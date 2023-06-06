# LSTM-Cell-Ruby
This repository is an LSTM cell (single cell) simulated in RUBY.
It is a building block of a complete LSTM to be developed further.

## LSTM Architecture
The following architecure is implemented in Ruby.
![LSTM diagram](https://github.com/ryan-n-may/LSTM-Cpp/blob/main/LSTM_draw.drawio.png)

## Forward Propagation
$$Z_g = W_{xg} x_t + W_{hg} h_{t-1} + b_g $$

$$g_t = tanh(Z_g) $$

$$Z_i = W_{xi} x_t + W_{hi} h_{t-1} + b_i $$

$$i_t = sigmoid(Z_i) $$

$$i_{gate} = g_t i_t $$

$$Z_f = W_{xf} x_t + W_{hf} h_{t-1} + b_g $$ 

$$f_t = sigmoid(Z_f) $$

$$f_{gate} = f_t $$

$$Z_o = W_{xo} x_t + W_{ho} h_{t-1} + b_g $$

$$o_t = sigmoid(Z_o) $$

$$o_{gate} = o_t $$

$$c_t = (c_{t-1}f_{gate}) + i_{gate} $$

$$h_t = o_{gate} tanh(c_t) $$

```ruby
def forwardPropagation()
  # 'g' operations
	@Zg = multiplyWithWeights(@Xtm1, @Wxg) + multiplyWithWeights(@Htm1, @Whg) + @Bg.transpose()
	@Gt = tanhVector(@Zg)
	# 'i' operations
	@Zi = multiplyWithWeights(@Xtm1, @Wxi) + multiplyWithWeights(@Htm1, @Whi) + @Bi.transpose()
	@It = sigmoidVector(@Zi)
	@Igate = @Gt *~ @It
	# 'f' operations
	@Zf = multiplyWithWeights(@Xtm1, @Wxf) + multiplyWithWeights(@Htm1, @Whf) + @Bf.transpose()
	@Ft = sigmoidVector(@Zf)
	@Fgate = @Ft
	# 'o' operations
	@Zo = multiplyWithWeights(@Xtm1, @Wxo) + multiplyWithWeights(@Htm1, @Who) + @Bg.transpose()
	@Ot = sigmoidVector(@Zo)
	@Ogate = @Ot
	# cell state operations
	@Ct = (@Ctm1 *~ @Fgate) + @Igate
	@Ht = @Ogate *~ tanhVector(@Ct)
end
```

## Backward propagation
```ruby
def backwardPropagation()
  # Calculating the gradients with respect to weights and output error
	# (see github for explanation in C++)
	# https://github.com/ryan-n-may/LSTM-Cpp/tree/main
	dE = @Yt - @Ht
	# Gradient with respect to gates and states
	dE_dot   = dE *~ tanhVector(@Ct)
	dE_dct   = dE *~ @Ot *~ invTanhVector(@Ct)
	dE_dit   = dE_dct *~ @It
	dE_dft   = dE_dct *~ @Ctm1
	dE_dctm1 = dE_dct *~ @Ft
	# Gradient with respect to output weights
	dE_dbo = dE *~ tanhVector(@Ct) *~ sigmoidVector(@Zo) *~ invSigmoidVector(@Zo)
	dE_dWxo = dE_dbo *~ @Xtm1 
	dE_dWho = dE_dbo *~ @Htm1
	# Gradient with respect to forget weights
	dE_dbf  = dE *~ @Ot *~ invTanhVector(@Ct) *~ @Ctm1 *~ sigmoidVector(@Zf) *~ invSigmoidVector(@Zf)
	dE_dWxf = dE_dbf *~ @Xtm1
	dE_dWhf = dE_dbf *~ @Htm1
	# Gradient with respect to input weights
	dE_dbi  = dE *~ @Ot *~ invTanhVector(@Ct) *~ @Gt *~ sigmoidVector(@Zi) *~ invSigmoidVector(@Zi)
	dE_dWxi = dE_dbi *~ @Xtm1
	dE_dWhi = dE_dbi *~ @Htm1
	# Gradient with respect to cell states @Gt
	dE_dbg  = dE *~ @Ot *~ @Ot *~ invTanhVector(@Ct) *~ @It *~ invTanhVector(@Zg) 
	dE_dWxg = dE_dbg *~ @Xtm1
	dE_dWhg = dE_dbg *~ @Htm1
	# Now we update the weights using the calculated gradients.
	# Modifying output weights
	@Wxo = updateWeights(@Wxo, dE_dWxo, @@Alpha)
	@Who = updateWeights(@Who, dE_dWho, @@Alpha)
	# Modifying forget weights	
	@Wxf = updateWeights(@Wxf, dE_dWxf, @@Alpha)
	@Whf = updateWeights(@Whf, dE_dWhf, @@Alpha)
	# Modifying input weights
	@Wxi = updateWeights(@Wxi, dE_dWxi, @@Alpha)
	@Whi = updateWeights(@Whi, dE_dWhi, @@Alpha)
	# Modifying g weights
	@Wxg = updateWeights(@Wxg, dE_dWxg, @@Alpha)
	@Whg = updateWeights(@Whg, dE_dWhg, @@Alpha)
	# Modifying bias vectors
	@Bf = @Bf + dE_dbf.transpose() # UNSURE IF THESE SHOULD ADD OR MINUS
	@Bi = @Bi + dE_dbi.transpose()
	@Bg = @Bg + dE_dbg.transpose()
end
```
## Weight modification
The figure here shows the weight matrices before traning (left) and after training (right)

![output](https://github.com/ryan-n-may/LSTM-Cell-Ruby/blob/main/output.png)

