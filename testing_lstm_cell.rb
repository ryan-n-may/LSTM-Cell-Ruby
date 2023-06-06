#!/usr/bin/env ruby
require_relative 'lstm_cell'
puts "Testing LSTM single cell"
# Initial cell states and cell target
ctm1_ = Matrix[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
xtm1_ = Matrix[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
yt_   = Matrix[[0.1,0.2,0.3,0.4,0.2,1.0,0.1,0.0,0.1,0.4,0.3,0.5,0.9,0.2,0.3,0.4]]
htm1_ = Matrix[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
# Initialising the cell
object = LSTM_CELL. new
object.init(16, 1.0)
object.plotWeights()
# Testing cell training
object.setState(ctm1_, htm1_, xtm1_, yt_)
object.forwardPropagation()
object.viewState()
for index in 0...550
	object.forwardPropagation()
	object.backwardPropagation()
	object.setState(object.getCt(), object.getHt(), xtm1_, yt_)
end
object.viewState()
object.plotWeights()
puts "Complete"