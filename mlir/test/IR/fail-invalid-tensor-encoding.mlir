// RUN: not mlir-opt %s

// Check that mlir-opt fails parsing when fed a tensor with an invalid encoding

!type = tensor<256x32xf16, #blocked>