  func.func private @func1(%arg0: index) -> memref<11x5xf32> {
    %cst_1 = arith.constant 0x4DAB5ADE : f32
    %cst_2 = arith.constant 1.840000e+04 : f16
    %false = arith.constant false
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index
    %c26 = arith.constant 26 : index
    %7 = tensor.empty(%c7, %c16) : tensor<?x?x28xi16>
    %124 = affine.for %arg1 = 0 to 92 iter_args(%arg2 = %7) -> (tensor<?x?x28xi16>) {
      %325 = tensor.empty(%c26, %c15) : tensor<?x?x28xi16>
      affine.yield %325 : tensor<?x?x28xi16>
    }
    %alloc_34 = memref.alloc() : memref<11x5xf32>
    return %alloc_34 : memref<11x5xf32>
  } 