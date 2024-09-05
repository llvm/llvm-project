// -----// IR Dump After TosaToLinalg (tosa-to-linalg) //----- //
func.func @test_add_0d(%arg0: tensor<2x4x32xf32>, %arg1: tensor<2x4x32xf32>) -> (tensor<1x4x32xf32>, tensor<2x1x32xf32>) {
  %0 = tensor.empty() : tensor<4x32xf32>
  %cst = arith.constant -3.40282347E+38 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<2x4x32xf32>) outs(%1 : tensor<4x32xf32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %4 = arith.maximumf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded = tensor.expand_shape %reduced [[0, 1], [2]] output_shape [1, 4, 32] : tensor<4x32xf32> into tensor<1x4x32xf32>
  %2 = tensor.empty() : tensor<2x32xf32>
  %cst_0 = arith.constant 3.40282347E+38 : f32
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %reduced_1 = linalg.reduce ins(%arg1 : tensor<2x4x32xf32>) outs(%3 : tensor<2x32xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %4 = arith.minimumf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded_2 = tensor.expand_shape %reduced_1 [[0], [1, 2]] output_shape [2, 1, 32] : tensor<2x32xf32> into tensor<2x1x32xf32>
  return %expanded, %expanded_2 : tensor<1x4x32xf32>, tensor<2x1x32xf32>
}

// -----// IR Dump After EmptyTensorElimination (eliminate-empty-tensors) //----- //
module {
  func.func @test_add_0d(%arg0: tensor<2x4x32xf32>, %arg1: tensor<2x4x32xf32>) -> (tensor<1x4x32xf32>, tensor<2x1x32xf32>) {
    %0 = tensor.empty() : tensor<4x32xf32>
    %cst = arith.constant -3.40282347E+38 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<2x4x32xf32>) outs(%1 : tensor<4x32xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %4 = arith.maximumf %in, %init : f32
        linalg.yield %4 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1], [2]] output_shape [1, 4, 32] : tensor<4x32xf32> into tensor<1x4x32xf32>
    %2 = tensor.empty() : tensor<2x32xf32>
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<2x32xf32>) -> tensor<2x32xf32>
    %reduced_1 = linalg.reduce ins(%arg1 : tensor<2x4x32xf32>) outs(%3 : tensor<2x32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %4 = arith.minimumf %in, %init : f32
        linalg.yield %4 : f32
      }
    %expanded_2 = tensor.expand_shape %reduced_1 [[0], [1, 2]] output_shape [2, 1, 32] : tensor<2x32xf32> into tensor<2x1x32xf32>
    return %expanded, %expanded_2 : tensor<1x4x32xf32>, tensor<2x1x32xf32>
  }
}


// -----// IR Dump After EmptyTensorToAllocTensor (empty-tensor-to-alloc-tensor) //----- //
func.func @test_add_0d(%arg0: tensor<2x4x32xf32>, %arg1: tensor<2x4x32xf32>) -> (tensor<1x4x32xf32>, tensor<2x1x32xf32>) {
  %cst = arith.constant 3.40282347E+38 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %0 = bufferization.alloc_tensor() : tensor<4x32xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<2x4x32xf32>) outs(%1 : tensor<4x32xf32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %4 = arith.maximumf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded = tensor.expand_shape %reduced [[0, 1], [2]] output_shape [1, 4, 32] : tensor<4x32xf32> into tensor<1x4x32xf32>
  %2 = bufferization.alloc_tensor() : tensor<2x32xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %reduced_1 = linalg.reduce ins(%arg1 : tensor<2x4x32xf32>) outs(%3 : tensor<2x32xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %4 = arith.minimumf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded_2 = tensor.expand_shape %reduced_1 [[0], [1, 2]] output_shape [2, 1, 32] : tensor<2x32xf32> into tensor<2x1x32xf32>
  return %expanded, %expanded_2 : tensor<1x4x32xf32>, tensor<2x1x32xf32>
}

// -----// IR Dump After OneShotBufferize (one-shot-bufferize) //----- //
module {
  func.func @test_add_0d(%arg0: tensor<2x4x32xf32>, %arg1: tensor<2x4x32xf32>) -> (tensor<1x4x32xf32>, tensor<2x1x32xf32>) {
    %0 = bufferization.to_memref %arg1 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>
    %cst = arith.constant 3.40282347E+38 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<4x32xf32>)
    linalg.reduce ins(%1 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<4x32xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %4 = arith.maximumf %in, %init : f32
        linalg.yield %4 : f32
      }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4, 32] : memref<4x32xf32> into memref<1x4x32xf32>
    %2 = bufferization.to_tensor %expand_shape : memref<1x4x32xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<2x32xf32>)
    linalg.reduce ins(%0 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc_1 : memref<2x32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %4 = arith.minimumf %in, %init : f32
        linalg.yield %4 : f32
      }
    %expand_shape_2 = memref.expand_shape %alloc_1 [[0], [1, 2]] output_shape [2, 1, 32] : memref<2x32xf32> into memref<2x1x32xf32>
    %3 = bufferization.to_tensor %expand_shape_2 : memref<2x1x32xf32>
    return %2, %3 : tensor<1x4x32xf32>, tensor<2x1x32xf32>
  }
}


// -----// IR Dump After FuncBufferize (func-bufferize) //----- //
module {
  func.func @test_add_0d(%arg0: memref<2x4x32xf32>, %arg1: memref<2x4x32xf32>) -> (memref<1x4x32xf32>, memref<2x1x32xf32>) {
    %0 = bufferization.to_tensor %arg1 : memref<2x4x32xf32>
    %1 = bufferization.to_tensor %arg0 : memref<2x4x32xf32>
    %2 = bufferization.to_memref %0 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>
    %3 = bufferization.to_memref %1 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>
    %cst = arith.constant 3.40282347E+38 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<4x32xf32>)
    linalg.reduce ins(%3 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<4x32xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %8 = arith.maximumf %in, %init : f32
        linalg.yield %8 : f32
      }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4, 32] : memref<4x32xf32> into memref<1x4x32xf32>
    %4 = bufferization.to_tensor %expand_shape : memref<1x4x32xf32>
    %5 = bufferization.to_memref %4 : memref<1x4x32xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<2x32xf32>)
    linalg.reduce ins(%2 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc_1 : memref<2x32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %8 = arith.minimumf %in, %init : f32
        linalg.yield %8 : f32
      }
    %expand_shape_2 = memref.expand_shape %alloc_1 [[0], [1, 2]] output_shape [2, 1, 32] : memref<2x32xf32> into memref<2x1x32xf32>
    %6 = bufferization.to_tensor %expand_shape_2 : memref<2x1x32xf32>
    %7 = bufferization.to_memref %6 : memref<2x1x32xf32>
    return %5, %7 : memref<1x4x32xf32>, memref<2x1x32xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module {
  func.func @test_add_0d(%arg0: memref<2x4x32xf32>, %arg1: memref<2x4x32xf32>) -> (memref<1x4x32xf32>, memref<2x1x32xf32>) {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %cast = memref.cast %arg1 : memref<2x4x32xf32> to memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>
    %cast_1 = memref.cast %arg0 : memref<2x4x32xf32> to memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<4x32xf32>)
    linalg.reduce ins(%cast_1 : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<4x32xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %0 = arith.maximumf %in, %init : f32
        linalg.yield %0 : f32
      }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4, 32] : memref<4x32xf32> into memref<1x4x32xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_2 : memref<2x32xf32>)
    linalg.reduce ins(%cast : memref<2x4x32xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc_2 : memref<2x32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %0 = arith.minimumf %in, %init : f32
        linalg.yield %0 : f32
      }
    %expand_shape_3 = memref.expand_shape %alloc_2 [[0], [1, 2]] output_shape [2, 1, 32] : memref<2x32xf32> into memref<2x1x32xf32>
    return %expand_shape, %expand_shape_3 : memref<1x4x32xf32>, memref<2x1x32xf32>
  }
}


// -----// IR Dump After ConvertLinalgToAffineLoopsPass (convert-linalg-to-affine-loops) //----- //
module {
  func.func @test_add_0d(%arg0: memref<2x4x32xf32>, %arg1: memref<2x4x32xf32>) -> (memref<1x4x32xf32>, memref<2x1x32xf32>) {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32>
    affine.for %arg2 = 0 to 4 {
      affine.for %arg3 = 0 to 32 {
        affine.store %cst, %alloc[%arg2, %arg3] : memref<4x32xf32>
      }
    }
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 32 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
          %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
          %2 = arith.maximumf %0, %1 : f32
          affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
        }
      }
    }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4, 32] : memref<4x32xf32> into memref<1x4x32xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 32 {
        affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
      }
    }
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 32 {
          %0 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
          %1 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
          %2 = arith.minimumf %0, %1 : f32
          affine.store %2, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
        }
      }
    }
    %expand_shape_2 = memref.expand_shape %alloc_1 [[0], [1, 2]] output_shape [2, 1, 32] : memref<2x32xf32> into memref<2x1x32xf32>
    return %expand_shape, %expand_shape_2 : memref<1x4x32xf32>, memref<2x1x32xf32>
  }
}


[FUSELOOPSINBLOCK LOG] DstLoop -> 
affine.for %arg2 = 0 to 4 {
  affine.for %arg3 = 0 to 32 {
    affine.store %cst, %alloc[%arg2, %arg3] : memref<4x32xf32>
  }
}
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 4 {
  affine.for %arg3 = 0 to 32 {
    affine.store %cst, %alloc[%arg2, %arg3] : memref<4x32xf32>
  }
}
[CHECKFUSIBILITY LOG] Same Loop
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
      %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
      %2 = arith.maximumf %0, %1 : f32
      affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
    }
  }
}
[CHECKFUSIBILITY LOG] Upper Bound is not same
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 32 {
    affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
  }
}
[CHECKFUSIBILITY LOG] Upper Bound is not same
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
      %1 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
      %2 = arith.minimumf %0, %1 : f32
      affine.store %2, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
    }
  }
}
[CHECKFUSIBILITY LOG] Upper Bound is not same
[FUSELOOPSINBLOCK LOG] DstLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
      %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
      %2 = arith.maximumf %0, %1 : f32
      affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
    }
  }
}
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 4 {
  affine.for %arg3 = 0 to 32 {
    affine.store %cst, %alloc[%arg2, %arg3] : memref<4x32xf32>
  }
}
[CHECKFUSIBILITY LOG] Upper Bound is not same
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
      %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
      %2 = arith.maximumf %0, %1 : f32
      affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
    }
  }
}
[CHECKFUSIBILITY LOG] Same Loop
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 32 {
    affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
  }
}
[CHECKFUSIBILITY LOG] SUCCESS
[FUSELOOPSINBLOCK LOG] DSTLoop SRCLoop FUSABLE
[CHECKFUSIBILITY LOG] New FUSED DSTLoop
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
      %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
      %2 = arith.maximumf %0, %1 : f32
      affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
    }
  }
  affine.for %arg3 = 0 to 32 {
    affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
  }
}
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
      %1 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
      %2 = arith.minimumf %0, %1 : f32
      affine.store %2, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
    }
  }
}
[CHECKFUSIBILITY LOG] SUCCESS
[FUSELOOPSINBLOCK LOG] DSTLoop SRCLoop FUSABLE
[CHECKFUSIBILITY LOG] New FUSED DSTLoop
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
      %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
      %2 = arith.maximumf %0, %1 : f32
      affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
    }
  }
  affine.for %arg3 = 0 to 32 {
    affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
  }
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
      %1 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
      %2 = arith.minimumf %0, %1 : f32
      affine.store %2, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
    }
  }
}
[CHECKFUSIBILITY LOG] Step is not same
[FUSELOOPSINBLOCK LOG] DstLoop -> 
affine.for %arg3 = 0 to 4 {
  affine.for %arg4 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
    %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
    %2 = arith.maximumf %0, %1 : f32
    affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
  }
}
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg3 = 0 to 4 {
  affine.for %arg4 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
    %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
    %2 = arith.maximumf %0, %1 : f32
    affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
  }
}
[CHECKFUSIBILITY LOG] Same Loop
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg3 = 0 to 32 {
  affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
}
[CHECKFUSIBILITY LOG] Upper Bound is not same
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg3 = 0 to 4 {
  affine.for %arg4 = 0 to 32 {
    %0 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
    %1 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
    %2 = arith.minimumf %0, %1 : f32
    affine.store %2, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
  }
}
[CHECKFUSIBILITY LOG] SUCCESS
[FUSELOOPSINBLOCK LOG] DSTLoop SRCLoop FUSABLE
[CHECKFUSIBILITY LOG] New FUSED DSTLoop
affine.for %arg3 = 0 to 4 {
  affine.for %arg4 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
    %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
    %2 = arith.maximumf %0, %1 : f32
    affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
  }
  affine.for %arg4 = 0 to 32 {
    %0 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
    %1 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
    %2 = arith.minimumf %0, %1 : f32
    affine.store %2, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
  }
}
[CHECKFUSIBILITY LOG] Step is not same
[FUSELOOPSINBLOCK LOG] DstLoop -> 
affine.for %arg4 = 0 to 32 {
  %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
  %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
  %2 = arith.maximumf %0, %1 : f32
  affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
}
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg4 = 0 to 32 {
  %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
  %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
  %2 = arith.maximumf %0, %1 : f32
  affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
}
[CHECKFUSIBILITY LOG] Same Loop
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg4 = 0 to 32 {
  %0 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
  %1 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
  %2 = arith.minimumf %0, %1 : f32
  affine.store %2, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
}
[CHECKFUSIBILITY LOG] SUCCESS
[FUSELOOPSINBLOCK LOG] DSTLoop SRCLoop FUSABLE
[CHECKFUSIBILITY LOG] New FUSED DSTLoop
affine.for %arg4 = 0 to 32 {
  %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
  %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
  %2 = arith.maximumf %0, %1 : f32
  affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
  %3 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
  %4 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
  %5 = arith.minimumf %3, %4 : f32
  affine.store %5, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
}
[FUSELOOPSINBLOCK LOG] DstLoop -> 
"affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
^bb0(%arg0: index):
  %0 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
  %1 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
  %2 = "arith.minimumf"(<<NULL VALUE>>, <<NULL VALUE>>) <{fastmath = #arith.fastmath<none>}> : (<<NULL TYPE>>, <<NULL TYPE>>) -> f32
  "affine.store"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> ()
  "affine.yield"() : () -> ()
}) : () -> ()
[FUSELOOPSINBLOCK LOG] 2 - DstLoop refernce dropped
[CHECKFUSIBILITY LOG] Step is not same
[FUSELOOPSINBLOCK LOG] DstLoop -> 
affine.for %arg3 = 0 to 32 {
  affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
}
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg3 = 0 to 4 {
  affine.for %arg4 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
    %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
    %2 = arith.maximumf %0, %1 : f32
    affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
    %3 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
    %4 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
    %5 = arith.minimumf %3, %4 : f32
    affine.store %5, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
  }
}
[CHECKFUSIBILITY LOG] Upper Bound is not same
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg3 = 0 to 32 {
  affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
}
[CHECKFUSIBILITY LOG] Same Loop
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
"affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
^bb0(%arg0: index):
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
  ^bb0(%arg1: index):
    %0 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
    %1 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
    %2 = "arith.minimumf"(<<NULL VALUE>>, <<NULL VALUE>>) <{fastmath = #arith.fastmath<none>}> : (<<NULL TYPE>>, <<NULL TYPE>>) -> f32
    "affine.store"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "affine.yield"() : () -> ()
}) : () -> ()
[FUSELOOPSINBLOCK LOG] 2 - SrcLoop refernce dropped
[FUSELOOPSINBLOCK LOG] DstLoop -> 
"affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
^bb0(%arg0: index):
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
  ^bb0(%arg1: index):
    %0 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
    %1 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
    %2 = "arith.minimumf"(<<NULL VALUE>>, <<NULL VALUE>>) <{fastmath = #arith.fastmath<none>}> : (<<NULL TYPE>>, <<NULL TYPE>>) -> f32
    "affine.store"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "affine.yield"() : () -> ()
}) : () -> ()
[FUSELOOPSINBLOCK LOG] 2 - DstLoop refernce dropped
[CHECKFUSIBILITY LOG] Step is not same
[FUSELOOPSINBLOCK LOG] DstLoop -> 
"affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (2)>}> ({
^bb0(%arg0: index):
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
  ^bb0(%arg1: index):
    "affine.store"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "affine.yield"() : () -> ()
}) : () -> ()
[FUSELOOPSINBLOCK LOG] 2 - DstLoop refernce dropped
[FUSELOOPSINBLOCK LOG] DstLoop -> 
"affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (2)>}> ({
^bb0(%arg0: index):
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
  ^bb0(%arg1: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
    ^bb0(%arg2: index):
      %0 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
      %1 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
      %2 = "arith.minimumf"(<<NULL VALUE>>, <<NULL VALUE>>) <{fastmath = #arith.fastmath<none>}> : (<<NULL TYPE>>, <<NULL TYPE>>) -> f32
      "affine.store"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "affine.yield"() : () -> ()
}) : () -> ()
[FUSELOOPSINBLOCK LOG] 2 - DstLoop refernce dropped
[CHECKFUSIBILITY LOG] Step is not same
// -----// IR Dump After TosaAffineFusion (tosa-affine-fusion) //----- //
module {
  func.func @test_add_0d(%arg0: memref<2x4x32xf32>, %arg1: memref<2x4x32xf32>) -> (memref<1x4x32xf32>, memref<2x1x32xf32>) {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32>
    affine.for %arg2 = 0 to 4 {
      affine.for %arg3 = 0 to 32 {
        affine.store %cst, %alloc[%arg2, %arg3] : memref<4x32xf32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 32 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
          %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
          %2 = arith.maximumf %0, %1 : f32
          affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
          %3 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
          %4 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
          %5 = arith.minimumf %3, %4 : f32
          affine.store %5, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
        }
      }
      affine.for %arg3 = 0 to 32 {
        affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
      }
    }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4, 32] : memref<4x32xf32> into memref<1x4x32xf32>
    %expand_shape_2 = memref.expand_shape %alloc_1 [[0], [1, 2]] output_shape [2, 1, 32] : memref<2x32xf32> into memref<2x1x32xf32>
    return %expand_shape, %expand_shape_2 : memref<1x4x32xf32>, memref<2x1x32xf32>
  }
}


module {
  func.func @test_add_0d(%arg0: memref<2x4x32xf32>, %arg1: memref<2x4x32xf32>) -> (memref<1x4x32xf32>, memref<2x1x32xf32>) {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32>
    affine.for %arg2 = 0 to 4 {
      affine.for %arg3 = 0 to 32 {
        affine.store %cst, %alloc[%arg2, %arg3] : memref<4x32xf32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 32 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
          %1 = affine.load %alloc[%arg3, %arg4] : memref<4x32xf32>
          %2 = arith.maximumf %0, %1 : f32
          affine.store %2, %alloc[%arg3, %arg4] : memref<4x32xf32>
          %3 = affine.load %arg1[%arg2, %arg3, %arg4] : memref<2x4x32xf32>
          %4 = affine.load %alloc_1[%arg2, %arg4] : memref<2x32xf32>
          %5 = arith.minimumf %3, %4 : f32
          affine.store %5, %alloc_1[%arg2, %arg4] : memref<2x32xf32>
        }
      }
      affine.for %arg3 = 0 to 32 {
        affine.store %cst_0, %alloc_1[%arg2, %arg3] : memref<2x32xf32>
      }
    }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4, 32] : memref<4x32xf32> into memref<1x4x32xf32>
    %expand_shape_2 = memref.expand_shape %alloc_1 [[0], [1, 2]] output_shape [2, 1, 32] : memref<2x32xf32> into memref<2x1x32xf32>
    return %expand_shape, %expand_shape_2 : memref<1x4x32xf32>, memref<2x1x32xf32>
  }
}

