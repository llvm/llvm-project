// -----// IR Dump After TosaToLinalg (tosa-to-linalg) //----- //
func.func @test_add_0d(%arg0: tensor<2x32xf32>, %arg1: tensor<2x32xf32>) -> (tensor<2x32xf32>, tensor<2x32xf32>) {
  %0 = tensor.empty() : tensor<2x32xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x32xf32>, tensor<2x32xf32>) outs(%0 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<2x32xf32>
  %2 = tensor.empty() : tensor<2x32xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x32xf32>, tensor<2x32xf32>) outs(%2 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<2x32xf32>
  return %1, %3 : tensor<2x32xf32>, tensor<2x32xf32>
}

// -----// IR Dump After EmptyTensorElimination (eliminate-empty-tensors) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_add_0d(%arg0: tensor<2x32xf32>, %arg1: tensor<2x32xf32>) -> (tensor<2x32xf32>, tensor<2x32xf32>) {
    %0 = tensor.empty() : tensor<2x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x32xf32>, tensor<2x32xf32>) outs(%0 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<2x32xf32>
    %2 = tensor.empty() : tensor<2x32xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x32xf32>, tensor<2x32xf32>) outs(%2 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<2x32xf32>
    return %1, %3 : tensor<2x32xf32>, tensor<2x32xf32>
  }
}


// -----// IR Dump After EmptyTensorToAllocTensor (empty-tensor-to-alloc-tensor) //----- //
func.func @test_add_0d(%arg0: tensor<2x32xf32>, %arg1: tensor<2x32xf32>) -> (tensor<2x32xf32>, tensor<2x32xf32>) {
  %0 = bufferization.alloc_tensor() : tensor<2x32xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x32xf32>, tensor<2x32xf32>) outs(%0 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<2x32xf32>
  %2 = bufferization.alloc_tensor() : tensor<2x32xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x32xf32>, tensor<2x32xf32>) outs(%2 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<2x32xf32>
  return %1, %3 : tensor<2x32xf32>, tensor<2x32xf32>
}

// -----// IR Dump After OneShotBufferize (one-shot-bufferize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_add_0d(%arg0: tensor<2x32xf32>, %arg1: tensor<2x32xf32>) -> (tensor<2x32xf32>, tensor<2x32xf32>) {
    %0 = bufferization.to_memref %arg1 : memref<2x32xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<2x32xf32, strided<[?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg1 : memref<2x32xf32, strided<[?, ?], offset: ?>>
    %3 = bufferization.to_memref %arg0 : memref<2x32xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %2 : memref<2x32xf32, strided<[?, ?], offset: ?>>, memref<2x32xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<2x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %6 = arith.addf %in, %in_1 : f32
      linalg.yield %6 : f32
    }
    %4 = bufferization.to_tensor %alloc : memref<2x32xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %0 : memref<2x32xf32, strided<[?, ?], offset: ?>>, memref<2x32xf32, strided<[?, ?], offset: ?>>) outs(%alloc_0 : memref<2x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %6 = arith.addf %in, %in_1 : f32
      linalg.yield %6 : f32
    }
    %5 = bufferization.to_tensor %alloc_0 : memref<2x32xf32>
    return %4, %5 : tensor<2x32xf32>, tensor<2x32xf32>
  }
}


// -----// IR Dump After FuncBufferize (func-bufferize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_add_0d(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>) -> (memref<2x32xf32>, memref<2x32xf32>) {
    %0 = bufferization.to_tensor %arg1 : memref<2x32xf32>
    %1 = bufferization.to_tensor %arg0 : memref<2x32xf32>
    %2 = bufferization.to_memref %0 : memref<2x32xf32, strided<[?, ?], offset: ?>>
    %3 = bufferization.to_memref %1 : memref<2x32xf32, strided<[?, ?], offset: ?>>
    %4 = bufferization.to_memref %0 : memref<2x32xf32, strided<[?, ?], offset: ?>>
    %5 = bufferization.to_memref %1 : memref<2x32xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %4 : memref<2x32xf32, strided<[?, ?], offset: ?>>, memref<2x32xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<2x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %10 = arith.addf %in, %in_1 : f32
      linalg.yield %10 : f32
    }
    %6 = bufferization.to_tensor %alloc : memref<2x32xf32>
    %7 = bufferization.to_memref %6 : memref<2x32xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %2 : memref<2x32xf32, strided<[?, ?], offset: ?>>, memref<2x32xf32, strided<[?, ?], offset: ?>>) outs(%alloc_0 : memref<2x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %10 = arith.addf %in, %in_1 : f32
      linalg.yield %10 : f32
    }
    %8 = bufferization.to_tensor %alloc_0 : memref<2x32xf32>
    %9 = bufferization.to_memref %8 : memref<2x32xf32>
    return %7, %9 : memref<2x32xf32>, memref<2x32xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_add_0d(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>) -> (memref<2x32xf32>, memref<2x32xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x32xf32>, memref<2x32xf32>) outs(%alloc : memref<2x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.addf %in, %in_1 : f32
      linalg.yield %0 : f32
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x32xf32>, memref<2x32xf32>) outs(%alloc_0 : memref<2x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.addf %in, %in_1 : f32
      linalg.yield %0 : f32
    }
    return %alloc, %alloc_0 : memref<2x32xf32>, memref<2x32xf32>
  }
}


// -----// IR Dump After ConvertLinalgToAffineLoopsPass (convert-linalg-to-affine-loops) //----- //
module {
  func.func @test_add_0d(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>) -> (memref<2x32xf32>, memref<2x32xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 32 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 32 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc_0[%arg2, %arg3] : memref<2x32xf32>
      }
    }
    return %alloc, %alloc_0 : memref<2x32xf32>, memref<2x32xf32>
  }
}


[FUSELOOPSINBLOCK LOG] DstLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
    %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
    %2 = arith.addf %0, %1 : f32
    affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
  }
}
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
    %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
    %2 = arith.addf %0, %1 : f32
    affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
  }
}
[CHECKFUSIBILITY LOG] Same Loop
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
    %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
    %2 = arith.addf %0, %1 : f32
    affine.store %2, %alloc_0[%arg2, %arg3] : memref<2x32xf32>
  }
}
[CHECKFUSIBILITY LOG] SUCCESS
[FUSELOOPSINBLOCK LOG] DSTLoop SRCLoop FUSABLE
[CHECKFUSIBILITY LOG] New FUSED DSTLoop
affine.for %arg2 = 0 to 2 {
  affine.for %arg3 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
    %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
    %2 = arith.addf %0, %1 : f32
    affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
  }
  affine.for %arg3 = 0 to 32 {
    %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
    %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
    %2 = arith.addf %0, %1 : f32
    affine.store %2, %alloc_0[%arg2, %arg3] : memref<2x32xf32>
  }
}
[CHECKFUSIBILITY LOG] Step is not same
[FUSELOOPSINBLOCK LOG] DstLoop -> 
affine.for %arg3 = 0 to 32 {
  %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
  %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
  %2 = arith.addf %0, %1 : f32
  affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
}
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg3 = 0 to 32 {
  %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
  %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
  %2 = arith.addf %0, %1 : f32
  affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
}
[CHECKFUSIBILITY LOG] Same Loop
[FUSELOOPSINBLOCK LOG] SrcLoop -> 
affine.for %arg3 = 0 to 32 {
  %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
  %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
  %2 = arith.addf %0, %1 : f32
  affine.store %2, %alloc_0[%arg2, %arg3] : memref<2x32xf32>
}
[CHECKFUSIBILITY LOG] SUCCESS
[FUSELOOPSINBLOCK LOG] DSTLoop SRCLoop FUSABLE
[CHECKFUSIBILITY LOG] New FUSED DSTLoop
affine.for %arg3 = 0 to 32 {
  %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
  %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
  %2 = arith.addf %0, %1 : f32
  affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
  %3 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
  %4 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
  %5 = arith.addf %3, %4 : f32
  affine.store %5, %alloc_0[%arg2, %arg3] : memref<2x32xf32>
}
[FUSELOOPSINBLOCK LOG] DstLoop -> 
"affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
^bb0(%arg0: index):
  %0 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
  %1 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
  %2 = "arith.addf"(<<NULL VALUE>>, <<NULL VALUE>>) <{fastmath = #arith.fastmath<none>}> : (<<NULL TYPE>>, <<NULL TYPE>>) -> f32
  "affine.store"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> ()
  "affine.yield"() : () -> ()
}) : () -> ()
[FUSELOOPSINBLOCK LOG] 2 - DstLoop refernce dropped
[CHECKFUSIBILITY LOG] Step is not same
[FUSELOOPSINBLOCK LOG] DstLoop -> 
"affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (2)>}> ({
^bb0(%arg0: index):
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
  ^bb0(%arg1: index):
    %0 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
    %1 = "affine.load"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> f32
    %2 = "arith.addf"(<<NULL VALUE>>, <<NULL VALUE>>) <{fastmath = #arith.fastmath<none>}> : (<<NULL TYPE>>, <<NULL TYPE>>) -> f32
    "affine.store"(<<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>, <<NULL VALUE>>) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (<<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>, <<NULL TYPE>>) -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "affine.yield"() : () -> ()
}) : () -> ()
[FUSELOOPSINBLOCK LOG] 2 - DstLoop refernce dropped
[CHECKFUSIBILITY LOG] Step is not same
// -----// IR Dump After TosaAffineFusion (tosa-affine-fusion) //----- //
module {
  func.func @test_add_0d(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>) -> (memref<2x32xf32>, memref<2x32xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 32 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
        %3 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
        %4 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
        %5 = arith.addf %3, %4 : f32
        affine.store %5, %alloc_0[%arg2, %arg3] : memref<2x32xf32>
      }
    }
    return %alloc, %alloc_0 : memref<2x32xf32>, memref<2x32xf32>
  }
}


module {
  func.func @test_add_0d(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>) -> (memref<2x32xf32>, memref<2x32xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 32 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc[%arg2, %arg3] : memref<2x32xf32>
        %3 = affine.load %arg0[%arg2, %arg3] : memref<2x32xf32>
        %4 = affine.load %arg1[%arg2, %arg3] : memref<2x32xf32>
        %5 = arith.addf %3, %4 : f32
        affine.store %5, %alloc_0[%arg2, %arg3] : memref<2x32xf32>
      }
    }
    return %alloc, %alloc_0 : memref<2x32xf32>, memref<2x32xf32>
  }
}

