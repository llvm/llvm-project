// RUN: mlir-opt -normalize-memrefs -allow-unregistered-dialect %s | FileCheck %s

// This file tests whether the memref type having non-trivial map layouts
// are normalized to trivial (identity) layouts.

// CHECK-DAG: #[[$REDUCE_MAP1:.*]] = affine_map<(d0, d1) -> ((d0 mod 2) * 2 + d1 mod 2 + (d0 floordiv 2) * 4 + (d1 floordiv 2) * 8)>
// CHECK-DAG: #[[$REDUCE_MAP2:.*]] = affine_map<(d0, d1) -> (d0 mod 2 + (d1 mod 2) * 2 + (d0 floordiv 2) * 8 + (d1 floordiv 2) * 4)>
// CHECK-DAG: #[[$REDUCE_MAP3:.*]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>

// CHECK-LABEL: func @permute()
func.func @permute() {
  %A = memref.alloc() : memref<64x256xf32, affine_map<(d0, d1) -> (d1, d0)>>
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 256 {
      %1 = affine.load %A[%i, %j] : memref<64x256xf32, affine_map<(d0, d1) -> (d1, d0)>>
      "prevent.dce"(%1) : (f32) -> ()
    }
  }
  memref.dealloc %A : memref<64x256xf32, affine_map<(d0, d1) -> (d1, d0)>>
  return
}
// The old memref alloc should disappear.
// CHECK-NOT:  memref<64x256xf32>
// CHECK:      [[MEM:%[0-9a-zA-Z_]+]] = memref.alloc() : memref<256x64xf32>
// CHECK-NEXT: affine.for %[[I:arg[0-9a-zA-Z_]+]] = 0 to 64 {
// CHECK-NEXT:   affine.for %[[J:arg[0-9a-zA-Z_]+]] = 0 to 256 {
// CHECK-NEXT:     affine.load [[MEM]][%[[J]], %[[I]]] : memref<256x64xf32>
// CHECK-NEXT:     "prevent.dce"
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: memref.dealloc [[MEM]]
// CHECK-NEXT: return

// CHECK-LABEL: func @shift
func.func @shift(%idx : index) {
  // CHECK-NEXT: memref.alloc() : memref<65xf32>
  %A = memref.alloc() : memref<64xf32, affine_map<(d0) -> (d0 + 1)>>
  // CHECK-NEXT: affine.load %{{.*}}[symbol(%arg0) + 1] : memref<65xf32>
  affine.load %A[%idx] : memref<64xf32, affine_map<(d0) -> (d0 + 1)>>
  affine.for %i = 0 to 64 {
    %1 = affine.load %A[%i] : memref<64xf32, affine_map<(d0) -> (d0 + 1)>>
    "prevent.dce"(%1) : (f32) -> ()
    // CHECK: %{{.*}} = affine.load %{{.*}}[%arg{{.*}} + 1] : memref<65xf32>
  }
  return
}

// CHECK-LABEL: func @high_dim_permute()
func.func @high_dim_permute() {
  // CHECK-NOT: memref<64x128x256xf32,
  %A = memref.alloc() : memref<64x128x256xf32, affine_map<(d0, d1, d2) -> (d2, d0, d1)>>
  // CHECK: %[[I:arg[0-9a-zA-Z_]+]]
  affine.for %i = 0 to 64 {
    // CHECK: %[[J:arg[0-9a-zA-Z_]+]]
    affine.for %j = 0 to 128 {
      // CHECK: %[[K:arg[0-9a-zA-Z_]+]]
      affine.for %k = 0 to 256 {
        %1 = affine.load %A[%i, %j, %k] : memref<64x128x256xf32, affine_map<(d0, d1, d2) -> (d2, d0, d1)>>
        // CHECK: %{{.*}} = affine.load %{{.*}}[%[[K]], %[[I]], %[[J]]] : memref<256x64x128xf32>
        "prevent.dce"(%1) : (f32) -> ()
      }
    }
  }
  return
}

// CHECK-LABEL: func @invalid_map
func.func @invalid_map() {
  %A = memref.alloc() : memref<64x128xf32, affine_map<(d0, d1) -> (d0, -d1 - 10)>>
  // CHECK: %{{.*}} = memref.alloc() : memref<64x128xf32,
  return
}

// A tiled layout.
// CHECK-LABEL: func @data_tiling
func.func @data_tiling(%idx : index) {
  // CHECK: memref.alloc() : memref<8x32x8x16xf32>
  %A = memref.alloc() : memref<64x512xf32, affine_map<(d0, d1) -> (d0 floordiv 8, d1 floordiv 16, d0 mod 8, d1 mod 16)>>
  // CHECK: affine.load %{{.*}}[symbol(%arg0) floordiv 8, symbol(%arg0) floordiv 16, symbol(%arg0) mod 8, symbol(%arg0) mod 16]
  %1 = affine.load %A[%idx, %idx] : memref<64x512xf32, affine_map<(d0, d1) -> (d0 floordiv 8, d1 floordiv 16, d0 mod 8, d1 mod 16)>>
  "prevent.dce"(%1) : (f32) -> ()
  return
}

// Strides 2 and 4 along respective dimensions.
// CHECK-LABEL: func @strided
func.func @strided() {
  %A = memref.alloc() : memref<64x128xf32, affine_map<(d0, d1) -> (2*d0, 4*d1)>>
  // CHECK: affine.for %[[IV0:.*]] =
  affine.for %i = 0 to 64 {
    // CHECK: affine.for %[[IV1:.*]] =
    affine.for %j = 0 to 128 {
      // CHECK: affine.load %{{.*}}[%[[IV0]] * 2, %[[IV1]] * 4] : memref<127x509xf32>
      %1 = affine.load %A[%i, %j] : memref<64x128xf32, affine_map<(d0, d1) -> (2*d0, 4*d1)>>
      "prevent.dce"(%1) : (f32) -> ()
    }
  }
  return
}

// Strided, but the strides are in the linearized space.
// CHECK-LABEL: func @strided_cumulative
func.func @strided_cumulative() {
  %A = memref.alloc() : memref<2x5xf32, affine_map<(d0, d1) -> (3*d0 + 17*d1)>>
  // CHECK: affine.for %[[IV0:.*]] =
  affine.for %i = 0 to 2 {
    // CHECK: affine.for %[[IV1:.*]] =
    affine.for %j = 0 to 5 {
      // CHECK: affine.load %{{.*}}[%[[IV0]] * 3 + %[[IV1]] * 17] : memref<72xf32>
      %1 = affine.load %A[%i, %j]  : memref<2x5xf32, affine_map<(d0, d1) -> (3*d0 + 17*d1)>>
      "prevent.dce"(%1) : (f32) -> ()
    }
  }
  return
}

// Symbolic operand for alloc, although unused. Tests replaceAllMemRefUsesWith
// when the index remap has symbols.
// CHECK-LABEL: func @symbolic_operands
func.func @symbolic_operands(%s : index) {
  // CHECK: memref.alloc() : memref<100xf32>
  %A = memref.alloc()[%s] : memref<10x10xf32, affine_map<(d0,d1)[s0] -> (10*d0 + d1)>>
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {
      // CHECK: affine.load %{{.*}}[%{{.*}} * 10 + %{{.*}}] : memref<100xf32>
      %1 = affine.load %A[%i, %j] : memref<10x10xf32, affine_map<(d0,d1)[s0] -> (10*d0 + d1)>>
      "prevent.dce"(%1) : (f32) -> ()
    }
  }
  return
}

// Semi-affine maps, normalization not implemented yet.
// CHECK-LABEL: func @semi_affine_layout_map
func.func @semi_affine_layout_map(%s0: index, %s1: index) {
  %A = memref.alloc()[%s0, %s1] : memref<256x1024xf32, affine_map<(d0, d1)[s0, s1] -> (d0*s0 + d1*s1)>>
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 1024 {
      // CHECK: memref<256x1024xf32, #map{{[0-9a-zA-Z_]+}}>
      affine.load %A[%i, %j] : memref<256x1024xf32, affine_map<(d0, d1)[s0, s1] -> (d0*s0 + d1*s1)>>
    }
  }
  return
}

// CHECK-LABEL: func @alignment
func.func @alignment() {
  %A = memref.alloc() {alignment = 32 : i64}: memref<64x128x256xf32, affine_map<(d0, d1, d2) -> (d2, d0, d1)>>
  // CHECK-NEXT: memref.alloc() {alignment = 32 : i64} : memref<256x64x128xf32>
  return
}

#tile = affine_map < (i)->(i floordiv 4, i mod 4) >

// Following test cases check the inter-procedural memref normalization.

// Test case 1: Check normalization for multiple memrefs in a function argument list.
// CHECK-LABEL: func @multiple_argument_type
// CHECK-SAME:  (%[[A:arg[0-9a-zA-Z_]+]]: memref<4x4xf64>, %[[B:arg[0-9a-zA-Z_]+]]: f64, %[[C:arg[0-9a-zA-Z_]+]]: memref<2x4xf64>, %[[D:arg[0-9a-zA-Z_]+]]: memref<24xf64>) -> f64
func.func @multiple_argument_type(%A: memref<16xf64, #tile>, %B: f64, %C: memref<8xf64, #tile>, %D: memref<24xf64>) -> f64 {
  %a = affine.load %A[0] : memref<16xf64, #tile>
  %p = arith.mulf %a, %a : f64
  affine.store %p, %A[10] : memref<16xf64, #tile>
  call @single_argument_type(%C): (memref<8xf64, #tile>) -> ()
  return %B : f64
}

// CHECK: %[[a:[0-9a-zA-Z_]+]] = affine.load %[[A]][0, 0] : memref<4x4xf64>
// CHECK: %[[p:[0-9a-zA-Z_]+]] = arith.mulf %[[a]], %[[a]] : f64
// CHECK: affine.store %[[p]], %[[A]][2, 2] : memref<4x4xf64>
// CHECK: call @single_argument_type(%[[C]]) : (memref<2x4xf64>) -> ()
// CHECK: return %[[B]] : f64

// Test case 2: Check normalization for single memref argument in a function.
// CHECK-LABEL: func @single_argument_type
// CHECK-SAME: (%[[C:arg[0-9a-zA-Z_]+]]: memref<2x4xf64>)
func.func @single_argument_type(%C : memref<8xf64, #tile>) {
  %a = memref.alloc(): memref<8xf64, #tile>
  %b = memref.alloc(): memref<16xf64, #tile>
  %d = arith.constant 23.0 : f64
  %e = memref.alloc(): memref<24xf64>
  call @single_argument_type(%a): (memref<8xf64, #tile>) -> ()
  call @single_argument_type(%C): (memref<8xf64, #tile>) -> ()
  call @multiple_argument_type(%b, %d, %a, %e): (memref<16xf64, #tile>, f64, memref<8xf64, #tile>, memref<24xf64>) -> f64
  return
}

// CHECK: %[[a:[0-9a-zA-Z_]+]] = memref.alloc() : memref<2x4xf64>
// CHECK: %[[b:[0-9a-zA-Z_]+]] = memref.alloc() : memref<4x4xf64>
// CHECK: %cst = arith.constant 2.300000e+01 : f64
// CHECK: %[[e:[0-9a-zA-Z_]+]] = memref.alloc() : memref<24xf64>
// CHECK: call @single_argument_type(%[[a]]) : (memref<2x4xf64>) -> ()
// CHECK: call @single_argument_type(%[[C]]) : (memref<2x4xf64>) -> ()
// CHECK: call @multiple_argument_type(%[[b]], %cst, %[[a]], %[[e]]) : (memref<4x4xf64>, f64, memref<2x4xf64>, memref<24xf64>) -> f64

// Test case 3: Check function returning any other type except memref.
// CHECK-LABEL: func @non_memref_ret
// CHECK-SAME: (%[[C:arg[0-9a-zA-Z_]+]]: memref<2x4xf64>) -> i1
func.func @non_memref_ret(%A: memref<8xf64, #tile>) -> i1 {
  %d = arith.constant 1 : i1
  return %d : i1
}

// Test cases here onwards deal with normalization of memref in function signature, caller site.

// Test case 4: Check successful memref normalization in case of inter/intra-recursive calls.
// CHECK-LABEL: func @ret_multiple_argument_type
// CHECK-SAME: (%[[A:arg[0-9a-zA-Z_]+]]: memref<4x4xf64>, %[[B:arg[0-9a-zA-Z_]+]]: f64, %[[C:arg[0-9a-zA-Z_]+]]: memref<2x4xf64>) -> (memref<2x4xf64>, f64)
func.func @ret_multiple_argument_type(%A: memref<16xf64, #tile>, %B: f64, %C: memref<8xf64, #tile>) -> (memref<8xf64, #tile>, f64) {
  %a = affine.load %A[0] : memref<16xf64, #tile>
  %p = arith.mulf %a, %a : f64
  %cond = arith.constant 1 : i1
  cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %res1, %res2 = call @ret_single_argument_type(%C) : (memref<8xf64, #tile>) -> (memref<16xf64, #tile>, memref<8xf64, #tile>)
    return %res2, %p: memref<8xf64, #tile>, f64
  ^bb2:
    return %C, %p: memref<8xf64, #tile>, f64
}

// CHECK:   %[[a:[0-9a-zA-Z_]+]] = affine.load %[[A]][0, 0] : memref<4x4xf64>
// CHECK:   %[[p:[0-9a-zA-Z_]+]] = arith.mulf %[[a]], %[[a]] : f64
// CHECK:   %true = arith.constant true
// CHECK:   cf.cond_br %true, ^bb1, ^bb2
// CHECK: ^bb1:  // pred: ^bb0
// CHECK:   %[[res:[0-9a-zA-Z_]+]]:2 = call @ret_single_argument_type(%[[C]]) : (memref<2x4xf64>) -> (memref<4x4xf64>, memref<2x4xf64>)
// CHECK:   return %[[res]]#1, %[[p]] : memref<2x4xf64>, f64
// CHECK: ^bb2:  // pred: ^bb0
// CHECK:   return %{{.*}}, %{{.*}} : memref<2x4xf64>, f64

// CHECK-LABEL: func @ret_single_argument_type
// CHECK-SAME: (%[[C:arg[0-9a-zA-Z_]+]]: memref<2x4xf64>) -> (memref<4x4xf64>, memref<2x4xf64>)
func.func @ret_single_argument_type(%C: memref<8xf64, #tile>) -> (memref<16xf64, #tile>, memref<8xf64, #tile>){
  %a = memref.alloc() : memref<8xf64, #tile>
  %b = memref.alloc() : memref<16xf64, #tile>
  %d = arith.constant 23.0 : f64
  call @ret_single_argument_type(%a) : (memref<8xf64, #tile>) -> (memref<16xf64, #tile>, memref<8xf64, #tile>)
  call @ret_single_argument_type(%C) : (memref<8xf64, #tile>) -> (memref<16xf64, #tile>, memref<8xf64, #tile>)
  %res1, %res2 = call @ret_multiple_argument_type(%b, %d, %a) : (memref<16xf64, #tile>, f64, memref<8xf64, #tile>) -> (memref<8xf64, #tile>, f64)
  %res3, %res4 = call @ret_single_argument_type(%res1) : (memref<8xf64, #tile>) -> (memref<16xf64, #tile>, memref<8xf64, #tile>)
  return %b, %a: memref<16xf64, #tile>, memref<8xf64, #tile>
}

// CHECK: %[[a:[0-9a-zA-Z_]+]] = memref.alloc() : memref<2x4xf64>
// CHECK: %[[b:[0-9a-zA-Z_]+]] = memref.alloc() : memref<4x4xf64>
// CHECK: %cst = arith.constant 2.300000e+01 : f64
// CHECK: %[[resA:[0-9a-zA-Z_]+]]:2 = call @ret_single_argument_type(%[[a]]) : (memref<2x4xf64>) -> (memref<4x4xf64>, memref<2x4xf64>)
// CHECK: %[[resB:[0-9a-zA-Z_]+]]:2 = call @ret_single_argument_type(%[[C]]) : (memref<2x4xf64>) -> (memref<4x4xf64>, memref<2x4xf64>)
// CHECK: %[[resC:[0-9a-zA-Z_]+]]:2 = call @ret_multiple_argument_type(%[[b]], %cst, %[[a]]) : (memref<4x4xf64>, f64, memref<2x4xf64>) -> (memref<2x4xf64>, f64)
// CHECK: %[[resD:[0-9a-zA-Z_]+]]:2 = call @ret_single_argument_type(%[[resC]]#0) : (memref<2x4xf64>) -> (memref<4x4xf64>, memref<2x4xf64>)
// CHECK: return %{{.*}}, %{{.*}} : memref<4x4xf64>, memref<2x4xf64>

// Test case set #5: To check normalization in a chain of interconnected functions.
// CHECK-LABEL: func @func_A
// CHECK-SAME: (%[[A:arg[0-9a-zA-Z_]+]]: memref<2x4xf64>)
func.func @func_A(%A: memref<8xf64, #tile>) {
  call @func_B(%A) : (memref<8xf64, #tile>) -> ()
  return
}
// CHECK: call @func_B(%[[A]]) : (memref<2x4xf64>) -> ()

// CHECK-LABEL: func @func_B
// CHECK-SAME: (%[[A:arg[0-9a-zA-Z_]+]]: memref<2x4xf64>)
func.func @func_B(%A: memref<8xf64, #tile>) {
  call @func_C(%A) : (memref<8xf64, #tile>) -> ()
  return
}
// CHECK: call @func_C(%[[A]]) : (memref<2x4xf64>) -> ()

// CHECK-LABEL: func @func_C
// CHECK-SAME: (%[[A:arg[0-9a-zA-Z_]+]]: memref<2x4xf64>)
func.func @func_C(%A: memref<8xf64, #tile>) {
  return
}

// Test case set #6: Checking if no normalization takes place in a scenario: A -> B -> C and B has an unsupported type.
// CHECK-LABEL: func @some_func_A
// CHECK-SAME: (%[[A:arg[0-9a-zA-Z_]+]]: memref<8xf64, #map{{[0-9a-zA-Z_]+}}>)
func.func @some_func_A(%A: memref<8xf64, #tile>) {
  call @some_func_B(%A) : (memref<8xf64, #tile>) -> ()
  return
}
// CHECK: call @some_func_B(%[[A]]) : (memref<8xf64, #map{{[0-9a-zA-Z_]+}}>) -> ()

// CHECK-LABEL: func @some_func_B
// CHECK-SAME: (%[[A:arg[0-9a-zA-Z_]+]]: memref<8xf64, #map{{[0-9a-zA-Z_]+}}>)
func.func @some_func_B(%A: memref<8xf64, #tile>) {
  "test.test"(%A) : (memref<8xf64, #tile>) -> ()
  call @some_func_C(%A) : (memref<8xf64, #tile>) -> ()
  return
}
// CHECK: call @some_func_C(%[[A]]) : (memref<8xf64, #map{{[0-9a-zA-Z_]+}}>) -> ()

// CHECK-LABEL: func @some_func_C
// CHECK-SAME: (%[[A:arg[0-9a-zA-Z_]+]]: memref<8xf64, #map{{[0-9a-zA-Z_]+}}>)
func.func @some_func_C(%A: memref<8xf64, #tile>) {
  return
}

// Test case set #7: Check normalization in case of external functions.
// CHECK-LABEL: func private @external_func_A
// CHECK-SAME: (memref<4x4xf64>)
func.func private @external_func_A(memref<16xf64, #tile>) -> ()

// CHECK-LABEL: func private @external_func_B
// CHECK-SAME: (memref<4x4xf64>, f64) -> memref<2x4xf64>
func.func private @external_func_B(memref<16xf64, #tile>, f64) -> (memref<8xf64, #tile>)

// CHECK-LABEL: func @simply_call_external()
func.func @simply_call_external() {
  %a = memref.alloc() : memref<16xf64, #tile>
  call @external_func_A(%a) : (memref<16xf64, #tile>) -> ()
  return
}
// CHECK: %[[a:[0-9a-zA-Z_]+]] = memref.alloc() : memref<4x4xf64>
// CHECK: call @external_func_A(%[[a]]) : (memref<4x4xf64>) -> ()

// CHECK-LABEL: func @use_value_of_external
// CHECK-SAME: (%[[A:arg[0-9a-zA-Z_]+]]: memref<4x4xf64>, %[[B:arg[0-9a-zA-Z_]+]]: f64) -> memref<2x4xf64>
func.func @use_value_of_external(%A: memref<16xf64, #tile>, %B: f64) -> (memref<8xf64, #tile>) {
  %res = call @external_func_B(%A, %B) : (memref<16xf64, #tile>, f64) -> (memref<8xf64, #tile>)
  return %res : memref<8xf64, #tile>
}
// CHECK: %[[res:[0-9a-zA-Z_]+]] = call @external_func_B(%[[A]], %[[B]]) : (memref<4x4xf64>, f64) -> memref<2x4xf64>
// CHECK: return %{{.*}} : memref<2x4xf64>

// CHECK-LABEL: func @affine_parallel_norm
func.func @affine_parallel_norm() ->  memref<8xf32, #tile> {
  %c = arith.constant 23.0 : f32
  %a = memref.alloc() : memref<8xf32, #tile>
  // CHECK: affine.parallel (%{{.*}}) = (0) to (8) reduce ("assign") -> (memref<2x4xf32>)
  %1 = affine.parallel (%i) = (0) to (8) reduce ("assign") ->  memref<8xf32, #tile> {
    affine.store %c, %a[%i] : memref<8xf32, #tile>
    // CHECK: affine.yield %{{.*}} : memref<2x4xf32>
    affine.yield %a : memref<8xf32, #tile>
  }
  return %1 : memref<8xf32, #tile>
}

#map = affine_map<(d0, d1)[s0] -> (d0 * 3 + s0 + d1)>
// CHECK-LABEL: func.func @map_symbol
func.func @map_symbol() -> memref<2x3xf32, #map> {
  %c1 = arith.constant 1 : index
  // The constant isn't propagated here and the utility can't compute a constant
  // upper bound for the memref dimension in the absence of that.
  // CHECK: memref.alloc()[%{{.*}}]
  %0 = memref.alloc()[%c1] : memref<2x3xf32, #map>
  return %0 : memref<2x3xf32, #map>
}

#neg = affine_map<(d0, d1) -> (d0, d1 - 100)>
// CHECK-LABEL: func.func @neg_map
func.func @neg_map() -> memref<2x3xf32, #neg> {
  // This isn't a valid map for normalization.
  // CHECK: memref.alloc() : memref<2x3xf32, #{{.*}}>
  %0 = memref.alloc() : memref<2x3xf32, #neg>
  return %0 : memref<2x3xf32, #neg>
}

// CHECK-LABEL: func @memref_with_strided_offset
func.func @memref_with_strided_offset(%arg0: tensor<128x512xf32>, %arg1: index, %arg2: index) -> tensor<16x512xf32> {
  %c0 = arith.constant 0 : index
  %0 = bufferization.to_memref %arg0 : tensor<128x512xf32> to memref<128x512xf32, strided<[?, ?], offset: ?>>
  %subview = memref.subview %0[%arg2, 0] [%arg1, 512] [1, 1] : memref<128x512xf32, strided<[?, ?], offset: ?>> to memref<?x512xf32, strided<[?, ?], offset: ?>>
  // CHECK: %{{.*}} = memref.cast %{{.*}} : memref<?x512xf32, strided<[?, ?], offset: ?>> to memref<16x512xf32, strided<[?, ?], offset: ?>>
  %cast = memref.cast %subview : memref<?x512xf32, strided<[?, ?], offset: ?>> to memref<16x512xf32, strided<[?, ?], offset: ?>>
  %1 = bufferization.to_tensor %cast : memref<16x512xf32, strided<[?, ?], offset: ?>> to tensor<16x512xf32>
  return %1 : tensor<16x512xf32>
}

#map0 = affine_map<(i,k) -> (2 * (i mod 2) + (k mod 2) + 4 * (i floordiv 2) + 8 * (k floordiv 2))>
#map1 = affine_map<(k,j) -> ((k mod 2) + 2 * (j mod 2) + 8 * (k floordiv 2) + 4 * (j floordiv 2))>
#map2 = affine_map<(i,j) -> (4 * i + j)>
// CHECK-LABEL: func @memref_load_with_reduction_map
func.func @memref_load_with_reduction_map(%arg0 :  memref<4x4xf32,#map2>) -> () {
  %0 = memref.alloc() : memref<4x8xf32,#map0>
  %1 = memref.alloc() : memref<8x4xf32,#map1>
  %2 = memref.alloc() : memref<4x4xf32,#map2>
  // CHECK-NOT:  memref<4x8xf32>
  // CHECK-NOT:  memref<8x4xf32>
  // CHECK-NOT:  memref<4x4xf32>
  %cst = arith.constant 3.0 : f32
  %cst0 = arith.constant 0 : index
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 8 {
        // CHECK: %[[INDEX0:.*]] = affine.apply #[[$REDUCE_MAP1]](%{{.*}}, %{{.*}})
        // CHECK: memref.load %alloc[%[[INDEX0]]] : memref<32xf32>
        %a = memref.load %0[%i, %k] : memref<4x8xf32,#map0>
        // CHECK: %[[INDEX1:.*]] = affine.apply #[[$REDUCE_MAP2]](%{{.*}}, %{{.*}})
        // CHECK: memref.load %alloc_0[%[[INDEX1]]] : memref<32xf32>
        %b = memref.load %1[%k, %j] :memref<8x4xf32,#map1>
        // CHECK: %[[INDEX2:.*]] = affine.apply #[[$REDUCE_MAP3]](%{{.*}}, %{{.*}})
        // CHECK: memref.load %alloc_1[%[[INDEX2]]] : memref<16xf32>
        %c = memref.load %2[%i, %j] : memref<4x4xf32,#map2>
        %3 = arith.mulf %a, %b : f32
        %4 = arith.addf %3, %c : f32
        affine.store %4, %arg0[%i, %j] : memref<4x4xf32,#map2>
      }
    }
  }
  return
}