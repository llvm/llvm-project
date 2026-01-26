// RUN: mlir-opt --test-xegpu-resolve-layout-conflicts -split-input-file %s | FileCheck %s

#load_lo = #xegpu.layout<inst_data = [8, 16]>
#prefetch_lo = #xegpu.layout<inst_data = [16, 16]>
#load_lo1 = #xegpu.layout<inst_data = [32, 16]>
gpu.module @test {

// CHECK-LABEL:   func.func @load_nd_with_conflicting_tensor_desc
// CHECK:           %{{.*}} = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK-NEXT:      %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [8, 16]>>
// CHECK-NEXT:      %{{.*}} = xegpu.load_nd %[[T1]][%{{.*}}, %{{.*}}] <{layout = #xegpu.layout<inst_data = [8, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<16x16xf16>
func.func @load_nd_with_conflicting_tensor_desc(%arg0: memref<64x64xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0 : memref<64x64xf16>
    -> !xegpu.tensor_desc<16x16xf16, #prefetch_lo>
  %1 = xegpu.load_nd %0 [%c0, %c0] {layout = #load_lo} : !xegpu.tensor_desc<16x16xf16, #prefetch_lo>
    -> vector<16x16xf16>
  xegpu.prefetch_nd %0 [%c0, %c0] {layout = #prefetch_lo} : !xegpu.tensor_desc<16x16xf16, #prefetch_lo>
  return %1 : vector<16x16xf16>
}

// CHECK-LABEL:   func.func @multiple_tensor_desc_conflicts
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [8, 16]>>
// CHECK-NEXT:      %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK-NEXT:      %[[T2:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [32, 16]>>
// CHECK-NEXT:      %{{.*}} = xegpu.load_nd %[[T0]][%[[C0]], %[[C0]]] <{layout = #xegpu.layout<inst_data = [8, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<32x16xf16>
// CHECK-NEXT:      %{{.*}} = xegpu.load_nd %[[T2]][%[[C0]], %[[C0]]] <{layout = #xegpu.layout<inst_data = [32, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [32, 16]>> -> vector<32x16xf16>
// CHECK-NEXT:      xegpu.prefetch_nd %[[T1]][%[[C0]], %[[C0]]] <{layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<32x16xf16, #xegpu.layout<inst_data = [16, 16]>>
func.func @multiple_tensor_desc_conflicts(%arg0: memref<64x64xf16>) -> vector<32x16xf16> {
  %c0 = arith.constant 0 : index
  %tdesc1 = xegpu.create_nd_tdesc %arg0 : memref<64x64xf16>
    -> !xegpu.tensor_desc<32x16xf16, #load_lo>
  %load1 = xegpu.load_nd %tdesc1 [%c0, %c0] {layout = #load_lo} : !xegpu.tensor_desc<32x16xf16, #load_lo>
    -> vector<32x16xf16>
  %load2 = xegpu.load_nd %tdesc1 [%c0, %c0] {layout = #load_lo1} : !xegpu.tensor_desc<32x16xf16, #load_lo>
    -> vector<32x16xf16>
  xegpu.prefetch_nd %tdesc1 [%c0, %c0] {layout = #prefetch_lo} : !xegpu.tensor_desc<32x16xf16, #load_lo>
  %result = arith.addf %load1, %load2 : vector<32x16xf16>
  return %result : vector<32x16xf16>
}

// CHECK-LABEL:   func.func @load_nd_with_conflicting_tensor_desc_in_loop
// CHECK:           %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK-NEXT:      %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<64x64xf16>
// CHECK-SAME:        -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [8, 16]>>
// CHECK-NEXT:      %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %[[T0]])
// CHECK-SAME:        -> (vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>) {
// CHECK-NEXT:        %{{.*}} = xegpu.load_nd %[[T1]][%{{.*}}] <{layout = #xegpu.layout<inst_data = [8, 16]>}> :
// CHECK-SAME:          !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [8, 16]>> -> vector<16x16xf16>
// CHECK:             scf.yield %{{.*}}, %{{.*}} : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK:           xegpu.prefetch_nd %[[T0]][%{{.*}}] <{layout = #xegpu.layout<inst_data = [16, 16]>}> :
// CHECK-SAME:        !xegpu.tensor_desc<16x16xf16, #xegpu.layout<inst_data = [16, 16]>>
// CHECK-NEXT:      return %{{.*}}#0 : vector<16x16xf16>
func.func @load_nd_with_conflicting_tensor_desc_in_loop(%arg0: memref<64x64xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant dense<0.0> : vector<16x16xf16>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<64x64xf16>
    -> !xegpu.tensor_desc<16x16xf16, #prefetch_lo>
  %1:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst, %tdesc = %0) -> (vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #prefetch_lo>) {
    %2 = xegpu.load_nd %tdesc [%c0, %c0] {layout = #load_lo} : !xegpu.tensor_desc<16x16xf16, #prefetch_lo>
      -> vector<16x16xf16>
    %3 = arith.addf %acc, %2 : vector<16x16xf16>
    scf.yield %3, %tdesc : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #prefetch_lo>
  }
  xegpu.prefetch_nd %0 [%c0, %c0] {layout = #prefetch_lo} : !xegpu.tensor_desc<16x16xf16, #prefetch_lo>
  return %1#0 : vector<16x16xf16>
}
}
