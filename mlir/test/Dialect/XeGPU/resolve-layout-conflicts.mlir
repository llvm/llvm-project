// RUN: mlir-opt --test-xegpu-resolve-layout-conflicts -split-input-file %s | FileCheck %s

#load_lo = #xegpu.layout<inst_data = [8, 16]>
#prefetch_lo = #xegpu.layout<inst_data = [16, 16]>
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
}
