// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(xegpu-wg-to-sg-distribute, xegpu-blocking, xegpu-sg-to-lane-distribute), xevm-attach-target{chip=pvc}, gpu.module(convert-xegpu-to-xevm))" \
// RUN: | FileCheck %s

#layout_1d = #xegpu.layout<sg_layout = [16], sg_data = [256], inst_data = [16], lane_layout = [16], lane_data = [1]>
#layout_2d = #xegpu.layout<sg_layout = [4, 4], sg_data = [16, 32], inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>

gpu.module @test {

  // CHECK-LABEL: gpu.func @test_load_store_matrix_1d
  // CHECK: memref.extract_aligned_pointer_as_index %arg0 : memref<8192xi8, 3> -> index
  // CHECK: gpu.subgroup_id
  // CHECK: gpu.lane_id
  // CHECK-COUNT-16: llvm.load %{{.*}} : !llvm.ptr<3> -> bf16
  // CHECK-COUNT-16: llvm.store %{{.*}}, %{{.*}} : bf16, !llvm.ptr<3>
  // CHECK-NOT: xegpu.load_matrix
  // CHECK-NOT: xegpu.store_matrix
  gpu.func @test_load_store_matrix_1d(%src: memref<8192xi8, 3>) {
    %c0 = arith.constant 0 : index
    %mdesc = xegpu.create_mem_desc %src : memref<8192xi8, 3> -> !xegpu.mem_desc<4096xbf16>
    %data = xegpu.load_matrix %mdesc[%c0] {layout = #layout_1d} : !xegpu.mem_desc<4096xbf16>, index -> vector<4096xbf16>
    xegpu.store_matrix %data, %mdesc[%c0] {layout = #layout_1d} : vector<4096xbf16>, !xegpu.mem_desc<4096xbf16>, index
    gpu.return
  }

  // CHECK-LABEL: gpu.func @test_load_store_matrix_2d
  // CHECK: memref.extract_aligned_pointer_as_index %arg0 : memref<16384xi8, 3> -> index
  // CHECK: gpu.subgroup_id
  // CHECK: gpu.lane_id
  // CHECK-COUNT-32: llvm.load %{{.*}} : !llvm.ptr<3> -> bf16
  // CHECK-COUNT-32: llvm.store %{{.*}}, %{{.*}} : bf16, !llvm.ptr<3>
  // CHECK-NOT: xegpu.load_matrix
  // CHECK-NOT: xegpu.store_matrix
  gpu.func @test_load_store_matrix_2d(%src: memref<16384xi8, 3>) {
    %c0 = arith.constant 0 : index
    %mdesc = xegpu.create_mem_desc %src : memref<16384xi8, 3> -> !xegpu.mem_desc<64x128xbf16>
    %data = xegpu.load_matrix %mdesc[%c0, %c0] {layout = #layout_2d} : !xegpu.mem_desc<64x128xbf16>, index, index -> vector<64x128xbf16>
    xegpu.store_matrix %data, %mdesc[%c0, %c0] {layout = #layout_2d} : vector<64x128xbf16>, !xegpu.mem_desc<64x128xbf16>, index, index
    gpu.return
  }
}
