// RUN: mlir-opt  -split-input-file -convert-xegpu-to-xevm --cse --canonicalize %s | FileCheck %s

gpu.module @test_kernel {
  //CHECK-LABEL: load_store_matrix_1
  gpu.func @load_store_matrix_1(%arg0: memref<4096xi8, 3>) -> vector<8xf32> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x32xf32>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf32>
    %tid_x = gpu.thread_id x
    %c0 = arith.constant 0 : index
    %1 = xegpu.load_matrix %0[%c0, %tid_x]: !xegpu.mem_desc<32x32xf32>, index, index -> vector<8xf32>
    gpu.return %1: vector<8xf32>
  }

  // e.g. for mem_desc<32x32xf16, @block=[16, 16], @strides=[1, 16]>
  // its memory layout tuple is ([2,2,16,16],[256,512,1,16])

  //CHECK-LABEL: load_store_matrix_2
  gpu.func @load_store_matrix_2(%arg0: memref<4096xi8, 3>) -> vector<8xf32> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x32xf16, #xegpu.mem_layout<stride = [1, 16], block = [16, 16]>>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf32>
    %tid_x = gpu.thread_id x
    %c0 = arith.constant 0 : index
    %1 = xegpu.load_matrix %0[%c0, %tid_x]: !xegpu.mem_desc<32x32xf32>, index, index -> vector<8xf32>
    gpu.return %1: vector<8xf32>
  }

  // e.g. for mem_desc<32x32xf16, @block=[16, 16]>
  // its memory layout tuple is ([2,2,16,16],[512,256,16,1])
  //CHECK-LABEL: load_store_matrix_3
  gpu.func @load_store_matrix_3(%arg0: memref<4096xi8, 3>) -> vector<8xf32> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x32xf16, #xegpu.mem_layout<block = [16, 16]>>
    //CHECK-COUNT-8: xegpu.load_matrix {{.*}} : !xegpu.mem_desc<32x32xf32>, index, index -> vector<8x16xf32>
    //CHECK-COUNT-8: vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<32x32xf32>
    %tid_x = gpu.thread_id x
    %c0 = arith.constant 0 : index
    %1 = xegpu.load_matrix %0[%c0, %tid_x]: !xegpu.mem_desc<32x32xf32>, index, index -> vector<8xf32>
    gpu.return %1: vector<8xf32>
  }

}
