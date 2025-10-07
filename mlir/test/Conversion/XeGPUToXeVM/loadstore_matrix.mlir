// RUN: mlir-opt  -split-input-file -convert-xegpu-to-xevm --cse --canonicalize %s | FileCheck %s

gpu.module @test_kernel {

  // e.g. for mem_desc<32x32xf16, @strides=[1, 16]>
  // its memory layout tuple is (blocked shape = [1,1,32,32],strides=[1024,1024,32,1])
  //CHECK-LABEL: load_store_matrix_1
  gpu.func @load_store_matrix_1(%arg0: memref<4096xi8, 3>) -> vector<1xf32> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x32xf32>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf32>
    %tid_x = gpu.thread_id x
    %c0 = arith.constant 0 : index
    %1 = xegpu.load_matrix %0[%c0, %tid_x]: !xegpu.mem_desc<32x32xf32>, index, index -> vector<1xf32>
    gpu.return %1: vector<1xf32>
  }

  // e.g. for mem_desc<32x64xf16, @block=[16, 16], @strides=[1, 16]>
  // its memory layout tuple is ([2,4,16,16],[256,512,1,16])
  //CHECK-LABEL: load_store_matrix_2
  gpu.func @load_store_matrix_2(%arg0: memref<4096xi8, 3>) -> vector<1xf16> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %tid_x = gpu.thread_id x
    %c13 = arith.constant 13 : index
    %1 = xegpu.load_matrix %0[%c13, %tid_x]: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index -> vector<1xf16>
    gpu.return %1: vector<1xf16>
  }

  // e.g. for mem_desc<32x64xf16, @block=[16, 16]>
  // its memory layout tuple is ([2,4,16,16],[1024,256,16,1])
  //CHECK-LABEL: load_store_matrix_3
  gpu.func @load_store_matrix_3(%arg0: memref<4096xi8, 3>) -> vector<1xf16> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %tid_x = gpu.thread_id x
    %c17 = arith.constant 17 : index
    %1 = xegpu.load_matrix %0[%c17, %tid_x]: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index -> vector<1xf16>
    gpu.return %1: vector<1xf16>
  }

}