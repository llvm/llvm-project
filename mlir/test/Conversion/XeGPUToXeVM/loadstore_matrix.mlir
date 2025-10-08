// RUN: mlir-opt  -split-input-file -convert-xegpu-to-xevm --cse --canonicalize %s | FileCheck %s

gpu.module @test_kernel [#xevm.target<chip = "pvc">] {

  // e.g. for mem_desc<32x32xf16, @strides=[1, 16]>
  // its memory layout tuple is (blocked shape = [1,1,32,32],strides=[1024,1024,32,1])
  //CHECK-LABEL: load_store_matrix_1
  gpu.func @load_store_matrix_1(%arg0: memref<4096xi8, 3>) -> f32 {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x32xf32>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f32
    %tid_x = gpu.thread_id x
    %c0 = arith.constant 0 : index
    %1 = xegpu.load_matrix %0[%c0, %tid_x]: !xegpu.mem_desc<32x32xf32>, index, index -> f32
    gpu.return %1: f32
  }

  // e.g. for mem_desc<32x64xf16, @block=[16, 16], @strides=[1, 32]>
  // its memory layout tuple is ([2,4,16,16],[256,512,1,16])
  //CHECK-LABEL: load_store_matrix_2
  gpu.func @load_store_matrix_2(%arg0: memref<4096xi8, 3>) -> f16 {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f16
    %tid_x = gpu.thread_id x
    %c13 = arith.constant 13 : index
    %1 = xegpu.load_matrix %0[%c13, %tid_x]: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index -> f16
    gpu.return %1: f16
  }

  // e.g. for mem_desc<32x64xf16, @block=[16, 16]>
  // its memory layout tuple is ([2,4,16,16],[1024,256,16,1])
  //CHECK-LABEL: load_store_matrix_3
  gpu.func @load_store_matrix_3(%arg0: memref<4096xi8, 3>) -> f16 {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f16
    %tid_x = gpu.thread_id x
    %c19 = arith.constant 19: index
    %1 = xegpu.load_matrix %0[%c19, %tid_x]: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index -> f16
    gpu.return %1: f16
  }
  
  // e.g. for mem_desc<32x64xf16, @block=[16, 16], @strides=[1, 16]>
  // its memory layout tuple is ([2,4,16,16],[256,512,1,16])
  //CHECK-LABEL: load_store_matrix_4
  gpu.func @load_store_matrix_4(%arg0: memref<4096xi8, 3>) -> vector<8xf16> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %tid_x = gpu.thread_id x
    %c16 = arith.constant 16 : index
    %1 = xegpu.load_matrix %0[%c16, %tid_x] {vec_length = 8 : i32, vec_direction = #xegpu.matrix_access_direction<col>}: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index -> vector<8xf16>
    gpu.return %1: vector<8xf16>
  }

  // e.g. for mem_desc<32x64xf16, @block=[16, 16]>
  // its memory layout tuple is ([2,4,16,16],[1024,256,16,1])
  //CHECK-LABEL: load_store_matrix_5
  gpu.func @load_store_matrix_5(%arg0: memref<4096xi8, 3>) -> vector<8xf16> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %c16 = arith.constant 16 : index
    %c48 = arith.constant 48 : index
    %1 = xegpu.load_matrix %0[%c16, %c48] {subgroup_block_io}: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index -> vector<8xf16>
    gpu.return %1: vector<8xf16>
  }

}