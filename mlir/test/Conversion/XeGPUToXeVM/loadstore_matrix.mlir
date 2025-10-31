// RUN: mlir-opt  -split-input-file -convert-xegpu-to-xevm -cse %s | FileCheck %s

gpu.module @test_kernel [#xevm.target<chip = "pvc">] {

 // e.g. for mem_desc<32x32xf16, @strides=[1, 16]>
  // its memory layout tuple is (blocked shape = [1,1,32,32],strides=[1024,1024,32,1])
  //CHECK-LABEL: load_store_matrix_1
  gpu.func @load_store_matrix_1(%arg0: memref<4096xi8, 3>) -> f32 {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x32xf32>

    //CHECK: %[[TID:.*]] = gpu.thread_id x
    //CHECK: %[[C1:.*]] = arith.constant 1 : index
    //CHECK: %[[MUL1:.*]] = arith.muli %[[TID]], %[[C1]] : index
    //CHECK: %[[C4:.*]] = arith.constant 4 : i32
    //CHECK: %[[MUL2:.*]] = arith.muli {{.*}}, %[[C4]] : i32
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f32

    %tid_x = gpu.thread_id x
    %c0 = arith.constant 0 : index
    %1 = xegpu.load_matrix %0[%c0, %tid_x]: !xegpu.mem_desc<32x32xf32>, index, index -> f32

    //CHECK: llvm.store {{.*}}, {{.*}} : f32, !llvm.ptr<3>

     xegpu.store_matrix %1, %0[%c0, %tid_x]: f32, !xegpu.mem_desc<32x32xf32>, index, index

    gpu.return %1: f32
  }

// e.g. for mem_desc<32x64xf16, @block=[16, 16], @strides=[1, 32]>
  // its memory layout tuple is ([2,4,16,16],[256,512,1,16])
  //CHECK-LABEL: load_store_matrix_2
  gpu.func @load_store_matrix_2(%arg0: memref<4096xi8, 3>) -> f16 {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[tid_x:.*]] = gpu.thread_id x
    //CHECK: %[[c13:.*]] = arith.constant 13 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_0:.*]] = arith.divsi %[[c13]], %[[c16]] : index
    //CHECK: %[[offsetx_1:.*]] = arith.remsi %[[c13]], %[[c16]] : index
    //CHECK: %[[offsety_0:.*]] = arith.divsi %[[tid_x]], %[[c16]] : index
    //CHECK: %[[offsety_1:.*]] = arith.remsi %[[tid_x]], %[[c16]] : index

    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[mul0:.*]] = arith.muli %[[offsetx_0]], %[[c256]] : index
    //CHECK: %[[add0:.*]] = arith.addi %[[mul0]], %[[c0]] : index
    //CHECK: %[[c512:.*]] = arith.constant 512 : index
    //CHECK: %[[mul1:.*]] = arith.muli %[[offsety_0]], %[[c512]] : index
    //CHECK: %[[add1:.*]] = arith.addi %[[mul1]], %[[add0]] : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[mul2:.*]] = arith.muli %[[offsetx_1]], %[[c1]] : index
    //CHECK: %[[add2:.*]] = arith.addi %[[mul2]], %[[add1]] : index
    //CHECK: %[[mul3:.*]] = arith.muli %[[offsety_1]], %[[c16]] : index
    //CHECK: %[[add3:.*]] = arith.addi %[[mul3]], %[[add2]] : index

    //CHECK: %[[loaded:.*]] = llvm.load {{.*}}: !llvm.ptr<3> -> f16
 

    %tid_x = gpu.thread_id x
    %c13 = arith.constant 13 : index
    %1 = xegpu.load_matrix %0[%c13, %tid_x]: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index -> f16

    //CHECK: llvm.store %[[loaded]], {{.*}} : f16, !llvm.ptr<3>
   
    xegpu.store_matrix %1, %0[%c13, %tid_x]: f16, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index 
    gpu.return %1: f16
  }


  // e.g. for mem_desc<32x64xf16, @block=[16, 16]>
  // its memory layout tuple is ([2,4,16,16],[1024,256,16,1])
  //CHECK-LABEL: load_store_matrix_3
  gpu.func @load_store_matrix_3(%arg0: memref<4096xi8, 3>) -> f16 {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[view:.*]] = memref.view %arg0[%[[c0]]][] : memref<4096xi8, 3> to memref<2048xf16, 3>
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>
    
    //CHECK: %[[tid_x:.*]] = gpu.thread_id x
    //CHECK: %[[c19:.*]] = arith.constant 19 : index
    %tid_x = gpu.thread_id x
    %c19 = arith.constant 19: index
    
    //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[view]] : memref<2048xf16, 3> -> index
    //CHECK: %[[basePtrI64:.*]] = arith.index_castui %[[intptr]] : index to i32
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_0:.*]] = arith.divsi %[[c19]], %[[c16]] : index
    //CHECK: %[[offsetx_1:.*]] = arith.remsi %[[c19]], %[[c16]] : index
    //CHECK: %[[offsety_0:.*]] = arith.divsi %[[tid_x]], %[[c16]] : index
    //CHECK: %[[offsety_1:.*]] = arith.remsi %[[tid_x]], %[[c16]] : index
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    //CHECK: %[[mul0:.*]] = arith.muli %[[offsetx_0]], %[[c1024]] : index
    //CHECK: %[[add0:.*]] = arith.addi %[[mul0]], %[[c0]] : index
    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[mul1:.*]] = arith.muli %[[offsety_0]], %[[c256]] : index
    //CHECK: %[[add1:.*]] = arith.addi %[[mul1]], %[[add0]] : index
    //CHECK: %[[mul2:.*]] = arith.muli %[[offsetx_1]], %[[c16]] : index
    //CHECK: %[[add2:.*]] = arith.addi %[[mul2]], %[[add1]] : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[mul3:.*]] = arith.muli %[[offsety_1]], %[[c1]] : index
    //CHECK: %[[add3:.*]] = arith.addi %[[mul3]], %[[add2]] : index

    //CHECK: %[[loaded:.*]] = llvm.load {{.*}} : !llvm.ptr<3> -> f16
    %1 = xegpu.load_matrix %0[%c19, %tid_x]: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index -> f16
    
    //CHECK: llvm.store %[[loaded]], {{.*}} : f16, !llvm.ptr<3>
    xegpu.store_matrix %1, %0[%c19, %tid_x]:  f16, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index
    
    //CHECK: gpu.return %[[loaded]] : f16
    gpu.return %1: f16
  }

   // e.g. for mem_desc<32x64xf16, @block=[16, 16], @strides=[1, 16]>
  // its memory layout tuple is ([2,4,16,16],[256,512,1,16])
  //CHECK-LABEL: load_store_matrix_4
  gpu.func @load_store_matrix_4(%arg0: memref<4096xi8, 3>) -> vector<8xf16> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>

    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[tid_x:.*]] = gpu.thread_id x

    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_0:.*]] = arith.divsi %[[c16]], %[[c16]] : index
    //CHECK: %[[offsetx_1:.*]] = arith.remsi %[[c16]], %[[c16]] : index
    //CHECK: %[[offsety_0:.*]] = arith.divsi %[[tid_x]], %[[c16]] : index
    //CHECK: %[[offsety_1:.*]] = arith.remsi %[[tid_x]], %[[c16]] : index

    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[mul0:.*]] = arith.muli %[[offsetx_0]], %[[c256]] : index
    //CHECK: %[[add0:.*]] = arith.addi %[[mul0]], %[[c0]] : index
    //CHECK: %[[c512:.*]] = arith.constant 512 : index
    //CHECK: %[[mul1:.*]] = arith.muli %[[offsety_0]], %[[c512]] : index
    //CHECK: %[[add1:.*]] = arith.addi %[[mul1]], %[[add0]] : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[mul2:.*]] = arith.muli %[[offsetx_1]], %[[c1]] : index
    //CHECK: %[[add2:.*]] = arith.addi %[[mul2]], %[[add1]] : index
    //CHECK: %[[mul3:.*]] = arith.muli %[[offsety_1]], %[[c16]] : index
    //CHECK: %[[add3:.*]] = arith.addi %[[mul3]], %[[add2]] : index

    //CHECK: %[[loaded:.*]] = llvm.load {{.*}}: !llvm.ptr<3> -> vector<8xf16>
     
    %tid_x = gpu.thread_id x
    %c16 = arith.constant 16 : index
    %1 = xegpu.load_matrix %0[%c16, %tid_x] : !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index -> vector<8xf16>

    //CHECK: llvm.store %[[loaded]], {{.*}} : vector<8xf16>, !llvm.ptr<3>
    xegpu.store_matrix %1, %0[%c16, %tid_x] : vector<8xf16>, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index

    gpu.return %1: vector<8xf16>
  }

 
  // e.g. for mem_desc<32x64xf16, @block=[16, 16]>
  // its memory layout tuple is ([2,4,16,16],[1024,256,16,1])
  //CHECK-LABEL: load_store_matrix_5
  gpu.func @load_store_matrix_5(%arg0: memref<4096xi8, 3>) -> vector<8xf16> {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[view:.*]] = memref.view %arg0[%[[c0]]][] : memref<4096xi8, 3> to memref<2048xf16, 3>
 
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>
 
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c48:.*]] = arith.constant 48 : index
  
    %c16 = arith.constant 16 : index
    %c48 = arith.constant 48 : index

    //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[view]] : memref<2048xf16, 3> -> index
    //CHECK: %[[basePtrI64:.*]] = arith.index_castui %[[intptr]] : index to i32
    //CHECK: %[[offset0:.*]] = arith.divsi %[[c16]], %[[c16]] : index
    //CHECK: %[[offset1:.*]] = arith.remsi %[[c16]], %[[c16]] : index
    //CHECK: %[[offset2:.*]] = arith.divsi %[[c48]], %[[c16]] : index
    //CHECK: %[[offset3:.*]] = arith.remsi %[[c48]], %[[c16]] : index
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    //CHECK: %[[mul0:.*]] = arith.muli %[[offset0]], %[[c1024]] : index
    //CHECK: %[[add0:.*]] = arith.addi %[[mul0]], %[[c0]] : index
    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[mul1:.*]] = arith.muli %[[offset2]], %[[c256]] : index
    //CHECK: %[[add1:.*]] = arith.addi %[[mul1]], %[[add0]] : index
    //CHECK: %[[mul2:.*]] = arith.muli %[[offset1]], %[[c16]] : index
    //CHECK: %[[add2:.*]] = arith.addi %[[mul2]], %[[add1]] : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[mul3:.*]] = arith.muli %[[offset3]], %[[c1]] : index
    //CHECK: %[[linearOffset:.*]] = arith.addi %[[mul3]], %[[add2]] : index
    //CHECK: %[[linearOffsetI64:.*]] = arith.index_castui %[[linearOffset]] : index to i32
    //CHECK: %[[c2:.*]] = arith.constant 2 : i32
    //CHECK: %[[byteOffset:.*]] = arith.muli %[[linearOffsetI64]], %[[c2]] : i32
    //CHECK: %[[finalPtr:.*]] = arith.addi %[[basePtrI64]], %[[byteOffset]] : i32
    //CHECK: %[[ptr:.*]] = llvm.inttoptr %[[finalPtr]] : i32 to !llvm.ptr<3>
    //CHECK: %[[loadedI16:.*]] = xevm.blockload %[[ptr]] : (!llvm.ptr<3>) -> vector<8xi16>
    //CHECK: %[[loaded:.*]] = vector.bitcast %[[loadedI16]] : vector<8xi16> to vector<8xf16>

    %1 = xegpu.load_matrix %0[%c16, %c48] {subgroup_block_io}: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index -> vector<8xf16>

    //CHECK: %[[storeDataI16:.*]] = vector.bitcast %[[loaded]] : vector<8xf16> to vector<8xi16>
    //CHECK: xevm.blockstore %[[ptr]], %[[storeDataI16]] : (!llvm.ptr<3>, vector<8xi16>) 

    xegpu.store_matrix %1, %0[%c16, %c48] {subgroup_block_io}: vector<8xf16>, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index

    gpu.return %1: vector<8xf16>
  }

}
