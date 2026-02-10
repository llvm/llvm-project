// RUN: mlir-opt  -split-input-file -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @test_kernel [#xevm.target<chip = "pvc">] {

  // e.g. for mem_desc<32x32xf16, @strides=[1, 16]>
  // its memory layout tuple is (blocked shape = [1,1,32,32],strides=[1024,1024,32,1])
  //CHECK-LABEL: load_store_matrix_plain
  gpu.func @load_store_matrix_plain(%arg0: memref<4096xi8, 3>) -> f32 {

    //CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<4096xi8, 3> -> index
    //CHECK: %[[C0:.*]] = arith.constant 0 : index
    //CHECK: %[[CAST0:.*]] = arith.index_castui %[[INTPTR]] : index to i32
    //CHECK: %[[CAST1:.*]] = arith.index_castui %[[C0]] : index to i32
    //CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
    //CHECK: %[[MUL:.*]] = arith.muli %[[CAST1]], %[[C1_I32]] : i32
    //CHECK: %[[ADD:.*]] = arith.addi %[[CAST0]], %[[MUL]] : i32
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

  //CHECK-LABEL: load_store_matrix_plain_2d_input
  gpu.func @load_store_matrix_plain_2d_input(%arg0: memref<1024xi8, 3>) -> f32 {
    %c0 = arith.constant 0 : index
    %view = memref.view %arg0[%c0][]: memref<1024xi8, 3> to memref<64x32xf32, 3>

    %subview = memref.subview %view[32, 0] [32, 32] [1, 1] : memref<64x32xf32, 3> to memref<32x32xf32, strided<[32, 1], offset: 1024>, 3>

    //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[base_buffer:.*]] : memref<32x32xf32, strided<[32, 1], offset: 1024>, 3> -> index
    //CHECK: %[[ptr_i32:.*]] = arith.index_castui %[[intptr]] : index to i32
    //CHECK: %[[offset_i32:.*]] = arith.index_castui %[[offset:.*]] : index to i32
    //CHECK: %[[c4_i32:.*]] = arith.constant 4 : i32
    //CHECK: %[[mul:.*]] = arith.muli %[[offset_i32]], %[[c4_i32]] : i32
    //CHECK: %[[add:.*]] = arith.addi %[[ptr_i32]], %[[mul]] : i32

    %0 = xegpu.create_mem_desc %subview : memref<32x32xf32, strided<[32, 1], offset: 1024>, 3> -> !xegpu.mem_desc<32x32xf32>

    //CHECK: %[[TID:.*]] = gpu.thread_id x
    //CHECK: %[[C1:.*]] = arith.constant 1 : index
    //CHECK: %[[MUL1:.*]] = arith.muli %[[TID]], %[[C1]] : index
    //CHECK: %[[MUL2:.*]] = arith.muli {{.*}}, {{.*}} : i32
    //CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f32

    %tid_x = gpu.thread_id x
    %1 = xegpu.load_matrix %0[%c0, %tid_x]: !xegpu.mem_desc<32x32xf32>, index, index -> f32

    //CHECK: llvm.store {{.*}}, {{.*}} : f32, !llvm.ptr<3>

    xegpu.store_matrix %1, %0[%c0, %tid_x]: f32, !xegpu.mem_desc<32x32xf32>, index, index

    gpu.return %1: f32
  }


// e.g. for mem_desc<32x64xf16, @block=[16, 16], @strides=[1, 32]>
  // its memory layout tuple is ([2,4,16,16],[256,512,1,16])
  //CHECK-LABEL: load_store_matrix_blocked_strided
  gpu.func @load_store_matrix_blocked_strided(%arg0: memref<4096xi8, 3>) -> f16 {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>

    //CHECK: %[[tid_x:.*]] = gpu.thread_id x
    //CHECK: %[[c13:.*]] = arith.constant 13 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_0:.*]] = arith.divsi %[[c13]], %[[c16]] : index
    //CHECK: %[[c16_0:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_1:.*]] = arith.remsi %[[c13]], %[[c16_0]] : index
    //CHECK: %[[c16_1:.*]] = arith.constant 16 : index
    //CHECK: %[[offsety_0:.*]] = arith.divsi %[[tid_x]], %[[c16_1]] : index
    //CHECK: %[[c16_2:.*]] = arith.constant 16 : index
    //CHECK: %[[offsety_1:.*]] = arith.remsi %[[tid_x]], %[[c16_2]] : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[mul0:.*]] = arith.muli %[[offsetx_0]], %[[c256]] : index
    //CHECK: %[[add0:.*]] = arith.addi %[[mul0]], %[[c0]] : index
    //CHECK: %[[c512:.*]] = arith.constant 512 : index
    //CHECK: %[[mul1:.*]] = arith.muli %[[offsety_0]], %[[c512]] : index
    //CHECK: %[[add1:.*]] = arith.addi %[[mul1]], %[[add0]] : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[mul2:.*]] = arith.muli %[[offsetx_1]], %[[c1]] : index
    //CHECK: %[[add2:.*]] = arith.addi %[[mul2]], %[[add1]] : index
    //CHECK: %[[c16_4:.*]] = arith.constant 16 : index
    //CHECK: %[[mul3:.*]] = arith.muli %[[offsety_1]], %[[c16_4]] : index
    //CHECK: %[[add3:.*]] = arith.addi %[[mul3]], %[[add2]] : index
    //CHECK: %[[cast:.*]] = arith.index_castui %[[add3]] : index to i32
    //CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
    //CHECK: %[[byte_offset:.*]] = arith.muli %[[cast]], %[[c2_i32]] : i32
    //CHECK: %[[final_ptr:.*]] = arith.addi {{.*}}, %[[byte_offset]] : i32
    //CHECK: %[[ptr:.*]] = llvm.inttoptr %[[final_ptr]] : i32 to !llvm.ptr<3>
    //CHECK: %[[loaded:.*]] = llvm.load %[[ptr]] : !llvm.ptr<3> -> f16

    %tid_x = gpu.thread_id x
    %c13 = arith.constant 13 : index
    %1 = xegpu.load_matrix %0[%c13, %tid_x]: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index -> f16

    //CHECK: llvm.store %[[loaded]], {{.*}} : f16, !llvm.ptr<3>

    xegpu.store_matrix %1, %0[%c13, %tid_x]: f16, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index
    gpu.return %1: f16
  }

  // e.g. for mem_desc<32x64xf16, @block=[16, 16]>
  // its memory layout tuple is ([2,4,16,16],[1024,256,16,1])
  //CHECK-LABEL: load_store_matrix_blocked_nostride
  gpu.func @load_store_matrix_blocked_nostride(%arg0: memref<4096xi8, 3>) -> f16 {
    //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<4096xi8, 3> -> index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[cast0:.*]] = arith.index_castui %[[intptr]] : index to i32
    //CHECK: %[[cast1:.*]] = arith.index_castui %[[c0]] : index to i32
    //CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
    //CHECK: %[[mul:.*]] = arith.muli %[[cast1]], %[[c1_i32]] : i32
    //CHECK: %[[add:.*]] = arith.addi %[[cast0]], %[[mul]] : i32
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>

    //CHECK: %[[tid_x:.*]] = gpu.thread_id x
    //CHECK: %[[c19:.*]] = arith.constant 19 : index
    %tid_x = gpu.thread_id x
    %c19 = arith.constant 19: index

    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_0:.*]] = arith.divsi %[[c19]], %[[c16]] : index
    //CHECK: %[[c16_0:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_1:.*]] = arith.remsi %[[c19]], %[[c16_0]] : index
    //CHECK: %[[c16_1:.*]] = arith.constant 16 : index
    //CHECK: %[[offsety_0:.*]] = arith.divsi %[[tid_x]], %[[c16_1]] : index
    //CHECK: %[[c16_2:.*]] = arith.constant 16 : index
    //CHECK: %[[offsety_1:.*]] = arith.remsi %[[tid_x]], %[[c16_2]] : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    //CHECK: %[[mul0:.*]] = arith.muli %[[offsetx_0]], %[[c1024]] : index
    //CHECK: %[[add0:.*]] = arith.addi %[[mul0]], %[[c0]] : index
    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[mul1:.*]] = arith.muli %[[offsety_0]], %[[c256]] : index
    //CHECK: %[[add1:.*]] = arith.addi %[[mul1]], %[[add0]] : index
    //CHECK: %[[c16_4:.*]] = arith.constant 16 : index
    //CHECK: %[[mul2:.*]] = arith.muli %[[offsetx_1]], %[[c16_4]] : index
    //CHECK: %[[add2:.*]] = arith.addi %[[mul2]], %[[add1]] : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[mul3:.*]] = arith.muli %[[offsety_1]], %[[c1]] : index
    //CHECK: %[[add3:.*]] = arith.addi %[[mul3]], %[[add2]] : index
    //CHECK: %[[cast:.*]] = arith.index_castui %[[add3]] : index to i32
    //CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
    //CHECK: %[[byte_offset:.*]] = arith.muli %[[cast]], %[[c2_i32]] : i32
    //CHECK: %[[final_ptr:.*]] = arith.addi {{.*}}, %[[byte_offset]] : i32
    //CHECK: %[[ptr:.*]] = llvm.inttoptr %[[final_ptr]] : i32 to !llvm.ptr<3>
    //CHECK: %[[loaded:.*]] = llvm.load %[[ptr]] : !llvm.ptr<3> -> f16
    %1 = xegpu.load_matrix %0[%c19, %tid_x]: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index -> f16

    //CHECK: llvm.store %[[loaded]], {{.*}} : f16, !llvm.ptr<3>
    xegpu.store_matrix %1, %0[%c19, %tid_x]:  f16, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index

    //CHECK: gpu.return %[[loaded]] : f16
    gpu.return %1: f16
  }

   // e.g. for mem_desc<32x64xf16, @block=[16, 16], @strides=[1, 16]>
  // its memory layout tuple is ([2,4,16,16],[256,512,1,16])
  //CHECK-LABEL: load_store_matrix_blocked_strided_return_vector
  gpu.func @load_store_matrix_blocked_strided_return_vector(%arg0: memref<4096xi8, 3>) -> vector<8xf16> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>

    //CHECK: %[[tid_x:.*]] = gpu.thread_id x
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c16_0:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_0:.*]] = arith.divsi %[[c16]], %[[c16_0]] : index
    //CHECK: %[[c16_1:.*]] = arith.constant 16 : index
    //CHECK: %[[offsetx_1:.*]] = arith.remsi %[[c16]], %[[c16_1]] : index
    //CHECK: %[[c16_2:.*]] = arith.constant 16 : index
    //CHECK: %[[offsety_0:.*]] = arith.divsi %[[tid_x]], %[[c16_2]] : index
    //CHECK: %[[c16_3:.*]] = arith.constant 16 : index
    //CHECK: %[[offsety_1:.*]] = arith.remsi %[[tid_x]], %[[c16_3]] : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[mul0:.*]] = arith.muli %[[offsetx_0]], %[[c256]] : index
    //CHECK: %[[add0:.*]] = arith.addi %[[mul0]], %[[c0]] : index
    //CHECK: %[[c512:.*]] = arith.constant 512 : index
    //CHECK: %[[mul1:.*]] = arith.muli %[[offsety_0]], %[[c512]] : index
    //CHECK: %[[add1:.*]] = arith.addi %[[mul1]], %[[add0]] : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[mul2:.*]] = arith.muli %[[offsetx_1]], %[[c1]] : index
    //CHECK: %[[add2:.*]] = arith.addi %[[mul2]], %[[add1]] : index
    //CHECK: %[[c16_5:.*]] = arith.constant 16 : index
    //CHECK: %[[mul3:.*]] = arith.muli %[[offsety_1]], %[[c16_5]] : index
    //CHECK: %[[add3:.*]] = arith.addi %[[mul3]], %[[add2]] : index
    //CHECK: %[[cast:.*]] = arith.index_castui %[[add3]] : index to i32
    //CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
    //CHECK: %[[byte_offset:.*]] = arith.muli %[[cast]], %[[c2_i32]] : i32
    //CHECK: %[[final_ptr:.*]] = arith.addi {{.*}}, %[[byte_offset]] : i32
    //CHECK: %[[ptr:.*]] = llvm.inttoptr %[[final_ptr]] : i32 to !llvm.ptr<3>
    //CHECK: %[[loaded:.*]] = llvm.load %[[ptr]] : !llvm.ptr<3> -> vector<8xf16>

    %tid_x = gpu.thread_id x
    %c16 = arith.constant 16 : index
    %1 = xegpu.load_matrix %0[%c16, %tid_x] : !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index -> vector<8xf16>

    //CHECK: llvm.store %[[loaded]], {{.*}} : vector<8xf16>, !llvm.ptr<3>
    xegpu.store_matrix %1, %0[%c16, %tid_x] : vector<8xf16>, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<stride = [1, 32], block = [16, 16]>>, index, index

    gpu.return %1: vector<8xf16>
  }


  // e.g. for mem_desc<32x64xf16, @block=[16, 16]>
  // its memory layout tuple is ([2,4,16,16],[1024,256,16,1])
  //CHECK-LABEL: load_store_matrix_blocked_subgroupblockio
  gpu.func @load_store_matrix_blocked_subgroupblockio(%arg0: memref<4096xi8, 3>) -> vector<8xf16> {

    //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<4096xi8, 3> -> index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[cast0:.*]] = arith.index_castui %[[intptr]] : index to i32
    //CHECK: %[[cast1:.*]] = arith.index_castui %[[c0]] : index to i32
    //CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
    //CHECK: %[[mul:.*]] = arith.muli %[[cast1]], %[[c1_i32]] : i32
    //CHECK: %[[add:.*]] = arith.addi %[[cast0]], %[[mul]] : i32
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c48:.*]] = arith.constant 48 : index
    //CHECK: %[[c16_0:.*]] = arith.constant 16 : index
    //CHECK: %[[divsi0:.*]] = arith.divsi %[[c16]], %[[c16_0]] : index
    //CHECK: %[[c16_1:.*]] = arith.constant 16 : index
    //CHECK: %[[remsi0:.*]] = arith.remsi %[[c16]], %[[c16_1]] : index
    //CHECK: %[[c16_2:.*]] = arith.constant 16 : index
    //CHECK: %[[divsi1:.*]] = arith.divsi %[[c48]], %[[c16_2]] : index
    //CHECK: %[[c16_3:.*]] = arith.constant 16 : index
    //CHECK: %[[remsi1:.*]] = arith.remsi %[[c48]], %[[c16_3]] : index
    //CHECK: %[[c0_4:.*]] = arith.constant 0 : index
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    //CHECK: %[[mul0:.*]] = arith.muli %[[divsi0]], %[[c1024]] : index
    //CHECK: %[[add0:.*]] = arith.addi %[[mul0]], %[[c0_4]] : index
    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[mul1:.*]] = arith.muli %[[divsi1]], %[[c256]] : index
    //CHECK: %[[add1:.*]] = arith.addi %[[mul1]], %[[add0]] : index
    //CHECK: %[[c16_5:.*]] = arith.constant 16 : index
    //CHECK: %[[mul2:.*]] = arith.muli %[[remsi0]], %[[c16_5]] : index
    //CHECK: %[[add2:.*]] = arith.addi %[[mul2]], %[[add1]] : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[mul3:.*]] = arith.muli %[[remsi1]], %[[c1]] : index
    //CHECK: %[[add3:.*]] = arith.addi %[[mul3]], %[[add2]] : index
    //CHECK: %[[cast:.*]] = arith.index_castui %[[add3]] : index to i32
    //CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
    //CHECK: %[[byte_offset:.*]] = arith.muli %[[cast]], %[[c2_i32]] : i32
    //CHECK: %[[final_ptr:.*]] = arith.addi %[[add]], %[[byte_offset]] : i32
    //CHECK: %[[ptr:.*]] = llvm.inttoptr %[[final_ptr]] : i32 to !llvm.ptr<3>
    //CHECK: %[[blockload:.*]] = xevm.blockload %[[ptr]] : (!llvm.ptr<3>) -> vector<8xi16>
    //CHECK: %[[loaded:.*]] = vector.bitcast %[[blockload]] : vector<8xi16> to vector<8xf16>
    
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>

    %c16 = arith.constant 16 : index
    %c48 = arith.constant 48 : index

    %1 = xegpu.load_matrix %0[%c16, %c48] {subgroup_block_io}: !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index -> vector<8xf16>

    //CHECK: %[[storeDataI16:.*]] = vector.bitcast %[[loaded]] : vector<8xf16> to vector<8xi16>
    //CHECK: xevm.blockstore %[[ptr2:.*]], %[[storeDataI16]] : (!llvm.ptr<3>, vector<8xi16>)

    xegpu.store_matrix %1, %0[%c16, %c48] {subgroup_block_io}: vector<8xf16>, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>, index, index

    gpu.return %1: vector<8xf16>
  }

  gpu.func @matrix_vector_materialization(%matrixdesc : !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>) {
    // CHECK: %[[XEVM_VECTOR:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xf16>
    // CHECK: %[[SOURCE_MATERIALIZE:.*]] = vector.shape_cast %[[XEVM_VECTOR]] : vector<16xf16> to vector<1x16xf16>
    // CHECK: %[[XEGPU_VECTOR:.*]] = arith.addf %[[SOURCE_MATERIALIZE]], %[[SOURCE_MATERIALIZE]] : vector<1x16xf16>
    // CHECK: %[[TARGET_MATERIALIZE:.*]] = vector.shape_cast %[[XEGPU_VECTOR]] : vector<1x16xf16> to vector<16xf16>
    // CHECK: llvm.store %[[TARGET_MATERIALIZE]], %{{.*}} : vector<16xf16>, !llvm.ptr<3>
    %loaded = xegpu.load_matrix %matrixdesc[16,0] : !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>> -> vector<1x16xf16>
    %loaded_2 = arith.addf %loaded, %loaded : vector<1x16xf16>
    xegpu.store_matrix %loaded_2, %matrixdesc[16,0] : vector<1x16xf16>, !xegpu.mem_desc<32x64xf16, #xegpu.mem_layout<block = [16, 16]>>
    gpu.return
  }

  //CHECK-LABEL: create_memdesc_from_subview
  gpu.func @create_memdesc_from_subview(%arg0: memref<256x16xbf16, 3>) -> vector<1x16xbf16> {

    %c0 = arith.constant 0 : index

  %smem_coop_a = memref.subview %arg0[64, 0][1, 16][1, 1] : memref<256x16xbf16, 3> to memref<1x16xbf16, strided<[16, 1], offset: 1024>, 3>

  //CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<1x16xbf16, strided<[16, 1], offset: 1024>, 3> -> index
  //CHECK: %[[C1024:.*]] = arith.constant 1024 : index
  //CHECK: %[[CAST0:.*]] = arith.index_castui %[[INTPTR]] : index to i32
  //CHECK: %[[CAST1:.*]] = arith.index_castui %[[C1024]] : index to i32
  //CHECK: %[[C2:.*]] = arith.constant 2 : i32
  //CHECK: %[[MUL:.*]] = arith.muli %[[CAST1]], %[[C2]] : i32
  //CHECK: %{{.*}} = arith.addi %[[CAST0]], %[[MUL]] : i32

  %mdesc_coop_a = xegpu.create_mem_desc %smem_coop_a : memref<1x16xbf16, strided<[16, 1], offset: 1024>, 3> -> !xegpu.mem_desc<1x16xbf16>

  %ret = xegpu.load_matrix%mdesc_coop_a[%c0, %c0]: !xegpu.mem_desc<1x16xbf16>, index, index -> vector<1x16xbf16>

  gpu.return  %ret : vector<1x16xbf16>

  }


}
