// RUN: mlir-opt --xegpu-blocking -split-input-file %s | FileCheck %s

#a = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#b = #xegpu.layout<inst_data = [16, 16], lane_layout = [1, 16], lane_data = [16, 1]>
#c = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>
gpu.module @test_kernel {
  gpu.func @test_gemm_with_one_to_n_lowering(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c16 : index
    %n = arith.muli %block_id_y, %c32 : index

    %c_tdesc = xegpu.create_nd_tdesc %C[%m, %n] : memref<1024x1024xf32> -> !xegpu.tensor_desc<16x32xf32, #c>
    %c_init = xegpu.load_nd %c_tdesc : !xegpu.tensor_desc<16x32xf32, #c> -> vector<16x32xf32>

    %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16, #a>
    %b_tdesc = xegpu.create_nd_tdesc %B[%c0, %n] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #b>
    %out:3 = scf.for %k = %c0 to %c1024 step %c32
      iter_args(%arg0 = %a_tdesc, %arg1 = %b_tdesc, %arg2 = %c_init)
      -> (!xegpu.tensor_desc<16x32xf16, #a>, !xegpu.tensor_desc<32x32xf16, #b>, vector<16x32xf32>) {
      //CHECK-COUNT-4: xegpu.load_nd {{.*}} -> vector<8x16xf16>
      %a = xegpu.load_nd %arg0 : !xegpu.tensor_desc<16x32xf16, #a> -> vector<16x32xf16>
      //CHECK-COUNT-4: xegpu.load_nd {{.*}} -> vector<16x16xf16>
      %b = xegpu.load_nd %arg1 : !xegpu.tensor_desc<32x32xf16, #b> -> vector<32x32xf16>
      //CHECK-COUNT-8: xegpu.dpas {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %c = xegpu.dpas %a, %b, %arg2 {layout_result_0 = #c}: vector<16x32xf16>, vector<32x32xf16>, vector<16x32xf32> -> vector<16x32xf32>
      //CHECK-COUNT-4: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>>
      %a_next_tdesc = xegpu.update_nd_offset %arg0, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16, #a>
      //CHECK-COUNT-4: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>>
      %b_next_tdesc = xegpu.update_nd_offset %arg1, [%c32, %c0] : !xegpu.tensor_desc<32x32xf16, #b>
      scf.yield %a_next_tdesc, %b_next_tdesc, %c
        : !xegpu.tensor_desc<16x32xf16, #a>, !xegpu.tensor_desc<32x32xf16, #b>, vector<16x32xf32>
    }
    //CHECK-COUNT-4: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>>
    xegpu.store_nd %out#2, %c_tdesc: vector<16x32xf32>, !xegpu.tensor_desc<16x32xf32, #c>
    gpu.return
  }
}

// -----
#l1 = #xegpu.layout<inst_data = [8, 16]>
#l2 = #xegpu.layout<inst_data = [16, 16]>
gpu.module @test_kernel {
  gpu.func @test_gemm_with_inst_data_only_attribute(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c16 : index
    %n = arith.muli %block_id_y, %c32 : index

    %c_tdesc = xegpu.create_nd_tdesc %C[%m, %n] : memref<1024x1024xf32> -> !xegpu.tensor_desc<16x32xf32, #l1>
    %c_init = xegpu.load_nd %c_tdesc : !xegpu.tensor_desc<16x32xf32, #l1> -> vector<16x32xf32>

    %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16, #l1>
    %b_tdesc = xegpu.create_nd_tdesc %B[%c0, %n] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #l2>
    %out:3 = scf.for %k = %c0 to %c1024 step %c32
      iter_args(%arg0 = %a_tdesc, %arg1 = %b_tdesc, %arg2 = %c_init)
      -> (!xegpu.tensor_desc<16x32xf16, #l1>, !xegpu.tensor_desc<32x32xf16, #l2>, vector<16x32xf32>) {
      //CHECK-COUNT-4: xegpu.load_nd {{.*}} -> vector<8x16xf16>
      %a = xegpu.load_nd %arg0 : !xegpu.tensor_desc<16x32xf16, #l1> -> vector<16x32xf16>
      //CHECK-COUNT-4: xegpu.load_nd {{.*}} -> vector<16x16xf16>
      %b = xegpu.load_nd %arg1 : !xegpu.tensor_desc<32x32xf16, #l2> -> vector<32x32xf16>
      //CHECK-COUNT-8: xegpu.dpas {{.*}} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %c = xegpu.dpas %a, %b, %arg2 {layout_result_0 = #l1}: vector<16x32xf16>, vector<32x32xf16>, vector<16x32xf32> -> vector<16x32xf32>
      //CHECK-COUNT-4: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<8x16xf16>
      %a_next_tdesc = xegpu.update_nd_offset %arg0, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16, #l1>
      //CHECK-COUNT-4: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<16x16xf16>
      %b_next_tdesc = xegpu.update_nd_offset %arg1, [%c32, %c0] : !xegpu.tensor_desc<32x32xf16, #l2>
      scf.yield %a_next_tdesc, %b_next_tdesc, %c
        : !xegpu.tensor_desc<16x32xf16, #l1>, !xegpu.tensor_desc<32x32xf16, #l2>, vector<16x32xf32>
    }
    //CHECK-COUNT-4: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %out#2, %c_tdesc: vector<16x32xf32>, !xegpu.tensor_desc<16x32xf32, #l1>
    gpu.return
  }
}

// -----
#l1 = #xegpu.layout<inst_data = [8, 16]>
#l2 = #xegpu.layout<inst_data = [16, 16]>
gpu.module @test_kernel {
  gpu.func @test_gemm_with_one_to_one_lowering(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c8 : index
    %n = arith.muli %block_id_y, %c32 : index

    %c_tdesc = xegpu.create_nd_tdesc %C[%m, %n] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x32xf32, #l1>

    //CHECK-COUNT-2: xegpu.load_nd {{.*}} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
    %c_init = xegpu.load_nd %c_tdesc : !xegpu.tensor_desc<8x32xf32, #l1> -> vector<8x32xf32>

    %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16, #l1>
    %b_tdesc = xegpu.create_nd_tdesc %B[%c0, %n] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16, #l2>
    %out:3 = scf.for %k = %c0 to %c1024 step %c16
      iter_args(%arg0 = %a_tdesc, %arg1 = %b_tdesc, %arg2 = %c_init)
      -> (!xegpu.tensor_desc<8x16xf16, #l1>, !xegpu.tensor_desc<16x32xf16, #l2>, vector<8x32xf32>) {
      //CHECK: xegpu.load_nd {{.*}} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %a = xegpu.load_nd %arg0 : !xegpu.tensor_desc<8x16xf16, #l1> -> vector<8x16xf16>
      //CHECK-COUNT-2: xegpu.load_nd {{.*}} : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
      %b = xegpu.load_nd %arg1 : !xegpu.tensor_desc<16x32xf16, #l2> -> vector<16x32xf16>
      %c = xegpu.dpas %a, %b, %arg2 {layout_result_0 = #l1}: vector<8x16xf16>, vector<16x32xf16>, vector<8x32xf32> -> vector<8x32xf32>
      //CHECK: xegpu.update_nd_offset {{.*}} [%c0, %c32] : !xegpu.tensor_desc<8x16xf16>
      %a_next_tdesc = xegpu.update_nd_offset %arg0, [%c0, %c32] : !xegpu.tensor_desc<8x16xf16, #l1>
      //CHECK-COUNT-2: xegpu.update_nd_offset {{.*}} [%c32, %c0] : !xegpu.tensor_desc<16x16xf16>
      %b_next_tdesc = xegpu.update_nd_offset %arg1, [%c32, %c0] : !xegpu.tensor_desc<16x32xf16, #l2>
      scf.yield %a_next_tdesc, %b_next_tdesc, %c
        : !xegpu.tensor_desc<8x16xf16, #l1>, !xegpu.tensor_desc<16x32xf16, #l2>, vector<8x32xf32>
    }
    //CHECK-COUNT-2: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %out#2, %c_tdesc: vector<8x32xf32>, !xegpu.tensor_desc<8x32xf32, #l1>
    gpu.return
  }
}

// -----
#a = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#b = #xegpu.layout<inst_data = [16, 16], lane_layout = [1, 16], lane_data = [16, 1]>
#c = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>
gpu.module @test_kernel {
  gpu.func @test_gemm_with_elemwise_preop(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c16 : index
    %n = arith.muli %block_id_y, %c32 : index

    %c_tdesc = xegpu.create_nd_tdesc %C[%m, %n] : memref<1024x1024xf32> -> !xegpu.tensor_desc<16x32xf32, #c>
    %c_init = xegpu.load_nd %c_tdesc : !xegpu.tensor_desc<16x32xf32, #c> -> vector<16x32xf32>

    %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16, #a>
    %b_tdesc = xegpu.create_nd_tdesc %B[%c0, %n] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #b>
    %out:3 = scf.for %k = %c0 to %c1024 step %c32
      iter_args(%arg0 = %a_tdesc, %arg1 = %b_tdesc, %arg2 = %c_init)
      -> (!xegpu.tensor_desc<16x32xf16, #a>, !xegpu.tensor_desc<32x32xf16, #b>, vector<16x32xf32>) {
      //CHECK-COUNT-4: xegpu.load_nd {{.*}} -> vector<8x16xf16>
      %a = xegpu.load_nd %arg0 : !xegpu.tensor_desc<16x32xf16, #a> -> vector<16x32xf16>
      //CHECK-COUNT-4: xegpu.load_nd {{.*}} -> vector<16x16xf16>
      %b = xegpu.load_nd %arg1 : !xegpu.tensor_desc<32x32xf16, #b> -> vector<32x32xf16>
      //CHECK-COUNT-4: math.exp {{.*}} : vector<8x16xf16>
      %e = math.exp %a {layout_result_0 = #a} : vector<16x32xf16>
      //CHECK-COUNT-8: xegpu.dpas {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %c = xegpu.dpas %e, %b, %arg2 {layout_result_0 = #c}: vector<16x32xf16>, vector<32x32xf16>, vector<16x32xf32> -> vector<16x32xf32>
      //CHECK-COUNT-4: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>>
      %a_next_tdesc = xegpu.update_nd_offset %arg0, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16, #a>
      //CHECK-COUNT-4: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>>
      %b_next_tdesc = xegpu.update_nd_offset %arg1, [%c32, %c0] : !xegpu.tensor_desc<32x32xf16, #b>
      scf.yield %a_next_tdesc, %b_next_tdesc, %c
        : !xegpu.tensor_desc<16x32xf16, #a>, !xegpu.tensor_desc<32x32xf16, #b>, vector<16x32xf32>
    }
    //CHECK-COUNT-4: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>>
    xegpu.store_nd %out#2, %c_tdesc: vector<16x32xf32>, !xegpu.tensor_desc<16x32xf32, #c>
    gpu.return
  }
}

// -----
#l = #xegpu.layout<inst_data = [8, 16]>
gpu.module @test_kernel {
  gpu.func @test_elementwise_with_inst_data_only(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c32 : index

    %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16, #l>
    %b_tdesc = xegpu.create_nd_tdesc %B[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16, #l>
    %c_tdesc = xegpu.create_nd_tdesc %C[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x32xf16, #l>

    %out:3 = scf.for %k = %c0 to %c1024 step %c32
      iter_args(%arg0 = %a_tdesc, %arg1 = %b_tdesc, %arg2 = %c_tdesc)
      -> (!xegpu.tensor_desc<16x32xf16, #l>, !xegpu.tensor_desc<16x32xf16, #l>, !xegpu.tensor_desc<16x32xf16, #l>) {
      //CHECK-COUNT-8: xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %a = xegpu.load_nd %arg0 : !xegpu.tensor_desc<16x32xf16, #l> -> vector<16x32xf16>
      %b = xegpu.load_nd %arg1 : !xegpu.tensor_desc<16x32xf16, #l> -> vector<16x32xf16>

      //CHECK-COUNT-4: arith.addf {{.*}} : vector<8x16xf16>
      %c = arith.addf %a, %b {layout_result_0 = #l} : vector<16x32xf16>

      //CHECK-COUNT-4: xegpu.store_nd {{.*}} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c, %arg2: vector<16x32xf16>, !xegpu.tensor_desc<16x32xf16, #l>

      //CHECK-COUNT-12: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<8x16xf16>
      %a_next_tdesc = xegpu.update_nd_offset %arg0, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16, #l>
      %b_next_tdesc = xegpu.update_nd_offset %arg1, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16, #l>
      %c_next_tdesc = xegpu.update_nd_offset %arg2, [%c0, %c32] : !xegpu.tensor_desc<16x32xf16, #l>
      scf.yield %a_next_tdesc, %b_next_tdesc, %c_next_tdesc
        : !xegpu.tensor_desc<16x32xf16, #l>, !xegpu.tensor_desc<16x32xf16, #l>, !xegpu.tensor_desc<16x32xf16, #l>
    }
    gpu.return
  }
}

// -----
#l = #xegpu.layout<inst_data = [8]>
gpu.module @test_kernel {
  gpu.func @test_elementwise_1D(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c32 : index

    %a_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32xf16, #l>
    %b_tdesc = xegpu.create_nd_tdesc %B[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32xf16, #l>
    %c_tdesc = xegpu.create_nd_tdesc %C[%m, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32xf16, #l>

    %out:3 = scf.for %k = %c0 to %c1024 step %c32
      iter_args(%arg0 = %a_tdesc, %arg1 = %b_tdesc, %arg2 = %c_tdesc)
      -> (!xegpu.tensor_desc<32xf16, #l>, !xegpu.tensor_desc<32xf16, #l>, !xegpu.tensor_desc<32xf16, #l>) {
      //CHECK-COUNT-8: xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<8xf16> -> vector<8xf16>
      %a = xegpu.load_nd %arg0 : !xegpu.tensor_desc<32xf16, #l> -> vector<32xf16>
      %b = xegpu.load_nd %arg1 : !xegpu.tensor_desc<32xf16, #l> -> vector<32xf16>

      //CHECK-COUNT-4: arith.addf {{.*}} : vector<8xf16>
      %c = arith.addf %a, %b {layout_result_0 = #l} : vector<32xf16>

      //CHECK-COUNT-4: xegpu.store_nd {{.*}} : vector<8xf16>, !xegpu.tensor_desc<8xf16>
      xegpu.store_nd %c, %arg2: vector<32xf16>, !xegpu.tensor_desc<32xf16, #l>

      //CHECK-COUNT-12: xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<8xf16>
      %a_next_tdesc = xegpu.update_nd_offset %arg0, [%c32] : !xegpu.tensor_desc<32xf16, #l>
      %b_next_tdesc = xegpu.update_nd_offset %arg1, [%c32] : !xegpu.tensor_desc<32xf16, #l>
      %c_next_tdesc = xegpu.update_nd_offset %arg2, [%c32] : !xegpu.tensor_desc<32xf16, #l>
      scf.yield %a_next_tdesc, %b_next_tdesc, %c_next_tdesc
        : !xegpu.tensor_desc<32xf16, #l>, !xegpu.tensor_desc<32xf16, #l>, !xegpu.tensor_desc<32xf16, #l>
    }
    gpu.return
  }
}
