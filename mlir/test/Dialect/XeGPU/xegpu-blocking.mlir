// RUN: mlir-opt --xegpu-blocking -split-input-file %s | FileCheck %s

#a = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#b = #xegpu.layout<inst_data = [16, 16], lane_layout = [1, 16], lane_data = [16, 1]>
#c = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>
gpu.module @test_kernel {
  gpu.func @gemm_with_one_to_n_lowering(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
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
  gpu.func @gemm_with_inst_data_only_attribute(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
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
  gpu.func @gemm_with_one_to_one_lowering(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
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
  gpu.func @gemm_with_elemwise_preop(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
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
  gpu.func @elementwise_with_inst_data_only(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf16>) {
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
  gpu.func @elementwise_1D(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf16>) {
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

// -----
#l = #xegpu.layout<inst_data = [16, 16]>
#r = #xegpu.layout<inst_data = [16]>
gpu.module @test_kernel  {
  gpu.func @reduce_dim_0(%a: memref<16x512xf32>, %b: memref<512xf32>)  kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %acc = arith.constant dense<0.0> : vector<64xf32>
    %c64 = arith.constant 64 : index
    %block_id_x = gpu.block_id x
    %m = arith.muli %block_id_x, %c64 : index
    %0 = xegpu.create_nd_tdesc %a[0, %m] : memref<16x512xf32> -> !xegpu.tensor_desc<16x64xf32, #l>
    %1 = xegpu.load_nd %0: !xegpu.tensor_desc<16x64xf32, #l> -> vector<16x64xf32>
    // CHECK: vector.multi_reduction <add>, {{.*}}, [[ACC:%[0-9A-Za-z]+]] [0] : vector<16x16xf32> to vector<16xf32>
    // CHECK-COUNT-3: vector.multi_reduction <add>, {{.*}}, [[ACC]] [0] : vector<16x16xf32> to vector<16xf32>
    %2 = vector.multi_reduction <add>, %1, %acc {layout_result_0 = #r} [0]: vector<16x64xf32> to vector<64xf32>
    %3 = xegpu.create_nd_tdesc %b[%m] : memref<512xf32> -> !xegpu.tensor_desc<64xf32, #r>
    xegpu.store_nd %2, %3: vector<64xf32>, !xegpu.tensor_desc<64xf32, #r>
    gpu.return
  }
}

// -----
#l = #xegpu.layout<inst_data = [16, 16]>
#r = #xegpu.layout<inst_data = [16]>
gpu.module @test_kernel   {
  gpu.func @reduce_dim_1(%a: memref<512x32xf32>, %b: memref<512xf32>)  kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %acc = arith.constant dense<0.0> : vector<32xf32>

    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y

    %m = arith.muli %block_id_x, %c32 : index
    %n = arith.muli %block_id_y, %c32 : index
    %0 = xegpu.create_nd_tdesc %a[%m, %n] : memref<512x32xf32> -> !xegpu.tensor_desc<32x128xf32, #l>
    %1 = xegpu.load_nd %0: !xegpu.tensor_desc<32x128xf32, #l> -> vector<32x128xf32>

    // CHECK: vector.multi_reduction <add>, {{.*}}, [[INIT:%[0-9A-Za-z]+]] [1] : vector<16x16xf32> to vector<16xf32>
    // CHECK-COUNT-1: vector.multi_reduction <add>, {{.*}}, [[INIT]] [1] : vector<16x16xf32> to vector<16xf32>

    %2 = vector.multi_reduction <add>, %1, %acc {layout_result_0 = #r} [1]: vector<32x128xf32> to vector<32xf32>
    %3 = xegpu.create_nd_tdesc %b[%n] : memref<512xf32> -> !xegpu.tensor_desc<32xf32, #r>
    xegpu.store_nd %2, %3: vector<32xf32>, !xegpu.tensor_desc<32xf32, #r>
    gpu.return
  }
}

// -----
#r = #xegpu.layout<inst_data = [16]>
#l = #xegpu.layout<inst_data = [16, 16]>
gpu.module @test_kernel   {
  gpu.func @broadcast_dim_0(%a: memref<512xf32>, %b: memref<16x512xf32>)  kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

    %c64 = arith.constant 64 : index
    %block_id_x = gpu.block_id x
    %m = arith.muli %block_id_x, %c64 : index
    %0 = xegpu.create_nd_tdesc %a[%m] : memref<512xf32> -> !xegpu.tensor_desc<64xf32, #r>
    %1 = xegpu.load_nd %0: !xegpu.tensor_desc<64xf32, #r> -> vector<64xf32>
    // CHECK-COUNT-4: vector.broadcast {{.*}} : vector<16xf32> to vector<16x16xf32>
    %2 = vector.broadcast  %1 {layout_result_0 = #l} : vector<64xf32> to vector<16x64xf32>
    %3 = xegpu.create_nd_tdesc %b[0, %m] : memref<16x512xf32> -> !xegpu.tensor_desc<16x64xf32, #l>
    xegpu.store_nd %2, %3: vector<16x64xf32>, !xegpu.tensor_desc<16x64xf32, #l>
    gpu.return
  }
}

// -----
#r = #xegpu.layout<inst_data = [16]>
#l = #xegpu.layout<inst_data = [16, 16]>
gpu.module @test_kernel  {
  gpu.func @broadcast_dim_1(%a: memref<512xf32>, %b: memref<16x512xf32>)  kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

    %c32 = arith.constant 32 : index
    %block_id_x = gpu.block_id x
    %m = arith.muli %block_id_x, %c32 : index
    %0 = xegpu.create_nd_tdesc %a[%m] : memref<512xf32> -> !xegpu.tensor_desc<32xf32, #r>
    %1 = xegpu.load_nd %0: !xegpu.tensor_desc<32xf32, #r> -> vector<32xf32>
    %11 = vector.shape_cast %1 :  vector<32xf32> to vector<32x1xf32>
    // CHECK-COUNT-8: vector.broadcast {{.*}}: vector<16x1xf32> to vector<16x16xf32>
    %2 = vector.broadcast  %11 {layout_result_0 = #l} : vector<32x1xf32> to vector<32x64xf32>
    %3 = xegpu.create_nd_tdesc %b[0, %m] : memref<16x512xf32> -> !xegpu.tensor_desc<32x64xf32, #l>
    xegpu.store_nd %2, %3: vector<32x64xf32>, !xegpu.tensor_desc<32x64xf32, #l>
    gpu.return
  }
}

// -----
#l = #xegpu.layout<inst_data = [16, 8]>
#t = #xegpu.layout<inst_data = [8, 16]>
gpu.module @test_kernel   {
  gpu.func @transpose(%a: memref<512x8xf32>, %b: memref<8x512xf32>)  kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

    %c32 = arith.constant 32 : index
    %block_id_x = gpu.block_id x
    %m = arith.muli %block_id_x, %c32 : index
    %0 = xegpu.create_nd_tdesc %a[%m, 0] : memref<512x8xf32> -> !xegpu.tensor_desc<32x8xf32, #l>
    %1 = xegpu.load_nd %0: !xegpu.tensor_desc<32x8xf32, #l> -> vector<32x8xf32>
    // CHECK-COUNT-2: vector.transpose {{.*}}  [1, 0] : vector<16x8xf32> to vector<8x16xf32>
    %2 = vector.transpose  %1, [1, 0] {layout_result_0 = #t} : vector<32x8xf32> to vector<8x32xf32>
    %3 = xegpu.create_nd_tdesc %b[0, %m] : memref<8x512xf32> -> !xegpu.tensor_desc<8x32xf32, #t>
    xegpu.store_nd %2, %3: vector<8x32xf32>, !xegpu.tensor_desc<8x32xf32, #t>
    gpu.return
  }
}

// -----
gpu.module @test_kernel {
  // CHECK-LABEL: test_prefetch_load_store_update
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-2: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.prefetch {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  // CHECK-COUNT-2: xegpu.update_offset {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
  // CHECK-COUNT-2: xegpu.load  {{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  // CHECK-COUNT-2: xegpu.store  {{.*}} : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>

  gpu.func @test_prefetch_load_store_update(%src: ui64)  {

    %cst = arith.constant dense<[
    0,   8,  16,  24,  32,  40,  48,  56,
    64,  72,  80,  88,  96, 104, 112, 120,
    128, 136, 144, 152, 160, 168, 176, 184,
    192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>
    xegpu.prefetch %tdesc: !xegpu.tensor_desc<32xf32,  #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>

    %delta = arith.constant dense<[
    32,   32,  32,  32,  32,  32,  32,  32,
    32,   32,  32,  32,  32,  32,  32,  64,
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 256
    ]> : vector<32xindex>
    %new_tdesc = xegpu.update_offset %tdesc, %delta
              : !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>, vector<32xindex>

    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>

    %ld_vec = xegpu.load %new_tdesc, %mask: !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>, vector<32xi1> -> vector<32xf32>

    %st_vec = arith.addf %ld_vec, %ld_vec : vector<32xf32>
    xegpu.store %st_vec, %tdesc, %mask:
                 vector<32xf32>,
                 !xegpu.tensor_desc<32xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<inst_data = [16]>>,
                 vector<32xi1>

    gpu.return
  }

}

// -----

gpu.module @test_kernel   {
  // CHECK-LABEL: test_prefetch_load_store_update_chunk
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK-COUNT-4: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK-COUNT-4: xegpu.prefetch {{.*}} : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK-COUNT-4: xegpu.update_offset {{.*}} : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xindex>
  // CHECK-COUNT-4: xegpu.load  {{.*}} : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xi1> -> vector<16x2xf32>
  // CHECK-COUNT-4: xegpu.store  {{.*}} : vector<16x2xf32>, !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xi1>

  gpu.func @test_prefetch_load_store_update_chunk(%src: ui64)  {

    %cst = arith.constant dense<[
      0,   8,  16,  24,  32,  40,  48,  56,
      64,  72,  80,  88,  96, 104, 112, 120,
      128, 136, 144, 152, 160, 168, 176, 184,
      192, 200, 208, 216, 224, 232, 240, 248
    ]> : vector<32xindex>

    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<32xindex> -> !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>
    xegpu.prefetch %tdesc: !xegpu.tensor_desc<32x4xf32,  #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>

    %delta = arith.constant dense<[
      32,   32,  32,  32,  32,  32,  32,  32,
      32,   32,  32,  32,  32,  32,  32,  64,
      128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 256
    ]> : vector<32xindex>
    %new_tdesc = xegpu.update_offset %tdesc, %delta
              : !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>, vector<32xindex>

    %c17 = arith.constant 17: index
    %mask = vector.create_mask %c17: vector<32xi1>

    %ld_vec = xegpu.load %new_tdesc, %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>, vector<32xi1> -> vector<32x4xf32>

    %st_vec = arith.addf %ld_vec, %ld_vec : vector<32x4xf32>
    xegpu.store %st_vec, %tdesc, %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>:
                 vector<32x4xf32>,
                 !xegpu.tensor_desc<32x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #xegpu.layout<inst_data = [16, 2]>>,
                 vector<32xi1>

    gpu.return
  }
}

// -----
#l = #xegpu.layout<inst_data = [2, 8, 2]>

// test the blocking pass on a 3D scattered tensor descriptor,
// Ops working 4x8x4xf32 scattered tensor_descs will be unrolled
// into 4 ops working 2x8x2xf32 scattered tensor_descs based on
// the given layout.
gpu.module @test_kernel   {
  // CHECK-LABEL: test_3d_scattered_tensor_desc
  // CHECK-SAME: [[arg0:%.+]]: ui64
  // CHECK: [[cst_1:%.+]] = arith.constant dense<{{.*}}[130, 138, 146, 154, 162, 170, 178, 186], [194, 202, 210, 218, 226, 234, 242, 250]]> : vector<2x8xindex>
  // CHECK: [[cst_2:%.+]] = arith.constant dense<{{.*}}[2, 10, 18, 26, 34, 42, 50, 58], [66, 74, 82, 90, 98, 106, 114, 122]]> : vector<2x8xindex>
  // CHECK: [[cst_3:%.+]] = arith.constant dense<{{.*}}[0, 8, 16, 24, 32, 40, 48, 56], [64, 72, 80, 88, 96, 104, 112, 120]]> : vector<2x8xindex>
  // CHECK: [[cst_4:%.+]] = arith.constant dense<{{.*}}[128, 136, 144, 152, 160, 168, 176, 184], [192, 200, 208, 216, 224, 232, 240, 248]]> : vector<2x8xindex>
  // CHECK-COUNT-4: xegpu.create_tdesc [[arg0]], {{.*}} : ui64, vector<2x8xindex> -> !xegpu.tensor_desc<2x8x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK-COUNT-4: xegpu.prefetch {{.*}} : !xegpu.tensor_desc<2x8x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  // CHECK-COUNT-4: xegpu.update_offset {{.*}} : !xegpu.tensor_desc<2x8x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<2x8xindex>
  // CHECK-COUNT-4: xegpu.load  {{.*}} : !xegpu.tensor_desc<2x8x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<2x8xi1> -> vector<2x8x2xf32>
  // CHECK-COUNT-4: xegpu.store  {{.*}} : vector<2x8x2xf32>, !xegpu.tensor_desc<2x8x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<2x8xi1>


  gpu.func @test_3d_scattered_tensor_desc(%src: ui64)  {
    %cst = arith.constant dense<[
      [0,   8,  16,  24,  32,  40,  48,  56],
      [64,  72,  80,  88,  96, 104, 112, 120],
      [128, 136, 144, 152, 160, 168, 176, 184],
      [192, 200, 208, 216, 224, 232, 240, 248]
    ]> : vector<4x8xindex>

    %tdesc = xegpu.create_tdesc %src, %cst : ui64, vector<4x8xindex> -> !xegpu.tensor_desc<4x8x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #l>
    xegpu.prefetch %tdesc: !xegpu.tensor_desc<4x8x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #l>

    %delta = arith.constant dense<[
      [32,   32,  32,  32,  32,  32,  32,  32],
      [32,   32,  32,  32,  32,  32,  32,  64],
      [128, 128, 128, 128, 128, 128, 128, 128],
      [128, 128, 128, 128, 128, 128, 128, 256]
    ]> : vector<4x8xindex>
    %new_tdesc = xegpu.update_offset %tdesc, %delta
              : !xegpu.tensor_desc<4x8x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #l>, vector<4x8xindex>

    %c4 = arith.constant 4: index
    %mask = vector.create_mask %c4, %c4: vector<4x8xi1>

    %ld_vec = xegpu.load %new_tdesc, %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<4x8x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #l>, vector<4x8xi1> -> vector<4x8x4xf32>

    %st_vec = arith.addf %ld_vec, %ld_vec {layout_result_0 = #l} : vector<4x8x4xf32>
    xegpu.store %st_vec, %tdesc, %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>:
                 vector<4x8x4xf32>,
                 !xegpu.tensor_desc<4x8x4xf32, #xegpu.scatter_tdesc_attr<chunk_size=4>, #l>,
                 vector<4x8xi1>
    gpu.return
  }
}

// -----
#a = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#b = #xegpu.layout<inst_data = [16, 16], lane_layout = [1, 16], lane_data = [16, 1]>
#c = #xegpu.layout<inst_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>

gpu.module @test_kernel {
  //CHECK-LABEL: gpu.func @convert_layout
  //CHECK-SAME: [[arg0:%.+]]: memref<16x16xf16>, [[arg1:%.+]]: memref<16x16xf16>, [[arg2:%.+]]: memref<16x16xf32>
  //CHECK: [[c8:%.+]] = arith.constant 8 : index
  //CHECK: [[c0:%.+]] = arith.constant 0 : index
  //CHECK: [[a:%.+]] = xegpu.create_nd_tdesc [[arg0]][[[c0]], [[c0]]] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>>
  //CHECK: [[b:%.+]] = xegpu.create_nd_tdesc [[arg1]][[[c0]], [[c0]]] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>>
  //CHECK: [[load_a:%.+]] = xegpu.load_nd [[a]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>> -> vector<16x16xf16>
  //CHECK: [[load_b:%.+]] = xegpu.load_nd [[b]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>> -> vector<16x16xf16>
  //CHECK: [[cvt:%.+]] = xegpu.convert_layout [[load_a]] <{input_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [16, 1]>, target_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>}> : vector<16x16xf16>
  //CHECK: [[a0:%.+]] = vector.extract_strided_slice [[cvt]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
  //CHECK: [[a1:%.+]] = vector.extract_strided_slice [[cvt]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
  //CHECK: [[dpas0:%.+]] = xegpu.dpas [[a0]], [[load_b]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>} : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  //CHECK: [[dpas1:%.+]] = xegpu.dpas [[a1]], [[load_b]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>} : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  //CHECK: [[c_tdesc_0:%.+]] = xegpu.create_nd_tdesc [[arg2]][[[c0]], [[c0]]] : memref<16x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>>
  //CHECK: [[c_tdesc_1:%.+]] = xegpu.create_nd_tdesc [[arg2]][[[c8]], [[c0]]] : memref<16x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>>
  //CHECK: xegpu.store_nd [[dpas0]], [[c_tdesc_0]]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>>
  //CHECK: xegpu.store_nd [[dpas1]], [[c_tdesc_1]]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [8, 1]>>

  gpu.func @convert_layout(%A: memref<16x16xf16>, %B: memref<16x16xf16>, %C: memref<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %a_tdesc = xegpu.create_nd_tdesc %A[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #b>
    %b_tdesc = xegpu.create_nd_tdesc %B[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #b>
    %a = xegpu.load_nd %a_tdesc : !xegpu.tensor_desc<16x16xf16, #b> -> vector<16x16xf16>
    %b = xegpu.load_nd %b_tdesc : !xegpu.tensor_desc<16x16xf16, #b> -> vector<16x16xf16>
    %e = xegpu.convert_layout %a <{input_layout = #b, target_layout = #a}> : vector<16x16xf16>
    %c = xegpu.dpas %e, %b {layout_result_0 = #c}: vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf32>
    %c_tdesc = xegpu.create_nd_tdesc %C[%c0, %c0] : memref<16x16xf32> -> !xegpu.tensor_desc<16x16xf32, #c>
    xegpu.store_nd %c, %c_tdesc: vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #c>
    gpu.return
  }
}

// -----

#lb = #xegpu.layout<inst_data = [8, 32, 2], lane_layout = [1, 16, 1], lane_data = [8, 1, 2]>
#b = #xegpu.layout<inst_data = [8, 16, 2], lane_layout = [1, 16, 1], lane_data = [8, 1, 2]>

gpu.module @test_kernel {
  //CHECK: gpu.func @convert_layout([[arg0:%.+]]: vector<8x32x2xf16>) -> vector<8x32x2xf16> {
  //CHECK: [[cst:%.+]] = arith.constant dense<0.000000e+00> : vector<8x32x2xf16>
  //CHECK: [[e1:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
  //CHECK: [[m1:%.+]] = math.exp [[e1]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16, 1], lane_data = [8, 1, 2]>} : vector<8x16x2xf16>
  //CHECK: [[r1:%.+]] = vector.insert_strided_slice [[m1]], [[cst]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<8x16x2xf16> into vector<8x32x2xf16>
  //CHECK: [[e2:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0, 16, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
  //CHECK: [[m2:%.+]] = math.exp [[e2]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16, 1], lane_data = [8, 1, 2]>} : vector<8x16x2xf16>
  //CHECK: [[r2:%.+]] = vector.insert_strided_slice [[m2]], [[r1]] {offsets = [0, 16, 0], strides = [1, 1, 1]} : vector<8x16x2xf16> into vector<8x32x2xf16>
  //CHECK: gpu.return [[r2]] : vector<8x32x2xf16>

  gpu.func @convert_layout(%B: vector<8x32x2xf16>) -> vector<8x32x2xf16> {
    %b = xegpu.convert_layout %B <{input_layout = #lb, target_layout = #b}> : vector<8x32x2xf16>
    %e = math.exp %b {layout_result_0 = #b} : vector<8x32x2xf16>
    gpu.return %e : vector<8x32x2xf16>
  }
}

// -----
gpu.module @test_kernel {
  //CHECK-LABEL: unroll_load_matrix
  gpu.func @unroll_load_matrix(%arg0: memref<4096xi8, 3>) -> vector<32x32xf32> {
    %0 = xegpu.create_mem_desc %arg0 : memref<4096xi8, 3> -> !xegpu.mem_desc<32x32xf32>
    //CHECK-COUNT-8: xegpu.load_matrix {{.*}} : !xegpu.mem_desc<32x32xf32>, index, index -> vector<8x16xf32>
    //CHECK-COUNT-8: vector.insert_strided_slice {{.*}} : vector<8x16xf32> into vector<32x32xf32>
    %1 = xegpu.load_matrix %0[0, 0] <{layout = #xegpu.layout<inst_data = [8, 16]>}>: !xegpu.mem_desc<32x32xf32> -> vector<32x32xf32>
    gpu.return %1: vector<32x32xf32>
  }
}

// -----
gpu.module @test_kernel {
  // CHECK-LABEL: unroll_store_matrix
  gpu.func @unroll_store_matrix(%value: vector<32x32xf32>, %arg0 : memref<32768xi8, 3>) {
    %mdesc = xegpu.create_mem_desc %arg0 : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    // CHECK-COUNT-8:  xegpu.store_matrix {{.*}} : vector<8x16xf32>, !xegpu.mem_desc<64x128xf32>, index, index
    xegpu.store_matrix %value, %mdesc[0, 0] {layout = #xegpu.layout<inst_data = [8, 16]>} : vector<32x32xf32>, !xegpu.mem_desc<64x128xf32>
    gpu.return
  }
}
