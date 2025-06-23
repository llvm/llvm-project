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

gpu.module @kernel  attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
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

gpu.module @kernel  attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
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

gpu.module @kernel  attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
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

gpu.module @kernel  attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
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

gpu.module @kernel  attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
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