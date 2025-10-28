// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

#map = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
module @gemm attributes {gpu.container_module} {
  func.func @test_fast_math(%input1: memref<256x256xf32>, %input2: memref<256x256xf32>) -> (memref<256x256xf32>, memref<256x256xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %input1_gpu = gpu.alloc () : memref<256x256xf32>
    gpu.memcpy %input1_gpu, %input2 : memref<256x256xf32>, memref<256x256xf32>
    %input2_gpu = gpu.alloc () : memref<256x256xf32>
    gpu.memcpy %input2_gpu, %input2 : memref<256x256xf32>, memref<256x256xf32>
    %result_gpu = gpu.alloc () : memref<256x256xf32>
    %result_gpu_with_fastmath = gpu.alloc () : memref<256x256xf32>
    // NOTE: Here we can't use [8, 64] wi threads following
    // the SG thread layout of [8, 4]. Because runtime will linearize
    // the x dimension first (we need y dimension to be linearized first).
    // So just use linearized thread layout of [512, 1] wi threads.
    gpu.launch_func  @math_kernels::@gpu_maximumf blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1) args(%input1_gpu : memref<256x256xf32>, %input2_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>)
    gpu.launch_func  @math_kernels::@gpu_maximumf_with_fastmath blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1) args(%input1_gpu : memref<256x256xf32>, %input2_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>)

    %result_host = memref.alloc() : memref<256x256xf32>
    %result_host_with_fastmath = memref.alloc() : memref<256x256xf32>
    gpu.memcpy %result_host, %result_gpu : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %input_gpu : memref<256x256xf32>
    gpu.dealloc %result_gpu : memref<256x256xf32>
    return %result_host, %result_host_with_fastmath : memref<256x256xf32>, memref<256x256xf32>
  }

  gpu.module @math_kernels   {
    gpu.func @gpu_maximumf(%input1_gpu : memref<256x256xf32>, %input2_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>) kernel  {
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %input_tdesc_1 = xegpu.create_nd_tdesc %input1_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val_1 = xegpu.load_nd %input_tdesc_1[%m, %n] : !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %input_tdesc_2 = xegpu.create_nd_tdesc %input2_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val_2 = xegpu.load_nd %input_tdesc_2[%m, %n] : !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %result_val = arith.maximumf %input_val_1, %input_val_2 : vector<256x256xf32>
      %result_tdesc = xegpu.create_nd_tdesc %result_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      xegpu.store_nd %result_val, %result_tdesc[%m, %n] : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #map>
      gpu.return
    }

    // Kernel with fastmath attribute
    gpu.func @gpu_maximumf_with_fastmath(%input1_gpu : memref<256x256xf32>, %input2_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>) kernel  {
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %input_tdesc_1 = xegpu.create_nd_tdesc %input1_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val_1 = xegpu.load_nd %input_tdesc_1[%m, %n] : !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %input_tdesc_2 = xegpu.create_nd_tdesc %input2_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val_2 = xegpu.load_nd %input_tdesc_2[%m, %n] : !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %result_val = arith.maximumf %input_val_1, %input_val_2 fastmath<fast> : vector<256x256xf32>
      %result_tdesc = xegpu.create_nd_tdesc %result_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      xegpu.store_nd %result_val, %result_tdesc[%m, %n] : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #map>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_f32 = arith.constant 2.2 : f32
    %c256 = arith.constant 256 : index
    %input_1 = memref.alloc() : memref<256x256xf32>
    %input_2 = memref.alloc() : memref<256x256xf32>
    %cpu_ref_result = memref.alloc() : memref<256x256xf32>

    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        memref.store %c2_f32, %input_1[%arg0, %arg1] : memref<256x256xf32>
        memref.store %c2_f32, %input_2[%arg0, %arg1] : memref<256x256xf32>
      }
    }

    // Run CPU version
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %val_1 = memref.load %input_1[%arg0, %arg1] : memref<256x256xf32>
        %val_2 = memref.load %input_2[%arg0, %arg1] : memref<256x256xf32>
        %res_val = arith.maximumf %val_1, %val_2 : f32
        memref.store %res_val, %cpu_ref_result[%arg0, %arg1] : memref<256x256xf32>
      }
    }

    // Run GPU version.
    %gpu_result, %gpu_result_fastmath = call @test_fast_math(%input_1, %input_2) : (memref<256x256xf32>, memref<256x256xf32>) -> (memref<256x256xf32>, memref<256x256xf32>)
    %gpu_result_cast = memref.cast %gpu_result : memref<256x256xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-256: [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,   98,   99,   100,   101,   102,   103,   104,   105,   106,   107,   108,   109,   110,   111,   112,   113,   114,   115,   116,   117,   118,   119,   120,   121,   122,   123,   124,   125,   126,   127,   128,   129,   130,   131,   132,   133,   134,   135,   136,   137,   138,   139,   140,   141,   142,   143,   144,   145,   146,   147,   148,   149,   150,   151,   152,   153,   154,   155,   156,   157,   158,   159,   160,   161,   162,   163,   164,   165,   166,   167,   168,   169,   170,   171,   172,   173,   174,   175,   176,   177,   178,   179,   180,   181,   182,   183,   184,   185,   186,   187,   188,   189,   190,   191,   192,   193,   194,   195,   196,   197,   198,   199,   200,   201,   202,   203,   204,   205,   206,   207,   208,   209,   210,   211,   212,   213,   214,   215,   216,   217,   218,   219,   220,   221,   222,   223,   224,   225,   226,   227,   228,   229,   230,   231,   232,   233,   234,   235,   236,   237,   238,   239,   240,   241,   242,   243,   244,   245,   246,   247,   248,   249,   250,   251,   252,   253,   254,   255]
    call @printMemrefF32(%gpu_result_cast) : (memref<*xf32>) -> ()

    memref.dealloc %input_1 : memref<256x256xf32>
    memref.dealloc %input_2 : memref<256x256xf32>
    memref.dealloc %cpu_ref_result : memref<256x256xf32>
    memref.dealloc %gpu_result : memref<256x256xf32>
    memref.dealloc %gpu_result_fastmath : memref<256x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
