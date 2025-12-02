// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

#a = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
#b = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>
#c = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>
#a_prefetch = #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 32], inst_data = [8, 16]>
#b_prefetch = #xegpu.layout<sg_layout = [4, 8], sg_data = [8, 32], inst_data = [8, 16]>
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<256x256xf16>, %B: memref<256x256xf16>, %C: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc () : memref<256x256xf16>
    gpu.memcpy %A_gpu, %A : memref<256x256xf16>, memref<256x256xf16>
    %B_gpu = gpu.alloc () : memref<256x256xf16>
    gpu.memcpy %B_gpu, %B : memref<256x256xf16>, memref<256x256xf16>
    %C_gpu = gpu.alloc () : memref<256x256xf32>
    gpu.memcpy %C_gpu, %C : memref<256x256xf32>, memref<256x256xf32>
    // NOTE: Here we can't use [8, 64] wi threads following
    // the SG thread layout of [8, 4]. Because runtime will linearize
    // the x dimension first (we need y dimension to be linearized first).
    // So just use linearized thread layout of [512, 1] wi threads.
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1) args(%A_gpu : memref<256x256xf16>, %B_gpu : memref<256x256xf16>, %C_gpu : memref<256x256xf32>)
    gpu.wait // Wait for the kernel to finish.
    gpu.memcpy %C, %C_gpu : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %A_gpu : memref<256x256xf16>
    gpu.dealloc %B_gpu : memref<256x256xf16>
    gpu.dealloc %C_gpu : memref<256x256xf32>
    return %C : memref<256x256xf32>
  }

  gpu.module @test_kernel   {
    gpu.func @test_kernel(%A: memref<256x256xf16>, %B: memref<256x256xf16>, %C: memref<256x256xf32>) kernel  {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c96 = arith.constant 96 : index
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %c_tdesc = xegpu.create_nd_tdesc %C : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #c>
      %c_init_value = xegpu.load_nd %c_tdesc[%m, %n] : !xegpu.tensor_desc<256x256xf32, #c> -> vector<256x256xf32>
      %a_tdesc = xegpu.create_nd_tdesc %A : memref<256x256xf16> -> !xegpu.tensor_desc<256x32xf16, #a>
      %b_tdesc = xegpu.create_nd_tdesc %B : memref<256x256xf16> -> !xegpu.tensor_desc<32x256xf16, #b>
      // Prefetch A 3 times.
      %a_prefetch_tdesc = xegpu.create_nd_tdesc %A : memref<256x256xf16> -> !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c0] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c32] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
      xegpu.prefetch_nd %a_prefetch_tdesc[%m, %c64] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
       // Prefetch B 3 times.
      %b_prefetch_tdesc = xegpu.create_nd_tdesc %B : memref<256x256xf16> -> !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c0, %n] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c32, %n] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
      xegpu.prefetch_nd %b_prefetch_tdesc[%c64, %n] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>

      %out = scf.for %k = %c0 to %c256 step %c32
        iter_args(%c_value = %c_init_value)
        -> (vector<256x256xf32>) {
        %a_value = xegpu.load_nd %a_tdesc[%m, %k]  : !xegpu.tensor_desc<256x32xf16, #a> -> vector<256x32xf16>
        %b_value = xegpu.load_nd %b_tdesc[%k, %n] : !xegpu.tensor_desc<32x256xf16, #b> -> vector<32x256xf16>
        // Prefetch next tiles.
        %prefetch_offset = arith.addi %k, %c96 : index
        xegpu.prefetch_nd %a_prefetch_tdesc[%m, %prefetch_offset] : !xegpu.tensor_desc<256x32xf16, #a_prefetch>
        xegpu.prefetch_nd %b_prefetch_tdesc[%prefetch_offset, %n] : !xegpu.tensor_desc<32x256xf16, #b_prefetch>
        %c_new_value = xegpu.dpas %a_value, %b_value, %c_value {layout_result_0 = #c}
          : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %c_new_value : vector<256x256xf32>
      }
      xegpu.store_nd %out, %c_tdesc[%m, %n] : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #c>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_f16 = arith.constant 1.0 : f16
    %c2_f16 = arith.constant 2.0 : f16
    %c256 = arith.constant 256 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<256x256xf16>
    %B = memref.alloc() : memref<256x256xf16>
    %C = memref.alloc() : memref<256x256xf32>
    %C_ref = memref.alloc() : memref<256x256xf32>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    // Intialize matrix A ; A[i, j] = j
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %t = index.castu %j : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %A[%i, %j] : memref<256x256xf16>
      }
    }

    // Initialize the B matrix
    // Make matrix B an identity matrix
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %i_i32 = index.castu %i : index to i32
        %j_i32 = index.castu %j : index to i32
        %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

        scf.if %i_j_same {
          memref.store %cf_1, %B[%i, %j] : memref<256x256xf16>
        } else {
          memref.store %cf_0, %B[%i, %j] : memref<256x256xf16>
        }
      }
    }

    // Initialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<256x256xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<256x256xf32>
      }
    }

    // Run GPU version.
    %2 = call @test(%A, %B, %C) : (memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf32>) -> memref<256x256xf32>
    %gpu_result_cast = memref.cast %2 : memref<256x256xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-256: [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,   98,   99,   100,   101,   102,   103,   104,   105,   106,   107,   108,   109,   110,   111,   112,   113,   114,   115,   116,   117,   118,   119,   120,   121,   122,   123,   124,   125,   126,   127,   128,   129,   130,   131,   132,   133,   134,   135,   136,   137,   138,   139,   140,   141,   142,   143,   144,   145,   146,   147,   148,   149,   150,   151,   152,   153,   154,   155,   156,   157,   158,   159,   160,   161,   162,   163,   164,   165,   166,   167,   168,   169,   170,   171,   172,   173,   174,   175,   176,   177,   178,   179,   180,   181,   182,   183,   184,   185,   186,   187,   188,   189,   190,   191,   192,   193,   194,   195,   196,   197,   198,   199,   200,   201,   202,   203,   204,   205,   206,   207,   208,   209,   210,   211,   212,   213,   214,   215,   216,   217,   218,   219,   220,   221,   222,   223,   224,   225,   226,   227,   228,   229,   230,   231,   232,   233,   234,   235,   236,   237,   238,   239,   240,   241,   242,   243,   244,   245,   246,   247,   248,   249,   250,   251,   252,   253,   254,   255]
    call @printMemrefF32(%gpu_result_cast) : (memref<*xf32>) -> ()

    memref.dealloc %A : memref<256x256xf16>
    memref.dealloc %B : memref<256x256xf16>
    memref.dealloc %C : memref<256x256xf32>
    memref.dealloc %C_ref : memref<256x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
