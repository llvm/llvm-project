// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// ---- BF16 ------

#sg_map_fp16_a = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
#sg_map_fp16_b = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#sg_map_fp16_c = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
// CHECK-LABEL: func @test_gemm_bf16({{.*}}) {
func.func @test_gemm_bf16(%a : memref<1024x1024xbf16>, %b: memref<1024x1024xbf16>, %c: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index

  %c0_1 = arith.constant 0 : i32
  %c1_1 = arith.constant 1 : i32


  scf.for %i= %c0 to %c1024 step %c8 {
    scf.for %j= %c0 to %c1024 step %c16 {
      // CHECK: xegpu.create_nd_tdesc
      // CHECK-SAME: memref<1024x1024xbf16>
      // CHECK-SAME: -> !xegpu.tensor_desc<8x16xbf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
      %1 = xegpu.create_nd_tdesc %a[%i, %c0] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16, #sg_map_fp16_a>

      // CHECK: xegpu.create_nd_tdesc
      // CHECK-SAME: memref<1024x1024xbf16>
      // CHECK-SAME: -> !xegpu.tensor_desc<16x16xbf16, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
      %2 = xegpu.create_nd_tdesc %b[%c0, %j] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16, #sg_map_fp16_b>

      %3 = arith.constant dense<0.0> : vector<8x1xf32>

      %tmp0, %tmp1, %result = scf.for %k= %c0 to %c1024 step %c16 iter_args(%subA = %1, %subB = %2, %subC = %3)
              -> (!xegpu.tensor_desc<8x16xbf16, #sg_map_fp16_a>, !xegpu.tensor_desc<16x16xbf16, #sg_map_fp16_b>, vector<8x1xf32>) {
        // CHECK: xegpu.load_nd
        // CHECK-SAME: vector<4x1x2xbf16>
        %4 = xegpu.load_nd %subA {vnni_axis = 1} : !xegpu.tensor_desc<8x16xbf16, #sg_map_fp16_a> -> vector<4x1x2xbf16>

        // CHECK: xegpu.load_nd
        // CHECK-SAME: vector<8x1x2xbf16>
        %5 = xegpu.load_nd %subB {vnni_axis = 0} : !xegpu.tensor_desc<16x16xbf16, #sg_map_fp16_b> -> vector<8x1x2xbf16>

        // CHECK: xegpu.dpas
        // CHECK-SAME: vector<4x1x2xbf16>, vector<8x1x2xbf16>, vector<8x1xf32> -> vector<8x1xf32>
        %6 = xegpu.dpas %4, %5, %subC  : vector<4x1x2xbf16>, vector<8x1x2xbf16>, vector<8x1xf32> -> vector<8x1xf32>

        %7 = xegpu.update_nd_offset %subA, [%c0, %c16] : !xegpu.tensor_desc<8x16xbf16, #sg_map_fp16_a>
            -> !xegpu.tensor_desc<8x16xbf16, #sg_map_fp16_a>

        %8 = xegpu.update_nd_offset %subB, [%c16, %c0] : !xegpu.tensor_desc<16x16xbf16, #sg_map_fp16_b>
            -> !xegpu.tensor_desc<16x16xbf16, #sg_map_fp16_b>

        scf.yield %7, %8, %6: !xegpu.tensor_desc<8x16xbf16, #sg_map_fp16_a>, !xegpu.tensor_desc<16x16xbf16, #sg_map_fp16_b>, vector<8x1xf32>
      }

      // CHECK: xegpu.create_nd_tdesc
      // CHECK-SAME: memref<1024x1024xf32>
      %9 = xegpu.create_nd_tdesc %c[%i, %j] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c>

      // CHECK: xegpu.store_nd
      // CHECK-SAME: vector<8x1xf32>
      xegpu.store_nd %result, %9 : vector<8x1xf32>, !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c>
    }
  }
  return
}
