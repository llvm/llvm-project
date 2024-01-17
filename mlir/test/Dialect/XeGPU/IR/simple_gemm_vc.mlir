// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// ---- BF16 VC ------

// CHECK-LABEL: func @test_gemm_vc_bf16({{.*}}) {
func.func @test_gemm_vc_bf16(%a : memref<1024x1024xbf16>, %b: memref<1024x1024xbf16>, %c: memref<1024x1024xf32>) {
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
      // CHECK-SAME: memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>
      %1 = xegpu.create_nd_tdesc %a[%i, %c0] {mode = vc} : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>

      // CHECK: xegpu.create_nd_tdesc
      // CHECK-SAME: memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>
      %2 = xegpu.create_nd_tdesc %b[%c0, %j] {mode = vc} : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>

      %3 = arith.constant dense<0.0> : vector<8x16xf32>

      %tmp0, %tmp1, %result = scf.for %k= %c0 to %c1024 step %c16
                                iter_args(%subA = %1, %subB = %2, %subC = %3)
                                  -> (!xegpu.tensor_desc<8x16xbf16>, !xegpu.tensor_desc<16x16xbf16>, vector<8x16xf32>) {
        // CHECK: xegpu.load_nd
        // CHECK-SAME: !xegpu.tensor_desc<8x16xbf16> -> vector<8x8x2xbf16>
        %4 = xegpu.load_nd %subA {mode = vc, vnni_axis = 1} : !xegpu.tensor_desc<8x16xbf16> -> vector<8x8x2xbf16>

        // CHECK: xegpu.load_nd
        // CHECK-SAME: !xegpu.tensor_desc<16x16xbf16> -> vector<8x16x2xbf16>
        %5 = xegpu.load_nd %subB {mode = vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xbf16> -> vector<8x16x2xbf16>

        // CHECK: xegpu.dpas
        // CHECK-SAME: vector<8x8x2xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
        %6 = xegpu.dpas %4, %5, %subC {mode = vc} : vector<8x8x2xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>

        %7 = xegpu.update_nd_offset %subA, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xbf16> -> !xegpu.tensor_desc<8x16xbf16>

        %8 = xegpu.update_nd_offset %subB, [%c16, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xbf16> -> !xegpu.tensor_desc<16x16xbf16>

        scf.yield %7, %8, %6: !xegpu.tensor_desc<8x16xbf16>, !xegpu.tensor_desc<16x16xbf16>, vector<8x16xf32>
      }

      // CHECK: xegpu.create_nd_tdesc
      // CHECK-SAME: memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %9 = xegpu.create_nd_tdesc %c[%i, %j] {mode = vc} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>

      // CHECK: xegpu.store_nd
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %result, %9 {mode = vc}: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    }
  }
  return
}
