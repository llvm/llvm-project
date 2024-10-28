// DEFINE: %{opts} =
// DEFINE: %{entry} = main
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -test-lower-to-arm-sme=%{opts} -test-lower-to-llvm -o %t
// DEFINE: %{run} = %mcr_aarch64_cmd %t \
// DEFINE:   -march=aarch64 -mattr=+sve,+sme \
// DEFINE:   -e %{entry} -entry-point-result=void \
// DEFINE:   -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils,%native_mlir_arm_runner_utils,%native_arm_sme_abi_shlib

// RUN: %{compile}

// RUN: %{run} | FileCheck %s

// Check result is the same when outerproducts are not combined into widening
// variant.

// REDEFINE: %{opts} = fuse-outer-products=false
// RUN: %{run} | FileCheck %s

func.func @main() {
  %c128 = arith.constant 128 : i32
  func.call @setArmSVLBits(%c128) : (i32) -> ()

  func.call @test_outerproduct_f16f16f32() : () -> ()

  // TODO: A bug in QEMU causes masked FMOPAs to hang [1]. Should be fixed in
  // 8.2.0, this test currently isn't run, once this version is available in CI
  // it can be run. The output without check lines in the function are correct
  // and have been verified on a version with the fix.
  // [1] https://gitlab.com/qemu-project/qemu/-/issues/1985
  //func.call @test_masked_outerproduct_f16f16f32() : () -> ()

  return
}

func.func @test_outerproduct_f16f16f32() {
  %undef = llvm.mlir.undef : vector<[4]xf16>

  %a0_data = arith.constant dense<[0., 2., 4., 6.]> : vector<4xf16>
  %b0_data = arith.constant dense<[1., 3., 5., 7.]> : vector<4xf16>
  %a1_data = arith.constant dense<[8., 10., 12., 14.]> : vector<4xf16>
  %b1_data = arith.constant dense<[9., 11., 13., 15.]> : vector<4xf16>

  %a0 = vector.scalable.insert %a0_data, %undef[0] : vector<4xf16> into vector<[4]xf16>
  %b0 = vector.scalable.insert %b0_data, %undef[0] : vector<4xf16> into vector<[4]xf16>
  %a1 = vector.scalable.insert %a1_data, %undef[0] : vector<4xf16> into vector<[4]xf16>
  %b1 = vector.scalable.insert %b1_data, %undef[0] : vector<4xf16> into vector<[4]xf16>

  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %acc = arith.constant dense<7.0> : vector<[4]x[4]xf32>
  %0 = vector.outerproduct %a0_ext, %b0_ext, %acc : vector<[4]xf32>, vector<[4]xf32>
  %1 = vector.outerproduct %a1_ext, %b1_ext, %0 : vector<[4]xf32>, vector<[4]xf32>

  // CHECK:      (  79,  95, 111, 127 )
  // CHECK-NEXT: (  99, 123, 147, 171 )
  // CHECK-NEXT: ( 119, 151, 183, 215 )
  // CHECK-NEXT: ( 139, 179, 219, 259 )
  vector.print %1 : vector<[4]x[4]xf32>

  return
}

func.func @test_masked_outerproduct_f16f16f32() {
  %undef = llvm.mlir.undef : vector<[4]xf16>

  %a0_data = arith.constant dense<[0., 2., 4., 6.]> : vector<4xf16>
  %b0_data = arith.constant dense<[1., 3., 5., 7.]> : vector<4xf16>
  %a1_data = arith.constant dense<[8., 10., 12., 14.]> : vector<4xf16>
  %b1_data = arith.constant dense<[9., 11., 13., 15.]> : vector<4xf16>

  %a0 = vector.scalable.insert %a0_data, %undef[0] : vector<4xf16> into vector<[4]xf16>
  %b0 = vector.scalable.insert %b0_data, %undef[0] : vector<4xf16> into vector<[4]xf16>
  %a1 = vector.scalable.insert %a1_data, %undef[0] : vector<4xf16> into vector<[4]xf16>
  %b1 = vector.scalable.insert %b1_data, %undef[0] : vector<4xf16> into vector<[4]xf16>

  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %acc = arith.constant dense<7.0> : vector<[4]x[4]xf32>

  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %mask0 = vector.create_mask %c2, %c3 : vector<[4]x[4]xi1>
  %mask1 = vector.create_mask %c3, %c2 : vector<[4]x[4]xi1>

  %0 = vector.mask %mask0 {
    vector.outerproduct %a0_ext, %b0_ext, %acc : vector<[4]xf32>, vector<[4]xf32>
  } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>

  %1 = vector.mask %mask1 {
    vector.outerproduct %a1_ext, %b1_ext, %0 : vector<[4]xf32>, vector<[4]xf32>
  } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>

  // TODO: CHECK these lines once QEMU is fixed.
  // (  79,  95,  7, 7 )
  // (  99, 123, 17, 7 )
  // ( 115, 139,  7, 7 )
  // (   7,   7,  7, 7 )
  vector.print %1 : vector<[4]x[4]xf32>

  return
}

func.func private @setArmSVLBits(%bits : i32)
