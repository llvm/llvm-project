// RUN: mlir-opt %s -convert-to-llvm | FileCheck %s
// RUN: mlir-opt %s -convert-vector-to-llvm='enable-x86' -convert-to-llvm -canonicalize='region-simplify=aggressive' -cse -sccp | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// The immediate synthesized by the dot product lowering must be an attribute.
// CHECK-LABEL: llvm.func @dot
// CHECK: llvm.call_intrinsic "llvm.x86.avx.dp.ps.256"(%{{.*}}, %{{.*}}) {immarg_positions = array<i32: 2>, immarg_values = [-1 : i8]}
// LLVM-LABEL: define <8 x float> @dot(
// LLVM: call <8 x float> @llvm.x86.avx.dp.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i8 -1)
func.func @dot(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
  %r = x86.avx.intr.dot %a, %b : vector<8xf32>
  return %r : vector<8xf32>
}

// Different immediate operands must remain constant after block merging.
// CHECK-LABEL: llvm.func @roundscale_branches
// CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.rndscale.ps.512"(%{{.*}}, %{{.*}}, %{{.*}}) {immarg_positions = array<i32: 1, 4>, immarg_values = [1 : i32, 4 : i32]}
// CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.rndscale.ps.512"(%{{.*}}, %{{.*}}, %{{.*}}) {immarg_positions = array<i32: 1, 4>, immarg_values = [2 : i32, 4 : i32]}
// LLVM-LABEL: define <16 x float> @roundscale_branches(
// LLVM: call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float> %{{.*}}, i32 1, <16 x float> %{{.*}}, i16 %{{.*}}, i32 4)
// LLVM: call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float> %{{.*}}, i32 2, <16 x float> %{{.*}}, i16 %{{.*}}, i32 4)
func.func @roundscale_branches(%cond: i1, %src: vector<16xf32>, %mask: i16) -> vector<16xf32> {
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %rounding = arith.constant 4 : i32
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %a = x86.avx512.mask.rndscale %src, %one, %src, %mask, %rounding : vector<16xf32>
  return %a : vector<16xf32>
^bb2:
  %b = x86.avx512.mask.rndscale %src, %two, %src, %mask, %rounding : vector<16xf32>
  return %b : vector<16xf32>
}

// CHECK-LABEL: llvm.func @scalef
// CHECK: llvm.call_intrinsic "llvm.x86.avx512.mask.scalef.pd.512"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {immarg_positions = array<i32: 4>, immarg_values = [4 : i32]}
// LLVM-LABEL: define <8 x double> @scalef(
// LLVM: call <8 x double> @llvm.x86.avx512.mask.scalef.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}, i8 %{{.*}}, i32 4)
func.func @scalef(%a: vector<8xf64>, %b: vector<8xf64>, %mask: i8) -> vector<8xf64> {
  %rounding = arith.constant 4 : i32
  %r = x86.avx512.mask.scalef %a, %b, %a, %mask, %rounding : vector<8xf64>
  return %r : vector<8xf64>
}
