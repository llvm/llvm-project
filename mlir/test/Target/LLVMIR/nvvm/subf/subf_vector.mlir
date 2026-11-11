// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// vector<2xf16> - vector<2xf16> -> vector<2xf16>
llvm.func @subf_vector_f16_f16(%a : vector<2xf16>, %b : vector<2xf16>) -> vector<2xf16> {
  // CHECK-LABEL: define <2 x half> @subf_vector_f16_f16(<2 x half> %0, <2 x half> %1) {
  // CHECK-NEXT: %3 = fneg <2 x half> %1
  // CHECK-NEXT: %4 = call <2 x half> @llvm.nvvm.fadd.v2f16(<2 x half> %0, <2 x half> %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = fneg <2 x half> %4
  // CHECK-NEXT: %6 = call <2 x half> @llvm.nvvm.fadd.v2f16(<2 x half> %4, <2 x half> %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = fneg <2 x half> %6
  // CHECK-NEXT: %8 = call <2 x half> @llvm.nvvm.fadd.ftz.v2f16(<2 x half> %6, <2 x half> %7, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %9 = fneg <2 x half> %8
  // CHECK-NEXT: %10 = call <2 x half> @llvm.nvvm.fadd.sat.v2f16(<2 x half> %8, <2 x half> %9, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %11 = fneg <2 x half> %10
  // CHECK-NEXT: %12 = call <2 x half> @llvm.nvvm.fadd.ftz.sat.v2f16(<2 x half> %10, <2 x half> %11, /* rnd=rn */ i32 1)
  // CHECK-NEXT: ret <2 x half> %4
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : vector<2xf16>
  %f2 = nvvm.subf %f1, %f1 rnd = <rn> : vector<2xf16>
  %f3 = nvvm.subf %f2, %f2 rnd = <rn> ftz = true : vector<2xf16>
  %f4 = nvvm.subf %f3, %f3 rnd = <rn> sat = <sat> : vector<2xf16>
  %f5 = nvvm.subf %f4, %f4 rnd = <rn> sat = <sat> ftz = true : vector<2xf16>
  llvm.return %f1 : vector<2xf16>
}

// vector<2xbf16> - vector<2xbf16> -> vector<2xbf16>
llvm.func @subf_vector_bf16_bf16(%a : vector<2xbf16>, %b : vector<2xbf16>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @subf_vector_bf16_bf16(<2 x bfloat> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = fneg <2 x bfloat> %1
  // CHECK-NEXT: %4 = call <2 x bfloat> @llvm.nvvm.fadd.v2bf16(<2 x bfloat> %0, <2 x bfloat> %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = fneg <2 x bfloat> %4
  // CHECK-NEXT: %6 = call <2 x bfloat> @llvm.nvvm.fadd.v2bf16(<2 x bfloat> %4, <2 x bfloat> %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: ret <2 x bfloat> %6
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : vector<2xbf16>
  %f2 = nvvm.subf %f1, %f1 rnd = <rn> : vector<2xbf16>
  llvm.return %f2 : vector<2xbf16>
}

// vector<2xf32> - vector<2xf32> -> vector<2xf32>
llvm.func @subf_vector_f32_f32_rn(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @subf_vector_f32_f32_rn(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = fneg <2 x float> %1
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %0, <2 x float> %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = fneg <2 x float> %4
  // CHECK-NEXT: %6 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %4, <2 x float> %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = fneg <2 x float> %6
  // CHECK-NEXT: %8 = extractelement <2 x float> %6, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x float> %7, i32 0
  // CHECK-NEXT: %10 = call float @llvm.nvvm.fadd.sat.f32(float %8, float %9, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %11 = insertelement <2 x float> poison, float %10, i32 0
  // CHECK-NEXT: %12 = extractelement <2 x float> %6, i32 1
  // CHECK-NEXT: %13 = extractelement <2 x float> %7, i32 1
  // CHECK-NEXT: %14 = call float @llvm.nvvm.fadd.sat.f32(float %12, float %13, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %15 = insertelement <2 x float> %11, float %14, i32 1
  // CHECK-NEXT: %16 = fneg <2 x float> %15
  // CHECK-NEXT: %17 = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %15, <2 x float> %16, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %18 = fneg <2 x float> %17
  // CHECK-NEXT: %19 = extractelement <2 x float> %17, i32 0
  // CHECK-NEXT: %20 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %21 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %19, float %20, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %22 = insertelement <2 x float> poison, float %21, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %17, i32 1
  // CHECK-NEXT: %24 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %25 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %23, float %24, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %26 = insertelement <2 x float> %22, float %25, i32 1
  // CHECK-NEXT: ret <2 x float> %17
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : vector<2xf32>
  %f2 = nvvm.subf %f1, %f1 rnd = <rn> : vector<2xf32>
  %f3 = nvvm.subf %f2, %f2 rnd = <rn> sat = <sat> : vector<2xf32>
  %f4 = nvvm.subf %f3, %f3 rnd = <rn> ftz = true : vector<2xf32>
  %f5 = nvvm.subf %f4, %f4 rnd = <rn> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @subf_vector_f32_f32_rm(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @subf_vector_f32_f32_rm(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = fneg <2 x float> %1
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %0, <2 x float> %3, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %5 = fneg <2 x float> %4
  // CHECK-NEXT: %6 = extractelement <2 x float> %4, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %5, i32 0
  // CHECK-NEXT: %8 = call float @llvm.nvvm.fadd.sat.f32(float %6, float %7, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %9 = insertelement <2 x float> poison, float %8, i32 0
  // CHECK-NEXT: %10 = extractelement <2 x float> %4, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x float> %5, i32 1
  // CHECK-NEXT: %12 = call float @llvm.nvvm.fadd.sat.f32(float %10, float %11, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %13 = insertelement <2 x float> %9, float %12, i32 1
  // CHECK-NEXT: %14 = fneg <2 x float> %13
  // CHECK-NEXT: %15 = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %13, <2 x float> %14, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %16 = fneg <2 x float> %15
  // CHECK-NEXT: %17 = extractelement <2 x float> %15, i32 0
  // CHECK-NEXT: %18 = extractelement <2 x float> %16, i32 0
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %17, float %18, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %20 = insertelement <2 x float> poison, float %19, i32 0
  // CHECK-NEXT: %21 = extractelement <2 x float> %15, i32 1
  // CHECK-NEXT: %22 = extractelement <2 x float> %16, i32 1
  // CHECK-NEXT: %23 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %21, float %22, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %24 = insertelement <2 x float> %20, float %23, i32 1
  // CHECK-NEXT: ret <2 x float> %24
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b rnd = <rm> : vector<2xf32>
  %f2 = nvvm.subf %f1, %f1 rnd = <rm> sat = <sat> : vector<2xf32>
  %f3 = nvvm.subf %f2, %f2 rnd = <rm> ftz = true : vector<2xf32>
  %f4 = nvvm.subf %f3, %f3 rnd = <rm> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @subf_vector_f32_f32_rp(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @subf_vector_f32_f32_rp(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = fneg <2 x float> %1
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %0, <2 x float> %3, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %5 = fneg <2 x float> %4
  // CHECK-NEXT: %6 = extractelement <2 x float> %4, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %5, i32 0
  // CHECK-NEXT: %8 = call float @llvm.nvvm.fadd.sat.f32(float %6, float %7, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %9 = insertelement <2 x float> poison, float %8, i32 0
  // CHECK-NEXT: %10 = extractelement <2 x float> %4, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x float> %5, i32 1
  // CHECK-NEXT: %12 = call float @llvm.nvvm.fadd.sat.f32(float %10, float %11, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %13 = insertelement <2 x float> %9, float %12, i32 1
  // CHECK-NEXT: %14 = fneg <2 x float> %13
  // CHECK-NEXT: %15 = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %13, <2 x float> %14, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %16 = fneg <2 x float> %15
  // CHECK-NEXT: %17 = extractelement <2 x float> %15, i32 0
  // CHECK-NEXT: %18 = extractelement <2 x float> %16, i32 0
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %17, float %18, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %20 = insertelement <2 x float> poison, float %19, i32 0
  // CHECK-NEXT: %21 = extractelement <2 x float> %15, i32 1
  // CHECK-NEXT: %22 = extractelement <2 x float> %16, i32 1
  // CHECK-NEXT: %23 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %21, float %22, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %24 = insertelement <2 x float> %20, float %23, i32 1
  // CHECK-NEXT: ret <2 x float> %24
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b rnd = <rp> : vector<2xf32>
  %f2 = nvvm.subf %f1, %f1 rnd = <rp> sat = <sat> : vector<2xf32>
  %f3 = nvvm.subf %f2, %f2 rnd = <rp> ftz = true : vector<2xf32>
  %f4 = nvvm.subf %f3, %f3 rnd = <rp> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @subf_vector_f32_f32_rz(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @subf_vector_f32_f32_rz(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = fneg <2 x float> %1
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %0, <2 x float> %3, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %5 = fneg <2 x float> %4
  // CHECK-NEXT: %6 = extractelement <2 x float> %4, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %5, i32 0
  // CHECK-NEXT: %8 = call float @llvm.nvvm.fadd.sat.f32(float %6, float %7, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %9 = insertelement <2 x float> poison, float %8, i32 0
  // CHECK-NEXT: %10 = extractelement <2 x float> %4, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x float> %5, i32 1
  // CHECK-NEXT: %12 = call float @llvm.nvvm.fadd.sat.f32(float %10, float %11, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %13 = insertelement <2 x float> %9, float %12, i32 1
  // CHECK-NEXT: %14 = fneg <2 x float> %13
  // CHECK-NEXT: %15 = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %13, <2 x float> %14, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %16 = fneg <2 x float> %15
  // CHECK-NEXT: %17 = extractelement <2 x float> %15, i32 0
  // CHECK-NEXT: %18 = extractelement <2 x float> %16, i32 0
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %17, float %18, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %20 = insertelement <2 x float> poison, float %19, i32 0
  // CHECK-NEXT: %21 = extractelement <2 x float> %15, i32 1
  // CHECK-NEXT: %22 = extractelement <2 x float> %16, i32 1
  // CHECK-NEXT: %23 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %21, float %22, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %24 = insertelement <2 x float> %20, float %23, i32 1
  // CHECK-NEXT: ret <2 x float> %24
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b rnd = <rz> : vector<2xf32>
  %f2 = nvvm.subf %f1, %f1 rnd = <rz> sat = <sat> : vector<2xf32>
  %f3 = nvvm.subf %f2, %f2 rnd = <rz> ftz = true : vector<2xf32>
  %f4 = nvvm.subf %f3, %f3 rnd = <rz> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

// vector<2xf64> - vector<2xf64> -> vector<2xf64>
llvm.func @subf_vector_f64_f64_rn(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @subf_vector_f64_f64_rn(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = fneg <2 x double> %1
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %3, i32 0
  // CHECK-NEXT: %6 = call double @llvm.nvvm.fadd.f64(double %4, double %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %3, i32 1
  // CHECK-NEXT: %10 = call double @llvm.nvvm.fadd.f64(double %8, double %9, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %11 = insertelement <2 x double> %7, double %10, i32 1
  // CHECK-NEXT: %12 = fneg <2 x double> %11
  // CHECK-NEXT: %13 = extractelement <2 x double> %11, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x double> %12, i32 0
  // CHECK-NEXT: %15 = call double @llvm.nvvm.fadd.f64(double %13, double %14, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %16 = insertelement <2 x double> poison, double %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x double> %11, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x double> %12, i32 1
  // CHECK-NEXT: %19 = call double @llvm.nvvm.fadd.f64(double %17, double %18, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %20 = insertelement <2 x double> %16, double %19, i32 1
  // CHECK-NEXT: ret <2 x double> %20
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : vector<2xf64>
  %f2 = nvvm.subf %f1, %f1 rnd = <rn> : vector<2xf64>
  llvm.return %f2 : vector<2xf64>
}

llvm.func @subf_vector_f64_f64_rm(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @subf_vector_f64_f64_rm(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = fneg <2 x double> %1
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %3, i32 0
  // CHECK-NEXT: %6 = call double @llvm.nvvm.fadd.f64(double %4, double %5, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %3, i32 1
  // CHECK-NEXT: %10 = call double @llvm.nvvm.fadd.f64(double %8, double %9, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %11 = insertelement <2 x double> %7, double %10, i32 1
  // CHECK-NEXT: ret <2 x double> %11
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b rnd = <rm> : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @subf_vector_f64_f64_rp(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @subf_vector_f64_f64_rp(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = fneg <2 x double> %1
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %3, i32 0
  // CHECK-NEXT: %6 = call double @llvm.nvvm.fadd.f64(double %4, double %5, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %3, i32 1
  // CHECK-NEXT: %10 = call double @llvm.nvvm.fadd.f64(double %8, double %9, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %11 = insertelement <2 x double> %7, double %10, i32 1
  // CHECK-NEXT: ret <2 x double> %11
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b rnd = <rp> : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @subf_vector_f64_f64_rz(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @subf_vector_f64_f64_rz(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = fneg <2 x double> %1
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %3, i32 0
  // CHECK-NEXT: %6 = call double @llvm.nvvm.fadd.f64(double %4, double %5, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %3, i32 1
  // CHECK-NEXT: %10 = call double @llvm.nvvm.fadd.f64(double %8, double %9, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %11 = insertelement <2 x double> %7, double %10, i32 1
  // CHECK-NEXT: ret <2 x double> %11
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b rnd = <rz> : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}
