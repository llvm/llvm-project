// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// vector<2xf16> + vector<2xf16> -> vector<2xf16>
llvm.func @addf_vector_f16_f16(%a : vector<2xf16>, %b : vector<2xf16>) -> vector<2xf16> {
  // CHECK-LABEL: define <2 x half> @addf_vector_f16_f16(<2 x half> %0, <2 x half> %1) {
  // CHECK-NEXT: %3 = call <2 x half> @llvm.nvvm.fadd.v2f16(<2 x half> %0, <2 x half> %1, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %4 = call <2 x half> @llvm.nvvm.fadd.v2f16(<2 x half> %3, <2 x half> %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = call <2 x half> @llvm.nvvm.fadd.ftz.v2f16(<2 x half> %4, <2 x half> %4, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %6 = call <2 x half> @llvm.nvvm.fadd.sat.v2f16(<2 x half> %5, <2 x half> %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = call <2 x half> @llvm.nvvm.fadd.ftz.sat.v2f16(<2 x half> %6, <2 x half> %6, /* rnd=rn */ i32 1)
  // CHECK-NEXT: ret <2 x half> %3
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : vector<2xf16>
  %f2 = nvvm.addf %f1, %f1 rnd = <rn> : vector<2xf16>
  %f3 = nvvm.addf %f2, %f2 rnd = <rn> ftz = true : vector<2xf16>
  %f4 = nvvm.addf %f3, %f3 rnd = <rn> sat = <sat> : vector<2xf16>
  %f5 = nvvm.addf %f4, %f4 rnd = <rn> sat = <sat> ftz = true : vector<2xf16>
  llvm.return %f1 : vector<2xf16>
}

// vector<2xbf16> + vector<2xbf16> -> vector<2xbf16>
llvm.func @addf_vector_bf16_bf16(%a : vector<2xbf16>, %b : vector<2xbf16>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @addf_vector_bf16_bf16(<2 x bfloat> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = call <2 x bfloat> @llvm.nvvm.fadd.v2bf16(<2 x bfloat> %0, <2 x bfloat> %1, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %4 = call <2 x bfloat> @llvm.nvvm.fadd.v2bf16(<2 x bfloat> %3, <2 x bfloat> %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: ret <2 x bfloat> %4
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : vector<2xbf16>
  %f2 = nvvm.addf %f1, %f1 rnd = <rn> : vector<2xbf16>
  llvm.return %f2 : vector<2xbf16>
}

// vector<2xf32> + vector<2xf32> -> vector<2xf32>
llvm.func @addf_vector_f32_f32_rn(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @addf_vector_f32_f32_rn(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %0, <2 x float> %1, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %3, <2 x float> %3, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = extractelement <2 x float> %4, i32 0
  // CHECK-NEXT: %6 = extractelement <2 x float> %4, i32 0
  // CHECK-NEXT: %7 = call float @llvm.nvvm.fadd.sat.f32(float %5, float %6, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x float> %4, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x float> %4, i32 1
  // CHECK-NEXT: %11 = call float @llvm.nvvm.fadd.sat.f32(float %9, float %10, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %12 = insertelement <2 x float> %8, float %11, i32 1
  // CHECK-NEXT: %13 = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %12, <2 x float> %12, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %14 = extractelement <2 x float> %13, i32 0
  // CHECK-NEXT: %15 = extractelement <2 x float> %13, i32 0
  // CHECK-NEXT: %16 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %14, float %15, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %17 = insertelement <2 x float> poison, float %16, i32 0
  // CHECK-NEXT: %18 = extractelement <2 x float> %13, i32 1
  // CHECK-NEXT: %19 = extractelement <2 x float> %13, i32 1
  // CHECK-NEXT: %20 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %18, float %19, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %21 = insertelement <2 x float> %17, float %20, i32 1
  // CHECK-NEXT: ret <2 x float> %13
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : vector<2xf32>
  %f2 = nvvm.addf %f1, %f1 rnd = <rn> : vector<2xf32>
  %f3 = nvvm.addf %f2, %f2 rnd = <rn> sat = <sat> : vector<2xf32>
  %f4 = nvvm.addf %f3, %f3 rnd = <rn> ftz = true : vector<2xf32>
  %f5 = nvvm.addf %f4, %f4 rnd = <rn> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @addf_vector_f32_f32_rm(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @addf_vector_f32_f32_rm(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %0, <2 x float> %1, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %4 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fadd.sat.f32(float %4, float %5, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %10 = call float @llvm.nvvm.fadd.sat.f32(float %8, float %9, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %11 = insertelement <2 x float> %7, float %10, i32 1
  // CHECK-NEXT: %12 = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %11, <2 x float> %11, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %13 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %15 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %13, float %14, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %16 = insertelement <2 x float> poison, float %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %17, float %18, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %20 = insertelement <2 x float> %16, float %19, i32 1
  // CHECK-NEXT: ret <2 x float> %20
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b rnd = <rm> : vector<2xf32>
  %f2 = nvvm.addf %f1, %f1 rnd = <rm> sat = <sat> : vector<2xf32>
  %f3 = nvvm.addf %f2, %f2 rnd = <rm> ftz = true : vector<2xf32>
  %f4 = nvvm.addf %f3, %f3 rnd = <rm> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @addf_vector_f32_f32_rp(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @addf_vector_f32_f32_rp(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %0, <2 x float> %1, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %4 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fadd.sat.f32(float %4, float %5, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %10 = call float @llvm.nvvm.fadd.sat.f32(float %8, float %9, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %11 = insertelement <2 x float> %7, float %10, i32 1
  // CHECK-NEXT: %12 = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %11, <2 x float> %11, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %13 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %15 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %13, float %14, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %16 = insertelement <2 x float> poison, float %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %17, float %18, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %20 = insertelement <2 x float> %16, float %19, i32 1
  // CHECK-NEXT: ret <2 x float> %20
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b rnd = <rp> : vector<2xf32>
  %f2 = nvvm.addf %f1, %f1 rnd = <rp> sat = <sat> : vector<2xf32>
  %f3 = nvvm.addf %f2, %f2 rnd = <rp> ftz = true : vector<2xf32>
  %f4 = nvvm.addf %f3, %f3 rnd = <rp> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @addf_vector_f32_f32_rz(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @addf_vector_f32_f32_rz(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %0, <2 x float> %1, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %4 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %6 = call float @llvm.nvvm.fadd.sat.f32(float %4, float %5, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %10 = call float @llvm.nvvm.fadd.sat.f32(float %8, float %9, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %11 = insertelement <2 x float> %7, float %10, i32 1
  // CHECK-NEXT: %12 = call <2 x float> @llvm.nvvm.fadd.ftz.v2f32(<2 x float> %11, <2 x float> %11, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %13 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %15 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %13, float %14, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %16 = insertelement <2 x float> poison, float %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fadd.ftz.sat.f32(float %17, float %18, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %20 = insertelement <2 x float> %16, float %19, i32 1
  // CHECK-NEXT: ret <2 x float> %20
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b rnd = <rz> : vector<2xf32>
  %f2 = nvvm.addf %f1, %f1 rnd = <rz> sat = <sat> : vector<2xf32>
  %f3 = nvvm.addf %f2, %f2 rnd = <rz> ftz = true : vector<2xf32>
  %f4 = nvvm.addf %f3, %f3 rnd = <rz> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

// vector<2xf64> + vector<2xf64> -> vector<2xf64>
llvm.func @addf_vector_f64_f64_rn(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @addf_vector_f64_f64_rn(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = call double @llvm.nvvm.fadd.f64(double %3, double %4, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %6 = insertelement <2 x double> poison, double %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %9 = call double @llvm.nvvm.fadd.f64(double %7, double %8, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %10 = insertelement <2 x double> %6, double %9, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x double> %10, i32 0
  // CHECK-NEXT: %12 = extractelement <2 x double> %10, i32 0
  // CHECK-NEXT: %13 = call double @llvm.nvvm.fadd.f64(double %11, double %12, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %14 = insertelement <2 x double> poison, double %13, i32 0
  // CHECK-NEXT: %15 = extractelement <2 x double> %10, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x double> %10, i32 1
  // CHECK-NEXT: %17 = call double @llvm.nvvm.fadd.f64(double %15, double %16, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %18 = insertelement <2 x double> %14, double %17, i32 1
  // CHECK-NEXT: ret <2 x double> %18
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : vector<2xf64>
  %f2 = nvvm.addf %f1, %f1 rnd = <rn> : vector<2xf64>
  llvm.return %f2 : vector<2xf64>
}

llvm.func @addf_vector_f64_f64_rm(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @addf_vector_f64_f64_rm(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = call double @llvm.nvvm.fadd.f64(double %3, double %4, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %6 = insertelement <2 x double> poison, double %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %9 = call double @llvm.nvvm.fadd.f64(double %7, double %8, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %10 = insertelement <2 x double> %6, double %9, i32 1
  // CHECK-NEXT: ret <2 x double> %10
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b rnd = <rm> : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @addf_vector_f64_f64_rp(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @addf_vector_f64_f64_rp(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = call double @llvm.nvvm.fadd.f64(double %3, double %4, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %6 = insertelement <2 x double> poison, double %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %9 = call double @llvm.nvvm.fadd.f64(double %7, double %8, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %10 = insertelement <2 x double> %6, double %9, i32 1
  // CHECK-NEXT: ret <2 x double> %10
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b rnd = <rp> : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @addf_vector_f64_f64_rz(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @addf_vector_f64_f64_rz(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = call double @llvm.nvvm.fadd.f64(double %3, double %4, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %6 = insertelement <2 x double> poison, double %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %9 = call double @llvm.nvvm.fadd.f64(double %7, double %8, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %10 = insertelement <2 x double> %6, double %9, i32 1
  // CHECK-NEXT: ret <2 x double> %10
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b rnd = <rz> : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}
