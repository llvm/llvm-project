// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// vector<2xf16> + vector<2xf16> -> vector<2xf16>
llvm.func @addf_vector_f16_f16(%a : vector<2xf16>, %b : vector<2xf16>) -> vector<2xf16> {
  // CHECK-LABEL: define <2 x half> @addf_vector_f16_f16(<2 x half> %0, <2 x half> %1) {
  // CHECK-NEXT: %3 = fadd <2 x half> %0, %1
  // CHECK-NEXT: %4 = fadd <2 x half> %3, %3
  // CHECK-NEXT: %5 = call <2 x half> @llvm.nvvm.add.rn.sat.v2f16(<2 x half> %4, <2 x half> %4)
  // CHECK-NEXT: %6 = call <2 x half> @llvm.nvvm.add.rn.ftz.sat.v2f16(<2 x half> %5, <2 x half> %5)
  // CHECK-NEXT: ret <2 x half> %3
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : vector<2xf16>
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf16>
  %f3 = nvvm.addf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>
  %f4 = nvvm.addf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>
  llvm.return %f1 : vector<2xf16>
}

// vector<2xbf16> + vector<2xbf16> -> vector<2xbf16>
llvm.func @addf_vector_bf16_bf16(%a : vector<2xbf16>, %b : vector<2xbf16>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @addf_vector_bf16_bf16(<2 x bfloat> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = fadd <2 x bfloat> %0, %1
  // CHECK-NEXT: %4 = fadd <2 x bfloat> %3, %3
  // CHECK-NEXT: ret <2 x bfloat> %4
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : vector<2xbf16>
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xbf16>
  llvm.return %f2 : vector<2xbf16>
}

// vector<2xf32> + vector<2xf32> -> vector<2xf32>
llvm.func @addf_vector_f32_f32_rn(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @addf_vector_f32_f32_rn(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.f(float %3, float %4)
  // CHECK-NEXT: %6 = insertelement <2 x float> poison, float %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %9 = call float @llvm.nvvm.add.rn.f(float %7, float %8)
  // CHECK-NEXT: %10 = insertelement <2 x float> %6, float %9, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x float> %10, i32 0
  // CHECK-NEXT: %12 = extractelement <2 x float> %10, i32 0
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rn.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> poison, float %13, i32 0
  // CHECK-NEXT: %15 = extractelement <2 x float> %10, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x float> %10, i32 1
  // CHECK-NEXT: %17 = call float @llvm.nvvm.add.rn.f(float %15, float %16)
  // CHECK-NEXT: %18 = insertelement <2 x float> %14, float %17, i32 1
  // CHECK-NEXT: %19 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %20 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %21 = call float @llvm.nvvm.add.rn.sat.f(float %19, float %20)
  // CHECK-NEXT: %22 = insertelement <2 x float> poison, float %21, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %24 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %25 = call float @llvm.nvvm.add.rn.sat.f(float %23, float %24)
  // CHECK-NEXT: %26 = insertelement <2 x float> %22, float %25, i32 1
  // CHECK-NEXT: %27 = extractelement <2 x float> %26, i32 0
  // CHECK-NEXT: %28 = extractelement <2 x float> %26, i32 0
  // CHECK-NEXT: %29 = call float @llvm.nvvm.add.rn.ftz.f(float %27, float %28)
  // CHECK-NEXT: %30 = insertelement <2 x float> poison, float %29, i32 0
  // CHECK-NEXT: %31 = extractelement <2 x float> %26, i32 1
  // CHECK-NEXT: %32 = extractelement <2 x float> %26, i32 1
  // CHECK-NEXT: %33 = call float @llvm.nvvm.add.rn.ftz.f(float %31, float %32)
  // CHECK-NEXT: %34 = insertelement <2 x float> %30, float %33, i32 1
  // CHECK-NEXT: %35 = extractelement <2 x float> %34, i32 0
  // CHECK-NEXT: %36 = extractelement <2 x float> %34, i32 0
  // CHECK-NEXT: %37 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %35, float %36)
  // CHECK-NEXT: %38 = insertelement <2 x float> poison, float %37, i32 0
  // CHECK-NEXT: %39 = extractelement <2 x float> %34, i32 1
  // CHECK-NEXT: %40 = extractelement <2 x float> %34, i32 1
  // CHECK-NEXT: %41 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %39, float %40)
  // CHECK-NEXT: %42 = insertelement <2 x float> %38, float %41, i32 1
  // CHECK-NEXT: ret <2 x float> %34
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : vector<2xf32>
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf32>
  %f3 = nvvm.addf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : vector<2xf32>
  %f4 = nvvm.addf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : vector<2xf32>
  %f5 = nvvm.addf %f4, %f4 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @addf_vector_f32_f32_rm(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @addf_vector_f32_f32_rm(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.f(float %3, float %4)
  // CHECK-NEXT: %6 = insertelement <2 x float> poison, float %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %9 = call float @llvm.nvvm.add.rm.f(float %7, float %8)
  // CHECK-NEXT: %10 = insertelement <2 x float> %6, float %9, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x float> %10, i32 0
  // CHECK-NEXT: %12 = extractelement <2 x float> %10, i32 0
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rm.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> poison, float %13, i32 0
  // CHECK-NEXT: %15 = extractelement <2 x float> %10, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x float> %10, i32 1
  // CHECK-NEXT: %17 = call float @llvm.nvvm.add.rm.sat.f(float %15, float %16)
  // CHECK-NEXT: %18 = insertelement <2 x float> %14, float %17, i32 1
  // CHECK-NEXT: %19 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %20 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %21 = call float @llvm.nvvm.add.rm.ftz.f(float %19, float %20)
  // CHECK-NEXT: %22 = insertelement <2 x float> poison, float %21, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %24 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %25 = call float @llvm.nvvm.add.rm.ftz.f(float %23, float %24)
  // CHECK-NEXT: %26 = insertelement <2 x float> %22, float %25, i32 1
  // CHECK-NEXT: %27 = extractelement <2 x float> %26, i32 0
  // CHECK-NEXT: %28 = extractelement <2 x float> %26, i32 0
  // CHECK-NEXT: %29 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %27, float %28)
  // CHECK-NEXT: %30 = insertelement <2 x float> poison, float %29, i32 0
  // CHECK-NEXT: %31 = extractelement <2 x float> %26, i32 1
  // CHECK-NEXT: %32 = extractelement <2 x float> %26, i32 1
  // CHECK-NEXT: %33 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %31, float %32)
  // CHECK-NEXT: %34 = insertelement <2 x float> %30, float %33, i32 1
  // CHECK-NEXT: ret <2 x float> %34
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf32>
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : vector<2xf32>
  %f3 = nvvm.addf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : vector<2xf32>
  %f4 = nvvm.addf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @addf_vector_f32_f32_rp(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @addf_vector_f32_f32_rp(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.f(float %3, float %4)
  // CHECK-NEXT: %6 = insertelement <2 x float> poison, float %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %9 = call float @llvm.nvvm.add.rp.f(float %7, float %8)
  // CHECK-NEXT: %10 = insertelement <2 x float> %6, float %9, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x float> %10, i32 0
  // CHECK-NEXT: %12 = extractelement <2 x float> %10, i32 0
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rp.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> poison, float %13, i32 0
  // CHECK-NEXT: %15 = extractelement <2 x float> %10, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x float> %10, i32 1
  // CHECK-NEXT: %17 = call float @llvm.nvvm.add.rp.sat.f(float %15, float %16)
  // CHECK-NEXT: %18 = insertelement <2 x float> %14, float %17, i32 1
  // CHECK-NEXT: %19 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %20 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %21 = call float @llvm.nvvm.add.rp.ftz.f(float %19, float %20)
  // CHECK-NEXT: %22 = insertelement <2 x float> poison, float %21, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %24 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %25 = call float @llvm.nvvm.add.rp.ftz.f(float %23, float %24)
  // CHECK-NEXT: %26 = insertelement <2 x float> %22, float %25, i32 1
  // CHECK-NEXT: %27 = extractelement <2 x float> %26, i32 0
  // CHECK-NEXT: %28 = extractelement <2 x float> %26, i32 0
  // CHECK-NEXT: %29 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %27, float %28)
  // CHECK-NEXT: %30 = insertelement <2 x float> poison, float %29, i32 0
  // CHECK-NEXT: %31 = extractelement <2 x float> %26, i32 1
  // CHECK-NEXT: %32 = extractelement <2 x float> %26, i32 1
  // CHECK-NEXT: %33 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %31, float %32)
  // CHECK-NEXT: %34 = insertelement <2 x float> %30, float %33, i32 1
  // CHECK-NEXT: ret <2 x float> %34
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xf32>
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : vector<2xf32>
  %f3 = nvvm.addf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : vector<2xf32>
  %f4 = nvvm.addf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @addf_vector_f32_f32_rz(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @addf_vector_f32_f32_rz(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.f(float %3, float %4)
  // CHECK-NEXT: %6 = insertelement <2 x float> poison, float %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %9 = call float @llvm.nvvm.add.rz.f(float %7, float %8)
  // CHECK-NEXT: %10 = insertelement <2 x float> %6, float %9, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x float> %10, i32 0
  // CHECK-NEXT: %12 = extractelement <2 x float> %10, i32 0
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rz.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> poison, float %13, i32 0
  // CHECK-NEXT: %15 = extractelement <2 x float> %10, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x float> %10, i32 1
  // CHECK-NEXT: %17 = call float @llvm.nvvm.add.rz.sat.f(float %15, float %16)
  // CHECK-NEXT: %18 = insertelement <2 x float> %14, float %17, i32 1
  // CHECK-NEXT: %19 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %20 = extractelement <2 x float> %18, i32 0
  // CHECK-NEXT: %21 = call float @llvm.nvvm.add.rz.ftz.f(float %19, float %20)
  // CHECK-NEXT: %22 = insertelement <2 x float> poison, float %21, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %24 = extractelement <2 x float> %18, i32 1
  // CHECK-NEXT: %25 = call float @llvm.nvvm.add.rz.ftz.f(float %23, float %24)
  // CHECK-NEXT: %26 = insertelement <2 x float> %22, float %25, i32 1
  // CHECK-NEXT: %27 = extractelement <2 x float> %26, i32 0
  // CHECK-NEXT: %28 = extractelement <2 x float> %26, i32 0
  // CHECK-NEXT: %29 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %27, float %28)
  // CHECK-NEXT: %30 = insertelement <2 x float> poison, float %29, i32 0
  // CHECK-NEXT: %31 = extractelement <2 x float> %26, i32 1
  // CHECK-NEXT: %32 = extractelement <2 x float> %26, i32 1
  // CHECK-NEXT: %33 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %31, float %32)
  // CHECK-NEXT: %34 = insertelement <2 x float> %30, float %33, i32 1
  // CHECK-NEXT: ret <2 x float> %34
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf32>
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : vector<2xf32>
  %f3 = nvvm.addf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : vector<2xf32>
  %f4 = nvvm.addf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

// vector<2xf64> + vector<2xf64> -> vector<2xf64>
llvm.func @addf_vector_f64_f64_rn(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @addf_vector_f64_f64_rn(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rn.d(double %3, double %4)
  // CHECK-NEXT: %6 = insertelement <2 x double> poison, double %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %9 = call double @llvm.nvvm.add.rn.d(double %7, double %8)
  // CHECK-NEXT: %10 = insertelement <2 x double> %6, double %9, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x double> %10, i32 0
  // CHECK-NEXT: %12 = extractelement <2 x double> %10, i32 0
  // CHECK-NEXT: %13 = call double @llvm.nvvm.add.rn.d(double %11, double %12)
  // CHECK-NEXT: %14 = insertelement <2 x double> poison, double %13, i32 0
  // CHECK-NEXT: %15 = extractelement <2 x double> %10, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x double> %10, i32 1
  // CHECK-NEXT: %17 = call double @llvm.nvvm.add.rn.d(double %15, double %16)
  // CHECK-NEXT: %18 = insertelement <2 x double> %14, double %17, i32 1
  // CHECK-NEXT: ret <2 x double> %18
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b : vector<2xf64>
  %f2 = nvvm.addf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf64>
  llvm.return %f2 : vector<2xf64>
}

llvm.func @addf_vector_f64_f64_rm(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @addf_vector_f64_f64_rm(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rm.d(double %3, double %4)
  // CHECK-NEXT: %6 = insertelement <2 x double> poison, double %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %9 = call double @llvm.nvvm.add.rm.d(double %7, double %8)
  // CHECK-NEXT: %10 = insertelement <2 x double> %6, double %9, i32 1
  // CHECK-NEXT: ret <2 x double> %10
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @addf_vector_f64_f64_rp(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @addf_vector_f64_f64_rp(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rp.d(double %3, double %4)
  // CHECK-NEXT: %6 = insertelement <2 x double> poison, double %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %9 = call double @llvm.nvvm.add.rp.d(double %7, double %8)
  // CHECK-NEXT: %10 = insertelement <2 x double> %6, double %9, i32 1
  // CHECK-NEXT: ret <2 x double> %10
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @addf_vector_f64_f64_rz(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @addf_vector_f64_f64_rz(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rz.d(double %3, double %4)
  // CHECK-NEXT: %6 = insertelement <2 x double> poison, double %5, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %8 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %9 = call double @llvm.nvvm.add.rz.d(double %7, double %8)
  // CHECK-NEXT: %10 = insertelement <2 x double> %6, double %9, i32 1
  // CHECK-NEXT: ret <2 x double> %10
  // CHECK-NEXT: }
  %f1 = nvvm.addf %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}
