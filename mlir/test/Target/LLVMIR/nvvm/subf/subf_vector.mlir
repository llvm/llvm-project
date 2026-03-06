// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// vector<2xf16> - vector<2xf16> -> vector<2xf16>
llvm.func @subf_vector_f16_f16(%a : vector<2xf16>, %b : vector<2xf16>) -> vector<2xf16> {
  // CHECK-LABEL: define <2 x half> @subf_vector_f16_f16(<2 x half> %0, <2 x half> %1) {
  // CHECK-NEXT: %3 = fneg <2 x half> %1
  // CHECK-NEXT: %4 = fadd <2 x half> %0, %3
  // CHECK-NEXT: %5 = fneg <2 x half> %4
  // CHECK-NEXT: %6 = fadd <2 x half> %4, %5
  // CHECK-NEXT: %7 = fneg <2 x half> %6
  // CHECK-NEXT: %8 = call <2 x half> @llvm.nvvm.add.rn.sat.v2f16(<2 x half> %6, <2 x half> %7)
  // CHECK-NEXT: %9 = fneg <2 x half> %8
  // CHECK-NEXT: %10 = call <2 x half> @llvm.nvvm.add.rn.ftz.sat.v2f16(<2 x half> %8, <2 x half> %9)
  // CHECK-NEXT: ret <2 x half> %4
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : vector<2xf16>
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf16>
  %f3 = nvvm.subf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>
  %f4 = nvvm.subf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>
  llvm.return %f1 : vector<2xf16>
}

// vector<2xbf16> - vector<2xbf16> -> vector<2xbf16>
llvm.func @subf_vector_bf16_bf16(%a : vector<2xbf16>, %b : vector<2xbf16>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @subf_vector_bf16_bf16(<2 x bfloat> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = fneg <2 x bfloat> %1
  // CHECK-NEXT: %4 = fadd <2 x bfloat> %0, %3
  // CHECK-NEXT: %5 = fneg <2 x bfloat> %4
  // CHECK-NEXT: %6 = fadd <2 x bfloat> %4, %5
  // CHECK-NEXT: ret <2 x bfloat> %6
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : vector<2xbf16>
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xbf16>
  llvm.return %f2 : vector<2xbf16>
}

// vector<2xf32> - vector<2xf32> -> vector<2xf32>
llvm.func @subf_vector_f32_f32_rn(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @subf_vector_f32_f32_rn(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = fneg <2 x float> %1
  // CHECK-NEXT: %4 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.f(float %4, float %5)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %10 = call float @llvm.nvvm.add.rn.f(float %8, float %9)
  // CHECK-NEXT: %11 = insertelement <2 x float> %7, float %10, i32 1
  // CHECK-NEXT: %12 = fneg <2 x float> %11
  // CHECK-NEXT: %13 = extractelement <2 x float> %11, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %15 = call float @llvm.nvvm.add.rn.f(float %13, float %14)
  // CHECK-NEXT: %16 = insertelement <2 x float> poison, float %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %11, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %19 = call float @llvm.nvvm.add.rn.f(float %17, float %18)
  // CHECK-NEXT: %20 = insertelement <2 x float> %16, float %19, i32 1
  // CHECK-NEXT: %21 = fneg <2 x float> %20
  // CHECK-NEXT: %22 = extractelement <2 x float> %20, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %21, i32 0
  // CHECK-NEXT: %24 = call float @llvm.nvvm.add.rn.sat.f(float %22, float %23)
  // CHECK-NEXT: %25 = insertelement <2 x float> poison, float %24, i32 0
  // CHECK-NEXT: %26 = extractelement <2 x float> %20, i32 1
  // CHECK-NEXT: %27 = extractelement <2 x float> %21, i32 1
  // CHECK-NEXT: %28 = call float @llvm.nvvm.add.rn.sat.f(float %26, float %27)
  // CHECK-NEXT: %29 = insertelement <2 x float> %25, float %28, i32 1
  // CHECK-NEXT: %30 = fneg <2 x float> %29
  // CHECK-NEXT: %31 = extractelement <2 x float> %29, i32 0
  // CHECK-NEXT: %32 = extractelement <2 x float> %30, i32 0
  // CHECK-NEXT: %33 = call float @llvm.nvvm.add.rn.ftz.f(float %31, float %32)
  // CHECK-NEXT: %34 = insertelement <2 x float> poison, float %33, i32 0
  // CHECK-NEXT: %35 = extractelement <2 x float> %29, i32 1
  // CHECK-NEXT: %36 = extractelement <2 x float> %30, i32 1
  // CHECK-NEXT: %37 = call float @llvm.nvvm.add.rn.ftz.f(float %35, float %36)
  // CHECK-NEXT: %38 = insertelement <2 x float> %34, float %37, i32 1
  // CHECK-NEXT: %39 = fneg <2 x float> %38
  // CHECK-NEXT: %40 = extractelement <2 x float> %38, i32 0
  // CHECK-NEXT: %41 = extractelement <2 x float> %39, i32 0
  // CHECK-NEXT: %42 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %40, float %41)
  // CHECK-NEXT: %43 = insertelement <2 x float> poison, float %42, i32 0
  // CHECK-NEXT: %44 = extractelement <2 x float> %38, i32 1
  // CHECK-NEXT: %45 = extractelement <2 x float> %39, i32 1
  // CHECK-NEXT: %46 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %44, float %45)
  // CHECK-NEXT: %47 = insertelement <2 x float> %43, float %46, i32 1
  // CHECK-NEXT: ret <2 x float> %38
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : vector<2xf32>
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf32>
  %f3 = nvvm.subf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : vector<2xf32>
  %f4 = nvvm.subf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : vector<2xf32>
  %f5 = nvvm.subf %f4, %f4 {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @subf_vector_f32_f32_rm(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @subf_vector_f32_f32_rm(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = fneg <2 x float> %1
  // CHECK-NEXT: %4 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.f(float %4, float %5)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %10 = call float @llvm.nvvm.add.rm.f(float %8, float %9)
  // CHECK-NEXT: %11 = insertelement <2 x float> %7, float %10, i32 1
  // CHECK-NEXT: %12 = fneg <2 x float> %11
  // CHECK-NEXT: %13 = extractelement <2 x float> %11, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %15 = call float @llvm.nvvm.add.rm.sat.f(float %13, float %14)
  // CHECK-NEXT: %16 = insertelement <2 x float> poison, float %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %11, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %19 = call float @llvm.nvvm.add.rm.sat.f(float %17, float %18)
  // CHECK-NEXT: %20 = insertelement <2 x float> %16, float %19, i32 1
  // CHECK-NEXT: %21 = fneg <2 x float> %20
  // CHECK-NEXT: %22 = extractelement <2 x float> %20, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %21, i32 0
  // CHECK-NEXT: %24 = call float @llvm.nvvm.add.rm.ftz.f(float %22, float %23)
  // CHECK-NEXT: %25 = insertelement <2 x float> poison, float %24, i32 0
  // CHECK-NEXT: %26 = extractelement <2 x float> %20, i32 1
  // CHECK-NEXT: %27 = extractelement <2 x float> %21, i32 1
  // CHECK-NEXT: %28 = call float @llvm.nvvm.add.rm.ftz.f(float %26, float %27)
  // CHECK-NEXT: %29 = insertelement <2 x float> %25, float %28, i32 1
  // CHECK-NEXT: %30 = fneg <2 x float> %29
  // CHECK-NEXT: %31 = extractelement <2 x float> %29, i32 0
  // CHECK-NEXT: %32 = extractelement <2 x float> %30, i32 0
  // CHECK-NEXT: %33 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %31, float %32)
  // CHECK-NEXT: %34 = insertelement <2 x float> poison, float %33, i32 0
  // CHECK-NEXT: %35 = extractelement <2 x float> %29, i32 1
  // CHECK-NEXT: %36 = extractelement <2 x float> %30, i32 1
  // CHECK-NEXT: %37 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %35, float %36)
  // CHECK-NEXT: %38 = insertelement <2 x float> %34, float %37, i32 1
  // CHECK-NEXT: ret <2 x float> %38
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf32>
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : vector<2xf32>
  %f3 = nvvm.subf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : vector<2xf32>
  %f4 = nvvm.subf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @subf_vector_f32_f32_rp(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @subf_vector_f32_f32_rp(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = fneg <2 x float> %1
  // CHECK-NEXT: %4 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.f(float %4, float %5)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %10 = call float @llvm.nvvm.add.rp.f(float %8, float %9)
  // CHECK-NEXT: %11 = insertelement <2 x float> %7, float %10, i32 1
  // CHECK-NEXT: %12 = fneg <2 x float> %11
  // CHECK-NEXT: %13 = extractelement <2 x float> %11, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %15 = call float @llvm.nvvm.add.rp.sat.f(float %13, float %14)
  // CHECK-NEXT: %16 = insertelement <2 x float> poison, float %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %11, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %19 = call float @llvm.nvvm.add.rp.sat.f(float %17, float %18)
  // CHECK-NEXT: %20 = insertelement <2 x float> %16, float %19, i32 1
  // CHECK-NEXT: %21 = fneg <2 x float> %20
  // CHECK-NEXT: %22 = extractelement <2 x float> %20, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %21, i32 0
  // CHECK-NEXT: %24 = call float @llvm.nvvm.add.rp.ftz.f(float %22, float %23)
  // CHECK-NEXT: %25 = insertelement <2 x float> poison, float %24, i32 0
  // CHECK-NEXT: %26 = extractelement <2 x float> %20, i32 1
  // CHECK-NEXT: %27 = extractelement <2 x float> %21, i32 1
  // CHECK-NEXT: %28 = call float @llvm.nvvm.add.rp.ftz.f(float %26, float %27)
  // CHECK-NEXT: %29 = insertelement <2 x float> %25, float %28, i32 1
  // CHECK-NEXT: %30 = fneg <2 x float> %29
  // CHECK-NEXT: %31 = extractelement <2 x float> %29, i32 0
  // CHECK-NEXT: %32 = extractelement <2 x float> %30, i32 0
  // CHECK-NEXT: %33 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %31, float %32)
  // CHECK-NEXT: %34 = insertelement <2 x float> poison, float %33, i32 0
  // CHECK-NEXT: %35 = extractelement <2 x float> %29, i32 1
  // CHECK-NEXT: %36 = extractelement <2 x float> %30, i32 1
  // CHECK-NEXT: %37 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %35, float %36)
  // CHECK-NEXT: %38 = insertelement <2 x float> %34, float %37, i32 1
  // CHECK-NEXT: ret <2 x float> %38
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xf32>
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : vector<2xf32>
  %f3 = nvvm.subf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : vector<2xf32>
  %f4 = nvvm.subf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

llvm.func @subf_vector_f32_f32_rz(%a : vector<2xf32>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @subf_vector_f32_f32_rz(<2 x float> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = fneg <2 x float> %1
  // CHECK-NEXT: %4 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x float> %3, i32 0
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.f(float %4, float %5)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %3, i32 1
  // CHECK-NEXT: %10 = call float @llvm.nvvm.add.rz.f(float %8, float %9)
  // CHECK-NEXT: %11 = insertelement <2 x float> %7, float %10, i32 1
  // CHECK-NEXT: %12 = fneg <2 x float> %11
  // CHECK-NEXT: %13 = extractelement <2 x float> %11, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x float> %12, i32 0
  // CHECK-NEXT: %15 = call float @llvm.nvvm.add.rz.sat.f(float %13, float %14)
  // CHECK-NEXT: %16 = insertelement <2 x float> poison, float %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %11, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x float> %12, i32 1
  // CHECK-NEXT: %19 = call float @llvm.nvvm.add.rz.sat.f(float %17, float %18)
  // CHECK-NEXT: %20 = insertelement <2 x float> %16, float %19, i32 1
  // CHECK-NEXT: %21 = fneg <2 x float> %20
  // CHECK-NEXT: %22 = extractelement <2 x float> %20, i32 0
  // CHECK-NEXT: %23 = extractelement <2 x float> %21, i32 0
  // CHECK-NEXT: %24 = call float @llvm.nvvm.add.rz.ftz.f(float %22, float %23)
  // CHECK-NEXT: %25 = insertelement <2 x float> poison, float %24, i32 0
  // CHECK-NEXT: %26 = extractelement <2 x float> %20, i32 1
  // CHECK-NEXT: %27 = extractelement <2 x float> %21, i32 1
  // CHECK-NEXT: %28 = call float @llvm.nvvm.add.rz.ftz.f(float %26, float %27)
  // CHECK-NEXT: %29 = insertelement <2 x float> %25, float %28, i32 1
  // CHECK-NEXT: %30 = fneg <2 x float> %29
  // CHECK-NEXT: %31 = extractelement <2 x float> %29, i32 0
  // CHECK-NEXT: %32 = extractelement <2 x float> %30, i32 0
  // CHECK-NEXT: %33 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %31, float %32)
  // CHECK-NEXT: %34 = insertelement <2 x float> poison, float %33, i32 0
  // CHECK-NEXT: %35 = extractelement <2 x float> %29, i32 1
  // CHECK-NEXT: %36 = extractelement <2 x float> %30, i32 1
  // CHECK-NEXT: %37 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %35, float %36)
  // CHECK-NEXT: %38 = insertelement <2 x float> %34, float %37, i32 1
  // CHECK-NEXT: ret <2 x float> %38
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf32>
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : vector<2xf32>
  %f3 = nvvm.subf %f2, %f2 {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : vector<2xf32>
  %f4 = nvvm.subf %f3, %f3 {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf32>
  llvm.return %f4 : vector<2xf32>
}

// vector<2xf64> - vector<2xf64> -> vector<2xf64>
llvm.func @subf_vector_f64_f64_rn(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @subf_vector_f64_f64_rn(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = fneg <2 x double> %1
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %3, i32 0
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rn.d(double %4, double %5)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %3, i32 1
  // CHECK-NEXT: %10 = call double @llvm.nvvm.add.rn.d(double %8, double %9)
  // CHECK-NEXT: %11 = insertelement <2 x double> %7, double %10, i32 1
  // CHECK-NEXT: %12 = fneg <2 x double> %11
  // CHECK-NEXT: %13 = extractelement <2 x double> %11, i32 0
  // CHECK-NEXT: %14 = extractelement <2 x double> %12, i32 0
  // CHECK-NEXT: %15 = call double @llvm.nvvm.add.rn.d(double %13, double %14)
  // CHECK-NEXT: %16 = insertelement <2 x double> poison, double %15, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x double> %11, i32 1
  // CHECK-NEXT: %18 = extractelement <2 x double> %12, i32 1
  // CHECK-NEXT: %19 = call double @llvm.nvvm.add.rn.d(double %17, double %18)
  // CHECK-NEXT: %20 = insertelement <2 x double> %16, double %19, i32 1
  // CHECK-NEXT: ret <2 x double> %20
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b : vector<2xf64>
  %f2 = nvvm.subf %f1, %f1 {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf64>
  llvm.return %f2 : vector<2xf64>
}

llvm.func @subf_vector_f64_f64_rm(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @subf_vector_f64_f64_rm(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = fneg <2 x double> %1
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %3, i32 0
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rm.d(double %4, double %5)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %3, i32 1
  // CHECK-NEXT: %10 = call double @llvm.nvvm.add.rm.d(double %8, double %9)
  // CHECK-NEXT: %11 = insertelement <2 x double> %7, double %10, i32 1
  // CHECK-NEXT: ret <2 x double> %11
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @subf_vector_f64_f64_rp(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @subf_vector_f64_f64_rp(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = fneg <2 x double> %1
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %3, i32 0
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rp.d(double %4, double %5)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %3, i32 1
  // CHECK-NEXT: %10 = call double @llvm.nvvm.add.rp.d(double %8, double %9)
  // CHECK-NEXT: %11 = insertelement <2 x double> %7, double %10, i32 1
  // CHECK-NEXT: ret <2 x double> %11
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @subf_vector_f64_f64_rz(%a : vector<2xf64>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @subf_vector_f64_f64_rz(<2 x double> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = fneg <2 x double> %1
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %3, i32 0
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rz.d(double %4, double %5)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %3, i32 1
  // CHECK-NEXT: %10 = call double @llvm.nvvm.add.rz.d(double %8, double %9)
  // CHECK-NEXT: %11 = insertelement <2 x double> %7, double %10, i32 1
  // CHECK-NEXT: ret <2 x double> %11
  // CHECK-NEXT: }
  %f1 = nvvm.subf %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}
