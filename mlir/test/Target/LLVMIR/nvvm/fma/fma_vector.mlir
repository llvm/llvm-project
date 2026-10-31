// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @fma_f16(%a: vector<2xf16>, %b: vector<2xf16>, %c: vector<2xf16>) -> vector<2xf16> {
  // CHECK-LABEL: define <2 x half> @fma_f16(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  // CHECK-NEXT: %4 = call <2 x half> @llvm.nvvm.fma.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = call <2 x half> @llvm.nvvm.fma.ftz.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %4, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %6 = call <2 x half> @llvm.nvvm.fma.sat.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %5, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %7 = call <2 x half> @llvm.nvvm.fma.ftz.sat.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %6, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %8 = call <2 x half> @llvm.nvvm.fma.relu.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %7, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %9 = call <2 x half> @llvm.nvvm.fma.ftz.relu.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %8, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %10 = call <2 x half> @llvm.nvvm.fma.rn.oob.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %9)
  // CHECK-NEXT: %11 = call <2 x half> @llvm.nvvm.fma.rn.oob.relu.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %10)
  // CHECK-NEXT: ret <2 x half> %11
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rn> : vector<2xf16>
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rn> ftz = true : vector<2xf16>
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rn> sat = <sat> : vector<2xf16>
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rn> sat = <sat> ftz = true : vector<2xf16>
  %f4 = nvvm.fma %a, %b, %f3, rnd = <rn> relu = true : vector<2xf16>
  %f5 = nvvm.fma %a, %b, %f4, rnd = <rn> ftz = true relu = true : vector<2xf16>
  %f6 = nvvm.fma %a, %b, %f5, rnd = <rn> oob = true : vector<2xf16>
  %f7 = nvvm.fma %a, %b, %f6, rnd = <rn> relu = true oob = true : vector<2xf16>
  llvm.return %f7 : vector<2xf16>
}

llvm.func @fma_bf16(%a: vector<2xbf16>, %b: vector<2xbf16>, %c: vector<2xbf16>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @fma_bf16(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %2) {
  // CHECK-NEXT: %4 = call <2 x bfloat> @llvm.nvvm.fma.v2bf16(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %2, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = call <2 x bfloat> @llvm.nvvm.fma.relu.v2bf16(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %4, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %6 = call <2 x bfloat> @llvm.nvvm.fma.rn.oob.v2bf16(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %5)
  // CHECK-NEXT: %7 = call <2 x bfloat> @llvm.nvvm.fma.rn.oob.relu.v2bf16(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %6)
  // CHECK-NEXT: ret <2 x bfloat> %7
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rn> : vector<2xbf16>
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rn> relu = true : vector<2xbf16>
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rn> oob = true : vector<2xbf16>
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rn> relu = true oob = true : vector<2xbf16>
  llvm.return %f3 : vector<2xbf16>
}

llvm.func @fma_f32_rn(%a: vector<2xf32>, %b: vector<2xf32>, %c: vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fma_f32_rn(<2 x float> %0, <2 x float> %1, <2 x float> %2) {
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fma.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %5 = call <2 x float> @llvm.nvvm.fma.ftz.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %4, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %6 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %5, i32 0
  // CHECK-NEXT: %9 = call float @llvm.nvvm.fma.sat.f32(float %6, float %7, float %8, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %10 = insertelement <2 x float> poison, float %9, i32 0
  // CHECK-NEXT: %11 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %12 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %13 = extractelement <2 x float> %5, i32 1
  // CHECK-NEXT: %14 = call float @llvm.nvvm.fma.sat.f32(float %11, float %12, float %13, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %15 = insertelement <2 x float> %10, float %14, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %18 = extractelement <2 x float> %15, i32 0
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fma.ftz.sat.f32(float %16, float %17, float %18, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %20 = insertelement <2 x float> poison, float %19, i32 0
  // CHECK-NEXT: %21 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %22 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %23 = extractelement <2 x float> %15, i32 1
  // CHECK-NEXT: %24 = call float @llvm.nvvm.fma.ftz.sat.f32(float %21, float %22, float %23, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %25 = insertelement <2 x float> %20, float %24, i32 1
  // CHECK-NEXT: ret <2 x float> %25
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rn> : vector<2xf32>
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rn> ftz = true : vector<2xf32>
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rn> sat = <sat> : vector<2xf32>
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rn> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f3 : vector<2xf32>
}

llvm.func @fma_f32_rm(%a: vector<2xf32>, %b: vector<2xf32>, %c: vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fma_f32_rm(<2 x float> %0, <2 x float> %1, <2 x float> %2) {
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fma.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %5 = call <2 x float> @llvm.nvvm.fma.ftz.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %4, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %6 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %5, i32 0
  // CHECK-NEXT: %9 = call float @llvm.nvvm.fma.sat.f32(float %6, float %7, float %8, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %10 = insertelement <2 x float> poison, float %9, i32 0
  // CHECK-NEXT: %11 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %12 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %13 = extractelement <2 x float> %5, i32 1
  // CHECK-NEXT: %14 = call float @llvm.nvvm.fma.sat.f32(float %11, float %12, float %13, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %15 = insertelement <2 x float> %10, float %14, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %18 = extractelement <2 x float> %15, i32 0
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fma.ftz.sat.f32(float %16, float %17, float %18, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %20 = insertelement <2 x float> poison, float %19, i32 0
  // CHECK-NEXT: %21 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %22 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %23 = extractelement <2 x float> %15, i32 1
  // CHECK-NEXT: %24 = call float @llvm.nvvm.fma.ftz.sat.f32(float %21, float %22, float %23, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %25 = insertelement <2 x float> %20, float %24, i32 1
  // CHECK-NEXT: ret <2 x float> %25
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rm> : vector<2xf32>
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rm> ftz = true : vector<2xf32>
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rm> sat = <sat> : vector<2xf32>
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rm> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f3 : vector<2xf32>
}

llvm.func @fma_f32_rp(%a: vector<2xf32>, %b: vector<2xf32>, %c: vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fma_f32_rp(<2 x float> %0, <2 x float> %1, <2 x float> %2) {
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fma.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %5 = call <2 x float> @llvm.nvvm.fma.ftz.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %4, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %6 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %5, i32 0
  // CHECK-NEXT: %9 = call float @llvm.nvvm.fma.sat.f32(float %6, float %7, float %8, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %10 = insertelement <2 x float> poison, float %9, i32 0
  // CHECK-NEXT: %11 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %12 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %13 = extractelement <2 x float> %5, i32 1
  // CHECK-NEXT: %14 = call float @llvm.nvvm.fma.sat.f32(float %11, float %12, float %13, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %15 = insertelement <2 x float> %10, float %14, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %18 = extractelement <2 x float> %15, i32 0
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fma.ftz.sat.f32(float %16, float %17, float %18, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %20 = insertelement <2 x float> poison, float %19, i32 0
  // CHECK-NEXT: %21 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %22 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %23 = extractelement <2 x float> %15, i32 1
  // CHECK-NEXT: %24 = call float @llvm.nvvm.fma.ftz.sat.f32(float %21, float %22, float %23, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %25 = insertelement <2 x float> %20, float %24, i32 1
  // CHECK-NEXT: ret <2 x float> %25
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rp> : vector<2xf32>
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rp> ftz = true : vector<2xf32>
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rp> sat = <sat> : vector<2xf32>
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rp> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f3 : vector<2xf32>
}

llvm.func @fma_f32_rz(%a: vector<2xf32>, %b: vector<2xf32>, %c: vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fma_f32_rz(<2 x float> %0, <2 x float> %1, <2 x float> %2) {
  // CHECK-NEXT: %4 = call <2 x float> @llvm.nvvm.fma.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %2, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %5 = call <2 x float> @llvm.nvvm.fma.ftz.v2f32(<2 x float> %0, <2 x float> %1, <2 x float> %4, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %6 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %7 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %5, i32 0
  // CHECK-NEXT: %9 = call float @llvm.nvvm.fma.sat.f32(float %6, float %7, float %8, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %10 = insertelement <2 x float> poison, float %9, i32 0
  // CHECK-NEXT: %11 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %12 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %13 = extractelement <2 x float> %5, i32 1
  // CHECK-NEXT: %14 = call float @llvm.nvvm.fma.sat.f32(float %11, float %12, float %13, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %15 = insertelement <2 x float> %10, float %14, i32 1
  // CHECK-NEXT: %16 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %17 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %18 = extractelement <2 x float> %15, i32 0
  // CHECK-NEXT: %19 = call float @llvm.nvvm.fma.ftz.sat.f32(float %16, float %17, float %18, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %20 = insertelement <2 x float> poison, float %19, i32 0
  // CHECK-NEXT: %21 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %22 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %23 = extractelement <2 x float> %15, i32 1
  // CHECK-NEXT: %24 = call float @llvm.nvvm.fma.ftz.sat.f32(float %21, float %22, float %23, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %25 = insertelement <2 x float> %20, float %24, i32 1
  // CHECK-NEXT: ret <2 x float> %25
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rz> : vector<2xf32>
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rz> ftz = true : vector<2xf32>
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rz> sat = <sat> : vector<2xf32>
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rz> sat = <sat> ftz = true : vector<2xf32>
  llvm.return %f3 : vector<2xf32>
}

llvm.func @fma_f64(%a: vector<2xf64>, %b: vector<2xf64>, %c: vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fma_f64(<2 x double> %0, <2 x double> %1, <2 x double> %2) {
  // CHECK-NEXT: %4 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %5 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %6 = extractelement <2 x double> %2, i32 0
  // CHECK-NEXT: %7 = call double @llvm.nvvm.fma.f64(double %4, double %5, double %6, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %8 = insertelement <2 x double> poison, double %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %11 = extractelement <2 x double> %2, i32 1
  // CHECK-NEXT: %12 = call double @llvm.nvvm.fma.f64(double %9, double %10, double %11, /* rnd=rn */ i32 1)
  // CHECK-NEXT: %13 = insertelement <2 x double> %8, double %12, i32 1
  // CHECK-NEXT: %14 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %15 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %16 = extractelement <2 x double> %13, i32 0
  // CHECK-NEXT: %17 = call double @llvm.nvvm.fma.f64(double %14, double %15, double %16, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %18 = insertelement <2 x double> poison, double %17, i32 0
  // CHECK-NEXT: %19 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %20 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %21 = extractelement <2 x double> %13, i32 1
  // CHECK-NEXT: %22 = call double @llvm.nvvm.fma.f64(double %19, double %20, double %21, /* rnd=rm */ i32 3)
  // CHECK-NEXT: %23 = insertelement <2 x double> %18, double %22, i32 1
  // CHECK-NEXT: %24 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %25 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %26 = extractelement <2 x double> %23, i32 0
  // CHECK-NEXT: %27 = call double @llvm.nvvm.fma.f64(double %24, double %25, double %26, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %28 = insertelement <2 x double> poison, double %27, i32 0
  // CHECK-NEXT: %29 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %30 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %31 = extractelement <2 x double> %23, i32 1
  // CHECK-NEXT: %32 = call double @llvm.nvvm.fma.f64(double %29, double %30, double %31, /* rnd=rp */ i32 2)
  // CHECK-NEXT: %33 = insertelement <2 x double> %28, double %32, i32 1
  // CHECK-NEXT: %34 = extractelement <2 x double> %0, i32 0
  // CHECK-NEXT: %35 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %36 = extractelement <2 x double> %33, i32 0
  // CHECK-NEXT: %37 = call double @llvm.nvvm.fma.f64(double %34, double %35, double %36, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %38 = insertelement <2 x double> poison, double %37, i32 0
  // CHECK-NEXT: %39 = extractelement <2 x double> %0, i32 1
  // CHECK-NEXT: %40 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %41 = extractelement <2 x double> %33, i32 1
  // CHECK-NEXT: %42 = call double @llvm.nvvm.fma.f64(double %39, double %40, double %41, /* rnd=rz */ i32 0)
  // CHECK-NEXT: %43 = insertelement <2 x double> %38, double %42, i32 1
  // CHECK-NEXT: ret <2 x double> %43
  // CHECK-NEXT: }
  %f0 = nvvm.fma %a, %b, %c, rnd = <rn> : vector<2xf64>
  %f1 = nvvm.fma %a, %b, %f0, rnd = <rm> : vector<2xf64>
  %f2 = nvvm.fma %a, %b, %f1, rnd = <rp> : vector<2xf64>
  %f3 = nvvm.fma %a, %b, %f2, rnd = <rz> : vector<2xf64>
  llvm.return %f3 : vector<2xf64>
}
