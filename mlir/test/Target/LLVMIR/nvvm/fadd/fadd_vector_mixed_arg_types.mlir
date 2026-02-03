// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// vector<2xf16> + vector<2xbf16> -> vector<2xf32>
llvm.func @fadd_vector_f16_bf16_f32(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rn.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rn.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rn(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rn(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rn.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rn.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rn_sat(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rn_sat(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rn.sat.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rn.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rn_ftz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rn_ftz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rn.ftz.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rn.ftz.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rn_sat_ftz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rn_sat_ftz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rm(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rm(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rm.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rm.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rm_sat(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rm_sat(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rm.sat.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rm.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rm_ftz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rm_ftz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rm.ftz.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rm.ftz.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rm_sat_ftz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rm_sat_ftz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rp(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rp(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rp.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rp.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rp_sat(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rp_sat(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rp.sat.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rp.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rp_ftz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rp_ftz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rp.ftz.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rp.ftz.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rp_sat_ftz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rp_sat_ftz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rz.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rz.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rz_sat(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rz_sat(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rz.sat.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rz.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rz_ftz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rz_ftz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rz.ftz.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rz.ftz.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_bf16_f32_rz_sat_ftz(%a : vector<2xf16>, %b : vector<2xbf16>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_bf16_f32_rz_sat_ftz(<2 x half> %0, <2 x bfloat> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x bfloat> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = fpext bfloat %4 to float
  // CHECK-NEXT: %7 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %5, float %6)
  // CHECK-NEXT: %8 = insertelement <2 x float> poison, float %7, i32 0
  // CHECK-NEXT: %9 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %10 = extractelement <2 x bfloat> %1, i32 1
  // CHECK-NEXT: %11 = fpext half %9 to float
  // CHECK-NEXT: %12 = fpext bfloat %10 to float
  // CHECK-NEXT: %13 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %11, float %12)
  // CHECK-NEXT: %14 = insertelement <2 x float> %8, float %13, i32 1
  // CHECK-NEXT: ret <2 x float> %14
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>, vector<2xbf16> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

// vector<2xf16> + vector<2xf32> -> vector<2xf32>
llvm.func @fadd_vector_f16_f32_f32(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rn(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rn(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rn_sat(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rn_sat(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rn_ftz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rn_ftz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.ftz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.ftz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rn_sat_ftz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rn_sat_ftz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rm(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rm(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rm_sat(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rm_sat(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rm_ftz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rm_ftz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.ftz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.ftz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rm_sat_ftz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rm_sat_ftz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rp(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rp(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rp.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rp_sat(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rp_sat(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rp.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rp_ftz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rp_ftz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.ftz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rp.ftz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rp_sat_ftz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rp_sat_ftz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rz_sat(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rz_sat(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rz_ftz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rz_ftz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.ftz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rz.ftz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_f16_f32_f32_rz_sat_ftz(%a : vector<2xf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_f16_f32_f32_rz_sat_ftz(<2 x half> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

// vector<2xf16> + vector<2xf64> -> vector<2xf64>
llvm.func @fadd_vector_f16_f64_f64(%a : vector<2xf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f16_f64_f64(<2 x half> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rn.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rn.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b : vector<2xf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_f16_f64_f64_rn(%a : vector<2xf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f16_f64_f64_rn(<2 x half> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rn.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rn.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_f16_f64_f64_rm(%a : vector<2xf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f16_f64_f64_rm(<2 x half> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rm.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rm.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_f16_f64_f64_rp(%a : vector<2xf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f16_f64_f64_rp(<2 x half> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rp.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rp.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_f16_f64_f64_rz(%a : vector<2xf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f16_f64_f64_rz(<2 x half> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x half> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext half %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rz.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x half> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext half %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rz.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

// vector<2xbf16> + vector<2xf32> -> vector<2xf32>
llvm.func @fadd_vector_bf16_f32_f32(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rn(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rn(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rn_sat(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rn_sat(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rn_ftz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rn_ftz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.ftz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.ftz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rn_sat_ftz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rn_sat_ftz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rm(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rm(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rm_sat(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rm_sat(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rm_ftz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rm_ftz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.ftz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.ftz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rm_sat_ftz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rm_sat_ftz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rp(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rp(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rp.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rp_sat(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rp_sat(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rp.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rp_ftz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rp_ftz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.ftz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rp.ftz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rp_sat_ftz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rp_sat_ftz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rz_sat(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rz_sat(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rz_ftz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rz_ftz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.ftz.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rz.ftz.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

llvm.func @fadd_vector_bf16_f32_f32_rz_sat_ftz(%a : vector<2xbf16>, %b : vector<2xf32>) -> vector<2xf32> {
  // CHECK-LABEL: define <2 x float> @fadd_vector_bf16_f32_f32_rz_sat_ftz(<2 x bfloat> %0, <2 x float> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x float> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to float
  // CHECK-NEXT: %6 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %5, float %4)
  // CHECK-NEXT: %7 = insertelement <2 x float> poison, float %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x float> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to float
  // CHECK-NEXT: %11 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %10, float %9)
  // CHECK-NEXT: %12 = insertelement <2 x float> %7, float %11, i32 1
  // CHECK-NEXT: ret <2 x float> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
  llvm.return %f1 : vector<2xf32>
}

// vector<2xbf16> + vector<2xf64> -> vector<2xf64>
llvm.func @fadd_vector_bf16_f64_f64(%a : vector<2xbf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_bf16_f64_f64(<2 x bfloat> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rn.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rn.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b : vector<2xbf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_bf16_f64_f64_rn(%a : vector<2xbf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_bf16_f64_f64_rn(<2 x bfloat> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rn.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rn.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xbf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_bf16_f64_f64_rm(%a : vector<2xbf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_bf16_f64_f64_rm(<2 x bfloat> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rm.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rm.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xbf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_bf16_f64_f64_rp(%a : vector<2xbf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_bf16_f64_f64_rp(<2 x bfloat> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rp.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rp.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xbf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_bf16_f64_f64_rz(%a : vector<2xbf16>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_bf16_f64_f64_rz(<2 x bfloat> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x bfloat> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext bfloat %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rz.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x bfloat> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext bfloat %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rz.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

// vector<2xf32> + vector<2xf64> -> vector<2xf64>
llvm.func @fadd_vector_f32_f64_f64(%a : vector<2xf32>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f32_f64_f64(<2 x float> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext float %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rn.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext float %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rn.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b : vector<2xf32>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_f32_f64_f64_rn(%a : vector<2xf32>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f32_f64_f64_rn(<2 x float> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext float %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rn.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext float %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rn.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf32>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_f32_f64_f64_rm(%a : vector<2xf32>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f32_f64_f64_rm(<2 x float> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext float %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rm.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext float %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rm.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf32>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_f32_f64_f64_rp(%a : vector<2xf32>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f32_f64_f64_rp(<2 x float> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext float %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rp.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext float %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rp.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xf32>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}

llvm.func @fadd_vector_f32_f64_f64_rz(%a : vector<2xf32>, %b : vector<2xf64>) -> vector<2xf64> {
  // CHECK-LABEL: define <2 x double> @fadd_vector_f32_f64_f64_rz(<2 x float> %0, <2 x double> %1) {
  // CHECK-NEXT: %3 = extractelement <2 x float> %0, i32 0
  // CHECK-NEXT: %4 = extractelement <2 x double> %1, i32 0
  // CHECK-NEXT: %5 = fpext float %3 to double
  // CHECK-NEXT: %6 = call double @llvm.nvvm.add.rz.d(double %5, double %4)
  // CHECK-NEXT: %7 = insertelement <2 x double> poison, double %6, i32 0
  // CHECK-NEXT: %8 = extractelement <2 x float> %0, i32 1
  // CHECK-NEXT: %9 = extractelement <2 x double> %1, i32 1
  // CHECK-NEXT: %10 = fpext float %8 to double
  // CHECK-NEXT: %11 = call double @llvm.nvvm.add.rz.d(double %10, double %9)
  // CHECK-NEXT: %12 = insertelement <2 x double> %7, double %11, i32 1
  // CHECK-NEXT: ret <2 x double> %12
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf32>, vector<2xf64> -> vector<2xf64>
  llvm.return %f1 : vector<2xf64>
}
