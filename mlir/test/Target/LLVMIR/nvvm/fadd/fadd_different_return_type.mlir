// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// f16 + f16 -> f32
llvm.func @fadd_f16_f16_rn(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rn(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rn_sat(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rn_sat(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rn_ftz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rn_ftz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.ftz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rn_sat_ftz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rn_sat_ftz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rm(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rm(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rm_sat(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rm_sat(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rm_ftz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rm_ftz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.ftz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rm_sat_ftz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rm_sat_ftz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rp(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rp(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rp_sat(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rp_sat(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rp_ftz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rp_ftz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.ftz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rp_sat_ftz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rp_sat_ftz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rz_sat(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rz_sat(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rz_ftz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rz_ftz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.ftz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : f16, f16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_f16_f16_rz_sat_ftz(%a : f16, %b : f16) -> f32 {
  // CHECK-LABEL: define float @fadd_f16_f16_rz_sat_ftz(half %0, half %1) {
  // CHECK-NEXT: %3 = fpext half %0 to float
  // CHECK-NEXT: %4 = fpext half %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : f16, f16 -> f32
  llvm.return %f1 : f32
}

// bf16 + bf16 -> f32
llvm.func @fadd_bf16_bf16_rn(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rn(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rn_sat(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rn_sat(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rn_ftz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rn_ftz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.ftz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, ftz=true} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rn_sat_ftz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rn_sat_ftz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rn.ftz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<sat>, ftz=true} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rm(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rm(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rm_sat(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rm_sat(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rm_ftz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rm_ftz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.ftz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, ftz=true} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rm_sat_ftz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rm_sat_ftz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rm.ftz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>, sat = #nvvm.sat_mode<sat>, ftz=true} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rp(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rp(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rp_sat(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rp_sat(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rp_ftz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rp_ftz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.ftz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, ftz=true} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rp_sat_ftz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rp_sat_ftz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rp.ftz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<sat>, ftz=true} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rz_sat(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rz_sat(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rz_ftz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rz_ftz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.ftz.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, ftz=true} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

llvm.func @fadd_bf16_bf16_rz_sat_ftz(%a : bf16, %b : bf16) -> f32 {
  // CHECK-LABEL: define float @fadd_bf16_bf16_rz_sat_ftz(bfloat %0, bfloat %1) {
  // CHECK-NEXT: %3 = fpext bfloat %0 to float
  // CHECK-NEXT: %4 = fpext bfloat %1 to float
  // CHECK-NEXT: %5 = call float @llvm.nvvm.add.rz.ftz.sat.f(float %3, float %4)
  // CHECK-NEXT: ret float %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<sat>, ftz=true} : bf16, bf16 -> f32
  llvm.return %f1 : f32
}

// f32 + f32 -> f64
llvm.func @fadd_f32_f32_rn(%a : f32, %b : f32) -> f64 {
  // CHECK-LABEL: define double @fadd_f32_f32_rn(float %0, float %1) {
  // CHECK-NEXT: %3 = fpext float %0 to double
  // CHECK-NEXT: %4 = fpext float %1 to double
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rn.d(double %3, double %4)
  // CHECK-NEXT: ret double %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : f32, f32 -> f64
  llvm.return %f1 : f64
}

llvm.func @fadd_f32_f32_rm(%a : f32, %b : f32) -> f64 {
  // CHECK-LABEL: define double @fadd_f32_f32_rm(float %0, float %1) {
  // CHECK-NEXT: %3 = fpext float %0 to double
  // CHECK-NEXT: %4 = fpext float %1 to double
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rm.d(double %3, double %4)
  // CHECK-NEXT: ret double %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rm>} : f32, f32 -> f64
  llvm.return %f1 : f64
}

llvm.func @fadd_f32_f32_rp(%a : f32, %b : f32) -> f64 {
  // CHECK-LABEL: define double @fadd_f32_f32_rp(float %0, float %1) {
  // CHECK-NEXT: %3 = fpext float %0 to double
  // CHECK-NEXT: %4 = fpext float %1 to double
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rp.d(double %3, double %4)
  // CHECK-NEXT: ret double %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rp>} : f32, f32 -> f64
  llvm.return %f1 : f64
}

llvm.func @fadd_f32_f32_rz(%a : f32, %b : f32) -> f64 {
  // CHECK-LABEL: define double @fadd_f32_f32_rz(float %0, float %1) {
  // CHECK-NEXT: %3 = fpext float %0 to double
  // CHECK-NEXT: %4 = fpext float %1 to double
  // CHECK-NEXT: %5 = call double @llvm.nvvm.add.rz.d(double %3, double %4)
  // CHECK-NEXT: ret double %5
  // CHECK-NEXT: }
  %f1 = nvvm.fadd %a, %b {rnd = #nvvm.fp_rnd_mode<rz>} : f32, f32 -> f64
  llvm.return %f1 : f64
}
