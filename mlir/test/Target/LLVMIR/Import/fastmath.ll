; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @fastmath_inst
define void @fastmath_inst(float %arg1, float %arg2, i1 %arg3) {
  ; CHECK: llvm.fadd %{{.*}}, %{{.*}}  {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32
  %1 = fadd nnan ninf float %arg1, %arg2
  ; CHECK: llvm.fsub %{{.*}}, %{{.*}}  {fastmathFlags = #llvm.fastmath<nsz>} : f32
  %2 = fsub nsz float %arg1, %arg2
  ; CHECK: llvm.fmul %{{.*}}, %{{.*}}  {fastmathFlags = #llvm.fastmath<arcp, contract>} : f32
  %3 = fmul arcp contract float %arg1, %arg2
  ; CHECK: llvm.fdiv %{{.*}}, %{{.*}}  {fastmathFlags = #llvm.fastmath<afn, reassoc>} : f32
  %4 = fdiv afn reassoc float %arg1, %arg2
  ; CHECK: llvm.fneg %{{.*}}  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %5 = fneg fast float %arg1
  ; CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<contract>} : i1, f32
  %6 = select contract i1 %arg3, float %arg1, float %arg2
  ret void
}

; // -----

; CHECK-LABEL: @fastmath_fcmp
define void @fastmath_fcmp(float %arg1, float %arg2) {
  ; CHECK:  llvm.fcmp "oge" %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f32
  %1 = fcmp nsz oge float %arg1, %arg2
  ret void
}

; // -----

declare float @fn(float)

; CHECK-LABEL: @fastmath_call
define void @fastmath_call(float %arg1) {
  ; CHECK:  llvm.call @fn(%{{.*}}) {fastmathFlags = #llvm.fastmath<ninf>} : (f32) -> f32
  %1 = call ninf float @fn(float %arg1)
  ret void
}

; // -----

declare float @llvm.exp.f32(float)
declare float @llvm.powi.f32.i32(float, i32)
declare float @llvm.pow.f32(float, float)
declare float @llvm.fmuladd.f32(float, float, float)
declare float @llvm.vector.reduce.fmin.v2f32(<2 x float>)
declare float @llvm.vector.reduce.fmax.v2f32(<2 x float>)
declare float @llvm.vector.reduce.fminimum.v2f32(<2 x float>)
declare float @llvm.vector.reduce.fmaximum.v2f32(<2 x float>)

; CHECK-LABEL: @fastmath_intr
define void @fastmath_intr(float %arg1, i32 %arg2, <2 x float> %arg3) {
  ; CHECK: llvm.intr.exp(%{{.*}}) {fastmathFlags = #llvm.fastmath<nnan, ninf>} : (f32) -> f32
  %1 = call nnan ninf float @llvm.exp.f32(float %arg1)
  ; CHECK: llvm.intr.powi(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (f32, i32) -> f32
  %2 = call fast float @llvm.powi.f32.i32(float %arg1, i32 %arg2)
  ; CHECK:  llvm.intr.pow(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (f32, f32) -> f32
  %3 = call fast float @llvm.pow.f32(float %arg1, float %arg1)
  ; CHECK: llvm.intr.fmuladd(%{{.*}}, %{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (f32, f32, f32) -> f32
  %4 = call fast float @llvm.fmuladd.f32(float %arg1, float %arg1, float %arg1)
  ; CHECK: %{{.*}} = llvm.intr.vector.reduce.fmin({{.*}}) {fastmathFlags = #llvm.fastmath<nnan>} : (vector<2xf32>) -> f32
  %5 = call nnan float @llvm.vector.reduce.fmin.v2f32(<2 x float> %arg3)
  ; CHECK: %{{.*}} = llvm.intr.vector.reduce.fmax({{.*}}) {fastmathFlags = #llvm.fastmath<nnan>} : (vector<2xf32>) -> f32
  %6 = call nnan float @llvm.vector.reduce.fmax.v2f32(<2 x float> %arg3)
  ; CHECK: %{{.*}} = llvm.intr.vector.reduce.fminimum({{.*}}) {fastmathFlags = #llvm.fastmath<nnan>} : (vector<2xf32>) -> f32
  %7 = call nnan float @llvm.vector.reduce.fminimum.v2f32(<2 x float> %arg3)
  ; CHECK: %{{.*}} = llvm.intr.vector.reduce.fmaximum({{.*}}) {fastmathFlags = #llvm.fastmath<nnan>} : (vector<2xf32>) -> f32
  %8 = call nnan float @llvm.vector.reduce.fmaximum.v2f32(<2 x float> %arg3)

  ret void
}
