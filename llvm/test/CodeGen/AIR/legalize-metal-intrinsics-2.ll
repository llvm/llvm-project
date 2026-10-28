; More of what Metal crashes on, leaves undefined or computes with NaN
; payloads lost (found on an M2 Max, llvm/utils/air-gpu groups llvm-half,
; llvm-int-narrow and llvm-math-ext). The Metal pipeline expands them; the
; portable passes, whose AIR the AGX backend compiles, keep them.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s
; RUN: llc -mtriple=air %s -o - | FileCheck %s --check-prefix=LLC
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE
; RUN: llc -mtriple=air -filetype=obj %s -o /dev/null

; CHECK-LABEL: define float @powi_f32(
; CHECK: [[NF:%.*]] = sitofp i32 %n to float
; CHECK: [[ABS:%.*]] = call float @llvm.fabs.f32(float %x)
; CHECK: [[P:%.*]] = call float @llvm.pow.f32(float [[ABS]], float [[NF]])
; CHECK: [[NEG:%.*]] = fsub float -0.000000e+00, [[P]]
; CHECK: select i1 {{%.*}}, float [[NEG]], float [[P]]
; PORTABLE-LABEL: define float @powi_f32(
; PORTABLE: call float @llvm.powi.f32.i32(
define float @powi_f32(float %x, i32 %n) {
  %r = call float @llvm.powi.f32.i32(float %x, i32 %n)
  ret float %r
}

; CHECK-LABEL: define i32 @lround_i32(
; CHECK: [[R:%.*]] = call float @llvm.round.f32(float %x)
; CHECK: fptosi float [[R]] to i32
define i32 @lround_i32(float %x) {
  %r = call i32 @llvm.lround.i32.f32(float %x)
  ret i32 %r
}

; CHECK-LABEL: define i64 @llrint_i64(
; CHECK: [[R:%.*]] = call float @llvm.rint.f32(float %x)
; CHECK: fptosi float [[R]] to i64
define i64 @llrint_i64(float %x) {
  %r = call i64 @llvm.llrint.i64.f32(float %x)
  ret i64 %r
}

; CHECK-LABEL: define float @canonicalize_f32(
; CHECK-NEXT: ret float %x
define float @canonicalize_f32(float %x) {
  %r = call float @llvm.canonicalize.f32(float %x)
  ret float %r
}

; CHECK-LABEL: define float @minimumnum_f32(
; CHECK: [[MN:%.*]] = call float @llvm.minnum.f32(float %x, float %y)
; CHECK: [[OR:%.*]] = or i32
; CHECK: [[EQ:%.*]] = fcmp oeq float %x, %y
; CHECK: select i1 [[EQ]], float {{%.*}}, float [[MN]]
define float @minimumnum_f32(float %x, float %y) {
  %r = call float @llvm.minimumnum.f32(float %x, float %y)
  ret float %r
}

; CHECK-LABEL: define float @sincos_f32(
; CHECK: [[S:%.*]] = call float @llvm.sin.f32(float %x)
; CHECK: [[C:%.*]] = call float @llvm.cos.f32(float %x)
; CHECK: fadd float [[S]], [[C]]
define float @sincos_f32(float %x) {
  %sc = call { float, float } @llvm.sincos.f32(float %x)
  %s = extractvalue { float, float } %sc, 0
  %c = extractvalue { float, float } %sc, 1
  %r = fadd float %s, %c
  ret float %r
}

; CHECK-LABEL: define i1 @umul_ovf_i64(
; CHECK: [[M:%.*]] = mul i64 %x, %y
; CHECK: [[Q:%.*]] = udiv i64 [[M]],
; CHECK: icmp ne i64 [[Q]], %y
; LLC-LABEL: define i1 @umul_ovf_i64(
; LLC-NOT: with.overflow
; LLC: ret i1
define i1 @umul_ovf_i64(i64 %x, i64 %y) {
  %s = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %x, i64 %y)
  %o = extractvalue { i64, i1 } %s, 1
  ret i1 %o
}

; Metal multiplies 32-bit operands with overflow right: kept.
; CHECK-LABEL: define i1 @umul_ovf_i32(
; CHECK: call { i32, i1 } @llvm.umul.with.overflow.i32(
define i1 @umul_ovf_i32(i32 %x, i32 %y) {
  %s = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %o = extractvalue { i32, i1 } %s, 1
  ret i1 %o
}

; CHECK-LABEL: define half @fabs_f16(
; CHECK: [[B:%.*]] = bitcast half %x to i16
; CHECK: [[A:%.*]] = and i16 [[B]], 32767
; CHECK: bitcast i16 [[A]] to half
; Float keeps air.fabs.
; CHECK-LABEL: define float @fabs_f32(
; CHECK: call float @llvm.fabs.f32(float %x)
define half @fabs_f16(half %x) {
  %r = call half @llvm.fabs.f16(half %x)
  ret half %r
}
define float @fabs_f32(float %x) {
  %r = call float @llvm.fabs.f32(float %x)
  ret float %r
}

; LLC-LABEL: define half @fneg_f16(
; LLC: [[B:%.*]] = bitcast half {{%.*}} to i16
; LLC: xor i16 [[B]], -32768
; LLC-LABEL: define float @fneg_f32(
; LLC: fsub float -0.000000e+00,
define half @fneg_f16(half %x) {
  %r = fneg half %x
  ret half %r
}
define float @fneg_f32(float %x) {
  %r = fneg float %x
  ret float %r
}

; CHECK-LABEL: define i32 @smulh_i32(
; CHECK: [[X:%.*]] = sext i32 %x to i64
; CHECK: [[Y:%.*]] = sext i32 %y to i64
; CHECK: [[P:%.*]] = mul i64 [[X]], [[Y]]
; CHECK: [[H:%.*]] = lshr i64 [[P]], 32
; CHECK: trunc i64 [[H]] to i32
define i32 @smulh_i32(i32 %x, i32 %y) {
  %r = call i32 @llvm.smulh.i32(i32 %x, i32 %y)
  ret i32 %r
}

; CHECK-LABEL: define i32 @umulh_i32(
; CHECK: zext i32 %x to i64
; CHECK-NOT: llvm.umulh
define i32 @umulh_i32(i32 %x, i32 %y) {
  %r = call i32 @llvm.umulh.i32(i32 %x, i32 %y)
  ret i32 %r
}

declare i32 @llvm.smulh.i32(i32, i32)
declare i32 @llvm.umulh.i32(i32, i32)
declare float @llvm.powi.f32.i32(float, i32)
declare i32 @llvm.lround.i32.f32(float)
declare i64 @llvm.llrint.i64.f32(float)
declare float @llvm.canonicalize.f32(float)
declare float @llvm.minimumnum.f32(float, float)
declare { float, float } @llvm.sincos.f32(float)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
declare half @llvm.fabs.f16(half)
declare float @llvm.fabs.f32(float)
