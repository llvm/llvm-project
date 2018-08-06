; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -verify-machineinstrs \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s

; Function Attrs: norecurse nounwind
define void @qpAdd(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %add = fadd fp128 %0, %0
  store fp128 %add, fp128* %res, align 16
  ret void
; CHECK-LABEL: qpAdd
; CHECK-NOT: bl __addtf3
; CHECK: xsaddqp
; CHECK: stxv
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @qpSub(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %sub = fsub fp128 %0, %0
  store fp128 %sub, fp128* %res, align 16
  ret void
; CHECK-LABEL: qpSub
; CHECK-NOT: bl __subtf3
; CHECK: xssubqp
; CHECK: stxv
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @qpMul(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %mul = fmul fp128 %0, %0
  store fp128 %mul, fp128* %res, align 16
  ret void
; CHECK-LABEL: qpMul
; CHECK-NOT: bl __multf3
; CHECK: xsmulqp
; CHECK: stxv
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @qpDiv(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %div = fdiv fp128 %0, %0
  store fp128 %div, fp128* %res, align 16
  ret void
; CHECK-LABEL: qpDiv
; CHECK-NOT: bl __divtf3
; CHECK: xsdivqp
; CHECK: stxv
; CHECK: blr
}

define void @testLdNSt(i8* nocapture readonly %PtrC, fp128* nocapture %PtrF) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %PtrC, i64 4
  %0 = bitcast i8* %add.ptr to fp128*
  %1 = load fp128, fp128* %0, align 16
  %2 = bitcast fp128* %PtrF to i8*
  %add.ptr1 = getelementptr inbounds i8, i8* %2, i64 8
  %3 = bitcast i8* %add.ptr1 to fp128*
  store fp128 %1, fp128* %3, align 16
  ret void
; CHECK-LABEL: testLdNSt
; CHECK: lxvx
; CHECK: stxvx
; CHECK-NEXT blr
}

define void @qpSqrt(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.sqrt.f128(fp128 %0)
  store fp128 %1, fp128* %res, align 16
  ret void

; CHECK-LABEL: qpSqrt
; CHECK-NOT: bl sqrtl
; CHECK: xssqrtqp
; CHECK: stxv
; CHECK: blr
}
declare fp128 @llvm.sqrt.f128(fp128 %Val)

define void @qpCpsgn(fp128* nocapture readonly %a, fp128* nocapture readonly %b,
                     fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %2 = tail call fp128 @llvm.copysign.f128(fp128 %0, fp128 %1)
  store fp128 %2, fp128* %res, align 16
  ret void

; CHECK-LABEL: qpCpsgn
; CHECK-NOT: rldimi
; CHECK: xscpsgnqp
; CHECK: stxv
; CHECK: blr
}
declare fp128 @llvm.copysign.f128(fp128 %Mag, fp128 %Sgn)

define void @qpAbs(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.fabs.f128(fp128 %0)
  store fp128 %1, fp128* %res, align 16
  ret void

; CHECK-LABEL: qpAbs
; CHECK-NOT: clrldi
; CHECK: xsabsqp
; CHECK: stxv
; CHECK: blr
}
declare fp128 @llvm.fabs.f128(fp128 %Val)

define void @qpNAbs(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.fabs.f128(fp128 %0)
  %neg = fsub fp128 0xL00000000000000008000000000000000, %1
  store fp128 %neg, fp128* %res, align 16
  ret void

; CHECK-LABEL: qpNAbs
; CHECK-NOT: bl __subtf3
; CHECK: xsnabsqp
; CHECK: stxv
; CHECK: blr
}

define void @qpNeg(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %sub = fsub fp128 0xL00000000000000008000000000000000, %0
  store fp128 %sub, fp128* %res, align 16
  ret void

; CHECK-LABEL: qpNeg
; CHECK-NOT: bl __subtf3
; CHECK: xsnegqp
; CHECK: stxv
; CHECK: blr
}

define fp128 @qp_sin(fp128* nocapture readonly %a) {
; CHECK-LABEL: qp_sin:
; CHECK:         lxv v2, 0(r3)
; CHECK:         bl sinf128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.sin.f128(fp128 %0)
  ret fp128 %1
}
declare fp128 @llvm.sin.f128(fp128 %Val)

define fp128 @qp_cos(fp128* nocapture readonly %a) {
; CHECK-LABEL: qp_cos:
; CHECK:         lxv v2, 0(r3)
; CHECK:         bl cosf128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.cos.f128(fp128 %0)
  ret fp128 %1
}
declare fp128 @llvm.cos.f128(fp128 %Val)

define fp128 @qp_log(fp128* nocapture readonly %a) {
; CHECK-LABEL: qp_log:
; CHECK:         lxv v2, 0(r3)
; CHECK:         bl logf128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.log.f128(fp128 %0)
  ret fp128 %1
}
declare fp128     @llvm.log.f128(fp128 %Val)

define fp128 @qp_log10(fp128* nocapture readonly %a) {
; CHECK-LABEL: qp_log10:
; CHECK:         lxv v2, 0(r3)
; CHECK:         bl log10f128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.log10.f128(fp128 %0)
  ret fp128 %1
}
declare fp128     @llvm.log10.f128(fp128 %Val)

define fp128 @qp_log2(fp128* nocapture readonly %a) {
; CHECK-LABEL: qp_log2:
; CHECK:         lxv v2, 0(r3)
; CHECK:         bl log2f128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.log2.f128(fp128 %0)
  ret fp128 %1
}
declare fp128     @llvm.log2.f128(fp128 %Val)

define fp128 @qp_minnum(fp128* nocapture readonly %a,
                        fp128* nocapture readonly %b) {
; CHECK-LABEL: qp_minnum:
; CHECK:         lxv v2, 0(r3)
; CHECK:         lxv v3, 0(r4)
; CHECK:         bl fminf128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %2 = tail call fp128 @llvm.minnum.f128(fp128 %0, fp128 %1)
  ret fp128 %2
}
declare fp128     @llvm.minnum.f128(fp128 %Val0, fp128 %Val1)

define fp128 @qp_maxnum(fp128* nocapture readonly %a,
                        fp128* nocapture readonly %b) {
; CHECK-LABEL: qp_maxnum:
; CHECK:         lxv v2, 0(r3)
; CHECK:         lxv v3, 0(r4)
; CHECK:         bl fmaxf128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %2 = tail call fp128 @llvm.maxnum.f128(fp128 %0, fp128 %1)
  ret fp128 %2
}
declare fp128     @llvm.maxnum.f128(fp128 %Val0, fp128 %Val1)

define fp128 @qp_pow(fp128* nocapture readonly %a,
                     fp128* nocapture readonly %b) {
; CHECK-LABEL: qp_pow:
; CHECK:         lxv v2, 0(r3)
; CHECK:         lxv v3, 0(r4)
; CHECK:         bl powf128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %2 = tail call fp128 @llvm.pow.f128(fp128 %0, fp128 %1)
  ret fp128 %2
}
declare fp128 @llvm.pow.f128(fp128 %Val, fp128 %Power)

define fp128 @qp_exp(fp128* nocapture readonly %a) {
; CHECK-LABEL: qp_exp:
; CHECK:         lxv v2, 0(r3)
; CHECK:         bl expf128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.exp.f128(fp128 %0)
  ret fp128 %1
}
declare fp128     @llvm.exp.f128(fp128 %Val)

define fp128 @qp_exp2(fp128* nocapture readonly %a) {
; CHECK-LABEL: qp_exp2:
; CHECK:         lxv v2, 0(r3)
; CHECK:         bl exp2f128
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.exp2.f128(fp128 %0)
  ret fp128 %1
}
declare fp128     @llvm.exp2.f128(fp128 %Val)

define void @qp_powi(fp128* nocapture readonly %a, i32* nocapture readonly %b,
                     fp128* nocapture %res) {
; CHECK-LABEL: qp_powi:
; CHECK:         lxv v2, 0(r3)
; CHECK:         lwz r3, 0(r4)
; CHECK:         bl __powikf2
; CHECK:         blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load i32, i32* %b, align 8
  %2 = tail call fp128 @llvm.powi.f128(fp128 %0, i32 %1)
  store fp128 %2, fp128* %res, align 16
  ret void
}
declare fp128 @llvm.powi.f128(fp128 %Val, i32 %power)

@a = common global fp128 0xL00000000000000000000000000000000, align 16
@b = common global fp128 0xL00000000000000000000000000000000, align 16

define fp128 @qp_frem() #0 {
entry:
  %0 = load fp128, fp128* @a, align 16
  %1 = load fp128, fp128* @b, align 16
  %rem = frem fp128 %0, %1
  ret fp128 %rem
; CHECK-LABEL: qp_frem
; CHECK: bl fmodf128
; CHECK: blr
}
