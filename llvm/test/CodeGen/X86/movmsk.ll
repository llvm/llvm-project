; RUN: llc -mcpu=core2 < %s | FileCheck %s
; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.6"

%0 = type { double }
%union.anon = type { float }

define i32 @double_signbit(double %d1) nounwind uwtable readnone ssp {
entry:
  %__x.addr.i = alloca double, align 8
  %__u.i = alloca %0, align 8
  %0 = bitcast double* %__x.addr.i to i8*
  %1 = bitcast %0* %__u.i to i8*
  store double %d1, double* %__x.addr.i, align 8
  %__f.i = getelementptr inbounds %0* %__u.i, i64 0, i32 0
  store double %d1, double* %__f.i, align 8
  %tmp = bitcast double %d1 to i64
; CHECK-NOT: shr
; CHECK: movmskpd
; CHECK-NEXT: and
  %tmp1 = lshr i64 %tmp, 63
  %shr.i = trunc i64 %tmp1 to i32
  ret i32 %shr.i
}

define i32 @double_add_signbit(double %d1, double %d2) nounwind uwtable readnone ssp {
entry:
  %__x.addr.i = alloca double, align 8
  %__u.i = alloca %0, align 8
  %add = fadd double %d1, %d2
  %0 = bitcast double* %__x.addr.i to i8*
  %1 = bitcast %0* %__u.i to i8*
  store double %add, double* %__x.addr.i, align 8
  %__f.i = getelementptr inbounds %0* %__u.i, i64 0, i32 0
  store double %add, double* %__f.i, align 8
  %tmp = bitcast double %add to i64
; CHECK-NOT: shr
; CHECK: movmskpd
; CHECK-NEXT: and
  %tmp1 = lshr i64 %tmp, 63
  %shr.i = trunc i64 %tmp1 to i32
  ret i32 %shr.i
}

define i32 @float_signbit(float %f1) nounwind uwtable readnone ssp {
entry:
  %__x.addr.i = alloca float, align 4
  %__u.i = alloca %union.anon, align 4
  %0 = bitcast float* %__x.addr.i to i8*
  %1 = bitcast %union.anon* %__u.i to i8*
  store float %f1, float* %__x.addr.i, align 4
  %__f.i = getelementptr inbounds %union.anon* %__u.i, i64 0, i32 0
  store float %f1, float* %__f.i, align 4
  %2 = bitcast float %f1 to i32
; CHECK-NOT: shr
; CHECK: movmskps
; CHECK-NEXT: and
  %shr.i = lshr i32 %2, 31
  ret i32 %shr.i
}

define i32 @float_add_signbit(float %f1, float %f2) nounwind uwtable readnone ssp {
entry:
  %__x.addr.i = alloca float, align 4
  %__u.i = alloca %union.anon, align 4
  %add = fadd float %f1, %f2
  %0 = bitcast float* %__x.addr.i to i8*
  %1 = bitcast %union.anon* %__u.i to i8*
  store float %add, float* %__x.addr.i, align 4
  %__f.i = getelementptr inbounds %union.anon* %__u.i, i64 0, i32 0
  store float %add, float* %__f.i, align 4
  %2 = bitcast float %add to i32
; CHECK-NOT: shr
; CHECK: movmskps
; CHECK-NEXT: and
  %shr.i = lshr i32 %2, 31
  ret i32 %shr.i
}

; PR11570
define void @float_call_signbit(double %n) {
entry:
; FIXME: This should also use movmskps; we don't form the FGETSIGN node
; in this case, though.
; CHECK: float_call_signbit:
; CHECK: movd %xmm0, %rdi
; FIXME
  %t0 = bitcast double %n to i64
  %tobool.i.i.i.i = icmp slt i64 %t0, 0
  tail call void @float_call_signbit_callee(i1 zeroext %tobool.i.i.i.i)
  ret void
}
declare void @float_call_signbit_callee(i1 zeroext)


; rdar://10247336
; movmskp{s|d} only set low 4/2 bits, high bits are known zero

define i32 @t1(<4 x float> %x, i32* nocapture %indexTable) nounwind uwtable readonly ssp {
entry:
; CHECK: t1:
; CHECK: movmskps
; CHECK-NOT: movslq
  %0 = tail call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %x) nounwind
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds i32* %indexTable, i64 %idxprom
  %1 = load i32* %arrayidx, align 4
  ret i32 %1
}

define i32 @t2(<4 x float> %x, i32* nocapture %indexTable) nounwind uwtable readonly ssp {
entry:
; CHECK: t2:
; CHECK: movmskpd
; CHECK-NOT: movslq
  %0 = bitcast <4 x float> %x to <2 x double>
  %1 = tail call i32 @llvm.x86.sse2.movmsk.pd(<2 x double> %0) nounwind
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds i32* %indexTable, i64 %idxprom
  %2 = load i32* %arrayidx, align 4
  ret i32 %2
}

declare i32 @llvm.x86.sse2.movmsk.pd(<2 x double>) nounwind readnone
declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone
