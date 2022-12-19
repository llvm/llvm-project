; RUN: llc -verify-machineinstrs < %s -mtriple=armv4t   | FileCheck %s -check-prefix=CHECK-v4A32
; RUN: llc -verify-machineinstrs < %s -mtriple=armv7a   | FileCheck %s -check-prefix=CHECK-v7A32
; RUN: llc -verify-machineinstrs < %s -mtriple=thumbv7a | FileCheck %s -check-prefix=CHECK-THUMB2
; FIXME: There are no tests for Thumb1 since dynamic stack alignment is not supported for
; Thumb1.

define i32 @f_bic_can_be_used_align() nounwind {
entry:
; CHECK-LABEL: f_bic_can_be_used_align:
; CHECK-v7A32: bfc        sp, #0, #8
; CHECK-v4A32: bic        sp, sp, #255
; CHECK-THUMB2:	mov	r4, sp
; CHECK-THUMB2-NEXT: bfc	r4, #0, #8
; CHECK-THUMB2-NEXT: mov	sp, r4
  %x = alloca i32, align 256
  store volatile i32 0, ptr %x, align 256
  ret i32 0
}

define i32 @f_too_large_for_bic_align() nounwind {
entry:
; CHECK-LABEL: f_too_large_for_bic_align:
; CHECK-v7A32: bfc sp, #0, #9
; CHECK-v4A32: lsr sp, sp, #9
; CHECK-v4A32: lsl sp, sp, #9
; CHECK-THUMB2:	mov	r4, sp
; CHECK-THUMB2-NEXT:	bfc	r4, #0, #9
; CHECK-THUMB2-NEXT:	mov	sp, r4
  %x = alloca i32, align 512
  store volatile i32 0, ptr %x, align 512
  ret i32 0
}

define ptr @f_alignedDPRCS2Spills(ptr %d) #0 {
entry:
; CHECK-LABEL: f_too_large_for_bic_align:
; CHECK-v7A32: bfc sp, #0, #12
; CHECK-v4A32: lsr sp, sp, #12
; CHECK-v4A32: lsl sp, sp, #12
; CHECK-THUMB2:      bfc	r4, #0, #12
; CHECK-THUMB2-NEXT: mov	sp, r4
  %a = alloca i8, align 4096
  %0 = load double, ptr %d, align 4
  %arrayidx1 = getelementptr inbounds double, ptr %d, i32 1
  %1 = load double, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds double, ptr %d, i32 2
  %2 = load double, ptr %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds double, ptr %d, i32 3
  %3 = load double, ptr %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds double, ptr %d, i32 4
  %4 = load double, ptr %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds double, ptr %d, i32 5
  %5 = load double, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds double, ptr %d, i32 6
  %6 = load double, ptr %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds double, ptr %d, i32 7
  %7 = load double, ptr %arrayidx7, align 4
  %arrayidx8 = getelementptr inbounds double, ptr %d, i32 8
  %8 = load double, ptr %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds double, ptr %d, i32 9
  %9 = load double, ptr %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds double, ptr %d, i32 10
  %10 = load double, ptr %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds double, ptr %d, i32 11
  %11 = load double, ptr %arrayidx11, align 4
  %arrayidx12 = getelementptr inbounds double, ptr %d, i32 12
  %12 = load double, ptr %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds double, ptr %d, i32 13
  %13 = load double, ptr %arrayidx13, align 4
  %arrayidx14 = getelementptr inbounds double, ptr %d, i32 14
  %14 = load double, ptr %arrayidx14, align 4
  %arrayidx15 = getelementptr inbounds double, ptr %d, i32 15
  %15 = load double, ptr %arrayidx15, align 4
  %arrayidx16 = getelementptr inbounds double, ptr %d, i32 16
  %16 = load double, ptr %arrayidx16, align 4
  %arrayidx17 = getelementptr inbounds double, ptr %d, i32 17
  %17 = load double, ptr %arrayidx17, align 4
  %arrayidx18 = getelementptr inbounds double, ptr %d, i32 18
  %18 = load double, ptr %arrayidx18, align 4
  %arrayidx19 = getelementptr inbounds double, ptr %d, i32 19
  %19 = load double, ptr %arrayidx19, align 4
  %arrayidx20 = getelementptr inbounds double, ptr %d, i32 20
  %20 = load double, ptr %arrayidx20, align 4
  %arrayidx21 = getelementptr inbounds double, ptr %d, i32 21
  %21 = load double, ptr %arrayidx21, align 4
  %arrayidx22 = getelementptr inbounds double, ptr %d, i32 22
  %22 = load double, ptr %arrayidx22, align 4
  %arrayidx23 = getelementptr inbounds double, ptr %d, i32 23
  %23 = load double, ptr %arrayidx23, align 4
  %arrayidx24 = getelementptr inbounds double, ptr %d, i32 24
  %24 = load double, ptr %arrayidx24, align 4
  %arrayidx25 = getelementptr inbounds double, ptr %d, i32 25
  %25 = load double, ptr %arrayidx25, align 4
  %arrayidx26 = getelementptr inbounds double, ptr %d, i32 26
  %26 = load double, ptr %arrayidx26, align 4
  %arrayidx27 = getelementptr inbounds double, ptr %d, i32 27
  %27 = load double, ptr %arrayidx27, align 4
  %arrayidx28 = getelementptr inbounds double, ptr %d, i32 28
  %28 = load double, ptr %arrayidx28, align 4
  %arrayidx29 = getelementptr inbounds double, ptr %d, i32 29
  %29 = load double, ptr %arrayidx29, align 4
  %div = fdiv double %29, %28
  %div30 = fdiv double %div, %27
  %div31 = fdiv double %div30, %26
  %div32 = fdiv double %div31, %25
  %div33 = fdiv double %div32, %24
  %div34 = fdiv double %div33, %23
  %div35 = fdiv double %div34, %22
  %div36 = fdiv double %div35, %21
  %div37 = fdiv double %div36, %20
  %div38 = fdiv double %div37, %19
  %div39 = fdiv double %div38, %18
  %div40 = fdiv double %div39, %17
  %div41 = fdiv double %div40, %16
  %div42 = fdiv double %div41, %15
  %div43 = fdiv double %div42, %14
  %div44 = fdiv double %div43, %13
  %div45 = fdiv double %div44, %12
  %div46 = fdiv double %div45, %11
  %div47 = fdiv double %div46, %10
  %div48 = fdiv double %div47, %9
  %div49 = fdiv double %div48, %8
  %div50 = fdiv double %div49, %7
  %div51 = fdiv double %div50, %6
  %div52 = fdiv double %div51, %5
  %div53 = fdiv double %div52, %4
  %div54 = fdiv double %div53, %3
  %div55 = fdiv double %div54, %2
  %div56 = fdiv double %div55, %1
  %div57 = fdiv double %div56, %0
  %div58 = fdiv double %0, %1
  %div59 = fdiv double %div58, %2
  %div60 = fdiv double %div59, %3
  %div61 = fdiv double %div60, %4
  %div62 = fdiv double %div61, %5
  %div63 = fdiv double %div62, %6
  %div64 = fdiv double %div63, %7
  %div65 = fdiv double %div64, %8
  %div66 = fdiv double %div65, %9
  %div67 = fdiv double %div66, %10
  %div68 = fdiv double %div67, %11
  %div69 = fdiv double %div68, %12
  %div70 = fdiv double %div69, %13
  %div71 = fdiv double %div70, %14
  %div72 = fdiv double %div71, %15
  %div73 = fdiv double %div72, %16
  %div74 = fdiv double %div73, %17
  %div75 = fdiv double %div74, %18
  %div76 = fdiv double %div75, %19
  %div77 = fdiv double %div76, %20
  %div78 = fdiv double %div77, %21
  %div79 = fdiv double %div78, %22
  %div80 = fdiv double %div79, %23
  %div81 = fdiv double %div80, %24
  %div82 = fdiv double %div81, %25
  %div83 = fdiv double %div82, %26
  %div84 = fdiv double %div83, %27
  %div85 = fdiv double %div84, %28
  %div86 = fdiv double %div85, %29
  %mul = fmul double %div57, %div86
  %conv = fptosi double %mul to i32
  %add.ptr = getelementptr inbounds i8, ptr %a, i32 %conv
  ret ptr %add.ptr
}
