; REQUIRES: asserts
; The regression tests need to test for order of emitted instructions, and
; therefore, the tests are a bit fragile/reliant on instruction scheduling. The
; test cases have been minimized as much as possible, but still most of the test
; cases could break if instruction scheduling heuristics for cortex-a53 change
; RUN: llc < %s -mcpu=cortex-a53 -mattr=+fix-cortex-a53-835769 -frame-pointer=non-leaf -stats 2>&1 \
; RUN:  | FileCheck %s
; RUN: llc < %s -mcpu=cortex-a53 -mattr=-fix-cortex-a53-835769 -frame-pointer=non-leaf -stats 2>&1 \
; RUN:  | FileCheck %s --check-prefix CHECK-NOWORKAROUND
; The following run lines are just to verify whether or not this pass runs by
; default for given CPUs. Given the fragility of the tests, this is only run on
; a test case where the scheduler has not freedom at all to reschedule the
; instructions, so the potentially massively different scheduling heuristics
; will not break the test case.
; RUN: llc < %s -mcpu=generic    -frame-pointer=non-leaf | FileCheck %s --check-prefix CHECK-BASIC-PASS-DISABLED
; RUN: llc < %s -mcpu=cortex-a53 -frame-pointer=non-leaf | FileCheck %s --check-prefix CHECK-BASIC-PASS-DISABLED
; RUN: llc < %s -mcpu=cortex-a57 -frame-pointer=non-leaf | FileCheck %s --check-prefix CHECK-BASIC-PASS-DISABLED
; RUN: llc < %s -mcpu=cyclone    -frame-pointer=non-leaf | FileCheck %s --check-prefix CHECK-BASIC-PASS-DISABLED

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

define i64 @f_load_madd_64(i64 %a, i64 %b, i64* nocapture readonly %c) #0 {
entry:
  %0 = load i64, i64* %c, align 8
  %mul = mul nsw i64 %0, %b
  %add = add nsw i64 %mul, %a
  ret i64 %add
}
; CHECK-LABEL: f_load_madd_64:
; CHECK:	ldr
; CHECK-NEXT:	nop
; CHECK-NEXT:	madd
; CHECK-NOWORKAROUND-LABEL: f_load_madd_64:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	madd
; CHECK-BASIC-PASS-DISABLED-LABEL: f_load_madd_64:
; CHECK-BASIC-PASS-DISABLED:  ldr
; CHECK-BASIC-PASS-DISABLED-NEXT:  madd


define i32 @f_load_madd_32(i32 %a, i32 %b, i32* nocapture readonly %c) #0 {
entry:
  %0 = load i32, i32* %c, align 4
  %mul = mul nsw i32 %0, %b
  %add = add nsw i32 %mul, %a
  ret i32 %add
}
; CHECK-LABEL: f_load_madd_32:
; CHECK:	ldr
; CHECK-NEXT:	madd
; CHECK-NOWORKAROUND-LABEL: f_load_madd_32:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	madd


define i64 @f_load_msub_64(i64 %a, i64 %b, i64* nocapture readonly %c) #0 {
entry:
  %0 = load i64, i64* %c, align 8
  %mul = mul nsw i64 %0, %b
  %sub = sub nsw i64 %a, %mul
  ret i64 %sub
}
; CHECK-LABEL: f_load_msub_64:
; CHECK:	ldr
; CHECK-NEXT:	nop
; CHECK-NEXT:	msub
; CHECK-NOWORKAROUND-LABEL: f_load_msub_64:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	msub


define i32 @f_load_msub_32(i32 %a, i32 %b, i32* nocapture readonly %c) #0 {
entry:
  %0 = load i32, i32* %c, align 4
  %mul = mul nsw i32 %0, %b
  %sub = sub nsw i32 %a, %mul
  ret i32 %sub
}
; CHECK-LABEL: f_load_msub_32:
; CHECK:	ldr
; CHECK-NEXT:	msub
; CHECK-NOWORKAROUND-LABEL: f_load_msub_32:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	msub


define i64 @f_load_mul_64(i64 %a, i64 %b, i64* nocapture readonly %c) #0 {
entry:
  %0 = load i64, i64* %c, align 8
  %mul = mul nsw i64 %0, %b
  ret i64 %mul
}
; CHECK-LABEL: f_load_mul_64:
; CHECK:	ldr
; CHECK-NEXT:	mul
; CHECK-NOWORKAROUND-LABEL: f_load_mul_64:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	mul


define i32 @f_load_mul_32(i32 %a, i32 %b, i32* nocapture readonly %c) #0 {
entry:
  %0 = load i32, i32* %c, align 4
  %mul = mul nsw i32 %0, %b
  ret i32 %mul
}
; CHECK-LABEL: f_load_mul_32:
; CHECK:	ldr
; CHECK-NEXT:	mul
; CHECK-NOWORKAROUND-LABEL: f_load_mul_32:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	mul


define i64 @f_load_mneg_64(i64 %a, i64 %b, i64* nocapture readonly %c) #0 {
entry:
  %0 = load i64, i64* %c, align 8
  %mul = sub i64 0, %b
  %sub = mul i64 %0, %mul
  ret i64 %sub
}
; CHECK-LABEL: f_load_mneg_64:
; CHECK-NOWORKAROUND-LABEL: f_load_mneg_64:
; FIXME: only add further checks here once LLVM actually produces
;        neg instructions
; FIXME-CHECK: ldr
; FIXME-CHECK-NEXT: nop
; FIXME-CHECK-NEXT: mneg
; FIXME-CHECK-NOWORKAROUND: ldr
; FIXME-CHECK-NOWORKAROUND-NEXT: mneg


define i32 @f_load_mneg_32(i32 %a, i32 %b, i32* nocapture readonly %c) #0 {
entry:
  %0 = load i32, i32* %c, align 4
  %mul = sub i32 0, %b
  %sub = mul i32 %0, %mul
  ret i32 %sub
}
; CHECK-LABEL: f_load_mneg_32:
; CHECK-NOWORKAROUND-LABEL: f_load_mneg_32:
; FIXME: only add further checks here once LLVM actually produces
;        neg instructions
; FIXME-CHECK: ldr
; FIXME-CHECK-NEXT: mneg
; FIXME-CHECK-NOWORKAROUND: ldr
; FIXME-CHECK-NOWORKAROUND-NEXT: mneg


define i64 @f_load_smaddl(i64 %a, i32 %b, i32 %c, i32* nocapture readonly %d) #0 {
entry:
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = mul nsw i64 %conv1, %conv
  %add = add nsw i64 %mul, %a
  %0 = load i32, i32* %d, align 4
  %conv2 = sext i32 %0 to i64
  %add3 = add nsw i64 %add, %conv2
  ret i64 %add3
}
; CHECK-LABEL: f_load_smaddl:
; CHECK:	ldrsw
; CHECK-NEXT:	nop
; CHECK-NEXT:	smaddl
; CHECK-NOWORKAROUND-LABEL: f_load_smaddl:
; CHECK-NOWORKAROUND:	ldrsw
; CHECK-NOWORKAROUND-NEXT:	smaddl


define i64 @f_load_smsubl_64(i64 %a, i32 %b, i32 %c, i32* nocapture readonly %d) #0 {
entry:
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = mul nsw i64 %conv1, %conv
  %sub = sub i64 %a, %mul
  %0 = load i32, i32* %d, align 4
  %conv2 = sext i32 %0 to i64
  %add = add nsw i64 %sub, %conv2
  ret i64 %add
}
; CHECK-LABEL: f_load_smsubl_64:
; CHECK:	ldrsw
; CHECK-NEXT:	nop
; CHECK-NEXT:	smsubl
; CHECK-NOWORKAROUND-LABEL: f_load_smsubl_64:
; CHECK-NOWORKAROUND:	ldrsw
; CHECK-NOWORKAROUND-NEXT:	smsubl


define i64 @f_load_smull(i64 %a, i32 %b, i32 %c, i32* nocapture readonly %d) #0 {
entry:
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = mul nsw i64 %conv1, %conv
  %0 = load i32, i32* %d, align 4
  %conv2 = sext i32 %0 to i64
  %div = sdiv i64 %mul, %conv2
  ret i64 %div
}
; CHECK-LABEL: f_load_smull:
; CHECK:	ldrsw
; CHECK-NEXT:	smull
; CHECK-NOWORKAROUND-LABEL: f_load_smull:
; CHECK-NOWORKAROUND:	ldrsw
; CHECK-NOWORKAROUND-NEXT:	smull


define i64 @f_load_smnegl_64(i64 %a, i32 %b, i32 %c, i32* nocapture readonly %d) #0 {
entry:
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = sub nsw i64 0, %conv
  %sub = mul i64 %conv1, %mul
  %0 = load i32, i32* %d, align 4
  %conv2 = sext i32 %0 to i64
  %div = sdiv i64 %sub, %conv2
  ret i64 %div
}
; CHECK-LABEL: f_load_smnegl_64:
; CHECK-NOWORKAROUND-LABEL: f_load_smnegl_64:
; FIXME: only add further checks here once LLVM actually produces
;        smnegl instructions


define i64 @f_load_umaddl(i64 %a, i32 %b, i32 %c, i32* nocapture readonly %d) #0 {
entry:
  %conv = zext i32 %b to i64
  %conv1 = zext i32 %c to i64
  %mul = mul i64 %conv1, %conv
  %add = add i64 %mul, %a
  %0 = load i32, i32* %d, align 4
  %conv2 = zext i32 %0 to i64
  %add3 = add i64 %add, %conv2
  ret i64 %add3
}
; CHECK-LABEL: f_load_umaddl:
; CHECK:	ldr
; CHECK-NEXT:	nop
; CHECK-NEXT:	umaddl
; CHECK-NOWORKAROUND-LABEL: f_load_umaddl:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	umaddl


define i64 @f_load_umsubl_64(i64 %a, i32 %b, i32 %c, i32* nocapture readonly %d) #0 {
entry:
  %conv = zext i32 %b to i64
  %conv1 = zext i32 %c to i64
  %mul = mul i64 %conv1, %conv
  %sub = sub i64 %a, %mul
  %0 = load i32, i32* %d, align 4
  %conv2 = zext i32 %0 to i64
  %add = add i64 %sub, %conv2
  ret i64 %add
}
; CHECK-LABEL: f_load_umsubl_64:
; CHECK:	ldr
; CHECK-NEXT:	nop
; CHECK-NEXT:	umsubl
; CHECK-NOWORKAROUND-LABEL: f_load_umsubl_64:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	umsubl


define i64 @f_load_umull(i64 %a, i32 %b, i32 %c, i32* nocapture readonly %d) #0 {
entry:
  %conv = zext i32 %b to i64
  %conv1 = zext i32 %c to i64
  %mul = mul i64 %conv1, %conv
  %0 = load i32, i32* %d, align 4
  %conv2 = zext i32 %0 to i64
  %div = udiv i64 %mul, %conv2
  ret i64 %div
}
; CHECK-LABEL: f_load_umull:
; CHECK:	ldr
; CHECK-NEXT:	umull
; CHECK-NOWORKAROUND-LABEL: f_load_umull:
; CHECK-NOWORKAROUND:	ldr
; CHECK-NOWORKAROUND-NEXT:	umull


define i64 @f_load_umnegl_64(i64 %a, i32 %b, i32 %c, i32* nocapture readonly %d) #0 {
entry:
  %conv = zext i32 %b to i64
  %conv1 = zext i32 %c to i64
  %mul = sub nsw i64 0, %conv
  %sub = mul i64 %conv1, %mul
  %0 = load i32, i32* %d, align 4
  %conv2 = zext i32 %0 to i64
  %div = udiv i64 %sub, %conv2
  ret i64 %div
}
; CHECK-LABEL: f_load_umnegl_64:
; CHECK-NOWORKAROUND-LABEL: f_load_umnegl_64:
; FIXME: only add further checks here once LLVM actually produces
;        umnegl instructions


define i64 @f_store_madd_64(i64 %a, i64 %b, i64* nocapture readonly %cp, i64* nocapture %e) #1 {
entry:
  %0 = load i64, i64* %cp, align 8
  store i64 %a, i64* %e, align 8
  %mul = mul nsw i64 %0, %b
  %add = add nsw i64 %mul, %a
  ret i64 %add
}
; CHECK-LABEL: f_store_madd_64:
; CHECK:	str
; CHECK-NEXT:	nop
; CHECK-NEXT:	madd
; CHECK-NOWORKAROUND-LABEL: f_store_madd_64:
; CHECK-NOWORKAROUND:	str
; CHECK-NOWORKAROUND-NEXT:	madd


define i32 @f_store_madd_32(i32 %a, i32 %b, i32* nocapture readonly %cp, i32* nocapture %e) #1 {
entry:
  %0 = load i32, i32* %cp, align 4
  store i32 %a, i32* %e, align 4
  %mul = mul nsw i32 %0, %b
  %add = add nsw i32 %mul, %a
  ret i32 %add
}
; CHECK-LABEL: f_store_madd_32:
; CHECK:	str
; CHECK-NEXT:	madd
; CHECK-NOWORKAROUND-LABEL: f_store_madd_32:
; CHECK-NOWORKAROUND:	str
; CHECK-NOWORKAROUND-NEXT:	madd


define i64 @f_store_msub_64(i64 %a, i64 %b, i64* nocapture readonly %cp, i64* nocapture %e) #1 {
entry:
  %0 = load i64, i64* %cp, align 8
  store i64 %a, i64* %e, align 8
  %mul = mul nsw i64 %0, %b
  %sub = sub nsw i64 %a, %mul
  ret i64 %sub
}
; CHECK-LABEL: f_store_msub_64:
; CHECK:	str
; CHECK-NEXT:	nop
; CHECK-NEXT:	msub
; CHECK-NOWORKAROUND-LABEL: f_store_msub_64:
; CHECK-NOWORKAROUND:	str
; CHECK-NOWORKAROUND-NEXT:	msub


define i32 @f_store_msub_32(i32 %a, i32 %b, i32* nocapture readonly %cp, i32* nocapture %e) #1 {
entry:
  %0 = load i32, i32* %cp, align 4
  store i32 %a, i32* %e, align 4
  %mul = mul nsw i32 %0, %b
  %sub = sub nsw i32 %a, %mul
  ret i32 %sub
}
; CHECK-LABEL: f_store_msub_32:
; CHECK:	str
; CHECK-NEXT:	msub
; CHECK-NOWORKAROUND-LABEL: f_store_msub_32:
; CHECK-NOWORKAROUND:	str
; CHECK-NOWORKAROUND-NEXT:	msub


define i64 @f_store_mul_64(i64 %a, i64 %b, i64* nocapture readonly %cp, i64* nocapture %e) #1 {
entry:
  %0 = load i64, i64* %cp, align 8
  store i64 %a, i64* %e, align 8
  %mul = mul nsw i64 %0, %b
  ret i64 %mul
}
; CHECK-LABEL: f_store_mul_64:
; CHECK:	str
; CHECK-NEXT:	mul
; CHECK-NOWORKAROUND-LABEL: f_store_mul_64:
; CHECK-NOWORKAROUND:	str
; CHECK-NOWORKAROUND-NEXT:	mul


define i32 @f_store_mul_32(i32 %a, i32 %b, i32* nocapture readonly %cp, i32* nocapture %e) #1 {
entry:
  %0 = load i32, i32* %cp, align 4
  store i32 %a, i32* %e, align 4
  %mul = mul nsw i32 %0, %b
  ret i32 %mul
}
; CHECK-LABEL: f_store_mul_32:
; CHECK:	str
; CHECK-NEXT:	mul
; CHECK-NOWORKAROUND-LABEL: f_store_mul_32:
; CHECK-NOWORKAROUND:	str
; CHECK-NOWORKAROUND-NEXT:	mul


define i64 @f_prefetch_madd_64(i64 %a, i64 %b, i64* nocapture readonly %cp, i64* nocapture %e) #1 {
entry:
  %0 = load i64, i64* %cp, align 8
  %1 = bitcast i64* %e to i8*
  tail call void @llvm.prefetch(i8* %1, i32 0, i32 0, i32 1)
  %mul = mul nsw i64 %0, %b
  %add = add nsw i64 %mul, %a
  ret i64 %add
}
; CHECK-LABEL: f_prefetch_madd_64:
; CHECK:	prfm
; CHECK-NEXT:   nop
; CHECK-NEXT:	madd
; CHECK-NOWORKAROUND-LABEL: f_prefetch_madd_64:
; CHECK-NOWORKAROUND:	prfm
; CHECK-NOWORKAROUND-NEXT:	madd

declare void @llvm.prefetch(i8* nocapture, i32, i32, i32) #2

define i32 @f_prefetch_madd_32(i32 %a, i32 %b, i32* nocapture readonly %cp, i32* nocapture %e) #1 {
entry:
  %0 = load i32, i32* %cp, align 4
  %1 = bitcast i32* %e to i8*
  tail call void @llvm.prefetch(i8* %1, i32 1, i32 0, i32 1)
  %mul = mul nsw i32 %0, %b
  %add = add nsw i32 %mul, %a
  ret i32 %add
}
; CHECK-LABEL: f_prefetch_madd_32:
; CHECK:	prfm
; CHECK-NEXT:	madd
; CHECK-NOWORKAROUND-LABEL: f_prefetch_madd_32:
; CHECK-NOWORKAROUND:	prfm
; CHECK-NOWORKAROUND-NEXT:	madd

define i64 @f_prefetch_msub_64(i64 %a, i64 %b, i64* nocapture readonly %cp, i64* nocapture %e) #1 {
entry:
  %0 = load i64, i64* %cp, align 8
  %1 = bitcast i64* %e to i8*
  tail call void @llvm.prefetch(i8* %1, i32 0, i32 1, i32 1)
  %mul = mul nsw i64 %0, %b
  %sub = sub nsw i64 %a, %mul
  ret i64 %sub
}
; CHECK-LABEL: f_prefetch_msub_64:
; CHECK:	prfm
; CHECK-NEXT:   nop
; CHECK-NEXT:	msub
; CHECK-NOWORKAROUND-LABEL: f_prefetch_msub_64:
; CHECK-NOWORKAROUND:	prfm
; CHECK-NOWORKAROUND-NEXT:	msub

define i32 @f_prefetch_msub_32(i32 %a, i32 %b, i32* nocapture readonly %cp, i32* nocapture %e) #1 {
entry:
  %0 = load i32, i32* %cp, align 4
  %1 = bitcast i32* %e to i8*
  tail call void @llvm.prefetch(i8* %1, i32 1, i32 1, i32 1)
  %mul = mul nsw i32 %0, %b
  %sub = sub nsw i32 %a, %mul
  ret i32 %sub
}
; CHECK-LABEL: f_prefetch_msub_32:
; CHECK:	prfm
; CHECK-NEXT:	msub
; CHECK-NOWORKAROUND-LABEL: f_prefetch_msub_32:
; CHECK-NOWORKAROUND:	prfm
; CHECK-NOWORKAROUND-NEXT:	msub

define i64 @f_prefetch_mul_64(i64 %a, i64 %b, i64* nocapture readonly %cp, i64* nocapture %e) #1 {
entry:
  %0 = load i64, i64* %cp, align 8
  %1 = bitcast i64* %e to i8*
  tail call void @llvm.prefetch(i8* %1, i32 0, i32 3, i32 1)
  %mul = mul nsw i64 %0, %b
  ret i64 %mul
}
; CHECK-LABEL: f_prefetch_mul_64:
; CHECK:	prfm
; CHECK-NEXT:	mul
; CHECK-NOWORKAROUND-LABEL: f_prefetch_mul_64:
; CHECK-NOWORKAROUND:	prfm
; CHECK-NOWORKAROUND-NEXT:	mul

define i32 @f_prefetch_mul_32(i32 %a, i32 %b, i32* nocapture readonly %cp, i32* nocapture %e) #1 {
entry:
  %0 = load i32, i32* %cp, align 4
  %1 = bitcast i32* %e to i8*
  tail call void @llvm.prefetch(i8* %1, i32 1, i32 3, i32 1)
  %mul = mul nsw i32 %0, %b
  ret i32 %mul
}
; CHECK-LABEL: f_prefetch_mul_32:
; CHECK:	prfm
; CHECK-NEXT:	mul
; CHECK-NOWORKAROUND-LABEL: f_prefetch_mul_32:
; CHECK-NOWORKAROUND:	prfm
; CHECK-NOWORKAROUND-NEXT:	mul

define i64 @fall_through(i64 %a, i64 %b, i64* nocapture readonly %c) #0 {
entry:
  %0 = load i64, i64* %c, align 8
  br label %block1

block1:
  %mul = mul nsw i64 %0, %b
  %add = add nsw i64 %mul, %a
  %tmp = ptrtoint i8* blockaddress(@fall_through, %block1) to i64
  %ret = add nsw i64 %tmp, %add
  ret i64 %ret
}
; CHECK-LABEL:	fall_through
; CHECK:	ldr
; CHECK-NEXT:	nop
; CHECK-NEXT:	.Ltmp
; CHECK-NEXT: 	%bb.
; CHECK-NEXT: 	madd
; CHECK-NOWORKAROUND-LABEL:	fall_through
; CHECK-NOWORKAROUND: 	ldr
; CHECK-NOWORKAROUND-NEXT:	.Ltmp
; CHECK-NOWORKAROUND-NEXT:	%bb.
; CHECK-NOWORKAROUND-NEXT:	madd

; No checks for this, just check it doesn't crash
define i32 @crash_check(i8** nocapture readnone %data) #0 {
entry:
  br label %while.cond

while.cond:
  br label %while.cond
}

attributes #0 = { nounwind readonly "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }


; CHECK-LABEL: ... Statistics Collected ...
; CHECK: 11 aarch64-fix-cortex-a53-835769 - Number of Nops added to work around erratum 835769
