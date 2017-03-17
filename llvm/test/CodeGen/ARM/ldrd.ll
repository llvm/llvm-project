; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -regalloc=fast -optimize-regalloc=0 -verify-machineinstrs | FileCheck %s -check-prefix=A8 -check-prefix=CHECK -check-prefix=NORMAL
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-m3 -regalloc=fast -optimize-regalloc=0 | FileCheck %s -check-prefix=M3 -check-prefix=CHECK -check-prefix=NORMAL
; rdar://6949835
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -regalloc=basic | FileCheck %s -check-prefix=BASIC -check-prefix=CHECK -check-prefix=NORMAL
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -regalloc=greedy | FileCheck %s -check-prefix=GREEDY -check-prefix=CHECK -check-prefix=NORMAL
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=swift | FileCheck %s -check-prefix=SWIFT -check-prefix=CHECK -check-prefix=NORMAL

; RUN: llc < %s -mtriple=thumbv7-apple-ios -arm-assume-misaligned-load-store | FileCheck %s -check-prefix=CHECK -check-prefix=CONSERVATIVE

; Magic ARM pair hints works best with linearscan / fast.

@b = external global i64*

; We use the following two to force values into specific registers.
declare i64* @get_ptr()
declare void @use_i64(i64 %v)

define void @test_ldrd(i64 %a) nounwind readonly "no-frame-pointer-elim"="true" {
; CHECK-LABEL: test_ldrd:
; NORMAL: bl{{x?}} _get_ptr
; A8: ldrd r0, r1, [r0]
; Cortex-M3 errata 602117: LDRD with base in list may result in incorrect base
; register when interrupted or faulted.
; M3-NOT: ldrd r[[REGNUM:[0-9]+]], {{r[0-9]+}}, [r[[REGNUM]]]
; CONSERVATIVE-NOT: ldrd
; NORMAL: bl{{x?}} _use_i64
  %ptr = call i64* @get_ptr()
  %v = load i64, i64* %ptr, align 8
  call void @use_i64(i64 %v)
  ret void
}

; rdar://10435045 mixed LDRi8/LDRi12
;
; In this case, LSR generate a sequence of LDRi8/LDRi12. We should be
; able to generate an LDRD pair here, but this is highly sensitive to
; regalloc hinting. So, this doubles as a register allocation
; test. RABasic currently does a better job within the inner loop
; because of its *lack* of hinting ability. Whereas RAGreedy keeps
; R0/R1/R2 live as the three arguments, forcing the LDRD's odd
; destination into R3. We then sensibly split LDRD again rather then
; evict another live range or use callee saved regs. Sorry if the test
; is sensitive to Regalloc changes, but it is an interesting case.
;
; CHECK-LABEL: f:
; BASIC: %bb
; BASIC: ldrd
; BASIC: str
; GREEDY: %bb
; GREEDY: ldrd
; GREEDY: str
define void @f(i32* nocapture %a, i32* nocapture %b, i32 %n) nounwind "no-frame-pointer-elim"="true" {
entry:
  %0 = add nsw i32 %n, -1                         ; <i32> [#uses=2]
  %1 = icmp sgt i32 %0, 0                         ; <i1> [#uses=1]
  br i1 %1, label %bb, label %return

bb:                                               ; preds = %bb, %entry
  %i.03 = phi i32 [ %tmp, %bb ], [ 0, %entry ]    ; <i32> [#uses=3]
  %scevgep = getelementptr i32, i32* %a, i32 %i.03     ; <i32*> [#uses=1]
  %scevgep4 = getelementptr i32, i32* %b, i32 %i.03    ; <i32*> [#uses=1]
  %tmp = add i32 %i.03, 1                         ; <i32> [#uses=3]
  %scevgep5 = getelementptr i32, i32* %a, i32 %tmp     ; <i32*> [#uses=1]
  %2 = load i32, i32* %scevgep, align 4                ; <i32> [#uses=1]
  %3 = load i32, i32* %scevgep5, align 4               ; <i32> [#uses=1]
  %4 = add nsw i32 %3, %2                         ; <i32> [#uses=1]
  store i32 %4, i32* %scevgep4, align 4
  %exitcond = icmp eq i32 %tmp, %0                ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}

; rdar://13978317
; Pair of loads not formed when lifetime markers are set.
%struct.Test = type { i32, i32, i32 }

@TestVar = external global %struct.Test

; CHECK-LABEL: Func1:
define void @Func1() nounwind ssp "no-frame-pointer-elim"="true" {
entry: 
; A8: movw [[BASE:r[0-9]+]], :lower16:{{.*}}TestVar{{.*}}
; A8: movt [[BASE]], :upper16:{{.*}}TestVar{{.*}}
; A8: ldrd [[FIELD1:r[0-9]+]], [[FIELD2:r[0-9]+]], {{\[}}[[BASE]], #4]
; A8-NEXT: add [[FIELD1]], [[FIELD2]]
; A8-NEXT: str [[FIELD1]], {{\[}}[[BASE]]{{\]}}
; CONSERVATIVE-NOT: ldrd
  %orig_blocks = alloca [256 x i16], align 2
  %0 = bitcast [256 x i16]* %orig_blocks to i8*call void @llvm.lifetime.start(i64 512, i8* %0) nounwind
  %tmp1 = load i32, i32* getelementptr inbounds (%struct.Test, %struct.Test* @TestVar, i32 0, i32 1), align 4
  %tmp2 = load i32, i32* getelementptr inbounds (%struct.Test, %struct.Test* @TestVar, i32 0, i32 2), align 4
  %add = add nsw i32 %tmp2, %tmp1
  store i32 %add, i32* getelementptr inbounds (%struct.Test, %struct.Test* @TestVar, i32 0, i32 0), align 4
  call void @llvm.lifetime.end(i64 512, i8* %0) nounwind
  ret void
}

declare void @extfunc(i32, i32, i32, i32)

; CHECK-LABEL: Func2:
; CONSERVATIVE-NOT: ldrd
; A8: ldrd
; CHECK: bl{{x?}} _extfunc
; A8: pop
define void @Func2(i32* %p) "no-frame-pointer-elim"="true" {
entry:
  %addr0 = getelementptr i32, i32* %p, i32 0
  %addr1 = getelementptr i32, i32* %p, i32 1
  %v0 = load i32, i32* %addr0
  %v1 = load i32, i32* %addr1
  ; try to force %v0/%v1 into non-adjacent registers
  call void @extfunc(i32 %v0, i32 0, i32 0, i32 %v1)
  ret void
}

; CHECK-LABEL: strd_spill_ldrd_reload:
; A8: strd r1, r0, [sp, #-8]!
; M3: strd r1, r0, [sp, #-8]!
; BASIC: strd r1, r0, [sp, #-8]!
; GREEDY: strd r0, r1, [sp, #-8]!
; CONSERVATIVE: strd r0, r1, [sp, #-8]!
; NORMAL: @ InlineAsm Start
; NORMAL: @ InlineAsm End
; A8: ldrd r2, r1, [sp]
; M3: ldrd r2, r1, [sp]
; BASIC: ldrd r2, r1, [sp]
; GREEDY: ldrd r1, r2, [sp]
; CONSERVATIVE: ldrd r1, r2, [sp]
; CHECK: bl{{x?}} _extfunc
define void @strd_spill_ldrd_reload(i32 %v0, i32 %v1) "no-frame-pointer-elim"="true" {
  ; force %v0 and %v1 to be spilled
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{lr}"()
  ; force the reloaded %v0, %v1 into different registers
  call void @extfunc(i32 0, i32 %v0, i32 %v1, i32 7)
  ret void
}

declare void @extfunc2(i32*, i32, i32)

; CHECK-LABEL: ldrd_postupdate_dec:
; NORMAL: ldrd r1, r2, [r0], #-8
; CONSERVATIVE-NOT: ldrd
; CHECK: bl{{x?}} _extfunc
define void @ldrd_postupdate_dec(i32* %p0) "no-frame-pointer-elim"="true" {
  %p0.1 = getelementptr i32, i32* %p0, i32 1
  %v0 = load i32, i32* %p0
  %v1 = load i32, i32* %p0.1
  %p1 = getelementptr i32, i32* %p0, i32 -2
  call void @extfunc2(i32* %p1, i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: ldrd_postupdate_inc:
; NORMAL: ldrd r1, r2, [r0], #8
; CONSERVATIVE-NOT: ldrd
; CHECK: bl{{x?}} _extfunc
define void @ldrd_postupdate_inc(i32* %p0) "no-frame-pointer-elim"="true" {
  %p0.1 = getelementptr i32, i32* %p0, i32 1
  %v0 = load i32, i32* %p0
  %v1 = load i32, i32* %p0.1
  %p1 = getelementptr i32, i32* %p0, i32 2
  call void @extfunc2(i32* %p1, i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: strd_postupdate_dec:
; NORMAL: strd r1, r2, [r0], #-8
; CONSERVATIVE-NOT: strd
; CHECK: bx lr
define i32* @strd_postupdate_dec(i32* %p0, i32 %v0, i32 %v1) "no-frame-pointer-elim"="true" {
  %p0.1 = getelementptr i32, i32* %p0, i32 1
  store i32 %v0, i32* %p0
  store i32 %v1, i32* %p0.1
  %p1 = getelementptr i32, i32* %p0, i32 -2
  ret i32* %p1
}

; CHECK-LABEL: strd_postupdate_inc:
; NORMAL: strd r1, r2, [r0], #8
; CONSERVATIVE-NOT: strd
; CHECK: bx lr
define i32* @strd_postupdate_inc(i32* %p0, i32 %v0, i32 %v1) "no-frame-pointer-elim"="true" {
  %p0.1 = getelementptr i32, i32* %p0, i32 1
  store i32 %v0, i32* %p0
  store i32 %v1, i32* %p0.1
  %p1 = getelementptr i32, i32* %p0, i32 2
  ret i32* %p1
}

; CHECK-LABEL: ldrd_strd_aa:
; NORMAL: ldrd [[TMP1:r[0-9]]], [[TMP2:r[0-9]]],
; NORMAL: strd [[TMP1]], [[TMP2]],
; CONSERVATIVE-NOT: ldrd
; CONSERVATIVE-NOT: strd
; CHECK: bx lr

define void @ldrd_strd_aa(i32* noalias nocapture %x, i32* noalias nocapture readonly %y) {
entry:
  %0 = load i32, i32* %y, align 4
  store i32 %0, i32* %x, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %y, i32 1
  %1 = load i32, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %x, i32 1
  store i32 %1, i32* %arrayidx3, align 4
  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
