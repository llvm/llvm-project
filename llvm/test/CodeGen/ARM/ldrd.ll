; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -regalloc=fast -optimize-regalloc=0 | FileCheck %s -check-prefix=A8
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-m3 -regalloc=fast -optimize-regalloc=0 | FileCheck %s -check-prefix=M3
; rdar://6949835
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -regalloc=basic | FileCheck %s -check-prefix=BASIC
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -regalloc=greedy | FileCheck %s -check-prefix=GREEDY

; Magic ARM pair hints works best with linearscan / fast.

; Cortex-M3 errata 602117: LDRD with base in list may result in incorrect base
; register when interrupted or faulted.

@b = external global i64*

define i64 @t(i64 %a) nounwind readonly {
entry:
; A8: t:
; A8:   ldrd r2, r3, [r2]

; M3: t:
; M3-NOT: ldrd

	%0 = load i64** @b, align 4
	%1 = load i64* %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
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
; BASIC: @f
; BASIC: %bb
; BASIC: ldrd
; BASIC: str
; GREEDY: @f
; GREEDY: %bb
; GREEDY: ldrd
; GREEDY: str
define void @f(i32* nocapture %a, i32* nocapture %b, i32 %n) nounwind {
entry:
  %0 = add nsw i32 %n, -1                         ; <i32> [#uses=2]
  %1 = icmp sgt i32 %0, 0                         ; <i1> [#uses=1]
  br i1 %1, label %bb, label %return

bb:                                               ; preds = %bb, %entry
  %i.03 = phi i32 [ %tmp, %bb ], [ 0, %entry ]    ; <i32> [#uses=3]
  %scevgep = getelementptr i32* %a, i32 %i.03     ; <i32*> [#uses=1]
  %scevgep4 = getelementptr i32* %b, i32 %i.03    ; <i32*> [#uses=1]
  %tmp = add i32 %i.03, 1                         ; <i32> [#uses=3]
  %scevgep5 = getelementptr i32* %a, i32 %tmp     ; <i32*> [#uses=1]
  %2 = load i32* %scevgep, align 4                ; <i32> [#uses=1]
  %3 = load i32* %scevgep5, align 4               ; <i32> [#uses=1]
  %4 = add nsw i32 %3, %2                         ; <i32> [#uses=1]
  store i32 %4, i32* %scevgep4, align 4
  %exitcond = icmp eq i32 %tmp, %0                ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}
