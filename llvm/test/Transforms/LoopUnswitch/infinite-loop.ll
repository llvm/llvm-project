; RUN: opt -loop-unswitch -disable-output -stats -info-output-file - < %s | FileCheck --check-prefix=STATS %s
; RUN: opt -loop-unswitch -simplifycfg -S < %s | FileCheck %s
; PR5373

; Loop unswitching shouldn't trivially unswitch the true case of condition %a
; in the code here because it leads to an infinite loop. While this doesn't
; contain any instructions with side effects, it's still a kind of side effect.
; It can trivially unswitch on the false cas of condition %a though.

; STATS: 2 loop-unswitch - Number of branches unswitched
; STATS: 1 loop-unswitch - Number of unswitches that are trivial

; CHECK: @func_16
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %a, label %entry.split, label %abort0.split

; CHECK: entry.split:
; CHECK-NEXT: br i1 %b, label %cond.end.us, label %abort1

; CHECK: cond.end.us:
; CHECK-NEXT: br label %cond.end.us

; CHECK: abort0.split:
; CHECK-NEXT: call void @end0() noreturn nounwind
; CHECK-NEXT: unreachable

; CHECK: abort1:
; CHECK-NEXT: call void @end1() noreturn nounwind
; CHECK-NEXT: unreachable

; CHECK: }

define void @func_16(i1 %a, i1 %b) nounwind {
entry:
  br label %for.body

for.body:
  br i1 %a, label %cond.end, label %abort0

cond.end:
  br i1 %b, label %for.body, label %abort1

abort0:
  call void @end0() noreturn nounwind
  unreachable

abort1:
  call void @end1() noreturn nounwind
  unreachable
}

declare void @end0() noreturn
declare void @end1() noreturn
