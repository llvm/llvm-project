; REQUIRES: asserts
; RUN: opt -simple-loop-unswitch -disable-output -stats -info-output-file - < %s | FileCheck --check-prefix=STATS %s
; RUN: opt -simple-loop-unswitch -S < %s | FileCheck %s
; PR5373

; Loop unswitching shouldn't trivially unswitch the true case of condition %a
; in the code here because it leads to an infinite loop. While this doesn't
; contain any instructions with side effects, it's still a kind of side effect.
; It can trivially unswitch on the false cas of condition %a though.

; STATS: 2 simple-loop-unswitch - Number of branches unswitched
; STATS: 2 simple-loop-unswitch - Number of unswitches that are trivial

; CHECK-LABEL: @func_16(
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %a, label %entry.split, label %abort0

; CHECK: entry.split:
; CHECK-NEXT: br i1 %b, label %entry.split.split, label %abort1

; CHECK: entry.split.split:
; CHECK-NEXT: br label %for.body

; CHECK: for.body:
; CHECK-NEXT: br label %cond.end

; CHECK: cond.end:
; CHECK-NEXT: br label %for.body

; CHECK: abort0:
; CHECK-NEXT: call void @end0() [[NOR_NUW:#[0-9]+]]
; CHECK-NEXT: unreachable

; CHECK: abort1:
; CHECK-NEXT: call void @end1() [[NOR_NUW]]
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

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes #1 = { noreturn }
; CHECK: attributes [[NOR_NUW]] = { noreturn nounwind }
