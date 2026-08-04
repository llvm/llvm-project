; RUN: opt < %s -passes=slsr -S -slsr-basis-distance-threshold=96 | FileCheck %s --check-prefixes=CHECK,REWRITE
; RUN: opt < %s -passes=slsr -S -slsr-basis-distance-threshold=2  | FileCheck %s --check-prefixes=CHECK,SKIP

; The register-pressure cost model skips a rewrite only when BOTH:
;   1. an operand of the candidate is used by a non-rewritable user in
;      another block, and
;   2. the basis' last same-block use is farther than
;      -slsr-basis-distance-threshold from the candidate.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

declare void @foo(i32)
declare void @use(i32)

define void @basis_too_far(i32 %b, i32 %s) {
; CHECK-LABEL: @basis_too_far(
; CHECK:         %t1 = add i32 %b, %s
; CHECK:         %s2 = shl i32 %s, 1
; REWRITE:       %t2 = add i32 %t1, %s
; SKIP:          %t2 = add i32 %b, %s2
entry:
  %t1 = add i32 %b, %s
  call void @foo(i32 %t1)
  call void @foo(i32 %b)
  call void @foo(i32 %b)
  call void @foo(i32 %b)
  %s2 = shl i32 %s, 1
  %t2 = add i32 %b, %s2
  call void @foo(i32 %t2)
  br label %next

next:
  call void @use(i32 %s2)
  ret void
}

define void @same_block_operand(i32 %b, i32 %s) {
; CHECK-LABEL: @same_block_operand(
; CHECK:         %t2 = add i32 %t1, %s
entry:
  %t1 = add i32 %b, %s
  call void @foo(i32 %t1)
  call void @foo(i32 %b)
  call void @foo(i32 %b)
  call void @foo(i32 %b)
  %s2 = shl i32 %s, 1
  %t2 = add i32 %b, %s2
  call void @foo(i32 %t2)
  call void @use(i32 %s2)
  ret void
}
