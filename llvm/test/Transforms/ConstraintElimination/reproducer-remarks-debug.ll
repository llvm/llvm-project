; RUN: opt -passes=constraint-elimination -constraint-elimination-dump-reproducers -pass-remarks=constraint-elimination -debug %s 2>&1 | FileCheck %s

; REQUIRES: asserts

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; CHECK:      Condition icmp eq ptr %a, null implied by dominating constraints
; CHECK-NEXT: %a <= 0
; CHECK-NEXT: Creating reproducer for   %c.2 = icmp eq ptr %a, null
; CHECK-NEXT:   found external input ptr %a
; CHECK-NEXT:   Materializing assumption icmp eq ptr %a, null

define i1 @test_ptr_null_constant(ptr %a) {
; CHECK-LABEL: define i1 @"{{.+}}test_ptr_null_constantrepro"(ptr %a) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = icmp eq ptr %a, null
; CHECK-NEXT:   call void @llvm.assume(i1 %0)
; CHECK-NEXT:   %c.2 = icmp eq ptr %a, null
; CHECK-NEXT:   ret i1 %c.2
; CHECK-NEXT: }
;
entry:
  %c.1 = icmp eq ptr %a, null
  br i1 %c.1, label %then, label %else

then:
  %c.2 = icmp eq ptr %a, null
  ret i1 %c.2

else:
  ret i1 false
}
