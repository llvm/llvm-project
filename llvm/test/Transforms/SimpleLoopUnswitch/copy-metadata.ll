; RUN: opt < %s -simple-loop-unswitch -verify-memoryssa -S | FileCheck %s

; This test checks if unswitched condition preserve make.implicit metadata.
define i32 @test(i1 %cond) {
; CHECK-LABEL: @test(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %{{.*}}, label %entry.split, label %loop_exit, !make.implicit !0
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br i1 %cond, label %continue, label %loop_exit, !make.implicit !0
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %continue

continue:
  call void @some_func()
  br label %loop_begin
; CHECK:       continue:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  ret i32 0
; CHECK:       loop_exit:
; CHECK-NEXT:    ret
}

declare void @some_func()

!0 = !{}
