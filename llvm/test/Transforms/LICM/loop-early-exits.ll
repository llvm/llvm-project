; RUN: opt -S -licm < %s | FileCheck %s

declare void @use(i64 %a)
declare void @use_nothrow(i64 %a) nounwind
declare void @use_nothing()

; We can move this udiv out of the loop as it comes before 
; the call instruction that may throw.
define void @throw_header1(i64 %x, i64 %y, i1* %cond) {
; CHECK-LABEL: throw_header1
; CHECK: %div = udiv i64 %x, %y
; CHECK-LABEL: loop
; CHECK: call void @use(i64 %div)
entry:
  br label %loop

loop:                                         ; preds = %entry, %for.inc
  %div = udiv i64 %x, %y
  call void @use(i64 %div)
  br label %loop
}

; We can not move this udiv out of the loop as it comes after 
; the call instruction that may throw.
define void @throw_header2(i64 %x, i64 %y, i1* %cond) {
; CHECK-LABEL: throw_header2
; CHECK-LABEL: loop
; CHECK: call void @use_nothing()
; CHECK: %div = udiv i64 %x, %y
entry:
  br label %loop

loop:                                         ; preds = %entry, %for.inc
  call void @use_nothing()
  %div = udiv i64 %x, %y
  call void @use_nothrow(i64 %div)
  br label %loop
}

; We can move this udiv out of the loop as it comes before 
; the call instruction that may throw.
define void @throw_body1(i64 %x, i64 %y, i1* %cond) {
; CHECK-LABEL: throw_body1
; CHECK: %div = udiv i64 %x, %y
; CHECK-LABEL: loop
entry:
  br label %loop

loop:                                         ; preds = %entry, %for.inc
  br label %body

body:
  %div = udiv i64 %x, %y
  call void @use(i64 %div)
  br i1 undef, label %loop, label %exit

exit:
  ret void
}

; We can not move this udiv out of the loop as it comes after 
; the call instruction that may throw.
define void @throw_body2(i64 %x, i64 %y, i1* %cond) {
; CHECK-LABEL: throw_body2
; CHECK-LABEL: loop
; CHECK: call void @use_nothing()
; CHECK: %div = udiv i64 %x, %y
entry:
  br label %loop

loop:                                         ; preds = %entry, %for.inc
  br label %body

body:
  call void @use_nothing()
  %div = udiv i64 %x, %y
  call void @use_nothrow(i64 %div)
  br i1 undef, label %loop, label %exit

exit:
  ret void
}
