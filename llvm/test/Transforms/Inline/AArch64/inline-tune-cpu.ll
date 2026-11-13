; RUN: opt < %s -mtriple=aarch64-apple-macosx -S -passes=inline | FileCheck %s
; Check that a different tune CPU does not prevent inlining, while a different
; feature set still does.

declare void @ext()

define void @callee_m1() #0 {
  call void @ext()
  ret void
}

define void @callee_tune_m5() #1 {
  call void @ext()
  ret void
}

define void @callee_tune_m5_sve() #2 {
  call void @ext()
  ret void
}

define void @caller_m1() #0 {
; CHECK-LABEL: define void @caller_m1(
; CHECK-NEXT:    call void @ext()
; CHECK-NEXT:    call void @callee_tune_m5_sve()
; CHECK-NEXT:    ret void
  call void @callee_tune_m5()
  call void @callee_tune_m5_sve()
  ret void
}

define void @caller_tune_m5() #1 {
; CHECK-LABEL: define void @caller_tune_m5(
; CHECK-NEXT:    call void @ext()
; CHECK-NEXT:    ret void
  call void @callee_m1()
  ret void
}

attributes #0 = { "target-cpu"="apple-m1" }
attributes #1 = { "target-cpu"="apple-m1" "tune-cpu"="apple-m5" }
attributes #2 = { "target-cpu"="apple-m1" "tune-cpu"="apple-m5" "target-features"="+sve" }
