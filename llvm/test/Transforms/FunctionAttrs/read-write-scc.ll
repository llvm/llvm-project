; RUN: opt -S -function-attrs < %s | FileCheck %s
; RUN: opt -S -passes=function-attrs < %s | FileCheck %s

@i = global i32 0

define void @foo() {
; CHECK-LABEL: define void @foo() #0 {
  store i32 1, ptr @i
  call void @bar()
  ret void
}

define void @bar() {
; CHECK-LABEL: define void @bar() #0 {
  %i = load i32, ptr @i
  call void @foo()
  ret void
}

; CHECK: attributes #0 = { nofree nosync nounwind }
