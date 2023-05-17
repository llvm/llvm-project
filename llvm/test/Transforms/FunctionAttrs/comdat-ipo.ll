; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; See PR26774

; CHECK-LABEL: define void @bar(ptr readonly %0) {
define void @bar(ptr readonly) {
  call void @foo(ptr %0)
  ret void
}


; CHECK-LABEL: define linkonce_odr void @foo(ptr readonly %0) {
define linkonce_odr void @foo(ptr readonly) {
  call void @bar(ptr %0)
  ret void
}
