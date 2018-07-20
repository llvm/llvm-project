; RUN: opt < %s -functionattrs -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; See PR26774

; CHECK-LABEL: define void @bar(i8* readonly) {
define void @bar(i8* readonly) {
  call void @foo(i8* %0)
  ret void
}


; CHECK-LABEL: define linkonce_odr void @foo(i8* readonly) {
define linkonce_odr void @foo(i8* readonly) {
  call void @bar(i8* %0)
  ret void
}
