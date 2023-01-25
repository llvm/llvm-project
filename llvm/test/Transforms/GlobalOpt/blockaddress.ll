; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@x = internal global ptr zeroinitializer

define void @f() {
; CHECK-LABEL: @f(

; Check that we don't hit an assert in Constant::IsThreadDependent()
; when storing this blockaddress into a global.

  store ptr blockaddress(@g, %here), ptr @x, align 8
  ret void
}

define void @g() {
entry:
  br label %here

; CHECK-LABEL: @g(

here:
  ret void
}
