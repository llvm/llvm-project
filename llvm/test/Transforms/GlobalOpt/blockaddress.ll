; RUN: opt < %s -globalopt -S | FileCheck %s

@x = internal global i8* zeroinitializer

define void @f() {
; CHECK: @f

; Check that we don't hit an assert in Constant::IsThreadDependent()
; when storing this blockaddress into a global.

  store i8* blockaddress(@g, %here), i8** @x, align 8
  ret void
}

define void @g() {
; CHECK: @g

here:
  ret void
}
