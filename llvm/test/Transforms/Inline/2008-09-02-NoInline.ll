; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

define i32 @fn2() noinline {
; CHECK-LABEL: define i32 @fn2()
entry:
  ret i32 1
}

define i32 @fn3() {
; CHECK-LABEL: define i32 @fn3()
entry:
  %r = call i32 @fn2()
; CHECK: call i32 @fn2()

  ret i32 %r
}
