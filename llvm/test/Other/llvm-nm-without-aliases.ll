; RUN: llvm-as %s -o %t
; RUN: llvm-nm --without-aliases %t | FileCheck %s
; RUN: llvm-nm %t | FileCheck --check-prefix=WITH %s

; CHECK-NOT: T a0bar
; CHECK-NOT: T a0foo
; CHECK: T bar
; CHECK: T foo

; WITH: T a0bar
; WITH: T a0foo
; WITH: T bar
; WITH: T foo

@a0foo = alias void (), ptr @foo

define void @foo() {
  ret void
}

@a0bar = alias void (), ptr @bar

define void @bar() {
  ret void
}
