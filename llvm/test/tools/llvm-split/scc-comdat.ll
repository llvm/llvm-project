; All functions in the same comdat group must
; be in the same module

; RUN: llvm-split -j=2 -preserve-locals -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK0 %s

; CHECK0: declare i32 @fun1
; CHECK0: declare i32 @fun2
; CHECK0: declare i32 @fun3

; CHECK1: define internal i32 @fun1
; CHECK1: define internal i32 @fun2
; CHECK1: define i32 @fun3

$fun = comdat any

define internal i32 @fun1() section ".text.funs" comdat($fun) {
entry:
  ret i32 0
}

define internal i32 @fun2() section ".text.funs" comdat($fun) {
entry:
  ret i32 0
}

define i32 @fun3() section ".text.funs" comdat($fun) {
entry:
  ret i32 0
}

