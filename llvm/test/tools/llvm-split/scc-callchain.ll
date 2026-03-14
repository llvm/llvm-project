; All of the functions in this module must end up
; in the same partition.

; RUN: llvm-split -j=2 -preserve-locals -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK0 %s

; CHECK0: declare dso_local i32 @funInternal0
; CHECK0: declare dso_local i32 @funInternal1
; CHECK0: declare dso_local i32 @funInternal2
; CHECK0: declare i32 @funExternal

; All functions are in the same file.
; Local functions are still local.
; CHECK1: define internal i32 @funInternal0
; CHECK1: define internal i32 @funInternal1
; CHECK1: define internal i32 @funInternal2
; CHECK1: define i32 @funExternal
; CHECK1: define i32 @funExternal2

define internal i32 @funInternal0() {
entry:
  ret i32 0
}

define internal i32 @funInternal1() {
entry:
  %x = call i32 @funInternal0()
  ret i32 %x
}

define internal i32 @funInternal2() {
entry:
  %x = call i32 @funInternal1()
  ret i32 %x
}

define i32 @funExternal() {
entry:
  %x = call i32 @funInternal2()
  ret i32 %x
}

define i32 @funExternal2() {
entry:
  %x = call i32 @funInternal0()
  ret i32 %x
}
