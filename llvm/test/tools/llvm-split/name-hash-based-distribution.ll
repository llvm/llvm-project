; RUN: llvm-split -o %t %s -j 2
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-NOT: define
; CHECK0: define void @D
; CHECK0-NOT: define

; CHECK1-NOT: define
; CHECK1: define void @A
; CHECK1: define void @B
; CHECK1: define void @C
; CHECK1-NOT: define

define void @A() {
  ret void
}

define void @B() {
  ret void
}

define void @C() {
  ret void
}

define void @D() {
  ret void
}
