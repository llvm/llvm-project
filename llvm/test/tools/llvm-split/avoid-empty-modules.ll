; RUN: llvm-split -o %t %s -j 2 -avoid-empty-modules
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-NOT: define
; CHECK0: declare extern_weak void @e
; CHECK0: define void @A
; CHECK0-NOT: define

; CHECK1-NOT: define
; CHECK1: declare extern_weak void @e
; CHECK1: define void @C
; CHECK1-NOT: define

declare extern_weak void @e(...)

define void @A() {
  ret void
}

define void @C() {
  ret void
}
