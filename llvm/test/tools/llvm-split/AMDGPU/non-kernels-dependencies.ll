; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=DEFINE %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=DEFINE %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=DEFINE %s

; 3 functions with each their own dependencies should go into 3
; distinct partitions.

; CHECK0: define void @C
; CHECK0: define internal void @HelperC

; CHECK1: define void @B
; CHECK1: define internal void @HelperB

; CHECK2: define void @A
; CHECK2: define internal void @HelperA


define void @A() {
  call void @HelperA()
  ret void
}

define internal void @HelperA() {
  ret void
}

define void @B() {
  call void @HelperB()
  ret void
}

define internal void @HelperB() {
  ret void
}

define void @C() {
  call void @HelperC()
  ret void
}

define internal void @HelperC() {
  ret void
}
