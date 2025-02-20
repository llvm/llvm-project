; REQUIRES: asserts
; RUN: opt -passes='no-op-cgscc' -disable-output -stats < %s 2>&1 | FileCheck %s --check-prefix=ONE
; RUN: opt -passes='cgscc(instcombine)' -disable-output -stats < %s 2>&1 | FileCheck %s --check-prefix=TWO

; ONE: 1 cgscc - Number of functions in the largest SCC
; TWO: 2 cgscc - Number of functions in the largest SCC

@g1 = constant ptr @f1
@g2 = constant ptr @f2

define void @f1() {
  %f = load ptr, ptr @g2
  call void %f()
  ret void
}

define void @f2() {
  %f = load ptr, ptr @g1
  call void %f()
  ret void
}
