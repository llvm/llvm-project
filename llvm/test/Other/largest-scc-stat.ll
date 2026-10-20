; REQUIRES: asserts
; RUN: opt -passes='no-op-cgscc' -disable-output -stats < %s 2>&1 | FileCheck %s --check-prefix=ONE
; RUN: opt -passes='cgscc(instcombine)' -disable-output -stats < %s 2>&1 | FileCheck %s --check-prefix=TWO

; ONE: 1 cgscc - Number of functions in the largest SCC
; TWO: 2 cgscc - Number of functions in the largest SCC

define void @f1() {
  %f = bitcast ptr @f2 to ptr
  call void %f()
  ret void
}

define void @f2() {
  %f = bitcast ptr @f1 to ptr
  call void %f()
  ret void
}
