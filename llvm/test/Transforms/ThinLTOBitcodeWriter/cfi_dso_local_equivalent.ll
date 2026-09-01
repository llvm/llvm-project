; REQUIRES: x86-registered-target

; Verify that the dso_local_equivalent use of f becomes a dso_local_equivalent
; use of the alias to f that is introduced for CFI.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

target triple = "x86_64-unknown-linux-gnu"

; M0: @vtable.{{[0-9a-f]+}} = external hidden constant [1 x ptr]
; M0: @f.{{[0-9a-f]+}} = hidden alias ptr, ptr @f
; M0: define internal void @f()
; M0: define ptr @use()
; M0-NEXT: ret ptr @vtable.{{[0-9a-f]+}}

; M1: @vtable = internal constant [1 x ptr] [ptr dso_local_equivalent @f.{{[0-9a-f]+}}]
; M1: @vtable.{{[0-9a-f]+}} = hidden alias ptr, ptr @vtable
; M1: declare !guid !{{[0-9]+}} hidden void @f.{{[0-9a-f]+}}()
@vtable = internal constant [1 x ptr] [
  ptr dso_local_equivalent @f
], !type !0

define internal void @f() {
  ret void
}

define ptr @use() {
  ret ptr @vtable
}

!0 = !{i32 0, !"typeid"}
