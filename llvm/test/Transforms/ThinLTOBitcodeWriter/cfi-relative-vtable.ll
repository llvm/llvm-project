; REQUIRES: x86-registered-target

; Verify that promoting an internal relative vtable in ThinLTOBitcodeWriter
; does not replace the vtable's self-reference in its own initializer with the
; promoted alias, which would break AsmPrinter lowering of dso_local_equivalent.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o %t.m1.bc %t
; RUN: llvm-dis -o - %t.m1.bc | FileCheck --check-prefix=M1 %s
; RUN: llc -filetype=null %t.m1.bc

target triple = "x86_64-unknown-linux-gnu"

; M0: @rel_vtable.{{[0-9a-f]+}} = external hidden constant [1 x i32]
; M0: @f.{{[0-9a-f]+}} = hidden alias void (), ptr @f
; M0: define internal void @f()
; M0: define ptr @use_rel()
; M0-NEXT: ret ptr @rel_vtable.{{[0-9a-f]+}}

; M1: @rel_vtable = internal constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @f.{{[0-9a-f]+}} to i64), i64 ptrtoint (ptr @rel_vtable to i64)) to i32)]
; M1: @rel_vtable.{{[0-9a-f]+}} = hidden alias [1 x i32], ptr @rel_vtable
; M1: declare !guid !{{[0-9]+}} hidden void @f.{{[0-9a-f]+}}()
@rel_vtable = internal constant [1 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @f to i64), i64 ptrtoint (ptr @rel_vtable to i64)) to i32)
], !type !0

define internal void @f() {
  ret void
}

define ptr @use_rel() {
  ret ptr @rel_vtable
}

!0 = !{i32 0, !"typeid"}
