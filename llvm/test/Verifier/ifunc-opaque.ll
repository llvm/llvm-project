; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define ptr @resolver() {
  ret ptr null
}

; CHECK: IFunc must have a Function resolver
; CHECK-NEXT: ptr @ifunc_getelementptr
@ifunc_getelementptr = ifunc void (), ptr getelementptr (i8, ptr @resolver, i32 4)


; Make sure nothing asserts on an unhandled constantexpr for the
; resolver.

; CHECK: IFunc must have a Function resolver
; CHECK-NEXT: ptr @ifunc_shl
@ifunc_shl = ifunc void (), ptr inttoptr (i64 add (i64 ptrtoint (ptr @resolver to i64), i64 4) to ptr)
