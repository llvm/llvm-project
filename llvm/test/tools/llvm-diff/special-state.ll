; Diff file with itself, assert no difference by return code
; RUN: llvm-diff %s %s

; Replace %newvar1 with %newvar2 in the phi node. This can only
; be detected to be different once BB1 has been processed.
; RUN: rm -f %t.ll
; RUN: cat %s | sed -e 's/i16/i8/' > %t.ll
; RUN: not llvm-diff %s %t.ll 2>&1 | FileCheck %s

; CHECK:      in function alloca:
; CHECK-NEXT:   in block %0 / %0:
; CHECK-NEXT:     >   %1 = alloca i8, align 1
; CHECK-NEXT:     >   ret ptr %1
; CHECK-NEXT:     <   %1 = alloca i16, align 1
; CHECK-NEXT:     <   ret ptr %1

define ptr @alloca() {
  %1 = alloca i16, align 1
  ret ptr %1
}

; CHECK:      in function load:
; CHECK-NEXT:   in block %0 / %0:
; CHECK-NEXT:     >   %1 = load i8, ptr %addr, align 4
; CHECK-NEXT:     <   %1 = load i16, ptr %addr, align 4

define void @load(ptr %addr) {
  %1 = load i16, ptr %addr, align 4
  ret void
}

; CHECK:      in function gep:
; CHECK-NEXT:   in block %0 / %0:
; CHECK-NEXT:     >   %1 = getelementptr %struct.ty1.0, ptr %addr, i32 0
; CHECK-NEXT:     <   %1 = getelementptr %struct.ty1, ptr %addr, i32 0

%struct.ty1 = type { i16 }

define void @gep(ptr %addr) {
  %1 = getelementptr %struct.ty1, ptr %addr, i32 0
  ret void
}
