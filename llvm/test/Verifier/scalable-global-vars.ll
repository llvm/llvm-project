; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

;; Global variables cannot be scalable vectors, since we don't
;; know the size at compile time.

; CHECK: Globals cannot contain scalable types
; CHECK-NEXT: ptr @ScalableVecGlobal
@ScalableVecGlobal = global <vscale x 4 x i32> zeroinitializer

; CHECK-NEXT: Globals cannot contain scalable types
; CHECK-NEXT: ptr @ScalableVecArrayGlobal
@ScalableVecArrayGlobal = global [ 8 x  <vscale x 4 x i32> ] zeroinitializer

; CHECK-NEXT: Globals cannot contain scalable types
; CHECK-NEXT: ptr @ScalableVecStructGlobal
@ScalableVecStructGlobal = global { i32,  <vscale x 4 x i32> } zeroinitializer

