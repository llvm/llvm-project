; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

;; Global variables cannot be scalable vectors, since we don't
;; know the size at compile time.

; CHECK: Globals cannot contain scalable vectors
; CHECK-NEXT: <vscale x 4 x i32>* @ScalableVecGlobal
@ScalableVecGlobal = global <vscale x 4 x i32> zeroinitializer

; CHECK-NEXT: Globals cannot contain scalable vectors
; CHECK-NEXT: { i32, <vscale x 4 x i32> }* @ScalableVecStructGlobal
@ScalableVecStructGlobal = global { i32,  <vscale x 4 x i32> } zeroinitializer

;; Global _pointers_ to scalable vectors are fine
; CHECK-NOT: Globals cannot contain scalable vectors
@ScalableVecPtr = global <vscale x 8 x i16>* zeroinitializer
