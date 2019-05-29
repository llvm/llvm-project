; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

;; Global variables cannot be scalable vectors, since we don't
;; know the size at compile time.

; CHECK: Globals cannot contain scalable vectors
; CHECK-NEXT: <vscale x 4 x i32>* @ScalableVecGlobal
@ScalableVecGlobal = global <vscale x 4 x i32> zeroinitializer

; CHECK: Globals cannot contain scalable vectors
; CHECK-NEXT: [64 x <vscale x 2 x double>]* @ScalableVecGlobalArray
@ScalableVecGlobalArray = global [64 x <vscale x 2 x double>] zeroinitializer

; CHECK: Globals cannot contain scalable vectors
; CHECK-NEXT: { <vscale x 16 x i64>, <vscale x 16 x i1> }* @ScalableVecGlobalStruct
@ScalableVecGlobalStruct = global { <vscale x 16 x i64>, <vscale x 16 x i1> } zeroinitializer

; CHECK: Globals cannot contain scalable vectors
; CHECK-NEXT: { [4 x i32], [2 x { <vscale x 4 x i64>, <vscale x 32 x i8> }] }* @ScalableVecMixed
@ScalableVecMixed = global { [4 x i32], [2 x { <vscale x 4 x i64>,  <vscale x 32 x i8> }]} zeroinitializer

;; Global _pointers_ to scalable vectors are fine
; CHECK-NOT: Globals cannot contain scalable vectors
@ScalableVecPtr = global <vscale x 8 x i16>* zeroinitializer
