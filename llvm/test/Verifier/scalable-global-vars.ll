; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

;; Global variables cannot be scalable vectors, since we don't
;; know the size at compile time.

; CHECK: Globals cannot contain scalable vectors
; CHECK-NEXT: <vscale x 4 x i32>* @ScalableVecGlobal
@ScalableVecGlobal = global <vscale x 4 x i32> zeroinitializer

;; Global _pointers_ to scalable vectors are fine
; CHECK-NOT: Globals cannot contain scalable vectors
@ScalableVecPtr = global <vscale x 8 x i16>* zeroinitializer

;; The following errors don't explicitly mention global variables, but
;; do still guarantee that the error will be caught.
; CHECK-DAG: Arrays cannot contain scalable vectors
; CHECK-DAG:  [64 x <vscale x 2 x double>]; ModuleID = '<stdin>'
@ScalableVecGlobalArray = global [64 x <vscale x 2 x double>] zeroinitializer

; CHECK-DAG: Structs cannot contain scalable vectors
; CHECK-DAG:  { <vscale x 16 x i64>, <vscale x 16 x i1> }; ModuleID = '<stdin>'
@ScalableVecGlobalStruct = global { <vscale x 16 x i64>, <vscale x 16 x i1> } zeroinitializer

; CHECK-DAG: Structs cannot contain scalable vectors
; CHECK-DAG  { <vscale x 4 x i64>, <vscale x 32 x i8> }; ModuleID = '<stdin>'
@ScalableVecMixed = global { [4 x i32], [2 x { <vscale x 4 x i64>,  <vscale x 32 x i8> }]} zeroinitializer
