; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

;; Arrays and Structs cannot contain scalable vectors, since we don't
;; know the size at compile time and the container types need to have
;; a known size.

; CHECK-DAG: Arrays cannot contain scalable vectors
; CHECK-DAG:  [2 x { i32, <vscale x 1 x i32> }]; ModuleID = '<stdin>'
; CHECK-DAG: Arrays cannot contain scalable vectors
; CHECK-DAG:  [4 x <vscale x 256 x i1>]; ModuleID = '<stdin>'
; CHECK-DAG: Arrays cannot contain scalable vectors
; CHECK-DAG:  [2 x <vscale x 4 x i32>]; ModuleID = '<stdin>'
; CHECK-DAG: Structs cannot contain scalable vectors
; CHECK-DAG:  { i64, [4 x <vscale x 256 x i1>] }; ModuleID = '<stdin>'
; CHECK-DAG: Structs cannot contain scalable vectors
; CHECK-DAG:  { i32, <vscale x 1 x i32> }; ModuleID = '<stdin>'
; CHECK-DAG: Structs cannot contain scalable vectors
; CHECK-DAG: { <vscale x 16 x i8>, <vscale x 2 x double> }; ModuleID = '<stdin>'
; CHECK-DAG: Structs cannot contain scalable vectors
; CHECK-DAG:  %sty = type { i64, <vscale x 32 x i16> }; ModuleID = '<stdin>'

%sty = type { i64, <vscale x 32 x i16> }

define void @scalable_aggregates() {
  %array = alloca [2 x <vscale x 4 x i32>]
  %struct = alloca { <vscale x 16 x i8>, <vscale x 2 x double> }
  %named_struct = alloca %sty
  %s_in_a = alloca [2 x { i32, <vscale x 1 x i32> } ]
  %a_in_s = alloca { i64, [4 x <vscale x 256 x i1> ] }
  ret void
}