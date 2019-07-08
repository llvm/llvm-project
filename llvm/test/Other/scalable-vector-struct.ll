; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

;; Structs cannot contain scalable vectors; make sure we detect them even
;; when nested inside other aggregates.

%ty = type [2 x { i32, <vscale x 1 x i32> }]
; CHECK: error: invalid element type for struct
; CHECK: %ty = type [2 x { i32, <vscale x 1 x i32> }]
