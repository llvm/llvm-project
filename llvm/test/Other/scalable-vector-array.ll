; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

;; Arrays cannot contain scalable vectors; make sure we detect them even
;; when nested inside other aggregates.

%ty = type { i64, [4 x <vscale x 256 x i1>] }
; CHECK: error: invalid array element type
; CHECK: %ty = type { i64, [4 x <vscale x 256 x i1>] }
