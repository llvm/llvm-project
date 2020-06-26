; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; CHECK:      Function: foo:
; CHECK-NEXT:   NoAlias: {}* %p, {}* %q

define void @foo({}* %p, {}* %q) {
  store {} {}, {}* %p
  store {} {}, {}* %q
  ret void
}
