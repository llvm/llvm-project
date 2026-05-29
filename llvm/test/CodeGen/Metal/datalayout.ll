; The Metal target's computed datalayout is fixed by AIR's typed-pointer
; bitcode contract; pin every field so an accidental change in
; Triple::computeDataLayout for `air` is caught immediately.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK: target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
; CHECK: target triple = "air"

define void @kernel(ptr %out) {
entry:
  store i32 0, ptr %out
  ret void
}
