; Regression test for https://github.com/llvm/llvm-project/issues/128560 -
; check that cbuffers are populated correctly when there aren't any other kinds
; of resource.

; RUN: opt -S -passes=dxil-translate-metadata %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define void @cbuffer_is_only_binding() {
  %cbuf = call target("dx.CBuffer", target("dx.Layout", {float}, 4, 0))
      @llvm.dx.resource.handlefrombinding(i32 1, i32 8, i32 1, i32 0, ptr null)
  ; CHECK: %CBuffer = type { float }

  ret void
}

; CHECK:      @[[CB0:.*]] = external constant %CBuffer

; CHECK: !{i32 0, ptr @[[CB0]], !""
