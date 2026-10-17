; RUN: not opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL only supports frexp on 32-bit floats.
; CHECK: LLVM ERROR: frexp is only supported for 32-bit floats in DXIL

define noundef half @frexp_half(half noundef %a) {
entry:
  %r = call { half, i32 } @llvm.frexp.f16.i32(half %a)
  %m = extractvalue { half, i32 } %r, 0
  ret half %m
}
