; The architecture name was renamed from "amdgcn" to "amdgpu". Check that the
; legacy "amdgcn" name still works when passed via -march.

; RUN: llc -march=amdgcn -mcpu=gfx900 < %s | FileCheck %s

; CHECK: f:
; CHECK: s_setpc_b64
define void @f() {
  ret void
}
