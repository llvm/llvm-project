; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define target("amdgpu.stridemark") @f() {
entry:
  ret target("amdgpu.stridemark") poison
}
; CHECK: Function returns a token but isn't an intrinsic
