; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f(target("amdgpu.stridemark") %A) {
entry:
  ret void
}
; CHECK: Function takes token but isn't an intrinsic
