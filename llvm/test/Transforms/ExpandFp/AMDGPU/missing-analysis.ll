; RUN: not opt -mtriple=amdgcn -passes=expand-fp -disable-output %s 2>&1 | FileCheck %s

; CHECK: 'runtime-libcall-info' analysis required
define void @empty() {
  ret void
}
