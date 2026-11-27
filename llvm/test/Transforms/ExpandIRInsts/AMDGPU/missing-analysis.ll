; RUN: not opt -mtriple=amdgpu -passes=expand-ir-insts -disable-output %s 2>&1 | FileCheck %s

; CHECK: 'LibcallLoweringModuleAnalysis' analysis required
define void @empty() {
  ret void
}
