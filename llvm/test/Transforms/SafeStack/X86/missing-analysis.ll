; RUN: not opt -mtriple=x86_64-- -passes=safe-stack -disable-output %s 2>&1 | FileCheck %s

; CHECK: 'LibcallLoweringModuleAnalysis' analysis required
define void @empty() safestack {
  ret void
}
