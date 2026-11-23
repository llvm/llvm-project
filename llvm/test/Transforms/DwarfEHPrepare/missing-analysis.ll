; REQUIRES: x86-registered-target
; RUN: not opt -mtriple=x86_64-- -passes=dwarf-eh-prepare -disable-output %s 2>&1 | FileCheck %s

; CHECK: 'LibcallLoweringModuleAnalysis' analysis required
define void @empty() {
  ret void
}
