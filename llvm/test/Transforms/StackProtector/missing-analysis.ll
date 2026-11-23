; REQUIRES: x86-registered-target
; RUN: not opt -mtriple=x86_64-- -passes=stack-protector -disable-output %s 2>&1 | FileCheck %s

; CHECK: 'LibcallLoweringModuleAnalysis' analysis required
define void @empty() sspreq {
  ret void
}
