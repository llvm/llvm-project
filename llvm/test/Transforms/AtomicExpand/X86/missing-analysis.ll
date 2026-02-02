; RUN: not opt -mtriple=x86_64-- -passes=atomic-expand -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: 'LibcallLoweringModuleAnalysis' analysis required
define i32 @test(ptr %ptr, i32 %value) {
  %res = atomicrmw xchg ptr %ptr, i32 %value seq_cst
  ret i32 %res
}
