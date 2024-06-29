; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s

define void @test(ptr %a) nounwind {
; CHECK-LABEL: test:
; CHECK: __atomic_load_16
; CHECK: __atomic_store_16
  %1 = load atomic i128, ptr %a seq_cst, align 16
  store atomic i128 %1, ptr %a seq_cst, align 16
  ret void
}
