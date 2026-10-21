; RUN: llc -mtriple=riscv32 -global-isel -stop-after=ir-translator -verify-machineinstrs %s -o - | FileCheck %s

define void @test_indirect_label(ptr %fn) "cfi_type_hash"="12345" {
entry:
  call void %fn(), !riscv_lpad_label !0
  ret void
}

!llvm.module.flags = !{!1, !2}
!1 = !{i32 8, !"cf-protection-branch", i32 1}
!2 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!0 = !{i32 12345}

; CHECK: PseudoCALLIndirect %{{[0-9]+}}(p0), 12345, csr_ilp32_lp64, implicit-def dead $x1