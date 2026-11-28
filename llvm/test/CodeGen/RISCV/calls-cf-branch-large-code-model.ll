; RUN: llc -mtriple=riscv64 -global-isel -global-isel-abort=0 -code-model=large -verify-machineinstrs %s -o - | FileCheck %s

define dso_local void @bar() #0 {
entry:
  call void @foo()
  ret void
}
declare void @foo() #1

attributes #0 = { "cfi_type_hash"="308871" }
attributes #1 = { "cfi_type_hash"="308871" }

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"} 
!2 = !{i32 1, !"Code Model", i32 4}

; CHECK: lpad 308871
; CHECK: auipc a0, %pcrel_hi(.LCPI0_0)
; CHECK: ld t2, %pcrel_lo(.Lpcrel_hi0)(a0)
; CHECK: jalr t2