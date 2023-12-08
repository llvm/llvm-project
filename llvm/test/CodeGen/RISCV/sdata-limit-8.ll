; RUN: llc -mtriple=riscv32 < %s | FileCheck -check-prefix=RV32 %s
; RUN: llc -mtriple=riscv64 < %s | FileCheck -check-prefix=RV64 %s

@v = dso_local global i32 0, align 4
@r = dso_local global i64 7, align 8

; SmallDataLimit set to 8, so we expect @v will be put in sbss
; and @r will be put in sdata.
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"SmallDataLimit", i32 8}

; RV32:    .section        .sbss
; RV32:    .section        .sdata
; RV64:    .section        .sbss
; RV64:    .section        .sdata
