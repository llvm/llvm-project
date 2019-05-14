; RUN: llc -mtriple=riscv32 < %s | FileCheck -check-prefix=RV32 %s
; RUN: llc -mtriple=riscv64 < %s | FileCheck -check-prefix=RV64 %s

@v = internal global i32 0, align 4
@r = internal global i64 7, align 8

; @v and @r are local symbols.
; SmallDataLimit set to 8, so we expect @v will be put in sbss
; and @r will be put in sdata.
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"SmallDataLimit", i32 8}

; RV32:    .section        .sbss
; RV32:    .section        .sdata
; RV64:    .section        .sbss
; RV64:    .section        .sdata
