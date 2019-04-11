; RUN: llc -mtriple=riscv32 < %s | FileCheck -check-prefix=RV32 %s
; RUN: llc -mtriple=riscv64 < %s | FileCheck -check-prefix=RV64 %s

@v = dso_local global i32 0, align 4
@r = dso_local global i64 7, align 8

; SmallDataLimit set to 0, so we expect no data will put in sbss and sdata.
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"SmallDataLimit", i32 0}

; RV32-NOT:    .section        .sbss
; RV32-NOT:    .section        .sdata
; RV64-NOT:    .section        .sbss
; RV64-NOT:    .section        .sdata
