; RUN: llc -mtriple=riscv32 -data-sections < %s | FileCheck -check-prefix=RV32 %s
; RUN: llc -mtriple=riscv64 -data-sections < %s | FileCheck -check-prefix=RV64 %s

; Append an unique name to each sdata/sbss section when -data-section.

@v = dso_local global i32 0, align 4
@r = dso_local global i64 7, align 8

; If a symbol has an explicit section name, we should honor it.
@vv = dso_local global i32 0, section ".sbss", align 4
@rr = dso_local global i64 7, section ".sdata", align 8
@bb = dso_local global i32 0, section ".sbss_like", align 4
@tt = dso_local global i64 7, section ".sdata_like", align 8
@nn = dso_local global i32 0, section ".custom_a", align 4
@yy = dso_local global i64 7, section ".custom_b", align 8

; SmallDataLimit set to 8, so we expect @v will be put in sbss
; and @r will be put in sdata.
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"SmallDataLimit", i32 8}

; RV32:    .section        .sbss.v,"aw"
; RV32:    .section        .sdata.r,"aw"
; RV32:    .section        .sbss,"aw"
; RV32:    .section        .sdata,"aw"
; RV32:    .section        .sbss_like,"aw"
; RV32:    .section        .sdata_like,"aw"
; RV32:    .section        .custom_a,"aw"
; RV32:    .section        .custom_b,"aw"

; RV64:    .section        .sbss.v,"aw"
; RV64:    .section        .sdata.r,"aw"
; RV64:    .section        .sbss,"aw"
; RV64:    .section        .sdata,"aw"
; RV64:    .section        .sbss_like,"aw"
; RV64:    .section        .sdata_like,"aw"
; RV64:    .section        .custom_a,"aw"
; RV64:    .section        .custom_b,"aw"
