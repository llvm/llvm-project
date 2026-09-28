; RUN: llc -mtriple=mipsel-sde-elf -relocation-model=static -mattr=+noabicalls -mgpopt < %s \
; RUN: | FileCheck %s
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -relocation-model=static -mgpopt -verify-machineinstrs < %s \
; RUN: | FileCheck %s --check-prefixes=CHECK,MIXED

; Static N64 defaults to noabicalls. Globals keep that section policy even when
; the last function uses abicalls; constant pools follow their owning function.

@i = internal unnamed_addr global i32 0, align 4

define i32 @geti() nounwind readonly {
entry:
; CHECK: lw ${{[0-9]+}}, %gp_rel(i)($gp)
  %0 = load i32, ptr @i, align 4
  ret i32 %0
}

define i32 @geti_abicalls() #1 {
; MIXED-LABEL: geti_abicalls:
; MIXED-DAG: %hi(i)
; MIXED-DAG: %lo(i)
  %v = load i32, ptr @i, align 4
  ret i32 %v
}

; MIXED: .section .sdata,
; MIXED: [[SMALL_CP:\.LCPI[0-9]+_0]]:
define double @constant_noabicalls() #0 {
; MIXED-LABEL: constant_noabicalls:
; MIXED: %gp_rel([[SMALL_CP]])($gp)
  ret double 0x400921FB54442D18
}

; MIXED: .section .rodata.cst8,
; MIXED: [[LARGE_CP:\.LCPI[0-9]+_0]]:
define double @constant_abicalls() #1 {
; MIXED-LABEL: constant_abicalls:
; MIXED: %hi([[LARGE_CP]])
; MIXED: %lo([[LARGE_CP]])
  ret double 0x400921FB54442D18
}

; CHECK: .type i,@object
; CHECK-NEXT: .section .sbss,
; CHECK: i:

attributes #0 = { "target-features"="+noabicalls" }
attributes #1 = { "target-features"="-noabicalls,+sym32" }
