; RUN: llc -mtriple=riscv32 --code-model=small \
; RUN:    -stop-after riscv-prera-expand-pseudo %s -o %t.mir
; RUN: llc -mtriple=riscv32 -run-pass riscv-expand-pseudo %t.mir -o - | \
; RUN:   FileCheck %s -check-prefix=RV32-SMALL
;
; RUN: llc -mtriple=riscv32 --code-model=medium --relocation-model=pic \
; RUN:   -stop-after riscv-prera-expand-pseudo %s -o %t.mir
; RUN: llc -mtriple=riscv32 -run-pass riscv-expand-pseudo %t.mir -o - | \
; RUN:   FileCheck %s -check-prefix=RV32-MED

; This tests the RISC-V-specific serialization and deserialization of
; `target-flags(...)`

@g_e = external global i32
@g_i = internal global i32 0
@t_un = external thread_local global i32
@t_ld = external thread_local(localdynamic) global i32
@t_ie = external thread_local(initialexec) global i32
@t_le = external thread_local(localexec) global i32

declare i32 @callee(i32) nounwind

define i32 @caller(i32 %a) nounwind {
; RV32-SMALL-LABEL: name: caller
; RV32-SMALL:      target-flags(riscv-hi) @g_e
; RV32-SMALL-NEXT: target-flags(riscv-lo) @g_e
; RV32-SMALL:      target-flags(riscv-hi) @g_i
; RV32-SMALL-NEXT: target-flags(riscv-lo) @g_i
; RV32-SMALL:      target-flags(riscv-tls-got-hi) @t_un
; RV32-SMALL-NEXT: target-flags(riscv-pcrel-lo) <mcsymbol .Lpcrel_hi0>
; RV32-SMALL:      target-flags(riscv-tls-got-hi) @t_ld
; RV32-SMALL-NEXT: target-flags(riscv-pcrel-lo) <mcsymbol .Lpcrel_hi1>
; RV32-SMALL:      target-flags(riscv-tls-got-hi) @t_ie
; RV32-SMALL-NEXT: target-flags(riscv-pcrel-lo) <mcsymbol .Lpcrel_hi2>
; RV32-SMALL:      target-flags(riscv-tprel-hi) @t_le
; RV32-SMALL-NEXT: target-flags(riscv-tprel-add) @t_le
; RV32-SMALL-NEXT: target-flags(riscv-tprel-lo) @t_le
; RV32-SMALL:      target-flags(riscv-plt) @callee
;
; RV32-MED-LABEL: name: caller
; RV32-MED:      target-flags(riscv-got-hi) @g_e
; RV32-MED-NEXT: target-flags(riscv-pcrel-lo) <mcsymbol .Lpcrel_hi0>
; RV32-MED:      target-flags(riscv-pcrel-hi) @g_i
; RV32-MED-NEXT: target-flags(riscv-pcrel-lo) <mcsymbol .Lpcrel_hi1>
; RV32-MED:      target-flags(riscv-tls-gd-hi) @t_un
; RV32-MED-NEXT: target-flags(riscv-pcrel-lo) <mcsymbol .Lpcrel_hi2>
; RV32-MED:      target-flags(riscv-plt) &__tls_get_addr
; RV32-MED:      target-flags(riscv-tls-gd-hi) @t_ld
; RV32-MED-NEXT: target-flags(riscv-pcrel-lo) <mcsymbol .Lpcrel_hi3>
; RV32-MED:      target-flags(riscv-plt) &__tls_get_addr
; RV32-MED:      target-flags(riscv-tls-got-hi) @t_ie
; RV32-MED-NEXT: target-flags(riscv-pcrel-lo) <mcsymbol .Lpcrel_hi4>
; RV32-MED:      target-flags(riscv-tprel-hi) @t_le
; RV32-MED-NEXT: target-flags(riscv-tprel-add) @t_le
; RV32-MED-NEXT: target-flags(riscv-tprel-lo) @t_le
; RV32-MED:      target-flags(riscv-plt) @callee
;
  %b = load i32, ptr @g_e
  %c = load i32, ptr @g_i
  %d = load i32, ptr @t_un
  %e = load i32, ptr @t_ld
  %f = load i32, ptr @t_ie
  %g = load i32, ptr @t_le
  %sum = bitcast i32 0 to i32
  %sum.a = add i32 %sum, %a
  %sum.b = add i32 %sum.a, %b
  %sum.c = add i32 %sum.b, %c
  %sum.d = add i32 %sum.c, %d
  %sum.e = add i32 %sum.d, %e
  %sum.f = add i32 %sum.e, %f
  %sum.g = add i32 %sum.f, %g
  %retval = call i32 @callee(i32 %sum.g)
  ret i32 %retval
}
