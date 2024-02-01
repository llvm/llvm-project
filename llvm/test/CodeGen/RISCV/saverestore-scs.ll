;; Check that shadow call stack doesn't interfere with save/restore

; RUN: llc -mtriple=riscv32 < %s | FileCheck %s -check-prefix=RV32I
; RUN: llc -mtriple=riscv64 < %s | FileCheck %s -check-prefix=RV64I
; RUN: llc -mtriple=riscv32 -mattr=+save-restore < %s | FileCheck %s -check-prefix=RV32I-SR
; RUN: llc -mtriple=riscv64 -mattr=+save-restore < %s | FileCheck %s -check-prefix=RV64I-SR
; RUN: llc -mtriple=riscv32 -mattr=+f,+save-restore -target-abi=ilp32f < %s | FileCheck %s -check-prefix=RV32I-FP-SR
; RUN: llc -mtriple=riscv64 -mattr=+f,+d,+save-restore -target-abi=lp64d < %s | FileCheck %s -check-prefix=RV64I-FP-SR

@var2 = global [30 x i32] zeroinitializer

define void @callee_scs() nounwind shadowcallstack {
; RV32I-LABEL: callee_scs:
; RV32I-NOT:     call t0, __riscv_save
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: callee_scs:
; RV64I-NOT:     call t0, __riscv_save
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: callee_scs:
; RV32I-SR:         call t0, __riscv_save_12
; RV32I-SR:         tail __riscv_restore_12
;
; RV64I-SR-LABEL: callee_scs:
; RV64I-SR:         call t0, __riscv_save_12
; RV64I-SR:         tail __riscv_restore_12
;
; RV32I-FP-SR-LABEL: callee_scs:
; RV32I-FP-SR:         call t0, __riscv_save_12
; RV32I-FP-SR:         tail __riscv_restore_12
;
; RV64I-FP-SR-LABEL: callee_scs:
; RV64I-FP-SR:         call t0, __riscv_save_12
; RV64I-FP-SR:         tail __riscv_restore_12
  %val = load [30 x i32], ptr @var2
  store volatile [30 x i32] %val, ptr @var2
  ret void
}
