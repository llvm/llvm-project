; RUN: llc -mtriple riscv32-unknown-elf -mattr=+experimental-xsfmclic \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32
; RUN: llc -mtriple riscv64-unknown-elf -mattr=+experimental-xsfmclic \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64

@size = external global i32

declare void @use(ptr)

define void @preemptible() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="all" {
; RV32-LABEL: preemptible:
; RV32:         sw t0, 16(sp)
; RV32:         .cfi_offset t0, -16
; RV32:         csrr t0, mcause
; RV32-NEXT:    sw t0, 28(sp)
; RV32-NEXT:    csrr t0, mepc
; RV32-NEXT:    csrsi mstatus, 8
; RV32-NEXT:    sw t0, 24(sp)
; RV32:         sw s0, 12(sp)
; RV32:         .cfi_offset s0, -20
; RV32:         addi s0, sp, 32
; RV32:         lw s0, 12(sp)
; RV32:         lw t0, 24(sp)
; RV32-NEXT:    csrci mstatus, 8
; RV32-NEXT:    csrw mepc, t0
; RV32-NEXT:    lw t0, 28(sp)
; RV32-NEXT:    csrw mcause, t0
; RV32:         lw t0, 16(sp)
; RV32:         .cfi_restore t0
; RV32:         .cfi_restore s0
; RV32:         mret
;
; RV64-LABEL: preemptible:
; RV64:         sd t0, 16(sp)
; RV64:         .cfi_offset t0, -32
; RV64:         csrr t0, mcause
; RV64-NEXT:    sd t0, 40(sp)
; RV64-NEXT:    csrr t0, mepc
; RV64-NEXT:    csrsi mstatus, 8
; RV64-NEXT:    sd t0, 32(sp)
; RV64:         sd s0, 8(sp)
; RV64:         .cfi_offset s0, -40
; RV64:         addi s0, sp, 48
; RV64:         ld s0, 8(sp)
; RV64:         ld t0, 32(sp)
; RV64-NEXT:    csrci mstatus, 8
; RV64-NEXT:    csrw mepc, t0
; RV64-NEXT:    ld t0, 40(sp)
; RV64-NEXT:    csrw mcause, t0
; RV64:         ld t0, 16(sp)
; RV64:         .cfi_restore t0
; RV64:         .cfi_restore s0
; RV64:         mret
  ret void
}

define void @preemptible_stack_swap() "interrupt"="SiFive-CLIC-preemptible-stack-swap" "frame-pointer"="all" {
; RV32-LABEL: preemptible_stack_swap:
; RV32:         csrrw sp, sf.mscratchcsw, sp
; RV32:         csrr t0, mcause
; RV32:         csrsi mstatus, 8
; RV32:         addi s0, sp, 32
; RV32:         lw s0, 12(sp)
; RV32:         csrci mstatus, 8
; RV32:         csrw mepc, t0
; RV32:         csrw mcause, t0
; RV32:         csrrw sp, sf.mscratchcsw, sp
; RV32-NEXT:    mret
;
; RV64-LABEL: preemptible_stack_swap:
; RV64:         csrrw sp, sf.mscratchcsw, sp
; RV64:         csrr t0, mcause
; RV64:         csrsi mstatus, 8
; RV64:         addi s0, sp, 48
; RV64:         ld s0, 8(sp)
; RV64:         csrci mstatus, 8
; RV64:         csrw mepc, t0
; RV64:         csrw mcause, t0
; RV64:         csrrw sp, sf.mscratchcsw, sp
; RV64-NEXT:    mret
  ret void
}

define void @preemptible_var_alloca() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="none" {
; RV32-LABEL: preemptible_var_alloca:
; RV32:         csrr t0, mcause
; RV32-NEXT:    sw t0, 76(sp)
; RV32-NEXT:    csrr t0, mepc
; RV32-NEXT:    csrsi mstatus, 8
; RV32-NEXT:    sw t0, 72(sp)
; RV32:         addi s0, sp, 80
; RV32:         sub a0, sp, a0
; RV32:         mv sp, a0
; RV32:         call use
; RV32:         addi sp, s0, -80
; RV32:         lw t0, 72(sp)
; RV32-NEXT:    csrci mstatus, 8
; RV32-NEXT:    csrw mepc, t0
; RV32-NEXT:    lw t0, 76(sp)
; RV32-NEXT:    csrw mcause, t0
; RV32:         mret
;
; RV64-LABEL: preemptible_var_alloca:
; RV64:         csrr t0, mcause
; RV64-NEXT:    sd t0, 152(sp)
; RV64-NEXT:    csrr t0, mepc
; RV64-NEXT:    csrsi mstatus, 8
; RV64-NEXT:    sd t0, 144(sp)
; RV64:         addi s0, sp, 160
; RV64:         sub a0, sp, a0
; RV64:         mv sp, a0
; RV64:         call use
; RV64:         addi sp, s0, -160
; RV64:         ld t0, 144(sp)
; RV64-NEXT:    csrci mstatus, 8
; RV64-NEXT:    csrw mepc, t0
; RV64-NEXT:    ld t0, 152(sp)
; RV64-NEXT:    csrw mcause, t0
; RV64:         mret
  %count = load volatile i32, ptr @size
  %object = alloca i8, i32 %count, align 16
  call void @use(ptr %object)
  ret void
}

define void @preemptible_realign() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="none" {
; RV32-LABEL: preemptible_realign:
; RV32:         csrr t0, mcause
; RV32-NEXT:    sw t0,
; RV32-NEXT:    csrr t0, mepc
; RV32-NEXT:    csrsi mstatus, 8
; RV32-NEXT:    sw t0,
; RV32:         addi s0, sp,
; RV32:         andi sp, sp, -64
; RV32:         call use
; RV32:         addi sp, s0,
; RV32:         lw t0, 184(sp)
; RV32-NEXT:    csrci mstatus, 8
; RV32-NEXT:    csrw mepc, t0
; RV32-NEXT:    lw t0, 188(sp)
; RV32-NEXT:    csrw mcause, t0
; RV32:         mret
;
; RV64-LABEL: preemptible_realign:
; RV64:         csrr t0, mcause
; RV64-NEXT:    sd t0,
; RV64-NEXT:    csrr t0, mepc
; RV64-NEXT:    csrsi mstatus, 8
; RV64-NEXT:    sd t0,
; RV64:         addi s0, sp,
; RV64:         andi sp, sp, -64
; RV64:         call use
; RV64:         addi sp, s0,
; RV64:         ld t0, 240(sp)
; RV64-NEXT:    csrci mstatus, 8
; RV64-NEXT:    csrw mepc, t0
; RV64-NEXT:    ld t0, 248(sp)
; RV64-NEXT:    csrw mcause, t0
; RV64:         mret
  %object = alloca [64 x i8], align 64
  call void @use(ptr %object)
  ret void
}

define void @large_frame_fp() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="all" {
; RV32-LABEL: large_frame_fp:
; RV32:         addi sp, sp, -2032
; RV32-NEXT:    .cfi_def_cfa_offset 2032
; RV32-NEXT:    sw t0, 2016(sp)
; RV32-NEXT:    .cfi_offset t0, -16
; RV32-NEXT:    csrr t0, mcause
; RV32-NEXT:    sw t0, 2028(sp)
; RV32-NEXT:    csrr t0, mepc
; RV32-NEXT:    csrsi mstatus, 8
; RV32-NEXT:    sw t0, 2024(sp)
; RV32:         call use
; RV32:         lw ra, 2020(sp)
; RV32:         lw t0, 2024(sp)
; RV32-NEXT:    csrci mstatus, 8
; RV32-NEXT:    csrw mepc, t0
; RV32-NEXT:    lw t0, 2028(sp)
; RV32-NEXT:    csrw mcause, t0
; RV32-NEXT:    lw t0, 2016(sp)
; RV32:         .cfi_restore t0
; RV32:         mret
;
; RV64-LABEL: large_frame_fp:
; RV64:         addi sp, sp, -2032
; RV64-NEXT:    .cfi_def_cfa_offset 2032
; RV64-NEXT:    sd t0, 2000(sp)
; RV64-NEXT:    .cfi_offset t0, -32
; RV64-NEXT:    csrr t0, mcause
; RV64-NEXT:    sd t0, 2024(sp)
; RV64-NEXT:    csrr t0, mepc
; RV64-NEXT:    csrsi mstatus, 8
; RV64-NEXT:    sd t0, 2016(sp)
; RV64:         call use
; RV64:         ld ra, 2008(sp)
; RV64:         ld t0, 2016(sp)
; RV64-NEXT:    csrci mstatus, 8
; RV64-NEXT:    csrw mepc, t0
; RV64-NEXT:    ld t0, 2024(sp)
; RV64-NEXT:    csrw mcause, t0
; RV64-NEXT:    ld t0, 2000(sp)
; RV64:         .cfi_restore t0
; RV64:         mret
  %object = alloca [4096 x i8], align 16
  call void @use(ptr %object)
  ret void
}

define void @realign_var_alloca() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="none" {
; RV32-LABEL: realign_var_alloca:
; RV32:         addi sp, sp, -128
; RV32-NEXT:    .cfi_def_cfa_offset 128
; RV32-NEXT:    sw t0, 112(sp)
; RV32-NEXT:    .cfi_offset t0, -16
; RV32-NEXT:    csrr t0, mcause
; RV32-NEXT:    sw t0, 124(sp)
; RV32-NEXT:    csrr t0, mepc
; RV32-NEXT:    csrsi mstatus, 8
; RV32-NEXT:    sw t0, 120(sp)
; RV32:         call use
; RV32:         lw ra, 116(sp)
; RV32:         lw t0, 120(sp)
; RV32-NEXT:    csrci mstatus, 8
; RV32-NEXT:    csrw mepc, t0
; RV32-NEXT:    lw t0, 124(sp)
; RV32-NEXT:    csrw mcause, t0
; RV32-NEXT:    lw t0, 112(sp)
; RV32:         .cfi_restore t0
; RV32:         mret
;
; RV64-LABEL: realign_var_alloca:
; RV64:         addi sp, sp, -192
; RV64-NEXT:    .cfi_def_cfa_offset 192
; RV64-NEXT:    sd t0, 160(sp)
; RV64-NEXT:    .cfi_offset t0, -32
; RV64-NEXT:    csrr t0, mcause
; RV64-NEXT:    sd t0, 184(sp)
; RV64-NEXT:    csrr t0, mepc
; RV64-NEXT:    csrsi mstatus, 8
; RV64-NEXT:    sd t0, 176(sp)
; RV64:         call use
; RV64:         ld ra, 168(sp)
; RV64:         ld t0, 176(sp)
; RV64-NEXT:    csrci mstatus, 8
; RV64-NEXT:    csrw mepc, t0
; RV64-NEXT:    ld t0, 184(sp)
; RV64-NEXT:    csrw mcause, t0
; RV64-NEXT:    ld t0, 160(sp)
; RV64:         .cfi_restore t0
; RV64:         mret
  %count = load volatile i32, ptr @size
  %object = alloca i8, i32 %count, align 64
  call void @use(ptr %object)
  ret void
}
