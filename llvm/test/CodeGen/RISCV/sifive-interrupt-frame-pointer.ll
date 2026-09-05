; RUN: llc -mtriple riscv32-unknown-elf -mattr=+experimental-xsfmclic \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32
; RUN: llc -mtriple riscv64-unknown-elf -mattr=+experimental-xsfmclic \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64

@size = external global i32

declare void @use(ptr)

define void @preemptible() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="all" {
; RV32-LABEL: preemptible:
; RV32:         sw t0, 16(sp)
; RV32:         sw s0, 12(sp)
; RV32:         addi s0, sp, 32
; RV32:         csrr t0, mcause
; RV32-NEXT:    sw t0, 28(sp)
; RV32-NEXT:    csrr t0, mepc
; RV32-NEXT:    sw t0, 24(sp)
; RV32-NEXT:    csrsi mstatus, 8
; RV32:         csrci mstatus, 8
; RV32-NEXT:    lw t0, 24(sp)
; RV32-NEXT:    csrw mepc, t0
; RV32-NEXT:    lw t0, 28(sp)
; RV32-NEXT:    csrw mcause, t0
; RV32:         lw t0, 16(sp)
; RV32:         lw s0, 12(sp)
; RV32:         mret
;
; RV64-LABEL: preemptible:
; RV64:         sd t0, 16(sp)
; RV64:         sd s0, 8(sp)
; RV64:         addi s0, sp, 48
; RV64:         csrr t0, mcause
; RV64-NEXT:    sd t0, 40(sp)
; RV64-NEXT:    csrr t0, mepc
; RV64-NEXT:    sd t0, 32(sp)
; RV64-NEXT:    csrsi mstatus, 8
; RV64:         csrci mstatus, 8
; RV64-NEXT:    ld t0, 32(sp)
; RV64-NEXT:    csrw mepc, t0
; RV64-NEXT:    ld t0, 40(sp)
; RV64-NEXT:    csrw mcause, t0
; RV64:         ld t0, 16(sp)
; RV64:         ld s0, 8(sp)
; RV64:         mret
  ret void
}

define void @preemptible_stack_swap() "interrupt"="SiFive-CLIC-preemptible-stack-swap" "frame-pointer"="all" {
; RV32-LABEL: preemptible_stack_swap:
; RV32:         csrrw sp, sf.mscratchcsw, sp
; RV32:         addi s0, sp, 32
; RV32:         csrr t0, mcause
; RV32:         csrsi mstatus, 8
; RV32:         csrci mstatus, 8
; RV32:         csrw mepc, t0
; RV32:         csrw mcause, t0
; RV32:         lw s0, 12(sp)
; RV32:         csrrw sp, sf.mscratchcsw, sp
; RV32-NEXT:    mret
;
; RV64-LABEL: preemptible_stack_swap:
; RV64:         csrrw sp, sf.mscratchcsw, sp
; RV64:         addi s0, sp, 48
; RV64:         csrr t0, mcause
; RV64:         csrsi mstatus, 8
; RV64:         csrci mstatus, 8
; RV64:         csrw mepc, t0
; RV64:         csrw mcause, t0
; RV64:         ld s0, 8(sp)
; RV64:         csrrw sp, sf.mscratchcsw, sp
; RV64-NEXT:    mret
  ret void
}

define void @preemptible_var_alloca() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="none" {
; RV32-LABEL: preemptible_var_alloca:
; RV32:         addi s0, sp, 80
; RV32:         csrr t0, mcause
; RV32-NEXT:    sw t0, -4(s0)
; RV32-NEXT:    csrr t0, mepc
; RV32-NEXT:    sw t0, -8(s0)
; RV32-NEXT:    csrsi mstatus, 8
; RV32:         sub a0, sp, a0
; RV32:         mv sp, a0
; RV32:         call use
; RV32:         addi sp, s0, -80
; RV32:         csrci mstatus, 8
; RV32-NEXT:    lw t0, -8(s0)
; RV32-NEXT:    csrw mepc, t0
; RV32-NEXT:    lw t0, -4(s0)
; RV32-NEXT:    csrw mcause, t0
; RV32:         mret
;
; RV64-LABEL: preemptible_var_alloca:
; RV64:         addi s0, sp, 160
; RV64:         csrr t0, mcause
; RV64-NEXT:    sd t0, -8(s0)
; RV64-NEXT:    csrr t0, mepc
; RV64-NEXT:    sd t0, -16(s0)
; RV64-NEXT:    csrsi mstatus, 8
; RV64:         sub a0, sp, a0
; RV64:         mv sp, a0
; RV64:         call use
; RV64:         addi sp, s0, -160
; RV64:         csrci mstatus, 8
; RV64-NEXT:    ld t0, -16(s0)
; RV64-NEXT:    csrw mepc, t0
; RV64-NEXT:    ld t0, -8(s0)
; RV64-NEXT:    csrw mcause, t0
; RV64:         mret
  %count = load volatile i32, ptr @size
  %object = alloca i8, i32 %count, align 16
  call void @use(ptr %object)
  ret void
}

define void @preemptible_realign() "interrupt"="SiFive-CLIC-preemptible" "frame-pointer"="none" {
; RV32-LABEL: preemptible_realign:
; RV32:         addi s0, sp,
; RV32:         csrr t0, mcause
; RV32-NEXT:    sw t0,
; RV32-NEXT:    csrr t0, mepc
; RV32-NEXT:    sw t0,
; RV32-NEXT:    csrsi mstatus, 8
; RV32:         andi sp, sp, -64
; RV32:         call use
; RV32:         addi sp, s0,
; RV32:         csrci mstatus, 8
; RV32-NEXT:    lw t0, 184(sp)
; RV32-NEXT:    csrw mepc, t0
; RV32-NEXT:    lw t0, 188(sp)
; RV32-NEXT:    csrw mcause, t0
; RV32:         mret
;
; RV64-LABEL: preemptible_realign:
; RV64:         addi s0, sp,
; RV64:         csrr t0, mcause
; RV64-NEXT:    sd t0,
; RV64-NEXT:    csrr t0, mepc
; RV64-NEXT:    sd t0,
; RV64-NEXT:    csrsi mstatus, 8
; RV64:         andi sp, sp, -64
; RV64:         call use
; RV64:         addi sp, s0,
; RV64:         csrci mstatus, 8
; RV64-NEXT:    ld t0, 240(sp)
; RV64-NEXT:    csrw mepc, t0
; RV64-NEXT:    ld t0, 248(sp)
; RV64-NEXT:    csrw mcause, t0
; RV64:         mret
  %object = alloca [64 x i8], align 64
  call void @use(ptr %object)
  ret void
}
