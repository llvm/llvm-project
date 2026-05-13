# RUN: llvm-mc -triple=riscv32 < %s -mattr=-relax -filetype=obj -o - \
# RUN: | llvm-objdump -d --no-leading-addr --no-show-raw-insn --symbolize-operands - \
# RUN: | FileCheck %s

# CHECK-LABEL: <.text>:
  .text
  .p2align  2
# CHECK: blez a0, <L4>
  blez a0, .LBB0_6
  li a3, 0
  li a2, 0
# CHECK: j <L1>
  j .LBB0_3
# CHECK-NEXT: <L0>:
.LBB0_2:
  addi a3, a3, 1
# CHECK: beq a3, a0, <L5>
  beq a3, a0, .LBB0_7
# CHECK-NEXT: <L1>:
.LBB0_3:
  slli a4, a3, 2
  add a4, a1, a4
  lw a5, 0(a4)
  lbu a4, 0(a5)
# CHECK: beqz a4, <L0>
  beqz a4, .LBB0_2
  addi a5, a5, 1
# CHECK: <L2>
# CHECK-NEXT: jal t5, <L2>
  jal t5, foo
# CHECK: <L3>
.LBB0_5:
  add a2, a2, a4
  lbu a4, 0(a5)
  addi a5, a5, 1
# CHECK: bnez a4, <L3>
  bnez a4, .LBB0_5
# CHECK-NEXT: j <L0>
  j .LBB0_2
# CHECK-NEXT: <L4>:
.LBB0_6:
  li a2, 0
# CHECK: <L5>:
.LBB0_7:
  mv a0, a2
# CHECK: <L6>:
# CHECK-NEXT: auipc ra, 0x0
# CHECK-NEXT: jalr ra <L6>
  call bar
# CHECK-NEXT: <L7>:
# CHECK-NEXT: beq a0, a2, <.text+0x17a>
  beq a0, a2, 0x122
# CHECK-NEXT: bge a0, a2, <L7>
  bge a0, a2, -0x4
  ret
