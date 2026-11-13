# RUN: llvm-mc %s -triple=riscv32 -mattr=+zca,+zcb -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zca,+zcb < %s \
# RUN:     | llvm-objdump --mattr=+zca,+zcb -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zca,+zcb,+relax < %s \
# RUN:     | llvm-objdump --mattr=+zca,+zcb -M no-aliases --no-print-imm-hex -dr - \
# RUN:     | FileCheck -check-prefix=CHECK-INST,CHECK-RELOC %s

# RUN: llvm-mc --verify %s -triple=riscv64 -mattr=+zca,+zcb

## This tests the instructions which accept %qc.access(...), which are the
## RVI, Zca, and Zcb loads and stores, on RV32 only.

lb a0, 0(a1), %qc.access(extern1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:15 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: lb a0, 0(a1)
# CHECK-ASM-SAME: , %qc.access(extern1)
# CHECK-ENC-SAME: [0x03,0x85,0x05,0x00]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_32 extern1
# CHECK-RELOC-NEXT: R_RISCV_RELAX

lbu a0, 0(a1), %qc.access(extern1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:16 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: lbu a0, 0(a1)
# CHECK-ASM-SAME: , %qc.access(extern1)
# CHECK-ENC-SAME: [0x03,0xc5,0x05,0x00]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_32 extern1
# CHECK-RELOC-NEXT: R_RISCV_RELAX

lh a2, 0(a3), %qc.access(extern2)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:15 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: lh a2, 0(a3)
# CHECK-ASM-SAME: , %qc.access(extern2)
# CHECK-ENC-SAME: [0x03,0x96,0x06,0x00]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_32 extern2
# CHECK-RELOC-NEXT: R_RISCV_RELAX

lhu a2, 0(a3), %qc.access(extern2)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:16 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: lhu a2, 0(a3)
# CHECK-ASM-SAME: , %qc.access(extern2)
# CHECK-ENC-SAME: [0x03,0xd6,0x06,0x00]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_32 extern2
# CHECK-RELOC-NEXT: R_RISCV_RELAX

lw a4, 0(a5), %qc.access(extern4)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:15 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: lw a4, 0(a5)
# CHECK-ASM-SAME: , %qc.access(extern4)
# CHECK-ENC-SAME: [0x03,0xa7,0x07,0x00]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_32 extern4
# CHECK-RELOC-NEXT: R_RISCV_RELAX

sb a0, 0(a1), %qc.access(extern1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:15 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: sb a0, 0(a1)
# CHECK-ASM-SAME: , %qc.access(extern1)
# CHECK-ENC-SAME: [0x23,0x80,0xa5,0x00]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_32 extern1
# CHECK-RELOC-NEXT: R_RISCV_RELAX

sh a2, 0(a3), %qc.access(extern2)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:15 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: sh a2, 0(a3)
# CHECK-ASM-SAME: , %qc.access(extern2)
# CHECK-ENC-SAME: [0x23,0x90,0xc6,0x00]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_32 extern2
# CHECK-RELOC-NEXT: R_RISCV_RELAX

sw a4, 0(a5), %qc.access(extern4)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:15 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: sw a4, 0(a5)
# CHECK-ASM-SAME: , %qc.access(extern4)
# CHECK-ENC-SAME: [0x23,0xa0,0xe7,0x00]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_32 extern4
# CHECK-RELOC-NEXT: R_RISCV_RELAX

## No c.lb

c.lbu a0, 0(a1), %qc.access(extern1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:18 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: c.lbu a0, 0(a1)
# CHECK-ASM-SAME: , %qc.access(extern1)
# CHECK-ENC-SAME: [0x88,0x81]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_16 extern1
# CHECK-RELOC-NEXT: R_RISCV_RELAX

c.lh a2, 0(a3), %qc.access(extern2)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: c.lh a2, 0(a3)
# CHECK-ASM-SAME: , %qc.access(extern2)
# CHECK-ENC-SAME: [0xd0,0x86]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_16 extern2
# CHECK-RELOC-NEXT: R_RISCV_RELAX

c.lhu a2, 0(a3), %qc.access(extern2)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:18 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: c.lhu a2, 0(a3)
# CHECK-ASM-SAME: , %qc.access(extern2)
# CHECK-ENC-SAME: [0x90,0x86]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_16 extern2
# CHECK-RELOC-NEXT: R_RISCV_RELAX

c.lw a4, 0(a5), %qc.access(extern4)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: c.lw a4, 0(a5)
# CHECK-ASM-SAME: , %qc.access(extern4)
# CHECK-ENC-SAME: [0x98,0x43]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_16 extern4
# CHECK-RELOC-NEXT: R_RISCV_RELAX

c.sb a0, 0(a1), %qc.access(extern1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: c.sb a0, 0(a1)
# CHECK-ASM-SAME: , %qc.access(extern1)
# CHECK-ENC-SAME: [0x88,0x89]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_16 extern1
# CHECK-RELOC-NEXT: R_RISCV_RELAX

c.sh a2, 0(a3), %qc.access(extern2)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: c.sh a2, 0(a3)
# CHECK-ASM-SAME: , %qc.access(extern2)
# CHECK-ENC-SAME: [0x90,0x8e]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_16 extern2
# CHECK-RELOC-NEXT: R_RISCV_RELAX

c.sw a4, 0(a5), %qc.access(extern4)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{unexpected extra operand for instruction}}
# expected-note@-3:1 {{instruction requires the following: RV32I Base Instruction Set}}
# CHECK-INST: c.sw a4, 0(a5)
# CHECK-ASM-SAME: , %qc.access(extern4)
# CHECK-ENC-SAME: [0x98,0xc3]
# CHECK-RELOC: R_RISCV_VENDOR QUALCOMM
# CHECK-RELOC-NEXT: R_RISCV_QC_ACCESS_16 extern4
# CHECK-RELOC-NEXT: R_RISCV_RELAX
