# RUN: llvm-mc -triple=riscv32 -mattr=+xqcili %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xqcili %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump --no-addresses -dr --mattr=+xqcili - \
# RUN:     | FileCheck -check-prefix=OBJ %s

## This test checks that we are lowering la/lla to qc.e.li correctly

.option pic
// ASM: .option pic

lla x6, abs_sym
// ASM: .Lpcrel_hi0:
// ASM-NEXT: auipc t1, %pcrel_hi(abs_sym)
// ASM-NEXT: addi t1, t1, %pcrel_lo(.Lpcrel_hi0)
// OBJ:      00000317               auipc t1, 0x0
// OBJ-NEXT: R_RISCV_PCREL_HI20 abs_sym
// OBJ-NEXT: 00030313               mv t1, t1
// OBJ-NEXT: R_RISCV_PCREL_LO12_I .Lpcrel_hi0

lla x6, same_section
// ASM: .Lpcrel_hi1:
// ASM-NEXT: auipc t1, %pcrel_hi(same_section)
// ASM-NEXT: addi t1, t1, %pcrel_lo(.Lpcrel_hi1)
// OBJ:      00000317               auipc t1, 0x0
// OBJ-NEXT: 02030313               addi t1, t1, 0x20

.option nopic
// ASM: .option nopic

lla x6, abs_sym
// ASM: qc.e.li t1, abs_sym
// OBJ: 031f 0000 0000         qc.e.li t1, 0x0
// OBJ-NEXT: R_RISCV_VENDOR QUALCOMM
// OBJ-NEXT: R_RISCV_QC_E_32 abs_sym

lla x6, same_section
// ASM: qc.e.li t1, same_section
// OBJ: 031f 0000 0000         qc.e.li t1, 0x0
// OBJ-NEXT: R_RISCV_VENDOR QUALCOMM
// OBJ-NEXT: R_RISCV_QC_E_32 same_section

la x6, same_section
// ASM: qc.e.li t1, same_section
// OBJ: 031f 0000 0000         qc.e.li t1, 0x0
// OBJ-NEXT: R_RISCV_VENDOR QUALCOMM
// OBJ-NEXT: R_RISCV_QC_E_32 same_section

la x6, abs_sym
// ASM: qc.e.li t1, abs_sym
// OBJ: 031f 0000 0000         qc.e.li t1, 0x0
// OBJ-NEXT: R_RISCV_VENDOR QUALCOMM
// OBJ-NEXT: R_RISCV_QC_E_32 abs_sym

same_section:
// ASM: same_section:
nop
// ASM: nop
// OBJ: 0001                   nop
