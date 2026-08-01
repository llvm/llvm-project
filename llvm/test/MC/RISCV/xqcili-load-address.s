# RUN: llvm-mc -triple=riscv32 -mattr=+xqcili %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xqcili %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump --no-addresses -dr --mattr=+xqcili - \
# RUN:     | FileCheck -check-prefix=OBJ %s

## This test checks that we are lowering la/lla to qc.e.li correctly

lla x6, abs_sym
// ASM: qc.e.li	t1, abs_sym
// OBJ: 031f 0000 0000        	qc.e.li	t1, 0x0
// OBJ-NEXT: R_RISCV_VENDOR	QUALCOMM
// OBJ-NEXT: R_RISCV_QC_E_32	abs_sym

lla x6, same_section
// ASM: qc.e.li	t1, same_section
// OBJ: 031f 0000 0000        	qc.e.li	t1, 0x0
// OBJ-NEXT: R_RISCV_VENDOR	QUALCOMM
// OBJ-NEXT: R_RISCV_QC_E_32	same_section

.option nopic
// ASM: .option	nopic

la x6, same_section
// ASM: qc.e.li	t1, same_section
// OBJ: 031f 0000 0000        	qc.e.li	t1, 0x0
// OBJ-NEXT: R_RISCV_VENDOR	QUALCOMM
// OBJ-NEXT: R_RISCV_QC_E_32	same_section

la x6, abs_sym
// ASM: qc.e.li	t1, abs_sym
// OBJ: 031f 0000 0000        	qc.e.li	t1, 0x0
// OBJ-NEXT: R_RISCV_VENDOR	QUALCOMM
// OBJ-NEXT: R_RISCV_QC_E_32	abs_sym

same_section:
// ASM: same_section:
nop
// ASM: nop
// OBJ: 0001                  	nop
