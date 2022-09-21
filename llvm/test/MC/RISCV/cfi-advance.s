# RUN: llvm-mc -filetype=obj -triple riscv32 %s -o %t.o  
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=CHECK %s
# RUN: llvm-dwarfdump --debug-frame %t.o 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-DWARFDUMP %s

# CHECK: 0x26 R_RISCV_SET8 - 0x0
# CHECK-DWARFDUMP:   DW_CFA_advance_loc1
# CHECK-DWARFDUMP-NEXT:  DW_CFA_def_cfa_offset
        .text
        .globl  test                            # -- Begin function test
        .p2align        1
        .type   test,@function
test:
        .cfi_startproc
        nop
        .zero 100, 0x90
        .cfi_def_cfa_offset 8
        nop
        .cfi_endproc
