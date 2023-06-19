# RUN: llvm-mc -filetype=obj -triple riscv32 %s -o %t.o  
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=CHECK %s
# RUN: llvm-dwarfdump --debug-frame %t.o 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-DWARFDUMP %s

# CHECK:      .rela.eh_frame {
# CHECK-NEXT:   0x1C R_RISCV_32_PCREL - 0x0
# CHECK-NEXT:   0x20 R_RISCV_ADD32 - 0x0
# CHECK-NEXT:   0x20 R_RISCV_SUB32 - 0x0
# CHECK-NEXT: }
# CHECK-DWARFDUMP: DW_CFA_advance_loc1
# CHECK-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset
# CHECK-DWARFDUMP-NEXT: DW_CFA_advance_loc2
# CHECK-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset
# CHECK-DWARFDUMP-NEXT: DW_CFA_advance_loc4
# CHECK-DWARFDUMP-NEXT: DW_CFA_def_cfa_offset
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
        .zero 255, 0x90
        .cfi_def_cfa_offset 8
        nop
        .zero 65535, 0x90
        .cfi_def_cfa_offset 8
        nop
        .cfi_endproc
