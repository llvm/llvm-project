# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=-relax %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=RELOC %s
# RUN: llvm-dwarfdump --debug-frame %t.o | FileCheck --check-prefix=DWARFDUMP %s

# RELOC:       Relocations [
# RELOC-NEXT:    .rela.eh_frame {
# RELOC-NEXT:       0x1C R_LARCH_32_PCREL .text 0x0
# RELOC-NEXT:    }
# RELOC-NEXT:  ]
# DWARFDUMP:       DW_CFA_advance_loc: 4
# DWARFDUMP-NEXT:  DW_CFA_def_cfa_offset: +8
# DWARFDUMP-NEXT:  DW_CFA_advance_loc: 8
# DWARFDUMP-NEXT:  DW_CFA_def_cfa_offset: +8

        .text
        .globl test
        .p2align 2
        .type   test,@function
test:
        .cfi_startproc
        nop
        .cfi_def_cfa_offset 8
        .p2align 3
        nop
        .cfi_def_cfa_offset 8
        nop
        .cfi_endproc
