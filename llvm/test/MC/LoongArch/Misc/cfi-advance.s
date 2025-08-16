# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=-relax %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=RELOC %s
# RUN: llvm-dwarfdump --debug-frame %t.o | FileCheck --check-prefix=DWARFDUMP %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=RELAX %s

# RELOC:       Relocations [
# RELOC:         .rela.eh_frame {
# RELOC-NEXT:       0x1C R_LARCH_32_PCREL .text 0x0
# RELOC-NEXT:    }
# RELOC-NEXT:  ]
# DWARFDUMP:       DW_CFA_advance_loc: 8
# DWARFDUMP-NEXT:  DW_CFA_def_cfa_offset: +8
# DWARFDUMP-NEXT:  DW_CFA_advance_loc: 4
# DWARFDUMP-NEXT:  DW_CFA_def_cfa_offset: +8

# RELAX:       Relocations [
# RELAX:         .rela.eh_frame {
# RELAX-NEXT:       0x1C R_LARCH_32_PCREL .L{{.*}} 0x0
# RELAX-NEXT:       0x20 R_LARCH_ADD32 .L{{.*}} 0x0
# RELAX-NEXT:       0x20 R_LARCH_SUB32 .L{{.*}} 0x0
# RELAX-NEXT:       0x25 R_LARCH_ADD6 .L{{.*}} 0x0
# RELAX-NEXT:       0x25 R_LARCH_SUB6 .L{{.*}} 0x0
# RELAX-NEXT:       0x28 R_LARCH_ADD6 .L{{.*}} 0x0
# RELAX-NEXT:       0x28 R_LARCH_SUB6 .L{{.*}} 0x0
# RELAX-NEXT:    }
# RELAX-NEXT:  ]

        .text
        .globl test
        .p2align 2
        .type   test,@function
test:
        .cfi_startproc
        call36 foo
        .cfi_def_cfa_offset 8
        .p2align 3
        nop
        .cfi_def_cfa_offset 8
        nop
        .cfi_endproc
