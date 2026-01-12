# RUN: llvm-mc --filetype=obj --gsframe -triple x86_64 %s -o %t.o
# RUN: llvm-readelf --sframe %t.o | FileCheck %s

## Tests that .cfi_escape sequences that are ok to pass through work.

        .align 1024
cfi_escape_ok:
        .cfi_startproc
        .long 0
        .cfi_def_cfa_offset 16
        ## Uninteresting register
## DW_CFA_expression,reg 0xc,length 2,DW_OP_breg6,SLEB(-8)
        .cfi_escape 0x10,0xc,0x2,0x76,0x78
## DW_CFA_nop
        .cfi_escape 0x0
        .cfi_escape 0x0,0x0,0x0,0x0
        ## Uninteresting register
## DW_CFA_val_offset,reg 0xc,ULEB scaled offset
        .cfi_escape 0x14,0xc,0x4
        .long 0
        .cfi_endproc

cfi_escape_gnu_args_fp:
        .cfi_startproc
        .long 0
## DW_CFA_GNU_args_size is OK if arg size is zero
        .cfi_escape 0x2e, 0x0
        .long 0
        .cfi_def_cfa_register 6
        .long 0
## DW_CFA_GNU_args_size is OK if cfa is FP
        .cfi_escape 0x2e, 0x20
        .cfi_endproc

cfi_escape_long_expr:
        .cfi_startproc
        .long 0
        .cfi_def_cfa_offset 16
## This is a long, but valid, dwarf expression without sframe
## implications. An FDE can still be created.
## DW_CFA_val_offset,rcx,ULEB scaled offset(16), DW_CFA_expr,r10,length,DW_OP_deref,SLEB(-8)
        .cfi_escape 0x14,0x2,0x2,0x10,0xa,0x2,0x76,0x78
        .long 0
        .cfi_endproc

# CHECK: Num FDEs: 3
