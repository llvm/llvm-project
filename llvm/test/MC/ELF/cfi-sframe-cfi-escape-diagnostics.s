# RUN: llvm-mc --filetype=obj --gsframe -triple x86_64 %s -o %t.o 2>&1 | FileCheck %s
# RUN: llvm-readelf --sframe %t.o | FileCheck %s --check-prefix=NOFDES

## Tests that .cfi_escape sequences that are unrepresentable in sframe warn
## and do not produce FDEs.

        .align 1024
cfi_escape_sp:
        .cfi_startproc
        .long 0
## Setting SP via other registers makes it unrepresentable in sframe
## DW_CFA_expression,reg 0x7,length 2,DW_OP_breg6,SLEB(-8)
# CHECK: {{.*}}.s:[[#@LINE+1]]:9: warning: skipping SFrame FDE; .cfi_escape DW_CFA_expression with SP reg 7
        .cfi_escape 0x10, 0x7, 0x2, 0x76, 0x78
        .long 0
.cfi_endproc

cfi_escape_args_sp:
        .cfi_startproc
        .long 0
## DW_CFA_GNU_args_size is not OK if cfa is SP
# CHECK: {{.*}}.s:[[#@LINE+1]]:9: warning: skipping SFrame FDE; .cfi_escape DW_CFA_GNU_args_size with non frame-pointer CFA
        .cfi_escape 0x2e, 0x20
        .cfi_endproc

cfi_escape_val_offset:
        .cfi_startproc
        .long 0
        .cfi_def_cfa_offset 16
## DW_CFA_val_offset,rbp,ULEB scaled offset(16)
# CHECK: {{.*}}.s:[[#@LINE+1]]:9: warning: skipping SFrame FDE;  .cfi_escape DW_CFA_val_offset with FP reg 6
        .cfi_escape 0x14,0x6,0x2
        .long 0
        .cfi_endproc

# NOFDES: Num FDEs: 0
