# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: llvm-mc -triple=x86_64-unknown-freebsd -filetype=obj -o t.o %s
# RUN: llvm-rtdyld -triple=x86_64-unknown-linux -verify -check=%s t.o -dummy-extern=extern_data=0x4200000000

# Generated from this C source:
#
# static int static_data[10];
# int global_data[10] = {1, 2};
# extern int extern_data[10];
#
# int *lea_static_data() { return &static_data[0]; }
# int *lea_global_data() { return &global_data[0]; }
# int *lea_extern_data() { return &extern_data[0]; }

        .text
        .file   "model.c"
        .globl  lea_static_data         # -- Begin function lea_static_data
        .p2align        4, 0x90
        .type   lea_static_data,@function
lea_static_data:                        # @lea_static_data
        .cfi_startproc
# %bb.0:
.Ltmp0:
        leaq    .Ltmp0(%rip), %rcx
# rtdyld-check: decode_operand(lea_static_got, 1) = section_addr(t.o, .got) - lea_static_data
lea_static_got:
        movabsq $_GLOBAL_OFFSET_TABLE_-.Ltmp0, %rax
        addq    %rax, %rcx
# rtdyld-check: decode_operand(lea_static_gotoff, 1) = static_data - section_addr(t.o, .got)
lea_static_gotoff:
        movabsq $static_data@GOTOFF, %rax
        addq    %rcx, %rax
        retq
.Lfunc_end0:
        .size   lea_static_data, .Lfunc_end0-lea_static_data
        .cfi_endproc


        .globl  lea_global_data         # -- Begin function lea_global_data
        .p2align        4, 0x90
        .type   lea_global_data,@function
lea_global_data:                        # @lea_global_data
        .cfi_startproc
# %bb.0:
.Ltmp1:
        leaq    .Ltmp1(%rip), %rcx
# rtdyld-check: decode_operand(lea_global_got, 1) = section_addr(t.o, .got) - lea_global_data
lea_global_got:
        movabsq $_GLOBAL_OFFSET_TABLE_-.Ltmp1, %rax
        addq    %rax, %rcx
# rtdyld-check: decode_operand(lea_global_gotoff, 1) = global_data - section_addr(t.o, .got)
lea_global_gotoff:
        movabsq $global_data@GOTOFF, %rax
        addq    %rcx, %rax
        retq
.Lfunc_end1:
        .size   lea_global_data, .Lfunc_end1-lea_global_data
        .cfi_endproc


        .globl  lea_extern_data         # -- Begin function lea_extern_data
        .p2align        4, 0x90
        .type   lea_extern_data,@function
lea_extern_data:                        # @lea_extern_data
        .cfi_startproc
# %bb.0:
.Ltmp2:
        leaq    .Ltmp2(%rip), %rax
# rtdyld-check: decode_operand(lea_extern_got, 1) = section_addr(t.o, .got) - lea_extern_data
lea_extern_got:
        movabsq $_GLOBAL_OFFSET_TABLE_-.Ltmp2, %rcx
        addq    %rcx, %rax
# extern_data is the only thing in the GOT, so it'll be slot 0.
# rtdyld-check: decode_operand(lea_extern_gotslot, 1) = 0
lea_extern_gotslot:
        movabsq $extern_data@GOT, %rcx
        movq    (%rax,%rcx), %rax
        retq
.Lfunc_end2:
        .size   lea_extern_data, .Lfunc_end2-lea_extern_data
        .cfi_endproc


        .type   global_data,@object     # @global_data
        .data
        .globl  global_data
        .p2align        4
global_data:
        .long   1                       # 0x1
        .long   2                       # 0x2
        .long   0                       # 0x0
        .long   0                       # 0x0
        .long   0                       # 0x0
        .long   0                       # 0x0
        .long   0                       # 0x0
        .long   0                       # 0x0
        .long   0                       # 0x0
        .long   0                       # 0x0
        .size   global_data, 40

        .type   static_data,@object     # @static_data
        .local  static_data
        .comm   static_data,40,16

