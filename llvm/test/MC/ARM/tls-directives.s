@ RUN: llvm-mc -triple thumbv7-apple-ios -filetype=obj -o %t %s
@ RUN: llvm-objdump --macho -p --indirect-symbols %t | FileCheck %s

@ CHECK: Indirect symbols for (__DATA,__thread_ptr)
@ CHECK: 0x0000001c     5 _a


@ CHECK: sectname __thread_data
@ CHECK: segname __DATA
@ CHECK: type S_THREAD_LOCAL_REGULAR

@ CHECK: sectname __thread_vars
@ CHECK: segname __DATA
@ CHECK: type S_THREAD_LOCAL_VARIABLES

@ CHECK: sectname __thread_bss
@ CHECK: segname __DATA
@ CHECK: type S_THREAD_LOCAL_ZEROFILL

@ CHECK: sectname __thread_ptr
@ CHECK: segname __DATA
@ CHECK: type S_THREAD_LOCAL_VARIABLE_POINTERS


        .section        __DATA,__thread_data,thread_local_regular
        .p2align        2
_b$tlv$init:
        .long 42

        .section        __DATA,__thread_vars,thread_local_variables
        .globl        _b
_b:
        .long        __tlv_bootstrap
        .long        0
        .long        _b$tlv$init

.tbss _c$tlv$init, 4, 2                 @ @c

        .globl        _c
_c:
        .long        __tlv_bootstrap
        .long        0
        .long        _c$tlv$init


        .section        __DATA,__thread_ptr,thread_local_variable_pointers
        .p2align        2
L_a$non_lazy_ptr:
        .indirect_symbol        _a
        .long        0
