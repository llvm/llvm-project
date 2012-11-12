# RUN: llvm-mc -filetype=obj %s -o %t.o -triple x86_64-pc-linux-gnu && llvm-objdump -s %t.o
# PR13581

# CHECK: Contents of section .debug_frame:
# CHECK-NEXT:  0000 14000000 ffffffff 01000178 100c0708  ...........x....
# CHECK-NEXT:  0010 90010000 00000000 1c000000 00000000  ................
# CHECK-NEXT:  0020 00000000 00000000 11000000 00000000  ................
# CHECK-NEXT:  0030 410e1086 02430d06                    A....C..

__cxx_global_var_init:                  # @__cxx_global_var_init
        .cfi_startproc
.Lfunc_begin0:
# BB#0:                                 # %entry
        pushq   %rbp
.Ltmp2:
        .cfi_def_cfa_offset 16
.Ltmp3:
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
.Ltmp4:
        .cfi_def_cfa_register %rbp
.Ltmp5:
        callq   _Z2rsv@PLT
        movl    %eax, _ZL1i(%rip)
        popq    %rbp
        ret
        .cfi_endproc
        .cfi_sections .debug_frame
