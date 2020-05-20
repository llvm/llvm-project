## This test checks that llvm-dwarfdump produces
## DW_MACINFO_invalid when parsing *_strx
## form in a debug_macinfo section.

# RUN: llvm-mc -triple x86_64-unknown-linux -filetype=obj %s -o -| \
# RUN:   llvm-dwarfdump -debug-macro - | FileCheck %s

#      CHECK: .debug_macinfo contents:
# CHECK-NEXT: 0x00000000:
# CHECK-NEXT: DW_MACINFO_invalid

       .section        .debug_macinfo,"",@progbits
.Lcu_macinfo_begin0:
       .byte   11                     # DW_MACRO_define_strx
