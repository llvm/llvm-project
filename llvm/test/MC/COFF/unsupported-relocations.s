// RUN: not llvm-mc -filetype=obj -triple i386-pc-win32 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-win32 %s 2>&1 | FileCheck %s
// CHECK: unsupported relocation type
        .text
        mov $_GLOBAL_OFFSET_TABLE_, %eax
