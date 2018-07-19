// RUN: llvm-mc -filetype=obj %s -o - -triple i686-pc-linux | llvm-objdump -d -r - | FileCheck --check-prefix=32 %s
// RUN: llvm-mc -filetype=obj %s -o - -triple x86_64-pc-linux | llvm-objdump -d -r - | FileCheck --check-prefix=64 %s

// 32: 0: 83 ff 00  cmpl $0, %edi
// 32:   00000002:  R_386_8 foo
// 64: 0: 83 ff 00  cmpl $0, %edi
// 64:  0000000000000002:  R_X86_64_8 foo+0
cmp $foo@ABS8, %edi
