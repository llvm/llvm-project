// RUN: llvm-mc -filetype=obj %s -o - -triple i686-pc-linux | llvm-objdump --no-print-imm-hex -d -r - | FileCheck --check-prefix=X86 %s
// RUN: llvm-mc -filetype=obj %s -o - -triple x86_64-pc-linux | llvm-objdump --no-print-imm-hex -d -r - | FileCheck --check-prefix=X64 %s

// X86: 0: 83 ff 00  cmpl $0, %edi
// X86:   00000002:  R_386_8 foo
// X64: 0: 83 ff 00  cmpl $0, %edi
// X64:  0000000000000002:  R_X86_64_8 foo
// X64-NEXT: 3: 3b 3c 25 00 00 00 00          cmpl    0, %edi
// X64-NEXT:  0000000000000006:  R_X86_64_32  $foo+0x3
// X64-NEXT: a: 3b 04 25 00 00 00 00          cmpl    0, %eax
// X64-NEXT:  000000000000000d:  R_X86_64_32  $foo-0x4
cmp $foo@ABS8, %edi
cmp (3+$foo)@ABS8, %edi
cmp ($foo-4)@ABS8, %eax
