# RUN: not llvm-mc -triple i386 %s -filetype=obj -o /dev/null 2>&1 | FileCheck --check-prefix=X86 %s
# RUN: not llvm-mc -triple=x86_64 %s -filetype=obj -o /dev/null 2>&1 | FileCheck --check-prefix=X64 %s

.nops 4, 3
# X86: :[[@LINE-1]]:1: error: illegal NOP size 3.
.nops 4, 4
# X86: :[[@LINE-1]]:1: error: illegal NOP size 4.
.nops 4, 5
# X86: :[[@LINE-1]]:1: error: illegal NOP size 5.
.nops 16, 15
# X86: :[[@LINE-1]]:1: error: illegal NOP size 15.
# X64: :[[@LINE-2]]:1: error: illegal NOP size 15.
