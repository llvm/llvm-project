// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t | FileCheck %s

.string "abcdefg
12345678"

// CHECK: Warning: unterminated string; newline inserted
