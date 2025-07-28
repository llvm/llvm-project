// RUN: not llvm-mc -filetype=obj -o /dev/null -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s

.intel_syntax noprefix
.code64
test_case:
// CHECK: error: value of -9223372036854775808 is too large for field of 4 bytes.
  mov rcx, EXAMPLE
.set EXAMPLE, (1ULL<<63ULL)
