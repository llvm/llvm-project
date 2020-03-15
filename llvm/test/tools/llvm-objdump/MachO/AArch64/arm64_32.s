// RUN: llvm-mc -triple arm64_32-apple-watchos %s -filetype=obj -o %t
// RUN: llvm-objdump -macho -d %t | FileCheck %s
// RUN: llvm-objdump -macho -private-headers %t | FileCheck %s --check-prefix=CHECK-HEADERS

// CHECK: ldr x0, [x2]
ldr x0, [x2]

// CHECK-HEADERS: MH_MAGIC ARM64_32 V8
