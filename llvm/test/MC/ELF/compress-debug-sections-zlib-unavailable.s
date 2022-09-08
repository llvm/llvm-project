// REQUIRES: !zlib
// RUN: not llvm-mc -filetype=obj -compress-debug-sections=zlib -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

// CHECK:     llvm-mc{{[^:]*}}: error: build tools with LLVM_ENABLE_ZLIB to enable --compress-debug-sections=zlib
// CHECK-NOT: {{.}}
