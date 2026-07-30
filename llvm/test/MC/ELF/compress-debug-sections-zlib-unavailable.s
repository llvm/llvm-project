// REQUIRES: !zlib
// RUN: not llvm-mc -filetype=obj -compress-debug-sections=zlib -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

// CHECK:     error: --compress-debug-sections: LLVM was not built with LLVM_ENABLE_ZLIB or did not find zlib at build time
// CHECK-NOT: {{.}}
