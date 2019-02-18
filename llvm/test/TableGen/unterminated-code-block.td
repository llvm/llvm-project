// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck %s

// CHECK: error: Unterminated Code Block

include "unterminated-code-block-include.inc" }]>;
