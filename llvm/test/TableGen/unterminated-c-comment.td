// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck %s

// CHECK: error: unterminated comment

include "unterminated-c-comment-include.inc" */
