; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define float @extractvalue() {
; CHECK: [[@LINE+1]]:13: error: extractvalue constexprs are no longer supported
  ret float extractvalue ({i32} {i32 3}, 0)
}
