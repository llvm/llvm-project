; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define private hidden void @function() {
; CHECK: symbol with local linkage must have default visibility
entry:
  ret void
}
