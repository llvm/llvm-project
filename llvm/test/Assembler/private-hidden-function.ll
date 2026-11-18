; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

define private hidden void @function() {
; CHECK: symbol with local linkage must have default visibility
entry:
  ret void
}
