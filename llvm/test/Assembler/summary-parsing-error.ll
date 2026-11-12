; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Reference to undefined global "does_not_exist"
^0 = gv: (name: "does_not_exist")
