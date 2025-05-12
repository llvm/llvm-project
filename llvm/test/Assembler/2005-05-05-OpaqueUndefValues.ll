; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: error: invalid type for undef constant

%t = type opaque
@x = global %t undef
