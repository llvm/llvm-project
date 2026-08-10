; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: error: identified structure type 'myTy' is recursive

%myTy = type { %myTy }
