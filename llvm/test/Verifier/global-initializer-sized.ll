; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@g = global target("opaque") undef

; CHECK: Global variable initializer must be sized
