; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

@llvm.used = appending global [1 x ptr] zeroinitializer, section "llvm.metadata"

; CHECK: wrong initializer for intrinsic global variable
; CHECK-NEXT: [1 x ptr] zeroinitializer
