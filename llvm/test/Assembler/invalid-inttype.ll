; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; i8388609 is the smallest integer type that can't be represented in LLVM IR
@i2 = common global i8388609 0, align 4
; CHECK: expected type
