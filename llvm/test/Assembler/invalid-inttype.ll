; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; i16777216 is the smallest integer type that can't be represented in LLVM IR
@i2 = common global i16777216 0, align 4
; CHECK: expected type
