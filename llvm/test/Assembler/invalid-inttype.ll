; RUN: not llvm-as --disable-output %s 2>&1 | FileCheck -DFILE=%s %s

; i8388609 is the smallest integer type that can't be represented in LLVM IR
; CHECK: [[FILE]]:[[@LINE+1]]:21: error: bitwidth for integer type out of range
@i2 = common global i8388609 0, align 4
