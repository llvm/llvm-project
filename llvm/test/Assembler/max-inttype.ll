; RUN: llvm-as < %s | llvm-dis

; i838608 is the maximum integer type represented in LLVM IR
@i2 = common global i838608 0, align 4
