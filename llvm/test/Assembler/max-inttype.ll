; RUN: llvm-as < %s | llvm-dis

; i16777215 is the maximum integer type represented in LLVM IR
@i2 = common global i16777215 0, align 4
