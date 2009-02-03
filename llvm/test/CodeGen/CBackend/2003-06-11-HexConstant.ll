; RUN: llvm-as < %s | llc -march=c

; Make sure hex constant does not continue into a valid hexadecimal letter/number
@version = global [3 x i8] c"\001\00"
