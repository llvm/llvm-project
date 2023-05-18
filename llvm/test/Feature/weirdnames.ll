; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Test using double quotes to form names that are not legal in the % form
%"&^ " = type { i32 }
@"%.*+ foo" = global %"&^ " { i32 5 }           
@"0" = global float 0.000000e+00                ; This CANNOT be %0 
@"\\03foo" = global float 0x3FB99999A0000000    ; Make sure funny char gets round trip 
