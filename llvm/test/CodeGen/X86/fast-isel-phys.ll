; RUN: llvm-as < %s | llc -fast-isel -march=x86

define i8 @t2(i8 %a, i8 %c) nounwind {
       %tmp = shl i8 %a, %c
       ret i8 %tmp
}

define i8 @t1(i8 %a) nounwind {
       %tmp = mul i8 %a, 17
       ret i8 %tmp
}
