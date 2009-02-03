; RUN: llvm-as < %s | opt -instcombine -inline | llvm-dis | not grep '-715827882'
; PR3142

define i32 @a(i32 %X) nounwind readnone {
entry:
       %0 = sub i32 0, %X
       %1 = sdiv i32 %0, -3
       ret i32 %1
}

define i32 @b(i32 %X) nounwind readnone {
entry:
       %0 = call i32 @a(i32 -2147483648)
       ret i32 %0
}

define i32 @c(i32 %X) nounwind readnone {
entry:
       %0 = sub i32 0, -2147483648
       %1 = sdiv i32 %0, -3
       ret i32 %1
}
