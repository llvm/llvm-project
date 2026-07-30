; RUN: llvm-as < %s | llvm-dis -disable-output
; RUN: verify-uselistorder < %s

!llvm.foo = !{!0}
!0 = !{i32 42}
@my.str = internal constant [4 x i8] c"foo\00"
