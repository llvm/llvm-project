; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnu | \
; RUN:   grep {mov r11, sp}
; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnu | \
; RUN:   grep {mov sp, r11}

define void @f(i32 %a) {
entry:
        %tmp = alloca i8, i32 %a                ; <i8*> [#uses=1]
        call void @g( i8* %tmp, i32 %a, i32 1, i32 2, i32 3 )
        ret void
}

declare void @g(i8*, i32, i32, i32, i32)
