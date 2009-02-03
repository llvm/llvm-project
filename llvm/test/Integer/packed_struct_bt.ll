; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll
; RUN: not grep cast %t2.ll
; RUN: grep {\\}>} %t2.ll
; END.

%struct.anon = type <{ i8, i35, i35, i35 }>
@foos = external global %struct.anon 
@bara = external global [2 x <{ i35, i8 }>]

;initializers should work for packed and non-packed the same way
@E1 = global <{i8, i35, i35}> <{i8 1, i35 2, i35 3}>
@E2 = global {i8, i35, i35} {i8 4, i35 5, i35 6}


define i35 @main() 
{
        %tmp = load i35*  getelementptr (%struct.anon* @foos, i32 0, i32 1)            ; <i35> [#uses=1]
        %tmp3 = load i35* getelementptr (%struct.anon* @foos, i32 0, i32 2)            ; <i35> [#uses=1]
        %tmp6 = load i35* getelementptr (%struct.anon* @foos, i32 0, i32 3)            ; <i35> [#uses=1]
        %tmp4 = add i35 %tmp3, %tmp             ; <i35> [#uses=1]
        %tmp7 = add i35 %tmp4, %tmp6            ; <i35> [#uses=1]
        ret i35 %tmp7
}

define i35 @bar() {
entry:
        %tmp = load i35* getelementptr([2 x <{ i35, i8 }>]* @bara, i32 0, i32 0, i32 0 )            ; <i35> [#uses=1]
        %tmp4 = load i35* getelementptr ([2 x <{ i35, i8 }>]* @bara, i32 0, i32 1, i32 0)           ; <i35> [#uses=1]
        %tmp5 = add i35 %tmp4, %tmp             ; <i35> [#uses=1]
        ret i35 %tmp5
}
