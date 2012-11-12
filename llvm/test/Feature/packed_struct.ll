; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll
; RUN: not grep cast %t2.ll
; RUN: grep "}>" %t2.ll
; END.

%struct.anon = type <{ i8, i32, i32, i32 }>
@foos = external global %struct.anon 
@bara = external global [2 x <{ i32, i8 }>]

;initializers should work for packed and non-packed the same way
@E1 = global <{i8, i32, i32}> <{i8 1, i32 2, i32 3}>
@E2 = global {i8, i32, i32} {i8 4, i32 5, i32 6}


define i32 @main() 
{
        %tmp = load i32*  getelementptr (%struct.anon* @foos, i32 0, i32 1)            ; <i32> [#uses=1]
        %tmp3 = load i32* getelementptr (%struct.anon* @foos, i32 0, i32 2)            ; <i32> [#uses=1]
        %tmp6 = load i32* getelementptr (%struct.anon* @foos, i32 0, i32 3)            ; <i32> [#uses=1]
        %tmp4 = add i32 %tmp3, %tmp             ; <i32> [#uses=1]
        %tmp7 = add i32 %tmp4, %tmp6            ; <i32> [#uses=1]
        ret i32 %tmp7
}

define i32 @bar() {
entry:
        %tmp = load i32* getelementptr([2 x <{ i32, i8 }>]* @bara, i32 0, i32 0, i32 0 )            ; <i32> [#uses=1]
        %tmp4 = load i32* getelementptr ([2 x <{ i32, i8 }>]* @bara, i32 0, i32 1, i32 0)           ; <i32> [#uses=1]
        %tmp5 = add i32 %tmp4, %tmp             ; <i32> [#uses=1]
        ret i32 %tmp5
}
