; try to check that we have the most important instructions, which shouldn't 
; appear otherwise
; RUN: llvm-as < %s | llc -march=alpha | grep jmp
; RUN: llvm-as < %s | llc -march=alpha | grep gprel32
; RUN: llvm-as < %s | llc -march=alpha | grep ldl
; RUN: llvm-as < %s | llc -march=alpha | grep rodata
; END.

target datalayout = "e-p:64:64"
target triple = "alphaev67-unknown-linux-gnu"
@str = internal constant [2 x i8] c"1\00"               ; <[2 x i8]*> [#uses=1]
@str1 = internal constant [2 x i8] c"2\00"              ; <[2 x i8]*> [#uses=1]
@str2 = internal constant [2 x i8] c"3\00"              ; <[2 x i8]*> [#uses=1]
@str3 = internal constant [2 x i8] c"4\00"              ; <[2 x i8]*> [#uses=1]
@str4 = internal constant [2 x i8] c"5\00"              ; <[2 x i8]*> [#uses=1]
@str5 = internal constant [2 x i8] c"6\00"              ; <[2 x i8]*> [#uses=1]
@str6 = internal constant [2 x i8] c"7\00"              ; <[2 x i8]*> [#uses=1]
@str7 = internal constant [2 x i8] c"8\00"              ; <[2 x i8]*> [#uses=1]

define i32 @main(i32 %x, i8** %y) {
entry:
        %x_addr = alloca i32            ; <i32*> [#uses=2]
        %y_addr = alloca i8**           ; <i8***> [#uses=1]
        %retval = alloca i32, align 4           ; <i32*> [#uses=2]
        %tmp = alloca i32, align 4              ; <i32*> [#uses=2]
        %foo = alloca i8*, align 8              ; <i8**> [#uses=9]
        %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
        store i32 %x, i32* %x_addr
        store i8** %y, i8*** %y_addr
        %tmp.upgrd.1 = load i32* %x_addr                ; <i32> [#uses=1]
        switch i32 %tmp.upgrd.1, label %bb15 [
                 i32 1, label %bb
                 i32 2, label %bb1
                 i32 3, label %bb3
                 i32 4, label %bb5
                 i32 5, label %bb7
                 i32 6, label %bb9
                 i32 7, label %bb11
                 i32 8, label %bb13
        ]

bb:             ; preds = %entry
        %tmp.upgrd.2 = getelementptr [2 x i8]* @str, i32 0, i64 0               ; <i8*> [#uses=1]
        store i8* %tmp.upgrd.2, i8** %foo
        br label %bb16

bb1:            ; preds = %entry
        %tmp2 = getelementptr [2 x i8]* @str1, i32 0, i64 0             ; <i8*> [#uses=1]
        store i8* %tmp2, i8** %foo
        br label %bb16

bb3:            ; preds = %entry
        %tmp4 = getelementptr [2 x i8]* @str2, i32 0, i64 0             ; <i8*> [#uses=1]
        store i8* %tmp4, i8** %foo
        br label %bb16

bb5:            ; preds = %entry
        %tmp6 = getelementptr [2 x i8]* @str3, i32 0, i64 0             ; <i8*> [#uses=1]
        store i8* %tmp6, i8** %foo
        br label %bb16

bb7:            ; preds = %entry
        %tmp8 = getelementptr [2 x i8]* @str4, i32 0, i64 0             ; <i8*> [#uses=1]
        store i8* %tmp8, i8** %foo
        br label %bb16

bb9:            ; preds = %entry
        %tmp10 = getelementptr [2 x i8]* @str5, i32 0, i64 0            ; <i8*> [#uses=1]
        store i8* %tmp10, i8** %foo
        br label %bb16

bb11:           ; preds = %entry
        %tmp12 = getelementptr [2 x i8]* @str6, i32 0, i64 0            ; <i8*> [#uses=1]
        store i8* %tmp12, i8** %foo
        br label %bb16

bb13:           ; preds = %entry
        %tmp14 = getelementptr [2 x i8]* @str7, i32 0, i64 0            ; <i8*> [#uses=1]
        store i8* %tmp14, i8** %foo
        br label %bb16

bb15:           ; preds = %entry
        br label %bb16

bb16:           ; preds = %bb15, %bb13, %bb11, %bb9, %bb7, %bb5, %bb3, %bb1, %bb
        %tmp17 = load i8** %foo         ; <i8*> [#uses=1]
        %tmp18 = call i32 (...)* @print( i8* %tmp17 )           ; <i32> [#uses=0]
        store i32 0, i32* %tmp
        %tmp19 = load i32* %tmp         ; <i32> [#uses=1]
        store i32 %tmp19, i32* %retval
        br label %return

return:         ; preds = %bb16
        %retval.upgrd.3 = load i32* %retval             ; <i32> [#uses=1]
        ret i32 %retval.upgrd.3
}

declare i32 @print(...)

