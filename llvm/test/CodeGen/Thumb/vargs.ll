; RUN: llc < %s -march=thumb
; RUN: llc < %s -mtriple=thumb-linux | grep pop | count 2
; RUN: llc < %s -mtriple=thumb-darwin | grep pop | count 2

@str = internal constant [4 x i8] c"%d\0A\00"           ; <[4 x i8]*> [#uses=1]

define void @f(i32 %a, ...) {
entry:
        %va = alloca i8*, align 4               ; <i8**> [#uses=4]
        %va.upgrd.1 = bitcast i8** %va to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_start( i8* %va.upgrd.1 )
        br label %bb

bb:             ; preds = %bb, %entry
        %a_addr.0 = phi i32 [ %a, %entry ], [ %tmp5, %bb ]              ; <i32> [#uses=2]
        %tmp = load volatile i8** %va           ; <i8*> [#uses=2]
        %tmp2 = getelementptr i8* %tmp, i32 4           ; <i8*> [#uses=1]
        store volatile i8* %tmp2, i8** %va
        %tmp5 = add i32 %a_addr.0, -1           ; <i32> [#uses=1]
        %tmp.upgrd.2 = icmp eq i32 %a_addr.0, 1         ; <i1> [#uses=1]
        br i1 %tmp.upgrd.2, label %bb7, label %bb

bb7:            ; preds = %bb
        %tmp3 = bitcast i8* %tmp to i32*                ; <i32*> [#uses=1]
        %tmp.upgrd.3 = load i32* %tmp3          ; <i32> [#uses=1]
        %tmp10 = call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @str, i32 0, i64 0), i32 %tmp.upgrd.3 )                ; <i32> [#uses=0]
        %va.upgrd.4 = bitcast i8** %va to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_end( i8* %va.upgrd.4 )
        ret void
}

declare void @llvm.va_start(i8*)

declare i32 @printf(i8*, ...)

declare void @llvm.va_end(i8*)
