; both globals are write only, delete them.

; RUN: llvm-as < %s | opt -globalopt | llvm-dis | \
; RUN:   not grep internal

@G0 = internal global [58 x i8] c"asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd\00"         ; <[58 x i8]*> [#uses=1]
@G1 = internal global [4 x i32] [ i32 1, i32 2, i32 3, i32 4 ]          ; <[4 x i32]*> [#uses=1]

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare void @llvm.memset.i32(i8*, i8, i32, i32)

define void @foo() {
        %Blah = alloca [58 x i8]                ; <[58 x i8]*> [#uses=1]
        %tmp3 = bitcast [58 x i8]* %Blah to i8*         ; <i8*> [#uses=1]
        call void @llvm.memcpy.i32( i8* bitcast ([4 x i32]* @G1 to i8*), i8* %tmp3, i32 16, i32 1 )
        call void @llvm.memset.i32( i8* getelementptr ([58 x i8]* @G0, i32 0, i32 0), i8 17, i32 58, i32 1 )
        ret void
}


