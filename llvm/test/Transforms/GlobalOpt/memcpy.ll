; RUN: llvm-as < %s | opt -globalopt | llvm-dis | \
; RUN:   grep {G1 = internal constant}

@G1 = internal global [58 x i8] c"asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd\00"         ; <[58 x i8]*> [#uses=1]

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

define void @foo() {
        %Blah = alloca [58 x i8]                ; <[58 x i8]*> [#uses=1]
        %tmp.0 = getelementptr [58 x i8]* %Blah, i32 0, i32 0           ; <i8*> [#uses=1]
        call void @llvm.memcpy.i32( i8* %tmp.0, i8* getelementptr ([58 x i8]* @G1, i32 0, i32 0), i32 58, i32 1 )
        ret void
}


