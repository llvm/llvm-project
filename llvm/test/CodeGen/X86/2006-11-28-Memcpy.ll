; PR1022, PR1023
; RUN: llvm-as < %s | llc -march=x86 | \
; RUN:   grep 3721182122 | count 2
; RUN: llvm-as < %s | llc -march=x86 | \
; RUN:   grep -E {movl	_?bytes2} | count 1

@fmt = constant [4 x i8] c"%x\0A\00"            ; <[4 x i8]*> [#uses=2]
@bytes = constant [4 x i8] c"\AA\BB\CC\DD"              ; <[4 x i8]*> [#uses=1]
@bytes2 = global [4 x i8] c"\AA\BB\CC\DD"               ; <[4 x i8]*> [#uses=1]

define i32 @test1() nounwind {
        %y = alloca i32         ; <i32*> [#uses=2]
        %c = bitcast i32* %y to i8*             ; <i8*> [#uses=1]
        %z = getelementptr [4 x i8]* @bytes, i32 0, i32 0               ; <i8*> [#uses=1]
        call void @llvm.memcpy.i32( i8* %c, i8* %z, i32 4, i32 1 )
        %r = load i32* %y               ; <i32> [#uses=1]
        %t = bitcast [4 x i8]* @fmt to i8*              ; <i8*> [#uses=1]
        %tmp = call i32 (i8*, ...)* @printf( i8* %t, i32 %r )           ; <i32> [#uses=0]
        ret i32 0
}

define void @test2() nounwind {
        %y = alloca i32         ; <i32*> [#uses=2]
        %c = bitcast i32* %y to i8*             ; <i8*> [#uses=1]
        %z = getelementptr [4 x i8]* @bytes2, i32 0, i32 0              ; <i8*> [#uses=1]
        call void @llvm.memcpy.i32( i8* %c, i8* %z, i32 4, i32 1 )
        %r = load i32* %y               ; <i32> [#uses=1]
        %t = bitcast [4 x i8]* @fmt to i8*              ; <i8*> [#uses=1]
        %tmp = call i32 (i8*, ...)* @printf( i8* %t, i32 %r )           ; <i32> [#uses=0]
        ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare i32 @printf(i8*, ...)

