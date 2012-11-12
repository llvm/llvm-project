; both globals are write only, delete them.

; RUN: opt < %s -globalopt -S | not grep internal

@G0 = internal global [58 x i8] c"asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd\00"         ; <[58 x i8]*> [#uses=1]
@G1 = internal global [4 x i32] [ i32 1, i32 2, i32 3, i32 4 ]          ; <[4 x i32]*> [#uses=1]

define void @foo() {
  %Blah = alloca [58 x i8]
  %tmp3 = bitcast [58 x i8]* %Blah to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast ([4 x i32]* @G1 to i8*), i8* %tmp3, i32 16, i32 1, i1 false)
  call void @llvm.memset.p0i8.i32(i8* getelementptr inbounds ([58 x i8]* @G0, i32 0, i32 0), i8 17, i32 58, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
