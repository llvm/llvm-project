; RUN: opt < %s -globalopt -S | FileCheck %s
; CHECK: G1 = internal unnamed_addr constant

@G1 = internal global [58 x i8] c"asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd\00"         ; <[58 x i8]*> [#uses=1]

define void @foo() {
  %Blah = alloca [58 x i8]
  %tmp.0 = getelementptr [58 x i8], [58 x i8]* %Blah, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %tmp.0, i8* align 1 getelementptr inbounds ([58 x i8], [58 x i8]* @G1, i32 0, i32 0), i32 58, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
