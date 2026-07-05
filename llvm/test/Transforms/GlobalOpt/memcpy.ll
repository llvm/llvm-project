; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK: G1 = internal unnamed_addr constant

@G1 = internal global [58 x i8] c"asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd\00"         ; <ptr> [#uses=1]

define void @foo() {
  %Blah = alloca [58 x i8]
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %Blah, ptr align 1 @G1, i32 58, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
