; RUN: llc -march=mipsel < %s | FileCheck %s 

%struct.S1 = type { i32, [41 x i8] }

@.str = private unnamed_addr constant [31 x i8] c"abcdefghijklmnopqrstuvwxyzABCD\00", align 1

define void @foo1(ptr %s1, i8 signext %n) nounwind {
entry:
; CHECK-NOT: call16(memcpy

  %arraydecay = getelementptr inbounds %struct.S1, ptr %s1, i32 0, i32 1, i32 0
  tail call void @llvm.memcpy.p0.p0.i32(ptr align 1 %arraydecay, ptr align 1 @.str, i32 31, i1 false)
  %arrayidx = getelementptr inbounds %struct.S1, ptr %s1, i32 0, i32 1, i32 40
  store i8 %n, ptr %arrayidx, align 1
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

