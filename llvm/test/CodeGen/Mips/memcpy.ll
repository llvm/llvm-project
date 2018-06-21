; RUN: llc -march=mipsel < %s | FileCheck %s 

%struct.S1 = type { i32, [41 x i8] }

@.str = private unnamed_addr constant [31 x i8] c"abcdefghijklmnopqrstuvwxyzABCD\00", align 1

define void @foo1(%struct.S1* %s1, i8 signext %n) nounwind {
entry:
; CHECK-NOT: call16(memcpy

  %arraydecay = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 1, i32 0
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %arraydecay, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str, i32 0, i32 0), i32 31, i32 1, i1 false)
  %arrayidx = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 1, i32 40
  store i8 %n, i8* %arrayidx, align 1
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

