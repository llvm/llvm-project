; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Generate MemOps for V4 and above.


define void @f(i8* nocapture %p) nounwind {
entry:
; CHECK:  memb(r{{[0-9]+}}{{ *}}+{{ *}}#10){{ *}}-={{ *}}#1
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 10
  %0 = load i8, i8* %add.ptr, align 1
  %conv = zext i8 %0 to i32
  %sub = add nsw i32 %conv, 255
  %conv1 = trunc i32 %sub to i8
  store i8 %conv1, i8* %add.ptr, align 1
  ret void
}

define void @g(i8* nocapture %p, i32 %i) nounwind {
entry:
; CHECK:  memb(r{{[0-9]+}}{{ *}}+{{ *}}#10){{ *}}-={{ *}}#1
  %add.ptr.sum = add i32 %i, 10
  %add.ptr1 = getelementptr inbounds i8, i8* %p, i32 %add.ptr.sum
  %0 = load i8, i8* %add.ptr1, align 1
  %conv = zext i8 %0 to i32
  %sub = add nsw i32 %conv, 255
  %conv2 = trunc i32 %sub to i8
  store i8 %conv2, i8* %add.ptr1, align 1
  ret void
}
