; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Generate MemOps for V4 and above.


define void @f(ptr nocapture %p) nounwind {
entry:
; CHECK:  memb(r{{[0-9]+}}+#10) -= #1
  %add.ptr = getelementptr inbounds i8, ptr %p, i32 10
  %0 = load i8, ptr %add.ptr, align 1
  %conv = zext i8 %0 to i32
  %sub = add nsw i32 %conv, 255
  %conv1 = trunc i32 %sub to i8
  store i8 %conv1, ptr %add.ptr, align 1
  ret void
}

define void @g(ptr nocapture %p, i32 %i) nounwind {
entry:
; CHECK:  memb(r{{[0-9]+}}+#10) -= #1
  %add.ptr.sum = add i32 %i, 10
  %add.ptr1 = getelementptr inbounds i8, ptr %p, i32 %add.ptr.sum
  %0 = load i8, ptr %add.ptr1, align 1
  %conv = zext i8 %0 to i32
  %sub = add nsw i32 %conv, 255
  %conv2 = trunc i32 %sub to i8
  store i8 %conv2, ptr %add.ptr1, align 1
  ret void
}
