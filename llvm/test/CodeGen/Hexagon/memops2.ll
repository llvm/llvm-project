; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Generate MemOps for V4 and above.


define void @f(i16* nocapture %p) nounwind {
entry:
; CHECK:  memh(r{{[0-9]+}}+#20) -= #1
  %add.ptr = getelementptr inbounds i16, i16* %p, i32 10
  %0 = load i16, i16* %add.ptr, align 2
  %conv2 = zext i16 %0 to i32
  %sub = add nsw i32 %conv2, 65535
  %conv1 = trunc i32 %sub to i16
  store i16 %conv1, i16* %add.ptr, align 2
  ret void
}

define void @g(i16* nocapture %p, i32 %i) nounwind {
entry:
; CHECK:  memh(r{{[0-9]+}}+#20) -= #1
  %add.ptr.sum = add i32 %i, 10
  %add.ptr1 = getelementptr inbounds i16, i16* %p, i32 %add.ptr.sum
  %0 = load i16, i16* %add.ptr1, align 2
  %conv3 = zext i16 %0 to i32
  %sub = add nsw i32 %conv3, 65535
  %conv2 = trunc i32 %sub to i16
  store i16 %conv2, i16* %add.ptr1, align 2
  ret void
}
