; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: memuh(r{{[0-9]+}}<<#2+##a)

@a = external global [5 x [2 x i16]]

define signext i16 @foo(i16 zeroext %l) nounwind readonly {
for.end:
  %idxprom = zext i16 %l to i32
  %arrayidx1 = getelementptr inbounds [5 x [2 x i16]], ptr @a, i32 0, i32 %idxprom, i32 0
  %0 = load i16, ptr %arrayidx1, align 2
  %conv = zext i16 %0 to i32
  %mul = mul nsw i32 %conv, 20
  %conv2 = trunc i32 %mul to i16
  ret i16 %conv2
}

