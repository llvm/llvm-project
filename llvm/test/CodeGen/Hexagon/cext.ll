; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: memub(r{{[0-9]+}}<<#1+##a)

@a = external global [5 x [2 x i8]]

define zeroext i8 @foo(i8 zeroext %l) nounwind readonly {
for.end:
  %idxprom = zext i8 %l to i32
  %arrayidx1 = getelementptr inbounds [5 x [2 x i8]], [5 x [2 x i8]]* @a, i32 0, i32 %idxprom, i32 0
  %0 = load i8, i8* %arrayidx1, align 1
  %conv = zext i8 %0 to i32
  %mul = mul nsw i32 %conv, 20
  %conv2 = trunc i32 %mul to i8
  ret i8 %conv2
}

