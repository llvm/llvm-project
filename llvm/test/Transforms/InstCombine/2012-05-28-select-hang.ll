; RUN: opt -S -instcombine < %s | FileCheck %s

@c = common global i8 0, align 1
@a = common global i8 0, align 1
@b = common global i8 0, align 1

define void @func() nounwind uwtable ssp {
entry:
  %0 = load i8* @c, align 1
  %conv = zext i8 %0 to i32
  %or = or i32 %conv, 1
  %conv1 = trunc i32 %or to i8
  store i8 %conv1, i8* @a, align 1
  %conv2 = zext i8 %conv1 to i32
  %neg = xor i32 %conv2, -1
  %and = and i32 1, %neg
  %conv3 = trunc i32 %and to i8
  store i8 %conv3, i8* @b, align 1
  %1 = load i8* @a, align 1
  %conv4 = zext i8 %1 to i32
  %conv5 = zext i8 %conv3 to i32
  %tobool = icmp ne i32 %conv4, 0
  br i1 %tobool, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %entry
  %tobool8 = icmp ne i32 %conv5, 0
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %2 = phi i1 [ false, %entry ], [ %tobool8, %land.rhs ]
  %land.ext = zext i1 %2 to i32
  %mul = mul nsw i32 3, %land.ext
  %conv9 = trunc i32 %mul to i8
  store i8 %conv9, i8* @a, align 1
  ret void

; CHECK: @func
; CHECK-NOT: select
}
