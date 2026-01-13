; RUN: opt -S -passes=instcombine < %s | FileCheck %s

@c = common global i8 0, align 1
@a = common global i8 0, align 1
@b = common global i8 0, align 1

define void @func() nounwind uwtable ssp {
entry:
  %0 = load i8, ptr @c, align 1
  %conv = zext i8 %0 to i32
  %or = or i32 %conv, 1
  %conv1 = trunc i32 %or to i8
  store i8 %conv1, ptr @a, align 1
  %conv2 = zext i8 %conv1 to i32
  %neg = xor i32 %conv2, -1
  %and = and i32 1, %neg
  %conv3 = trunc i32 %and to i8
  store i8 %conv3, ptr @b, align 1
  %1 = load i8, ptr @a, align 1
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
  store i8 %conv9, ptr @a, align 1
  ret void

; CHECK-LABEL: @func(
; CHECK-NOT: select
}

define i1 @select_no_infinite_loop(i2 %arg0, i2 %arg1, i1 %arg2, i2 %arg3) {
; CHECK-LABEL: define i1 @select_no_infinite_loop(
; CHECK-SAME: i2 [[ARG0:%.*]], i2 [[ARG1:%.*]], i1 [[ARG2:%.*]], i2 [[ARG3:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[DOTFR:%.*]] = freeze i2 [[ARG3]]
; CHECK-NEXT:    [[I:%.*]] = icmp ne i2 [[DOTFR]], -1
; CHECK-NEXT:    ret i1 [[I]]
;
entry:
  %cmp0 = icmp sgt i2 0, %arg0
  %zext = zext i1 %arg2 to i2
  %sel0 = select i1 %cmp0, i2 0, i2 %zext
  %trunc = trunc i2 %sel0 to i1
  %sel1 = select i1 %trunc, i2 %arg1, i2 0
  %cmp1 = icmp sle i1 %cmp0, %trunc
  %sel2 = select i1 %cmp1, i2 0, i2 %arg1
  %and = and i2 %zext, %sel2
  %sel3 = select i1 %arg2, i2 0, i2 %arg3
  %div = sdiv i2 1, %sel3
  %cmp2 = icmp uge i2 %sel1, %and
  %sext = sext i1 %cmp2 to i2
  %cmp3 = icmp sgt i2 %div, %sext
  %cmp4 = icmp sgt i2 %sel0, 0
  %lshr = lshr i1 %cmp3, %cmp4
  ret i1 %lshr
}
