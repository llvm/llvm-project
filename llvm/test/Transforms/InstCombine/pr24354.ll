; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; This used to crash opt

@c = common global i32 0, align 4
@b = common global i32 0, align 4
@a = common global i16 0, align 2
@d = common global i32 0, align 4

define void @fn3() {
; CHECK: @fn3
bb:
  %tmp = load i32, ptr @c, align 4
  %tmp1 = icmp eq i32 %tmp, 0
  br i1 %tmp1, label %bb2, label %bb6

bb2:                                              ; preds = %bb
  %tmp3 = load i32, ptr @b, align 4
  %tmp.i = add nsw i32 255, %tmp3
  %tmp5 = icmp ugt i32 %tmp.i, 254
  br label %bb6

bb6:                                              ; preds = %bb, %bb2
  %tmp7 = phi i1 [ true, %bb ], [ %tmp5, %bb2 ]
  %tmp8 = zext i1 %tmp7 to i32
  %tmp10 = icmp eq i32 %tmp8, 0
  %tmp12 = load i16, ptr @a, align 2
  %tmp14 = icmp ne i16 %tmp12, 0
  %tmp16 = select i1 %tmp10, i1 false, i1 %tmp14
  %tmp17 = zext i1 %tmp16 to i32
  store i32 %tmp17, ptr @d, align 4
  ret void
}
