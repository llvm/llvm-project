; RUN: opt -O2 -disable-output < %s
; PR15714

%struct.ham = type { i32 }

@global5 = common global i32 0, align 4
@global6 = common global i32 0, align 4
@global7 = common global i32 0, align 4
@global = common global i32 0, align 4
@global8 = common global %struct.ham zeroinitializer, align 4
@global9 = common global i32 0, align 4
@global10 = common global i32 0, align 4
@global11 = common global i32 0, align 4

define void @zot12() {
bb:
  store i32 0, i32* @global5, align 4
  store i32 0, i32* @global6, align 4
  br label %bb2

bb1:                                              ; preds = %bb11
  %tmp = load i32, i32* @global5, align 4
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %tmp3 = phi i32 [ %tmp, %bb1 ], [ 0, %bb ]
  %tmp4 = xor i32 %tmp3, zext (i1 icmp ne (i64 ptrtoint (i32* @global5 to i64), i64 1) to i32)
  store i32 %tmp4, i32* @global5, align 4
  %tmp5 = icmp eq i32 %tmp3, zext (i1 icmp ne (i64 ptrtoint (i32* @global5 to i64), i64 1) to i32)
  br i1 %tmp5, label %bb8, label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = tail call i32 @quux13()
  br label %bb8

bb8:                                              ; preds = %bb6, %bb2
  %tmp9 = load i32, i32* @global7, align 4
  %tmp10 = icmp eq i32 %tmp9, 0
  br i1 %tmp10, label %bb11, label %bb15

bb11:                                             ; preds = %bb8
  %tmp12 = load i32, i32* @global6, align 4
  %tmp13 = add nsw i32 %tmp12, 1
  store i32 %tmp13, i32* @global6, align 4
  %tmp14 = icmp slt i32 %tmp13, 42
  br i1 %tmp14, label %bb1, label %bb15

bb15:                                             ; preds = %bb11, %bb8
  ret void
}

define i32 @quux13() {
bb:
  store i32 1, i32* @global5, align 4
  ret i32 1
}

define void @wombat() {
bb:
  tail call void @zot12()
  ret void
}

define void @wombat14() {
bb:
  tail call void @blam()
  ret void
}

define void @blam() {
bb:
  store i32 ptrtoint (i32* @global to i32), i32* getelementptr inbounds (%struct.ham, %struct.ham* @global8, i64 0, i32 0), align 4
  store i32 0, i32* @global9, align 4
  %tmp = load i32, i32* getelementptr inbounds (%struct.ham, %struct.ham* @global8, i64 0, i32 0), align 4
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %tmp2 = phi i32 [ 0, %bb ], [ %tmp11, %bb1 ]
  %tmp3 = phi i32 [ %tmp, %bb ], [ %tmp10, %bb1 ]
  %tmp4 = icmp sgt i32 %tmp3, 0
  %tmp5 = zext i1 %tmp4 to i32
  %tmp6 = urem i32 %tmp5, 5
  %tmp7 = mul i32 %tmp3, -80
  %tmp8 = or i32 %tmp7, %tmp6
  %tmp9 = icmp eq i32 %tmp8, 0
  %tmp10 = zext i1 %tmp9 to i32
  %tmp11 = add nsw i32 %tmp2, 1
  %tmp12 = icmp eq i32 %tmp11, 20
  br i1 %tmp12, label %bb13, label %bb1

bb13:                                             ; preds = %bb1
  store i32 %tmp10, i32* getelementptr inbounds (%struct.ham, %struct.ham* @global8, i64 0, i32 0), align 4
  store i32 0, i32* @global10, align 4
  store i32 %tmp6, i32* @global11, align 4
  store i32 20, i32* @global9, align 4
  ret void
}
