; This function contains two selects which differ only by having reversed
; operand order. They are in fact equivalent, since the icmps that preceede
; them also are the same except for reversed conditions. SelectionDAG will
; discover this and return the other SELECT_CCMASK when the operands are
; canonicalized. This must be handled by SystemZDAGToDAGISel::Select, or
; instruction selection will fail.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13


@g_1531 = external global [7 x i64], align 8
@g_277 = external global <{ i64, i8, i8, i8, i8, i8, i8 }>, align 8
@g_62.6 = external global i32, align 4

define dso_local void @fun() {
entry:
  %tmp = add nuw nsw i16 0, 238
  %tmp4 = sub nsw i16 %tmp, 0
  store i64 4, ptr getelementptr inbounds (<{ i64, i8, i8, i8, i8, i8, i8 }>, ptr @g_277, i64 0, i32 0), align 8
  %tmp5 = load i64, ptr getelementptr inbounds ([7 x i64], ptr @g_1531, i64 0, i64 5), align 8
  %tmp6 = trunc i64 %tmp5 to i32
  %tmp7 = trunc i64 %tmp5 to i16
  %tmp8 = shl i32 %tmp6, 24
  %tmp9 = ashr exact i32 %tmp8, 24
  %tmp10 = urem i16 %tmp7, %tmp4
  %tmp11 = icmp eq i16 %tmp10, 0
  %tmp12 = select i1 %tmp11, i32 0, i32 %tmp9
  %tmp13 = icmp sge i32 %tmp12, undef
  %tmp14 = zext i1 %tmp13 to i32
  %tmp15 = or i32 0, %tmp14
  %tmp16 = icmp ne i16 %tmp10, 0
  %tmp17 = select i1 %tmp16, i32 %tmp9, i32 0
  %tmp18 = icmp sge i32 %tmp17, undef
  %tmp19 = zext i1 %tmp18 to i32
  %tmp20 = or i32 %tmp15, %tmp19
  store i32 %tmp20, ptr @g_62.6, align 4
  unreachable
}
