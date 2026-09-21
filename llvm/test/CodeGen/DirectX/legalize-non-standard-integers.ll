; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define i1 @preserve_i1(i32 %value) {
; CHECK-LABEL: define i1 @preserve_i1(
; CHECK: [[RESULT:%.*]] = trunc i32 %value to i1
; CHECK: ret i1 [[RESULT]]
  %result = trunc i32 %value to i1
  ret i1 %result
}

define i1 @packed_v3i1(<3 x i1> %value) {
; CHECK-LABEL: define i1 @packed_v3i1(
; CHECK-NOT: {{(^| )i3( |$)}}
; CHECK: [[E0:%.*]] = extractelement <3 x i1> %value, i64 0
; CHECK: [[Z0:%.*]] = zext i1 [[E0]] to i32
; CHECK: [[E1:%.*]] = extractelement <3 x i1> %value, i64 1
; CHECK: [[Z1:%.*]] = zext i1 [[E1]] to i32
; CHECK: [[S1:%.*]] = shl i32 [[Z1]], 1
; CHECK: [[E2:%.*]] = extractelement <3 x i1> %value, i64 2
; CHECK: [[Z2:%.*]] = zext i1 [[E2]] to i32
; CHECK: [[S2:%.*]] = shl i32 [[Z2]], 2
; CHECK: [[PACKED:%.*]] = or i32
; CHECK: ret i1
  %packed = bitcast <3 x i1> %value to i3
  %result = icmp ne i3 %packed, 0
  ret i1 %result
}

define i32 @wrap_i3(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: define i32 @wrap_i3(
; CHECK-NOT: {{(^| )i3( |$)}}
; CHECK: [[SUM:%.*]] = add i32 %lhs, %rhs
; CHECK: [[WRAPPED:%.*]] = and i32 [[SUM]], 7
; CHECK: ret i32 [[WRAPPED]]
  %lhs.i3 = trunc i32 %lhs to i3
  %rhs.i3 = trunc i32 %rhs to i3
  %sum = add i3 %lhs.i3, %rhs.i3
  %result = zext i3 %sum to i32
  ret i32 %result
}

define i32 @nuw_i8(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: define i32 @nuw_i8(
; CHECK: [[LHS:%.*]] = and i32 %lhs, 255
; CHECK: [[RHS:%.*]] = and i32 %rhs, 255
; CHECK: [[SUM:%.*]] = add nuw i32 [[LHS]], [[RHS]]
; CHECK-NEXT: ret i32 [[SUM]]
  %lhs.i8 = trunc i32 %lhs to i8
  %rhs.i8 = trunc i32 %rhs to i8
  %sum = add nuw i8 %lhs.i8, %rhs.i8
  %result = zext i8 %sum to i32
  ret i32 %result
}

define i1 @signed_i5(i32 %value) {
; CHECK-LABEL: define i1 @signed_i5(
; CHECK-NOT: {{(^| )i5( |$)}}
; CHECK: [[SHL:%.*]] = shl i32 %value, 27
; CHECK: [[SIGNED:%.*]] = ashr i32 [[SHL]], 27
; CHECK: ret i1
  %narrow = trunc i32 %value to i5
  %result = icmp slt i5 %narrow, -2
  ret i1 %result
}

define i64 @wrap_i37(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: define i64 @wrap_i37(
; CHECK-NOT: {{(^| )i37( |$)}}
; CHECK: [[PRODUCT:%.*]] = mul i64 %lhs, %rhs
; CHECK: [[WRAPPED:%.*]] = and i64 [[PRODUCT]], 137438953471
; CHECK: ret i64 [[WRAPPED]]
  %lhs.i37 = trunc i64 %lhs to i37
  %rhs.i37 = trunc i64 %rhs to i37
  %product = mul i37 %lhs.i37, %rhs.i37
  %result = zext i37 %product to i64
  ret i64 %result
}

define i1 @trunc_i3_to_i1(i32 %value) {
; CHECK-LABEL: define i1 @trunc_i3_to_i1(
; CHECK-NOT: {{(^| )i3( |$)}}
; CHECK: [[RESULT:%.*]] = trunc i32 %value to i1
; CHECK: ret i1 [[RESULT]]
  %narrow = trunc i32 %value to i3
  %result = trunc i3 %narrow to i1
  ret i1 %result
}

define i16 @trunc_i37_to_i16(i64 %value) {
; CHECK-LABEL: define i16 @trunc_i37_to_i16(
; CHECK-NOT: {{(^| )i37( |$)}}
; CHECK: [[RESULT:%.*]] = trunc i64 %value to i16
; CHECK: ret i16 [[RESULT]]
  %narrow = trunc i64 %value to i37
  %result = trunc i37 %narrow to i16
  ret i16 %result
}

define i32 @fptosi_to_i3(float %value) {
; CHECK-LABEL: define i32 @fptosi_to_i3(
; CHECK-NOT: {{(^| )i3( |$)}}
; CHECK: [[CONVERT:%.*]] = fptosi float %value to i32
; CHECK: [[RESULT:%.*]] = and i32 [[CONVERT]], 7
; CHECK: ret i32 [[RESULT]]
  %narrow = fptosi float %value to i3
  %result = zext i3 %narrow to i32
  ret i32 %result
}

define float @sitofp_from_i3(i32 %value) {
; CHECK-LABEL: define float @sitofp_from_i3(
; CHECK-NOT: {{(^| )i3( |$)}}
; CHECK: [[SHL:%.*]] = shl i32 %value, 29
; CHECK: [[SIGNED:%.*]] = ashr i32 [[SHL]], 29
; CHECK: [[RESULT:%.*]] = sitofp i32 [[SIGNED]] to float
; CHECK: ret float [[RESULT]]
  %narrow = trunc i32 %value to i3
  %result = sitofp i3 %narrow to float
  ret float %result
}

define i32 @ptrtoint_to_i3(ptr %value) {
; CHECK-LABEL: define i32 @ptrtoint_to_i3(
; CHECK-NOT: {{(^| )i3( |$)}}
; CHECK: [[CONVERT:%.*]] = ptrtoint ptr %value to i32
; CHECK: [[RESULT:%.*]] = and i32 [[CONVERT]], 7
; CHECK: ret i32 [[RESULT]]
  %narrow = ptrtoint ptr %value to i3
  %result = zext i3 %narrow to i32
  ret i32 %result
}

define ptr @inttoptr_from_i37(i64 %value) {
; CHECK-LABEL: define ptr @inttoptr_from_i37(
; CHECK-NOT: {{(^| )i37( |$)}}
; CHECK: [[MASKED:%.*]] = and i64 %value, 137438953471
; CHECK: [[RESULT:%.*]] = inttoptr i64 [[MASKED]] to ptr
; CHECK: ret ptr [[RESULT]]
  %narrow = trunc i64 %value to i37
  %result = inttoptr i37 %narrow to ptr
  ret ptr %result
}

define i32 @weighted_select(i1 %condition, i32 %lhs, i32 %rhs) {
; CHECK-LABEL: define i32 @weighted_select(
; CHECK: [[SELECT:%.*]] = select i1 %condition, i32 %lhs, i32 %rhs, !prof [[PROF:![0-9]+]]
; CHECK: [[RESULT:%.*]] = and i32 [[SELECT]], 7
; CHECK: ret i32 [[RESULT]]
  %lhs.i3 = trunc i32 %lhs to i3
  %rhs.i3 = trunc i32 %rhs to i3
  %selected = select i1 %condition, i3 %lhs.i3, i3 %rhs.i3, !prof !0
  %result = zext i3 %selected to i32
  ret i32 %result
}

; CHECK: [[PROF]] = !{!"branch_weights", i32 1, i32 2}

!0 = !{!"branch_weights", i32 1, i32 2}
