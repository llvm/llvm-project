; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @foo(i32) local_unnamed_addr #0  {
  %2 = icmp sgt i32 %0, 2
  %3 = add nsw i32 %0, 20
  %4 = add nsw i32 %0, -20
  select i1 %2, i32 %3, i32 %4, !prof !1
  ret i32 %5
; CHECK-LABEL: @foo
; CHECK: select i1 %2, {{.*}} !prof ![[MD1:[0-9]+]]
}

define void @min_max_bitcast(<4 x float> %a, <4 x float> %b, <4 x i32>* %ptr1, <4 x i32>* %ptr2) {
  %cmp = fcmp olt <4 x float> %a, %b
  %bc1 = bitcast <4 x float> %a to <4 x i32>
  %bc2 = bitcast <4 x float> %b to <4 x i32>
  %sel1 = select <4 x i1> %cmp, <4 x i32> %bc1, <4 x i32> %bc2, !prof !1
  %sel2 = select <4 x i1> %cmp, <4 x i32> %bc2, <4 x i32> %bc1, !prof !1
  store <4 x i32> %sel1, <4 x i32>* %ptr1
  store <4 x i32> %sel2, <4 x i32>* %ptr2
  ret void
; CHECK-LABEL: @min_max_bitcast
; CHECK: select {{.*}} %cmp,{{.*}}!prof ![[MD1]]
}

define i32 @foo2(i32, i32) local_unnamed_addr #0  {
  %3 = icmp sgt i32 %0, 2
  %4 = add nsw i32 %0, %1
  %5 = sub nsw i32 %0, %1
  select i1 %3, i32 %4, i32 %5, !prof !1
  ret i32 %6
; CHECK-LABEL: @foo2
; CHECK: select i1 %3, {{.*}}, !prof ![[MD1]]
}

; condition swapped
define i64 @test43(i32 %a) nounwind {
  %a_ext = sext i32 %a to i64
  %is_a_nonnegative = icmp sgt i32 %a, -1
  %max = select i1 %is_a_nonnegative, i64 %a_ext, i64 0, !prof !1
  ret i64 %max
; CHECK-LABEL: @test43
; CHECK: select {{.*}}, i64 0, i64 %a_ext, !prof ![[MD3:[0-9]+]]
}

define <2 x i32> @scalar_select_of_vectors_sext(<2 x i1> %cca, i1 %ccb) {
  %ccax = sext <2 x i1> %cca to <2 x i32>
  %r = select i1 %ccb, <2 x i32> %ccax, <2 x i32> <i32 0, i32 0>, !prof !1
  ret <2 x i32> %r
; CHECK-LABEL: @scalar_select_of_vectors_sext(
; CHECK-NEXT:    [[FOLD_R:%.*]] = select i1 %ccb, {{.*}}, !prof ![[MD1]]
; CHECK-NEXT:    [[R:%.*]] = sext <2 x i1> [[FOLD_R]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[R]]
}


define i16 @t7(i32 %a) {
  %1 = icmp slt i32 %a, -32768
  %2 = trunc i32 %a to i16
  %3 = select i1 %1, i16 %2, i16 -32768, !prof !1
  ret i16 %3
}
; CHECK-LABEL: @t7
; CHECK-NEXT: icmp
; CHECK-NEXT: select i1 %1{{.*}}, !prof ![[MD1]]
; CHECK-NEXT: trunc


define i32 @abs_nabs_x01(i32 %x) {
  %cmp = icmp sgt i32 %x, -1
  %sub = sub nsw i32 0, %x
  %cond = select i1 %cmp, i32 %sub, i32 %x, !prof !1
  %cmp1 = icmp sgt i32 %cond, -1
  %sub16 = sub nsw i32 0, %cond
  %cond18 = select i1 %cmp1, i32 %cond, i32 %sub16, !prof !2
  ret i32 %cond18
; CHECK-LABEL: @abs_nabs_x01(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp sgt i32 %x, -1
; CHECK-NEXT: [[NEG:%[a-z0-9]+]] = sub nsw i32 0, %x
; CHECK-NEXT: [[SEL:%[a-z0-9]+]] = select i1 [[CMP]], i32 %x, i32 [[NEG]], !prof ![[MD1]]
}


; SMAX(SMAX(x, y), x) -> SMAX(x, y)
define i32 @test30(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y, !prof !1
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x, !prof !2
  ret i32 %retval
; CHECK-LABEL: @test30
; CHECK: select {{.*}}, !prof ![[MD1]]
}

define i32 @test70(i32 %x) {
entry:
  %cmp = icmp slt i32 %x, 75
  %cond = select i1 %cmp, i32 75, i32 %x, !prof !1
  %cmp3 = icmp slt i32 %cond, 36
  %retval = select i1 %cmp3, i32 36, i32 %cond, !prof !2
  ret i32 %retval
; CHECK-LABEL: @test70
; CHECK: select {{.*}}, !prof ![[MD1]]
}


; SMIN(SMIN(X, 92), 11) -> SMIN(X, 11)
define i32 @test72(i32 %x) {
  %cmp = icmp sgt i32 %x, 92
  %cond = select i1 %cmp, i32 92, i32 %x, !prof !1
  %cmp3 = icmp sgt i32 %cond, 11
  %retval = select i1 %cmp3, i32 11, i32 %cond, !prof !2
  ret i32 %retval
; CHECK-LABEL: @test72
; CHECK: select {{.*}}, !prof ![[MD2:[0-9]+]]
}

; SMAX(SMAX(X, 36), 75) -> SMAX(X, 75)
define i32 @test74(i32 %x) {
  %cmp = icmp slt i32 %x, 36
  %cond = select i1 %cmp, i32 36, i32 %x, !prof !1
  %cmp3 = icmp slt i32 %cond, 75
  %retval = select i1 %cmp3, i32 75, i32 %cond, !prof !2
  ret i32 %retval
; CHECK-LABEL: @test74
; CHECK: select {{.*}}, !prof ![[MD2]]
}

!1 = !{!"branch_weights", i32 2, i32 10}
!2 = !{!"branch_weights", i32 3, i32 10}

; CHECK-DAG: ![[MD1]] = !{!"branch_weights", i32 2, i32 10}
; CHECK-DAG: ![[MD2]] = !{!"branch_weights", i32 3, i32 10}
; CHECK-DAG: ![[MD3]] = !{!"branch_weights", i32 10, i32 2}

