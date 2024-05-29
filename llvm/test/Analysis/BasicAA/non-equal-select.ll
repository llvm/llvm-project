; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
@G = global [10 x i32] zeroinitializer, align 4

define void @select_in_gep1(i1 %c, i64 %x) {
entry:
; CHECK-LABEL: Function: select_in_gep1
; CHECK: NoAlias: i32* %arrayidx1, i32* %arrayidx2
  %add1_ = add nsw i64 %x, 1
  %add2_ = add nsw i64 %x, 2
  %select_ = select i1 %c, i64 %add1_, i64 %add2_
  %arrayidx1 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %select_
  store i32 42, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %x
  store i32 43, ptr %arrayidx2, align 4
  ret void
}

define void @select_in_gep2(i1 %c, i64 %x) {
entry:
  ; TODO: should be "NoAlias" here as well.
; CHECK-LABEL: Function: select_in_gep2
; CHECK: MayAlias:     i32* %arrayidx1, i32* %arrayidx2
  %add1_ = add nsw i64 %x, 1
  %add2_ = add nsw i64 %x, 2
  %add3_ = add nsw i64 %x, 3
  %select_ = select i1 %c, i64 %add1_, i64 %add2_
  %arrayidx1 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %select_
  store i32 42, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %add3_
  store i32 43, ptr %arrayidx2, align 4
  ret void
}

define void @two_selects_in_gep_same_cond(i1 %c, i64 %x) {
entry:
; CHECK-LABEL: Function: two_selects_in_gep_same_cond
; CHECK: NoAlias: i32* %arrayidx1, i32* %arrayidx2
  %add1_ = add nsw i64 %x, 1
  %add2_ = add nsw i64 %x, 2
  %select1_ = select i1 %c, i64 %x, i64 %add1_
  %select2_ = select i1 %c, i64 %add2_, i64 %x
  %arrayidx1 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %select1_
  store i32 42, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %select2_
  store i32 43, ptr %arrayidx2, align 4
  ret void
}

define void @two_selects_in_gep_different_cond1(i1 %c1, i1 %c2, i64 %x) {
entry:
; CHECK-LABEL: Function: two_selects_in_gep_different_cond1
; CHECK: NoAlias: i32* %arrayidx1, i32* %arrayidx2
  %add1_ = add nsw i64 %x, 1
  %add2_ = add nsw i64 %x, 2
  %add3_ = add nsw i64 %x, 3
  %add4_ = add nsw i64 %x, 4
  %select1_ = select i1 %c1, i64 %add1_, i64 %add2_
  %select2_ = select i1 %c2, i64 %add3_, i64 %add4_
  %arrayidx1 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %select1_
  store i32 42, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %select2_
  store i32 43, ptr %arrayidx2, align 4
  ret void
}

define void @two_selects_in_gep_different_cond2(i1 %c1, i1 %c2, i64 %x) {
entry:
; CHECK-LABEL: Function: two_selects_in_gep_different_cond2
; CHECK: MayAlias: i32* %arrayidx1, i32* %arrayidx2
  %add1_ = add nsw i64 %x, 1
  %add2_ = add nsw i64 %x, 2
  %select1_ = select i1 %c1, i64 %x, i64 %add1_
  %select2_ = select i1 %c2, i64 %x, i64 %add2_
  %arrayidx1 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %select1_
  store i32 42, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds [10 x i32], ptr @G, i64 0, i64 %select2_
  store i32 43, ptr %arrayidx2, align 4
  ret void
}
