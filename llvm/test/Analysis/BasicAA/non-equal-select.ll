; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
@G = global [10 x i32] zeroinitializer, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define void @select_in_gep1(i1 %c, i64 noundef %x) {
entry:
; CHECK: Function: select_in_gep1
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

define void @select_in_gep2(i1 %c, i64 noundef %x) {
entry:
  ; TODO: should be "NoAlias" here as well.
; CHECK: Function: select_in_gep2
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
