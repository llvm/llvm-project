; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

@a = external local_unnamed_addr global ptr, align 8


define void @no_alias_phi(i1 %c, i64 %x) {
; CHECK-LABEL: no_alias_phi
; CHECK:  NoAlias:  i32* %arrayidx0, i32* %arrayidx1
entry:
  br i1 %c, label %bb1, label %bb2

bb1:
  %add1_ = add nsw i64 %x, 1
  br label %end

bb2:
  %add2_ = add nsw i64 %x, 2
  br label %end

end:
  %cond = phi i64 [%add1_, %bb1 ], [ %add2_, %bb2 ]
  %arrayidx0 = getelementptr inbounds i32, ptr @a, i64 %cond
  %arrayidx1 = getelementptr inbounds i32, ptr @a, i64 %x
  store i32 123, ptr %arrayidx0, align 4
  store i32 456, ptr %arrayidx1, align 4
  ret void
}

define void @may_alias_phi(i1 %c, i64 %x) {
; CHECK-LABEL: may_alias_phi
; CHECK:  MayAlias:  i32* %arrayidx0, i32* %arrayidx1
entry:
  br i1 %c, label %bb1, label %bb2

bb1:
  %add1_ = add nsw i64 %x, 1
  br label %end

bb2:
  br label %end

end:
  %cond = phi i64 [%add1_, %bb1 ], [ %x, %bb2 ]
  %arrayidx0 = getelementptr inbounds i32, ptr @a, i64 %cond
  %arrayidx1 = getelementptr inbounds i32, ptr @a, i64 %x
  store i32 123, ptr %arrayidx0, align 4
  store i32 456, ptr %arrayidx1, align 4
  ret void
}

define void @no_alias_phi_same_bb(i1 %c, i64 %x) {
; CHECK-LABEL: no_alias_phi_same_bb
; CHECK:  NoAlias:  i32* %arrayidx0, i32* %arrayidx1
entry:
  br i1 %c, label %bb1, label %bb2

bb1:
  %add1_ = add nsw i64 %x, 1
  %add3_ = add nsw i64 %x, 3
  br label %end

bb2:
  %add2_ = add nsw i64 %x, 2
  %add4_ = add nsw i64 %x, 4
  br label %end

end:
  %phi1_ = phi i64 [%add1_, %bb1 ], [ %add2_, %bb2 ]
  %phi2_ = phi i64 [%add3_, %bb1 ], [ %add4_, %bb2 ]
  %arrayidx0 = getelementptr inbounds i32, ptr @a, i64 %phi1_
  %arrayidx1 = getelementptr inbounds i32, ptr @a, i64 %phi2_
  store i32 123, ptr %arrayidx0, align 4
  store i32 456, ptr %arrayidx1, align 4
  ret void
}

define void @no_alias_phi_same_bb_2(i1 %c, i64 %x) {
; CHECK-LABEL: no_alias_phi_same_bb_2
; CHECK:  NoAlias:  i32* %arrayidx0, i32* %arrayidx1
entry:
  br i1 %c, label %bb1, label %bb2

bb1:
  %add1_ = add nsw i64 %x, 1
  br label %end

bb2:
  %add2_ = add nsw i64 %x, 2
  br label %end

end:
  %phi1_ = phi i64 [%x, %bb1 ], [ %add2_, %bb2 ]
  %phi2_ = phi i64 [%add1_, %bb1 ], [ %x, %bb2 ]
  %arrayidx0 = getelementptr inbounds i32, ptr @a, i64 %phi1_
  %arrayidx1 = getelementptr inbounds i32, ptr @a, i64 %phi2_
  store i32 123, ptr %arrayidx0, align 4
  store i32 456, ptr %arrayidx1, align 4
  ret void
}

define void @may_alias_phi_same_bb(i1 %c, i64 %x) {
; CHECK-LABEL: may_alias_phi_same_bb
; CHECK:  MayAlias:  i32* %arrayidx0, i32* %arrayidx1
entry:
  br i1 %c, label %bb1, label %bb2

bb1:
  br label %end

bb2:
  %add2_ = add nsw i64 %x, 2
  %add4_ = add nsw i64 %x, 4
  br label %end

end:
  %phi1_ = phi i64 [%x, %bb1 ], [ %add2_, %bb2 ]
  %phi2_ = phi i64 [%x, %bb1 ], [ %add4_, %bb2 ]
  %arrayidx0 = getelementptr inbounds i32, ptr @a, i64 %phi1_
  %arrayidx1 = getelementptr inbounds i32, ptr @a, i64 %phi2_
  store i32 123, ptr %arrayidx0, align 4
  store i32 456, ptr %arrayidx1, align 4
  ret void
}
