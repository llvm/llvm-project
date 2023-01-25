; RUN: opt -opaque-pointers=0 -S -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=2 < %s | FileCheck --check-prefixes=CHECK,VEC %s
; RUN: opt -opaque-pointers=0 -S -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=1 < %s | FileCheck --check-prefixes=CHECK %s

; CHECK-LABEL: @postinc
; CHECK-LABEL: scalar.ph:
; CHECK: %bc.resume.val = phi i32 [ %n.vec, %middle.block ], [ 0, %entry ]
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32 [ {{.*}}, %for.body ], [ %n.vec, %middle.block ]
; CHECK: ret i32 %[[RET]]
define i32 @postinc(i32 %k)  {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %inc = add nsw i32 %inc.phi, 1
  %cmp = icmp eq i32 %inc, %k
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %inc
}

; CHECK-LABEL: @preinc
; CHECK-LABEL: middle.block:
; CHECK: %[[v3:.+]] = sub i32 %n.vec, 1
; CHECK-LABEL: scalar.ph:
; CHECK: %bc.resume.val = phi i32 [ %n.vec, %middle.block ], [ 0, %entry ]
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32 [ {{.*}}, %for.body ], [ %[[v3]], %middle.block ]
; CHECK: ret i32 %[[RET]]
define i32 @preinc(i32 %k)  {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %inc = add nsw i32 %inc.phi, 1
  %cmp = icmp eq i32 %inc, %k
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %inc.phi
}

; CHECK-LABEL: @constpre
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32 [ {{.*}}, %for.body ], [ 2, %middle.block ]
; CHECK: ret i32 %[[RET]]
define i32 @constpre()  {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 32, %entry ], [ %inc, %for.body ]
  %inc = sub nsw i32 %inc.phi, 2
  %cmp = icmp eq i32 %inc, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %inc.phi
}

; CHECK-LABEL: @geppre
; CHECK-LABEL: middle.block:
; CHECK: %ind.escape = getelementptr i32, i32* %ptr, i64 124
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32* [ {{.*}}, %for.body ], [ %ind.escape, %middle.block ]
; CHECK: ret i32* %[[RET]]
define i32* @geppre(i32* %ptr) {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %ptr.phi = phi i32* [ %ptr, %entry ], [ %inc.ptr, %for.body ]
  %inc = add nsw i32 %inc.phi, 1
  %inc.ptr = getelementptr i32, i32* %ptr.phi, i32 4
  %cmp = icmp eq i32 %inc, 32
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32* %ptr.phi
}

; CHECK-LABEL: @both
; CHECK-LABEL: middle.block:
; CHECK: %[[END:.*]] = sub i64 %n.vec, 1
; CHECK: %ind.escape = getelementptr i32, i32* %base, i64 %[[END]]
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32* [ %inc.lag1, %for.body ], [ %ind.escape, %middle.block ]
; CHECK: ret i32* %[[RET]]

define i32* @both(i32 %k)  {
entry:
  %base = getelementptr inbounds i32, i32* undef, i64 1
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %inc.lag1 = phi i32* [ %base, %entry ], [ %tmp, %for.body]
  %inc.lag2 = phi i32* [ undef, %entry ], [ %inc.lag1, %for.body]
  %tmp = getelementptr inbounds i32, i32* %inc.lag1, i64 1
  %inc = add nsw i32 %inc.phi, 1
  %cmp = icmp eq i32 %inc, %k
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32* %inc.lag1
}

; CHECK-LABEL: @multiphi
; CHECK-LABEL: scalar.ph:
; CHECK: %bc.resume.val = phi i32 [ %n.vec, %middle.block ], [ 0, %entry ]
; CHECK-LABEL: for.end:
; CHECK: %phi = phi i32 [ {{.*}}, %for.body ], [ %n.vec, %middle.block ]
; CHECK: %phi2 = phi i32 [ {{.*}}, %for.body ], [ %n.vec, %middle.block ]
; CHECK: store i32 %phi2, i32* %p
; CHECK: ret i32 %phi
define i32 @multiphi(i32 %k, i32* %p)  {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %inc = add nsw i32 %inc.phi, 1
  %cmp = icmp eq i32 %inc, %k
  br i1 %cmp, label %for.end, label %for.body

for.end:
  %phi = phi i32 [ %inc, %for.body ]
  %phi2 = phi i32 [ %inc, %for.body ]
  store i32 %phi2, i32* %p
  ret i32 %phi
}

; CHECK-LABEL: @PR30742
; CHECK:   %[[T15:.+]] = add nsw i32 %tmp03, -7
; CHECK: vector.ph
; CHECK:   %[[N_MOD_VF:.+]] = urem i32 %[[T5:.+]], 2
; CHECK:   %[[N_VEC:.+]] = sub i32 %[[T5]], %[[N_MOD_VF]]
; CHECK: middle.block
; CHECK:   %[[CMP:.+]] = icmp eq i32 %[[T5]], %[[N_VEC]]
; CHECK:   %ind.escape = add i32 %[[T15]],
; CHECK:   br i1 %[[CMP]], label %BB3, label %scalar.ph
define void @PR30742() {
BB0:
  br label %BB1

BB1:
  %tmp00 = load i32, i32* undef, align 16
  %tmp01 = sub i32 %tmp00, undef
  %tmp02 = icmp slt i32 %tmp01, 1
  %tmp03 = select i1 %tmp02, i32 1, i32 %tmp01
  %tmp04 = add nsw i32 %tmp03, -7
  br label %BB2

BB2:
  %tmp05 = phi i32 [ %tmp04, %BB1 ], [ %tmp06, %BB2 ]
  %tmp06 = add i32 %tmp05, -8
  %tmp07 = icmp sgt i32 %tmp06, 0
  br i1 %tmp07, label %BB2, label %BB3

BB3:
  %tmp08 = phi i32 [ %tmp05, %BB2 ]
  %tmp09 = sub i32 %tmp00, undef
  %tmp10 = icmp slt i32 %tmp09, 1
  %tmp11 = select i1 %tmp10, i32 1, i32 %tmp09
  %tmp12 = add nsw i32 %tmp11, -7
  br label %BB4

BB4:
  %tmp13 = phi i32 [ %tmp12, %BB3 ], [ %tmp14, %BB4 ]
  %tmp14 = add i32 %tmp13, -8
  %tmp15 = icmp sgt i32 %tmp14, 0
  br i1 %tmp15, label %BB4, label %BB1
}

; CHECK-LABEL: @iv_scalar_steps_and_outside_users
; CHECK-LABEL: scalar.ph:
; CHECK-NEXT:    %bc.resume.val = phi i64 [ 1002, %middle.block ], [ 0, %entry ]
; CHECK-LABEL: exit:
; CHECK-NEXT:    %iv.lcssa = phi i64 [ %iv, %loop ], [ 1001, %middle.block ]
;
define i64 @iv_scalar_steps_and_outside_users(i64* %ptr) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw i64 %iv, 1
  %gep.ptr = getelementptr inbounds i64, i64* %ptr, i64 %iv
  store i64 %iv, i64* %gep.ptr
  %exitcond = icmp ugt i64 %iv, 1000
  br i1 %exitcond, label %exit, label %loop

exit:
  %iv.lcssa = phi i64 [ %iv, %loop ]
  ret i64 %iv.lcssa
}


; %iv.2 is dead in the vector loop and only used outside the loop.
define i32 @iv_2_dead_in_loop_only_used_outside(i64* %ptr) {
; CHECK-LABEL: @iv_2_dead_in_loop_only_used_outside
; CHECK-LABEL: vector.body:
; CHECK-NEXT:   [[INDEX:%.+]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.+]], %vector.body ]
; VEC-NEXT:     [[VEC_IND:%.+]] = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ [[VEC_IND_NEXT:%.+]], %vector.body ]
; CHECK:        [[IV_0:%.+]] = add i64 [[INDEX]], 0
; VEC-NOT:      add i64 [[INDEX]], 1
; CHECK-NOT:    [[IV_2_0:%.+]] = add i32 %offset.idx, 0
; CHECK-LABEL: scalar.ph:
; CHECK-NEXT:    {{.+}} = phi i64 [ 1002, %middle.block ], [ 0, %entry ]
; CHECK-NEXT:    {{.+}} = phi i32 [ 2004, %middle.block ], [ 0, %entry ]
; CHECK-LABEL: exit:
; CHECK-NEXT:    %iv.2.lcssa = phi i32 [ %iv.2, %loop ], [ 2002, %middle.block ]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.2 = phi i32 [ 0, %entry ], [ %iv.2.next, %loop ]
  %iv.next = add nuw i64 %iv, 1
  %iv.2.next = add nuw i32 %iv.2, 2
  %gep.ptr = getelementptr inbounds i64, i64* %ptr, i64 %iv
  store i64 %iv, i64* %gep.ptr
  %exitcond = icmp ugt i64 %iv, 1000
  br i1 %exitcond, label %exit, label %loop

exit:
  %iv.2.lcssa = phi i32 [ %iv.2, %loop ]
  ret i32 %iv.2.lcssa
}
