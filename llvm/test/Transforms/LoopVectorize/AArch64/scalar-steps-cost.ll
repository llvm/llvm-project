; REQUIRES: asserts
; RUN: opt -p loop-vectorize -mtriple=arm64-apple-macosx -S -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

define i32 @scalar_steps_all_lanes(ptr %start, ptr %end) {
; CHECK-LABEL: LV: Checking a loop in 'scalar_steps_all_lanes'
; CHECK: Cost of 0 for VF 2: {{.*}} = SCALAR-STEPS {{.*}}, ir<1>, {{.*}}
; CHECK: Cost of 0 for VF 4: {{.*}} = SCALAR-STEPS {{.*}}, ir<1>, {{.*}}
entry:
  br label %loop

loop:
  %iv = phi ptr [ %start, %entry ], [ %iv.next, %loop ]
  %rdx = phi i32 [ 0, %entry ], [ %rdx.next, %loop ]
  %l = load i32, ptr %iv, align 1
  %1 = and i32 %l, 1
  %rdx.next = or i32 %rdx, %1
  %iv.next = getelementptr i8, ptr %iv, i64 1
  %ec = icmp ult ptr %iv, %end
  br i1 %ec, label %loop, label %exit

exit:
  ret i32 %rdx.next
}

define void @scalar_steps_first_lane_only(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: LV: Checking a loop in 'scalar_steps_first_lane_only'
; CHECK: Cost of 0 for VF 2: {{.*}} = SCALAR-STEPS {{.*}}, ir<1>, {{.*}}
; CHECK: Cost of 0 for VF 4: {{.*}} = SCALAR-STEPS {{.*}}, ir<1>, {{.*}}
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep1 = getelementptr inbounds i32, ptr %b, i64 %iv
  %ld1 = load i32, ptr %gep1
  %gep2 = getelementptr inbounds i32, ptr %c, i64 %iv
  %ld2 = load i32, ptr %gep2
  %add = add nsw i32 %ld2, %ld1
  %gep3 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %gep3
  %iv.next = add i64 %iv, 1
  %iv.trunc = trunc i64 %iv.next to i32
  %exitcond = icmp eq i32 %iv.trunc, 128
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

