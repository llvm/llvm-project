; REQUIRES: asserts
; RUN: opt -S -passes='loop-vectorize' -vectorizer-maximize-bandwidth -enable-epilogue-vectorization=false -debug-only=loop-vectorize 2>&1 < %s | FileCheck %s

target triple = "aarch64"

; CHECK: LV: Checking a loop in 'test_vf_8_better_vf_than_16_given_tripcount_of_24'
; CHECK: Cost for VF 2: 15 (Estimated cost per lane: 7.5)
; CHECK: Cost for VF 4: 9 (Estimated cost per lane: 2.2)
; CHECK: Cost for VF 8: 6 (Estimated cost per lane: 0.8)
; CHECK: Cost for VF 16: 6 (Estimated cost per lane: 0.4)
; CHECK: LV: VF 8 has lower cost than VF 16 when taking the cost of the remaining scalar loop iterations into consideration for a maximum trip count of 24.
; CHECK: LV: Selecting VF: 8.

define void @test_vf_8_better_vf_than_16_given_tripcount_of_24(ptr %a, ptr %b){
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %a.iv = getelementptr inbounds nuw i8, ptr %a, i64 %iv
  %a.ld = load i8, ptr %a.iv
  %a.ld.zext = zext i8 %a.ld to i32
  %b.iv = getelementptr inbounds nuw i8, ptr %b, i64 %iv
  %b.ld = load i8, ptr %b.iv
  %b.ld.zext = zext i8 %b.ld to i32
  %a.plus.b = add i32 %a.ld.zext, %b.ld.zext
  %a.plus.b.trunc = trunc i32 %a.plus.b to i8
  store i8 %a.plus.b.trunc, ptr %a.iv
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 24
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                 ; preds = %for.body
  ret void
}
