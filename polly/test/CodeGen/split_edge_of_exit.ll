; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -verify-region-info -disable-output < %s
;
; This is a scop directly precedented by a region, i.e. the scop's entry is the
; region's exit block. This test is to ensure that the RegionInfo is correctly
; preserved.
;
; CHECK: Valid Region for Scop: region2 => return
;
define void @f1(i64* %A, i64 %N) nounwind {
entry:
  br label %region1

region1:
  %indvar1 = phi i64 [ 0, %entry ], [ %indvar1.next, %region1 ]
  fence seq_cst
  %indvar1.next = add nsw i64 %indvar1, 1
  %exitcond1 = icmp eq i64 %indvar1.next, %N
  br i1 %exitcond1, label %region2, label %region1

region2:
  %indvar2 = phi i64 [ 0, %region1 ], [ %indvar2.next, %region2 ]
  %scevgep2 = getelementptr i64, i64* %A, i64 %indvar2
  store i64 %indvar2, i64* %scevgep2
  %indvar2.next = add nsw i64 %indvar2, 1
  %exitcond2 = icmp eq i64 %indvar2.next, %N
  br i1 %exitcond2, label %return, label %region2

return:
  ret void
}
