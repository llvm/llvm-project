; RUN: opt < %s -instrprof -do-counter-promotion=true -speculative-counter-promotion-max-exiting=3 -S | FileCheck --check-prefix=PROMO %s
; RUN: opt < %s --passes=instrprof -do-counter-promotion=true -speculative-counter-promotion-max-exiting=3 -S | FileCheck --check-prefix=PROMO %s

$__llvm_profile_raw_version = comdat any

@g = common local_unnamed_addr global i32 0, align 4
@__llvm_profile_raw_version = constant i64 72057594037927940, comdat
@__profn_foo = private constant [3 x i8] c"foo"

define void @foo(i32 %arg) local_unnamed_addr {
bb:
  %tmp = add nsw i32 %arg, -1
  br label %bb1

bb1:                                              ; preds = %bb11, %bb
  %tmp2 = phi i32 [ 0, %bb ], [ %tmp12, %bb11 ]
  %tmp3 = icmp sgt i32 %tmp2, %arg
  br i1 %tmp3, label %bb7, label %bb4

bb4:                                              ; preds = %bb1
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 63969943867, i32 5, i32 1)
  tail call void @bar(i32 1)
  %tmp5 = load i32, i32* @g, align 4
  %tmp6 = icmp sgt i32 %tmp5, 100
  br i1 %tmp6, label %bb14, label %bb11

bb7:                                              ; preds = %bb1
  %tmp8 = icmp slt i32 %tmp2, %tmp
  br i1 %tmp8, label %bb9, label %bb10

bb9:                                              ; preds = %bb7
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 63969943867, i32 5, i32 2)
  tail call void @bar(i32 2)
  br label %bb11

bb10:                                             ; preds = %bb7
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 63969943867, i32 5, i32 3)
  tail call void @bar(i32 3)
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9, %bb4
  %tmp12 = add nuw nsw i32 %tmp2, 1
  %tmp13 = icmp slt i32 %tmp2, 99
  br i1 %tmp13, label %bb1, label %bb14

bb14:                                             ; preds = %bb4.bb14_crit_edge, %bb11
  tail call void @bar(i32 0)
  br label %bb15
; PROMO-LABEL: bb14:
; PROMO: %[[MERGE1:[a-z0-9]+]] = phi {{.*}}
; PROMO-NEXT: %[[MERGE2:[a-z0-9.]+]] = phi {{.*}}
; PROMO-NEXT: %[[MERGE3:[a-z0-9.]+]] = phi {{.*}}
; PROMO-NEXT: %[[PROMO3:[a-z0-9.]+]] = load{{.*}}@__profc_foo{{.*}}1)
; PROMO-NEXT: {{.*}} = add {{.*}}%[[PROMO3]], %[[MERGE3]]
; PROMO-NEXT: store{{.*}}@__profc_foo{{.*}}1)
; PROMO-NEXT: %[[PROMO2:[a-z0-9.]+]] = load{{.*}}@__profc_foo{{.*}}2)
; PROMO-NEXT: {{.*}} = add {{.*}}%[[PROMO2]], %[[MERGE2]]
; PROMO-NEXT: store{{.*}}@__profc_foo{{.*}}2)
; PROMO-NEXT: %[[PROMO1:[a-z0-9.]+]] = load{{.*}}@__profc_foo{{.*}}3)
; PROMO-NEXT: {{.*}} = add {{.*}}%[[PROMO1]], %[[MERGE1]]
; PROMO-NEXT: store{{.*}}@__profc_foo{{.*}}3)

bb15:                                             ; preds = %bb14
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 63969943867, i32 5, i32 4)
  tail call void @bar(i32 1)
  ret void
}

declare void @bar(i32) local_unnamed_addr

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(i8*, i64, i32, i32) #0

attributes #0 = { nounwind }
