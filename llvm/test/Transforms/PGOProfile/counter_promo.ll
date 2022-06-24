; RUN: opt < %s --passes=pgo-instr-gen,instrprof -do-counter-promotion=true -skip-ret-exit-block=0 -S | FileCheck --check-prefix=PROMO --check-prefix=NONATOMIC_PROMO %s
; RUN: opt < %s --passes=pgo-instr-gen,instrprof -do-counter-promotion=true -atomic-counter-update-promoted -skip-ret-exit-block=0 -S | FileCheck --check-prefix=PROMO --check-prefix=ATOMIC_PROMO %s

define void @foo(i32 %n, i32 %N) {
; PROMO-LABEL: @foo
; PROMO: {{.*}} = load {{.*}} @__profc_foo{{.*}} 3)
; PROMO-NEXT: add
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}3)
bb:
  %tmp = add nsw i32 %n, 1
  %tmp1 = add nsw i32 %n, -1
  br label %bb2

bb2:                                              ; preds = %bb9, %bb
; PROMO: phi {{.*}}
; PROMO-NEXT: phi {{.*}}
; PROMO-NEXT: phi {{.*}}
; PROMO-NEXT: phi {{.*}}
  %i.0 = phi i32 [ 0, %bb ], [ %tmp10, %bb9 ]
  %tmp3 = icmp slt i32 %i.0, %tmp
  br i1 %tmp3, label %bb4, label %bb5

bb4:                                              ; preds = %bb2
  tail call void @bar(i32 1)
  br label %bb9

bb5:                                              ; preds = %bb2
  %tmp6 = icmp slt i32 %i.0, %tmp1
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb5
  tail call void @bar(i32 2)
  br label %bb9

bb8:                                              ; preds = %bb5
  tail call void @bar(i32 3)
  br label %bb9

bb9:                                              ; preds = %bb8, %bb7, %bb4
; PROMO: %[[LIVEOUT3:[a-z0-9]+]] = phi {{.*}}
; PROMO-NEXT: %[[LIVEOUT2:[a-z0-9]+]] = phi {{.*}}
; PROMO-NEXT: %[[LIVEOUT1:[a-z0-9]+]] = phi {{.*}}
  %tmp10 = add nsw i32 %i.0, 1
  %tmp11 = icmp slt i32 %tmp10, %N
  br i1 %tmp11, label %bb2, label %bb12

bb12:                                             ; preds = %bb9
  ret void
; NONATOMIC_PROMO: %[[PROMO1:[a-z0-9.]+]] = load {{.*}} @__profc_foo{{.*}} 0)
; NONATOMIC_PROMO-NEXT: add {{.*}} %[[PROMO1]], %[[LIVEOUT1]] 
; NONATOMIC_PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}0)
; NONATOMIC_PROMO-NEXT: %[[PROMO2:[a-z0-9.]+]] = load {{.*}} @__profc_foo{{.*}} 1)
; NONATOMIC_PROMO-NEXT: add {{.*}} %[[PROMO2]], %[[LIVEOUT2]]
; NONATOMIC_PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}1)
; NONATOMIC_PROMO-NEXT: %[[PROMO3:[a-z0-9.]+]] = load {{.*}} @__profc_foo{{.*}} 2)
; NONATOMIC_PROMO-NEXT: add {{.*}} %[[PROMO3]], %[[LIVEOUT3]]
; NONATOMIC_PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}2)
; ATOMIC_PROMO: atomicrmw add {{.*}} @__profc_foo{{.*}}0), i64 %[[LIVEOUT1]] seq_cst
; ATOMIC_PROMO-NEXT: atomicrmw add {{.*}} @__profc_foo{{.*}}1), i64 %[[LIVEOUT2]] seq_cst
; ATOMIC_PROMO-NEXT: atomicrmw add {{.*}} @__profc_foo{{.*}}2), i64 %[[LIVEOUT3]] seq_cst
; PROMO-NOT: @__profc_foo{{.*}})


}

declare void @bar(i32)
