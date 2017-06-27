; RUN: opt < %s -pgo-instr-gen -instrprof -do-counter-promotion=true -speculative-counter-promotion -S | FileCheck --check-prefix=PROMO %s
; RUN: opt < %s --passes=pgo-instr-gen,instrprof -do-counter-promotion=true -speculative-counter-promotion -S | FileCheck --check-prefix=PROMO %s

@g = common local_unnamed_addr global i32 0, align 4

define void @foo(i32 %arg) local_unnamed_addr {
; PROMO-LABEL: @foo
bb:
  %tmp = add nsw i32 %arg, -1
  br label %bb1
bb1:                                              ; preds = %bb11, %bb
  %tmp2 = phi i32 [ 0, %bb ], [ %tmp12, %bb11 ]
  %tmp3 = icmp sgt i32 %tmp2, %arg
  br i1 %tmp3, label %bb7, label %bb4

bb4:                                              ; preds = %bb1
  tail call void @bar(i32 1)
  %tmp5 = load i32, i32* @g, align 4
  %tmp6 = icmp sgt i32 %tmp5, 100
  br i1 %tmp6, label %bb15_0, label %bb11

bb7:                                              ; preds = %bb1
  %tmp8 = icmp slt i32 %tmp2, %tmp
  br i1 %tmp8, label %bb9, label %bb10

bb9:                                              ; preds = %bb7
  tail call void @bar(i32 2)
  br label %bb11

bb10:                                             ; preds = %bb7
  tail call void @bar(i32 3)
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9, %bb4
  %tmp12 = add nuw nsw i32 %tmp2, 1
  %tmp13 = icmp slt i32 %tmp2, 99
  br i1 %tmp13, label %bb1, label %bb14

bb14:                                             ; preds = %bb11
; PROMO-LABEL: bb14:
  tail call void @bar(i32 0)
  br label %bb15
; PROMO:  %pgocount.promoted{{.*}} = load {{.*}} @__profc_foo{{.*}} 0)
; PROMO-NEXT: add 
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}0)
; PROMO-NEXT:  %pgocount.promoted{{.*}} = load {{.*}} @__profc_foo{{.*}} 1)
; PROMO-NEXT: add 
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}1)
; PROMO-NEXT:  %pgocount.promoted{{.*}} = load {{.*}} @__profc_foo{{.*}} 2)
; PROMO-NEXT: add 
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}2)
; PROMO-NEXT:  %pgocount{{.*}} = load {{.*}} @__profc_foo{{.*}} 3)
; PROMO-NEXT: add 
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}3)


bb15_0:                                             ; preds = %bb11
; PROMO-LABEL: bb15_0:
  br label %bb15
; PROMO:  %pgocount.promoted{{.*}} = load {{.*}} @__profc_foo{{.*}} 0)
; PROMO-NEXT: add 
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}0)
; PROMO-NEXT:  %pgocount.promoted{{.*}} = load {{.*}} @__profc_foo{{.*}} 1)
; PROMO-NEXT: add 
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}1)
; PROMO-NEXT:  %pgocount.promoted{{.*}} = load {{.*}} @__profc_foo{{.*}} 2)
; PROMO-NEXT: add 
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}2)
; PROMO-NEXT:  %pgocount{{.*}} = load {{.*}} @__profc_foo{{.*}} 4)
; PROMO-NEXT: add 
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}4)
; PROMO-NOT: @__profc_foo


bb15:                                             ; preds = %bb14, %bb4
  tail call void @bar(i32 1)
  ret void
}

declare void @bar(i32) local_unnamed_addr
