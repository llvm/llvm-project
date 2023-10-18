; RUN: opt < %s --passes=pgo-instr-gen,instrprof -do-counter-promotion=true -sampled-instr=true -skip-ret-exit-block=0 -S | FileCheck --check-prefixes=SAMPLING,PROMO %s

; SAMPLING: $__llvm_profile_sampling = comdat any
; SAMPLING: @__llvm_profile_sampling = thread_local global i16 0, comdat

define void @foo(i32 %n, i32 %N) {
; SAMPLING-LABEL: @foo
; SAMPLING:  %[[VV0:[0-9]+]] = load i16, ptr @__llvm_profile_sampling, align 2
; SAMPLING:  %[[VV1:[0-9]+]] = icmp ule i16 %[[VV0]], 200
; SAMPLING:  br i1 %[[VV1]], label {{.*}}, label {{.*}}, !prof !0
; SAMPLING: {{.*}} = load {{.*}} @__profc_foo{{.*}} 3)
; SAMPLING-NEXT: add
; SAMPLING-NEXT: store {{.*}}@__profc_foo{{.*}}3)
bb:
  %tmp = add nsw i32 %n, 1
  %tmp1 = add nsw i32 %n, -1
  br label %bb2

bb2:
; PROMO: phi {{.*}}
; PROMO-NEXT: phi {{.*}}
; PROMO-NEXT: phi {{.*}}
; PROMO-NEXT: phi {{.*}}
  %i.0 = phi i32 [ 0, %bb ], [ %tmp10, %bb9 ]
  %tmp3 = icmp slt i32 %i.0, %tmp
  br i1 %tmp3, label %bb4, label %bb5

bb4:
  tail call void @bar(i32 1)
  br label %bb9

bb5:
  %tmp6 = icmp slt i32 %i.0, %tmp1
  br i1 %tmp6, label %bb7, label %bb8

bb7:
  tail call void @bar(i32 2)
  br label %bb9

bb8:
  tail call void @bar(i32 3)
  br label %bb9

bb9:
; SAMPLING:       phi {{.*}}
; SAMPLING-NEXT:  %[[V1:[0-9]+]] = add i16 {{.*}}, 1
; SAMPLING-NEXT:  store i16 %[[V1]], ptr @__llvm_profile_sampling, align 2
; SAMPLING:       phi {{.*}}
; SAMPLING-NEXT:  %[[V2:[0-9]+]] = add i16 {{.*}}, 1
; SAMPLING-NEXT:  store i16 %[[V2]], ptr @__llvm_profile_sampling, align 2
; SAMPLING:       phi {{.*}}
; SAMPLING-NEXT:  %[[V3:[0-9]+]] = add i16 {{.*}}, 1
; SAMPLING-NEXT:  store i16 %[[V3]], ptr @__llvm_profile_sampling, align 2
; PROMO: %[[LIVEOUT3:[a-z0-9]+]] = phi {{.*}}
; PROMO-NEXT: %[[LIVEOUT2:[a-z0-9]+]] = phi {{.*}}
; PROMO-NEXT: %[[LIVEOUT1:[a-z0-9]+]] = phi {{.*}}
  %tmp10 = add nsw i32 %i.0, 1
  %tmp11 = icmp slt i32 %tmp10, %N
  br i1 %tmp11, label %bb2, label %bb12

bb12:
  ret void
; PROMO: %[[CHECK1:[a-z0-9.]+]] = load {{.*}} @__profc_foo{{.*}}
; PROMO-NEXT: add {{.*}} %[[CHECK1]], %[[LIVEOUT1]]
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}
; PROMO-NEXT: %[[CHECK2:[a-z0-9.]+]] = load {{.*}} @__profc_foo{{.*}} 1)
; PROMO-NEXT: add {{.*}} %[[CHECK2]], %[[LIVEOUT2]]
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}1)
; PROMO-NEXT: %[[CHECK3:[a-z0-9.]+]] = load {{.*}} @__profc_foo{{.*}} 2)
; PROMO-NEXT: add {{.*}} %[[CHECK3]], %[[LIVEOUT3]]
; PROMO-NEXT: store {{.*}}@__profc_foo{{.*}}2)
; PROMO-NOT: @__profc_foo{{.*}})

}

declare void @bar(i32)

; SAMPLING: !0 = !{!"branch_weights", i32 200, i32 65336}
