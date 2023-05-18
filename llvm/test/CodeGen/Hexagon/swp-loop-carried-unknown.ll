; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that the pipeliner schedules a store before the load in which there is a
; loop carried dependence. Previously, the loop carried dependence wasn't added
; and the load from iteration n was scheduled prior to the store from iteration
; n-1.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: memh({{.*}}) =
; CHECK: = memuh({{.*}})
; CHECK: endloop0

%s.0 = type { i16, i16 }

; Function Attrs: nounwind
define void @f0() local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v22, %b1 ]
  %v1 = load ptr, ptr undef, align 4
  %v3 = load i16, ptr %v1, align 2
  %v4 = add i16 0, %v3
  %v5 = add i16 %v4, 0
  %v6 = add i16 %v5, 0
  %v7 = add i16 %v6, 0
  %v8 = add i16 %v7, 0
  %v9 = add i16 %v8, 0
  %v10 = add i16 %v9, 0
  %v11 = add i16 %v10, 0
  %v12 = add i16 %v11, 0
  %v13 = add i16 %v12, 0
  %v14 = add i16 %v13, 0
  %v15 = add i16 %v14, 0
  %v16 = add i16 %v15, 0
  %v17 = add i16 %v16, 0
  %v18 = add i16 %v17, 0
  %v19 = add i16 %v18, 0
  %v20 = load ptr, ptr undef, align 4
  store i16 %v19, ptr undef, align 2
  %v21 = getelementptr inbounds %s.0, ptr %v20, i32 0, i32 1
  store i16 0, ptr %v21, align 2
  %v22 = add nuw nsw i32 %v0, 1
  %v23 = icmp eq i32 %v22, 6
  br i1 %v23, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
