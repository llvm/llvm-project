; RUN: llc -mtriple=hexagon -O3 -fp-contract=fast -pipeliner-max-stages=3 < %s
; REQUIRES: asserts

; Check Phis are generated correctly in epilogs after setting -swp-max-stages=3

@g0 = private unnamed_addr constant [6 x i8] c"s4116\00", align 1

; Function Attrs: noinline nounwind
define void @f0(i32 %a0, i32 %a1, ptr nocapture readonly %a2, ptr nocapture readonly %a3, ptr nocapture readonly %a4, i32 %a5) #0 {
b0:
  %v0 = tail call i32 @f1() #2
  %v1 = sitofp i32 %v0 to double
  %v2 = add nsw i32 %a1, -1
  %v3 = icmp sgt i32 %a1, 1
  br i1 %v3, label %b1, label %b3

b1:                                               ; preds = %b1, %b0
  %v4 = phi float [ %v13, %b1 ], [ 0.000000e+00, %b0 ]
  %v5 = phi ptr [ %v16, %b1 ], [ %a2, %b0 ]
  %v6 = phi ptr [ %v17, %b1 ], [ %a4, %b0 ]
  %v7 = phi i32 [ %v14, %b1 ], [ 0, %b0 ]
  %v8 = load float, ptr %v5, align 4
  %v9 = load i32, ptr %v6, align 4
  %v10 = getelementptr inbounds [1000 x float], ptr %a3, i32 %v9, i32 %a5
  %v11 = load float, ptr %v10, align 4
  %v12 = fmul float %v8, %v11
  %v13 = fadd float %v4, %v12
  %v14 = add nuw nsw i32 %v7, 1
  %v15 = icmp slt i32 %v14, %v2
  %v16 = getelementptr float, ptr %v5, i32 1
  %v17 = getelementptr i32, ptr %v6, i32 1
  br i1 %v15, label %b1, label %b2

b2:                                               ; preds = %b1
  %v18 = fpext float %v13 to double
  br label %b3

b3:                                               ; preds = %b2, %b0
  %v19 = phi double [ %v18, %b2 ], [ 0.000000e+00, %b0 ]
  tail call void @f2(double %v19, i32 %a1, i32 %a1, double %v1, ptr @g0) #2
  ret void
}

declare i32 @f1(...) #1

declare void @f2(double, i32, i32, double, ptr) #1

attributes #0 = { noinline nounwind "target-cpu"="hexagonv60" }
attributes #1 = { "target-cpu"="hexagonv60" }
attributes #2 = { nounwind }
