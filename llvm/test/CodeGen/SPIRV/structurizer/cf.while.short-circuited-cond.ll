; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; [numthreads(1, 1, 1)]
; void main() {
;   int a, b;
;   while (a && b) {
;   }
;
;   while (a || b) {
;   }
;   while (a && ((a || b) && b)) {
;   }
;
;   while (a ? a : b) {
;   }
;
;   int x, y;
;   while (x + (x && y)) {
;   }
; }

; CHECK: %[[#func_10:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb54:]] = OpLabel
; CHECK:                  OpBranch %[[#bb55:]]
; CHECK:    %[[#bb55:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb56:]] %[[#bb57:]] None
; CHECK:                  OpBranch %[[#bb58:]]
; CHECK:    %[[#bb58:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb59:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb60:]] %[[#bb59:]]
; CHECK:    %[[#bb60:]] = OpLabel
; CHECK:                  OpBranch %[[#bb59:]]
; CHECK:    %[[#bb59:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb57:]] %[[#bb56:]]
; CHECK:    %[[#bb57:]] = OpLabel
; CHECK:                  OpBranch %[[#bb55:]]
; CHECK:    %[[#bb56:]] = OpLabel
; CHECK:                  OpBranch %[[#bb61:]]
; CHECK:    %[[#bb61:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb62:]] %[[#bb63:]] None
; CHECK:                  OpBranch %[[#bb64:]]
; CHECK:    %[[#bb64:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb65:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb65:]] %[[#bb66:]]
; CHECK:    %[[#bb66:]] = OpLabel
; CHECK:                  OpBranch %[[#bb65:]]
; CHECK:    %[[#bb65:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb63:]] %[[#bb62:]]
; CHECK:    %[[#bb63:]] = OpLabel
; CHECK:                  OpBranch %[[#bb61:]]
; CHECK:    %[[#bb62:]] = OpLabel
; CHECK:                  OpBranch %[[#bb67:]]
; CHECK:    %[[#bb67:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb68:]] %[[#bb69:]] None
; CHECK:                  OpBranch %[[#bb70:]]
; CHECK:    %[[#bb70:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb71:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb72:]] %[[#bb71:]]
; CHECK:    %[[#bb72:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb73:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb74:]] %[[#bb75:]]
; CHECK:    %[[#bb74:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb76:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb76:]] %[[#bb77:]]
; CHECK:    %[[#bb75:]] = OpLabel
; CHECK:    %[[#bb77:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb76:]] %[[#bb78:]]
; CHECK:    %[[#bb78:]] = OpLabel
; CHECK:                  OpBranch %[[#bb76:]]
; CHECK:    %[[#bb76:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb79:]] %[[#bb73:]]
; CHECK:    %[[#bb79:]] = OpLabel
; CHECK:                  OpBranch %[[#bb73:]]
; CHECK:    %[[#bb73:]] = OpLabel
; CHECK:                  OpBranch %[[#bb71:]]
; CHECK:    %[[#bb71:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb69:]] %[[#bb68:]]
; CHECK:    %[[#bb69:]] = OpLabel
; CHECK:                  OpBranch %[[#bb67:]]
; CHECK:    %[[#bb68:]] = OpLabel
; CHECK:                  OpBranch %[[#bb80:]]
; CHECK:    %[[#bb80:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb81:]] %[[#bb82:]] None
; CHECK:                  OpBranch %[[#bb83:]]
; CHECK:    %[[#bb83:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb84:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb85:]] %[[#bb86:]]
; CHECK:    %[[#bb85:]] = OpLabel
; CHECK:                  OpBranch %[[#bb84:]]
; CHECK:    %[[#bb86:]] = OpLabel
; CHECK:                  OpBranch %[[#bb84:]]
; CHECK:    %[[#bb84:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb82:]] %[[#bb81:]]
; CHECK:    %[[#bb82:]] = OpLabel
; CHECK:                  OpBranch %[[#bb80:]]
; CHECK:    %[[#bb81:]] = OpLabel
; CHECK:                  OpBranch %[[#bb87:]]
; CHECK:    %[[#bb87:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb88:]] %[[#bb89:]] None
; CHECK:                  OpBranch %[[#bb90:]]
; CHECK:    %[[#bb90:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb91:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb92:]] %[[#bb91:]]
; CHECK:    %[[#bb92:]] = OpLabel
; CHECK:                  OpBranch %[[#bb91:]]
; CHECK:    %[[#bb91:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb89:]] %[[#bb88:]]
; CHECK:    %[[#bb89:]] = OpLabel
; CHECK:                  OpBranch %[[#bb87:]]
; CHECK:    %[[#bb88:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_52:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb93:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %a, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %while.cond
  %3 = load i32, ptr %b, align 4
  %tobool1 = icmp ne i32 %3, 0
  br label %land.end

land.end:                                         ; preds = %land.rhs, %while.cond
  %4 = phi i1 [ false, %while.cond ], [ %tobool1, %land.rhs ]
  br i1 %4, label %while.body, label %while.end

while.body:                                       ; preds = %land.end
  br label %while.cond

while.end:                                        ; preds = %land.end
  br label %while.cond2

while.cond2:                                      ; preds = %while.body5, %while.end
  %5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %6 = load i32, ptr %a, align 4
  %tobool3 = icmp ne i32 %6, 0
  br i1 %tobool3, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %while.cond2
  %7 = load i32, ptr %b, align 4
  %tobool4 = icmp ne i32 %7, 0
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %while.cond2
  %8 = phi i1 [ true, %while.cond2 ], [ %tobool4, %lor.rhs ]
  br i1 %8, label %while.body5, label %while.end6

while.body5:                                      ; preds = %lor.end
  br label %while.cond2

while.end6:                                       ; preds = %lor.end
  br label %while.cond7

while.cond7:                                      ; preds = %while.body16, %while.end6
  %9 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %10 = load i32, ptr %a, align 4
  %tobool8 = icmp ne i32 %10, 0
  br i1 %tobool8, label %land.rhs9, label %land.end15

land.rhs9:                                        ; preds = %while.cond7
  %11 = load i32, ptr %a, align 4
  %tobool10 = icmp ne i32 %11, 0
  br i1 %tobool10, label %land.rhs12, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %land.rhs9
  %12 = load i32, ptr %b, align 4
  %tobool11 = icmp ne i32 %12, 0
  br i1 %tobool11, label %land.rhs12, label %land.end14

land.rhs12:                                       ; preds = %lor.lhs.false, %land.rhs9
  %13 = load i32, ptr %b, align 4
  %tobool13 = icmp ne i32 %13, 0
  br label %land.end14

land.end14:                                       ; preds = %land.rhs12, %lor.lhs.false
  %14 = phi i1 [ false, %lor.lhs.false ], [ %tobool13, %land.rhs12 ]
  br label %land.end15

land.end15:                                       ; preds = %land.end14, %while.cond7
  %15 = phi i1 [ false, %while.cond7 ], [ %14, %land.end14 ]
  br i1 %15, label %while.body16, label %while.end17

while.body16:                                     ; preds = %land.end15
  br label %while.cond7

while.end17:                                      ; preds = %land.end15
  br label %while.cond18

while.cond18:                                     ; preds = %while.body21, %while.end17
  %16 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %17 = load i32, ptr %a, align 4
  %tobool19 = icmp ne i32 %17, 0
  br i1 %tobool19, label %cond.true, label %cond.false

cond.true:                                        ; preds = %while.cond18
  %18 = load i32, ptr %a, align 4
  br label %cond.end

cond.false:                                       ; preds = %while.cond18
  %19 = load i32, ptr %b, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %18, %cond.true ], [ %19, %cond.false ]
  %tobool20 = icmp ne i32 %cond, 0
  br i1 %tobool20, label %while.body21, label %while.end22

while.body21:                                     ; preds = %cond.end
  br label %while.cond18

while.end22:                                      ; preds = %cond.end
  br label %while.cond23

while.cond23:                                     ; preds = %while.body29, %while.end22
  %20 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %21 = load i32, ptr %x, align 4
  %22 = load i32, ptr %x, align 4
  %tobool24 = icmp ne i32 %22, 0
  br i1 %tobool24, label %land.rhs25, label %land.end27

land.rhs25:                                       ; preds = %while.cond23
  %23 = load i32, ptr %y, align 4
  %tobool26 = icmp ne i32 %23, 0
  br label %land.end27

land.end27:                                       ; preds = %land.rhs25, %while.cond23
  %24 = phi i1 [ false, %while.cond23 ], [ %tobool26, %land.rhs25 ]
  %conv = zext i1 %24 to i32
  %add = add nsw i32 %21, %conv
  %tobool28 = icmp ne i32 %add, 0
  br i1 %tobool28, label %while.body29, label %while.end30

while.body29:                                     ; preds = %land.end27
  br label %while.cond23

while.end30:                                      ; preds = %land.end27
  ret void
}

; Function Attrs: convergent norecurse
define void @main.1() #1 {
entry:
  call void @main()
  ret void
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #2

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.loop() #2

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}


