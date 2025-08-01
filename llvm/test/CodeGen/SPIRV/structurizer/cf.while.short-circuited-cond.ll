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

; CHECK: %[[#func_11:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb87:]] = OpLabel
; CHECK:                  OpBranch %[[#bb88:]]
; CHECK:     %[[#bb88]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb89:]] %[[#bb90:]] None
; CHECK:                  OpBranch %[[#bb91:]]
; CHECK:     %[[#bb91]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb92:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb93:]] %[[#bb94:]]
; CHECK:     %[[#bb94]] = OpLabel
; CHECK:                  OpBranch %[[#bb92]]
; CHECK:     %[[#bb93]] = OpLabel
; CHECK:                  OpBranch %[[#bb92]]
; CHECK:     %[[#bb92]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb90]] %[[#bb89]]
; CHECK:     %[[#bb89]] = OpLabel
; CHECK:                  OpBranch %[[#bb95:]]
; CHECK:     %[[#bb95]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb96:]] %[[#bb97:]] None
; CHECK:                  OpBranch %[[#bb98:]]
; CHECK:     %[[#bb98]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb99:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb100:]] %[[#bb101:]]
; CHECK:    %[[#bb101]] = OpLabel
; CHECK:                  OpBranch %[[#bb99]]
; CHECK:    %[[#bb100]] = OpLabel
; CHECK:                  OpBranch %[[#bb99]]
; CHECK:     %[[#bb99]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb97]] %[[#bb96]]
; CHECK:     %[[#bb96]] = OpLabel
; CHECK:                  OpBranch %[[#bb102:]]
; CHECK:    %[[#bb102]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb103:]] %[[#bb104:]] None
; CHECK:                  OpBranch %[[#bb105:]]
; CHECK:    %[[#bb105]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb106:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb107:]] %[[#bb108:]]
; CHECK:    %[[#bb108]] = OpLabel
; CHECK:                  OpBranch %[[#bb106]]
; CHECK:    %[[#bb107]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb109:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb110:]] %[[#bb109]]
; CHECK:    %[[#bb110]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb111:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb112:]] %[[#bb113:]]
; CHECK:    %[[#bb113]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb114:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb114]] %[[#bb115:]]
; CHECK:    %[[#bb115]] = OpLabel
; CHECK:                  OpBranch %[[#bb114]]
; CHECK:    %[[#bb114]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb116:]] %[[#bb111]]
; CHECK:    %[[#bb116]] = OpLabel
; CHECK:                  OpBranch %[[#bb111]]
; CHECK:    %[[#bb112]] = OpLabel
; CHECK:                  OpBranch %[[#bb111]]
; CHECK:    %[[#bb111]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb109]] %[[#bb117:]]
; CHECK:    %[[#bb117]] = OpLabel
; CHECK:                  OpBranch %[[#bb109]]
; CHECK:    %[[#bb109]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb106]] %[[#bb118:]]
; CHECK:    %[[#bb118]] = OpLabel
; CHECK:                  OpBranch %[[#bb106]]
; CHECK:    %[[#bb106]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb119:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb119]] %[[#bb120:]]
; CHECK:    %[[#bb120]] = OpLabel
; CHECK:    %[[#bb119]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb104]] %[[#bb103]]
; CHECK:    %[[#bb103]] = OpLabel
; CHECK:                  OpBranch %[[#bb121:]]
; CHECK:    %[[#bb121]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb122:]] %[[#bb123:]] None
; CHECK:                  OpBranch %[[#bb124:]]
; CHECK:    %[[#bb124]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb125:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb126:]] %[[#bb127:]]
; CHECK:    %[[#bb127]] = OpLabel
; CHECK:                  OpBranch %[[#bb125]]
; CHECK:    %[[#bb126]] = OpLabel
; CHECK:                  OpBranch %[[#bb125]]
; CHECK:    %[[#bb125]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb123]] %[[#bb122]]
; CHECK:    %[[#bb122]] = OpLabel
; CHECK:                  OpBranch %[[#bb128:]]
; CHECK:    %[[#bb128]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb129:]] %[[#bb130:]] None
; CHECK:                  OpBranch %[[#bb131:]]
; CHECK:    %[[#bb131]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb132:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb133:]] %[[#bb134:]]
; CHECK:    %[[#bb134]] = OpLabel
; CHECK:                  OpBranch %[[#bb132]]
; CHECK:    %[[#bb133]] = OpLabel
; CHECK:                  OpBranch %[[#bb132]]
; CHECK:    %[[#bb132]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb130]] %[[#bb129]]
; CHECK:    %[[#bb129]] = OpLabel
; CHECK:                  OpReturn
; CHECK:    %[[#bb130]] = OpLabel
; CHECK:                  OpBranch %[[#bb128]]
; CHECK:    %[[#bb123]] = OpLabel
; CHECK:                  OpBranch %[[#bb121]]
; CHECK:    %[[#bb104]] = OpLabel
; CHECK:                  OpBranch %[[#bb102]]
; CHECK:     %[[#bb97]] = OpLabel
; CHECK:                  OpBranch %[[#bb95]]
; CHECK:     %[[#bb90]] = OpLabel
; CHECK:                  OpBranch %[[#bb88]]
; CHECK:                  OpFunctionEnd

; CHECK: %[[#func_85:]] = OpFunction %[[#void]] None %[[#]]
; CHECK:   %[[#bb135:]] = OpLabel
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


