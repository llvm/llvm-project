; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=3

;
; int fn() { return true; }
;
; int process() {
;   int a = 0;
;   int b = 0;
;   int val = 0;
;
;   // Use in control flow
;   if (a || b) val++;
;
;   // Operand with side effects
;   if (fn() || fn()) val++;
;   if (a || fn()) val++;
;   if (fn() || b) val++;
;   return val;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_10:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb52:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_11:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb53:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb54:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb55:]] %[[#bb56:]]
; CHECK:    %[[#bb55:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb57:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb57:]] %[[#bb58:]]
; CHECK:    %[[#bb56:]] = OpLabel
; CHECK:    %[[#bb58:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb57:]] %[[#bb59:]]
; CHECK:    %[[#bb59:]] = OpLabel
; CHECK:                  OpBranch %[[#bb57:]]
; CHECK:    %[[#bb57:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb60:]] %[[#bb54:]]
; CHECK:    %[[#bb60:]] = OpLabel
; CHECK:                  OpBranch %[[#bb54:]]
; CHECK:    %[[#bb54:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb61:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb62:]] %[[#bb63:]]
; CHECK:    %[[#bb62:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb64:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb64:]] %[[#bb65:]]
; CHECK:    %[[#bb63:]] = OpLabel
; CHECK:    %[[#bb65:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb64:]] %[[#bb66:]]
; CHECK:    %[[#bb66:]] = OpLabel
; CHECK:                  OpBranch %[[#bb64:]]
; CHECK:    %[[#bb64:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb67:]] %[[#bb61:]]
; CHECK:    %[[#bb67:]] = OpLabel
; CHECK:                  OpBranch %[[#bb61:]]
; CHECK:    %[[#bb61:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb68:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb69:]] %[[#bb70:]]
; CHECK:    %[[#bb69:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb71:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb71:]] %[[#bb72:]]
; CHECK:    %[[#bb70:]] = OpLabel
; CHECK:    %[[#bb72:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb71:]] %[[#bb73:]]
; CHECK:    %[[#bb73:]] = OpLabel
; CHECK:                  OpBranch %[[#bb71:]]
; CHECK:    %[[#bb71:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb74:]] %[[#bb68:]]
; CHECK:    %[[#bb74:]] = OpLabel
; CHECK:                  OpBranch %[[#bb68:]]
; CHECK:    %[[#bb68:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb75:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb76:]] %[[#bb77:]]
; CHECK:    %[[#bb76:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb78:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb78:]] %[[#bb79:]]
; CHECK:    %[[#bb77:]] = OpLabel
; CHECK:    %[[#bb79:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb78:]] %[[#bb80:]]
; CHECK:    %[[#bb80:]] = OpLabel
; CHECK:                  OpBranch %[[#bb78:]]
; CHECK:    %[[#bb78:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb81:]] %[[#bb75:]]
; CHECK:    %[[#bb81:]] = OpLabel
; CHECK:                  OpBranch %[[#bb75:]]
; CHECK:    %[[#bb75:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_48:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb82:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_50:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb83:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z2fnv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %val = alloca i32, align 4
  store i32 0, ptr %a, align 4
  store i32 0, ptr %b, align 4
  store i32 0, ptr %val, align 4
  %1 = load i32, ptr %a, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %2 = load i32, ptr %b, align 4
  %tobool1 = icmp ne i32 %2, 0
  br i1 %tobool1, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false, %entry
  %3 = load i32, ptr %val, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %val, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %lor.lhs.false
  %call2 = call spir_func noundef i32 @_Z2fnv() #3 [ "convergencectrl"(token %0) ]
  %tobool3 = icmp ne i32 %call2, 0
  br i1 %tobool3, label %if.then7, label %lor.lhs.false4

lor.lhs.false4:                                   ; preds = %if.end
  %call5 = call spir_func noundef i32 @_Z2fnv() #3 [ "convergencectrl"(token %0) ]
  %tobool6 = icmp ne i32 %call5, 0
  br i1 %tobool6, label %if.then7, label %if.end9

if.then7:                                         ; preds = %lor.lhs.false4, %if.end
  %4 = load i32, ptr %val, align 4
  %inc8 = add nsw i32 %4, 1
  store i32 %inc8, ptr %val, align 4
  br label %if.end9

if.end9:                                          ; preds = %if.then7, %lor.lhs.false4
  %5 = load i32, ptr %a, align 4
  %tobool10 = icmp ne i32 %5, 0
  br i1 %tobool10, label %if.then14, label %lor.lhs.false11

lor.lhs.false11:                                  ; preds = %if.end9
  %call12 = call spir_func noundef i32 @_Z2fnv() #3 [ "convergencectrl"(token %0) ]
  %tobool13 = icmp ne i32 %call12, 0
  br i1 %tobool13, label %if.then14, label %if.end16

if.then14:                                        ; preds = %lor.lhs.false11, %if.end9
  %6 = load i32, ptr %val, align 4
  %inc15 = add nsw i32 %6, 1
  store i32 %inc15, ptr %val, align 4
  br label %if.end16

if.end16:                                         ; preds = %if.then14, %lor.lhs.false11
  %call17 = call spir_func noundef i32 @_Z2fnv() #3 [ "convergencectrl"(token %0) ]
  %tobool18 = icmp ne i32 %call17, 0
  br i1 %tobool18, label %if.then21, label %lor.lhs.false19

lor.lhs.false19:                                  ; preds = %if.end16
  %7 = load i32, ptr %b, align 4
  %tobool20 = icmp ne i32 %7, 0
  br i1 %tobool20, label %if.then21, label %if.end23

if.then21:                                        ; preds = %lor.lhs.false19, %if.end16
  %8 = load i32, ptr %val, align 4
  %inc22 = add nsw i32 %8, 1
  store i32 %inc22, ptr %val, align 4
  br label %if.end23

if.end23:                                         ; preds = %if.then21, %lor.lhs.false19
  %9 = load i32, ptr %val, align 4
  ret i32 %9
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %call1 = call spir_func noundef i32 @_Z7processv() #3 [ "convergencectrl"(token %0) ]
  ret void
}

; Function Attrs: convergent norecurse
define void @main.1() #2 {
entry:
  call void @main()
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent }

!llvm.module.flags = !{!0, !1, !2}


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}


