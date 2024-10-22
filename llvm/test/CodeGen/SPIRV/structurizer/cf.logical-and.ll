; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=1

;
; int fn() { return true; }
;
; int process() {
;   // Use in control flow
;   int a = 0;
;   int b = 0;
;   int val = 0;
;   if (a && b) val++;
;
;   // Operand with side effects
;   if (fn() && fn()) val++;
;
;   if (a && fn())
;     val++;
;
;   if (fn() && b)
;     val++;
;   return val;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK:  %[[#func_9:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb43:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_10:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb44:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb45:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb46:]] %[[#bb45:]]
; CHECK:    %[[#bb46:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb47:]] %[[#bb45:]]
; CHECK:    %[[#bb47:]] = OpLabel
; CHECK:                  OpBranch %[[#bb45:]]
; CHECK:    %[[#bb45:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb48:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb49:]] %[[#bb48:]]
; CHECK:    %[[#bb49:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb50:]] %[[#bb48:]]
; CHECK:    %[[#bb50:]] = OpLabel
; CHECK:                  OpBranch %[[#bb48:]]
; CHECK:    %[[#bb48:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb51:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb52:]] %[[#bb51:]]
; CHECK:    %[[#bb52:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb53:]] %[[#bb51:]]
; CHECK:    %[[#bb53:]] = OpLabel
; CHECK:                  OpBranch %[[#bb51:]]
; CHECK:    %[[#bb51:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb54:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb55:]] %[[#bb54:]]
; CHECK:    %[[#bb55:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb56:]] %[[#bb54:]]
; CHECK:    %[[#bb56:]] = OpLabel
; CHECK:                  OpBranch %[[#bb54:]]
; CHECK:    %[[#bb54:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_39:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb57:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_41:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb58:]] = OpLabel
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
  br i1 %tobool, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %2 = load i32, ptr %b, align 4
  %tobool1 = icmp ne i32 %2, 0
  br i1 %tobool1, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %3 = load i32, ptr %val, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %val, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  %call2 = call spir_func noundef i32 @_Z2fnv() #3 [ "convergencectrl"(token %0) ]
  %tobool3 = icmp ne i32 %call2, 0
  br i1 %tobool3, label %land.lhs.true4, label %if.end9

land.lhs.true4:                                   ; preds = %if.end
  %call5 = call spir_func noundef i32 @_Z2fnv() #3 [ "convergencectrl"(token %0) ]
  %tobool6 = icmp ne i32 %call5, 0
  br i1 %tobool6, label %if.then7, label %if.end9

if.then7:                                         ; preds = %land.lhs.true4
  %4 = load i32, ptr %val, align 4
  %inc8 = add nsw i32 %4, 1
  store i32 %inc8, ptr %val, align 4
  br label %if.end9

if.end9:                                          ; preds = %if.then7, %land.lhs.true4, %if.end
  %5 = load i32, ptr %a, align 4
  %tobool10 = icmp ne i32 %5, 0
  br i1 %tobool10, label %land.lhs.true11, label %if.end16

land.lhs.true11:                                  ; preds = %if.end9
  %call12 = call spir_func noundef i32 @_Z2fnv() #3 [ "convergencectrl"(token %0) ]
  %tobool13 = icmp ne i32 %call12, 0
  br i1 %tobool13, label %if.then14, label %if.end16

if.then14:                                        ; preds = %land.lhs.true11
  %6 = load i32, ptr %val, align 4
  %inc15 = add nsw i32 %6, 1
  store i32 %inc15, ptr %val, align 4
  br label %if.end16

if.end16:                                         ; preds = %if.then14, %land.lhs.true11, %if.end9
  %call17 = call spir_func noundef i32 @_Z2fnv() #3 [ "convergencectrl"(token %0) ]
  %tobool18 = icmp ne i32 %call17, 0
  br i1 %tobool18, label %land.lhs.true19, label %if.end23

land.lhs.true19:                                  ; preds = %if.end16
  %7 = load i32, ptr %b, align 4
  %tobool20 = icmp ne i32 %7, 0
  br i1 %tobool20, label %if.then21, label %if.end23

if.then21:                                        ; preds = %land.lhs.true19
  %8 = load i32, ptr %val, align 4
  %inc22 = add nsw i32 %8, 1
  store i32 %inc22, ptr %val, align 4
  br label %if.end23

if.end23:                                         ; preds = %if.then21, %land.lhs.true19, %if.end16
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


