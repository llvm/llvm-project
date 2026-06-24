; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; int process() {
;   int c = 0;
;   int val = 0;
;
;   // Both then and else
;   if (c) {
;     val = val + 1;
;   } else {
;     val = val + 2;
;   }
;
;   // No else
;   if (c)
;     val = 1;
;
;   // Empty then
;   if (c) {
;   } else {
;     val = 2;
;   }
;
;   // Null body
;   if (c)
;     ;
;
;   if (int d = val) {
;     c = true;
;   }
;
;   return val;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_10:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb34:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb35:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb36:]] %[[#bb37:]]
; CHECK:    %[[#bb36:]] = OpLabel
; CHECK:                  OpBranch %[[#bb35:]]
; CHECK:    %[[#bb37:]] = OpLabel
; CHECK:                  OpBranch %[[#bb35:]]
; CHECK:    %[[#bb35:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb38:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb39:]] %[[#bb38:]]
; CHECK:    %[[#bb39:]] = OpLabel
; CHECK:                  OpBranch %[[#bb38:]]
; CHECK:    %[[#bb38:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb40:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb40:]] %[[#bb41:]]
; CHECK:    %[[#bb41:]] = OpLabel
; CHECK:                  OpBranch %[[#bb40:]]
; CHECK:    %[[#bb40:]] = OpLabel
; CHECK:                  OpBranch %[[#bb42:]]
; CHECK:    %[[#bb42:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb43:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb44:]] %[[#bb43:]]
; CHECK:    %[[#bb44:]] = OpLabel
; CHECK:                  OpBranch %[[#bb43:]]
; CHECK:    %[[#bb43:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_30:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb45:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_32:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb46:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %c = alloca i32, align 4
  %val = alloca i32, align 4
  %d = alloca i32, align 4
  store i32 0, ptr %c, align 4
  store i32 0, ptr %val, align 4
  %1 = load i32, ptr %c, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load i32, ptr %val, align 4
  %add = add nsw i32 %2, 1
  store i32 %add, ptr %val, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %3 = load i32, ptr %val, align 4
  %add1 = add nsw i32 %3, 2
  store i32 %add1, ptr %val, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %4 = load i32, ptr %c, align 4
  %tobool2 = icmp ne i32 %4, 0
  br i1 %tobool2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.end
  store i32 1, ptr %val, align 4
  br label %if.end4

if.end4:                                          ; preds = %if.then3, %if.end
  %5 = load i32, ptr %c, align 4
  %tobool5 = icmp ne i32 %5, 0
  br i1 %tobool5, label %if.then6, label %if.else7

if.then6:                                         ; preds = %if.end4
  br label %if.end8

if.else7:                                         ; preds = %if.end4
  store i32 2, ptr %val, align 4
  br label %if.end8

if.end8:                                          ; preds = %if.else7, %if.then6
  %6 = load i32, ptr %c, align 4
  %tobool9 = icmp ne i32 %6, 0
  br i1 %tobool9, label %if.then10, label %if.end11

if.then10:                                        ; preds = %if.end8
  br label %if.end11

if.end11:                                         ; preds = %if.then10, %if.end8
  %7 = load i32, ptr %val, align 4
  store i32 %7, ptr %d, align 4
  %8 = load i32, ptr %d, align 4
  %tobool12 = icmp ne i32 %8, 0
  br i1 %tobool12, label %if.then13, label %if.end14

if.then13:                                        ; preds = %if.end11
  store i32 1, ptr %c, align 4
  br label %if.end14

if.end14:                                         ; preds = %if.then13, %if.end11
  %9 = load i32, ptr %val, align 4
  ret i32 %9
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

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


