; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=3


;
; int process() {
;   int c1 = 0;
;   int c2 = 1;
;   int c3 = 0;
;   int c4 = 1;
;   int val = 0;
;
;   if (c1) {
;     if (c2)
;       val = 1;
;   } else {
;     if (c3) {
;       val = 2;
;     } else {
;       if (c4) {
;         val = 3;
;       }
;     }
;   }
;   return val;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_11:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb30:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb31:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb32:]] %[[#bb33:]]
; CHECK:    %[[#bb32:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb34:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb35:]] %[[#bb34:]]
; CHECK:    %[[#bb33:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb36:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb37:]] %[[#bb38:]]
; CHECK:    %[[#bb35:]] = OpLabel
; CHECK:                  OpBranch %[[#bb34:]]
; CHECK:    %[[#bb37:]] = OpLabel
; CHECK:                  OpBranch %[[#bb36:]]
; CHECK:    %[[#bb38:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb39:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb40:]] %[[#bb39:]]
; CHECK:    %[[#bb34:]] = OpLabel
; CHECK:                  OpBranch %[[#bb31:]]
; CHECK:    %[[#bb40:]] = OpLabel
; CHECK:                  OpBranch %[[#bb39:]]
; CHECK:    %[[#bb39:]] = OpLabel
; CHECK:                  OpBranch %[[#bb36:]]
; CHECK:    %[[#bb36:]] = OpLabel
; CHECK:                  OpBranch %[[#bb31:]]
; CHECK:    %[[#bb31:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_26:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb41:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_28:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb42:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %c1 = alloca i32, align 4
  %c2 = alloca i32, align 4
  %c3 = alloca i32, align 4
  %c4 = alloca i32, align 4
  %val = alloca i32, align 4
  store i32 0, ptr %c1, align 4
  store i32 1, ptr %c2, align 4
  store i32 0, ptr %c3, align 4
  store i32 1, ptr %c4, align 4
  store i32 0, ptr %val, align 4
  %1 = load i32, ptr %c1, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load i32, ptr %c2, align 4
  %tobool1 = icmp ne i32 %2, 0
  br i1 %tobool1, label %if.then2, label %if.end

if.then2:                                         ; preds = %if.then
  store i32 1, ptr %val, align 4
  br label %if.end

if.end:                                           ; preds = %if.then2, %if.then
  br label %if.end10

if.else:                                          ; preds = %entry
  %3 = load i32, ptr %c3, align 4
  %tobool3 = icmp ne i32 %3, 0
  br i1 %tobool3, label %if.then4, label %if.else5

if.then4:                                         ; preds = %if.else
  store i32 2, ptr %val, align 4
  br label %if.end9

if.else5:                                         ; preds = %if.else
  %4 = load i32, ptr %c4, align 4
  %tobool6 = icmp ne i32 %4, 0
  br i1 %tobool6, label %if.then7, label %if.end8

if.then7:                                         ; preds = %if.else5
  store i32 3, ptr %val, align 4
  br label %if.end8

if.end8:                                          ; preds = %if.then7, %if.else5
  br label %if.end9

if.end9:                                          ; preds = %if.end8, %if.then4
  br label %if.end10

if.end10:                                         ; preds = %if.end9, %if.end
  %5 = load i32, ptr %val, align 4
  ret i32 %5
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


