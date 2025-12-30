; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; [numthreads(1, 1, 1)]
; void main() {
;   int a, b;
;   int cond = 1;
;
;   while(cond) {
;     switch(b) {
;       default:
;         if (cond) {
;           if (cond)
;             return;
;           else
;             return;
;         }
;     }
;     return;
;   }
; }

; CHECK:  %[[#func_8:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb22:]] = OpLabel
; CHECK:                  OpBranch %[[#bb23:]]
; CHECK:    %[[#bb23:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb24:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb25:]] %[[#bb24:]]
; CHECK:    %[[#bb25:]] = OpLabel
; CHECK:                  OpBranch %[[#bb26:]]
; CHECK:    %[[#bb26:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb27:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb27:]] %[[#bb28:]]
; CHECK:    %[[#bb28:]] = OpLabel
; CHECK:                  OpBranch %[[#bb27:]]
; CHECK:    %[[#bb27:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb29:]] %[[#bb24:]]
; CHECK:    %[[#bb29:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb30:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb30:]] %[[#bb31:]]
; CHECK:    %[[#bb31:]] = OpLabel
; CHECK:                  OpBranch %[[#bb30:]]
; CHECK:    %[[#bb30:]] = OpLabel
; CHECK:                  OpBranch %[[#bb24:]]
; CHECK:    %[[#bb24:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_20:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb32:]] = OpLabel
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
  %cond = alloca i32, align 4
  store i32 1, ptr %cond, align 4
  br label %while.cond

while.cond:                                       ; preds = %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %cond, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %3 = load i32, ptr %b, align 4
  switch i32 %3, label %sw.default [
  ]

sw.default:                                       ; preds = %while.body
  %4 = load i32, ptr %cond, align 4
  %tobool1 = icmp ne i32 %4, 0
  br i1 %tobool1, label %if.then, label %if.end

if.then:                                          ; preds = %sw.default
  %5 = load i32, ptr %cond, align 4
  %tobool2 = icmp ne i32 %5, 0
  br i1 %tobool2, label %if.then3, label %if.else

if.then3:                                         ; preds = %if.then
  br label %while.end

if.else:                                          ; preds = %if.then
  br label %while.end

if.end:                                           ; preds = %sw.default
  br label %sw.epilog

sw.epilog:                                        ; preds = %if.end
  br label %while.end

while.end:                                        ; preds = %if.then3, %if.else, %sw.epilog, %while.cond
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


