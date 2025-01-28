; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; [numthreads(1, 1, 1)]
; void main() {
;   int val=0, i=0, j=0, k=0;
;
;   while (i < 10) {
;     val = val + i;
;     while (j < 20) {
;       while (k < 30) {
;         val = val + k;
;         ++k;
;       }
;
;       val = val * 2;
;       ++j;
;     }
;
;     ++i;
;   }
; }

; CHECK: %[[#func_12:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb39:]] = OpLabel
; CHECK:                  OpBranch %[[#bb40:]]
; CHECK:     %[[#bb40]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb41:]] %[[#bb42:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb43:]] %[[#bb41]]
; CHECK:     %[[#bb41]] = OpLabel
; CHECK:                  OpReturn
; CHECK:     %[[#bb43]] = OpLabel
; CHECK:                  OpBranch %[[#bb44:]]
; CHECK:     %[[#bb44]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb45:]] %[[#bb46:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb47:]] %[[#bb45]]
; CHECK:     %[[#bb45]] = OpLabel
; CHECK:                  OpBranch %[[#bb42]]
; CHECK:     %[[#bb42]] = OpLabel
; CHECK:                  OpBranch %[[#bb40]]
; CHECK:     %[[#bb47]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb48:]] %[[#bb49:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb49]] %[[#bb48]]
; CHECK:     %[[#bb48]] = OpLabel
; CHECK:                  OpBranch %[[#bb46]]
; CHECK:     %[[#bb46]] = OpLabel
; CHECK:                  OpBranch %[[#bb44]]
; CHECK:     %[[#bb49]] = OpLabel
; CHECK:                  OpBranch %[[#bb47]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_37:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb50:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  store i32 0, ptr %j, align 4
  store i32 0, ptr %k, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.end9, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 10
  br i1 %cmp, label %while.body, label %while.end11

while.body:                                       ; preds = %while.cond
  %3 = load i32, ptr %val, align 4
  %4 = load i32, ptr %i, align 4
  %add = add nsw i32 %3, %4
  store i32 %add, ptr %val, align 4
  br label %while.cond1

while.cond1:                                      ; preds = %while.end, %while.body
  %5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %1) ]
  %6 = load i32, ptr %j, align 4
  %cmp2 = icmp slt i32 %6, 20
  br i1 %cmp2, label %while.body3, label %while.end9

while.body3:                                      ; preds = %while.cond1
  br label %while.cond4

while.cond4:                                      ; preds = %while.body6, %while.body3
  %7 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %5) ]
  %8 = load i32, ptr %k, align 4
  %cmp5 = icmp slt i32 %8, 30
  br i1 %cmp5, label %while.body6, label %while.end

while.body6:                                      ; preds = %while.cond4
  %9 = load i32, ptr %val, align 4
  %10 = load i32, ptr %k, align 4
  %add7 = add nsw i32 %9, %10
  store i32 %add7, ptr %val, align 4
  %11 = load i32, ptr %k, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, ptr %k, align 4
  br label %while.cond4

while.end:                                        ; preds = %while.cond4
  %12 = load i32, ptr %val, align 4
  %mul = mul nsw i32 %12, 2
  store i32 %mul, ptr %val, align 4
  %13 = load i32, ptr %j, align 4
  %inc8 = add nsw i32 %13, 1
  store i32 %inc8, ptr %j, align 4
  br label %while.cond1

while.end9:                                       ; preds = %while.cond1
  %14 = load i32, ptr %i, align 4
  %inc10 = add nsw i32 %14, 1
  store i32 %inc10, ptr %i, align 4
  br label %while.cond

while.end11:                                      ; preds = %while.cond
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


