; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=2563170

;
; int process() {
;   int val = 0;
;
;   for (int i = 0; i < 10; ++i) {
;     val = val + i;
;
;     for (int j = 0; j < 2; ++j) {
;       for (int k = 0; k < 2; ++k) {
;         val = val + k;
;       }
;
;       val = val * 2;
;
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
; CHECK:    %[[#bb41:]] = OpLabel
; CHECK:                  OpBranch %[[#bb42:]]
; CHECK:    %[[#bb42:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb43:]] %[[#bb44:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb45:]] %[[#bb43:]]
; CHECK:    %[[#bb45:]] = OpLabel
; CHECK:                  OpBranch %[[#bb46:]]
; CHECK:    %[[#bb46:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb47:]] %[[#bb48:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb49:]] %[[#bb47:]]
; CHECK:    %[[#bb49:]] = OpLabel
; CHECK:                  OpBranch %[[#bb50:]]
; CHECK:    %[[#bb50:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb51:]] %[[#bb52:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb53:]] %[[#bb51:]]
; CHECK:    %[[#bb53:]] = OpLabel
; CHECK:                  OpBranch %[[#bb52:]]
; CHECK:    %[[#bb52:]] = OpLabel
; CHECK:                  OpBranch %[[#bb50:]]
; CHECK:    %[[#bb51:]] = OpLabel
; CHECK:                  OpBranch %[[#bb48:]]
; CHECK:    %[[#bb48:]] = OpLabel
; CHECK:                  OpBranch %[[#bb46:]]
; CHECK:    %[[#bb47:]] = OpLabel
; CHECK:                  OpBranch %[[#bb44:]]
; CHECK:    %[[#bb44:]] = OpLabel
; CHECK:                  OpBranch %[[#bb42:]]
; CHECK:    %[[#bb43:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_37:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb54:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_39:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb55:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc11, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 10
  br i1 %cmp, label %for.body, label %for.end13

for.body:                                         ; preds = %for.cond
  %3 = load i32, ptr %val, align 4
  %4 = load i32, ptr %i, align 4
  %add = add nsw i32 %3, %4
  store i32 %add, ptr %val, align 4
  store i32 0, ptr %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc8, %for.body
  %5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %1) ]
  %6 = load i32, ptr %j, align 4
  %cmp2 = icmp slt i32 %6, 2
  br i1 %cmp2, label %for.body3, label %for.end10

for.body3:                                        ; preds = %for.cond1
  store i32 0, ptr %k, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %7 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %5) ]
  %8 = load i32, ptr %k, align 4
  %cmp5 = icmp slt i32 %8, 2
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %9 = load i32, ptr %val, align 4
  %10 = load i32, ptr %k, align 4
  %add7 = add nsw i32 %9, %10
  store i32 %add7, ptr %val, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %11 = load i32, ptr %k, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, ptr %k, align 4
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  %12 = load i32, ptr %val, align 4
  %mul = mul nsw i32 %12, 2
  store i32 %mul, ptr %val, align 4
  br label %for.inc8

for.inc8:                                         ; preds = %for.end
  %13 = load i32, ptr %j, align 4
  %inc9 = add nsw i32 %13, 1
  store i32 %inc9, ptr %j, align 4
  br label %for.cond1

for.end10:                                        ; preds = %for.cond1
  br label %for.inc11

for.inc11:                                        ; preds = %for.end10
  %14 = load i32, ptr %i, align 4
  %inc12 = add nsw i32 %14, 1
  store i32 %inc12, ptr %i, align 4
  br label %for.cond

for.end13:                                        ; preds = %for.cond
  %15 = load i32, ptr %val, align 4
  ret i32 %15
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.loop() #1

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


