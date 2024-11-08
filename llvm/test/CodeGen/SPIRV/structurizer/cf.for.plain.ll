; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; CHECK-DAG: OpName %[[#process:]] "_Z7processv"
; CHECK-DAG: OpName %[[#val:]] "val"
; CHECK-DAG: OpName %[[#i:]] "i"

; CHECK-DAG: %[[#int_ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#int_pfty:]] = OpTypePointer Function %[[#int_ty]]
; CHECK-DAG: %[[#false:]] = OpConstantFalse


; CHECK: %[[#process]] = OpFunction %[[#]] DontInline %[[#]]
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()

  ; CHECK-DAG: %[[#val]] = OpVariable %[[#int_pfty]] Function
  ; CHECK-DAG:   %[[#i]] = OpVariable %[[#int_pfty]] Function
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond
  ; CHECK: OpBranch %[[#for_cond:]]

for.cond:                                         ; preds = %for.inc, %entry
  ; CHECK: %[[#for_cond]] = OpLabel
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 10

  ; CHECK: OpLoopMerge %[[#for_end:]] %[[#for_inc:]]
  ; CHECK: OpBranchConditional %[[#]] %[[#for_body:]] %[[#for_end]]
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32, ptr %i, align 4
  store i32 %3, ptr %val, align 4
  br label %for.inc
  ; CHECK: %[[#for_body]] = OpLabel
  ; CHECK:                 OpBranch %[[#for_inc]]

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond
  ; CHECK: %[[#for_inc]] = OpLabel
  ; CHECK:                 OpBranch %[[#for_cond]]

for.end:                                          ; preds = %for.cond
  br label %for.cond1
  ; CHECK: %[[#for_end]] = OpLabel
  ; CHECK:                 OpBranch %[[#for_cond1:]]

for.cond1:                                        ; preds = %for.cond1, %for.end
  %5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  store i32 0, ptr %val, align 4
  br label %for.cond1
  ; CHECK: %[[#for_cond1]] = OpLabel
  ; CHECK: OpLoopMerge %[[#unreachable:]] %[[#for_cond1]]
  ; CHECK: OpBranchConditional %[[#false]] %[[#unreachable]] %[[#for_cond1]]

  ; CHECK: %[[#unreachable]] = OpLabel
  ; CHECK-NEXT:                OpUnreachable
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.loop() #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #2 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %call1 = call spir_func noundef i32 @_Z7processv() #4 [ "convergencectrl"(token %0) ]
  ret void
}

; Function Attrs: convergent norecurse
define void @main.1() #3 {
entry:
  call void @main()
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { convergent }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
