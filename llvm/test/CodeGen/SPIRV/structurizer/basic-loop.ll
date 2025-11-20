; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {

; CHECK: %[[#entry:]] = OpLabel
; CHECK:                OpBranch %[[#header:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %1 = alloca i32, align 4
  br label %header

; CHECK: %[[#header]] = OpLabel
; CHECK:                OpLoopMerge %[[#merge:]] %[[#continue:]] None
; CHECK:                OpBranchConditional %[[#]] %[[#body:]] %[[#merge]]
header:
  %2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  br i1 true, label %body, label %merge

; CHECK: %[[#merge]] = OpLabel
; CHECK:               OpReturnValue %[[#]]
merge:
  ret i32 0

; CHECK: %[[#body]] = OpLabel
; CHECK:              OpBranch %[[#continue]]
body:
  store i32 0, ptr %1
  br label %continue

continue:
  br label %header
; CHECK: %[[#continue]] = OpLabel
; CHECK:                  OpBranch %[[#header]]
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.loop() #1


attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent }

!llvm.module.flags = !{!0, !1, !2}


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}

