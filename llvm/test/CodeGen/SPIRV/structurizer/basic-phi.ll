; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {

; CHECK-DAG:    %[[#int_0:]] = OpConstant %[[#]] 0
; CHECK-DAG:    %[[#int_1:]] = OpConstant %[[#]] 1

; CHECK: %[[#entry:]] = OpLabel
; CHECK:   %[[#var:]] = OpVariable %[[#]] Function
; CHECK:                OpSelectionMerge %[[#merge:]] None
; CHECK:                OpBranchConditional %[[#]] %[[#left:]] %[[#right:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  br i1 true, label %left, label %right

; CHECK:      %[[#right]] = OpLabel
; CHECK-NEXT:               OpStore %[[#var]] %[[#int_1]]
; CHECK-NEXT:               OpBranch %[[#merge]]
right:
  br label %end

; CHECK:      %[[#left]] = OpLabel
; CHECK-NEXT:              OpStore %[[#var]] %[[#int_0]]
; CHECK-NEXT:              OpBranch %[[#merge]]
left:
  br label %end

; CHECK: %[[#merge]] = OpLabel
; CHECK:  %[[#tmp:]] = OpLoad %[[#]] %[[#var]]
; CHECK:               OpReturnValue %[[#tmp]]
end:
  %1 = phi i32 [ 0, %left ], [ 1, %right ]
  ret i32 %1
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
