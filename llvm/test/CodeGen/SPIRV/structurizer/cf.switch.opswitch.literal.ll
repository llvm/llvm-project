; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; [numthreads(1, 1, 1)]
; void main() {
; ///////////////////////////////
; // 32-bit int literal switch //
; ///////////////////////////////
;   switch (0) {
;   case 0:
;     return;
;   default:
;     return;
;   }
;
; ///////////////////////////////
; // 64-bit int literal switch //
; ///////////////////////////////
;   switch (12345678910) {
;   case 12345678910:
;     return;
;   }
;
;   return;
; }

; CHECK: %[[#func_3:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb6:]] = OpLabel
; CHECK:                 OpReturn
; CHECK:                 OpFunctionEnd
; CHECK: %[[#func_4:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb7:]] = OpLabel
; CHECK:                 OpReturn
; CHECK:                 OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
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

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}


