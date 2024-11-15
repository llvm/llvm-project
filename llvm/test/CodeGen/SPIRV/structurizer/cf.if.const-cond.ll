; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; int process() {
;   int a = 0;
;   int b = 0;
;
;   if (3 + 5) {
;     a = 1;
;   } else {
;     a = 0;
;   }
;
;   if (4 + 3 > 7 || 4 + 3 < 8) {
;     b = 2;
;   }
;
;   if (4 + 3 > 7 && true) {
;     b = 0;
;   }
;
;   if (true)
;     ;
;
;   if (false) {}
;
;   return a + b;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK:  %[[#func_9:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb19:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_15:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb20:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_17:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb21:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 0, ptr %a, align 4
  store i32 0, ptr %b, align 4
  store i32 1, ptr %a, align 4
  store i32 2, ptr %b, align 4
  %1 = load i32, ptr %a, align 4
  %2 = load i32, ptr %b, align 4
  %add = add nsw i32 %1, %2
  ret i32 %add
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


