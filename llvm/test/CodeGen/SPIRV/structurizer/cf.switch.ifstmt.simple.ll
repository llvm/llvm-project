; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=5

;
; int process() {
;   int b = 0;
;   const int t = 50;
;
;   switch(int d = 5) {
;     case t:
;       b = t;
;     case 4:
;     case 5:
;       b = 5;
;       break;
;     default:
;       break;
;   }
;
;   return b;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_13:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb25:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb26:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb27:]] %[[#bb28:]]
; CHECK:    %[[#bb27:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb29:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb30:]] 50 %[[#bb31:]] 4 %[[#bb29:]] 5 %[[#bb32:]]
; CHECK:    %[[#bb28:]] = OpLabel
; CHECK:    %[[#bb30:]] = OpLabel
; CHECK:                  OpBranch %[[#bb29:]]
; CHECK:    %[[#bb31:]] = OpLabel
; CHECK:                  OpBranch %[[#bb29:]]
; CHECK:    %[[#bb32:]] = OpLabel
; CHECK:                  OpBranch %[[#bb29:]]
; CHECK:    %[[#bb29:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb33:]] %[[#bb26:]]
; CHECK:    %[[#bb33:]] = OpLabel
; CHECK:                  OpBranch %[[#bb26:]]
; CHECK:    %[[#bb26:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_21:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb34:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_23:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb35:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %b = alloca i32, align 4
  %t = alloca i32, align 4
  %d = alloca i32, align 4
  store i32 0, ptr %b, align 4
  store i32 50, ptr %t, align 4
  store i32 5, ptr %d, align 4
  %1 = load i32, ptr %d, align 4
  switch i32 %1, label %sw.default [
    i32 50, label %sw.bb
    i32 4, label %sw.bb1
    i32 5, label %sw.bb1
  ]

sw.bb:                                            ; preds = %entry
  store i32 50, ptr %b, align 4
  br label %sw.bb1

sw.bb1:                                           ; preds = %entry, %entry, %sw.bb
  store i32 5, ptr %b, align 4
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb1
  %2 = load i32, ptr %b, align 4
  ret i32 %2
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


