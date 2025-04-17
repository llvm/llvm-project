; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; int foo() { return 200; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   int result;
;
;   int a = 0;
;   switch(a) {
;     case -3:
;       result = -300;
;       break;
;     case 0:
;       result = 0;
;       break;
;     case 1:
;       result = 100;
;       break;
;     case 2:
;       result = foo();
;       break;
;     default:
;       result = 777;
;       break;
;   }
;
;   switch(int c = a) {
;     case -4:
;       result = -400;
;       break;
;     case 4:
;       result = 400;
;       break;
;   }
; }

; CHECK: %[[#func_14:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb25:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_15:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb26:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb27:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb28:]] 4294967293 %[[#bb29:]] 0 %[[#bb30:]] 1 %[[#bb31:]] 2 %[[#bb32:]]
; CHECK:    %[[#bb28:]] = OpLabel
; CHECK:                  OpBranch %[[#bb27:]]
; CHECK:    %[[#bb29:]] = OpLabel
; CHECK:                  OpBranch %[[#bb27:]]
; CHECK:    %[[#bb30:]] = OpLabel
; CHECK:                  OpBranch %[[#bb27:]]
; CHECK:    %[[#bb31:]] = OpLabel
; CHECK:                  OpBranch %[[#bb27:]]
; CHECK:    %[[#bb32:]] = OpLabel
; CHECK:                  OpBranch %[[#bb27:]]
; CHECK:    %[[#bb27:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb33:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb33:]] 4294967292 %[[#bb34:]] 4 %[[#bb35:]]
; CHECK:    %[[#bb34:]] = OpLabel
; CHECK:                  OpBranch %[[#bb33:]]
; CHECK:    %[[#bb35:]] = OpLabel
; CHECK:                  OpBranch %[[#bb33:]]
; CHECK:    %[[#bb33:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_23:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb36:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z3foov() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 200
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %result = alloca i32, align 4
  %a = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %1 = load i32, ptr %a, align 4
  switch i32 %1, label %sw.default [
    i32 -3, label %sw.bb
    i32 0, label %sw.bb1
    i32 1, label %sw.bb2
    i32 2, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  store i32 -300, ptr %result, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  store i32 0, ptr %result, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  store i32 100, ptr %result, align 4
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  %call4 = call spir_func noundef i32 @_Z3foov() #3 [ "convergencectrl"(token %0) ]
  store i32 %call4, ptr %result, align 4
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  store i32 777, ptr %result, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %2 = load i32, ptr %a, align 4
  store i32 %2, ptr %c, align 4
  %3 = load i32, ptr %c, align 4
  switch i32 %3, label %sw.epilog7 [
    i32 -4, label %sw.bb5
    i32 4, label %sw.bb6
  ]

sw.bb5:                                           ; preds = %sw.epilog
  store i32 -400, ptr %result, align 4
  br label %sw.epilog7

sw.bb6:                                           ; preds = %sw.epilog
  store i32 400, ptr %result, align 4
  br label %sw.epilog7

sw.epilog7:                                       ; preds = %sw.epilog, %sw.bb6, %sw.bb5
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


