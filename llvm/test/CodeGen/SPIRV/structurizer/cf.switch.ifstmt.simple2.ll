; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=5

;
; int foo() { return 200; }
;
; int process() {
;   int a = 0;
;   int b = 0;
;   int c = 0;
;   const int r = 20;
;   const int s = 40;
;   const int t = 3*r+2*s;
;
;   switch(int d = 5) {
;     case 1:
;       b += 1;
;       c += foo();
;     case 2:
;       b += 2;
;       break;
;     case 3:
;     {
;       b += 3;
;       break;
;     }
;     case t:
;       b += t;
;     case 4:
;     case 5:
;       b += 5;
;       break;
;     case 6: {
;     case 7:
;       break;}
;     default:
;       break;
;   }
;
;   return a + b + c;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_18:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb52:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_19:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb53:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb54:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb55:]] %[[#bb56:]]
; CHECK:    %[[#bb55:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb57:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb58:]] %[[#bb59:]]
; CHECK:    %[[#bb56:]] = OpLabel
; CHECK:    %[[#bb58:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb60:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb61:]] %[[#bb62:]]
; CHECK:    %[[#bb59:]] = OpLabel
; CHECK:    %[[#bb61:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb63:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb64:]] 1 %[[#bb65:]] 2 %[[#bb63:]] 3 %[[#bb66:]] 140 %[[#bb67:]] 4 %[[#bb68:]] 5 %[[#bb69:]] 6 %[[#bb70:]] 7 %[[#bb71:]]
; CHECK:    %[[#bb62:]] = OpLabel
; CHECK:    %[[#bb64:]] = OpLabel
; CHECK:                  OpBranch %[[#bb63:]]
; CHECK:    %[[#bb65:]] = OpLabel
; CHECK:                  OpBranch %[[#bb63:]]
; CHECK:    %[[#bb66:]] = OpLabel
; CHECK:                  OpBranch %[[#bb63:]]
; CHECK:    %[[#bb67:]] = OpLabel
; CHECK:                  OpBranch %[[#bb63:]]
; CHECK:    %[[#bb68:]] = OpLabel
; CHECK:                  OpBranch %[[#bb63:]]
; CHECK:    %[[#bb69:]] = OpLabel
; CHECK:                  OpBranch %[[#bb63:]]
; CHECK:    %[[#bb70:]] = OpLabel
; CHECK:                  OpBranch %[[#bb63:]]
; CHECK:    %[[#bb71:]] = OpLabel
; CHECK:                  OpBranch %[[#bb63:]]
; CHECK:    %[[#bb63:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb72:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb73:]] 1 %[[#bb72:]] 2 %[[#bb74:]] 3 %[[#bb75:]]
; CHECK:    %[[#bb73:]] = OpLabel
; CHECK:                  OpBranch %[[#bb72:]]
; CHECK:    %[[#bb74:]] = OpLabel
; CHECK:                  OpBranch %[[#bb72:]]
; CHECK:    %[[#bb75:]] = OpLabel
; CHECK:                  OpBranch %[[#bb72:]]
; CHECK:    %[[#bb72:]] = OpLabel
; CHECK:                  OpBranch %[[#bb60:]]
; CHECK:    %[[#bb60:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb76:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb77:]] 1 %[[#bb76:]] 2 %[[#bb78:]]
; CHECK:    %[[#bb77:]] = OpLabel
; CHECK:                  OpBranch %[[#bb76:]]
; CHECK:    %[[#bb78:]] = OpLabel
; CHECK:                  OpBranch %[[#bb76:]]
; CHECK:    %[[#bb76:]] = OpLabel
; CHECK:                  OpBranch %[[#bb57:]]
; CHECK:    %[[#bb57:]] = OpLabel
; CHECK:                  OpBranch %[[#bb54:]]
; CHECK:    %[[#bb54:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_48:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb79:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_50:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb80:]] = OpLabel
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
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %r = alloca i32, align 4
  %s = alloca i32, align 4
  %t = alloca i32, align 4
  %d = alloca i32, align 4
  store i32 0, ptr %a, align 4
  store i32 0, ptr %b, align 4
  store i32 0, ptr %c, align 4
  store i32 20, ptr %r, align 4
  store i32 40, ptr %s, align 4
  store i32 140, ptr %t, align 4
  store i32 5, ptr %d, align 4
  %1 = load i32, ptr %d, align 4
  switch i32 %1, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb3
    i32 3, label %sw.bb5
    i32 140, label %sw.bb7
    i32 4, label %sw.bb9
    i32 5, label %sw.bb9
    i32 6, label %sw.bb11
    i32 7, label %sw.bb12
  ]

sw.bb:                                            ; preds = %entry
  %2 = load i32, ptr %b, align 4
  %add = add nsw i32 %2, 1
  store i32 %add, ptr %b, align 4
  %call1 = call spir_func noundef i32 @_Z3foov() #3 [ "convergencectrl"(token %0) ]
  %3 = load i32, ptr %c, align 4
  %add2 = add nsw i32 %3, %call1
  store i32 %add2, ptr %c, align 4
  br label %sw.bb3

sw.bb3:                                           ; preds = %entry, %sw.bb
  %4 = load i32, ptr %b, align 4
  %add4 = add nsw i32 %4, 2
  store i32 %add4, ptr %b, align 4
  br label %sw.epilog

sw.bb5:                                           ; preds = %entry
  %5 = load i32, ptr %b, align 4
  %add6 = add nsw i32 %5, 3
  store i32 %add6, ptr %b, align 4
  br label %sw.epilog

sw.bb7:                                           ; preds = %entry
  %6 = load i32, ptr %b, align 4
  %add8 = add nsw i32 %6, 140
  store i32 %add8, ptr %b, align 4
  br label %sw.bb9

sw.bb9:                                           ; preds = %entry, %entry, %sw.bb7
  %7 = load i32, ptr %b, align 4
  %add10 = add nsw i32 %7, 5
  store i32 %add10, ptr %b, align 4
  br label %sw.epilog

sw.bb11:                                          ; preds = %entry
  br label %sw.bb12

sw.bb12:                                          ; preds = %entry, %sw.bb11
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb12, %sw.bb9, %sw.bb5, %sw.bb3
  %8 = load i32, ptr %a, align 4
  %9 = load i32, ptr %b, align 4
  %add13 = add nsw i32 %8, %9
  %10 = load i32, ptr %c, align 4
  %add14 = add nsw i32 %add13, %10
  ret i32 %add14
}

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


