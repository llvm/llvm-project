; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; static int foo() { return 200; }
;
; static int process() {
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

; CHECK: %[[#func:]] = OpFunction %[[#]] DontInline %[[#]]
; CHECK: %[[#bb30:]] = OpLabel
; CHECK:               OpSelectionMerge %[[#bb31:]] None
; CHECK:               OpBranchConditional %[[#]] %[[#bb32:]] %[[#bb33:]]

; CHECK:  %[[#bb33]] = OpLabel
; CHECK:               OpUnreachable

; CHECK:  %[[#bb32]] = OpLabel
; CHECK:               OpSelectionMerge %[[#bb34:]] None
; CHECK:               OpBranchConditional %[[#]] %[[#bb35:]] %[[#bb36:]]

; CHECK:  %[[#bb36]] = OpLabel
; CHECK:               OpUnreachable

; CHECK:  %[[#bb35]] = OpLabel
; CHECK:               OpSelectionMerge %[[#bb37:]] None
; CHECK:               OpBranchConditional %[[#]] %[[#bb38:]] %[[#bb39:]]

; CHECK:  %[[#bb39]] = OpLabel
; CHECK:               OpUnreachable

; CHECK:  %[[#bb38]] = OpLabel
; CHECK:               OpSelectionMerge %[[#bb40:]] None
; CHECK:               OpSwitch %[[#]] %[[#bb41:]] 1 %[[#bb42:]] 2 %[[#bb43:]] 3 %[[#bb44:]] 140 %[[#bb45:]] 4 %[[#bb46:]] 5 %[[#bb47:]] 6 %[[#bb48:]] 7 %[[#bb49:]]

; CHECK:  %[[#bb49]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]
; CHECK:  %[[#bb48]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]
; CHECK:  %[[#bb47]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]
; CHECK:  %[[#bb46]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]
; CHECK:  %[[#bb45]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]
; CHECK:  %[[#bb44]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]
; CHECK:  %[[#bb43]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]
; CHECK:  %[[#bb42]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]
; CHECK:  %[[#bb41]] = OpLabel
; CHECK:               OpBranch %[[#bb40]]

; CHECK:  %[[#bb40]] = OpLabel
; CHECK:               OpSelectionMerge %[[#bb50:]] None
; CHECK:               OpSwitch %[[#]] %[[#bb50]] 1 %[[#bb51:]] 2 %[[#bb52:]] 3 %[[#bb53:]]
; CHECK:  %[[#bb53]] = OpLabel
; CHECK:               OpBranch %[[#bb50]]
; CHECK:  %[[#bb52]] = OpLabel
; CHECK:               OpBranch %[[#bb50]]
; CHECK:  %[[#bb51]] = OpLabel
; CHECK:               OpBranch %[[#bb50]]
; CHECK:  %[[#bb50]] = OpLabel
; CHECK:               OpBranch %[[#bb37]]

; CHECK:  %[[#bb37]] = OpLabel
; CHECK:               OpSelectionMerge %[[#bb54:]] None
; CHECK:               OpSwitch %[[#]] %[[#bb54]] 1 %[[#bb55:]] 2 %[[#bb56:]]
; CHECK:  %[[#bb56]] = OpLabel
; CHECK:               OpBranch %[[#bb54]]
; CHECK:  %[[#bb55]] = OpLabel
; CHECK:               OpBranch %[[#bb54]]
; CHECK:  %[[#bb54]] = OpLabel
; CHECK:               OpBranch %[[#bb34]]

; CHECK:  %[[#bb34]] = OpLabel
; CHECK:               OpBranch %[[#bb31]]

; CHECK:  %[[#bb31]] = OpLabel
; CHECK:               OpReturn
; CHECK:               OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse
define void @main() #0 {
entry:
  %a.i = alloca i32, align 4
  %b.i = alloca i32, align 4
  %c.i = alloca i32, align 4
  %r.i = alloca i32, align 4
  %s.i = alloca i32, align 4
  %t.i = alloca i32, align 4
  %d.i = alloca i32, align 4
  %0 = call token @llvm.experimental.convergence.entry()
  store i32 0, ptr %a.i, align 4
  store i32 0, ptr %b.i, align 4
  store i32 0, ptr %c.i, align 4
  store i32 20, ptr %r.i, align 4
  store i32 40, ptr %s.i, align 4
  store i32 140, ptr %t.i, align 4
  store i32 5, ptr %d.i, align 4
  %1 = load i32, ptr %d.i, align 4
  switch i32 %1, label %sw.default.i [
    i32 1, label %sw.bb.i
    i32 2, label %sw.bb3.i
    i32 3, label %sw.bb5.i
    i32 140, label %sw.bb7.i
    i32 4, label %sw.bb9.i
    i32 5, label %sw.bb9.i
    i32 6, label %sw.bb11.i
    i32 7, label %sw.bb12.i
  ]

sw.bb.i:
  %2 = load i32, ptr %b.i, align 4
  %add.i = add nsw i32 %2, 1
  store i32 %add.i, ptr %b.i, align 4
  %3 = load i32, ptr %c.i, align 4
  %add2.i = add nsw i32 %3, 200
  store i32 %add2.i, ptr %c.i, align 4
  br label %sw.bb3.i

sw.bb3.i:
  %4 = load i32, ptr %b.i, align 4
  %add4.i = add nsw i32 %4, 2
  store i32 %add4.i, ptr %b.i, align 4
  br label %_ZL7processv.exit

sw.bb5.i:
  %5 = load i32, ptr %b.i, align 4
  %add6.i = add nsw i32 %5, 3
  store i32 %add6.i, ptr %b.i, align 4
  br label %_ZL7processv.exit

sw.bb7.i:
  %6 = load i32, ptr %b.i, align 4
  %add8.i = add nsw i32 %6, 140
  store i32 %add8.i, ptr %b.i, align 4
  br label %sw.bb9.i

sw.bb9.i:
  %7 = load i32, ptr %b.i, align 4
  %add10.i = add nsw i32 %7, 5
  store i32 %add10.i, ptr %b.i, align 4
  br label %_ZL7processv.exit

sw.bb11.i:
  br label %sw.bb12.i

sw.bb12.i:
  br label %_ZL7processv.exit

sw.default.i:
  br label %_ZL7processv.exit

_ZL7processv.exit:
  %8 = load i32, ptr %a.i, align 4
  %9 = load i32, ptr %b.i, align 4
  %add13.i = add nsw i32 %8, %9
  %10 = load i32, ptr %c.i, align 4
  %add14.i = add nsw i32 %add13.i, %10
  ret void
}

declare token @llvm.experimental.convergence.entry() #1

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
