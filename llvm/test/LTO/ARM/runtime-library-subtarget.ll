; Check that user-defined runtime library function __addsf3vfp is not removed
;
; RUN: llvm-as <%s >%t1
; RUN: llvm-lto -o %t2 %t1 -mcpu arm1176jz-s
; RUN: llvm-nm %t2 | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

; CHECK: ___addsf3vfp

define float @__addsf3vfp(float %a, float %b) #0 {
entry:
  %add = fadd float %a, %b
  ret float %add
}

attributes #0 = { "target-cpu"="arm1176jzf-s"}
