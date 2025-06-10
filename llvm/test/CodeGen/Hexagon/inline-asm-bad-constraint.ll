; RUN: not llc -mtriple=hexagon < %s 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate output register for constraint 'r'

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred() #0 {
entry:
  %a0 = alloca <16 x i32>, align 64
  %0 = call <16 x i32> asm sideeffect "$0 = vmem(r0)", "=r"()
  store <16 x i32> %0, ptr %a0, align 64
  ret void
}

attributes #0 = { noinline nounwind }
