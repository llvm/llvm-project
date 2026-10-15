; RUN: opt %s -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @callbr_results() {
; CHECK-LABEL: for function 'callbr_results'
; CHECK: {{^ +}}%uniform.use =
; CHECK: DIVERGENT: %source.use =
; CHECK: DIVERGENT: %dependent.use =
entry:
  %uniform = callbr i32 asm sideeffect "", "=r,r,!i"(i32 0) #0
      to label %source_block [label %exit]
source_block:
  %uniform.use = add i32 %uniform, 1
  %source = callbr i32 asm sideeffect "", "=r,!i"()
      to label %dependent_block [label %exit]
dependent_block:
  %source.use = add i32 %source, 1
  %dependent = callbr i32 asm sideeffect "", "=r,r,!i"(i32 %source) #0
      to label %use [label %exit]
use:
  %dependent.use = add i32 %dependent, 1
  br label %exit
exit:
  ret void
}

define ptx_kernel void @invoke_results() personality ptr @personality {
; CHECK-LABEL: for function 'invoke_results'
; CHECK: {{^ +}}%uniform.use =
; CHECK: DIVERGENT: %source.use =
; CHECK: DIVERGENT: %dependent.use =
entry:
  %uniform = invoke i32 @dependent(i32 0)
      to label %source_block unwind label %unwind
source_block:
  %uniform.use = add i32 %uniform, 1
  %source = invoke i32 @source()
      to label %dependent_block unwind label %unwind
dependent_block:
  %source.use = add i32 %source, 1
  %dependent = invoke i32 @dependent(i32 %source)
      to label %use unwind label %unwind
use:
  %dependent.use = add i32 %dependent, 1
  ret void
unwind:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

define ptx_kernel void @callbr_result_in_divergent_cycle(i32 %arg) {
; CHECK-LABEL: for function 'callbr_result_in_divergent_cycle'
; CHECK: CYCLES ASSUMED DIVERGENT:
; CHECK: {{^ +}}%value = callbr
; CHECK: DIVERGENT: %use =
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %cond = icmp eq i32 %lane, 0
  br i1 %cond, label %left, label %right
left:
  %value = callbr i32 asm sideeffect "", "=r,r,!i"(i32 %arg) #0
      to label %right [label %exit]
right:
  br label %left
exit:
  %use = add i32 %value, 1
  ret void
}

declare i32 @source()
declare i32 @dependent(i32) #0
declare i32 @personality(...)
declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
attributes #0 = { nodivergencesource }
