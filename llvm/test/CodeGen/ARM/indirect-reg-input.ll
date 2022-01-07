; RUN: not llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - 2>&1 | FileCheck %s

; Check for error message:
; CHECK: error: inline asm not supported yet: don't know how to handle tied indirect register inputs

%struct.my_stack = type { %struct.myjmp_buf }
%struct.myjmp_buf = type { [6 x i32] }

define void @switch_to_stack(%struct.my_stack* %stack) nounwind {
entry:
  %regs = getelementptr inbounds %struct.my_stack, %struct.my_stack* %stack, i32 0, i32 0
  tail call void asm "\0A", "=*r,*0"(%struct.myjmp_buf* elementtype(%struct.myjmp_buf) %regs, %struct.myjmp_buf* elementtype(%struct.myjmp_buf) %regs)
  ret void
}
