; RUN: llc < %s -mtriple=arm-apple-darwin  | FileCheck %s
; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s

define i32 @get_stack() nounwind {
entry:
; CHECK-LABEL: get_stack:
; CHECK: mov   r0, sp
	%sp = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %sp
}

define void @set_stack(i32 %val) nounwind {
entry:
; CHECK-LABEL: set_stack:
; CHECK: mov   sp, r0
  call void @llvm.write_register.i32(metadata !0, i32 %val)
  ret void
}

declare i32 @llvm.read_register.i32(metadata) nounwind
declare void @llvm.write_register.i32(metadata, i32) nounwind

; register unsigned long current_stack_pointer asm("sp");
; CHECK-NOT: .asciz  "sp"
!0 = !{!"sp\00"}
