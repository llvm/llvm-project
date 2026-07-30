; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s

define i64 @get_stack() nounwind {
entry:
; CHECK-LABEL: get_stack:
; CHECK: mov   x0, sp
	%sp = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %sp
}

define void @set_stack(i64 %val) nounwind {
entry:
; CHECK-LABEL: set_stack:
; CHECK: mov   sp, x0
  call void @llvm.write_register.i64(metadata !0, i64 %val)
  ret void
}

declare i64 @llvm.read_register.i64(metadata) nounwind
declare void @llvm.write_register.i64(metadata, i64) nounwind

; register unsigned long current_stack_pointer asm("sp");
; CHECK-NOT: .asciz  "sp"
!0 = !{!"sp\00"}
