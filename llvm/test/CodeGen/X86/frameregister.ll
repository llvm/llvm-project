; RUN: llc < %s -mtriple=x86_64-apple-darwin  | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnueabi | FileCheck %s
; RUN: opt < %s -O3 -S -mtriple=x86_64-linux-gnueabi | FileCheck %s --check-prefix=OPT

define i64 @get_frame() #0 {
entry:
; CHECK-LABEL: get_frame:
; CHECK: movq	%rbp, %rax
  %sp = call i64 @llvm.read_register.i64(metadata !0)
; OPT: @llvm.read_register.i64
  ret i64 %sp
}

define void @set_frame(i64 %val) #0 {
entry:
; CHECK-LABEL: set_frame:
; CHECK: movq	%rdi, %rbp
  call void @llvm.write_register.i64(metadata !0, i64 %val)
; OPT: @llvm.write_register.i64
  ret void
}

declare i64 @llvm.read_register.i64(metadata) nounwind
declare void @llvm.write_register.i64(metadata, i64) nounwind

; register unsigned long current_stack_pointer asm("rbp");
; CHECK-NOT: .asciz  "rbp"
!0 = !{!"rbp\00"}

attributes #0 = { nounwind "frame-pointer"="all" }
