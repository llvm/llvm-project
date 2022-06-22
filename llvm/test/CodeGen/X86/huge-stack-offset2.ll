; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s --check-prefix=CHECK

; Test how we handle pathologically large stack frames when RAX is live through
; the prologue and epilogue.

declare void @bar(ptr)
declare void @llvm.va_start(ptr)

; For stack frames between 2GB and 16GB, do multiple adjustments.

define i32 @stack_frame_8gb(i32 %x, ...) nounwind {
; CHECK-LABEL: stack_frame_8gb:
; CHECK:      subq ${{.*}}, %rsp # imm = 0x7FFFFFFF
; CHECK:      subq ${{.*}}, %rsp # imm = 0x7FFFFFFF
; CHECK:      subq ${{.*}}, %rsp # imm = 0x7FFFFFFF
; CHECK:      subq ${{.*}}, %rsp # imm = 0x7FFFFFFF
; CHECK:      subq ${{.*}}, %rsp
; CHECK:      callq bar
; CHECK:      addq ${{.*}}, %rsp # imm = 0x7FFFFFFF
; CHECK:      addq ${{.*}}, %rsp # imm = 0x7FFFFFFF
; CHECK:      addq ${{.*}}, %rsp # imm = 0x7FFFFFFF
; CHECK:      addq ${{.*}}, %rsp # imm = 0x7FFFFFFF
; CHECK:      addq ${{.*}}, %rsp
; CHECK:      retq
  %1 = alloca [u0x200000000 x i8]
  %va = alloca i8, i32 24
  call void @llvm.va_start(ptr %va)
  call void @bar(ptr %1)
  ret i32 %x
}

; For stack frames larger than 16GB, spill EAX instead of doing a linear number
; of adjustments.

; This function should have a frame size of 0x4000000D0. The 0xD0 is 208 bytes
; from 24 bytes of va_list, 176 bytes of spilled varargs regparms, and 8 bytes
; of alignment. We subtract 8 less and add 8 more in the prologue and epilogue
; respectively to account for the PUSH.

define i32 @stack_frame_16gb(i32 %x, ...) nounwind {
; CHECK-LABEL: stack_frame_16gb:
; CHECK:      pushq %rax
; CHECK-NEXT: movabsq ${{.*}}, %rax # imm = 0xFFFFFFFBFFFFFF38
; CHECK-NEXT: addq %rsp, %rax
; CHECK-NEXT: xchgq %rax, (%rsp)
; CHECK-NEXT: movq (%rsp), %rsp
; CHECK:      callq bar
; CHECK:      pushq %rax
; CHECK-NEXT: movabsq ${{.*}}, %rax # imm = 0x4000000D8
; CHECK-NEXT: addq %rsp, %rax
; CHECK-NEXT: xchgq %rax, (%rsp)
; CHECK-NEXT: movq (%rsp), %rsp
; CHECK:      retq
  %1 = alloca [u0x400000000 x i8]
  %va = alloca i8, i32 24
  call void @llvm.va_start(ptr %va)
  call void @bar(ptr %1)
  ret i32 %x
}

