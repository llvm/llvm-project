; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-unknown -O0 < %s | FileCheck %s -check-prefix=CHECK0

%struct.interrupt_frame = type { i64, i64, i64, i64, i64 }

@llvm.used = appending global [3 x i8*] [i8* bitcast (void (%struct.interrupt_frame*)* @test_isr_no_ecode to i8*), i8* bitcast (void (%struct.interrupt_frame*, i64)* @test_isr_ecode to i8*), i8* bitcast (void (%struct.interrupt_frame*, i64)* @test_isr_clobbers to i8*)], section "llvm.metadata"

; Spills rax, putting original esp at +8.
; No stack adjustment if declared with no error code
define x86_intrcc void @test_isr_no_ecode(%struct.interrupt_frame* %frame) {
  ; CHECK-LABEL: test_isr_no_ecode:
  ; CHECK: pushq %rax
  ; CHECK: movq 24(%rsp), %rax
  ; CHECK: popq %rax
  ; CHECK: iretq
  ; CHECK0-LABEL: test_isr_no_ecode:
  ; CHECK0: pushq %rax
  ; CHECK0: leaq 8(%rsp), %rax
  ; CHECK0: movq 16(%rax), %rax
  ; CHECK0: popq %rax
  ; CHECK0: iretq
  %pflags = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %frame, i32 0, i32 2
  %flags = load i64, i64* %pflags, align 4
  call void asm sideeffect "", "r"(i64 %flags)
  ret void
}

; Spills rax and rcx, putting original rsp at +16. Stack is adjusted up another 8 bytes
; before return, popping the error code.
define x86_intrcc void @test_isr_ecode(%struct.interrupt_frame* %frame, i64 %ecode) {
  ; CHECK-LABEL: test_isr_ecode
  ; CHECK: pushq %rax
  ; CHECK: pushq %rcx
  ; CHECK: movq 16(%rsp), %rax
  ; CHECK: movq 40(%rsp), %rcx
  ; CHECK: popq %rcx
  ; CHECK: popq %rax
  ; CHECK: addq $8, %rsp
  ; CHECK: iretq
  ; CHECK0-LABEL: test_isr_ecode
  ; CHECK0: pushq %rax
  ; CHECK0: pushq %rcx
  ; CHECK0: movq 16(%rsp), %rax
  ; CHECK0: leaq 24(%rsp), %rcx
  ; CHECK0: movq 16(%rcx), %rcx
  ; CHECK0: popq %rcx
  ; CHECK0: popq %rax
  ; CHECK0: addq $8, %rsp
  ; CHECK0: iretq
  %pflags = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %frame, i32 0, i32 2
  %flags = load i64, i64* %pflags, align 4
  call void asm sideeffect "", "r,r"(i64 %flags, i64 %ecode)
  ret void
}

; All clobbered registers must be saved
define x86_intrcc void @test_isr_clobbers(%struct.interrupt_frame* %frame, i64 %ecode) {
  call void asm sideeffect "", "~{rax},~{rbx},~{rbp},~{r11},~{xmm0}"()
  ; CHECK-LABEL: test_isr_clobbers
  ; CHECK-SSE-NEXT: pushq %rax
  ; CHECK-SSE-NEXT; pushq %r11
  ; CHECK-SSE-NEXT: pushq %rbp
  ; CHECK-SSE-NEXT: pushq %rbx
  ; CHECK-SSE-NEXT: movaps %xmm0
  ; CHECK-SSE-NEXT: movaps %xmm0
  ; CHECK-SSE-NEXT: popq %rbx
  ; CHECK-SSE-NEXT: popq %rbp
  ; CHECK-SSE-NEXT: popq %r11
  ; CHECK-SSE-NEXT: popq %rax
  ; CHECK-SSE-NEXT: addq $8, %rsp
  ; CHECK-SSE-NEXT: iretq
  ; CHECK0-LABEL: test_isr_clobbers
  ; CHECK0-SSE-NEXT: pushq %rax
  ; CHECK0-SSE-NEXT; pushq %r11
  ; CHECK0-SSE-NEXT: pushq %rbp
  ; CHECK0-SSE-NEXT: pushq %rbx
  ; CHECK0-SSE-NEXT: movaps %xmm0
  ; CHECK0-SSE-NEXT: movaps %xmm0
  ; CHECK0-SSE-NEXT: popq %rbx
  ; CHECK0-SSE-NEXT: popq %rbp
  ; CHECK0-SSE-NEXT: popq %r11
  ; CHECK0-SSE-NEXT: popq %rax
  ; CHECK0-SSE-NEXT: addq $8, %rsp
  ; CHECK0-SSE-NEXT: iretq
  ret void
}