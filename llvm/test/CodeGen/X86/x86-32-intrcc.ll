; RUN: llc -mtriple=i686-unknown-unknown < %s | FileCheck %s
; RUN: llc -mtriple=i686-unknown-unknown -O0 < %s | FileCheck %s -check-prefix=CHECK0

%struct.interrupt_frame = type { i32, i32, i32, i32, i32 }

@llvm.used = appending global [3 x i8*] [i8* bitcast (void (%struct.interrupt_frame*)* @test_isr_no_ecode to i8*), i8* bitcast (void (%struct.interrupt_frame*, i32)* @test_isr_ecode to i8*), i8* bitcast (void (%struct.interrupt_frame*, i32)* @test_isr_clobbers to i8*)], section "llvm.metadata"

; Spills eax, putting original esp at +4.
; No stack adjustment if declared with no error code
define x86_intrcc void @test_isr_no_ecode(%struct.interrupt_frame* %frame) {
  ; CHECK-LABEL: test_isr_no_ecode:
  ; CHECK: pushl %eax
  ; CHECK: movl 12(%esp), %eax
  ; CHECK: popl %eax
  ; CHECK: iretl
  ; CHECK0-LABEL: test_isr_no_ecode:
  ; CHECK0: pushl %eax
  ; CHECK0: leal 4(%esp), %eax
  ; CHECK0: movl 8(%eax), %eax
  ; CHECK0: popl %eax
  ; CHECK0: iretl
  %pflags = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %frame, i32 0, i32 2
  %flags = load i32, i32* %pflags, align 4
  call void asm sideeffect "", "r"(i32 %flags)
  ret void
}

; Spills eax and ecx, putting original esp at +8. Stack is adjusted up another 4 bytes
; before return, popping the error code.
define x86_intrcc void @test_isr_ecode(%struct.interrupt_frame* %frame, i32 %ecode) {
  ; CHECK-LABEL: test_isr_ecode
  ; CHECK: pushl %ecx
  ; CHECK: pushl %eax
  ; CHECK: movl 8(%esp), %eax
  ; CHECK: movl 20(%esp), %ecx
  ; CHECK: popl %eax
  ; CHECK: popl %ecx
  ; CHECK: addl $4, %esp
  ; CHECK: iretl
  ; CHECK0-LABEL: test_isr_ecode
  ; CHECK0: pushl %ecx
  ; CHECK0: pushl %eax
  ; CHECK0: movl 8(%esp), %eax
  ; CHECK0: leal 12(%esp), %ecx
  ; CHECK0: movl 8(%ecx), %ecx
  ; CHECK0: popl %eax
  ; CHECK0: popl %ecx
  ; CHECK0: addl $4, %esp
  ; CHECK0: iretl
  %pflags = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %frame, i32 0, i32 2
  %flags = load i32, i32* %pflags, align 4
  call x86_fastcallcc void asm sideeffect "", "r,r"(i32 %flags, i32 %ecode)
  ret void
}

; All clobbered registers must be saved
define x86_intrcc void @test_isr_clobbers(%struct.interrupt_frame* %frame, i32 %ecode) {
  call void asm sideeffect "", "~{eax},~{ebx},~{ebp}"()
  ; CHECK-LABEL: test_isr_clobbers
  ; CHECK-SSE-NEXT: pushl %ebp
  ; CHECK-SSE-NEXT: pushl %ebx
  ; CHECK-SSE-NEXT; pushl %eax
  ; CHECK-SSE-NEXT: popl %eax
  ; CHECK-SSE-NEXT: popl %ebx
  ; CHECK-SSE-NEXT: popl %ebp
  ; CHECK-SSE-NEXT: addl $4, %esp
  ; CHECK-SSE-NEXT: iretl
  ; CHECK0-LABEL: test_isr_clobbers
  ; CHECK0-SSE-NEXT: pushl %ebp
  ; CHECK0-SSE-NEXT: pushl %ebx
  ; CHECK0-SSE-NEXT; pushl %eax
  ; CHECK0-SSE-NEXT: popl %eax
  ; CHECK0-SSE-NEXT: popl %ebx
  ; CHECK0-SSE-NEXT: popl %ebp
  ; CHECK0-SSE-NEXT: addl $4, %esp
  ; CHECK0-SSE-NEXT: iretl
  ret void
}

