; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-unknown -O0 < %s | FileCheck %s -check-prefix=CHECK0

%struct.interrupt_frame = type { i64, i64, i64, i64, i64 }

@sink_address = dso_local global i64* null
@sink_i32 = dso_local global i64 0

; Spills rax, putting original esp at +8.
; No stack adjustment if declared with no error code
define x86_intrcc void @test_isr_no_ecode(%struct.interrupt_frame* byval(%struct.interrupt_frame) %frame) {
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
define x86_intrcc void @test_isr_ecode(%struct.interrupt_frame* byval(%struct.interrupt_frame) %frame, i64 %ecode) {
  ; CHECK-LABEL: test_isr_ecode
  ; CHECK: pushq %rax
  ; CHECK: pushq %rax
  ; CHECK: pushq %rcx
  ; CHECK: movq 24(%rsp), %rax
  ; CHECK: movq 48(%rsp), %rcx
  ; CHECK: popq %rcx
  ; CHECK: popq %rax
  ; CHECK: addq $16, %rsp
  ; CHECK: iretq
  ; CHECK0-LABEL: test_isr_ecode
  ; CHECK0: pushq %rax
  ; CHECK0: pushq %rax
  ; CHECK0: pushq %rcx
  ; CHECK0: movq 24(%rsp), %rcx
  ; CHECK0: leaq 32(%rsp), %rax
  ; CHECK0: movq 16(%rax), %rax
  ; CHECK0: popq %rcx
  ; CHECK0: popq %rax
  ; CHECK0: addq $16, %rsp
  ; CHECK0: iretq
  %pflags = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %frame, i32 0, i32 2
  %flags = load i64, i64* %pflags, align 4
  call void asm sideeffect "", "r,r"(i64 %flags, i64 %ecode)
  ret void
}

; All clobbered registers must be saved
define x86_intrcc void @test_isr_clobbers(%struct.interrupt_frame* byval(%struct.interrupt_frame) %frame, i64 %ecode) {
  call void asm sideeffect "", "~{rax},~{rbx},~{rbp},~{r11},~{xmm0}"()
  ; CHECK-LABEL: test_isr_clobbers

  ; CHECK: pushq %rax
  ; CHECK: pushq %rbp
  ; CHECK: pushq %r11
  ; CHECK: pushq %rbx
  ; CHECK: movaps %xmm0
  ; CHECK: movaps {{.*}}, %xmm0
  ; CHECK: popq %rbx
  ; CHECK: popq %r11
  ; CHECK: popq %rbp
  ; CHECK: popq %rax
  ; CHECK: addq $16, %rsp
  ; CHECK: iretq
  ; CHECK0-LABEL: test_isr_clobbers

  ; CHECK0: pushq %rax
  ; CHECK0: pushq %rbp
  ; CHECK0: pushq %r11
  ; CHECK0: pushq %rbx
  ; CHECK0: movaps %xmm0
  ; CHECK0: movaps {{.*}}, %xmm0
  ; CHECK0: popq %rbx
  ; CHECK0: popq %r11
  ; CHECK0: popq %rbp
  ; CHECK0: popq %rax
  ; CHECK0: addq $16, %rsp
  ; CHECK0: iretq
  ret void
}

@f80 = common dso_local global x86_fp80 0xK00000000000000000000, align 4

; Test that the presence of x87 does not crash the FP stackifier
define x86_intrcc void @test_isr_x87(%struct.interrupt_frame* byval(%struct.interrupt_frame) %frame) {
  ; CHECK-LABEL: test_isr_x87
  ; CHECK-DAG: fldt f80
  ; CHECK-DAG: fld1
  ; CHECK: faddp
  ; CHECK-NEXT: fstpt f80
  ; CHECK-NEXT: iretq
entry:
  %ld = load x86_fp80, x86_fp80* @f80, align 4
  %add = fadd x86_fp80 %ld, 0xK3FFF8000000000000000
  store x86_fp80 %add, x86_fp80* @f80, align 4
  ret void
}

; Use a frame pointer to check the offsets. No return address, arguments start
; at RBP+4.
define dso_local x86_intrcc void @test_fp_1(%struct.interrupt_frame* byval(%struct.interrupt_frame) %p) #0 {
  ; CHECK-LABEL: test_fp_1:
  ; CHECK: # %bb.0: # %entry
  ; CHECK-NEXT: pushq %rbp
  ; CHECK-NEXT: movq %rsp, %rbp
  ; CHECK: cld
  ; CHECK-DAG: leaq 8(%rbp), %[[R1:[^ ]*]]
  ; CHECK-DAG: leaq 40(%rbp), %[[R2:[^ ]*]]
  ; CHECK: movq %[[R1]], sink_address
  ; CHECK: movq %[[R2]], sink_address
  ; CHECK: popq %rbp
  ; CHECK: iretq
entry:
  %arrayidx = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %p, i64 0, i32 0
  %arrayidx2 = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %p, i64 0, i32 4
  store volatile i64* %arrayidx, i64** @sink_address
  store volatile i64* %arrayidx2, i64** @sink_address
  ret void
}

; The error code is between RBP and the interrupt_frame.
define dso_local x86_intrcc void @test_fp_2(%struct.interrupt_frame* byval(%struct.interrupt_frame) %p, i64 %err) #0 {
  ; CHECK-LABEL: test_fp_2:
  ; CHECK: # %bb.0: # %entry
  ; This RAX push is just to align the stack.
  ; CHECK-NEXT: pushq %rax
  ; CHECK-NEXT: pushq %rbp
  ; CHECK-NEXT: movq %rsp, %rbp
  ; CHECK: cld
  ; CHECK-DAG: movq 16(%rbp), %[[R3:[^ ]*]]
  ; CHECK-DAG: leaq 24(%rbp), %[[R1:[^ ]*]]
  ; CHECK-DAG: leaq 56(%rbp), %[[R2:[^ ]*]]
  ; CHECK: movq %[[R1]], sink_address(%rip)
  ; CHECK: movq %[[R2]], sink_address(%rip)
  ; CHECK: movq %[[R3]], sink_i32(%rip)
  ; CHECK: popq %rbp
  ; Pop off both the error code and the 8 byte alignment adjustment from the
  ; prologue.
  ; CHECK: addq $16, %rsp
  ; CHECK: iretq
entry:
  %arrayidx = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %p, i64 0, i32 0
  %arrayidx2 = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %p, i64 0, i32 4
  store volatile i64* %arrayidx, i64** @sink_address
  store volatile i64* %arrayidx2, i64** @sink_address
  store volatile i64 %err, i64* @sink_i32
  ret void
}

; Test argument copy elision when copied to a local alloca.
define x86_intrcc void @test_copy_elide(%struct.interrupt_frame* byval(%struct.interrupt_frame) %frame, i64 %err) #0 {
  ; CHECK-LABEL: test_copy_elide:
  ; CHECK: # %bb.0: # %entry
  ; This RAX push is just to align the stack.
  ; CHECK-NEXT: pushq %rax
  ; CHECK-NEXT: pushq %rbp
  ; CHECK-NEXT: movq %rsp, %rbp
  ; CHECK: cld
  ; CHECK: leaq 16(%rbp), %[[R1:[^ ]*]]
  ; CHECK: movq %[[R1]], sink_address(%rip)
entry:
  %err.addr = alloca i64, align 4
  store i64 %err, i64* %err.addr, align 4
  store volatile i64* %err.addr, i64** @sink_address
  ret void
}


attributes #0 = { nounwind "frame-pointer"="all" }
