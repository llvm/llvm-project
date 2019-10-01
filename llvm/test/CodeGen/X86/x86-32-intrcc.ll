; RUN: llc -mtriple=i686-unknown-unknown < %s | FileCheck %s
; RUN: llc -mtriple=i686-unknown-unknown -O0 < %s | FileCheck %s -check-prefix=CHECK0

%struct.interrupt_frame = type { i32, i32, i32, i32, i32 }

@sink_address = global i32* null
@sink_i32 = global i32 0


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
  ; CHECK: pushl %ebp
  ; CHECK: pushl %ebx
  ; CHECK: pushl %eax
  ; CHECK: popl %eax
  ; CHECK: popl %ebx
  ; CHECK: popl %ebp
  ; CHECK: addl $4, %esp
  ; CHECK: iretl
  ; CHECK0-LABEL: test_isr_clobbers
  ; CHECK0: pushl %ebp
  ; CHECK0: pushl %ebx
  ; CHECK0: pushl %eax
  ; CHECK0: popl %eax
  ; CHECK0: popl %ebx
  ; CHECK0: popl %ebp
  ; CHECK0: addl $4, %esp
  ; CHECK0: iretl
  ret void
}

@f80 = common global x86_fp80 0xK00000000000000000000, align 4

; Test that the presence of x87 does not crash the FP stackifier
define x86_intrcc void @test_isr_x87(%struct.interrupt_frame* %frame) {
  ; CHECK-LABEL: test_isr_x87
  ; CHECK-DAG: fldt f80
  ; CHECK-DAG: fld1
  ; CHECK: faddp
  ; CHECK-NEXT: fstpt f80
  ; CHECK-NEXT: iretl
entry:
  %ld = load x86_fp80, x86_fp80* @f80, align 4
  %add = fadd x86_fp80 %ld, 0xK3FFF8000000000000000
  store x86_fp80 %add, x86_fp80* @f80, align 4
  ret void
}

; Use a frame pointer to check the offsets. No return address, arguments start
; at EBP+4.
define dso_local x86_intrcc void @test_fp_1(%struct.interrupt_frame* %p) #0 {
  ; CHECK-LABEL: test_fp_1:
  ; CHECK: # %bb.0: # %entry
  ; CHECK-NEXT: pushl %ebp
  ; CHECK-NEXT: movl %esp, %ebp
  ; CHECK: cld
  ; CHECK-DAG: leal 4(%ebp), %[[R1:[^ ]*]]
  ; CHECK-DAG: leal 20(%ebp), %[[R2:[^ ]*]]
  ; CHECK: movl %[[R1]], sink_address
  ; CHECK: movl %[[R2]], sink_address
  ; CHECK: popl %ebp
  ; CHECK: iretl
entry:
  %arrayidx = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %p, i32 0, i32 0
  %arrayidx2 = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %p, i32 0, i32 4
  store volatile i32* %arrayidx, i32** @sink_address
  store volatile i32* %arrayidx2, i32** @sink_address
  ret void
}

; The error code is between EBP and the interrupt_frame.
define dso_local x86_intrcc void @test_fp_2(%struct.interrupt_frame* %p, i32 %err) #0 {
  ; CHECK-LABEL: test_fp_2:
  ; CHECK: # %bb.0: # %entry
  ; CHECK-NEXT: pushl %ebp
  ; CHECK-NEXT: movl %esp, %ebp
  ; CHECK: cld
  ; CHECK-DAG: movl 4(%ebp), %[[R3:[^ ]*]]
  ; CHECK-DAG: leal 8(%ebp), %[[R1:[^ ]*]]
  ; CHECK-DAG: leal 24(%ebp), %[[R2:[^ ]*]]
  ; CHECK: movl %[[R1]], sink_address
  ; CHECK: movl %[[R2]], sink_address
  ; CHECK: movl %[[R3]], sink_i32
  ; CHECK: popl %ebp
  ; CHECK: iretl
entry:
  %arrayidx = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %p, i32 0, i32 0
  %arrayidx2 = getelementptr inbounds %struct.interrupt_frame, %struct.interrupt_frame* %p, i32 0, i32 4
  store volatile i32* %arrayidx, i32** @sink_address
  store volatile i32* %arrayidx2, i32** @sink_address
  store volatile i32 %err, i32* @sink_i32
  ret void
}

; Test argument copy elision when copied to a local alloca.
define x86_intrcc void @test_copy_elide(%struct.interrupt_frame* %frame, i32 %err) #0 {
  ; CHECK-LABEL: test_copy_elide:
  ; CHECK: # %bb.0: # %entry
  ; CHECK-NEXT: pushl %ebp
  ; CHECK-NEXT: movl %esp, %ebp
  ; CHECK: cld
  ; CHECK: leal 4(%ebp), %[[R1:[^ ]*]]
  ; CHECK: movl %[[R1]], sink_address
entry:
  %err.addr = alloca i32, align 4
  store i32 %err, i32* %err.addr, align 4
  store volatile i32* %err.addr, i32** @sink_address
  ret void
}

attributes #0 = { nounwind "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
