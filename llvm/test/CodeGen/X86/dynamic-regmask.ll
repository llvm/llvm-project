; RUN: llc -mtriple=x86_64-pc-win32 -stop-after machine-sink %s -o %t.mir
; RUN: FileCheck %s < %t.mir
; RUN: llc %t.mir -mtriple=x86_64-pc-win32 -run-pass machine-sink
; Check that callee saved registers are printed in a format that can then be parsed.

declare x86_regcallcc i32 @callee(i32 %a0, i32 %b0, i32 %c0, i32 %d0, i32 %e0)

define i32 @caller(i32 %a0) nounwind {
  %b1 = call x86_regcallcc i32 @callee(i32 %a0, i32 %a0, i32 %a0, i32 %a0, i32 %a0)
  %b2 = add i32 %b1, %a0
  ret i32 %b2
}
; CHECK:    name: caller
; CHECK:    CALL64pcrel32 @callee, CustomRegMask(%bh,%bl,%bp,%bpl,%bx,%ebp,%ebx,%esp,%rbp,%rbx,%rsp,%sp,%spl,%r10,%r11,%r12,%r13,%r14,%r15,%xmm8,%xmm9,%xmm10,%xmm11,%xmm12,%xmm13,%xmm14,%xmm15,%r10b,%r11b,%r12b,%r13b,%r14b,%r15b,%r10d,%r11d,%r12d,%r13d,%r14d,%r15d,%r10w,%r11w,%r12w,%r13w,%r14w,%r15w)
; CHECK:    RET 0, %eax

define x86_regcallcc {i32, i32, i32} @test_callee(i32 %a0, i32 %b0, i32 %c0, i32 %d0, i32 %e0) nounwind {
  %b1 = mul i32 7, %e0
  %b2 = udiv i32 5, %e0
  %b3 = mul i32 7, %d0
  %b4 = insertvalue {i32, i32, i32} undef, i32 %b1, 0
  %b5 = insertvalue {i32, i32, i32} %b4, i32 %b2, 1
  %b6 = insertvalue {i32, i32, i32} %b5, i32 %b3, 2
  ret {i32, i32, i32} %b6
}
; CHECK: name:            test_callee
; CHECK: calleeSavedRegisters: [ '%rbx', '%rbp', '%rsp', '%r10', '%r11', '%r12',
; CHECK:                         '%r13', '%r14', '%r15', '%xmm8', '%xmm9', '%xmm10',
; CHECK:                         '%xmm11', '%xmm12', '%xmm13', '%xmm14', '%xmm15' ]
; CHECK: RET 0, %eax, %ecx, %edx
