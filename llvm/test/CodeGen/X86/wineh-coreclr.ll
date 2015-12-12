; RUN: llc -mtriple=x86_64-pc-windows-coreclr -verify-machineinstrs < %s | FileCheck %s

declare void @ProcessCLRException()
declare void @f(i32)
declare void @g(i8 addrspace(1)*)
declare i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token)

; Simplified IR for pseudo-C# like the following:
; void test1() {
;   try {
;     f(1);
;     try {
;       f(2);
;     } catch (type1) {
;       f(3);
;     } catch (type2) {
;       f(4);
;       try {
;         f(5);
;       } fault {
;         f(6);
;       }
;     }
;   } finally {
;     f(7);
;   }
;   f(8);
; }

; CHECK-LABEL: test1:     # @test1
; CHECK-NEXT: [[L_begin:.*func_begin.*]]:
define void @test1() personality i8* bitcast (void ()* @ProcessCLRException to i8*) {
entry:
; CHECK: # %entry
; CHECK: leaq [[FPOffset:[0-9]+]](%rsp), %rbp
; CHECK: .seh_endprologue
; CHECK: movq %rsp, [[PSPSymOffset:[0-9]+]](%rsp)
; CHECK: [[L_before_f1:.+]]:
; CHECK-NEXT: movl $1, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[L_after_f1:.+]]:
  invoke void @f(i32 1)
    to label %inner_try unwind label %finally.pad
inner_try:
; CHECK: # %inner_try
; CHECK: [[L_before_f2:.+]]:
; CHECK-NEXT: movl $2, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[L_after_f2:.+]]:
  invoke void @f(i32 2)
    to label %finally.clone unwind label %catch1.pad
catch1.pad:
  %cs1 = catchswitch within none [label %catch1.body, label %catch2.body] unwind label %finally.pad
catch1.body:
  %catch1 = catchpad within %cs1 [i32 1]
; CHECK: .seh_proc [[L_catch1:[^ ]+]]
; CHECK: .seh_stackalloc [[FuncletFrameSize:[0-9]+]]
;                        ^ all funclets use the same frame size
; CHECK: movq [[PSPSymOffset]](%rcx), %rcx
;                              ^ establisher frame pointer passed in rcx
; CHECK: movq %rcx, [[PSPSymOffset]](%rsp)
; CHECK: leaq [[FPOffset]](%rcx), %rbp
; CHECK: .seh_endprologue
; CHECK: movq %rdx, %rcx
;             ^ exception pointer passed in rdx
; CHECK-NEXT: callq g
  %exn1 = call i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token %catch1)
  call void @g(i8 addrspace(1)* %exn1)
; CHECK: [[L_before_f3:.+]]:
; CHECK-NEXT: movl $3, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[L_after_f3:.+]]:
  invoke void @f(i32 3)
    to label %catch1.ret unwind label %finally.pad
catch1.ret:
  catchret from %catch1 to label %finally.clone
catch2.body:
  %catch2 = catchpad within %cs1 [i32 2]
; CHECK: .seh_proc [[L_catch2:[^ ]+]]
; CHECK: .seh_stackalloc [[FuncletFrameSize:[0-9]+]]
;                        ^ all funclets use the same frame size
; CHECK: movq [[PSPSymOffset]](%rcx), %rcx
;                              ^ establisher frame pointer passed in rcx
; CHECK: movq %rcx, [[PSPSymOffset]](%rsp)
; CHECK: leaq [[FPOffset]](%rcx), %rbp
; CHECK: .seh_endprologue
; CHECK: movq %rdx, %rcx
;             ^ exception pointer passed in rdx
; CHECK-NEXT: callq g
  %exn2 = call i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token %catch2)
  call void @g(i8 addrspace(1)* %exn2)
; CHECK: [[L_before_f4:.+]]:
; CHECK-NEXT: movl $4, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[L_after_f4:.+]]:
  invoke void @f(i32 4)
    to label %try_in_catch unwind label %finally.pad
try_in_catch:
; CHECK: # %try_in_catch
; CHECK: [[L_before_f5:.+]]:
; CHECK-NEXT: movl $5, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[L_after_f5:.+]]:
  invoke void @f(i32 5)
    to label %catch2.ret unwind label %fault.pad
fault.pad:
; CHECK: .seh_proc [[L_fault:[^ ]+]]
  %fault = cleanuppad within none [i32 undef]
; CHECK: .seh_stackalloc [[FuncletFrameSize:[0-9]+]]
;                        ^ all funclets use the same frame size
; CHECK: movq [[PSPSymOffset]](%rcx), %rcx
;                              ^ establisher frame pointer passed in rcx
; CHECK: movq %rcx, [[PSPSymOffset]](%rsp)
; CHECK: leaq [[FPOffset]](%rcx), %rbp
; CHECK: .seh_endprologue
; CHECK: [[L_before_f6:.+]]:
; CHECK-NEXT: movl $6, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[L_after_f6:.+]]:
  invoke void @f(i32 6)
    to label %fault.ret unwind label %finally.pad
fault.ret:
  cleanupret from %fault unwind label %finally.pad
catch2.ret:
  catchret from %catch2 to label %finally.clone
finally.clone:
  call void @f(i32 7)
  br label %tail
finally.pad:
; CHECK: .seh_proc [[L_finally:[^ ]+]]
  %finally = cleanuppad within none []
; CHECK: .seh_stackalloc [[FuncletFrameSize:[0-9]+]]
;                        ^ all funclets use the same frame size
; CHECK: movq [[PSPSymOffset]](%rcx), %rcx
;                              ^ establisher frame pointer passed in rcx
; CHECK: movq %rcx, [[PSPSymOffset]](%rsp)
; CHECK: leaq [[FPOffset]](%rcx), %rbp
; CHECK: .seh_endprologue
; CHECK-NEXT: movl $7, %ecx
; CHECK-NEXT: callq f
  call void @f(i32 7)
  cleanupret from %finally unwind to caller
tail:
  call void @f(i32 8)
  ret void
; CHECK: [[L_end:.*func_end.*]]:
}

; FIXME: Verify that the new clauses are correct and re-enable these checks.

; Now check for EH table in xdata (following standard xdata)
; CHECKX-LABEL: .section .xdata
; standard xdata comes here
; CHECKX:      .long 4{{$}}
;                   ^ number of funclets
; CHECKX-NEXT: .long [[L_catch1]]-[[L_begin]]
;                   ^ offset from L_begin to start of 1st funclet
; CHECKX-NEXT: .long [[L_catch2]]-[[L_begin]]
;                   ^ offset from L_begin to start of 2nd funclet
; CHECKX-NEXT: .long [[L_fault]]-[[L_begin]]
;                   ^ offset from L_begin to start of 3rd funclet
; CHECKX-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset from L_begin to start of 4th funclet
; CHECKX-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset from L_begin to end of last funclet
; CHECKX-NEXT: .long 7
;                   ^ number of EH clauses
; Clause 1: call f(2) is guarded by catch1
; CHECKX-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECKX-NEXT: .long ([[L_before_f2]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECKX-NEXT: .long ([[L_after_f2]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECKX-NEXT: .long [[L_catch1]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECKX-NEXT: .long [[L_catch2]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECKX-NEXT: .long 1
;                   ^ type token of catch (from catchpad)
; Clause 2: call f(2) is also guarded by catch2
; CHECKX-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECKX-NEXT: .long ([[L_before_f2]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECKX-NEXT: .long ([[L_after_f2]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECKX-NEXT: .long [[L_catch2]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECKX-NEXT: .long [[L_fault]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECKX-NEXT: .long 2
;                   ^ type token of catch (from catchpad)
; Clause 3: calls f(1) and f(2) are guarded by finally
; CHECKX-NEXT: .long 2
;                   ^ flags (2 => finally handler)
; CHECKX-NEXT: .long ([[L_before_f1]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECKX-NEXT: .long ([[L_after_f2]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECKX-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECKX-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECKX-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 4: call f(3) is guarded by finally
;           This is a "duplicate" because the protected range (f(3))
;           is in funclet catch1 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECKX-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECKX-NEXT: .long ([[L_before_f3]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECKX-NEXT: .long ([[L_after_f3]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECKX-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECKX-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECKX-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 5: call f(5) is guarded by fault
; CHECKX-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECKX-NEXT: .long ([[L_before_f5]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECKX-NEXT: .long ([[L_after_f5]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECKX-NEXT: .long [[L_fault]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECKX-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECKX-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 6: calls f(4) and f(5) are guarded by finally
;           This is a "duplicate" because the protected range (f(4)-f(5))
;           is in funclet catch2 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECKX-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECKX-NEXT: .long ([[L_before_f4]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECKX-NEXT: .long ([[L_after_f5]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECKX-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECKX-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECKX-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 7: call f(6) is guarded by finally
;           This is a "duplicate" because the protected range (f(3))
;           is in funclet catch1 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECKX-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECKX-NEXT: .long ([[L_before_f6]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECKX-NEXT: .long ([[L_after_f6]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECKX-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECKX-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECKX-NEXT: .long 0
;                   ^ type token slot (null for finally)
