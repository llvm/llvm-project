; RUN: llc -mtriple=x86_64-pc-windows-coreclr < %s | FileCheck %s

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
;     } catch (type2) [
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
; CHECK: .seh_endprologue
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
; CHECK: .seh_proc [[L_catch1:[^ ]+]]
  %catch1 = catchpad [i32 1]
    to label %catch1.body unwind label %catch2.pad
catch1.body:
; CHECK: leaq {{[0-9]+}}(%rcx), %rbp
;                        ^ establisher frame pointer passed in rcx
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
    to label %catch1.ret unwind label %catch.end
catch1.ret:
  catchret %catch1 to label %finally.clone
catch2.pad:
; CHECK: .seh_proc [[L_catch2:[^ ]+]]
  %catch2 = catchpad [i32 2]
    to label %catch2.body unwind label %catch.end
catch2.body:
; CHECK: leaq {{[0-9]+}}(%rcx), %rbp
;                        ^ establisher frame pointer passed in rcx
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
    to label %try_in_catch unwind label %catch.end
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
  %fault = cleanuppad [i32 undef]
; CHECK: leaq {{[0-9]+}}(%rcx), %rbp
;                        ^ establisher frame pointer passed in rcx
; CHECK: .seh_endprologue
; CHECK: [[L_before_f6:.+]]:
; CHECK-NEXT: movl $6, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[L_after_f6:.+]]:
  invoke void @f(i32 6)
    to label %fault.ret unwind label %fault.end
fault.ret:
  cleanupret %fault unwind label %catch.end
fault.end:
  cleanupendpad %fault unwind label %catch.end
catch2.ret:
  catchret %catch2 to label %finally.clone
catch.end:
  catchendpad unwind label %finally.pad
finally.clone:
  call void @f(i32 7)
  br label %tail
finally.pad:
; CHECK: .seh_proc [[L_finally:[^ ]+]]
  %finally = cleanuppad []
; CHECK: leaq {{[0-9]+}}(%rcx), %rbp
;                        ^ establisher frame pointer passed in rcx
; CHECK: .seh_endprologue
; CHECK: [[L_before_f7:.+]]:
; CHECK-NEXT: movl $7, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[L_after_f7:.+]]:
  invoke void @f(i32 7)
    to label %finally.ret unwind label %finally.end
finally.ret:
  cleanupret %finally unwind to caller
finally.end:
   cleanupendpad %finally unwind to caller
tail:
  call void @f(i32 8)
  ret void
; CHECK: [[L_end:.*func_end.*]]:
}

; Now check for EH table in xdata (following standard xdata)
; CHECK-LABEL: .section .xdata
; standard xdata comes here
; CHECK:      .long 4{{$}}
;                   ^ number of funclets
; CHECK-NEXT: .long [[L_catch1]]-[[L_begin]]
;                   ^ offset from L_begin to start of 1st funclet
; CHECK-NEXT: .long [[L_catch2]]-[[L_begin]]
;                   ^ offset from L_begin to start of 2nd funclet
; CHECK-NEXT: .long [[L_fault]]-[[L_begin]]
;                   ^ offset from L_begin to start of 3rd funclet
; CHECK-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset from L_begin to start of 4th funclet
; CHECK-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset from L_begin to end of last funclet
; CHECK-NEXT: .long 7
;                   ^ number of EH clauses
; Clause 1: call f(2) is guarded by catch1
; CHECK-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECK-NEXT: .long ([[L_before_f2]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[L_after_f2]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[L_catch1]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[L_catch2]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 1
;                   ^ type token of catch (from catchpad)
; Clause 2: call f(2) is also guarded by catch2
; CHECK-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECK-NEXT: .long ([[L_before_f2]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[L_after_f2]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[L_catch2]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[L_fault]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 2
;                   ^ type token of catch (from catchpad)
; Clause 3: calls f(1) and f(2) are guarded by finally
; CHECK-NEXT: .long 2
;                   ^ flags (2 => finally handler)
; CHECK-NEXT: .long ([[L_before_f1]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[L_after_f2]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 4: call f(3) is guarded by finally
;           This is a "duplicate" because the protected range (f(3))
;           is in funclet catch1 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECK-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECK-NEXT: .long ([[L_before_f3]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[L_after_f3]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 5: call f(5) is guarded by fault
; CHECK-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[L_before_f5]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[L_after_f5]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[L_fault]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 6: calls f(4) and f(5) are guarded by finally
;           This is a "duplicate" because the protected range (f(4)-f(5))
;           is in funclet catch2 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECK-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECK-NEXT: .long ([[L_before_f4]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[L_after_f5]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 7: call f(6) is guarded by finally
;           This is a "duplicate" because the protected range (f(3))
;           is in funclet catch1 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECK-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECK-NEXT: .long ([[L_before_f6]]-[[L_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[L_after_f6]]-[[L_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[L_finally]]-[[L_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[L_end]]-[[L_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for finally)
