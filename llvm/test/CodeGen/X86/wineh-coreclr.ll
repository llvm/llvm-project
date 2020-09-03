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
;
; CHECK-LABEL: test1:     # @test1
; CHECK-NEXT: [[test1_begin:.*func_begin.*]]:
define void @test1() personality i8* bitcast (void ()* @ProcessCLRException to i8*) {
entry:
; CHECK: # %entry
; CHECK: leaq [[FPOffset:[0-9]+]](%rsp), %rbp
; CHECK: .seh_endprologue
; CHECK: movq %rsp, [[PSPSymOffset:[0-9]+]](%rsp)
; CHECK: [[test1_before_f1:.+]]:
; CHECK-NEXT: movl $1, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test1_after_f1:.+]]:
  invoke void @f(i32 1)
    to label %inner_try unwind label %finally
inner_try:
; CHECK: # %inner_try
; CHECK: [[test1_before_f2:.+]]:
; CHECK-NEXT: movl $2, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test1_after_f2:.+]]:
  invoke void @f(i32 2)
    to label %finally.clone unwind label %exn.dispatch
exn.dispatch:
  %catchswitch = catchswitch within none [label %catch1, label %catch2] unwind label %finally
catch1:
  %catch.pad1 = catchpad within %catchswitch [i32 1]
; CHECK: .seh_proc [[test1_catch1:[^ ]+]]
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
  %exn1 = call i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token %catch.pad1)
  call void @g(i8 addrspace(1)* %exn1) [ "funclet"(token %catch.pad1) ]
; CHECK: [[test1_before_f3:.+]]:
; CHECK-NEXT: movl $3, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test1_after_f3:.+]]:
  invoke void @f(i32 3) [ "funclet"(token %catch.pad1) ]
    to label %catch1.ret unwind label %finally
catch1.ret:
  catchret from %catch.pad1 to label %finally.clone
catch2:
  %catch.pad2 = catchpad within %catchswitch [i32 2]
; CHECK: .seh_proc [[test1_catch2:[^ ]+]]
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
  %exn2 = call i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token %catch.pad2)
  call void @g(i8 addrspace(1)* %exn2) [ "funclet"(token %catch.pad2) ]
; CHECK: [[test1_before_f4:.+]]:
; CHECK-NEXT: movl $4, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test1_after_f4:.+]]:
  invoke void @f(i32 4) [ "funclet"(token %catch.pad2) ]
    to label %try_in_catch unwind label %finally
try_in_catch:
; CHECK: # %try_in_catch
; CHECK: [[test1_before_f5:.+]]:
; CHECK-NEXT: movl $5, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test1_after_f5:.+]]:
  invoke void @f(i32 5) [ "funclet"(token %catch.pad2) ]
    to label %catch2.ret unwind label %fault
fault:
; CHECK: .seh_proc [[test1_fault:[^ ]+]]
  %fault.pad = cleanuppad within %catch.pad2 [i32 undef]
; CHECK: .seh_stackalloc [[FuncletFrameSize:[0-9]+]]
;                        ^ all funclets use the same frame size
; CHECK: movq [[PSPSymOffset]](%rcx), %rcx
;                              ^ establisher frame pointer passed in rcx
; CHECK: movq %rcx, [[PSPSymOffset]](%rsp)
; CHECK: leaq [[FPOffset]](%rcx), %rbp
; CHECK: .seh_endprologue
; CHECK: [[test1_before_f6:.+]]:
; CHECK-NEXT: movl $6, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test1_after_f6:.+]]:
  invoke void @f(i32 6) [ "funclet"(token %fault.pad) ]
    to label %fault.ret unwind label %finally
fault.ret:
  cleanupret from %fault.pad unwind label %finally
catch2.ret:
  catchret from %catch.pad2 to label %finally.clone
finally.clone:
  call void @f(i32 7)
  br label %tail
finally:
; CHECK: .seh_proc [[test1_finally:[^ ]+]]
  %finally.pad = cleanuppad within none []
; CHECK: .seh_stackalloc [[FuncletFrameSize:[0-9]+]]
;                        ^ all funclets use the same frame size
; CHECK: movq [[PSPSymOffset]](%rcx), %rcx
;                              ^ establisher frame pointer passed in rcx
; CHECK: movq %rcx, [[PSPSymOffset]](%rsp)
; CHECK: leaq [[FPOffset]](%rcx), %rbp
; CHECK: .seh_endprologue
; CHECK-NEXT: movl $7, %ecx
; CHECK-NEXT: callq f
  call void @f(i32 7) [ "funclet"(token %finally.pad) ]
  cleanupret from %finally.pad unwind to caller
tail:
  call void @f(i32 8)
  ret void
; CHECK: [[test1_end:.*func_end.*]]:
}

; Now check for EH table in xdata (following standard xdata)
; CHECK-LABEL: .section .xdata
; standard xdata comes here
; CHECK:      .long 4{{$}}
;                   ^ number of funclets
; CHECK-NEXT: .long [[test1_catch1]]-[[test1_begin]]
;                   ^ offset from L_begin to start of 1st funclet
; CHECK-NEXT: .long [[test1_catch2]]-[[test1_begin]]
;                   ^ offset from L_begin to start of 2nd funclet
; CHECK-NEXT: .long [[test1_fault]]-[[test1_begin]]
;                   ^ offset from L_begin to start of 3rd funclet
; CHECK-NEXT: .long [[test1_finally]]-[[test1_begin]]
;                   ^ offset from L_begin to start of 4th funclet
; CHECK-NEXT: .long [[test1_end]]-[[test1_begin]]
;                   ^ offset from L_begin to end of last funclet
; CHECK-NEXT: .long 7
;                   ^ number of EH clauses
; Clause 1: call f(2) is guarded by catch1
; CHECK-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECK-NEXT: .long ([[test1_before_f2]]-[[test1_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test1_after_f2]]-[[test1_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test1_catch1]]-[[test1_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test1_catch2]]-[[test1_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 1
;                   ^ type token of catch (from catchpad)
; Clause 2: call f(2) is also guarded by catch2
; CHECK-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECK-NEXT: .long ([[test1_before_f2]]-[[test1_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test1_after_f2]]-[[test1_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test1_catch2]]-[[test1_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test1_fault]]-[[test1_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 2
;                   ^ type token of catch (from catchpad)
; Clause 3: calls f(1) and f(2) are guarded by finally
; CHECK-NEXT: .long 2
;                   ^ flags (2 => finally handler)
; CHECK-NEXT: .long ([[test1_before_f1]]-[[test1_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test1_after_f2]]-[[test1_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test1_finally]]-[[test1_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test1_end]]-[[test1_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 4: call f(3) is guarded by finally
;           This is a "duplicate" because the protected range (f(3))
;           is in funclet catch1 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECK-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECK-NEXT: .long ([[test1_before_f3]]-[[test1_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test1_after_f3]]-[[test1_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test1_finally]]-[[test1_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test1_end]]-[[test1_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 5: call f(5) is guarded by fault
; CHECK-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[test1_before_f5]]-[[test1_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test1_after_f5]]-[[test1_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test1_fault]]-[[test1_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test1_finally]]-[[test1_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 6: calls f(4) and f(5) are guarded by finally
;           This is a "duplicate" because the protected range (f(4)-f(5))
;           is in funclet catch2 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECK-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECK-NEXT: .long ([[test1_before_f4]]-[[test1_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test1_after_f5]]-[[test1_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test1_finally]]-[[test1_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test1_end]]-[[test1_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for finally)
; Clause 7: call f(6) is guarded by finally
;           This is a "duplicate" because the protected range (f(3))
;           is in funclet catch1 but the finally's immediate parent
;           is the main function, not that funclet.
; CHECK-NEXT: .long 10
;                   ^ flags (2 => finally handler | 8 => duplicate)
; CHECK-NEXT: .long ([[test1_before_f6]]-[[test1_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test1_after_f6]]-[[test1_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test1_finally]]-[[test1_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test1_end]]-[[test1_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for finally)

; Test with a cleanup that has no cleanupret, and thus needs its unwind dest
; inferred from an inner catchswitch
;
; corresponds to C# along the lines of:
; void test2() {
;   try {
;     try {
;       f(1);
;     } fault {
;       try {
;         f(2);
;       } catch(type1) {
;       }
;       __unreachable();
;     }
;   } catch(type2) {
;   }
; }
;
define void @test2() personality i8* bitcast (void ()* @ProcessCLRException to i8*) {
entry:
  invoke void @f(i32 1)
    to label %exit unwind label %fault
fault:
  %fault.pad = cleanuppad within none [i32 undef]
  invoke void @f(i32 2) ["funclet"(token %fault.pad)]
    to label %unreachable unwind label %exn.dispatch.inner
exn.dispatch.inner:
  %catchswitch.inner = catchswitch within %fault.pad [label %catch1] unwind label %exn.dispatch.outer
catch1:
  %catch.pad1 = catchpad within %catchswitch.inner [i32 1]
  catchret from %catch.pad1 to label %unreachable
exn.dispatch.outer:
  %catchswitch.outer = catchswitch within none [label %catch2] unwind to caller
catch2:
  %catch.pad2 = catchpad within %catchswitch.outer [i32 2]
  catchret from %catch.pad2 to label %exit
exit:
  ret void
unreachable:
  unreachable
}
; CHECK-LABEL: test2:     # @test2
; CHECK-NEXT: [[test2_begin:.*func_begin.*]]:
; CHECK: .seh_endprologue
; CHECK: [[test2_before_f1:.+]]:
; CHECK-NEXT: movl $1, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test2_after_f1:.+]]:
; CHECK: .seh_proc [[test2_catch1:[^ ]+]]
; CHECK: .seh_proc [[test2_catch2:[^ ]+]]
; CHECK: .seh_proc [[test2_fault:[^ ]+]]
; CHECK: .seh_endprologue
; CHECK: [[test2_before_f2:.+]]:
; CHECK-NEXT: movl $2, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test2_after_f2:.+]]:
; CHECK: int3
; CHECK: [[test2_end:.*func_end.*]]:


; Now check for EH table in xdata (following standard xdata)
; CHECK-LABEL: .section .xdata
; standard xdata comes here
; CHECK:      .long 3{{$}}
;                   ^ number of funclets
; CHECK-NEXT: .long [[test2_catch1]]-[[test2_begin]]
;                   ^ offset from L_begin to start of 2nd funclet
; CHECK-NEXT: .long [[test2_catch2]]-[[test2_begin]]
;                   ^ offset from L_begin to start of 3rd funclet
; CHECK-NEXT: .long [[test2_fault]]-[[test2_begin]]
;                   ^ offset from L_begin to start of 1st funclet
; CHECK-NEXT: .long [[test2_end]]-[[test2_begin]]
;                   ^ offset from L_begin to end of last funclet
; CHECK-NEXT: .long 4
;                   ^ number of EH clauses
; Clause 1: call f(1) is guarded by fault
; CHECK-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[test2_before_f1]]-[[test2_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test2_after_f1]]-[[test2_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test2_fault]]-[[test2_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test2_end]]-[[test2_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 2: call f(1) is also guarded by catch2
; CHECK-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECK-NEXT: .long ([[test2_before_f1]]-[[test2_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test2_after_f1]]-[[test2_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test2_catch2]]-[[test2_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test2_fault]]-[[test2_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 2
;                   ^ type token of catch (from catchpad)
; Clause 3: calls f(2) is guarded by catch1
; CHECK-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECK-NEXT: .long ([[test2_before_f2]]-[[test2_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test2_after_f2]]-[[test2_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test2_catch1]]-[[test2_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test2_catch2]]-[[test2_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 1
;                   ^ type token of catch (from catchpad)
; Clause 4: call f(2) is also guarded by catch2
;           This is a "duplicate" because the protected range (f(2))
;           is in funclet fault but catch2's immediate parent
;           is the main function, not that funclet.
; CHECK-NEXT: .long 8
;                   ^ flags (0 => catch handler | 8 => duplicate)
; CHECK-NEXT: .long ([[test2_before_f2]]-[[test2_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test2_after_f2]]-[[test2_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test2_catch2]]-[[test2_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test2_fault]]-[[test2_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 2
;                   ^ type token of catch (from catchpad)

; Test with several cleanups that need to infer their unwind dests from each
; other, the inner one needing to make the inference from an invoke, ignoring
; not-really-unwinding calls/unwind-to-caller catchswitches, as well as some
; internal invokes/catchswitches
;
; Corresponds to something like:
; void test3() {
;   try {
;     f(1);
;   } fault { // fault1
;     try {
;       try {
;         f(2);
;         __unreachable();
;       } fault { // fault2
;         try {
;           f(3);
;         } fault { // fault3
;           try {
;             f(4);
;           } fault { // fault4
;             f(5); // no unwind edge (e.g. front-end knew it wouldn't throw but
;                    didn't bother to specify nounwind)
;             try {
;               try {
;                 f(6);
;               } catch(type 1) {
;                 goto __unreachable;
;               }
;             } catch (type 2) { // marked "unwinds to caller" because we allow
;                                // that if the unwind won't be taken (see
;                                // SimplifyUnreachable & RemoveUnwindEdge)
;               goto _unreachable;
;             }
;             f(7);
;             __unreachable();
;           }
;         }
;       }
;     } fault { // fault 5
;     }
;   }
; }
;
; CHECK-LABEL: test3:     # @test3
; CHECK-NEXT: [[test3_begin:.*func_begin.*]]:
define void @test3() personality i8* bitcast (void ()* @ProcessCLRException to i8*) {
entry:
; CHECK: .seh_endprologue
; CHECK: [[test3_before_f1:.+]]:
; CHECK-NEXT: movl $1, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test3_after_f1:.+]]:
  invoke void @f(i32 1)
    to label %exit unwind label %fault1
fault1:
 ; check lines below since this gets reordered to end-of-func
  %fault.pad1 = cleanuppad within none [i32 undef]
  invoke void @f(i32 2) ["funclet"(token %fault.pad1)]
    to label %unreachable unwind label %fault2
fault2:
 ; check lines below since this gets reordered to end-of-func
  %fault.pad2 = cleanuppad within %fault.pad1 [i32 undef]
  invoke void @f(i32 3) ["funclet"(token %fault.pad2)]
    to label %unreachable unwind label %fault3
fault3:
 ; check lines below since this gets reordered to end-of-func
  %fault.pad3 = cleanuppad within %fault.pad2 [i32 undef]
  invoke void @f(i32 4) ["funclet"(token %fault.pad3)]
    to label %unreachable unwind label %fault4
fault4:
; CHECK: .seh_proc [[test3_fault4:[^ ]+]]
  %fault.pad4 = cleanuppad within %fault.pad3 [i32 undef]
; CHECK: .seh_endprologue
  call void @f(i32 5) ["funclet"(token %fault.pad4)]
; CHECK: [[test3_before_f6:.+]]:
; CHECK-NEXT: movl $6, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test3_after_f6:.+]]:
  invoke void @f(i32 6) ["funclet"(token %fault.pad4)]
    to label %fault4.cont unwind label %exn.dispatch1
fault4.cont:
; CHECK: # %fault4.cont
; CHECK: [[test3_before_f7:.+]]:
; CHECK-NEXT: movl $7, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test3_after_f7:.+]]:
  invoke void @f(i32 7) ["funclet"(token %fault.pad4)]
    to label %unreachable unwind label %fault5
exn.dispatch1:
  %catchswitch1 = catchswitch within %fault.pad4 [label %catch1] unwind label %exn.dispatch2
catch1:
  %catch.pad1 = catchpad within %catchswitch1 [i32 1]
; CHECK: .seh_proc [[test3_catch1:[^ ]+]]
  catchret from %catch.pad1 to label %unreachable
exn.dispatch2:
  %catchswitch2 = catchswitch within %fault.pad4 [label %catch2] unwind to caller
catch2:
  %catch.pad2 = catchpad within %catchswitch2 [i32 2]
; CHECK: .seh_proc [[test3_catch2:[^ ]+]]
  catchret from %catch.pad2 to label %unreachable
fault5:
; CHECK: .seh_proc [[test3_fault5:[^ ]+]]
  %fault.pad5 = cleanuppad within %fault.pad1 [i32 undef]
; CHECK: .seh_endprologue
cleanupret from %fault.pad5 unwind to caller
exit:
  ret void
unreachable:
  unreachable
; CHECK: .seh_proc [[test3_fault3:[^ ]+]]
; CHECK: # %fault3
; CHECK: .seh_endprologue
; CHECK: [[test3_before_f4:.+]]:
; CHECK-NEXT: movl $4, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test3_after_f4:.+]]:
; CHECK: int3
; CHECK: .seh_proc [[test3_fault2:[^ ]+]]
; CHECK: # %fault2
; CHECK: .seh_endprologue
; CHECK: [[test3_before_f3:.+]]:
; CHECK-NEXT: movl $3, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test3_after_f3:.+]]:
; CHECK: int3
; CHECK: .seh_proc [[test3_fault1:[^ ]+]]
; CHECK: # %fault1
; CHECK: .seh_endprologue
; CHECK: [[test3_before_f2:.+]]:
; CHECK-NEXT: movl $2, %ecx
; CHECK-NEXT: callq f
; CHECK-NEXT: [[test3_after_f2:.+]]:
; CHECK: int3
; CHECK: [[test3_end:.*func_end.*]]:
}

; Now check for EH table in xdata (following standard xdata)
; CHECK-LABEL: .section .xdata
; standard xdata comes here
; CHECK:      .long 7{{$}}
;                   ^ number of funclets
; CHECK-NEXT: .long [[test3_fault4]]-[[test3_begin]]
;                   ^ offset from L_begin to start of 1st funclet
; CHECK-NEXT: .long [[test3_catch1]]-[[test3_begin]]
;                   ^ offset from L_begin to start of 2nd funclet
; CHECK-NEXT: .long [[test3_catch2]]-[[test3_begin]]
;                   ^ offset from L_begin to start of 3rd funclet
; CHECK-NEXT: .long [[test3_fault5]]-[[test3_begin]]
;                   ^ offset from L_begin to start of 4th funclet
; CHECK-NEXT: .long [[test3_fault3]]-[[test3_begin]]
;                   ^ offset from L_begin to start of 5th funclet
; CHECK-NEXT: .long [[test3_fault2]]-[[test3_begin]]
;                   ^ offset from L_begin to start of 6th funclet
; CHECK-NEXT: .long [[test3_fault1]]-[[test3_begin]]
;                   ^ offset from L_begin to start of 7th funclet
; CHECK-NEXT: .long [[test3_end]]-[[test3_begin]]
;                   ^ offset from L_begin to end of last funclet
; CHECK-NEXT: .long 10
;                   ^ number of EH clauses
; Clause 1: call f(1) is guarded by fault1
; CHECK-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[test3_before_f1]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f1]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_fault1]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_end]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 3: call f(6) is guarded by catch1
; CHECK-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECK-NEXT: .long ([[test3_before_f6]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f6]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_catch1]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_catch2]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 1
;                   ^ type token of catch (from catchpad)
; Clause 3: call f(6) is also guarded by catch2
; CHECK-NEXT: .long 0
;                   ^ flags (0 => catch handler)
; CHECK-NEXT: .long ([[test3_before_f6]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f6]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_catch2]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_fault5]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 2
;                   ^ type token of catch (from catchpad)
; Clause 4: call f(7) is guarded by fault5
;           This is a "duplicate" because the protected range (f(6)-f(7))
;           is in funclet fault4 but fault5's immediate parent
;           is fault1, not that funclet.
; CHECK-NEXT: .long 12
;                   ^ flags (4 => fault handler | 8 => duplicate)
; CHECK-NEXT: .long ([[test3_before_f7]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f7]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_fault5]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_fault3]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 5: call f(4) is guarded by fault4
; CHECK-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[test3_before_f4]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f4]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_fault4]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_catch1]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 6: call f(4) is also guarded by fault5
;           This is a "duplicate" because the protected range (f(4))
;           is in funclet fault3 but fault5's immediate parent
;           is fault1, not that funclet.
; CHECK-NEXT: .long 12
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[test3_before_f4]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f4]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_fault5]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_fault3]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 7: call f(3) is guarded by fault3
; CHECK-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[test3_before_f3]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f3]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_fault3]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_fault2]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 8: call f(3) is guarded by fault5
;           This is a "duplicate" because the protected range (f(3))
;           is in funclet fault2 but fault5's immediate parent
;           is fault1, not that funclet.
; CHECK-NEXT: .long 12
;                   ^ flags (4 => fault handler | 8 => duplicate)
; CHECK-NEXT: .long ([[test3_before_f3]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f3]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_fault5]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_fault3]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 9: call f(2) is guarded by fault2
; CHECK-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[test3_before_f2]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f2]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_fault2]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_fault1]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
; Clause 10: call f(2) is guarded by fault5
; CHECK-NEXT: .long 4
;                   ^ flags (4 => fault handler)
; CHECK-NEXT: .long ([[test3_before_f2]]-[[test3_begin]])+1
;                   ^ offset of start of clause
; CHECK-NEXT: .long ([[test3_after_f2]]-[[test3_begin]])+1
;                   ^ offset of end of clause
; CHECK-NEXT: .long [[test3_fault5]]-[[test3_begin]]
;                   ^ offset of start of handler
; CHECK-NEXT: .long [[test3_fault3]]-[[test3_begin]]
;                   ^ offset of end of handler
; CHECK-NEXT: .long 0
;                   ^ type token slot (null for fault)
