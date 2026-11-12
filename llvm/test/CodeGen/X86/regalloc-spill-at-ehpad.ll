; RUN: llc -regalloc=greedy -mtriple=x86_64-pc-windows-msvc  < %s -o - | FileCheck %s

; This test checks for proper handling of a condition where the greedy register
; allocator encounters a very short interval that contains no uses but does
; contain an EH pad unwind edge, which requires spilling.  Previously the
; register allocator marked a interval like this as unspillable, resulting in
; a compilation failure.


; The following checks that the value %p is reloaded within the catch handler.
; CHECK-LABEL: "?catch$8@?0?test@4HA":
; CHECK:           .seh_endprologue
; CHECK:           movq    -16(%rbp), %rax
; CHECK:           movb    $0, (%rax)

define ptr @test(ptr %a) personality ptr @__CxxFrameHandler3 {
entry:
  %call = call i32 @f()
  br i1 undef, label %if.end, label %if.else

if.else:                                          ; preds = %entry
  br i1 undef, label %cond.false.i, label %if.else.else

if.else.else:                                     ; preds = %if.else
  br i1 undef, label %cond.true.i, label %cond.false.i

cond.true.i:                                      ; preds = %if.else.else
  br label %invoke.cont

cond.false.i:                                     ; preds = %if.else.else, %if.else
  %call.i = invoke i32 @f()
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %cond.false.i
  %tmp0 = catchswitch within none [label %catch] unwind label %ehcleanup

catch:                                            ; preds = %catch.dispatch
  %tmp1 = catchpad within %tmp0 [ptr null, i32 64, ptr null]
  store i8 0, ptr %a, align 8
  invoke void @_CxxThrowException(ptr null, ptr null) [ "funclet"(token %tmp1) ]
          to label %noexc unwind label %ehcleanup

noexc:                                            ; preds = %catch
  unreachable

invoke.cont:                                      ; preds = %cond.false.i, %cond.true.i
  %cond.i = phi i32 [ %call, %cond.true.i ], [ %call.i, %cond.false.i ]
  %cmp = icmp eq i32 %cond.i, -1
  %tmp3 = select i1 %cmp, i32 4, i32 0
  br label %if.end

if.end:                                           ; preds = %invoke.cont, %entry
  %state.0 = phi i32 [ %tmp3, %invoke.cont ], [ 4, %entry ]
  invoke void @g(ptr %a, i32 %state.0)
          to label %invoke.cont.1 unwind label %ehcleanup

invoke.cont.1:                                    ; preds = %if.end
  ret ptr %a

ehcleanup:                                        ; preds = %if.end, %catch, %catch.dispatch
  %tmp4 = cleanuppad within none []
  cleanupret from %tmp4 unwind to caller
}

%eh.ThrowInfo = type { i32, i32, i32, i32 }

declare i32 @__CxxFrameHandler3(...)

declare void @_CxxThrowException(ptr, ptr)

declare i32 @f()
declare void @g(ptr, i32)
