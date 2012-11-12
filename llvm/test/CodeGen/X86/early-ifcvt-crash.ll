; RUN: llc < %s -x86-early-ifcvt -verify-machineinstrs
; RUN: llc < %s -x86-early-ifcvt -stress-early-ifcvt -verify-machineinstrs
;
; Run these tests with and without -stress-early-ifcvt to exercise heuristics.
;
target triple = "x86_64-apple-macosx10.8.0"

; MachineTraceMetrics::Ensemble::addLiveIns crashes because the first operand
; on an inline asm instruction is not a vreg def.
; <rdar://problem/12472811>
define void @f1() nounwind {
entry:
  br i1 undef, label %if.then6.i, label %if.end.i

if.then6.i:
  br label %if.end.i

if.end.i:
  br i1 undef, label %if.end25.i, label %if.else17.i

if.else17.i:
  %shl24.i = shl i32 undef, undef
  br label %if.end25.i

if.end25.i:
  %storemerge31.i = phi i32 [ %shl24.i, %if.else17.i ], [ 0, %if.end.i ]
  store i32 %storemerge31.i, i32* undef, align 4
  %0 = tail call i32 asm sideeffect "", "=r,r,i,i"(i32 undef, i32 15, i32 1) nounwind
  %conv = trunc i32 %0 to i8
  store i8 %conv, i8* undef, align 1
  unreachable
}
