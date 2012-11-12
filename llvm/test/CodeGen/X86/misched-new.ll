; RUN: llc < %s -march=x86-64 -mcpu=core2 -x86-early-ifcvt -enable-misched \
; RUN:          -misched=shuffle -misched-bottomup -verify-machineinstrs \
; RUN:     | FileCheck %s
; REQUIRES: asserts
;
; Interesting MachineScheduler cases.
;
; FIXME: There should be an assert in the coalescer that we're not rematting
; "not-quite-dead" copies, but that breaks a lot of tests <rdar://problem/11148682>.

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

; From oggenc.
; After coalescing, we have a dead superreg (RAX) definition.
;
; CHECK: xorl %esi, %esi
; CHECK: movl $32, %ecx
; CHECK: rep;movsl
define fastcc void @_preextrapolate_helper() nounwind uwtable ssp {
entry:
  br i1 undef, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %entry
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* undef, i8* null, i64 128, i32 4, i1 false) nounwind
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

; The machine verifier checks that EFLAGS kill flags are updated when
; the scheduler reorders cmovel instructions.
;
; CHECK: test
; CHECK: cmovel
; CHECK: cmovel
; CHECK: call
define void @foo(i32 %b) nounwind uwtable ssp {
entry:
  %tobool = icmp ne i32 %b, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %v1 = phi i32 [1, %entry], [2, %if.then]
  %v2 = phi i32 [3, %entry], [4, %if.then]
  call void @bar(i32 %v1, i32 %v2)
  ret void
}

declare void @bar(i32,i32)
