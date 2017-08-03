; RUN: llc < %s -march=x86-64 -mcpu=core2 -x86-early-ifcvt -enable-misched \
; RUN:          -misched=shuffle -misched-bottomup -verify-machineinstrs \
; RUN:     | FileCheck %s
; RUN: llc < %s -march=x86-64 -mcpu=core2 -x86-early-ifcvt -enable-misched \
; RUN:          -misched=shuffle -misched-topdown -verify-machineinstrs \
; RUN:     | FileCheck %s --check-prefix TOPDOWN
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

; Test that the DAG builder can handle an undef vreg on ExitSU.
; CHECK: hasundef
; CHECK: call

%t0 = type { i32, i32, i8 }
%t6 = type { i32 (...)**, %t7* }
%t7 = type { i32 (...)** }

define void @hasundef() unnamed_addr uwtable ssp align 2 {
  %1 = alloca %t0, align 8
  br i1 undef, label %3, label %2

; <label>:2                                       ; preds = %0
  unreachable

; <label>:3                                       ; preds = %0
  br i1 undef, label %4, label %5

; <label>:4                                       ; preds = %3
  call void undef(%t6* undef, %t0* %1)
  unreachable

; <label>:5                                       ; preds = %3
  ret void
}

; Test top-down subregister liveness tracking. Self-verification
; catches any pressure set underflow.
; rdar://12797931.
;
; TOPDOWN: @testSubregTracking
; TOPDOWN: divb
; TOPDOWN: movzbl %al
; TOPDOWN: ret
define void @testSubregTracking() nounwind uwtable ssp align 2 {
  %tmp = load i8, i8* undef, align 1
  %tmp6 = sub i8 0, %tmp
  %tmp7 = load i8, i8* undef, align 1
  %tmp8 = udiv i8 %tmp6, %tmp7
  %tmp9 = zext i8 %tmp8 to i64
  %tmp10 = load i8, i8* undef, align 1
  %tmp11 = zext i8 %tmp10 to i64
  %tmp12 = mul i64 %tmp11, %tmp9
  %tmp13 = urem i8 %tmp6, %tmp7
  %tmp14 = zext i8 %tmp13 to i32
  %tmp15 = add nsw i32 %tmp14, 0
  %tmp16 = add i32 %tmp15, 0
  store i32 %tmp16, i32* undef, align 4
  %tmp17 = add i64 0, %tmp12
  store i64 %tmp17, i64* undef, align 8
  ret void
}
