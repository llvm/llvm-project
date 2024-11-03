; RUN: llc -max-registers-for-gc-values=256 -verify-machineinstrs -stop-after twoaddressinstruction < %s | FileCheck --check-prefixes CHECK,CHECK-LV %s
; RUN: llc -max-registers-for-gc-values=256 -verify-machineinstrs -stop-after twoaddressinstruction -early-live-intervals < %s | FileCheck --check-prefixes CHECK,CHECK-LIS %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo() gc "statepoint-example"
declare void @bar(i8 addrspace(1)*) gc "statepoint-example"

declare i32* @fake_personality_function()

; Simplest possible test demonstrating the problem

; CHECK-LABEL: name: test
; CHECK:  bb.0
; CHECK-LV:     %0:gr64 = COPY killed $rdi
; CHECK-LIS:    %0:gr64 = COPY $rdi
; CHECK:        %1:gr64 = COPY %0
; CHECK:        %1:gr64 = STATEPOINT 2, 5, 0, undef %2:gr64, 2, 0, 2, 0, 2, 0, 2, 1, %1(tied-def 0), 2, 0, 2, 1, 0, 0
; CHECK-LV:     TEST64rr killed %0, %0, implicit-def $eflags
; CHECK-LIS:    TEST64rr %0, %0, implicit-def $eflags
; CHECK:        JCC_1 %bb.2, 4, implicit killed $eflags
; CHECK:        JMP_1 %bb.1
; CHECK:      bb.1
; CHECK-LV:     $rdi = COPY killed %1
; CHECK-LV:     STATEPOINT 2, 5, 1, undef %3:gr64, killed $rdi, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0
; CHECK-LIS:    $rdi = COPY %1
; CHECK-LIS:    STATEPOINT 2, 5, 1, undef %3:gr64, $rdi, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0
; CHECK:        RET 0
; CHECK:      bb.2
; CHECK:        RET 0
define void @test(i8 addrspace(1)* %a)  gc "statepoint-example" {
entry:
  %not7 = icmp eq i8 addrspace(1)* %a, null
  %statepoint_token1745 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2, i32 5, void ()* nonnull elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) [ "deopt"(), "gc-live"(i8 addrspace(1)* %a) ]
  br i1 %not7, label %zero, label %not_zero

not_zero:
  %a.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token1745, i32 0, i32 0) ; (%a, %a)
  %statepoint_token1752 = call token (i64, i32, void (i8 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidp1i8f(i64 2, i32 5, void (i8 addrspace(1)*)* nonnull elementtype(void (i8 addrspace(1)*)) @bar, i32 1, i32 0, i8 addrspace(1)* %a.relocated, i32 0, i32 0) [ "deopt"(), "gc-live"() ]
  ret void

zero:
  ret void
}

; A bit more complex test, where both registers are used in same successor BB

; CHECK-LABEL: name: test2
; CHECK:  bb.2
; CHECK:        %1:gr64 = STATEPOINT 2882400000, 0, 0, undef %11:gr64, 2, 0, 2, 0, 2, 0, 2, 1, %1(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64
; CHECK:        %10:gr64 = COPY %1
; CHECK:        %10:gr64 = STATEPOINT 2882400000, 0, 0, undef %13:gr64, 2, 0, 2, 0, 2, 0, 2, 1, %10(tied-def 0), 2, 0, 2, 1, 0, 0, csr_64
; CHECK:        JMP_1 %bb.3
; CHECK:      bb.3
; CHECK:        %18:gr8 = COPY %17.sub_8bit
; CHECK-LV:     TEST8rr killed %18, %18, implicit-def $eflags
; CHECK-LIS:    TEST8rr %18, %18, implicit-def $eflags
; CHECK:        JCC_1 %bb.5, 5, implicit killed $eflags
; CHECK:        JMP_1 %bb.4
; CHECK:      bb.4
; CHECK:      bb.5
; CHECK:        %3:gr64 = COPY %10
; CHECK-LV:     %4:gr64 = COPY killed %10
; CHECK-LIS:    %4:gr64 = COPY %10
; CHECK:        %4:gr64 = nuw ADD64ri32 %4, 8, implicit-def dead $eflags
; CHECK:        TEST64rr killed %1, %1, implicit-def $eflags
; CHECK:        JCC_1 %bb.1, 5, implicit killed $eflags
; CHECK:        JMP_1 %bb.6
define void @test2(i8 addrspace(1)* %this, i32 %0, i32 addrspace(1)* %p0, i8 addrspace(1)* %p1) gc "statepoint-example" personality i32* ()* @fake_personality_function {
preheader:
  br label %loop.head

loop.head:
  %phi1 = phi i32 addrspace(1)* [ %p0, %preheader ], [ %addr.i.i46797.remat64523, %tail ]
  %v1 = phi i8 addrspace(1)* [ %p1, %preheader ], [ %v3, %tail ]
  %not3= icmp ne i32 addrspace(1)* %phi1, null
  br i1 %not3, label %BB1, label %BB3

BB3:
  %token1 = call token (i64, i32, i8 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i8f(i64 2882400000, i32 0, i8 addrspace(1)* ()* elementtype(i8 addrspace(1)* ()) undef, i32 0, i32 0, i32 0, i32 0) [ "deopt"(), "gc-live"(i8 addrspace(1)* %v1) ]
  %v2 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %token1, i32 0, i32 0) ; (%v1, %v1)
  %cond = icmp eq i8 addrspace(1)* null, %v2
  %token2 = invoke token (i64, i32, i8 addrspace(1)* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i8f(i64 2882400000, i32 0, i8 addrspace(1)* ()* elementtype(i8 addrspace(1)* ()) undef, i32 0, i32 0, i32 0, i32 0) [ "deopt"(), "gc-live"(i8 addrspace(1)* %v2, i32 addrspace(1)* %phi1) ]
          to label %BB2 unwind label %BB6

BB2:
  %v3 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %token2, i32 0, i32 0) ; (%v2, %v2)
  %.remat64522 = getelementptr inbounds i8, i8 addrspace(1)* %v3, i64 8
  %addr.i.i46797.remat64523 = bitcast i8 addrspace(1)* %.remat64522 to i32 addrspace(1)*
  br i1 undef, label %BB4, label %tail

BB4:
  %dummy = ptrtoint i64* undef to i64
  br label %tail

tail:
  br i1 %cond, label %BB1, label %loop.head

BB1:
  ret void

BB6:
  %lpad.split-lp = landingpad token
          cleanup
  ret void

}


declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32 immarg, i32 immarg) #5
declare token @llvm.experimental.gc.statepoint.p0f_p1i8f(i64 immarg, i32 immarg, i8 addrspace(1)* ()*, i32 immarg, i32 immarg, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 immarg, i32 immarg, void ()*, i32 immarg, i32 immarg, ...)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidp1i8f(i64 immarg, i32 immarg, void (i8 addrspace(1)*)*, i32 immarg, i32 immarg, ...)
