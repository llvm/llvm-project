; REQUIRES: asserts
; RUN: not  --crash llc -o /dev/null %s -max-registers-for-gc-values=15 -use-registers-for-gc-values-in-landing-pad=true -verify-regalloc 2>&1 | FileCheck %s

; The test checks the verification catch the case when RA splits live interval in the
; way the def is located after invoke statepoint while use is in landing pad.

; CHECK: *** Bad machine code: Register not marked live out of predecessor ***

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @wombat(i8 addrspace(1)* %arg, i32 %arg1, i32 addrspace(1)* %arg2) gc "statepoint-example" personality i32* ()* @widget {
bb:
  %tmp = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* null, align 8
  %tmp3 = load i32, i32 addrspace(1)* null, align 4
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 24
  %tmp5 = load i32, i32 addrspace(1)* %tmp4, align 4
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 40
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = load i32, i32 addrspace(1)* null, align 4
  %tmp9 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* undef, align 8
  %tmp10 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 88
  %tmp11 = load i32, i32 addrspace(1)* %tmp10, align 4
  %tmp12 = getelementptr inbounds i8, i8 addrspace(1)* %arg, i64 96
  %tmp13 = bitcast i8 addrspace(1)* %tmp12 to i8 addrspace(1)* addrspace(1)*
  %tmp14 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %tmp13, align 8
  %tmp15 = getelementptr inbounds i8, i8 addrspace(1)* %arg, i64 104
  %tmp16 = bitcast i8 addrspace(1)* %tmp15 to i8 addrspace(1)* addrspace(1)*
  %tmp17 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %tmp16, align 8
  %tmp18 = add i32 %tmp3, -1
  %tmp19 = load atomic i64, i64 addrspace(1)* undef unordered, align 8
  %tmp20 = invoke token (i64, i32, i32 (i32, i8 addrspace(1)*, i32, i32, i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32i32p1i8i32i32i32f(i64 1, i32 16, i32 (i32, i8 addrspace(1)*, i32, i32, i32)* nonnull @wombat.1, i32 5, i32 0, i32 0, i8 addrspace(1)* null, i32 undef, i32 %arg1, i32 0, i32 0, i32 0) [ "deopt"(i32 %tmp18, i8 addrspace(1)* %tmp, i32 %arg1, i32 %tmp3, i32 %tmp5, i32 %tmp7, i32 %tmp8, i8 addrspace(1)* %tmp9, i32 %tmp11, i8 addrspace(1)* %tmp14, i8 addrspace(1)* %tmp17), "gc-live"(i8 addrspace(1)* %tmp, i8 addrspace(1)* %tmp9, i8 addrspace(1)* %tmp14, i8 addrspace(1)* %tmp17) ]
          to label %bb21 unwind label %bb26

bb21:                                             ; preds = %bb
  %tmp22 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tmp20, i32 0, i32 0) ; (%tmp, %tmp)
  %tmp23 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tmp20, i32 2, i32 2) ; (%tmp14, %tmp14)
  %tmp24 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tmp20, i32 3, i32 3) ; (%tmp17, %tmp17)
  %tmp25 = call token (i64, i32, void (i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidi32f(i64 2882400000, i32 0, void (i32)* nonnull @quux, i32 1, i32 2, i32 10, i32 0, i32 0) [ "deopt"(i32 %tmp18, i8 addrspace(1)* %tmp22, i32 %arg1, i32 %tmp3, i32 %tmp5, i32 %tmp7, i32 %tmp8, i32 %tmp11, i8 addrspace(1)* %tmp23, i8 addrspace(1)* %tmp24), "gc-live"() ]
  ret void

bb26:                                             ; preds = %bb
  %tmp27 = landingpad token
          cleanup
  %tmp28 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tmp27, i32 1, i32 1) ; (%tmp9, %tmp9)
  %tmp29 = call token (i64, i32, void (i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidi32f(i64 2882400000, i32 0, void (i32)* nonnull @quux, i32 1, i32 0, i32 -271, i32 0, i32 0) [ "deopt"(i32 %arg1, i32 %tmp3, i32 %tmp5, i32 %tmp8, i8 addrspace(1)* %tmp28, i32 %tmp11), "gc-live"() ]
  unreachable
}

declare i32* @widget()

declare i32 @wombat.1(i32, i8 addrspace(1)*, i32, i32, i32)

declare void @quux(i32)

declare token @llvm.experimental.gc.statepoint.p0f_isVoidi32f(i64 immarg, i32 immarg, void (i32)*, i32 immarg, i32 immarg, ...)

; Function Attrs: nounwind readonly
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32 immarg, i32 immarg) #0

declare token @llvm.experimental.gc.statepoint.p0f_i32i32p1i8i32i32i32f(i64 immarg, i32 immarg, i32 (i32, i8 addrspace(1)*, i32, i32, i32)*, i32 immarg, i32 immarg, ...)

attributes #0 = { nounwind readonly }
