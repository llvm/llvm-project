; RUN: opt -o %t %s -passes=instcombine,simplifycfg -simplifycfg-require-and-preserve-domtree=1 -thinlto-bc -verify-assumption-cache
; RUN: llvm-dis -o - %t | FileCheck %s

; Test that the simplifycfg pass correctly updates the assumption cache
; when it clones the llvm.assume call as part of creating a critical
; edge. To do that, we set up a pass pipeline such that (1) an assumption
; cache is created for foo before simplifycfg updates it, and (2) foo's
; assumption cache is verified after simplifycfg has run. To satisfy 1, we
; run the instcombine pass first in our pipeline. To satisfy 2, we use the
; ThinLTOBitcodeWriter pass to write bitcode (that pass uses the assumption
; cache). That ensures that the pass manager does not call releaseMemory()
; on the AssumptionCacheTracker before the end of the pipeline, which would
; wipe out the bad assumption cache before it is verified.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.F = type { i8 }
%class.B = type { i8 }
%class.A = type { %class.C }
%class.C = type { ptr }

define void @foo(ptr %this, ptr %out) {
entry:
  %call = tail call i32 @_ZNK1F5beginEv(ptr %this)
  %call2 = tail call i32 @_ZNK1F3endEv(ptr %this)
  %cmp.i22 = icmp eq i32 %call, %call2
  br i1 %cmp.i22, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %frame_node.sroa.0.023 = phi i32 [ %inc.i, %_ZN10unique_ptrD2Ev.exit ], [ %call, %while.body.preheader ]
  %call8 = tail call ptr @_Znwm(i64 8)
  %inc.i = add nsw i32 %frame_node.sroa.0.023, 1
  %cmp = icmp eq i32 %inc.i, %call2
  br i1 %cmp, label %_ZN10unique_ptrD2Ev.exit, label %if.then

if.then:
  tail call void @_ZN1B6appendEv(ptr %out)
  br label %_ZN10unique_ptrD2Ev.exit

_ZN10unique_ptrD2Ev.exit:
  %vtable.i.i = load ptr, ptr %call8, align 8
  %x3 = tail call i1 @llvm.type.test(ptr %vtable.i.i, metadata !"foo")
  ; CHECK: call void @llvm.assume
  ; CHECK: call void @llvm.assume
  tail call void @llvm.assume(i1 %x3) #5
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

declare void @llvm.lifetime.start.p0(ptr nocapture)

declare i32 @_ZNK1F5beginEv(ptr)

declare i32 @_ZNK1F3endEv(ptr)

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

declare noalias nonnull ptr @_Znwm(i64)

declare void @_ZN1B6appendEv(ptr)

declare void @llvm.lifetime.end.p0(ptr nocapture)

declare i1 @llvm.type.test(ptr, metadata)

declare void @llvm.assume(i1)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 5.0.0 "}
