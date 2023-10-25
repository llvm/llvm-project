; RUN: not --crash opt -passes='consthoist' -S -o - -consthoist-gep=1 -mtriple=x86_64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; My changes fixed this likely by accident, please update as necessary when
; you work on this:
; XFAIL: *

; Matching assertion strings is not easy as they might differ on different
; platforms. So limit this to x86_64-linux.
; REQUIRES: x86_64-linux

; This is a reproducer for https://github.com/llvm/llvm-project/issues/52689
;
; opt: ../lib/Transforms/Scalar/ConstantHoisting.cpp:919: bool llvm::ConstantHoistingPass::emitBaseConstants(llvm::GlobalVariable *): Assertion `UsesNum == (ReBasesNum + NotRebasedNum) && "Not all uses are rebased"' failed.

; CHECK: UsesNum == (ReBasesNum + NotRebasedNum)
; CHECK-SAME: Not all uses are rebased

@g_77 = external global [5 x i32]

define internal ptr @func_29(i1 %p1, i1 %p2, ptr %p3) {
entry:
  br i1 %p1, label %crit_edge, label %if.else3089

crit_edge:                                        ; preds = %entry
  br label %for.cond1063

for.cond1063:                                     ; preds = %cleanup1660, %crit_edge
  %l_323.sroa.0.0 = phi ptr [ getelementptr inbounds ([5 x i32], ptr @g_77, i32 0, i32 3), %cleanup1660 ], [ null, %crit_edge ]
  %l_323.sroa.2.0 = phi ptr [ getelementptr inbounds ([5 x i32], ptr @g_77, i32 0, i32 3), %cleanup1660 ], [ null, %crit_edge ]
  br i1 %p2, label %cleanup1660.thread, label %cleanup1674

cleanup1660.thread:                               ; preds = %for.cond1063
  br label %cleanup1674

cleanup1660:                                      ; No predecessors!
  br label %for.cond1063

cleanup1674:                                      ; preds = %cleanup1660.thread, %for.cond1063
  store ptr getelementptr inbounds ([5 x i32], ptr @g_77, i32 0, i32 1), ptr %p3, align 1
  ret ptr null

if.else3089:                                      ; preds = %entry
  store ptr getelementptr inbounds ([5 x i32], ptr @g_77, i32 0, i32 1), ptr %p3, align 1
  ret ptr null
}
