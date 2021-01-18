; RUN: opt -loop-reduce %s -o - -S | FileCheck %s
; RUN: opt -passes='loop(loop-reduce)' %s -o - -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define dso_local i32 @test1() local_unnamed_addr {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
; It's ok to modify this test in the future should be able to split critical
; edges here, just noting that this is the critical edge that we care about.
; CHECK: callbr void asm sideeffect "", "X,X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@test1, %cond.true.i), i8* blockaddress(@test1, %for.end))
; CHECK-NEXT: to label %asm.fallthrough.i.i [label %cond.true.i, label %for.end]
  callbr void asm sideeffect "", "X,X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@test1, %cond.true.i), i8* blockaddress(@test1, %for.end))
          to label %asm.fallthrough.i.i [label %cond.true.i, label %for.end]

asm.fallthrough.i.i:                              ; preds = %for.cond
  unreachable

cond.true.i:                                      ; preds = %for.cond
  br label %do.body.i.i.do.body.i.i_crit_edge

do.body.i.i.do.body.i.i_crit_edge:                ; preds = %do.body.i.i.do.body.i.i_crit_edge, %cond.true.i
  %pgocount711 = phi i64 [ %0, %do.body.i.i.do.body.i.i_crit_edge ], [ 0, %cond.true.i ]
  %0 = add nuw nsw i64 %pgocount711, 1
  br i1 undef, label %do.body.i.i.rdrand_int.exit.i_crit_edge, label %do.body.i.i.do.body.i.i_crit_edge

do.body.i.i.rdrand_int.exit.i_crit_edge:          ; preds = %do.body.i.i.do.body.i.i_crit_edge
  %1 = add i64 %0, undef
  br i1 undef, label %for.end, label %for.inc

for.inc:                                          ; preds = %do.body.i.i.rdrand_int.exit.i_crit_edge
  br label %for.cond

for.end:                                          ; preds = %do.body.i.i.rdrand_int.exit.i_crit_edge, %for.cond
  %pgocount.promoted24 = phi i64 [ undef, %for.cond ], [ %1, %do.body.i.i.rdrand_int.exit.i_crit_edge ]
  ret i32 undef
}
