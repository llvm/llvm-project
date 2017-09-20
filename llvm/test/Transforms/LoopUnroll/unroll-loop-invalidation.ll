; This test exercises that we don't corrupt a loop-analysis when running loop
; unrolling in a way that deletes a loop. To do that, we first ensure the
; analysis is cached, then unroll the loop (deleting it) and make sure that the
; next function doesn't get a cache "hit" for this stale analysis result.
;
; RUN: opt -S -passes='loop(require<access-info>),unroll,loop(print-access-info)' -debug-pass-manager < %s 2>&1 | FileCheck %s
;
; CHECK: Starting llvm::Function pass manager run.
; CHECK: Running pass: FunctionToLoopPassAdaptor
; CHECK: Running analysis: LoopAnalysis
; CHECK: Running analysis: InnerAnalysisManagerProxy<
; CHECK: Starting Loop pass manager run.
; CHECK: Running pass: RequireAnalysisPass<{{.*}}LoopAccessAnalysis
; CHECK: Running analysis: LoopAccessAnalysis on inner1.header
; CHECK: Finished Loop pass manager run.
; CHECK: Starting Loop pass manager run.
; CHECK: Running pass: RequireAnalysisPass<{{.*}}LoopAccessAnalysis
; CHECK: Running analysis: LoopAccessAnalysis on inner2.header
; CHECK: Finished Loop pass manager run.
; CHECK: Starting Loop pass manager run.
; CHECK: Running pass: RequireAnalysisPass<{{.*}}LoopAccessAnalysis
; CHECK: Running analysis: LoopAccessAnalysis on outer.header
; CHECK: Finished Loop pass manager run.
; CHECK: Running pass: LoopUnrollPass
; CHECK: Clearing all analysis results for: <invalidated loop>
; CHECK: Clearing all analysis results for: <invalidated loop>
; CHECK: Invalidating all non-preserved analyses for: test
; CHECK: Invalidating all non-preserved analyses for: inner1.header
; CHECK: Invalidating analysis: LoopAccessAnalysis on inner1.header
; CHECK: Invalidating all non-preserved analyses for: inner1.header.1
; CHECK-NOT: Invalidating analysis: LoopAccessAnalysis on inner1.header.1
; CHECK: Running pass: FunctionToLoopPassAdaptor
; CHECK: Starting Loop pass manager run.
; CHECK: Running pass: LoopAccessInfoPrinterPass
; CHECK: Running analysis: LoopAccessAnalysis on inner1.header
; CHECK: Loop access info in function 'test':
; CHECK:   inner1.header:
; CHECK: Finished Loop pass manager run.
; CHECK: Starting Loop pass manager run.
; CHECK: Running pass: LoopAccessInfoPrinterPass
; CHECK: Running analysis: LoopAccessAnalysis on inner1.header.1
; CHECK: Loop access info in function 'test':
; CHECK:   inner1.header.1:
; CHECK: Finished Loop pass manager run.

target triple = "x86_64-unknown-linux-gnu"

define void @test(i32 %inner1.count) {
; CHECK-LABEL: define void @test(
bb:
  br label %outer.ph

outer.ph:
  br label %outer.header

outer.header:
  %outer.i = phi i32 [ 0, %outer.ph ], [ %outer.i.next, %outer.latch ]
  br label %inner1.ph

inner1.ph:
  br label %inner1.header

inner1.header:
  %inner1.i = phi i32 [ 0, %inner1.ph ], [ %inner1.i.next, %inner1.header ]
  %inner1.i.next = add i32 %inner1.i, 1
  %inner1.cond = icmp eq i32 %inner1.i, %inner1.count
  br i1 %inner1.cond, label %inner1.exit, label %inner1.header
; We should have two unrolled copies of this loop and nothing else.
;
; CHECK-NOT:     icmp eq
; CHECK-NOT:     br i1
; CHECK:         %[[COND1:.*]] = icmp eq i32 %{{.*}}, %inner1.count
; CHECK:         br i1 %[[COND1]],
; CHECK-NOT:     icmp eq
; CHECK-NOT:     br i1
; CHECK:         %[[COND2:.*]] = icmp eq i32 %{{.*}}, %inner1.count
; CHECK:         br i1 %[[COND2]],
; CHECK-NOT:     icmp eq
; CHECK-NOT:     br i1


inner1.exit:
  br label %inner2.ph

inner2.ph:
  br label %inner2.header

inner2.header:
  %inner2.i = phi i32 [ 0, %inner2.ph ], [ %inner2.i.next, %inner2.header ]
  %inner2.i.next = add i32 %inner2.i, 1
  %inner2.cond = icmp eq i32 %inner2.i, 4
  br i1 %inner2.cond, label %inner2.exit, label %inner2.header

inner2.exit:
  br label %outer.latch

outer.latch:
  %outer.i.next = add i32 %outer.i, 1
  %outer.cond = icmp eq i32 %outer.i.next, 2
  br i1 %outer.cond, label %outer.exit, label %outer.header

outer.exit:
  br label %exit

exit:
  ret void
}
