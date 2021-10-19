; Test -sanitizer-coverage-trace-compares=1 and how it prunes backedge compares.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-compares=1  -sanitizer-coverage-prune-blocks=1 -S | FileCheck %s --check-prefix=PRUNE
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-compares=1  -sanitizer-coverage-prune-blocks=0 -S | FileCheck %s --check-prefix=NOPRUNE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo(i32* nocapture readnone %a, i32 %n) local_unnamed_addr {
entry:
  br label %do.body

do.body:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  tail call void (...) @bar()
  %inc = add nuw nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc, %n
;PRUNE-LABEL: foo
;PRUNE-NOT: __sanitizer_cov_trace_cmp4
;PRUNE: ret void

;NOPRUNE-LABEL: foo
;NOPRUNE: call void @__sanitizer_cov_trace_cmp4
;NOPRUNE-NEXT: icmp
;NOPRUNE: ret void

  br i1 %cmp, label %do.body, label %do.end

do.end:
  ret void
}

declare dso_local void @bar(...) local_unnamed_addr
