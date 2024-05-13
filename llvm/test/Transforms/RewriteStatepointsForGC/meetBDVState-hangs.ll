; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s

; Regression test to incorrectly testing fixed state causing infinite loop.
; CHECK: test
target triple = "x86_64-unknown-linux-gnu"

declare void @bar(ptr addrspace(1) nocapture readonly)
declare noalias ptr addrspace(1) @foo()

define ptr addrspace(1) @test(i1 %c, i1 %c1, i1 %c2, i1 %c3, i1 %c4, i1 %c5, i1 %c.exit) gc "statepoint-example" {
entry:
  br i1 %c, label %ph.L, label %ph.R
ph.L:
  %ph.L.p.b = call noalias nonnull ptr addrspace(1) @foo()
  %ph.L.p = getelementptr i8, ptr addrspace(1) %ph.L.p.b, i64 8
  br label %ph.M
ph.R:
  %ph.R.p = call noalias nonnull ptr addrspace(1) @foo()
  br label %ph.M
ph.M:
  %ph.M.p = phi ptr addrspace(1) [ %ph.L.p, %ph.L ], [ %ph.R.p, %ph.R ]
  br label %header

header:
  %header.p = phi ptr addrspace(1) [ %ph.M.p, %ph.M ], [ %backedge.p, %backedge]
  br i1 %c1, label %loop.M, label %loop.R

loop.R:
  br i1 %c2, label %loop.R.M, label %loop.R.R

loop.R.R:
  %loop.R.R.p = call noalias nonnull ptr addrspace(1) @foo()
  br label %loop.R.M

loop.R.M:
  %loop.R.M.p = phi ptr addrspace(1) [ %header.p, %loop.R ], [ %loop.R.R.p, %loop.R.R ]
  br label %loop.M

loop.M:
  %loop.M.p = phi ptr addrspace(1) [ %loop.R.M.p, %loop.R.M ], [ %header.p, %header ]
  br i1 %c4, label %backedge, label %pre.backedge.R

pre.backedge.R:
  br i1 %c5, label %pre.backedge.R.L, label %pre.backedge.R.R
pre.backedge.R.L:
  %pre.backedge.R.L.p.b = call noalias nonnull ptr addrspace(1) @foo()
  %pre.backedge.R.L.p = getelementptr i8, ptr addrspace(1) %pre.backedge.R.L.p.b, i64 8
  br label %pre.backedge.R.M
pre.backedge.R.R:
  %pre.backedge.R.R.p = call noalias nonnull ptr addrspace(1) @foo()
  br label %pre.backedge.R.M
pre.backedge.R.M:
  %pre.backedge.R.M.p = phi ptr addrspace(1) [ %pre.backedge.R.L.p, %pre.backedge.R.L ], [ %pre.backedge.R.R.p, %pre.backedge.R.R ]
  br label %backedge

backedge:
  %backedge.p = phi ptr addrspace(1) [ %pre.backedge.R.M.p, %pre.backedge.R.M ], [ %loop.M.p, %loop.M ]
  br i1 %c.exit, label %header, label %exit

exit:                                                ; preds = %3, %1
  call void @bar(ptr addrspace(1) align 8 %header.p) [ "deopt"() ]
  ret ptr addrspace(1) %header.p
}
