; NOTE: Do not autogenerate
;
; Kernels whose trip counts are runtime values. A runtime trip count is not by
; itself a runtime bound: the disjoint-memory kernels need no bound requirement
; and the same-array kernels discharge theirs from the constant array extent.
;
; With the option off the pass leaves this module unchanged.
; RUN: opt -S -passes=no-op-loopnest %s -o %t.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %s -o %t.out
; RUN: diff -u %t.noop %t.out
;
; DEFINE: %{policy} = -loop-interchange-profitabilities=instorder,vectorize
; DEFINE: %{prepare} = -loop-interchange-outer-epilogue-fission -loop-interchange-print-prepared-plan
; DEFINE: %{remarks} = -pass-remarks=loop-interchange -pass-remarks-analysis=loop-interchange -pass-remarks-missed=loop-interchange
;
; Preparation is analysis-only: a complete plan is printed and then discarded,
; so the IR is unchanged with the option on.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.prep 2> %t.prep.stderr
; RUN: FileCheck %s --check-prefix=PREP --input-file=%t.prep.stderr --implicit-check-not='loop-interchange:'
; RUN: diff -u %t.out %t.prep
;
; Each printed plan has passed both dependence matrices, legality,
; profitability, and validation. The ordinary miss that follows it closes the
; decision.
; PREP-LABEL: loop-interchange: prepared function=checksum_kernel_rect{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=8 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=2
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
; PREP-LABEL: loop-interchange: prepared function=checksum_kernel_latch{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=0 epilogue-insts=8 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=2
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
; PREP-LABEL: loop-interchange: prepared function=same_array_diag{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[SAME_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[SAME_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
; PREP-LABEL: loop-interchange: prepared function=nested_same_array_diag{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[NESTED_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[NESTED_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
;
; The guarded bypass of the inner loop is rejected at discovery.
; PREP-LABEL: loop-interchange: rejected function=checksum_kernel_zerotrip{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=exit-to-latch region is not one dominated acyclic path{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @checksum_kernel_rect(ptr noalias %A, i64 %n, ptr noalias %diag,
                                  ptr noalias %epcount, ptr noalias %sumout) {
entry:
  %n.pos = icmp sgt i64 %n, 0
  br i1 %n.pos, label %outer.ph, label %done
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %sum.i = phi i64 [ 0, %outer.ph ], [ %sum.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sum.j = phi i64 [ %sum.i, %outer.header ], [ %sum.next.j, %inner.header ]
  %aidx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %av = load i64, ptr %aidx, align 8
  %sum.next.j = add i64 %sum.j, %av
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %sum.next = phi i64 [ %sum.next.j, %inner.header ]
  %ii1 = add i64 %i, 1
  %dp = getelementptr inbounds i64, ptr %diag, i64 %i
  %dv = load i64, ptr %dp, align 8
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  %ecv = load i64, ptr %epcount, align 8
  %ecn = add i64 %ecv, 1
  store i64 %ecn, ptr %epcount, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %sum.res = phi i64 [ %sum.next, %outer.latch ]
  store i64 %sum.res, ptr %sumout, align 8
  br label %done
done:
  ret void
}

define void @checksum_kernel_latch(ptr noalias %A, i64 %n, ptr %diag,
                                   ptr noalias %epcount, ptr %sumout) {
entry:
  %n.pos = icmp sgt i64 %n, 0
  br i1 %n.pos, label %outer.ph, label %done
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %sum.i = phi i64 [ 0, %outer.ph ], [ %sum.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sum.j = phi i64 [ %sum.i, %outer.header ], [ %sum.next.j, %inner.header ]
  %aidx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %av = load i64, ptr %aidx, align 8
  %sum.next.j = add i64 %sum.j, %av
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %outer.latch, label %inner.header
outer.latch:
  %sum.next = phi i64 [ %sum.next.j, %inner.header ]
  %ii1 = add i64 %i, 1
  %dp = getelementptr inbounds i64, ptr %diag, i64 %i
  %dv = load i64, ptr %dp, align 8
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  %ecv = load i64, ptr %epcount, align 8
  %ecn = add i64 %ecv, 1
  store i64 %ecn, ptr %epcount, align 8
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %sum.res = phi i64 [ %sum.next, %outer.latch ]
  store i64 %sum.res, ptr %sumout, align 8
  br label %done
done:
  ret void
}

define void @same_array_diag(ptr noalias dereferenceable(128) %X,
                             ptr noalias %sumout) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %sum.i = phi i64 [ 0, %entry ], [ %sum.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sum.j = phi i64 [ %sum.i, %outer.header ], [ %sum.next.j, %inner.header ]
  %xidx = getelementptr inbounds [4 x i64], ptr %X, i64 %j, i64 %i
  %xv = load i64, ptr %xidx, align 8
  %sum.next.j = add i64 %sum.j, %xv
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %sum.next = phi i64 [ %sum.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x i64], ptr %X, i64 %i, i64 %i
  %dv = load i64, ptr %ddiag, align 8
  %dn = add i64 %dv, 100
  store i64 %dn, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %sum.res = phi i64 [ %sum.next, %outer.latch ]
  store i64 %sum.res, ptr %sumout, align 8
  ret void
}

define void @nested_same_array_diag(ptr noalias dereferenceable(128) %X,
                                    ptr noalias %rowsum) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %outer.latch ]
  %sum.i = phi i64 [ 0, %root.header ], [ %sum.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sum.j = phi i64 [ %sum.i, %outer.header ], [ %sum.next.j, %inner.header ]
  %xidx = getelementptr inbounds [4 x i64], ptr %X, i64 %j, i64 %i
  %xv = load i64, ptr %xidx, align 8
  %sum.next.j = add i64 %sum.j, %xv
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %sum.next = phi i64 [ %sum.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x i64], ptr %X, i64 %i, i64 %i
  %dv = load i64, ptr %ddiag, align 8
  %dn = add i64 %dv, 1
  store i64 %dn, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %outer.exit, label %outer.header
outer.exit:
  %sum.res = phi i64 [ %sum.next, %outer.latch ]
  %rk = getelementptr inbounds i64, ptr %rowsum, i64 %k
  %rkv = load i64, ptr %rk, align 8
  %rksum = add i64 %rkv, %sum.res
  store i64 %rksum, ptr %rk, align 8
  br label %sibling.header
sibling.header:
  %s = phi i64 [ 0, %outer.exit ], [ %s.next, %sibling.header ]
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 2
  br i1 %s.ec, label %root.latch, label %sibling.header
root.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 2
  br i1 %k.ec, label %exit, label %root.header
exit:
  ret void
}

define void @checksum_kernel_zerotrip(ptr noalias %A, i64 %n, i64 %m,
                                      ptr noalias %diag, ptr noalias %epcount,
                                      ptr noalias %sumout) {
entry:
  %n.pos = icmp sgt i64 %n, 0
  br i1 %n.pos, label %outer.ph, label %done
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %sum.i = phi i64 [ 0, %outer.ph ], [ %sum.next, %outer.latch ]
  %m.pos = icmp sgt i64 %m, 0
  br i1 %m.pos, label %inner.ph, label %epilogue
inner.ph:
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.ph ], [ %j.next, %inner.header ]
  %sum.j = phi i64 [ %sum.i, %inner.ph ], [ %sum.next.j, %inner.header ]
  %aidx = getelementptr inbounds [4 x i64], ptr %A, i64 %i, i64 %j
  %av = load i64, ptr %aidx, align 8
  %sum.next.j = add i64 %sum.j, %av
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %m
  br i1 %j.ec, label %inner.exit, label %inner.header
inner.exit:
  %sum.inner = phi i64 [ %sum.next.j, %inner.header ]
  br label %epilogue
epilogue:
  %sum.next = phi i64 [ %sum.i, %outer.header ], [ %sum.inner, %inner.exit ]
  %ii1 = add i64 %i, 1
  %dp = getelementptr inbounds i64, ptr %diag, i64 %i
  %dv = load i64, ptr %dp, align 8
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  %ecv = load i64, ptr %epcount, align 8
  %ecn = add i64 %ecv, 1
  store i64 %ecn, ptr %epcount, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %sum.res = phi i64 [ %sum.next, %outer.latch ]
  store i64 %sum.res, ptr %sumout, align 8
  br label %done
done:
  ret void
}
