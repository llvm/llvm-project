; NOTE: Do not autogenerate
;
; Guarded runtime-bound candidates whose row stride W comes from the exact byte
; coefficients of a column read and a diagonal write. Each accepted candidate is
; versioned behind one unsigned trip guard 'trip u> W'. The versioned copy has
; its proven epilogue distributed into a sibling loop and its reduction nest
; interchanged, and the fallback clone keeps the original loop. W == 1 is left
; alone because interchange is not profitable there.
;
; With the option off the pass leaves this module unchanged.
; RUN: opt -S -passes=no-op-loopnest %s -o %t.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %s -o %t.out
; RUN: diff -u %t.noop %t.out
;
; DEFINE: %{policy} = -loop-interchange-profitabilities=instorder,vectorize
; DEFINE: %{prepare} = -loop-interchange-outer-epilogue-fission -loop-interchange-print-prepared-plan -loop-interchange-outer-epilogue-runtime-versioning
; DEFINE: %{remarks} = -pass-remarks=loop-interchange -pass-remarks-analysis=loop-interchange -pass-remarks-missed=loop-interchange
; DEFINE: %{applied} = \
; DEFINE:   --func=cand_w2 --func=cand_w3 --func=cand_w7 \
; DEFINE:   --func=cand_canonical --func=cand_byte_w4
; DEFINE: %{no_alias_metadata} = \
; DEFINE:   --implicit-check-not='!alias.scope' \
; DEFINE:   --implicit-check-not='!noalias'
;
; With the option on, every function in %{applied} is versioned and every other
; function is unchanged.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.prep 2> %t.prep.stderr
; RUN: FileCheck %s --check-prefix=PREP --input-file=%t.prep.stderr --implicit-check-not='loop-interchange:'
; RUN: llvm-extract -S --delete %{applied} %t.out -o %t.out.rest
; RUN: llvm-extract -S --delete %{applied} %t.prep -o %t.prep.rest
; RUN: diff -u -I '^; ModuleID' %t.out.rest %t.prep.rest
; RUN: FileCheck %s --check-prefix=APPLIED --input-file=%t.prep %{no_alias_metadata} --implicit-check-not='{{^define }}' --implicit-check-not='!llvm.loop' --implicit-check-not='{{^[^[:space:]]*[.]lver[.]check[^:]*:}}'
;
; The updated LoopInfo must match a fresh rebuild, sibling order included, and
; a second run of the pass must change nothing.
; RUN: opt -passes='loop(loop-interchange),print<loops>' -cache-line-size=64 %{policy} -loop-interchange-outer-epilogue-fission -loop-interchange-outer-epilogue-runtime-versioning -disable-output %s 2>&1 | FileCheck %s --check-prefix=APPLY-LOOPS
; RUN: opt -passes='print<loops>' -disable-output %t.prep 2>&1 | FileCheck %s --check-prefix=APPLY-LOOPS
; RUN: opt -S -passes='loop(loop-interchange,loop-interchange)' -cache-line-size=64 %{policy} %{prepare} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.twice 2> %t.twice.stderr
; RUN: diff -u %t.prep %t.twice
;
; The W == 1 candidate is the profitability control: its virtual extracted nest
; is rejected before any runtime work, so the function is never versioned.
; PREP-LABEL: loop-interchange: rejected function=cand_w1{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing default profitability{{$}}
;
; Each printed plan has passed both dependence matrices, legality,
; profitability, validation, and the runtime versioning checks. The two
; remarks after it come from the apply step. Wmin is the recovered row stride.
; PREP-LABEL: loop-interchange: prepared function=cand_w2{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=8 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=6
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=2
; PREP-SAME:  requirement-ids=[[W2_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[W2_ID]]:object-containment/runtime/runtime{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=2.
; PREP-LABEL: loop-interchange: prepared function=cand_w3{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=8 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=6
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=3
; PREP-SAME:  requirement-ids=[[W3_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[W3_ID]]:object-containment/runtime/runtime{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=3.
; PREP-LABEL: loop-interchange: prepared function=cand_w7{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=8 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=6
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=7
; PREP-SAME:  requirement-ids=[[W7_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[W7_ID]]:object-containment/runtime/runtime{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=7.
; PREP-LABEL: loop-interchange: prepared function=cand_canonical{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=1335
; PREP-SAME:  requirement-ids=[[CANON_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[CANON_ID]]:object-containment/runtime/runtime{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=1335.
;
; The same stride recovered from explicit byte arithmetic reaches the same plan.
; PREP-LABEL: loop-interchange: prepared function=cand_byte_w4{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=9 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=6
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[BYTE_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[BYTE_ID]]:object-containment/runtime/runtime{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
;
; The versioned module. Every definition is listed, so the '{{^define }}'
; exclusion on that FileCheck invocation makes the list exhaustive. Each
; applied function gains one guard block, one untransformed fallback clone, one
; distributed and interchanged versioned nest, and one sibling epilogue loop on
; that path. Each applied function also keeps the entry bypass that skips the
; whole nest when the trip is not positive, so the entry compare and its branch
; are asserted here. The fallback clone keeps the whole original epilogue, so
; its epilogue block is checked instruction by instruction, including the inner
; loop's exit into it and its stores. The same invocation also excludes every
; '!llvm.loop' attachment and every '.lver.check' label definition it does not
; match here, so the two markers per applied function and the single guard per
; applied function are exhaustive too. Block labels are plain directives
; because the printed IR separates blocks with a blank line. @cand_w1 was
; already shown unchanged by the llvm-extract --delete pair above and carries
; only its label.
;
; APPLIED-LABEL: define {{.*}}@cand_w1(
;
; The three reductions are bound from each loop's dedicated exit through the
; shared join to their own consumer stores, so a permuted join fails.
; APPLIED-LABEL: define {{.*}}@cand_w2(
; APPLIED:         entry:
; APPLIED-NEXT:      %pos = icmp sgt i64 %n, 0
; APPLIED-NEXT:      br i1 %pos, label %outer.header.lver.check, label %ret
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 2
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %p.next.lver.orig = phi i64 [ %p.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %u.next.lver.orig = phi i64 [ %u.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %v.next.lver.orig = phi i64 [ %v.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %dp.lver.orig = getelementptr inbounds [2 x i64], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %dv.lver.orig = load i64, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ii1.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %dn.lver.orig = add i64 %dv.lver.orig, %ii1.lver.orig
; APPLIED-NEXT:      store i64 %dn.lver.orig, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ec.lver.orig = load i64, ptr %epc, align 8
; APPLIED-NEXT:      %ecn.lver.orig = add i64 %ec.lver.orig, 1
; APPLIED-NEXT:      store i64 %ecn.lver.orig, ptr %epc, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_W2:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_W2:[0-9]+]]
; APPLIED:         exit.loopexit:
; APPLIED-NEXT:      %[[P_FB:[^ ]+]] = phi i64 [ %p.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      %[[U_FB:[^ ]+]] = phi i64 [ %u.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      %[[V_FB:[^ ]+]] = phi i64 [ %v.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      br label %exit
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %[[P_FAST:[^ ]+]] = phi i64 [ %p.next, %inner.header.split ]
; APPLIED-NEXT:      %[[U_FAST:[^ ]+]] = phi i64 [ %u.next, %inner.header.split ]
; APPLIED-NEXT:      %[[V_FAST:[^ ]+]] = phi i64 [ %v.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr inbounds [2 x i64], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dv.epil = load i64, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ii1.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %dn.epil = add i64 %dv.epil, %ii1.epil
; APPLIED-NEXT:      store i64 %dn.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ec.epil = load i64, ptr %epc, align 8
; APPLIED-NEXT:      %ecn.epil = add i64 %ec.epil, 1
; APPLIED-NEXT:      store i64 %ecn.epil, ptr %epc, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         exit.loopexit1.cont:
; APPLIED-NEXT:      br label %exit
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %[[P_RES:[^ ]+]] = phi i64 [ %[[P_FB]], %exit.loopexit ], [ %[[P_FAST]], %exit.loopexit1.cont ]
; APPLIED-NEXT:      %[[U_RES:[^ ]+]] = phi i64 [ %[[U_FB]], %exit.loopexit ], [ %[[U_FAST]], %exit.loopexit1.cont ]
; APPLIED-NEXT:      %[[V_RES:[^ ]+]] = phi i64 [ %[[V_FB]], %exit.loopexit ], [ %[[V_FAST]], %exit.loopexit1.cont ]
; APPLIED-NEXT:      %r0 = getelementptr inbounds i64, ptr %red, i64 0
; APPLIED-NEXT:      store i64 %[[P_RES]], ptr %r0, align 8
; APPLIED-NEXT:      %r1 = getelementptr inbounds i64, ptr %red, i64 1
; APPLIED-NEXT:      store i64 %[[U_RES]], ptr %r1, align 8
; APPLIED-NEXT:      %r2 = getelementptr inbounds i64, ptr %red, i64 2
; APPLIED-NEXT:      store i64 %[[V_RES]], ptr %r2, align 8
; APPLIED-NEXT:      br label %ret
;
; APPLIED-LABEL: define {{.*}}@cand_w3(
; APPLIED:         entry:
; APPLIED-NEXT:      %pos = icmp sgt i64 %n, 0
; APPLIED-NEXT:      br i1 %pos, label %outer.header.lver.check, label %ret
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 3
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %p.next.lver.orig = phi i64 [ %p.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %u.next.lver.orig = phi i64 [ %u.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %v.next.lver.orig = phi i64 [ %v.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %dp.lver.orig = getelementptr inbounds [3 x i64], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %dv.lver.orig = load i64, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ii1.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %dn.lver.orig = add i64 %dv.lver.orig, %ii1.lver.orig
; APPLIED-NEXT:      store i64 %dn.lver.orig, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ec.lver.orig = load i64, ptr %epc, align 8
; APPLIED-NEXT:      %ecn.lver.orig = add i64 %ec.lver.orig, 1
; APPLIED-NEXT:      store i64 %ecn.lver.orig, ptr %epc, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_W3:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_W3:[0-9]+]]
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %p.res.ph2 = phi i64 [ %p.next, %inner.header.split ]
; APPLIED-NEXT:      %u.res.ph3 = phi i64 [ %u.next, %inner.header.split ]
; APPLIED-NEXT:      %v.res.ph4 = phi i64 [ %v.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr inbounds [3 x i64], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dv.epil = load i64, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ii1.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %dn.epil = add i64 %dv.epil, %ii1.epil
; APPLIED-NEXT:      store i64 %dn.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ec.epil = load i64, ptr %epc, align 8
; APPLIED-NEXT:      %ecn.epil = add i64 %ec.epil, 1
; APPLIED-NEXT:      store i64 %ecn.epil, ptr %epc, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %p.res = phi i64 [ %p.res.ph, %exit.loopexit ], [ %p.res.ph2, %exit.loopexit1.cont ]
; APPLIED-NEXT:      %u.res = phi i64 [ %u.res.ph, %exit.loopexit ], [ %u.res.ph3, %exit.loopexit1.cont ]
; APPLIED-NEXT:      %v.res = phi i64 [ %v.res.ph, %exit.loopexit ], [ %v.res.ph4, %exit.loopexit1.cont ]
; APPLIED-LABEL: define {{.*}}@cand_w7(
; APPLIED:         entry:
; APPLIED-NEXT:      %pos = icmp sgt i64 %n, 0
; APPLIED-NEXT:      br i1 %pos, label %outer.header.lver.check, label %ret
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 7
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %p.next.lver.orig = phi i64 [ %p.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %u.next.lver.orig = phi i64 [ %u.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %v.next.lver.orig = phi i64 [ %v.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %dp.lver.orig = getelementptr inbounds [7 x i64], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %dv.lver.orig = load i64, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ii1.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %dn.lver.orig = add i64 %dv.lver.orig, %ii1.lver.orig
; APPLIED-NEXT:      store i64 %dn.lver.orig, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ec.lver.orig = load i64, ptr %epc, align 8
; APPLIED-NEXT:      %ecn.lver.orig = add i64 %ec.lver.orig, 1
; APPLIED-NEXT:      store i64 %ecn.lver.orig, ptr %epc, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_W7:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_W7:[0-9]+]]
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %p.res.ph2 = phi i64 [ %p.next, %inner.header.split ]
; APPLIED-NEXT:      %u.res.ph3 = phi i64 [ %u.next, %inner.header.split ]
; APPLIED-NEXT:      %v.res.ph4 = phi i64 [ %v.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr inbounds [7 x i64], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dv.epil = load i64, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ii1.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %dn.epil = add i64 %dv.epil, %ii1.epil
; APPLIED-NEXT:      store i64 %dn.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ec.epil = load i64, ptr %epc, align 8
; APPLIED-NEXT:      %ecn.epil = add i64 %ec.epil, 1
; APPLIED-NEXT:      store i64 %ecn.epil, ptr %epc, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %p.res = phi i64 [ %p.res.ph, %exit.loopexit ], [ %p.res.ph2, %exit.loopexit1.cont ]
; APPLIED-NEXT:      %u.res = phi i64 [ %u.res.ph, %exit.loopexit ], [ %u.res.ph3, %exit.loopexit1.cont ]
; APPLIED-NEXT:      %v.res = phi i64 [ %v.res.ph, %exit.loopexit ], [ %v.res.ph4, %exit.loopexit1.cont ]
;
; The canonical leading dimension reaches the same shape with one reduction and
; the guard 'trip u> 1335'.
; APPLIED-LABEL: define {{.*}}@cand_canonical(
; APPLIED:         entry:
; APPLIED-NEXT:      %pos = icmp sgt i64 %n, 0
; APPLIED-NEXT:      br i1 %pos, label %outer.header.lver.check, label %ret
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 1335
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %s.next.lver.orig = phi i64 [ %s.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %dp.lver.orig = getelementptr inbounds [1335 x i64], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %dv.lver.orig = load i64, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ii1.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %dn.lver.orig = add i64 %dv.lver.orig, %ii1.lver.orig
; APPLIED-NEXT:      store i64 %dn.lver.orig, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_CANON:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_CANON:[0-9]+]]
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %s.res.ph2 = phi i64 [ %s.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr inbounds [1335 x i64], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dv.epil = load i64, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ii1.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %dn.epil = add i64 %dv.epil, %ii1.epil
; APPLIED-NEXT:      store i64 %dn.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %s.res = phi i64 [ %s.res.ph, %exit.loopexit ], [ %s.res.ph2, %exit.loopexit1.cont ]
; APPLIED-NEXT:      store i64 %s.res, ptr %sumout, align 8
; APPLIED-NEXT:      br label %ret
;
; The flattened-byte candidate keeps its byte arithmetic in the extracted
; epilogue.
; APPLIED-LABEL: define {{.*}}@cand_byte_w4(
; APPLIED:         entry:
; APPLIED-NEXT:      %pos = icmp sgt i64 %n, 0
; APPLIED-NEXT:      br i1 %pos, label %outer.header.lver.check, label %ret
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 4
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %p.next.lver.orig = phi i64 [ %p.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %u.next.lver.orig = phi i64 [ %u.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %v.next.lver.orig = phi i64 [ %v.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %diagoff.lver.orig = mul i64 %i.lver.orig, 40
; APPLIED-NEXT:      %dp.lver.orig = getelementptr i8, ptr %A, i64 %diagoff.lver.orig
; APPLIED-NEXT:      %dv.lver.orig = load i64, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ii1.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %dn.lver.orig = add i64 %dv.lver.orig, %ii1.lver.orig
; APPLIED-NEXT:      store i64 %dn.lver.orig, ptr %dp.lver.orig, align 8
; APPLIED-NEXT:      %ec.lver.orig = load i64, ptr %epc, align 8
; APPLIED-NEXT:      %ecn.lver.orig = add i64 %ec.lver.orig, 1
; APPLIED-NEXT:      store i64 %ecn.lver.orig, ptr %epc, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_BYTE:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_BYTE:[0-9]+]]
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %p.res.ph2 = phi i64 [ %p.next, %inner.header.split ]
; APPLIED-NEXT:      %u.res.ph3 = phi i64 [ %u.next, %inner.header.split ]
; APPLIED-NEXT:      %v.res.ph4 = phi i64 [ %v.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %diagoff.epil = mul i64 %epilogue.iv, 40
; APPLIED-NEXT:      %dp.epil = getelementptr i8, ptr %A, i64 %diagoff.epil
; APPLIED-NEXT:      %dv.epil = load i64, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ii1.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %dn.epil = add i64 %dv.epil, %ii1.epil
; APPLIED-NEXT:      store i64 %dn.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ec.epil = load i64, ptr %epc, align 8
; APPLIED-NEXT:      %ecn.epil = add i64 %ec.epil, 1
; APPLIED-NEXT:      store i64 %ecn.epil, ptr %epc, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %p.res = phi i64 [ %p.res.ph, %exit.loopexit ], [ %p.res.ph2, %exit.loopexit1.cont ]
; APPLIED-NEXT:      %u.res = phi i64 [ %u.res.ph, %exit.loopexit ], [ %u.res.ph3, %exit.loopexit1.cont ]
; APPLIED-NEXT:      %v.res = phi i64 [ %v.res.ph, %exit.loopexit ], [ %v.res.ph4, %exit.loopexit1.cont ]
;
; Every attachment the file carries is one of the ten matched above: a versioned
; and a fallback marker for each of the five applied functions. Each is a
; distinct loop id whose second operand is the shared marker name.
; APPLIED:         ![[FB_W2]] = distinct !{![[FB_W2]], ![[MARK:[0-9]+]]}
; APPLIED-NEXT:    ![[MARK]] = !{!"llvm.loop.interchange.runtime_versioned"}
; APPLIED-NEXT:    ![[FAST_W2]] = distinct !{![[FAST_W2]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_W3]] = distinct !{![[FB_W3]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_W3]] = distinct !{![[FAST_W3]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_W7]] = distinct !{![[FB_W7]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_W7]] = distinct !{![[FAST_W7]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_CANON]] = distinct !{![[FB_CANON]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_CANON]] = distinct !{![[FAST_CANON]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_BYTE]] = distinct !{![[FB_BYTE]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_BYTE]] = distinct !{![[FAST_BYTE]], ![[MARK]]}
;
; The same loop-info lines are checked against the loop manager's live
; LoopInfo and against a fresh rebuild of the transformed module. The block
; lists inside a nest are wildcarded between the header and the latch because
; LoopInfo stores them in discovery order, which an incremental update and a
; fresh walk reach differently. The loop order, the depths, the headers, the
; latches, and the epilogue loop's two blocks are exact. Every versioned
; function leaves the sibling order fallback, epilogue, versioned.
;
; APPLY-LOOPS-LABEL: Loop info for function 'cand_w1':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'cand_w2':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'cand_w3':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'cand_w7':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'cand_canonical':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'cand_byte_w4':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;-------------------------------------------------------------------------------
; Guarded outer-epilogue candidates. Row stride W is recovered from the exact
; byte coefficients of the column read and the diagonal write. The four
; candidates below spell those coefficients with typed `[W x i64]` GEPs (a
; typed A[j][i] column read and a typed A[i][i] diagonal write). @cand_byte_w4
; below spells the same relation as explicit byte arithmetic. The selected
; outer trip and the inner trip are the same runtime %n, so each candidate
; exposes one ExactTrip and one predicate `n u> W`. N and E share %A, while
; %A, %red, and %epc are mutually noalias. The E-to-N dependence is real: For
; trip n > W the diagonal write A[i][i] is read back by a later column read.
; Three exact integer reductions (%p sum, %u sum of x<<1, %v sum of x*3)
; escape, and E counts its iterations in %epc. For W>1, the versioned loop runs
; when 0<n<=W and the fallback clone runs when n>W. W=1 is unchanged because
; interchange is not profitable.
;-------------------------------------------------------------------------------
define void @cand_w1(ptr noalias dereferenceable(8) %A, i64 %n,
                     ptr noalias %red, ptr noalias %epc) {
entry:
  %pos = icmp sgt i64 %n, 0
  br i1 %pos, label %outer.ph, label %ret
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi i64 [ 0, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi i64 [ 0, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi i64 [ 0, %outer.ph ], [ %v.next, %outer.latch ]
  %col = getelementptr inbounds [1 x i64], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi i64 [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi i64 [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi i64 [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %addr = getelementptr inbounds [1 x i64], ptr %col, i64 %j, i64 0
  %x = load i64, ptr %addr, align 8
  %p.next.j = add i64 %p.j, %x
  %x2 = shl i64 %x, 1
  %u.next.j = add i64 %u.j, %x2
  %x3 = mul i64 %x, 3
  %v.next.j = add i64 %v.j, %x3
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %n
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi i64 [ %p.next.j, %inner.header ]
  %u.next = phi i64 [ %u.next.j, %inner.header ]
  %v.next = phi i64 [ %v.next.j, %inner.header ]
  %dp = getelementptr inbounds [1 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %dp, align 8
  %ii1 = add i64 %i, 1
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  %ec = load i64, ptr %epc, align 8
  %ecn = add i64 %ec, 1
  store i64 %ecn, ptr %epc, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi i64 [ %p.next, %outer.latch ]
  %u.res = phi i64 [ %u.next, %outer.latch ]
  %v.res = phi i64 [ %v.next, %outer.latch ]
  %r0 = getelementptr inbounds i64, ptr %red, i64 0
  store i64 %p.res, ptr %r0, align 8
  %r1 = getelementptr inbounds i64, ptr %red, i64 1
  store i64 %u.res, ptr %r1, align 8
  %r2 = getelementptr inbounds i64, ptr %red, i64 2
  store i64 %v.res, ptr %r2, align 8
  br label %ret
ret:
  ret void
}


define void @cand_w2(ptr noalias dereferenceable(32) %A, i64 %n,
                     ptr noalias %red, ptr noalias %epc) {
entry:
  %pos = icmp sgt i64 %n, 0
  br i1 %pos, label %outer.ph, label %ret
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi i64 [ 0, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi i64 [ 0, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi i64 [ 0, %outer.ph ], [ %v.next, %outer.latch ]
  %col = getelementptr inbounds [2 x i64], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi i64 [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi i64 [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi i64 [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %addr = getelementptr inbounds [2 x i64], ptr %col, i64 %j, i64 0
  %x = load i64, ptr %addr, align 8
  %p.next.j = add i64 %p.j, %x
  %x2 = shl i64 %x, 1
  %u.next.j = add i64 %u.j, %x2
  %x3 = mul i64 %x, 3
  %v.next.j = add i64 %v.j, %x3
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %n
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi i64 [ %p.next.j, %inner.header ]
  %u.next = phi i64 [ %u.next.j, %inner.header ]
  %v.next = phi i64 [ %v.next.j, %inner.header ]
  %dp = getelementptr inbounds [2 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %dp, align 8
  %ii1 = add i64 %i, 1
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  %ec = load i64, ptr %epc, align 8
  %ecn = add i64 %ec, 1
  store i64 %ecn, ptr %epc, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi i64 [ %p.next, %outer.latch ]
  %u.res = phi i64 [ %u.next, %outer.latch ]
  %v.res = phi i64 [ %v.next, %outer.latch ]
  %r0 = getelementptr inbounds i64, ptr %red, i64 0
  store i64 %p.res, ptr %r0, align 8
  %r1 = getelementptr inbounds i64, ptr %red, i64 1
  store i64 %u.res, ptr %r1, align 8
  %r2 = getelementptr inbounds i64, ptr %red, i64 2
  store i64 %v.res, ptr %r2, align 8
  br label %ret
ret:
  ret void
}


define void @cand_w3(ptr noalias dereferenceable(72) %A, i64 %n,
                     ptr noalias %red, ptr noalias %epc) {
entry:
  %pos = icmp sgt i64 %n, 0
  br i1 %pos, label %outer.ph, label %ret
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi i64 [ 0, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi i64 [ 0, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi i64 [ 0, %outer.ph ], [ %v.next, %outer.latch ]
  %col = getelementptr inbounds [3 x i64], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi i64 [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi i64 [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi i64 [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %addr = getelementptr inbounds [3 x i64], ptr %col, i64 %j, i64 0
  %x = load i64, ptr %addr, align 8
  %p.next.j = add i64 %p.j, %x
  %x2 = shl i64 %x, 1
  %u.next.j = add i64 %u.j, %x2
  %x3 = mul i64 %x, 3
  %v.next.j = add i64 %v.j, %x3
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %n
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi i64 [ %p.next.j, %inner.header ]
  %u.next = phi i64 [ %u.next.j, %inner.header ]
  %v.next = phi i64 [ %v.next.j, %inner.header ]
  %dp = getelementptr inbounds [3 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %dp, align 8
  %ii1 = add i64 %i, 1
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  %ec = load i64, ptr %epc, align 8
  %ecn = add i64 %ec, 1
  store i64 %ecn, ptr %epc, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi i64 [ %p.next, %outer.latch ]
  %u.res = phi i64 [ %u.next, %outer.latch ]
  %v.res = phi i64 [ %v.next, %outer.latch ]
  %r0 = getelementptr inbounds i64, ptr %red, i64 0
  store i64 %p.res, ptr %r0, align 8
  %r1 = getelementptr inbounds i64, ptr %red, i64 1
  store i64 %u.res, ptr %r1, align 8
  %r2 = getelementptr inbounds i64, ptr %red, i64 2
  store i64 %v.res, ptr %r2, align 8
  br label %ret
ret:
  ret void
}


define void @cand_w7(ptr noalias dereferenceable(392) %A, i64 %n,
                     ptr noalias %red, ptr noalias %epc) {
entry:
  %pos = icmp sgt i64 %n, 0
  br i1 %pos, label %outer.ph, label %ret
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi i64 [ 0, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi i64 [ 0, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi i64 [ 0, %outer.ph ], [ %v.next, %outer.latch ]
  %col = getelementptr inbounds [7 x i64], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi i64 [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi i64 [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi i64 [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %addr = getelementptr inbounds [7 x i64], ptr %col, i64 %j, i64 0
  %x = load i64, ptr %addr, align 8
  %p.next.j = add i64 %p.j, %x
  %x2 = shl i64 %x, 1
  %u.next.j = add i64 %u.j, %x2
  %x3 = mul i64 %x, 3
  %v.next.j = add i64 %v.j, %x3
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %n
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi i64 [ %p.next.j, %inner.header ]
  %u.next = phi i64 [ %u.next.j, %inner.header ]
  %v.next = phi i64 [ %v.next.j, %inner.header ]
  %dp = getelementptr inbounds [7 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %dp, align 8
  %ii1 = add i64 %i, 1
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  %ec = load i64, ptr %epc, align 8
  %ecn = add i64 %ec, 1
  store i64 %ecn, ptr %epc, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi i64 [ %p.next, %outer.latch ]
  %u.res = phi i64 [ %u.next, %outer.latch ]
  %v.res = phi i64 [ %v.next, %outer.latch ]
  %r0 = getelementptr inbounds i64, ptr %red, i64 0
  store i64 %p.res, ptr %r0, align 8
  %r1 = getelementptr inbounds i64, ptr %red, i64 1
  store i64 %u.res, ptr %r1, align 8
  %r2 = getelementptr inbounds i64, ptr %red, i64 2
  store i64 %v.res, ptr %r2, align 8
  br label %ret
ret:
  ret void
}

;-------------------------------------------------------------------------------
; This test case uses the canonical leading dimension. Row stride 1335 is
; recovered from the inner byte coefficient of the `[1335 x i64]` GEPs A[j][i]
; and A[i][i], divided once by the 8-byte element size. Both trips are the
; runtime %n (one ExactTrip), so a positive trip of at most 1335 takes the
; versioned loop and a larger one takes the fallback. %A carries the exact
; minimum accessible-byte bound 1335*1335*8 == 14257800.
;-------------------------------------------------------------------------------
define void @cand_canonical(ptr noalias dereferenceable(14257800) %A, i64 %n,
                            ptr noalias %sumout) {
entry:
  %pos = icmp sgt i64 %n, 0
  br i1 %pos, label %outer.ph, label %ret
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %s.i = phi i64 [ 0, %outer.ph ], [ %s.next, %outer.latch ]
  %col = getelementptr inbounds [1335 x i64], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %s.j = phi i64 [ %s.i, %outer.header ], [ %s.next.j, %inner.header ]
  %addr = getelementptr inbounds [1335 x i64], ptr %col, i64 %j, i64 0
  %x = load i64, ptr %addr, align 8
  %s.next.j = add i64 %s.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %n
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %s.next = phi i64 [ %s.next.j, %inner.header ]
  %dp = getelementptr inbounds [1335 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %dp, align 8
  %ii1 = add i64 %i, 1
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %s.res = phi i64 [ %s.next, %outer.latch ]
  store i64 %s.res, ptr %sumout, align 8
  br label %ret
ret:
  ret void
}


;-------------------------------------------------------------------------------
; Flattened-byte candidate, semantically identical to a W=4 member of the family
; above. Every address is built from explicit `mul` + `getelementptr i8` byte
; arithmetic, without an array type, a multi-index GEP, or a source element
; type in the addressing. W == 4 comes solely from the raw inner byte coefficient
; 32 divided exactly once by the 8-byte element size, and the diagonal byte
; coefficient 40 == 8*(4 + 1) agrees with the column coefficient 8 modulo W.
;
; Its accessible-byte lower bound is the exact minimum 4*4*8 == 128 that its
; accepted 4-by-4 byte range needs. Because both trips are the same runtime %n,
; this is a full runtime-bound candidate: it gets the same single `trip u> 4`
; guard, the same whole-loop fallback clone, and the same extracted epilogue
; after the versioned loop.
;-------------------------------------------------------------------------------
define void @cand_byte_w4(ptr noalias dereferenceable(128) %A, i64 %n,
                          ptr noalias %red, ptr noalias %epc) {
entry:
  %pos = icmp sgt i64 %n, 0
  br i1 %pos, label %outer.ph, label %ret
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi i64 [ 0, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi i64 [ 0, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi i64 [ 0, %outer.ph ], [ %v.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr i8, ptr %A, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi i64 [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi i64 [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi i64 [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %rowoff = mul i64 %j, 32
  %addr = getelementptr i8, ptr %col, i64 %rowoff
  %x = load i64, ptr %addr, align 8
  %p.next.j = add i64 %p.j, %x
  %x2 = shl i64 %x, 1
  %u.next.j = add i64 %u.j, %x2
  %x3 = mul i64 %x, 3
  %v.next.j = add i64 %v.j, %x3
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %n
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi i64 [ %p.next.j, %inner.header ]
  %u.next = phi i64 [ %u.next.j, %inner.header ]
  %v.next = phi i64 [ %v.next.j, %inner.header ]
  %diagoff = mul i64 %i, 40
  %dp = getelementptr i8, ptr %A, i64 %diagoff
  %dv = load i64, ptr %dp, align 8
  %ii1 = add i64 %i, 1
  %dn = add i64 %dv, %ii1
  store i64 %dn, ptr %dp, align 8
  %ec = load i64, ptr %epc, align 8
  %ecn = add i64 %ec, 1
  store i64 %ecn, ptr %epc, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi i64 [ %p.next, %outer.latch ]
  %u.res = phi i64 [ %u.next, %outer.latch ]
  %v.res = phi i64 [ %v.next, %outer.latch ]
  %r0 = getelementptr inbounds i64, ptr %red, i64 0
  store i64 %p.res, ptr %r0, align 8
  %r1 = getelementptr inbounds i64, ptr %red, i64 1
  store i64 %u.res, ptr %r1, align 8
  %r2 = getelementptr inbounds i64, ptr %red, i64 2
  store i64 %v.res, ptr %r2, align 8
  br label %ret
ret:
  ret void
}
