; NOTE: Do not autogenerate
;
; Runtime-bound outer-epilogue candidates. Every selected outer loop here has a
; runtime trip count and one span requirement that only a runtime check can
; discharge, so the pass versions the loop behind a single unsigned trip guard.
; The versioned copy gets its proven epilogue distributed into a sibling loop
; and its reduction nest interchanged. The fallback clone keeps the original
; loop. The remaining test cases cover the checks that reject or skip a
; candidate.
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
; DEFINE:   --func=version_toplevel_dynamic --func=version_nested_dynamic \
; DEFINE:   --func=version_fabs_epilogue --func=version_blocks_16 \
; DEFINE:   --func=version_insts_128
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
; Lowering the block limit from 16 to 15 rejects version_blocks_16 without
; changing it.
; RUN: llvm-extract -func=version_blocks_16 -S %s -o %t.b16.ll
; RUN: opt -S -passes=no-op-loopnest %t.b16.ll -o %t.b16.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} -loop-interchange-max-runtime-versioning-blocks=15 %t.b16.ll -o %t.b16.max15 2> %t.b16.max15.stderr
; RUN: diff -u %t.b16.noop %t.b16.max15
; RUN: FileCheck %s --check-prefix=LIMIT-BLOCKS --input-file=%t.b16.max15.stderr --implicit-check-not='loop-interchange:'
;
; Lowering the instruction limit from 128 to 127 rejects version_insts_128
; without changing it.
; RUN: llvm-extract -func=version_insts_128 -S %s -o %t.i128.ll
; RUN: opt -S -passes=no-op-loopnest %t.i128.ll -o %t.i128.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} -loop-interchange-max-runtime-versioning-instructions=127 %t.i128.ll -o %t.i128.max127 2> %t.i128.max127.stderr
; RUN: diff -u %t.i128.noop %t.i128.max127
; RUN: FileCheck %s --check-prefix=LIMIT-INSTS --input-file=%t.i128.max127.stderr --implicit-check-not='loop-interchange:'
;
; A zero expansion budget rejects the guard during classification, so the plan
; never reaches the runtime versioning checks.
; RUN: llvm-extract -func=version_toplevel_dynamic -S %s -o %t.top.ll
; RUN: opt -S -passes=no-op-loopnest %t.top.ll -o %t.top.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} -loop-interchange-runtime-trip-expansion-budget=0 %t.top.ll -o %t.top.budget0 2> %t.top.budget0.stderr
; RUN: diff -u %t.top.noop %t.top.budget0
; RUN: FileCheck %s --check-prefix=BUDGET0 --input-file=%t.top.budget0.stderr --implicit-check-not='loop-interchange:'
;
; With runtime versioning off, the same positive is rejected with the disabled
; reason and nothing changes.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-outer-epilogue-fission -loop-interchange-print-prepared-plan -loop-interchange-outer-epilogue-runtime-versioning=false %t.top.ll -o %t.top.runtimeoff 2> %t.top.runtimeoff.stderr
; RUN: diff -u %t.top.noop %t.top.runtimeoff
; RUN: FileCheck %s --check-prefix=RUNTIME-OFF --input-file=%t.top.runtimeoff.stderr --implicit-check-not='loop-interchange:'
;
; A SCEV cache primed before the transform must agree with a fresh computation
; over the transformed function, for a top-level and for a nested candidate.
; Each transformed module is checked to be versioned, so neither comparison can
; pass over an untransformed function.
; RUN: opt -disable-output -passes='print<scalar-evolution>' %t.top.ll 2> %t.top.before
; RUN: opt -disable-output -cache-line-size=64 -verify-scev %{policy} -loop-interchange-outer-epilogue-fission -loop-interchange-outer-epilogue-runtime-versioning -passes='print<scalar-evolution>,loop-interchange,print<scalar-evolution>' %t.top.ll 2> %t.top.primed
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 -verify-scev %{policy} -loop-interchange-outer-epilogue-fission -loop-interchange-outer-epilogue-runtime-versioning %t.top.ll -o %t.top.transformed
; RUN: opt -disable-output -passes='print<scalar-evolution>' %t.top.transformed 2> %t.top.after
; RUN: cat %t.top.before %t.top.after > %t.top.expected
; RUN: diff -u %t.top.expected %t.top.primed
; RUN: FileCheck %s --check-prefix=VERSIONED-TOP --input-file=%t.top.transformed
; RUN: llvm-extract -func=version_nested_dynamic -S %s -o %t.nested.ll
; RUN: opt -disable-output -passes='print<scalar-evolution>' %t.nested.ll 2> %t.nested.before
; RUN: opt -disable-output -cache-line-size=64 -verify-scev %{policy} -loop-interchange-outer-epilogue-fission -loop-interchange-outer-epilogue-runtime-versioning -passes='print<scalar-evolution>,loop-interchange,print<scalar-evolution>' %t.nested.ll 2> %t.nested.primed
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 -verify-scev %{policy} -loop-interchange-outer-epilogue-fission -loop-interchange-outer-epilogue-runtime-versioning %t.nested.ll -o %t.nested.transformed
; RUN: opt -disable-output -passes='print<scalar-evolution>' %t.nested.transformed 2> %t.nested.after
; RUN: cat %t.nested.before %t.nested.after > %t.nested.expected
; RUN: diff -u %t.nested.expected %t.nested.primed
; RUN: FileCheck %s --check-prefix=VERSIONED-NESTED --input-file=%t.nested.transformed
;
; Each printed plan has passed both dependence matrices, legality,
; profitability, validation, and the runtime versioning checks. The two
; remarks after it come from the apply step.
; PREP-LABEL: loop-interchange: prepared function=version_toplevel_dynamic{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=6
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[TOP_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[TOP_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
; PREP-LABEL: loop-interchange: prepared function=version_nested_dynamic{{ }}
; PREP-SAME:  outer=i.header inner=j.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[NESTED_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[NESTED_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
;
; Whole-loop cloning rejects a noduplicate call, an indirect branch, and a
; token-like live-out.
; PREP-LABEL: loop-interchange: rejected function=version_noduplicate_epilogue{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop is not safe to clone{{$}}
; PREP-LABEL: loop-interchange: rejected function=version_indirectbr{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop is not safe to clone{{$}}
; PREP-LABEL: loop-interchange: rejected function=version_tokenlike_liveout{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop is not safe to clone{{$}}
; A token-like value that is not a reachable live-out passes conditional clone
; safety but not the versioning control scan.
; PREP-LABEL: loop-interchange: rejected function=version_tokenlike_unreachable_use{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop contains unsafe versioning control{{$}}
; PREP-LABEL: loop-interchange: rejected function=version_tokenlike_nouse{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop contains unsafe versioning control{{$}}
; A live-out whose only user is unreachable lacks the LCSSA PHI needed for
; versioning.
; PREP-LABEL: loop-interchange: rejected function=version_liveout_unreachable_use{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop has a live-out used in unreachable code{{$}}
; PREP-LABEL: loop-interchange: rejected function=version_liveout_unreachable_phi_edge{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop has a live-out used in unreachable code{{$}}
;
; A clone-safe intrinsic does not block versioning.
; PREP-LABEL: loop-interchange: prepared function=version_fabs_epilogue{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[FABS_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[FABS_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
;
; At each limit the accepted side is prepared, and the side one step over is
; rejected with its own reason.
; PREP-LABEL: loop-interchange: prepared function=version_blocks_16{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=13 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[BLOCKS_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[BLOCKS_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
; PREP-LABEL: loop-interchange: rejected function=version_blocks_17{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop exceeds the runtime block limit{{$}}
; PREP-LABEL: loop-interchange: prepared function=version_insts_128{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=2 epilogue-insts=110 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[INSTS_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[INSTS_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
; PREP-LABEL: loop-interchange: rejected function=version_insts_129{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=selected outer loop exceeds the runtime instruction limit{{$}}
;
; A resampled freeze in the selected outer header fails the existing legality
; check on the virtual extracted nest, before the versioning checks run.
; PREP-LABEL: loop-interchange: rejected function=version_outer_header_freeze{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality{{$}}
;
; The marked, optsize, and minsize test cases do not print a
; 'loop-interchange:' line. Discovery skips the marked loop, and the size check
; skips the other two before preparation.
;
; The versioned module. Every definition is listed, so the RUN line's
; '{{^define }}' exclusion makes the list exhaustive. Each applied function
; gains one guard block, one untransformed fallback clone, one distributed and
; interchanged versioned nest, and one sibling epilogue loop on that path. The
; fallback clone keeps the whole original epilogue, so its epilogue block is
; checked instruction by instruction, including the inner loop's exit into it
; and its stores. The two limit-boundary positives carry large bodies, so their
; fallback epilogue is checked for its label, every store, and its terminator.
; That same invocation also excludes every '!llvm.loop' attachment and every
; '.lver.check' label definition it does not match here, so the two markers per
; applied function and the single guard per applied function are exhaustive
; too. Block labels are plain directives because the printed IR separates
; blocks with a blank line. The functions that were left alone were already
; shown unchanged by the llvm-extract --delete pair above and carry only their
; label.

;
; APPLIED-LABEL: define {{.*}}@version_toplevel_dynamic(
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 4
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %p.next.lver.orig = phi double [ %p.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %u.next.lver.orig = phi double [ %u.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %v.next.lver.orig = phi double [ %v.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %ddiag.lver.orig = getelementptr inbounds [4 x double], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %dval.lver.orig = load double, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      %dnew.lver.orig = fmul double %dval.lver.orig, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.lver.orig, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_TOP:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_TOP:[0-9]+]]
; APPLIED:         exit.loopexit:
; APPLIED-NEXT:      %[[P_FB:[^ ]+]] = phi double [ %p.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      %[[U_FB:[^ ]+]] = phi double [ %u.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      %[[V_FB:[^ ]+]] = phi double [ %v.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      br label %exit
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %[[P_FAST:[^ ]+]] = phi double [ %p.next, %inner.header.split ]
; APPLIED-NEXT:      %[[U_FAST:[^ ]+]] = phi double [ %u.next, %inner.header.split ]
; APPLIED-NEXT:      %[[V_FAST:[^ ]+]] = phi double [ %v.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds [4 x double], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         exit.loopexit1.cont:
; APPLIED-NEXT:      br label %exit
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %[[P_RES:[^ ]+]] = phi double [ %[[P_FB]], %exit.loopexit ], [ %[[P_FAST]], %exit.loopexit1.cont ]
; APPLIED-NEXT:      %[[U_RES:[^ ]+]] = phi double [ %[[U_FB]], %exit.loopexit ], [ %[[U_FAST]], %exit.loopexit1.cont ]
; APPLIED-NEXT:      %[[V_RES:[^ ]+]] = phi double [ %[[V_FB]], %exit.loopexit ], [ %[[V_FAST]], %exit.loopexit1.cont ]
; APPLIED-NEXT:      %r1 = getelementptr inbounds double, ptr %R, i64 1
; APPLIED-NEXT:      %r2 = getelementptr inbounds double, ptr %R, i64 2
; APPLIED-NEXT:      store double %[[P_RES]], ptr %R, align 8
; APPLIED-NEXT:      store double %[[U_RES]], ptr %r1, align 8
; APPLIED-NEXT:      store double %[[V_RES]], ptr %r2, align 8
; APPLIED-NEXT:      ret void
;
; The nested candidate expands its trip once in the function entry and keeps
; the enclosing loop's phi in the guard block.
; APPLIED-LABEL: define {{.*}}@version_nested_dynamic(
; APPLIED:         entry:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      br label %i.header.lver.check
; APPLIED:         i.header.lver.check:
; APPLIED-NEXT:      %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 4
; APPLIED-NEXT:      br i1 %ident.check, label %i.header.ph.lver.orig, label %j.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %i.epilogue.lver.orig, label %j.header.lver.orig
; APPLIED:         i.epilogue.lver.orig:
; APPLIED-NEXT:      %chk.next.lver.orig = phi double [ %chk.next.j.lver.orig, %j.header.lver.orig ]
; APPLIED-NEXT:      %ddiag.lver.orig = getelementptr inbounds [4 x double], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %dval.lver.orig = load double, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      %dnew.lver.orig = fmul double %dval.lver.orig, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.lver.orig, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      br label %i.latch.lver.orig
; APPLIED:         i.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %i.exit.loopexit, label %i.header.lver.orig, !llvm.loop ![[FB_NESTED:[0-9]+]]
; APPLIED:         {{^i[.]latch:}}
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %j.header.split, label %i.header, !llvm.loop ![[FAST_NESTED:[0-9]+]]
; APPLIED:         i.exit.loopexit1:
; APPLIED-NEXT:      %chk.lcssa.ph2 = phi double [ %chk.next, %j.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds [4 x double], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %i.exit.loopexit1.cont, label %epilogue.header
; APPLIED:         i.exit.loopexit1.cont:
; APPLIED-NEXT:      br label %i.exit
; APPLIED:         {{^i[.]exit:}}
; APPLIED-NEXT:      %chk.lcssa = phi double [ %chk.lcssa.ph, %i.exit.loopexit ], [ %chk.lcssa.ph2, %i.exit.loopexit1.cont ]
; APPLIED-NEXT:      br label %sibling.header
;
; APPLIED-LABEL: define {{.*}}@version_noduplicate_epilogue(
; APPLIED-LABEL: define {{.*}}@version_indirectbr(
; APPLIED-LABEL: define {{.*}}@version_tokenlike_liveout(
; APPLIED-LABEL: define {{.*}}@version_tokenlike_unreachable_use(
; APPLIED-LABEL: define {{.*}}@version_tokenlike_nouse(
; APPLIED-LABEL: define {{.*}}@version_liveout_unreachable_use(
; APPLIED-LABEL: define {{.*}}@version_liveout_unreachable_phi_edge(
; APPLIED-LABEL: define {{.*}}@version_fabs_epilogue(
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 4
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %chk.next.lver.orig = phi double [ %chk.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %ddiag.lver.orig = getelementptr inbounds [4 x double], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %dval.lver.orig = load double, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      %dnew.lver.orig = fmul double %dval.lver.orig, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.lver.orig, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_FABS:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_FABS:[0-9]+]]
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %chk.res.ph2 = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds [4 x double], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         exit.loopexit1.cont:
; APPLIED-NEXT:      br label %exit
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %chk.res = phi double [ %chk.res.ph, %exit.loopexit ], [ %chk.res.ph2, %exit.loopexit1.cont ]
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
;
; The already-versioned test case keeps its input loop attachment and gains
; nothing else.
; APPLIED-LABEL: define {{.*}}@version_marked(
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %exit, label %outer.header, !llvm.loop ![[MARKED:[0-9]+]]
; APPLIED-LABEL: define {{.*}}@version_optsize(
; APPLIED-LABEL: define {{.*}}@version_minsize(
;
; The two limit-boundary positives carry large bodies, so they are checked for
; the guard, its polarity, both markers, the fallback epilogue's store and
; terminator, and the epilogue loop's two blocks.
; APPLIED-LABEL: define {{.*}}@version_blocks_16(
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 4
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED:           store double %dnew.lver.orig, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      br label %pad01.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_BLOCKS:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_BLOCKS:[0-9]+]]
; APPLIED:         epilogue.header:
; APPLIED:         epilogue.latch:
; APPLIED-LABEL: define {{.*}}@version_blocks_17(
; APPLIED-LABEL: define {{.*}}@version_insts_128(
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 4
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED:           store double %dnew.lver.orig, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      br label %pad.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_INSTS:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_INSTS:[0-9]+]]
; APPLIED:         epilogue.header:
; APPLIED:         epilogue.latch:
; APPLIED-LABEL: define {{.*}}@version_insts_129(
; APPLIED-LABEL: define {{.*}}@version_outer_header_freeze(
;
; Every attachment the file carries is one of the eleven matched above: a
; versioned and a fallback marker for each of the five applied functions, plus
; the input attachment of @version_marked. Each is a distinct loop id whose
; second operand is the shared marker name.
; APPLIED:         ![[FB_TOP]] = distinct !{![[FB_TOP]], ![[MARK:[0-9]+]]}
; APPLIED-NEXT:    ![[MARK]] = !{!"llvm.loop.interchange.runtime_versioned"}
; APPLIED-NEXT:    ![[FAST_TOP]] = distinct !{![[FAST_TOP]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_NESTED]] = distinct !{![[FB_NESTED]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_NESTED]] = distinct !{![[FAST_NESTED]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_FABS]] = distinct !{![[FB_FABS]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_FABS]] = distinct !{![[FAST_FABS]], ![[MARK]]}
; APPLIED-NEXT:    ![[MARKED]] = distinct !{![[MARKED]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_BLOCKS]] = distinct !{![[FB_BLOCKS]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_BLOCKS]] = distinct !{![[FAST_BLOCKS]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_INSTS]] = distinct !{![[FB_INSTS]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_INSTS]] = distinct !{![[FAST_INSTS]], ![[MARK]]}
;
; The same loop-info lines are checked against the loop manager's live
; LoopInfo and against a fresh rebuild of the transformed module. The block
; lists inside a nest are wildcarded between the header and the latch because
; LoopInfo stores them in discovery order, which an incremental update and a
; fresh walk reach differently. The loop order, the depths, the headers, the
; latches, and the epilogue loop's two blocks are exact. A top-level candidate
; leaves the sibling order fallback, epilogue, versioned. The nested candidate
; leaves versioned, epilogue, fallback, then the unrelated trailing sibling.
;
; APPLY-LOOPS-LABEL: Loop info for function 'version_toplevel_dynamic':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_nested_dynamic':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %i.header.lver.check<header>,{{.*}}%k.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %j.header<header>,{{.*}}%j.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:         Loop at depth 3 containing: %i.header<header>,{{.*}}%i.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %i.header.lver.orig<header>,{{.*}}%i.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:         Loop at depth 3 containing: %j.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %sibling.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_noduplicate_epilogue':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_indirectbr':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header>,{{.*}}%inner.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_fabs_epilogue':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_marked':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_optsize':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_minsize':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_blocks_16':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_blocks_17':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_insts_128':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_insts_129':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'version_outer_header_freeze':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header><latch><exiting>
;
; Each lowered limit rejects the test case that sits exactly at the default
; limit, with its own reason, and changes nothing.
; LIMIT-BLOCKS: loop-interchange: rejected function=version_blocks_16{{ }}
; LIMIT-BLOCKS-SAME: outer=outer.header inner=inner.header reason=selected outer loop exceeds the runtime block limit{{$}}
; LIMIT-INSTS: loop-interchange: rejected function=version_insts_128{{ }}
; LIMIT-INSTS-SAME: outer=outer.header inner=inner.header reason=selected outer loop exceeds the runtime instruction limit{{$}}
; BUDGET0: loop-interchange: rejected function=version_toplevel_dynamic{{ }}
; BUDGET0-SAME: outer=outer.header inner=inner.header reason=runtime-trip-expansion-too-costly{{$}}
; RUNTIME-OFF: loop-interchange: rejected function=version_toplevel_dynamic{{ }}
; RUNTIME-OFF-SAME: outer=outer.header inner=inner.header reason=runtime outer-epilogue versioning is disabled{{$}}
;
; The two SCEV comparisons run over versioned functions.
; VERSIONED-TOP: .lver.check
; VERSIONED-NESTED: .lver.check

target datalayout = "e-m:e-p3:16:16-i64:64-f80:128-n8:16:32:64-S128"

;-------------------------------------------------------------------------------
; Top-level dynamic candidate: runtime outer trip, three reassociated reductions
; escaping to a shared exit (three live-outs). The versioned/fallback join must
; repair all three live-outs.
;-------------------------------------------------------------------------------
define void @version_toplevel_dynamic(ptr noalias dereferenceable(128) %A,
                                      ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi double [ 0.000000e+00, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi double [ 0.000000e+00, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi double [ 0.000000e+00, %outer.ph ], [ %v.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi double [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi double [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi double [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %p.next.j = fadd reassoc double %p.j, %x
  %u.next.j = fadd reassoc double %u.j, %x
  %v.next.j = fadd reassoc double %v.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi double [ %p.next.j, %inner.header ]
  %u.next = phi double [ %u.next.j, %inner.header ]
  %v.next = phi double [ %v.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi double [ %p.next, %outer.latch ]
  %u.res = phi double [ %u.next, %outer.latch ]
  %v.res = phi double [ %v.next, %outer.latch ]
  %r1 = getelementptr inbounds double, ptr %R, i64 1
  %r2 = getelementptr inbounds double, ptr %R, i64 2
  store double %p.res, ptr %R, align 8
  store double %u.res, ptr %r1, align 8
  store double %v.res, ptr %r2, align 8
  ret void
}

; Versioned-loop live-outs arrive after E. Fallback live-outs arrive through
; their dedicated exit.

;-------------------------------------------------------------------------------
; Nested dynamic candidate: the selected loop %i.header is the middle of a
; three-deep nest, imperfect (diagonal epilogue) so the inner (j,i) pair is not
; interchangeable, and the enclosing %k.header is imperfect too (it stores the
; partial result). The middle loop is versioned and the clone becomes a sibling
; under the enclosing loop. The selected loop's preheader is %k.header itself,
; which has entry and backedge predecessors. An unrelated trailing sibling
; makes the required versioned, E, fallback, sibling order observable after
; reconstruction.
;-------------------------------------------------------------------------------
define void @version_nested_dynamic(ptr noalias dereferenceable(128) %A,
                                    ptr noalias %R, i64 %n) {
entry:
  br label %k.header
k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %i.header
i.header:
  %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
  %chk.i = phi double [ 0.000000e+00, %k.header ], [ %chk.next, %i.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %j.header
j.header:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.header ]
  %chk.j = phi double [ %chk.i, %i.header ], [ %chk.next.j, %j.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %i.epilogue, label %j.header
i.epilogue:
  %chk.next = phi double [ %chk.next.j, %j.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %i.latch
i.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %i.exit, label %i.header
i.exit:
  %chk.lcssa = phi double [ %chk.next, %i.latch ]
  br label %sibling.header
sibling.header:
  %sibling.iv = phi i64 [ 0, %i.exit ], [ %sibling.next, %sibling.header ]
  %sibling.next = add i64 %sibling.iv, 1
  %sibling.ec = icmp eq i64 %sibling.next, 2
  br i1 %sibling.ec, label %sibling.exit, label %sibling.header
sibling.exit:
  br label %k.epilogue
k.epilogue:
  %rk = getelementptr inbounds double, ptr %R, i64 %k
  store double %chk.lcssa, ptr %rk, align 8
  br label %k.latch
k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, %n
  br i1 %k.ec, label %exit, label %k.header
exit:
  ret void
}

; The nested join has the same post-E versioned and dedicated-fallback shape.

;-------------------------------------------------------------------------------
; A clone-safe-looking, memory-free call in N still carries `noduplicate`, so
; whole-loop cloning must reject it.
;-------------------------------------------------------------------------------
define void @version_noduplicate_epilogue(ptr noalias dereferenceable(128) %A,
                                          ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  call void @sink() #0
  %chk.next.j = fadd reassoc double %chk.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}


;-------------------------------------------------------------------------------
; An indirect branch in N is outside the loop-control boundaries but still makes
; the selected outer unsafe to clone.
;-------------------------------------------------------------------------------
define void @version_indirectbr(ptr noalias dereferenceable(128) %A,
                                ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.latch ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.latch ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %x
  indirectbr ptr blockaddress(@version_indirectbr, %inner.latch), [ label %inner.latch ]
inner.latch:
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.latch ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

;-------------------------------------------------------------------------------
; A live-out of a target extension type that behaves like a token: the selected
; outer header defines it and the shared exit uses it. Conditional cloning would
; have to merge the two loops' definitions with a PHI at the shared join, and a
; token-like value cannot be a PHI operand, so the selected outer loop is not
; safe to clone conditionally and the candidate is rejected before any mutation.
;-------------------------------------------------------------------------------
define void @version_tokenlike_liveout(ptr noalias dereferenceable(128) %A,
                                       ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi double [ 0.000000e+00, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi double [ 0.000000e+00, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi double [ 0.000000e+00, %outer.ph ], [ %v.next, %outer.latch ]
  %token = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") poison)
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi double [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi double [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi double [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %p.next.j = fadd reassoc double %p.j, %x
  %u.next.j = fadd reassoc double %u.j, %x
  %v.next.j = fadd reassoc double %v.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi double [ %p.next.j, %inner.header ]
  %u.next = phi double [ %u.next.j, %inner.header ]
  %v.next = phi double [ %v.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi double [ %p.next, %outer.latch ]
  %u.res = phi double [ %u.next, %outer.latch ]
  %v.res = phi double [ %v.next, %outer.latch ]
  %r1 = getelementptr inbounds double, ptr %R, i64 1
  %r2 = getelementptr inbounds double, ptr %R, i64 2
  store double %p.res, ptr %R, align 8
  store double %u.res, ptr %r1, align 8
  store double %v.res, ptr %r2, align 8
  %token.use = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") %token)
  ret void
}

;-------------------------------------------------------------------------------
; The same token-like value, but its only consumer sits in an unreachable
; block. Conditional clone safety judges live-outs by reachability, while the
; live-out repair that whole-loop versioning performs does not, so the
; versioning control scan rejects any token-like value inside the selected outer
; loop before any mutation.
;-------------------------------------------------------------------------------
define void @version_tokenlike_unreachable_use(ptr noalias dereferenceable(128) %A,
                                               ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi double [ 0.000000e+00, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi double [ 0.000000e+00, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi double [ 0.000000e+00, %outer.ph ], [ %v.next, %outer.latch ]
  %token = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") poison)
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi double [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi double [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi double [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %p.next.j = fadd reassoc double %p.j, %x
  %u.next.j = fadd reassoc double %u.j, %x
  %v.next.j = fadd reassoc double %v.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi double [ %p.next.j, %inner.header ]
  %u.next = phi double [ %u.next.j, %inner.header ]
  %v.next = phi double [ %v.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi double [ %p.next, %outer.latch ]
  %u.res = phi double [ %u.next, %outer.latch ]
  %v.res = phi double [ %v.next, %outer.latch ]
  %r1 = getelementptr inbounds double, ptr %R, i64 1
  %r2 = getelementptr inbounds double, ptr %R, i64 2
  store double %p.res, ptr %R, align 8
  store double %u.res, ptr %r1, align 8
  store double %v.res, ptr %r2, align 8
  ret void
unreach:
  %token.use = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") %token)
  ret void
}

;-------------------------------------------------------------------------------
; The same token-like value is now unused. Nothing is live-out, so
; conditional clone safety accepts the loop. The versioning control scan still
; rejects the token-like value, because the interchange of the versioned copy
; could not carry it through the LCSSA repair.
;-------------------------------------------------------------------------------
define void @version_tokenlike_nouse(ptr noalias dereferenceable(128) %A,
                                     ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi double [ 0.000000e+00, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi double [ 0.000000e+00, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi double [ 0.000000e+00, %outer.ph ], [ %v.next, %outer.latch ]
  %token = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") poison)
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %p.j = phi double [ %p.i, %outer.header ], [ %p.next.j, %inner.header ]
  %u.j = phi double [ %u.i, %outer.header ], [ %u.next.j, %inner.header ]
  %v.j = phi double [ %v.i, %outer.header ], [ %v.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %p.next.j = fadd reassoc double %p.j, %x
  %u.next.j = fadd reassoc double %u.j, %x
  %v.next.j = fadd reassoc double %v.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi double [ %p.next.j, %inner.header ]
  %u.next = phi double [ %u.next.j, %inner.header ]
  %v.next = phi double [ %v.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi double [ %p.next, %outer.latch ]
  %u.res = phi double [ %u.next, %outer.latch ]
  %v.res = phi double [ %v.next, %outer.latch ]
  %r1 = getelementptr inbounds double, ptr %R, i64 1
  %r2 = getelementptr inbounds double, ptr %R, i64 2
  store double %p.res, ptr %R, align 8
  store double %u.res, ptr %r1, align 8
  store double %v.res, ptr %r2, align 8
  ret void
}

;-------------------------------------------------------------------------------
; An ordinary double is defined in a conditional inner block, and its only user
; is in an unreachable block. LCSSA formation and conditional clone safety
; ignore that user, but findDefsUsedOutsideOfLoop counts it. Whole-loop
; versioning would then have to synthesize a join PHI for a value lacking an
; LCSSA PHI and not dominating the exiting block. The runtime versioning checks
; reject the loop before any mutation.
;-------------------------------------------------------------------------------
define void @version_liveout_unreachable_use(ptr noalias dereferenceable(128) %A,
                                            ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi double [ 0.000000e+00, %outer.ph ], [ %p.next, %outer.latch ]
  %u.i = phi double [ 0.000000e+00, %outer.ph ], [ %u.next, %outer.latch ]
  %v.i = phi double [ 0.000000e+00, %outer.ph ], [ %v.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.latch ]
  %p.j = phi double [ %p.i, %outer.header ], [ %p.next.j, %inner.latch ]
  %u.j = phi double [ %u.i, %outer.header ], [ %u.next.j, %inner.latch ]
  %v.j = phi double [ %v.i, %outer.header ], [ %v.next.j, %inner.latch ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %p.next.j = fadd reassoc double %p.j, %x
  %u.next.j = fadd reassoc double %u.j, %x
  %v.next.j = fadd reassoc double %v.j, %x
  %c = fcmp ogt double %x, 0.000000e+00
  br i1 %c, label %inner.then, label %inner.latch
inner.then:
  %y = fmul double %x, 2.000000e+00
  br label %inner.latch
inner.latch:
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi double [ %p.next.j, %inner.latch ]
  %u.next = phi double [ %u.next.j, %inner.latch ]
  %v.next = phi double [ %v.next.j, %inner.latch ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi double [ %p.next, %outer.latch ]
  %u.res = phi double [ %u.next, %outer.latch ]
  %v.res = phi double [ %v.next, %outer.latch ]
  %r1 = getelementptr inbounds double, ptr %R, i64 1
  %r2 = getelementptr inbounds double, ptr %R, i64 2
  store double %p.res, ptr %R, align 8
  store double %u.res, ptr %r1, align 8
  store double %v.res, ptr %r2, align 8
  ret void
unreach:
  %uy = fadd double %y, 1.000000e+00
  ret void
}

;-------------------------------------------------------------------------------
; A PHI in a reachable join block now consumes the same conditional inner
; definition along an incoming edge from an unreachable block. LCSSA attributes
; the PHI use to its incoming block and ignores that use, so the value lacks an
; exit PHI. findDefsUsedOutsideOfLoop still records the definition. The live-out
; predicate applies LCSSA's attribution and rejects the candidate before any
; mutation.
;-------------------------------------------------------------------------------
define void @version_liveout_unreachable_phi_edge(ptr noalias dereferenceable(128) %A,
                                                 ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %p.i = phi double [ 0.0, %outer.ph ], [ %p.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.latch ]
  %p.j = phi double [ %p.i, %outer.header ], [ %p.next.j, %inner.latch ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %p.next.j = fadd reassoc double %p.j, %x
  %c = fcmp ogt double %x, 0.0
  br i1 %c, label %inner.then, label %inner.latch
inner.then:
  %y = fmul double %x, 2.0
  br label %inner.latch
inner.latch:
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %p.next = phi double [ %p.next.j, %inner.latch ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.5
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %p.res = phi double [ %p.next, %outer.latch ]
  store double %p.res, ptr %R, align 8
  br label %join
unreach:
  br label %join
join:
  %uy = phi double [ 0.0, %exit ], [ %y, %unreach ]
  store double %uy, ptr %R, align 8
  ret void
}

;-------------------------------------------------------------------------------
; A clone-safe intrinsic call (llvm.fabs) in N does not block whole-loop
; versioning, and E stays within the supported extraction slice.
;-------------------------------------------------------------------------------
define void @version_fabs_epilogue(ptr noalias dereferenceable(128) %A,
                                   ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %xabs = call double @llvm.fabs.f64(double %x)
  %chk.next.j = fadd reassoc double %chk.j, %xabs
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}


;-------------------------------------------------------------------------------
; Already-versioned marker: the loop carries
; llvm.loop.interchange.runtime_versioned, so candidate discovery must skip it
; and never create a second version.
;-------------------------------------------------------------------------------
define void @version_marked(ptr noalias dereferenceable(128) %A,
                            ptr noalias %R, i64 %n) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header, !llvm.loop !0
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}


;-------------------------------------------------------------------------------
; optsize function: hasOptSize() rejects fission before preparation, and the
; normal interchange path leaves the imperfect nest unchanged.
;-------------------------------------------------------------------------------
define void @version_optsize(ptr noalias dereferenceable(128) %A,
                             ptr noalias %R, i64 %n) #1 {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}


;-------------------------------------------------------------------------------
; minsize function: also rejected before preparation under hasOptSize() (minsize
; implies optsize). The function is unchanged.
;-------------------------------------------------------------------------------
define void @version_minsize(ptr noalias dereferenceable(128) %A,
                             ptr noalias %R, i64 %n) #2 {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %x
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}


;-------------------------------------------------------------------------------
; This test case sits at the block-count limit and is accepted. The selected
; outer loop %outer.header has exactly 16 basic blocks: outer.header,
; inner.header, epilogue, pad01..pad12 (a straight-line multi-block epilogue
; path), and outer.latch. Entry and exit are outside the loop. The 16-block
; limit accepts this loop. The 17-block sibling below is one block over.
;-------------------------------------------------------------------------------
define void @version_blocks_16(ptr dereferenceable(128) %A, ptr %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %pad01
pad01:
  br label %pad02
pad02:
  br label %pad03
pad03:
  br label %pad04
pad04:
  br label %pad05
pad05:
  br label %pad06
pad06:
  br label %pad07
pad07:
  br label %pad08
pad08:
  br label %pad09
pad09:
  br label %pad10
pad10:
  br label %pad11
pad11:
  br label %pad12
pad12:
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}


;-------------------------------------------------------------------------------
; Adding pad13 to the 16-block control gives 17 loop blocks, one above the
; default block limit.
;-------------------------------------------------------------------------------
define void @version_blocks_17(ptr dereferenceable(128) %A, ptr %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %pad01
pad01:
  br label %pad02
pad02:
  br label %pad03
pad03:
  br label %pad04
pad04:
  br label %pad05
pad05:
  br label %pad06
pad06:
  br label %pad07
pad07:
  br label %pad08
pad08:
  br label %pad09
pad09:
  br label %pad10
pad10:
  br label %pad11
pad11:
  br label %pad12
pad12:
  br label %pad13
pad13:
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

;-------------------------------------------------------------------------------
; This test case sits at the instruction-count limit and is accepted. The
; selected outer loop has exactly 128 non-debug instructions: 22 in the header,
; inner loop, epilogue, pad branch, and latch, plus a sequentially named chain
; of 106 padding adds (%p1..%p106) in %pad, all inside the loop. That is the
; 128-instruction limit. The loop has only five blocks, so the block limit does
; not discriminate here.
;-------------------------------------------------------------------------------
define void @version_insts_128(ptr dereferenceable(128) %A, ptr %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %pad
pad:
  %p1 = add i64 %i, 0
  %p2 = add i64 %p1, 1
  %p3 = add i64 %p2, 1
  %p4 = add i64 %p3, 1
  %p5 = add i64 %p4, 1
  %p6 = add i64 %p5, 1
  %p7 = add i64 %p6, 1
  %p8 = add i64 %p7, 1
  %p9 = add i64 %p8, 1
  %p10 = add i64 %p9, 1
  %p11 = add i64 %p10, 1
  %p12 = add i64 %p11, 1
  %p13 = add i64 %p12, 1
  %p14 = add i64 %p13, 1
  %p15 = add i64 %p14, 1
  %p16 = add i64 %p15, 1
  %p17 = add i64 %p16, 1
  %p18 = add i64 %p17, 1
  %p19 = add i64 %p18, 1
  %p20 = add i64 %p19, 1
  %p21 = add i64 %p20, 1
  %p22 = add i64 %p21, 1
  %p23 = add i64 %p22, 1
  %p24 = add i64 %p23, 1
  %p25 = add i64 %p24, 1
  %p26 = add i64 %p25, 1
  %p27 = add i64 %p26, 1
  %p28 = add i64 %p27, 1
  %p29 = add i64 %p28, 1
  %p30 = add i64 %p29, 1
  %p31 = add i64 %p30, 1
  %p32 = add i64 %p31, 1
  %p33 = add i64 %p32, 1
  %p34 = add i64 %p33, 1
  %p35 = add i64 %p34, 1
  %p36 = add i64 %p35, 1
  %p37 = add i64 %p36, 1
  %p38 = add i64 %p37, 1
  %p39 = add i64 %p38, 1
  %p40 = add i64 %p39, 1
  %p41 = add i64 %p40, 1
  %p42 = add i64 %p41, 1
  %p43 = add i64 %p42, 1
  %p44 = add i64 %p43, 1
  %p45 = add i64 %p44, 1
  %p46 = add i64 %p45, 1
  %p47 = add i64 %p46, 1
  %p48 = add i64 %p47, 1
  %p49 = add i64 %p48, 1
  %p50 = add i64 %p49, 1
  %p51 = add i64 %p50, 1
  %p52 = add i64 %p51, 1
  %p53 = add i64 %p52, 1
  %p54 = add i64 %p53, 1
  %p55 = add i64 %p54, 1
  %p56 = add i64 %p55, 1
  %p57 = add i64 %p56, 1
  %p58 = add i64 %p57, 1
  %p59 = add i64 %p58, 1
  %p60 = add i64 %p59, 1
  %p61 = add i64 %p60, 1
  %p62 = add i64 %p61, 1
  %p63 = add i64 %p62, 1
  %p64 = add i64 %p63, 1
  %p65 = add i64 %p64, 1
  %p66 = add i64 %p65, 1
  %p67 = add i64 %p66, 1
  %p68 = add i64 %p67, 1
  %p69 = add i64 %p68, 1
  %p70 = add i64 %p69, 1
  %p71 = add i64 %p70, 1
  %p72 = add i64 %p71, 1
  %p73 = add i64 %p72, 1
  %p74 = add i64 %p73, 1
  %p75 = add i64 %p74, 1
  %p76 = add i64 %p75, 1
  %p77 = add i64 %p76, 1
  %p78 = add i64 %p77, 1
  %p79 = add i64 %p78, 1
  %p80 = add i64 %p79, 1
  %p81 = add i64 %p80, 1
  %p82 = add i64 %p81, 1
  %p83 = add i64 %p82, 1
  %p84 = add i64 %p83, 1
  %p85 = add i64 %p84, 1
  %p86 = add i64 %p85, 1
  %p87 = add i64 %p86, 1
  %p88 = add i64 %p87, 1
  %p89 = add i64 %p88, 1
  %p90 = add i64 %p89, 1
  %p91 = add i64 %p90, 1
  %p92 = add i64 %p91, 1
  %p93 = add i64 %p92, 1
  %p94 = add i64 %p93, 1
  %p95 = add i64 %p94, 1
  %p96 = add i64 %p95, 1
  %p97 = add i64 %p96, 1
  %p98 = add i64 %p97, 1
  %p99 = add i64 %p98, 1
  %p100 = add i64 %p99, 1
  %p101 = add i64 %p100, 1
  %p102 = add i64 %p101, 1
  %p103 = add i64 %p102, 1
  %p104 = add i64 %p103, 1
  %p105 = add i64 %p104, 1
  %p106 = add i64 %p105, 1
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

;-------------------------------------------------------------------------------
; The extra %p107 add raises the count to 129 non-debug instructions. The
; candidate must be rejected for exceeding the default limit.
;-------------------------------------------------------------------------------
define void @version_insts_129(ptr dereferenceable(128) %A, ptr %R, i64 %n) {
entry:
  br label %outer.ph
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %pad
pad:
  %p1 = add i64 %i, 0
  %p2 = add i64 %p1, 1
  %p3 = add i64 %p2, 1
  %p4 = add i64 %p3, 1
  %p5 = add i64 %p4, 1
  %p6 = add i64 %p5, 1
  %p7 = add i64 %p6, 1
  %p8 = add i64 %p7, 1
  %p9 = add i64 %p8, 1
  %p10 = add i64 %p9, 1
  %p11 = add i64 %p10, 1
  %p12 = add i64 %p11, 1
  %p13 = add i64 %p12, 1
  %p14 = add i64 %p13, 1
  %p15 = add i64 %p14, 1
  %p16 = add i64 %p15, 1
  %p17 = add i64 %p16, 1
  %p18 = add i64 %p17, 1
  %p19 = add i64 %p18, 1
  %p20 = add i64 %p19, 1
  %p21 = add i64 %p20, 1
  %p22 = add i64 %p21, 1
  %p23 = add i64 %p22, 1
  %p24 = add i64 %p23, 1
  %p25 = add i64 %p24, 1
  %p26 = add i64 %p25, 1
  %p27 = add i64 %p26, 1
  %p28 = add i64 %p27, 1
  %p29 = add i64 %p28, 1
  %p30 = add i64 %p29, 1
  %p31 = add i64 %p30, 1
  %p32 = add i64 %p31, 1
  %p33 = add i64 %p32, 1
  %p34 = add i64 %p33, 1
  %p35 = add i64 %p34, 1
  %p36 = add i64 %p35, 1
  %p37 = add i64 %p36, 1
  %p38 = add i64 %p37, 1
  %p39 = add i64 %p38, 1
  %p40 = add i64 %p39, 1
  %p41 = add i64 %p40, 1
  %p42 = add i64 %p41, 1
  %p43 = add i64 %p42, 1
  %p44 = add i64 %p43, 1
  %p45 = add i64 %p44, 1
  %p46 = add i64 %p45, 1
  %p47 = add i64 %p46, 1
  %p48 = add i64 %p47, 1
  %p49 = add i64 %p48, 1
  %p50 = add i64 %p49, 1
  %p51 = add i64 %p50, 1
  %p52 = add i64 %p51, 1
  %p53 = add i64 %p52, 1
  %p54 = add i64 %p53, 1
  %p55 = add i64 %p54, 1
  %p56 = add i64 %p55, 1
  %p57 = add i64 %p56, 1
  %p58 = add i64 %p57, 1
  %p59 = add i64 %p58, 1
  %p60 = add i64 %p59, 1
  %p61 = add i64 %p60, 1
  %p62 = add i64 %p61, 1
  %p63 = add i64 %p62, 1
  %p64 = add i64 %p63, 1
  %p65 = add i64 %p64, 1
  %p66 = add i64 %p65, 1
  %p67 = add i64 %p66, 1
  %p68 = add i64 %p67, 1
  %p69 = add i64 %p68, 1
  %p70 = add i64 %p69, 1
  %p71 = add i64 %p70, 1
  %p72 = add i64 %p71, 1
  %p73 = add i64 %p72, 1
  %p74 = add i64 %p73, 1
  %p75 = add i64 %p74, 1
  %p76 = add i64 %p75, 1
  %p77 = add i64 %p76, 1
  %p78 = add i64 %p77, 1
  %p79 = add i64 %p78, 1
  %p80 = add i64 %p79, 1
  %p81 = add i64 %p80, 1
  %p82 = add i64 %p81, 1
  %p83 = add i64 %p82, 1
  %p84 = add i64 %p83, 1
  %p85 = add i64 %p84, 1
  %p86 = add i64 %p85, 1
  %p87 = add i64 %p86, 1
  %p88 = add i64 %p87, 1
  %p89 = add i64 %p88, 1
  %p90 = add i64 %p89, 1
  %p91 = add i64 %p90, 1
  %p92 = add i64 %p91, 1
  %p93 = add i64 %p92, 1
  %p94 = add i64 %p93, 1
  %p95 = add i64 %p94, 1
  %p96 = add i64 %p95, 1
  %p97 = add i64 %p96, 1
  %p98 = add i64 %p97, 1
  %p99 = add i64 %p98, 1
  %p100 = add i64 %p99, 1
  %p101 = add i64 %p100, 1
  %p102 = add i64 %p101, 1
  %p103 = add i64 %p102, 1
  %p104 = add i64 %p103, 1
  %p105 = add i64 %p104, 1
  %p106 = add i64 %p105, 1
  %p107 = add i64 %p106, 1
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}


;-------------------------------------------------------------------------------
; A freeze in the selected outer header is sampled once per original outer
; iteration and shared by all child iterations. Interchange would resample it
; in the new inner loop, and runtime versioning cannot make the guarded copy
; sound. The otherwise-runtime-bound candidate must remain unversioned.
;-------------------------------------------------------------------------------
define void @version_outer_header_freeze(
    ptr noalias dereferenceable(128) %A, ptr noalias %R, i64 %n) {
entry:
  br label %outer.ph

outer.ph:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %choice = freeze i1 poison
  %col = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [4 x double], ptr %col, i64 %j, i64 0
  %x = load double, ptr %idx, align 8
  %selected = select i1 %choice, double %x, double 0.000000e+00
  %chk.next.j = fadd reassoc double %chk.j, %selected
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %n
  br i1 %j.ec, label %epilogue, label %inner.header

epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %dp = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %dv = load double, ptr %dp, align 8
  %dn = fmul double %dv, 1.500000e+00
  store double %dn, ptr %dp, align 8
  br label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header

exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}


declare void @sink()
declare double @llvm.fabs.f64(double)

attributes #0 = { noduplicate memory(none) nounwind willreturn }
attributes #1 = { optsize }
attributes #2 = { minsize }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.interchange.runtime_versioned"}
