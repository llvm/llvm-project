; NOTE: Do not autogenerate
;
; Trip-count, byte-offset, and storage boundaries of the static bound proof:
; constant trips at, below, and above the row extent, a contextual exact trip
; behind a truncation, an enclosing AddRec trip, and the storage kinds an
; object-containment requirement may or may not read. N denotes the retained
; nest, E the post-inner epilogue, and W the row stride in elements. A test case
; whose recovered bound is a runtime value is versioned behind one unsigned
; trip guard 'trip u> Wmin', and the versioned copy of that pair has its
; epilogue distributed into a sibling loop.
;
; DEFINE: %{outer_backedges} = \
; DEFINE:   --implicit-check-not='{{^Loop %outer.header: (backedge-taken|constant max backedge-taken) count is }}'
;
; With the option off the pass leaves this module unchanged.
; RUN: opt -S -passes=no-op-loopnest %s -o %t.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %s -o %t.out
; RUN: diff -u %t.noop %t.out
;
; The three contextual test cases must keep the outer backedge-taken counts
; their names promise: a truncation for the two twins, an enclosing AddRec for
; the third.
; RUN: llvm-extract -S --recursive --keep-const-init --func=bound_context_exact_trunc %s -o %t.exact-trunc.ll
; RUN: llvm-extract -S --recursive --keep-const-init --func=bound_context_nofact_trunc %s -o %t.nofact-trunc.ll
; RUN: llvm-extract -S --recursive --keep-const-init --func=bound_context_exact_addrec %s -o %t.exact-addrec.ll
; RUN: opt -passes='print<scalar-evolution>' -disable-output %t.exact-trunc.ll 2>&1 | FileCheck %s --check-prefix=EXACT-TRUNC-SCEV %{outer_backedges}
; RUN: opt -passes='print<scalar-evolution>' -disable-output %t.nofact-trunc.ll 2>&1 | FileCheck %s --check-prefix=NOFACT-TRUNC-SCEV %{outer_backedges}
; RUN: opt -passes='print<scalar-evolution>' -disable-output %t.exact-addrec.ll 2>&1 | FileCheck %s --check-prefix=ADDREC-SCEV %{outer_backedges}
;
; The profitability policy is fixed so that the small synthetic kernels are
; judged by instruction order rather than by the cache model.
; DEFINE: %{policy} = -loop-interchange-profitabilities=instorder,vectorize
; %{prepare} also enables runtime versioning, so the runtime-classified
; test cases of this file are guarded and applied.
; DEFINE: %{prepare} = -loop-interchange-outer-epilogue-fission -loop-interchange-print-prepared-plan -loop-interchange-outer-epilogue-runtime-versioning
; DEFINE: %{remarks} = -pass-remarks=loop-interchange -pass-remarks-analysis=loop-interchange -pass-remarks-missed=loop-interchange
; DEFINE: %{applied} = \
; DEFINE:   --func=bound_typed_canonical --func=bound_same_trip_wmin \
; DEFINE:   --func=bound_byte_canonical --func=bound_offset_high32 \
; DEFINE:   --func=bound_storage_defined_global \
; DEFINE:   --func=bound_storage_external_global \
; DEFINE:   --func=bound_storage_replaceable_global \
; DEFINE:   --func=bound_runtime_canonical \
; DEFINE:   --func=bound_const_below --func=bound_const_eq \
; DEFINE:   --func=bound_dominating_guard --func=bound_max_too_broad \
; DEFINE:   --func=bound_assume --func=bound_scunknown_range \
; DEFINE:   --func=bound_ineffective_min_metadata --func=bound_nofact \
; DEFINE:   --func=bound_widen_i32 --func=bound_context_exact_trunc \
; DEFINE:   --func=bound_context_nofact_trunc \
; DEFINE:   --func=bound_context_exact_addrec
; DEFINE: %{no_versioning} = \
; DEFINE:   --implicit-check-not='.lver' \
; DEFINE:   --implicit-check-not='!alias.scope' \
; DEFINE:   --implicit-check-not='!noalias' \
; DEFINE:   --implicit-check-not='llvm.loop.interchange.runtime_versioned' \
; DEFINE:   --implicit-check-not='LoopVersioning'
; DEFINE: %{no_alias_metadata} = \
; DEFINE:   --implicit-check-not='!alias.scope' \
; DEFINE:   --implicit-check-not='!noalias'
;
; With the option on, every function in %{applied} is transformed and every
; other function is unchanged.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.prep 2> %t.prep.stderr
; RUN: FileCheck %s --check-prefix=PREP --input-file=%t.prep.stderr --implicit-check-not='loop-interchange:'
; RUN: llvm-extract -S --delete %{applied} %t.out -o %t.out.rest
; RUN: llvm-extract -S --delete %{applied} %t.prep -o %t.prep.rest
; RUN: diff -u -I '^; ModuleID' %t.out.rest %t.prep.rest
; RUN: FileCheck %s --check-prefix=APPLIED --input-file=%t.prep %{no_alias_metadata} --implicit-check-not='{{^define }}' --implicit-check-not='!llvm.loop' --implicit-check-not='{{^[^[:space:]]*[.]lver[.]check[^:]*:}}'
; RUN: llvm-extract -S --delete --func=bound_same_trip_wmin --func=bound_runtime_canonical --func=bound_ineffective_min_metadata --func=bound_nofact --func=bound_context_nofact_trunc %t.prep -o %t.prep.static
; RUN: FileCheck %s --check-prefix=STATIC --input-file=%t.prep.static %{no_versioning} --implicit-check-not='{{^define }}'
;
; The updated LoopInfo must match a fresh rebuild, sibling order included, and
; a second run of the pass must change nothing.
; RUN: opt -passes='loop(loop-interchange),print<loops>' -cache-line-size=64 %{policy} -loop-interchange-outer-epilogue-fission -loop-interchange-outer-epilogue-runtime-versioning -disable-output %s 2>&1 | FileCheck %s --check-prefix=APPLY-LOOPS
; RUN: opt -passes='print<loops>' -disable-output %t.prep 2>&1 | FileCheck %s --check-prefix=APPLY-LOOPS
; RUN: opt -S -passes='loop(loop-interchange,loop-interchange)' -cache-line-size=64 %{policy} %{prepare} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.twice
; RUN: diff -u %t.prep %t.twice
;
; With runtime versioning turned off, the canonical runtime test case reports
; the disabled reason and nothing changes.
; RUN: llvm-extract -S --func=bound_runtime_canonical %s -o %t.rtcanon.ll
; RUN: opt -S -passes=no-op-loopnest %t.rtcanon.ll -o %t.rtcanon.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} -loop-interchange-outer-epilogue-runtime-versioning=false %t.rtcanon.ll -o %t.rt.off 2> %t.rt.off.stderr
; RUN: diff -u %t.rtcanon.noop %t.rt.off
; RUN: FileCheck %s --check-prefix=RUNTIME-OFF --input-file=%t.rt.off.stderr --implicit-check-not='loop-interchange:'
;
; Runtime versioning is off by default, so naming the fission option alone
; leaves all five runtime test cases unchanged.
; RUN: llvm-extract -S --func=bound_same_trip_wmin --func=bound_runtime_canonical --func=bound_ineffective_min_metadata --func=bound_nofact --func=bound_context_nofact_trunc %s -o %t.five.ll
; RUN: opt -S -passes=no-op-loopnest %t.five.ll -o %t.five.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-outer-epilogue-fission %t.five.ll -o %t.five.default
; RUN: diff -u %t.five.noop %t.five.default
;
; Disabling runtime versioning leaves this statically bounded test case prepared
; and distributed without a guard.
; RUN: llvm-extract -S --func=bound_const_eq %s -o %t.consteq.ll
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} -loop-interchange-outer-epilogue-runtime-versioning=false %t.consteq.ll -o %t.consteq.off 2> %t.consteq.off.stderr
; RUN: FileCheck %s --check-prefix=STATIC-RUNTIME-OFF --input-file=%t.consteq.off.stderr --implicit-check-not='loop-interchange:'
; RUN: FileCheck %s --check-prefix=STATIC-RUNTIME-OFF-IR --input-file=%t.consteq.off --implicit-check-not='.lver'
;
; Each printed plan has passed both dependence matrices, legality,
; profitability, and validation, and a runtime-classified plan has also passed
; the runtime versioning checks. The two remarks after it come from the
; apply step. A runtime-classified plan reports Wmin where a static one
; reports W. W is recovered from the element-normalized byte stride, not from
; the storage size, and each generalized N/E pair has its own modular and
; containment requirements.
; PREP-LABEL: loop-interchange: prepared function=bound_typed_canonical{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[TYPED_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[TYPED_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=1335.
; PREP-LABEL: loop-interchange: prepared function=bound_same_trip_wmin{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=8 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=2 reductions=2
; PREP-SAME:  byte-offset-proofs=2 requirements=4 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[#WMIN_ID:]]:modular-outer-span/runtime/runtime
; PREP-SAME:  ,[[#WMIN_ID]]:object-containment/static/by-constant-max,[[#WMIN_ID+1]]:modular-outer-span/runtime/runtime,[[#WMIN_ID+1]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
; PREP-LABEL: loop-interchange: rejected function=bound_distinct_runtime{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=second-distinct-runtime-trip-object-containment{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_outer_static_inner_runtime{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=second-distinct-runtime-trip-object-containment{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_byte_canonical{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[BYTE_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[BYTE_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=1335.
; PREP-LABEL: loop-interchange: rejected function=bound_byte_nondivisible{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
;
; The high-offset store is independent of N; it needs no byte-offset bound
; proof. Do not reinterpret a high-32-bit offset as the low address.
; PREP-LABEL: loop-interchange: prepared function=bound_offset_high32{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=2
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
;
; optsize and minsize functions skip preparation rather than reject a
; discovered candidate, so they print nothing. bound_const_eq is the same shape
; without the size attribute.
; PREP-LABEL: loop-interchange: prepared function=bound_storage_defined_global{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[DEFINED_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[DEFINED_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: prepared function=bound_storage_external_global{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[EXTERNAL_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[EXTERNAL_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: prepared function=bound_storage_replaceable_global{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[REPLACEABLE_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[REPLACEABLE_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: rejected function=bound_storage_missing{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_storage_undersized{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_storage_nullable{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_storage_weak{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_runtime_canonical{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=1335
; PREP-SAME:  requirement-ids=[[RUNTIME_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[RUNTIME_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=1335.
;
; W-1 and W trips are safe. W+1 and W+2 must not be accepted by comparing
; backedges directly to W. The i8 wraparound case below has 256 trips, not zero.
; PREP-LABEL: loop-interchange: prepared function=bound_const_below{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[BELOW_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[BELOW_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: prepared function=bound_const_eq{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[CONST_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[CONST_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: rejected function=bound_const_above1{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=known-unsafe-trip-exceeds-bound{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_const_above2{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=known-unsafe-trip-exceeds-bound{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_dominating_guard{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[GUARD_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[GUARD_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
;
; This body discharges by constant maximum despite its name. The truncation
; twins below exercise the contextual exact-trip strategy.
; PREP-LABEL: loop-interchange: prepared function=bound_max_too_broad{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[MAX_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[MAX_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: prepared function=bound_assume{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[ASSUME_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[ASSUME_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: prepared function=bound_scunknown_range{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[RANGE_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[RANGE_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: prepared function=bound_ineffective_min_metadata{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[MIN_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[MIN_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
; PREP-LABEL: loop-interchange: prepared function=bound_nofact{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[NOFACT_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[NOFACT_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
; PREP-LABEL: loop-interchange: prepared function=bound_widen_i32{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=1
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[WIDEN_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[WIDEN_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: rejected function=bound_overflow{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=known-unsafe-trip-exceeds-bound{{$}}
;
; Each pair has its own requirement ID. The modular row uses the exact trip
; in context, while the containment row independently uses a constant maximum.
; PREP-LABEL: loop-interchange: prepared function=bound_context_exact_trunc{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[TRUNC_ID:[0-9]+]]:modular-outer-span/static/by-context-exact,[[TRUNC_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
; PREP-LABEL: loop-interchange: prepared function=bound_context_nofact_trunc{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[NOFACT_TRUNC_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[NOFACT_TRUNC_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=runtime-bound, Wmin=4.
; PREP-LABEL: loop-interchange: prepared function=bound_context_exact_addrec{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=3 routing-depth=3 fission-rows=0 routing-rows=1 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[ADDREC_ID:[0-9]+]]:modular-outer-span/static/by-context-exact,[[ADDREC_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=4.
;
; EXACT-TRUNC-SCEV-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'bound_context_exact_trunc':
; EXACT-TRUNC-SCEV: {{^}}Determining loop execution counts for: @bound_context_exact_trunc{{$}}
; EXACT-TRUNC-SCEV: {{^}}Loop %outer.header: backedge-taken count is (zext i3 (trunc i64 %n to i3) to i64){{$}}
; EXACT-TRUNC-SCEV-NEXT: {{^}}Loop %outer.header: constant max backedge-taken count is i64 7{{$}}
; NOFACT-TRUNC-SCEV-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'bound_context_nofact_trunc':
; NOFACT-TRUNC-SCEV: {{^}}Determining loop execution counts for: @bound_context_nofact_trunc{{$}}
; NOFACT-TRUNC-SCEV: {{^}}Loop %outer.header: backedge-taken count is (zext i3 (trunc i64 %n to i3) to i64){{$}}
; NOFACT-TRUNC-SCEV-NEXT: {{^}}Loop %outer.header: constant max backedge-taken count is i64 7{{$}}
; ADDREC-SCEV-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'bound_context_exact_addrec':
; ADDREC-SCEV: {{^}}Determining loop execution counts for: @bound_context_exact_addrec{{$}}
; ADDREC-SCEV: {{^}}Loop %outer.header: backedge-taken count is {0,+,1}<nuw><nsw><%top.header>{{$}}
; ADDREC-SCEV-NEXT: {{^}}Loop %outer.header: constant max backedge-taken count is i64 999999{{$}}

; The transformed module. Every definition is listed, so the RUN line's
; '{{^define }}' exclusion makes the list exhaustive. Each statically bounded
; applied function gains one sibling epilogue loop whose latch branches to the
; original continuation, and its nest is interchanged. The emptied original
; epilogue block, the epilogue loop, and the continuation are checked
; instruction by instruction. Each runtime-bounded applied function gains one
; guard block, one untransformed fallback clone, one distributed and
; interchanged versioned nest, and one sibling epilogue loop on that path. The
; fallback clone keeps the whole original epilogue, so its epilogue block is
; checked instruction by instruction, including the inner loop's exit into it
; and its stores. The same invocation also excludes every '!llvm.loop'
; attachment and every '.lver.check' label definition it does not match here,
; so the two markers per versioned function and the single guard per versioned
; function are exhaustive too. Block labels are plain directives because the
; printed IR separates blocks with a blank line. The other functions were
; already shown unchanged by the llvm-extract --delete pair above and carry
; only their label.
;
; APPLIED-LABEL: define {{.*}}@bound_typed_canonical(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds [1335 x double], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
;
; The two byte-offset proofs constrain the same trip with row strides 4 and 6.
; One guard compares that trip with their minimum, Wmin = 4.
; APPLIED-LABEL: define {{.*}}@bound_same_trip_wmin(
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 4
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %chk.next.lver.orig = phi double [ %chk.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %adiag.lver.orig = getelementptr inbounds [4 x double], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %adv.lver.orig = load double, ptr %adiag.lver.orig, align 8
; APPLIED-NEXT:      %adn.lver.orig = fmul double %adv.lver.orig, 1.500000e+00
; APPLIED-NEXT:      store double %adn.lver.orig, ptr %adiag.lver.orig, align 8
; APPLIED-NEXT:      %bdiag.lver.orig = getelementptr inbounds [6 x double], ptr %B, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %bdv.lver.orig = load double, ptr %bdiag.lver.orig, align 8
; APPLIED-NEXT:      %bdn.lver.orig = fmul double %bdv.lver.orig, 1.500000e+00
; APPLIED-NEXT:      store double %bdn.lver.orig, ptr %bdiag.lver.orig, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_WMIN:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_WMIN:[0-9]+]]
; APPLIED:         exit.loopexit:
; APPLIED-NEXT:      %chk.res.ph = phi double [ %chk.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      br label %exit
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %chk.res.ph2 = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %adiag.epil = getelementptr inbounds [4 x double], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %adv.epil = load double, ptr %adiag.epil, align 8
; APPLIED-NEXT:      %adn.epil = fmul double %adv.epil, 1.500000e+00
; APPLIED-NEXT:      store double %adn.epil, ptr %adiag.epil, align 8
; APPLIED-NEXT:      %bdiag.epil = getelementptr inbounds [6 x double], ptr %B, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %bdv.epil = load double, ptr %bdiag.epil, align 8
; APPLIED-NEXT:      %bdn.epil = fmul double %bdv.epil, 1.500000e+00
; APPLIED-NEXT:      store double %bdn.epil, ptr %bdiag.epil, align 8
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
; APPLIED-LABEL: define {{.*}}@bound_distinct_runtime(
; APPLIED-LABEL: define {{.*}}@bound_outer_static_inner_runtime(
; APPLIED-LABEL: define {{.*}}@bound_byte_canonical(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %diagoff.epil = mul i64 %epilogue.iv, 10688
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds i8, ptr @byte_canonical_obj, i64 %diagoff.epil
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@bound_byte_nondivisible(
; APPLIED-LABEL: define {{.*}}@bound_offset_high32(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %lo.epil = getelementptr inbounds i8, ptr @offset_high32_obj, i64 16
; APPLIED-NEXT:      %hi.epil = getelementptr inbounds i8, ptr @offset_high32_obj, i64 4294967312
; APPLIED-NEXT:      %lv.epil = load double, ptr %lo.epil, align 8
; APPLIED-NEXT:      %hn.epil = fmul double %lv.epil, 1.500000e+00
; APPLIED-NEXT:      store double %hn.epil, ptr %hi.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@bound_optsize_canonical(
; APPLIED-LABEL: define {{.*}}@bound_minsize_canonical(
; APPLIED-LABEL: define {{.*}}@bound_storage_defined_global(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %diagoff.epil = mul i64 %epilogue.iv, 40
; APPLIED-NEXT:      %ddiag.epil = getelementptr i8, ptr @storage_defined_obj, i64 %diagoff.epil
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@bound_storage_external_global(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %diagoff.epil = mul i64 %epilogue.iv, 40
; APPLIED-NEXT:      %ddiag.epil = getelementptr i8, ptr @storage_external_obj, i64 %diagoff.epil
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@bound_storage_replaceable_global(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %diagoff.epil = mul i64 %epilogue.iv, 40
; APPLIED-NEXT:      %ddiag.epil = getelementptr i8, ptr @storage_replaceable_obj, i64 %diagoff.epil
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@bound_storage_missing(
; APPLIED-LABEL: define {{.*}}@bound_storage_undersized(
; APPLIED-LABEL: define {{.*}}@bound_storage_nullable(
; APPLIED-LABEL: define {{.*}}@bound_storage_weak(
;
; The canonical runtime test case is versioned behind the guard 'trip u> 1335'.
; APPLIED-LABEL: define {{.*}}@bound_runtime_canonical(
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = add i64 %n, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 1335
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           br i1 %j.ec.lver.orig, label %epilogue.lver.orig, label %inner.header.lver.orig
; APPLIED:         epilogue.lver.orig:
; APPLIED-NEXT:      %chk.next.lver.orig = phi double [ %chk.next.j.lver.orig, %inner.header.lver.orig ]
; APPLIED-NEXT:      %ddiag.lver.orig = getelementptr inbounds [1335 x double], ptr %A, i64 %i.lver.orig, i64 %i.lver.orig
; APPLIED-NEXT:      %dval.lver.orig = load double, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      %dnew.lver.orig = fmul double %dval.lver.orig, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.lver.orig, ptr %ddiag.lver.orig, align 8
; APPLIED-NEXT:      br label %outer.latch.lver.orig
; APPLIED:         outer.latch.lver.orig:
; APPLIED-NEXT:      %i.next.lver.orig = add i64 %i.lver.orig, 1
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %n
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_RTCANON:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_RTCANON:[0-9]+]]
; APPLIED:         exit.loopexit:
; APPLIED-NEXT:      %chk.res.ph = phi double [ %chk.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      br label %exit
; APPLIED:         exit.loopexit1:
; APPLIED-NEXT:      %chk.res.ph2 = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds [1335 x double], ptr %A, i64 %epilogue.iv, i64 %epilogue.iv
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
; APPLIED-LABEL: define {{.*}}@bound_const_below(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
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
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 3
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@bound_const_eq(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
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
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@bound_const_above1(
; APPLIED-LABEL: define {{.*}}@bound_const_above2(
; APPLIED-LABEL: define {{.*}}@bound_dominating_guard(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
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
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      br label %done
; APPLIED-LABEL: define {{.*}}@bound_max_too_broad(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
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
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %nn
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      br label %done
; APPLIED-LABEL: define {{.*}}@bound_assume(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
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
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@bound_scunknown_range(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
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
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
;
; The check block holds the shared smin call, now without its !range metadata,
; and the inner increment keeps its nuw and nsw flags in the fallback and in
; the versioned loop.
; APPLIED-LABEL: define {{.*}}@bound_ineffective_min_metadata(
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %mn = call i64 @llvm.smin.i64(i64 %m, i64 %k){{$}}
; APPLIED-NEXT:      %0 = add i64 %mn, -1
; APPLIED-NEXT:      %1 = zext i64 %0 to i65
; APPLIED-NEXT:      %2 = add nuw i65 %1, 1
; APPLIED-NEXT:      %ident.check = icmp ugt i65 %2, 4
; APPLIED-NEXT:      br i1 %ident.check, label %outer.header.ph.lver.orig, label %inner.header.preheader
; APPLIED:           %j.next.lver.orig = add nuw nsw i64 %j.lver.orig, 1
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
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %mn
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_MIN:[0-9]+]]
; APPLIED:           %j.next = add nuw nsw i64 %j, 1
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %mn
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_MIN:[0-9]+]]
; APPLIED:         exit.loopexit:
; APPLIED-NEXT:      %chk.res.ph = phi double [ %chk.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      br label %exit
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
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %mn
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         exit.loopexit1.cont:
; APPLIED-NEXT:      br label %exit
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %chk.res = phi double [ %chk.res.ph, %exit.loopexit ], [ %chk.res.ph2, %exit.loopexit1.cont ]
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
;
; APPLIED-LABEL: define {{.*}}@bound_nofact(
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
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_NOFACT:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %n
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_NOFACT:[0-9]+]]
; APPLIED:         exit.loopexit:
; APPLIED-NEXT:      %chk.res.ph = phi double [ %chk.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      br label %exit
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
; APPLIED-LABEL: define {{.*}}@bound_widen_i32(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i32 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %i64.epil = zext i32 %epilogue.iv to i64
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds [4 x double], ptr @widen_obj, i64 %i64.epil, i64 %i64.epil
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i32 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i32 %i.next.epil, %n
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      br label %done
; APPLIED-LABEL: define {{.*}}@bound_overflow(
; APPLIED-LABEL: define {{.*}}@bound_context_exact_trunc(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
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
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %nn
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      br label %done
;
; The guard truncates %n to the i3 backedge-taken count, zero-extends it to
; i65, adds one to form the trip, and compares the trip there. The versioned
; pair sits inside the entry branch.
; APPLIED-LABEL: define {{.*}}@bound_context_nofact_trunc(
; APPLIED:         outer.header.lver.check:
; APPLIED-NEXT:      %0 = trunc i64 %n to i3
; APPLIED-NEXT:      %1 = zext i3 %0 to i65
; APPLIED-NEXT:      %2 = add nuw nsw i65 %1, 1
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
; APPLIED-NEXT:      %i.ec.lver.orig = icmp eq i64 %i.next.lver.orig, %nn
; APPLIED-NEXT:      br i1 %i.ec.lver.orig, label %exit.loopexit, label %outer.header.lver.orig, !llvm.loop ![[FB_TRUNC:[0-9]+]]
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i.next, %nn
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header, !llvm.loop ![[FAST_TRUNC:[0-9]+]]
; APPLIED:         exit.loopexit:
; APPLIED-NEXT:      %chk.res.ph = phi double [ %chk.next.lver.orig, %outer.latch.lver.orig ]
; APPLIED-NEXT:      br label %exit
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
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %nn
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.loopexit1.cont, label %epilogue.header
; APPLIED:         exit.loopexit1.cont:
; APPLIED-NEXT:      br label %exit
; APPLIED:         {{^exit:}}
; APPLIED-NEXT:      %chk.res = phi double [ %chk.res.ph, %exit.loopexit ], [ %chk.res.ph2, %exit.loopexit1.cont ]
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      br label %done
; APPLIED:         done:
; APPLIED-NEXT:      ret void
;
; APPLIED-LABEL: define {{.*}}@bound_context_exact_addrec(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         outer.exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
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
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, %t
; APPLIED-NEXT:      br i1 %i.ec.epil, label %outer.exit.cont, label %epilogue.header
; APPLIED:         outer.exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      br label %top.latch
;
; Every attachment the file carries is one of the ten matched above: a versioned
; and a fallback marker for each of the five versioned functions. Each is a
; distinct loop id whose second operand is the shared marker name. The range
; metadata of @bound_scunknown_range separates the fourth id from the fifth, so
; the chain restarts there.
; APPLIED:         ![[FB_WMIN]] = distinct !{![[FB_WMIN]], ![[MARK:[0-9]+]]}
; APPLIED-NEXT:    ![[MARK]] = !{!"llvm.loop.interchange.runtime_versioned"}
; APPLIED-NEXT:    ![[FAST_WMIN]] = distinct !{![[FAST_WMIN]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_RTCANON]] = distinct !{![[FB_RTCANON]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_RTCANON]] = distinct !{![[FAST_RTCANON]], ![[MARK]]}
; APPLIED:         ![[FB_MIN]] = distinct !{![[FB_MIN]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_MIN]] = distinct !{![[FAST_MIN]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_NOFACT]] = distinct !{![[FB_NOFACT]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_NOFACT]] = distinct !{![[FAST_NOFACT]], ![[MARK]]}
; APPLIED-NEXT:    ![[FB_TRUNC]] = distinct !{![[FB_TRUNC]], ![[MARK]]}
; APPLIED-NEXT:    ![[FAST_TRUNC]] = distinct !{![[FAST_TRUNC]], ![[MARK]]}
;
; With the five versioned functions deleted, the remaining module is free of
; guards, versioning markers, and alias metadata. The definition exclusion
; makes this list exhaustive.
; STATIC-LABEL: define {{.*}}@bound_typed_canonical(
; STATIC-LABEL: define {{.*}}@bound_distinct_runtime(
; STATIC-LABEL: define {{.*}}@bound_outer_static_inner_runtime(
; STATIC-LABEL: define {{.*}}@bound_byte_canonical(
; STATIC-LABEL: define {{.*}}@bound_byte_nondivisible(
; STATIC-LABEL: define {{.*}}@bound_offset_high32(
; STATIC-LABEL: define {{.*}}@bound_optsize_canonical(
; STATIC-LABEL: define {{.*}}@bound_minsize_canonical(
; STATIC-LABEL: define {{.*}}@bound_storage_defined_global(
; STATIC-LABEL: define {{.*}}@bound_storage_external_global(
; STATIC-LABEL: define {{.*}}@bound_storage_replaceable_global(
; STATIC-LABEL: define {{.*}}@bound_storage_missing(
; STATIC-LABEL: define {{.*}}@bound_storage_undersized(
; STATIC-LABEL: define {{.*}}@bound_storage_nullable(
; STATIC-LABEL: define {{.*}}@bound_storage_weak(
; STATIC-LABEL: define {{.*}}@bound_const_below(
; STATIC-LABEL: define {{.*}}@bound_const_eq(
; STATIC-LABEL: define {{.*}}@bound_const_above1(
; STATIC-LABEL: define {{.*}}@bound_const_above2(
; STATIC-LABEL: define {{.*}}@bound_dominating_guard(
; STATIC-LABEL: define {{.*}}@bound_max_too_broad(
; STATIC-LABEL: define {{.*}}@bound_assume(
; STATIC-LABEL: define {{.*}}@bound_scunknown_range(
; STATIC-LABEL: define {{.*}}@bound_widen_i32(
; STATIC-LABEL: define {{.*}}@bound_overflow(
; STATIC-LABEL: define {{.*}}@bound_context_exact_trunc(
; STATIC-LABEL: define {{.*}}@bound_context_exact_addrec(
;
; Compare incremental LoopInfo with a fresh reconstruction. Block order inside
; a nest is wildcarded because the two traversals may discover blocks
; differently. The loop order, the depths, and the epilogue loop's two blocks
; are exact. Every versioned function also states its headers and latches
; exactly and leaves the sibling order fallback, epilogue, versioned.
;
; APPLY-LOOPS-LABEL: Loop info for function 'bound_typed_canonical':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_same_trip_wmin':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'bound_byte_canonical':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_offset_high32':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_storage_defined_global':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_storage_external_global':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_storage_replaceable_global':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_runtime_canonical':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'bound_const_below':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_const_eq':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_dominating_guard':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_max_too_broad':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_assume':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_scunknown_range':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_ineffective_min_metadata':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'bound_nofact':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'bound_widen_i32':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_context_exact_trunc':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'bound_context_nofact_trunc':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %outer.header.lver.orig<header>,{{.*}}%outer.latch.lver.orig<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header.lver.orig<header><latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}%inner.header.split<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}%outer.latch<latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'bound_context_exact_addrec':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %top.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:         Loop at depth 3 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
;
; With runtime versioning disabled, the runtime test case reports the disabled
; reason and remains unchanged.
; RUNTIME-OFF: loop-interchange: rejected function=bound_runtime_canonical{{ }}
; RUNTIME-OFF-SAME: outer=outer.header inner=inner.header reason=runtime outer-epilogue versioning is disabled{{$}}
;
; Disabling runtime versioning does not affect static preparation or epilogue
; extraction.
; STATIC-RUNTIME-OFF: loop-interchange: prepared function=bound_const_eq{{ }}
; STATIC-RUNTIME-OFF-SAME: outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; STATIC-RUNTIME-OFF-SAME: absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; STATIC-RUNTIME-OFF-SAME: byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; STATIC-RUNTIME-OFF-SAME: requirement-ids=[[CONSTEQ_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[CONSTEQ_ID]]:object-containment/static/by-constant-max{{$}}
; STATIC-RUNTIME-OFF-IR: epilogue.header:

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @bound_typed_canonical(ptr noalias dereferenceable(14257800) %A,
                                   ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [1335 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [1335 x double], ptr %col, i64 %j, i64 0
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %A, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_same_trip_wmin(ptr noalias dereferenceable(128) %A,
                                  ptr noalias dereferenceable(288) %B,
                                  ptr noalias %R, i64 %n) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %acol = getelementptr inbounds [4 x double], ptr %A, i64 0, i64 %i
  %bcol = getelementptr inbounds [6 x double], ptr %B, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %aidx = getelementptr inbounds [4 x double], ptr %acol, i64 %j, i64 0
  %av = load double, ptr %aidx, align 8
  %bidx = getelementptr inbounds [6 x double], ptr %bcol, i64 %j, i64 0
  %bv = load double, ptr %bidx, align 8
  %sum = fadd reassoc double %chk.j, %av
  %chk.next.j = fadd reassoc double %sum, %bv
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %adiag = getelementptr inbounds [4 x double], ptr %A, i64 %i, i64 %i
  %adv = load double, ptr %adiag, align 8
  %adn = fmul double %adv, 1.500000e+00
  store double %adn, ptr %adiag, align 8
  %bdiag = getelementptr inbounds [6 x double], ptr %B, i64 %i, i64 %i
  %bdv = load double, ptr %bdiag, align 8
  %bdn = fmul double %bdv, 1.500000e+00
  store double %bdn, ptr %bdiag, align 8
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

@distinct_runtime_obj = internal global [1024 x double] zeroinitializer, align 8

define void @bound_distinct_runtime(ptr noalias %R, i64 %n, i64 %m) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %row = mul i64 %j, 4
  %off = add i64 %row, %i
  %idx = getelementptr inbounds double, ptr @distinct_runtime_obj, i64 %off
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %m
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %drow = mul i64 %i, 4
  %doff = add i64 %drow, %i
  %ddiag = getelementptr inbounds double, ptr @distinct_runtime_obj, i64 %doff
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

define void @bound_outer_static_inner_runtime(ptr noalias %R, i64 %m) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %row = mul i64 %j, 4
  %off = add i64 %row, %i
  %idx = getelementptr inbounds double, ptr @distinct_runtime_obj, i64 %off
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %m
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %drow = mul i64 %i, 4
  %doff = add i64 %drow, %i
  %ddiag = getelementptr inbounds double, ptr @distinct_runtime_obj, i64 %doff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

@byte_canonical_obj = common global [14268552 x i8] zeroinitializer, align 8

define void @bound_byte_canonical(ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr inbounds i8, ptr @byte_canonical_obj, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 10680
  %idx = getelementptr inbounds i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 10688
  %ddiag = getelementptr inbounds i8, ptr @byte_canonical_obj, i64 %diagoff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

@byte_nondiv_obj = internal global [256 x i8] zeroinitializer, align 8

define void @bound_byte_nondivisible(ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr inbounds i8, ptr @byte_nondiv_obj, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 52
  %idx = getelementptr inbounds i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 4
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 60
  %ddiag = getelementptr inbounds i8, ptr @byte_nondiv_obj, i64 %diagoff
  %dval = load double, ptr %ddiag, align 4
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 4
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

@offset_high32_obj = external global [4294967320 x i8], align 8

define void @bound_offset_high32(ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [4 x double], ptr @offset_high32_obj, i64 0, i64 %i
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
  %lo = getelementptr inbounds i8, ptr @offset_high32_obj, i64 16
  %hi = getelementptr inbounds i8, ptr @offset_high32_obj, i64 4294967312
  %lv = load double, ptr %lo, align 8
  %hn = fmul double %lv, 1.500000e+00
  store double %hn, ptr %hi, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_optsize_canonical(ptr noalias dereferenceable(128) %A,
                                     ptr noalias %R) optsize {
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_minsize_canonical(ptr noalias dereferenceable(128) %A,
                                     ptr noalias %R) minsize {
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

@storage_defined_obj = internal global [16 x double] zeroinitializer, align 8
@storage_external_obj = external global [16 x double], align 8
@storage_replaceable_obj = weak global [16 x double] zeroinitializer, align 8
@storage_weak_obj = extern_weak global [16 x double], align 8

define void @bound_storage_defined_global(ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr i8, ptr @storage_defined_obj, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 32
  %idx = getelementptr i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 40
  %ddiag = getelementptr i8, ptr @storage_defined_obj, i64 %diagoff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_storage_external_global(ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr i8, ptr @storage_external_obj, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 32
  %idx = getelementptr i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 40
  %ddiag = getelementptr i8, ptr @storage_external_obj, i64 %diagoff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_storage_replaceable_global(ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr i8, ptr @storage_replaceable_obj, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 32
  %idx = getelementptr i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 40
  %ddiag = getelementptr i8, ptr @storage_replaceable_obj, i64 %diagoff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_storage_missing(ptr noalias %A, ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr i8, ptr %A, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 32
  %idx = getelementptr i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 40
  %ddiag = getelementptr i8, ptr %A, i64 %diagoff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_storage_undersized(ptr noalias dereferenceable(127) %A,
                                      ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr i8, ptr %A, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 32
  %idx = getelementptr i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 40
  %ddiag = getelementptr i8, ptr %A, i64 %diagoff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_storage_nullable(
    ptr noalias dereferenceable_or_null(128) %A, ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr i8, ptr %A, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 32
  %idx = getelementptr i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 40
  %ddiag = getelementptr i8, ptr %A, i64 %diagoff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_storage_weak(ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %coloff = mul i64 %i, 8
  %col = getelementptr i8, ptr @storage_weak_obj, i64 %coloff
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %rowoff = mul i64 %j, 32
  %idx = getelementptr i8, ptr %col, i64 %rowoff
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %diagoff = mul i64 %i, 40
  %ddiag = getelementptr i8, ptr @storage_weak_obj, i64 %diagoff
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_runtime_canonical(ptr noalias dereferenceable(14257800) %A,
                                     ptr noalias %R, i64 %n) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %col = getelementptr inbounds [1335 x double], ptr %A, i64 0, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %idx = getelementptr inbounds [1335 x double], ptr %col, i64 %j, i64 0
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %A, i64 %i, i64 %i
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

define void @bound_const_below(ptr noalias dereferenceable(128) %A,
                               ptr noalias %R) {
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 3
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_const_eq(ptr noalias dereferenceable(128) %A,
                            ptr noalias %R) {
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 4
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_const_above1(ptr noalias dereferenceable(128) %A,
                                ptr noalias %R) {
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 5
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_const_above2(ptr noalias dereferenceable(128) %A,
                                ptr noalias %R) {
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 6
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_dominating_guard(ptr noalias dereferenceable(128) %A,
                                    ptr noalias %R, i64 %n) {
entry:
  %lo = icmp uge i64 %n, 1
  %hi = icmp ule i64 %n, 4
  %ok = and i1 %lo, %hi
  br i1 %ok, label %outer.ph, label %done
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  br label %done
done:
  ret void
}

define void @bound_max_too_broad(ptr noalias dereferenceable(128) %A,
                                 ptr noalias %R, i64 %n) {
entry:
  %m7 = and i64 %n, 7
  %nn = add i64 %m7, 1
  %hi = icmp ule i64 %nn, 4
  br i1 %hi, label %outer.ph, label %done
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %nn
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  br label %done
done:
  ret void
}

define void @bound_assume(ptr noalias dereferenceable(128) %A, ptr noalias %R,
                          i64 %n) {
entry:
  %lo = icmp uge i64 %n, 1
  call void @llvm.assume(i1 %lo)
  %hi = icmp ule i64 %n, 4
  call void @llvm.assume(i1 %hi)
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

define void @bound_scunknown_range(ptr noalias dereferenceable(128) %A,
                                   ptr noalias %R, ptr %np) {
entry:
  %n = load i64, ptr %np, align 8, !range !1
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

define void @bound_ineffective_min_metadata(
    ptr noalias dereferenceable(128) %A, ptr noalias %R, i64 %m, i64 %k) {
entry:
  %mn = call i64 @llvm.smin.i64(i64 %m, i64 %k), !range !0
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
  %v = load double, ptr %idx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %v
  %j.next = add nuw nsw i64 %j, 1
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
  %i.ec = icmp eq i64 %i.next, %mn
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @bound_nofact(ptr noalias dereferenceable(128) %A, ptr noalias %R,
                          i64 %n) {
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

@widen_obj = internal global [4096 x double] zeroinitializer, align 8

define void @bound_widen_i32(ptr noalias %R, i32 %n) {
entry:
  %lo = icmp uge i32 %n, 1
  %hi = icmp ule i32 %n, 4
  %ok = and i1 %lo, %hi
  br i1 %ok, label %outer.ph, label %done
outer.ph:
  br label %outer.header
outer.header:
  %i = phi i32 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %outer.ph ], [ %chk.next, %outer.latch ]
  %i64 = zext i32 %i to i64
  %col = getelementptr inbounds [4 x double], ptr @widen_obj, i64 0, i64 %i64
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
  %ddiag = getelementptr inbounds [4 x double], ptr @widen_obj, i64 %i64, i64 %i64
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i32 %i, 1
  %i.ec = icmp eq i32 %i.next, %n
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  br label %done
done:
  ret void
}

@overflow_obj = internal global [2048 x double] zeroinitializer, align 8

define void @bound_overflow(ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i8 = phi i8 [ 0, %entry ], [ %i8.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %i = zext i8 %i8 to i64
  %col = getelementptr inbounds [4 x double], ptr @overflow_obj, i64 0, i64 %i
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
  %ddiag = getelementptr inbounds [4 x double], ptr @overflow_obj, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i8.next = add i8 %i8, 1
  %i.ec = icmp eq i8 %i8.next, 0
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

declare void @llvm.assume(i1)
declare i64 @llvm.smin.i64(i64, i64)

!0 = !{i64 0, i64 5}
!1 = !{i64 1, i64 5}

define void @bound_context_exact_trunc(ptr noalias dereferenceable(128) %A, ptr noalias %R, i64 %n) {
entry:
  %m7 = and i64 %n, 7
  %nn = add nuw nsw i64 %m7, 1
  %nn32 = trunc i64 %nn to i32
  %hi = icmp ule i32 %nn32, 4
  br i1 %hi, label %outer.ph, label %done
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %nn
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  br label %done
done:
  ret void
}
define void @bound_context_nofact_trunc(ptr noalias dereferenceable(128) %A, ptr noalias %R, i64 %n) {
entry:
  %m7 = and i64 %n, 7
  %nn = add nuw nsw i64 %m7, 1
  %nn32 = trunc i64 %nn to i32
  %hi = icmp ne i32 %nn32, 12345
  br i1 %hi, label %outer.ph, label %done
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %nn
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  br label %done
done:
  ret void
}

; The enclosing AddRec supplies the selected outer trip. Its guard limits
; entry to the imperfect pair without changing the enclosing loop's span.
define void @bound_context_exact_addrec(ptr noalias dereferenceable(128) %A, ptr noalias %R) {
entry:
  br label %top.header
top.header:
  %t = phi i64 [ 1, %entry ], [ %t.next, %top.latch ]
  %hi = icmp ule i64 %t, 4
  br i1 %hi, label %outer.ph, label %top.latch
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
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, %t
  br i1 %i.ec, label %outer.exit, label %outer.header
outer.exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  br label %top.latch
top.latch:
  %t.next = add nuw nsw i64 %t, 1
  %t.ec = icmp eq i64 %t.next, 1000001
  br i1 %t.ec, label %done, label %top.header
done:
  ret void
}
