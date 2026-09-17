; NOTE: Do not autogenerate
;
; Trip-count, byte-offset, and storage boundaries of the static bound proof:
; constant trips at, below, and above the row extent, a contextual exact trip
; behind a truncation, an enclosing AddRec trip, and the storage kinds an
; object-containment requirement may or may not read. N denotes the retained
; nest, E the post-inner epilogue, and W the row stride in elements.
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
; profitability, and validation. W is recovered from the element-normalized
; byte stride, not from the storage size, and each generalized N/E pair has
; its own modular and containment requirements.
; PREP-LABEL: loop-interchange: prepared function=bound_typed_canonical{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[TYPED_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[TYPED_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_same_trip_wmin{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=8 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=2 reductions=2
; PREP-SAME:  byte-offset-proofs=2 requirements=4 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[#WMIN_ID:]]:modular-outer-span/runtime/runtime
; PREP-SAME:  ,[[#WMIN_ID]]:object-containment/static/by-constant-max,[[#WMIN_ID+1]]:modular-outer-span/runtime/runtime,[[#WMIN_ID+1]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  loop-interchange: rejected function=bound_same_trip_wmin outer=outer.header inner=inner.header reason=recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_distinct_runtime{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=second-distinct-runtime-trip-object-containment{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_outer_static_inner_runtime{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=second-distinct-runtime-trip-object-containment{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_byte_canonical{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[BYTE_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[BYTE_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_byte_nondivisible{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
;
; The high-offset store is independent of N; it needs no byte-offset bound
; proof. Do not reinterpret a high-32-bit offset as the low address.
; PREP-LABEL: loop-interchange: prepared function=bound_offset_high32{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=2
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
;
; optsize and minsize functions skip preparation rather than reject a
; discovered candidate, so they print nothing. bound_const_eq is the same shape
; without the size attribute.
; PREP-LABEL: loop-interchange: prepared function=bound_storage_defined_global{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[DEFINED_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[DEFINED_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_storage_external_global{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[EXTERNAL_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[EXTERNAL_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_storage_replaceable_global{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[REPLACEABLE_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[REPLACEABLE_ID]]:object-containment/static/by-constant-max{{$}}
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
; PREP-NEXT:  loop-interchange: rejected function=bound_runtime_canonical outer=outer.header inner=inner.header reason=recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check{{$}}
;
; W-1 and W trips are safe. W+1 and W+2 must not be accepted by comparing
; backedges directly to W. The i8 wraparound case below has 256 trips, not zero.
; PREP-LABEL: loop-interchange: prepared function=bound_const_below{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[BELOW_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[BELOW_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_const_eq{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[CONST_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[CONST_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_const_above1{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=known-unsafe-trip-exceeds-bound{{$}}
; PREP-LABEL: loop-interchange: rejected function=bound_const_above2{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=known-unsafe-trip-exceeds-bound{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_dominating_guard{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[GUARD_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[GUARD_ID]]:object-containment/static/by-constant-max{{$}}
;
; This body discharges by constant maximum despite its name. The truncation
; twins below exercise the contextual exact-trip strategy.
; PREP-LABEL: loop-interchange: prepared function=bound_max_too_broad{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[MAX_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[MAX_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_assume{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[ASSUME_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[ASSUME_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_scunknown_range{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[RANGE_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[RANGE_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_ineffective_min_metadata{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[MIN_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[MIN_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  loop-interchange: rejected function=bound_ineffective_min_metadata outer=outer.header inner=inner.header reason=recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_nofact{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[NOFACT_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[NOFACT_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  loop-interchange: rejected function=bound_nofact outer=outer.header inner=inner.header reason=recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check{{$}}
; PREP-LABEL: loop-interchange: prepared function=bound_widen_i32{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=1
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[WIDEN_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[WIDEN_ID]]:object-containment/static/by-constant-max{{$}}
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
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
; PREP-LABEL: loop-interchange: prepared function=bound_context_nofact_trunc{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; PREP-SAME:  requirement-ids=[[NOFACT_TRUNC_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[NOFACT_TRUNC_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  loop-interchange: rejected function=bound_context_nofact_trunc outer=outer.header inner=inner.header reason=recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
; PREP-LABEL: loop-interchange: prepared function=bound_context_exact_addrec{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=3 routing-depth=3 fission-rows=0 routing-rows=1 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[ADDREC_ID:[0-9]+]]:modular-outer-span/static/by-context-exact,[[ADDREC_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
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
;

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
