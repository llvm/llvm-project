; NOTE: Do not autogenerate
;
; Candidate routing: which loop pair of a function's collected chains the
; feature prepares, in which order, under which candidate budget, nest-depth
; limits, and profitability policy, and how the remaining pairs reach ordinary
; interchange. The routing cases are numbered r1 to r16; each function's
; comment states its case.
;
; With the option off the pass leaves these functions unchanged; the others
; are interchanged by the ordinary path, so %t.default is the reference.
; RUN: llvm-extract -S %s -o %t.unchanged.ll -func=r1_r2_length3_simple_ancestor -func=r3_length3_nonsimple_ancestor -func=r4_overdepth_tail_candidate -func=r4_uncomputable_ancestor -func=r6_runtime_then_static -func=r7_rejected_then_static -func=r8_ineligible_then_static -func=r10_r11_budget_slot_sequence -func=r12_two_static_collected_order -func=r13_explicit_cache_nested_chain -func=r14_strict_fp -func=r15_nested_scalar_all -func=top_level_middle_epilogue_order
; RUN: opt -S -passes=no-op-loopnest %t.unchanged.ll -o %t.unchanged.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %t.unchanged.ll -o %t.unchanged.out
; RUN: diff -u %t.unchanged.noop %t.unchanged.out
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %s -o %t.default
;
; The profitability policy is fixed so that the small synthetic kernels are
; judged by instruction order rather than by the cache model.
; DEFINE: %{policy} = -loop-interchange-profitabilities=instorder,vectorize
; DEFINE: %{prepare} = -loop-interchange-outer-epilogue-fission -loop-interchange-print-prepared-plan
; DEFINE: %{remarks} = -pass-remarks=loop-interchange -pass-remarks-analysis=loop-interchange -pass-remarks-missed=loop-interchange
; DEFINE: %{prep_checks} = --implicit-check-not='loop-interchange:'
;
; Preparation is analysis-only: a complete plan is printed and then discarded,
; so every function keeps its ordinary result. PREP-COMMON holds the decisions
; every configuration makes; PREP-LATER those reachable with at least two
; candidate slots; CHAIN3 those needing an eligible length-three chain; PREP
; and BUDGET2 distinguish reaching the third attempt from stopping at two.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.prep 2> %t.prep.stderr
; RUN: FileCheck %s --check-prefixes=PREP-COMMON,PREP-LATER,CHAIN3,PREP --input-file=%t.prep.stderr %{prep_checks}
; RUN: diff -u %t.default %t.prep
;
; A zero candidate budget does no feature work. One slot stops after a
; rejection or a runtime plan but skips an ineligible chain for free. An
; exhausted budget never suppresses ordinary routing.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-outer-epilogue-fission-candidates=0 -loop-interchange-outer-epilogue-fission=false %{remarks} %s -o %t.budget0.off 2> %t.budget0.off.stderr
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-outer-epilogue-fission-candidates=0 %{prepare} %{remarks} %s -o %t.budget0.on 2> %t.budget0.on.stderr
; RUN: diff -u %t.default %t.budget0.off
; RUN: diff -u %t.budget0.off %t.budget0.on
; RUN: diff -u %t.budget0.off.stderr %t.budget0.on.stderr
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-outer-epilogue-fission-candidates=1 -loop-interchange-outer-epilogue-fission=false %s -o %t.budget1.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-outer-epilogue-fission-candidates=1 %{prepare} %{remarks} %s -o %t.budget1.on 2> %t.budget1.stderr
; RUN: diff -u %t.default %t.budget1.off
; RUN: diff -u %t.budget1.off %t.budget1.on
; RUN: FileCheck %s --check-prefixes=PREP-COMMON,CHAIN3,BUDGET1 --input-file=%t.budget1.stderr %{prep_checks}
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-outer-epilogue-fission-candidates=2 -loop-interchange-outer-epilogue-fission=false %s -o %t.budget2.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-outer-epilogue-fission-candidates=2 %{prepare} %{remarks} %s -o %t.budget2.on 2> %t.budget2.stderr
; RUN: diff -u %t.default %t.budget2.off
; RUN: diff -u %t.budget2.off %t.budget2.on
; RUN: FileCheck %s --check-prefixes=PREP-COMMON,PREP-LATER,CHAIN3,BUDGET2 --input-file=%t.budget2.stderr %{prep_checks}
;
; Chain depth and logical pair depth are separate gates. At [2,2], r1 is
; skipped while nested length-two chains still prepare. At [2,3], r1 prepares.
; At [3,3], r1 reaches ordinary legality but its depth-two fission pair is
; ineligible, so the feature-on and feature-off diagnostics agree.
; Raising the maximum to eleven reaches r4's absolute-prefix rejection.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-min-loop-nest-depth=2 -loop-interchange-max-loop-nest-depth=2 -loop-interchange-outer-epilogue-fission=false %s -o %t.depth22.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-min-loop-nest-depth=2 -loop-interchange-max-loop-nest-depth=2 %{prepare} %{remarks} %s -o %t.depth22.on 2> %t.depth22.stderr
; RUN: diff -u %t.depth22.off %t.depth22.on
; RUN: FileCheck %s --check-prefixes=PREP-COMMON,PREP-LATER,PREP --input-file=%t.depth22.stderr %{prep_checks}
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-min-loop-nest-depth=2 -loop-interchange-max-loop-nest-depth=3 -loop-interchange-outer-epilogue-fission=false %s -o %t.depth23.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-min-loop-nest-depth=2 -loop-interchange-max-loop-nest-depth=3 %{prepare} %{remarks} %s -o %t.depth23.on 2> %t.depth23.stderr
; RUN: diff -u %t.depth23.off %t.depth23.on
; RUN: FileCheck %s --check-prefixes=PREP-COMMON,PREP-LATER,CHAIN3,PREP --input-file=%t.depth23.stderr %{prep_checks}
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-min-loop-nest-depth=3 -loop-interchange-max-loop-nest-depth=3 -loop-interchange-outer-epilogue-fission=false %{remarks} %s -o %t.depth33.off 2> %t.depth33.off.stderr
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-min-loop-nest-depth=3 -loop-interchange-max-loop-nest-depth=3 %{prepare} %{remarks} %s -o %t.depth33.on 2> %t.depth33.on.stderr
; RUN: diff -u %t.depth33.off %t.depth33.on
; RUN: diff -u %t.depth33.off.stderr %t.depth33.on.stderr
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-min-loop-nest-depth=2 -loop-interchange-max-loop-nest-depth=11 -loop-interchange-outer-epilogue-fission=false %s -o %t.depth211.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-min-loop-nest-depth=2 -loop-interchange-max-loop-nest-depth=11 %{prepare} %{remarks} %s -o %t.depth211.on 2> %t.depth211.stderr
; RUN: diff -u %t.depth211.off %t.depth211.on
; RUN: FileCheck %s --check-prefixes=PREP-COMMON,PREP-LATER,CHAIN3,PREP,DEPTH11 --input-file=%t.depth211.stderr %{prep_checks}
;
; Cache uses the collected-chain root. r1 and the top-level middle pair remain
; positive; the nested cache candidates abstain. The diagnostic's existing
; "default profitability" wording is also used for an explicit cache policy.
; Ignore bypasses cost, not strict-FP or dependence legality. It also makes
; both ordinary r9 chains interchange; do not impose the default r9 shape.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 -loop-interchange-profitabilities=cache -loop-interchange-outer-epilogue-fission=false %s -o %t.cache.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 -loop-interchange-profitabilities=cache %{prepare} %{remarks} %s -o %t.cache.on 2> %t.cache.stderr
; RUN: diff -u %t.cache.off %t.cache.on
; RUN: FileCheck %s --check-prefixes=CHAIN3,CACHE --input-file=%t.cache.stderr %{prep_checks}
; RUN: FileCheck %s --check-prefix=CACHE-IR --input-file=%t.cache.on
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 -loop-interchange-profitabilities=ignore -loop-interchange-outer-epilogue-fission=false %s -o %t.ignore.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 -loop-interchange-profitabilities=ignore %{prepare} %{remarks} %s -o %t.ignore.on 2> %t.ignore.stderr
; RUN: diff -u %t.ignore.off %t.ignore.on
; RUN: FileCheck %s --check-prefixes=PREP-COMMON,PREP-LATER,CHAIN3,PREP --input-file=%t.ignore.stderr %{prep_checks}
; RUN: FileCheck %s --check-prefix=IGNORE --input-file=%t.ignore.stderr
; RUN: FileCheck %s --check-prefix=IGNORE-IR --input-file=%t.ignore.on
;
; Assertions-only traces distinguish the two matrix domains (absolute
; ancestry versus the collected chain) and place each failure before or after
; the speculative matrix calls.
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r1_r2_length3_simple_ancestor %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=MATRIX3 %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r3_length3_nonsimple_ancestor %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=MATRIX-REJECT --implicit-check-not='loop-interchange: prepared function=' %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r15_nested_scalar_all %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=MATRIX-NESTED %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r16_depth3_nonempty_fission_row %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=MATRIX-R16 %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r6_runtime_then_static %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=SCAN-RUNTIME --implicit-check-not='loop-interchange: prepared a complete' --implicit-check-not='loop-interchange: rejected function=' %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r12_two_static_collected_order %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=SCAN-ORDER --implicit-check-not='loop-interchange: discovered a closed' --implicit-check-not='loop-interchange: prepared a complete' %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r14_strict_fp %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=STRICT --implicit-check-not='loop-interchange: prepared function=' %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r4_uncomputable_ancestor %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=COMPUTE-SKIP %{prep_checks} %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r4_overdepth_tail_candidate %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=DEPTH-SKIP %{prep_checks} %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=r8_ineligible_then_static %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=SCAN-SKIP --implicit-check-not="Couldn't compute backedge count" %}
;
; Every feature line per function is listed; the implicit exclusion rejects
; any other, including a second selected static plan or a runtime decline
; when a later static plan exists.
; CHAIN3-LABEL: loop-interchange: prepared function=r1_r2_length3_simple_ancestor{{ }}
; CHAIN3-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; CHAIN3-SAME:  absolute-depth=3 routing-depth=3 fission-rows=0 routing-rows=1 cross-deps=1 reductions=2
; CHAIN3-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; CHAIN3-SAME:  requirement-ids=[[R1_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R1_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-COMMON-LABEL: loop-interchange: rejected function=r9_no_static_two_ordinary{{ }}
; PREP-COMMON-SAME:  outer=for.j.header inner=for.r.header reason=post-inner region has no material epilogue memory slice{{$}}
; PREP-LATER-NEXT:   loop-interchange: rejected function=r9_no_static_two_ordinary outer=for.k.header inner=for.l.header reason=post-inner region has no material epilogue memory slice{{$}}
; CACHE-LABEL: loop-interchange: rejected function=r9_no_static_two_ordinary{{ }}
; CACHE-SAME:  outer=for.j.header inner=for.r.header reason=post-inner region has no material epilogue memory slice{{$}}
; CACHE-NEXT:  loop-interchange: rejected function=r9_no_static_two_ordinary outer=for.k.header inner=for.l.header reason=post-inner region has no material epilogue memory slice{{$}}
; CACHE-NEXT:  remark: <unknown>:0:0: Insufficient information to calculate the cost of loop for interchange.
; CACHE-NEXT:  remark: <unknown>:0:0: Insufficient information to calculate the cost of loop for interchange.
;
; PREP-COMMON-LABEL: loop-interchange: rejected function=r10_r11_budget_slot_sequence{{ }}
; PREP-COMMON-SAME:  outer=reject.outer inner=reject.inner reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-COMMON-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; PREP-LABEL: loop-interchange: prepared function=r10_r11_budget_slot_sequence{{ }}
; PREP-SAME:  outer=static.outer inner=static.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-SAME:  requirement-ids=[[R10_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R10_ID]]:object-containment/static/by-constant-max{{$}}
; BUDGET2-LABEL: loop-interchange: prepared function=r10_r11_budget_slot_sequence{{ }}
; BUDGET2-SAME:  outer=runtime.outer inner=runtime.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; BUDGET2-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; BUDGET2-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; BUDGET2-SAME:  requirement-ids=[[B2_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[B2_ID]]:object-containment/static/by-constant-max{{$}}
; BUDGET2-NEXT:  loop-interchange: rejected function=r10_r11_budget_slot_sequence outer=runtime.outer inner=runtime.inner reason=recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check{{$}}
; BUDGET2-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check
; CACHE-LABEL: loop-interchange: rejected function=r10_r11_budget_slot_sequence{{ }}
; CACHE-SAME:  outer=reject.outer inner=reject.inner reason=cross-partition dependence is not statically proved N-to-E{{$}}
; CACHE:       loop-interchange: rejected function=r10_r11_budget_slot_sequence outer=runtime.outer inner=runtime.inner reason=virtual extracted nest failed existing default profitability{{$}}
; CACHE:       loop-interchange: rejected function=r10_r11_budget_slot_sequence outer=static.outer inner=static.inner reason=virtual extracted nest failed existing default profitability{{$}}
;
; PREP-COMMON-LABEL: loop-interchange: prepared function=r12_two_static_collected_order{{ }}
; PREP-COMMON-SAME:  outer=first.outer inner=first.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-COMMON-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-COMMON-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-COMMON-SAME:  requirement-ids=[[R12_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R12_ID]]:object-containment/static/by-constant-max{{$}}
; CACHE-LABEL: loop-interchange: rejected function=r12_two_static_collected_order{{ }}
; CACHE-SAME:  outer=first.outer inner=first.inner reason=virtual extracted nest failed existing default profitability{{$}}
; CACHE:       loop-interchange: rejected function=r12_two_static_collected_order outer=second.outer inner=second.inner reason=virtual extracted nest failed existing default profitability{{$}}
; PREP-COMMON-LABEL: loop-interchange: prepared function=r13_explicit_cache_nested_chain{{ }}
; PREP-COMMON-SAME:  outer=candidate.outer inner=candidate.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-COMMON-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-COMMON-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-COMMON-SAME:  requirement-ids=[[R13_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R13_ID]]:object-containment/static/by-constant-max{{$}}
; CACHE-LABEL: loop-interchange: rejected function=r13_explicit_cache_nested_chain{{ }}
; CACHE-SAME:  outer=candidate.outer inner=candidate.inner reason=virtual extracted nest failed existing default profitability{{$}}
;
; PREP-COMMON-LABEL: loop-interchange: rejected function=r14_strict_fp{{ }}
; PREP-COMMON-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality{{$}}
; PREP-COMMON-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: virtual extracted nest failed existing interchange legality
; PREP-COMMON-NEXT:  remark: <unknown>:0:0: Only outer loops with induction or reduction PHI nodes can be interchanged currently.
; CACHE-LABEL: loop-interchange: rejected function=r14_strict_fp{{ }}
; CACHE-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality{{$}}
; PREP-COMMON-LABEL: loop-interchange: prepared function=r15_nested_scalar_all{{ }}
; PREP-COMMON-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-COMMON-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-COMMON-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-COMMON-SAME:  requirement-ids=[[R15_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R15_ID]]:object-containment/static/by-constant-max{{$}}
; CACHE-LABEL: loop-interchange: rejected function=r15_nested_scalar_all{{ }}
; CACHE-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing default profitability{{$}}
;
; unsafe_prefix_then_ordinary declines in discovery. Its all-unknown matrix
; failure and later success belong to ordinary routing.
; PREP-COMMON-LABEL: loop-interchange: rejected function=unsafe_prefix_then_ordinary{{ }}
; PREP-COMMON-SAME:  outer=loop.j2.header inner=loop.k2.header reason=post-inner region has no material epilogue memory slice{{$}}
; PREP-LATER-NEXT:   loop-interchange: rejected function=unsafe_prefix_then_ordinary outer=loop.x2.header inner=loop.y2.header reason=post-inner region has no material epilogue memory slice{{$}}
; CACHE-LABEL: loop-interchange: rejected function=unsafe_prefix_then_ordinary{{ }}
; CACHE-SAME:  outer=loop.j2.header inner=loop.k2.header reason=post-inner region has no material epilogue memory slice{{$}}
; CACHE-NEXT:  loop-interchange: rejected function=unsafe_prefix_then_ordinary outer=loop.x2.header inner=loop.y2.header reason=post-inner region has no material epilogue memory slice{{$}}
; CHAIN3-LABEL: loop-interchange: rejected function=r3_length3_nonsimple_ancestor{{ }}
; CHAIN3-SAME:  outer=outer.header inner=inner.header reason=routing dependence matrix is unavailable{{$}}
; CHAIN3-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: routing dependence matrix is unavailable
; DEPTH11-LABEL: loop-interchange: rejected function=r4_overdepth_tail_candidate{{ }}
; DEPTH11-SAME:  outer=l9.header inner=l10.header reason=virtual extracted nest has unknown surrounding dependence context{{$}}
; DEPTH11-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: virtual extracted nest has unknown surrounding dependence context
;
; r4_uncomputable_ancestor has no attempted pair; the extracted trace below
; checks the chain-wide exclusion.
; PREP-COMMON-LABEL: loop-interchange: prepared function=r5_first_static_then_later_ordinary{{ }}
; PREP-COMMON-SAME:  outer=first.outer inner=first.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-COMMON-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-COMMON-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-COMMON-SAME:  requirement-ids=[[R5_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R5_ID]]:object-containment/static/by-constant-max{{$}}
; CACHE-LABEL: loop-interchange: rejected function=r5_first_static_then_later_ordinary{{ }}
; CACHE-SAME:  outer=first.outer inner=first.inner reason=virtual extracted nest failed existing default profitability{{$}}
; CACHE:       loop-interchange: rejected function=r5_first_static_then_later_ordinary outer=second.outer inner=second.inner reason=post-inner region has no material epilogue memory slice{{$}}
;
; PREP-LATER-LABEL: loop-interchange: prepared function=r6_runtime_then_static{{ }}
; PREP-LATER-SAME:  outer=static.outer inner=static.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-LATER-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-LATER-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-LATER-SAME:  requirement-ids=[[R6_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R6_ID]]:object-containment/static/by-constant-max{{$}}
; BUDGET1-LABEL: loop-interchange: prepared function=r6_runtime_then_static{{ }}
; BUDGET1-SAME:  outer=runtime.outer inner=runtime.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; BUDGET1-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; BUDGET1-SAME:  byte-offset-proofs=1 requirements=2 bound=runtime-bound Wmin=4
; BUDGET1-SAME:  requirement-ids=[[B1_ID:[0-9]+]]:modular-outer-span/runtime/runtime,[[B1_ID]]:object-containment/static/by-constant-max{{$}}
; BUDGET1-NEXT:  loop-interchange: rejected function=r6_runtime_then_static outer=runtime.outer inner=runtime.inner reason=recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check{{$}}
; BUDGET1-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: recovered selected-outer array bound is a runtime value; safe distribution requires a runtime bound check
; CACHE-LABEL: loop-interchange: rejected function=r6_runtime_then_static{{ }}
; CACHE-SAME:  outer=runtime.outer inner=runtime.inner reason=virtual extracted nest failed existing default profitability{{$}}
; CACHE:       loop-interchange: rejected function=r6_runtime_then_static outer=static.outer inner=static.inner reason=virtual extracted nest failed existing default profitability{{$}}
;
; PREP-COMMON-LABEL: loop-interchange: rejected function=r7_rejected_then_static{{ }}
; PREP-COMMON-SAME:  outer=reject.outer inner=reject.inner reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-COMMON-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; PREP-LATER-LABEL: loop-interchange: prepared function=r7_rejected_then_static{{ }}
; PREP-LATER-SAME:  outer=static.outer inner=static.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-LATER-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-LATER-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-LATER-SAME:  requirement-ids=[[R7_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R7_ID]]:object-containment/static/by-constant-max{{$}}
; CACHE-LABEL: loop-interchange: rejected function=r7_rejected_then_static{{ }}
; CACHE-SAME:  outer=reject.outer inner=reject.inner reason=cross-partition dependence is not statically proved N-to-E{{$}}
; CACHE:       loop-interchange: rejected function=r7_rejected_then_static outer=static.outer inner=static.inner reason=virtual extracted nest failed existing default profitability{{$}}
;
; PREP-COMMON-LABEL: loop-interchange: prepared function=r8_ineligible_then_static{{ }}
; PREP-COMMON-SAME:  outer=static.outer inner=static.inner path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-COMMON-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-COMMON-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=4
; PREP-COMMON-SAME:  requirement-ids=[[R8_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[R8_ID]]:object-containment/static/by-constant-max{{$}}
; CACHE-LABEL: loop-interchange: rejected function=r8_ineligible_then_static{{ }}
; CACHE-SAME:  outer=static.outer inner=static.inner reason=virtual extracted nest failed existing default profitability{{$}}
;
; PREP-COMMON-LABEL: loop-interchange: prepared function=top_level_middle_epilogue_order{{ }}
; PREP-COMMON-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=4
; PREP-COMMON-SAME:  absolute-depth=2 routing-depth=2 fission-rows=1 routing-rows=1 cross-deps=0 reductions=0
; PREP-COMMON-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-COMMON-NEXT:  remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; PREP-COMMON-NEXT:  remark: <unknown>:0:0: Computed dependence info, invoking the transform.
; PREP-COMMON-NEXT:  remark: <unknown>:0:0: Computed dependence info, invoking the transform.
; CACHE-LABEL: loop-interchange: prepared function=top_level_middle_epilogue_order{{ }}
; CACHE-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=4
; CACHE-SAME:  absolute-depth=2 routing-depth=2 fission-rows=1 routing-rows=1 cross-deps=0 reductions=0
; CACHE-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; CACHE-NEXT:  remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; CACHE-NEXT:  remark: <unknown>:0:0: Computed dependence info, invoking the transform.
; CACHE-NEXT:  remark: <unknown>:0:0: Computed dependence info, invoking the transform.
; The depth-3 positive keeps both speculative matrices nonempty, so its
; ancestor-prefix proof is not vacuous when it accepts. absolute-depth=3 and
; routing-depth=3 give the two domains, and fission-rows=1 and routing-rows=1
; give their sizes. MATRIX-R16 below checks the ancestor column index and the
; complete printed dependence behind each row.
; CHAIN3-LABEL: loop-interchange: prepared function=r16_depth3_nonempty_fission_row{{ }}
; CHAIN3-SAME:  outer=l2.header inner=l3.header path-blocks=1 epilogue-insts=2 rematerialized=0
; CHAIN3-SAME:  absolute-depth=3 routing-depth=3 fission-rows=1 routing-rows=1 cross-deps=0 reductions=0
; CHAIN3-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; CHAIN3-NEXT:  remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; CHAIN3-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
;
; IGNORE-LABEL: loop-interchange: rejected function=r9_no_static_two_ordinary{{ }}
; IGNORE-SAME:  outer=for.j.header inner=for.r.header reason=post-inner region has no material epilogue memory slice{{$}}
; IGNORE-NEXT:  loop-interchange: rejected function=r9_no_static_two_ordinary outer=for.k.header inner=for.l.header reason=post-inner region has no material epilogue memory slice{{$}}
; IGNORE-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; IGNORE-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; IGNORE-NEXT:  remark: <unknown>:0:0: Computed dependence info, invoking the transform.
; IGNORE-NEXT:  loop-interchange: rejected function=r10_r11_budget_slot_sequence{{ }}
;
; Both ordinary interchanges are visible through their PHIs and branches.
; IGNORE-IR-LABEL: define void @r9_no_static_two_ordinary(
; IGNORE-IR:       for.j.header:
; IGNORE-IR-NEXT:    %j = phi i64 [ %j.next, %for.j.latch ], [ 0, %[[J_PH:[^ ]+]] ]
; IGNORE-IR:       for.r.header:
; IGNORE-IR-NEXT:    %r = phi i64 [ %[[R_NEXT:[^ ,]+]], %[[R_LATCH:[^ ]+]] ], [ 0, %[[R_PH:[^ ]+]] ]
; IGNORE-IR-NEXT:    br label %[[J_PH]]
; IGNORE-IR:         %a.element = getelementptr [64 x i8], ptr %A, i64 %j, i64 %r
; IGNORE-IR-NEXT:    store i8 0, ptr %a.element, align 1
; IGNORE-IR:       [[R_LATCH]]:
; IGNORE-IR-NEXT:    %[[R_NEXT]] = add i64 %r, 1
; IGNORE-IR:       for.k.header:
; IGNORE-IR-NEXT:    %k = phi i64 [ %k.next, %for.k.latch ], [ 0, %[[K_PH:[^ ]+]] ]
; IGNORE-IR:       for.l.header:
; IGNORE-IR-NEXT:    %l = phi i64 [ %[[L_NEXT:[^ ,]+]], %[[L_LATCH:[^ ]+]] ], [ 0, %[[L_PH:[^ ]+]] ]
; IGNORE-IR-NEXT:    br label %[[K_PH]]
; IGNORE-IR:         %b.element = getelementptr [64 x i8], ptr %B, i64 %l, i64 %k
; IGNORE-IR-NEXT:    store i8 0, ptr %b.element, align 1
; IGNORE-IR:       [[L_LATCH]]:
; IGNORE-IR-NEXT:    %[[L_NEXT]] = add i64 %l, 1
; IGNORE-IR:       {{^}}}{{$}}
;
; MATRIX3-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'r1_r2_length3_simple_ancestor' with 1 path block(s), 4 instruction(s), and 0 rematerialized loop-local definition(s).
; MATRIX3-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 3 loads/stores over 20 instructions.
; MATRIX3-NEXT:  loop-interchange: flattened byte-offset proof accepted an N/E pair:
; MATRIX3-NEXT:   N:   %av = load i64, ptr %aidx, align 8
; MATRIX3-NEXT:   E:   store i64 %diag.next, ptr %diag, align 8
; MATRIX3-NEXT:   width = 4, element bytes = 8, bounded modular equality implies equal outer iteration.
; MATRIX3-NEXT:  Found 1 Loads and Stores to analyze
; MATRIX3-NEXT:  Found 4 Loads and Stores to analyze
; MATRIX3-NEXT:  Found output dependency between Src and Dst
; MATRIX3-NEXT:   Src:  store i64 %ancestor.value, ptr %AncestorOut, align 8
; MATRIX3-NEXT:   Dst:  store i64 %ancestor.value, ptr %AncestorOut, align 8
; MATRIX3:       loop-interchange: prepared a complete outer-epilogue fission plan in function 'r1_r2_length3_simple_ancestor': Outer 'outer.header', Inner 'inner.header', absolute columns 1/2, routing columns 1/2, 1 cross dependence(s).
; MATRIX3-NEXT:  loop-interchange: prepared function=r1_r2_length3_simple_ancestor{{ }}
; MATRIX3-SAME:  rematerialized=0{{ }}
; MATRIX3:       Processing LoopList of size = 3 containing the following loops:
; MATRIX3:       Found 6 Loads and Stores to analyze
; MATRIX3:       Dependency matrix before interchange:
; MATRIX3-NEXT:  {{^}}* I I{{ *$}}
; MATRIX3-NEXT:  {{^}}* * I{{ *$}}
; MATRIX3-NEXT:  {{^}}* = I{{ *$}}
; MATRIX3-NEXT:  Processing InnerLoopId = 2 and OuterLoopId = 1
; MATRIX3:       Cannot prove legality, not interchanging loops 'root.header' and 'outer.header'
;
; MATRIX-REJECT-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'r3_length3_nonsimple_ancestor' with 1 path block(s), 4 instruction(s), and 0 rematerialized loop-local definition(s).
; MATRIX-REJECT:       width = 4, element bytes = 8, bounded modular equality implies equal outer iteration.
; MATRIX-REJECT-NEXT:  Found 1 Loads and Stores to analyze
; MATRIX-REJECT-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'r3_length3_nonsimple_ancestor': Outer 'outer.header', Inner 'inner.header': routing dependence matrix is unavailable
; MATRIX-REJECT-NEXT:  loop-interchange: rejected function=r3_length3_nonsimple_ancestor outer=outer.header inner=inner.header reason=routing dependence matrix is unavailable
; MATRIX-REJECT-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: routing dependence matrix is unavailable
; MATRIX-REJECT-NEXT:  Processing LoopList of size = 3 containing the following loops:
; MATRIX-REJECT:       Populating dependency matrix failed
;
; MATRIX-NESTED-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'r15_nested_scalar_all' with 1 path block(s), 4 instruction(s), and 0 rematerialized loop-local definition(s).
; MATRIX-NESTED:       loop-interchange: flattened byte-offset proof accepted an N/E pair:
; MATRIX-NESTED-NEXT:   N:   %dread = load double, ptr %didx, align 8
; MATRIX-NESTED-NEXT:   E:   store double %dn, ptr %ddiag, align 8
; MATRIX-NESTED-NEXT:   width = 1335, element bytes = 8, bounded modular equality implies equal outer iteration.
; MATRIX-NESTED-NEXT:  Found 1 Loads and Stores to analyze
; MATRIX-NESTED-NEXT:  Found 1 Loads and Stores to analyze
; MATRIX-NESTED:       loop-interchange: prepared a complete outer-epilogue fission plan in function 'r15_nested_scalar_all': Outer 'outer.header', Inner 'inner.header', absolute columns 1/2, routing columns 0/1, 1 cross dependence(s).
; MATRIX-NESTED-NEXT:  loop-interchange: prepared function=r15_nested_scalar_all{{ }}
; MATRIX-NESTED-SAME:  rematerialized=0{{ }}
; MATRIX-NESTED:       Processing LoopList of size = 2 containing the following loops:
; MATRIX-NESTED:       Dependency matrix before interchange:
; MATRIX-NESTED-NEXT:  {{^}}* I{{ *$}}
; MATRIX-NESTED-NEXT:  {{^}}= I{{ *$}}
; MATRIX-NESTED-NEXT:  Processing InnerLoopId = 1 and OuterLoopId = 0
; MATRIX-NESTED:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
; The two speculative matrices carry independent domains. The plan summary
; gives the column indices: absolute columns 1/2 puts the selected outer at
; index 1, so index 0 is the ancestor column that the prefix proof reads, and
; routing columns 1/2 indexes the collected chain. The printer reports sizes
; only, so each row is bound here by the complete dependence that produced it.
; The ordinary depth-3 matrix that follows preparation is anchored as whole
; rows, and it is a third, independent domain.
; MATRIX-R16-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'r16_depth3_nonempty_fission_row' with 1 path block(s), 2 instruction(s), and 0 rematerialized loop-local definition(s).
; MATRIX-R16-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 3 loads/stores over 16 instructions.
; MATRIX-R16-NEXT:  Found 2 Loads and Stores to analyze
; MATRIX-R16-NEXT:  Found anti dependency between Src and Dst
; MATRIX-R16-NEXT:   Src:  %v = load double, ptr %ap, align 8
; MATRIX-R16-NEXT:   Dst:  store double %w, ptr %ap, align 8
; MATRIX-R16-NEXT:  Found 2 Loads and Stores to analyze
; MATRIX-R16-NEXT:  Found anti dependency between Src and Dst
; MATRIX-R16-NEXT:   Src:  %v = load double, ptr %ap, align 8
; MATRIX-R16-NEXT:   Dst:  store double %w, ptr %ap, align 8
; MATRIX-R16-NEXT:  Checking if loops 'l2.header' and 'l3.header' are tightly nested
; MATRIX-R16-NEXT:  Checking instructions in Loop header and Loop latch
; MATRIX-R16-NEXT:  Loops are perfectly nested
; MATRIX-R16-NEXT:  Cost = -1
; MATRIX-R16-NEXT:  {{^}}loop-interchange: prepared a complete outer-epilogue fission plan in function 'r16_depth3_nonempty_fission_row': Outer 'l2.header', Inner 'l3.header', absolute columns 1/2, routing columns 1/2, 0 cross dependence(s).{{$}}
; MATRIX-R16-NEXT:  loop-interchange: prepared function=r16_depth3_nonempty_fission_row{{ }}
; MATRIX-R16-SAME:  rematerialized=0{{ }}
; MATRIX-R16-SAME:  absolute-depth=3 routing-depth=3 fission-rows=1 routing-rows=1{{ }}
; MATRIX-R16-NEXT:  Processing LoopList of size = 3 containing the following loops:
; MATRIX-R16:       Dependency matrix before interchange:
; MATRIX-R16-NEXT:  {{^}}= = ={{ *$}}
; MATRIX-R16-NEXT:  {{^}}* = I{{ *$}}
; MATRIX-R16-NEXT:  Processing InnerLoopId = 2 and OuterLoopId = 1
; MATRIX-R16:       Cannot prove legality, not interchanging loops 'l1.header' and 'l2.header'
;
; SCAN-RUNTIME-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'r6_runtime_then_static'
; SCAN-RUNTIME:       loop-interchange: prepared a complete outer-epilogue fission plan in function 'r6_runtime_then_static': Outer 'runtime.outer', Inner 'runtime.inner', absolute columns 1/2, routing columns 0/1, 1 cross dependence(s).
; SCAN-RUNTIME:       loop-interchange: discovered a closed outer-loop epilogue in function 'r6_runtime_then_static'
; SCAN-RUNTIME:       loop-interchange: prepared a complete outer-epilogue fission plan in function 'r6_runtime_then_static': Outer 'static.outer', Inner 'static.inner', absolute columns 1/2, routing columns 0/1, 1 cross dependence(s).
; SCAN-RUNTIME-NEXT:  loop-interchange: prepared function=r6_runtime_then_static outer=static.outer inner=static.inner{{ }}
; SCAN-RUNTIME-SAME:  rematerialized=0{{ }}
; SCAN-RUNTIME-NEXT:  Processing LoopList of size = 2 containing the following loops:
; SCAN-RUNTIME:       Cannot prove legality, not interchanging loops 'runtime.outer' and 'runtime.inner'
; SCAN-RUNTIME:       Cannot prove legality, not interchanging loops 'static.outer' and 'static.inner'
;
; SCAN-ORDER-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'r12_two_static_collected_order'
; SCAN-ORDER:       loop-interchange: prepared a complete outer-epilogue fission plan in function 'r12_two_static_collected_order': Outer 'first.outer', Inner 'first.inner', absolute columns 1/2, routing columns 0/1, 1 cross dependence(s).
; SCAN-ORDER-NEXT:  loop-interchange: prepared function=r12_two_static_collected_order outer=first.outer inner=first.inner{{ }}
; SCAN-ORDER-SAME:  rematerialized=0{{ }}
; SCAN-ORDER-NEXT:  Processing LoopList of size = 2 containing the following loops:
; SCAN-ORDER:       Cannot prove legality, not interchanging loops 'first.outer' and 'first.inner'
; SCAN-ORDER-NEXT:  Processing LoopList of size = 2 containing the following loops:
; SCAN-ORDER:       Cannot prove legality, not interchanging loops 'second.outer' and 'second.inner'
;
; STRICT-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'r14_strict_fp' with 1 path block(s), 4 instruction(s), and 0 rematerialized loop-local definition(s).
; STRICT-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 3 loads/stores over 20 instructions.
; STRICT-NEXT:  Found 1 Loads and Stores to analyze
; STRICT-NEXT:  Found 1 Loads and Stores to analyze
; STRICT-NEXT:  Failed to recognize PHI as an induction or reduction.
; STRICT-NEXT:  Failed to find inner loop inductions or found unsupported reductions.
; STRICT-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'r14_strict_fp': Outer 'outer.header', Inner 'inner.header': virtual extracted nest failed existing interchange legality
; STRICT-NEXT:  loop-interchange: rejected function=r14_strict_fp outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality
; STRICT-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: virtual extracted nest failed existing interchange legality
; STRICT-NEXT:  Processing LoopList of size = 2 containing the following loops:
; STRICT:       remark: <unknown>:0:0: Only outer loops with induction or reduction PHI nodes can be interchanged currently.
; STRICT:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
;
; COMPUTE-SKIP:      remark: <unknown>:0:0: Computed dependence info, invoking the transform.
; COMPUTE-SKIP-NEXT: Couldn't compute backedge count
; COMPUTE-SKIP-NEXT: Couldn't compute backedge count
; COMPUTE-SKIP-NEXT: Not valid loop candidate for interchange
; DEPTH-SKIP:        remark: <unknown>:0:0: Computed dependence info, invoking the transform.
; DEPTH-SKIP-NEXT:   Unsupported depth of loop nest 11, the supported range is [2, 10].
; DEPTH-SKIP-NEXT:   remark: <unknown>:0:0: Unsupported depth of loop nest, the supported range is [2, 10].
;
; SCAN-SKIP:       remark: <unknown>:0:0: Computed dependence info, invoking the transform.
; SCAN-SKIP-NEXT:  Couldn't compute backedge count
; SCAN-SKIP-NEXT:  loop-interchange: discovered a closed outer-loop epilogue in function 'r8_ineligible_then_static'
; SCAN-SKIP:       loop-interchange: prepared a complete outer-epilogue fission plan in function 'r8_ineligible_then_static': Outer 'static.outer', Inner 'static.inner', absolute columns 1/2, routing columns 0/1, 1 cross dependence(s).
; SCAN-SKIP-NEXT:  loop-interchange: prepared function=r8_ineligible_then_static outer=static.outer inner=static.inner{{ }}
; SCAN-SKIP-SAME:  rematerialized=0{{ }}
; SCAN-SKIP-NEXT:  Couldn't compute backedge count
; SCAN-SKIP-NEXT:  Not valid loop candidate for interchange
; SCAN-SKIP-NEXT:  Processing LoopList of size = 2 containing the following loops:
; SCAN-SKIP:       Cannot prove legality, not interchanging loops 'static.outer' and 'static.inner'
;
; CACHE-IR-LABEL: define void @r9_no_static_two_ordinary(
; CACHE-IR:       for.k.header:
; CACHE-IR:         br label %for.l.header
; CACHE-IR:       for.l.header:
; CACHE-IR:         %l = phi i64 [ 0, %for.k.header ], [ %l.next, %for.l.header ]
; CACHE-IR:       {{^}}}{{$}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @r1_r2_length3_simple_ancestor(
    ptr noalias %AncestorIn, ptr noalias %AncestorOut,
    ptr noalias dereferenceable(128) %A, ptr noalias %R) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  %ancestor.value = load i64, ptr %AncestorIn, align 8
  store i64 %ancestor.value, ptr %AncestorOut, align 8
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %outer.latch ]
  %sum.i = phi i64 [ 0, %root.header ], [ %sum.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sum.j = phi i64 [ %sum.i, %outer.header ], [ %sum.next.j, %inner.header ]
  %aidx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %av = load i64, ptr %aidx, align 8
  %sum.next.j = add i64 %sum.j, %av
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header
epilogue:
  %sum.next = phi i64 [ %sum.next.j, %inner.header ]
  %diag = getelementptr inbounds [4 x i64], ptr %A, i64 %i, i64 %i
  %diag.value = load i64, ptr %diag, align 8
  %diag.next = add i64 %diag.value, 1
  store i64 %diag.next, ptr %diag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %root.latch, label %outer.header
root.latch:
  %sum.result = phi i64 [ %sum.next, %outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

define void @r9_no_static_two_ordinary(ptr noalias %A, ptr noalias %B) {
entry:
  br label %for.i.header
for.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.i.latch ]
  br label %for.j.header
for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %j.next, %for.j.latch ]
  br label %for.r.header
for.r.header:
  %r = phi i64 [ 0, %for.j.header ], [ %r.next, %for.r.header ]
  %a.element = getelementptr [64 x i8], ptr %A, i64 %j, i64 %r
  store i8 0, ptr %a.element
  %r.next = add i64 %r, 1
  %r.done = icmp eq i64 %r.next, 64
  br i1 %r.done, label %for.j.latch, label %for.r.header
for.j.latch:
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 64
  br i1 %j.done, label %for.k.header, label %for.j.header
for.k.header:
  %k = phi i64 [ 0, %for.j.latch ], [ %k.next, %for.k.latch ]
  br label %for.l.header
for.l.header:
  %l = phi i64 [ 0, %for.k.header ], [ %l.next, %for.l.header ]
  %b.element = getelementptr [64 x i8], ptr %B, i64 %l, i64 %k
  store i8 0, ptr %b.element
  %l.next = add i64 %l, 1
  %l.done = icmp eq i64 %l.next, 64
  br i1 %l.done, label %for.k.latch, label %for.l.header
for.k.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 64
  br i1 %k.done, label %for.i.latch, label %for.k.header
for.i.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 64
  br i1 %i.done, label %exit, label %for.i.header
exit:
  ret void
}

define void @r10_r11_budget_slot_sequence(
    ptr noalias %Limit, ptr noalias %B, ptr noalias %S,
    ptr noalias %Scratch, ptr noalias dereferenceable(128) %RuntimeA,
    ptr noalias dereferenceable(128) %StaticA, ptr noalias %R, i64 %n) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %ineligible.outer
ineligible.outer:
  %u = phi i64 [ 0, %root.header ], [ %u.next, %ineligible.outer.latch ]
  br label %ineligible.inner
ineligible.inner:
  %v = phi i64 [ 0, %ineligible.outer ], [ %v.next, %ineligible.inner ]
  %bidx = getelementptr inbounds [4 x i64], ptr %B, i64 %v, i64 %u
  store i64 %u, ptr %bidx, align 8
  %v.next = add i64 %v, 1
  %v.done = icmp eq i64 %v.next, 4
  br i1 %v.done, label %ineligible.outer.latch, label %ineligible.inner
ineligible.outer.latch:
  %limit = load volatile i64, ptr %Limit, align 8
  %u.next = add i64 %u, 1
  %u.continue = icmp ult i64 %u.next, %limit
  br i1 %u.continue, label %ineligible.outer, label %reject.outer
reject.outer:
  %ri = phi i64 [ 0, %ineligible.outer.latch ], [ %ri.next, %reject.outer.latch ]
  %si = getelementptr inbounds i64, ptr %S, i64 %ri
  br label %reject.inner
reject.inner:
  %rj = phi i64 [ 0, %reject.outer ], [ %rj.next, %reject.inner ]
  %sv = load i64, ptr %si, align 8
  %scratch = getelementptr inbounds [4 x i64], ptr %Scratch, i64 %rj, i64 %ri
  store i64 %sv, ptr %scratch, align 8
  %rj.next = add i64 %rj, 1
  %rj.done = icmp eq i64 %rj.next, 4
  br i1 %rj.done, label %reject.epilogue, label %reject.inner
reject.epilogue:
  %ri.plus1 = add i64 %ri, 1
  %si.plus1 = getelementptr inbounds i64, ptr %S, i64 %ri.plus1
  store i64 %ri, ptr %si.plus1, align 8
  br label %reject.outer.latch
reject.outer.latch:
  %ri.next = add i64 %ri, 1
  %ri.done = icmp eq i64 %ri.next, 4
  br i1 %ri.done, label %runtime.outer, label %reject.outer
runtime.outer:
  %ti = phi i64 [ 0, %reject.outer.latch ], [ %ti.next, %runtime.outer.latch ]
  %tsum.i = phi i64 [ 0, %reject.outer.latch ], [ %tsum.next, %runtime.outer.latch ]
  br label %runtime.inner
runtime.inner:
  %tj = phi i64 [ 0, %runtime.outer ], [ %tj.next, %runtime.inner ]
  %tsum.j = phi i64 [ %tsum.i, %runtime.outer ], [ %tsum.next.j, %runtime.inner ]
  %tidx = getelementptr inbounds [4 x i64], ptr %RuntimeA, i64 %tj, i64 %ti
  %tv = load i64, ptr %tidx, align 8
  %tsum.next.j = add i64 %tsum.j, %tv
  %tj.next = add i64 %tj, 1
  %tj.done = icmp eq i64 %tj.next, 4
  br i1 %tj.done, label %runtime.epilogue, label %runtime.inner
runtime.epilogue:
  %tsum.next = phi i64 [ %tsum.next.j, %runtime.inner ]
  %tdiag = getelementptr inbounds [4 x i64], ptr %RuntimeA, i64 %ti, i64 %ti
  %tdv = load i64, ptr %tdiag, align 8
  %tdn = add i64 %tdv, 1
  store i64 %tdn, ptr %tdiag, align 8
  br label %runtime.outer.latch
runtime.outer.latch:
  %ti.next = add i64 %ti, 1
  %ti.done = icmp eq i64 %ti.next, %n
  br i1 %ti.done, label %static.outer, label %runtime.outer
static.outer:
  %i = phi i64 [ 0, %runtime.outer.latch ], [ %i.next, %static.outer.latch ]
  %sum.i = phi i64 [ %tsum.next, %runtime.outer.latch ], [ %sum.next, %static.outer.latch ]
  br label %static.inner
static.inner:
  %j = phi i64 [ 0, %static.outer ], [ %j.next, %static.inner ]
  %sum.j = phi i64 [ %sum.i, %static.outer ], [ %sum.next.j, %static.inner ]
  %idx = getelementptr inbounds [4 x i64], ptr %StaticA, i64 %j, i64 %i
  %value = load i64, ptr %idx, align 8
  %sum.next.j = add i64 %sum.j, %value
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %static.epilogue, label %static.inner
static.epilogue:
  %sum.next = phi i64 [ %sum.next.j, %static.inner ]
  %diag = getelementptr inbounds [4 x i64], ptr %StaticA, i64 %i, i64 %i
  %dv = load i64, ptr %diag, align 8
  %dn = add i64 %dv, 1
  store i64 %dn, ptr %diag, align 8
  br label %static.outer.latch
static.outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %root.latch, label %static.outer
root.latch:
  %sum.result = phi i64 [ %sum.next, %static.outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

define void @r12_two_static_collected_order(
    ptr noalias dereferenceable(128) %A0,
    ptr noalias dereferenceable(128) %A1, ptr noalias %R) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %first.outer
first.outer:
  %i0 = phi i64 [ 0, %root.header ], [ %i0.next, %first.outer.latch ]
  %sum0.i = phi i64 [ 0, %root.header ], [ %sum0.next, %first.outer.latch ]
  br label %first.inner
first.inner:
  %j0 = phi i64 [ 0, %first.outer ], [ %j0.next, %first.inner ]
  %sum0.j = phi i64 [ %sum0.i, %first.outer ], [ %sum0.next.j, %first.inner ]
  %idx0 = getelementptr inbounds [4 x i64], ptr %A0, i64 %j0, i64 %i0
  %value0 = load i64, ptr %idx0, align 8
  %sum0.next.j = add i64 %sum0.j, %value0
  %j0.next = add i64 %j0, 1
  %j0.done = icmp eq i64 %j0.next, 4
  br i1 %j0.done, label %first.epilogue, label %first.inner
first.epilogue:
  %sum0.next = phi i64 [ %sum0.next.j, %first.inner ]
  %diag0 = getelementptr inbounds [4 x i64], ptr %A0, i64 %i0, i64 %i0
  %dv0 = load i64, ptr %diag0, align 8
  %dn0 = add i64 %dv0, 11
  store i64 %dn0, ptr %diag0, align 8
  br label %first.outer.latch
first.outer.latch:
  %i0.next = add i64 %i0, 1
  %i0.done = icmp eq i64 %i0.next, 4
  br i1 %i0.done, label %second.outer, label %first.outer
second.outer:
  %i1 = phi i64 [ 0, %first.outer.latch ], [ %i1.next, %second.outer.latch ]
  %sum1.i = phi i64 [ %sum0.next, %first.outer.latch ], [ %sum1.next, %second.outer.latch ]
  br label %second.inner
second.inner:
  %j1 = phi i64 [ 0, %second.outer ], [ %j1.next, %second.inner ]
  %sum1.j = phi i64 [ %sum1.i, %second.outer ], [ %sum1.next.j, %second.inner ]
  %idx1 = getelementptr inbounds [4 x i64], ptr %A1, i64 %j1, i64 %i1
  %value1 = load i64, ptr %idx1, align 8
  %sum1.next.j = add i64 %sum1.j, %value1
  %j1.next = add i64 %j1, 1
  %j1.done = icmp eq i64 %j1.next, 4
  br i1 %j1.done, label %second.epilogue, label %second.inner
second.epilogue:
  %sum1.next = phi i64 [ %sum1.next.j, %second.inner ]
  %diag1 = getelementptr inbounds [4 x i64], ptr %A1, i64 %i1, i64 %i1
  %dv1 = load i64, ptr %diag1, align 8
  %dn1 = add i64 %dv1, 22
  store i64 %dn1, ptr %diag1, align 8
  br label %second.outer.latch
second.outer.latch:
  %i1.next = add i64 %i1, 1
  %i1.done = icmp eq i64 %i1.next, 4
  br i1 %i1.done, label %root.latch, label %second.outer
root.latch:
  %sum.result = phi i64 [ %sum1.next, %second.outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

define void @r13_explicit_cache_nested_chain(
    ptr noalias dereferenceable(128) %A, ptr noalias %B, ptr noalias %R) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %candidate.outer
candidate.outer:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %candidate.outer.latch ]
  %sum.i = phi i64 [ 0, %root.header ], [ %sum.next, %candidate.outer.latch ]
  br label %candidate.inner
candidate.inner:
  %j = phi i64 [ 0, %candidate.outer ], [ %j.next, %candidate.inner ]
  %sum.j = phi i64 [ %sum.i, %candidate.outer ], [ %sum.next.j, %candidate.inner ]
  %idx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %value = load i64, ptr %idx, align 8
  %sum.next.j = add i64 %sum.j, %value
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %candidate.epilogue, label %candidate.inner
candidate.epilogue:
  %sum.next = phi i64 [ %sum.next.j, %candidate.inner ]
  %diag = getelementptr inbounds [4 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %diag, align 8
  %dn = add i64 %dv, 1
  store i64 %dn, ptr %diag, align 8
  br label %candidate.outer.latch
candidate.outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %sibling.header, label %candidate.outer
sibling.header:
  %s = phi i64 [ 0, %candidate.outer.latch ], [ %s.next, %sibling.header ]
  %bp = getelementptr inbounds i64, ptr %B, i64 %s
  store i64 %s, ptr %bp, align 8
  %s.next = add i64 %s, 1
  %s.done = icmp eq i64 %s.next, 4
  br i1 %s.done, label %root.latch, label %sibling.header
root.latch:
  %sum.result = phi i64 [ %sum.next, %sibling.header ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

define void @r14_strict_fp(ptr noalias %A, ptr noalias %D, ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %aidx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %chk.next.j = fadd double %chk.j, %av
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
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

define void @r15_nested_scalar_all(
    ptr noalias dereferenceable(14257800) %D, ptr noalias %R) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %root.header ], [ %chk.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %didx = getelementptr inbounds [1335 x double], ptr %D, i64 %j, i64 %i
  %dread = load double, ptr %didx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %dread
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dv = load double, ptr %ddiag, align 8
  %dn = fadd double %dv, 1.000000e+00
  store double %dn, ptr %ddiag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %outer.exit, label %outer.header
outer.exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  %rk = getelementptr inbounds double, ptr %R, i64 %k
  store double %chk.res, ptr %rk, align 8
  br label %sibling.header
sibling.header:
  %s = phi i64 [ 0, %outer.exit ], [ %s.next, %sibling.header ]
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 2
  br i1 %s.ec, label %root.latch, label %sibling.header
root.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 8
  br i1 %k.ec, label %exit, label %root.header
exit:
  ret void
}

define void @unsafe_prefix_then_ordinary(ptr %A, ptr %C) {
entry:
  br label %loop.h.header
loop.h.header:
  %h = phi i64 [ 0, %entry ], [ %h.next, %loop.h.latch ]
  br label %loop.i2.header
loop.i2.header:
  %i = phi i64 [ 0, %loop.h.header ], [ %i.next, %loop.i2.latch ]
  br label %loop.j2.header
loop.j2.header:
  %j = phi i64 [ 0, %loop.i2.header ], [ %j.next, %loop.j2.latch ]
  br label %loop.k2.header
loop.k2.header:
  %k = phi i64 [ 0, %loop.j2.header ], [ %k.next, %loop.k2.latch ]
  %c.ptr = getelementptr double, ptr %C, i64 %k
  %c.val = load double, ptr %c.ptr, align 8
  %sum = fadd double %c.val, 1.000000e+00
  %jrow = mul nuw nsw i64 %j, 8
  %aidx = add nuw nsw i64 %jrow, %k
  %a.ptr = getelementptr double, ptr %A, i64 %aidx
  store double %sum, ptr %a.ptr, align 8
  br label %loop.k2.latch
loop.k2.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, 8
  br i1 %k.done, label %loop.j2.latch, label %loop.k2.header
loop.j2.latch:
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 8
  br i1 %j.done, label %loop.x2.header, label %loop.j2.header
loop.x2.header:
  %x = phi i64 [ 0, %loop.j2.latch ], [ %x.next, %loop.x2.latch ]
  br label %loop.y2.header
loop.y2.header:
  %y = phi i64 [ 0, %loop.x2.header ], [ %y.next, %loop.y2.latch ]
  %row = mul nuw nsw i64 %y, 8
  %idx = add nuw nsw i64 %row, %x
  %axy.ptr = getelementptr double, ptr %A, i64 %idx
  %old = load double, ptr %axy.ptr, align 8
  %new = fadd double %old, 2.000000e+00
  store double %new, ptr %axy.ptr, align 8
  br label %loop.y2.latch
loop.y2.latch:
  %y.next = add nuw nsw i64 %y, 1
  %y.done = icmp eq i64 %y.next, 8
  br i1 %y.done, label %loop.x2.latch, label %loop.y2.header
loop.x2.latch:
  %x.next = add nuw nsw i64 %x, 1
  %x.done = icmp eq i64 %x.next, 8
  br i1 %x.done, label %loop.i2.latch, label %loop.x2.header
loop.i2.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 8
  br i1 %i.done, label %loop.h.latch, label %loop.i2.header
loop.h.latch:
  %h.next = add nuw nsw i64 %h, 1
  %h.done = icmp eq i64 %h.next, 8
  br i1 %h.done, label %exit, label %loop.h.header
exit:
  ret void
}

define void @r3_length3_nonsimple_ancestor(
    ptr noalias %Ancestor, ptr noalias dereferenceable(128) %A,
    ptr noalias %R) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  %ancestor.value = load volatile i64, ptr %Ancestor, align 8
  store volatile i64 %ancestor.value, ptr %Ancestor, align 8
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %outer.latch ]
  %sum.i = phi i64 [ 0, %root.header ], [ %sum.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sum.j = phi i64 [ %sum.i, %outer.header ], [ %sum.next.j, %inner.header ]
  %aidx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %av = load i64, ptr %aidx, align 8
  %sum.next.j = add i64 %sum.j, %av
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header
epilogue:
  %sum.next = phi i64 [ %sum.next.j, %inner.header ]
  %diag = getelementptr inbounds [4 x i64], ptr %A, i64 %i, i64 %i
  %diag.value = load i64, ptr %diag, align 8
  %diag.next = add i64 %diag.value, 1
  store i64 %diag.next, ptr %diag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %root.latch, label %outer.header
root.latch:
  %sum.result = phi i64 [ %sum.next, %outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

define void @r4_overdepth_tail_candidate(ptr noalias %A, ptr noalias %E) {
entry:
  br label %l0.header
l0.header:
  %v0 = phi i64 [ 0, %entry ], [ %v0.next, %l0.latch ]
  br label %l1.header
l1.header:
  %v1 = phi i64 [ 0, %l0.header ], [ %v1.next, %l1.latch ]
  br label %l2.header
l2.header:
  %v2 = phi i64 [ 0, %l1.header ], [ %v2.next, %l2.latch ]
  br label %l3.header
l3.header:
  %v3 = phi i64 [ 0, %l2.header ], [ %v3.next, %l3.latch ]
  br label %l4.header
l4.header:
  %v4 = phi i64 [ 0, %l3.header ], [ %v4.next, %l4.latch ]
  br label %l5.header
l5.header:
  %v5 = phi i64 [ 0, %l4.header ], [ %v5.next, %l5.latch ]
  br label %l6.header
l6.header:
  %v6 = phi i64 [ 0, %l5.header ], [ %v6.next, %l6.latch ]
  br label %l7.header
l7.header:
  %v7 = phi i64 [ 0, %l6.header ], [ %v7.next, %l7.latch ]
  br label %l8.header
l8.header:
  %v8 = phi i64 [ 0, %l7.header ], [ %v8.next, %l8.latch ]
  br label %l9.header
l9.header:
  %v9 = phi i64 [ 0, %l8.header ], [ %v9.next, %l9.latch ]
  br label %l10.header
l10.header:
  %v10 = phi i64 [ 0, %l9.header ], [ %v10.next, %l10.header ]
  %idx = getelementptr inbounds [4 x i64], ptr %A, i64 %v10, i64 %v9
  %value = load i64, ptr %idx, align 8
  store i64 %value, ptr %idx, align 8
  %v10.next = add i64 %v10, 1
  %v10.done = icmp eq i64 %v10.next, 4
  br i1 %v10.done, label %l9.epilogue, label %l10.header
l9.epilogue:
  %ep = getelementptr inbounds i64, ptr %E, i64 %v9
  store i64 %v9, ptr %ep, align 8
  br label %l9.latch
l9.latch:
  %v9.next = add i64 %v9, 1
  %v9.done = icmp eq i64 %v9.next, 4
  br i1 %v9.done, label %l8.latch, label %l9.header
l8.latch:
  %v8.next = add i64 %v8, 1
  %v8.done = icmp eq i64 %v8.next, 2
  br i1 %v8.done, label %l7.latch, label %l8.header
l7.latch:
  %v7.next = add i64 %v7, 1
  %v7.done = icmp eq i64 %v7.next, 2
  br i1 %v7.done, label %l6.latch, label %l7.header
l6.latch:
  %v6.next = add i64 %v6, 1
  %v6.done = icmp eq i64 %v6.next, 2
  br i1 %v6.done, label %l5.latch, label %l6.header
l5.latch:
  %v5.next = add i64 %v5, 1
  %v5.done = icmp eq i64 %v5.next, 2
  br i1 %v5.done, label %l4.latch, label %l5.header
l4.latch:
  %v4.next = add i64 %v4, 1
  %v4.done = icmp eq i64 %v4.next, 2
  br i1 %v4.done, label %l3.latch, label %l4.header
l3.latch:
  %v3.next = add i64 %v3, 1
  %v3.done = icmp eq i64 %v3.next, 2
  br i1 %v3.done, label %l2.latch, label %l3.header
l2.latch:
  %v2.next = add i64 %v2, 1
  %v2.done = icmp eq i64 %v2.next, 2
  br i1 %v2.done, label %l1.latch, label %l2.header
l1.latch:
  %v1.next = add i64 %v1, 1
  %v1.done = icmp eq i64 %v1.next, 2
  br i1 %v1.done, label %l0.latch, label %l1.header
l0.latch:
  %v0.next = add i64 %v0, 1
  %v0.done = icmp eq i64 %v0.next, 2
  br i1 %v0.done, label %exit, label %l0.header
exit:
  ret void
}

define void @r4_uncomputable_ancestor(
    ptr noalias %Limit, ptr noalias dereferenceable(128) %A,
    ptr noalias %R) {
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
  %aidx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %av = load i64, ptr %aidx, align 8
  %sum.next.j = add i64 %sum.j, %av
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header
epilogue:
  %sum.next = phi i64 [ %sum.next.j, %inner.header ]
  %diag = getelementptr inbounds [4 x i64], ptr %A, i64 %i, i64 %i
  %diag.value = load i64, ptr %diag, align 8
  %diag.next = add i64 %diag.value, 1
  store i64 %diag.next, ptr %diag, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %root.latch, label %outer.header
root.latch:
  %sum.result = phi i64 [ %sum.next, %outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %limit = load volatile i64, ptr %Limit, align 8
  %k.next = add i64 %k, 1
  %k.continue = icmp ult i64 %k.next, %limit
  br i1 %k.continue, label %root.header, label %exit
exit:
  ret void
}

define void @r5_first_static_then_later_ordinary(
    ptr noalias dereferenceable(128) %A, ptr noalias %B, ptr noalias %R) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %first.outer
first.outer:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %first.outer.latch ]
  %sum.i = phi i64 [ 0, %root.header ], [ %sum.next, %first.outer.latch ]
  br label %first.inner
first.inner:
  %j = phi i64 [ 0, %first.outer ], [ %j.next, %first.inner ]
  %sum.j = phi i64 [ %sum.i, %first.outer ], [ %sum.next.j, %first.inner ]
  %aidx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %av = load i64, ptr %aidx, align 8
  %sum.next.j = add i64 %sum.j, %av
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %first.epilogue, label %first.inner
first.epilogue:
  %sum.next = phi i64 [ %sum.next.j, %first.inner ]
  %diag = getelementptr inbounds [4 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %diag, align 8
  %dn = add i64 %dv, 1
  store i64 %dn, ptr %diag, align 8
  br label %first.outer.latch
first.outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %second.outer, label %first.outer
second.outer:
  %x = phi i64 [ 0, %first.outer.latch ], [ %x.next, %second.outer.latch ]
  br label %second.inner
second.inner:
  %y = phi i64 [ 0, %second.outer ], [ %y.next, %second.inner ]
  %bidx = getelementptr inbounds [4 x i64], ptr %B, i64 %y, i64 %x
  store i64 %x, ptr %bidx, align 8
  %y.next = add i64 %y, 1
  %y.done = icmp eq i64 %y.next, 4
  br i1 %y.done, label %second.outer.latch, label %second.inner
second.outer.latch:
  %x.next = add i64 %x, 1
  %x.done = icmp eq i64 %x.next, 4
  br i1 %x.done, label %root.latch, label %second.outer
root.latch:
  %sum.result = phi i64 [ %sum.next, %second.outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

define void @r6_runtime_then_static(
    ptr noalias dereferenceable(128) %A0,
    ptr noalias dereferenceable(128) %A1, ptr noalias %R, i64 %n) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %runtime.outer
runtime.outer:
  %ri = phi i64 [ 0, %root.header ], [ %ri.next, %runtime.outer.latch ]
  %rsum.i = phi i64 [ 0, %root.header ], [ %rsum.next, %runtime.outer.latch ]
  br label %runtime.inner
runtime.inner:
  %rj = phi i64 [ 0, %runtime.outer ], [ %rj.next, %runtime.inner ]
  %rsum.j = phi i64 [ %rsum.i, %runtime.outer ], [ %rsum.next.j, %runtime.inner ]
  %ridx = getelementptr inbounds [4 x i64], ptr %A0, i64 %rj, i64 %ri
  %rv = load i64, ptr %ridx, align 8
  %rsum.next.j = add i64 %rsum.j, %rv
  %rj.next = add i64 %rj, 1
  %rj.done = icmp eq i64 %rj.next, 4
  br i1 %rj.done, label %runtime.epilogue, label %runtime.inner
runtime.epilogue:
  %rsum.next = phi i64 [ %rsum.next.j, %runtime.inner ]
  %rdiag = getelementptr inbounds [4 x i64], ptr %A0, i64 %ri, i64 %ri
  %rdv = load i64, ptr %rdiag, align 8
  %rdn = add i64 %rdv, 1
  store i64 %rdn, ptr %rdiag, align 8
  br label %runtime.outer.latch
runtime.outer.latch:
  %ri.next = add i64 %ri, 1
  %ri.done = icmp eq i64 %ri.next, %n
  br i1 %ri.done, label %static.outer, label %runtime.outer
static.outer:
  %si = phi i64 [ 0, %runtime.outer.latch ], [ %si.next, %static.outer.latch ]
  %ssum.i = phi i64 [ %rsum.next, %runtime.outer.latch ], [ %ssum.next, %static.outer.latch ]
  br label %static.inner
static.inner:
  %sj = phi i64 [ 0, %static.outer ], [ %sj.next, %static.inner ]
  %ssum.j = phi i64 [ %ssum.i, %static.outer ], [ %ssum.next.j, %static.inner ]
  %sidx = getelementptr inbounds [4 x i64], ptr %A1, i64 %sj, i64 %si
  %sv = load i64, ptr %sidx, align 8
  %ssum.next.j = add i64 %ssum.j, %sv
  %sj.next = add i64 %sj, 1
  %sj.done = icmp eq i64 %sj.next, 4
  br i1 %sj.done, label %static.epilogue, label %static.inner
static.epilogue:
  %ssum.next = phi i64 [ %ssum.next.j, %static.inner ]
  %sdiag = getelementptr inbounds [4 x i64], ptr %A1, i64 %si, i64 %si
  %sdv = load i64, ptr %sdiag, align 8
  %sdn = add i64 %sdv, 2
  store i64 %sdn, ptr %sdiag, align 8
  br label %static.outer.latch
static.outer.latch:
  %si.next = add i64 %si, 1
  %si.done = icmp eq i64 %si.next, 4
  br i1 %si.done, label %root.latch, label %static.outer
root.latch:
  %final.sum = phi i64 [ %ssum.next, %static.outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %final.sum, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

define void @r7_rejected_then_static(
    ptr noalias %S, ptr noalias %Scratch,
    ptr noalias dereferenceable(128) %A, ptr noalias %R) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %reject.outer
reject.outer:
  %ri = phi i64 [ 0, %root.header ], [ %ri.next, %reject.outer.latch ]
  %si = getelementptr inbounds i64, ptr %S, i64 %ri
  br label %reject.inner
reject.inner:
  %rj = phi i64 [ 0, %reject.outer ], [ %rj.next, %reject.inner ]
  %sv = load i64, ptr %si, align 8
  %scratch = getelementptr inbounds [4 x i64], ptr %Scratch, i64 %rj, i64 %ri
  store i64 %sv, ptr %scratch, align 8
  %rj.next = add i64 %rj, 1
  %rj.done = icmp eq i64 %rj.next, 4
  br i1 %rj.done, label %reject.epilogue, label %reject.inner
reject.epilogue:
  %ri.plus1 = add i64 %ri, 1
  %si.plus1 = getelementptr inbounds i64, ptr %S, i64 %ri.plus1
  store i64 %ri, ptr %si.plus1, align 8
  br label %reject.outer.latch
reject.outer.latch:
  %ri.next = add i64 %ri, 1
  %ri.done = icmp eq i64 %ri.next, 4
  br i1 %ri.done, label %static.outer, label %reject.outer
static.outer:
  %i = phi i64 [ 0, %reject.outer.latch ], [ %i.next, %static.outer.latch ]
  %sum.i = phi i64 [ 0, %reject.outer.latch ], [ %sum.next, %static.outer.latch ]
  br label %static.inner
static.inner:
  %j = phi i64 [ 0, %static.outer ], [ %j.next, %static.inner ]
  %sum.j = phi i64 [ %sum.i, %static.outer ], [ %sum.next.j, %static.inner ]
  %idx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %v = load i64, ptr %idx, align 8
  %sum.next.j = add i64 %sum.j, %v
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %static.epilogue, label %static.inner
static.epilogue:
  %sum.next = phi i64 [ %sum.next.j, %static.inner ]
  %diag = getelementptr inbounds [4 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %diag, align 8
  %dn = add i64 %dv, 1
  store i64 %dn, ptr %diag, align 8
  br label %static.outer.latch
static.outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %root.latch, label %static.outer
root.latch:
  %sum.result = phi i64 [ %sum.next, %static.outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

define void @r8_ineligible_then_static(
    ptr noalias %Limit, ptr noalias %B,
    ptr noalias dereferenceable(128) %A, ptr noalias %R) {
entry:
  br label %root.header
root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %ineligible.outer
ineligible.outer:
  %x = phi i64 [ 0, %root.header ], [ %x.next, %ineligible.outer.latch ]
  br label %ineligible.inner
ineligible.inner:
  %y = phi i64 [ 0, %ineligible.outer ], [ %y.next, %ineligible.inner ]
  %bidx = getelementptr inbounds [4 x i64], ptr %B, i64 %y, i64 %x
  store i64 %x, ptr %bidx, align 8
  %y.next = add i64 %y, 1
  %y.done = icmp eq i64 %y.next, 4
  br i1 %y.done, label %ineligible.outer.latch, label %ineligible.inner
ineligible.outer.latch:
  %limit = load volatile i64, ptr %Limit, align 8
  %x.next = add i64 %x, 1
  %x.continue = icmp ult i64 %x.next, %limit
  br i1 %x.continue, label %ineligible.outer, label %static.outer
static.outer:
  %i = phi i64 [ 0, %ineligible.outer.latch ], [ %i.next, %static.outer.latch ]
  %sum.i = phi i64 [ 0, %ineligible.outer.latch ], [ %sum.next, %static.outer.latch ]
  br label %static.inner
static.inner:
  %j = phi i64 [ 0, %static.outer ], [ %j.next, %static.inner ]
  %sum.j = phi i64 [ %sum.i, %static.outer ], [ %sum.next.j, %static.inner ]
  %idx = getelementptr inbounds [4 x i64], ptr %A, i64 %j, i64 %i
  %v = load i64, ptr %idx, align 8
  %sum.next.j = add i64 %sum.j, %v
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %static.epilogue, label %static.inner
static.epilogue:
  %sum.next = phi i64 [ %sum.next.j, %static.inner ]
  %diag = getelementptr inbounds [4 x i64], ptr %A, i64 %i, i64 %i
  %dv = load i64, ptr %diag, align 8
  %dn = add i64 %dv, 1
  store i64 %dn, ptr %diag, align 8
  br label %static.outer.latch
static.outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %root.latch, label %static.outer
root.latch:
  %sum.result = phi i64 [ %sum.next, %static.outer.latch ]
  %rk = getelementptr inbounds i64, ptr %R, i64 %k
  store i64 %sum.result, ptr %rk, align 8
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 2
  br i1 %k.done, label %exit, label %root.header
exit:
  ret void
}

; The entry guard skips the selected pair and both auxiliary siblings.
; The pair retains the shared pure-data kernel. Each sibling runs once and
; writes only its own disjoint auxiliary storage.
define void @top_level_middle_epilogue_order(
    ptr noalias %A, ptr noalias %D, ptr noalias %L,
    i64 %bias, i1 %take, ptr noalias %Before, ptr noalias %After) {
entry:
  br i1 %take, label %before.ph, label %done
before.ph:
  br label %before.header
before.header:
  %before.iv = phi i64 [ 0, %before.ph ], [ %before.next, %before.header ]
  store i64 101, ptr %Before, align 8
  %before.next = add i64 %before.iv, 1
  %before.done = icmp eq i64 %before.next, 1
  br i1 %before.done, label %outer.ph, label %before.header
outer.ph:
  %seed = add i64 %bias, 11
  %limit = add i64 3, 1
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %r.iv.1 = add i64 %i, %seed
  %r.invariant.1 = add i64 %bias, 17
  %r.invariant.2 = mul i64 %r.invariant.1, 3
  %r.iv.2 = add i64 %r.iv.1, %r.invariant.2
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %ap = getelementptr [4 x i64], ptr %A, i64 %j, i64 %i
  %av = load i64, ptr %ap, align 8
  %an = add i64 %av, 1
  store i64 %an, ptr %ap, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header
epilogue:
  %dp = getelementptr i64, ptr %D, i64 %i
  store i64 %r.iv.2, ptr %dp, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, %limit
  br i1 %i.done, label %exit, label %outer.header
exit:
  br label %after.ph
after.ph:
  br label %after.header
after.header:
  %after.iv = phi i64 [ 0, %after.ph ], [ %after.next, %after.header ]
  store i64 202, ptr %After, align 8
  %after.next = add i64 %after.iv, 1
  %after.done = icmp eq i64 %after.next, 1
  br i1 %after.done, label %done, label %after.header
done:
  ret void
}

; One accepted absolute-depth-3 candidate whose retained nest carries exactly
; one row in each matrix, so the ancestor-prefix proof is not vacuous.
define void @r16_depth3_nonempty_fission_row(ptr noalias %A, ptr noalias %D) {
entry:
  br label %l1.header

l1.header:
  %a = phi i64 [ 0, %entry ], [ %a.next, %l1.latch ]
  br label %l2.header

l2.header:
  %i = phi i64 [ 0, %l1.header ], [ %i.next, %l2.latch ]
  br label %l3.header

l3.header:
  %j = phi i64 [ 0, %l2.header ], [ %j.next, %l3.header ]
  %ap = getelementptr inbounds [4 x [4 x double]], ptr %A, i64 %a, i64 %j, i64 %i
  %v = load double, ptr %ap, align 8
  %w = fadd double %v, 1.000000e+00
  store double %w, ptr %ap, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %l3.header

epilogue:
  %dp = getelementptr inbounds double, ptr %D, i64 %i
  store double 1.000000e+00, ptr %dp, align 8
  br label %l2.latch

l2.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %l1.latch, label %l2.header

l1.latch:
  %a.next = add nuw nsw i64 %a, 1
  %a.done = icmp eq i64 %a.next, 4
  br i1 %a.done, label %exit, label %l1.header

exit:
  ret void
}
