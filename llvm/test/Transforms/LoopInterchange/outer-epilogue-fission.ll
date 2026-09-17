; NOTE: Do not autogenerate
;
; Outer-loop epilogues with static bounds: the motivating checksum shape, the
; dependence directions between the nest N and the epilogue E, and the operand
; roles E may consume. Every positive is prepared and applied; the PREP lines
; are the printed plans and the APPLIED lines the transformed IR.
;
; With the option off the pass leaves this module unchanged.
; RUN: opt -S -passes=no-op-loopnest %s -o %t.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %s -o %t.out
; RUN: diff -u %t.noop %t.out
;
; The profitability policy is fixed so that the small synthetic kernels are
; judged by instruction order rather than by the cache model.
; DEFINE: %{policy} = -loop-interchange-profitabilities=instorder,vectorize
; DEFINE: %{prepare} = -loop-interchange-outer-epilogue-fission -loop-interchange-print-prepared-plan
; DEFINE: %{remarks} = -pass-remarks=loop-interchange -pass-remarks-analysis=loop-interchange -pass-remarks-missed=loop-interchange
; DEFINE: %{applied} = \
; DEFINE:   --func=checksum_diagonal_epilogue --func=n_to_e_flow \
; DEFINE:   --func=n_to_e_anti --func=n_to_e_output \
; DEFINE:   --func=multi_block_epilogue --func=nested_diagonal_epilogue \
; DEFINE:   --func=flattened_byte_gep_latch \
; DEFINE:   --func=e_data_from_shared_preheader_pure \
; DEFINE:   --func=e_data_from_inner_preheader_pure \
; DEFINE:   --func=e_data_from_outer_preheader_invariant \
; DEFINE:   --func=e_operand_closure_dag \
; DEFINE:   --func=n_nonconvergent_call_control \
; DEFINE:   --func=n_nonconvergent_outer_header_control \
; DEFINE:   --func=n_convergent_call_outside_outer --func=raw_lt_n_to_e
; DEFINE: %{no_versioning} = \
; DEFINE:   --implicit-check-not='.lver' \
; DEFINE:   --implicit-check-not='!alias.scope' \
; DEFINE:   --implicit-check-not='!noalias' \
; DEFINE:   --implicit-check-not='llvm.loop.interchange.runtime_versioned' \
; DEFINE:   --implicit-check-not='LoopVersioning'
;
; With the option on, every function in %{applied} is transformed and every
; other function is unchanged.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.prep 2> %t.prep.stderr
; RUN: FileCheck %s --check-prefix=PREP --input-file=%t.prep.stderr --implicit-check-not='loop-interchange:'
; RUN: llvm-extract -S --delete %{applied} %t.out -o %t.out.rest
; RUN: llvm-extract -S --delete %{applied} %t.prep -o %t.prep.rest
; RUN: diff -u -I '^; ModuleID' %t.out.rest %t.prep.rest
; RUN: FileCheck %s --check-prefix=APPLIED --input-file=%t.prep %{no_versioning} --implicit-check-not='{{^define }}'
;
; A zero memory-instruction ratio fails the shared N/E budget before any
; dependence matrix is built, so nothing is transformed.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-mem-instr-ratio=0 -loop-interchange-outer-epilogue-fission=false %s -o %t.ratio0.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-mem-instr-ratio=0 %{prepare} %{remarks} %s -o %t.ratio0.on 2> %t.ratio0.stderr
; RUN: diff -u %t.ratio0.off %t.ratio0.on
; RUN: FileCheck %s --check-prefix=MEMZERO --input-file=%t.ratio0.stderr --implicit-check-not='loop-interchange:'
;
; The updated LoopInfo must match a fresh rebuild, sibling order included, and
; a second run of the pass must change nothing.
; RUN: opt -passes='loop(loop-interchange),print<loops>' -cache-line-size=64 %{policy} -loop-interchange-outer-epilogue-fission -disable-output %s 2>&1 | FileCheck %s --check-prefix=APPLY-LOOPS
; RUN: opt -passes='print<loops>' -disable-output %t.prep 2>&1 | FileCheck %s --check-prefix=APPLY-LOOPS
; RUN: opt -S -passes='loop(loop-interchange,loop-interchange)' -cache-line-size=64 %{policy} %{prepare} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.twice
; RUN: diff -u %t.prep %t.twice
;
; A SCEV cache primed before the transform must agree with a fresh computation
; over the transformed function, for a top-level and for a nested candidate.
; RUN: llvm-extract -func=checksum_diagonal_epilogue -S %s -o %t.one.ll
; RUN: opt -disable-output -passes='print<scalar-evolution>' %t.one.ll 2> %t.one.before
; RUN: opt -disable-output -cache-line-size=64 -verify-scev %{policy} -loop-interchange-outer-epilogue-fission -passes='print<scalar-evolution>,loop-interchange,print<scalar-evolution>' %t.one.ll 2> %t.one.primed
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 -verify-scev %{policy} -loop-interchange-outer-epilogue-fission %t.one.ll -o %t.one.transformed
; RUN: opt -disable-output -passes='print<scalar-evolution>' %t.one.transformed 2> %t.one.after
; RUN: cat %t.one.before %t.one.after > %t.one.expected
; RUN: diff -u %t.one.expected %t.one.primed
; RUN: llvm-extract -func=nested_diagonal_epilogue -S %s -o %t.nested.ll
; RUN: opt -disable-output -passes='print<scalar-evolution>' %t.nested.ll 2> %t.nested.before
; RUN: opt -disable-output -cache-line-size=64 -verify-scev %{policy} -loop-interchange-outer-epilogue-fission -passes='print<scalar-evolution>,loop-interchange,print<scalar-evolution>' %t.nested.ll 2> %t.nested.primed
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 -verify-scev %{policy} -loop-interchange-outer-epilogue-fission %t.nested.ll -o %t.nested.transformed
; RUN: opt -disable-output -passes='print<scalar-evolution>' %t.nested.transformed 2> %t.nested.after
; RUN: cat %t.nested.before %t.nested.after > %t.nested.expected
; RUN: diff -u %t.nested.expected %t.nested.primed
;
; Each printed plan has passed both dependence matrices, legality,
; profitability, and validation. The two remarks after it come from the apply
; step.
; PREP-LABEL: loop-interchange: prepared function=checksum_diagonal_epilogue{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=6
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[CHECKSUM_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[CHECKSUM_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=1335.
; PREP-LABEL: loop-interchange: prepared function=n_to_e_flow{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=5 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[FLOW_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[FLOW_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=1335.
; PREP-LABEL: loop-interchange: prepared function=n_to_e_anti{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=1
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; PREP-LABEL: loop-interchange: prepared function=n_to_e_output{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=3 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[OUTPUT_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[OUTPUT_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=1335.
; PREP-LABEL: loop-interchange: prepared function=multi_block_epilogue{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=2 epilogue-insts=8 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=2
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; PREP-LABEL: loop-interchange: prepared function=nested_diagonal_epilogue{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=4 rematerialized=0
; PREP-SAME:  absolute-depth=3 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[NESTED_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[NESTED_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=1335.
; PREP-LABEL: loop-interchange: prepared function=flattened_byte_gep_latch{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=0 epilogue-insts=6 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=2
; PREP-SAME:  byte-offset-proofs=1 requirements=2 bound=static-bound W=1335
; PREP-SAME:  requirement-ids=[[BYTE_ID:[0-9]+]]:modular-outer-span/static/by-constant-max,[[BYTE_ID]]:object-containment/static/by-constant-max{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, W=1335.
;
; The shared and distinct pure operand chains include the two loop-local
; invariant definitions. The freeze outside the outer loop is retained rather
; than rematerialized. Disjoint N/E memory needs no bound requirement.
; PREP-LABEL: loop-interchange: prepared function=e_data_from_shared_preheader_pure{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=4
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=1 routing-rows=1 cross-deps=0 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; PREP-LABEL: loop-interchange: prepared function=e_data_from_inner_preheader_pure{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=4
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=1 routing-rows=1 cross-deps=0 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; PREP-LABEL: loop-interchange: prepared function=e_data_from_outer_preheader_invariant{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=1 routing-rows=1 cross-deps=0 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
;
; The operand chain records each loop-local definition once, in post-order,
; including the shared operand and the duplicated operand: a, b, c, k1, k2, d,
; e. The APPLIED lines below check that order.
; PREP-LABEL: loop-interchange: prepared function=e_operand_closure_dag{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=7
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
;
; The three convergence controls each reach a complete static plan. Two place
; a plain call where the paired negative (in the rejections test) places a
; convergent one; the third keeps its convergent call outside the outer loop.
; PREP-LABEL: loop-interchange: prepared function=n_nonconvergent_call_control{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; PREP-LABEL: loop-interchange: prepared function=n_nonconvergent_outer_header_control{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; PREP-LABEL: loop-interchange: prepared function=n_convergent_call_outside_outer{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
;
; The raw N/E distance is +1, a decisive LT direction, so this plan needs
; neither a byte-offset proof nor a bound requirement.
; PREP-LABEL: loop-interchange: prepared function=raw_lt_n_to_e{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=3 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=1 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
;
; MEMZERO-LABEL: loop-interchange: rejected function=checksum_diagonal_epilogue{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: union N/E memory budget or simple-access check failed
; MEMZERO-NEXT:  remark: <unknown>:0:0: Number of loads/stores exceeded, the supported maximum can be increased with option -loop-interchange-max-mem-instr-ratio.
; MEMZERO-LABEL: loop-interchange: rejected function=n_to_e_flow{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=n_to_e_anti{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=n_to_e_output{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=multi_block_epilogue{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=nested_diagonal_epilogue{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=flattened_byte_gep_latch{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=e_data_from_shared_preheader_pure{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=e_data_from_inner_preheader_pure{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=e_data_from_outer_preheader_invariant{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: union N/E memory budget or simple-access check failed
; MEMZERO-NEXT:  remark: <unknown>:0:0: Number of loads/stores exceeded, the supported maximum can be increased with option -loop-interchange-max-mem-instr-ratio.
; MEMZERO-LABEL: loop-interchange: rejected function=e_operand_closure_dag{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=n_nonconvergent_call_control{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=n_nonconvergent_outer_header_control{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=n_convergent_call_outside_outer{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=raw_lt_n_to_e{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: union N/E memory budget or simple-access check failed
; MEMZERO-NEXT:  remark: <unknown>:0:0: Number of loads/stores exceeded, the supported maximum can be increased with option -loop-interchange-max-mem-instr-ratio.
;
; The transformed module. Every definition is listed, so the RUN line's
; '{{^define }}' exclusion makes the list exhaustive. Each applied function
; gains one sibling epilogue loop whose latch branches to the original
; continuation, and its nest is interchanged. The emptied original epilogue
; block, the epilogue loop, and the continuation are checked instruction by
; instruction. Block labels are plain directives because the printed IR
; separates blocks with a blank line. The other functions were already shown
; unchanged by the llvm-extract --delete pair above and carry only their label.
;
; APPLIED-LABEL: define {{.*}}@checksum_diagonal_epilogue(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %pchk.res = phi double [ %pchk.next, %inner.header.split ]
; APPLIED-NEXT:      %uchk.res = phi double [ %uchk.next, %inner.header.split ]
; APPLIED-NEXT:      %vchk.res = phi double [ %vchk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds [1335 x double], ptr %UNEW, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dval.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dnew.epil = fmul double %dval.epil, 1.500000e+00
; APPLIED-NEXT:      store double %dnew.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header, !prof !0
; APPLIED:         exit.cont:
; APPLIED-NEXT:      %r1 = getelementptr inbounds double, ptr %R, i64 1
; APPLIED-NEXT:      %r2 = getelementptr inbounds double, ptr %R, i64 2
; APPLIED-NEXT:      store double %pchk.res, ptr %R, align 8
; APPLIED-NEXT:      store double %uchk.res, ptr %r1, align 8
; APPLIED-NEXT:      store double %vchk.res, ptr %r2, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@n_to_e_flow(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %seidx.epil = getelementptr inbounds [1335 x double], ptr %S, i64 1334, i64 %epilogue.iv
; APPLIED-NEXT:      %sval.epil = load double, ptr %seidx.epil, align 8
; APPLIED-NEXT:      %bdbl.epil = fadd reassoc double %sval.epil, %sval.epil
; APPLIED-NEXT:      %bi.epil = getelementptr inbounds double, ptr %B, i64 %epilogue.iv
; APPLIED-NEXT:      store double %bdbl.epil, ptr %bi.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@n_to_e_anti(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %si.epil = getelementptr inbounds double, ptr %S, i64 %epilogue.iv
; APPLIED-NEXT:      %ival.epil = uitofp i64 %epilogue.iv to double
; APPLIED-NEXT:      store double %ival.epil, ptr %si.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@n_to_e_output(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %seidx.epil = getelementptr inbounds [1335 x double], ptr %S, i64 1334, i64 %epilogue.iv
; APPLIED-NEXT:      %ival.epil = uitofp i64 %epilogue.iv to double
; APPLIED-NEXT:      store double %ival.epil, ptr %seidx.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@multi_block_epilogue(
; APPLIED:         epilogue.first:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr inbounds double, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      %dv.epil = load double, ptr %dp.epil, align 8
; APPLIED-NEXT:      %dn.epil = fadd double %dv.epil, 1.000000e+00
; APPLIED-NEXT:      store double %dn.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      %ep.epil = getelementptr inbounds double, ptr %E, i64 %epilogue.iv
; APPLIED-NEXT:      %ev.epil = load double, ptr %ep.epil, align 8
; APPLIED-NEXT:      %en.epil = fadd double %ev.epil, 2.000000e+00
; APPLIED-NEXT:      store double %en.epil, ptr %ep.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@nested_diagonal_epilogue(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         outer.exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %ddiag.epil = getelementptr inbounds [1335 x double], ptr %D, i64 %epilogue.iv, i64 %epilogue.iv
; APPLIED-NEXT:      %dv.epil = load double, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      %dn.epil = fadd double %dv.epil, 1.000000e+00
; APPLIED-NEXT:      store double %dn.epil, ptr %ddiag.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %i.next.epil, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %outer.exit.cont, label %epilogue.header
; APPLIED:         outer.exit.cont:
; APPLIED-NEXT:      %rk = getelementptr inbounds double, ptr %R, i64 %k
; APPLIED-NEXT:      store double %chk.res, ptr %rk, align 8
; APPLIED-NEXT:      br label %sibling.header
; APPLIED-LABEL: define {{.*}}@flattened_byte_gep_latch(
; APPLIED:         outer.latch:
; APPLIED-NEXT:      %i.next = add i64 %i, 1
; APPLIED-NEXT:      %i.ec = icmp eq i64 %i, 1335
; APPLIED-NEXT:      br i1 %i.ec, label %inner.header.split, label %outer.header
; APPLIED:         exit:
; APPLIED-NEXT:      %chk.res = phi double [ %chk.next, %inner.header.split ]
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 1, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %e.byte.epil = mul i64 %epilogue.iv, 10688
; APPLIED-NEXT:      %e.base.epil = getelementptr i8, ptr getelementptr inbounds (i8, ptr @byte_common, i64 64), i64 %e.byte.epil
; APPLIED-NEXT:      %e.ptr.epil = getelementptr i8, ptr %e.base.epil, i64 -10688
; APPLIED-NEXT:      %e.val.epil = load double, ptr %e.ptr.epil, align 8
; APPLIED-NEXT:      %e.new.epil = fmul double %e.val.epil, 1.500000e+00
; APPLIED-NEXT:      store double %e.new.epil, ptr %e.ptr.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.ec.epil = icmp eq i64 %epilogue.iv, 1335
; APPLIED-NEXT:      br i1 %i.ec.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      store double %chk.res, ptr %R, align 8
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@e_data_from_shared_preheader_pure(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %r.iv.1.epil = add i64 %epilogue.iv, %seed
; APPLIED-NEXT:      %r.invariant.1.epil = add i64 %bias, 17
; APPLIED-NEXT:      %r.invariant.2.epil = mul i64 %r.invariant.1.epil, 3
; APPLIED-NEXT:      %r.iv.2.epil = add i64 %r.iv.1.epil, %r.invariant.2.epil
; APPLIED-NEXT:      %dp.epil = getelementptr i64, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      store i64 %r.iv.2.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, %limit
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      br label %done
; APPLIED-LABEL: define {{.*}}@e_data_from_inner_preheader_pure(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %r.iv.1.epil = add i64 %epilogue.iv, %seed
; APPLIED-NEXT:      %r.invariant.1.epil = add i64 %bias, 17
; APPLIED-NEXT:      %r.invariant.2.epil = mul i64 %r.invariant.1.epil, 3
; APPLIED-NEXT:      %r.iv.2.epil = add i64 %r.iv.1.epil, %r.invariant.2.epil
; APPLIED-NEXT:      %dp.epil = getelementptr i64, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      store i64 %r.iv.2.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, %limit
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      br label %done
; APPLIED-LABEL: define {{.*}}@e_data_from_outer_preheader_invariant(
; APPLIED:         inner.ph:
; APPLIED-NEXT:      %seed = freeze i64 %bias
; APPLIED-NEXT:      %limit = add i64 3, 1
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr i64, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      store i64 %seed, ptr %dp.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, %limit
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      br label %done
; APPLIED-LABEL: define {{.*}}@e_operand_closure_dag(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %a.epil = add i64 %epilogue.iv, %seed
; APPLIED-NEXT:      %b.epil = mul i64 %a.epil, 3
; APPLIED-NEXT:      %c.epil = add i64 %b.epil, %b.epil
; APPLIED-NEXT:      %k1.epil = add i64 %bias, 17
; APPLIED-NEXT:      %k2.epil = mul i64 %k1.epil, %k1.epil
; APPLIED-NEXT:      %d.epil = add i64 %c.epil, %k2.epil
; APPLIED-NEXT:      %e.epil = sub i64 %d.epil, %a.epil
; APPLIED-NEXT:      %dp.epil = getelementptr i64, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      store i64 %e.epil, ptr %dp.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add nuw nsw i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@n_nonconvergent_call_control(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr i32, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      store i32 7, ptr %dp.epil, align 4
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add nuw nsw i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@n_nonconvergent_outer_header_control(
; APPLIED:         outer.header:
; APPLIED:           %v = call i32 @fission_plain_seed(i64 %i)
; APPLIED-NOT:       call i32 @fission_plain_seed
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr i32, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      store i32 7, ptr %dp.epil, align 4
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add nuw nsw i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@n_convergent_call_outside_outer(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr i32, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      store i32 7, ptr %dp.epil, align 4
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add nuw nsw i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@raw_lt_n_to_e(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 1, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %previous.epil = sub nuw nsw i64 %epilogue.iv, 1
; APPLIED-NEXT:      %ep.epil = getelementptr inbounds i64, ptr %A, i64 %previous.epil
; APPLIED-NEXT:      store i64 0, ptr %ep.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add nuw nsw i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, 5
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      ret void
; The same loop-info lines are checked against the loop manager's live
; LoopInfo and against a fresh rebuild of the transformed module. The block
; lists inside the nest are wildcarded because LoopInfo stores them in
; discovery order, which an incremental update and a fresh walk reach
; differently; the loop order, the depths, and the epilogue loop's two blocks
; are exact.
;
; APPLY-LOOPS-LABEL: Loop info for function 'checksum_diagonal_epilogue':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'n_to_e_flow':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'n_to_e_anti':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'n_to_e_output':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'multi_block_epilogue':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'nested_diagonal_epilogue':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %root.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:         Loop at depth 3 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %sibling.header<header><latch><exiting>
; APPLY-LOOPS-LABEL: Loop info for function 'flattened_byte_gep_latch':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'e_data_from_shared_preheader_pure':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'e_data_from_inner_preheader_pure':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'e_data_from_outer_preheader_invariant':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'e_operand_closure_dag':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'n_nonconvergent_call_control':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'n_nonconvergent_outer_header_control':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'n_convergent_call_outside_outer':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'raw_lt_n_to_e':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
;

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@byte_common = internal global [14268552 x i8] zeroinitializer, align 8

; The motivating shape, in Fortran (column-major, so the first subscript is
; the unit-stride one; every array has leading dimension 1335):
;
;   do i = 1, 1335
;     do j = 1, 1335
;       pchk = pchk + p(i, j)
;       uchk = uchk + u(i, j)
;       vchk = vchk + v(i, j)
;     end do
;     u(i, i) = u(i, i) * 1.5d0
;   end do
;
; The three reductions hold the first subscript fixed and step the second, a
; 10,680-byte stride between consecutive j. The update of u(i, i) after the
; inner loop makes the nest imperfect. Distributing that update into its own
; loop and interchanging the remaining nest makes i the inner subscript, so
; the reads become unit stride.
; The trip count equals the leading dimension, so the bound is static here.
define void @checksum_diagonal_epilogue(
    ptr noalias %PNEW, ptr noalias dereferenceable(14257800) %UNEW,
    ptr noalias %VNEW, ptr noalias %R) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %pchk.i = phi double [ 0.000000e+00, %entry ], [ %pchk.next, %outer.latch ]
  %uchk.i = phi double [ 0.000000e+00, %entry ], [ %uchk.next, %outer.latch ]
  %vchk.i = phi double [ 0.000000e+00, %entry ], [ %vchk.next, %outer.latch ]
  %pcol = getelementptr inbounds [1335 x double], ptr %PNEW, i64 0, i64 %i
  %ucol = getelementptr inbounds [1335 x double], ptr %UNEW, i64 0, i64 %i
  %vcol = getelementptr inbounds [1335 x double], ptr %VNEW, i64 0, i64 %i
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %pchk.j = phi double [ %pchk.i, %outer.header ], [ %pchk.next.j, %inner.header ]
  %uchk.j = phi double [ %uchk.i, %outer.header ], [ %uchk.next.j, %inner.header ]
  %vchk.j = phi double [ %vchk.i, %outer.header ], [ %vchk.next.j, %inner.header ]
  %pidx = getelementptr inbounds [1335 x double], ptr %pcol, i64 %j, i64 0
  %pv = load double, ptr %pidx, align 8
  %pchk.next.j = fadd reassoc double %pchk.j, %pv
  %uidx = getelementptr inbounds [1335 x double], ptr %ucol, i64 %j, i64 0
  %uv = load double, ptr %uidx, align 8
  %uchk.next.j = fadd reassoc double %uchk.j, %uv
  %vidx = getelementptr inbounds [1335 x double], ptr %vcol, i64 %j, i64 0
  %vv = load double, ptr %vidx, align 8
  %vchk.next.j = fadd reassoc double %vchk.j, %vv
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header

epilogue:
  %pchk.next = phi double [ %pchk.next.j, %inner.header ]
  %uchk.next = phi double [ %uchk.next.j, %inner.header ]
  %vchk.next = phi double [ %vchk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %UNEW, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %exit, label %outer.header, !prof !0

exit:
  %pchk.res = phi double [ %pchk.next, %outer.latch ]
  %uchk.res = phi double [ %uchk.next, %outer.latch ]
  %vchk.res = phi double [ %vchk.next, %outer.latch ]
  %r1 = getelementptr inbounds double, ptr %R, i64 1
  %r2 = getelementptr inbounds double, ptr %R, i64 2
  store double %pchk.res, ptr %R, align 8
  store double %uchk.res, ptr %r1, align 8
  store double %vchk.res, ptr %r2, align 8
  ret void
}

define void @n_to_e_flow(
    ptr noalias %A, ptr noalias dereferenceable(14257800) %S,
    ptr noalias %B, ptr noalias %R) {
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
  %chk.next.j = fadd reassoc double %chk.j, %av
  %sidx = getelementptr inbounds [1335 x double], ptr %S, i64 %j, i64 %i
  store double %av, ptr %sidx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header

epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %seidx = getelementptr inbounds [1335 x double], ptr %S, i64 1334, i64 %i
  %sval = load double, ptr %seidx, align 8
  %bdbl = fadd reassoc double %sval, %sval
  %bi = getelementptr inbounds double, ptr %B, i64 %i
  store double %bdbl, ptr %bi, align 8
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

define void @n_to_e_anti(ptr noalias %A, ptr noalias %S, ptr noalias %B,
                         ptr noalias %R) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %si = getelementptr inbounds double, ptr %S, i64 %i
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %aidx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %av
  %sread = load double, ptr %si, align 8
  %bidx = getelementptr inbounds [1335 x double], ptr %B, i64 %j, i64 %i
  store double %sread, ptr %bidx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header

epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ival = uitofp i64 %i to double
  store double %ival, ptr %si, align 8
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

define void @n_to_e_output(
    ptr noalias %A, ptr noalias dereferenceable(14257800) %S,
    ptr noalias %R) {
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
  %chk.next.j = fadd reassoc double %chk.j, %av
  %sidx = getelementptr inbounds [1335 x double], ptr %S, i64 %j, i64 %i
  store double %av, ptr %sidx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header

epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %seidx = getelementptr inbounds [1335 x double], ptr %S, i64 1334, i64 %i
  %ival = uitofp i64 %i to double
  store double %ival, ptr %seidx, align 8
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

define void @multi_block_epilogue(ptr noalias %A, ptr noalias %D,
                                  ptr noalias %E, ptr noalias %R) {
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
  %chk.next.j = fadd reassoc double %chk.j, %av
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue.first, label %inner.header

epilogue.first:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %dp = getelementptr inbounds double, ptr %D, i64 %i
  %dv = load double, ptr %dp, align 8
  %dn = fadd double %dv, 1.000000e+00
  store double %dn, ptr %dp, align 8
  br label %epilogue.second

epilogue.second:
  %ep = getelementptr inbounds double, ptr %E, i64 %i
  %ev = load double, ptr %ep, align 8
  %en = fadd double %ev, 2.000000e+00
  store double %en, ptr %ep, align 8
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

define void @nested_diagonal_epilogue(
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

define void @flattened_byte_gep_latch(ptr noalias %R) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 1, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 1, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %n.row = mul i64 %j, 1335
  %n.column.bias = add i64 %i, -1336
  %n.element = add i64 %n.row, %n.column.bias
  %n.ptr = getelementptr [8 x i8], ptr getelementptr inbounds (i8, ptr @byte_common, i64 64), i64 %n.element
  %n.val = load double, ptr %n.ptr, align 8
  %chk.next.j = fadd reassoc double %chk.j, %n.val
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j, 1335
  br i1 %j.ec, label %outer.latch, label %inner.header

outer.latch:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %e.byte = mul i64 %i, 10688
  %e.base = getelementptr i8, ptr getelementptr inbounds (i8, ptr @byte_common, i64 64), i64 %e.byte
  %e.ptr = getelementptr i8, ptr %e.base, i64 -10688
  %e.val = load double, ptr %e.ptr, align 8
  %e.new = fmul double %e.val, 1.500000e+00
  store double %e.new, ptr %e.ptr, align 8
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i, 1335
  br i1 %i.ec, label %exit, label %outer.header

exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @e_data_from_shared_preheader_pure(ptr noalias %A, ptr noalias %D,
                            ptr noalias %L, i64 %bias, i1 %take) {
entry:
  br i1 %take, label %outer.ph, label %done
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
  br label %done
done:
  ret void
}

define void @e_data_from_inner_preheader_pure(ptr noalias %A, ptr noalias %D,
                            ptr noalias %L, i64 %bias, i1 %take) {
entry:
  br i1 %take, label %outer.ph, label %done
outer.ph:
  %seed = add i64 %bias, 11
  %limit = add i64 3, 1
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  br label %inner.ph

inner.ph:
  %r.iv.1 = add i64 %i, %seed
  %r.invariant.1 = add i64 %bias, 17
  %r.invariant.2 = mul i64 %r.invariant.1, 3
  %r.iv.2 = add i64 %r.iv.1, %r.invariant.2
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %inner.ph ], [ %j.next, %inner.header ]
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
  br label %done
done:
  ret void
}

; The retained operand: the invariant freeze stays in the block that becomes
; the new outer preheader and E reads it directly rather than through a clone.
; The epilogue's trip limit is the same retained value as the nest's.
define void @e_data_from_outer_preheader_invariant(ptr noalias %A, ptr noalias %D,
                                      ptr noalias %L, i64 %bias, i1 %take) {
entry:
  br i1 %take, label %outer.ph, label %done
outer.ph:
  %seed = freeze i64 %bias
  %limit = add i64 3, 1
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  br label %inner.ph
inner.ph:
  %column = getelementptr i64, ptr %A, i64 %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %inner.ph ], [ %j.next, %inner.header ]
  %ap = getelementptr [4 x i64], ptr %column, i64 %j
  %av = load i64, ptr %ap, align 8
  %an = add i64 %av, 1
  store i64 %an, ptr %ap, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header
epilogue:
  %dp = getelementptr i64, ptr %D, i64 %i
  store i64 %seed, ptr %dp, align 8
  br label %outer.latch
outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, %limit
  br i1 %i.done, label %exit, label %outer.header
exit:
  br label %done
done:
  ret void
}


; E's stored value roots an operand chain with a shared operand, a duplicated
; operand, and two loop-invariant leaves.
define void @e_operand_closure_dag(
    ptr noalias %A, ptr noalias %D, ptr noalias %K, i64 %bias) {
entry:
  br label %outer.ph

outer.ph:
  %seed = freeze i64 %bias
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  %a = add i64 %i, %seed
  %b = mul i64 %a, 3
  %c = add i64 %b, %b
  %k1 = add i64 %bias, 17
  %k2 = mul i64 %k1, %k1
  %d = add i64 %c, %k2
  %e = sub i64 %d, %a
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %ap = getelementptr [4 x i64], ptr %A, i64 %j, i64 %i
  store i64 %j, ptr %ap, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:
  %dp = getelementptr i64, ptr %D, i64 %i
  store i64 %e, ptr %dp, align 8
  br label %outer.latch

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}

; The non-convergent twin of n_convergent_controlled_in_nest (in the
; rejections test): the same shape and call position with a plain callee.
define void @n_nonconvergent_call_control(ptr noalias %A, ptr noalias %D) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %v = call i32 @fission_plain_seed(i64 %j)
  %ap = getelementptr [4 x i32], ptr %A, i64 %j, i64 %i
  store i32 %v, ptr %ap, align 4
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header
epilogue:
  %dp = getelementptr i32, ptr %D, i64 %i
  store i32 7, ptr %dp, align 4
  br label %outer.latch
outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %exit, label %outer.header
exit:
  ret void
}

; The non-convergent twin of n_convergent_uncontrolled_in_outer_header. Its
; distinct inner preheader keeps the outer header out of that role.
define void @n_nonconvergent_outer_header_control(
    ptr noalias %A, ptr noalias %D) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %v = call i32 @fission_plain_seed(i64 %i)
  br label %inner.ph

inner.ph:
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %inner.ph ], [ %j.next, %inner.header ]
  %ap = getelementptr [4 x i32], ptr %A, i64 %j, i64 %i
  store i32 %v, ptr %ap, align 4
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:
  %dp = getelementptr i32, ptr %D, i64 %i
  store i32 7, ptr %dp, align 4
  br label %outer.latch

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}

; The only convergent call sits in the outer preheader, outside Outer, so the
; retained nest holds no convergent operation.
define void @n_convergent_call_outside_outer(
    ptr noalias %A, ptr noalias %D, i64 %bias) {
entry:
  br label %outer.ph

outer.ph:
  %v = call i32 @fission_convergent_seed(i64 %bias)
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %outer.ph ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %ap = getelementptr [4 x i32], ptr %A, i64 %j, i64 %i
  store i32 %v, ptr %ap, align 4
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:
  %dp = getelementptr i32, ptr %D, i64 %i
  store i32 7, ptr %dp, align 4
  br label %outer.latch

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}

; The raw less-than control for the selected-outer direction negatives in
; outer-epilogue-fission-rejections.ll.
define void @raw_lt_n_to_e(ptr noalias %A, ptr noalias %B) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 1, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %np = getelementptr inbounds i64, ptr %A, i64 %i
  %nv = load i64, ptr %np, align 8
  %bp = getelementptr inbounds [8 x i64], ptr %B, i64 %j, i64 %i
  store i64 %nv, ptr %bp, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:
  %previous = sub nuw nsw i64 %i, 1
  %ep = getelementptr inbounds i64, ptr %A, i64 %previous
  store i64 0, ptr %ep, align 8
  br label %outer.latch

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 5
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}

; Callees for the convergence controls. A call result never reaches E.
declare i32 @fission_plain_seed(i64) memory(none) nounwind willreturn
declare i32 @fission_convergent_seed(i64) convergent memory(none) nounwind willreturn

!0 = !{!"branch_weights", i32 1, i32 100}
