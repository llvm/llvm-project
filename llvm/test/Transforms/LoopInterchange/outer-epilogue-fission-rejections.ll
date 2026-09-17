; NOTE: Do not autogenerate
;
; Rejections: dependence, operand-chain, legality, and structure negatives,
; including unsupported epilogue data and control roles. N denotes the
; retained nest and E the post-inner epilogue. Two positive controls separate
; nest-to-nest aliasing and the matrix memory budget from the negatives.
;
; With the option off the pass leaves this module unchanged.
; RUN: opt -S -passes=no-op-loopnest %s -o %t.noop
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %s -o %t.out
; RUN: diff -u %t.noop %t.out
;
; The profitability policy is fixed so that the small synthetic kernels are
; judged by instruction order rather than by the cache model. The positive
; N/E, closed-slice, reassociated-FP, and pure-data controls live in
; outer-epilogue-fission.ll.
; DEFINE: %{policy} = -loop-interchange-profitabilities=instorder,vectorize
; DEFINE: %{prepare} = -loop-interchange-outer-epilogue-fission -loop-interchange-print-prepared-plan
; DEFINE: %{remarks} = -pass-remarks=loop-interchange -pass-remarks-analysis=loop-interchange -pass-remarks-missed=loop-interchange
; DEFINE: %{applied} = \
; DEFINE:   --func=noalias_in_nest --func=matrix_memory_ratio
; DEFINE: %{no_versioning} = \
; DEFINE:   --implicit-check-not='.lver' \
; DEFINE:   --implicit-check-not='!alias.scope' \
; DEFINE:   --implicit-check-not='!noalias' \
; DEFINE:   --implicit-check-not='llvm.loop.interchange.runtime_versioned' \
; DEFINE:   --implicit-check-not='LoopVersioning'
;
; Preparation rejects the four convergence negatives right after discovery
; and the memory-budget check, so no later proof stage may report them.
; DEFINE: %{convergence_owner} = \
; DEFINE:   --implicit-check-not='cross-partition' \
; DEFINE:   --implicit-check-not='fission-context' \
; DEFINE:   --implicit-check-not='routing dependence' \
; DEFINE:   --implicit-check-not='interchange legality'
;
; The three selected-outer negatives stop at the N-to-E proof, before either
; speculative matrix is built.
; DEFINE: %{selected_owner} = \
; DEFINE:   --implicit-check-not='fission-context' \
; DEFINE:   --implicit-check-not='routing dependence'
;
; With the option on, the two positive controls are transformed and every
; other function is unchanged.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.prep 2> %t.prep.stderr
; RUN: FileCheck %s --check-prefixes=PREP,DISCOVERY,MATRIX-READY --input-file=%t.prep.stderr --implicit-check-not='loop-interchange:'
; RUN: llvm-extract -S --delete %{applied} %t.out -o %t.out.rest
; RUN: llvm-extract -S --delete %{applied} %t.prep -o %t.prep.rest
; RUN: diff -u -I '^; ModuleID' %t.out.rest %t.prep.rest
; RUN: FileCheck %s --check-prefix=APPLIED --input-file=%t.prep %{no_versioning} --implicit-check-not='{{^define }}'
;
; The updated LoopInfo must match a fresh rebuild, sibling order included, and
; a second run of the pass must change nothing.
; RUN: opt -passes='loop(loop-interchange),print<loops>' -cache-line-size=64 %{policy} -loop-interchange-outer-epilogue-fission -disable-output %s 2>&1 | FileCheck %s --check-prefix=APPLY-LOOPS
; RUN: opt -passes='print<loops>' -disable-output %t.prep 2>&1 | FileCheck %s --check-prefix=APPLY-LOOPS
; RUN: opt -S -passes='loop(loop-interchange,loop-interchange)' -cache-line-size=64 %{policy} %{prepare} -verify-each -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa %s -o %t.twice
; RUN: diff -u %t.prep %t.twice
;
; At ratio zero, discovery still reports its own failures; only candidates
; that finish discovery reach the shared budget rejection. At ratio one the
; later dependence and legality paths run again, but matrix_memory_ratio now
; fails the matrix memory limit computed after the epilogue is excluded.
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-mem-instr-ratio=0 -loop-interchange-outer-epilogue-fission=false %s -o %t.ratio0.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-mem-instr-ratio=0 %{prepare} %{remarks} %s -o %t.ratio0.on 2> %t.ratio0.stderr
; RUN: diff -u %t.ratio0.off %t.ratio0.on
; RUN: FileCheck %s --check-prefixes=MEMZERO,DISCOVERY --input-file=%t.ratio0.stderr --implicit-check-not='loop-interchange:'
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-mem-instr-ratio=1 -loop-interchange-outer-epilogue-fission=false %s -o %t.ratio1.off
; RUN: opt -S -passes=loop-interchange -cache-line-size=64 %{policy} -loop-interchange-max-mem-instr-ratio=1 %{prepare} %{remarks} %s -o %t.ratio1.on 2> %t.ratio1.stderr
; RUN: llvm-extract -S --delete --func=noalias_in_nest %t.ratio1.off -o %t.ratio1.off.rest
; RUN: llvm-extract -S --delete --func=noalias_in_nest %t.ratio1.on -o %t.ratio1.on.rest
; RUN: diff -u -I '^; ModuleID' %t.ratio1.off.rest %t.ratio1.on.rest
; RUN: FileCheck %s --check-prefix=RATIO-LIMITED --input-file=%t.ratio1.on %{no_versioning}
; RUN: FileCheck %s --check-prefixes=PREP,DISCOVERY,MATRIX-LIMIT --input-file=%t.ratio1.stderr --implicit-check-not='loop-interchange:'
;
; These whole-module traces reach the two matrix guards. The same N/N alias
; rejection and noalias control precede the ratio-sensitive function in both
; traces, and ordinary routing starts only after the feature has decided.
; RUN: %if asserts %{ opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -loop-interchange-max-mem-instr-ratio=1 -debug-only=loop-interchange -verify-each %s -o %t.matrix1.debug 2> %t.matrix1.debug.stderr %}
; RUN: %if asserts %{ FileCheck %s --check-prefixes=MATRIX-TRACE,MATRIX-TRACE-LIMIT --input-file=%t.matrix1.debug.stderr %}
; RUN: %if asserts %{ diff -u %t.ratio1.on %t.matrix1.debug %}
; RUN: %if asserts %{ opt -S -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -verify-each %s -o %t.matrix4.debug 2> %t.matrix4.debug.stderr %}
; RUN: %if asserts %{ FileCheck %s --check-prefixes=MATRIX-TRACE,MATRIX-TRACE-READY --input-file=%t.matrix4.debug.stderr %}
; RUN: %if asserts %{ diff -u %t.prep %t.matrix4.debug %}
;
; The raw dependence directions below are those between N and E, before the
; ordinary path normalizes them. LE and NE pass the raw N/E check and both
; matrices, then fail tight nesting. GE fails the raw direction check before
; either matrix is built.
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=direction_le %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=LE-PREP --implicit-check-not='loop-interchange: prepared function=' %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=direction_ge %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=GE-PREP --implicit-check-not='loop-interchange: prepared function=' %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=direction_ne %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=NE-PREP --implicit-check-not='loop-interchange: prepared function=' %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=n_convergent_controlled_in_nest %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=CONV-NEST %{convergence_owner} %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=n_convergent_uncontrolled_in_outer_header %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=CONV-HEADER %{convergence_owner} %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=n_convergent_uncontrolled_in_inner_body %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=CONV-BODY %{convergence_owner} %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=n_convergent_uncontrolled_in_inner_ph %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=CONV-PH %{convergence_owner} %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=direction_selected_le %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=SEL-LE %{selected_owner} %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=direction_selected_ge %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=SEL-GE %{selected_owner} %}
; RUN: %if asserts %{ llvm-extract -S --recursive --keep-const-init --func=direction_selected_ne %s -o - | opt -passes=loop-interchange -cache-line-size=64 %{policy} %{prepare} %{remarks} -debug-only=loop-interchange -disable-output 2>&1 | FileCheck %s --check-prefix=SEL-NE %{selected_owner} %}
;
; Each negative reports its first rejection; the implicit exclusion rejects
; any other feature line.
; PREP-LABEL: loop-interchange: rejected function=e_to_n_flow{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=e_to_n_flow{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: union N/E memory budget or simple-access check failed
; MEMZERO-NEXT:  remark: <unknown>:0:0: Number of loads/stores exceeded, the supported maximum can be increased with option -loop-interchange-max-mem-instr-ratio.
; PREP-LABEL: loop-interchange: rejected function=e_to_n_anti{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=e_to_n_anti{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; PREP-LABEL: loop-interchange: rejected function=e_to_n_output{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=e_to_n_output{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=e_consumes_reduction{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue consumes an inner/reduction or unsupported external SSA value{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=e_consumes_inner_iv{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue consumes an inner/reduction or unsupported external SSA value{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=e_escapes{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue-defined SSA value escapes the closed slice{{$}}
; PREP-LABEL: loop-interchange: rejected function=exact_recurrence{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=exact_recurrence{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; PREP-LABEL: loop-interchange: rejected function=partly_exact_recurrence{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=partly_exact_recurrence{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=call_in_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue contains an unsupported operation or metadata{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=convergent_call_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue contains an unsupported operation or metadata{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=deopt_call_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue contains an unsupported operation or metadata{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=constrained_fp_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue contains an unsupported operation or metadata{{$}}
; PREP-LABEL: loop-interchange: rejected function=unknown_alias{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=unknown_alias{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; PREP-LABEL: loop-interchange: rejected function=non_affine_address{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=non_affine_address{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=volatile_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue contains an unsupported operation or metadata{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=atomic_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue contains an unsupported operation or metadata{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=fence_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue contains an unsupported operation or metadata{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=conditional_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=exit-to-latch path contains conditional or non-straight-line control{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=multiple_epilogues{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=exit-to-latch path contains conditional or non-straight-line control{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=address_taken_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=post-inner path block cannot be collapsed after extraction{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=triangular_bounds{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=child range is not rectangular with respect to the outer loop{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=data_dependent_bounds{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=child range is not rectangular with respect to the outer loop{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=loop_local_outer_bound{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=outer latch control depends on an unsupported loop-local value{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=noncanonical_latch{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=latch/exit control is not the supported canonical form{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=unsupported_metadata{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=loop carries metadata without a fission policy{{$}}
; PREP-LABEL: loop-interchange: rejected function=outer_header_freeze{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=outer_header_freeze{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; PREP-LABEL: loop-interchange: rejected function=invariant_address_n_to_e_flow{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=invariant_address_n_to_e_flow{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; PREP-LABEL: loop-interchange: rejected function=invariant_address_n_to_e_output{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=virtual extracted nest failed existing interchange legality{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=invariant_address_n_to_e_output{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; PREP-LABEL: loop-interchange: rejected function=direction_le{{ }}
; PREP-SAME:  outer=le.outer inner=le.inner reason=virtual extracted nest failed existing interchange legality{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=direction_le{{ }}
; MEMZERO-SAME:  outer=le.outer inner=le.inner reason=union N/E memory budget or simple-access check failed{{$}}
; PREP-LABEL: loop-interchange: rejected function=direction_ge{{ }}
; PREP-SAME:  outer=ge.outer inner=ge.inner reason=cross-partition dependence is not statically proved N-to-E{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=direction_ge{{ }}
; MEMZERO-SAME:  outer=ge.outer inner=ge.inner reason=union N/E memory budget or simple-access check failed{{$}}
; PREP-LABEL: loop-interchange: rejected function=direction_ne{{ }}
; PREP-SAME:  outer=ne.outer inner=ne.inner reason=virtual extracted nest failed existing interchange legality{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=direction_ne{{ }}
; MEMZERO-SAME:  outer=ne.outer inner=ne.inner reason=union N/E memory budget or simple-access check failed{{$}}
;
; These four roles and e_consumes_inner_iv above fail in discovery. Even
; loop-invariant loop-local latch control is outside the supported control
; partition. Their pure-data twins in outer-epilogue-fission.ll are positive.
; DISCOVERY-LABEL: loop-interchange: rejected function=e_data_from_shared_preheader_freeze{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue consumes an inner/reduction or unsupported external SSA value{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=e_data_from_inner_preheader_load{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue consumes an inner/reduction or unsupported external SSA value{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=e_control_from_shared_preheader{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=outer latch control depends on an unsupported loop-local value{{$}}
; DISCOVERY-LABEL: loop-interchange: rejected function=e_control_from_inner_preheader{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=outer latch control depends on an unsupported loop-local value{{$}}
; PREP-NEXT:     remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; MEMZERO-NEXT:  remark: <unknown>:0:0: Number of loads/stores exceeded, the supported maximum can be increased with option -loop-interchange-max-mem-instr-ratio.
; The invalid leaf records a, b, c and then fails at k1, so invalidity
; propagates after partial recording. Discovery rejects it, and no run prints
; a prepared line for it.
; DISCOVERY-LABEL: loop-interchange: rejected function=e_operand_closure_invalid_leaf{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=epilogue consumes an inner/reduction or unsupported external SSA value{{$}}
;
; Four negatives place convergent calls in the scanned regions. Preparation
; rejects them after discovery and the memory-budget check, before any later
; proof stage. The controls in outer-epilogue-fission.ll use plain callees.
; PREP-LABEL: loop-interchange: rejected function=n_convergent_controlled_in_nest{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=retained nest contains an unsupported convergent operation{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: retained nest contains an unsupported convergent operation
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; PREP-LABEL: loop-interchange: rejected function=n_convergent_uncontrolled_in_outer_header{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=retained nest contains an unsupported convergent operation{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: retained nest contains an unsupported convergent operation
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; PREP-LABEL: loop-interchange: rejected function=n_convergent_uncontrolled_in_inner_body{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=retained nest contains an unsupported convergent operation{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: retained nest contains an unsupported convergent operation
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; PREP-LABEL: loop-interchange: rejected function=n_convergent_uncontrolled_in_inner_ph{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=retained nest contains an unsupported convergent operation{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: retained nest contains an unsupported convergent operation
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
;
; The three selected-outer directions own the generalized raw classes at the
; selected outer itself. Their raw vectors [<=|<], [=>|<], and [<>] are bound
; by the function-scoped DA rows above and stay unnormalized there.
; PREP-LABEL: loop-interchange: rejected function=direction_selected_le{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
; PREP-LABEL: loop-interchange: rejected function=direction_selected_ge{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
; PREP-LABEL: loop-interchange: rejected function=direction_selected_ne{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; PREP-NEXT:  remark: <unknown>:0:0: Cannot interchange loops due to dependences.
;
; The budget check precedes the convergence scan, so at ratio zero a
; convergence negative reports the budget failure instead.
; MEMZERO-LABEL: loop-interchange: rejected function=n_convergent_controlled_in_nest{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=n_convergent_uncontrolled_in_outer_header{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=n_convergent_uncontrolled_in_inner_body{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=n_convergent_uncontrolled_in_inner_ph{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=direction_selected_le{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=direction_selected_ge{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-LABEL: loop-interchange: rejected function=direction_selected_ne{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
;
; Both N/E pairs are independent, but the uncertain N/N pair fails the first
; matrix. Adding noalias to A alone gives a complete plan with no requirement.
; PREP-LABEL: loop-interchange: rejected function=unknown_alias_in_nest{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header reason=fission-context dependence matrix is unavailable{{$}}
; PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: fission-context dependence matrix is unavailable
; PREP-NEXT:  remark: <unknown>:0:0: All loops have dependencies in all directions.
; MEMZERO-LABEL: loop-interchange: rejected function=unknown_alias_in_nest{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: union N/E memory budget or simple-access check failed
; MEMZERO-NEXT:  remark: <unknown>:0:0: Number of loads/stores exceeded, the supported maximum can be increased with option -loop-interchange-max-mem-instr-ratio.
; PREP-LABEL: loop-interchange: prepared function=noalias_in_nest{{ }}
; PREP-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=2 rematerialized=0
; PREP-SAME:  absolute-depth=2 routing-depth=2 fission-rows=0 routing-rows=0 cross-deps=0 reductions=0
; PREP-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; PREP-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; PREP-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; MEMZERO-LABEL: loop-interchange: rejected function=noalias_in_nest{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: union N/E memory budget or simple-access check failed
; MEMZERO-NEXT:  remark: <unknown>:0:0: Number of loads/stores exceeded, the supported maximum can be increased with option -loop-interchange-max-mem-instr-ratio.
;
; N and E together have 9 memory instructions among 92. Excluding the
; 66-instruction E slice leaves 8 among 26, which ratio 1 rejects (8 > 26/8)
; and the default ratio accepts.
; MATRIX-READY-LABEL: loop-interchange: prepared function=matrix_memory_ratio{{ }}
; MATRIX-READY-SAME:  outer=outer.header inner=inner.header path-blocks=1 epilogue-insts=66 rematerialized=0
; MATRIX-READY-SAME:  absolute-depth=2 routing-depth=2 fission-rows=1 routing-rows=1 cross-deps=0 reductions=0
; MATRIX-READY-SAME:  byte-offset-proofs=0 requirements=0 bound=static-bound, no-bound-requirement requirement-ids={{$}}
; MATRIX-READY-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; MATRIX-READY-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; MATRIX-LIMIT-LABEL: loop-interchange: rejected function=matrix_memory_ratio{{ }}
; MATRIX-LIMIT-SAME:  outer=outer.header inner=inner.header reason=fission-context dependence matrix is unavailable{{$}}
; MATRIX-LIMIT-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: fission-context dependence matrix is unavailable
; MATRIX-LIMIT-NEXT:  remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; MEMZERO-LABEL: loop-interchange: rejected function=matrix_memory_ratio{{ }}
; MEMZERO-SAME:  outer=outer.header inner=inner.header reason=union N/E memory budget or simple-access check failed{{$}}
; MEMZERO-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: union N/E memory budget or simple-access check failed
; MEMZERO-NEXT:  remark: <unknown>:0:0: Number of loads/stores exceeded, the supported maximum can be increased with option -loop-interchange-max-mem-instr-ratio.
; The reproducer of https://github.com/llvm/llvm-project/issues/47457 fails
; the straight-line exit-to-latch check in discovery, so it reports the same
; reason under every memory ratio and prints no prepared line.
; DISCOVERY-LABEL: loop-interchange: rejected function=issue47457_guarded_epilogue{{ }}
; DISCOVERY-SAME:  outer=outer.header inner=inner.header reason=exit-to-latch path contains conditional or non-straight-line control{{$}}
;
; The speculative matrix calls must not emit the ordinary missed remarks.
; -NEXT ties the failing matrix to the feature rejection before the ordinary
; route emits its own miss.
; MATRIX-TRACE-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'unknown_alias_in_nest' with 1 path block(s), 2 instruction(s), and 0 rematerialized loop-local definition(s).
; MATRIX-TRACE-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 3 loads/stores over 16 instructions.
; MATRIX-TRACE-NEXT:  Found 2 Loads and Stores to analyze
; MATRIX-TRACE-NEXT:  Found anti dependency between Src and Dst
; MATRIX-TRACE-NEXT:   Src:  %v = load i64, ptr %ap, align 8
; MATRIX-TRACE-NEXT:   Dst:  store i64 %v, ptr %bp, align 8
; MATRIX-TRACE-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'unknown_alias_in_nest': Outer 'outer.header', Inner 'inner.header': fission-context dependence matrix is unavailable
; MATRIX-TRACE-NEXT:  loop-interchange: rejected function=unknown_alias_in_nest outer=outer.header inner=inner.header reason=fission-context dependence matrix is unavailable
; MATRIX-TRACE-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: fission-context dependence matrix is unavailable
; MATRIX-TRACE-NEXT:  Processing LoopList of size = 2 containing the following loops:
; MATRIX-TRACE:       remark: <unknown>:0:0: All loops have dependencies in all directions.
; MATRIX-TRACE-NEXT:  Populating dependency matrix failed
; MATRIX-TRACE-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'noalias_in_nest' with 1 path block(s), 2 instruction(s), and 0 rematerialized loop-local definition(s).
; MATRIX-TRACE-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 3 loads/stores over 16 instructions.
; MATRIX-TRACE-NEXT:  Found 2 Loads and Stores to analyze
; MATRIX-TRACE-NEXT:  Found 2 Loads and Stores to analyze
; MATRIX-TRACE-NEXT:  Checking if loops 'outer.header' and 'inner.header' are tightly nested
; MATRIX-TRACE-NEXT:  Checking instructions in Loop header and Loop latch
; MATRIX-TRACE-NEXT:  Loops are perfectly nested
; MATRIX-TRACE-NEXT:  Cost = -2
; MATRIX-TRACE-NEXT:  loop-interchange: prepared a complete outer-epilogue fission plan in function 'noalias_in_nest': Outer 'outer.header', Inner 'inner.header', absolute columns 0/1, routing columns 0/1, 0 cross dependence(s).
; MATRIX-TRACE-NEXT:  loop-interchange: prepared function=noalias_in_nest{{ }}
; MATRIX-TRACE-SAME:  rematerialized=0{{ }}
; MATRIX-TRACE-NEXT:  Splitting the inner loop latch
; MATRIX-TRACE-NEXT:  splitting InnerLoopHeader done
; MATRIX-TRACE-NEXT:  adjustLoopBranches called
; MATRIX-TRACE-NEXT:  remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; MATRIX-TRACE-NEXT:  remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; MATRIX-TRACE-NEXT:  loop-interchange: materialized outer-epilogue fission + interchange, top-level sibling, in function 'noalias_in_nest'.
; MATRIX-TRACE:       No Valid candidates for loop interchange.
; MATRIX-TRACE-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'matrix_memory_ratio' with 1 path block(s), 66 instruction(s), and 0 rematerialized loop-local definition(s).
; MATRIX-TRACE-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 9 loads/stores over 92 instructions.
; MATRIX-TRACE-NEXT:  Found 8 Loads and Stores to analyze
; MATRIX-TRACE-LIMIT-NEXT: loop-interchange: outer-epilogue preparation rejected candidate in function 'matrix_memory_ratio': Outer 'outer.header', Inner 'inner.header': fission-context dependence matrix is unavailable
; MATRIX-TRACE-LIMIT-NEXT: loop-interchange: rejected function=matrix_memory_ratio outer=outer.header inner=inner.header reason=fission-context dependence matrix is unavailable
; MATRIX-TRACE-LIMIT-NEXT: remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: fission-context dependence matrix is unavailable
; MATRIX-TRACE-READY-NEXT: Found anti dependency between Src and Dst
; MATRIX-TRACE-READY:      Src:  %dv = load i64, ptr %dp, align 8
; MATRIX-TRACE-READY-NEXT: Dst:  store i64 %dn, ptr %dp, align 8
; MATRIX-TRACE-READY-NEXT: Found 8 Loads and Stores to analyze
; MATRIX-TRACE-READY:      loop-interchange: prepared a complete outer-epilogue fission plan in function 'matrix_memory_ratio': Outer 'outer.header', Inner 'inner.header', absolute columns 0/1, routing columns 0/1, 0 cross dependence(s).
; MATRIX-TRACE-READY-NEXT: loop-interchange: prepared function=matrix_memory_ratio{{ }}
; MATRIX-TRACE-READY-SAME: rematerialized=0{{ }}
; MATRIX-TRACE-READY-NEXT: Splitting the inner loop latch
; MATRIX-TRACE-READY-NEXT: splitting InnerLoopHeader done
; MATRIX-TRACE-READY-NEXT: adjustLoopBranches called
; MATRIX-TRACE-READY-NEXT: remark: <unknown>:0:0: Loop interchanged with enclosing loop.
; MATRIX-TRACE-READY-NEXT: remark: <unknown>:0:0: Distributed a proven outer-loop epilogue into its own loop before interchanging the reduction nest; bound=static-bound, no-bound-requirement.
; MATRIX-TRACE-READY-NEXT: loop-interchange: materialized outer-epilogue fission + interchange, top-level sibling, in function 'matrix_memory_ratio'.
; MATRIX-TRACE-READY:      No Valid candidates for loop interchange.
;
; Only the ratio-limited run still routes this function through ordinary
; interchange, so the trailing legality trace belongs to that run alone.
; MATRIX-TRACE-LIMIT:      Processing LoopList of size = 2 containing the following loops:
; MATRIX-TRACE-LIMIT:      Found 9 Loads and Stores to analyze
; MATRIX-TRACE-LIMIT:      Dependency matrix before interchange:
; MATRIX-TRACE-LIMIT-NEXT: {{^}}= ={{ *$}}
; MATRIX-TRACE-LIMIT-NEXT: Processing InnerLoopId = 1 and OuterLoopId = 0
; MATRIX-TRACE-LIMIT:      remark: <unknown>:0:0: Cannot interchange loops because they are not tightly nested.
; MATRIX-TRACE-LIMIT-NEXT: Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
;
; LE-PREP-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'direction_le' with 1 path block(s), 8 instruction(s), and 0 rematerialized loop-local definition(s).
; LE-PREP-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 4 loads/stores over 23 instructions.
; LE-PREP-NEXT:  Found 2 Loads and Stores to analyze
; LE-PREP-NEXT:  Found 2 Loads and Stores to analyze
; LE-PREP-NEXT:  Checking if loops 'le.outer' and 'le.inner' are tightly nested
; LE-PREP-NEXT:  Checking instructions in Loop header and Loop latch
; LE-PREP-NEXT:  Loops not tightly nested
; LE-PREP-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'direction_le': Outer 'le.outer', Inner 'le.inner': virtual extracted nest failed existing interchange legality
; LE-PREP-NEXT:  loop-interchange: rejected function=direction_le outer=le.outer inner=le.inner reason=virtual extracted nest failed existing interchange legality
; LE-PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: virtual extracted nest failed existing interchange legality
; LE-PREP-NEXT:  Processing LoopList of size = 3 containing the following loops:
; LE-PREP:       Dependency matrix before interchange:
; LE-PREP-NEXT:  {{^}}* = I{{ *$}}
; LE-PREP-NEXT:  Processing InnerLoopId = 2 and OuterLoopId = 1
; LE-PREP:       Cannot prove legality, not interchanging loops 'le.root' and 'le.outer'
;
; GE-PREP-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'direction_ge' with 1 path block(s), 7 instruction(s), and 0 rematerialized loop-local definition(s).
; GE-PREP-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 4 loads/stores over 22 instructions.
; GE-PREP-NEXT:  loop-interchange: rejected an unknown, reversed, confused, or assumption-bearing N/E dependence.
; GE-PREP-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'direction_ge': Outer 'ge.outer', Inner 'ge.inner': cross-partition dependence is not statically proved N-to-E
; GE-PREP-NEXT:  loop-interchange: rejected function=direction_ge outer=ge.outer inner=ge.inner reason=cross-partition dependence is not statically proved N-to-E
; GE-PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; GE-PREP-NEXT:  Processing LoopList of size = 3 containing the following loops:
; GE-PREP:       Negative dependence vector normalized.
; GE-PREP:       Dependency matrix before interchange:
; GE-PREP-NEXT:  {{^}}* = I{{ *$}}
; GE-PREP-NEXT:  Processing InnerLoopId = 2 and OuterLoopId = 1
; GE-PREP:       Cannot prove legality, not interchanging loops 'ge.root' and 'ge.outer'
;
; NE-PREP-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'direction_ne' with 1 path block(s), 7 instruction(s), and 0 rematerialized loop-local definition(s).
; NE-PREP-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 4 loads/stores over 21 instructions.
; NE-PREP-NEXT:  Found 2 Loads and Stores to analyze
; NE-PREP-NEXT:  Found 2 Loads and Stores to analyze
; NE-PREP-NEXT:  Checking if loops 'ne.outer' and 'ne.inner' are tightly nested
; NE-PREP-NEXT:  Checking instructions in Loop header and Loop latch
; NE-PREP-NEXT:  Loops not tightly nested
; NE-PREP-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'direction_ne': Outer 'ne.outer', Inner 'ne.inner': virtual extracted nest failed existing interchange legality
; NE-PREP-NEXT:  loop-interchange: rejected function=direction_ne outer=ne.outer inner=ne.inner reason=virtual extracted nest failed existing interchange legality
; NE-PREP-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: virtual extracted nest failed existing interchange legality
; NE-PREP-NEXT:  Processing LoopList of size = 3 containing the following loops:
; NE-PREP:       Dependency matrix before interchange:
; NE-PREP-NEXT:  {{^}}* = I{{ *$}}
; NE-PREP-NEXT:  Processing InnerLoopId = 2 and OuterLoopId = 1
; NE-PREP:       Cannot prove legality, not interchanging loops 'ne.root' and 'ne.outer'
; Each convergence negative shows discovery, then the memory budget, then the
; convergence rejection with nothing between them. The implicit exclusions
; fail any scan that reaches a later proof stage.
; CONV-NEST-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'n_convergent_controlled_in_nest' with 1 path block(s), 2 instruction(s), and 0 rematerialized loop-local definition(s).
; CONV-NEST-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 2 loads/stores over 17 instructions.
; CONV-NEST-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'n_convergent_controlled_in_nest': Outer 'outer.header', Inner 'inner.header': retained nest contains an unsupported convergent operation
; CONV-NEST-NEXT:  loop-interchange: rejected function=n_convergent_controlled_in_nest outer=outer.header inner=inner.header reason=retained nest contains an unsupported convergent operation
; CONV-NEST-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: retained nest contains an unsupported convergent operation
; CONV-NEST-NEXT:  Processing LoopList of size = 2 containing the following loops:
; CONV-NEST:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
; CONV-HEADER-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'n_convergent_uncontrolled_in_outer_header' with 1 path block(s), 2 instruction(s), and 0 rematerialized loop-local definition(s).
; CONV-HEADER-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 2 loads/stores over 16 instructions.
; CONV-HEADER-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'n_convergent_uncontrolled_in_outer_header': Outer 'outer.header', Inner 'inner.header': retained nest contains an unsupported convergent operation
; CONV-HEADER-NEXT:  loop-interchange: rejected function=n_convergent_uncontrolled_in_outer_header outer=outer.header inner=inner.header reason=retained nest contains an unsupported convergent operation
; CONV-HEADER-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: retained nest contains an unsupported convergent operation
; CONV-HEADER-NEXT:  Processing LoopList of size = 2 containing the following loops:
; CONV-HEADER:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
; CONV-BODY-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'n_convergent_uncontrolled_in_inner_body' with 1 path block(s), 2 instruction(s), and 0 rematerialized loop-local definition(s).
; CONV-BODY-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 2 loads/stores over 16 instructions.
; CONV-BODY-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'n_convergent_uncontrolled_in_inner_body': Outer 'outer.header', Inner 'inner.header': retained nest contains an unsupported convergent operation
; CONV-BODY-NEXT:  loop-interchange: rejected function=n_convergent_uncontrolled_in_inner_body outer=outer.header inner=inner.header reason=retained nest contains an unsupported convergent operation
; CONV-BODY-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: retained nest contains an unsupported convergent operation
; CONV-BODY-NEXT:  Processing LoopList of size = 2 containing the following loops:
; CONV-BODY:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
; CONV-PH-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'n_convergent_uncontrolled_in_inner_ph' with 1 path block(s), 2 instruction(s), and 0 rematerialized loop-local definition(s).
; CONV-PH-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 2 loads/stores over 16 instructions.
; CONV-PH-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'n_convergent_uncontrolled_in_inner_ph': Outer 'outer.header', Inner 'inner.header': retained nest contains an unsupported convergent operation
; CONV-PH-NEXT:  loop-interchange: rejected function=n_convergent_uncontrolled_in_inner_ph outer=outer.header inner=inner.header reason=retained nest contains an unsupported convergent operation
; CONV-PH-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: retained nest contains an unsupported convergent operation
; CONV-PH-NEXT:  Processing LoopList of size = 2 containing the following loops:
; CONV-PH:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
;
; The selected-outer negatives stop at the N-to-E proof. GE additionally
; reports the reversed/unknown N/E dependence that the proof refused.
; SEL-LE-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'direction_selected_le' with 1 path block(s), 4 instruction(s), and 0 rematerialized loop-local definition(s).
; SEL-LE-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 3 loads/stores over 19 instructions.
; SEL-LE-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'direction_selected_le': Outer 'outer.header', Inner 'inner.header': cross-partition dependence is not statically proved N-to-E
; SEL-LE-NEXT:  loop-interchange: rejected function=direction_selected_le outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E
; SEL-LE-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; SEL-LE-NEXT:  Processing LoopList of size = 2 containing the following loops:
; SEL-LE:       Dependency matrix before interchange:
; SEL-LE-NEXT:  {{^}}* I{{ *$}}
; SEL-LE-NEXT:  Processing InnerLoopId = 1 and OuterLoopId = 0
; SEL-LE:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
; SEL-GE-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'direction_selected_ge' with 1 path block(s), 3 instruction(s), and 0 rematerialized loop-local definition(s).
; SEL-GE-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 3 loads/stores over 18 instructions.
; SEL-GE-NEXT:  loop-interchange: rejected an unknown, reversed, confused, or assumption-bearing N/E dependence.
; SEL-GE-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'direction_selected_ge': Outer 'outer.header', Inner 'inner.header': cross-partition dependence is not statically proved N-to-E
; SEL-GE-NEXT:  loop-interchange: rejected function=direction_selected_ge outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E
; SEL-GE-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; SEL-GE-NEXT:  Processing LoopList of size = 2 containing the following loops:
; SEL-GE:       Dependency matrix before interchange:
; SEL-GE-NEXT:  {{^}}* I{{ *$}}
; SEL-GE-NEXT:  Processing InnerLoopId = 1 and OuterLoopId = 0
; SEL-GE:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
; SEL-NE-LABEL: loop-interchange: discovered a closed outer-loop epilogue in function 'direction_selected_ne' with 1 path block(s), 3 instruction(s), and 0 rematerialized loop-local definition(s).
; SEL-NE-NEXT:  loop-interchange: outer-epilogue preparation memory budget: 3 loads/stores over 17 instructions.
; SEL-NE-NEXT:  loop-interchange: outer-epilogue preparation rejected candidate in function 'direction_selected_ne': Outer 'outer.header', Inner 'inner.header': cross-partition dependence is not statically proved N-to-E
; SEL-NE-NEXT:  loop-interchange: rejected function=direction_selected_ne outer=outer.header inner=inner.header reason=cross-partition dependence is not statically proved N-to-E
; SEL-NE-NEXT:  remark: <unknown>:0:0: did not distribute the discovered outer-loop epilogue: cross-partition dependence is not statically proved N-to-E
; SEL-NE-NEXT:  Processing LoopList of size = 2 containing the following loops:
; SEL-NE:       Dependency matrix before interchange:
; SEL-NE-NEXT:  {{^}}* I{{ *$}}
; SEL-NE-NEXT:  Processing InnerLoopId = 1 and OuterLoopId = 0
; SEL-NE:       Cannot prove legality, not interchanging loops 'outer.header' and 'inner.header'
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
; APPLIED-LABEL: define {{.*}}@e_to_n_flow(
; APPLIED-LABEL: define {{.*}}@e_to_n_anti(
; APPLIED-LABEL: define {{.*}}@e_to_n_output(
; APPLIED-LABEL: define {{.*}}@e_consumes_reduction(
; APPLIED-LABEL: define {{.*}}@e_consumes_inner_iv(
; APPLIED-LABEL: define {{.*}}@e_escapes(
; APPLIED-LABEL: define {{.*}}@exact_recurrence(
; APPLIED-LABEL: define {{.*}}@partly_exact_recurrence(
; APPLIED-LABEL: define {{.*}}@call_in_epilogue(
; APPLIED-LABEL: define {{.*}}@convergent_call_epilogue(
; APPLIED-LABEL: define {{.*}}@deopt_call_epilogue(
; APPLIED-LABEL: define {{.*}}@constrained_fp_epilogue(
; APPLIED-LABEL: define {{.*}}@unknown_alias(
; APPLIED-LABEL: define {{.*}}@non_affine_address(
; APPLIED-LABEL: define {{.*}}@volatile_epilogue(
; APPLIED-LABEL: define {{.*}}@atomic_epilogue(
; APPLIED-LABEL: define {{.*}}@fence_epilogue(
; APPLIED-LABEL: define {{.*}}@conditional_epilogue(
; APPLIED-LABEL: define {{.*}}@multiple_epilogues(
; APPLIED-LABEL: define {{.*}}@address_taken_epilogue(
; APPLIED-LABEL: define {{.*}}@triangular_bounds(
; APPLIED-LABEL: define {{.*}}@data_dependent_bounds(
; APPLIED-LABEL: define {{.*}}@loop_local_outer_bound(
; APPLIED-LABEL: define {{.*}}@noncanonical_latch(
; APPLIED-LABEL: define {{.*}}@unsupported_metadata(
; APPLIED-LABEL: define {{.*}}@outer_header_freeze(
; APPLIED-LABEL: define {{.*}}@invariant_address_n_to_e_flow(
; APPLIED-LABEL: define {{.*}}@invariant_address_n_to_e_output(
; APPLIED-LABEL: define {{.*}}@direction_le(
; APPLIED-LABEL: define {{.*}}@direction_ge(
; APPLIED-LABEL: define {{.*}}@direction_ne(
; APPLIED-LABEL: define {{.*}}@e_data_from_shared_preheader_freeze(
; APPLIED-LABEL: define {{.*}}@e_data_from_inner_preheader_load(
; APPLIED-LABEL: define {{.*}}@e_control_from_shared_preheader(
; APPLIED-LABEL: define {{.*}}@e_control_from_inner_preheader(
; APPLIED-LABEL: define {{.*}}@e_operand_closure_invalid_leaf(
; APPLIED-LABEL: define {{.*}}@n_convergent_controlled_in_nest(
; APPLIED-LABEL: define {{.*}}@n_convergent_uncontrolled_in_outer_header(
; APPLIED-LABEL: define {{.*}}@n_convergent_uncontrolled_in_inner_body(
; APPLIED-LABEL: define {{.*}}@n_convergent_uncontrolled_in_inner_ph(
; APPLIED-LABEL: define {{.*}}@direction_selected_le(
; APPLIED-LABEL: define {{.*}}@direction_selected_ge(
; APPLIED-LABEL: define {{.*}}@direction_selected_ne(
; APPLIED-LABEL: define {{.*}}@unknown_alias_in_nest(
; APPLIED-LABEL: define {{.*}}@noalias_in_nest(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %dp.epil = getelementptr i64, ptr %D, i64 %epilogue.iv
; APPLIED-NEXT:      store i64 %epilogue.iv, ptr %dp.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@matrix_memory_ratio(
; APPLIED:         epilogue:
; APPLIED-NEXT:      br label %outer.latch
; APPLIED:         exit:
; APPLIED-NEXT:      br label %epilogue.preheader
; APPLIED:         epilogue.preheader:
; APPLIED-NEXT:      br label %epilogue.header
; APPLIED:         epilogue.header:
; APPLIED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; APPLIED-NEXT:      %v01.epil = add i64 %seed, %epilogue.iv
; APPLIED-NEXT:      %v02.epil = add i64 %v01.epil, 1
; APPLIED-NEXT:      %v03.epil = add i64 %v02.epil, 1
; APPLIED-NEXT:      %v04.epil = add i64 %v03.epil, 1
; APPLIED-NEXT:      %v05.epil = add i64 %v04.epil, 1
; APPLIED-NEXT:      %v06.epil = add i64 %v05.epil, 1
; APPLIED-NEXT:      %v07.epil = add i64 %v06.epil, 1
; APPLIED-NEXT:      %v08.epil = add i64 %v07.epil, 1
; APPLIED-NEXT:      %v09.epil = add i64 %v08.epil, 1
; APPLIED-NEXT:      %v10.epil = add i64 %v09.epil, 1
; APPLIED-NEXT:      %v11.epil = add i64 %v10.epil, 1
; APPLIED-NEXT:      %v12.epil = add i64 %v11.epil, 1
; APPLIED-NEXT:      %v13.epil = add i64 %v12.epil, 1
; APPLIED-NEXT:      %v14.epil = add i64 %v13.epil, 1
; APPLIED-NEXT:      %v15.epil = add i64 %v14.epil, 1
; APPLIED-NEXT:      %v16.epil = add i64 %v15.epil, 1
; APPLIED-NEXT:      %v17.epil = add i64 %v16.epil, 1
; APPLIED-NEXT:      %v18.epil = add i64 %v17.epil, 1
; APPLIED-NEXT:      %v19.epil = add i64 %v18.epil, 1
; APPLIED-NEXT:      %v20.epil = add i64 %v19.epil, 1
; APPLIED-NEXT:      %v21.epil = add i64 %v20.epil, 1
; APPLIED-NEXT:      %v22.epil = add i64 %v21.epil, 1
; APPLIED-NEXT:      %v23.epil = add i64 %v22.epil, 1
; APPLIED-NEXT:      %v24.epil = add i64 %v23.epil, 1
; APPLIED-NEXT:      %v25.epil = add i64 %v24.epil, 1
; APPLIED-NEXT:      %v26.epil = add i64 %v25.epil, 1
; APPLIED-NEXT:      %v27.epil = add i64 %v26.epil, 1
; APPLIED-NEXT:      %v28.epil = add i64 %v27.epil, 1
; APPLIED-NEXT:      %v29.epil = add i64 %v28.epil, 1
; APPLIED-NEXT:      %v30.epil = add i64 %v29.epil, 1
; APPLIED-NEXT:      %v31.epil = add i64 %v30.epil, 1
; APPLIED-NEXT:      %v32.epil = add i64 %v31.epil, 1
; APPLIED-NEXT:      %v33.epil = add i64 %v32.epil, 1
; APPLIED-NEXT:      %v34.epil = add i64 %v33.epil, 1
; APPLIED-NEXT:      %v35.epil = add i64 %v34.epil, 1
; APPLIED-NEXT:      %v36.epil = add i64 %v35.epil, 1
; APPLIED-NEXT:      %v37.epil = add i64 %v36.epil, 1
; APPLIED-NEXT:      %v38.epil = add i64 %v37.epil, 1
; APPLIED-NEXT:      %v39.epil = add i64 %v38.epil, 1
; APPLIED-NEXT:      %v40.epil = add i64 %v39.epil, 1
; APPLIED-NEXT:      %v41.epil = add i64 %v40.epil, 1
; APPLIED-NEXT:      %v42.epil = add i64 %v41.epil, 1
; APPLIED-NEXT:      %v43.epil = add i64 %v42.epil, 1
; APPLIED-NEXT:      %v44.epil = add i64 %v43.epil, 1
; APPLIED-NEXT:      %v45.epil = add i64 %v44.epil, 1
; APPLIED-NEXT:      %v46.epil = add i64 %v45.epil, 1
; APPLIED-NEXT:      %v47.epil = add i64 %v46.epil, 1
; APPLIED-NEXT:      %v48.epil = add i64 %v47.epil, 1
; APPLIED-NEXT:      %v49.epil = add i64 %v48.epil, 1
; APPLIED-NEXT:      %v50.epil = add i64 %v49.epil, 1
; APPLIED-NEXT:      %v51.epil = add i64 %v50.epil, 1
; APPLIED-NEXT:      %v52.epil = add i64 %v51.epil, 1
; APPLIED-NEXT:      %v53.epil = add i64 %v52.epil, 1
; APPLIED-NEXT:      %v54.epil = add i64 %v53.epil, 1
; APPLIED-NEXT:      %v55.epil = add i64 %v54.epil, 1
; APPLIED-NEXT:      %v56.epil = add i64 %v55.epil, 1
; APPLIED-NEXT:      %v57.epil = add i64 %v56.epil, 1
; APPLIED-NEXT:      %v58.epil = add i64 %v57.epil, 1
; APPLIED-NEXT:      %v59.epil = add i64 %v58.epil, 1
; APPLIED-NEXT:      %v60.epil = add i64 %v59.epil, 1
; APPLIED-NEXT:      %v61.epil = add i64 %v60.epil, 1
; APPLIED-NEXT:      %v62.epil = add i64 %v61.epil, 1
; APPLIED-NEXT:      %v63.epil = add i64 %v62.epil, 1
; APPLIED-NEXT:      %v64.epil = add i64 %v63.epil, 1
; APPLIED-NEXT:      %outp.epil = getelementptr i64, ptr %Out, i64 %epilogue.iv
; APPLIED-NEXT:      store i64 %v64.epil, ptr %outp.epil, align 8
; APPLIED-NEXT:      br label %epilogue.latch
; APPLIED:         epilogue.latch:
; APPLIED-NEXT:      %i.next.epil = add i64 %epilogue.iv, 1
; APPLIED-NEXT:      %i.done.epil = icmp eq i64 %i.next.epil, 4
; APPLIED-NEXT:      br i1 %i.done.epil, label %exit.cont, label %epilogue.header
; APPLIED:         exit.cont:
; APPLIED-NEXT:      ret void
; APPLIED-LABEL: define {{.*}}@issue47457_guarded_epilogue(
;
; The same loop-info lines are checked against the loop manager's live
; LoopInfo and against a fresh rebuild of the transformed module. The block
; lists inside the nest are wildcarded because LoopInfo stores them in
; discovery order, which an incremental update and a fresh walk reach
; differently; the loop order, the depths, and the epilogue loop's two blocks
; are exact.
;
; APPLY-LOOPS-LABEL: Loop info for function 'noalias_in_nest':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
; APPLY-LOOPS-LABEL: Loop info for function 'matrix_memory_ratio':
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %epilogue.header<header>,%epilogue.latch<latch><exiting>
; APPLY-LOOPS-NEXT: Loop at depth 1 containing: %inner.header<header>,{{.*}}
; APPLY-LOOPS-NEXT:     Loop at depth 2 containing: %outer.header<header>,{{.*}}
;
; At ratio one the ratio-limited function never reaches a matrix, so it keeps
; its shape while the noalias control is still distributed.
; RATIO-LIMITED-LABEL: define {{.*}}@noalias_in_nest(
; RATIO-LIMITED:         epilogue.header:
; RATIO-LIMITED-NEXT:      %epilogue.iv = phi i64 [ 0, %epilogue.preheader ], [ %i.next.epil, %epilogue.latch ]
; RATIO-LIMITED-LABEL: define {{.*}}@matrix_memory_ratio(
; RATIO-LIMITED-NOT:     %epilogue.iv = phi

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@address_taken_epilogue_block =
    constant ptr blockaddress(@address_taken_epilogue, %epilogue.second)

define void @e_to_n_flow(ptr noalias %A, ptr noalias %S, ptr noalias %B,
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
  %sv = load double, ptr %si, align 8
  %bidx = getelementptr inbounds [1335 x double], ptr %B, i64 %j, i64 %i
  store double %sv, ptr %bidx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %i1 = add i64 %i, 1
  %si1 = getelementptr inbounds double, ptr %S, i64 %i1
  %ival = uitofp i64 %i to double
  store double %ival, ptr %si1, align 8
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

define void @e_to_n_anti(ptr noalias %A, ptr noalias %S, ptr noalias %B,
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
  store double %av, ptr %si, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %i1 = add i64 %i, 1
  %si1 = getelementptr inbounds double, ptr %S, i64 %i1
  %sr = load double, ptr %si1, align 8
  %bi = getelementptr inbounds double, ptr %B, i64 %i
  store double %sr, ptr %bi, align 8
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

define void @e_to_n_output(ptr noalias %A, ptr noalias %S, ptr noalias %R) {
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
  store double %av, ptr %si, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %i1 = add i64 %i, 1
  %si1 = getelementptr inbounds double, ptr %S, i64 %i1
  %ival = uitofp i64 %i to double
  store double %ival, ptr %si1, align 8
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

define void @e_consumes_reduction(ptr noalias %A, ptr noalias %D,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %scaled = fmul double %chk.next, %chk.next
  store double %scaled, ptr %ddiag, align 8
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

define void @e_consumes_inner_iv(ptr noalias %A, ptr noalias %D,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %j.exit = phi i64 [ %j.next, %inner.header ]
  %j.value = uitofp i64 %j.exit to double
  %dp = getelementptr inbounds double, ptr %D, i64 %i
  store double %j.value, ptr %dp, align 8
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

define void @e_escapes(ptr noalias %A, ptr noalias %D, ptr noalias %R) {
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
  %esc = phi double [ %dnew, %outer.latch ]
  %r1 = getelementptr inbounds double, ptr %R, i64 1
  store double %chk.res, ptr %R, align 8
  store double %esc, ptr %r1, align 8
  ret void
}

define void @exact_recurrence(ptr noalias %A, ptr noalias %D, ptr noalias %R) {
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

define void @partly_exact_recurrence(ptr noalias %A, ptr noalias %B,
                                     ptr noalias %D, ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chkA.i = phi double [ 0.000000e+00, %entry ], [ %chkA.next, %outer.latch ]
  %chkB.i = phi double [ 0.000000e+00, %entry ], [ %chkB.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chkA.j = phi double [ %chkA.i, %outer.header ], [ %chkA.next.j, %inner.header ]
  %chkB.j = phi double [ %chkB.i, %outer.header ], [ %chkB.next.j, %inner.header ]
  %aidx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %chkA.next.j = fadd reassoc double %chkA.j, %av
  %bidx = getelementptr inbounds [1335 x double], ptr %B, i64 %j, i64 %i
  %bv = load double, ptr %bidx, align 8
  %chkB.next.j = fadd double %chkB.j, %bv
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chkA.next = phi double [ %chkA.next.j, %inner.header ]
  %chkB.next = phi double [ %chkB.next.j, %inner.header ]
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
  %chkA.res = phi double [ %chkA.next, %outer.latch ]
  %chkB.res = phi double [ %chkB.next, %outer.latch ]
  %r1 = getelementptr inbounds double, ptr %R, i64 1
  store double %chkA.res, ptr %R, align 8
  store double %chkB.res, ptr %r1, align 8
  ret void
}

declare void @sink(double)
declare void @barrier() #0
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)

define void @call_in_epilogue(ptr noalias %A, ptr noalias %D, ptr noalias %R) {
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
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  call void @sink(double %dval)
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

define void @convergent_call_epilogue(ptr noalias %A, ptr noalias %D,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  call void @barrier() #0
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

define void @deopt_call_epilogue(ptr noalias %A, ptr noalias %D,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  call void @sink(double %dval) [ "deopt"(i32 0) ]
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

define void @constrained_fp_epilogue(ptr noalias %A, ptr noalias %D,
                                     ptr noalias %R) #1 {
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
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = call double @llvm.experimental.constrained.fadd.f64(double %dval, double 1.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict") #1
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

define void @unknown_alias(ptr %X, ptr %Y, ptr %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %xidx = getelementptr inbounds [1335 x double], ptr %X, i64 %j, i64 %i
  %xv = load double, ptr %xidx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %xv
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ydiag = getelementptr inbounds [1335 x double], ptr %Y, i64 %i, i64 %i
  %yv = load double, ptr %ydiag, align 8
  %yn = fmul double %yv, 1.500000e+00
  store double %yn, ptr %ydiag, align 8
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

define void @non_affine_address(ptr noalias %A,
                                ptr noalias dereferenceable(14257800) %D,
                                ptr noalias %IdxV, ptr noalias %R) {
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
  %didx = getelementptr inbounds [1335 x double], ptr %D, i64 %j, i64 %i
  %dv = load double, ptr %didx, align 8
  %pair = fadd reassoc double %av, %dv
  %chk.next.j = fadd reassoc double %chk.j, %pair
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ivp = getelementptr inbounds i64, ptr %IdxV, i64 %i
  %dynidx = load i64, ptr %ivp, align 8
  %ddyn = getelementptr inbounds double, ptr %D, i64 %dynidx
  %ival = uitofp i64 %i to double
  store double %ival, ptr %ddyn, align 8
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

define void @volatile_epilogue(ptr noalias %A, ptr noalias %D,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dval = load volatile double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store volatile double %dnew, ptr %ddiag, align 8
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

define void @atomic_epilogue(ptr noalias %A, ptr noalias %D,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dval = load atomic double, ptr %ddiag monotonic, align 8
  %dnew = fadd double %dval, 1.000000e+00
  store atomic double %dnew, ptr %ddiag monotonic, align 8
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

define void @fence_epilogue(ptr noalias %A, ptr noalias %D, ptr noalias %R) {
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
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  fence seq_cst
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

define void @conditional_epilogue(ptr noalias %A, ptr noalias %D,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %inner.exit, label %inner.header
inner.exit:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %odd = and i64 %i, 1
  %do = icmp eq i64 %odd, 0
  br i1 %do, label %do.epilogue, label %outer.latch
do.epilogue:
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

define void @multiple_epilogues(ptr noalias %A, ptr noalias %D, ptr noalias %E,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue.select, label %inner.header
epilogue.select:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %odd = and i64 %i, 1
  %choose.first = icmp eq i64 %odd, 0
  br i1 %choose.first, label %epilogue1, label %epilogue2
epilogue1:
  %ddiag = getelementptr inbounds [1335 x double], ptr %D, i64 %i, i64 %i
  %dval = load double, ptr %ddiag, align 8
  %dnew = fmul double %dval, 1.500000e+00
  store double %dnew, ptr %ddiag, align 8
  br label %outer.latch
epilogue2:
  %ediag = getelementptr inbounds [1335 x double], ptr %E, i64 %i, i64 %i
  %eval = load double, ptr %ediag, align 8
  %enew = fmul double %eval, 2.000000e+00
  store double %enew, ptr %ediag, align 8
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

define void @address_taken_epilogue(ptr noalias %A, ptr noalias %D,
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
  store double 1.000000e+00, ptr %dp, align 8
  br label %epilogue.second
epilogue.second:
  %ep = getelementptr inbounds double, ptr %E, i64 %i
  store double 2.000000e+00, ptr %ep, align 8
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

define void @triangular_bounds(ptr noalias %A, ptr noalias %D, ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ %i, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %aidx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %av
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
  %i.ec = icmp eq i64 %i.next, 1334
  br i1 %i.ec, label %exit, label %outer.header
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @data_dependent_bounds(ptr noalias %A, ptr noalias %Bounds,
                                   ptr noalias %D, ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %boundp = getelementptr inbounds i64, ptr %Bounds, i64 %i
  %bound = load i64, ptr %boundp, align 8
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %aidx = getelementptr [1335 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %av
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, %bound
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %dp = getelementptr inbounds double, ptr %D, i64 %i
  %dv = load double, ptr %dp, align 8
  %dn = fadd double %dv, 1.000000e+00
  store double %dn, ptr %dp, align 8
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

define void @loop_local_outer_bound(ptr noalias %A, ptr noalias %D,
                                    ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %limit = sub nsw i64 8, %i
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %aidx = getelementptr inbounds [4 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %av
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %dp = getelementptr inbounds double, ptr %D, i64 %i
  store double 1.000000e+00, ptr %dp, align 8
  br label %outer.latch
outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %continue = icmp slt i64 %i.next, %limit
  br i1 %continue, label %outer.header, label %exit
exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

define void @noncanonical_latch(ptr noalias %A, ptr noalias %D,
                                ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.latch ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.latch ]
  %aidx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %av
  %j.ec = icmp eq i64 %j, 1334
  br i1 %j.ec, label %epilogue, label %inner.latch
inner.latch:
  %j.next = add i64 %j, 1
  br label %inner.header
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

define void @unsupported_metadata(ptr noalias %A, ptr noalias %D,
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
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header, !llvm.loop !0
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

define void @outer_header_freeze(ptr noalias %A, ptr noalias %D,
                                 ptr noalias %R) {
entry:
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %choice = freeze i1 poison
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %aidx = getelementptr inbounds [4 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %selected = select i1 %choice, double %av, double 0.000000e+00
  %chk.next.j = fadd reassoc double %chk.j, %selected
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 4
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %dp = getelementptr inbounds double, ptr %D, i64 %i
  %dv = load double, ptr %dp, align 8
  %dn = fmul double %dv, 1.500000e+00
  store double %dn, ptr %dp, align 8
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

define void @invariant_address_n_to_e_flow(
    ptr noalias %A, ptr noalias %S, ptr noalias %B, ptr noalias %R) {
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
  store double %av, ptr %si, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue, label %inner.header
epilogue:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
  %sval = load double, ptr %si, align 8
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

define void @invariant_address_n_to_e_output(
    ptr noalias %A, ptr noalias %S, ptr noalias %R) {
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
  store double %av, ptr %si, align 8
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

; Generalized raw directions at the selected outer loop. Each function is a
; valid depth-three candidate chain. N stores from the selected outer header,
; while the post-inner epilogue E reads the same base. A leading-zero typed GEP keeps
; the selected-outer IV and root-IV expression in separate subscripts. Both
; addresses use the same selected-outer IV, so that component is equality.
; Their distinct root-IV expressions produce the generalized ancestor
; component.
define void @direction_le(ptr noalias %A, ptr noalias %Scratch,
                          ptr noalias %R) {
entry:
  br label %le.root
le.root:
  %k = phi i64 [ 0, %entry ], [ %k.next, %le.root.latch ]
  br label %le.outer
le.outer:
  %i = phi i64 [ 0, %le.root ], [ %i.next, %le.latch ]
  %n.root = add i64 %k, 10
  %n.ptr = getelementptr [4 x [128 x i64]], ptr %A, i64 0, i64 %i, i64 %n.root
  store i64 %i, ptr %n.ptr, align 8
  br label %le.inner
le.inner:
  %j = phi i64 [ 0, %le.outer ], [ %j.next, %le.inner ]
  %scratch.ptr = getelementptr [4 x [4 x i64]], ptr %Scratch, i64 %k, i64 %j, i64 %i
  store i64 %j, ptr %scratch.ptr, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %le.epilogue, label %le.inner
le.epilogue:
  %e.twice = shl i64 %k, 1
  %e.root = or disjoint i64 %e.twice, 1
  %e.ptr = getelementptr [4 x [128 x i64]], ptr %A, i64 0, i64 %i, i64 %e.root
  %e.value = load i64, ptr %e.ptr, align 8
  %r.row = mul i64 %k, 4
  %r.index = add i64 %r.row, %i
  %r.ptr = getelementptr i64, ptr %R, i64 %r.index
  store i64 %e.value, ptr %r.ptr, align 8
  br label %le.latch
le.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %le.root.latch, label %le.outer
le.root.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 10
  br i1 %k.done, label %exit, label %le.root
exit:
  ret void
}

define void @direction_ge(ptr noalias %A, ptr noalias %Scratch,
                          ptr noalias %R) {
entry:
  br label %ge.root
ge.root:
  %k = phi i64 [ 0, %entry ], [ %k.next, %ge.root.latch ]
  br label %ge.outer
ge.outer:
  %i = phi i64 [ 0, %ge.root ], [ %i.next, %ge.latch ]
  %n.root = mul i64 %k, 6
  %n.ptr = getelementptr [4 x [128 x i64]], ptr %A, i64 0, i64 %i, i64 %n.root
  store i64 %i, ptr %n.ptr, align 8
  br label %ge.inner
ge.inner:
  %j = phi i64 [ 0, %ge.outer ], [ %j.next, %ge.inner ]
  %scratch.ptr = getelementptr [4 x [4 x i64]], ptr %Scratch, i64 %k, i64 %j, i64 %i
  store i64 %j, ptr %scratch.ptr, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %ge.epilogue, label %ge.inner
ge.epilogue:
  %e.root = add i64 %k, 60
  %e.ptr = getelementptr [4 x [128 x i64]], ptr %A, i64 0, i64 %i, i64 %e.root
  %e.value = load i64, ptr %e.ptr, align 8
  %r.row = mul i64 %k, 4
  %r.index = add i64 %r.row, %i
  %r.ptr = getelementptr i64, ptr %R, i64 %r.index
  store i64 %e.value, ptr %r.ptr, align 8
  br label %ge.latch
ge.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %ge.root.latch, label %ge.outer
ge.root.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 13
  br i1 %k.done, label %exit, label %ge.root
exit:
  ret void
}

define void @direction_ne(ptr noalias %A, ptr noalias %Scratch,
                          ptr noalias %R) {
entry:
  br label %ne.root
ne.root:
  %k = phi i64 [ 0, %entry ], [ %k.next, %ne.root.latch ]
  br label %ne.outer
ne.outer:
  %i = phi i64 [ 0, %ne.root ], [ %i.next, %ne.latch ]
  %n.ptr = getelementptr [4 x [128 x i64]], ptr %A, i64 0, i64 %i, i64 %k
  store i64 %i, ptr %n.ptr, align 8
  br label %ne.inner
ne.inner:
  %j = phi i64 [ 0, %ne.outer ], [ %j.next, %ne.inner ]
  %scratch.ptr = getelementptr [4 x [4 x i64]], ptr %Scratch, i64 %k, i64 %j, i64 %i
  store i64 %j, ptr %scratch.ptr, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %ne.epilogue, label %ne.inner
ne.epilogue:
  %e.root = sub i64 5, %k
  %e.ptr = getelementptr [4 x [128 x i64]], ptr %A, i64 0, i64 %i, i64 %e.root
  %e.value = load i64, ptr %e.ptr, align 8
  %r.row = mul i64 %k, 4
  %r.index = add i64 %r.row, %i
  %r.ptr = getelementptr i64, ptr %R, i64 %r.index
  store i64 %e.value, ptr %r.ptr, align 8
  br label %ne.latch
ne.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %ne.root.latch, label %ne.outer
ne.root.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %exit, label %ne.root
exit:
  ret void
}

attributes #0 = { convergent nounwind }
attributes #1 = { strictfp }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}

define void @e_data_from_shared_preheader_freeze(ptr noalias %A, ptr noalias %D,
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
  %bad = freeze i64 %i
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
  store i64 %bad, ptr %dp, align 8
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

define void @e_data_from_inner_preheader_load(ptr noalias %A, ptr noalias %D,
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
  %bad = load i64, ptr %L, align 8
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
  store i64 %bad, ptr %dp, align 8
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

define void @e_control_from_shared_preheader(ptr noalias %A, ptr noalias %D,
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
  %limit.copy = add i64 3, 1
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
  %i.done = icmp eq i64 %i.next, %limit.copy
  br i1 %i.done, label %exit, label %outer.header
exit:
  br label %done
done:
  ret void
}

define void @e_control_from_inner_preheader(ptr noalias %A, ptr noalias %D,
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
  %limit.copy = add i64 3, 1
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
  %i.done = icmp eq i64 %i.next, %limit.copy
  br i1 %i.done, label %exit, label %outer.header
exit:
  br label %done
done:
  ret void
}

declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.loop()
declare i32 @fission_plain_seed(i64) memory(none) nounwind willreturn
declare i32 @fission_convergent_seed(i64) convergent memory(none) nounwind willreturn

; The invalid-leaf twin of e_operand_closure_dag: the same operand chain with
; a loaded leaf, so the walk records part of the chain and then stops.
define void @e_operand_closure_invalid_leaf(
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
  %k1 = load i64, ptr %K, align 8
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

; The controlled convergent call sits inside the nest, whose outer header
; also serves as the inner preheader.
define void @n_convergent_controlled_in_nest(ptr noalias %A, ptr noalias %D) convergent {
entry:
  %entrytok = call token @llvm.experimental.convergence.entry()
  br label %outer.header
outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %outertok = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %entrytok) ]
  br label %inner.header
inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %innertok = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %outertok) ]
  %v = call i32 @fission_convergent_seed(i64 %j) [ "convergencectrl"(token %innertok) ]
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

; The uncontrolled negatives isolate the outer header alone, a block inside
; Inner, and a retained block outside Inner but inside Outer.
define void @n_convergent_uncontrolled_in_outer_header(
    ptr noalias %A, ptr noalias %D) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %v = call i32 @fission_convergent_seed(i64 %i)
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

define void @n_convergent_uncontrolled_in_inner_body(
    ptr noalias %A, ptr noalias %D) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.ph

inner.ph:
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %inner.ph ], [ %j.next, %inner.header ]
  %v = call i32 @fission_convergent_seed(i64 %j)
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

define void @n_convergent_uncontrolled_in_inner_ph(
    ptr noalias %A, ptr noalias %D) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.ph

inner.ph:
  %v = call i32 @fission_convergent_seed(i64 %i)
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

; Raw LE, GE, and NE between N and E at the selected outer loop.
define void @direction_selected_le(ptr noalias %A, ptr noalias %B) {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.latch, %entry
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:                                     ; preds = %inner.header, %outer.header
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %n.index = add i64 %i, 10
  %np = getelementptr inbounds i64, ptr %A, i64 %n.index
  %nv = load i64, ptr %np, align 8
  %bp = getelementptr inbounds [32 x i64], ptr %B, i64 %j, i64 %i
  store i64 %nv, ptr %bp, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:                                         ; preds = %inner.header
  %e.twice = shl i64 %i, 1
  %previous = or disjoint i64 %e.twice, 1
  %ep = getelementptr inbounds i64, ptr %A, i64 %previous
  store i64 0, ptr %ep, align 8
  br label %outer.latch

outer.latch:                                      ; preds = %epilogue
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 10
  br i1 %i.done, label %exit, label %outer.header

exit:                                             ; preds = %outer.latch
  ret void
}

define void @direction_selected_ge(ptr noalias %A, ptr noalias %B) {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.latch, %entry
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:                                     ; preds = %inner.header, %outer.header
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %n.index = mul i64 %i, 6
  %np = getelementptr inbounds i64, ptr %A, i64 %n.index
  %nv = load i64, ptr %np, align 8
  %bp = getelementptr inbounds [32 x i64], ptr %B, i64 %j, i64 %i
  store i64 %nv, ptr %bp, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:                                         ; preds = %inner.header
  %previous = add i64 %i, 60
  %ep = getelementptr inbounds i64, ptr %A, i64 %previous
  store i64 0, ptr %ep, align 8
  br label %outer.latch

outer.latch:                                      ; preds = %epilogue
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 13
  br i1 %i.done, label %exit, label %outer.header

exit:                                             ; preds = %outer.latch
  ret void
}

define void @direction_selected_ne(ptr noalias %A, ptr noalias %B) {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.latch, %entry
  %i = phi i64 [ 1, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:                                     ; preds = %inner.header, %outer.header
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %np = getelementptr inbounds i64, ptr %A, i64 %i
  %nv = load i64, ptr %np, align 8
  %bp = getelementptr inbounds [8 x i64], ptr %B, i64 %j, i64 %i
  store i64 %nv, ptr %bp, align 8
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:                                         ; preds = %inner.header
  %previous = sub nuw nsw i64 5, %i
  %ep = getelementptr inbounds i64, ptr %A, i64 %previous
  store i64 0, ptr %ep, align 8
  br label %outer.latch

outer.latch:                                      ; preds = %epilogue
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 5
  br i1 %i.done, label %exit, label %outer.header

exit:                                             ; preds = %outer.latch
  ret void
}

; The alias pair. Only the unknown-alias nest loses every direction.
define void @unknown_alias_in_nest(ptr %A, ptr %B, ptr noalias %D) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %ap = getelementptr [8 x i64], ptr %A, i64 %j, i64 %i
  %bp = getelementptr [8 x i64], ptr %B, i64 %j, i64 %i
  %v = load i64, ptr %ap, align 8
  store i64 %v, ptr %bp, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:
  %dp = getelementptr i64, ptr %D, i64 %i
  store i64 %i, ptr %dp, align 8
  br label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}

define void @noalias_in_nest(ptr noalias %A, ptr %B, ptr noalias %D) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %ap = getelementptr [8 x i64], ptr %A, i64 %j, i64 %i
  %bp = getelementptr [8 x i64], ptr %B, i64 %j, i64 %i
  %v = load i64, ptr %ap, align 8
  store i64 %v, ptr %bp, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:
  %dp = getelementptr i64, ptr %D, i64 %i
  store i64 %i, ptr %dp, align 8
  br label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}

; A retained nest with four memory streams under a long scalar epilogue.
define void @matrix_memory_ratio(ptr noalias %A, ptr noalias %B,
                                 ptr noalias %C, ptr noalias %D,
                                 ptr noalias %Out, i64 %seed) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %ap = getelementptr [8 x i64], ptr %A, i64 %j, i64 %i
  %av = load i64, ptr %ap, align 8
  %an = add i64 %av, 1
  store i64 %an, ptr %ap, align 8
  %bp = getelementptr [8 x i64], ptr %B, i64 %j, i64 %i
  %bv = load i64, ptr %bp, align 8
  %bn = add i64 %bv, 1
  store i64 %bn, ptr %bp, align 8
  %cp = getelementptr [8 x i64], ptr %C, i64 %j, i64 %i
  %cv = load i64, ptr %cp, align 8
  %cn = add i64 %cv, 1
  store i64 %cn, ptr %cp, align 8
  %dp = getelementptr [8 x i64], ptr %D, i64 %j, i64 %i
  %dv = load i64, ptr %dp, align 8
  %dn = add i64 %dv, 1
  store i64 %dn, ptr %dp, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 4
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:
  %v01 = add i64 %seed, %i
  %v02 = add i64 %v01, 1
  %v03 = add i64 %v02, 1
  %v04 = add i64 %v03, 1
  %v05 = add i64 %v04, 1
  %v06 = add i64 %v05, 1
  %v07 = add i64 %v06, 1
  %v08 = add i64 %v07, 1
  %v09 = add i64 %v08, 1
  %v10 = add i64 %v09, 1
  %v11 = add i64 %v10, 1
  %v12 = add i64 %v11, 1
  %v13 = add i64 %v12, 1
  %v14 = add i64 %v13, 1
  %v15 = add i64 %v14, 1
  %v16 = add i64 %v15, 1
  %v17 = add i64 %v16, 1
  %v18 = add i64 %v17, 1
  %v19 = add i64 %v18, 1
  %v20 = add i64 %v19, 1
  %v21 = add i64 %v20, 1
  %v22 = add i64 %v21, 1
  %v23 = add i64 %v22, 1
  %v24 = add i64 %v23, 1
  %v25 = add i64 %v24, 1
  %v26 = add i64 %v25, 1
  %v27 = add i64 %v26, 1
  %v28 = add i64 %v27, 1
  %v29 = add i64 %v28, 1
  %v30 = add i64 %v29, 1
  %v31 = add i64 %v30, 1
  %v32 = add i64 %v31, 1
  %v33 = add i64 %v32, 1
  %v34 = add i64 %v33, 1
  %v35 = add i64 %v34, 1
  %v36 = add i64 %v35, 1
  %v37 = add i64 %v36, 1
  %v38 = add i64 %v37, 1
  %v39 = add i64 %v38, 1
  %v40 = add i64 %v39, 1
  %v41 = add i64 %v40, 1
  %v42 = add i64 %v41, 1
  %v43 = add i64 %v42, 1
  %v44 = add i64 %v43, 1
  %v45 = add i64 %v44, 1
  %v46 = add i64 %v45, 1
  %v47 = add i64 %v46, 1
  %v48 = add i64 %v47, 1
  %v49 = add i64 %v48, 1
  %v50 = add i64 %v49, 1
  %v51 = add i64 %v50, 1
  %v52 = add i64 %v51, 1
  %v53 = add i64 %v52, 1
  %v54 = add i64 %v53, 1
  %v55 = add i64 %v54, 1
  %v56 = add i64 %v55, 1
  %v57 = add i64 %v56, 1
  %v58 = add i64 %v57, 1
  %v59 = add i64 %v58, 1
  %v60 = add i64 %v59, 1
  %v61 = add i64 %v60, 1
  %v62 = add i64 %v61, 1
  %v63 = add i64 %v62, 1
  %v64 = add i64 %v63, 1
  %outp = getelementptr i64, ptr %Out, i64 %i
  store i64 %v64, ptr %outp, align 8
  br label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 4
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}

; Loop-simplified transcription of the reproducer in
; https://github.com/llvm/llvm-project/issues/47457 (Bugzilla 48113). The
; epilogue increments an external global through a pointer and conditionally
; loads through an external pointer, and the internal counter is
; register-promoted into the outer induction variable.
@a = global i8 0, align 1
@d = global i8 0, align 1
@b = global [1 x [2 x i32]] zeroinitializer, align 4
@c = global [1 x [9 x i8]] zeroinitializer, align 1
@e = global ptr null, align 8
@f = internal global i32 0, align 4
@g = global i16 0, align 2

define void @issue47457_guarded_epilogue() {
entry:
  %f.initial = load i32, ptr @f, align 4
  %f.guard = icmp ult i32 %f.initial, 3
  br i1 %f.guard, label %outer.ph, label %exit

outer.ph:
  br label %outer.header

outer.header:
  %f.iv = phi i32 [ %f.initial, %outer.ph ], [ %f.next, %outer.latch ]
  %f.idx = zext i32 %f.iv to i64
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %cp = getelementptr inbounds [9 x i8], ptr @c, i64 %j, i64 %f.idx
  %cv = load i8, ptr %cp, align 1
  %cv.ext = sext i8 %cv to i32
  %bp = getelementptr inbounds [2 x i32], ptr @b, i64 %j, i64 1
  store i32 %cv.ext, ptr %bp, align 4
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 3
  br i1 %j.done, label %epilogue, label %inner.header

epilogue:
  %d.old = load i8, ptr @d, align 1
  %d.new = add i8 %d.old, 1
  store i8 %d.new, ptr @d, align 1
  %a.val = load i8, ptr @a, align 1
  %a.set = icmp ne i8 %a.val, 0
  br i1 %a.set, label %if.then, label %outer.latch

if.then:
  %e.ptr = load ptr, ptr @e, align 8
  %e.val = load i8, ptr %e.ptr, align 1
  %e.ext = sext i8 %e.val to i16
  store i16 %e.ext, ptr @g, align 2
  br label %outer.latch

outer.latch:
  %f.next = add nuw nsw i32 %f.iv, 1
  %f.done = icmp eq i32 %f.next, 3
  br i1 %f.done, label %outer.exit, label %outer.header

outer.exit:
  %f.final = phi i32 [ %f.next, %outer.latch ]
  store i32 %f.final, ptr @f, align 4
  br label %exit

exit:
  ret void
}
