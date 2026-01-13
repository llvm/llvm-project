;; Test that the -print-pipeline-passes option correctly prints some explicitly specified pipelines.

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(adce),function(adce)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-0
; CHECK-0: function(adce),function(adce)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='module(rpo-function-attrs,require<globals-aa>,function(float2int,lower-constant-intrinsics,loop(loop-rotate)),invalidate<globals-aa>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-1
; CHECK-1: rpo-function-attrs,require<globals-aa>,function(float2int,lower-constant-intrinsics,loop(loop-rotate<header-duplication;no-prepare-for-lto>)),invalidate<globals-aa>

;; Test that we get ClassName printed when there is no ClassName to pass-name mapping (as is the case for the BitcodeWriterPass).
; RUN: opt -o /dev/null -disable-verify -print-pipeline-passes -passes='function(mem2reg)' < %s -disable-pipeline-verification | FileCheck %s --match-full-lines --check-prefixes=CHECK-3
; CHECK-3: function(mem2reg),BitcodeWriterPass

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(loop-mssa(indvars))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-4
; CHECK-4: function(loop-mssa(indvars))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='cgscc(argpromotion,require<no-op-cgscc>,no-op-cgscc,devirt<7>(inline,no-op-cgscc)),function(loop(require<no-op-loop>))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-5
; CHECK-5: cgscc(argpromotion,require<no-op-cgscc>,no-op-cgscc,devirt<7>(inline,no-op-cgscc)),function(loop(require<no-op-loop>))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(ee-instrument<>,ee-instrument<post-inline>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-6
; CHECK-6: function(ee-instrument<>,ee-instrument<post-inline>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='loop(simple-loop-unswitch<nontrivial;trivial>,simple-loop-unswitch<no-nontrivial;no-trivial>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-7
; CHECK-7: function(loop(simple-loop-unswitch<nontrivial;trivial>,simple-loop-unswitch<no-nontrivial;no-trivial>))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(mldst-motion<split-footer-bb>,mldst-motion<no-split-footer-bb>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-8
; CHECK-8: function(mldst-motion<split-footer-bb>,mldst-motion<no-split-footer-bb>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(lower-matrix-intrinsics<>,lower-matrix-intrinsics<minimal>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-9
; CHECK-9: function(lower-matrix-intrinsics<>,lower-matrix-intrinsics<minimal>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(loop-unroll<>,loop-unroll<partial;peeling;runtime;upperbound;profile-peeling;full-unroll-max=5;O1>,loop-unroll<no-partial;no-peeling;no-runtime;no-upperbound;no-profile-peeling;full-unroll-max=7;O1>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-10
; CHECK-10: function(loop-unroll<O2>,loop-unroll<partial;peeling;runtime;upperbound;profile-peeling;full-unroll-max=5;O1>,loop-unroll<no-partial;no-peeling;no-runtime;no-upperbound;no-profile-peeling;full-unroll-max=7;O1>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(gvn<>,gvn<pre;load-pre;split-backedge-load-pre;memdep;memoryssa>,gvn<no-pre;no-load-pre;no-split-backedge-load-pre;no-memdep;no-memoryssa>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-11
; CHECK-11: function(gvn<>,gvn<pre;load-pre;split-backedge-load-pre;no-memdep;memoryssa>,gvn<no-pre;no-load-pre;no-split-backedge-load-pre;memdep;no-memoryssa>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(early-cse<>,early-cse<memssa>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-12
; CHECK-12: function(early-cse<>,early-cse<memssa>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='msan,module(msan,msan<>,msan<recover;kernel;eager-checks;track-origins=5>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-13
; CHECK-13: msan<track-origins=0>,msan<track-origins=0>,msan<track-origins=0>,msan<recover;kernel;eager-checks;track-origins=5>

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='module(hwasan<>,hwasan<kernel;recover>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-14
; CHECK-14: hwasan<>,hwasan<kernel;recover>

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='module(loop-extract<>,loop-extract<single>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-16
; CHECK-16: loop-extract<>,loop-extract<single>

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(print<stack-lifetime><may>,print<stack-lifetime><must>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-17
; CHECK-17: function(print<stack-lifetime><may>,print<stack-lifetime><must>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(simplifycfg<bonus-inst-threshold=5;forward-switch-cond;switch-to-lookup;keep-loops;hoist-common-insts;hoist-loads-stores-with-cond-faulting;sink-common-insts;speculate-blocks;simplify-cond-branch;speculate-unpredictables>,simplifycfg<bonus-inst-threshold=7;no-forward-switch-cond;no-switch-to-lookup;no-keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;no-speculate-blocks;no-simplify-cond-branch;no-speculate-unpredictables>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-18
; CHECK-18: function(simplifycfg<bonus-inst-threshold=5;forward-switch-cond;no-switch-range-to-icmp;no-switch-to-arithmetic;switch-to-lookup;keep-loops;hoist-common-insts;hoist-loads-stores-with-cond-faulting;sink-common-insts;speculate-blocks;simplify-cond-branch;speculate-unpredictables>,simplifycfg<bonus-inst-threshold=7;no-forward-switch-cond;no-switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;no-keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;no-speculate-blocks;no-simplify-cond-branch;no-speculate-unpredictables>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only>,loop-vectorize<interleave-forced-only;vectorize-forced-only>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-19
; CHECK-19: function(loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only;>,loop-vectorize<interleave-forced-only;vectorize-forced-only;>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='inliner-wrapper,inliner-wrapper-no-mandatory-first' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-20
; CHECK-20: cgscc(inline<only-mandatory>,inline),cgscc(inline)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='scc-oz-module-inliner' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-21
; CHECK-21: require<globals-aa>,function(invalidate<aa>),require<profile-summary>,cgscc(devirt<4>(inline,{{.*}},instcombine{{.*}}))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='cgscc(function<eager-inv>(no-op-function)),function<eager-inv>(no-op-function)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-22
; CHECK-22: cgscc(function<eager-inv>(no-op-function)),function<eager-inv>(no-op-function)

;; Test that the loop-nest-pass lnicm is printed with the other loop-passes in the pipeline.
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(loop-mssa(licm,loop-rotate,loop-deletion,lnicm,loop-rotate))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-23
; CHECK-23: function(loop-mssa(licm<allowspeculation>,loop-rotate<header-duplication;no-prepare-for-lto>,loop-deletion,lnicm<allowspeculation>,loop-rotate<header-duplication;no-prepare-for-lto>))

;; Test that -debugify and -check-debugify is printed correctly.
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='debugify,no-op-function,check-debugify' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-24
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -enable-debugify -passes='no-op-function' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-24
; CHECK-24: debugify,function(no-op-function),check-debugify

;; Test that LICM & LNICM with options.
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(loop-mssa(licm<allowspeculation>,licm<no-allowspeculation>,lnicm<allowspeculation>,lnicm<no-allowspeculation>))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-25
; CHECK-25: function(loop-mssa(licm<allowspeculation>,licm<no-allowspeculation>,lnicm<allowspeculation>,lnicm<no-allowspeculation>))

;; Test coro-cond.
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='coro-cond(no-op-module)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-26
; CHECK-26: coro-cond(no-op-module)

;; Test that -print-pipeline-passes is parsable (implicitly done with -print-pipeline-passes) for various default pipelines.
; RUN: opt -disable-output -passes='default<O0>' < %s
; RUN: opt -disable-output -passes='default<O1>' < %s
; RUN: opt -disable-output -passes='default<O2>' < %s
; RUN: opt -disable-output -passes='default<O3>' < %s

;; Test SeparateConstOffsetFromGEPPass option.
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='separate-const-offset-from-gep<lower-gep>' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-27
; CHECK-27: function(separate-const-offset-from-gep<lower-gep>)

;; Test InstCombine options - the first pass checks default settings, and the second checks customized options.
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(instcombine,instcombine<no-verify-fixpoint;max-iterations=42>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-28
; CHECK-28: function(instcombine<max-iterations=1;verify-fixpoint>,instcombine<max-iterations=42;no-verify-fixpoint>)

;; Test function-attrs
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='cgscc(function-attrs<skip-non-recursive-function-attrs>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-29
; CHECK-29: cgscc(function-attrs<skip-non-recursive-function-attrs>)

;; Test cgscc -> function adaptor
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='cgscc(function<eager-inv;no-rerun>(no-op-function))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-30
; CHECK-30: cgscc(function<eager-inv;no-rerun>(no-op-function))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='cgscc(function<no-rerun;eager-inv>(no-op-function))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-31
; CHECK-31: cgscc(function<eager-inv;no-rerun>(no-op-function))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='cgscc(function<no-rerun>(no-op-function))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-32
; CHECK-32: cgscc(function<no-rerun>(no-op-function))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(loop(loop-rotate<no-header-duplication;no-prepare-for-lto>))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-33
; CHECK-33: function(loop(loop-rotate<no-header-duplication;no-prepare-for-lto>))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='globaldce' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-34
; CHECK-34: globaldce

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='globaldce<vfe-linkage-unit-visibility>' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-35
; CHECK-35: globaldce<vfe-linkage-unit-visibility>

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='speculative-execution<only-if-divergent-target>' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-36
; CHECK-36: function(speculative-execution<only-if-divergent-target>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='module(asan<>,asan<kernel;use-after-scope>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-37
; CHECK-37: asan<>,asan<kernel;use-after-scope>
