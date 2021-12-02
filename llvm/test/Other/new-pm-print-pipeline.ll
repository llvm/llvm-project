;; Test that the -print-pipeline-passes option correctly prints some explicitly specified pipelines.

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(adce),function(simplifycfg<bonus-inst-threshold=123;no-forward-switch-cond;switch-to-lookup;keep-loops;no-hoist-common-insts;sink-common-insts>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-0
; CHECK-0: function(adce),function(simplifycfg<bonus-inst-threshold=123;no-forward-switch-cond;switch-to-lookup;keep-loops;no-hoist-common-insts;sink-common-insts>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='module(rpo-function-attrs,require<globals-aa>,function(float2int,lower-constant-intrinsics,loop(loop-rotate)),invalidate<globals-aa>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-1
; CHECK-1: rpo-function-attrs,require<globals-aa>,function(float2int,lower-constant-intrinsics,loop(loop-rotate)),invalidate<globals-aa>

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='repeat<5>(function(mem2reg)),invalidate<all>' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-2
; CHECK-2: repeat<5>(function(mem2reg)),invalidate<all>

;; Test that we get ClassName printed when there is no ClassName to pass-name mapping (as is the case for the BitcodeWriterPass).
; RUN: opt -o /dev/null -disable-verify -print-pipeline-passes -passes='function(mem2reg)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-3
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

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(gvn<>,gvn<pre;load-pre;split-backedge-load-pre;memdep>,gvn<no-pre;no-load-pre;no-split-backedge-load-pre;no-memdep>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-11
; CHECK-11: function(gvn<>,gvn<pre;load-pre;split-backedge-load-pre;memdep>,gvn<no-pre;no-load-pre;no-split-backedge-load-pre;no-memdep>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(early-cse<>,early-cse<memssa>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-12
; CHECK-12: function(early-cse<>,early-cse<memssa>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='msan-module,function(msan,msan<>,msan<recover;kernel;track-origins=5>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-13
; CHECK-13: msan-module,function(msan<track-origins=0>,msan<track-origins=0>,msan<recover;kernel;track-origins=5>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='module(hwasan<>,hwasan<kernel;recover>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-14
; CHECK-14: hwasan<>,hwasan<kernel;recover>

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(asan<>,asan<kernel>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-15
; CHECK-15: function(asan<>,asan<kernel>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='module(loop-extract<>,loop-extract<single>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-16
; CHECK-16: loop-extract<>,loop-extract<single>

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(print<stack-lifetime><may>,print<stack-lifetime><must>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-17
; CHECK-17: function(print<stack-lifetime><may>,print<stack-lifetime><must>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(simplifycfg<bonus-inst-threshold=5;forward-switch-cond;switch-to-lookup;keep-loops;hoist-common-insts;sink-common-insts>,simplifycfg<bonus-inst-threshold=7;no-forward-switch-cond;no-switch-to-lookup;no-keep-loops;no-hoist-common-insts;no-sink-common-insts>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-18
; CHECK-18: function(simplifycfg<bonus-inst-threshold=5;forward-switch-cond;switch-to-lookup;keep-loops;hoist-common-insts;sink-common-insts>,simplifycfg<bonus-inst-threshold=7;no-forward-switch-cond;no-switch-to-lookup;no-keep-loops;no-hoist-common-insts;no-sink-common-insts>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only>,loop-vectorize<interleave-forced-only;vectorize-forced-only>)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-19
; CHECK-19: function(loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only;>,loop-vectorize<interleave-forced-only;vectorize-forced-only;>)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='inliner-wrapper,inliner-wrapper-no-mandatory-first' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-20
; CHECK-20: cgscc(inline<only-mandatory>,inline),cgscc(inline)

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='scc-oz-module-inliner' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-21
; CHECK-21: require<globals-aa>,function(invalidate<aa>),require<profile-summary>,cgscc(devirt<4>(inline<only-mandatory>,inline,{{.*}},instcombine{{.*}}))

; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='cgscc(function<eager-inv>(no-op-function)),function<eager-inv>(no-op-function)' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-22
; CHECK-22: cgscc(function<eager-inv>(no-op-function)),function<eager-inv>(no-op-function)

;; Test that the loop-nest-pass lnicm is printed with the other loop-passes in the pipeline.
; RUN: opt -disable-output -disable-verify -print-pipeline-passes -passes='function(loop-mssa(licm,loop-rotate,loop-deletion,lnicm,loop-rotate))' < %s | FileCheck %s --match-full-lines --check-prefixes=CHECK-23
; CHECK-23: function(loop-mssa(licm,loop-rotate,loop-deletion,lnicm,loop-rotate))
