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
