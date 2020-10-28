; REQUIRES: have_tf_api
; RUN: opt -passes='print<inliner-size-estimator>' -S < %S/Inputs/size-estimator.ll 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -passes='print<inliner-size-estimator>' -ml-inliner-ir2native-model=%S/../../../../unittests/Analysis/Inputs/ir2native_x86_64_model -S < %S/Inputs/size-estimator.ll 2>&1 | FileCheck %s

; DEFAULT: [InlineSizeEstimatorAnalysis] size estimate for branches: None
; CHECK: [InlineSizeEstimatorAnalysis] size estimate for branches: 28