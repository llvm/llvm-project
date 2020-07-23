; REQUIRES: !have_tf_api
; RUN: opt -passes='print<inliner-size-estimator>' -S < %S/Inputs/size-estimator.ll 2>&1 | FileCheck %s

; CHECK: [InlineSizeEstimatorAnalysis] size estimate for branches: None