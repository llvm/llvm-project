; Check that, in the absence of dependencies, we emit an error message when
; trying to use ML-driven inlining.
; REQUIRES: !have_tf_aot
; RUN: not opt -passes=scc-oz-module-inliner -enable-ml-inliner=development -S < %s 2>&1 | FileCheck %s
; RUN: not opt -passes=scc-oz-module-inliner -enable-ml-inliner=release -S < %s 2>&1 | FileCheck %s

declare i64 @f1()

; CHECK: Could not setup Inlining Advisor for the requested mode and/or options