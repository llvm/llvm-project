; The default inliner doesn't elide @adder, it believes it's too costly to inline 
; adder into switcher. The ML inliner carries out that inlining, resulting in
; a smaller result (part of it is that adder gets elided).
;
; This test uses Inputs/test-module.ll, as it will share it with a similar test
; for the 'development' mode.
;
; REQUIRES: have_tf_aot
; RUN: opt -passes=scc-oz-module-inliner -enable-ml-inliner=release -S < %S/Inputs/test-module.ll 2>&1 | FileCheck %S/Inputs/test-module.ll --check-prefix=CHECK
; RUN: opt -passes=scc-oz-module-inliner -enable-ml-inliner=default -S < %S/Inputs/test-module.ll 2>&1 | FileCheck %S/Inputs/test-module.ll --check-prefix=DEFAULT
