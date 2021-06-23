; The default inliner doesn't elide @adder, it believes it's too costly to inline 
; adder into switcher. The ML inliner carries out that inlining, resulting in
; a smaller result (part of it is that adder gets elided).
;
; This test uses Inputs/test-module.ll, as it shares it with a similar test
; for the 'release' mode.
;
; REQUIRES: have_tf_api
; RUN: rm -rf %t && mkdir %t
; RUN: %python %S/../../../../lib/Analysis/models/generate_mock_model.py %S/../../../../lib/Analysis/models/inlining/config.py %t
; RUN: opt -passes=scc-oz-module-inliner -enable-ml-inliner=default -S < %S/Inputs/test-module.ll 2>&1 | FileCheck %S/Inputs/test-module.ll --check-prefix=DEFAULT
; RUN: opt -passes=scc-oz-module-inliner -enable-ml-inliner=development -ml-inliner-model-under-training=%t -S < %S/Inputs/test-module.ll 2>&1 | FileCheck %S/Inputs/test-module.ll --check-prefix=CHECK
