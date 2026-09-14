; Check that, in the absence of dependencies or a selected model, we emit an
; error message when trying to use ML-driven inlining.
; REQUIRES: !have_tf_aot
; REQUIRES: !have_tflite
; RUN: not opt -passes=scc-oz-module-inliner -enable-ml-inliner=development -S < %s 2>&1 | FileCheck %s
; RUN: not opt -passes=scc-oz-module-inliner -enable-ml-inliner=release -S < %s 2>&1 | FileCheck %s
; RUN: %if have_mlir_lowering_inliner %{ not opt -passes=scc-oz-module-inliner -enable-ml-inliner=release -mlgo-model=default -S < %s 2>&1 | FileCheck %s %}
; RUN: %if have_mlir_lowering_inliner %{ not opt -passes=scc-oz-module-inliner -enable-ml-inliner=release -mlgo-model=invalid_model -S < %s 2>&1 | FileCheck %s --check-prefix=INVALID %}

declare i64 @f1()

; CHECK: Could not setup Inlining Advisor for the requested mode and/or options
; INVALID: {{.*}}opt{{.*}}: for the --mlgo-model option: Cannot find option named 'invalid_model'!
