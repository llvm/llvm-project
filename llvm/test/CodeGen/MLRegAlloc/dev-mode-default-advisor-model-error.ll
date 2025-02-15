; REQUIRES: have_tflite
; REQUIRES: x86_64-linux
;
; Checking that if we specify a model, but do not specify the development
; advisor, we get an error.
;
; RUN: not llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=default \
; RUN:   -regalloc-model=/model_foo %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: A model has been passed in, but the default eviction advisor analysis was requested. The model will not be used.

define i32 @foo() {
  ret i32 0
}
