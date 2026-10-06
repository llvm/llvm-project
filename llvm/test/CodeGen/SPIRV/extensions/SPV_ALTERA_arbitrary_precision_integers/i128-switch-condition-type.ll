; The fix for the crash on odd-width switch condition types is orthogonal to
; the arbitrary precision integers extension: with the extension enabled, an
; odd-width type still lowers to itself rather than being widened.

; RUN: llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#I128:]] = OpTypeInt 128 0
; CHECK: %[[#COND:]] = OpFunctionParameter %[[#I128]]
; CHECK: OpSwitch %[[#COND]] %[[#]] 0 0 0 0 %[[#]] 1 0 0 0 %[[#]] 2 0 0 0 %[[#]]

define spir_func void @apint_switch(i128 %n, ptr %out) {
entry:
  switch i128 %n, label %d [
    i128 0, label %a
    i128 1, label %b
    i128 2, label %c
  ]
a:
  br label %m
b:
  br label %m
c:
  br label %m
d:
  br label %m
m:
  %r = phi i128 [ 40, %d ], [ 20, %a ], [ 30, %b ], [ 50, %c ]
  %add = add i128 %r, %n
  store i128 %add, ptr %out
  ret void
}
