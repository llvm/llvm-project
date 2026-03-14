;; Source:
;; void kk(int x){
;;   switch(x) {
;;     default: return;
;;   }
;; }

;; Command:
;; clang -cc1 -triple spir -emit-llvm -o test/SPIRV/OpSwitchEmpty.ll OpSwitchEmpty.cl -disable-llvm-passes

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[#X:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: OpSwitch %[[#X]] %[[#DEFAULT:]]{{$}}
; CHECK-SPIRV: %[[#DEFAULT]] = OpLabel

define spir_func void @kk(i32 %x) {
entry:
  switch i32 %x, label %sw.default [
  ]

sw.default:                                       ; preds = %entry
  ret void
}
