; When logging arguments is specified, emit the entry sled accordingly.

; RUN: llc -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -filetype=asm -o - -mtriple=x86_64-darwin-unknown < %s | FileCheck %s

define i32 @callee(i32 %arg) nounwind noinline uwtable "function-instrument"="xray-always" "xray-log-args"="1" {
  ret i32 %arg
}
; CHECK-LABEL: Lxray_synthetic_0:
; CHECK:	.quad	{{\.?}}Lxray_sled_0
; CHECK:	.quad	{{_?}}callee
; CHECK:	.byte	3
; CHECK:	.byte	1
; CHECK:	.{{(zero|space)}}	14
; CHECK:	.quad	{{\.?}}Lxray_sled_1
; CHECK:	.quad	{{_?}}callee
; CHECK:	.byte	1
; CHECK:	.byte	1
; CHECK:	.{{(zero|space)}}	14

define i32 @caller(i32 %arg) nounwind noinline uwtable "function-instrument"="xray-always" "xray-log-args"="1" {
  %retval = tail call i32 @callee(i32 %arg)
  ret i32 %retval
}
; CHECK-LABEL: Lxray_synthetic_1:
; CHECK:	.quad	{{\.?}}Lxray_sled_2
; CHECK:	.quad	{{_?}}caller
; CHECK:	.byte	3
; CHECK:	.byte	1
; CHECK:	.{{(zero|space)}}	14
; CHECK:	.quad	{{\.?}}Lxray_sled_3
; CHECK:	.quad	{{_?}}caller
; CHECK:	.byte	2
; CHECK:	.byte	1
; CHECK:	.{{(zero|space)}}	14
