; When logging arguments is specified, emit the entry sled accordingly.

; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-darwin-unknown < %s | FileCheck %s

define i32 @callee(i32 %arg) nounwind noinline uwtable "function-instrument"="xray-always" "xray-log-args"="1" {
  ret i32 %arg
}
; CHECK-LABEL: callee:
; CHECK-NEXT:  Lfunc_begin0:

; CHECK-LABEL: Lxray_sleds_start0:
; CHECK-NEXT:  Ltmp0:
; CHECK-NEXT:   .quad {{\.?}}Lxray_sled_0-{{\.?}}Ltmp0
; CHECK-NEXT:   .quad {{\.?}}Lfunc_begin0-({{\.?}}Ltmp0+8)
; CHECK-NEXT:   .byte 0x03
; CHECK-NEXT:   .byte 0x01
; CHECK-NEXT:   .byte 0x02
; CHECK:        .{{(zero|space)}}  13
; CHECK:       Ltmp1:
; CHECK-NEXT:   .quad {{\.?}}Lxray_sled_1-{{\.?}}Ltmp1
; CHECK-NEXT:   .quad {{\.?}}Lfunc_begin0-({{\.?}}Ltmp1+8)
; CHECK-NEXT:   .byte 0x01
; CHECK-NEXT:   .byte 0x01
; CHECK-NEXT:   .byte 0x02
; CHECK:  .{{(zero|space)}}  13

define i32 @caller(i32 %arg) nounwind noinline uwtable "function-instrument"="xray-always" "xray-log-args"="1" {
  %retval = tail call i32 @callee(i32 %arg)
  ret i32 %retval
}
; CHECK-LABEL: Lxray_sleds_start1:
; CHECK-NEXT:  Ltmp3:
; CHECK-NEXT:   .quad {{\.?}}Lxray_sled_2-{{\.?}}Ltmp3
; CHECK-NEXT:   .quad {{\.?}}Lfunc_begin1-({{\.?}}Ltmp3+8)
; CHECK-NEXT:   .byte 0x03
; CHECK-NEXT:   .byte 0x01
; CHECK-NEXT:   .byte 0x02
; CHECK:  .{{(zero|space)}}  13
; CHECK:       Ltmp4:
; CHECK-NEXT:   .quad {{\.?}}Lxray_sled_3-{{\.?}}Ltmp4
; CHECK-NEXT:   .quad {{\.?}}Lfunc_begin1-({{\.?}}Ltmp4+8)
; CHECK-NEXT:   .byte 0x02
; CHECK-NEXT:   .byte 0x01
; CHECK-NEXT:   .byte 0x02
; CHECK:  .{{(zero|space)}}  13
