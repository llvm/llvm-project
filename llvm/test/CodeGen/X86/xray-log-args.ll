; When logging arguments is specified, emit the entry sled accordingly.

; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=x86_64-darwin-unknown    < %s | FileCheck %s --check-prefixes=CHECK,CHECK-MACOS

define i32 @callee(i32 %arg) nounwind noinline uwtable "function-instrument"="xray-always" "xray-log-args"="1" {
  ret i32 %arg
}
; CHECK-LABEL: callee:
; CHECK-NEXT:  Lfunc_begin0:

; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX-NEXT:  .Ltmp0:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_0-.Ltmp0
; CHECK-LINUX-NEXT:    .quad .Lfunc_begin0-(.Ltmp0+8)
; CHECK-LINUX-NEXT:    .byte 0x03
; CHECK-LINUX-NEXT:    .byte 0x01
; CHECK-LINUX-NEXT:    .byte 0x02
; CHECK-LINUX:         .zero 13
; CHECK-LINUX:       .Ltmp1:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_1-.Ltmp1
; CHECK-LINUX-NEXT:    .quad .Lfunc_begin0-(.Ltmp1+8)
; CHECK-LINUX-NEXT:    .byte 0x01
; CHECK-LINUX-NEXT:    .byte 0x01
; CHECK-LINUX-NEXT:    .byte 0x02
; CHECK-LINUX:         .zero 13

; CHECK-MACOS-LABEL: Lxray_sleds_start0:
; CHECK-MACOS-NEXT:  Ltmp0:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_0-Ltmp0
; CHECK-MACOS-NEXT:    .quad Lfunc_begin0-(Ltmp0+8)
; CHECK-MACOS-NEXT:    .byte 0x03
; CHECK-MACOS-NEXT:    .byte 0x01
; CHECK-MACOS-NEXT:    .byte 0x02
; CHECK-MACOS:         .space 13
; CHECK-MACOS:       Ltmp1:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_1-Ltmp1
; CHECK-MACOS-NEXT:    .quad Lfunc_begin0-(Ltmp1+8)
; CHECK-MACOS-NEXT:    .byte 0x01
; CHECK-MACOS-NEXT:    .byte 0x01
; CHECK-MACOS-NEXT:    .byte 0x02
; CHECK-MACOS:         .space 13

define i32 @caller(i32 %arg) nounwind noinline uwtable "function-instrument"="xray-always" "xray-log-args"="1" {
  %retval = tail call i32 @callee(i32 %arg)
  ret i32 %retval
}

; CHECK-LINUX-LABEL: .Lxray_sleds_start1:
; CHECK-LINUX-NEXT:  .Ltmp3:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_2-.Ltmp3
; CHECK-LINUX-NEXT:    .quad .Lfunc_begin1-(.Ltmp3+8)
; CHECK-LINUX-NEXT:    .byte 0x03
; CHECK-LINUX-NEXT:    .byte 0x01
; CHECK-LINUX-NEXT:    .byte 0x02
; CHECK-LINUX:         .zero 13
; CHECK-LINUX:       .Ltmp4:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_3-.Ltmp4
; CHECK-LINUX-NEXT:    .quad .Lfunc_begin1-(.Ltmp4+8)
; CHECK-LINUX-NEXT:    .byte 0x02
; CHECK-LINUX-NEXT:    .byte 0x01
; CHECK-LINUX-NEXT:    .byte 0x02
; CHECK-LINUX:         .zero 13

; CHECK-MACOS-LABEL: Lxray_sleds_start1:
; CHECK-MACOS-NEXT:  Ltmp3:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_2-Ltmp3
; CHECK-MACOS-NEXT:    .quad Lfunc_begin1-(Ltmp3+8)
; CHECK-MACOS-NEXT:    .byte 0x03
; CHECK-MACOS-NEXT:    .byte 0x01
; CHECK-MACOS-NEXT:    .byte 0x02
; CHECK-MACOS:         .space 13
; CHECK-MACOS:       Ltmp4:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_3-Ltmp4
; CHECK-MACOS-NEXT:    .quad Lfunc_begin1-(Ltmp4+8)
; CHECK-MACOS-NEXT:    .byte 0x02
; CHECK-MACOS-NEXT:    .byte 0x01
; CHECK-MACOS-NEXT:    .byte 0x02
; CHECK-MACOS:         .space 13
