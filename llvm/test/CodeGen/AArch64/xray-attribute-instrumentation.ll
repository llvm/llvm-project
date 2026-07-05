; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=aarch64-apple-darwin      < %s | FileCheck %s -check-prefixes=CHECK,CHECK-MACOS

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: foo:
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b  #32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b  #32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK-NEXT:  ret
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,foo{{$}}
; CHECK-LINUX-LABEL: Lxray_sleds_start0:
; CHECK-LINUX:         .xword .Lxray_sled_0
; CHECK-LINUX:         .xword .Lxray_sled_1
; CHECK-LINUX-LABEL: Lxray_sleds_end0:

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start0:
; CHECK-MACOS:         .quad Lxray_sled_0
; CHECK-MACOS:         .quad Lxray_sled_1
; CHECK-MACOS-LABEL: Lxray_sleds_end0:

define i32 @bar() nounwind noinline uwtable "function-instrument"="xray-never" "function-instrument"="xray-always" {
; CHECK-LABEL: bar:
; CHECK-LABEL: Lxray_sled_2:
; CHECK-NEXT:  b  #32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
  ret i32 0
; CHECK-LABEL: Lxray_sled_3:
; CHECK-NEXT:  b  #32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK-NEXT:  ret
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,bar{{$}}
; CHECK-LINUX-LABEL: Lxray_sleds_start1:
; CHECK-LINUX:         .xword .Lxray_sled_2
; CHECK-LINUX:         .xword .Lxray_sled_3
; CHECK-LINUX-LABEL: Lxray_sleds_end1:

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start1:
; CHECK-MACOS:         .quad Lxray_sled_2
; CHECK-MACOS:         .quad Lxray_sled_3
; CHECK-MACOS-LABEL: Lxray_sleds_end1:

define i32 @instrumented() nounwind noinline uwtable "xray-instruction-threshold"="1" {
; CHECK-LABEL: instrumented:
; CHECK-LABEL: Lxray_sled_4:
; CHECK-NEXT:  b  #32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
  ret i32 0
; CHECK-LABEL: Lxray_sled_5:
; CHECK-NEXT:  b  #32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK-NEXT:  ret
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,instrumented{{$}}
; CHECK-LINUX-LABEL: Lxray_sleds_start2:
; CHECK-LINUX:         .xword .Lxray_sled_4
; CHECK-LINUX:         .xword .Lxray_sled_5
; CHECK-LINUX-LABEL: Lxray_sleds_end2:

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start2:
; CHECK-MACOS:         .quad Lxray_sled_4
; CHECK-MACOS:         .quad Lxray_sled_5
; CHECK-MACOS-LABEL: Lxray_sleds_end2:

define i32 @not_instrumented() nounwind noinline uwtable "xray-instruction-threshold"="1" "function-instrument"="xray-never" {
; CHECK-LABEL: not_instrumented
; CHECK-NOT: Lxray_sled_6
  ret i32 0
; CHECK:  ret
}
