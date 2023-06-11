; RUN: llc -mtriple=armv7-unknown-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=armv7-apple-ios6.0.0    < %s | FileCheck %s --check-prefixes=CHECK,CHECK-IOS

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp0:
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp1:
; CHECK-NEXT:  bx lr
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",%progbits,foo{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX:         .long .Lxray_sled_0
; CHECK-LINUX:         .long .Lxray_sled_1
; CHECK-LINUX-LABEL: .Lxray_sleds_end0:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",%progbits,foo{{$}}
; CHECK-LINUX:         .long .Lxray_sleds_start0
; CHECK-LINUX-NEXT:    .long .Lxray_sleds_end0

; CHECK-IOS-LABEL: .section __DATA,xray_instr_map{{$}}
; CHECK-IOS-LABEL: Lxray_sleds_start0:
; CHECK-IOS:         .long Lxray_sled_0
; CHECK-IOS:         .long Lxray_sled_1
; CHECK-IOS-LABEL: Lxray_sleds_end0:
; CHECK-IOS-LABEL: .section __DATA,xray_fn_idx{{$}}
; CHECK-IOS:         .long Lxray_sleds_start0
; CHECK-IOS-NEXT:    .long Lxray_sleds_end0
