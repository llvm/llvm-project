; RUN: llc -mtriple=armv6-unknown-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=armv6-apple-ios6.0.0    < %s | FileCheck %s --check-prefixes=CHECK,CHECK-IOS

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  Ltmp[[#]]:
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK-NEXT:  bx	lr
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",%progbits,foo{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX:         .long .Lxray_sled_0
; CHECK-LINUX:         .long .Lxray_sled_1
; CHECK-LINUX-LABEL: .Lxray_sleds_end0:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"ao",%progbits,foo{{$}}
; CHECK-LINUX:         .long .Lxray_sleds_start0-.Lxray_fn_idx0
; CHECK-LINUX-NEXT:    .long 2

; CHECK-IOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-IOS-LABEL: lxray_sleds_start0:
; CHECK-IOS:         .long Lxray_sled_0
; CHECK-IOS:         .long Lxray_sled_1
; CHECK-IOS-LABEL: Lxray_sleds_end0:
; CHECK-IOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-IOS:         .long lxray_sleds_start0-lxray_fn_idx0
; CHECK-IOS-NEXT:    .long 2
