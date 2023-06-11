; RUN: llc -mtriple=aarch64-linux-gnu    < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=aarch64-apple-darwin < %s | FileCheck %s --check-prefixes=CHECK,CHECK-MACOS

define i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp0:
  ret i32 0
; CHECK-NEXT:  mov	w0, wzr
; CHECK-NEXT:  .p2align	2
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp1:
; CHECK-NEXT:  ret
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,callee{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX-NEXT:  .Ltmp2:
; CHECK-LINUX:         .xword .Lxray_sled_0-.Ltmp2
; CHECK-LINUX:       .Ltmp3:
; CHECK-LINUX-NEXT:    .xword .Lxray_sled_1-.Ltmp3
; CHECK-LINUX-LABEL: Lxray_sleds_end0:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",@progbits,callee{{$}}
; CHECK-LINUX:         .xword .Lxray_sleds_start0
; CHECK-LINUX-NEXT:    .xword .Lxray_sleds_end0

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map{{$}}
; CHECK-MACOS-LABEL: Lxray_sleds_start0:
; CHECK-MACOS-NEXT:  Ltmp2:
; CHECK-MACOS:         .quad Lxray_sled_0-Ltmp2
; CHECK-MACOS:       Ltmp3:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_1-Ltmp3
; CHECK-MACOS-LABEL: Lxray_sleds_end0:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx{{$}}
; CHECK-MACOS:         .quad Lxray_sleds_start0
; CHECK-MACOS-NEXT:    .quad Lxray_sleds_end0

define i32 @caller() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_2:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp4:
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_3:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp5:
  %retval = tail call i32 @callee()
; CHECK-LINUX: b	callee
; CHECK-MACOS: b	_callee
  ret i32 %retval
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,caller{{$}}
; CHECK-LINUX-LABEL: Lxray_sleds_start1:
; CHECK-LINUX:         .xword .Lxray_sled_2
; CHECK-LINUX:         .xword .Lxray_sled_3
; CHECK-LINUX-LABEL: Lxray_sleds_end1:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",@progbits,caller{{$}}
; CHECK-LINUX:         .xword .Lxray_sleds_start1
; CHECK-LINUX-NEXT:    .xword .Lxray_sleds_end1

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map{{$}}
; CHECK-MACOS-LABEL: Lxray_sleds_start1:
; CHECK-MACOS:         .quad Lxray_sled_2
; CHECK-MACOS:         .quad Lxray_sled_3
; CHECK-MACOS-LABEL: Lxray_sleds_end1:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx{{$}}
; CHECK-MACOS:         .quad Lxray_sleds_start1
; CHECK-MACOS-NEXT:    .quad Lxray_sleds_end1
