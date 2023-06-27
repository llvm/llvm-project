; RUN: llc -mtriple=aarch64-linux-gnu    < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=aarch64-apple-darwin < %s | FileCheck %s --check-prefixes=CHECK,CHECK-MACOS

define i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b	#32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
  ret i32 0
; CHECK-NEXT:  mov	w0, wzr
; CHECK-NEXT:  .p2align	2
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b	#32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK-NEXT:  ret
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,callee{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX-NEXT:  [[TMP:.Ltmp[0-9]+]]:
; CHECK-LINUX:         .xword .Lxray_sled_0-[[TMP]]
; CHECK-LINUX:       [[TMP:.Ltmp[0-9]+]]:
; CHECK-LINUX-NEXT:    .xword .Lxray_sled_1-[[TMP]]
; CHECK-LINUX-LABEL: Lxray_sleds_end0:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"ao",@progbits,callee{{$}}
; CHECK-LINUX:         .xword .Lxray_sleds_start0
; CHECK-LINUX-NEXT:    .xword 2

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start0:
; CHECK-MACOS-NEXT:  [[TMP:Ltmp[0-9]+]]:
; CHECK-MACOS:         .quad Lxray_sled_0-[[TMP]]
; CHECK-MACOS:       [[TMP:Ltmp[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_1-[[TMP]]
; CHECK-MACOS-LABEL: Lxray_sleds_end0:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-MACOS:       [[IDX:lxray_fn_idx[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad lxray_sleds_start0-[[IDX]]
; CHECK-MACOS-NEXT:    .quad 2

define i32 @caller() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_2:
; CHECK-NEXT:  b	#32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_3:
; CHECK-NEXT:  b	#32
; CHECK-COUNT-7:  nop
; CHECK-NEXT:  Ltmp[[#]]:
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
; CHECK-LINUX-LABEL: .section xray_fn_idx,"ao",@progbits,caller{{$}}
; CHECK-LINUX:       [[IDX:\.Lxray_fn_idx[0-9]+]]:
; CHECK-LINUX-NEXT:    .xword .Lxray_sleds_start1-[[IDX]]
; CHECK-LINUX-NEXT:    .xword 2

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start1:
; CHECK-MACOS:         .quad Lxray_sled_2
; CHECK-MACOS:         .quad Lxray_sled_3
; CHECK-MACOS-LABEL: Lxray_sleds_end1:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-MACOS:       [[IDX:lxray_fn_idx[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad lxray_sleds_start1-[[IDX]]
; CHECK-MACOS-NEXT:    .quad 2
