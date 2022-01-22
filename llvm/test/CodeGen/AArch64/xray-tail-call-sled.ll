; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck %s

define i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: .Lxray_sled_0:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: .Ltmp0:
  ret i32 0
; CHECK-NEXT:  mov	w0, wzr
; CHECK-NEXT:  .p2align	2
; CHECK-LABEL: .Lxray_sled_1:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: .Ltmp1:
; CHECK-NEXT:  ret
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: .Lxray_sleds_start0:
; CHECK-NEXT:  .Ltmp2:
; CHECK:       .xword .Lxray_sled_0-.Ltmp2
; CHECK:       .Ltmp3:
; CHECK-NEXT:  .xword .Lxray_sled_1-.Ltmp3
; CHECK-LABEL: Lxray_sleds_end0:
; CHECK-LABEL: xray_fn_idx
; CHECK:       .xword .Lxray_sleds_start0
; CHECK-NEXT:  .xword .Lxray_sleds_end0

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
; CHECK-LABEL: .Ltmp4:
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
; CHECK-LABEL: .Ltmp5:
  %retval = tail call i32 @callee()
; CHECK:       b	callee
  ret i32 %retval
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start1:
; CHECK:       .xword .Lxray_sled_2
; CHECK:       .xword .Lxray_sled_3
; CHECK-LABEL: Lxray_sleds_end1:
; CHECK:       .section xray_fn_idx,{{.*}}
; CHECK:       .xword .Lxray_sleds_start1
; CHECK-NEXT:  .xword .Lxray_sleds_end1
