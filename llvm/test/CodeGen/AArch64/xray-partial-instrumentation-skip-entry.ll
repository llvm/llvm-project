; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" "xray-skip-entry" {
; CHECK-NOT: Lxray_sled_0:
  ret i32 0
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b  #32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp0:
; CHECK-NEXT:  ret
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0
; CHECK:       .xword .Lxray_sled_0
; CHECK-LABEL: Lxray_sleds_end0
