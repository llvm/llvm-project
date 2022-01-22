; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: foo:
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
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b  #32
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

; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0
; CHECK:       .xword .Lxray_sled_0
; CHECK:       .xword .Lxray_sled_1
; CHECK-LABEL: Lxray_sleds_end0

define i32 @bar() nounwind noinline uwtable "function-instrument"="xray-never" "function-instrument"="xray-always" {
; CHECK-LABEL: bar:
; CHECK-LABEL: Lxray_sled_2:
; CHECK-NEXT:  b  #32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp4:
  ret i32 0
; CHECK-LABEL: Lxray_sled_3:
; CHECK-NEXT:  b  #32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp5:
; CHECK-NEXT:  ret
}

; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start1
; CHECK:       .xword .Lxray_sled_2
; CHECK:       .xword .Lxray_sled_3
; CHECK-LABEL: Lxray_sleds_end1

define i32 @instrumented() nounwind noinline uwtable "xray-instruction-threshold"="1" {
; CHECK-LABEL: instrumented:
; CHECK-LABEL: Lxray_sled_4:
; CHECK-NEXT:  b  #32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp8:
  ret i32 0
; CHECK-LABEL: Lxray_sled_5:
; CHECK-NEXT:  b  #32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp9:
; CHECK-NEXT:  ret
}

; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start2
; CHECK:       .xword .Lxray_sled_4
; CHECK:       .xword .Lxray_sled_5
; CHECK-LABEL: Lxray_sleds_end2

define i32 @not_instrumented() nounwind noinline uwtable "xray-instruction-threshold"="1" "function-instrument"="xray-never" {
; CHECK-LABEL: not_instrumented
; CHECK-NOT: .Lxray_sled_6
  ret i32 0
; CHECK:  ret
}
