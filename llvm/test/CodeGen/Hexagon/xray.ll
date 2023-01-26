; RUN: llc -mtriple=hexagon-unknown-elf < %s | FileCheck %s
; RUN: llc -mtriple=hexagon-unknown-linux-musl  < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: .Lxray_sled_0:
; CHECK:       jump .Ltmp0
; CHECK:         nop
; CHECK:         nop
; CHECK:         nop
; CHECK:         nop
; CHECK-LABEL: .Ltmp0:
  ret i32 0
; CHECK-LABEL: .Lxray_sled_1:
; CHECK:       jump .Ltmp1
; CHECK:         nop
; CHECK:         nop
; CHECK:         nop
; CHECK:         nop
; CHECK-LABEL: .Ltmp1:
; CHECK:       jumpr r31
}
; CHECK:       .section xray_instr_map,"ao",@progbits,foo
; CHECK-NEXT:  .Lxray_sleds_start0:
; CHECK-NEXT:  .Ltmp2:
; CHECK-NEXT:  .word .Lxray_sled_0-.Ltmp2
; CHECK-NEXT:  .word .Lfunc_begin0-(.Ltmp2+4)
; CHECK-NEXT:  .byte 0x00
; CHECK-NEXT:  .byte 0x01
; CHECK-NEXT:  .byte 0x02
; CHECK-NEXT:  .space 5
; CHECK-LABEL: .Lxray_sleds_end0:
; CHECK-LABEL: xray_fn_idx
; CHECK:       .word .Lxray_sleds_start0{{$}}
; CHECK-NEXT:  .word .Lxray_sleds_end0{{$}}
