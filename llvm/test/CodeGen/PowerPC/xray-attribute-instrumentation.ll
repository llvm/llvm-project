; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: foo:
; CHECK-NEXT:  .Lfunc_begin0:
; CHECK:       .Ltmp[[#l:]]:
; CHECK-NEXT:         b .Ltmp[[#l+1]]
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionEntry
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
; CHECK-NEXT:  .Ltmp[[#l+1]]:
  ret i32 0
; CHECK:       .Ltmp[[#l+2]]:
; CHECK-NEXT:         blr
; CHECK-NEXT:         nop
; CHECK-NEXT:         std 0, -8(1)
; CHECK-NEXT:         mflr 0
; CHECK-NEXT:         bl __xray_FunctionExit
; CHECK-NEXT:         nop
; CHECK-NEXT:         mtlr 0
}
; CHECK-LABEL: xray_instr_map,"ao",@progbits,foo{{$}}
; CHECK:      .Lxray_sleds_start0:
; CHECK-NEXT: [[TMP:.Ltmp[0-9]+]]:
; CHECK-NEXT:         .quad   .Ltmp[[#l]]-[[TMP]]
; CHECK-NEXT:         .quad   .Lfunc_begin0-([[TMP]]+8)
; CHECK-NEXT:         .byte   0x00
; CHECK-NEXT:         .byte   0x01
; CHECK-NEXT:         .byte   0x02
; CHECK-NEXT:         .space  13
; CHECK-NEXT: [[TMP:.Ltmp[0-9]+]]:
; CHECK-NEXT:         .quad   .Ltmp[[#l+2]]-[[TMP]]
; CHECK-NEXT:         .quad   .Lfunc_begin0-([[TMP]]+8)
; CHECK-NEXT:         .byte   0x01
; CHECK-NEXT:         .byte   0x01
; CHECK-NEXT:         .byte   0x02
; CHECK-NEXT:         .space  13
; CHECK-NEXT: .Lxray_sleds_end0:
; CHECK-LABEL: xray_fn_idx,"ao",@progbits,foo{{$}}
; CHECK:              .p2align        4
; CHECK-NEXT: [[IDX:.Lxray_fn_idx[0-9]+]]:
; CHECK-NEXT:         .quad .Lxray_sleds_start0-[[IDX]]
; CHECK-NEXT:         .quad 2
; CHECK-NEXT:         .text
