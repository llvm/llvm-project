; RUN: llc --mtriple=loongarch64 %s -o - | FileCheck %s
; RUN: llc --mtriple=loongarch64 -filetype=obj %s -o %t
; RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: foo:
; CHECK-LABEL: .Lfunc_begin0:
; CHECK:       .p2align 2
; CHECK-LABEL: .Lxray_sled_begin0:
; CHECK-NEXT:  b .Lxray_sled_end0
; CHECK-COUNT-11:  nop
; CHECK-LABEL: .Lxray_sled_end0:
  ret i32 0
; CHECK-LABEL: .Lxray_sled_begin1:
; CHECK-NEXT:  b .Lxray_sled_end1
; CHECK-COUNT-11:  nop
; CHECK-NEXT: .Lxray_sled_end1:
; CHECK-NEXT:  ret
; CHECK-NEXT: .Lfunc_end0:
}

; CHECK-LABEL: .section xray_instr_map
; CHECK-NEXT: .Lxray_sleds_start0:
; CHECK-NEXT: [[TMP:.Ltmp[0-9]+]]:
; CHECK-NEXT: .dword .Lxray_sled_begin0-[[TMP]]
; CHECK-NEXT: .dword .Lfunc_begin0-([[TMP]]+8)
; CHECK-NEXT: .byte 0x00
; CHECK-NEXT: .byte 0x01
; CHECK-NEXT: .byte 0x02
; CHECK-NEXT: .space 13
; CHECK-NEXT: [[TMP:.Ltmp[0-9]+]]:
; CHECK-NEXT: .dword .Lxray_sled_begin1-[[TMP]]
; CHECK-NEXT: .dword .Lfunc_begin0-([[TMP]]+8)
; CHECK-NEXT: .byte 0x01
; CHECK-NEXT: .byte 0x01
; CHECK-NEXT: .byte 0x02
; CHECK-NEXT: .space 13
; CHECK-NEXT: .Lxray_sleds_end0:

; CHECK-LABEL:  .section xray_fn_idx
; CHECK:      [[IDX:.Lxray_fn_idx[0-9]+]]:
; CHECK:      .dword .Lxray_sleds_start0-[[IDX]]
; CHECK-NEXT: .dword 2

; RELOC:      Section ([[#]]) .relaxray_instr_map {
; RELOC-NEXT:   0x0 R_LARCH_64_PCREL .text 0x0
; RELOC-NEXT:   0x8 R_LARCH_64_PCREL .text 0x0
; RELOC-NEXT:   0x20 R_LARCH_64_PCREL .text 0x34
; RELOC-NEXT:   0x28 R_LARCH_64_PCREL .text 0x0
; RELOC-NEXT: }
; RELOC-NEXT: Section ([[#]]) .relaxray_fn_idx {
; RELOC-NEXT:   0x0 R_LARCH_64_PCREL xray_instr_map 0x0
; RELOC-NEXT: }
; RELOC-NEXT: Section ([[#]]) .rela.eh_frame {
; RELOC-NEXT:   0x1C R_LARCH_32_PCREL .text 0x0
; RELOC-NEXT: }
