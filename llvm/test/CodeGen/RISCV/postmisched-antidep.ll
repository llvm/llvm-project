; RUN: llc -mtriple=riscv64 -mcpu=sifive-u74 \
; RUN:   -misched-postra-break-antideps=all -o - %s \
; RUN:   | FileCheck %s --check-prefix=ALL
; RUN: llc -mtriple=riscv64 -mcpu=sifive-u74 \
; RUN:   -misched-postra-break-antideps=none -o - %s \
; RUN:   | FileCheck %s --check-prefix=NONE

define i64 @sdiv4096(i64 %x) {
; ALL-LABEL: sdiv4096:
; ALL:       # %bb.0:
; ALL-NEXT:    srai a1, a0, 63
; ALL-NEXT:    srli a2, a1, 52
; ALL-NEXT:    add a5, a0, a2
; ALL-NEXT:    srai a0, a5, 12
; ALL-NEXT:    ret
;
; NONE-LABEL: sdiv4096:
; NONE:       # %bb.0:
; NONE-NEXT:    srai a1, a0, 63
; NONE-NEXT:    srli a1, a1, 52
; NONE-NEXT:    add a0, a0, a1
; NONE-NEXT:    srai a0, a0, 12
; NONE-NEXT:    ret
  %result = sdiv i64 %x, 4096
  ret i64 %result
}
