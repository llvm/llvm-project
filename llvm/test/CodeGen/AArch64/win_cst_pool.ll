; RUN: llc < %s -mtriple=aarch64-win32-gnu | FileCheck -check-prefix=MINGW %s

define double @double() {
  ret double 0x0000000000800000
}
; MINGW:              .section        .rdata,"dr"
; MINGW-NEXT:         .p2align  3
; MINGW-NEXT: [[LABEL:\.LC.*]]:
; MINGW-NEXT:         .xword   8388608
; MINGW:      double:
; MINGW:               adrp    x8, [[LABEL]]
; MINGW-NEXT:          ldr     d0, [x8, [[LABEL]]]
; MINGW-NEXT:          ret
