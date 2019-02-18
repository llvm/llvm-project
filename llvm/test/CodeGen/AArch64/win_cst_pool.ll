; RUN: llc < %s -mtriple=aarch64-win32-msvc | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-win32-gnu | FileCheck -check-prefix=MINGW %s

define double @double() {
  ret double 0x0000000000800000
}
; CHECK:              .globl  __real@0000000000800000
; CHECK-NEXT:         .section        .rdata,"dr",discard,__real@0000000000800000
; CHECK-NEXT:         .p2align  3
; CHECK-NEXT: __real@0000000000800000:
; CHECK-NEXT:         .xword   8388608
; CHECK:      double:
; CHECK:               adrp    x8, __real@0000000000800000
; CHECK-NEXT:          ldr     d0, [x8, __real@0000000000800000]
; CHECK-NEXT:          ret

; MINGW:              .section        .rdata,"dr"
; MINGW-NEXT:         .p2align  3
; MINGW-NEXT: [[LABEL:\.LC.*]]:
; MINGW-NEXT:         .xword   8388608
; MINGW:      double:
; MINGW:               adrp    x8, [[LABEL]]
; MINGW-NEXT:          ldr     d0, [x8, [[LABEL]]]
; MINGW-NEXT:          ret
