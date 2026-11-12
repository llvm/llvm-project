; RUN: llc -mtriple=sparcv9 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=V9
; RUN: llc -mtriple=sparc -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=V8

; The quad routines take their operands and result by pointer, in stack
; objects aligned as the ABI aligns long double: 16 bytes on SPARC V9, and 8
; on 32-bit SPARC.

define fp128 @add(fp128 %a, fp128 %b) {
; V9-LABEL: name: add
; V9:       stack:
; V9-COUNT-3: size: 16, alignment: 16
; V8-LABEL: name: add
; V8:       stack:
; V8-COUNT-3: size: 16, alignment: 8
  %r = fadd fp128 %a, %b
  ret fp128 %r
}
