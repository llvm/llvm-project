; RUN: llc -O0 --mtriple=loongarch32 -global-isel -stop-after=irtranslator -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=LA32
; RUN: llc -O0 --mtriple=loongarch64 -global-isel -stop-after=irtranslator -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=LA64

define void @foo() {
  ; LA32-LABEL: name: foo
  ; LA32: bb.1.entry:
  ; LA32-NEXT: PseudoRET

  ; LA64-LABEL: name: foo
  ; LA64: bb.1.entry:
  ; LA64-NEXT: PseudoRET
entry:
  ret void
}
