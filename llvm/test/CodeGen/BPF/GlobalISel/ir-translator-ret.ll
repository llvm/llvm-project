; RUN: llc -mtriple=bpfel -global-isel -verify-machineinstrs -stop-after=irtranslator < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -global-isel -verify-machineinstrs < %s | FileCheck --check-prefix=ISEL %s

; CHECK: name: f
; CHECK: RET
define void @f() {
; ISEL-LABEL: f:
; ISEL:       # %bb.0:
; ISEL-NEXT:  exit
; ISEL-NEXT: .Lfunc_end0:
  ret void
}
