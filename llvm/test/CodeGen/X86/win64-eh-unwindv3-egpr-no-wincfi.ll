; RUN: llc -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; An egpr leaf with no WinCFI (saves nothing) must NOT get a .seh_proc or
; .seh_unwindversion: the per-function marker is gated on MF.hasWinCFI().

; CHECK-LABEL:  leaf:
; CHECK-NOT:    .seh_proc
; CHECK-NOT:    .seh_unwindversion
; CHECK:        retq

define dso_local void @leaf() uwtable #0 {
entry:
  ret void
}

attributes #0 = { "target-features"="+egpr" }
