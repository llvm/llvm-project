; RUN: llc -mtriple=x86_64-- -O2 -enable-machine-outliner -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: asserts

; Confirm that passes that can add CFI instructions run before CFI instruction inserter.

; CHECK-LABEL: Pass Arguments:
; CHECK:       Check CFA info and insert CFI instructions if needed
; CHECK-NOT:   X86 Optimize Call Frame
; CHECK-NOT:   Prologue/Epilogue Insertion & Frame Finalization
; CHECK-NOT:   Machine Outliner

define void @f() {
  ret void
}
