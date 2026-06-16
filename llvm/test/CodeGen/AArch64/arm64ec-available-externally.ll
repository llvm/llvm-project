; RUN: llc -mtriple arm64ec-windows-msvc -o - %s | FileCheck %s

; Arm64EC Regression Test: The Arm64EC Call Lowering was placing "available
; externally" items in COMDATs, which is not permitted by the module verifier.

define available_externally float @f() {
entry:
  ret float 0x0
}

define i32 @caller() {
entry:
  call float @f()
  ret i32 0
}

; Normal function gets an entry thunk, but not an exit thunk.
; CHECK-DAG:    $ientry_thunk$cdecl$i8$v:
; CHECK-NOT:    $iexit_thunk$cdecl$i8$v:

; Available Externally function gets an exit thunk, but not an entry thunk.
; CHECK-DAG:    $iexit_thunk$cdecl$f$v:
; CHECK-DAG:    "#f$exit_thunk":
; CHECK-NOT:    $ientry_thunk$cdecl$f$v:
