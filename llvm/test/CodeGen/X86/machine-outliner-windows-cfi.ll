; RUN: llc -enable-machine-outliner -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s
; RUN: llc -enable-machine-outliner -mtriple=x86_64-apple-darwin < %s | FileCheck %s --check-prefix=DARWIN

; X86 previously outlined from Windows-CFI functions, which can corrupt .seh_*
; frame state around WinEH/funclets (#213862). Mirror AArch64 and refuse
; outlining when usesWindowsCFI() is true.

define i32 @f1(i32 %a, i32 %b) nounwind {
entry:
  %a2 = mul i32 %a, %b
  %a3 = add i32 %a2, %b
  %a4 = mul i32 %a3, %a2
  %a5 = add i32 %a4, %a3
  ret i32 %a5
}

define i32 @f2(i32 %a, i32 %b) nounwind {
entry:
  %a2 = mul i32 %a, %b
  %a3 = add i32 %a2, %b
  %a4 = mul i32 %a3, %a2
  %a5 = add i32 %a4, %a3
  ret i32 %a5
}

; CHECK-LABEL: f1:
; CHECK-NOT: OUTLINED_FUNCTION
; CHECK: imull
; CHECK-LABEL: f2:
; CHECK-NOT: OUTLINED_FUNCTION
; CHECK: imull

; DARWIN-LABEL: _f1:
; DARWIN: OUTLINED_FUNCTION
; DARWIN-LABEL: _f2:
; DARWIN: OUTLINED_FUNCTION
