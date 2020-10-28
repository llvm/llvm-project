; RUN: llc -mtriple=aarch64-linux %s               -o - | \
; RUN:   FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-linux %s -filetype=obj -o - |  \
; RUN:   llvm-readelf -S - | FileCheck %s --check-prefix=OBJ

define dso_local i32 @f() #0 {
entry:
  ret i32 0
}

define dso_local i32 @g() #1 {
entry:
  ret i32 0
}

attributes #0 = { "sign-return-address"="non-leaf" }

attributes #1 = { "sign-return-address"="none" }

; No common attribute, no note section
; ASM-NOT: .note.gnu.property
; OBJ-NOT: .note.gnu.property
