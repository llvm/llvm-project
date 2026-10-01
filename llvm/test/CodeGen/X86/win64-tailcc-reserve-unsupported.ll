; RUN: sed 's/UNWIND_MODE/2/' %s | not --crash llc -mtriple=x86_64-pc-windows-msvc 2>&1 | FileCheck %s --check-prefix=V2
; RUN: sed 's/UNWIND_MODE/3/' %s | not --crash llc -mtriple=x86_64-pc-windows-msvc 2>&1 | FileCheck %s --check-prefix=V3
; RUN: sed 's/UNWIND_MODE/0/' %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=V1

; The reserve is only described by classic (V1) unwind info, and is not handled
; in functions with EH funclets.

; V2: Can't handle guaranteed tail calls that change the stack argument size with Windows x64 unwind v2 or v3 yet
; V3: Can't handle guaranteed tail calls that change the stack argument size with Windows x64 unwind v2 or v3 yet
; V1: reserve:

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define tailcc void @reserve(i64 %a, i64 %b) {
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 UNWIND_MODE}
