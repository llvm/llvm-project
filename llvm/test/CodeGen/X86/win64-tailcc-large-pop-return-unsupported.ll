; RUN: sed 's/UNWIND_MODE/2/' %s | not --crash llc -mtriple=x86_64-pc-windows-msvc 2>&1 | FileCheck %s

; The moved return address is only described by classic (V1) unwind info.

; CHECK: Can't handle a return that pops more than 65535 bytes with Windows x64 unwind v2 or v3 yet

define tailcc void @big([9000 x i64] %a) {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 UNWIND_MODE}
