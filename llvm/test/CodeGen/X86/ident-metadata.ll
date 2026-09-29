; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s
; Verify that llvm.ident metadata is emitted as .ident
; directives in assembly files, and in the .comment section in ELF object files.

; CHECK: .ident  "clang version x.x"
; CHECK-NEXT: .ident  "something else"
!llvm.ident = !{!0, !1}
!0 = !{!"clang version x.x"}
!1 = !{!"something else"}
