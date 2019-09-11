; RUN: llvm-as -o %t %s
; RUN: llvm-lto2 dump-symtab %t | FileCheck %s

; CHECK: target triple: x86_64-unknown-linux-gnu
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-NOT: linker opts:
!0 = !{!"/include:foo"}
!llvm.linker.options = !{ !0 }

; CHECK: {{^dependent libraries: \"foo\" \"b a r\" \"baz\"$}}
!1 = !{!"foo"}
!2 = !{!"b a r"}
!3 = !{!"baz"}
!llvm.dependent-libraries = !{!1, !2, !3}

@g1 = global i32 0

; CHECK-NOT: fallback g1
@g2 = weak alias i32, i32* @g1
