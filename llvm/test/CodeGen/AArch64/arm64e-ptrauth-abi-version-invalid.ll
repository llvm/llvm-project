; RUN: not llc -mtriple arm64e-apple-darwin -o - %s 2> %t | FileCheck %s
; RUN: FileCheck < %t %s --check-prefix=ERROR

; ERROR: error: invalid ptrauth ABI version: 64
; CHECK: .ptrauth_abi_version 63

!0 = !{ i32 64, i1 false }
!1 = !{ !0 }
!2 = !{ i32 6, !"ptrauth.abi-version", !1 }
!llvm.module.flags = !{ !2 }
