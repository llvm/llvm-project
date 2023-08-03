; RUN: llc -mtriple arm64e-apple-darwin -o - %s | FileCheck %s

; CHECK: .ptrauth_abi_version 0

!0 = !{ i32 0, i1 false }
!1 = !{ !0 }
!2 = !{ i32 6, !"ptrauth.abi-version", !1 }
!llvm.module.flags = !{ !2 }
