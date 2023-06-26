; RUN: llc -mtriple arm64e-apple-darwin -o - %s 2> %t | FileCheck %s
; RUN: FileCheck < %t %s --check-prefix=WARNING

; WARNING: warning: incompatible ptrauth ABI versions: 5 (user) and 2 (user), falling back to 63 (user)
; CHECK: .ptrauth_abi_version 63

!0 = !{ i32 5, i1 false }
!1 = !{ i32 2, i1 false }
!2 = !{ !0, !1 }
!3 = !{ i32 6, !"ptrauth.abi-version", !2 }
!llvm.module.flags = !{ !3 }
