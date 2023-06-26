; RUN: llc -mtriple arm64e-apple-darwin -o - %s | FileCheck %s

; CHECK-NOT: .ptrauth_abi_version
