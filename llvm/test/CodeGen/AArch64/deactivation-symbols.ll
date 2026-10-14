; RUN: llc < %s -O0 -mtriple=aarch64-none-linux-gnu -mattr=+pauth | FileCheck --check-prefixes=CHECK,O0 %s
; RUN: llc < %s -O2 -mtriple=aarch64-none-linux-gnu -mattr=+pauth | FileCheck --check-prefixes=CHECK,O2 %s

@ds = external global i8

declare void @f(ptr %p)

; CHECK: call:
define void @call(ptr %p) {
  ; CHECK: [[LABEL:.L.*]]:
  ; CHECK-NEXT: .reloc [[LABEL]], R_AARCH64_PATCHINST, ds
  ; CHECK-NEXT: bl f
  notail call void @f(ptr %p) [ "deactivation-symbol"(ptr @ds) ]
  ret void
}

; CHECK: pauth_sign_zero:
define i64 @pauth_sign_zero(i64 %p) {
  ; O0: mov x8, xzr
  ; CHECK: [[LABEL:.L.*]]:
  ; CHECK-NEXT: .reloc [[LABEL]], R_AARCH64_PATCHINST, ds
  ; O0-NEXT: pacia x0, x8
  ; O2-NEXT: paciza x0
  %signed = call i64 @llvm.ptrauth.sign(i64 %p, i32 0, i64 0) [ "deactivation-symbol"(ptr @ds) ]
  ret i64 %signed
}

; CHECK: pauth_sign_const:
define i64 @pauth_sign_const(i64 %p) {
  ; CHECK: mov x16, #12345
  ; CHECK-NEXT: [[LABEL:.L.*]]:
  ; CHECK-NEXT: .reloc [[LABEL]], R_AARCH64_PATCHINST, ds
  ; CHECK-NEXT: pacia x0, x16
  %signed = call i64 @llvm.ptrauth.sign(i64 %p, i32 0, i64 12345) [ "deactivation-symbol"(ptr @ds) ]
  ret i64 %signed
}

; CHECK: pauth_sign:
define i64 @pauth_sign(i64 %p, i64 %d) {
  ; CHECK: [[LABEL:.L.*]]:
  ; CHECK-NEXT: .reloc [[LABEL]], R_AARCH64_PATCHINST, ds
  ; CHECK-NEXT: pacia x0, x1
  %signed = call i64 @llvm.ptrauth.sign(i64 %p, i32 0, i64 %d) [ "deactivation-symbol"(ptr @ds) ]
  ret i64 %signed
}

; CHECK: pauth_auth_zero:
define i64 @pauth_auth_zero(i64 %p) {
  ; CHECK: [[LABEL:.L.*]]:
  ; CHECK-NEXT: .reloc [[LABEL]], R_AARCH64_PATCHINST, ds
  ; CHECK-NEXT: autiza x0
  %authed = call i64 @llvm.ptrauth.auth(i64 %p, i32 0, i64 0) [ "deactivation-symbol"(ptr @ds) ]
  ret i64 %authed
}

; CHECK: pauth_auth_const:
define i64 @pauth_auth_const(i64 %p) {
  ; CHECK: mov x8, #12345
  ; CHECK-NEXT: [[LABEL:.L.*]]:
  ; CHECK-NEXT: .reloc [[LABEL]], R_AARCH64_PATCHINST, ds
  ; CHECK-NEXT: autia x0, x8
  %authed = call i64 @llvm.ptrauth.auth(i64 %p, i32 0, i64 12345) [ "deactivation-symbol"(ptr @ds) ]
  ret i64 %authed
}

; CHECK: pauth_auth:
define i64 @pauth_auth(i64 %p, i64 %d) {
  ; CHECK: [[LABEL:.L.*]]:
  ; CHECK-NEXT: .reloc [[LABEL]], R_AARCH64_PATCHINST, ds
  ; CHECK-NEXT: autia x0, x1
  %authed = call i64 @llvm.ptrauth.auth(i64 %p, i32 0, i64 %d) [ "deactivation-symbol"(ptr @ds) ]
  ret i64 %authed
}
