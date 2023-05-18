; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

; Test that we don't abort fast-isle for ret
define <8 x i8> @ret_v8i8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: ret_v8i8
; CHECK:       add.8b v0, v0, v1
  %1 = add <8 x i8> %a, %b
  ret <8 x i8> %1
}
