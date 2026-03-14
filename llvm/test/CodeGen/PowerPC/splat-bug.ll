; RUN: llc -verify-machineinstrs -mcpu=ppc64 -O0 -fast-isel=false < %s | FileCheck %s

; Checks for a previous bug where vspltisb/vaddubm were issued in place
; of vsplitsh/vadduhm.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@a = external global <16 x i8>

define void @foo() nounwind ssp {
; CHECK: foo:
  store <16 x i8> <i8 0, i8 16, i8 0, i8 16, i8 0, i8 16, i8 0, i8 16, i8 0, i8 16, i8 0, i8 16, i8 0, i8 16, i8 0, i8 16>, ptr @a
; CHECK: vspltish [[REG:[0-9]+]], 8
; CHECK: vadduhm {{[0-9]+}}, [[REG]], [[REG]]
  ret void
}

