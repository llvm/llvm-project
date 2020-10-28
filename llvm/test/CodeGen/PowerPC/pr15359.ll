; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr7 -filetype=obj %s -o - | \
; RUN: llvm-readobj --symbols - | FileCheck %s

target datalayout = "E-p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@nextIdx = external thread_local global i32

define fastcc void @func() nounwind {
entry:
  store i32 42, i32* @nextIdx
  ret void
}

; Verify that nextIdx has symbol type TLS.
;
; CHECK:      Symbol {
; CHECK:        Name: nextIdx
; CHECK-NEXT:   Value:
; CHECK-NEXT:   Size:
; CHECK-NEXT:   Binding:
; CHECK-NEXT:   Type: TLS
