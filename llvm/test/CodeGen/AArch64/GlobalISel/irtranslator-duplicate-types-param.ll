; RUN: llc -O0 -o - -verify-machineinstrs %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Check we don't crash due to encountering the same struct param type twice.
; CHECK-LABEL: param_two_struct
; CHECK: add
; CHECK: ret
define i64 @param_two_struct([2 x i64] %t.coerce, [2 x i64] %s.coerce) {
entry:
  %t.coerce.fca.0.extract = extractvalue [2 x i64] %t.coerce, 0
  %s.coerce.fca.1.extract = extractvalue [2 x i64] %s.coerce, 1
  %add = add nsw i64 %s.coerce.fca.1.extract, %t.coerce.fca.0.extract
  ret i64 %add
}
