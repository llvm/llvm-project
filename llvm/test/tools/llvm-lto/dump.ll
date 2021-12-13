; RUN: llvm-as < %s > %t1
; RUN: llvm-lto --dump-linked-module %t1 2>&1 | FileCheck %s

; CHEKCK: target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: define void @f() {
define void @f() {
; CHECK-NEXT: entry:
entry:
; CHECK-NEXT: ret void
  ret void
; CHECK-NEXT: }
}
