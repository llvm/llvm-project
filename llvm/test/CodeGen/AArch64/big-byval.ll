; RUN: llc -o - %s -verify-machineinstrs | FileCheck %s
target triple = "aarch64--"

; Make sure we don't fail machine verification because the memcpy callframe
; setup is nested inside the extfunc callframe setup.
; CHECK-LABEL: func:
; CHECK: bl memcpy
; CHECK: bl extfunc
declare void @extfunc([4096 x i64]* byval %p)
define void @func([4096 x i64]* %z) {
  call void @extfunc([4096 x i64]* byval %z)
  ret void
}
