; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s

define ptr @irg_imm16(ptr %p) {
entry:
; CHECK-LABEL: irg_imm16:
; CHECK: mov w[[R:[0-9]+]], #16
; CHECK: irg x0, x0, x[[R]]
; CHECK: ret
  %q = call ptr @llvm.aarch64.irg(ptr %p, i64 16)
  ret ptr %q
}

define ptr @irg_imm0(ptr %p) {
entry:
; CHECK-LABEL: irg_imm0:
; CHECK: irg x0, x0{{$}}
; CHECK: ret
  %q = call ptr @llvm.aarch64.irg(ptr %p, i64 0)
  ret ptr %q
}

define ptr @irg_reg(ptr %p, i64 %ex) {
entry:
; CHECK-LABEL: irg_reg:
; CHECK: irg x0, x0, x1
; CHECK: ret
  %q = call ptr @llvm.aarch64.irg(ptr %p, i64 %ex)
  ret ptr %q
}

; undef argument in irg is treated specially
define ptr @irg_sp() {
entry:
; CHECK-LABEL: irg_sp:
; CHECK: irg x0, sp{{$}}
; CHECK: ret
  %q = call ptr @llvm.aarch64.irg.sp(i64 0)
  ret ptr %q
}

declare ptr @llvm.aarch64.irg(ptr %p, i64 %exclude)
declare ptr @llvm.aarch64.irg.sp(i64 %exclude)
