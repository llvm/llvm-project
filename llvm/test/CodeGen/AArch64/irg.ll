; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s

define i8* @irg_imm16(i8* %p) {
entry:
; CHECK-LABEL: irg_imm16:
; CHECK: mov w[[R:[0-9]+]], #16
; CHECK: irg x0, x0, x[[R]]
; CHECK: ret
  %q = call i8* @llvm.aarch64.irg(i8* %p, i64 16)
  ret i8* %q
}

define i8* @irg_imm0(i8* %p) {
entry:
; CHECK-LABEL: irg_imm0:
; CHECK: irg x0, x0{{$}}
; CHECK: ret
  %q = call i8* @llvm.aarch64.irg(i8* %p, i64 0)
  ret i8* %q
}

define i8* @irg_reg(i8* %p, i64 %ex) {
entry:
; CHECK-LABEL: irg_reg:
; CHECK: irg x0, x0, x1
; CHECK: ret
  %q = call i8* @llvm.aarch64.irg(i8* %p, i64 %ex)
  ret i8* %q
}

; undef argument in irg is treated specially
define i8* @irg_sp() {
entry:
; CHECK-LABEL: irg_sp:
; CHECK: irg x0, sp{{$}}
; CHECK: ret
  %q = call i8* @llvm.aarch64.irg.sp(i64 0)
  ret i8* %q
}

declare i8* @llvm.aarch64.irg(i8* %p, i64 %exclude)
declare i8* @llvm.aarch64.irg.sp(i64 %exclude)
