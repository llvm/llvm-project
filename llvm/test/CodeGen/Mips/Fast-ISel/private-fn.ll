; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 < %s | FileCheck %s

define private void @priv() {
  ret void
}

define ptr @addr() {
; CHECK-LABEL: addr:
; CHECK:       lw $[[R:[0-9]+]], %got($priv)(${{[0-9]+}})
; CHECK-NEXT:  addiu ${{[0-9]+}}, $[[R]], %lo($priv)
  ret ptr @priv
}

define void @call() {
; CHECK-LABEL: call:
; CHECK:       lw $[[R:[0-9]+]], %got($priv)(${{[0-9]+}})
; CHECK-NEXT:  addiu $25, $[[R]], %lo($priv)
; CHECK-NEXT:  .reloc {{.*}}, R_MIPS_JALR, priv
  call void @priv()
  ret void
}
