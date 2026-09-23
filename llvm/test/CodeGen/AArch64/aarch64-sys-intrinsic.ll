; RUN: llc < %s -mtriple=aarch64 -asm-verbose=false | FileCheck %s --check-prefix=CHECK
; RUN: llc < %s -mtriple=aarch64 -mattr=+predres -asm-verbose=false | FileCheck %s --check-prefix=PREDRES
; RUN: llc < %s -mtriple=aarch64 -mattr=+specres2 -asm-verbose=false | FileCheck %s --check-prefix=SPECRES2
; RUN: llc < %s -mtriple=aarch64 -mattr=+gcie -asm-verbose=false | FileCheck %s --check-prefix=GCIE
; RUN: llc < %s -mtriple=aarch64 -mattr=+poe2 -asm-verbose=false | FileCheck %s --check-prefix=POE2

declare void @llvm.aarch64.sys(i32 immarg, i32 immarg, i32 immarg, i32 immarg,
                               i64)

define void @sys_random(i64 %x) {
; CHECK-LABEL: sys_random:
; CHECK:       sys #0, c7, c10, #6, x0
; CHECK-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 0, i32 7, i32 10, i32 6, i64 %x)
  ret void
}

define void @sys_ic_iallu() {
; CHECK-LABEL: sys_ic_iallu:
; CHECK:       ic iallu
; CHECK-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 0, i32 7, i32 5, i32 0, i64 0)
  ret void
}

define void @sys_dc_cvac(i64 %x) {
; CHECK-LABEL: sys_dc_cvac:
; CHECK:       dc cvac, x0
; CHECK-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 3, i32 7, i32 10, i32 1, i64 %x)
  ret void
}

define void @sys_at_s1e2w(i64 %x) {
; CHECK-LABEL: sys_at_s1e2w:
; CHECK:       at s1e2w, x0
; CHECK-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 4, i32 7, i32 8, i32 1, i64 %x)
  ret void
}

define void @sys_tlbi_vmalle1() {
; CHECK-LABEL: sys_tlbi_vmalle1:
; CHECK:       tlbi vmalle1
; CHECK-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 0, i32 8, i32 7, i32 0, i64 0)
  ret void
}

define void @sys_cfp_rctx(i64 %x) {
; PREDRES-LABEL: sys_cfp_rctx:
; PREDRES:       cfp rctx, x0
; PREDRES-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 3, i32 7, i32 3, i32 4, i64 %x)
  ret void
}

define void @sys_dvp_rctx(i64 %x) {
; PREDRES-LABEL: sys_dvp_rctx:
; PREDRES:       dvp rctx, x0
; PREDRES-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 3, i32 7, i32 3, i32 5, i64 %x)
  ret void
}

define void @sys_cpp_rctx(i64 %x) {
; PREDRES-LABEL: sys_cpp_rctx:
; PREDRES:       cpp rctx, x0
; PREDRES-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 3, i32 7, i32 3, i32 7, i64 %x)
  ret void
}

define void @sys_cosp_rctx(i64 %x) {
; SPECRES2-LABEL: sys_cosp_rctx:
; SPECRES2:       cosp rctx, x0
; SPECRES2-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 3, i32 7, i32 3, i32 6, i64 %x)
  ret void
}

define void @sys_gic_cdaff(i64 %x) {
; GCIE-LABEL: sys_gic_cdaff:
; GCIE:       gic cdaff, x0
; GCIE-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 0, i32 12, i32 1, i32 3, i64 %x)
  ret void
}

define void @sys_gsb_sys() {
; GCIE-LABEL: sys_gsb_sys:
; GCIE:       gsb sys
; GCIE-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 0, i32 12, i32 0, i32 0, i64 0)
  ret void
}

define void @sys_plbi_vmalle1() {
; POE2-LABEL: sys_plbi_vmalle1:
; POE2:       plbi vmalle1
; POE2-NEXT:  ret
entry:
  call void @llvm.aarch64.sys(i32 0, i32 10, i32 7, i32 0, i64 0)
  ret void
}
