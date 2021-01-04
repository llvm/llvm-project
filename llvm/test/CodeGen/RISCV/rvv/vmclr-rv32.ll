; RUN: llc -mtriple=riscv32 -mattr=+experimental-v,+experimental-zfh -verify-machineinstrs \
; RUN:   < %s | FileCheck %s
declare <vscale x 1 x i1> @llvm.riscv.vmclr.nxv1i1(
  i32);

define <vscale x 1 x i1> @intrinsic_vmclr_m_pseudo_nxv1i1(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmclr_m_pseudo_nxv1i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf8
; CHECK:       vmclr.m {{v[0-9]+}}
  %a = call <vscale x 1 x i1> @llvm.riscv.vmclr.nxv1i1(
    i32 %0)

  ret <vscale x 1 x i1> %a
}

declare <vscale x 2 x i1> @llvm.riscv.vmclr.nxv2i1(
  i32);

define <vscale x 2 x i1> @intrinsic_vmclr_m_pseudo_nxv2i1(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmclr_m_pseudo_nxv2i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf4
; CHECK:       vmclr.m {{v[0-9]+}}
  %a = call <vscale x 2 x i1> @llvm.riscv.vmclr.nxv2i1(
    i32 %0)

  ret <vscale x 2 x i1> %a
}

declare <vscale x 4 x i1> @llvm.riscv.vmclr.nxv4i1(
  i32);

define <vscale x 4 x i1> @intrinsic_vmclr_m_pseudo_nxv4i1(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmclr_m_pseudo_nxv4i1
; CHECK:       vsetvli {{.*}}, a0, e8,mf2
; CHECK:       vmclr.m {{v[0-9]+}}
  %a = call <vscale x 4 x i1> @llvm.riscv.vmclr.nxv4i1(
    i32 %0)

  ret <vscale x 4 x i1> %a
}

declare <vscale x 8 x i1> @llvm.riscv.vmclr.nxv8i1(
  i32);

define <vscale x 8 x i1> @intrinsic_vmclr_m_pseudo_nxv8i1(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmclr_m_pseudo_nxv8i1
; CHECK:       vsetvli {{.*}}, a0, e8,m1
; CHECK:       vmclr.m {{v[0-9]+}}
  %a = call <vscale x 8 x i1> @llvm.riscv.vmclr.nxv8i1(
    i32 %0)

  ret <vscale x 8 x i1> %a
}

declare <vscale x 16 x i1> @llvm.riscv.vmclr.nxv16i1(
  i32);

define <vscale x 16 x i1> @intrinsic_vmclr_m_pseudo_nxv16i1(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmclr_m_pseudo_nxv16i1
; CHECK:       vsetvli {{.*}}, a0, e8,m2
; CHECK:       vmclr.m {{v[0-9]+}}
  %a = call <vscale x 16 x i1> @llvm.riscv.vmclr.nxv16i1(
    i32 %0)

  ret <vscale x 16 x i1> %a
}

declare <vscale x 32 x i1> @llvm.riscv.vmclr.nxv32i1(
  i32);

define <vscale x 32 x i1> @intrinsic_vmclr_m_pseudo_nxv32i1(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmclr_m_pseudo_nxv32i1
; CHECK:       vsetvli {{.*}}, a0, e8,m4
; CHECK:       vmclr.m {{v[0-9]+}}
  %a = call <vscale x 32 x i1> @llvm.riscv.vmclr.nxv32i1(
    i32 %0)

  ret <vscale x 32 x i1> %a
}

declare <vscale x 64 x i1> @llvm.riscv.vmclr.nxv64i1(
  i32);

define <vscale x 64 x i1> @intrinsic_vmclr_m_pseudo_nxv64i1(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmclr_m_pseudo_nxv64i1
; CHECK:       vsetvli {{.*}}, a0, e8,m8
; CHECK:       vmclr.m {{v[0-9]+}}
  %a = call <vscale x 64 x i1> @llvm.riscv.vmclr.nxv64i1(
    i32 %0)

  ret <vscale x 64 x i1> %a
}
