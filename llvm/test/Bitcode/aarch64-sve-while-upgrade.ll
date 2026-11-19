; RUN: opt -S -passes=verify < %s | FileCheck %s
; RUN: llvm-dis < %S/Inputs/aarch64-sve-while-upgrade.bc | FileCheck %s

define <vscale x 16 x i1> @whilerw_b(ptr %a, ptr %b) {
; CHECK-LABEL: @whilerw_b(
; CHECK: call <vscale x 16 x i1> @llvm.aarch64.sve.whilerw.nxv16i1.p0(ptr %a, ptr %b)
  %r = call <vscale x 16 x i1> @llvm.aarch64.sve.whilerw.b.nxv16i1.p0(ptr %a, ptr %b)
  ret <vscale x 16 x i1> %r
}

define <vscale x 8 x i1> @whilerw_h(ptr %a, ptr %b) {
; CHECK-LABEL: @whilerw_h(
; CHECK: call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.nxv8i1.p0(ptr %a, ptr %b)
  %r = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nxv8i1.p0(ptr %a, ptr %b)
  ret <vscale x 8 x i1> %r
}

define <vscale x 4 x i1> @whilerw_s(ptr %a, ptr %b) {
; CHECK-LABEL: @whilerw_s(
; CHECK: call <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.nxv4i1.p0(ptr %a, ptr %b)
  %r = call <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nxv4i1.p0(ptr %a, ptr %b)
  ret <vscale x 4 x i1> %r
}

define <vscale x 2 x i1> @whilerw_d(ptr %a, ptr %b) {
; CHECK-LABEL: @whilerw_d(
; CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.nxv2i1.p0(ptr %a, ptr %b)
  %r = call <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nxv2i1.p0(ptr %a, ptr %b)
  ret <vscale x 2 x i1> %r
}

define <vscale x 16 x i1> @whilewr_b(ptr %a, ptr %b) {
; CHECK-LABEL: @whilewr_b(
; CHECK: call <vscale x 16 x i1> @llvm.aarch64.sve.whilewr.nxv16i1.p0(ptr %a, ptr %b)
  %r = call <vscale x 16 x i1> @llvm.aarch64.sve.whilewr.b.nxv16i1.p0(ptr %a, ptr %b)
  ret <vscale x 16 x i1> %r
}

define <vscale x 8 x i1> @whilewr_h(ptr %a, ptr %b) {
; CHECK-LABEL: @whilewr_h(
; CHECK: call <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.nxv8i1.p0(ptr %a, ptr %b)
  %r = call <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.h.nxv8i1.p0(ptr %a, ptr %b)
  ret <vscale x 8 x i1> %r
}

define <vscale x 4 x i1> @whilewr_s(ptr %a, ptr %b) {
; CHECK-LABEL: @whilewr_s(
; CHECK: call <vscale x 4 x i1> @llvm.aarch64.sve.whilewr.nxv4i1.p0(ptr %a, ptr %b)
  %r = call <vscale x 4 x i1> @llvm.aarch64.sve.whilewr.s.nxv4i1.p0(ptr %a, ptr %b)
  ret <vscale x 4 x i1> %r
}

define <vscale x 2 x i1> @whilewr_d(ptr %a, ptr %b) {
; CHECK-LABEL: @whilewr_d(
; CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.whilewr.nxv2i1.p0(ptr %a, ptr %b)
  %r = call <vscale x 2 x i1> @llvm.aarch64.sve.whilewr.d.nxv2i1.p0(ptr %a, ptr %b)
  ret <vscale x 2 x i1> %r
}

define <vscale x 8 x i1> @whilerw_addrspace1(ptr addrspace(1) %a,
                                               ptr addrspace(1) %b) {
; CHECK-LABEL: @whilerw_addrspace1(
; CHECK: call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.nxv8i1.p1(ptr addrspace(1) %a, ptr addrspace(1) %b)
  %r = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nxv8i1.p1(ptr addrspace(1) %a, ptr addrspace(1) %b)
  ret <vscale x 8 x i1> %r
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.whilerw.b.nxv16i1.p0(ptr, ptr)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nxv8i1.p0(ptr, ptr)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nxv4i1.p0(ptr, ptr)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nxv2i1.p0(ptr, ptr)
declare <vscale x 16 x i1> @llvm.aarch64.sve.whilewr.b.nxv16i1.p0(ptr, ptr)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilewr.h.nxv8i1.p0(ptr, ptr)
declare <vscale x 4 x i1> @llvm.aarch64.sve.whilewr.s.nxv4i1.p0(ptr, ptr)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilewr.d.nxv2i1.p0(ptr, ptr)
declare <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nxv8i1.p1(ptr addrspace(1), ptr addrspace(1))
