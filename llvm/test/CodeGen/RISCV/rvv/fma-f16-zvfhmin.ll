; RUN: llc -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -target-abi=lp64d \
; RUN:   -mattr=+v,+d,+zfhmin,+zvfhmin -verify-machineinstrs < %s | FileCheck %s
;
; Exact f16 FMA needs enough intermediate precision to avoid double rounding.

target triple = "riscv64-unknown-linux-gnu"

define void @fma_v8f16(ptr noalias nocapture writeonly %out,
                       ptr noalias nocapture readonly %x,
                       ptr noalias nocapture readonly %y,
                       ptr noalias nocapture readonly %z) {
; CHECK-LABEL: fma_v8f16:
; CHECK-NOT:     vfmadd.vv
; CHECK:         vfwmacc.vv
; CHECK-NOT:     vfmadd.vv
; CHECK:         vfncvt.rod.f.f.w
; CHECK:         vfncvt.f.f.w
entry:
  %vx = load <8 x half>, ptr %x, align 2
  %vy = load <8 x half>, ptr %y, align 2
  %vz = load <8 x half>, ptr %z, align 2
  %vr = call <8 x half> @llvm.fma.v8f16(<8 x half> %vx,
                                        <8 x half> %vy,
                                        <8 x half> %vz)
  store <8 x half> %vr, ptr %out, align 2
  ret void
}

declare <8 x half> @llvm.fma.v8f16(<8 x half>, <8 x half>, <8 x half>)

declare <8 x half> @llvm.experimental.constrained.fma.v8f16(
    <8 x half>, <8 x half>, <8 x half>, metadata, metadata)

define void @strict_fma_v8f16(ptr noalias nocapture writeonly %out,
                              ptr noalias nocapture readonly %x,
                              ptr noalias nocapture readonly %y,
                              ptr noalias nocapture readonly %z) strictfp {
; CHECK-LABEL: strict_fma_v8f16:
; CHECK-NOT:     vfmadd.vv
; CHECK:         vsetvli {{.*}}e64
; CHECK:         vfmadd.vv
; CHECK:         vfncvt.rod.f.f.w
; CHECK:         vfncvt.f.f.w
entry:
  %vx = load <8 x half>, ptr %x, align 2
  %vy = load <8 x half>, ptr %y, align 2
  %vz = load <8 x half>, ptr %z, align 2
  %vr = call <8 x half> @llvm.experimental.constrained.fma.v8f16(
      <8 x half> %vx, <8 x half> %vy, <8 x half> %vz,
      metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  store <8 x half> %vr, ptr %out, align 2
  ret void
}
