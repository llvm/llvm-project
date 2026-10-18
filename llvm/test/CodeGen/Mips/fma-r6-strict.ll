; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 -verify-machineinstrs | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 -verify-machineinstrs -stop-after=finalize-isel | FileCheck %s --check-prefix=MIR
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs -stop-after=finalize-isel | FileCheck %s --check-prefix=MIR
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 -verify-machineinstrs | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 -verify-machineinstrs -stop-after=finalize-isel | FileCheck %s --check-prefix=MIR
; RUN: llc < %s -mtriple=mips64el -mcpu=mips64r6 -target-abi=n64 -verify-machineinstrs -disable-strictnode-mutation | FileCheck %s --check-prefix=ASM

; Strict operations must retain their exceptions even when their results are
; unused. MIR checks also reject the nofpexcept flag.

define void @unused_fma_f32(float %a, float %b, float %c) strictfp {
; ASM-LABEL: unused_fma_f32:
; ASM: maddf.s
; MIR-LABEL: name: unused_fma_f32
; MIR: :fgr32 = MADDF_S
  %r = call float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

define void @unused_msub_f32(float %a, float %b, float %c) strictfp {
; ASM-LABEL: unused_msub_f32:
; ASM: msubf.s
; MIR-LABEL: name: unused_msub_f32
; MIR: :fgr32 = MSUBF_S
  %neg = fneg float %a
  %r = call float @llvm.experimental.constrained.fma.f32(float %neg, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

; The first FMA must raise exceptions before the observing call, and the
; second FMA must raise exceptions after it. The first FMA executes in the
; call's delay slot, before control transfers to observe.
define void @ordered_f32(float %a, float %b, float %c) strictfp {
; ASM-LABEL: ordered_f32:
; ASM: jal observe
; ASM-NEXT: maddf.s
; ASM: msubf.s
; MIR-LABEL: name: ordered_f32
; MIR: :fgr32 = MADDF_S
; MIR: JAL @observe
; MIR: :fgr32 = MSUBF_S
  %r0 = call float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  call void @observe()
  %neg = fneg float %a
  %r2 = call float @llvm.experimental.constrained.fma.f32(float %neg, float %b, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

define void @unused_fma_f64(double %a, double %b, double %c) strictfp {
; ASM-LABEL: unused_fma_f64:
; ASM: maddf.d
; MIR-LABEL: name: unused_fma_f64
; MIR: :fgr64 = MADDF_D
  %r = call double @llvm.experimental.constrained.fma.f64(double %a, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

define void @unused_msub_f64(double %a, double %b, double %c) strictfp {
; ASM-LABEL: unused_msub_f64:
; ASM: msubf.d
; MIR-LABEL: name: unused_msub_f64
; MIR: :fgr64 = MSUBF_D
  %neg = fneg double %a
  %r = call double @llvm.experimental.constrained.fma.f64(double %neg, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

; The first FMA must raise exceptions before the observing call, and the
; second FMA must raise exceptions after it. Check the call's delay slot.
define void @ordered_f64(double %a, double %b, double %c) strictfp {
; ASM-LABEL: ordered_f64:
; ASM: jal observe
; ASM-NEXT: maddf.d
; ASM: msubf.d
; MIR-LABEL: name: ordered_f64
; MIR: :fgr64 = MADDF_D
; MIR: JAL @observe
; MIR: :fgr64 = MSUBF_D
  %r0 = call double @llvm.experimental.constrained.fma.f64(double %a, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  call void @observe()
  %neg = fneg double %a
  %r2 = call double @llvm.experimental.constrained.fma.f64(double %neg, double %b, double %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret void
}

declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
declare void @observe()
