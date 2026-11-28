; RUN: llc -mtriple=riscv64 -mattr=+zve32f,+zvfhmin,+zfhmin,+d -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,HARD
; RUN: llc -mtriple=riscv64 -mattr=+zve32f,+zvfhmin,+zfhmin,-d -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,SOFT
; RUN: llc -mtriple=riscv32 -mattr=+zve32f,+zvl64b,+zvfhmin,+d -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,HARD
; RUN: llc -mtriple=riscv32 -mattr=+zve32f,+zvfhmin,-d -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,SOFT
; RUN: llc -enable-new-pm -mtriple=riscv64 -mattr=+zve32f,+zvl64b,+zvfhmin,+d -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,HARD

; A vector f32 FMA followed by ordinary narrowing can round twice. Check
; that the fallback computes the fused result in scalar double precision.
; CHECK-LABEL: fixed:
; CHECK-NOT: vfmadd
; CHECK-NOT: vfmacc
; HARD: fmadd.d
; SOFT: call fma
; CHECK: ret
; CHECK-LABEL: scalable:
; CHECK-NOT: vfmadd
; CHECK-NOT: vfmacc
; HARD: fmadd.d
; SOFT: call fma
; CHECK: ret
; CHECK-LABEL: fixed_strict:
; CHECK-NOT: vfmadd
; CHECK-NOT: vfmacc
; HARD: fmadd.d
; SOFT: call fma
; CHECK: ret
; CHECK-LABEL: scalable_strict:
; CHECK-NOT: vfmadd
; CHECK-NOT: vfmacc
; HARD: fmadd.d
; SOFT: call fma
; CHECK: ret

define void @fixed(ptr %out, ptr %x, ptr %y, ptr %z) {
  %a = load <4 x half>, ptr %x, align 2
  %b = load <4 x half>, ptr %y, align 2
  %c = load <4 x half>, ptr %z, align 2
  %r = call <4 x half> @llvm.fma.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  store <4 x half> %r, ptr %out, align 2
  ret void
}

define <vscale x 4 x half> @scalable(<vscale x 4 x half> %a, <vscale x 4 x half> %b, <vscale x 4 x half> %c) {
  %r = call <vscale x 4 x half> @llvm.fma.nxv4f16(<vscale x 4 x half> %a, <vscale x 4 x half> %b, <vscale x 4 x half> %c)
  ret <vscale x 4 x half> %r
}

define void @fixed_strict(ptr %out, ptr %x, ptr %y, ptr %z) strictfp {
  %a = load <4 x half>, ptr %x, align 2
  %b = load <4 x half>, ptr %y, align 2
  %c = load <4 x half>, ptr %z, align 2
  %r = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  store <4 x half> %r, ptr %out, align 2
  ret void
}

define <vscale x 4 x half> @scalable_strict(<vscale x 4 x half> %a, <vscale x 4 x half> %b, <vscale x 4 x half> %c) noinline optnone strictfp {
  %r = call <vscale x 4 x half> @llvm.experimental.constrained.fma.nxv4f16(<vscale x 4 x half> %a, <vscale x 4 x half> %b, <vscale x 4 x half> %c, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret <vscale x 4 x half> %r
}
