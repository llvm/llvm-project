; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32r6 -mattr=+soft-float < %s | FileCheck %s --check-prefix=SOFT --implicit-check-not=rint.s --implicit-check-not=rint.d
; RUN: llc -mtriple=mips64el-linux-gnu -mcpu=mips64r6 -mattr=+soft-float < %s | FileCheck %s --check-prefix=SOFT --implicit-check-not=rint.s --implicit-check-not=rint.d
; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32r6 -mattr=+single-float -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=SINGLE32 --implicit-check-not=RINT_D
; RUN: llc -mtriple=mips64el-linux-gnu -mcpu=mips64r6 -mattr=+single-float < %s | FileCheck %s --check-prefix=SINGLE --implicit-check-not=rint.d

; Soft-float must use libcalls for both types. Single-float can use RINT_S,
; but double precision must still use a libcall, including for strict rint.
define float @rint_f32(float %x) {
; SOFT-LABEL: rint_f32:
; SOFT:       jal rintf
; SINGLE-LABEL: rint_f32:
; SINGLE:       rint.s
; SINGLE32-LABEL: name: rint_f32
; SINGLE32:       nofpexcept RINT_S
  %r = call float @llvm.rint.f32(float %x)
  ret float %r
}

define double @rint_f64(double %x) {
; SOFT-LABEL: rint_f64:
; SOFT:       jal rint{{$}}
; SINGLE-LABEL: rint_f64:
; SINGLE:       jal rint{{$}}
; SINGLE32-LABEL: name: rint_f64
; SINGLE32:       JAL &rint,
  %r = call double @llvm.rint.f64(double %x)
  ret double %r
}

define float @strict_rint_f32(float %x) strictfp {
; SOFT-LABEL: strict_rint_f32:
; SOFT:       jal rintf
; SINGLE-LABEL: strict_rint_f32:
; SINGLE:       rint.s
; SINGLE32-LABEL: name: strict_rint_f32
; SINGLE32:       = RINT_S
  %r = call float @llvm.experimental.constrained.rint.f32(float %x, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret float %r
}

define double @strict_rint_f64(double %x) strictfp {
; SOFT-LABEL: strict_rint_f64:
; SOFT:       jal rint{{$}}
; SINGLE-LABEL: strict_rint_f64:
; SINGLE:       jal rint{{$}}
; SINGLE32-LABEL: name: strict_rint_f64
; SINGLE32:       JAL &rint,
  %r = call double @llvm.experimental.constrained.rint.f64(double %x, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %r
}
