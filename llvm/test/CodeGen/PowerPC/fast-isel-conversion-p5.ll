; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr5 | FileCheck %s --check-prefix=ELF64

; Test sitofp

define void @sitofp_double_i32(i32 %a, double %b) nounwind ssp {
entry:
; ELF64: sitofp_double_i32
  %b.addr = alloca double, align 8
  %conv = sitofp i32 %a to double
; ELF64: std {{[0-9]+}}, -[[OFFSET:[0-9]+]](1)
; ELF64: lfd {{[0-9]+}}, -[[OFFSET]](1)
; ELF64: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @sitofp_double_i64(i64 %a, double %b) nounwind ssp {
entry:
; ELF64: sitofp_double_i64
  %b.addr = alloca double, align 8
  %conv = sitofp i64 %a to double
; ELF64: std {{[0-9]+}}, -[[OFFSET:[0-9]+]](1)
; ELF64: lfd {{[0-9]+}}, -[[OFFSET]](1)
; ELF64: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @sitofp_double_i16(i16 %a, double %b) nounwind ssp {
entry:
; ELF64: sitofp_double_i16
  %b.addr = alloca double, align 8
  %conv = sitofp i16 %a to double
; ELF64: extsh
; ELF64: std {{[0-9]+}}, -[[OFFSET:[0-9]+]](1)
; ELF64: lfd {{[0-9]+}}, -[[OFFSET]](1)
; ELF64: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @sitofp_double_i8(i8 %a, double %b) nounwind ssp {
entry:
; ELF64: sitofp_double_i8
  %b.addr = alloca double, align 8
  %conv = sitofp i8 %a to double
; ELF64: extsb
; ELF64: std {{[0-9]+}}, -[[OFFSET:[0-9]+]](1)
; ELF64: lfd {{[0-9]+}}, -[[OFFSET]](1)
; ELF64: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

; Test fptosi

define void @fptosi_float_i32(float %a) nounwind ssp {
entry:
; ELF64: fptosi_float_i32
  %b.addr = alloca i32, align 4
  %conv = fptosi float %a to i32
; ELF64: fctiwz
; ELF64: stfd
; ELF64: lwa
  store i32 %conv, i32* %b.addr, align 4
  ret void
}

define void @fptosi_float_i64(float %a) nounwind ssp {
entry:
; ELF64: fptosi_float_i64
  %b.addr = alloca i64, align 4
  %conv = fptosi float %a to i64
; ELF64: fctidz
; ELF64: stfd
; ELF64: ld
  store i64 %conv, i64* %b.addr, align 4
  ret void
}

define void @fptosi_double_i32(double %a) nounwind ssp {
entry:
; ELF64: fptosi_double_i32
  %b.addr = alloca i32, align 8
  %conv = fptosi double %a to i32
; ELF64: fctiwz
; ELF64: stfd
; ELF64: lwa
  store i32 %conv, i32* %b.addr, align 8
  ret void
}

define void @fptosi_double_i64(double %a) nounwind ssp {
entry:
; ELF64: fptosi_double_i64
  %b.addr = alloca i64, align 8
  %conv = fptosi double %a to i64
; ELF64: fctidz
; ELF64: stfd
; ELF64: ld
  store i64 %conv, i64* %b.addr, align 8
  ret void
}

; Test fptoui

define void @fptoui_float_i32(float %a) nounwind ssp {
entry:
; ELF64: fptoui_float_i32
  %b.addr = alloca i32, align 4
  %conv = fptoui float %a to i32
; ELF64: fctidz
; ELF64: stfd
; ELF64: lwz
  store i32 %conv, i32* %b.addr, align 4
  ret void
}

define void @fptoui_float_i64(float %a) nounwind ssp {
entry:
; ELF64: fptoui_float_i64
  %b.addr = alloca i64, align 4
  %conv = fptoui float %a to i64
; ELF64: fctiduz
; ELF64: stfd
; ELF64: ld
  store i64 %conv, i64* %b.addr, align 4
  ret void
}

define void @fptoui_double_i32(double %a) nounwind ssp {
entry:
; ELF64: fptoui_double_i32
  %b.addr = alloca i32, align 8
  %conv = fptoui double %a to i32
; ELF64: fctidz
; ELF64: stfd
; ELF64: lwz
  store i32 %conv, i32* %b.addr, align 8
  ret void
}

define void @fptoui_double_i64(double %a) nounwind ssp {
entry:
; ELF64: fptoui_double_i64
  %b.addr = alloca i64, align 8
  %conv = fptoui double %a to i64
; ELF64: fctiduz
; ELF64: stfd
; ELF64: ld
  store i64 %conv, i64* %b.addr, align 8
  ret void
}
