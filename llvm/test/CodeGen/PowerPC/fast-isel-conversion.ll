; FIXME: FastISel currently returns false if it hits code that uses VSX
; registers and with -fast-isel-abort=1 turned on the test case will then fail.
; When fastisel better supports VSX fix up this test case.
;
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx | FileCheck %s
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx | FileCheck %s
; RUN: llc < %s -O0 -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=970 -mattr=-vsx | FileCheck %s --check-prefix=PPC970

;; Tests for 970 don't use -fast-isel-abort=1 because we intentionally punt
;; to SelectionDAG in some cases.

; Test sitofp

define void @sitofp_single_i64(i64 %a, float %b) nounwind {
entry:
; CHECK: sitofp_single_i64
; PPC970: sitofp_single_i64
  %b.addr = alloca float, align 4
  %conv = sitofp i64 %a to float
; CHECK: std
; CHECK: lfd
; CHECK: fcfids
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
; PPC970: frsp
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @sitofp_single_i32(i32 %a, float %b) nounwind {
entry:
; CHECK: sitofp_single_i32
; PPC970: sitofp_single_i32
  %b.addr = alloca float, align 4
  %conv = sitofp i32 %a to float
; CHECK: std
; CHECK-NEXT: li
; CHECK-NEXT: lfiwax
; CHECK-NEXT: fcfids
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
; PPC970: frsp
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @sitofp_single_i16(i16 %a, float %b) nounwind {
entry:
; CHECK: sitofp_single_i16
; PPC970: sitofp_single_i16
  %b.addr = alloca float, align 4
  %conv = sitofp i16 %a to float
; CHECK: extsh
; CHECK: std
; CHECK: lfd
; CHECK: fcfids
; PPC970: extsh
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
; PPC970: frsp
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @sitofp_single_i8(i8 %a) nounwind {
entry:
; CHECK: sitofp_single_i8
; PPC970: sitofp_single_i8
  %b.addr = alloca float, align 4
  %conv = sitofp i8 %a to float
; CHECK: extsb
; CHECK: std
; CHECK: lfd
; CHECK: fcfids
; PPC970: extsb
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
; PPC970: frsp
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @sitofp_double_i32(i32 %a, double %b) nounwind {
entry:
; CHECK: sitofp_double_i32
; PPC970: sitofp_double_i32
  %b.addr = alloca double, align 8
  %conv = sitofp i32 %a to double
; CHECK: std
; CHECK-NOT: ori
; CHECK: li
; CHECK-NOT: ori
; CHECK: lfiwax
; CHECK: fcfid
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @sitofp_double_i64(i64 %a, double %b) nounwind {
entry:
; CHECK: sitofp_double_i64
; PPC970: sitofp_double_i64
  %b.addr = alloca double, align 8
  %conv = sitofp i64 %a to double
; CHECK: std
; CHECK: lfd
; CHECK: fcfid
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @sitofp_double_i16(i16 %a, double %b) nounwind {
entry:
; CHECK: sitofp_double_i16
; PPC970: sitofp_double_i16
  %b.addr = alloca double, align 8
  %conv = sitofp i16 %a to double
; CHECK: extsh
; CHECK: std
; CHECK: lfd
; CHECK: fcfid
; PPC970: extsh
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @sitofp_double_i8(i8 %a, double %b) nounwind {
entry:
; CHECK: sitofp_double_i8
; PPC970: sitofp_double_i8
  %b.addr = alloca double, align 8
  %conv = sitofp i8 %a to double
; CHECK: extsb
; CHECK: std
; CHECK: lfd
; CHECK: fcfid
; PPC970: extsb
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

; Test uitofp

define void @uitofp_single_i64(i64 %a, float %b) nounwind {
entry:
; CHECK: uitofp_single_i64
; PPC970: uitofp_single_i64
  %b.addr = alloca float, align 4
  %conv = uitofp i64 %a to float
; CHECK: std
; CHECK: lfd
; CHECK: fcfidus
; PPC970-NOT: fcfidus
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @uitofp_single_i32(i32 %a, float %b) nounwind {
entry:
; CHECK: uitofp_single_i32
; PPC970: uitofp_single_i32
  %b.addr = alloca float, align 4
  %conv = uitofp i32 %a to float
; CHECK: std
; CHECK-NOT: ori
; CHECK: li
; CHECK-NOT: ori
; CHECK: lfiwzx
; CHECK: fcfidus
; PPC970-NOT: lfiwzx
; PPC970-NOT: fcfidus
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @uitofp_single_i16(i16 %a, float %b) nounwind {
entry:
; CHECK: uitofp_single_i16
; PPC970: uitofp_single_i16
  %b.addr = alloca float, align 4
  %conv = uitofp i16 %a to float
; CHECK: clrldi {{[0-9]+}}, {{[0-9]+}}, 48
; CHECK: std
; CHECK: lfd
; CHECK: fcfidus
; PPC970: clrlwi {{[0-9]+}}, {{[0-9]+}}, 16
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
; PPC970: frsp
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @uitofp_single_i8(i8 %a) nounwind {
entry:
; CHECK: uitofp_single_i8
; PPC970: uitofp_single_i8
  %b.addr = alloca float, align 4
  %conv = uitofp i8 %a to float
; CHECK: clrldi {{[0-9]+}}, {{[0-9]+}}, 56
; CHECK: std
; CHECK: lfd
; CHECK: fcfidus
; PPC970: clrlwi {{[0-9]+}}, {{[0-9]+}}, 24
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
; PPC970: frsp
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @uitofp_double_i64(i64 %a, double %b) nounwind {
entry:
; CHECK: uitofp_double_i64
; PPC970: uitofp_double_i64
  %b.addr = alloca double, align 8
  %conv = uitofp i64 %a to double
; CHECK: std
; CHECK: lfd
; CHECK: fcfidu
; PPC970-NOT: fcfidu
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @uitofp_double_i32(i32 %a, double %b) nounwind {
entry:
; CHECK: uitofp_double_i32
; PPC970: uitofp_double_i32
  %b.addr = alloca double, align 8
  %conv = uitofp i32 %a to double
; CHECK: std
; CHECK-NEXT: li
; CHECK-NEXT: lfiwzx
; CHECK-NEXT: fcfidu
; CHECKLE: fcfidu
; PPC970-NOT: lfiwzx
; PPC970-NOT: fcfidu
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @uitofp_double_i16(i16 %a, double %b) nounwind {
entry:
; CHECK: uitofp_double_i16
; PPC970: uitofp_double_i16
  %b.addr = alloca double, align 8
  %conv = uitofp i16 %a to double
; CHECK: clrldi {{[0-9]+}}, {{[0-9]+}}, 48
; CHECK: std
; CHECK: lfd
; CHECK: fcfidu
; PPC970: clrlwi {{[0-9]+}}, {{[0-9]+}}, 16
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @uitofp_double_i8(i8 %a, double %b) nounwind {
entry:
; CHECK: uitofp_double_i8
; PPC970: uitofp_double_i8
  %b.addr = alloca double, align 8
  %conv = uitofp i8 %a to double
; CHECK: clrldi {{[0-9]+}}, {{[0-9]+}}, 56
; CHECK: std
; CHECK: lfd
; CHECK: fcfidu
; PPC970: clrlwi {{[0-9]+}}, {{[0-9]+}}, 24
; PPC970: std
; PPC970: lfd
; PPC970: fcfid
  store double %conv, double* %b.addr, align 8
  ret void
}

; Test fptosi

define void @fptosi_float_i32(float %a) nounwind {
entry:
; CHECK: fptosi_float_i32
; PPC970: fptosi_float_i32
  %b.addr = alloca i32, align 4
  %conv = fptosi float %a to i32
; CHECK: fctiwz
; CHECK: stfd
; CHECK: lwa
; PPC970: fctiwz
; PPC970: stfd
; PPC970: lwa
  store i32 %conv, i32* %b.addr, align 4
  ret void
}

define void @fptosi_float_i64(float %a) nounwind {
entry:
; CHECK: fptosi_float_i64
; PPC970: fptosi_float_i64
  %b.addr = alloca i64, align 4
  %conv = fptosi float %a to i64
; CHECK: fctidz
; CHECK: stfd
; CHECK: ld
; PPC970: fctidz
; PPC970: stfd
; PPC970: ld
  store i64 %conv, i64* %b.addr, align 4
  ret void
}

define void @fptosi_double_i32(double %a) nounwind {
entry:
; CHECK: fptosi_double_i32
; PPC970: fptosi_double_i32
  %b.addr = alloca i32, align 8
  %conv = fptosi double %a to i32
; CHECK: fctiwz
; CHECK: stfd
; CHECK: lwa
; PPC970: fctiwz
; PPC970: stfd
; PPC970: lwa
  store i32 %conv, i32* %b.addr, align 8
  ret void
}

define void @fptosi_double_i64(double %a) nounwind {
entry:
; CHECK: fptosi_double_i64
; PPC970: fptosi_double_i64
  %b.addr = alloca i64, align 8
  %conv = fptosi double %a to i64
; CHECK: fctidz
; CHECK: stfd
; CHECK: ld
; PPC970: fctidz
; PPC970: stfd
; PPC970: ld
  store i64 %conv, i64* %b.addr, align 8
  ret void
}

; Test fptoui

define void @fptoui_float_i32(float %a) nounwind {
entry:
; CHECK: fptoui_float_i32
; PPC970: fptoui_float_i32
  %b.addr = alloca i32, align 4
  %conv = fptoui float %a to i32
; CHECK: fctiwuz
; CHECK: stfd
; CHECK: lwz
; PPC970: fctidz
; PPC970: stfd
; PPC970: lwz
  store i32 %conv, i32* %b.addr, align 4
  ret void
}

define void @fptoui_float_i64(float %a) nounwind {
entry:
; CHECK: fptoui_float_i64
; PPC970: fptoui_float_i64
  %b.addr = alloca i64, align 4
  %conv = fptoui float %a to i64
; CHECK: fctiduz
; CHECK: stfd
; CHECK: ld
; PPC970-NOT: fctiduz
  store i64 %conv, i64* %b.addr, align 4
  ret void
}

define void @fptoui_double_i32(double %a) nounwind {
entry:
; CHECK: fptoui_double_i32
; PPC970: fptoui_double_i32
  %b.addr = alloca i32, align 8
  %conv = fptoui double %a to i32
; CHECK: fctiwuz
; CHECK: stfd
; CHECK: lwz
; PPC970: fctidz
; PPC970: stfd
; PPC970: lwz
  store i32 %conv, i32* %b.addr, align 8
  ret void
}

define void @fptoui_double_i64(double %a) nounwind {
entry:
; CHECK: fptoui_double_i64
; PPC970: fptoui_double_i64
  %b.addr = alloca i64, align 8
  %conv = fptoui double %a to i64
; CHECK: fctiduz
; CHECK: stfd
; CHECK: ld
; PPC970-NOT: fctiduz
  store i64 %conv, i64* %b.addr, align 8
  ret void
}
