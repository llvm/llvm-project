; RUN: opt -S -passes=instcombine < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32)

; SVE intrinsics fmul and fadd should be replaced with regular fmul and fadd
declare <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
define <vscale x 8 x half> @replace_fmul_intrinsic_half(<vscale x 8 x half> %a, <vscale x 8 x half> %b) #0 {
; CHECK-LABEL: @replace_fmul_intrinsic_half
; CHECK-NEXT:  %1 = fmul fast <vscale x 8 x half> %a, %b
; CHECK-NEXT:  ret <vscale x 8 x half> %1
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %2 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %1, <vscale x 8 x half> %a, <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %2
}

declare <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
define <vscale x 4 x float> @replace_fmul_intrinsic_float(<vscale x 4 x float> %a, <vscale x 4 x float> %b) #0 {
; CHECK-LABEL: @replace_fmul_intrinsic_float
; CHECK-NEXT:  %1 = fmul fast <vscale x 4 x float> %a, %b
; CHECK-NEXT:  ret <vscale x 4 x float> %1
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call fast <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1> %1, <vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %2
}

declare <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)
define <vscale x 2 x double> @replace_fmul_intrinsic_double(<vscale x 2 x double> %a, <vscale x 2 x double> %b) #0 {
; CHECK-LABEL: @replace_fmul_intrinsic_double
; CHECK-NEXT:  %1 = fmul fast <vscale x 2 x double> %a, %b
; CHECK-NEXT:  ret <vscale x 2 x double> %1
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call fast <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> %1, <vscale x 2 x double> %a, <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %2
}

declare <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
define <vscale x 8 x half> @replace_fadd_intrinsic_half(<vscale x 8 x half> %a, <vscale x 8 x half> %b) #0 {
; CHECK-LABEL: @replace_fadd_intrinsic_half
; CHECK-NEXT:  %1 = fadd fast <vscale x 8 x half> %a, %b
; CHECK-NEXT:  ret <vscale x 8 x half> %1
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %2 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %1, <vscale x 8 x half> %a, <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %2
}

declare <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
define <vscale x 4 x float> @replace_fadd_intrinsic_float(<vscale x 4 x float> %a, <vscale x 4 x float> %b) #0 {
; CHECK-LABEL: @replace_fadd_intrinsic_float
; CHECK-NEXT:  %1 = fadd fast <vscale x 4 x float> %a, %b
; CHECK-NEXT:  ret <vscale x 4 x float> %1
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call fast <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1> %1, <vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %2
}

declare <vscale x 2 x double> @llvm.aarch64.sve.fadd.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)
define <vscale x 2 x double> @replace_fadd_intrinsic_double(<vscale x 2 x double> %a, <vscale x 2 x double> %b) #0 {
; CHECK-LABEL: @replace_fadd_intrinsic_double
; CHECK-NEXT:  %1 = fadd fast <vscale x 2 x double> %a, %b
; CHECK-NEXT:  ret <vscale x 2 x double> %1
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call fast <vscale x 2 x double> @llvm.aarch64.sve.fadd.nxv2f64(<vscale x 2 x i1> %1, <vscale x 2 x double> %a, <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %2
}

declare <vscale x 8 x half> @llvm.aarch64.sve.fsub.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
define <vscale x 8 x half> @replace_fsub_intrinsic_half(<vscale x 8 x half> %a, <vscale x 8 x half> %b) #0 {
; CHECK-LABEL: @replace_fsub_intrinsic_half
; CHECK-NEXT:  %1 = fsub fast <vscale x 8 x half> %a, %b
; CHECK-NEXT:  ret <vscale x 8 x half> %1
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %2 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fsub.nxv8f16(<vscale x 8 x i1> %1, <vscale x 8 x half> %a, <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %2
}

declare <vscale x 4 x float> @llvm.aarch64.sve.fsub.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
define <vscale x 4 x float> @replace_fsub_intrinsic_float(<vscale x 4 x float> %a, <vscale x 4 x float> %b) #0 {
; CHECK-LABEL: @replace_fsub_intrinsic_float
; CHECK-NEXT:  %1 = fsub fast <vscale x 4 x float> %a, %b
; CHECK-NEXT:  ret <vscale x 4 x float> %1
  %1 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %2 = tail call fast <vscale x 4 x float> @llvm.aarch64.sve.fsub.nxv4f32(<vscale x 4 x i1> %1, <vscale x 4 x float> %a, <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %2
}


declare <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)
define <vscale x 2 x double> @replace_fsub_intrinsic_double(<vscale x 2 x double> %a, <vscale x 2 x double> %b) #0 {
; CHECK-LABEL: @replace_fsub_intrinsic_double
; CHECK-NEXT:  %1 = fsub fast <vscale x 2 x double> %a, %b
; CHECK-NEXT:  ret <vscale x 2 x double> %1
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call fast <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1> %1, <vscale x 2 x double> %a, <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %2
}

define <vscale x 2 x double> @no_replace_on_non_ptrue_all(<vscale x 2 x double> %a, <vscale x 2 x double> %b) #0 {
; CHECK-LABEL: @no_replace_on_non_ptrue_all
; CHECK-NEXT:  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 5)
; CHECK-NEXT:  %2 = tail call fast <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1> %1, <vscale x 2 x double> %a, <vscale x 2 x double> %b)
; CHECK-NEXT:  ret <vscale x 2 x double> %2
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 5)
  %2 = tail call fast <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1> %1, <vscale x 2 x double> %a, <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %2
}

define <vscale x 2 x double> @replace_fsub_intrinsic_no_fast_flag(<vscale x 2 x double> %a, <vscale x 2 x double> %b) #0 {
; CHECK-LABEL: @replace_fsub_intrinsic_no_fast_flag
; CHECK-NEXT:  %1 = fsub <vscale x 2 x double> %a, %b
; CHECK-NEXT:  ret <vscale x 2 x double> %1
  %1 = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %2 = tail call <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1> %1, <vscale x 2 x double> %a, <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %2
}

attributes #0 = { "target-features"="+sve" }
