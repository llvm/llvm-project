; RUN: llc -enable-ppc-gen-scalar-mass -verify-machineinstrs -O3 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck --check-prefix=CHECK-LNX %s
; RUN: llc -enable-ppc-gen-scalar-mass -verify-machineinstrs -O3 -mtriple=powerpc-ibm-aix-xcoff < %s | FileCheck --check-prefix=CHECK-AIX %s

declare float @llvm.pow.f32 (float, float);
declare double @llvm.pow.f64 (double, double);

; afn flag powf with 0.25
define float @llvmintr_powf_f32_afn025(float %a) {
; CHECK-LNX-LABEL: llvmintr_powf_f32_afn025:
; CHECK-LNX:       bl __xl_powf
; CHECK-LNX:       blr
;
; CHECK-AIX-LABEL: llvmintr_powf_f32_afn025:
; CHECK-AIX:       bl .__xl_powf[PR]
; CHECK-AIX:       blr
entry:
  %call = tail call afn float @llvm.pow.f32(float %a, float 2.500000e-01)
  ret float %call
}

; afn flag pow with 0.25
define double @llvmintr_pow_f64_afn025(double %a) {
; CHECK-LNX-LABEL: llvmintr_pow_f64_afn025:
; CHECK-LNX:       bl __xl_pow
; CHECK-LNX:       blr
;
; CHECK-AIX-LABEL: llvmintr_pow_f64_afn025:
; CHECK-AIX:       bl .__xl_pow[PR]
; CHECK-AIX:       blr
entry:
  %call = tail call afn double @llvm.pow.f64(double %a, double 2.500000e-01)
  ret double %call
}

; afn flag powf with 0.75
define float @llvmintr_powf_f32_afn075(float %a) {
; CHECK-LNX-LABEL: llvmintr_powf_f32_afn075:
; CHECK-LNX:       bl __xl_powf
; CHECK-LNX:       blr
;
; CHECK-AIX-LABEL: llvmintr_powf_f32_afn075:
; CHECK-AIX:       # %bb.0: # %entry
; CHECK-AIX:       bl .__xl_powf[PR]
; CHECK-AIX:       blr
entry:
  %call = tail call afn float @llvm.pow.f32(float %a, float 7.500000e-01)
  ret float %call
}

; afn flag pow with 0.75
define double @llvmintr_pow_f64_afn075(double %a) {
; CHECK-LNX-LABEL: llvmintr_pow_f64_afn075:
; CHECK-LNX:       bl __xl_pow
; CHECK-LNX:       blr
;
; CHECK-AIX-LABEL: llvmintr_pow_f64_afn075:
; CHECK-AIX:       bl .__xl_pow[PR]
; CHECK-AIX:       blr
entry:
  %call = tail call afn double @llvm.pow.f64(double %a, double 7.500000e-01)
  ret double %call
}

; afn flag powf with 0.50
define float @llvmintr_powf_f32_afn050(float %a) {
; CHECK-LNX-LABEL: llvmintr_powf_f32_afn050:
; CHECK-LNX:       # %bb.0: # %entry
; CHECK-LNX:       bl __xl_powf
; CHECK-LNX:       blr
;
; CHECK-AIX-LABEL: llvmintr_powf_f32_afn050:
; CHECK-AIX:       # %bb.0: # %entry
; CHECK-AIX:       bl .__xl_powf[PR]
; CHECK-AIX:       blr
entry:
  %call = tail call afn float @llvm.pow.f32(float %a, float 5.000000e-01)
  ret float %call
}

; afn flag pow with 0.50
define double @llvmintr_pow_f64_afn050(double %a) {
; CHECK-LNX-LABEL: llvmintr_pow_f64_afn050:
; CHECK-LNX:       # %bb.0: # %entry
; CHECK-LNX:       bl __xl_pow
; CHECK-LNX:       blr
;
; CHECK-AIX-LABEL: llvmintr_pow_f64_afn050:
; CHECK-AIX:       # %bb.0: # %entry
; CHECK-AIX:       bl .__xl_pow[PR]
; CHECK-AIX:       blr
entry:
  %call = tail call afn double @llvm.pow.f64(double %a, double 5.000000e-01)
  ret double %call
}
