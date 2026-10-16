; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck -check-prefixes=FUNC,GCN,FMA_F64 %s
; RUN: llc -mtriple=amdgpu8.02 < %s | FileCheck -check-prefixes=FUNC,GCN,FMA_F64 %s
; RUN: llc -mtriple=amdgpu9.0a < %s | FileCheck -check-prefixes=FUNC,GCN,FMAC_F64 %s
; RUN: llc -mtriple=amdgpu11.00 < %s | FileCheck -check-prefixes=FUNC,GCN,FMA_F64 %s
; RUN: llc -mtriple=amdgpu12.50 < %s | FileCheck -check-prefixes=FUNC,GCN,FMAC_F64 %s

declare double @llvm.fma.f64(double, double, double) nounwind readnone
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>) nounwind readnone
declare double @llvm.fabs.f64(double) nounwind readnone

; FUNC-LABEL: {{^}}fma_f64:
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define double @fma_f64(double %in1, double %in2, double %in3) {
   %r3 = tail call double @llvm.fma.f64(double %in1, double %in2, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_v2f64:
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define <2 x double> @fma_v2f64(<2 x double> %in1, <2 x double> %in2, <2 x double> %in3) {
   %r3 = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %in1, <2 x double> %in2, <2 x double> %in3)
  ret <2 x double> %r3
}

; FUNC-LABEL: {{^}}fma_v4f64:
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define <4 x double> @fma_v4f64(<4 x double> %in1, <4 x double> %in2, <4 x double> %in3) {
   %r3 = tail call <4 x double> @llvm.fma.v4f64(<4 x double> %in1, <4 x double> %in2, <4 x double> %in3)
  ret <4 x double> %r3
}

; FUNC-LABEL: {{^}}fma_f64_abs_src0:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], |v\[[0-9]+:[0-9]+\]|, v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_abs_src0(double %in1, double %in2, double %in3) {
   %fabs = call double @llvm.fabs.f64(double %in1)
   %r3 = tail call double @llvm.fma.f64(double %fabs, double %in2, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_abs_src1:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], \|v\[[0-9]+:[0-9]+\]\|, v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_abs_src1(double %in1, double %in2, double %in3) {
   %fabs = call double @llvm.fabs.f64(double %in2)
   %r3 = tail call double @llvm.fma.f64(double %in1, double %fabs, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_abs_src2:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], \|v\[[0-9]+:[0-9]+\]\|}}
define double @fma_f64_abs_src2(double %in1, double %in2, double %in3) {
   %fabs = call double @llvm.fabs.f64(double %in3)
   %r3 = tail call double @llvm.fma.f64(double %in1, double %in2, double %fabs)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_neg_src0:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_neg_src0(double %in1, double %in2, double %in3) {
   %fsub = fsub double -0.000000e+00, %in1
   %r3 = tail call double @llvm.fma.f64(double %fsub, double %in2, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_neg_src1:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_neg_src1(double %in1, double %in2, double %in3) {
   %fsub = fsub double -0.000000e+00, %in2
   %r3 = tail call double @llvm.fma.f64(double %in1, double %fsub, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_neg_src2:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_neg_src2(double %in1, double %in2, double %in3) {
   %fsub = fsub double -0.000000e+00, %in3
   %r3 = tail call double @llvm.fma.f64(double %in1, double %in2, double %fsub)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_abs_neg_src0:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|, v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_abs_neg_src0(double %in1, double %in2, double %in3) {
   %fabs = call double @llvm.fabs.f64(double %in1)
   %fsub = fsub double -0.000000e+00, %fabs
   %r3 = tail call double @llvm.fma.f64(double %fsub, double %in2, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_abs_neg_src1:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|, v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_abs_neg_src1(double %in1, double %in2, double %in3) {
   %fabs = call double @llvm.fabs.f64(double %in2)
   %fsub = fsub double -0.000000e+00, %fabs
   %r3 = tail call double @llvm.fma.f64(double %in1, double %fsub, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_abs_neg_src2:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -\|v\[[0-9]+:[0-9]+\]\|}}
define double @fma_f64_abs_neg_src2(double %in1, double %in2, double %in3) {
   %fabs = call double @llvm.fabs.f64(double %in3)
   %fsub = fsub double -0.000000e+00, %fabs
   %r3 = tail call double @llvm.fma.f64(double %in1, double %in2, double %fsub)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_lit_src0:
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], 2.0, v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], 2.0, v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_lit_src0(double %in2, double %in3) {
   %r3 = tail call double @llvm.fma.f64(double +2.0, double %in2, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_lit_src1:
; FMA_F64: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], 2.0, v\[[0-9]+:[0-9]+\]}}
; FMAC_F64: v_fmac_f64_e32 {{v\[[0-9]+:[0-9]+\], 2.0, v\[[0-9]+:[0-9]+\]}}
define double @fma_f64_lit_src1(double %in1, double %in3) {
   %r3 = tail call double @llvm.fma.f64(double %in1, double +2.0, double %in3)
  ret double %r3
}

; FUNC-LABEL: {{^}}fma_f64_lit_src2:
; GCN: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], 2.0}}
define double @fma_f64_lit_src2(double %in1, double %in2) {
   %r3 = tail call double @llvm.fma.f64(double %in1, double %in2, double +2.0)
  ret double %r3
}
