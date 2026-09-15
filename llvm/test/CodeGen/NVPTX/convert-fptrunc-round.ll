; Split since some rounding modes are unsupported on older hardware.
; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/sm80.ll > %t/all-sm80.ll
; RUN: cat %t/common.ll %t/sm80.ll %t/sm90.ll > %t/all-sm90.ll
; RUN: llc < %t/common.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx71 | FileCheck --check-prefixes=SM70,SM70-NOFTZ %s
; RUN: %if ptxas-sm_70 && ptxas-isa-7.1 %{ llc < %t/common.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx71 | %ptxas-verify -arch=sm_70 %}
; RUN: llc < %t/common.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx71 -denormal-fp-math-f32=preserve-sign | FileCheck --check-prefixes=SM70,SM70-FTZ %s
; RUN: %if ptxas-sm_70 && ptxas-isa-7.1 %{ llc < %t/common.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx71 -denormal-fp-math-f32=preserve-sign | %ptxas-verify -arch=sm_70 %}
; RUN: llc < %t/all-sm80.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck --check-prefix=SM80-PTX70 %s
; RUN: %if ptxas-sm_80 && ptxas-isa-7.0 %{ llc < %t/all-sm80.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}
; RUN: llc < %t/all-sm80.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx71 | FileCheck --check-prefixes=SM80,SM80-NOFTZ %s
; RUN: %if ptxas-sm_80 && ptxas-isa-7.1 %{ llc < %t/all-sm80.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx71 | %ptxas-verify -arch=sm_80 %}
; RUN: llc < %t/all-sm80.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx71 -denormal-fp-math-f32=preserve-sign | FileCheck --check-prefixes=SM80,SM80-FTZ %s
; RUN: %if ptxas-sm_80 && ptxas-isa-7.1 %{ llc < %t/all-sm80.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx71 -denormal-fp-math-f32=preserve-sign | %ptxas-verify -arch=sm_80 %}
; RUN: llc < %t/all-sm90.ll -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | FileCheck --check-prefixes=SM90,SM90-NOFTZ %s
; RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{ llc < %t/all-sm90.ll -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | %ptxas-verify -arch=sm_90 %}
; RUN: llc < %t/all-sm90.ll -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 -denormal-fp-math-f32=preserve-sign | FileCheck --check-prefixes=SM90,SM90-FTZ %s
; RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{ llc < %t/all-sm90.ll -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 -denormal-fp-math-f32=preserve-sign | %ptxas-verify -arch=sm_90 %}

; RUN: not llc < %t/all-sm90.ll -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx71 -o /dev/null 2>&1 | FileCheck --check-prefix=SM70-ERR %s
; RUN: not llc < %t/all-sm90.ll -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx71 -o /dev/null 2>&1 | FileCheck --check-prefix=SM80-ERR %s
; RUN: not llc < %t/unsupported-rounding.ll -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 -o /dev/null 2>&1 | FileCheck --check-prefix=ROUND-ERR %s

;--- common.ll
declare half @llvm.fptrunc.round.f16.f32(float, metadata)
declare bfloat @llvm.fptrunc.round.bf16.f32(float, metadata)
declare half @llvm.fptrunc.round.f16.f64(double, metadata)
declare float @llvm.fptrunc.round.f32.f64(double, metadata)
declare bfloat @llvm.fptrunc.round.bf16.f64(double, metadata)

define half @cvt_f16_bf16(bfloat %a) {
; SM70-LABEL: cvt_f16_bf16(
; SM70:       ld.param.b16 [[A:%r[0-9]+]], [cvt_f16_bf16_param_0];
; SM70-NEXT:  shl.b32 [[EXT:%r[0-9]+]], [[A]], 16;
; SM70-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: cvt_f16_bf16(
; SM80-PTX70:       shl.b32 [[EXT:%r[0-9]+]], {{%r[0-9]+}}, 16;
; SM80-PTX70-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: cvt_f16_bf16(
; SM80:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_f16_bf16_param_0];
; SM80-NEXT:  cvt.f32.bf16 [[EXT:%r[0-9]+]], [[A]];
; SM80-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: cvt_f16_bf16(
; SM90:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_f16_bf16_param_0];
; SM90-NEXT:  cvt.rn.f16.bf16 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %ext = fpext bfloat %a to float
  %res = fptrunc float %ext to half
  ret half %res
}

define bfloat @cvt_bf16_f16(half %a) {
; SM70-LABEL:       cvt_bf16_f16(
; SM70-NOFTZ:       cvt.f32.f16 [[EXT:%r[0-9]+]], %rs{{[0-9]+}};
; SM70-FTZ:         cvt.ftz.f32.f16 [[EXT:%r[0-9]+]], %rs{{[0-9]+}};
; SM70-NEXT:        bfe.u32 {{%r[0-9]+}}, [[EXT]], 16, 1;
; SM70:             shr.u32 [[RES:%r[0-9]+]], {{%r[0-9]+}}, 16;
; SM70-NEXT:        st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:        ret;
;
; SM80-PTX70-LABEL: cvt_bf16_f16(
; SM80-PTX70:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_bf16_f16_param_0];
; SM80-PTX70-NEXT:  cvt.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  cvt.rn.bf16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL:       cvt_bf16_f16(
; SM80:             ld.param.b16 [[A:%rs[0-9]+]], [cvt_bf16_f16_param_0];
; SM80-NOFTZ-NEXT:  cvt.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-FTZ-NEXT:    cvt.ftz.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-NEXT:        cvt.rn.bf16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-NEXT:        st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:        ret;
;
; SM90-LABEL: cvt_bf16_f16(
; SM90:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_bf16_f16_param_0];
; SM90-NEXT:  cvt.rn.bf16.f16 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %ext = fpext half %a to float
  %res = fptrunc float %ext to bfloat
  ret bfloat %res
}

define half @cvt_rn_f16_bf16(bfloat %a) {
; SM70-LABEL: cvt_rn_f16_bf16(
; SM70:       ld.param.b16 [[A:%r[0-9]+]], [cvt_rn_f16_bf16_param_0];
; SM70-NEXT:  shl.b32 [[EXT:%r[0-9]+]], [[A]], 16;
; SM70-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: cvt_rn_f16_bf16(
; SM80-PTX70:       shl.b32 [[EXT:%r[0-9]+]], {{%r[0-9]+}}, 16;
; SM80-PTX70-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: cvt_rn_f16_bf16(
; SM80:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rn_f16_bf16_param_0];
; SM80-NEXT:  cvt.f32.bf16 [[EXT:%r[0-9]+]], [[A]];
; SM80-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: cvt_rn_f16_bf16(
; SM90:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rn_f16_bf16_param_0];
; SM90-NEXT:  cvt.rn.f16.bf16 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %ext = fpext bfloat %a to float
  %res = call half @llvm.fptrunc.round.f16.f32(float %ext, metadata !"round.tonearest")
  ret half %res
}

define half @cvt_rz_f16_bf16(bfloat %a) {
; SM70-LABEL: cvt_rz_f16_bf16(
; SM70:       ld.param.b16 [[A:%r[0-9]+]], [cvt_rz_f16_bf16_param_0];
; SM70-NEXT:  shl.b32 [[EXT:%r[0-9]+]], [[A]], 16;
; SM70-NEXT:  cvt.rz.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: cvt_rz_f16_bf16(
; SM80-PTX70:       shl.b32 [[EXT:%r[0-9]+]], {{%r[0-9]+}}, 16;
; SM80-PTX70-NEXT:  cvt.rz.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: cvt_rz_f16_bf16(
; SM80:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rz_f16_bf16_param_0];
; SM80-NEXT:  cvt.f32.bf16 [[EXT:%r[0-9]+]], [[A]];
; SM80-NEXT:  cvt.rz.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: cvt_rz_f16_bf16(
; SM90:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rz_f16_bf16_param_0];
; SM90-NEXT:  cvt.rz.f16.bf16 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %ext = fpext bfloat %a to float
  %res = call half @llvm.fptrunc.round.f16.f32(float %ext, metadata !"round.towardzero")
  ret half %res
}

define half @cvt_rm_f16_bf16(bfloat %a) {
; SM70-LABEL: cvt_rm_f16_bf16(
; SM70:       ld.param.b16 [[A:%r[0-9]+]], [cvt_rm_f16_bf16_param_0];
; SM70-NEXT:  shl.b32 [[EXT:%r[0-9]+]], [[A]], 16;
; SM70-NEXT:  cvt.rm.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: cvt_rm_f16_bf16(
; SM80-PTX70:       shl.b32 [[EXT:%r[0-9]+]], {{%r[0-9]+}}, 16;
; SM80-PTX70-NEXT:  cvt.rm.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: cvt_rm_f16_bf16(
; SM80:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rm_f16_bf16_param_0];
; SM80-NEXT:  cvt.f32.bf16 [[EXT:%r[0-9]+]], [[A]];
; SM80-NEXT:  cvt.rm.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL:       cvt_rm_f16_bf16(
; SM90:             ld.param.b16 [[A:%rs[0-9]+]], [cvt_rm_f16_bf16_param_0];
; SM90-NOFTZ-NEXT:  cvt.rm.f16.bf16 [[RES:%rs[0-9]+]], [[A]];
; SM90-FTZ-NEXT:    cvt.ftz.f32.bf16 [[EXT:%r[0-9]+]], [[A]];
; SM90-FTZ-NEXT:    cvt.rm.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM90-NEXT:        st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:        ret;
  %ext = fpext bfloat %a to float
  %res = call half @llvm.fptrunc.round.f16.f32(float %ext, metadata !"round.downward")
  ret half %res
}

define half @cvt_rp_f16_bf16(bfloat %a) {
; SM70-LABEL: cvt_rp_f16_bf16(
; SM70:       ld.param.b16 [[A:%r[0-9]+]], [cvt_rp_f16_bf16_param_0];
; SM70-NEXT:  shl.b32 [[EXT:%r[0-9]+]], [[A]], 16;
; SM70-NEXT:  cvt.rp.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: cvt_rp_f16_bf16(
; SM80-PTX70:       shl.b32 [[EXT:%r[0-9]+]], {{%r[0-9]+}}, 16;
; SM80-PTX70-NEXT:  cvt.rp.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: cvt_rp_f16_bf16(
; SM80:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rp_f16_bf16_param_0];
; SM80-NEXT:  cvt.f32.bf16 [[EXT:%r[0-9]+]], [[A]];
; SM80-NEXT:  cvt.rp.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL:       cvt_rp_f16_bf16(
; SM90:             ld.param.b16 [[A:%rs[0-9]+]], [cvt_rp_f16_bf16_param_0];
; SM90-NOFTZ-NEXT:  cvt.rp.f16.bf16 [[RES:%rs[0-9]+]], [[A]];
; SM90-FTZ-NEXT:    cvt.ftz.f32.bf16 [[EXT:%r[0-9]+]], [[A]];
; SM90-FTZ-NEXT:    cvt.rp.f16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM90-NEXT:        st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:        ret;
  %ext = fpext bfloat %a to float
  %res = call half @llvm.fptrunc.round.f16.f32(float %ext, metadata !"round.upward")
  ret half %res
}

define half @fptrunc_rn_f16_f32(float %a) {
; SM70-LABEL: fptrunc_rn_f16_f32(
; SM70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rn_f16_f32_param_0];
; SM70-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: fptrunc_rn_f16_f32(
; SM80-PTX70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rn_f16_f32_param_0];
; SM80-PTX70-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rn_f16_f32(
; SM80:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rn_f16_f32_param_0];
; SM80-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rn_f16_f32(
; SM90:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rn_f16_f32_param_0];
; SM90-NEXT:  cvt.rn.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call half @llvm.fptrunc.round.f16.f32(float %a, metadata !"round.tonearest")
  ret half %res
}

define half @fptrunc_rz_f16_f32(float %a) {
; SM70-LABEL: fptrunc_rz_f16_f32(
; SM70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rz_f16_f32_param_0];
; SM70-NEXT:  cvt.rz.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: fptrunc_rz_f16_f32(
; SM80-PTX70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rz_f16_f32_param_0];
; SM80-PTX70-NEXT:  cvt.rz.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rz_f16_f32(
; SM80:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rz_f16_f32_param_0];
; SM80-NEXT:  cvt.rz.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rz_f16_f32(
; SM90:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rz_f16_f32_param_0];
; SM90-NEXT:  cvt.rz.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call half @llvm.fptrunc.round.f16.f32(float %a, metadata !"round.towardzero")
  ret half %res
}

define half @fptrunc_rm_f16_f32(float %a) {
; SM70-LABEL: fptrunc_rm_f16_f32(
; SM70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rm_f16_f32_param_0];
; SM70-NEXT:  cvt.rm.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: fptrunc_rm_f16_f32(
; SM80-PTX70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rm_f16_f32_param_0];
; SM80-PTX70-NEXT:  cvt.rm.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rm_f16_f32(
; SM80:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rm_f16_f32_param_0];
; SM80-NEXT:  cvt.rm.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rm_f16_f32(
; SM90:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rm_f16_f32_param_0];
; SM90-NEXT:  cvt.rm.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call half @llvm.fptrunc.round.f16.f32(float %a, metadata !"round.downward")
  ret half %res
}

define half @fptrunc_rp_f16_f32(float %a) {
; SM70-LABEL: fptrunc_rp_f16_f32(
; SM70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rp_f16_f32_param_0];
; SM70-NEXT:  cvt.rp.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: fptrunc_rp_f16_f32(
; SM80-PTX70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rp_f16_f32_param_0];
; SM80-PTX70-NEXT:  cvt.rp.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rp_f16_f32(
; SM80:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rp_f16_f32_param_0];
; SM80-NEXT:  cvt.rp.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rp_f16_f32(
; SM90:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rp_f16_f32_param_0];
; SM90-NEXT:  cvt.rp.f16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call half @llvm.fptrunc.round.f16.f32(float %a, metadata !"round.upward")
  ret half %res
}

define half @fptrunc_rn_f16_f64(double %a) {
; SM70-LABEL: fptrunc_rn_f16_f64(
; SM70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_f16_f64_param_0];
; SM70-NEXT:  cvt.rn.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: fptrunc_rn_f16_f64(
; SM80-PTX70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_f16_f64_param_0];
; SM80-PTX70-NEXT:  cvt.rn.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rn_f16_f64(
; SM80:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_f16_f64_param_0];
; SM80-NEXT:  cvt.rn.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rn_f16_f64(
; SM90:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_f16_f64_param_0];
; SM90-NEXT:  cvt.rn.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call half @llvm.fptrunc.round.f16.f64(double %a, metadata !"round.tonearest")
  ret half %res
}

define half @fptrunc_rz_f16_f64(double %a) {
; SM70-LABEL: fptrunc_rz_f16_f64(
; SM70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_f16_f64_param_0];
; SM70-NEXT:  cvt.rz.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: fptrunc_rz_f16_f64(
; SM80-PTX70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_f16_f64_param_0];
; SM80-PTX70-NEXT:  cvt.rz.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rz_f16_f64(
; SM80:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_f16_f64_param_0];
; SM80-NEXT:  cvt.rz.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rz_f16_f64(
; SM90:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_f16_f64_param_0];
; SM90-NEXT:  cvt.rz.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call half @llvm.fptrunc.round.f16.f64(double %a, metadata !"round.towardzero")
  ret half %res
}

define half @fptrunc_rm_f16_f64(double %a) {
; SM70-LABEL: fptrunc_rm_f16_f64(
; SM70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_f16_f64_param_0];
; SM70-NEXT:  cvt.rm.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: fptrunc_rm_f16_f64(
; SM80-PTX70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_f16_f64_param_0];
; SM80-PTX70-NEXT:  cvt.rm.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rm_f16_f64(
; SM80:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_f16_f64_param_0];
; SM80-NEXT:  cvt.rm.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rm_f16_f64(
; SM90:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_f16_f64_param_0];
; SM90-NEXT:  cvt.rm.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call half @llvm.fptrunc.round.f16.f64(double %a, metadata !"round.downward")
  ret half %res
}

define half @fptrunc_rp_f16_f64(double %a) {
; SM70-LABEL: fptrunc_rp_f16_f64(
; SM70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_f16_f64_param_0];
; SM70-NEXT:  cvt.rp.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM70-NEXT:  ret;
;
; SM80-PTX70-LABEL: fptrunc_rp_f16_f64(
; SM80-PTX70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_f16_f64_param_0];
; SM80-PTX70-NEXT:  cvt.rp.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rp_f16_f64(
; SM80:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_f16_f64_param_0];
; SM80-NEXT:  cvt.rp.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rp_f16_f64(
; SM90:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_f16_f64_param_0];
; SM90-NEXT:  cvt.rp.f16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call half @llvm.fptrunc.round.f16.f64(double %a, metadata !"round.upward")
  ret half %res
}

define float @fptrunc_rn_f32_f64(double %a) {
; SM70-LABEL:       fptrunc_rn_f32_f64(
; SM70:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_f32_f64_param_0];
; SM70-NOFTZ-NEXT:  cvt.rn.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM70-FTZ-NEXT:    cvt.rn.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM70-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM70-NEXT:        ret;
;
; SM80-PTX70-LABEL: fptrunc_rn_f32_f64(
; SM80-PTX70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_f32_f64_param_0];
; SM80-PTX70-NEXT:  cvt.rn.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b32 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL:       fptrunc_rn_f32_f64(
; SM80:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_f32_f64_param_0];
; SM80-NOFTZ-NEXT:  cvt.rn.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-FTZ-NEXT:    cvt.rn.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM80-NEXT:        ret;
;
; SM90-LABEL:       fptrunc_rn_f32_f64(
; SM90:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_f32_f64_param_0];
; SM90-NOFTZ-NEXT:  cvt.rn.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM90-FTZ-NEXT:    cvt.rn.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM90-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM90-NEXT:        ret;
  %res = call float @llvm.fptrunc.round.f32.f64(double %a, metadata !"round.tonearest")
  ret float %res
}

define float @fptrunc_rz_f32_f64(double %a) {
; SM70-LABEL:       fptrunc_rz_f32_f64(
; SM70:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_f32_f64_param_0];
; SM70-NOFTZ-NEXT:  cvt.rz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM70-FTZ-NEXT:    cvt.rz.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM70-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM70-NEXT:        ret;
;
; SM80-PTX70-LABEL: fptrunc_rz_f32_f64(
; SM80-PTX70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_f32_f64_param_0];
; SM80-PTX70-NEXT:  cvt.rz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b32 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL:       fptrunc_rz_f32_f64(
; SM80:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_f32_f64_param_0];
; SM80-NOFTZ-NEXT:  cvt.rz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-FTZ-NEXT:    cvt.rz.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM80-NEXT:        ret;
;
; SM90-LABEL:       fptrunc_rz_f32_f64(
; SM90:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_f32_f64_param_0];
; SM90-NOFTZ-NEXT:  cvt.rz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM90-FTZ-NEXT:    cvt.rz.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM90-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM90-NEXT:        ret;
  %res = call float @llvm.fptrunc.round.f32.f64(double %a, metadata !"round.towardzero")
  ret float %res
}

define float @fptrunc_rm_f32_f64(double %a) {
; SM70-LABEL:       fptrunc_rm_f32_f64(
; SM70:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_f32_f64_param_0];
; SM70-NOFTZ-NEXT:  cvt.rm.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM70-FTZ-NEXT:    cvt.rm.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM70-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM70-NEXT:        ret;
;
; SM80-PTX70-LABEL: fptrunc_rm_f32_f64(
; SM80-PTX70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_f32_f64_param_0];
; SM80-PTX70-NEXT:  cvt.rm.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b32 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL:       fptrunc_rm_f32_f64(
; SM80:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_f32_f64_param_0];
; SM80-NOFTZ-NEXT:  cvt.rm.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-FTZ-NEXT:    cvt.rm.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM80-NEXT:        ret;
;
; SM90-LABEL:       fptrunc_rm_f32_f64(
; SM90:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_f32_f64_param_0];
; SM90-NOFTZ-NEXT:  cvt.rm.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM90-FTZ-NEXT:    cvt.rm.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM90-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM90-NEXT:        ret;
  %res = call float @llvm.fptrunc.round.f32.f64(double %a, metadata !"round.downward")
  ret float %res
}

define float @fptrunc_rp_f32_f64(double %a) {
; SM70-LABEL:       fptrunc_rp_f32_f64(
; SM70:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_f32_f64_param_0];
; SM70-NOFTZ-NEXT:  cvt.rp.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM70-FTZ-NEXT:    cvt.rp.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM70-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM70-NEXT:        ret;
;
; SM80-PTX70-LABEL: fptrunc_rp_f32_f64(
; SM80-PTX70:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_f32_f64_param_0];
; SM80-PTX70-NEXT:  cvt.rp.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b32 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL:       fptrunc_rp_f32_f64(
; SM80:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_f32_f64_param_0];
; SM80-NOFTZ-NEXT:  cvt.rp.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-FTZ-NEXT:    cvt.rp.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM80-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM80-NEXT:        ret;
;
; SM90-LABEL:       fptrunc_rp_f32_f64(
; SM90:             ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_f32_f64_param_0];
; SM90-NOFTZ-NEXT:  cvt.rp.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM90-FTZ-NEXT:    cvt.rp.ftz.f32.f64 [[RES:%r[0-9]+]], [[A]];
; SM90-NEXT:        st.param.b32 [func_retval0], [[RES]];
; SM90-NEXT:        ret;
  %res = call float @llvm.fptrunc.round.f32.f64(double %a, metadata !"round.upward")
  ret float %res
}

define bfloat @cvt_rn_bf16_f16(half %a) {
; SM70-LABEL:       cvt_rn_bf16_f16(
; SM70:             ld.param.b16 [[RS1:%rs[0-9]+]], [cvt_rn_bf16_f16_param_0];
; SM70-NOFTZ-NEXT:  cvt.f32.f16 [[R1:%r[0-9]+]], [[RS1]];
; SM70-FTZ-NEXT:    cvt.ftz.f32.f16 [[R1:%r[0-9]+]], [[RS1]];
; SM70-NEXT:        bfe.u32 [[R2:%r[0-9]+]], [[R1]], 16, 1;
; SM70-NEXT:        add.s32 [[R3:%r[0-9]+]], [[R2]], [[R1]];
; SM70-NEXT:        add.s32 [[R4:%r[0-9]+]], [[R3]], 32767;
; SM70-NOFTZ-NEXT:  setp.nan.f32 [[P1:%p[0-9]+]], [[R1]], [[R1]];
; SM70-FTZ-NEXT:    setp.nan.ftz.f32 [[P1:%p[0-9]+]], [[R1]], [[R1]];
; SM70-NEXT:        or.b32 [[R5:%r[0-9]+]], [[R1]], 4194304;
; SM70-NEXT:        selp.b32 [[R6:%r[0-9]+]], [[R5]], [[R4]], [[P1]];
; SM70-NEXT:        shr.u32 [[R7:%r[0-9]+]], [[R6]], 16;
; SM70-NEXT:        st.param.b16 [func_retval0], [[R7]];
; SM70-NEXT:        ret;
;
; SM80-PTX70-LABEL: cvt_rn_bf16_f16(
; SM80-PTX70:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rn_bf16_f16_param_0];
; SM80-PTX70-NEXT:  cvt.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  cvt.rn.bf16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL:       cvt_rn_bf16_f16(
; SM80:             ld.param.b16 [[A:%rs[0-9]+]], [cvt_rn_bf16_f16_param_0];
; SM80-NOFTZ-NEXT:  cvt.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-FTZ-NEXT:    cvt.ftz.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-NEXT:        cvt.rn.bf16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-NEXT:        st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:        ret;
;
; SM90-LABEL: cvt_rn_bf16_f16(
; SM90:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rn_bf16_f16_param_0];
; SM90-NEXT:  cvt.rn.bf16.f16 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %ext = fpext half %a to float
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %ext, metadata !"round.tonearest")
  ret bfloat %res
}

define bfloat @fptrunc_rn_bf16_f32(float %a) {
; SM70-LABEL:       fptrunc_rn_bf16_f32(
; SM70:             ld.param.b32 [[R1:%r[0-9]+]], [fptrunc_rn_bf16_f32_param_0];
; SM70-NEXT:        bfe.u32 [[R2:%r[0-9]+]], [[R1]], 16, 1;
; SM70-NEXT:        add.s32 [[R3:%r[0-9]+]], [[R2]], [[R1]];
; SM70-NEXT:        add.s32 [[R4:%r[0-9]+]], [[R3]], 32767;
; SM70-NOFTZ-NEXT:  setp.nan.f32 [[P1:%p[0-9]+]], [[R1]], [[R1]];
; SM70-FTZ-NEXT:    setp.nan.ftz.f32 [[P1:%p[0-9]+]], [[R1]], [[R1]];
; SM70-NEXT:        or.b32 [[R5:%r[0-9]+]], [[R1]], 4194304;
; SM70-NEXT:        selp.b32 [[R6:%r[0-9]+]], [[R5]], [[R4]], [[P1]];
; SM70-NEXT:        shr.u32 [[R7:%r[0-9]+]], [[R6]], 16;
; SM70-NEXT:        st.param.b16 [func_retval0], [[R7]];
; SM70-NEXT:        ret;
;
; SM80-PTX70-LABEL: fptrunc_rn_bf16_f32(
; SM80-PTX70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rn_bf16_f32_param_0];
; SM80-PTX70-NEXT:  cvt.rn.bf16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rn_bf16_f32(
; SM80:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rn_bf16_f32_param_0];
; SM80-NEXT:  cvt.rn.bf16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rn_bf16_f32(
; SM90:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rn_bf16_f32_param_0];
; SM90-NEXT:  cvt.rn.bf16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %a, metadata !"round.tonearest")
  ret bfloat %res
}

define bfloat @fptrunc_rn_bf16_f64(double %a) {
; SM70-LABEL:       fptrunc_rn_bf16_f64(
; SM70:             ld.param.b64 [[RD1:%rd[0-9]+]], [fptrunc_rn_bf16_f64_param_0];
; SM70-NEXT:        abs.f64 [[RD2:%rd[0-9]+]], [[RD1]];
; SM70-NOFTZ-NEXT:  cvt.rn.f32.f64 [[R1:%r[0-9]+]], [[RD1]];
; SM70-NOFTZ-NEXT:  cvt.f64.f32 [[RD3:%rd[0-9]+]], [[R1]];
; SM70-FTZ-NEXT:    cvt.rn.ftz.f32.f64 [[R1:%r[0-9]+]], [[RD1]];
; SM70-FTZ-NEXT:    cvt.ftz.f64.f32 [[RD3:%rd[0-9]+]], [[R1]];
; SM70-NEXT:        abs.f64 [[RD4:%rd[0-9]+]], [[RD3]];
; SM70-NEXT:        setp.gt.f64 [[P1:%p[0-9]+]], [[RD2]], [[RD4]];
; SM70-NEXT:        selp.b32 [[R2:%r[0-9]+]], 1, -1, [[P1]];
; SM70-NEXT:        add.s32 [[R3:%r[0-9]+]], [[R1]], [[R2]];
; SM70-NEXT:        and.b32 [[R4:%r[0-9]+]], [[R1]], 1;
; SM70-NEXT:        setp.ne.b32 [[P2:%p[0-9]+]], [[R4]], 0;
; SM70-NEXT:        selp.b32 [[R5:%r[0-9]+]], [[R1]], [[R3]], [[P2]];
; SM70-NEXT:        setp.equ.f64 [[P3:%p[0-9]+]], [[RD1]], [[RD3]];
; SM70-NEXT:        selp.b32 [[R6:%r[0-9]+]], [[R1]], [[R5]], [[P3]];
; SM70-NEXT:        bfe.u32 [[R7:%r[0-9]+]], [[R6]], 16, 1;
; SM70-NEXT:        add.s32 [[R8:%r[0-9]+]], [[R7]], [[R6]];
; SM70-NEXT:        add.s32 [[R9:%r[0-9]+]], [[R8]], 32767;
; SM70-NEXT:        or.b32 [[R10:%r[0-9]+]], [[R6]], 4194304;
; SM70-NEXT:        setp.nan.f64 [[P4:%p[0-9]+]], [[RD1]], [[RD1]];
; SM70-NEXT:        selp.b32 [[R11:%r[0-9]+]], [[R10]], [[R9]], [[P4]];
; SM70-NEXT:        shr.u32 [[R12:%r[0-9]+]], [[R11]], 16;
; SM70-NEXT:        st.param.b16 [func_retval0], [[R12]];
; SM70-NEXT:        ret;
;
; SM80-PTX70-LABEL: fptrunc_rn_bf16_f64(
; SM80-PTX70:       ld.param.b64 [[RD1:%rd[0-9]+]], [fptrunc_rn_bf16_f64_param_0];
; SM80-PTX70-NEXT:  abs.f64 [[RD2:%rd[0-9]+]], [[RD1]];
; SM80-PTX70-NEXT:  cvt.rn.f32.f64 [[R1:%r[0-9]+]], [[RD1]];
; SM80-PTX70-NEXT:  cvt.f64.f32 [[RD3:%rd[0-9]+]], [[R1]];
; SM80-PTX70-NEXT:  abs.f64 [[RD4:%rd[0-9]+]], [[RD3]];
; SM80-PTX70-NEXT:  setp.gt.f64 [[P1:%p[0-9]+]], [[RD2]], [[RD4]];
; SM80-PTX70-NEXT:  selp.b32 [[R2:%r[0-9]+]], 1, -1, [[P1]];
; SM80-PTX70-NEXT:  add.s32 [[R3:%r[0-9]+]], [[R1]], [[R2]];
; SM80-PTX70-NEXT:  and.b32 [[R4:%r[0-9]+]], [[R1]], 1;
; SM80-PTX70-NEXT:  setp.ne.b32 [[P2:%p[0-9]+]], [[R4]], 0;
; SM80-PTX70-NEXT:  selp.b32 [[R5:%r[0-9]+]], [[R1]], [[R3]], [[P2]];
; SM80-PTX70-NEXT:  setp.equ.f64 [[P3:%p[0-9]+]], [[RD1]], [[RD3]];
; SM80-PTX70-NEXT:  selp.b32 [[R6:%r[0-9]+]], [[R1]], [[R5]], [[P3]];
; SM80-PTX70-NEXT:  cvt.rn.bf16.f32 [[RS1:%rs[0-9]+]], [[R6]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RS1]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL:       fptrunc_rn_bf16_f64(
; SM80:             ld.param.b64 [[RD1:%rd[0-9]+]], [fptrunc_rn_bf16_f64_param_0];
; SM80-NEXT:        abs.f64 [[RD2:%rd[0-9]+]], [[RD1]];
; SM80-NOFTZ-NEXT:  cvt.rn.f32.f64 [[R1:%r[0-9]+]], [[RD1]];
; SM80-NOFTZ-NEXT:  cvt.f64.f32 [[RD3:%rd[0-9]+]], [[R1]];
; SM80-FTZ-NEXT:    cvt.rn.ftz.f32.f64 [[R1:%r[0-9]+]], [[RD1]];
; SM80-FTZ-NEXT:    cvt.ftz.f64.f32 [[RD3:%rd[0-9]+]], [[R1]];
; SM80-NEXT:        abs.f64 [[RD4:%rd[0-9]+]], [[RD3]];
; SM80-NEXT:        setp.gt.f64 [[P1:%p[0-9]+]], [[RD2]], [[RD4]];
; SM80-NEXT:        selp.b32 [[R2:%r[0-9]+]], 1, -1, [[P1]];
; SM80-NEXT:        add.s32 [[R3:%r[0-9]+]], [[R1]], [[R2]];
; SM80-NEXT:        and.b32 [[R4:%r[0-9]+]], [[R1]], 1;
; SM80-NEXT:        setp.ne.b32 [[P2:%p[0-9]+]], [[R4]], 0;
; SM80-NEXT:        selp.b32 [[R5:%r[0-9]+]], [[R1]], [[R3]], [[P2]];
; SM80-NEXT:        setp.equ.f64 [[P3:%p[0-9]+]], [[RD1]], [[RD3]];
; SM80-NEXT:        selp.b32 [[R6:%r[0-9]+]], [[R1]], [[R5]], [[P3]];
; SM80-NEXT:        cvt.rn.bf16.f32 [[RS1:%rs[0-9]+]], [[R6]];
; SM80-NEXT:        st.param.b16 [func_retval0], [[RS1]];
; SM80-NEXT:        ret;
;
; SM90-LABEL: fptrunc_rn_bf16_f64(
; SM90:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rn_bf16_f64_param_0];
; SM90-NEXT:  cvt.rn.bf16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call bfloat @llvm.fptrunc.round.bf16.f64(double %a, metadata !"round.tonearest")
  ret bfloat %res
}

;--- sm80.ll

define bfloat @cvt_rz_bf16_f16(half %a) {
; SM70-ERR: error: {{.*}}in function cvt_rz_bf16_f16{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.towardzero requires sm_80 or higher
;
; SM80-PTX70-LABEL: cvt_rz_bf16_f16(
; SM80-PTX70:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rz_bf16_f16_param_0];
; SM80-PTX70-NEXT:  cvt.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  cvt.rz.bf16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL:       cvt_rz_bf16_f16(
; SM80:             ld.param.b16 [[A:%rs[0-9]+]], [cvt_rz_bf16_f16_param_0];
; SM80-NOFTZ-NEXT:  cvt.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-FTZ-NEXT:    cvt.ftz.f32.f16 [[EXT:%r[0-9]+]], [[A]];
; SM80-NEXT:        cvt.rz.bf16.f32 [[RES:%rs[0-9]+]], [[EXT]];
; SM80-NEXT:        st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:        ret;
;
; SM90-LABEL: cvt_rz_bf16_f16(
; SM90:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rz_bf16_f16_param_0];
; SM90-NEXT:  cvt.rz.bf16.f16 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %ext = fpext half %a to float
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %ext, metadata !"round.towardzero")
  ret bfloat %res
}

define bfloat @fptrunc_rz_bf16_f32(float %a) {
; SM70-ERR: error: {{.*}}in function fptrunc_rz_bf16_f32{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.towardzero requires sm_80 or higher
;
; SM80-PTX70-LABEL: fptrunc_rz_bf16_f32(
; SM80-PTX70:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rz_bf16_f32_param_0];
; SM80-PTX70-NEXT:  cvt.rz.bf16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-PTX70-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-PTX70-NEXT:  ret;
;
; SM80-LABEL: fptrunc_rz_bf16_f32(
; SM80:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rz_bf16_f32_param_0];
; SM80-NEXT:  cvt.rz.bf16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM80-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM80-NEXT:  ret;
;
; SM90-LABEL: fptrunc_rz_bf16_f32(
; SM90:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rz_bf16_f32_param_0];
; SM90-NEXT:  cvt.rz.bf16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %a, metadata !"round.towardzero")
  ret bfloat %res
}

;--- sm90.ll
define bfloat @cvt_rm_bf16_f16(half %a) {
; SM70-ERR: error: {{.*}}in function cvt_rm_bf16_f16{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.downward requires sm_90 or higher
; SM80-ERR: error: {{.*}}in function cvt_rm_bf16_f16{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.downward requires sm_90 or higher
;
; SM90-LABEL: cvt_rm_bf16_f16(
; SM90:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rm_bf16_f16_param_0];
; SM90-NEXT:  cvt.rm.bf16.f16 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %ext = fpext half %a to float
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %ext, metadata !"round.downward")
  ret bfloat %res
}

define bfloat @cvt_rp_bf16_f16(half %a) {
; SM70-ERR: error: {{.*}}in function cvt_rp_bf16_f16{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.upward requires sm_90 or higher
; SM80-ERR: error: {{.*}}in function cvt_rp_bf16_f16{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.upward requires sm_90 or higher
;
; SM90-LABEL: cvt_rp_bf16_f16(
; SM90:       ld.param.b16 [[A:%rs[0-9]+]], [cvt_rp_bf16_f16_param_0];
; SM90-NEXT:  cvt.rp.bf16.f16 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %ext = fpext half %a to float
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %ext, metadata !"round.upward")
  ret bfloat %res
}

define bfloat @fptrunc_rm_bf16_f32(float %a) {
; SM70-ERR: error: {{.*}}in function fptrunc_rm_bf16_f32{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.downward requires sm_90 or higher
; SM80-ERR: error: {{.*}}in function fptrunc_rm_bf16_f32{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.downward requires sm_90 or higher
;
; SM90-LABEL: fptrunc_rm_bf16_f32(
; SM90:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rm_bf16_f32_param_0];
; SM90-NEXT:  cvt.rm.bf16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %a, metadata !"round.downward")
  ret bfloat %res
}

define bfloat @fptrunc_rp_bf16_f32(float %a) {
; SM70-ERR: error: {{.*}}in function fptrunc_rp_bf16_f32{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.upward requires sm_90 or higher
; SM80-ERR: error: {{.*}}in function fptrunc_rp_bf16_f32{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.upward requires sm_90 or higher
;
; SM90-LABEL: fptrunc_rp_bf16_f32(
; SM90:       ld.param.b32 [[A:%r[0-9]+]], [fptrunc_rp_bf16_f32_param_0];
; SM90-NEXT:  cvt.rp.bf16.f32 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %a, metadata !"round.upward")
  ret bfloat %res
}

define bfloat @fptrunc_rz_bf16_f64(double %a) {
; SM70-ERR: error: {{.*}}in function fptrunc_rz_bf16_f64{{.*}}llvm.fptrunc.round from f64 to bf16 with rounding mode round.towardzero requires sm_90 or higher
; SM80-ERR: error: {{.*}}in function fptrunc_rz_bf16_f64{{.*}}llvm.fptrunc.round from f64 to bf16 with rounding mode round.towardzero requires sm_90 or higher
;
; SM90-LABEL: fptrunc_rz_bf16_f64(
; SM90:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rz_bf16_f64_param_0];
; SM90-NEXT:  cvt.rz.bf16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call bfloat @llvm.fptrunc.round.bf16.f64(double %a, metadata !"round.towardzero")
  ret bfloat %res
}

define bfloat @fptrunc_rm_bf16_f64(double %a) {
; SM70-ERR: error: {{.*}}in function fptrunc_rm_bf16_f64{{.*}}llvm.fptrunc.round from f64 to bf16 with rounding mode round.downward requires sm_90 or higher
; SM80-ERR: error: {{.*}}in function fptrunc_rm_bf16_f64{{.*}}llvm.fptrunc.round from f64 to bf16 with rounding mode round.downward requires sm_90 or higher
;
; SM90-LABEL: fptrunc_rm_bf16_f64(
; SM90:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rm_bf16_f64_param_0];
; SM90-NEXT:  cvt.rm.bf16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call bfloat @llvm.fptrunc.round.bf16.f64(double %a, metadata !"round.downward")
  ret bfloat %res
}

define bfloat @fptrunc_rp_bf16_f64(double %a) {
; SM70-ERR: error: {{.*}}in function fptrunc_rp_bf16_f64{{.*}}llvm.fptrunc.round from f64 to bf16 with rounding mode round.upward requires sm_90 or higher
; SM80-ERR: error: {{.*}}in function fptrunc_rp_bf16_f64{{.*}}llvm.fptrunc.round from f64 to bf16 with rounding mode round.upward requires sm_90 or higher
;
; SM90-LABEL: fptrunc_rp_bf16_f64(
; SM90:       ld.param.b64 [[A:%rd[0-9]+]], [fptrunc_rp_bf16_f64_param_0];
; SM90-NEXT:  cvt.rp.bf16.f64 [[RES:%rs[0-9]+]], [[A]];
; SM90-NEXT:  st.param.b16 [func_retval0], [[RES]];
; SM90-NEXT:  ret;
  %res = call bfloat @llvm.fptrunc.round.bf16.f64(double %a, metadata !"round.upward")
  ret bfloat %res
}

;--- unsupported-rounding.ll
declare half @llvm.fptrunc.round.f16.f32(float, metadata)
declare bfloat @llvm.fptrunc.round.bf16.f32(float, metadata)
declare float @llvm.fptrunc.round.f32.f64(double, metadata)

define half @fptrunc_rna_f16_f32(float %a) {
; ROUND-ERR: error: {{.*}}in function fptrunc_rna_f16_f32{{.*}}llvm.fptrunc.round from f32 to f16 with rounding mode round.tonearestaway is not supported on this target
  %res = call half @llvm.fptrunc.round.f16.f32(float %a, metadata !"round.tonearestaway")
  ret half %res
}

define bfloat @fptrunc_rna_bf16_f32(float %a) {
; ROUND-ERR: error: {{.*}}in function fptrunc_rna_bf16_f32{{.*}}llvm.fptrunc.round from f32 to bf16 with rounding mode round.tonearestaway is not supported on this target
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %a, metadata !"round.tonearestaway")
  ret bfloat %res
}

define float @fptrunc_rna_f32_f64(double %a) {
; ROUND-ERR: error: {{.*}}in function fptrunc_rna_f32_f64{{.*}}llvm.fptrunc.round from f64 to f32 with rounding mode round.tonearestaway is not supported on this target
  %res = call float @llvm.fptrunc.round.f32.f64(double %a, metadata !"round.tonearestaway")
  ret float %res
}
