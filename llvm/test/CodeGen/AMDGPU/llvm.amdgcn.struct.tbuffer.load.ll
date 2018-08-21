;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefix=GCN %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}tbuffer_load:
; GCN: v_mov_b32_e32 [[ZEROREG:v[0-9]+]], 0
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 idxen
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, dfmt:15, nfmt:3, 0 idxen glc
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, dfmt:6, nfmt:1, 0 idxen slc
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, dfmt:6, nfmt:1, 0 idxen
; GCN: s_waitcnt
define amdgpu_vs {<4 x float>, <4 x float>, <4 x float>, <4 x float>} @tbuffer_load(<4 x i32> inreg) {
main_body:
    %vdata     = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 78, i32 0)
    %vdata_glc = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 63, i32 1)
    %vdata_slc = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 22, i32 2)
    %vdata_f32 = call <4 x float> @llvm.amdgcn.struct.tbuffer.load.v4f32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 22, i32 0)
    %vdata.f     = bitcast <4 x i32> %vdata to <4 x float>
    %vdata_glc.f = bitcast <4 x i32> %vdata_glc to <4 x float>
    %vdata_slc.f = bitcast <4 x i32> %vdata_slc to <4 x float>
    %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %vdata.f, 0
    %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %vdata_glc.f, 1
    %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %vdata_slc.f, 2
    %r3 = insertvalue {<4 x float>, <4 x float>, <4 x float>, <4 x float>} %r2, <4 x float> %vdata_f32, 3
    ret {<4 x float>, <4 x float>, <4 x float>, <4 x float>} %r3
}

; GCN-LABEL: {{^}}tbuffer_load_immoffs:
; GCN: v_mov_b32_e32 [[ZEROREG:v[0-9]+]], 0
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 idxen offset:42
define amdgpu_vs <4 x float> @tbuffer_load_immoffs(<4 x i32> inreg) {
main_body:
    %vdata   = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 42, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <4 x i32> %vdata to <4 x float>
    ret <4 x float> %vdata.f
}

; GCN-LABEL: {{^}}tbuffer_load_immoffs_large
; GCN: v_mov_b32_e32 [[ZEROREG:v[0-9]+]], 0
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, dfmt:15, nfmt:2, 61 idxen offset:4095
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:3, {{s[0-9]+}} idxen offset:73
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, dfmt:13, nfmt:4, {{s[0-9]+}} idxen offset:1
; GCN: s_waitcnt
define amdgpu_vs {<4 x float>, <4 x float>, <4 x float>} @tbuffer_load_immoffs_large(<4 x i32> inreg, i32 inreg %soffs) {
    %vdata     = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 4095, i32 61, i32 47, i32 0)
    %vdata_glc = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 73, i32 %soffs, i32 62, i32 0)
    %vdata_slc = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 1, i32 %soffs, i32 77, i32 0)
    %vdata.f     = bitcast <4 x i32> %vdata to <4 x float>
    %vdata_glc.f = bitcast <4 x i32> %vdata_glc to <4 x float>
    %vdata_slc.f = bitcast <4 x i32> %vdata_slc to <4 x float>
    %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %vdata.f, 0
    %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %vdata_glc.f, 1
    %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %vdata_slc.f, 2
    ret {<4 x float>, <4 x float>, <4 x float>} %r2
}

; GCN-LABEL: {{^}}tbuffer_load_idx:
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 idxen
define amdgpu_vs <4 x float> @tbuffer_load_idx(<4 x i32> inreg, i32 %vindex) {
main_body:
    %vdata   = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 %vindex, i32 0, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <4 x i32> %vdata to <4 x float>
    ret <4 x float> %vdata.f
}

; GCN-LABEL: {{^}}tbuffer_load_ofs:
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 idxen offen
define amdgpu_vs <4 x float> @tbuffer_load_ofs(<4 x i32> inreg, i32 %voffs) {
main_body:
    %vdata   = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 %voffs, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <4 x i32> %vdata to <4 x float>
    ret <4 x float> %vdata.f
}

; GCN-LABEL: {{^}}tbuffer_load_ofs_imm:
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 idxen offen offset:52
define amdgpu_vs <4 x float> @tbuffer_load_ofs_imm(<4 x i32> inreg, i32 %voffs) {
main_body:
    %ofs = add i32 %voffs, 52
    %vdata   = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 %ofs, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <4 x i32> %vdata to <4 x float>
    ret <4 x float> %vdata.f
}

; GCN-LABEL: {{^}}tbuffer_load_both:
; GCN: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 idxen offen
define amdgpu_vs <4 x float> @tbuffer_load_both(<4 x i32> inreg, i32 %vindex, i32 %voffs) {
main_body:
    %vdata   = call <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32> %0, i32 %vindex, i32 %voffs, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <4 x i32> %vdata to <4 x float>
    ret <4 x float> %vdata.f
}


; GCN-LABEL: {{^}}buffer_load_xy:
; GCN: tbuffer_load_format_xy {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:13, nfmt:4, 0 idxen
define amdgpu_vs <2 x float> @buffer_load_xy(<4 x i32> inreg %rsrc) {
    %vdata = call <2 x i32> @llvm.amdgcn.struct.tbuffer.load.v2i32(<4 x i32> %rsrc, i32 0, i32 0, i32 0, i32 77, i32 0)
    %vdata.f = bitcast <2 x i32> %vdata to <2 x float>
    ret <2 x float> %vdata.f
}

; GCN-LABEL: {{^}}buffer_load_x:
; GCN: tbuffer_load_format_x {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:13, nfmt:4, 0 idxen
define amdgpu_vs float @buffer_load_x(<4 x i32> inreg %rsrc) {
    %vdata = call i32 @llvm.amdgcn.struct.tbuffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0, i32 0, i32 77, i32 0)
    %vdata.f = bitcast i32 %vdata to float
    ret float %vdata.f
}

declare i32 @llvm.amdgcn.struct.tbuffer.load.i32(<4 x i32>, i32, i32, i32, i32, i32)
declare <2 x i32> @llvm.amdgcn.struct.tbuffer.load.v2i32(<4 x i32>, i32, i32, i32, i32, i32)
declare <4 x i32> @llvm.amdgcn.struct.tbuffer.load.v4i32(<4 x i32>, i32, i32, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.struct.tbuffer.load.v4f32(<4 x i32>, i32, i32, i32, i32, i32)

