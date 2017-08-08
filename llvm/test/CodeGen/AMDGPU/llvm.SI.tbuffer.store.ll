;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}test1:
;CHECK: tbuffer_store_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 offen offset:32 glc slc
define amdgpu_vs void @test1(i32 %a1, i32 %vaddr) {
    %vdata = insertelement <4 x i32> undef, i32 %a1, i32 0
    call void @llvm.SI.tbuffer.store.v4i32(<4 x i32> undef, <4 x i32> %vdata,
        i32 4, i32 %vaddr, i32 0, i32 32, i32 14, i32 4, i32 1, i32 0, i32 1,
        i32 1, i32 0)
    ret void
}

;CHECK-LABEL: {{^}}test1_idx:
;CHECK: tbuffer_store_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 idxen offset:32 glc slc
define amdgpu_vs void @test1_idx(i32 %a1, i32 %vaddr) {
    %vdata = insertelement <4 x i32> undef, i32 %a1, i32 0
    call void @llvm.SI.tbuffer.store.v4i32(<4 x i32> undef, <4 x i32> %vdata,
        i32 4, i32 %vaddr, i32 0, i32 32, i32 14, i32 4, i32 0, i32 1, i32 1,
        i32 1, i32 0)
    ret void
}

;CHECK-LABEL: {{^}}test1_scalar_offset:
;CHECK: tbuffer_store_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, {{s[0-9]+}} idxen offset:32 glc slc
define amdgpu_vs void @test1_scalar_offset(i32 %a1, i32 %vaddr, i32 inreg %soffset) {
    %vdata = insertelement <4 x i32> undef, i32 %a1, i32 0
    call void @llvm.SI.tbuffer.store.v4i32(<4 x i32> undef, <4 x i32> %vdata,
        i32 4, i32 %vaddr, i32 %soffset, i32 32, i32 14, i32 4, i32 0, i32 1, i32 1,
        i32 1, i32 0)
    ret void
}

;CHECK-LABEL: {{^}}test1_no_glc_slc:
;CHECK: tbuffer_store_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:14, nfmt:4, 0 offen offset:32
define amdgpu_vs void @test1_no_glc_slc(i32 %a1, i32 %vaddr) {
    %vdata = insertelement <4 x i32> undef, i32 %a1, i32 0
    call void @llvm.SI.tbuffer.store.v4i32(<4 x i32> undef, <4 x i32> %vdata,
        i32 4, i32 %vaddr, i32 0, i32 32, i32 14, i32 4, i32 1, i32 0, i32 0,
        i32 0, i32 0)
    ret void
}

;CHECK-LABEL: {{^}}test2:
;CHECK: tbuffer_store_format_xyz {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:13, nfmt:4, 0 offen offset:24 glc slc
define amdgpu_vs void @test2(i32 %a1, i32 %vaddr) {
    %vdata = insertelement <4 x i32> undef, i32 %a1, i32 0
    call void @llvm.SI.tbuffer.store.v4i32(<4 x i32> undef, <4 x i32> %vdata,
        i32 3, i32 %vaddr, i32 0, i32 24, i32 13, i32 4, i32 1, i32 0, i32 1,
        i32 1, i32 0)
    ret void
}

;CHECK-LABEL: {{^}}test3:
;CHECK: tbuffer_store_format_xy {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:11, nfmt:4, 0 offen offset:16 glc slc
define amdgpu_vs void @test3(i32 %a1, i32 %vaddr) {
    %vdata = insertelement <2 x i32> undef, i32 %a1, i32 0
    call void @llvm.SI.tbuffer.store.v2i32(<4 x i32> undef, <2 x i32> %vdata,
        i32 2, i32 %vaddr, i32 0, i32 16, i32 11, i32 4, i32 1, i32 0, i32 1,
        i32 1, i32 0)
    ret void
}

;CHECK-LABEL: {{^}}test4:
;CHECK: tbuffer_store_format_x {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, dfmt:4, nfmt:4, 0 offen offset:8 glc slc
define amdgpu_vs void @test4(i32 %vdata, i32 %vaddr) {
    call void @llvm.SI.tbuffer.store.i32(<4 x i32> undef, i32 %vdata,
        i32 1, i32 %vaddr, i32 0, i32 8, i32 4, i32 4, i32 1, i32 0, i32 1,
        i32 1, i32 0)
    ret void
}

declare void @llvm.SI.tbuffer.store.i32(<4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
declare void @llvm.SI.tbuffer.store.v2i32(<4 x i32>, <2 x i32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
declare void @llvm.SI.tbuffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
