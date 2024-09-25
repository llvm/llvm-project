;RUN: llc < %s -mtriple=amdgcn -mcpu=verde -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}test1:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_swap v0, off, s[0:3], 0 glc
;CHECK: s_movk_i32 [[SOFS:s[0-9]+]], 0x1ffc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, v1, s[0:3], 0 offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, v1, s[0:3], 0 offen offset:42 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, off, s[0:3], [[SOFS]] offset:4 glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, off, s[0:3], 0{{$}}
;CHECK: buffer_atomic_swap v0, off, s[0:3], 0 glc
define amdgpu_ps float @test1(<4 x i32> inreg %rsrc, i32 %data, i32 %voffset) {
main_body:
  %o1 = call i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  %o3 = call i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32 %o1, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %off5 = add i32 %voffset, 42
  %o5 = call i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32 %o3, <4 x i32> %rsrc, i32 %off5, i32 0, i32 0)
  %o6 = call i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32 %o5, <4 x i32> %rsrc, i32 4, i32 8188, i32 0)
  %unused = call i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32 %o6, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  %o7 = bitcast i32 %o6 to float
  %out = call float @llvm.amdgcn.raw.buffer.atomic.swap.f32(float %o7, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  ret float %out
}

;CHECK-LABEL: {{^}}test2:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_add v0, v1, s[0:3], 0 offen glc{{$}}
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_sub v0, v1, s[0:3], 0 offen glc slc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_smin v0, v1, s[0:3], 0 offen glc{{$}}
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_umin v0, v1, s[0:3], 0 offen glc slc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_smax v0, v1, s[0:3], 0 offen glc{{$}}
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_umax v0, v1, s[0:3], 0 offen glc slc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_and v0, v1, s[0:3], 0 offen glc{{$}}
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_or v0, v1, s[0:3], 0 offen glc slc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_xor v0, v1, s[0:3], 0 offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_inc v0, v1, s[0:3], 0 offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_dec v0, v1, s[0:3], 0 offen glc
define amdgpu_ps float @test2(<4 x i32> inreg %rsrc, i32 %data, i32 %voffset) {
main_body:
  %t1 = call i32 @llvm.amdgcn.raw.buffer.atomic.add.i32(i32 %data, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %t2 = call i32 @llvm.amdgcn.raw.buffer.atomic.sub.i32(i32 %t1, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 2)
  %t3 = call i32 @llvm.amdgcn.raw.buffer.atomic.smin.i32(i32 %t2, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %t4 = call i32 @llvm.amdgcn.raw.buffer.atomic.umin.i32(i32 %t3, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 2)
  %t5 = call i32 @llvm.amdgcn.raw.buffer.atomic.smax.i32(i32 %t4, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %t6 = call i32 @llvm.amdgcn.raw.buffer.atomic.umax.i32(i32 %t5, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 2)
  %t7 = call i32 @llvm.amdgcn.raw.buffer.atomic.and.i32(i32 %t6, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %t8 = call i32 @llvm.amdgcn.raw.buffer.atomic.or.i32(i32 %t7, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 2)
  %t9 = call i32 @llvm.amdgcn.raw.buffer.atomic.xor.i32(i32 %t8, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %t10 = call i32 @llvm.amdgcn.raw.buffer.atomic.inc.i32(i32 %t9, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %t11 = call i32 @llvm.amdgcn.raw.buffer.atomic.dec.i32(i32 %t10, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %out = bitcast i32 %t11 to float
  ret float %out
}

; Ideally, we would teach tablegen & friends that cmpswap only modifies the
; first vgpr. Since we don't do that yet, the register allocator will have to
; create copies which we don't bother to track here.
;
;CHECK-LABEL: {{^}}test3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: s_movk_i32 [[SOFS:s[0-9]+]], 0x1ffc
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, v2, s[0:3], 0 offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, v2, s[0:3], 0 offen offset:44 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], [[SOFS]] offset:4 glc
define amdgpu_ps float @test3(<4 x i32> inreg %rsrc, i32 %data, i32 %cmp, i32 %vindex, i32 %voffset) {
main_body:
  %o1 = call i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i32(i32 %data, i32 %cmp, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  %o3 = call i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i32(i32 %o1, i32 %cmp, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %ofs.5 = add i32 %voffset, 44
  %o5 = call i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i32(i32 %o3, i32 %cmp, <4 x i32> %rsrc, i32 %ofs.5, i32 0, i32 0)
  %o6 = call i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i32(i32 %o5, i32 %cmp, <4 x i32> %rsrc, i32 4, i32 8188, i32 0)

; Detecting the no-return variant doesn't work right now because of how the
; intrinsic is replaced by an instruction that feeds into an EXTRACT_SUBREG.
; Since there probably isn't a reasonable use-case of cmpswap that discards
; the return value, that seems okay.
;
;  %unused = call i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i32(i32 %o6, i32 %cmp, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  %out = bitcast i32 %o6 to float
  ret float %out
}

;CHECK-LABEL: {{^}}test4:
;CHECK: buffer_atomic_add v0,
define amdgpu_ps float @test4() {
main_body:
  %v = call i32 @llvm.amdgcn.raw.buffer.atomic.add.i32(i32 1, <4 x i32> undef, i32 4, i32 0, i32 0)
  %v.float = bitcast i32 %v to float
  ret float %v.float
}

;CHECK-LABEL: {{^}}test5:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_cmpswap_x2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;CHECK-DAG: s_movk_i32 [[SOFS:s[0-9]+]], 0x1ffc
;CHECK: buffer_atomic_cmpswap_x2 {{v\[[0-9]+:[0-9]+\]}}, v4, s[0:3], 0 offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap_x2 {{v\[[0-9]+:[0-9]+\]}}, v4, s[0:3], 0 offen offset:44 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap_x2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], [[SOFS]] offset:4 glc
define amdgpu_ps float @test5(<4 x i32> inreg %rsrc, i64 %data, i64 %cmp, i32 %vindex, i32 %voffset) {
main_body:
  %o1 = call i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64 %data, i64 %cmp, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  %o3 = call i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64 %o1, i64 %cmp, <4 x i32> %rsrc, i32 %voffset, i32 0, i32 0)
  %ofs.5 = add i32 %voffset, 44
  %o5 = call i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64 %o3, i64 %cmp, <4 x i32> %rsrc, i32 %ofs.5, i32 0, i32 0)
  %o6 = call i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64 %o5, i64 %cmp, <4 x i32> %rsrc, i32 4, i32 8188, i32 0)
  %out = sitofp i64 %o6 to float
  ret float %out
}

declare i32 @llvm.amdgcn.raw.buffer.atomic.swap.i32(i32, <4 x i32>, i32, i32, i32) #0
declare float @llvm.amdgcn.raw.buffer.atomic.swap.f32(float, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.add.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.sub.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.smin.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.umin.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.smax.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.umax.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.and.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.or.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.xor.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.inc.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.dec.i32(i32, <4 x i32>, i32, i32, i32) #0
declare i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i32(i32, i32, <4 x i32>, i32, i32, i32) #0
declare i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64, i64, <4 x i32>, i32, i32, i32) #0

attributes #0 = { nounwind }
