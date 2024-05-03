;RUN: llc < %s -mtriple=amdgcn -mcpu=verde -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=SICI
;RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=VI

;CHECK-LABEL: {{^}}test1:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_swap v0, off, s[0:3], 0 glc
;VI: s_movk_i32 [[SOFS:s[0-9]+]], 0x1ffc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, v2, s[0:3], 0 offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, v[1:2], s[0:3], 0 idxen offen glc
;SICI: v_mov_b32_e32 v1, 0x2000
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, v2, s[0:3], 0 offen offset:42 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;SICI: buffer_atomic_swap v0, v1, s[0:3], 0 offen glc
;VI: buffer_atomic_swap v0, off, s[0:3], [[SOFS]] offset:4 glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, off, s[0:3], 0{{$}}
define amdgpu_ps float @test1(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex, i32 %voffset) {
main_body:
  %o1 = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %o2 = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32 %o1, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %o3 = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32 %o2, <4 x i32> %rsrc, i32 0, i32 %voffset, i1 0)
  %o4 = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32 %o3, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i1 0)
  %ofs.5 = add i32 %voffset, 42
  %o5 = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32 %o4, <4 x i32> %rsrc, i32 0, i32 %ofs.5, i1 0)
  %o6 = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32 %o5, <4 x i32> %rsrc, i32 0, i32 8192, i1 0)
  %unused = call i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32 %o6, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %out = bitcast i32 %o6 to float
  ret float %out
}

;CHECK-LABEL: {{^}}test11:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_swap_x2 v[3:4], off, s[0:3], 0 glc
;VI: s_movk_i32 [[SOFS:s[0-9]+]], 0x1ffc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap_x2 v[3:4], v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap_x2 v[3:4], v2, s[0:3], 0 offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap_x2 v[3:4], v[1:2], s[0:3], 0 idxen offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap_x2 v[3:4], v2, s[0:3], 0 offen offset:42 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;SICI: buffer_atomic_swap_x2 v[3:4], v0, s[0:3], 0 offen glc
;VI: buffer_atomic_swap_x2 v[3:4], off, s[0:3], [[SOFS]] offset:4 glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap_x2 v[3:4], off, s[0:3], 0{{$}}
define amdgpu_ps float @test11(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex, i32 %voffset) {
main_body:
  %o0 = sext i32 %data to i64
  %o1 = call i64 @llvm.amdgcn.buffer.atomic.swap.i64(i64 %o0, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %o2 = call i64 @llvm.amdgcn.buffer.atomic.swap.i64(i64 %o1, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %o3 = call i64 @llvm.amdgcn.buffer.atomic.swap.i64(i64 %o2, <4 x i32> %rsrc, i32 0, i32 %voffset, i1 0)
  %o4 = call i64 @llvm.amdgcn.buffer.atomic.swap.i64(i64 %o3, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i1 0)
  %ofs.5 = add i32 %voffset, 42
  %o5 = call i64 @llvm.amdgcn.buffer.atomic.swap.i64(i64 %o4, <4 x i32> %rsrc, i32 0, i32 %ofs.5, i1 0)
  %o6 = call i64 @llvm.amdgcn.buffer.atomic.swap.i64(i64 %o5, <4 x i32> %rsrc, i32 0, i32 8192, i1 0)
  %unused = call i64 @llvm.amdgcn.buffer.atomic.swap.i64(i64 %o6, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %o7 = trunc i64 %o6 to i32
  %out = bitcast i32 %o7 to float
  ret float %out
}

;CHECK-LABEL: {{^}}test2:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_add v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_sub v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_smin v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_umin v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_smax v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_umax v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_and v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_or v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_xor v0, v1, s[0:3], 0 idxen glc
define amdgpu_ps float @test2(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex) {
main_body:
  %t1 = call i32 @llvm.amdgcn.buffer.atomic.add.i32(i32 %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t2 = call i32 @llvm.amdgcn.buffer.atomic.sub.i32(i32 %t1, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t3 = call i32 @llvm.amdgcn.buffer.atomic.smin.i32(i32 %t2, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t4 = call i32 @llvm.amdgcn.buffer.atomic.umin.i32(i32 %t3, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t5 = call i32 @llvm.amdgcn.buffer.atomic.smax.i32(i32 %t4, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t6 = call i32 @llvm.amdgcn.buffer.atomic.umax.i32(i32 %t5, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t7 = call i32 @llvm.amdgcn.buffer.atomic.and.i32(i32 %t6, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t8 = call i32 @llvm.amdgcn.buffer.atomic.or.i32(i32 %t7, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t9 = call i32 @llvm.amdgcn.buffer.atomic.xor.i32(i32 %t8, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %out = bitcast i32 %t9 to float
  ret float %out
}

;CHECK-LABEL: {{^}}test3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_add_x2 v[0:1], v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_sub_x2 v[0:1], v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_smin_x2 v[0:1], v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_umin_x2 v[0:1], v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_smax_x2 v[0:1], v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_umax_x2 v[0:1], v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_and_x2 v[0:1], v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_or_x2 v[0:1], v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_xor_x2 v[0:1], v2, s[0:3], 0 idxen glc
define amdgpu_ps float @test3(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex) {
main_body:
  %t0 = sext i32 %data to i64
  %t1 = call i64 @llvm.amdgcn.buffer.atomic.add.i64(i64 %t0, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t2 = call i64 @llvm.amdgcn.buffer.atomic.sub.i64(i64 %t1, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t3 = call i64 @llvm.amdgcn.buffer.atomic.smin.i64(i64 %t2, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t4 = call i64 @llvm.amdgcn.buffer.atomic.umin.i64(i64 %t3, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t5 = call i64 @llvm.amdgcn.buffer.atomic.smax.i64(i64 %t4, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t6 = call i64 @llvm.amdgcn.buffer.atomic.umax.i64(i64 %t5, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t7 = call i64 @llvm.amdgcn.buffer.atomic.and.i64(i64 %t6, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t8 = call i64 @llvm.amdgcn.buffer.atomic.or.i64(i64 %t7, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t9 = call i64 @llvm.amdgcn.buffer.atomic.xor.i64(i64 %t8, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %t10 = trunc i64 %t9 to i32
  %out = bitcast i32 %t10 to float
  ret float %out
}

; Ideally, we would teach tablegen & friends that cmpswap only modifies the
; first vgpr. Since we don't do that yet, the register allocator will have to
; create copies which we don't bother to track here.
;
;CHECK-LABEL: {{^}}test4:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 glc
;CHECK: s_waitcnt vmcnt(0)
;VI: s_movk_i32 [[SOFS:s[0-9]+]], 0x1ffc
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, v2, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, v3, s[0:3], 0 offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, v[2:3], s[0:3], 0 idxen offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, v3, s[0:3], 0 offen offset:44 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;SICI: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, s[0:3], 0 offen glc
;VI: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], [[SOFS]] offset:4 glc
define amdgpu_ps float @test4(<4 x i32> inreg %rsrc, i32 %data, i32 %cmp, i32 %vindex, i32 %voffset) {
main_body:
  %o1 = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32 %data, i32 %cmp, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %o2 = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32 %o1, i32 %cmp, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  %o3 = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32 %o2, i32 %cmp, <4 x i32> %rsrc, i32 0, i32 %voffset, i1 0)
  %o4 = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32 %o3, i32 %cmp, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i1 0)
  %ofs.5 = add i32 %voffset, 44
  %o5 = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32 %o4, i32 %cmp, <4 x i32> %rsrc, i32 0, i32 %ofs.5, i1 0)
  %o6 = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32 %o5, i32 %cmp, <4 x i32> %rsrc, i32 0, i32 8192, i1 0)

; Detecting the no-return variant doesn't work right now because of how the
; intrinsic is replaced by an instruction that feeds into an EXTRACT_SUBREG.
; Since there probably isn't a reasonable use-case of cmpswap that discards
; the return value, that seems okay.
;
;  %unused = call i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32 %o6, i32 %cmp, <4 x i32> %rsrc, i32 0, i32 0, i1 0)
  %out = bitcast i32 %o6 to float
  ret float %out
}

;CHECK-LABEL: {{^}}test7:
;CHECK: buffer_atomic_add v0,
define amdgpu_ps float @test7() {
main_body:
  %v = call i32 @llvm.amdgcn.buffer.atomic.add.i32(i32 1, <4 x i32> undef, i32 0, i32 4, i1 false)
  %v.float = bitcast i32 %v to float
  ret float %v.float
}

declare i32 @llvm.amdgcn.buffer.atomic.swap.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.add.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.sub.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.smin.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.umin.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.smax.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.umax.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.and.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.or.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.xor.i32(i32, <4 x i32>, i32, i32, i1) #0
declare i32 @llvm.amdgcn.buffer.atomic.cmpswap(i32, i32, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.swap.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.add.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.sub.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.smin.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.umin.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.smax.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.umax.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.and.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.or.i64(i64, <4 x i32>, i32, i32, i1) #0
declare i64 @llvm.amdgcn.buffer.atomic.xor.i64(i64, <4 x i32>, i32, i32, i1) #0

attributes #0 = { nounwind }
