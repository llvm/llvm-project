;RUN: llc < %s -march=amdgcn -mcpu=verde -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}test1:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_swap v0, {{v[0-9]+}}, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, {{v[0-9]+}}, s[0:3], 0 idxen glc
;CHECK: s_movk_i32 [[SOFS:s[0-9]+]], 0x1ffc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, {{v\[[0-9]+:[0-9]+\]}}, s[0:3], 0 idxen offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, {{v\[[0-9]+:[0-9]+\]}}, s[0:3], 0 idxen offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, v[1:2], s[0:3], 0 idxen offen offset:42 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, {{v[0-9]+}}, s[0:3], [[SOFS]] idxen offset:4 glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_swap v0, {{v[0-9]+}}, s[0:3], 0 idxen{{$}}
;CHECK: buffer_atomic_swap v0, {{v[0-9]+}}, s[0:3], 0 idxen glc
define amdgpu_ps float @test1(ptr addrspace(8) inreg %rsrc, i32 %data, i32 %vindex, i32 %voffset) {
main_body:
  %o1 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.swap.i32(i32 %data, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %o2 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.swap.i32(i32 %o1, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %o3 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.swap.i32(i32 %o2, ptr addrspace(8) %rsrc, i32 0, i32 %voffset, i32 0, i32 0)
  %o4 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.swap.i32(i32 %o3, ptr addrspace(8) %rsrc, i32 %vindex, i32 %voffset, i32 0, i32 0)
  %ofs.5 = add i32 %voffset, 42
  %o5 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.swap.i32(i32 %o4, ptr addrspace(8) %rsrc, i32 0, i32 %ofs.5, i32 0, i32 0)
  %o6 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.swap.i32(i32 %o5, ptr addrspace(8) %rsrc, i32 0, i32 4, i32 8188, i32 0)
  %unused = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.swap.i32(i32 %o6, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %o7 = bitcast i32 %o6 to float
  %out = call float @llvm.amdgcn.struct.ptr.buffer.atomic.swap.f32(float %o7, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  ret float %out
}

;CHECK-LABEL: {{^}}test2:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_add v0, v1, s[0:3], 0 idxen glc{{$}}
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_sub v0, v1, s[0:3], 0 idxen glc slc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_smin v0, v1, s[0:3], 0 idxen glc{{$}}
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_umin v0, v1, s[0:3], 0 idxen glc slc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_smax v0, v1, s[0:3], 0 idxen glc{{$}}
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_umax v0, v1, s[0:3], 0 idxen glc slc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_and v0, v1, s[0:3], 0 idxen glc{{$}}
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_or v0, v1, s[0:3], 0 idxen glc slc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_xor v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_inc v0, v1, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_dec v0, v1, s[0:3], 0 idxen glc
define amdgpu_ps float @test2(ptr addrspace(8) inreg %rsrc, i32 %data, i32 %vindex) {
main_body:
  %t1 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.add.i32(i32 %data, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %t2 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.sub.i32(i32 %t1, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 2)
  %t3 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.smin.i32(i32 %t2, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %t4 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.umin.i32(i32 %t3, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 2)
  %t5 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.smax.i32(i32 %t4, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %t6 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.umax.i32(i32 %t5, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 2)
  %t7 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.and.i32(i32 %t6, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %t8 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.or.i32(i32 %t7, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 2)
  %t9 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.xor.i32(i32 %t8, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %t10 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.inc.i32(i32 %t9, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %t11 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.dec.i32(i32 %t10, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %out = bitcast i32 %t11 to float
  ret float %out
}

; Ideally, we would teach tablegen & friends that cmpswap only modifies the
; first vgpr. Since we don't do that yet, the register allocator will have to
; create copies which we don't bother to track here.
;
;CHECK-LABEL: {{^}}test3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, s[0:3], 0 idxen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: s_movk_i32 [[SOFS:s[0-9]+]], 0x1ffc
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, s[0:3], 0 idxen offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, s[0:3], 0 idxen offen glc
;CHECK: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, s[0:3], 0 idxen offen offset:44 glc
;CHECK-DAG: s_waitcnt vmcnt(0)
;CHECK: buffer_atomic_cmpswap {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, s[0:3], [[SOFS]] idxen offset:4 glc
define amdgpu_ps float @test3(ptr addrspace(8) inreg %rsrc, i32 %data, i32 %cmp, i32 %vindex, i32 %voffset) {
main_body:
  %o1 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %data, i32 %cmp, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %o2 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %o1, i32 %cmp, ptr addrspace(8) %rsrc, i32 %vindex, i32 0, i32 0, i32 0)
  %o3 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %o2, i32 %cmp, ptr addrspace(8) %rsrc, i32 0, i32 %voffset, i32 0, i32 0)
  %o4 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %o3, i32 %cmp, ptr addrspace(8) %rsrc, i32 %vindex, i32 %voffset, i32 0, i32 0)
  %offs.5 = add i32 %voffset, 44
  %o5 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %o4, i32 %cmp, ptr addrspace(8) %rsrc, i32 0, i32 %offs.5, i32 0, i32 0)
  %o6 = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %o5, i32 %cmp, ptr addrspace(8) %rsrc, i32 0, i32 4, i32 8188, i32 0)

; Detecting the no-return variant doesn't work right now because of how the
; intrinsic is replaced by an instruction that feeds into an EXTRACT_SUBREG.
; Since there probably isn't a reasonable use-case of cmpswap that discards
; the return value, that seems okay.
;
;  %unused = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %o6, i32 %cmp, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %out = bitcast i32 %o6 to float
  ret float %out
}

;CHECK-LABEL: {{^}}test4:
;CHECK: buffer_atomic_add v0,
define amdgpu_ps float @test4() {
main_body:
  %v = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.add.i32(i32 1, ptr addrspace(8) undef, i32 0, i32 4, i32 0, i32 0)
  %v.float = bitcast i32 %v to float
  ret float %v.float
}

declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.swap.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare float @llvm.amdgcn.struct.ptr.buffer.atomic.swap.f32(float, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.add.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.sub.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.smin.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.umin.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.smax.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.umax.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.and.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.or.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.xor.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.inc.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.dec.i32(i32, ptr addrspace(8), i32, i32, i32, i32) #0
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32, i32, ptr addrspace(8), i32, i32, i32, i32) #0

attributes #0 = { nounwind }
