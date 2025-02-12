; RUN: llc -mtriple=amdgcn -mcpu=gfx942 --stop-after=si-fix-sgpr-copies < %s | FileCheck %s

; iglp.opt should not be flagged as clobbering the memory operand for the global_load, and we should be able to
; lower into the scalar version (i.e. should not need to lower into vector version with waterfall loop)
; CHECK-NOT: WATERFALL

define amdgpu_kernel void @_attn_forward_fp8e5_128x32x64_BW128(ptr addrspace(1) %in, ptr addrspace(3) %out) {
.lr.ph:
  br label %1

1:                                                ; preds = %1, %.lr.ph
  %addr = phi ptr addrspace(1) [ null, %.lr.ph ], [ %gep, %1 ]
  %offset = phi i64 [ 0, %.lr.ph ], [ %nextOff, %1 ]
  %inc = phi i32 [0, %.lr.ph], [ %incCond, %1 ] 
  %rsrc = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1) %addr, i16 0, i32 0, i32 0)
  %load = tail call <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  %load.bc = bitcast <2 x i32> %load to <8 x i8>
  %load.elem = extractelement <8 x i8> %load.bc, i64 0
  tail call void @llvm.amdgcn.iglp.opt(i32 0)
  %vec = insertelement <4 x i8> zeroinitializer, i8 %load.elem, i64 0
  %vec.bc = bitcast <4 x i8> %vec to <2 x half>
  %shuff = shufflevector <2 x half> %vec.bc, <2 x half> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %gep = getelementptr i8, ptr addrspace(1) %in, i64 %offset
  %unmaskedload49 = load <1 x i64>, ptr addrspace(1) null, align 8
  %nextOff = extractelement <1 x i64> %unmaskedload49, i64 0
  %incCond = add i32 %inc, 1
  %cond = icmp eq i32 %incCond, 32
  br i1 %cond, label %2, label %1 

2:
  store <4 x half> %shuff, ptr addrspace(3) %out, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1) readnone, i16, i32, i32) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg) #1
