--- |
  ; ModuleID = '../llvm/test/CodeGen/AMDGPU/carryout-selection.ll'
  source_filename = "../llvm/test/CodeGen/AMDGPU/carryout-selection.ll"
  target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
  target triple = "amdgcn"
  
  define amdgpu_kernel void @sadd64rr(ptr addrspace(1) %out, i64 %a, i64 %b) #0 {
  entry:
    %sadd64rr.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %sadd64rr.kernarg.segment, i64 36, !amdgpu.uniform !0
    %0 = load <3 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <3 x i64> %0, i32 0
    %1 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %a.load2 = extractelement <3 x i64> %0, i32 1
    %b.load3 = extractelement <3 x i64> %0, i32 2
    %add = add i64 %a.load2, %b.load3
    store i64 %add, ptr addrspace(1) %1, align 8
    ret void
  }
  
  define amdgpu_kernel void @sadd64ri(ptr addrspace(1) %out, i64 %a) {
  entry:
    %sadd64ri.kernarg.segment = call nonnull align 16 dereferenceable(272) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %sadd64ri.kernarg.segment, i64 36, !amdgpu.uniform !0
    %0 = load <2 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <2 x i64> %0, i32 0
    %1 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %a.load2 = extractelement <2 x i64> %0, i32 1
    %add = add i64 20015998343286, %a.load2
    store i64 %add, ptr addrspace(1) %1, align 8
    ret void
  }
  
  define amdgpu_kernel void @vadd64rr(ptr addrspace(1) %out, i64 %a) {
  entry:
    %vadd64rr.kernarg.segment = call nonnull align 16 dereferenceable(272) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %vadd64rr.kernarg.segment, i64 36, !amdgpu.uniform !0
    %0 = load <2 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <2 x i64> %0, i32 0
    %1 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %a.load2 = extractelement <2 x i64> %0, i32 1
    %tid = call i32 @llvm.amdgcn.workitem.id.x()
    %tid.ext = sext i32 %tid to i64
    %add = add i64 %a.load2, %tid.ext
    store i64 %add, ptr addrspace(1) %1, align 8
    ret void
  }
  
  define amdgpu_kernel void @vadd64ri(ptr addrspace(1) %out) {
  entry:
    %vadd64ri.kernarg.segment = call nonnull align 16 dereferenceable(264) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %vadd64ri.kernarg.segment, i64 36, !amdgpu.uniform !0
    %out.load = load ptr addrspace(1), ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %tid = call i32 @llvm.amdgcn.workitem.id.x()
    %tid.ext = sext i32 %tid to i64
    %add = add i64 20015998343286, %tid.ext
    store i64 %add, ptr addrspace(1) %out.load, align 8
    ret void
  }
  
  ; Function Attrs: nounwind
  define amdgpu_kernel void @suaddo32(ptr addrspace(1) %out, ptr addrspace(1) %carryout, i32 %a, i32 %b) #1 {
    %suaddo32.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %suaddo32.kernarg.segment, i64 36, !amdgpu.uniform !0
    %out.load = load ptr addrspace(1), ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %a.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %suaddo32.kernarg.segment, i64 52, !amdgpu.uniform !0
    %1 = load <2 x i32>, ptr addrspace(4) %a.kernarg.offset, align 4, !invariant.load !0
    %a.load1 = extractelement <2 x i32> %1, i32 0
    %b.load2 = extractelement <2 x i32> %1, i32 1
    %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a.load1, i32 %b.load2)
    %val = extractvalue { i32, i1 } %uadd, 0
    store i32 %val, ptr addrspace(1) %out.load, align 4
    ret void
  }
  
  ; Function Attrs: nounwind
  define amdgpu_kernel void @uaddo32_vcc_user(ptr addrspace(1) %out, ptr addrspace(1) %carryout, i32 %a, i32 %b) #1 {
    %uaddo32_vcc_user.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %uaddo32_vcc_user.kernarg.segment, i64 36, !amdgpu.uniform !0
    %1 = load <2 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <2 x i64> %1, i32 0
    %2 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %carryout.load2 = extractelement <2 x i64> %1, i32 1
    %3 = inttoptr i64 %carryout.load2 to ptr addrspace(1)
    %a.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %uaddo32_vcc_user.kernarg.segment, i64 52, !amdgpu.uniform !0
    %4 = load <2 x i32>, ptr addrspace(4) %a.kernarg.offset, align 4, !invariant.load !0
    %a.load3 = extractelement <2 x i32> %4, i32 0
    %b.load4 = extractelement <2 x i32> %4, i32 1
    %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a.load3, i32 %b.load4)
    %val = extractvalue { i32, i1 } %uadd, 0
    %carry = extractvalue { i32, i1 } %uadd, 1
    store i32 %val, ptr addrspace(1) %2, align 4
    store i1 %carry, ptr addrspace(1) %3, align 1
    ret void
  }
  
  ; Function Attrs: nounwind
  define amdgpu_kernel void @suaddo64(ptr addrspace(1) %out, ptr addrspace(1) %carryout, i64 %a, i64 %b) #1 {
    %suaddo64.kernarg.segment = call nonnull align 16 dereferenceable(288) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %suaddo64.kernarg.segment, i64 36, !amdgpu.uniform !0
    %1 = load <4 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <4 x i64> %1, i32 0
    %2 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %carryout.load2 = extractelement <4 x i64> %1, i32 1
    %3 = inttoptr i64 %carryout.load2 to ptr addrspace(1)
    %a.load3 = extractelement <4 x i64> %1, i32 2
    %b.load4 = extractelement <4 x i64> %1, i32 3
    %uadd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a.load3, i64 %b.load4)
    %val = extractvalue { i64, i1 } %uadd, 0
    %carry = extractvalue { i64, i1 } %uadd, 1
    store i64 %val, ptr addrspace(1) %2, align 8
    store i1 %carry, ptr addrspace(1) %3, align 1
    ret void
  }
  
  ; Function Attrs: nounwind
  define amdgpu_kernel void @vuaddo64(ptr addrspace(1) %out, ptr addrspace(1) %carryout, i64 %a) #1 {
    %vuaddo64.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %vuaddo64.kernarg.segment, i64 36, !amdgpu.uniform !0
    %1 = load <3 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <3 x i64> %1, i32 0
    %2 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %carryout.load2 = extractelement <3 x i64> %1, i32 1
    %3 = inttoptr i64 %carryout.load2 to ptr addrspace(1)
    %a.load3 = extractelement <3 x i64> %1, i32 2
    %tid = call i32 @llvm.amdgcn.workitem.id.x()
    %tid.ext = sext i32 %tid to i64
    %uadd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a.load3, i64 %tid.ext)
    %val = extractvalue { i64, i1 } %uadd, 0
    %carry = extractvalue { i64, i1 } %uadd, 1
    store i64 %val, ptr addrspace(1) %2, align 8
    store i1 %carry, ptr addrspace(1) %3, align 1
    ret void
  }
  
  define amdgpu_kernel void @ssub64rr(ptr addrspace(1) %out, i64 %a, i64 %b) #0 {
  entry:
    %ssub64rr.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %ssub64rr.kernarg.segment, i64 36, !amdgpu.uniform !0
    %0 = load <3 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <3 x i64> %0, i32 0
    %1 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %a.load2 = extractelement <3 x i64> %0, i32 1
    %b.load3 = extractelement <3 x i64> %0, i32 2
    %sub = sub i64 %a.load2, %b.load3
    store i64 %sub, ptr addrspace(1) %1, align 8
    ret void
  }
  
  define amdgpu_kernel void @ssub64ri(ptr addrspace(1) %out, i64 %a) {
  entry:
    %ssub64ri.kernarg.segment = call nonnull align 16 dereferenceable(272) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %ssub64ri.kernarg.segment, i64 36, !amdgpu.uniform !0
    %0 = load <2 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <2 x i64> %0, i32 0
    %1 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %a.load2 = extractelement <2 x i64> %0, i32 1
    %sub = sub i64 20015998343286, %a.load2
    store i64 %sub, ptr addrspace(1) %1, align 8
    ret void
  }
  
  define amdgpu_kernel void @vsub64rr(ptr addrspace(1) %out, i64 %a) {
  entry:
    %vsub64rr.kernarg.segment = call nonnull align 16 dereferenceable(272) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %vsub64rr.kernarg.segment, i64 36, !amdgpu.uniform !0
    %0 = load <2 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <2 x i64> %0, i32 0
    %1 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %a.load2 = extractelement <2 x i64> %0, i32 1
    %tid = call i32 @llvm.amdgcn.workitem.id.x()
    %tid.ext = sext i32 %tid to i64
    %sub = sub i64 %a.load2, %tid.ext
    store i64 %sub, ptr addrspace(1) %1, align 8
    ret void
  }
  
  define amdgpu_kernel void @vsub64ri(ptr addrspace(1) %out) {
  entry:
    %vsub64ri.kernarg.segment = call nonnull align 16 dereferenceable(264) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %vsub64ri.kernarg.segment, i64 36, !amdgpu.uniform !0
    %out.load = load ptr addrspace(1), ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %tid = call i32 @llvm.amdgcn.workitem.id.x()
    %tid.ext = sext i32 %tid to i64
    %sub = sub i64 20015998343286, %tid.ext
    store i64 %sub, ptr addrspace(1) %out.load, align 8
    ret void
  }
  
  ; Function Attrs: nounwind
  define amdgpu_kernel void @susubo32(ptr addrspace(1) %out, ptr addrspace(1) %carryout, i32 %a, i32 %b) #1 {
    %susubo32.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %susubo32.kernarg.segment, i64 36, !amdgpu.uniform !0
    %out.load = load ptr addrspace(1), ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %a.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %susubo32.kernarg.segment, i64 52, !amdgpu.uniform !0
    %1 = load <2 x i32>, ptr addrspace(4) %a.kernarg.offset, align 4, !invariant.load !0
    %a.load1 = extractelement <2 x i32> %1, i32 0
    %b.load2 = extractelement <2 x i32> %1, i32 1
    %usub = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a.load1, i32 %b.load2)
    %val = extractvalue { i32, i1 } %usub, 0
    store i32 %val, ptr addrspace(1) %out.load, align 4
    ret void
  }
  
  ; Function Attrs: nounwind
  define amdgpu_kernel void @usubo32_vcc_user(ptr addrspace(1) %out, ptr addrspace(1) %carryout, i32 %a, i32 %b) #1 {
    %usubo32_vcc_user.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %usubo32_vcc_user.kernarg.segment, i64 36, !amdgpu.uniform !0
    %1 = load <2 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <2 x i64> %1, i32 0
    %2 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %carryout.load2 = extractelement <2 x i64> %1, i32 1
    %3 = inttoptr i64 %carryout.load2 to ptr addrspace(1)
    %a.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %usubo32_vcc_user.kernarg.segment, i64 52, !amdgpu.uniform !0
    %4 = load <2 x i32>, ptr addrspace(4) %a.kernarg.offset, align 4, !invariant.load !0
    %a.load3 = extractelement <2 x i32> %4, i32 0
    %b.load4 = extractelement <2 x i32> %4, i32 1
    %usub = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a.load3, i32 %b.load4)
    %val = extractvalue { i32, i1 } %usub, 0
    %carry = extractvalue { i32, i1 } %usub, 1
    store i32 %val, ptr addrspace(1) %2, align 4
    store i1 %carry, ptr addrspace(1) %3, align 1
    ret void
  }
  
  ; Function Attrs: nounwind
  define amdgpu_kernel void @susubo64(ptr addrspace(1) %out, ptr addrspace(1) %carryout, i64 %a, i64 %b) #1 {
    %susubo64.kernarg.segment = call nonnull align 16 dereferenceable(288) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %susubo64.kernarg.segment, i64 36, !amdgpu.uniform !0
    %1 = load <4 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <4 x i64> %1, i32 0
    %2 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %carryout.load2 = extractelement <4 x i64> %1, i32 1
    %3 = inttoptr i64 %carryout.load2 to ptr addrspace(1)
    %a.load3 = extractelement <4 x i64> %1, i32 2
    %b.load4 = extractelement <4 x i64> %1, i32 3
    %usub = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a.load3, i64 %b.load4)
    %val = extractvalue { i64, i1 } %usub, 0
    %carry = extractvalue { i64, i1 } %usub, 1
    store i64 %val, ptr addrspace(1) %2, align 8
    store i1 %carry, ptr addrspace(1) %3, align 1
    ret void
  }
  
  ; Function Attrs: nounwind
  define amdgpu_kernel void @vusubo64(ptr addrspace(1) %out, ptr addrspace(1) %carryout, i64 %a) #1 {
    %vusubo64.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %vusubo64.kernarg.segment, i64 36, !amdgpu.uniform !0
    %1 = load <3 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %out.load1 = extractelement <3 x i64> %1, i32 0
    %2 = inttoptr i64 %out.load1 to ptr addrspace(1)
    %carryout.load2 = extractelement <3 x i64> %1, i32 1
    %3 = inttoptr i64 %carryout.load2 to ptr addrspace(1)
    %a.load3 = extractelement <3 x i64> %1, i32 2
    %tid = call i32 @llvm.amdgcn.workitem.id.x()
    %tid.ext = sext i32 %tid to i64
    %usub = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a.load3, i64 %tid.ext)
    %val = extractvalue { i64, i1 } %usub, 0
    %carry = extractvalue { i64, i1 } %usub, 1
    store i64 %val, ptr addrspace(1) %2, align 8
    store i1 %carry, ptr addrspace(1) %3, align 1
    ret void
  }
  
  define amdgpu_kernel void @sudiv64(ptr addrspace(1) %out, i64 %x, i64 %y) {
    %sudiv64.kernarg.segment = call nonnull align 16 dereferenceable(280) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %out.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %sudiv64.kernarg.segment, i64 36, !amdgpu.uniform !0
    %1 = load <3 x i64>, ptr addrspace(4) %out.kernarg.offset, align 4, !invariant.load !0
    %x.load2 = extractelement <3 x i64> %1, i32 1
    %y.load3 = extractelement <3 x i64> %1, i32 2
    %2 = or i64 %x.load2, %y.load3
    %3 = and i64 %2, -4294967296
    %4 = icmp ne i64 %3, 0
    br i1 %4, label %12, label %Flow, !amdgpu.uniform !0
  
  Flow:                                             ; preds = %12, %0
    %5 = phi i64 [ %13, %12 ], [ poison, %0 ]
    %6 = phi i1 [ false, %12 ], [ true, %0 ]
    br i1 %6, label %7, label %14, !amdgpu.uniform !0
  
  7:                                                ; preds = %Flow
    %8 = trunc i64 %y.load3 to i32
    %9 = trunc i64 %x.load2 to i32
    %10 = udiv i32 %9, %8
    %11 = zext i32 %10 to i64
    br label %14, !amdgpu.uniform !0
  
  12:                                               ; preds = %0
    %13 = udiv i64 %x.load2, %y.load3
    br label %Flow, !amdgpu.uniform !0
  
  14:                                               ; preds = %7, %Flow
    %15 = phi i64 [ %5, %Flow ], [ %11, %7 ]
    %out.load1 = extractelement <3 x i64> %1, i32 0
    %16 = inttoptr i64 %out.load1 to ptr addrspace(1)
    store i64 %15, ptr addrspace(1) %16, align 8
    ret void
  }
  
  ; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
  declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64) #2
  
  ; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
  declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #2
  
  ; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
  declare { i64, i1 } @llvm.usub.with.overflow.i64(i64, i64) #2
  
  ; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
  declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #2
  
  ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
  declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3
  
  ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
  declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #3
  
  attributes #0 = { "amdgpu-memory-bound"="true" "amdgpu-wave-limiter"="true" }
  attributes #1 = { nounwind }
  attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
  attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
  
  !0 = !{}
...
---
name:            sadd64rr
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 28, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     true
  waveLimiter:     true
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset + 16, align 4, addrspace 4)
    %13:sreg_32 = COPY %12.sub1
    %14:sreg_32 = COPY %12.sub0
    %15:sreg_32 = COPY %11.sub3
    %16:sreg_32 = COPY %11.sub2
    %17:sreg_32 = COPY %11.sub1
    %18:sreg_32 = COPY %11.sub0
    %19:sreg_64 = REG_SEQUENCE killed %18, %subreg.sub0, killed %17, %subreg.sub1
    %20:sreg_32 = COPY %19.sub1
    %21:sreg_32 = COPY %19.sub0
    %22:sreg_32 = S_MOV_B32 61440
    %23:sreg_32 = S_MOV_B32 -1
    %24:sgpr_128 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1, killed %23, %subreg.sub2, killed %22, %subreg.sub3
    %25:sreg_64 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1
    %26:sreg_64 = REG_SEQUENCE killed %14, %subreg.sub0, killed %13, %subreg.sub1
    %27:sreg_64 = S_ADD_U64_PSEUDO killed %25, killed %26, implicit-def dead $scc
    %28:vreg_64 = COPY %27
    BUFFER_STORE_DWORDX2_OFFSET killed %28, killed %24, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.1, addrspace 1)
    S_ENDPGM 0
...
---
name:            sadd64ri
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 27, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 16
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_32 = COPY %11.sub1
    %13:sreg_32 = COPY %11.sub0
    %14:sreg_64 = REG_SEQUENCE killed %13, %subreg.sub0, killed %12, %subreg.sub1
    %15:sreg_32 = COPY %14.sub1
    %16:sreg_32 = COPY %14.sub0
    %17:sreg_32 = S_MOV_B32 61440
    %18:sreg_32 = S_MOV_B32 -1
    %19:sgpr_128 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1, killed %18, %subreg.sub2, killed %17, %subreg.sub3
    %20:sreg_32 = COPY %11.sub3
    %21:sreg_32 = COPY %11.sub2
    %22:sreg_64 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1
    %23:sreg_32 = S_MOV_B32 4660
    %24:sreg_32 = S_MOV_B32 1450743926
    %25:sreg_64 = REG_SEQUENCE killed %24, %subreg.sub0, killed %23, %subreg.sub1
    %26:sreg_64 = S_ADD_U64_PSEUDO killed %22, killed %25, implicit-def dead $scc
    %27:vreg_64 = COPY %26
    BUFFER_STORE_DWORDX2_OFFSET killed %27, killed %19, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.1, addrspace 1)
    S_ENDPGM 0
...
---
name:            vadd64rr
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 23, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 25, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$vgpr0', virtual-reg: '%0' }
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 16
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    liveins: $vgpr0, $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %0:vgpr_32(s32) = COPY $vgpr0
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_32 = COPY %11.sub1
    %13:sreg_32 = COPY %11.sub0
    %14:sreg_64 = REG_SEQUENCE killed %13, %subreg.sub0, killed %12, %subreg.sub1
    %15:sreg_32 = COPY %14.sub1
    %16:sreg_32 = COPY %14.sub0
    %17:sreg_32 = S_MOV_B32 61440
    %18:sreg_32 = S_MOV_B32 -1
    %19:sgpr_128 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1, killed %18, %subreg.sub2, killed %17, %subreg.sub3
    %20:sreg_32 = COPY %11.sub3
    %21:sreg_32 = COPY %11.sub2
    %22:sreg_64 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1
    %23:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    %24:vreg_64 = REG_SEQUENCE %0(s32), %subreg.sub0, killed %23, %subreg.sub1
    %25:vreg_64 = V_ADD_U64_PSEUDO killed %22, killed %24, implicit-def dead $vcc, implicit $exec
    BUFFER_STORE_DWORDX2_OFFSET killed %25, killed %19, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.1, addrspace 1)
    S_ENDPGM 0
...
---
name:            vadd64ri
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 17, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 22, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$vgpr0', virtual-reg: '%0' }
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 8
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    liveins: $vgpr0, $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %0:vgpr_32(s32) = COPY $vgpr0
    %11:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_32 = COPY %11.sub1
    %13:sreg_32 = COPY %11.sub0
    %14:sreg_32 = S_MOV_B32 61440
    %15:sreg_32 = S_MOV_B32 -1
    %16:sgpr_128 = REG_SEQUENCE killed %13, %subreg.sub0, killed %12, %subreg.sub1, killed %15, %subreg.sub2, killed %14, %subreg.sub3
    %17:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    %18:vreg_64 = REG_SEQUENCE %0(s32), %subreg.sub0, killed %17, %subreg.sub1
    %19:sreg_32 = S_MOV_B32 4660
    %20:sreg_32 = S_MOV_B32 1450743926
    %21:sreg_64 = REG_SEQUENCE killed %20, %subreg.sub0, killed %19, %subreg.sub1
    %22:vreg_64 = V_ADD_U64_PSEUDO killed %18, killed %21, implicit-def dead $vcc, implicit $exec
    BUFFER_STORE_DWORDX2_OFFSET killed %22, killed %16, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.out.load, addrspace 1)
    S_ENDPGM 0
...
---
name:            suaddo32
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.a.kernarg.offset, align 4, addrspace 4)
    %13:sreg_32 = COPY %11.sub1
    %14:sreg_32 = COPY %11.sub0
    %15:sreg_32 = S_MOV_B32 61440
    %16:sreg_32 = S_MOV_B32 -1
    %17:sgpr_128 = REG_SEQUENCE killed %14, %subreg.sub0, killed %13, %subreg.sub1, killed %16, %subreg.sub2, killed %15, %subreg.sub3
    %18:sreg_32 = COPY %12.sub0
    %19:sreg_32 = COPY %12.sub1
    %20:sreg_32 = S_ADD_I32 killed %18, killed %19, implicit-def dead $scc
    %21:vgpr_32 = COPY %20
    BUFFER_STORE_DWORD_OFFSET killed %21, killed %17, 0, 0, 0, 0, implicit $exec :: (store (s32) into %ir.out.load, addrspace 1)
    S_ENDPGM 0
...
---
name:            uaddo32_vcc_user
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 20, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 24, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 26, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 28, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 29, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 30, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 31, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 32, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.a.kernarg.offset, align 4, addrspace 4)
    %13:sreg_32 = COPY %11.sub1
    %14:sreg_32 = COPY %11.sub0
    %15:sreg_64 = REG_SEQUENCE killed %14, %subreg.sub0, killed %13, %subreg.sub1
    %16:sreg_32 = COPY %15.sub1
    %17:sreg_32 = COPY %15.sub0
    %18:sreg_32 = S_MOV_B32 61440
    %19:sreg_32 = S_MOV_B32 -1
    %20:sgpr_128 = REG_SEQUENCE killed %17, %subreg.sub0, killed %16, %subreg.sub1, %19, %subreg.sub2, %18, %subreg.sub3
    %21:sreg_32 = COPY %11.sub3
    %22:sreg_32 = COPY %11.sub2
    %23:sreg_64 = REG_SEQUENCE killed %22, %subreg.sub0, killed %21, %subreg.sub1
    %24:sreg_32 = COPY %23.sub1
    %25:sreg_32 = COPY %23.sub0
    %26:sgpr_128 = REG_SEQUENCE killed %25, %subreg.sub0, killed %24, %subreg.sub1, %19, %subreg.sub2, %18, %subreg.sub3
    %27:sreg_32 = COPY %12.sub0
    %28:sreg_32 = COPY %12.sub1
    %31:vgpr_32 = COPY killed %28
    %29:vgpr_32, %30:sreg_64_xexec = V_ADD_CO_U32_e64 killed %27, %31, 0, implicit $exec
    BUFFER_STORE_DWORD_OFFSET killed %29, killed %20, 0, 0, 0, 0, implicit $exec :: (store (s32) into %ir.2, addrspace 1)
    %32:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, killed %30, implicit $exec
    BUFFER_STORE_BYTE_OFFSET killed %32, killed %26, 0, 0, 0, 0, implicit $exec :: (store (s8) into %ir.3, addrspace 1)
    S_ENDPGM 0
...
---
name:            suaddo64
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_256, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 25, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 28, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 29, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 30, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 31, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 32, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 33, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 34, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 35, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 36, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 32
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sgpr_256 = S_LOAD_DWORDX8_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s256) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_32 = COPY %11.sub1
    %13:sreg_32 = COPY %11.sub0
    %14:sreg_64 = REG_SEQUENCE killed %13, %subreg.sub0, killed %12, %subreg.sub1
    %15:sreg_32 = COPY %14.sub1
    %16:sreg_32 = COPY %14.sub0
    %17:sreg_32 = S_MOV_B32 61440
    %18:sreg_32 = S_MOV_B32 -1
    %19:sgpr_128 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1, %18, %subreg.sub2, %17, %subreg.sub3
    %20:sreg_32 = COPY %11.sub3
    %21:sreg_32 = COPY %11.sub2
    %22:sreg_64 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1
    %23:sreg_32 = COPY %22.sub1
    %24:sreg_32 = COPY %22.sub0
    %25:sgpr_128 = REG_SEQUENCE killed %24, %subreg.sub0, killed %23, %subreg.sub1, %18, %subreg.sub2, %17, %subreg.sub3
    %26:sreg_32 = COPY %11.sub5
    %27:sreg_32 = COPY %11.sub4
    %28:sreg_32 = COPY %11.sub7
    %29:sreg_32 = COPY %11.sub6
    %30:sreg_32, %31:sreg_64_xexec = S_UADDO_PSEUDO killed %27, killed %29, implicit-def dead $scc
    %32:sreg_32, %33:sreg_64_xexec = S_ADD_CO_PSEUDO killed %26, killed %28, killed %31, implicit-def dead $scc
    %34:sreg_64 = REG_SEQUENCE killed %30, %subreg.sub0, killed %32, %subreg.sub1
    %35:vreg_64 = COPY %34
    BUFFER_STORE_DWORDX2_OFFSET killed %35, killed %19, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.2, addrspace 1)
    %36:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, killed %33, implicit $exec
    BUFFER_STORE_BYTE_OFFSET killed %36, killed %25, 0, 0, 0, 0, implicit $exec :: (store (s8) into %ir.3, addrspace 1)
    S_ENDPGM 0
...
---
name:            vuaddo64
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 28, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 29, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 30, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 31, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 32, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 33, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 34, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 35, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 36, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 37, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$vgpr0', virtual-reg: '%0' }
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    liveins: $vgpr0, $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %0:vgpr_32(s32) = COPY $vgpr0
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset + 16, align 4, addrspace 4)
    %13:sreg_32 = COPY %12.sub1
    %14:sreg_32 = COPY %12.sub0
    %15:sreg_32 = COPY %11.sub3
    %16:sreg_32 = COPY %11.sub2
    %17:sreg_32 = COPY %11.sub1
    %18:sreg_32 = COPY %11.sub0
    %19:sreg_64 = REG_SEQUENCE killed %18, %subreg.sub0, killed %17, %subreg.sub1
    %20:sreg_32 = COPY %19.sub1
    %21:sreg_32 = COPY %19.sub0
    %22:sreg_32 = S_MOV_B32 61440
    %23:sreg_32 = S_MOV_B32 -1
    %24:sgpr_128 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1, %23, %subreg.sub2, %22, %subreg.sub3
    %25:sreg_64 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1
    %26:sreg_32 = COPY %25.sub1
    %27:sreg_32 = COPY %25.sub0
    %28:sgpr_128 = REG_SEQUENCE killed %27, %subreg.sub0, killed %26, %subreg.sub1, %23, %subreg.sub2, %22, %subreg.sub3
    %29:vgpr_32, %30:sreg_64_xexec = V_ADD_CO_U32_e64 killed %14, %0(s32), 0, implicit $exec
    %31:sreg_32 = S_MOV_B32 0
    %34:vgpr_32 = COPY killed %13
    %35:vgpr_32 = COPY killed %31
    %32:vgpr_32, %33:sreg_64_xexec = V_ADDC_U32_e64 %34, %35, killed %30, 0, implicit $exec
    %36:vreg_64 = REG_SEQUENCE killed %29, %subreg.sub0, killed %32, %subreg.sub1
    BUFFER_STORE_DWORDX2_OFFSET killed %36, killed %24, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.2, addrspace 1)
    %37:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, killed %33, implicit $exec
    BUFFER_STORE_BYTE_OFFSET killed %37, killed %28, 0, 0, 0, 0, implicit $exec :: (store (s8) into %ir.3, addrspace 1)
    S_ENDPGM 0
...
---
name:            ssub64rr
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 28, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     true
  waveLimiter:     true
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset + 16, align 4, addrspace 4)
    %13:sreg_32 = COPY %12.sub1
    %14:sreg_32 = COPY %12.sub0
    %15:sreg_32 = COPY %11.sub3
    %16:sreg_32 = COPY %11.sub2
    %17:sreg_32 = COPY %11.sub1
    %18:sreg_32 = COPY %11.sub0
    %19:sreg_64 = REG_SEQUENCE killed %18, %subreg.sub0, killed %17, %subreg.sub1
    %20:sreg_32 = COPY %19.sub1
    %21:sreg_32 = COPY %19.sub0
    %22:sreg_32 = S_MOV_B32 61440
    %23:sreg_32 = S_MOV_B32 -1
    %24:sgpr_128 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1, killed %23, %subreg.sub2, killed %22, %subreg.sub3
    %25:sreg_64 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1
    %26:sreg_64 = REG_SEQUENCE killed %14, %subreg.sub0, killed %13, %subreg.sub1
    %27:sreg_64 = S_SUB_U64_PSEUDO killed %25, killed %26, implicit-def dead $scc
    %28:vreg_64 = COPY %27
    BUFFER_STORE_DWORDX2_OFFSET killed %28, killed %24, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.1, addrspace 1)
    S_ENDPGM 0
...
---
name:            ssub64ri
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 27, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 16
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_32 = COPY %11.sub1
    %13:sreg_32 = COPY %11.sub0
    %14:sreg_64 = REG_SEQUENCE killed %13, %subreg.sub0, killed %12, %subreg.sub1
    %15:sreg_32 = COPY %14.sub1
    %16:sreg_32 = COPY %14.sub0
    %17:sreg_32 = S_MOV_B32 61440
    %18:sreg_32 = S_MOV_B32 -1
    %19:sgpr_128 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1, killed %18, %subreg.sub2, killed %17, %subreg.sub3
    %20:sreg_32 = COPY %11.sub3
    %21:sreg_32 = COPY %11.sub2
    %22:sreg_64 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1
    %23:sreg_32 = S_MOV_B32 4660
    %24:sreg_32 = S_MOV_B32 1450743926
    %25:sreg_64 = REG_SEQUENCE killed %24, %subreg.sub0, killed %23, %subreg.sub1
    %26:sreg_64 = S_SUB_U64_PSEUDO killed %25, killed %22, implicit-def dead $scc
    %27:vreg_64 = COPY %26
    BUFFER_STORE_DWORDX2_OFFSET killed %27, killed %19, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.1, addrspace 1)
    S_ENDPGM 0
...
---
name:            vsub64rr
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 23, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 25, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$vgpr0', virtual-reg: '%0' }
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 16
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    liveins: $vgpr0, $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %0:vgpr_32(s32) = COPY $vgpr0
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_32 = COPY %11.sub1
    %13:sreg_32 = COPY %11.sub0
    %14:sreg_64 = REG_SEQUENCE killed %13, %subreg.sub0, killed %12, %subreg.sub1
    %15:sreg_32 = COPY %14.sub1
    %16:sreg_32 = COPY %14.sub0
    %17:sreg_32 = S_MOV_B32 61440
    %18:sreg_32 = S_MOV_B32 -1
    %19:sgpr_128 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1, killed %18, %subreg.sub2, killed %17, %subreg.sub3
    %20:sreg_32 = COPY %11.sub3
    %21:sreg_32 = COPY %11.sub2
    %22:sreg_64 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1
    %23:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    %24:vreg_64 = REG_SEQUENCE %0(s32), %subreg.sub0, killed %23, %subreg.sub1
    %25:vreg_64 = V_SUB_U64_PSEUDO killed %22, killed %24, implicit-def dead $vcc, implicit $exec
    BUFFER_STORE_DWORDX2_OFFSET killed %25, killed %19, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.1, addrspace 1)
    S_ENDPGM 0
...
---
name:            vsub64ri
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 17, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 22, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$vgpr0', virtual-reg: '%0' }
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 8
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    liveins: $vgpr0, $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %0:vgpr_32(s32) = COPY $vgpr0
    %11:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_32 = COPY %11.sub1
    %13:sreg_32 = COPY %11.sub0
    %14:sreg_32 = S_MOV_B32 61440
    %15:sreg_32 = S_MOV_B32 -1
    %16:sgpr_128 = REG_SEQUENCE killed %13, %subreg.sub0, killed %12, %subreg.sub1, killed %15, %subreg.sub2, killed %14, %subreg.sub3
    %17:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    %18:vreg_64 = REG_SEQUENCE %0(s32), %subreg.sub0, killed %17, %subreg.sub1
    %19:sreg_32 = S_MOV_B32 4660
    %20:sreg_32 = S_MOV_B32 1450743926
    %21:sreg_64 = REG_SEQUENCE killed %20, %subreg.sub0, killed %19, %subreg.sub1
    %22:vreg_64 = V_SUB_U64_PSEUDO killed %21, killed %18, implicit-def dead $vcc, implicit $exec
    BUFFER_STORE_DWORDX2_OFFSET killed %22, killed %16, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.out.load, addrspace 1)
    S_ENDPGM 0
...
---
name:            susubo32
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.a.kernarg.offset, align 4, addrspace 4)
    %13:sreg_32 = COPY %11.sub1
    %14:sreg_32 = COPY %11.sub0
    %15:sreg_32 = S_MOV_B32 61440
    %16:sreg_32 = S_MOV_B32 -1
    %17:sgpr_128 = REG_SEQUENCE killed %14, %subreg.sub0, killed %13, %subreg.sub1, killed %16, %subreg.sub2, killed %15, %subreg.sub3
    %18:sreg_32 = COPY %12.sub0
    %19:sreg_32 = COPY %12.sub1
    %20:sreg_32 = S_SUB_I32 killed %18, killed %19, implicit-def dead $scc
    %21:vgpr_32 = COPY %20
    BUFFER_STORE_DWORD_OFFSET killed %21, killed %17, 0, 0, 0, 0, implicit $exec :: (store (s32) into %ir.out.load, addrspace 1)
    S_ENDPGM 0
...
---
name:            usubo32_vcc_user
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 20, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 24, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 26, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 28, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 29, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 30, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 31, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 32, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.a.kernarg.offset, align 4, addrspace 4)
    %13:sreg_32 = COPY %11.sub1
    %14:sreg_32 = COPY %11.sub0
    %15:sreg_64 = REG_SEQUENCE killed %14, %subreg.sub0, killed %13, %subreg.sub1
    %16:sreg_32 = COPY %15.sub1
    %17:sreg_32 = COPY %15.sub0
    %18:sreg_32 = S_MOV_B32 61440
    %19:sreg_32 = S_MOV_B32 -1
    %20:sgpr_128 = REG_SEQUENCE killed %17, %subreg.sub0, killed %16, %subreg.sub1, %19, %subreg.sub2, %18, %subreg.sub3
    %21:sreg_32 = COPY %11.sub3
    %22:sreg_32 = COPY %11.sub2
    %23:sreg_64 = REG_SEQUENCE killed %22, %subreg.sub0, killed %21, %subreg.sub1
    %24:sreg_32 = COPY %23.sub1
    %25:sreg_32 = COPY %23.sub0
    %26:sgpr_128 = REG_SEQUENCE killed %25, %subreg.sub0, killed %24, %subreg.sub1, %19, %subreg.sub2, %18, %subreg.sub3
    %27:sreg_32 = COPY %12.sub0
    %28:sreg_32 = COPY %12.sub1
    %31:vgpr_32 = COPY killed %28
    %29:vgpr_32, %30:sreg_64_xexec = V_SUB_CO_U32_e64 killed %27, %31, 0, implicit $exec
    BUFFER_STORE_DWORD_OFFSET killed %29, killed %20, 0, 0, 0, 0, implicit $exec :: (store (s32) into %ir.2, addrspace 1)
    %32:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, killed %30, implicit $exec
    BUFFER_STORE_BYTE_OFFSET killed %32, killed %26, 0, 0, 0, 0, implicit $exec :: (store (s8) into %ir.3, addrspace 1)
    S_ENDPGM 0
...
---
name:            susubo64
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_256, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 25, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 28, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 29, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 30, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 31, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 32, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 33, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 34, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 35, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 36, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 32
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    liveins: $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %11:sgpr_256 = S_LOAD_DWORDX8_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s256) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_32 = COPY %11.sub1
    %13:sreg_32 = COPY %11.sub0
    %14:sreg_64 = REG_SEQUENCE killed %13, %subreg.sub0, killed %12, %subreg.sub1
    %15:sreg_32 = COPY %14.sub1
    %16:sreg_32 = COPY %14.sub0
    %17:sreg_32 = S_MOV_B32 61440
    %18:sreg_32 = S_MOV_B32 -1
    %19:sgpr_128 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1, %18, %subreg.sub2, %17, %subreg.sub3
    %20:sreg_32 = COPY %11.sub3
    %21:sreg_32 = COPY %11.sub2
    %22:sreg_64 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1
    %23:sreg_32 = COPY %22.sub1
    %24:sreg_32 = COPY %22.sub0
    %25:sgpr_128 = REG_SEQUENCE killed %24, %subreg.sub0, killed %23, %subreg.sub1, %18, %subreg.sub2, %17, %subreg.sub3
    %26:sreg_32 = COPY %11.sub5
    %27:sreg_32 = COPY %11.sub4
    %28:sreg_64 = REG_SEQUENCE killed %27, %subreg.sub0, killed %26, %subreg.sub1
    %29:sreg_32 = COPY %11.sub7
    %30:sreg_32 = COPY %11.sub6
    %31:sreg_64 = REG_SEQUENCE killed %30, %subreg.sub0, killed %29, %subreg.sub1
    %33:vreg_64 = COPY %31
    %32:sreg_64_xexec = V_CMP_GT_U64_e64 %28, %33, implicit $exec
    %34:sreg_64 = S_SUB_U64_PSEUDO %28, %31, implicit-def dead $scc
    %35:vreg_64 = COPY %34
    BUFFER_STORE_DWORDX2_OFFSET killed %35, killed %19, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.2, addrspace 1)
    %36:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, killed %32, implicit $exec
    BUFFER_STORE_BYTE_OFFSET killed %36, killed %25, 0, 0, 0, 0, implicit $exec :: (store (s8) into %ir.3, addrspace 1)
    S_ENDPGM 0
...
---
name:            vusubo64
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 3, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 12, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 13, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 14, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 15, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 21, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 28, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 29, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 30, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 31, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 32, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 33, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 34, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$vgpr0', virtual-reg: '%0' }
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%5' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    liveins: $vgpr0, $sgpr4_sgpr5
  
    %5:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %0:vgpr_32(s32) = COPY $vgpr0
    %11:sgpr_128 = S_LOAD_DWORDX4_IMM %5(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %12:sreg_64_xexec = S_LOAD_DWORDX2_IMM %5(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset + 16, align 4, addrspace 4)
    %13:sreg_32 = COPY %12.sub1
    %14:sreg_32 = COPY %12.sub0
    %15:sreg_32 = COPY %11.sub3
    %16:sreg_32 = COPY %11.sub2
    %17:sreg_32 = COPY %11.sub1
    %18:sreg_32 = COPY %11.sub0
    %19:sreg_64 = REG_SEQUENCE killed %18, %subreg.sub0, killed %17, %subreg.sub1
    %20:sreg_32 = COPY %19.sub1
    %21:sreg_32 = COPY %19.sub0
    %22:sreg_32 = S_MOV_B32 61440
    %23:sreg_32 = S_MOV_B32 -1
    %24:sgpr_128 = REG_SEQUENCE killed %21, %subreg.sub0, killed %20, %subreg.sub1, %23, %subreg.sub2, %22, %subreg.sub3
    %25:sreg_64 = REG_SEQUENCE killed %16, %subreg.sub0, killed %15, %subreg.sub1
    %26:sreg_32 = COPY %25.sub1
    %27:sreg_32 = COPY %25.sub0
    %28:sgpr_128 = REG_SEQUENCE killed %27, %subreg.sub0, killed %26, %subreg.sub1, %23, %subreg.sub2, %22, %subreg.sub3
    %29:sreg_64 = REG_SEQUENCE killed %14, %subreg.sub0, killed %13, %subreg.sub1
    %30:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    %31:vreg_64 = REG_SEQUENCE %0(s32), %subreg.sub0, killed %30, %subreg.sub1
    %32:sreg_64_xexec = V_CMP_GT_U64_e64 %29, %31, implicit $exec
    %33:vreg_64 = V_SUB_U64_PSEUDO %29, %31, implicit-def dead $vcc, implicit $exec
    BUFFER_STORE_DWORDX2_OFFSET killed %33, killed %24, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.2, addrspace 1)
    %34:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, killed %32, implicit $exec
    BUFFER_STORE_BYTE_OFFSET killed %34, killed %28, 0, 0, 0, 0, implicit $exec :: (store (s8) into %ir.3, addrspace 1)
    S_ENDPGM 0
...
---
name:            sudiv64
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: sgpr_192, preferred-register: '', flags: [  ] }
  - { id: 1, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 2, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 3, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 4, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 5, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 6, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 7, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 8, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 9, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 10, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 11, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 12, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 13, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 14, class: sgpr_64, preferred-register: '', flags: [  ] }
  - { id: 15, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 16, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 17, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 18, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 19, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 20, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 21, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 22, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 23, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 24, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 25, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 26, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 27, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 28, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 29, class: sgpr_192, preferred-register: '', flags: [  ] }
  - { id: 30, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 31, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 32, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 33, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 34, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 35, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 36, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 37, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 38, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 39, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 40, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 41, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 42, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 43, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 44, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 45, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 46, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 47, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 48, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 49, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 50, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 51, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 52, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 53, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 54, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 55, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 56, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 57, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 58, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 59, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 60, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 61, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 62, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 63, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 64, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 65, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 66, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 67, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 68, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 69, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 70, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 71, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 72, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 73, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 74, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 75, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 76, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 77, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 78, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 79, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 80, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 81, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 82, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 83, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 84, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 85, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 86, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 87, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 88, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 89, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 90, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 91, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 92, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 93, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 94, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 95, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 96, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 97, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 98, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 99, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 100, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 101, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 102, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 103, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 104, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 105, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 106, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 107, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 108, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 109, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 110, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 111, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 112, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 113, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 114, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 115, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 116, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 117, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 118, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 119, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 120, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 121, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 122, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 123, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 124, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 125, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 126, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 127, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 128, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 129, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 130, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 131, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 132, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 133, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 134, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 135, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 136, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 137, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 138, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 139, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 140, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 141, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 142, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 143, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 144, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 145, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 146, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 147, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 148, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 149, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 150, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 151, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 152, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 153, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 154, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 155, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 156, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 157, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 158, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 159, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 160, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 161, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 162, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 163, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 164, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 165, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 166, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 167, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 168, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 169, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 170, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 171, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 172, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 173, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 174, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 175, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 176, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 177, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 178, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 179, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 180, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 181, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 182, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 183, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 184, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 185, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 186, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 187, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 188, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 189, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 190, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 191, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 192, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 193, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 194, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 195, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 196, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 197, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 198, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 199, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 200, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 201, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 202, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 203, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 204, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 205, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 206, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 207, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 208, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 209, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 210, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 211, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 212, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 213, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 214, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 215, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 216, class: sreg_64_xexec, preferred-register: '', flags: [  ] }
  - { id: 217, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 218, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 219, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 220, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 221, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 222, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 223, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 224, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 225, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 226, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 227, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 228, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 229, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 230, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 231, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 232, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 233, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 234, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 235, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 236, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 237, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 238, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 239, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 240, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 241, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 242, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 243, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 244, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 245, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 246, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 247, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 248, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 249, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 250, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 251, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 252, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 253, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 254, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 255, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 256, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 257, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 258, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 259, class: sreg_64, preferred-register: '', flags: [  ] }
  - { id: 260, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 261, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 262, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 263, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 264, class: sgpr_128, preferred-register: '', flags: [  ] }
  - { id: 265, class: vreg_64, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$sgpr4_sgpr5', virtual-reg: '%13' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 24
  maxKernArgAlign: 8
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: true
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    dispatchPtr:     { reg: '$sgpr0_sgpr1' }
    queuePtr:        { reg: '$sgpr2_sgpr3' }
    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
    dispatchID:      { reg: '$sgpr6_sgpr7' }
    workGroupIDX:    { reg: '$sgpr8' }
    workGroupIDY:    { reg: '$sgpr9' }
    workGroupIDZ:    { reg: '$sgpr10' }
    privateSegmentWaveByteOffset: { reg: '$sgpr11' }
    workItemIDX:     { reg: '$vgpr0' }
    workItemIDY:     { reg: '$vgpr1' }
    workItemIDZ:     { reg: '$vgpr2' }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       10
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0 (%ir-block.0):
    successors: %bb.3(0x50000000), %bb.1(0x30000000)
    liveins: $sgpr4_sgpr5
  
    %13:sgpr_64(p4) = COPY $sgpr4_sgpr5
    %21:sgpr_128 = S_LOAD_DWORDX4_IMM %13(p4), 9, 0 :: (dereferenceable invariant load (s128) from %ir.out.kernarg.offset, align 4, addrspace 4)
    %22:sreg_64_xexec = S_LOAD_DWORDX2_IMM %13(p4), 13, 0 :: (dereferenceable invariant load (s64) from %ir.out.kernarg.offset + 16, align 4, addrspace 4)
    %23:sreg_32 = COPY %22.sub1
    %24:sreg_32 = COPY %22.sub0
    %25:sreg_32 = COPY %21.sub3
    %26:sreg_32 = COPY %21.sub2
    %27:sreg_32 = COPY %21.sub1
    %28:sreg_32 = COPY %21.sub0
    %29:sgpr_192 = REG_SEQUENCE killed %28, %subreg.sub0, killed %27, %subreg.sub1, %26, %subreg.sub2, %25, %subreg.sub3, %24, %subreg.sub4, %23, %subreg.sub5
    %0:sgpr_192 = COPY %29
    %30:sreg_64 = REG_SEQUENCE %26, %subreg.sub0, %25, %subreg.sub1
    %1:sreg_64 = COPY %30
    %31:sreg_64 = REG_SEQUENCE %24, %subreg.sub0, %23, %subreg.sub1
    %2:sreg_64 = COPY %31
    %32:sreg_64 = S_OR_B64 %30, %31, implicit-def dead $scc
    %33:sreg_32 = COPY %32.sub1
    %34:sreg_32 = S_MOV_B32 0
    %35:sreg_64 = REG_SEQUENCE killed %34, %subreg.sub0, killed %33, %subreg.sub1
    %36:sreg_64 = S_MOV_B64 0
    %38:vreg_64 = COPY killed %36
    %37:sreg_64 = V_CMP_NE_U64_e64 killed %35, %38, implicit $exec
    %20:sreg_64 = S_MOV_B64 -1
    %19:sreg_64 = IMPLICIT_DEF
    %39:sreg_64 = S_AND_B64 $exec, killed %37, implicit-def dead $scc
    $vcc = COPY %39
    S_CBRANCH_VCCNZ %bb.3, implicit $vcc
    S_BRANCH %bb.1
  
  bb.1.Flow:
    successors: %bb.2(0x40000000), %bb.4(0x40000000)
  
    %3:sreg_64 = PHI %19, %bb.0, %6, %bb.3
    %4:sreg_64_xexec = PHI %20, %bb.0, %40, %bb.3
    %224:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, %4, implicit $exec
    %225:sreg_32 = S_MOV_B32 1
    %226:sreg_32 = COPY %224
    S_CMP_LG_U32 killed %226, killed %225, implicit-def $scc
    S_CBRANCH_SCC1 %bb.4, implicit $scc
    S_BRANCH %bb.2
  
  bb.2 (%ir-block.7):
    successors: %bb.4(0x80000000)
  
    %227:sreg_32 = COPY %2.sub0
    %228:sreg_32 = COPY %1.sub0
    %229:sreg_32 = S_MOV_B32 0
    %230:sreg_32 = S_SUB_I32 killed %229, %227, implicit-def dead $scc
    %231:vgpr_32 = V_CVT_F32_U32_e32 %227, implicit $mode, implicit $exec
    %232:vgpr_32 = nofpexcept V_RCP_IFLAG_F32_e32 killed %231, implicit $mode, implicit $exec
    %233:vgpr_32 = nofpexcept V_MUL_F32_e32 1333788670, killed %232, implicit $mode, implicit $exec
    %234:vgpr_32 = nofpexcept V_CVT_U32_F32_e32 killed %233, implicit $mode, implicit $exec
    %236:sreg_32 = COPY %234
    %235:sreg_32 = S_MUL_I32 killed %230, %236
    %237:vgpr_32 = V_MUL_HI_U32_e64 %234, killed %235, implicit $exec
    %239:sreg_32 = COPY %234
    %240:sreg_32 = COPY %237
    %238:sreg_32 = S_ADD_I32 %239, killed %240, implicit-def dead $scc
    %242:vgpr_32 = COPY killed %238
    %241:vgpr_32 = V_MUL_HI_U32_e64 %228, %242, implicit $exec
    %243:sreg_32 = S_MOV_B32 1
    %245:sreg_32 = COPY %241
    %244:sreg_32 = S_ADD_I32 %245, %243, implicit-def dead $scc
    %247:sreg_32 = COPY %241
    %246:sreg_32 = S_MUL_I32 %247, %227
    %248:sreg_32 = S_SUB_I32 %228, killed %246, implicit-def dead $scc
    %249:sreg_32 = S_SUB_I32 %248, %227, implicit-def dead $scc
    S_CMP_GE_U32 %248, %227, implicit-def $scc
    %250:sreg_32 = S_CSELECT_B32 killed %249, %248, implicit $scc
    %252:sreg_32 = COPY %241
    %251:sreg_32 = S_CSELECT_B32 killed %244, %252, implicit $scc
    %253:sreg_32 = S_ADD_I32 %251, %243, implicit-def dead $scc
    S_CMP_GE_U32 killed %250, %227, implicit-def $scc
    %254:sreg_32 = S_CSELECT_B32 killed %253, %251, implicit $scc
    %255:sreg_32 = S_MOV_B32 0
    %256:sreg_64 = REG_SEQUENCE killed %254, %subreg.sub0, killed %255, %subreg.sub1
    %5:sreg_64 = COPY %256
    S_BRANCH %bb.4
  
  bb.3 (%ir-block.12):
    successors: %bb.1(0x80000000)
  
    %41:sreg_32 = COPY %2.sub0
    %42:vgpr_32 = V_CVT_F32_U32_e64 %41, 0, 0, implicit $mode, implicit $exec
    %43:sreg_32 = COPY %2.sub1
    %44:vgpr_32 = V_CVT_F32_U32_e64 %43, 0, 0, implicit $mode, implicit $exec
    %45:sgpr_32 = S_MOV_B32 1333788672
    %46:vgpr_32 = nofpexcept V_FMA_F32_e64 0, killed %44, 0, killed %45, 0, killed %42, 0, 0, implicit $mode, implicit $exec
    %47:vgpr_32 = nofpexcept V_RCP_F32_e64 0, killed %46, 0, 0, implicit $mode, implicit $exec
    %48:sgpr_32 = S_MOV_B32 1602224124
    %49:vgpr_32 = nofpexcept V_MUL_F32_e64 0, killed %47, 0, killed %48, 0, 0, implicit $mode, implicit $exec
    %50:sgpr_32 = S_MOV_B32 796917760
    %51:vgpr_32 = nofpexcept V_MUL_F32_e64 0, %49, 0, killed %50, 0, 0, implicit $mode, implicit $exec
    %52:vgpr_32 = nofpexcept V_TRUNC_F32_e64 0, killed %51, 0, 0, implicit $mode, implicit $exec
    %53:sgpr_32 = S_MOV_B32 -813694976
    %54:vgpr_32 = nofpexcept V_FMA_F32_e64 0, %52, 0, killed %53, 0, %49, 0, 0, implicit $mode, implicit $exec
    %55:vgpr_32 = nofpexcept V_CVT_U32_F32_e64 0, killed %54, 0, 0, implicit $mode, implicit $exec
    %56:sreg_64 = S_MOV_B64 0
    %57:sreg_64 = S_SUB_U64_PSEUDO killed %56, %2, implicit-def dead $scc
    %58:sreg_32 = COPY %57.sub1
    %60:sreg_32 = COPY %55
    %59:sreg_32 = S_MUL_I32 %58, %60
    %61:sreg_32 = COPY %57.sub0
    %62:vgpr_32 = V_MUL_HI_U32_e64 %61, %55, implicit $exec
    %63:vgpr_32 = nofpexcept V_CVT_U32_F32_e64 0, %52, 0, 0, implicit $mode, implicit $exec
    %65:sreg_32 = COPY %63
    %64:sreg_32 = S_MUL_I32 %61, %65
    %67:sreg_32 = COPY %62
    %66:sreg_32 = S_ADD_I32 killed %67, killed %64, implicit-def dead $scc
    %68:sreg_32 = S_ADD_I32 killed %66, killed %59, implicit-def dead $scc
    %69:vgpr_32 = V_MUL_HI_U32_e64 %55, %68, implicit $exec
    %71:sreg_32 = COPY %55
    %70:sreg_32 = S_MUL_I32 %71, %68
    %72:sreg_64 = REG_SEQUENCE killed %70, %subreg.sub0, killed %69, %subreg.sub1
    %74:sreg_32 = COPY %55
    %73:sreg_32 = S_MUL_I32 %61, %74
    %75:vgpr_32 = V_MUL_HI_U32_e64 %55, %73, implicit $exec
    %76:sreg_32 = S_MOV_B32 0
    %77:sreg_64 = REG_SEQUENCE killed %75, %subreg.sub0, %76, %subreg.sub1
    %78:sreg_64 = S_ADD_U64_PSEUDO killed %77, killed %72, implicit-def dead $scc
    %79:sreg_32 = COPY %78.sub0
    %80:sreg_32 = COPY %78.sub1
    %81:vgpr_32 = V_MUL_HI_U32_e64 %63, %68, implicit $exec
    %82:vgpr_32 = V_MUL_HI_U32_e64 %63, %73, implicit $exec
    %84:sreg_32 = COPY %63
    %83:sreg_32 = S_MUL_I32 %84, %73
    %85:sreg_64 = REG_SEQUENCE killed %83, %subreg.sub0, killed %82, %subreg.sub1
    %86:sreg_32 = COPY %85.sub0
    %87:sreg_32 = COPY %85.sub1
    %88:sreg_32 = S_MOV_B32 0
    %89:sreg_32 = S_ADD_U32 killed %79, killed %86, implicit-def $scc
    %90:sreg_32 = S_ADDC_U32 killed %80, killed %87, implicit-def $scc, implicit $scc
    %92:sreg_32 = COPY %81
    %91:sreg_32 = S_ADDC_U32 killed %92, %88, implicit-def dead $scc, implicit $scc
    %94:sreg_32 = COPY %63
    %93:sreg_32 = S_MUL_I32 %94, %68
    %95:sreg_64 = REG_SEQUENCE killed %93, %subreg.sub0, killed %91, %subreg.sub1
    %96:sreg_64 = REG_SEQUENCE killed %89, %subreg.sub0, killed %90, %subreg.sub1
    %97:sreg_32 = COPY %96.sub1
    %98:sreg_64 = REG_SEQUENCE killed %97, %subreg.sub0, %88, %subreg.sub1
    %99:sreg_64 = S_ADD_U64_PSEUDO killed %98, killed %95, implicit-def dead $scc
    %100:sreg_32 = COPY %99.sub0
    %103:sreg_32 = COPY %55
    %101:sreg_32, %102:sreg_64_xexec = S_UADDO_PSEUDO %103, killed %100, implicit-def dead $scc
    %104:sreg_32 = COPY %99.sub1
    %107:sreg_32 = COPY %63
    %105:sreg_32, %106:sreg_64_xexec = S_ADD_CO_PSEUDO %107, killed %104, killed %102, implicit-def dead $scc
    %108:sreg_32 = S_MUL_I32 %61, %105
    %110:vgpr_32 = COPY %101
    %109:vgpr_32 = V_MUL_HI_U32_e64 %61, %110, implicit $exec
    %112:sreg_32 = COPY %109
    %111:sreg_32 = S_ADD_I32 killed %112, killed %108, implicit-def dead $scc
    %113:sreg_32 = S_MUL_I32 %58, %101
    %114:sreg_32 = S_ADD_I32 killed %111, killed %113, implicit-def dead $scc
    %116:vgpr_32 = COPY %114
    %115:vgpr_32 = V_MUL_HI_U32_e64 %105, %116, implicit $exec
    %117:sreg_32 = S_MUL_I32 %61, %101
    %119:vgpr_32 = COPY %117
    %118:vgpr_32 = V_MUL_HI_U32_e64 %105, %119, implicit $exec
    %120:sreg_32 = S_MUL_I32 %105, %117
    %121:sreg_64 = REG_SEQUENCE killed %120, %subreg.sub0, killed %118, %subreg.sub1
    %122:sreg_32 = COPY %121.sub0
    %123:sreg_32 = COPY %121.sub1
    %125:vgpr_32 = COPY %114
    %124:vgpr_32 = V_MUL_HI_U32_e64 %101, %125, implicit $exec
    %126:sreg_32 = S_MUL_I32 %101, %114
    %127:sreg_64 = REG_SEQUENCE killed %126, %subreg.sub0, killed %124, %subreg.sub1
    %129:vgpr_32 = COPY %117
    %128:vgpr_32 = V_MUL_HI_U32_e64 %101, %129, implicit $exec
    %130:sreg_64 = REG_SEQUENCE killed %128, %subreg.sub0, %76, %subreg.sub1
    %131:sreg_64 = S_ADD_U64_PSEUDO killed %130, killed %127, implicit-def dead $scc
    %132:sreg_32 = COPY %131.sub0
    %133:sreg_32 = COPY %131.sub1
    %134:sreg_32 = S_ADD_U32 killed %132, killed %122, implicit-def $scc
    %135:sreg_32 = S_ADDC_U32 killed %133, killed %123, implicit-def $scc, implicit $scc
    %137:sreg_32 = COPY %115
    %136:sreg_32 = S_ADDC_U32 killed %137, %88, implicit-def dead $scc, implicit $scc
    %138:sreg_32 = S_MUL_I32 %105, %114
    %139:sreg_64 = REG_SEQUENCE killed %138, %subreg.sub0, killed %136, %subreg.sub1
    %140:sreg_64 = REG_SEQUENCE killed %134, %subreg.sub0, killed %135, %subreg.sub1
    %141:sreg_32 = COPY %140.sub1
    %142:sreg_64 = REG_SEQUENCE killed %141, %subreg.sub0, %88, %subreg.sub1
    %143:sreg_64 = S_ADD_U64_PSEUDO killed %142, killed %139, implicit-def dead $scc
    %144:sreg_32 = COPY %143.sub0
    %145:sreg_32, %146:sreg_64_xexec = S_UADDO_PSEUDO %101, killed %144, implicit-def dead $scc
    %147:sreg_32 = COPY %143.sub1
    %148:sreg_32, %149:sreg_64_xexec = S_ADD_CO_PSEUDO %105, killed %147, killed %146, implicit-def dead $scc
    %150:sreg_32 = COPY %1.sub0
    %152:vgpr_32 = COPY %148
    %151:vgpr_32 = V_MUL_HI_U32_e64 %150, %152, implicit $exec
    %153:sreg_32 = S_MUL_I32 %150, %148
    %154:sreg_64 = REG_SEQUENCE killed %153, %subreg.sub0, killed %151, %subreg.sub1
    %156:vgpr_32 = COPY %145
    %155:vgpr_32 = V_MUL_HI_U32_e64 %150, %156, implicit $exec
    %157:sreg_64 = REG_SEQUENCE killed %155, %subreg.sub0, %76, %subreg.sub1
    %158:sreg_64 = S_ADD_U64_PSEUDO killed %157, killed %154, implicit-def dead $scc
    %159:sreg_32 = COPY %158.sub0
    %160:sreg_32 = COPY %158.sub1
    %161:sreg_32 = COPY %1.sub1
    %163:vgpr_32 = COPY %148
    %162:vgpr_32 = V_MUL_HI_U32_e64 %161, %163, implicit $exec
    %165:vgpr_32 = COPY %145
    %164:vgpr_32 = V_MUL_HI_U32_e64 %161, %165, implicit $exec
    %166:sreg_32 = S_MUL_I32 %161, %145
    %167:sreg_64 = REG_SEQUENCE killed %166, %subreg.sub0, killed %164, %subreg.sub1
    %168:sreg_32 = COPY %167.sub0
    %169:sreg_32 = COPY %167.sub1
    %170:sreg_32 = S_ADD_U32 killed %159, killed %168, implicit-def $scc
    %171:sreg_32 = S_ADDC_U32 killed %160, killed %169, implicit-def $scc, implicit $scc
    %173:sreg_32 = COPY %162
    %172:sreg_32 = S_ADDC_U32 killed %173, %88, implicit-def dead $scc, implicit $scc
    %174:sreg_32 = S_MUL_I32 %161, %148
    %175:sreg_64 = REG_SEQUENCE killed %174, %subreg.sub0, killed %172, %subreg.sub1
    %176:sreg_64 = REG_SEQUENCE killed %170, %subreg.sub0, killed %171, %subreg.sub1
    %177:sreg_32 = COPY %176.sub1
    %178:sreg_64 = REG_SEQUENCE killed %177, %subreg.sub0, %88, %subreg.sub1
    %179:sreg_64 = S_ADD_U64_PSEUDO killed %178, killed %175, implicit-def dead $scc
    %180:sreg_32 = COPY %179.sub1
    %181:sreg_32 = S_MUL_I32 %41, %180
    %182:sreg_32 = COPY %179.sub0
    %184:vgpr_32 = COPY %182
    %183:vgpr_32 = V_MUL_HI_U32_e64 %41, %184, implicit $exec
    %186:sreg_32 = COPY %183
    %185:sreg_32 = S_ADD_I32 killed %186, killed %181, implicit-def dead $scc
    %187:sreg_32 = S_MUL_I32 %43, %182
    %188:sreg_32 = S_ADD_I32 killed %185, killed %187, implicit-def dead $scc
    %189:sreg_32 = S_SUB_I32 %161, %188, implicit-def dead $scc
    %190:sreg_32 = S_MUL_I32 %41, %182
    %191:sreg_32, %192:sreg_64_xexec = S_USUBO_PSEUDO %150, killed %190, implicit-def dead $scc
    %193:sreg_32, %194:sreg_64_xexec = S_SUB_CO_PSEUDO killed %189, %43, %192, implicit-def dead $scc
    %195:sreg_32, %196:sreg_64_xexec = S_USUBO_PSEUDO %191, %41, implicit-def dead $scc
    %197:sreg_32, %198:sreg_64_xexec = S_SUB_CO_PSEUDO killed %193, %88, killed %196, implicit-def dead $scc
    S_CMP_GE_U32 %197, %43, implicit-def $scc
    %199:sreg_32 = S_MOV_B32 -1
    %200:sreg_32 = S_CSELECT_B32 %199, %88, implicit $scc
    S_CMP_GE_U32 killed %195, %41, implicit-def $scc
    %201:sreg_32 = S_CSELECT_B32 %199, %88, implicit $scc
    S_CMP_EQ_U32 %197, %43, implicit-def $scc
    %202:sreg_32 = S_CSELECT_B32 killed %201, killed %200, implicit $scc
    %203:sreg_32 = COPY killed %202
    %204:sreg_64 = REG_SEQUENCE %182, %subreg.sub0, %180, %subreg.sub1
    %205:sreg_64 = S_MOV_B64 1
    %206:sreg_64 = S_ADD_U64_PSEUDO %204, killed %205, implicit-def dead $scc
    %207:sreg_32 = COPY %206.sub0
    %208:sreg_64 = S_MOV_B64 2
    %209:sreg_64 = S_ADD_U64_PSEUDO %204, killed %208, implicit-def dead $scc
    %210:sreg_32 = COPY %209.sub0
    S_CMP_LG_U32 killed %203, %88, implicit-def $scc
    %211:sreg_32 = S_CSELECT_B32 killed %210, killed %207, implicit $scc
    %212:sreg_32 = COPY %206.sub1
    %213:sreg_32 = COPY %209.sub1
    %214:sreg_32 = S_CSELECT_B32 killed %213, killed %212, implicit $scc
    %215:sreg_32, %216:sreg_64_xexec = S_SUB_CO_PSEUDO %161, %188, %192, implicit-def dead $scc
    S_CMP_GE_U32 %215, %43, implicit-def $scc
    %217:sreg_32 = S_CSELECT_B32 %199, %88, implicit $scc
    S_CMP_GE_U32 %191, %41, implicit-def $scc
    %218:sreg_32 = S_CSELECT_B32 %199, %88, implicit $scc
    S_CMP_EQ_U32 %215, %43, implicit-def $scc
    %219:sreg_32 = S_CSELECT_B32 killed %218, killed %217, implicit $scc
    %220:sreg_32 = COPY killed %219
    S_CMP_LG_U32 killed %220, %88, implicit-def $scc
    %221:sreg_32 = S_CSELECT_B32 killed %214, %180, implicit $scc
    %222:sreg_32 = S_CSELECT_B32 killed %211, %182, implicit $scc
    %223:sreg_64 = REG_SEQUENCE killed %222, %subreg.sub0, killed %221, %subreg.sub1
    %40:sreg_64 = S_MOV_B64 0
    %6:sreg_64 = COPY %223
    S_BRANCH %bb.1
  
  bb.4 (%ir-block.14):
    %7:sreg_64 = PHI %3, %bb.1, %5, %bb.2
    %257:sreg_32 = COPY %0.sub1
    %258:sreg_32 = COPY %0.sub0
    %259:sreg_64 = REG_SEQUENCE killed %258, %subreg.sub0, killed %257, %subreg.sub1
    %260:sreg_32 = COPY %259.sub1
    %261:sreg_32 = COPY %259.sub0
    %262:sreg_32 = S_MOV_B32 61440
    %263:sreg_32 = S_MOV_B32 -1
    %264:sgpr_128 = REG_SEQUENCE killed %261, %subreg.sub0, killed %260, %subreg.sub1, killed %263, %subreg.sub2, killed %262, %subreg.sub3
    %265:vreg_64 = COPY %7
    BUFFER_STORE_DWORDX2_OFFSET %265, killed %264, 0, 0, 0, 0, implicit $exec :: (store (s64) into %ir.16, addrspace 1)
    S_ENDPGM 0
...
