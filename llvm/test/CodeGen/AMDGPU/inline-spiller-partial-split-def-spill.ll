; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -greedy-compact-slot-indexes=false -split-spill-mode=speed -verify-machineinstrs -stop-after=greedy -o - %s | FileCheck %s

; The buffer resource built from %0 is spilled before the loop and reloaded on
; every iteration. Live range splitting creates a sibling of it that only defines
; sub3. All siblings share one stack slot, so the full-width spill of that
; sibling must not write its undefined lanes over the resource in the slot.
; Reload the slot before the partial copy so that the spill writes the other
; lanes back unchanged, and do not hoist the spill to the copy source.

; CHECK-LABEL: name: partial_split_def_spill
; CHECK: SI_SPILL_S128_SAVE %{{[0-9]+}}, %stack.0
; CHECK: bb.1
; CHECK: [[RSRC:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE %stack.0
; CHECK-NEXT: BUFFER_LOAD_DWORDX4_OFFEN %{{[0-9]+}}, [[RSRC]],
; CHECK-NEXT: undef [[SPLIT:%[0-9]+]].sub3:sgpr_128 = lr-split COPY [[RSRC]].sub3
; CHECK-NOT: SI_SPILL_S128_SAVE
; CHECK: [[RMW:%[0-9]+]]:sgpr_128 = SI_SPILL_S128_RESTORE %stack.0
; CHECK-NEXT: [[RMW]].sub3:sgpr_128 = lr-split COPY [[SPLIT]].sub3
; CHECK-NEXT: SI_SPILL_S128_SAVE [[RMW]], %stack.0

define amdgpu_kernel void @partial_split_def_spill(ptr addrspace(1) inreg %0, ptr addrspace(1) %1, i32 %2, <4 x float> %3) #0 {
.lr.ph:
  %4 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %5 = insertelement <4 x i32> zeroinitializer, i32 %4, i64 0
  br label %6

6:                                                ; preds = %6, %.lr.ph
  %7 = phi i32 [ 1, %.lr.ph ], [ 0, %6 ]
  %8 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i32(ptr addrspace(1) %0, i16 0, i32 2147483646, i32 1)
  %9 = load <1 x i32>, ptr addrspace(1) null, align 4
  %10 = extractelement <1 x i32> %9, i64 0
  %11 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %8, i32 %10, i32 0, i32 0)
  %12 = sdiv i32 %7, %2
  %13 = insertelement <4 x i32> zeroinitializer, i32 %12, i64 0
  %14 = shufflevector <4 x i32> %13, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
  %15 = shufflevector <4 x i32> %5, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
  %16 = or <4 x i32> %15, <i32 40, i32 41, i32 42, i32 43>
  %17 = or <4 x i32> %14, %16
  %18 = or <4 x i32> %5, <i32 48, i32 49, i32 50, i32 51>
  %19 = or <4 x i32> %14, %18
  %20 = or <4 x i32> %15, <i32 56, i32 57, i32 58, i32 59>
  %21 = or <4 x i32> %14, %20
  %22 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i32(ptr addrspace(1) %1, i16 0, i32 1, i32 0)
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %22, ptr addrspace(3) null, i32 1, i32 0, i32 0, i32 0, i32 0)
  %23 = or <4 x i32> %15, <i32 24, i32 25, i32 26, i32 27>
  %24 = or <4 x i32> %14, %23
  %25 = sub <4 x i32> zeroinitializer, %24
  %26 = icmp ult <4 x i32> %25, splat (i32 101)
  %27 = or <4 x i32> %15, <i32 32, i32 33, i32 34, i32 35>
  %28 = or <4 x i32> %14, %27
  %29 = sub <4 x i32> zeroinitializer, %28
  %30 = icmp ult <4 x i32> %29, splat (i32 101)
  %31 = sub <4 x i32> zeroinitializer, %17
  %32 = icmp ult <4 x i32> %31, splat (i32 101)
  %33 = sub <4 x i32> zeroinitializer, %19
  %34 = icmp ult <4 x i32> %33, splat (i32 101)
  %35 = sub <4 x i32> zeroinitializer, %21
  %36 = icmp ult <4 x i32> %35, splat (i32 101)
  %37 = icmp slt <4 x i32> %24, zeroinitializer
  %38 = icmp slt <4 x i32> %28, zeroinitializer
  %39 = icmp slt <4 x i32> %17, zeroinitializer
  %40 = icmp slt <4 x i32> %19, zeroinitializer
  %41 = icmp slt <4 x i32> %21, zeroinitializer
  %42 = and <4 x i1> %37, %26
  %43 = and <4 x i1> %38, %30
  %44 = and <4 x i1> %40, %34
  %45 = and <4 x i1> %41, %36
  %46 = select <4 x i1> %42, <4 x float> %3, <4 x float> zeroinitializer
  %47 = bitcast <4 x i32> %11 to <8 x bfloat>
  %48 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> zeroinitializer, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %49 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> %47, <16 x float> %48, i32 0, i32 0, i32 0)
  %50 = shufflevector <16 x float> %49, <16 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %51 = bitcast <4 x i32> %13 to <4 x float>
  %52 = fadd <4 x float> %50, %51
  %53 = select <4 x i1> %43, <4 x float> %52, <4 x float> zeroinitializer
  %54 = and <4 x i1> %39, %32
  %55 = shufflevector <16 x float> %48, <16 x float> zeroinitializer, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %56 = select <4 x i1> <i1 false, i1 true, i1 false, i1 false>, <4 x float> %3, <4 x float> zeroinitializer
  %57 = fadd <4 x float> %55, %56
  %58 = fmul <4 x float> %46, %53
  %59 = select <4 x i1> %54, <4 x float> zeroinitializer, <4 x float> splat (float -inf)
  %60 = fmul <4 x float> %58, %59
  %61 = select <4 x i1> %44, <4 x float> splat (float +qnan), <4 x float> zeroinitializer
  %62 = fmul <4 x float> %60, %61
  %63 = select <4 x i1> %45, <4 x float> %57, <4 x float> zeroinitializer
  %64 = fmul <4 x float> %62, %63
  %65 = tail call float @llvm.vector.reduce.fmaximum.v4f32(<4 x float> %64)
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %65, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %66 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i32(ptr addrspace(1) null, i16 0, i32 0, i32 1)
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> zeroinitializer, ptr addrspace(8) %66, i32 0, i32 0, i32 0)
  br label %6
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i32(ptr addrspace(1) readnone, i16, i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fmaximum.v4f32(<4 x float>) #6

; uselistorder directives
uselistorder ptr @llvm.amdgcn.mfma.f32.32x32x16.bf16, { 1, 0 }
uselistorder ptr @llvm.amdgcn.make.buffer.rsrc.p8.p1.i32, { 2, 1, 0 }

attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-no-queue-ptr" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
