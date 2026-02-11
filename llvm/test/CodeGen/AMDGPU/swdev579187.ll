; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 < %s | FileCheck %s

; This testcase hit a situation where reverting scheduling after the scheduler's
; rematerialization stage would cause incoherent MI and slot orders, hitting an
; assert later on in the RP trackers. This is fixed by ensuring that all MIs that
; should be alive at the end of the stage are marked non-debug before scheduling
; is reverted, but is extremely sensitive to scheduling and rematerialization
; decisions.  

%f8 = type { i8 }

@shared = external addrspace(3) global [16384 x i8]

define amdgpu_kernel void @test_revert_schedule(i32 %arg0, i32 %arg1, ptr addrspace(3) %p15, ptr addrspace(3) %lds, ptr addrspace(3) %0, ptr addrspace(3) %p14, i32 %1, ptr addrspace(3) %2, ptr addrspace(3) %3, i32 %4, i32 %5, ptr addrspace(3) %p12, i32 %x7, ptr addrspace(3) %p7, i32 %a7, ptr addrspace(3) %6, i1 %loopcond, i32 %a5, i32 %a3, i32 %a4, i32 %a2, <4 x i8> %7, <4 x i8> %8) #4 {
; CHECK-LABEL: test_revert_schedule:
entry:
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %10 = lshr i32 %9, 3
  %n6 = and i32 %9, 1
  %m3 = shl i32 %n6, 1
  %n4 = or i32 %9, %arg0
  %d2 = or i32 %9, 1
  %s = and i32 %9, 3
  %d = or i32 %s, 20
  %n = or i32 %d, 1
  %11 = getelementptr i8, ptr addrspace(3) %lds, i32 %n
  %p6 = getelementptr %f8, ptr addrspace(3) %11, i32 %m3
  %12 = getelementptr i8, ptr addrspace(3) %p6, i32 4
  %d3 = or i32 %s, 24
  %x = xor i32 %9, 1
  %x2 = xor i32 %x, %d2
  %x6 = xor i32 1, %s
  %n2 = or i32 %x2, %x6
  %13 = shl i32 %n2, 1
  %x4 = xor i32 1, %d
  %n3 = or i32 %x4, %arg0
  %x3 = xor i32 1, %d3
  %x5 = and i32 %9, 31
  br label %loop

loop:                                    ; preds = %loop, %entry
  %phi0 = phi i32 [ 0, %entry ], [ %x5, %loop ]
  %14 = phi <4 x i8> [ zeroinitializer, %entry ], [ %29, %loop ]
  %15 = phi <4 x i8> [ zeroinitializer, %entry ], [ %27, %loop ]
  %p19 = getelementptr %f8, ptr addrspace(3) %lds, i32 %phi0
  store <4 x i32> zeroinitializer, ptr addrspace(3) %p19, align 16
  %16 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %15, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac = bitcast <16 x i8> %16 to <2 x i64>
  %17 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac2 = bitcast <16 x i8> %17 to <2 x i64>
  %ae = extractelement <2 x i64> %ac, i64 0
  %ae2 = extractelement <2 x i64> %ac2, i64 0
  %18 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae, i64 0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %19 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae2, i64 0, <4 x float> %18, i32 0, i32 0, i32 0)
  %20 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %21 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 1, i32 0, i32 0)
  %22 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 14336, i32 0, i32 0)
  %23 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 15360, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.waitcnt(i32 0)
  %24 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %25 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %arg0, i32 0, i32 0)
  %26 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 1, i32 0, i32 0)
  %p3 = getelementptr i8, ptr addrspace(3) %3, i32 %x
  %27 = load <4 x i8>, ptr addrspace(3) %p3, align 4
  %n5 = sub i32 0, %x6
  %28 = shl i32 %n5, 1
  %p = getelementptr i8, ptr addrspace(3) %3, i32 %28
  %29 = load <4 x i8>, ptr addrspace(3) %p, align 4
  %30 = load <4 x i8>, ptr addrspace(3) null, align 8
  tail call void @llvm.amdgcn.sched.group.barrier(i32 0, i32 0, i32 0)
  br i1 %loopcond, label %loop, label %exit

exit:                                             ; preds = %loop
  %31 = shl i32 %9, 1
  %32 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %33 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 16384, i32 0, i32 0)
  %34 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 17408, i32 0, i32 0)
  %35 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %x7, i32 0, i32 0)
  %36 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %1, i32 0, i32 0)
  %37 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 19456, i32 0, i32 0)
  %d4 = sdiv i32 %5, 64
  %m5 = shl i32 %d4, 1
  %a = or i32 19456, %m5
  %38 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %a, i32 0, i32 0)
  %39 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %a7, i32 0, i32 0)
  %40 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 20480, i32 0, i32 0)
  %41 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 1, i32 0, i32 0)
  %42 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %a3, i32 0, i32 0)
  %43 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %a2, i32 0, i32 0)
  %44 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 22528, i32 0, i32 0)
  %45 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 23552, i32 0, i32 0)
  %46 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %arg0, i32 0, i32 0)
  store <4 x i32> %24, ptr addrspace(3) null, align 16
  %p18 = getelementptr %f8, ptr addrspace(3) @shared, i32 %1
  store <4 x i32> %25, ptr addrspace(3) %p18, align 16
  %p17 = getelementptr %f8, ptr addrspace(3) %p18, i32 512
  %47 = or i32 %x5, 1
  %p16 = getelementptr %f8, ptr addrspace(3) %p17, i32 %47
  store <4 x i32> %26, ptr addrspace(3) %p16, align 16
  %48 = load <4 x i8>, ptr addrspace(3) %p15, align 8
  %bc16 = bitcast <4 x i32> %20 to <2 x i64>
  %be21 = extractelement <2 x i64> %bc16, i64 0
  %bc11 = bitcast <4 x i32> %22 to <2 x i64>
  %be22 = extractelement <2 x i64> %bc11, i64 0
  %as6 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %27, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac7 = bitcast <16 x i8> %as6 to <2 x i64>
  %ae6 = extractelement <2 x i64> %ac7, i64 0
  %49 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %50 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae6, i64 0, <4 x float> %49, i32 0, i32 0, i32 0)
  %as4 = shufflevector <4 x i8> %30, <4 x i8> zeroinitializer, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ac6 = bitcast <16 x i8> %as4 to <2 x i64>
  %ae7 = extractelement <2 x i64> %ac6, i64 1
  %51 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 72340172838076673, i64 0, <4 x float> %50, i32 0, i32 0, i32 0)
  %52 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %51, i32 0, i32 0, i32 0)
  %53 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae7, i64 0, <4 x float> %52, i32 0, i32 0, i32 0)
  %54 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %53, i32 0, i32 0, i32 0)
  %55 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %54, i32 0, i32 0, i32 0)
  %56 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be21, <4 x float> %55, i32 0, i32 0, i32 0)
  %bc4 = bitcast <4 x i32> %21 to <2 x i64>
  %be29 = extractelement <2 x i64> %bc4, i64 1
  %be14 = extractelement <2 x i64> %bc4, i64 0
  %57 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be21, <4 x float> %56, i32 0, i32 0, i32 0)
  %58 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be14, <4 x float> %57, i32 0, i32 0, i32 0)
  %59 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be29, <4 x float> %58, i32 0, i32 0, i32 0)
  %60 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %59, i32 0, i32 0, i32 0)
  %61 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %60, i32 0, i32 0, i32 0)
  %as8 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %8, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac9 = bitcast <16 x i8> %as8 to <2 x i64>
  %as13 = shufflevector <4 x i8> %7, <4 x i8> zeroinitializer, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ac18 = bitcast <16 x i8> %as13 to <2 x i64>
  %ae18 = extractelement <2 x i64> %ac18, i64 1
  %62 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %19, i32 0, i32 0, i32 0)
  %ae9 = extractelement <2 x i64> %ac9, i64 0
  %63 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae18, i64 0, <4 x float> %62, i32 0, i32 0, i32 0)
  %64 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae9, i64 0, <4 x float> %63, i32 0, i32 0, i32 0)
  %65 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %64, i32 0, i32 0, i32 0)
  %as5 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %29, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac8 = bitcast <16 x i8> %as5 to <2 x i64>
  %ae8 = extractelement <2 x i64> %ac8, i64 0
  %66 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %65, i32 0, i32 0, i32 0)
  %67 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %66, i32 0, i32 0, i32 0)
  %68 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae8, i64 0, <4 x float> %67, i32 0, i32 0, i32 0)
  %69 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %68, i32 0, i32 0, i32 0)
  %70 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %69, i32 0, i32 0, i32 0)
  %71 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %70, i32 0, i32 0, i32 0)
  %72 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %71, i32 0, i32 0, i32 0)
  %73 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %72, i32 0, i32 0, i32 0)
  %bc13 = bitcast <4 x i32> %23 to <2 x i64>
  %74 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be22, <4 x float> %73, i32 0, i32 0, i32 0)
  %be20 = extractelement <2 x i64> %bc13, i64 0
  %be26 = extractelement <2 x i64> %bc11, i64 1
  %75 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be26, <4 x float> %74, i32 0, i32 0, i32 0)
  %76 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be20, <4 x float> %75, i32 0, i32 0, i32 0)
  %77 = load <4 x i8>, ptr addrspace(3) null, align 8
  %78 = getelementptr i8, ptr addrspace(3) @shared, i32 %28
  %p10 = getelementptr %f8, ptr addrspace(3) %78, i32 %arg1
  %79 = load <4 x i8>, ptr addrspace(3) %p10, align 4
  %80 = getelementptr i8, ptr addrspace(3) @shared, i32 %n4
  %p11 = getelementptr %f8, ptr addrspace(3) %80, i32 %arg1
  %81 = getelementptr i8, ptr addrspace(3) %p11, i32 4
  %82 = load <4 x i8>, ptr addrspace(3) %81, align 4
  %83 = load <4 x i8>, ptr addrspace(3) %p11, align 8
  %p4 = getelementptr %f8, ptr addrspace(3) %0, i32 %m3
  %84 = load <4 x i8>, ptr addrspace(3) %p4, align 4
  %85 = getelementptr i8, ptr addrspace(3) %lds, i32 %n5
  %p8 = getelementptr %f8, ptr addrspace(3) %85, i32 %arg1
  %86 = load <4 x i8>, ptr addrspace(3) %p8, align 4
  %87 = load <4 x i8>, ptr addrspace(3) %85, align 8
  %88 = load <4 x i8>, ptr addrspace(3) %11, align 4
  %p5 = getelementptr %f8, ptr addrspace(3) %lds, i32 %m3
  %89 = load <4 x i8>, ptr addrspace(3) %p5, align 4
  %p13 = getelementptr i8, ptr addrspace(3) inttoptr (i32 8192 to ptr addrspace(3)), i32 %13
  %90 = load <4 x i8>, ptr addrspace(3) %p13, align 4
  %p2 = getelementptr i8, ptr addrspace(3) %3, i32 %n3
  %91 = load <4 x i8>, ptr addrspace(3) %p2, align 16
  %p9 = getelementptr i8, ptr addrspace(3) inttoptr (i32 8192 to ptr addrspace(3)), i32 %x3
  %92 = load <4 x i8>, ptr addrspace(3) %p9, align 16
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %bc = bitcast <4 x i32> %32 to <2 x i64>
  %be = extractelement <2 x i64> %bc, i64 1
  %as14 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %79, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %bc18 = bitcast <4 x i32> %35 to <2 x i64>
  %be30 = extractelement <2 x i64> %bc18, i64 0
  %ac17 = bitcast <16 x i8> %as14 to <2 x i64>
  %ae17 = extractelement <2 x i64> %ac17, i64 1
  %as16 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %82, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %as10 = shufflevector <4 x i8> %83, <4 x i8> %48, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ac15 = bitcast <16 x i8> %as16 to <2 x i64>
  %ae13 = extractelement <2 x i64> %ac15, i64 0
  %bc15 = bitcast <4 x i32> %36 to <2 x i64>
  %be8 = extractelement <2 x i64> %bc15, i64 0
  %ac10 = bitcast <16 x i8> %as10 to <2 x i64>
  %ae10 = extractelement <2 x i64> %ac10, i64 1
  %as9 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %84, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %bc7 = bitcast <4 x i32> %38 to <2 x i64>
  %be7 = extractelement <2 x i64> %bc7, i64 0
  %ac16 = bitcast <16 x i8> %as9 to <2 x i64>
  %ae11 = extractelement <2 x i64> %ac16, i64 1
  %as11 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %86, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %as7 = shufflevector <4 x i8> %87, <4 x i8> %77, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ac11 = bitcast <16 x i8> %as11 to <2 x i64>
  %ae15 = extractelement <2 x i64> %ac11, i64 0
  %bc6 = bitcast <4 x i32> %39 to <2 x i64>
  %be17 = extractelement <2 x i64> %bc6, i64 0
  %ac14 = bitcast <16 x i8> %as7 to <2 x i64>
  %ae12 = extractelement <2 x i64> %ac14, i64 1
  %as12 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %88, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac13 = bitcast <16 x i8> %as12 to <2 x i64>
  %ae16 = extractelement <2 x i64> %ac13, i64 0
  %bc14 = bitcast <4 x i32> %42 to <2 x i64>
  %be28 = extractelement <2 x i64> %bc14, i64 0
  %as15 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %89, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac12 = bitcast <16 x i8> %as15 to <2 x i64>
  %ae14 = extractelement <2 x i64> %ac12, i64 0
  %bc12 = bitcast <4 x i32> %43 to <2 x i64>
  %be19 = extractelement <2 x i64> %bc12, i64 0
  %bc2 = bitcast <4 x i32> %46 to <2 x i64>
  %be15 = extractelement <2 x i64> %bc2, i64 1
  %93 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be15, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %bc17 = bitcast <4 x i32> %33 to <2 x i64>
  %be32 = extractelement <2 x i64> %bc17, i64 0
  %bc19 = bitcast <4 x i32> %34 to <2 x i64>
  %be33 = extractelement <2 x i64> %bc19, i64 0
  %94 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae17, i64 %be33, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %95 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae13, i64 0, <4 x float> %94, i32 0, i32 0, i32 0)
  %96 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae10, i64 0, <4 x float> %95, i32 0, i32 0, i32 0)
  %bc5 = bitcast <4 x i32> %37 to <2 x i64>
  %be10 = extractelement <2 x i64> %bc5, i64 0
  %97 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be10, <4 x float> %96, i32 0, i32 0, i32 0)
  %be12 = extractelement <2 x i64> %bc5, i64 1
  %98 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae11, i64 %be12, <4 x float> %97, i32 0, i32 0, i32 0)
  %bc10 = bitcast <4 x i32> %40 to <2 x i64>
  %be16 = extractelement <2 x i64> %bc10, i64 0
  %99 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae15, i64 %be16, <4 x float> %98, i32 0, i32 0, i32 0)
  %be6 = extractelement <2 x i64> %bc10, i64 1
  %100 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae12, i64 %be6, <4 x float> %99, i32 0, i32 0, i32 0)
  %bc8 = bitcast <4 x i32> %41 to <2 x i64>
  %be3 = extractelement <2 x i64> %bc8, i64 0
  %101 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae16, i64 %be3, <4 x float> %100, i32 0, i32 0, i32 0)
  %be11 = extractelement <2 x i64> %bc8, i64 1
  %102 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be11, <4 x float> %101, i32 0, i32 0, i32 0)
  %bc9 = bitcast <4 x i32> %44 to <2 x i64>
  %103 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae14, i64 0, <4 x float> %102, i32 0, i32 0, i32 0)
  %be18 = extractelement <2 x i64> %bc9, i64 1
  %104 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be18, <4 x float> %103, i32 0, i32 0, i32 0)
  %bc3 = bitcast <4 x i32> %45 to <2 x i64>
  %be4 = extractelement <2 x i64> %bc3, i64 0
  %105 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be4, <4 x float> %104, i32 0, i32 0, i32 0)
  %be25 = extractelement <2 x i64> %bc3, i64 1
  %106 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be25, <4 x float> %105, i32 0, i32 0, i32 0)
  %be2 = extractelement <2 x i64> %bc, i64 0
  %107 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 72340172838076673, i64 %be2, <4 x float> %61, i32 0, i32 0, i32 0)
  %108 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be, <4 x float> %107, i32 0, i32 0, i32 0)
  %109 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be30, <4 x float> %108, i32 0, i32 0, i32 0)
  %be31 = extractelement <2 x i64> %bc18, i64 1
  %110 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be31, <4 x float> %109, i32 0, i32 0, i32 0)
  %111 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be8, <4 x float> %110, i32 0, i32 0, i32 0)
  %be23 = extractelement <2 x i64> %bc15, i64 1
  %112 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be23, <4 x float> %111, i32 0, i32 0, i32 0)
  %113 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be7, <4 x float> %112, i32 0, i32 0, i32 0)
  %be5 = extractelement <2 x i64> %bc7, i64 1
  %114 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be5, <4 x float> %113, i32 0, i32 0, i32 0)
  %as = shufflevector <4 x i8> zeroinitializer, <4 x i8> %90, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %115 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be17, <4 x float> %114, i32 0, i32 0, i32 0)
  %ac4 = bitcast <16 x i8> %as to <2 x i64>
  %ae4 = extractelement <2 x i64> %ac4, i64 1
  %be13 = extractelement <2 x i64> %bc6, i64 1
  %116 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae4, i64 %be13, <4 x float> %115, i32 0, i32 0, i32 0)
  %as3 = shufflevector <4 x i8> %91, <4 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac5 = bitcast <16 x i8> %as3 to <2 x i64>
  %ae3 = extractelement <2 x i64> %ac5, i64 0
  %117 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be28, <4 x float> %116, i32 0, i32 0, i32 0)
  %be9 = extractelement <2 x i64> %bc14, i64 1
  %118 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be9, <4 x float> %117, i32 0, i32 0, i32 0)
  %as2 = shufflevector <4 x i8> %92, <4 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac3 = bitcast <16 x i8> %as2 to <2 x i64>
  %ae5 = extractelement <2 x i64> %ac3, i64 0
  %be27 = extractelement <2 x i64> %bc2, i64 0
  %119 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae5, i64 %be19, <4 x float> %118, i32 0, i32 0, i32 0)
  %120 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be27, <4 x float> %119, i32 0, i32 0, i32 0)
  %121 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be32, <4 x float> %76, i32 0, i32 0, i32 0)
  %be24 = extractelement <2 x i64> %bc9, i64 0
  %122 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae3, i64 0, <4 x float> %121, i32 0, i32 0, i32 0)
  %123 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be24, <4 x float> %122, i32 0, i32 0, i32 0)
  %n7 = and i32 %10, 31
  %m4 = and i32 %31, 56
  %m = mul i32 %n7, %arg0
  %a6 = or i32 %m, %m4
  %ce4 = extractelement <4 x float> %93, i64 0
  %m2 = and i32 %9, 48
  %124 = getelementptr float, ptr addrspace(3) @shared, i32 %m2
  %p21 = getelementptr float, ptr addrspace(3) %124, i32 %n6
  store float %ce4, ptr addrspace(3) null, align 4
  %ce3 = extractelement <4 x float> %120, i64 0
  store float %ce3, ptr addrspace(3) %p21, align 4
  %ce2 = extractelement <4 x float> %106, i64 0
  store float %ce2, ptr addrspace(3) null, align 4
  %ce = extractelement <4 x float> %123, i64 0
  store float %ce, ptr addrspace(3) %p15, align 4
  %sx = sext i32 %a6 to i64
  %p20 = getelementptr i16, ptr addrspace(1) null, i64 %sx
  %125 = atomicrmw fadd ptr addrspace(1) %p20, <2 x bfloat> zeroinitializer syncscope("agent") monotonic, align 4
  ret void

  uselistorder i32 %9, { 0, 1, 2, 3, 6, 4, 5, 7, 8 }
}

attributes #4 = { "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="2" }
