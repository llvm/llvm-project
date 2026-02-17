; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 -debug-only=machine-scheduler 2>&1 < %s | FileCheck -check-prefix=DEBUG %s
; REQUIRES: asserts

; This testcase hit a situation where reverting scheduling after the scheduler's
; rematerialization stage would cause incoherent MI and slot orders, hitting an
; assert later on in the RP trackers. This is fixed by ensuring that all MIs that
; should be alive at the end of the stage are marked non-debug before scheduling
; is reverted, but is extremely sensitive to scheduling and rematerialization
; decisions.

%f8 = type { i8 }

@shared = external addrspace(3) global [16384 x i8]

define amdgpu_kernel void @test_revert_schedule(i32 %arg0, i32 %arg1, ptr addrspace(3) %p15, ptr addrspace(3) %lds, ptr addrspace(3) %arg, ptr addrspace(3) %p14, i32 %arg2, ptr addrspace(3) %arg3, ptr addrspace(3) %arg4, i32 %arg5, i32 %arg6, ptr addrspace(3) %p12, i32 %x7, ptr addrspace(3) %p7, i32 %a7, ptr addrspace(3) %arg7, i1 %loopcond, i32 %a5, i32 %a3, i32 %a4, i32 %a2, <4 x i8> %arg8, <4 x i8> %arg9) #0 {
; DEBUG: test_revert_schedule
; DEBUG: Region 3 cannot meet occupancy target, interrupting re-scheduling in all regions
; DEBUG: ==== ROLLBACK ====
entry:
  %i = tail call i32 @llvm.amdgcn.workitem.id.x()
  %i10 = lshr i32 %i, 3
  %n6 = and i32 %i, 1
  %m3 = shl i32 %n6, 1
  %n4 = or i32 %i, %arg0
  %d2 = or i32 %i, 1
  %s = and i32 %i, 3
  %d = or i32 %s, 20
  %n = or i32 %d, 1
  %i11 = getelementptr i8, ptr addrspace(3) %lds, i32 %n
  %p6 = getelementptr %f8, ptr addrspace(3) %i11, i32 %m3
  %i12 = getelementptr i8, ptr addrspace(3) %p6, i32 4
  %d3 = or i32 %s, 24
  %x = xor i32 %i, 1
  %x2 = xor i32 %x, %d2
  %x6 = xor i32 1, %s
  %n2 = or i32 %x2, %x6
  %i13 = shl i32 %n2, 1
  %x4 = xor i32 1, %d
  %n3 = or i32 %x4, %arg0
  %x3 = xor i32 1, %d3
  %x5 = and i32 %i, 31
  br label %loop

loop:                                             ; preds = %loop, %entry
  %phi0 = phi i32 [ 0, %entry ], [ %x5, %loop ]
  %i14 = phi <4 x i8> [ zeroinitializer, %entry ], [ %i29, %loop ]
  %i15 = phi <4 x i8> [ zeroinitializer, %entry ], [ %i27, %loop ]
  %p19 = getelementptr %f8, ptr addrspace(3) %lds, i32 %phi0
  store <4 x i32> zeroinitializer, ptr addrspace(3) %p19, align 16
  %i16 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i15, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac = bitcast <16 x i8> %i16 to <2 x i64>
  %i17 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac2 = bitcast <16 x i8> %i17 to <2 x i64>
  %ae = extractelement <2 x i64> %ac, i64 0
  %ae2 = extractelement <2 x i64> %ac2, i64 0
  %i18 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae, i64 0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %i19 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae2, i64 0, <4 x float> %i18, i32 0, i32 0, i32 0)
  %i20 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i21 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 1, i32 0, i32 0)
  %i22 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 14336, i32 0, i32 0)
  %i23 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 15360, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.waitcnt(i32 0)
  %i24 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i25 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %arg0, i32 0, i32 0)
  %i26 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 1, i32 0, i32 0)
  %p3 = getelementptr i8, ptr addrspace(3) %arg4, i32 %x
  %i27 = load <4 x i8>, ptr addrspace(3) %p3, align 4
  %n5 = sub i32 0, %x6
  %i28 = shl i32 %n5, 1
  %p = getelementptr i8, ptr addrspace(3) %arg4, i32 %i28
  %i29 = load <4 x i8>, ptr addrspace(3) %p, align 4
  %i30 = load <4 x i8>, ptr addrspace(3) zeroinitializer, align 8
  tail call void @llvm.amdgcn.sched.group.barrier(i32 0, i32 0, i32 0)
  br i1 %loopcond, label %loop, label %exit

exit:                                             ; preds = %loop
  %i31 = shl i32 %i, 1
  %i32 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i33 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 16384, i32 0, i32 0)
  %i34 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 17408, i32 0, i32 0)
  %i35 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %x7, i32 0, i32 0)
  %i36 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %arg2, i32 0, i32 0)
  %i37 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 19456, i32 0, i32 0)
  %d4 = sdiv i32 %arg6, 64
  %m5 = shl i32 %d4, 1
  %a = or i32 19456, %m5
  %i38 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %a, i32 0, i32 0)
  %i39 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %a7, i32 0, i32 0)
  %i40 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 20480, i32 0, i32 0)
  %i41 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 1, i32 0, i32 0)
  %i42 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %a3, i32 0, i32 0)
  %i43 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %a2, i32 0, i32 0)
  %i44 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 22528, i32 0, i32 0)
  %i45 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 23552, i32 0, i32 0)
  %i46 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %arg0, i32 0, i32 0)
  store <4 x i32> %i24, ptr addrspace(3) zeroinitializer, align 16
  %p18 = getelementptr %f8, ptr addrspace(3) @shared, i32 %arg2
  store <4 x i32> %i25, ptr addrspace(3) %p18, align 16
  %p17 = getelementptr %f8, ptr addrspace(3) %p18, i32 512
  %i47 = or i32 %x5, 1
  %p16 = getelementptr %f8, ptr addrspace(3) %p17, i32 %i47
  store <4 x i32> %i26, ptr addrspace(3) %p16, align 16
  %i48 = load <4 x i8>, ptr addrspace(3) %p15, align 8
  %bc16 = bitcast <4 x i32> %i20 to <2 x i64>
  %be21 = extractelement <2 x i64> %bc16, i64 0
  %bc11 = bitcast <4 x i32> %i22 to <2 x i64>
  %be22 = extractelement <2 x i64> %bc11, i64 0
  %as6 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i27, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac7 = bitcast <16 x i8> %as6 to <2 x i64>
  %ae6 = extractelement <2 x i64> %ac7, i64 0
  %i49 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %i50 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae6, i64 0, <4 x float> %i49, i32 0, i32 0, i32 0)
  %as4 = shufflevector <4 x i8> %i30, <4 x i8> zeroinitializer, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ac6 = bitcast <16 x i8> %as4 to <2 x i64>
  %ae7 = extractelement <2 x i64> %ac6, i64 1
  %i51 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 72340172838076673, i64 0, <4 x float> %i50, i32 0, i32 0, i32 0)
  %i52 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i51, i32 0, i32 0, i32 0)
  %i53 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae7, i64 0, <4 x float> %i52, i32 0, i32 0, i32 0)
  %i54 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i53, i32 0, i32 0, i32 0)
  %i55 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i54, i32 0, i32 0, i32 0)
  %i56 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be21, <4 x float> %i55, i32 0, i32 0, i32 0)
  %bc4 = bitcast <4 x i32> %i21 to <2 x i64>
  %be29 = extractelement <2 x i64> %bc4, i64 1
  %be14 = extractelement <2 x i64> %bc4, i64 0
  %i57 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be21, <4 x float> %i56, i32 0, i32 0, i32 0)
  %i58 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be14, <4 x float> %i57, i32 0, i32 0, i32 0)
  %i59 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be29, <4 x float> %i58, i32 0, i32 0, i32 0)
  %i60 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i59, i32 0, i32 0, i32 0)
  %i61 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i60, i32 0, i32 0, i32 0)
  %as8 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %arg9, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac9 = bitcast <16 x i8> %as8 to <2 x i64>
  %as13 = shufflevector <4 x i8> %arg8, <4 x i8> zeroinitializer, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ac18 = bitcast <16 x i8> %as13 to <2 x i64>
  %ae18 = extractelement <2 x i64> %ac18, i64 1
  %i62 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i19, i32 0, i32 0, i32 0)
  %ae9 = extractelement <2 x i64> %ac9, i64 0
  %i63 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae18, i64 0, <4 x float> %i62, i32 0, i32 0, i32 0)
  %i64 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae9, i64 0, <4 x float> %i63, i32 0, i32 0, i32 0)
  %i65 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i64, i32 0, i32 0, i32 0)
  %as5 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i29, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac8 = bitcast <16 x i8> %as5 to <2 x i64>
  %ae8 = extractelement <2 x i64> %ac8, i64 0
  %i66 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i65, i32 0, i32 0, i32 0)
  %i67 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i66, i32 0, i32 0, i32 0)
  %i68 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae8, i64 0, <4 x float> %i67, i32 0, i32 0, i32 0)
  %i69 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i68, i32 0, i32 0, i32 0)
  %i70 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i69, i32 0, i32 0, i32 0)
  %i71 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i70, i32 0, i32 0, i32 0)
  %i72 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i71, i32 0, i32 0, i32 0)
  %i73 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 0, <4 x float> %i72, i32 0, i32 0, i32 0)
  %bc13 = bitcast <4 x i32> %i23 to <2 x i64>
  %i74 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be22, <4 x float> %i73, i32 0, i32 0, i32 0)
  %be20 = extractelement <2 x i64> %bc13, i64 0
  %be26 = extractelement <2 x i64> %bc11, i64 1
  %i75 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be26, <4 x float> %i74, i32 0, i32 0, i32 0)
  %i76 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be20, <4 x float> %i75, i32 0, i32 0, i32 0)
  %i77 = load <4 x i8>, ptr addrspace(3) zeroinitializer, align 8
  %i78 = getelementptr i8, ptr addrspace(3) @shared, i32 %i28
  %p10 = getelementptr %f8, ptr addrspace(3) %i78, i32 %arg1
  %i79 = load <4 x i8>, ptr addrspace(3) %p10, align 4
  %i80 = getelementptr i8, ptr addrspace(3) @shared, i32 %n4
  %p11 = getelementptr %f8, ptr addrspace(3) %i80, i32 %arg1
  %i81 = getelementptr i8, ptr addrspace(3) %p11, i32 4
  %i82 = load <4 x i8>, ptr addrspace(3) %i81, align 4
  %i83 = load <4 x i8>, ptr addrspace(3) %p11, align 8
  %p4 = getelementptr %f8, ptr addrspace(3) %arg, i32 %m3
  %i84 = load <4 x i8>, ptr addrspace(3) %p4, align 4
  %i85 = getelementptr i8, ptr addrspace(3) %lds, i32 %n5
  %p8 = getelementptr %f8, ptr addrspace(3) %i85, i32 %arg1
  %i86 = load <4 x i8>, ptr addrspace(3) %p8, align 4
  %i87 = load <4 x i8>, ptr addrspace(3) %i85, align 8
  %i88 = load <4 x i8>, ptr addrspace(3) %i11, align 4
  %p5 = getelementptr %f8, ptr addrspace(3) %lds, i32 %m3
  %i89 = load <4 x i8>, ptr addrspace(3) %p5, align 4
  %p13 = getelementptr i8, ptr addrspace(3) inttoptr (i32 8192 to ptr addrspace(3)), i32 %i13
  %i90 = load <4 x i8>, ptr addrspace(3) %p13, align 4
  %p2 = getelementptr i8, ptr addrspace(3) %arg4, i32 %n3
  %i91 = load <4 x i8>, ptr addrspace(3) %p2, align 16
  %p9 = getelementptr i8, ptr addrspace(3) inttoptr (i32 8192 to ptr addrspace(3)), i32 %x3
  %i92 = load <4 x i8>, ptr addrspace(3) %p9, align 16
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %bc = bitcast <4 x i32> %i32 to <2 x i64>
  %be = extractelement <2 x i64> %bc, i64 1
  %as14 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i79, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %bc18 = bitcast <4 x i32> %i35 to <2 x i64>
  %be30 = extractelement <2 x i64> %bc18, i64 0
  %ac17 = bitcast <16 x i8> %as14 to <2 x i64>
  %ae17 = extractelement <2 x i64> %ac17, i64 1
  %as16 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i82, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %as10 = shufflevector <4 x i8> %i83, <4 x i8> %i48, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ac15 = bitcast <16 x i8> %as16 to <2 x i64>
  %ae13 = extractelement <2 x i64> %ac15, i64 0
  %bc15 = bitcast <4 x i32> %i36 to <2 x i64>
  %be8 = extractelement <2 x i64> %bc15, i64 0
  %ac10 = bitcast <16 x i8> %as10 to <2 x i64>
  %ae10 = extractelement <2 x i64> %ac10, i64 1
  %as9 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i84, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %bc7 = bitcast <4 x i32> %i38 to <2 x i64>
  %be7 = extractelement <2 x i64> %bc7, i64 0
  %ac16 = bitcast <16 x i8> %as9 to <2 x i64>
  %ae11 = extractelement <2 x i64> %ac16, i64 1
  %as11 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i86, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %as7 = shufflevector <4 x i8> %i87, <4 x i8> %i77, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ac11 = bitcast <16 x i8> %as11 to <2 x i64>
  %ae15 = extractelement <2 x i64> %ac11, i64 0
  %bc6 = bitcast <4 x i32> %i39 to <2 x i64>
  %be17 = extractelement <2 x i64> %bc6, i64 0
  %ac14 = bitcast <16 x i8> %as7 to <2 x i64>
  %ae12 = extractelement <2 x i64> %ac14, i64 1
  %as12 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i88, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac13 = bitcast <16 x i8> %as12 to <2 x i64>
  %ae16 = extractelement <2 x i64> %ac13, i64 0
  %bc14 = bitcast <4 x i32> %i42 to <2 x i64>
  %be28 = extractelement <2 x i64> %bc14, i64 0
  %as15 = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i89, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac12 = bitcast <16 x i8> %as15 to <2 x i64>
  %ae14 = extractelement <2 x i64> %ac12, i64 0
  %bc12 = bitcast <4 x i32> %i43 to <2 x i64>
  %be19 = extractelement <2 x i64> %bc12, i64 0
  %bc2 = bitcast <4 x i32> %i46 to <2 x i64>
  %be15 = extractelement <2 x i64> %bc2, i64 1
  %i93 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be15, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %bc17 = bitcast <4 x i32> %i33 to <2 x i64>
  %be32 = extractelement <2 x i64> %bc17, i64 0
  %bc19 = bitcast <4 x i32> %i34 to <2 x i64>
  %be33 = extractelement <2 x i64> %bc19, i64 0
  %i94 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae17, i64 %be33, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %i95 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae13, i64 0, <4 x float> %i94, i32 0, i32 0, i32 0)
  %i96 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae10, i64 0, <4 x float> %i95, i32 0, i32 0, i32 0)
  %bc5 = bitcast <4 x i32> %i37 to <2 x i64>
  %be10 = extractelement <2 x i64> %bc5, i64 0
  %i97 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be10, <4 x float> %i96, i32 0, i32 0, i32 0)
  %be12 = extractelement <2 x i64> %bc5, i64 1
  %i98 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae11, i64 %be12, <4 x float> %i97, i32 0, i32 0, i32 0)
  %bc10 = bitcast <4 x i32> %i40 to <2 x i64>
  %be16 = extractelement <2 x i64> %bc10, i64 0
  %i99 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae15, i64 %be16, <4 x float> %i98, i32 0, i32 0, i32 0)
  %be6 = extractelement <2 x i64> %bc10, i64 1
  %i100 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae12, i64 %be6, <4 x float> %i99, i32 0, i32 0, i32 0)
  %bc8 = bitcast <4 x i32> %i41 to <2 x i64>
  %be3 = extractelement <2 x i64> %bc8, i64 0
  %i101 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae16, i64 %be3, <4 x float> %i100, i32 0, i32 0, i32 0)
  %be11 = extractelement <2 x i64> %bc8, i64 1
  %i102 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be11, <4 x float> %i101, i32 0, i32 0, i32 0)
  %bc9 = bitcast <4 x i32> %i44 to <2 x i64>
  %i103 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae14, i64 0, <4 x float> %i102, i32 0, i32 0, i32 0)
  %be18 = extractelement <2 x i64> %bc9, i64 1
  %i104 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be18, <4 x float> %i103, i32 0, i32 0, i32 0)
  %bc3 = bitcast <4 x i32> %i45 to <2 x i64>
  %be4 = extractelement <2 x i64> %bc3, i64 0
  %i105 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be4, <4 x float> %i104, i32 0, i32 0, i32 0)
  %be25 = extractelement <2 x i64> %bc3, i64 1
  %i106 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be25, <4 x float> %i105, i32 0, i32 0, i32 0)
  %be2 = extractelement <2 x i64> %bc, i64 0
  %i107 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 72340172838076673, i64 %be2, <4 x float> %i61, i32 0, i32 0, i32 0)
  %i108 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be, <4 x float> %i107, i32 0, i32 0, i32 0)
  %i109 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be30, <4 x float> %i108, i32 0, i32 0, i32 0)
  %be31 = extractelement <2 x i64> %bc18, i64 1
  %i110 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be31, <4 x float> %i109, i32 0, i32 0, i32 0)
  %i111 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be8, <4 x float> %i110, i32 0, i32 0, i32 0)
  %be23 = extractelement <2 x i64> %bc15, i64 1
  %i112 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be23, <4 x float> %i111, i32 0, i32 0, i32 0)
  %i113 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be7, <4 x float> %i112, i32 0, i32 0, i32 0)
  %be5 = extractelement <2 x i64> %bc7, i64 1
  %i114 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be5, <4 x float> %i113, i32 0, i32 0, i32 0)
  %as = shufflevector <4 x i8> zeroinitializer, <4 x i8> %i90, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %i115 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be17, <4 x float> %i114, i32 0, i32 0, i32 0)
  %ac4 = bitcast <16 x i8> %as to <2 x i64>
  %ae4 = extractelement <2 x i64> %ac4, i64 1
  %be13 = extractelement <2 x i64> %bc6, i64 1
  %i116 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae4, i64 %be13, <4 x float> %i115, i32 0, i32 0, i32 0)
  %as3 = shufflevector <4 x i8> %i91, <4 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac5 = bitcast <16 x i8> %as3 to <2 x i64>
  %ae3 = extractelement <2 x i64> %ac5, i64 0
  %i117 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be28, <4 x float> %i116, i32 0, i32 0, i32 0)
  %be9 = extractelement <2 x i64> %bc14, i64 1
  %i118 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be9, <4 x float> %i117, i32 0, i32 0, i32 0)
  %as2 = shufflevector <4 x i8> %i92, <4 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %ac3 = bitcast <16 x i8> %as2 to <2 x i64>
  %ae5 = extractelement <2 x i64> %ac3, i64 0
  %be27 = extractelement <2 x i64> %bc2, i64 0
  %i119 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae5, i64 %be19, <4 x float> %i118, i32 0, i32 0, i32 0)
  %i120 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be27, <4 x float> %i119, i32 0, i32 0, i32 0)
  %i121 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be32, <4 x float> %i76, i32 0, i32 0, i32 0)
  %be24 = extractelement <2 x i64> %bc9, i64 0
  %i122 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %ae3, i64 0, <4 x float> %i121, i32 0, i32 0, i32 0)
  %i123 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 0, i64 %be24, <4 x float> %i122, i32 0, i32 0, i32 0)
  %n7 = and i32 %i10, 31
  %m4 = and i32 %i31, 56
  %m = mul i32 %n7, %arg0
  %a6 = or i32 %m, %m4
  %ce4 = extractelement <4 x float> %i93, i64 0
  %m2 = and i32 %i, 48
  %i124 = getelementptr float, ptr addrspace(3) @shared, i32 %m2
  %p21 = getelementptr float, ptr addrspace(3) %i124, i32 %n6
  store float %ce4, ptr addrspace(3) zeroinitializer, align 4
  %ce3 = extractelement <4 x float> %i120, i64 0
  store float %ce3, ptr addrspace(3) %p21, align 4
  %ce2 = extractelement <4 x float> %i106, i64 0
  store float %ce2, ptr addrspace(3) zeroinitializer, align 4
  %ce = extractelement <4 x float> %i123, i64 0
  store float %ce, ptr addrspace(3) %p15, align 4
  %sx = sext i32 %a6 to i64
  %p20 = getelementptr i16, ptr addrspace(1) null, i64 %sx
  %i125 = atomicrmw fadd ptr addrspace(1) %p20, <2 x bfloat> zeroinitializer syncscope("agent") monotonic, align 4
  ret void
}

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #5

attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="2" "target-cpu"="gfx942" }
attributes #1 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) "target-cpu"="gfx942" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) "target-cpu"="gfx942" }
attributes #3 = { nocallback nofree nounwind willreturn "target-cpu"="gfx942" }
attributes #4 = { convergent nocallback nofree nounwind willreturn "target-cpu"="gfx942" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="gfx942" }
