// RUN: mlir-translate -mlir-to-rocdlir %s | FileCheck %s

llvm.func @rocdl_special_regs() -> !llvm.i32 {
  // CHECK-LABEL: rocdl_special_regs
  // CHECK: call i32 @llvm.amdgcn.workitem.id.x()
  %1 = rocdl.workitem.id.x : !llvm.i32
  // CHECK: call i32 @llvm.amdgcn.workitem.id.y()
  %2 = rocdl.workitem.id.y : !llvm.i32
  // CHECK: call i32 @llvm.amdgcn.workitem.id.z()
  %3 = rocdl.workitem.id.z : !llvm.i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.x()
  %4 = rocdl.workgroup.id.x : !llvm.i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.y()
  %5 = rocdl.workgroup.id.y : !llvm.i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.z()
  %6 = rocdl.workgroup.id.z : !llvm.i32
  // CHECK: call i64 @__ockl_get_local_size(i32 0)
  %7 = rocdl.workgroup.dim.x : !llvm.i64
  // CHECK: call i64 @__ockl_get_local_size(i32 1)
  %8 = rocdl.workgroup.dim.y : !llvm.i64
  // CHECK: call i64 @__ockl_get_local_size(i32 2)
  %9 = rocdl.workgroup.dim.z : !llvm.i64
  // CHECK: call i64 @__ockl_get_global_size(i32 0)
  %10 = rocdl.grid.dim.x : !llvm.i64
  // CHECK: call i64 @__ockl_get_global_size(i32 1)
  %11 = rocdl.grid.dim.y : !llvm.i64
  // CHECK: call i64 @__ockl_get_global_size(i32 2)
  %12 = rocdl.grid.dim.z : !llvm.i64
  llvm.return %1 : !llvm.i32
}

llvm.func @kernel_func() attributes {gpu.kernel} {
  // CHECK-LABEL: amdgpu_kernel void @kernel_func
  llvm.return
}

llvm.func @rocdl.barrier() {
  // CHECK:      fence syncscope("workgroup") release
  // CHECK-NEXT: call void @llvm.amdgcn.s.barrier()
  // CHECK-NEXT: fence syncscope("workgroup") acquire
  rocdl.barrier
  llvm.return
}

llvm.func @rocdl.xdlops(%arg0 : !llvm.float, %arg1 : !llvm.float,
                   %arg2 : !llvm.vec<32 x float>, %arg3 : !llvm.i32,
                   %arg4 : !llvm.vec<16 x float>, %arg5 : !llvm.vec<4 x float>,
                   %arg6 : !llvm.vec<4 x half>, %arg7 : !llvm.vec<32 x i32>,
                   %arg8 : !llvm.vec<16 x i32>, %arg9 : !llvm.vec<4 x i32>,
                   %arg10 : !llvm.vec<2 x i16>) -> !llvm.vec<32 x float> {
  // CHECK-LABEL: rocdl.xdlops
  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %{{.*}}, float %{{.*}}, <32 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r0 = rocdl.mfma.f32.32x32x1f32 %arg0, %arg1, %arg2, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<32 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float %{{.*}}, float %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r1 = rocdl.mfma.f32.16x16x1f32 %arg0, %arg1, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r2 = rocdl.mfma.f32.16x16x4f32 %arg0, %arg1, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r3 = rocdl.mfma.f32.4x4x1f32 %arg0, %arg1, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float %{{.*}}, float %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r4= rocdl.mfma.f32.32x32x2f32 %arg0, %arg1, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <32 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r5 = rocdl.mfma.f32.32x32x4f16 %arg6, %arg6, %arg2, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<32 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r6 = rocdl.mfma.f32.16x16x4f16 %arg6, %arg6, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r7 = rocdl.mfma.f32.4x4x4f16 %arg6, %arg6, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r8 = rocdl.mfma.f32.32x32x8f16 %arg6, %arg6, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r9 = rocdl.mfma.f32.16x16x16f16 %arg6, %arg6, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 %{{.*}}, i32 %{{.*}}, <32 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r10 = rocdl.mfma.i32.32x32x4i8 %arg3, %arg3, %arg7, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<32 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x i32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32 %{{.*}}, i32 %{{.*}}, <16 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r11 = rocdl.mfma.i32.16x16x4i8 %arg3, %arg3, %arg8, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<16 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x i32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r12 = rocdl.mfma.i32.4x4x4i8 %arg3, %arg3, %arg9, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<4 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x i32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32 %{{.*}}, i32 %{{.*}}, <16 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r13 = rocdl.mfma.i32.32x32x8i8 %arg3, %arg3, %arg8, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<16 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x i32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r14 = rocdl.mfma.i32.16x16x16i8 %arg3, %arg3, %arg9, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<4 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x i32>

  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <32 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r15 = rocdl.mfma.f32.32x32x2bf16 %arg10, %arg10, %arg2, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<32 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r16 = rocdl.mfma.f32.16x16x2bf16 %arg10, %arg10, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r17 = rocdl.mfma.f32.4x4x2bf16 %arg10, %arg10, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r18 = rocdl.mfma.f32.32x32x4bf16 %arg10, %arg10, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r19 = rocdl.mfma.f32.16x16x8bf16 %arg10, %arg10, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  llvm.return %r0 : !llvm.vec<32 x float>
}

llvm.func @rocdl.mubuf(%rsrc : !llvm.vec<4 x i32>, %vindex : !llvm.i32,
                       %offset : !llvm.i32, %glc : !llvm.i1,
                       %slc : !llvm.i1, %vdata1 : !llvm.vec<1 x float>,
                       %vdata2 : !llvm.vec<2 x float>, %vdata4 : !llvm.vec<4 x float>) {
  // CHECK-LABEL: rocdl.mubuf
  // CHECK: call <1 x float> @llvm.amdgcn.buffer.load.v1f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  %r1 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<1 x float>
  // CHECK: call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  %r2 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<2 x float>
  // CHECK: call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  %r4 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<4 x float>

  // CHECK: call void @llvm.amdgcn.buffer.store.v1f32(<1 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  rocdl.buffer.store %vdata1, %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<1 x float>
  // CHECK: call void @llvm.amdgcn.buffer.store.v2f32(<2 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  rocdl.buffer.store %vdata2, %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<2 x float>
  // CHECK: call void @llvm.amdgcn.buffer.store.v4f32(<4 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  rocdl.buffer.store %vdata4, %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<4 x float>

  llvm.return
}

