// RUN: mlir-translate -mlir-to-rocdlir %s | FileCheck %s

llvm.func @rocdl_special_regs() -> i32 {
  // CHECK-LABEL: rocdl_special_regs
  // CHECK: call i32 @llvm.amdgcn.workitem.id.x()
  %1 = rocdl.workitem.id.x : i32
  // CHECK: call i32 @llvm.amdgcn.workitem.id.y()
  %2 = rocdl.workitem.id.y : i32
  // CHECK: call i32 @llvm.amdgcn.workitem.id.z()
  %3 = rocdl.workitem.id.z : i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.x()
  %4 = rocdl.workgroup.id.x : i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.y()
  %5 = rocdl.workgroup.id.y : i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.z()
  %6 = rocdl.workgroup.id.z : i32
  // CHECK: call i64 @__ockl_get_local_size(i32 0)
  %7 = rocdl.workgroup.dim.x : i64
  // CHECK: call i64 @__ockl_get_local_size(i32 1)
  %8 = rocdl.workgroup.dim.y : i64
  // CHECK: call i64 @__ockl_get_local_size(i32 2)
  %9 = rocdl.workgroup.dim.z : i64
  // CHECK: call i64 @__ockl_get_global_size(i32 0)
  %10 = rocdl.grid.dim.x : i64
  // CHECK: call i64 @__ockl_get_global_size(i32 1)
  %11 = rocdl.grid.dim.y : i64
  // CHECK: call i64 @__ockl_get_global_size(i32 2)
  %12 = rocdl.grid.dim.z : i64
  llvm.return %1 : i32
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

llvm.func @rocdl.xdlops(%arg0 : f32, %arg1 : f32,
                   %arg2 : vector<32 x f32>, %arg3 : i32,
                   %arg4 : vector<16 x f32>, %arg5 : vector<4xf32>,
                   %arg6 : vector<4xf16>, %arg7 : vector<32 x i32>,
                   %arg8 : vector<16 x i32>, %arg9 : vector<4xi32>,
                   %arg10 : vector<2xi16>) -> vector<32 x f32> {
  // CHECK-LABEL: rocdl.xdlops
  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %{{.*}}, float %{{.*}}, <32 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r0 = rocdl.mfma.f32.32x32x1f32 %arg0, %arg1, %arg2, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<32 x f32>,
                            i32, i32, i32) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float %{{.*}}, float %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r1 = rocdl.mfma.f32.16x16x1f32 %arg0, %arg1, %arg4, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r2 = rocdl.mfma.f32.16x16x4f32 %arg0, %arg1, %arg5, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r3 = rocdl.mfma.f32.4x4x1f32 %arg0, %arg1, %arg5, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float %{{.*}}, float %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r4= rocdl.mfma.f32.32x32x2f32 %arg0, %arg1, %arg4, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <32 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r5 = rocdl.mfma.f32.32x32x4f16 %arg6, %arg6, %arg2, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<32 x f32>,
                            i32, i32, i32) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r6 = rocdl.mfma.f32.16x16x4f16 %arg6, %arg6, %arg4, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r7 = rocdl.mfma.f32.4x4x4f16 %arg6, %arg6, %arg5, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r8 = rocdl.mfma.f32.32x32x8f16 %arg6, %arg6, %arg4, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r9 = rocdl.mfma.f32.16x16x16f16 %arg6, %arg6, %arg5, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 %{{.*}}, i32 %{{.*}}, <32 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r10 = rocdl.mfma.i32.32x32x4i8 %arg3, %arg3, %arg7, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<32 x i32>,
                            i32, i32, i32) -> vector<32 x i32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32 %{{.*}}, i32 %{{.*}}, <16 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r11 = rocdl.mfma.i32.16x16x4i8 %arg3, %arg3, %arg8, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<16 x i32>,
                            i32, i32, i32) -> vector<16 x i32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r12 = rocdl.mfma.i32.4x4x4i8 %arg3, %arg3, %arg9, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<4xi32>,
                            i32, i32, i32) -> vector<4xi32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32 %{{.*}}, i32 %{{.*}}, <16 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r13 = rocdl.mfma.i32.32x32x8i8 %arg3, %arg3, %arg8, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<16 x i32>,
                            i32, i32, i32) -> vector<16 x i32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r14 = rocdl.mfma.i32.16x16x16i8 %arg3, %arg3, %arg9, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<4xi32>,
                            i32, i32, i32) -> vector<4xi32>

  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <32 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r15 = rocdl.mfma.f32.32x32x2bf16 %arg10, %arg10, %arg2, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<32 x f32>,
                            i32, i32, i32) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r16 = rocdl.mfma.f32.16x16x2bf16 %arg10, %arg10, %arg4, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r17 = rocdl.mfma.f32.4x4x2bf16 %arg10, %arg10, %arg5, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r18 = rocdl.mfma.f32.32x32x4bf16 %arg10, %arg10, %arg4, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r19 = rocdl.mfma.f32.16x16x8bf16 %arg10, %arg10, %arg5, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  llvm.return %r0 : vector<32 x f32>
}

llvm.func @rocdl.mubuf(%rsrc : vector<4xi32>, %vindex : i32,
                       %offset : i32, %glc : i1,
                       %slc : i1, %vdata1 : vector<1xf32>,
                       %vdata2 : vector<2xf32>, %vdata4 : vector<4xf32>) {
  // CHECK-LABEL: rocdl.mubuf
  // CHECK: call <1 x float> @llvm.amdgcn.buffer.load.v1f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  %r1 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<1xf32>
  // CHECK: call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  %r2 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<2xf32>
  // CHECK: call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  %r4 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<4xf32>

  // CHECK: call void @llvm.amdgcn.buffer.store.v1f32(<1 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  rocdl.buffer.store %vdata1, %rsrc, %vindex, %offset, %glc, %slc : vector<1xf32>
  // CHECK: call void @llvm.amdgcn.buffer.store.v2f32(<2 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  rocdl.buffer.store %vdata2, %rsrc, %vindex, %offset, %glc, %slc : vector<2xf32>
  // CHECK: call void @llvm.amdgcn.buffer.store.v4f32(<4 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 %{{.*}}, i1 %{{.*}})
  rocdl.buffer.store %vdata4, %rsrc, %vindex, %offset, %glc, %slc : vector<4xf32>

  llvm.return
}

