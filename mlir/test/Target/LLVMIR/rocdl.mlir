// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

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

  // CHECK: call i32 @llvm.amdgcn.workitem.id.x(),{{.*}} !range ![[$RANGE:[0-9]+]]
  %13 = rocdl.workitem.id.x {range = array<i32: 0, 64>} : i32

  llvm.return %1 : i32
}

llvm.func @kernel_func() attributes {rocdl.kernel} {
  // CHECK-LABEL: amdgpu_kernel void @kernel_func()
  // CHECK: #[[$KERNEL_ATTRS:[0-9]+]]
  llvm.return
}

llvm.func @kernel_func_workgroups()
    attributes {rocdl.kernel, rocdl.max_flat_work_group_size = 1024 : index} {
  // CHECK-LABEL: amdgpu_kernel void @kernel_func_workgroups()
  // CHECK: #[[$KERNEL_WORKGROUP_ATTRS:[0-9]+]]
  llvm.return
}

llvm.func @known_block_sizes()
    attributes {rocdl.kernel,
      rocdl.flat_work_group_size = "128,128",
      rocdl.reqd_work_group_size = array<i32: 16, 4, 2>} {
  // CHECK-LABEL: amdgpu_kernel void @known_block_sizes()
  // CHECK: #[[$KNOWN_BLOCK_SIZE_ATTRS:[0-9]+]]
  // CHECK: !reqd_work_group_size ![[$REQD_WORK_GROUP_SIZE:[0-9]+]]
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
                   %arg2 : vector<32 x f32>, %arg3: i32,
                   %arg4 : vector<16 x f32>, %arg5 : vector<4xf32>,
                   %arg6 : vector<4xf16>, %arg7 : vector<32 x i32>,
                   %arg8 : vector<16 x i32>, %arg9 : vector<4xi32>,
                   %arg10 : vector<2xi16>, %arg11 : i64) -> vector<32 x f32> {
  %csti32 = llvm.mlir.constant(42 : i32) : i32

  // CHECK-LABEL: rocdl.xdlops
  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %{{.*}}, float %{{.*}}, <32 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r0 = rocdl.mfma.f32.32x32x1f32 %arg0, %arg1, %arg2, %csti32, %csti32, %csti32 :
                            (f32, f32, vector<32 x f32>,
                            i32, i32, i32) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float %{{.*}}, float %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r1 = rocdl.mfma.f32.16x16x1f32 %arg0, %arg1, %arg4, %csti32, %csti32, %csti32 :
                            (f32, f32, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r2 = rocdl.mfma.f32.16x16x4f32 %arg0, %arg1, %arg5, %csti32, %csti32, %csti32 :
                            (f32, f32, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r3 = rocdl.mfma.f32.4x4x1f32 %arg0, %arg1, %arg5, %csti32, %csti32, %csti32 :
                            (f32, f32, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float %{{.*}}, float %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r4= rocdl.mfma.f32.32x32x2f32 %arg0, %arg1, %arg4, %csti32, %csti32, %csti32 :
                            (f32, f32, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <32 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r5 = rocdl.mfma.f32.32x32x4f16 %arg6, %arg6, %arg2, %csti32, %csti32, %csti32 :
                            (vector<4xf16>, vector<4xf16>, vector<32 x f32>,
                            i32, i32, i32) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r6 = rocdl.mfma.f32.16x16x4f16 %arg6, %arg6, %arg4, %csti32, %csti32, %csti32 :
                            (vector<4xf16>, vector<4xf16>, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r7 = rocdl.mfma.f32.4x4x4f16 %arg6, %arg6, %arg5, %csti32, %csti32, %csti32 :
                            (vector<4xf16>, vector<4xf16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r8 = rocdl.mfma.f32.32x32x8f16 %arg6, %arg6, %arg4, %csti32, %csti32, %csti32 :
                            (vector<4xf16>, vector<4xf16>, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r9 = rocdl.mfma.f32.16x16x16f16 %arg6, %arg6, %arg5, %csti32, %csti32, %csti32 :
                            (vector<4xf16>, vector<4xf16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 %{{.*}}, i32 %{{.*}}, <32 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r10 = rocdl.mfma.i32.32x32x4i8 %arg3, %arg3, %arg7, %csti32, %csti32, %csti32 :
                            (i32, i32, vector<32 x i32>,
                            i32, i32, i32) -> vector<32 x i32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32 %{{.*}}, i32 %{{.*}}, <16 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r11 = rocdl.mfma.i32.16x16x4i8 %arg3, %arg3, %arg8, %csti32, %csti32, %csti32 :
                            (i32, i32, vector<16 x i32>,
                            i32, i32, i32) -> vector<16 x i32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r12 = rocdl.mfma.i32.4x4x4i8 %arg3, %arg3, %arg9, %csti32, %csti32, %csti32 :
                            (i32, i32, vector<4xi32>,
                            i32, i32, i32) -> vector<4xi32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32 %{{.*}}, i32 %{{.*}}, <16 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r13 = rocdl.mfma.i32.32x32x8i8 %arg3, %arg3, %arg8, %csti32, %csti32, %csti32 :
                            (i32, i32, vector<16 x i32>,
                            i32, i32, i32) -> vector<16 x i32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r14 = rocdl.mfma.i32.16x16x16i8 %arg3, %arg3, %arg9, %csti32, %csti32, %csti32 :
                            (i32, i32, vector<4xi32>,
                            i32, i32, i32) -> vector<4xi32>

  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <32 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r15 = rocdl.mfma.f32.32x32x2bf16 %arg10, %arg10, %arg2, %csti32, %csti32, %csti32 :
                            (vector<2xi16>, vector<2xi16>, vector<32 x f32>,
                            i32, i32, i32) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r16 = rocdl.mfma.f32.16x16x2bf16 %arg10, %arg10, %arg4, %csti32, %csti32, %csti32 :
                            (vector<2xi16>, vector<2xi16>, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r17 = rocdl.mfma.f32.4x4x2bf16 %arg10, %arg10, %arg5, %csti32, %csti32, %csti32 :
                            (vector<2xi16>, vector<2xi16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r18 = rocdl.mfma.f32.32x32x4bf16 %arg10, %arg10, %arg4, %csti32, %csti32, %csti32 :
                            (vector<2xi16>, vector<2xi16>, vector<16 x f32>,
                            i32, i32, i32) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r19 = rocdl.mfma.f32.16x16x8bf16 %arg10, %arg10, %arg5, %csti32, %csti32, %csti32 :
                            (vector<2xi16>, vector<2xi16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8(i64 %{{.*}}, i64 %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r20 = rocdl.mfma.f32.16x16x32.bf8.bf8 %arg11, %arg11, %arg5, %csti32, %csti32, %csti32 :
                            (i64, i64, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %{{.*}}, i64 %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r21 = rocdl.mfma.f32.16x16x32.bf8.fp8 %arg11, %arg11, %arg5, %csti32, %csti32, %csti32 :
                            (i64, i64, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8(i64 %{{.*}}, i64 %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r22 = rocdl.mfma.f32.16x16x32.fp8.bf8 %arg11, %arg11, %arg5, %csti32, %csti32, %csti32 :
                            (i64, i64, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{.*}}, i64 %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r23 = rocdl.mfma.f32.16x16x32.fp8.fp8 %arg11, %arg11, %arg5, %csti32, %csti32, %csti32 :
                            (i64, i64, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64 %{{.*}}, i64 %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r24 = rocdl.mfma.f32.32x32x16.bf8.bf8 %arg11, %arg11, %arg4, %csti32, %csti32, %csti32 :
                            (i64, i64, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8(i64 %{{.*}}, i64 %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r25 = rocdl.mfma.f32.32x32x16.bf8.fp8 %arg11, %arg11, %arg4, %csti32, %csti32, %csti32 :
                            (i64, i64, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8(i64 %{{.*}}, i64 %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r26 = rocdl.mfma.f32.32x32x16.fp8.bf8 %arg11, %arg11, %arg4, %csti32, %csti32, %csti32 :
                            (i64, i64, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64 %{{.*}}, i64 %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r27 = rocdl.mfma.f32.32x32x16.bf8.bf8 %arg11, %arg11, %arg4, %csti32, %csti32, %csti32 :
                            (i64, i64, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>
  llvm.return %r0 : vector<32 x f32>
}

llvm.func @rocdl.mubuf(%rsrc : vector<4xi32>, %vindex : i32,
                       %offset : i32, %vdata1 : vector<1xf32>,
                       %vdata2 : vector<2xf32>, %vdata4 : vector<4xf32>) {
  %glc = llvm.mlir.constant(false) : i1
  %slc = llvm.mlir.constant(true) : i1
  // CHECK-LABEL: rocdl.mubuf
  // CHECK: call <1 x float> @llvm.amdgcn.buffer.load.v1f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 {{.*}}, i1 {{.*}})
  // CHECK: call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 {{.*}}, i1 {{.*}})
  // CHECK: call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 {{.*}}, i1 {{.*}})

  // CHECK: call void @llvm.amdgcn.buffer.store.v1f32(<1 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 {{.*}}, i1 {{.*}})
  // CHECK: call void @llvm.amdgcn.buffer.store.v2f32(<2 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 {{.*}}, i1 {{.*}})
  // CHECK: call void @llvm.amdgcn.buffer.store.v4f32(<4 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i1 {{.*}}, i1 {{.*}})

  %r1 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<1xf32>
  %r2 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<2xf32>
  %r4 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<4xf32>

  rocdl.buffer.store %vdata1, %rsrc, %vindex, %offset, %glc, %slc : vector<1xf32>
  rocdl.buffer.store %vdata2, %rsrc, %vindex, %offset, %glc, %slc : vector<2xf32>
  rocdl.buffer.store %vdata4, %rsrc, %vindex, %offset, %glc, %slc : vector<4xf32>

  llvm.return
}

llvm.func @rocdl.raw.buffer(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : i32,
                        %vdata2 : vector<2xi32>,
                        %vdata4 : vector<4xi32>) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.buffer
  // CHECK: call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call <2 x i32> @llvm.amdgcn.raw.buffer.load.v2i32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  // CHECK: call void @llvm.amdgcn.raw.buffer.store.i32(i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  %r1 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : i32
  %r2 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : vector<2xi32>
  %r4 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : vector<4xi32>

  rocdl.raw.buffer.store %vdata1, %rsrc, %offset, %soffset, %aux : i32
  rocdl.raw.buffer.store %vdata2, %rsrc, %offset, %soffset, %aux : vector<2xi32>
  rocdl.raw.buffer.store %vdata4, %rsrc, %offset, %soffset, %aux : vector<4xi32>

  llvm.return
}

llvm.func @rocdl.raw.buffer.atomic.f32(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : f32) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.buffer.atomic.f32
  // CHECK: call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call float @llvm.amdgcn.raw.buffer.atomic.fmax.f32(float %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  rocdl.raw.buffer.atomic.fadd %vdata1, %rsrc, %offset, %soffset, %aux : f32
  rocdl.raw.buffer.atomic.fmax %vdata1, %rsrc, %offset, %soffset, %aux : f32

  llvm.return
}

llvm.func @rocdl.raw.buffer.atomic.i32(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : i32) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.buffer.atomic.i32
  // CHECK: call i32 @llvm.amdgcn.raw.buffer.atomic.smax.i32(i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call i32 @llvm.amdgcn.raw.buffer.atomic.umin.i32(i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  rocdl.raw.buffer.atomic.smax %vdata1, %rsrc, %offset, %soffset, %aux : i32
  rocdl.raw.buffer.atomic.umin %vdata1, %rsrc, %offset, %soffset, %aux : i32

  llvm.return
}

llvm.func @rocdl.raw.buffer.atomic.cmpswap(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %src : i32, %cmp : i32) -> i32 {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.buffer.atomic.cmpswap
  // CHECK: [[val:%.+]] = call i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i32(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: ret i32 [[val]]

  %val = rocdl.raw.buffer.atomic.cmpswap(%src, %cmp, %rsrc, %offset, %soffset, %aux) : i32, vector<4xi32>
  llvm.return %val : i32
}

// CHECK-DAG: attributes #[[$KERNEL_ATTRS]] = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-implicitarg-num-bytes"="56" }
// CHECK-DAG: attributes #[[$KERNEL_WORKGROUP_ATTRS]] = { "amdgpu-flat-work-group-size"="1,1024"
// CHECK-DAG: attributes #[[$KNOWN_BLOCK_SIZE_ATTRS]] = { "amdgpu-flat-work-group-size"="128,128"
// CHECK-DAG: ![[$RANGE]] = !{i32 0, i32 64}
// CHECK-DAG: ![[$REQD_WORK_GROUP_SIZE]] = !{i32 16, i32 4, i32 2}
