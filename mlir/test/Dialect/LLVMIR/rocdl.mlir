// RUN: mlir-opt %s | FileCheck %s

func @rocdl_special_regs() -> !llvm.i32 {
  // CHECK-LABEL: rocdl_special_regs
  // CHECK: rocdl.workitem.id.x : !llvm.i32
  %0 = rocdl.workitem.id.x : !llvm.i32
  // CHECK: rocdl.workitem.id.y : !llvm.i32
  %1 = rocdl.workitem.id.y : !llvm.i32
  // CHECK: rocdl.workitem.id.z : !llvm.i32
  %2 = rocdl.workitem.id.z : !llvm.i32
  // CHECK: rocdl.workgroup.id.x : !llvm.i32
  %3 = rocdl.workgroup.id.x : !llvm.i32
  // CHECK: rocdl.workgroup.id.y : !llvm.i32
  %4 = rocdl.workgroup.id.y : !llvm.i32
  // CHECK: rocdl.workgroup.id.z : !llvm.i32
  %5 = rocdl.workgroup.id.z : !llvm.i32
  // CHECK: rocdl.workgroup.dim.x : !llvm.i32
  %6 = rocdl.workgroup.dim.x : !llvm.i32
  // CHECK: rocdl.workgroup.dim.y : !llvm.i32
  %7 = rocdl.workgroup.dim.y : !llvm.i32
  // CHECK: rocdl.workgroup.dim.z : !llvm.i32
  %8 = rocdl.workgroup.dim.z : !llvm.i32
  // CHECK: rocdl.grid.dim.x : !llvm.i32
  %9 = rocdl.grid.dim.x : !llvm.i32
  // CHECK: rocdl.grid.dim.y : !llvm.i32
  %10 = rocdl.grid.dim.y : !llvm.i32
  // CHECK: rocdl.grid.dim.z : !llvm.i32
  %11 = rocdl.grid.dim.z : !llvm.i32
  llvm.return %0 : !llvm.i32
}

func @rocdl.barrier() {
  // CHECK: rocdl.barrier
  rocdl.barrier
  llvm.return
}

func @rocdl.xdlops(%arg0 : !llvm.float, %arg1 : !llvm.float,
                   %arg2 : !llvm.vec<32 x float>, %arg3 : !llvm.i32,
                   %arg4 : !llvm.vec<16 x float>, %arg5 : !llvm.vec<4 x float>,
                   %arg6 : !llvm.vec<4 x half>, %arg7 : !llvm.vec<32 x i32>,
                   %arg8 : !llvm.vec<16 x i32>, %arg9 : !llvm.vec<4 x i32>,
                   %arg10 : !llvm.vec<2 x i16>) -> !llvm.vec<32 x float> {
  // CHECK-LABEL: rocdl.xdlops
  // CHECK: rocdl.mfma.f32.32x32x1f32 {{.*}} : (!llvm.float, !llvm.float, !llvm.vec<32 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>
  %r0 = rocdl.mfma.f32.32x32x1f32 %arg0, %arg1, %arg2, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<32 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>

  // CHECK: rocdl.mfma.f32.16x16x1f32 {{.*}} : (!llvm.float, !llvm.float, !llvm.vec<16 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>
  %r1 = rocdl.mfma.f32.16x16x1f32 %arg0, %arg1, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: rocdl.mfma.f32.16x16x4f32 {{.*}} : (!llvm.float, !llvm.float, !llvm.vec<4 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>
  %r2 = rocdl.mfma.f32.16x16x4f32 %arg0, %arg1, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: rocdl.mfma.f32.4x4x1f32 {{.*}} : (!llvm.float, !llvm.float, !llvm.vec<4 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>
  %r3 = rocdl.mfma.f32.4x4x1f32 %arg0, %arg1, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: rocdl.mfma.f32.32x32x2f32 {{.*}} : (!llvm.float, !llvm.float, !llvm.vec<16 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>
  %r4= rocdl.mfma.f32.32x32x2f32 %arg0, %arg1, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.float, !llvm.float, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: rocdl.mfma.f32.32x32x4f16 {{.*}} : (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<32 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>
  %r5 = rocdl.mfma.f32.32x32x4f16 %arg6, %arg6, %arg2, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<32 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>

  // CHECK: rocdl.mfma.f32.16x16x4f16 {{.*}} : (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<16 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>
  %r6 = rocdl.mfma.f32.16x16x4f16 %arg6, %arg6, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: rocdl.mfma.f32.4x4x4f16 {{.*}} : (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<4 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>
  %r7 = rocdl.mfma.f32.4x4x4f16 %arg6, %arg6, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: rocdl.mfma.f32.32x32x8f16 {{.*}} : (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<16 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>
  %r8 = rocdl.mfma.f32.32x32x8f16 %arg6, %arg6, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: rocdl.mfma.f32.16x16x16f16 {{.*}} : (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<4 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>
  %r9 = rocdl.mfma.f32.16x16x16f16 %arg6, %arg6, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.vec<4 x half>, !llvm.vec<4 x half>, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: rocdl.mfma.i32.32x32x4i8 {{.*}} : (!llvm.i32, !llvm.i32, !llvm.vec<32 x i32>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x i32>
  %r10 = rocdl.mfma.i32.32x32x4i8 %arg3, %arg3, %arg7, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<32 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x i32>

  // CHECK: rocdl.mfma.i32.16x16x4i8 {{.*}} : (!llvm.i32, !llvm.i32, !llvm.vec<16 x i32>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x i32>
  %r11 = rocdl.mfma.i32.16x16x4i8 %arg3, %arg3, %arg8, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<16 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x i32>

  // CHECK: rocdl.mfma.i32.4x4x4i8 {{.*}} : (!llvm.i32, !llvm.i32, !llvm.vec<4 x i32>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x i32>
  %r12 = rocdl.mfma.i32.4x4x4i8 %arg3, %arg3, %arg9, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<4 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x i32>

  // CHECK: rocdl.mfma.i32.32x32x8i8 {{.*}} : (!llvm.i32, !llvm.i32, !llvm.vec<16 x i32>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x i32>
  %r13 = rocdl.mfma.i32.32x32x8i8 %arg3, %arg3, %arg8, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<16 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x i32>

  // CHECK: rocdl.mfma.i32.16x16x16i8 {{.*}} : (!llvm.i32, !llvm.i32, !llvm.vec<4 x i32>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x i32>
  %r14 = rocdl.mfma.i32.16x16x16i8 %arg3, %arg3, %arg9, %arg3, %arg3, %arg3 :
                            (!llvm.i32, !llvm.i32, !llvm.vec<4 x i32>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x i32>

  // CHECK: rocdl.mfma.f32.32x32x2bf16 {{.*}} : (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<32 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>
  %r15 = rocdl.mfma.f32.32x32x2bf16 %arg10, %arg10, %arg2, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<32 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<32 x float>

  // CHECK: rocdl.mfma.f32.16x16x2bf16 {{.*}} : (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<16 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>
  %r16 = rocdl.mfma.f32.16x16x2bf16 %arg10, %arg10, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: rocdl.mfma.f32.4x4x2bf16 {{.*}} : (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<4 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>
  %r17 = rocdl.mfma.f32.4x4x2bf16 %arg10, %arg10, %arg5, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<4 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>

  // CHECK: rocdl.mfma.f32.32x32x4bf16 {{.*}} : (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<16 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>
  %r18 = rocdl.mfma.f32.32x32x4bf16 %arg10, %arg10, %arg4, %arg3, %arg3, %arg3 :
                            (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<16 x float>,
                            !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<16 x float>

  // CHECK: rocdl.mfma.f32.16x16x8bf16 {{.*}} : (!llvm.vec<2 x i16>, !llvm.vec<2 x i16>, !llvm.vec<4 x float>, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.vec<4 x float>
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
  // CHECK: %{{.*}} = rocdl.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !llvm.vec<1 x float>
  %r1 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<1 x float>
  // CHECK: %{{.*}} = rocdl.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !llvm.vec<2 x float>
  %r2 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<2 x float>
  // CHECK: %{{.*}} = rocdl.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !llvm.vec<4 x float>
  %r4 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<4 x float>

  // CHECK: rocdl.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !llvm.vec<1 x float>
  rocdl.buffer.store %vdata1, %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<1 x float>
  // CHECK: rocdl.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !llvm.vec<2 x float>
  rocdl.buffer.store %vdata2, %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<2 x float>
  // CHECK: rocdl.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !llvm.vec<4 x float>
  rocdl.buffer.store %vdata4, %rsrc, %vindex, %offset, %glc, %slc : !llvm.vec<4 x float>

  llvm.return
}

