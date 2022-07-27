// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

func.func @rocdl_special_regs() -> i32 {
  // CHECK-LABEL: rocdl_special_regs
  // CHECK: rocdl.workitem.id.x : i32
  %0 = rocdl.workitem.id.x : i32
  // CHECK: rocdl.workitem.id.y : i32
  %1 = rocdl.workitem.id.y : i32
  // CHECK: rocdl.workitem.id.z : i32
  %2 = rocdl.workitem.id.z : i32
  // CHECK: rocdl.workgroup.id.x : i32
  %3 = rocdl.workgroup.id.x : i32
  // CHECK: rocdl.workgroup.id.y : i32
  %4 = rocdl.workgroup.id.y : i32
  // CHECK: rocdl.workgroup.id.z : i32
  %5 = rocdl.workgroup.id.z : i32
  // CHECK: rocdl.workgroup.dim.x : i32
  %6 = rocdl.workgroup.dim.x : i32
  // CHECK: rocdl.workgroup.dim.y : i32
  %7 = rocdl.workgroup.dim.y : i32
  // CHECK: rocdl.workgroup.dim.z : i32
  %8 = rocdl.workgroup.dim.z : i32
  // CHECK: rocdl.grid.dim.x : i32
  %9 = rocdl.grid.dim.x : i32
  // CHECK: rocdl.grid.dim.y : i32
  %10 = rocdl.grid.dim.y : i32
  // CHECK: rocdl.grid.dim.z : i32
  %11 = rocdl.grid.dim.z : i32
  llvm.return %0 : i32
}

func.func @rocdl.barrier() {
  // CHECK: rocdl.barrier
  rocdl.barrier
  llvm.return
}

func.func @rocdl.xdlops(%arg0 : f32, %arg1 : f32,
                   %arg2 : vector<32xf32>, %arg3 : i32,
                   %arg4 : vector<16xf32>, %arg5 : vector<4xf32>,
                   %arg6 : vector<4xf16>, %arg7 : vector<32xi32>,
                   %arg8 : vector<16xi32>, %arg9 : vector<4xi32>,
                   %arg10 : vector<2xi16>, %arg11 : vector<4xi16>,
                   %arg12 : vector<4xf64>, %arg13 : f64,
                   %arg14 : i64, %arg15 : vector<2xf32>) {
  // CHECK-LABEL: rocdl.xdlops
  // CHECK: rocdl.mfma.f32.32x32x1f32 {{.*}} : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
  %r0 = rocdl.mfma.f32.32x32x1f32 %arg0, %arg1, %arg2, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<32xf32>,
                            i32, i32, i32) -> vector<32xf32>

  // CHECK: rocdl.mfma.f32.16x16x1f32 {{.*}} : (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r1 = rocdl.mfma.f32.16x16x1f32 %arg0, %arg1, %arg4, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.f32.4x4x1f32 {{.*}} : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r2 = rocdl.mfma.f32.4x4x1f32 %arg0, %arg1, %arg5, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.f32.32x32x2f32 {{.*}} : (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r3= rocdl.mfma.f32.32x32x2f32 %arg0, %arg1, %arg4, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.f32.16x16x4f32 {{.*}} : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r4 = rocdl.mfma.f32.16x16x4f32 %arg0, %arg1, %arg5, %arg3, %arg3, %arg3 :
                            (f32, f32, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.f32.32x32x4f16 {{.*}} : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
  %r5 = rocdl.mfma.f32.32x32x4f16 %arg6, %arg6, %arg2, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<32xf32>,
                            i32, i32, i32) -> vector<32xf32>

  // CHECK: rocdl.mfma.f32.16x16x4f16 {{.*}} : (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r6 = rocdl.mfma.f32.16x16x4f16 %arg6, %arg6, %arg4, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.f32.4x4x4f16 {{.*}} : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r7 = rocdl.mfma.f32.4x4x4f16 %arg6, %arg6, %arg5, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.f32.32x32x8f16 {{.*}} : (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r8 = rocdl.mfma.f32.32x32x8f16 %arg6, %arg6, %arg4, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.f32.16x16x16f16 {{.*}} : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r9 = rocdl.mfma.f32.16x16x16f16 %arg6, %arg6, %arg5, %arg3, %arg3, %arg3 :
                            (vector<4xf16>, vector<4xf16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.i32.32x32x4i8 {{.*}} : (i32, i32, vector<32xi32>, i32, i32, i32) -> vector<32xi32>
  %r10 = rocdl.mfma.i32.32x32x4i8 %arg3, %arg3, %arg7, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<32xi32>,
                            i32, i32, i32) -> vector<32xi32>

  // CHECK: rocdl.mfma.i32.16x16x4i8 {{.*}} : (i32, i32, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  %r11 = rocdl.mfma.i32.16x16x4i8 %arg3, %arg3, %arg8, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<16xi32>,
                            i32, i32, i32) -> vector<16xi32>

  // CHECK: rocdl.mfma.i32.4x4x4i8 {{.*}} : (i32, i32, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  %r12 = rocdl.mfma.i32.4x4x4i8 %arg3, %arg3, %arg9, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<4xi32>,
                            i32, i32, i32) -> vector<4xi32>

  // CHECK: rocdl.mfma.i32.32x32x8i8 {{.*}} : (i32, i32, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  %r13 = rocdl.mfma.i32.32x32x8i8 %arg3, %arg3, %arg8, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<16xi32>,
                            i32, i32, i32) -> vector<16xi32>

  // CHECK: rocdl.mfma.i32.16x16x16i8 {{.*}} : (i32, i32, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  %r14 = rocdl.mfma.i32.16x16x16i8 %arg3, %arg3, %arg9, %arg3, %arg3, %arg3 :
                            (i32, i32, vector<4xi32>,
                            i32, i32, i32) -> vector<4xi32>

  // CHECK: rocdl.mfma.f32.32x32x2bf16 {{.*}} : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
  %r15 = rocdl.mfma.f32.32x32x2bf16 %arg10, %arg10, %arg2, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<32xf32>,
                            i32, i32, i32) -> vector<32xf32>

  // CHECK: rocdl.mfma.f32.16x16x2bf16 {{.*}} : (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r16 = rocdl.mfma.f32.16x16x2bf16 %arg10, %arg10, %arg4, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.f32.4x4x2bf16 {{.*}} : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r17 = rocdl.mfma.f32.4x4x2bf16 %arg10, %arg10, %arg5, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.f32.32x32x4bf16 {{.*}} : (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r18 = rocdl.mfma.f32.32x32x4bf16 %arg10, %arg10, %arg4, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.f32.16x16x8bf16 {{.*}} : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r19 = rocdl.mfma.f32.16x16x8bf16 %arg10, %arg10, %arg5, %arg3, %arg3, %arg3 :
                            (vector<2xi16>, vector<2xi16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>


  // CHECK: rocdl.mfma.f32.32x32x4bf16.1k {{.*}} : (vector<4xi16>, vector<4xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
  %r20 = rocdl.mfma.f32.32x32x4bf16.1k %arg11, %arg11, %arg2, %arg3, %arg3, %arg3 :
                            (vector<4xi16>, vector<4xi16>, vector<32xf32>,
                            i32, i32, i32) -> vector<32xf32>

  // CHECK: rocdl.mfma.f32.16x16x4bf16.1k {{.*}} : (vector<4xi16>, vector<4xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r21 = rocdl.mfma.f32.16x16x4bf16.1k %arg11, %arg11, %arg4, %arg3, %arg3, %arg3 :
                            (vector<4xi16>, vector<4xi16>, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.f32.4x4x4bf16.1k {{.*}} : (vector<4xi16>, vector<4xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r22 = rocdl.mfma.f32.4x4x4bf16.1k %arg11, %arg11, %arg5, %arg3, %arg3, %arg3 :
                            (vector<4xi16>, vector<4xi16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.f32.32x32x8bf16.1k {{.*}} : (vector<4xi16>, vector<4xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r23 = rocdl.mfma.f32.32x32x8bf16.1k %arg11, %arg11, %arg4, %arg3, %arg3, %arg3 :
                            (vector<4xi16>, vector<4xi16>, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k {{.*}} : (vector<4xi16>, vector<4xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r24 = rocdl.mfma.f32.16x16x16bf16.1k %arg11, %arg11, %arg5, %arg3, %arg3, %arg3 :
                            (vector<4xi16>, vector<4xi16>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.f64.16x16x4f64 {{.*}} : (f64, f64, vector<4xf64>, i32, i32, i32) -> vector<4xf64>
  %r25 = rocdl.mfma.f64.16x16x4f64 %arg13, %arg13, %arg12, %arg3, %arg3, %arg3 :
                            (f64, f64, vector<4xf64>,
                            i32, i32, i32) -> vector<4xf64>

  // CHECK: rocdl.mfma.f64.4x4x4f64 {{.*}} : (f64, f64, f64, i32, i32, i32) -> f64
  %r26 = rocdl.mfma.f64.4x4x4f64 %arg13, %arg13, %arg13, %arg3, %arg3, %arg3 :
                            (f64, f64, f64,
                            i32, i32, i32) -> f64

  // CHECK: rocdl.mfma.i32.16x16x32.i8 {{.*}} : (i64, i64, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  %r27 = rocdl.mfma.i32.16x16x32.i8 %arg14, %arg14, %arg9, %arg3, %arg3, %arg3 :
                            (i64, i64, vector<4xi32>,
                            i32, i32, i32) -> vector<4xi32>

  // CHECK: rocdl.mfma.i32.32x32x16.i8 {{.*}} : (i64, i64, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  %r28 = rocdl.mfma.i32.32x32x16.i8 %arg14, %arg14, %arg8, %arg3, %arg3, %arg3 :
                            (i64, i64, vector<16xi32>,
                            i32, i32, i32) -> vector<16xi32>

  // CHECK: rocdl.mfma.f32.16x16x8.xf32 {{.*}} : (vector<2xf32>, vector<2xf32>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r29 = rocdl.mfma.f32.16x16x8.xf32 %arg15, %arg15, %arg5, %arg3, %arg3, %arg3 :
                            (vector<2xf32>, vector<2xf32>, vector<4xf32>,
                            i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.f32.32x32x4.xf32 {{.*}} : (vector<2xf32>, vector<2xf32>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r30 = rocdl.mfma.f32.32x32x4.xf32 %arg15, %arg15, %arg4, %arg3, %arg3, %arg3 :
                            (vector<2xf32>, vector<2xf32>, vector<16xf32>,
                            i32, i32, i32) -> vector<16xf32>

  llvm.return
}

llvm.func @rocdl.mubuf(%rsrc : vector<4xi32>, %vindex : i32,
                       %offset : i32, %glc : i1,
                       %slc : i1, %vdata1 : vector<1xf32>,
                       %vdata2 : vector<2xf32>, %vdata4 : vector<4xf32>) {
  // CHECK-LABEL: rocdl.mubuf
  // CHECK: %{{.*}} = rocdl.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<1xf32>
  %r1 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<1xf32>
  // CHECK: %{{.*}} = rocdl.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<2xf32>
  %r2 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<2xf32>
  // CHECK: %{{.*}} = rocdl.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<4xf32>
  %r4 = rocdl.buffer.load %rsrc, %vindex, %offset, %glc, %slc : vector<4xf32>

  // CHECK: rocdl.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<1xf32>
  rocdl.buffer.store %vdata1, %rsrc, %vindex, %offset, %glc, %slc : vector<1xf32>
  // CHECK: rocdl.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<2xf32>
  rocdl.buffer.store %vdata2, %rsrc, %vindex, %offset, %glc, %slc : vector<2xf32>
  // CHECK: rocdl.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<4xf32>
  rocdl.buffer.store %vdata4, %rsrc, %vindex, %offset, %glc, %slc : vector<4xf32>

  llvm.return
}

llvm.func @rocdl.raw.buffer(%rsrc : vector<4xi32>,
                       %offset : i32, %soffset : i32,
                       %aux : i32, %vdata1 : f32,
                       %vdata2 : vector<2xf32>, %vdata4 : vector<4xf32>) {
  // CHECK-LABEL: rocdl.raw.buffer
  // CHECK: %{{.*}} = rocdl.raw.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} : f32
  // CHECK: %{{.*}} = rocdl.raw.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<2xf32>
  // CHECK: %{{.*}} = rocdl.raw.buffer.load %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<4xf32>

  // CHECK: rocdl.raw.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : f32
  // CHECK: rocdl.raw.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<2xf32>
  // CHECK: rocdl.raw.buffer.store %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : vector<4xf32>

  // CHECK: rocdl.raw.buffer.atomic.fadd %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : f32

  %r1 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : f32
  %r2 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : vector<2xf32>
  %r4 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : vector<4xf32>

  rocdl.raw.buffer.store %vdata1, %rsrc, %offset, %soffset, %aux : f32
  rocdl.raw.buffer.store %vdata2, %rsrc, %offset, %soffset, %aux : vector<2xf32>
  rocdl.raw.buffer.store %vdata4, %rsrc, %offset, %offset, %aux : vector<4xf32>

  rocdl.raw.buffer.atomic.fadd %vdata1, %rsrc, %offset, %soffset, %aux : f32

  llvm.return
}

// -----

// expected-error@below {{attribute attached to unexpected op}}
func.func private @expected_llvm_func() attributes { rocdl.kernel }
