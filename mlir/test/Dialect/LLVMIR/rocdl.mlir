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

func.func @rocdl.fmed3.scalar(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-LABEL: rocdl.fmed3.scalar
  // CHECK: %0 = rocdl.fmed3 %arg0, %arg1, %arg2 : f32
  %0 = rocdl.fmed3 %a, %b, %c : f32
  llvm.return %0 : f32
}

func.func @rocdl.fmed3.vector(%a: vector<4xf16>, %b: vector<4xf16>, %c: vector<4xf16>) -> vector<4xf16> {
  // CHECK-LABEL: rocdl.fmed3.vector
  // CHECK: %0 = rocdl.fmed3 %arg0, %arg1, %arg2 : vector<4xf16>
  %0 = rocdl.fmed3 %a, %b, %c : vector<4xf16>
  llvm.return %0 : vector<4xf16>
}

func.func @rocdl.barrier() {
  // CHECK: rocdl.barrier
  rocdl.barrier
  llvm.return
}

func.func @rocdl.sched_barrier() {
  // CHECK: rocdl.sched.barrier
  rocdl.sched.barrier 0
  llvm.return
}

func.func @rocdl_sched_group_barrier() {
  // CHECK: rocdl.sched.group.barrier
  rocdl.sched.group.barrier 8, 1, 0
  llvm.return
}

func.func @rocdl_iglp_opt() {
  // CHECK: rocdl.iglp.opt
  rocdl.iglp.opt 0
  llvm.return
}

func.func @rocdl.setprio() {
  // CHECK: rocdl.s.setprio
  rocdl.s.setprio 0
  llvm.return
}

func.func @rocdl.xdlops(%arg0 : f32, %arg1 : f32,
                   %arg2 : vector<32xf32>, %arg3 : i32,
                   %arg4 : vector<16xf32>, %arg5 : vector<4xf32>,
                   %arg6 : vector<4xf16>, %arg7 : vector<32xi32>,
                   %arg8 : vector<16xi32>, %arg9 : vector<4xi32>,
                   %arg10 : vector<2xi16>, %arg11 : vector<4xi16>,
                   %arg12 : vector<4xf64>, %arg13 : f64,
                   %arg14 : i64, %arg15 : vector<2xf32>,
                   %arg16: vector<8xbf16>, %arg17 : vector<8xf16>) {
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

  // CHECK: rocdl.mfma.f32.16x16x32.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r31 = rocdl.mfma.f32.16x16x32.bf16 %arg16, %arg16, %arg5, %arg3, %arg3, %arg3 :
                              (vector<8xbf16>, vector<8xbf16>, vector<4xf32>,
                               i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.mfma.i32.16x16x64.i8 {{.*}} : (vector<4xi32>, vector<4xi32>, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  %r32 = rocdl.mfma.i32.16x16x64.i8 %arg9, %arg9, %arg9, %arg3, %arg3, %arg3 :
                              (vector<4xi32>, vector<4xi32>, vector<4xi32>,
                               i32, i32, i32) -> vector<4xi32>

  // CHECK: rocdl.mfma.f32.16x16x32.f16 {{.*}} : (vector<8xf16>, vector<8xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xi32>
  %r33 = rocdl.mfma.f32.16x16x32.f16 %arg17, %arg17, %arg5, %arg3, %arg3, %arg3 :
                               (vector<8xf16>, vector<8xf16>, vector<4xf32>,
                                i32, i32, i32) -> vector<4xi32>

  // CHECK: rocdl.mfma.f32.32x32x16.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r34 = rocdl.mfma.f32.32x32x16.bf16 %arg16, %arg16, %arg4, %arg3, %arg3, %arg3 :
                               (vector<8xbf16>, vector<8xbf16>, vector<16xf32>,
                                i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.mfma.i32.32x32x32.i8 {{.*}} : (vector<4xi32>, vector<4xi32>, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  %r35 = rocdl.mfma.i32.32x32x32.i8 %arg9, %arg9, %arg8, %arg3, %arg3, %arg3 :
                               (vector<4xi32>, vector<4xi32>, vector<16xi32>,
                                i32, i32, i32) -> vector<16xi32>

  // CHECK: rocdl.mfma.f32.32x32x16.f16 {{.*}} : (vector<8xf16>, vector<8xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r36 = rocdl.mfma.f32.32x32x16.f16 %arg17, %arg17, %arg4, %arg3, %arg3, %arg3 :
                               (vector<8xf16>, vector<8xf16>, vector<16xf32>,
                                i32, i32, i32) -> vector<16xf32>

  llvm.return
}


llvm.func @rocdl.smfmac(%arg0 : i32,
                   %arg1 : vector<4 x f16>,
                   %arg2 : vector<8 x f16>,
                   %arg3 : vector<4 x f32>,
                   %arg4 : vector<16 x f32>,
                   %arg5 : vector<4 x i16>,
                   %arg6 : vector<8 x i16>,
                   %arg7 : vector<2xi32>,
                   %arg8 : vector<4xi32>,
                   %arg9 : vector<16xi32>) -> vector<4 x f32> {
  %csti32 = llvm.mlir.constant(42 : i32) : i32

  // CHECK-LABEL: rocdl.smfmac
  // CHECK: rocdl.smfmac.f32.16x16x32.f16 %{{.*}} : (vector<4xf16>, vector<8xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r0 = rocdl.smfmac.f32.16x16x32.f16 %arg1, %arg2, %arg3, %csti32, %csti32, %csti32 :
                                (vector<4xf16>, vector<8xf16>, vector<4xf32>,
                                 i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.smfmac.f32.32x32x16.f16 %{{.*}} : (vector<4xf16>, vector<8xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r1 = rocdl.smfmac.f32.32x32x16.f16 %arg1, %arg2, %arg4, %csti32, %csti32, %csti32 :
                                (vector<4xf16>, vector<8xf16>, vector<16xf32>,
                                 i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.smfmac.f32.16x16x32.bf16 %{{.*}} : (vector<4xi16>, vector<8xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r2 = rocdl.smfmac.f32.16x16x32.bf16 %arg5, %arg6, %arg3, %csti32, %csti32, %csti32 :
                                (vector<4xi16>, vector<8xi16>, vector<4xf32>,
                                 i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.smfmac.f32.32x32x16.bf16 %{{.*}} : (vector<4xi16>, vector<8xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r3 = rocdl.smfmac.f32.32x32x16.bf16 %arg5, %arg6, %arg4, %csti32, %csti32, %csti32 :
                                (vector<4xi16>, vector<8xi16>, vector<16xf32>,
                                 i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.smfmac.i32.16x16x64.i8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  %r4 = rocdl.smfmac.i32.16x16x64.i8 %arg7, %arg8, %arg8, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<4xi32>,
                                 i32, i32, i32) -> vector<4xi32>

  // CHECK: rocdl.smfmac.i32.32x32x32.i8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  %r5 = rocdl.smfmac.i32.32x32x32.i8 %arg7, %arg8, %arg9, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<16xi32>,
                                 i32, i32, i32) -> vector<16xi32>

  // CHECK: rocdl.smfmac.f32.16x16x64.bf8.bf8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r6 = rocdl.smfmac.f32.16x16x64.bf8.bf8 %arg7, %arg8, %arg3, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<4xf32>,
                                 i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.smfmac.f32.16x16x64.bf8.fp8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r7 = rocdl.smfmac.f32.16x16x64.bf8.fp8 %arg7, %arg8, %arg3, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<4xf32>,
                                 i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.smfmac.f32.16x16x64.fp8.bf8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r8 = rocdl.smfmac.f32.16x16x64.fp8.bf8 %arg7, %arg8, %arg3, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<4xf32>,
                                 i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.smfmac.f32.16x16x64.fp8.fp8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  %r9 = rocdl.smfmac.f32.16x16x64.fp8.fp8 %arg7, %arg8, %arg3, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<4xf32>,
                                 i32, i32, i32) -> vector<4xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.bf8.bf8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r10 = rocdl.smfmac.f32.32x32x32.bf8.bf8 %arg7, %arg8, %arg4, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<16xf32>,
                                 i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.bf8.fp8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r11 = rocdl.smfmac.f32.32x32x32.bf8.fp8 %arg7, %arg8, %arg4, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<16xf32>,
                                 i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.fp8.bf8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r12 = rocdl.smfmac.f32.32x32x32.fp8.bf8 %arg7, %arg8, %arg4, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<16xf32>,
                                 i32, i32, i32) -> vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.fp8.fp8 %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  %r13 = rocdl.smfmac.f32.32x32x32.fp8.fp8 %arg7, %arg8, %arg4, %csti32, %csti32, %csti32 :
                                (vector<2xi32>, vector<4xi32>, vector<16xf32>,
                                 i32, i32, i32) -> vector<16xf32>

  llvm.return %r0 : vector<4 x f32>
}

llvm.func @rocdl.mfma.scale.f32.32x32x64.f8f6f4(%arg0 : i32,
                   %arg1 : vector<16 x f32>, %arg2 : vector<8xi32>,
                   %arg3 : vector<6xi32>, %arg4 : vector<4xi32>) {
  %cst0 = llvm.mlir.constant(0 : i32) : i32
  %cst1 = llvm.mlir.constant(1 : i32) : i32
  %cst2 = llvm.mlir.constant(2 : i32) : i32
  %cst3 = llvm.mlir.constant(3 : i32) : i32
  %cst4 = llvm.mlir.constant(4 : i32) : i32

  // CHECK-LABEL: rocdl.mfma.scale.f32.32x32x64.f8f6f4
  // fp8 * fp8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r00 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg2, %arg1, %cst0, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp8 * bf8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r01 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg2, %arg1, %cst0, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp8 * fp6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r02 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg3, %arg1, %cst0, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp8 * bf6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r03 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg3, %arg1, %cst0, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp8 * fp4
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r04 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg4, %arg1, %cst0, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf8 * fp8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r10 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg2, %arg1, %cst1, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf8 * bf8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r11 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg2, %arg1, %cst1, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf8 * fp6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r12 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg3, %arg1, %cst1, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf8 * bf6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r13 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg3, %arg1, %cst1, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf8 * fp4
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<8xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r14 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg4, %arg1, %cst1, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp6 * fp8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r20 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg2, %arg1, %cst2, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp6 * bf8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r21 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg2, %arg1, %cst2, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp6 * fp6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r22 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg3, %arg1, %cst2, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp6 * bf6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r23 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg3, %arg1, %cst2, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp6 * fp4
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r24 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg4, %arg1, %cst2, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf6 * fp8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r30 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg2, %arg1, %cst3, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf6 * bf8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r31 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg2, %arg1, %cst3, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf6 * fp6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r32 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg3, %arg1, %cst3, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf6 * bf6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r33 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg3, %arg1, %cst3, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // bf6 * fp4
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<6xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r34 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg4, %arg1, %cst3, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp4 * fp8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r40 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg2, %arg1, %cst4, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp4 * bf8
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r41 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg2, %arg1, %cst4, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp4 * fp6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<4xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r42 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg3, %arg1, %cst4, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp4 * bf6
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<4xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r43 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg3, %arg1, %cst4, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  // fp4 * fp4
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 {{.*}} : (vector<4xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  %r44 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg4, %arg1, %cst4, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>

  llvm.return
}

llvm.func @rocdl.mfma.scale.f32.16x16x128.f8f6f4(%arg0 : i32,
                   %arg1 : vector<4 x f32>, %arg2 : vector<8xi32>,
                   %arg3 : vector<6xi32>, %arg4 : vector<4xi32>) {
  %cst0 = llvm.mlir.constant(0 : i32) : i32
  %cst1 = llvm.mlir.constant(1 : i32) : i32
  %cst2 = llvm.mlir.constant(2 : i32) : i32
  %cst3 = llvm.mlir.constant(3 : i32) : i32
  %cst4 = llvm.mlir.constant(4 : i32) : i32

  // CHECK-LABEL: rocdl.mfma.scale.f32.16x16x128.f8f6f4
  // fp8 * fp8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r00 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg2, %arg1, %cst0, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp8 * bf8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r01 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg2, %arg1, %cst0, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp8 * fp6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r02 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg3, %arg1, %cst0, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp8 * bf6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r03 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg3, %arg1, %cst0, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp8 * fp4
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r04 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg4, %arg1, %cst0, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf8 * fp8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r10 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg2, %arg1, %cst1, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf8 * bf8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r11 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg2, %arg1, %cst1, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf8 * fp6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r12 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg3, %arg1, %cst1, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf8 * bf6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r13 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg3, %arg1, %cst1, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf8 * fp4
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<8xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r14 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg4, %arg1, %cst1, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<8xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp6 * fp8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r20 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg2, %arg1, %cst2, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp6 * bf8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r21 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg2, %arg1, %cst2, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp6 * fp6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r22 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg3, %arg1, %cst2, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp6 * bf6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r23 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg3, %arg1, %cst2, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp6 * fp4
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r24 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg4, %arg1, %cst2, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf6 * fp8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r30 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg2, %arg1, %cst3, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf6 * bf8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r31 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg2, %arg1, %cst3, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf6 * fp6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r32 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg3, %arg1, %cst3, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf6 * bf6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r33 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg3, %arg1, %cst3, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // bf6 * fp4
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<6xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r34 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg4, %arg1, %cst3, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<6xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp4 * fp8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r40 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg2, %arg1, %cst4, %cst0, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp4 * bf8
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r41 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg2, %arg1, %cst4, %cst1, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp4 * fp6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<4xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r42 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg3, %arg1, %cst4, %cst2, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp4 * bf6
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<4xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r43 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg3, %arg1, %cst4, %cst3, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  // fp4 * fp4
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} : (vector<4xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  %r44 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg4, %arg1, %cst4, %cst4, %cst0, %arg0, %cst0, %arg0 :
                              (vector<4xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>

  llvm.return
}

llvm.func @rocdl.ds.read.tr(%ptr : !llvm.ptr<3>) -> vector<4xf16> {
  // CHECK-LABEL: rocdl.ds.read.tr
  // CHECK: rocdl.ds.read.tr4.b64 {{.*}} : <3> -> vector<2xi32>
  %r0 = rocdl.ds.read.tr4.b64 %ptr : !llvm.ptr<3> -> vector<2xi32>
  // CHECK: rocdl.ds.read.tr6.b96 {{.*}} : <3> -> vector<3xi32>
  %r1 = rocdl.ds.read.tr6.b96 %ptr : !llvm.ptr<3> -> vector<3xi32>
  // CHECK: rocdl.ds.read.tr8.b64 {{.*}} : <3> -> vector<2xi32>
  %r2 = rocdl.ds.read.tr8.b64 %ptr : !llvm.ptr<3> -> vector<2xi32>
  // CHECK: rocdl.ds.read.tr16.b64 {{.*}} : <3> -> vector<4xf16>
  %r3 = rocdl.ds.read.tr16.b64 %ptr : !llvm.ptr<3> -> vector<4xf16>
  // CHECK: rocdl.ds.read.tr16.b64 {{.*}} : <3> -> vector<4xbf16>
  %r4 = rocdl.ds.read.tr16.b64 %ptr : !llvm.ptr<3> -> vector<4xbf16>
  llvm.return %r3 : vector<4xf16>
}

llvm.func @rocdl.load.to.lds(%src : !llvm.ptr<7>, %dst: !llvm.ptr<3>) {
  // CHECK-LABEL @rocdl.load.to.lds
  //CHECK: rocdl.load.to.lds %{{.*}}, %{{.*}}, 4, 0, 0 : <7>
  rocdl.load.to.lds %src, %dst, 4, 0, 0 : <7>
  llvm.return
}

llvm.func @rocdl.global.load.lds(%src : !llvm.ptr<1>, %dst: !llvm.ptr<3>) {
  // CHECK-LABEL @rocdl.global.load.lds
  //CHECK: rocdl.global.load.lds %{{.*}}, %{{.*}}, 4, 0, 0
  rocdl.global.load.lds %src, %dst, 4, 0, 0
  llvm.return
}

llvm.func @rocdl.make.buffer.rsrc(%ptr : !llvm.ptr,
                                  %stride : i16,
                                  %numRecords : i64,
                                  %flags : i32) -> !llvm.ptr<8> {
  // CHECK-LABEL: rocdl.make.buffer.rsrc
  // CHECK: %{{.*}} = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.ptr to <8>
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : !llvm.ptr to !llvm.ptr<8>
  llvm.return %rsrc : !llvm.ptr<8>
}

llvm.func @rocdl.raw.ptr.buffer.f32(%rsrc : !llvm.ptr<8>,
                       %offset : i32, %soffset : i32,
                       %aux : i32, %vdata1 : f32,
                       %vdata2 : vector<2xf32>, %vdata4 : vector<4xf32>) {
  // CHECK-LABEL: rocdl.raw.ptr.buffer.f32
  // CHECK: %{{.*}} = rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}} %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}} %{{.*}}, %{{.*}} : vector<2xf32>
  // CHECK: %{{.*}} = rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}} %{{.*}}, %{{.*}} : vector<4xf32>

  // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
  // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
  // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>

  // CHECK: rocdl.raw.ptr.buffer.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
  // CHECK: rocdl.raw.ptr.buffer.atomic.fmax %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32

  %r1 = rocdl.raw.ptr.buffer.load %rsrc, %offset, %soffset, %aux : f32
  %r2 = rocdl.raw.ptr.buffer.load %rsrc, %offset, %soffset, %aux : vector<2xf32>
  %r4 = rocdl.raw.ptr.buffer.load %rsrc, %offset, %soffset, %aux : vector<4xf32>

  rocdl.raw.ptr.buffer.store %vdata1, %rsrc, %offset, %soffset, %aux : f32
  rocdl.raw.ptr.buffer.store %vdata2, %rsrc, %offset, %soffset, %aux : vector<2xf32>
  rocdl.raw.ptr.buffer.store %vdata4, %rsrc, %offset, %offset, %aux : vector<4xf32>

  rocdl.raw.ptr.buffer.atomic.fadd %vdata1, %rsrc, %offset, %soffset, %aux : f32
  rocdl.raw.ptr.buffer.atomic.fmax %vdata1, %rsrc, %offset, %soffset, %aux : f32

  llvm.return
}

llvm.func @rocdl.raw.ptr.buffer.load.lds(%rsrc : !llvm.ptr<8>, %dstLds : !llvm.ptr<3>,
                       %size: i32, %voffset : i32, %soffset : i32, %offset : i32,
                       %aux : i32) {
  // CHECK-LABEL: rocdl.raw.ptr.buffer.load.lds
  // CHECK: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  rocdl.raw.ptr.buffer.load.lds %rsrc, %dstLds, %size, %voffset, %soffset, %offset, %aux

  llvm.return
}

llvm.func @rocdl.raw.ptr.buffer.i32(%rsrc : !llvm.ptr<8>,
                       %offset : i32, %soffset : i32,
                       %aux : i32, %vdata1 : i32,
                       %vdata2 : vector<2xi32>, %vdata4 : vector<4xi32>) {
  // CHECK-LABEL: rocdl.raw.ptr.buffer.i32
  // CHECK: rocdl.raw.ptr.buffer.atomic.smax %{{.*}}, %{{.*}}, %{{.*}} %{{.*}}, %{{.*}} : i32
  // CHECK: rocdl.raw.ptr.buffer.atomic.umin %{{.*}}, %{{.*}}, %{{.*}} %{{.*}}, %{{.*}} : i32
  // CHECK: %{{.*}} = rocdl.raw.ptr.buffer.atomic.cmpswap %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32

  rocdl.raw.ptr.buffer.atomic.smax %vdata1, %rsrc, %offset, %soffset, %aux : i32
  rocdl.raw.ptr.buffer.atomic.umin %vdata1, %rsrc, %offset, %soffset, %aux : i32
  %val = rocdl.raw.ptr.buffer.atomic.cmpswap %vdata1, %vdata1, %rsrc, %offset, %soffset, %aux : i32
  llvm.return
}

// -----

llvm.func @rocdl.raw.buffer.f32(%rsrc : vector<4xi32>,
                       %offset : i32, %soffset : i32,
                       %aux : i32, %vdata1 : f32,
                       %vdata2 : vector<2xf32>, %vdata4 : vector<4xf32>) {
  // CHECK-LABEL: rocdl.raw.buffer.f32
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
  rocdl.raw.buffer.atomic.fmax %vdata1, %rsrc, %offset, %soffset, %aux : f32

  llvm.return
}


llvm.func @rocdl.raw.buffer.i32(%rsrc : vector<4xi32>,
                       %offset : i32, %soffset : i32,
                       %aux : i32, %vdata1 : i32,
                       %vdata2 : vector<2xi32>, %vdata4 : vector<4xi32>) {
  // CHECK-LABEL: rocdl.raw.buffer.i32
  // CHECK: rocdl.raw.buffer.atomic.smax %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : i32
  // CHECK: rocdl.raw.buffer.atomic.umin %{{.*}} %{{.*}} %{{.*}} %{{.*}} %{{.*}} : i32
  // CHECK: %{{.*}} = rocdl.raw.buffer.atomic.cmpswap(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, vector<4xi32>

  rocdl.raw.buffer.atomic.smax %vdata1, %rsrc, %offset, %soffset, %aux : i32
  rocdl.raw.buffer.atomic.umin %vdata1, %rsrc, %offset, %soffset, %aux : i32
  %val = rocdl.raw.buffer.atomic.cmpswap(%vdata1, %vdata1, %rsrc, %offset, %soffset, %aux) : i32, vector<4xi32>
  llvm.return
}

llvm.func @rocdl_8bit_floats(%source: i32, %source_half: f16, %source_bfloat: bf16, %stoch: i32) -> i32 {
// CHECK-LABEL: @rocdl_8bit_floats
// CHECK: rocdl.cvt.f32.bf8
// CHECK: rocdl.cvt.f32.fp8
// CHECK: rocdl.cvt.scalef32.f32.bf8
// CHECK: rocdl.cvt.scalef32.f32.fp8
// CHECK: rocdl.cvt.scalef32.pk.f16.bf8
// CHECK: rocdl.cvt.scalef32.pk.f16.fp8
// CHECK: rocdl.cvt.scalef32.pk.bf16.bf8
// CHECK: rocdl.cvt.scalef32.pk.bf16.fp8
// CHECK: rocdl.cvt.scalef32.f16.fp8
// CHECK: rocdl.cvt.scalef32.f16.bf8
// CHECK: rocdl.cvt.pk.bf8.f32
// CHECK: rocdl.cvt.pk.fp8.f32
// CHECK: rocdl.cvt.pk.f32.bf8
// CHECK: rocdl.cvt.pk.f32.fp8
// CHECK: rocdl.cvt.sr.bf8.f32
// CHECK: rocdl.cvt.sr.fp8.f32
// CHECK: rocdl.cvt.scalef32.sr.fp8.f32
// CHECK: rocdl.cvt.scalef32.sr.fp8.f16
// CHECK: rocdl.cvt.scalef32.sr.fp8.bf16
// CHECK: rocdl.cvt.sr.bf8.f32
// CHECK: rocdl.cvt.scalef32.sr.bf8.f32
// CHECK: rocdl.cvt.scalef32.sr.bf8.f16
// CHECK: rocdl.cvt.scalef32.sr.bf8.bf16
// CHECK: rocdl.cvt.scalef32.pk.f32.fp8
// CHECK: rocdl.cvt.scalef32.pk.f32.bf8
  %c4 = llvm.mlir.constant(1.0 : f32) : f32
  %v1 = rocdl.cvt.f32.bf8 %source[0] : f32
  %v2 = rocdl.cvt.f32.fp8 %source[0] : f32
  %v1_scaled = rocdl.cvt.scalef32.f32.bf8 %source[0], %c4 : f32
  %v2_scaled = rocdl.cvt.scalef32.f32.fp8 %source[0], %c4 : f32
  %v3_scaled = rocdl.cvt.scalef32.pk.f16.bf8 %source[false], %c4 : vector<2xf16>
  %v4_scaled = rocdl.cvt.scalef32.pk.f16.fp8 %source[false], %c4 : vector<2xf16>
  %v3_scaled_bf16 = rocdl.cvt.scalef32.pk.bf16.bf8 %source[false], %c4 : vector<2xbf16>
  %v4_scaled_bf16 = rocdl.cvt.scalef32.pk.bf16.fp8 %source[false], %c4 : vector<2xbf16>
  %v5 = rocdl.cvt.scalef32.f16.fp8 %source[0], %c4 -> %v3_scaled[false] : vector<2xf16>
  %v6  = rocdl.cvt.scalef32.f16.bf8 %source[0], %c4 -> %v3_scaled[false] : vector<2xf16>
  %source2 = rocdl.cvt.pk.bf8.f32 %v1, %v2 -> %source[false] : i32
  %source3 = rocdl.cvt.pk.fp8.f32 %v1, %v2 -> %source2[false] : i32
  %source2_ext = rocdl.cvt.pk.f32.bf8 %source[false] : vector<2xf32>
  %source3_ext = rocdl.cvt.pk.f32.fp8 %source[false] : vector<2xf32>
  %source4 = rocdl.cvt.sr.bf8.f32 %v1, %stoch -> %source3[2] : i32
  %source5 = rocdl.cvt.sr.fp8.f32 %v2, %stoch -> %source4[3] : i32
  %source5_scaled = rocdl.cvt.scalef32.sr.fp8.f32 %v2, %stoch, %c4 -> %source4[3] : i32
  %source5_scaled_half = rocdl.cvt.scalef32.sr.fp8.f16 %source_half, %stoch, %c4 -> %source4[3] : i32
  %source5_scaled_bfloat = rocdl.cvt.scalef32.sr.fp8.bf16 %source_bfloat, %stoch, %c4 -> %source4[3] : i32
  %source6 = rocdl.cvt.sr.bf8.f32 %v1, %stoch -> %source3[3] : i32
  %source6_scaled  = rocdl.cvt.scalef32.sr.bf8.f32 %v2, %stoch, %c4 -> %source3[3] : i32
  %source6_scaled_half = rocdl.cvt.scalef32.sr.bf8.f16 %source_half, %stoch, %c4 -> %source3[3] : i32
  %source6_scaled_bfloat =  rocdl.cvt.scalef32.sr.bf8.bf16 %source_bfloat, %stoch, %c4 -> %source3[3] : i32
  %source7_scaled = rocdl.cvt.scalef32.pk.f32.fp8 %source[false], %c4 : vector<2xf32>
  %source8_scaled = rocdl.cvt.scalef32.pk.f32.bf8 %source[false], %c4 : vector<2xf32>
  llvm.return %source5 : i32
}

llvm.func @rocdl_8bit_packed_v2i16(%sourceA: f32, %sourceB: f32, %old: vector<2xi16>) -> vector<2xi16> {
// CHECK-LABEL: @rocdl_8bit_packed_v2i16
// CHECK: rocdl.cvt.scalef32.pk.fp8.f32
  %c0 = llvm.mlir.constant(1.0 : f32) : f32
  %source_scaled = rocdl.cvt.scalef32.pk.fp8.f32 %sourceA, %sourceB, %c0 -> %old[false] : vector<2xi16>
  %source2_scaled = rocdl.cvt.scalef32.pk.bf8.f32 %sourceA, %sourceB, %c0 -> %old[false] : vector<2xi16>
  llvm.return %source_scaled : vector<2xi16>
}

llvm.func @rocdl_v2f16_v2i16(%source: vector<2xf16>, %source2: vector<2xbf16>, %old: vector<2xi16>) -> vector<2xi16> {
// CHECK-LABEL: @rocdl_v2f16_v2i16
// CHECK: rocdl.cvt.scalef32.pk.fp8.f16
  %c0 = llvm.mlir.constant(1.0 : f32) : f32
  %source_scaled = rocdl.cvt.scalef32.pk.fp8.f16 %source, %c0 -> %old[false] : vector<2xi16>
  %source2_scaled = rocdl.cvt.scalef32.pk.fp8.bf16 %source2, %c0 -> %old[false] : vector<2xi16>
  %source3_scaled = rocdl.cvt.scalef32.pk.bf8.f16 %source, %c0 -> %old[false] : vector<2xi16>
  %source4_scaled = rocdl.cvt.scalef32.pk.bf8.bf16 %source2, %c0 -> %old[false] : vector<2xi16>
  llvm.return %source_scaled : vector<2xi16>
}

// CHECK-LABEL: @rocdl_6_bit_floats
// CHECK-SAME: (%[[V32F6:.+]]: vector<6xi32>, %[[V16F32:.+]]: vector<16xf32>, %[[V32F32:.+]]: vector<32xf32>, %[[V32F16:.+]]: vector<32xf16>, %[[V32BF16:.+]]: vector<32xbf16>, %[[SEED:.+]]: i32, %[[SCALE:.+]]: f32)
llvm.func @rocdl_6_bit_floats(
    %v32f6: vector<6xi32>, %v16f32: vector<16xf32>, %v32f32: vector<32xf32>,
    %v32f16: vector<32xf16>, %v32bf16: vector<32xbf16>, %seed: i32,
    %scale: f32) {
  // CHECK-NEXT: rocdl.cvt.scalef32.2xpk16.bf6.f32 %[[V16F32]], %[[V16F32]], %[[SCALE]]
  %f32_to_bf6 = rocdl.cvt.scalef32.2xpk16.bf6.f32 %v16f32, %v16f32, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.2xpk16.fp6.f32 %[[V16F32]], %[[V16F32]], %[[SCALE]]
  %f32_to_fp6 = rocdl.cvt.scalef32.2xpk16.fp6.f32 %v16f32, %v16f32, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.bf6.f16 %[[V32F16]], %[[SCALE]]
  %f16_to_bf6 = rocdl.cvt.scalef32.pk32.bf6.f16 %v32f16, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.fp6.f16 %[[V32F16]], %[[SCALE]]
  %f16_to_fp6 = rocdl.cvt.scalef32.pk32.fp6.f16 %v32f16, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.bf6.bf16 %[[V32BF16]], %[[SCALE]]
  %bf16_to_bf6 = rocdl.cvt.scalef32.pk32.bf6.bf16 %v32bf16, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.fp6.bf16 %[[V32BF16]], %[[SCALE]]
  %bf16_to_fp6 = rocdl.cvt.scalef32.pk32.fp6.bf16 %v32bf16, %scale : vector<6xi32>

  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.f32.bf6 %[[V32F6]], %[[SCALE]]
  %bf6_to_f32 = rocdl.cvt.scalef32.pk32.f32.bf6 %v32f6, %scale : vector<32xf32>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.f32.fp6 %[[V32F6]], %[[SCALE]]
  %fp6_to_f32 = rocdl.cvt.scalef32.pk32.f32.fp6 %v32f6, %scale : vector<32xf32>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.f16.bf6 %[[V32F6]], %[[SCALE]]
  %bf6_to_f16 = rocdl.cvt.scalef32.pk32.f16.bf6 %v32f6, %scale : vector<32xf16>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.f16.fp6 %[[V32F6]], %[[SCALE]]
  %fp6_to_f16 = rocdl.cvt.scalef32.pk32.f16.fp6 %v32f6, %scale : vector<32xf16>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.bf16.bf6 %[[V32F6]], %[[SCALE]]
  %bf6_to_bf16 = rocdl.cvt.scalef32.pk32.bf16.bf6 %v32f6, %scale : vector<32xbf16>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk32.bf16.fp6 %[[V32F6]], %[[SCALE]]
  %fp6_to_bf16 = rocdl.cvt.scalef32.pk32.bf16.fp6 %v32f6, %scale : vector<32xbf16>

  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk32.bf6.f32 %[[V32F32]], %[[SEED]], %[[SCALE]]
  %f32_to_bf6_sr = rocdl.cvt.scalef32.sr.pk32.bf6.f32 %v32f32, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk32.fp6.f32 %[[V32F32]], %[[SEED]], %[[SCALE]]
  %f32_to_fp6_sr = rocdl.cvt.scalef32.sr.pk32.fp6.f32 %v32f32, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk32.bf6.f16 %[[V32F16]], %[[SEED]], %[[SCALE]]
  %f16_to_bf6_sr = rocdl.cvt.scalef32.sr.pk32.bf6.f16 %v32f16, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk32.fp6.f16 %[[V32F16]], %[[SEED]], %[[SCALE]]
  %f16_to_fp6_sr = rocdl.cvt.scalef32.sr.pk32.fp6.f16 %v32f16, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk32.bf6.bf16 %[[V32BF16]], %[[SEED]], %[[SCALE]]
  %bf16_to_bf6_sr = rocdl.cvt.scalef32.sr.pk32.bf6.bf16 %v32bf16, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk32.fp6.bf16 %[[V32BF16]], %[[SEED]], %[[SCALE]]
  %bf16_to_fp6_sr = rocdl.cvt.scalef32.sr.pk32.fp6.bf16 %v32bf16, %seed, %scale : vector<6xi32>

  llvm.return
}

// CHECK-LABEL: @rocdl_4_bit_floats
// CHECK-SAME: (%[[V8F4:.+]]: i32, %[[F32:.+]]: f32, %[[V2F32:.+]]: vector<2xf32>, %[[V2F16:.+]]: vector<2xf16>, %[[V2BF16:.+]]: vector<2xbf16>, %[[SEED:.+]]: i32, %[[SCALE:.+]]: f32)
llvm.func @rocdl_4_bit_floats(
    %v8f4: i32, %f32: f32, %v2f32: vector<2xf32>, %v2f16: vector<2xf16>,
    %v2bf16: vector<2xbf16>, %seed: i32, %scale: f32) {

  // CHECK-NEXT: rocdl.cvt.scalef32.pk.fp4.f32 %[[F32]], %[[F32]], %[[SCALE]] -> %[[V8F4]][0]
  %f32_to_fp4 = rocdl.cvt.scalef32.pk.fp4.f32 %f32, %f32, %scale -> %v8f4[0] : i32
  // CHECK-NEXT: rocdl.cvt.scalef32.pk.fp4.f16 %[[V2F16]], %[[SCALE]] -> %[[V8F4]][1]
  %f16_to_fp4 = rocdl.cvt.scalef32.pk.fp4.f16 %v2f16, %scale -> %v8f4[1] : i32
  // CHECK-NEXT: rocdl.cvt.scalef32.pk.fp4.bf16 %[[V2BF16]], %[[SCALE]] -> %[[V8F4]][0]
  %bf16_to_fp4 = rocdl.cvt.scalef32.pk.fp4.bf16 %v2bf16, %scale -> %v8f4[0] : i32

  // CHECK-NEXT: rocdl.cvt.scalef32.pk.f32.fp4 %[[V8F4]][0], %[[SCALE]]
  %fp4_to_f32 = rocdl.cvt.scalef32.pk.f32.fp4 %v8f4[0], %scale : vector<2xf32>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk.f16.fp4 %[[V8F4]][1], %[[SCALE]]
  %fp4_to_f16 = rocdl.cvt.scalef32.pk.f16.fp4 %v8f4[1], %scale : vector<2xf16>
  // CHECK-NEXT: rocdl.cvt.scalef32.pk.bf16.fp4 %[[V8F4]][0], %[[SCALE]]
  %fp4_to_bf16 = rocdl.cvt.scalef32.pk.bf16.fp4 %v8f4[0], %scale : vector<2xbf16>

  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk.fp4.f32 %[[V2F32]], %[[SEED]], %[[SCALE]] -> %[[V8F4]][0]
  %f32_to_fp4_sr = rocdl.cvt.scalef32.sr.pk.fp4.f32 %v2f32, %seed, %scale -> %v8f4[0] : i32
  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk.fp4.f16 %[[V2F16]], %[[SEED]], %[[SCALE]] -> %[[V8F4]][1]
  %f16_to_fp4_sr = rocdl.cvt.scalef32.sr.pk.fp4.f16 %v2f16, %seed, %scale -> %v8f4[1] : i32
  // CHECK-NEXT: rocdl.cvt.scalef32.sr.pk.fp4.bf16 %[[V2BF16]], %[[SEED]], %[[SCALE]] -> %[[V8F4]][0]
  %bf16_to_fp4_sr = rocdl.cvt.scalef32.sr.pk.fp4.bf16 %v2bf16, %seed, %scale -> %v8f4[0] : i32

  llvm.return
}

llvm.func @rocdl.s.waitcnt() {
  // CHECK-LABEL: rocdl.s.waitcnt
  // CHECK: rocdl.s.waitcnt 0
  rocdl.s.waitcnt 0
  llvm.return
}

llvm.func @rocdl.s.sleep() {
  // CHECK-LABEL: rocdl.s.sleep
  // CHECK: rocdl.s.sleep 0
  rocdl.s.sleep 0
  llvm.return
}

llvm.func @rocdl.s.barrier() {
  // CHECK-LABEL: rocdl.s.barrier
  // CHECK: rocdl.s.barrier
  rocdl.s.barrier
  llvm.return
}

llvm.func @rocdl.s.barrier.init(%ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.s.barrier.init
  // CHECK: rocdl.s.barrier.init %[[PTR:.+]], 1
  rocdl.s.barrier.init %ptr, 1
  llvm.return
}

llvm.func @rocdl.s.barrier.signal() {
  // CHECK-LABEL: rocdl.s.barrier.signal
  // CHECK: rocdl.s.barrier.signal -1
  rocdl.s.barrier.signal -1
  llvm.return
}

llvm.func @rocdl.s.barrier.signal.var(%ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.s.barrier.signal.var
  // CHECK: rocdl.s.barrier.signal.var %[[PTR:.+]], 1
  rocdl.s.barrier.signal.var %ptr, 1
  llvm.return
}

llvm.func @rocdl.s.barrier.join(%ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.s.barrier.join
  // CHECK: rocdl.s.barrier.join %[[PTR:.+]]
  rocdl.s.barrier.join %ptr
  llvm.return
}

llvm.func @rocdl.s.barrier.leave() {
  // CHECK-LABEL: rocdl.s.barrier.leave
  // CHECK: rocdl.s.barrier.leave 1
  rocdl.s.barrier.leave 1
  llvm.return
}

llvm.func @rocdl.s.barrier.wait() {
  // CHECK-LABEL: rocdl.s.barrier.wait
  // CHECK: rocdl.s.barrier.wait -1
  rocdl.s.barrier.wait -1
  llvm.return
}

llvm.func @rocdl.s.barrier.signal.isfirst() {
  // CHECK-LABEL: rocdl.s.barrier.signal.isfirst
  // CHECK: rocdl.s.barrier.signal.isfirst 1
  %0 = rocdl.s.barrier.signal.isfirst 1 : i1
  llvm.return
}

llvm.func @rocdl.s.get.barrier.state() {
  // CHECK-LABEL: rocdl.s.get.barrier.state
  // CHECK: rocdl.s.get.barrier.state 1
  %0 = rocdl.s.get.barrier.state 1 : i32
  llvm.return
}

llvm.func @rocdl.s.wait.dscnt() {
  // CHECK-LABEL: rocdl.s.wait.dscnt
  // CHECK: rocdl.s.wait.dscnt 0
  rocdl.s.wait.dscnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.loadcnt() {
  // CHECK-LABEL: rocdl.s.wait.loadcnt
  // CHECK: rocdl.s.wait.loadcnt 0
  rocdl.s.wait.loadcnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.storecnt() {
  // CHECK-LABEL: rocdl.s.wait.storecnt
  // CHECK: rocdl.s.wait.storecnt 0
  rocdl.s.wait.storecnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.expcnt() {
  // CHECK-LABEL: rocdl.s.wait.expcnt
  // CHECK: rocdl.s.wait.expcnt 0
  rocdl.s.wait.expcnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.asynccnt() {
  // CHECK-LABEL: rocdl.s.wait.asynccnt
  // CHECK: rocdl.s.wait.asynccnt 0
  rocdl.s.wait.asynccnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.tensorcnt() {
  // CHECK-LABEL: rocdl.s.wait.tensorcnt
  // CHECK: rocdl.s.wait.tensorcnt 0
  rocdl.s.wait.tensorcnt 0
  llvm.return
}

// -----

llvm.func @rocdl.readfirstlane(%src : f32) -> f32 {
  // CHECK-LABEL: rocdl.readfirstlane
  // CHECK: rocdl.readfirstlane %{{.*}} : f32
  %ret = rocdl.readfirstlane %src : f32
  llvm.return %ret : f32
}

llvm.func @rocdl.readlane(%src : f32) -> f32 {
  %cst0 = llvm.mlir.constant(0 : i32) : i32

  // CHECK-LABEL: rocdl.readlane
  // CHECK: rocdl.readlane %{{.*}} %{{.*}}
  %ret = rocdl.readlane %src, %cst0 : (f32, i32) -> f32
  llvm.return %ret : f32
}

// -----

llvm.func @rocdl.permlanex16(%src : f32) -> f32 {
  %cst0 = llvm.mlir.constant(-1 : i32) : i32
  // CHECK-LABEL: rocdl.permlanex16
  // CHECK: rocdl.permlanex16 %{{.*}} %{{.*}}
  %ret = rocdl.permlanex16 %src, %src, %cst0, %cst0, 0, -1 : f32, i32
  llvm.return %ret : f32
}

// -----

llvm.func @rocdl.permlane16.swap(%src : i32) -> !llvm.struct<(i32, i32)> {
  // CHECK-LABEL: rocdl.permlane16.swap
  // CHECK: rocdl.permlane16.swap %{{.*}} %{{.*}}
  %res = rocdl.permlane16.swap %src, %src, 0, -1  : (i32, i32) -> !llvm.struct<(i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32)>
}

llvm.func @rocdl.permlane32.swap(%src : i32) -> !llvm.struct<(i32, i32)> {
  // CHECK-LABEL: rocdl.permlane32.swap
  // CHECK: rocdl.permlane32.swap %{{.*}} %{{.*}}
  %res = rocdl.permlane32.swap %src, %src, 0, -1  : (i32, i32) -> !llvm.struct<(i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32)>
}

// -----

// CHECK-LABEL: rocdl.cvt.scale.pk8
llvm.func @rocdl.cvt.scale.pk8(%i32: i32, %v2xi32: vector<2xi32>, %scale: i32) {

  // CHECK: rocdl.cvt.scale.pk8.f16.fp4
  %0 =      rocdl.cvt.scale.pk8.f16.fp4 %i32, %scale[0] : vector<8xf16>
  // CHECK: rocdl.cvt.scale.pk8.bf16.fp4
  %1 =      rocdl.cvt.scale.pk8.bf16.fp4 %i32, %scale[0] : vector<8xbf16>
  // CHECK: rocdl.cvt.scale.pk8.f32.fp4
  %2 =      rocdl.cvt.scale.pk8.f32.fp4 %i32, %scale[0] : vector<8xf32>

  // CHECK: rocdl.cvt.scale.pk8.f16.fp8
  %3 =      rocdl.cvt.scale.pk8.f16.fp8 %v2xi32, %scale[0] : vector<8xf16>
  // CHECK: rocdl.cvt.scale.pk8.bf16.fp8
  %4 =      rocdl.cvt.scale.pk8.bf16.fp8 %v2xi32, %scale[0] : vector<8xbf16>
  // CHECK: rocdl.cvt.scale.pk8.f32.fp8
  %5 =      rocdl.cvt.scale.pk8.f32.fp8 %v2xi32, %scale[0] : vector<8xf32>

  // CHECK: rocdl.cvt.scale.pk8.f16.bf8
  %6 =      rocdl.cvt.scale.pk8.f16.bf8 %v2xi32, %scale[0] : vector<8xf16>
  // CHECK: rocdl.cvt.scale.pk8.bf16.bf8
  %7 =      rocdl.cvt.scale.pk8.bf16.bf8 %v2xi32, %scale[0] : vector<8xbf16>
  // CHECK: rocdl.cvt.scale.pk8.f32.bf8
  %8 =      rocdl.cvt.scale.pk8.f32.bf8 %v2xi32, %scale[0] : vector<8xf32>

  llvm.return
}

// -----

// CHECK-LABEL: rocdl.cvt.scalef32.pk8
llvm.func @rocdl.cvt.scalef32.pk8(%v8xf32: vector<8xf32>,
                                  %v8xf16: vector<8xf16>,
                                  %v8xbf16: vector<8xbf16>,
                                  %scale: f32) {

  // CHECK: rocdl.cvt.scalef32.pk8.fp8.f32
  %0 =      rocdl.cvt.scalef32.pk8.fp8.f32 %v8xf32, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.pk8.bf8.f32
  %1 =      rocdl.cvt.scalef32.pk8.bf8.f32 %v8xf32, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.pk8.fp4.f32
  %2 =      rocdl.cvt.scalef32.pk8.fp4.f32 %v8xf32, %scale : i32

  // CHECK: rocdl.cvt.scalef32.pk8.fp8.f16
  %3 =      rocdl.cvt.scalef32.pk8.fp8.f16 %v8xf16, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.pk8.bf8.f16
  %4 =      rocdl.cvt.scalef32.pk8.bf8.f16 %v8xf16, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.pk8.fp4.f16
  %5 =      rocdl.cvt.scalef32.pk8.fp4.f16 %v8xf16, %scale : i32

  // CHECK: rocdl.cvt.scalef32.pk8.fp8.bf16
  %6 =      rocdl.cvt.scalef32.pk8.fp8.bf16 %v8xbf16, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.pk8.bf8.bf16
  %7 =      rocdl.cvt.scalef32.pk8.bf8.bf16 %v8xbf16, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.pk8.fp4.bf16
  %8 =      rocdl.cvt.scalef32.pk8.fp4.bf16 %v8xbf16, %scale : i32

  llvm.return
}

// -----

// CHECK-LABEL: rocdl.cvt.scalef32.sr.pk8
llvm.func @rocdl.cvt.scalef32.sr.pk8(%v8xf32: vector<8xf32>,
                                     %v8xf16: vector<8xf16>,
                                     %v8xbf16: vector<8xbf16>,
                                     %seed: i32,
                                     %scale: f32) {

  // CHECK: rocdl.cvt.scalef32.sr.pk8.fp8.f32
  %0 =      rocdl.cvt.scalef32.sr.pk8.fp8.f32 %v8xf32, %seed, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.sr.pk8.bf8.f32
  %1 =      rocdl.cvt.scalef32.sr.pk8.bf8.f32 %v8xf32, %seed, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.sr.pk8.fp4.f32
  %2 =      rocdl.cvt.scalef32.sr.pk8.fp4.f32 %v8xf32, %seed, %scale : i32

  // CHECK: rocdl.cvt.scalef32.sr.pk8.fp8.f16
  %3 =      rocdl.cvt.scalef32.sr.pk8.fp8.f16 %v8xf16, %seed, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.sr.pk8.bf8.f16
  %4 =      rocdl.cvt.scalef32.sr.pk8.bf8.f16 %v8xf16, %seed, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.sr.pk8.fp4.f16
  %5 =      rocdl.cvt.scalef32.sr.pk8.fp4.f16 %v8xf16, %seed, %scale : i32

  // CHECK: rocdl.cvt.scalef32.sr.pk8.fp8.bf16
  %6 =      rocdl.cvt.scalef32.sr.pk8.fp8.bf16 %v8xbf16, %seed, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.sr.pk8.bf8.bf16
  %7 =      rocdl.cvt.scalef32.sr.pk8.bf8.bf16 %v8xbf16, %seed, %scale : vector<2xi32>
  // CHECK: rocdl.cvt.scalef32.sr.pk8.fp4.bf16
  %8 =      rocdl.cvt.scalef32.sr.pk8.fp4.bf16 %v8xbf16, %seed, %scale : i32

  llvm.return
}

// -----

// CHECK-LABEL: rocdl.cvt.scale.pk16
llvm.func @rocdl.cvt.scale.pk16(%v3xi32: vector<3xi32>, %scale:i32) {

  // CHECK: rocdl.cvt.scale.pk16.f16.fp6
  %0 =      rocdl.cvt.scale.pk16.f16.fp6 %v3xi32, %scale[0] : vector<16xf16>
  // CHECK: rocdl.cvt.scale.pk16.bf16.fp6
  %1 =      rocdl.cvt.scale.pk16.bf16.fp6 %v3xi32, %scale[0] : vector<16xbf16>
  // CHECK: rocdl.cvt.scale.pk16.f32.fp6
  %2 =      rocdl.cvt.scale.pk16.f32.fp6 %v3xi32, %scale[0] : vector<16xf32>

  // CHECK: rocdl.cvt.scale.pk16.f16.bf6
  %3 =      rocdl.cvt.scale.pk16.f16.bf6 %v3xi32, %scale[0] : vector<16xf16>
  // CHECK: rocdl.cvt.scale.pk16.bf16.bf6
  %4 =      rocdl.cvt.scale.pk16.bf16.bf6 %v3xi32, %scale[0] : vector<16xbf16>
  // CHECK: rocdl.cvt.scale.pk16.f32.bf6
  %5 =      rocdl.cvt.scale.pk16.f32.bf6 %v3xi32, %scale[0] : vector<16xf32>

  llvm.return
}

// -----

// expected-error@below {{attribute attached to unexpected op}}
func.func private @expected_llvm_func() attributes { rocdl.kernel }

// -----

// Just check these don't emit errors.
gpu.module @module_1 [#rocdl.target<O = 1, chip = "gfx900", abi = "500", link = ["my_device_lib.bc"], flags = {fast, daz, unsafe_math}>] {
}

gpu.module @module_2 [#rocdl.target<chip = "gfx900">, #rocdl.target<chip = "gfx90a">] {
}

gpu.module @module_3 [#rocdl.target<O = 1, chip = "gfx900", abi = "600", link = ["my_device_lib.bc"], flags = {fast, daz, unsafe_math}>] {
}
