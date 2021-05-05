// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_sdot
llvm.func @arm_sve_sdot(%arg0: !llvm.vec<?x16 x i8>,
                        %arg1: !llvm.vec<?x16 x i8>,
                        %arg2: !llvm.vec<?x4 x i32>)
                        -> !llvm.vec<?x4 x i32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.sdot.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.sdot"(%arg2, %arg0, %arg1) :
    (!llvm.vec<?x4 x i32>, !llvm.vec<?x16 x i8>, !llvm.vec<?x16 x i8>)
        -> !llvm.vec<?x4 x i32>
  llvm.return %0 : !llvm.vec<?x4 x i32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_smmla
llvm.func @arm_sve_smmla(%arg0: !llvm.vec<?x16 x i8>,
                         %arg1: !llvm.vec<?x16 x i8>,
                         %arg2: !llvm.vec<?x4 x i32>)
                         -> !llvm.vec<?x4 x i32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.smmla.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.smmla"(%arg2, %arg0, %arg1) :
    (!llvm.vec<?x4 x i32>, !llvm.vec<?x16 x i8>, !llvm.vec<?x16 x i8>)
        -> !llvm.vec<?x4 x i32>
  llvm.return %0 : !llvm.vec<?x4 x i32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_udot
llvm.func @arm_sve_udot(%arg0: !llvm.vec<?x16 x i8>,
                        %arg1: !llvm.vec<?x16 x i8>,
                        %arg2: !llvm.vec<?x4 x i32>)
                        -> !llvm.vec<?x4 x i32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.udot.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.udot"(%arg2, %arg0, %arg1) :
    (!llvm.vec<?x4 x i32>, !llvm.vec<?x16 x i8>, !llvm.vec<?x16 x i8>)
        -> !llvm.vec<?x4 x i32>
  llvm.return %0 : !llvm.vec<?x4 x i32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_ummla
llvm.func @arm_sve_ummla(%arg0: !llvm.vec<?x16 x i8>,
                         %arg1: !llvm.vec<?x16 x i8>,
                         %arg2: !llvm.vec<?x4 x i32>)
                         -> !llvm.vec<?x4 x i32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.ummla.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.ummla"(%arg2, %arg0, %arg1) :
    (!llvm.vec<?x4 x i32>, !llvm.vec<?x16 x i8>, !llvm.vec<?x16 x i8>)
        -> !llvm.vec<?x4 x i32>
  llvm.return %0 : !llvm.vec<?x4 x i32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_arithi
llvm.func @arm_sve_arithi(%arg0: !llvm.vec<? x 4 x i32>,
                          %arg1: !llvm.vec<? x 4 x i32>,
                          %arg2: !llvm.vec<? x 4 x i32>)
                          -> !llvm.vec<? x 4 x i32> {
  // CHECK: mul <vscale x 4 x i32>
  %0 = llvm.mul %arg0, %arg1 : !llvm.vec<? x 4 x i32>
  // CHECK: add <vscale x 4 x i32>
  %1 = llvm.add %0, %arg2 : !llvm.vec<? x 4 x i32>
  llvm.return %1 : !llvm.vec<? x 4 x i32>
}

// CHECK-LABEL: define <vscale x 4 x float> @arm_sve_arithf
llvm.func @arm_sve_arithf(%arg0: !llvm.vec<? x 4 x f32>,
                          %arg1: !llvm.vec<? x 4 x f32>,
                          %arg2: !llvm.vec<? x 4 x f32>)
                          -> !llvm.vec<? x 4 x f32> {
  // CHECK: fmul <vscale x 4 x float>
  %0 = llvm.fmul %arg0, %arg1 : !llvm.vec<? x 4 x f32>
  // CHECK: fadd <vscale x 4 x float>
  %1 = llvm.fadd %0, %arg2 : !llvm.vec<? x 4 x f32>
  llvm.return %1 : !llvm.vec<? x 4 x f32>
}

// CHECK-LABEL: define i64 @get_vector_scale()
llvm.func @get_vector_scale() -> i64 {
  // CHECK: call i64 @llvm.vscale.i64()
  %0 = "arm_sve.vscale"() : () -> i64
  llvm.return %0 : i64
}
