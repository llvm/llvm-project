// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: @arm_sme_zero
llvm.func @arm_sme_zero() {
  // CHECK: call void @llvm.aarch64.sme.zero(i32 0)
  "arm_sme.intr.zero"() <{tile_mask = 0 : i32}> : () -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_fmopa
llvm.func @arm_sme_fmopa(%nxv2f64 : vector<[2]xf64>,
                         %nxv4f32 : vector<[4]xf32>,
                         %nxv8f16 : vector<[8]xf16>,
                         %nxv8bf16: vector<[8]xbf16>,
                         %nxv2i1  : vector<[2]xi1>,
                         %nxv4i1  : vector<[4]xi1>,
                         %nxv8i1  : vector<[8]xi1>) {
  // CHECK: call void @llvm.aarch64.sme.mopa.nxv2f64
  "arm_sme.intr.mopa"(%nxv2i1, %nxv2i1, %nxv2f64, %nxv2f64) <{tile_id = 0 : i32}> :
    (vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.nxv4f32
  "arm_sme.intr.mopa"(%nxv4i1, %nxv4i1, %nxv4f32, %nxv4f32) <{tile_id = 0 : i32}> :
    (vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.wide.nxv8f16
  "arm_sme.intr.mopa.wide"(%nxv8i1, %nxv8i1, %nxv8f16, %nxv8f16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.wide.nxv8bf16
  "arm_sme.intr.mopa.wide"(%nxv8i1, %nxv8i1, %nxv8bf16, %nxv8bf16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_imopa
llvm.func @arm_sme_imopa(%nxv8i16 : vector<[8]xi16>,
                         %nxv16i8 : vector<[16]xi8>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>) {
  // CHECK: call void @llvm.aarch64.sme.smopa.wide.nxv8i16
  "arm_sme.intr.smopa.wide"(%nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umopa.wide.nxv8i16
  "arm_sme.intr.umopa.wide"(%nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumopa.wide.nxv8i16
  "arm_sme.intr.sumopa.wide"(%nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmopa.wide.nxv8i16
  "arm_sme.intr.usmopa.wide"(%nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.smopa.wide.nxv16i8
  "arm_sme.intr.smopa.wide"(%nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) <{tile_id = 0 : i32}> :
    (vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umopa.wide.nxv16i8
  "arm_sme.intr.umopa.wide"(%nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) <{tile_id = 0 : i32}> :
    (vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumopa.wide.nxv16i8
  "arm_sme.intr.sumopa.wide"(%nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) <{tile_id = 0 : i32}> :
    (vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmopa.wide.nxv16i8
  "arm_sme.intr.usmopa.wide"(%nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) <{tile_id = 0 : i32}> :
    (vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_fmops
llvm.func @arm_sme_fmops(%nxv2f64 : vector<[2]xf64>,
                         %nxv4f32 : vector<[4]xf32>,
                         %nxv8f16 : vector<[8]xf16>,
                         %nxv8bf16: vector<[8]xbf16>,
                         %nxv2i1  : vector<[2]xi1>,
                         %nxv4i1  : vector<[4]xi1>,
                         %nxv8i1  : vector<[8]xi1>) {
  // CHECK: call void @llvm.aarch64.sme.mops.nxv2f64
  "arm_sme.intr.mops"(%nxv2i1, %nxv2i1, %nxv2f64, %nxv2f64) <{tile_id = 0 : i32}> :
    (vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.nxv4f32
  "arm_sme.intr.mops"(%nxv4i1, %nxv4i1, %nxv4f32, %nxv4f32) <{tile_id = 0 : i32}> :
    (vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.wide.nxv8f16
  "arm_sme.intr.mops.wide"(%nxv8i1, %nxv8i1, %nxv8f16, %nxv8f16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.wide.nxv8bf16
  "arm_sme.intr.mops.wide"(%nxv8i1, %nxv8i1, %nxv8bf16, %nxv8bf16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_imops
llvm.func @arm_sme_imops(%nxv8i16 : vector<[8]xi16>,
                         %nxv16i8 : vector<[16]xi8>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>) {
  // CHECK: call void @llvm.aarch64.sme.smops.wide.nxv8i16
  "arm_sme.intr.smops.wide"(%nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umops.wide.nxv8i16
  "arm_sme.intr.umops.wide"(%nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumops.wide.nxv8i16
  "arm_sme.intr.sumops.wide"(%nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmops.wide.nxv8i16
  "arm_sme.intr.usmops.wide"(%nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) <{tile_id = 0 : i32}> :
    (vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.smops.wide.nxv16i8
  "arm_sme.intr.smops.wide"(%nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) <{tile_id = 0 : i32}> :
    (vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umops.wide.nxv16i8
  "arm_sme.intr.umops.wide"(%nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) <{tile_id = 0 : i32}> :
    (vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumops.wide.nxv16i8
  "arm_sme.intr.sumops.wide"(%nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) <{tile_id = 0 : i32}> :
    (vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmops.wide.nxv16i8
  "arm_sme.intr.usmops.wide"(%nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) <{tile_id = 0 : i32}> :
    (vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_load
llvm.func @arm_sme_load(%nxv1i1  : vector<[1]xi1>,
                        %nxv2i1  : vector<[2]xi1>,
                        %nxv4i1  : vector<[4]xi1>,
                        %nxv8i1  : vector<[8]xi1>,
                        %nxv16i1 : vector<[16]xi1>,
                        %ptr    : !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.ld1q.horiz
  "arm_sme.intr.ld1q.horiz"(%nxv1i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[1]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1d.horiz
  "arm_sme.intr.ld1d.horiz"(%nxv2i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[2]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1w.horiz
  "arm_sme.intr.ld1w.horiz"(%nxv4i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[4]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1h.horiz
  "arm_sme.intr.ld1h.horiz"(%nxv8i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[8]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1b.horiz
  "arm_sme.intr.ld1b.horiz"(%nxv16i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[16]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1q.vert
  "arm_sme.intr.ld1q.vert"(%nxv1i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[1]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1d.vert
  "arm_sme.intr.ld1d.vert"(%nxv2i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[2]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1w.vert
  "arm_sme.intr.ld1w.vert"(%nxv4i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[4]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1h.vert
  "arm_sme.intr.ld1h.vert"(%nxv8i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[8]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1b.vert
  "arm_sme.intr.ld1b.vert"(%nxv16i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[16]xi1>, !llvm.ptr, i32) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_store
llvm.func @arm_sme_store(%nxv1i1  : vector<[1]xi1>,
                         %nxv2i1  : vector<[2]xi1>,
                         %nxv4i1  : vector<[4]xi1>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>,
                         %ptr    : !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.st1q.horiz
  "arm_sme.intr.st1q.horiz"(%nxv1i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[1]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1d.horiz
  "arm_sme.intr.st1d.horiz"(%nxv2i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[2]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1w.horiz
  "arm_sme.intr.st1w.horiz"(%nxv4i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[4]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1h.horiz
  "arm_sme.intr.st1h.horiz"(%nxv8i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[8]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1b.horiz
  "arm_sme.intr.st1b.horiz"(%nxv16i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[16]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1q.vert
  "arm_sme.intr.st1q.vert"(%nxv1i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[1]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1d.vert
  "arm_sme.intr.st1d.vert"(%nxv2i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[2]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1w.vert
  "arm_sme.intr.st1w.vert"(%nxv4i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[4]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1h.vert
  "arm_sme.intr.st1h.vert"(%nxv8i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[8]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1b.vert
  "arm_sme.intr.st1b.vert"(%nxv16i1, %ptr, %c0) <{tile_id = 0 : i32}> :
              (vector<[16]xi1>, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.str
  "arm_sme.intr.str"(%c0, %ptr, %c0) : (i32, !llvm.ptr, i32) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_vector_to_tile_horiz
llvm.func @arm_sme_vector_to_tile_horiz(%tileslice : i32,
                                        %nxv16i1 : vector<[16]xi1>,
                                        %nxv8i1 : vector<[8]xi1>,
                                        %nxv4i1 : vector<[4]xi1>,
                                        %nxv2i1 : vector<[2]xi1>,
                                        %nxv1i1 : vector<[1]xi1>,
                                        %nxv16i8 : vector<[16]xi8>,
                                        %nxv8i16 : vector<[8]xi16>,
                                        %nxv4i32 : vector<[4]xi32>,
                                        %nxv2i64 : vector<[2]xi64>,
                                        %nxv1i128 : vector<[1]xi128>,
                                        %nxv8f16 : vector<[8]xf16>,
                                        %nxv8bf16 : vector<[8]xbf16>,
                                        %nxv4f32 : vector<[4]xf32>,
                                        %nxv2f64 : vector<[2]xf64>) {
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv16i8
  "arm_sme.intr.write.horiz"(%tileslice, %nxv16i1, %nxv16i8) <{tile_id = 0 : i32}> :
      (i32, vector<[16]xi1>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv8i16
  "arm_sme.intr.write.horiz"(%tileslice, %nxv8i1, %nxv8i16) <{tile_id = 0 : i32}> :
      (i32, vector<[8]xi1>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv4i32
  "arm_sme.intr.write.horiz"(%tileslice, %nxv4i1, %nxv4i32) <{tile_id = 0 : i32}> :
      (i32, vector<[4]xi1>, vector<[4]xi32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv2i64
  "arm_sme.intr.write.horiz"(%tileslice, %nxv2i1, %nxv2i64) <{tile_id = 0 : i32}> :
      (i32, vector<[2]xi1>, vector<[2]xi64>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv1i128
  "arm_sme.intr.write.horiz"(%tileslice, %nxv1i1, %nxv1i128) <{tile_id = 0 : i32}> :
      (i32, vector<[1]xi1>, vector<[1]xi128>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv8f16
  "arm_sme.intr.write.horiz"(%tileslice, %nxv8i1, %nxv8f16) <{tile_id = 0 : i32}> :
      (i32, vector<[8]xi1>, vector<[8]xf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv8bf16
  "arm_sme.intr.write.horiz"(%tileslice, %nxv8i1, %nxv8bf16) <{tile_id = 0 : i32}> :
      (i32, vector<[8]xi1>, vector<[8]xbf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv4f32
  "arm_sme.intr.write.horiz"(%tileslice, %nxv4i1, %nxv4f32) <{tile_id = 0 : i32}> :
      (i32, vector<[4]xi1>, vector<[4]xf32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.horiz.nxv2f64
  "arm_sme.intr.write.horiz"(%tileslice, %nxv2i1, %nxv2f64) <{tile_id = 0 : i32}> :
      (i32, vector<[2]xi1>, vector<[2]xf64>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_vector_to_tile_vert
llvm.func @arm_sme_vector_to_tile_vert(%tileslice : i32,
                                       %nxv16i1 : vector<[16]xi1>,
                                       %nxv8i1 : vector<[8]xi1>,
                                       %nxv4i1 : vector<[4]xi1>,
                                       %nxv2i1 : vector<[2]xi1>,
                                       %nxv1i1 : vector<[1]xi1>,
                                       %nxv16i8 : vector<[16]xi8>,
                                       %nxv8i16 : vector<[8]xi16>,
                                       %nxv4i32 : vector<[4]xi32>,
                                       %nxv2i64 : vector<[2]xi64>,
                                       %nxv1i128 : vector<[1]xi128>,
                                       %nxv8f16 : vector<[8]xf16>,
                                       %nxv8bf16 : vector<[8]xbf16>,
                                       %nxv4f32 : vector<[4]xf32>,
                                       %nxv2f64 : vector<[2]xf64>) {
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv16i8
  "arm_sme.intr.write.vert"(%tileslice, %nxv16i1, %nxv16i8) <{tile_id = 0 : i32}> :
      (i32, vector<[16]xi1>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv8i16
  "arm_sme.intr.write.vert"(%tileslice, %nxv8i1, %nxv8i16) <{tile_id = 0 : i32}> :
      (i32, vector<[8]xi1>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv4i32
  "arm_sme.intr.write.vert"(%tileslice, %nxv4i1, %nxv4i32) <{tile_id = 0 : i32}> :
      (i32, vector<[4]xi1>, vector<[4]xi32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv2i64
  "arm_sme.intr.write.vert"(%tileslice, %nxv2i1, %nxv2i64) <{tile_id = 0 : i32}> :
      (i32, vector<[2]xi1>, vector<[2]xi64>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv1i128
  "arm_sme.intr.write.vert"(%tileslice, %nxv1i1, %nxv1i128) <{tile_id = 0 : i32}> :
      (i32, vector<[1]xi1>, vector<[1]xi128>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv8f16
  "arm_sme.intr.write.vert"(%tileslice, %nxv8i1, %nxv8f16) <{tile_id = 0 : i32}> :
      (i32, vector<[8]xi1>, vector<[8]xf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv8bf16
  "arm_sme.intr.write.vert"(%tileslice, %nxv8i1, %nxv8bf16) <{tile_id = 0 : i32}> :
      (i32, vector<[8]xi1>, vector<[8]xbf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv4f32
  "arm_sme.intr.write.vert"(%tileslice, %nxv4i1, %nxv4f32) <{tile_id = 0 : i32}> :
      (i32, vector<[4]xi1>, vector<[4]xf32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.write.vert.nxv2f64
  "arm_sme.intr.write.vert"(%tileslice, %nxv2i1, %nxv2f64) <{tile_id = 0 : i32}> :
      (i32, vector<[2]xi1>, vector<[2]xf64>) -> ()
  llvm.return
}

// -----


llvm.func @arm_sme_tile_slice_to_vector_horiz(%tileslice : i32,
                                              %nxv16i1   : vector<[16]xi1>,
                                              %nxv8i1    : vector<[8]xi1>,
                                              %nxv4i1    : vector<[4]xi1>,
                                              %nxv2i1    : vector<[2]xi1>,
                                              %nxv1i1    : vector<[1]xi1>,
                                              %nxv16i8   : vector<[16]xi8>,
                                              %nxv8i16   : vector<[8]xi16>,
                                              %nxv4i32   : vector<[4]xi32>,
                                              %nxv2i64   : vector<[2]xi64>,
                                              %nxv1i128  : vector<[1]xi128>,
                                              %nxv8f16   : vector<[8]xf16>,
                                              %nxv8bf16  : vector<[8]xbf16>,
                                              %nxv4f32   : vector<[4]xf32>,
                                              %nxv2f64   : vector<[2]xf64>) {
  // CHECK: call <vscale x 16 x i8> @llvm.aarch64.sme.read.horiz.nxv16i8
  %res0 = "arm_sme.intr.read.horiz"(%nxv16i8, %nxv16i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[16]xi8>, vector<[16]xi1>, i32) -> vector<[16]xi8>
  // CHECK: call <vscale x 8 x i16> @llvm.aarch64.sme.read.horiz.nxv8i16
  %res1 = "arm_sme.intr.read.horiz"(%nxv8i16, %nxv8i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[8]xi16>, vector<[8]xi1>, i32) -> vector<[8]xi16>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sme.read.horiz.nxv4i32
  %res2 = "arm_sme.intr.read.horiz"(%nxv4i32, %nxv4i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>
  // CHECK: call <vscale x 2 x i64> @llvm.aarch64.sme.read.horiz.nxv2i64
  %res3 = "arm_sme.intr.read.horiz"(%nxv2i64, %nxv2i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[2]xi64>, vector<[2]xi1>, i32) -> vector<[2]xi64>
  // CHECK: call <vscale x 1 x i128> @llvm.aarch64.sme.read.horiz.nxv1i128
  %res4 = "arm_sme.intr.read.horiz"(%nxv1i128, %nxv1i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[1]xi128>, vector<[1]xi1>, i32) -> vector<[1]xi128>
  // CHECK: call <vscale x 8 x half> @llvm.aarch64.sme.read.horiz.nxv8f16
  %res5 = "arm_sme.intr.read.horiz"(%nxv8f16, %nxv8i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[8]xf16>, vector<[8]xi1>, i32) -> vector<[8]xf16>
  // CHECK: call <vscale x 8 x bfloat> @llvm.aarch64.sme.read.horiz.nxv8bf16
  %res6 = "arm_sme.intr.read.horiz"(%nxv8bf16, %nxv8i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[8]xbf16>, vector<[8]xi1>, i32) -> vector<[8]xbf16>
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sme.read.horiz.nxv4f32
  %res7 = "arm_sme.intr.read.horiz"(%nxv4f32, %nxv4i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  // CHECK: call <vscale x 2 x double> @llvm.aarch64.sme.read.horiz.nxv2f64
  %res8 = "arm_sme.intr.read.horiz"(%nxv2f64, %nxv2i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[2]xf64>, vector<[2]xi1>, i32) -> vector<[2]xf64>
  llvm.return
}

// -----

llvm.func @arm_sme_tile_slice_to_vector_vert(%tileslice : i32,
                                              %nxv16i1  : vector<[16]xi1>,
                                              %nxv8i1   : vector<[8]xi1>,
                                              %nxv4i1   : vector<[4]xi1>,
                                              %nxv2i1   : vector<[2]xi1>,
                                              %nxv1i1   : vector<[1]xi1>,
                                              %nxv16i8  : vector<[16]xi8>,
                                              %nxv8i16  : vector<[8]xi16>,
                                              %nxv4i32  : vector<[4]xi32>,
                                              %nxv2i64  : vector<[2]xi64>,
                                              %nxv1i128 : vector<[1]xi128>,
                                              %nxv8f16  : vector<[8]xf16>,
                                              %nxv8bf16 : vector<[8]xbf16>,
                                              %nxv4f32  : vector<[4]xf32>,
                                              %nxv2f64  : vector<[2]xf64>) {
  // CHECK: call <vscale x 16 x i8> @llvm.aarch64.sme.read.vert.nxv16i8
  %res0 = "arm_sme.intr.read.vert"(%nxv16i8, %nxv16i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[16]xi8>, vector<[16]xi1>, i32) -> vector<[16]xi8>
  // CHECK: call <vscale x 8 x i16> @llvm.aarch64.sme.read.vert.nxv8i16
  %res1 = "arm_sme.intr.read.vert"(%nxv8i16, %nxv8i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[8]xi16>, vector<[8]xi1>, i32) -> vector<[8]xi16>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sme.read.vert.nxv4i32
  %res2 = "arm_sme.intr.read.vert"(%nxv4i32, %nxv4i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>
  // CHECK: call <vscale x 2 x i64> @llvm.aarch64.sme.read.vert.nxv2i64
  %res3 = "arm_sme.intr.read.vert"(%nxv2i64, %nxv2i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[2]xi64>, vector<[2]xi1>, i32) -> vector<[2]xi64>
  // CHECK: call <vscale x 1 x i128> @llvm.aarch64.sme.read.vert.nxv1i128
  %res4 = "arm_sme.intr.read.vert"(%nxv1i128, %nxv1i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[1]xi128>, vector<[1]xi1>, i32) -> vector<[1]xi128>
  // CHECK: call <vscale x 8 x half> @llvm.aarch64.sme.read.vert.nxv8f16
  %res5 = "arm_sme.intr.read.vert"(%nxv8f16, %nxv8i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[8]xf16>, vector<[8]xi1>, i32) -> vector<[8]xf16>
  // CHECK: call <vscale x 8 x bfloat> @llvm.aarch64.sme.read.vert.nxv8bf16
  %res6 = "arm_sme.intr.read.vert"(%nxv8bf16, %nxv8i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[8]xbf16>, vector<[8]xi1>, i32) -> vector<[8]xbf16>
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sme.read.vert.nxv4f32
  %res7 = "arm_sme.intr.read.vert"(%nxv4f32, %nxv4i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  // CHECK: call <vscale x 2 x double> @llvm.aarch64.sme.read.vert.nxv2f64
  %res8 = "arm_sme.intr.read.vert"(%nxv2f64, %nxv2i1, %tileslice) <{tile_id = 0 : i32}>
    : (vector<[2]xf64>, vector<[2]xi1>, i32) -> vector<[2]xf64>
  llvm.return
}

// -----

llvm.func @arm_sme_streaming_vl() {
  // CHECK: call i64 @llvm.aarch64.sme.cntsb()
  %svl_b = "arm_sme.intr.cntsb"() : () -> i64
  // CHECK: call i64 @llvm.aarch64.sme.cntsh()
  %svl_h = "arm_sme.intr.cntsh"() : () -> i64
  // CHECK: call i64 @llvm.aarch64.sme.cntsw()
  %svl_w = "arm_sme.intr.cntsw"() : () -> i64
  // CHECK: call i64 @llvm.aarch64.sme.cntsd()
  %svl_d = "arm_sme.intr.cntsd"() : () -> i64
  llvm.return
}
