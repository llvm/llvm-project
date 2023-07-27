// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: @arm_sme_zero
llvm.func @arm_sme_zero() {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.zero(i32 0)
  "arm_sme.intr.zero"(%c0) : (i32) -> ()
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
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.mopa.nxv2f64
  "arm_sme.intr.mopa"(%c0, %nxv2i1, %nxv2i1, %nxv2f64, %nxv2f64) :
    (i32, vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.nxv4f32
  "arm_sme.intr.mopa"(%c0, %nxv4i1, %nxv4i1, %nxv4f32, %nxv4f32) :
    (i32, vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.wide.nxv8f16
  "arm_sme.intr.mopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8f16, %nxv8f16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mopa.wide.nxv8bf16
  "arm_sme.intr.mopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8bf16, %nxv8bf16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_imopa
llvm.func @arm_sme_imopa(%nxv8i16 : vector<[8]xi16>,
                         %nxv16i8 : vector<[16]xi8>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.smopa.wide.nxv8i16
  "arm_sme.intr.smopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umopa.wide.nxv8i16
  "arm_sme.intr.umopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumopa.wide.nxv8i16
  "arm_sme.intr.sumopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmopa.wide.nxv8i16
  "arm_sme.intr.usmopa.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.smopa.wide.nxv16i8
  "arm_sme.intr.smopa.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umopa.wide.nxv16i8
  "arm_sme.intr.umopa.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumopa.wide.nxv16i8
  "arm_sme.intr.sumopa.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmopa.wide.nxv16i8
  "arm_sme.intr.usmopa.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
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
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.mops.nxv2f64
  "arm_sme.intr.mops"(%c0, %nxv2i1, %nxv2i1, %nxv2f64, %nxv2f64) :
    (i32, vector<[2]xi1>, vector<[2]xi1>, vector<[2]xf64>, vector<[2]xf64>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.nxv4f32
  "arm_sme.intr.mops"(%c0, %nxv4i1, %nxv4i1, %nxv4f32, %nxv4f32) :
    (i32, vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.wide.nxv8f16
  "arm_sme.intr.mops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8f16, %nxv8f16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xf16>, vector<[8]xf16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.mops.wide.nxv8bf16
  "arm_sme.intr.mops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8bf16, %nxv8bf16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xbf16>, vector<[8]xbf16>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_imops
llvm.func @arm_sme_imops(%nxv8i16 : vector<[8]xi16>,
                         %nxv16i8 : vector<[16]xi8>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.smops.wide.nxv8i16
  "arm_sme.intr.smops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umops.wide.nxv8i16
  "arm_sme.intr.umops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumops.wide.nxv8i16
  "arm_sme.intr.sumops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmops.wide.nxv8i16
  "arm_sme.intr.usmops.wide"(%c0, %nxv8i1, %nxv8i1, %nxv8i16, %nxv8i16) :
    (i32, vector<[8]xi1>, vector<[8]xi1>, vector<[8]xi16>, vector<[8]xi16>) -> ()
  // CHECK: call void @llvm.aarch64.sme.smops.wide.nxv16i8
  "arm_sme.intr.smops.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.umops.wide.nxv16i8
  "arm_sme.intr.umops.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.sumops.wide.nxv16i8
  "arm_sme.intr.sumops.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  // CHECK: call void @llvm.aarch64.sme.usmops.wide.nxv16i8
  "arm_sme.intr.usmops.wide"(%c0, %nxv16i1, %nxv16i1, %nxv16i8, %nxv16i8) :
    (i32, vector<[16]xi1>, vector<[16]xi1>, vector<[16]xi8>, vector<[16]xi8>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_load
llvm.func @arm_sme_load(%nxv1i1  : vector<[1]xi1>,
                        %nxv2i1  : vector<[2]xi1>,
                        %nxv4i1  : vector<[4]xi1>,
                        %nxv8i1  : vector<[8]xi1>,
                        %nxv16i1 : vector<[16]xi1>,
                        %p8      : !llvm.ptr<i8>,
                        %p16     : !llvm.ptr<i16>,
                        %p32     : !llvm.ptr<i32>,
                        %p64     : !llvm.ptr<i64>,
                        %p128    : !llvm.ptr<i128>) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.ld1q.horiz
  "arm_sme.intr.ld1q.horiz"(%nxv1i1, %p128, %c0, %c0) :
              (vector<[1]xi1>, !llvm.ptr<i128>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1d.horiz
  "arm_sme.intr.ld1d.horiz"(%nxv2i1, %p64, %c0, %c0) :
              (vector<[2]xi1>, !llvm.ptr<i64>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1w.horiz
  "arm_sme.intr.ld1w.horiz"(%nxv4i1, %p32, %c0, %c0) :
              (vector<[4]xi1>, !llvm.ptr<i32>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1h.horiz
  "arm_sme.intr.ld1h.horiz"(%nxv8i1, %p16, %c0, %c0) :
              (vector<[8]xi1>, !llvm.ptr<i16>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1b.horiz
  "arm_sme.intr.ld1b.horiz"(%nxv16i1, %p8, %c0, %c0) :
              (vector<[16]xi1>, !llvm.ptr<i8>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1q.vert
  "arm_sme.intr.ld1q.vert"(%nxv1i1, %p128, %c0, %c0) :
              (vector<[1]xi1>, !llvm.ptr<i128>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1d.vert
  "arm_sme.intr.ld1d.vert"(%nxv2i1, %p64, %c0, %c0) :
              (vector<[2]xi1>, !llvm.ptr<i64>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1w.vert
  "arm_sme.intr.ld1w.vert"(%nxv4i1, %p32, %c0, %c0) :
              (vector<[4]xi1>, !llvm.ptr<i32>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1h.vert
  "arm_sme.intr.ld1h.vert"(%nxv8i1, %p16, %c0, %c0) :
              (vector<[8]xi1>, !llvm.ptr<i16>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.ld1b.vert
  "arm_sme.intr.ld1b.vert"(%nxv16i1, %p8, %c0, %c0) :
              (vector<[16]xi1>, !llvm.ptr<i8>, i32, i32) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_store
llvm.func @arm_sme_store(%nxv1i1  : vector<[1]xi1>,
                         %nxv2i1  : vector<[2]xi1>,
                         %nxv4i1  : vector<[4]xi1>,
                         %nxv8i1  : vector<[8]xi1>,
                         %nxv16i1 : vector<[16]xi1>,
                         %p8      : !llvm.ptr<i8>,
                         %p16     : !llvm.ptr<i16>,
                         %p32     : !llvm.ptr<i32>,
                         %p64     : !llvm.ptr<i64>,
                         %p128    : !llvm.ptr<i128>) {
  %c0 = llvm.mlir.constant(0 : index) : i32
  // CHECK: call void @llvm.aarch64.sme.st1q.horiz
  "arm_sme.intr.st1q.horiz"(%nxv1i1, %p128, %c0, %c0) :
              (vector<[1]xi1>, !llvm.ptr<i128>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1d.horiz
  "arm_sme.intr.st1d.horiz"(%nxv2i1, %p64, %c0, %c0) :
              (vector<[2]xi1>, !llvm.ptr<i64>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1w.horiz
  "arm_sme.intr.st1w.horiz"(%nxv4i1, %p32, %c0, %c0) :
              (vector<[4]xi1>, !llvm.ptr<i32>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1h.horiz
  "arm_sme.intr.st1h.horiz"(%nxv8i1, %p16, %c0, %c0) :
              (vector<[8]xi1>, !llvm.ptr<i16>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1b.horiz
  "arm_sme.intr.st1b.horiz"(%nxv16i1, %p8, %c0, %c0) :
              (vector<[16]xi1>, !llvm.ptr<i8>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1q.vert
  "arm_sme.intr.st1q.vert"(%nxv1i1, %p128, %c0, %c0) :
              (vector<[1]xi1>, !llvm.ptr<i128>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1d.vert
  "arm_sme.intr.st1d.vert"(%nxv2i1, %p64, %c0, %c0) :
              (vector<[2]xi1>, !llvm.ptr<i64>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1w.vert
  "arm_sme.intr.st1w.vert"(%nxv4i1, %p32, %c0, %c0) :
              (vector<[4]xi1>, !llvm.ptr<i32>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1h.vert
  "arm_sme.intr.st1h.vert"(%nxv8i1, %p16, %c0, %c0) :
              (vector<[8]xi1>, !llvm.ptr<i16>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.st1b.vert
  "arm_sme.intr.st1b.vert"(%nxv16i1, %p8, %c0, %c0) :
              (vector<[16]xi1>, !llvm.ptr<i8>, i32, i32) -> ()
  // CHECK: call void @llvm.aarch64.sme.str
  "arm_sme.intr.str"(%c0, %p8) : (i32, !llvm.ptr<i8>) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: @arm_sme_toggle_za
llvm.func @arm_sme_toggle_za() {
  // CHECK: call void @llvm.aarch64.sme.za.enable()
  "arm_sme.intr.za.enable"() : () -> ()
  // CHECK: call void @llvm.aarch64.sme.za.disable()
  "arm_sme.intr.za.disable"() : () -> ()
  llvm.return
}
