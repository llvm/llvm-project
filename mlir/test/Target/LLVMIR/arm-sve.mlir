// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Dot intrinsics
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_sdot
llvm.func @arm_sve_sdot(%arg0: vector<[16]xi8>,
                        %arg1: vector<[16]xi8>,
                        %arg2: vector<[4]xi32>)
                        -> vector<[4]xi32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.sdot.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.sdot"(%arg2, %arg0, %arg1) :
    (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>)
        -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_udot
llvm.func @arm_sve_udot(%arg0: vector<[16]xi8>,
                        %arg1: vector<[16]xi8>,
                        %arg2: vector<[4]xi32>)
                        -> vector<[4]xi32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.udot.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.udot"(%arg2, %arg0, %arg1) :
    (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>)
        -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

//===----------------------------------------------------------------------===//
// MMLA intrinsics
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_smmla
llvm.func @arm_sve_smmla(%arg0: vector<[16]xi8>,
                         %arg1: vector<[16]xi8>,
                         %arg2: vector<[4]xi32>)
                         -> vector<[4]xi32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.smmla.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.smmla"(%arg2, %arg0, %arg1) :
    (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>)
        -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_ummla
llvm.func @arm_sve_ummla(%arg0: vector<[16]xi8>,
                         %arg1: vector<[16]xi8>,
                         %arg2: vector<[4]xi32>)
                         -> vector<[4]xi32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.ummla.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.ummla"(%arg2, %arg0, %arg1) :
    (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>)
        -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_usmmla
llvm.func @arm_sve_usmmla(%arg0: vector<[16]xi8>,
                         %arg1: vector<[16]xi8>,
                         %arg2: vector<[4]xi32>)
                         -> vector<[4]xi32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.usmmla.nxv4i32(<vscale x 4
  %0 = "arm_sve.intr.usmmla"(%arg2, %arg0, %arg1) :
    (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>)
        -> vector<[4]xi32>
  llvm.return %0 : vector<[4]xi32>
}

// CHECK-LABEL: define <vscale x 4 x float> @arm_sve_bfmmla
llvm.func @arm_sve_bfmmla(%arg0: vector<[8]xbf16>,
                          %arg1: vector<[8]xbf16>,
                          %arg2: vector<[4]xf32>)
                          -> vector<[4]xf32> {
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sve.fmmla.nxv4f32.nxv8bf16(<vscale x 4 x float> %2, <vscale x 8 x bfloat> %0, <vscale x 8 x bfloat> %1)
  %0 = "arm_sve.intr.bfmmla"(%arg2, %arg0, %arg1) :
    (vector<[4]xf32>, vector<[8]xbf16>, vector<[8]xbf16>)
        -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

//===----------------------------------------------------------------------===//
// Masked Arith intrinsics (integer)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_arithi_masked
llvm.func @arm_sve_arithi_masked(%arg0: vector<[4]xi32>,
                                 %arg1: vector<[4]xi32>,
                                 %arg2: vector<[4]xi32>,
                                 %arg3: vector<[4]xi32>,
                                 %arg4: vector<[4]xi32>,
                                 %arg5: vector<[4]xi1>)
                                 -> vector<[4]xi32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.add.nxv4i32
  %0 = "arm_sve.intr.add"(%arg5, %arg0, %arg1) : (vector<[4]xi1>,
                                                  vector<[4]xi32>,
                                                  vector<[4]xi32>)
                                                  -> vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.sub.nxv4i32
  %1 = "arm_sve.intr.sub"(%arg5, %0, %arg1) : (vector<[4]xi1>,
                                               vector<[4]xi32>,
                                               vector<[4]xi32>)
                                               -> vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.mul.nxv4i32
  %2 = "arm_sve.intr.mul"(%arg5, %1, %arg3) : (vector<[4]xi1>,
                                               vector<[4]xi32>,
                                               vector<[4]xi32>)
                                               -> vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.sdiv.nxv4i32
  %3 = "arm_sve.intr.sdiv"(%arg5, %2, %arg4) : (vector<[4]xi1>,
                                               vector<[4]xi32>,
                                               vector<[4]xi32>)
                                               -> vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.udiv.nxv4i32
  %4 = "arm_sve.intr.udiv"(%arg5, %3, %arg4) : (vector<[4]xi1>,
                                               vector<[4]xi32>,
                                               vector<[4]xi32>)
                                               -> vector<[4]xi32>
  llvm.return %4 : vector<[4]xi32>
}

//===----------------------------------------------------------------------===//
// Masked Arith intrinsics (fp)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define <vscale x 4 x float> @arm_sve_arithf_masked
llvm.func @arm_sve_arithf_masked(%arg0: vector<[4]xf32>,
                                 %arg1: vector<[4]xf32>,
                                 %arg2: vector<[4]xf32>,
                                 %arg3: vector<[4]xf32>,
                                 %arg4: vector<[4]xf32>,
                                 %arg5: vector<[4]xi1>)
                                 -> vector<[4]xf32> {
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32
  %0 = "arm_sve.intr.fadd"(%arg5, %arg0, %arg1) : (vector<[4]xi1>,
                                                   vector<[4]xf32>,
                                                   vector<[4]xf32>)
                                                   -> vector<[4]xf32>
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sve.fsub.nxv4f32
  %1 = "arm_sve.intr.fsub"(%arg5, %0, %arg2) : (vector<[4]xi1>,
                                                vector<[4]xf32>,
                                                vector<[4]xf32>)
                                                -> vector<[4]xf32>
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32
  %2 = "arm_sve.intr.fmul"(%arg5, %1, %arg3) : (vector<[4]xi1>,
                                                vector<[4]xf32>,
                                                vector<[4]xf32>)
                                                -> vector<[4]xf32>
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sve.fdiv.nxv4f32
  %3 = "arm_sve.intr.fdiv"(%arg5, %2, %arg4) : (vector<[4]xi1>,
                                                vector<[4]xf32>,
                                                vector<[4]xf32>)
                                                -> vector<[4]xf32>
  llvm.return %3 : vector<[4]xf32>
}

//===----------------------------------------------------------------------===//
// TODO: This is a duplicate of other tests, remove
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_abs_diff
llvm.func @arm_sve_abs_diff(%arg0: vector<[4]xi32>,
                            %arg1: vector<[4]xi32>)
                            -> vector<[4]xi32> {
  // CHECK: sub <vscale x 4 x i32>
  %0 = llvm.sub %arg0, %arg0  : vector<[4]xi32>
  // CHECK: icmp sge <vscale x 4 x i32>
  %1 = llvm.icmp "sge" %arg0, %arg1 : vector<[4]xi32>
  // CHECK: icmp slt <vscale x 4 x i32>
  %2 = llvm.icmp "slt" %arg0, %arg1 : vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.sub.nxv4i32
  %3 = "arm_sve.intr.sub"(%1, %arg0, %arg1) : (vector<[4]xi1>,
                                               vector<[4]xi32>,
                                               vector<[4]xi32>)
                                               -> vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.sub.nxv4i32
  %4 = "arm_sve.intr.sub"(%2, %arg1, %arg0) : (vector<[4]xi1>,
                                               vector<[4]xi32>,
                                               vector<[4]xi32>)
                                               -> vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.add.nxv4i32
  %5 = "arm_sve.intr.add"(%1, %0, %3) : (vector<[4]xi1>,
                                         vector<[4]xi32>,
                                         vector<[4]xi32>)
                                         -> vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.add.nxv4i32
  %6 = "arm_sve.intr.add"(%2, %5, %4) : (vector<[4]xi1>,
                                         vector<[4]xi32>,
                                         vector<[4]xi32>)
                                         -> vector<[4]xi32>
  llvm.return %6 : vector<[4]xi32>
}

//===----------------------------------------------------------------------===//
// From/To bool intrinsics
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @arm_sve_convert_from_svbool(
// CHECK-SAME:                               <vscale x 16 x i1> %[[SVBOOL:[0-9]+]])
llvm.func @arm_sve_convert_from_svbool(%nxv16i1 : vector<[16]xi1>) {
  // CHECK: %[[RES0:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %[[SVBOOL]])
  %res0 = "arm_sve.intr.convert.from.svbool"(%nxv16i1)
    : (vector<[16]xi1>) -> vector<[8]xi1>
  // CHECK: %[[RES1:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %[[SVBOOL]])
  %res1 = "arm_sve.intr.convert.from.svbool"(%nxv16i1)
    : (vector<[16]xi1>) -> vector<[4]xi1>
  // CHECK: %[[RES2:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %[[SVBOOL]])
  %res2 = "arm_sve.intr.convert.from.svbool"(%nxv16i1)
    : (vector<[16]xi1>) -> vector<[2]xi1>
  // CHECK: %[[RES3:.*]] = call <vscale x 1 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv1i1(<vscale x 16 x i1> %[[SVBOOL]])
  %res3 = "arm_sve.intr.convert.from.svbool"(%nxv16i1)
    : (vector<[16]xi1>) -> vector<[1]xi1>
  llvm.return
}

// CHECK-LABEL: arm_sve_convert_to_svbool(
// CHECK-SAME:                            <vscale x 8 x i1> %[[P8:[0-9]+]],
// CHECK-SAME:                            <vscale x 4 x i1> %[[P4:[0-9]+]],
// CHECK-SAME:                            <vscale x 2 x i1> %[[P2:[0-9]+]],
// CHECK-SAME:                            <vscale x 1 x i1> %[[P1:[0-9]+]])
llvm.func @arm_sve_convert_to_svbool(
                                       %nxv8i1  : vector<[8]xi1>,
                                       %nxv4i1  : vector<[4]xi1>,
                                       %nxv2i1  : vector<[2]xi1>,
                                       %nxv1i1  : vector<[1]xi1>
) {
  // CHECK-NEXT: %[[RES0:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[P8]])
  %res0 = "arm_sve.intr.convert.to.svbool"(%nxv8i1)
    : (vector<[8]xi1>) -> vector<[16]xi1>
  // CHECK-NEXT: %[[RES1:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[P4]])
  %res1 = "arm_sve.intr.convert.to.svbool"(%nxv4i1)
    : (vector<[4]xi1>) -> vector<[16]xi1>
  // CHECK-NEXT: %[[RES2:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[P2]])
  %res2 = "arm_sve.intr.convert.to.svbool"(%nxv2i1)
    : (vector<[2]xi1>) -> vector<[16]xi1>
  // CHECK-NEXT: %[[RES3:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv1i1(<vscale x 1 x i1> %[[P1]])
  %res3 = "arm_sve.intr.convert.to.svbool"(%nxv1i1)
    : (vector<[1]xi1>) -> vector<[16]xi1>
  llvm.return
}

//===----------------------------------------------------------------------===//
// Zip intrinsics (fp)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: arm_sve_zip_x2(
// CHECK-SAME:                 <vscale x 16 x i8> %[[V1:[0-9]+]],
// CHECK-SAME:                 <vscale x 8 x i16> %[[V2:[0-9]+]],
// CHECK-SAME:                 <vscale x 4 x i32> %[[V3:[0-9]+]],
// CHECK-SAME:                 <vscale x 2 x i64> %[[V4:[0-9]+]])
llvm.func @arm_sve_zip_x2(%nxv16i8: vector<[16]xi8>, %nxv8i16: vector<[8]xi16>, %nxv4i32: vector<[4]xi32>, %nxv2i64: vector<[2]xi64>) {
  // CHECK: call { <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.zip.x2.nxv16i8(<vscale x 16 x i8> %[[V1]], <vscale x 16 x i8> %[[V1]])
  %0 = "arm_sve.intr.zip.x2"(%nxv16i8, %nxv16i8) : (vector<[16]xi8>, vector<[16]xi8>)
    -> !llvm.struct<(vector<[16]xi8>, vector<[16]xi8>)>
  // CHECK: call { <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.zip.x2.nxv8i16(<vscale x 8 x i16> %[[V2]], <vscale x 8 x i16> %[[V2]])
  %1 = "arm_sve.intr.zip.x2"(%nxv8i16, %nxv8i16) : (vector<[8]xi16>, vector<[8]xi16>)
    -> !llvm.struct<(vector<[8]xi16>, vector<[8]xi16>)>
  // CHECK: call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.zip.x2.nxv4i32(<vscale x 4 x i32> %[[V3]], <vscale x 4 x i32> %[[V3]])
  %2 = "arm_sve.intr.zip.x2"(%nxv4i32, %nxv4i32) : (vector<[4]xi32>, vector<[4]xi32>)
    -> !llvm.struct<(vector<[4]xi32>, vector<[4]xi32>)>
  // CHECK: call { <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.zip.x2.nxv2i64(<vscale x 2 x i64> %[[V4]], <vscale x 2 x i64> %[[V4]])
  %3 = "arm_sve.intr.zip.x2"(%nxv2i64, %nxv2i64) : (vector<[2]xi64>, vector<[2]xi64>)
     -> !llvm.struct<(vector<[2]xi64>, vector<[2]xi64>)>
  llvm.return
}

// CHECK-LABEL: arm_sve_zip_x4(
// CHECK-SAME:                 <vscale x 16 x i8> %[[V1:[0-9]+]],
// CHECK-SAME:                 <vscale x 8 x i16> %[[V2:[0-9]+]],
// CHECK-SAME:                 <vscale x 4 x i32> %[[V3:[0-9]+]],
// CHECK-SAME:                 <vscale x 2 x i64> %[[V4:[0-9]+]])
llvm.func @arm_sve_zip_x4(%nxv16i8: vector<[16]xi8>, %nxv8i16: vector<[8]xi16>, %nxv4i32: vector<[4]xi32>, %nxv2i64: vector<[2]xi64>) {
  // CHECK: call { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } @llvm.aarch64.sve.zip.x4.nxv16i8(<vscale x 16 x i8> %[[V1]], <vscale x 16 x i8> %[[V1]], <vscale x 16 x i8> %[[V1]], <vscale x 16 x i8> %[[V1]])
  %0 = "arm_sve.intr.zip.x4"(%nxv16i8, %nxv16i8, %nxv16i8, %nxv16i8) : (vector<[16]xi8>, vector<[16]xi8>, vector<[16]xi8>, vector<[16]xi8>)
    -> !llvm.struct<(vector<[16]xi8>, vector<[16]xi8>, vector<[16]xi8>, vector<[16]xi8>)>
  // CHECK: call { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } @llvm.aarch64.sve.zip.x4.nxv8i16(<vscale x 8 x i16> %[[V2]], <vscale x 8 x i16> %[[V2]], <vscale x 8 x i16> %[[V2]], <vscale x 8 x i16> %[[V2]])
  %1 = "arm_sve.intr.zip.x4"(%nxv8i16, %nxv8i16, %nxv8i16, %nxv8i16) : (vector<[8]xi16>, vector<[8]xi16>, vector<[8]xi16>, vector<[8]xi16>)
    -> !llvm.struct<(vector<[8]xi16>, vector<[8]xi16>, vector<[8]xi16>, vector<[8]xi16>)>
  // CHECK: call { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.aarch64.sve.zip.x4.nxv4i32(<vscale x 4 x i32> %[[V3]], <vscale x 4 x i32> %[[V3]], <vscale x 4 x i32> %[[V3]], <vscale x 4 x i32> %[[V3]])
  %2 = "arm_sve.intr.zip.x4"(%nxv4i32, %nxv4i32, %nxv4i32, %nxv4i32) : (vector<[4]xi32>, vector<[4]xi32>, vector<[4]xi32>, vector<[4]xi32>)
    -> !llvm.struct<(vector<[4]xi32>, vector<[4]xi32>, vector<[4]xi32>, vector<[4]xi32>)>
  // CHECK: call { <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64> } @llvm.aarch64.sve.zip.x4.nxv2i64(<vscale x 2 x i64> %[[V4]], <vscale x 2 x i64> %[[V4]], <vscale x 2 x i64> %[[V4]], <vscale x 2 x i64> %[[V4]])
  %3 = "arm_sve.intr.zip.x4"(%nxv2i64, %nxv2i64, %nxv2i64, %nxv2i64) : (vector<[2]xi64>, vector<[2]xi64>, vector<[2]xi64>, vector<[2]xi64>)
     -> !llvm.struct<(vector<[2]xi64>, vector<[2]xi64>, vector<[2]xi64>, vector<[2]xi64>)>
  llvm.return
}

//===----------------------------------------------------------------------===//
// While* intrinsics (fp)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: arm_sve_whilelt(
// CHECK-SAME:                  i64 %[[BASE:[0-9]+]],
// CHECK-SAME:                  i64 %[[N:[0-9]+]]
llvm.func @arm_sve_whilelt(%base: i64, %n: i64) {
  // call <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i64(i64 %[[BASE]], i64 %[[N]])
  %1 = "arm_sve.intr.whilelt"(%base, %n) : (i64, i64) -> vector<[2]xi1>
  // call <vscale x 4 x i1> @llvm.aarch64.sve.whilelt.nxv4i1.i64(i64 %[[BASE]], i64 %[[N]])
  %2 = "arm_sve.intr.whilelt"(%base, %n) : (i64, i64) -> vector<[4]xi1>
  // call <vscale x 8 x i1> @llvm.aarch64.sve.whilelt.nxv8i1.i64(i64 %[[BASE]], i64 %[[N]])
  %3 = "arm_sve.intr.whilelt"(%base, %n) : (i64, i64) -> vector<[8]xi1>
  // call <vscale x 16 x i1> @llvm.aarch64.sve.whilelt.nxv16i1.i64(i64 %[[BASE]], i64 %[[N]])
  %4 = "arm_sve.intr.whilelt"(%base, %n) : (i64, i64) -> vector<[16]xi1>
  llvm.return
}

//===----------------------------------------------------------------------===//
// Psel intrinsics (fp)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: arm_sve_psel(
// CHECK-SAME:               <vscale x 16 x i1> %[[PN:[0-9]+]],
// CHECK-SAME:               <vscale x 2 x i1> %[[P1:[0-9]+]],
// CHECK-SAME:               <vscale x 4 x i1> %[[P2:[0-9]+]],
// CHECK-SAME:               <vscale x 8 x i1> %[[P3:[0-9]+]],
// CHECK-SAME:               <vscale x 16 x i1> %[[P4:[0-9]+]],
// CHECK-SAME:               i32 %[[INDEX:[0-9]+]])
llvm.func @arm_sve_psel(%pn: vector<[16]xi1>, %p1: vector<[2]xi1>, %p2: vector<[4]xi1>, %p3: vector<[8]xi1>, %p4: vector<[16]xi1>, %index: i32) {
  // CHECK: call <vscale x 16 x i1> @llvm.aarch64.sve.psel.nxv16i1.nxv2i1(<vscale x 16 x i1> %[[PN]], <vscale x 2 x i1> %[[P1]], i32 %[[INDEX]])
  "arm_sve.intr.psel"(%pn, %p1, %index) : (vector<[16]xi1>, vector<[2]xi1>, i32) -> vector<[16]xi1>
  // CHECK: call <vscale x 16 x i1> @llvm.aarch64.sve.psel.nxv16i1.nxv4i1(<vscale x 16 x i1> %[[PN]], <vscale x 4 x i1> %[[P2]], i32 %[[INDEX]])
  "arm_sve.intr.psel"(%pn, %p2, %index) : (vector<[16]xi1>, vector<[4]xi1>, i32) -> vector<[16]xi1>
  // CHECK: call <vscale x 16 x i1> @llvm.aarch64.sve.psel.nxv16i1.nxv8i1(<vscale x 16 x i1> %[[PN]], <vscale x 8 x i1> %[[P3]], i32 %[[INDEX]])
  "arm_sve.intr.psel"(%pn, %p3, %index) : (vector<[16]xi1>, vector<[8]xi1>, i32) -> vector<[16]xi1>
  // CHECK: call <vscale x 16 x i1> @llvm.aarch64.sve.psel.nxv16i1.nxv16i1(<vscale x 16 x i1> %[[PN]], <vscale x 16 x i1> %[[P4]], i32 %[[INDEX]])
  "arm_sve.intr.psel"(%pn, %p4, %index) : (vector<[16]xi1>, vector<[16]xi1>, i32) -> vector<[16]xi1>
  llvm.return
}

//===----------------------------------------------------------------------===//
// Dupq intrinsics (fp)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @arm_sve_dupq_lane
// CHECK-SAME: <vscale x 16 x i8> %[[V0:[0-9]+]]
// CHECK-SAME: <vscale x 8 x i16> %[[V1:[0-9]+]]
// CHECK-SAME: <vscale x 8 x half> %[[V2:[0-9]+]]
// CHECK-SAME: <vscale x 8 x bfloat> %[[V3:[0-9]+]]
// CHECK-SAME: <vscale x 4 x i32> %[[V4:[0-9]+]]
// CHECK-SAME: <vscale x 4 x float> %[[V5:[0-9]+]]
// CHECK-SAME: <vscale x 2 x i64> %[[V6:[0-9]+]]
// CHECK-SAME: <vscale x 2 x double> %[[V7:[0-9]+]]
llvm.func @arm_sve_dupq_lane(%nxv16i8: vector<[16]xi8>, %nxv8i16:  vector<[8]xi16>,
                             %nxv8f16: vector<[8]xf16>, %nxv8bf16: vector<[8]xbf16>,
                             %nxv4i32: vector<[4]xi32>, %nxv4f32:  vector<[4]xf32>,
                             %nxv2i64: vector<[2]xi64>, %nxv2f64:  vector<[2]xf64>) {
  // CHECK: call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %[[V0]], i64 0)
  %0 = "arm_sve.intr.dupq_lane"(%nxv16i8) <{lane = 0 : i64}> : (vector<[16]xi8>) -> vector<[16]xi8>
  // CHECK: call <vscale x 8 x i16> @llvm.aarch64.sve.dupq.lane.nxv8i16(<vscale x 8 x i16> %[[V1]], i64 1)
  %1 = "arm_sve.intr.dupq_lane"(%nxv8i16) <{lane = 1 : i64}> : (vector<[8]xi16>) -> vector<[8]xi16>
  // CHECK: call <vscale x 8 x half> @llvm.aarch64.sve.dupq.lane.nxv8f16(<vscale x 8 x half> %[[V2]], i64 2)
  %2 = "arm_sve.intr.dupq_lane"(%nxv8f16) <{lane = 2 : i64}> : (vector<[8]xf16>) -> vector<[8]xf16>
  // CHECK: call <vscale x 8 x bfloat> @llvm.aarch64.sve.dupq.lane.nxv8bf16(<vscale x 8 x bfloat> %[[V3]], i64 3)
  %3 = "arm_sve.intr.dupq_lane"(%nxv8bf16) <{lane = 3 : i64}> : (vector<[8]xbf16>) -> vector<[8]xbf16>
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %[[V4]], i64 4)
  %4 = "arm_sve.intr.dupq_lane"(%nxv4i32) <{lane = 4 : i64}> : (vector<[4]xi32>) -> vector<[4]xi32>
  // CHECK: call <vscale x 4 x float> @llvm.aarch64.sve.dupq.lane.nxv4f32(<vscale x 4 x float> %[[V5]], i64 5)
  %5 = "arm_sve.intr.dupq_lane"(%nxv4f32) <{lane = 5 : i64}> : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK: call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %[[V6]], i64 6)
  %6 = "arm_sve.intr.dupq_lane"(%nxv2i64) <{lane = 6 : i64}> : (vector<[2]xi64>) -> vector<[2]xi64>
  // CHECK: call <vscale x 2 x double> @llvm.aarch64.sve.dupq.lane.nxv2f64(<vscale x 2 x double> %[[V7]], i64 7)
  %7 = "arm_sve.intr.dupq_lane"(%nxv2f64) <{lane = 7 : i64}> : (vector<[2]xf64>) -> vector<[2]xf64>
  llvm.return
}
