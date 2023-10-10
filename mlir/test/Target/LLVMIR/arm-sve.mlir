// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

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

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_arithi
llvm.func @arm_sve_arithi(%arg0: vector<[4]xi32>,
                          %arg1: vector<[4]xi32>,
                          %arg2: vector<[4]xi32>)
                          -> vector<[4]xi32> {
  // CHECK: mul <vscale x 4 x i32>
  %0 = llvm.mul %arg0, %arg1 : vector<[4]xi32>
  // CHECK: add <vscale x 4 x i32>
  %1 = llvm.add %0, %arg2 : vector<[4]xi32>
  llvm.return %1 : vector<[4]xi32>
}

// CHECK-LABEL: define <vscale x 4 x float> @arm_sve_arithf
llvm.func @arm_sve_arithf(%arg0: vector<[4]xf32>,
                          %arg1: vector<[4]xf32>,
                          %arg2: vector<[4]xf32>)
                          -> vector<[4]xf32> {
  // CHECK: fmul <vscale x 4 x float>
  %0 = llvm.fmul %arg0, %arg1 : vector<[4]xf32>
  // CHECK: fadd <vscale x 4 x float>
  %1 = llvm.fadd %0, %arg2 : vector<[4]xf32>
  llvm.return %1 : vector<[4]xf32>
}

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

// CHECK-LABEL: define <vscale x 4 x i1> @arm_sve_mask_genf
llvm.func @arm_sve_mask_genf(%arg0: vector<[4]xf32>,
                             %arg1: vector<[4]xf32>)
                             -> vector<[4]xi1> {
  // CHECK: fcmp oeq <vscale x 4 x float>
  %0 = llvm.fcmp "oeq" %arg0, %arg1 : vector<[4]xf32>
  llvm.return %0 : vector<[4]xi1>
}

// CHECK-LABEL: define <vscale x 4 x i1> @arm_sve_mask_geni
llvm.func @arm_sve_mask_geni(%arg0: vector<[4]xi32>,
                             %arg1: vector<[4]xi32>)
                             -> vector<[4]xi1> {
  // CHECK: icmp uge <vscale x 4 x i32>
  %0 = llvm.icmp "uge" %arg0, %arg1 : vector<[4]xi32>
  llvm.return %0 : vector<[4]xi1>
}

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

// CHECK-LABEL: define void @memcopy
llvm.func @memcopy(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>,
                   %arg2: i64, %arg3: i64, %arg4: i64,
                   %arg5: !llvm.ptr<f32>, %arg6: !llvm.ptr<f32>,
                   %arg7: i64, %arg8: i64, %arg9: i64,
                   %arg10: i64) {
  %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                       array<1 x i64>, array<1 x i64>)>
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                     array<1 x i64>,
                                                     array<1 x i64>)>
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                     array<1 x i64>,
                                                     array<1 x i64>)>
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                     array<1 x i64>,
                                                     array<1 x i64>)>
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                        array<1 x i64>,
                                                        array<1 x i64>)>
  %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                        array<1 x i64>,
                                                        array<1 x i64>)>
  %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                       array<1 x i64>,
                                       array<1 x i64>)>
  %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                     array<1 x i64>,
                                                     array<1 x i64>)>
  %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                     array<1 x i64>,
                                                     array<1 x i64>)>
  %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                     array<1 x i64>,
                                                     array<1 x i64>)>
  %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                         array<1 x i64>,
                                                         array<1 x i64>)>
  %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                         array<1 x i64>,
                                                         array<1 x i64>)>
  %12 = llvm.mlir.constant(0 : index) : i64
  %13 = llvm.mlir.constant(4 : index) : i64
  // CHECK: [[VL:%[0-9]+]] = call i64 @llvm.vscale.i64()
  %14 = "llvm.intr.vscale"() : () -> i64
  // CHECK: mul i64 [[VL]], 4
  %15 = llvm.mul %14, %13  : i64
  llvm.br ^bb1(%12 : i64)
^bb1(%16: i64):
  %17 = llvm.icmp "slt" %16, %arg10 : i64
  llvm.cond_br %17, ^bb2, ^bb3
^bb2:
  // CHECK: extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] }
  %18 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                array<1 x i64>,
                                                array<1 x i64>)>
  // CHECK: getelementptr float, ptr
  %19 = llvm.getelementptr %18[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %20 = llvm.bitcast %19 : !llvm.ptr<f32> to !llvm.ptr<vector<[4]xf32>>
  // CHECK: load <vscale x 4 x float>, ptr
  %21 = llvm.load %20 : !llvm.ptr<vector<[4]xf32>>
  // CHECK: extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] }
  %22 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64,
                                                 array<1 x i64>,
                                                 array<1 x i64>)>
  // CHECK: getelementptr float, ptr
  %23 = llvm.getelementptr %22[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %24 = llvm.bitcast %23 : !llvm.ptr<f32> to !llvm.ptr<vector<[4]xf32>>
  // CHECK: store <vscale x 4 x float> %{{[0-9]+}}, ptr %{{[0-9]+}}
  llvm.store %21, %24 : !llvm.ptr<vector<[4]xf32>>
  %25 = llvm.add %16, %15  : i64
  llvm.br ^bb1(%25 : i64)
^bb3:
  llvm.return
}

// CHECK-LABEL: define i64 @get_vector_scale()
llvm.func @get_vector_scale() -> i64 {
  // CHECK: call i64 @llvm.vscale.i64()
  %0 = "llvm.intr.vscale"() : () -> i64
  llvm.return %0 : i64
}

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
