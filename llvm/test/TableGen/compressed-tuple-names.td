// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s 2>&1 | FileCheck %s

// Verify that the `CompressedTupleNameStride` option produces the expected
// compact synthesized tuple names rather than the default names.

include "reg-with-subregs-common.td"

class getSubRegs<int Size> {
  list<SubRegIndex> Ret =
      !foreach(X, !range(Size), !cast<SubRegIndex>("sub"#X));
}

// A 64-wide tuple with no alignment constraints.
let CompressedTupleNameStride = 1 in
def Tuples64X_S1 : RegisterTuples<
    getSubRegs<64>.Ret, !foreach(X, !range(64), (decimate (shl GPR32, X), 1))>;

// A 5-wide tuple with 8-alignment.
let CompressedTupleNameStride = 2 in
def Tuples5X_S2 : RegisterTuples<
    getSubRegs<5>.Ret, !foreach(X, !range(5), (decimate (shl GPR32, X), 8))>;


def GPR2048 : RegisterClass<"", [v64i32], 2048, (add Tuples64X_S1)>;
def GPR160 : RegisterClass<"", [v5i32], 160, (add Tuples5X_S2)>;


//  CHECK: enum : unsigned {
//  CHECK-NEXT:   NoRegister,
//
//  CHECK:        R0_R1_R2_R3_R4_R5_R6_R7_R8_R9_R10_R11_R12_R13_R14_R15_R16_R17_R18_R19_R20_R21_R22_R23_R24_R25_R26_R27_R28_R29_R30_R31 = 1255,
//  CHECK-NEXT:   R1_R2_R3_R4_R5_R6_R7_R8_R9_R10_R11_R12_R13_R14_R15_R16_R17_R18_R19_R20_R21_R22_R23_R24_R25_R26_R27_R28_R29_R30_R31_R32 = 1256,
//
//  CHECK:        R0_TO_R4_BY_2 = 1480,
//  CHECK-NEXT:   R8_TO_R12_BY_2 = 1481,
//  CHECK-NEXT:   R16_TO_R20_BY_2 = 1482,
//  CHECK-NEXT:   R24_TO_R28_BY_2 = 1483,
//  CHECK-NEXT:   R32_TO_R36_BY_2 = 1484,
//  CHECK-NEXT:   R40_TO_R44_BY_2 = 1485,
//  CHECK-NEXT:   R48_TO_R52_BY_2 = 1486,
//
//  CHECK:        R0_TO_R63_BY_1 = 1512,
//  CHECK-NEXT:   R1_TO_R64_BY_1 = 1513,
//  CHECK-NEXT:   R2_TO_R65_BY_1 = 1514,
//  CHECK-NEXT:   R3_TO_R66_BY_1 = 1515,
//  CHECK-NEXT:   R4_TO_R67_BY_1 = 1516,
//  CHECK-NEXT:   R5_TO_R68_BY_1 = 1517,
//  CHECK-NEXT:   R6_TO_R69_BY_1 = 1518,
//
//  CHECK:        NUM_TARGET_REGS
//  CHECK-NEXT: };
//
//  CHECK:        enum {
//  CHECK-NEXT:   GPR32RegClassID = 0,
//  CHECK-NEXT:   GPR_64RegClassID = 1,
//  CHECK-NEXT:   GPR160RegClassID = 2,
//  CHECK-NEXT:   GPR_1024RegClassID = 3,
//  CHECK-NEXT:   GPR_1024_with_sub0_sub1_sub2_sub3_sub4RegClassID = 4,
//  CHECK-NEXT:   GPR_1024_with_sub1_sub2_sub3_sub4_sub5RegClassID = 5,
//  CHECK-NEXT:   GPR_1024_with_sub2_sub3_sub4_sub5_sub6RegClassID = 6,
//  CHECK-NEXT:   GPR_1024_with_sub3_sub4_sub5_sub6_sub7RegClassID = 7,
//  CHECK-NEXT:   GPR_1024_with_sub4_sub5_sub6_sub7_sub8RegClassID = 8,
//  CHECK-NEXT:   GPR_1024_with_sub5_sub6_sub7_sub8_sub9RegClassID = 9,
//  CHECK-NEXT:   GPR_1024_with_sub6_sub7_sub8_sub9_sub10RegClassID = 10,
//  CHECK-NEXT:   GPR_1024_with_sub7_sub8_sub9_sub10_sub11RegClassID = 11,
//  CHECK-NEXT:   GPR2048RegClassID = 12,
//  CHECK-NEXT:   GPR2048_with_sub0_sub1_sub2_sub3_sub4RegClassID = 13,
//  CHECK-NEXT:   GPR2048_with_sub1_sub2_sub3_sub4_sub5RegClassID = 14,
//  CHECK-NEXT:   GPR2048_with_sub2_sub3_sub4_sub5_sub6RegClassID = 15,
//  CHECK-NEXT:   GPR2048_with_sub3_sub4_sub5_sub6_sub7RegClassID = 16,
//  CHECK-NEXT:   GPR2048_with_sub4_sub5_sub6_sub7_sub8RegClassID = 17,
//  CHECK-NEXT:   GPR2048_with_sub5_sub6_sub7_sub8_sub9RegClassID = 18,
//  CHECK-NEXT:   GPR2048_with_sub6_sub7_sub8_sub9_sub10RegClassID = 19,
//  CHECK-NEXT:   GPR2048_with_sub7_sub8_sub9_sub10_sub11RegClassID = 20,
//
//  CHECK:      extern const char TestTargetRegStrings[] = {
//  CHECK:   /* {{[0-9]+}} */ "R0_R1_R2_R3_R4_R5_R6_R7_R8_R9_R10_R11_R12_R13_R14_R15_R16_R17_R18_R19_R20_R21_R22_R23_R24_R25_R26_R27_R28_R29_R30_R31\000"
//  CHECK:        /* {{[0-9]+}} */ "R0_TO_R63_BY_1\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R110_TO_R173_BY_1\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R10_TO_R73_BY_1\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R120_TO_R183_BY_1\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R20_TO_R83_BY_1\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R130_TO_R193_BY_1\000"
//  CHECK:        /* {{[0-9]+}} */ "R0_TO_R4_BY_2\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R112_TO_R116_BY_2\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R232_TO_R236_BY_2\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R32_TO_R36_BY_2\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R152_TO_R156_BY_2\000"
//  CHECK-NEXT:   /* {{[0-9]+}} */ "R72_TO_R76_BY_2\000"
//
//  CHECK:         // GPR160 Register Class...
//  CHECK-NEXT:   const MCPhysReg GPR160[] = {
//  CHECK-NEXT:     R0_TO_R4_BY_2, R8_TO_R12_BY_2, R16_TO_R20_BY_2, R24_TO_R28_BY_2, R32_TO_R36_BY_2, R40_TO_R44_BY_2, R48_TO_R52_BY_2, R56_TO_R60_BY_2, R64_TO_R68_BY_2, R72_TO_R76_BY_2, R80_TO_R84_BY_2, R88_TO_R92_BY_2, R96_TO_R100_BY_2, R104_TO_R108_BY_2, R112_TO_R116_BY_2, R120_TO_R124_BY_2, R128_TO_R132_BY_2, R136_TO_R140_BY_2, R144_TO_R148_BY_2, R152_TO_R156_BY_2, R160_TO_R164_BY_2, R168_TO_R172_BY_2, R176_TO_R180_BY_2, R184_TO_R188_BY_2, R192_TO_R196_BY_2, R200_TO_R204_BY_2, R208_TO_R212_BY_2, R216_TO_R220_BY_2, R224_TO_R228_BY_2, R232_TO_R236_BY_2, R240_TO_R244_BY_2, R248_TO_R252_BY_2,
