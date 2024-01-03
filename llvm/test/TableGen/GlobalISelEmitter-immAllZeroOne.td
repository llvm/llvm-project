// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -optimize-match-table=false -I %p/../../include -I %p/Common %s -o - | FileCheck -check-prefixes=GISEL-NOOPT %s
// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -optimize-match-table=true -I %p/../../include -I %p/Common %s -o - | FileCheck -check-prefixes=GISEL-OPT %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// GISEL-OPT: GIM_SwitchOpcode
// GISEL-OPT-NEXT: /*TargetOpcode::G_SHL*/
// GISEL-OPT-NEXT: /*TargetOpcode::G_LSHR*/
// GISEL-OPT-NEXT: // Label


// GISEL-OPT: GIM_Try,
// GISEL-OPT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s16,
// GISEL-OPT: GIM_CheckOpcodeIsEither, /*MI*/1, GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR), GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR_TRUNC),
// GISEL-OPT: GIM_CheckIsBuildVectorAllZeros, /*MI*/1,

// GISEL-OPT: GIM_Try,
// GISEL-OPT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s16,
// GISEL-OPT: GIM_CheckOpcodeIsEither, /*MI*/1, GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR), GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR_TRUNC),
// GISEL-OPT: GIM_CheckIsBuildVectorAllOnes, /*MI*/1,



// GISEL-NOOPT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_LSHR),
// GISEL-NOOPT: // MIs[0] Operand 2
// GISEL-NOOPT-NEXT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s16,
// GISEL-NOOPT-NEXT: GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// GISEL-NOOPT-NEXT: GIM_CheckOpcodeIsEither, /*MI*/1, GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR), GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR_TRUNC),
// GISEL-NOOPT-NEXT: GIM_CheckIsBuildVectorAllOnes, /*MI*/1,
// GISEL-NOOPT-NEXT: // MIs[1] Operand 0
// GISEL-NOOPT-NEXT: GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_v4s16,
// GISEL-NOOPT-NEXT: GIM_CheckIsSafeToFold, /*InsnID*/1,
// GISEL-NOOPT-NEXT: // (srl:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$src0, immAllOnesV:{ *:[v4i16] })  =>  (VFOOONES:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$src0)
def VFOOONES : I<(outs VecReg128:$dst), (ins VecReg128:$src0),
  [(set v4i32:$dst, (srl v4i32:$src0, (v4i16 immAllOnesV)))]
>;


// GISEL-NOOPT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_SHL),
// GISEL-NOOPT: // MIs[0] Operand 2
// GISEL-NOOPT-NEXT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s16,
// GISEL-NOOPT-NEXT: GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// GISEL-NOOPT-NEXT: GIM_CheckOpcodeIsEither, /*MI*/1, GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR), GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR_TRUNC),
// GISEL-NOOPT-NEXT: GIM_CheckIsBuildVectorAllZeros, /*MI*/1,
// GISEL-NOOPT-NEXT: // MIs[1] Operand 0
// GISEL-NOOPT-NEXT: GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_v4s16,
// GISEL-NOOPT-NEXT: GIM_CheckIsSafeToFold, /*InsnID*/1,
// GISEL-NOOPT-NEXT: // (shl:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$src0, immAllZerosV:{ *:[v4i16] })  =>  (VFOOZERO:{ *:[v4i32] } v4i32:{ *:[v4i32] }:$src0)
def VFOOZERO : I<(outs VecReg128:$dst), (ins VecReg128:$src0),
  [(set v4i32:$dst, (shl v4i32:$src0, (v4i16 immAllZerosV)))]
>;
