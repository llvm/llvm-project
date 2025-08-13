// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../../include -I %p/../Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def GPR : RegisterClass<"MyTarget", [i64], 64, (add R0)>;
def ANDI : I<(outs GPR:$dst), (ins GPR:$src1, i64imm:$src2), []>;

// CHECK-LABEL: GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(59), // Rule ID 0 //
// CHECK-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT: // MIs[0] DstI[dst]
// CHECK-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s64,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT: // MIs[0] rs1
// CHECK-NEXT: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s64,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPRRegClassID),
// CHECK-NEXT: // MIs[0] Operand 2
// CHECK-NEXT: GIM_RootCheckType, /*Op*/2, /*Type*/GILLT_s64,
// CHECK-NEXT: GIM_CheckConstantInt, /*MI*/0, /*Op*/2, GIMT_Encode8(9223372036854775808),
// CHECK-NEXT: // (and:{ *:[i64] } GPR:{ *:[i64] }:$rs1, -9223372036854775808:{ *:[i64] })  =>  (ANDI:{ *:[i64] } GPR:{ *:[i64] }:$rs1, -9223372036854775808:{ *:[i64] })
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::ANDI),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/1, // rs1
// CHECK-NEXT: GIR_AddImm, /*InsnID*/0, /*Imm*/GIMT_Encode8(9223372036854775808),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
def : Pat<(and GPR:$rs1, 0x8000000000000000),
          (ANDI GPR:$rs1, 0x8000000000000000)>;
