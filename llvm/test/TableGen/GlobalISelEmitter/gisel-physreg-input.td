// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../../include %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def SPECIAL : Register<"special"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
def Special32 : RegisterClass<"MyTarget", [i32], 32, (add SPECIAL)>;


class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}

// Try a nested physical register

// GISEL: GIM_Try,
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_STORE),
// GISEL-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// GISEL-NEXT: // MIs[0] src0
// GISEL-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // MIs[0] Operand 1
// GISEL-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// GISEL-NEXT: GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_MUL),
// GISEL-NEXT: // MIs[1] Operand 0
// GISEL-NEXT: GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: // MIs[1] src1
// GISEL-NEXT: GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // MIs[1] Operand 2
// GISEL-NEXT: GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::Special32RegClassID),
// GISEL-NEXT: GIM_CheckIsSafeToFold, /*NumInsns*/1,
// GISEL-NEXT: // (st GPR32:{ *:[i32] }:$src0, (mul:{ *:[i32] } GPR32:{ *:[i32] }:$src1, SPECIAL:{ *:[i32] }))  =>  (MULM_PHYS GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// GISEL-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::SPECIAL), /*AddRegisterRegFlags*/GIMT_Encode2(RegState::Define),
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/2, // SPECIAL
// GISEL-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::MULM_PHYS),
// GISEL-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // src0
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
// GISEL-NEXT: GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// GISEL-NEXT: GIR_RootConstrainSelectedInstOperands,
// GISEL-NEXT: // GIR_Coverage, 0,
// GISEL-NEXT: GIR_EraseRootFromParent_Done,
def MULM_PHYS : I<(outs), (ins GPR32:$src0, GPR32:$src1),
    [(st GPR32:$src0, (mul GPR32:$src1, SPECIAL))]> {
  let Uses = [SPECIAL];
}

// Try nested physical registers and check on duplicated copies

// GISEL: GIM_Try,
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_STORE),
// GISEL-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// GISEL-NEXT: // MIs[0] src0
// GISEL-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // MIs[0] Operand 1
// GISEL-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// GISEL-NEXT: GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_MUL),
// GISEL-NEXT: // MIs[1] Operand 0
// GISEL-NEXT: GIM_CheckType, /*MI*/1, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: // MIs[1] Operand 1
// GISEL-NEXT: GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // MIs[1] Operand 2
// GISEL-NEXT: GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::Special32RegClassID),
// GISEL-NEXT: GIM_CheckIsSafeToFold, /*NumInsns*/1,
// GISEL-NEXT: // (st GPR32:{ *:[i32] }:$src0, (mul:{ *:[i32] } R0:{ *:[i32] }, SPECIAL:{ *:[i32] }))  =>  (MULMR0_PHYS GPR32:{ *:[i32] }:$src0)
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/2, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// GISEL-NEXT: GIR_AddRegister, /*InsnID*/2, GIMT_Encode2(MyTarget::SPECIAL), /*AddRegisterRegFlags*/GIMT_Encode2(RegState::Define),
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/2, // SPECIAL
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// GISEL-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::R0), /*AddRegisterRegFlags*/GIMT_Encode2(RegState::Define),
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/1, // R0
// GISEL-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::MULMR0_PHYS),
// GISEL-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // src0
// GISEL-NEXT: GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// GISEL-NEXT: GIR_RootConstrainSelectedInstOperands,
// GISEL-NEXT: // GIR_Coverage, 1,
// GISEL-NEXT: GIR_EraseRootFromParent_Done,
def MULMR0_PHYS : I<(outs), (ins GPR32:$src0),
    [(st GPR32:$src0, (mul R0, SPECIAL))]> {
  let Uses = [R0, SPECIAL];
}

// Try a normal physical register use.

// GISEL: GIM_Try,
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ADD),
// GISEL-NEXT: // MIs[0] DstI[dst]
// GISEL-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // MIs[0] src0
// GISEL-NEXT: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // MIs[0] Operand 2
// GISEL-NEXT: GIM_RootCheckType, /*Op*/2, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_RootCheckRegBankForClass, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::Special32RegClassID),
// GISEL-NEXT: // (add:{ *:[i32] } GPR32:{ *:[i32] }:$src0, SPECIAL:{ *:[i32] })  =>  (ADD_PHYS:{ *:[i32] } GPR32:{ *:[i32] }:$src0)
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// GISEL-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::SPECIAL), /*AddRegisterRegFlags*/GIMT_Encode2(RegState::Define),
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/2, // SPECIAL
// GISEL-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::ADD_PHYS),
// GISEL-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// GISEL-NEXT: GIR_RootToRootCopy, /*OpIdx*/1, // src0
// GISEL-NEXT: GIR_RootConstrainSelectedInstOperands,
// GISEL-NEXT: // GIR_Coverage, 2,
// GISEL-NEXT: GIR_EraseRootFromParent_Done,
def ADD_PHYS : I<(outs GPR32:$dst), (ins GPR32:$src0),
    [(set GPR32:$dst, (add GPR32:$src0, SPECIAL))]> {
  let Uses = [SPECIAL];
}

// Try using the name of the physreg in another operand.

// GISEL: GIM_Try,
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// GISEL-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_MUL),
// GISEL-NEXT: // MIs[0] DstI[dst]
// GISEL-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // MIs[0] SPECIAL
// GISEL-NEXT: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // MIs[0] Operand 2
// GISEL-NEXT: GIM_RootCheckType, /*Op*/2, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_RootCheckRegBankForClass, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::Special32RegClassID),
// GISEL-NEXT: // (mul:{ *:[i32] } GPR32:{ *:[i32] }:$SPECIAL, SPECIAL:{ *:[i32] })  =>  (MUL_PHYS:{ *:[i32] } GPR32:{ *:[i32] }:$SPECIAL)
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// GISEL-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::SPECIAL), /*AddRegisterRegFlags*/GIMT_Encode2(RegState::Define),
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/2, // SPECIAL
// GISEL-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::MUL_PHYS),
// GISEL-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// GISEL-NEXT: GIR_RootToRootCopy, /*OpIdx*/1, // SPECIAL
// GISEL-NEXT: GIR_RootConstrainSelectedInstOperands,
// GISEL-NEXT: // GIR_Coverage, 3,
// GISEL-NEXT: GIR_EraseRootFromParent_Done,
def MUL_PHYS : I<(outs GPR32:$dst), (ins GPR32:$SPECIAL),
    [(set GPR32:$dst, (mul GPR32:$SPECIAL, SPECIAL))]> {
  let Uses = [SPECIAL];
}

// Try giving the physical operand a name
// def ADD_PHYS : I<(outs GPR32:$dst), (ins GPR32:$src0),
//     [(set GPR32:$dst, (add GPR32:$src0, SPECIAL:$special))]> {
//   let Uses = [SPECIAL];
// }
