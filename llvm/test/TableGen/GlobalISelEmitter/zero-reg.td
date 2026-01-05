// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../../include -I %p/../Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def P0 : Register<"p0"> { let Namespace = "MyTarget"; }
def PR32 : RegisterClass<"MyTarget", [i32], 32, (add P0)>;
def PR32Op : RegisterOperand<PR32>;

def pred : PredicateOperand<OtherVT,
                            (ops PR32:$FR),
                            (ops (i32 zero_reg))> {}
class PredI<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = !con(IOps, (ins pred:$pred));
  let Pattern = Pat;
}

def INST : PredI<(outs GPR32:$dst), (ins GPR32:$src), []>;

// CHECK: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_LOAD),
// CHECK-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT: // MIs[0] DstI[dst]
// CHECK-NEXT: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // (ld:{ *:[i32] } GPR32:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (INST:{ *:[i32] } GPR32:{ *:[i32] }:$src)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::INST),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/1, // src
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/0, GIMT_Encode2(MyTarget::NoRegister), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/1, /*MergeInsnID's*/0,
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
def : Pat<(i32 (load GPR32:$src)),
          (INST GPR32:$src)>;
