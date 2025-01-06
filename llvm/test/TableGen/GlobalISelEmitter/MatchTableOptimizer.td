// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=true -I %p/../../../include -I %p/../Common | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Two LOADs with same output size but different input size, hence their
// GIM_CheckPointerToAny should *not* be merged
def LOAD8 : I<(outs GPR8:$dst), (ins GPR8:$src), []>;
def LOAD32 : I<(outs GPR8:$dst), (ins GPR32:$src), []>;
// CHECK: Label 1: @{{[0-9]+}}
// CHECK-NEXT: GIM_Try, /*On fail goto*//*Label [[L1_ID:[0-9]+]]*/ GIMT_Encode4([[L1_AT:[0-9]+]]),
// CHECK-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:   GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR8RegClassID),
// CHECK-NEXT: GIM_Try, /*On fail goto*//*Label [[L2_ID:[0-9]+]]*/ GIMT_Encode4([[L2_AT:[0-9]+]]),
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR8RegClassID),
// CHECK-NEXT: // (ld:{ *:[i8] } GPR8:{ *:[i8] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8:{ *:[i8] } GPR8:{ *:[i8] }:$src)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 0,
// CHECK-NEXT: GIR_Done,
// CHECK-NEXT: // Label [[L2_ID]]: @[[L2_AT]]
// CHECK-NEXT: GIM_Try, /*On fail goto*//*Label [[L3_ID:[0-9]+]]*/ GIMT_Encode4([[L3_AT:[0-9]+]]),
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // (ld:{ *:[i8] } GPR32:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD32:{ *:[i8] } GPR32:{ *:[i32] }:$src)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD32),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 1,
// CHECK-NEXT: GIR_Done,
// CHECK-NEXT: // Label [[L3_ID]]: @[[L3_AT]]
// CHECK-NEXT: GIM_Reject,
// CHECK-NEXT: // Label [[L1_ID]]: @[[L1_AT]]
def : Pat<(i8 (load GPR8:$src)),
          (LOAD8 GPR8:$src)>;
def : Pat<(i8 (load GPR32:$src)),
          (LOAD32 GPR32:$src)>;

// Two LOADs with same output size and input size, hence their
// GIM_CheckPointerToAny *should* be merged
def S0 : Register<"s0"> { let Namespace = "MyTarget"; }
def GPR16 : RegisterClass<"MyTarget", [i16], 16, (add S0)>;
def LOAD16 : I<(outs GPR16:$dst), (ins GPR16:$src), []>;
def LOAD16Imm : I<(outs GPR16:$dst), (ins GPR16:$src), []>;
// CHECK: // Label 2: @{{[0-9]+}}
// CHECK-NEXT: GIM_Try, /*On fail goto*//*Label [[L1_ID:[0-9]+]]*/ GIMT_Encode4([[L1_AT:[0-9]+]]),
// CHECK-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT: GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR16RegClassID),
// CHECK-NEXT: GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT: GIM_Try, /*On fail goto*//*Label [[L2_ID:[0-9]+]]*/ GIMT_Encode4([[L2_AT:[0-9]+]]),
// CHECK-NEXT: GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT: GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s16,
// CHECK-NEXT: GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s16,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR16RegClassID),
// CHECK-NEXT: GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT: GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT: // (ld:{ *:[i16] } (add:{ *:[i16] } GPR16:{ *:[i16] }:$src, 10:{ *:[i16] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16Imm:{ *:[i16] } GPR16:{ *:[i16] }:$src)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16Imm),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT: GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 3,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
// CHECK-NEXT: // Label [[L2_ID]]: @[[L2_AT]]
// CHECK-NEXT: GIM_Try, /*On fail goto*//*Label [[L3_ID:[0-9]+]]*/ GIMT_Encode4([[L3_AT:[0-9]+]]),
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR16RegClassID),
// CHECK-NEXT: // (ld:{ *:[i16] } GPR16:{ *:[i16] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16:{ *:[i16] } GPR16:{ *:[i16] }:$src)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 2,
// CHECK-NEXT: GIR_Done,
// CHECK-NEXT: // Label [[L3_ID]]: @[[L3_AT]]
// CHECK-NEXT: GIM_Reject,
// CHECK-NEXT: // Label [[L1_ID]]: @[[L1_AT]]
def : Pat<(i16 (load GPR16:$src)),
          (LOAD16 GPR16:$src)>;
def : Pat<(i16 (load (add GPR16:$src, 10))),
          (LOAD16Imm GPR16:$src)>;
