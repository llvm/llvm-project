// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common -optimize-match-table=false %s -o %T/context-non-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK_NOPT -input-file=%T/context-non-optimized.cpp
// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common -optimize-match-table=true  %s -o %T/context-optimized.cpp
// RUN: FileCheck %s --check-prefixes=CHECK_OPT -input-file=%T/context-optimized.cpp



include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def test_atomic_op_frag : PatFrag<(ops node:$ptr, node:$val),
	                                             (atomic_swap node:$ptr, node:$val)> {
  let GISelPredicateCode = [{ return !MRI.use_nodbg_empty(MI.getOperand(0).getReg()); }];
  let IsAtomic = 1;
  let MemoryVT = i32;
}

def INSN : I<(outs GPR32:$dst), (ins GPR32Op:$src1, GPR32Op:$src2), []>;

def : Pat<(test_atomic_op_frag GPR32:$ptr, GPR32:$val) ,
	      (INSN GPR32:$ptr, GPR32:$val)>;

// CHECK_NOPT-LABEL: const uint8_t *MyTargetInstructionSelector::getMatchTable() const {
// CHECK_NOPT-NEXT:    constexpr static uint8_t MatchTable0[] = {
// CHECK_NOPT-NEXT:      GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(52), // Rule ID 0 //
// CHECK_NOPT-NEXT:        GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK_NOPT-NEXT:        GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ATOMICRMW_XCHG),
// CHECK_NOPT-NEXT:        GIM_CheckMemorySizeEqualTo, /*MI*/0, /*MMO*/0, /*Size*/GIMT_Encode4(4),
// CHECK_NOPT-NEXT:        // MIs[0] DstI[dst]
// CHECK_NOPT-NEXT:        GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK_NOPT-NEXT:        GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK_NOPT-NEXT:        // MIs[0] ptr
// CHECK_NOPT-NEXT:        GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK_NOPT-NEXT:        GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK_NOPT-NEXT:        // MIs[0] val
// CHECK_NOPT-NEXT:        GIM_RootCheckType, /*Op*/2, /*Type*/GILLT_s32,
// CHECK_NOPT-NEXT:        GIM_RootCheckRegBankForClass, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK_NOPT-NEXT:        GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_test_atomic_op_frag),
// CHECK_NOPT-NEXT:        // (atomic_swap:{ *:[i32] } GPR32:{ *:[i32] }:$ptr, GPR32:{ *:[i32] }:$val)<<P:Predicate_test_atomic_op_frag>>  =>  (INSN:{ *:[i32] } GPR32:{ *:[i32] }:$ptr, GPR32:{ *:[i32] }:$val)
// CHECK_NOPT-NEXT:        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::INSN),
// CHECK_NOPT-NEXT:        GIR_RootConstrainSelectedInstOperands,
// CHECK_NOPT-NEXT:        // GIR_Coverage, 0,
// CHECK_NOPT-NEXT:        GIR_Done,
// CHECK_NOPT-NEXT:      // Label 0: @52
// CHECK_NOPT-NEXT:      GIM_Reject,
// CHECK_NOPT-NEXT:      };
// CHECK_NOPT-NEXT:    return MatchTable0;
// CHECK_NOPT-NEXT:  }

// CHECK_OPT-LABEL: const uint8_t *MyTargetInstructionSelector::getMatchTable() const {
// CHECK_OPT-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK_OPT-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(49), // Rule ID 0 //
// CHECK_OPT-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ATOMICRMW_XCHG),
// CHECK_OPT-NEXT:       GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// CHECK_OPT-NEXT:       GIM_RootCheckType, /*Op*/2, /*Type*/GILLT_s32,
// CHECK_OPT-NEXT:       GIM_CheckMemorySizeEqualTo, /*MI*/0, /*MMO*/0, /*Size*/GIMT_Encode4(4),
// CHECK_OPT-NEXT:       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK_OPT-NEXT:       // MIs[0] ptr
// CHECK_OPT-NEXT:       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// CHECK_OPT-NEXT:       GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK_OPT-NEXT:       GIM_RootCheckRegBankForClass, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK_OPT-NEXT:       GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_test_atomic_op_frag),
// CHECK_OPT-NEXT:       // (atomic_swap:{ *:[i32] } GPR32:{ *:[i32] }:$ptr, GPR32:{ *:[i32] }:$val)<<P:Predicate_test_atomic_op_frag>>  =>  (INSN:{ *:[i32] } GPR32:{ *:[i32] }:$ptr, GPR32:{ *:[i32] }:$val)
// CHECK_OPT-NEXT:       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::INSN),
// CHECK_OPT-NEXT:       GIR_RootConstrainSelectedInstOperands,
// CHECK_OPT-NEXT:       // GIR_Coverage, 0,
// CHECK_OPT-NEXT:       GIR_Done,
// CHECK_OPT-NEXT:     // Label 0: @49
// CHECK_OPT-NEXT:     GIM_Reject,
// CHECK_OPT-NEXT:     };
// CHECK_OPT-NEXT:   return MatchTable0;
// CHECK_OPT-NEXT:  }
