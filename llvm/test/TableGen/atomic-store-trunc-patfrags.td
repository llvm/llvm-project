// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common %s | FileCheck -check-prefix=SDAG %s
// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -I %p/../../include -I %p/Common %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def atomic_store_nontrunc : PatFrag<(ops node:$val, node:$ptr),
                                    (atomic_store node:$val, node:$ptr)> {
  let IsAtomic = 1;
  let IsTruncStore = 0;
}

def atomic_store_trunc : PatFrag<(ops node:$val, node:$ptr),
                                 (atomic_store node:$val, node:$ptr)> {
  let IsAtomic = 1;
  let IsTruncStore = 1;
}

def STORE : I<(outs), (ins GPR32:$val, GPR32:$ptr), []>;
def TRUNCSTORE : I<(outs), (ins GPR32:$val, GPR32:$ptr), []>;

// SDAG-LABEL: CheckNodePredicate(SDValue Op, unsigned PredNo) const
// SDAG: // Predicate_atomic_store_nontrunc
// SDAG: if (cast<AtomicSDNode>(N)->getMemoryVT().getSizeInBits() != cast<AtomicSDNode>(N)->getVal().getValueSizeInBits()) return false;
// SDAG: // Predicate_atomic_store_trunc
// SDAG: if (cast<AtomicSDNode>(N)->getMemoryVT().getSizeInBits() == cast<AtomicSDNode>(N)->getVal().getValueSizeInBits()) return false;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_STORE),
// GISEL-NEXT: GIM_CheckAtomicOrderingOrStrongerThan, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::Unordered,
// GISEL-NEXT: GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// GISEL: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::STORE),
def : Pat<(atomic_store_nontrunc i32:$val, i32:$ptr),
          (STORE GPR32:$val, GPR32:$ptr)>;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_STORE),
// GISEL-NEXT: GIM_CheckAtomicOrderingOrStrongerThan, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::Unordered,
// GISEL-NEXT: GIM_CheckMemorySizeLessThanLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// GISEL: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::TRUNCSTORE),
def : Pat<(atomic_store_trunc i32:$val, i32:$ptr),
          (TRUNCSTORE GPR32:$val, GPR32:$ptr)>;
