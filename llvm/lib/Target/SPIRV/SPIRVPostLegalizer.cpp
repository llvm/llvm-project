//===-- SPIRVPostLegalizer.cpp - amend info after legalization -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass partially applies pre-legalization logic to new instructions
// inserted as a result of legalization:
// - assigns SPIR-V types to registers for new instructions.
// - inserts ASSIGN_TYPE pseudo-instructions required for type folding.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Support/Debug.h"
#include <stack>

#define DEBUG_TYPE "spirv-postlegalizer"

using namespace llvm;

namespace {
class SPIRVPostLegalizer : public MachineFunctionPass {
public:
  static char ID;
  SPIRVPostLegalizer() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

namespace llvm {
//  Defined in SPIRVPreLegalizer.cpp.
extern void updateRegType(Register Reg, Type *Ty, SPIRVType *SpirvTy,
                          SPIRVGlobalRegistry *GR, MachineIRBuilder &MIB,
                          MachineRegisterInfo &MRI);
extern void processInstr(MachineInstr &MI, MachineIRBuilder &MIB,
                         MachineRegisterInfo &MRI, SPIRVGlobalRegistry *GR,
                         SPIRVType *KnownResType);
} // namespace llvm

static SPIRVType *deduceIntTypeFromResult(Register ResVReg,
                                          MachineIRBuilder &MIB,
                                          SPIRVGlobalRegistry *GR) {
  const LLT &Ty = MIB.getMRI()->getType(ResVReg);
  return GR->getOrCreateSPIRVIntegerType(Ty.getScalarSizeInBits(), MIB);
}

static SPIRVType *deduceTypeFromSingleOperand(MachineInstr *I,
                                              MachineIRBuilder &MIB,
                                              SPIRVGlobalRegistry *GR,
                                              unsigned OpIdx) {
  Register OpReg = I->getOperand(OpIdx).getReg();
  if (SPIRVType *OpType = GR->getSPIRVTypeForVReg(OpReg)) {
    if (SPIRVType *CompType = GR->getScalarOrVectorComponentType(OpType)) {
      Register ResVReg = I->getOperand(0).getReg();
      const LLT &ResLLT = MIB.getMRI()->getType(ResVReg);
      if (ResLLT.isVector())
        return GR->getOrCreateSPIRVVectorType(CompType, ResLLT.getNumElements(),
                                              MIB, false);
      return CompType;
    }
  }
  return nullptr;
}

static SPIRVType *deduceTypeFromOperandRange(MachineInstr *I,
                                             MachineIRBuilder &MIB,
                                             SPIRVGlobalRegistry *GR,
                                             unsigned StartOp, unsigned EndOp) {
  SPIRVType *ResType = nullptr;
  for (unsigned i = StartOp; i < EndOp; ++i) {
    if (SPIRVType *Type = deduceTypeFromSingleOperand(I, MIB, GR, i)) {
#ifdef EXPENSIVE_CHECKS
      assert(!ResType || Type == ResType && "Conflicting type from operands.");
      ResType = Type;
#else
      return Type;
#endif
    }
  }
  return ResType;
}

static SPIRVType *deduceTypeFromResultRegister(MachineInstr *Use,
                                               Register UseRegister,
                                               SPIRVGlobalRegistry *GR,
                                               MachineIRBuilder &MIB) {
  for (const MachineOperand &MO : Use->defs()) {
    if (!MO.isReg())
      continue;
    if (SPIRVType *OpType = GR->getSPIRVTypeForVReg(MO.getReg())) {
      if (SPIRVType *CompType = GR->getScalarOrVectorComponentType(OpType)) {
        const LLT &ResLLT = MIB.getMRI()->getType(UseRegister);
        if (ResLLT.isVector())
          return GR->getOrCreateSPIRVVectorType(
              CompType, ResLLT.getNumElements(), MIB, false);
        return CompType;
      }
    }
  }
  return nullptr;
}

static SPIRVType *deducePointerTypeFromResultRegister(MachineInstr *Use,
                                                      Register UseRegister,
                                                      SPIRVGlobalRegistry *GR,
                                                      MachineIRBuilder &MIB) {
  assert(Use->getOpcode() == TargetOpcode::G_LOAD ||
         Use->getOpcode() == TargetOpcode::G_STORE);

  Register ValueReg = Use->getOperand(0).getReg();
  SPIRVType *ValueType = GR->getSPIRVTypeForVReg(ValueReg);
  if (!ValueType)
    return nullptr;

  return GR->getOrCreateSPIRVPointerType(ValueType, MIB,
                                         SPIRV::StorageClass::Function);
}

static SPIRVType *deduceTypeFromPointerOperand(MachineInstr *Use,
                                               Register UseRegister,
                                               SPIRVGlobalRegistry *GR,
                                               MachineIRBuilder &MIB) {
  assert(Use->getOpcode() == TargetOpcode::G_LOAD ||
         Use->getOpcode() == TargetOpcode::G_STORE);

  Register PtrReg = Use->getOperand(1).getReg();
  SPIRVType *PtrType = GR->getSPIRVTypeForVReg(PtrReg);
  if (!PtrType)
    return nullptr;

  return GR->getPointeeType(PtrType);
}

static SPIRVType *deduceTypeFromUses(Register Reg, MachineFunction &MF,
                                     SPIRVGlobalRegistry *GR,
                                     MachineIRBuilder &MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineInstr &Use : MRI.use_nodbg_instructions(Reg)) {
    SPIRVType *ResType = nullptr;
    LLVM_DEBUG(dbgs() << "Looking at use " << Use);
    switch (Use.getOpcode()) {
    case TargetOpcode::G_BUILD_VECTOR:
    case TargetOpcode::G_EXTRACT_VECTOR_ELT:
    case TargetOpcode::G_UNMERGE_VALUES:
    case TargetOpcode::G_ADD:
    case TargetOpcode::G_SUB:
    case TargetOpcode::G_MUL:
    case TargetOpcode::G_SDIV:
    case TargetOpcode::G_UDIV:
    case TargetOpcode::G_SREM:
    case TargetOpcode::G_UREM:
    case TargetOpcode::G_FADD:
    case TargetOpcode::G_FSUB:
    case TargetOpcode::G_FMUL:
    case TargetOpcode::G_FDIV:
    case TargetOpcode::G_FREM:
    case TargetOpcode::G_FMA:
    case TargetOpcode::COPY:
    case TargetOpcode::G_STRICT_FMA:
      ResType = deduceTypeFromResultRegister(&Use, Reg, GR, MIB);
      break;
    case TargetOpcode::G_LOAD:
    case TargetOpcode::G_STORE:
      if (Reg == Use.getOperand(1).getReg())
        ResType = deducePointerTypeFromResultRegister(&Use, Reg, GR, MIB);
      else
        ResType = deduceTypeFromPointerOperand(&Use, Reg, GR, MIB);
      break;
    case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
    case TargetOpcode::G_INTRINSIC: {
      auto IntrinsicID = cast<GIntrinsic>(Use).getIntrinsicID();
      if (IntrinsicID == Intrinsic::spv_insertelt) {
        if (Reg == Use.getOperand(2).getReg())
          ResType = deduceTypeFromResultRegister(&Use, Reg, GR, MIB);
      } else if (IntrinsicID == Intrinsic::spv_extractelt) {
        if (Reg == Use.getOperand(2).getReg())
          ResType = deduceTypeFromResultRegister(&Use, Reg, GR, MIB);
      }
      break;
    }
    }
    if (ResType) {
      LLVM_DEBUG(dbgs() << "Deduced type from use " << *ResType);
      return ResType;
    }
  }
  return nullptr;
}

static SPIRVType *deduceGEPType(MachineInstr *I, SPIRVGlobalRegistry *GR,
                                MachineIRBuilder &MIB) {
  LLVM_DEBUG(dbgs() << "Deducing GEP type for: " << *I);
  Register PtrReg = I->getOperand(3).getReg();
  SPIRVType *PtrType = GR->getSPIRVTypeForVReg(PtrReg);
  if (!PtrType) {
    LLVM_DEBUG(dbgs() << "  Could not get type for pointer operand.\n");
    return nullptr;
  }

  SPIRVType *PointeeType = GR->getPointeeType(PtrType);
  if (!PointeeType) {
    LLVM_DEBUG(dbgs() << "  Could not get pointee type from pointer type.\n");
    return nullptr;
  }

  MachineRegisterInfo *MRI = MIB.getMRI();

  // The first index (operand 4) steps over the pointer, so the type doesn't
  // change.
  for (unsigned i = 5; i < I->getNumOperands(); ++i) {
    LLVM_DEBUG(dbgs() << "  Traversing index " << i
                      << ", current type: " << *PointeeType);
    switch (PointeeType->getOpcode()) {
    case SPIRV::OpTypeArray:
    case SPIRV::OpTypeRuntimeArray:
    case SPIRV::OpTypeVector: {
      Register ElemTypeReg = PointeeType->getOperand(1).getReg();
      PointeeType = GR->getSPIRVTypeForVReg(ElemTypeReg);
      break;
    }
    case SPIRV::OpTypeStruct: {
      MachineOperand &IdxOp = I->getOperand(i);
      if (!IdxOp.isReg()) {
        LLVM_DEBUG(dbgs() << "  Index is not a register.\n");
        return nullptr;
      }
      MachineInstr *Def = MRI->getVRegDef(IdxOp.getReg());
      if (!Def) {
        LLVM_DEBUG(
            dbgs() << "  Could not find definition for index register.\n");
        return nullptr;
      }

      uint64_t IndexVal = foldImm(IdxOp, MRI);
      if (IndexVal >= PointeeType->getNumOperands() - 1) {
        LLVM_DEBUG(dbgs() << "  Struct index out of bounds.\n");
        return nullptr;
      }

      Register MemberTypeReg = PointeeType->getOperand(IndexVal + 1).getReg();
      PointeeType = GR->getSPIRVTypeForVReg(MemberTypeReg);
      break;
    }
    default:
      LLVM_DEBUG(dbgs() << "  Unknown type opcode for GEP traversal.\n");
      return nullptr;
    }

    if (!PointeeType) {
      LLVM_DEBUG(dbgs() << "  Could not resolve next pointee type.\n");
      return nullptr;
    }
  }
  LLVM_DEBUG(dbgs() << "  Final pointee type: " << *PointeeType);

  SPIRV::StorageClass::StorageClass SC = GR->getPointerStorageClass(PtrType);
  SPIRVType *Res = GR->getOrCreateSPIRVPointerType(PointeeType, MIB, SC);
  LLVM_DEBUG(dbgs() << "  Deduced GEP type: " << *Res);
  return Res;
}

static SPIRVType *deduceResultTypeFromOperands(MachineInstr *I,
                                               SPIRVGlobalRegistry *GR,
                                               MachineIRBuilder &MIB) {
  Register ResVReg = I->getOperand(0).getReg();
  switch (I->getOpcode()) {
  case TargetOpcode::G_CONSTANT:
  case TargetOpcode::G_ANYEXT:
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ZEXT:
    return deduceIntTypeFromResult(ResVReg, MIB, GR);
  case TargetOpcode::G_BUILD_VECTOR:
    return deduceTypeFromOperandRange(I, MIB, GR, 1, I->getNumOperands());
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return deduceTypeFromOperandRange(I, MIB, GR, 1, 3);
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
  case TargetOpcode::G_INTRINSIC: {
    auto IntrinsicID = cast<GIntrinsic>(I)->getIntrinsicID();
    if (IntrinsicID == Intrinsic::spv_gep)
      return deduceGEPType(I, GR, MIB);
    break;
  }
  case TargetOpcode::G_LOAD: {
    SPIRVType *PtrType = deduceTypeFromSingleOperand(I, MIB, GR, 1);
    return PtrType ? GR->getPointeeType(PtrType) : nullptr;
  }
  default:
    if (I->getNumDefs() == 1 && I->getNumOperands() > 1 &&
        I->getOperand(1).isReg())
      return deduceTypeFromSingleOperand(I, MIB, GR, 1);
  }
  return nullptr;
}

static bool deduceAndAssignTypeForGUnmerge(MachineInstr *I, MachineFunction &MF,
                                           SPIRVGlobalRegistry *GR,
                                           MachineIRBuilder &MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register SrcReg = I->getOperand(I->getNumOperands() - 1).getReg();
  SPIRVType *ScalarType = nullptr;
  if (SPIRVType *DefType = GR->getSPIRVTypeForVReg(SrcReg)) {
    assert(DefType->getOpcode() == SPIRV::OpTypeVector);
    ScalarType = GR->getSPIRVTypeForVReg(DefType->getOperand(1).getReg());
  }

  if (!ScalarType) {
    // If we could not deduce the type from the source, try to deduce it from
    // the uses of the results.
    for (unsigned i = 0; i < I->getNumDefs(); ++i) {
      Register DefReg = I->getOperand(i).getReg();
      ScalarType = deduceTypeFromUses(DefReg, MF, GR, MIB);
      if (ScalarType) {
        ScalarType = GR->getScalarOrVectorComponentType(ScalarType);
        break;
      }
    }
  }

  if (!ScalarType)
    return false;

  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    Register DefReg = I->getOperand(i).getReg();
    if (GR->getSPIRVTypeForVReg(DefReg))
      continue;

    LLT DefLLT = MRI.getType(DefReg);
    SPIRVType *ResType =
        DefLLT.isVector()
            ? GR->getOrCreateSPIRVVectorType(
                  ScalarType, DefLLT.getNumElements(), *I,
                  *MF.getSubtarget<SPIRVSubtarget>().getInstrInfo())
            : ScalarType;
    setRegClassType(DefReg, ResType, GR, &MRI, MF);
  }
  return true;
}

static bool deduceAndAssignSpirvType(MachineInstr *I, MachineFunction &MF,
                                     SPIRVGlobalRegistry *GR,
                                     MachineIRBuilder &MIB) {
  LLVM_DEBUG(dbgs() << "\nProcessing instruction: " << *I);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register ResVReg = I->getOperand(0).getReg();

  // G_UNMERGE_VALUES is handled separately because it has multiple definitions,
  // unlike the other instructions which have a single result register. The main
  // deduction logic is designed for the single-definition case.
  if (I->getOpcode() == TargetOpcode::G_UNMERGE_VALUES)
    return deduceAndAssignTypeForGUnmerge(I, MF, GR, MIB);

  LLVM_DEBUG(dbgs() << "Inferring type from operands\n");
  SPIRVType *ResType = deduceResultTypeFromOperands(I, GR, MIB);
  if (!ResType) {
    LLVM_DEBUG(dbgs() << "Inferring type from uses\n");
    ResType = deduceTypeFromUses(ResVReg, MF, GR, MIB);
  }

  if (!ResType)
    return false;

  LLVM_DEBUG(dbgs() << "Assigned type to " << *I << ": " << *ResType);
  GR->assignSPIRVTypeToVReg(ResType, ResVReg, MF);

  if (!MRI.getRegClassOrNull(ResVReg)) {
    LLVM_DEBUG(dbgs() << "Updating the register class.\n");
    setRegClassType(ResVReg, ResType, GR, &MRI, *GR->CurMF, true);
  }
  return true;
}

static bool requiresSpirvType(MachineInstr &I, SPIRVGlobalRegistry *GR,
                              MachineRegisterInfo &MRI) {
  LLVM_DEBUG(dbgs() << "Checking if instruction requires a SPIR-V type: "
                    << I;);
  if (I.getNumDefs() == 0) {
    LLVM_DEBUG(dbgs() << "Instruction does not have a definition.\n");
    return false;
  }

  if (!I.isPreISelOpcode()) {
    LLVM_DEBUG(dbgs() << "Instruction is not a generic instruction.\n");
    return false;
  }

  Register ResultRegister = I.defs().begin()->getReg();
  if (GR->getSPIRVTypeForVReg(ResultRegister)) {
    LLVM_DEBUG(dbgs() << "Instruction already has a SPIR-V type.\n");
    if (!MRI.getRegClassOrNull(ResultRegister)) {
      LLVM_DEBUG(dbgs() << "Updating the register class.\n");
      setRegClassType(ResultRegister, GR->getSPIRVTypeForVReg(ResultRegister),
                      GR, &MRI, *GR->CurMF, true);
    }
    return false;
  }

  return true;
}

static void registerSpirvTypeForNewInstructions(MachineFunction &MF,
                                                SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<MachineInstr *, 8> Worklist;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &I : MBB) {
      if (requiresSpirvType(I, GR, MRI)) {
        Worklist.push_back(&I);
      }
    }
  }

  if (Worklist.empty()) {
    LLVM_DEBUG(dbgs() << "Initial worklist is empty.\n");
    return;
  }

  LLVM_DEBUG(dbgs() << "Initial worklist:\n";
             for (auto *I : Worklist) { I->dump(); });

  bool Changed;
  do {
    Changed = false;
    SmallVector<MachineInstr *, 8> NextWorklist;

    for (MachineInstr *I : Worklist) {
      MachineIRBuilder MIB(*I);
      if (deduceAndAssignSpirvType(I, MF, GR, MIB)) {
        Changed = true;
      } else {
        NextWorklist.push_back(I);
      }
    }
    Worklist = std::move(NextWorklist);
    LLVM_DEBUG(dbgs() << "Worklist size: " << Worklist.size() << "\n");
  } while (Changed);

  if (Worklist.empty())
    return;

  for (auto *I : Worklist) {
    MachineIRBuilder MIB(*I);
    LLVM_DEBUG(dbgs() << "Assigning default type to results in " << *I);
    for (unsigned Idx = 0; Idx < I->getNumDefs(); ++Idx) {
      Register ResVReg = I->getOperand(Idx).getReg();
      if (GR->getSPIRVTypeForVReg(ResVReg))
        continue;
      const LLT &ResLLT = MRI.getType(ResVReg);
      SPIRVType *ResType = nullptr;
      if (ResLLT.isVector()) {
        SPIRVType *CompType = GR->getOrCreateSPIRVIntegerType(
            ResLLT.getElementType().getSizeInBits(), MIB);
        ResType = GR->getOrCreateSPIRVVectorType(
            CompType, ResLLT.getNumElements(), MIB, false);
      } else {
        ResType = GR->getOrCreateSPIRVIntegerType(ResLLT.getSizeInBits(), MIB);
      }
      setRegClassType(ResVReg, ResType, GR, &MRI, MF, true);
    }
  }
}

static bool hasAssignType(Register Reg, MachineRegisterInfo &MRI) {
  for (MachineInstr &UseInstr : MRI.use_nodbg_instructions(Reg)) {
    if (UseInstr.getOpcode() == SPIRV::ASSIGN_TYPE) {
      return true;
    }
  }
  return false;
}

static void generateAssignType(MachineInstr &MI, Register ResultRegister,
                               SPIRVType *ResultType, SPIRVGlobalRegistry *GR,
                               MachineRegisterInfo &MRI) {
  LLVM_DEBUG(dbgs() << "  Adding ASSIGN_TYPE for ResultRegister: "
                    << printReg(ResultRegister, MRI.getTargetRegisterInfo())
                    << " with type: " << *ResultType);
  MachineIRBuilder MIB(MI);
  updateRegType(ResultRegister, nullptr, ResultType, GR, MIB, MRI);

  // Tablegen definition assumes SPIRV::ASSIGN_TYPE pseudo-instruction is
  // present after each auto-folded instruction to take a type reference
  // from.
  Register NewReg =
      MRI.createGenericVirtualRegister(MRI.getType(ResultRegister));
  const auto *RegClass = GR->getRegClass(ResultType);
  MRI.setRegClass(NewReg, RegClass);
  MRI.setRegClass(ResultRegister, RegClass);

  GR->assignSPIRVTypeToVReg(ResultType, ResultRegister, MIB.getMF());
  // This is to make it convenient for Legalizer to get the SPIRVType
  // when processing the actual MI (i.e. not pseudo one).
  GR->assignSPIRVTypeToVReg(ResultType, NewReg, MIB.getMF());
  // Copy MIFlags from Def to ASSIGN_TYPE instruction. It's required to
  // keep the flags after instruction selection.
  const uint32_t Flags = MI.getFlags();
  MIB.buildInstr(SPIRV::ASSIGN_TYPE)
      .addDef(ResultRegister)
      .addUse(NewReg)
      .addUse(GR->getSPIRVTypeID(ResultType))
      .setMIFlags(Flags);
  for (unsigned I = 0, E = MI.getNumDefs(); I != E; ++I) {
    MachineOperand &MO = MI.getOperand(I);
    if (MO.getReg() == ResultRegister) {
      MO.setReg(NewReg);
      break;
    }
  }
}

static void ensureAssignTypeForTypeFolding(MachineFunction &MF,
                                           SPIRVGlobalRegistry *GR) {
  LLVM_DEBUG(dbgs() << "Entering ensureAssignTypeForTypeFolding for function "
                    << MF.getName() << "\n");
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isTypeFoldingSupported(MI.getOpcode()))
        continue;

      LLVM_DEBUG(dbgs() << "Processing instruction: " << MI);

      Register ResultRegister = MI.defs().begin()->getReg();
      if (hasAssignType(ResultRegister, MRI)) {
        LLVM_DEBUG(dbgs() << "  Instruction already has ASSIGN_TYPE\n");
        continue;
      }

      SPIRVType *ResultType = GR->getSPIRVTypeForVReg(ResultRegister);
      assert(ResultType);
      generateAssignType(MI, ResultRegister, ResultType, GR, MRI);
    }
  }
}

// Do a preorder traversal of the CFG starting from the BB |Start|.
// point. Calls |op| on each basic block encountered during the traversal.
void visit(MachineFunction &MF, MachineBasicBlock &Start,
           std::function<void(MachineBasicBlock *)> op) {
  std::stack<MachineBasicBlock *> ToVisit;
  SmallPtrSet<MachineBasicBlock *, 8> Seen;

  ToVisit.push(&Start);
  Seen.insert(ToVisit.top());
  while (ToVisit.size() != 0) {
    MachineBasicBlock *MBB = ToVisit.top();
    ToVisit.pop();

    op(MBB);

    for (auto Succ : MBB->successors()) {
      if (Seen.contains(Succ))
        continue;
      ToVisit.push(Succ);
      Seen.insert(Succ);
    }
  }
}

// Do a preorder traversal of the CFG starting from the given function's entry
// point. Calls |op| on each basic block encountered during the traversal.
void visit(MachineFunction &MF, std::function<void(MachineBasicBlock *)> op) {
  visit(MF, *MF.begin(), std::move(op));
}

bool SPIRVPostLegalizer::runOnMachineFunction(MachineFunction &MF) {
  // Initialize the type registry.
  const SPIRVSubtarget &ST = MF.getSubtarget<SPIRVSubtarget>();
  SPIRVGlobalRegistry *GR = ST.getSPIRVGlobalRegistry();
  GR->setCurrentFunc(MF);
  registerSpirvTypeForNewInstructions(MF, GR);
  ensureAssignTypeForTypeFolding(MF, GR);
  return true;
}

INITIALIZE_PASS(SPIRVPostLegalizer, DEBUG_TYPE, "SPIRV post legalizer", false,
                false)

char SPIRVPostLegalizer::ID = 0;

FunctionPass *llvm::createSPIRVPostLegalizerPass() {
  return new SPIRVPostLegalizer();
}
