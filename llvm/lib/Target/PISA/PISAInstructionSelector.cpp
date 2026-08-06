//===-- PISAInstructionSelector.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PISAEnum.h"
#include "PISA.h"
#include "PISAGenInstrInfo.inc"
#include "PISAInstrInfo.h"
#include "PISARegisterBankInfo.h"
#include "PISARegisterInfo.h"
#include "PISATargetMachine.h"
#include "PISAUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/IR/PISAIntrinsicUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PISAAddrSpace.h"

namespace llvm {
namespace PISA {
struct Name2InstrEntry {
  StringTable::Offset Name;
  unsigned Opcode;
};

#define GET_Name2InstrOpTable_DECL
#define GET_Name2InstrOpTable_IMPL
#include "PISAGenSearchableTables.inc"

} // namespace PISA
} // namespace llvm

#define DEBUG_TYPE "pisa-isel"

using namespace llvm;
using namespace MIPatternMatch;

namespace {

/// Returns an APFloat from Val converted to the appropriate size.
static APFloat getAPFloatFromSize(double Val, unsigned Size) {
  if (Size == 32)
    return APFloat(float(Val));
  if (Size == 64)
    return APFloat(Val);
  if (Size != 16)
    llvm_unreachable("Unsupported FPConstant size");
  bool Ignored;
  APFloat APF(Val);
  APF.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Ignored);
  return APF;
}

struct SyncScopeInfo {
  static const StringMap<unsigned> Name2Opcode;
  DenseMap<llvm::SyncScope::ID, unsigned> ID2Opcode;
  SyncScope::ID SystemGenericID = SyncScope::System;

  SyncScopeInfo() {}
  SyncScopeInfo(llvm::LLVMContext &Context) {
    for (const auto &[Name, Opcode] : Name2Opcode) {
      auto ID = Context.getOrInsertSyncScopeID(Name);
      ID2Opcode.emplace_or_assign(ID, Opcode);
    }
    ID2Opcode.emplace_or_assign(SyncScope::SingleThread, PISA::fence_subgroup);
    ID2Opcode.emplace_or_assign(SyncScope::System, PISA::fence_global_system);
    SystemGenericID = Context.getOrInsertSyncScopeID("system-generic");
  }
};

const StringMap<unsigned> SyncScopeInfo::Name2Opcode = {
    {"workitem", PISA::fence_subgroup},
    {"workitem-shared", PISA::fence_subgroup},
    {"workitem-global", PISA::fence_subgroup},
    {"workitem-generic", PISA::fence_subgroup},
    {"subgroup", PISA::fence_subgroup},
    {"subgroup-shared", PISA::fence_subgroup},
    {"subgroup-global", PISA::fence_subgroup},
    {"subgroup-generic", PISA::fence_subgroup},
    {"workgroup", PISA::fence_generic_workgroup},
    {"workgroup-shared", PISA::fence_shared_workgroup},
    {"workgroup-global", PISA::fence_global_workgroup},
    {"workgroup-generic", PISA::fence_generic_workgroup},
    {"gpu", PISA::fence_generic_gpu},
    {"gpu-shared", PISA::fence_shared_gpu},
    {"gpu-global", PISA::fence_global_gpu},
    {"gpu-generic", PISA::fence_generic_gpu},
    {"system", PISA::fence_global_system},
    {"system-shared", PISA::fence_shared_gpu},
    {"system-global", PISA::fence_global_system},
};

#define GET_GLOBALISEL_PREDICATE_BITSET
#include "PISAGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATE_BITSET

class PISAInstructionSelector : public InstructionSelector {
  const PISASubtarget &STI;
  const PISAInstrInfo &TII;
  const PISARegisterInfo &TRI;
  const RegisterBankInfo &RBI;
  MachineRegisterInfo *MRI = nullptr;
  SyncScopeInfo SSI;
  MachineFrameInfo *MFI = nullptr;

public:
  PISAInstructionSelector(const PISATargetMachine &TM, const PISASubtarget &ST,
                          const RegisterBankInfo &RBI);
  void setupMF(MachineFunction &MF, GISelValueTracking *KB,
               CodeGenCoverage *CoverageInfo, ProfileSummaryInfo *PSI,
               BlockFrequencyInfo *BFI) override;
  bool select(MachineInstr &I) override;
  static const char *getName() { return DEBUG_TYPE; }

#define GET_GLOBALISEL_PREDICATES_DECL
#include "PISAGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATES_DECL

#define GET_GLOBALISEL_TEMPORARIES_DECL
#include "PISAGenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_DECL

private:
  // tblgen-erated 'select' implementation, used as the initial selector for
  // the patterns that don't require complex C++.
  bool selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const;
  bool selectFrameIndex(MachineInstr &I) const;
  bool selectGlobalValue(MachineInstr &I) const;
  bool selectPhiOrImplicitDef(MachineInstr &I) const;
  bool selectAddSubWithOverflow(MachineInstr &I) const;
  bool selectSignedAddSubWithOverflow(MachineInstr &I) const;
  bool selectBitcast(MachineInstr &I) const;
  // NOLINTBEGIN(readability-identifier-naming)
  bool selectG_BUILD_VECTOR(MachineInstr &I) const;
  bool selectConvPtrInt(MachineInstr &I) const;
  bool selectCopy(MachineInstr &I) const;
  bool selectG_INTRINSIC(MachineInstr &I) const;
  bool selectG_INTRINSIC_WITH_SIDEFFECTS(MachineInstr &I) const;
  bool selectDbgValue(MachineInstr &I) const;
  bool selectG_FCANONICALIZE(MachineInstr &I) const;
  bool selectG_LROUND(MachineInstr &I) const;
  bool selectG_MERGE_VALUES(MachineInstr &MI) const;
  bool selectG_UNMERGE_VALUES(MachineInstr &MI) const;
  bool selectG_BFX(MachineInstr &I, bool Signed) const;
  bool selectG_CONSTANT(MachineInstr &I) const;
  bool selectG_FENCE(MachineInstr &I) const;
  bool selectG_INSERT_SUBVECTOR(MachineInstr &MI) const;
  bool selectG_EXTRACT_SUBVECTOR(MachineInstr &MI) const;
  bool selectG_INSERT_VECTOR_ELT(MachineInstr &MI) const;
  bool selectG_EXTRACT_VECTOR_ELT(MachineInstr &MI) const;
  bool selectG_ADDRSPACE_CAST(llvm::MachineInstr &I) const;
  MachineInstr *findPtrBaseDefinition(MachineInstr *StartDef,
                                      const MachineRegisterInfo &MRI) const;

  InstructionSelector::ComplexRendererFns
  SelectAddr_rr(MachineOperand &Root) const;
  InstructionSelector::ComplexRendererFns
  SelectAddr_ri(MachineOperand &Root) const;
  InstructionSelector::ComplexRendererFns
  selectParamSlot_ii(MachineOperand &Root) const;
  InstructionSelector::ComplexRendererFns
  selectParamSlot_ir(MachineOperand &Root) const;

  // NOLINTEND(readability-identifier-naming)

  bool emitBuildVector(MachineInstr &InsertPt, Register DstReg,
                       ArrayRef<Register> Srcs) const;

  bool useFastFP(const MachineInstr &I) const {
    return I.getFlag(MachineInstr::MIFlag::FmAfn);
  }
};

} // namespace

#define GET_GLOBALISEL_IMPL
#include "PISAGenGlobalISel.inc"
#undef GET_GLOBALISEL_IMPL

PISAInstructionSelector::PISAInstructionSelector(const PISATargetMachine &TM,
                                                 const PISASubtarget &ST,
                                                 const RegisterBankInfo &RBI)
    : InstructionSelector(), STI(ST), TII(*ST.getInstrInfo()),
      TRI(*ST.getRegisterInfo()), RBI(RBI),
#define GET_GLOBALISEL_PREDICATES_INIT
#include "PISAGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATES_INIT
#define GET_GLOBALISEL_TEMPORARIES_INIT
#include "PISAGenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_INIT
{
}

void PISAInstructionSelector::setupMF(MachineFunction &MF,
                                      GISelValueTracking *KB,
                                      CodeGenCoverage *CoverageInfo,
                                      ProfileSummaryInfo *PSI,
                                      BlockFrequencyInfo *BFI) {
  constexpr unsigned SharedAS = unsigned(PISAAS::AddressSpace::SHARED);

  SSI = SyncScopeInfo(MF.getFunction().getContext());
  InstructionSelector::setupMF(MF, KB, CoverageInfo, PSI, BFI);
  MRI = &MF.getRegInfo();
  MFI = &MF.getFrameInfo();
  // Setup the correct stack ID.
  for (int Idx = MFI->getObjectIndexBegin(), EndIdx = MFI->getObjectIndexEnd();
       Idx != EndIdx; ++Idx) {
    if (auto *AI = MFI->getObjectAllocation(Idx)) {
      const auto AS = AI->getAddressSpace();
      Type *AllocatedTy = AI->getAllocatedType();
      if (auto *ATy = dyn_cast<ArrayType>(AllocatedTy))
        AllocatedTy = ATy->getElementType();

      auto *TExtTy = dyn_cast<TargetExtType>(AllocatedTy);

      if (AS == SharedAS) {
        assert(MF.getFunction().getCallingConv() == CallingConv::PISA_KERNEL &&
               "shared variables are not supported in functions!");
        unsigned StackID = TargetStackID::PISAShared;
        if (TExtTy)
          StackID = llvm::StringSwitch<unsigned>(TExtTy->getName())
                        .Default(TargetStackID::PISAShared);
        MFI->setStackID(Idx, StackID);
      }
    }
  }
}

bool PISAInstructionSelector::select(MachineInstr &I) {
  assert(I.getParent() && "Instruction should be in a basic block!");
  assert(I.getParent()->getParent() && "Instruction should be in a function!");

  unsigned Opcode = I.getOpcode();

  // If it's not a GMIR instruction, we've selected it already.
  if (!I.isPreISelOpcode() && Opcode != TargetOpcode::DBG_VALUE) {
    if (I.isCopy())
      return selectCopy(I);
    constrainSelectedInstRegOperands(I, TII, TRI, RBI);
    return true;
  }

  switch (Opcode) {
  default:
    break;
  case TargetOpcode::G_ADDRSPACE_CAST:
    if (selectG_ADDRSPACE_CAST(I))
      return true;
    break;
  }

  if (selectImpl(I, *CoverageInfo))
    return true;

  if (I.getNumOperands() != I.getNumExplicitOperands()) {
    LLVM_DEBUG(errs() << "Generic instr has unexpected implicit operands\n");
    return false;
  }

  switch (Opcode) {
  case TargetOpcode::G_FRAME_INDEX:
    return selectFrameIndex(I);
  case TargetOpcode::G_GLOBAL_VALUE:
    return selectGlobalValue(I);
  case TargetOpcode::G_PHI:
  case TargetOpcode::G_IMPLICIT_DEF:
    return selectPhiOrImplicitDef(I);
  case TargetOpcode::G_UADDO:
  case TargetOpcode::G_USUBO:
  case TargetOpcode::G_UADDE:
  case TargetOpcode::G_USUBE:
    return selectAddSubWithOverflow(I);
  case TargetOpcode::G_SADDO:
  case TargetOpcode::G_SSUBO:
  case TargetOpcode::G_SADDE:
  case TargetOpcode::G_SSUBE:
    return selectSignedAddSubWithOverflow(I);
  case TargetOpcode::G_BITCAST:
    return selectBitcast(I);
  case TargetOpcode::G_UNMERGE_VALUES:
    return selectG_UNMERGE_VALUES(I);
  case TargetOpcode::G_CONCAT_VECTORS:
  case TargetOpcode::G_MERGE_VALUES:
    return selectG_MERGE_VALUES(I);
  case TargetOpcode::G_BUILD_VECTOR:
    return selectG_BUILD_VECTOR(I);
  case TargetOpcode::G_PTRTOINT:
  case TargetOpcode::G_INTTOPTR:
    return selectConvPtrInt(I);
  case TargetOpcode::G_FREEZE:
    return selectCopy(I);
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_CONVERGENT:
    return selectG_INTRINSIC(I);
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
  case TargetOpcode::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS:
    return selectG_INTRINSIC_WITH_SIDEFFECTS(I);
  case TargetOpcode::DBG_VALUE:
    return selectDbgValue(I);
  case TargetOpcode::G_FCANONICALIZE:
    return selectG_FCANONICALIZE(I);
  case TargetOpcode::G_LROUND:
  case TargetOpcode::G_LLROUND:
    return selectG_LROUND(I);
  case TargetOpcode::G_UBFX:
    return selectG_BFX(I, false);
  case TargetOpcode::G_SBFX:
    return selectG_BFX(I, true);
  case TargetOpcode::G_CONSTANT:
    return selectG_CONSTANT(I);
  case TargetOpcode::G_FENCE:
    return selectG_FENCE(I);
  case TargetOpcode::G_INSERT_SUBVECTOR:
    return selectG_INSERT_SUBVECTOR(I);
  case TargetOpcode::G_EXTRACT_SUBVECTOR:
    return selectG_EXTRACT_SUBVECTOR(I);
  case TargetOpcode::G_INSERT_VECTOR_ELT:
    return selectG_INSERT_VECTOR_ELT(I);
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
    return selectG_EXTRACT_VECTOR_ELT(I);
  default:
    return false;
  }
}

MachineInstr *PISAInstructionSelector::findPtrBaseDefinition(
    MachineInstr *StartDef, const MachineRegisterInfo &MRI) const {
  if (!StartDef)
    return nullptr;

  SmallVector<MachineInstr *, 8> WorkList;
  SmallPtrSet<MachineInstr *, 8> Visited;
  MachineInstr *DefMI = StartDef;
  WorkList.push_back(DefMI);

  while (!WorkList.empty()) {
    DefMI = WorkList.pop_back_val();
    if (!Visited.insert(DefMI).second)
      continue;

    auto Opcode = DefMI->getOpcode();
    switch (Opcode) {
    default:
      // If we see an instruction that we don't recognize, stop looking and try
      // to select the G_ADDRSPACE_CAST as is.
      WorkList.clear();
      break;
    case TargetOpcode::G_PTR_ADD: {
      MachineInstr *BaseDef =
          getDefIgnoringCopies(DefMI->getOperand(1).getReg(), MRI);
      if (BaseDef)
        WorkList.push_back(BaseDef);
    } break;
    case TargetOpcode::G_PHI:
      for (unsigned I = 1, E = DefMI->getNumOperands(); I < E; I += 2) {
        MachineInstr *PhiDef =
            getDefIgnoringCopies(DefMI->getOperand(I).getReg(), MRI);
        if (PhiDef)
          WorkList.push_back(PhiDef);
      }
      break;
    case TargetOpcode::G_SELECT:
      for (unsigned I = 2, E = DefMI->getNumOperands(); I < E; I++) {
        MachineInstr *SelectDef =
            getDefIgnoringCopies(DefMI->getOperand(I).getReg(), MRI);
        if (SelectDef)
          WorkList.push_back(SelectDef);
      }
      break;
    }
  }

  return DefMI;
}

bool PISAInstructionSelector::selectG_ADDRSPACE_CAST(
    llvm::MachineInstr &I) const {
  constexpr unsigned GenericAS =
      static_cast<unsigned>(PISAAS::AddressSpace::GENERIC);
  [[maybe_unused]] constexpr unsigned SharedAS =
      static_cast<unsigned>(PISAAS::AddressSpace::SHARED);
  [[maybe_unused]] constexpr unsigned PrivateAS =
      static_cast<unsigned>(PISAAS::AddressSpace::PRIVATE);

  auto [Dst, DstTy, Src, SrcTy] = I.getFirst2RegLLTs();
  const unsigned DstAS = DstTy.getAddressSpace();
  [[maybe_unused]] const unsigned SrcAS = SrcTy.getAddressSpace();

  if (DstAS != GenericAS)
    return false;

  MachineInstr *DefMI =
      findPtrBaseDefinition(getDefIgnoringCopies(Src, *MRI), *MRI);

  if (!DefMI || DefMI->getOpcode() != TargetOpcode::G_FRAME_INDEX)
    return false;

  int FI = DefMI->getOperand(1).getIndex();
  switch (MFI->getStackID(FI)) {
  default: // No custom selection needed
    break;
  }

  return false;
}

bool PISAInstructionSelector::selectG_INTRINSIC(MachineInstr &I) const {
  // While we want to select special reg intrinsics in tablegen using
  // SpecialRegPat<>, there is an issue in llvm that prevents us from doing
  // so when the output value is a pointer. It fails in:
  // GlobalISelEmitter::createAndImportSelDAGMatcher()
  // because isOutOperandAPointer() will always return false for G_INTRINSIC
  // as its OutOperandList = (outs). This then causes addTypeCheckPredicate()
  // to not generate a pointer matcher, so tablegen fails saying that we have
  // an unsupported type (even though it is an iPTR). Until this is fixed,
  // manually select these cases here.

  switch (cast<GIntrinsic>(I).getIntrinsicID()) {
  default:
    return false;
  }
}

bool PISAInstructionSelector::selectG_INTRINSIC_WITH_SIDEFFECTS(
    MachineInstr &I) const {
  // Find the intrinsic ID.
  unsigned IntrinID = cast<GIntrinsic>(I).getIntrinsicID();
  switch (IntrinID) {
  default:
    return false;
  }
}

bool PISAInstructionSelector::selectFrameIndex(MachineInstr &I) const {
  I.setDesc(TII.get(PISA::addrof_32b));
  // need to set dst reg-class
  const Register DefReg = I.getOperand(0).getReg();
  const LLT DefTy = MRI->getType(DefReg);

  const RegClassOrRegBank &RegClassOrBank = MRI->getRegClassOrRegBank(DefReg);
  const TargetRegisterClass *DefRC =
      RegClassOrBank.dyn_cast<const TargetRegisterClass *>();
  if (!DefRC)
    DefRC = TRI.getRegClassFromLLT(DefTy);
  return RBI.constrainGenericRegister(DefReg, *DefRC, *MRI);
}

bool PISAInstructionSelector::selectGlobalValue(MachineInstr &I) const {
  I.setDesc(TII.get(PISA::addrof_64b));

  Register DefReg = I.getOperand(0).getReg();
  const LLT DefTy = MRI->getType(DefReg);

  const RegClassOrRegBank &RegClassOrBank = MRI->getRegClassOrRegBank(DefReg);
  const TargetRegisterClass *DefRC =
      RegClassOrBank.dyn_cast<const TargetRegisterClass *>();
  if (!DefRC)
    DefRC = TRI.getRegClassFromLLT(DefTy);

  return RBI.constrainGenericRegister(DefReg, *DefRC, *MRI);
}

bool PISAInstructionSelector::selectPhiOrImplicitDef(MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  const Register DefReg = I.getOperand(0).getReg();
  const LLT DefTy = MRI.getType(DefReg);

  const RegClassOrRegBank &RegClassOrBank = MRI.getRegClassOrRegBank(DefReg);

  const TargetRegisterClass *DefRC =
      RegClassOrBank.dyn_cast<const TargetRegisterClass *>();

  if (!DefRC)
    DefRC = TRI.getRegClassFromLLT(DefTy);

  unsigned Opcode = (I.getOpcode() == TargetOpcode::G_PHI)
                        ? TargetOpcode::PHI
                        : TargetOpcode::IMPLICIT_DEF;
  I.setDesc(TII.get(Opcode));

  return RBI.constrainGenericRegister(DefReg, *DefRC, MRI);
}

bool PISAInstructionSelector::selectAddSubWithOverflow(MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  const DebugLoc &DL = I.getDebugLoc();

  const bool IsAdd = I.getOpcode() == TargetOpcode::G_UADDO ||
                     I.getOpcode() == TargetOpcode::G_UADDE;
  const bool HasCarryIn = I.getOpcode() == TargetOpcode::G_UADDE ||
                          I.getOpcode() == TargetOpcode::G_USUBE;

  auto [DstReg, CarryOutReg, Src0Reg, Src1Reg] = I.getFirst4Regs();
  const auto *DstRC = TRI.getRegClassFromLLT(MRI->getType(DstReg));
  const auto *PredRC = &PISA::PredRegClass;

  // Is there any reason to support 16b addc/subb?
  assert(DstRC == &PISA::Reg32bRegClass);

  auto Src0C = getIConstantVRegValWithLookThrough(Src0Reg, *MRI);
  auto Src1C = getIConstantVRegValWithLookThrough(Src1Reg, *MRI);
  unsigned RegReg = Src0C ? 0 : (Src1C ? 1 : 2);

  const bool HasCarryOut = !HasCarryIn || !MRI->use_nodbg_empty(CarryOutReg);

  unsigned PISAOps[] = {PISA::uaddc_co_32b_ir,    PISA::uaddc_co_32b_ri,
                        PISA::uaddc_co_32b_rr,    PISA::usubb_co_32b_ir,
                        PISA::usubb_co_32b_ri,    PISA::usubb_co_32b_rr,
                        PISA::uaddc_ci_32b_ir,    PISA::uaddc_ci_32b_ri,
                        PISA::uaddc_ci_32b_rr,    PISA::usubb_ci_32b_ir,
                        PISA::usubb_ci_32b_ri,    PISA::usubb_ci_32b_rr,
                        PISA::uaddc_ci_co_32b_ir, PISA::uaddc_ci_co_32b_ri,
                        PISA::uaddc_ci_co_32b_rr, PISA::usubb_ci_co_32b_ir,
                        PISA::usubb_ci_co_32b_ri, PISA::usubb_ci_co_32b_rr};

  unsigned CarryOutOpc = PISAOps[(IsAdd ? 0 : 3) + RegReg];
  unsigned CarryInOpc = PISAOps[6 + (IsAdd ? 0 : 3) + RegReg];
  unsigned CarryInOutOpc = PISAOps[12 + (IsAdd ? 0 : 3) + RegReg];

  unsigned Opc = (HasCarryIn && HasCarryOut)
                     ? CarryInOutOpc
                     : (HasCarryOut ? CarryOutOpc : CarryInOpc);

  auto MIB = BuildMI(*BB, &I, DL, TII.get(Opc), DstReg);
  if (HasCarryOut)
    MIB.addDef(CarryOutReg);
  if (Src0C)
    MIB.addImm(Src0C->Value.getSExtValue());
  else
    MIB.add(I.getOperand(2));
  if (Src1C && !Src0C)
    MIB.addImm(Src1C->Value.getSExtValue());
  else
    MIB.add(I.getOperand(3));
  if (HasCarryIn)
    MIB.add(I.getOperand(4));

  if (!RBI.constrainGenericRegister(DstReg, *DstRC, *MRI) ||
      !RBI.constrainGenericRegister(CarryOutReg, *PredRC, *MRI) ||
      !RBI.constrainGenericRegister(Src0Reg, *DstRC, *MRI) ||
      !RBI.constrainGenericRegister(Src1Reg, *DstRC, *MRI))
    return false;
  if (HasCarryIn &&
      !RBI.constrainGenericRegister(I.getOperand(4).getReg(), *PredRC, *MRI))
    return false;

  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectSignedAddSubWithOverflow(
    MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  const DebugLoc &DL = I.getDebugLoc();

  const bool IsAdd = I.getOpcode() == TargetOpcode::G_SADDO ||
                     I.getOpcode() == TargetOpcode::G_SADDE;
  const bool HasCarryIn = I.getOpcode() == TargetOpcode::G_SADDE ||
                          I.getOpcode() == TargetOpcode::G_SSUBE;

  auto [DstReg, CarryOutReg, Src0Reg, Src1Reg] = I.getFirst4Regs();
  const auto *DstRC = TRI.getRegClassFromLLT(MRI->getType(DstReg));
  const auto *PredRC = &PISA::PredRegClass;

  assert(DstRC == &PISA::Reg32bRegClass);

  auto Src0C = getIConstantVRegValWithLookThrough(Src0Reg, *MRI);
  auto Src1C = getIConstantVRegValWithLookThrough(Src1Reg, *MRI);
  unsigned RegReg = Src0C ? 0 : (Src1C ? 1 : 2);

  unsigned PISAOps[] = {PISA::uaddc_co_32b_ir,    PISA::uaddc_co_32b_ri,
                        PISA::uaddc_co_32b_rr,    PISA::usubb_co_32b_ir,
                        PISA::usubb_co_32b_ri,    PISA::usubb_co_32b_rr,
                        PISA::uaddc_ci_co_32b_ir, PISA::uaddc_ci_co_32b_ri,
                        PISA::uaddc_ci_co_32b_rr, PISA::usubb_ci_co_32b_ir,
                        PISA::usubb_ci_co_32b_ri, PISA::usubb_ci_co_32b_rr};
  unsigned NoCarryOpc = PISAOps[(IsAdd ? 0 : 3) + RegReg];
  unsigned CarryOpc = PISAOps[6 + (IsAdd ? 0 : 3) + RegReg];

  auto CarryOutPredReg = MRI->createVirtualRegister(PredRC);
  const auto *IntRC = TRI.getRegClassFromLLT(LLT::integer(16));
  auto CarryOutIntReg = MRI->createVirtualRegister(IntRC);
  auto CarryInIntReg = MRI->createVirtualRegister(IntRC);
  auto MIB =
      BuildMI(*BB, &I, DL, TII.get(HasCarryIn ? CarryOpc : NoCarryOpc), DstReg)
          .addDef(CarryOutPredReg);
  if (Src0C)
    MIB.addImm(Src0C->Value.getSExtValue());
  else
    MIB.add(I.getOperand(2));
  if (Src1C && !Src0C)
    MIB.addImm(Src1C->Value.getSExtValue());
  else
    MIB.add(I.getOperand(3));
  if (HasCarryIn)
    MIB.add(I.getOperand(4));

  // overflow is produced when MSB of arguments are same and
  // result of add/sub operation changes MSB value. Overflow
  // will occur if carry-in != carry-out for unsigned add/sub.
  BuildMI(*BB, &I, DL, TII.get(PISA::sel_16_iip), CarryOutIntReg)
      .addReg(CarryOutPredReg)
      .addImm(1)
      .addImm(0);
  if (HasCarryIn) {
    BuildMI(*BB, &I, DL, TII.get(PISA::sel_16_iip), CarryInIntReg)
        .add(I.getOperand(4))
        .addImm(1)
        .addImm(0);
    BuildMI(*BB, &I, DL, TII.get(PISA::ucmp_eq_16b_prr), CarryOutReg)
        .addReg(CarryOutIntReg)
        .addReg(CarryInIntReg);
  } else {
    BuildMI(*BB, &I, DL, TII.get(PISA::ucmp_eq_16b_pri), CarryOutReg)
        .addReg(CarryOutIntReg)
        .addImm(0);
  }

  if (!RBI.constrainGenericRegister(DstReg, *DstRC, *MRI) ||
      !RBI.constrainGenericRegister(CarryOutReg, *PredRC, *MRI) ||
      !RBI.constrainGenericRegister(Src0Reg, *DstRC, *MRI) ||
      !RBI.constrainGenericRegister(Src1Reg, *DstRC, *MRI) ||
      !RBI.constrainGenericRegister(CarryOutPredReg, *PredRC, *MRI) ||
      !RBI.constrainGenericRegister(CarryOutIntReg, *IntRC, *MRI) ||
      !RBI.constrainGenericRegister(CarryInIntReg, *IntRC, *MRI))
    return false;
  if (HasCarryIn &&
      !RBI.constrainGenericRegister(I.getOperand(4).getReg(), *PredRC, *MRI))
    return false;

  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectBitcast(MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  DebugLoc DL = I.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg = I.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);

  if (DstTy.getSizeInBits() != SrcTy.getSizeInBits()) {
    assert(false && "Wrong bitcast operands!");
    return false;
  }

  BuildMI(*BB, &I, DL, TII.get(TargetOpcode::COPY))
      .addDef(DstReg)
      .addReg(SrcReg);

  if (!RBI.constrainGenericRegister(SrcReg, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(DstReg, *DstRC, MRI))
    return false;

  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_MERGE_VALUES(MachineInstr &MI) const {
  MachineBasicBlock *BB = MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  const int NumSrc = MI.getNumOperands() - 1;

  auto [DstReg, DstTy, SrcReg, SrcTy] = MI.getFirst2RegLLTs();
  const unsigned SrcNumElts = SrcTy.isVector() ? SrcTy.getNumElements() : 1;
  const unsigned SrcEltSize = SrcTy.getScalarSizeInBits();

  // Subregister code below can only operate on vector destination.
  // - create bitcasts in case of scalar destination (G_MERGE_VALUES)
  if (!DstTy.isVector()) {
    const TargetRegisterClass *DstRC = TRI.getRegClassFromLLT(DstTy);
    if (!DstRC || !RBI.constrainGenericRegister(DstReg, *DstRC, *MRI))
      return false;

    auto DstNumElts = DstTy.getSizeInBits() / SrcTy.getScalarSizeInBits();
    DstTy = LLT::fixed_vector(DstNumElts, SrcTy.getScalarType());
    DstReg = MRI->createGenericVirtualRegister(DstTy);
  }

  assert(DstTy.isVector() && "expecting vector destination in merge_values");
  assert((NumSrc <= 4) && "expecting vector of 4 or less in merge_values");

  const TargetRegisterClass *DstRC = TRI.getRegClassFromLLT(DstTy);
  if (!DstRC || !RBI.constrainGenericRegister(DstReg, *DstRC, *MRI))
    return false;

  // G_CONCAT_VECTORS of sub-vectors that each map to a nameable composite
  // sub-register (.xy / .zw): write each source directly into its slice via
  // INSERT_SUBREG, instead of unpacking to scalars and rebuilding. Register
  // coalescing then folds each source's producer into the destination slice
  // (the write-side counterpart of the .xy/.zw read optimization).
  if (MI.getOpcode() == TargetOpcode::G_CONCAT_VECTORS && SrcTy.isVector()) {
    bool AllComposite = SrcNumElts > 0;
    for (int I = 0; I < NumSrc && AllComposite; ++I) {
      if (MRI->getType(MI.getOperand(I + 1).getReg()) != SrcTy)
        AllComposite = false;
      else if (unsigned Comp = TRI.getCompositeSubRegIdx(
                   SrcEltSize, I * SrcNumElts, SrcNumElts)) {
        if (!TRI.getSubClassWithSubReg(DstRC, Comp))
          AllComposite = false;
      } else
        AllComposite = false;
    }
    if (AllComposite) {
      // If all sources are consecutive outputs of a single G_UNMERGE_VALUES
      // whose source has the same type as our destination, the concat just
      // rebuilds the original register -- emit a COPY instead of INSERT_SUBREGs
      // to avoid a self-copy round-trip the coalescer cannot eliminate.
      bool IsIdentityConcat = false;
      if (NumSrc >= 2) {
        MachineInstr *FirstDef = MRI->getVRegDef(MI.getOperand(1).getReg());
        if (FirstDef &&
            FirstDef->getOpcode() == TargetOpcode::G_UNMERGE_VALUES) {
          unsigned NumUnmergeOuts = FirstDef->getNumOperands() - 1;
          Register UnmergeSrc = FirstDef->getOperand(NumUnmergeOuts).getReg();
          if ((int)NumUnmergeOuts == NumSrc &&
              MRI->getType(UnmergeSrc) == DstTy) {
            IsIdentityConcat = true;
            for (int I = 0; I < NumSrc && IsIdentityConcat; ++I) {
              MachineInstr *Def =
                  MRI->getVRegDef(MI.getOperand(I + 1).getReg());
              if (Def != FirstDef || MI.getOperand(I + 1).getReg() !=
                                         FirstDef->getOperand(I).getReg())
                IsIdentityConcat = false;
            }
          }
        }
      }
      if (IsIdentityConcat) {
        MachineInstr *FirstDef = MRI->getVRegDef(MI.getOperand(1).getReg());
        unsigned NumUnmergeOuts = FirstDef->getNumOperands() - 1;
        Register UnmergeSrc = FirstDef->getOperand(NumUnmergeOuts).getReg();
        BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::COPY), DstReg)
            .addReg(UnmergeSrc);
        if (!RBI.constrainGenericRegister(UnmergeSrc, *DstRC, *MRI))
          return false;
        MI.eraseFromParent();
        return true;
      }

      auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);
      Register CurVec = MRI->createVirtualRegister(DstRC);
      BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::IMPLICIT_DEF), CurVec);
      if (!RBI.constrainGenericRegister(CurVec, *DstRC, *MRI))
        return false;
      for (int I = 0; I < NumSrc; ++I) {
        unsigned Comp =
            TRI.getCompositeSubRegIdx(SrcEltSize, I * SrcNumElts, SrcNumElts);
        Register SrcReg = MI.getOperand(I + 1).getReg();
        if (!SrcRC || !RBI.constrainGenericRegister(SrcReg, *SrcRC, *MRI))
          return false;
        Register CurDst =
            (I + 1 == NumSrc) ? DstReg : MRI->createVirtualRegister(DstRC);
        BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::INSERT_SUBREG), CurDst)
            .addReg(CurVec)
            .addReg(SrcReg)
            .addImm(Comp);
        if (!RBI.constrainGenericRegister(CurDst, *DstRC, *MRI))
          return false;
        CurVec = CurDst;
      }
      MI.eraseFromParent();
      return true;
    }
  }

  SmallVector<Register> Regs;
  for (int I = 0, E = NumSrc; I != E; ++I) {
    MachineOperand &Src = MI.getOperand(I + 1);
    Register SrcReg = Src.getReg();

    if (SrcNumElts == 1) {
      const TargetRegisterClass *SrcRC = TRI.getRegClassFromLLT(SrcTy);
      if (!SrcRC || !RBI.constrainGenericRegister(SrcReg, *SrcRC, *MRI))
        return false;
      Regs.push_back(SrcReg);

    } else {
      auto *SubRC = TRI.getRegClassFromLLT(SrcTy.getScalarType());
      for (unsigned J = 0; J < SrcNumElts; J++) {
        auto SubRegIdx = TRI.getSubRegIdx(SrcEltSize, J);
        // Constrain SrcReg to the subclass supporting this subreg index
        // (needed when RC contains physicals without the subreg, e.g.
        // RegV2_16b with Reg32b members that lack sub16_N). Sources may not
        // have an RC set yet if their defining instruction hasn't been
        // selected (selection order is bottom-up), so derive from LLT.
        auto *SrcRC = MRI->getRegClassOrNull(SrcReg);
        if (!SrcRC)
          SrcRC = TRI.getRegClassFromLLT(SrcTy);
        if (SrcRC) {
          if (auto *NarrowRC = TRI.getSubClassWithSubReg(SrcRC, SubRegIdx))
            RBI.constrainGenericRegister(SrcReg, *NarrowRC, *MRI);
        }
        auto SubSrcReg =
            MRI->createGenericVirtualRegister(SrcTy.getScalarType());
        BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::COPY), SubSrcReg)
            .addReg(SrcReg, {}, SubRegIdx);
        if (!SubRC || !RBI.constrainGenericRegister(SubSrcReg, *SubRC, *MRI))
          return false;
        Regs.push_back(SubSrcReg);
      }
    }
  }
  if (!emitBuildVector(MI, DstReg, Regs))
    return false;

  // copy back the value in case of scalar destination
  auto OrigDstReg = MI.getOperand(0).getReg();
  if (!MRI->getType(OrigDstReg).isVector()) {
    BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::COPY), OrigDstReg)
        .addReg(DstReg);
  }

  MI.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_UNMERGE_VALUES(MachineInstr &MI) const {
  MachineBasicBlock *BB = MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  const int NumDst = MI.getNumOperands() - 1;

  Register SrcReg = MI.getOperand(NumDst).getReg();
  LLT SrcTy = MRI->getType(SrcReg);

  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI->getType(DstReg);
  const unsigned DstNumElts = DstTy.isVector() ? DstTy.getNumElements() : 1;
  const unsigned DstEltSize = DstTy.getScalarSizeInBits();

  // Subregister code below can only operate on vector sources.
  // - create bitcasts in case of scalar sources
  if (!SrcTy.isVector()) {
    const TargetRegisterClass *SrcRC = TRI.getRegClassFromLLT(SrcTy);
    if (!SrcRC || !RBI.constrainGenericRegister(SrcReg, *SrcRC, *MRI))
      return false;

    auto SrcNumElts = SrcTy.getSizeInBits() / DstTy.getScalarSizeInBits();
    SrcTy = LLT::fixed_vector(SrcNumElts, DstTy.getScalarType());
    auto NewSrcReg = MRI->createGenericVirtualRegister(SrcTy);
    BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::COPY), NewSrcReg)
        .addReg(SrcReg);
    SrcReg = NewSrcReg;
  }

  assert(SrcTy.isVector() && "expecting vector source in unmerge_values");
  assert((NumDst <= 4) && "expecting vector of 4 or less in unmerge_values");

  const TargetRegisterClass *SrcRC = TRI.getRegClassFromLLT(SrcTy);
  if (!SrcRC || !RBI.constrainGenericRegister(SrcReg, *SrcRC, *MRI))
    return false;

  for (int I = 0, E = NumDst; I != E; ++I) {
    Register DstReg = MI.getOperand(I).getReg();
    SmallVector<Register> Regs;
    for (unsigned J = 0; J < DstNumElts; J++) {
      auto SubRegIdx = TRI.getSubRegIdx(DstEltSize, I * DstNumElts + J);
      // Constrain SrcReg to the subclass supporting this subreg index.
      if (auto *CurSrcRC = MRI->getRegClassOrNull(SrcReg)) {
        if (auto *NarrowRC = TRI.getSubClassWithSubReg(CurSrcRC, SubRegIdx))
          MRI->constrainRegClass(SrcReg, NarrowRC);
      }
      auto SubDstReg =
          (DstNumElts == 1)
              ? DstReg
              : MRI->createGenericVirtualRegister(DstTy.getScalarType());
      BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::COPY), SubDstReg)
          .addReg(SrcReg, {}, SubRegIdx);
      Regs.push_back(SubDstReg);
    }

    if (DstNumElts > 1) {
      auto *SubRC = TRI.getRegClassFromLLT(DstTy.getScalarType());
      for (Register R : Regs) {
        if (!RBI.constrainGenericRegister(R, *SubRC, *MRI))
          return false;
      }
      if (!emitBuildVector(MI, DstReg, Regs))
        return false;
    }

    const TargetRegisterClass *DstRC = TRI.getRegClassFromLLT(DstTy);
    if (DstRC && !RBI.constrainGenericRegister(DstReg, *DstRC, *MRI))
      return false;
  }

  MI.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectConvPtrInt(MachineInstr &I) const {
  // Custom select G_PTRTOINT and G_INTTOPTR
  MachineBasicBlock *BB = I.getParent();
  DebugLoc DL = I.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg = I.getOperand(1).getReg();

  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);

  assert(DstTy.getSizeInBits() == SrcTy.getSizeInBits() &&
         "mismatch in int2ptr src/dst sizes");
  BuildMI(*BB, &I, DL, TII.get(TargetOpcode::COPY))
      .addDef(DstReg)
      .addReg(SrcReg);

  if (!RBI.constrainGenericRegister(SrcReg, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(DstReg, *DstRC, MRI))
    return false;

  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectCopy(MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  DebugLoc DL = I.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg = I.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);

  if (auto *SrcDefI = MRI.getVRegDef(SrcReg);
      SrcDefI && SrcDefI->isInlineAsm() && !SrcTy.isValid()) {
    assert(DstTy.isValid());
    SrcTy = DstTy;
  } else if (MRI.hasOneNonDBGUse(DstReg) &&
             MRI.use_instr_nodbg_begin(DstReg)->isInlineAsm()) {
    assert(SrcTy.isValid());
    DstTy = SrcTy;
  }

  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);

  if (DstTy.getSizeInBits() != SrcTy.getSizeInBits()) {
    assert(false && "Operands with different bit size!");
    return false;
  }

  BuildMI(*BB, &I, DL, TII.get(TargetOpcode::COPY))
      .addDef(DstReg)
      .addReg(SrcReg);

  if (!RBI.constrainGenericRegister(SrcReg, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(DstReg, *DstRC, MRI))
    return false;

  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::emitBuildVector(MachineInstr &InsertPt,
                                              Register DstReg,
                                              ArrayRef<Register> Srcs) const {
  assert(!Srcs.empty());
  MachineBasicBlock *BB = InsertPt.getParent();
  DebugLoc DL = InsertPt.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  // A sequence of INSERT_SUBREG's that are later lowered
  // to COPY's is generated.

  const uint32_t NumSrcs = Srcs.size();

  LLT DstTy = MRI.getType(DstReg);
  LLT EltTy = MRI.getType(Srcs[0]);
  const unsigned SrcSize = EltTy.getSizeInBits();
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);

  Register CurVecReg = MRI.createVirtualRegister(DstRC);
  BuildMI(*BB, &InsertPt, InsertPt.getDebugLoc(),
          TII.get(TargetOpcode::IMPLICIT_DEF), CurVecReg);

  if (!RBI.constrainGenericRegister(CurVecReg, *DstRC, MRI))
    return false;

  // Use the large-vector insert path when the class has no per-element
  // sub-registers (e.g. RegV64_32b).
  bool HasSubRegSupport =
      TRI.getSubClassWithSubReg(DstRC, TRI.getSubRegIdx(SrcSize, 0));
  if (!HasSubRegSupport || Srcs.size() > 4) {
    assert(EltTy.getScalarSizeInBits() == 32);
    for (auto [Index, EltReg] : llvm::enumerate(Srcs)) {
      // lookup target opcode
      std::string OpcodeName = "insert_" + std::to_string(Index) + "_v" +
                               std::to_string(DstTy.getNumElements()) + "i" +
                               std::to_string(DstTy.getScalarSizeInBits()) +
                               "_i32_r";
      auto *Entry = PISA::lookupName2InstrOpEntry(OpcodeName);
      assert(Entry && "unable to find insert instruction");
      auto Opcode = Entry->Opcode;

      Register CurDstReg =
          (Index + 1 == NumSrcs) ? DstReg : MRI.createVirtualRegister(DstRC);
      BuildMI(*BB, &InsertPt, DL, TII.get(Opcode), CurDstReg)
          .addReg(CurVecReg)
          .addReg(EltReg);

      if (!RBI.constrainGenericRegister(CurDstReg, *DstRC, MRI))
        return false;
      CurVecReg = CurDstReg;
    }
  } else {
    // Narrow DstRC to the subclass supporting the subreg indices we'll use
    // (needed when RC contains physicals without those subregs, e.g.
    // RegV2_16b with Reg32b members that lack sub16_N).
    if (NumSrcs > 0) {
      unsigned FirstSubRegIdx = TRI.getSubRegIdx(SrcSize, 0);
      if (auto *NarrowRC = TRI.getSubClassWithSubReg(DstRC, FirstSubRegIdx)) {
        DstRC = NarrowRC;
        MRI.constrainRegClass(CurVecReg, DstRC);
      }
    }
    for (auto [Idx, EltReg] : llvm::enumerate(Srcs)) {
      unsigned SubRegIdx = TRI.getSubRegIdx(SrcSize, Idx);
      Register CurDstReg =
          (Idx + 1 == NumSrcs) ? DstReg : MRI.createVirtualRegister(DstRC);

      // skip building INSERT_SUBREG if element is undef, since element
      // is already undef in current vector
      if (!getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF, EltReg, MRI)) {
        BuildMI(*BB, &InsertPt, DL, TII.get(TargetOpcode::INSERT_SUBREG),
                CurDstReg)
            .addReg(CurVecReg)
            .addReg(EltReg)
            .addImm(SubRegIdx);
      } else if (CurDstReg == DstReg) {
        BuildMI(*BB, &InsertPt, DL, TII.get(TargetOpcode::COPY))
            .addDef(CurDstReg)
            .addReg(CurVecReg);
      } else {
        continue;
      }

      if (!RBI.constrainGenericRegister(CurDstReg, *DstRC, MRI))
        return false;

      CurVecReg = CurDstReg;
    }
  }

  return true;
}

bool PISAInstructionSelector::selectG_BUILD_VECTOR(MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  DebugLoc DL = I.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  bool AllUndef = true;

  SmallVector<Register> SrcRegs;
  uint32_t LastDefinedIdx = 1;
  for (uint32_t It = 1; It < I.getNumOperands(); It++) {
    auto SrcReg = I.getOperand(It).getReg();
    MachineInstr *Def = getDefIgnoringCopies(SrcReg, MRI);
    if (Def->getOpcode() != TargetOpcode::G_IMPLICIT_DEF) {
      AllUndef = false;
      LastDefinedIdx = It;
    }
    SrcRegs.push_back(SrcReg);
  }

  if (SrcRegs.size() > LastDefinedIdx)
    SrcRegs.resize(LastDefinedIdx);

  Register DstReg = I.getOperand(0).getReg();
  if (AllUndef) {
    // replace vector of IMPLICIT_DEFs with a single IMPLICIT_DEF
    BuildMI(*BB, &I, DL, TII.get(TargetOpcode::IMPLICIT_DEF)).addDef(DstReg);
    auto *DstRC = TRI.getRegClassFromLLT(MRI.getType(DstReg));
    if (!RBI.constrainGenericRegister(DstReg, *DstRC, MRI))
      return false;
  } else if (!emitBuildVector(I, DstReg, SrcRegs))
    return false;

  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectDbgValue(MachineInstr &I) const {
  // Retain DBG_VALUE MIR opcode, but make it target specific by
  // adding a phony operand. This way, AsmPrinter won't print it
  // as target independent instruction. If we don't add the phony
  // operand, AsmPrinter thinks it's the machine independent
  // version so it print it itself and never invokes PISAAsmPrinter.
  // If we select DBG_VALUE to a custom pseudo op then other
  // passes need to be taught to not mark the pseudo op as "dead".
  // To keep it simple, therefore, we keep DBG_VALUE operation
  // but attach a phony operand. This is still legal because
  // DBG_VALUE is treated as a variadic operation.
  I.addOperand(*I.getParent()->getParent(), MachineOperand::CreateImm(1));
  return true;
}

bool PISAInstructionSelector::selectG_FCANONICALIZE(MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  DebugLoc DL = I.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg = I.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);
  auto BitSize = DstTy.getScalarSizeInBits();

  // https://llvm.org/docs/LangRef.html#i-intr-llvm-canonicalize
  auto &Ctx = I.getMF()->getFunction().getContext();
  auto *CFP = ConstantFP::get(Ctx, getAPFloatFromSize(1.0, BitSize));

  unsigned Opcode;
  switch (BitSize) {
  case 16:
    Opcode = PISA::fmul_hf_ri;
    break;
  case 32:
    Opcode = PISA::fmul_f_ri;
    break;
  case 64:
    Opcode = PISA::fmul_df_ri;
    break;
  default:
    llvm_unreachable("illegal dst type for fcanonicalize");
  }
  BuildMI(*BB, &I, DL, TII.get(Opcode))
      .addDef(DstReg)
      .addReg(SrcReg)
      .addFPImm(CFP);

  if (!RBI.constrainGenericRegister(SrcReg, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(DstReg, *DstRC, MRI))
    return false;

  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_LROUND(MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  DebugLoc DL = I.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg = I.getOperand(1).getReg();

  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);
  auto SrcSize = SrcTy.getScalarSizeInBits();
  auto DstSize = DstTy.getScalarSizeInBits();
  assert((SrcSize == 32 || SrcSize == 64) && "illegal Src type for lround");
  assert((DstSize == 32 || DstSize == 64) && "illegal Dst type for lround");

  auto &Ctx = I.getMF()->getFunction().getContext();
  auto *FPHalf = ConstantFP::get(Ctx, getAPFloatFromSize(0.5, SrcSize));
  auto *FPZero = ConstantFP::get(Ctx, getAPFloatFromSize(0.0, SrcSize));

  unsigned F2i[/*src/64*/ 2][/*dst/64*/ 2] = {
      {PISA::f2i_u32_f_rz_r, PISA::f2i_u64_f_rz_r},
      {PISA::f2i_u32_df_rz_r, PISA::f2i_u64_df_rz_r},
  };

  // https://en.cppreference.com/w/cpp/numeric/math/round
  // => lrint(copysign(0.5 + fabs(x), x));
  Register FAbsReg = MRI.createVirtualRegister(SrcRC);
  Register FAddReg = MRI.createVirtualRegister(SrcRC);
  Register FNegReg = MRI.createVirtualRegister(SrcRC);
  Register FSelReg = MRI.createVirtualRegister(SrcRC);

  auto *CmpRC = TRI.getRegClassFromLLT(LLT::integer(1));
  Register CmpReg = MRI.createVirtualRegister(CmpRC);

  unsigned Opcode;
  Opcode = (SrcSize == 64) ? PISA::fabs_df_r : PISA::fabs_f_r;
  BuildMI(*BB, &I, DL, TII.get(Opcode)).addDef(FAbsReg).addReg(SrcReg);
  Opcode = (SrcSize == 64) ? PISA::fadd_df_ri : PISA::fadd_f_ri;
  BuildMI(*BB, &I, DL, TII.get(Opcode))
      .addDef(FAddReg)
      .addReg(FAbsReg)
      .addFPImm(FPHalf);
  Opcode = (SrcSize == 64) ? PISA::fneg_df_r : PISA::fneg_f_r;
  BuildMI(*BB, &I, DL, TII.get(Opcode)).addDef(FNegReg).addReg(FAddReg);
  Opcode = (SrcSize == 64) ? PISA::fcmp_lt_df_pri : PISA::fcmp_lt_f_pri;
  BuildMI(*BB, &I, DL, TII.get(Opcode))
      .addDef(CmpReg)
      .addReg(SrcReg)
      .addFPImm(FPZero);
  Opcode = (SrcSize == 64) ? PISA::sel_df_rrp : PISA::sel_f_rrp;
  BuildMI(*BB, &I, DL, TII.get(Opcode))
      .addDef(FSelReg)
      .addReg(CmpReg)
      .addReg(FNegReg)
      .addReg(FAddReg);
  Opcode = F2i[SrcSize / 64][DstSize / 64];
  BuildMI(*BB, &I, DL, TII.get(Opcode)).addDef(DstReg).addReg(FSelReg);

  if (!RBI.constrainGenericRegister(SrcReg, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(DstReg, *DstRC, MRI))
    return false;
  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_BFX(MachineInstr &I, bool Signed) const {
  MachineBasicBlock *BB = I.getParent();
  DebugLoc DL = I.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  auto [Dst, Src0, Src1, Src2] = I.getFirst4Regs();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src0);
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);
  assert((SrcTy.getScalarSizeInBits() == 32) && "illegal Src type for bfe");
  assert((DstTy.getScalarSizeInBits() == 32) && "illegal Dst type for bfe");

  // Res = G_UBFX (Base,LSB,Width)
  // Res = bfe (Base,Width,Offset)
  unsigned OpcodesU[] = {PISA::ubfe_32b_rrr, PISA::ubfe_32b_rri,
                         PISA::ubfe_32b_rir, PISA::ubfe_32b_rii,
                         PISA::ubfe_32b_irr, PISA::ubfe_32b_iri,
                         PISA::ubfe_32b_iir, PISA::ubfe_32b_iii};
  unsigned OpcodesS[] = {PISA::sbfe_32b_rrr, PISA::sbfe_32b_rri,
                         PISA::sbfe_32b_rir, PISA::sbfe_32b_rii,
                         PISA::sbfe_32b_irr, PISA::sbfe_32b_iri,
                         PISA::sbfe_32b_iir, PISA::sbfe_32b_iii};
  auto &Opcodes = Signed ? OpcodesS : OpcodesU;
  auto WidthImm = getIConstantVRegValWithLookThrough(Src2, MRI);
  auto OffsetImm = getIConstantVRegValWithLookThrough(Src1, MRI);
  auto SourceImm = getIConstantVRegValWithLookThrough(Src0, MRI);
  unsigned OpIdx = SourceImm ? 4 : 0;
  if (WidthImm)
    OpIdx += 2;
  if (OffsetImm)
    OpIdx += 1;
  auto MIB = BuildMI(*BB, &I, DL, TII.get(Opcodes[OpIdx])).addDef(Dst);
  if (SourceImm) {
    auto SourceImmValue = Signed ? SourceImm->Value.getSExtValue()
                                 : SourceImm->Value.getZExtValue();
    MIB.addImm(SourceImmValue);
  } else {
    MIB.addReg(Src0);
  }
  if (WidthImm) {
    auto WidthImmValue = Signed ? WidthImm->Value.getSExtValue()
                                : WidthImm->Value.getZExtValue();
    MIB.addImm(WidthImmValue);
  } else {
    MIB.addReg(Src2);
  }
  if (OffsetImm) {
    auto OffsetImmValue = Signed ? OffsetImm->Value.getSExtValue()
                                 : OffsetImm->Value.getZExtValue();
    MIB.addImm(OffsetImmValue);
  } else {
    MIB.addReg(Src1);
  }

  if (!RBI.constrainGenericRegister(Src0, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(Src1, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(Src2, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(Dst, *DstRC, MRI))
    return false;
  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_CONSTANT(MachineInstr &I) const {
  MachineBasicBlock *BB = I.getParent();
  DebugLoc DL = I.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  auto Dst = I.getOperand(0).getReg();
  auto Src = I.getOperand(1).getCImm()->getZExtValue();
  LLT DstTy = MRI.getType(Dst);
  unsigned DstSize = DstTy.getScalarSizeInBits();

  // For non-boolean constants (e.g. G_CONSTANT with a float dest type that
  // selectImpl couldn't match), emit a MOV immediate directly.
  if (DstSize != 1) {
    unsigned MovOpc;
    switch (DstSize) {
    case 8:
      MovOpc = PISA::mov_i8_i;
      break;
    case 16:
      MovOpc = PISA::mov_i16_i;
      break;
    case 32:
      MovOpc = PISA::mov_i32_i;
      break;
    case 64:
      MovOpc = PISA::mov_i64_i;
      break;
    default:
      LLVM_DEBUG(errs() << "Unsupported G_CONSTANT size: " << DstSize << "\n");
      return false;
    }
    auto *DstRC = TRI.getRegClassFromLLT(LLT::integer(DstSize));
    BuildMI(*BB, &I, DL, TII.get(MovOpc), Dst).addImm(Src);
    if (!RBI.constrainGenericRegister(Dst, *DstRC, MRI))
      return false;
    I.eraseFromParent();
    return true;
  }

  auto *DstRC = TRI.getRegClassFromLLT(DstTy);

  auto *ConstRC = TRI.getRegClassFromLLT(LLT::integer(32));
  Register ConstReg = MRI.createVirtualRegister(ConstRC);
  BuildMI(*BB, &I, DL, TII.get(PISA::mov_i32_i), ConstReg).addImm(0);

  unsigned Opcodes[] = {PISA::ucmp_ne_32b_prr, PISA::ucmp_eq_32b_prr};
  BuildMI(*BB, &I, DL, TII.get(Opcodes[Src & 1]), Dst)
      .addReg(ConstReg)
      .addReg(ConstReg);

  if (!RBI.constrainGenericRegister(ConstReg, *ConstRC, MRI) ||
      !RBI.constrainGenericRegister(Dst, *DstRC, MRI))
    return false;
  I.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_FENCE(MachineInstr &MI) const {
  MachineBasicBlock *BB = MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  // G_FENCE has two operands:
  // Operand 0: AtomicOrdering
  // Operand 1: SyncScope
  unsigned MemOrder = MI.getOperand(0).getImm();
  SyncScope::ID Ord = SyncScope::ID(MI.getOperand(1).getImm());

  // HW has no system-generic fence; split into shared + global fences
  if (Ord == SSI.SystemGenericID) {
    BuildMI(*BB, &MI, DL, TII.get(PISA::fence_shared_gpu)).addImm(MemOrder);
    BuildMI(*BB, &MI, DL, TII.get(PISA::fence_global_system)).addImm(MemOrder);
    MI.eraseFromParent();
    return true;
  }

  auto Entry = SSI.ID2Opcode.find(Ord);
  if (Entry == SSI.ID2Opcode.end())
    llvm_unreachable("unimplemented fence syncscope");

  if (Entry->second == PISA::fence_subgroup)
    BuildMI(*BB, &MI, DL, TII.get(Entry->second));
  else
    BuildMI(*BB, &MI, DL, TII.get(Entry->second)).addImm(MemOrder);
  MI.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_INSERT_SUBVECTOR(MachineInstr &MI) const {
  MachineBasicBlock *BB = MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  auto [Dst, DstTy, Src0, Src0Ty, Src1, Src1Ty] = MI.getFirst3RegLLTs();
  assert(DstTy.getScalarSizeInBits() == 32);
  auto Index = MI.getOperand(3).getImm();
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *Src1RC = TRI.getRegClassFromLLT(Src1Ty);

  // lookup target opcode
  std::string OpcodeName = "insert_" + std::to_string(Index) + "_v" +
                           std::to_string(DstTy.getNumElements()) + "i" +
                           std::to_string(DstTy.getScalarSizeInBits()) + "_v" +
                           std::to_string(Src1Ty.getNumElements()) + "i" +
                           std::to_string(Src1Ty.getScalarSizeInBits()) + "_r";
  auto *Entry = PISA::lookupName2InstrOpEntry(OpcodeName);
  assert(Entry && "unable to find insert instruction");
  auto Opcode = Entry->Opcode;

  BuildMI(*BB, &MI, DL, TII.get(Opcode), Dst).addReg(Src0).addReg(Src1);

  if (!RBI.constrainGenericRegister(Src0, *DstRC, MRI) ||
      !RBI.constrainGenericRegister(Src1, *Src1RC, MRI) ||
      !RBI.constrainGenericRegister(Dst, *DstRC, MRI))
    return false;

  MI.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_EXTRACT_SUBVECTOR(
    MachineInstr &MI) const {
  MachineBasicBlock *BB = MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  auto [Dst, DstTy, Src, SrcTy] = MI.getFirst2RegLLTs();
  auto Index = MI.getOperand(2).getImm();
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);

  // If the extracted sub-vector is a nameable composite sub-register slice of
  // the source (e.g. lanes 0,1 -> .xy or lanes 2,3 -> .zw of a 4-lane vector),
  // emit a single sub-register COPY. Register coalescing folds it so the
  // consumer addresses the slice directly, without gather copies. This mirrors
  // how the low .xy half is already handled and extends it to .zw.
  // The .xy/.zw composites are only defined on 4-lane classes (Reg*bx4); on
  // wider sources (v5-v8, ...) they are not real sub-registers and must not be
  // used, so restrict to <=4-lane sources.
  unsigned CompIdx =
      (SrcTy.isVector() && SrcTy.getNumElements() <= 4)
          ? TRI.getCompositeSubRegIdx(DstTy.getScalarSizeInBits(), Index,
                                      DstTy.getNumElements())
          : 0;
  if (CompIdx && TRI.getSubClassWithSubReg(SrcRC, CompIdx)) {
    BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::COPY), Dst)
        .addReg(Src, {}, CompIdx);
    if (!RBI.constrainGenericRegister(Src, *SrcRC, MRI) ||
        !RBI.constrainGenericRegister(Dst, *DstRC, MRI))
      return false;
    MI.eraseFromParent();
    return true;
  }

  assert(DstTy.getScalarSizeInBits() == 32);

  // lookup target opcode
  std::string OpcodeName = "extract_" + std::to_string(Index) + "_v" +
                           std::to_string(DstTy.getNumElements()) + "i" +
                           std::to_string(DstTy.getScalarSizeInBits()) + "_v" +
                           std::to_string(SrcTy.getNumElements()) + "i" +
                           std::to_string(SrcTy.getScalarSizeInBits()) + "_r";
  auto *Entry = PISA::lookupName2InstrOpEntry(OpcodeName);
  assert(Entry && "unable to find insert instruction");
  auto Opcode = Entry->Opcode;

  BuildMI(*BB, &MI, DL, TII.get(Opcode), Dst).addReg(Src);

  if (!RBI.constrainGenericRegister(Src, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(Dst, *DstRC, MRI))
    return false;

  MI.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_INSERT_VECTOR_ELT(
    MachineInstr &MI) const {
  MachineBasicBlock *BB = MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  auto [Dst, DstTy, Src0, Src0Ty, Src1, Src1Ty, Idx, IdxTy] =
      MI.getFirst4RegLLTs();
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *Src1RC = TRI.getRegClassFromLLT(Src1Ty);
  auto *IdxRC = TRI.getRegClassFromLLT(IdxTy);

  // Check if Index is a constant
  auto IndexConst = getIConstantVRegValWithLookThrough(Idx, MRI);

  if (IndexConst) {
    // Handle constant index case
    auto Index = IndexConst->Value.getZExtValue();

    // Use insert instruction for index >= 4 or when the dest register
    // class lacks per-element sub-register structure (e.g. v64).
    if (Index >= 4 || DstTy.getNumElements() > 32) {
      assert(DstTy.getScalarSizeInBits() == 32);
      std::string OpcodeName =
          "insert_" + std::to_string(Index) + "_v" +
          std::to_string(DstTy.getNumElements()) + "i" +
          std::to_string(DstTy.getScalarSizeInBits()) + "_i" +
          std::to_string(Src1Ty.getScalarSizeInBits()) + "_r";
      auto *Entry = PISA::lookupName2InstrOpEntry(OpcodeName);
      assert(Entry && "unable to find insert instruction");
      auto Opcode = Entry->Opcode;

      BuildMI(*BB, &MI, DL, TII.get(Opcode), Dst).addReg(Src0).addReg(Src1);
    } else { // use swizzle
      auto SubRegIdx = TRI.getSubRegIdx(DstTy.getScalarSizeInBits(), Index);
      // Constrain Src0 (the vector input) and Dst to support the subreg.
      if (auto *NarrowRC = TRI.getSubClassWithSubReg(DstRC, SubRegIdx))
        DstRC = NarrowRC;
      BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::INSERT_SUBREG), Dst)
          .addReg(Src0)
          .addReg(Src1)
          .addImm(SubRegIdx);
    }
  } else {
    // Handle dynamic index case - use insert.dynamic instruction
    assert(DstTy.getScalarSizeInBits() == 32);
    std::string OpcodeName = "insert_dynamic_v" +
                             std::to_string(DstTy.getNumElements()) + "i" +
                             std::to_string(DstTy.getScalarSizeInBits());
    auto *Entry = PISA::lookupName2InstrOpEntry(OpcodeName);
    assert(Entry && "unable to find insert.dynamic instruction");
    auto Opcode = Entry->Opcode;

    BuildMI(*BB, &MI, DL, TII.get(Opcode), Dst)
        .addReg(Src0)
        .addReg(Src1)
        .addReg(Idx);
  }

  if (!RBI.constrainGenericRegister(Src0, *DstRC, MRI) ||
      !RBI.constrainGenericRegister(Src1, *Src1RC, MRI) ||
      !RBI.constrainGenericRegister(Dst, *DstRC, MRI) ||
      !RBI.constrainGenericRegister(Idx, *IdxRC, MRI))
    return false;
  MI.eraseFromParent();
  return true;
}

bool PISAInstructionSelector::selectG_EXTRACT_VECTOR_ELT(
    MachineInstr &MI) const {
  MachineBasicBlock *BB = MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  auto [Dst, DstTy, Src, SrcTy, Idx, IdxTy] = MI.getFirst3RegLLTs();
  auto *DstRC = TRI.getRegClassFromLLT(DstTy);
  auto *SrcRC = TRI.getRegClassFromLLT(SrcTy);
  auto *IdxRC = TRI.getRegClassFromLLT(IdxTy);

  // Check if Index is a constant
  auto IndexConst = getIConstantVRegValWithLookThrough(Idx, MRI);

  if (IndexConst) {
    // Handle constant index case
    auto Index = IndexConst->Value.getZExtValue();

    // Use extract instruction for index >= 4 or when the source register
    // class lacks per-element sub-register structure (e.g. v64).
    if (Index >= 4 || SrcTy.getNumElements() > 32) {
      assert(DstTy.getScalarSizeInBits() == 32);
      std::string OpcodeName =
          "extract_" + std::to_string(Index) + "_i" +
          std::to_string(DstTy.getScalarSizeInBits()) + "_v" +
          std::to_string(SrcTy.getNumElements()) + "i" +
          std::to_string(SrcTy.getScalarSizeInBits()) + "_r";
      auto *Entry = PISA::lookupName2InstrOpEntry(OpcodeName);
      assert(Entry && "unable to find insert instruction");
      auto Opcode = Entry->Opcode;

      BuildMI(*BB, &MI, DL, TII.get(Opcode), Dst).addReg(Src);
    } else { // use swizzle
      auto SubRegIdx = TRI.getSubRegIdx(SrcTy.getScalarSizeInBits(), Index);
      // Constrain Src to the subclass supporting this subreg index.
      if (auto *CurSrcRC = MRI.getRegClassOrNull(Src)) {
        if (auto *NarrowRC = TRI.getSubClassWithSubReg(CurSrcRC, SubRegIdx))
          MRI.constrainRegClass(Src, NarrowRC);
      }
      BuildMI(*BB, &MI, DL, TII.get(TargetOpcode::COPY))
          .addDef(Dst)
          .addReg(Src, {}, SubRegIdx);
    }
  } else {
    std::string OpcodeName = "extract_dynamic_v" +
                             std::to_string(SrcTy.getNumElements()) + "i" +
                             std::to_string(SrcTy.getScalarSizeInBits());
    auto *Entry = PISA::lookupName2InstrOpEntry(OpcodeName);
    assert(Entry && "unable to find extract.dynamic instruction");
    auto Opcode = Entry->Opcode;

    BuildMI(*BB, &MI, DL, TII.get(Opcode), Dst).addReg(Src).addReg(Idx);
  }

  if (!RBI.constrainGenericRegister(Src, *SrcRC, MRI) ||
      !RBI.constrainGenericRegister(Dst, *DstRC, MRI) ||
      !RBI.constrainGenericRegister(Idx, *IdxRC, MRI))
    return false;
  MI.eraseFromParent();
  return true;
}

InstructionSelector::ComplexRendererFns
PISAInstructionSelector::SelectAddr_ri(MachineOperand &Root) const {
  Register Addr = Root.getReg();
  Register Base;
  int64_t Offset;
  if (!mi_match(Addr, *MRI, m_GPtrAdd(m_Reg(Base), m_ICst(Offset)))) {
    Base = Addr;
    Offset = 0;
  }
  MachineInstr *BaseDef = getDefIgnoringCopies(Base, *MRI);
  assert(BaseDef && "unexpected no definition for base register");

  // If the first operand of the def is not a register, it means it's not a
  // G_PTR_ADD. We can directly use the Base and Offset to render the address.
  if (!BaseDef->getOperand(0).isReg())
    return {{[=](MachineInstrBuilder &MIB) { MIB.addReg(Base); },
             [=](MachineInstrBuilder &MIB) { MIB.addImm(Offset); }}};

  if (BaseDef->getOpcode() == TargetOpcode::G_FRAME_INDEX)
    return {{[=](MachineInstrBuilder &MIB) { MIB.add(BaseDef->getOperand(1)); },
             [=](MachineInstrBuilder &MIB) { MIB.addImm(Offset); }}};

  // %7  = G_CONSTANT i32 4
  // %6  = G_PTR_ADD %0, %7
  // %18 = G_CONSTANT i32 2
  // %17 = G_PTR_ADD %6, %18
  // => %17 = G_PTR_ADD %0, (%7 + %18)
  {
    MachineOperand &Op = BaseDef->getOperand(0);
    assert(Op.isReg() && "unexpected non-register operand for G_PTR_ADD");
    Register OAddr = Op.getReg();
    Register OBase = Base;
    int64_t OOffset = Offset;

    while (mi_match(OAddr, *MRI, m_GPtrAdd(m_Reg(OBase), m_ICst(OOffset)))) {
      BaseDef = getDefIgnoringCopies(OBase, *MRI);
      assert(BaseDef && "unexpected no definition for base register");
      if (BaseDef->getOpcode() == TargetOpcode::G_FRAME_INDEX)
        break;
      Base = OBase;
      Offset += OOffset;
      OAddr = BaseDef->getOperand(0).getReg();
    }
  }

  // Assume that _rr is selected when the offset doesn't fit in 32-bit
  assert(isInt<32>(Offset) && "unexpected offset that doesn't fit in 32-bit");

  if (BaseDef->getOpcode() == TargetOpcode::G_FRAME_INDEX)
    return {{[=](MachineInstrBuilder &MIB) { MIB.add(BaseDef->getOperand(1)); },
             [=](MachineInstrBuilder &MIB) { MIB.addImm(Offset); }}};

  return {{[=](MachineInstrBuilder &MIB) { MIB.addReg(Base); },
           [=](MachineInstrBuilder &MIB) { MIB.addImm(Offset); }}};
}

InstructionSelector::ComplexRendererFns
PISAInstructionSelector::SelectAddr_rr(MachineOperand &Root) const {
  Register BaseReg;
  int64_t ImmOff;
  // Skip base-plus-immediate chains that fit the signed 32-bit displacement
  // accepted by memory operands. Larger constants are matched here so the
  // offset remains in a register instead of being expanded into an add chain.
  Register Addr = Root.getReg();
  int64_t TotalImmOff = 0;
  while (mi_match(Addr, *MRI, m_GPtrAdd(m_Reg(BaseReg), m_ICst(ImmOff)))) {
    TotalImmOff += ImmOff;
    Addr = BaseReg;
  }
  if (TotalImmOff != 0 && isInt<32>(TotalImmOff))
    return std::nullopt;

  MachineInstr *MI = getOpcodeDef(TargetOpcode::G_PTR_ADD, Root.getReg(), *MRI);
  if (!MI)
    return std::nullopt;

  Register Base = MI->getOperand(1).getReg();
  Register Offset = MI->getOperand(2).getReg();
  MachineInstr *Def = getDefIgnoringCopies(Base, *MRI);

  if (Def->getOpcode() == TargetOpcode::G_FRAME_INDEX) {
    return {{
        [=](MachineInstrBuilder &MIB) { MIB.add(Def->getOperand(1)); },
        [=](MachineInstrBuilder &MIB) {
          MIB.addReg(Offset);
        } // [FrameIndex + $Offset]
    }};
  }
  return {{
      [=](MachineInstrBuilder &MIB) { MIB.addReg(Base); },
      [=](MachineInstrBuilder &MIB) { MIB.addReg(Offset); } // [$Base + $Offset]
  }};
}

InstructionSelector::ComplexRendererFns
PISAInstructionSelector::selectParamSlot_ii(MachineOperand &Root) const {
  Register Addr = Root.getReg();
  Register Base;
  MachineInstr *BaseDef = getDefIgnoringCopies(Addr, *MRI);
  int64_t Offset = 0, Cst;
  while (mi_match(Addr, *MRI, m_GPtrAdd(m_Reg(Base), m_ICst(Cst)))) {
    BaseDef = getDefIgnoringCopies(Base, *MRI);
    Offset += Cst;
    if (!BaseDef || BaseDef->getOpcode() == PISA::G_PISA_PARAM_SLOT)
      break;
    if (BaseDef->getOpcode() != TargetOpcode::G_PTR_ADD)
      return std::nullopt;
    Addr = BaseDef->getOperand(0).getReg();
  }

  if (!BaseDef || BaseDef->getOpcode() != PISA::G_PISA_PARAM_SLOT)
    return std::nullopt;

  unsigned Slot = BaseDef->getOperand(1).getImm();
  const char *ArgName = nullptr;
  if (BaseDef->getNumOperands() > 3 && BaseDef->getOperand(3).isSymbol())
    ArgName = BaseDef->getOperand(3).getSymbolName();
  return {{[=](MachineInstrBuilder &MIB) { MIB.addImm(Slot); },
           [=](MachineInstrBuilder &MIB) {
             MIB.addImm(Offset);
             if (ArgName)
               MIB.addExternalSymbol(ArgName);
           }}};
}

InstructionSelector::ComplexRendererFns
PISAInstructionSelector::selectParamSlot_ir(MachineOperand &Root) const {
  MachineInstr *MI = Root.getParent();
  MachineBasicBlock *BB = MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  Register Addr = getSrcRegIgnoringCopies(Root.getReg(), *MRI);
  Register Base, OffsetReg;
  MachineInstr *BaseDef = nullptr;
  // Support matching a chain of G_PTR_ADD to retrieve the base. The offset is
  // also calculated when visiting a G_PTR_ADD in the chain.
  // TODO: Support matching other possible translated getelementptr sequence.
  SmallVector<Register> OffsetRegs;
  while (mi_match(Addr, *MRI, m_GPtrAdd(m_Reg(Base), m_Reg(OffsetReg)))) {
    OffsetRegs.push_back(OffsetReg);
    BaseDef = getDefIgnoringCopies(Base, *MRI);
    if (!BaseDef || BaseDef->getOpcode() == PISA::G_PISA_PARAM_SLOT)
      break;
    if (BaseDef->getOpcode() != TargetOpcode::G_PTR_ADD)
      return std::nullopt;
    Addr = BaseDef->getOperand(0).getReg();
  }
  assert(!BaseDef || BaseDef->getOpcode() == PISA::G_PISA_PARAM_SLOT);
  if (!BaseDef)
    return std::nullopt;

  // Create instructions to re-calculate offset after a match is found.
  assert(!OffsetRegs.empty());
  OffsetReg = OffsetRegs[0];
  unsigned AddOpcodes[] = {PISA::iadd_64b_rr, PISA::iadd_64b_ri,
                           PISA::iadd_64b_ir, PISA::iadd_64b_ii};
  for (unsigned I = 1, E = OffsetRegs.size(); I != E; ++I) {
    Register CurOffReg = OffsetRegs[I];
    auto *RC = &PISA::Reg64bRegClass;
    assert(MRI->getRegClassOrNull(OffsetReg) == RC ||
           TRI.getRegClassFromLLT(MRI->getType(OffsetReg)) == RC ||
           TRI.getRegClassFromLLT(MRI->getType(CurOffReg)) == RC);
    auto OffsetC = getIConstantVRegValWithLookThrough(OffsetReg, *MRI);
    auto CurOffC = getIConstantVRegValWithLookThrough(CurOffReg, *MRI);
    Register Tmp = MRI->createVirtualRegister(RC);
    unsigned OpIdx = OffsetC.has_value() << 1 | CurOffC.has_value();
    auto MIB = BuildMI(*BB, MI, DL, TII.get(AddOpcodes[OpIdx]), Tmp);
    if (OffsetC)
      MIB.addImm(OffsetC->Value.getSExtValue());
    else
      MIB.addReg(OffsetReg);
    if (CurOffC)
      MIB.addImm(CurOffC->Value.getSExtValue());
    else
      MIB.addReg(CurOffReg);
    OffsetReg = Tmp;
  }

  // Per PISA spec, register offset in ld.param address operand must be 32-bit.
  // Truncate the 64-bit offset to 32-bit.
  Register TruncReg = MRI->createVirtualRegister(&PISA::Reg32bRegClass);
  BuildMI(*BB, MI, DL, TII.get(PISA::trunc_32b_64b_r), TruncReg)
      .addReg(OffsetReg);
  OffsetReg = TruncReg;

  unsigned Slot = BaseDef->getOperand(1).getImm();
  const char *ArgName = nullptr;
  if (BaseDef->getNumOperands() > 3 && BaseDef->getOperand(3).isSymbol())
    ArgName = BaseDef->getOperand(3).getSymbolName();
  return {{[=](MachineInstrBuilder &MIB) { MIB.addImm(Slot); },
           [=](MachineInstrBuilder &MIB) {
             MIB.addReg(OffsetReg);
             if (ArgName)
               MIB.addExternalSymbol(ArgName);
           }}};
}

namespace llvm {
InstructionSelector *
createPISAInstructionSelector(const PISATargetMachine &TM,
                              const PISASubtarget &Subtarget,
                              const RegisterBankInfo &RBI) {
  return new PISAInstructionSelector(TM, Subtarget, RBI);
}
} // namespace llvm
