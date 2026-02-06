#include "WebAssemblyRegisterBankInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_TARGET_REGBANK_IMPL

#include "WebAssemblyGenRegisterBank.inc"

namespace llvm {
namespace WebAssembly {
enum PartialMappingIdx {
  PMI_None = -1,
  PMI_I32 = 1,
  PMI_I64,
  PMI_F32,
  PMI_F64,
  PMI_Min = PMI_I32,
};

enum ValueMappingIdx {
  InvalidIdx = 0,
  I32Idx = 1,
  I64Idx = 5,
  F32Idx = 9,
  F64Idx = 13
};

const RegisterBankInfo::PartialMapping PartMappings[]{{0, 32, I32RegBank},
                                                      {0, 64, I64RegBank},
                                                      {0, 32, F32RegBank},
                                                      {0, 64, F64RegBank}};

const RegisterBankInfo::ValueMapping ValueMappings[] = {
    // invalid
    {nullptr, 0},
    // up to 4 operands as I32
    {&PartMappings[PMI_I32 - PMI_Min], 1},
    {&PartMappings[PMI_I32 - PMI_Min], 1},
    {&PartMappings[PMI_I32 - PMI_Min], 1},
    {&PartMappings[PMI_I32 - PMI_Min], 1},
    // up to 4 operands as I64
    {&PartMappings[PMI_I64 - PMI_Min], 1},
    {&PartMappings[PMI_I64 - PMI_Min], 1},
    {&PartMappings[PMI_I64 - PMI_Min], 1},
    {&PartMappings[PMI_I64 - PMI_Min], 1},
    // up to 4 operands as F32
    {&PartMappings[PMI_F32 - PMI_Min], 1},
    {&PartMappings[PMI_F32 - PMI_Min], 1},
    {&PartMappings[PMI_F32 - PMI_Min], 1},
    {&PartMappings[PMI_F32 - PMI_Min], 1},
    // up to 4 operands as F64
    {&PartMappings[PMI_F64 - PMI_Min], 1},
    {&PartMappings[PMI_F64 - PMI_Min], 1},
    {&PartMappings[PMI_F64 - PMI_Min], 1},
    {&PartMappings[PMI_F64 - PMI_Min], 1}};

} // namespace WebAssembly
} // namespace llvm

using namespace llvm;

WebAssemblyRegisterBankInfo::WebAssemblyRegisterBankInfo(
    const TargetRegisterInfo &TRI) {}

bool WebAssemblyRegisterBankInfo::isPHIWithFPConstraints(
    const MachineInstr &MI, const MachineRegisterInfo &MRI,
    const WebAssemblyRegisterInfo &TRI, const unsigned Depth) const {
  if (!MI.isPHI() || Depth > MaxFPRSearchDepth)
    return false;

  return any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
                [&](const MachineInstr &UseMI) {
                  if (onlyUsesFP(UseMI, MRI, TRI, Depth + 1))
                    return true;
                  return isPHIWithFPConstraints(UseMI, MRI, TRI, Depth + 1);
                });
}

bool WebAssemblyRegisterBankInfo::hasFPConstraints(
    const MachineInstr &MI, const MachineRegisterInfo &MRI,
    const WebAssemblyRegisterInfo &TRI, unsigned Depth) const {
  unsigned Op = MI.getOpcode();
  // if (Op == TargetOpcode::G_INTRINSIC && isFPIntrinsic(MRI, MI))
  //   return true;

  // Do we have an explicit floating point instruction?
  if (isPreISelGenericFloatingPointOpcode(Op))
    return true;

  // No. Check if we have a copy-like instruction. If we do, then we could
  // still be fed by floating point instructions.
  if (Op != TargetOpcode::COPY && !MI.isPHI() &&
      !isPreISelGenericOptimizationHint(Op))
    return false;

  // Check if we already know the register bank.
  auto *RB = getRegBank(MI.getOperand(0).getReg(), MRI, TRI);
  if (RB == &WebAssembly::F32RegBank || RB == &WebAssembly::F64RegBank)
    return true;
  if (RB == &WebAssembly::I32RegBank || RB == &WebAssembly::I64RegBank)
    return false;

  // We don't know anything.
  //
  // If we have a phi, we may be able to infer that it will be assigned a FPR
  // based off of its inputs.
  if (!MI.isPHI() || Depth > MaxFPRSearchDepth)
    return false;

  return any_of(MI.explicit_uses(), [&](const MachineOperand &Op) {
    return Op.isReg() &&
           onlyDefinesFP(*MRI.getVRegDef(Op.getReg()), MRI, TRI, Depth + 1);
  });
}

bool WebAssemblyRegisterBankInfo::onlyUsesFP(const MachineInstr &MI,
                                             const MachineRegisterInfo &MRI,
                                             const WebAssemblyRegisterInfo &TRI,
                                             unsigned Depth) const {
  switch (MI.getOpcode()) {
  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI:
  case TargetOpcode::G_FPTOSI_SAT:
  case TargetOpcode::G_FPTOUI_SAT:
  case TargetOpcode::G_FCMP:
  case TargetOpcode::G_LROUND:
  case TargetOpcode::G_LLROUND:
    return true;
  default:
    break;
  }
  return hasFPConstraints(MI, MRI, TRI, Depth);
}

bool WebAssemblyRegisterBankInfo::onlyDefinesFP(
    const MachineInstr &MI, const MachineRegisterInfo &MRI,
    const WebAssemblyRegisterInfo &TRI, unsigned Depth) const {
  switch (MI.getOpcode()) {
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
  case TargetOpcode::G_INSERT_VECTOR_ELT:
  case TargetOpcode::G_BUILD_VECTOR:
  case TargetOpcode::G_BUILD_VECTOR_TRUNC:
    return true;
  default:
    break;
  }
  return hasFPConstraints(MI, MRI, TRI, Depth);
}

bool WebAssemblyRegisterBankInfo::prefersFPUse(
    const MachineInstr &MI, const MachineRegisterInfo &MRI,
    const WebAssemblyRegisterInfo &TRI, unsigned Depth) const {
  switch (MI.getOpcode()) {
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
    return MRI.getType(MI.getOperand(0).getReg()).getSizeInBits() ==
           MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
  }
  return onlyDefinesFP(MI, MRI, TRI, Depth);
}

const RegisterBankInfo::InstructionMapping &
WebAssemblyRegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {

  unsigned Opc = MI.getOpcode();
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const WebAssemblySubtarget &STI = MF.getSubtarget<WebAssemblySubtarget>();
  const WebAssemblyRegisterInfo &TRI = *STI.getRegisterInfo();

  if ((Opc != TargetOpcode::COPY && !isPreISelGenericOpcode(Opc)) ||
      Opc == TargetOpcode::G_PHI) {
    const RegisterBankInfo::InstructionMapping &Mapping =
        getInstrMappingImpl(MI);
    if (Mapping.isValid())
      return Mapping;
  }

  using namespace TargetOpcode;

  unsigned NumOperands = MI.getNumOperands();
  const ValueMapping *OperandsMapping = nullptr;
  unsigned MappingID = DefaultMappingID;

  // Check if LLT sizes match sizes of available register banks.
  for (const MachineOperand &Op : MI.operands()) {
    if (Op.isReg()) {
      LLT RegTy = MRI.getType(Op.getReg());

      if (RegTy.isScalar() &&
          (RegTy.getSizeInBits() != 32 && RegTy.getSizeInBits() != 64))
        return getInvalidInstructionMapping();

      if (RegTy.isVector() && RegTy.getSizeInBits() != 128)
        return getInvalidInstructionMapping();
    }
  }

  if (MI.isDebugValue()) {
    const MachineOperand &MO = MI.getOperand(0);

    if (!MO.isReg())
      return getInstructionMapping(DefaultMappingID, /*Cost=*/1, nullptr, 0);

    auto &CurBankOrClass = MRI.getRegClassOrRegBank(MO.getReg());

    const RegisterBank *NewBank;

    if (auto *CurBank = CurBankOrClass.dyn_cast<const RegisterBank *>()) {
      NewBank = CurBank;
    } else if (auto *CurClass =
                   CurBankOrClass.dyn_cast<const TargetRegisterClass *>()) {
      getRegBankFromRegClass(*CurClass, MRI.getType(MO.getReg())).dump();
      NewBank = &getRegBankFromRegClass(*CurClass, MRI.getType(MO.getReg()));
    } else {
      llvm_unreachable("Encountered DBG_VALUE with an unmapped register.");
    }

    WebAssembly::ValueMappingIdx ValueMappingIdx;

    switch (NewBank->getID()) {
    case WebAssembly::I32RegBankID:
      ValueMappingIdx = WebAssembly::I32Idx;
      break;
    case WebAssembly::I64RegBankID:
      ValueMappingIdx = WebAssembly::I64Idx;
      break;
    case WebAssembly::F32RegBankID:
      ValueMappingIdx = WebAssembly::F32Idx;
      break;
    case WebAssembly::F64RegBankID:
      ValueMappingIdx = WebAssembly::F64Idx;
      break;
    default:
      llvm_unreachable("Encountered unexpected register bank.");
    }

    return getInstructionMapping(
        MappingID, /*Cost=*/1,
        getOperandsMapping({&WebAssembly::ValueMappings[ValueMappingIdx]}),
        NumOperands);
  }

  switch (Opc) {
  case G_BR:
    return getInstructionMapping(MappingID, /*Cost=*/1,
                                 getOperandsMapping({nullptr}), NumOperands);
  case G_TRAP:
  case G_DEBUGTRAP:
    return getInstructionMapping(MappingID, /*Cost=*/1, getOperandsMapping({}),
                                 0);
  case COPY: {
    Register DstReg = MI.getOperand(0).getReg();
    if (DstReg.isPhysical()) {
      if (DstReg.id() == WebAssembly::SP32) {
        return getInstructionMapping(
            MappingID, /*Cost=*/1,
            getOperandsMapping(
                {&WebAssembly::ValueMappings[WebAssembly::I32Idx]}),
            1);
      }

      if (DstReg.id() == WebAssembly::SP64) {
        return getInstructionMapping(
            MappingID, /*Cost=*/1,
            getOperandsMapping(
                {&WebAssembly::ValueMappings[WebAssembly::I64Idx]}),
            1);
      }

      llvm_unreachable("Trying to copy into WASM physical register other "
                       "than sp32 or sp64?");
    }
    break;
  }
  case G_INTRINSIC:
  case G_INTRINSIC_W_SIDE_EFFECTS: {
    switch (cast<GIntrinsic>(MI).getIntrinsicID()) {
    default:
      break;
    }
    return getInstructionMapping(DefaultMappingID, /*Cost=*/1, nullptr, 0);
  }
  }

  const LLT Op0Ty = MRI.getType(MI.getOperand(0).getReg());
  unsigned Op0Size = Op0Ty.getSizeInBits();

  auto &Op0IntValueMapping =
      WebAssembly::ValueMappings[Op0Size == 64 ? WebAssembly::I64Idx
                                               : WebAssembly::I32Idx];
  auto &Op0FloatValueMapping =
      WebAssembly::ValueMappings[Op0Size == 64 ? WebAssembly::F64Idx
                                               : WebAssembly::F32Idx];
  auto &Pointer0ValueMapping =
      WebAssembly::ValueMappings[MI.getMF()->getDataLayout()
                                             .getPointerSizeInBits(0) == 64
                                     ? WebAssembly::I64Idx
                                     : WebAssembly::I32Idx];

  switch (Opc) {
  case G_AND:
  case G_OR:
  case G_XOR:
  case G_SHL:
  case G_ASHR:
  case G_LSHR:
  case G_PTR_ADD:
  case G_PTRMASK:
  case G_INTTOPTR:
  case G_PTRTOINT:
  case G_ADD:
  case G_SUB:
  case G_MUL:
  case G_SDIV:
  case G_SREM:
  case G_UDIV:
  case G_UREM:
  case G_CTLZ:
  case G_CTLZ_ZERO_UNDEF:
  case G_CTTZ:
  case G_CTTZ_ZERO_UNDEF:
  case G_CTPOP:
  case G_FSHL:
  case G_FSHR:
  case G_ROTR:
  case G_ROTL:
    OperandsMapping = &Op0IntValueMapping;
    break;
  case G_FADD:
  case G_FSUB:
  case G_FDIV:
  case G_FMUL:
  case G_FNEG:
  case G_FABS:
  case G_FCEIL:
  case G_FFLOOR:
  case G_FSQRT:
  case G_INTRINSIC_TRUNC:
  case G_FNEARBYINT:
  case G_FRINT:
  case G_INTRINSIC_ROUNDEVEN:
  case G_FMINIMUM:
  case G_FMAXIMUM:
  case G_FMINNUM:
  case G_FMAXNUM:
  case G_FMINNUM_IEEE:
  case G_FMAXNUM_IEEE:
  case G_FMA:
  case G_FREM:
  case G_FCOPYSIGN:
  case G_FCANONICALIZE:
  case G_STRICT_FMUL:
    OperandsMapping = &Op0FloatValueMapping;
    break;
  case G_SEXT_INREG:
    OperandsMapping =
        getOperandsMapping({&Op0IntValueMapping, &Op0IntValueMapping, nullptr});
    break;
  case G_FRAME_INDEX:
    OperandsMapping = getOperandsMapping({&Op0IntValueMapping, nullptr});
    break;
  case G_VASTART:
    OperandsMapping = &Op0IntValueMapping;
    break;
  case G_ZEXT:
  case G_ANYEXT:
  case G_SEXT:
  case G_TRUNC: {
    const LLT Op1Ty = MRI.getType(MI.getOperand(1).getReg());
    unsigned Op1Size = Op1Ty.getSizeInBits();

    auto &Op1IntValueMapping =
        WebAssembly::ValueMappings[Op1Size == 64 ? WebAssembly::I64Idx
                                                 : WebAssembly::I32Idx];
    OperandsMapping =
        getOperandsMapping({&Op0IntValueMapping, &Op1IntValueMapping});
    break;
  }
  case G_LOAD:
  case G_ZEXTLOAD:
  case G_SEXTLOAD: {
    if (MRI.getType(MI.getOperand(1).getReg()).getAddressSpace() != 0)
      break;

    auto *LoadValueMapping = &Op0IntValueMapping;
    if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
               [&](const MachineInstr &UseMI) {
                 // If we have at least one direct or indirect use
                 // in a FP instruction,
                 // assume this was a floating point load in the IR. If it was
                 // not, we would have had a bitcast before reaching that
                 // instruction.
                 //
                 // Int->FP conversion operations are also captured in
                 // prefersFPUse().

                 if (isPHIWithFPConstraints(UseMI, MRI, TRI))
                   return true;

                 return onlyUsesFP(UseMI, MRI, TRI) ||
                        prefersFPUse(UseMI, MRI, TRI);
               }))
      LoadValueMapping = &Op0FloatValueMapping;
    OperandsMapping =
        getOperandsMapping({LoadValueMapping, &Pointer0ValueMapping});
    break;
  }
  case G_STORE: {
    if (MRI.getType(MI.getOperand(1).getReg()).getAddressSpace() != 0)
      break;

    Register VReg = MI.getOperand(0).getReg();
    if (!VReg)
      break;
    MachineInstr *DefMI = MRI.getVRegDef(VReg);
    if (onlyDefinesFP(*DefMI, MRI, TRI)) {
      OperandsMapping =
          getOperandsMapping({&Op0FloatValueMapping, &Pointer0ValueMapping});
    } else {
      OperandsMapping =
          getOperandsMapping({&Op0IntValueMapping, &Pointer0ValueMapping});
    }
    break;
  }
  case G_MEMCPY:
  case G_MEMMOVE: {
    if (MRI.getType(MI.getOperand(0).getReg()).getAddressSpace() != 0)
      break;
    if (MRI.getType(MI.getOperand(1).getReg()).getAddressSpace() != 0)
      break;

    const LLT Op2Ty = MRI.getType(MI.getOperand(2).getReg());
    unsigned Op2Size = Op2Ty.getSizeInBits();
    auto &Op2IntValueMapping =
        WebAssembly::ValueMappings[Op2Size == 64 ? WebAssembly::I64Idx
                                                 : WebAssembly::I32Idx];
    OperandsMapping =
        getOperandsMapping({&Pointer0ValueMapping, &Pointer0ValueMapping,
                            &Op2IntValueMapping, nullptr});
    break;
  }
  case G_MEMSET: {
    if (MRI.getType(MI.getOperand(0).getReg()).getAddressSpace() != 0)
      break;
    const LLT Op1Ty = MRI.getType(MI.getOperand(1).getReg());
    unsigned Op1Size = Op1Ty.getSizeInBits();
    auto &Op1IntValueMapping =
        WebAssembly::ValueMappings[Op1Size == 64 ? WebAssembly::I64Idx
                                                 : WebAssembly::I32Idx];

    const LLT Op2Ty = MRI.getType(MI.getOperand(2).getReg());
    unsigned Op2Size = Op2Ty.getSizeInBits();
    auto &Op2IntValueMapping =
        WebAssembly::ValueMappings[Op2Size == 64 ? WebAssembly::I64Idx
                                                 : WebAssembly::I32Idx];

    OperandsMapping =
        getOperandsMapping({&Pointer0ValueMapping, &Op1IntValueMapping,
                            &Op2IntValueMapping, nullptr});
    break;
  }
  case G_GLOBAL_VALUE:
  case G_CONSTANT:
    OperandsMapping = getOperandsMapping({&Op0IntValueMapping, nullptr});
    break;
  case G_FCONSTANT:
    OperandsMapping = getOperandsMapping({&Op0FloatValueMapping, nullptr});
    break;
  case G_IMPLICIT_DEF:
    OperandsMapping = &Op0IntValueMapping;
    break;
  case G_ICMP: {
    const LLT Op2Ty = MRI.getType(MI.getOperand(2).getReg());
    unsigned Op2Size = Op2Ty.getSizeInBits();

    auto &Op2IntValueMapping =
        WebAssembly::ValueMappings[Op2Size == 64 ? WebAssembly::I64Idx
                                                 : WebAssembly::I32Idx];

    OperandsMapping =
        getOperandsMapping({&Op0IntValueMapping, nullptr, &Op2IntValueMapping,
                            &Op2IntValueMapping});
    break;
  }
  case G_FCMP: {
    const LLT Op2Ty = MRI.getType(MI.getOperand(2).getReg());
    unsigned Op2Size = Op2Ty.getSizeInBits();

    auto &Op2FloatValueMapping =
        WebAssembly::ValueMappings[Op2Size == 64 ? WebAssembly::F64Idx
                                                 : WebAssembly::F32Idx];

    OperandsMapping =
        getOperandsMapping({&Op0IntValueMapping, nullptr, &Op2FloatValueMapping,
                            &Op2FloatValueMapping});
    break;
  }
  case G_BRCOND:
    OperandsMapping = getOperandsMapping({&Op0IntValueMapping, nullptr});
    break;
  case G_JUMP_TABLE:
    OperandsMapping = getOperandsMapping({&Op0IntValueMapping, nullptr});
    break;
  case G_BRJT:
    OperandsMapping = getOperandsMapping(
        {&Op0IntValueMapping, nullptr, &Pointer0ValueMapping});
    break;
  case COPY: {
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();

    const RegisterBank *DstRB = getRegBank(DstReg, MRI, TRI);
    const RegisterBank *SrcRB = getRegBank(SrcReg, MRI, TRI);

    if (!DstRB)
      DstRB = SrcRB;
    else if (!SrcRB)
      SrcRB = DstRB;

    assert(DstRB && SrcRB && "Both RegBank were nullptr");
    TypeSize DstSize = getSizeInBits(DstReg, MRI, TRI);
    TypeSize SrcSize = getSizeInBits(SrcReg, MRI, TRI);
    assert(DstSize == SrcSize &&
           "Trying to copy between different sized regbanks? Why?");

    WebAssembly::ValueMappingIdx DstValMappingIdx = WebAssembly::InvalidIdx;
    switch (DstRB->getID()) {
    case WebAssembly::I32RegBankID:
      DstValMappingIdx = WebAssembly::I32Idx;
      break;
    case WebAssembly::I64RegBankID:
      DstValMappingIdx = WebAssembly::I64Idx;
      break;
    case WebAssembly::F32RegBankID:
      DstValMappingIdx = WebAssembly::F32Idx;
      break;
    case WebAssembly::F64RegBankID:
      DstValMappingIdx = WebAssembly::F64Idx;
      break;
    default:
      break;
    }

    WebAssembly::ValueMappingIdx SrcValMappingIdx = WebAssembly::InvalidIdx;
    switch (SrcRB->getID()) {
    case WebAssembly::I32RegBankID:
      SrcValMappingIdx = WebAssembly::I32Idx;
      break;
    case WebAssembly::I64RegBankID:
      SrcValMappingIdx = WebAssembly::I64Idx;
      break;
    case WebAssembly::F32RegBankID:
      SrcValMappingIdx = WebAssembly::F32Idx;
      break;
    case WebAssembly::F64RegBankID:
      SrcValMappingIdx = WebAssembly::F64Idx;
      break;
    default:
      break;
    }

    OperandsMapping =
        getOperandsMapping({&WebAssembly::ValueMappings[DstValMappingIdx],
                            &WebAssembly::ValueMappings[SrcValMappingIdx]});
    return getInstructionMapping(
        MappingID, /*Cost=*/copyCost(*DstRB, *SrcRB, DstSize), OperandsMapping,
        // We only care about the mapping of the destination for COPY.
        1);
  }
  case G_SELECT: {
    // Try to minimize the number of copies. If we have more floating point
    // constrained values than not, then we'll put everything on FPR. Otherwise,
    // everything has to be on GPR.
    unsigned NumFP = 0;

    // Check if the uses of the result always produce floating point values.
    //
    // For example:
    //
    // %z = G_SELECT %cond %x %y
    // fpr = G_FOO %z ...
    if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
               [&](MachineInstr &MI) { return onlyUsesFP(MI, MRI, TRI); }))
      ++NumFP;

    // Check if the defs of the source values always produce floating point
    // values.
    //
    // For example:
    //
    // %x = G_SOMETHING_ALWAYS_FLOAT %a ...
    // %z = G_SELECT %cond %x %y
    //
    // Also check whether or not the sources have already been decided to be
    // FPR. Keep track of this.
    //
    // This doesn't check the condition, since it's just whatever is in NZCV.
    // This isn't passed explicitly in a register to fcsel/csel.
    for (unsigned Idx = 2; Idx < 4; ++Idx) {
      Register VReg = MI.getOperand(Idx).getReg();
      MachineInstr *DefMI = MRI.getVRegDef(VReg);
      if (getRegBank(VReg, MRI, TRI) == &WebAssembly::F32RegBank ||
          getRegBank(VReg, MRI, TRI) == &WebAssembly::F64RegBank ||
          onlyDefinesFP(*DefMI, MRI, TRI))
        ++NumFP;
    }

    // If we have more FP constraints than not, then move everything over to
    // FPR.
    if (NumFP >= 2) {
      OperandsMapping =
          getOperandsMapping({&Op0FloatValueMapping,
                              &WebAssembly::ValueMappings[WebAssembly::I32Idx],
                              &Op0FloatValueMapping, &Op0FloatValueMapping});

    } else {
      OperandsMapping =
          getOperandsMapping({&Op0IntValueMapping,
                              &WebAssembly::ValueMappings[WebAssembly::I32Idx],
                              &Op0IntValueMapping, &Op0IntValueMapping});
    }
    break;
  }
  case G_FPTOSI:
  case G_FPTOSI_SAT:
  case G_FPTOUI:
  case G_FPTOUI_SAT: {
    const LLT Op1Ty = MRI.getType(MI.getOperand(1).getReg());
    unsigned Op1Size = Op1Ty.getSizeInBits();

    auto &Op1FloatValueMapping =
        WebAssembly::ValueMappings[Op1Size == 64 ? WebAssembly::F64Idx
                                                 : WebAssembly::F32Idx];

    OperandsMapping =
        getOperandsMapping({&Op0IntValueMapping, &Op1FloatValueMapping});
    break;
  }
  case G_SITOFP:
  case G_UITOFP: {
    const LLT Op1Ty = MRI.getType(MI.getOperand(1).getReg());
    unsigned Op1Size = Op1Ty.getSizeInBits();

    auto &Op1IntValueMapping =
        WebAssembly::ValueMappings[Op1Size == 64 ? WebAssembly::I64Idx
                                                 : WebAssembly::I32Idx];

    OperandsMapping =
        getOperandsMapping({&Op0FloatValueMapping, &Op1IntValueMapping});
    break;
  }
  case G_FPEXT:
  case G_FPTRUNC: {
    const LLT Op1Ty = MRI.getType(MI.getOperand(1).getReg());
    unsigned Op1Size = Op1Ty.getSizeInBits();

    auto &Op1FloatValueMapping =
        WebAssembly::ValueMappings[Op1Size == 64 ? WebAssembly::F64Idx
                                                 : WebAssembly::F32Idx];

    OperandsMapping =
        getOperandsMapping({&Op0FloatValueMapping, &Op1FloatValueMapping});
    break;
  }
  }

  if (!OperandsMapping)
    return getInvalidInstructionMapping();

  return getInstructionMapping(MappingID, /*Cost=*/1, OperandsMapping,
                               NumOperands);
}
