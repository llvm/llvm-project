#include "WebAssemblyRegisterBankInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/CodeGen/TargetOpcodes.h"

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
  I64Idx = 4,
  F32Idx = 7,
  F64Idx = 10
};

const RegisterBankInfo::PartialMapping PartMappings[]{{0, 32, I32RegBank},
                                                      {0, 64, I64RegBank},
                                                      {0, 32, F32RegBank},
                                                      {0, 64, F64RegBank}};

const RegisterBankInfo::ValueMapping ValueMappings[] = {
    // invalid
    {nullptr, 0},
    // up to 3 operands as I32
    {&PartMappings[PMI_I32 - PMI_Min], 1},
    {&PartMappings[PMI_I32 - PMI_Min], 1},
    {&PartMappings[PMI_I32 - PMI_Min], 1},
    // up to 3 operands as I64
    {&PartMappings[PMI_I64 - PMI_Min], 1},
    {&PartMappings[PMI_I64 - PMI_Min], 1},
    {&PartMappings[PMI_I64 - PMI_Min], 1},
    // up to 3 operands as F32
    {&PartMappings[PMI_F32 - PMI_Min], 1},
    {&PartMappings[PMI_F32 - PMI_Min], 1},
    {&PartMappings[PMI_F32 - PMI_Min], 1},
    // up to 3 operands as F64
    {&PartMappings[PMI_F64 - PMI_Min], 1},
    {&PartMappings[PMI_F64 - PMI_Min], 1},
    {&PartMappings[PMI_F64 - PMI_Min], 1}};

} // namespace WebAssembly
} // namespace llvm

using namespace llvm;

WebAssemblyRegisterBankInfo::WebAssemblyRegisterBankInfo(
    const TargetRegisterInfo &TRI) {}

// Instructions where use operands are floating point registers.
// Def operands are general purpose.
static bool isFloatingPointOpcodeUse(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI:
  case TargetOpcode::G_FCMP:
    return true;
  default:
    return isPreISelGenericFloatingPointOpcode(Opc);
  }
}

// Instructions where def operands are floating point registers.
// Use operands are general purpose.
static bool isFloatingPointOpcodeDef(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
    return true;
  default:
    return isPreISelGenericFloatingPointOpcode(Opc);
  }
}

static bool isAmbiguous(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_STORE:
  case TargetOpcode::G_PHI:
  case TargetOpcode::G_SELECT:
  case TargetOpcode::G_IMPLICIT_DEF:
  case TargetOpcode::G_UNMERGE_VALUES:
  case TargetOpcode::G_MERGE_VALUES:
    return true;
  default:
    return false;
  }
}

const RegisterBankInfo::InstructionMapping &
WebAssemblyRegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {

  unsigned Opc = MI.getOpcode();
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();

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

  switch (Opc) {
  case G_BR:
    return getInstructionMapping(MappingID, /*Cost=*/1,
                                 getOperandsMapping({nullptr}), NumOperands);
  case G_TRAP:
    return getInstructionMapping(MappingID, /*Cost=*/1, nullptr, 0);
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
  case G_INTTOPTR:
  case G_PTRTOINT:
  case G_ADD:
  case G_SUB:
  case G_MUL:
  case G_SDIV:
  case G_SREM:
  case G_UDIV:
  case G_UREM:
    OperandsMapping = &Op0IntValueMapping;
    break;
  case G_SEXT_INREG:
    OperandsMapping =
        getOperandsMapping({&Op0IntValueMapping, &Op0IntValueMapping, nullptr});
    break;
  case G_FRAME_INDEX:
    OperandsMapping = getOperandsMapping({&Op0IntValueMapping, nullptr});
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
  case G_STORE:
    if (MRI.getType(MI.getOperand(1).getReg()).getAddressSpace() != 0)
      break;
    OperandsMapping =
        getOperandsMapping({&Op0IntValueMapping, &Pointer0ValueMapping});
    break;
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
    unsigned Op2Size = Op1Ty.getSizeInBits();
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
  case G_BRCOND:
    OperandsMapping = getOperandsMapping({&Op0IntValueMapping, nullptr});
    break;
  case COPY: {
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();
    // Check if one of the register is not a generic register.
    if ((DstReg.isPhysical() || !MRI.getType(DstReg).isValid()) ||
        (SrcReg.isPhysical() || !MRI.getType(SrcReg).isValid())) {
      const RegisterBank *DstRB = getRegBank(DstReg, MRI, TRI);
      const RegisterBank *SrcRB = getRegBank(SrcReg, MRI, TRI);
      if (!DstRB)
        DstRB = SrcRB;
      else if (!SrcRB)
        SrcRB = DstRB;
      // If both RB are null that means both registers are generic.
      // We shouldn't be here.
      assert(DstRB && SrcRB && "Both RegBank were nullptr");
      TypeSize DstSize = getSizeInBits(DstReg, MRI, TRI);
      TypeSize SrcSize = getSizeInBits(SrcReg, MRI, TRI);
      assert(DstSize == SrcSize &&
             "Trying to copy between different sized regbanks? Why?");

      return getInstructionMapping(
          DefaultMappingID, copyCost(*DstRB, *SrcRB, DstSize),
          getCopyMapping(DstRB->getID(), SrcRB->getID(), Size),
          // We only care about the mapping of the destination.
          /*NumOperands*/ 1);
    }
  }
  }
  if (!OperandsMapping)
    return getInvalidInstructionMapping();

  return getInstructionMapping(MappingID, /*Cost=*/1, OperandsMapping,
                               NumOperands);
}
