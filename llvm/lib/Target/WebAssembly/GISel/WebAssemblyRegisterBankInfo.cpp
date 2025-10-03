#include "WebAssemblyRegisterBankInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/CodeGen/TargetOpcodes.h"
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
  case G_DEBUGTRAP:
    return getInstructionMapping(MappingID, /*Cost=*/1, getOperandsMapping({}),
                                 0);
  case COPY:
    Register DstReg = MI.getOperand(0).getReg();
    if (DstReg.isPhysical()) {
      if (DstReg.id() == WebAssembly::SP32) {
        return getInstructionMapping(
            MappingID, /*Cost=*/1,
            getOperandsMapping(
                {&WebAssembly::ValueMappings[WebAssembly::I32Idx]}),
            1);
      } else if (DstReg.id() == WebAssembly::SP64) {
        return getInstructionMapping(
            MappingID, /*Cost=*/1,
            getOperandsMapping(
                {&WebAssembly::ValueMappings[WebAssembly::I64Idx]}),
            1);
      } else {
        llvm_unreachable("Trying to copy into WASM physical register other "
                         "than sp32 or sp64?");
      }
    }
    break;
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
  case G_CTLZ:
  case G_CTLZ_ZERO_UNDEF:
  case G_CTTZ:
  case G_CTTZ_ZERO_UNDEF:
  case G_CTPOP:
  case G_FSHL:
  case G_FSHR:
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
  case G_SELECT:
    OperandsMapping = getOperandsMapping(
        {&Op0IntValueMapping, &WebAssembly::ValueMappings[WebAssembly::I32Idx],
         &Op0IntValueMapping, &Op0IntValueMapping});
    break;
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
