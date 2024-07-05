#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Metadata.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "spirv-nonsemantic-debug-info"

namespace llvm {
struct SPIRVEmitNonSemanticDI : public MachineFunctionPass {
  static char ID;
  SPIRVTargetMachine *TM;
  SPIRVEmitNonSemanticDI(SPIRVTargetMachine *TM);
  SPIRVEmitNonSemanticDI();

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool IsGlobalDIEmitted = false;
  bool emitGlobalDI(MachineFunction &MF);
};

void initializeSPIRVEmitNonSemanticDIPass(PassRegistry &);

FunctionPass *createSPIRVEmitNonSemanticDIPass(SPIRVTargetMachine *TM) {
  return new SPIRVEmitNonSemanticDI(TM);
}
} // namespace llvm

using namespace llvm;

INITIALIZE_PASS(SPIRVEmitNonSemanticDI, DEBUG_TYPE,
                "SPIRV NonSemantic.Shader.DebugInfo.100 emitter", false, false)

char SPIRVEmitNonSemanticDI::ID = 0;

SPIRVEmitNonSemanticDI::SPIRVEmitNonSemanticDI(SPIRVTargetMachine *TM)
    : MachineFunctionPass(ID), TM(TM) {
  initializeSPIRVEmitNonSemanticDIPass(*PassRegistry::getPassRegistry());
}

SPIRVEmitNonSemanticDI::SPIRVEmitNonSemanticDI() : MachineFunctionPass(ID) {
  initializeSPIRVEmitNonSemanticDIPass(*PassRegistry::getPassRegistry());
}

bool SPIRVEmitNonSemanticDI::emitGlobalDI(MachineFunction &MF) {
  if (MF.begin() == MF.end()) {
    IsGlobalDIEmitted = false;
    return false;
  }
  const MachineModuleInfo &MMI = MF.getMMI();
  const Module *M = MMI.getModule();
  const NamedMDNode *DbgCu = M->getNamedMetadata("llvm.dbg.cu");
  if (!DbgCu)
    return false;
  std::string FilePath;
  unsigned SourceLanguage = 0;
  for (const auto *Op : DbgCu->operands()) {
    if (const auto *CompileUnit = dyn_cast<DICompileUnit>(Op)) {
      DIFile *File = CompileUnit->getFile();
      FilePath = ((File->getDirectory() + sys::path::get_separator() +
                   File->getFilename()))
                     .str();
      SourceLanguage = CompileUnit->getSourceLanguage();
      break;
    }
  }
  const NamedMDNode *ModuleFlags = M->getNamedMetadata("llvm.module.flags");
  int64_t DwarfVersion = 0;
  int64_t DebugInfoVersion = 0;
  for (const auto *Op : ModuleFlags->operands()) {
    const MDOperand &MaybeStrOp = Op->getOperand(1);
    if (MaybeStrOp.equalsStr("Dwarf Version"))
      DwarfVersion =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
              ->getSExtValue();
    else if (MaybeStrOp.equalsStr("Debug Info Version"))
      DebugInfoVersion =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
              ->getSExtValue();
  }
  const SPIRVInstrInfo *TII = TM->getSubtargetImpl()->getInstrInfo();
  const SPIRVRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
  const RegisterBankInfo *RBI = TM->getSubtargetImpl()->getRegBankInfo();
  SPIRVGlobalRegistry *GR = TM->getSubtargetImpl()->getSPIRVGlobalRegistry();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineBasicBlock &MBB = *MF.begin();
  MachineIRBuilder MIRBuilder(MBB, MBB.begin());

  const Register StrReg = MRI.createVirtualRegister(&SPIRV::IDRegClass);
  MRI.setType(StrReg, LLT::scalar(32));
  MachineInstrBuilder MIB = MIRBuilder.buildInstr(SPIRV::OpString);
  MIB.addDef(StrReg);
  addStringImm(FilePath, MIB);

  const SPIRVType *VoidTyMI =
      GR->getOrCreateSPIRVType(Type::getVoidTy(M->getContext()), MIRBuilder);
  GR->assignSPIRVTypeToVReg(VoidTyMI, GR->getSPIRVTypeID(VoidTyMI), MF);

  const Register DebugSourceResIdReg =
      MRI.createVirtualRegister(&SPIRV::IDRegClass);
  MRI.setType(DebugSourceResIdReg, LLT::scalar(32));
  MIB = MIRBuilder.buildInstr(SPIRV::OpExtInst)
            .addDef(DebugSourceResIdReg)
            .addUse(GR->getSPIRVTypeID(VoidTyMI))
            .addImm(static_cast<int64_t>(
                SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100))
            .addImm(SPIRV::NonSemanticExtInst::DebugSource)
            .addUse(StrReg);
  MIB.constrainAllUses(*TII, *TRI, *RBI);

  const SPIRVType *I32Ty =
      GR->getOrCreateSPIRVType(Type::getInt32Ty(M->getContext()), MIRBuilder);
  GR->assignSPIRVTypeToVReg(I32Ty, GR->getSPIRVTypeID(I32Ty), MF);

  const Register DwarfVersionReg =
      MRI.createVirtualRegister(&SPIRV::IDRegClass);
  MRI.setType(DwarfVersionReg, LLT::scalar(32));
  MIRBuilder.buildInstr(SPIRV::OpConstantI)
      .addDef(DwarfVersionReg)
      .addUse(GR->getSPIRVTypeID(I32Ty))
      .addImm(DwarfVersion);

  const Register DebugInfoVersionReg =
      MRI.createVirtualRegister(&SPIRV::IDRegClass);
  MRI.setType(DebugInfoVersionReg, LLT::scalar(32));
  MIRBuilder.buildInstr(SPIRV::OpConstantI)
      .addDef(DebugInfoVersionReg)
      .addUse(GR->getSPIRVTypeID(I32Ty))
      .addImm(DebugInfoVersion);

  const Register SourceLanguageReg =
      MRI.createVirtualRegister(&SPIRV::IDRegClass);
  MRI.setType(SourceLanguageReg, LLT::scalar(32));
  MIRBuilder.buildInstr(SPIRV::OpConstantI)
      .addDef(SourceLanguageReg)
      .addUse(GR->getSPIRVTypeID(I32Ty))
      .addImm(SourceLanguage);

  const Register DebugCompUnitResIdReg =
      MRI.createVirtualRegister(&SPIRV::IDRegClass);
  MRI.setType(DebugCompUnitResIdReg, LLT::scalar(32));
  MIB = MIRBuilder.buildInstr(SPIRV::OpExtInst)
            .addDef(DebugCompUnitResIdReg)
            .addUse(GR->getSPIRVTypeID(VoidTyMI))
            .addImm(static_cast<int64_t>(
                SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100))
            .addImm(SPIRV::NonSemanticExtInst::DebugCompilationUnit)
            .addUse(DebugInfoVersionReg)
            .addUse(DwarfVersionReg)
            .addUse(DebugSourceResIdReg)
            .addUse(SourceLanguageReg);
  MIB.constrainAllUses(*TII, *TRI, *RBI);

  return true;
}

bool SPIRVEmitNonSemanticDI::runOnMachineFunction(MachineFunction &MF) {
  bool Res = false;
  if (!IsGlobalDIEmitted) {
    IsGlobalDIEmitted = true;
    Res = emitGlobalDI(MF);
  }
  return Res;
}
