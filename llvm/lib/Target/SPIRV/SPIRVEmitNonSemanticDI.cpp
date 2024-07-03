#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
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

INITIALIZE_PASS(SPIRVEmitNonSemanticDI, "spirv-nonsemantic-debug-info",
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
  MachineModuleInfo &MMI = MF.getMMI();
  const Module *M = MMI.getModule();
  NamedMDNode *DbgCu = M->getNamedMetadata("llvm.dbg.cu");
  if (!DbgCu) {
    return false;
  }
  std::string FilePath;
  unsigned SourceLanguage;
  unsigned NumOp = DbgCu->getNumOperands();
  if (NumOp) {
    if (const auto *CompileUnit =
            dyn_cast<DICompileUnit>(DbgCu->getOperand(0))) {
      DIFile *File = CompileUnit->getFile();
      FilePath = ((File->getDirectory() + "/" + File->getFilename())).str();
      SourceLanguage = CompileUnit->getSourceLanguage();
    }
  }
  NamedMDNode *ModuleFlags = M->getNamedMetadata("llvm.module.flags");
  int64_t DwarfVersion = 0;
  int64_t DebugInfoVersion = 0;
  for (auto *Op : ModuleFlags->operands()) {
    const MDOperand &StrOp = Op->getOperand(1);
    if (StrOp.equalsStr("Dwarf Version")) {
      DwarfVersion =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
              ->getSExtValue();
    } else if (StrOp.equalsStr("Debug Info Version")) {
      DebugInfoVersion =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
              ->getSExtValue();
    }
  }
  const SPIRVInstrInfo *TII = TM->getSubtargetImpl()->getInstrInfo();
  const SPIRVRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
  const RegisterBankInfo *RBI = TM->getSubtargetImpl()->getRegBankInfo();
  SPIRVGlobalRegistry *GR = TM->getSubtargetImpl()->getSPIRVGlobalRegistry();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF) {
    MachineIRBuilder MIRBuilder(MBB, MBB.begin());

    MachineInstrBuilder MIB = MIRBuilder.buildInstr(SPIRV::OpString);
    Register StrReg = MRI.createVirtualRegister(&SPIRV::IDRegClass);
    MachineOperand StrRegOp = MachineOperand::CreateReg(StrReg, true);
    MIB.add(StrRegOp);
    addStringImm(FilePath, MIB);

    const MachineInstr *VoidTyMI =
        GR->getOrCreateSPIRVType(Type::getVoidTy(M->getContext()), MIRBuilder);

    MIB = MIRBuilder.buildInstr(SPIRV::OpExtInst);
    Register DebugSourceResIdReg =
        MRI.createVirtualRegister(&SPIRV::IDRegClass);
    MIB.addDef(DebugSourceResIdReg);              // Result ID
    MIB.addUse(VoidTyMI->getOperand(0).getReg()); // Result Type
    MIB.addImm(static_cast<int64_t>(
        SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100)); // Set ID
    MIB.addImm(SPIRV::NonSemanticExtInst::DebugSource);            //
    MIB.addUse(StrReg);
    MIB.constrainAllUses(*TII, *TRI, *RBI);

    Register DwarfVersionReg = GR->buildConstantInt(DwarfVersion, MIRBuilder);
    Register DebugInfoVersionReg =
        GR->buildConstantInt(DebugInfoVersion, MIRBuilder);
    Register SourceLanguageReg =
        GR->buildConstantInt(SourceLanguage, MIRBuilder);

    MIB = MIRBuilder.buildInstr(SPIRV::OpExtInst);
    Register DebugCompUnitResIdReg =
        MRI.createVirtualRegister(&SPIRV::IDRegClass);
    MIB.addDef(DebugCompUnitResIdReg);            // Result ID
    MIB.addUse(VoidTyMI->getOperand(0).getReg()); // Result Type
    MIB.addImm(static_cast<int64_t>(
        SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100)); // Set ID
    MIB.addImm(SPIRV::NonSemanticExtInst::DebugCompilationUnit);
    MIB.addUse(DebugInfoVersionReg);
    MIB.addUse(DwarfVersionReg);
    MIB.addUse(DebugSourceResIdReg);
    MIB.addUse(SourceLanguageReg);
    MIB.constrainAllUses(*TII, *TRI, *RBI);
  }

  return true;
}

bool SPIRVEmitNonSemanticDI::runOnMachineFunction(MachineFunction &MF) {
  bool Res = false;
  if (!IsGlobalDIEmitted) {
    Res = emitGlobalDI(MF);
    IsGlobalDIEmitted = true;
  }
  return Res;
}
