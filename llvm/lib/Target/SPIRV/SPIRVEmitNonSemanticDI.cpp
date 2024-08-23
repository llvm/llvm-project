#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVTargetMachine.h"
#include "llvm/ADT/SmallString.h"
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
} // namespace llvm

using namespace llvm;

INITIALIZE_PASS(SPIRVEmitNonSemanticDI, DEBUG_TYPE,
                "SPIRV NonSemantic.Shader.DebugInfo.100 emitter", false, false)

char SPIRVEmitNonSemanticDI::ID = 0;

MachineFunctionPass *
llvm::createSPIRVEmitNonSemanticDIPass(SPIRVTargetMachine *TM) {
  return new SPIRVEmitNonSemanticDI(TM);
}

SPIRVEmitNonSemanticDI::SPIRVEmitNonSemanticDI(SPIRVTargetMachine *TM)
    : MachineFunctionPass(ID), TM(TM) {
  initializeSPIRVEmitNonSemanticDIPass(*PassRegistry::getPassRegistry());
}

SPIRVEmitNonSemanticDI::SPIRVEmitNonSemanticDI() : MachineFunctionPass(ID) {
  initializeSPIRVEmitNonSemanticDIPass(*PassRegistry::getPassRegistry());
}

bool SPIRVEmitNonSemanticDI::emitGlobalDI(MachineFunction &MF) {
  // If this MachineFunction doesn't have any BB repeat procedure
  // for the next
  if (MF.begin() == MF.end()) {
    IsGlobalDIEmitted = false;
    return false;
  }

  // Required variables to get from metadata search
  LLVMContext *Context;
  SmallString<128> FilePath;
  unsigned SourceLanguage = 0;
  int64_t DwarfVersion = 0;
  int64_t DebugInfoVersion = 0;

  // Searching through the Module metadata to find nescessary
  // information like DwarfVersion or SourceLanguage
  {
    const MachineModuleInfo &MMI =
        getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
    const Module *M = MMI.getModule();
    Context = &M->getContext();
    const NamedMDNode *DbgCu = M->getNamedMetadata("llvm.dbg.cu");
    if (!DbgCu)
      return false;
    for (const auto *Op : DbgCu->operands()) {
      if (const auto *CompileUnit = dyn_cast<DICompileUnit>(Op)) {
        DIFile *File = CompileUnit->getFile();
        sys::path::append(FilePath, File->getDirectory(), File->getFilename());
        SourceLanguage = CompileUnit->getSourceLanguage();
        break;
      }
    }
    const NamedMDNode *ModuleFlags = M->getNamedMetadata("llvm.module.flags");
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
  }
  // NonSemantic.Shader.DebugInfo.100 global DI instruction emitting
  {
    // Required LLVM variables for emitting logic
    const SPIRVInstrInfo *TII = TM->getSubtargetImpl()->getInstrInfo();
    const SPIRVRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
    const RegisterBankInfo *RBI = TM->getSubtargetImpl()->getRegBankInfo();
    SPIRVGlobalRegistry *GR = TM->getSubtargetImpl()->getSPIRVGlobalRegistry();
    MachineRegisterInfo &MRI = MF.getRegInfo();
    MachineBasicBlock &MBB = *MF.begin();

    // To correct placement of a OpLabel instruction during SPIRVAsmPrinter
    // emission all new instructions needs to be placed after OpFunction
    // and before first terminator
    MachineIRBuilder MIRBuilder(MBB, MBB.getFirstTerminator());

    // Emit OpString with FilePath which is required by DebugSource
    const Register StrReg = MRI.createVirtualRegister(&SPIRV::IDRegClass);
    MRI.setType(StrReg, LLT::scalar(32));
    MachineInstrBuilder MIB = MIRBuilder.buildInstr(SPIRV::OpString);
    MIB.addDef(StrReg);
    addStringImm(FilePath, MIB);

    const SPIRVType *VoidTy =
        GR->getOrCreateSPIRVType(Type::getVoidTy(*Context), MIRBuilder);

    // Emit DebugSource which is required by DebugCompilationUnit
    const Register DebugSourceResIdReg =
        MRI.createVirtualRegister(&SPIRV::IDRegClass);
    MRI.setType(DebugSourceResIdReg, LLT::scalar(32));
    MIB = MIRBuilder.buildInstr(SPIRV::OpExtInst)
              .addDef(DebugSourceResIdReg)
              .addUse(GR->getSPIRVTypeID(VoidTy))
              .addImm(static_cast<int64_t>(
                  SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100))
              .addImm(SPIRV::NonSemanticExtInst::DebugSource)
              .addUse(StrReg);
    MIB.constrainAllUses(*TII, *TRI, *RBI);
    GR->assignSPIRVTypeToVReg(VoidTy, DebugSourceResIdReg, MF);

    const SPIRVType *I32Ty =
        GR->getOrCreateSPIRVType(Type::getInt32Ty(*Context), MIRBuilder);

    // Convert DwarfVersion, DebugInfo and SourceLanguage integers to OpConstant
    // instructions required by DebugCompilationUnit
    const Register DwarfVersionReg =
        GR->buildConstantInt(DwarfVersion, MIRBuilder, I32Ty, false);
    const Register DebugInfoVersionReg =
        GR->buildConstantInt(DebugInfoVersion, MIRBuilder, I32Ty, false);
    const Register SourceLanguageReg =
        GR->buildConstantInt(SourceLanguage, MIRBuilder, I32Ty, false);

    // Emit DebugCompilationUnit
    const Register DebugCompUnitResIdReg =
        MRI.createVirtualRegister(&SPIRV::IDRegClass);
    MRI.setType(DebugCompUnitResIdReg, LLT::scalar(32));
    MIB = MIRBuilder.buildInstr(SPIRV::OpExtInst)
              .addDef(DebugCompUnitResIdReg)
              .addUse(GR->getSPIRVTypeID(VoidTy))
              .addImm(static_cast<int64_t>(
                  SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100))
              .addImm(SPIRV::NonSemanticExtInst::DebugCompilationUnit)
              .addUse(DebugInfoVersionReg)
              .addUse(DwarfVersionReg)
              .addUse(DebugSourceResIdReg)
              .addUse(SourceLanguageReg);
    MIB.constrainAllUses(*TII, *TRI, *RBI);
    GR->assignSPIRVTypeToVReg(VoidTy, DebugCompUnitResIdReg, MF);
  }
  return true;
}

bool SPIRVEmitNonSemanticDI::runOnMachineFunction(MachineFunction &MF) {
  bool Res = false;
  // emitGlobalDI needs to be executed only once to avoid
  // emitting duplicates
  if (!IsGlobalDIEmitted) {
    IsGlobalDIEmitted = true;
    Res = emitGlobalDI(MF);
  }
  return Res;
}
