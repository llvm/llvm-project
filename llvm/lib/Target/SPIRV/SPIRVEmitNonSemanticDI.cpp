#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVTargetMachine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
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

enum BaseTypeAttributeEncoding {
  Unspecified = 0,
  Address = 1,
  Boolean = 2,
  Float = 3,
  Signed = 4,
  SignedChar = 5,
  Unsigned = 6,
  UnsignedChar = 7
};

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
  SmallPtrSet<DIBasicType *, 12> BasicTypes;
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

    // This traversal is the only supported way to access
    // instruction related DI metadata like DIBasicType
    for (auto &F : *M) {
      for (auto &BB : F) {
        for (auto &I : BB) {
          for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
            DILocalVariable *LocalVariable = DVR.getVariable();
            if (auto *BasicType =
                    dyn_cast<DIBasicType>(LocalVariable->getType()))
              BasicTypes.insert(BasicType);
          }
        }
      }
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

    const auto EmitOpString = [&](StringRef SR) {
      const Register StrReg = MRI.createVirtualRegister(&SPIRV::IDRegClass);
      MRI.setType(StrReg, LLT::scalar(32));
      MachineInstrBuilder MIB = MIRBuilder.buildInstr(SPIRV::OpString);
      MIB.addDef(StrReg);
      addStringImm(SR, MIB);
      return StrReg;
    };

    // Emit OpString with FilePath which is required by DebugSource
    const Register FilePathStrReg = EmitOpString(FilePath);

    const SPIRVType *VoidTy =
        GR->getOrCreateSPIRVType(Type::getVoidTy(*Context), MIRBuilder);

    const auto EmitDIInstruction =
        [&](SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst,
            std::initializer_list<Register> Registers) {
          const Register InstReg =
              MRI.createVirtualRegister(&SPIRV::IDRegClass);
          MRI.setType(InstReg, LLT::scalar(32));
          MachineInstrBuilder MIB =
              MIRBuilder.buildInstr(SPIRV::OpExtInst)
                  .addDef(InstReg)
                  .addUse(GR->getSPIRVTypeID(VoidTy))
                  .addImm(static_cast<int64_t>(
                      SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100))
                  .addImm(Inst);
          for (auto Reg : Registers) {
            MIB.addUse(Reg);
          }
          MIB.constrainAllUses(*TII, *TRI, *RBI);
          GR->assignSPIRVTypeToVReg(VoidTy, InstReg, MF);
          return InstReg;
        };

    // Emit DebugSource which is required by DebugCompilationUnit
    const Register DebugSourceResIdReg = EmitDIInstruction(
        SPIRV::NonSemanticExtInst::DebugSource, {FilePathStrReg});

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

    [[maybe_unused]]
    const Register DebugCompUnitResIdReg =
        EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugCompilationUnit,
                          {DebugInfoVersionReg, DwarfVersionReg,
                           DebugSourceResIdReg, SourceLanguageReg});

    // We aren't extracting any DebugInfoFlags now so we
    // emitting zero to use as <id>Flags argument for DebugBasicType
    const Register I32ZeroReg =
        GR->buildConstantInt(0, MIRBuilder, I32Ty, false);

    for (auto *BasicType : BasicTypes) {
      const Register BasicTypeStrReg = EmitOpString(BasicType->getName());

      const Register ConstIntBitwidthReg = GR->buildConstantInt(
          BasicType->getSizeInBits(), MIRBuilder, I32Ty, false);

      uint64_t AttributeEncoding = BaseTypeAttributeEncoding::Unspecified;
      switch (BasicType->getEncoding()) {
      case dwarf::DW_ATE_signed:
        AttributeEncoding = BaseTypeAttributeEncoding::Signed;
        break;
      case dwarf::DW_ATE_unsigned:
        AttributeEncoding = BaseTypeAttributeEncoding::Unsigned;
        break;
      case dwarf::DW_ATE_unsigned_char:
        AttributeEncoding = BaseTypeAttributeEncoding::UnsignedChar;
        break;
      case dwarf::DW_ATE_signed_char:
        AttributeEncoding = BaseTypeAttributeEncoding::SignedChar;
        break;
      case dwarf::DW_ATE_float:
        AttributeEncoding = BaseTypeAttributeEncoding::Float;
        break;
      case dwarf::DW_ATE_boolean:
        AttributeEncoding = BaseTypeAttributeEncoding::Boolean;
        break;
      case dwarf::DW_ATE_address:
        AttributeEncoding = BaseTypeAttributeEncoding::Address;
      }

      const Register AttributeEncodingReg =
          GR->buildConstantInt(AttributeEncoding, MIRBuilder, I32Ty, false);

      [[maybe_unused]]
      const Register BasicTypeReg =
          EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypeBasic,
                            {BasicTypeStrReg, ConstIntBitwidthReg,
                             AttributeEncodingReg, I32ZeroReg});
    }
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
