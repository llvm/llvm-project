#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
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

enum SourceLanguage {
  Unknown = 0,
  ESSL = 1,
  GLSL = 2,
  OpenCL_C = 3,
  OpenCL_CPP = 4,
  HLSL = 5,
  CPP_for_OpenCL = 6,
  SYCL = 7,
  HERO_C = 8,
  NZSL = 9,
  WGSL = 10,
  Slang = 11,
  Zig = 12
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
  SmallVector<SmallString<128>> FilePaths;
  SmallVector<int64_t> LLVMSourceLanguages;
  int64_t DwarfVersion = 0;
  int64_t DebugInfoVersion = 0;
  SmallPtrSet<DIBasicType *, 12> BasicTypes;
  SmallPtrSet<DIDerivedType *, 12> PointerDerivedTypes;
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
        FilePaths.emplace_back();
        sys::path::append(FilePaths.back(), File->getDirectory(),
                          File->getFilename());
        LLVMSourceLanguages.push_back(CompileUnit->getSourceLanguage());
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
                    dyn_cast<DIBasicType>(LocalVariable->getType())) {
              BasicTypes.insert(BasicType);
            } else if (auto *DerivedType =
                           dyn_cast<DIDerivedType>(LocalVariable->getType())) {
              if (DerivedType->getTag() == dwarf::DW_TAG_pointer_type) {
                PointerDerivedTypes.insert(DerivedType);
                // DIBasicType can be unreachable from DbgRecord and only
                // pointed on from other DI types
                // DerivedType->getBaseType is null when pointer
                // is representing a void type
                if (DerivedType->getBaseType())
                  BasicTypes.insert(
                      cast<DIBasicType>(DerivedType->getBaseType()));
              }
            }
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

    const SPIRVType *I32Ty =
        GR->getOrCreateSPIRVType(Type::getInt32Ty(*Context), MIRBuilder);

    const Register DwarfVersionReg =
        GR->buildConstantInt(DwarfVersion, MIRBuilder, I32Ty, false);

    const Register DebugInfoVersionReg =
        GR->buildConstantInt(DebugInfoVersion, MIRBuilder, I32Ty, false);

    for (unsigned Idx = 0; Idx < LLVMSourceLanguages.size(); ++Idx) {
      const Register FilePathStrReg = EmitOpString(FilePaths[Idx]);

      const Register DebugSourceResIdReg = EmitDIInstruction(
          SPIRV::NonSemanticExtInst::DebugSource, {FilePathStrReg});

      SourceLanguage SpirvSourceLanguage = SourceLanguage::Unknown;
      switch (LLVMSourceLanguages[Idx]) {
      case dwarf::DW_LANG_OpenCL:
        SpirvSourceLanguage = SourceLanguage::OpenCL_C;
        break;
      case dwarf::DW_LANG_OpenCL_CPP:
        SpirvSourceLanguage = SourceLanguage::OpenCL_CPP;
        break;
      case dwarf::DW_LANG_CPP_for_OpenCL:
        SpirvSourceLanguage = SourceLanguage::CPP_for_OpenCL;
        break;
      case dwarf::DW_LANG_GLSL:
        SpirvSourceLanguage = SourceLanguage::GLSL;
        break;
      case dwarf::DW_LANG_HLSL:
        SpirvSourceLanguage = SourceLanguage::HLSL;
        break;
      case dwarf::DW_LANG_SYCL:
        SpirvSourceLanguage = SourceLanguage::SYCL;
        break;
      case dwarf::DW_LANG_Zig:
        SpirvSourceLanguage = SourceLanguage::Zig;
      }

      const Register SourceLanguageReg =
          GR->buildConstantInt(SpirvSourceLanguage, MIRBuilder, I32Ty, false);

      [[maybe_unused]]
      const Register DebugCompUnitResIdReg =
          EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugCompilationUnit,
                            {DebugInfoVersionReg, DwarfVersionReg,
                             DebugSourceResIdReg, SourceLanguageReg});
    }

    // We aren't extracting any DebugInfoFlags now so we
    // emitting zero to use as <id>Flags argument for DebugBasicType
    const Register I32ZeroReg =
        GR->buildConstantInt(0, MIRBuilder, I32Ty, false, false);

    // We need to store pairs because further instructions reference
    // the DIBasicTypes and size will be always small so there isn't
    // need for any kind of map
    SmallVector<std::pair<const DIBasicType *const, const Register>, 12>
        BasicTypeRegPairs;
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

      const Register BasicTypeReg =
          EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugTypeBasic,
                            {BasicTypeStrReg, ConstIntBitwidthReg,
                             AttributeEncodingReg, I32ZeroReg});
      BasicTypeRegPairs.emplace_back(BasicType, BasicTypeReg);
    }

    if (PointerDerivedTypes.size()) {
      for (const auto *PointerDerivedType : PointerDerivedTypes) {

        assert(PointerDerivedType->getDWARFAddressSpace().has_value());
        const Register StorageClassReg = GR->buildConstantInt(
            addressSpaceToStorageClass(
                PointerDerivedType->getDWARFAddressSpace().value(),
                *TM->getSubtargetImpl()),
            MIRBuilder, I32Ty, false);

        // If the Pointer is representing a void type it's getBaseType
        // is a nullptr
        const auto *MaybeNestedBasicType =
            cast_or_null<DIBasicType>(PointerDerivedType->getBaseType());
        if (MaybeNestedBasicType) {
          for (const auto &BasicTypeRegPair : BasicTypeRegPairs) {
            const auto &[DefinedBasicType, BasicTypeReg] = BasicTypeRegPair;
            if (DefinedBasicType == MaybeNestedBasicType) {
              [[maybe_unused]]
              const Register DebugPointerTypeReg = EmitDIInstruction(
                  SPIRV::NonSemanticExtInst::DebugTypePointer,
                  {BasicTypeReg, StorageClassReg, I32ZeroReg});
            }
          }
        } else {
          const Register DebugInfoNoneReg =
              EmitDIInstruction(SPIRV::NonSemanticExtInst::DebugInfoNone, {});
          [[maybe_unused]]
          const Register DebugPointerTypeReg = EmitDIInstruction(
              SPIRV::NonSemanticExtInst::DebugTypePointer,
              {DebugInfoNoneReg, StorageClassReg, I32ZeroReg});
        }
      }
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
