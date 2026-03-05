#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVectorExtras.h"
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "spirv-nonsemantic-debug-info"

using namespace llvm;

namespace {
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

struct SPIRVEmitNonSemanticDI : public MachineFunctionPass {
  static char ID;
  SPIRVTargetMachine *TM;
  DenseMap<const DICompileUnit *, Register> CompileUnitRegMap;
  SPIRVEmitNonSemanticDI(SPIRVTargetMachine *TM = nullptr)
      : MachineFunctionPass(ID), TM(TM) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool emitGlobalDI(MachineFunction &MF);
  static SourceLanguage
  convertDWARFToSPIRVSourceLanguage(int64_t LLVMSourceLanguage);
  static Register emitOpString(MachineRegisterInfo &MRI,
                               MachineIRBuilder &MIRBuilder, StringRef SR);
  static Register
  emitDIInstruction(MachineRegisterInfo &MRI, MachineIRBuilder &MIRBuilder,
                    SPIRVGlobalRegistry *GR, const SPIRVTypeInst &VoidTy,
                    const SPIRVInstrInfo *TII, const SPIRVRegisterInfo *TRI,
                    const RegisterBankInfo *RBI, MachineFunction &MF,
                    SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst,
                    ArrayRef<Register> Registers);
};
} // anonymous namespace

INITIALIZE_PASS(SPIRVEmitNonSemanticDI, DEBUG_TYPE,
                "SPIRV NonSemantic.Shader.DebugInfo.100 emitter", false, false)

char SPIRVEmitNonSemanticDI::ID = 0;

MachineFunctionPass *
llvm::createSPIRVEmitNonSemanticDIPass(SPIRVTargetMachine *TM) {
  return new SPIRVEmitNonSemanticDI(TM);
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

SourceLanguage SPIRVEmitNonSemanticDI::convertDWARFToSPIRVSourceLanguage(
    int64_t LLVMSourceLanguage) {
  switch (LLVMSourceLanguage) {
  case dwarf::DW_LANG_OpenCL:
    return SourceLanguage::OpenCL_C;
  case dwarf::DW_LANG_OpenCL_CPP:
    return SourceLanguage::OpenCL_CPP;
  case dwarf::DW_LANG_CPP_for_OpenCL:
    return SourceLanguage::CPP_for_OpenCL;
  case dwarf::DW_LANG_GLSL:
    return SourceLanguage::GLSL;
  case dwarf::DW_LANG_HLSL:
    return SourceLanguage::HLSL;
  case dwarf::DW_LANG_SYCL:
    return SourceLanguage::SYCL;
  case dwarf::DW_LANG_Zig:
    return SourceLanguage::Zig;
  default:
    return SourceLanguage::Unknown;
  }
}

static const Module *getModule(MachineFunction &MF) {
  return MF.getFunction().getParent();
}

Register SPIRVEmitNonSemanticDI::emitOpString(MachineRegisterInfo &MRI,
                                              MachineIRBuilder &MIRBuilder,
                                              StringRef SR) {
  const Register StrReg = MRI.createVirtualRegister(&SPIRV::IDRegClass);
  MRI.setType(StrReg, LLT::scalar(32));
  MachineInstrBuilder MIB = MIRBuilder.buildInstr(SPIRV::OpString);
  MIB.addDef(StrReg);
  addStringImm(SR, MIB);
  return StrReg;
}

Register SPIRVEmitNonSemanticDI::emitDIInstruction(
    MachineRegisterInfo &MRI, MachineIRBuilder &MIRBuilder,
    SPIRVGlobalRegistry *GR, const SPIRVTypeInst &VoidTy,
    const SPIRVInstrInfo *TII, const SPIRVRegisterInfo *TRI,
    const RegisterBankInfo *RBI, MachineFunction &MF,
    SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst,
    ArrayRef<Register> Registers) {
  const Register InstReg = MRI.createVirtualRegister(&SPIRV::IDRegClass);
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
}

bool SPIRVEmitNonSemanticDI::emitGlobalDI(MachineFunction &MF) {
  // If this MachineFunction doesn't have any BB repeat procedure
  // for the next
  if (MF.begin() == MF.end()) {
    return false;
  }

  // Required variables to get from metadata search
  LLVMContext *Context;
  SmallVector<const DICompileUnit *> CompileUnits;
  int64_t DwarfVersion = 0;
  int64_t DebugInfoVersion = 0;
  SetVector<DIBasicType *> BasicTypes;
  SetVector<DIDerivedType *> PointerDerivedTypes;
  // Searching through the Module metadata to find nescessary
  // information like DwarfVersion or SourceLanguage
  {
    const Module *M = getModule(MF);
    Context = &M->getContext();
    const NamedMDNode *DbgCu = M->getNamedMetadata("llvm.dbg.cu");
    if (!DbgCu)
      return false;
    CompileUnits = map_to_vector(
        make_filter_range(DbgCu->operands(), llvm::IsaPred<DICompileUnit>),
        llvm::CastTo<DICompileUnit>);
    const NamedMDNode *ModuleFlags = M->getNamedMetadata("llvm.module.flags");
    assert(ModuleFlags && "Expected llvm.module.flags metadata to be present");
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
                if (auto *BT = dyn_cast_or_null<DIBasicType>(
                        DerivedType->getBaseType()))
                  BasicTypes.insert(BT);
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

    const SPIRVTypeInst VoidTy =
        GR->getOrCreateSPIRVType(Type::getVoidTy(*Context), MIRBuilder,
                                 SPIRV::AccessQualifier::ReadWrite, false);

    const SPIRVTypeInst I32Ty =
        GR->getOrCreateSPIRVType(Type::getInt32Ty(*Context), MIRBuilder,
                                 SPIRV::AccessQualifier::ReadWrite, false);

    const Register DwarfVersionReg =
        GR->buildConstantInt(DwarfVersion, MIRBuilder, I32Ty, false);

    const Register DebugInfoVersionReg =
        GR->buildConstantInt(DebugInfoVersion, MIRBuilder, I32Ty, false);

    for (const DICompileUnit *CompileUnit : CompileUnits) {
      const DIFile *File = CompileUnit->getFile();
      SmallString<128> FilePath;
      sys::path::append(FilePath, File->getDirectory(), File->getFilename());
      const Register FilePathStrReg = emitOpString(MRI, MIRBuilder, FilePath);

      const Register DebugSourceResIdReg = emitDIInstruction(
          MRI, MIRBuilder, GR, VoidTy, TII, TRI, RBI, MF,
          SPIRV::NonSemanticExtInst::DebugSource, {FilePathStrReg});

      SourceLanguage SpirvSourceLanguage = convertDWARFToSPIRVSourceLanguage(
          CompileUnit->getSourceLanguage().getUnversionedName());

      const Register SourceLanguageReg =
          GR->buildConstantInt(SpirvSourceLanguage, MIRBuilder, I32Ty, false);

      const Register DebugCompUnitResIdReg =
          emitDIInstruction(MRI, MIRBuilder, GR, VoidTy, TII, TRI, RBI, MF,
                            SPIRV::NonSemanticExtInst::DebugCompilationUnit,
                            {DebugInfoVersionReg, DwarfVersionReg,
                             DebugSourceResIdReg, SourceLanguageReg});

      // Store the register for this compile unit.
      CompileUnitRegMap[CompileUnit] = DebugCompUnitResIdReg;
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
      const Register BasicTypeStrReg =
          emitOpString(MRI, MIRBuilder, BasicType->getName());

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
          emitDIInstruction(MRI, MIRBuilder, GR, VoidTy, TII, TRI, RBI, MF,
                            SPIRV::NonSemanticExtInst::DebugTypeBasic,
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
            dyn_cast_or_null<DIBasicType>(PointerDerivedType->getBaseType());
        if (MaybeNestedBasicType) {
          for (const auto &BasicTypeRegPair : BasicTypeRegPairs) {
            const auto &[DefinedBasicType, BasicTypeReg] = BasicTypeRegPair;
            if (DefinedBasicType == MaybeNestedBasicType) {
              [[maybe_unused]] const Register DebugPointerTypeReg =
                  emitDIInstruction(
                      MRI, MIRBuilder, GR, VoidTy, TII, TRI, RBI, MF,
                      SPIRV::NonSemanticExtInst::DebugTypePointer,
                      {BasicTypeReg, StorageClassReg, I32ZeroReg});
            }
          }
        } else {
          const Register DebugInfoNoneReg =
              emitDIInstruction(MRI, MIRBuilder, GR, VoidTy, TII, TRI, RBI, MF,
                                SPIRV::NonSemanticExtInst::DebugInfoNone, {});
          [[maybe_unused]] const Register DebugPointerTypeReg =
              emitDIInstruction(
                  MRI, MIRBuilder, GR, VoidTy, TII, TRI, RBI, MF,
                  SPIRV::NonSemanticExtInst::DebugTypePointer,
                  {DebugInfoNoneReg, StorageClassReg, I32ZeroReg});
        }
      }
    }
  }
  return true;
}

bool SPIRVEmitNonSemanticDI::runOnMachineFunction(MachineFunction &MF) {
  CompileUnitRegMap.clear();
  bool Res = emitGlobalDI(MF);
  return Res;
}
