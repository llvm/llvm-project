#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "spirv-nonsemantic-debug-info"

namespace {
using namespace llvm;

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

enum TypesMapping {
  PrimitiveIntArray,
  PrimitiveStringArray,
  DebugSourceArray,
  DebugCompilationUnitArray,
  DebugTypeBasicArray,
  DebugTypePointerArray,
  DebugInfoNoneArray,
  DebugLineArray
};

struct DebugSource {
  size_t FileId;

  explicit constexpr DebugSource(const size_t FileId) noexcept
      : FileId(FileId) {}

  explicit constexpr DebugSource(const ArrayRef<size_t> AR) : FileId(AR[0]) {}

  friend bool operator==(const DebugSource &Lhs, const DebugSource &Rhs) {
    return Lhs.FileId == Rhs.FileId;
  }
};

struct DebugCompilationUnit {
  size_t DebugInfoVersionId;
  size_t DwarfVersionId;
  size_t DebugSourceId;
  size_t LanguageId;

  constexpr DebugCompilationUnit(const size_t DebugInfoVersionId,
                                 const size_t DwarfVersionId,
                                 const size_t DebugSourceId,
                                 const size_t LanguageId) noexcept
      : DebugInfoVersionId(DebugInfoVersionId), DwarfVersionId(DwarfVersionId),
        DebugSourceId(DebugSourceId), LanguageId(LanguageId) {}

  explicit constexpr DebugCompilationUnit(const ArrayRef<size_t> AR) noexcept
      : DebugInfoVersionId(AR[0]), DwarfVersionId(AR[1]), DebugSourceId(AR[2]),
        LanguageId(AR[3]) {}

  friend bool operator==(const DebugCompilationUnit &Lhs,
                         const DebugCompilationUnit &Rhs) {
    return Lhs.DebugInfoVersionId == Rhs.DebugInfoVersionId &
           Lhs.DwarfVersionId == Rhs.DwarfVersionId &
           Lhs.DebugSourceId == Rhs.DebugSourceId &
           Lhs.LanguageId == Rhs.LanguageId;
  }
};

struct DebugTypeBasic {
  size_t NameId;
  size_t SizeId;
  size_t BaseTypeEncodingId;
  size_t FlagsId;

  explicit constexpr DebugTypeBasic(const ArrayRef<size_t> AR)
      : NameId(AR[0]), SizeId(AR[1]), BaseTypeEncodingId(AR[2]),
        FlagsId(AR[3]) {}

  friend bool operator==(const DebugTypeBasic &Lhs, const DebugTypeBasic &Rhs) {
    return Lhs.NameId == Rhs.NameId && Lhs.SizeId == Rhs.SizeId &&
           Lhs.BaseTypeEncodingId == Rhs.BaseTypeEncodingId &&
           Lhs.FlagsId == Rhs.FlagsId;
  }
};

struct DebugTypePointer {
  size_t BaseTypeId;
  size_t StorageClassId;
  size_t FlagsId;

  DebugTypePointer(const size_t BaseTypeId, const size_t StorageClassId,
                   const size_t FlagsId) noexcept
      : BaseTypeId(BaseTypeId), StorageClassId(StorageClassId),
        FlagsId(FlagsId) {}

  explicit DebugTypePointer(const ArrayRef<size_t> AR) noexcept
      : BaseTypeId(AR[0]), StorageClassId(AR[1]), FlagsId(AR[2]) {}

  friend bool operator==(const DebugTypePointer &Lhs,
                         const DebugTypePointer &Rhs) {
    return Lhs.BaseTypeId == Rhs.BaseTypeId &&
           Lhs.StorageClassId == Rhs.StorageClassId &&
           Lhs.FlagsId == Rhs.FlagsId;
  }
};

struct DebugLine {
  size_t DebugSourceId;
  size_t LineStartId;
  size_t LineEndId;
  size_t ColumnStartId;
  size_t ColumnEndId;

  explicit DebugLine(const ArrayRef<size_t> AR)
      : DebugSourceId(AR[0]), LineStartId(AR[1]), LineEndId(AR[2]),
        ColumnStartId(AR[3]), ColumnEndId(AR[4]) {}

  friend bool operator==(const DebugLine &Lhs, const DebugLine &Rhs) {
    return Lhs.DebugSourceId == Rhs.DebugSourceId &&
           Lhs.LineStartId == Rhs.LineStartId &&
           Lhs.LineEndId == Rhs.LineEndId &&
           Lhs.ColumnStartId == Rhs.ColumnStartId &&
           Lhs.ColumnEndId == Rhs.ColumnEndId;
  }
};

struct DebugInfoNone;

template <typename T> struct DebugTypeContainer;

template <> struct DebugTypeContainer<int64_t> {
  static constexpr TypesMapping TM = PrimitiveIntArray;
};

template <> struct DebugTypeContainer<StringRef> {
  static constexpr TypesMapping TM = PrimitiveStringArray;
};

template <> struct DebugTypeContainer<DebugSource> {
  static constexpr TypesMapping TM = DebugSourceArray;
  static constexpr SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst =
      SPIRV::NonSemanticExtInst::DebugSource;
};

template <> struct DebugTypeContainer<DebugCompilationUnit> {
  static constexpr TypesMapping TM = DebugCompilationUnitArray;
  static constexpr SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst =
      SPIRV::NonSemanticExtInst::DebugCompilationUnit;
};

template <> struct DebugTypeContainer<DebugTypeBasic> {
  static constexpr TypesMapping TM = DebugTypeBasicArray;
  static constexpr SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst =
      SPIRV::NonSemanticExtInst::DebugTypeBasic;
};

template <> struct DebugTypeContainer<DebugTypePointer> {
  static constexpr TypesMapping TM = DebugTypePointerArray;
  static constexpr SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst =
      SPIRV::NonSemanticExtInst::DebugTypePointer;
};

template <> struct DebugTypeContainer<DebugInfoNone> {
  static constexpr TypesMapping TM = DebugInfoNoneArray;
  static constexpr SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst =
      SPIRV::NonSemanticExtInst::DebugInfoNone;
};

template <> struct DebugTypeContainer<DebugLine> {
  static constexpr TypesMapping TM = DebugLineArray;
  static constexpr SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst =
      SPIRV::NonSemanticExtInst::DebugLine;
};

template <typename T> struct DebugLiveContainer {
  SmallVector<T> Values;
  SmallVector<size_t> BackIdx;
};
///  @class LiveRepository
///  @brief Container for the connection graph between newly emitted Debug
///  instructions.
///
///  The architecture consists of two tiers:
///  - The **Main tier**, represented by an array of `Instructions`.
///  - The **Concrete Type tier**, represented by separate arrays for each
///  corresponding debug type.
///
///  The `Instructions` array contains records that include:
///  - `Type`: An enum value pointing to the specific debug type array in the
///  Concrete Type tier.
///  - `Id`: An index into the corresponding array in the Concrete Type tier.
///
///  The class is designed to ensure safe and controlled access to its internal
///  state. All interactions with the repository are done through the
///  `getOrCreateIdx` function, which restricts direct manipulation of the
///  underlying data structures, thus minimizing the risk of memory errors or
///  logical mistakes.
///
///  Users of this class interact exclusively with the indexes of the
///  `Instructions` array and never directly access the internal concrete type
///  arrays. This property flattens the instruction access,
//   making it easily retrievable.
///
///
///       getOrCreateIdx() <--+ Iidx
///             V             |
/// +-------------------------+-+
/// |[Instructions]--{Cidx,Type}|<----+
/// +------+--------------+-----+     |
///        v              v           |
/// [PrimitiveType]  [ConcreteType_0] |
///      {Data}        {Instruction}  |
///    (i32)(i32)      (Iidx) (Iidx)  |
///                        |    |     |
///                        +----+-----+
///
///  Class is designed to be easily removable by simply replacing the
///  `getOrConstructIdx` method with another mechanism.
class LiveRepository {
  DebugLiveContainer<int64_t> PrimitiveInts;
  DebugLiveContainer<StringRef> PrimitiveStrings;
  DebugLiveContainer<DebugSource> DebugSources;
  DebugLiveContainer<DebugLine> DebugLines;
  DebugLiveContainer<DebugCompilationUnit> DebugCompilationUnits;
  DebugLiveContainer<DebugTypeBasic> DebugTypeBasics;
  DebugLiveContainer<DebugTypePointer> DebugTypePointers;

  SmallVector<Register> Registers;
  SmallVector<std::pair<TypesMapping, unsigned>> Instructions;

  template <typename T> constexpr SmallVector<T> &values() {
    if constexpr (std::is_same_v<T, int64_t>) {
      return PrimitiveInts.Values;
    } else if constexpr (std::is_same_v<T, StringRef>) {
      return PrimitiveStrings.Values;
    } else if constexpr (std::is_same_v<T, DebugLine>) {
      return DebugLines.Values;
    } else if constexpr (std::is_same_v<T, DebugSource>) {
      return DebugSources.Values;
    } else if constexpr (std::is_same_v<T, DebugCompilationUnit>) {
      return DebugCompilationUnits.Values;
    } else if constexpr (std::is_same_v<T, DebugTypeBasic>) {
      return DebugTypeBasics.Values;
    } else if constexpr (std::is_same_v<T, DebugTypePointer>) {
      return DebugTypePointers.Values;
    }
    llvm_unreachable("unreachable");
  }

  template <typename T> constexpr SmallVector<size_t> &backIdx() {
    if constexpr (std::is_same_v<T, int64_t>) {
      return PrimitiveInts.BackIdx;
    } else if constexpr (std::is_same_v<T, StringRef>) {
      return PrimitiveStrings.BackIdx;
    } else if constexpr (std::is_same_v<T, DebugLine>) {
      return DebugLines.BackIdx;
    } else if constexpr (std::is_same_v<T, DebugSource>) {
      return DebugSources.BackIdx;
    } else if constexpr (std::is_same_v<T, DebugCompilationUnit>) {
      return DebugCompilationUnits.BackIdx;
    } else if constexpr (std::is_same_v<T, DebugTypeBasic>) {
      return DebugTypeBasics.BackIdx;
    } else if constexpr (std::is_same_v<T, DebugTypePointer>) {
      return DebugTypePointers.BackIdx;
    }
    llvm_unreachable("unreachable");
  }

  template <typename T>
  static std::pair<size_t, bool>
  emplaceOrReturnDuplicate(T Val, SmallVectorImpl<T> &SV) {
    for (unsigned Idx = 0; Idx < SV.size(); ++Idx) {
      if (Val == SV[Idx]) {
        return {Idx, true};
      }
    }
    SV.emplace_back(Val);
    return {SV.size() - 1, false};
  }

  static Register emitOpString(const StringRef SR,
                               MachineIRBuilder &MIRBuilder) {
    MachineRegisterInfo *MRI = MIRBuilder.getMRI();
    const Register StrReg = MRI->createVirtualRegister(&SPIRV::IDRegClass);
    MRI->setType(StrReg, LLT::scalar(32));
    MachineInstrBuilder MIB = MIRBuilder.buildInstr(SPIRV::OpString);
    MIB.addDef(StrReg);
    addStringImm(SR, MIB);
    return StrReg;
  }

  Register emitDIInstruction(SPIRV::NonSemanticExtInst::NonSemanticExtInst Inst,
                             const ArrayRef<size_t> Ids,
                             MachineIRBuilder &MIRBuilder,
                             const SPIRVTargetMachine *const TM) {
    const SPIRVInstrInfo *TII = TM->getSubtargetImpl()->getInstrInfo();
    const SPIRVRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
    const RegisterBankInfo *RBI = TM->getSubtargetImpl()->getRegBankInfo();
    SPIRVGlobalRegistry *GR = TM->getSubtargetImpl()->getSPIRVGlobalRegistry();
    MachineRegisterInfo *MRI = MIRBuilder.getMRI();
    const SPIRVType *VoidTy = GR->getOrCreateSPIRVType(
        Type::getVoidTy(MIRBuilder.getContext()), MIRBuilder);
    const Register InstReg = MRI->createVirtualRegister(&SPIRV::IDRegClass);
    MRI->setType(InstReg, LLT::scalar(32));
    MachineInstrBuilder MIB =
        MIRBuilder.buildInstr(SPIRV::OpExtInst)
            .addDef(InstReg)
            .addUse(GR->getSPIRVTypeID(VoidTy))
            .addImm(SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100)
            .addImm(Inst);
    for (const auto Idx : Ids) {
      MIB.addUse(Registers[Idx]);
    }
    MIB.constrainAllUses(*TII, *TRI, *RBI);
    GR->assignSPIRVTypeToVReg(VoidTy, InstReg, MIRBuilder.getMF());
    return InstReg;
  }

public:
  /// @brief Retrieves or creates an index for the int64_t primitive type.
  ///
  /// It's backed by OpConstantI instruction.
  /// It either retrieves an existing index or creates a new one based on the
  /// provided parameters. The function currently has a linear, cache-friendly
  /// search time for duplicate entries. It assumes that no single type of
  /// instruction will become so dominant in the module and every entry of that
  /// type will be unique.
  ///
  /// @param[in] Val A int64_t value to retreive or construct.
  /// @param[in] MIRBuilder A reference to the `MachineIRBuilder` used for
  /// constructing IR.
  /// @param[in] TM A pointer to the target machine (`SPIRVTargetMachine`) used
  /// for this operation.
  ///
  /// @return The index (`size_t`) of the retrieved duplicate or newly created
  /// entry.
  ///
  size_t getOrCreateIdx(const int64_t Val, MachineIRBuilder &MIRBuilder,
                        const SPIRVTargetMachine *TM) {
    auto &SV = values<int64_t>();
    const auto [ConcreteIdx, IsDuplicate] = emplaceOrReturnDuplicate(Val, SV);
    if (IsDuplicate) {
      return backIdx<int64_t>()[ConcreteIdx];
    }
    Instructions.emplace_back(DebugTypeContainer<int64_t>::TM, ConcreteIdx);
    SPIRVGlobalRegistry *GR = TM->getSubtargetImpl()->getSPIRVGlobalRegistry();
    const SPIRVType *I32Ty = GR->getOrCreateSPIRVType(
        Type::getInt32Ty(MIRBuilder.getContext()), MIRBuilder);
    Registers.emplace_back(GR->buildConstantInt(Val, MIRBuilder, I32Ty, false));
    backIdx<int64_t>().emplace_back(Instructions.size() - 1);
    return Instructions.size() - 1;
  }

  /// @brief Retrieves or creates an index for the StringRef primitive type.
  ///
  /// It's backed by OpString instruction.
  /// It either retrieves an existing index or creates a new one based on the
  /// provided parameters. The function currently has a linear, cache-friendly
  /// search time for duplicate entries. It assumes that no single type of
  /// instruction will become so dominant in the module and every entry of that
  /// type will be unique.
  ///
  /// @param[in] Val A StringRef value to retreive or construct.
  /// @param[in] MIRBuilder A reference to the `MachineIRBuilder` used for
  /// constructing IR.
  /// @param[in] TM A pointer to the target machine (`SPIRVTargetMachine`) used
  /// for this operation.
  ///
  /// @return The index (`size_t`) of the retrieved duplicate or newly created
  /// entry.
  ///
  size_t getOrCreateIdx(const StringRef Val, MachineIRBuilder &MIRBuilder) {
    auto &SV = values<StringRef>();
    const auto [ConcreteIdx, IsDuplicate] = emplaceOrReturnDuplicate(Val, SV);
    if (IsDuplicate) {
      return backIdx<StringRef>()[ConcreteIdx];
    }
    Instructions.emplace_back(DebugTypeContainer<StringRef>::TM, ConcreteIdx);
    Registers.emplace_back(emitOpString(Val, MIRBuilder));
    backIdx<StringRef>().emplace_back(Instructions.size() - 1);
    return Instructions.size() - 1;
  }

  /// @brief Retrieves or creates an index for the specified type.
  /// It either retrieves an existing index or creates a new one based on the
  /// provided parameters. The function currently has a linear, cache-friendly
  /// search time for duplicate entries. It assumes that no single type of
  /// instruction will become so dominant in the module and every entry of that
  /// type will be unique.:w
  ///
  /// @tparam T The specific type being used for this instantiation.
  ///
  /// @param[in] arrayRef A reference to an array containing indices (of type
  /// `size_t`).
  /// @param[in] MIRBuilder A reference to the `MachineIRBuilder` used for
  /// constructing IR.
  /// @param[in] TM A pointer to the target machine (`SPIRVTargetMachine`) used
  /// for this operation.
  ///
  /// @return The index (`size_t`) of the retrieved duplicate or newly created
  /// entry.
  ///
  template <typename T>
  constexpr size_t getOrCreateIdx(ArrayRef<size_t> Args,
                                  MachineIRBuilder &MIRBuilder,
                                  const SPIRVTargetMachine *TM) {
    auto &SV = values<T>();
    const auto [ConcreteIdx, IsDuplicate] =
        emplaceOrReturnDuplicate(T(Args), SV);
    if (IsDuplicate) {
      return backIdx<T>()[ConcreteIdx];
    }
    Instructions.emplace_back(DebugTypeContainer<T>::TM, ConcreteIdx);
    Registers.emplace_back(
        emitDIInstruction(DebugTypeContainer<T>::Inst, Args, MIRBuilder, TM));
    backIdx<T>().emplace_back(Instructions.size() - 1);
    return Instructions.size() - 1;
  }
};

template <>
size_t
LiveRepository::getOrCreateIdx<DebugInfoNone>(ArrayRef<size_t>,
                                              MachineIRBuilder &MIRBuilder,
                                              const SPIRVTargetMachine *TM) {
  static std::optional<size_t> DebugInfoNoneIdx = std::nullopt;
  if (!DebugInfoNoneIdx.has_value()) {
    Instructions.emplace_back(DebugTypeContainer<DebugInfoNone>::TM, 0);
    Registers.emplace_back(emitDIInstruction(
        DebugTypeContainer<DebugInfoNone>::Inst, {}, MIRBuilder, TM));
    DebugInfoNoneIdx.emplace(Instructions.size() - 1);
  }
  return DebugInfoNoneIdx.value();
}

size_t emitDebugSource(const DIFile *File, MachineIRBuilder &MIRBuilder,
                       SPIRVTargetMachine *TM, LiveRepository &LR) {
  SmallString<128> FilePath;
  sys::path::append(FilePath, File->getDirectory(), File->getFilename());
  const size_t FilePathId =
      LR.getOrCreateIdx(StringRef(FilePath.c_str()), MIRBuilder);
  return LR.getOrCreateIdx<DebugSource>({FilePathId}, MIRBuilder, TM);
}

size_t emitDebugCompilationUnits(const Module *M, MachineIRBuilder &MIRBuilder,
                                 const SPIRVTargetMachine *TM,
                                 LiveRepository &LR) {
  std::optional<size_t> DwarfVersionId = std::nullopt;
  std::optional<size_t> DebugInfoVersionId = std::nullopt;
  const NamedMDNode *ModuleFlags = M->getNamedMetadata("llvm.module.flags");
  for (const auto *Op : ModuleFlags->operands()) {
    const MDOperand &MaybeStrOp = Op->getOperand(1);
    if (MaybeStrOp.equalsStr("Dwarf Version")) {
      const int64_t DwarfVersion =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
              ->getSExtValue();
      DwarfVersionId = LR.getOrCreateIdx(DwarfVersion, MIRBuilder, TM);
    } else if (MaybeStrOp.equalsStr("Debug Info Version")) {
      const int64_t DebugInfoVersion =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
              ->getSExtValue();
      DebugInfoVersionId = LR.getOrCreateIdx(DebugInfoVersion, MIRBuilder, TM);
    }
  }
  assert(DwarfVersionId.has_value() && DebugInfoVersionId.has_value());
  const NamedMDNode *DbgCu = M->getNamedMetadata("llvm.dbg.cu");
  for (const auto *Op : DbgCu->operands()) {
    if (const auto *CompileUnit = dyn_cast<DICompileUnit>(Op)) {
      const DIFile *File = CompileUnit->getFile();
      SmallString<128> FilePath;
      sys::path::append(FilePath, File->getDirectory(), File->getFilename());

      const size_t FilePathId =
          LR.getOrCreateIdx(StringRef(FilePath.c_str()), MIRBuilder);
      const size_t DebugSourceId =
          LR.getOrCreateIdx<DebugSource>({FilePathId}, MIRBuilder, TM);

      SourceLanguage SpirvSourceLanguage;
      switch (CompileUnit->getSourceLanguage()) {
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
        break;
      default:
        SpirvSourceLanguage = SourceLanguage::Unknown;
      }

      const size_t SpirvSourceLanguageId =
          LR.getOrCreateIdx(SpirvSourceLanguage, MIRBuilder, TM);
      LR.getOrCreateIdx<DebugCompilationUnit>(
          {DebugInfoVersionId.value(), DwarfVersionId.value(), DebugSourceId,
           SpirvSourceLanguageId},
          MIRBuilder, TM);
    }
  }
  return 0;
}

size_t emitDebugTypeBasic(const DIBasicType *BT, size_t I32ZeroIdx,
                          MachineIRBuilder &MIRBuilder,
                          const SPIRVTargetMachine *TM, LiveRepository &LR) {

  const size_t BasicTypeStrId = LR.getOrCreateIdx(BT->getName(), MIRBuilder);

  const size_t ConstIntBitWidthId =
      LR.getOrCreateIdx(BT->getSizeInBits(), MIRBuilder, TM);

  uint64_t AttributeEncoding;
  switch (BT->getEncoding()) {
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
    break;
  default:
    AttributeEncoding = BaseTypeAttributeEncoding::Unspecified;
  }

  const size_t AttributeEncodingId =
      LR.getOrCreateIdx(AttributeEncoding, MIRBuilder, TM);

  return LR.getOrCreateIdx<DebugTypeBasic>(
      {BasicTypeStrId, ConstIntBitWidthId, AttributeEncodingId, I32ZeroIdx},
      MIRBuilder, TM);
}

size_t emitDebugTypePointer(const DIDerivedType *DT, const size_t BasicTypeIdx,
                            const size_t I32ZeroIdx,
                            MachineIRBuilder &MIRBuilder,
                            const SPIRVTargetMachine *TM, LiveRepository &LR) {
  assert(DT->getDWARFAddressSpace().has_value());

  size_t StorageClassIdx = LR.getOrCreateIdx(
      addressSpaceToStorageClass(DT->getDWARFAddressSpace().value(),
                                 *TM->getSubtargetImpl()),
      MIRBuilder, TM);

  return LR.getOrCreateIdx<DebugTypePointer>(
      {BasicTypeIdx, StorageClassIdx, I32ZeroIdx}, MIRBuilder, TM);
}

size_t emitDebugTypePointer(const DIDerivedType *DT, const size_t I32ZeroIdx,
                            MachineIRBuilder &MIRBuilder,
                            const SPIRVTargetMachine *TM, LiveRepository &LR) {
  assert(DT->getDWARFAddressSpace().has_value());

  size_t StorageClassIdx = LR.getOrCreateIdx(
      addressSpaceToStorageClass(DT->getDWARFAddressSpace().value(),
                                 *TM->getSubtargetImpl()),
      MIRBuilder, TM);
  size_t DebugInfoNoneIdx =
      LR.getOrCreateIdx<DebugInfoNone>({}, MIRBuilder, TM);
  return LR.getOrCreateIdx<DebugTypePointer>(
      {DebugInfoNoneIdx, StorageClassIdx, I32ZeroIdx}, MIRBuilder, TM);
}
} // namespace

namespace llvm {
struct SPIRVEmitNonSemanticDI : MachineFunctionPass {
  static char ID;
  SPIRVTargetMachine *TM = nullptr;
  SPIRVEmitNonSemanticDI(SPIRVTargetMachine *TM);
  SPIRVEmitNonSemanticDI();

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool IsGlobalDIEmitted = false;
  bool emitGlobalDI(MachineFunction &MF, const Module *M,
                    LiveRepository &LR) const;
  bool emitLineDI(MachineFunction &MF, LiveRepository &LR) const;
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

bool SPIRVEmitNonSemanticDI::emitGlobalDI(MachineFunction &MF, const Module *M,
                                          LiveRepository &LR) const {
  MachineBasicBlock &MBB = *MF.begin();

  // To ensure correct placement of an OpLabel instruction during
  // SPIRVAsmPrinter emission, all new instructions must be positioned after
  // OpFunction and before the first terminator.
  MachineIRBuilder MIRBuilder(MBB, MBB.getFirstTerminator());

  emitDebugCompilationUnits(M, MIRBuilder, TM, LR);

  for (auto &F : *M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
          const DILocalVariable *LocalVariable = DVR.getVariable();
          if (const auto *BasicType =
                  dyn_cast<DIBasicType>(LocalVariable->getType())) {
            // Currently, we are not extracting any DebugInfoFlags,
            // so we emit zero as the <id>Flags argument for DebugBasicType.
            const size_t I32ZeroIdx = LR.getOrCreateIdx(0, MIRBuilder, TM);
            emitDebugTypeBasic(BasicType, I32ZeroIdx, MIRBuilder, TM, LR);
            continue;
          }
          if (const auto *DerivedType =
                  dyn_cast<DIDerivedType>(LocalVariable->getType())) {
            if (DerivedType->getTag() == dwarf::DW_TAG_pointer_type) {
              const size_t I32ZeroIdx = LR.getOrCreateIdx(0, MIRBuilder, TM);
              // DIBasicType may be unreachable from DbgRecord and can only be
              // referenced by other Debug Information (DI) types. Note:
              // DerivedType->getBaseType returns null when the pointer
              // represents a void type.
              if (const auto *BasicType = dyn_cast_if_present<DIBasicType>(
                      DerivedType->getBaseType())) {
                const size_t BTIdx = emitDebugTypeBasic(BasicType, I32ZeroIdx,
                                                        MIRBuilder, TM, LR);
                emitDebugTypePointer(DerivedType, BTIdx, I32ZeroIdx, MIRBuilder,
                                     TM, LR);
              } else {
                emitDebugTypePointer(DerivedType, I32ZeroIdx, MIRBuilder, TM,
                                     LR);
              }
            }
          }
        }
      }
    }
  }
  return true;
}

bool SPIRVEmitNonSemanticDI::emitLineDI(MachineFunction &MF,
                                        LiveRepository &LR) const {
  bool IsModified = false;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getDebugLoc().get()) {
        MachineIRBuilder MIRBuilder(MBB, MI);
        DebugLoc DL = MI.getDebugLoc();
        assert(DL.getScope() && "DL.getScope() must exist and be DISubprogram");
        const auto *File = cast<DISubprogram>(DL.getScope())->getFile();
        const size_t ScopeIdx = emitDebugSource(File, MIRBuilder, TM, LR);
        const size_t LineIdx = LR.getOrCreateIdx(DL.getLine(), MIRBuilder, TM);
        const size_t ColIdx = LR.getOrCreateIdx(DL.getCol(), MIRBuilder, TM);
        LR.getOrCreateIdx<DebugLine>(
            {ScopeIdx, LineIdx, LineIdx, ColIdx, ColIdx}, MIRBuilder, TM);
        IsModified = true;
      }
    }
  }
  return IsModified;
}

bool SPIRVEmitNonSemanticDI::runOnMachineFunction(MachineFunction &MF) {
  if (MF.begin() == MF.end()) {
    return false;
  }
  static bool IsDIInModule = true;
  bool IsFunctionModified = false;
  if (IsDIInModule) {
    LiveRepository LR;
    // emitGlobalDI should be executed only once to prevent
    // the emission of duplicate entries.
    if (!IsGlobalDIEmitted) {
      IsGlobalDIEmitted = true;
      const MachineModuleInfo &MMI =
          getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
      const Module *M = MMI.getModule();
      if (!M || !M->getNamedMetadata("llvm.dbg.cu")) {
        IsDIInModule = false;
        return false;
      }
      IsFunctionModified = emitGlobalDI(MF, M, LR);
    }
    IsFunctionModified |= emitLineDI(MF, LR);
  }
  return IsFunctionModified;
}
