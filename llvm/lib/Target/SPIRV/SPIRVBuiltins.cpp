//===- SPIRVBuiltins.cpp - SPIR-V Built-in Functions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering builtin function calls and types using their
// demangled names and TableGen records.
//
//===----------------------------------------------------------------------===//

#include "SPIRVBuiltins.h"
#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include <regex>
#include <string>
#include <tuple>

#define DEBUG_TYPE "spirv-builtins"

namespace llvm {
namespace SPIRV {
#define GET_BuiltinGroup_DECL
#include "SPIRVGenTables.inc"

struct DemangledBuiltin {
  StringRef Name;
  InstructionSet::InstructionSet Set;
  BuiltinGroup Group;
  uint8_t MinNumArgs;
  uint8_t MaxNumArgs;
};

#define GET_DemangledBuiltins_DECL
#define GET_DemangledBuiltins_IMPL

struct IncomingCall {
  const std::string BuiltinName;
  const DemangledBuiltin *Builtin;

  const Register ReturnRegister;
  const SPIRVType *ReturnType;
  const SmallVectorImpl<Register> &Arguments;

  IncomingCall(const std::string BuiltinName, const DemangledBuiltin *Builtin,
               const Register ReturnRegister, const SPIRVType *ReturnType,
               const SmallVectorImpl<Register> &Arguments)
      : BuiltinName(std::move(BuiltinName)), Builtin(Builtin),
        ReturnRegister(ReturnRegister), ReturnType(ReturnType),
        Arguments(Arguments) {}

  bool isSpirvOp() const { return BuiltinName.rfind("__spirv_", 0) == 0; }
};

struct NativeBuiltin {
  StringRef Name;
  InstructionSet::InstructionSet Set;
  uint32_t Opcode;
};

#define GET_NativeBuiltins_DECL
#define GET_NativeBuiltins_IMPL

struct GroupBuiltin {
  StringRef Name;
  uint32_t Opcode;
  uint32_t GroupOperation;
  bool IsElect;
  bool IsAllOrAny;
  bool IsAllEqual;
  bool IsBallot;
  bool IsInverseBallot;
  bool IsBallotBitExtract;
  bool IsBallotFindBit;
  bool IsLogical;
  bool NoGroupOperation;
  bool HasBoolArg;
};

#define GET_GroupBuiltins_DECL
#define GET_GroupBuiltins_IMPL

struct IntelSubgroupsBuiltin {
  StringRef Name;
  uint32_t Opcode;
  bool IsBlock;
  bool IsWrite;
  bool IsMedia;
};

#define GET_IntelSubgroupsBuiltins_DECL
#define GET_IntelSubgroupsBuiltins_IMPL

struct AtomicFloatingBuiltin {
  StringRef Name;
  uint32_t Opcode;
};

#define GET_AtomicFloatingBuiltins_DECL
#define GET_AtomicFloatingBuiltins_IMPL
struct GroupUniformBuiltin {
  StringRef Name;
  uint32_t Opcode;
  bool IsLogical;
};

#define GET_GroupUniformBuiltins_DECL
#define GET_GroupUniformBuiltins_IMPL

struct GetBuiltin {
  StringRef Name;
  InstructionSet::InstructionSet Set;
  BuiltIn::BuiltIn Value;
};

using namespace BuiltIn;
#define GET_GetBuiltins_DECL
#define GET_GetBuiltins_IMPL

struct ImageQueryBuiltin {
  StringRef Name;
  InstructionSet::InstructionSet Set;
  uint32_t Component;
};

#define GET_ImageQueryBuiltins_DECL
#define GET_ImageQueryBuiltins_IMPL

struct IntegerDotProductBuiltin {
  StringRef Name;
  uint32_t Opcode;
  bool IsSwapReq;
};

#define GET_IntegerDotProductBuiltins_DECL
#define GET_IntegerDotProductBuiltins_IMPL

struct ConvertBuiltin {
  StringRef Name;
  InstructionSet::InstructionSet Set;
  bool IsDestinationSigned;
  bool IsSaturated;
  bool IsRounded;
  bool IsBfloat16;
  bool IsTF32;
  FPRoundingMode::FPRoundingMode RoundingMode;
};

struct VectorLoadStoreBuiltin {
  StringRef Name;
  InstructionSet::InstructionSet Set;
  uint32_t Number;
  uint32_t ElementCount;
  bool IsRounded;
  FPRoundingMode::FPRoundingMode RoundingMode;
};

using namespace FPRoundingMode;
#define GET_ConvertBuiltins_DECL
#define GET_ConvertBuiltins_IMPL

using namespace InstructionSet;
#define GET_VectorLoadStoreBuiltins_DECL
#define GET_VectorLoadStoreBuiltins_IMPL

#define GET_CLMemoryScope_DECL
#define GET_CLSamplerAddressingMode_DECL
#define GET_CLMemoryFenceFlags_DECL
#define GET_ExtendedBuiltins_DECL
#include "SPIRVGenTables.inc"
} // namespace SPIRV

//===----------------------------------------------------------------------===//
// Misc functions for looking up builtins and veryfying requirements using
// TableGen records
//===----------------------------------------------------------------------===//

namespace SPIRV {
/// Parses the name part of the demangled builtin call.
std::string lookupBuiltinNameHelper(StringRef DemangledCall,
                                    FPDecorationId *DecorationId) {
  StringRef PassPrefix = "(anonymous namespace)::";
  std::string BuiltinName;
  // Itanium Demangler result may have "(anonymous namespace)::" prefix
  if (DemangledCall.starts_with(PassPrefix))
    BuiltinName = DemangledCall.substr(PassPrefix.size());
  else
    BuiltinName = DemangledCall;
  // Extract the builtin function name and types of arguments from the call
  // skeleton.
  BuiltinName = BuiltinName.substr(0, BuiltinName.find('('));

  // Account for possible "__spirv_ocl_" prefix in SPIR-V friendly LLVM IR
  if (BuiltinName.rfind("__spirv_ocl_", 0) == 0)
    BuiltinName = BuiltinName.substr(12);

  // Check if the extracted name contains type information between angle
  // brackets. If so, the builtin is an instantiated template - needs to have
  // the information after angle brackets and return type removed.
  std::size_t Pos1 = BuiltinName.rfind('<');
  if (Pos1 != std::string::npos && BuiltinName.back() == '>') {
    std::size_t Pos2 = BuiltinName.rfind(' ', Pos1);
    if (Pos2 == std::string::npos)
      Pos2 = 0;
    else
      ++Pos2;
    BuiltinName = BuiltinName.substr(Pos2, Pos1 - Pos2);
    BuiltinName = BuiltinName.substr(BuiltinName.find_last_of(' ') + 1);
  }

  // Check if the extracted name begins with:
  // - "__spirv_ImageSampleExplicitLod"
  // - "__spirv_ImageRead"
  // - "__spirv_ImageWrite"
  // - "__spirv_ImageQuerySizeLod"
  // - "__spirv_UDotKHR"
  // - "__spirv_SDotKHR"
  // - "__spirv_SUDotKHR"
  // - "__spirv_SDotAccSatKHR"
  // - "__spirv_UDotAccSatKHR"
  // - "__spirv_SUDotAccSatKHR"
  // - "__spirv_ReadClockKHR"
  // - "__spirv_SubgroupBlockReadINTEL"
  // - "__spirv_SubgroupImageBlockReadINTEL"
  // - "__spirv_SubgroupImageMediaBlockReadINTEL"
  // - "__spirv_SubgroupImageMediaBlockWriteINTEL"
  // - "__spirv_Convert"
  // - "__spirv_Round"
  // - "__spirv_UConvert"
  // - "__spirv_SConvert"
  // - "__spirv_FConvert"
  // - "__spirv_SatConvert"
  // and maybe contains return type information at the end "_R<type>".
  // If so, extract the plain builtin name without the type information.
  static const std::regex SpvWithR(
      "(__spirv_(ImageSampleExplicitLod|ImageRead|ImageWrite|ImageQuerySizeLod|"
      "UDotKHR|"
      "SDotKHR|SUDotKHR|SDotAccSatKHR|UDotAccSatKHR|SUDotAccSatKHR|"
      "ReadClockKHR|SubgroupBlockReadINTEL|SubgroupImageBlockReadINTEL|"
      "SubgroupImageMediaBlockReadINTEL|SubgroupImageMediaBlockWriteINTEL|"
      "Convert|Round|"
      "UConvert|SConvert|FConvert|SatConvert)[^_]*)(_R[^_]*_?(\\w+)?.*)?");
  std::smatch Match;
  if (std::regex_match(BuiltinName, Match, SpvWithR) && Match.size() > 1) {
    std::ssub_match SubMatch;
    if (DecorationId && Match.size() > 3) {
      SubMatch = Match[4];
      *DecorationId = demangledPostfixToDecorationId(SubMatch.str());
    }
    SubMatch = Match[1];
    BuiltinName = SubMatch.str();
  }

  return BuiltinName;
}
} // namespace SPIRV

/// Looks up the demangled builtin call in the SPIRVBuiltins.td records using
/// the provided \p DemangledCall and specified \p Set.
///
/// The lookup follows the following algorithm, returning the first successful
/// match:
/// 1. Search with the plain demangled name (expecting a 1:1 match).
/// 2. Search with the prefix before or suffix after the demangled name
/// signyfying the type of the first argument.
///
/// \returns Wrapper around the demangled call and found builtin definition.
static std::unique_ptr<const SPIRV::IncomingCall>
lookupBuiltin(StringRef DemangledCall,
              SPIRV::InstructionSet::InstructionSet Set,
              Register ReturnRegister, const SPIRVType *ReturnType,
              const SmallVectorImpl<Register> &Arguments) {
  std::string BuiltinName = SPIRV::lookupBuiltinNameHelper(DemangledCall);

  SmallVector<StringRef, 10> BuiltinArgumentTypes;
  StringRef BuiltinArgs =
      DemangledCall.slice(DemangledCall.find('(') + 1, DemangledCall.find(')'));
  BuiltinArgs.split(BuiltinArgumentTypes, ',', -1, false);

  // Look up the builtin in the defined set. Start with the plain demangled
  // name, expecting a 1:1 match in the defined builtin set.
  const SPIRV::DemangledBuiltin *Builtin;
  if ((Builtin = SPIRV::lookupBuiltin(BuiltinName, Set)))
    return std::make_unique<SPIRV::IncomingCall>(
        BuiltinName, Builtin, ReturnRegister, ReturnType, Arguments);

  // If the initial look up was unsuccessful and the demangled call takes at
  // least 1 argument, add a prefix or suffix signifying the type of the first
  // argument and repeat the search.
  if (BuiltinArgumentTypes.size() >= 1) {
    char FirstArgumentType = BuiltinArgumentTypes[0][0];
    // Prefix to be added to the builtin's name for lookup.
    // For example, OpenCL "abs" taking an unsigned value has a prefix "u_".
    std::string Prefix;

    switch (FirstArgumentType) {
    // Unsigned:
    case 'u':
      if (Set == SPIRV::InstructionSet::OpenCL_std)
        Prefix = "u_";
      else if (Set == SPIRV::InstructionSet::GLSL_std_450)
        Prefix = "u";
      break;
    // Signed:
    case 'c':
    case 's':
    case 'i':
    case 'l':
      if (Set == SPIRV::InstructionSet::OpenCL_std)
        Prefix = "s_";
      else if (Set == SPIRV::InstructionSet::GLSL_std_450)
        Prefix = "s";
      break;
    // Floating-point:
    case 'f':
    case 'd':
    case 'h':
      if (Set == SPIRV::InstructionSet::OpenCL_std ||
          Set == SPIRV::InstructionSet::GLSL_std_450)
        Prefix = "f";
      break;
    }

    // If argument-type name prefix was added, look up the builtin again.
    if (!Prefix.empty() &&
        (Builtin = SPIRV::lookupBuiltin(Prefix + BuiltinName, Set)))
      return std::make_unique<SPIRV::IncomingCall>(
          BuiltinName, Builtin, ReturnRegister, ReturnType, Arguments);

    // If lookup with a prefix failed, find a suffix to be added to the
    // builtin's name for lookup. For example, OpenCL "group_reduce_max" taking
    // an unsigned value has a suffix "u".
    std::string Suffix;

    switch (FirstArgumentType) {
    // Unsigned:
    case 'u':
      Suffix = "u";
      break;
    // Signed:
    case 'c':
    case 's':
    case 'i':
    case 'l':
      Suffix = "s";
      break;
    // Floating-point:
    case 'f':
    case 'd':
    case 'h':
      Suffix = "f";
      break;
    }

    // If argument-type name suffix was added, look up the builtin again.
    if (!Suffix.empty() &&
        (Builtin = SPIRV::lookupBuiltin(BuiltinName + Suffix, Set)))
      return std::make_unique<SPIRV::IncomingCall>(
          BuiltinName, Builtin, ReturnRegister, ReturnType, Arguments);
  }

  // No builtin with such name was found in the set.
  return nullptr;
}

static MachineInstr *getBlockStructInstr(Register ParamReg,
                                         MachineRegisterInfo *MRI) {
  // We expect the following sequence of instructions:
  //   %0:_(pN) = G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.spv.alloca)
  //   or       = G_GLOBAL_VALUE @block_literal_global
  //   %1:_(pN) = G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.spv.bitcast), %0
  //   %2:_(p4) = G_ADDRSPACE_CAST %1:_(pN)
  MachineInstr *MI = MRI->getUniqueVRegDef(ParamReg);
  assert(MI->getOpcode() == TargetOpcode::G_ADDRSPACE_CAST &&
         MI->getOperand(1).isReg());
  Register BitcastReg = MI->getOperand(1).getReg();
  MachineInstr *BitcastMI = MRI->getUniqueVRegDef(BitcastReg);
  assert(isSpvIntrinsic(*BitcastMI, Intrinsic::spv_bitcast) &&
         BitcastMI->getOperand(2).isReg());
  Register ValueReg = BitcastMI->getOperand(2).getReg();
  MachineInstr *ValueMI = MRI->getUniqueVRegDef(ValueReg);
  return ValueMI;
}

// Return an integer constant corresponding to the given register and
// defined in spv_track_constant.
// TODO: maybe unify with prelegalizer pass.
static unsigned getConstFromIntrinsic(Register Reg, MachineRegisterInfo *MRI) {
  MachineInstr *DefMI = MRI->getUniqueVRegDef(Reg);
  assert(DefMI->getOpcode() == TargetOpcode::G_CONSTANT &&
         DefMI->getOperand(1).isCImm());
  return DefMI->getOperand(1).getCImm()->getValue().getZExtValue();
}

// Return type of the instruction result from spv_assign_type intrinsic.
// TODO: maybe unify with prelegalizer pass.
static const Type *getMachineInstrType(MachineInstr *MI) {
  MachineInstr *NextMI = MI->getNextNode();
  if (!NextMI)
    return nullptr;
  if (isSpvIntrinsic(*NextMI, Intrinsic::spv_assign_name))
    if ((NextMI = NextMI->getNextNode()) == nullptr)
      return nullptr;
  Register ValueReg = MI->getOperand(0).getReg();
  if ((!isSpvIntrinsic(*NextMI, Intrinsic::spv_assign_type) &&
       !isSpvIntrinsic(*NextMI, Intrinsic::spv_assign_ptr_type)) ||
      NextMI->getOperand(1).getReg() != ValueReg)
    return nullptr;
  Type *Ty = getMDOperandAsType(NextMI->getOperand(2).getMetadata(), 0);
  assert(Ty && "Type is expected");
  return Ty;
}

static const Type *getBlockStructType(Register ParamReg,
                                      MachineRegisterInfo *MRI) {
  // In principle, this information should be passed to us from Clang via
  // an elementtype attribute. However, said attribute requires that
  // the function call be an intrinsic, which is not. Instead, we rely on being
  // able to trace this to the declaration of a variable: OpenCL C specification
  // section 6.12.5 should guarantee that we can do this.
  MachineInstr *MI = getBlockStructInstr(ParamReg, MRI);
  if (MI->getOpcode() == TargetOpcode::G_GLOBAL_VALUE)
    return MI->getOperand(1).getGlobal()->getType();
  assert(isSpvIntrinsic(*MI, Intrinsic::spv_alloca) &&
         "Blocks in OpenCL C must be traceable to allocation site");
  return getMachineInstrType(MI);
}

//===----------------------------------------------------------------------===//
// Helper functions for building misc instructions
//===----------------------------------------------------------------------===//

/// Helper function building either a resulting scalar or vector bool register
/// depending on the expected \p ResultType.
///
/// \returns Tuple of the resulting register and its type.
static std::tuple<Register, SPIRVType *>
buildBoolRegister(MachineIRBuilder &MIRBuilder, const SPIRVType *ResultType,
                  SPIRVGlobalRegistry *GR) {
  LLT Type;
  SPIRVType *BoolType = GR->getOrCreateSPIRVBoolType(MIRBuilder, true);

  if (ResultType->getOpcode() == SPIRV::OpTypeVector) {
    unsigned VectorElements = ResultType->getOperand(2).getImm();
    BoolType = GR->getOrCreateSPIRVVectorType(BoolType, VectorElements,
                                              MIRBuilder, true);
    const FixedVectorType *LLVMVectorType =
        cast<FixedVectorType>(GR->getTypeForSPIRVType(BoolType));
    Type = LLT::vector(LLVMVectorType->getElementCount(), 1);
  } else {
    Type = LLT::scalar(1);
  }

  Register ResultRegister =
      MIRBuilder.getMRI()->createGenericVirtualRegister(Type);
  MIRBuilder.getMRI()->setRegClass(ResultRegister, GR->getRegClass(ResultType));
  GR->assignSPIRVTypeToVReg(BoolType, ResultRegister, MIRBuilder.getMF());
  return std::make_tuple(ResultRegister, BoolType);
}

/// Helper function for building either a vector or scalar select instruction
/// depending on the expected \p ResultType.
static bool buildSelectInst(MachineIRBuilder &MIRBuilder,
                            Register ReturnRegister, Register SourceRegister,
                            const SPIRVType *ReturnType,
                            SPIRVGlobalRegistry *GR) {
  Register TrueConst, FalseConst;

  if (ReturnType->getOpcode() == SPIRV::OpTypeVector) {
    unsigned Bits = GR->getScalarOrVectorBitWidth(ReturnType);
    uint64_t AllOnes = APInt::getAllOnes(Bits).getZExtValue();
    TrueConst =
        GR->getOrCreateConsIntVector(AllOnes, MIRBuilder, ReturnType, true);
    FalseConst = GR->getOrCreateConsIntVector(0, MIRBuilder, ReturnType, true);
  } else {
    TrueConst = GR->buildConstantInt(1, MIRBuilder, ReturnType, true);
    FalseConst = GR->buildConstantInt(0, MIRBuilder, ReturnType, true);
  }

  return MIRBuilder.buildSelect(ReturnRegister, SourceRegister, TrueConst,
                                FalseConst);
}

/// Helper function for building a load instruction loading into the
/// \p DestinationReg.
static Register buildLoadInst(SPIRVType *BaseType, Register PtrRegister,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR, LLT LowLevelType,
                              Register DestinationReg = Register(0)) {
  if (!DestinationReg.isValid())
    DestinationReg = createVirtualRegister(BaseType, GR, MIRBuilder);
  // TODO: consider using correct address space and alignment (p0 is canonical
  // type for selection though).
  MachinePointerInfo PtrInfo = MachinePointerInfo();
  MIRBuilder.buildLoad(DestinationReg, PtrRegister, PtrInfo, Align());
  return DestinationReg;
}

/// Helper function for building a load instruction for loading a builtin global
/// variable of \p BuiltinValue value.
static Register buildBuiltinVariableLoad(
    MachineIRBuilder &MIRBuilder, SPIRVType *VariableType,
    SPIRVGlobalRegistry *GR, SPIRV::BuiltIn::BuiltIn BuiltinValue, LLT LLType,
    Register Reg = Register(0), bool isConst = true, bool hasLinkageTy = true) {
  Register NewRegister =
      MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::pIDRegClass);
  MIRBuilder.getMRI()->setType(
      NewRegister,
      LLT::pointer(storageClassToAddressSpace(SPIRV::StorageClass::Function),
                   GR->getPointerSize()));
  SPIRVType *PtrType = GR->getOrCreateSPIRVPointerType(
      VariableType, MIRBuilder, SPIRV::StorageClass::Input);
  GR->assignSPIRVTypeToVReg(PtrType, NewRegister, MIRBuilder.getMF());

  // Set up the global OpVariable with the necessary builtin decorations.
  Register Variable = GR->buildGlobalVariable(
      NewRegister, PtrType, getLinkStringForBuiltIn(BuiltinValue), nullptr,
      SPIRV::StorageClass::Input, nullptr, /* isConst= */ isConst,
      /* HasLinkageTy */ hasLinkageTy, SPIRV::LinkageType::Import, MIRBuilder,
      false);

  // Load the value from the global variable.
  Register LoadedRegister =
      buildLoadInst(VariableType, Variable, MIRBuilder, GR, LLType, Reg);
  MIRBuilder.getMRI()->setType(LoadedRegister, LLType);
  return LoadedRegister;
}

/// Helper external function for inserting ASSIGN_TYPE instuction between \p Reg
/// and its definition, set the new register as a destination of the definition,
/// assign SPIRVType to both registers. If SpirvTy is provided, use it as
/// SPIRVType in ASSIGN_TYPE, otherwise create it from \p Ty. Defined in
/// SPIRVPreLegalizer.cpp.
extern void insertAssignInstr(Register Reg, Type *Ty, SPIRVType *SpirvTy,
                              SPIRVGlobalRegistry *GR, MachineIRBuilder &MIB,
                              MachineRegisterInfo &MRI);

// TODO: Move to TableGen.
static SPIRV::MemorySemantics::MemorySemantics
getSPIRVMemSemantics(std::memory_order MemOrder) {
  switch (MemOrder) {
  case std::memory_order_relaxed:
    return SPIRV::MemorySemantics::None;
  case std::memory_order_acquire:
    return SPIRV::MemorySemantics::Acquire;
  case std::memory_order_release:
    return SPIRV::MemorySemantics::Release;
  case std::memory_order_acq_rel:
    return SPIRV::MemorySemantics::AcquireRelease;
  case std::memory_order_seq_cst:
    return SPIRV::MemorySemantics::SequentiallyConsistent;
  default:
    report_fatal_error("Unknown CL memory scope");
  }
}

static SPIRV::Scope::Scope getSPIRVScope(SPIRV::CLMemoryScope ClScope) {
  switch (ClScope) {
  case SPIRV::CLMemoryScope::memory_scope_work_item:
    return SPIRV::Scope::Invocation;
  case SPIRV::CLMemoryScope::memory_scope_work_group:
    return SPIRV::Scope::Workgroup;
  case SPIRV::CLMemoryScope::memory_scope_device:
    return SPIRV::Scope::Device;
  case SPIRV::CLMemoryScope::memory_scope_all_svm_devices:
    return SPIRV::Scope::CrossDevice;
  case SPIRV::CLMemoryScope::memory_scope_sub_group:
    return SPIRV::Scope::Subgroup;
  }
  report_fatal_error("Unknown CL memory scope");
}

static Register buildConstantIntReg32(uint64_t Val,
                                      MachineIRBuilder &MIRBuilder,
                                      SPIRVGlobalRegistry *GR) {
  return GR->buildConstantInt(
      Val, MIRBuilder, GR->getOrCreateSPIRVIntegerType(32, MIRBuilder), true);
}

static Register buildScopeReg(Register CLScopeRegister,
                              SPIRV::Scope::Scope Scope,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR,
                              MachineRegisterInfo *MRI) {
  if (CLScopeRegister.isValid()) {
    auto CLScope =
        static_cast<SPIRV::CLMemoryScope>(getIConstVal(CLScopeRegister, MRI));
    Scope = getSPIRVScope(CLScope);

    if (CLScope == static_cast<unsigned>(Scope)) {
      MRI->setRegClass(CLScopeRegister, &SPIRV::iIDRegClass);
      return CLScopeRegister;
    }
  }
  return buildConstantIntReg32(Scope, MIRBuilder, GR);
}

static void setRegClassIfNull(Register Reg, MachineRegisterInfo *MRI,
                              SPIRVGlobalRegistry *GR) {
  if (MRI->getRegClassOrNull(Reg))
    return;
  SPIRVType *SpvType = GR->getSPIRVTypeForVReg(Reg);
  MRI->setRegClass(Reg,
                   SpvType ? GR->getRegClass(SpvType) : &SPIRV::iIDRegClass);
}

static Register buildMemSemanticsReg(Register SemanticsRegister,
                                     Register PtrRegister, unsigned &Semantics,
                                     MachineIRBuilder &MIRBuilder,
                                     SPIRVGlobalRegistry *GR) {
  if (SemanticsRegister.isValid()) {
    MachineRegisterInfo *MRI = MIRBuilder.getMRI();
    std::memory_order Order =
        static_cast<std::memory_order>(getIConstVal(SemanticsRegister, MRI));
    Semantics =
        getSPIRVMemSemantics(Order) |
        getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));
    if (static_cast<unsigned>(Order) == Semantics) {
      MRI->setRegClass(SemanticsRegister, &SPIRV::iIDRegClass);
      return SemanticsRegister;
    }
  }
  return buildConstantIntReg32(Semantics, MIRBuilder, GR);
}

static bool buildOpFromWrapper(MachineIRBuilder &MIRBuilder, unsigned Opcode,
                               const SPIRV::IncomingCall *Call,
                               Register TypeReg,
                               ArrayRef<uint32_t> ImmArgs = {}) {
  auto MIB = MIRBuilder.buildInstr(Opcode);
  if (TypeReg.isValid())
    MIB.addDef(Call->ReturnRegister).addUse(TypeReg);
  unsigned Sz = Call->Arguments.size() - ImmArgs.size();
  for (unsigned i = 0; i < Sz; ++i)
    MIB.addUse(Call->Arguments[i]);
  for (uint32_t ImmArg : ImmArgs)
    MIB.addImm(ImmArg);
  return true;
}

/// Helper function for translating atomic init to OpStore.
static bool buildAtomicInitInst(const SPIRV::IncomingCall *Call,
                                MachineIRBuilder &MIRBuilder) {
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, SPIRV::OpStore, Call, Register(0));

  assert(Call->Arguments.size() == 2 &&
         "Need 2 arguments for atomic init translation");
  MIRBuilder.buildInstr(SPIRV::OpStore)
      .addUse(Call->Arguments[0])
      .addUse(Call->Arguments[1]);
  return true;
}

/// Helper function for building an atomic load instruction.
static bool buildAtomicLoadInst(const SPIRV::IncomingCall *Call,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  Register TypeReg = GR->getSPIRVTypeID(Call->ReturnType);
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, SPIRV::OpAtomicLoad, Call, TypeReg);

  Register PtrRegister = Call->Arguments[0];
  // TODO: if true insert call to __translate_ocl_memory_sccope before
  // OpAtomicLoad and the function implementation. We can use Translator's
  // output for transcoding/atomic_explicit_arguments.cl as an example.
  Register ScopeRegister =
      Call->Arguments.size() > 1
          ? Call->Arguments[1]
          : buildConstantIntReg32(SPIRV::Scope::Device, MIRBuilder, GR);
  Register MemSemanticsReg;
  if (Call->Arguments.size() > 2) {
    // TODO: Insert call to __translate_ocl_memory_order before OpAtomicLoad.
    MemSemanticsReg = Call->Arguments[2];
  } else {
    int Semantics =
        SPIRV::MemorySemantics::SequentiallyConsistent |
        getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));
    MemSemanticsReg = buildConstantIntReg32(Semantics, MIRBuilder, GR);
  }

  MIRBuilder.buildInstr(SPIRV::OpAtomicLoad)
      .addDef(Call->ReturnRegister)
      .addUse(TypeReg)
      .addUse(PtrRegister)
      .addUse(ScopeRegister)
      .addUse(MemSemanticsReg);
  return true;
}

/// Helper function for building an atomic store instruction.
static bool buildAtomicStoreInst(const SPIRV::IncomingCall *Call,
                                 MachineIRBuilder &MIRBuilder,
                                 SPIRVGlobalRegistry *GR) {
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, SPIRV::OpAtomicStore, Call,
                              Register(0));

  Register ScopeRegister =
      buildConstantIntReg32(SPIRV::Scope::Device, MIRBuilder, GR);
  Register PtrRegister = Call->Arguments[0];
  int Semantics =
      SPIRV::MemorySemantics::SequentiallyConsistent |
      getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));
  Register MemSemanticsReg = buildConstantIntReg32(Semantics, MIRBuilder, GR);
  MIRBuilder.buildInstr(SPIRV::OpAtomicStore)
      .addUse(PtrRegister)
      .addUse(ScopeRegister)
      .addUse(MemSemanticsReg)
      .addUse(Call->Arguments[1]);
  return true;
}

/// Helper function for building an atomic compare-exchange instruction.
static bool buildAtomicCompareExchangeInst(
    const SPIRV::IncomingCall *Call, const SPIRV::DemangledBuiltin *Builtin,
    unsigned Opcode, MachineIRBuilder &MIRBuilder, SPIRVGlobalRegistry *GR) {
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));

  bool IsCmpxchg = Call->Builtin->Name.contains("cmpxchg");
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();

  Register ObjectPtr = Call->Arguments[0];   // Pointer (volatile A *object.)
  Register ExpectedArg = Call->Arguments[1]; // Comparator (C* expected).
  Register Desired = Call->Arguments[2];     // Value (C Desired).
  SPIRVType *SpvDesiredTy = GR->getSPIRVTypeForVReg(Desired);
  LLT DesiredLLT = MRI->getType(Desired);

  assert(GR->getSPIRVTypeForVReg(ObjectPtr)->getOpcode() ==
         SPIRV::OpTypePointer);
  unsigned ExpectedType = GR->getSPIRVTypeForVReg(ExpectedArg)->getOpcode();
  (void)ExpectedType;
  assert(IsCmpxchg ? ExpectedType == SPIRV::OpTypeInt
                   : ExpectedType == SPIRV::OpTypePointer);
  assert(GR->isScalarOfType(Desired, SPIRV::OpTypeInt));

  SPIRVType *SpvObjectPtrTy = GR->getSPIRVTypeForVReg(ObjectPtr);
  assert(SpvObjectPtrTy->getOperand(2).isReg() && "SPIRV type is expected");
  auto StorageClass = static_cast<SPIRV::StorageClass::StorageClass>(
      SpvObjectPtrTy->getOperand(1).getImm());
  auto MemSemStorage = getMemSemanticsForStorageClass(StorageClass);

  Register MemSemEqualReg;
  Register MemSemUnequalReg;
  uint64_t MemSemEqual =
      IsCmpxchg
          ? SPIRV::MemorySemantics::None
          : SPIRV::MemorySemantics::SequentiallyConsistent | MemSemStorage;
  uint64_t MemSemUnequal =
      IsCmpxchg
          ? SPIRV::MemorySemantics::None
          : SPIRV::MemorySemantics::SequentiallyConsistent | MemSemStorage;
  if (Call->Arguments.size() >= 4) {
    assert(Call->Arguments.size() >= 5 &&
           "Need 5+ args for explicit atomic cmpxchg");
    auto MemOrdEq =
        static_cast<std::memory_order>(getIConstVal(Call->Arguments[3], MRI));
    auto MemOrdNeq =
        static_cast<std::memory_order>(getIConstVal(Call->Arguments[4], MRI));
    MemSemEqual = getSPIRVMemSemantics(MemOrdEq) | MemSemStorage;
    MemSemUnequal = getSPIRVMemSemantics(MemOrdNeq) | MemSemStorage;
    if (static_cast<unsigned>(MemOrdEq) == MemSemEqual)
      MemSemEqualReg = Call->Arguments[3];
    if (static_cast<unsigned>(MemOrdNeq) == MemSemEqual)
      MemSemUnequalReg = Call->Arguments[4];
  }
  if (!MemSemEqualReg.isValid())
    MemSemEqualReg = buildConstantIntReg32(MemSemEqual, MIRBuilder, GR);
  if (!MemSemUnequalReg.isValid())
    MemSemUnequalReg = buildConstantIntReg32(MemSemUnequal, MIRBuilder, GR);

  Register ScopeReg;
  auto Scope = IsCmpxchg ? SPIRV::Scope::Workgroup : SPIRV::Scope::Device;
  if (Call->Arguments.size() >= 6) {
    assert(Call->Arguments.size() == 6 &&
           "Extra args for explicit atomic cmpxchg");
    auto ClScope = static_cast<SPIRV::CLMemoryScope>(
        getIConstVal(Call->Arguments[5], MRI));
    Scope = getSPIRVScope(ClScope);
    if (ClScope == static_cast<unsigned>(Scope))
      ScopeReg = Call->Arguments[5];
  }
  if (!ScopeReg.isValid())
    ScopeReg = buildConstantIntReg32(Scope, MIRBuilder, GR);

  Register Expected = IsCmpxchg
                          ? ExpectedArg
                          : buildLoadInst(SpvDesiredTy, ExpectedArg, MIRBuilder,
                                          GR, LLT::scalar(64));
  MRI->setType(Expected, DesiredLLT);
  Register Tmp = !IsCmpxchg ? MRI->createGenericVirtualRegister(DesiredLLT)
                            : Call->ReturnRegister;
  if (!MRI->getRegClassOrNull(Tmp))
    MRI->setRegClass(Tmp, GR->getRegClass(SpvDesiredTy));
  GR->assignSPIRVTypeToVReg(SpvDesiredTy, Tmp, MIRBuilder.getMF());

  MIRBuilder.buildInstr(Opcode)
      .addDef(Tmp)
      .addUse(GR->getSPIRVTypeID(SpvDesiredTy))
      .addUse(ObjectPtr)
      .addUse(ScopeReg)
      .addUse(MemSemEqualReg)
      .addUse(MemSemUnequalReg)
      .addUse(Desired)
      .addUse(Expected);
  if (!IsCmpxchg) {
    MIRBuilder.buildInstr(SPIRV::OpStore).addUse(ExpectedArg).addUse(Tmp);
    MIRBuilder.buildICmp(CmpInst::ICMP_EQ, Call->ReturnRegister, Tmp, Expected);
  }
  return true;
}

/// Helper function for building atomic instructions.
static bool buildAtomicRMWInst(const SPIRV::IncomingCall *Call, unsigned Opcode,
                               MachineIRBuilder &MIRBuilder,
                               SPIRVGlobalRegistry *GR) {
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));

  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  Register ScopeRegister =
      Call->Arguments.size() >= 4 ? Call->Arguments[3] : Register();

  assert(Call->Arguments.size() <= 4 &&
         "Too many args for explicit atomic RMW");
  ScopeRegister = buildScopeReg(ScopeRegister, SPIRV::Scope::Workgroup,
                                MIRBuilder, GR, MRI);

  Register PtrRegister = Call->Arguments[0];
  unsigned Semantics = SPIRV::MemorySemantics::None;
  Register MemSemanticsReg =
      Call->Arguments.size() >= 3 ? Call->Arguments[2] : Register();
  MemSemanticsReg = buildMemSemanticsReg(MemSemanticsReg, PtrRegister,
                                         Semantics, MIRBuilder, GR);
  Register ValueReg = Call->Arguments[1];
  Register ValueTypeReg = GR->getSPIRVTypeID(Call->ReturnType);
  // support cl_ext_float_atomics
  if (Call->ReturnType->getOpcode() == SPIRV::OpTypeFloat) {
    if (Opcode == SPIRV::OpAtomicIAdd) {
      Opcode = SPIRV::OpAtomicFAddEXT;
    } else if (Opcode == SPIRV::OpAtomicISub) {
      // Translate OpAtomicISub applied to a floating type argument to
      // OpAtomicFAddEXT with the negative value operand
      Opcode = SPIRV::OpAtomicFAddEXT;
      Register NegValueReg =
          MRI->createGenericVirtualRegister(MRI->getType(ValueReg));
      MRI->setRegClass(NegValueReg, GR->getRegClass(Call->ReturnType));
      GR->assignSPIRVTypeToVReg(Call->ReturnType, NegValueReg,
                                MIRBuilder.getMF());
      MIRBuilder.buildInstr(TargetOpcode::G_FNEG)
          .addDef(NegValueReg)
          .addUse(ValueReg);
      insertAssignInstr(NegValueReg, nullptr, Call->ReturnType, GR, MIRBuilder,
                        MIRBuilder.getMF().getRegInfo());
      ValueReg = NegValueReg;
    }
  }
  MIRBuilder.buildInstr(Opcode)
      .addDef(Call->ReturnRegister)
      .addUse(ValueTypeReg)
      .addUse(PtrRegister)
      .addUse(ScopeRegister)
      .addUse(MemSemanticsReg)
      .addUse(ValueReg);
  return true;
}

/// Helper function for building an atomic floating-type instruction.
static bool buildAtomicFloatingRMWInst(const SPIRV::IncomingCall *Call,
                                       unsigned Opcode,
                                       MachineIRBuilder &MIRBuilder,
                                       SPIRVGlobalRegistry *GR) {
  assert(Call->Arguments.size() == 4 &&
         "Wrong number of atomic floating-type builtin");
  Register PtrReg = Call->Arguments[0];
  Register ScopeReg = Call->Arguments[1];
  Register MemSemanticsReg = Call->Arguments[2];
  Register ValueReg = Call->Arguments[3];
  MIRBuilder.buildInstr(Opcode)
      .addDef(Call->ReturnRegister)
      .addUse(GR->getSPIRVTypeID(Call->ReturnType))
      .addUse(PtrReg)
      .addUse(ScopeReg)
      .addUse(MemSemanticsReg)
      .addUse(ValueReg);
  return true;
}

/// Helper function for building atomic flag instructions (e.g.
/// OpAtomicFlagTestAndSet).
static bool buildAtomicFlagInst(const SPIRV::IncomingCall *Call,
                                unsigned Opcode, MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  bool IsSet = Opcode == SPIRV::OpAtomicFlagTestAndSet;
  Register TypeReg = GR->getSPIRVTypeID(Call->ReturnType);
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              IsSet ? TypeReg : Register(0));

  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  Register PtrRegister = Call->Arguments[0];
  unsigned Semantics = SPIRV::MemorySemantics::SequentiallyConsistent;
  Register MemSemanticsReg =
      Call->Arguments.size() >= 2 ? Call->Arguments[1] : Register();
  MemSemanticsReg = buildMemSemanticsReg(MemSemanticsReg, PtrRegister,
                                         Semantics, MIRBuilder, GR);

  assert((Opcode != SPIRV::OpAtomicFlagClear ||
          (Semantics != SPIRV::MemorySemantics::Acquire &&
           Semantics != SPIRV::MemorySemantics::AcquireRelease)) &&
         "Invalid memory order argument!");

  Register ScopeRegister =
      Call->Arguments.size() >= 3 ? Call->Arguments[2] : Register();
  ScopeRegister =
      buildScopeReg(ScopeRegister, SPIRV::Scope::Device, MIRBuilder, GR, MRI);

  auto MIB = MIRBuilder.buildInstr(Opcode);
  if (IsSet)
    MIB.addDef(Call->ReturnRegister).addUse(TypeReg);

  MIB.addUse(PtrRegister).addUse(ScopeRegister).addUse(MemSemanticsReg);
  return true;
}

/// Helper function for building barriers, i.e., memory/control ordering
/// operations.
static bool buildBarrierInst(const SPIRV::IncomingCall *Call, unsigned Opcode,
                             MachineIRBuilder &MIRBuilder,
                             SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  const auto *ST =
      static_cast<const SPIRVSubtarget *>(&MIRBuilder.getMF().getSubtarget());
  if ((Opcode == SPIRV::OpControlBarrierArriveINTEL ||
       Opcode == SPIRV::OpControlBarrierWaitINTEL) &&
      !ST->canUseExtension(SPIRV::Extension::SPV_INTEL_split_barrier)) {
    std::string DiagMsg = std::string(Builtin->Name) +
                          ": the builtin requires the following SPIR-V "
                          "extension: SPV_INTEL_split_barrier";
    report_fatal_error(DiagMsg.c_str(), false);
  }

  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call, Register(0));

  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  unsigned MemFlags = getIConstVal(Call->Arguments[0], MRI);
  unsigned MemSemantics = SPIRV::MemorySemantics::None;

  if (MemFlags & SPIRV::CLK_LOCAL_MEM_FENCE)
    MemSemantics |= SPIRV::MemorySemantics::WorkgroupMemory;

  if (MemFlags & SPIRV::CLK_GLOBAL_MEM_FENCE)
    MemSemantics |= SPIRV::MemorySemantics::CrossWorkgroupMemory;

  if (MemFlags & SPIRV::CLK_IMAGE_MEM_FENCE)
    MemSemantics |= SPIRV::MemorySemantics::ImageMemory;

  if (Opcode == SPIRV::OpMemoryBarrier)
    MemSemantics = getSPIRVMemSemantics(static_cast<std::memory_order>(
                       getIConstVal(Call->Arguments[1], MRI))) |
                   MemSemantics;
  else if (Opcode == SPIRV::OpControlBarrierArriveINTEL)
    MemSemantics |= SPIRV::MemorySemantics::Release;
  else if (Opcode == SPIRV::OpControlBarrierWaitINTEL)
    MemSemantics |= SPIRV::MemorySemantics::Acquire;
  else
    MemSemantics |= SPIRV::MemorySemantics::SequentiallyConsistent;

  Register MemSemanticsReg =
      MemFlags == MemSemantics
          ? Call->Arguments[0]
          : buildConstantIntReg32(MemSemantics, MIRBuilder, GR);
  Register ScopeReg;
  SPIRV::Scope::Scope Scope = SPIRV::Scope::Workgroup;
  SPIRV::Scope::Scope MemScope = Scope;
  if (Call->Arguments.size() >= 2) {
    assert(
        ((Opcode != SPIRV::OpMemoryBarrier && Call->Arguments.size() == 2) ||
         (Opcode == SPIRV::OpMemoryBarrier && Call->Arguments.size() == 3)) &&
        "Extra args for explicitly scoped barrier");
    Register ScopeArg = (Opcode == SPIRV::OpMemoryBarrier) ? Call->Arguments[2]
                                                           : Call->Arguments[1];
    SPIRV::CLMemoryScope CLScope =
        static_cast<SPIRV::CLMemoryScope>(getIConstVal(ScopeArg, MRI));
    MemScope = getSPIRVScope(CLScope);
    if (!(MemFlags & SPIRV::CLK_LOCAL_MEM_FENCE) ||
        (Opcode == SPIRV::OpMemoryBarrier))
      Scope = MemScope;
    if (CLScope == static_cast<unsigned>(Scope))
      ScopeReg = Call->Arguments[1];
  }

  if (!ScopeReg.isValid())
    ScopeReg = buildConstantIntReg32(Scope, MIRBuilder, GR);

  auto MIB = MIRBuilder.buildInstr(Opcode).addUse(ScopeReg);
  if (Opcode != SPIRV::OpMemoryBarrier)
    MIB.addUse(buildConstantIntReg32(MemScope, MIRBuilder, GR));
  MIB.addUse(MemSemanticsReg);
  return true;
}

/// Helper function for building extended bit operations.
static bool buildExtendedBitOpsInst(const SPIRV::IncomingCall *Call,
                                    unsigned Opcode,
                                    MachineIRBuilder &MIRBuilder,
                                    SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  const auto *ST =
      static_cast<const SPIRVSubtarget *>(&MIRBuilder.getMF().getSubtarget());
  if ((Opcode == SPIRV::OpBitFieldInsert ||
       Opcode == SPIRV::OpBitFieldSExtract ||
       Opcode == SPIRV::OpBitFieldUExtract || Opcode == SPIRV::OpBitReverse) &&
      !ST->canUseExtension(SPIRV::Extension::SPV_KHR_bit_instructions)) {
    std::string DiagMsg = std::string(Builtin->Name) +
                          ": the builtin requires the following SPIR-V "
                          "extension: SPV_KHR_bit_instructions";
    report_fatal_error(DiagMsg.c_str(), false);
  }

  // Generate SPIRV instruction accordingly.
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));

  auto MIB = MIRBuilder.buildInstr(Opcode)
                 .addDef(Call->ReturnRegister)
                 .addUse(GR->getSPIRVTypeID(Call->ReturnType));
  for (unsigned i = 0; i < Call->Arguments.size(); ++i)
    MIB.addUse(Call->Arguments[i]);

  return true;
}

/// Helper function for building Intel's bindless image instructions.
static bool buildBindlessImageINTELInst(const SPIRV::IncomingCall *Call,
                                        unsigned Opcode,
                                        MachineIRBuilder &MIRBuilder,
                                        SPIRVGlobalRegistry *GR) {
  // Generate SPIRV instruction accordingly.
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));

  MIRBuilder.buildInstr(Opcode)
      .addDef(Call->ReturnRegister)
      .addUse(GR->getSPIRVTypeID(Call->ReturnType))
      .addUse(Call->Arguments[0]);

  return true;
}

/// Helper function for building Intel's OpBitwiseFunctionINTEL instruction.
static bool buildTernaryBitwiseFunctionINTELInst(
    const SPIRV::IncomingCall *Call, unsigned Opcode,
    MachineIRBuilder &MIRBuilder, SPIRVGlobalRegistry *GR) {
  // Generate SPIRV instruction accordingly.
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));

  auto MIB = MIRBuilder.buildInstr(Opcode)
                 .addDef(Call->ReturnRegister)
                 .addUse(GR->getSPIRVTypeID(Call->ReturnType));
  for (unsigned i = 0; i < Call->Arguments.size(); ++i)
    MIB.addUse(Call->Arguments[i]);

  return true;
}

/// Helper function for building Intel's 2d block io instructions.
static bool build2DBlockIOINTELInst(const SPIRV::IncomingCall *Call,
                                    unsigned Opcode,
                                    MachineIRBuilder &MIRBuilder,
                                    SPIRVGlobalRegistry *GR) {
  // Generate SPIRV instruction accordingly.
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call, Register(0));

  auto MIB = MIRBuilder.buildInstr(Opcode)
                 .addDef(Call->ReturnRegister)
                 .addUse(GR->getSPIRVTypeID(Call->ReturnType));
  for (unsigned i = 0; i < Call->Arguments.size(); ++i)
    MIB.addUse(Call->Arguments[i]);

  return true;
}

static bool buildPipeInst(const SPIRV::IncomingCall *Call, unsigned Opcode,
                          unsigned Scope, MachineIRBuilder &MIRBuilder,
                          SPIRVGlobalRegistry *GR) {
  switch (Opcode) {
  case SPIRV::OpCommitReadPipe:
  case SPIRV::OpCommitWritePipe:
    return buildOpFromWrapper(MIRBuilder, Opcode, Call, Register(0));
  case SPIRV::OpGroupCommitReadPipe:
  case SPIRV::OpGroupCommitWritePipe:
  case SPIRV::OpGroupReserveReadPipePackets:
  case SPIRV::OpGroupReserveWritePipePackets: {
    Register ScopeConstReg =
        MIRBuilder.buildConstant(LLT::scalar(32), Scope).getReg(0);
    MachineRegisterInfo *MRI = MIRBuilder.getMRI();
    MRI->setRegClass(ScopeConstReg, &SPIRV::iIDRegClass);
    MachineInstrBuilder MIB;
    MIB = MIRBuilder.buildInstr(Opcode);
    // Add Return register and type.
    if (Opcode == SPIRV::OpGroupReserveReadPipePackets ||
        Opcode == SPIRV::OpGroupReserveWritePipePackets)
      MIB.addDef(Call->ReturnRegister)
          .addUse(GR->getSPIRVTypeID(Call->ReturnType));

    MIB.addUse(ScopeConstReg);
    for (unsigned int i = 0; i < Call->Arguments.size(); ++i)
      MIB.addUse(Call->Arguments[i]);

    return true;
  }
  default:
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));
  }
}

static unsigned getNumComponentsForDim(SPIRV::Dim::Dim dim) {
  switch (dim) {
  case SPIRV::Dim::DIM_1D:
  case SPIRV::Dim::DIM_Buffer:
    return 1;
  case SPIRV::Dim::DIM_2D:
  case SPIRV::Dim::DIM_Cube:
  case SPIRV::Dim::DIM_Rect:
    return 2;
  case SPIRV::Dim::DIM_3D:
    return 3;
  default:
    report_fatal_error("Cannot get num components for given Dim");
  }
}

/// Helper function for obtaining the number of size components.
static unsigned getNumSizeComponents(SPIRVType *imgType) {
  assert(imgType->getOpcode() == SPIRV::OpTypeImage);
  auto dim = static_cast<SPIRV::Dim::Dim>(imgType->getOperand(2).getImm());
  unsigned numComps = getNumComponentsForDim(dim);
  bool arrayed = imgType->getOperand(4).getImm() == 1;
  return arrayed ? numComps + 1 : numComps;
}

//===----------------------------------------------------------------------===//
// Implementation functions for each builtin group
//===----------------------------------------------------------------------===//

static bool generateExtInst(const SPIRV::IncomingCall *Call,
                            MachineIRBuilder &MIRBuilder,
                            SPIRVGlobalRegistry *GR, const CallBase &CB) {
  // Lookup the extended instruction number in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  uint32_t Number =
      SPIRV::lookupExtendedBuiltin(Builtin->Name, Builtin->Set)->Number;
  // fmin_common and fmax_common are now deprecated, and we should use fmin and
  // fmax with NotInf and NotNaN flags instead. Keep original number to add
  // later the NoNans and NoInfs flags.
  uint32_t OrigNumber = Number;
  const SPIRVSubtarget &ST =
      cast<SPIRVSubtarget>(MIRBuilder.getMF().getSubtarget());
  if (ST.canUseExtension(SPIRV::Extension::SPV_KHR_float_controls2) &&
      (Number == SPIRV::OpenCLExtInst::fmin_common ||
       Number == SPIRV::OpenCLExtInst::fmax_common)) {
    Number = (Number == SPIRV::OpenCLExtInst::fmin_common)
                 ? SPIRV::OpenCLExtInst::fmin
                 : SPIRV::OpenCLExtInst::fmax;
  }

  // Build extended instruction.
  auto MIB =
      MIRBuilder.buildInstr(SPIRV::OpExtInst)
          .addDef(Call->ReturnRegister)
          .addUse(GR->getSPIRVTypeID(Call->ReturnType))
          .addImm(static_cast<uint32_t>(SPIRV::InstructionSet::OpenCL_std))
          .addImm(Number);

  for (auto Argument : Call->Arguments)
    MIB.addUse(Argument);
  MIB.getInstr()->copyIRFlags(CB);
  if (OrigNumber == SPIRV::OpenCLExtInst::fmin_common ||
      OrigNumber == SPIRV::OpenCLExtInst::fmax_common) {
    // Add NoNans and NoInfs flags to fmin/fmax instruction.
    MIB.getInstr()->setFlag(MachineInstr::MIFlag::FmNoNans);
    MIB.getInstr()->setFlag(MachineInstr::MIFlag::FmNoInfs);
  }
  return true;
}

static bool generateRelationalInst(const SPIRV::IncomingCall *Call,
                                   MachineIRBuilder &MIRBuilder,
                                   SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  Register CompareRegister;
  SPIRVType *RelationType;
  std::tie(CompareRegister, RelationType) =
      buildBoolRegister(MIRBuilder, Call->ReturnType, GR);

  // Build relational instruction.
  auto MIB = MIRBuilder.buildInstr(Opcode)
                 .addDef(CompareRegister)
                 .addUse(GR->getSPIRVTypeID(RelationType));

  for (auto Argument : Call->Arguments)
    MIB.addUse(Argument);

  // Build select instruction.
  return buildSelectInst(MIRBuilder, Call->ReturnRegister, CompareRegister,
                         Call->ReturnType, GR);
}

static bool generateGroupInst(const SPIRV::IncomingCall *Call,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  const SPIRV::GroupBuiltin *GroupBuiltin =
      SPIRV::lookupGroupBuiltin(Builtin->Name);

  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  if (Call->isSpirvOp()) {
    if (GroupBuiltin->NoGroupOperation) {
      SmallVector<uint32_t, 1> ImmArgs;
      if (GroupBuiltin->Opcode ==
              SPIRV::OpSubgroupMatrixMultiplyAccumulateINTEL &&
          Call->Arguments.size() > 4)
        ImmArgs.push_back(getConstFromIntrinsic(Call->Arguments[4], MRI));
      return buildOpFromWrapper(MIRBuilder, GroupBuiltin->Opcode, Call,
                                GR->getSPIRVTypeID(Call->ReturnType), ImmArgs);
    }

    // Group Operation is a literal
    Register GroupOpReg = Call->Arguments[1];
    const MachineInstr *MI = getDefInstrMaybeConstant(GroupOpReg, MRI);
    if (!MI || MI->getOpcode() != TargetOpcode::G_CONSTANT)
      report_fatal_error(
          "Group Operation parameter must be an integer constant");
    uint64_t GrpOp = MI->getOperand(1).getCImm()->getValue().getZExtValue();
    Register ScopeReg = Call->Arguments[0];
    auto MIB = MIRBuilder.buildInstr(GroupBuiltin->Opcode)
                   .addDef(Call->ReturnRegister)
                   .addUse(GR->getSPIRVTypeID(Call->ReturnType))
                   .addUse(ScopeReg)
                   .addImm(GrpOp);
    for (unsigned i = 2; i < Call->Arguments.size(); ++i)
      MIB.addUse(Call->Arguments[i]);
    return true;
  }

  Register Arg0;
  if (GroupBuiltin->HasBoolArg) {
    SPIRVType *BoolType = GR->getOrCreateSPIRVBoolType(MIRBuilder, true);
    Register BoolReg = Call->Arguments[0];
    SPIRVType *BoolRegType = GR->getSPIRVTypeForVReg(BoolReg);
    if (!BoolRegType)
      report_fatal_error("Can't find a register's type definition");
    MachineInstr *ArgInstruction = getDefInstrMaybeConstant(BoolReg, MRI);
    if (ArgInstruction->getOpcode() == TargetOpcode::G_CONSTANT) {
      if (BoolRegType->getOpcode() != SPIRV::OpTypeBool)
        Arg0 = GR->buildConstantInt(getIConstVal(BoolReg, MRI), MIRBuilder,
                                    BoolType, true);
    } else {
      if (BoolRegType->getOpcode() == SPIRV::OpTypeInt) {
        Arg0 = MRI->createGenericVirtualRegister(LLT::scalar(1));
        MRI->setRegClass(Arg0, &SPIRV::iIDRegClass);
        GR->assignSPIRVTypeToVReg(BoolType, Arg0, MIRBuilder.getMF());
        MIRBuilder.buildICmp(
            CmpInst::ICMP_NE, Arg0, BoolReg,
            GR->buildConstantInt(0, MIRBuilder, BoolRegType, true));
        insertAssignInstr(Arg0, nullptr, BoolType, GR, MIRBuilder,
                          MIRBuilder.getMF().getRegInfo());
      } else if (BoolRegType->getOpcode() != SPIRV::OpTypeBool) {
        report_fatal_error("Expect a boolean argument");
      }
      // if BoolReg is a boolean register, we don't need to do anything
    }
  }

  Register GroupResultRegister = Call->ReturnRegister;
  SPIRVType *GroupResultType = Call->ReturnType;

  // TODO: maybe we need to check whether the result type is already boolean
  // and in this case do not insert select instruction.
  const bool HasBoolReturnTy =
      GroupBuiltin->IsElect || GroupBuiltin->IsAllOrAny ||
      GroupBuiltin->IsAllEqual || GroupBuiltin->IsLogical ||
      GroupBuiltin->IsInverseBallot || GroupBuiltin->IsBallotBitExtract;

  if (HasBoolReturnTy)
    std::tie(GroupResultRegister, GroupResultType) =
        buildBoolRegister(MIRBuilder, Call->ReturnType, GR);

  auto Scope = Builtin->Name.starts_with("sub_group") ? SPIRV::Scope::Subgroup
                                                      : SPIRV::Scope::Workgroup;
  Register ScopeRegister = buildConstantIntReg32(Scope, MIRBuilder, GR);

  Register VecReg;
  if (GroupBuiltin->Opcode == SPIRV::OpGroupBroadcast &&
      Call->Arguments.size() > 2) {
    // For OpGroupBroadcast "LocalId must be an integer datatype. It must be a
    // scalar, a vector with 2 components, or a vector with 3 components.",
    // meaning that we must create a vector from the function arguments if
    // it's a work_group_broadcast(val, local_id_x, local_id_y) or
    // work_group_broadcast(val, local_id_x, local_id_y, local_id_z) call.
    Register ElemReg = Call->Arguments[1];
    SPIRVType *ElemType = GR->getSPIRVTypeForVReg(ElemReg);
    if (!ElemType || ElemType->getOpcode() != SPIRV::OpTypeInt)
      report_fatal_error("Expect an integer <LocalId> argument");
    unsigned VecLen = Call->Arguments.size() - 1;
    VecReg = MRI->createGenericVirtualRegister(
        LLT::fixed_vector(VecLen, MRI->getType(ElemReg)));
    MRI->setRegClass(VecReg, &SPIRV::vIDRegClass);
    SPIRVType *VecType =
        GR->getOrCreateSPIRVVectorType(ElemType, VecLen, MIRBuilder, true);
    GR->assignSPIRVTypeToVReg(VecType, VecReg, MIRBuilder.getMF());
    auto MIB =
        MIRBuilder.buildInstr(TargetOpcode::G_BUILD_VECTOR).addDef(VecReg);
    for (unsigned i = 1; i < Call->Arguments.size(); i++) {
      MIB.addUse(Call->Arguments[i]);
      setRegClassIfNull(Call->Arguments[i], MRI, GR);
    }
    insertAssignInstr(VecReg, nullptr, VecType, GR, MIRBuilder,
                      MIRBuilder.getMF().getRegInfo());
  }

  // Build work/sub group instruction.
  auto MIB = MIRBuilder.buildInstr(GroupBuiltin->Opcode)
                 .addDef(GroupResultRegister)
                 .addUse(GR->getSPIRVTypeID(GroupResultType))
                 .addUse(ScopeRegister);

  if (!GroupBuiltin->NoGroupOperation)
    MIB.addImm(GroupBuiltin->GroupOperation);
  if (Call->Arguments.size() > 0) {
    MIB.addUse(Arg0.isValid() ? Arg0 : Call->Arguments[0]);
    setRegClassIfNull(Call->Arguments[0], MRI, GR);
    if (VecReg.isValid())
      MIB.addUse(VecReg);
    else
      for (unsigned i = 1; i < Call->Arguments.size(); i++)
        MIB.addUse(Call->Arguments[i]);
  }

  // Build select instruction.
  if (HasBoolReturnTy)
    buildSelectInst(MIRBuilder, Call->ReturnRegister, GroupResultRegister,
                    Call->ReturnType, GR);
  return true;
}

static bool generateIntelSubgroupsInst(const SPIRV::IncomingCall *Call,
                                       MachineIRBuilder &MIRBuilder,
                                       SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  MachineFunction &MF = MIRBuilder.getMF();
  const auto *ST = static_cast<const SPIRVSubtarget *>(&MF.getSubtarget());
  const SPIRV::IntelSubgroupsBuiltin *IntelSubgroups =
      SPIRV::lookupIntelSubgroupsBuiltin(Builtin->Name);

  if (IntelSubgroups->IsMedia &&
      !ST->canUseExtension(SPIRV::Extension::SPV_INTEL_media_block_io)) {
    std::string DiagMsg = std::string(Builtin->Name) +
                          ": the builtin requires the following SPIR-V "
                          "extension: SPV_INTEL_media_block_io";
    report_fatal_error(DiagMsg.c_str(), false);
  } else if (!IntelSubgroups->IsMedia &&
             !ST->canUseExtension(SPIRV::Extension::SPV_INTEL_subgroups)) {
    std::string DiagMsg = std::string(Builtin->Name) +
                          ": the builtin requires the following SPIR-V "
                          "extension: SPV_INTEL_subgroups";
    report_fatal_error(DiagMsg.c_str(), false);
  }

  uint32_t OpCode = IntelSubgroups->Opcode;
  if (Call->isSpirvOp()) {
    bool IsSet = OpCode != SPIRV::OpSubgroupBlockWriteINTEL &&
                 OpCode != SPIRV::OpSubgroupImageBlockWriteINTEL &&
                 OpCode != SPIRV::OpSubgroupImageMediaBlockWriteINTEL;
    return buildOpFromWrapper(MIRBuilder, OpCode, Call,
                              IsSet ? GR->getSPIRVTypeID(Call->ReturnType)
                                    : Register(0));
  }

  if (IntelSubgroups->IsBlock) {
    // Minimal number or arguments set in TableGen records is 1
    if (SPIRVType *Arg0Type = GR->getSPIRVTypeForVReg(Call->Arguments[0])) {
      if (Arg0Type->getOpcode() == SPIRV::OpTypeImage) {
        // TODO: add required validation from the specification:
        // "'Image' must be an object whose type is OpTypeImage with a 'Sampled'
        // operand of 0 or 2. If the 'Sampled' operand is 2, then some
        // dimensions require a capability."
        switch (OpCode) {
        case SPIRV::OpSubgroupBlockReadINTEL:
          OpCode = SPIRV::OpSubgroupImageBlockReadINTEL;
          break;
        case SPIRV::OpSubgroupBlockWriteINTEL:
          OpCode = SPIRV::OpSubgroupImageBlockWriteINTEL;
          break;
        }
      }
    }
  }

  // TODO: opaque pointers types should be eventually resolved in such a way
  // that validation of block read is enabled with respect to the following
  // specification requirement:
  // "'Result Type' may be a scalar or vector type, and its component type must
  // be equal to the type pointed to by 'Ptr'."
  // For example, function parameter type should not be default i8 pointer, but
  // depend on the result type of the instruction where it is used as a pointer
  // argument of OpSubgroupBlockReadINTEL

  // Build Intel subgroups instruction
  MachineInstrBuilder MIB =
      IntelSubgroups->IsWrite
          ? MIRBuilder.buildInstr(OpCode)
          : MIRBuilder.buildInstr(OpCode)
                .addDef(Call->ReturnRegister)
                .addUse(GR->getSPIRVTypeID(Call->ReturnType));
  for (size_t i = 0; i < Call->Arguments.size(); ++i)
    MIB.addUse(Call->Arguments[i]);
  return true;
}

static bool generateGroupUniformInst(const SPIRV::IncomingCall *Call,
                                     MachineIRBuilder &MIRBuilder,
                                     SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  MachineFunction &MF = MIRBuilder.getMF();
  const auto *ST = static_cast<const SPIRVSubtarget *>(&MF.getSubtarget());
  if (!ST->canUseExtension(
          SPIRV::Extension::SPV_KHR_uniform_group_instructions)) {
    std::string DiagMsg = std::string(Builtin->Name) +
                          ": the builtin requires the following SPIR-V "
                          "extension: SPV_KHR_uniform_group_instructions";
    report_fatal_error(DiagMsg.c_str(), false);
  }
  const SPIRV::GroupUniformBuiltin *GroupUniform =
      SPIRV::lookupGroupUniformBuiltin(Builtin->Name);
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();

  Register GroupResultReg = Call->ReturnRegister;
  Register ScopeReg = Call->Arguments[0];
  Register ValueReg = Call->Arguments[2];

  // Group Operation
  Register ConstGroupOpReg = Call->Arguments[1];
  const MachineInstr *Const = getDefInstrMaybeConstant(ConstGroupOpReg, MRI);
  if (!Const || Const->getOpcode() != TargetOpcode::G_CONSTANT)
    report_fatal_error(
        "expect a constant group operation for a uniform group instruction",
        false);
  const MachineOperand &ConstOperand = Const->getOperand(1);
  if (!ConstOperand.isCImm())
    report_fatal_error("uniform group instructions: group operation must be an "
                       "integer constant",
                       false);

  auto MIB = MIRBuilder.buildInstr(GroupUniform->Opcode)
                 .addDef(GroupResultReg)
                 .addUse(GR->getSPIRVTypeID(Call->ReturnType))
                 .addUse(ScopeReg);
  addNumImm(ConstOperand.getCImm()->getValue(), MIB);
  MIB.addUse(ValueReg);

  return true;
}

static bool generateKernelClockInst(const SPIRV::IncomingCall *Call,
                                    MachineIRBuilder &MIRBuilder,
                                    SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  MachineFunction &MF = MIRBuilder.getMF();
  const auto *ST = static_cast<const SPIRVSubtarget *>(&MF.getSubtarget());
  if (!ST->canUseExtension(SPIRV::Extension::SPV_KHR_shader_clock)) {
    std::string DiagMsg = std::string(Builtin->Name) +
                          ": the builtin requires the following SPIR-V "
                          "extension: SPV_KHR_shader_clock";
    report_fatal_error(DiagMsg.c_str(), false);
  }

  Register ResultReg = Call->ReturnRegister;

  // Deduce the `Scope` operand from the builtin function name.
  SPIRV::Scope::Scope ScopeArg =
      StringSwitch<SPIRV::Scope::Scope>(Builtin->Name)
          .EndsWith("device", SPIRV::Scope::Scope::Device)
          .EndsWith("work_group", SPIRV::Scope::Scope::Workgroup)
          .EndsWith("sub_group", SPIRV::Scope::Scope::Subgroup);
  Register ScopeReg = buildConstantIntReg32(ScopeArg, MIRBuilder, GR);

  MIRBuilder.buildInstr(SPIRV::OpReadClockKHR)
      .addDef(ResultReg)
      .addUse(GR->getSPIRVTypeID(Call->ReturnType))
      .addUse(ScopeReg);

  return true;
}

// These queries ask for a single size_t result for a given dimension index,
// e.g. size_t get_global_id(uint dimindex). In SPIR-V, the builtins
// corresponding to these values are all vec3 types, so we need to extract the
// correct index or return DefaultValue (0 or 1 depending on the query). We also
// handle extending or truncating in case size_t does not match the expected
// result type's bitwidth.
//
// For a constant index >= 3 we generate:
//  %res = OpConstant %SizeT DefaultValue
//
// For other indices we generate:
//  %g = OpVariable %ptr_V3_SizeT Input
//  OpDecorate %g BuiltIn XXX
//  OpDecorate %g LinkageAttributes "__spirv_BuiltInXXX"
//  OpDecorate %g Constant
//  %loadedVec = OpLoad %V3_SizeT %g
//
//  Then, if the index is constant < 3, we generate:
//    %res = OpCompositeExtract %SizeT %loadedVec idx
//  If the index is dynamic, we generate:
//    %tmp = OpVectorExtractDynamic %SizeT %loadedVec %idx
//    %cmp = OpULessThan %bool %idx %const_3
//    %res = OpSelect %SizeT %cmp %tmp %const_<DefaultValue>
//
//  If the bitwidth of %res does not match the expected return type, we add an
//  extend or truncate.
static bool genWorkgroupQuery(const SPIRV::IncomingCall *Call,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR,
                              SPIRV::BuiltIn::BuiltIn BuiltinValue,
                              uint64_t DefaultValue) {
  Register IndexRegister = Call->Arguments[0];
  const unsigned ResultWidth = Call->ReturnType->getOperand(1).getImm();
  const unsigned PointerSize = GR->getPointerSize();
  const SPIRVType *PointerSizeType =
      GR->getOrCreateSPIRVIntegerType(PointerSize, MIRBuilder);
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  auto IndexInstruction = getDefInstrMaybeConstant(IndexRegister, MRI);

  // Set up the final register to do truncation or extension on at the end.
  Register ToTruncate = Call->ReturnRegister;

  // If the index is constant, we can statically determine if it is in range.
  bool IsConstantIndex =
      IndexInstruction->getOpcode() == TargetOpcode::G_CONSTANT;

  // If it's out of range (max dimension is 3), we can just return the constant
  // default value (0 or 1 depending on which query function).
  if (IsConstantIndex && getIConstVal(IndexRegister, MRI) >= 3) {
    Register DefaultReg = Call->ReturnRegister;
    if (PointerSize != ResultWidth) {
      DefaultReg = MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
      MRI->setRegClass(DefaultReg, &SPIRV::iIDRegClass);
      GR->assignSPIRVTypeToVReg(PointerSizeType, DefaultReg,
                                MIRBuilder.getMF());
      ToTruncate = DefaultReg;
    }
    auto NewRegister =
        GR->buildConstantInt(DefaultValue, MIRBuilder, PointerSizeType, true);
    MIRBuilder.buildCopy(DefaultReg, NewRegister);
  } else { // If it could be in range, we need to load from the given builtin.
    auto Vec3Ty =
        GR->getOrCreateSPIRVVectorType(PointerSizeType, 3, MIRBuilder, true);
    Register LoadedVector =
        buildBuiltinVariableLoad(MIRBuilder, Vec3Ty, GR, BuiltinValue,
                                 LLT::fixed_vector(3, PointerSize));
    // Set up the vreg to extract the result to (possibly a new temporary one).
    Register Extracted = Call->ReturnRegister;
    if (!IsConstantIndex || PointerSize != ResultWidth) {
      Extracted = MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
      MRI->setRegClass(Extracted, &SPIRV::iIDRegClass);
      GR->assignSPIRVTypeToVReg(PointerSizeType, Extracted, MIRBuilder.getMF());
    }
    // Use Intrinsic::spv_extractelt so dynamic vs static extraction is
    // handled later: extr = spv_extractelt LoadedVector, IndexRegister.
    MachineInstrBuilder ExtractInst = MIRBuilder.buildIntrinsic(
        Intrinsic::spv_extractelt, ArrayRef<Register>{Extracted}, true, false);
    ExtractInst.addUse(LoadedVector).addUse(IndexRegister);

    // If the index is dynamic, need check if it's < 3, and then use a select.
    if (!IsConstantIndex) {
      insertAssignInstr(Extracted, nullptr, PointerSizeType, GR, MIRBuilder,
                        *MRI);

      auto IndexType = GR->getSPIRVTypeForVReg(IndexRegister);
      auto BoolType = GR->getOrCreateSPIRVBoolType(MIRBuilder, true);

      Register CompareRegister =
          MRI->createGenericVirtualRegister(LLT::scalar(1));
      MRI->setRegClass(CompareRegister, &SPIRV::iIDRegClass);
      GR->assignSPIRVTypeToVReg(BoolType, CompareRegister, MIRBuilder.getMF());

      // Use G_ICMP to check if idxVReg < 3.
      MIRBuilder.buildICmp(
          CmpInst::ICMP_ULT, CompareRegister, IndexRegister,
          GR->buildConstantInt(3, MIRBuilder, IndexType, true));

      // Get constant for the default value (0 or 1 depending on which
      // function).
      Register DefaultRegister =
          GR->buildConstantInt(DefaultValue, MIRBuilder, PointerSizeType, true);

      // Get a register for the selection result (possibly a new temporary one).
      Register SelectionResult = Call->ReturnRegister;
      if (PointerSize != ResultWidth) {
        SelectionResult =
            MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
        MRI->setRegClass(SelectionResult, &SPIRV::iIDRegClass);
        GR->assignSPIRVTypeToVReg(PointerSizeType, SelectionResult,
                                  MIRBuilder.getMF());
      }
      // Create the final G_SELECT to return the extracted value or the default.
      MIRBuilder.buildSelect(SelectionResult, CompareRegister, Extracted,
                             DefaultRegister);
      ToTruncate = SelectionResult;
    } else {
      ToTruncate = Extracted;
    }
  }
  // Alter the result's bitwidth if it does not match the SizeT value extracted.
  if (PointerSize != ResultWidth)
    MIRBuilder.buildZExtOrTrunc(Call->ReturnRegister, ToTruncate);
  return true;
}

static bool generateBuiltinVar(const SPIRV::IncomingCall *Call,
                               MachineIRBuilder &MIRBuilder,
                               SPIRVGlobalRegistry *GR) {
  // Lookup the builtin variable record.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  SPIRV::BuiltIn::BuiltIn Value =
      SPIRV::lookupGetBuiltin(Builtin->Name, Builtin->Set)->Value;

  if (Value == SPIRV::BuiltIn::GlobalInvocationId)
    return genWorkgroupQuery(Call, MIRBuilder, GR, Value, 0);

  // Build a load instruction for the builtin variable.
  unsigned BitWidth = GR->getScalarOrVectorBitWidth(Call->ReturnType);
  LLT LLType;
  if (Call->ReturnType->getOpcode() == SPIRV::OpTypeVector)
    LLType =
        LLT::fixed_vector(Call->ReturnType->getOperand(2).getImm(), BitWidth);
  else
    LLType = LLT::scalar(BitWidth);

  return buildBuiltinVariableLoad(MIRBuilder, Call->ReturnType, GR, Value,
                                  LLType, Call->ReturnRegister);
}

static bool generateAtomicInst(const SPIRV::IncomingCall *Call,
                               MachineIRBuilder &MIRBuilder,
                               SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  switch (Opcode) {
  case SPIRV::OpStore:
    return buildAtomicInitInst(Call, MIRBuilder);
  case SPIRV::OpAtomicLoad:
    return buildAtomicLoadInst(Call, MIRBuilder, GR);
  case SPIRV::OpAtomicStore:
    return buildAtomicStoreInst(Call, MIRBuilder, GR);
  case SPIRV::OpAtomicCompareExchange:
  case SPIRV::OpAtomicCompareExchangeWeak:
    return buildAtomicCompareExchangeInst(Call, Builtin, Opcode, MIRBuilder,
                                          GR);
  case SPIRV::OpAtomicIAdd:
  case SPIRV::OpAtomicISub:
  case SPIRV::OpAtomicOr:
  case SPIRV::OpAtomicXor:
  case SPIRV::OpAtomicAnd:
  case SPIRV::OpAtomicExchange:
    return buildAtomicRMWInst(Call, Opcode, MIRBuilder, GR);
  case SPIRV::OpMemoryBarrier:
    return buildBarrierInst(Call, SPIRV::OpMemoryBarrier, MIRBuilder, GR);
  case SPIRV::OpAtomicFlagTestAndSet:
  case SPIRV::OpAtomicFlagClear:
    return buildAtomicFlagInst(Call, Opcode, MIRBuilder, GR);
  default:
    if (Call->isSpirvOp())
      return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                                GR->getSPIRVTypeID(Call->ReturnType));
    return false;
  }
}

static bool generateAtomicFloatingInst(const SPIRV::IncomingCall *Call,
                                       MachineIRBuilder &MIRBuilder,
                                       SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode = SPIRV::lookupAtomicFloatingBuiltin(Builtin->Name)->Opcode;

  switch (Opcode) {
  case SPIRV::OpAtomicFAddEXT:
  case SPIRV::OpAtomicFMinEXT:
  case SPIRV::OpAtomicFMaxEXT:
    return buildAtomicFloatingRMWInst(Call, Opcode, MIRBuilder, GR);
  default:
    return false;
  }
}

static bool generateBarrierInst(const SPIRV::IncomingCall *Call,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  return buildBarrierInst(Call, Opcode, MIRBuilder, GR);
}

static bool generateCastToPtrInst(const SPIRV::IncomingCall *Call,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  if (Opcode == SPIRV::OpGenericCastToPtrExplicit) {
    SPIRV::StorageClass::StorageClass ResSC =
        GR->getPointerStorageClass(Call->ReturnRegister);
    if (!isGenericCastablePtr(ResSC))
      return false;

    MIRBuilder.buildInstr(Opcode)
        .addDef(Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(Call->ReturnType))
        .addUse(Call->Arguments[0])
        .addImm(ResSC);
  } else {
    MIRBuilder.buildInstr(TargetOpcode::G_ADDRSPACE_CAST)
        .addDef(Call->ReturnRegister)
        .addUse(Call->Arguments[0]);
  }
  return true;
}

static bool generateDotOrFMulInst(const StringRef DemangledCall,
                                  const SPIRV::IncomingCall *Call,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRVGlobalRegistry *GR) {
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, SPIRV::OpDot, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));

  bool IsVec = GR->getSPIRVTypeForVReg(Call->Arguments[0])->getOpcode() ==
               SPIRV::OpTypeVector;
  // Use OpDot only in case of vector args and OpFMul in case of scalar args.
  uint32_t OC = IsVec ? SPIRV::OpDot : SPIRV::OpFMulS;
  bool IsSwapReq = false;

  const auto *ST =
      static_cast<const SPIRVSubtarget *>(&MIRBuilder.getMF().getSubtarget());
  if (GR->isScalarOrVectorOfType(Call->ReturnRegister, SPIRV::OpTypeInt) &&
      (ST->canUseExtension(SPIRV::Extension::SPV_KHR_integer_dot_product) ||
       ST->isAtLeastSPIRVVer(VersionTuple(1, 6)))) {
    const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
    const SPIRV::IntegerDotProductBuiltin *IntDot =
        SPIRV::lookupIntegerDotProductBuiltin(Builtin->Name);
    if (IntDot) {
      OC = IntDot->Opcode;
      IsSwapReq = IntDot->IsSwapReq;
    } else if (IsVec) {
      // Handling "dot" and "dot_acc_sat" builtins which use vectors of
      // integers.
      LLVMContext &Ctx = MIRBuilder.getContext();
      SmallVector<StringRef, 10> TypeStrs;
      SPIRV::parseBuiltinTypeStr(TypeStrs, DemangledCall, Ctx);
      bool IsFirstSigned = TypeStrs[0].trim()[0] != 'u';
      bool IsSecondSigned = TypeStrs[1].trim()[0] != 'u';

      if (Call->BuiltinName == "dot") {
        if (IsFirstSigned && IsSecondSigned)
          OC = SPIRV::OpSDot;
        else if (!IsFirstSigned && !IsSecondSigned)
          OC = SPIRV::OpUDot;
        else {
          OC = SPIRV::OpSUDot;
          if (!IsFirstSigned)
            IsSwapReq = true;
        }
      } else if (Call->BuiltinName == "dot_acc_sat") {
        if (IsFirstSigned && IsSecondSigned)
          OC = SPIRV::OpSDotAccSat;
        else if (!IsFirstSigned && !IsSecondSigned)
          OC = SPIRV::OpUDotAccSat;
        else {
          OC = SPIRV::OpSUDotAccSat;
          if (!IsFirstSigned)
            IsSwapReq = true;
        }
      }
    }
  }

  MachineInstrBuilder MIB = MIRBuilder.buildInstr(OC)
                                .addDef(Call->ReturnRegister)
                                .addUse(GR->getSPIRVTypeID(Call->ReturnType));

  if (IsSwapReq) {
    MIB.addUse(Call->Arguments[1]);
    MIB.addUse(Call->Arguments[0]);
    // needed for dot_acc_sat* builtins
    for (size_t i = 2; i < Call->Arguments.size(); ++i)
      MIB.addUse(Call->Arguments[i]);
  } else {
    for (size_t i = 0; i < Call->Arguments.size(); ++i)
      MIB.addUse(Call->Arguments[i]);
  }

  // Add Packed Vector Format for Integer dot product builtins if arguments are
  // scalar
  if (!IsVec && OC != SPIRV::OpFMulS)
    MIB.addImm(SPIRV::PackedVectorFormat4x8Bit);

  return true;
}

static bool generateWaveInst(const SPIRV::IncomingCall *Call,
                             MachineIRBuilder &MIRBuilder,
                             SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  SPIRV::BuiltIn::BuiltIn Value =
      SPIRV::lookupGetBuiltin(Builtin->Name, Builtin->Set)->Value;

  // For now, we only support a single Wave intrinsic with a single return type.
  assert(Call->ReturnType->getOpcode() == SPIRV::OpTypeInt);
  LLT LLType = LLT::scalar(GR->getScalarOrVectorBitWidth(Call->ReturnType));

  return buildBuiltinVariableLoad(
      MIRBuilder, Call->ReturnType, GR, Value, LLType, Call->ReturnRegister,
      /* isConst= */ false, /* hasLinkageTy= */ false);
}

// We expect a builtin
//     Name(ptr sret([RetType]) %result, Type %operand1, Type %operand1)
// where %result is a pointer to where the result of the builtin execution
// is to be stored, and generate the following instructions:
//     Res = Opcode RetType Operand1 Operand1
//     OpStore RetVariable Res
static bool generateICarryBorrowInst(const SPIRV::IncomingCall *Call,
                                     MachineIRBuilder &MIRBuilder,
                                     SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  Register SRetReg = Call->Arguments[0];
  SPIRVType *PtrRetType = GR->getSPIRVTypeForVReg(SRetReg);
  SPIRVType *RetType = GR->getPointeeType(PtrRetType);
  if (!RetType)
    report_fatal_error("The first parameter must be a pointer");
  if (RetType->getOpcode() != SPIRV::OpTypeStruct)
    report_fatal_error("Expected struct type result for the arithmetic with "
                       "overflow builtins");

  SPIRVType *OpType1 = GR->getSPIRVTypeForVReg(Call->Arguments[1]);
  SPIRVType *OpType2 = GR->getSPIRVTypeForVReg(Call->Arguments[2]);
  if (!OpType1 || !OpType2 || OpType1 != OpType2)
    report_fatal_error("Operands must have the same type");
  if (OpType1->getOpcode() == SPIRV::OpTypeVector)
    switch (Opcode) {
    case SPIRV::OpIAddCarryS:
      Opcode = SPIRV::OpIAddCarryV;
      break;
    case SPIRV::OpISubBorrowS:
      Opcode = SPIRV::OpISubBorrowV;
      break;
    }

  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  Register ResReg = MRI->createVirtualRegister(&SPIRV::iIDRegClass);
  if (const TargetRegisterClass *DstRC =
          MRI->getRegClassOrNull(Call->Arguments[1])) {
    MRI->setRegClass(ResReg, DstRC);
    MRI->setType(ResReg, MRI->getType(Call->Arguments[1]));
  } else {
    MRI->setType(ResReg, LLT::scalar(64));
  }
  GR->assignSPIRVTypeToVReg(RetType, ResReg, MIRBuilder.getMF());
  MIRBuilder.buildInstr(Opcode)
      .addDef(ResReg)
      .addUse(GR->getSPIRVTypeID(RetType))
      .addUse(Call->Arguments[1])
      .addUse(Call->Arguments[2]);
  MIRBuilder.buildInstr(SPIRV::OpStore).addUse(SRetReg).addUse(ResReg);
  return true;
}

static bool generateGetQueryInst(const SPIRV::IncomingCall *Call,
                                 MachineIRBuilder &MIRBuilder,
                                 SPIRVGlobalRegistry *GR) {
  // Lookup the builtin record.
  SPIRV::BuiltIn::BuiltIn Value =
      SPIRV::lookupGetBuiltin(Call->Builtin->Name, Call->Builtin->Set)->Value;
  const bool IsDefaultOne = (Value == SPIRV::BuiltIn::GlobalSize ||
                             Value == SPIRV::BuiltIn::NumWorkgroups ||
                             Value == SPIRV::BuiltIn::WorkgroupSize ||
                             Value == SPIRV::BuiltIn::EnqueuedWorkgroupSize);
  return genWorkgroupQuery(Call, MIRBuilder, GR, Value, IsDefaultOne ? 1 : 0);
}

static bool generateImageSizeQueryInst(const SPIRV::IncomingCall *Call,
                                       MachineIRBuilder &MIRBuilder,
                                       SPIRVGlobalRegistry *GR) {
  // Lookup the image size query component number in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  uint32_t Component =
      SPIRV::lookupImageQueryBuiltin(Builtin->Name, Builtin->Set)->Component;
  // Query result may either be a vector or a scalar. If return type is not a
  // vector, expect only a single size component. Otherwise get the number of
  // expected components.
  unsigned NumExpectedRetComponents =
      Call->ReturnType->getOpcode() == SPIRV::OpTypeVector
          ? Call->ReturnType->getOperand(2).getImm()
          : 1;
  // Get the actual number of query result/size components.
  SPIRVType *ImgType = GR->getSPIRVTypeForVReg(Call->Arguments[0]);
  unsigned NumActualRetComponents = getNumSizeComponents(ImgType);
  Register QueryResult = Call->ReturnRegister;
  SPIRVType *QueryResultType = Call->ReturnType;
  if (NumExpectedRetComponents != NumActualRetComponents) {
    unsigned Bitwidth = Call->ReturnType->getOpcode() == SPIRV::OpTypeInt
                            ? Call->ReturnType->getOperand(1).getImm()
                            : 32;
    QueryResult = MIRBuilder.getMRI()->createGenericVirtualRegister(
        LLT::fixed_vector(NumActualRetComponents, Bitwidth));
    MIRBuilder.getMRI()->setRegClass(QueryResult, &SPIRV::vIDRegClass);
    SPIRVType *IntTy = GR->getOrCreateSPIRVIntegerType(Bitwidth, MIRBuilder);
    QueryResultType = GR->getOrCreateSPIRVVectorType(
        IntTy, NumActualRetComponents, MIRBuilder, true);
    GR->assignSPIRVTypeToVReg(QueryResultType, QueryResult, MIRBuilder.getMF());
  }
  bool IsDimBuf = ImgType->getOperand(2).getImm() == SPIRV::Dim::DIM_Buffer;
  unsigned Opcode =
      IsDimBuf ? SPIRV::OpImageQuerySize : SPIRV::OpImageQuerySizeLod;
  auto MIB = MIRBuilder.buildInstr(Opcode)
                 .addDef(QueryResult)
                 .addUse(GR->getSPIRVTypeID(QueryResultType))
                 .addUse(Call->Arguments[0]);
  if (!IsDimBuf)
    MIB.addUse(buildConstantIntReg32(0, MIRBuilder, GR)); // Lod id.
  if (NumExpectedRetComponents == NumActualRetComponents)
    return true;
  if (NumExpectedRetComponents == 1) {
    // Only 1 component is expected, build OpCompositeExtract instruction.
    unsigned ExtractedComposite =
        Component == 3 ? NumActualRetComponents - 1 : Component;
    assert(ExtractedComposite < NumActualRetComponents &&
           "Invalid composite index!");
    Register TypeReg = GR->getSPIRVTypeID(Call->ReturnType);
    SPIRVType *NewType = nullptr;
    if (QueryResultType->getOpcode() == SPIRV::OpTypeVector) {
      Register NewTypeReg = QueryResultType->getOperand(1).getReg();
      if (TypeReg != NewTypeReg &&
          (NewType = GR->getSPIRVTypeForVReg(NewTypeReg)) != nullptr)
        TypeReg = NewTypeReg;
    }
    MIRBuilder.buildInstr(SPIRV::OpCompositeExtract)
        .addDef(Call->ReturnRegister)
        .addUse(TypeReg)
        .addUse(QueryResult)
        .addImm(ExtractedComposite);
    if (NewType != nullptr)
      insertAssignInstr(Call->ReturnRegister, nullptr, NewType, GR, MIRBuilder,
                        MIRBuilder.getMF().getRegInfo());
  } else {
    // More than 1 component is expected, fill a new vector.
    auto MIB = MIRBuilder.buildInstr(SPIRV::OpVectorShuffle)
                   .addDef(Call->ReturnRegister)
                   .addUse(GR->getSPIRVTypeID(Call->ReturnType))
                   .addUse(QueryResult)
                   .addUse(QueryResult);
    for (unsigned i = 0; i < NumExpectedRetComponents; ++i)
      MIB.addImm(i < NumActualRetComponents ? i : 0xffffffff);
  }
  return true;
}

static bool generateImageMiscQueryInst(const SPIRV::IncomingCall *Call,
                                       MachineIRBuilder &MIRBuilder,
                                       SPIRVGlobalRegistry *GR) {
  assert(Call->ReturnType->getOpcode() == SPIRV::OpTypeInt &&
         "Image samples query result must be of int type!");

  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  Register Image = Call->Arguments[0];
  SPIRV::Dim::Dim ImageDimensionality = static_cast<SPIRV::Dim::Dim>(
      GR->getSPIRVTypeForVReg(Image)->getOperand(2).getImm());
  (void)ImageDimensionality;

  switch (Opcode) {
  case SPIRV::OpImageQuerySamples:
    assert(ImageDimensionality == SPIRV::Dim::DIM_2D &&
           "Image must be of 2D dimensionality");
    break;
  case SPIRV::OpImageQueryLevels:
    assert((ImageDimensionality == SPIRV::Dim::DIM_1D ||
            ImageDimensionality == SPIRV::Dim::DIM_2D ||
            ImageDimensionality == SPIRV::Dim::DIM_3D ||
            ImageDimensionality == SPIRV::Dim::DIM_Cube) &&
           "Image must be of 1D/2D/3D/Cube dimensionality");
    break;
  }

  MIRBuilder.buildInstr(Opcode)
      .addDef(Call->ReturnRegister)
      .addUse(GR->getSPIRVTypeID(Call->ReturnType))
      .addUse(Image);
  return true;
}

// TODO: Move to TableGen.
static SPIRV::SamplerAddressingMode::SamplerAddressingMode
getSamplerAddressingModeFromBitmask(unsigned Bitmask) {
  switch (Bitmask & SPIRV::CLK_ADDRESS_MODE_MASK) {
  case SPIRV::CLK_ADDRESS_CLAMP:
    return SPIRV::SamplerAddressingMode::Clamp;
  case SPIRV::CLK_ADDRESS_CLAMP_TO_EDGE:
    return SPIRV::SamplerAddressingMode::ClampToEdge;
  case SPIRV::CLK_ADDRESS_REPEAT:
    return SPIRV::SamplerAddressingMode::Repeat;
  case SPIRV::CLK_ADDRESS_MIRRORED_REPEAT:
    return SPIRV::SamplerAddressingMode::RepeatMirrored;
  case SPIRV::CLK_ADDRESS_NONE:
    return SPIRV::SamplerAddressingMode::None;
  default:
    report_fatal_error("Unknown CL address mode");
  }
}

static unsigned getSamplerParamFromBitmask(unsigned Bitmask) {
  return (Bitmask & SPIRV::CLK_NORMALIZED_COORDS_TRUE) ? 1 : 0;
}

static SPIRV::SamplerFilterMode::SamplerFilterMode
getSamplerFilterModeFromBitmask(unsigned Bitmask) {
  if (Bitmask & SPIRV::CLK_FILTER_LINEAR)
    return SPIRV::SamplerFilterMode::Linear;
  if (Bitmask & SPIRV::CLK_FILTER_NEAREST)
    return SPIRV::SamplerFilterMode::Nearest;
  return SPIRV::SamplerFilterMode::Nearest;
}

static bool generateReadImageInst(const StringRef DemangledCall,
                                  const SPIRV::IncomingCall *Call,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRVGlobalRegistry *GR) {
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, SPIRV::OpImageRead, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));
  Register Image = Call->Arguments[0];
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  bool HasOclSampler = DemangledCall.contains_insensitive("ocl_sampler");
  bool HasMsaa = DemangledCall.contains_insensitive("msaa");
  if (HasOclSampler) {
    Register Sampler = Call->Arguments[1];

    if (!GR->isScalarOfType(Sampler, SPIRV::OpTypeSampler) &&
        getDefInstrMaybeConstant(Sampler, MRI)->getOperand(1).isCImm()) {
      uint64_t SamplerMask = getIConstVal(Sampler, MRI);
      Sampler = GR->buildConstantSampler(
          Register(), getSamplerAddressingModeFromBitmask(SamplerMask),
          getSamplerParamFromBitmask(SamplerMask),
          getSamplerFilterModeFromBitmask(SamplerMask), MIRBuilder);
    }
    SPIRVType *ImageType = GR->getSPIRVTypeForVReg(Image);
    SPIRVType *SampledImageType =
        GR->getOrCreateOpTypeSampledImage(ImageType, MIRBuilder);
    Register SampledImage = MRI->createVirtualRegister(&SPIRV::iIDRegClass);

    MIRBuilder.buildInstr(SPIRV::OpSampledImage)
        .addDef(SampledImage)
        .addUse(GR->getSPIRVTypeID(SampledImageType))
        .addUse(Image)
        .addUse(Sampler);

    Register Lod = GR->buildConstantFP(APFloat::getZero(APFloat::IEEEsingle()),
                                       MIRBuilder);

    if (Call->ReturnType->getOpcode() != SPIRV::OpTypeVector) {
      SPIRVType *TempType =
          GR->getOrCreateSPIRVVectorType(Call->ReturnType, 4, MIRBuilder, true);
      Register TempRegister =
          MRI->createGenericVirtualRegister(GR->getRegType(TempType));
      MRI->setRegClass(TempRegister, GR->getRegClass(TempType));
      GR->assignSPIRVTypeToVReg(TempType, TempRegister, MIRBuilder.getMF());
      MIRBuilder.buildInstr(SPIRV::OpImageSampleExplicitLod)
          .addDef(TempRegister)
          .addUse(GR->getSPIRVTypeID(TempType))
          .addUse(SampledImage)
          .addUse(Call->Arguments[2]) // Coordinate.
          .addImm(SPIRV::ImageOperand::Lod)
          .addUse(Lod);
      MIRBuilder.buildInstr(SPIRV::OpCompositeExtract)
          .addDef(Call->ReturnRegister)
          .addUse(GR->getSPIRVTypeID(Call->ReturnType))
          .addUse(TempRegister)
          .addImm(0);
    } else {
      MIRBuilder.buildInstr(SPIRV::OpImageSampleExplicitLod)
          .addDef(Call->ReturnRegister)
          .addUse(GR->getSPIRVTypeID(Call->ReturnType))
          .addUse(SampledImage)
          .addUse(Call->Arguments[2]) // Coordinate.
          .addImm(SPIRV::ImageOperand::Lod)
          .addUse(Lod);
    }
  } else if (HasMsaa) {
    MIRBuilder.buildInstr(SPIRV::OpImageRead)
        .addDef(Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(Call->ReturnType))
        .addUse(Image)
        .addUse(Call->Arguments[1]) // Coordinate.
        .addImm(SPIRV::ImageOperand::Sample)
        .addUse(Call->Arguments[2]);
  } else {
    MIRBuilder.buildInstr(SPIRV::OpImageRead)
        .addDef(Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(Call->ReturnType))
        .addUse(Image)
        .addUse(Call->Arguments[1]); // Coordinate.
  }
  return true;
}

static bool generateWriteImageInst(const SPIRV::IncomingCall *Call,
                                   MachineIRBuilder &MIRBuilder,
                                   SPIRVGlobalRegistry *GR) {
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, SPIRV::OpImageWrite, Call,
                              Register(0));
  MIRBuilder.buildInstr(SPIRV::OpImageWrite)
      .addUse(Call->Arguments[0])  // Image.
      .addUse(Call->Arguments[1])  // Coordinate.
      .addUse(Call->Arguments[2]); // Texel.
  return true;
}

static bool generateSampleImageInst(const StringRef DemangledCall,
                                    const SPIRV::IncomingCall *Call,
                                    MachineIRBuilder &MIRBuilder,
                                    SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  if (Call->Builtin->Name.contains_insensitive(
          "__translate_sampler_initializer")) {
    // Build sampler literal.
    uint64_t Bitmask = getIConstVal(Call->Arguments[0], MRI);
    Register Sampler = GR->buildConstantSampler(
        Call->ReturnRegister, getSamplerAddressingModeFromBitmask(Bitmask),
        getSamplerParamFromBitmask(Bitmask),
        getSamplerFilterModeFromBitmask(Bitmask), MIRBuilder);
    return Sampler.isValid();
  } else if (Call->Builtin->Name.contains_insensitive("__spirv_SampledImage")) {
    // Create OpSampledImage.
    Register Image = Call->Arguments[0];
    SPIRVType *ImageType = GR->getSPIRVTypeForVReg(Image);
    SPIRVType *SampledImageType =
        GR->getOrCreateOpTypeSampledImage(ImageType, MIRBuilder);
    Register SampledImage =
        Call->ReturnRegister.isValid()
            ? Call->ReturnRegister
            : MRI->createVirtualRegister(&SPIRV::iIDRegClass);
    MIRBuilder.buildInstr(SPIRV::OpSampledImage)
        .addDef(SampledImage)
        .addUse(GR->getSPIRVTypeID(SampledImageType))
        .addUse(Image)
        .addUse(Call->Arguments[1]); // Sampler.
    return true;
  } else if (Call->Builtin->Name.contains_insensitive(
                 "__spirv_ImageSampleExplicitLod")) {
    // Sample an image using an explicit level of detail.
    std::string ReturnType = DemangledCall.str();
    if (DemangledCall.contains("_R")) {
      ReturnType = ReturnType.substr(ReturnType.find("_R") + 2);
      ReturnType = ReturnType.substr(0, ReturnType.find('('));
    }
    SPIRVType *Type =
        Call->ReturnType
            ? Call->ReturnType
            : GR->getOrCreateSPIRVTypeByName(ReturnType, MIRBuilder, true);
    if (!Type) {
      std::string DiagMsg =
          "Unable to recognize SPIRV type name: " + ReturnType;
      report_fatal_error(DiagMsg.c_str());
    }
    MIRBuilder.buildInstr(SPIRV::OpImageSampleExplicitLod)
        .addDef(Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(Type))
        .addUse(Call->Arguments[0]) // Image.
        .addUse(Call->Arguments[1]) // Coordinate.
        .addImm(SPIRV::ImageOperand::Lod)
        .addUse(Call->Arguments[3]);
    return true;
  }
  return false;
}

static bool generateSelectInst(const SPIRV::IncomingCall *Call,
                               MachineIRBuilder &MIRBuilder) {
  MIRBuilder.buildSelect(Call->ReturnRegister, Call->Arguments[0],
                         Call->Arguments[1], Call->Arguments[2]);
  return true;
}

static bool generateConstructInst(const SPIRV::IncomingCall *Call,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRVGlobalRegistry *GR) {
  createContinuedInstructions(MIRBuilder, SPIRV::OpCompositeConstruct, 3,
                              SPIRV::OpCompositeConstructContinuedINTEL,
                              Call->Arguments, Call->ReturnRegister,
                              GR->getSPIRVTypeID(Call->ReturnType));
  return true;
}

static bool generateCoopMatrInst(const SPIRV::IncomingCall *Call,
                                 MachineIRBuilder &MIRBuilder,
                                 SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;
  bool IsSet = Opcode != SPIRV::OpCooperativeMatrixStoreKHR &&
               Opcode != SPIRV::OpCooperativeMatrixStoreCheckedINTEL &&
               Opcode != SPIRV::OpCooperativeMatrixPrefetchINTEL;
  unsigned ArgSz = Call->Arguments.size();
  unsigned LiteralIdx = 0;
  switch (Opcode) {
  // Memory operand is optional and is literal.
  case SPIRV::OpCooperativeMatrixLoadKHR:
    LiteralIdx = ArgSz > 3 ? 3 : 0;
    break;
  case SPIRV::OpCooperativeMatrixStoreKHR:
    LiteralIdx = ArgSz > 4 ? 4 : 0;
    break;
  case SPIRV::OpCooperativeMatrixLoadCheckedINTEL:
    LiteralIdx = ArgSz > 7 ? 7 : 0;
    break;
  case SPIRV::OpCooperativeMatrixStoreCheckedINTEL:
    LiteralIdx = ArgSz > 8 ? 8 : 0;
    break;
  // Cooperative Matrix Operands operand is optional and is literal.
  case SPIRV::OpCooperativeMatrixMulAddKHR:
    LiteralIdx = ArgSz > 3 ? 3 : 0;
    break;
  };

  SmallVector<uint32_t, 1> ImmArgs;
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  if (Opcode == SPIRV::OpCooperativeMatrixPrefetchINTEL) {
    const uint32_t CacheLevel = getConstFromIntrinsic(Call->Arguments[3], MRI);
    auto MIB = MIRBuilder.buildInstr(SPIRV::OpCooperativeMatrixPrefetchINTEL)
                   .addUse(Call->Arguments[0])  // pointer
                   .addUse(Call->Arguments[1])  // rows
                   .addUse(Call->Arguments[2])  // columns
                   .addImm(CacheLevel)          // cache level
                   .addUse(Call->Arguments[4]); // memory layout
    if (ArgSz > 5)
      MIB.addUse(Call->Arguments[5]); // stride
    if (ArgSz > 6) {
      const uint32_t MemOp = getConstFromIntrinsic(Call->Arguments[6], MRI);
      MIB.addImm(MemOp); // memory operand
    }
    return true;
  }
  if (LiteralIdx > 0)
    ImmArgs.push_back(getConstFromIntrinsic(Call->Arguments[LiteralIdx], MRI));
  Register TypeReg = GR->getSPIRVTypeID(Call->ReturnType);
  if (Opcode == SPIRV::OpCooperativeMatrixLengthKHR) {
    SPIRVType *CoopMatrType = GR->getSPIRVTypeForVReg(Call->Arguments[0]);
    if (!CoopMatrType)
      report_fatal_error("Can't find a register's type definition");
    MIRBuilder.buildInstr(Opcode)
        .addDef(Call->ReturnRegister)
        .addUse(TypeReg)
        .addUse(CoopMatrType->getOperand(0).getReg());
    return true;
  }
  return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                            IsSet ? TypeReg : Register(0), ImmArgs);
}

static bool generateSpecConstantInst(const SPIRV::IncomingCall *Call,
                                     MachineIRBuilder &MIRBuilder,
                                     SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;
  const MachineRegisterInfo *MRI = MIRBuilder.getMRI();

  switch (Opcode) {
  case SPIRV::OpSpecConstant: {
    // Build the SpecID decoration.
    unsigned SpecId =
        static_cast<unsigned>(getIConstVal(Call->Arguments[0], MRI));
    buildOpDecorate(Call->ReturnRegister, MIRBuilder, SPIRV::Decoration::SpecId,
                    {SpecId});
    // Determine the constant MI.
    Register ConstRegister = Call->Arguments[1];
    const MachineInstr *Const = getDefInstrMaybeConstant(ConstRegister, MRI);
    assert(Const &&
           (Const->getOpcode() == TargetOpcode::G_CONSTANT ||
            Const->getOpcode() == TargetOpcode::G_FCONSTANT) &&
           "Argument should be either an int or floating-point constant");
    // Determine the opcode and built the OpSpec MI.
    const MachineOperand &ConstOperand = Const->getOperand(1);
    if (Call->ReturnType->getOpcode() == SPIRV::OpTypeBool) {
      assert(ConstOperand.isCImm() && "Int constant operand is expected");
      Opcode = ConstOperand.getCImm()->getValue().getZExtValue()
                   ? SPIRV::OpSpecConstantTrue
                   : SPIRV::OpSpecConstantFalse;
    }
    auto MIB = MIRBuilder.buildInstr(Opcode)
                   .addDef(Call->ReturnRegister)
                   .addUse(GR->getSPIRVTypeID(Call->ReturnType));

    if (Call->ReturnType->getOpcode() != SPIRV::OpTypeBool) {
      if (Const->getOpcode() == TargetOpcode::G_CONSTANT)
        addNumImm(ConstOperand.getCImm()->getValue(), MIB);
      else
        addNumImm(ConstOperand.getFPImm()->getValueAPF().bitcastToAPInt(), MIB);
    }
    return true;
  }
  case SPIRV::OpSpecConstantComposite: {
    createContinuedInstructions(MIRBuilder, Opcode, 3,
                                SPIRV::OpSpecConstantCompositeContinuedINTEL,
                                Call->Arguments, Call->ReturnRegister,
                                GR->getSPIRVTypeID(Call->ReturnType));
    return true;
  }
  default:
    return false;
  }
}

static bool generateExtendedBitOpsInst(const SPIRV::IncomingCall *Call,
                                       MachineIRBuilder &MIRBuilder,
                                       SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  return buildExtendedBitOpsInst(Call, Opcode, MIRBuilder, GR);
}

static bool generateBindlessImageINTELInst(const SPIRV::IncomingCall *Call,
                                           MachineIRBuilder &MIRBuilder,
                                           SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  return buildBindlessImageINTELInst(Call, Opcode, MIRBuilder, GR);
}

static bool
generateTernaryBitwiseFunctionINTELInst(const SPIRV::IncomingCall *Call,
                                        MachineIRBuilder &MIRBuilder,
                                        SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  return buildTernaryBitwiseFunctionINTELInst(Call, Opcode, MIRBuilder, GR);
}

static bool generate2DBlockIOINTELInst(const SPIRV::IncomingCall *Call,
                                       MachineIRBuilder &MIRBuilder,
                                       SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  return build2DBlockIOINTELInst(Call, Opcode, MIRBuilder, GR);
}

static bool generatePipeInst(const SPIRV::IncomingCall *Call,
                             MachineIRBuilder &MIRBuilder,
                             SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  unsigned Scope = SPIRV::Scope::Workgroup;
  if (Builtin->Name.contains("sub_group"))
    Scope = SPIRV::Scope::Subgroup;

  return buildPipeInst(Call, Opcode, Scope, MIRBuilder, GR);
}

static bool generatePredicatedLoadStoreInst(const SPIRV::IncomingCall *Call,
                                            MachineIRBuilder &MIRBuilder,
                                            SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  bool IsSet = Opcode != SPIRV::OpPredicatedStoreINTEL;
  unsigned ArgSz = Call->Arguments.size();
  SmallVector<uint32_t, 1> ImmArgs;
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  // Memory operand is optional and is literal.
  if (ArgSz > 3)
    ImmArgs.push_back(
        getConstFromIntrinsic(Call->Arguments[/*Literal index*/ 3], MRI));

  Register TypeReg = GR->getSPIRVTypeID(Call->ReturnType);
  return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                            IsSet ? TypeReg : Register(0), ImmArgs);
}

static bool buildNDRange(const SPIRV::IncomingCall *Call,
                         MachineIRBuilder &MIRBuilder,
                         SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  SPIRVType *PtrType = GR->getSPIRVTypeForVReg(Call->Arguments[0]);
  assert(PtrType->getOpcode() == SPIRV::OpTypePointer &&
         PtrType->getOperand(2).isReg());
  Register TypeReg = PtrType->getOperand(2).getReg();
  SPIRVType *StructType = GR->getSPIRVTypeForVReg(TypeReg);
  MachineFunction &MF = MIRBuilder.getMF();
  Register TmpReg = MRI->createVirtualRegister(&SPIRV::iIDRegClass);
  GR->assignSPIRVTypeToVReg(StructType, TmpReg, MF);
  // Skip the first arg, it's the destination pointer. OpBuildNDRange takes
  // three other arguments, so pass zero constant on absence.
  unsigned NumArgs = Call->Arguments.size();
  assert(NumArgs >= 2);
  Register GlobalWorkSize = Call->Arguments[NumArgs < 4 ? 1 : 2];
  Register LocalWorkSize =
      NumArgs == 2 ? Register(0) : Call->Arguments[NumArgs < 4 ? 2 : 3];
  Register GlobalWorkOffset = NumArgs <= 3 ? Register(0) : Call->Arguments[1];
  if (NumArgs < 4) {
    Register Const;
    SPIRVType *SpvTy = GR->getSPIRVTypeForVReg(GlobalWorkSize);
    if (SpvTy->getOpcode() == SPIRV::OpTypePointer) {
      MachineInstr *DefInstr = MRI->getUniqueVRegDef(GlobalWorkSize);
      assert(DefInstr && isSpvIntrinsic(*DefInstr, Intrinsic::spv_gep) &&
             DefInstr->getOperand(3).isReg());
      Register GWSPtr = DefInstr->getOperand(3).getReg();
      // TODO: Maybe simplify generation of the type of the fields.
      unsigned Size = Call->Builtin->Name == "ndrange_3D" ? 3 : 2;
      unsigned BitWidth = GR->getPointerSize() == 64 ? 64 : 32;
      Type *BaseTy = IntegerType::get(MF.getFunction().getContext(), BitWidth);
      Type *FieldTy = ArrayType::get(BaseTy, Size);
      SPIRVType *SpvFieldTy = GR->getOrCreateSPIRVType(
          FieldTy, MIRBuilder, SPIRV::AccessQualifier::ReadWrite, true);
      GlobalWorkSize = MRI->createVirtualRegister(&SPIRV::iIDRegClass);
      GR->assignSPIRVTypeToVReg(SpvFieldTy, GlobalWorkSize, MF);
      MIRBuilder.buildInstr(SPIRV::OpLoad)
          .addDef(GlobalWorkSize)
          .addUse(GR->getSPIRVTypeID(SpvFieldTy))
          .addUse(GWSPtr);
      const SPIRVSubtarget &ST =
          cast<SPIRVSubtarget>(MIRBuilder.getMF().getSubtarget());
      Const = GR->getOrCreateConstIntArray(0, Size, *MIRBuilder.getInsertPt(),
                                           SpvFieldTy, *ST.getInstrInfo());
    } else {
      Const = GR->buildConstantInt(0, MIRBuilder, SpvTy, true);
    }
    if (!LocalWorkSize.isValid())
      LocalWorkSize = Const;
    if (!GlobalWorkOffset.isValid())
      GlobalWorkOffset = Const;
  }
  assert(LocalWorkSize.isValid() && GlobalWorkOffset.isValid());
  MIRBuilder.buildInstr(SPIRV::OpBuildNDRange)
      .addDef(TmpReg)
      .addUse(TypeReg)
      .addUse(GlobalWorkSize)
      .addUse(LocalWorkSize)
      .addUse(GlobalWorkOffset);
  return MIRBuilder.buildInstr(SPIRV::OpStore)
      .addUse(Call->Arguments[0])
      .addUse(TmpReg);
}

// TODO: maybe move to the global register.
static SPIRVType *
getOrCreateSPIRVDeviceEventPointer(MachineIRBuilder &MIRBuilder,
                                   SPIRVGlobalRegistry *GR) {
  LLVMContext &Context = MIRBuilder.getMF().getFunction().getContext();
  unsigned SC1 = storageClassToAddressSpace(SPIRV::StorageClass::Generic);
  Type *PtrType = PointerType::get(Context, SC1);
  return GR->getOrCreateSPIRVType(PtrType, MIRBuilder,
                                  SPIRV::AccessQualifier::ReadWrite, true);
}

static bool buildEnqueueKernel(const SPIRV::IncomingCall *Call,
                               MachineIRBuilder &MIRBuilder,
                               SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  const DataLayout &DL = MIRBuilder.getDataLayout();
  bool IsSpirvOp = Call->isSpirvOp();
  bool HasEvents = Call->Builtin->Name.contains("events") || IsSpirvOp;
  const SPIRVType *Int32Ty = GR->getOrCreateSPIRVIntegerType(32, MIRBuilder);

  // Make vararg instructions before OpEnqueueKernel.
  // Local sizes arguments: Sizes of block invoke arguments. Clang generates
  // local size operands as an array, so we need to unpack them.
  SmallVector<Register, 16> LocalSizes;
  if (Call->Builtin->Name.contains("_varargs") || IsSpirvOp) {
    const unsigned LocalSizeArrayIdx = HasEvents ? 9 : 6;
    Register GepReg = Call->Arguments[LocalSizeArrayIdx];
    MachineInstr *GepMI = MRI->getUniqueVRegDef(GepReg);
    assert(isSpvIntrinsic(*GepMI, Intrinsic::spv_gep) &&
           GepMI->getOperand(3).isReg());
    Register ArrayReg = GepMI->getOperand(3).getReg();
    MachineInstr *ArrayMI = MRI->getUniqueVRegDef(ArrayReg);
    const Type *LocalSizeTy = getMachineInstrType(ArrayMI);
    assert(LocalSizeTy && "Local size type is expected");
    const uint64_t LocalSizeNum =
        cast<ArrayType>(LocalSizeTy)->getNumElements();
    unsigned SC = storageClassToAddressSpace(SPIRV::StorageClass::Generic);
    const LLT LLType = LLT::pointer(SC, GR->getPointerSize());
    const SPIRVType *PointerSizeTy = GR->getOrCreateSPIRVPointerType(
        Int32Ty, MIRBuilder, SPIRV::StorageClass::Function);
    for (unsigned I = 0; I < LocalSizeNum; ++I) {
      Register Reg = MRI->createVirtualRegister(&SPIRV::pIDRegClass);
      MRI->setType(Reg, LLType);
      GR->assignSPIRVTypeToVReg(PointerSizeTy, Reg, MIRBuilder.getMF());
      auto GEPInst = MIRBuilder.buildIntrinsic(
          Intrinsic::spv_gep, ArrayRef<Register>{Reg}, true, false);
      GEPInst
          .addImm(GepMI->getOperand(2).getImm())            // In bound.
          .addUse(ArrayMI->getOperand(0).getReg())          // Alloca.
          .addUse(buildConstantIntReg32(0, MIRBuilder, GR)) // Indices.
          .addUse(buildConstantIntReg32(I, MIRBuilder, GR));
      LocalSizes.push_back(Reg);
    }
  }

  // SPIRV OpEnqueueKernel instruction has 10+ arguments.
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpEnqueueKernel)
                 .addDef(Call->ReturnRegister)
                 .addUse(GR->getSPIRVTypeID(Int32Ty));

  // Copy all arguments before block invoke function pointer.
  const unsigned BlockFIdx = HasEvents ? 6 : 3;
  for (unsigned i = 0; i < BlockFIdx; i++)
    MIB.addUse(Call->Arguments[i]);

  // If there are no event arguments in the original call, add dummy ones.
  if (!HasEvents) {
    MIB.addUse(buildConstantIntReg32(0, MIRBuilder, GR)); // Dummy num events.
    Register NullPtr = GR->getOrCreateConstNullPtr(
        MIRBuilder, getOrCreateSPIRVDeviceEventPointer(MIRBuilder, GR));
    MIB.addUse(NullPtr); // Dummy wait events.
    MIB.addUse(NullPtr); // Dummy ret event.
  }

  MachineInstr *BlockMI = getBlockStructInstr(Call->Arguments[BlockFIdx], MRI);
  assert(BlockMI->getOpcode() == TargetOpcode::G_GLOBAL_VALUE);
  // Invoke: Pointer to invoke function.
  MIB.addGlobalAddress(BlockMI->getOperand(1).getGlobal());

  Register BlockLiteralReg = Call->Arguments[BlockFIdx + 1];
  // Param: Pointer to block literal.
  MIB.addUse(BlockLiteralReg);

  Type *PType = const_cast<Type *>(getBlockStructType(BlockLiteralReg, MRI));
  // TODO: these numbers should be obtained from block literal structure.
  // Param Size: Size of block literal structure.
  MIB.addUse(buildConstantIntReg32(DL.getTypeStoreSize(PType), MIRBuilder, GR));
  // Param Aligment: Aligment of block literal structure.
  MIB.addUse(buildConstantIntReg32(DL.getPrefTypeAlign(PType).value(),
                                   MIRBuilder, GR));

  for (unsigned i = 0; i < LocalSizes.size(); i++)
    MIB.addUse(LocalSizes[i]);
  return true;
}

static bool generateEnqueueInst(const SPIRV::IncomingCall *Call,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  switch (Opcode) {
  case SPIRV::OpRetainEvent:
  case SPIRV::OpReleaseEvent:
    return MIRBuilder.buildInstr(Opcode).addUse(Call->Arguments[0]);
  case SPIRV::OpCreateUserEvent:
  case SPIRV::OpGetDefaultQueue:
    return MIRBuilder.buildInstr(Opcode)
        .addDef(Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(Call->ReturnType));
  case SPIRV::OpIsValidEvent:
    return MIRBuilder.buildInstr(Opcode)
        .addDef(Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(Call->ReturnType))
        .addUse(Call->Arguments[0]);
  case SPIRV::OpSetUserEventStatus:
    return MIRBuilder.buildInstr(Opcode)
        .addUse(Call->Arguments[0])
        .addUse(Call->Arguments[1]);
  case SPIRV::OpCaptureEventProfilingInfo:
    return MIRBuilder.buildInstr(Opcode)
        .addUse(Call->Arguments[0])
        .addUse(Call->Arguments[1])
        .addUse(Call->Arguments[2]);
  case SPIRV::OpBuildNDRange:
    return buildNDRange(Call, MIRBuilder, GR);
  case SPIRV::OpEnqueueKernel:
    return buildEnqueueKernel(Call, MIRBuilder, GR);
  default:
    return false;
  }
}

static bool generateAsyncCopy(const SPIRV::IncomingCall *Call,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  bool IsSet = Opcode == SPIRV::OpGroupAsyncCopy;
  Register TypeReg = GR->getSPIRVTypeID(Call->ReturnType);
  if (Call->isSpirvOp())
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              IsSet ? TypeReg : Register(0));

  auto Scope = buildConstantIntReg32(SPIRV::Scope::Workgroup, MIRBuilder, GR);

  switch (Opcode) {
  case SPIRV::OpGroupAsyncCopy: {
    SPIRVType *NewType =
        Call->ReturnType->getOpcode() == SPIRV::OpTypeEvent
            ? nullptr
            : GR->getOrCreateSPIRVTypeByName("spirv.Event", MIRBuilder, true);
    Register TypeReg = GR->getSPIRVTypeID(NewType ? NewType : Call->ReturnType);
    unsigned NumArgs = Call->Arguments.size();
    Register EventReg = Call->Arguments[NumArgs - 1];
    bool Res = MIRBuilder.buildInstr(Opcode)
                   .addDef(Call->ReturnRegister)
                   .addUse(TypeReg)
                   .addUse(Scope)
                   .addUse(Call->Arguments[0])
                   .addUse(Call->Arguments[1])
                   .addUse(Call->Arguments[2])
                   .addUse(Call->Arguments.size() > 4
                               ? Call->Arguments[3]
                               : buildConstantIntReg32(1, MIRBuilder, GR))
                   .addUse(EventReg);
    if (NewType != nullptr)
      insertAssignInstr(Call->ReturnRegister, nullptr, NewType, GR, MIRBuilder,
                        MIRBuilder.getMF().getRegInfo());
    return Res;
  }
  case SPIRV::OpGroupWaitEvents:
    return MIRBuilder.buildInstr(Opcode)
        .addUse(Scope)
        .addUse(Call->Arguments[0])
        .addUse(Call->Arguments[1]);
  default:
    return false;
  }
}

static bool generateConvertInst(const StringRef DemangledCall,
                                const SPIRV::IncomingCall *Call,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  // Lookup the conversion builtin in the TableGen records.
  const SPIRV::ConvertBuiltin *Builtin =
      SPIRV::lookupConvertBuiltin(Call->Builtin->Name, Call->Builtin->Set);

  if (!Builtin && Call->isSpirvOp()) {
    const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
    unsigned Opcode =
        SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;
    return buildOpFromWrapper(MIRBuilder, Opcode, Call,
                              GR->getSPIRVTypeID(Call->ReturnType));
  }

  assert(Builtin && "Conversion builtin not found.");
  if (Builtin->IsSaturated)
    buildOpDecorate(Call->ReturnRegister, MIRBuilder,
                    SPIRV::Decoration::SaturatedConversion, {});
  if (Builtin->IsRounded)
    buildOpDecorate(Call->ReturnRegister, MIRBuilder,
                    SPIRV::Decoration::FPRoundingMode,
                    {(unsigned)Builtin->RoundingMode});

  std::string NeedExtMsg;              // no errors if empty
  bool IsRightComponentsNumber = true; // check if input/output accepts vectors
  unsigned Opcode = SPIRV::OpNop;
  if (GR->isScalarOrVectorOfType(Call->Arguments[0], SPIRV::OpTypeInt)) {
    // Int -> ...
    if (GR->isScalarOrVectorOfType(Call->ReturnRegister, SPIRV::OpTypeInt)) {
      // Int -> Int
      if (Builtin->IsSaturated)
        Opcode = Builtin->IsDestinationSigned ? SPIRV::OpSatConvertUToS
                                              : SPIRV::OpSatConvertSToU;
      else
        Opcode = Builtin->IsDestinationSigned ? SPIRV::OpUConvert
                                              : SPIRV::OpSConvert;
    } else if (GR->isScalarOrVectorOfType(Call->ReturnRegister,
                                          SPIRV::OpTypeFloat)) {
      // Int -> Float
      if (Builtin->IsBfloat16) {
        const auto *ST = static_cast<const SPIRVSubtarget *>(
            &MIRBuilder.getMF().getSubtarget());
        if (!ST->canUseExtension(
                SPIRV::Extension::SPV_INTEL_bfloat16_conversion))
          NeedExtMsg = "SPV_INTEL_bfloat16_conversion";
        IsRightComponentsNumber =
            GR->getScalarOrVectorComponentCount(Call->Arguments[0]) ==
            GR->getScalarOrVectorComponentCount(Call->ReturnRegister);
        Opcode = SPIRV::OpConvertBF16ToFINTEL;
      } else {
        bool IsSourceSigned =
            DemangledCall[DemangledCall.find_first_of('(') + 1] != 'u';
        Opcode = IsSourceSigned ? SPIRV::OpConvertSToF : SPIRV::OpConvertUToF;
      }
    }
  } else if (GR->isScalarOrVectorOfType(Call->Arguments[0],
                                        SPIRV::OpTypeFloat)) {
    // Float -> ...
    if (GR->isScalarOrVectorOfType(Call->ReturnRegister, SPIRV::OpTypeInt)) {
      // Float -> Int
      if (Builtin->IsBfloat16) {
        const auto *ST = static_cast<const SPIRVSubtarget *>(
            &MIRBuilder.getMF().getSubtarget());
        if (!ST->canUseExtension(
                SPIRV::Extension::SPV_INTEL_bfloat16_conversion))
          NeedExtMsg = "SPV_INTEL_bfloat16_conversion";
        IsRightComponentsNumber =
            GR->getScalarOrVectorComponentCount(Call->Arguments[0]) ==
            GR->getScalarOrVectorComponentCount(Call->ReturnRegister);
        Opcode = SPIRV::OpConvertFToBF16INTEL;
      } else {
        Opcode = Builtin->IsDestinationSigned ? SPIRV::OpConvertFToS
                                              : SPIRV::OpConvertFToU;
      }
    } else if (GR->isScalarOrVectorOfType(Call->ReturnRegister,
                                          SPIRV::OpTypeFloat)) {
      if (Builtin->IsTF32) {
        const auto *ST = static_cast<const SPIRVSubtarget *>(
            &MIRBuilder.getMF().getSubtarget());
        if (!ST->canUseExtension(
                SPIRV::Extension::SPV_INTEL_tensor_float32_conversion))
          NeedExtMsg = "SPV_INTEL_tensor_float32_conversion";
        IsRightComponentsNumber =
            GR->getScalarOrVectorComponentCount(Call->Arguments[0]) ==
            GR->getScalarOrVectorComponentCount(Call->ReturnRegister);
        Opcode = SPIRV::OpRoundFToTF32INTEL;
      } else {
        // Float -> Float
        Opcode = SPIRV::OpFConvert;
      }
    }
  }

  if (!NeedExtMsg.empty()) {
    std::string DiagMsg = std::string(Builtin->Name) +
                          ": the builtin requires the following SPIR-V "
                          "extension: " +
                          NeedExtMsg;
    report_fatal_error(DiagMsg.c_str(), false);
  }
  if (!IsRightComponentsNumber) {
    std::string DiagMsg =
        std::string(Builtin->Name) +
        ": result and argument must have the same number of components";
    report_fatal_error(DiagMsg.c_str(), false);
  }
  assert(Opcode != SPIRV::OpNop &&
         "Conversion between the types not implemented!");

  MIRBuilder.buildInstr(Opcode)
      .addDef(Call->ReturnRegister)
      .addUse(GR->getSPIRVTypeID(Call->ReturnType))
      .addUse(Call->Arguments[0]);
  return true;
}

static bool generateVectorLoadStoreInst(const SPIRV::IncomingCall *Call,
                                        MachineIRBuilder &MIRBuilder,
                                        SPIRVGlobalRegistry *GR) {
  // Lookup the vector load/store builtin in the TableGen records.
  const SPIRV::VectorLoadStoreBuiltin *Builtin =
      SPIRV::lookupVectorLoadStoreBuiltin(Call->Builtin->Name,
                                          Call->Builtin->Set);
  // Build extended instruction.
  auto MIB =
      MIRBuilder.buildInstr(SPIRV::OpExtInst)
          .addDef(Call->ReturnRegister)
          .addUse(GR->getSPIRVTypeID(Call->ReturnType))
          .addImm(static_cast<uint32_t>(SPIRV::InstructionSet::OpenCL_std))
          .addImm(Builtin->Number);
  for (auto Argument : Call->Arguments)
    MIB.addUse(Argument);
  if (Builtin->Name.contains("load") && Builtin->ElementCount > 1)
    MIB.addImm(Builtin->ElementCount);

  // Rounding mode should be passed as a last argument in the MI for builtins
  // like "vstorea_halfn_r".
  if (Builtin->IsRounded)
    MIB.addImm(static_cast<uint32_t>(Builtin->RoundingMode));
  return true;
}

static bool generateLoadStoreInst(const SPIRV::IncomingCall *Call,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;
  bool IsLoad = Opcode == SPIRV::OpLoad;
  // Build the instruction.
  auto MIB = MIRBuilder.buildInstr(Opcode);
  if (IsLoad) {
    MIB.addDef(Call->ReturnRegister);
    MIB.addUse(GR->getSPIRVTypeID(Call->ReturnType));
  }
  // Add a pointer to the value to load/store.
  MIB.addUse(Call->Arguments[0]);
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  // Add a value to store.
  if (!IsLoad)
    MIB.addUse(Call->Arguments[1]);
  // Add optional memory attributes and an alignment.
  unsigned NumArgs = Call->Arguments.size();
  if ((IsLoad && NumArgs >= 2) || NumArgs >= 3)
    MIB.addImm(getConstFromIntrinsic(Call->Arguments[IsLoad ? 1 : 2], MRI));
  if ((IsLoad && NumArgs >= 3) || NumArgs >= 4)
    MIB.addImm(getConstFromIntrinsic(Call->Arguments[IsLoad ? 2 : 3], MRI));
  return true;
}

namespace SPIRV {
// Try to find a builtin function attributes by a demangled function name and
// return a tuple <builtin group, op code, ext instruction number>, or a special
// tuple value <-1, 0, 0> if the builtin function is not found.
// Not all builtin functions are supported, only those with a ready-to-use op
// code or instruction number defined in TableGen.
// TODO: consider a major rework of mapping demangled calls into a builtin
// functions to unify search and decrease number of individual cases.
std::tuple<int, unsigned, unsigned>
mapBuiltinToOpcode(const StringRef DemangledCall,
                   SPIRV::InstructionSet::InstructionSet Set) {
  Register Reg;
  SmallVector<Register> Args;
  std::unique_ptr<const IncomingCall> Call =
      lookupBuiltin(DemangledCall, Set, Reg, nullptr, Args);
  if (!Call)
    return std::make_tuple(-1, 0, 0);

  switch (Call->Builtin->Group) {
  case SPIRV::Relational:
  case SPIRV::Atomic:
  case SPIRV::Barrier:
  case SPIRV::CastToPtr:
  case SPIRV::ImageMiscQuery:
  case SPIRV::SpecConstant:
  case SPIRV::Enqueue:
  case SPIRV::AsyncCopy:
  case SPIRV::LoadStore:
  case SPIRV::CoopMatr:
    if (const auto *R =
            SPIRV::lookupNativeBuiltin(Call->Builtin->Name, Call->Builtin->Set))
      return std::make_tuple(Call->Builtin->Group, R->Opcode, 0);
    break;
  case SPIRV::Extended:
    if (const auto *R = SPIRV::lookupExtendedBuiltin(Call->Builtin->Name,
                                                     Call->Builtin->Set))
      return std::make_tuple(Call->Builtin->Group, 0, R->Number);
    break;
  case SPIRV::VectorLoadStore:
    if (const auto *R = SPIRV::lookupVectorLoadStoreBuiltin(Call->Builtin->Name,
                                                            Call->Builtin->Set))
      return std::make_tuple(SPIRV::Extended, 0, R->Number);
    break;
  case SPIRV::Group:
    if (const auto *R = SPIRV::lookupGroupBuiltin(Call->Builtin->Name))
      return std::make_tuple(Call->Builtin->Group, R->Opcode, 0);
    break;
  case SPIRV::AtomicFloating:
    if (const auto *R = SPIRV::lookupAtomicFloatingBuiltin(Call->Builtin->Name))
      return std::make_tuple(Call->Builtin->Group, R->Opcode, 0);
    break;
  case SPIRV::IntelSubgroups:
    if (const auto *R = SPIRV::lookupIntelSubgroupsBuiltin(Call->Builtin->Name))
      return std::make_tuple(Call->Builtin->Group, R->Opcode, 0);
    break;
  case SPIRV::GroupUniform:
    if (const auto *R = SPIRV::lookupGroupUniformBuiltin(Call->Builtin->Name))
      return std::make_tuple(Call->Builtin->Group, R->Opcode, 0);
    break;
  case SPIRV::IntegerDot:
    if (const auto *R =
            SPIRV::lookupIntegerDotProductBuiltin(Call->Builtin->Name))
      return std::make_tuple(Call->Builtin->Group, R->Opcode, 0);
    break;
  case SPIRV::WriteImage:
    return std::make_tuple(Call->Builtin->Group, SPIRV::OpImageWrite, 0);
  case SPIRV::Select:
    return std::make_tuple(Call->Builtin->Group, TargetOpcode::G_SELECT, 0);
  case SPIRV::Construct:
    return std::make_tuple(Call->Builtin->Group, SPIRV::OpCompositeConstruct,
                           0);
  case SPIRV::KernelClock:
    return std::make_tuple(Call->Builtin->Group, SPIRV::OpReadClockKHR, 0);
  default:
    return std::make_tuple(-1, 0, 0);
  }
  return std::make_tuple(-1, 0, 0);
}

std::optional<bool> lowerBuiltin(const StringRef DemangledCall,
                                 SPIRV::InstructionSet::InstructionSet Set,
                                 MachineIRBuilder &MIRBuilder,
                                 const Register OrigRet, const Type *OrigRetTy,
                                 const SmallVectorImpl<Register> &Args,
                                 SPIRVGlobalRegistry *GR, const CallBase &CB) {
  LLVM_DEBUG(dbgs() << "Lowering builtin call: " << DemangledCall << "\n");

  // Lookup the builtin in the TableGen records.
  SPIRVType *SpvType = GR->getSPIRVTypeForVReg(OrigRet);
  assert(SpvType && "Inconsistent return register: expected valid type info");
  std::unique_ptr<const IncomingCall> Call =
      lookupBuiltin(DemangledCall, Set, OrigRet, SpvType, Args);

  if (!Call) {
    LLVM_DEBUG(dbgs() << "Builtin record was not found!\n");
    return std::nullopt;
  }

  // TODO: check if the provided args meet the builtin requirments.
  assert(Args.size() >= Call->Builtin->MinNumArgs &&
         "Too few arguments to generate the builtin");
  if (Call->Builtin->MaxNumArgs && Args.size() > Call->Builtin->MaxNumArgs)
    LLVM_DEBUG(dbgs() << "More arguments provided than required!\n");

  // Match the builtin with implementation based on the grouping.
  switch (Call->Builtin->Group) {
  case SPIRV::Extended:
    return generateExtInst(Call.get(), MIRBuilder, GR, CB);
  case SPIRV::Relational:
    return generateRelationalInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Group:
    return generateGroupInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Variable:
    return generateBuiltinVar(Call.get(), MIRBuilder, GR);
  case SPIRV::Atomic:
    return generateAtomicInst(Call.get(), MIRBuilder, GR);
  case SPIRV::AtomicFloating:
    return generateAtomicFloatingInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Barrier:
    return generateBarrierInst(Call.get(), MIRBuilder, GR);
  case SPIRV::CastToPtr:
    return generateCastToPtrInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Dot:
  case SPIRV::IntegerDot:
    return generateDotOrFMulInst(DemangledCall, Call.get(), MIRBuilder, GR);
  case SPIRV::Wave:
    return generateWaveInst(Call.get(), MIRBuilder, GR);
  case SPIRV::ICarryBorrow:
    return generateICarryBorrowInst(Call.get(), MIRBuilder, GR);
  case SPIRV::GetQuery:
    return generateGetQueryInst(Call.get(), MIRBuilder, GR);
  case SPIRV::ImageSizeQuery:
    return generateImageSizeQueryInst(Call.get(), MIRBuilder, GR);
  case SPIRV::ImageMiscQuery:
    return generateImageMiscQueryInst(Call.get(), MIRBuilder, GR);
  case SPIRV::ReadImage:
    return generateReadImageInst(DemangledCall, Call.get(), MIRBuilder, GR);
  case SPIRV::WriteImage:
    return generateWriteImageInst(Call.get(), MIRBuilder, GR);
  case SPIRV::SampleImage:
    return generateSampleImageInst(DemangledCall, Call.get(), MIRBuilder, GR);
  case SPIRV::Select:
    return generateSelectInst(Call.get(), MIRBuilder);
  case SPIRV::Construct:
    return generateConstructInst(Call.get(), MIRBuilder, GR);
  case SPIRV::SpecConstant:
    return generateSpecConstantInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Enqueue:
    return generateEnqueueInst(Call.get(), MIRBuilder, GR);
  case SPIRV::AsyncCopy:
    return generateAsyncCopy(Call.get(), MIRBuilder, GR);
  case SPIRV::Convert:
    return generateConvertInst(DemangledCall, Call.get(), MIRBuilder, GR);
  case SPIRV::VectorLoadStore:
    return generateVectorLoadStoreInst(Call.get(), MIRBuilder, GR);
  case SPIRV::LoadStore:
    return generateLoadStoreInst(Call.get(), MIRBuilder, GR);
  case SPIRV::IntelSubgroups:
    return generateIntelSubgroupsInst(Call.get(), MIRBuilder, GR);
  case SPIRV::GroupUniform:
    return generateGroupUniformInst(Call.get(), MIRBuilder, GR);
  case SPIRV::KernelClock:
    return generateKernelClockInst(Call.get(), MIRBuilder, GR);
  case SPIRV::CoopMatr:
    return generateCoopMatrInst(Call.get(), MIRBuilder, GR);
  case SPIRV::ExtendedBitOps:
    return generateExtendedBitOpsInst(Call.get(), MIRBuilder, GR);
  case SPIRV::BindlessINTEL:
    return generateBindlessImageINTELInst(Call.get(), MIRBuilder, GR);
  case SPIRV::TernaryBitwiseINTEL:
    return generateTernaryBitwiseFunctionINTELInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Block2DLoadStore:
    return generate2DBlockIOINTELInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Pipe:
    return generatePipeInst(Call.get(), MIRBuilder, GR);
  case SPIRV::PredicatedLoadStore:
    return generatePredicatedLoadStoreInst(Call.get(), MIRBuilder, GR);
  }
  return false;
}

Type *parseBuiltinCallArgumentType(StringRef TypeStr, LLVMContext &Ctx) {
  // Parse strings representing OpenCL builtin types.
  if (hasBuiltinTypePrefix(TypeStr)) {
    // OpenCL builtin types in demangled call strings have the following format:
    // e.g. ocl_image2d_ro
    [[maybe_unused]] bool IsOCLBuiltinType = TypeStr.consume_front("ocl_");
    assert(IsOCLBuiltinType && "Invalid OpenCL builtin prefix");

    // Check if this is pointer to a builtin type and not just pointer
    // representing a builtin type. In case it is a pointer to builtin type,
    // this will require additional handling in the method calling
    // parseBuiltinCallArgumentBaseType(...) as this function only retrieves the
    // base types.
    if (TypeStr.ends_with("*"))
      TypeStr = TypeStr.slice(0, TypeStr.find_first_of(" *"));

    return parseBuiltinTypeNameToTargetExtType("opencl." + TypeStr.str() + "_t",
                                               Ctx);
  }

  // Parse type name in either "typeN" or "type vector[N]" format, where
  // N is the number of elements of the vector.
  Type *BaseType;
  unsigned VecElts = 0;

  BaseType = parseBasicTypeName(TypeStr, Ctx);
  if (!BaseType)
    // Unable to recognize SPIRV type name.
    return nullptr;

  // Handle "typeN*" or "type vector[N]*".
  TypeStr.consume_back("*");

  if (TypeStr.consume_front(" vector["))
    TypeStr = TypeStr.substr(0, TypeStr.find(']'));

  TypeStr.getAsInteger(10, VecElts);
  if (VecElts > 0)
    BaseType = VectorType::get(
        BaseType->isVoidTy() ? Type::getInt8Ty(Ctx) : BaseType, VecElts, false);

  return BaseType;
}

bool parseBuiltinTypeStr(SmallVector<StringRef, 10> &BuiltinArgsTypeStrs,
                         const StringRef DemangledCall, LLVMContext &Ctx) {
  auto Pos1 = DemangledCall.find('(');
  if (Pos1 == StringRef::npos)
    return false;
  auto Pos2 = DemangledCall.find(')');
  if (Pos2 == StringRef::npos || Pos1 > Pos2)
    return false;
  DemangledCall.slice(Pos1 + 1, Pos2)
      .split(BuiltinArgsTypeStrs, ',', -1, false);
  return true;
}

Type *parseBuiltinCallArgumentBaseType(const StringRef DemangledCall,
                                       unsigned ArgIdx, LLVMContext &Ctx) {
  SmallVector<StringRef, 10> BuiltinArgsTypeStrs;
  parseBuiltinTypeStr(BuiltinArgsTypeStrs, DemangledCall, Ctx);
  if (ArgIdx >= BuiltinArgsTypeStrs.size())
    return nullptr;
  StringRef TypeStr = BuiltinArgsTypeStrs[ArgIdx].trim();
  return parseBuiltinCallArgumentType(TypeStr, Ctx);
}

struct BuiltinType {
  StringRef Name;
  uint32_t Opcode;
};

#define GET_BuiltinTypes_DECL
#define GET_BuiltinTypes_IMPL

struct OpenCLType {
  StringRef Name;
  StringRef SpirvTypeLiteral;
};

#define GET_OpenCLTypes_DECL
#define GET_OpenCLTypes_IMPL

#include "SPIRVGenTables.inc"
} // namespace SPIRV

//===----------------------------------------------------------------------===//
// Misc functions for parsing builtin types.
//===----------------------------------------------------------------------===//

static Type *parseTypeString(const StringRef Name, LLVMContext &Context) {
  if (Name.starts_with("void"))
    return Type::getVoidTy(Context);
  else if (Name.starts_with("int") || Name.starts_with("uint"))
    return Type::getInt32Ty(Context);
  else if (Name.starts_with("float"))
    return Type::getFloatTy(Context);
  else if (Name.starts_with("half"))
    return Type::getHalfTy(Context);
  report_fatal_error("Unable to recognize type!");
}

//===----------------------------------------------------------------------===//
// Implementation functions for builtin types.
//===----------------------------------------------------------------------===//

static SPIRVType *getNonParameterizedType(const TargetExtType *ExtensionType,
                                          const SPIRV::BuiltinType *TypeRecord,
                                          MachineIRBuilder &MIRBuilder,
                                          SPIRVGlobalRegistry *GR) {
  unsigned Opcode = TypeRecord->Opcode;
  // Create or get an existing type from GlobalRegistry.
  return GR->getOrCreateOpTypeByOpcode(ExtensionType, MIRBuilder, Opcode);
}

static SPIRVType *getSamplerType(MachineIRBuilder &MIRBuilder,
                                 SPIRVGlobalRegistry *GR) {
  // Create or get an existing type from GlobalRegistry.
  return GR->getOrCreateOpTypeSampler(MIRBuilder);
}

static SPIRVType *getPipeType(const TargetExtType *ExtensionType,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR) {
  assert(ExtensionType->getNumIntParameters() == 1 &&
         "Invalid number of parameters for SPIR-V pipe builtin!");
  // Create or get an existing type from GlobalRegistry.
  return GR->getOrCreateOpTypePipe(MIRBuilder,
                                   SPIRV::AccessQualifier::AccessQualifier(
                                       ExtensionType->getIntParameter(0)));
}

static SPIRVType *getCoopMatrType(const TargetExtType *ExtensionType,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRVGlobalRegistry *GR) {
  assert(ExtensionType->getNumIntParameters() == 4 &&
         "Invalid number of parameters for SPIR-V coop matrices builtin!");
  assert(ExtensionType->getNumTypeParameters() == 1 &&
         "SPIR-V coop matrices builtin type must have a type parameter!");
  const SPIRVType *ElemType =
      GR->getOrCreateSPIRVType(ExtensionType->getTypeParameter(0), MIRBuilder,
                               SPIRV::AccessQualifier::ReadWrite, true);
  // Create or get an existing type from GlobalRegistry.
  return GR->getOrCreateOpTypeCoopMatr(
      MIRBuilder, ExtensionType, ElemType, ExtensionType->getIntParameter(0),
      ExtensionType->getIntParameter(1), ExtensionType->getIntParameter(2),
      ExtensionType->getIntParameter(3), true);
}

static SPIRVType *getSampledImageType(const TargetExtType *OpaqueType,
                                      MachineIRBuilder &MIRBuilder,
                                      SPIRVGlobalRegistry *GR) {
  SPIRVType *OpaqueImageType = GR->getImageType(
      OpaqueType, SPIRV::AccessQualifier::ReadOnly, MIRBuilder);
  // Create or get an existing type from GlobalRegistry.
  return GR->getOrCreateOpTypeSampledImage(OpaqueImageType, MIRBuilder);
}

static SPIRVType *getInlineSpirvType(const TargetExtType *ExtensionType,
                                     MachineIRBuilder &MIRBuilder,
                                     SPIRVGlobalRegistry *GR) {
  assert(ExtensionType->getNumIntParameters() == 3 &&
         "Inline SPIR-V type builtin takes an opcode, size, and alignment "
         "parameter");
  auto Opcode = ExtensionType->getIntParameter(0);

  SmallVector<MCOperand> Operands;
  for (Type *Param : ExtensionType->type_params()) {
    if (const TargetExtType *ParamEType = dyn_cast<TargetExtType>(Param)) {
      if (ParamEType->getName() == "spirv.IntegralConstant") {
        assert(ParamEType->getNumTypeParameters() == 1 &&
               "Inline SPIR-V integral constant builtin must have a type "
               "parameter");
        assert(ParamEType->getNumIntParameters() == 1 &&
               "Inline SPIR-V integral constant builtin must have a "
               "value parameter");

        auto OperandValue = ParamEType->getIntParameter(0);
        auto *OperandType = ParamEType->getTypeParameter(0);

        const SPIRVType *OperandSPIRVType = GR->getOrCreateSPIRVType(
            OperandType, MIRBuilder, SPIRV::AccessQualifier::ReadWrite, true);

        Operands.push_back(MCOperand::createReg(GR->buildConstantInt(
            OperandValue, MIRBuilder, OperandSPIRVType, true)));
        continue;
      } else if (ParamEType->getName() == "spirv.Literal") {
        assert(ParamEType->getNumTypeParameters() == 0 &&
               "Inline SPIR-V literal builtin does not take type "
               "parameters");
        assert(ParamEType->getNumIntParameters() == 1 &&
               "Inline SPIR-V literal builtin must have an integer "
               "parameter");

        auto OperandValue = ParamEType->getIntParameter(0);

        Operands.push_back(MCOperand::createImm(OperandValue));
        continue;
      }
    }
    const SPIRVType *TypeOperand = GR->getOrCreateSPIRVType(
        Param, MIRBuilder, SPIRV::AccessQualifier::ReadWrite, true);
    Operands.push_back(MCOperand::createReg(GR->getSPIRVTypeID(TypeOperand)));
  }

  return GR->getOrCreateUnknownType(ExtensionType, MIRBuilder, Opcode,
                                    Operands);
}

static SPIRVType *getVulkanBufferType(const TargetExtType *ExtensionType,
                                      MachineIRBuilder &MIRBuilder,
                                      SPIRVGlobalRegistry *GR) {
  assert(ExtensionType->getNumTypeParameters() == 1 &&
         "Vulkan buffers have exactly one type for the type of the buffer.");
  assert(ExtensionType->getNumIntParameters() == 2 &&
         "Vulkan buffer have 2 integer parameters: storage class and is "
         "writable.");

  auto *T = ExtensionType->getTypeParameter(0);
  auto SC = static_cast<SPIRV::StorageClass::StorageClass>(
      ExtensionType->getIntParameter(0));
  bool IsWritable = ExtensionType->getIntParameter(1);
  return GR->getOrCreateVulkanBufferType(MIRBuilder, T, SC, IsWritable);
}

static SPIRVType *getLayoutType(const TargetExtType *ExtensionType,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  return GR->getOrCreateLayoutType(MIRBuilder, ExtensionType);
}

namespace SPIRV {
TargetExtType *parseBuiltinTypeNameToTargetExtType(std::string TypeName,
                                                   LLVMContext &Context) {
  StringRef NameWithParameters = TypeName;

  // Pointers-to-opaque-structs representing OpenCL types are first translated
  // to equivalent SPIR-V types. OpenCL builtin type names should have the
  // following format: e.g. %opencl.event_t
  if (NameWithParameters.starts_with("opencl.")) {
    const SPIRV::OpenCLType *OCLTypeRecord =
        SPIRV::lookupOpenCLType(NameWithParameters);
    if (!OCLTypeRecord)
      report_fatal_error("Missing TableGen record for OpenCL type: " +
                         NameWithParameters);
    NameWithParameters = OCLTypeRecord->SpirvTypeLiteral;
    // Continue with the SPIR-V builtin type...
  }

  // Names of the opaque structs representing a SPIR-V builtins without
  // parameters should have the following format: e.g. %spirv.Event
  assert(NameWithParameters.starts_with("spirv.") &&
         "Unknown builtin opaque type!");

  // Parameterized SPIR-V builtins names follow this format:
  // e.g. %spirv.Image._void_1_0_0_0_0_0_0, %spirv.Pipe._0
  if (!NameWithParameters.contains('_'))
    return TargetExtType::get(Context, NameWithParameters);

  SmallVector<StringRef> Parameters;
  unsigned BaseNameLength = NameWithParameters.find('_') - 1;
  SplitString(NameWithParameters.substr(BaseNameLength + 1), Parameters, "_");

  SmallVector<Type *, 1> TypeParameters;
  bool HasTypeParameter = !isDigit(Parameters[0][0]);
  if (HasTypeParameter)
    TypeParameters.push_back(parseTypeString(Parameters[0], Context));
  SmallVector<unsigned> IntParameters;
  for (unsigned i = HasTypeParameter ? 1 : 0; i < Parameters.size(); i++) {
    unsigned IntParameter = 0;
    bool ValidLiteral = !Parameters[i].getAsInteger(10, IntParameter);
    (void)ValidLiteral;
    assert(ValidLiteral &&
           "Invalid format of SPIR-V builtin parameter literal!");
    IntParameters.push_back(IntParameter);
  }
  return TargetExtType::get(Context,
                            NameWithParameters.substr(0, BaseNameLength),
                            TypeParameters, IntParameters);
}

SPIRVType *lowerBuiltinType(const Type *OpaqueType,
                            SPIRV::AccessQualifier::AccessQualifier AccessQual,
                            MachineIRBuilder &MIRBuilder,
                            SPIRVGlobalRegistry *GR) {
  // In LLVM IR, SPIR-V and OpenCL builtin types are represented as either
  // target(...) target extension types or pointers-to-opaque-structs. The
  // approach relying on structs is deprecated and works only in the non-opaque
  // pointer mode (-opaque-pointers=0).
  // In order to maintain compatibility with LLVM IR generated by older versions
  // of Clang and LLVM/SPIR-V Translator, the pointers-to-opaque-structs are
  // "translated" to target extension types. This translation is temporary and
  // will be removed in the future release of LLVM.
  const TargetExtType *BuiltinType = dyn_cast<TargetExtType>(OpaqueType);
  if (!BuiltinType)
    BuiltinType = parseBuiltinTypeNameToTargetExtType(
        OpaqueType->getStructName().str(), MIRBuilder.getContext());

  unsigned NumStartingVRegs = MIRBuilder.getMRI()->getNumVirtRegs();

  const StringRef Name = BuiltinType->getName();
  LLVM_DEBUG(dbgs() << "Lowering builtin type: " << Name << "\n");

  SPIRVType *TargetType;
  if (Name == "spirv.Type") {
    TargetType = getInlineSpirvType(BuiltinType, MIRBuilder, GR);
  } else if (Name == "spirv.VulkanBuffer") {
    TargetType = getVulkanBufferType(BuiltinType, MIRBuilder, GR);
  } else if (Name == "spirv.Layout") {
    TargetType = getLayoutType(BuiltinType, MIRBuilder, GR);
  } else {
    // Lookup the demangled builtin type in the TableGen records.
    const SPIRV::BuiltinType *TypeRecord = SPIRV::lookupBuiltinType(Name);
    if (!TypeRecord)
      report_fatal_error("Missing TableGen record for builtin type: " + Name);

    // "Lower" the BuiltinType into TargetType. The following get<...>Type
    // methods use the implementation details from TableGen records or
    // TargetExtType parameters to either create a new OpType<...> machine
    // instruction or get an existing equivalent SPIRVType from
    // GlobalRegistry.

    switch (TypeRecord->Opcode) {
    case SPIRV::OpTypeImage:
      TargetType = GR->getImageType(BuiltinType, AccessQual, MIRBuilder);
      break;
    case SPIRV::OpTypePipe:
      TargetType = getPipeType(BuiltinType, MIRBuilder, GR);
      break;
    case SPIRV::OpTypeDeviceEvent:
      TargetType = GR->getOrCreateOpTypeDeviceEvent(MIRBuilder);
      break;
    case SPIRV::OpTypeSampler:
      TargetType = getSamplerType(MIRBuilder, GR);
      break;
    case SPIRV::OpTypeSampledImage:
      TargetType = getSampledImageType(BuiltinType, MIRBuilder, GR);
      break;
    case SPIRV::OpTypeCooperativeMatrixKHR:
      TargetType = getCoopMatrType(BuiltinType, MIRBuilder, GR);
      break;
    default:
      TargetType =
          getNonParameterizedType(BuiltinType, TypeRecord, MIRBuilder, GR);
      break;
    }
  }

  // Emit OpName instruction if a new OpType<...> instruction was added
  // (equivalent type was not found in GlobalRegistry).
  if (NumStartingVRegs < MIRBuilder.getMRI()->getNumVirtRegs())
    buildOpName(GR->getSPIRVTypeID(TargetType), Name, MIRBuilder);

  return TargetType;
}
} // namespace SPIRV
} // namespace llvm
