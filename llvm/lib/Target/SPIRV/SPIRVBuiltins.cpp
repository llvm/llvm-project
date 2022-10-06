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
#include "SPIRVUtils.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
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
      : BuiltinName(BuiltinName), Builtin(Builtin),
        ReturnRegister(ReturnRegister), ReturnType(ReturnType),
        Arguments(Arguments) {}
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

struct ConvertBuiltin {
  StringRef Name;
  InstructionSet::InstructionSet Set;
  bool IsDestinationSigned;
  bool IsSaturated;
  bool IsRounded;
  FPRoundingMode::FPRoundingMode RoundingMode;
};

struct VectorLoadStoreBuiltin {
  StringRef Name;
  InstructionSet::InstructionSet Set;
  uint32_t Number;
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
  // Extract the builtin function name and types of arguments from the call
  // skeleton.
  std::string BuiltinName =
      DemangledCall.substr(0, DemangledCall.find('(')).str();

  // Check if the extracted name contains type information between angle
  // brackets. If so, the builtin is an instantiated template - needs to have
  // the information after angle brackets and return type removed.
  if (BuiltinName.find('<') && BuiltinName.back() == '>') {
    BuiltinName = BuiltinName.substr(0, BuiltinName.find('<'));
    BuiltinName = BuiltinName.substr(BuiltinName.find_last_of(" ") + 1);
  }

  // Check if the extracted name begins with "__spirv_ImageSampleExplicitLod"
  // contains return type information at the end "_R<type>", if so extract the
  // plain builtin name without the type information.
  if (StringRef(BuiltinName).contains("__spirv_ImageSampleExplicitLod") &&
      StringRef(BuiltinName).contains("_R")) {
    BuiltinName = BuiltinName.substr(0, BuiltinName.find("_R"));
  }

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
  SPIRVType *BoolType = GR->getOrCreateSPIRVBoolType(MIRBuilder);

  if (ResultType->getOpcode() == SPIRV::OpTypeVector) {
    unsigned VectorElements = ResultType->getOperand(2).getImm();
    BoolType =
        GR->getOrCreateSPIRVVectorType(BoolType, VectorElements, MIRBuilder);
    const FixedVectorType *LLVMVectorType =
        cast<FixedVectorType>(GR->getTypeForSPIRVType(BoolType));
    Type = LLT::vector(LLVMVectorType->getElementCount(), 1);
  } else {
    Type = LLT::scalar(1);
  }

  Register ResultRegister =
      MIRBuilder.getMRI()->createGenericVirtualRegister(Type);
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
    uint64_t AllOnes = APInt::getAllOnesValue(Bits).getZExtValue();
    TrueConst = GR->getOrCreateConsIntVector(AllOnes, MIRBuilder, ReturnType);
    FalseConst = GR->getOrCreateConsIntVector(0, MIRBuilder, ReturnType);
  } else {
    TrueConst = GR->buildConstantInt(1, MIRBuilder, ReturnType);
    FalseConst = GR->buildConstantInt(0, MIRBuilder, ReturnType);
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
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  if (!DestinationReg.isValid()) {
    DestinationReg = MRI->createVirtualRegister(&SPIRV::IDRegClass);
    MRI->setType(DestinationReg, LLT::scalar(32));
    GR->assignSPIRVTypeToVReg(BaseType, DestinationReg, MIRBuilder.getMF());
  }
  // TODO: consider using correct address space and alignment (p0 is canonical
  // type for selection though).
  MachinePointerInfo PtrInfo = MachinePointerInfo();
  MIRBuilder.buildLoad(DestinationReg, PtrRegister, PtrInfo, Align());
  return DestinationReg;
}

/// Helper function for building a load instruction for loading a builtin global
/// variable of \p BuiltinValue value.
static Register buildBuiltinVariableLoad(MachineIRBuilder &MIRBuilder,
                                         SPIRVType *VariableType,
                                         SPIRVGlobalRegistry *GR,
                                         SPIRV::BuiltIn::BuiltIn BuiltinValue,
                                         LLT LLType,
                                         Register Reg = Register(0)) {
  Register NewRegister =
      MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::IDRegClass);
  MIRBuilder.getMRI()->setType(NewRegister,
                               LLT::pointer(0, GR->getPointerSize()));
  SPIRVType *PtrType = GR->getOrCreateSPIRVPointerType(
      VariableType, MIRBuilder, SPIRV::StorageClass::Input);
  GR->assignSPIRVTypeToVReg(PtrType, NewRegister, MIRBuilder.getMF());

  // Set up the global OpVariable with the necessary builtin decorations.
  Register Variable = GR->buildGlobalVariable(
      NewRegister, PtrType, getLinkStringForBuiltIn(BuiltinValue), nullptr,
      SPIRV::StorageClass::Input, nullptr, true, true,
      SPIRV::LinkageType::Import, MIRBuilder, false);

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
extern Register insertAssignInstr(Register Reg, Type *Ty, SPIRVType *SpirvTy,
                                  SPIRVGlobalRegistry *GR,
                                  MachineIRBuilder &MIB,
                                  MachineRegisterInfo &MRI);

// TODO: Move to TableGen.
static SPIRV::MemorySemantics::MemorySemantics
getSPIRVMemSemantics(std::memory_order MemOrder) {
  switch (MemOrder) {
  case std::memory_order::memory_order_relaxed:
    return SPIRV::MemorySemantics::None;
  case std::memory_order::memory_order_acquire:
    return SPIRV::MemorySemantics::Acquire;
  case std::memory_order::memory_order_release:
    return SPIRV::MemorySemantics::Release;
  case std::memory_order::memory_order_acq_rel:
    return SPIRV::MemorySemantics::AcquireRelease;
  case std::memory_order::memory_order_seq_cst:
    return SPIRV::MemorySemantics::SequentiallyConsistent;
  default:
    llvm_unreachable("Unknown CL memory scope");
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
  llvm_unreachable("Unknown CL memory scope");
}

static Register buildConstantIntReg(uint64_t Val, MachineIRBuilder &MIRBuilder,
                                    SPIRVGlobalRegistry *GR,
                                    unsigned BitWidth = 32) {
  SPIRVType *IntType = GR->getOrCreateSPIRVIntegerType(BitWidth, MIRBuilder);
  return GR->buildConstantInt(Val, MIRBuilder, IntType);
}

/// Helper function for building an atomic load instruction.
static bool buildAtomicLoadInst(const SPIRV::IncomingCall *Call,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  Register PtrRegister = Call->Arguments[0];
  // TODO: if true insert call to __translate_ocl_memory_sccope before
  // OpAtomicLoad and the function implementation. We can use Translator's
  // output for transcoding/atomic_explicit_arguments.cl as an example.
  Register ScopeRegister;
  if (Call->Arguments.size() > 1)
    ScopeRegister = Call->Arguments[1];
  else
    ScopeRegister = buildConstantIntReg(SPIRV::Scope::Device, MIRBuilder, GR);

  Register MemSemanticsReg;
  if (Call->Arguments.size() > 2) {
    // TODO: Insert call to __translate_ocl_memory_order before OpAtomicLoad.
    MemSemanticsReg = Call->Arguments[2];
  } else {
    int Semantics =
        SPIRV::MemorySemantics::SequentiallyConsistent |
        getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));
    MemSemanticsReg = buildConstantIntReg(Semantics, MIRBuilder, GR);
  }

  MIRBuilder.buildInstr(SPIRV::OpAtomicLoad)
      .addDef(Call->ReturnRegister)
      .addUse(GR->getSPIRVTypeID(Call->ReturnType))
      .addUse(PtrRegister)
      .addUse(ScopeRegister)
      .addUse(MemSemanticsReg);
  return true;
}

/// Helper function for building an atomic store instruction.
static bool buildAtomicStoreInst(const SPIRV::IncomingCall *Call,
                                 MachineIRBuilder &MIRBuilder,
                                 SPIRVGlobalRegistry *GR) {
  Register ScopeRegister =
      buildConstantIntReg(SPIRV::Scope::Device, MIRBuilder, GR);
  Register PtrRegister = Call->Arguments[0];
  int Semantics =
      SPIRV::MemorySemantics::SequentiallyConsistent |
      getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));
  Register MemSemanticsReg = buildConstantIntReg(Semantics, MIRBuilder, GR);

  MIRBuilder.buildInstr(SPIRV::OpAtomicStore)
      .addUse(PtrRegister)
      .addUse(ScopeRegister)
      .addUse(MemSemanticsReg)
      .addUse(Call->Arguments[1]);
  return true;
}

/// Helper function for building an atomic compare-exchange instruction.
static bool buildAtomicCompareExchangeInst(const SPIRV::IncomingCall *Call,
                                           MachineIRBuilder &MIRBuilder,
                                           SPIRVGlobalRegistry *GR) {
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode =
      SPIRV::lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;
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
    if (MemOrdEq == MemSemEqual)
      MemSemEqualReg = Call->Arguments[3];
    if (MemOrdNeq == MemSemEqual)
      MemSemUnequalReg = Call->Arguments[4];
  }
  if (!MemSemEqualReg.isValid())
    MemSemEqualReg = buildConstantIntReg(MemSemEqual, MIRBuilder, GR);
  if (!MemSemUnequalReg.isValid())
    MemSemUnequalReg = buildConstantIntReg(MemSemUnequal, MIRBuilder, GR);

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
    ScopeReg = buildConstantIntReg(Scope, MIRBuilder, GR);

  Register Expected = IsCmpxchg
                          ? ExpectedArg
                          : buildLoadInst(SpvDesiredTy, ExpectedArg, MIRBuilder,
                                          GR, LLT::scalar(32));
  MRI->setType(Expected, DesiredLLT);
  Register Tmp = !IsCmpxchg ? MRI->createGenericVirtualRegister(DesiredLLT)
                            : Call->ReturnRegister;
  GR->assignSPIRVTypeToVReg(SpvDesiredTy, Tmp, MIRBuilder.getMF());

  SPIRVType *IntTy = GR->getOrCreateSPIRVIntegerType(32, MIRBuilder);
  MIRBuilder.buildInstr(Opcode)
      .addDef(Tmp)
      .addUse(GR->getSPIRVTypeID(IntTy))
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

/// Helper function for building an atomic load instruction.
static bool buildAtomicRMWInst(const SPIRV::IncomingCall *Call, unsigned Opcode,
                               MachineIRBuilder &MIRBuilder,
                               SPIRVGlobalRegistry *GR) {
  const MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  Register ScopeRegister;
  SPIRV::Scope::Scope Scope = SPIRV::Scope::Workgroup;
  if (Call->Arguments.size() >= 4) {
    assert(Call->Arguments.size() == 4 && "Extra args for explicit atomic RMW");
    auto CLScope = static_cast<SPIRV::CLMemoryScope>(
        getIConstVal(Call->Arguments[5], MRI));
    Scope = getSPIRVScope(CLScope);
    if (CLScope == static_cast<unsigned>(Scope))
      ScopeRegister = Call->Arguments[5];
  }
  if (!ScopeRegister.isValid())
    ScopeRegister = buildConstantIntReg(Scope, MIRBuilder, GR);

  Register PtrRegister = Call->Arguments[0];
  Register MemSemanticsReg;
  unsigned Semantics = SPIRV::MemorySemantics::None;
  if (Call->Arguments.size() >= 3) {
    std::memory_order Order =
        static_cast<std::memory_order>(getIConstVal(Call->Arguments[2], MRI));
    Semantics =
        getSPIRVMemSemantics(Order) |
        getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));
    if (Order == Semantics)
      MemSemanticsReg = Call->Arguments[3];
  }
  if (!MemSemanticsReg.isValid())
    MemSemanticsReg = buildConstantIntReg(Semantics, MIRBuilder, GR);

  MIRBuilder.buildInstr(Opcode)
      .addDef(Call->ReturnRegister)
      .addUse(GR->getSPIRVTypeID(Call->ReturnType))
      .addUse(PtrRegister)
      .addUse(ScopeRegister)
      .addUse(MemSemanticsReg)
      .addUse(Call->Arguments[1]);
  return true;
}

/// Helper function for building barriers, i.e., memory/control ordering
/// operations.
static bool buildBarrierInst(const SPIRV::IncomingCall *Call, unsigned Opcode,
                             MachineIRBuilder &MIRBuilder,
                             SPIRVGlobalRegistry *GR) {
  const MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  unsigned MemFlags = getIConstVal(Call->Arguments[0], MRI);
  unsigned MemSemantics = SPIRV::MemorySemantics::None;

  if (MemFlags & SPIRV::CLK_LOCAL_MEM_FENCE)
    MemSemantics |= SPIRV::MemorySemantics::WorkgroupMemory;

  if (MemFlags & SPIRV::CLK_GLOBAL_MEM_FENCE)
    MemSemantics |= SPIRV::MemorySemantics::CrossWorkgroupMemory;

  if (MemFlags & SPIRV::CLK_IMAGE_MEM_FENCE)
    MemSemantics |= SPIRV::MemorySemantics::ImageMemory;

  if (Opcode == SPIRV::OpMemoryBarrier) {
    std::memory_order MemOrder =
        static_cast<std::memory_order>(getIConstVal(Call->Arguments[1], MRI));
    MemSemantics = getSPIRVMemSemantics(MemOrder) | MemSemantics;
  } else {
    MemSemantics |= SPIRV::MemorySemantics::SequentiallyConsistent;
  }

  Register MemSemanticsReg;
  if (MemFlags == MemSemantics)
    MemSemanticsReg = Call->Arguments[0];
  else
    MemSemanticsReg = buildConstantIntReg(MemSemantics, MIRBuilder, GR);

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
    ScopeReg = buildConstantIntReg(Scope, MIRBuilder, GR);

  auto MIB = MIRBuilder.buildInstr(Opcode).addUse(ScopeReg);
  if (Opcode != SPIRV::OpMemoryBarrier)
    MIB.addUse(buildConstantIntReg(MemScope, MIRBuilder, GR));
  MIB.addUse(MemSemanticsReg);
  return true;
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
    llvm_unreachable("Cannot get num components for given Dim");
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
                            SPIRVGlobalRegistry *GR) {
  // Lookup the extended instruction number in the TableGen records.
  const SPIRV::DemangledBuiltin *Builtin = Call->Builtin;
  uint32_t Number =
      SPIRV::lookupExtendedBuiltin(Builtin->Name, Builtin->Set)->Number;

  // Build extended instruction.
  auto MIB =
      MIRBuilder.buildInstr(SPIRV::OpExtInst)
          .addDef(Call->ReturnRegister)
          .addUse(GR->getSPIRVTypeID(Call->ReturnType))
          .addImm(static_cast<uint32_t>(SPIRV::InstructionSet::OpenCL_std))
          .addImm(Number);

  for (auto Argument : Call->Arguments)
    MIB.addUse(Argument);
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
  const MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  Register Arg0;
  if (GroupBuiltin->HasBoolArg) {
    Register ConstRegister = Call->Arguments[0];
    auto ArgInstruction = getDefInstrMaybeConstant(ConstRegister, MRI);
    // TODO: support non-constant bool values.
    assert(ArgInstruction->getOpcode() == TargetOpcode::G_CONSTANT &&
           "Only constant bool value args are supported");
    if (GR->getSPIRVTypeForVReg(Call->Arguments[0])->getOpcode() !=
        SPIRV::OpTypeBool)
      Arg0 = GR->buildConstantInt(getIConstVal(ConstRegister, MRI), MIRBuilder,
                                  GR->getOrCreateSPIRVBoolType(MIRBuilder));
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

  auto Scope = Builtin->Name.startswith("sub_group") ? SPIRV::Scope::Subgroup
                                                     : SPIRV::Scope::Workgroup;
  Register ScopeRegister = buildConstantIntReg(Scope, MIRBuilder, GR);

  // Build work/sub group instruction.
  auto MIB = MIRBuilder.buildInstr(GroupBuiltin->Opcode)
                 .addDef(GroupResultRegister)
                 .addUse(GR->getSPIRVTypeID(GroupResultType))
                 .addUse(ScopeRegister);

  if (!GroupBuiltin->NoGroupOperation)
    MIB.addImm(GroupBuiltin->GroupOperation);
  if (Call->Arguments.size() > 0) {
    MIB.addUse(Arg0.isValid() ? Arg0 : Call->Arguments[0]);
    for (unsigned i = 1; i < Call->Arguments.size(); i++)
      MIB.addUse(Call->Arguments[i]);
  }

  // Build select instruction.
  if (HasBoolReturnTy)
    buildSelectInst(MIRBuilder, Call->ReturnRegister, GroupResultRegister,
                    Call->ReturnType, GR);
  return true;
}

// These queries ask for a single size_t result for a given dimension index, e.g
// size_t get_global_id(uint dimindex). In SPIR-V, the builtins corresonding to
// these values are all vec3 types, so we need to extract the correct index or
// return defaultVal (0 or 1 depending on the query). We also handle extending
// or tuncating in case size_t does not match the expected result type's
// bitwidth.
//
// For a constant index >= 3 we generate:
//  %res = OpConstant %SizeT 0
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
//    %res = OpSelect %SizeT %cmp %tmp %const_0
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
    Register defaultReg = Call->ReturnRegister;
    if (PointerSize != ResultWidth) {
      defaultReg = MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
      GR->assignSPIRVTypeToVReg(PointerSizeType, defaultReg,
                                MIRBuilder.getMF());
      ToTruncate = defaultReg;
    }
    auto NewRegister =
        GR->buildConstantInt(DefaultValue, MIRBuilder, PointerSizeType);
    MIRBuilder.buildCopy(defaultReg, NewRegister);
  } else { // If it could be in range, we need to load from the given builtin.
    auto Vec3Ty =
        GR->getOrCreateSPIRVVectorType(PointerSizeType, 3, MIRBuilder);
    Register LoadedVector =
        buildBuiltinVariableLoad(MIRBuilder, Vec3Ty, GR, BuiltinValue,
                                 LLT::fixed_vector(3, PointerSize));
    // Set up the vreg to extract the result to (possibly a new temporary one).
    Register Extracted = Call->ReturnRegister;
    if (!IsConstantIndex || PointerSize != ResultWidth) {
      Extracted = MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
      GR->assignSPIRVTypeToVReg(PointerSizeType, Extracted, MIRBuilder.getMF());
    }
    // Use Intrinsic::spv_extractelt so dynamic vs static extraction is
    // handled later: extr = spv_extractelt LoadedVector, IndexRegister.
    MachineInstrBuilder ExtractInst = MIRBuilder.buildIntrinsic(
        Intrinsic::spv_extractelt, ArrayRef<Register>{Extracted}, true);
    ExtractInst.addUse(LoadedVector).addUse(IndexRegister);

    // If the index is dynamic, need check if it's < 3, and then use a select.
    if (!IsConstantIndex) {
      insertAssignInstr(Extracted, nullptr, PointerSizeType, GR, MIRBuilder,
                        *MRI);

      auto IndexType = GR->getSPIRVTypeForVReg(IndexRegister);
      auto BoolType = GR->getOrCreateSPIRVBoolType(MIRBuilder);

      Register CompareRegister =
          MRI->createGenericVirtualRegister(LLT::scalar(1));
      GR->assignSPIRVTypeToVReg(BoolType, CompareRegister, MIRBuilder.getMF());

      // Use G_ICMP to check if idxVReg < 3.
      MIRBuilder.buildICmp(CmpInst::ICMP_ULT, CompareRegister, IndexRegister,
                           GR->buildConstantInt(3, MIRBuilder, IndexType));

      // Get constant for the default value (0 or 1 depending on which
      // function).
      Register DefaultRegister =
          GR->buildConstantInt(DefaultValue, MIRBuilder, PointerSizeType);

      // Get a register for the selection result (possibly a new temporary one).
      Register SelectionResult = Call->ReturnRegister;
      if (PointerSize != ResultWidth) {
        SelectionResult =
            MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
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
  case SPIRV::OpAtomicLoad:
    return buildAtomicLoadInst(Call, MIRBuilder, GR);
  case SPIRV::OpAtomicStore:
    return buildAtomicStoreInst(Call, MIRBuilder, GR);
  case SPIRV::OpAtomicCompareExchange:
  case SPIRV::OpAtomicCompareExchangeWeak:
    return buildAtomicCompareExchangeInst(Call, MIRBuilder, GR);
  case SPIRV::OpAtomicIAdd:
  case SPIRV::OpAtomicISub:
  case SPIRV::OpAtomicOr:
  case SPIRV::OpAtomicXor:
  case SPIRV::OpAtomicAnd:
    return buildAtomicRMWInst(Call, Opcode, MIRBuilder, GR);
  case SPIRV::OpMemoryBarrier:
    return buildBarrierInst(Call, SPIRV::OpMemoryBarrier, MIRBuilder, GR);
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

static bool generateDotOrFMulInst(const SPIRV::IncomingCall *Call,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRVGlobalRegistry *GR) {
  unsigned Opcode = GR->getSPIRVTypeForVReg(Call->Arguments[0])->getOpcode();
  bool IsVec = Opcode == SPIRV::OpTypeVector;
  // Use OpDot only in case of vector args and OpFMul in case of scalar args.
  MIRBuilder.buildInstr(IsVec ? SPIRV::OpDot : SPIRV::OpFMulS)
      .addDef(Call->ReturnRegister)
      .addUse(GR->getSPIRVTypeID(Call->ReturnType))
      .addUse(Call->Arguments[0])
      .addUse(Call->Arguments[1]);
  return true;
}

static bool generateGetQueryInst(const SPIRV::IncomingCall *Call,
                                 MachineIRBuilder &MIRBuilder,
                                 SPIRVGlobalRegistry *GR) {
  // Lookup the builtin record.
  SPIRV::BuiltIn::BuiltIn Value =
      SPIRV::lookupGetBuiltin(Call->Builtin->Name, Call->Builtin->Set)->Value;
  uint64_t IsDefault = (Value == SPIRV::BuiltIn::GlobalSize ||
                        Value == SPIRV::BuiltIn::WorkgroupSize ||
                        Value == SPIRV::BuiltIn::EnqueuedWorkgroupSize);
  return genWorkgroupQuery(Call, MIRBuilder, GR, Value, IsDefault ? 1 : 0);
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
  SPIRVType *RetTy = Call->ReturnType;
  unsigned NumExpectedRetComponents = RetTy->getOpcode() == SPIRV::OpTypeVector
                                          ? RetTy->getOperand(2).getImm()
                                          : 1;
  // Get the actual number of query result/size components.
  SPIRVType *ImgType = GR->getSPIRVTypeForVReg(Call->Arguments[0]);
  unsigned NumActualRetComponents = getNumSizeComponents(ImgType);
  Register QueryResult = Call->ReturnRegister;
  SPIRVType *QueryResultType = Call->ReturnType;
  if (NumExpectedRetComponents != NumActualRetComponents) {
    QueryResult = MIRBuilder.getMRI()->createGenericVirtualRegister(
        LLT::fixed_vector(NumActualRetComponents, 32));
    SPIRVType *IntTy = GR->getOrCreateSPIRVIntegerType(32, MIRBuilder);
    QueryResultType = GR->getOrCreateSPIRVVectorType(
        IntTy, NumActualRetComponents, MIRBuilder);
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
    MIB.addUse(buildConstantIntReg(0, MIRBuilder, GR)); // Lod id.
  if (NumExpectedRetComponents == NumActualRetComponents)
    return true;
  if (NumExpectedRetComponents == 1) {
    // Only 1 component is expected, build OpCompositeExtract instruction.
    unsigned ExtractedComposite =
        Component == 3 ? NumActualRetComponents - 1 : Component;
    assert(ExtractedComposite < NumActualRetComponents &&
           "Invalid composite index!");
    MIRBuilder.buildInstr(SPIRV::OpCompositeExtract)
        .addDef(Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(Call->ReturnType))
        .addUse(QueryResult)
        .addImm(ExtractedComposite);
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
  // TODO: Add support for other image query builtins.
  Register Image = Call->Arguments[0];

  assert(Call->ReturnType->getOpcode() == SPIRV::OpTypeInt &&
         "Image samples query result must be of int type!");
  assert(GR->getSPIRVTypeForVReg(Image)->getOperand(2).getImm() == 1 &&
         "Image must be of 2D dimensionality");
  MIRBuilder.buildInstr(SPIRV::OpImageQuerySamples)
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
    llvm_unreachable("Unknown CL address mode");
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
  Register Image = Call->Arguments[0];
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();

  if (DemangledCall.contains_insensitive("ocl_sampler")) {
    Register Sampler = Call->Arguments[1];

    if (!GR->isScalarOfType(Sampler, SPIRV::OpTypeSampler) &&
        getDefInstrMaybeConstant(Sampler, MRI)->getOperand(1).isCImm()) {
      uint64_t SamplerMask = getIConstVal(Sampler, MRI);
      Sampler = GR->buildConstantSampler(
          Register(), getSamplerAddressingModeFromBitmask(SamplerMask),
          getSamplerParamFromBitmask(SamplerMask),
          getSamplerFilterModeFromBitmask(SamplerMask), MIRBuilder,
          GR->getSPIRVTypeForVReg(Sampler));
    }
    SPIRVType *ImageType = GR->getSPIRVTypeForVReg(Image);
    SPIRVType *SampledImageType =
        GR->getOrCreateOpTypeSampledImage(ImageType, MIRBuilder);
    Register SampledImage = MRI->createVirtualRegister(&SPIRV::IDRegClass);

    MIRBuilder.buildInstr(SPIRV::OpSampledImage)
        .addDef(SampledImage)
        .addUse(GR->getSPIRVTypeID(SampledImageType))
        .addUse(Image)
        .addUse(Sampler);

    Register Lod = GR->buildConstantFP(APFloat::getZero(APFloat::IEEEsingle()),
                                       MIRBuilder);
    SPIRVType *TempType = Call->ReturnType;
    bool NeedsExtraction = false;
    if (TempType->getOpcode() != SPIRV::OpTypeVector) {
      TempType =
          GR->getOrCreateSPIRVVectorType(Call->ReturnType, 4, MIRBuilder);
      NeedsExtraction = true;
    }
    LLT LLType = LLT::scalar(GR->getScalarOrVectorBitWidth(TempType));
    Register TempRegister = MRI->createGenericVirtualRegister(LLType);
    GR->assignSPIRVTypeToVReg(TempType, TempRegister, MIRBuilder.getMF());

    MIRBuilder.buildInstr(SPIRV::OpImageSampleExplicitLod)
        .addDef(NeedsExtraction ? TempRegister : Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(TempType))
        .addUse(SampledImage)
        .addUse(Call->Arguments[2]) // Coordinate.
        .addImm(SPIRV::ImageOperand::Lod)
        .addUse(Lod);

    if (NeedsExtraction)
      MIRBuilder.buildInstr(SPIRV::OpCompositeExtract)
          .addDef(Call->ReturnRegister)
          .addUse(GR->getSPIRVTypeID(Call->ReturnType))
          .addUse(TempRegister)
          .addImm(0);
  } else if (DemangledCall.contains_insensitive("msaa")) {
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
  if (Call->Builtin->Name.contains_insensitive(
          "__translate_sampler_initializer")) {
    // Build sampler literal.
    uint64_t Bitmask = getIConstVal(Call->Arguments[0], MIRBuilder.getMRI());
    Register Sampler = GR->buildConstantSampler(
        Call->ReturnRegister, getSamplerAddressingModeFromBitmask(Bitmask),
        getSamplerParamFromBitmask(Bitmask),
        getSamplerFilterModeFromBitmask(Bitmask), MIRBuilder, Call->ReturnType);
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
            : MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::IDRegClass);
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
    SPIRVType *Type = GR->getOrCreateSPIRVTypeByName(ReturnType, MIRBuilder);
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
    auto MIB = MIRBuilder.buildInstr(Opcode)
                   .addDef(Call->ReturnRegister)
                   .addUse(GR->getSPIRVTypeID(Call->ReturnType));
    for (unsigned i = 0; i < Call->Arguments.size(); i++)
      MIB.addUse(Call->Arguments[i]);
    return true;
  }
  default:
    return false;
  }
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
  case SPIRV::OpBuildNDRange: {
    MachineRegisterInfo *MRI = MIRBuilder.getMRI();
    SPIRVType *PtrType = GR->getSPIRVTypeForVReg(Call->Arguments[0]);
    assert(PtrType->getOpcode() == SPIRV::OpTypePointer &&
           PtrType->getOperand(2).isReg());
    Register TypeReg = PtrType->getOperand(2).getReg();
    SPIRVType *StructType = GR->getSPIRVTypeForVReg(TypeReg);
    Register TmpReg = MRI->createVirtualRegister(&SPIRV::IDRegClass);
    GR->assignSPIRVTypeToVReg(StructType, TmpReg, MIRBuilder.getMF());
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
        unsigned Size = Call->Builtin->Name.equals("ndrange_3D") ? 3 : 2;
        unsigned BitWidth = GR->getPointerSize() == 64 ? 64 : 32;
        Type *BaseTy = IntegerType::get(
            MIRBuilder.getMF().getFunction().getContext(), BitWidth);
        Type *FieldTy = ArrayType::get(BaseTy, Size);
        SPIRVType *SpvFieldTy = GR->getOrCreateSPIRVType(FieldTy, MIRBuilder);
        GlobalWorkSize = MRI->createVirtualRegister(&SPIRV::IDRegClass);
        GR->assignSPIRVTypeToVReg(SpvFieldTy, GlobalWorkSize,
                                  MIRBuilder.getMF());
        MIRBuilder.buildInstr(SPIRV::OpLoad)
            .addDef(GlobalWorkSize)
            .addUse(GR->getSPIRVTypeID(SpvFieldTy))
            .addUse(GWSPtr);
        Const = GR->getOrCreateConsIntArray(0, MIRBuilder, SpvFieldTy);
      } else {
        Const = GR->buildConstantInt(0, MIRBuilder, SpvTy);
      }
      if (!LocalWorkSize.isValid())
        LocalWorkSize = Const;
      if (!GlobalWorkOffset.isValid())
        GlobalWorkOffset = Const;
    }
    MIRBuilder.buildInstr(Opcode)
        .addDef(TmpReg)
        .addUse(TypeReg)
        .addUse(GlobalWorkSize)
        .addUse(LocalWorkSize)
        .addUse(GlobalWorkOffset);
    return MIRBuilder.buildInstr(SPIRV::OpStore)
        .addUse(Call->Arguments[0])
        .addUse(TmpReg);
  }
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
  auto Scope = buildConstantIntReg(SPIRV::Scope::Workgroup, MIRBuilder, GR);

  switch (Opcode) {
  case SPIRV::OpGroupAsyncCopy:
    return MIRBuilder.buildInstr(Opcode)
        .addDef(Call->ReturnRegister)
        .addUse(GR->getSPIRVTypeID(Call->ReturnType))
        .addUse(Scope)
        .addUse(Call->Arguments[0])
        .addUse(Call->Arguments[1])
        .addUse(Call->Arguments[2])
        .addUse(buildConstantIntReg(1, MIRBuilder, GR))
        .addUse(Call->Arguments[3]);
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

  if (Builtin->IsSaturated)
    buildOpDecorate(Call->ReturnRegister, MIRBuilder,
                    SPIRV::Decoration::SaturatedConversion, {});
  if (Builtin->IsRounded)
    buildOpDecorate(Call->ReturnRegister, MIRBuilder,
                    SPIRV::Decoration::FPRoundingMode, {Builtin->RoundingMode});

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
      bool IsSourceSigned =
          DemangledCall[DemangledCall.find_first_of('(') + 1] != 'u';
      Opcode = IsSourceSigned ? SPIRV::OpConvertSToF : SPIRV::OpConvertUToF;
    }
  } else if (GR->isScalarOrVectorOfType(Call->Arguments[0],
                                        SPIRV::OpTypeFloat)) {
    // Float -> ...
    if (GR->isScalarOrVectorOfType(Call->ReturnRegister, SPIRV::OpTypeInt))
      // Float -> Int
      Opcode = Builtin->IsDestinationSigned ? SPIRV::OpConvertFToS
                                            : SPIRV::OpConvertFToU;
    else if (GR->isScalarOrVectorOfType(Call->ReturnRegister,
                                        SPIRV::OpTypeFloat))
      // Float -> Float
      Opcode = SPIRV::OpFConvert;
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

  // Rounding mode should be passed as a last argument in the MI for builtins
  // like "vstorea_halfn_r".
  if (Builtin->IsRounded)
    MIB.addImm(static_cast<uint32_t>(Builtin->RoundingMode));
  return true;
}

/// Lowers a builtin funtion call using the provided \p DemangledCall skeleton
/// and external instruction \p Set.
namespace SPIRV {
Optional<bool> lowerBuiltin(const StringRef DemangledCall,
                            SPIRV::InstructionSet::InstructionSet Set,
                            MachineIRBuilder &MIRBuilder,
                            const Register OrigRet, const Type *OrigRetTy,
                            const SmallVectorImpl<Register> &Args,
                            SPIRVGlobalRegistry *GR) {
  LLVM_DEBUG(dbgs() << "Lowering builtin call: " << DemangledCall << "\n");

  // SPIR-V type and return register.
  Register ReturnRegister = OrigRet;
  SPIRVType *ReturnType = nullptr;
  if (OrigRetTy && !OrigRetTy->isVoidTy()) {
    ReturnType = GR->assignTypeToVReg(OrigRetTy, OrigRet, MIRBuilder);
  } else if (OrigRetTy && OrigRetTy->isVoidTy()) {
    ReturnRegister = MIRBuilder.getMRI()->createVirtualRegister(&IDRegClass);
    MIRBuilder.getMRI()->setType(ReturnRegister, LLT::scalar(32));
    ReturnType = GR->assignTypeToVReg(OrigRetTy, ReturnRegister, MIRBuilder);
  }

  // Lookup the builtin in the TableGen records.
  std::unique_ptr<const IncomingCall> Call =
      lookupBuiltin(DemangledCall, Set, ReturnRegister, ReturnType, Args);

  if (!Call) {
    LLVM_DEBUG(dbgs() << "Builtin record was not found!");
    return {};
  }

  // TODO: check if the provided args meet the builtin requirments.
  assert(Args.size() >= Call->Builtin->MinNumArgs &&
         "Too few arguments to generate the builtin");
  if (Call->Builtin->MaxNumArgs && Args.size() <= Call->Builtin->MaxNumArgs)
    LLVM_DEBUG(dbgs() << "More arguments provided than required!");

  // Match the builtin with implementation based on the grouping.
  switch (Call->Builtin->Group) {
  case SPIRV::Extended:
    return generateExtInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Relational:
    return generateRelationalInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Group:
    return generateGroupInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Variable:
    return generateBuiltinVar(Call.get(), MIRBuilder, GR);
  case SPIRV::Atomic:
    return generateAtomicInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Barrier:
    return generateBarrierInst(Call.get(), MIRBuilder, GR);
  case SPIRV::Dot:
    return generateDotOrFMulInst(Call.get(), MIRBuilder, GR);
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
  }
  return false;
}

struct DemangledType {
  StringRef Name;
  uint32_t Opcode;
};

#define GET_DemangledTypes_DECL
#define GET_DemangledTypes_IMPL

struct ImageType {
  StringRef Name;
  StringRef SampledType;
  AccessQualifier::AccessQualifier Qualifier;
  Dim::Dim Dimensionality;
  bool Arrayed;
  bool Depth;
  bool Multisampled;
  bool Sampled;
  ImageFormat::ImageFormat Format;
};

struct PipeType {
  StringRef Name;
  AccessQualifier::AccessQualifier Qualifier;
};

using namespace AccessQualifier;
using namespace Dim;
using namespace ImageFormat;
#define GET_ImageTypes_DECL
#define GET_ImageTypes_IMPL
#define GET_PipeTypes_DECL
#define GET_PipeTypes_IMPL
#include "SPIRVGenTables.inc"
} // namespace SPIRV

//===----------------------------------------------------------------------===//
// Misc functions for parsing builtin types and looking up implementation
// details in TableGenerated tables.
//===----------------------------------------------------------------------===//

static const SPIRV::DemangledType *findBuiltinType(StringRef Name) {
  if (Name.startswith("opencl."))
    return SPIRV::lookupBuiltinType(Name);
  if (!Name.startswith("spirv."))
    return nullptr;
  // Some SPIR-V builtin types have a complex list of parameters as part of
  // their name (e.g. spirv.Image._void_1_0_0_0_0_0_0). Those parameters often
  // are numeric literals which cannot be easily represented by TableGen
  // records and should be parsed instead.
  unsigned BaseTypeNameLength =
      Name.contains('_') ? Name.find('_') - 1 : Name.size();
  return SPIRV::lookupBuiltinType(Name.substr(0, BaseTypeNameLength).str());
}

static std::unique_ptr<const SPIRV::ImageType>
lookupOrParseBuiltinImageType(StringRef Name) {
  if (Name.startswith("opencl.")) {
    // Lookup OpenCL builtin image type lowering details in TableGen records.
    const SPIRV::ImageType *Record = SPIRV::lookupImageType(Name);
    return std::unique_ptr<SPIRV::ImageType>(new SPIRV::ImageType(*Record));
  }
  if (!Name.startswith("spirv."))
    llvm_unreachable("Unknown builtin image type name/literal");
  // Parse the literals of SPIR-V image builtin parameters. The name should
  // have the following format:
  // spirv.Image._Type_Dim_Depth_Arrayed_MS_Sampled_ImageFormat_AccessQualifier
  // e.g. %spirv.Image._void_1_0_0_0_0_0_0
  StringRef TypeParametersString = Name.substr(strlen("spirv.Image."));
  SmallVector<StringRef> TypeParameters;
  SplitString(TypeParametersString, TypeParameters, "_");
  assert(TypeParameters.size() == 8 &&
         "Wrong number of literals in SPIR-V builtin image type");

  StringRef SampledType = TypeParameters[0];
  unsigned Dim, Depth, Arrayed, Multisampled, Sampled, Format, AccessQual;
  bool AreParameterLiteralsValid =
      !(TypeParameters[1].getAsInteger(10, Dim) ||
        TypeParameters[2].getAsInteger(10, Depth) ||
        TypeParameters[3].getAsInteger(10, Arrayed) ||
        TypeParameters[4].getAsInteger(10, Multisampled) ||
        TypeParameters[5].getAsInteger(10, Sampled) ||
        TypeParameters[6].getAsInteger(10, Format) ||
        TypeParameters[7].getAsInteger(10, AccessQual));
  assert(AreParameterLiteralsValid &&
         "Invalid format of SPIR-V image type parameter literals.");

  return std::unique_ptr<SPIRV::ImageType>(new SPIRV::ImageType{
      Name, SampledType, SPIRV::AccessQualifier::AccessQualifier(AccessQual),
      SPIRV::Dim::Dim(Dim), static_cast<bool>(Arrayed),
      static_cast<bool>(Depth), static_cast<bool>(Multisampled),
      static_cast<bool>(Sampled), SPIRV::ImageFormat::ImageFormat(Format)});
}

static std::unique_ptr<const SPIRV::PipeType>
lookupOrParseBuiltinPipeType(StringRef Name) {
  if (Name.startswith("opencl.")) {
    // Lookup OpenCL builtin pipe type lowering details in TableGen records.
    const SPIRV::PipeType *Record = SPIRV::lookupPipeType(Name);
    return std::unique_ptr<SPIRV::PipeType>(new SPIRV::PipeType(*Record));
  }
  if (!Name.startswith("spirv."))
    llvm_unreachable("Unknown builtin pipe type name/literal");
  // Parse the access qualifier literal in the name of the SPIR-V pipe type.
  // The name should have the following format:
  // spirv.Pipe._AccessQualifier
  // e.g. %spirv.Pipe._1
  if (Name.endswith("_0"))
    return std::unique_ptr<SPIRV::PipeType>(
        new SPIRV::PipeType{Name, SPIRV::AccessQualifier::ReadOnly});
  if (Name.endswith("_1"))
    return std::unique_ptr<SPIRV::PipeType>(
        new SPIRV::PipeType{Name, SPIRV::AccessQualifier::WriteOnly});
  if (Name.endswith("_2"))
    return std::unique_ptr<SPIRV::PipeType>(
        new SPIRV::PipeType{Name, SPIRV::AccessQualifier::ReadWrite});
  llvm_unreachable("Unknown pipe type access qualifier literal");
}

//===----------------------------------------------------------------------===//
// Implementation functions for builtin types.
//===----------------------------------------------------------------------===//

static SPIRVType *getNonParametrizedType(const StructType *OpaqueType,
                                         const SPIRV::DemangledType *TypeRecord,
                                         MachineIRBuilder &MIRBuilder,
                                         SPIRVGlobalRegistry *GR) {
  unsigned Opcode = TypeRecord->Opcode;
  // Create or get an existing type from GlobalRegistry.
  return GR->getOrCreateOpTypeByOpcode(OpaqueType, MIRBuilder, Opcode);
}

static SPIRVType *getSamplerType(MachineIRBuilder &MIRBuilder,
                                 SPIRVGlobalRegistry *GR) {
  // Create or get an existing type from GlobalRegistry.
  return GR->getOrCreateOpTypeSampler(MIRBuilder);
}

static SPIRVType *getPipeType(const StructType *OpaqueType,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR) {
  // Lookup pipe type lowering details in TableGen records or parse the
  // name/literal for details.
  std::unique_ptr<const SPIRV::PipeType> Record =
      lookupOrParseBuiltinPipeType(OpaqueType->getName());
  // Create or get an existing type from GlobalRegistry.
  return GR->getOrCreateOpTypePipe(MIRBuilder, Record.get()->Qualifier);
}

static SPIRVType *
getImageType(const StructType *OpaqueType,
             SPIRV::AccessQualifier::AccessQualifier AccessQual,
             MachineIRBuilder &MIRBuilder, SPIRVGlobalRegistry *GR) {
  // Lookup image type lowering details in TableGen records or parse the
  // name/literal for details.
  std::unique_ptr<const SPIRV::ImageType> Record =
      lookupOrParseBuiltinImageType(OpaqueType->getName());

  SPIRVType *SampledType =
      GR->getOrCreateSPIRVTypeByName(Record.get()->SampledType, MIRBuilder);
  return GR->getOrCreateOpTypeImage(
      MIRBuilder, SampledType, Record.get()->Dimensionality,
      Record.get()->Depth, Record.get()->Arrayed, Record.get()->Multisampled,
      Record.get()->Sampled, Record.get()->Format,
      AccessQual == SPIRV::AccessQualifier::WriteOnly
          ? SPIRV::AccessQualifier::WriteOnly
          : Record.get()->Qualifier);
}

static SPIRVType *getSampledImageType(const StructType *OpaqueType,
                                      MachineIRBuilder &MIRBuilder,
                                      SPIRVGlobalRegistry *GR) {
  StringRef TypeParametersString =
      OpaqueType->getName().substr(strlen("spirv.SampledImage."));
  LLVMContext &Context = MIRBuilder.getMF().getFunction().getContext();
  Type *ImageOpaqueType = StructType::getTypeByName(
      Context, "spirv.Image." + TypeParametersString.str());
  SPIRVType *TargetImageType =
      GR->getOrCreateSPIRVType(ImageOpaqueType, MIRBuilder);
  return GR->getOrCreateOpTypeSampledImage(TargetImageType, MIRBuilder);
}

namespace SPIRV {
SPIRVType *lowerBuiltinType(const StructType *OpaqueType,
                            SPIRV::AccessQualifier::AccessQualifier AccessQual,
                            MachineIRBuilder &MIRBuilder,
                            SPIRVGlobalRegistry *GR) {
  assert(OpaqueType->hasName() &&
         "Structs representing builtin types must have a parsable name");
  unsigned NumStartingVRegs = MIRBuilder.getMRI()->getNumVirtRegs();

  const StringRef Name = OpaqueType->getName();
  LLVM_DEBUG(dbgs() << "Lowering builtin type: " << Name << "\n");

  // Lookup the demangled builtin type in the TableGen records.
  const SPIRV::DemangledType *TypeRecord = findBuiltinType(Name);
  if (!TypeRecord)
    report_fatal_error("Missing TableGen record for builtin type: " + Name);

  // "Lower" the BuiltinType into TargetType. The following get<...>Type methods
  // use the implementation details from TableGen records to either create a new
  // OpType<...> machine instruction or get an existing equivalent SPIRVType
  // from GlobalRegistry.
  SPIRVType *TargetType;
  switch (TypeRecord->Opcode) {
  case SPIRV::OpTypeImage:
    TargetType = getImageType(OpaqueType, AccessQual, MIRBuilder, GR);
    break;
  case SPIRV::OpTypePipe:
    TargetType = getPipeType(OpaqueType, MIRBuilder, GR);
    break;
  case SPIRV::OpTypeSampler:
    TargetType = getSamplerType(MIRBuilder, GR);
    break;
  case SPIRV::OpTypeSampledImage:
    TargetType = getSampledImageType(OpaqueType, MIRBuilder, GR);
    break;
  default:
    TargetType = getNonParametrizedType(OpaqueType, TypeRecord, MIRBuilder, GR);
    break;
  }

  // Emit OpName instruction if a new OpType<...> instruction was added
  // (equivalent type was not found in GlobalRegistry).
  if (NumStartingVRegs < MIRBuilder.getMRI()->getNumVirtRegs())
    buildOpName(GR->getSPIRVTypeID(TargetType), OpaqueType->getName(),
                MIRBuilder);

  return TargetType;
}
} // namespace SPIRV
} // namespace llvm
