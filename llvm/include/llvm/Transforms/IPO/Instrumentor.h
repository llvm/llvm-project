//===- Transforms/IPO/Instrumentor.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A highly configurable instrumentation pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INSTRUMENTOR_H
#define LLVM_TRANSFORMS_IPO_INSTRUMENTOR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Transforms/Utils/Instrumentation.h"

#include <bitset>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>

namespace llvm {
namespace instrumentor {

struct InstrumentationConfig;
struct InstrumentationOpportunity;

struct InstrumentorIRBuilderTy {
  InstrumentorIRBuilderTy(Module &M, FunctionAnalysisManager &FAM)
      : M(M), Ctx(M.getContext()), FAM(FAM),
        IRB(Ctx, ConstantFolder(),
            IRBuilderCallbackInserter(
                [&](Instruction *I) { NewInsts[I] = Epoche; })) {}

  ~InstrumentorIRBuilderTy() {
    for (auto *I : ToBeErased) {
      if (!I->getType()->isVoidTy())
        I->replaceAllUsesWith(PoisonValue::get(I->getType()));
      I->eraseFromParent();
    }
  }

  /// Get a temporary alloca to communicate (large) values with the runtime.
  AllocaInst *getAlloca(Function *Fn, Type *Ty, bool MatchType = false) {
    const DataLayout &DL = Fn->getDataLayout();
    auto *&AllocaList = AllocaMap[{Fn, DL.getTypeAllocSize(Ty)}];
    if (!AllocaList)
      AllocaList = new AllocaListTy;
    AllocaInst *AI = nullptr;
    for (auto *&ListAI : *AllocaList) {
      if (MatchType && ListAI->getAllocatedType() != Ty)
        continue;
      AI = ListAI;
      ListAI = *AllocaList->rbegin();
      break;
    }
    if (AI)
      AllocaList->pop_back();
    else
      AI = new AllocaInst(Ty, DL.getAllocaAddrSpace(), "",
                          Fn->getEntryBlock().begin());
    UsedAllocas[AI] = AllocaList;
    return AI;
  }

  /// Return the temporary allocas.
  void returnAllocas() {
    for (auto [AI, List] : UsedAllocas)
      List->push_back(AI);
    UsedAllocas.clear();
  }

  /// Commonly used values for IR inspection and creation.
  ///{

  Module &M;

  /// The underying LLVM context.
  LLVMContext &Ctx;

  const DataLayout &DL = M.getDataLayout();

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);
  Constant *NullPtrVal = Constant::getNullValue(PtrTy);
  ///}

  /// Mapping to remember temporary allocas for reuse.
  using AllocaListTy = SmallVector<AllocaInst *>;
  DenseMap<std::pair<Function *, unsigned>, AllocaListTy *> AllocaMap;
  DenseMap<AllocaInst *, SmallVector<AllocaInst *> *> UsedAllocas;

  void eraseLater(Instruction *I) { ToBeErased.insert(I); }
  SmallPtrSet<Instruction *, 32> ToBeErased;

  FunctionAnalysisManager &FAM;

  IRBuilder<ConstantFolder, IRBuilderCallbackInserter> IRB;

  /// Each instrumentation, i.a., of an instruction, is happening in a dedicated
  /// epoche. The epoche allows to determine if instrumentation instructions
  /// were already around, due to prior instrumentations, or have been
  /// introduced to support the current instrumentation, i.a., compute
  /// information about the current instruction.
  unsigned Epoche = 0;

  /// A mapping from instrumentation instructions to the epoche they have been
  /// created.
  DenseMap<Instruction *, unsigned> NewInsts;
};

using GetterCallbackTy = std::function<Value *(
    Value &, Type &, InstrumentationConfig &, InstrumentorIRBuilderTy &)>;
using SetterCallbackTy = std::function<Value *(
    Value &, Value &, InstrumentationConfig &, InstrumentorIRBuilderTy &)>;

struct IRTArg {
  enum IRArgFlagTy {
    NONE = 0,
    STRING = 1 << 0,
    REPLACABLE = 1 << 1,
    REPLACABLE_CUSTOM = 1 << 2,
    POTENTIALLY_INDIRECT = 1 << 3,
    INDIRECT_HAS_SIZE = 1 << 4,

    LAST,
  };

  IRTArg(Type *Ty, StringRef Name, StringRef Description, unsigned Flags,
         GetterCallbackTy GetterCB, SetterCallbackTy SetterCB = nullptr,
         bool Enabled = true, bool NoCache = false)
      : Enabled(Enabled), Ty(Ty), Name(Name), Description(Description),
        Flags(Flags), GetterCB(std::move(GetterCB)),
        SetterCB(std::move(SetterCB)), NoCache(NoCache) {}

  bool Enabled;
  Type *Ty;
  StringRef Name;
  StringRef Description;
  unsigned Flags;
  GetterCallbackTy GetterCB;
  SetterCallbackTy SetterCB;
  bool NoCache;
};

struct InstrumentationCaches {
  DenseMap<std::tuple<unsigned, StringRef, StringRef>, Value *> DirectArgCache;
  DenseMap<std::tuple<unsigned, StringRef, StringRef>, Value *>
      IndirectArgCache;
};

struct IRTCallDescription {
  IRTCallDescription(InstrumentationOpportunity &IConf, Type *RetTy = nullptr);

  FunctionType *createLLVMSignature(InstrumentationConfig &IConf,
                                    LLVMContext &Ctx, const DataLayout &DL,
                                    bool ForceIndirection);
  CallInst *createLLVMCall(Value *&V, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB, const DataLayout &DL,
                           InstrumentationCaches &ICaches);

  bool isReplacable(IRTArg &IRTA) const {
    return (IRTA.Flags & (IRTArg::REPLACABLE | IRTArg::REPLACABLE_CUSTOM));
  }

  bool isPotentiallyIndirect(IRTArg &IRTA) const {
    return ((IRTA.Flags & IRTArg::POTENTIALLY_INDIRECT) ||
            ((IRTA.Flags & IRTArg::REPLACABLE) && NumReplaceableArgs > 1));
  }

  bool RequiresIndirection = false;
  bool MightRequireIndirection = false;
  unsigned NumReplaceableArgs = 0;
  InstrumentationOpportunity &IO;
  Type *RetTy = nullptr;
};

struct InstrumentationLocation {
  enum KindTy {
    MODULE_PRE,
    MODULE_POST,
    GLOBAL_PRE,
    GLOBAL_POST,
    FUNCTION_PRE,
    FUNCTION_POST,
    BASIC_BLOCK_PRE,
    BASIC_BLOCK_POST,
    INSTRUCTION_PRE,
    INSTRUCTION_POST,
    SPECIAL_VALUE,
    Last = SPECIAL_VALUE,
  };

  InstrumentationLocation(KindTy Kind) : Kind(Kind) {
    assert(Kind != INSTRUCTION_PRE && Kind != INSTRUCTION_POST &&
           "Opcode required!");
  }

  InstrumentationLocation(unsigned Opcode, bool IsPRE)
      : Kind(IsPRE ? INSTRUCTION_PRE : INSTRUCTION_POST), Opcode(Opcode) {}

  KindTy getKind() const { return Kind; }

  static StringRef getKindStr(KindTy Kind) {
    switch (Kind) {
    case MODULE_PRE:
      return "module_pre";
    case MODULE_POST:
      return "module_post";
    case GLOBAL_PRE:
      return "global_pre";
    case GLOBAL_POST:
      return "global_post";
    case FUNCTION_PRE:
      return "function_pre";
    case FUNCTION_POST:
      return "function_post";
    case BASIC_BLOCK_PRE:
      return "basic_block_pre";
    case BASIC_BLOCK_POST:
      return "basic_block_post";
    case INSTRUCTION_PRE:
      return "instruction_pre";
    case INSTRUCTION_POST:
      return "instruction_post";
    case SPECIAL_VALUE:
      return "special_value";
    }
    llvm_unreachable("Invalid kind!");
  }
  static KindTy getKindFromStr(StringRef S) {
    return StringSwitch<KindTy>(S)
        .Case("module_pre", MODULE_PRE)
        .Case("module_post", MODULE_POST)
        .Case("global_pre", GLOBAL_PRE)
        .Case("global_post", GLOBAL_POST)
        .Case("function_pre", FUNCTION_PRE)
        .Case("function_post", FUNCTION_POST)
        .Case("basic_block_pre", BASIC_BLOCK_PRE)
        .Case("basic_block_post", BASIC_BLOCK_POST)
        .Case("instruction_pre", INSTRUCTION_PRE)
        .Case("instruction_post", INSTRUCTION_POST)
        .Case("special_value", SPECIAL_VALUE)
        .Default(Last);
  }

  static bool isPRE(KindTy Kind) {
    switch (Kind) {
    case MODULE_PRE:
    case GLOBAL_PRE:
    case FUNCTION_PRE:
    case BASIC_BLOCK_PRE:
    case INSTRUCTION_PRE:
      return true;
    case MODULE_POST:
    case GLOBAL_POST:
    case FUNCTION_POST:
    case BASIC_BLOCK_POST:
    case INSTRUCTION_POST:
    case SPECIAL_VALUE:
      return false;
    }
    llvm_unreachable("Invalid kind!");
  }
  bool isPRE() const { return isPRE(Kind); }

  unsigned getOpcode() const {
    assert((Kind == INSTRUCTION_PRE || Kind == INSTRUCTION_POST) &&
           "Expected instruction!");
    return Opcode;
  }

private:
  const KindTy Kind;
  const unsigned Opcode = -1;
};

struct BaseConfigurationOpportunity {
  enum KindTy {
    STRING,
    BOOLEAN,
  };

  static BaseConfigurationOpportunity *getBoolOption(InstrumentationConfig &IC,
                                                     StringRef Name,
                                                     StringRef Description,
                                                     bool B);
  static BaseConfigurationOpportunity *
  getStringOption(InstrumentationConfig &IC, StringRef Name,
                  StringRef Description, StringRef Value);
  union ValueTy {
    bool B;
    int64_t I;
    StringRef S;
  };

  void setBool(bool B) {
    assert(Kind == BOOLEAN && "Not a boolean!");
    V.B = B;
  }
  bool getBool() const {
    assert(Kind == BOOLEAN && "Not a boolean!");
    return V.B;
  }
  void setString(StringRef S) {
    assert(Kind == STRING && "Not a string!");
    V.S = S;
  }
  StringRef getString() const {
    assert(Kind == STRING && "Not a string!");
    return V.S;
  }

  StringRef Name;
  StringRef Description;
  KindTy Kind;
  ValueTy V = {0};
};

struct InstrumentorIRBuilderTy;
struct InstrumentationConfig {
  virtual ~InstrumentationConfig() {}

  InstrumentationConfig() : SS(StringAllocator) {
    RuntimePrefix = BaseConfigurationOpportunity::getStringOption(
        *this, "runtime_prefix", "The runtime API prefix.", "__instrumentor_");
    TargetRegex = BaseConfigurationOpportunity::getStringOption(
        *this, "target_regex",
        "Regular expression to be matched against the module target. "
        "Only targets that match this regex will be instrumented",
        "");
    HostEnabled = BaseConfigurationOpportunity::getBoolOption(
        *this, "host_enabled", "Instrument non-GPU targets", true);
    GPUEnabled = BaseConfigurationOpportunity::getBoolOption(
        *this, "gpu_enabled", "Instrument GPU targets", true);
  }

  bool ReadConfig = true;

  virtual void populate(InstrumentorIRBuilderTy &IIRB);
  StringRef getRTName() const { return RuntimePrefix->getString(); }

  std::string getRTName(StringRef Prefix, StringRef Name,
                        StringRef Suffix1 = "", StringRef Suffix2 = "") const {
    return (getRTName() + Prefix + Name + Suffix1 + Suffix2).str();
  }

  void addBaseChoice(BaseConfigurationOpportunity *BCO) {
    BaseConfigurationOpportunities.push_back(BCO);
  }
  SmallVector<BaseConfigurationOpportunity *> BaseConfigurationOpportunities;

  BaseConfigurationOpportunity *RuntimePrefix;
  BaseConfigurationOpportunity *TargetRegex;
  BaseConfigurationOpportunity *HostEnabled;
  BaseConfigurationOpportunity *GPUEnabled;

  EnumeratedArray<StringMap<InstrumentationOpportunity *>,
                  InstrumentationLocation::KindTy>
      IChoices;
  void addChoice(InstrumentationOpportunity &IO);

  template <typename Ty, typename... ArgsTy>
  static Ty *allocate(ArgsTy &&...Args) {
    static SpecificBumpPtrAllocator<Ty> Allocator;
    Ty *Obj = Allocator.Allocate();
    new (Obj) Ty(std::forward<ArgsTy>(Args)...);
    return Obj;
  }

  BumpPtrAllocator StringAllocator;
  StringSaver SS;
};

template <typename EnumTy> struct BaseConfigTy {
  std::bitset<static_cast<int>(EnumTy::NumConfig)> Options;

  BaseConfigTy(bool Enable = true) {
    if (Enable)
      Options.set();
  }

  bool has(EnumTy Opt) const { return Options.test(static_cast<int>(Opt)); }
  void set(EnumTy Opt, bool Value = true) {
    Options.set(static_cast<int>(Opt), Value);
  }
};

struct InstrumentationOpportunity {
  InstrumentationOpportunity(const InstrumentationLocation IP) : IP(IP) {}
  virtual ~InstrumentationOpportunity() {}

  InstrumentationLocation IP;

  SmallVector<IRTArg> IRTArgs;
  bool Enabled = true;

  /// Helpers to cast values, pass them to the runtime, and replace them. To be
  /// used as part of the getter/setter of a InstrumentationOpportunity.
  ///{
  static Value *forceCast(Value &V, Type &Ty, InstrumentorIRBuilderTy &IIRB);
  static Value *getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB) {
    return forceCast(V, Ty, IIRB);
  }

  static Value *replaceValue(Value &V, Value &NewV,
                             InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  ///}

  virtual Value *instrument(Value *&V, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB,
                            InstrumentationCaches &ICaches) {
    if (CB && !CB(*V))
      return nullptr;

    const DataLayout &DL = IIRB.IRB.GetInsertBlock()->getDataLayout();
    IRTCallDescription IRTCallDesc(*this, getRetTy(V->getContext()));
    auto *CI = IRTCallDesc.createLLVMCall(V, IConf, IIRB, DL, ICaches);
    return CI;
  }

  virtual Type *getRetTy(LLVMContext &Ctx) const { return nullptr; }
  virtual StringRef getName() const = 0;

  unsigned getOpcode() const { return IP.getOpcode(); }
  InstrumentationLocation::KindTy getLocationKind() const {
    return IP.getKind();
  }

  /// An optional callback that takes the value that is about to be
  /// instrumented and can return false if it should be skipped.
  using CallbackTy = std::function<bool(Value &)>;

  CallbackTy CB = nullptr;

  static Value *getIdPre(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getIdPost(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB);

  void addCommonArgs(InstrumentationConfig &IConf, LLVMContext &Ctx,
                     bool PassId) {
    const auto CB = IP.isPRE() ? getIdPre : getIdPost;
    if (PassId)
      IRTArgs.push_back(
          IRTArg(IntegerType::getInt32Ty(Ctx), "id",
                 "A unique ID associated with the given instrumentor call",
                 IRTArg::NONE, CB, nullptr, true, true));
  }

  static int32_t getIdFromEpoche(uint32_t Epoche) {
    static DenseMap<uint32_t, int32_t> EpocheIdMap;
    static int32_t GlobalId = 0;
    int32_t &EpochId = EpocheIdMap[Epoche];
    if (EpochId == 0)
      EpochId = ++GlobalId;
    return EpochId;
  }
};

template <unsigned Opcode>
struct InstructionIO : public InstrumentationOpportunity {
  InstructionIO(bool IsPRE)
      : InstrumentationOpportunity(InstrumentationLocation(Opcode, IsPRE)) {}
  virtual ~InstructionIO() {}

  unsigned getOpcode() const { return Opcode; }

  StringRef getName() const override {
    return Instruction::getOpcodeName(Opcode);
  }
};

struct StoreIO : public InstructionIO<Instruction::Store> {
  StoreIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~StoreIO() {};

  enum ConfigKind {
    PassPointer = 0,
    ReplacePointer,
    PassPointerAS,
    PassStoredValue,
    PassStoredValueSize,
    PassAlignment,
    PassValueTypeId,
    PassAtomicityOrdering,
    PassSyncScopeId,
    PassIsVolatile,
    PassId,
    NumConfig,
  };

  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr) {
    if (UserConfig)
      Config = *UserConfig;

    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (Config.has(PassPointer))
      IRTArgs.push_back(
          IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
                 ((IsPRE && Config.has(ReplacePointer)) ? IRTArg::REPLACABLE
                                                        : IRTArg::NONE),
                 getPointer, setPointer));
    if (Config.has(PassPointerAS))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                               "The address space of the accessed pointer.",
                               IRTArg::NONE, getPointerAS));
    if (Config.has(PassStoredValue))
      IRTArgs.push_back(
          IRTArg(getValueType(IIRB.Ctx), "value", "The stored value.",
                 IRTArg::POTENTIALLY_INDIRECT | (Config.has(PassStoredValueSize)
                                                     ? IRTArg::INDIRECT_HAS_SIZE
                                                     : IRTArg::NONE),
                 getValue));
    if (Config.has(PassStoredValueSize))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "value_size",
                               "The size of the stored value.", IRTArg::NONE,
                               getValueSize));
    if (Config.has(PassAlignment))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                               "The known access alignment.", IRTArg::NONE,
                               getAlignment));
    if (Config.has(PassValueTypeId))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                               "The type id of the stored value.", IRTArg::NONE,
                               getValueTypeId));
    if (Config.has(PassAtomicityOrdering))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                               "The atomicity ordering of the store.",
                               IRTArg::NONE, getAtomicityOrdering));
    if (Config.has(PassSyncScopeId))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                               "The sync scope id of the store.", IRTArg::NONE,
                               getSyncScopeId));
    if (Config.has(PassIsVolatile))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                               "Flag indicating a volatile store.",
                               IRTArg::NONE, isVolatile));

    addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
    IConf.addChoice(*this);
  }

  static Value *getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *getPointerAS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getValueTypeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *getAtomicityOrdering(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB);
  static Value *getSyncScopeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *isVolatile(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<StoreIO>(IsPRE);
      AIC->init(IConf, IIRB);
    }
  }
};

struct LoadIO : public InstructionIO<Instruction::Load> {
  LoadIO(bool IsPRE) : InstructionIO(IsPRE) {}
  virtual ~LoadIO() {};

  enum ConfigKind {
    PassPointer = 0,
    ReplacePointer,
    PassPointerAS,
    PassValue,
    ReplaceValue,
    PassValueSize,
    PassAlignment,
    PassValueTypeId,
    PassAtomicityOrdering,
    PassSyncScopeId,
    PassIsVolatile,
    PassId,
    NumConfig,
  };

  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;

  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr) {
    bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
    if (UserConfig)
      Config = *UserConfig;
    if (Config.has(PassPointer))
      IRTArgs.push_back(
          IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
                 ((IsPRE && Config.has(ReplacePointer)) ? IRTArg::REPLACABLE
                                                        : IRTArg::NONE),
                 getPointer, setPointer));
    if (Config.has(PassPointerAS))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                               "The address space of the accessed pointer.",
                               IRTArg::NONE, getPointerAS));
    if (!IsPRE && Config.has(PassValue))
      IRTArgs.push_back(IRTArg(
          getValueType(IIRB.Ctx), "value", "The loaded value.",
          Config.has(ReplaceValue)
              ? IRTArg::REPLACABLE | IRTArg::POTENTIALLY_INDIRECT |
                    (Config.has(PassValueSize) ? IRTArg::INDIRECT_HAS_SIZE
                                               : IRTArg::NONE)
              : IRTArg::NONE,
          getValue, Config.has(ReplaceValue) ? replaceValue : nullptr));
    if (Config.has(PassValueSize))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "value_size",
                               "The size of the loaded value.", IRTArg::NONE,
                               getValueSize));
    if (Config.has(PassAlignment))
      IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                               "The known access alignment.", IRTArg::NONE,
                               getAlignment));
    if (Config.has(PassValueTypeId))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                               "The type id of the loaded value.", IRTArg::NONE,
                               getValueTypeId));
    if (Config.has(PassAtomicityOrdering))
      IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                               "The atomicity ordering of the load.",
                               IRTArg::NONE, getAtomicityOrdering));
    if (Config.has(PassSyncScopeId))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                               "The sync scope id of the load.", IRTArg::NONE,
                               getSyncScopeId));
    if (Config.has(PassIsVolatile))
      IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                               "Flag indicating a volatile load.", IRTArg::NONE,
                               isVolatile));
    addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
    IConf.addChoice(*this);
  }

  static Value *getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);
  static Value *getPointerAS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB);
  static Value *getValueTypeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *getAtomicityOrdering(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB);
  static Value *getSyncScopeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB);
  static Value *isVolatile(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB);

  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<LoadIO>(IsPRE);
      AIC->init(IConf, IIRB);
    }
  }
};

} // namespace instrumentor

class InstrumentorPass : public PassInfoMixin<InstrumentorPass> {
  using InstrumentationConfig = instrumentor::InstrumentationConfig;
  using InstrumentorIRBuilderTy = instrumentor::InstrumentorIRBuilderTy;
  InstrumentationConfig *UserIConf;
  InstrumentorIRBuilderTy *UserIIRB;

  PreservedAnalyses run(Module &M, FunctionAnalysisManager &FAM,
                        InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB);

public:
  InstrumentorPass(InstrumentationConfig *IC = nullptr,
                   InstrumentorIRBuilderTy *IIRB = nullptr)
      : UserIConf(IC), UserIIRB(IIRB) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INSTRUMENTOR_H
