//===- Transforms/IPO/Instrumentor.h --------------------------------------===//
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
#include "llvm/Transforms/IPO/InstrumentorUtils.h"
#include "llvm/Transforms/Utils/Instrumentation.h"

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>

namespace llvm {
namespace instrumentor {

struct InstrumentationConfig;
struct InstrumentationOpportunity;

/// Callback type for getting/setting a value for a instrumented opportunity.
///{
using GetterCallbackTy = std::function<Value *(
    Value &, Type &, InstrumentationConfig &, InstrumentorIRBuilderTy &)>;
using SetterCallbackTy = std::function<Value *(
    Value &, Value &, InstrumentationConfig &, InstrumentorIRBuilderTy &)>;
///}

/// Helper to represent an argument to a instrumentation runtime function.
struct IRTArg {
  /// Flags describing the possible properties of an argument.
  enum IRArgFlagTy {
    NONE = 0,
    STRING = 1 << 0,
    REPLACABLE = 1 << 1,
    REPLACABLE_CUSTOM = 1 << 2,
    POTENTIALLY_INDIRECT = 1 << 3,
    INDIRECT_HAS_SIZE = 1 << 4,
    LAST,
  };

  /// Construct an argument.
  IRTArg(Type *Ty, StringRef Name, StringRef Description, unsigned Flags,
         GetterCallbackTy GetterCB, SetterCallbackTy SetterCB = nullptr,
         bool Enabled = true, bool NoCache = false)
      : Enabled(Enabled), Ty(Ty), Name(Name), Description(Description),
        Flags(Flags), GetterCB(std::move(GetterCB)),
        SetterCB(std::move(SetterCB)), NoCache(NoCache) {}

  /// Whether the argument is enabled and should be passed to the function call.
  bool Enabled;

  /// The type of the argument.
  Type *Ty;

  /// A string with the name of the argument.
  StringRef Name;

  /// A string with the description of the argument.
  StringRef Description;

  /// The flags that describe the properties of the argument. Multiple flags may
  /// be specified.
  unsigned Flags;

  /// The callback for getting the value of the argument.
  GetterCallbackTy GetterCB;

  /// The callback for consuming the output value of the argument.
  SetterCallbackTy SetterCB;

  /// Whether the argument value can be cached between the PRE and POST calls.
  bool NoCache;
};

/// Helper to represent an instrumentation runtime function that is related to
/// an instrumentation opportunity.
struct IRTCallDescription {
  /// Construct an instrumentation function description linked to the \p IO
  /// instrumentation opportunity and \p RetTy return type.
  IRTCallDescription(InstrumentationOpportunity &IO, Type *RetTy = nullptr);

  /// Create the type of the instrumentation function.
  FunctionType *createLLVMSignature(InstrumentationConfig &IConf,
                                    LLVMContext &Ctx, const DataLayout &DL,
                                    bool ForceIndirection);

  /// Create a call instruction that calls to the instrumentation function and
  /// passes the corresponding arguments.
  CallInst *createLLVMCall(Value *&V, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB, const DataLayout &DL,
                           InstrumentationCaches &ICaches);

  /// Create a string representation of the function declaration in C. Two
  /// strings are returned: the function definition with direct arguments and
  /// the function with any indirect argument.
  std::pair<std::string, std::string>
  createCSignature(const InstrumentationConfig &IConf) const;

  /// Create a string representation of the function definition in C. The
  /// function body implements a stub and only prints the passed arguments. Two
  /// strings are returned: the function definition with direct arguments and
  /// the function with any indirect argument.
  std::pair<std::string, std::string> createCBodies() const;

  /// Return whether the \p IRTA argument can be replaced.
  bool isReplacable(IRTArg &IRTA) const {
    return (IRTA.Flags & (IRTArg::REPLACABLE | IRTArg::REPLACABLE_CUSTOM));
  }

  /// Return whether the function may have any indirect argument.
  bool isPotentiallyIndirect(IRTArg &IRTA) const {
    return ((IRTA.Flags & IRTArg::POTENTIALLY_INDIRECT) ||
            ((IRTA.Flags & IRTArg::REPLACABLE) && NumReplaceableArgs > 1));
  }

  /// Whether the function requires indirection in some argument.
  bool RequiresIndirection = false;

  /// Whether any argument may require indirection.
  bool MightRequireIndirection = false;

  /// The number of arguments that can be replaced.
  unsigned NumReplaceableArgs = 0;

  /// The instrumentation opportunity which it is linked to.
  InstrumentationOpportunity &IO;

  /// The return type of the instrumentation function.
  Type *RetTy = nullptr;
};

/// Helper to represent an instrumentation location, which is composed of an
/// instrumentation opportunity type and a position.
struct InstrumentationLocation {
  /// The supported location kinds, which are composed of a opportunity type and
  /// position. The PRE position indicates the instrumentation function call is
  /// inserted before the instrumented event occurs. The POST position indicates
  /// the instrumentation call is inserted after the event occurs. Some
  /// opportunity types may only support one position.
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
    Last = INSTRUCTION_POST,
  };

  /// Construct an instrumentation location that is not instrumenting an
  /// instruction.
  InstrumentationLocation(KindTy Kind) : Kind(Kind) {
    assert(Kind != INSTRUCTION_PRE && Kind != INSTRUCTION_POST &&
           "Opcode required!");
  }

  /// Construct an instrumentation location belonging to the instrumentation of
  /// an instruction.
  InstrumentationLocation(unsigned Opcode, bool IsPRE)
      : Kind(IsPRE ? INSTRUCTION_PRE : INSTRUCTION_POST), Opcode(Opcode) {}

  /// Return the type and position.
  KindTy getKind() const { return Kind; }

  /// Return the string representation given a location kind. This is the string
  /// used in the configuration file.
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
    }
    llvm_unreachable("Invalid kind!");
  }

  /// Return the location kind described by a string.
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
        .Default(Last);
  }

  /// Return whether a location kind is positioned before the event occurs.
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
      return false;
    }
    llvm_unreachable("Invalid kind!");
  }

  /// Return whether the instrumentation location is before the event occurs.
  bool isPRE() const { return isPRE(Kind); }

  /// Get the opcode of the instruction instrumentation location. This function
  /// may not be called by a non-instruction instrumentation location.
  unsigned getOpcode() const {
    assert((Kind == INSTRUCTION_PRE || Kind == INSTRUCTION_POST) &&
           "Expected instruction!");
    return Opcode;
  }

private:
  /// The kind (type and position) of the instrumentation location.
  const KindTy Kind;

  /// The opcode for instruction instrumentation locations.
  const unsigned Opcode = -1;
};

/// An option for the base configuration.
struct BaseConfigurationOption {
  /// The possible types of options.
  enum KindTy {
    STRING,
    BOOLEAN,
  };

  /// Create a boolean option with \p Name name, \p Description description and
  /// \p DefaultValue as boolean default value.
  static BaseConfigurationOption *getBoolOption(InstrumentationConfig &IC,
                                                StringRef Name,
                                                StringRef Description,
                                                bool DefaultValue);

  /// Create a string option with \p Name name, \p Description description and
  /// \p DefaultValue as string default value.
  static BaseConfigurationOption *getStringOption(InstrumentationConfig &IC,
                                                  StringRef Name,
                                                  StringRef Description,
                                                  StringRef DefaultValue);

  /// Helper union that holds any possible option type.
  union ValueTy {
    bool Bool;
    StringRef String;
  };

  /// Set and get of the boolean value. Only valid if it is a boolean option.
  ///{
  void setBool(bool B) {
    assert(Kind == BOOLEAN && "Not a boolean!");
    Value.Bool = B;
  }
  bool getBool() const {
    assert(Kind == BOOLEAN && "Not a boolean!");
    return Value.Bool;
  }
  ///}

  /// Set and get the string value. Only valid if it is a boolean option.
  ///{
  void setString(StringRef S) {
    assert(Kind == STRING && "Not a string!");
    Value.String = S;
  }
  StringRef getString() const {
    assert(Kind == STRING && "Not a string!");
    return Value.String;
  }
  ///}

  /// The information of the option.
  ///{
  StringRef Name;
  StringRef Description;
  KindTy Kind;
  ValueTy Value = {0};
  ///}
};

/// The class that contains the configuration for the instrumentor. It holds the
/// information for each instrumented opportunity, including the base
/// configuration options. Another class may inherit from this one to modify the
/// default behavior.
struct InstrumentationConfig {
  virtual ~InstrumentationConfig() {}

  /// Construct an instrumentation configuration with the base options.
  InstrumentationConfig() : SS(StringAllocator) {
    RuntimePrefix = BaseConfigurationOption::getStringOption(
        *this, "runtime_prefix", "The runtime API prefix.", "__instrumentor_");
    RuntimeStubsFile = BaseConfigurationOption::getStringOption(
        *this, "runtime_stubs_file",
        "The file into which runtime stubs should be written.", "test.c");
    TargetRegex = BaseConfigurationOption::getStringOption(
        *this, "target_regex",
        "Regular expression to be matched against the module target. "
        "Only targets that match this regex will be instrumented",
        "");
    HostEnabled = BaseConfigurationOption::getBoolOption(
        *this, "host_enabled", "Instrument non-GPU targets", true);
    GPUEnabled = BaseConfigurationOption::getBoolOption(
        *this, "gpu_enabled", "Instrument GPU targets", true);
  }

  /// Populate the instrumentation opportunities.
  virtual void populate(InstrumentorIRBuilderTy &IIRB);

  /// Get the runtime prefix for the instrumentation runtime functions.
  StringRef getRTName() const { return RuntimePrefix->getString(); }

  /// Get the instrumentation function name.
  std::string getRTName(StringRef Prefix, StringRef Name,
                        StringRef Suffix1 = "", StringRef Suffix2 = "") const {
    return (getRTName() + Prefix + Name + Suffix1 + Suffix2).str();
  }

  /// Add the base configuration option \p BCO into the list of base options.
  void addBaseChoice(BaseConfigurationOption *BCO) {
    BaseConfigurationOptions.push_back(BCO);
  }

  /// Register instrumentation opportunity \p IO.
  void addChoice(InstrumentationOpportunity &IO, LLVMContext &Ctx);

  /// Allocate an object of type \p Ty using a bump allocator and construct it
  /// with the \p Args arguments. The object may not be freed manually.
  template <typename Ty, typename... ArgsTy>
  static Ty *allocate(ArgsTy &&...Args) {
    static SpecificBumpPtrAllocator<Ty> Allocator;
    Ty *Obj = Allocator.Allocate();
    new (Obj) Ty(std::forward<ArgsTy>(Args)...);
    return Obj;
  }

  /// The list of enabled base configuration options.
  SmallVector<BaseConfigurationOption *> BaseConfigurationOptions;

  /// The base configuration options.
  BaseConfigurationOption *RuntimePrefix;
  BaseConfigurationOption *RuntimeStubsFile;
  BaseConfigurationOption *TargetRegex;
  BaseConfigurationOption *HostEnabled;
  BaseConfigurationOption *GPUEnabled;

  /// The map registered instrumentation opportunities. The map is indexed by
  /// the instrumentation location kind and then by the opportunity name. Notice
  /// that an instrumentation location may have more than one instrumentation
  /// opportunity registered.
  EnumeratedArray<StringMap<InstrumentationOpportunity *>,
                  InstrumentationLocation::KindTy>
      IChoices;

  /// Utilities for allocating and building strings.
  ///{
  BumpPtrAllocator StringAllocator;
  StringSaver SS;
  ///}
};

/// Base class for instrumentation opportunities. All opportunities should
/// inherit from this class and implement the virtual class members.
struct InstrumentationOpportunity {
  virtual ~InstrumentationOpportunity() {}

  /// Construct an opportunity with location \p IP.
  InstrumentationOpportunity(const InstrumentationLocation IP) : IP(IP) {}

  /// The instrumentation location of the opportunity.
  InstrumentationLocation IP;

  /// The list of possible arguments for the instrumentation runtime function.
  /// The order within the array determines the order of arguments. Arguments
  /// may be disabled and will not be passed to the function call.
  SmallVector<IRTArg> IRTArgs;

  /// Whether the opportunity is enabled.
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

  /// Instrument the value \p V using the configuration \p IConf, and
  /// potentially, the caches \p ICaches.
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

  /// Get the return type for the instrumentation runtime function.
  virtual Type *getRetTy(LLVMContext &Ctx) const { return nullptr; }

  /// Get the name of the instrumentation opportunity.
  virtual StringRef getName() const = 0;

  /// Get the opcode of the instruction instrumentation opportunity. Only valid
  /// if it is instruction instrumentation.
  unsigned getOpcode() const { return IP.getOpcode(); }

  /// Get the location kind of the instrumentation opportunity.
  InstrumentationLocation::KindTy getLocationKind() const {
    return IP.getKind();
  }

  /// An optional callback that takes the value that is about to be
  /// instrumented and can return false if it should be skipped.
  ///{
  using CallbackTy = std::function<bool(Value &)>;
  CallbackTy CB = nullptr;
  ///}

  /// Add arguments available in all instrumentation opportunities.
  void addCommonArgs(InstrumentationConfig &IConf, LLVMContext &Ctx,
                     bool PassId) {
    const auto CB = IP.isPRE() ? getIdPre : getIdPost;
    if (PassId) {
      IRTArgs.push_back(
          IRTArg(IntegerType::getInt32Ty(Ctx), "id",
                 "A unique ID associated with the given instrumentor call",
                 IRTArg::NONE, CB, nullptr, true, true));
    }
  }

  /// Get the opportunity identifier for the pre and post positions.
  ///{
  static Value *getIdPre(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB);
  static Value *getIdPost(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB);
  ///}

  /// Compute the opportunity identifier for the current instrumentation epoch
  /// \p CurrentEpoch. The identifiers are assigned consecutively as the epoch
  /// advances. Epochs may have no identifier assigned (e.g., because no id was
  /// requested). This function always returns the same identifier when called
  /// multiple times with the same epoch.
  static int32_t getIdFromEpoch(uint32_t CurrentEpoch) {
    static DenseMap<uint32_t, int32_t> EpochIdMap;
    static int32_t GlobalId = 0;
    int32_t &EpochId = EpochIdMap[CurrentEpoch];
    if (EpochId == 0)
      EpochId = ++GlobalId;
    return EpochId;
  }
};

/// The base instrumentation opportunity class for instruction opportunities.
/// Each instruction opportunity should inherit from this class and implement
/// the virtual class members.
template <unsigned Opcode>
struct InstructionIO : public InstrumentationOpportunity {
  virtual ~InstructionIO() {}

  /// Construct an instruction opportunity.
  InstructionIO(bool IsPRE)
      : InstrumentationOpportunity(InstrumentationLocation(Opcode, IsPRE)) {}

  /// Get the name of the instruction.
  StringRef getName() const override {
    return Instruction::getOpcodeName(Opcode);
  }
};

/// The instrumentation opportunity for store instructions.
struct StoreIO : public InstructionIO<Instruction::Store> {
  virtual ~StoreIO() {};

  /// Construct a store instruction opportunity.
  StoreIO(bool IsPRE) : InstructionIO(IsPRE) {}

  /// The selector of arguments for store opportunities.
  ///{
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

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;
  ///}

  /// Get the type of the stored value.
  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  /// Initialize the store opportunity using the instrumentation config \p IConf
  /// and the user config \p UserConfig.
  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr);

  /// Getters and setters for the arguments of the instrumentation function for
  /// the store opportunity.
  ///{
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
  ///}

  /// Create the store opportunities for pre and post positions. The
  /// opportunities are also initialized with the arguments for their
  /// instrumentation calls.
  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<StoreIO>(IsPRE);
      AIC->init(IConf, IIRB);
    }
  }
};

/// The instrumentation opportunity for load instructions.
struct LoadIO : public InstructionIO<Instruction::Load> {
  virtual ~LoadIO() {};

  /// Construct a load opportunity.
  LoadIO(bool IsPRE) : InstructionIO(IsPRE) {}

  /// The selector of arguments for load opportunities.
  ///{
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

  using ConfigTy = BaseConfigTy<ConfigKind>;
  ConfigTy Config;
  ///}

  /// Get the type of the loaded value.
  virtual Type *getValueType(LLVMContext &Ctx) const {
    return IntegerType::getInt64Ty(Ctx);
  }

  /// Initialize the load opportunity using the instrumentation config \p IConf
  /// and the user config \p UserConfig.
  void init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
            ConfigTy *UserConfig = nullptr);

  /// Getters and setters for the arguments of the instrumentation function for
  /// the load opportunity.
  ///{
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
  ///}

  /// Create the store opportunities for PRE and POST positions.
  static void populate(InstrumentationConfig &IConf,
                       InstrumentorIRBuilderTy &IIRB) {
    for (auto IsPRE : {true, false}) {
      auto *AIC = IConf.allocate<LoadIO>(IsPRE);
      AIC->init(IConf, IIRB);
    }
  }
};

} // namespace instrumentor

/// The Instrumentor pass.
class InstrumentorPass : public PassInfoMixin<InstrumentorPass> {
  using InstrumentationConfig = instrumentor::InstrumentationConfig;
  using InstrumentorIRBuilderTy = instrumentor::InstrumentorIRBuilderTy;

  /// The configuration and IR builder provided by the user.
  InstrumentationConfig *UserIConf;
  InstrumentorIRBuilderTy *UserIIRB;

  PreservedAnalyses run(Module &M, InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB, bool ReadConfig);

public:
  /// Construct an instrumentor pass that will use the instrumentation
  /// configuration \p IC and the IR builder \p IIRB. If an IR builder is not
  /// provided, a default builder is used. When the configuration is not
  /// provided, it is read from the config file if available and otherwise a
  /// default configuration is used.
  InstrumentorPass(InstrumentationConfig *IC = nullptr,
                   InstrumentorIRBuilderTy *IIRB = nullptr)
      : UserIConf(IC), UserIIRB(IIRB) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INSTRUMENTOR_H
