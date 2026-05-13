//===-- Instrumentor.cpp - Highly configurable instrumentation pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The implementation of the Instrumentor, a highly configurable instrumentation
// pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Instrumentor.h"
#include "llvm/Transforms/IPO/InstrumentorConfigFile.h"
#include "llvm/Transforms/IPO/InstrumentorStubPrinter.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <type_traits>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "instrumentor"

namespace {

/// The user option to specify an output JSON file to write the configuration.
static cl::opt<std::string> WriteConfigFile(
    "instrumentor-write-config-file",
    cl::desc(
        "Write the instrumentor configuration into the specified JSON file"),
    cl::init(""));

/// The user option to specify an input JSON file to read the configuration.
static cl::opt<std::string> ReadConfigFile(
    "instrumentor-read-config-file",
    cl::desc(
        "Read the instrumentor configuration from the specified JSON file"),
    cl::init(""));

/// Set the debug location, if not set, after changing the insertion point of
/// the IR builder \p IRB.
template <typename IRBuilderTy> void ensureDbgLoc(IRBuilderTy &IRB) {
  if (IRB.getCurrentDebugLocation())
    return;
  auto *BB = IRB.GetInsertBlock();
  if (auto *SP = BB->getParent()->getSubprogram())
    IRB.SetCurrentDebugLocation(DILocation::get(BB->getContext(), 0, 0, SP));
}

/// Attempt to cast \p V to type \p Ty.
template <typename IRBTy>
Value *tryToCast(IRBTy &IRB, Value *V, Type *Ty, const DataLayout &DL,
                 bool AllowTruncate = false) {
  if (!V)
    return Constant::getAllOnesValue(Ty);
  Type *VTy = V->getType();
  if (VTy == Ty)
    return V;
  if (VTy->isAggregateType())
    return V;
  TypeSize RequestedSize = DL.getTypeSizeInBits(Ty);
  TypeSize ValueSize = DL.getTypeSizeInBits(VTy);
  bool ShouldTruncate = RequestedSize < ValueSize;
  if (ShouldTruncate && !AllowTruncate)
    return V;
  if (ShouldTruncate && AllowTruncate)
    return tryToCast(IRB,
                     IRB.CreateIntCast(V, IRB.getIntNTy(RequestedSize),
                                       /*IsSigned=*/false),
                     Ty, DL, AllowTruncate);
  if (VTy->isPointerTy() && Ty->isPointerTy())
    return IRB.CreatePointerBitCastOrAddrSpaceCast(V, Ty);
  if (VTy->isIntegerTy() && Ty->isIntegerTy())
    return IRB.CreateIntCast(V, Ty, /*IsSigned=*/false);
  if (VTy->isFloatingPointTy() && Ty->isIntOrPtrTy()) {
    return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getIntNTy(ValueSize)), Ty,
                     DL, AllowTruncate);
  }
  return IRB.CreateBitOrPointerCast(V, Ty);
}

/// Get a constant integer/boolean of type \p IT and value \p Val.
template <typename Ty>
Constant *getCI(Type *IT, Ty Val, bool IsSigned = false) {
  return ConstantInt::get(IT, Val, IsSigned);
}

/// The core of the instrumentor pass, which instruments the module as the
/// instrumentation configuration mandates.
class InstrumentorImpl final {
public:
  /// Construct an instrumentor implementation using the configuration \p IConf.
  InstrumentorImpl(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
                   Module &M)
      : IConf(IConf), M(M), IIRB(IIRB) {
    IConf.populate(IIRB);
  }

  /// Instrument the module, public entry point.
  bool instrument();

private:
  /// Indicate if the module should be instrumented based on the target.
  bool shouldInstrumentTarget();

  /// Indicate if the function \p Fn should be instrumented.
  bool shouldInstrumentFunction(Function &Fn);

  /// Instrument instruction \p I if needed, and use the argument caches in \p
  /// ICaches.
  bool instrumentInstruction(Instruction &I, InstrumentationCaches &ICaches);

  /// Instrument function \p Fn.
  bool instrumentFunction(Function &Fn);

  /// The instrumentation opportunities for instructions indexed by
  /// their opcode.
  DenseMap<unsigned, InstrumentationOpportunity *> InstChoicesPRE,
      InstChoicesPOST;

  /// The instrumentor configuration.
  InstrumentationConfig &IConf;

  /// The underlying module.
  Module &M;

protected:
  /// A special IR builder that keeps track of the inserted instructions.
  InstrumentorIRBuilderTy &IIRB;
};

} // end anonymous namespace

bool InstrumentorImpl::shouldInstrumentTarget() {
  const Triple &T = M.getTargetTriple();
  const bool IsGPU = T.isAMDGPU() || T.isNVPTX();

  bool RegexMatches = true;
  const auto TargetRegexStr = IConf.TargetRegex->getString();
  if (!TargetRegexStr.empty()) {
    llvm::Regex TargetRegex(TargetRegexStr);
    std::string ErrMsg;
    if (!TargetRegex.isValid(ErrMsg)) {
      IIRB.Ctx.diagnose(DiagnosticInfoInstrumentation(
          Twine("failed to parse target regex: ") + ErrMsg, DS_Warning));
      return false;
    }
    RegexMatches = TargetRegex.match(T.str());
  }

  // Only instrument the module if the target has to be instrumented.
  return ((IsGPU && IConf.GPUEnabled->getBool()) ||
          (!IsGPU && IConf.HostEnabled->getBool())) &&
         RegexMatches;
}

bool InstrumentorImpl::shouldInstrumentFunction(Function &Fn) {
  if (Fn.isDeclaration())
    return false;
  return !Fn.getName().starts_with(IConf.getRTName()) ||
         Fn.hasFnAttribute("instrument");
}

bool InstrumentorImpl::instrumentInstruction(Instruction &I,
                                             InstrumentationCaches &ICaches) {
  bool Changed = false;

  // Skip instrumentation instructions.
  if (IIRB.NewInsts.contains(&I))
    return Changed;

  // Count epochs eagerly.
  ++IIRB.Epoch;

  Value *IPtr = &I;
  if (auto *IO = InstChoicesPRE.lookup(I.getOpcode())) {
    IIRB.IRB.SetInsertPoint(&I);
    ensureDbgLoc(IIRB.IRB);
    Changed |= bool(IO->instrument(IPtr, IConf, IIRB, ICaches));
  }

  if (auto *IO = InstChoicesPOST.lookup(I.getOpcode())) {
    IIRB.IRB.SetInsertPoint(I.getNextNode());
    ensureDbgLoc(IIRB.IRB);
    Changed |= bool(IO->instrument(IPtr, IConf, IIRB, ICaches));
  }
  IIRB.returnAllocas();

  return Changed;
}

bool InstrumentorImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;
  if (!shouldInstrumentFunction(Fn))
    return Changed;

  InstrumentationCaches ICaches;
  SmallVector<Instruction *> FinalTIs;
  ReversePostOrderTraversal<Function *> RPOT(&Fn);
  for (auto &It : RPOT) {
    for (auto &I : *It)
      Changed |= instrumentInstruction(I, ICaches);

    auto *TI = It->getTerminator();
    if (!TI->getNumSuccessors())
      FinalTIs.push_back(TI);
  }

  Value *FPtr = &Fn;
  for (auto &[Name, IO] :
       IConf.IChoices[InstrumentationLocation::FUNCTION_PRE]) {
    if (!IO->Enabled)
      continue;
    // Count epochs eagerly.
    ++IIRB.Epoch;

    IIRB.IRB.SetInsertPoint(
        cast<Function>(FPtr)->getEntryBlock().getFirstInsertionPt());
    ensureDbgLoc(IIRB.IRB);
    Changed |= bool(IO->instrument(FPtr, IConf, IIRB, ICaches));
    IIRB.returnAllocas();
  }

  for (auto &[Name, IO] :
       IConf.IChoices[InstrumentationLocation::FUNCTION_POST]) {
    if (!IO->Enabled)
      continue;
    // Count epochs eagerly.
    ++IIRB.Epoch;

    for (Instruction *FinalTI : FinalTIs) {
      IIRB.IRB.SetInsertPoint(FinalTI);
      ensureDbgLoc(IIRB.IRB);
      Changed |= bool(IO->instrument(FPtr, IConf, IIRB, ICaches));
      IIRB.returnAllocas();
    }
  }
  return Changed;
}

bool InstrumentorImpl::instrument() {
  bool Changed = false;
  if (!shouldInstrumentTarget())
    return Changed;

  for (auto &[Name, IO] :
       IConf.IChoices[InstrumentationLocation::INSTRUCTION_PRE])
    if (IO->Enabled)
      InstChoicesPRE[IO->getOpcode()] = IO;
  for (auto &[Name, IO] :
       IConf.IChoices[InstrumentationLocation::INSTRUCTION_POST])
    if (IO->Enabled)
      InstChoicesPOST[IO->getOpcode()] = IO;

  for (Function &Fn : M)
    Changed |= instrumentFunction(Fn);

  return Changed;
}

InstrumentorPass::InstrumentorPass(IntrusiveRefCntPtr<vfs::FileSystem> FS,
                                   InstrumentationConfig *IC,
                                   InstrumentorIRBuilderTy *IIRB)
    : FS(FS), UserIConf(IC), UserIIRB(IIRB) {
  if (!FS)
    this->FS = vfs::getRealFileSystem();
}

PreservedAnalyses InstrumentorPass::run(Module &M, InstrumentationConfig &IConf,
                                        InstrumentorIRBuilderTy &IIRB,
                                        bool ReadConfig) {
  InstrumentorImpl Impl(IConf, IIRB, M);
  if (ReadConfig && !readConfigFromJSON(IConf, ReadConfigFile, IIRB.Ctx, *FS))
    return PreservedAnalyses::all();

  writeConfigToJSON(IConf, WriteConfigFile, IIRB.Ctx);

  printRuntimeStub(IConf, IConf.RuntimeStubsFile->getString(), IIRB.Ctx);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

PreservedAnalyses InstrumentorPass::run(Module &M, ModuleAnalysisManager &MAM) {
  // Only create them if the user did not provide them.
  std::unique_ptr<InstrumentationConfig> IConfInt(
      !UserIConf ? new InstrumentationConfig() : nullptr);
  std::unique_ptr<InstrumentorIRBuilderTy> IIRBInt(
      !UserIIRB ? new InstrumentorIRBuilderTy(M) : nullptr);

  auto *IConf = IConfInt ? IConfInt.get() : UserIConf;
  auto *IIRB = IIRBInt ? IIRBInt.get() : UserIIRB;

  auto PA = run(M, *IConf, *IIRB, !UserIConf);

  assert(!verifyModule(M, &errs()));
  return PA;
}

std::unique_ptr<BaseConfigurationOption>
BaseConfigurationOption::createBoolOption(InstrumentationConfig &IConf,
                                          StringRef Name, StringRef Description,
                                          bool DefaultValue) {
  auto BCO =
      std::make_unique<BaseConfigurationOption>(Name, Description, BOOLEAN);
  BCO->setBool(DefaultValue);
  IConf.addBaseChoice(BCO.get());
  return BCO;
}

std::unique_ptr<BaseConfigurationOption>
BaseConfigurationOption::createStringOption(InstrumentationConfig &IConf,
                                            StringRef Name,
                                            StringRef Description,
                                            StringRef DefaultValue) {
  auto BCO =
      std::make_unique<BaseConfigurationOption>(Name, Description, STRING);
  BCO->setString(DefaultValue);
  IConf.addBaseChoice(BCO.get());
  return BCO;
}

void InstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  /// List of all instrumentation opportunities.
  FunctionIO::populate(*this, IIRB);
  AllocaIO::populate(*this, IIRB);
  LoadIO::populate(*this, IIRB);
  StoreIO::populate(*this, IIRB);
}

void InstrumentationConfig::addChoice(InstrumentationOpportunity &IO,
                                      LLVMContext &Ctx) {
  auto *&ICPtr = IChoices[IO.getLocationKind()][IO.getName()];
  if (ICPtr) {
    Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("registered two instrumentation opportunities for the same "
              "location (") +
            ICPtr->getName() + Twine(" vs ") + IO.getName() + Twine(")"),
        DS_Warning));
  }
  ICPtr = &IO;
}

Value *InstrumentationOpportunity::getIdPre(Value &V, Type &Ty,
                                            InstrumentationConfig &IConf,
                                            InstrumentorIRBuilderTy &IIRB) {
  return getCI(&Ty, getIdFromEpoch(IIRB.Epoch));
}

Value *InstrumentationOpportunity::getIdPost(Value &V, Type &Ty,
                                             InstrumentationConfig &IConf,
                                             InstrumentorIRBuilderTy &IIRB) {
  return getCI(&Ty, -getIdFromEpoch(IIRB.Epoch), /*IsSigned=*/true);
}

Value *InstrumentationOpportunity::forceCast(Value &V, Type &Ty,
                                             InstrumentorIRBuilderTy &IIRB) {
  if (V.getType()->isVoidTy())
    return Ty.isVoidTy() ? &V : Constant::getNullValue(&Ty);
  return tryToCast(IIRB.IRB, &V, &Ty,
                   IIRB.IRB.GetInsertBlock()->getDataLayout());
}

Value *InstrumentationOpportunity::replaceValue(Value &V, Value &NewV,
                                                InstrumentationConfig &IConf,
                                                InstrumentorIRBuilderTy &IIRB) {
  if (V.getType()->isVoidTy())
    return &V;

  auto *NewVCasted = &NewV;
  if (auto *I = dyn_cast<Instruction>(&NewV)) {
    IRBuilderBase::InsertPointGuard IPG(IIRB.IRB);
    IIRB.IRB.SetInsertPoint(I->getNextNode());
    ensureDbgLoc(IIRB.IRB);
    NewVCasted = tryToCast(IIRB.IRB, &NewV, V.getType(), IIRB.DL,
                           /*AllowTruncate=*/true);
  }
  V.replaceUsesWithIf(NewVCasted, [&](Use &U) {
    if (IIRB.NewInsts.lookup(cast<Instruction>(U.getUser())) == IIRB.Epoch)
      return false;
    return !isa<LifetimeIntrinsic>(U.getUser()) && !U.getUser()->isDroppable();
  });

  return &V;
}

IRTCallDescription::IRTCallDescription(InstrumentationOpportunity &IO,
                                       Type *RetTy)
    : IO(IO), RetTy(RetTy) {
  for (auto &It : IO.IRTArgs) {
    if (!It.Enabled)
      continue;
    NumReplaceableArgs += bool(It.Flags & IRTArg::REPLACABLE);
    MightRequireIndirection |= It.Flags & IRTArg::POTENTIALLY_INDIRECT;
  }
  if (NumReplaceableArgs > 1)
    MightRequireIndirection = RequiresIndirection = true;
}

FunctionType *IRTCallDescription::createLLVMSignature(
    InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
    const DataLayout &DL, bool ForceIndirection) {
  assert(((ForceIndirection && MightRequireIndirection) ||
          (!ForceIndirection && !RequiresIndirection)) &&
         "Wrong indirection setting!");

  SmallVector<Type *> ParamTypes;
  for (auto &It : IO.IRTArgs) {
    if (!It.Enabled)
      continue;
    if (!ForceIndirection || !isPotentiallyIndirect(It)) {
      ParamTypes.push_back(It.Ty);
      if (!RetTy && NumReplaceableArgs == 1 && (It.Flags & IRTArg::REPLACABLE))
        RetTy = It.Ty;
      continue;
    }

    // The indirection pointer and the size of the value.
    ParamTypes.push_back(IIRB.PtrTy);
    if (!(It.Flags & IRTArg::INDIRECT_HAS_SIZE))
      ParamTypes.push_back(IIRB.Int32Ty);
  }
  if (!RetTy)
    RetTy = IIRB.VoidTy;

  return FunctionType::get(RetTy, ParamTypes, /*isVarArg=*/false);
}

CallInst *IRTCallDescription::createLLVMCall(Value *&V,
                                             InstrumentationConfig &IConf,
                                             InstrumentorIRBuilderTy &IIRB,
                                             const DataLayout &DL,
                                             InstrumentationCaches &ICaches) {
  SmallVector<Value *> CallParams;

  IRBuilderBase::InsertPointGuard IRP(IIRB.IRB);
  auto IP = IIRB.IRB.GetInsertPoint();

  bool ForceIndirection = RequiresIndirection;
  for (auto &It : IO.IRTArgs) {
    if (!It.Enabled)
      continue;
    auto *&Param = ICaches.DirectArgCache[{IIRB.Epoch, IO.getName(), It.Name}];
    if (!Param || It.NoCache)
      // Avoid passing the caches to the getter.
      Param = It.GetterCB(*V, *It.Ty, IConf, IIRB);
    assert(Param);

    if (Param->getType()->isVoidTy()) {
      Param = Constant::getNullValue(It.Ty);
    } else if (Param->getType()->isAggregateType() ||
               DL.getTypeSizeInBits(Param->getType()) >
                   DL.getTypeSizeInBits(It.Ty)) {
      if (!isPotentiallyIndirect(It)) {
        IIRB.Ctx.diagnose(DiagnosticInfoInstrumentation(
            Twine("indirection needed for ") + It.Name + Twine(" in ") +
                IO.getName() +
                Twine(", but not indicated. Instrumentation is skipped"),
            DS_Warning));
        return nullptr;
      }
      ForceIndirection = true;
    } else {
      Param = tryToCast(IIRB.IRB, Param, It.Ty, DL);
    }
    CallParams.push_back(Param);
  }

  if (ForceIndirection) {
    Function *Fn = IIRB.IRB.GetInsertBlock()->getParent();

    unsigned Offset = 0;
    for (auto &It : IO.IRTArgs) {
      if (!It.Enabled)
        continue;

      if (!isPotentiallyIndirect(It)) {
        ++Offset;
        continue;
      }
      auto *&CallParam = CallParams[Offset++];
      if (!(It.Flags & IRTArg::INDIRECT_HAS_SIZE)) {
        CallParams.insert(&CallParam + 1, IIRB.IRB.getInt32(DL.getTypeStoreSize(
                                              CallParam->getType())));
        Offset += 1;
      }

      auto *&CachedParam =
          ICaches.IndirectArgCache[{IIRB.Epoch, IO.getName(), It.Name}];
      if (CachedParam) {
        CallParam = CachedParam;
        continue;
      }

      auto *AI = IIRB.getAlloca(Fn, CallParam->getType());
      IIRB.IRB.CreateStore(CallParam, AI);
      CallParam = CachedParam = AI;
    }
  }

  if (!ForceIndirection)
    IIRB.IRB.SetInsertPoint(IP);
  ensureDbgLoc(IIRB.IRB);

  auto *FnTy = createLLVMSignature(IConf, IIRB, DL, ForceIndirection);
  auto CompleteName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(),
                      ForceIndirection ? "_ind" : "");
  auto FC = IIRB.IRB.GetInsertBlock()->getModule()->getOrInsertFunction(
      CompleteName, FnTy);
  auto *CI = IIRB.IRB.CreateCall(FC, CallParams);
  CI->addFnAttr(Attribute::get(IIRB.Ctx, Attribute::WillReturn));

  for (unsigned I = 0, E = IO.IRTArgs.size(); I < E; ++I) {
    if (!IO.IRTArgs[I].Enabled)
      continue;
    if (!isReplacable(IO.IRTArgs[I]))
      continue;
    bool IsCustomReplaceable = IO.IRTArgs[I].Flags & IRTArg::REPLACABLE_CUSTOM;
    Value *NewValue = FnTy->isVoidTy() || IsCustomReplaceable
                          ? ICaches.DirectArgCache[{IIRB.Epoch, IO.getName(),
                                                    IO.IRTArgs[I].Name}]
                          : CI;
    assert(NewValue);
    if (ForceIndirection && !IsCustomReplaceable &&
        isPotentiallyIndirect(IO.IRTArgs[I])) {
      auto *Q =
          ICaches
              .IndirectArgCache[{IIRB.Epoch, IO.getName(), IO.IRTArgs[I].Name}];
      NewValue = IIRB.IRB.CreateLoad(V->getType(), Q);
    }
    V = IO.IRTArgs[I].SetterCB(*V, *NewValue, IConf, IIRB);
  }
  return CI;
}

template <typename Ty> constexpr static Value *getValue(Ty &ValueOrUse) {
  if constexpr (std::is_same<Ty, Use>::value)
    return ValueOrUse.get();
  else
    return static_cast<Value *>(&ValueOrUse);
}

template <typename Range>
static Value *createValuePack(const Range &R, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  auto *Fn = IIRB.IRB.GetInsertBlock()->getParent();
  auto *I32Ty = IIRB.IRB.getInt32Ty();
  SmallVector<Constant *> ConstantValues;
  SmallVector<std::pair<Value *, uint32_t>> Values;
  SmallVector<Type *> Types;
  for (auto &RE : R) {
    Value *V = getValue(RE);
    if (!V->getType()->isSized())
      continue;
    auto VSize = IIRB.DL.getTypeAllocSize(V->getType());
    ConstantValues.push_back(getCI(I32Ty, VSize));
    Types.push_back(I32Ty);
    ConstantValues.push_back(getCI(I32Ty, V->getType()->getTypeID()));
    Types.push_back(I32Ty);
    if (uint32_t MisAlign = VSize % 8) {
      Types.push_back(ArrayType::get(IIRB.Int8Ty, 8 - MisAlign));
      ConstantValues.push_back(ConstantArray::getNullValue(Types.back()));
    }
    Types.push_back(V->getType());
    if (auto *C = dyn_cast<Constant>(V)) {
      ConstantValues.push_back(C);
      continue;
    }
    Values.push_back({V, ConstantValues.size()});
    ConstantValues.push_back(Constant::getNullValue(V->getType()));
  }
  if (Types.empty())
    return ConstantPointerNull::get(IIRB.PtrTy);

  StructType *STy = StructType::get(Fn->getContext(), Types, /*isPacked=*/true);
  Constant *Initializer = ConstantStruct::get(STy, ConstantValues);

  GlobalVariable *&GV = IConf.ConstantGlobalsCache[Initializer];
  if (!GV)
    GV = new GlobalVariable(*Fn->getParent(), STy, false,
                            GlobalValue::InternalLinkage, Initializer,
                            IConf.getRTName("", "value_pack"));

  auto *AI = IIRB.getAlloca(Fn, STy);
  IIRB.IRB.CreateMemCpy(AI, AI->getAlign(), GV, MaybeAlign(GV->getAlignment()),
                        IIRB.DL.getTypeAllocSize(STy));
  for (auto [Param, Idx] : Values) {
    auto *Ptr = IIRB.IRB.CreateStructGEP(STy, AI, Idx);
    IIRB.IRB.CreateStore(Param, Ptr);
  }
  return AI;
}

template <typename Range>
static void readValuePack(const Range &R, Value &Pack,
                          InstrumentorIRBuilderTy &IIRB,
                          function_ref<void(int, Value *)> SetterCB) {
  auto *Fn = IIRB.IRB.GetInsertBlock()->getParent();
  auto &DL = Fn->getDataLayout();
  SmallVector<Value *> ParameterValues;
  unsigned Offset = 0;
  for (const auto &[Idx, RE] : enumerate(R)) {
    Value *V = getValue(RE);
    if (!V->getType()->isSized())
      continue;
    Offset += 8;
    auto VSize = DL.getTypeAllocSize(V->getType());
    auto Padding = alignTo(VSize, 8) - VSize;
    Offset += Padding;
    auto *Ptr = IIRB.IRB.CreateConstInBoundsGEP1_32(IIRB.Int8Ty, &Pack, Offset);
    auto *NewV = IIRB.IRB.CreateLoad(V->getType(), Ptr);
    SetterCB(Idx, NewV);
    Offset += VSize;
  }
}

/// FunctionIO
/// {
void FunctionIO::init(InstrumentationConfig &IConf,
                      InstrumentorIRBuilderTy &IIRB, ConfigTy *UserConfig) {
  using namespace std::placeholders;
  if (UserConfig)
    Config = *UserConfig;

  bool IsPRE = getLocationKind() == InstrumentationLocation::FUNCTION_PRE;
  if (Config.has(PassAddress))
    IRTArgs.push_back(IRTArg(IIRB.PtrTy, "address", "The function address.",
                             IRTArg::NONE, getFunctionAddress));
  if (Config.has(PassName))
    IRTArgs.push_back(IRTArg(IIRB.PtrTy, "name", "The function name.",
                             IRTArg::STRING, getFunctionName));
  if (Config.has(PassNumArguments))
    IRTArgs.push_back(
        IRTArg(IIRB.Int32Ty, "num_arguments",
               "Number of function arguments (without varargs).", IRTArg::NONE,
               std::bind(&FunctionIO::getNumArguments, this, _1, _2, _3, _4)));
  if (Config.has(PassArguments))
    IRTArgs.push_back(
        IRTArg(IIRB.PtrTy, "arguments", "Description of the arguments.",
               IsPRE && Config.has(ReplaceArguments) ? IRTArg::REPLACABLE_CUSTOM
                                                     : IRTArg::NONE,
               std::bind(&FunctionIO::getArguments, this, _1, _2, _3, _4),
               std::bind(&FunctionIO::setArguments, this, _1, _2, _3, _4)));
  if (Config.has(PassIsMain))
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_main",
                             "Flag to indicate it is the main function.",
                             IRTArg::NONE, isMainFunction));
  addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
  IConf.addChoice(*this, IIRB.Ctx);
}

Value *FunctionIO::getFunctionAddress(Value &V, Type &Ty,
                                      InstrumentationConfig &IConf,
                                      InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  if (Fn.isIntrinsic())
    return Constant::getNullValue(&Ty);
  return &V;
}
Value *FunctionIO::getFunctionName(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  return IConf.getGlobalString(IConf.DemangleFunctionNames->getBool()
                                   ? demangle(Fn.getName())
                                   : Fn.getName(),
                               IIRB);
}
Value *FunctionIO::getNumArguments(Value &V, Type &Ty,
                                   InstrumentationConfig &IConf,
                                   InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  if (!Config.ArgFilter)
    return getCI(&Ty, Fn.arg_size());
  auto FRange = make_filter_range(Fn.args(), Config.ArgFilter);
  return getCI(&Ty, std::distance(FRange.begin(), FRange.end()));
}
Value *FunctionIO::getArguments(Value &V, Type &Ty,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  if (!Config.ArgFilter)
    return createValuePack(Fn.args(), IConf, IIRB);
  return createValuePack(make_filter_range(Fn.args(), Config.ArgFilter), IConf,
                         IIRB);
}
Value *FunctionIO::setArguments(Value &V, Value &NewV,
                                InstrumentationConfig &IConf,
                                InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  auto *AIt = Fn.arg_begin();
  auto CB = [&](int Idx, Value *ReplV) {
    while (Config.ArgFilter && !Config.ArgFilter(*AIt))
      ++AIt;
    Fn.getArg(Idx)->replaceUsesWithIf(ReplV, [&](Use &U) {
      return IIRB.NewInsts.lookup(cast<Instruction>(U.getUser())) != IIRB.Epoch;
    });
    ++AIt;
  };
  if (!Config.ArgFilter)
    readValuePack(Fn.args(), NewV, IIRB, CB);
  else
    readValuePack(make_filter_range(Fn.args(), Config.ArgFilter), NewV, IIRB,
                  CB);
  return &Fn;
}
Value *FunctionIO::isMainFunction(Value &V, Type &Ty,
                                  InstrumentationConfig &IConf,
                                  InstrumentorIRBuilderTy &IIRB) {
  auto &Fn = cast<Function>(V);
  return getCI(&Ty, Fn.getName() == "main");
}

///}

/// AllocaIO
///{
void AllocaIO::init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
                    ConfigTy *UserConfig) {
  if (UserConfig)
    Config = *UserConfig;

  bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
  if (!IsPRE && Config.has(PassAddress))
    IRTArgs.push_back(
        IRTArg(IIRB.PtrTy, "address", "The allocated memory address.",
               Config.has(ReplaceAddress) ? IRTArg::REPLACABLE : IRTArg::NONE,
               InstrumentationOpportunity::getValue,
               InstrumentationOpportunity::replaceValue));
  if (Config.has(PassSize))
    IRTArgs.push_back(IRTArg(
        IIRB.Int64Ty, "size", "The allocation size.",
        (IsPRE && Config.has(ReplaceSize)) ? IRTArg::REPLACABLE : IRTArg::NONE,
        getSize, setSize));
  if (Config.has(PassAlignment))
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                             "The allocation alignment.", IRTArg::NONE,
                             getAlignment));

  addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
  IConf.addChoice(*this, IIRB.Ctx);
}

Value *AllocaIO::getSize(Value &V, Type &Ty, InstrumentationConfig &IO,
                         InstrumentorIRBuilderTy &IIRB) {
  auto &AI = cast<AllocaInst>(V);
  const DataLayout &DL = AI.getDataLayout();
  Value *SizeValue = nullptr;
  TypeSize TypeSize = DL.getTypeAllocSize(AI.getAllocatedType());
  if (TypeSize.isFixed()) {
    SizeValue = getCI(&Ty, TypeSize.getFixedValue());
  } else {
    auto *NullPtr = ConstantPointerNull::get(AI.getType());
    SizeValue = IIRB.IRB.CreatePtrToInt(
        IIRB.IRB.CreateGEP(AI.getAllocatedType(), NullPtr,
                           {IIRB.IRB.getInt32(1)}),
        &Ty);
  }
  if (AI.isArrayAllocation())
    SizeValue = IIRB.IRB.CreateMul(
        SizeValue, IIRB.IRB.CreateZExtOrBitCast(AI.getArraySize(), &Ty));
  return SizeValue;
}

Value *AllocaIO::setSize(Value &V, Value &NewV, InstrumentationConfig &IO,
                         InstrumentorIRBuilderTy &IIRB) {
  auto &AI = cast<AllocaInst>(V);
  const DataLayout &DL = AI.getDataLayout();
  auto *NewAI = IIRB.IRB.CreateAlloca(IIRB.IRB.getInt8Ty(),
                                      DL.getAllocaAddrSpace(), &NewV);
  NewAI->setAlignment(AI.getAlign());
  AI.replaceAllUsesWith(NewAI);
  IIRB.eraseLater(&AI);
  return NewAI;
}

Value *AllocaIO::getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  return getCI(&Ty, cast<AllocaInst>(V).getAlign().value());
}
///}

void StoreIO::init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
                   ConfigTy *UserConfig) {
  if (UserConfig)
    Config = *UserConfig;

  bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
  if (Config.has(PassPointer)) {
    IRTArgs.push_back(
        IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
               ((IsPRE && Config.has(ReplacePointer)) ? IRTArg::REPLACABLE
                                                      : IRTArg::NONE),
               getPointer, setPointer));
  }
  if (Config.has(PassPointerAS)) {
    IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                             "The address space of the accessed pointer.",
                             IRTArg::NONE, getPointerAS));
  }
  if (Config.has(PassStoredValue)) {
    IRTArgs.push_back(
        IRTArg(getValueType(IIRB), "value", "The stored value.",
               IRTArg::POTENTIALLY_INDIRECT |
                   (Config.has(PassStoredValueSize) ? IRTArg::INDIRECT_HAS_SIZE
                                                    : IRTArg::NONE),
               getValue));
  }
  if (Config.has(PassStoredValueSize)) {
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "value_size",
                             "The size of the stored value.", IRTArg::NONE,
                             getValueSize));
  }
  if (Config.has(PassAlignment)) {
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                             "The known access alignment.", IRTArg::NONE,
                             getAlignment));
  }
  if (Config.has(PassValueTypeId)) {
    IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                             "The type id of the stored value.", IRTArg::NONE,
                             getValueTypeId));
  }
  if (Config.has(PassAtomicityOrdering)) {
    IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                             "The atomicity ordering of the store.",
                             IRTArg::NONE, getAtomicityOrdering));
  }
  if (Config.has(PassSyncScopeId)) {
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                             "The sync scope id of the store.", IRTArg::NONE,
                             getSyncScopeId));
  }
  if (Config.has(PassIsVolatile)) {
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                             "Flag indicating a volatile store.", IRTArg::NONE,
                             isVolatile));
  }

  addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
  IConf.addChoice(*this, IIRB.Ctx);
}

Value *StoreIO::getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return SI.getPointerOperand();
}

Value *StoreIO::setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  SI.setOperand(SI.getPointerOperandIndex(), &NewV);
  return &SI;
}

Value *StoreIO::getPointerAS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, SI.getPointerAddressSpace());
}

Value *StoreIO::getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                         InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return SI.getValueOperand();
}

Value *StoreIO::getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  auto &DL = SI.getDataLayout();
  return getCI(&Ty, DL.getTypeStoreSize(SI.getValueOperand()->getType()));
}

Value *StoreIO::getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                             InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, SI.getAlign().value());
}

Value *StoreIO::getValueTypeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, SI.getValueOperand()->getType()->getTypeID());
}

Value *StoreIO::getAtomicityOrdering(Value &V, Type &Ty,
                                     InstrumentationConfig &IConf,
                                     InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, uint64_t(SI.getOrdering()));
}

Value *StoreIO::getSyncScopeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                               InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, uint64_t(SI.getSyncScopeID()));
}

Value *StoreIO::isVolatile(Value &V, Type &Ty, InstrumentationConfig &IConf,
                           InstrumentorIRBuilderTy &IIRB) {
  auto &SI = cast<StoreInst>(V);
  return getCI(&Ty, SI.isVolatile());
}

void LoadIO::init(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
                  ConfigTy *UserConfig) {
  bool IsPRE = getLocationKind() == InstrumentationLocation::INSTRUCTION_PRE;
  if (UserConfig)
    Config = *UserConfig;
  if (Config.has(PassPointer)) {
    IRTArgs.push_back(
        IRTArg(IIRB.PtrTy, "pointer", "The accessed pointer.",
               ((IsPRE && Config.has(ReplacePointer)) ? IRTArg::REPLACABLE
                                                      : IRTArg::NONE),
               getPointer, setPointer));
  }
  if (Config.has(PassPointerAS)) {
    IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "pointer_as",
                             "The address space of the accessed pointer.",
                             IRTArg::NONE, getPointerAS));
  }
  if (!IsPRE && Config.has(PassValue)) {
    IRTArgs.push_back(
        IRTArg(getValueType(IIRB), "value", "The loaded value.",
               Config.has(ReplaceValue)
                   ? IRTArg::REPLACABLE | IRTArg::POTENTIALLY_INDIRECT |
                         (Config.has(PassValueSize) ? IRTArg::INDIRECT_HAS_SIZE
                                                    : IRTArg::NONE)
                   : IRTArg::NONE,
               getValue, Config.has(ReplaceValue) ? replaceValue : nullptr));
  }
  if (Config.has(PassValueSize)) {
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "value_size",
                             "The size of the loaded value.", IRTArg::NONE,
                             getValueSize));
  }
  if (Config.has(PassAlignment)) {
    IRTArgs.push_back(IRTArg(IIRB.Int64Ty, "alignment",
                             "The known access alignment.", IRTArg::NONE,
                             getAlignment));
  }
  if (Config.has(PassValueTypeId)) {
    IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "value_type_id",
                             "The type id of the loaded value.", IRTArg::NONE,
                             getValueTypeId));
  }
  if (Config.has(PassAtomicityOrdering)) {
    IRTArgs.push_back(IRTArg(IIRB.Int32Ty, "atomicity_ordering",
                             "The atomicity ordering of the load.",
                             IRTArg::NONE, getAtomicityOrdering));
  }
  if (Config.has(PassSyncScopeId)) {
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "sync_scope_id",
                             "The sync scope id of the load.", IRTArg::NONE,
                             getSyncScopeId));
  }
  if (Config.has(PassIsVolatile)) {
    IRTArgs.push_back(IRTArg(IIRB.Int8Ty, "is_volatile",
                             "Flag indicating a volatile load.", IRTArg::NONE,
                             isVolatile));
  }

  addCommonArgs(IConf, IIRB.Ctx, Config.has(PassId));
  IConf.addChoice(*this, IIRB.Ctx);
}

Value *LoadIO::getPointer(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return LI.getPointerOperand();
}

Value *LoadIO::setPointer(Value &V, Value &NewV, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  LI.setOperand(LI.getPointerOperandIndex(), &NewV);
  return &LI;
}

Value *LoadIO::getPointerAS(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, LI.getPointerAddressSpace());
}

Value *LoadIO::getValue(Value &V, Type &Ty, InstrumentationConfig &IConf,
                        InstrumentorIRBuilderTy &IIRB) {
  return &V;
}

Value *LoadIO::getValueSize(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  auto &DL = LI.getDataLayout();
  return getCI(&Ty, DL.getTypeStoreSize(LI.getType()));
}

Value *LoadIO::getAlignment(Value &V, Type &Ty, InstrumentationConfig &IConf,
                            InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, LI.getAlign().value());
}

Value *LoadIO::getValueTypeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, LI.getType()->getTypeID());
}

Value *LoadIO::getAtomicityOrdering(Value &V, Type &Ty,
                                    InstrumentationConfig &IConf,
                                    InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, uint64_t(LI.getOrdering()));
}

Value *LoadIO::getSyncScopeId(Value &V, Type &Ty, InstrumentationConfig &IConf,
                              InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, uint64_t(LI.getSyncScopeID()));
}

Value *LoadIO::isVolatile(Value &V, Type &Ty, InstrumentationConfig &IConf,
                          InstrumentorIRBuilderTy &IIRB) {
  auto &LI = cast<LoadInst>(V);
  return getCI(&Ty, LI.isVolatile());
}
