//===-- Instrumentor.cpp - Highly configurable instrumentation pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Instrumentor.h"
#include "llvm/Transforms/IPO/InstrumentorConfigFile.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
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

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <string>
#include <system_error>
#include <type_traits>

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "instrumentor"

static cl::opt<std::string> WriteJSONConfig(
    "instrumentor-write-config-file",
    cl::desc(
        "Write the instrumentor configuration into the specified JSON file"),
    cl::init(""));
static cl::opt<std::string> ReadJSONConfig(
    "instrumentor-read-config-file",
    cl::desc(
        "Read the instrumentor configuration from the specified JSON file"),
    cl::init(""));

namespace {

template <typename IRBuilderTy> void ensureDbgLoc(IRBuilderTy &IRB) {
  if (IRB.getCurrentDebugLocation())
    return;
  auto *BB = IRB.GetInsertBlock();
  if (auto *SP = BB->getParent()->getSubprogram())
    IRB.SetCurrentDebugLocation(DILocation::get(BB->getContext(), 0, 0, SP));
}

template <typename IRBTy>
Value *tryToCast(IRBTy &IRB, Value *V, Type *Ty, const DataLayout &DL,
                 bool AllowTruncate = false) {
  if (!V)
    return Constant::getAllOnesValue(Ty);
  auto *VTy = V->getType();
  if (VTy == Ty)
    return V;
  if (VTy->isAggregateType())
    return V;
  auto RequestedSize = DL.getTypeSizeInBits(Ty);
  auto ValueSize = DL.getTypeSizeInBits(VTy);
  bool IsTruncate = RequestedSize < ValueSize;
  if (IsTruncate && !AllowTruncate)
    return V;
  if (IsTruncate && AllowTruncate)
    return tryToCast(IRB,
                     IRB.CreateIntCast(V, IRB.getIntNTy(RequestedSize),
                                       /*IsSigned=*/false),
                     Ty, DL, AllowTruncate);
  if (VTy->isPointerTy() && Ty->isPointerTy())
    return IRB.CreatePointerBitCastOrAddrSpaceCast(V, Ty);
  if (VTy->isIntegerTy() && Ty->isIntegerTy())
    return IRB.CreateIntCast(V, Ty, /*IsSigned=*/false);
  if (VTy->isFloatingPointTy() && Ty->isIntOrPtrTy()) {
    switch (ValueSize) {
    case 64:
      return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getInt64Ty()), Ty, DL,
                       AllowTruncate);
    case 32:
      return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getInt32Ty()), Ty, DL,
                       AllowTruncate);
    case 16:
      return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getInt16Ty()), Ty, DL,
                       AllowTruncate);
    case 8:
      return tryToCast(IRB, IRB.CreateBitCast(V, IRB.getInt8Ty()), Ty, DL,
                       AllowTruncate);
    default:
      llvm_unreachable("unsupported floating point size");
    }
  }
  return IRB.CreateBitOrPointerCast(V, Ty);
}

template <typename Ty> Constant *getCI(Type *IT, Ty Val) {
  return ConstantInt::get(IT, Val);
}

class InstrumentorImpl final {
public:
  InstrumentorImpl(InstrumentationConfig &IConf, InstrumentorIRBuilderTy &IIRB,
                   Module &M, FunctionAnalysisManager &FAM)
      : IConf(IConf), M(M), FAM(FAM), IIRB(IIRB) {
    IConf.populate(IIRB);
  }

  /// Instrument the module, public entry point.
  bool instrument();

private:
  bool shouldInstrumentTarget();
  bool shouldInstrumentFunction(Function &Fn);

  bool instrumentInstruction(Instruction &I, InstrumentationCaches &ICaches);
  bool instrumentFunction(Function &Fn);

  /// The instrumentation opportunities for instructions indexed by
  /// their opcode.
  DenseMap<unsigned, InstrumentationOpportunity *> InstChoicesPRE,
      InstChoicesPOST;

  /// The instrumentor configuration.
  InstrumentationConfig &IConf;

  /// The underlying module.
  Module &M;

  FunctionAnalysisManager &FAM;

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
      errs() << "WARNING: failed to parse target regex: " << ErrMsg << "\n";
      return false;
    }
    RegexMatches = TargetRegex.match(T.str());
  }

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
  ++IIRB.Epoche;

  Value *IPtr = &I;
  if (auto *IO = InstChoicesPRE.lookup(I.getOpcode())) {
    IIRB.IRB.SetInsertPoint(&I);
    ensureDbgLoc(IIRB.IRB);
    Changed |= bool(IO->instrument(IPtr, IConf, IIRB, ICaches));
  }

  if (auto *IO = InstChoicesPOST.lookup(I.getOpcode())) {
    IIRB.IRB.SetInsertPoint(I.getNextNonDebugInstruction());
    ensureDbgLoc(IIRB.IRB);
    Changed |= bool(IO->instrument(IPtr, IConf, IIRB, ICaches));
  }
  IIRB.returnAllocas();

  return Changed;
};

bool InstrumentorImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;
  if (!shouldInstrumentFunction(Fn))
    return Changed;

  InstrumentationCaches ICaches;
  ReversePostOrderTraversal<Function *> RPOT(&Fn);
  for (auto &It : RPOT)
    for (auto &I : *It)
      Changed |= instrumentInstruction(I, ICaches);

  return Changed;
}

bool InstrumentorImpl::instrument() {
  bool Changed = false;
  if (!shouldInstrumentTarget())
    return Changed;

  for (auto &It : IConf.IChoices[InstrumentationLocation::INSTRUCTION_PRE])
    if (It.second->Enabled)
      InstChoicesPRE[It.second->getOpcode()] = It.second;
  for (auto &It : IConf.IChoices[InstrumentationLocation::INSTRUCTION_POST])
    if (It.second->Enabled)
      InstChoicesPOST[It.second->getOpcode()] = It.second;

  for (Function &Fn : M)
    Changed |= instrumentFunction(Fn);

  return Changed;
}

PreservedAnalyses InstrumentorPass::run(Module &M, FunctionAnalysisManager &FAM,
                                        InstrumentationConfig &IConf,
                                        InstrumentorIRBuilderTy &IIRB) {
  InstrumentorImpl Impl(IConf, IIRB, M, FAM);
  if (IConf.ReadConfig && !readConfigFromJSON(IConf, ReadJSONConfig))
    return PreservedAnalyses::all();

  writeConfigToJSON(IConf, WriteJSONConfig);

  bool Changed = Impl.instrument();
  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

PreservedAnalyses InstrumentorPass::run(Module &M, ModuleAnalysisManager &MAM) {
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  InstrumentationConfig *IConf =
      UserIConf ? UserIConf : new InstrumentationConfig();
  InstrumentorIRBuilderTy *IIRB =
      UserIIRB ? UserIIRB : new InstrumentorIRBuilderTy(M, FAM);

  auto PA = run(M, FAM, *IConf, *IIRB);

  if (!UserIIRB)
    delete IIRB;
  if (!UserIConf)
    delete IConf;

  assert(!verifyModule(M, &errs()));

  return PA;
}

BaseConfigurationOpportunity *
BaseConfigurationOpportunity::getBoolOption(InstrumentationConfig &IConf,
                                            StringRef Name,
                                            StringRef Description, bool Value) {
  auto *BCO = new BaseConfigurationOpportunity();
  BCO->Name = Name;
  BCO->Description = Description;
  BCO->Kind = BOOLEAN;
  BCO->V.B = Value;
  IConf.addBaseChoice(BCO);
  return BCO;
}

BaseConfigurationOpportunity *BaseConfigurationOpportunity::getStringOption(
    InstrumentationConfig &IConf, StringRef Name, StringRef Description,
    StringRef Value) {
  auto *BCO = new BaseConfigurationOpportunity();
  BCO->Name = Name;
  BCO->Description = Description;
  BCO->Kind = STRING;
  BCO->V.S = Value;
  IConf.addBaseChoice(BCO);
  return BCO;
}

void InstrumentationConfig::populate(InstrumentorIRBuilderTy &IIRB) {
  /// List of all instrumentation opportunities.
  LoadIO::populate(*this, IIRB);
  StoreIO::populate(*this, IIRB);
}

void InstrumentationConfig::addChoice(InstrumentationOpportunity &IO) {
  auto *&ICPtr = IChoices[IO.getLocationKind()][IO.getName()];
  if (ICPtr && IO.getLocationKind() != InstrumentationLocation::SPECIAL_VALUE) {
    errs() << "WARNING: registered two instrumentation opportunities for the "
              "same location ("
           << ICPtr->getName() << " vs " << IO.getName() << ")!\n";
  }
  ICPtr = &IO;
}

Value *InstrumentationOpportunity::getIdPre(Value &V, Type &Ty,
                                            InstrumentationConfig &IConf,
                                            InstrumentorIRBuilderTy &IIRB) {
  return getCI(&Ty, getIdFromEpoche(IIRB.Epoche));
}

Value *InstrumentationOpportunity::getIdPost(Value &V, Type &Ty,
                                             InstrumentationConfig &IConf,
                                             InstrumentorIRBuilderTy &IIRB) {
  return getCI(&Ty, -getIdFromEpoche(IIRB.Epoche));
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
    if (IIRB.NewInsts.lookup(cast<Instruction>(U.getUser())) == IIRB.Epoche)
      return false;
    if (isa<LifetimeIntrinsic>(U.getUser()) || U.getUser()->isDroppable())
      return false;
    return true;
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

FunctionType *
IRTCallDescription::createLLVMSignature(InstrumentationConfig &IConf,
                                        LLVMContext &Ctx, const DataLayout &DL,
                                        bool ForceIndirection) {
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
    ParamTypes.push_back(PointerType::get(Ctx, 0));
    if (!(It.Flags & IRTArg::INDIRECT_HAS_SIZE))
      ParamTypes.push_back(IntegerType::getInt32Ty(Ctx));
  }
  if (!RetTy)
    RetTy = Type::getVoidTy(Ctx);

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
    auto *&Param = ICaches.DirectArgCache[{IIRB.Epoche, IO.getName(), It.Name}];
    if (!Param || It.NoCache)
      // Avoid passing the caches to the getter.
      Param = It.GetterCB(*V, *It.Ty, IConf, IIRB);
    if (!Param)
      errs() << IO.getName() << " : " << It.Name << "\n";
    assert(Param);

    if (Param->getType()->isVoidTy()) {
      Param = Constant::getNullValue(It.Ty);
    } else if (Param->getType()->isAggregateType() ||
               DL.getTypeSizeInBits(Param->getType()) >
                   DL.getTypeSizeInBits(It.Ty)) {
      if (!isPotentiallyIndirect(It)) {
        errs() << "WARNING: Indirection needed for " << It.Name << " of " << *V
               << " in " << IO.getName() << ", but not indicated\n. Got "
               << *Param << " expected " << *It.Ty
               << "; instrumentation is skipped";
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
          ICaches.IndirectArgCache[{IIRB.Epoche, IO.getName(), It.Name}];
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

  auto *FnTy =
      createLLVMSignature(IConf, V->getContext(), DL, ForceIndirection);
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
                          ? ICaches.DirectArgCache[{IIRB.Epoche, IO.getName(),
                                                    IO.IRTArgs[I].Name}]
                          : CI;
    assert(NewValue);
    if (ForceIndirection && !IsCustomReplaceable &&
        isPotentiallyIndirect(IO.IRTArgs[I])) {
      auto *Q = ICaches.IndirectArgCache[{IIRB.Epoche, IO.getName(),
                                          IO.IRTArgs[I].Name}];
      NewValue = IIRB.IRB.CreateLoad(V->getType(), Q);
    }
    V = IO.IRTArgs[I].SetterCB(*V, *NewValue, IConf, IIRB);
  }
  return CI;
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
