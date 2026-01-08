//===-- SPIRVPrepareFunctions.cpp - modify function signatures --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass modifies function signatures containing aggregate arguments
// and/or return value before IRTranslator. Information about the original
// signatures is stored in metadata. It is used during call lowering to
// restore correct SPIR-V types of function arguments and return values.
// This pass also substitutes some llvm intrinsic calls with calls to newly
// generated functions (as the Khronos LLVM/SPIR-V Translator does).
//
// NOTE: this pass is a module-level one due to the necessity to modify
// GVs/functions.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include <regex>

using namespace llvm;

namespace {

class SPIRVPrepareFunctions : public ModulePass {
  const SPIRVTargetMachine &TM;
  bool substituteIntrinsicCalls(Function *F);
  Function *removeAggregateTypesFromSignature(Function *F);
  bool removeAggregateTypesFromCalls(Function *F);

public:
  static char ID;
  SPIRVPrepareFunctions(const SPIRVTargetMachine &TM)
      : ModulePass(ID), TM(TM) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override { return "SPIRV prepare functions"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
};

static cl::list<std::string> SPVAllowUnknownIntrinsics(
    "spv-allow-unknown-intrinsics", cl::CommaSeparated,
    cl::desc("Emit unknown intrinsics as calls to external functions. A "
             "comma-separated input list of intrinsic prefixes must be "
             "provided, and only intrinsics carrying a listed prefix get "
             "emitted as described."),
    cl::value_desc("intrinsic_prefix_0,intrinsic_prefix_1"), cl::ValueOptional);
} // namespace

char SPIRVPrepareFunctions::ID = 0;

INITIALIZE_PASS(SPIRVPrepareFunctions, "prepare-functions",
                "SPIRV prepare functions", false, false)

static std::string lowerLLVMIntrinsicName(IntrinsicInst *II) {
  Function *IntrinsicFunc = II->getCalledFunction();
  assert(IntrinsicFunc && "Missing function");
  std::string FuncName = IntrinsicFunc->getName().str();
  llvm::replace(FuncName, '.', '_');
  FuncName = "spirv." + FuncName;
  return FuncName;
}

static Function *getOrCreateFunction(Module *M, Type *RetTy,
                                     ArrayRef<Type *> ArgTypes,
                                     StringRef Name) {
  FunctionType *FT = FunctionType::get(RetTy, ArgTypes, false);
  Function *F = M->getFunction(Name);
  if (F && F->getFunctionType() == FT)
    return F;
  Function *NewF = Function::Create(FT, GlobalValue::ExternalLinkage, Name, M);
  if (F)
    NewF->setDSOLocal(F->isDSOLocal());
  NewF->setCallingConv(CallingConv::SPIR_FUNC);
  return NewF;
}

static bool lowerIntrinsicToFunction(IntrinsicInst *Intrinsic) {
  // For @llvm.memset.* intrinsic cases with constant value and length arguments
  // are emulated via "storing" a constant array to the destination. For other
  // cases we wrap the intrinsic in @spirv.llvm_memset_* function and expand the
  // intrinsic to a loop via expandMemSetAsLoop().
  if (auto *MSI = dyn_cast<MemSetInst>(Intrinsic))
    if (isa<Constant>(MSI->getValue()) && isa<ConstantInt>(MSI->getLength()))
      return false; // It is handled later using OpCopyMemorySized.

  Module *M = Intrinsic->getModule();
  std::string FuncName = lowerLLVMIntrinsicName(Intrinsic);
  if (Intrinsic->isVolatile())
    FuncName += ".volatile";
  // Redirect @llvm.intrinsic.* call to @spirv.llvm_intrinsic_*
  Function *F = M->getFunction(FuncName);
  if (F) {
    Intrinsic->setCalledFunction(F);
    return true;
  }
  // TODO copy arguments attributes: nocapture writeonly.
  FunctionCallee FC =
      M->getOrInsertFunction(FuncName, Intrinsic->getFunctionType());
  auto IntrinsicID = Intrinsic->getIntrinsicID();
  Intrinsic->setCalledFunction(FC);

  F = dyn_cast<Function>(FC.getCallee());
  assert(F && "Callee must be a function");

  switch (IntrinsicID) {
  case Intrinsic::memset: {
    auto *MSI = static_cast<MemSetInst *>(Intrinsic);
    Argument *Dest = F->getArg(0);
    Argument *Val = F->getArg(1);
    Argument *Len = F->getArg(2);
    Argument *IsVolatile = F->getArg(3);
    Dest->setName("dest");
    Val->setName("val");
    Len->setName("len");
    IsVolatile->setName("isvolatile");
    BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", F);
    IRBuilder<> IRB(EntryBB);
    auto *MemSet = IRB.CreateMemSet(Dest, Val, Len, MSI->getDestAlign(),
                                    MSI->isVolatile());
    IRB.CreateRetVoid();
    expandMemSetAsLoop(cast<MemSetInst>(MemSet));
    MemSet->eraseFromParent();
    break;
  }
  case Intrinsic::bswap: {
    BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", F);
    IRBuilder<> IRB(EntryBB);
    auto *BSwap = IRB.CreateIntrinsic(Intrinsic::bswap, Intrinsic->getType(),
                                      F->getArg(0));
    IRB.CreateRet(BSwap);
    IntrinsicLowering IL(M->getDataLayout());
    IL.LowerIntrinsicCall(BSwap);
    break;
  }
  default:
    break;
  }
  return true;
}

static std::string getAnnotation(Value *AnnoVal, Value *OptAnnoVal) {
  if (auto *Ref = dyn_cast_or_null<GetElementPtrInst>(AnnoVal))
    AnnoVal = Ref->getOperand(0);
  if (auto *Ref = dyn_cast_or_null<BitCastInst>(OptAnnoVal))
    OptAnnoVal = Ref->getOperand(0);

  std::string Anno;
  if (auto *C = dyn_cast_or_null<Constant>(AnnoVal)) {
    StringRef Str;
    if (getConstantStringInfo(C, Str))
      Anno = Str;
  }
  // handle optional annotation parameter in a way that Khronos Translator do
  // (collect integers wrapped in a struct)
  if (auto *C = dyn_cast_or_null<Constant>(OptAnnoVal);
      C && C->getNumOperands()) {
    Value *MaybeStruct = C->getOperand(0);
    if (auto *Struct = dyn_cast<ConstantStruct>(MaybeStruct)) {
      for (unsigned I = 0, E = Struct->getNumOperands(); I != E; ++I) {
        if (auto *CInt = dyn_cast<ConstantInt>(Struct->getOperand(I)))
          Anno += (I == 0 ? ": " : ", ") +
                  std::to_string(CInt->getType()->getIntegerBitWidth() == 1
                                     ? CInt->getZExtValue()
                                     : CInt->getSExtValue());
      }
    } else if (auto *Struct = dyn_cast<ConstantAggregateZero>(MaybeStruct)) {
      // { i32 i32 ... } zeroinitializer
      for (unsigned I = 0, E = Struct->getType()->getStructNumElements();
           I != E; ++I)
        Anno += I == 0 ? ": 0" : ", 0";
    }
  }
  return Anno;
}

static SmallVector<Metadata *> parseAnnotation(Value *I,
                                               const std::string &Anno,
                                               LLVMContext &Ctx,
                                               Type *Int32Ty) {
  // Try to parse the annotation string according to the following rules:
  // annotation := ({kind} | {kind:value,value,...})+
  // kind := number
  // value := number | string
  static const std::regex R(
      "\\{(\\d+)(?:[:,](\\d+|\"[^\"]*\")(?:,(\\d+|\"[^\"]*\"))*)?\\}");
  SmallVector<Metadata *> MDs;
  int Pos = 0;
  for (std::sregex_iterator
           It = std::sregex_iterator(Anno.begin(), Anno.end(), R),
           ItEnd = std::sregex_iterator();
       It != ItEnd; ++It) {
    if (It->position() != Pos)
      return SmallVector<Metadata *>{};
    Pos = It->position() + It->length();
    std::smatch Match = *It;
    SmallVector<Metadata *> MDsItem;
    for (std::size_t i = 1; i < Match.size(); ++i) {
      std::ssub_match SMatch = Match[i];
      std::string Item = SMatch.str();
      if (Item.length() == 0)
        break;
      if (Item[0] == '"') {
        Item = Item.substr(1, Item.length() - 2);
        // Acceptable format of the string snippet is:
        static const std::regex RStr("^(\\d+)(?:,(\\d+))*$");
        if (std::smatch MatchStr; std::regex_match(Item, MatchStr, RStr)) {
          for (std::size_t SubIdx = 1; SubIdx < MatchStr.size(); ++SubIdx)
            if (std::string SubStr = MatchStr[SubIdx].str(); SubStr.length())
              MDsItem.push_back(ConstantAsMetadata::get(
                  ConstantInt::get(Int32Ty, std::stoi(SubStr))));
        } else {
          MDsItem.push_back(MDString::get(Ctx, Item));
        }
      } else if (int32_t Num; llvm::to_integer(StringRef(Item), Num, 10)) {
        MDsItem.push_back(
            ConstantAsMetadata::get(ConstantInt::get(Int32Ty, Num)));
      } else {
        MDsItem.push_back(MDString::get(Ctx, Item));
      }
    }
    if (MDsItem.size() == 0)
      return SmallVector<Metadata *>{};
    MDs.push_back(MDNode::get(Ctx, MDsItem));
  }
  return Pos == static_cast<int>(Anno.length()) ? std::move(MDs)
                                                : SmallVector<Metadata *>{};
}

static void lowerPtrAnnotation(IntrinsicInst *II) {
  LLVMContext &Ctx = II->getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);

  // Retrieve an annotation string from arguments.
  Value *PtrArg = nullptr;
  if (auto *BI = dyn_cast<BitCastInst>(II->getArgOperand(0)))
    PtrArg = BI->getOperand(0);
  else
    PtrArg = II->getOperand(0);
  std::string Anno =
      getAnnotation(II->getArgOperand(1),
                    4 < II->arg_size() ? II->getArgOperand(4) : nullptr);

  // Parse the annotation.
  SmallVector<Metadata *> MDs = parseAnnotation(II, Anno, Ctx, Int32Ty);

  // If the annotation string is not parsed successfully we don't know the
  // format used and output it as a general UserSemantic decoration.
  // Otherwise MDs is a Metadata tuple (a decoration list) in the format
  // expected by `spirv.Decorations`.
  if (MDs.size() == 0) {
    auto UserSemantic = ConstantAsMetadata::get(ConstantInt::get(
        Int32Ty, static_cast<uint32_t>(SPIRV::Decoration::UserSemantic)));
    MDs.push_back(MDNode::get(Ctx, {UserSemantic, MDString::get(Ctx, Anno)}));
  }

  // Build the internal intrinsic function.
  IRBuilder<> IRB(II->getParent());
  IRB.SetInsertPoint(II);
  IRB.CreateIntrinsic(
      Intrinsic::spv_assign_decoration, {PtrArg->getType()},
      {PtrArg, MetadataAsValue::get(Ctx, MDNode::get(Ctx, MDs))});
  II->replaceAllUsesWith(II->getOperand(0));
}

static void lowerFunnelShifts(IntrinsicInst *FSHIntrinsic) {
  // Get a separate function - otherwise, we'd have to rework the CFG of the
  // current one. Then simply replace the intrinsic uses with a call to the new
  // function.
  // Generate LLVM IR for  i* @spirv.llvm_fsh?_i* (i* %a, i* %b, i* %c)
  Module *M = FSHIntrinsic->getModule();
  FunctionType *FSHFuncTy = FSHIntrinsic->getFunctionType();
  Type *FSHRetTy = FSHFuncTy->getReturnType();
  const std::string FuncName = lowerLLVMIntrinsicName(FSHIntrinsic);
  Function *FSHFunc =
      getOrCreateFunction(M, FSHRetTy, FSHFuncTy->params(), FuncName);

  if (!FSHFunc->empty()) {
    FSHIntrinsic->setCalledFunction(FSHFunc);
    return;
  }
  BasicBlock *RotateBB = BasicBlock::Create(M->getContext(), "rotate", FSHFunc);
  IRBuilder<> IRB(RotateBB);
  Type *Ty = FSHFunc->getReturnType();
  // Build the actual funnel shift rotate logic.
  // In the comments, "int" is used interchangeably with "vector of int
  // elements".
  FixedVectorType *VectorTy = dyn_cast<FixedVectorType>(Ty);
  Type *IntTy = VectorTy ? VectorTy->getElementType() : Ty;
  unsigned BitWidth = IntTy->getIntegerBitWidth();
  ConstantInt *BitWidthConstant = IRB.getInt({BitWidth, BitWidth});
  Value *BitWidthForInsts =
      VectorTy
          ? IRB.CreateVectorSplat(VectorTy->getNumElements(), BitWidthConstant)
          : BitWidthConstant;
  Value *RotateModVal =
      IRB.CreateURem(/*Rotate*/ FSHFunc->getArg(2), BitWidthForInsts);
  Value *FirstShift = nullptr, *SecShift = nullptr;
  if (FSHIntrinsic->getIntrinsicID() == Intrinsic::fshr) {
    // Shift the less significant number right, the "rotate" number of bits
    // will be 0-filled on the left as a result of this regular shift.
    FirstShift = IRB.CreateLShr(FSHFunc->getArg(1), RotateModVal);
  } else {
    // Shift the more significant number left, the "rotate" number of bits
    // will be 0-filled on the right as a result of this regular shift.
    FirstShift = IRB.CreateShl(FSHFunc->getArg(0), RotateModVal);
  }
  // We want the "rotate" number of the more significant int's LSBs (MSBs) to
  // occupy the leftmost (rightmost) "0 space" left by the previous operation.
  // Therefore, subtract the "rotate" number from the integer bitsize...
  Value *SubRotateVal = IRB.CreateSub(BitWidthForInsts, RotateModVal);
  if (FSHIntrinsic->getIntrinsicID() == Intrinsic::fshr) {
    // ...and left-shift the more significant int by this number, zero-filling
    // the LSBs.
    SecShift = IRB.CreateShl(FSHFunc->getArg(0), SubRotateVal);
  } else {
    // ...and right-shift the less significant int by this number, zero-filling
    // the MSBs.
    SecShift = IRB.CreateLShr(FSHFunc->getArg(1), SubRotateVal);
  }
  // A simple binary addition of the shifted ints yields the final result.
  IRB.CreateRet(IRB.CreateOr(FirstShift, SecShift));

  FSHIntrinsic->setCalledFunction(FSHFunc);
}

static void lowerConstrainedFPCmpIntrinsic(
    ConstrainedFPCmpIntrinsic *ConstrainedCmpIntrinsic,
    SmallVector<Instruction *> &EraseFromParent) {
  if (!ConstrainedCmpIntrinsic)
    return;
  // Extract the floating-point values being compared
  Value *LHS = ConstrainedCmpIntrinsic->getArgOperand(0);
  Value *RHS = ConstrainedCmpIntrinsic->getArgOperand(1);
  FCmpInst::Predicate Pred = ConstrainedCmpIntrinsic->getPredicate();
  IRBuilder<> Builder(ConstrainedCmpIntrinsic);
  Value *FCmp = Builder.CreateFCmp(Pred, LHS, RHS);
  ConstrainedCmpIntrinsic->replaceAllUsesWith(FCmp);
  EraseFromParent.push_back(dyn_cast<Instruction>(ConstrainedCmpIntrinsic));
}

static void lowerExpectAssume(IntrinsicInst *II) {
  // If we cannot use the SPV_KHR_expect_assume extension, then we need to
  // ignore the intrinsic and move on. It should be removed later on by LLVM.
  // Otherwise we should lower the intrinsic to the corresponding SPIR-V
  // instruction.
  // For @llvm.assume we have OpAssumeTrueKHR.
  // For @llvm.expect we have OpExpectKHR.
  //
  // We need to lower this into a builtin and then the builtin into a SPIR-V
  // instruction.
  if (II->getIntrinsicID() == Intrinsic::assume) {
    Function *F = Intrinsic::getOrInsertDeclaration(
        II->getModule(), Intrinsic::SPVIntrinsics::spv_assume);
    II->setCalledFunction(F);
  } else if (II->getIntrinsicID() == Intrinsic::expect) {
    Function *F = Intrinsic::getOrInsertDeclaration(
        II->getModule(), Intrinsic::SPVIntrinsics::spv_expect,
        {II->getOperand(0)->getType()});
    II->setCalledFunction(F);
  } else {
    llvm_unreachable("Unknown intrinsic");
  }
}

static bool toSpvLifetimeIntrinsic(IntrinsicInst *II, Intrinsic::ID NewID) {
  auto *LifetimeArg0 = II->getArgOperand(0);

  // If the lifetime argument is a poison value, the intrinsic has no effect.
  if (isa<PoisonValue>(LifetimeArg0)) {
    II->eraseFromParent();
    return true;
  }

  IRBuilder<> Builder(II);
  auto *Alloca = cast<AllocaInst>(LifetimeArg0);
  std::optional<TypeSize> Size =
      Alloca->getAllocationSize(Alloca->getDataLayout());
  Value *SizeVal = Builder.getInt64(Size ? *Size : -1);
  Builder.CreateIntrinsic(NewID, Alloca->getType(), {SizeVal, LifetimeArg0});
  II->eraseFromParent();
  return true;
}

// Substitutes calls to LLVM intrinsics with either calls to SPIR-V intrinsics
// or calls to proper generated functions. Returns True if F was modified.
bool SPIRVPrepareFunctions::substituteIntrinsicCalls(Function *F) {
  bool Changed = false;
  const SPIRVSubtarget &STI = TM.getSubtarget<SPIRVSubtarget>(*F);
  SmallVector<Instruction *> EraseFromParent;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : make_early_inc_range(BB)) {
      auto Call = dyn_cast<CallInst>(&I);
      if (!Call)
        continue;
      Function *CF = Call->getCalledFunction();
      if (!CF || !CF->isIntrinsic())
        continue;
      auto *II = cast<IntrinsicInst>(Call);
      switch (II->getIntrinsicID()) {
      case Intrinsic::memset:
      case Intrinsic::bswap:
        Changed |= lowerIntrinsicToFunction(II);
        break;
      case Intrinsic::fshl:
      case Intrinsic::fshr:
        lowerFunnelShifts(II);
        Changed = true;
        break;
      case Intrinsic::assume:
      case Intrinsic::expect:
        if (STI.canUseExtension(SPIRV::Extension::SPV_KHR_expect_assume))
          lowerExpectAssume(II);
        Changed = true;
        break;
      case Intrinsic::lifetime_start:
        if (!STI.isShader()) {
          Changed |= toSpvLifetimeIntrinsic(
              II, Intrinsic::SPVIntrinsics::spv_lifetime_start);
        } else {
          II->eraseFromParent();
          Changed = true;
        }
        break;
      case Intrinsic::lifetime_end:
        if (!STI.isShader()) {
          Changed |= toSpvLifetimeIntrinsic(
              II, Intrinsic::SPVIntrinsics::spv_lifetime_end);
        } else {
          II->eraseFromParent();
          Changed = true;
        }
        break;
      case Intrinsic::ptr_annotation:
        lowerPtrAnnotation(II);
        Changed = true;
        break;
      case Intrinsic::experimental_constrained_fcmp:
      case Intrinsic::experimental_constrained_fcmps:
        lowerConstrainedFPCmpIntrinsic(dyn_cast<ConstrainedFPCmpIntrinsic>(II),
                                       EraseFromParent);
        Changed = true;
        break;
      default:
        if (TM.getTargetTriple().getVendor() == Triple::AMD ||
            any_of(SPVAllowUnknownIntrinsics, [II](auto &&Prefix) {
              if (Prefix.empty())
                return false;
              return II->getCalledFunction()->getName().starts_with(Prefix);
            }))
          Changed |= lowerIntrinsicToFunction(II);
        break;
      }
    }
  }
  for (auto *I : EraseFromParent)
    I->eraseFromParent();
  return Changed;
}

static void
addFunctionTypeMutation(NamedMDNode *NMD,
                        SmallVector<std::pair<int, Type *>> ChangedTys,
                        StringRef Name) {

  LLVMContext &Ctx = NMD->getParent()->getContext();
  Type *I32Ty = IntegerType::getInt32Ty(Ctx);

  SmallVector<Metadata *> MDArgs;
  MDArgs.push_back(MDString::get(Ctx, Name));
  transform(ChangedTys, std::back_inserter(MDArgs), [=, &Ctx](auto &&CTy) {
    return MDNode::get(
        Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32Ty, CTy.first, true)),
              ValueAsMetadata::get(Constant::getNullValue(CTy.second))});
  });
  NMD->addOperand(MDNode::get(Ctx, MDArgs));
}
// Returns F if aggregate argument/return types are not present or cloned F
// function with the types replaced by i32 types. The change in types is
// noted in 'spv.cloned_funcs' metadata for later restoration.
Function *
SPIRVPrepareFunctions::removeAggregateTypesFromSignature(Function *F) {
  bool IsRetAggr = F->getReturnType()->isAggregateType();
  // Allow intrinsics with aggregate return type to reach GlobalISel
  if (F->isIntrinsic() && IsRetAggr)
    return F;

  IRBuilder<> B(F->getContext());

  bool HasAggrArg = llvm::any_of(F->args(), [](Argument &Arg) {
    return Arg.getType()->isAggregateType();
  });
  bool DoClone = IsRetAggr || HasAggrArg;
  if (!DoClone)
    return F;
  SmallVector<std::pair<int, Type *>, 4> ChangedTypes;
  Type *RetType = IsRetAggr ? B.getInt32Ty() : F->getReturnType();
  if (IsRetAggr)
    ChangedTypes.push_back(std::pair<int, Type *>(-1, F->getReturnType()));
  SmallVector<Type *, 4> ArgTypes;
  for (const auto &Arg : F->args()) {
    if (Arg.getType()->isAggregateType()) {
      ArgTypes.push_back(B.getInt32Ty());
      ChangedTypes.push_back(
          std::pair<int, Type *>(Arg.getArgNo(), Arg.getType()));
    } else
      ArgTypes.push_back(Arg.getType());
  }
  FunctionType *NewFTy =
      FunctionType::get(RetType, ArgTypes, F->getFunctionType()->isVarArg());
  Function *NewF =
      Function::Create(NewFTy, F->getLinkage(), F->getAddressSpace(),
                       F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  auto NewFArgIt = NewF->arg_begin();
  for (auto &Arg : F->args()) {
    StringRef ArgName = Arg.getName();
    NewFArgIt->setName(ArgName);
    VMap[&Arg] = &(*NewFArgIt++);
  }
  SmallVector<ReturnInst *, 8> Returns;

  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);
  NewF->takeName(F);

  addFunctionTypeMutation(
      NewF->getParent()->getOrInsertNamedMetadata("spv.cloned_funcs"),
      std::move(ChangedTypes), NewF->getName());

  for (auto *U : make_early_inc_range(F->users())) {
    if (CallInst *CI;
        (CI = dyn_cast<CallInst>(U)) && CI->getCalledFunction() == F)
      CI->mutateFunctionType(NewF->getFunctionType());
    if (auto *C = dyn_cast<Constant>(U))
      C->handleOperandChange(F, NewF);
    else
      U->replaceUsesOfWith(F, NewF);
  }

  // register the mutation
  if (RetType != F->getReturnType())
    TM.getSubtarget<SPIRVSubtarget>(*F).getSPIRVGlobalRegistry()->addMutated(
        NewF, F->getReturnType());
  return NewF;
}

// Mutates indirect callsites iff if aggregate argument/return types are present
// with the types replaced by i32 types. The change in types is noted in
// 'spv.mutated_callsites' metadata for later restoration.
bool SPIRVPrepareFunctions::removeAggregateTypesFromCalls(Function *F) {
  if (F->isDeclaration() || F->isIntrinsic())
    return false;

  SmallVector<std::pair<CallBase *, FunctionType *>> Calls;
  for (auto &&I : instructions(F)) {
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (!CB->getCalledOperand() || CB->getCalledFunction())
        continue;
      if (CB->getType()->isAggregateType() ||
          any_of(CB->args(),
                 [](auto &&Arg) { return Arg->getType()->isAggregateType(); }))
        Calls.emplace_back(CB, nullptr);
    }
  }

  if (Calls.empty())
    return false;

  IRBuilder<> B(F->getContext());

  for (auto &&[CB, NewFnTy] : Calls) {
    SmallVector<std::pair<int, Type *>> ChangedTypes;
    SmallVector<Type *> NewArgTypes;

    Type *RetTy = CB->getType();
    if (RetTy->isAggregateType()) {
      ChangedTypes.emplace_back(-1, RetTy);
      RetTy = B.getInt32Ty();
    }

    for (auto &&Arg : CB->args()) {
      if (Arg->getType()->isAggregateType()) {
        NewArgTypes.push_back(B.getInt32Ty());
        ChangedTypes.emplace_back(Arg.getOperandNo(), Arg->getType());
      } else {
        NewArgTypes.push_back(Arg->getType());
      }
    }
    NewFnTy = FunctionType::get(RetTy, NewArgTypes,
                                CB->getFunctionType()->isVarArg());

    if (!CB->hasName())
      CB->setName("spv.mutated_callsite." + F->getName());
    else
      CB->setName("spv.named_mutated_callsite." + F->getName() + "." +
                  CB->getName());

    addFunctionTypeMutation(
        F->getParent()->getOrInsertNamedMetadata("spv.mutated_callsites"),
        std::move(ChangedTypes), CB->getName());
  }

  for (auto &&[CB, NewFTy] : Calls) {
    if (NewFTy->getReturnType() != CB->getType())
      TM.getSubtarget<SPIRVSubtarget>(*F).getSPIRVGlobalRegistry()->addMutated(
          CB, CB->getType());
    CB->mutateFunctionType(NewFTy);
  }

  return true;
}

bool SPIRVPrepareFunctions::runOnModule(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    Changed |= substituteIntrinsicCalls(&F);
    Changed |= sortBlocks(F);
    Changed |= removeAggregateTypesFromCalls(&F);
  }

  std::vector<Function *> FuncsWorklist;
  for (auto &F : M)
    FuncsWorklist.push_back(&F);

  for (auto *F : FuncsWorklist) {
    Function *NewF = removeAggregateTypesFromSignature(F);

    if (NewF != F) {
      F->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

ModulePass *
llvm::createSPIRVPrepareFunctionsPass(const SPIRVTargetMachine &TM) {
  return new SPIRVPrepareFunctions(TM);
}
