//===-- ExpandVariadicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is an optimization pass for variadic functions. If called from codegen,
// it can serve as the implementation of variadic functions for a given target.
//
// The strategy is to turn the ... part of a variadic function into a va_list
// and fix up the call sites. The majority of the pass is target independent.
// The exceptions are the va_list type itself and the rules for where to store
// variables in memory such that va_arg can iterate over them given a va_list.
//
// The majority of the plumbing is splitting the variadic function into a
// single basic block that packs the variadic arguments into a va_list and
// a second function that does the work of the original. That packing is
// exactly what is done by va_start. Further, the transform from ... to va_list
// replaced va_start with an operation to copy a va_list from the new argument,
// which is exactly a va_copy. This is useful for reducing target-dependence.
//
// A va_list instance is a forward iterator, where the primary operation va_arg
// is dereference-then-increment. This interface forces significant convergent
// evolution between target specific implementations. The variation in runtime
// data layout is limited to that representable by the iterator, parameterised
// by the type passed to the va_arg instruction.
//
// Therefore the majority of the target specific subtlety is packing arguments
// into a stack allocated buffer such that a va_list can be initialised with it
// and the va_arg expansion for the target will find the arguments at runtime.
//
// The aggregate effect is to unblock other transforms, most critically the
// general purpose inliner. Known calls to variadic functions become zero cost.
//
// Consistency with clang is primarily tested by emitting va_arg using clang
// then expanding the variadic functions using this pass, followed by trying
// to constant fold the functions to no-ops.
//
// Target specific behaviour is tested in IR - mainly checking that values are
// put into positions in call frames that make sense for that particular target.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ExpandVariadics.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cstdio>

#define DEBUG_TYPE "expand-variadics"

using namespace llvm;

cl::opt<ExpandVariadicsMode> ExpandVariadicsModeOption(
    DEBUG_TYPE "-override", cl::desc("Override the behaviour of " DEBUG_TYPE),
    cl::init(ExpandVariadicsMode::Unspecified),
    cl::values(clEnumValN(ExpandVariadicsMode::Unspecified, "unspecified",
                          "Use the implementation defaults"),
               clEnumValN(ExpandVariadicsMode::Disable, "disable",
                          "Disable the pass entirely"),
               clEnumValN(ExpandVariadicsMode::Optimize, "optimize",
                          "Optimise without changing ABI"),
               clEnumValN(ExpandVariadicsMode::Lowering, "lowering",
                          "Change variadic calling convention")));

namespace {

// Instances of this class encapsulate the target-dependant behaviour as a
// function of triple. Implementing a new ABI is adding a case to the switch
// in create(llvm::Triple) at the end of this file.
class VariadicABIInfo {
protected:
  VariadicABIInfo() {}

public:
  static std::unique_ptr<VariadicABIInfo> create(llvm::Triple const &Triple);

  // Whether a valist instance is passed by value or by address
  // I.e. does it need to be alloca'ed and stored into, or can
  // it be passed directly in a SSA register
  virtual bool vaListPassedInSSARegister() = 0;

  // The type of a va_list iterator object
  virtual Type *vaListType(LLVMContext &Ctx) = 0;

  // The type of a va_list as a function argument as lowered by C
  virtual Type *vaListParameterType(Module &M) = 0;

  // Initialize an allocated va_list object to point to an already
  // initialized contiguous memory region.
  // Return the value to pass as the va_list argument
  virtual Value *initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                                  AllocaInst *, Value * /*buffer*/) = 0;

  struct VAArgSlotInfo {
    Align Align;   // With respect to the call frame
    bool Indirect; // Passed via a pointer
    bool Unknown;  // Cannot analyse this type, cannot transform the call
  };
  virtual VAArgSlotInfo slotInfo(const DataLayout &DL, Type *Parameter) = 0;

  // Targets implemented so far all have the same trivial lowering for these
  bool vaEndIsNop() { return true; }
  bool vaCopyIsMemcpy() { return true; }

  virtual ~VariadicABIInfo() {}
};

// Module implements getFunction() which returns nullptr on missing declaration
// and getOrInsertFunction which creates one when absent. Intrinsics.h only
// implements getDeclaration which creates one when missing. Checking whether
// an intrinsic exists thus inserts it in the module and it then needs to be
// deleted again to clean up.
// The right name for the two functions on intrinsics would match Module::,
// but doing that in a single change would introduce nullptr dereferences
// where currently there are none. The minimal collateral damage approach
// would split the change over a release to help downstream branches. As it
// is unclear what approach will be preferred, implementing the trivial
// function here in the meantime to decouple from that discussion.
Function *getPreexistingDeclaration(Module *M, Intrinsic::ID id,
                                    ArrayRef<Type *> Tys = std::nullopt) {
  auto *FT = Intrinsic::getType(M->getContext(), id, Tys);
  return M->getFunction(Tys.empty() ? Intrinsic::getName(id)
                                    : Intrinsic::getName(id, Tys, M, FT));
}

class ExpandVariadics : public ModulePass {

  // The pass construction sets the default to optimize when called from middle
  // end and lowering when called from the backend. The command line variable
  // overrides that. This is useful for testing and debugging. It also allows
  // building an applications with variadic functions wholly removed if one
  // has sufficient control over the dependencies, e.g. a statically linked
  // clang that has no variadic function calls remaining in the binary.
  static ExpandVariadicsMode
  withCommandLineOverride(ExpandVariadicsMode LLVMRequested) {
    ExpandVariadicsMode UserRequested = ExpandVariadicsModeOption;
    return (UserRequested == ExpandVariadicsMode::Unspecified) ? LLVMRequested
                                                               : UserRequested;
  }

public:
  static char ID;
  const ExpandVariadicsMode Mode;
  std::unique_ptr<VariadicABIInfo> ABI;

  ExpandVariadics(ExpandVariadicsMode Mode)
      : ModulePass(ID), Mode(withCommandLineOverride(Mode)) {}
  StringRef getPassName() const override { return "Expand variadic functions"; }

  // Rewrite a variadic call site
  bool expandCall(Module &M, IRBuilder<> &Builder, CallBase *CB, FunctionType *,
                  Function *NF);

  Function *replaceAllUsesWithNewDeclaration(Module &M,
                                             Function *OriginalFunction);
  Function *deriveFixedArityReplacement(Module &M, IRBuilder<> &Builder,
                                        Function *OriginalFunction);
  Function *defineVariadicWrapper(Module &M, IRBuilder<> &Builder,
                                  Function *VariadicWrapper,
                                  Function *FixedArityReplacement);

  bool runOnModule(Module &M) override;
  bool runOnFunction(Module &M, IRBuilder<> &Builder, Function *F);

  bool rewriteABI() { return Mode == ExpandVariadicsMode::Lowering; }

  void memcpyVAListPointers(const DataLayout &DL, IRBuilder<> &Builder,
                            Value *Dst, Value *Src) {
    auto &Ctx = Builder.getContext();
    Type *VaListTy = ABI->vaListType(Ctx);
    uint64_t Size = DL.getTypeAllocSize(VaListTy).getFixedValue();
    Builder.CreateMemCpyInline(Dst, {}, Src, {},
                               ConstantInt::get(Type::getInt32Ty(Ctx), Size));
  }

  template <Intrinsic::ID ID, typename InstructionType>
  bool expandIntrinsicUsers(Module &M, IRBuilder<> &Builder,
                            PointerType *ArgType) {
    bool Changed = false;
    const DataLayout &DL = M.getDataLayout();
    if (Function *Intrinsic = getPreexistingDeclaration(&M, ID, {ArgType})) {
      for (User *U : llvm::make_early_inc_range(Intrinsic->users())) {
        if (auto *I = dyn_cast<InstructionType>(U)) {
          Changed |= expandVAIntrinsicCall(Builder, DL, I);
        }
      }
      if (Intrinsic->use_empty())
        Intrinsic->eraseFromParent();
    }
    return Changed;
  }

  bool expandVAIntrinsicCall(IRBuilder<> &Builder, const DataLayout &DL,
                             VAStartInst *Inst);

  bool expandVAIntrinsicCall(IRBuilder<> &, const DataLayout &,
                             VAEndInst *Inst);

  bool expandVAIntrinsicCall(IRBuilder<> &Builder, const DataLayout &DL,
                             VACopyInst *Inst);

  FunctionType *inlinableVariadicFunctionType(Module &M, FunctionType *FTy) {
    // The type of "FTy" with the ... removed and a va_list appended
    SmallVector<Type *> ArgTypes(FTy->param_begin(), FTy->param_end());
    ArgTypes.push_back(ABI->vaListParameterType(M));
    bool IsVarArgs = false;
    return FunctionType::get(FTy->getReturnType(), ArgTypes, IsVarArgs);
  }

  static ConstantInt *sizeOfAlloca(LLVMContext &Ctx, const DataLayout &DL,
                                   AllocaInst *Alloced) {
    Type *AllocaType = Alloced->getAllocatedType();
    TypeSize AllocaTypeSize = DL.getTypeAllocSize(AllocaType);
    uint64_t AsInt = AllocaTypeSize.getFixedValue();
    return ConstantInt::get(Type::getInt64Ty(Ctx), AsInt);
  }

  bool expansionApplicableToFunction(Module &M, Function *F) {
    if (F->isIntrinsic() || !F->isVarArg() ||
        F->hasFnAttribute(Attribute::Naked)) {
      return false;
    }

    if (F->getCallingConv() != CallingConv::C)
      return false;

    if (!rewriteABI()) {
      // e.g. can't replace a weak function unless changing the original symbol
      if (GlobalValue::isInterposableLinkage(F->getLinkage())) {
        return false;
      }
    }

    if (!rewriteABI()) {
      // If optimising, err on the side of leaving things alone
      for (const Use &U : F->uses()) {
        const auto *CB = dyn_cast<CallBase>(U.getUser());

        if (!CB)
          return false;

        if (CB->isMustTailCall())
          return false;

        if (!CB->isCallee(&U) ||
            CB->getFunctionType() != F->getFunctionType()) {
          return false;
        }
      }
    }

    // Branch funnels look like variadic functions but aren't:
    //
    // define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...) {
    //  musttail call void (...) @llvm.icall.branch.funnel(ptr %0, ptr @vt1_1,
    //  ptr @vf1_1, ...) ret void
    // }
    //
    // %1 = call i32 @__typeid_typeid1_0_branch_funnel(ptr nest %vtable, ptr
    // %obj, i32 1)
    //
    // If this function contains a branch funnel intrinsic, don't transform it.

    if (Function *Funnel =
            getPreexistingDeclaration(&M, Intrinsic::icall_branch_funnel)) {
      for (const User *U : Funnel->users()) {
        if (auto *I = dyn_cast<CallBase>(U)) {
          if (F == I->getFunction()) {
            return false;
          }
        }
      }
    }

    return true;
  }

  bool callinstRewritable(CallBase *CB) {
    if (CallInst *CI = dyn_cast<CallInst>(CB)) {
      if (CI->isMustTailCall()) {
        // Cannot expand musttail calls
        return false;
      }

      return true;
    }

    if (isa<InvokeInst>(CB)) {
      // Invoke not implemented in initial implementation of pass
      return false;
    }

    // Other unimplemented derivative of CallBase
    return false;
  }

  class ExpandedCallFrame {
    // Helper for constructing an alloca instance containing the arguments bound
    // to the variadic ... parameter, rearranged to allow indexing through a
    // va_list iterator
    enum { N = 4 };
    SmallVector<Type *, N> FieldTypes;
    enum Tag { Store, Memcpy, Padding };
    SmallVector<std::tuple<Value *, uint64_t, Tag>, N> Source;

    template <Tag tag> void append(Type *FieldType, Value *V, uint64_t Bytes) {
      FieldTypes.push_back(FieldType);
      Source.push_back({V, Bytes, tag});
    }

  public:
    void store(LLVMContext &Ctx, Type *T, Value *V) { append<Store>(T, V, 0); }

    void memcpy(LLVMContext &Ctx, Type *T, Value *V, uint64_t Bytes) {
      append<Memcpy>(T, V, Bytes);
    }

    void padding(LLVMContext &Ctx, uint64_t By) {
      append<Padding>(ArrayType::get(Type::getInt8Ty(Ctx), By), nullptr, 0);
    }

    size_t size() const { return FieldTypes.size(); }
    bool empty() const { return FieldTypes.empty(); }

    StructType *asStruct(LLVMContext &Ctx, StringRef Name) {
      const bool IsPacked = true;
      return StructType::create(Ctx, FieldTypes,
                                (Twine(Name) + ".vararg").str(), IsPacked);
    }

    void initializeStructAlloca(const DataLayout &DL, IRBuilder<> &Builder,
                                AllocaInst *Alloced) {

      StructType *VarargsTy = cast<StructType>(Alloced->getAllocatedType());

      for (size_t I = 0; I < size(); I++) {

        auto [V, bytes, tag] = Source[I];

        if (tag == Padding) {
          assert(V == nullptr);
          continue;
        }

        auto Dst = Builder.CreateStructGEP(VarargsTy, Alloced, I);

        assert(V != nullptr);

        if (tag == Store) {
          Builder.CreateStore(V, Dst);
        }

        if (tag == Memcpy) {
          Builder.CreateMemCpy(Dst, {}, V, {}, bytes);
        }
      }
    }
  };
};

bool ExpandVariadics::runOnModule(Module &M) {
  bool Changed = false;

  if (Mode == ExpandVariadicsMode::Disable)
    return Changed;

  llvm::Triple Triple(M.getTargetTriple());

  if (Triple.getArch() == Triple::UnknownArch) {
    // If we don't know the triple, we can't lower varargs
    return false;
  }

  ABI = VariadicABIInfo::create(Triple);
  if (!ABI) {
    if (Mode == ExpandVariadicsMode::Lowering) {
      report_fatal_error(
          "Requested variadic lowering is unimplemented on this target");
    }
    return Changed;
  }

  auto &Ctx = M.getContext();
  IRBuilder<> Builder(Ctx);

  // At pass input, va_start intrinsics only occur in variadic functions, as
  // checked by the IR verifier.

  // The lowering pass needs to run on all variadic functions.
  // The optimise could run on only those that call va_start
  // in exchange for additional book keeping to avoid transforming
  // the same function multiple times when it contains multiple va_start.
  // Leaving that compile time optimisation for a later patch.

  for (Function &F : llvm::make_early_inc_range(M))
    Changed |= runOnFunction(M, Builder, &F);

  // After runOnFunction, all known calls to known variadic functions have been
  // replaced. va_start intrinsics are presently (and invalidly!) only present
  // in functions that used to be variadic and have now been replaced to take a
  // va_list instead. If lowering as opposed to optimising, calls to unknown
  // variadic functions have also been replaced.

  unsigned Addrspace = 0; // Sufficient for current targets
  {
    PointerType *ArgType = PointerType::get(Ctx, Addrspace);
    // expand vastart before vacopy as vastart may introduce a vacopy
    Changed |= expandIntrinsicUsers<Intrinsic::vastart, VAStartInst>(M, Builder,
                                                                     ArgType);
    Changed |=
        expandIntrinsicUsers<Intrinsic::vaend, VAEndInst>(M, Builder, ArgType);
    Changed |= expandIntrinsicUsers<Intrinsic::vacopy, VACopyInst>(M, Builder,
                                                                   ArgType);
  }

  if (Mode != ExpandVariadicsMode::Lowering) {
    return Changed; // Done
  }

  for (Function &F : llvm::make_early_inc_range(M)) {
    if (F.isDeclaration())
      continue;

    // Now need to track down indirect calls. Can't find those
    // by walking uses of variadic functions, need to crawl the instruction
    // stream. Fortunately this is only necessary for the ABI rewrite case.
    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        if (CallBase *CB = dyn_cast<CallBase>(&I)) {
          if (CB->isIndirectCall()) {
            FunctionType *FTy = CB->getFunctionType();
            if (FTy->isVarArg()) {
              Changed |= expandCall(M, Builder, CB, FTy, 0);
            }
          }
        }
      }
    }
  }

  return Changed;
}

bool ExpandVariadics::runOnFunction(Module &M, IRBuilder<> &Builder,
                                    Function *OriginalFunction) {
  bool Changed = false;

  // fprintf(stderr, "Called runOn: %s\n",
  // OriginalFunction->getName().str().c_str());

  // TODO: Check what F.hasExactDefinition() does

  // This check might be too coarse - there are probably cases where
  // splitting a function is bad but it's usable without splitting
  if (!expansionApplicableToFunction(M, OriginalFunction))
    return false;

  // TODO: Leave "thunk" attribute functions alone?

  // Need more tests than this. Weak etc. Some are in expansionApplicable.

  if (OriginalFunction->isDeclaration()) {
    if (Mode == ExpandVariadicsMode::Optimize) {
      return false;
    }
  }

  const bool OriginalFunctionIsDeclaration = OriginalFunction->isDeclaration();

  // Declare a new function and redirect every use to that new function
  Function *VariadicWrapper =
      replaceAllUsesWithNewDeclaration(M, OriginalFunction);
  assert(VariadicWrapper->isDeclaration());
  assert(OriginalFunction->use_empty());

  // Create a new function taking va_list containing the implementation of the
  // original
  Function *FixedArityReplacement =
      deriveFixedArityReplacement(M, Builder, OriginalFunction);
  assert(OriginalFunction->isDeclaration());
  assert(FixedArityReplacement->isDeclaration() ==
         OriginalFunctionIsDeclaration);
  assert(VariadicWrapper->isDeclaration());

  // Create a single block forwarding wrapper that turns a ... into a va_list
  Function *VariadicWrapperDefine =
      defineVariadicWrapper(M, Builder, VariadicWrapper, FixedArityReplacement);
  assert(VariadicWrapperDefine == VariadicWrapper);
  assert(!VariadicWrapper->isDeclaration());

  // We now have:
  // 1. the original function, now as a declaration with no uses
  // 2. a variadic function that unconditionally calls a fixed arity replacement
  // 3. a fixed arity function equivalent to the original function

  // Replace known calls to the variadic with calls to the va_list equivalent
  for (User *U : llvm::make_early_inc_range(VariadicWrapper->users())) {
    if (CallBase *CB = dyn_cast<CallBase>(U)) {
      Value *calledOperand = CB->getCalledOperand();
      if (VariadicWrapper == calledOperand) {
        Changed |=
            expandCall(M, Builder, CB, VariadicWrapper->getFunctionType(),
                       FixedArityReplacement);
      }
    }
  }

  Function *const ExternallyAccessible =
      rewriteABI() ? FixedArityReplacement : VariadicWrapper;
  Function *const InternalOnly =
      rewriteABI() ? VariadicWrapper : FixedArityReplacement;

  // care needed over other attributes, metadata etc

  ExternallyAccessible->setLinkage(OriginalFunction->getLinkage());
  ExternallyAccessible->setVisibility(OriginalFunction->getVisibility());
  ExternallyAccessible->setComdat(OriginalFunction->getComdat());
  ExternallyAccessible->takeName(OriginalFunction);

  InternalOnly->setVisibility(GlobalValue::DefaultVisibility);
  InternalOnly->setLinkage(GlobalValue::InternalLinkage);

  OriginalFunction->eraseFromParent();

  InternalOnly->removeDeadConstantUsers();

  if (rewriteABI()) {
    // All known calls to the function have been removed by expandCall
    // Resolve everything else by replace all uses with

    VariadicWrapper->replaceAllUsesWith(FixedArityReplacement);

    assert(VariadicWrapper->use_empty());
    VariadicWrapper->eraseFromParent();
  }

  return Changed;
}

Function *
ExpandVariadics::replaceAllUsesWithNewDeclaration(Module &M,
                                                  Function *OriginalFunction) {
  auto &Ctx = M.getContext();
  Function &F = *OriginalFunction;
  FunctionType *FTy = F.getFunctionType();
  Function *NF = Function::Create(FTy, F.getLinkage(), F.getAddressSpace());

  NF->setName(F.getName() + ".varargs");
  NF->IsNewDbgInfoFormat = F.IsNewDbgInfoFormat;

  // Could give it the same visibility/linkage as the original
  F.getParent()->getFunctionList().insert(F.getIterator(), NF);

  // might have a shorthand
  AttrBuilder ParamAttrs(Ctx);
  AttributeList Attrs = NF->getAttributes();
  Attrs = Attrs.addParamAttributes(Ctx, FTy->getNumParams(), ParamAttrs);
  NF->setAttributes(Attrs);

  OriginalFunction->replaceAllUsesWith(NF);
  return NF;
}

Function *
ExpandVariadics::deriveFixedArityReplacement(Module &M, IRBuilder<> &Builder,
                                             Function *OriginalFunction) {
  Function &F = *OriginalFunction;
  // The purpose here is split the variadic function F into two functions
  // One is a variadic function that bundles the passed argument into a va_list
  // and passes it to the second function. The second function does whatever
  // the original F does, except that it takes a va_list instead of the ...

  assert(expansionApplicableToFunction(M, &F));

  auto &Ctx = M.getContext();

  // Returned value isDeclaration() is equal to F.isDeclaration()
  // but that invariant is not satisfied throughout this function
  const bool FunctionIsDefinition = !F.isDeclaration();

  FunctionType *FTy = F.getFunctionType();
  SmallVector<Type *> ArgTypes(FTy->param_begin(), FTy->param_end());
  ArgTypes.push_back(ABI->vaListParameterType(M));

  FunctionType *NFTy = inlinableVariadicFunctionType(M, FTy);
  Function *NF = Function::Create(NFTy, F.getLinkage(), F.getAddressSpace());

  // Note - same attribute handling as DeadArgumentElimination
  NF->copyAttributesFrom(&F);
  //  NF->setComdat(F.getComdat()); // beware weak
  F.getParent()->getFunctionList().insert(F.getIterator(), NF);
  NF->setName(F.getName() + ".valist");
  NF->IsNewDbgInfoFormat = F.IsNewDbgInfoFormat;

  AttrBuilder ParamAttrs(Ctx);

  AttributeList Attrs = NF->getAttributes();
  Attrs = Attrs.addParamAttributes(Ctx, NFTy->getNumParams() - 1, ParamAttrs);
  NF->setAttributes(Attrs);

  // Splice the implementation into the new function with minimal changes
  if (FunctionIsDefinition) {
    NF->splice(NF->begin(), &F);

    auto NewArg = NF->arg_begin();
    for (Argument &Arg : F.args()) {
      Arg.replaceAllUsesWith(NewArg);
      NewArg->setName(Arg.getName()); // takeName without killing the old one
      ++NewArg;
    }
    NewArg->setName("varargs");
  }

  SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
  F.getAllMetadata(MDs);
  for (auto [KindID, Node] : MDs)
    NF->addMetadata(KindID, *Node);
  F.clearMetadata();

  return NF;
}

Function *
ExpandVariadics::defineVariadicWrapper(Module &M, IRBuilder<> &Builder,
                                       Function *VariadicWrapper,
                                       Function *FixedArityReplacement) {
  auto &Ctx = Builder.getContext();
  const DataLayout &DL = M.getDataLayout();
  assert(VariadicWrapper->isDeclaration());
  Function &F = *VariadicWrapper;

  assert(F.isDeclaration());
  Type *VaListTy = ABI->vaListType(Ctx);

  auto *BB = BasicBlock::Create(Ctx, "entry", &F);
  Builder.SetInsertPoint(BB);

  AllocaInst *VaListInstance =
      Builder.CreateAlloca(VaListTy, nullptr, "va_list");

  Builder.CreateLifetimeStart(VaListInstance,
                              sizeOfAlloca(Ctx, DL, VaListInstance));

  Builder.CreateIntrinsic(Intrinsic::vastart, {DL.getAllocaPtrType(Ctx)},
                          {VaListInstance});

  SmallVector<Value *> Args;
  for (Argument &A : F.args())
    Args.push_back(&A);

  Args.push_back(VaListInstance);

  CallInst *Result = Builder.CreateCall(FixedArityReplacement, Args);
  Result->setTailCallKind(CallInst::TCK_Tail);

  Builder.CreateIntrinsic(Intrinsic::vaend, {DL.getAllocaPtrType(Ctx)},
                          {VaListInstance});
  Builder.CreateLifetimeEnd(VaListInstance,
                            sizeOfAlloca(Ctx, DL, VaListInstance));

  if (Result->getType()->isVoidTy())
    Builder.CreateRetVoid();
  else
    Builder.CreateRet(Result);

  return VariadicWrapper;
}

bool ExpandVariadics::expandCall(Module &M, IRBuilder<> &Builder, CallBase *CB,
                                 FunctionType *VarargFunctionType,
                                 Function *NF) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();

  if (!callinstRewritable(CB)) {
    if (rewriteABI()) {
      report_fatal_error("Cannot lower callbase instruction");
    }
    return Changed;
  }

  // This is tricky. The call instruction's function type might not match
  // the type of the caller. When optimising, can leave it unchanged.
  // Webassembly detects that inconsistency and repairs it.
  FunctionType *FuncType = CB->getFunctionType();
  if (FuncType != VarargFunctionType) {
    if (!rewriteABI()) {
      return Changed;
    }
    FuncType = VarargFunctionType;
  }

  auto &Ctx = CB->getContext();

  Align MaxFieldAlign(1);

  // The strategy is to allocate a call frame containing the variadic
  // arguments laid out such that a target specific va_list can be initialized
  // with it, such that target specific va_arg instructions will correctly
  // iterate over it. This means getting the alignment right and sometimes
  // embedding a pointer to the value instead of embedding the value itself.

  Function *CBF = CB->getParent()->getParent();

  ExpandedCallFrame Frame;

  uint64_t CurrentOffset = 0;

  for (unsigned I = FuncType->getNumParams(), E = CB->arg_size(); I < E; ++I) {
    Value *ArgVal = CB->getArgOperand(I);
    bool IsByVal = CB->paramHasAttr(I, Attribute::ByVal);

    // The call argument is either passed by value, or is a pointer passed byval
    // The varargs frame either stores the value directly or a pointer to it

    // The type of the value being passed, decoded from byval metadata if
    // required
    Type *const UnderlyingType =
        IsByVal ? CB->getParamByValType(I) : ArgVal->getType();
    const uint64_t UnderlyingSize =
        DL.getTypeAllocSize(UnderlyingType).getFixedValue();

    // The type to be written into the call frame
    Type *FrameFieldType = UnderlyingType;

    // The value to copy from when initialising the frame alloca
    Value *SourceValue = ArgVal;

    // TODO, slotInfo should probably return the right alignment even
    // when returning true for indirect, somewhat messy
    VariadicABIInfo::VAArgSlotInfo slotInfo = ABI->slotInfo(DL, UnderlyingType);

#if 0
    {
      fprintf(stdout, "Underlying type for param %u (byval %u, indir %u)\n", I,
              IsByVal, slotInfo.Indirect);
      UnderlyingType->dump();
    }
#endif

    if (slotInfo.Unknown) {
      if (rewriteABI()) {
        report_fatal_error("Variadic lowering unimplemented on given type");
      } else {
        return Changed;
      }
    }

    if (slotInfo.Indirect) {
      // The va_arg lowering loads through a pointer. Set up an alloca to aim
      // that pointer at.
      Builder.SetInsertPointPastAllocas(CBF);
      Builder.SetCurrentDebugLocation(CB->getStableDebugLoc());
      Value *CallerCopy =
          Builder.CreateAlloca(UnderlyingType, nullptr, "IndirectAlloca");

      Builder.SetInsertPoint(CB);
      if (IsByVal)
        Builder.CreateMemCpy(CallerCopy, {}, ArgVal, {}, UnderlyingSize);
      else
        Builder.CreateStore(ArgVal, CallerCopy);

      // Indirection now handled, pass the alloca ptr by value
      FrameFieldType = DL.getAllocaPtrType(Ctx);
      SourceValue = CallerCopy;
    }

    // Alignment of the value within the frame
    // This probably needs to be controllable as a function of type
    Align DataAlign = slotInfo.Align;

    MaxFieldAlign = std::max(MaxFieldAlign, DataAlign);

    uint64_t DataAlignV = DataAlign.value();
    if (uint64_t Rem = CurrentOffset % DataAlignV) {
      // Inject explicit padding to deal with alignment requirements
      uint64_t Padding = DataAlignV - Rem;
      Frame.padding(Ctx, Padding);
      CurrentOffset += Padding;
    }

    if (slotInfo.Indirect) {
      Frame.store(Ctx, FrameFieldType, SourceValue);
    } else {
      if (IsByVal) {
        Frame.memcpy(Ctx, FrameFieldType, SourceValue, UnderlyingSize);
      } else {
        Frame.store(Ctx, FrameFieldType, SourceValue);
      }
    }

    CurrentOffset += DL.getTypeAllocSize(FrameFieldType).getFixedValue();
  }

  if (Frame.empty()) {
    // Not passing any arguments, hopefully va_arg won't try to read any
    // Creating a single byte frame containing nothing to point the va_list
    // instance as that is less special-casey in the compiler and probably
    // easier to interpret in a debugger.
    Frame.padding(Ctx, 1);
  }

  StructType *VarargsTy = Frame.asStruct(Ctx, CBF->getName());

  // The struct instance needs to be at least MaxFieldAlign for the alignment of
  // the fields to be correct at runtime. Use the native stack alignment instead
  // if that's greater as that tends to give better codegen.
  // This is an awkward way to guess whether there is a known stack alignment
  // without hitting an assert in DL.getStackAlignment, 1024 is an arbitrary
  // number likely to be greater than the natural stack alignment.
  // TODO: DL.getStackAlignment could return a MaybeAlign instead of assert
  Align AllocaAlign = MaxFieldAlign;
  if (DL.exceedsNaturalStackAlignment(Align(1024))) {
    AllocaAlign = std::max(AllocaAlign, DL.getStackAlignment());
  }

  // Put the alloca to hold the variadic args in the entry basic block.
  Builder.SetInsertPointPastAllocas(CBF);

  // SetCurrentDebugLocation when the builder SetInsertPoint method does not
  Builder.SetCurrentDebugLocation(CB->getStableDebugLoc());

  // The awkward construction here is to set the alignment on the instance
  Changed = true;
  AllocaInst *Alloced = Builder.Insert(
      new AllocaInst(VarargsTy, DL.getAllocaAddrSpace(), nullptr, AllocaAlign),
      "vararg_buffer");
  assert(Alloced->getAllocatedType() == VarargsTy);

  // Initialize the fields in the struct
  Builder.SetInsertPoint(CB);
  Builder.CreateLifetimeStart(Alloced, sizeOfAlloca(Ctx, DL, Alloced));
  Frame.initializeStructAlloca(DL, Builder, Alloced);

  const unsigned NumArgs = FuncType->getNumParams();
  SmallVector<Value *> Args;
  Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);

  // Initialize a va_list pointing to that struct and pass it as the last
  // argument
  AllocaInst *VaList = nullptr;
  {
    if (!ABI->vaListPassedInSSARegister()) {
      Type *VaListTy = ABI->vaListType(Ctx);
      Builder.SetInsertPointPastAllocas(CBF);
      Builder.SetCurrentDebugLocation(CB->getStableDebugLoc());
      VaList = Builder.CreateAlloca(VaListTy, nullptr, "va_list");
      Builder.SetInsertPoint(CB);
      Builder.CreateLifetimeStart(VaList, sizeOfAlloca(Ctx, DL, VaList));
    }
    Args.push_back(ABI->initializeVAList(Ctx, Builder, VaList, Alloced));
  }

  // Attributes excluding any on the vararg arguments
  AttributeList PAL = CB->getAttributes();
  if (!PAL.isEmpty()) {
    SmallVector<AttributeSet, 8> ArgAttrs;
    for (unsigned ArgNo = 0; ArgNo < NumArgs; ArgNo++)
      ArgAttrs.push_back(PAL.getParamAttrs(ArgNo));
    PAL =
        AttributeList::get(Ctx, PAL.getFnAttrs(), PAL.getRetAttrs(), ArgAttrs);
  }

  SmallVector<OperandBundleDef, 1> OpBundles;
  CB->getOperandBundlesAsDefs(OpBundles);

  CallBase *NewCB = nullptr;

  // Assert won't be true once InvokeInst is implemented in a later patch,
  // current invariant is established by callinstRewritable() at the start
  assert(isa<CallInst>(CB));

  if (CallInst *CI = dyn_cast<CallInst>(CB)) {

    Value *Dst = NF ? NF : CI->getCalledOperand();
    FunctionType *NFTy = inlinableVariadicFunctionType(M, VarargFunctionType);

    NewCB = CallInst::Create(NFTy, Dst, Args, OpBundles, "", CI);

    CallInst::TailCallKind TCK = CI->getTailCallKind();
    assert(TCK != CallInst::TCK_MustTail); // guarded at prologue

    // It doesn't get to be a tail call any more
    // might want to guard this with arch, x64 and aarch64 document that
    // varargs can't be tail called anyway
    // Not totally convinced this is necessary but dead store elimination
    // will discard the stores to the Alloca and pass uninitialized memory along
    // instead when the function is marked tailcall
    if (TCK == CallInst::TCK_Tail) {
      TCK = CallInst::TCK_None;
    }
    CI->setTailCallKind(TCK);

  } else {
    llvm_unreachable("unreachable because callinstRewritable() returned false");
  }

  if (VaList)
    Builder.CreateLifetimeEnd(VaList, sizeOfAlloca(Ctx, DL, VaList));

  Builder.CreateLifetimeEnd(Alloced, sizeOfAlloca(Ctx, DL, Alloced));

  NewCB->setAttributes(PAL);
  NewCB->takeName(CB);
  NewCB->setCallingConv(CB->getCallingConv());

  NewCB->setDebugLoc(DebugLoc());

  // DeadArgElim and ArgPromotion copy exactly this metadata
  NewCB->copyMetadata(*CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});

  CB->replaceAllUsesWith(NewCB);
  CB->eraseFromParent();
  return Changed;
}

bool ExpandVariadics::expandVAIntrinsicCall(IRBuilder<> &Builder,
                                            const DataLayout &DL,
                                            VAStartInst *Inst) {
  // TODO: Document or remove this action at a distance trickery
  Function *ContainingFunction = Inst->getFunction();
  if (ContainingFunction->isVarArg())
    return false;

  // The last argument is a vaListParameterType
  Argument *PassedVaList =
      ContainingFunction->getArg(ContainingFunction->arg_size() - 1);

  // va_start takes a pointer to a va_list, e.g. one on the stack
  Value *VaStartArg = Inst->getArgList();

  Builder.SetInsertPoint(Inst);

  // If the va_list is itself a ptr, emitting a vacopy call requires an alloca
  // which is then removed, simpler to build the store directly.
  if (ABI->vaListPassedInSSARegister()) {
    Builder.CreateStore(PassedVaList, VaStartArg);
  } else {
    // Otherwise emit a vacopy to pick up target-specific handling if any
    auto &Ctx = Builder.getContext();
    Builder.CreateIntrinsic(Intrinsic::vacopy, {DL.getAllocaPtrType(Ctx)},
                            {VaStartArg, PassedVaList});
  }

  Inst->eraseFromParent();
  return true;
}

bool ExpandVariadics::expandVAIntrinsicCall(IRBuilder<> &, const DataLayout &,
                                            VAEndInst *Inst) {
  assert(ABI->vaEndIsNop());
  // A no-op on all the architectures implemented so far
  Inst->eraseFromParent();
  return true;
}

bool ExpandVariadics::expandVAIntrinsicCall(IRBuilder<> &Builder,
                                            const DataLayout &DL,
                                            VACopyInst *Inst) {
  // TODO: This looks be wrong for non-struct va_list, check it using wasm
  assert(ABI->vaCopyIsMemcpy());
  Builder.SetInsertPoint(Inst);
  memcpyVAListPointers(DL, Builder, Inst->getDest(), Inst->getSrc());
  Inst->eraseFromParent();
  return true;
}

template <uint32_t MinAlign, uint32_t MaxAlign> Align clampAlign(Align A) {
  // Uses 0 as a sentinel to mean inactive
  if (MinAlign && A < MinAlign)
    A = Align(MinAlign);

  if (MaxAlign && A > MaxAlign)
    A = Align(MaxAlign);

  return A;
}

bool simpleScalarType(Type *Parameter) {
  // This is a stop-gap. The MVP can optimise x64 and aarch64 on linux
  // for sufficiently simple calls.
  if (Parameter->isDoubleTy())
    return true;

  if (Parameter->isIntegerTy(32))
    return true;
  if (Parameter->isIntegerTy(64))
    return true;

  if (Parameter->isPointerTy()) {
    return true;
  }

  return false;
}

struct AArch64 final : public VariadicABIInfo {
  // https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
  // big endian, little endian ILP32 have their own triples

  bool vaListPassedInSSARegister() override { return false; }

  Type *vaListType(LLVMContext &Ctx) override {
#if 0
    typedef struct  va_list {
      void * stack; // next stack param
      void * gr_top; // end of GP arg reg save area
      void * vr_top; // end of FP/SIMD arg reg save area
      int gr_offs; // offset from  gr_top to next GP register arg
      int vr_offs; // offset from  vr_top to next FP/SIMD register arg
    } va_list;
#endif

    auto I32 = Type::getInt32Ty(Ctx);
    auto Ptr = PointerType::getUnqual(Ctx);

    return StructType::get(Ctx, {Ptr, Ptr, Ptr, I32, I32});
  }

  Type *vaListParameterType(Module &M) override {
    return PointerType::getUnqual(M.getContext());
  }

  Value *initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                          AllocaInst *VaList, Value *VoidBuffer) override {
    assert(VaList->getAllocatedType() == vaListType(Ctx));

    Type *VaListTy = vaListType(Ctx);
    Type *I32 = Type::getInt32Ty(Ctx);
    Constant *Zero = ConstantInt::get(I32, 0);
    Constant *Null = ConstantPointerNull::get(PointerType::getUnqual(Ctx));

    Value *Idxs[2] = {
        ConstantInt::get(I32, 0),
        nullptr,
    };

    Idxs[1] = ConstantInt::get(I32, 0);
    Builder.CreateStore(
        VoidBuffer, Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "stack"));

    // The general and vector regions are unused, given by the zero offsets,
    // with nullptr a reasonable value to use for the pointer fields. That is
    // all arguments are packed into the "stack" area, leaving the specialised
    // two area unused.

    Idxs[1] = ConstantInt::get(I32, 1);
    Builder.CreateStore(
        Null, Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "gr_top"));

    Idxs[1] = ConstantInt::get(I32, 2);
    Builder.CreateStore(
        Null, Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "vr_top"));

    Idxs[1] = ConstantInt::get(I32, 3);
    Builder.CreateStore(
        Zero, Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "gr_offs"));

    Idxs[1] = ConstantInt::get(I32, 4);
    Builder.CreateStore(
        Zero, Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "vr_offs"));

    return VaList;
  }

  VAArgSlotInfo slotInfo(const DataLayout &DL, Type *Parameter) override {
    Align A = clampAlign<8, 0u>(DL.getABITypeAlign(Parameter));

    bool Indirect = false; // true for some non-simple types on aarch64
    bool Unknown = !simpleScalarType(Parameter);
    return {A, Indirect, Unknown};
  }
};

struct Wasm final : public VariadicABIInfo {
  bool vaListPassedInSSARegister() override { return true; }

  Type *vaListType(LLVMContext &Ctx) override {
    return PointerType::getUnqual(Ctx);
  }

  Type *vaListParameterType(Module &M) override {
    return PointerType::getUnqual(M.getContext());
  }

  Value *initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                          AllocaInst * /*va_list*/, Value *buffer) override {
    return buffer;
  }

  VAArgSlotInfo slotInfo(const DataLayout &DL, Type *Parameter) override {
    LLVMContext &Ctx = Parameter->getContext();
    Align A = clampAlign<4, 0>(DL.getABITypeAlign(Parameter));

    // TODO, test empty record
    if (auto s = dyn_cast<StructType>(Parameter)) {
      if (s->getNumElements() > 1) {
        return {DL.getABITypeAlign(PointerType::getUnqual(Ctx)), true, false};
      }
    }

    return {A, false, false};
  }
};

struct X64SystemV final : public VariadicABIInfo {
  bool vaListPassedInSSARegister() override { return false; }

  Type *vaListType(LLVMContext &Ctx) override {
    auto I32 = Type::getInt32Ty(Ctx);
    auto Ptr = PointerType::getUnqual(Ctx);
    return ArrayType::get(StructType::get(Ctx, {I32, I32, Ptr, Ptr}), 1);
  }

  Type *vaListParameterType(Module &M) override {
    return PointerType::getUnqual(M.getContext());
  }

  Value *initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                          AllocaInst *VaList, Value *VoidBuffer) override {
    assert(VaList->getAllocatedType() == vaListType(Ctx));

    Type *VaListTy = vaListType(Ctx);

    Type *I32 = Type::getInt32Ty(Ctx);
    Type *I64 = Type::getInt64Ty(Ctx);

    Value *Idxs[3] = {
        ConstantInt::get(I64, 0),
        ConstantInt::get(I32, 0),
        nullptr,
    };

    // The magic numbers here set up a va_list instance that has the general
    // purpose and floating point regions empty, such that only the overflow
    // area is used. That means a single contiguous struct can be the backing
    // store and simpler code to optimise in the inlining case.

    Idxs[2] = ConstantInt::get(I32, 0);
    Builder.CreateStore(
        ConstantInt::get(I32, 48),
        Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "gp_offset"));

    Idxs[2] = ConstantInt::get(I32, 1);
    Builder.CreateStore(
        ConstantInt::get(I32, 6 * 8 + 8 * 16),
        Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "fp_offset"));

    Idxs[2] = ConstantInt::get(I32, 2);
    Builder.CreateStore(
        VoidBuffer,
        Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "overfow_arg_area"));

    Idxs[2] = ConstantInt::get(I32, 3);
    Builder.CreateStore(
        ConstantPointerNull::get(PointerType::getUnqual(Ctx)),
        Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "reg_save_area"));

    return VaList;
  }

  VAArgSlotInfo slotInfo(const DataLayout &DL, Type *Parameter) override {
    // TODO: Make this comment look less scary
    // SystemV X64 documented behaviour:
    // Slots are at least eight byte aligned and at most 16 byte aligned.
    // If the type needs more than sixteen byte alignment, it still only gets
    // that much alignment on the stack.
    // X64 behaviour in clang:
    // Slots are at least eight byte aligned and at most naturally aligned
    // This matches clang, not the ABI docs.

    Align A = clampAlign<8, 0u>(DL.getABITypeAlign(Parameter));
    bool Indirect = false;
    bool Unknown = !simpleScalarType(Parameter);
    return {A, Indirect, Unknown};
  }
};

std::unique_ptr<VariadicABIInfo>
VariadicABIInfo::create(llvm::Triple const &Triple) {

  switch (Triple.getArch()) {

  case Triple::aarch64: {
    return std::make_unique<AArch64>();
  }

  case Triple::wasm32: {
    return std::make_unique<Wasm>();
  }

  case Triple::x86_64: {
    if (Triple.isOSLinux()) {
      return std::make_unique<X64SystemV>();
    }

    break;
  }

  default:
    break;
  }

  return {};
}

} // namespace

char ExpandVariadics::ID = 0;

INITIALIZE_PASS(ExpandVariadics, DEBUG_TYPE, "Expand variadic functions", false,
                false)

ModulePass *llvm::createExpandVariadicsPass(ExpandVariadicsMode M) {
  return new ExpandVariadics(M);
}

PreservedAnalyses ExpandVariadicsPass::run(Module &M, ModuleAnalysisManager &) {
  return ExpandVariadics(Mode).runOnModule(M) ? PreservedAnalyses::none()
                                              : PreservedAnalyses::all();
}

ExpandVariadicsPass::ExpandVariadicsPass(OptimizationLevel Level)
    : ExpandVariadicsPass(Level == OptimizationLevel::O0
                              ? ExpandVariadicsMode::Disable
                              : ExpandVariadicsMode::Optimize) {}

ExpandVariadicsPass::ExpandVariadicsPass(ExpandVariadicsMode M) : Mode(M) {}
