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
// The strategy is to turn the ... part of a varidic function into a va_list
// and fix up the call sites. This is completely effective if the calling
// convention can declare that to be the right thing, e.g. on GPUs or where
// the application is wholly statically linked. In the usual case, it will
// replace known calls to known variadic functions with calls that are amenable
// to inlining and other optimisations.
//
// The target-dependent parts are in class VariadicABIInfo. Enabling a new
// target means adding a case to VariadicABIInfo::create() along with tests.
// This will be especially simple if the va_list representation is a char*.
//
// The majority of the plumbing is splitting the variadic function into a
// single basic block that packs the variadic arguments into a va_list and
// a second function that does the work of the original. The target specific
// part is packing arguments into a contiguous buffer that the clang expansion
// of va_arg will do the right thing with.
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

#include <cstdio>

#define DEBUG_TYPE "expand-variadics"

using namespace llvm;

cl::opt<ExpandVariadicsMode> ExpandVariadicsModeOption(
    DEBUG_TYPE "-override", cl::desc("Override the behaviour of " DEBUG_TYPE),
    cl::init(ExpandVariadicsMode::unspecified),
    cl::values(clEnumValN(ExpandVariadicsMode::unspecified, "unspecified",
                          "Use the implementation defaults"),
               clEnumValN(ExpandVariadicsMode::disable, "disable",
                          "Disable the pass entirely"),
               clEnumValN(ExpandVariadicsMode::optimize, "optimize",
                          "Optimise without changing ABI"),
               clEnumValN(ExpandVariadicsMode::lowering, "lowering",
                          "Change variadic calling convention")));

namespace {

// Module implements getFunction() which returns nullptr on missing declaration
// and getOrInsertFunction which creates one when absent. Intrinsics.h
// implements getDeclaration which creates one when missing. This should be
// changed to be consistent with Module()'s naming. Implementing as a local
// function here in the meantime to decouple from that process.
Function *getPreexistingDeclaration(Module *M, Intrinsic::ID id,
                                    ArrayRef<Type *> Tys = std::nullopt) {
  auto *FT = Intrinsic::getType(M->getContext(), id, Tys);
  return M->getFunction(Tys.empty() ? Intrinsic::getName(id)
                                    : Intrinsic::getName(id, Tys, M, FT));
}

// Lots of targets use a void* pointed at a buffer for va_list.
// Some use more complicated iterator constructs. Type erase that
// so the rest of the pass can operation on either.
// Virtual functions where different targets want different behaviour,
// normal where all implemented targets presently have the same.
struct VAListInterface {
  virtual ~VAListInterface() {}

  // Whether a valist instance is passed by value or by address
  // I.e. does it need to be alloca'ed and stored into, or can
  // it be passed directly in a SSA register
  virtual bool passedInSSARegister() = 0;

  // The type of a va_list iterator object
  virtual Type *vaListType(LLVMContext &Ctx) = 0;

  // The type of a va_list as a function argument as lowered by C
  virtual Type *vaListParameterType(Module &M) = 0;

  // Initialise an allocated va_list object to point to an already
  // initialised contiguous memory region.
  // Return the value to pass as the va_list argument
  virtual Value *initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                                  AllocaInst *, Value * /*buffer*/) = 0;

  // Simple lowering suffices for va_end, va_copy for current targets
  bool vaEndIsNop() { return true; }
  bool vaCopyIsMemcpy() { return true; }
};

// The majority case - a void* into an alloca
struct VoidPtr final : public VAListInterface {
  bool passedInSSARegister() override { return true; }

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
};

struct VoidPtrAllocaAddrspace final : public VAListInterface {

  bool passedInSSARegister() override { return true; }

  Type *vaListType(LLVMContext &Ctx) override {
    return PointerType::getUnqual(Ctx);
  }

  Type *vaListParameterType(Module &M) override {
    const DataLayout &DL = M.getDataLayout();
    return DL.getAllocaPtrType(M.getContext());
  }

  Value *initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                          AllocaInst * /*va_list*/, Value *buffer) override {
    return buffer;
  }
};

// SystemV as used by X64 Linux and others
struct SystemV final : public VAListInterface {
  bool passedInSSARegister() override { return false; }

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
};

class VariadicABIInfo {

  VariadicABIInfo(uint32_t MinAlign, uint32_t MaxAlign,
                  std::unique_ptr<VAListInterface> VAList)
      : MinAlign(MinAlign), MaxAlign(MaxAlign), VAList(std::move(VAList)) {}

  template <typename T>
  static VariadicABIInfo create(uint32_t MinAlign, uint32_t MaxAlign) {
    return {MinAlign, MaxAlign, std::make_unique<T>()};
  }

public:
  const uint32_t MinAlign;
  const uint32_t MaxAlign;
  std::unique_ptr<VAListInterface> VAList;

  VariadicABIInfo() : VariadicABIInfo(0, 0, nullptr) {}
  explicit operator bool() const { return static_cast<bool>(VAList); }

  VariadicABIInfo(VariadicABIInfo &&Self)
      : MinAlign(Self.MinAlign), MaxAlign(Self.MaxAlign),
        VAList(Self.VAList.release()) {}

  VariadicABIInfo &operator=(VariadicABIInfo &&Other) {
    this->~VariadicABIInfo();
    new (this) VariadicABIInfo(std::move(Other));
    return *this;
  }

  static VariadicABIInfo create(llvm::Triple const &Triple) {
    const bool IsLinuxABI = Triple.isOSLinux() || Triple.isOSCygMing();

    switch (Triple.getArch()) {

    case Triple::r600:
    case Triple::amdgcn: {
      return create<VoidPtrAllocaAddrspace>(1, 0);
    }

    case Triple::nvptx:
    case Triple::nvptx64: {
      return create<VoidPtr>(4, 0);
    }

    case Triple::x86: {
      // These seem to all fall out the same, despite getTypeStackAlign
      // implying otherwise.

      if (Triple.isOSDarwin()) {
        // X86_32ABIInfo::getTypeStackAlignInBytes is misleading for this.
        // The slotSize(4) implies a minimum alignment
        // The AllowHigherAlign = true means there is no maximum alignment.

        return create<VoidPtr>(4, 0);
      }
      if (Triple.getOS() == llvm::Triple::Win32) {
        return create<VoidPtr>(4, 0);
      }

      if (IsLinuxABI) {
        return create<VoidPtr>(4, 0);
      }

      break;
    }

    case Triple::x86_64: {
      if (Triple.isWindowsMSVCEnvironment() || Triple.isOSWindows()) {
        // x64 msvc emit vaarg passes > 8 byte values by pointer
        // however the variadic call instruction created does not, e.g.
        // a <4 x f32> will be passed as itself, not as a pointer or byval.
        // Postponing resolution of that for now.
        // Expected min/max align of 8.
        return {};
      }

      // SystemV X64 documented behaviour:
      // Slots are at least eight byte aligned and at most 16 byte aligned.
      // If the type needs more than sixteen byte alignment, it still only gets
      // that much alignment on the stack.
      // X64 behaviour in clang:
      // Slots are at least eight byte aligned and at most naturally aligned
      // This matches clang, not the ABI docs.

      if (Triple.isOSDarwin()) {
        return create<SystemV>(8, 8);
      }

      if (IsLinuxABI) {
        return create<SystemV>(8, 8);
      }

      break;
    }

    default:
      break;
    }

    return {};
  }
};

class ExpandVariadics : public ModulePass {

  // The pass construction sets the default (optimize when called from middle
  // end, lowering when called from the backend). The command line variable
  // overrides that. This is useful for testing and debugging. It also allows
  // building an applications with variadic functions wholly removed if one
  // has sufficient control over the dependencies, e.g. a statically linked
  // clang that has no variadic function calls remaining in the binary.
  static ExpandVariadicsMode
  withCommandLineOverride(ExpandVariadicsMode LLVMRequested) {
    ExpandVariadicsMode UserRequested = ExpandVariadicsModeOption;
    return (UserRequested == ExpandVariadicsMode::unspecified) ? LLVMRequested
                                                               : UserRequested;
  }

public:
  static char ID;
  const ExpandVariadicsMode Mode;
  VariadicABIInfo ABI;

  ExpandVariadics(ExpandVariadicsMode Mode)
      : ModulePass(ID), Mode(withCommandLineOverride(Mode)) {}
  StringRef getPassName() const override { return "Expand variadic functions"; }

  // Rewrite a variadic call site
  bool expandCall(Module &M, IRBuilder<> &Builder, CallBase *CB, FunctionType *,
                  Function *NF);

  // Given a variadic function, return a function taking a va_list that can be
  // called instead of the original. Mutates F.
  Function *deriveInlinableVariadicFunctionPair(Module &M, IRBuilder<> &Builder,
                                                Function &F);

  bool runOnFunction(Module &M, IRBuilder<> &Builder, Function *F);

  // Entry point
  bool runOnModule(Module &M) override;

  bool rewriteABI() { return Mode == ExpandVariadicsMode::lowering; }

  void memcpyVAListPointers(const DataLayout &DL, IRBuilder<> &Builder,
                            Value *Dst, Value *Src) {
    auto &Ctx = Builder.getContext();
    Type *VaListTy = ABI.VAList->vaListType(Ctx);
    uint64_t Size = DL.getTypeAllocSize(VaListTy).getFixedValue();
    // todo: on amdgcn this should be in terms of addrspace 5
    Builder.CreateMemCpyInline(Dst, {}, Src, {},
                               ConstantInt::get(Type::getInt32Ty(Ctx), Size));
  }

  bool expandVAIntrinsicCall(IRBuilder<> &Builder, const DataLayout &DL,
                             VAStartInst *Inst);

  bool expandVAIntrinsicCall(IRBuilder<> &, const DataLayout &,
                             VAEndInst *Inst);

  bool expandVAIntrinsicCall(IRBuilder<> &Builder, const DataLayout &DL,
                             VACopyInst *Inst);

  template <Intrinsic::ID ID, typename InstructionType>
  bool expandIntrinsicUsers(Module &M, IRBuilder<> &Builder,
                            PointerType *ArgType) {
    bool Changed = false;
    const DataLayout &DL = M.getDataLayout();
    if (Function *Intrinsic = getPreexistingDeclaration(&M, ID, {ArgType})) {
      for (User *U : Intrinsic->users()) {
        if (auto *I = dyn_cast<InstructionType>(U)) {
          Changed |= expandVAIntrinsicCall(Builder, DL, I);
        }
      }
      if (Intrinsic->use_empty())
        Intrinsic->eraseFromParent();
    }
    return Changed;
  }

  FunctionType *inlinableVariadicFunctionType(Module &M, FunctionType *FTy) {
    SmallVector<Type *> ArgTypes(FTy->param_begin(), FTy->param_end());
    ArgTypes.push_back(ABI.VAList->vaListParameterType(M));
    return FunctionType::get(FTy->getReturnType(), ArgTypes,
                             /*IsVarArgs*/ false);
  }

  static ConstantInt *sizeOfAlloca(LLVMContext &Ctx, const DataLayout &DL,
                                   AllocaInst *Alloced) {
    Type *AllocaType = Alloced->getAllocatedType();
    TypeSize AllocaTypeSize = DL.getTypeAllocSize(AllocaType);
    uint64_t AsInt = AllocaTypeSize.getFixedValue();
    return ConstantInt::get(Type::getInt64Ty(Ctx), AsInt);
  }

  static SmallSet<unsigned, 2> supportedAddressSpaces(const DataLayout &DL) {
    // FIXME: It looks like a module can contain arbitrary integers for address
    // spaces in which case we might need to check _lots_ of cases. Maybe add a
    // rule to the verifier that the vastart/vaend intrinsics can have arguments
    // in 0 or in allocaaddrspace but nowhere else
    SmallSet<unsigned, 2> Set;
    Set.insert(0); // things tend to end up in zero
    Set.insert(
        DL.getAllocaAddrSpace()); // the argument should be in alloca addrspace
    return Set;
  }

  // this could be partially target specific
  bool expansionApplicableToFunction(Module &M, Function *F) {
    if (F->isIntrinsic() || !F->isVarArg() ||
        F->hasFnAttribute(Attribute::Naked)) {
      return false;
    }

    // TODO: work out what to do with the cs_chain functions documented as
    // non-variadic that are variadic in some lit tests
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
        if (rewriteABI()) {
          // Todo: Sema?
          report_fatal_error("Cannot lower musttail variadic call");
        } else {
          return false;
        }
      }
    }

    return true;
  }

  class ExpandedCallFrame {
    // Helper for constructing an alloca instance containing the arguments bound
    // to the variadic ... parameter, rearranged to allow indexing through a
    // va_list iterator
    //
    // The awkward memory layout is to allow direct access to a contiguous array
    // of types for the conversion to a struct type
    enum { N = 4 };
    SmallVector<Type *, N> FieldTypes;
    enum Tag { IsByVal, NotByVal, Padding };
    SmallVector<std::pair<Value *, Tag>, N> Fields;

    template <Tag tag> void append(Type *T, Value *V) {
      FieldTypes.push_back(T);
      Fields.push_back({V, tag});
    }

  public:
    void value(Type *T, Value *V) { append<NotByVal>(T, V); }

    void byVal(Type *T, Value *V) { append<IsByVal>(T, V); }

    void padding(LLVMContext &Ctx, uint64_t By) {
      append<Padding>(ArrayType::get(Type::getInt8Ty(Ctx), By), nullptr);
    }

    size_t size() const { return FieldTypes.size(); }
    bool empty() const { return FieldTypes.empty(); }

    StructType *asStruct(LLVMContext &Ctx, StringRef Name) {
      const bool IsPacked = true;
      return StructType::create(Ctx, FieldTypes,
                                (Twine(Name) + ".vararg").str(), IsPacked);
    }

    void initialiseStructAlloca(const DataLayout &DL, IRBuilder<> &Builder,
                                AllocaInst *Alloced) {

      StructType *VarargsTy = cast<StructType>(Alloced->getAllocatedType());

      for (size_t I = 0; I < size(); I++) {
        auto [V, tag] = Fields[I];
        if (!V)
          continue;

        auto R = Builder.CreateStructGEP(VarargsTy, Alloced, I);
        if (tag == IsByVal) {
          Type *ByValType = FieldTypes[I];
          Builder.CreateMemCpy(R, {}, V, {},
                               DL.getTypeAllocSize(ByValType).getFixedValue());
        } else {
          Builder.CreateStore(V, R);
        }
      }
    }
  };
};

bool ExpandVariadics::runOnModule(Module &M) {
  bool Changed = false;
  if (Mode == ExpandVariadicsMode::disable)
    return Changed;

  llvm::Triple Triple(M.getTargetTriple());

  if (Triple.getArch() == Triple::UnknownArch) {
    // If we don't know the triple, we can't lower varargs
    return false;
  }

  ABI = VariadicABIInfo::create(Triple);
  if (!ABI) {
    if (Mode == ExpandVariadicsMode::lowering) {
      report_fatal_error(
          "Requested variadic lowering is unimplemented on this target");
    }
    return Changed;
  }

  const DataLayout &DL = M.getDataLayout();
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
  // in functions thart used to be variadic and have now been mutated to take a
  // va_list instead. If lowering as opposed to optimising, calls to unknown
  // variadic functions have also been replaced.

  // Warning: Intrinsics acting on other ones are missed
  auto CandidateAddressSpaces = supportedAddressSpaces(DL);

  for (unsigned Addrspace : CandidateAddressSpaces) {
    PointerType *ArgType = PointerType::get(Ctx, Addrspace);
    Changed |= expandIntrinsicUsers<Intrinsic::vastart, VAStartInst>(M, Builder,
                                                                     ArgType);
    Changed |=
        expandIntrinsicUsers<Intrinsic::vaend, VAEndInst>(M, Builder, ArgType);
    Changed |= expandIntrinsicUsers<Intrinsic::vacopy, VACopyInst>(M, Builder,
                                                                   ArgType);
  }

  // Variadic intrinsics are now gone. The va_start have been replaced with the
  // equivalent of a va_copy from the newly appended va_list argument, va_end
  // and va_copy are removed. All that remains is for the lowering pass to find
  // indirect calls and rewrite those as well.

  if (Mode == ExpandVariadicsMode::lowering) {
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
  }

  return Changed;
}

bool ExpandVariadics::runOnFunction(Module &M, IRBuilder<> &Builder,
                                    Function *F) {
  bool Changed = false;

  // fprintf(stderr, "Called runOn: %s\n", F->getName().str().c_str());

  // This check might be too coarse - there are probably cases where
  // splitting a function is bad but it's usable without splitting
  if (!expansionApplicableToFunction(M, F))
    return false;

  // TODO: Leave "thunk" attribute functions alone?

  // Need more tests than this. Weak etc. Some are in expansionApplicable.
  if (F->isDeclaration() && !rewriteABI()) {
    return false;
  }

  // TODO: Is the lazy construction here still useful?
  Function *Equivalent = deriveInlinableVariadicFunctionPair(M, Builder, *F);

  for (User *U : llvm::make_early_inc_range(F->users())) {
    // TODO: A test where the call instruction takes a variadic function as
    // a parameter other than the one it is calling
    if (CallBase *CB = dyn_cast<CallBase>(U)) {
      Value *calledOperand = CB->getCalledOperand();
      if (F == calledOperand) {
        Changed |= expandCall(M, Builder, CB, F->getFunctionType(), Equivalent);
      }
    }
  }

  if (rewriteABI()) {
    // No direct calls remain to F, remaining uses are things like address
    // escaping, modulo errors in this implementation.
    for (User *U : llvm::make_early_inc_range(F->users()))
      if (CallBase *CB = dyn_cast<CallBase>(U)) {
        Value *calledOperand = CB->getCalledOperand();
        if (F == calledOperand) {
          report_fatal_error(
              "ExpandVA abi requires eliminating call uses first\n");
        }
      }

    Changed = true;
    // Converting the original variadic function in-place into the equivalent
    // one.
    Equivalent->setLinkage(F->getLinkage());
    Equivalent->setVisibility(F->getVisibility());
    Equivalent->takeName(F);

    // Indirect calls still need to be patched up
    // DAE bitcasts it, todo: check block addresses
    F->replaceAllUsesWith(Equivalent);
    F->eraseFromParent();
  }

  return Changed;
}

Function *ExpandVariadics::deriveInlinableVariadicFunctionPair(
    Module &M, IRBuilder<> &Builder, Function &F) {
  // The purpose here is split the variadic function F into two functions
  // One is a variadic function that bundles the passed argument into a va_list
  // and passes it to the second function. The second function does whatever
  // the original F does, except that it takes a va_list instead of the ...

  assert(expansionApplicableToFunction(M, &F));

  auto &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  // Returned value isDeclaration() is equal to F.isDeclaration()
  // but that invariant is not satisfied throughout this function
  const bool FunctionIsDefinition = !F.isDeclaration();

  FunctionType *FTy = F.getFunctionType();
  SmallVector<Type *> ArgTypes(FTy->param_begin(), FTy->param_end());
  ArgTypes.push_back(ABI.VAList->vaListParameterType(M));

  FunctionType *NFTy = inlinableVariadicFunctionType(M, F.getFunctionType());
  Function *NF = Function::Create(NFTy, F.getLinkage(), F.getAddressSpace());

  // Note - same attribute handling as DeadArgumentElimination
  NF->copyAttributesFrom(&F);
  NF->setComdat(F.getComdat()); // beware weak
  F.getParent()->getFunctionList().insert(F.getIterator(), NF);
  NF->setName(F.getName() + ".valist");
  NF->IsNewDbgInfoFormat = F.IsNewDbgInfoFormat;

  // New function is default visibility and internal
  // Need to set visibility before linkage to avoid an assert in setVisibility
  NF->setVisibility(GlobalValue::DefaultVisibility);
  NF->setLinkage(GlobalValue::InternalLinkage);

  AttrBuilder ParamAttrs(Ctx);
  ParamAttrs.addAttribute(Attribute::NoAlias);

  // TODO: When can the va_list argument have addAlignmentAttr called on it?
  // It improves codegen lot in the non-inlined case. Probably target
  // specific.

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

  if (FunctionIsDefinition) {
    // The blocks have been stolen so it's now a declaration
    assert(F.isDeclaration());
    Type *VaListTy = ABI.VAList->vaListType(Ctx);

    auto *BB = BasicBlock::Create(Ctx, "entry", &F);
    Builder.SetInsertPoint(BB);

    Value *VaListInstance = Builder.CreateAlloca(VaListTy, nullptr, "va_list");

    Builder.CreateIntrinsic(Intrinsic::vastart, {DL.getAllocaPtrType(Ctx)},
                            {VaListInstance});

    SmallVector<Value *> Args;
    for (Argument &A : F.args())
      Args.push_back(&A);

    // Shall we put the extra arg in alloca addrspace? Probably yes
    VaListInstance = Builder.CreatePointerBitCastOrAddrSpaceCast(
        VaListInstance, ABI.VAList->vaListParameterType(M));
    Args.push_back(VaListInstance);

    CallInst *Result = Builder.CreateCall(NF, Args);
    Result->setTailCallKind(CallInst::TCK_Tail);

    assert(ABI.VAList->vaEndIsNop()); // If this changes, insert a va_end here

    if (Result->getType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(Result);
  }

  assert(F.isDeclaration() == NF->isDeclaration());

  return NF;
}

bool ExpandVariadics::expandCall(Module &M, IRBuilder<> &Builder, CallBase *CB,
                                 FunctionType *VarargFunctionType,
                                 Function *NF) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();

  if (!callinstRewritable(CB)) {
    return Changed;
  }

  // This is something of a problem because the call instructions' idea of the
  // function type doesn't necessarily match reality, before or after this
  // pass
  // Since the plan here is to build a new instruction there is no
  // particular benefit to trying to preserve an incorrect initial type
  // If the types don't match and we aren't changing ABI, leave it alone
  // in case someone is deliberately doing dubious type punning through a
  // varargs.
  FunctionType *FuncType = CB->getFunctionType();
  if (FuncType != VarargFunctionType) {
    if (!rewriteABI()) {
      return Changed;
    }
    FuncType = VarargFunctionType;
  }

  auto &Ctx = CB->getContext();

  // Align the struct on ABI.MinAlign to start with
  Align MaxFieldAlign(ABI.MinAlign ? ABI.MinAlign : 1);

  // The strategy here is to allocate a call frame containing the variadic
  // arguments laid out such that a target specific va_list can be initialised
  // with it, such that target specific va_arg instructions will correctly
  // iterate over it. Primarily this means getting the alignment right.

  ExpandedCallFrame Frame;

  uint64_t CurrentOffset = 0;
  for (unsigned I = FuncType->getNumParams(), E = CB->arg_size(); I < E; ++I) {
    Value *ArgVal = CB->getArgOperand(I);
    bool IsByVal = CB->paramHasAttr(I, Attribute::ByVal);
    Type *ArgType = IsByVal ? CB->getParamByValType(I) : ArgVal->getType();
    Align DataAlign = DL.getABITypeAlign(ArgType);

    uint64_t DataAlignV = DataAlign.value();

    // Currently using 0 as a sentinel to mean ignored
    if (ABI.MinAlign && DataAlignV < ABI.MinAlign)
      DataAlignV = ABI.MinAlign;
    if (ABI.MaxAlign && DataAlignV > ABI.MaxAlign)
      DataAlignV = ABI.MaxAlign;

    DataAlign = Align(DataAlignV);
    MaxFieldAlign = std::max(MaxFieldAlign, DataAlign);

    if (uint64_t Rem = CurrentOffset % DataAlignV) {
      // Inject explicit padding to deal with alignment requirements
      uint64_t Padding = DataAlignV - Rem;
      Frame.padding(Ctx, Padding);
      CurrentOffset += Padding;
    }

    if (IsByVal) {
      Frame.byVal(ArgType, ArgVal);
    } else {
      Frame.value(ArgType, ArgVal);
    }
    CurrentOffset += DL.getTypeAllocSize(ArgType).getFixedValue();
  }

  if (Frame.empty()) {
    // Not passing any arguments, hopefully va_arg won't try to read any
    // Creating a single byte frame containing nothing to point the va_list
    // instance as that is less special-casey in the compiler and probably
    // easier to interpret in a debugger.
    Frame.padding(Ctx, 1);
  }

  Function *CBF = CB->getParent()->getParent();

  StructType *VarargsTy = Frame.asStruct(Ctx, CBF->getName());

  // Put the alloca to hold the variadic args in the entry basic block.
  // The clumsy construction is to set a the alignment on the instance
  Builder.SetInsertPointPastAllocas(CBF);

  // The struct instance needs to be at least MaxFieldAlign for the alignment of
  // the fields to be correct at runtime. Use the native stack alignment instead
  // if that's greater as that tends to give better codegen.
  Align AllocaAlign = MaxFieldAlign;
  if (DL.exceedsNaturalStackAlignment(Align(1024))) {
    // TODO: DL.getStackAlignment could return a MaybeAlign instead of assert
    AllocaAlign = std::max(AllocaAlign, DL.getStackAlignment());
  }

  AllocaInst *Alloced = Builder.Insert(
      new AllocaInst(VarargsTy, DL.getAllocaAddrSpace(), nullptr, AllocaAlign),
      "vararg_buffer");
  Changed = true;
  assert(Alloced->getAllocatedType() == VarargsTy);

  // Initialise the fields in the struct
  Builder.SetInsertPoint(CB);

  Builder.CreateLifetimeStart(Alloced, sizeOfAlloca(Ctx, DL, Alloced));

  Frame.initialiseStructAlloca(DL, Builder, Alloced);

  unsigned NumArgs = FuncType->getNumParams();

  SmallVector<Value *> Args;
  Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);

  // Initialise a va_list pointing to that struct and pass it as the last
  // argument
  AllocaInst *VaList = nullptr;
  {
    if (!ABI.VAList->passedInSSARegister()) {
      Type *VaListTy = ABI.VAList->vaListType(Ctx);
      Builder.SetInsertPointPastAllocas(CBF);
      VaList = Builder.CreateAlloca(VaListTy, nullptr, "va_list");
      Builder.SetInsertPoint(CB);
      Builder.CreateLifetimeStart(VaList, sizeOfAlloca(Ctx, DL, VaList));
    }
    Args.push_back(ABI.VAList->initializeVAList(Ctx, Builder, VaList, Alloced));
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
  // TODO, other instructions? Haven't managed to write variadic inline asm yet
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
    // will discard the stores to the Alloca and pass uninitialised memory along
    // instead when the function is marked tailcall
    if (TCK == CallInst::TCK_Tail) {
      TCK = CallInst::TCK_None;
    }
    CI->setTailCallKind(TCK);

  } else if (InvokeInst *II = dyn_cast<InvokeInst>(CB)) {
    assert(NF);
    NewCB = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args, OpBundles, "", CB);
  } else {
    report_fatal_error("Unimplemented variadic lowering for CallInst");
  }

  if (VaList)
    Builder.CreateLifetimeEnd(VaList, sizeOfAlloca(Ctx, DL, VaList));

  Builder.CreateLifetimeEnd(Alloced, sizeOfAlloca(Ctx, DL, Alloced));

  NewCB->setAttributes(PAL);
  NewCB->takeName(CB);
  NewCB->setCallingConv(CB->getCallingConv());

  NewCB->setDebugLoc(DebugLoc());

  // I think this is upsetting the debug handling (DISubprogram attached to more
  // than one function) Need to move metadata, not copy it?
  NewCB->copyMetadata(*CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});

  CB->replaceAllUsesWith(NewCB);
  CB->eraseFromParent();
  return Changed;
}

bool ExpandVariadics::expandVAIntrinsicCall(IRBuilder<> &Builder,
                                            const DataLayout &DL,
                                            VAStartInst *Inst) {
  Function *ContainingFunction = Inst->getFunction();
  if (ContainingFunction->isVarArg())
    return false;

  // The last argument is a vaListParameterType
  Argument *PassedVaList =
      ContainingFunction->getArg(ContainingFunction->arg_size() - 1);

  // va_start takes a pointer to a va_list, e.g. one on the stack
  Value *VaStartArg = Inst->getArgList();

  Builder.SetInsertPoint(Inst);
  if (ABI.VAList->passedInSSARegister()) {
    Builder.CreateStore(PassedVaList, VaStartArg);
  } else {
    // src and dst are both pointers
    memcpyVAListPointers(DL, Builder, VaStartArg, PassedVaList);
  }

  Inst->eraseFromParent();
  return true;
}

bool ExpandVariadics::expandVAIntrinsicCall(IRBuilder<> &, const DataLayout &,
                                            VAEndInst *Inst) {
  // A no-op on all the architectures implemented so far
  Inst->eraseFromParent();
  return true;
}

bool ExpandVariadics::expandVAIntrinsicCall(IRBuilder<> &Builder,
                                            const DataLayout &DL,
                                            VACopyInst *Inst) {
  Builder.SetInsertPoint(Inst);
  memcpyVAListPointers(DL, Builder, Inst->getDest(), Inst->getSrc());
  Inst->eraseFromParent();
  return true;
}

} // namespace

char ExpandVariadics::ID = 0;

INITIALIZE_PASS(ExpandVariadics, DEBUG_TYPE, "Expand variadic functions", false,
                false)

ModulePass *llvm::createExpandVariadicsPass(ExpandVariadicsMode Mode) {
  return new ExpandVariadics(Mode);
}

PreservedAnalyses ExpandVariadicsPass::run(Module &M, ModuleAnalysisManager &) {
  return ExpandVariadics(ConstructedMode).runOnModule(M)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}

ExpandVariadicsPass::ExpandVariadicsPass(OptimizationLevel Level)
    : ExpandVariadicsPass(Level == OptimizationLevel::O0
                              ? ExpandVariadicsMode::disable
                              : ExpandVariadicsMode::optimize) {}

ExpandVariadicsPass::ExpandVariadicsPass(ExpandVariadicsMode Mode)
    : ConstructedMode(Mode) {}
