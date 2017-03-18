//===- ThinLTOBitcodeWriter.cpp - Bitcode writing pass for ThinLTO --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass prepares a module containing type metadata for ThinLTO by splitting
// it into regular and thin LTO parts if possible, and writing both parts to
// a multi-module bitcode file. Modules that do not contain type metadata are
// written unmodified as a single module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/TypeMetadataUtils.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
using namespace llvm;

namespace {

// Promote each local-linkage entity defined by ExportM and used by ImportM by
// changing visibility and appending the given ModuleId.
void promoteInternals(Module &ExportM, Module &ImportM, StringRef ModuleId) {
  auto PromoteInternal = [&](GlobalValue &ExportGV) {
    if (!ExportGV.hasLocalLinkage())
      return;

    GlobalValue *ImportGV = ImportM.getNamedValue(ExportGV.getName());
    if (!ImportGV || ImportGV->use_empty())
      return;

    std::string NewName = (ExportGV.getName() + ModuleId).str();

    ExportGV.setName(NewName);
    ExportGV.setLinkage(GlobalValue::ExternalLinkage);
    ExportGV.setVisibility(GlobalValue::HiddenVisibility);

    ImportGV->setName(NewName);
    ImportGV->setVisibility(GlobalValue::HiddenVisibility);
  };

  for (auto &F : ExportM)
    PromoteInternal(F);
  for (auto &GV : ExportM.globals())
    PromoteInternal(GV);
  for (auto &GA : ExportM.aliases())
    PromoteInternal(GA);
  for (auto &IF : ExportM.ifuncs())
    PromoteInternal(IF);
}

// Promote all internal (i.e. distinct) type ids used by the module by replacing
// them with external type ids formed using the module id.
//
// Note that this needs to be done before we clone the module because each clone
// will receive its own set of distinct metadata nodes.
void promoteTypeIds(Module &M, StringRef ModuleId) {
  DenseMap<Metadata *, Metadata *> LocalToGlobal;
  auto ExternalizeTypeId = [&](CallInst *CI, unsigned ArgNo) {
    Metadata *MD =
        cast<MetadataAsValue>(CI->getArgOperand(ArgNo))->getMetadata();

    if (isa<MDNode>(MD) && cast<MDNode>(MD)->isDistinct()) {
      Metadata *&GlobalMD = LocalToGlobal[MD];
      if (!GlobalMD) {
        std::string NewName =
            (to_string(LocalToGlobal.size()) + ModuleId).str();
        GlobalMD = MDString::get(M.getContext(), NewName);
      }

      CI->setArgOperand(ArgNo,
                        MetadataAsValue::get(M.getContext(), GlobalMD));
    }
  };

  if (Function *TypeTestFunc =
          M.getFunction(Intrinsic::getName(Intrinsic::type_test))) {
    for (const Use &U : TypeTestFunc->uses()) {
      auto CI = cast<CallInst>(U.getUser());
      ExternalizeTypeId(CI, 1);
    }
  }

  if (Function *TypeCheckedLoadFunc =
          M.getFunction(Intrinsic::getName(Intrinsic::type_checked_load))) {
    for (const Use &U : TypeCheckedLoadFunc->uses()) {
      auto CI = cast<CallInst>(U.getUser());
      ExternalizeTypeId(CI, 2);
    }
  }

  for (GlobalObject &GO : M.global_objects()) {
    SmallVector<MDNode *, 1> MDs;
    GO.getMetadata(LLVMContext::MD_type, MDs);

    GO.eraseMetadata(LLVMContext::MD_type);
    for (auto MD : MDs) {
      auto I = LocalToGlobal.find(MD->getOperand(1));
      if (I == LocalToGlobal.end()) {
        GO.addMetadata(LLVMContext::MD_type, *MD);
        continue;
      }
      GO.addMetadata(
          LLVMContext::MD_type,
          *MDNode::get(M.getContext(),
                       ArrayRef<Metadata *>{MD->getOperand(0), I->second}));
    }
  }
}

// Drop unused globals, and drop type information from function declarations.
// FIXME: If we made functions typeless then there would be no need to do this.
void simplifyExternals(Module &M) {
  FunctionType *EmptyFT =
      FunctionType::get(Type::getVoidTy(M.getContext()), false);

  for (auto I = M.begin(), E = M.end(); I != E;) {
    Function &F = *I++;
    if (F.isDeclaration() && F.use_empty()) {
      F.eraseFromParent();
      continue;
    }

    if (!F.isDeclaration() || F.getFunctionType() == EmptyFT)
      continue;

    Function *NewF =
        Function::Create(EmptyFT, GlobalValue::ExternalLinkage, "", &M);
    NewF->setVisibility(F.getVisibility());
    NewF->takeName(&F);
    F.replaceAllUsesWith(ConstantExpr::getBitCast(NewF, F.getType()));
    F.eraseFromParent();
  }

  for (auto I = M.global_begin(), E = M.global_end(); I != E;) {
    GlobalVariable &GV = *I++;
    if (GV.isDeclaration() && GV.use_empty()) {
      GV.eraseFromParent();
      continue;
    }
  }
}

void filterModule(
    Module *M, function_ref<bool(const GlobalValue *)> ShouldKeepDefinition) {
  for (Function &F : *M) {
    if (ShouldKeepDefinition(&F))
      continue;

    F.deleteBody();
    F.setComdat(nullptr);
    F.clearMetadata();
  }

  for (GlobalVariable &GV : M->globals()) {
    if (ShouldKeepDefinition(&GV))
      continue;

    GV.setInitializer(nullptr);
    GV.setLinkage(GlobalValue::ExternalLinkage);
    GV.setComdat(nullptr);
    GV.clearMetadata();
  }

  for (Module::alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E;) {
    GlobalAlias *GA = &*I++;
    if (ShouldKeepDefinition(GA))
      continue;

    GlobalObject *GO;
    if (I->getValueType()->isFunctionTy())
      GO = Function::Create(cast<FunctionType>(GA->getValueType()),
                            GlobalValue::ExternalLinkage, "", M);
    else
      GO = new GlobalVariable(
          *M, GA->getValueType(), false, GlobalValue::ExternalLinkage,
          (Constant *)nullptr, "", (GlobalVariable *)nullptr,
          GA->getThreadLocalMode(), GA->getType()->getAddressSpace());
    GO->takeName(GA);
    GA->replaceAllUsesWith(GO);
    GA->eraseFromParent();
  }
}

void forEachVirtualFunction(Constant *C, function_ref<void(Function *)> Fn) {
  if (auto *F = dyn_cast<Function>(C))
    return Fn(F);
  if (isa<GlobalValue>(C))
    return;
  for (Value *Op : C->operands())
    forEachVirtualFunction(cast<Constant>(Op), Fn);
}

// If it's possible to split M into regular and thin LTO parts, do so and write
// a multi-module bitcode file with the two parts to OS. Otherwise, write only a
// regular LTO bitcode file to OS.
void splitAndWriteThinLTOBitcode(
    raw_ostream &OS, function_ref<AAResults &(Function &)> AARGetter,
    Module &M) {
  std::string ModuleId = getUniqueModuleId(&M);
  if (ModuleId.empty()) {
    // We couldn't generate a module ID for this module, just write it out as a
    // regular LTO module.
    WriteBitcodeToFile(&M, OS);
    return;
  }

  promoteTypeIds(M, ModuleId);

  // Returns whether a global has attached type metadata. Such globals may
  // participate in CFI or whole-program devirtualization, so they need to
  // appear in the merged module instead of the thin LTO module.
  auto HasTypeMetadata = [&](const GlobalObject *GO) {
    SmallVector<MDNode *, 1> MDs;
    GO->getMetadata(LLVMContext::MD_type, MDs);
    return !MDs.empty();
  };

  // Collect the set of virtual functions that are eligible for virtual constant
  // propagation. Each eligible function must not access memory, must return
  // an integer of width <=64 bits, must take at least one argument, must not
  // use its first argument (assumed to be "this") and all arguments other than
  // the first one must be of <=64 bit integer type.
  //
  // Note that we test whether this copy of the function is readnone, rather
  // than testing function attributes, which must hold for any copy of the
  // function, even a less optimized version substituted at link time. This is
  // sound because the virtual constant propagation optimizations effectively
  // inline all implementations of the virtual function into each call site,
  // rather than using function attributes to perform local optimization.
  std::set<const Function *> EligibleVirtualFns;
  for (GlobalVariable &GV : M.globals())
    if (HasTypeMetadata(&GV))
      forEachVirtualFunction(GV.getInitializer(), [&](Function *F) {
        auto *RT = dyn_cast<IntegerType>(F->getReturnType());
        if (!RT || RT->getBitWidth() > 64 || F->arg_empty() ||
            !F->arg_begin()->use_empty())
          return;
        for (auto &Arg : make_range(std::next(F->arg_begin()), F->arg_end())) {
          auto *ArgT = dyn_cast<IntegerType>(Arg.getType());
          if (!ArgT || ArgT->getBitWidth() > 64)
            return;
        }
        if (computeFunctionBodyMemoryAccess(*F, AARGetter(*F)) == MAK_ReadNone)
          EligibleVirtualFns.insert(F);
      });

  ValueToValueMapTy VMap;
  std::unique_ptr<Module> MergedM(
      CloneModule(&M, VMap, [&](const GlobalValue *GV) -> bool {
        if (auto *F = dyn_cast<Function>(GV))
          return EligibleVirtualFns.count(F);
        if (auto *GVar = dyn_cast_or_null<GlobalVariable>(GV->getBaseObject()))
          return HasTypeMetadata(GVar);
        return false;
      }));
  StripDebugInfo(*MergedM);

  for (Function &F : *MergedM)
    if (!F.isDeclaration()) {
      // Reset the linkage of all functions eligible for virtual constant
      // propagation. The canonical definitions live in the thin LTO module so
      // that they can be imported.
      F.setLinkage(GlobalValue::AvailableExternallyLinkage);
      F.setComdat(nullptr);
    }

  // Remove all globals with type metadata, as well as aliases pointing to them,
  // from the thin LTO module.
  filterModule(&M, [&](const GlobalValue *GV) {
    if (auto *GVar = dyn_cast_or_null<GlobalVariable>(GV->getBaseObject()))
      return !HasTypeMetadata(GVar);
    return true;
  });

  promoteInternals(*MergedM, M, ModuleId);
  promoteInternals(M, *MergedM, ModuleId);

  simplifyExternals(*MergedM);

  SmallVector<char, 0> Buffer;
  BitcodeWriter W(Buffer);

  // FIXME: Try to re-use BSI and PFI from the original module here.
  ModuleSummaryIndex Index = buildModuleSummaryIndex(M, nullptr, nullptr);
  W.writeModule(&M, /*ShouldPreserveUseListOrder=*/false, &Index,
                /*GenerateHash=*/true);

  W.writeModule(MergedM.get());

  OS << Buffer;
}

// Returns whether this module needs to be split because it uses type metadata.
bool requiresSplit(Module &M) {
  SmallVector<MDNode *, 1> MDs;
  for (auto &GO : M.global_objects()) {
    GO.getMetadata(LLVMContext::MD_type, MDs);
    if (!MDs.empty())
      return true;
  }

  return false;
}

void writeThinLTOBitcode(raw_ostream &OS,
                         function_ref<AAResults &(Function &)> AARGetter,
                         Module &M, const ModuleSummaryIndex *Index) {
  // See if this module has any type metadata. If so, we need to split it.
  if (requiresSplit(M))
    return splitAndWriteThinLTOBitcode(OS, AARGetter, M);

  // Otherwise we can just write it out as a regular module.
  WriteBitcodeToFile(&M, OS, /*ShouldPreserveUseListOrder=*/false, Index,
                     /*GenerateHash=*/true);
}

class WriteThinLTOBitcode : public ModulePass {
  raw_ostream &OS; // raw_ostream to print on

public:
  static char ID; // Pass identification, replacement for typeid
  WriteThinLTOBitcode() : ModulePass(ID), OS(dbgs()) {
    initializeWriteThinLTOBitcodePass(*PassRegistry::getPassRegistry());
  }

  explicit WriteThinLTOBitcode(raw_ostream &o)
      : ModulePass(ID), OS(o) {
    initializeWriteThinLTOBitcodePass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "ThinLTO Bitcode Writer"; }

  bool runOnModule(Module &M) override {
    const ModuleSummaryIndex *Index =
        &(getAnalysis<ModuleSummaryIndexWrapperPass>().getIndex());
    writeThinLTOBitcode(OS, LegacyAARGetter(*this), M, Index);
    return true;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<ModuleSummaryIndexWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};
} // anonymous namespace

char WriteThinLTOBitcode::ID = 0;
INITIALIZE_PASS_BEGIN(WriteThinLTOBitcode, "write-thinlto-bitcode",
                      "Write ThinLTO Bitcode", false, true)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(ModuleSummaryIndexWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(WriteThinLTOBitcode, "write-thinlto-bitcode",
                    "Write ThinLTO Bitcode", false, true)

ModulePass *llvm::createWriteThinLTOBitcodePass(raw_ostream &Str) {
  return new WriteThinLTOBitcode(Str);
}
