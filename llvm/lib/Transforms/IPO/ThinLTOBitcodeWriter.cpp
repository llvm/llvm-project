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
#include "llvm/Transforms/Utils/Cloning.h"
using namespace llvm;

namespace {

// Produce a unique identifier for this module by taking the MD5 sum of the
// names of the module's strong external symbols. This identifier is
// normally guaranteed to be unique, or the program would fail to link due to
// multiply defined symbols.
//
// If the module has no strong external symbols (such a module may still have a
// semantic effect if it performs global initialization), we cannot produce a
// unique identifier for this module, so we return the empty string, which
// causes the entire module to be written as a regular LTO module.
std::string getModuleId(Module *M) {
  MD5 Md5;
  bool ExportsSymbols = false;
  auto AddGlobal = [&](GlobalValue &GV) {
    if (GV.isDeclaration() || GV.getName().startswith("llvm.") ||
        !GV.hasExternalLinkage())
      return;
    ExportsSymbols = true;
    Md5.update(GV.getName());
    Md5.update(ArrayRef<uint8_t>{0});
  };

  for (auto &F : *M)
    AddGlobal(F);
  for (auto &GV : M->globals())
    AddGlobal(GV);
  for (auto &GA : M->aliases())
    AddGlobal(GA);
  for (auto &IF : M->ifuncs())
    AddGlobal(IF);

  if (!ExportsSymbols)
    return "";

  MD5::MD5Result R;
  Md5.final(R);

  SmallString<32> Str;
  MD5::stringifyResult(R, Str);
  return ("$" + Str).str();
}

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

// If it's possible to split M into regular and thin LTO parts, do so and write
// a multi-module bitcode file with the two parts to OS. Otherwise, write only a
// regular LTO bitcode file to OS.
void splitAndWriteThinLTOBitcode(raw_ostream &OS, Module &M) {
  std::string ModuleId = getModuleId(&M);
  if (ModuleId.empty()) {
    // We couldn't generate a module ID for this module, just write it out as a
    // regular LTO module.
    WriteBitcodeToFile(&M, OS);
    return;
  }

  promoteTypeIds(M, ModuleId);

  auto IsInMergedM = [&](const GlobalValue *GV) {
    auto *GVar = dyn_cast<GlobalVariable>(GV->getBaseObject());
    if (!GVar)
      return false;

    SmallVector<MDNode *, 1> MDs;
    GVar->getMetadata(LLVMContext::MD_type, MDs);
    return !MDs.empty();
  };

  ValueToValueMapTy VMap;
  std::unique_ptr<Module> MergedM(CloneModule(&M, VMap, IsInMergedM));
  StripDebugInfo(*MergedM);

  filterModule(&M, [&](const GlobalValue *GV) { return !IsInMergedM(GV); });

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

void writeThinLTOBitcode(raw_ostream &OS, Module &M,
                         const ModuleSummaryIndex *Index) {
  // See if this module has any type metadata. If so, we need to split it.
  if (requiresSplit(M))
    return splitAndWriteThinLTOBitcode(OS, M);

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
    writeThinLTOBitcode(OS, M, Index);
    return true;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ModuleSummaryIndexWrapperPass>();
  }
};
} // anonymous namespace

char WriteThinLTOBitcode::ID = 0;
INITIALIZE_PASS_BEGIN(WriteThinLTOBitcode, "write-thinlto-bitcode",
                      "Write ThinLTO Bitcode", false, true)
INITIALIZE_PASS_DEPENDENCY(ModuleSummaryIndexWrapperPass)
INITIALIZE_PASS_END(WriteThinLTOBitcode, "write-thinlto-bitcode",
                    "Write ThinLTO Bitcode", false, true)

ModulePass *llvm::createWriteThinLTOBitcodePass(raw_ostream &Str) {
  return new WriteThinLTOBitcode(Str);
}
