//===--- ModuleSplitter.cpp - Module Splitter -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/ModuleSplitter/ModuleSplitter.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
#define DEBUG_TYPE "llvm-module-split"

//===----------------------------------------------------------------------===//
// LLVMModuleAndContext
//===----------------------------------------------------------------------===//

Expected<bool> LLVMModuleAndContext::create(
    function_ref<Expected<std::unique_ptr<llvm::Module>>(llvm::LLVMContext &)>
        CreateModule) {
  assert(!Module && "already have a module");
  auto ModuleOr = CreateModule(*Ctx);
  if (!ModuleOr)
    return ModuleOr.takeError();

  Module = std::move(*ModuleOr);
  return true;
}

void LLVMModuleAndContext::reset() {
  Module.reset();
  Ctx.reset();
}

//===----------------------------------------------------------------------===//
// StringConstantTable
//===----------------------------------------------------------------------===//

namespace {
/// Large strings are very inefficiently encoded in LLVM bitcode (each `char` is
/// encoded as a `uint64_t`). The LLVM bitcode reader is also very inefficiently
/// reads strings back, performing 3 ultimate copies of the data. This is made
/// worse by the fact the `getLazyBitcodeModule` does not lazily parse constants
/// from the LLVM bitcode. Thus, when per-function splitting a module with N
/// functions and M large string constants, we form 3*M*N copies of the large
/// strings.
///
/// This class is part of a workaround of this inefficiency. When processing a
/// module for splitting, we track any string global constants and their indices
/// in this table. If a module is going to be roundtripped through bitcode to be
/// lazily loaded, we externalize the strings by setting the corresponding
/// constants to `zeroinitializer` in the module before it is written to
/// bitcode. As we materialize constants on the other side, we check for a
/// materialized global variable that matches an entry in the string table and
/// directly copy the data over into the new LLVM context.
///
/// We can generalize this optimization to other large data types as necessary.
///
/// This class is used in an `RCRef` to be shared across multiple threads.
class StringConstantTable
    : public ThreadSafeRefCountedBase<StringConstantTable> {
  /// An entry in the string table consists of a global variable, its module
  /// index, and the a reference to the string data. Because the string data is
  /// owned by the original LLVM context, we have to ensure it stays alive.
  struct Entry {
    unsigned Idx;
    const llvm::GlobalVariable *Var;
    StringRef Value;
  };

public:
  /// If `Value` denotes a string constant, record the data at index `GvIdx`.
  void recordIfStringConstant(unsigned GvIdx, const llvm::GlobalValue &Value) {
    auto Var = dyn_cast<llvm::GlobalVariable>(&Value);
    if (Var && Var->isConstant() && Var->hasInternalLinkage()) {
      auto *Init =
          dyn_cast<llvm::ConstantDataSequential>(Var->getInitializer());
      if (Init && Init->isCString())
        StringConstants.push_back(Entry{GvIdx, Var, Init->getAsString()});
    }
  }

  /// Before writing the main Module to bitcode, externalize large string
  /// constants by stubbing out their values. Take ownership of the main Module
  /// so the string data stays alive.
  llvm::Module &externalizeStrings(LLVMModuleAndContext &&Module) {
    MainModule = std::move(Module);
    // Stub the initializers. The global variable is an internal constant, so it
    // must have an initializer.
    for (Entry &E : StringConstants) {
      auto *Stub =
          llvm::Constant::getNullValue(E.Var->getInitializer()->getType());
      // `const_cast` is OK because we own the module now.
      const_cast<llvm::GlobalVariable *>(E.Var)->setInitializer(Stub);
    }
    return *MainModule;
  }

  /// This is an iterator over the entries in the string table.
  class Injector {
    using const_iterator = std::vector<Entry>::const_iterator;

  public:
    /// Given a global variable in a materialized module and its index, if it is
    /// a string constant found in the table, copy the data over into the new
    /// LLVM context and set the initializer.
    void materializeIfStringConstant(unsigned GvIdx,
                                     llvm::GlobalVariable &Var) {
      while (It != Et && It->Idx < GvIdx)
        ++It;
      if (It == Et || It->Idx != GvIdx)
        return;
      Var.setInitializer(llvm::ConstantDataArray::getString(
          Var.getType()->getContext(), It->Value, /*AddNull=*/false));
    }

  private:
    explicit Injector(const_iterator It, const_iterator Et) : It(It), Et(Et) {}

    const_iterator It, Et;

    friend class StringConstantTable;
  };

  Injector begin() const {
    return Injector(StringConstants.begin(), StringConstants.end());
  }

private:
  std::vector<Entry> StringConstants;
  LLVMModuleAndContext MainModule;
};

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

class LLVMModuleSplitterImpl {
public:
  explicit LLVMModuleSplitterImpl(LLVMModuleAndContext Module)
      : MainModule(std::move(Module)) {}

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void split(LLVMSplitProcessFn ProcessFn,
             llvm::SmallVectorImpl<llvm::Function> &Anchors);

private:
  struct ValueInfo {
    /// The immediate global value dependencies of a value.
    SmallVector<const llvm::GlobalValue *> Dependencies;
    /// Map each global value to its index in the module. We will use this to
    /// materialize global values from bitcode.
    unsigned GvIdx;
  };

  struct TransitiveDeps {
    /// The transitive dependencies.
    llvm::MapVector<const llvm::GlobalValue *, unsigned> Deps;
    /// True if computation is complete.
    bool Complete = false;
    /// The assigned module index.
    std::optional<unsigned> MutIdx;
  };

  /// Collect the immediate global value dependencies of `Value`. `Orig` is the
  /// original transitive value, which is not equal to `Value` when it is used
  /// in a constant.
  void collectImmediateDependencies(const llvm::Value *Value,
                                    const llvm::GlobalValue *Orig);

  /// The main LLVM module being split.
  LLVMModuleAndContext MainModule;

  /// The value info for each global value in the module.
  llvm::DenseMap<const llvm::Value *, ValueInfo> Infos;

  /// The transitive dependencies of each global value.
  llvm::MapVector<const llvm::GlobalValue *, TransitiveDeps> TransDeps;

  /// Users of split "anchors". These are global values where we don't want
  /// their users to be split into different modules because it will cause the
  /// symbol to be duplicated.
  llvm::MapVector<const llvm::GlobalValue *, llvm::SetVector<TransitiveDeps *>>
      SplitAnchorUsers;
};
} // namespace

static LLVMModuleAndContext readAndMaterializeDependencies(
    MemoryBufferRef &Buf,
    const llvm::MapVector<const llvm::GlobalValue *, unsigned> &Set,
    const StringConstantTable &Strtab) {

  // First, create a lazy module with an internal bitcode materializer.
  // TODO: Not sure how to make lazy loading metadata work.
  LLVMModuleAndContext Result;
  {
    auto CreateOr = Result.create(
        [&](llvm::LLVMContext &Ctx) -> Expected<std::unique_ptr<Module>> {
          return llvm::cantFail(
              llvm::getLazyBitcodeModule(Buf, Ctx,
                                         /*ShouldLazyLoadMetadata=*/false));
        });
    if (!CreateOr)
      return LLVMModuleAndContext();

    Result->setModuleInlineAsm("");
  }

  SmallVector<unsigned> SortIndices =
      llvm::to_vector(llvm::make_second_range(Set));
  llvm::sort(SortIndices, std::less<unsigned>());
  auto *IdxIt = SortIndices.begin();
  auto *IdxEnd = SortIndices.end();

  // The global value indices go from globals, functions, then aliases. This
  // mirrors the order in which global values are deleted by LLVM's GlobalDCE.
  unsigned CurIdx = 0;
  StringConstantTable::Injector It = Strtab.begin();
  // We need to keep the IR "valid" for the verifier because `materializeAll`
  // may invoke it. It doesn't matter since we're deleting the globals anyway.
  for (llvm::GlobalVariable &Global : Result->globals()) {
    if (IdxIt != IdxEnd && CurIdx == *IdxIt) {
      ++IdxIt;
      llvm::cantFail(Global.materialize());
      It.materializeIfStringConstant(CurIdx, Global);
    } else {
      Global.setInitializer(nullptr);
      Global.setComdat(nullptr);
      Global.setLinkage(llvm::GlobalValue::ExternalLinkage);
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generates
      // `R_X86_64_PC32` instead of `R_X86_64_REX_GOTPCRELX`
      // for these symbols and building shared library from
      // a static archive of this module will error with an `fPIC` confusion.
      Global.setDSOLocal(false);
    }
    ++CurIdx;
  }
  for (llvm::Function &Func : Result->functions()) {
    if (IdxIt != IdxEnd && CurIdx == *IdxIt) {
      ++IdxIt;
      llvm::cantFail(Func.materialize());
    } else {
      Func.deleteBody();
      Func.setComdat(nullptr);
      Func.setLinkage(llvm::GlobalValue::ExternalLinkage);
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generates
      // `R_X86_64_PC32` instead of `R_X86_64_REX_GOTPCRELX`
      // for these symbols and building shared library from
      // a static archive of this module will error with an `fPIC` confusion.
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generation get confused.
      Func.setDSOLocal(false);
    }
    ++CurIdx;
  }

  // Finalize materialization of the module.
  llvm::cantFail(Result->materializeAll());

  // Now that the module is materialized, we can start deleting stuff. Just
  // delete declarations with no uses.
  for (llvm::GlobalVariable &Global :
       llvm::make_early_inc_range(Result->globals())) {
    if (Global.isDeclaration() && Global.use_empty())
      Global.eraseFromParent();
  }
  for (llvm::Function &Func : llvm::make_early_inc_range(Result->functions())) {
    if (Func.isDeclaration() && Func.use_empty())
      Func.eraseFromParent();
  }
  return Result;
}

namespace llvm {
/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
void splitPerAnchored(LLVMModuleAndContext Module, LLVMSplitProcessFn ProcessFn,
                      llvm::SmallVectorImpl<llvm::Function> &Anchors) {
  LLVMModuleSplitterImpl Impl(std::move(Module));
  Impl.split(ProcessFn, Anchors);
}
} // namespace llvm

void LLVMModuleSplitterImpl::split(
    LLVMSplitProcessFn ProcessFn,
    llvm::SmallVectorImpl<llvm::Function> &Anchors) {
  // The use-def list is sparse. Use it to build a sparse dependency graph
  // between global values.
  IntrusiveRefCntPtr<StringConstantTable> Strtab(new StringConstantTable());
  unsigned GvIdx = 0;

  auto ComputeDeps = [&](const llvm::GlobalValue &value) {
    Strtab->recordIfStringConstant(GvIdx, value);
    Infos[&value].GvIdx = GvIdx++;
    collectImmediateDependencies(&value, &value);
  };
  // NOTE: The visitation of globals then functions has to line up with
  // `readAndMaterializeDependencies`.
  for (const llvm::GlobalVariable &global : MainModule->globals()) {
    ComputeDeps(global);
    if (!global.hasInternalLinkage() && !global.hasPrivateLinkage())
      TransDeps[&global];
  }
  for (const llvm::Function &Fn : MainModule->functions()) {
    ComputeDeps(Fn);
    if (!Fn.isDeclaration() && (Fn.hasExternalLinkage() || Fn.hasWeakLinkage()))
      TransDeps[&Fn];
  }

  // If there is only one (or fewer) exported functions, forward the main
  // module.
  if (TransDeps.size() <= 1)
    return ProcessFn(forwardModule(std::move(MainModule)), std::nullopt,
                     /*numFunctionBase=*/0);

  // Now for each export'd global value, compute the transitive set of
  // dependencies using DFS.
  SmallVector<const llvm::GlobalValue *> Worklist;
  for (auto &[Value, Deps] : TransDeps) {
    Worklist.clear();
    Worklist.push_back(Value);
    while (!Worklist.empty()) {
      const llvm::GlobalValue *It = Worklist.pop_back_val();

      auto [iter, inserted] = Deps.Deps.insert({It, -1});
      if (!inserted) {
        // Already visited.
        continue;
      }
      // Pay the cost of the name lookup only on a miss.
      const ValueInfo &Info = Infos.at(It);
      iter->second = Info.GvIdx;

      // If this value depends on another value that is going to be split, we
      // don't want to duplicate the symbol. Keep all the users together.
      if (It != Value) {
        if (auto *DepIt = TransDeps.find(It); DepIt != TransDeps.end()) {
          auto &Users = SplitAnchorUsers[It];
          Users.insert(&Deps);
          // Make sure to include the other value in its own user list.
          Users.insert(&DepIt->second);
          // We don't have to recurse since the subgraph will get processed.
          continue;
        }
      }

      // If this value depends on a mutable global, keep track of it. We have to
      // put all users of a mutable global in the same module.
      if (auto *Global = dyn_cast<llvm::GlobalVariable>(It);
          Global && !Global->isConstant())
        SplitAnchorUsers[Global].insert(&Deps);

      // Recursive on dependencies.
      llvm::append_range(Worklist, Info.Dependencies);
    }

    Deps.Complete = true;
  }

  // For each mutable global, grab all the transitive users and put them in one
  // module. If global A has user set A* and global B has user set B* where
  // A* and B* have an empty intersection, all values in A* will be assigned 0
  // and all values in B* will be assigned 1. If global C has user set C* that
  // overlaps both A* and B*, it will overwrite both to 2.
  SmallVector<SmallVector<TransitiveDeps *>> Bucketing(SplitAnchorUsers.size());
  for (auto [CurMutIdx, Bucket, Users] :
       llvm::enumerate(Bucketing, llvm::make_second_range(SplitAnchorUsers))) {
    for (TransitiveDeps *Deps : Users) {
      if (Deps->MutIdx && *Deps->MutIdx != CurMutIdx) {
        auto &OtherBucket = Bucketing[*Deps->MutIdx];
        for (TransitiveDeps *Other : OtherBucket) {
          Bucket.push_back(Other);
          Other->MutIdx = CurMutIdx;
        }
        OtherBucket.clear();
        assert(*Deps->MutIdx == CurMutIdx);
      } else {
        Bucket.push_back(Deps);
        Deps->MutIdx = CurMutIdx;
      }
    }
  }

  // Now that we have assigned buckets to each value, merge the transitive
  // dependency sets of all values belonging to the same set.
  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned>> Buckets(
      Bucketing.size());
  for (auto [Deps, Bucket] : llvm::zip(Bucketing, Buckets)) {
    for (TransitiveDeps *Dep : Deps) {
      for (auto &NamedValue : Dep->Deps)
        Bucket.insert(NamedValue);
    }
  }

  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned> *>
      SetsToProcess;
  SetsToProcess.reserve(Buckets.size() + TransDeps.size());

  // Clone each mutable global bucket into its own module.
  for (auto &Bucket : Buckets) {
    if (Bucket.empty())
      continue;
    SetsToProcess.push_back(&Bucket);
  }

  for (auto &[Root, Deps] : TransDeps) {
    // Skip values included in another transitive dependency set and values
    // included in mutable global sets.
    if (!Deps.MutIdx)
      SetsToProcess.push_back(&Deps.Deps);
  }

  if (SetsToProcess.size() <= 1)
    return ProcessFn(forwardModule(std::move(MainModule)), std::nullopt,
                     /*numFunctionBase=*/0);

  // Sort the sets by to schedule the larger modules first.
  llvm::sort(SetsToProcess,
             [](auto *Lhs, auto *Rhs) { return Lhs->size() > Rhs->size(); });

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  std::string BufStr;
  llvm::raw_string_ostream BufOS(BufStr);
  {
    llvm::Module &Module = Strtab->externalizeStrings(std::move(MainModule));
    llvm::WriteBitcodeToFile(Module, BufOS);
  }

  auto Buf = WritableMemoryBuffer::getNewUninitMemBuffer(BufStr.size());
  memcpy(Buf->getBufferStart(), BufStr.c_str(), BufStr.size());

  unsigned NumFunctions = 0;
  for (auto [Idx, Set] : llvm::enumerate(SetsToProcess)) {
    unsigned Next = NumFunctions + Set->size();
    auto MakeModule =
        [Set = std::move(*Set),
         Buf = MemoryBufferRef((*Buf).MemoryBuffer::getBuffer(), ""),
         Strtab = Strtab]() mutable {
          return readAndMaterializeDependencies(Buf, Set, *Strtab);
        };
    ProcessFn(std::move(MakeModule), Idx, NumFunctions);
    NumFunctions = Next;
  }
}

void LLVMModuleSplitterImpl::collectImmediateDependencies(
    const llvm::Value *Value, const llvm::GlobalValue *Orig) {
  for (const llvm::Value *User : Value->users()) {
    // Recurse into pure constant users.
    if (isa<llvm::Constant>(User) && !isa<llvm::GlobalValue>(User)) {
      collectImmediateDependencies(User, Orig);
      continue;
    }

    if (auto *Inst = dyn_cast<llvm::Instruction>(User)) {
      const llvm::Function *Func = Inst->getParent()->getParent();
      Infos[Func].Dependencies.push_back(Orig);
    } else if (auto *GlobalVal = dyn_cast<llvm::GlobalValue>(User)) {
      Infos[GlobalVal].Dependencies.push_back(Orig);
    } else {
      llvm_unreachable("unexpected user of global value");
    }
  }
}

namespace {
/// This class provides support for splitting an LLVM module into multiple
/// parts.
/// TODO: Clean up the splitters here (some code duplication) when we can move
/// to per function llvm compilation.
class LLVMModulePerFunctionSplitterImpl {
public:
  LLVMModulePerFunctionSplitterImpl(LLVMModuleAndContext Module)
      : MainModule(std::move(Module)) {}

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void
  split(LLVMSplitProcessFn ProcessFn,
        llvm::StringMap<llvm::GlobalValue::LinkageTypes> &SymbolLinkageTypes,
        unsigned NumFunctionBase);

private:
  struct ValueInfo {
    const llvm::Value *Value = nullptr;
    bool CanBeSplit = true;
    llvm::SmallPtrSet<const llvm::GlobalValue *, 4> Dependencies;
    llvm::SmallPtrSet<const llvm::GlobalValue *, 4> Users;
    /// Map each global value to its index in the module. We will use this to
    /// materialize global values from bitcode.
    unsigned GvIdx;
    bool UserEmpty = true;
  };

  /// Collect all of the immediate global value users of `value`.
  void collectValueUsers(const llvm::GlobalValue *Value);

  /// Propagate use information through the module.
  void propagateUseInfo();

  /// The main LLVM module being split.
  LLVMModuleAndContext MainModule;

  /// The value info for each global value in the module.
  llvm::MapVector<const llvm::GlobalValue *, ValueInfo> ValueInfos;
};
} // namespace

namespace llvm {
/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void splitPerFunction(
    LLVMModuleAndContext Module, LLVMSplitProcessFn ProcessFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &SymbolLinkageTypes,
    unsigned NumFunctionBase) {
  LLVMModulePerFunctionSplitterImpl Impl(std::move(Module));
  Impl.split(ProcessFn, SymbolLinkageTypes, NumFunctionBase);
}
} // namespace llvm

/// Split the LLVM module into multiple modules using the provided process
/// function.
void LLVMModulePerFunctionSplitterImpl::split(
    LLVMSplitProcessFn ProcessFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &SymbolLinkageTypes,
    unsigned NumFunctionBase) {
  // Compute the value info for each global in the module.
  // NOTE: The visitation of globals then functions has to line up with
  // `readAndMaterializeDependencies`.
  IntrusiveRefCntPtr<StringConstantTable> Strtab(new StringConstantTable());
  unsigned GvIdx = 0;
  auto ComputeUsers = [&](const llvm::GlobalValue &Value) {
    Strtab->recordIfStringConstant(GvIdx, Value);
    ValueInfos[&Value].GvIdx = GvIdx++;
    collectValueUsers(&Value);
  };
  llvm::for_each(MainModule->globals(), ComputeUsers);
  llvm::for_each(MainModule->functions(), ComputeUsers);

  // With use information collected, propagate it to the dependencies.
  propagateUseInfo();

  // Now we can split the module.
  // We split the module per function and cloning any necessary dependencies:
  // - For function dependencies, only clone the declaration unless its
  //   coroutine related.
  // - For other internal values, clone as is.
  // This is much fine-grained splitting, which enables significantly higher
  // levels of parallelism (and smaller generated artifacts).
  // LLVM LTO style optimization may suffer a bit here since we don't have
  // the full callstack present anymore in each cloned module.
  llvm::DenseSet<const llvm::Value *> SplitValues;
  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned>>
      SetsToProcess;

  // Hoist these collections to re-use memory allocations.
  llvm::ValueToValueMapTy ValueMap;
  SmallPtrSet<const llvm::Value *, 4> SplitDeps;
  auto SplitValue = [&](const llvm::GlobalValue *Root) {
    // If the function is already split, e.g. if it was a dependency of
    // another function, skip it.
    if (SplitValues.count(Root))
      return;

    auto &ValueInfo = ValueInfos[Root];
    ValueMap.clear();
    SplitDeps.clear();
    auto ShouldSplit = [&](const llvm::GlobalValue *GlobalVal,
                           const struct ValueInfo &Info) {
      // Only clone root and the declaration of its dependencies.
      if (GlobalVal == Root) {
        SplitDeps.insert(GlobalVal);
        return true;
      }

      if ((Info.CanBeSplit || Info.UserEmpty) &&
          isa_and_nonnull<llvm::Function>(GlobalVal))
        return false;

      if (ValueInfo.Dependencies.contains(GlobalVal)) {
        SplitDeps.insert(GlobalVal);
        return true;
      }

      return false;
    };

    auto &Set = SetsToProcess.emplace_back();
    for (auto &[GlobalVal, Info] : ValueInfos) {
      if (ShouldSplit(GlobalVal, Info))
        Set.insert({GlobalVal, Info.GvIdx});
    }
    if (Set.empty())
      SetsToProcess.pop_back();

    // Record the split values.
    SplitValues.insert(SplitDeps.begin(), SplitDeps.end());
  };

  [[maybe_unused]] int64_t Count = 0;
  SmallVector<const llvm::GlobalValue *> ToSplit;
  unsigned UnnamedGlobal = NumFunctionBase;

  for (auto &Global : MainModule->globals()) {
    if (Global.hasInternalLinkage() || Global.hasPrivateLinkage()) {
      if (!Global.hasName()) {
        // Give unnamed GlobalVariable a unique name so that MCLink will not get
        // confused to name them while generating linked code since the IR
        // values can be different in each splits (for X86 backend.)
        // asan build inserts these unnamed GlobalVariables.
        Global.setName("__llvm_split_unnamed" + Twine(UnnamedGlobal++));
      }

      SymbolLinkageTypes.insert({Global.getName().str(), Global.getLinkage()});
      Global.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
      continue;
    }

    if (Global.hasExternalLinkage())
      continue;

    // TODO: Add special handling for `llvm.global_ctors` and
    // `llvm.global_dtors`, because otherwise they end up tying almost all
    // symbols into the same split.
    LLVM_DEBUG(llvm::dbgs()
                   << (Count++) << ": split global: " << Global << "\n";);
    ToSplit.emplace_back(&Global);
  }

  for (auto &Fn : MainModule->functions()) {
    if (Fn.isDeclaration())
      continue;

    ValueInfo &Info = ValueInfos[&Fn];
    if (Fn.hasInternalLinkage() || Fn.hasPrivateLinkage()) {
      // Avoid renaming when linking in MCLink.
      SymbolLinkageTypes.insert({Fn.getName().str(), Fn.getLinkage()});
      Fn.setLinkage(llvm::Function::LinkageTypes::WeakAnyLinkage);
    }

    if (Info.CanBeSplit || Info.UserEmpty) {
      LLVM_DEBUG(llvm::dbgs()
                     << (Count++) << ": split fn: " << Fn.getName() << "\n";);
      ToSplit.emplace_back(&Fn);
    }
  }

  // Run this now since we just changed the linkages.
  for (const llvm::GlobalValue *Value : ToSplit)
    SplitValue(Value);

  if (SetsToProcess.size() <= 1)
    return ProcessFn(forwardModule(std::move(MainModule)), std::nullopt,
                     NumFunctionBase);

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  std::string BufStr;
  llvm::raw_string_ostream BufOS(BufStr);
  {
    llvm::Module &Module = Strtab->externalizeStrings(std::move(MainModule));
    llvm::WriteBitcodeToFile(Module, BufOS);
  }

  auto Buf = WritableMemoryBuffer::getNewUninitMemBuffer(BufStr.size());
  memcpy(Buf->getBufferStart(), BufStr.c_str(), BufStr.size());
  unsigned NumFunctions = 0;
  for (auto [Idx, Set] : llvm::enumerate(SetsToProcess)) {
    unsigned Next = NumFunctions + Set.size();
    // Giving each function a unique ID across all splits for proper MC level
    // linking and codegen into one object file where duplicated functions
    // in each split will be deduplicated (with the linking).
    auto MakeModule =
        [Set = std::move(Set),
         Buf = MemoryBufferRef((*Buf).MemoryBuffer::getBuffer(), ""),
         Strtab = Strtab]() mutable {
          return readAndMaterializeDependencies(Buf, Set, *Strtab);
        };
    ProcessFn(std::move(MakeModule), Idx, NumFunctions);
    NumFunctions = Next;
  }
}

/// Collect all of the immediate global value users of `value`.
void LLVMModulePerFunctionSplitterImpl::collectValueUsers(
    const llvm::GlobalValue *Value) {
  SmallVector<const llvm::User *> Worklist(Value->users());

  while (!Worklist.empty()) {
    const llvm::User *UserIt = Worklist.pop_back_val();

    // Recurse into pure constant users.
    if (isa<llvm::Constant>(UserIt) && !isa<llvm::GlobalValue>(UserIt)) {
      Worklist.append(UserIt->user_begin(), UserIt->user_end());
      continue;
    }

    if (const auto *Inst = dyn_cast<llvm::Instruction>(UserIt)) {
      const llvm::Function *Func = Inst->getParent()->getParent();
      ValueInfos[Value].Users.insert(Func);
      ValueInfos[Func];
    } else if (const auto *GlobalVal = dyn_cast<llvm::GlobalValue>(UserIt)) {
      ValueInfos[Value].Users.insert(GlobalVal);
      ValueInfos[GlobalVal];
    } else {
      llvm_unreachable("unexpected user of global value");
    }
  }

  // If the current value is a mutable global variable, then it can't be
  // split.
  if (auto *Global = dyn_cast<llvm::GlobalVariable>(Value))
    ValueInfos[Value].CanBeSplit = Global->isConstant();
}

/// Propagate use information through the module.
void LLVMModulePerFunctionSplitterImpl::propagateUseInfo() {
  std::vector<ValueInfo *> Worklist;

  // Each value depends on itself. Seed the iteration with that.
  for (auto &[Value, Info] : ValueInfos) {
    if (auto Func = llvm::dyn_cast<llvm::Function>(Value)) {
      if (Func->isDeclaration())
        continue;
    }

    Info.Dependencies.insert(Value);
    Info.Value = Value;
    Worklist.push_back(&Info);
    if (!Info.CanBeSplit) {
      // If a value cannot be split, its users are also its dependencies.
      llvm::set_union(Info.Dependencies, Info.Users);
    }
  }

  while (!Worklist.empty()) {
    ValueInfo *Info = Worklist.back();
    Worklist.pop_back();

    // Propagate the dependencies of this value to its users.
    for (const llvm::GlobalValue *User : Info->Users) {
      ValueInfo &UserInfo = ValueInfos.find(User)->second;
      if (Info == &UserInfo)
        continue;
      bool Changed = false;

      // Merge dependency to user if current value is not a function that will
      // be split into a separate module.
      bool MergeToUserDep = true;
      if (llvm::isa_and_nonnull<llvm::Function>(Info->Value)) {
        MergeToUserDep = !Info->CanBeSplit;
      }

      // If there is a change, add the user info to the worklist.
      if (MergeToUserDep) {
        if (llvm::set_union(UserInfo.Dependencies, Info->Dependencies))
          Changed = true;
      }

      // If the value cannot be split, its users cannot be split either.
      if (!Info->CanBeSplit && UserInfo.CanBeSplit) {
        UserInfo.CanBeSplit = false;
        Changed = true;
        // If a value cannot be split, its users are also its dependencies.
        llvm::set_union(UserInfo.Dependencies, UserInfo.Users);
      }

      if (Changed) {
        UserInfo.Value = User;
        Worklist.push_back(&UserInfo);
      }
    }

    if (Info->CanBeSplit || isa_and_nonnull<llvm::GlobalValue>(Info->Value))
      continue;

    // If a value cannot be split, propagate its dependencies up to its
    // dependencies.
    for (const llvm::GlobalValue *Dep : Info->Dependencies) {
      ValueInfo &DepInfo = ValueInfos.find(Dep)->second;
      if (Info == &DepInfo)
        continue;
      if (llvm::set_union(DepInfo.Dependencies, Info->Dependencies)) {
        DepInfo.Value = Dep;
        Worklist.push_back(&DepInfo);
      }
    }
  }

  for (auto &[Value, Info] : ValueInfos) {
    Info.UserEmpty = Info.Users.empty() ||
                     (Info.Users.size() == 1 && Info.Users.contains(Value));
  }
}
