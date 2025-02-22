//===--- ModuleSplitter.cpp - Module Splitter -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ModuleSplitter.h"

#include "mlir/Support/LLVM.h"
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
  if (Error Err = ModuleOr.takeError())
    return Err;

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
    MemoryBuffer &Buf,
    const llvm::MapVector<const llvm::GlobalValue *, unsigned> &Set,
    const StringConstantTable &Strtab) {

  // First, create a lazy module with an internal bitcode materializer.
  // TODO: Not sure how to make lazy loading metadata work.
  LLVMModuleAndContext Result;
  {
    (void)Result.create(
        [&](llvm::LLVMContext &Ctx) -> Expected<std::unique_ptr<Module>> {
          return llvm::cantFail(llvm::getLazyBitcodeModule(
              llvm::MemoryBufferRef(Buf.getBuffer(), "<split-module>"), Ctx,
              /*ShouldLazyLoadMetadata=*/false));
        });
    Result->setModuleInlineAsm("");
  }

  SmallVector<unsigned> SortIndices =
      llvm::to_vector(llvm::make_second_range(Set));
  llvm::sort(SortIndices, std::less<unsigned>());
  auto IdxIt = SortIndices.begin();
  auto IdxEnd = SortIndices.end();

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

/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
void splitPerAnchored(LLVMModuleAndContext Module, LLVMSplitProcessFn ProcessFn,
                      llvm::SmallVectorImpl<llvm::Function> &Anchors) {
  LLVMModuleSplitterImpl impl(std::move(Module));
  impl.split(ProcessFn, Anchors);
}

void LLVMModuleSplitterImpl::split(
    LLVMSplitProcessFn processFn,
    llvm::SmallVectorImpl<llvm::Function> &Anchors) {
  // The use-def list is sparse. Use it to build a sparse dependency graph
  // between global values.
  auto strtab = RCRef<StringConstantTable>::create();
  unsigned gvIdx = 0;
  auto computeDeps = [&](const llvm::GlobalValue &value) {
    strtab->recordIfStringConstant(gvIdx, value);
    infos[&value].gvIdx = gvIdx++;
    collectImmediateDependencies(&value, &value);
  };
  // NOTE: The visitation of globals then functions has to line up with
  // `readAndMaterializeDependencies`.
  for (const llvm::GlobalVariable &global : mainModule->globals()) {
    computeDeps(global);
    if (!global.hasInternalLinkage() && !global.hasPrivateLinkage())
      transitiveDeps[&global];
  }
  for (const llvm::Function &fn : mainModule->functions()) {
    computeDeps(fn);
    if (!fn.isDeclaration() && (fn.hasExternalLinkage() || fn.hasWeakLinkage()))
      transitiveDeps[&fn];
  }

  // If there is only one (or fewer) exported functions, forward the main
  // module.
  if (transitiveDeps.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt,
                     /*numFunctionBase=*/0);

  // Now for each export'd global value, compute the transitive set of
  // dependencies using DFS.
  SmallVector<const llvm::GlobalValue *> worklist;
  for (auto &[value, deps] : transitiveDeps) {
    worklist.clear();
    worklist.push_back(value);
    while (!worklist.empty()) {
      const llvm::GlobalValue *it = worklist.pop_back_val();

      auto [iter, inserted] = deps.deps.insert({it, -1});
      if (!inserted) {
        // Already visited.
        continue;
      }
      // Pay the cost of the name lookup only on a miss.
      const ValueInfo &info = infos.at(it);
      iter->second = info.gvIdx;

      // If this value depends on another value that is going to be split, we
      // don't want to duplicate the symbol. Keep all the users together.
      if (it != value) {
        if (auto depIt = transitiveDeps.find(it);
            depIt != transitiveDeps.end()) {
          auto &users = splitAnchorUsers[it];
          users.insert(&deps);
          // Make sure to include the other value in its own user list.
          users.insert(&depIt->second);
          // We don't have to recurse since the subgraph will get processed.
          continue;
        }
      }

      // If this value depends on a mutable global, keep track of it. We have to
      // put all users of a mutable global in the same module.
      if (auto *global = dyn_cast<llvm::GlobalVariable>(it);
          global && !global->isConstant())
        splitAnchorUsers[global].insert(&deps);

      // Recursive on dependencies.
      llvm::append_range(worklist, info.dependencies);
    }

    deps.complete = true;
  }

  // For each mutable global, grab all the transitive users and put them in one
  // module. If global A has user set A* and global B has user set B* where
  // A* and B* have an empty intersection, all values in A* will be assigned 0
  // and all values in B* will be assigned 1. If global C has user set C* that
  // overlaps both A* and B*, it will overwrite both to 2.
  SmallVector<SmallVector<TransitiveDeps *>> bucketing(splitAnchorUsers.size());
  for (auto [curMutIdx, bucket, users] :
       llvm::enumerate(bucketing, llvm::make_second_range(splitAnchorUsers))) {
    for (TransitiveDeps *deps : users) {
      if (deps->mutIdx && *deps->mutIdx != curMutIdx) {
        auto &otherBucket = bucketing[*deps->mutIdx];
        for (TransitiveDeps *other : otherBucket) {
          bucket.push_back(other);
          other->mutIdx = curMutIdx;
        }
        otherBucket.clear();
        assert(*deps->mutIdx == curMutIdx);
      } else {
        bucket.push_back(deps);
        deps->mutIdx = curMutIdx;
      }
    }
  }

  // Now that we have assigned buckets to each value, merge the transitive
  // dependency sets of all values belonging to the same set.
  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned>> buckets(
      bucketing.size());
  for (auto [deps, bucket] : llvm::zip(bucketing, buckets)) {
    for (TransitiveDeps *dep : deps) {
      for (auto &namedValue : dep->deps)
        bucket.insert(namedValue);
    }
  }

  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned> *>
      setsToProcess;
  setsToProcess.reserve(buckets.size() + transitiveDeps.size());

  // Clone each mutable global bucket into its own module.
  for (auto &bucket : buckets) {
    if (bucket.empty())
      continue;
    setsToProcess.push_back(&bucket);
  }

  for (auto &[root, deps] : transitiveDeps) {
    // Skip values included in another transitive dependency set and values
    // included in mutable global sets.
    if (!deps.mutIdx)
      setsToProcess.push_back(&deps.deps);
  }

  if (setsToProcess.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt,
                     /*numFunctionBase=*/0);

  // Sort the sets by to schedule the larger modules first.
  llvm::sort(setsToProcess,
             [](auto *lhs, auto *rhs) { return lhs->size() > rhs->size(); });

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  auto buf = WriteableBuffer::get();
  {
    CompilerTimeTraceScope traceScope("writeMainModuleBitcode");
    llvm::Module &module = strtab->externalizeStrings(std::move(mainModule));
    llvm::WriteBitcodeToFile(module, *buf);
  }

  unsigned numFunctions = 0;
  for (auto [idx, set] : llvm::enumerate(setsToProcess)) {
    unsigned next = numFunctions + set->size();
    auto makeModule = [set = std::move(*set), buf = BufferRef(buf.copy()),
                       strtab = strtab.copy()]() mutable {
      return readAndMaterializeDependencies(std::move(buf), set, *strtab,
                                            /*ignoreFns=*/{});
    };
    processFn(std::move(makeModule), idx, numFunctions);
    numFunctions = next;
  }
}

void LLVMModuleSplitterImpl::collectImmediateDependencies(
    const llvm::Value *value, const llvm::GlobalValue *orig) {
  for (const llvm::Value *user : value->users()) {
    // Recurse into pure constant users.
    if (isa<llvm::Constant>(user) && !isa<llvm::GlobalValue>(user)) {
      collectImmediateDependencies(user, orig);
      continue;
    }

    if (auto *inst = dyn_cast<llvm::Instruction>(user)) {
      const llvm::Function *func = inst->getParent()->getParent();
      infos[func].dependencies.push_back(orig);
    } else if (auto *globalVal = dyn_cast<llvm::GlobalValue>(user)) {
      infos[globalVal].dependencies.push_back(orig);
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
  LLVMModulePerFunctionSplitterImpl(LLVMModuleAndContext module)
      : mainModule(std::move(module)) {}

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void
  split(LLVMSplitProcessFn processFn,
        llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
        unsigned numFunctionBase);

private:
  struct ValueInfo {
    const llvm::Value *value = nullptr;
    bool canBeSplit = true;
    llvm::SmallPtrSet<const llvm::GlobalValue *, 4> dependencies;
    llvm::SmallPtrSet<const llvm::GlobalValue *, 4> users;
    /// Map each global value to its index in the module. We will use this to
    /// materialize global values from bitcode.
    unsigned gvIdx;
    bool userEmpty = true;
  };

  /// Collect all of the immediate global value users of `value`.
  void collectValueUsers(const llvm::GlobalValue *value);

  /// Propagate use information through the module.
  void propagateUseInfo();

  /// The main LLVM module being split.
  LLVMModuleAndContext mainModule;

  /// The value info for each global value in the module.
  llvm::MapVector<const llvm::GlobalValue *, ValueInfo> valueInfos;
};
} // namespace

static void
checkDuplicates(llvm::MapVector<const llvm::GlobalValue *, unsigned> &set,
                llvm::StringSet<> &seenFns, llvm::StringSet<> &dupFns) {
  for (auto [gv, _] : set) {
    if (auto fn = dyn_cast<llvm::Function>(gv)) {
      if (!seenFns.insert(fn->getName()).second) {
        dupFns.insert(fn->getName());
      }
    }
  }
}

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void KGEN::splitPerFunction(
    LLVMModuleAndContext module, LLVMSplitProcessFn processFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    unsigned numFunctionBase) {
  CompilerTimeTraceScope traceScope("splitPerFunction");
  LLVMModulePerFunctionSplitterImpl impl(std::move(module));
  impl.split(processFn, symbolLinkageTypes, numFunctionBase);
}

/// Split the LLVM module into multiple modules using the provided process
/// function.
void LLVMModulePerFunctionSplitterImpl::split(
    LLVMSplitProcessFn processFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    unsigned numFunctionBase) {
  // Compute the value info for each global in the module.
  // NOTE: The visitation of globals then functions has to line up with
  // `readAndMaterializeDependencies`.
  auto strtab = RCRef<StringConstantTable>::create();
  unsigned gvIdx = 0;
  auto computeUsers = [&](const llvm::GlobalValue &value) {
    strtab->recordIfStringConstant(gvIdx, value);
    valueInfos[&value].gvIdx = gvIdx++;
    collectValueUsers(&value);
  };
  llvm::for_each(mainModule->globals(), computeUsers);
  llvm::for_each(mainModule->functions(), computeUsers);

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
  llvm::DenseSet<const llvm::Value *> splitValues;
  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned>>
      setsToProcess;

  // Hoist these collections to re-use memory allocations.
  llvm::ValueToValueMapTy valueMap;
  SmallPtrSet<const llvm::Value *, 4> splitDeps;
  auto splitValue = [&](const llvm::GlobalValue *root) {
    // If the function is already split, e.g. if it was a dependency of
    // another function, skip it.
    if (splitValues.count(root))
      return;

    auto &valueInfo = valueInfos[root];
    valueMap.clear();
    splitDeps.clear();
    auto shouldSplit = [&](const llvm::GlobalValue *globalVal,
                           const ValueInfo &info) {
      // Only clone root and the declaration of its dependencies.
      if (globalVal == root) {
        splitDeps.insert(globalVal);
        return true;
      }

      if ((info.canBeSplit || info.userEmpty) &&
          isa_and_nonnull<llvm::Function>(globalVal))
        return false;

      if (valueInfo.dependencies.contains(globalVal)) {
        splitDeps.insert(globalVal);
        return true;
      }

      return false;
    };

    auto &set = setsToProcess.emplace_back();
    for (auto &[globalVal, info] : valueInfos) {
      if (shouldSplit(globalVal, info))
        set.insert({globalVal, info.gvIdx});
    }
    if (set.empty())
      setsToProcess.pop_back();

    // Record the split values.
    splitValues.insert(splitDeps.begin(), splitDeps.end());
  };

  [[maybe_unused]] int64_t count = 0;
  SmallVector<const llvm::GlobalValue *> toSplit;
  unsigned unnamedGlobal = numFunctionBase;
  for (auto &global : mainModule->globals()) {
    if (global.hasInternalLinkage() || global.hasPrivateLinkage()) {
      if (!global.hasName()) {
        // Give unnamed GlobalVariable a unique name so that MCLink will not get
        // confused to name them while generating linked code since the IR
        // values can be different in each splits (for X86 backend.)
        // asan build inserts these unnamed GlobalVariables.
        global.setName("__mojo_unnamed" + Twine(unnamedGlobal++));
      }

      symbolLinkageTypes.insert({global.getName().str(), global.getLinkage()});
      global.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
      continue;
    }

    if (global.hasExternalLinkage())
      continue;

    // TODO: Add special handling for `llvm.global_ctors` and
    // `llvm.global_dtors`, because otherwise they end up tying almost all
    // symbols into the same split.
    LLVM_DEBUG(llvm::dbgs()
                   << (count++) << ": split global: " << global << "\n";);
    toSplit.emplace_back(&global);
  }

  for (auto &fn : mainModule->functions()) {
    if (fn.isDeclaration())
      continue;

    ValueInfo &info = valueInfos[&fn];
    if (fn.hasInternalLinkage() || fn.hasPrivateLinkage()) {
      // Avoid renaming when linking in MCLink.
      symbolLinkageTypes.insert({fn.getName().str(), fn.getLinkage()});
      fn.setLinkage(llvm::Function::LinkageTypes::WeakAnyLinkage);
    }

    if (info.canBeSplit || info.userEmpty) {
      LLVM_DEBUG(llvm::dbgs()
                     << (count++) << ": split fn: " << fn.getName() << "\n";);
      toSplit.emplace_back(&fn);
    }
  }

  // Run this now since we just changed the linkages.
  for (const llvm::GlobalValue *value : toSplit)
    splitValue(value);

  if (setsToProcess.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt,
                     numFunctionBase);

  auto duplicatedFns = std::move(mainModule.duplicatedFns);

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  auto buf = WriteableBuffer::get();
  {
    CompilerTimeTraceScope traceScope("writeMainModuleBitcode");
    llvm::Module &module = strtab->externalizeStrings(std::move(mainModule));
    llvm::WriteBitcodeToFile(module, *buf);
  }

  unsigned numFunctions = numFunctionBase;
  llvm::StringSet<> seenFns;
  for (auto [idx, set] : llvm::enumerate(setsToProcess)) {
    // Giving each function a unique ID across all splits for proper MC level
    // linking and codegen into one object file where duplicated functions
    // in each split will be deduplicated (with the linking).
    llvm::StringSet<> currDuplicatedFns = duplicatedFns;
    checkDuplicates(set, seenFns, currDuplicatedFns);

    unsigned next = numFunctions + set.size();
    auto makeModule = [set = std::move(set), buf = BufferRef(buf.copy()),
                       strtab = strtab.copy(), currDuplicatedFns]() mutable {
      return readAndMaterializeDependencies(std::move(buf), set, *strtab,
                                            currDuplicatedFns);
    };
    processFn(std::move(makeModule), idx, numFunctions);
    numFunctions = next;
  }
}

/// Collect all of the immediate global value users of `value`.
void LLVMModulePerFunctionSplitterImpl::collectValueUsers(
    const llvm::GlobalValue *value) {
  SmallVector<const llvm::User *> worklist(value->users());

  while (!worklist.empty()) {
    const llvm::User *userIt = worklist.pop_back_val();

    // Recurse into pure constant users.
    if (isa<llvm::Constant>(userIt) && !isa<llvm::GlobalValue>(userIt)) {
      worklist.append(userIt->user_begin(), userIt->user_end());
      continue;
    }

    if (const auto *inst = dyn_cast<llvm::Instruction>(userIt)) {
      const llvm::Function *func = inst->getParent()->getParent();
      valueInfos[value].users.insert(func);
      valueInfos[func];
    } else if (const auto *globalVal = dyn_cast<llvm::GlobalValue>(userIt)) {
      valueInfos[value].users.insert(globalVal);
      valueInfos[globalVal];
    } else {
      llvm_unreachable("unexpected user of global value");
    }
  }

  // If the current value is a mutable global variable, then it can't be
  // split.
  if (auto *global = dyn_cast<llvm::GlobalVariable>(value))
    valueInfos[value].canBeSplit = global->isConstant();
}

/// Propagate use information through the module.
void LLVMModulePerFunctionSplitterImpl::propagateUseInfo() {
  std::vector<ValueInfo *> worklist;

  // Each value depends on itself. Seed the iteration with that.
  for (auto &[value, info] : valueInfos) {
    if (auto func = llvm::dyn_cast<llvm::Function>(value)) {
      if (func->isDeclaration())
        continue;
    }

    info.dependencies.insert(value);
    info.value = value;
    worklist.push_back(&info);
    if (!info.canBeSplit) {
      // If a value cannot be split, its users are also its dependencies.
      llvm::set_union(info.dependencies, info.users);
    }
  }

  while (!worklist.empty()) {
    ValueInfo *info = worklist.back();
    worklist.pop_back();

    // Propagate the dependencies of this value to its users.
    for (const llvm::GlobalValue *user : info->users) {
      ValueInfo &userInfo = valueInfos.find(user)->second;
      if (info == &userInfo)
        continue;
      bool changed = false;

      // Merge dependency to user if current value is not a function that will
      // be split into a separate module.
      bool mergeToUserDep = true;
      if (llvm::isa_and_nonnull<llvm::Function>(info->value)) {
        mergeToUserDep = !info->canBeSplit;
      }

      // If there is a change, add the user info to the worklist.
      if (mergeToUserDep) {
        if (llvm::set_union(userInfo.dependencies, info->dependencies))
          changed = true;
      }

      // If the value cannot be split, its users cannot be split either.
      if (!info->canBeSplit && userInfo.canBeSplit) {
        userInfo.canBeSplit = false;
        changed = true;
        // If a value cannot be split, its users are also its dependencies.
        llvm::set_union(userInfo.dependencies, userInfo.users);
      }

      if (changed) {
        userInfo.value = user;
        worklist.push_back(&userInfo);
      }
    }

    if (info->canBeSplit || isa_and_nonnull<llvm::GlobalValue>(info->value))
      continue;

    // If a value cannot be split, propagate its dependencies up to its
    // dependencies.
    for (const llvm::GlobalValue *dep : info->dependencies) {
      ValueInfo &depInfo = valueInfos.find(dep)->second;
      if (info == &depInfo)
        continue;
      if (llvm::set_union(depInfo.dependencies, info->dependencies)) {
        depInfo.value = dep;
        worklist.push_back(&depInfo);
      }
    }
  }

  for (auto &[value, info] : valueInfos) {
    info.userEmpty = info.users.empty() ||
                     (info.users.size() == 1 && info.users.contains(value));
  }
}
