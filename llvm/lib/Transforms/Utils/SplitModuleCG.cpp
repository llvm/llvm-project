//===- SplitModuleCG.cpp - Split a module by its call graph ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the llvm::SplitModuleCG class, which splits a module
// into partitions based on its call graph.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SplitModuleCG.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
using namespace llvm;

#define DEBUG_TYPE "split-module-cg"

static cl::opt<bool>
    EnablePrintSimplifiedCallGraph("enable-print-simplified-callgraph",
                                   cl::Hidden, cl::init(false),
                                   cl::desc("print SimplifiedCallGraph"));

namespace {

using PartitionID = unsigned;

} // namespace

/// Returns whether duplicate definitions of \p F across partitions may be
/// downgraded to available_externally. This is safe for external functions
/// (either originally external or promoted by externalize), and for
/// weak_odr/linkonce_odr functions whose equivalent definitions can be
/// deduplicated to reduce codegen. Interposable linkages (weak/linkonce
/// non-ODR) are excluded since downgrading them would change their
/// optimization semantics.
static bool canDowngradeToAvailableExternally(const Function &F) {
  return !F.isDeclaration() &&
         (F.hasExternalLinkage() || F.hasWeakODRLinkage() ||
          F.hasLinkOnceODRLinkage());
}

/// Fallback for getUniqueModuleId when the module has no exportable symbols
/// to seed a hash (e.g. every definition is in a comdat or has linkonce_odr
/// linkage), in which case getUniqueModuleId returns "". Hashes the module
/// identifier and the names of all named global values so that such modules
/// can still be split with promoted locals renamed uniquely per module.
static std::string computeFallbackSuffix(const Module &M) {
  MD5 Md5;
  Md5.update(M.getModuleIdentifier());
  Md5.update(ArrayRef<uint8_t>{0});
  for (const GlobalValue &GV : M.global_values()) {
    if (!GV.hasName())
      continue;
    Md5.update(GV.getName());
    Md5.update(ArrayRef<uint8_t>{0});
  }
  MD5::MD5Result R;
  Md5.final(R);
  SmallString<32> Str;
  MD5::stringifyResult(R, Str);
  return ("." + Str).str();
}

std::vector<DenseSet<const Function *>> SplitModuleCG::doPartitioning() {
  LLVM_DEBUG(dbgs() << "\n--Partitioning Starts--\n");
  assert(NumPartitions != 0 && "Partition count must be at least 1");
  std::vector<DenseSet<const Function *>> Partitions;
  Partitions.resize(NumPartitions);

  auto ComparePartitions = [](const std::pair<PartitionID, CostType> &LHS,
                              const std::pair<PartitionID, CostType> &RHS) {
    // When two partitions have the same cost, assign to the one with the
    // biggest ID first. This allows us to put things in P0 last, because P0 may
    // have other stuff added later.
    if (LHS.second == RHS.second)
      return LHS.first < RHS.first;
    return LHS.second > RHS.second;
  };

  std::vector<std::pair<PartitionID, CostType>> BalancingQueue;
  for (unsigned I = 0; I < NumPartitions; ++I)
    BalancingQueue.emplace_back(I, 0);

  for (auto &CurFn : FWDWorkList) {
    // Normal "load-balancing", assign to partition with least pressure.
    auto [PID, _] = BalancingQueue.back();

    // Insert the root function and its dependencies into the partition,
    // tracking the cost of newly inserted functions so the balancing queue
    // can be updated. CurFn.Dependencies includes the root F itself.
    auto &FnsInPart = Partitions[PID];
    CostType AddedCost = 0;
    for (const Function *Dep : CurFn.Dependencies)
      if (FnsInPart.insert(Dep).second)
        AddedCost += FuncsCosts.lookup(Dep);

    // Update the cost of the selected partition, which is the entry at the
    // back of the sorted queue, before re-sorting.
    BalancingQueue.back().second += AddedCost;

    sort(BalancingQueue, ComparePartitions);
  }

  return Partitions;
}

void SplitModuleCG::calculateFunctionCosts() {
  ModuleCost = 0;
  for (auto &Fn : M) {
    if (Fn.isDeclaration())
      continue;

    CostType FnCost = 0;
    for (const auto &BB : Fn)
      FnCost += std::distance(BB.begin(), BB.end());
    assert(FnCost != 0);
    FuncsCosts[&Fn] = FnCost;
    // Signed overflow is UB, so perform the check in unsigned arithmetic,
    // where wraparound is well-defined.
    assert(static_cast<uint64_t>(ModuleCost) + static_cast<uint64_t>(FnCost) >=
               static_cast<uint64_t>(ModuleCost) &&
           "Overflow!");
    ModuleCost += FnCost;
  }
}

void SplitModuleCG::dealWithMpart(Module &MPart, unsigned I) {
  // Downgrade duplicate definitions of external functions to
  // available_externally. The first partition to define such a function keeps
  // the real definition; all other partitions get available_externally copies.
  for (auto &PartFunc : MPart.functions()) {
    if (PartFunc.isDeclaration())
      continue;
    // Look up the corresponding function in the original module M to check
    // its ExternalFunction status.
    auto *OrigFn = M.getFunction(PartFunc.getName());
    if (!ExternalFunction.contains(OrigFn))
      continue;
    if (!ExternalFunction[OrigFn]) {
      PartFunc.setLinkage(GlobalValue::AvailableExternallyLinkage);
      PartFunc.setComdat(nullptr);
    } else {
      ExternalFunction[OrigFn] = false;
    }
  }

  // Erase declarations that are no longer used.
  for (auto &GV : make_early_inc_range(MPart.global_values()))
    if (GV.isDeclaration() && GV.use_empty())
      GV.eraseFromParent();

  // Rename GlobalValues whose linkage was promoted from internal to external,
  // to avoid duplicate symbols across partitions in ThinLTO. Use the naming
  // convention "name.llvm.<suffix>" so the promoted internal cannot clash with
  // an external that happens to share the same name. The suffix is derived
  // from the module via getUniqueModuleId, so it is consistent across all
  // partitions.
  std::string Suffix = getUniqueModuleId(&M);
  if (Suffix.empty())
    Suffix = computeFallbackSuffix(M);
  for (auto &GV : MPart.global_values()) {
    // Only rename symbols that were promoted from internal to external: skip
    // those that are still internal, and those that were already external in
    // the source module (recorded in OriginalExternals).
    if (GV.hasLocalLinkage() || OriginalExternals.contains(GV.getName()))
      continue;
    GV.setName((GV.getName() + ".llvm" + Suffix).str());
  }

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << MPart.getModuleIdentifier() << "  : \n");
  for (auto &F : MPart)
    if (!F.isDeclaration())
      LLVM_DEBUG(dbgs() << "   [Function: ] " << I << "  " << F.getName() << " "
                        << F.getLinkage() << "\n");
#endif
}

FunctionWithDependencies::FunctionWithDependencies(
    SimplifiedCallGraph &SCG,
    const DenseMap<const Function *, CostType> &FnCosts, const Function *F)
    : F(F) {
  assert(!F->isDeclaration());

  // Collect F and all non-declaration functions transitively called by F.
  SmallVector<const Function *> WorkList({F});
  Dependencies.insert(F);

  while (!WorkList.empty()) {
    const auto *CurFn = WorkList.pop_back_val();
    assert(!CurFn->isDeclaration());

    // Walk the callees of CurFn recorded in SimplifiedCallGraph and
    // add them to Dependencies, recursing transitively via the WorkList.
    for (auto &SCGNode : *SCG.at(CurFn)) {
      auto *Callee = SCGNode->getFunction();
      if (!Callee || Callee->isDeclaration())
        continue;
      if (Dependencies.insert(Callee).second)
        WorkList.push_back(Callee);
    }
  }

  for (const auto *Dep : Dependencies)
    TotalCost += FnCosts.lookup(Dep);
}

void SplitModuleCG::createWorkList() {
  // First, find all the entry functions with an in-degree of 0
  // (i.e., those that are not called by any function).
  for (auto &SCGNode : SCG->values()) {
    Function *F = SCGNode->getFunction();
    if (F && SCGNode->getNumReferences() == 0)
      EntryFuncs.insert(F);
  }

  // Second, find all the dependencies of each entry function.
  for (auto *F : EntryFuncs)
    FWDWorkList.emplace_back(*SCG, FuncsCosts, F);

  // Third, find all the functions that are not in the worklist.
  DenseSet<const Function *> SeenFunctions;
  for (const auto &Fwd : FWDWorkList)
    SeenFunctions.insert(Fwd.Dependencies.begin(), Fwd.Dependencies.end());
  for (auto &F : M) {
    // This function may be in a cycle, and therefore is not a dependency of
    // any root, which is treated as a root function here.
    if (F.isDeclaration() || SeenFunctions.contains(&F))
      continue;
    FWDWorkList.emplace_back(*SCG, FuncsCosts, &F);
    auto &Fwd = FWDWorkList.back();
    EntryFuncs.insert(&F);
    SeenFunctions.insert(Fwd.Dependencies.begin(), Fwd.Dependencies.end());
  }
}

void SplitModuleCG::sortWorkList() {
  // Sort the worklist so the most expensive roots are seen first.
  sort(FWDWorkList, [](const auto &A, const auto &B) {
    // Sort by total cost, and if the total cost is identical, sort
    // alphabetically
    if (A.TotalCost == B.TotalCost)
      return A.F->getName() < B.F->getName();
    return A.TotalCost > B.TotalCost;
  });

  LLVM_DEBUG(dbgs() << "Number of callgraphs to be allocated: "
                    << FWDWorkList.size() << "   Module cost: " << ModuleCost
                    << "\n");
  LLVM_DEBUG(dbgs() << "callgraphs: \n");
#ifndef NDEBUG
  for (auto &Fwd : FWDWorkList)
    LLVM_DEBUG(dbgs() << "[root] " << Fwd.F->getName()
                      << " (totalCost:" << Fwd.TotalCost
                      << ";   root function cost: " << FuncsCosts[Fwd.F]
                      << ";   has dependency: " << Fwd.Dependencies.size()
                      << ")\n");
#endif
}

void SplitModuleCG::splitModule(ModuleCreationCallback ModuleCallback,
                                ContextCreationCallback MakeCtx) {
  for (Function &F : M) {
    if (F.hasLocalLinkage() && F.hasOneUse() && !F.hasAddressTaken())
      continue;
    F.externalize();
    // Record functions that may be defined in multiple partitions so that
    // dealWithMpart can downgrade duplicates to available_externally.
    if (canDowngradeToAvailableExternally(F))
      ExternalFunction[&F] = true;
  }
  for (GlobalVariable &GV : M.globals())
    GV.externalize();
  for (GlobalAlias &GA : M.aliases())
    GA.externalize();
  for (GlobalIFunc &GI : M.ifuncs())
    GI.externalize();
  // TODO: The verifier requires an alias to point to a definition and an
  // ifunc resolver to be a definition, but the aliasee/resolver may be
  // assigned to a different partition than its alias/ifunc. Handling this
  // is deferred to a follow-up patch.

  // Sort the worklist here, after all potential renaming has occurred.
  sortWorkList();

  // Assign callgraphs into NumPartitions partitions.
  auto Partitions = doPartitioning();
  assert(Partitions.size() == NumPartitions);

  auto ShouldCloneDefinition = [&](unsigned I, const GlobalValue *GV) {
    const auto &FnsInPart = Partitions[I];

    // Functions go in their assigned partition.
    if (const auto *FnToClone = dyn_cast<Function>(GV))
      return FnsInPart.contains(FnToClone);
    // Everything else goes in the first partition.
    return I == 0;
  };

  // TODO: Consider parallelizing the per-partition CloneModule call itself.
  // Today the loop below serially clones M into NumPartitions partitions in
  // the main thread, then enqueues the opt+codegen work on a thread pool. If
  // CloneModule becomes a bottleneck for large modules, the clones could be
  // produced in parallel too — but that would require either per-thread
  // LLVMContexts for the clone step or a thread-safe CloneModule, neither of
  // which is straightforward. dealWithMpart's handling of duplicate
  // definitions is also order-dependent: the first partition processed keeps
  // the real definition of an externalized function and later ones downgrade
  // their copies to available_externally. Parallelizing would make that
  // choice non-deterministic unless the owner partition were pre-assigned.
  DefaultThreadPool SplitThreadPool(
      heavyweight_hardware_concurrency(NumPartitions));
  for (unsigned I = 0; I < NumPartitions; ++I) {
    ValueToValueMapTy VMap;
    std::unique_ptr<Module> MPart(
        CloneModule(M, VMap, [&](const GlobalValue *GV) {
          return ShouldCloneDefinition(I, GV);
        }));

    dealWithMpart(*MPart, I);

    // Serialize the cloned partition to bitcode and re-parse it inside the
    // worker thread's own LLVMContext. This round-trip is required because
    // LLVM's Module / LLVMContext are not safe to share across threads:
    // CloneModule above runs in the main thread's context, but the worker
    // thread created below needs its own context to run opt + codegen
    // concurrently without racing on shared internal state. So bitcode
    // serialization is the supported way to move a Module between contexts.
    SmallString<0> BC;
    raw_svector_ostream BCOS(BC);
    WriteBitcodeToFile(*MPart, BCOS);
    MPart.reset();
    SplitThreadPool.async(
        [&, I](SmallString<0> BC) {
          // Each partition is parsed into its own context, created by the
          // caller, as LLVMContext cannot be shared across threads.
          std::unique_ptr<LLVMContext> Ctx = MakeCtx();
          // Give each partition a distinct module name reflecting its index,
          // so that diagnostics and debug output identify the partition.
          std::string PartName = ("split-module-cg." + Twine(I)).str();
          Expected<std::unique_ptr<Module>> MOrErr =
              parseBitcodeFile(MemoryBufferRef(BC.str(), PartName), *Ctx);
          BC = SmallString<0>();
          if (!MOrErr)
            report_fatal_error(MOrErr.takeError());
          ModuleCallback(std::move(MOrErr.get()), I);
        },
        std::move(BC));
  }
  // The inner lambda (which runs in a worker thread) captures our local
  // variables, so we need to wait for the worker threads to terminate before
  // we can leave the function scope.
  SplitThreadPool.wait();
}

SplitModuleCG::SplitModuleCG(Module &M, unsigned LimitPartition)
    : NumPartitions(LimitPartition), M(M), CG(M) {
  // Track existing non-local symbols. This ensures that when we promote
  // internal symbols to external for partitioning, we can handle renaming
  // and avoid conflicts.
  for (const auto &GV : M.global_values())
    if (!GV.hasLocalLinkage())
      OriginalExternals.insert(GV.getName());

  calculateFunctionCosts();

  // Construct a simplified call graph to facilitate worklist generation.
  SCG = std::make_unique<SimplifiedCallGraph>(CG);

  // Populate the worklist with root functions and their transitive
  // dependencies. This worklist serves as the foundation for the
  // subsequent module partitioning.
  createWorkList();

  if (NumPartitions == 0) {
    // Auto mode: one partition per call-graph root, but capped to the
    // available hardware parallelism so that a module with a very large
    // number of entry functions does not create an excessive number of
    // partitions (and thus output objects).
    unsigned MaxPartitions =
        heavyweight_hardware_concurrency().compute_thread_count();
    NumPartitions = std::min<unsigned>(EntryFuncs.size(), MaxPartitions);
  } else if (NumPartitions > EntryFuncs.size()) {
    // Do not create more partitions than there are call-graph roots.
    NumPartitions = EntryFuncs.size();
  }
  if (NumPartitions == 0)
    NumPartitions = 1;
}

SimplifiedCallGraph::SimplifiedCallGraph(CallGraph &CG) {
  for (auto &NodePair : CG) {
    auto &CGNode = NodePair.second;
    Function *F = CGNode->getFunction();
    if (!F || F->isDeclaration())
      continue;

    SimplifiedCallGraphNode *SCGNode = getOrInsertFunction(F);

    for (const auto &CGNodeItem : *CGNode) {
      Function *Called = CGNodeItem.second->getFunction();
      if (!Called || Called->isDeclaration())
        continue;
      SCGNode->addCalledFunction(getOrInsertFunction(Called));
    }
  }

  if (EnablePrintSimplifiedCallGraph)
    print();
}

void SimplifiedCallGraph::print() {
#ifndef NDEBUG
  for (auto &SCGItem : FunctionMap) {
    LLVM_DEBUG(dbgs() << "Call graph node for function: '"
                      << SCGItem.first->getName() << "' #uses="
                      << SCGItem.second->getNumReferences() << "\n");

    for (const auto &Callee : *SCGItem.second)
      LLVM_DEBUG(dbgs() << "          Calls function : '"
                        << Callee->getFunction()->getName() << " '\n");
  }
#endif
}

SimplifiedCallGraphNode *
SimplifiedCallGraph::getOrInsertFunction(const Function *F) {
  auto &SCGN = FunctionMap[F];
  if (SCGN)
    return SCGN.get();

  SCGN = std::make_unique<SimplifiedCallGraphNode>(const_cast<Function *>(F));
  return SCGN.get();
}
