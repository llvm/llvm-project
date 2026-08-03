#include "llvm/Transforms/Utils/SplitModuleCG.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/MD5.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <thread>
using namespace llvm;

#define DEBUG_TYPE "split-module-CG"

namespace {

static cl::opt<bool> enablePrintSimplifyCallGraph(
    "enable-print-simplify-callgraph", cl::Hidden, cl::init(false),
    cl::desc("print SimplifyCallGraph"));

using PartitionID = unsigned;

static void externalize(GlobalValue *GV) {
  if (GV->hasLocalLinkage()) {
    GV->setLinkage(GlobalValue::ExternalLinkage);
    GV->setVisibility(GlobalValue::HiddenVisibility);
  }

  // Unnamed entities must be named consistently between modules. setName will
  // give a distinct name to each such entity.
  if (!GV->hasName())
    GV->setName("__llvmsplit_unnamed");
}

} // namespace

std::vector<DenseSet<const Function *>> SplitModuleCG::doPartitioning() {
  LLVM_DEBUG(dbgs() << "\n--Partitioning Starts--\n");
  // Performs all of the partitioning work on M.
  std::vector<DenseSet<const Function *>> Partitions;
  Partitions.resize(N);
  if (N == 0)
    return Partitions;

  auto ComparePartitions = [](const std::pair<PartitionID, CostType> &a,
                              const std::pair<PartitionID, CostType> &b) {
    // When two partitions have the same cost, assign to the one with the
    // biggest ID first. This allows us to put things in P0 last, because P0 may
    // have other stuff added later.
    if (a.second == b.second)
      return a.first < b.first;
    return a.second > b.second;
  };

  std::vector<std::pair<PartitionID, CostType>> BalancingQueue;
  for (unsigned I = 0; I < N; ++I)
    BalancingQueue.emplace_back(I, 0);

  // Helper function to handle assigning a function to a partition. This takes
  // care of updating the balancing queue.
  const auto AssignToPartition = [&](PartitionID PID,
                                     const FunctionWithDependencies &FWD) {
    // Insert the root function and its dependencies into the partition,
    // tracking the cost of newly inserted functions so the balancing queue
    // can be updated.
    auto &FnsInPart = Partitions[PID];
    CostType AddedCost = 0;
    if (FnsInPart.insert(FWD.F).second)
      AddedCost += FuncsCosts.at(FWD.F);
    for (const Function *Dep : FWD.Dependencies)
      if (FnsInPart.insert(Dep).second)
        AddedCost += FuncsCosts.lookup(Dep);

    // Update the balancing queue. We scan backwards because in the common
    // case the target partition is at the end of the sorted queue.
    for (auto &[QueuePID, Cost] : reverse(BalancingQueue)) {
      if (QueuePID != PID)
        continue;
      Cost += AddedCost;
    }

    sort(BalancingQueue, ComparePartitions);
  };

  for (auto &CurFn : FWDWorkList) {
    // Normal "load-balancing", assign to partition with least pressure.
    auto [PID, CurCost] = BalancingQueue.back();
    AssignToPartition(PID, CurFn);
  }

  return Partitions;
}

void SplitModuleCG::calculateFunctionCosts() {
  ModuleCost = 0;
  for (auto &Fn : M) {
    if (Fn.isDeclaration())
      continue;

    CostType FnCost = 0;
    for (const auto &BB : Fn) {
      CostType CostVal = std::distance(BB.begin(), BB.end());
      FnCost += CostVal;
    }
    assert(FnCost != 0);
    FuncsCosts[&Fn] = FnCost;
    assert((ModuleCost + FnCost) >= ModuleCost && "Overflow!");
    ModuleCost += FnCost;
  }
}

void SplitModuleCG::dealWithMpart(Module &MPart, unsigned I,
                                   function_ref<bool(const GlobalValue *)> NeedsConservativeImport) {
  // Collect promoted symbols (those that were local but are now external due
  // to externalize(), and therefore are not in the OriginalExternals set
  // captured at construction time).
  //
  // Note: here we only *record* the rename in PromotedRenames; we do not
  // perform the actual renaming immediately. The rename is applied after the
  // opt pipeline has completed. This is intentional: deferring the rename
  // minimizes the impact of renaming on subsequent optimizations.
  auto checkPromoted = [&](const GlobalValue &GV) {
    // now is external (not local), but not in external set.
    if (!GV.hasLocalLinkage() && !OriginalExternals.contains(GV.getName())) {
      if (PromotedRenames.count(GV.getName()))
        return;
      // Use the naming convention "name.llvm.<suffix>" so the
      // promoted local cannot clash with an external that happens to share
      // the same name in another module/partition.
      std::string Suffix = getUniqueModuleId(&M);
      std::string NewName = (GV.getName() + ".llvm" + Suffix).str();
      PromotedRenames[GV.getName()] = NewName;
    }
  };

  auto AvailableExternalizeFunc = [&](llvm::Function &Func) {
    Func.setLinkage(GlobalValue::AvailableExternallyLinkage);
    Func.setComdat(nullptr);
  };

  for (const auto &GV : MPart.global_values())
    checkPromoted(GV);
  // Clean-up conservatively imported GVs without any users.
  for (auto &GV : make_early_inc_range(MPart.globals())) {
    if (NeedsConservativeImport(&GV) && GV.use_empty())
      GV.eraseFromParent();
  }

  for (auto &func : MPart.functions()) {
    auto Fn = M.getFunction(func.getName());
    if (externalFunction.count(Fn) && !func.isDeclaration()) {
      if (!externalFunction[Fn]) {
        AvailableExternalizeFunc(func);
      } else {
        externalFunction[Fn] = false;
      }
    }
  }

  LLVM_DEBUG(dbgs() << MPart.getModuleIdentifier() << "  : \n");
  for (auto &F : MPart) {
    if (!F.isDeclaration())
      LLVM_DEBUG(dbgs() << "   [Function: ] " << I << "  " << F.getName() << " "
                        << F.getLinkage() << "\n");
  }
}

void SplitModuleCG::createWorkList() {
  // First, find all the entry functions with an in-degree of 0
  // (i.e., those that are not called by any function).
  for (auto &NodePair : *SCG) {
    SimplifyCallGraphNode *SCGNode = NodePair.second.get();
    Function *F = SCGNode->getFunction();
    if (F && SCGNode->getNumReferences() == 0) {
      EntryFuncs.insert(F);
    }
  }

  // Second, find all the dependencies of each entry function.
  for (auto *F : EntryFuncs) {
    FWDWorkList.emplace_back(*SCG, FuncsCosts, F);
  }

  // Third, find all the functions that are not in the worklist.
  DenseSet<const Function *> SeenFunctions;
  for (const auto &FWD : FWDWorkList) {
    SeenFunctions.insert(FWD.F);
    SeenFunctions.insert(FWD.Dependencies.begin(), FWD.Dependencies.end());
  }
  for (auto &F : M) {
    // This function may be in a ring, and therefore is not a dependency of
    // any root, which is treated as a root function here.
    if (!F.isDeclaration() && !SeenFunctions.count(&F)) {
      FWDWorkList.emplace_back(*SCG, FuncsCosts, &F);
      auto &FWD = FWDWorkList.back();
      EntryFuncs.insert(&F);
      SeenFunctions.insert(FWD.F);
      SeenFunctions.insert(FWD.Dependencies.begin(), FWD.Dependencies.end());
    }
  }

  // Sort the worklist so the most expensive roots are seen first.
  sort(FWDWorkList, [&](auto &A, auto &B) {
    // Sort by total cost, and if the total cost is identical, sort
    // alphabetically
    if (A.TotalCost == B.TotalCost)
      return A.F->getName() < B.F->getName();
    return A.TotalCost > B.TotalCost;
  });

  LLVM_DEBUG(dbgs() << "Number of callgraphs to be allocated: "
                    << FWDWorkList.size() << "   Module cost: "
                    << ModuleCost << "\n");
  LLVM_DEBUG(dbgs() << "callgraphs: \n");
  for (auto FWD : FWDWorkList) {
    LLVM_DEBUG(dbgs() << "[root] " << FWD.F->getName() << " (totalCost:"
                      << FWD.TotalCost << ";   root function cost: "
                      << FuncsCosts[FWD.F] << ";   has dependency: "
                      << FWD.Dependencies.size() << "\n");
  }
}

void SplitModuleCG::SplitModule(ModuleCreationCallback ModuleCallback,
                                const llvm::lto::Config &C) {
  for (Function &F : M) {
    if (F.hasLocalLinkage() && F.hasOneUse() && !F.hasAddressTaken())
      continue;
    externalize(&F);
    if (!F.isDeclaration() &&
        (F.hasExternalLinkage() || !F.isDefinitionExact()))
      externalFunction[&F] = true;
  }
  for (GlobalVariable &GV : M.globals())
    externalize(&GV);
  for (GlobalAlias &GA : M.aliases())
    externalize(&GA);
  for (GlobalIFunc &GI : M.ifuncs())
    externalize(&GI);

  // Assign callgraphs into N partitions.
  auto Partitions = doPartitioning();
  assert(Partitions.size() == N);

  const auto NeedsConservativeImport = [&](const GlobalValue *GV) {
    // Conservatively clone private/internal globals into every partition;
    // unused copies are removed by dealWithMpart afterwards.
    const auto *Var = dyn_cast<GlobalVariable>(GV);
    return Var && Var->hasLocalLinkage();
  };

  auto ShouldCloneDefinition = [&](unsigned I, const GlobalValue *GV) {
    const auto &FnsInPart = Partitions[I];

    // Functions go in their assigned partition.
    if (const auto *newFn = dyn_cast<Function>(GV)) {
      const auto *Fn = M.getFunction(newFn->getName());
      return FnsInPart.contains(Fn);
    }
    if (NeedsConservativeImport(GV))
      return true;
    // Everything else goes in the first partition.
    return I == 0;
  };

  // TODO: Consider parallelizing the per-partition CloneModule call itself.
  // Today the loop below serially clones M into N partitions in the main
  // thread, then spawns N worker threads to run opt+codegen. If CloneModule
  // becomes a bottleneck for large modules, the clones could be produced in
  // parallel too — but that would require either per-thread LLVMContexts
  // for the clone step or a thread-safe CloneModule, neither of which is
  // straightforward.
  std::vector<std::thread> Threads;
  Threads.reserve(N);
  std::vector<std::unique_ptr<Module>> MPartInCtxs;
  MPartInCtxs.resize(N);
  for (unsigned I = 0; I < N; ++I) {
    ValueToValueMapTy VMap;
    std::unique_ptr<Module> MPart(
      CloneModule(M, VMap, [&](const GlobalValue *GV) {
        return ShouldCloneDefinition(I, GV);
    }));

    dealWithMpart(*MPart, I, NeedsConservativeImport);

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
    Threads.emplace_back([&, I](SmallString<0> BC) {
      llvm::lto::LTOLLVMContext Ctx(C);
      Expected<std::unique_ptr<Module>> MOrErr = parseBitcodeFile(
          MemoryBufferRef(BC.str(), "ld-temp.o"), Ctx);
      BC = SmallString<0>();
      if (!MOrErr)
        report_fatal_error("Failed to read bitcode");
      ModuleCallback(std::move(MOrErr.get()), I);
    }, std::move(BC));
  }
  for (auto &T : Threads)
    T.join();
}

SplitModuleCG::SplitModuleCG(Module &M, unsigned LimitPartition)
    : N(LimitPartition), M(M), CG(M) {
  // Track existing non-local symbols. This ensures that when we promote
  // internal symbols to external for partitioning, we can handle renaming
  // and avoid conflicts.
  for (const auto &GV : M.global_values())
    if (!GV.hasLocalLinkage())
      OriginalExternals.insert(GV.getName());

  calculateFunctionCosts();

  // Construct a simplified call graph to facilitate worklist generation.
  SCG = std::make_unique<SimplifyCallGraph>(CG, M);

  // Populate the worklist with root functions and their transitive
  // dependencies. This worklist serves as the foundation for the
  // subsequent module partitioning.
  createWorkList();

  if (N == 0 || N > EntryFuncs.size()) {
    N = EntryFuncs.size();
  }
  N = N == 0 ? 1 : N;
}

void SimplifyCallGraph::createSimplifyCallGraph() {
  for (auto &NodePair : CG) {
    CallGraphNode *CGNode = NodePair.second.get();
    Function *F = CGNode->getFunction();
    if (!F || F->isDeclaration())
      continue;

    SimplifyCallGraphNode *SCGNode = getOrInsertFunction(F);

    for (const auto &CGNodeItem : *CGNode) {
      Function *Called = CGNodeItem.second->getFunction();
      if (!Called || Called->isDeclaration())
        continue;
      SCGNode->addCalledFunction(getOrInsertFunction(Called));
    }
  }

  if (enablePrintSimplifyCallGraph)
    print();
}


void SimplifyCallGraph::print() {
  for (auto &SCGItem : FunctionMap) {
    LLVM_DEBUG(dbgs() << "Call graph node for function: '"
                      << SCGItem.first->getName() << "' #uses="
                      << SCGItem.second->getNumReferences() << "\n");

    for (const auto &callee : *SCGItem.second) {
      LLVM_DEBUG(dbgs() <<"          Calls function : '"
                        << callee->getFunction()->getName() << " '\n");
    }
  }
}

SimplifyCallGraphNode *
SimplifyCallGraph::getOrInsertFunction(const Function *F) {
  auto &SCGN = FunctionMap[F];
  if (SCGN)
    return SCGN.get();

  SCGN =
      std::make_unique<SimplifyCallGraphNode>(this, const_cast<Function *>(F));
  return SCGN.get();
}
