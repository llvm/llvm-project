#include "llvm/Transforms/Utils/SplitModuleCG.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/MD5.h"
#include "llvm/Transforms/Utils/Cloning.h"
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
    auto &FnsInPart = Partitions[PID];
    FnsInPart.insert(FWD.F);
    for (const Function *Dep : FWD.Dependencies) {
      FnsInPart.insert(Dep);
    }

    // Update the balancing queue. we scan backwards because in the common case
    // the partition is at the end.
    for (auto &[QueuePID, Cost] : reverse(BalancingQueue)) {
      if (QueuePID == PID) {
        CostType NewCost = 0;
        for (auto *Fn : Partitions[PID])
          NewCost += FuncsCosts.at(Fn);
        Cost = NewCost;
      }
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
  // collect symbols to rename
  auto checkPromoted = [&](const GlobalValue &GV) {
    // now is external (not local), but not in external set.
    if (!GV.hasLocalLinkage() && !OriginalExternals.contains(GV.getName())) {
      if (PromotedRenames.count(GV.getName()))
        return;
      MD5 Hash;
      Hash.update(M.getModuleIdentifier());
      MD5::MD5Result Result;
      Hash.final(Result);
      SmallString<32> HashStr;
      MD5::stringifyResult(Result, HashStr);
      std::string NewName = (GV.getName() + "." + HashStr.str().substr(0, 8)).str();
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

  // TODO: Consider optimizing the alias, replacing the determined alias with
  // the determined aliasee.

  // Assign callgraphs into N partitions.
  auto Partitions = doPartitioning();
  assert(Partitions.size() == N);

  // local GVs need to be conservatively imported into [dependency] every module,
 	// and then cleaned up afterwards.
  const auto NeedsConservativeImport = [&](const GlobalValue *GV) {
    // We conservatively import private/internal GVs into every module and clean
    // them up afterwards.
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

  // TODO: In the future, it may be considered to also include clonemodule in
  // parallel to reduce compilation time.
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

    // If not clone module in multi-thread, we also need to clone
    // the module obtained through segmentation into a new context
    // to avoid data races.
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

SplitModuleCG::SplitModuleCG(Module &M,
                             const ModuleSummaryIndex &CombinedIndex,
                             unsigned LimitPartition)
    : M(M), CG(M), N(LimitPartition) {
  // Track existing non-local symbols. This ensures that when we promote
  // internal symbols to external for partitioning, we can handle renaming
  // and avoid conflicts.
  for (const auto &GV : M.global_values())
    if (!GV.hasLocalLinkage())
      OriginalExternals.insert(GV.getName());

  calculateFunctionCosts();

  // Construct a simplified call graph to facilitate worklist generation.
  SCG = std::make_unique<SimplifyCallGraph>(CG, CombinedIndex, M);
  // TODO: When the SCG is established, the special cases of comdat and
  // initarray need to be considered.

  // Populate the worklist with root functions and their transitive
  // dependencies. This worklist serves as the foundation for the
  // subsequent module partitioning.
  createWorkList();

  if (N == 0 || N > EntryFuncs.size()) {
    N = EntryFuncs.size();
  }
  N = N == 0 ? 1 : N;
}

void SimplifyCallGraph::createSimplifyCallGraph(
    const ModuleSummaryIndex &CombinedIndex) {
  for (auto &NodePair : CG) {
    CallGraphNode *CGNode = NodePair.second.get();
    Function *F = CGNode->getFunction();
    if (!F || F->isDeclaration())
      continue;

    SimplifyCallGraphNode *SCGNode = getOrInsertFunction(F);

    //TODO: Trace indirect call usage for the current function.

    for (const auto &CGNodeItem : *CGNode) {
      Function *Called = CGNodeItem.second->getFunction();
      if (!Called) {
        //TODO: Deal with indirect call. 
        // 1. Check if the instruction has a callees metadata.
        // 2. Check if this is an indirect call with profile data.
        // 3. Check if this is an alias to a function.
      }
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
