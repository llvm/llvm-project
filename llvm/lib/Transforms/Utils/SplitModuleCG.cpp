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

#define DEBUG_TYPE "split-module-cg"

namespace {

static cl::opt<bool>
    enablePrintSimplifiedCallGraph("enable-print-simplified-callgraph",
                                   cl::Hidden, cl::init(false),
                                   cl::desc("print SimplifiedCallGraph"));

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

} // namespace

std::vector<DenseSet<const Function *>> SplitModuleCG::doPartitioning() {
  LLVM_DEBUG(dbgs() << "\n--Partitioning Starts--\n");
  // Performs all of the partitioning work on M.
  assert(N != 0 && "Partition count must be at least 1");
  std::vector<DenseSet<const Function *>> Partitions;
  Partitions.resize(N);

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
  for (unsigned I = 0; I < N; ++I)
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

    // Update the balancing queue. We scan backwards because in the common
    // case the target partition is at the end of the sorted queue.
    for (auto &[QueuePID, Cost] : reverse(BalancingQueue)) {
      if (QueuePID != PID)
        continue;
      Cost += AddedCost;
    }

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
    assert((ModuleCost + FnCost) >= ModuleCost && "Overflow!");
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
    // its externalFunction status.
    auto *OrigFn = M.getFunction(PartFunc.getName());
    if (!externalFunction.contains(OrigFn))
      continue;
    if (!externalFunction[OrigFn]) {
      PartFunc.setLinkage(GlobalValue::AvailableExternallyLinkage);
      PartFunc.setComdat(nullptr);
    } else {
      externalFunction[OrigFn] = false;
    }
  }

  // Rename GlobalValues whose linkage was promoted from local to external,
  // to avoid duplicate symbols across partitions in ThinLTO. Use the naming
  // convention "name.llvm.<suffix>" so the promoted local cannot clash with
  // an external that happens to share the same name. The suffix is derived
  // from the module via getUniqueModuleId, so it is consistent across all
  // partitions.
  std::string Suffix = getUniqueModuleId(&M);
  for (auto &GV : MPart.global_values()) {
    // Only rename symbols that were promoted from local to external: skip
    // those that are still local, and those that were already external in
    // the source module (recorded in OriginalExternals).
    if (GV.hasLocalLinkage() || OriginalExternals.contains(GV.getName()))
      continue;
    // Skip declarations of functions that were not explicitly externalized
    // (e.g. skipped by the hasOneUse check). Their definitions in other
    // partitions remain internal and are not renamed, so declarations must
    // keep the original name to stay consistent.
    auto *Fn = dyn_cast<Function>(&GV);
    if (Fn && Fn->isDeclaration() &&
        !externalFunction.contains(M.getFunction(Fn->getName())))
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

void SplitModuleCG::createWorkList() {
  // First, find all the entry functions with an in-degree of 0
  // (i.e., those that are not called by any function).
  for (auto &SCGNode : SCG->values()) {
    Function *F = SCGNode->getFunction();
    if (F && SCGNode->getNumReferences() == 0)
      EntryFuncs.insert(F);
  }

  // Second, find all the dependencies of each entry function.
  for (auto *F : EntryFuncs) {
    FWDWorkList.emplace_back(*SCG, FuncsCosts, F);
  }

  // Third, find all the functions that are not in the worklist.
  DenseSet<const Function *> SeenFunctions;
  for (const auto &Fwd : FWDWorkList) {
    SeenFunctions.insert(Fwd.Dependencies.begin(), Fwd.Dependencies.end());
  }
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

  // Sort the worklist so the most expensive roots are seen first.
  sort(FWDWorkList, [&](auto &A, auto &B) {
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
  for (auto Fwd : FWDWorkList)
    LLVM_DEBUG(dbgs() << "[root] " << Fwd.F->getName()
                      << " (totalCost:" << Fwd.TotalCost
                      << ";   root function cost: " << FuncsCosts[Fwd.F]
                      << ";   has dependency: " << Fwd.Dependencies.size()
                      << "\n");
#endif
}

void SplitModuleCG::SplitModule(ModuleCreationCallback ModuleCallback,
                                const llvm::lto::Config &C) {
  for (Function &F : M) {
    if (F.hasLocalLinkage() && F.hasOneUse() && !F.hasAddressTaken())
      continue;
    externalize(&F);
    // Record functions that may be defined in multiple partitions so that
    // dealWithMpart can downgrade duplicates to available_externally.
    if (canDowngradeToAvailableExternally(F))
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

  auto ShouldCloneDefinition = [&](unsigned I, const GlobalValue *GV) {
    const auto &FnsInPart = Partitions[I];

    // Functions go in their assigned partition.
    if (const auto *FnToClone = dyn_cast<Function>(GV))
      return FnsInPart.contains(FnToClone);
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
  for (unsigned I = 0; I < N; ++I) {
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
    Threads.emplace_back(
        [&, I](SmallString<0> BC) {
          llvm::lto::LTOLLVMContext Ctx(C);
          Expected<std::unique_ptr<Module>> MOrErr =
              parseBitcodeFile(MemoryBufferRef(BC.str(), "ld-temp.o"), Ctx);
          BC = SmallString<0>();
          if (!MOrErr)
            report_fatal_error("Failed to read bitcode");
          ModuleCallback(std::move(MOrErr.get()), I);
        },
        std::move(BC));
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
  SCG = std::make_unique<SimplifiedCallGraph>(CG);

  // Populate the worklist with root functions and their transitive
  // dependencies. This worklist serves as the foundation for the
  // subsequent module partitioning.
  createWorkList();

  if (N == 0 || N > EntryFuncs.size())
    N = EntryFuncs.size();
  if (N == 0)
    N = 1;
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

  if (enablePrintSimplifiedCallGraph)
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
