//===-------- SYCLSplitModule.cpp - Split a module into call graphs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SYCLSplitModule.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SYCLUtils.h"

#include <map>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "sycl-split-module"

static bool isKernel(const Function &F) {
  return F.getCallingConv() == CallingConv::SPIR_KERNEL ||
         F.getCallingConv() == CallingConv::AMDGPU_KERNEL;
}

static bool isEntryPoint(const Function &F) {
  // Skip declarations, if any: they should not be included into a vector of
  // entry points groups or otherwise we will end up with incorrectly generated
  // list of symbols.
  if (F.isDeclaration())
    return false;

  // Kernels are always considered to be entry points
  return isKernel(F);
}

namespace {

// A vector that contains all entry point functions in a split module.
using EntryPointSet = SetVector<const Function *>;

/// Represents a named group entry points.
struct EntryPointGroup {
  std::string GroupName;
  EntryPointSet Functions;

  EntryPointGroup() = default;
  EntryPointGroup(const EntryPointGroup &) = default;
  EntryPointGroup &operator=(const EntryPointGroup &) = default;
  EntryPointGroup(EntryPointGroup &&) = default;
  EntryPointGroup &operator=(EntryPointGroup &&) = default;

  EntryPointGroup(StringRef GroupName,
                  EntryPointSet Functions = EntryPointSet())
      : GroupName(GroupName), Functions(std::move(Functions)) {}

  void clear() {
    GroupName.clear();
    Functions.clear();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const {
    constexpr size_t INDENT = 4;
    dbgs().indent(INDENT) << "ENTRY POINTS"
                          << " " << GroupName << " {\n";
    for (const Function *F : Functions)
      dbgs().indent(INDENT) << "  " << F->getName() << "\n";

    dbgs().indent(INDENT) << "}\n";
  }
#endif
};

/// Annotates an llvm::Module with information necessary to perform and track
/// the result of device code (llvm::Module instances) splitting:
/// - entry points group from the module.
class ModuleDesc {
  std::unique_ptr<Module> M;
  EntryPointGroup EntryPoints;

public:
  ModuleDesc() = delete;
  ModuleDesc(const ModuleDesc &) = delete;
  ModuleDesc &operator=(const ModuleDesc &) = delete;
  ModuleDesc(ModuleDesc &&) = default;
  ModuleDesc &operator=(ModuleDesc &&) = default;

  ModuleDesc(std::unique_ptr<Module> M,
             EntryPointGroup EntryPoints = EntryPointGroup())
      : M(std::move(M)), EntryPoints(std::move(EntryPoints)) {
    assert(this->M && "Module should be non-null");
  }

  Module &getModule() { return *M; }
  const Module &getModule() const { return *M; }

  std::unique_ptr<Module> releaseModule() {
    EntryPoints.clear();
    return std::move(M);
  }

  std::string makeSymbolTable() const {
    SmallString<0> Data;
    raw_svector_ostream OS(Data);
    for (const Function *F : EntryPoints.Functions)
      OS << F->getName() << '\n';

    return std::string(OS.str());
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const {
    dbgs() << "ModuleDesc[" << M->getName() << "] {\n";
    EntryPoints.dump();
    dbgs() << "}\n";
  }
#endif
};

// Represents "dependency" or "use" graph of global objects (functions and
// global variables) in a module. It is used during device code split to
// understand which global variables and functions (other than entry points)
// should be included into a split module.
//
// Nodes of the graph represent LLVM's GlobalObjects, edges "A" -> "B" represent
// the fact that if "A" is included into a module, then "B" should be included
// as well.
//
// Examples of dependencies which are represented in this graph:
// - Function FA calls function FB
// - Function FA uses global variable GA
// - Global variable GA references (initialized with) function FB
// - Function FA stores address of a function FB somewhere
//
// The following cases are treated as dependencies between global objects:
// 1. Global object A is used within by a global object B in any way (store,
//    bitcast, phi node, call, etc.): "A" -> "B" edge will be added to the
//    graph;
// 2. function A performs an indirect call of a function with signature S and
//    there is a function B with signature S. "A" -> "B" edge will be added to
//    the graph;
class DependencyGraph {
public:
  using GlobalSet = SmallPtrSet<const GlobalValue *, 16>;

  DependencyGraph(const Module &M) {
    // Group functions by their signature to handle case (2) described above
    DenseMap<const FunctionType *, DependencyGraph::GlobalSet>
        FuncTypeToFuncsMap;
    for (const auto &F : M.functions()) {
      // Kernels can't be called (either directly or indirectly) in SYCL
      if (isKernel(F))
        continue;

      FuncTypeToFuncsMap[F.getFunctionType()].insert(&F);
    }

    for (const auto &F : M.functions()) {
      // case (1), see comment above the class definition
      for (const Value *U : F.users())
        addUserToGraphRecursively(cast<const User>(U), &F);

      // case (2), see comment above the class definition
      for (const auto &I : instructions(F)) {
        const auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->isIndirectCall()) // Direct calls were handled above
          continue;

        const FunctionType *Signature = CI->getFunctionType();
        const auto &PotentialCallees = FuncTypeToFuncsMap[Signature];
        Graph[&F].insert(PotentialCallees.begin(), PotentialCallees.end());
      }
    }

    // And every global variable (but their handling is a bit simpler)
    for (const auto &GV : M.globals())
      for (const Value *U : GV.users())
        addUserToGraphRecursively(cast<const User>(U), &GV);
  }

  iterator_range<GlobalSet::const_iterator>
  dependencies(const GlobalValue *Val) const {
    auto It = Graph.find(Val);
    return (It == Graph.end())
               ? make_range(EmptySet.begin(), EmptySet.end())
               : make_range(It->second.begin(), It->second.end());
  }

private:
  void addUserToGraphRecursively(const User *Root, const GlobalValue *V) {
    SmallVector<const User *, 8> WorkList;
    WorkList.push_back(Root);

    while (!WorkList.empty()) {
      const User *U = WorkList.pop_back_val();
      if (const auto *I = dyn_cast<const Instruction>(U)) {
        const auto *UFunc = I->getFunction();
        Graph[UFunc].insert(V);
      } else if (isa<const Constant>(U)) {
        if (const auto *GV = dyn_cast<const GlobalVariable>(U))
          Graph[GV].insert(V);
        // This could be a global variable or some constant expression (like
        // bitcast or gep). We trace users of this constant further to reach
        // global objects they are used by and add them to the graph.
        for (const auto *UU : U->users())
          WorkList.push_back(UU);
      } else
        llvm_unreachable("Unhandled type of function user");
    }
  }

  DenseMap<const GlobalValue *, GlobalSet> Graph;
  SmallPtrSet<const GlobalValue *, 1> EmptySet;
};

void collectFunctionsAndGlobalVariablesToExtract(
    SetVector<const GlobalValue *> &GVs, const Module &M,
    const EntryPointGroup &ModuleEntryPoints, const DependencyGraph &DG) {
  // We start with module entry points
  for (const auto *F : ModuleEntryPoints.Functions)
    GVs.insert(F);

  // Non-discardable global variables are also include into the initial set
  for (const auto &GV : M.globals())
    if (!GV.isDiscardableIfUnused())
      GVs.insert(&GV);

  // GVs has SetVector type. This type inserts a value only if it is not yet
  // present there. So, recursion is not expected here.
  size_t Idx = 0;
  while (Idx < GVs.size()) {
    const GlobalValue *Obj = GVs[Idx++];

    for (const GlobalValue *Dep : DG.dependencies(Obj)) {
      if (const auto *Func = dyn_cast<const Function>(Dep)) {
        if (!Func->isDeclaration())
          GVs.insert(Func);
      } else
        GVs.insert(Dep); // Global variables are added unconditionally
    }
  }
}

ModuleDesc extractSubModule(const Module &M,
                            const SetVector<const GlobalValue *> &GVs,
                            EntryPointGroup ModuleEntryPoints) {
  // For each group of entry points collect all dependencies.
  ValueToValueMapTy VMap;
  // Clone definitions only for needed globals. Others will be added as
  // declarations and removed later.
  std::unique_ptr<Module> SubM = CloneModule(
      M, VMap, [&](const GlobalValue *GV) { return GVs.count(GV); });
  // Replace entry points with cloned ones.
  EntryPointSet NewEPs;
  const EntryPointSet &EPs = ModuleEntryPoints.Functions;
  std::for_each(EPs.begin(), EPs.end(), [&](const Function *F) {
    NewEPs.insert(cast<Function>(VMap[F]));
  });
  ModuleEntryPoints.Functions = std::move(NewEPs);
  return ModuleDesc{std::move(SubM), std::move(ModuleEntryPoints)};
}

// The function produces a copy of input LLVM IR module M with only those
// functions and globals that can be called from entry points that are specified
// in ModuleEntryPoints vector, in addition to the entry point functions.
ModuleDesc extractCallGraph(const Module &M, EntryPointGroup ModuleEntryPoints,
                            const DependencyGraph &DG) {
  SetVector<const GlobalValue *> GVs;
  collectFunctionsAndGlobalVariablesToExtract(GVs, M, ModuleEntryPoints, DG);

  ModuleDesc SplitM = extractSubModule(M, GVs, std::move(ModuleEntryPoints));
  LLVM_DEBUG(SplitM.dump());
  return SplitM;
}

using EntryPointGroupVec = SmallVector<EntryPointGroup, 0>;

/// Module Splitter.
/// It gets a module (in a form of module descriptor, to get additional info)
/// and a collection of entry points groups. Each group specifies subset entry
/// points from input module that should be included in a split module.
class ModuleSplitter {
private:
  ModuleDesc Input;
  EntryPointGroupVec Groups;
  DependencyGraph DG;

private:
  EntryPointGroup drawEntryPointGroup() {
    assert(Groups.size() > 0 && "Reached end of entry point groups list.");
    EntryPointGroup Group = std::move(Groups.back());
    Groups.pop_back();
    return Group;
  }

public:
  ModuleSplitter(ModuleDesc MD, EntryPointGroupVec GroupVec)
      : Input(std::move(MD)), Groups(std::move(GroupVec)),
        DG(Input.getModule()) {
    assert(!Groups.empty() && "Entry points groups collection is empty!");
  }

  /// Gets next subsequence of entry points in an input module and provides
  /// split submodule containing these entry points and their dependencies.
  ModuleDesc getNextSplit() {
    return extractCallGraph(Input.getModule(), drawEntryPointGroup(), DG);
  }

  /// Check that there are still submodules to split.
  bool hasMoreSplits() const { return Groups.size() > 0; }
};

} // namespace

static EntryPointGroupVec selectEntryPointGroups(const Module &M,
                                                 IRSplitMode Mode) {
  // std::map is used here to ensure stable ordering of entry point groups,
  // which is based on their contents, this greatly helps LIT tests
  std::map<std::string, EntryPointSet> EntryPointsMap;

  static constexpr char ATTR_SYCL_MODULE_ID[] = "sycl-module-id";
  for (const auto &F : M.functions()) {
    if (!isEntryPoint(F))
      continue;

    std::string Key;
    switch (Mode) {
    case IRSplitMode::IRSM_PER_KERNEL:
      Key = F.getName();
      break;
    case IRSplitMode::IRSM_PER_TU:
      Key = F.getFnAttribute(ATTR_SYCL_MODULE_ID).getValueAsString();
      break;
    case IRSplitMode::IRSM_NONE:
      llvm_unreachable("");
    }

    EntryPointsMap[Key].insert(&F);
  }

  EntryPointGroupVec Groups;
  if (EntryPointsMap.empty()) {
    // No entry points met, record this.
    Groups.emplace_back("-", EntryPointSet());
  } else {
    Groups.reserve(EntryPointsMap.size());
    // Start with properties of a source module
    for (auto &[Key, EntryPoints] : EntryPointsMap)
      Groups.emplace_back(Key, std::move(EntryPoints));
  }

  return Groups;
}

namespace llvm {

std::optional<IRSplitMode> convertStringToSplitMode(StringRef S) {
  static const StringMap<IRSplitMode> Values = {
      {"source", IRSplitMode::IRSM_PER_TU},
      {"kernel", IRSplitMode::IRSM_PER_KERNEL},
      {"none", IRSplitMode::IRSM_NONE}};

  auto It = Values.find(S);
  if (It == Values.end())
    return std::nullopt;

  return It->second;
}

void SYCLSplitModule(std::unique_ptr<Module> M, IRSplitMode Mode,
                     PostSYCLSplitCallbackType Callback) {
  SmallVector<ModuleAndSYCLMetadata, 0> OutputImages;
  if (Mode == IRSplitMode::IRSM_NONE) {
    auto MD = ModuleDesc(std::move(M));
    auto Symbols = MD.makeSymbolTable();
    Callback(std::move(MD.releaseModule()), std::move(Symbols));
    return;
  }

  EntryPointGroupVec Groups = selectEntryPointGroups(*M, Mode);
  ModuleDesc MD = std::move(M);
  ModuleSplitter Splitter(std::move(MD), std::move(Groups));
  while (Splitter.hasMoreSplits()) {
    ModuleDesc MD = Splitter.getNextSplit();
    auto Symbols = MD.makeSymbolTable();
    Callback(std::move(MD.releaseModule()), std::move(Symbols));
  }
}

} // namespace llvm
