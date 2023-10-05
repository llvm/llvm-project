//===-- GlobalDCE.cpp - DCE unreachable internal functions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform is designed to eliminate unreachable internal globals from the
// program.  It uses an aggressive algorithm, searching out globals that are
// known to be alive.  After it finds all of the globals which are needed, it
// deletes whatever is left over.  This allows it to delete recursive chunks of
// the program which are unreachable.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TypeMetadataUtils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"

#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

#define DEBUG_TYPE "globaldce"

static cl::opt<bool>
    ClEnableVFE("enable-vfe", cl::Hidden, cl::init(true),
                cl::desc("Enable virtual function elimination"));

static cl::opt<std::string> ClReadSummary(
    "globaldce-read-summary",
    cl::desc("Read summary from given bitcode before running pass"),
    cl::Hidden);

STATISTIC(NumAliases  , "Number of global aliases removed");
STATISTIC(NumFunctions, "Number of functions removed");
STATISTIC(NumIFuncs,    "Number of indirect functions removed");
STATISTIC(NumVariables, "Number of global variables removed");
STATISTIC(NumVFuncs,    "Number of virtual functions removed");

namespace llvm {

// Returning a representative summary for the vtable, also set isSafe.
static const GlobalVarSummary *
getVTableFuncsForTId(const TypeIdOffsetVtableInfo &P, bool &isSafe) {
  // Find a representative copy of the vtable initializer.
  const GlobalVarSummary *VS = nullptr;
  bool LocalFound = false;
  for (auto &S : P.VTableVI.getSummaryList()) {
    if (GlobalValue::isLocalLinkage(S->linkage())) {
      if (LocalFound) {
        isSafe = false;
        return nullptr;
      }
      LocalFound = true;
    }
    auto *CurVS = cast<GlobalVarSummary>(S->getBaseObject());
    // Ignore if vTableFuncs is empty and vtable is available_externally.
    if (!CurVS->vTableFuncs().empty() ||
        !GlobalValue::isAvailableExternallyLinkage(S->linkage())) {
      VS = CurVS;
      if (VS->getVCallVisibility() == GlobalObject::VCallVisibilityPublic) {
        isSafe = false;
        return VS;
      }
    }
  }

  if (!VS) {
    isSafe = false;
    return nullptr;
  }
  if (!VS->isLive()) {
    isSafe = true;
    return nullptr;
  }
  isSafe = true;
  return VS;
}

static void collectSafeVTables(
    ModuleSummaryIndex &Summary,
    DenseMap<GlobalValue::GUID, std::vector<StringRef>> &NameByGUID,
    std::map<ValueInfo, std::vector<VirtFuncOffset>> &VFESafeVTablesAndFns) {
  // Update VFESafeVTablesAndFns with information from summary.
  for (auto &P : Summary.typeIdCompatibleVtableMap()) {
    NameByGUID[GlobalValue::getGUID(P.first)].push_back(P.first);
    LLVM_DEBUG(dbgs() << "TId " << GlobalValue::getGUID(P.first) << " "
                      << P.first << "\n");
  }
  llvm::errs() << "VFEThinLTO number of TIds: " << NameByGUID.size() << "\n";

  // VFESafeVTablesAndFns: map from VI for vTable to VI for vfunc
  std::map<ValueInfo, std::set<GlobalValue::GUID>> vFuncSet;
  unsigned numSafeVFuncs = 0;
  // Collect stats for VTables (safe, public-visibility, other).
  std::set<ValueInfo> vTablePublicVis;
  std::set<ValueInfo> vTableOther;
  for (auto &TidSummary : Summary.typeIdCompatibleVtableMap()) {
    for (const TypeIdOffsetVtableInfo &P : TidSummary.second) {
      LLVM_DEBUG(dbgs() << "TId-vTable " << TidSummary.first << " "
                        << P.VTableVI.name() << " " << P.AddressPointOffset
                        << "\n");
      bool isSafe = false;
      const GlobalVarSummary *VS = getVTableFuncsForTId(P, isSafe);
      if (!isSafe && VS)
        vTablePublicVis.insert(P.VTableVI);
      if ((isSafe && !VS) || (!isSafe && !VS))
        vTableOther.insert(P.VTableVI);
      if (!isSafe || !VS) {
        continue;
      }

      // Go through VS->vTableFuncs
      for (auto VTP : VS->vTableFuncs()) {
        if (vFuncSet.find(P.VTableVI) == vFuncSet.end() ||
            !vFuncSet[P.VTableVI].count(VTP.FuncVI.getGUID())) {
          VFESafeVTablesAndFns[P.VTableVI].push_back(VTP);
          LLVM_DEBUG(dbgs()
                     << "vTable " << P.VTableVI.name() << " "
                     << VTP.FuncVI.name() << " " << VTP.VTableOffset << "\n");
          ++numSafeVFuncs;
        }
        vFuncSet[P.VTableVI].insert(VTP.FuncVI.getGUID());
      }
    }
  }
  llvm::errs() << "VFEThinLTO number of vTables: " << vFuncSet.size() << " "
               << vTablePublicVis.size() << " " << vTableOther.size() << "\n";
  llvm::errs() << "VFEThinLTO numSafeVFuncs: " << numSafeVFuncs << "\n";
}

static void checkVTableLoadsIndex(
    ModuleSummaryIndex &Summary,
    DenseMap<GlobalValue::GUID, std::vector<StringRef>> &NameByGUID,
    std::map<ValueInfo, std::vector<VirtFuncOffset>> &VFESafeVTablesAndFns,
    std::map<ValueInfo, std::vector<ValueInfo>> &VFuncsAndCallers) {
  // Go through Function summarys for intrinsics, also funcHasNonVTableRef to
  // erase entries from VFESafeVTableAndFns.
  for (auto &PI : Summary) {
    for (auto &S : PI.second.SummaryList) {
      auto *FS = dyn_cast<FunctionSummary>(S.get());
      if (!FS)
        continue;
      // We should ignore Tid if there is a type.checked.load with Offset not
      // ConstantInt. Currently ModuleSummaryAnalysis will update TypeTests.
      for (GlobalValue::GUID G : FS->type_tests()) {
        if (NameByGUID.find(G) == NameByGUID.end())
          continue;
        auto TidSummary =
            Summary.getTypeIdCompatibleVtableSummary(NameByGUID[G][0]);
        for (const TypeIdOffsetVtableInfo &P : *TidSummary) {
          LLVM_DEBUG(dbgs() << "unsafe-vtable due to type_tests: "
                            << P.VTableVI.name() << "\n");
          VFESafeVTablesAndFns.erase(P.VTableVI);
        }
      }
      // Go through vTableFuncs to find the potential callees
      auto CheckVLoad = [&](GlobalValue::GUID TId, uint64_t Offset) {
        if (!NameByGUID.count(TId)) {
          return;
        }
        auto TidSummary =
            Summary.getTypeIdCompatibleVtableSummary(NameByGUID[TId][0]);
        for (const TypeIdOffsetVtableInfo &P : *TidSummary) {
          uint64_t VTableOffset = P.AddressPointOffset;
          bool isSafe = false;
          const GlobalVarSummary *VS = getVTableFuncsForTId(P, isSafe);
          if (!isSafe || !VS)
            continue;
          unsigned foundCnt = 0;
          for (auto VTP : VS->vTableFuncs()) {
            if (VTP.VTableOffset != VTableOffset + Offset)
              continue;
            assert(foundCnt == 0);
            foundCnt = 1;
            VFuncsAndCallers[VTP.FuncVI].push_back(Summary.getValueInfo(PI));
          }
          if (foundCnt == 0) {
            // A vtable is unsafe if a given vtable for the typeId doesn't have
            // the offset.
            VFESafeVTablesAndFns.erase(P.VTableVI);
            LLVM_DEBUG(dbgs() << "unsafe-vtable can't find offset "
                              << P.VTableVI.name() << " "
                              << (VTableOffset + Offset) << "\n");
          }
        }
      };
      for (FunctionSummary::VFuncId VF : FS->type_checked_load_vcalls()) {
        CheckVLoad(VF.GUID, VF.Offset);
      }
      for (const FunctionSummary::ConstVCall &VC :
           FS->type_checked_load_const_vcalls()) {
        CheckVLoad(VC.VFunc.GUID, VC.VFunc.Offset);
      }
    }
  }
}

void runVFEOnIndex(ModuleSummaryIndex &Summary,
                   function_ref<bool(GlobalValue::GUID)> isRetained) {
  if (Summary.typeIdCompatibleVtableMap().empty())
    return;

  DenseMap<GlobalValue::GUID, std::vector<StringRef>> NameByGUID;
  std::map<ValueInfo, std::vector<VirtFuncOffset>> VFESafeVTablesAndFns;
  collectSafeVTables(Summary, NameByGUID, VFESafeVTablesAndFns);

  // Go through Function summarys for intrinsics, also funcHasNonVTableRef to
  // erase entries from VFESafeVTableAndFns.
  std::map<ValueInfo, std::vector<ValueInfo>> VFuncsAndCallers;
  checkVTableLoadsIndex(Summary, NameByGUID, VFESafeVTablesAndFns,
                        VFuncsAndCallers);
  llvm::errs() << "VFEThinLTO number of vTables: "
               << VFESafeVTablesAndFns.size() << "\n";

  // Generate list of vfuncs that can be removed.
  std::set<ValueInfo> candidateSet;
  for (auto entry : VFESafeVTablesAndFns) {
    for (auto vfunc : entry.second) {
      if (Summary.isFuncWithNonVtableRef(vfunc.FuncVI.getGUID())) {
        LLVM_DEBUG(dbgs() << "unsafe-vfunc with non-vtable ref "
                          << vfunc.FuncVI.name() << "\n");
        continue;
      }
      if (!candidateSet.count(vfunc.FuncVI)) {
        candidateSet.insert(vfunc.FuncVI);
      }
    }
  }

  // It is possible to use a workList to recursively mark vfuncs as removable.
  // For now, only remove vfuncs that are not in VFuncsAndCallers.
  std::set<ValueInfo> removable;
  for (auto vfunc : candidateSet) {
    if (!VFuncsAndCallers.count(vfunc) && !isRetained(vfunc.getGUID()))
      removable.insert(vfunc);
  }

  for (auto fVI : removable) {
    Summary.addFuncToBeRemoved(fVI.getGUID());
  }
  llvm::errs() << "VFEThinLTO removable " << removable.size() << "\n";
}
} // end namespace llvm

/// Returns true if F is effectively empty.
static bool isEmptyFunction(Function *F) {
  // Skip external functions.
  if (F->isDeclaration())
    return false;
  BasicBlock &Entry = F->getEntryBlock();
  for (auto &I : Entry) {
    if (I.isDebugOrPseudoInst())
      continue;
    if (auto *RI = dyn_cast<ReturnInst>(&I))
      return !RI->getReturnValue();
    break;
  }
  return false;
}

/// Compute the set of GlobalValue that depends from V.
/// The recursion stops as soon as a GlobalValue is met.
void GlobalDCEPass::ComputeDependencies(Value *V,
                                        SmallPtrSetImpl<GlobalValue *> &Deps) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    Function *Parent = I->getParent()->getParent();
    Deps.insert(Parent);
  } else if (auto *GV = dyn_cast<GlobalValue>(V)) {
    Deps.insert(GV);
  } else if (auto *CE = dyn_cast<Constant>(V)) {
    // Avoid walking the whole tree of a big ConstantExprs multiple times.
    auto Where = ConstantDependenciesCache.find(CE);
    if (Where != ConstantDependenciesCache.end()) {
      auto const &K = Where->second;
      Deps.insert(K.begin(), K.end());
    } else {
      SmallPtrSetImpl<GlobalValue *> &LocalDeps = ConstantDependenciesCache[CE];
      for (User *CEUser : CE->users())
        ComputeDependencies(CEUser, LocalDeps);
      Deps.insert(LocalDeps.begin(), LocalDeps.end());
    }
  }
}

void GlobalDCEPass::UpdateGVDependencies(GlobalValue &GV) {
  SmallPtrSet<GlobalValue *, 8> Deps;
  for (User *User : GV.users())
    ComputeDependencies(User, Deps);
  Deps.erase(&GV); // Remove self-reference.
  for (GlobalValue *GVU : Deps) {
    // If this is a dep from a vtable to a virtual function, and we have
    // complete information about all virtual call sites which could call
    // though this vtable, then skip it, because the call site information will
    // be more precise.
    if (VFESafeVTables.count(GVU) && isa<Function>(&GV)) {
      LLVM_DEBUG(dbgs() << "Ignoring dep " << GVU->getName() << " -> "
                        << GV.getName() << "\n");
      continue;
    }
    GVDependencies[GVU].insert(&GV);
  }
}

/// Mark Global value as Live
void GlobalDCEPass::MarkLive(GlobalValue &GV,
                             SmallVectorImpl<GlobalValue *> *Updates) {
  auto const Ret = AliveGlobals.insert(&GV);
  if (!Ret.second)
    return;

  if (Updates)
    Updates->push_back(&GV);
  if (Comdat *C = GV.getComdat()) {
    for (auto &&CM : make_range(ComdatMembers.equal_range(C))) {
      MarkLive(*CM.second, Updates); // Recursion depth is only two because only
                                     // globals in the same comdat are visited.
    }
  }
}

void GlobalDCEPass::ScanVTables(Module &M) {
  SmallVector<MDNode *, 2> Types;
  LLVM_DEBUG(dbgs() << "Building type info -> vtable map\n");

  for (GlobalVariable &GV : M.globals()) {
    Types.clear();
    GV.getMetadata(LLVMContext::MD_type, Types);
    if (GV.isDeclaration() || Types.empty())
      continue;

    // Use the typeid metadata on the vtable to build a mapping from typeids to
    // the list of (GV, offset) pairs which are the possible vtables for that
    // typeid.
    for (MDNode *Type : Types) {
      Metadata *TypeID = Type->getOperand(1).get();

      uint64_t Offset =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Type->getOperand(0))->getValue())
              ->getZExtValue();

      TypeIdMap[TypeID].insert(std::make_pair(&GV, Offset));
    }

    // If the type corresponding to the vtable is private to this translation
    // unit, we know that we can see all virtual functions which might use it,
    // so VFE is safe.
    if (auto GO = dyn_cast<GlobalObject>(&GV)) {
      GlobalObject::VCallVisibility TypeVis = GO->getVCallVisibility();
      if (TypeVis == GlobalObject::VCallVisibilityTranslationUnit ||
          (InLTOPostLink &&
           TypeVis == GlobalObject::VCallVisibilityLinkageUnit)) {
        LLVM_DEBUG(dbgs() << GV.getName() << " is safe for VFE\n");
        VFESafeVTables.insert(&GV);
      }
    }
  }
}

void GlobalDCEPass::ScanVTableLoad(Function *Caller, Metadata *TypeId,
                                   uint64_t CallOffset) {
  for (const auto &VTableInfo : TypeIdMap[TypeId]) {
    GlobalVariable *VTable = VTableInfo.first;
    uint64_t VTableOffset = VTableInfo.second;

    Constant *Ptr =
        getPointerAtOffset(VTable->getInitializer(), VTableOffset + CallOffset,
                           *Caller->getParent(), VTable);
    if (!Ptr) {
      LLVM_DEBUG(dbgs() << "can't find pointer in vtable!\n");
      VFESafeVTables.erase(VTable);
      continue;
    }

    auto Callee = dyn_cast<Function>(Ptr->stripPointerCasts());
    if (!Callee) {
      LLVM_DEBUG(dbgs() << "vtable entry is not function pointer!\n");
      VFESafeVTables.erase(VTable);
      continue;
    }

    LLVM_DEBUG(dbgs() << "vfunc dep " << Caller->getName() << " -> "
                      << Callee->getName() << "\n");
    GVDependencies[Caller].insert(Callee);
  }
}

void GlobalDCEPass::ScanTypeCheckedLoadIntrinsics(Module &M) {
  LLVM_DEBUG(dbgs() << "Scanning type.checked.load intrinsics\n");
  Function *TypeCheckedLoadFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::type_checked_load));
  Function *TypeCheckedLoadRelativeFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::type_checked_load_relative));

  auto scan = [&](Function *CheckedLoadFunc) {
    if (!CheckedLoadFunc)
      return;

    for (auto *U : CheckedLoadFunc->users()) {
      auto CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;

      auto *Offset = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      Value *TypeIdValue = CI->getArgOperand(2);
      auto *TypeId = cast<MetadataAsValue>(TypeIdValue)->getMetadata();

      if (Offset) {
        ScanVTableLoad(CI->getFunction(), TypeId, Offset->getZExtValue());
      } else {
        // type.checked.load with a non-constant offset, so assume every entry
        // in every matching vtable is used.
        for (const auto &VTableInfo : TypeIdMap[TypeId]) {
          VFESafeVTables.erase(VTableInfo.first);
        }
      }
    }
  };

  scan(TypeCheckedLoadFunc);
  scan(TypeCheckedLoadRelativeFunc);
}

void GlobalDCEPass::AddVirtualFunctionDependencies(Module &M) {
  if (ImportSummary)
    return; // only use results from Index with ThinLTO
  if (!ClEnableVFE)
    return;

  // If the Virtual Function Elim module flag is present and set to zero, then
  // the vcall_visibility metadata was inserted for another optimization (WPD)
  // and we may not have type checked loads on all accesses to the vtable.
  // Don't attempt VFE in that case.
  auto *Val = mdconst::dyn_extract_or_null<ConstantInt>(
      M.getModuleFlag("Virtual Function Elim"));
  if (!Val || Val->isZero())
    return;

  ScanVTables(M);

  if (VFESafeVTables.empty())
    return;

  ScanTypeCheckedLoadIntrinsics(M);

  LLVM_DEBUG(
    dbgs() << "VFE safe vtables:\n";
    for (auto *VTable : VFESafeVTables)
      dbgs() << "  " << VTable->getName() << "\n";
  );
}

PreservedAnalyses GlobalDCEPass::run(Module &M, ModuleAnalysisManager &MAM) {
  // Handle the command-line summary arguments. This code is for testing
  // purposes only, so we handle errors directly.
  std::unique_ptr<ModuleSummaryIndex> Summary =
      std::make_unique<ModuleSummaryIndex>(/*HaveGVs=*/false);
  if (!ClReadSummary.empty()) {
    ExitOnError ExitOnErr("-globaldce-read-summary: " + ClReadSummary + ": ");
    auto ReadSummaryFile =
        ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(ClReadSummary)));
    if (Expected<std::unique_ptr<ModuleSummaryIndex>> SummaryOrErr =
            getModuleSummaryIndex(*ReadSummaryFile)) {
      Summary = std::move(*SummaryOrErr);
    }
    ImportSummary = Summary.get();
    auto isRetained = [&](GlobalValue::GUID CalleeGUID) { return false; };
    runVFEOnIndex(*(const_cast<ModuleSummaryIndex *>(ImportSummary)),
                  isRetained);
  }
  bool Changed = false;

  // The algorithm first computes the set L of global variables that are
  // trivially live.  Then it walks the initialization of these variables to
  // compute the globals used to initialize them, which effectively builds a
  // directed graph where nodes are global variables, and an edge from A to B
  // means B is used to initialize A.  Finally, it propagates the liveness
  // information through the graph starting from the nodes in L. Nodes note
  // marked as alive are discarded.

  // Remove empty functions from the global ctors list.
  Changed |= optimizeGlobalCtorsList(
      M, [](uint32_t, Function *F) { return isEmptyFunction(F); });

  // Collect the set of members for each comdat.
  for (Function &F : M)
    if (Comdat *C = F.getComdat())
      ComdatMembers.insert(std::make_pair(C, &F));
  for (GlobalVariable &GV : M.globals())
    if (Comdat *C = GV.getComdat())
      ComdatMembers.insert(std::make_pair(C, &GV));
  for (GlobalAlias &GA : M.aliases())
    if (Comdat *C = GA.getComdat())
      ComdatMembers.insert(std::make_pair(C, &GA));

  // Add dependencies between virtual call sites and the virtual functions they
  // might call, if we have that information.
  AddVirtualFunctionDependencies(M);

  // Loop over the module, adding globals which are obviously necessary.
  for (GlobalObject &GO : M.global_objects()) {
    GO.removeDeadConstantUsers();
    // Functions with external linkage are needed if they have a body.
    // Externally visible & appending globals are needed, if they have an
    // initializer.
    if (!GO.isDeclaration())
      if (!GO.isDiscardableIfUnused())
        MarkLive(GO);

    UpdateGVDependencies(GO);
  }

  // Compute direct dependencies of aliases.
  for (GlobalAlias &GA : M.aliases()) {
    GA.removeDeadConstantUsers();
    // Externally visible aliases are needed.
    if (!GA.isDiscardableIfUnused())
      MarkLive(GA);

    UpdateGVDependencies(GA);
  }

  // Compute direct dependencies of ifuncs.
  for (GlobalIFunc &GIF : M.ifuncs()) {
    GIF.removeDeadConstantUsers();
    // Externally visible ifuncs are needed.
    if (!GIF.isDiscardableIfUnused())
      MarkLive(GIF);

    UpdateGVDependencies(GIF);
  }

  // Propagate liveness from collected Global Values through the computed
  // dependencies.
  SmallVector<GlobalValue *, 8> NewLiveGVs{AliveGlobals.begin(),
                                           AliveGlobals.end()};
  while (!NewLiveGVs.empty()) {
    GlobalValue *LGV = NewLiveGVs.pop_back_val();
    for (auto *GVD : GVDependencies[LGV])
      MarkLive(*GVD, &NewLiveGVs);
  }

  // Now that all globals which are needed are in the AliveGlobals set, we loop
  // through the program, deleting those which are not alive.
  //

  // The first pass is to drop initializers of global variables which are dead.
  std::vector<GlobalVariable *> DeadGlobalVars; // Keep track of dead globals
  for (GlobalVariable &GV : M.globals())
    if (!AliveGlobals.count(&GV)) {
      DeadGlobalVars.push_back(&GV);         // Keep track of dead globals
      if (GV.hasInitializer()) {
        Constant *Init = GV.getInitializer();
        GV.setInitializer(nullptr);
        if (isSafeToDestroyConstant(Init))
          Init->destroyConstant();
      }
    }

  // The second pass drops the bodies of functions which are dead...
  std::vector<Function *> DeadFunctions;
  std::set<Function *> DeadFunctionsSet;
  auto funcRemovedInIndex = [&](Function *F) -> bool {
    if (!ImportSummary)
      return false;
    auto &vfuncsRemoved = ImportSummary->funcsToBeRemoved();
    // If function is internalized, its current GUID will be different
    // from the GUID in funcsToBeRemoved. Treat it as a global and search
    // again.
    if (vfuncsRemoved.count(F->getGUID()))
      return true;
    if (vfuncsRemoved.count(GlobalValue::getGUID(F->getName())))
      return true;
    return false;
  };
  for (Function &F : M) {
    if (!AliveGlobals.count(&F)) {
      DeadFunctions.push_back(&F); // Keep track of dead globals
      DeadFunctionsSet.insert(&F);
      if (!F.isDeclaration())
        F.deleteBody();
    } else if (funcRemovedInIndex(&F)) {
      DeadFunctions.push_back(&F); // Keep track of dead globals
      DeadFunctionsSet.insert(&F);
      if (!F.isDeclaration())
        F.deleteBody();
    }
  }

  // The third pass drops targets of aliases which are dead...
  std::vector<GlobalAlias*> DeadAliases;
  for (GlobalAlias &GA : M.aliases())
    if (!AliveGlobals.count(&GA)) {
      DeadAliases.push_back(&GA);
      GA.setAliasee(nullptr);
    }

  // The fourth pass drops targets of ifuncs which are dead...
  std::vector<GlobalIFunc*> DeadIFuncs;
  for (GlobalIFunc &GIF : M.ifuncs())
    if (!AliveGlobals.count(&GIF)) {
      DeadIFuncs.push_back(&GIF);
      GIF.setResolver(nullptr);
    }

  // Now that all interferences have been dropped, delete the actual objects
  // themselves.
  auto EraseUnusedGlobalValue = [&](GlobalValue *GV) {
    GV->removeDeadConstantUsers();
    GV->eraseFromParent();
    Changed = true;
  };

  NumFunctions += DeadFunctions.size();
  for (Function *F : DeadFunctions) {
    if (!F->use_empty()) {
      // Virtual functions might still be referenced by one or more vtables,
      // but if we've proven them to be unused then it's safe to replace the
      // virtual function pointers with null, allowing us to remove the
      // function itself.
      ++NumVFuncs;

      // Detect vfuncs that are referenced as "relative pointers" which are used
      // in Swift vtables, i.e. entries in the form of:
      //
      //   i32 trunc (i64 sub (i64 ptrtoint @f, i64 ptrtoint ...)) to i32)
      //
      // In this case, replace the whole "sub" expression with constant 0 to
      // avoid leaving a weird sub(0, symbol) expression behind.
      replaceRelativePointerUsersWithZero(F);

      F->replaceNonMetadataUsesWith(ConstantPointerNull::get(F->getType()));
    }
    EraseUnusedGlobalValue(F);
  }

  NumVariables += DeadGlobalVars.size();
  for (GlobalVariable *GV : DeadGlobalVars)
    EraseUnusedGlobalValue(GV);

  NumAliases += DeadAliases.size();
  for (GlobalAlias *GA : DeadAliases)
    EraseUnusedGlobalValue(GA);

  NumIFuncs += DeadIFuncs.size();
  for (GlobalIFunc *GIF : DeadIFuncs)
    EraseUnusedGlobalValue(GIF);

  // Make sure that all memory is released
  AliveGlobals.clear();
  ConstantDependenciesCache.clear();
  GVDependencies.clear();
  ComdatMembers.clear();
  TypeIdMap.clear();
  VFESafeVTables.clear();

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

void GlobalDCEPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<GlobalDCEPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  if (InLTOPostLink)
    OS << "<vfe-linkage-unit-visibility>";
}
