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
#include "IPOOptions.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TypeMetadataUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "globaldce"

namespace {
class GlobalDCELegacyPass : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  GlobalDCELegacyPass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    // Note: GlobalDCEPass does not use any analyses, so we're safe to call the
    // new-pm style pass with a default-initialized analysis manager here
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  GlobalDCEPass Impl;
};
} // namespace

char GlobalDCELegacyPass::ID = 0;
INITIALIZE_PASS(GlobalDCELegacyPass, "globaldce", "Dead Global Elimination",
                false, false)

// Public interface to the GlobalDCEPass.
ModulePass *llvm::createGlobalDCEPass() { return new GlobalDCELegacyPass(); }

STATISTIC(NumAssumeUses, "Number of assumption uses of dead globals dropped");
STATISTIC(NumAliases  , "Number of global aliases removed");
STATISTIC(NumFunctions, "Number of functions removed");
STATISTIC(NumIFuncs,    "Number of indirect functions removed");
STATISTIC(NumVariables, "Number of global variables removed");
STATISTIC(NumVFuncs,    "Number of virtual functions removed");

/// Returns true if F is effectively empty.
static bool isEmptyFunction(Function *F) {
  // Skip external functions.
  if (F->isDeclaration())
    return false;
  BasicBlock &Entry = F->getEntryBlock();
  for (auto &I : Entry) {
    if (isa<PseudoProbeInst>(I))
      continue;
    if (auto *RI = dyn_cast<ReturnInst>(&I))
      return !RI->getReturnValue();
    break;
  }
  return false;
}

/// Returns true if I can be erased once all of its users are erased.
static bool canBeEphemeral(const Instruction *I) {
  if (I->mayHaveSideEffects() || I->isTerminator() || I->isEHPad())
    return false;
  // Type tests have special semantics, keep them.
  if (auto *II = dyn_cast<IntrinsicInst>(I))
    return II->getIntrinsicID() != Intrinsic::type_test &&
           II->getIntrinsicID() != Intrinsic::public_type_test;
  return true;
}

/// Collect the assumptions and the instructions whose uses are all, directly
/// or transitively, in assumptions.
void GlobalDCEPass::collectEphemeralValues(Module &M) {
  Function *AssumeFn = Intrinsic::getDeclarationIfExists(&M, Intrinsic::assume);
  if (!AssumeFn)
    return;

  SmallVector<Instruction *, 16> Worklist;
  for (User *U : AssumeFn->users())
    if (auto *Assume = dyn_cast<AssumeInst>(U))
      Worklist.push_back(Assume);

  // An instruction becomes ephemeral when its last non-ephemeral use is found
  // to be ephemeral. Uses in a cycle are never all ephemeral.
  DenseMap<Instruction *, unsigned> NumOtherUses;
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!EphemeralValues.insert(I).second)
      continue;
    for (Value *Op : I->operands()) {
      auto *OpI = dyn_cast<Instruction>(Op);
      if (!OpI || !canBeEphemeral(OpI))
        continue;
      auto It = NumOtherUses.try_emplace(OpI, OpI->getNumUses()).first;
      if (--It->second == 0)
        Worklist.push_back(OpI);
    }
  }
}

/// Drop the uses of dead globals from the assumptions in live functions, and
/// erase the instructions that only computed them.
bool GlobalDCEPass::dropAssumesOnDeadGlobals(Module &M,
                                             FunctionAnalysisManager *FAM) {
  SmallVector<Use *, 8> AssumeUses;
  SmallSetVector<Instruction *, 8> DeadInsts;
  SmallPtrSet<Constant *, 8> VisitedConstants;
  SmallVector<Value *, 16> Worklist;
  for (GlobalValue &GV : M.global_values())
    if (!AliveGlobals.contains(&GV))
      Worklist.push_back(&GV);

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    for (Use &U : V->uses()) {
      User *Usr = U.getUser();
      if (auto *I = dyn_cast<Instruction>(Usr)) {
        if (!AliveGlobals.contains(I->getFunction()))
          continue;
        assert(EphemeralValues.contains(I) && "dead global used by live code");
        if (isa<AssumeInst>(I))
          AssumeUses.push_back(&U);
        else if (DeadInsts.insert(I))
          Worklist.push_back(I);
      } else if (auto *C = dyn_cast<Constant>(Usr)) {
        if (!isa<GlobalValue>(C) && VisitedConstants.insert(C).second)
          Worklist.push_back(C);
      }
    }
  }
  if (AssumeUses.empty())
    return false;

  SmallSetVector<Function *, 4> ChangedFunctions;
  SmallVector<WeakTrackingVH, 16> MaybeDead;
  for (Use *U : AssumeUses) {
    auto *Assume = cast<AssumeInst>(U->getUser());
    Value::dropDroppableUse(*U);
    ChangedFunctions.insert(Assume->getFunction());
    MaybeDead.push_back(Assume);
  }
  NumAssumeUses += AssumeUses.size();

  for (Instruction *I : DeadInsts)
    for (Value *Op : I->operands())
      if (auto *OpI = dyn_cast<Instruction>(Op);
          OpI && !DeadInsts.contains(OpI))
        MaybeDead.push_back(OpI);
  for (Instruction *I : DeadInsts)
    I->dropAllReferences();
  for (Instruction *I : DeadInsts)
    I->eraseFromParent();
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(MaybeDead);

  if (FAM) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    for (Function *F : ChangedFunctions)
      FAM->invalidate(*F, PA);
  }
  return true;
}

/// Compute the set of GlobalValue that depends from V.
/// The recursion stops as soon as a GlobalValue is met.
void GlobalDCEPass::ComputeDependencies(Value *V,
                                        SmallPtrSetImpl<GlobalValue *> &Deps) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    // A use that only feeds assumptions does not keep a global alive.
    if (EphemeralValues.contains(I))
      return;
    Function *Parent = I->getParent()->getParent();
    Deps.insert(Parent);
  } else if (auto *GV = dyn_cast<GlobalValue>(V)) {
    Deps.insert(GV);
  } else if (auto *CE = dyn_cast<Constant>(V)) {
    // Avoid walking the whole tree of a big ConstantExprs multiple times.
    auto Where = ConstantDependenciesCache.find(CE);
    if (Where == ConstantDependenciesCache.end()) {
      // Compute into a local set first: the recursion can insert into the
      // cache, which would invalidate references into it.
      SmallPtrSet<GlobalValue *, 8> LocalDeps;
      for (User *CEUser : CE->users())
        ComputeDependencies(CEUser, LocalDeps);
      Where =
          ConstantDependenciesCache.try_emplace(CE, std::move(LocalDeps)).first;
    }
    Deps.insert_range(Where->second);
  }
}

void GlobalDCEPass::UpdateGVDependencies(GlobalValue &GV) {
  SmallPtrSet<GlobalValue *, 8> Deps;
  for (Use &U : GV.uses()) {
    // A called function is needed even if the call only feeds assumptions.
    if (auto *CB = dyn_cast<CallBase>(U.getUser()); CB && CB->isCallee(&U))
      Deps.insert(CB->getFunction());
    else
      ComputeDependencies(U.getUser(), Deps);
  }
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
    GlobalObject::VCallVisibility TypeVis = GV.getVCallVisibility();
    if (TypeVis == GlobalObject::VCallVisibilityTranslationUnit ||
        (InLTOPostLink &&
         TypeVis == GlobalObject::VCallVisibilityLinkageUnit)) {
      LLVM_DEBUG(dbgs() << GV.getName() << " is safe for VFE\n");
      VFESafeVTables.insert(&GV);
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
    VFuncDependencies[Caller].insert({VTable, Callee});
  }
}

void GlobalDCEPass::ScanTypeCheckedLoadIntrinsics(Module &M) {
  LLVM_DEBUG(dbgs() << "Scanning type.checked.load intrinsics\n");
  Function *TypeCheckedLoadFunc =
      Intrinsic::getDeclarationIfExists(&M, Intrinsic::type_checked_load);
  Function *TypeCheckedLoadRelativeFunc = Intrinsic::getDeclarationIfExists(
      &M, Intrinsic::type_checked_load_relative);

  // Keep track of which (function, type id, offset) triples we've already
  // scanned, to prevent repeated work.
  DenseSet<std::tuple<Function *, Metadata *, uint64_t>> ScannedLoads;

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
        if (ScannedLoads
                .insert({CI->getFunction(), TypeId, Offset->getZExtValue()})
                .second)
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
  if (!getIPOOptions(M).enable_vfe)
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
  bool Changed = false;

  // Only the analyses of the deleted functions need to be cleared.
  FunctionAnalysisManager *FAM = nullptr;
  if (MAM.isPassRegistered<FunctionAnalysisManagerModuleProxy>()) {
    if (auto *FAMProxy =
            MAM.getCachedResult<FunctionAnalysisManagerModuleProxy>(M))
      FAM = &FAMProxy->getManager();
  }

  // The algorithm first computes the set L of global variables that are
  // trivially live.
  //
  // Then it walks the initialization of these variables to compute the globals
  // used to initialize them, which effectively builds a directed graph where
  // nodes are global variables, and an edge from A to B means B is used to
  // initialize A.
  //
  // Finally, it propagates the liveness information through the graph starting
  // from the nodes in L. A virtual call only keeps a virtual function alive
  // when its vtable is also live.
  //
  // Nodes not marked as alive are discarded.

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

  // Uses that only feed assumptions do not keep globals alive.
  if (DropAssumes)
    collectEphemeralValues(M);

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

    // A virtual call keeps a virtual function alive only if the vtable it is
    // loaded from is alive too. If the vtable is not live, defer the function:
    // the vtable may become live later.
    if (auto *F = dyn_cast<Function>(LGV)) {
      if (auto It = VFuncDependencies.find(F); It != VFuncDependencies.end()) {
        for (auto [VTable, VFunc] : It->second) {
          if (AliveGlobals.contains(VTable))
            MarkLive(*VFunc, &NewLiveGVs);
          else
            PendingVFuncs[VTable].push_back(VFunc);
        }
      }
    }

    // If LGV is a vtable that became alive later, mark its deferred functions
    // live.
    if (auto *VT = dyn_cast<GlobalVariable>(LGV)) {
      if (auto It = PendingVFuncs.find(VT); It != PendingVFuncs.end()) {
        for (auto *VFunc : It->second)
          MarkLive(*VFunc, &NewLiveGVs);
      }
    }
  }

  // Now that all globals which are needed are in the AliveGlobals set, we loop
  // through the program, deleting those which are not alive.
  //

  // Live functions can still refer to dead globals in their assumptions.
  if (DropAssumes)
    Changed |= dropAssumesOnDeadGlobals(M, FAM);

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
  for (Function &F : M)
    if (!AliveGlobals.count(&F)) {
      DeadFunctions.push_back(&F);         // Keep track of dead globals
      if (FAM)
        FAM->clear(F, F.getName());
      if (!F.isDeclaration())
        F.deleteBody();
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
  VFuncDependencies.clear();
  PendingVFuncs.clear();
  ComdatMembers.clear();
  TypeIdMap.clear();
  VFESafeVTables.clear();
  EphemeralValues.clear();

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<FunctionAnalysisManagerModuleProxy>();
  PA.preserveSet<AllAnalysesOn<Function>>();
  return PA;
}

void GlobalDCEPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<GlobalDCEPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  if (InLTOPostLink && DropAssumes)
    OS << "<vfe-linkage-unit-visibility;drop-assumes>";
  else if (InLTOPostLink)
    OS << "<vfe-linkage-unit-visibility>";
  else if (DropAssumes)
    OS << "<drop-assumes>";
}
