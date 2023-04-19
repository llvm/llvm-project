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
#include "llvm/IR/GlobalPtrAuthInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "globaldce"

static cl::opt<bool>
    ClEnableVFE("enable-vfe", cl::Hidden, cl::init(true),
                cl::desc("Enable virtual function elimination"));

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
    // If this is a dep from a vtable to a virtual function, and it's within the
    // range specified in !vcall_visibility, and we have complete information
    // about all virtual call sites which could call though this vtable, then
    // skip it, because the call site information will be more precise.
    if (VFESafeVTablesAndFns.count(GVU) &&
        VFESafeVTablesAndFns[GVU].contains(&GV)) {
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

void GlobalDCEPass::PropagateLivenessInGlobalValues() {
  // Propagate liveness from collected Global Values through the computed
  // dependencies.
  SmallVector<GlobalValue *, 8> NewLiveGVs{AliveGlobals.begin(),
                                           AliveGlobals.end()};
  while (!NewLiveGVs.empty()) {
    GlobalValue *LGV = NewLiveGVs.pop_back_val();
    for (auto *GVD : GVDependencies[LGV])
      MarkLive(*GVD, &NewLiveGVs);
  }
}

/// Recursively iterate over the (sub-)constants in the vtable and look for
/// vptrs, if their offset is within [RangeStart..RangeEnd), add them to VFuncs.
static void FindVirtualFunctionsInVTable(Module &M, Constant *C,
                                         uint64_t RangeStart, uint64_t RangeEnd,
                                         SmallPtrSet<GlobalValue *, 8> *VFuncs,
                                         uint64_t BaseOffset = 0) {
  if (auto *GV = dyn_cast<GlobalValue>(C)) {
    if (RangeStart <= BaseOffset && BaseOffset < RangeEnd) {
      if (auto *F = dyn_cast<Function>(GV))
        VFuncs->insert(F);
      else if (auto PAI = GlobalPtrAuthInfo::analyze(GV))
        if (isa<Function>(PAI->getPointer()->stripPointerCasts()))
          VFuncs->insert(GV);
    }

    // Do not recurse outside of the current global.
    return;
  }

  if (auto *S = dyn_cast<ConstantStruct>(C)) {
    StructType *STy = dyn_cast<StructType>(S->getType());
    const StructLayout *SL = M.getDataLayout().getStructLayout(STy);
    for (auto EI : llvm::enumerate(STy->elements())) {
      auto Offset = SL->getElementOffset(EI.index());
      unsigned Op = SL->getElementContainingOffset(Offset);
      FindVirtualFunctionsInVTable(M, cast<Constant>(S->getOperand(Op)),
                                   RangeStart, RangeEnd, VFuncs,
                                   BaseOffset + Offset);
    }
  } else if (auto *A = dyn_cast<ConstantArray>(C)) {
    ArrayType *ATy = A->getType();
    auto EltSize = M.getDataLayout().getTypeAllocSize(ATy->getElementType());
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i) {
      FindVirtualFunctionsInVTable(M, cast<Constant>(A->getOperand(i)),
                                   RangeStart, RangeEnd, VFuncs,
                                   BaseOffset + EltSize * i);
    }
  } else {
    for (auto &Op : C->operands()) {
      FindVirtualFunctionsInVTable(M, cast<Constant>(Op), RangeStart, RangeEnd,
                                   VFuncs, BaseOffset);
    }
  }
}

void GlobalDCEPass::ScanVTables(Module &M) {
  SmallVector<MDNode *, 2> Types;
  LLVM_DEBUG(dbgs() << "Building type info -> vtable map\n");

  auto *LTOPostLinkMD =
      cast_or_null<ConstantAsMetadata>(M.getModuleFlag("LTOPostLink"));
  bool LTOPostLink =
      LTOPostLinkMD && !cast<ConstantInt>(LTOPostLinkMD->getValue())->isZero();

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
          (LTOPostLink &&
           TypeVis == GlobalObject::VCallVisibilityLinkageUnit)) {
        LLVM_DEBUG(dbgs() << GV.getName() << " is safe for VFE\n");

        // Find and record all the vfunctions that are within the offset range
        // specified in the !vcall_visibility attribute.
        auto Range = GO->getVTableOffsetRange();
        SmallPtrSet<GlobalValue *, 8> VFuncs;
        FindVirtualFunctionsInVTable(M, GV.getInitializer(), std::get<0>(Range),
                                     std::get<1>(Range), &VFuncs);
        VFESafeVTablesAndFns[&GV] = VFuncs;
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
      VFESafeVTablesAndFns.erase(VTable);
      continue;
    }

    Ptr = Ptr->stripPointerCasts();

    GlobalValue *Callee = dyn_cast<Function>(Ptr);
    if (!Callee)
      if (GlobalPtrAuthInfo::analyze(Ptr))
        Callee = dyn_cast<GlobalValue>(Ptr);

    if (!Callee) {
      LLVM_DEBUG(dbgs() << "vtable entry is not function pointer or a .ptrauth "
                           "global variable!\n");
      VFESafeVTablesAndFns.erase(VTable);
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

  if (!TypeCheckedLoadFunc)
    return;

  for (auto *U : TypeCheckedLoadFunc->users()) {
    auto CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    auto *Offset = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    Value *TypeIdValue = CI->getArgOperand(2);
    auto *TypeId = cast<MetadataAsValue>(TypeIdValue)->getMetadata();

    if (Offset) {
      ScanVTableLoad(CI->getFunction(), TypeId, Offset->getZExtValue());
    } else {
      // type.checked.load with a non-constant offset, so assume every entry in
      // every matching vtable is used.
      for (const auto &VTableInfo : TypeIdMap[TypeId]) {
        VFESafeVTablesAndFns.erase(VTableInfo.first);
      }
    }
  }
}

void GlobalDCEPass::AddVirtualFunctionDependencies(Module &M) {
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

  if (VFESafeVTablesAndFns.empty())
    return;

  ScanTypeCheckedLoadIntrinsics(M);

  LLVM_DEBUG(dbgs() << "VFE safe vtables:\n";
             for (auto &Entry
                  : VFESafeVTablesAndFns) dbgs()
             << "  " << Entry.first->getName() << "\n";);
}

static bool RemoveConditionalTargetsFromUsedList(Module &M) {
  auto *Used = M.getGlobalVariable("llvm.used");
  if (!Used)
    return false;

  auto *UsedConditional = M.getNamedMetadata("llvm.used.conditional");
  if (!UsedConditional)
    return false;
  if (UsedConditional->getNumOperands() == 0)
    return false;

  // Construct a set of conditionally used targets.
  SmallPtrSet<GlobalValue *, 8> Targets;
  for (auto *M : UsedConditional->operands()) {
    assert(M->getNumOperands() == 3);
    auto *V = mdconst::extract_or_null<GlobalValue>(M->getOperand(0));
    if (!V)
      continue;
    Targets.insert(V);
  }

  if (Targets.empty())
    return false;

  // Now remove all targets from @llvm.used.
  SmallPtrSet<GlobalValue *, 8> NewUsedArray;
  const ConstantArray *UsedList = cast<ConstantArray>(Used->getInitializer());
  for (Value *Op : UsedList->operands()) {
    GlobalValue *G = cast<GlobalValue>(Op->stripPointerCasts());
    if (Targets.contains(G))
      continue;
    NewUsedArray.insert(G);
  }
  Used = setUsedInitializer(*Used, NewUsedArray);
  return true;
}

// Parse one entry from !llvm.used.conditional list as a triplet of
// { target, type, dependencies } and evaluate the conditional dependency, i.e.
// check liveness of all dependencies and based on type conclude whether the
// target is supposed to be declared alive. If yes, return the target, otherwise
// return nullptr.
GlobalValue *GlobalDCEPass::TargetFromConditionalUsedIfLive(MDNode *M) {
  assert(M->getNumOperands() == 3);
  auto *Target = mdconst::extract_or_null<GlobalValue>(M->getOperand(0));
  if (!Target)
    return nullptr;

  auto *DependenciesMD = dyn_cast_or_null<MDNode>(M->getOperand(2).get());
  SmallPtrSet<GlobalValue *, 8> Dependencies;
  if (DependenciesMD == nullptr) {
    Dependencies.insert(nullptr);
  } else {
    for (auto &DependencyMD : DependenciesMD->operands()) {
      auto *Dependency = DependencyMD.get();
      if (!Dependency)
        continue; // Allow null, skip.
      auto *C =
          mdconst::extract_or_null<Constant>(Dependency)->stripPointerCasts();
      if (dyn_cast<UndefValue>(C))
        continue; // Allow undef, skip.
      Dependencies.insert(cast<GlobalValue>(C));
    }
  }

  bool AllDependenciesAlive = Dependencies.empty() ? false : true;
  bool AnyDependencyAlive = false;
  for (auto *Dep : Dependencies) {
    bool Live = AliveGlobals.count(Dep) != 0;
    if (Live)
      AnyDependencyAlive = true;
    else
      AllDependenciesAlive = false;
  }

  auto *Type = mdconst::extract_or_null<ConstantInt>(M->getOperand(1));
  switch (Type->getValue().getSExtValue()) {
  case 0:
    return AnyDependencyAlive ? Target : nullptr;
  case 1:
    return AllDependenciesAlive ? Target : nullptr;
  default:
    llvm_unreachable("bad !llvm.used.conditional type");
  }
}

void GlobalDCEPass::PropagateLivenessToConditionallyUsed(Module &M) {
  auto *Used = M.getGlobalVariable("llvm.used");
  if (!Used)
    return;
  auto *UsedConditional = M.getNamedMetadata("llvm.used.conditional");
  if (!UsedConditional)
    return;

  SmallPtrSet<GlobalValue *, 8> NewUsedArray;
  const ConstantArray *UsedList = cast<ConstantArray>(Used->getInitializer());
  for (Value *Op : UsedList->operands()) {
    NewUsedArray.insert(cast<GlobalValue>(Op->stripPointerCasts()));
  }

  // Repeat the liveness propagation iteraticely, one iteration might force
  // other conditionally used globals to become alive.
  while (true) {
    PropagateLivenessInGlobalValues();

    unsigned OldSize = NewUsedArray.size();
    for (auto *M : UsedConditional->operands()) {
      auto *Target = TargetFromConditionalUsedIfLive(M);
      if (!Target) continue;

      NewUsedArray.insert(Target);
      MarkLive(*Target);
      LLVM_DEBUG(dbgs() << "Conditionally used target alive: "
                        << Target->getName() << "\n");
    }

    unsigned NewSize = NewUsedArray.size();
    LLVM_DEBUG(dbgs() << "Conditionally used iteration end, old size: "
                      << OldSize << " new size: " << NewSize << "\n");

    // Stop the iteration once we reach a steady state (no new additions to
    // @llvm.used).
    if (NewSize == OldSize) break;
  }

  Used = setUsedInitializer(*Used, NewUsedArray);
  MarkLive(*Used);
}

PreservedAnalyses GlobalDCEPass::run(Module &M, ModuleAnalysisManager &MAM) {
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

  // Process the !llvm.used.conditional list and (temporarily, see below)
  // remove all "targets" from @llvm.used. No effect if `!llvm.used.conditional`
  // is not present in the module.
  bool UsedConditionalPresent = RemoveConditionalTargetsFromUsedList(M);

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

  // Step 2 of !llvm.used.conditional processing: If any conditionally used
  // "targets" are alive, put them back into @llvm.used.
  if (UsedConditionalPresent) {
    PropagateLivenessToConditionallyUsed(M);
  }

  PropagateLivenessInGlobalValues();

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
  for (Function &F : M)
    if (!AliveGlobals.count(&F)) {
      DeadFunctions.push_back(&F);         // Keep track of dead globals
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
  for (GlobalVariable *GV : DeadGlobalVars) {
    if (!GV->use_empty()) {
      // Normally, a vtable only contain Function references that are eliminated
      // by VFE, and their "leftover uses" are handled by the for loop above.
      // But with ptrauth on, we can also get "leftover uses" of GlobalVariables
      // because the vtable references the .ptrauth wrappers instead. So we need
      // to apply the same use-erasing logic as above. The same reasoning as
      // above applies: These are proven to be unused, so they're safe to
      // replace with null.
      GV->replaceNonMetadataUsesWith(ConstantPointerNull::get(GV->getType()));
    }
    EraseUnusedGlobalValue(GV);
  }

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
  VFESafeVTablesAndFns.clear();

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
