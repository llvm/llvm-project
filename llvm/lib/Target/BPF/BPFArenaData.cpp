//===-- BPFArenaData.cpp - Keep global data in arena ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In a language without address spaces a pointer to a global variable has to be
// like a pointer to what is on the heap, which is in arena: the same insn
// dereferences both, and a pointer that is stored in arena is a number when
// it's loaded back. So global variables and constants are in arena too.
//
// The loader places two sections into the arena map of the object:
//   .addr_space.1      variables that are not zero
//   .bss.addr_space.1  variables that are zero. It takes no space in the file.
// It resolves pointers to variables in these sections that are stored in
// .addr_space.1.
//
// What is not moved:
// - Variables that have a section: maps, struct_ops, license.
// - Variables with pointers to functions: the kernel looks for them in
//   read-only data.
// - What is passed to helpers and kfuncs: they take memory that the kernel
//   knows about, and strings from read-only maps only. What else uses such
//   a constant gets a copy in arena. There can't be two copies of a variable.
//
// The object needs an arena map. Add one when there is none:
//   struct {
//       __uint(type, BPF_MAP_TYPE_ARENA);
//       __uint(map_flags, BPF_F_MMAPABLE);
//   } arena SEC(".maps");
// The loader picks the size. The program declares the map to have another.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFCORE.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-arena-data"

static constexpr StringLiteral ArenaSec = ".addr_space.1";
static constexpr StringLiteral ArenaBssSec = ".bss.addr_space.1";
static constexpr StringLiteral MapsSec = ".maps";
static constexpr uint64_t BPF_MAP_TYPE_ARENA = 33;
static constexpr uint64_t BPF_F_MMAPABLE = 1 << 10;

namespace {

class ArenaData {
  Module &M;
  function_ref<const TargetLibraryInfo &(Function &)> GetTLI;

  bool isKernelUse(const Use &U) const;
  bool canMove(const GlobalVariable &GV) const;
  void move(GlobalVariable &GV);
  bool checkPointers(const GlobalVariable &GV, const Constant *C) const;
  bool hasArenaMap() const;
  void addArenaMap();

public:
  ArenaData(Module &M,
            function_ref<const TargetLibraryInfo &(Function &)> GetTLI)
      : M(M), GetTLI(GetTLI) {}
  bool run();
};

} // namespace

static bool hasFuncPtr(const Constant *C) {
  if (isa<Function>(C))
    return true;
  if (isa<GlobalValue>(C))
    return false;
  return any_of(C->operands(), [](const Use &U) {
    return hasFuncPtr(cast<Constant>(U.get()));
  });
}

bool llvm::isBPFArenaGlobal(const GlobalVariable &GV) {
  return GV.getSection() == ArenaSec || GV.getSection() == ArenaBssSec;
}

/// The address, maybe with an offset, is passed to a helper or a kfunc.
bool ArenaData::isKernelUse(const Use &U) const {
  const User *Usr = U.getUser();

  if (const auto *CB = dyn_cast<CallBase>(Usr))
    return CB->isArgOperand(&U) && !isBPFArenaArg(*CB, U) &&
           isBPFKernelFunc(*CB,
                           GetTLI(*const_cast<Function *>(CB->getFunction())));
  if (isa<ConstantExpr>(Usr))
    return any_of(Usr->uses(), [&](const Use &U) { return isKernelUse(U); });
  return false;
}

bool ArenaData::canMove(const GlobalVariable &GV) const {
  return !GV.isDeclaration() && !GV.hasSection() && !GV.isThreadLocal() &&
         GV.getAddressSpace() == 0 && !GV.getName().starts_with("llvm.") &&
         !hasFuncPtr(GV.getInitializer());
}

void ArenaData::move(GlobalVariable &GV) {
  const Constant *Init = GV.getInitializer();

  GV.setSection(Init->isNullValue() || isa<UndefValue>(Init) ? ArenaBssSec
                                                             : ArenaSec);
  // One section for constants, strings that can be merged and variables.
  GV.setConstant(false);
  GV.setUnnamedAddr(GlobalValue::UnnamedAddr::None);
}

/// The loader resolves pointers to what is in arena only.
bool ArenaData::checkPointers(const GlobalVariable &GV,
                              const Constant *C) const {
  if (const auto *Target = dyn_cast<GlobalValue>(C)) {
    const auto *TargetGV = dyn_cast<GlobalVariable>(Target);

    if (TargetGV && isBPFArenaGlobal(*TargetGV))
      return true;
    M.getContext().diagnose(DiagnosticInfoGeneric(
        Twine("'") + GV.getName() + "' is in arena and has a pointer to '" +
        Target->getName() + "' that is not"));
    return false;
  }
  return all_of(C->operands(), [&](const Use &U) {
    return checkPointers(GV, cast<Constant>(U.get()));
  });
}

/// __uint(type, BPF_MAP_TYPE_ARENA) is 'int (*type)[BPF_MAP_TYPE_ARENA]'.
bool ArenaData::hasArenaMap() const {
  for (const GlobalVariable &GV : M.globals()) {
    SmallVector<DIGlobalVariableExpression *, 1> GVEs;

    if (GV.getSection() != MapsSec)
      continue;
    GV.getDebugInfo(GVEs);
    for (const DIGlobalVariableExpression *GVE : GVEs) {
      const auto *Ty =
          dyn_cast_or_null<DICompositeType>(GVE->getVariable()->getType());

      if (!Ty)
        continue;
      for (const DINode *N : Ty->getElements()) {
        const auto *Member = dyn_cast<DIDerivedType>(N);

        if (!Member || Member->getName() != "type")
          continue;
        const auto *Ptr =
            dyn_cast_or_null<DIDerivedType>(Member->getBaseType());
        const auto *Arr =
            Ptr ? dyn_cast_or_null<DICompositeType>(Ptr->getBaseType())
                : nullptr;

        if (!Arr || Arr->getElements().size() != 1)
          continue;
        const auto *Range = dyn_cast<DISubrange>(Arr->getElements()[0]);
        const auto *Cnt =
            Range ? dyn_cast_if_present<ConstantInt *>(Range->getCount())
                  : nullptr;

        if (Cnt && Cnt->getZExtValue() == BPF_MAP_TYPE_ARENA)
          return true;
      }
    }
  }
  return false;
}

void ArenaData::addArenaMap() {
  LLVMContext &C = M.getContext();
  Type *Ptr = PointerType::get(C, 0);
  Type *Ty = StructType::get(C, {Ptr, Ptr});
  auto *GV = new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage,
                                ConstantAggregateZero::get(Ty), "arena");

  GV->setSection(MapsSec);
  GV->setAlignment(Align(8));
  // Not llvm.used: that is SHF_GNU_RETAIN, which gives the map a section of its
  // own next to .maps of the maps that the program defines.
  appendToCompilerUsed(M, {GV});

  // The definition of the map is its type in BTF.
  if (M.debug_compile_units().empty())
    return;

  DICompileUnit *CU = *M.debug_compile_units_begin();
  DIBuilder DIB(M, /*AllowUnresolved=*/true, CU);
  DIFile *File = CU->getFile();
  DIType *Int = DIB.createBasicType("int", 32, dwarf::DW_ATE_signed);
  auto MakeUint = [&](StringRef Name, uint64_t Val,
                      uint64_t Off) -> Metadata * {
    DIType *Arr = DIB.createArrayType(
        Val * 32, 32, Int,
        DIB.getOrCreateArray(DIB.getOrCreateSubrange(0, Val)));

    return DIB.createMemberType(CU, Name, File, 0, 64, 64, Off,
                                DINode::FlagZero,
                                DIB.createPointerType(Arr, 64));
  };
  DIType *DITy = DIB.createStructType(
      CU, "", File, 0, 128, 64, DINode::FlagZero, nullptr,
      DIB.getOrCreateArray({MakeUint("type", BPF_MAP_TYPE_ARENA, 0),
                            MakeUint("map_flags", BPF_F_MMAPABLE, 64)}));

  GV->addDebugInfo(DIB.createGlobalVariableExpression(
      CU, GV->getName(), "", File, 0, DITy, /*IsLocalToUnit=*/false));
  DIB.finalize();
}

bool ArenaData::run() {
  SmallVector<GlobalVariable *, 32> Moved;
  bool Changed = false;

  for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
    if (!canMove(GV))
      continue;

    if (none_of(GV.uses(), [&](const Use &U) { return isKernelUse(U); })) {
      move(GV);
      Moved.push_back(&GV);
      continue;
    }
    if (!GV.isConstant() ||
        all_of(GV.uses(), [&](const Use &U) { return isKernelUse(U); }))
      continue;

    auto *Copy =
        new GlobalVariable(M, GV.getValueType(), false, GV.getLinkage(),
                           GV.getInitializer(), GV.getName() + ".arena", &GV);

    Copy->setAlignment(GV.getAlign());
    GV.replaceUsesWithIf(Copy, [&](Use &U) { return !isKernelUse(U); });
    move(*Copy);
    Moved.push_back(Copy);
  }
  Changed |= !Moved.empty();

  for (GlobalVariable *GV : Moved)
    checkPointers(*GV, GV->getInitializer());

  if (!hasArenaMap()) {
    addArenaMap();
    Changed = true;
  }
  return Changed;
}

namespace {

struct BPFArenaDataLegacy : public ModulePass {
  static char ID;
  BPFArenaDataLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override { return "BPF Arena Data"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    auto GetTLI = [this](Function &F) -> const TargetLibraryInfo & {
      return getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };
    return ArenaData(M, GetTLI).run();
  }
};

} // namespace

char BPFArenaDataLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(BPFArenaDataLegacy, DEBUG_TYPE, "BPF Arena Data", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(BPFArenaDataLegacy, DEBUG_TYPE, "BPF Arena Data", false,
                    false)

ModulePass *llvm::createBPFArenaDataPass() { return new BPFArenaDataLegacy(); }

PreservedAnalyses BPFArenaDataPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> const TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  return ArenaData(M, GetTLI).run() ? PreservedAnalyses::none()
                                    : PreservedAnalyses::all();
}
