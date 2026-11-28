//===-- BPFArenaData.cpp - Keep global data in arena ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In a program that keeps its memory in arena a pointer to a global variable
// has to be like a pointer to what is on the heap: the same insn dereferences
// both, and a pointer that is stored in arena is a number when it's loaded
// back. So global variables are in arena too.
//
// The loader places two sections into the arena map of the program:
//   .addr_space.1      variables that are not zero
//   .bss.addr_space.1  variables that are zero. It takes no space in the file.
// It resolves pointers to variables in these sections that are stored in
// .addr_space.1.
//
// What is not moved:
// - Variables that have a section: maps, struct_ops, license.
// - Variables with pointers to functions or to labels: the kernel looks for
//   them in read-only data and in the table of a computed goto.
// - What gets to helpers and kfuncs: they take memory that the kernel knows
//   about, and strings from read-only maps only. What else uses such
//   a constant gets a copy in arena. There can't be two copies of a variable,
//   or of a constant whose address matters: 'p == name' (no unnamed_addr).
// - Variables with fields that have a type tag, like __kptr_untrusted:
//   the kernel looks for them in the BTF of a map value.
// - Variables in which the program stores a pointer of the kernel, if it only
//   compares what it loads from there, keeps it as a number or passes it to
//   a helper or a kfunc. See BPFArenaKptr.cpp.
// - Constants that are not static. The loader can set them before the program
//   is loaded, and the kernel knows the values in read-only data: it drops
//   the branches that they make dead.
// - Static constants that are only loaded from, at constant offsets, or copied
//   to a local. There is no pointer to them.
// The last two don't apply to a constant that holds a pointer to a variable.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFArena.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-arena-data"

/// A pointer to a function or to a label.
static bool hasCodePtr(const Constant *C) {
  if (isa<Function, BlockAddress>(C))
    return true;
  if (isa<GlobalValue>(C))
    return false;
  return any_of(C->operands(), [](const Use &U) {
    return hasCodePtr(cast<Constant>(U.get()));
  });
}

static bool hasPtr(const Constant *C) {
  return isa<GlobalValue>(C) || any_of(C->operands(), [](const Use &U) {
           return hasPtr(cast<Constant>(U.get()));
         });
}

/// A load at a constant offset, or a copy of a constant size: the initial value
/// of a local. It becomes stores of numbers when the source is a constant.
static bool isDirectLoad(const Use &U) {
  const User *Usr = U.getUser();

  if (const auto *GEP = dyn_cast<GEPOperator>(Usr))
    return GEP->hasAllConstantIndices() && all_of(GEP->uses(), isDirectLoad);
  if (const auto *MT = dyn_cast<MemTransferInst>(Usr))
    return &U == &MT->getRawSourceUse() && isa<ConstantInt>(MT->getLength());
  return isa<LoadInst>(Usr);
}

/// The kernel manages the fields that have a type tag (__kptr_untrusted and
/// the like). It finds them in the BTF of a map value, and what the program
/// loads from there is a pointer for the verifier.
static bool hasTypeTag(const DIType *Ty,
                       SmallPtrSetImpl<const DIType *> &Seen) {
  if (!Ty || !Seen.insert(Ty).second)
    return false;
  if (const auto *DT = dyn_cast<DIDerivedType>(Ty)) {
    if (DT->getTag() != dwarf::DW_TAG_pointer_type)
      return hasTypeTag(DT->getBaseType(), Seen);
    DINodeArray Annotations = DT->getAnnotations();
    return Annotations &&
           any_of(Annotations->operands(), [](const MDOperand &Op) {
             const auto *MD = cast<MDNode>(Op);

             return cast<MDString>(MD->getOperand(0))->getString() ==
                        "btf_type_tag" &&
                    cast<MDString>(MD->getOperand(1))->getString() != "arena";
           });
  }
  const auto *CT = dyn_cast<DICompositeType>(Ty);
  return CT && (hasTypeTag(CT->getBaseType(), Seen) ||
                any_of(CT->getElements(), [&](const DINode *N) {
                  return hasTypeTag(dyn_cast<DIType>(N), Seen);
                }));
}

static bool hasTypeTag(const GlobalVariable &GV) {
  SmallVector<DIGlobalVariableExpression *, 1> GVEs;
  SmallPtrSet<const DIType *, 8> Seen;

  GV.getDebugInfo(GVEs);
  return any_of(GVEs, [&](const DIGlobalVariableExpression *E) {
    return hasTypeTag(E->getVariable()->getType(), Seen);
  });
}

static bool canMove(const GlobalVariable &GV) {
  return !GV.isDeclaration() && !GV.hasSection() && !GV.isThreadLocal() &&
         !hasTypeTag(GV) && GV.getAddressSpace() == 0 &&
         !GV.getName().starts_with("llvm.") && !hasCodePtr(GV.getInitializer());
}

static void move(GlobalVariable &GV) {
  const Constant *Init = GV.getInitializer();

  GV.setSection(Init->isNullValue() || isa<UndefValue>(Init)
                    ? ".bss.addr_space.1"
                    : ".addr_space.1");
  // One section for constants, strings that can be merged and variables.
  GV.setConstant(false);
  GV.setUnnamedAddr(GlobalValue::UnnamedAddr::None);
}

/// The loader resolves pointers to what is in arena only.
static bool checkPointers(const GlobalVariable &GV, const Constant *C) {
  if (const auto *Target = dyn_cast<GlobalValue>(C)) {
    const auto *TargetGV = dyn_cast<GlobalVariable>(Target);

    if (TargetGV && isBPFArenaGlobal(*TargetGV))
      return true;
    GV.getContext().diagnose(DiagnosticInfoGeneric(
        Twine("'") + GV.getName() + "' is in arena and has a pointer to '" +
        Target->getName() + "' that is not"));
    return false;
  }
  return all_of(C->operands(), [&](const Use &U) {
    return checkPointers(GV, cast<Constant>(U.get()));
  });
}

static bool moveData(Module &M) {
  BPFKernelMem KM(M);
  SmallVector<GlobalVariable *, 32> Moved;
  SmallPtrSet<const GlobalVariable *, 8> KernelPtrs;
  DenseMap<const GlobalVariable *, bool> Known;

  auto Decide = [&](const GlobalVariable &GV) {
    // A variable in another file of the program.
    if (isBPFArenaGlobal(GV) || (GV.isDeclaration() && !GV.hasSection()))
      return true;
    if (!canMove(GV))
      return false;
    if (GV.isConstant() && !hasPtr(GV.getInitializer()) &&
        (!GV.hasLocalLinkage() || all_of(GV.uses(), [&](const Use &U) {
          return isDirectLoad(U) || KM.isKernelUse(U);
        })))
      return false;
    // The program gets a copy of a constant that the kernel needs.
    return (GV.isConstant() && GV.hasGlobalUnnamedAddr()) || !KM.isPassed(&GV);
  };
  // The search for pointers of the kernel asks for each use of a variable, and
  // again until nothing changes.
  auto Moves = [&](const GlobalVariable &GV) {
    auto It = Known.find(&GV);

    if (It == Known.end())
      It = Known.try_emplace(&GV, Decide(GV)).first;
    return It->second;
  };

  findBPFKernelPtrGlobals(M, Moves, KernelPtrs);
  for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
    if (!canMove(GV) || !Moves(GV) || KernelPtrs.count(&GV))
      continue;

    if (!KM.isPassed(&GV)) {
      move(GV);
      Moved.push_back(&GV);
      continue;
    }

    auto *Copy =
        new GlobalVariable(M, GV.getValueType(), false, GV.getLinkage(),
                           GV.getInitializer(), GV.getName() + ".arena", &GV);

    Copy->setAlignment(GV.getAlign());
    GV.replaceUsesWithIf(Copy, [&](Use &U) { return !KM.isKernelUse(U); });
    move(*Copy);
    Moved.push_back(Copy);
  }

  for (GlobalVariable *GV : Moved)
    checkPointers(*GV, GV->getInitializer());
  return !Moved.empty();
}

namespace {

struct BPFArenaDataLegacy : public ModulePass {
  static char ID;
  BPFArenaDataLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override { return "BPF Arena Data"; }

  bool runOnModule(Module &M) override { return moveData(M); }
};

} // namespace

char BPFArenaDataLegacy::ID = 0;

INITIALIZE_PASS(BPFArenaDataLegacy, DEBUG_TYPE, "BPF Arena Data", false, false)

ModulePass *llvm::createBPFArenaDataPass() { return new BPFArenaDataLegacy(); }

PreservedAnalyses BPFArenaDataPass::run(Module &M, ModuleAnalysisManager &AM) {
  return moveData(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
