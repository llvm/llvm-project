//===-- Instrumentation.cpp - TransformUtils Infrastructure ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the common initialization infrastructure for the
// Instrumentation library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

static cl::opt<bool> ClIgnoreRedundantInstrumentation(
    "ignore-redundant-instrumentation",
    cl::desc("Ignore redundant instrumentation"), cl::Hidden, cl::init(false));

/// Check if module has flag attached, if not add the flag.
bool llvm::checkIfAlreadyInstrumented(Module &M, StringRef Flag) {
  if (!M.getModuleFlag(Flag)) {
    M.addModuleFlag(Module::ModFlagBehavior::Override, Flag, 1);
    return false;
  }
  if (ClIgnoreRedundantInstrumentation)
    return true;
  std::string diagInfo =
      "Redundant instrumentation detected, with module flag: " +
      std::string(Flag);
  M.getContext().diagnose(
      DiagnosticInfoInstrumentation(diagInfo, DiagnosticSeverity::DS_Warning));
  return true;
}

/// Moves I before IP. Returns new insert point.
static BasicBlock::iterator moveBeforeInsertPoint(BasicBlock::iterator I,
                                                  BasicBlock::iterator IP) {
  // If I is IP, move the insert point down.
  if (I == IP) {
    ++IP;
  } else {
    // Otherwise, move I before IP and return IP.
    I->moveBefore(IP);
  }
  return IP;
}

/// Instrumentation passes often insert conditional checks into entry blocks.
/// Call this function before splitting the entry block to move instructions
/// that must remain in the entry block up before the split point. Static
/// allocas and llvm.localescape calls, for example, must remain in the entry
/// block.
BasicBlock::iterator llvm::PrepareToSplitEntryBlock(BasicBlock &BB,
                                                    BasicBlock::iterator IP) {
  assert(&BB.getParent()->getEntryBlock() == &BB);
  for (auto I = IP, E = BB.end(); I != E; ++I) {
    bool KeepInEntry = false;
    if (auto *AI = dyn_cast<AllocaInst>(I)) {
      if (AI->isStaticAlloca())
        KeepInEntry = true;
    } else if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      if (II->getIntrinsicID() == llvm::Intrinsic::localescape)
        KeepInEntry = true;
    }
    if (KeepInEntry)
      IP = moveBeforeInsertPoint(I, IP);
  }
  return IP;
}

// Create a constant for Str so that we can pass it to the run-time lib.
GlobalVariable *llvm::createPrivateGlobalForString(Module &M, StringRef Str,
                                                   bool AllowMerging,
                                                   Twine NamePrefix) {
  Constant *StrConst = ConstantDataArray::getString(M.getContext(), Str);
  // We use private linkage for module-local strings. If they can be merged
  // with another one, we set the unnamed_addr attribute.
  GlobalVariable *GV =
      new GlobalVariable(M, StrConst->getType(), true,
                         GlobalValue::PrivateLinkage, StrConst, NamePrefix);
  if (AllowMerging)
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  GV->setAlignment(Align(1)); // Strings may not be merged w/o setting
                              // alignment explicitly.
  return GV;
}

Comdat *llvm::getOrCreateFunctionComdat(Function &F, Triple &T) {
  if (auto Comdat = F.getComdat())
    return Comdat;
  assert(F.hasName());
  Module *M = F.getParent();

  // Make a new comdat for the function. Use the "no duplicates" selection kind
  // if the object file format supports it. For COFF we restrict it to non-weak
  // symbols.
  Comdat *C = M->getOrInsertComdat(F.getName());
  if (T.isOSBinFormatELF() || (T.isOSBinFormatCOFF() && !F.isWeakForLinker()))
    C->setSelectionKind(Comdat::NoDeduplicate);
  F.setComdat(C);
  return C;
}

void llvm::setGlobalVariableLargeSection(const Triple &TargetTriple,
                                         GlobalVariable &GV) {
  // Limit to x86-64 ELF.
  if (TargetTriple.getArch() != Triple::x86_64 ||
      TargetTriple.getObjectFormat() != Triple::ELF)
    return;
  // Limit to medium/large code models.
  std::optional<CodeModel::Model> CM = GV.getParent()->getCodeModel();
  if (!CM || (*CM != CodeModel::Medium && *CM != CodeModel::Large))
    return;
  GV.setCodeModel(CodeModel::Large);
}

static bool isSanitizerInstrumentationBegin(const Instruction &I) {
  const auto *II = dyn_cast<IntrinsicInst>(&I);
  return II &&
         II->getIntrinsicID() == Intrinsic::sanitizer_instrumentation_begin;
}

static bool isSanitizerInstrumentationEnd(const Instruction &I) {
  const auto *II = dyn_cast<IntrinsicInst>(&I);
  return II && II->getIntrinsicID() == Intrinsic::sanitizer_instrumentation_end;
}

bool llvm::hasSanitizerInstrumentationRegions(const Function &F) {
  // Most modules never use the markers, so avoid scanning their functions.
  const Module *M = F.getParent();
  auto IsUsed = [M](Intrinsic::ID ID) {
    const Function *Decl = Intrinsic::getDeclarationIfExists(M, ID);
    return Decl && !Decl->use_empty();
  };
  if (!IsUsed(Intrinsic::sanitizer_instrumentation_begin) &&
      !IsUsed(Intrinsic::sanitizer_instrumentation_end))
    return false;

  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      if (isSanitizerInstrumentationBegin(I) ||
          isSanitizerInstrumentationEnd(I))
        return true;
  return false;
}

// Lattice top of the region analysis: this block has not been reached yet, so
// it does not constrain the depth of its successors.
static constexpr unsigned UnreachedDepth = ~0U;

/// Return the region nesting depth after \p I, given the depth before it.
static unsigned regionDepthAfter(const Instruction &I, unsigned Depth) {
  if (isSanitizerInstrumentationBegin(I))
    return Depth + 1;
  // Saturate at zero: an unbalanced end() is malformed input, and wrapping
  // around would spuriously instrument the rest of the function.
  if (isSanitizerInstrumentationEnd(I) && Depth)
    return Depth - 1;
  return Depth;
}

static const char *const SanitizerRegionsDiagnosedMD =
    "sanitizer.regions.diagnosed";

/// Find the first problem with the region markers of \p F, given the fixed
/// point of the region analysis, and report it.
///
/// With the "must" (minimum) join the analysis is exact as long as all paths
/// to a block agree on the nesting depth, so the first point where they
/// disagree is a block whose reached predecessors leave with different depths.
/// Joins are checked first, so that an end() or a return that only looks
/// unbalanced because of an earlier bad join is not blamed instead.
static void diagnoseSanitizerInstrumentationRegions(
    Function &F, const DenseMap<const BasicBlock *, unsigned> &DepthIn,
    const DenseMap<const BasicBlock *, unsigned> &DepthOut) {
  if (F.getMetadata(SanitizerRegionsDiagnosedMD))
    return;

  const Instruction *Where = nullptr;
  const char *Msg = nullptr;
  for (const BasicBlock &BB : F) {
    if (&BB == &F.getEntryBlock() || DepthIn.lookup(&BB) == UnreachedDepth)
      continue;
    unsigned Min = UnreachedDepth, Max = 0;
    for (const BasicBlock *Pred : predecessors(&BB)) {
      unsigned Out = DepthOut.lookup(Pred);
      if (Out == UnreachedDepth)
        continue;
      Min = std::min(Min, Out);
      Max = std::max(Max, Out);
    }
    if (Min != Max) {
      // Point at the first instruction that has a location, if any.
      Where = &*BB.getFirstNonPHIOrDbgOrAlloca();
      for (const Instruction &I : BB)
        if (I.getDebugLoc()) {
          Where = &I;
          break;
        }
      Msg = Min == 0
                ? "control flow merges from paths with different sanitizer "
                  "instrumentation region nesting; the code that follows is "
                  "treated as outside the region"
                : "control flow merges from paths with different sanitizer "
                  "instrumentation region nesting; the code that follows is "
                  "treated as nested in the smaller number of regions";
      break;
    }
  }

  for (const BasicBlock &BB : F) {
    if (Msg)
      break;
    unsigned Depth = DepthIn.lookup(&BB);
    if (Depth == UnreachedDepth)
      continue;
    for (const Instruction &I : BB) {
      if (isSanitizerInstrumentationEnd(I) && !Depth) {
        Where = &I;
        Msg = "sanitizer instrumentation region end without a matching begin";
        break;
      }
      if (isa<ReturnInst>(I) && Depth) {
        // Only returns are checked. Not closing a region before a call that
        // does not return (BUG(), panic()) is common and harmless, and
        // unwinding out of a function cannot be checked consistently: a call
        // without a landing pad leaves no terminator to look at.
        Where = &I;
        Msg = "sanitizer instrumentation region is not closed before return";
        break;
      }
      Depth = regionDepthAfter(I, Depth);
    }
  }

  if (!Msg)
    return;
  F.getContext().diagnose(DiagnosticInfoSanitizerInstrumentationRegion(
      F, Where->getDebugLoc(),
      Twine("in function '") + F.getName() + "': " + Msg));
  // Several sanitizer passes may analyze the same function; warn only once.
  F.setMetadata(SanitizerRegionsDiagnosedMD, MDNode::get(F.getContext(), {}));
}

void llvm::computeSanitizerInstrumentationRegions(
    Function &F, SmallPtrSetImpl<const Instruction *> &InRegion) {
  DenseMap<const BasicBlock *, unsigned> DepthIn;
  DenseMap<const BasicBlock *, unsigned> DepthOut;
  const BasicBlock *Entry = &F.getEntryBlock();
  for (const BasicBlock &BB : F) {
    DepthIn[&BB] = UnreachedDepth;
    DepthOut[&BB] = UnreachedDepth;
  }
  DepthIn[Entry] = 0;

  // Both maps only ever decrease and are bounded below by zero, so this
  // fixed-point iteration terminates. A block is only rescanned when its input
  // depth decreased, or when it has not been scanned yet.
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (const BasicBlock &BB : F) {
      bool Rescan = DepthOut[&BB] == UnreachedDepth;
      if (&BB != Entry) {
        unsigned In = UnreachedDepth;
        for (const BasicBlock *Pred : predecessors(&BB))
          In = std::min(In, DepthOut[Pred]);
        if (In < DepthIn[&BB]) {
          DepthIn[&BB] = In;
          Rescan = true;
        }
      }

      unsigned Depth = DepthIn[&BB];
      if (!Rescan || Depth == UnreachedDepth)
        continue;
      for (const Instruction &I : BB)
        Depth = regionDepthAfter(I, Depth);
      if (Depth < DepthOut[&BB]) {
        DepthOut[&BB] = Depth;
        Changed = true;
      }
    }
  }

  for (const BasicBlock &BB : F) {
    unsigned Depth = DepthIn[&BB];
    if (Depth == UnreachedDepth)
      continue;
    for (const Instruction &I : BB) {
      Depth = regionDepthAfter(I, Depth);
      if (Depth > 0)
        InRegion.insert(&I);
    }
  }

  diagnoseSanitizerInstrumentationRegions(F, DepthIn, DepthOut);
}
