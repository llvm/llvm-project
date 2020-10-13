//===-- CodeGen/AsmPrinter/WinCFGuard.cpp - Control Flow Guard Impl ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing the metadata for Windows Control Flow
// Guard, including address-taken functions and valid longjmp targets.
//
//===----------------------------------------------------------------------===//

#include "WinCFGuard.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCStreamer.h"

#include <vector>

using namespace llvm;

WinCFGuard::WinCFGuard(AsmPrinter *A) : AsmPrinterHandler(), Asm(A) {}

WinCFGuard::~WinCFGuard() {}

void WinCFGuard::endFunction(const MachineFunction *MF) {

  // Skip functions without any longjmp targets.
  if (MF->getLongjmpTargets().empty())
    return;

  // Copy the function's longjmp targets to a module-level list.
  LongjmpTargets.insert(LongjmpTargets.end(), MF->getLongjmpTargets().begin(),
                        MF->getLongjmpTargets().end());
}

/// Returns true if this function's address is escaped in a way that might make
/// it an indirect call target. Function::hasAddressTaken gives different
/// results when a function is called directly with a function prototype
/// mismatch, which requires a cast.
static bool isPossibleIndirectCallTarget(const Function *F) {
  SmallVector<const Value *, 4> Users{F};
  while (!Users.empty()) {
    const Value *FnOrCast = Users.pop_back_val();
    for (const Use &U : FnOrCast->uses()) {
      const User *FnUser = U.getUser();
      if (isa<BlockAddress>(FnUser))
        continue;
      if (const auto *Call = dyn_cast<CallBase>(FnUser)) {
        if (!Call->isCallee(&U))
          return true;
      } else if (isa<Instruction>(FnUser)) {
        // Consider any other instruction to be an escape. This has some weird
        // consequences like no-op intrinsics being an escape or a store *to* a
        // function address being an escape.
        return true;
      } else if (const auto *C = dyn_cast<Constant>(FnUser)) {
        // If this is a constant pointer cast of the function, don't consider
        // this escape. Analyze the uses of the cast as well. This ensures that
        // direct calls with mismatched prototypes don't end up in the CFG
        // table. Consider other constants, such as vtable initializers, to
        // escape the function.
        if (C->stripPointerCasts() == F)
          Users.push_back(FnUser);
        else
          return true;
      }
    }
  }
  return false;
}

/// Returns true if this function should be added to the Guard Address Taken IAT
/// Entry Table (GIATs) instead of the Guard Function ID Table (GFIDs).
static bool isIATAddressTaken(const Function *F) {
  if (F->hasDLLImportStorageClass()) {
    return true;
  }
  return false;
}

void WinCFGuard::endModule() {
  const Module *M = Asm->MMI->getModule();
  std::vector<const Function *> GFIDsEntries;
  std::vector<const Function *> GIATsEntries;
  for (const Function &F : *M) {
    if (isPossibleIndirectCallTarget(&F)) {
      if (isIATAddressTaken(&F)) {
        // If the possible call target is reached via the IAT, add it to the
        // GIATs table instead of the GFIDs table.
        GIATsEntries.push_back(&F);
      } else {
        // Otherwise add it to the GFIDs table.
        GFIDsEntries.push_back(&F);
      }
    }
  }

  if (GFIDsEntries.empty() && GIATsEntries.empty() && LongjmpTargets.empty())
    return;

  // Emit the symbol index of each GFIDs entry to form the GFIDs table.
  auto &OS = *Asm->OutStreamer;
  OS.SwitchSection(Asm->OutContext.getObjectFileInfo()->getGFIDsSection());
  for (const Function *F : GFIDsEntries)
    OS.EmitCOFFSymbolIndex(Asm->getSymbol(F));

  // Emit the symbol index of each GIATs entry to form the GIATs table.
  OS.SwitchSection(Asm->OutContext.getObjectFileInfo()->getGIATsSection());
  for (const Function *F : GIATsEntries) {
    OS.EmitCOFFSymbolIndex(Asm->OutContext.getOrCreateSymbol(
        Twine("__imp_") + Asm->getSymbol(F)->getName()));
  }

  // Emit the symbol index of each longjmp target to form the GLJMP table.
  OS.SwitchSection(Asm->OutContext.getObjectFileInfo()->getGLJMPSection());
  for (const MCSymbol *S : LongjmpTargets) {
    OS.EmitCOFFSymbolIndex(S);
  }
}
