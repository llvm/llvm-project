//===-- NVPTXDwarfDebug.cpp - NVPTX DwarfDebug Implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helper functions for NVPTX-specific debug information
// processing.
//
//===----------------------------------------------------------------------===//

#include "NVPTXDwarfDebug.h"
#include "NVPTXSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

// Command line option to control inlined_at enhancement to lineinfo support.
// Valid only when debuginfo emissionkind is DebugDirectivesOnly or
// LineTablesOnly.
static cl::opt<bool> LineInfoWithInlinedAt(
    "line-info-inlined-at",
    cl::desc("Emit line with inlined_at enhancement for NVPTX"), cl::init(true),
    cl::Hidden);

NVPTXDwarfDebug::NVPTXDwarfDebug(AsmPrinter *A) : DwarfDebug(A) {}

// Collect all inlined_at locations for the current function.
void NVPTXDwarfDebug::collectInlinedAtLocations(const MachineFunction &MF) {
  const DISubprogram *SP = MF.getFunction().getSubprogram();
  assert(SP && "expecting valid subprogram here");

  // inlined_at support requires PTX 7.2 or later.
  const NVPTXSubtarget &STI = MF.getSubtarget<NVPTXSubtarget>();
  if (STI.getPTXVersion() < 72)
    return;

  if (!(SP->getUnit()->isDebugDirectivesOnly() ||
        SP->getUnit()->getEmissionKind() == DICompileUnit::LineTablesOnly) ||
      !LineInfoWithInlinedAt) // No enhanced lineinfo, we are done.
    return;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      const DebugLoc &DL = MI.getDebugLoc();
      if (!DL)
        continue;
      const DILocation *InlinedAt = DL.getInlinedAt();
      while (InlinedAt) {
        if (!InlinedAtLocs.insert(InlinedAt).second)
          break;
        InlinedAt = InlinedAt->getInlinedAt();
      }
    }
  }
}

// NVPTX-specific source line recording with inlined_at support.
void NVPTXDwarfDebug::recordSourceLineAndInlinedAt(const MachineInstr &MI,
                                                   unsigned Flags) {
  const DebugLoc &DL = MI.getDebugLoc();
  // Maintain a work list of .loc to be emitted. If we are emitting the
  // inlined_at directive, we might need to emit additional .loc prior
  // to it for the location contained in the inlined_at.
  SmallVector<const DILocation *, 8> WorkList;
  DenseSet<const DILocation *> WorkListSet;
  const DILocation *EmitLoc = DL.get();

  const DISubprogram *SP = MI.getMF()->getFunction().getSubprogram();
  const NVPTXSubtarget &STI = MI.getMF()->getSubtarget<NVPTXSubtarget>();
  const bool EnhancedLineinfo =
      LineInfoWithInlinedAt && (STI.getPTXVersion() >= 72) && SP &&
      (SP->getUnit()->isDebugDirectivesOnly() ||
       SP->getUnit()->getEmissionKind() == DICompileUnit::LineTablesOnly);

  while (EmitLoc) {
    // Get the scope for the current location.
    const DIScope *Scope = EmitLoc->getScope();
    if (!Scope)
      break; // scope is null, we are done.

    // Check if this loc is already in work list, if so, we are done.
    if (WorkListSet.contains(EmitLoc))
      break;

    // Add this location to the work list.
    WorkList.push_back(EmitLoc);
    WorkListSet.insert(EmitLoc);

    if (!EnhancedLineinfo) // No enhanced lineinfo, we are done.
      break;

    const DILocation *IA = EmitLoc->getInlinedAt();
    // Check if this has inlined_at information, and if we have not yet
    // emitted the .loc for the inlined_at location.
    if (IA && InlinedAtLocs.contains(IA))
      EmitLoc = IA;
    else // We are done
      break;
  }

  const unsigned CUID = Asm->OutStreamer->getContext().getDwarfCompileUnitID();
  // Traverse the work list, and emit .loc.
  while (!WorkList.empty()) {
    const DILocation *Current = WorkList.pop_back_val();
    const DIScope *Scope = Current->getScope();

    if (!Scope)
      llvm_unreachable("we shouldn't be here for null scope");

    const DILocation *InlinedAt = Current->getInlinedAt();
    StringRef Fn = Scope->getFilename();
    const unsigned Line = Current->getLine();
    const unsigned Col = Current->getColumn();
    unsigned Discriminator = 0;
    if (Line != 0 && getDwarfVersion() >= 4)
      if (const DILexicalBlockFile *LBF = dyn_cast<DILexicalBlockFile>(Scope))
        Discriminator = LBF->getDiscriminator();

    const unsigned FileNo = static_cast<DwarfCompileUnit &>(*getUnits()[CUID])
                                .getOrCreateSourceID(Scope->getFile());
    // Remove this location from the work list if it is in the inlined_at
    // locations set.
    if (EnhancedLineinfo && InlinedAtLocs.contains(Current))
      InlinedAtLocs.erase(Current);

    if (EnhancedLineinfo && InlinedAt) {
      const unsigned FileIA = static_cast<DwarfCompileUnit &>(*getUnits()[CUID])
                                  .getOrCreateSourceID(InlinedAt->getFile());
      const DISubprogram *SubProgram = getDISubprogram(Current->getScope());
      DwarfStringPoolEntryRef Entry = InfoHolder.getStringPool().getEntry(
          *Asm, SubProgram->getLinkageName());
      Asm->OutStreamer->emitDwarfLocDirectiveWithInlinedAt(
          FileNo, Line, Col, FileIA, InlinedAt->getLine(),
          InlinedAt->getColumn(), Entry.getSymbol(), Flags, 0, Discriminator,
          Fn);
    } else {
      Asm->OutStreamer->emitDwarfLocDirective(FileNo, Line, Col, Flags, 0,
                                              Discriminator, Fn);
    }
  }
}

// NVPTX-specific debug info initialization.
void NVPTXDwarfDebug::initializeTargetDebugInfo(const MachineFunction &MF) {
  InlinedAtLocs.clear();
  collectInlinedAtLocations(MF);
}

// NVPTX-specific source line recording with inlined_at support.
void NVPTXDwarfDebug::recordTargetSourceLine(const MachineInstr &MI,
                                             const DebugLoc &DL,
                                             unsigned Flags) {
  // Call NVPTX-specific implementation that handles inlined_at.
  recordSourceLineAndInlinedAt(MI, Flags);
}
