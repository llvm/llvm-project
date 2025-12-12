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

// NVPTX-specific source line recording with inlined_at support.
void NVPTXDwarfDebug::recordSourceLineAndInlinedAt(const MachineInstr &MI,
                                                   unsigned Flags) {
  const DebugLoc &DL = MI.getDebugLoc();
  // Maintain a work list of .loc to be emitted. If we are emitting the
  // inlined_at directive, we might need to emit additional .loc prior
  // to it for the location contained in the inlined_at.
  SmallVector<const DILocation *, 8> WorkList;
  SmallDenseSet<const DILocation *, 8> WorkListSet;
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
    // Check if this has inlined_at information, and if the parent location
    // has not yet been emitted. If already emitted, we don't need to
    // re-emit the parent chain.
    if (IA && !EmittedInlinedAtLocs.contains(IA))
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
    // Mark this location as emitted so we don't re-emit the parent chain
    // for subsequent instructions that share the same inlined_at parent.
    if (EnhancedLineinfo)
      EmittedInlinedAtLocs.insert(Current);
  }
}

// NVPTX-specific debug info initialization.
void NVPTXDwarfDebug::initializeTargetDebugInfo(const MachineFunction &MF) {
  // Clear the set of emitted inlined_at locations for each new function.
  EmittedInlinedAtLocs.clear();
}

// NVPTX-specific source line recording with inlined_at support.
void NVPTXDwarfDebug::recordTargetSourceLine(const MachineInstr &MI,
                                             const DebugLoc &DL,
                                             unsigned Flags) {
  // Call NVPTX-specific implementation that handles inlined_at.
  recordSourceLineAndInlinedAt(MI, Flags);
}
