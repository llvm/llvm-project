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

/// NVPTX-specific source line recording with inlined_at support.
///
/// Why this exists:
/// NVPTX supports an "enhanced lineinfo" mode where inlining context is carried
/// via line-table directives, rather than full DWARF DIEs. This is conceptually
/// similar to proposals[1] for richer DWARF line tables that carry inline call
/// context and callee identity in the line table. NVPTX implements this via
/// target-specific `.loc` extensions in the PTX ISA[3].
///
/// How it impacts PTX assembly generation:
/// - When enabled (PTX ISA >= 7.2 + line-tables-only / debug-directives-only),
///   we emit multiple consecutive `.loc` directives for a single inlined
///   instruction: the instruction's own location and its `inlined_at` parent
///   chain.
/// - During emission we use `MCStreamer::emitDwarfLocDirectiveWithInlinedAt` to
///   emit an enhanced `.loc` directive[3] that carries the extra
///   `function_name` and `inlined_at` operands in the PTX assembly stream.
///
/// Example (conceptual PTX `.loc` sequence for an inlined callsite):
///   .loc 1 16 3                            // caller location
///   .loc 1  5 3, function_name $L__info_stringN, inlined_at 1 16 3
///                                         // inlined callee location
///   Here, $L__info_stringN is a label (or label+immediate) referring into
///   `.debug_str`.
///
/// How this impacts DWARF :
/// DWARF generation tools that consume this PTX(e.g. ptxas assembler) can use
/// the `inlined_at` and `function_name` operands to extend the DWARF v2
/// line table information.
/// This adds:
/// - a `context` column[2]: the `inlined_at <file> <line> <col>` information
///   populates an inlining "context" (a reference to the parent/callsite row)
///   enabling reconstruction of inline call chains from the line table.
/// - a `function_name` column[2]: the `.loc ... function_name <sym>` identifies
///   the inlined callee associated with a non-zero context.
///
/// References:
/// - [1] DWARF line tables / Two-Level Line Tables:
///   https://wiki.dwarfstd.org/TwoLevelLineTables.md
/// - [2] DWARF issue tracking for Two-Level Line Tables:
///   https://dwarfstd.org/issues/140906.1.html
/// - [3] NVIDIA PTX ISA `.loc` (debugging directives; PTX ISA 7.2+):
///   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#debugging-directives-loc
void NVPTXDwarfDebug::recordTargetSourceLine(const DebugLoc &DL,
                                             unsigned Flags) {
  // Maintain a work list of .loc to be emitted. If we are emitting the
  // inlined_at directive, we might need to emit additional .loc prior
  // to it for the location contained in the inlined_at.
  SmallVector<const DILocation *, 8> WorkList;
  SmallDenseSet<const DILocation *, 8> WorkListSet;
  const DILocation *EmitLoc = DL.get();

  if (!EmitLoc)
    return;

  const MachineFunction *MF = Asm->MF;
  if (!MF)
    return;

  const DISubprogram *SP = MF->getFunction().getSubprogram();
  const NVPTXSubtarget &STI = MF->getSubtarget<NVPTXSubtarget>();
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
    else // We are done.
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

/// NVPTX-specific debug info initialization.
void NVPTXDwarfDebug::initializeTargetDebugInfo(const MachineFunction &MF) {
  // Clear the set of emitted inlined_at locations for each new function.
  EmittedInlinedAtLocs.clear();
}
