//===- llvm/CodeGen/PseudoProbePrinter.cpp - Pseudo Probe Emission -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing pseudo probe info into asm files.
//
//===----------------------------------------------------------------------===//

#include "PseudoProbePrinter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PseudoProbe.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

#define DEBUG_TYPE "pseudoprobe"

PseudoProbeHandler::~PseudoProbeHandler() = default;

PseudoProbeHandler::PseudoProbeHandler(AsmPrinter *A, Module *M) : Asm(A) {
  NamedMDNode *FuncInfo = M->getNamedMetadata(PseudoProbeDescMetadataName);
  assert(FuncInfo && "Pseudo probe descriptors are missing");
  for (const auto *Operand : FuncInfo->operands()) {
    const auto *MD = cast<MDNode>(Operand);
    auto GUID =
        mdconst::dyn_extract<ConstantInt>(MD->getOperand(0))->getZExtValue();
    auto Name = cast<MDString>(MD->getOperand(2))->getString();
    // We may see pairs with same name but different GUIDs here in LTO mode, due
    // to static same-named functions inlined from other modules into this
    // module. Function profiles with the same name will be merged no matter
    // whether they are collected on the same function. Therefore we just pick
    // up the last <Name, GUID> pair here to represent the same-named function
    // collection and all probes from the collection will be merged into a
    // single profile eventually.
    Names[Name] = GUID;
  }

  LLVM_DEBUG(dump());
}

void PseudoProbeHandler::emitPseudoProbe(uint64_t Guid, uint64_t Index,
                                         uint64_t Type, uint64_t Attr,
                                         const DILocation *DebugLoc) {
  // Gather all the inlined-at nodes.
  // When it's done ReversedInlineStack looks like ([66, B], [88, A])
  // which means, Function A inlines function B at calliste with a probe id 88,
  // and B inlines C at probe 66 where C is represented by Guid.
  SmallVector<InlineSite, 8> ReversedInlineStack;
  auto *InlinedAt = DebugLoc ? DebugLoc->getInlinedAt() : nullptr;
  while (InlinedAt) {
    const DISubprogram *SP = InlinedAt->getScope()->getSubprogram();
    // Use linkage name for C++ if possible.
    auto Name = SP->getLinkageName();
    if (Name.empty())
      Name = SP->getName();
    assert(Names.count(Name) && "Pseudo probe descriptor missing for function");
    uint64_t CallerGuid = Names[Name];
    uint64_t CallerProbeId = PseudoProbeDwarfDiscriminator::extractProbeIndex(
        InlinedAt->getDiscriminator());
    ReversedInlineStack.emplace_back(CallerGuid, CallerProbeId);
    InlinedAt = InlinedAt->getInlinedAt();
  }

  SmallVector<InlineSite, 8> InlineStack(ReversedInlineStack.rbegin(),
                                         ReversedInlineStack.rend());
  Asm->OutStreamer->emitPseudoProbe(Guid, Index, Type, Attr, InlineStack);
}

#ifndef NDEBUG
void PseudoProbeHandler::dump() const {
  dbgs() << "\n=============================\n";
  dbgs() << "\nFunction Name to GUID map:\n";
  dbgs() << "\n=============================\n";
  for (const auto &Item : Names)
    dbgs() << "Func: " << Item.first << "   GUID: " << Item.second << "\n";
}
#endif
