//=====- NVPTXTargetStreamer.cpp - NVPTXTargetStreamer class ------------=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the NVPTXTargetStreamer class.
//
//===----------------------------------------------------------------------===//

#include "NVPTXTargetStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"

using namespace llvm;

//
// NVPTXTargetStreamer Implemenation
//
NVPTXTargetStreamer::NVPTXTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

NVPTXTargetStreamer::~NVPTXTargetStreamer() = default;

void NVPTXTargetStreamer::emitDwarfFileDirective(StringRef Directive) {
  DwarfFiles.emplace_back(Directive);
}

static bool isDwarfSection(const MCObjectFileInfo *FI,
                           const MCSection *Section) {
  // FIXME: the checks for the DWARF sections are very fragile and should be
  // fixed up in a followup patch.
  if (!Section || Section->getKind().isText() ||
      Section->getKind().isWriteable())
    return false;
  return Section == FI->getDwarfAbbrevSection() ||
         Section == FI->getDwarfInfoSection() ||
         Section == FI->getDwarfMacinfoSection() ||
         Section == FI->getDwarfFrameSection() ||
         Section == FI->getDwarfAddrSection() ||
         Section == FI->getDwarfRangesSection() ||
         Section == FI->getDwarfARangesSection() ||
         Section == FI->getDwarfLocSection() ||
         Section == FI->getDwarfStrSection() ||
         Section == FI->getDwarfLineSection() ||
         Section == FI->getDwarfStrOffSection() ||
         Section == FI->getDwarfLineStrSection() ||
         Section == FI->getDwarfPubNamesSection() ||
         Section == FI->getDwarfPubTypesSection() ||
         Section == FI->getDwarfSwiftASTSection() ||
         Section == FI->getDwarfTypesDWOSection() ||
         Section == FI->getDwarfAbbrevDWOSection() ||
         Section == FI->getDwarfAccelObjCSection() ||
         Section == FI->getDwarfAccelNamesSection() ||
         Section == FI->getDwarfAccelTypesSection() ||
         Section == FI->getDwarfAccelNamespaceSection() ||
         Section == FI->getDwarfLocDWOSection() ||
         Section == FI->getDwarfStrDWOSection() ||
         Section == FI->getDwarfCUIndexSection() ||
         Section == FI->getDwarfInfoDWOSection() ||
         Section == FI->getDwarfLineDWOSection() ||
         Section == FI->getDwarfTUIndexSection() ||
         Section == FI->getDwarfStrOffDWOSection() ||
         Section == FI->getDwarfDebugNamesSection() ||
         Section == FI->getDwarfDebugInlineSection() ||
         Section == FI->getDwarfGnuPubNamesSection() ||
         Section == FI->getDwarfGnuPubTypesSection();
}

void NVPTXTargetStreamer::changeSection(const MCSection *CurSection,
                                        MCSection *Section,
                                        const MCExpr *SubSection,
                                        raw_ostream &OS) {
  assert(!SubSection && "SubSection is not null!");
  const MCObjectFileInfo *FI = getStreamer().getContext().getObjectFileInfo();
  // FIXME: remove comment once debug info is properly supported.
  // Emit closing brace for DWARF sections only.
  if (isDwarfSection(FI, CurSection))
    OS << "//\t}\n";
  if (isDwarfSection(FI, Section)) {
    // Emit DWARF .file directives in the outermost scope.
    for (const std::string &S : DwarfFiles)
      getStreamer().EmitRawText(S.data());
    DwarfFiles.clear();
    OS << "//\t.section";
    Section->PrintSwitchToSection(*getStreamer().getContext().getAsmInfo(),
                                  FI->getTargetTriple(), OS, SubSection);
    // DWARF sections are enclosed into braces - emit the open one.
    OS << "//\t{\n";
  }
}
