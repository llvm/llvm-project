//===- tools/dsymutil/PseudoProbe.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PseudoProbe.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace dsymutil {

void PseudoProbeCollector::emit(StringRef SecName, StringRef Contents,
                                uint32_t Alignment) {
  MCSection *Section = Streamer.getContext().getMachOSection(
      PseudoProbeSegmentName, SecName, /*TypeAndAttributes=*/0,
      SectionKind::getMetadata());
  Section->setAlignment(Align(Alignment));
  Streamer.switchSection(Section);
  Streamer.emitBytes(Contents);
}

void PseudoProbeCollector::collectFromObject(
    const object::MachOObjectFile &Obj) {
  // The probe sections carry no relocations.
  for (const object::SectionRef &Section : Obj.sections()) {
    Expected<StringRef> NameOrErr =
        Obj.getSectionName(Section.getRawDataRefImpl());
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr != PseudoProbeSectionName &&
        *NameOrErr != PseudoProbeDescSectionName)
      continue;
    Expected<StringRef> ContentsOrErr = Section.getContents();
    if (!ContentsOrErr) {
      consumeError(ContentsOrErr.takeError());
      continue;
    }
    emit(*NameOrErr, *ContentsOrErr, Section.getAlignment().value());
  }
}

} // end namespace dsymutil
} // end namespace llvm
