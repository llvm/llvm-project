//= AArch64WinCOFFObjectWriter.cpp - AArch64 Windows COFF Object Writer C++ =//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "MCTargetDesc/AArch64FixupKinds.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWinCOFFObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

namespace {

class AArch64WinCOFFObjectWriter : public MCWinCOFFObjectTargetWriter {
public:
  AArch64WinCOFFObjectWriter()
    : MCWinCOFFObjectTargetWriter(COFF::IMAGE_FILE_MACHINE_ARM64) {
  }

  ~AArch64WinCOFFObjectWriter() override = default;

  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsCrossSection,
                        const MCAsmBackend &MAB) const override;

   bool recordRelocation(const MCFixup &) const override;
};

} // end anonymous namespace

unsigned
AArch64WinCOFFObjectWriter::getRelocType(MCContext &Ctx,
                                         const MCValue &Target,
                                         const MCFixup &Fixup,
                                         bool IsCrossSection,
                                         const MCAsmBackend &MAB) const {
  const MCFixupKindInfo &Info = MAB.getFixupKindInfo(Fixup.getKind());
  report_fatal_error(Twine("unsupported relocation type: ") + Info.Name);
}

bool AArch64WinCOFFObjectWriter::recordRelocation(const MCFixup &Fixup) const {
  return true;
}

namespace llvm {

MCObjectWriter *createAArch64WinCOFFObjectWriter(raw_pwrite_stream &OS) {
  MCWinCOFFObjectTargetWriter *MOTW = new AArch64WinCOFFObjectWriter();
  return createWinCOFFObjectWriter(MOTW, OS);
}

} // end namespace llvm
