//===-- ParasolELFObjectWriter.cpp - Parasol ELF Writer -------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

// #include "MCTargetDesc/ParasolFixupKinds.h"
// #include "MCTargetDesc/ParasolMCExpr.h"
#include "MCTargetDesc/ParasolMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

// Update this number whenever the instruction format changes
#define ABI_VERSION 1

namespace {
class ParasolELFObjectWriter : public MCELFObjectTargetWriter {
public:
  ParasolELFObjectWriter(bool Is64Bit, uint8_t OSABI)
      : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_Parasol,
                                /*HasRelocationAddend*/ true, ABI_VERSION) {}

  ~ParasolELFObjectWriter() override = default;

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;

  bool needsRelocateWithSymbol(const MCValue &Val, const MCSymbol &Sym,
                               unsigned Type) const override;
};
} // namespace

unsigned ParasolELFObjectWriter::getRelocType(MCContext &Ctx,
                                              const MCValue &Target,
                                              const MCFixup &Fixup,
                                              bool IsPCRel) const {
  MCFixupKind Kind = Fixup.getKind();
  if (Kind >= FirstLiteralRelocationKind)
    return Kind - FirstLiteralRelocationKind;

  return 0;
}

bool ParasolELFObjectWriter::needsRelocateWithSymbol(const MCValue &,
                                                     const MCSymbol &,
                                                     unsigned Type) const {
  return false;
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createParasolELFObjectWriter(bool Is64Bit, uint8_t OSABI) {
  return std::make_unique<ParasolELFObjectWriter>(Is64Bit, OSABI);
}
