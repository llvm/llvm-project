//
//===----------------------------------------------------------------------===//
//
// This file implements SQELF object file writer information.
//
//===----------------------------------------------------------------------===//
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/SQELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSQELFObjectWriter.h"
#include "llvm/MC/MCValue.h"
// TODO(fzkaria): Consider removing -- for now we are copying some constants
#include "llvm/BinaryFormat/ELF.h"

using namespace llvm;

class SQELFObjectWriter : public MCObjectWriter {
private:
  raw_pwrite_stream &OS;
  /// The target specific ELF writer instance.
  std::unique_ptr<MCSQELFObjectTargetWriter> TargetObjectWriter;

  BinaryFormat::SQELF::Metadata createMetadata(MCAssembler &Asm);

public:
  SQELFObjectWriter(std::unique_ptr<MCSQELFObjectTargetWriter> MOTW,
                    raw_pwrite_stream &OS)
      : OS(OS), TargetObjectWriter(std::move(MOTW)) {}

  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;

  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override;

  uint64_t writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
};

void SQELFObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                 const MCAsmLayout &Layout) {}

void SQELFObjectWriter::recordRelocation(MCAssembler &Asm,
                                         const MCAsmLayout &Layout,
                                         const MCFragment *Fragment,
                                         const MCFixup &Fixup, MCValue Target,
                                         uint64_t &FixedValue) {}

BinaryFormat::SQELF::Metadata
SQELFObjectWriter::createMetadata(MCAssembler &Asm) {
  const std::string Arch = std::string(
      ELF::convertEMachineToArchName(TargetObjectWriter->getEMachine()));

  return BinaryFormat::SQELF::Metadata{"Relocatable", Arch, ELF::EV_CURRENT};
}

uint64_t SQELFObjectWriter::writeObject(MCAssembler &Asm,
                                        const MCAsmLayout &Layout) {
  BinaryFormat::SQELF::Metadata M = createMetadata(Asm);
  BinaryFormat::SQELF OF{};
  OF.setMetadata(M);

  OS << OF;
  return 0;
}

std::unique_ptr<MCObjectWriter>
llvm::createSQELFObjectWriter(std::unique_ptr<MCSQELFObjectTargetWriter> MOTW,
                              raw_pwrite_stream &OS) {
  return std::make_unique<SQELFObjectWriter>(std::move(MOTW), OS);
}