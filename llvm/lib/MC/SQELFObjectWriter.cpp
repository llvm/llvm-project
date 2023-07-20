//
//===----------------------------------------------------------------------===//
//
// This file implements SQELF object file writer information.
//
//===----------------------------------------------------------------------===//
#include "llvm/MC/MCSQELFObjectWriter.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

class SQELFObjectWriter : public MCObjectWriter {
  raw_pwrite_stream &OS;
  /// The target specific ELF writer instance.
  std::unique_ptr<MCSQELFObjectTargetWriter> TargetObjectWriter;

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

uint64_t SQELFObjectWriter::writeObject(MCAssembler &Asm,
                                        const MCAsmLayout &Layout) {
  return 0;
}

std::unique_ptr<MCObjectWriter>
llvm::createSQELFObjectWriter(std::unique_ptr<MCSQELFObjectTargetWriter> MOTW,
                              raw_pwrite_stream &OS) {
  return std::make_unique<SQELFObjectWriter>(std::move(MOTW), OS);
}