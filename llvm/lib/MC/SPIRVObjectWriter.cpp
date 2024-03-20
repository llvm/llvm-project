//===- llvm/MC/MCSPIRVObjectWriter.cpp - SPIR-V Object Writer ----*- C++ *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCSPIRVObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;

namespace {
class SPIRVObjectWriter : public MCObjectWriter {
  ::support::endian::Writer W;

  /// The target specific SPIR-V writer instance.
  std::unique_ptr<MCSPIRVObjectTargetWriter> TargetObjectWriter;

public:
  SPIRVObjectWriter(std::unique_ptr<MCSPIRVObjectTargetWriter> MOTW,
                    raw_pwrite_stream &OS)
      : W(OS, llvm::endianness::little), TargetObjectWriter(std::move(MOTW)) {}

  ~SPIRVObjectWriter() override {}

private:
  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override {}

  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override {}

  uint64_t writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
  void writeHeader(const MCAssembler &Asm);
};
} // namespace

void SPIRVObjectWriter::writeHeader(const MCAssembler &Asm) {
  constexpr uint32_t MagicNumber = 0x07230203;
  constexpr uint32_t GeneratorMagicNumber = 0;
  constexpr uint32_t Schema = 0;

  // Construct SPIR-V version and Bound
  const MCAssembler::VersionInfoType &VIT = Asm.getVersionInfo();
  uint32_t VersionNumber = 0 | (VIT.Major << 16) | (VIT.Minor << 8);
  uint32_t Bound = VIT.Update;

  W.write<uint32_t>(MagicNumber);
  W.write<uint32_t>(VersionNumber);
  W.write<uint32_t>(GeneratorMagicNumber);
  W.write<uint32_t>(Bound);
  W.write<uint32_t>(Schema);
}

uint64_t SPIRVObjectWriter::writeObject(MCAssembler &Asm,
                                        const MCAsmLayout &Layout) {
  uint64_t StartOffset = W.OS.tell();
  writeHeader(Asm);
  for (const MCSection &S : Asm)
    Asm.writeSectionData(W.OS, &S, Layout);
  return W.OS.tell() - StartOffset;
}

std::unique_ptr<MCObjectWriter>
llvm::createSPIRVObjectWriter(std::unique_ptr<MCSPIRVObjectTargetWriter> MOTW,
                              raw_pwrite_stream &OS) {
  return std::make_unique<SPIRVObjectWriter>(std::move(MOTW), OS);
}
