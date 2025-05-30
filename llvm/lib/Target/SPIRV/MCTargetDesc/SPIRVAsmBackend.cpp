//===-- SPIRVAsmBackend.cpp - SPIR-V Assembler Backend ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSPIRVObjectWriter.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;

namespace {

class SPIRVAsmBackend : public MCAsmBackend {
public:
  SPIRVAsmBackend(llvm::endianness Endian) : MCAsmBackend(Endian) {}

  void applyFixup(const MCFragment &, const MCFixup &, const MCValue &Target,
                  MutableArrayRef<char> Data, uint64_t Value,
                  bool IsResolved) override {}

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return std::make_unique<MCSPIRVObjectTargetWriter>();
  }

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override {
    return false;
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createSPIRVAsmBackend(const Target &T,
                                          const MCSubtargetInfo &STI,
                                          const MCRegisterInfo &MRI,
                                          const MCTargetOptions &) {
  return new SPIRVAsmBackend(llvm::endianness::little);
}
