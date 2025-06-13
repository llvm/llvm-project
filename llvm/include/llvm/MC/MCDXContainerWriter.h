//===- llvm/MC/MCDXContainerWriter.h - DXContainer Writer -*- C++ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCDXCONTAINERWRITER_H
#define LLVM_MC_MCDXCONTAINERWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

class raw_pwrite_stream;

class MCDXContainerTargetWriter : public MCObjectTargetWriter {
protected:
  MCDXContainerTargetWriter() {}

public:
  virtual ~MCDXContainerTargetWriter();

  Triple::ObjectFormatType getFormat() const override {
    return Triple::DXContainer;
  }
  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::DXContainer;
  }
};

class DXContainerObjectWriter final : public MCObjectWriter {
  support::endian::Writer W;
  std::unique_ptr<MCDXContainerTargetWriter> TargetObjectWriter;

public:
  DXContainerObjectWriter(std::unique_ptr<MCDXContainerTargetWriter> MOTW,
                          raw_pwrite_stream &OS)
      : W(OS, llvm::endianness::little), TargetObjectWriter(std::move(MOTW)) {}

  uint64_t writeObject() override;
};
} // end namespace llvm

#endif // LLVM_MC_MCDXCONTAINERWRITER_H
