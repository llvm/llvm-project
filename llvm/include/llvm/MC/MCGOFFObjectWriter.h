//===- MCGOFFObjectWriter.h - GOFF Object Writer ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCGOFFOBJECTWRITER_H
#define LLVM_MC_MCGOFFOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"

namespace llvm {
class MCObjectWriter;
class raw_pwrite_stream;

class MCGOFFObjectTargetWriter : public MCObjectTargetWriter {
protected:
  MCGOFFObjectTargetWriter() = default;

public:
  ~MCGOFFObjectTargetWriter() override = default;

  Triple::ObjectFormatType getFormat() const override { return Triple::GOFF; }

  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::GOFF;
  }
};

class GOFFObjectWriter : public MCObjectWriter {
  // The target specific GOFF writer instance.
  std::unique_ptr<MCGOFFObjectTargetWriter> TargetObjectWriter;

  // The stream used to write the GOFF records.
  raw_pwrite_stream &OS;

public:
  GOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                   raw_pwrite_stream &OS);
  ~GOFFObjectWriter() override;

  // Implementation of the MCObjectWriter interface.
  void recordRelocation(const MCFragment &F, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override {}

  uint64_t writeObject() override;
};

/// \brief Construct a new GOFF writer instance.
///
/// \param MOTW - The target-specific GOFF writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
std::unique_ptr<MCObjectWriter>
createGOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                       raw_pwrite_stream &OS);
} // namespace llvm

#endif
