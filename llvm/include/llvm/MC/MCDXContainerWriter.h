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

class LLVM_ABI MCDXContainerTargetWriter : public MCObjectTargetWriter {
protected:
  MCDXContainerTargetWriter() {}

public:
  ~MCDXContainerTargetWriter() override;

  Triple::ObjectFormatType getFormat() const override {
    return Triple::DXContainer;
  }
  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::DXContainer;
  }
};

/// Contains PDB output file name.
static constexpr StringLiteral PdbFileNameSectionName = "PDBNAME";
/// Contains module hash.
static constexpr StringLiteral ModuleHashSectionName = "PDBHASH";

struct MCDXContainerPart {
  StringRef Name;
  StringRef Data;
};

class MCDXContainerBaseWriter {
protected:
  virtual ArrayRef<MCDXContainerPart> getParts() {
    llvm_unreachable("Unimplemented");
  }

  virtual bool shouldSkipSection(StringRef SectionName, size_t SectionSize) {
    // Skip empty and auxiliary sections.
    return SectionSize == 0 || SectionName == PdbFileNameSectionName ||
           SectionName == ModuleHashSectionName;
  }

public:
  MCDXContainerBaseWriter() {}
  virtual ~MCDXContainerBaseWriter();

  void write(raw_ostream &OS, const Triple &TT);
};

class LLVM_ABI DXContainerObjectWriter final : public MCDXContainerBaseWriter,
                                               public MCObjectWriter {
  support::endian::Writer W;
  std::unique_ptr<MCDXContainerTargetWriter> TargetObjectWriter;
  SmallVector<MCDXContainerPart> Parts;
  SmallVector<SmallString<0>> SectionBuffers;

  void clearParts();

protected:
  ArrayRef<MCDXContainerPart> getParts() override;

public:
  DXContainerObjectWriter(std::unique_ptr<MCDXContainerTargetWriter> MOTW,
                          raw_pwrite_stream &OS)
      : W(OS, llvm::endianness::little), TargetObjectWriter(std::move(MOTW)) {}

  uint64_t writeObject() override;
};

} // end namespace llvm

#endif // LLVM_MC_MCDXCONTAINERWRITER_H
