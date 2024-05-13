//===-- llvm/MC/MCSPIRVObjectWriter.h - SPIR-V Object Writer -----*- C++ *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSPIRVOBJECTWRITER_H
#define LLVM_MC_MCSPIRVOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {

class MCSPIRVObjectTargetWriter : public MCObjectTargetWriter {
protected:
  explicit MCSPIRVObjectTargetWriter() {}

public:
  Triple::ObjectFormatType getFormat() const override { return Triple::SPIRV; }
  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::SPIRV;
  }
};

/// Construct a new SPIR-V writer instance.
///
/// \param MOTW - The target specific SPIR-V writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
std::unique_ptr<MCObjectWriter>
createSPIRVObjectWriter(std::unique_ptr<MCSPIRVObjectTargetWriter> MOTW,
                        raw_pwrite_stream &OS);

} // namespace llvm

#endif
