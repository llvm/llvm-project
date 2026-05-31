//===- llvm/MC/MCAIRLibObjectWriter.h - AIRLib Writer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCAIRLIBOBJECTWRITER_H
#define LLVM_MC_MCAIRLIBOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

class raw_pwrite_stream;

/// Target-side hook of the AIRLib object writer.
///
/// AIRLib has no per-target encoding knobs today: AIR is the only payload
/// and the metallib container is fully assembled inside the target by the
/// embedder pass. This class exists only to satisfy the
/// `MCObjectTargetWriter` / `MCObjectWriter` split that MC requires and to
/// let `classof` dispatch correctly on `Triple::AIRLib`.
class MCAIRLibTargetWriter : public MCObjectTargetWriter {
protected:
  MCAIRLibTargetWriter() {}

public:
  ~MCAIRLibTargetWriter() override;

  Triple::ObjectFormatType getFormat() const override {
    return Triple::AIRLib;
  }
  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::AIRLib;
  }
};

/// Core AIRLib object writer.
///
/// The AIR `.metallib` container is an opaque, monolithic byte blob. The
/// target's embedder pass serializes the entire metallib into the contents
/// of a section named ".metallib" (via a private GlobalVariable). This
/// writer simply locates that section and writes its bytes verbatim to the
/// output stream.
class AIRLibObjectWriter final : public MCObjectWriter {
  support::endian::Writer W;
  std::unique_ptr<MCAIRLibTargetWriter> TargetObjectWriter;

public:
  AIRLibObjectWriter(std::unique_ptr<MCAIRLibTargetWriter> MOTW,
                       raw_pwrite_stream &OS)
      : W(OS, llvm::endianness::little), TargetObjectWriter(std::move(MOTW)) {}

  uint64_t writeObject() override;
};

} // end namespace llvm

#endif // LLVM_MC_MCAIRLIBOBJECTWRITER_H
