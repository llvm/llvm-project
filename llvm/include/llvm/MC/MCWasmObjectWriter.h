//===-- llvm/MC/MCWasmObjectWriter.h - Wasm Object Writer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWASMOBJECTWRITER_H
#define LLVM_MC_MCWASMOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include <memory>

namespace llvm {

class MCFixup;
class MCValue;
class raw_pwrite_stream;

class MCWasmObjectTargetWriter : public MCObjectTargetWriter {
  const unsigned Is64Bit : 1;

protected:
  explicit MCWasmObjectTargetWriter(bool Is64Bit_);

public:
  virtual ~MCWasmObjectTargetWriter();

  virtual Triple::ObjectFormatType getFormat() const { return Triple::Wasm; }
  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::Wasm;
  }

  virtual unsigned getRelocType(const MCValue &Target,
                                const MCFixup &Fixup) const = 0;

  /// \name Accessors
  /// @{
  bool is64Bit() const { return Is64Bit; }
  /// @}
};

/// Construct a new Wasm writer instance.
///
/// \param MOTW - The target specific Wasm writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
std::unique_ptr<MCObjectWriter>
createWasmObjectWriter(std::unique_ptr<MCWasmObjectTargetWriter> MOTW,
                       raw_pwrite_stream &OS);

} // namespace llvm

#endif
