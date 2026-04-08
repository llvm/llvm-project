//===- GsymCreatorV2.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMCREATORV2_H
#define LLVM_DEBUGINFO_GSYM_GSYMCREATORV2_H

#include "llvm/DebugInfo/GSYM/GsymCreator.h"

namespace llvm {
namespace gsym {

/// GsymCreatorV2 emits GSYM V2 data with a GlobalData-based section layout.
class GsymCreatorV2 : public GsymCreator {
  uint64_t calculateHeaderAndTableSize() const override;
  std::unique_ptr<GsymCreator> createNew(bool Quiet) const override;

public:
  GsymCreatorV2(bool Quiet = false) : GsymCreator(Quiet) {}

  uint8_t getAddressOffsetSize() const override;
  uint8_t getStringOffsetSize() const override;
  LLVM_ABI llvm::Error encode(FileWriter &O) const override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMCREATORV2_H
