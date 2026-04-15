//===- GsymCreatorV1.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMCREATORV1_H
#define LLVM_DEBUGINFO_GSYM_GSYMCREATORV1_H

#include "llvm/DebugInfo/GSYM/GsymCreator.h"
#include "llvm/DebugInfo/GSYM/Header.h"

namespace llvm {
namespace gsym {

class GsymCreatorV1 : public GsymCreator {
  uint64_t calculateHeaderAndTableSize() const override;
  std::unique_ptr<GsymCreator> createNew(bool Quiet) const override {
    return std::make_unique<GsymCreatorV1>(Quiet);
  }

public:
  GsymCreatorV1(bool Quiet = false) : GsymCreator(Quiet) {}

  uint8_t getStringOffsetSize() const override {
    return Header::getStringOffsetSize();
  }
  LLVM_ABI llvm::Error encode(FileWriter &O) const override;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMCREATORV1_H
