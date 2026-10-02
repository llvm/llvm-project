//===-- DirectXContainerObjectWriter.cpp - DX object writer ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains DXContainer object writers for the DirectX backend.
//
//===----------------------------------------------------------------------===//

#include "DirectXContainerObjectWriter.h"
#include "DirectXMCOptions.h"
#include "llvm/MC/MCDXContainerWriter.h"

using namespace llvm;

namespace {
class DirectXContainerObjectWriter : public MCDXContainerTargetWriter {
  const DirectXMCOptions &CLOpts;

public:
  DirectXContainerObjectWriter(const DirectXMCOptions &CLOpts)
      : MCDXContainerTargetWriter(), CLOpts(CLOpts) {}

  // The ILDB part, present when the module has debug info, is embedded unless
  // it is stripped or written to a PDB. -dx-embed-debug overrides the latter
  // two, and slim debug omits ILDB from every output.
  bool shouldSkipSection(StringRef SectionName) const override {
    return SectionName == "ILDB" &&
           (CLOpts.slim_debug ||
            (!CLOpts.embed_debug &&
             (CLOpts.strip_debug || !CLOpts.pdb_path.empty())));
  }
};
} // namespace

std::unique_ptr<MCObjectTargetWriter>
llvm::createDXContainerTargetObjectWriter(const DirectXMCOptions &CLOpts) {
  return std::make_unique<DirectXContainerObjectWriter>(CLOpts);
}
