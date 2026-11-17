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
#include "llvm/MC/MCDXContainerWriter.h"

using namespace llvm;

cl::opt<bool> dxil::EmbedDebug("dx-embed-debug",
                               cl::desc("Embed PDB in shader container"));
cl::opt<bool> dxil::SlimDebug("dx-slim-debug",
                              cl::desc("Generate slim PDB without ILDB part"));
cl::opt<std::string> dxil::PdbDebugPath(
    "dx-pdb-path",
    cl::desc("Write debug information to the given file, or automatically "
             "named file in directory when ending in '/'"),
    cl::value_desc("filename"));

static cl::opt<bool>
    StripDebug("dx-strip-debug",
               cl::desc("Strip debug information from shader bytecode"));

namespace {
class DirectXContainerObjectWriter : public MCDXContainerTargetWriter {
public:
  DirectXContainerObjectWriter() : MCDXContainerTargetWriter() {}

  // The ILDB part, present when the module has debug info, is embedded unless
  // it is stripped or written to a PDB. -dx-embed-debug overrides the latter
  // two, and slim debug omits ILDB from every output.
  bool shouldSkipSection(StringRef SectionName) const override {
    return SectionName == "ILDB" &&
           (dxil::SlimDebug ||
            (!dxil::EmbedDebug && (StripDebug || !dxil::PdbDebugPath.empty())));
  }
};
} // namespace

std::unique_ptr<MCObjectTargetWriter>
llvm::createDXContainerTargetObjectWriter() {
  return std::make_unique<DirectXContainerObjectWriter>();
}
