//===-- DirectXContainerObjectWriter.h - DX object writer ------*- C++ -*--===//
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

#ifndef LLVM_DIRECTX_DIRECTXCONTAINEROBJECTWRITER_H
#define LLVM_DIRECTX_DIRECTXCONTAINEROBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

std::unique_ptr<MCObjectTargetWriter> createDXContainerTargetObjectWriter();

namespace dxil {
extern cl::opt<bool> EmbedDebug;
extern cl::opt<bool> SlimDebug;
extern cl::opt<std::string> PdbDebugPath;
} // namespace dxil
}

#endif // LLVM_DIRECTX_DIRECTXCONTAINEROBJECTWRITER_H
