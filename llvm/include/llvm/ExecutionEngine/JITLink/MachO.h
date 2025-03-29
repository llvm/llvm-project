//===------- MachO.h - Generic JIT link function for MachO ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic jit-link functions for MachO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_MACHO_H
#define LLVM_EXECUTIONENGINE_JITLINK_MACHO_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Shared/MachOObjectFormat.h"

namespace llvm {
namespace jitlink {

/// Create a LinkGraph from a MachO relocatable object.
///
/// Note: The graph does not take ownership of the underlying buffer, nor copy
/// its contents. The caller is responsible for ensuring that the object buffer
/// outlives the graph.
Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromMachOObject(MemoryBufferRef ObjectBuffer,
                               std::shared_ptr<orc::SymbolStringPool> SSP);

/// jit-link the given ObjBuffer, which must be a MachO object file.
///
/// Uses conservative defaults for GOT and stub handling based on the target
/// platform.
void link_MachO(std::unique_ptr<LinkGraph> G,
                std::unique_ptr<JITLinkContext> Ctx);

/// Get a pointer to the standard MachO data section (creates an empty
/// section with RW- permissions and standard lifetime if one does not
/// already exist).
inline Section &getMachODefaultRWDataSection(LinkGraph &G) {
  if (auto *DataSec = G.findSectionByName(orc::MachODataDataSectionName))
    return *DataSec;
  return G.createSection(orc::MachODataDataSectionName,
                         orc::MemProt::Read | orc::MemProt::Write);
}

/// Get a pointer to the standard MachO text section (creates an empty
/// section with R-X permissions and standard lifetime if one does not
/// already exist).
inline Section &getMachODefaultTextSection(LinkGraph &G) {
  if (auto *TextSec = G.findSectionByName(orc::MachOTextTextSectionName))
    return *TextSec;
  return G.createSection(orc::MachOTextTextSectionName,
                         orc::MemProt::Read | orc::MemProt::Exec);
}

/// Gets or creates a MachO header for the current LinkGraph.
Expected<Symbol &> getOrCreateLocalMachOHeader(LinkGraph &G);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_MACHO_H
