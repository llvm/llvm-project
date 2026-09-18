//===------ GOFFLinkGraphBuilder.h - GOFF LinkGraph builder -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic GOFF LinkGraph building code.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_EXECUTIONENGINE_JITLINK_GOFFLINKGRAPHBUILDER_H
#define LIB_EXECUTIONENGINE_JITLINK_GOFFLINKGRAPHBUILDER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/TargetParser/SubtargetFeature.h"

namespace llvm {
namespace jitlink {

// Builder for GOFF LinkGraphs.
class GOFFLinkGraphBuilder {
public:
  virtual ~GOFFLinkGraphBuilder() = default;
  Expected<std::unique_ptr<LinkGraph>> buildGraph();

public:
  GOFFLinkGraphBuilder(const object::GOFFObjectFile &Obj,
                       std::shared_ptr<orc::SymbolStringPool> SSP, Triple TT,
                       SubtargetFeatures Features,
                       LinkGraph::GetEdgeKindNameFunction GetEdgeKindName);
  LinkGraph &getGraph() const { return *G; }
  const object::GOFFObjectFile &getObject() const { return Obj; }

private:
  // Process all sections in the GOFF file.
  Error processSections();

  // Process ESD symbols.
  Error processSymbols();

  // Process all relocations for all sections.
  Error processRelocations();

private:
  const object::GOFFObjectFile &Obj;
  std::unique_ptr<LinkGraph> G;

  struct SectionEntry {
    jitlink::Section *Section;
    jitlink::Block *Block;
    object::SectionRef SectionData;
  };

  DenseMap<uint32_t, jitlink::Symbol *> SymbolMap;
  DenseMap<uint32_t, SectionEntry> SectionMap;
};

} // namespace jitlink
} // namespace llvm

#endif // LIB_EXECUTIONENGINE_JITLINK_GOFFLINKGRAPHBUILDER_H
