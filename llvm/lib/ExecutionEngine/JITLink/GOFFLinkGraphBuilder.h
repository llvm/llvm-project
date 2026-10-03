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

protected:
  GOFFLinkGraphBuilder(const object::GOFFObjectFile &Obj,
                       std::shared_ptr<orc::SymbolStringPool> SSP, Triple TT,
                       SubtargetFeatures Features,
                       LinkGraph::GetEdgeKindNameFunction GetEdgeKindName);
  LinkGraph &getGraph() const { return *G; }

  const object::GOFFObjectFile &getObject() const { return Obj; }

  // Process all sections in the GOFF file.
  Error processSections();

  // Process ESD symbols.
  Error processSymbols();

  // Process all relocations for all sections.
  virtual Error processRelocations() = 0;

  void setGraphSymbol(uint32_t SymIndex, Symbol &Sym) {
    assert(!SymbolMap.contains(SymIndex) && "Duplicate symbol at index");
    SymbolMap[SymIndex] = &Sym;
  }

  Symbol *getGraphSymbol(uint32_t SymIndex) const {
    if (SymbolMap.contains(SymIndex))
      return SymbolMap.at(SymIndex);

    return nullptr;
  }

  void setGraphBlock(uint32_t SecIndex, Section *Section, Block *B,
                     object::SectionRef SectionData) {
    assert(!SectionMap.contains(SecIndex) &&
           "Duplicate section block at index");
    SectionMap[SecIndex] = {Section, B, SectionData};
  }

  Block *getGraphBlock(uint32_t SecIndex) const {
    if (SectionMap.contains(SecIndex))
      return SectionMap.at(SecIndex).Block;

    return nullptr;
  }

  object::GOFFObjectFile::section_iterator_range sections() const {
    return Obj.sections();
  }

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
