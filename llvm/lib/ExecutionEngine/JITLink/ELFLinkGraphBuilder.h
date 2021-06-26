//===------- ELFLinkGraphBuilder.h - ELF LinkGraph builder ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic ELF LinkGraph building code.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_EXECUTIONENGINE_JITLINK_ELFLINKGRAPHBUILDER_H
#define LIB_EXECUTIONENGINE_JITLINK_ELFLINKGRAPHBUILDER_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace jitlink {

/// Common link-graph building code shared between all ELFFiles.
class ELFLinkGraphBuilderBase {
public:
  virtual ~ELFLinkGraphBuilderBase();
};

/// Ling-graph building code that's specific to the given ELFT, but common
/// across all architectures.
template <typename ELFT>
class ELFLinkGraphBuilder : public ELFLinkGraphBuilderBase {
public:
  ELFLinkGraphBuilder(const object::ELFFile<ELFT> &Obj, Triple TT,
                      StringRef FileName,
                      LinkGraph::GetEdgeKindNameFunction GetEdgeKindName);

protected:
  std::unique_ptr<LinkGraph> G;
  const object::ELFFile<ELFT> &Obj;
};

template <typename ELFT>
ELFLinkGraphBuilder<ELFT>::ELFLinkGraphBuilder(
    const object::ELFFile<ELFT> &Obj, Triple TT, StringRef FileName,
    LinkGraph::GetEdgeKindNameFunction GetEdgeKindName)
    : G(std::make_unique<LinkGraph>(FileName.str(), Triple(std::move(TT)),
                                    ELFT::Is64Bits ? 8 : 4,
                                    support::endianness(ELFT::TargetEndianness),
                                    std::move(GetEdgeKindName))),
      Obj(Obj) {}

} // end namespace jitlink
} // end namespace llvm

#endif // LIB_EXECUTIONENGINE_JITLINK_ELFLINKGRAPHBUILDER_H
