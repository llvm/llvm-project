//===- LinkGraphLayer.h - Add LinkGraphs to an ExecutionSession -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LinkGraphLayer and associated utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LINKGRAPHLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_LINKGRAPHLAYER_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <atomic>
#include <memory>

namespace llvm::orc {

class LinkGraphLayer {
public:
  LinkGraphLayer(ExecutionSession &ES) : ES(ES) {}

  virtual ~LinkGraphLayer();

  ExecutionSession &getExecutionSession() { return ES; }

  /// Adds a LinkGraph to the JITDylib for the given ResourceTracker.
  virtual Error add(ResourceTrackerSP RT, std::unique_ptr<jitlink::LinkGraph> G,
                    MaterializationUnit::Interface I);

  /// Adds a LinkGraph to the JITDylib for the given ResourceTracker. The
  /// interface for the graph will be built using getLinkGraphInterface.
  Error add(ResourceTrackerSP RT, std::unique_ptr<jitlink::LinkGraph> G) {
    auto LGI = getInterface(*G);
    return add(std::move(RT), std::move(G), std::move(LGI));
  }

  /// Adds a LinkGraph to the given JITDylib.
  Error add(JITDylib &JD, std::unique_ptr<jitlink::LinkGraph> G,
            MaterializationUnit::Interface I) {
    return add(JD.getDefaultResourceTracker(), std::move(G), std::move(I));
  }

  /// Adds a LinkGraph to the given JITDylib. The interface for the object will
  /// be built using getLinkGraphInterface.
  Error add(JITDylib &JD, std::unique_ptr<jitlink::LinkGraph> G) {
    return add(JD.getDefaultResourceTracker(), std::move(G));
  }

  /// Emit should materialize the given IR.
  virtual void emit(std::unique_ptr<MaterializationResponsibility> R,
                    std::unique_ptr<jitlink::LinkGraph> G) = 0;

  /// Get the interface for the given LinkGraph.
  MaterializationUnit::Interface getInterface(jitlink::LinkGraph &G);

  /// Get the JITSymbolFlags for the given symbol.
  static JITSymbolFlags getJITSymbolFlagsForSymbol(jitlink::Symbol &Sym);

private:
  ExecutionSession &ES;
  std::atomic<uint64_t> Counter{0};
};

/// MaterializationUnit for wrapping LinkGraphs.
class LinkGraphMaterializationUnit : public MaterializationUnit {
public:
  LinkGraphMaterializationUnit(LinkGraphLayer &LGLayer,
                               std::unique_ptr<jitlink::LinkGraph> G,
                               Interface I)
      : MaterializationUnit(I), LGLayer(LGLayer), G(std::move(G)) {}

  LinkGraphMaterializationUnit(LinkGraphLayer &LGLayer,
                               std::unique_ptr<jitlink::LinkGraph> G)
      : MaterializationUnit(LGLayer.getInterface(*G)), LGLayer(LGLayer),
        G(std::move(G)) {}

  StringRef getName() const override;

  void materialize(std::unique_ptr<MaterializationResponsibility> MR) override {
    LGLayer.emit(std::move(MR), std::move(G));
  }

private:
  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override;

  LinkGraphLayer &LGLayer;
  std::unique_ptr<jitlink::LinkGraph> G;
};

inline Error LinkGraphLayer::add(ResourceTrackerSP RT,
                                 std::unique_ptr<jitlink::LinkGraph> G,
                                 MaterializationUnit::Interface I) {
  auto &JD = RT->getJITDylib();

  return JD.define(std::make_unique<LinkGraphMaterializationUnit>(
                       *this, std::move(G), std::move(I)),
                   std::move(RT));
}

} // end namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_LINKGRAPHLAYER_H
