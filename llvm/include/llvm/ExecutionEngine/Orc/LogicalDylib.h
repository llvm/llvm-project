//===--- LogicalDylib.h - Simulates dylib-style symbol lookup ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simulates symbol resolution inside a dylib.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LOGICALDYLIB_H
#define LLVM_EXECUTIONENGINE_ORC_LOGICALDYLIB_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include <string>
#include <vector>

namespace llvm {
namespace orc {

template <typename BaseLayerT,
          typename LogicalModuleResources,
          typename LogicalDylibResources>
class LogicalDylib {
public:
  typedef typename BaseLayerT::ModuleSetHandleT BaseLayerModuleSetHandleT;
private:

  typedef std::vector<BaseLayerModuleSetHandleT> BaseLayerHandleList;

  struct LogicalModule {
    // Make this move-only to ensure they don't get duplicated across moves of
    // LogicalDylib or anything like that.
    LogicalModule(LogicalModule &&RHS)
        : Resources(std::move(RHS.Resources)),
          BaseLayerHandles(std::move(RHS.BaseLayerHandles)) {}
    LogicalModule() = default;
    LogicalModuleResources Resources;
    BaseLayerHandleList BaseLayerHandles;
  };
  typedef std::vector<LogicalModule> LogicalModuleList;

public:

  typedef typename BaseLayerHandleList::iterator BaseLayerHandleIterator;
  typedef typename LogicalModuleList::size_type LogicalModuleHandle;

  LogicalDylib(BaseLayerT &BaseLayer) : BaseLayer(BaseLayer) {}

  ~LogicalDylib() {
    for (auto &LM : LogicalModules)
      for (auto BLH : LM.BaseLayerHandles)
        BaseLayer.removeModuleSet(BLH);
  }

  // If possible, remove this and ~LogicalDylib once the work in the dtor is
  // moved to members (eg: self-unregistering base layer handles).
  LogicalDylib(LogicalDylib &&RHS)
      : BaseLayer(std::move(RHS.BaseLayer)),
        LogicalModules(std::move(RHS.LogicalModules)),
        DylibResources(std::move(RHS.DylibResources)) {}

  LogicalModuleHandle createLogicalModule() {
    LogicalModules.push_back(LogicalModule());
    return LogicalModules.size() - 1;
  }

  void addToLogicalModule(LogicalModuleHandle LMH,
                          BaseLayerModuleSetHandleT BaseLayerHandle) {
    LogicalModules[LMH].BaseLayerHandles.push_back(BaseLayerHandle);
  }

  LogicalModuleResources& getLogicalModuleResources(LogicalModuleHandle LMH) {
    return LogicalModules[LMH].Resources;
  }

  BaseLayerHandleIterator moduleHandlesBegin(LogicalModuleHandle LMH) {
    return LogicalModules[LMH].BaseLayerHandles.begin();
  }

  BaseLayerHandleIterator moduleHandlesEnd(LogicalModuleHandle LMH) {
    return LogicalModules[LMH].BaseLayerHandles.end();
  }

  JITSymbol findSymbolInLogicalModule(LogicalModuleHandle LMH,
                                      const std::string &Name,
                                      bool ExportedSymbolsOnly) {

    if (auto StubSym =
          LogicalModules[LMH].Resources.findSymbol(Name, ExportedSymbolsOnly))
      return StubSym;

    for (auto BLH : LogicalModules[LMH].BaseLayerHandles)
      if (auto Symbol = BaseLayer.findSymbolIn(BLH, Name, ExportedSymbolsOnly))
        return Symbol;
    return nullptr;
  }

  JITSymbol findSymbolInternally(LogicalModuleHandle LMH,
                                 const std::string &Name) {
    if (auto Symbol = findSymbolInLogicalModule(LMH, Name, false))
      return Symbol;

    for (typename LogicalModuleList::size_type I = 0, E = LogicalModules.size();
         I != E; ++I) {
      if (I != LMH)
        if (auto Symbol = findSymbolInLogicalModule(I, Name, false))
          return Symbol;
    }

    return nullptr;
  }

  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    for (typename LogicalModuleList::size_type I = 0, E = LogicalModules.size();
         I != E; ++I)
      if (auto Sym = findSymbolInLogicalModule(I, Name, ExportedSymbolsOnly))
        return Sym;
    return nullptr;
  }

  LogicalDylibResources& getDylibResources() { return DylibResources; }

  LogicalModuleResources*
  getLogicalModuleResourcesForSymbol(const std::string &Name,
                                     bool ExportedSymbolsOnly) {
    for (typename LogicalModuleList::size_type I = 0, E = LogicalModules.size();
         I != E; ++I)
      if (auto Sym = LogicalModules[I].Resources.findSymbol(Name, ExportedSymbolsOnly))
        return &LogicalModules[I]->Resources;
    return nullptr;
  }

protected:
  BaseLayerT BaseLayer;
  LogicalModuleList LogicalModules;
  LogicalDylibResources DylibResources;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_LOGICALDYLIB_H
