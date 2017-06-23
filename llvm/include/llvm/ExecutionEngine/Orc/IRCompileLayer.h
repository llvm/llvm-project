//===- IRCompileLayer.h -- Eagerly compile IR for JIT -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the definition for a basic, eagerly compiling layer of the JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_IRCOMPILELAYER_H
#define LLVM_EXECUTIONENGINE_ORC_IRCOMPILELAYER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <string>

namespace llvm {

class Module;

namespace orc {

/// @brief Eager IR compiling layer.
///
///   This layer accepts sets of LLVM IR Modules (via addModuleSet). It
/// immediately compiles each IR module to an object file (each IR Module is
/// compiled separately). The resulting set of object files is then added to
/// the layer below, which must implement the object layer concept.
template <typename BaseLayerT, typename CompileFtor>
class IRCompileLayer {
public:
  /// @brief Handle to a set of compiled modules.
  using ModuleSetHandleT = typename BaseLayerT::ObjHandleT;

  /// @brief Construct an IRCompileLayer with the given BaseLayer, which must
  ///        implement the ObjectLayer concept.
  IRCompileLayer(BaseLayerT &BaseLayer, CompileFtor Compile)
      : BaseLayer(BaseLayer), Compile(std::move(Compile)) {}

  /// @brief Get a reference to the compiler functor.
  CompileFtor& getCompiler() { return Compile; }

  /// @brief Compile each module in the given module set, then add the resulting
  ///        set of objects to the base layer along with the memory manager and
  ///        symbol resolver.
  ///
  /// @return A handle for the added modules.
  template <typename ModuleSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                MemoryManagerPtrT MemMgr,
                                SymbolResolverPtrT Resolver) {
    assert(Ms.size() == 1);
    using CompileResult = decltype(Compile(*Ms.front()));
    auto Obj = std::make_shared<CompileResult>(Compile(*Ms.front()));
    return BaseLayer.addObject(std::move(Obj), std::move(MemMgr),
                               std::move(Resolver));
  }

  /// @brief Remove the module set associated with the handle H.
  void removeModuleSet(ModuleSetHandleT H) { BaseLayer.removeObject(H); }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of the given symbol in the context of the set of
  ///        compiled modules represented by the handle H. This call is
  ///        forwarded to the base layer's implementation.
  /// @param H The handle for the module set to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given module set.
  JITSymbol findSymbolIn(ModuleSetHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbolIn(H, Name, ExportedSymbolsOnly);
  }

  /// @brief Immediately emit and finalize the moduleOB set represented by the
  ///        given handle.
  /// @param H Handle for module set to emit/finalize.
  void emitAndFinalize(ModuleSetHandleT H) {
    BaseLayer.emitAndFinalize(H);
  }

private:
  BaseLayerT &BaseLayer;
  CompileFtor Compile;
};

} // end namespace orc

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_IRCOMPILINGLAYER_H
