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
///   This layer immediately compiles each IR module added via addModule to an
/// object file and adds this module file to the layer below, which must
/// implement the object layer concept.
template <typename BaseLayerT, typename CompileFtor>
class IRCompileLayer {
public:

  /// @brief Handle to a compiled module.
  using ModuleHandleT = typename BaseLayerT::ObjHandleT;

  /// @brief Construct an IRCompileLayer with the given BaseLayer, which must
  ///        implement the ObjectLayer concept.
  IRCompileLayer(BaseLayerT &BaseLayer, CompileFtor Compile)
      : BaseLayer(BaseLayer), Compile(std::move(Compile)) {}

  /// @brief Get a reference to the compiler functor.
  CompileFtor& getCompiler() { return Compile; }

  /// @brief Compile the module, and add the resulting object to the base layer
  ///        along with the given memory manager and symbol resolver.
  ///
  /// @return A handle for the added module.
  ModuleHandleT addModule(std::shared_ptr<Module> M,
                          std::shared_ptr<JITSymbolResolver> Resolver) {
    using CompileResult = decltype(Compile(*M));
    auto Obj = std::make_shared<CompileResult>(Compile(*M));
    return BaseLayer.addObject(std::move(Obj), std::move(Resolver));
  }

  /// @brief Remove the module associated with the handle H.
  void removeModule(ModuleHandleT H) { BaseLayer.removeObject(H); }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of the given symbol in compiled module represented
  ///        by the handle H. This call is forwarded to the base layer's
  ///        implementation.
  /// @param H The handle for the module to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given module.
  JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbolIn(H, Name, ExportedSymbolsOnly);
  }

  /// @brief Immediately emit and finalize the module represented by the given
  ///        handle.
  /// @param H Handle for module to emit/finalize.
  void emitAndFinalize(ModuleHandleT H) {
    BaseLayer.emitAndFinalize(H);
  }

private:
  BaseLayerT &BaseLayer;
  CompileFtor Compile;
};

} // end namespace orc

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_IRCOMPILINGLAYER_H
