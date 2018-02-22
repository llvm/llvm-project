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
#include "llvm/ExecutionEngine/Orc/Core.h"
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

  /// @brief Construct an IRCompileLayer with the given BaseLayer, which must
  ///        implement the ObjectLayer concept.
  IRCompileLayer(BaseLayerT &BaseLayer, CompileFtor Compile)
      : BaseLayer(BaseLayer), Compile(std::move(Compile)) {}

  /// @brief Get a reference to the compiler functor.
  CompileFtor& getCompiler() { return Compile; }

  /// @brief Compile the module, and add the resulting object to the base layer
  ///        along with the given memory manager and symbol resolver.
  Error addModule(VModuleKey K, std::shared_ptr<Module> M) {
    return BaseLayer.addObject(std::move(K), Compile(*M));
  }

  /// @brief Remove the module associated with the VModuleKey K.
  Error removeModule(VModuleKey K) { return BaseLayer.removeObject(K); }

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
  /// @param K The VModuleKey for the module to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given module.
  JITSymbol findSymbolIn(VModuleKey K, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbolIn(K, Name, ExportedSymbolsOnly);
  }

  /// @brief Immediately emit and finalize the module represented by the given
  ///        handle.
  /// @param K The VModuleKey for the module to emit/finalize.
  Error emitAndFinalize(VModuleKey K) { return BaseLayer.emitAndFinalize(K); }

private:
  BaseLayerT &BaseLayer;
  CompileFtor Compile;
};

} // end namespace orc

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_IRCOMPILINGLAYER_H
