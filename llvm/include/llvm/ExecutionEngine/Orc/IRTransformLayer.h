//===- IRTransformLayer.h - Run all IR through a functor --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Run all IR passed in through a user supplied functor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_IRTRANSFORMLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_IRTRANSFORMLAYER_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include <memory>
#include <string>

namespace llvm {
class Module;
namespace orc {

/// @brief IR mutating layer.
///
///   This layer applies a user supplied transform to each module that is added,
/// then adds the transformed module to the layer below.
template <typename BaseLayerT, typename TransformFtor>
class IRTransformLayer {
public:

  /// @brief Handle to a set of added modules.
  using ModuleHandleT = typename BaseLayerT::ModuleHandleT;

  /// @brief Construct an IRTransformLayer with the given BaseLayer
  IRTransformLayer(BaseLayerT &BaseLayer,
                   TransformFtor Transform = TransformFtor())
    : BaseLayer(BaseLayer), Transform(std::move(Transform)) {}

  /// @brief Apply the transform functor to the module, then add the module to
  ///        the layer below, along with the memory manager and symbol resolver.
  ///
  /// @return A handle for the added modules.
  ModuleHandleT addModule(std::shared_ptr<Module> M,
                          std::shared_ptr<JITSymbolResolver> Resolver) {
    return BaseLayer.addModule(Transform(std::move(M)), std::move(Resolver));
  }

  /// @brief Remove the module associated with the handle H.
  void removeModule(ModuleHandleT H) { BaseLayer.removeModule(H); }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of the given symbol in the context of the module
  ///        represented by the handle H. This call is forwarded to the base
  ///        layer's implementation.
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

  /// @brief Access the transform functor directly.
  TransformFtor& getTransform() { return Transform; }

  /// @brief Access the mumate functor directly.
  const TransformFtor& getTransform() const { return Transform; }

private:
  BaseLayerT &BaseLayer;
  TransformFtor Transform;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_IRTRANSFORMLAYER_H
