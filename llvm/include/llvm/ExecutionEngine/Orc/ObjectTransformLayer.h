//===- ObjectTransformLayer.h - Run all objects through functor -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Run all objects passed in through a user supplied functor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_OBJECTTRANSFORMLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_OBJECTTRANSFORMLAYER_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include <algorithm>
#include <memory>
#include <string>

namespace llvm {
namespace orc {

/// @brief Object mutating layer.
///
///   This layer accepts sets of ObjectFiles (via addObject). It
/// immediately applies the user supplied functor to each object, then adds
/// the set of transformed objects to the layer below.
template <typename BaseLayerT, typename TransformFtor>
class ObjectTransformLayer {
public:
  /// @brief Handle to a set of added objects.
  using ObjHandleT = typename BaseLayerT::ObjHandleT;

  /// @brief Construct an ObjectTransformLayer with the given BaseLayer
  ObjectTransformLayer(BaseLayerT &BaseLayer,
                       TransformFtor Transform = TransformFtor())
      : BaseLayer(BaseLayer), Transform(std::move(Transform)) {}

  /// @brief Apply the transform functor to each object in the object set, then
  ///        add the resulting set of objects to the base layer, along with the
  ///        memory manager and symbol resolver.
  ///
  /// @return A handle for the added objects.
  template <typename ObjectPtr>
  Expected<ObjHandleT> addObject(ObjectPtr Obj,
                                 std::shared_ptr<JITSymbolResolver> Resolver) {
    return BaseLayer.addObject(Transform(std::move(Obj)), std::move(Resolver));
  }

  /// @brief Remove the object set associated with the handle H.
  Error removeObject(ObjHandleT H) { return BaseLayer.removeObject(H); }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of the given symbol in the context of the set of
  ///        objects represented by the handle H. This call is forwarded to the
  ///        base layer's implementation.
  /// @param H The handle for the object set to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given object set.
  JITSymbol findSymbolIn(ObjHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbolIn(H, Name, ExportedSymbolsOnly);
  }

  /// @brief Immediately emit and finalize the object set represented by the
  ///        given handle.
  /// @param H Handle for object set to emit/finalize.
  Error emitAndFinalize(ObjHandleT H) {
    return BaseLayer.emitAndFinalize(H);
  }

  /// @brief Map section addresses for the objects associated with the handle H.
  void mapSectionAddress(ObjHandleT H, const void *LocalAddress,
                         JITTargetAddress TargetAddr) {
    BaseLayer.mapSectionAddress(H, LocalAddress, TargetAddr);
  }

  /// @brief Access the transform functor directly.
  TransformFtor &getTransform() { return Transform; }

  /// @brief Access the mumate functor directly.
  const TransformFtor &getTransform() const { return Transform; }

private:
  BaseLayerT &BaseLayer;
  TransformFtor Transform;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_OBJECTTRANSFORMLAYER_H
