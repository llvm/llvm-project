//===- IRCompileLayer.h -- Eagerly compile IR for JIT -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <string>

namespace llvm {

class Module;

namespace orc {

class IRCompileLayer : public IRLayer {
public:
  class IRCompiler {
  public:
    IRCompiler(IRSymbolMapper::ManglingOptions MO) : MO(std::move(MO)) {}
    virtual ~IRCompiler();
    const IRSymbolMapper::ManglingOptions &getManglingOptions() const {
      return MO;
    }
    virtual Expected<std::unique_ptr<MemoryBuffer>> operator()(Module &M) = 0;

  protected:
    IRSymbolMapper::ManglingOptions &manglingOptions() { return MO; }

  private:
    IRSymbolMapper::ManglingOptions MO;
  };

  using NotifyCompiledFunction =
      std::function<void(VModuleKey K, ThreadSafeModule TSM)>;

  IRCompileLayer(ExecutionSession &ES, ObjectLayer &BaseLayer,
                 std::unique_ptr<IRCompiler> Compile);

  IRCompiler &getCompiler() { return *Compile; }

  void setNotifyCompiled(NotifyCompiledFunction NotifyCompiled);

  void emit(std::unique_ptr<MaterializationResponsibility> R,
            ThreadSafeModule TSM) override;

private:
  mutable std::mutex IRLayerMutex;
  ObjectLayer &BaseLayer;
  std::unique_ptr<IRCompiler> Compile;
  const IRSymbolMapper::ManglingOptions *ManglingOpts;
  NotifyCompiledFunction NotifyCompiled = NotifyCompiledFunction();
};

/// Eager IR compiling layer.
///
///   This layer immediately compiles each IR module added via addModule to an
/// object file and adds this module file to the layer below, which must
/// implement the object layer concept.
template <typename BaseLayerT, typename CompileFtor>
class LegacyIRCompileLayer {
public:
  /// Callback type for notifications when modules are compiled.
  using NotifyCompiledCallback =
      std::function<void(VModuleKey K, std::unique_ptr<Module>)>;

  /// Construct an LegacyIRCompileLayer with the given BaseLayer, which must
  ///        implement the ObjectLayer concept.
  LLVM_ATTRIBUTE_DEPRECATED(
      LegacyIRCompileLayer(
          BaseLayerT &BaseLayer, CompileFtor Compile,
          NotifyCompiledCallback NotifyCompiled = NotifyCompiledCallback()),
      "ORCv1 layers (layers with the 'Legacy' prefix) are deprecated. Please "
      "use "
      "the ORCv2 IRCompileLayer instead");

  /// Legacy layer constructor with deprecation acknowledgement.
  LegacyIRCompileLayer(
      ORCv1DeprecationAcknowledgement, BaseLayerT &BaseLayer,
      CompileFtor Compile,
      NotifyCompiledCallback NotifyCompiled = NotifyCompiledCallback())
      : BaseLayer(BaseLayer), Compile(std::move(Compile)),
        NotifyCompiled(std::move(NotifyCompiled)) {}

  /// Get a reference to the compiler functor.
  CompileFtor& getCompiler() { return Compile; }

  /// (Re)set the NotifyCompiled callback.
  void setNotifyCompiled(NotifyCompiledCallback NotifyCompiled) {
    this->NotifyCompiled = std::move(NotifyCompiled);
  }

  /// Compile the module, and add the resulting object to the base layer
  ///        along with the given memory manager and symbol resolver.
  Error addModule(VModuleKey K, std::unique_ptr<Module> M) {
    auto Obj = Compile(*M);
    if (!Obj)
      return Obj.takeError();
    if (auto Err = BaseLayer.addObject(std::move(K), std::move(*Obj)))
      return Err;
    if (NotifyCompiled)
      NotifyCompiled(std::move(K), std::move(M));
    return Error::success();
  }

  /// Remove the module associated with the VModuleKey K.
  Error removeModule(VModuleKey K) { return BaseLayer.removeObject(K); }

  /// Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// Get the address of the given symbol in compiled module represented
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

  /// Immediately emit and finalize the module represented by the given
  ///        handle.
  /// @param K The VModuleKey for the module to emit/finalize.
  Error emitAndFinalize(VModuleKey K) { return BaseLayer.emitAndFinalize(K); }

private:
  BaseLayerT &BaseLayer;
  CompileFtor Compile;
  NotifyCompiledCallback NotifyCompiled;
};

template <typename BaseLayerT, typename CompileFtor>
LegacyIRCompileLayer<BaseLayerT, CompileFtor>::LegacyIRCompileLayer(
    BaseLayerT &BaseLayer, CompileFtor Compile,
    NotifyCompiledCallback NotifyCompiled)
    : BaseLayer(BaseLayer), Compile(std::move(Compile)),
      NotifyCompiled(std::move(NotifyCompiled)) {}

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_IRCOMPILINGLAYER_H
