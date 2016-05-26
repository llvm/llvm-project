//===-- LambdaResolverMM - Redirect symbol lookup via a functor -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//   Defines a RuntimeDyld::SymbolResolver subclass that uses a user-supplied
// functor for symbol resolution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LAMBDARESOLVER_H
#define LLVM_EXECUTIONENGINE_ORC_LAMBDARESOLVER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include <memory>

namespace llvm {
namespace orc {

template <typename DylibLookupFtorT, typename ExternalLookupFtorT>
class LambdaResolver : public RuntimeDyld::SymbolResolver {
public:

  LambdaResolver(DylibLookupFtorT DylibLookupFtor,
                 ExternalLookupFtorT ExternalLookupFtor)
      : DylibLookupFtor(DylibLookupFtor),
        ExternalLookupFtor(ExternalLookupFtor) {}

  RuntimeDyld::SymbolInfo
  findSymbolInLogicalDylib(const std::string &Name) final {
    return DylibLookupFtor(Name);
  }

  RuntimeDyld::SymbolInfo findSymbol(const std::string &Name) final {
    return ExternalLookupFtor(Name);
  }

private:
  DylibLookupFtorT DylibLookupFtor;
  ExternalLookupFtorT ExternalLookupFtor;
};

template <typename DylibLookupFtorT,
          typename ExternalLookupFtorT>
std::unique_ptr<LambdaResolver<DylibLookupFtorT, ExternalLookupFtorT>>
createLambdaResolver(DylibLookupFtorT DylibLookupFtor,
                     ExternalLookupFtorT ExternalLookupFtor) {
  typedef LambdaResolver<DylibLookupFtorT, ExternalLookupFtorT> LR;
  return make_unique<LR>(std::move(DylibLookupFtor),
                         std::move(ExternalLookupFtor));
}

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_LAMBDARESOLVER_H
