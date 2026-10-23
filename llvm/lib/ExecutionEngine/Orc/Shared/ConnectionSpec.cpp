//===- ConnectionSpec.cpp - Connection spec parsing -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/ConnectionSpec.h"

using namespace llvm;
using namespace llvm::orc;

static Error makeConnectionSpecError(StringRef Spec, const Twine &Msg) {
  return make_error<StringError>("in connection spec '" + Spec.str() +
                                     "': " + Msg,
                                 inconvertibleErrorCode());
}

Expected<ConnectionSpec> ConnectionSpec::parse(StringRef Spec) {
  if (!Spec.contains('='))
    return makeConnectionSpecError(
        Spec, "expected '<transport>[:<action>]=<descriptor>', but found "
              "no '='");

  auto [LHS, Descriptor] = Spec.split('=');
  auto [Transport, Action] = LHS.split(':');

  if (Transport.empty())
    return makeConnectionSpecError(Spec, "empty transport name");

  // An action is optional, but writing the ':' that introduces one and then
  // omitting it is an error.
  if (LHS.contains(':') && Action.empty())
    return makeConnectionSpecError(Spec, "empty action name");

  return ConnectionSpec(Transport.str(), Action.str(), Descriptor.str());
}
