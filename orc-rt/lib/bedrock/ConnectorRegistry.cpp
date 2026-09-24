//===- ConnectorRegistry.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the
// orc-rt/bedrock/ConnectorRegistry.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/ConnectorRegistry.h"

#include "orc-rt-internal/support/StringExtras.h"

#include <cassert>

using namespace orc_rt;

namespace orc_rt {

Error ConnectorRegistry::registerConnector(std::string Transport,
                                           ConnectorFn Connector) noexcept {
  std::scoped_lock<std::mutex> Lock(M);
  if (Connectors.count(Transport))
    return make_error<StringError>(
        (StringOutputStream()
         << "A connector is already registered for transport \"" << Transport
         << "\"")
            .str());
  Connectors[std::move(Transport)] = std::move(Connector);
  return Error::success();
}

Error ConnectorRegistry::connect(GetAttachInfoFn GetAttachInfo,
                                 const ConnectionSpec &CS) noexcept {
  ConnectorFn *Connector = nullptr;
  {
    std::scoped_lock<std::mutex> Lock(M);
    auto I = Connectors.find(CS.transport());
    if (I == Connectors.end())
      return make_error<StringError>((StringOutputStream()
                                      << "In connection spec \"" << CS.str()
                                      << "\", unrecognized transport \""
                                      << CS.transport() << "\"")
                                         .str());
    Connector = &I->second;
  }

  // Run the connector without the lock: it blocks on IO, and may register
  // further connectors or connect again. The pointer stays good because
  // unordered_map does not move its elements on insert, and nothing removes
  // them.
  return (*Connector)(std::move(GetAttachInfo), CS);
}

} // namespace orc_rt
