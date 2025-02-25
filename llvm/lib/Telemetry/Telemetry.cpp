//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the basic framework for Telemetry.
/// Refer to its documentation at llvm/docs/Telemetry.rst for more details.
//===---------------------------------------------------------------------===//

#include "llvm/Telemetry/Telemetry.h"

namespace llvm {
namespace telemetry {

void TelemetryInfo::serialize(Serializer &serializer) const {
  serializer.write("SessionId", SessionId);
}

Error Manager::dispatch(TelemetryInfo *Entry) {
  if (Error Err = preDispatch(Entry))
    return Err;

  Error AllErrs = Error::success();
  for (auto &Dest : Destinations) {
    AllErrs = joinErrors(std::move(AllErrs), Dest->receiveEntry(Entry));
  }
  return AllErrs;
}

void Manager::addDestination(std::unique_ptr<Destination> Dest) {
  Destinations.push_back(std::move(Dest));
}

Error Manager::preDispatch(TelemetryInfo *Entry) { return Error::success(); }

} // namespace telemetry
} // namespace llvm
