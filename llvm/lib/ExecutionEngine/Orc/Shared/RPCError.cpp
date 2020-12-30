//===--------------- RPCError.cpp - RPCERror implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RPC Error type implmentations.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/RPCUtils.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

char llvm::orc::shared::RPCFatalError::ID = 0;
char llvm::orc::shared::ConnectionClosed::ID = 0;
char llvm::orc::shared::ResponseAbandoned::ID = 0;
char llvm::orc::shared::CouldNotNegotiate::ID = 0;

namespace llvm {
namespace orc {
namespace shared {

std::error_code ConnectionClosed::convertToErrorCode() const {
  return orcError(OrcErrorCode::RPCConnectionClosed);
}

void ConnectionClosed::log(raw_ostream &OS) const {
  OS << "RPC connection already closed";
}

std::error_code ResponseAbandoned::convertToErrorCode() const {
  return orcError(OrcErrorCode::RPCResponseAbandoned);
}

void ResponseAbandoned::log(raw_ostream &OS) const {
  OS << "RPC response abandoned";
}

CouldNotNegotiate::CouldNotNegotiate(std::string Signature)
    : Signature(std::move(Signature)) {}

std::error_code CouldNotNegotiate::convertToErrorCode() const {
  return orcError(OrcErrorCode::RPCCouldNotNegotiateFunction);
}

void CouldNotNegotiate::log(raw_ostream &OS) const {
  OS << "Could not negotiate RPC function " << Signature;
}

} // end namespace shared
} // end namespace orc
} // end namespace llvm
