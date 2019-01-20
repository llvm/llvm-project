//===--------------- RPCUtils.cpp - RPCUtils implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RPCUtils implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RPCUtils.h"

char llvm::orc::rpc::RPCFatalError::ID = 0;
char llvm::orc::rpc::ConnectionClosed::ID = 0;
char llvm::orc::rpc::ResponseAbandoned::ID = 0;
char llvm::orc::rpc::CouldNotNegotiate::ID = 0;

namespace llvm {
namespace orc {
namespace rpc {

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


} // end namespace rpc
} // end namespace orc
} // end namespace llvm
