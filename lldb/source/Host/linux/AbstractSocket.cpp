//===-- AbstractSocket.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/linux/AbstractSocket.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

AbstractSocket::AbstractSocket() : DomainSocket(ProtocolUnixAbstract) {}

AbstractSocket::AbstractSocket(NativeSocket socket, bool should_close)
    : DomainSocket(ProtocolUnixAbstract, socket, should_close) {}

size_t AbstractSocket::GetNameOffset() const { return 1; }

void AbstractSocket::DeleteSocketFile(llvm::StringRef name) {}
