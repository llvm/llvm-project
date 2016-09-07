//===-- AbstractSocket.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/linux/AbstractSocket.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

AbstractSocket::AbstractSocket(bool child_processes_inherit, Error &error)
    : DomainSocket(ProtocolUnixAbstract, child_processes_inherit, error) {}

size_t AbstractSocket::GetNameOffset() const { return 1; }

void AbstractSocket::DeleteSocketFile(llvm::StringRef name) {}
