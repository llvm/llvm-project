//===-- StreamCallback.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StreamCallback.h"

#include <string>

using namespace lldb_private;

StreamCallback::StreamCallback(lldb::LogOutputCallback callback, void *baton)
    : llvm::raw_ostream(true), m_callback(callback), m_baton(baton) {}

void StreamCallback::write_impl(const char *Ptr, size_t Size) {
  m_callback(std::string(Ptr, Size).c_str(), m_baton);
}

uint64_t StreamCallback::current_pos() const { return 0; }
