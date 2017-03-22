//===-- NativeThreadNetBSD.cpp -------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeThreadNetBSD.h"
#include "NativeRegisterContextNetBSD.h"

#include "NativeProcessNetBSD.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_netbsd;

NativeThreadNetBSD::NativeThreadNetBSD(NativeProcessNetBSD *process,
                                       lldb::tid_t tid)
    : NativeThreadProtocol(process, tid) {}
