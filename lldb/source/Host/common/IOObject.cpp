//===-- IOObject.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/IOObject.h"

using namespace lldb_private;

const IOObject::WaitableHandle IOObject::kInvalidHandleValue = -1;
