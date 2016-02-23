//===-- ThisThread.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ThisThread.h"

#include <pthread.h>
#include "llvm/ADT/SmallVector.h"

using namespace lldb_private;

void
ThisThread::SetName(llvm::StringRef name)
{
#if defined (__APPLE__)
    ::pthread_setname_np(name.str().c_str());
#endif
}

void
ThisThread::GetName(llvm::SmallVectorImpl<char> &name)
{
    // FIXME - implement this.
}
