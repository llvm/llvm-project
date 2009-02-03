//===- llvm/System/Win32/Mutex.inc - Win32 Mutex Implementation -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 specific (non-pthread) Mutex class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include "llvm/System/Mutex.h"

namespace llvm {
using namespace sys;

Mutex::Mutex(bool /*recursive*/)
{
  data_ = new CRITICAL_SECTION;
  InitializeCriticalSection((LPCRITICAL_SECTION)data_);
}

Mutex::~Mutex()
{
  DeleteCriticalSection((LPCRITICAL_SECTION)data_);
  delete (LPCRITICAL_SECTION)data_;
  data_ = 0;
}

bool 
Mutex::acquire()
{
  EnterCriticalSection((LPCRITICAL_SECTION)data_);
  return true;
}

bool 
Mutex::release()
{
  LeaveCriticalSection((LPCRITICAL_SECTION)data_);
  return true;
}

bool 
Mutex::tryacquire()
{
  return TryEnterCriticalSection((LPCRITICAL_SECTION)data_);
}

}
