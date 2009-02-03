//===- llvm/System/Unix/Mutex.inc - Unix Mutex Implementation ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific (non-pthread) Mutex class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

namespace llvm
{
using namespace sys;

Mutex::Mutex( bool recursive)
{
}

Mutex::~Mutex()
{
}

bool 
Mutex::acquire()
{
  return true;
}

bool 
Mutex::release()
{
  return true;
}

bool 
Mutex::tryacquire( void )
{
  return true;
}

}
