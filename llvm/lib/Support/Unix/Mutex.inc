//===- llvm/Support/Unix/Mutex.inc - Unix Mutex Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

MutexImpl::MutexImpl( bool recursive)
{
}

MutexImpl::~MutexImpl()
{
}

bool
MutexImpl::release()
{
  return true;
}

bool
MutexImpl::tryacquire( void )
{
  return true;
}

}
