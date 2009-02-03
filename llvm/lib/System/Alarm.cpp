//===- Alarm.cpp - Alarm Generation Support ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Alarm functionality 
//
//===----------------------------------------------------------------------===//

#include "llvm/System/Alarm.h"
#include "llvm/Config/config.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Alarm.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Win32/Alarm.inc"
#endif
