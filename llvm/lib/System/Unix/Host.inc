 //===- llvm/System/Unix/Host.inc --------------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the UNIX Host support.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include <llvm/Config/config.h>
#include "Unix.h"
#include <sys/utsname.h>
#include <string>

using namespace llvm;

std::string llvm::sys::getOSName() {
  struct utsname info;

  if (uname(&info))
    return "";

  return info.sysname;
}

std::string llvm::sys::getOSVersion() {
  struct utsname info;

  if (uname(&info))
    return "";

  return info.release;
}
