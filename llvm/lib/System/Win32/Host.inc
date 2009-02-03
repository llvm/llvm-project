//===- llvm/System/Win32/Host.inc -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 Host support.
//
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include <cstdio>
#include <string>

using namespace llvm;

std::string sys::getOSName() {
  return "Windows";
}

std::string sys::getOSVersion() {
  OSVERSIONINFO osvi;

  memset(&osvi, 0, sizeof(osvi));
  osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);

  if (!GetVersionEx(&osvi))
    return "";

  char buf[64];
  sprintf(buf, "%d.%d", (int)osvi.dwMajorVersion, (int)osvi.dwMinorVersion);

  return buf;
}
