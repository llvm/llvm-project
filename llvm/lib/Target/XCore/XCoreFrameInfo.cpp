//===-- XCoreFrameInfo.cpp - Frame info for XCore Target ---------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains XCore frame information that doesn't fit anywhere else
// cleanly...
//
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "XCoreFrameInfo.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// XCoreFrameInfo:
//===----------------------------------------------------------------------===//

XCoreFrameInfo::XCoreFrameInfo(const TargetMachine &tm):
  TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 4, 0)
{
  // Do nothing
}
