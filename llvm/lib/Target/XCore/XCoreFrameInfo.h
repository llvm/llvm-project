//===-- XCoreFrameInfo.h - Frame info for XCore Target -----------*- C++ -*-==//
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

#ifndef XCOREFRAMEINFO_H
#define XCOREFRAMEINFO_H

#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class XCoreFrameInfo: public TargetFrameInfo {

  public:
    XCoreFrameInfo(const TargetMachine &tm);

    //! Stack slot size (4 bytes)
    static int stackSlotSize() {
      return 4;
    }
  };
}

#endif // XCOREFRAMEINFO_H
