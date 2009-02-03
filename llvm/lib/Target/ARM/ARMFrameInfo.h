//===-- ARMTargetFrameInfo.h - Define TargetFrameInfo for ARM ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef ARM_FRAMEINFO_H
#define ARM_FRAMEINFO_H

#include "ARM.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "ARMSubtarget.h"

namespace llvm {

class ARMFrameInfo : public TargetFrameInfo {
public:
  explicit ARMFrameInfo(const ARMSubtarget &ST)
    : TargetFrameInfo(StackGrowsDown, ST.getStackAlignment(), 0) {
  }
};

} // End llvm namespace

#endif
