//====---- IA64Subtarget.h - Define Subtarget for the IA64 -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the IA64 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef IA64SUBTARGET_H
#define IA64SUBTARGET_H

#include "llvm/Target/TargetSubtarget.h"

namespace llvm {

class IA64Subtarget : public TargetSubtarget {
public:
  IA64Subtarget();
};

} // End llvm namespace

#endif
