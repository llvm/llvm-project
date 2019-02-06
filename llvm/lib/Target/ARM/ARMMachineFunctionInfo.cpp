//===-- ARMMachineFunctionInfo.cpp - ARM machine function info ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"

using namespace llvm;

void ARMFunctionInfo::anchor() {}

ARMFunctionInfo::ARMFunctionInfo(MachineFunction &MF)
    : isThumb(MF.getSubtarget<ARMSubtarget>().isThumb()),
      hasThumb2(MF.getSubtarget<ARMSubtarget>().hasThumb2()) {}
