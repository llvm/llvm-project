//=== BPFCallingConv.h ----- BPF Calling Convention Routines ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains BPF Calling Convention Routines declaration
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFCALLINGCONV_H
#define LLVM_LIB_TARGET_BPF_BPFCALLINGCONV_H

#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {

bool CC_BPF32(unsigned ValNo, MVT ValVT, MVT LocVT,
              CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
              CCState &State);

bool CC_BPF32(unsigned ValNo, MVT ValVT, MVT LocVT,
              CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
              CCState &State);
bool CC_BPF64(unsigned ValNo, MVT ValVT, MVT LocVT,
              CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
              CCState &State);
bool RetCC_BPF32(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                 CCState &State);
bool RetCC_BPF64(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                 CCState &State);

} // namespace llvm

#endif
