//===-- RISCVCallingConv.h - RISC-V Custom CC Routines ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the custom routines for the RISC-V Calling Convention.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {

/// RISCVCCAssignFn - This target-specific function extends the default
/// CCValAssign with additional information used to lower RISC-V calling
/// conventions.
typedef bool RISCVCCAssignFn(unsigned ValNo, MVT ValVT, MVT LocVT,
                             CCValAssign::LocInfo LocInfo,
                             ISD::ArgFlagsTy ArgFlags, CCState &State,
                             bool IsRet, Type *OrigTy);

bool CC_RISCV(unsigned ValNo, MVT ValVT, MVT LocVT,
              CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
              CCState &State, bool IsRet, Type *OrigTy);

bool CC_RISCV_FastCC(unsigned ValNo, MVT ValVT, MVT LocVT,
                     CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                     CCState &State, bool IsRet, Type *OrigTy);

bool CC_RISCV_GHC(unsigned ValNo, MVT ValVT, MVT LocVT,
                  CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                  CCState &State);

namespace RISCV {

ArrayRef<MCPhysReg> getArgGPRs(const RISCVABI::ABI ABI);

} // end namespace RISCV

} // end namespace llvm
