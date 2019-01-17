//=== X86CallingConv.h - X86 Custom Calling Convention Routines -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the X86 Calling Convention that
// aren't done by tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86CALLINGCONV_H
#define LLVM_LIB_TARGET_X86_X86CALLINGCONV_H

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/IR/CallingConv.h"

namespace llvm {

/// When regcall calling convention compiled to 32 bit arch, special treatment
/// is required for 64 bit masks.
/// The value should be assigned to two GPRs.
/// \return true if registers were allocated and false otherwise.
bool CC_X86_32_RegCall_Assign2Regs(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                   CCValAssign::LocInfo &LocInfo,
                                   ISD::ArgFlagsTy &ArgFlags, CCState &State);

/// Vectorcall calling convention has special handling for vector types or
/// HVA for 64 bit arch.
/// For HVAs shadow registers might be allocated on the first pass
/// and actual XMM registers are allocated on the second pass.
/// For vector types, actual XMM registers are allocated on the first pass.
/// \return true if registers were allocated and false otherwise.
bool CC_X86_64_VectorCall(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                          CCValAssign::LocInfo &LocInfo,
                          ISD::ArgFlagsTy &ArgFlags, CCState &State);

/// Vectorcall calling convention has special handling for vector types or
/// HVA for 32 bit arch.
/// For HVAs actual XMM registers are allocated on the second pass.
/// For vector types, actual XMM registers are allocated on the first pass.
/// \return true if registers were allocated and false otherwise.
bool CC_X86_32_VectorCall(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                          CCValAssign::LocInfo &LocInfo,
                          ISD::ArgFlagsTy &ArgFlags, CCState &State);

inline bool CC_X86_AnyReg_Error(unsigned &, MVT &, MVT &,
                                CCValAssign::LocInfo &, ISD::ArgFlagsTy &,
                                CCState &) {
  llvm_unreachable("The AnyReg calling convention is only supported by the " \
                   "stackmap and patchpoint intrinsics.");
  // gracefully fallback to X86 C calling convention on Release builds.
  return false;
}

bool CC_X86_32_MCUInReg(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                        CCValAssign::LocInfo &LocInfo,
                        ISD::ArgFlagsTy &ArgFlags, CCState &State);

} // End llvm namespace

#endif

