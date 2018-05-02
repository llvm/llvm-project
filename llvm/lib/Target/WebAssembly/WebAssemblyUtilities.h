//===-- WebAssemblyUtilities - WebAssembly Utility Functions ---*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the WebAssembly-specific
/// utility functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYUTILITIES_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYUTILITIES_H

namespace llvm {

class MachineBasicBlock;
class MachineInstr;
class MachineLoop;
class WebAssemblyFunctionInfo;

namespace WebAssembly {

bool isArgument(const MachineInstr &MI);
bool isCopy(const MachineInstr &MI);
bool isTee(const MachineInstr &MI);
bool isChild(const MachineInstr &MI, const WebAssemblyFunctionInfo &MFI);
bool isCallIndirect(const MachineInstr &MI);

} // end namespace WebAssembly

/// Return the "bottom" block of a loop. This differs from
/// MachineLoop::getBottomBlock in that it works even if the loop is
/// discontiguous.
MachineBasicBlock *LoopBottom(const MachineLoop *Loop);

} // end namespace llvm

#endif
