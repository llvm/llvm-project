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
/// \brief This file contains the declaration of the WebAssembly-specific
/// utility functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYUTILITIES_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYUTILITIES_H

namespace llvm {

class MachineInstr;
class WebAssemblyFunctionInfo;

namespace WebAssembly {

bool isArgument(const MachineInstr &MI);
bool isCopy(const MachineInstr &MI);
bool isTee(const MachineInstr &MI);
bool isChild(const MachineInstr &MI, const WebAssemblyFunctionInfo &MFI);
bool isCallIndirect(const MachineInstr &MI);

} // end namespace WebAssembly
} // end namespace llvm

#endif
