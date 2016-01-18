// WebAssemblyMachineFunctionInfo.h-WebAssembly machine function info-*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares WebAssembly-specific per-machine-function
/// information.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYMACHINEFUNCTIONINFO_H

#include "WebAssemblyRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

namespace llvm {

/// This class is derived from MachineFunctionInfo and contains private
/// WebAssembly-specific information for each MachineFunction.
class WebAssemblyFunctionInfo final : public MachineFunctionInfo {
  MachineFunction &MF;

  unsigned NumArguments;

public:
  explicit WebAssemblyFunctionInfo(MachineFunction &MF)
      : MF(MF), NumArguments(0) {}
  ~WebAssemblyFunctionInfo() override;

  void setNumArguments(unsigned N) { NumArguments = N; }
  unsigned getNumArguments() const { return NumArguments; }
};

} // end namespace llvm

#endif
