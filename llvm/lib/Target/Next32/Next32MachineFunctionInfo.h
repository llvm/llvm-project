//==- Next32MachineFunctionInfo.h - Next32 machine function info -*- C++ --*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares Next32-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_NEXT32MACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_NEXT32_NEXT32MACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// Next32FunctionInfo - This class is derived from MachineFunctionInfo and
/// contains private Next32-specific information for each MachineFunction.
class Next32MachineFunctionInfo : public MachineFunctionInfo {
  /// HasTopLevelStackFrame - True if a top level stack frame was created
  /// on function prologue
  bool HasTopLevelStackFrame = false;

public:
  Next32MachineFunctionInfo() = default;
  explicit Next32MachineFunctionInfo(const Function &F,
                                     const TargetSubtargetInfo *STI) {}

  bool hasTopLevelStackFrame() const;
  void setHasTopLevelStackFrame(bool s = true);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_NEXT32_NEXT32MACHINEFUNCTIONINFO_H
