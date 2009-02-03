//===-- PPCFrameInfo.h - Define TargetFrameInfo for PowerPC -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_FRAMEINFO_H
#define POWERPC_FRAMEINFO_H

#include "PPC.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class PPCFrameInfo: public TargetFrameInfo {
  const TargetMachine &TM;

public:
  PPCFrameInfo(const TargetMachine &tm, bool LP64)
    : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0), TM(tm) {
  }

  /// getReturnSaveOffset - Return the previous frame offset to save the
  /// return address.
  static unsigned getReturnSaveOffset(bool LP64, bool isMacho) {
    if (isMacho)
      return LP64 ? 16 : 8;
    // For ELF 32 ABI:
    return 4;
  }

  /// getFramePointerSaveOffset - Return the previous frame offset to save the
  /// frame pointer.
  static unsigned getFramePointerSaveOffset(bool LP64, bool isMacho) {
    // For MachO ABI:
    // Use the TOC save slot in the PowerPC linkage area for saving the frame
    // pointer (if needed.)  LLVM does not generate code that uses the TOC (R2
    // is treated as a caller saved register.)
    if (isMacho)
      return LP64 ? 40 : 20;
    
    // For ELF 32 ABI:
    // Save it right before the link register
    return -4U;
  }
  
  /// getLinkageSize - Return the size of the PowerPC ABI linkage area.
  ///
  static unsigned getLinkageSize(bool LP64, bool isMacho) {
    if (isMacho)
      return 6 * (LP64 ? 8 : 4);
    
    // For ELF 32 ABI:
    return 8;
  }

  /// getMinCallArgumentsSize - Return the size of the minium PowerPC ABI
  /// argument area.
  static unsigned getMinCallArgumentsSize(bool LP64, bool isMacho) {
    // For Macho ABI:
    // The prolog code of the callee may store up to 8 GPR argument registers to
    // the stack, allowing va_start to index over them in memory if its varargs.
    // Because we cannot tell if this is needed on the caller side, we have to
    // conservatively assume that it is needed.  As such, make sure we have at
    // least enough stack space for the caller to store the 8 GPRs.
    if (isMacho)
      return 8 * (LP64 ? 8 : 4);
    
    // For ELF 32 ABI:
    // There is no default stack allocated for the 8 first GPR arguments.
    return 0;
  }

  /// getMinCallFrameSize - Return the minimum size a call frame can be using
  /// the PowerPC ABI.
  static unsigned getMinCallFrameSize(bool LP64, bool isMacho) {
    // The call frame needs to be at least big enough for linkage and 8 args.
    return getLinkageSize(LP64, isMacho) +
           getMinCallArgumentsSize(LP64, isMacho);
  }
  
};

} // End llvm namespace

#endif
