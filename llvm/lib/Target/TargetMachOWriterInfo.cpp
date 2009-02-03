//===-- llvm/Target/TargetMachOWriterInfo.h - MachO Writer Info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetMachOWriterInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachOWriterInfo.h"
#include "llvm/CodeGen/MachineRelocation.h"
using namespace llvm;

TargetMachOWriterInfo::~TargetMachOWriterInfo() {}

MachineRelocation
TargetMachOWriterInfo::GetJTRelocation(unsigned Offset,
                                       MachineBasicBlock *MBB) const {
  // FIXME: do something about PIC
  return MachineRelocation::getBB(Offset, MachineRelocation::VANILLA, MBB);
}
