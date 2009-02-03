//===-- llvm/Target/TargetELFWriterInfo.h - ELF Writer Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetELFWriterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETELFWRITERINFO_H
#define LLVM_TARGET_TARGETELFWRITERINFO_H

namespace llvm {

  //===--------------------------------------------------------------------===//
  //                          TargetELFWriterInfo
  //===--------------------------------------------------------------------===//

  class TargetELFWriterInfo {
    // EMachine - This field is the target specific value to emit as the
    // e_machine member of the ELF header.
    unsigned short EMachine;
  public:
    enum MachineType {
      NoMachine,
      EM_386 = 3
    };

    explicit TargetELFWriterInfo(MachineType machine) : EMachine(machine) {}
    virtual ~TargetELFWriterInfo() {}

    unsigned short getEMachine() const { return EMachine; }
  };

} // end llvm namespace

#endif // LLVM_TARGET_TARGETELFWRITERINFO_H
