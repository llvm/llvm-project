//===-- X86ELFWriterInfo.h - ELF Writer Info for X86 ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the X86 backend.
//
//===----------------------------------------------------------------------===//

#ifndef X86_ELF_WRITER_INFO_H
#define X86_ELF_WRITER_INFO_H

#include "llvm/Target/TargetELFWriterInfo.h"

namespace llvm {

  class X86ELFWriterInfo : public TargetELFWriterInfo {
  public:
    X86ELFWriterInfo();
    virtual ~X86ELFWriterInfo();
  };

} // end llvm namespace

#endif // X86_ELF_WRITER_INFO_H
