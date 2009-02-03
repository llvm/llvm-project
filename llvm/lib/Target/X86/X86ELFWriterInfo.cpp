//===-- X86ELFWriterInfo.cpp - ELF Writer Info for the X86 backend --------===//
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

#include "X86ELFWriterInfo.h"
using namespace llvm;

X86ELFWriterInfo::X86ELFWriterInfo() : TargetELFWriterInfo(EM_386) {}
X86ELFWriterInfo::~X86ELFWriterInfo() {}
