//===-- NVPTXutil.h - Functions exported to CodeGen --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the functions that can be used in CodeGen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_NVPTX_UTIL_H
#define LLVM_TARGET_NVPTX_UTIL_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"

namespace llvm {
bool isParamLoad(const MachineInstr *);
uint64_t encode_leb128(const char *str);
}

#endif
