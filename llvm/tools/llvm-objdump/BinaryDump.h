//===-- BinaryDump.h - raw-binary dumper ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJDUMP_BINARYDUMP_H
#define LLVM_TOOLS_LLVM_OBJDUMP_BINARYDUMP_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {
class Error;
namespace object {
class BinaryObjectFile;
class RelocationRef;
} // namespace object

namespace objdump {

Error getBinaryRelocationValueString(const object::BinaryObjectFile *Obj,
                                     const object::RelocationRef &RelRef,
                                     llvm::SmallVectorImpl<char> &Result);

} // namespace objdump
} // namespace llvm

#endif
