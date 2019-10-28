//===- TableGen.cpp - Top-Level TableGen implementation for Clang ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for all of the LLDB TableGen
// backends. A "TableGen backend" is just a function. See
// "$LLVM_ROOT/utils/TableGen/TableGenBackends.h" for more info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLDB_UTILS_TABLEGEN_TABLEGENBACKENDS_H
#define LLVM_LLDB_UTILS_TABLEGEN_TABLEGENBACKENDS_H

#include <string>

namespace llvm {
class raw_ostream;
class RecordKeeper;
} // namespace llvm

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace lldb_private {

void EmitOptionDefs(RecordKeeper &RK, raw_ostream &OS);

} // namespace lldb_private

#endif
