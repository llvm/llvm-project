//===-- BinaryDump.cpp - raw-binary dumper ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the raw binary dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "BinaryDump.h"

#include "llvm-objdump.h"
#include "llvm/Object/BinaryObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::object;

namespace {
class BinaryDumper : public objdump::Dumper {
  const BinaryObjectFile &Obj;

public:
  BinaryDumper(const BinaryObjectFile &O) : Dumper(O), Obj(O) {}
};
} // namespace

std::unique_ptr<objdump::Dumper>
objdump::createBinaryDumper(const BinaryObjectFile &Obj) {
  return std::make_unique<BinaryDumper>(Obj);
}

Error objdump::getBinaryRelocationValueString(const BinaryObjectFile *Obj,
                                              const RelocationRef &RelRef,
                                              SmallVectorImpl<char> &Result) {
  return Error::success();
}
