//===----- FunctionInfo.cpp - ABI Function Information ----------- C++ ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include <optional>

using namespace llvm;
using namespace llvm::abi;

std::unique_ptr<FunctionInfo>
FunctionInfo::create(CallingConv::ID CC, const Type *ReturnType,
                     ArrayRef<const Type *> ArgTypes,
                     std::optional<unsigned> NumRequired) {

  assert(!NumRequired || *NumRequired <= ArgTypes.size());

  void *Buffer = operator new(totalSizeToAlloc<ArgEntry>(ArgTypes.size()));

  // FunctionInfo overloads operator delete, so we can use std::unique_ptr
  // without worrying about sized deallocation of trailing objects.
  std::unique_ptr<FunctionInfo> FI(
      new (Buffer) FunctionInfo(CC, ReturnType, ArgTypes.size(), NumRequired));

  ArgEntry *Args = FI->getTrailingObjects();
  for (unsigned I = 0; I < ArgTypes.size(); ++I)
    new (&Args[I]) ArgEntry(ArgTypes[I]);

  return FI;
}
