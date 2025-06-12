//===----- ABIFunctionInfo.cpp - ABI Function Information --------- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/ABIFunctionInfo.h"

using namespace llvm;
using namespace llvm::abi;

ABIFunctionInfo *ABIFunctionInfo::create(llvm::CallingConv::ID CC,
                                         const Type *ReturnType,
                                         llvm::ArrayRef<const Type *> ArgTypes,
                                         const FunctionABIInfo &ABIInfo,
                                         RequiredArgs Required) {

  assert(!Required.allowsOptionalArgs() ||
         Required.getNumRequiredArgs() <= ArgTypes.size());

  void *Buffer = operator new(
      totalSizeToAlloc<ABIFunctionInfoArgInfo>(ArgTypes.size()));

  ABIFunctionInfo *FI =
      new (Buffer) ABIFunctionInfo(ReturnType, ArgTypes.size());

  FI->ABIInfo = ABIInfo;
  FI->ABIInfo.CC = CC;
  FI->Required = Required;

  auto Args = FI->arguments();
  for (unsigned I = 0; I < ArgTypes.size(); ++I) {
    Args[I].ABIType = ArgTypes[I];
    Args[I].ArgInfo = ABIArgInfo::getDirect();
  }

  return FI;
}
