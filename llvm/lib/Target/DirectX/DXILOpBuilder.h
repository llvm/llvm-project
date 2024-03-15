//===- DXILOpBuilder.h - Helper class for build DIXLOp functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains class to help build DXIL op functions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILOPBUILDER_H
#define LLVM_LIB_TARGET_DIRECTX_DXILOPBUILDER_H

#include "DXILConstants.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {
class Module;
class IRBuilderBase;
class CallInst;
class Value;
class Type;
class FunctionType;
class Use;

namespace dxil {

class DXILOpBuilder {
public:
  DXILOpBuilder(Module &M, IRBuilderBase &B) : M(M), B(B) {}
  CallInst *createDXILOpCall(dxil::OpCode OpCode, Type *ReturnTy,
                             Type *OverloadTy,
                             llvm::iterator_range<Use *> Args);
  Type *getOverloadTy(dxil::OpCode OpCode, FunctionType *FT);
  static const char *getOpCodeName(dxil::OpCode DXILOp);

private:
  Module &M;
  IRBuilderBase &B;
};

} // namespace dxil
} // namespace llvm

#endif
