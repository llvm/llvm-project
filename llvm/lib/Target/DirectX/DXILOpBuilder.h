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
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
class Module;
class IRBuilderBase;
class CallInst;
class Value;
class Type;
class FunctionType;

namespace dxil {

class DXILOpBuilder {
public:
  DXILOpBuilder(Module &M);

  IRBuilder<> &getIRB() { return IRB; }

  /// Create a call instruction for the given DXIL op. The arguments
  /// must be valid for an overload of the operation.
  CallInst *createOp(dxil::OpCode Op, ArrayRef<Value *> Args,
                     Type *RetTy = nullptr);

  /// Try to create a call instruction for the given DXIL op. Fails if the
  /// overload is invalid.
  Expected<CallInst *> tryCreateOp(dxil::OpCode Op, ArrayRef<Value *> Args,
                                   Type *RetTy = nullptr);

  /// Return the name of the given opcode.
  static const char *getOpCodeName(dxil::OpCode DXILOp);

private:
  /// Gets a specific overload type of the function for the given DXIL op. If
  /// the operation is not overloaded, \c OverloadType may be nullptr.
  FunctionType *getOpFunctionType(dxil::OpCode OpCode,
                                  Type *OverloadType = nullptr);

  Module &M;
  IRBuilder<> IRB;
  VersionTuple DXILVersion;
  Triple::EnvironmentType ShaderStage;
};

} // namespace dxil
} // namespace llvm

#endif
