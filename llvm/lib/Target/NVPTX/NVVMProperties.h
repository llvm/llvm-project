//===-- NVVMProperties - NVVM annotation utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations for NVVM attribute and metadata query
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVVMPROPERTIES_H
#define LLVM_LIB_TARGET_NVPTX_NVVMPROPERTIES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Alignment.h"
#include <cstdint>
#include <optional>

namespace llvm {

class Argument;
class CallInst;
class Module;
class Value;

void clearAnnotationCache(const Module *);

inline bool isKernelFunction(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

bool isTexture(const Value &);
bool isSurface(const Value &);
bool isSampler(const Value &);
bool isImage(const Value &);
bool isImageReadOnly(const Value &);
bool isImageWriteOnly(const Value &);
bool isImageReadWrite(const Value &);
bool isManaged(const Value &);

SmallVector<unsigned, 3> getMaxNTID(const Function &);
SmallVector<unsigned, 3> getReqNTID(const Function &);
SmallVector<unsigned, 3> getClusterDim(const Function &);

std::optional<uint64_t> getOverallMaxNTID(const Function &);
std::optional<uint64_t> getOverallReqNTID(const Function &);
std::optional<uint64_t> getOverallClusterRank(const Function &);

std::optional<unsigned> getMaxClusterRank(const Function &);
std::optional<unsigned> getMinCTASm(const Function &);
std::optional<unsigned> getMaxNReg(const Function &);

bool hasBlocksAreClusters(const Function &);

bool isParamGridConstant(const Argument &);

inline MaybeAlign getAlign(const Function &F, unsigned Index) {
  return F.getAttributes().getAttributes(Index).getStackAlignment();
}
MaybeAlign getAlign(const CallInst &, unsigned);

} // namespace llvm

#endif
