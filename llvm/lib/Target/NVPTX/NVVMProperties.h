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

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Alignment.h"
#include <cstdint>
#include <optional>

namespace llvm {

class Argument;
class CallBase;
class GlobalVariable;
class Module;
class Value;

void clearAnnotationCache(const Module *);

inline bool isKernelFunction(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

enum class PTXOpaqueType { None, Texture, Surface, Sampler };

PTXOpaqueType getPTXOpaqueType(const GlobalVariable &);
PTXOpaqueType getPTXOpaqueType(const Argument &);
PTXOpaqueType getPTXOpaqueType(const Value &);

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

/// Maps the name of each nvvm.abi_preserve* attribute that is present to its
/// register count, in PTX emission order. An absent attribute has no entry.
using ABIPreserve = SmallMapVector<StringRef, unsigned, 2>;

/// On a function, the attributes are looked up on the function definition or
/// declaration. On a callsite, the attributes are looked up on the call only;
/// they are not inherited from the callee.
ABIPreserve getABIPreserve(const Function &);
ABIPreserve getABIPreserve(const CallBase &);

inline MaybeAlign getStackAlign(const Function &F, unsigned Index) {
  return F.getAttributes().getAttributes(Index).getStackAlignment();
}
MaybeAlign getStackAlign(const CallBase &, unsigned);

} // namespace llvm

#endif
