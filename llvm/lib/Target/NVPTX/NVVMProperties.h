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
#include <string>

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

/// Extract the PTX abi_preserve directive string implied by
/// the NVVM preserve_n_* attributes, e.g. ".abi_preserve 8". Returns an empty
/// string when none are present. Entries are separated by '\n' when
/// \p Multiline is set, and by ' ' otherwise.
///
/// On a function, the attributes are looked up on the function definition or
/// declaration. On a callsite, the attributes are looked up on the call.
std::string getABIPreserveDirectives(const Function &, bool Multiline);
std::string getABIPreserveDirectives(const CallBase &, bool Multiline);

inline MaybeAlign getStackAlign(const Function &F, unsigned Index) {
  return F.getAttributes().getAttributes(Index).getStackAlignment();
}
MaybeAlign getStackAlign(const CallBase &, unsigned);

} // namespace llvm

#endif
