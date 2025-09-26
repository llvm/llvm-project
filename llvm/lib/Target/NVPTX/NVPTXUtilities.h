//===-- NVPTXUtilities - Utilities -----------------------------*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVVM specific utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXUTILITIES_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXUTILITIES_H

#include "NVPTX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdarg>
#include <set>
#include <string>
#include <vector>

namespace llvm {

class TargetMachine;

void clearAnnotationCache(const Module *);

bool isTexture(const Value &);
bool isSurface(const Value &);
bool isSampler(const Value &);
bool isImage(const Value &);
bool isImageReadOnly(const Value &);
bool isImageWriteOnly(const Value &);
bool isImageReadWrite(const Value &);
bool isManaged(const Value &);

StringRef getTextureName(const Value &);
StringRef getSurfaceName(const Value &);
StringRef getSamplerName(const Value &);

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

inline bool isKernelFunction(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

bool isParamGridConstant(const Argument &);

inline MaybeAlign getAlign(const Function &F, unsigned Index) {
  return F.getAttributes().getAttributes(Index).getStackAlignment();
}

MaybeAlign getAlign(const CallInst &, unsigned);
Function *getMaybeBitcastedCallee(const CallBase *CB);

// PTX ABI requires all scalar argument/return values to have
// bit-size as a power of two of at least 32 bits.
inline unsigned promoteScalarArgumentSize(unsigned size) {
  if (size <= 32)
    return 32;
  if (size <= 64)
    return 64;
  if (size <= 128)
    return 128;
  return size;
}

bool shouldEmitPTXNoReturn(const Value *V, const TargetMachine &TM);

inline bool shouldPassAsArray(Type *Ty) {
  return Ty->isAggregateType() || Ty->isVectorTy() ||
         Ty->getScalarSizeInBits() >= 128 || Ty->isHalfTy() || Ty->isBFloatTy();
}

namespace NVPTX {
// Returns a list of vector types that we prefer to fit into a single PTX
// register. NOTE: This must be kept in sync with the register classes
// defined in NVPTXRegisterInfo.td.
inline auto packed_types() {
  static const auto PackedTypes = {MVT::v4i8, MVT::v2f16, MVT::v2bf16,
                                   MVT::v2i16, MVT::v2f32};
  return PackedTypes;
}

// Checks if the type VT can fit into a single register.
inline bool isPackedVectorTy(EVT VT) {
  return any_of(packed_types(), [VT](EVT OVT) { return OVT == VT; });
}

// Checks if two or more of the type ET can fit into a single register.
inline bool isPackedElementTy(EVT ET) {
  return any_of(packed_types(),
                [ET](EVT OVT) { return OVT.getVectorElementType() == ET; });
}

inline std::string getValidPTXIdentifier(StringRef Name) {
  std::string ValidName;
  ValidName.reserve(Name.size() + 4);
  for (char C : Name)
    // While PTX also allows '%' at the start of identifiers, LLVM will throw a
    // fatal error for '%' in symbol names in MCSymbol::print. Exclude for now.
    if (isAlnum(C) || C == '_' || C == '$')
      ValidName.push_back(C);
    else
      ValidName.append({'_', '$', '_'});

  return ValidName;
}

inline std::string OrderingToString(Ordering Order) {
  switch (Order) {
  case Ordering::NotAtomic:
    return "NotAtomic";
  case Ordering::Relaxed:
    return "Relaxed";
  case Ordering::Acquire:
    return "Acquire";
  case Ordering::Release:
    return "Release";
  case Ordering::AcquireRelease:
    return "AcquireRelease";
  case Ordering::SequentiallyConsistent:
    return "SequentiallyConsistent";
  case Ordering::Volatile:
    return "Volatile";
  case Ordering::RelaxedMMIO:
    return "RelaxedMMIO";
  }
  report_fatal_error(formatv("Unknown NVPTX::Ordering \"{}\".",
                             static_cast<OrderingUnderlyingType>(Order)));
}

inline raw_ostream &operator<<(raw_ostream &O, Ordering Order) {
  O << OrderingToString(Order);
  return O;
}

inline std::string ScopeToString(Scope S) {
  switch (S) {
  case Scope::Thread:
    return "Thread";
  case Scope::System:
    return "System";
  case Scope::Block:
    return "Block";
  case Scope::Cluster:
    return "Cluster";
  case Scope::Device:
    return "Device";
  case Scope::DefaultDevice:
    return "DefaultDevice";
  }
  report_fatal_error(formatv("Unknown NVPTX::Scope \"{}\".",
                             static_cast<ScopeUnderlyingType>(S)));
}

inline raw_ostream &operator<<(raw_ostream &O, Scope S) {
  O << ScopeToString(S);
  return O;
}

inline std::string AddressSpaceToString(AddressSpace A) {
  switch (A) {
  case AddressSpace::Generic:
    return "generic";
  case AddressSpace::Global:
    return "global";
  case AddressSpace::Const:
    return "const";
  case AddressSpace::Shared:
    return "shared";
  case AddressSpace::SharedCluster:
    return "shared::cluster";
  case AddressSpace::Param:
    return "param";
  case AddressSpace::Local:
    return "local";
  }
  report_fatal_error(formatv("Unknown NVPTX::AddressSpace \"{}\".",
                             static_cast<AddressSpaceUnderlyingType>(A)));
}

inline raw_ostream &operator<<(raw_ostream &O, AddressSpace A) {
  O << AddressSpaceToString(A);
  return O;
}

} // namespace NVPTX
} // namespace llvm

#endif
