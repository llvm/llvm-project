//===-- NVPTXUtilities - Utilities -----------------------------*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations for PTX-specific utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXUTILITIES_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXUTILITIES_H

#include "NVPTX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdarg>
#include <string>

namespace llvm {

class DataLayout;
class TargetMachine;

Function *getMaybeBitcastedCallee(const CallBase *CB);

/// Since function arguments are passed via .param space, we may want to
/// increase their alignment in a way that ensures that we can effectively
/// vectorize their loads & stores. We can increase alignment only if the
/// function has internal or private linkage as for other linkage types callers
/// may already rely on default alignment. To allow using 128-bit vectorized
/// loads/stores, this function ensures that alignment is 16 or greater.
Align getFunctionParamOptimizedAlign(const Function *F, Type *ArgTy,
                                     const DataLayout &DL);

Align getFunctionArgumentAlignment(const Function *F, Type *Ty, unsigned Idx,
                                   const DataLayout &DL);

Align getFunctionByValParamAlign(const Function *F, Type *ArgTy,
                                 Align InitialAlign, const DataLayout &DL);

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
  static const auto PackedTypes = {MVT::v4i8,  MVT::v2f16, MVT::v2bf16,
                                   MVT::v2i16, MVT::v2f32, MVT::v2i32};
  return PackedTypes;
}

// Checks if the type VT can fit into a single register.
inline bool isPackedVectorTy(EVT VT) {
  return any_of(packed_types(), equal_to(VT));
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

inline const char *addressSpaceToString(AddressSpace A,
                                        bool UseParamSubqualifiers = false) {
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
  case AddressSpace::EntryParam:
    return UseParamSubqualifiers ? "param::entry" : "param";
  case AddressSpace::DeviceParam:
    return UseParamSubqualifiers ? "param::func" : "param";
  case AddressSpace::Local:
    return "local";
  }
  report_fatal_error(formatv("Unknown NVPTX::AddressSpace \"{}\".",
                             static_cast<AddressSpaceUnderlyingType>(A)));
}

inline raw_ostream &operator<<(raw_ostream &O, AddressSpace A) {
  O << addressSpaceToString(A);
  return O;
}

} // namespace NVPTX
} // namespace llvm

#endif
