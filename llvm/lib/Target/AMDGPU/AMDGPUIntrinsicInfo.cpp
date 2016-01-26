//===- AMDGPUIntrinsicInfo.cpp - AMDGPU Intrinsic Information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU Implementation of the IntrinsicInfo class.
//
//===-----------------------------------------------------------------------===//

#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
#include "AMDGPUGenIntrinsics.inc"
#undef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN

AMDGPUIntrinsicInfo::AMDGPUIntrinsicInfo()
    : TargetIntrinsicInfo() {}

static const char *const IntrinsicNameTable[] = {
#define GET_INTRINSIC_NAME_TABLE
#include "AMDGPUGenIntrinsics.inc"
#undef GET_INTRINSIC_NAME_TABLE
};

std::string AMDGPUIntrinsicInfo::getName(unsigned IntrID, Type **Tys,
                                         unsigned numTys) const {
  if (IntrID < Intrinsic::num_intrinsics) {
    return nullptr;
  }
  assert(IntrID < AMDGPUIntrinsic::num_AMDGPU_intrinsics &&
         "Invalid intrinsic ID");

  std::string Result(IntrinsicNameTable[IntrID - Intrinsic::num_intrinsics]);
  return Result;
}

unsigned AMDGPUIntrinsicInfo::lookupName(const char *NameData,
                                         unsigned Len) const {
  StringRef Name(NameData, Len);
  if (!Name.startswith("llvm."))
    return 0; // All intrinsics start with 'llvm.'

  // Look for a name match in our table.  If the intrinsic is not overloaded,
  // require an exact match. If it is overloaded, require a prefix match. The
  // AMDGPU enum enum starts at Intrinsic::num_intrinsics.
  int Idx = Intrinsic::lookupLLVMIntrinsicByName(IntrinsicNameTable, Name);
  if (Idx >= 0) {
    bool IsPrefixMatch = Name.size() > strlen(IntrinsicNameTable[Idx]);
    return IsPrefixMatch == isOverloaded(Idx + 1)
               ? Intrinsic::num_intrinsics + Idx
               : 0;
  }

  // Fall back on GCC builtin names.
  return getIntrinsicForGCCBuiltin("AMDGPU", NameData);
}

bool AMDGPUIntrinsicInfo::isOverloaded(unsigned id) const {
// Overload Table
#define GET_INTRINSIC_OVERLOAD_TABLE
#include "AMDGPUGenIntrinsics.inc"
#undef GET_INTRINSIC_OVERLOAD_TABLE
}

Function *AMDGPUIntrinsicInfo::getDeclaration(Module *M, unsigned IntrID,
                                              Type **Tys,
                                              unsigned numTys) const {
  llvm_unreachable("Not implemented");
}
