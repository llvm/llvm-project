//===- CBuffer.h - HLSL constant buffer handling ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains utilities to work with constant buffers in HLSL.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_CBUFFER_H
#define LLVM_FRONTEND_HLSL_CBUFFER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include <optional>

namespace llvm {
class Module;
class GlobalVariable;
class NamedMDNode;

namespace hlsl {

struct CBufferMember {
  GlobalVariable *GV;
  size_t Offset;

  CBufferMember(GlobalVariable *GV, size_t Offset) : GV(GV), Offset(Offset) {}
};

struct CBufferMapping {
  GlobalVariable *Handle;
  SmallVector<CBufferMember> Members;

  CBufferMapping(GlobalVariable *Handle) : Handle(Handle) {}
};

class CBufferMetadata {
  NamedMDNode *MD;
  SmallVector<CBufferMapping> Mappings;

  CBufferMetadata(NamedMDNode *MD) : MD(MD) {}

public:
  static std::optional<CBufferMetadata> get(Module &M);

  using iterator = SmallVector<CBufferMapping>::iterator;
  iterator begin() { return Mappings.begin(); }
  iterator end() { return Mappings.end(); }

  void eraseFromModule();
};

APInt translateCBufArrayOffset(const DataLayout &DL, APInt Offset,
                               ArrayType *Ty);

} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_CBUFFER_H
