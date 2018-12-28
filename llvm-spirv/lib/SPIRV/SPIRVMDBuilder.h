//===- SPIRVMDBuilder.h -  SPIR-V metadata builder header file --*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares classes for creating SPIR-V metadata.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_SPIRVMDBUILDER_H
#define SPIRV_SPIRVMDBUILDER_H

#include "SPIRVInternal.h"
#include "llvm/IR/Metadata.h"

#include <functional>
using namespace llvm;

namespace SPIRV {

class SPIRVMDBuilder {
public:
  template <typename ParentT> struct MDWrapper;
  struct NamedMDWrapper {
    NamedMDWrapper(NamedMDNode &Named, SPIRVMDBuilder &BB)
        : NMD(Named), B(BB) {}
    MDWrapper<NamedMDWrapper> addOp() {
      return MDWrapper<NamedMDWrapper>(*this, B);
    }
    NamedMDWrapper &addOp(MDWrapper<NamedMDWrapper> &MD) {
      NMD.addOperand(MD.M);
      return *this;
    }
    NamedMDNode &NMD;
    SPIRVMDBuilder &B;
  };
  template <typename ParentT> struct MDWrapper {
    MDWrapper(ParentT &Parent, SPIRVMDBuilder &Builder)
        : M(nullptr), P(Parent), B(Builder) {}
    MDWrapper &add(unsigned I) {
      V.push_back(ConstantAsMetadata::get(getUInt32(&B.M, I)));
      return *this;
    }
    MDWrapper &addU16(unsigned short I) {
      V.push_back(ConstantAsMetadata::get(getUInt16(&B.M, I)));
      return *this;
    }
    MDWrapper &add(StringRef S) {
      V.push_back(MDString::get(B.C, S));
      return *this;
    }
    MDWrapper &add(Function *F) {
      V.push_back(ConstantAsMetadata::get(F));
      return *this;
    }
    MDWrapper &add(SmallVectorImpl<StringRef> &S) {
      for (auto &I : S)
        add(I);
      return *this;
    }
    MDWrapper &addOp(MDNode *Node) {
      V.push_back(Node);
      return *this;
    }
    MDWrapper<MDWrapper> addOp() { return MDWrapper<MDWrapper>(*this, B); }
    MDWrapper &addOp(MDWrapper<MDWrapper> &MD) {
      V.push_back(MD.M);
      return *this;
    }
    /// Generate the scheduled MDNode and return the parent.
    /// If \param Ptr is not nullptr, save the generated MDNode.
    ParentT &done(MDNode **Ptr = nullptr) {
      M = MDNode::get(B.C, V);
      if (Ptr)
        *Ptr = M;
      return P.addOp(*this);
    }
    MDNode *M;
    ParentT &P;
    SPIRVMDBuilder &B;
    SmallVector<Metadata *, 10> V;
  };
  explicit SPIRVMDBuilder(Module &Mod) : M(Mod), C(Mod.getContext()) {}
  NamedMDWrapper addNamedMD(StringRef Name) {
    return NamedMDWrapper(*M.getOrInsertNamedMetadata(Name), *this);
  }
  SPIRVMDBuilder &eraseNamedMD(StringRef Name) {
    if (auto N = M.getNamedMetadata(Name))
      M.eraseNamedMetadata(N);
    return *this;
  }
  friend struct NamedMDWrapper;

private:
  Module &M;
  LLVMContext &C;
};

} /* namespace SPIRV */

#endif // SPIRV_SPIRVMDBUILDER_H
