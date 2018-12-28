//===- SPIRVMDWalker.h -  SPIR-V metadata walker header file ----*- C++ -*-===//
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
/// This file declares classes for walking SPIR-V metadata.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_SPIRVMDWALKER_H
#define SPIRV_SPIRVMDWALKER_H

#include "SPIRVInternal.h"
#include "llvm/IR/Metadata.h"

#include <functional>
using namespace llvm;

namespace SPIRV {

class SPIRVMDWalker {
public:
  template <typename ParentT> struct MDWrapper;

  struct NamedMDWrapper {
    NamedMDWrapper(NamedMDNode *Named, SPIRVMDWalker &WW)
        : NMD(Named), W(WW), I(0), Q(true) {
      E = Named ? Named->getNumOperands() : 0;
    }

    operator bool() const { return NMD; }

    bool atEnd() const { return !(NMD && I < E); }

    MDWrapper<NamedMDWrapper> nextOp() {
      if (!Q)
        assert(I < E && "out of bound");
      return MDWrapper<NamedMDWrapper>(
          (NMD && I < E) ? NMD->getOperand(I++) : nullptr, *this, W);
    }

    NamedMDWrapper &setQuiet(bool Quiet) {
      Q = Quiet;
      return *this;
    }

    NamedMDNode *NMD;
    SPIRVMDWalker &W;
    unsigned I;
    unsigned E;
    bool Q; // Quiet
  };

  template <typename ParentT> struct MDWrapper {
    MDWrapper(MDNode *Node, ParentT &Parent, SPIRVMDWalker &Walker)
        : M(Node), P(Parent), W(Walker), I(0), Q(false) {
      E = Node ? Node->getNumOperands() : 0;
    }

    operator bool() const { return M; }

    bool atEnd() const { return !(M && I < E); }

    template <typename T> MDWrapper &get(T &V) {
      if (!Q)
        assert(I < E && "out of bound");
      if (atEnd())
        return *this;
      V = mdconst::dyn_extract<ConstantInt>(M->getOperand(I++))->getZExtValue();
      return *this;
    }

    MDWrapper &get(std::string &S) {
      if (!Q)
        assert(I < E && "out of bound");
      if (atEnd())
        return *this;
      Metadata *Op = M->getOperand(I++);
      if (!Op)
        S = "";
      else if (auto Str = dyn_cast<MDString>(Op))
        S = Str->getString().str();
      else
        S = "";
      return *this;
    }

    MDWrapper &get(Function *&F) {
      if (!Q)
        assert(I < E && "out of bound");
      if (atEnd())
        return *this;
      F = mdconst::dyn_extract<Function>(M->getOperand(I++));
      return *this;
    }

    MDWrapper &get(SmallVectorImpl<std::string> &SV) {
      if (atEnd())
        return *this;
      while (I < E) {
        std::string S;
        get(S);
        SV.push_back(S);
      }
      return *this;
    }

    MDWrapper<MDWrapper> nextOp() {
      if (!Q)
        assert(I < E && "out of bound");
      return MDWrapper<MDWrapper>(
          (M && I < E) ? dyn_cast<MDNode>(M->getOperand(I++)) : nullptr, *this,
          W);
    }

    ParentT &done() { return P; }

    MDWrapper &setQuiet(bool Quiet) {
      Q = Quiet;
      return *this;
    }

    MDNode *M;
    ParentT &P;
    SPIRVMDWalker &W;
    SmallVector<Metadata *, 10> V;
    unsigned I;
    unsigned E;
    bool Q; // Quiet
  };

  explicit SPIRVMDWalker(Module &Mod) : M(Mod), C(Mod.getContext()) {}

  NamedMDWrapper getNamedMD(StringRef Name) {
    return NamedMDWrapper(M.getNamedMetadata(Name), *this);
  }

  friend struct NamedMDWrapper;

private:
  Module &M;
  LLVMContext &C;
};

} /* namespace SPIRV */

#endif // SPIRV_SPIRVMDWALKER_H
