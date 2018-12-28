//===- OCLTypeToSPIRV.h - Adapt types from OCL for SPIRV --------*- C++ -*-===//
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
//
// This file implements adaptation of OCL types for SPIRV. It does not modify
// the module. Instead, it returns adapted function type based on kernel
// argument metadata. Later LLVM/SPIRV translator will translate the adapted
// type instead of the original type.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_OCLTYPETOSPIRV_H
#define SPIRV_OCLTYPETOSPIRV_H

#include "LLVMSPIRVLib.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Pass.h"

#include <map>
#include <set>

using namespace llvm;

namespace SPIRV {

class OCLTypeToSPIRV : public ModulePass {
public:
  OCLTypeToSPIRV();
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;

  /// \return Adapted type based on kernel argument metadata. If \p V is
  ///   a function, returns function type.
  /// E.g. for a function with argument of read only opencl.image_2d_t* type
  /// returns a function with argument of type opencl.image2d_t.read_only*.
  Type *getAdaptedType(Value *V);

  static char ID;

private:
  Module *M;
  LLVMContext *Ctx;
  unsigned CLVer;
  std::map<Value *, Type *> AdaptedTy; // Adapted types for values
  std::set<Function *> WorkSet;        // Functions to be adapted

  void adaptFunctionArguments(Function *F);
  void adaptArgumentsByMetadata(Function *F);
  void adaptArgumentsBySamplerUse(Module &M);
  void adaptFunction(Function *F);
  void addAdaptedType(Value *V, Type *T);
  void addWork(Function *F);
};

} // namespace SPIRV

#endif // SPIRV_OCLTYPETOSPIRV_H
