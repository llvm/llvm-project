//===- SPIRVLowerMemmove.cpp - Lower llvm.memmove to llvm.memcpys ---------===//
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
// This file implements lowering llvm.memmove into several llvm.memcpys.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvmemmove"

#include "SPIRVInternal.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace SPIRV;

namespace SPIRV {
cl::opt<bool> SPIRVLowerMemmoveValidate(
    "spvmemmove-validate",
    cl::desc("Validate module after lowering llvm.memmove instructions into "
             "llvm.memcpy"));

class SPIRVLowerMemmove : public ModulePass,
                          public InstVisitor<SPIRVLowerMemmove> {
public:
  SPIRVLowerMemmove() : ModulePass(ID), Context(nullptr) {
    initializeSPIRVLowerMemmovePass(*PassRegistry::getPassRegistry());
  }
  virtual void visitMemMoveInst(MemMoveInst &I) {
    IRBuilder<> Builder(I.getParent());
    Builder.SetInsertPoint(&I);
    auto *Dest = I.getRawDest();
    auto *Src = I.getRawSource();
    auto *SrcTy = Src->getType();
    if (!isa<ConstantInt>(I.getLength()))
      // ToDo: for non-constant length, could use a loop to copy a
      // fixed length chunk at a time. For now simply fail
      report_fatal_error("llvm.memmove of non-constant length not supported",
                         false);
    auto *Length = cast<ConstantInt>(I.getLength());
    if (isa<BitCastInst>(Src))
      // The source could be bit-cast from another type,
      // need the original type for the allocation of the temporary variable
      SrcTy = cast<BitCastInst>(Src)->getOperand(0)->getType();
    auto Align = I.getSourceAlignment();
    auto Volatile = I.isVolatile();
    Value *NumElements = nullptr;
    uint64_t ElementsCount = 1;
    if (SrcTy->isArrayTy()) {
      NumElements = Builder.getInt32(SrcTy->getArrayNumElements());
      ElementsCount = SrcTy->getArrayNumElements();
    }
    if (Mod->getDataLayout().getTypeSizeInBits(SrcTy->getPointerElementType()) *
            ElementsCount !=
        Length->getZExtValue() * 8)
      report_fatal_error("Size of the memcpy should match the allocated memory",
                         false);

    auto *Alloca =
        Builder.CreateAlloca(SrcTy->getPointerElementType(), NumElements);
    Alloca->setAlignment(Align);
    Builder.CreateLifetimeStart(Alloca);
    Builder.CreateMemCpy(Alloca, Align, Src, Align, Length, Volatile);
    auto *SecondCpy = Builder.CreateMemCpy(Dest, I.getDestAlignment(), Alloca,
                                           Align, Length, Volatile);
    Builder.CreateLifetimeEnd(Alloca);

    SecondCpy->takeName(&I);
    I.replaceAllUsesWith(SecondCpy);
    I.dropAllReferences();
    I.eraseFromParent();
  }
  bool runOnModule(Module &M) override {
    Context = &M.getContext();
    Mod = &M;
    visit(M);

    if (SPIRVLowerMemmoveValidate) {
      LLVM_DEBUG(dbgs() << "After SPIRVLowerMemmove:\n" << M);
      std::string Err;
      raw_string_ostream ErrorOS(Err);
      if (verifyModule(M, &ErrorOS)) {
        Err = std::string("Fails to verify module: ") + Err;
        report_fatal_error(Err.c_str(), false);
      }
    }
    return true;
  }

  static char ID;

private:
  LLVMContext *Context;
  Module *Mod;
};

char SPIRVLowerMemmove::ID = 0;
} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerMemmove, "spvmemmove",
                "Lower llvm.memmove into llvm.memcpy", false, false)

ModulePass *llvm::createSPIRVLowerMemmove() { return new SPIRVLowerMemmove(); }
