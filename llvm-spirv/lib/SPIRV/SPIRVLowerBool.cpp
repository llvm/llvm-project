//===- SPIRVLowerBool.cpp - Lower instructions with bool operands ---------===//
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
// This file implements lowering instructions with bool operands.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvbool"

#include "SPIRVInternal.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace SPIRV;

namespace SPIRV {
cl::opt<bool> SPIRVLowerBoolValidate(
    "spvbool-validate",
    cl::desc("Validate module after lowering boolean instructions for SPIR-V"));

class SPIRVLowerBool : public ModulePass, public InstVisitor<SPIRVLowerBool> {
public:
  SPIRVLowerBool() : ModulePass(ID), Context(nullptr) {
    initializeSPIRVLowerBoolPass(*PassRegistry::getPassRegistry());
  }
  void replace(Instruction *I, Instruction *NewI) {
    NewI->takeName(I);
    I->replaceAllUsesWith(NewI);
    I->dropAllReferences();
    I->eraseFromParent();
  }
  bool isBoolType(Type *Ty) {
    if (Ty->isIntegerTy(1))
      return true;
    if (auto VT = dyn_cast<VectorType>(Ty))
      return isBoolType(VT->getElementType());
    return false;
  }
  virtual void visitTruncInst(TruncInst &I) {
    if (isBoolType(I.getType())) {
      auto Op = I.getOperand(0);
      auto Zero = getScalarOrVectorConstantInt(Op->getType(), 0, false);
      auto Cmp = new ICmpInst(&I, CmpInst::ICMP_NE, Op, Zero);
      replace(&I, Cmp);
    }
  }
  virtual void visitZExtInst(ZExtInst &I) {
    auto Op = I.getOperand(0);
    if (isBoolType(Op->getType())) {
      auto Ty = I.getType();
      auto Zero = getScalarOrVectorConstantInt(Ty, 0, false);
      auto One = getScalarOrVectorConstantInt(Ty, 1, false);
      assert(Zero && One && "Couldn't create constant int");
      auto Sel = SelectInst::Create(Op, One, Zero, "", &I);
      replace(&I, Sel);
    }
  }
  virtual void visitSExtInst(SExtInst &I) {
    auto Op = I.getOperand(0);
    if (isBoolType(Op->getType())) {
      auto Ty = I.getType();
      auto Zero = getScalarOrVectorConstantInt(Ty, 0, false);
      auto One = getScalarOrVectorConstantInt(Ty, ~0, false);
      assert(Zero && One && "Couldn't create constant int");
      auto Sel = SelectInst::Create(Op, One, Zero, "", &I);
      replace(&I, Sel);
    }
  }
  bool runOnModule(Module &M) override {
    Context = &M.getContext();
    visit(M);

    if (SPIRVLowerBoolValidate) {
      LLVM_DEBUG(dbgs() << "After SPIRVLowerBool:\n" << M);
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
};

char SPIRVLowerBool::ID = 0;
} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerBool, "spvbool",
                "Lower instructions with bool operands", false, false)

ModulePass *llvm::createSPIRVLowerBool() { return new SPIRVLowerBool(); }
