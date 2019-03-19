//===- OCL20To12.cpp - Transform OCL 2.0 builtins to OCL 1.2 builtins -----===//
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
// This file implements transform OCL 2.0 builtins to OCL 1.2 builtins.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "ocl20to12"

#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {
class OCL20To12 : public ModulePass, public InstVisitor<OCL20To12> {
public:
  OCL20To12() : ModulePass(ID), M(nullptr), Ctx(nullptr) {
    initializeOCL20To12Pass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  virtual void visitCallInst(CallInst &CI);

  /// Transform atomic_work_item_fence to mem_fence.
  ///   atomic_work_item_fence(flag, relaxed, work_group) =>
  ///       mem_fence(flag)
  void visitCallAtomicWorkItemFence(CallInst *CI);

  static char ID;

private:
  Module *M;
  LLVMContext *Ctx;
};

char OCL20To12::ID = 0;

bool OCL20To12::runOnModule(Module &Module) {
  M = &Module;
  if (getOCLVersion(M) >= kOCLVer::CL20)
    return false;

  Ctx = &M->getContext();
  visit(*M);

  LLVM_DEBUG(dbgs() << "After OCL20To12:\n" << *M);

  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

void OCL20To12::visitCallInst(CallInst &CI) {
  LLVM_DEBUG(dbgs() << "[visistCallInst] " << CI << '\n');
  auto F = CI.getCalledFunction();
  if (!F)
    return;

  auto MangledName = F->getName();
  std::string DemangledName;
  if (!oclIsBuiltin(MangledName, &DemangledName))
    return;
  LLVM_DEBUG(dbgs() << "DemangledName = " << DemangledName.c_str() << '\n');

  if (DemangledName == kOCLBuiltinName::AtomicWorkItemFence) {
    visitCallAtomicWorkItemFence(&CI);
    return;
  }
}

void OCL20To12::visitCallAtomicWorkItemFence(CallInst *CI) {
  auto Lit = getAtomicWorkItemFenceLiterals(CI);
  if (std::get<1>(Lit) != OCLLegacyAtomicMemOrder ||
      std::get<2>(Lit) != OCLLegacyAtomicMemScope)
    report_fatal_error("OCL 2.0 builtin atomic_work_item_fence used in 1.2",
                       false);

  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(M, CI,
                    [=](CallInst *, std::vector<Value *> &Args) {
                      Args.resize(1);
                      Args[0] = getInt32(M, std::get<0>(Lit));
                      return kOCLBuiltinName::MemFence;
                    },
                    &Attrs);
}

} // namespace SPIRV

INITIALIZE_PASS(OCL20To12, "ocl20to12",
                "Translate OCL 2.0 builtins to OCL 1.2 builtins", false, false)

ModulePass *llvm::createOCL20To12() { return new OCL20To12(); }
