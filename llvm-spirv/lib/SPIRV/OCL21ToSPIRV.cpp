//===- OCL21ToSPIRV.cpp - Transform OCL21 to SPIR-V builtins ----*- C++ -*-===//
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
// This file implements translation of OCL21 builtin functions.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "cl21tospv"

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

#include <set>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

class OCL21ToSPIRV : public ModulePass, public InstVisitor<OCL21ToSPIRV> {
public:
  OCL21ToSPIRV() : ModulePass(ID), M(nullptr), Ctx(nullptr), CLVer(0) {
    initializeOCL21ToSPIRVPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  virtual void visitCallInst(CallInst &CI);

  /// Transform SPIR-V convert function
  //    __spirv{N}Op{ConvertOpName}(src, dummy)
  ///   =>
  ///   __spirv_{ConvertOpName}_R{TargeTyName}
  void visitCallConvert(CallInst *CI, StringRef MangledName, Op OC);

  /// Transform SPIR-V decoration
  ///   x = __spirv_{OpName};
  ///   y = __spirv{N}Op{Decorate}(x, type, value, dummy)
  ///   =>
  ///   y = __spirv_{OpName}{Postfix(type,value)}
  void visitCallDecorate(CallInst *CI, StringRef MangledName);

  /// Transform sub_group_barrier to __spirv_ControlBarrier.
  /// sub_group_barrier(scope, flag) =>
  ///   __spirv_ControlBarrier(subgroup, map(scope), map(flag))
  void visitCallSubGroupBarrier(CallInst *CI);

  /// Transform OCL C++ builtin function to SPIR-V builtin function.
  /// Assuming there is no argument changes.
  /// Should be called at last.
  void transBuiltin(CallInst *CI, Op OC);

  static char ID;

private:
  ConstantInt *addInt32(int I) { return getInt32(M, I); }

  Module *M;
  LLVMContext *Ctx;
  unsigned CLVer; /// OpenCL version as major*10+minor
  std::set<Value *> ValuesToDelete;
};

char OCL21ToSPIRV::ID = 0;

bool OCL21ToSPIRV::runOnModule(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();

  auto Src = getSPIRVSource(&Module);
  if (std::get<0>(Src) != spv::SourceLanguageOpenCL_CPP)
    return false;

  CLVer = std::get<1>(Src);
  if (CLVer < kOCLVer::CL21)
    return false;

  LLVM_DEBUG(dbgs() << "Enter OCL21ToSPIRV:\n");
  visit(*M);

  for (auto &I : ValuesToDelete)
    if (auto Inst = dyn_cast<Instruction>(I))
      Inst->eraseFromParent();
  for (auto &I : ValuesToDelete)
    if (auto GV = dyn_cast<GlobalValue>(I))
      GV->eraseFromParent();

  LLVM_DEBUG(dbgs() << "After OCL21ToSPIRV:\n" << *M);
  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

// The order of handling OCL builtin functions is important.
// Workgroup functions need to be handled before pipe functions since
// there are functions fall into both categories.
void OCL21ToSPIRV::visitCallInst(CallInst &CI) {
  LLVM_DEBUG(dbgs() << "[visistCallInst] " << CI << '\n');
  auto F = CI.getCalledFunction();
  if (!F)
    return;

  auto MangledName = F->getName();
  std::string DemangledName;

  if (oclIsBuiltin(MangledName, &DemangledName)) {
    if (DemangledName == kOCLBuiltinName::SubGroupBarrier) {
      visitCallSubGroupBarrier(&CI);
      return;
    }
  }

  if (!oclIsBuiltin(MangledName, &DemangledName, true))
    return;
  LLVM_DEBUG(dbgs() << "DemangledName:" << DemangledName << '\n');
  StringRef Ref(DemangledName);

  Op OC = OpNop;
  if (!OpCodeNameMap::rfind(Ref.str(), &OC))
    return;
  LLVM_DEBUG(dbgs() << "maps to opcode " << OC << '\n');

  if (isCvtOpCode(OC)) {
    visitCallConvert(&CI, MangledName, OC);
    return;
  }
  if (OC == OpDecorate) {
    visitCallDecorate(&CI, MangledName);
    return;
  }
  transBuiltin(&CI, OC);
}

void OCL21ToSPIRV::visitCallConvert(CallInst *CI, StringRef MangledName,
                                    Op OC) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.pop_back();
        return getSPIRVFuncName(
            OC, kSPIRVPostfix::Divider +
                    getPostfixForReturnType(CI, OC == OpSConvert ||
                                                    OC == OpConvertFToS ||
                                                    OC == OpSatConvertUToS));
      },
      &Attrs);
  ValuesToDelete.insert(CI);
  ValuesToDelete.insert(CI->getCalledFunction());
}

void OCL21ToSPIRV::visitCallDecorate(CallInst *CI, StringRef MangledName) {
  auto Target = cast<CallInst>(CI->getArgOperand(0));
  assert(Target->getCalledFunction() && "Unexpected indirect call");
  auto F = Target->getCalledFunction();
  auto Name = F->getName().str();
  std::string DemangledName;
  oclIsBuiltin(Name, &DemangledName);
  BuiltinFuncMangleInfo Info;
  F->setName(mangleBuiltin(
      DemangledName + kSPIRVPostfix::Divider +
          getPostfix(getArgAsDecoration(CI, 1), getArgAsInt(CI, 2)),
      getTypes(getArguments(CI)), &Info));
  CI->replaceAllUsesWith(Target);
  ValuesToDelete.insert(CI);
  ValuesToDelete.insert(CI->getCalledFunction());
}

void OCL21ToSPIRV::visitCallSubGroupBarrier(CallInst *CI) {
  LLVM_DEBUG(dbgs() << "[visitCallSubGroupBarrier] " << *CI << '\n');
  auto Lit = getBarrierLiterals(CI);
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(M, CI,
                      [=](CallInst *, std::vector<Value *> &Args) {
                        Args.resize(3);
                        // Execution scope
                        Args[0] = addInt32(map<Scope>(std::get<2>(Lit)));
                        // Memory scope
                        Args[1] = addInt32(map<Scope>(std::get<1>(Lit)));
                        // Use sequential consistent memory order by default.
                        // But if the flags argument is set to 0, we use
                        // None(Relaxed) memory order.
                        unsigned MemFenceFlag = std::get<0>(Lit);
                        OCLMemOrderKind MemOrder =
                            MemFenceFlag ? OCLMO_seq_cst : OCLMO_relaxed;
                        Args[2] = addInt32(mapOCLMemSemanticToSPIRV(
                            MemFenceFlag, MemOrder)); // Memory semantics
                        return getSPIRVFuncName(OpControlBarrier);
                      },
                      &Attrs);
}

void OCL21ToSPIRV::transBuiltin(CallInst *CI, Op OC) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  assert(OC != OpExtInst && "not supported");
  mutateCallInstSPIRV(M, CI,
                      [=](CallInst *, std::vector<Value *> &Args) {
                        return getSPIRVFuncName(OC);
                      },
                      &Attrs);
  ValuesToDelete.insert(CI);
  ValuesToDelete.insert(CI->getCalledFunction());
}

} // namespace SPIRV

INITIALIZE_PASS(OCL21ToSPIRV, "cl21tospv", "Transform OCL 2.1 to SPIR-V", false,
                false)

ModulePass *llvm::createOCL21ToSPIRV() { return new OCL21ToSPIRV(); }
