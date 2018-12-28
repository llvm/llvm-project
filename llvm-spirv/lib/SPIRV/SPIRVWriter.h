//===- SPIRVWriter.h - Converts LLVM to SPIR-V ------------------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
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
/// This file contains declaration of LLVMToSPIRV class which implements
/// conversion of LLVM intermediate language to SPIR-V
/// binary.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRVWRITER_H
#define SPIRVWRITER_H

#include "OCLTypeToSPIRV.h"
#include "OCLUtil.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVEntry.h"
#include "SPIRVEnum.h"
#include "SPIRVExtInst.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "SPIRVMDWalker.h"
#include "SPIRVModule.h"
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"

#include "llvm/IR/IntrinsicInst.h"

#include <memory>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

class LLVMToSPIRVDbgTran;

class LLVMToSPIRV : public ModulePass {
public:
  LLVMToSPIRV(SPIRVModule *SMod = nullptr);

  virtual StringRef getPassName() const override { return "LLVMToSPIRV"; }

  bool runOnModule(Module &Mod) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<OCLTypeToSPIRV>();
  }

  static char ID;

  SPIRVType *transType(Type *T);
  SPIRVType *transSPIRVOpaqueType(Type *T);

  SPIRVValue *getTranslatedValue(const Value *) const;

  // Translation functions
  bool transAddressingMode();
  bool transAlign(Value *V, SPIRVValue *BV);
  std::vector<SPIRVValue *> transArguments(CallInst *, SPIRVBasicBlock *);
  std::vector<SPIRVWord> transArguments(CallInst *, SPIRVBasicBlock *,
                                        SPIRVEntry *);
  bool transSourceLanguage();
  bool transExtension();
  bool transBuiltinSet();
  SPIRVValue *transIntrinsicInst(IntrinsicInst *Intrinsic, SPIRVBasicBlock *BB);
  SPIRVValue *transCallInst(CallInst *Call, SPIRVBasicBlock *BB);
  bool transDecoration(Value *V, SPIRVValue *BV);
  SPIRVWord transFunctionControlMask(CallInst *);
  SPIRVWord transFunctionControlMask(Function *);
  SPIRVFunction *transFunctionDecl(Function *F);
  bool transGlobalVariables();

  Op transBoolOpCode(SPIRVValue *Opn, Op OC);
  // Translate LLVM module to SPIR-V module.
  // Returns true if succeeds.
  bool translate();
  bool transExecutionMode();
  SPIRVValue *transConstant(Value *V);
  SPIRVValue *transValue(Value *V, SPIRVBasicBlock *BB,
                         bool CreateForward = true);
  SPIRVValue *transValueWithoutDecoration(Value *V, SPIRVBasicBlock *BB,
                                          bool CreateForward = true);

  typedef DenseMap<Type *, SPIRVType *> LLVMToSPIRVTypeMap;
  typedef DenseMap<Value *, SPIRVValue *> LLVMToSPIRVValueMap;

private:
  Module *M;
  LLVMContext *Ctx;
  SPIRVModule *BM;
  LLVMToSPIRVTypeMap TypeMap;
  LLVMToSPIRVValueMap ValueMap;
  SPIRVWord SrcLang;
  SPIRVWord SrcLangVer;
  std::unique_ptr<LLVMToSPIRVDbgTran> DbgTran;

  SPIRVType *mapType(Type *T, SPIRVType *BT);
  SPIRVValue *mapValue(Value *V, SPIRVValue *BV);
  SPIRVType *getSPIRVType(Type *T) { return TypeMap[T]; }
  SPIRVErrorLog &getErrorLog() { return BM->getErrorLog(); }
  llvm::IntegerType *getSizetType();
  std::vector<SPIRVValue *> transValue(const std::vector<Value *> &Values,
                                       SPIRVBasicBlock *BB);
  std::vector<SPIRVWord> transValue(const std::vector<Value *> &Values,
                                    SPIRVBasicBlock *BB, SPIRVEntry *Entry);
  SPIRVInstruction *transBinaryInst(BinaryOperator *B, SPIRVBasicBlock *BB);
  SPIRVInstruction *transCmpInst(CmpInst *Cmp, SPIRVBasicBlock *BB);
  SPIRVInstruction *transLifetimeIntrinsicInst(Op OC, IntrinsicInst *Intrinsic,
                                               SPIRVBasicBlock *BB);

  void dumpUsers(Value *V);

  template <class ExtInstKind>
  bool oclGetExtInstIndex(const std::string &MangledName,
                          const std::string &DemangledName,
                          SPIRVWord *EntryPoint);
  void
  oclGetMutatedArgumentTypesByBuiltin(llvm::FunctionType *FT,
                                      std::map<unsigned, Type *> &ChangedType,
                                      Function *F);
  bool isBuiltinTransToInst(Function *F);
  bool isBuiltinTransToExtInst(Function *F,
                               SPIRVExtInstSetKind *BuiltinSet = nullptr,
                               SPIRVWord *EntryPoint = nullptr,
                               SmallVectorImpl<std::string> *Dec = nullptr);
  bool oclIsKernel(Function *F);
  bool transOCLKernelMetadata();
  SPIRVInstruction *transBuiltinToInst(const std::string &DemangledName,
                                       const std::string &MangledName,
                                       CallInst *CI, SPIRVBasicBlock *BB);
  SPIRVInstruction *transBuiltinToInstWithoutDecoration(Op OC, CallInst *CI,
                                                        SPIRVBasicBlock *BB);
  void mutateFuncArgType(const std::map<unsigned, Type *> &ChangedType,
                         Function *F);

  SPIRVValue *transSpcvCast(CallInst *CI, SPIRVBasicBlock *BB);
  SPIRVValue *oclTransSpvcCastSampler(CallInst *CI, SPIRVBasicBlock *BB);
  SPIRV::SPIRVInstruction *transUnaryInst(UnaryInstruction *U,
                                          SPIRVBasicBlock *BB);

  /// Add a 32 bit integer constant.
  /// \return Id of the constant.
  SPIRVId addInt32(int);
  void transFunction(Function *I);
  SPIRV::SPIRVLinkageTypeKind transLinkageType(const GlobalValue *GV);
};

} // namespace SPIRV

#endif // SPIRVWRITER_H
