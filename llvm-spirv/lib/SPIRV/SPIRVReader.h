//===- SPIRVReader.h - Converts SPIR-V to LLVM ------------------*- C++ -*-===//
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
/// This file contains declaration of SPIRVToLLVM class which implements
/// conversion of SPIR-V binary to LLVM IR.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRVREADER_H
#define SPIRVREADER_H

#include "SPIRVModule.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/GlobalValue.h" // llvm::GlobalValue::LinkageTypes

namespace llvm {
class Module;
class Type;
class Value;
class Instruction;
class CallInst;
class BasicBlock;
class Function;
class GlobalVariable;
class LLVMContext;
class MDString;
class IntrinsicInst;
class LoadInst;
class BranchInst;
class BinaryOperator;
class Value;
} // namespace llvm
using namespace llvm;

namespace SPIRV {
class SPIRVFunctionParameter;
class SPIRVConstantSampler;
class SPIRVConstantPipeStorage;
class SPIRVLoopMerge;
class SPIRVToLLVMDbgTran;
class SPIRVToLLVM {
public:
  SPIRVToLLVM(Module *LLVMModule, SPIRVModule *TheSPIRVModule);

  std::string getOCLBuiltinName(SPIRVInstruction *BI);
  std::string getOCLConvertBuiltinName(SPIRVInstruction *BI);
  std::string getOCLGenericCastToPtrName(SPIRVInstruction *BI);

  Type *transType(SPIRVType *BT, bool IsClassMember = false);
  std::string transTypeToOCLTypeName(SPIRVType *BT, bool IsSigned = true);
  std::vector<Type *> transTypeVector(const std::vector<SPIRVType *> &);
  bool translate();
  bool transAddressingModel();

  Value *transValue(SPIRVValue *, Function *F, BasicBlock *,
                    bool CreatePlaceHolder = true);
  Value *transValueWithoutDecoration(SPIRVValue *, Function *F, BasicBlock *,
                                     bool CreatePlaceHolder = true);
  Value *transDeviceEvent(SPIRVValue *BV, Function *F, BasicBlock *BB);
  bool transDecoration(SPIRVValue *, Value *);
  bool transAlign(SPIRVValue *, Value *);
  Instruction *transOCLBuiltinFromExtInst(SPIRVExtInst *BC, BasicBlock *BB);
  std::vector<Value *> transValue(const std::vector<SPIRVValue *> &,
                                  Function *F, BasicBlock *);
  Function *transFunction(SPIRVFunction *F);
  Value *transBlockInvoke(SPIRVValue *Invoke, BasicBlock *BB);
  Instruction *transEnqueueKernelBI(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transWGSizeQueryBI(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transSGSizeQueryBI(SPIRVInstruction *BI, BasicBlock *BB);
  bool transFPContractMetadata();
  bool transKernelMetadata();
  bool transNonTemporalMetadata(Instruction *I);
  bool transSourceLanguage();
  bool transSourceExtension();
  void transGeneratorMD();
  Value *transConvertInst(SPIRVValue *BV, Function *F, BasicBlock *BB);
  Instruction *transBuiltinFromInst(const std::string &FuncName,
                                    SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transOCLBuiltinFromInst(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transSPIRVBuiltinFromInst(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transOCLBarrierFence(SPIRVInstruction *BI, BasicBlock *BB);
  void transOCLVectorLoadStore(std::string &UnmangledName,
                               std::vector<SPIRVWord> &BArgs);

  /// Post-process translated LLVM module for OpenCL.
  bool postProcessOCL();

  /// \brief Post-process OpenCL builtin functions returning struct type.
  ///
  /// Some OpenCL builtin functions are translated to SPIR-V instructions with
  /// struct type result, e.g. NDRange creation functions. Such functions
  /// need to be post-processed to return the struct through sret argument.
  bool postProcessOCLBuiltinReturnStruct(Function *F);

  /// \brief Post-process OpenCL builtin functions having array argument.
  ///
  /// These functions are translated to functions with array type argument
  /// first, then post-processed to have pointer arguments.
  bool
  postProcessOCLBuiltinWithArrayArguments(Function *F,
                                          const std::string &DemangledName);

  /// \brief Post-process OpImageSampleExplicitLod.
  ///   sampled_image = __spirv_SampledImage__(image, sampler);
  ///   return __spirv_ImageSampleExplicitLod__(sampled_image, image_operands,
  ///                                           ...);
  /// =>
  ///   read_image(image, sampler, ...)
  /// \return transformed call instruction.
  Instruction *postProcessOCLReadImage(SPIRVInstruction *BI, CallInst *CI,
                                       const std::string &DemangledName);

  /// \brief Post-process OpImageWrite.
  ///   return write_image(image, coord, color, image_operands, ...);
  /// =>
  ///   write_image(image, coord, ..., color)
  /// \return transformed call instruction.
  CallInst *postProcessOCLWriteImage(SPIRVInstruction *BI, CallInst *CI,
                                     const std::string &DemangledName);

  /// \brief Post-process OpBuildNDRange.
  ///   OpBuildNDRange GlobalWorkSize, LocalWorkSize, GlobalWorkOffset
  /// =>
  ///   call ndrange_XD(GlobalWorkOffset, GlobalWorkSize, LocalWorkSize)
  /// \return transformed call instruction.
  CallInst *postProcessOCLBuildNDRange(SPIRVInstruction *BI, CallInst *CI,
                                       const std::string &DemangledName);

  /// \brief Expand OCL builtin functions with scalar argument, e.g.
  /// step, smoothstep.
  /// gentype func (fp edge, gentype x)
  /// =>
  /// gentype func (gentype edge, gentype x)
  /// \return transformed call instruction.
  CallInst *expandOCLBuiltinWithScalarArg(CallInst *CI,
                                          const std::string &FuncName);

  /// \brief Post-process OpGroupAll and OpGroupAny instructions translation.
  /// i1 func (<n x i1> arg)
  /// =>
  /// i32 func (<n x i32> arg)
  /// \return transformed call instruction.
  Instruction *postProcessGroupAllAny(CallInst *CI,
                                      const std::string &DemangledName);

  typedef DenseMap<SPIRVType *, Type *> SPIRVToLLVMTypeMap;
  typedef DenseMap<SPIRVValue *, Value *> SPIRVToLLVMValueMap;
  typedef DenseMap<SPIRVValue *, Value *> SPIRVBlockToLLVMStructMap;
  typedef DenseMap<SPIRVFunction *, Function *> SPIRVToLLVMFunctionMap;
  typedef DenseMap<GlobalVariable *, SPIRVBuiltinVariableKind> BuiltinVarMap;

  // A SPIRV value may be translated to a load instruction of a placeholder
  // global variable. This map records load instruction of these placeholders
  // which are supposed to be replaced by the real values later.
  typedef std::map<SPIRVValue *, LoadInst *> SPIRVToLLVMPlaceholderMap;

private:
  Module *M;
  BuiltinVarMap BuiltinGVMap;
  LLVMContext *Context;
  SPIRVModule *BM;
  SPIRVToLLVMTypeMap TypeMap;
  SPIRVToLLVMValueMap ValueMap;
  SPIRVToLLVMFunctionMap FuncMap;
  SPIRVBlockToLLVMStructMap BlockMap;
  SPIRVToLLVMPlaceholderMap PlaceholderMap;
  std::unique_ptr<SPIRVToLLVMDbgTran> DbgTran;

  Type *mapType(SPIRVType *BT, Type *T);

  // If a value is mapped twice, the existing mapped value is a placeholder,
  // which must be a load instruction of a global variable whose name starts
  // with kPlaceholderPrefix.
  Value *mapValue(SPIRVValue *BV, Value *V);

  bool isSPIRVBuiltinVariable(GlobalVariable *GV,
                              SPIRVBuiltinVariableKind *Kind = nullptr);

  // OpenCL function always has NoUnwind attribute.
  // Change this if it is no longer true.
  bool isFuncNoUnwind() const { return true; }
  bool isSPIRVCmpInstTransToLLVMInst(SPIRVInstruction *BI) const;
  bool transOCLBuiltinsFromVariables();
  bool transOCLBuiltinFromVariable(GlobalVariable *GV,
                                   SPIRVBuiltinVariableKind Kind);
  MDString *transOCLKernelArgTypeName(SPIRVFunctionParameter *);
  Value *mapFunction(SPIRVFunction *BF, Function *F);
  Value *getTranslatedValue(SPIRVValue *BV);
  Type *getTranslatedType(SPIRVType *BT);
  IntrinsicInst *getLifetimeStartIntrinsic(Instruction *I);
  SPIRVErrorLog &getErrorLog();
  void setCallingConv(CallInst *Call);
  void setAttrByCalledFunc(CallInst *Call);
  Type *transFPType(SPIRVType *T);
  BinaryOperator *transShiftLogicalBitwiseInst(SPIRVValue *BV, BasicBlock *BB,
                                               Function *F);
  Instruction *transCmpInst(SPIRVValue *BV, BasicBlock *BB, Function *F);
  void transOCLBuiltinFromInstPreproc(SPIRVInstruction *BI, Type *&RetTy,
                                      std::vector<SPIRVValue *> &Args);
  Instruction *transOCLBuiltinPostproc(SPIRVInstruction *BI, CallInst *CI,
                                       BasicBlock *BB,
                                       const std::string &DemangledName);
  std::string transOCLImageTypeName(SPIRV::SPIRVTypeImage *ST);
  std::string transOCLSampledImageTypeName(SPIRV::SPIRVTypeSampledImage *ST);
  std::string transOCLPipeTypeName(
      SPIRV::SPIRVTypePipe *ST, bool UseSPIRVFriendlyFormat = false,
      SPIRVAccessQualifierKind PipeAccess = AccessQualifierReadOnly);
  std::string transOCLPipeStorageTypeName(SPIRV::SPIRVTypePipeStorage *PST);
  std::string transOCLImageTypeAccessQualifier(SPIRV::SPIRVTypeImage *ST);
  std::string transOCLPipeTypeAccessQualifier(SPIRV::SPIRVTypePipe *ST);

  Value *oclTransConstantSampler(SPIRV::SPIRVConstantSampler *BCS,
                                 BasicBlock *BB);
  Value *oclTransConstantPipeStorage(SPIRV::SPIRVConstantPipeStorage *BCPS);
  void setName(llvm::Value *V, SPIRVValue *BV);
  void setLLVMLoopMetadata(SPIRVLoopMerge *LM, BranchInst *BI);
  void insertImageNameAccessQualifier(SPIRV::SPIRVTypeImage *ST,
                                      std::string &Name);
  template <class Source, class Func> bool foreachFuncCtlMask(Source, Func);
  llvm::GlobalValue::LinkageTypes transLinkageType(const SPIRVValue *V);
  Instruction *transOCLAllAny(SPIRVInstruction *BI, BasicBlock *BB);
  Instruction *transOCLRelational(SPIRVInstruction *BI, BasicBlock *BB);

  CallInst *transOCLBarrier(BasicBlock *BB, SPIRVWord ExecScope,
                            SPIRVWord MemSema, SPIRVWord MemScope);

  CallInst *transOCLMemFence(BasicBlock *BB, SPIRVWord MemSema,
                             SPIRVWord MemScope);
}; // class SPIRVToLLVM

} // namespace SPIRV

#endif // SPIRVREADER_H
