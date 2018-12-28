//===- SPIRVModule.h - Class to represent a SPIR-V module -------*- C++ -*-===//
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
/// This file defines Module class for SPIR-V.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVMODULE_H
#define SPIRV_LIBSPIRV_SPIRVMODULE_H

#include "SPIRVEntry.h"

#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace SPIRV {

class SPIRVBasicBlock;
class SPIRVConstant;
class SPIRVEntry;
class SPIRVFunction;
class SPIRVInstruction;
class SPIRVType;
class SPIRVTypeArray;
class SPIRVTypeBool;
class SPIRVTypeFloat;
class SPIRVTypeFunction;
class SPIRVTypeInt;
class SPIRVTypeOpaque;
class SPIRVTypePointer;
class SPIRVTypeImage;
class SPIRVTypeSampler;
class SPIRVTypeSampledImage;
class SPIRVTypePipeStorage;
class SPIRVTypeStruct;
class SPIRVTypeVector;
class SPIRVTypeVoid;
class SPIRVTypeDeviceEvent;
class SPIRVTypeQueue;
class SPIRVTypePipe;
class SPIRVTypeVmeImageINTEL;
class SPIRVValue;
class SPIRVVariable;
class SPIRVDecorateGeneric;
class SPIRVDecorationGroup;
class SPIRVGroupDecorate;
class SPIRVGroupMemberDecorate;
class SPIRVGroupDecorateGeneric;
class SPIRVInstTemplateBase;

typedef SPIRVBasicBlock SPIRVLabel;
struct SPIRVTypeImageDescriptor;

class SPIRVModule {
public:
  typedef std::map<SPIRVCapabilityKind, SPIRVCapability *> SPIRVCapMap;

  static SPIRVModule *createSPIRVModule();
  SPIRVModule();
  virtual ~SPIRVModule();

  // Object query functions
  virtual bool exist(SPIRVId) const = 0;
  virtual bool exist(SPIRVId, SPIRVEntry **) const = 0;
  template <class T> T *get(SPIRVId Id) const {
    return static_cast<T *>(getEntry(Id));
  }
  virtual SPIRVEntry *getEntry(SPIRVId) const = 0;
  virtual bool hasDebugInfo() const = 0;

  // Error handling functions
  virtual SPIRVErrorLog &getErrorLog() = 0;
  virtual SPIRVErrorCode getError(std::string &) = 0;

  // Module query functions
  virtual SPIRVAddressingModelKind getAddressingModel() = 0;
  virtual const SPIRVCapMap &getCapability() const = 0;
  virtual bool hasCapability(SPIRVCapabilityKind) const = 0;
  virtual SPIRVExtInstSetKind getBuiltinSet(SPIRVId) const = 0;
  virtual SPIRVFunction *getEntryPoint(SPIRVExecutionModelKind,
                                       unsigned) const = 0;
  virtual std::set<std::string> &getExtension() = 0;
  virtual SPIRVFunction *getFunction(unsigned) const = 0;
  virtual SPIRVVariable *getVariable(unsigned) const = 0;
  virtual SPIRVMemoryModelKind getMemoryModel() const = 0;
  virtual unsigned getNumFunctions() const = 0;
  virtual unsigned getNumEntryPoints(SPIRVExecutionModelKind) const = 0;
  virtual unsigned getNumVariables() const = 0;
  virtual SourceLanguage getSourceLanguage(SPIRVWord *) const = 0;
  virtual std::set<std::string> &getSourceExtension() = 0;
  virtual SPIRVValue *getValue(SPIRVId TheId) const = 0;
  virtual std::vector<SPIRVValue *>
  getValues(const std::vector<SPIRVId> &) const = 0;
  virtual std::vector<SPIRVId>
  getIds(const std::vector<SPIRVEntry *> &) const = 0;
  virtual std::vector<SPIRVId>
  getIds(const std::vector<SPIRVValue *> &) const = 0;
  virtual SPIRVType *getValueType(SPIRVId TheId) const = 0;
  virtual std::vector<SPIRVType *>
  getValueTypes(const std::vector<SPIRVId> &) const = 0;
  virtual SPIRVConstant *getLiteralAsConstant(unsigned Literal) = 0;
  virtual bool isEntryPoint(SPIRVExecutionModelKind, SPIRVId) const = 0;
  virtual unsigned short getGeneratorId() const = 0;
  virtual unsigned short getGeneratorVer() const = 0;
  virtual SPIRVWord getSPIRVVersion() const = 0;
  virtual const std::vector<SPIRVExtInst *> &getDebugInstVec() const = 0;
  virtual const std::vector<SPIRVString *> &getStringVec() const = 0;

  // Module changing functions
  virtual bool importBuiltinSet(const std::string &, SPIRVId *) = 0;
  virtual bool importBuiltinSetWithId(const std::string &, SPIRVId) = 0;
  virtual void setAddressingModel(SPIRVAddressingModelKind) = 0;
  virtual void setAlignment(SPIRVValue *, SPIRVWord) = 0;
  virtual void setMemoryModel(SPIRVMemoryModelKind) = 0;
  virtual void setName(SPIRVEntry *, const std::string &) = 0;
  virtual void setSourceLanguage(SourceLanguage, SPIRVWord) = 0;
  virtual void optimizeDecorates() = 0;
  virtual void setAutoAddCapability(bool E) { AutoAddCapability = E; }
  virtual void setValidateCapability(bool E) { ValidateCapability = E; }
  virtual void setGeneratorId(unsigned short) = 0;
  virtual void setGeneratorVer(unsigned short) = 0;
  virtual void resolveUnknownStructFields() = 0;
  virtual void setSPIRVVersion(SPIRVWord) = 0;

  void setMinSPIRVVersion(SPIRVWord Ver) {
    setSPIRVVersion(std::max(Ver, getSPIRVVersion()));
  }

  // Object creation functions
  template <class T> T *add(T *Entry) {
    addEntry(Entry);
    return Entry;
  }
  virtual SPIRVEntry *addEntry(SPIRVEntry *) = 0;
  virtual SPIRVBasicBlock *addBasicBlock(SPIRVFunction *,
                                         SPIRVId Id = SPIRVID_INVALID) = 0;
  virtual SPIRVString *getString(const std::string &Str) = 0;
  virtual SPIRVMemberName *addMemberName(SPIRVTypeStruct *ST,
                                         SPIRVWord MemberNumber,
                                         const std::string &Name) = 0;
  virtual void addUnknownStructField(SPIRVTypeStruct *, unsigned Idx,
                                     SPIRVId Id) = 0;
  virtual void addLine(SPIRVEntry *E, SPIRVId FileNameId, SPIRVWord Line,
                       SPIRVWord Column) = 0;
  virtual const std::shared_ptr<const SPIRVLine> &getCurrentLine() const = 0;
  virtual void setCurrentLine(const std::shared_ptr<const SPIRVLine> &) = 0;
  virtual const SPIRVDecorateGeneric *addDecorate(SPIRVDecorateGeneric *) = 0;
  virtual SPIRVDecorationGroup *addDecorationGroup() = 0;
  virtual SPIRVDecorationGroup *
  addDecorationGroup(SPIRVDecorationGroup *Group) = 0;
  virtual SPIRVGroupDecorate *
  addGroupDecorate(SPIRVDecorationGroup *Group,
                   const std::vector<SPIRVEntry *> &Targets) = 0;
  virtual SPIRVGroupMemberDecorate *
  addGroupMemberDecorate(SPIRVDecorationGroup *Group,
                         const std::vector<SPIRVEntry *> &Targets) = 0;
  virtual SPIRVGroupDecorateGeneric *
  addGroupDecorateGeneric(SPIRVGroupDecorateGeneric *GDec) = 0;
  virtual void addEntryPoint(SPIRVExecutionModelKind, SPIRVId) = 0;
  virtual SPIRVForward *addForward(SPIRVType *Ty) = 0;
  virtual SPIRVForward *addForward(SPIRVId, SPIRVType *Ty) = 0;
  virtual SPIRVFunction *addFunction(SPIRVFunction *) = 0;
  virtual SPIRVFunction *addFunction(SPIRVTypeFunction *,
                                     SPIRVId Id = SPIRVID_INVALID) = 0;
  virtual SPIRVEntry *replaceForward(SPIRVForward *, SPIRVEntry *) = 0;
  virtual void eraseInstruction(SPIRVInstruction *, SPIRVBasicBlock *) = 0;

  // Type creation functions
  virtual SPIRVTypeArray *addArrayType(SPIRVType *, SPIRVConstant *) = 0;
  virtual SPIRVTypeBool *addBoolType() = 0;
  virtual SPIRVTypeFloat *addFloatType(unsigned) = 0;
  virtual SPIRVTypeFunction *
  addFunctionType(SPIRVType *, const std::vector<SPIRVType *> &) = 0;
  virtual SPIRVTypeImage *addImageType(SPIRVType *,
                                       const SPIRVTypeImageDescriptor &) = 0;
  virtual SPIRVTypeImage *addImageType(SPIRVType *,
                                       const SPIRVTypeImageDescriptor &,
                                       SPIRVAccessQualifierKind) = 0;
  virtual SPIRVTypeSampler *addSamplerType() = 0;
  virtual SPIRVTypePipeStorage *addPipeStorageType() = 0;
  virtual SPIRVTypeSampledImage *addSampledImageType(SPIRVTypeImage *T) = 0;
  virtual SPIRVTypeInt *addIntegerType(unsigned) = 0;
  virtual SPIRVTypeOpaque *addOpaqueType(const std::string &) = 0;
  virtual SPIRVTypePointer *addPointerType(SPIRVStorageClassKind,
                                           SPIRVType *) = 0;
  virtual SPIRVTypeStruct *openStructType(unsigned, const std::string &) = 0;
  virtual void closeStructType(SPIRVTypeStruct *, bool) = 0;
  virtual SPIRVTypeVector *addVectorType(SPIRVType *, SPIRVWord) = 0;
  virtual SPIRVTypeVoid *addVoidType() = 0;
  virtual SPIRVType *addOpaqueGenericType(Op) = 0;
  virtual SPIRVTypeDeviceEvent *addDeviceEventType() = 0;
  virtual SPIRVTypeQueue *addQueueType() = 0;
  virtual SPIRVTypePipe *addPipeType() = 0;
  virtual SPIRVType *addSubgroupAvcINTELType(Op) = 0;
  virtual SPIRVTypeVmeImageINTEL *addVmeImageINTELType(SPIRVTypeImage *) = 0;
  virtual void createForwardPointers() = 0;

  // Constants creation functions
  virtual SPIRVValue *
  addCompositeConstant(SPIRVType *, const std::vector<SPIRVValue *> &) = 0;
  virtual SPIRVValue *addConstant(SPIRVValue *) = 0;
  virtual SPIRVValue *addConstant(SPIRVType *, uint64_t) = 0;
  virtual SPIRVValue *addDoubleConstant(SPIRVTypeFloat *, double) = 0;
  virtual SPIRVValue *addFloatConstant(SPIRVTypeFloat *, float) = 0;
  virtual SPIRVValue *addIntegerConstant(SPIRVTypeInt *, uint64_t) = 0;
  virtual SPIRVValue *addNullConstant(SPIRVType *) = 0;
  virtual SPIRVValue *addUndef(SPIRVType *TheType) = 0;
  virtual SPIRVValue *addSamplerConstant(SPIRVType *TheType, SPIRVWord AddrMode,
                                         SPIRVWord ParametricMode,
                                         SPIRVWord FilterMode) = 0;
  virtual SPIRVValue *addPipeStorageConstant(SPIRVType *TheType,
                                             SPIRVWord PacketSize,
                                             SPIRVWord PacketAlign,
                                             SPIRVWord Capacity) = 0;

  // Instruction creation functions
  virtual SPIRVInstruction *addPtrAccessChainInst(SPIRVType *, SPIRVValue *,
                                                  std::vector<SPIRVValue *>,
                                                  SPIRVBasicBlock *, bool) = 0;
  virtual SPIRVInstruction *
  addAsyncGroupCopy(SPIRVValue *Scope, SPIRVValue *Dest, SPIRVValue *Src,
                    SPIRVValue *NumElems, SPIRVValue *Stride, SPIRVValue *Event,
                    SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addBinaryInst(Op, SPIRVType *, SPIRVValue *,
                                          SPIRVValue *, SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addBranchConditionalInst(SPIRVValue *, SPIRVLabel *,
                                                     SPIRVLabel *,
                                                     SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addBranchInst(SPIRVLabel *, SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addExtInst(SPIRVType *, SPIRVWord, SPIRVWord,
                                       const std::vector<SPIRVWord> &,
                                       SPIRVBasicBlock *,
                                       SPIRVInstruction * = nullptr) = 0;
  virtual SPIRVInstruction *addExtInst(SPIRVType *, SPIRVWord, SPIRVWord,
                                       const std::vector<SPIRVValue *> &,
                                       SPIRVBasicBlock *,
                                       SPIRVInstruction * = nullptr) = 0;
  virtual SPIRVEntry *addDebugInfo(SPIRVWord, SPIRVType *,
                                   const std::vector<SPIRVWord> &) = 0;
  virtual void addCapability(SPIRVCapabilityKind) = 0;
  template <typename T> void addCapabilities(const T &Caps) {
    for (auto I : Caps)
      addCapability(I);
  }
  /// Used by SPIRV entries to add required capability internally.
  /// Should not be used by users directly.
  virtual void addCapabilityInternal(SPIRVCapabilityKind) = 0;
  virtual SPIRVInstruction *addCallInst(SPIRVFunction *,
                                        const std::vector<SPIRVWord> &,
                                        SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *
  addCompositeConstructInst(SPIRVType *, const std::vector<SPIRVId> &,
                            SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *
  addCompositeExtractInst(SPIRVType *, SPIRVValue *,
                          const std::vector<SPIRVWord> &,
                          SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *
  addCompositeInsertInst(SPIRVValue *, SPIRVValue *,
                         const std::vector<SPIRVWord> &, SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addCopyObjectInst(SPIRVType *, SPIRVValue *,
                                              SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addCopyMemoryInst(SPIRVValue *, SPIRVValue *,
                                              const std::vector<SPIRVWord> &,
                                              SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *
  addCopyMemorySizedInst(SPIRVValue *, SPIRVValue *, SPIRVValue *,
                         const std::vector<SPIRVWord> &, SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addCmpInst(Op, SPIRVType *, SPIRVValue *,
                                       SPIRVValue *, SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addControlBarrierInst(SPIRVValue *ExecKind,
                                                  SPIRVValue *MemKind,
                                                  SPIRVValue *MemSema,
                                                  SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addGroupInst(Op OpCode, SPIRVType *Type,
                                         Scope Scope,
                                         const std::vector<SPIRVValue *> &Ops,
                                         SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstTemplateBase *addInstTemplate(Op OC, SPIRVBasicBlock *BB,
                                                 SPIRVType *Ty) = 0;
  virtual SPIRVInstTemplateBase *
  addInstTemplate(Op OC, const std::vector<SPIRVWord> &Ops, SPIRVBasicBlock *BB,
                  SPIRVType *Ty) = 0;
  virtual SPIRVInstruction *addLoadInst(SPIRVValue *,
                                        const std::vector<SPIRVWord> &,
                                        SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addLifetimeInst(Op OC, SPIRVValue *Object,
                                            SPIRVWord Size,
                                            SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addMemoryBarrierInst(Scope ScopeKind,
                                                 SPIRVWord MemFlag,
                                                 SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addPhiInst(SPIRVType *, std::vector<SPIRVValue *>,
                                       SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addUnreachableInst(SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addReturnInst(SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addReturnValueInst(SPIRVValue *,
                                               SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addSelectInst(SPIRVValue *, SPIRVValue *,
                                          SPIRVValue *, SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addSelectionMergeInst(SPIRVId MergeBlock,
                                                  SPIRVWord SelectionControl,
                                                  SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addLoopMergeInst(
      SPIRVId MergeBlock, SPIRVId ContinueTarget, SPIRVWord LoopControl,
      std::vector<SPIRVWord> LoopControlParameters, SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addStoreInst(SPIRVValue *, SPIRVValue *,
                                         const std::vector<SPIRVWord> &,
                                         SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addSwitchInst(
      SPIRVValue *, SPIRVBasicBlock *,
      const std::vector<std::pair<std::vector<SPIRVWord>, SPIRVBasicBlock *>> &,
      SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addFModInst(SPIRVType *TheType, SPIRVId TheDividend,
                                        SPIRVId TheDivisor,
                                        SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addVectorTimesScalarInst(SPIRVType *TheType,
                                                     SPIRVId TheVector,
                                                     SPIRVId TheScalar,
                                                     SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addUnaryInst(Op, SPIRVType *, SPIRVValue *,
                                         SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addVariable(SPIRVType *, bool, SPIRVLinkageTypeKind,
                                        SPIRVValue *, const std::string &,
                                        SPIRVStorageClassKind,
                                        SPIRVBasicBlock *) = 0;
  virtual SPIRVValue *
  addVectorShuffleInst(SPIRVType *Type, SPIRVValue *Vec1, SPIRVValue *Vec2,
                       const std::vector<SPIRVWord> &Components,
                       SPIRVBasicBlock *BB) = 0;
  virtual SPIRVInstruction *addVectorExtractDynamicInst(SPIRVValue *,
                                                        SPIRVValue *,
                                                        SPIRVBasicBlock *) = 0;
  virtual SPIRVInstruction *addVectorInsertDynamicInst(SPIRVValue *,
                                                       SPIRVValue *,
                                                       SPIRVValue *,
                                                       SPIRVBasicBlock *) = 0;
  virtual SPIRVId getExtInstSetId(SPIRVExtInstSetKind Kind) const = 0;

  // I/O functions
  friend spv_ostream &operator<<(spv_ostream &O, SPIRVModule &M);
  friend std::istream &operator>>(std::istream &I, SPIRVModule &M);

protected:
  bool AutoAddCapability;
  bool ValidateCapability;
};


#ifdef _SPIRV_SUPPORT_TEXT_FMT

/// Convert SPIR-V between binary and internel text formats.
/// This function is not thread safe and should not be used in multi-thread
/// applications unless guarded by a critical section.
bool ConvertSPIRV(std::istream &IS, spv_ostream &OS, std::string &ErrMsg,
                  bool FromText, bool ToText);

/// Convert SPIR-V between binary and internel text formats.
/// This function is not thread safe and should not be used in multi-thread
/// applications unless guarded by a critical section.
bool ConvertSPIRV(std::string &Input, std::string &Out, std::string &ErrMsg,
                  bool ToText);
#endif
} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVMODULE_H
