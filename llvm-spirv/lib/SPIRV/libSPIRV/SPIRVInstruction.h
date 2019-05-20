//===- SPIRVInstruction.h - Class to represent SPIRV instruction -- C++ -*-===//
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
/// This file defines Instruction class for SPIR-V.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVINSTRUCTION_H
#define SPIRV_LIBSPIRV_SPIRVINSTRUCTION_H

#include "SPIRVBasicBlock.h"
#include "SPIRVEnum.h"
#include "SPIRVIsValidEnum.h"
#include "SPIRVOpCode.h"
#include "SPIRVStream.h"
#include "SPIRVValue.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace SPIRV {

typedef std::vector<SPIRVValue *> ValueVec;
typedef std::pair<ValueVec::iterator, ValueVec::iterator> ValueRange;

class SPIRVBasicBlock;
class SPIRVFunction;

bool isSpecConstantOpAllowedOp(Op OC);

class SPIRVComponentExecutionScope {
public:
  SPIRVComponentExecutionScope(Scope TheScope = ScopeInvocation)
      : ExecScope(TheScope) {}
  Scope ExecScope;
};

class SPIRVComponentMemorySemanticsMask {
public:
  SPIRVComponentMemorySemanticsMask(SPIRVWord TheSema = SPIRVWORD_MAX)
      : MemSema(TheSema) {}
  SPIRVWord MemSema;
};

class SPIRVComponentOperands {
public:
  SPIRVComponentOperands(){};
  SPIRVComponentOperands(const std::vector<SPIRVValue *> &TheOperands)
      : Operands(TheOperands){};
  SPIRVComponentOperands(std::vector<SPIRVValue *> &&TheOperands)
      : Operands(std::move(TheOperands)){};
  std::vector<SPIRVValue *> getCompOperands() { return Operands; }
  std::vector<SPIRVType *> getCompOperandTypes() {
    std::vector<SPIRVType *> Tys;
    for (auto &I : getCompOperands())
      Tys.push_back(I->getType());
    return Tys;
  }

protected:
  std::vector<SPIRVValue *> Operands;
};

class SPIRVInstruction : public SPIRVValue {
public:
  // Complete constructor for instruction with type and id
  SPIRVInstruction(unsigned TheWordCount, Op TheOC, SPIRVType *TheType,
                   SPIRVId TheId, SPIRVBasicBlock *TheBB);
  // Complete constructor for instruction with module, type and id
  SPIRVInstruction(unsigned TheWordCount, Op TheOC, SPIRVType *TheType,
                   SPIRVId TheId, SPIRVBasicBlock *TheBB, SPIRVModule *TheBM);
  // Complete constructor for instruction with id but no type
  SPIRVInstruction(unsigned TheWordCount, Op TheOC, SPIRVId TheId,
                   SPIRVBasicBlock *TheBB);
  // Complete constructor for instruction without type and id
  SPIRVInstruction(unsigned TheWordCount, Op TheOC, SPIRVBasicBlock *TheBB);
  // Complete constructor for instruction with type but no id
  SPIRVInstruction(unsigned TheWordCount, Op TheOC, SPIRVType *TheType,
                   SPIRVBasicBlock *TheBB);
  // Special constructor for debug instruction
  SPIRVInstruction(unsigned TheWordCount, Op TheOC, SPIRVType *TheType,
                   SPIRVId TheId, SPIRVModule *TheBM,
                   SPIRVBasicBlock *TheBB = nullptr);

  // Incomplete constructor
  SPIRVInstruction(Op TheOC = OpNop)
      : SPIRVValue(TheOC), BB(NULL), DebugScope(nullptr) {}

  bool isInst() const override { return true; }
  SPIRVBasicBlock *getParent() const { return BB; }
  SPIRVInstruction *getPrevious() const { return BB->getPrevious(this); }
  SPIRVInstruction *getNext() const { return BB->getNext(this); }
  virtual std::vector<SPIRVValue *> getOperands();
  std::vector<SPIRVType *> getOperandTypes();
  static std::vector<SPIRVType *>
  getOperandTypes(const std::vector<SPIRVValue *> &Ops);

  void setParent(SPIRVBasicBlock *);
  void setScope(SPIRVEntry *) override;
  void addFPRoundingMode(SPIRVFPRoundingModeKind Kind) {
    addDecorate(DecorationFPRoundingMode, Kind);
  }
  void eraseFPRoundingMode() { eraseDecorate(DecorationFPRoundingMode); }
  void setSaturatedConversion(bool Enable) {
    if (Enable)
      addDecorate(DecorationSaturatedConversion);
    else
      eraseDecorate(DecorationSaturatedConversion);
  }
  bool hasFPRoundingMode(SPIRVFPRoundingModeKind *Kind = nullptr) {
    SPIRVWord V;
    auto Found = hasDecorate(DecorationFPRoundingMode, 0, &V);
    if (Found && Kind)
      *Kind = static_cast<SPIRVFPRoundingModeKind>(V);
    return Found;
  }
  bool isSaturatedConversion() {
    return hasDecorate(DecorationSaturatedConversion) ||
           OpCode == OpSatConvertSToU || OpCode == OpSatConvertUToS;
  }

  SPIRVBasicBlock *getBasicBlock() const { return BB; }

  void setBasicBlock(SPIRVBasicBlock *TheBB) {
    BB = TheBB;
    if (TheBB)
      setModule(TheBB->getModule());
  }

  void setDebugScope(SPIRVEntry *Scope) { DebugScope = Scope; }

  SPIRVEntry *getDebugScope() const { return DebugScope; }

protected:
  void validate() const override { SPIRVValue::validate(); }

private:
  SPIRVBasicBlock *BB;
  SPIRVEntry *DebugScope;
};

class SPIRVInstTemplateBase : public SPIRVInstruction {
public:
  /// Create an empty instruction. Mainly for getting format information,
  /// e.g. whether an operand is literal.
  static SPIRVInstTemplateBase *create(Op TheOC) {
    auto Inst = static_cast<SPIRVInstTemplateBase *>(SPIRVEntry::create(TheOC));
    assert(Inst);
    Inst->init();
    return Inst;
  }
  /// Create a instruction without operands.
  static SPIRVInstTemplateBase *create(Op TheOC, SPIRVType *TheType,
                                       SPIRVId TheId, SPIRVBasicBlock *TheBB,
                                       SPIRVModule *TheModule) {
    auto Inst = create(TheOC);
    Inst->init(TheType, TheId, TheBB, TheModule);
    return Inst;
  }
  /// Create a complete and valid instruction.
  static SPIRVInstTemplateBase *create(Op TheOC, SPIRVType *TheType,
                                       SPIRVId TheId,
                                       const std::vector<SPIRVWord> &TheOps,
                                       SPIRVBasicBlock *TheBB,
                                       SPIRVModule *TheModule) {
    auto Inst = create(TheOC);
    Inst->init(TheType, TheId, TheBB, TheModule);
    Inst->setOpWords(TheOps);
    Inst->validate();
    return Inst;
  }
  SPIRVInstTemplateBase(Op OC = OpNop)
      : SPIRVInstruction(OC), HasVariWC(false) {
    init();
  }
  ~SPIRVInstTemplateBase() override {}
  SPIRVInstTemplateBase *init(SPIRVType *TheType, SPIRVId TheId,
                              SPIRVBasicBlock *TheBB, SPIRVModule *TheModule) {
    assert((TheBB || TheModule) && "Invalid BB or Module");
    if (TheBB)
      setBasicBlock(TheBB);
    else {
      setModule(TheModule);
    }
    setId(hasId() ? TheId : SPIRVID_INVALID);
    setType(hasType() ? TheType : nullptr);
    return this;
  }
  virtual void init() {}
  virtual void initImpl(Op OC, bool HasId = true, SPIRVWord WC = 0,
                        bool VariWC = false, unsigned Lit1 = ~0U,
                        unsigned Lit2 = ~0U, unsigned Lit3 = ~0U) {
    OpCode = OC;
    if (!HasId) {
      setHasNoId();
      setHasNoType();
    }
    if (WC)
      SPIRVEntry::setWordCount(WC);
    setHasVariableWordCount(VariWC);
    addLit(Lit1);
    addLit(Lit2);
    addLit(Lit3);
  }
  bool isOperandLiteral(unsigned I) const override { return Lit.count(I); }
  void addLit(unsigned L) {
    if (L != ~0U)
      Lit.insert(L);
  }
  /// \return Expected number of operands. If the instruction has variable
  /// number of words, return the minimum.
  SPIRVWord getExpectedNumOperands() const {
    assert(WordCount > 0 && "Word count not initialized");
    auto Exp = WordCount - 1;
    if (hasId())
      --Exp;
    if (hasType())
      --Exp;
    return Exp;
  }
  virtual void setOpWordsAndValidate(const std::vector<SPIRVWord> &TheOps) {
    setOpWords(TheOps);
    validate();
  }
  virtual void setOpWords(const std::vector<SPIRVWord> &TheOps) {
    SPIRVWord WC = TheOps.size() + 1;
    if (hasId())
      ++WC;
    if (hasType())
      ++WC;
    if (WordCount) {
      if (WordCount == WC) {
        // do nothing
      } else {
        assert(HasVariWC && WC >= WordCount && "Invalid word count");
        SPIRVEntry::setWordCount(WC);
      }
    } else
      SPIRVEntry::setWordCount(WC);
    Ops = TheOps;
  }
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    auto NumOps = WordCount - 1;
    if (hasId())
      --NumOps;
    if (hasType())
      --NumOps;
    Ops.resize(NumOps);
  }

  std::vector<SPIRVWord> &getOpWords() { return Ops; }

  const std::vector<SPIRVWord> &getOpWords() const { return Ops; }

  SPIRVWord getOpWord(int I) const { return Ops[I]; }

  /// Get operand as value.
  /// If the operand is a literal, return it as a uint32 constant.
  SPIRVValue *getOpValue(int I) {
    return isOperandLiteral(I) ? Module->getLiteralAsConstant(Ops[I])
                               : getValue(Ops[I]);
  }

  // Get the offset of operands.
  // Some instructions skip literals when returning operands.
  size_t getOperandOffset() const {
    if (hasExecScope() && !isGroupOpCode(OpCode) && !isPipeOpCode(OpCode))
      return 1;
    return 0;
  }

  // Get operands which are values.
  // Drop execution scope and group operation literals.
  // Return other literals as uint32 constants.
  std::vector<SPIRVValue *> getOperands() override {
    std::vector<SPIRVValue *> VOps;
    auto Offset = getOperandOffset();
    for (size_t I = 0, E = Ops.size() - Offset; I != E; ++I)
      VOps.push_back(getOperand(I));
    return VOps;
  }

  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    std::vector<SPIRVEntry *> Operands;
    for (size_t I = getOperandOffset(), E = Ops.size(); I < E; ++I)
      if (!isOperandLiteral(I))
        Operands.push_back(getEntry(Ops[I]));
    return Operands;
  }

  virtual SPIRVValue *getOperand(unsigned I) {
    return getOpValue(I + getOperandOffset());
  }

  bool hasExecScope() const { return SPIRV::hasExecScope(OpCode); }

  bool hasGroupOperation() const { return SPIRV::hasGroupOperation(OpCode); }

  bool getSPIRVGroupOperation(SPIRVGroupOperationKind &GroupOp) const {
    if (!hasGroupOperation())
      return false;
    GroupOp = static_cast<SPIRVGroupOperationKind>(Ops[1]);
    return true;
  }

  Scope getExecutionScope() const {
    if (!hasExecScope())
      return ScopeInvocation;
    return static_cast<Scope>(
        static_cast<SPIRVConstant *>(getValue(Ops[0]))->getZExtIntValue());
  }

  bool hasVariableWordCount() const { return HasVariWC; }

  void setHasVariableWordCount(bool VariWC) { HasVariWC = VariWC; }

protected:
  void encode(spv_ostream &O) const override {
    auto E = getEncoder(O);
    if (hasType())
      E << Type;
    if (hasId())
      E << Id;
    E << Ops;
  }
  void decode(std::istream &I) override {
    auto D = getDecoder(I);
    if (hasType())
      D >> Type;
    if (hasId())
      D >> Id;
    D >> Ops;
  }
  std::vector<SPIRVWord> Ops;
  bool HasVariWC;
  std::unordered_set<unsigned> Lit; // Literal operand index
};

template <typename BT = SPIRVInstTemplateBase, Op OC = OpNop, bool HasId = true,
          SPIRVWord WC = 0, bool HasVariableWC = false, unsigned Literal1 = ~0U,
          unsigned Literal2 = ~0U, unsigned Literal3 = ~0U>
class SPIRVInstTemplate : public BT {
public:
  typedef BT BaseTy;
  SPIRVInstTemplate() { init(); }
  ~SPIRVInstTemplate() override {}
  void init() override {
    this->initImpl(OC, HasId, WC, HasVariableWC, Literal1, Literal2, Literal3);
  }
};

class SPIRVMemoryAccess {
public:
  SPIRVMemoryAccess(const std::vector<SPIRVWord> &TheMemoryAccess)
      : TheMemoryAccessMask(0), Alignment(0) {
    memoryAccessUpdate(TheMemoryAccess);
  }

  SPIRVMemoryAccess() : TheMemoryAccessMask(0), Alignment(0) {}

  void memoryAccessUpdate(const std::vector<SPIRVWord> &MemoryAccess) {
    if (!MemoryAccess.size())
      return;
    assert((MemoryAccess.size() == 1 || MemoryAccess.size() == 2) &&
           "Invalid memory access operand size");
    TheMemoryAccessMask = MemoryAccess[0];
    if (MemoryAccess[0] & MemoryAccessAlignedMask) {
      assert(MemoryAccess.size() == 2 && "Alignment operand is missing");
      Alignment = MemoryAccess[1];
    }
  }
  SPIRVWord isVolatile() const {
    return getMemoryAccessMask() & MemoryAccessVolatileMask;
  }
  SPIRVWord isNonTemporal() const {
    return getMemoryAccessMask() & MemoryAccessNontemporalMask;
  }
  SPIRVWord getMemoryAccessMask() const { return TheMemoryAccessMask; }
  SPIRVWord getAlignment() const { return Alignment; }

protected:
  SPIRVWord TheMemoryAccessMask;
  SPIRVWord Alignment;
};

class SPIRVVariable : public SPIRVInstruction {
public:
  // Complete constructor for integer constant
  SPIRVVariable(SPIRVType *TheType, SPIRVId TheId, SPIRVValue *TheInitializer,
                const std::string &TheName,
                SPIRVStorageClassKind TheStorageClass, SPIRVBasicBlock *TheBB,
                SPIRVModule *TheM)
      : SPIRVInstruction(TheInitializer ? 5 : 4, OpVariable, TheType, TheId,
                         TheBB, TheM),
        StorageClass(TheStorageClass) {
    if (TheInitializer)
      Initializer.push_back(TheInitializer->getId());
    Name = TheName;
    validate();
  }
  // Incomplete constructor
  SPIRVVariable()
      : SPIRVInstruction(OpVariable), StorageClass(StorageClassFunction) {}

  SPIRVStorageClassKind getStorageClass() const { return StorageClass; }
  SPIRVValue *getInitializer() const {
    if (Initializer.empty())
      return nullptr;
    assert(Initializer.size() == 1);
    return getValue(Initializer[0]);
  }
  bool isConstant() const { return hasDecorate(DecorationConstant); }
  bool isBuiltin(SPIRVBuiltinVariableKind *BuiltinKind = nullptr) const {
    SPIRVWord Kind;
    bool Found = hasDecorate(DecorationBuiltIn, 0, &Kind);
    if (!Found)
      return false;
    if (BuiltinKind)
      *BuiltinKind = static_cast<SPIRVBuiltinVariableKind>(Kind);
    return true;
  }
  void setBuiltin(SPIRVBuiltinVariableKind Kind) {
    assert(isValid(Kind));
    addDecorate(new SPIRVDecorate(DecorationBuiltIn, this, Kind));
  }
  void setIsConstant(bool Is) {
    if (Is)
      addDecorate(new SPIRVDecorate(DecorationConstant, this));
    else
      eraseDecorate(DecorationConstant);
  }
  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    if (SPIRVValue *V = getInitializer())
      return std::vector<SPIRVEntry *>(1, V);
    return std::vector<SPIRVEntry *>();
  }

protected:
  void validate() const override {
    SPIRVValue::validate();
    assert(isValid(StorageClass));
    assert(Initializer.size() == 1 || Initializer.empty());
    assert(getType()->isTypePointer());
  }
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Initializer.resize(WordCount - 4);
  }
  _SPIRV_DEF_ENCDEC4(Type, Id, StorageClass, Initializer)

  SPIRVStorageClassKind StorageClass;
  std::vector<SPIRVId> Initializer;
};

class SPIRVStore : public SPIRVInstruction, public SPIRVMemoryAccess {
public:
  const static SPIRVWord FixedWords = 3;
  // Complete constructor
  SPIRVStore(SPIRVId PointerId, SPIRVId ValueId,
             const std::vector<SPIRVWord> &TheMemoryAccess,
             SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(FixedWords + TheMemoryAccess.size(), OpStore, TheBB),
        SPIRVMemoryAccess(TheMemoryAccess), MemoryAccess(TheMemoryAccess),
        PtrId(PointerId), ValId(ValueId) {
    setAttr();
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVStore()
      : SPIRVInstruction(OpStore), SPIRVMemoryAccess(), PtrId(SPIRVID_INVALID),
        ValId(SPIRVID_INVALID) {
    setAttr();
  }

  SPIRVValue *getSrc() const { return getValue(ValId); }
  SPIRVValue *getDst() const { return getValue(PtrId); }

protected:
  void setAttr() {
    setHasNoType();
    setHasNoId();
  }

  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    MemoryAccess.resize(TheWordCount - FixedWords);
  }
  void encode(spv_ostream &O) const override {
    getEncoder(O) << PtrId << ValId << MemoryAccess;
  }

  void decode(std::istream &I) override {
    getDecoder(I) >> PtrId >> ValId >> MemoryAccess;
    memoryAccessUpdate(MemoryAccess);
  }

  void validate() const override {
    SPIRVInstruction::validate();
    if (getSrc()->isForward() || getDst()->isForward())
      return;
    assert(getValueType(PtrId)->getPointerElementType() ==
               getValueType(ValId) &&
           "Inconsistent operand types");
  }

private:
  std::vector<SPIRVWord> MemoryAccess;
  SPIRVId PtrId;
  SPIRVId ValId;
};

class SPIRVLoad : public SPIRVInstruction, public SPIRVMemoryAccess {
public:
  const static SPIRVWord FixedWords = 4;
  // Complete constructor
  SPIRVLoad(SPIRVId TheId, SPIRVId PointerId,
            const std::vector<SPIRVWord> &TheMemoryAccess,
            SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(
            FixedWords + TheMemoryAccess.size(), OpLoad,
            TheBB->getValueType(PointerId)->getPointerElementType(), TheId,
            TheBB),
        SPIRVMemoryAccess(TheMemoryAccess), PtrId(PointerId),
        MemoryAccess(TheMemoryAccess) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVLoad()
      : SPIRVInstruction(OpLoad), SPIRVMemoryAccess(), PtrId(SPIRVID_INVALID) {}

  SPIRVValue *getSrc() const { return Module->get<SPIRVValue>(PtrId); }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    MemoryAccess.resize(TheWordCount - FixedWords);
  }

  void encode(spv_ostream &O) const override {
    getEncoder(O) << Type << Id << PtrId << MemoryAccess;
  }

  void decode(std::istream &I) override {
    getDecoder(I) >> Type >> Id >> PtrId >> MemoryAccess;
    memoryAccessUpdate(MemoryAccess);
  }

  void validate() const override {
    SPIRVInstruction::validate();
    assert((getValue(PtrId)->isForward() ||
            Type == getValueType(PtrId)->getPointerElementType()) &&
           "Inconsistent types");
  }

private:
  SPIRVId PtrId;
  std::vector<SPIRVWord> MemoryAccess;
};

class SPIRVBinary : public SPIRVInstTemplateBase {
protected:
  void validate() const override {
    SPIRVId Op1 = Ops[0];
    SPIRVId Op2 = Ops[1];
    SPIRVType *Op1Ty, *Op2Ty;
    SPIRVInstruction::validate();
    if (getValue(Op1)->isForward() || getValue(Op2)->isForward())
      return;
    if (getValueType(Op1)->isTypeVector()) {
      Op1Ty = getValueType(Op1)->getVectorComponentType();
      Op2Ty = getValueType(Op2)->getVectorComponentType();
      assert(getValueType(Op1)->getVectorComponentCount() ==
                 getValueType(Op2)->getVectorComponentCount() &&
             "Inconsistent Vector component width");
    } else {
      Op1Ty = getValueType(Op1);
      Op2Ty = getValueType(Op2);
    }

    (void)Op1Ty;
    (void)Op2Ty;
    if (isBinaryOpCode(OpCode)) {
      assert(getValueType(Op1) == getValueType(Op2) &&
             "Invalid type for binary instruction");
      assert((Op1Ty->isTypeInt() || Op2Ty->isTypeFloat()) &&
             "Invalid type for Binary instruction");
      assert((Op1Ty->getBitWidth() == Op2Ty->getBitWidth()) &&
             "Inconsistent BitWidth");
    } else if (isShiftOpCode(OpCode)) {
      assert((Op1Ty->isTypeInt() || Op2Ty->isTypeInt()) &&
             "Invalid type for shift instruction");
    } else if (isLogicalOpCode(OpCode)) {
      assert((Op1Ty->isTypeBool() || Op2Ty->isTypeBool()) &&
             "Invalid type for logical instruction");
    } else if (isBitwiseOpCode(OpCode)) {
      assert((Op1Ty->isTypeInt() || Op2Ty->isTypeInt()) &&
             "Invalid type for bitwise instruction");
      assert((Op1Ty->getIntegerBitWidth() == Op2Ty->getIntegerBitWidth()) &&
             "Inconsistent BitWidth");
    } else {
      assert(0 && "Invalid op code!");
    }
  }
};

template <Op OC>
class SPIRVBinaryInst
    : public SPIRVInstTemplate<SPIRVBinary, OC, true, 5, false> {};

/* ToDo: SMod and FMod to be added */
#define _SPIRV_OP(x) typedef SPIRVBinaryInst<Op##x> SPIRV##x;
_SPIRV_OP(IAdd)
_SPIRV_OP(FAdd)
_SPIRV_OP(ISub)
_SPIRV_OP(FSub)
_SPIRV_OP(IMul)
_SPIRV_OP(FMul)
_SPIRV_OP(UDiv)
_SPIRV_OP(SDiv)
_SPIRV_OP(FDiv)
_SPIRV_OP(SRem)
_SPIRV_OP(FRem)
_SPIRV_OP(UMod)
_SPIRV_OP(ShiftLeftLogical)
_SPIRV_OP(ShiftRightLogical)
_SPIRV_OP(ShiftRightArithmetic)
_SPIRV_OP(LogicalAnd)
_SPIRV_OP(LogicalOr)
_SPIRV_OP(LogicalEqual)
_SPIRV_OP(LogicalNotEqual)
_SPIRV_OP(BitwiseAnd)
_SPIRV_OP(BitwiseOr)
_SPIRV_OP(BitwiseXor)
_SPIRV_OP(Dot)
#undef _SPIRV_OP

template <Op TheOpCode> class SPIRVInstNoOperand : public SPIRVInstruction {
public:
  // Complete constructor
  SPIRVInstNoOperand(SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(1, TheOpCode, TheBB) {
    setAttr();
    validate();
  }
  // Incomplete constructor
  SPIRVInstNoOperand() : SPIRVInstruction(TheOpCode) { setAttr(); }

protected:
  void setAttr() {
    setHasNoId();
    setHasNoType();
  }
  _SPIRV_DEF_ENCDEC0
};

typedef SPIRVInstNoOperand<OpReturn> SPIRVReturn;
typedef SPIRVInstNoOperand<OpUnreachable> SPIRVUnreachable;

class SPIRVReturnValue : public SPIRVInstruction {
public:
  static const Op OC = OpReturnValue;
  // Complete constructor
  SPIRVReturnValue(SPIRVValue *TheReturnValue, SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(2, OC, TheBB), ReturnValueId(TheReturnValue->getId()) {
    setAttr();
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVReturnValue() : SPIRVInstruction(OC), ReturnValueId(SPIRVID_INVALID) {
    setAttr();
  }

  SPIRVValue *getReturnValue() const { return getValue(ReturnValueId); }

protected:
  void setAttr() {
    setHasNoId();
    setHasNoType();
  }
  _SPIRV_DEF_ENCDEC1(ReturnValueId)
  void validate() const override { SPIRVInstruction::validate(); }
  SPIRVId ReturnValueId;
};

class SPIRVBranch : public SPIRVInstruction {
public:
  static const Op OC = OpBranch;
  // Complete constructor
  SPIRVBranch(SPIRVLabel *TheTargetLabel, SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(2, OC, TheBB), TargetLabelId(TheTargetLabel->getId()) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVBranch() : SPIRVInstruction(OC), TargetLabelId(SPIRVID_INVALID) {
    setHasNoId();
    setHasNoType();
  }
  SPIRVValue *getTargetLabel() const { return getValue(TargetLabelId); }

protected:
  _SPIRV_DEF_ENCDEC1(TargetLabelId)
  void validate() const override {
    SPIRVInstruction::validate();
    assert(WordCount == 2);
    assert(OpCode == OC);
    assert(getTargetLabel()->isLabel() || getTargetLabel()->isForward());
  }
  SPIRVId TargetLabelId;
};

class SPIRVBranchConditional : public SPIRVInstruction {
public:
  static const Op OC = OpBranchConditional;
  // Complete constructor
  SPIRVBranchConditional(SPIRVValue *TheCondition, SPIRVLabel *TheTrueLabel,
                         SPIRVLabel *TheFalseLabel, SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(4, OC, TheBB), ConditionId(TheCondition->getId()),
        TrueLabelId(TheTrueLabel->getId()),
        FalseLabelId(TheFalseLabel->getId()) {
    validate();
  }
  SPIRVBranchConditional(SPIRVValue *TheCondition, SPIRVLabel *TheTrueLabel,
                         SPIRVLabel *TheFalseLabel, SPIRVBasicBlock *TheBB,
                         SPIRVWord TrueWeight, SPIRVWord FalseWeight)
      : SPIRVInstruction(6, OC, TheBB), ConditionId(TheCondition->getId()),
        TrueLabelId(TheTrueLabel->getId()),
        FalseLabelId(TheFalseLabel->getId()) {
    BranchWeights.push_back(TrueWeight);
    BranchWeights.push_back(FalseWeight);
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVBranchConditional()
      : SPIRVInstruction(OC), ConditionId(SPIRVID_INVALID),
        TrueLabelId(SPIRVID_INVALID), FalseLabelId(SPIRVID_INVALID) {
    setHasNoId();
    setHasNoType();
  }
  SPIRVValue *getCondition() const { return getValue(ConditionId); }
  SPIRVLabel *getTrueLabel() const { return get<SPIRVLabel>(TrueLabelId); }
  SPIRVLabel *getFalseLabel() const { return get<SPIRVLabel>(FalseLabelId); }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    BranchWeights.resize(TheWordCount - 4);
  }
  _SPIRV_DEF_ENCDEC4(ConditionId, TrueLabelId, FalseLabelId, BranchWeights)
  void validate() const override {
    SPIRVInstruction::validate();
    assert(WordCount == 4 || WordCount == 6);
    assert(WordCount == BranchWeights.size() + 4);
    assert(OpCode == OC);
    assert(getCondition()->isForward() ||
           getCondition()->getType()->isTypeBool());
    assert(getTrueLabel()->isForward() || getTrueLabel()->isLabel());
    assert(getFalseLabel()->isForward() || getFalseLabel()->isLabel());
  }
  SPIRVId ConditionId;
  SPIRVId TrueLabelId;
  SPIRVId FalseLabelId;
  std::vector<SPIRVWord> BranchWeights;
};

class SPIRVPhi : public SPIRVInstruction {
public:
  static const Op OC = OpPhi;
  static const SPIRVWord FixedWordCount = 3;
  SPIRVPhi(SPIRVType *TheType, SPIRVId TheId,
           const std::vector<SPIRVValue *> &ThePairs, SPIRVBasicBlock *BB)
      : SPIRVInstruction(ThePairs.size() + FixedWordCount, OC, TheType, TheId,
                         BB) {
    Pairs = getIds(ThePairs);
    validate();
    assert(BB && "Invalid BB");
  }
  SPIRVPhi() : SPIRVInstruction(OC) {}
  std::vector<SPIRVValue *> getPairs() { return getValues(Pairs); }
  void addPair(SPIRVValue *Value, SPIRVBasicBlock *BB) {
    Pairs.push_back(Value->getId());
    Pairs.push_back(BB->getId());
    WordCount = Pairs.size() + FixedWordCount;
    validate();
  }
  void setPairs(const std::vector<SPIRVValue *> &ThePairs) {
    Pairs = getIds(ThePairs);
    WordCount = Pairs.size() + FixedWordCount;
    validate();
  }
  void foreachPair(
      std::function<void(SPIRVValue *, SPIRVBasicBlock *, size_t)> Func) {
    for (size_t I = 0, E = Pairs.size() / 2; I != E; ++I) {
      SPIRVEntry *Value, *BB;
      if (!Module->exist(Pairs[2 * I], &Value) ||
          !Module->exist(Pairs[2 * I + 1], &BB))
        continue;
      Func(static_cast<SPIRVValue *>(Value), static_cast<SPIRVBasicBlock *>(BB),
           I);
    }
  }
  void
  foreachPair(std::function<void(SPIRVValue *, SPIRVBasicBlock *)> Func) const {
    for (size_t I = 0, E = Pairs.size() / 2; I != E; ++I) {
      SPIRVEntry *Value, *BB;
      if (!Module->exist(Pairs[2 * I], &Value) ||
          !Module->exist(Pairs[2 * I + 1], &BB))
        continue;
      Func(static_cast<SPIRVValue *>(Value),
           static_cast<SPIRVBasicBlock *>(BB));
    }
  }
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Pairs.resize(TheWordCount - FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC3(Type, Id, Pairs)
  void validate() const override {
    assert(WordCount == Pairs.size() + FixedWordCount);
    assert(OpCode == OC);
    assert(Pairs.size() % 2 == 0);
    foreachPair([=](SPIRVValue *IncomingV, SPIRVBasicBlock *IncomingBB) {
      assert(IncomingV->isForward() || IncomingV->getType() == Type);
      assert(IncomingBB->isBasicBlock() || IncomingBB->isForward());
    });
    SPIRVInstruction::validate();
  }

protected:
  std::vector<SPIRVId> Pairs;
};

class SPIRVCompare : public SPIRVInstTemplateBase {
protected:
  void validate() const override {
    auto Op1 = Ops[0];
    auto Op2 = Ops[1];
    SPIRVType *Op1Ty, *Op2Ty, *ResTy;
    SPIRVInstruction::validate();
    if (getValue(Op1)->isForward() || getValue(Op2)->isForward())
      return;

    (void)Op1Ty;
    (void)Op2Ty;
    (void)ResTy;
    if (getValueType(Op1)->isTypeVector()) {
      Op1Ty = getValueType(Op1)->getVectorComponentType();
      Op2Ty = getValueType(Op2)->getVectorComponentType();
      ResTy = Type->getVectorComponentType();
      assert(getValueType(Op1)->getVectorComponentCount() ==
                 getValueType(Op2)->getVectorComponentCount() &&
             "Inconsistent Vector component width");
    } else {
      Op1Ty = getValueType(Op1);
      Op2Ty = getValueType(Op2);
      ResTy = Type;
    }
    assert(isCmpOpCode(OpCode) && "Invalid op code for cmp inst");
    assert((ResTy->isTypeBool() || ResTy->isTypeInt()) &&
           "Invalid type for compare instruction");
    assert(Op1Ty == Op2Ty && "Inconsistent types");
  }
};

template <Op OC>
class SPIRVCmpInst
    : public SPIRVInstTemplate<SPIRVCompare, OC, true, 5, false> {};

#define _SPIRV_OP(x) typedef SPIRVCmpInst<Op##x> SPIRV##x;
_SPIRV_OP(IEqual)
_SPIRV_OP(FOrdEqual)
_SPIRV_OP(FUnordEqual)
_SPIRV_OP(INotEqual)
_SPIRV_OP(FOrdNotEqual)
_SPIRV_OP(FUnordNotEqual)
_SPIRV_OP(ULessThan)
_SPIRV_OP(SLessThan)
_SPIRV_OP(FOrdLessThan)
_SPIRV_OP(FUnordLessThan)
_SPIRV_OP(UGreaterThan)
_SPIRV_OP(SGreaterThan)
_SPIRV_OP(FOrdGreaterThan)
_SPIRV_OP(FUnordGreaterThan)
_SPIRV_OP(ULessThanEqual)
_SPIRV_OP(SLessThanEqual)
_SPIRV_OP(FOrdLessThanEqual)
_SPIRV_OP(FUnordLessThanEqual)
_SPIRV_OP(UGreaterThanEqual)
_SPIRV_OP(SGreaterThanEqual)
_SPIRV_OP(FOrdGreaterThanEqual)
_SPIRV_OP(FUnordGreaterThanEqual)
_SPIRV_OP(LessOrGreater)
_SPIRV_OP(Ordered)
_SPIRV_OP(Unordered)
#undef _SPIRV_OP

class SPIRVSelect : public SPIRVInstruction {
public:
  // Complete constructor
  SPIRVSelect(SPIRVId TheId, SPIRVId TheCondition, SPIRVId TheOp1,
              SPIRVId TheOp2, SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(6, OpSelect, TheBB->getValueType(TheOp1), TheId,
                         TheBB),
        Condition(TheCondition), Op1(TheOp1), Op2(TheOp2) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVSelect()
      : SPIRVInstruction(OpSelect), Condition(SPIRVID_INVALID),
        Op1(SPIRVID_INVALID), Op2(SPIRVID_INVALID) {}
  SPIRVValue *getCondition() { return getValue(Condition); }
  SPIRVValue *getTrueValue() { return getValue(Op1); }
  SPIRVValue *getFalseValue() { return getValue(Op2); }

protected:
  _SPIRV_DEF_ENCDEC5(Type, Id, Condition, Op1, Op2)
  void validate() const override {
    SPIRVInstruction::validate();
    if (getValue(Condition)->isForward() || getValue(Op1)->isForward() ||
        getValue(Op2)->isForward())
      return;

    SPIRVType *ConTy = getValueType(Condition)->isTypeVector()
                           ? getValueType(Condition)->getVectorComponentType()
                           : getValueType(Condition);
    (void)ConTy;
    assert(ConTy->isTypeBool() && "Invalid type");
    assert(getType() == getValueType(Op1) && getType() == getValueType(Op2) &&
           "Inconsistent type");
  }
  SPIRVId Condition;
  SPIRVId Op1;
  SPIRVId Op2;
};

class SPIRVSelectionMerge : public SPIRVInstruction {
public:
  static const Op OC = OpSelectionMerge;
  static const SPIRVWord FixedWordCount = 3;

  SPIRVSelectionMerge(SPIRVId TheMergeBlock, SPIRVWord TheSelectionControl,
                      SPIRVBasicBlock *BB)
      : SPIRVInstruction(3, OC, BB), MergeBlock(TheMergeBlock),
        SelectionControl(TheSelectionControl) {
    validate();
    assert(BB && "Invalid BB");
  }

  SPIRVSelectionMerge()
      : SPIRVInstruction(OC), MergeBlock(SPIRVID_INVALID),
        SelectionControl(SPIRVWORD_MAX) {
    setHasNoId();
    setHasNoType();
  }

  SPIRVId getMergeBlock() { return MergeBlock; }
  SPIRVWord getSelectionControl() { return SelectionControl; }

  _SPIRV_DEF_ENCDEC2(MergeBlock, SelectionControl)

protected:
  SPIRVId MergeBlock;
  SPIRVWord SelectionControl;
};

class SPIRVLoopMerge : public SPIRVInstruction {
public:
  static const Op OC = OpLoopMerge;
  static const SPIRVWord FixedWordCount = 4;

  SPIRVLoopMerge(SPIRVId TheMergeBlock, SPIRVId TheContinueTarget,
                 SPIRVWord TheLoopControl,
                 std::vector<SPIRVWord> TheLoopControlParameters,
                 SPIRVBasicBlock *BB)
      : SPIRVInstruction(FixedWordCount + TheLoopControlParameters.size(), OC,
                         BB),
        MergeBlock(TheMergeBlock), ContinueTarget(TheContinueTarget),
        LoopControl(TheLoopControl),
        LoopControlParameters(TheLoopControlParameters) {
    validate();
    assert(BB && "Invalid BB");
  }

  SPIRVLoopMerge()
      : SPIRVInstruction(OC), MergeBlock(SPIRVID_MAX),
        LoopControl(SPIRVWORD_MAX) {
    setHasNoId();
    setHasNoType();
  }

  SPIRVId getMergeBlock() { return MergeBlock; }
  SPIRVId getContinueTarget() { return ContinueTarget; }
  SPIRVWord getLoopControl() { return LoopControl; }

  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    LoopControlParameters.resize(TheWordCount - FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC4(MergeBlock, ContinueTarget, LoopControl,
                     LoopControlParameters)

protected:
  SPIRVId MergeBlock;
  SPIRVId ContinueTarget;
  SPIRVWord LoopControl;
  std::vector<SPIRVWord> LoopControlParameters;
};

class SPIRVSwitch : public SPIRVInstruction {
public:
  static const Op OC = OpSwitch;
  static const SPIRVWord FixedWordCount = 3;
  typedef std::vector<SPIRVWord> LiteralTy;
  typedef std::pair<LiteralTy, SPIRVBasicBlock *> PairTy;

  SPIRVSwitch(SPIRVValue *TheSelect, SPIRVBasicBlock *TheDefault,
              const std::vector<PairTy> &ThePairs, SPIRVBasicBlock *BB)
      : SPIRVInstruction(FixedWordCount, OC, BB), Select(TheSelect->getId()),
        Default(TheDefault->getId()) {

    if (!ThePairs.empty())
      SPIRVEntry::setWordCount(
          ThePairs.size() * (ThePairs.at(0).first.size() + 1) + FixedWordCount);

    for (auto &I : ThePairs) {
      for (auto &U : I.first)
        Pairs.push_back(U);
      Pairs.push_back(I.second->getId());
    }
    validate();
    assert(BB && "Invalid BB");
  }
  SPIRVSwitch()
      : SPIRVInstruction(OC), Select(SPIRVWORD_MAX), Default(SPIRVWORD_MAX) {
    setHasNoId();
    setHasNoType();
  }
  std::vector<SPIRVValue *> getPairs() { return getValues(Pairs); }
  SPIRVValue *getSelect() const { return getValue(Select); }
  SPIRVBasicBlock *getDefault() const {
    return static_cast<SPIRVBasicBlock *>(getValue(Default));
  }
  size_t getLiteralSize() const {
    unsigned ByteWidth = getSelect()->getType()->getBitWidth() / 8;
    unsigned Remainder = (ByteWidth % sizeof(SPIRVWord)) != 0;
    return (ByteWidth / sizeof(SPIRVWord)) + Remainder;
  }
  size_t getPairSize() const { return getLiteralSize() + 1; }
  size_t getNumPairs() const { return Pairs.size() / getPairSize(); }
  void
  foreachPair(std::function<void(LiteralTy, SPIRVBasicBlock *)> Func) const {
    unsigned PairSize = getPairSize();
    for (size_t I = 0, E = getNumPairs(); I != E; ++I) {
      SPIRVEntry *BB;
      LiteralTy Literals;
      if (!Module->exist(Pairs[PairSize * I + getLiteralSize()], &BB))
        continue;

      for (size_t J = 0; J < getLiteralSize(); ++J) {
        Literals.push_back(Pairs.at(PairSize * I + J));
      }
      Func(Literals, static_cast<SPIRVBasicBlock *>(BB));
    }
  }
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Pairs.resize(TheWordCount - FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC3(Select, Default, Pairs)
  void validate() const override {
    assert(WordCount == Pairs.size() + FixedWordCount);
    assert(OpCode == OC);
    assert(Pairs.size() % getPairSize() == 0);
    foreachPair([=](LiteralTy Literals, SPIRVBasicBlock *BB) {
      assert(BB->isBasicBlock() || BB->isForward());
    });
    SPIRVInstruction::validate();
  }

protected:
  SPIRVId Select;
  SPIRVId Default;
  std::vector<SPIRVWord> Pairs;
};

class SPIRVFMod : public SPIRVInstruction {
public:
  static const Op OC = OpFMod;
  static const SPIRVWord FixedWordCount = 4;
  // Complete constructor
  SPIRVFMod(SPIRVType *TheType, SPIRVId TheId, SPIRVId TheDividend,
            SPIRVId TheDivisor, SPIRVBasicBlock *BB)
      : SPIRVInstruction(5, OC, TheType, TheId, BB), Dividend(TheDividend),
        Divisor(TheDivisor) {
    validate();
    assert(BB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVFMod()
      : SPIRVInstruction(OC), Dividend(SPIRVID_INVALID),
        Divisor(SPIRVID_INVALID) {}
  SPIRVValue *getDividend() const { return getValue(Dividend); }
  SPIRVValue *getDivisor() const { return getValue(Divisor); }

  std::vector<SPIRVValue *> getOperands() override {
    std::vector<SPIRVId> Operands;
    Operands.push_back(Dividend);
    Operands.push_back(Divisor);
    return getValues(Operands);
  }

  void setWordCount(SPIRVWord FixedWordCount) override {
    SPIRVEntry::setWordCount(FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC4(Type, Id, Dividend, Divisor)
  void validate() const override {
    SPIRVInstruction::validate();
    if (getValue(Dividend)->isForward() || getValue(Divisor)->isForward())
      return;
    SPIRVInstruction::validate();
  }

protected:
  SPIRVId Dividend;
  SPIRVId Divisor;
};

class SPIRVVectorTimesScalar : public SPIRVInstruction {
public:
  static const Op OC = OpVectorTimesScalar;
  static const SPIRVWord FixedWordCount = 4;
  // Complete constructor
  SPIRVVectorTimesScalar(SPIRVType *TheType, SPIRVId TheId, SPIRVId TheVector,
                         SPIRVId TheScalar, SPIRVBasicBlock *BB)
      : SPIRVInstruction(5, OC, TheType, TheId, BB), Vector(TheVector),
        Scalar(TheScalar) {
    validate();
    assert(BB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVVectorTimesScalar()
      : SPIRVInstruction(OC), Vector(SPIRVID_INVALID), Scalar(SPIRVID_INVALID) {
  }
  SPIRVValue *getVector() const { return getValue(Vector); }
  SPIRVValue *getScalar() const { return getValue(Scalar); }

  std::vector<SPIRVValue *> getOperands() override {
    std::vector<SPIRVId> Operands;
    Operands.push_back(Vector);
    Operands.push_back(Scalar);
    return getValues(Operands);
  }

  void setWordCount(SPIRVWord FixedWordCount) override {
    SPIRVEntry::setWordCount(FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC4(Type, Id, Vector, Scalar)
  void validate() const override {
    SPIRVInstruction::validate();
    if (getValue(Vector)->isForward() || getValue(Scalar)->isForward())
      return;

    assert(getValueType(Vector)->isTypeVector() &&
           getValueType(Vector)->getVectorComponentType()->isTypeFloat() &&
           "First operand must be a vector of floating-point type");
    assert(getValueType(getId())->isTypeVector() &&
           getValueType(getId())->getVectorComponentType()->isTypeFloat() &&
           "Result type must be a vector of floating-point type");
    assert(
        getValueType(Vector)->getVectorComponentType() ==
            getValueType(getId())->getVectorComponentType() &&
        "Scalar must have the same type as the Component Type in Result Type");
    SPIRVInstruction::validate();
  }

protected:
  SPIRVId Vector;
  SPIRVId Scalar;
};

class SPIRVUnary : public SPIRVInstTemplateBase {
protected:
  void validate() const override {
    auto Op = Ops[0];
    SPIRVInstruction::validate();
    if (getValue(Op)->isForward())
      return;
    if (isGenericNegateOpCode(OpCode)) {
      SPIRVType *ResTy =
          Type->isTypeVector() ? Type->getVectorComponentType() : Type;
      SPIRVType *OpTy = Type->isTypeVector()
                            ? getValueType(Op)->getVectorComponentType()
                            : getValueType(Op);

      (void)ResTy;
      (void)OpTy;
      assert(getType() == getValueType(Op) && "Inconsistent type");
      assert((ResTy->isTypeInt() || ResTy->isTypeFloat()) &&
             "Invalid type for Generic Negate instruction");
      assert((ResTy->getBitWidth() == OpTy->getBitWidth()) &&
             "Invalid bitwidth for Generic Negate instruction");
      assert((Type->isTypeVector()
                  ? (Type->getVectorComponentCount() ==
                     getValueType(Op)->getVectorComponentCount())
                  : 1) &&
             "Invalid vector component Width for Generic Negate instruction");
    }
  }
};

template <Op OC>
class SPIRVUnaryInst
    : public SPIRVInstTemplate<SPIRVUnary, OC, true, 4, false> {};

#define _SPIRV_OP(x) typedef SPIRVUnaryInst<Op##x> SPIRV##x;
_SPIRV_OP(ConvertFToU)
_SPIRV_OP(ConvertFToS)
_SPIRV_OP(ConvertSToF)
_SPIRV_OP(ConvertUToF)
_SPIRV_OP(UConvert)
_SPIRV_OP(SConvert)
_SPIRV_OP(FConvert)
_SPIRV_OP(SatConvertSToU)
_SPIRV_OP(SatConvertUToS)
_SPIRV_OP(ConvertPtrToU)
_SPIRV_OP(ConvertUToPtr)
_SPIRV_OP(PtrCastToGeneric)
_SPIRV_OP(GenericCastToPtr)
_SPIRV_OP(Bitcast)
_SPIRV_OP(SNegate)
_SPIRV_OP(FNegate)
_SPIRV_OP(Not)
_SPIRV_OP(LogicalNot)
_SPIRV_OP(IsNan)
_SPIRV_OP(IsInf)
_SPIRV_OP(IsFinite)
_SPIRV_OP(IsNormal)
_SPIRV_OP(SignBitSet)
_SPIRV_OP(Any)
_SPIRV_OP(All)
#undef _SPIRV_OP

class SPIRVAccessChainBase : public SPIRVInstTemplateBase {
public:
  SPIRVValue *getBase() { return this->getValue(this->Ops[0]); }
  std::vector<SPIRVValue *> getIndices() const {
    std::vector<SPIRVWord> IndexWords(this->Ops.begin() + 1, this->Ops.end());
    return this->getValues(IndexWords);
  }
  bool isInBounds() {
    return OpCode == OpInBoundsAccessChain ||
           OpCode == OpInBoundsPtrAccessChain;
  }
  bool hasPtrIndex() {
    return OpCode == OpPtrAccessChain || OpCode == OpInBoundsPtrAccessChain;
  }
};

template <Op OC, unsigned FixedWC>
class SPIRVAccessChainGeneric
    : public SPIRVInstTemplate<SPIRVAccessChainBase, OC, true, FixedWC, true> {
};

typedef SPIRVAccessChainGeneric<OpAccessChain, 4> SPIRVAccessChain;
typedef SPIRVAccessChainGeneric<OpInBoundsAccessChain, 4>
    SPIRVInBoundsAccessChain;
typedef SPIRVAccessChainGeneric<OpPtrAccessChain, 5> SPIRVPtrAccessChain;
typedef SPIRVAccessChainGeneric<OpInBoundsPtrAccessChain, 5>
    SPIRVInBoundsPtrAccessChain;

template <Op OC, SPIRVWord FixedWordCount>
class SPIRVFunctionCallGeneric : public SPIRVInstruction {
public:
  SPIRVFunctionCallGeneric(SPIRVType *TheType, SPIRVId TheId,
                           const std::vector<SPIRVWord> &TheArgs,
                           SPIRVBasicBlock *BB)
      : SPIRVInstruction(TheArgs.size() + FixedWordCount, OC, TheType, TheId,
                         BB),
        Args(TheArgs) {
    SPIRVFunctionCallGeneric::validate();
    assert(BB && "Invalid BB");
  }
  SPIRVFunctionCallGeneric(SPIRVType *TheType, SPIRVId TheId,
                           const std::vector<SPIRVValue *> &TheArgs,
                           SPIRVBasicBlock *BB)
      : SPIRVInstruction(TheArgs.size() + FixedWordCount, OC, TheType, TheId,
                         BB) {
    Args = getIds(TheArgs);
    SPIRVFunctionCallGeneric::validate();
    assert(BB && "Invalid BB");
  }

  SPIRVFunctionCallGeneric(SPIRVModule *BM, SPIRVWord ResId, SPIRVType *TheType,
                           const std::vector<SPIRVWord> &TheArgs)
      : SPIRVInstruction(TheArgs.size() + FixedWordCount, OC, TheType, ResId,
                         BM),
        Args(TheArgs) {}

  SPIRVFunctionCallGeneric() : SPIRVInstruction(OC) {}
  const std::vector<SPIRVWord> &getArguments() const { return Args; }
  void setArguments(const std::vector<SPIRVWord> &A) {
    Args = A;
    setWordCount(Args.size() + FixedWordCount);
  }
  std::vector<SPIRVValue *> getArgumentValues() { return getValues(Args); }
  std::vector<SPIRVType *> getArgumentValueTypes() const {
    std::vector<SPIRVType *> ArgTypes;
    for (auto &I : Args)
      ArgTypes.push_back(getValue(I)->getType());
    return ArgTypes;
  }
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Args.resize(TheWordCount - FixedWordCount);
  }
  void validate() const override { SPIRVInstruction::validate(); }

protected:
  std::vector<SPIRVWord> Args;
};

class SPIRVFunctionCall : public SPIRVFunctionCallGeneric<OpFunctionCall, 4> {
public:
  SPIRVFunctionCall(SPIRVId TheId, SPIRVFunction *TheFunction,
                    const std::vector<SPIRVWord> &TheArgs, SPIRVBasicBlock *BB);
  SPIRVFunctionCall() : FunctionId(SPIRVID_INVALID) {}
  SPIRVFunction *getFunction() const { return get<SPIRVFunction>(FunctionId); }
  _SPIRV_DEF_ENCDEC4(Type, Id, FunctionId, Args)
  void validate() const override;
  bool isOperandLiteral(unsigned Index) const override { return false; }

protected:
  SPIRVId FunctionId;
};

class SPIRVExtInst : public SPIRVFunctionCallGeneric<OpExtInst, 5> {
public:
  SPIRVExtInst(SPIRVType *TheType, SPIRVId TheId, SPIRVId TheBuiltinSet,
               SPIRVWord TheEntryPoint, const std::vector<SPIRVWord> &TheArgs,
               SPIRVBasicBlock *BB)
      : SPIRVFunctionCallGeneric(TheType, TheId, TheArgs, BB),
        ExtSetId(TheBuiltinSet), ExtOp(TheEntryPoint) {
    setExtSetKindById();
    SPIRVExtInst::validate();
  }
  SPIRVExtInst(SPIRVType *TheType, SPIRVId TheId, SPIRVId TheBuiltinSet,
               SPIRVWord TheEntryPoint,
               const std::vector<SPIRVValue *> &TheArgs, SPIRVBasicBlock *BB)
      : SPIRVFunctionCallGeneric(TheType, TheId, TheArgs, BB),
        ExtSetId(TheBuiltinSet), ExtOp(TheEntryPoint) {
    setExtSetKindById();
    SPIRVExtInst::validate();
  }

  SPIRVExtInst(SPIRVModule *BM, SPIRVId ResId, SPIRVType *TheType,
               SPIRVExtInstSetKind SetKind, SPIRVWord SetId, SPIRVWord InstId,
               const std::vector<SPIRVWord> &Args)
      : SPIRVFunctionCallGeneric(BM, ResId, TheType, Args), ExtSetKind(SetKind),
        ExtSetId(SetId), ExtOp(InstId) {}

  SPIRVExtInst(SPIRVExtInstSetKind SetKind = SPIRVEIS_Count,
               unsigned ExtOC = SPIRVWORD_MAX)
      : ExtSetKind(SetKind), ExtSetId(SPIRVWORD_MAX), ExtOp(ExtOC) {}
  void setExtSetId(unsigned Set) { ExtSetId = Set; }
  void setExtOp(unsigned ExtOC) { ExtOp = ExtOC; }
  SPIRVId getExtSetId() const { return ExtSetId; }
  SPIRVWord getExtOp() const { return ExtOp; }

  SPIRVExtInstSetKind getExtSetKind() const { return ExtSetKind; }

  void setExtSetKindById() {
    assert(Module && "Invalid module");
    ExtSetKind = Module->getBuiltinSet(ExtSetId);
    assert((ExtSetKind == SPIRVEIS_OpenCL || ExtSetKind == SPIRVEIS_Debug) &&
           "not supported");
  }
  void encode(spv_ostream &O) const override {
    getEncoder(O) << Type << Id << ExtSetId;
    switch (ExtSetKind) {
    case SPIRVEIS_OpenCL:
      getEncoder(O) << ExtOpOCL;
      break;
    case SPIRVEIS_Debug:
      getEncoder(O) << ExtOpDebug;
      break;
    default:
      assert(0 && "not supported");
      getEncoder(O) << ExtOp;
    }
    getEncoder(O) << Args;
  }
  void decode(std::istream &I) override {
    getDecoder(I) >> Type >> Id >> ExtSetId;
    setExtSetKindById();
    switch (ExtSetKind) {
    case SPIRVEIS_OpenCL:
      getDecoder(I) >> ExtOpOCL;
      break;
    case SPIRVEIS_Debug:
      getDecoder(I) >> ExtOpDebug;
      break;
    default:
      assert(0 && "not supported");
      getDecoder(I) >> ExtOp;
    }
    getDecoder(I) >> Args;
  }
  void validate() const override {
    SPIRVFunctionCallGeneric::validate();
    validateBuiltin(ExtSetId, ExtOp);
  }
  bool isOperandLiteral(unsigned Index) const override {
    assert(ExtSetKind == SPIRVEIS_OpenCL &&
           "Unsupported extended instruction set");
    auto EOC = static_cast<OCLExtOpKind>(ExtOp);
    switch (EOC) {
    default:
      return false;
    case OpenCLLIB::Vloadn:
    case OpenCLLIB::Vload_halfn:
    case OpenCLLIB::Vloada_halfn:
      return Index == 2;
    case OpenCLLIB::Vstore_half_r:
    case OpenCLLIB::Vstore_halfn_r:
    case OpenCLLIB::Vstorea_halfn_r:
      return Index == 3;
    }
  }

protected:
  SPIRVExtInstSetKind ExtSetKind;
  SPIRVId ExtSetId;
  union {
    SPIRVWord ExtOp;
    OCLExtOpKind ExtOpOCL;
    SPIRVDebugExtOpKind ExtOpDebug;
  };
};

class SPIRVCompositeConstruct : public SPIRVInstruction {
public:
  const static Op OC = OpCompositeConstruct;
  const static SPIRVWord FixedWordCount = 3;
  // Complete constructor
  SPIRVCompositeConstruct(SPIRVType *TheType, SPIRVId TheId,
                          const std::vector<SPIRVId> &TheConstituents,
                          SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(TheConstituents.size() + FixedWordCount, OC, TheType,
                         TheId, TheBB),
        Constituents(TheConstituents) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVCompositeConstruct() : SPIRVInstruction(OC) {}

  const std::vector<SPIRVValue *> getConstituents() const {
    return getValues(Constituents);
  }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Constituents.resize(TheWordCount - FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC3(Type, Id, Constituents)
  void validate() const override {
    SPIRVInstruction::validate();
    switch (getValueType(this->getId())->getOpCode()) {
    case OpTypeVector:
      assert(getConstituents().size() > 1 &&
             "There must be at least two Constituent operands in vector");
    case OpTypeArray:
    case OpTypeStruct:
      break;
    default:
      static_assert("Invalid type", "");
    }
  }
  std::vector<SPIRVId> Constituents;
};

class SPIRVCompositeExtract : public SPIRVInstruction {
public:
  const static Op OC = OpCompositeExtract;
  // Complete constructor
  SPIRVCompositeExtract(SPIRVType *TheType, SPIRVId TheId,
                        SPIRVValue *TheComposite,
                        const std::vector<SPIRVWord> &TheIndices,
                        SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(TheIndices.size() + 4, OC, TheType, TheId, TheBB),
        Composite(TheComposite->getId()), Indices(TheIndices) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVCompositeExtract() : SPIRVInstruction(OC), Composite(SPIRVID_INVALID) {}

  SPIRVValue *getComposite() { return getValue(Composite); }
  const std::vector<SPIRVWord> &getIndices() const { return Indices; }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Indices.resize(TheWordCount - 4);
  }
  _SPIRV_DEF_ENCDEC4(Type, Id, Composite, Indices)
  // ToDo: validate the result type is consistent with the base type and indices
  // need to trace through the base type for struct types
  void validate() const override {
    SPIRVInstruction::validate();
    assert(getValueType(Composite)->isTypeArray() ||
           getValueType(Composite)->isTypeStruct() ||
           getValueType(Composite)->isTypeVector());
  }
  SPIRVId Composite;
  std::vector<SPIRVWord> Indices;
};

class SPIRVCompositeInsert : public SPIRVInstruction {
public:
  const static Op OC = OpCompositeInsert;
  const static SPIRVWord FixedWordCount = 5;
  // Complete constructor
  SPIRVCompositeInsert(SPIRVId TheId, SPIRVValue *TheObject,
                       SPIRVValue *TheComposite,
                       const std::vector<SPIRVWord> &TheIndices,
                       SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(TheIndices.size() + FixedWordCount, OC,
                         TheComposite->getType(), TheId, TheBB),
        Object(TheObject->getId()), Composite(TheComposite->getId()),
        Indices(TheIndices) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVCompositeInsert()
      : SPIRVInstruction(OC), Object(SPIRVID_INVALID),
        Composite(SPIRVID_INVALID) {}

  SPIRVValue *getObject() { return getValue(Object); }
  SPIRVValue *getComposite() { return getValue(Composite); }
  const std::vector<SPIRVWord> &getIndices() const { return Indices; }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Indices.resize(TheWordCount - FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC5(Type, Id, Object, Composite, Indices)
  // ToDo: validate the object type is consistent with the base type and indices
  // need to trace through the base type for struct types
  void validate() const override {
    SPIRVInstruction::validate();
    assert(OpCode == OC);
    assert(WordCount == Indices.size() + FixedWordCount);
    assert(getValueType(Composite)->isTypeArray() ||
           getValueType(Composite)->isTypeStruct() ||
           getValueType(Composite)->isTypeVector());
    assert(Type == getValueType(Composite));
  }
  SPIRVId Object;
  SPIRVId Composite;
  std::vector<SPIRVWord> Indices;
};

class SPIRVCopyObject : public SPIRVInstruction {
public:
  const static Op OC = OpCopyObject;

  // Complete constructor
  SPIRVCopyObject(SPIRVType *TheType, SPIRVId TheId, SPIRVValue *TheOperand,
                  SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(4, OC, TheType, TheId, TheBB),
        Operand(TheOperand->getId()) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVCopyObject() : SPIRVInstruction(OC), Operand(SPIRVID_INVALID) {}

  SPIRVValue *getOperand() { return getValue(Operand); }

protected:
  _SPIRV_DEF_ENCDEC3(Type, Id, Operand)

  void validate() const override { SPIRVInstruction::validate(); }
  SPIRVId Operand;
};

class SPIRVCopyMemory : public SPIRVInstruction, public SPIRVMemoryAccess {
public:
  const static Op OC = OpCopyMemory;
  const static SPIRVWord FixedWords = 3;
  // Complete constructor
  SPIRVCopyMemory(SPIRVValue *TheTarget, SPIRVValue *TheSource,
                  const std::vector<SPIRVWord> &TheMemoryAccess,
                  SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(FixedWords + TheMemoryAccess.size(), OC, TheBB),
        SPIRVMemoryAccess(TheMemoryAccess), MemoryAccess(TheMemoryAccess),
        Target(TheTarget->getId()), Source(TheSource->getId()) {
    validate();
    assert(TheBB && "Invalid BB");
  }

  // Incomplete constructor
  SPIRVCopyMemory()
      : SPIRVInstruction(OC), SPIRVMemoryAccess(), Target(SPIRVID_INVALID),
        Source(SPIRVID_INVALID) {
    setHasNoId();
    setHasNoType();
  }

  SPIRVValue *getSource() { return getValue(Source); }
  SPIRVValue *getTarget() { return getValue(Target); }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    MemoryAccess.resize(TheWordCount - FixedWords);
  }

  void encode(spv_ostream &O) const override {
    getEncoder(O) << Target << Source << MemoryAccess;
  }

  void decode(std::istream &I) override {
    getDecoder(I) >> Target >> Source >> MemoryAccess;
    memoryAccessUpdate(MemoryAccess);
  }

  void validate() const override {
    assert((getValueType(Id) == getValueType(Source)) && "Inconsistent type");
    assert(getValueType(Id)->isTypePointer() && "Invalid type");
    assert(!(getValueType(Id)->getPointerElementType()->isTypeVoid()) &&
           "Invalid type");
    SPIRVInstruction::validate();
  }

  std::vector<SPIRVWord> MemoryAccess;
  SPIRVId Target;
  SPIRVId Source;
};

class SPIRVCopyMemorySized : public SPIRVInstruction, public SPIRVMemoryAccess {
public:
  const static Op OC = OpCopyMemorySized;
  const static SPIRVWord FixedWords = 4;
  // Complete constructor
  SPIRVCopyMemorySized(SPIRVValue *TheTarget, SPIRVValue *TheSource,
                       SPIRVValue *TheSize,
                       const std::vector<SPIRVWord> &TheMemoryAccess,
                       SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(FixedWords + TheMemoryAccess.size(), OC, TheBB),
        SPIRVMemoryAccess(TheMemoryAccess), MemoryAccess(TheMemoryAccess),
        Target(TheTarget->getId()), Source(TheSource->getId()),
        Size(TheSize->getId()) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVCopyMemorySized()
      : SPIRVInstruction(OC), SPIRVMemoryAccess(), Target(SPIRVID_INVALID),
        Source(SPIRVID_INVALID), Size(0) {
    setHasNoId();
    setHasNoType();
  }

  SPIRVValue *getSource() { return getValue(Source); }
  SPIRVValue *getTarget() { return getValue(Target); }
  SPIRVValue *getSize() { return getValue(Size); }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    MemoryAccess.resize(TheWordCount - FixedWords);
  }

  void encode(spv_ostream &O) const override {
    getEncoder(O) << Target << Source << Size << MemoryAccess;
  }

  void decode(std::istream &I) override {
    getDecoder(I) >> Target >> Source >> Size >> MemoryAccess;
    memoryAccessUpdate(MemoryAccess);
  }

  void validate() const override { SPIRVInstruction::validate(); }

  std::vector<SPIRVWord> MemoryAccess;
  SPIRVId Target;
  SPIRVId Source;
  SPIRVId Size;
};

class SPIRVVectorExtractDynamic : public SPIRVInstruction {
public:
  const static Op OC = OpVectorExtractDynamic;
  // Complete constructor
  SPIRVVectorExtractDynamic(SPIRVId TheId, SPIRVValue *TheVector,
                            SPIRVValue *TheIndex, SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(5, OC, TheVector->getType()->getVectorComponentType(),
                         TheId, TheBB),
        VectorId(TheVector->getId()), IndexId(TheIndex->getId()) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVVectorExtractDynamic()
      : SPIRVInstruction(OC), VectorId(SPIRVID_INVALID),
        IndexId(SPIRVID_INVALID) {}

  SPIRVValue *getVector() { return getValue(VectorId); }
  SPIRVValue *getIndex() const { return getValue(IndexId); }

protected:
  _SPIRV_DEF_ENCDEC4(Type, Id, VectorId, IndexId)
  void validate() const override {
    SPIRVInstruction::validate();
    if (getValue(VectorId)->isForward())
      return;
    assert(getValueType(VectorId)->isTypeVector());
  }
  SPIRVId VectorId;
  SPIRVId IndexId;
};

class SPIRVVectorInsertDynamic : public SPIRVInstruction {
public:
  const static Op OC = OpVectorInsertDynamic;
  // Complete constructor
  SPIRVVectorInsertDynamic(SPIRVId TheId, SPIRVValue *TheVector,
                           SPIRVValue *TheComponent, SPIRVValue *TheIndex,
                           SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(6, OC, TheVector->getType(), TheId, TheBB),
        VectorId(TheVector->getId()), IndexId(TheIndex->getId()),
        ComponentId(TheComponent->getId()) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVVectorInsertDynamic()
      : SPIRVInstruction(OC), VectorId(SPIRVID_INVALID),
        IndexId(SPIRVID_INVALID), ComponentId(SPIRVID_INVALID) {}

  SPIRVValue *getVector() { return getValue(VectorId); }
  SPIRVValue *getIndex() const { return getValue(IndexId); }
  SPIRVValue *getComponent() { return getValue(ComponentId); }

protected:
  _SPIRV_DEF_ENCDEC5(Type, Id, VectorId, ComponentId, IndexId)
  void validate() const override {
    SPIRVInstruction::validate();
    if (getValue(VectorId)->isForward())
      return;
    assert(getValueType(VectorId)->isTypeVector());
  }
  SPIRVId VectorId;
  SPIRVId IndexId;
  SPIRVId ComponentId;
};

class SPIRVVectorShuffle : public SPIRVInstruction {
public:
  const static Op OC = OpVectorShuffle;
  const static SPIRVWord FixedWordCount = 5;
  // Complete constructor
  SPIRVVectorShuffle(SPIRVId TheId, SPIRVType *TheType, SPIRVValue *TheVector1,
                     SPIRVValue *TheVector2,
                     const std::vector<SPIRVWord> &TheComponents,
                     SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(TheComponents.size() + FixedWordCount, OC, TheType,
                         TheId, TheBB),
        Vector1(TheVector1->getId()), Vector2(TheVector2->getId()),
        Components(TheComponents) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVVectorShuffle()
      : SPIRVInstruction(OC), Vector1(SPIRVID_INVALID),
        Vector2(SPIRVID_INVALID) {}

  SPIRVValue *getVector1() { return getValue(Vector1); }
  SPIRVValue *getVector2() { return getValue(Vector2); }
  const std::vector<SPIRVWord> &getComponents() const { return Components; }

protected:
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Components.resize(TheWordCount - FixedWordCount);
  }
  _SPIRV_DEF_ENCDEC5(Type, Id, Vector1, Vector2, Components)
  void validate() const override {
    SPIRVInstruction::validate();
    assert(OpCode == OC);
    assert(WordCount == Components.size() + FixedWordCount);
    assert(Type->isTypeVector());
    assert(Type->getVectorComponentType() ==
           getValueType(Vector1)->getVectorComponentType());
    if (getValue(Vector1)->isForward() || getValue(Vector2)->isForward())
      return;
    assert(getValueType(Vector1) == getValueType(Vector2));
    assert(Components.size() == Type->getVectorComponentCount());
    assert(Components.size() > 1);
  }
  SPIRVId Vector1;
  SPIRVId Vector2;
  std::vector<SPIRVWord> Components;
};

class SPIRVControlBarrier : public SPIRVInstruction {
public:
  static const Op OC = OpControlBarrier;
  // Complete constructor
  SPIRVControlBarrier(SPIRVValue *TheScope, SPIRVValue *TheMemScope,
                      SPIRVValue *TheMemSema, SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(4, OC, TheBB), ExecScope(TheScope->getId()),
        MemScope(TheMemScope->getId()), MemSema(TheMemSema->getId()) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVControlBarrier() : SPIRVInstruction(OC), ExecScope(ScopeInvocation) {
    setHasNoId();
    setHasNoType();
  }
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
  }
  SPIRVValue *getExecScope() const { return getValue(ExecScope); }
  SPIRVValue *getMemScope() const { return getValue(MemScope); }
  SPIRVValue *getMemSemantic() const { return getValue(MemSema); }
  std::vector<SPIRVValue *> getOperands() override {
    std::vector<SPIRVId> Operands;
    Operands.push_back(ExecScope);
    Operands.push_back(MemScope);
    Operands.push_back(MemSema);
    return getValues(Operands);
  }

protected:
  _SPIRV_DEF_ENCDEC3(ExecScope, MemScope, MemSema)
  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == 4);
    SPIRVInstruction::validate();
  }
  SPIRVId ExecScope;
  SPIRVId MemScope;
  SPIRVId MemSema;
};

template <Op OC> class SPIRVLifetime : public SPIRVInstruction {
public:
  // Complete constructor
  SPIRVLifetime(SPIRVId TheObject, SPIRVWord TheSize, SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(3, OC, TheBB), Object(TheObject), Size(TheSize) {
    validate();
    assert(TheBB && "Invalid BB");
  };
  // Incomplete constructor
  SPIRVLifetime()
      : SPIRVInstruction(OC), Object(SPIRVID_INVALID), Size(SPIRVWORD_MAX) {
    setHasNoId();
    setHasNoType();
  }
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityKernel);
  }
  SPIRVValue *getObject() { return getValue(Object); };
  SPIRVWord getSize() { return Size; };

protected:
  void validate() const override {
    auto ObjType = getValue(Object)->getType();
    // Type must be an OpTypePointer with Storage Class Function.
    assert(ObjType->isTypePointer() && "Objects type must be a pointer");
    assert(static_cast<SPIRVTypePointer *>(ObjType)->getStorageClass() ==
               StorageClassFunction &&
           "Invalid storage class");
    // Size must be 0 if Pointer is a pointer to a non-void type or the
    // Addresses capability is not being used. If Size is non-zero, it is the
    // number of bytes of memory whose lifetime is starting. Its type must be an
    // integer type scalar. It is treated as unsigned; if its type has
    // Signedness of 1, its sign bit cannot be set.
    if (!(ObjType->getPointerElementType()->isTypeVoid() ||
          // (void *) is i8* in LLVM IR
          ObjType->getPointerElementType()->isTypeInt(8)) ||
        !Module->hasCapability(CapabilityAddresses))
      assert(Size == 0 && "Size must be 0");
  }
  _SPIRV_DEF_ENCDEC2(Object, Size)
  SPIRVId Object;
  SPIRVWord Size;
};

typedef SPIRVLifetime<OpLifetimeStart> SPIRVLifetimeStart;
typedef SPIRVLifetime<OpLifetimeStop> SPIRVLifetimeStop;

class SPIRVGroupAsyncCopy : public SPIRVInstruction {
public:
  static const Op OC = OpGroupAsyncCopy;
  static const SPIRVWord WC = 9;
  // Complete constructor
  SPIRVGroupAsyncCopy(SPIRVValue *TheScope, SPIRVId TheId, SPIRVValue *TheDest,
                      SPIRVValue *TheSrc, SPIRVValue *TheNumElems,
                      SPIRVValue *TheStride, SPIRVValue *TheEvent,
                      SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(WC, OC, TheEvent->getType(), TheId, TheBB),
        ExecScope(TheScope->getId()), Destination(TheDest->getId()),
        Source(TheSrc->getId()), NumElements(TheNumElems->getId()),
        Stride(TheStride->getId()), Event(TheEvent->getId()) {
    validate();
    assert(TheBB && "Invalid BB");
  }
  // Incomplete constructor
  SPIRVGroupAsyncCopy()
      : SPIRVInstruction(OC), ExecScope(SPIRVID_INVALID),
        Destination(SPIRVID_INVALID), Source(SPIRVID_INVALID),
        NumElements(SPIRVID_INVALID), Stride(SPIRVID_INVALID),
        Event(SPIRVID_INVALID) {}
  SPIRVValue *getExecScope() const { return getValue(ExecScope); }
  SPIRVValue *getDestination() const { return getValue(Destination); }
  SPIRVValue *getSource() const { return getValue(Source); }
  SPIRVValue *getNumElements() const { return getValue(NumElements); }
  SPIRVValue *getStride() const { return getValue(Stride); }
  SPIRVValue *getEvent() const { return getValue(Event); }
  std::vector<SPIRVValue *> getOperands() override {
    std::vector<SPIRVId> Operands;
    Operands.push_back(Destination);
    Operands.push_back(Source);
    Operands.push_back(NumElements);
    Operands.push_back(Stride);
    Operands.push_back(Event);
    return getValues(Operands);
  }

protected:
  _SPIRV_DEF_ENCDEC8(Type, Id, ExecScope, Destination, Source, NumElements,
                     Stride, Event)
  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == WC);
    SPIRVInstruction::validate();
  }
  SPIRVId ExecScope;
  SPIRVId Destination;
  SPIRVId Source;
  SPIRVId NumElements;
  SPIRVId Stride;
  SPIRVId Event;
};

enum SPIRVOpKind { SPIRVOPK_Id, SPIRVOPK_Literal, SPIRVOPK_Count };

class SPIRVDevEnqInstBase : public SPIRVInstTemplateBase {
public:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityDeviceEnqueue);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVDevEnqInstBase, Op##x, __VA_ARGS__> SPIRV##x;
// CL 2.0 enqueue kernel builtins
_SPIRV_OP(EnqueueMarker, true, 7)
_SPIRV_OP(EnqueueKernel, true, 13, true)
_SPIRV_OP(GetKernelNDrangeSubGroupCount, true, 8)
_SPIRV_OP(GetKernelNDrangeMaxSubGroupSize, true, 8)
_SPIRV_OP(GetKernelWorkGroupSize, true, 7)
_SPIRV_OP(GetKernelPreferredWorkGroupSizeMultiple, true, 7)
_SPIRV_OP(RetainEvent, false, 2)
_SPIRV_OP(ReleaseEvent, false, 2)
_SPIRV_OP(CreateUserEvent, true, 3)
_SPIRV_OP(IsValidEvent, true, 4)
_SPIRV_OP(SetUserEventStatus, false, 3)
_SPIRV_OP(CaptureEventProfilingInfo, false, 4)
_SPIRV_OP(GetDefaultQueue, true, 3)
_SPIRV_OP(BuildNDRange, true, 6)
#undef _SPIRV_OP

class SPIRVPipeInstBase : public SPIRVInstTemplateBase {
public:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityPipes);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVPipeInstBase, Op##x, __VA_ARGS__> SPIRV##x;
// CL 2.0 pipe builtins
_SPIRV_OP(ReadPipe, true, 7)
_SPIRV_OP(WritePipe, true, 7)
_SPIRV_OP(ReservedReadPipe, true, 9)
_SPIRV_OP(ReservedWritePipe, true, 9)
_SPIRV_OP(ReserveReadPipePackets, true, 7)
_SPIRV_OP(ReserveWritePipePackets, true, 7)
_SPIRV_OP(CommitReadPipe, false, 5)
_SPIRV_OP(CommitWritePipe, false, 5)
_SPIRV_OP(IsValidReserveId, true, 4)
_SPIRV_OP(GetNumPipePackets, true, 6)
_SPIRV_OP(GetMaxPipePackets, true, 6)
#undef _SPIRV_OP

class SPIRVPipeStorageInstBase : public SPIRVInstTemplateBase {
public:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityPipeStorage, CapabilityPipes);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVPipeStorageInstBase, Op##x, __VA_ARGS__>      \
      SPIRV##x;

_SPIRV_OP(CreatePipeFromPipeStorage, true, 4)
#undef _SPIRV_OP

class SPIRVGroupInstBase : public SPIRVInstTemplateBase {
public:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityGroups);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVGroupInstBase, Op##x, __VA_ARGS__> SPIRV##x;
// Group instructions
_SPIRV_OP(GroupWaitEvents, false, 4)
_SPIRV_OP(GroupAll, true, 5)
_SPIRV_OP(GroupAny, true, 5)
_SPIRV_OP(GroupBroadcast, true, 6)
_SPIRV_OP(GroupIAdd, true, 6, false, 1)
_SPIRV_OP(GroupFAdd, true, 6, false, 1)
_SPIRV_OP(GroupFMin, true, 6, false, 1)
_SPIRV_OP(GroupUMin, true, 6, false, 1)
_SPIRV_OP(GroupSMin, true, 6, false, 1)
_SPIRV_OP(GroupFMax, true, 6, false, 1)
_SPIRV_OP(GroupUMax, true, 6, false, 1)
_SPIRV_OP(GroupSMax, true, 6, false, 1)
_SPIRV_OP(GroupReserveReadPipePackets, true, 8)
_SPIRV_OP(GroupReserveWritePipePackets, true, 8)
_SPIRV_OP(GroupCommitReadPipe, false, 6)
_SPIRV_OP(GroupCommitWritePipe, false, 6)
#undef _SPIRV_OP

class SPIRVAtomicInstBase : public SPIRVInstTemplateBase {
public:
  SPIRVCapVec getRequiredCapability() const override {
    SPIRVCapVec CapVec;
    // Most of atomic instructions do not require any capabilities
    // ... unless they operate on 64-bit integers.
    if (hasType() && getType()->isTypeInt(64)) {
      // In SPIRV 1.2 spec only 2 atomic instructions have no result type:
      // 1. OpAtomicStore - need to check type of the Value operand
      // 2. OpAtomicFlagClear - doesn't require Int64Atomics capability.
      CapVec.push_back(CapabilityInt64Atomics);
    }
    // Per the spec OpAtomicCompareExchangeWeak, OpAtomicFlagTestAndSet and
    // OpAtomicFlagClear instructions require kernel capability. But this
    // capability should be added by setting OpenCL memory model.
    return CapVec;
  }

  // Overriding the following method only because of OpAtomicStore.
  // We have to declare Int64Atomics capability if the Value operand is int64.
  void setOpWords(const std::vector<SPIRVWord> &TheOps) override {
    SPIRVInstTemplateBase::setOpWords(TheOps);
    static const unsigned ValueOperandIndex = 3;
    if (getOpCode() == OpAtomicStore &&
        getOperand(ValueOperandIndex)->getType()->isTypeInt(64))
      Module->addCapability(CapabilityInt64Atomics);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVAtomicInstBase, Op##x, __VA_ARGS__> SPIRV##x;
// Atomic builtins
_SPIRV_OP(AtomicFlagTestAndSet, true, 6)
_SPIRV_OP(AtomicFlagClear, false, 4)
_SPIRV_OP(AtomicLoad, true, 6)
_SPIRV_OP(AtomicStore, false, 5)
_SPIRV_OP(AtomicExchange, true, 7)
_SPIRV_OP(AtomicCompareExchange, true, 9)
_SPIRV_OP(AtomicCompareExchangeWeak, true, 9)
_SPIRV_OP(AtomicIIncrement, true, 6)
_SPIRV_OP(AtomicIDecrement, true, 6)
_SPIRV_OP(AtomicIAdd, true, 7)
_SPIRV_OP(AtomicISub, true, 7)
_SPIRV_OP(AtomicUMin, true, 7)
_SPIRV_OP(AtomicUMax, true, 7)
_SPIRV_OP(AtomicSMin, true, 7)
_SPIRV_OP(AtomicSMax, true, 7)
_SPIRV_OP(AtomicAnd, true, 7)
_SPIRV_OP(AtomicOr, true, 7)
_SPIRV_OP(AtomicXor, true, 7)
_SPIRV_OP(MemoryBarrier, false, 3)
#undef _SPIRV_OP

class SPIRVImageInstBase : public SPIRVInstTemplateBase {
public:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityImageBasic);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVImageInstBase, Op##x, __VA_ARGS__> SPIRV##x;
// Image instructions
_SPIRV_OP(SampledImage, true, 5)
_SPIRV_OP(ImageSampleImplicitLod, true, 5, true)
_SPIRV_OP(ImageSampleExplicitLod, true, 7, true, 2)
_SPIRV_OP(ImageRead, true, 5, true, 2)
_SPIRV_OP(ImageWrite, false, 4, true, 3)
_SPIRV_OP(ImageQueryFormat, true, 4)
_SPIRV_OP(ImageQueryOrder, true, 4)
_SPIRV_OP(ImageQuerySizeLod, true, 5)
_SPIRV_OP(ImageQuerySize, true, 4)
_SPIRV_OP(ImageQueryLod, true, 5)
_SPIRV_OP(ImageQueryLevels, true, 4)
_SPIRV_OP(ImageQuerySamples, true, 4)
#undef _SPIRV_OP

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVInstTemplateBase, Op##x, __VA_ARGS__> SPIRV##x;
// Other instructions
_SPIRV_OP(SpecConstantOp, true, 4, true, 0)
_SPIRV_OP(GenericPtrMemSemantics, true, 4, false)
_SPIRV_OP(GenericCastToPtrExplicit, true, 5, false, 1)
#undef _SPIRV_OP

class SPIRVSubgroupShuffleINTELInstBase : public SPIRVInstTemplateBase {
protected:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupShuffleINTEL);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVSubgroupShuffleINTELInstBase, Op##x,          \
                            __VA_ARGS__>                                       \
      SPIRV##x;
// Intel Subgroup Shuffle Instructions
_SPIRV_OP(SubgroupShuffleINTEL, true, 5)
_SPIRV_OP(SubgroupShuffleDownINTEL, true, 6)
_SPIRV_OP(SubgroupShuffleUpINTEL, true, 6)
_SPIRV_OP(SubgroupShuffleXorINTEL, true, 5)
#undef _SPIRV_OP

class SPIRVSubgroupBufferBlockIOINTELInstBase : public SPIRVInstTemplateBase {
protected:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupBufferBlockIOINTEL);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVSubgroupBufferBlockIOINTELInstBase, Op##x,    \
                            __VA_ARGS__>                                       \
      SPIRV##x;
// Intel Subgroup Buffer Block Read and Write Instructions
_SPIRV_OP(SubgroupBlockReadINTEL, true, 4)
_SPIRV_OP(SubgroupBlockWriteINTEL, false, 3)
#undef _SPIRV_OP

class SPIRVSubgroupImageBlockIOINTELInstBase : public SPIRVInstTemplateBase {
protected:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupImageBlockIOINTEL);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVSubgroupImageBlockIOINTELInstBase, Op##x,     \
                            __VA_ARGS__>                                       \
      SPIRV##x;
// Intel Subgroup Image Block Read and Write Instructions
_SPIRV_OP(SubgroupImageBlockReadINTEL, true, 5)
_SPIRV_OP(SubgroupImageBlockWriteINTEL, false, 4)
#undef _SPIRV_OP

class SPIRVSubgroupImageMediaBlockIOINTELInstBase
    : public SPIRVInstTemplateBase {
protected:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupImageMediaBlockIOINTEL);
  }
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVSubgroupImageMediaBlockIOINTELInstBase,       \
                            Op##x, __VA_ARGS__>                                \
      SPIRV##x;
// Intel Subgroup Image Media Block Read and Write Instructions
_SPIRV_OP(SubgroupImageMediaBlockReadINTEL, true, 7)
_SPIRV_OP(SubgroupImageMediaBlockWriteINTEL, false, 6)
#undef _SPIRV_OP

class SPIRVSubgroupAVCIntelInstBase : public SPIRVInstTemplateBase {
protected:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupAvcMotionEstimationINTEL);
  }

  SPIRVExtSet getRequiredExtensions() const override {
    return getSet(SPV_INTEL_device_side_avc_motion_estimation);
  }
};

// Intel Subgroup AVC Motion Estimation Instructions
typedef SPIRVInstTemplate<SPIRVSubgroupAVCIntelInstBase, OpVmeImageINTEL, true,
                          5>
    SPIRVVmeImageINTEL;

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVSubgroupAVCIntelInstBase,                     \
                            OpSubgroupAvc##x##INTEL, __VA_ARGS__>              \
      SPIRVSubgroupAvc##x##INTEL;
_SPIRV_OP(MceGetDefaultInterBaseMultiReferencePenalty, true, 5)
_SPIRV_OP(MceSetInterBaseMultiReferencePenalty, true, 5)
_SPIRV_OP(MceGetDefaultInterShapePenalty, true, 5)
_SPIRV_OP(MceSetInterShapePenalty, true, 5)
_SPIRV_OP(MceGetDefaultInterDirectionPenalty, true, 5)
_SPIRV_OP(MceSetInterDirectionPenalty, true, 5)
_SPIRV_OP(MceGetDefaultInterMotionVectorCostTable, true, 5)
_SPIRV_OP(MceGetDefaultHighPenaltyCostTable, true, 3)
_SPIRV_OP(MceGetDefaultMediumPenaltyCostTable, true, 3)
_SPIRV_OP(MceGetDefaultLowPenaltyCostTable, true, 3)
_SPIRV_OP(MceSetMotionVectorCostFunction, true, 7)
_SPIRV_OP(MceSetAcOnlyHaar, true, 4)
_SPIRV_OP(MceSetSourceInterlacedFieldPolarity, true, 5)
_SPIRV_OP(MceSetSingleReferenceInterlacedFieldPolarity, true, 5)
_SPIRV_OP(MceSetDualReferenceInterlacedFieldPolarities, true, 6)
_SPIRV_OP(MceConvertToImePayload, true, 4)
_SPIRV_OP(MceConvertToImeResult, true, 4)
_SPIRV_OP(MceConvertToRefPayload, true, 4)
_SPIRV_OP(MceConvertToRefResult, true, 4)
_SPIRV_OP(MceConvertToSicPayload, true, 4)
_SPIRV_OP(MceConvertToSicResult, true, 4)
_SPIRV_OP(MceGetMotionVectors, true, 4)
_SPIRV_OP(MceGetInterDistortions, true, 4)
_SPIRV_OP(MceGetBestInterDistortions, true, 4)
_SPIRV_OP(MceGetInterMajorShape, true, 4)
_SPIRV_OP(MceGetInterMinorShape, true, 4)
_SPIRV_OP(MceGetInterDirections, true, 4)
_SPIRV_OP(MceGetInterMotionVectorCount, true, 4)
_SPIRV_OP(MceGetInterReferenceIds, true, 4)
_SPIRV_OP(MceGetInterReferenceInterlacedFieldPolarities, true, 6)
_SPIRV_OP(ImeInitialize, true, 6)
_SPIRV_OP(ImeSetSingleReference, true, 6)
_SPIRV_OP(ImeSetDualReference, true, 7)
_SPIRV_OP(ImeRefWindowSize, true, 5)
_SPIRV_OP(ImeAdjustRefOffset, true, 7)
_SPIRV_OP(ImeConvertToMcePayload, true, 4)
_SPIRV_OP(ImeSetMaxMotionVectorCount, true, 5)
_SPIRV_OP(ImeSetUnidirectionalMixDisable, true, 4)
_SPIRV_OP(ImeSetEarlySearchTerminationThreshold, true, 5)
_SPIRV_OP(ImeSetWeightedSad, true, 5)
_SPIRV_OP(ImeEvaluateWithSingleReference, true, 6)
_SPIRV_OP(ImeEvaluateWithDualReference, true, 7)
_SPIRV_OP(ImeEvaluateWithSingleReferenceStreamin, true, 7)
_SPIRV_OP(ImeEvaluateWithDualReferenceStreamin, true, 8)
_SPIRV_OP(ImeEvaluateWithSingleReferenceStreamout, true, 6)
_SPIRV_OP(ImeEvaluateWithDualReferenceStreamout, true, 7)
_SPIRV_OP(ImeEvaluateWithSingleReferenceStreaminout, true, 7)
_SPIRV_OP(ImeEvaluateWithDualReferenceStreaminout, true, 8)
_SPIRV_OP(ImeConvertToMceResult, true, 4)
_SPIRV_OP(ImeGetSingleReferenceStreamin, true, 4)
_SPIRV_OP(ImeGetDualReferenceStreamin, true, 4)
_SPIRV_OP(ImeStripSingleReferenceStreamout, true, 4)
_SPIRV_OP(ImeStripDualReferenceStreamout, true, 4)
_SPIRV_OP(ImeGetStreamoutSingleReferenceMajorShapeMotionVectors, true, 5)
_SPIRV_OP(ImeGetStreamoutSingleReferenceMajorShapeDistortions, true, 5)
_SPIRV_OP(ImeGetStreamoutSingleReferenceMajorShapeReferenceIds, true, 5)
_SPIRV_OP(ImeGetStreamoutDualReferenceMajorShapeMotionVectors, true, 6)
_SPIRV_OP(ImeGetStreamoutDualReferenceMajorShapeDistortions, true, 6)
_SPIRV_OP(ImeGetStreamoutDualReferenceMajorShapeReferenceIds, true, 6)
_SPIRV_OP(ImeGetBorderReached, true, 5)
_SPIRV_OP(ImeGetTruncatedSearchIndication, true, 4)
_SPIRV_OP(ImeGetUnidirectionalEarlySearchTermination, true, 4)
_SPIRV_OP(ImeGetWeightingPatternMinimumMotionVector, true, 4)
_SPIRV_OP(ImeGetWeightingPatternMinimumDistortion, true, 4)
_SPIRV_OP(FmeInitialize, true, 10)
_SPIRV_OP(BmeInitialize, true, 11)
_SPIRV_OP(RefConvertToMcePayload, true, 4)
_SPIRV_OP(RefSetBidirectionalMixDisable, true, 4)
_SPIRV_OP(RefSetBilinearFilterEnable, true, 4)
_SPIRV_OP(RefEvaluateWithSingleReference, true, 6)
_SPIRV_OP(RefEvaluateWithDualReference, true, 7)
_SPIRV_OP(RefEvaluateWithMultiReference, true, 6)
_SPIRV_OP(RefEvaluateWithMultiReferenceInterlaced, true, 7)
_SPIRV_OP(RefConvertToMceResult, true, 4)
_SPIRV_OP(SicInitialize, true, 4)
_SPIRV_OP(SicConfigureSkc, true, 9)
_SPIRV_OP(SicGetMotionVectorMask, true, 5)
_SPIRV_OP(SicConvertToMcePayload, true, 4)
_SPIRV_OP(SicSetIntraLumaShapePenalty, true, 5)
_SPIRV_OP(SicSetBilinearFilterEnable, true, 4)
_SPIRV_OP(SicSetSkcForwardTransformEnable, true, 5)
_SPIRV_OP(SicSetBlockBasedRawSkipSad, true, 5)
_SPIRV_OP(SicEvaluateWithSingleReference, true, 6)
_SPIRV_OP(SicEvaluateWithDualReference, true, 7)
_SPIRV_OP(SicEvaluateWithMultiReference, true, 6)
_SPIRV_OP(SicEvaluateWithMultiReferenceInterlaced, true, 7)
_SPIRV_OP(SicConvertToMceResult, true, 4)
_SPIRV_OP(SicGetInterRawSads, true, 4)
#undef _SPIRV_OP

class SPIRVSubgroupAVCIntelInstBaseIntra : public SPIRVInstTemplateBase {
protected:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupAvcMotionEstimationIntraINTEL);
  }
};

// Intel Subgroup AVC Motion Estimation Intra Instructions
#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVSubgroupAVCIntelInstBaseIntra,                \
                            OpSubgroupAvc##x##INTEL, __VA_ARGS__>              \
      SPIRVSubgroupAvc##x##INTEL;
_SPIRV_OP(MceGetDefaultIntraLumaShapePenalty, true, 5)
_SPIRV_OP(MceGetDefaultIntraLumaModePenalty, true, 5)
_SPIRV_OP(MceGetDefaultNonDcLumaIntraPenalty, true, 3)
_SPIRV_OP(SicConfigureIpeLuma, true, 11)
_SPIRV_OP(SicSetIntraLumaModeCostFunction, true, 7)
_SPIRV_OP(SicEvaluateIpe, true, 5)
_SPIRV_OP(SicGetIpeLumaShape, true, 4)
_SPIRV_OP(SicGetBestIpeLumaDistortion, true, 4)
_SPIRV_OP(SicGetPackedIpeLumaModes, true, 4)
_SPIRV_OP(SicGetPackedSkcLumaCountThreshold, true, 4)
_SPIRV_OP(SicGetPackedSkcLumaSumThreshold, true, 4)
#undef _SPIRV_OP

class SPIRVSubgroupAVCIntelInstBaseChroma : public SPIRVInstTemplateBase {
protected:
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupAvcMotionEstimationChromaINTEL);
  }
};

// Intel Subgroup AVC Motion Estimation Chroma Instructions
#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVInstTemplate<SPIRVSubgroupAVCIntelInstBaseChroma,               \
                            OpSubgroupAvc##x##INTEL, __VA_ARGS__>              \
      SPIRVSubgroupAvc##x##INTEL;
_SPIRV_OP(MceGetDefaultIntraChromaModeBasePenalty, true, 3)
_SPIRV_OP(SicConfigureIpeLumaChroma, true, 14)
_SPIRV_OP(SicSetIntraChromaModeCostFunction, true, 5)
_SPIRV_OP(SicGetBestIpeChromaDistortion, true, 4)
_SPIRV_OP(SicGetIpeChromaMode, true, 4)
#undef _SPIRV_OP

SPIRVSpecConstantOp *createSpecConstantOpInst(SPIRVInstruction *Inst);
SPIRVInstruction *createInstFromSpecConstantOp(SPIRVSpecConstantOp *C);
} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVINSTRUCTION_H
