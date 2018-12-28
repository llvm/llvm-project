//===- SPIRVValue.h - Class to represent a SPIR-V Value ---------*- C++ -*-===//
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
/// This file defines the values defined in SPIR-V spec with op codes.
///
/// The name of the SPIR-V values follow the op code name in the spec.
/// This is for readability and ease of using macro to handle types.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVVALUE_H
#define SPIRV_LIBSPIRV_SPIRVVALUE_H

#include "SPIRVDecorate.h"
#include "SPIRVEntry.h"
#include "SPIRVType.h"

#include <iostream>
#include <map>
#include <memory>

namespace SPIRV {

class SPIRVValue : public SPIRVEntry {
public:
  // Complete constructor for value with id and type
  SPIRVValue(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode,
             SPIRVType *TheType, SPIRVId TheId)
      : SPIRVEntry(M, TheWordCount, TheOpCode, TheId), Type(TheType) {
    validate();
  }
  // Complete constructor for value with type but without id
  SPIRVValue(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode,
             SPIRVType *TheType)
      : SPIRVEntry(M, TheWordCount, TheOpCode), Type(TheType) {
    setHasNoId();
    SPIRVValue::validate();
  }
  // Complete constructor for value with id but without type
  SPIRVValue(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode, SPIRVId TheId)
      : SPIRVEntry(M, TheWordCount, TheOpCode, TheId), Type(NULL) {
    setHasNoType();
    SPIRVValue::validate();
  }
  // Complete constructor for value without id and type
  SPIRVValue(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode)
      : SPIRVEntry(M, TheWordCount, TheOpCode), Type(NULL) {
    setHasNoId();
    setHasNoType();
    SPIRVValue::validate();
  }
  // Incomplete constructor
  SPIRVValue(Op TheOpCode) : SPIRVEntry(TheOpCode), Type(NULL) {}

  bool hasType() const { return !(Attrib & SPIRVEA_NOTYPE); }
  SPIRVType *getType() const {
    assert(hasType() && "value has no type");
    return Type;
  }
  bool isVolatile() const;
  bool hasAlignment(SPIRVWord *Result = 0) const;
  bool hasNoSignedWrap() const;
  bool hasNoUnsignedWrap() const;

  void setAlignment(SPIRVWord);
  void setVolatile(bool IsVolatile);
  void setNoSignedWrap(bool HasNoSignedWrap);
  void setNoUnsignedWrap(bool HasNoUnsignedWrap);

  void validate() const override {
    SPIRVEntry::validate();
    assert((!hasType() || Type) && "Invalid type");
  }

  void setType(SPIRVType *Ty) {
    Type = Ty;
    assert(!Ty || !Ty->isTypeVoid() || OpCode == OpFunction);
    if (Ty && (!Ty->isTypeVoid() || OpCode == OpFunction))
      setHasType();
    else
      setHasNoType();
  }

  SPIRVCapVec getRequiredCapability() const override {
    SPIRVCapVec CV;
    if (!hasType())
      return CV;
    return Type->getRequiredCapability();
  }

protected:
  void setHasNoType() { Attrib |= SPIRVEA_NOTYPE; }
  void setHasType() { Attrib &= ~SPIRVEA_NOTYPE; }

  SPIRVType *Type; // Value Type
};

class SPIRVConstant : public SPIRVValue {
public:
  // Complete constructor for integer constant
  SPIRVConstant(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                uint64_t TheValue)
      : SPIRVValue(M, 0, OpConstant, TheType, TheId) {
    Union.UInt64Val = TheValue;
    recalculateWordCount();
    validate();
  }
  // Complete constructor for float constant
  SPIRVConstant(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                float TheValue)
      : SPIRVValue(M, 0, OpConstant, TheType, TheId) {
    Union.FloatVal = TheValue;
    recalculateWordCount();
    validate();
  }
  // Complete constructor for double constant
  SPIRVConstant(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                double TheValue)
      : SPIRVValue(M, 0, OpConstant, TheType, TheId) {
    Union.DoubleVal = TheValue;
    recalculateWordCount();
    validate();
  }
  // Incomplete constructor
  SPIRVConstant() : SPIRVValue(OpConstant), NumWords(0) {}
  uint64_t getZExtIntValue() const { return Union.UInt64Val; }
  float getFloatValue() const { return Union.FloatVal; }
  double getDoubleValue() const { return Union.DoubleVal; }

protected:
  void recalculateWordCount() {
    NumWords = Type->getBitWidth() / 32;
    if (NumWords < 1)
      NumWords = 1;
    WordCount = 3 + NumWords;
  }
  void validate() const override {
    SPIRVValue::validate();
    assert(NumWords >= 1 && NumWords <= 2 && "Invalid constant size");
  }
  void encode(spv_ostream &O) const override {
    getEncoder(O) << Type << Id;
    for (unsigned I = 0; I < NumWords; ++I)
      getEncoder(O) << Union.Words[I];
  }
  void setWordCount(SPIRVWord WordCount) override {
    SPIRVValue::setWordCount(WordCount);
    NumWords = WordCount - 3;
  }
  void decode(std::istream &I) override {
    getDecoder(I) >> Type >> Id;
    for (unsigned J = 0; J < NumWords; ++J)
      getDecoder(I) >> Union.Words[J];
  }

  unsigned NumWords;
  union UnionType {
    uint64_t UInt64Val;
    float FloatVal;
    double DoubleVal;
    SPIRVWord Words[2];
    UnionType() { UInt64Val = 0; }
  } Union;
};

template <Op OC> class SPIRVConstantEmpty : public SPIRVValue {
public:
  // Complete constructor
  SPIRVConstantEmpty(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId)
      : SPIRVValue(M, 3, OC, TheType, TheId) {
    validate();
  }
  // Incomplete constructor
  SPIRVConstantEmpty() : SPIRVValue(OC) {}

protected:
  void validate() const override { SPIRVValue::validate(); }
  _SPIRV_DEF_ENCDEC2(Type, Id)
};

template <Op OC> class SPIRVConstantBool : public SPIRVConstantEmpty<OC> {
public:
  // Complete constructor
  SPIRVConstantBool(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId)
      : SPIRVConstantEmpty<OC>(M, TheType, TheId) {}
  // Incomplete constructor
  SPIRVConstantBool() {}

protected:
  void validate() const override {
    SPIRVConstantEmpty<OC>::validate();
    assert(this->Type->isTypeBool() && "Invalid type");
  }
};

typedef SPIRVConstantBool<OpConstantTrue> SPIRVConstantTrue;
typedef SPIRVConstantBool<OpConstantFalse> SPIRVConstantFalse;

class SPIRVConstantNull : public SPIRVConstantEmpty<OpConstantNull> {
public:
  // Complete constructor
  SPIRVConstantNull(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId)
      : SPIRVConstantEmpty(M, TheType, TheId) {
    validate();
  }
  // Incomplete constructor
  SPIRVConstantNull() {}

protected:
  void validate() const override {
    SPIRVConstantEmpty::validate();
    assert((Type->isTypeComposite() || Type->isTypeOpaque() ||
            Type->isTypeEvent() || Type->isTypePointer() ||
            Type->isTypeReserveId() || Type->isTypeDeviceEvent() ||
            (Type->isTypeSubgroupAvcINTEL() &&
             !Type->isTypeSubgroupAvcMceINTEL())) &&
           "Invalid type");
  }
};

class SPIRVUndef : public SPIRVConstantEmpty<OpUndef> {
public:
  // Complete constructor
  SPIRVUndef(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId)
      : SPIRVConstantEmpty(M, TheType, TheId) {
    validate();
  }
  // Incomplete constructor
  SPIRVUndef() {}

protected:
  void validate() const override { SPIRVConstantEmpty::validate(); }
};

class SPIRVConstantComposite : public SPIRVValue {
public:
  // Complete constructor for composite constant
  SPIRVConstantComposite(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                         const std::vector<SPIRVValue *> TheElements)
      : SPIRVValue(M, TheElements.size() + 3, OpConstantComposite, TheType,
                   TheId) {
    Elements = getIds(TheElements);
    validate();
  }
  // Incomplete constructor
  SPIRVConstantComposite() : SPIRVValue(OpConstantComposite) {}
  std::vector<SPIRVValue *> getElements() const { return getValues(Elements); }
  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    std::vector<SPIRVValue *> Elements = getElements();
    return std::vector<SPIRVEntry *>(Elements.begin(), Elements.end());
  }

protected:
  void validate() const override {
    SPIRVValue::validate();
    for (auto &I : Elements)
      getValue(I)->validate();
  }
  void setWordCount(SPIRVWord WordCount) override {
    SPIRVEntry::setWordCount(WordCount);
    Elements.resize(WordCount - 3);
  }
  _SPIRV_DEF_ENCDEC3(Type, Id, Elements)

  std::vector<SPIRVId> Elements;
};

class SPIRVConstantSampler : public SPIRVValue {
public:
  const static Op OC = OpConstantSampler;
  const static SPIRVWord WC = 6;
  // Complete constructor
  SPIRVConstantSampler(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                       SPIRVWord TheAddrMode, SPIRVWord TheNormalized,
                       SPIRVWord TheFilterMode)
      : SPIRVValue(M, WC, OC, TheType, TheId), AddrMode(TheAddrMode),
        Normalized(TheNormalized), FilterMode(TheFilterMode) {
    validate();
  }
  // Incomplete constructor
  SPIRVConstantSampler()
      : SPIRVValue(OC), AddrMode(SPIRVSAM_Invalid), Normalized(SPIRVWORD_MAX),
        FilterMode(SPIRVSFM_Invalid) {}

  SPIRVWord getAddrMode() const { return AddrMode; }

  SPIRVWord getFilterMode() const { return FilterMode; }

  SPIRVWord getNormalized() const { return Normalized; }
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityLiteralSampler);
  }

protected:
  SPIRVWord AddrMode;
  SPIRVWord Normalized;
  SPIRVWord FilterMode;
  void validate() const override {
    SPIRVValue::validate();
    assert(OpCode == OC);
    assert(WordCount == WC);
    assert(Type->isTypeSampler());
  }
  _SPIRV_DEF_ENCDEC5(Type, Id, AddrMode, Normalized, FilterMode)
};

class SPIRVConstantPipeStorage : public SPIRVValue {
public:
  const static Op OC = OpConstantPipeStorage;
  const static SPIRVWord WC = 6;
  // Complete constructor
  SPIRVConstantPipeStorage(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                           SPIRVWord ThePacketSize, SPIRVWord ThePacketAlign,
                           SPIRVWord TheCapacity)
      : SPIRVValue(M, WC, OC, TheType, TheId), PacketSize(ThePacketSize),
        PacketAlign(ThePacketAlign), Capacity(TheCapacity) {
    validate();
  }
  // Incomplete constructor
  SPIRVConstantPipeStorage()
      : SPIRVValue(OC), PacketSize(0), PacketAlign(0), Capacity(0) {}

  SPIRVWord getPacketSize() const { return PacketSize; }

  SPIRVWord getPacketAlign() const { return PacketAlign; }

  SPIRVWord getCapacity() const { return Capacity; }
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityPipes, CapabilityPipeStorage);
  }

protected:
  SPIRVWord PacketSize;
  SPIRVWord PacketAlign;
  SPIRVWord Capacity;
  void validate() const override {
    SPIRVValue::validate();
    assert(OpCode == OC);
    assert(WordCount == WC);
    assert(Type->isTypePipeStorage());
  }
  _SPIRV_DEF_ENCDEC5(Type, Id, PacketSize, PacketAlign, Capacity)
};

class SPIRVForward : public SPIRVValue, public SPIRVComponentExecutionModes {
public:
  const static Op OC = OpForward;
  // Complete constructor
  SPIRVForward(SPIRVModule *TheModule, SPIRVType *TheTy, SPIRVId TheId)
      : SPIRVValue(TheModule, 0, OC, TheId) {
    if (TheTy)
      setType(TheTy);
  }
  SPIRVForward() : SPIRVValue(OC) { assert(0 && "should never be called"); }
  _SPIRV_DEF_ENCDEC1(Id)
  friend class SPIRVFunction;

protected:
  void validate() const override {}
};

} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVVALUE_H
