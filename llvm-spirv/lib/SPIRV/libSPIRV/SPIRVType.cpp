//===- SPIRVtype.cpp – Class to represent a SPIR-V type ---------*- C++ -*-===//
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
/// This file implements the types defined in SPIRV spec with op codes.
///
//===----------------------------------------------------------------------===//

#include "SPIRVType.h"
#include "SPIRVDecorate.h"
#include "SPIRVModule.h"
#include "SPIRVValue.h"

#include <cassert>

namespace SPIRV {

SPIRVType *SPIRVType::getArrayElementType() const {
  assert(OpCode == OpTypeArray && "Not array type");
  return static_cast<const SPIRVTypeArray *>(this)->getElementType();
}

uint64_t SPIRVType::getArrayLength() const {
  assert(OpCode == OpTypeArray && "Not array type");
  return static_cast<const SPIRVTypeArray *>(this)
      ->getLength()
      ->getZExtIntValue();
}

SPIRVWord SPIRVType::getBitWidth() const {
  if (isTypeVector())
    return getVectorComponentType()->getBitWidth();
  if (isTypeBool())
    return 1;
  return isTypeInt() ? getIntegerBitWidth() : getFloatBitWidth();
}

SPIRVWord SPIRVType::getFloatBitWidth() const {
  assert(OpCode == OpTypeFloat && "Not a float type");
  return static_cast<const SPIRVTypeFloat *>(this)->getBitWidth();
}

SPIRVWord SPIRVType::getIntegerBitWidth() const {
  assert((OpCode == OpTypeInt || OpCode == OpTypeBool) &&
         "Not an integer type");
  if (isTypeBool())
    return 1;
  return static_cast<const SPIRVTypeInt *>(this)->getBitWidth();
}

SPIRVType *SPIRVType::getFunctionReturnType() const {
  assert(OpCode == OpTypeFunction);
  return static_cast<const SPIRVTypeFunction *>(this)->getReturnType();
}

SPIRVType *SPIRVType::getPointerElementType() const {
  assert(OpCode == OpTypePointer && "Not a pointer type");
  return static_cast<const SPIRVTypePointer *>(this)->getElementType();
}

SPIRVStorageClassKind SPIRVType::getPointerStorageClass() const {
  assert(OpCode == OpTypePointer && "Not a pointer type");
  return static_cast<const SPIRVTypePointer *>(this)->getStorageClass();
}

SPIRVType *SPIRVType::getStructMemberType(size_t Index) const {
  assert(OpCode == OpTypeStruct && "Not struct type");
  return static_cast<const SPIRVTypeStruct *>(this)->getMemberType(Index);
}

SPIRVWord SPIRVType::getStructMemberCount() const {
  assert(OpCode == OpTypeStruct && "Not struct type");
  return static_cast<const SPIRVTypeStruct *>(this)->getMemberCount();
}

SPIRVWord SPIRVType::getVectorComponentCount() const {
  assert(OpCode == OpTypeVector && "Not vector type");
  return static_cast<const SPIRVTypeVector *>(this)->getComponentCount();
}

SPIRVType *SPIRVType::getVectorComponentType() const {
  assert(OpCode == OpTypeVector && "Not vector type");
  return static_cast<const SPIRVTypeVector *>(this)->getComponentType();
}

bool SPIRVType::isTypeVoid() const { return OpCode == OpTypeVoid; }
bool SPIRVType::isTypeArray() const { return OpCode == OpTypeArray; }

bool SPIRVType::isTypeBool() const { return OpCode == OpTypeBool; }

bool SPIRVType::isTypeComposite() const {
  return isTypeVector() || isTypeArray() || isTypeStruct();
}

bool SPIRVType::isTypeFloat(unsigned Bits) const {
  return isType<SPIRVTypeFloat>(this, Bits);
}

bool SPIRVType::isTypeOCLImage() const {
  return isTypeImage() &&
         static_cast<const SPIRVTypeImage *>(this)->isOCLImage();
}

bool SPIRVType::isTypePipe() const { return OpCode == OpTypePipe; }

bool SPIRVType::isTypePipeStorage() const {
  return OpCode == OpTypePipeStorage;
}

bool SPIRVType::isTypeReserveId() const { return OpCode == OpTypeReserveId; }

bool SPIRVType::isTypeInt(unsigned Bits) const {
  return isType<SPIRVTypeInt>(this, Bits);
}

bool SPIRVType::isTypePointer() const { return OpCode == OpTypePointer; }

bool SPIRVType::isTypeOpaque() const { return OpCode == OpTypeOpaque; }

bool SPIRVType::isTypeEvent() const { return OpCode == OpTypeEvent; }

bool SPIRVType::isTypeDeviceEvent() const {
  return OpCode == OpTypeDeviceEvent;
}

bool SPIRVType::isTypeSampler() const { return OpCode == OpTypeSampler; }

bool SPIRVType::isTypeImage() const { return OpCode == OpTypeImage; }

bool SPIRVType::isTypeStruct() const { return OpCode == OpTypeStruct; }

bool SPIRVType::isTypeVector() const { return OpCode == OpTypeVector; }

bool SPIRVType::isTypeVectorBool() const {
  return isTypeVector() && getVectorComponentType()->isTypeBool();
}

bool SPIRVType::isTypeVectorInt() const {
  return isTypeVector() && getVectorComponentType()->isTypeInt();
}

bool SPIRVType::isTypeVectorFloat() const {
  return isTypeVector() && getVectorComponentType()->isTypeFloat();
}

bool SPIRVType::isTypeVectorOrScalarBool() const {
  return isTypeBool() || isTypeVectorBool();
}

bool SPIRVType::isTypeSubgroupAvcINTEL() const {
  return isSubgroupAvcINTELTypeOpCode(OpCode);
}

bool SPIRVType::isTypeSubgroupAvcMceINTEL() const {
  return OpCode == OpTypeAvcMcePayloadINTEL ||
         OpCode == OpTypeAvcMceResultINTEL;
}

bool SPIRVType::isTypeVectorOrScalarInt() const {
  return isTypeInt() || isTypeVectorInt();
}

bool SPIRVType::isTypeVectorOrScalarFloat() const {
  return isTypeFloat() || isTypeVectorFloat();
}

bool SPIRVTypeStruct::isPacked() const {
  return hasDecorate(DecorationCPacked);
}

void SPIRVTypeStruct::setPacked(bool Packed) {
  if (Packed)
    addDecorate(new SPIRVDecorate(DecorationCPacked, this));
  else
    eraseDecorate(DecorationCPacked);
}

SPIRVTypeArray::SPIRVTypeArray(SPIRVModule *M, SPIRVId TheId,
                               SPIRVType *TheElemType, SPIRVConstant *TheLength)
    : SPIRVType(M, 4, OpTypeArray, TheId), ElemType(TheElemType),
      Length(TheLength->getId()) {
  validate();
}

void SPIRVTypeArray::validate() const {
  SPIRVEntry::validate();
  ElemType->validate();
  assert(getValue(Length)->getType()->isTypeInt() &&
         get<SPIRVConstant>(Length)->getZExtIntValue() > 0);
}

SPIRVConstant *SPIRVTypeArray::getLength() const {
  return get<SPIRVConstant>(Length);
}

_SPIRV_IMP_ENCDEC3(SPIRVTypeArray, Id, ElemType, Length)

void SPIRVTypeForwardPointer::encode(spv_ostream &O) const {
  getEncoder(O) << Pointer << SC;
}

void SPIRVTypeForwardPointer::decode(std::istream &I) {
  auto Decoder = getDecoder(I);
  SPIRVId PointerId;
  Decoder >> PointerId >> SC;
}
} // namespace SPIRV
