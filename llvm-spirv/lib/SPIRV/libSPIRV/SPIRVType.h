//===- SPIRVType.h - Class to represent a SPIR-V Type -----------*- C++ -*-===//
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
/// This file defines the types defined in SPIRV spec with op codes.
///
/// The name of the SPIR-V types follow the op code name in the spec, e.g.
/// SPIR-V type with op code name OpTypeInt is named as SPIRVTypeInt. This is
/// for readability and ease of using macro to handle types.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVTYPE_H
#define SPIRV_LIBSPIRV_SPIRVTYPE_H

#include "SPIRVEntry.h"
#include "SPIRVStream.h"

#include <cassert>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

namespace SPIRV {

class SPIRVType : public SPIRVEntry {
public:
  // Complete constructor
  SPIRVType(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode, SPIRVId TheId)
      : SPIRVEntry(M, TheWordCount, TheOpCode, TheId) {}
  // Incomplete constructor
  SPIRVType(Op TheOpCode) : SPIRVEntry(TheOpCode) {}

  SPIRVType *getArrayElementType() const;
  uint64_t getArrayLength() const;
  unsigned getBitWidth() const;
  unsigned getFloatBitWidth() const;
  SPIRVType *getFunctionReturnType() const;
  unsigned getIntegerBitWidth() const;
  SPIRVType *getPointerElementType() const;
  SPIRVStorageClassKind getPointerStorageClass() const;
  SPIRVType *getStructMemberType(size_t) const;
  SPIRVWord getStructMemberCount() const;
  SPIRVWord getVectorComponentCount() const;
  SPIRVType *getVectorComponentType() const;

  bool isTypeVoid() const;
  bool isTypeArray() const;
  bool isTypeBool() const;
  bool isTypeComposite() const;
  bool isTypeEvent() const;
  bool isTypeDeviceEvent() const;
  bool isTypeReserveId() const;
  bool isTypeFloat(unsigned Bits = 0) const;
  bool isTypeImage() const;
  bool isTypeOCLImage() const;
  bool isTypePipe() const;
  bool isTypePipeStorage() const;
  bool isTypeInt(unsigned Bits = 0) const;
  bool isTypeOpaque() const;
  bool isTypePointer() const;
  bool isTypeSampler() const;
  bool isTypeStruct() const;
  bool isTypeVector() const;
  bool isTypeVectorInt() const;
  bool isTypeVectorFloat() const;
  bool isTypeVectorBool() const;
  bool isTypeVectorOrScalarInt() const;
  bool isTypeVectorOrScalarFloat() const;
  bool isTypeVectorOrScalarBool() const;
  bool isTypeSubgroupAvcINTEL() const;
  bool isTypeSubgroupAvcMceINTEL() const;
};

class SPIRVTypeVoid : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeVoid(SPIRVModule *M, SPIRVId TheId)
      : SPIRVType(M, 2, OpTypeVoid, TheId) {}
  // Incomplete constructor
  SPIRVTypeVoid() : SPIRVType(OpTypeVoid) {}

protected:
  _SPIRV_DEF_ENCDEC1(Id)
};

class SPIRVTypeBool : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeBool(SPIRVModule *M, SPIRVId TheId)
      : SPIRVType(M, 2, OpTypeBool, TheId) {}
  // Incomplete constructor
  SPIRVTypeBool() : SPIRVType(OpTypeBool) {}

protected:
  _SPIRV_DEF_ENCDEC1(Id)
};

class SPIRVTypeInt : public SPIRVType {
public:
  static const Op OC = OpTypeInt;
  // Complete constructor
  SPIRVTypeInt(SPIRVModule *M, SPIRVId TheId, unsigned TheBitWidth,
               bool ItIsSigned)
      : SPIRVType(M, 4, OC, TheId), BitWidth(TheBitWidth),
        IsSigned(ItIsSigned) {
    validate();
  }
  // Incomplete constructor
  SPIRVTypeInt() : SPIRVType(OC), BitWidth(0), IsSigned(false) {}

  unsigned getBitWidth() const { return BitWidth; }
  bool isSigned() const { return IsSigned; }
  SPIRVCapVec getRequiredCapability() const override {
    SPIRVCapVec CV;
    switch (BitWidth) {
    case 8:
      CV.push_back(CapabilityInt8);
      break;
    case 16:
      CV.push_back(CapabilityInt16);
      break;
    case 64:
      CV.push_back(CapabilityInt64);
      break;
    default:
      break;
    }
    return CV;
  }

protected:
  _SPIRV_DEF_ENCDEC3(Id, BitWidth, IsSigned)
  void validate() const override {
    SPIRVEntry::validate();
    assert(BitWidth > 1 && BitWidth <= 64 && "Invalid bit width");
  }

private:
  unsigned BitWidth; // Bit width
  bool IsSigned;     // Whether it is signed
};

class SPIRVTypeFloat : public SPIRVType {
public:
  static const Op OC = OpTypeFloat;
  // Complete constructor
  SPIRVTypeFloat(SPIRVModule *M, SPIRVId TheId, unsigned TheBitWidth)
      : SPIRVType(M, 3, OC, TheId), BitWidth(TheBitWidth) {}
  // Incomplete constructor
  SPIRVTypeFloat() : SPIRVType(OC), BitWidth(0) {}

  unsigned getBitWidth() const { return BitWidth; }

  SPIRVCapVec getRequiredCapability() const override {
    SPIRVCapVec CV;
    if (isTypeFloat(16)) {
      CV.push_back(CapabilityFloat16Buffer);
      auto Extensions = getModule()->getExtension();
      if (std::any_of(Extensions.begin(), Extensions.end(),
                      [](const std::string &I) { return I == "cl_khr_fp16"; }))
        CV.push_back(CapabilityFloat16);
    } else if (isTypeFloat(64))
      CV.push_back(CapabilityFloat64);
    return CV;
  }

protected:
  _SPIRV_DEF_ENCDEC2(Id, BitWidth)
  void validate() const override {
    SPIRVEntry::validate();
    assert(BitWidth >= 16 && BitWidth <= 64 && "Invalid bit width");
  }

private:
  unsigned BitWidth; // Bit width
};

class SPIRVTypePointer : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypePointer(SPIRVModule *M, SPIRVId TheId,
                   SPIRVStorageClassKind TheStorageClass,
                   SPIRVType *ElementType)
      : SPIRVType(M, 4, OpTypePointer, TheId),
        ElemStorageClass(TheStorageClass), ElemTypeId(ElementType->getId()) {
    validate();
  }
  // Incomplete constructor
  SPIRVTypePointer()
      : SPIRVType(OpTypePointer), ElemStorageClass(StorageClassFunction),
        ElemTypeId(0) {}

  SPIRVType *getElementType() const {
    return static_cast<SPIRVType *>(getEntry(ElemTypeId));
  }
  SPIRVStorageClassKind getStorageClass() const { return ElemStorageClass; }
  SPIRVCapVec getRequiredCapability() const override {
    auto Cap = getVec(CapabilityAddresses);
    if (getElementType()->isTypeFloat(16))
      Cap.push_back(CapabilityFloat16Buffer);
    auto C = getCapability(ElemStorageClass);
    Cap.insert(Cap.end(), C.begin(), C.end());
    return Cap;
  }
  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    return std::vector<SPIRVEntry *>(1, getEntry(ElemTypeId));
  }

protected:
  _SPIRV_DEF_ENCDEC3(Id, ElemStorageClass, ElemTypeId)
  void validate() const override {
    SPIRVEntry::validate();
    assert(isValid(ElemStorageClass));
  }

private:
  SPIRVStorageClassKind ElemStorageClass; // Storage Class
  SPIRVId ElemTypeId;
};

class SPIRVTypeForwardPointer : public SPIRVEntryNoId<OpTypeForwardPointer> {
public:
  SPIRVTypeForwardPointer(SPIRVModule *M, SPIRVTypePointer *Pointer,
                          SPIRVStorageClassKind SC)
      : SPIRVEntryNoId(M, 3), Pointer(Pointer), SC(SC) {}

  SPIRVTypeForwardPointer()
      : Pointer(nullptr), SC(StorageClassUniformConstant) {}

  SPIRVTypePointer *getPointer() const { return Pointer; }
  _SPIRV_DCL_ENCDEC
private:
  SPIRVTypePointer *Pointer;
  SPIRVStorageClassKind SC;
};

class SPIRVTypeVector : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeVector(SPIRVModule *M, SPIRVId TheId, SPIRVType *TheCompType,
                  SPIRVWord TheCompCount)
      : SPIRVType(M, 4, OpTypeVector, TheId), CompType(TheCompType),
        CompCount(TheCompCount) {
    validate();
  }
  // Incomplete constructor
  SPIRVTypeVector()
      : SPIRVType(OpTypeVector), CompType(nullptr), CompCount(0) {}

  SPIRVType *getComponentType() const { return CompType; }
  SPIRVWord getComponentCount() const { return CompCount; }
  bool isValidIndex(SPIRVWord Index) const { return Index < CompCount; }
  SPIRVCapVec getRequiredCapability() const override {
    SPIRVCapVec V(getComponentType()->getRequiredCapability());
    // Even though the capability name is "Vector16", it describes
    // usage of 8-component or 16-component vectors.
    if (CompCount >= 8)
      V.push_back(CapabilityVector16);
    return V;
  }

  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    return std::vector<SPIRVEntry *>(1, CompType);
  }

protected:
  _SPIRV_DEF_ENCDEC3(Id, CompType, CompCount)
  void validate() const override {
    SPIRVEntry::validate();
    CompType->validate();
    assert(CompCount == 2 || CompCount == 3 || CompCount == 4 ||
           CompCount == 8 || CompCount == 16);
  }

private:
  SPIRVType *CompType; // Component Type
  SPIRVWord CompCount; // Component Count
};

class SPIRVConstant;
class SPIRVTypeArray : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeArray(SPIRVModule *M, SPIRVId TheId, SPIRVType *TheElemType,
                 SPIRVConstant *TheLength);
  // Incomplete constructor
  SPIRVTypeArray()
      : SPIRVType(OpTypeArray), ElemType(nullptr), Length(SPIRVID_INVALID) {}

  SPIRVType *getElementType() const { return ElemType; }
  SPIRVConstant *getLength() const;
  SPIRVCapVec getRequiredCapability() const override {
    return getElementType()->getRequiredCapability();
  }
  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    std::vector<SPIRVEntry *> Operands(2, ElemType);
    Operands[1] = (SPIRVEntry *)getLength();
    return Operands;
  }

protected:
  _SPIRV_DCL_ENCDEC
  void validate() const override;

private:
  SPIRVType *ElemType; // Element Type
  SPIRVId Length;      // Array Length
};

class SPIRVTypeOpaque : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeOpaque(SPIRVModule *M, SPIRVId TheId, const std::string &TheName)
      : SPIRVType(M, 2 + getSizeInWords(TheName), OpTypeOpaque, TheId) {
    Name = TheName;
    validate();
  }
  // Incomplete constructor
  SPIRVTypeOpaque() : SPIRVType(OpTypeOpaque) {}

protected:
  _SPIRV_DEF_ENCDEC2(Id, Name)
  void validate() const override { SPIRVEntry::validate(); }
};

struct SPIRVTypeImageDescriptor {
  SPIRVImageDimKind Dim;
  SPIRVWord Depth;
  SPIRVWord Arrayed;
  SPIRVWord MS;
  SPIRVWord Sampled;
  SPIRVWord Format;
  static std::tuple<
      std::tuple<SPIRVImageDimKind, SPIRVWord, SPIRVWord, SPIRVWord, SPIRVWord>,
      SPIRVWord>
  getAsTuple(const SPIRVTypeImageDescriptor &Desc) {
    return std::make_tuple(std::make_tuple(Desc.Dim, Desc.Depth, Desc.Arrayed,
                                           Desc.MS, Desc.Sampled),
                           Desc.Format);
  }
  SPIRVTypeImageDescriptor()
      : Dim(Dim1D), Depth(0), Arrayed(0), MS(0), Sampled(0), Format(0) {}
  SPIRVTypeImageDescriptor(SPIRVImageDimKind Dim, SPIRVWord Cont, SPIRVWord Arr,
                           SPIRVWord Comp, SPIRVWord Mult, SPIRVWord F)
      : Dim(Dim), Depth(Cont), Arrayed(Arr), MS(Comp), Sampled(Mult),
        Format(F) {}
};

template <>
inline void SPIRVMap<std::string, SPIRVTypeImageDescriptor>::init() {
#define _SPIRV_OP(x, ...)                                                      \
  {                                                                            \
    SPIRVTypeImageDescriptor S(__VA_ARGS__);                                   \
    add(#x, S);                                                                \
  }
  _SPIRV_OP(image1d_t, Dim1D, 0, 0, 0, 0, 0)
  _SPIRV_OP(image1d_buffer_t, DimBuffer, 0, 0, 0, 0, 0)
  _SPIRV_OP(image1d_array_t, Dim1D, 0, 1, 0, 0, 0)
  _SPIRV_OP(image2d_t, Dim2D, 0, 0, 0, 0, 0)
  _SPIRV_OP(image2d_array_t, Dim2D, 0, 1, 0, 0, 0)
  _SPIRV_OP(image2d_depth_t, Dim2D, 1, 0, 0, 0, 0)
  _SPIRV_OP(image2d_array_depth_t, Dim2D, 1, 1, 0, 0, 0)
  _SPIRV_OP(image2d_msaa_t, Dim2D, 0, 0, 1, 0, 0)
  _SPIRV_OP(image2d_array_msaa_t, Dim2D, 0, 1, 1, 0, 0)
  _SPIRV_OP(image2d_msaa_depth_t, Dim2D, 1, 0, 1, 0, 0)
  _SPIRV_OP(image2d_array_msaa_depth_t, Dim2D, 1, 1, 1, 0, 0)
  _SPIRV_OP(image3d_t, Dim3D, 0, 0, 0, 0, 0)
#undef _SPIRV_OP
}
typedef SPIRVMap<std::string, SPIRVTypeImageDescriptor> OCLSPIRVImageTypeMap;

// Comparision function required to use the struct as map key.
inline bool operator<(const SPIRVTypeImageDescriptor &A,
                      const SPIRVTypeImageDescriptor &B) {
  return SPIRVTypeImageDescriptor::getAsTuple(A) <
         SPIRVTypeImageDescriptor::getAsTuple(B);
}

class SPIRVTypeImage : public SPIRVType {
public:
  const static Op OC = OpTypeImage;
  const static SPIRVWord FixedWC = 9;
  SPIRVTypeImage(SPIRVModule *M, SPIRVId TheId, SPIRVId TheSampledType,
                 const SPIRVTypeImageDescriptor &TheDesc)
      : SPIRVType(M, FixedWC, OC, TheId), SampledType(TheSampledType),
        Desc(TheDesc) {
    validate();
  }
  SPIRVTypeImage(SPIRVModule *M, SPIRVId TheId, SPIRVId TheSampledType,
                 const SPIRVTypeImageDescriptor &TheDesc,
                 SPIRVAccessQualifierKind TheAcc)
      : SPIRVType(M, FixedWC + 1, OC, TheId), SampledType(TheSampledType),
        Desc(TheDesc) {
    Acc.push_back(TheAcc);
    validate();
  }
  SPIRVTypeImage() : SPIRVType(OC), SampledType(SPIRVID_INVALID), Desc() {}
  const SPIRVTypeImageDescriptor &getDescriptor() const { return Desc; }
  bool isOCLImage() const { return Desc.Sampled == 0 && Desc.Format == 0; }
  bool hasAccessQualifier() const { return !Acc.empty(); }
  SPIRVAccessQualifierKind getAccessQualifier() const {
    assert(hasAccessQualifier());
    return Acc[0];
  }
  SPIRVCapVec getRequiredCapability() const override {
    SPIRVCapVec CV;
    CV.push_back(CapabilityImageBasic);
    if (Desc.Dim == SPIRVImageDimKind::Dim1D)
      CV.push_back(CapabilitySampled1D);
    else if (Desc.Dim == SPIRVImageDimKind::DimBuffer)
      CV.push_back(CapabilitySampledBuffer);
    if (Acc.size() > 0 && Acc[0] == AccessQualifierReadWrite)
      CV.push_back(CapabilityImageReadWrite);
    if (Desc.MS)
      CV.push_back(CapabilityImageMipmap);
    return CV;
  }
  SPIRVType *getSampledType() const { return get<SPIRVType>(SampledType); }

  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    return std::vector<SPIRVEntry *>(1, get<SPIRVType>(SampledType));
  }

protected:
  _SPIRV_DEF_ENCDEC9(Id, SampledType, Desc.Dim, Desc.Depth, Desc.Arrayed,
                     Desc.MS, Desc.Sampled, Desc.Format, Acc)
  // The validation assumes OpenCL image or sampler type.
  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == FixedWC + Acc.size());
    assert(SampledType != SPIRVID_INVALID && "Invalid sampled type");
    assert(Desc.Dim <= 5);
    assert(Desc.Depth <= 1);
    assert(Desc.Arrayed <= 1);
    assert(Desc.MS <= 1);
    assert(Desc.Sampled == 0); // For OCL only
    assert(Desc.Format == 0);  // For OCL only
    assert(Acc.size() <= 1);
  }
  void setWordCount(SPIRVWord TheWC) override {
    WordCount = TheWC;
    Acc.resize(WordCount - FixedWC);
  }

private:
  SPIRVId SampledType;
  SPIRVTypeImageDescriptor Desc;
  std::vector<SPIRVAccessQualifierKind> Acc;
};

class SPIRVTypeSampler : public SPIRVType {
public:
  const static Op OC = OpTypeSampler;
  const static SPIRVWord FixedWC = 2;
  SPIRVTypeSampler(SPIRVModule *M, SPIRVId TheId)
      : SPIRVType(M, FixedWC, OC, TheId) {
    validate();
  }
  SPIRVTypeSampler() : SPIRVType(OC) {}

protected:
  _SPIRV_DEF_ENCDEC1(Id)
  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == FixedWC);
  }
};

class SPIRVTypeSampledImage : public SPIRVType {
public:
  const static Op OC = OpTypeSampledImage;
  const static SPIRVWord FixedWC = 3;
  SPIRVTypeSampledImage(SPIRVModule *M, SPIRVId TheId, SPIRVTypeImage *TheImgTy)
      : SPIRVType(M, FixedWC, OC, TheId), ImgTy(TheImgTy) {
    validate();
  }
  SPIRVTypeSampledImage() : SPIRVType(OC), ImgTy(nullptr) {}

  const SPIRVTypeImage *getImageType() const { return ImgTy; }

  void setImageType(SPIRVTypeImage *TheImgTy) { ImgTy = TheImgTy; }

  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    return std::vector<SPIRVEntry *>(1, ImgTy);
  }

protected:
  SPIRVTypeImage *ImgTy;
  _SPIRV_DEF_ENCDEC2(Id, ImgTy)
  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == FixedWC);
    assert(ImgTy && ImgTy->isTypeImage());
  }
};

class SPIRVTypePipeStorage : public SPIRVType {
public:
  const static Op OC = OpTypePipeStorage;
  const static SPIRVWord FixedWC = 2;
  SPIRVTypePipeStorage(SPIRVModule *M, SPIRVId TheId)
      : SPIRVType(M, FixedWC, OC, TheId) {
    validate();
  }
  SPIRVTypePipeStorage() : SPIRVType(OC) {}

protected:
  _SPIRV_DEF_ENCDEC1(Id)
  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == FixedWC);
  }
};

class SPIRVTypeStruct : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeStruct(SPIRVModule *M, SPIRVId TheId,
                  const std::vector<SPIRVType *> &TheMemberTypes,
                  const std::string &TheName)
      : SPIRVType(M, 2 + TheMemberTypes.size(), OpTypeStruct, TheId) {
    MemberTypeIdVec.resize(TheMemberTypes.size());
    for (auto &T : TheMemberTypes)
      MemberTypeIdVec.push_back(T->getId());
    Name = TheName;
    validate();
  }
  SPIRVTypeStruct(SPIRVModule *M, SPIRVId TheId, unsigned NumMembers,
                  const std::string &TheName)
      : SPIRVType(M, 2 + NumMembers, OpTypeStruct, TheId) {
    Name = TheName;
    validate();
    MemberTypeIdVec.resize(NumMembers);
  }
  // Incomplete constructor
  SPIRVTypeStruct() : SPIRVType(OpTypeStruct) {}

  SPIRVWord getMemberCount() const { return MemberTypeIdVec.size(); }
  SPIRVType *getMemberType(size_t I) const {
    return static_cast<SPIRVType *>(getEntry(MemberTypeIdVec[I]));
  }
  void setMemberType(size_t I, SPIRVType *Ty) {
    MemberTypeIdVec[I] = Ty->getId();
  }

  bool isPacked() const;
  void setPacked(bool Packed);

  void setWordCount(SPIRVWord WordCount) override {
    SPIRVType::setWordCount(WordCount);
    MemberTypeIdVec.resize(WordCount - 2);
  }

  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    std::vector<SPIRVEntry *> Operands(MemberTypeIdVec.size());
    for (size_t I = 0, E = MemberTypeIdVec.size(); I < E; ++I)
      Operands[I] = getEntry(MemberTypeIdVec[I]);
    return Operands;
  }

protected:
  _SPIRV_DEF_ENCDEC2(Id, MemberTypeIdVec)

  void validate() const override { SPIRVEntry::validate(); }

private:
  std::vector<SPIRVId> MemberTypeIdVec; // Member Type Ids
};

class SPIRVTypeFunction : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeFunction(SPIRVModule *M, SPIRVId TheId, SPIRVType *TheReturnType,
                    const std::vector<SPIRVType *> &TheParameterTypes)
      : SPIRVType(M, 3 + TheParameterTypes.size(), OpTypeFunction, TheId),
        ReturnType(TheReturnType), ParamTypeVec(TheParameterTypes) {
    validate();
  }
  // Incomplete constructor
  SPIRVTypeFunction() : SPIRVType(OpTypeFunction), ReturnType(NULL) {}

  SPIRVType *getReturnType() const { return ReturnType; }
  SPIRVWord getNumParameters() const { return ParamTypeVec.size(); }
  SPIRVType *getParameterType(unsigned I) const { return ParamTypeVec[I]; }
  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    std::vector<SPIRVEntry *> Operands(1 + ParamTypeVec.size(), ReturnType);
    std::copy(ParamTypeVec.begin(), ParamTypeVec.end(), ++Operands.begin());
    return Operands;
  }

protected:
  _SPIRV_DEF_ENCDEC3(Id, ReturnType, ParamTypeVec)
  void setWordCount(SPIRVWord WordCount) override {
    SPIRVType::setWordCount(WordCount);
    ParamTypeVec.resize(WordCount - 3);
  }
  void validate() const override {
    SPIRVEntry::validate();
    ReturnType->validate();
    for (auto T : ParamTypeVec)
      T->validate();
  }

private:
  SPIRVType *ReturnType;                 // Return Type
  std::vector<SPIRVType *> ParamTypeVec; // Parameter Types
};

class SPIRVTypeOpaqueGeneric : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeOpaqueGeneric(Op TheOpCode, SPIRVModule *M, SPIRVId TheId)
      : SPIRVType(M, 2, TheOpCode, TheId) {
    validate();
  }

  // Incomplete constructor
  SPIRVTypeOpaqueGeneric(Op TheOpCode)
      : SPIRVType(TheOpCode), Opn(SPIRVID_INVALID) {}

  SPIRVValue *getOperand() { return getValue(Opn); }

protected:
  _SPIRV_DEF_ENCDEC1(Id)
  void validate() const override { SPIRVEntry::validate(); }
  SPIRVId Opn;
};

template <Op TheOpCode>
class SPIRVOpaqueGenericType : public SPIRVTypeOpaqueGeneric {
public:
  // Complete constructor
  SPIRVOpaqueGenericType(SPIRVModule *M, SPIRVId TheId)
      : SPIRVTypeOpaqueGeneric(TheOpCode, M, TheId) {}
  // Incomplete constructor
  SPIRVOpaqueGenericType() : SPIRVTypeOpaqueGeneric(TheOpCode) {}
};

#define _SPIRV_OP(x) typedef SPIRVOpaqueGenericType<OpType##x> SPIRVType##x;
_SPIRV_OP(Event)
_SPIRV_OP(ReserveId)
#undef _SPIRV_OP

class SPIRVTypeDeviceEvent : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeDeviceEvent(SPIRVModule *M, SPIRVId TheId)
      : SPIRVType(M, 2, OpTypeDeviceEvent, TheId) {
    validate();
  }

  // Incomplete constructor
  SPIRVTypeDeviceEvent() : SPIRVType(OpTypeDeviceEvent) {}

  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityDeviceEnqueue);
  }

protected:
  _SPIRV_DEF_ENCDEC1(Id)
  void validate() const override { SPIRVEntry::validate(); }
};

class SPIRVTypeQueue : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeQueue(SPIRVModule *M, SPIRVId TheId)
      : SPIRVType(M, 2, OpTypeQueue, TheId) {
    validate();
  }

  // Incomplete constructor
  SPIRVTypeQueue() : SPIRVType(OpTypeQueue) {}

  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityDeviceEnqueue);
  }

protected:
  _SPIRV_DEF_ENCDEC1(Id)
};

class SPIRVTypePipe : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypePipe(SPIRVModule *M, SPIRVId TheId,
                SPIRVAccessQualifierKind AccessQual = AccessQualifierReadOnly)
      : SPIRVType(M, 3, OpTypePipe, TheId), AccessQualifier(AccessQual) {
    validate();
  }

  // Incomplete constructor
  SPIRVTypePipe()
      : SPIRVType(OpTypePipe), AccessQualifier(AccessQualifierReadOnly) {}

  SPIRVAccessQualifierKind getAccessQualifier() const {
    return AccessQualifier;
  }
  void setPipeAcessQualifier(SPIRVAccessQualifierKind AccessQual) {
    AccessQualifier = AccessQual;
    assert(isValid(AccessQualifier));
  }
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityPipes);
  }

protected:
  _SPIRV_DEF_ENCDEC2(Id, AccessQualifier)
  void validate() const override { SPIRVEntry::validate(); }

private:
  SPIRVAccessQualifierKind AccessQualifier; // Access Qualifier
};

template <typename T2, typename T1>
bool isType(const T1 *Ty, unsigned Bits = 0) {
  bool Is = Ty->getOpCode() == T2::OC;
  if (!Is)
    return false;
  if (Bits == 0)
    return true;
  return static_cast<const T2 *>(Ty)->getBitWidth() == Bits;
}

// SPV_INTEL_device_side_avc_motion_estimation extension types
class SPIRVTypeVmeImageINTEL : public SPIRVType {
public:
  const static Op OC = OpTypeVmeImageINTEL;
  const static SPIRVWord FixedWC = 3;
  SPIRVTypeVmeImageINTEL(SPIRVModule *M, SPIRVId TheId,
                         SPIRVTypeImage *TheImgTy)
      : SPIRVType(M, FixedWC, OC, TheId), ImgTy(TheImgTy) {
    validate();
  }

  SPIRVTypeVmeImageINTEL() : SPIRVType(OC), ImgTy(nullptr) {}

  const SPIRVTypeImage *getImageType() const { return ImgTy; }
  void setImageType(SPIRVTypeImage *TheImgTy) { ImgTy = TheImgTy; }

  virtual std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    return std::vector<SPIRVEntry *>(1, ImgTy);
  }

  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupAvcMotionEstimationINTEL);
  }

protected:
  SPIRVTypeImage *ImgTy;
  _SPIRV_DEF_ENCDEC2(Id, ImgTy)

  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == FixedWC);
    assert(ImgTy && ImgTy->isTypeImage());
  }
};

class SPIRVTypeSubgroupINTEL;
template <>
inline void SPIRVMap<std::string, Op, SPIRVTypeSubgroupINTEL>::init() {
#define _SPIRV_OP(x, y)                                                        \
  add("opencl.intel_sub_group_avc_" #x, OpTypeAvc##y##INTEL);
  _SPIRV_OP(mce_payload_t, McePayload)
  _SPIRV_OP(mce_result_t, MceResult)
  _SPIRV_OP(sic_payload_t, SicPayload)
  _SPIRV_OP(sic_result_t, SicResult)
  _SPIRV_OP(ime_result_single_reference_streamout_t,
            ImeResultSingleReferenceStreamout)
  _SPIRV_OP(ime_result_dual_reference_streamout_t,
            ImeResultDualReferenceStreamout)
  _SPIRV_OP(ime_single_reference_streamin_t, ImeSingleReferenceStreamin)
  _SPIRV_OP(ime_dual_reference_streamin_t, ImeDualReferenceStreamin)
  _SPIRV_OP(ime_payload_t, ImePayload)
  _SPIRV_OP(ime_result_t, ImeResult)
  _SPIRV_OP(ref_payload_t, RefPayload)
  _SPIRV_OP(ref_result_t, RefResult);
#undef _SPIRV_OP
}
typedef SPIRVMap<std::string, Op, SPIRVTypeSubgroupINTEL>
    OCLSubgroupINTELTypeOpCodeMap;

class SPIRVTypeSubgroupAvcINTEL : public SPIRVType {
public:
  // Complete constructor
  SPIRVTypeSubgroupAvcINTEL(Op TheOpCode, SPIRVModule *M, SPIRVId TheId)
      : SPIRVType(M, 2, TheOpCode, TheId) {
    validate();
  }

  // Incomplete constructor
  SPIRVTypeSubgroupAvcINTEL(Op TheOpCode)
      : SPIRVType(TheOpCode), Opn(SPIRVID_INVALID) {}

  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilitySubgroupAvcMotionEstimationINTEL);
  }

  SPIRVValue *getOperand() { return getValue(Opn); }

protected:
  _SPIRV_DEF_ENCDEC1(Id)
  void validate() const override { SPIRVEntry::validate(); }
  SPIRVId Opn;
};

template <Op TheOpCode>
class SPIRVSubgroupAvcINTELType : public SPIRVTypeSubgroupAvcINTEL {
public:
  // Complete constructor
  SPIRVSubgroupAvcINTELType(SPIRVModule *M, SPIRVId TheId)
      : SPIRVTypeSubgroupAvcINTEL(TheOpCode, M, TheId) {}

  // Incomplete constructor
  SPIRVSubgroupAvcINTELType() : SPIRVTypeSubgroupAvcINTEL(TheOpCode) {}
};

#define _SPIRV_OP(x)                                                           \
  typedef SPIRVSubgroupAvcINTELType<OpType##x##INTEL> SPIRVType##x##INTEL;
_SPIRV_OP(AvcMcePayload)
_SPIRV_OP(AvcImePayload)
_SPIRV_OP(AvcRefPayload)
_SPIRV_OP(AvcSicPayload)
_SPIRV_OP(AvcMceResult)
_SPIRV_OP(AvcImeResult)
_SPIRV_OP(AvcImeResultSingleReferenceStreamout)
_SPIRV_OP(AvcImeResultDualReferenceStreamout)
_SPIRV_OP(AvcImeSingleReferenceStreamin)
_SPIRV_OP(AvcImeDualReferenceStreamin)
_SPIRV_OP(AvcRefResult)
_SPIRV_OP(AvcSicResult)
#undef _SPIRV_OP
} // namespace SPIRV
#endif // SPIRV_LIBSPIRV_SPIRVTYPE_H
