//===- SPIRVEnum.h - SPIR-V enums -------------------------------*- C++ -*-===//
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
/// This file defines SPIR-V enums.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVENUM_H
#define SPIRV_LIBSPIRV_SPIRVENUM_H

#include "SPIRVOpCode.h"
#include "spirv.hpp"
#include <cstdint>
using namespace spv;

namespace SPIRV {

typedef uint32_t SPIRVWord;
typedef uint32_t SPIRVId;
#define SPIRVID_MAX ~0U
#define SPIRVID_INVALID ~0U
#define SPIRVWORD_MAX ~0U

inline bool isValidId(SPIRVId Id) { return Id != SPIRVID_INVALID && Id != 0; }

inline SPIRVWord mkWord(unsigned WordCount, Op OpCode) {
  return (WordCount << 16) | OpCode;
}

const static unsigned KSpirvMemOrderSemanticMask = 0x1F;

enum SPIRVVersion : SPIRVWord {
  SPIRV_1_0 = 0x00010000,
  SPIRV_1_1 = 0x00010100
};

enum SPIRVGeneratorKind {
  SPIRVGEN_KhronosLLVMSPIRVTranslator = 6,
  SPIRVGEN_KhronosSPIRVAssembler = 7,
};

enum SPIRVInstructionSchemaKind {
  SPIRVISCH_Default,
};

enum SPIRVExtInstSetKind {
  SPIRVEIS_OpenCL,
  SPIRVEIS_Debug,
  SPIRVEIS_Count,
};

enum SPIRVSamplerAddressingModeKind {
  SPIRVSAM_None = 0,
  SPIRVSAM_ClampEdge = 2,
  SPIRVSAM_Clamp = 4,
  SPIRVSAM_Repeat = 6,
  SPIRVSAM_RepeatMirrored = 8,
  SPIRVSAM_Invalid = 255,
};

enum SPIRVSamplerFilterModeKind {
  SPIRVSFM_Nearest = 16,
  SPIRVSFM_Linear = 32,
  SPIRVSFM_Invalid = 255,
};

typedef spv::Capability SPIRVCapabilityKind;
typedef spv::ExecutionModel SPIRVExecutionModelKind;
typedef spv::ExecutionMode SPIRVExecutionModeKind;
typedef spv::AccessQualifier SPIRVAccessQualifierKind;
typedef spv::AddressingModel SPIRVAddressingModelKind;
typedef spv::LinkageType SPIRVLinkageTypeKind;
typedef spv::MemoryModel SPIRVMemoryModelKind;
typedef spv::StorageClass SPIRVStorageClassKind;
typedef spv::FunctionControlMask SPIRVFunctionControlMaskKind;
typedef spv::FPRoundingMode SPIRVFPRoundingModeKind;
typedef spv::FunctionParameterAttribute SPIRVFuncParamAttrKind;
typedef spv::BuiltIn SPIRVBuiltinVariableKind;
typedef spv::MemoryAccessMask SPIRVMemoryAccessKind;
typedef spv::GroupOperation SPIRVGroupOperationKind;
typedef spv::Dim SPIRVImageDimKind;
typedef std::vector<SPIRVCapabilityKind> SPIRVCapVec;

enum SPIRVExtensionKind {
  SPV_INTEL_device_side_avc_motion_estimation,
  SPV_KHR_no_integer_wrap_decoration
};

typedef std::set<SPIRVExtensionKind> SPIRVExtSet;

template <> inline void SPIRVMap<SPIRVExtensionKind, std::string>::init() {
  add(SPV_INTEL_device_side_avc_motion_estimation,
      "SPV_INTEL_device_side_avc_motion_estimation");
  add(SPV_KHR_no_integer_wrap_decoration, "SPV_KHR_no_integer_wrap_decoration");
};

template <> inline void SPIRVMap<SPIRVExtInstSetKind, std::string>::init() {
  add(SPIRVEIS_OpenCL, "OpenCL.std");
  add(SPIRVEIS_Debug, "SPIRV.debug");
}
typedef SPIRVMap<SPIRVExtInstSetKind, std::string> SPIRVBuiltinSetNameMap;

template <typename K> SPIRVCapVec getCapability(K Key) {
  SPIRVCapVec V;
  SPIRVMap<K, SPIRVCapVec>::find(Key, &V);
  return V;
}

#define ADD_VEC_INIT(Cap, ...)                                                 \
  {                                                                            \
    SPIRVCapabilityKind C[] = __VA_ARGS__;                                     \
    SPIRVCapVec V(C, C + sizeof(C) / sizeof(C[0]));                            \
    add(Cap, V);                                                               \
  }

template <> inline void SPIRVMap<SPIRVCapabilityKind, SPIRVCapVec>::init() {
  ADD_VEC_INIT(CapabilityShader, {CapabilityMatrix});
  ADD_VEC_INIT(CapabilityGeometry, {CapabilityShader});
  ADD_VEC_INIT(CapabilityTessellation, {CapabilityShader});
  ADD_VEC_INIT(CapabilityVector16, {CapabilityKernel});
  ADD_VEC_INIT(CapabilityFloat16Buffer, {CapabilityKernel});
  ADD_VEC_INIT(CapabilityInt64Atomics, {CapabilityInt64});
  ADD_VEC_INIT(CapabilityImageBasic, {CapabilityKernel});
  ADD_VEC_INIT(CapabilityImageReadWrite, {CapabilityImageBasic});
  ADD_VEC_INIT(CapabilityImageMipmap, {CapabilityImageBasic});
  ADD_VEC_INIT(CapabilityPipes, {CapabilityKernel});
  ADD_VEC_INIT(CapabilityDeviceEnqueue, {CapabilityKernel});
  ADD_VEC_INIT(CapabilityLiteralSampler, {CapabilityKernel});
  ADD_VEC_INIT(CapabilityAtomicStorage, {CapabilityShader});
  ADD_VEC_INIT(CapabilityTessellationPointSize, {CapabilityTessellation});
  ADD_VEC_INIT(CapabilityGeometryPointSize, {CapabilityGeometry});
  ADD_VEC_INIT(CapabilityImageGatherExtended, {CapabilityShader});
  ADD_VEC_INIT(CapabilityStorageImageMultisample, {CapabilityShader});
  ADD_VEC_INIT(CapabilityUniformBufferArrayDynamicIndexing, {CapabilityShader});
  ADD_VEC_INIT(CapabilitySampledImageArrayDynamicIndexing, {CapabilityShader});
  ADD_VEC_INIT(CapabilityStorageBufferArrayDynamicIndexing, {CapabilityShader});
  ADD_VEC_INIT(CapabilityStorageImageArrayDynamicIndexing, {CapabilityShader});
  ADD_VEC_INIT(CapabilityClipDistance, {CapabilityShader});
  ADD_VEC_INIT(CapabilityCullDistance, {CapabilityShader});
  ADD_VEC_INIT(CapabilityImageCubeArray, {CapabilitySampledCubeArray});
  ADD_VEC_INIT(CapabilitySampleRateShading, {CapabilityShader});
  ADD_VEC_INIT(CapabilityImageRect, {CapabilitySampledRect});
  ADD_VEC_INIT(CapabilitySampledRect, {CapabilityShader});
  ADD_VEC_INIT(CapabilityGenericPointer, {CapabilityAddresses});
  ADD_VEC_INIT(CapabilityInt8, {CapabilityKernel});
  ADD_VEC_INIT(CapabilityInputAttachment, {CapabilityShader});
  ADD_VEC_INIT(CapabilitySparseResidency, {CapabilityShader});
  ADD_VEC_INIT(CapabilityMinLod, {CapabilityShader});
  ADD_VEC_INIT(CapabilityImage1D, {CapabilitySampled1D});
  ADD_VEC_INIT(CapabilitySampledCubeArray, {CapabilityShader});
  ADD_VEC_INIT(CapabilityImageBuffer, {CapabilitySampledBuffer});
  ADD_VEC_INIT(CapabilityImageMSArray, {CapabilityShader});
  ADD_VEC_INIT(CapabilityStorageImageExtendedFormats, {CapabilityShader});
  ADD_VEC_INIT(CapabilityImageQuery, {CapabilityShader});
  ADD_VEC_INIT(CapabilityDerivativeControl, {CapabilityShader});
  ADD_VEC_INIT(CapabilityInterpolationFunction, {CapabilityShader});
  ADD_VEC_INIT(CapabilityTransformFeedback, {CapabilityShader});
  ADD_VEC_INIT(CapabilityGeometryStreams, {CapabilityGeometry});
  ADD_VEC_INIT(CapabilityStorageImageReadWithoutFormat, {CapabilityShader});
  ADD_VEC_INIT(CapabilityStorageImageWriteWithoutFormat, {CapabilityShader});
  ADD_VEC_INIT(CapabilityMultiViewport, {CapabilityGeometry});
  ADD_VEC_INIT(CapabilitySubgroupAvcMotionEstimationINTEL, {CapabilityGroups});
  ADD_VEC_INIT(CapabilitySubgroupAvcMotionEstimationIntraINTEL,
               {CapabilitySubgroupAvcMotionEstimationINTEL});
  ADD_VEC_INIT(CapabilitySubgroupAvcMotionEstimationChromaINTEL,
               {CapabilitySubgroupAvcMotionEstimationIntraINTEL});
}

template <> inline void SPIRVMap<SPIRVExecutionModelKind, SPIRVCapVec>::init() {
  ADD_VEC_INIT(ExecutionModelVertex, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModelTessellationControl, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModelTessellationEvaluation, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModelGeometry, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModelFragment, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModelGLCompute, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModelKernel, {CapabilityKernel});
}

template <> inline void SPIRVMap<SPIRVExecutionModeKind, SPIRVCapVec>::init() {
  ADD_VEC_INIT(ExecutionModeInvocations, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModeSpacingEqual, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeSpacingFractionalEven, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeSpacingFractionalOdd, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeVertexOrderCw, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeVertexOrderCcw, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModePixelCenterInteger, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModeOriginUpperLeft, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModeOriginLowerLeft, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModeEarlyFragmentTests, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModePointMode, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeXfb, {CapabilityTransformFeedback});
  ADD_VEC_INIT(ExecutionModeDepthReplacing, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModeDepthGreater, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModeDepthLess, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModeDepthUnchanged, {CapabilityShader});
  ADD_VEC_INIT(ExecutionModeLocalSizeHint, {CapabilityKernel});
  ADD_VEC_INIT(ExecutionModeInputPoints, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModeInputLines, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModeInputLinesAdjacency, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModeTriangles,
               {CapabilityGeometry, CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeInputTrianglesAdjacency, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModeQuads, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeIsolines, {CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeOutputVertices,
               {CapabilityGeometry, CapabilityTessellation});
  ADD_VEC_INIT(ExecutionModeOutputPoints, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModeOutputLineStrip, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModeOutputTriangleStrip, {CapabilityGeometry});
  ADD_VEC_INIT(ExecutionModeVecTypeHint, {CapabilityKernel});
  ADD_VEC_INIT(ExecutionModeContractionOff, {CapabilityKernel});
}

template <> inline void SPIRVMap<SPIRVMemoryModelKind, SPIRVCapVec>::init() {
  ADD_VEC_INIT(MemoryModelSimple, {CapabilityShader});
  ADD_VEC_INIT(MemoryModelGLSL450, {CapabilityShader});
  ADD_VEC_INIT(MemoryModelOpenCL, {CapabilityKernel});
}

template <> inline void SPIRVMap<SPIRVStorageClassKind, SPIRVCapVec>::init() {
  ADD_VEC_INIT(StorageClassUniform, {CapabilityShader});
  ADD_VEC_INIT(StorageClassOutput, {CapabilityShader});
  ADD_VEC_INIT(StorageClassPrivate, {CapabilityShader});
  ADD_VEC_INIT(StorageClassGeneric, {CapabilityGenericPointer});
  ADD_VEC_INIT(StorageClassPushConstant, {CapabilityShader});
  ADD_VEC_INIT(StorageClassAtomicCounter, {CapabilityAtomicStorage});
}

template <> inline void SPIRVMap<SPIRVImageDimKind, SPIRVCapVec>::init() {
  ADD_VEC_INIT(Dim1D, {CapabilitySampled1D});
  ADD_VEC_INIT(DimCube, {CapabilityShader});
  ADD_VEC_INIT(DimRect, {CapabilitySampledRect});
  ADD_VEC_INIT(DimBuffer, {CapabilitySampledBuffer});
  ADD_VEC_INIT(DimSubpassData, {CapabilityInputAttachment});
}

template <> inline void SPIRVMap<ImageFormat, SPIRVCapVec>::init() {
  ADD_VEC_INIT(ImageFormatRgba32f, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRgba16f, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatR32f, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRgba8, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRgba8Snorm, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRg32f, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg16f, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR11fG11fB10f,
               {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR16f, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRgba16, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRgb10A2, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg16, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg8, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR16, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR8, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRgba16Snorm, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg16Snorm, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg8Snorm, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR16Snorm, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR8Snorm, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRgba32i, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRgba16i, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRgba8i, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatR32i, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRg32i, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg16i, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg8i, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR16i, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR8i, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRgba32ui, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRgba16ui, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRgba8ui, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatR32ui, {CapabilityShader});
  ADD_VEC_INIT(ImageFormatRgb10a2ui, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg32ui, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatRg16ui, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR16ui, {CapabilityStorageImageExtendedFormats});
  ADD_VEC_INIT(ImageFormatR8ui, {CapabilityStorageImageExtendedFormats});
}

template <> inline void SPIRVMap<ImageOperandsMask, SPIRVCapVec>::init() {
  ADD_VEC_INIT(ImageOperandsBiasMask, {CapabilityShader});
  ADD_VEC_INIT(ImageOperandsOffsetMask, {CapabilityImageGatherExtended});
  ADD_VEC_INIT(ImageOperandsMinLodMask, {CapabilityMinLod});
}

template <> inline void SPIRVMap<Decoration, SPIRVCapVec>::init() {
  ADD_VEC_INIT(DecorationRelaxedPrecision, {CapabilityShader});
  ADD_VEC_INIT(DecorationSpecId, {CapabilityShader});
  ADD_VEC_INIT(DecorationBlock, {CapabilityShader});
  ADD_VEC_INIT(DecorationBufferBlock, {CapabilityShader});
  ADD_VEC_INIT(DecorationRowMajor, {CapabilityMatrix});
  ADD_VEC_INIT(DecorationColMajor, {CapabilityMatrix});
  ADD_VEC_INIT(DecorationArrayStride, {CapabilityShader});
  ADD_VEC_INIT(DecorationMatrixStride, {CapabilityMatrix});
  ADD_VEC_INIT(DecorationGLSLShared, {CapabilityShader});
  ADD_VEC_INIT(DecorationGLSLPacked, {CapabilityShader});
  ADD_VEC_INIT(DecorationCPacked, {CapabilityKernel});
  ADD_VEC_INIT(DecorationNoPerspective, {CapabilityShader});
  ADD_VEC_INIT(DecorationFlat, {CapabilityShader});
  ADD_VEC_INIT(DecorationPatch, {CapabilityTessellation});
  ADD_VEC_INIT(DecorationCentroid, {CapabilityShader});
  ADD_VEC_INIT(DecorationSample, {CapabilitySampleRateShading});
  ADD_VEC_INIT(DecorationInvariant, {CapabilityShader});
  ADD_VEC_INIT(DecorationConstant, {CapabilityKernel});
  ADD_VEC_INIT(DecorationUniform, {CapabilityShader});
  ADD_VEC_INIT(DecorationSaturatedConversion, {CapabilityKernel});
  ADD_VEC_INIT(DecorationStream, {CapabilityGeometryStreams});
  ADD_VEC_INIT(DecorationLocation, {CapabilityShader});
  ADD_VEC_INIT(DecorationComponent, {CapabilityShader});
  ADD_VEC_INIT(DecorationIndex, {CapabilityShader});
  ADD_VEC_INIT(DecorationBinding, {CapabilityShader});
  ADD_VEC_INIT(DecorationDescriptorSet, {CapabilityShader});
  ADD_VEC_INIT(DecorationOffset, {CapabilityShader});
  ADD_VEC_INIT(DecorationXfbBuffer, {CapabilityTransformFeedback});
  ADD_VEC_INIT(DecorationXfbStride, {CapabilityTransformFeedback});
  ADD_VEC_INIT(DecorationFuncParamAttr, {CapabilityKernel});
  ADD_VEC_INIT(DecorationFPRoundingMode, {CapabilityKernel});
  ADD_VEC_INIT(DecorationFPFastMathMode, {CapabilityKernel});
  ADD_VEC_INIT(DecorationLinkageAttributes, {CapabilityLinkage});
  ADD_VEC_INIT(DecorationNoContraction, {CapabilityShader});
  ADD_VEC_INIT(DecorationInputAttachmentIndex, {CapabilityInputAttachment});
  ADD_VEC_INIT(DecorationAlignment, {CapabilityKernel});
  ADD_VEC_INIT(DecorationRegisterINTEL, {CapabilityFPGAMemoryAttributesINTEL});
  ADD_VEC_INIT(DecorationMemoryINTEL, {CapabilityFPGAMemoryAttributesINTEL});
  ADD_VEC_INIT(DecorationNumbanksINTEL, {CapabilityFPGAMemoryAttributesINTEL});
  ADD_VEC_INIT(DecorationBankwidthINTEL, {CapabilityFPGAMemoryAttributesINTEL});
  ADD_VEC_INIT(DecorationMaxconcurrencyINTEL,
               {CapabilityFPGAMemoryAttributesINTEL});
  ADD_VEC_INIT(DecorationSinglepumpINTEL,
               {CapabilityFPGAMemoryAttributesINTEL});
  ADD_VEC_INIT(DecorationDoublepumpINTEL,
               {CapabilityFPGAMemoryAttributesINTEL});
}

template <> inline void SPIRVMap<BuiltIn, SPIRVCapVec>::init() {
  ADD_VEC_INIT(BuiltInPosition, {CapabilityShader});
  ADD_VEC_INIT(BuiltInPointSize, {CapabilityShader});
  ADD_VEC_INIT(BuiltInClipDistance, {CapabilityClipDistance});
  ADD_VEC_INIT(BuiltInCullDistance, {CapabilityCullDistance});
  ADD_VEC_INIT(BuiltInVertexId, {CapabilityShader});
  ADD_VEC_INIT(BuiltInInstanceId, {CapabilityShader});
  ADD_VEC_INIT(BuiltInPrimitiveId,
               {CapabilityGeometry, CapabilityTessellation});
  ADD_VEC_INIT(BuiltInInvocationId,
               {CapabilityGeometry, CapabilityTessellation});
  ADD_VEC_INIT(BuiltInLayer, {CapabilityGeometry});
  ADD_VEC_INIT(BuiltInViewportIndex, {CapabilityMultiViewport});
  ADD_VEC_INIT(BuiltInTessLevelOuter, {CapabilityTessellation});
  ADD_VEC_INIT(BuiltInTessLevelInner, {CapabilityTessellation});
  ADD_VEC_INIT(BuiltInTessCoord, {CapabilityTessellation});
  ADD_VEC_INIT(BuiltInPatchVertices, {CapabilityTessellation});
  ADD_VEC_INIT(BuiltInFragCoord, {CapabilityShader});
  ADD_VEC_INIT(BuiltInPointCoord, {CapabilityShader});
  ADD_VEC_INIT(BuiltInFrontFacing, {CapabilityShader});
  ADD_VEC_INIT(BuiltInSampleId, {CapabilitySampleRateShading});
  ADD_VEC_INIT(BuiltInSamplePosition, {CapabilitySampleRateShading});
  ADD_VEC_INIT(BuiltInSampleMask, {CapabilitySampleRateShading});
  ADD_VEC_INIT(BuiltInFragDepth, {CapabilityShader});
  ADD_VEC_INIT(BuiltInHelperInvocation, {CapabilityShader});
  ADD_VEC_INIT(BuiltInWorkDim, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInGlobalSize, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInEnqueuedWorkgroupSize, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInGlobalOffset, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInGlobalLinearId, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInSubgroupSize, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInSubgroupMaxSize, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInNumSubgroups, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInNumEnqueuedSubgroups, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInSubgroupId, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInSubgroupLocalInvocationId, {CapabilityKernel});
  ADD_VEC_INIT(BuiltInVertexIndex, {CapabilityShader});
  ADD_VEC_INIT(BuiltInInstanceIndex, {CapabilityShader});
}

template <> inline void SPIRVMap<MemorySemanticsMask, SPIRVCapVec>::init() {
  ADD_VEC_INIT(MemorySemanticsUniformMemoryMask, {CapabilityShader});
  ADD_VEC_INIT(MemorySemanticsAtomicCounterMemoryMask,
               {CapabilityAtomicStorage});
}

#undef ADD_VEC_INIT

inline unsigned getImageDimension(SPIRVImageDimKind K) {
  switch (K) {
  case Dim1D:
    return 1;
  case Dim2D:
    return 2;
  case Dim3D:
    return 3;
  case DimCube:
    return 2;
  case DimRect:
    return 2;
  case DimBuffer:
    return 1;
  default:
    return 0;
  }
}

/// Extract memory order part of SPIR-V memory semantics.
inline unsigned extractSPIRVMemOrderSemantic(unsigned Sema) {
  return Sema & KSpirvMemOrderSemanticMask;
}

} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVENUM_H
