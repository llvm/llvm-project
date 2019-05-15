//===- SPIRVNameMapEnum.h - SPIR-V NameMap enums ----------------*- C++ -*-===//
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
/// This file defines SPIR-V NameMap enums.
///
//===----------------------------------------------------------------------===//
// WARNING:
//
// This file has been generated using `tools/spirv-tool/gen_spirv.bash` and
// should not be modified manually. If the file needs to be updated, edit the
// script and any other source file instead, before re-generating this file.
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVNAMEMAPENUM_H
#define SPIRV_LIBSPIRV_SPIRVNAMEMAPENUM_H

#include "SPIRVEnum.h"
#include "spirv.hpp"

using namespace spv;

namespace SPIRV {

template <> inline void SPIRVMap<SourceLanguage, std::string>::init() {
  add(SourceLanguageUnknown, "Unknown");
  add(SourceLanguageESSL, "ESSL");
  add(SourceLanguageGLSL, "GLSL");
  add(SourceLanguageOpenCL_C, "OpenCL_C");
  add(SourceLanguageOpenCL_CPP, "OpenCL_CPP");
}
SPIRV_DEF_NAMEMAP(SourceLanguage, SPIRVSourceLanguageNameMap)

template <> inline void SPIRVMap<ExecutionModel, std::string>::init() {
  add(ExecutionModelVertex, "Vertex");
  add(ExecutionModelTessellationControl, "TessellationControl");
  add(ExecutionModelTessellationEvaluation, "TessellationEvaluation");
  add(ExecutionModelGeometry, "Geometry");
  add(ExecutionModelFragment, "Fragment");
  add(ExecutionModelGLCompute, "GLCompute");
  add(ExecutionModelKernel, "Kernel");
}
SPIRV_DEF_NAMEMAP(ExecutionModel, SPIRVExecutionModelNameMap)

template <> inline void SPIRVMap<AddressingModel, std::string>::init() {
  add(AddressingModelLogical, "Logical");
  add(AddressingModelPhysical32, "Physical32");
  add(AddressingModelPhysical64, "Physical64");
}
SPIRV_DEF_NAMEMAP(AddressingModel, SPIRVAddressingModelNameMap)

template <> inline void SPIRVMap<MemoryModel, std::string>::init() {
  add(MemoryModelSimple, "Simple");
  add(MemoryModelGLSL450, "GLSL450");
  add(MemoryModelOpenCL, "OpenCL");
}
SPIRV_DEF_NAMEMAP(MemoryModel, SPIRVMemoryModelNameMap)

template <> inline void SPIRVMap<ExecutionMode, std::string>::init() {
  add(ExecutionModeInvocations, "Invocations");
  add(ExecutionModeSpacingEqual, "SpacingEqual");
  add(ExecutionModeSpacingFractionalEven, "SpacingFractionalEven");
  add(ExecutionModeSpacingFractionalOdd, "SpacingFractionalOdd");
  add(ExecutionModeVertexOrderCw, "VertexOrderCw");
  add(ExecutionModeVertexOrderCcw, "VertexOrderCcw");
  add(ExecutionModePixelCenterInteger, "PixelCenterInteger");
  add(ExecutionModeOriginUpperLeft, "OriginUpperLeft");
  add(ExecutionModeOriginLowerLeft, "OriginLowerLeft");
  add(ExecutionModeEarlyFragmentTests, "EarlyFragmentTests");
  add(ExecutionModePointMode, "PointMode");
  add(ExecutionModeXfb, "Xfb");
  add(ExecutionModeDepthReplacing, "DepthReplacing");
  add(ExecutionModeDepthGreater, "DepthGreater");
  add(ExecutionModeDepthLess, "DepthLess");
  add(ExecutionModeDepthUnchanged, "DepthUnchanged");
  add(ExecutionModeLocalSize, "LocalSize");
  add(ExecutionModeLocalSizeHint, "LocalSizeHint");
  add(ExecutionModeInputPoints, "InputPoints");
  add(ExecutionModeInputLines, "InputLines");
  add(ExecutionModeInputLinesAdjacency, "InputLinesAdjacency");
  add(ExecutionModeTriangles, "Triangles");
  add(ExecutionModeInputTrianglesAdjacency, "InputTrianglesAdjacency");
  add(ExecutionModeQuads, "Quads");
  add(ExecutionModeIsolines, "Isolines");
  add(ExecutionModeOutputVertices, "OutputVertices");
  add(ExecutionModeOutputPoints, "OutputPoints");
  add(ExecutionModeOutputLineStrip, "OutputLineStrip");
  add(ExecutionModeOutputTriangleStrip, "OutputTriangleStrip");
  add(ExecutionModeVecTypeHint, "VecTypeHint");
  add(ExecutionModeContractionOff, "ContractionOff");
}
SPIRV_DEF_NAMEMAP(ExecutionMode, SPIRVExecutionModeNameMap)

template <> inline void SPIRVMap<StorageClass, std::string>::init() {
  add(StorageClassUniformConstant, "UniformConstant");
  add(StorageClassInput, "Input");
  add(StorageClassUniform, "Uniform");
  add(StorageClassOutput, "Output");
  add(StorageClassWorkgroup, "Workgroup");
  add(StorageClassCrossWorkgroup, "CrossWorkgroup");
  add(StorageClassPrivate, "Private");
  add(StorageClassFunction, "Function");
  add(StorageClassGeneric, "Generic");
  add(StorageClassPushConstant, "PushConstant");
  add(StorageClassAtomicCounter, "AtomicCounter");
  add(StorageClassImage, "Image");
}
SPIRV_DEF_NAMEMAP(StorageClass, SPIRVStorageClassNameMap)

template <> inline void SPIRVMap<Dim, std::string>::init() {
  add(Dim1D, "1D");
  add(Dim2D, "2D");
  add(Dim3D, "3D");
  add(DimCube, "Cube");
  add(DimRect, "Rect");
  add(DimBuffer, "Buffer");
  add(DimSubpassData, "SubpassData");
}
SPIRV_DEF_NAMEMAP(Dim, SPIRVDimNameMap)

template <> inline void SPIRVMap<SamplerAddressingMode, std::string>::init() {
  add(SamplerAddressingModeNone, "None");
  add(SamplerAddressingModeClampToEdge, "ClampToEdge");
  add(SamplerAddressingModeClamp, "Clamp");
  add(SamplerAddressingModeRepeat, "Repeat");
  add(SamplerAddressingModeRepeatMirrored, "RepeatMirrored");
}
SPIRV_DEF_NAMEMAP(SamplerAddressingMode, SPIRVSamplerAddressingModeNameMap)

template <> inline void SPIRVMap<SamplerFilterMode, std::string>::init() {
  add(SamplerFilterModeNearest, "Nearest");
  add(SamplerFilterModeLinear, "Linear");
}
SPIRV_DEF_NAMEMAP(SamplerFilterMode, SPIRVSamplerFilterModeNameMap)

template <> inline void SPIRVMap<ImageFormat, std::string>::init() {
  add(ImageFormatUnknown, "Unknown");
  add(ImageFormatRgba32f, "Rgba32f");
  add(ImageFormatRgba16f, "Rgba16f");
  add(ImageFormatR32f, "R32f");
  add(ImageFormatRgba8, "Rgba8");
  add(ImageFormatRgba8Snorm, "Rgba8Snorm");
  add(ImageFormatRg32f, "Rg32f");
  add(ImageFormatRg16f, "Rg16f");
  add(ImageFormatR11fG11fB10f, "R11fG11fB10f");
  add(ImageFormatR16f, "R16f");
  add(ImageFormatRgba16, "Rgba16");
  add(ImageFormatRgb10A2, "Rgb10A2");
  add(ImageFormatRg16, "Rg16");
  add(ImageFormatRg8, "Rg8");
  add(ImageFormatR16, "R16");
  add(ImageFormatR8, "R8");
  add(ImageFormatRgba16Snorm, "Rgba16Snorm");
  add(ImageFormatRg16Snorm, "Rg16Snorm");
  add(ImageFormatRg8Snorm, "Rg8Snorm");
  add(ImageFormatR16Snorm, "R16Snorm");
  add(ImageFormatR8Snorm, "R8Snorm");
  add(ImageFormatRgba32i, "Rgba32i");
  add(ImageFormatRgba16i, "Rgba16i");
  add(ImageFormatRgba8i, "Rgba8i");
  add(ImageFormatR32i, "R32i");
  add(ImageFormatRg32i, "Rg32i");
  add(ImageFormatRg16i, "Rg16i");
  add(ImageFormatRg8i, "Rg8i");
  add(ImageFormatR16i, "R16i");
  add(ImageFormatR8i, "R8i");
  add(ImageFormatRgba32ui, "Rgba32ui");
  add(ImageFormatRgba16ui, "Rgba16ui");
  add(ImageFormatRgba8ui, "Rgba8ui");
  add(ImageFormatR32ui, "R32ui");
  add(ImageFormatRgb10a2ui, "Rgb10a2ui");
  add(ImageFormatRg32ui, "Rg32ui");
  add(ImageFormatRg16ui, "Rg16ui");
  add(ImageFormatRg8ui, "Rg8ui");
  add(ImageFormatR16ui, "R16ui");
  add(ImageFormatR8ui, "R8ui");
}
SPIRV_DEF_NAMEMAP(ImageFormat, SPIRVImageFormatNameMap)

template <> inline void SPIRVMap<ImageChannelOrder, std::string>::init() {
  add(ImageChannelOrderR, "R");
  add(ImageChannelOrderA, "A");
  add(ImageChannelOrderRG, "RG");
  add(ImageChannelOrderRA, "RA");
  add(ImageChannelOrderRGB, "RGB");
  add(ImageChannelOrderRGBA, "RGBA");
  add(ImageChannelOrderBGRA, "BGRA");
  add(ImageChannelOrderARGB, "ARGB");
  add(ImageChannelOrderIntensity, "Intensity");
  add(ImageChannelOrderLuminance, "Luminance");
  add(ImageChannelOrderRx, "Rx");
  add(ImageChannelOrderRGx, "RGx");
  add(ImageChannelOrderRGBx, "RGBx");
  add(ImageChannelOrderDepth, "Depth");
  add(ImageChannelOrderDepthStencil, "DepthStencil");
}
SPIRV_DEF_NAMEMAP(ImageChannelOrder, SPIRVImageChannelOrderNameMap)

template <> inline void SPIRVMap<ImageChannelDataType, std::string>::init() {
  add(ImageChannelDataTypeSnormInt8, "SnormInt8");
  add(ImageChannelDataTypeSnormInt16, "SnormInt16");
  add(ImageChannelDataTypeUnormInt8, "UnormInt8");
  add(ImageChannelDataTypeUnormInt16, "UnormInt16");
  add(ImageChannelDataTypeUnormShort565, "UnormShort565");
  add(ImageChannelDataTypeUnormShort555, "UnormShort555");
  add(ImageChannelDataTypeUnormInt101010, "UnormInt101010");
  add(ImageChannelDataTypeSignedInt8, "SignedInt8");
  add(ImageChannelDataTypeSignedInt16, "SignedInt16");
  add(ImageChannelDataTypeSignedInt32, "SignedInt32");
  add(ImageChannelDataTypeUnsignedInt8, "UnsignedInt8");
  add(ImageChannelDataTypeUnsignedInt16, "UnsignedInt16");
  add(ImageChannelDataTypeUnsignedInt32, "UnsignedInt32");
  add(ImageChannelDataTypeHalfFloat, "HalfFloat");
  add(ImageChannelDataTypeFloat, "Float");
  add(ImageChannelDataTypeUnormInt24, "UnormInt24");
  add(ImageChannelDataTypeUnormInt101010_2, "UnormInt101010_2");
}
SPIRV_DEF_NAMEMAP(ImageChannelDataType, SPIRVImageChannelDataTypeNameMap)

template <> inline void SPIRVMap<FPRoundingMode, std::string>::init() {
  add(FPRoundingModeRTE, "RTE");
  add(FPRoundingModeRTZ, "RTZ");
  add(FPRoundingModeRTP, "RTP");
  add(FPRoundingModeRTN, "RTN");
}
SPIRV_DEF_NAMEMAP(FPRoundingMode, SPIRVFPRoundingModeNameMap)

template <> inline void SPIRVMap<LinkageType, std::string>::init() {
  add(LinkageTypeExport, "Export");
  add(LinkageTypeImport, "Import");
  add(LinkageTypeInternal, "Internal");
}
SPIRV_DEF_NAMEMAP(LinkageType, SPIRVLinkageTypeNameMap)

template <> inline void SPIRVMap<AccessQualifier, std::string>::init() {
  add(AccessQualifierReadOnly, "ReadOnly");
  add(AccessQualifierWriteOnly, "WriteOnly");
  add(AccessQualifierReadWrite, "ReadWrite");
}
SPIRV_DEF_NAMEMAP(AccessQualifier, SPIRVAccessQualifierNameMap)

template <>
inline void SPIRVMap<FunctionParameterAttribute, std::string>::init() {
  add(FunctionParameterAttributeZext, "Zext");
  add(FunctionParameterAttributeSext, "Sext");
  add(FunctionParameterAttributeByVal, "ByVal");
  add(FunctionParameterAttributeSret, "Sret");
  add(FunctionParameterAttributeNoAlias, "NoAlias");
  add(FunctionParameterAttributeNoCapture, "NoCapture");
  add(FunctionParameterAttributeNoWrite, "NoWrite");
  add(FunctionParameterAttributeNoReadWrite, "NoReadWrite");
}
SPIRV_DEF_NAMEMAP(FunctionParameterAttribute,
                  SPIRVFunctionParameterAttributeNameMap)

template <> inline void SPIRVMap<Decoration, std::string>::init() {
  add(DecorationRelaxedPrecision, "RelaxedPrecision");
  add(DecorationSpecId, "SpecId");
  add(DecorationBlock, "Block");
  add(DecorationBufferBlock, "BufferBlock");
  add(DecorationRowMajor, "RowMajor");
  add(DecorationColMajor, "ColMajor");
  add(DecorationArrayStride, "ArrayStride");
  add(DecorationMatrixStride, "MatrixStride");
  add(DecorationGLSLShared, "GLSLShared");
  add(DecorationGLSLPacked, "GLSLPacked");
  add(DecorationCPacked, "CPacked");
  add(DecorationBuiltIn, "BuiltIn");
  add(DecorationNoPerspective, "NoPerspective");
  add(DecorationFlat, "Flat");
  add(DecorationPatch, "Patch");
  add(DecorationCentroid, "Centroid");
  add(DecorationSample, "Sample");
  add(DecorationInvariant, "Invariant");
  add(DecorationRestrict, "Restrict");
  add(DecorationAliased, "Aliased");
  add(DecorationVolatile, "Volatile");
  add(DecorationConstant, "Constant");
  add(DecorationCoherent, "Coherent");
  add(DecorationNonWritable, "NonWritable");
  add(DecorationNonReadable, "NonReadable");
  add(DecorationUniform, "Uniform");
  add(DecorationSaturatedConversion, "SaturatedConversion");
  add(DecorationStream, "Stream");
  add(DecorationLocation, "Location");
  add(DecorationComponent, "Component");
  add(DecorationIndex, "Index");
  add(DecorationBinding, "Binding");
  add(DecorationDescriptorSet, "DescriptorSet");
  add(DecorationOffset, "Offset");
  add(DecorationXfbBuffer, "XfbBuffer");
  add(DecorationXfbStride, "XfbStride");
  add(DecorationFuncParamAttr, "FuncParamAttr");
  add(DecorationFPRoundingMode, "FPRoundingMode");
  add(DecorationFPFastMathMode, "FPFastMathMode");
  add(DecorationLinkageAttributes, "LinkageAttributes");
  add(DecorationNoContraction, "NoContraction");
  add(DecorationInputAttachmentIndex, "InputAttachmentIndex");
  add(DecorationAlignment, "Alignment");
  add(DecorationMaxByteOffset, "MaxByteOffset");
  add(DecorationNoSignedWrap, "NoSignedWrap");
  add(DecorationNoUnsignedWrap, "NoUnsignedWrap");
  add(DecorationRegisterINTEL, "RegisterINTEL");
  add(DecorationMemoryINTEL, "MemoryINTEL");
  add(DecorationNumbanksINTEL, "NumbanksINTEL");
  add(DecorationBankwidthINTEL, "BankwidthINTEL");
  add(DecorationMaxconcurrencyINTEL, "MaxconcurrencyINTEL");
  add(DecorationSinglepumpINTEL, "SinglepumpINTEL");
  add(DecorationDoublepumpINTEL, "DoublepumpINTEL");
}
SPIRV_DEF_NAMEMAP(Decoration, SPIRVDecorationNameMap)

template <> inline void SPIRVMap<BuiltIn, std::string>::init() {
  add(BuiltInPosition, "BuiltInPosition");
  add(BuiltInPointSize, "BuiltInPointSize");
  add(BuiltInClipDistance, "BuiltInClipDistance");
  add(BuiltInCullDistance, "BuiltInCullDistance");
  add(BuiltInVertexId, "BuiltInVertexId");
  add(BuiltInInstanceId, "BuiltInInstanceId");
  add(BuiltInPrimitiveId, "BuiltInPrimitiveId");
  add(BuiltInInvocationId, "BuiltInInvocationId");
  add(BuiltInLayer, "BuiltInLayer");
  add(BuiltInViewportIndex, "BuiltInViewportIndex");
  add(BuiltInTessLevelOuter, "BuiltInTessLevelOuter");
  add(BuiltInTessLevelInner, "BuiltInTessLevelInner");
  add(BuiltInTessCoord, "BuiltInTessCoord");
  add(BuiltInPatchVertices, "BuiltInPatchVertices");
  add(BuiltInFragCoord, "BuiltInFragCoord");
  add(BuiltInPointCoord, "BuiltInPointCoord");
  add(BuiltInFrontFacing, "BuiltInFrontFacing");
  add(BuiltInSampleId, "BuiltInSampleId");
  add(BuiltInSamplePosition, "BuiltInSamplePosition");
  add(BuiltInSampleMask, "BuiltInSampleMask");
  add(BuiltInFragDepth, "BuiltInFragDepth");
  add(BuiltInHelperInvocation, "BuiltInHelperInvocation");
  add(BuiltInNumWorkgroups, "BuiltInNumWorkgroups");
  add(BuiltInWorkgroupSize, "BuiltInWorkgroupSize");
  add(BuiltInWorkgroupId, "BuiltInWorkgroupId");
  add(BuiltInLocalInvocationId, "BuiltInLocalInvocationId");
  add(BuiltInGlobalInvocationId, "BuiltInGlobalInvocationId");
  add(BuiltInLocalInvocationIndex, "BuiltInLocalInvocationIndex");
  add(BuiltInWorkDim, "BuiltInWorkDim");
  add(BuiltInGlobalSize, "BuiltInGlobalSize");
  add(BuiltInEnqueuedWorkgroupSize, "BuiltInEnqueuedWorkgroupSize");
  add(BuiltInGlobalOffset, "BuiltInGlobalOffset");
  add(BuiltInGlobalLinearId, "BuiltInGlobalLinearId");
  add(BuiltInSubgroupSize, "BuiltInSubgroupSize");
  add(BuiltInSubgroupMaxSize, "BuiltInSubgroupMaxSize");
  add(BuiltInNumSubgroups, "BuiltInNumSubgroups");
  add(BuiltInNumEnqueuedSubgroups, "BuiltInNumEnqueuedSubgroups");
  add(BuiltInSubgroupId, "BuiltInSubgroupId");
  add(BuiltInSubgroupLocalInvocationId, "BuiltInSubgroupLocalInvocationId");
  add(BuiltInVertexIndex, "BuiltInVertexIndex");
  add(BuiltInInstanceIndex, "BuiltInInstanceIndex");
}
SPIRV_DEF_NAMEMAP(BuiltIn, SPIRVBuiltInNameMap)

template <> inline void SPIRVMap<Scope, std::string>::init() {
  add(ScopeCrossDevice, "CrossDevice");
  add(ScopeDevice, "Device");
  add(ScopeWorkgroup, "Workgroup");
  add(ScopeSubgroup, "Subgroup");
  add(ScopeInvocation, "Invocation");
}
SPIRV_DEF_NAMEMAP(Scope, SPIRVScopeNameMap)

template <> inline void SPIRVMap<GroupOperation, std::string>::init() {
  add(GroupOperationReduce, "Reduce");
  add(GroupOperationInclusiveScan, "InclusiveScan");
  add(GroupOperationExclusiveScan, "ExclusiveScan");
}
SPIRV_DEF_NAMEMAP(GroupOperation, SPIRVGroupOperationNameMap)

template <> inline void SPIRVMap<KernelEnqueueFlags, std::string>::init() {
  add(KernelEnqueueFlagsNoWait, "NoWait");
  add(KernelEnqueueFlagsWaitKernel, "WaitKernel");
  add(KernelEnqueueFlagsWaitWorkGroup, "WaitWorkGroup");
}
SPIRV_DEF_NAMEMAP(KernelEnqueueFlags, SPIRVKernelEnqueueFlagsNameMap)

template <> inline void SPIRVMap<Capability, std::string>::init() {
  add(CapabilityMatrix, "Matrix");
  add(CapabilityShader, "Shader");
  add(CapabilityGeometry, "Geometry");
  add(CapabilityTessellation, "Tessellation");
  add(CapabilityAddresses, "Addresses");
  add(CapabilityLinkage, "Linkage");
  add(CapabilityKernel, "Kernel");
  add(CapabilityVector16, "Vector16");
  add(CapabilityFloat16Buffer, "Float16Buffer");
  add(CapabilityFloat16, "Float16");
  add(CapabilityFloat64, "Float64");
  add(CapabilityInt64, "Int64");
  add(CapabilityInt64Atomics, "Int64Atomics");
  add(CapabilityImageBasic, "ImageBasic");
  add(CapabilityImageReadWrite, "ImageReadWrite");
  add(CapabilityImageMipmap, "ImageMipmap");
  add(CapabilityPipes, "Pipes");
  add(CapabilityPipeStorage, "PipeStorage");
  add(CapabilityGroups, "Groups");
  add(CapabilityDeviceEnqueue, "DeviceEnqueue");
  add(CapabilityLiteralSampler, "LiteralSampler");
  add(CapabilityAtomicStorage, "AtomicStorage");
  add(CapabilityInt16, "Int16");
  add(CapabilityTessellationPointSize, "TessellationPointSize");
  add(CapabilityGeometryPointSize, "GeometryPointSize");
  add(CapabilityImageGatherExtended, "ImageGatherExtended");
  add(CapabilityStorageImageMultisample, "StorageImageMultisample");
  add(CapabilityUniformBufferArrayDynamicIndexing,
      "UniformBufferArrayDynamicIndexing");
  add(CapabilitySampledImageArrayDynamicIndexing,
      "SampledImageArrayDynamicIndexing");
  add(CapabilityStorageBufferArrayDynamicIndexing,
      "StorageBufferArrayDynamicIndexing");
  add(CapabilityStorageImageArrayDynamicIndexing,
      "StorageImageArrayDynamicIndexing");
  add(CapabilityClipDistance, "ClipDistance");
  add(CapabilityCullDistance, "CullDistance");
  add(CapabilityImageCubeArray, "ImageCubeArray");
  add(CapabilitySampleRateShading, "SampleRateShading");
  add(CapabilityImageRect, "ImageRect");
  add(CapabilitySampledRect, "SampledRect");
  add(CapabilityGenericPointer, "GenericPointer");
  add(CapabilityInt8, "Int8");
  add(CapabilityInputAttachment, "InputAttachment");
  add(CapabilitySparseResidency, "SparseResidency");
  add(CapabilityMinLod, "MinLod");
  add(CapabilitySampled1D, "Sampled1D");
  add(CapabilityImage1D, "Image1D");
  add(CapabilitySampledCubeArray, "SampledCubeArray");
  add(CapabilitySampledBuffer, "SampledBuffer");
  add(CapabilityImageBuffer, "ImageBuffer");
  add(CapabilityImageMSArray, "ImageMSArray");
  add(CapabilityStorageImageExtendedFormats, "StorageImageExtendedFormats");
  add(CapabilityImageQuery, "ImageQuery");
  add(CapabilityDerivativeControl, "DerivativeControl");
  add(CapabilityInterpolationFunction, "InterpolationFunction");
  add(CapabilityTransformFeedback, "TransformFeedback");
  add(CapabilityGeometryStreams, "GeometryStreams");
  add(CapabilityStorageImageReadWithoutFormat, "StorageImageReadWithoutFormat");
  add(CapabilityStorageImageWriteWithoutFormat,
      "StorageImageWriteWithoutFormat");
  add(CapabilityMultiViewport, "MultiViewport");
  add(CapabilitySubgroupShuffleINTEL, "SubgroupShuffleINTEL");
  add(CapabilitySubgroupBufferBlockIOINTEL, "SubgroupBufferBlockIOINTEL");
  add(CapabilitySubgroupImageBlockIOINTEL, "SubgroupImageBlockIOINTEL");
  add(CapabilitySubgroupImageMediaBlockIOINTEL,
      "SubgroupImageMediaBlockIOINTEL");
  add(CapabilitySubgroupAvcMotionEstimationINTEL,
      "SubgroupAvcMotionEstimationINTEL");
  add(CapabilitySubgroupAvcMotionEstimationIntraINTEL,
      "SubgroupAvcMotionEstimationIntraINTEL");
  add(CapabilitySubgroupAvcMotionEstimationChromaINTEL,
      "SubgroupAvcMotionEstimationChromaINTEL");
  add(CapabilityFPGAMemoryAttributesINTEL, "FPGAMemoryAttributesINTEL");
  add(CapabilityFPGALoopControlsINTEL, "FPGALoopControlsINTEL");
}
SPIRV_DEF_NAMEMAP(Capability, SPIRVCapabilityNameMap)

} /* namespace SPIRV */

#endif // SPIRV_LIBSPIRV_SPIRVNAMEMAPENUM_H
