//===- SPIRVIsValidEnum.h - SPIR-V isValid enums ----------------*- C++ -*-===//
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
/// This file defines SPIR-V isValid enums.
///
//===----------------------------------------------------------------------===//
// WARNING:
//
// This file has been generated using `tools/spirv-tool/gen_spirv.bash` and
// should not be modified manually. If the file needs to be updated, edit the
// script and any other source file instead, before re-generating this file.
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVISVALIDENUM_H
#define SPIRV_LIBSPIRV_SPIRVISVALIDENUM_H

#include "SPIRVEnum.h"
#include "spirv.hpp"

using namespace spv;

namespace SPIRV {

inline bool isValid(spv::SourceLanguage V) {
  switch (V) {
  case SourceLanguageUnknown:
  case SourceLanguageESSL:
  case SourceLanguageGLSL:
  case SourceLanguageOpenCL_C:
  case SourceLanguageOpenCL_CPP:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ExecutionModel V) {
  switch (V) {
  case ExecutionModelVertex:
  case ExecutionModelTessellationControl:
  case ExecutionModelTessellationEvaluation:
  case ExecutionModelGeometry:
  case ExecutionModelFragment:
  case ExecutionModelGLCompute:
  case ExecutionModelKernel:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::AddressingModel V) {
  switch (V) {
  case AddressingModelLogical:
  case AddressingModelPhysical32:
  case AddressingModelPhysical64:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::MemoryModel V) {
  switch (V) {
  case MemoryModelSimple:
  case MemoryModelGLSL450:
  case MemoryModelOpenCL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ExecutionMode V) {
  switch (V) {
  case ExecutionModeInvocations:
  case ExecutionModeSpacingEqual:
  case ExecutionModeSpacingFractionalEven:
  case ExecutionModeSpacingFractionalOdd:
  case ExecutionModeVertexOrderCw:
  case ExecutionModeVertexOrderCcw:
  case ExecutionModePixelCenterInteger:
  case ExecutionModeOriginUpperLeft:
  case ExecutionModeOriginLowerLeft:
  case ExecutionModeEarlyFragmentTests:
  case ExecutionModePointMode:
  case ExecutionModeXfb:
  case ExecutionModeDepthReplacing:
  case ExecutionModeDepthGreater:
  case ExecutionModeDepthLess:
  case ExecutionModeDepthUnchanged:
  case ExecutionModeLocalSize:
  case ExecutionModeLocalSizeHint:
  case ExecutionModeInputPoints:
  case ExecutionModeInputLines:
  case ExecutionModeInputLinesAdjacency:
  case ExecutionModeTriangles:
  case ExecutionModeInputTrianglesAdjacency:
  case ExecutionModeQuads:
  case ExecutionModeIsolines:
  case ExecutionModeOutputVertices:
  case ExecutionModeOutputPoints:
  case ExecutionModeOutputLineStrip:
  case ExecutionModeOutputTriangleStrip:
  case ExecutionModeVecTypeHint:
  case ExecutionModeContractionOff:
  case ExecutionModeInitializer:
  case ExecutionModeFinalizer:
  case ExecutionModeSubgroupSize:
  case ExecutionModeSubgroupsPerWorkgroup:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::StorageClass V) {
  switch (V) {
  case StorageClassUniformConstant:
  case StorageClassInput:
  case StorageClassUniform:
  case StorageClassOutput:
  case StorageClassWorkgroup:
  case StorageClassCrossWorkgroup:
  case StorageClassPrivate:
  case StorageClassFunction:
  case StorageClassGeneric:
  case StorageClassPushConstant:
  case StorageClassAtomicCounter:
  case StorageClassImage:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Dim V) {
  switch (V) {
  case Dim1D:
  case Dim2D:
  case Dim3D:
  case DimCube:
  case DimRect:
  case DimBuffer:
  case DimSubpassData:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::SamplerAddressingMode V) {
  switch (V) {
  case SamplerAddressingModeNone:
  case SamplerAddressingModeClampToEdge:
  case SamplerAddressingModeClamp:
  case SamplerAddressingModeRepeat:
  case SamplerAddressingModeRepeatMirrored:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::SamplerFilterMode V) {
  switch (V) {
  case SamplerFilterModeNearest:
  case SamplerFilterModeLinear:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ImageFormat V) {
  switch (V) {
  case ImageFormatUnknown:
  case ImageFormatRgba32f:
  case ImageFormatRgba16f:
  case ImageFormatR32f:
  case ImageFormatRgba8:
  case ImageFormatRgba8Snorm:
  case ImageFormatRg32f:
  case ImageFormatRg16f:
  case ImageFormatR11fG11fB10f:
  case ImageFormatR16f:
  case ImageFormatRgba16:
  case ImageFormatRgb10A2:
  case ImageFormatRg16:
  case ImageFormatRg8:
  case ImageFormatR16:
  case ImageFormatR8:
  case ImageFormatRgba16Snorm:
  case ImageFormatRg16Snorm:
  case ImageFormatRg8Snorm:
  case ImageFormatR16Snorm:
  case ImageFormatR8Snorm:
  case ImageFormatRgba32i:
  case ImageFormatRgba16i:
  case ImageFormatRgba8i:
  case ImageFormatR32i:
  case ImageFormatRg32i:
  case ImageFormatRg16i:
  case ImageFormatRg8i:
  case ImageFormatR16i:
  case ImageFormatR8i:
  case ImageFormatRgba32ui:
  case ImageFormatRgba16ui:
  case ImageFormatRgba8ui:
  case ImageFormatR32ui:
  case ImageFormatRgb10a2ui:
  case ImageFormatRg32ui:
  case ImageFormatRg16ui:
  case ImageFormatRg8ui:
  case ImageFormatR16ui:
  case ImageFormatR8ui:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ImageChannelOrder V) {
  switch (V) {
  case ImageChannelOrderR:
  case ImageChannelOrderA:
  case ImageChannelOrderRG:
  case ImageChannelOrderRA:
  case ImageChannelOrderRGB:
  case ImageChannelOrderRGBA:
  case ImageChannelOrderBGRA:
  case ImageChannelOrderARGB:
  case ImageChannelOrderIntensity:
  case ImageChannelOrderLuminance:
  case ImageChannelOrderRx:
  case ImageChannelOrderRGx:
  case ImageChannelOrderRGBx:
  case ImageChannelOrderDepth:
  case ImageChannelOrderDepthStencil:
  case ImageChannelOrderABGR:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ImageChannelDataType V) {
  switch (V) {
  case ImageChannelDataTypeSnormInt8:
  case ImageChannelDataTypeSnormInt16:
  case ImageChannelDataTypeUnormInt8:
  case ImageChannelDataTypeUnormInt16:
  case ImageChannelDataTypeUnormShort565:
  case ImageChannelDataTypeUnormShort555:
  case ImageChannelDataTypeUnormInt101010:
  case ImageChannelDataTypeSignedInt8:
  case ImageChannelDataTypeSignedInt16:
  case ImageChannelDataTypeSignedInt32:
  case ImageChannelDataTypeUnsignedInt8:
  case ImageChannelDataTypeUnsignedInt16:
  case ImageChannelDataTypeUnsignedInt32:
  case ImageChannelDataTypeHalfFloat:
  case ImageChannelDataTypeFloat:
  case ImageChannelDataTypeUnormInt24:
  case ImageChannelDataTypeUnormInt101010_2:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::FPRoundingMode V) {
  switch (V) {
  case FPRoundingModeRTE:
  case FPRoundingModeRTZ:
  case FPRoundingModeRTP:
  case FPRoundingModeRTN:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::LinkageType V) {
  switch (V) {
  case LinkageTypeExport:
  case LinkageTypeImport:
  case LinkageTypeInternal:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::AccessQualifier V) {
  switch (V) {
  case AccessQualifierReadOnly:
  case AccessQualifierWriteOnly:
  case AccessQualifierReadWrite:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::FunctionParameterAttribute V) {
  switch (V) {
  case FunctionParameterAttributeZext:
  case FunctionParameterAttributeSext:
  case FunctionParameterAttributeByVal:
  case FunctionParameterAttributeSret:
  case FunctionParameterAttributeNoAlias:
  case FunctionParameterAttributeNoCapture:
  case FunctionParameterAttributeNoWrite:
  case FunctionParameterAttributeNoReadWrite:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Decoration V) {
  switch (V) {
  case DecorationRelaxedPrecision:
  case DecorationSpecId:
  case DecorationBlock:
  case DecorationBufferBlock:
  case DecorationRowMajor:
  case DecorationColMajor:
  case DecorationArrayStride:
  case DecorationMatrixStride:
  case DecorationGLSLShared:
  case DecorationGLSLPacked:
  case DecorationCPacked:
  case DecorationBuiltIn:
  case DecorationNoPerspective:
  case DecorationFlat:
  case DecorationPatch:
  case DecorationCentroid:
  case DecorationSample:
  case DecorationInvariant:
  case DecorationRestrict:
  case DecorationAliased:
  case DecorationVolatile:
  case DecorationConstant:
  case DecorationCoherent:
  case DecorationNonWritable:
  case DecorationNonReadable:
  case DecorationUniform:
  case DecorationSaturatedConversion:
  case DecorationStream:
  case DecorationLocation:
  case DecorationComponent:
  case DecorationIndex:
  case DecorationBinding:
  case DecorationDescriptorSet:
  case DecorationOffset:
  case DecorationXfbBuffer:
  case DecorationXfbStride:
  case DecorationFuncParamAttr:
  case DecorationFPRoundingMode:
  case DecorationFPFastMathMode:
  case DecorationLinkageAttributes:
  case DecorationNoContraction:
  case DecorationInputAttachmentIndex:
  case DecorationAlignment:
  case DecorationMaxByteOffset:
  case DecorationRegisterINTEL:
  case DecorationMemoryINTEL:
  case DecorationNumbanksINTEL:
  case DecorationBankwidthINTEL:
  case DecorationMaxconcurrencyINTEL:
  case DecorationSinglepumpINTEL:
  case DecorationDoublepumpINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::BuiltIn V) {
  switch (V) {
  case BuiltInPosition:
  case BuiltInPointSize:
  case BuiltInClipDistance:
  case BuiltInCullDistance:
  case BuiltInVertexId:
  case BuiltInInstanceId:
  case BuiltInPrimitiveId:
  case BuiltInInvocationId:
  case BuiltInLayer:
  case BuiltInViewportIndex:
  case BuiltInTessLevelOuter:
  case BuiltInTessLevelInner:
  case BuiltInTessCoord:
  case BuiltInPatchVertices:
  case BuiltInFragCoord:
  case BuiltInPointCoord:
  case BuiltInFrontFacing:
  case BuiltInSampleId:
  case BuiltInSamplePosition:
  case BuiltInSampleMask:
  case BuiltInFragDepth:
  case BuiltInHelperInvocation:
  case BuiltInNumWorkgroups:
  case BuiltInWorkgroupSize:
  case BuiltInWorkgroupId:
  case BuiltInLocalInvocationId:
  case BuiltInGlobalInvocationId:
  case BuiltInLocalInvocationIndex:
  case BuiltInWorkDim:
  case BuiltInGlobalSize:
  case BuiltInEnqueuedWorkgroupSize:
  case BuiltInGlobalOffset:
  case BuiltInGlobalLinearId:
  case BuiltInSubgroupSize:
  case BuiltInSubgroupMaxSize:
  case BuiltInNumSubgroups:
  case BuiltInNumEnqueuedSubgroups:
  case BuiltInSubgroupId:
  case BuiltInSubgroupLocalInvocationId:
  case BuiltInVertexIndex:
  case BuiltInInstanceIndex:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Scope V) {
  switch (V) {
  case ScopeCrossDevice:
  case ScopeDevice:
  case ScopeWorkgroup:
  case ScopeSubgroup:
  case ScopeInvocation:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::GroupOperation V) {
  switch (V) {
  case GroupOperationReduce:
  case GroupOperationInclusiveScan:
  case GroupOperationExclusiveScan:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::KernelEnqueueFlags V) {
  switch (V) {
  case KernelEnqueueFlagsNoWait:
  case KernelEnqueueFlagsWaitKernel:
  case KernelEnqueueFlagsWaitWorkGroup:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Capability V) {
  switch (V) {
  case CapabilityMatrix:
  case CapabilityShader:
  case CapabilityGeometry:
  case CapabilityTessellation:
  case CapabilityAddresses:
  case CapabilityLinkage:
  case CapabilityKernel:
  case CapabilityVector16:
  case CapabilityFloat16Buffer:
  case CapabilityFloat16:
  case CapabilityFloat64:
  case CapabilityInt64:
  case CapabilityInt64Atomics:
  case CapabilityImageBasic:
  case CapabilityImageReadWrite:
  case CapabilityImageMipmap:
  case CapabilityPipes:
  case CapabilityGroups:
  case CapabilityDeviceEnqueue:
  case CapabilityLiteralSampler:
  case CapabilityAtomicStorage:
  case CapabilityInt16:
  case CapabilityTessellationPointSize:
  case CapabilityGeometryPointSize:
  case CapabilityImageGatherExtended:
  case CapabilityStorageImageMultisample:
  case CapabilityUniformBufferArrayDynamicIndexing:
  case CapabilitySampledImageArrayDynamicIndexing:
  case CapabilityStorageBufferArrayDynamicIndexing:
  case CapabilityStorageImageArrayDynamicIndexing:
  case CapabilityClipDistance:
  case CapabilityCullDistance:
  case CapabilityImageCubeArray:
  case CapabilitySampleRateShading:
  case CapabilityImageRect:
  case CapabilitySampledRect:
  case CapabilityGenericPointer:
  case CapabilityInt8:
  case CapabilityInputAttachment:
  case CapabilitySparseResidency:
  case CapabilityMinLod:
  case CapabilitySampled1D:
  case CapabilityImage1D:
  case CapabilitySampledCubeArray:
  case CapabilitySampledBuffer:
  case CapabilityImageBuffer:
  case CapabilityImageMSArray:
  case CapabilityStorageImageExtendedFormats:
  case CapabilityImageQuery:
  case CapabilityDerivativeControl:
  case CapabilityInterpolationFunction:
  case CapabilityTransformFeedback:
  case CapabilityGeometryStreams:
  case CapabilityStorageImageReadWithoutFormat:
  case CapabilityStorageImageWriteWithoutFormat:
  case CapabilityMultiViewport:
  case CapabilitySubgroupDispatch:
  case CapabilityNamedBarrier:
  case CapabilityPipeStorage:
  case CapabilityFPGAMemoryAttributesINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Op V) {
  switch (V) {
  case OpNop:
  case OpUndef:
  case OpSourceContinued:
  case OpSource:
  case OpSourceExtension:
  case OpName:
  case OpMemberName:
  case OpString:
  case OpLine:
  case OpExtension:
  case OpExtInstImport:
  case OpExtInst:
  case OpMemoryModel:
  case OpEntryPoint:
  case OpExecutionMode:
  case OpCapability:
  case OpTypeVoid:
  case OpTypeBool:
  case OpTypeInt:
  case OpTypeFloat:
  case OpTypeVector:
  case OpTypeMatrix:
  case OpTypeImage:
  case OpTypeSampler:
  case OpTypeSampledImage:
  case OpTypeArray:
  case OpTypeRuntimeArray:
  case OpTypeStruct:
  case OpTypeOpaque:
  case OpTypePointer:
  case OpTypeFunction:
  case OpTypeEvent:
  case OpTypeDeviceEvent:
  case OpTypeReserveId:
  case OpTypeQueue:
  case OpTypePipe:
  case OpTypeForwardPointer:
  case OpConstantTrue:
  case OpConstantFalse:
  case OpConstant:
  case OpConstantComposite:
  case OpConstantSampler:
  case OpConstantNull:
  case OpSpecConstantTrue:
  case OpSpecConstantFalse:
  case OpSpecConstant:
  case OpSpecConstantComposite:
  case OpSpecConstantOp:
  case OpFunction:
  case OpFunctionParameter:
  case OpFunctionEnd:
  case OpFunctionCall:
  case OpVariable:
  case OpImageTexelPointer:
  case OpLoad:
  case OpStore:
  case OpCopyMemory:
  case OpCopyMemorySized:
  case OpAccessChain:
  case OpInBoundsAccessChain:
  case OpPtrAccessChain:
  case OpArrayLength:
  case OpGenericPtrMemSemantics:
  case OpInBoundsPtrAccessChain:
  case OpDecorate:
  case OpMemberDecorate:
  case OpDecorationGroup:
  case OpGroupDecorate:
  case OpGroupMemberDecorate:
  case OpVectorExtractDynamic:
  case OpVectorInsertDynamic:
  case OpVectorShuffle:
  case OpCompositeConstruct:
  case OpCompositeExtract:
  case OpCompositeInsert:
  case OpCopyObject:
  case OpTranspose:
  case OpSampledImage:
  case OpImageSampleImplicitLod:
  case OpImageSampleExplicitLod:
  case OpImageSampleDrefImplicitLod:
  case OpImageSampleDrefExplicitLod:
  case OpImageSampleProjImplicitLod:
  case OpImageSampleProjExplicitLod:
  case OpImageSampleProjDrefImplicitLod:
  case OpImageSampleProjDrefExplicitLod:
  case OpImageFetch:
  case OpImageGather:
  case OpImageDrefGather:
  case OpImageRead:
  case OpImageWrite:
  case OpImage:
  case OpImageQueryFormat:
  case OpImageQueryOrder:
  case OpImageQuerySizeLod:
  case OpImageQuerySize:
  case OpImageQueryLod:
  case OpImageQueryLevels:
  case OpImageQuerySamples:
  case OpConvertFToU:
  case OpConvertFToS:
  case OpConvertSToF:
  case OpConvertUToF:
  case OpUConvert:
  case OpSConvert:
  case OpFConvert:
  case OpQuantizeToF16:
  case OpConvertPtrToU:
  case OpSatConvertSToU:
  case OpSatConvertUToS:
  case OpConvertUToPtr:
  case OpPtrCastToGeneric:
  case OpGenericCastToPtr:
  case OpGenericCastToPtrExplicit:
  case OpBitcast:
  case OpSNegate:
  case OpFNegate:
  case OpIAdd:
  case OpFAdd:
  case OpISub:
  case OpFSub:
  case OpIMul:
  case OpFMul:
  case OpUDiv:
  case OpSDiv:
  case OpFDiv:
  case OpUMod:
  case OpSRem:
  case OpSMod:
  case OpFRem:
  case OpFMod:
  case OpVectorTimesScalar:
  case OpMatrixTimesScalar:
  case OpVectorTimesMatrix:
  case OpMatrixTimesVector:
  case OpMatrixTimesMatrix:
  case OpOuterProduct:
  case OpDot:
  case OpIAddCarry:
  case OpISubBorrow:
  case OpUMulExtended:
  case OpSMulExtended:
  case OpAny:
  case OpAll:
  case OpIsNan:
  case OpIsInf:
  case OpIsFinite:
  case OpIsNormal:
  case OpSignBitSet:
  case OpLessOrGreater:
  case OpOrdered:
  case OpUnordered:
  case OpLogicalEqual:
  case OpLogicalNotEqual:
  case OpLogicalOr:
  case OpLogicalAnd:
  case OpLogicalNot:
  case OpSelect:
  case OpIEqual:
  case OpINotEqual:
  case OpUGreaterThan:
  case OpSGreaterThan:
  case OpUGreaterThanEqual:
  case OpSGreaterThanEqual:
  case OpULessThan:
  case OpSLessThan:
  case OpULessThanEqual:
  case OpSLessThanEqual:
  case OpFOrdEqual:
  case OpFUnordEqual:
  case OpFOrdNotEqual:
  case OpFUnordNotEqual:
  case OpFOrdLessThan:
  case OpFUnordLessThan:
  case OpFOrdGreaterThan:
  case OpFUnordGreaterThan:
  case OpFOrdLessThanEqual:
  case OpFUnordLessThanEqual:
  case OpFOrdGreaterThanEqual:
  case OpFUnordGreaterThanEqual:
  case OpShiftRightLogical:
  case OpShiftRightArithmetic:
  case OpShiftLeftLogical:
  case OpBitwiseOr:
  case OpBitwiseXor:
  case OpBitwiseAnd:
  case OpNot:
  case OpBitFieldInsert:
  case OpBitFieldSExtract:
  case OpBitFieldUExtract:
  case OpBitReverse:
  case OpBitCount:
  case OpDPdx:
  case OpDPdy:
  case OpFwidth:
  case OpDPdxFine:
  case OpDPdyFine:
  case OpFwidthFine:
  case OpDPdxCoarse:
  case OpDPdyCoarse:
  case OpFwidthCoarse:
  case OpEmitVertex:
  case OpEndPrimitive:
  case OpEmitStreamVertex:
  case OpEndStreamPrimitive:
  case OpControlBarrier:
  case OpMemoryBarrier:
  case OpAtomicLoad:
  case OpAtomicStore:
  case OpAtomicExchange:
  case OpAtomicCompareExchange:
  case OpAtomicCompareExchangeWeak:
  case OpAtomicIIncrement:
  case OpAtomicIDecrement:
  case OpAtomicIAdd:
  case OpAtomicISub:
  case OpAtomicSMin:
  case OpAtomicUMin:
  case OpAtomicSMax:
  case OpAtomicUMax:
  case OpAtomicAnd:
  case OpAtomicOr:
  case OpAtomicXor:
  case OpPhi:
  case OpLoopMerge:
  case OpSelectionMerge:
  case OpLabel:
  case OpBranch:
  case OpBranchConditional:
  case OpSwitch:
  case OpKill:
  case OpReturn:
  case OpReturnValue:
  case OpUnreachable:
  case OpLifetimeStart:
  case OpLifetimeStop:
  case OpGroupAsyncCopy:
  case OpGroupWaitEvents:
  case OpGroupAll:
  case OpGroupAny:
  case OpGroupBroadcast:
  case OpGroupIAdd:
  case OpGroupFAdd:
  case OpGroupFMin:
  case OpGroupUMin:
  case OpGroupSMin:
  case OpGroupFMax:
  case OpGroupUMax:
  case OpGroupSMax:
  case OpReadPipe:
  case OpWritePipe:
  case OpReservedReadPipe:
  case OpReservedWritePipe:
  case OpReserveReadPipePackets:
  case OpReserveWritePipePackets:
  case OpCommitReadPipe:
  case OpCommitWritePipe:
  case OpIsValidReserveId:
  case OpGetNumPipePackets:
  case OpGetMaxPipePackets:
  case OpGroupReserveReadPipePackets:
  case OpGroupReserveWritePipePackets:
  case OpGroupCommitReadPipe:
  case OpGroupCommitWritePipe:
  case OpEnqueueMarker:
  case OpEnqueueKernel:
  case OpGetKernelNDrangeSubGroupCount:
  case OpGetKernelNDrangeMaxSubGroupSize:
  case OpGetKernelWorkGroupSize:
  case OpGetKernelPreferredWorkGroupSizeMultiple:
  case OpRetainEvent:
  case OpReleaseEvent:
  case OpCreateUserEvent:
  case OpIsValidEvent:
  case OpSetUserEventStatus:
  case OpCaptureEventProfilingInfo:
  case OpGetDefaultQueue:
  case OpBuildNDRange:
  case OpImageSparseSampleImplicitLod:
  case OpImageSparseSampleExplicitLod:
  case OpImageSparseSampleDrefImplicitLod:
  case OpImageSparseSampleDrefExplicitLod:
  case OpImageSparseSampleProjImplicitLod:
  case OpImageSparseSampleProjExplicitLod:
  case OpImageSparseSampleProjDrefImplicitLod:
  case OpImageSparseSampleProjDrefExplicitLod:
  case OpImageSparseFetch:
  case OpImageSparseGather:
  case OpImageSparseDrefGather:
  case OpImageSparseTexelsResident:
  case OpNoLine:
  case OpAtomicFlagTestAndSet:
  case OpAtomicFlagClear:
  case OpImageSparseRead:
  case OpSizeOf:
  case OpTypePipeStorage:
  case OpConstantPipeStorage:
  case OpCreatePipeFromPipeStorage:
  case OpGetKernelLocalSizeForSubgroupCount:
  case OpGetKernelMaxNumSubgroups:
  case OpTypeNamedBarrier:
  case OpNamedBarrierInitialize:
  case OpMemoryNamedBarrier:
  case OpModuleProcessed:
  case OpForward:
  case OpSubgroupShuffleINTEL:
  case OpSubgroupShuffleDownINTEL:
  case OpSubgroupShuffleUpINTEL:
  case OpSubgroupShuffleXorINTEL:
  case OpSubgroupBlockReadINTEL:
  case OpSubgroupBlockWriteINTEL:
  case OpSubgroupImageBlockReadINTEL:
  case OpSubgroupImageBlockWriteINTEL:
  case OpSubgroupImageMediaBlockReadINTEL:
  case OpSubgroupImageMediaBlockWriteINTEL:
  case OpVmeImageINTEL:
  case OpTypeVmeImageINTEL:
  case OpTypeAvcImePayloadINTEL:
  case OpTypeAvcRefPayloadINTEL:
  case OpTypeAvcSicPayloadINTEL:
  case OpTypeAvcMcePayloadINTEL:
  case OpTypeAvcMceResultINTEL:
  case OpTypeAvcImeResultINTEL:
  case OpTypeAvcImeResultSingleReferenceStreamoutINTEL:
  case OpTypeAvcImeResultDualReferenceStreamoutINTEL:
  case OpTypeAvcImeSingleReferenceStreaminINTEL:
  case OpTypeAvcImeDualReferenceStreaminINTEL:
  case OpTypeAvcRefResultINTEL:
  case OpTypeAvcSicResultINTEL:
  case OpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL:
  case OpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL:
  case OpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL:
  case OpSubgroupAvcMceSetInterShapePenaltyINTEL:
  case OpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL:
  case OpSubgroupAvcMceSetInterDirectionPenaltyINTEL:
  case OpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL:
  case OpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL:
  case OpSubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL:
  case OpSubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL:
  case OpSubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL:
  case OpSubgroupAvcMceSetMotionVectorCostFunctionINTEL:
  case OpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL:
  case OpSubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL:
  case OpSubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL:
  case OpSubgroupAvcMceSetAcOnlyHaarINTEL:
  case OpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL:
  case OpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL:
  case OpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL:
  case OpSubgroupAvcMceConvertToImePayloadINTEL:
  case OpSubgroupAvcMceConvertToImeResultINTEL:
  case OpSubgroupAvcMceConvertToRefPayloadINTEL:
  case OpSubgroupAvcMceConvertToRefResultINTEL:
  case OpSubgroupAvcMceConvertToSicPayloadINTEL:
  case OpSubgroupAvcMceConvertToSicResultINTEL:
  case OpSubgroupAvcMceGetMotionVectorsINTEL:
  case OpSubgroupAvcMceGetInterDistortionsINTEL:
  case OpSubgroupAvcMceGetBestInterDistortionsINTEL:
  case OpSubgroupAvcMceGetInterMajorShapeINTEL:
  case OpSubgroupAvcMceGetInterMinorShapeINTEL:
  case OpSubgroupAvcMceGetInterDirectionsINTEL:
  case OpSubgroupAvcMceGetInterMotionVectorCountINTEL:
  case OpSubgroupAvcMceGetInterReferenceIdsINTEL:
  case OpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL:
  case OpSubgroupAvcImeInitializeINTEL:
  case OpSubgroupAvcImeSetSingleReferenceINTEL:
  case OpSubgroupAvcImeSetDualReferenceINTEL:
  case OpSubgroupAvcImeRefWindowSizeINTEL:
  case OpSubgroupAvcImeAdjustRefOffsetINTEL:
  case OpSubgroupAvcImeConvertToMcePayloadINTEL:
  case OpSubgroupAvcImeSetMaxMotionVectorCountINTEL:
  case OpSubgroupAvcImeSetUnidirectionalMixDisableINTEL:
  case OpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL:
  case OpSubgroupAvcImeSetWeightedSadINTEL:
  case OpSubgroupAvcImeEvaluateWithSingleReferenceINTEL:
  case OpSubgroupAvcImeEvaluateWithDualReferenceINTEL:
  case OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL:
  case OpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL:
  case OpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL:
  case OpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL:
  case OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL:
  case OpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL:
  case OpSubgroupAvcImeConvertToMceResultINTEL:
  case OpSubgroupAvcImeGetSingleReferenceStreaminINTEL:
  case OpSubgroupAvcImeGetDualReferenceStreaminINTEL:
  case OpSubgroupAvcImeStripSingleReferenceStreamoutINTEL:
  case OpSubgroupAvcImeStripDualReferenceStreamoutINTEL:
  case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL:
  case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL:
  case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL:
  case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL:
  case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL:
  case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL:
  case OpSubgroupAvcImeGetBorderReachedINTEL:
  case OpSubgroupAvcImeGetTruncatedSearchIndicationINTEL:
  case OpSubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL:
  case OpSubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL:
  case OpSubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL:
  case OpSubgroupAvcFmeInitializeINTEL:
  case OpSubgroupAvcBmeInitializeINTEL:
  case OpSubgroupAvcRefConvertToMcePayloadINTEL:
  case OpSubgroupAvcRefSetBidirectionalMixDisableINTEL:
  case OpSubgroupAvcRefSetBilinearFilterEnableINTEL:
  case OpSubgroupAvcRefEvaluateWithSingleReferenceINTEL:
  case OpSubgroupAvcRefEvaluateWithDualReferenceINTEL:
  case OpSubgroupAvcRefEvaluateWithMultiReferenceINTEL:
  case OpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL:
  case OpSubgroupAvcRefConvertToMceResultINTEL:
  case OpSubgroupAvcSicInitializeINTEL:
  case OpSubgroupAvcSicConfigureSkcINTEL:
  case OpSubgroupAvcSicConfigureIpeLumaINTEL:
  case OpSubgroupAvcSicConfigureIpeLumaChromaINTEL:
  case OpSubgroupAvcSicGetMotionVectorMaskINTEL:
  case OpSubgroupAvcSicConvertToMcePayloadINTEL:
  case OpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL:
  case OpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL:
  case OpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL:
  case OpSubgroupAvcSicSetBilinearFilterEnableINTEL:
  case OpSubgroupAvcSicSetSkcForwardTransformEnableINTEL:
  case OpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL:
  case OpSubgroupAvcSicEvaluateIpeINTEL:
  case OpSubgroupAvcSicEvaluateWithSingleReferenceINTEL:
  case OpSubgroupAvcSicEvaluateWithDualReferenceINTEL:
  case OpSubgroupAvcSicEvaluateWithMultiReferenceINTEL:
  case OpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL:
  case OpSubgroupAvcSicConvertToMceResultINTEL:
  case OpSubgroupAvcSicGetIpeLumaShapeINTEL:
  case OpSubgroupAvcSicGetBestIpeLumaDistortionINTEL:
  case OpSubgroupAvcSicGetBestIpeChromaDistortionINTEL:
  case OpSubgroupAvcSicGetPackedIpeLumaModesINTEL:
  case OpSubgroupAvcSicGetIpeChromaModeINTEL:
  case OpSubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL:
  case OpSubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL:
  case OpSubgroupAvcSicGetInterRawSadsINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValidImageOperandsMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= ImageOperandsBiasMask;
  ValidMask |= ImageOperandsLodMask;
  ValidMask |= ImageOperandsGradMask;
  ValidMask |= ImageOperandsConstOffsetMask;
  ValidMask |= ImageOperandsOffsetMask;
  ValidMask |= ImageOperandsConstOffsetsMask;
  ValidMask |= ImageOperandsSampleMask;
  ValidMask |= ImageOperandsMinLodMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidFPFastMathModeMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= FPFastMathModeNotNaNMask;
  ValidMask |= FPFastMathModeNotInfMask;
  ValidMask |= FPFastMathModeNSZMask;
  ValidMask |= FPFastMathModeAllowRecipMask;
  ValidMask |= FPFastMathModeFastMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidSelectionControlMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= SelectionControlFlattenMask;
  ValidMask |= SelectionControlDontFlattenMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidLoopControlMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= LoopControlUnrollMask;
  ValidMask |= LoopControlDontUnrollMask;
  ValidMask |= LoopControlDependencyInfiniteMask;
  ValidMask |= LoopControlDependencyLengthMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidFunctionControlMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= FunctionControlInlineMask;
  ValidMask |= FunctionControlDontInlineMask;
  ValidMask |= FunctionControlPureMask;
  ValidMask |= FunctionControlConstMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidMemorySemanticsMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= MemorySemanticsAcquireMask;
  ValidMask |= MemorySemanticsReleaseMask;
  ValidMask |= MemorySemanticsAcquireReleaseMask;
  ValidMask |= MemorySemanticsSequentiallyConsistentMask;
  ValidMask |= MemorySemanticsUniformMemoryMask;
  ValidMask |= MemorySemanticsSubgroupMemoryMask;
  ValidMask |= MemorySemanticsWorkgroupMemoryMask;
  ValidMask |= MemorySemanticsCrossWorkgroupMemoryMask;
  ValidMask |= MemorySemanticsAtomicCounterMemoryMask;
  ValidMask |= MemorySemanticsImageMemoryMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidMemoryAccessMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= MemoryAccessVolatileMask;
  ValidMask |= MemoryAccessAlignedMask;
  ValidMask |= MemoryAccessNontemporalMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidKernelProfilingInfoMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= KernelProfilingInfoCmdExecTimeMask;

  return (Mask & ~ValidMask) == 0;
}

} /* namespace SPIRV */

#endif // SPIRV_LIBSPIRV_SPIRVISVALIDENUM_H
