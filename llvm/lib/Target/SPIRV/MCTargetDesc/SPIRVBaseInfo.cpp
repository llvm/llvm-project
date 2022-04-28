//===-- SPIRVBaseInfo.cpp -  Top level definitions for SPIRV ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the SPIRV target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#include "SPIRVBaseInfo.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace SPIRV {

#define CASE(CLASS, ATTR)                                                      \
  case CLASS::ATTR:                                                            \
    return #ATTR;
#define CASE_SUF(CLASS, SF, ATTR)                                              \
  case CLASS::SF##_##ATTR:                                                     \
    return #ATTR;

// Implement getEnumName(Enum e) helper functions.
// TODO: re-implement all the functions using TableGen.
StringRef getCapabilityName(Capability e) {
  switch (e) {
    CASE(Capability, Matrix)
    CASE(Capability, Shader)
    CASE(Capability, Geometry)
    CASE(Capability, Tessellation)
    CASE(Capability, Addresses)
    CASE(Capability, Linkage)
    CASE(Capability, Kernel)
    CASE(Capability, Vector16)
    CASE(Capability, Float16Buffer)
    CASE(Capability, Float16)
    CASE(Capability, Float64)
    CASE(Capability, Int64)
    CASE(Capability, Int64Atomics)
    CASE(Capability, ImageBasic)
    CASE(Capability, ImageReadWrite)
    CASE(Capability, ImageMipmap)
    CASE(Capability, Pipes)
    CASE(Capability, Groups)
    CASE(Capability, DeviceEnqueue)
    CASE(Capability, LiteralSampler)
    CASE(Capability, AtomicStorage)
    CASE(Capability, Int16)
    CASE(Capability, TessellationPointSize)
    CASE(Capability, GeometryPointSize)
    CASE(Capability, ImageGatherExtended)
    CASE(Capability, StorageImageMultisample)
    CASE(Capability, UniformBufferArrayDynamicIndexing)
    CASE(Capability, SampledImageArrayDymnamicIndexing)
    CASE(Capability, ClipDistance)
    CASE(Capability, CullDistance)
    CASE(Capability, ImageCubeArray)
    CASE(Capability, SampleRateShading)
    CASE(Capability, ImageRect)
    CASE(Capability, SampledRect)
    CASE(Capability, GenericPointer)
    CASE(Capability, Int8)
    CASE(Capability, InputAttachment)
    CASE(Capability, SparseResidency)
    CASE(Capability, MinLod)
    CASE(Capability, Sampled1D)
    CASE(Capability, Image1D)
    CASE(Capability, SampledCubeArray)
    CASE(Capability, SampledBuffer)
    CASE(Capability, ImageBuffer)
    CASE(Capability, ImageMSArray)
    CASE(Capability, StorageImageExtendedFormats)
    CASE(Capability, ImageQuery)
    CASE(Capability, DerivativeControl)
    CASE(Capability, InterpolationFunction)
    CASE(Capability, TransformFeedback)
    CASE(Capability, GeometryStreams)
    CASE(Capability, StorageImageReadWithoutFormat)
    CASE(Capability, StorageImageWriteWithoutFormat)
    CASE(Capability, MultiViewport)
    CASE(Capability, SubgroupDispatch)
    CASE(Capability, NamedBarrier)
    CASE(Capability, PipeStorage)
    CASE(Capability, GroupNonUniform)
    CASE(Capability, GroupNonUniformVote)
    CASE(Capability, GroupNonUniformArithmetic)
    CASE(Capability, GroupNonUniformBallot)
    CASE(Capability, GroupNonUniformShuffle)
    CASE(Capability, GroupNonUniformShuffleRelative)
    CASE(Capability, GroupNonUniformClustered)
    CASE(Capability, GroupNonUniformQuad)
    CASE(Capability, SubgroupBallotKHR)
    CASE(Capability, DrawParameters)
    CASE(Capability, SubgroupVoteKHR)
    CASE(Capability, StorageBuffer16BitAccess)
    CASE(Capability, StorageUniform16)
    CASE(Capability, StoragePushConstant16)
    CASE(Capability, StorageInputOutput16)
    CASE(Capability, DeviceGroup)
    CASE(Capability, MultiView)
    CASE(Capability, VariablePointersStorageBuffer)
    CASE(Capability, VariablePointers)
    CASE(Capability, AtomicStorageOps)
    CASE(Capability, SampleMaskPostDepthCoverage)
    CASE(Capability, StorageBuffer8BitAccess)
    CASE(Capability, UniformAndStorageBuffer8BitAccess)
    CASE(Capability, StoragePushConstant8)
    CASE(Capability, DenormPreserve)
    CASE(Capability, DenormFlushToZero)
    CASE(Capability, SignedZeroInfNanPreserve)
    CASE(Capability, RoundingModeRTE)
    CASE(Capability, RoundingModeRTZ)
    CASE(Capability, Float16ImageAMD)
    CASE(Capability, ImageGatherBiasLodAMD)
    CASE(Capability, FragmentMaskAMD)
    CASE(Capability, StencilExportEXT)
    CASE(Capability, ImageReadWriteLodAMD)
    CASE(Capability, SampleMaskOverrideCoverageNV)
    CASE(Capability, GeometryShaderPassthroughNV)
    CASE(Capability, ShaderViewportIndexLayerEXT)
    CASE(Capability, ShaderViewportMaskNV)
    CASE(Capability, ShaderStereoViewNV)
    CASE(Capability, PerViewAttributesNV)
    CASE(Capability, FragmentFullyCoveredEXT)
    CASE(Capability, MeshShadingNV)
    CASE(Capability, ShaderNonUniformEXT)
    CASE(Capability, RuntimeDescriptorArrayEXT)
    CASE(Capability, InputAttachmentArrayDynamicIndexingEXT)
    CASE(Capability, UniformTexelBufferArrayDynamicIndexingEXT)
    CASE(Capability, StorageTexelBufferArrayDynamicIndexingEXT)
    CASE(Capability, UniformBufferArrayNonUniformIndexingEXT)
    CASE(Capability, SampledImageArrayNonUniformIndexingEXT)
    CASE(Capability, StorageBufferArrayNonUniformIndexingEXT)
    CASE(Capability, StorageImageArrayNonUniformIndexingEXT)
    CASE(Capability, InputAttachmentArrayNonUniformIndexingEXT)
    CASE(Capability, UniformTexelBufferArrayNonUniformIndexingEXT)
    CASE(Capability, StorageTexelBufferArrayNonUniformIndexingEXT)
    CASE(Capability, RayTracingNV)
    CASE(Capability, SubgroupShuffleINTEL)
    CASE(Capability, SubgroupBufferBlockIOINTEL)
    CASE(Capability, SubgroupImageBlockIOINTEL)
    CASE(Capability, SubgroupImageMediaBlockIOINTEL)
    CASE(Capability, SubgroupAvcMotionEstimationINTEL)
    CASE(Capability, SubgroupAvcMotionEstimationIntraINTEL)
    CASE(Capability, SubgroupAvcMotionEstimationChromaINTEL)
    CASE(Capability, GroupNonUniformPartitionedNV)
    CASE(Capability, VulkanMemoryModelKHR)
    CASE(Capability, VulkanMemoryModelDeviceScopeKHR)
    CASE(Capability, ImageFootprintNV)
    CASE(Capability, FragmentBarycentricNV)
    CASE(Capability, ComputeDerivativeGroupQuadsNV)
    CASE(Capability, ComputeDerivativeGroupLinearNV)
    CASE(Capability, FragmentDensityEXT)
    CASE(Capability, PhysicalStorageBufferAddressesEXT)
    CASE(Capability, CooperativeMatrixNV)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getSourceLanguageName(SourceLanguage e) {
  switch (e) {
    CASE(SourceLanguage, Unknown)
    CASE(SourceLanguage, ESSL)
    CASE(SourceLanguage, GLSL)
    CASE(SourceLanguage, OpenCL_C)
    CASE(SourceLanguage, OpenCL_CPP)
    CASE(SourceLanguage, HLSL)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getExecutionModelName(ExecutionModel e) {
  switch (e) {
    CASE(ExecutionModel, Vertex)
    CASE(ExecutionModel, TessellationControl)
    CASE(ExecutionModel, TessellationEvaluation)
    CASE(ExecutionModel, Geometry)
    CASE(ExecutionModel, Fragment)
    CASE(ExecutionModel, GLCompute)
    CASE(ExecutionModel, Kernel)
    CASE(ExecutionModel, TaskNV)
    CASE(ExecutionModel, MeshNV)
    CASE(ExecutionModel, RayGenerationNV)
    CASE(ExecutionModel, IntersectionNV)
    CASE(ExecutionModel, AnyHitNV)
    CASE(ExecutionModel, ClosestHitNV)
    CASE(ExecutionModel, MissNV)
    CASE(ExecutionModel, CallableNV)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getAddressingModelName(AddressingModel e) {
  switch (e) {
    CASE(AddressingModel, Logical)
    CASE(AddressingModel, Physical32)
    CASE(AddressingModel, Physical64)
    CASE(AddressingModel, PhysicalStorageBuffer64EXT)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getMemoryModelName(MemoryModel e) {
  switch (e) {
    CASE(MemoryModel, Simple)
    CASE(MemoryModel, GLSL450)
    CASE(MemoryModel, OpenCL)
    CASE(MemoryModel, VulkanKHR)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getExecutionModeName(ExecutionMode e) {
  switch (e) {
    CASE(ExecutionMode, Invocations)
    CASE(ExecutionMode, SpacingEqual)
    CASE(ExecutionMode, SpacingFractionalEven)
    CASE(ExecutionMode, SpacingFractionalOdd)
    CASE(ExecutionMode, VertexOrderCw)
    CASE(ExecutionMode, VertexOrderCcw)
    CASE(ExecutionMode, PixelCenterInteger)
    CASE(ExecutionMode, OriginUpperLeft)
    CASE(ExecutionMode, OriginLowerLeft)
    CASE(ExecutionMode, EarlyFragmentTests)
    CASE(ExecutionMode, PointMode)
    CASE(ExecutionMode, Xfb)
    CASE(ExecutionMode, DepthReplacing)
    CASE(ExecutionMode, DepthGreater)
    CASE(ExecutionMode, DepthLess)
    CASE(ExecutionMode, DepthUnchanged)
    CASE(ExecutionMode, LocalSize)
    CASE(ExecutionMode, LocalSizeHint)
    CASE(ExecutionMode, InputPoints)
    CASE(ExecutionMode, InputLines)
    CASE(ExecutionMode, InputLinesAdjacency)
    CASE(ExecutionMode, Triangles)
    CASE(ExecutionMode, InputTrianglesAdjacency)
    CASE(ExecutionMode, Quads)
    CASE(ExecutionMode, Isolines)
    CASE(ExecutionMode, OutputVertices)
    CASE(ExecutionMode, OutputPoints)
    CASE(ExecutionMode, OutputLineStrip)
    CASE(ExecutionMode, OutputTriangleStrip)
    CASE(ExecutionMode, VecTypeHint)
    CASE(ExecutionMode, ContractionOff)
    CASE(ExecutionMode, Initializer)
    CASE(ExecutionMode, Finalizer)
    CASE(ExecutionMode, SubgroupSize)
    CASE(ExecutionMode, SubgroupsPerWorkgroup)
    CASE(ExecutionMode, SubgroupsPerWorkgroupId)
    CASE(ExecutionMode, LocalSizeId)
    CASE(ExecutionMode, LocalSizeHintId)
    CASE(ExecutionMode, PostDepthCoverage)
    CASE(ExecutionMode, DenormPreserve)
    CASE(ExecutionMode, DenormFlushToZero)
    CASE(ExecutionMode, SignedZeroInfNanPreserve)
    CASE(ExecutionMode, RoundingModeRTE)
    CASE(ExecutionMode, RoundingModeRTZ)
    CASE(ExecutionMode, StencilRefReplacingEXT)
    CASE(ExecutionMode, OutputLinesNV)
    CASE(ExecutionMode, DerivativeGroupQuadsNV)
    CASE(ExecutionMode, DerivativeGroupLinearNV)
    CASE(ExecutionMode, OutputTrianglesNV)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getStorageClassName(StorageClass e) {
  switch (e) {
    CASE(StorageClass, UniformConstant)
    CASE(StorageClass, Input)
    CASE(StorageClass, Uniform)
    CASE(StorageClass, Output)
    CASE(StorageClass, Workgroup)
    CASE(StorageClass, CrossWorkgroup)
    CASE(StorageClass, Private)
    CASE(StorageClass, Function)
    CASE(StorageClass, Generic)
    CASE(StorageClass, PushConstant)
    CASE(StorageClass, AtomicCounter)
    CASE(StorageClass, Image)
    CASE(StorageClass, StorageBuffer)
    CASE(StorageClass, CallableDataNV)
    CASE(StorageClass, IncomingCallableDataNV)
    CASE(StorageClass, RayPayloadNV)
    CASE(StorageClass, HitAttributeNV)
    CASE(StorageClass, IncomingRayPayloadNV)
    CASE(StorageClass, ShaderRecordBufferNV)
    CASE(StorageClass, PhysicalStorageBufferEXT)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getDimName(Dim dim) {
  switch (dim) {
    CASE_SUF(Dim, DIM, 1D)
    CASE_SUF(Dim, DIM, 2D)
    CASE_SUF(Dim, DIM, 3D)
    CASE_SUF(Dim, DIM, Cube)
    CASE_SUF(Dim, DIM, Rect)
    CASE_SUF(Dim, DIM, Buffer)
    CASE_SUF(Dim, DIM, SubpassData)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getSamplerAddressingModeName(SamplerAddressingMode e) {
  switch (e) {
    CASE(SamplerAddressingMode, None)
    CASE(SamplerAddressingMode, ClampToEdge)
    CASE(SamplerAddressingMode, Clamp)
    CASE(SamplerAddressingMode, Repeat)
    CASE(SamplerAddressingMode, RepeatMirrored)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getSamplerFilterModeName(SamplerFilterMode e) {
  switch (e) {
    CASE(SamplerFilterMode, Nearest)
    CASE(SamplerFilterMode, Linear)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getImageFormatName(ImageFormat e) {
  switch (e) {
    CASE(ImageFormat, Unknown)
    CASE(ImageFormat, Rgba32f)
    CASE(ImageFormat, Rgba16f)
    CASE(ImageFormat, R32f)
    CASE(ImageFormat, Rgba8)
    CASE(ImageFormat, Rgba8Snorm)
    CASE(ImageFormat, Rg32f)
    CASE(ImageFormat, Rg16f)
    CASE(ImageFormat, R11fG11fB10f)
    CASE(ImageFormat, R16f)
    CASE(ImageFormat, Rgba16)
    CASE(ImageFormat, Rgb10A2)
    CASE(ImageFormat, Rg16)
    CASE(ImageFormat, Rg8)
    CASE(ImageFormat, R16)
    CASE(ImageFormat, R8)
    CASE(ImageFormat, Rgba16Snorm)
    CASE(ImageFormat, Rg16Snorm)
    CASE(ImageFormat, Rg8Snorm)
    CASE(ImageFormat, R16Snorm)
    CASE(ImageFormat, R8Snorm)
    CASE(ImageFormat, Rgba32i)
    CASE(ImageFormat, Rgba16i)
    CASE(ImageFormat, Rgba8i)
    CASE(ImageFormat, R32i)
    CASE(ImageFormat, Rg32i)
    CASE(ImageFormat, Rg16i)
    CASE(ImageFormat, Rg8i)
    CASE(ImageFormat, R16i)
    CASE(ImageFormat, R8i)
    CASE(ImageFormat, Rgba32ui)
    CASE(ImageFormat, Rgba16ui)
    CASE(ImageFormat, Rgba8ui)
    CASE(ImageFormat, R32ui)
    CASE(ImageFormat, Rgb10a2ui)
    CASE(ImageFormat, Rg32ui)
    CASE(ImageFormat, Rg16ui)
    CASE(ImageFormat, Rg8ui)
    CASE(ImageFormat, R16ui)
    CASE(ImageFormat, R8ui)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getImageChannelOrderName(ImageChannelOrder e) {
  switch (e) {
    CASE(ImageChannelOrder, R)
    CASE(ImageChannelOrder, A)
    CASE(ImageChannelOrder, RG)
    CASE(ImageChannelOrder, RA)
    CASE(ImageChannelOrder, RGB)
    CASE(ImageChannelOrder, RGBA)
    CASE(ImageChannelOrder, BGRA)
    CASE(ImageChannelOrder, ARGB)
    CASE(ImageChannelOrder, Intensity)
    CASE(ImageChannelOrder, Luminance)
    CASE(ImageChannelOrder, Rx)
    CASE(ImageChannelOrder, RGx)
    CASE(ImageChannelOrder, RGBx)
    CASE(ImageChannelOrder, Depth)
    CASE(ImageChannelOrder, DepthStencil)
    CASE(ImageChannelOrder, sRGB)
    CASE(ImageChannelOrder, sRGBx)
    CASE(ImageChannelOrder, sRGBA)
    CASE(ImageChannelOrder, sBGRA)
    CASE(ImageChannelOrder, ABGR)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getImageChannelDataTypeName(ImageChannelDataType e) {
  switch (e) {
    CASE(ImageChannelDataType, SnormInt8)
    CASE(ImageChannelDataType, SnormInt16)
    CASE(ImageChannelDataType, UnormInt8)
    CASE(ImageChannelDataType, UnormInt16)
    CASE(ImageChannelDataType, UnormShort565)
    CASE(ImageChannelDataType, UnormShort555)
    CASE(ImageChannelDataType, UnormInt101010)
    CASE(ImageChannelDataType, SignedInt8)
    CASE(ImageChannelDataType, SignedInt16)
    CASE(ImageChannelDataType, SignedInt32)
    CASE(ImageChannelDataType, UnsignedInt8)
    CASE(ImageChannelDataType, UnsignedInt16)
    CASE(ImageChannelDataType, UnsigendInt32)
    CASE(ImageChannelDataType, HalfFloat)
    CASE(ImageChannelDataType, Float)
    CASE(ImageChannelDataType, UnormInt24)
    CASE(ImageChannelDataType, UnormInt101010_2)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

std::string getImageOperandName(uint32_t e) {
  std::string nameString = "";
  std::string sep = "";
  if (e == static_cast<uint32_t>(ImageOperand::None))
    return "None";
  if (e == static_cast<uint32_t>(ImageOperand::Bias))
    return "Bias";
  if (e & static_cast<uint32_t>(ImageOperand::Bias)) {
    nameString += sep + "Bias";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::Lod))
    return "Lod";
  if (e & static_cast<uint32_t>(ImageOperand::Lod)) {
    nameString += sep + "Lod";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::Grad))
    return "Grad";
  if (e & static_cast<uint32_t>(ImageOperand::Grad)) {
    nameString += sep + "Grad";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::ConstOffset))
    return "ConstOffset";
  if (e & static_cast<uint32_t>(ImageOperand::ConstOffset)) {
    nameString += sep + "ConstOffset";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::Offset))
    return "Offset";
  if (e & static_cast<uint32_t>(ImageOperand::Offset)) {
    nameString += sep + "Offset";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::ConstOffsets))
    return "ConstOffsets";
  if (e & static_cast<uint32_t>(ImageOperand::ConstOffsets)) {
    nameString += sep + "ConstOffsets";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::Sample))
    return "Sample";
  if (e & static_cast<uint32_t>(ImageOperand::Sample)) {
    nameString += sep + "Sample";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::MinLod))
    return "MinLod";
  if (e & static_cast<uint32_t>(ImageOperand::MinLod)) {
    nameString += sep + "MinLod";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::MakeTexelAvailableKHR))
    return "MakeTexelAvailableKHR";
  if (e & static_cast<uint32_t>(ImageOperand::MakeTexelAvailableKHR)) {
    nameString += sep + "MakeTexelAvailableKHR";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::MakeTexelVisibleKHR))
    return "MakeTexelVisibleKHR";
  if (e & static_cast<uint32_t>(ImageOperand::MakeTexelVisibleKHR)) {
    nameString += sep + "MakeTexelVisibleKHR";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::NonPrivateTexelKHR))
    return "NonPrivateTexelKHR";
  if (e & static_cast<uint32_t>(ImageOperand::NonPrivateTexelKHR)) {
    nameString += sep + "NonPrivateTexelKHR";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::VolatileTexelKHR))
    return "VolatileTexelKHR";
  if (e & static_cast<uint32_t>(ImageOperand::VolatileTexelKHR)) {
    nameString += sep + "VolatileTexelKHR";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::SignExtend))
    return "SignExtend";
  if (e & static_cast<uint32_t>(ImageOperand::SignExtend)) {
    nameString += sep + "SignExtend";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(ImageOperand::ZeroExtend))
    return "ZeroExtend";
  if (e & static_cast<uint32_t>(ImageOperand::ZeroExtend)) {
    nameString += sep + "ZeroExtend";
    sep = "|";
  };
  return nameString;
}

std::string getFPFastMathModeName(uint32_t e) {
  std::string nameString = "";
  std::string sep = "";
  if (e == static_cast<uint32_t>(FPFastMathMode::None))
    return "None";
  if (e == static_cast<uint32_t>(FPFastMathMode::NotNaN))
    return "NotNaN";
  if (e & static_cast<uint32_t>(FPFastMathMode::NotNaN)) {
    nameString += sep + "NotNaN";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(FPFastMathMode::NotInf))
    return "NotInf";
  if (e & static_cast<uint32_t>(FPFastMathMode::NotInf)) {
    nameString += sep + "NotInf";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(FPFastMathMode::NSZ))
    return "NSZ";
  if (e & static_cast<uint32_t>(FPFastMathMode::NSZ)) {
    nameString += sep + "NSZ";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(FPFastMathMode::AllowRecip))
    return "AllowRecip";
  if (e & static_cast<uint32_t>(FPFastMathMode::AllowRecip)) {
    nameString += sep + "AllowRecip";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(FPFastMathMode::Fast))
    return "Fast";
  if (e & static_cast<uint32_t>(FPFastMathMode::Fast)) {
    nameString += sep + "Fast";
    sep = "|";
  };
  return nameString;
}

StringRef getFPRoundingModeName(FPRoundingMode e) {
  switch (e) {
    CASE(FPRoundingMode, RTE)
    CASE(FPRoundingMode, RTZ)
    CASE(FPRoundingMode, RTP)
    CASE(FPRoundingMode, RTN)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getLinkageTypeName(LinkageType e) {
  switch (e) {
    CASE(LinkageType, Export)
    CASE(LinkageType, Import)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getAccessQualifierName(AccessQualifier e) {
  switch (e) {
    CASE(AccessQualifier, ReadOnly)
    CASE(AccessQualifier, WriteOnly)
    CASE(AccessQualifier, ReadWrite)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getFunctionParameterAttributeName(FunctionParameterAttribute e) {
  switch (e) {
    CASE(FunctionParameterAttribute, Zext)
    CASE(FunctionParameterAttribute, Sext)
    CASE(FunctionParameterAttribute, ByVal)
    CASE(FunctionParameterAttribute, Sret)
    CASE(FunctionParameterAttribute, NoAlias)
    CASE(FunctionParameterAttribute, NoCapture)
    CASE(FunctionParameterAttribute, NoWrite)
    CASE(FunctionParameterAttribute, NoReadWrite)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getDecorationName(Decoration e) {
  switch (e) {
    CASE(Decoration, RelaxedPrecision)
    CASE(Decoration, SpecId)
    CASE(Decoration, Block)
    CASE(Decoration, BufferBlock)
    CASE(Decoration, RowMajor)
    CASE(Decoration, ColMajor)
    CASE(Decoration, ArrayStride)
    CASE(Decoration, MatrixStride)
    CASE(Decoration, GLSLShared)
    CASE(Decoration, GLSLPacked)
    CASE(Decoration, CPacked)
    CASE(Decoration, BuiltIn)
    CASE(Decoration, NoPerspective)
    CASE(Decoration, Flat)
    CASE(Decoration, Patch)
    CASE(Decoration, Centroid)
    CASE(Decoration, Sample)
    CASE(Decoration, Invariant)
    CASE(Decoration, Restrict)
    CASE(Decoration, Aliased)
    CASE(Decoration, Volatile)
    CASE(Decoration, Constant)
    CASE(Decoration, Coherent)
    CASE(Decoration, NonWritable)
    CASE(Decoration, NonReadable)
    CASE(Decoration, Uniform)
    CASE(Decoration, UniformId)
    CASE(Decoration, SaturatedConversion)
    CASE(Decoration, Stream)
    CASE(Decoration, Location)
    CASE(Decoration, Component)
    CASE(Decoration, Index)
    CASE(Decoration, Binding)
    CASE(Decoration, DescriptorSet)
    CASE(Decoration, Offset)
    CASE(Decoration, XfbBuffer)
    CASE(Decoration, XfbStride)
    CASE(Decoration, FuncParamAttr)
    CASE(Decoration, FPRoundingMode)
    CASE(Decoration, FPFastMathMode)
    CASE(Decoration, LinkageAttributes)
    CASE(Decoration, NoContraction)
    CASE(Decoration, InputAttachmentIndex)
    CASE(Decoration, Alignment)
    CASE(Decoration, MaxByteOffset)
    CASE(Decoration, AlignmentId)
    CASE(Decoration, MaxByteOffsetId)
    CASE(Decoration, NoSignedWrap)
    CASE(Decoration, NoUnsignedWrap)
    CASE(Decoration, ExplicitInterpAMD)
    CASE(Decoration, OverrideCoverageNV)
    CASE(Decoration, PassthroughNV)
    CASE(Decoration, ViewportRelativeNV)
    CASE(Decoration, SecondaryViewportRelativeNV)
    CASE(Decoration, PerPrimitiveNV)
    CASE(Decoration, PerViewNV)
    CASE(Decoration, PerVertexNV)
    CASE(Decoration, NonUniformEXT)
    CASE(Decoration, CountBuffer)
    CASE(Decoration, UserSemantic)
    CASE(Decoration, RestrictPointerEXT)
    CASE(Decoration, AliasedPointerEXT)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getBuiltInName(BuiltIn e) {
  switch (e) {
    CASE(BuiltIn, Position)
    CASE(BuiltIn, PointSize)
    CASE(BuiltIn, ClipDistance)
    CASE(BuiltIn, CullDistance)
    CASE(BuiltIn, VertexId)
    CASE(BuiltIn, InstanceId)
    CASE(BuiltIn, PrimitiveId)
    CASE(BuiltIn, InvocationId)
    CASE(BuiltIn, Layer)
    CASE(BuiltIn, ViewportIndex)
    CASE(BuiltIn, TessLevelOuter)
    CASE(BuiltIn, TessLevelInner)
    CASE(BuiltIn, TessCoord)
    CASE(BuiltIn, PatchVertices)
    CASE(BuiltIn, FragCoord)
    CASE(BuiltIn, PointCoord)
    CASE(BuiltIn, FrontFacing)
    CASE(BuiltIn, SampleId)
    CASE(BuiltIn, SamplePosition)
    CASE(BuiltIn, SampleMask)
    CASE(BuiltIn, FragDepth)
    CASE(BuiltIn, HelperInvocation)
    CASE(BuiltIn, NumWorkgroups)
    CASE(BuiltIn, WorkgroupSize)
    CASE(BuiltIn, WorkgroupId)
    CASE(BuiltIn, LocalInvocationId)
    CASE(BuiltIn, GlobalInvocationId)
    CASE(BuiltIn, LocalInvocationIndex)
    CASE(BuiltIn, WorkDim)
    CASE(BuiltIn, GlobalSize)
    CASE(BuiltIn, EnqueuedWorkgroupSize)
    CASE(BuiltIn, GlobalOffset)
    CASE(BuiltIn, GlobalLinearId)
    CASE(BuiltIn, SubgroupSize)
    CASE(BuiltIn, SubgroupMaxSize)
    CASE(BuiltIn, NumSubgroups)
    CASE(BuiltIn, NumEnqueuedSubgroups)
    CASE(BuiltIn, SubgroupId)
    CASE(BuiltIn, SubgroupLocalInvocationId)
    CASE(BuiltIn, VertexIndex)
    CASE(BuiltIn, InstanceIndex)
    CASE(BuiltIn, SubgroupEqMask)
    CASE(BuiltIn, SubgroupGeMask)
    CASE(BuiltIn, SubgroupGtMask)
    CASE(BuiltIn, SubgroupLeMask)
    CASE(BuiltIn, SubgroupLtMask)
    CASE(BuiltIn, BaseVertex)
    CASE(BuiltIn, BaseInstance)
    CASE(BuiltIn, DrawIndex)
    CASE(BuiltIn, DeviceIndex)
    CASE(BuiltIn, ViewIndex)
    CASE(BuiltIn, BaryCoordNoPerspAMD)
    CASE(BuiltIn, BaryCoordNoPerspCentroidAMD)
    CASE(BuiltIn, BaryCoordNoPerspSampleAMD)
    CASE(BuiltIn, BaryCoordSmoothAMD)
    CASE(BuiltIn, BaryCoordSmoothCentroid)
    CASE(BuiltIn, BaryCoordSmoothSample)
    CASE(BuiltIn, BaryCoordPullModel)
    CASE(BuiltIn, FragStencilRefEXT)
    CASE(BuiltIn, ViewportMaskNV)
    CASE(BuiltIn, SecondaryPositionNV)
    CASE(BuiltIn, SecondaryViewportMaskNV)
    CASE(BuiltIn, PositionPerViewNV)
    CASE(BuiltIn, ViewportMaskPerViewNV)
    CASE(BuiltIn, FullyCoveredEXT)
    CASE(BuiltIn, TaskCountNV)
    CASE(BuiltIn, PrimitiveCountNV)
    CASE(BuiltIn, PrimitiveIndicesNV)
    CASE(BuiltIn, ClipDistancePerViewNV)
    CASE(BuiltIn, CullDistancePerViewNV)
    CASE(BuiltIn, LayerPerViewNV)
    CASE(BuiltIn, MeshViewCountNV)
    CASE(BuiltIn, MeshViewIndices)
    CASE(BuiltIn, BaryCoordNV)
    CASE(BuiltIn, BaryCoordNoPerspNV)
    CASE(BuiltIn, FragSizeEXT)
    CASE(BuiltIn, FragInvocationCountEXT)
    CASE(BuiltIn, LaunchIdNV)
    CASE(BuiltIn, LaunchSizeNV)
    CASE(BuiltIn, WorldRayOriginNV)
    CASE(BuiltIn, WorldRayDirectionNV)
    CASE(BuiltIn, ObjectRayOriginNV)
    CASE(BuiltIn, ObjectRayDirectionNV)
    CASE(BuiltIn, RayTminNV)
    CASE(BuiltIn, RayTmaxNV)
    CASE(BuiltIn, InstanceCustomIndexNV)
    CASE(BuiltIn, ObjectToWorldNV)
    CASE(BuiltIn, WorldToObjectNV)
    CASE(BuiltIn, HitTNV)
    CASE(BuiltIn, HitKindNV)
    CASE(BuiltIn, IncomingRayFlagsNV)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

std::string getSelectionControlName(uint32_t e) {
  std::string nameString = "";
  std::string sep = "";
  if (e == static_cast<uint32_t>(SelectionControl::None))
    return "None";
  if (e == static_cast<uint32_t>(SelectionControl::Flatten))
    return "Flatten";
  if (e & static_cast<uint32_t>(SelectionControl::Flatten)) {
    nameString += sep + "Flatten";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(SelectionControl::DontFlatten))
    return "DontFlatten";
  if (e & static_cast<uint32_t>(SelectionControl::DontFlatten)) {
    nameString += sep + "DontFlatten";
    sep = "|";
  };
  return nameString;
}

std::string getLoopControlName(uint32_t e) {
  std::string nameString = "";
  std::string sep = "";
  if (e == static_cast<uint32_t>(LoopControl::None))
    return "None";
  if (e == static_cast<uint32_t>(LoopControl::Unroll))
    return "Unroll";
  if (e & static_cast<uint32_t>(LoopControl::Unroll)) {
    nameString += sep + "Unroll";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(LoopControl::DontUnroll))
    return "DontUnroll";
  if (e & static_cast<uint32_t>(LoopControl::DontUnroll)) {
    nameString += sep + "DontUnroll";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(LoopControl::DependencyInfinite))
    return "DependencyInfinite";
  if (e & static_cast<uint32_t>(LoopControl::DependencyInfinite)) {
    nameString += sep + "DependencyInfinite";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(LoopControl::DependencyLength))
    return "DependencyLength";
  if (e & static_cast<uint32_t>(LoopControl::DependencyLength)) {
    nameString += sep + "DependencyLength";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(LoopControl::MinIterations))
    return "MinIterations";
  if (e & static_cast<uint32_t>(LoopControl::MinIterations)) {
    nameString += sep + "MinIterations";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(LoopControl::MaxIterations))
    return "MaxIterations";
  if (e & static_cast<uint32_t>(LoopControl::MaxIterations)) {
    nameString += sep + "MaxIterations";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(LoopControl::IterationMultiple))
    return "IterationMultiple";
  if (e & static_cast<uint32_t>(LoopControl::IterationMultiple)) {
    nameString += sep + "IterationMultiple";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(LoopControl::PeelCount))
    return "PeelCount";
  if (e & static_cast<uint32_t>(LoopControl::PeelCount)) {
    nameString += sep + "PeelCount";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(LoopControl::PartialCount))
    return "PartialCount";
  if (e & static_cast<uint32_t>(LoopControl::PartialCount)) {
    nameString += sep + "PartialCount";
    sep = "|";
  };
  return nameString;
}

std::string getFunctionControlName(uint32_t e) {
  std::string nameString = "";
  std::string sep = "";
  if (e == static_cast<uint32_t>(FunctionControl::None))
    return "None";
  if (e == static_cast<uint32_t>(FunctionControl::Inline))
    return "Inline";
  if (e & static_cast<uint32_t>(FunctionControl::Inline)) {
    nameString += sep + "Inline";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(FunctionControl::DontInline))
    return "DontInline";
  if (e & static_cast<uint32_t>(FunctionControl::DontInline)) {
    nameString += sep + "DontInline";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(FunctionControl::Pure))
    return "Pure";
  if (e & static_cast<uint32_t>(FunctionControl::Pure)) {
    nameString += sep + "Pure";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(FunctionControl::Const))
    return "Const";
  if (e & static_cast<uint32_t>(FunctionControl::Const)) {
    nameString += sep + "Const";
    sep = "|";
  };
  return nameString;
}

std::string getMemorySemanticsName(uint32_t e) {
  std::string nameString = "";
  std::string sep = "";
  if (e == static_cast<uint32_t>(MemorySemantics::None))
    return "None";
  if (e == static_cast<uint32_t>(MemorySemantics::Acquire))
    return "Acquire";
  if (e & static_cast<uint32_t>(MemorySemantics::Acquire)) {
    nameString += sep + "Acquire";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::Release))
    return "Release";
  if (e & static_cast<uint32_t>(MemorySemantics::Release)) {
    nameString += sep + "Release";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::AcquireRelease))
    return "AcquireRelease";
  if (e & static_cast<uint32_t>(MemorySemantics::AcquireRelease)) {
    nameString += sep + "AcquireRelease";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::SequentiallyConsistent))
    return "SequentiallyConsistent";
  if (e & static_cast<uint32_t>(MemorySemantics::SequentiallyConsistent)) {
    nameString += sep + "SequentiallyConsistent";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::UniformMemory))
    return "UniformMemory";
  if (e & static_cast<uint32_t>(MemorySemantics::UniformMemory)) {
    nameString += sep + "UniformMemory";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::SubgroupMemory))
    return "SubgroupMemory";
  if (e & static_cast<uint32_t>(MemorySemantics::SubgroupMemory)) {
    nameString += sep + "SubgroupMemory";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::WorkgroupMemory))
    return "WorkgroupMemory";
  if (e & static_cast<uint32_t>(MemorySemantics::WorkgroupMemory)) {
    nameString += sep + "WorkgroupMemory";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory))
    return "CrossWorkgroupMemory";
  if (e & static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory)) {
    nameString += sep + "CrossWorkgroupMemory";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::AtomicCounterMemory))
    return "AtomicCounterMemory";
  if (e & static_cast<uint32_t>(MemorySemantics::AtomicCounterMemory)) {
    nameString += sep + "AtomicCounterMemory";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::ImageMemory))
    return "ImageMemory";
  if (e & static_cast<uint32_t>(MemorySemantics::ImageMemory)) {
    nameString += sep + "ImageMemory";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::OutputMemoryKHR))
    return "OutputMemoryKHR";
  if (e & static_cast<uint32_t>(MemorySemantics::OutputMemoryKHR)) {
    nameString += sep + "OutputMemoryKHR";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::MakeAvailableKHR))
    return "MakeAvailableKHR";
  if (e & static_cast<uint32_t>(MemorySemantics::MakeAvailableKHR)) {
    nameString += sep + "MakeAvailableKHR";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemorySemantics::MakeVisibleKHR))
    return "MakeVisibleKHR";
  if (e & static_cast<uint32_t>(MemorySemantics::MakeVisibleKHR)) {
    nameString += sep + "MakeVisibleKHR";
    sep = "|";
  };
  return nameString;
}

std::string getMemoryOperandName(uint32_t e) {
  std::string nameString = "";
  std::string sep = "";
  if (e == static_cast<uint32_t>(MemoryOperand::None))
    return "None";
  if (e == static_cast<uint32_t>(MemoryOperand::Volatile))
    return "Volatile";
  if (e & static_cast<uint32_t>(MemoryOperand::Volatile)) {
    nameString += sep + "Volatile";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemoryOperand::Aligned))
    return "Aligned";
  if (e & static_cast<uint32_t>(MemoryOperand::Aligned)) {
    nameString += sep + "Aligned";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemoryOperand::Nontemporal))
    return "Nontemporal";
  if (e & static_cast<uint32_t>(MemoryOperand::Nontemporal)) {
    nameString += sep + "Nontemporal";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemoryOperand::MakePointerAvailableKHR))
    return "MakePointerAvailableKHR";
  if (e & static_cast<uint32_t>(MemoryOperand::MakePointerAvailableKHR)) {
    nameString += sep + "MakePointerAvailableKHR";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemoryOperand::MakePointerVisibleKHR))
    return "MakePointerVisibleKHR";
  if (e & static_cast<uint32_t>(MemoryOperand::MakePointerVisibleKHR)) {
    nameString += sep + "MakePointerVisibleKHR";
    sep = "|";
  }
  if (e == static_cast<uint32_t>(MemoryOperand::NonPrivatePointerKHR))
    return "NonPrivatePointerKHR";
  if (e & static_cast<uint32_t>(MemoryOperand::NonPrivatePointerKHR)) {
    nameString += sep + "NonPrivatePointerKHR";
    sep = "|";
  };
  return nameString;
}

StringRef getScopeName(Scope e) {
  switch (e) {
    CASE(Scope, CrossDevice)
    CASE(Scope, Device)
    CASE(Scope, Workgroup)
    CASE(Scope, Subgroup)
    CASE(Scope, Invocation)
    CASE(Scope, QueueFamilyKHR)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getGroupOperationName(GroupOperation e) {
  switch (e) {
    CASE(GroupOperation, Reduce)
    CASE(GroupOperation, InclusiveScan)
    CASE(GroupOperation, ExclusiveScan)
    CASE(GroupOperation, ClusteredReduce)
    CASE(GroupOperation, PartitionedReduceNV)
    CASE(GroupOperation, PartitionedInclusiveScanNV)
    CASE(GroupOperation, PartitionedExclusiveScanNV)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getKernelEnqueueFlagsName(KernelEnqueueFlags e) {
  switch (e) {
    CASE(KernelEnqueueFlags, NoWait)
    CASE(KernelEnqueueFlags, WaitKernel)
    CASE(KernelEnqueueFlags, WaitWorkGroup)
    break;
  }
  llvm_unreachable("Unexpected operand");
}

StringRef getKernelProfilingInfoName(KernelProfilingInfo e) {
  switch (e) {
    CASE(KernelProfilingInfo, None)
    CASE(KernelProfilingInfo, CmdExecTime)
    break;
  }
  llvm_unreachable("Unexpected operand");
}
} // namespace SPIRV
} // namespace llvm
