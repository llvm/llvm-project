//===-- SPIRVBaseInfo.h -  Top level definitions for SPIRV ------*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_SPIRV_MCTARGETDESC_SPIRVBASEINFO_H
#define LLVM_LIB_TARGET_SPIRV_MCTARGETDESC_SPIRVBASEINFO_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {
namespace SPIRV {
enum class Capability : uint32_t {
  Matrix = 0,
  Shader = 1,
  Geometry = 2,
  Tessellation = 3,
  Addresses = 4,
  Linkage = 5,
  Kernel = 6,
  Vector16 = 7,
  Float16Buffer = 8,
  Float16 = 9,
  Float64 = 10,
  Int64 = 11,
  Int64Atomics = 12,
  ImageBasic = 13,
  ImageReadWrite = 14,
  ImageMipmap = 15,
  Pipes = 17,
  Groups = 18,
  DeviceEnqueue = 19,
  LiteralSampler = 20,
  AtomicStorage = 21,
  Int16 = 22,
  TessellationPointSize = 23,
  GeometryPointSize = 24,
  ImageGatherExtended = 25,
  StorageImageMultisample = 27,
  UniformBufferArrayDynamicIndexing = 28,
  SampledImageArrayDymnamicIndexing = 29,
  ClipDistance = 32,
  CullDistance = 33,
  ImageCubeArray = 34,
  SampleRateShading = 35,
  ImageRect = 36,
  SampledRect = 37,
  GenericPointer = 38,
  Int8 = 39,
  InputAttachment = 40,
  SparseResidency = 41,
  MinLod = 42,
  Sampled1D = 43,
  Image1D = 44,
  SampledCubeArray = 45,
  SampledBuffer = 46,
  ImageBuffer = 47,
  ImageMSArray = 48,
  StorageImageExtendedFormats = 49,
  ImageQuery = 50,
  DerivativeControl = 51,
  InterpolationFunction = 52,
  TransformFeedback = 53,
  GeometryStreams = 54,
  StorageImageReadWithoutFormat = 55,
  StorageImageWriteWithoutFormat = 56,
  MultiViewport = 57,
  SubgroupDispatch = 58,
  NamedBarrier = 59,
  PipeStorage = 60,
  GroupNonUniform = 61,
  GroupNonUniformVote = 62,
  GroupNonUniformArithmetic = 63,
  GroupNonUniformBallot = 64,
  GroupNonUniformShuffle = 65,
  GroupNonUniformShuffleRelative = 66,
  GroupNonUniformClustered = 67,
  GroupNonUniformQuad = 68,
  SubgroupBallotKHR = 4423,
  DrawParameters = 4427,
  SubgroupVoteKHR = 4431,
  StorageBuffer16BitAccess = 4433,
  StorageUniform16 = 4434,
  StoragePushConstant16 = 4435,
  StorageInputOutput16 = 4436,
  DeviceGroup = 4437,
  MultiView = 4439,
  VariablePointersStorageBuffer = 4441,
  VariablePointers = 4442,
  AtomicStorageOps = 4445,
  SampleMaskPostDepthCoverage = 4447,
  StorageBuffer8BitAccess = 4448,
  UniformAndStorageBuffer8BitAccess = 4449,
  StoragePushConstant8 = 4450,
  DenormPreserve = 4464,
  DenormFlushToZero = 4465,
  SignedZeroInfNanPreserve = 4466,
  RoundingModeRTE = 4467,
  RoundingModeRTZ = 4468,
  Float16ImageAMD = 5008,
  ImageGatherBiasLodAMD = 5009,
  FragmentMaskAMD = 5010,
  StencilExportEXT = 5013,
  ImageReadWriteLodAMD = 5015,
  SampleMaskOverrideCoverageNV = 5249,
  GeometryShaderPassthroughNV = 5251,
  ShaderViewportIndexLayerEXT = 5254,
  ShaderViewportMaskNV = 5255,
  ShaderStereoViewNV = 5259,
  PerViewAttributesNV = 5260,
  FragmentFullyCoveredEXT = 5265,
  MeshShadingNV = 5266,
  ShaderNonUniformEXT = 5301,
  RuntimeDescriptorArrayEXT = 5302,
  InputAttachmentArrayDynamicIndexingEXT = 5303,
  UniformTexelBufferArrayDynamicIndexingEXT = 5304,
  StorageTexelBufferArrayDynamicIndexingEXT = 5305,
  UniformBufferArrayNonUniformIndexingEXT = 5306,
  SampledImageArrayNonUniformIndexingEXT = 5307,
  StorageBufferArrayNonUniformIndexingEXT = 5308,
  StorageImageArrayNonUniformIndexingEXT = 5309,
  InputAttachmentArrayNonUniformIndexingEXT = 5310,
  UniformTexelBufferArrayNonUniformIndexingEXT = 5311,
  StorageTexelBufferArrayNonUniformIndexingEXT = 5312,
  RayTracingNV = 5340,
  SubgroupShuffleINTEL = 5568,
  SubgroupBufferBlockIOINTEL = 5569,
  SubgroupImageBlockIOINTEL = 5570,
  SubgroupImageMediaBlockIOINTEL = 5579,
  SubgroupAvcMotionEstimationINTEL = 5696,
  SubgroupAvcMotionEstimationIntraINTEL = 5697,
  SubgroupAvcMotionEstimationChromaINTEL = 5698,
  GroupNonUniformPartitionedNV = 5297,
  VulkanMemoryModelKHR = 5345,
  VulkanMemoryModelDeviceScopeKHR = 5346,
  ImageFootprintNV = 5282,
  FragmentBarycentricNV = 5284,
  ComputeDerivativeGroupQuadsNV = 5288,
  ComputeDerivativeGroupLinearNV = 5350,
  FragmentDensityEXT = 5291,
  PhysicalStorageBufferAddressesEXT = 5347,
  CooperativeMatrixNV = 5357,
};
StringRef getCapabilityName(Capability e);

enum class SourceLanguage : uint32_t {
  Unknown = 0,
  ESSL = 1,
  GLSL = 2,
  OpenCL_C = 3,
  OpenCL_CPP = 4,
  HLSL = 5,
};
StringRef getSourceLanguageName(SourceLanguage e);

enum class AddressingModel : uint32_t {
  Logical = 0,
  Physical32 = 1,
  Physical64 = 2,
  PhysicalStorageBuffer64EXT = 5348,
};
StringRef getAddressingModelName(AddressingModel e);

enum class ExecutionModel : uint32_t {
  Vertex = 0,
  TessellationControl = 1,
  TessellationEvaluation = 2,
  Geometry = 3,
  Fragment = 4,
  GLCompute = 5,
  Kernel = 6,
  TaskNV = 5267,
  MeshNV = 5268,
  RayGenerationNV = 5313,
  IntersectionNV = 5314,
  AnyHitNV = 5315,
  ClosestHitNV = 5316,
  MissNV = 5317,
  CallableNV = 5318,
};
StringRef getExecutionModelName(ExecutionModel e);

enum class MemoryModel : uint32_t {
  Simple = 0,
  GLSL450 = 1,
  OpenCL = 2,
  VulkanKHR = 3,
};
StringRef getMemoryModelName(MemoryModel e);

enum class ExecutionMode : uint32_t {
  Invocations = 0,
  SpacingEqual = 1,
  SpacingFractionalEven = 2,
  SpacingFractionalOdd = 3,
  VertexOrderCw = 4,
  VertexOrderCcw = 5,
  PixelCenterInteger = 6,
  OriginUpperLeft = 7,
  OriginLowerLeft = 8,
  EarlyFragmentTests = 9,
  PointMode = 10,
  Xfb = 11,
  DepthReplacing = 12,
  DepthGreater = 14,
  DepthLess = 15,
  DepthUnchanged = 16,
  LocalSize = 17,
  LocalSizeHint = 18,
  InputPoints = 19,
  InputLines = 20,
  InputLinesAdjacency = 21,
  Triangles = 22,
  InputTrianglesAdjacency = 23,
  Quads = 24,
  Isolines = 25,
  OutputVertices = 26,
  OutputPoints = 27,
  OutputLineStrip = 28,
  OutputTriangleStrip = 29,
  VecTypeHint = 30,
  ContractionOff = 31,
  Initializer = 33,
  Finalizer = 34,
  SubgroupSize = 35,
  SubgroupsPerWorkgroup = 36,
  SubgroupsPerWorkgroupId = 37,
  LocalSizeId = 38,
  LocalSizeHintId = 39,
  PostDepthCoverage = 4446,
  DenormPreserve = 4459,
  DenormFlushToZero = 4460,
  SignedZeroInfNanPreserve = 4461,
  RoundingModeRTE = 4462,
  RoundingModeRTZ = 4463,
  StencilRefReplacingEXT = 5027,
  OutputLinesNV = 5269,
  DerivativeGroupQuadsNV = 5289,
  DerivativeGroupLinearNV = 5290,
  OutputTrianglesNV = 5298,
};
StringRef getExecutionModeName(ExecutionMode e);

enum class StorageClass : uint32_t {
  UniformConstant = 0,
  Input = 1,
  Uniform = 2,
  Output = 3,
  Workgroup = 4,
  CrossWorkgroup = 5,
  Private = 6,
  Function = 7,
  Generic = 8,
  PushConstant = 9,
  AtomicCounter = 10,
  Image = 11,
  StorageBuffer = 12,
  CallableDataNV = 5328,
  IncomingCallableDataNV = 5329,
  RayPayloadNV = 5338,
  HitAttributeNV = 5339,
  IncomingRayPayloadNV = 5342,
  ShaderRecordBufferNV = 5343,
  PhysicalStorageBufferEXT = 5349,
};
StringRef getStorageClassName(StorageClass e);

enum class Dim : uint32_t {
  DIM_1D = 0,
  DIM_2D = 1,
  DIM_3D = 2,
  DIM_Cube = 3,
  DIM_Rect = 4,
  DIM_Buffer = 5,
  DIM_SubpassData = 6,
};
StringRef getDimName(Dim e);

enum class SamplerAddressingMode : uint32_t {
  None = 0,
  ClampToEdge = 1,
  Clamp = 2,
  Repeat = 3,
  RepeatMirrored = 4,
};
StringRef getSamplerAddressingModeName(SamplerAddressingMode e);

enum class SamplerFilterMode : uint32_t {
  Nearest = 0,
  Linear = 1,
};
StringRef getSamplerFilterModeName(SamplerFilterMode e);

enum class ImageFormat : uint32_t {
  Unknown = 0,
  Rgba32f = 1,
  Rgba16f = 2,
  R32f = 3,
  Rgba8 = 4,
  Rgba8Snorm = 5,
  Rg32f = 6,
  Rg16f = 7,
  R11fG11fB10f = 8,
  R16f = 9,
  Rgba16 = 10,
  Rgb10A2 = 11,
  Rg16 = 12,
  Rg8 = 13,
  R16 = 14,
  R8 = 15,
  Rgba16Snorm = 16,
  Rg16Snorm = 17,
  Rg8Snorm = 18,
  R16Snorm = 19,
  R8Snorm = 20,
  Rgba32i = 21,
  Rgba16i = 22,
  Rgba8i = 23,
  R32i = 24,
  Rg32i = 25,
  Rg16i = 26,
  Rg8i = 27,
  R16i = 28,
  R8i = 29,
  Rgba32ui = 30,
  Rgba16ui = 31,
  Rgba8ui = 32,
  R32ui = 33,
  Rgb10a2ui = 34,
  Rg32ui = 35,
  Rg16ui = 36,
  Rg8ui = 37,
  R16ui = 38,
  R8ui = 39,
};
StringRef getImageFormatName(ImageFormat e);

enum class ImageChannelOrder : uint32_t {
  R = 0,
  A = 1,
  RG = 2,
  RA = 3,
  RGB = 4,
  RGBA = 5,
  BGRA = 6,
  ARGB = 7,
  Intensity = 8,
  Luminance = 9,
  Rx = 10,
  RGx = 11,
  RGBx = 12,
  Depth = 13,
  DepthStencil = 14,
  sRGB = 15,
  sRGBx = 16,
  sRGBA = 17,
  sBGRA = 18,
  ABGR = 19,
};
StringRef getImageChannelOrderName(ImageChannelOrder e);

enum class ImageChannelDataType : uint32_t {
  SnormInt8 = 0,
  SnormInt16 = 1,
  UnormInt8 = 2,
  UnormInt16 = 3,
  UnormShort565 = 4,
  UnormShort555 = 5,
  UnormInt101010 = 6,
  SignedInt8 = 7,
  SignedInt16 = 8,
  SignedInt32 = 9,
  UnsignedInt8 = 10,
  UnsignedInt16 = 11,
  UnsigendInt32 = 12,
  HalfFloat = 13,
  Float = 14,
  UnormInt24 = 15,
  UnormInt101010_2 = 16,
};
StringRef getImageChannelDataTypeName(ImageChannelDataType e);

enum class ImageOperand : uint32_t {
  None = 0x0,
  Bias = 0x1,
  Lod = 0x2,
  Grad = 0x4,
  ConstOffset = 0x8,
  Offset = 0x10,
  ConstOffsets = 0x20,
  Sample = 0x40,
  MinLod = 0x80,
  MakeTexelAvailableKHR = 0x100,
  MakeTexelVisibleKHR = 0x200,
  NonPrivateTexelKHR = 0x400,
  VolatileTexelKHR = 0x800,
  SignExtend = 0x1000,
  ZeroExtend = 0x2000,
};
std::string getImageOperandName(uint32_t e);

enum class FPFastMathMode : uint32_t {
  None = 0x0,
  NotNaN = 0x1,
  NotInf = 0x2,
  NSZ = 0x4,
  AllowRecip = 0x8,
  Fast = 0x10,
};
std::string getFPFastMathModeName(uint32_t e);

enum class FPRoundingMode : uint32_t {
  RTE = 0,
  RTZ = 1,
  RTP = 2,
  RTN = 3,
};
StringRef getFPRoundingModeName(FPRoundingMode e);

enum class LinkageType : uint32_t {
  Export = 0,
  Import = 1,
};
StringRef getLinkageTypeName(LinkageType e);

enum class AccessQualifier : uint32_t {
  ReadOnly = 0,
  WriteOnly = 1,
  ReadWrite = 2,
};
StringRef getAccessQualifierName(AccessQualifier e);

enum class FunctionParameterAttribute : uint32_t {
  Zext = 0,
  Sext = 1,
  ByVal = 2,
  Sret = 3,
  NoAlias = 4,
  NoCapture = 5,
  NoWrite = 6,
  NoReadWrite = 7,
};
StringRef getFunctionParameterAttributeName(FunctionParameterAttribute e);

enum class Decoration : uint32_t {
  RelaxedPrecision = 0,
  SpecId = 1,
  Block = 2,
  BufferBlock = 3,
  RowMajor = 4,
  ColMajor = 5,
  ArrayStride = 6,
  MatrixStride = 7,
  GLSLShared = 8,
  GLSLPacked = 9,
  CPacked = 10,
  BuiltIn = 11,
  NoPerspective = 13,
  Flat = 14,
  Patch = 15,
  Centroid = 16,
  Sample = 17,
  Invariant = 18,
  Restrict = 19,
  Aliased = 20,
  Volatile = 21,
  Constant = 22,
  Coherent = 23,
  NonWritable = 24,
  NonReadable = 25,
  Uniform = 26,
  UniformId = 27,
  SaturatedConversion = 28,
  Stream = 29,
  Location = 30,
  Component = 31,
  Index = 32,
  Binding = 33,
  DescriptorSet = 34,
  Offset = 35,
  XfbBuffer = 36,
  XfbStride = 37,
  FuncParamAttr = 38,
  FPRoundingMode = 39,
  FPFastMathMode = 40,
  LinkageAttributes = 41,
  NoContraction = 42,
  InputAttachmentIndex = 43,
  Alignment = 44,
  MaxByteOffset = 45,
  AlignmentId = 46,
  MaxByteOffsetId = 47,
  NoSignedWrap = 4469,
  NoUnsignedWrap = 4470,
  ExplicitInterpAMD = 4999,
  OverrideCoverageNV = 5248,
  PassthroughNV = 5250,
  ViewportRelativeNV = 5252,
  SecondaryViewportRelativeNV = 5256,
  PerPrimitiveNV = 5271,
  PerViewNV = 5272,
  PerVertexNV = 5273,
  NonUniformEXT = 5300,
  CountBuffer = 5634,
  UserSemantic = 5635,
  RestrictPointerEXT = 5355,
  AliasedPointerEXT = 5356,
};
StringRef getDecorationName(Decoration e);

enum class BuiltIn : uint32_t {
  Position = 0,
  PointSize = 1,
  ClipDistance = 3,
  CullDistance = 4,
  VertexId = 5,
  InstanceId = 6,
  PrimitiveId = 7,
  InvocationId = 8,
  Layer = 9,
  ViewportIndex = 10,
  TessLevelOuter = 11,
  TessLevelInner = 12,
  TessCoord = 13,
  PatchVertices = 14,
  FragCoord = 15,
  PointCoord = 16,
  FrontFacing = 17,
  SampleId = 18,
  SamplePosition = 19,
  SampleMask = 20,
  FragDepth = 22,
  HelperInvocation = 23,
  NumWorkgroups = 24,
  WorkgroupSize = 25,
  WorkgroupId = 26,
  LocalInvocationId = 27,
  GlobalInvocationId = 28,
  LocalInvocationIndex = 29,
  WorkDim = 30,
  GlobalSize = 31,
  EnqueuedWorkgroupSize = 32,
  GlobalOffset = 33,
  GlobalLinearId = 34,
  SubgroupSize = 36,
  SubgroupMaxSize = 37,
  NumSubgroups = 38,
  NumEnqueuedSubgroups = 39,
  SubgroupId = 40,
  SubgroupLocalInvocationId = 41,
  VertexIndex = 42,
  InstanceIndex = 43,
  SubgroupEqMask = 4416,
  SubgroupGeMask = 4417,
  SubgroupGtMask = 4418,
  SubgroupLeMask = 4419,
  SubgroupLtMask = 4420,
  BaseVertex = 4424,
  BaseInstance = 4425,
  DrawIndex = 4426,
  DeviceIndex = 4438,
  ViewIndex = 4440,
  BaryCoordNoPerspAMD = 4492,
  BaryCoordNoPerspCentroidAMD = 4493,
  BaryCoordNoPerspSampleAMD = 4494,
  BaryCoordSmoothAMD = 4495,
  BaryCoordSmoothCentroid = 4496,
  BaryCoordSmoothSample = 4497,
  BaryCoordPullModel = 4498,
  FragStencilRefEXT = 5014,
  ViewportMaskNV = 5253,
  SecondaryPositionNV = 5257,
  SecondaryViewportMaskNV = 5258,
  PositionPerViewNV = 5261,
  ViewportMaskPerViewNV = 5262,
  FullyCoveredEXT = 5264,
  TaskCountNV = 5274,
  PrimitiveCountNV = 5275,
  PrimitiveIndicesNV = 5276,
  ClipDistancePerViewNV = 5277,
  CullDistancePerViewNV = 5278,
  LayerPerViewNV = 5279,
  MeshViewCountNV = 5280,
  MeshViewIndices = 5281,
  BaryCoordNV = 5286,
  BaryCoordNoPerspNV = 5287,
  FragSizeEXT = 5292,
  FragInvocationCountEXT = 5293,
  LaunchIdNV = 5319,
  LaunchSizeNV = 5320,
  WorldRayOriginNV = 5321,
  WorldRayDirectionNV = 5322,
  ObjectRayOriginNV = 5323,
  ObjectRayDirectionNV = 5324,
  RayTminNV = 5325,
  RayTmaxNV = 5326,
  InstanceCustomIndexNV = 5327,
  ObjectToWorldNV = 5330,
  WorldToObjectNV = 5331,
  HitTNV = 5332,
  HitKindNV = 5333,
  IncomingRayFlagsNV = 5351,
};
StringRef getBuiltInName(BuiltIn e);

enum class SelectionControl : uint32_t {
  None = 0x0,
  Flatten = 0x1,
  DontFlatten = 0x2,
};
std::string getSelectionControlName(uint32_t e);

enum class LoopControl : uint32_t {
  None = 0x0,
  Unroll = 0x1,
  DontUnroll = 0x2,
  DependencyInfinite = 0x4,
  DependencyLength = 0x8,
  MinIterations = 0x10,
  MaxIterations = 0x20,
  IterationMultiple = 0x40,
  PeelCount = 0x80,
  PartialCount = 0x100,
};
std::string getLoopControlName(uint32_t e);

enum class FunctionControl : uint32_t {
  None = 0x0,
  Inline = 0x1,
  DontInline = 0x2,
  Pure = 0x4,
  Const = 0x8,
};
std::string getFunctionControlName(uint32_t e);

enum class MemorySemantics : uint32_t {
  None = 0x0,
  Acquire = 0x2,
  Release = 0x4,
  AcquireRelease = 0x8,
  SequentiallyConsistent = 0x10,
  UniformMemory = 0x40,
  SubgroupMemory = 0x80,
  WorkgroupMemory = 0x100,
  CrossWorkgroupMemory = 0x200,
  AtomicCounterMemory = 0x400,
  ImageMemory = 0x800,
  OutputMemoryKHR = 0x1000,
  MakeAvailableKHR = 0x2000,
  MakeVisibleKHR = 0x4000,
};
std::string getMemorySemanticsName(uint32_t e);

enum class MemoryOperand : uint32_t {
  None = 0x0,
  Volatile = 0x1,
  Aligned = 0x2,
  Nontemporal = 0x4,
  MakePointerAvailableKHR = 0x8,
  MakePointerVisibleKHR = 0x10,
  NonPrivatePointerKHR = 0x20,
};
std::string getMemoryOperandName(uint32_t e);

enum class Scope : uint32_t {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
  QueueFamilyKHR = 5,
};
StringRef getScopeName(Scope e);

enum class GroupOperation : uint32_t {
  Reduce = 0,
  InclusiveScan = 1,
  ExclusiveScan = 2,
  ClusteredReduce = 3,
  PartitionedReduceNV = 6,
  PartitionedInclusiveScanNV = 7,
  PartitionedExclusiveScanNV = 8,
};
StringRef getGroupOperationName(GroupOperation e);

enum class KernelEnqueueFlags : uint32_t {
  NoWait = 0,
  WaitKernel = 1,
  WaitWorkGroup = 2,
};
StringRef getKernelEnqueueFlagsName(KernelEnqueueFlags e);

enum class KernelProfilingInfo : uint32_t {
  None = 0x0,
  CmdExecTime = 0x1,
};
StringRef getKernelProfilingInfoName(KernelProfilingInfo e);
} // namespace SPIRV
} // namespace llvm

// Return a string representation of the operands from startIndex onwards.
// Templated to allow both MachineInstr and MCInst to use the same logic.
template <class InstType>
std::string getSPIRVStringOperand(const InstType &MI, unsigned StartIndex) {
  std::string s; // Iteratively append to this string.

  const unsigned NumOps = MI.getNumOperands();
  bool IsFinished = false;
  for (unsigned i = StartIndex; i < NumOps && !IsFinished; ++i) {
    const auto &Op = MI.getOperand(i);
    if (!Op.isImm()) // Stop if we hit a register operand.
      break;
    assert((Op.getImm() >> 32) == 0 && "Imm operand should be i32 word");
    const uint32_t Imm = Op.getImm(); // Each i32 word is up to 4 characters.
    for (unsigned ShiftAmount = 0; ShiftAmount < 32; ShiftAmount += 8) {
      char c = (Imm >> ShiftAmount) & 0xff;
      if (c == 0) { // Stop if we hit a null-terminator character.
        IsFinished = true;
        break;
      } else {
        s += c; // Otherwise, append the character to the result string.
      }
    }
  }
  return s;
}

#endif // LLVM_LIB_TARGET_SPIRV_MCTARGETDESC_SPIRVBASEINFO_H
