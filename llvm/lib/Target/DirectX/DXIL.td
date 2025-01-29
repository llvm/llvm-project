//- DXIL.td - Describe DXIL operation -------------------------*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a target description file for DXIL operations.
///
//===----------------------------------------------------------------------===//

include "llvm/IR/Intrinsics.td"

// Abstract class to represent major and minor version values
class Version<int major, int minor> {
  int Major = major;
  int Minor = minor;
}

// Valid DXIL Version records
foreach i = 0...8 in {
  def DXIL1_ #i : Version<1, i>;
}

class DXILOpParamType {
  int isOverload = 0;
}

let isOverload = 1 in {
  def OverloadTy : DXILOpParamType;
}
def VoidTy : DXILOpParamType;
def Int1Ty : DXILOpParamType;
def Int8Ty : DXILOpParamType;
def Int16Ty : DXILOpParamType;
def Int32Ty : DXILOpParamType;
def Int64Ty : DXILOpParamType;
def HalfTy : DXILOpParamType;
def FloatTy : DXILOpParamType;
def DoubleTy : DXILOpParamType;
def ResRetHalfTy : DXILOpParamType;
def ResRetFloatTy : DXILOpParamType;
def ResRetDoubleTy : DXILOpParamType;
def ResRetInt16Ty : DXILOpParamType;
def ResRetInt32Ty : DXILOpParamType;
def ResRetInt64Ty : DXILOpParamType;
def HandleTy : DXILOpParamType;
def ResBindTy : DXILOpParamType;
def ResPropsTy : DXILOpParamType;
def SplitDoubleTy : DXILOpParamType;

class DXILOpClass;

defset list<DXILOpClass> OpClasses = {
  def acceptHitAndEndSearch : DXILOpClass;
  def allocateNodeOutputRecords : DXILOpClass;
  def allocateRayQuery : DXILOpClass;
  def annotateHandle : DXILOpClass;
  def annotateNodeHandle : DXILOpClass;
  def annotateNodeRecordHandle : DXILOpClass;
  def atomicBinOp : DXILOpClass;
  def atomicCompareExchange : DXILOpClass;
  def attributeAtVertex : DXILOpClass;
  def barrier : DXILOpClass;
  def barrierByMemoryHandle : DXILOpClass;
  def barrierByMemoryType : DXILOpClass;
  def barrierByNodeRecordHandle : DXILOpClass;
  def binary : DXILOpClass;
  def binaryWithCarryOrBorrow : DXILOpClass;
  def binaryWithTwoOuts : DXILOpClass;
  def bitcastF16toI16 : DXILOpClass;
  def bitcastF32toI32 : DXILOpClass;
  def bitcastF64toI64 : DXILOpClass;
  def bitcastI16toF16 : DXILOpClass;
  def bitcastI32toF32 : DXILOpClass;
  def bitcastI64toF64 : DXILOpClass;
  def bufferLoad : DXILOpClass;
  def bufferStore : DXILOpClass;
  def bufferUpdateCounter : DXILOpClass;
  def calculateLOD : DXILOpClass;
  def callShader : DXILOpClass;
  def cbufferLoad : DXILOpClass;
  def cbufferLoadLegacy : DXILOpClass;
  def checkAccessFullyMapped : DXILOpClass;
  def coverage : DXILOpClass;
  def createHandle : DXILOpClass;
  def createHandleForLib : DXILOpClass;
  def createHandleFromBinding : DXILOpClass;
  def createHandleFromHeap : DXILOpClass;
  def createNodeInputRecordHandle : DXILOpClass;
  def createNodeOutputHandle : DXILOpClass;
  def cutStream : DXILOpClass;
  def cycleCounterLegacy : DXILOpClass;
  def discard : DXILOpClass;
  def dispatchMesh : DXILOpClass;
  def dispatchRaysDimensions : DXILOpClass;
  def dispatchRaysIndex : DXILOpClass;
  def domainLocation : DXILOpClass;
  def dot2 : DXILOpClass;
  def dot2AddHalf : DXILOpClass;
  def dot3 : DXILOpClass;
  def dot4 : DXILOpClass;
  def dot4AddPacked : DXILOpClass;
  def emitIndices : DXILOpClass;
  def emitStream : DXILOpClass;
  def emitThenCutStream : DXILOpClass;
  def evalCentroid : DXILOpClass;
  def evalSampleIndex : DXILOpClass;
  def evalSnapped : DXILOpClass;
  def finishedCrossGroupSharing : DXILOpClass;
  def flattenedThreadIdInGroup : DXILOpClass;
  def geometryIndex : DXILOpClass;
  def getDimensions : DXILOpClass;
  def getInputRecordCount : DXILOpClass;
  def getMeshPayload : DXILOpClass;
  def getNodeRecordPtr : DXILOpClass;
  def getRemainingRecursionLevels : DXILOpClass;
  def groupId : DXILOpClass;
  def gsInstanceID : DXILOpClass;
  def hitKind : DXILOpClass;
  def ignoreHit : DXILOpClass;
  def incrementOutputCount : DXILOpClass;
  def indexNodeHandle : DXILOpClass;
  def innerCoverage : DXILOpClass;
  def instanceID : DXILOpClass;
  def instanceIndex : DXILOpClass;
  def isHelperLane : DXILOpClass;
  def isSpecialFloat : DXILOpClass;
  def legacyDoubleToFloat : DXILOpClass;
  def legacyDoubleToSInt32 : DXILOpClass;
  def legacyDoubleToUInt32 : DXILOpClass;
  def legacyF16ToF32 : DXILOpClass;
  def legacyF32ToF16 : DXILOpClass;
  def loadInput : DXILOpClass;
  def loadOutputControlPoint : DXILOpClass;
  def loadPatchConstant : DXILOpClass;
  def makeDouble : DXILOpClass;
  def minPrecXRegLoad : DXILOpClass;
  def minPrecXRegStore : DXILOpClass;
  def nodeOutputIsValid : DXILOpClass;
  def objectRayDirection : DXILOpClass;
  def objectRayOrigin : DXILOpClass;
  def objectToWorld : DXILOpClass;
  def outputComplete : DXILOpClass;
  def outputControlPointID : DXILOpClass;
  def pack4x8 : DXILOpClass;
  def primitiveID : DXILOpClass;
  def primitiveIndex : DXILOpClass;
  def quadOp : DXILOpClass;
  def quadReadLaneAt : DXILOpClass;
  def quadVote : DXILOpClass;
  def quaternary : DXILOpClass;
  def rawBufferLoad : DXILOpClass;
  def rawBufferStore : DXILOpClass;
  def rayFlags : DXILOpClass;
  def rayQuery_Abort : DXILOpClass;
  def rayQuery_CommitNonOpaqueTriangleHit : DXILOpClass;
  def rayQuery_CommitProceduralPrimitiveHit : DXILOpClass;
  def rayQuery_Proceed : DXILOpClass;
  def rayQuery_StateMatrix : DXILOpClass;
  def rayQuery_StateScalar : DXILOpClass;
  def rayQuery_StateVector : DXILOpClass;
  def rayQuery_TraceRayInline : DXILOpClass;
  def rayTCurrent : DXILOpClass;
  def rayTMin : DXILOpClass;
  def renderTargetGetSampleCount : DXILOpClass;
  def renderTargetGetSamplePosition : DXILOpClass;
  def reportHit : DXILOpClass;
  def sample : DXILOpClass;
  def sampleBias : DXILOpClass;
  def sampleCmp : DXILOpClass;
  def sampleCmpBias : DXILOpClass;
  def sampleCmpGrad : DXILOpClass;
  def sampleCmpLevel : DXILOpClass;
  def sampleCmpLevelZero : DXILOpClass;
  def sampleGrad : DXILOpClass;
  def sampleIndex : DXILOpClass;
  def sampleLevel : DXILOpClass;
  def setMeshOutputCounts : DXILOpClass;
  def splitDouble : DXILOpClass;
  def startInstanceLocation : DXILOpClass;
  def startVertexLocation : DXILOpClass;
  def storeOutput : DXILOpClass;
  def storePatchConstant : DXILOpClass;
  def storePrimitiveOutput : DXILOpClass;
  def storeVertexOutput : DXILOpClass;
  def tempRegLoad : DXILOpClass;
  def tempRegStore : DXILOpClass;
  def tertiary : DXILOpClass;
  def texture2DMSGetSamplePosition : DXILOpClass;
  def textureGather : DXILOpClass;
  def textureGatherCmp : DXILOpClass;
  def textureGatherRaw : DXILOpClass;
  def textureLoad : DXILOpClass;
  def textureStore : DXILOpClass;
  def textureStoreSample : DXILOpClass;
  def threadId : DXILOpClass;
  def threadIdInGroup : DXILOpClass;
  def traceRay : DXILOpClass;
  def unary : DXILOpClass;
  def unaryBits : DXILOpClass;
  def unpack4x8 : DXILOpClass;
  def viewID : DXILOpClass;
  def waveActiveAllEqual : DXILOpClass;
  def waveActiveBallot : DXILOpClass;
  def waveActiveBit : DXILOpClass;
  def waveActiveOp : DXILOpClass;
  def waveAllOp : DXILOpClass;
  def waveAllTrue : DXILOpClass;
  def waveAnyTrue : DXILOpClass;
  def waveGetLaneCount : DXILOpClass;
  def waveGetLaneIndex : DXILOpClass;
  def waveIsFirstLane : DXILOpClass;
  def waveMatch : DXILOpClass;
  def waveMatrix_Accumulate : DXILOpClass;
  def waveMatrix_Annotate : DXILOpClass;
  def waveMatrix_Depth : DXILOpClass;
  def waveMatrix_Fill : DXILOpClass;
  def waveMatrix_LoadGroupShared : DXILOpClass;
  def waveMatrix_LoadRawBuf : DXILOpClass;
  def waveMatrix_Multiply : DXILOpClass;
  def waveMatrix_ScalarOp : DXILOpClass;
  def waveMatrix_StoreGroupShared : DXILOpClass;
  def waveMatrix_StoreRawBuf : DXILOpClass;
  def waveMultiPrefixBitCount : DXILOpClass;
  def waveMultiPrefixOp : DXILOpClass;
  def wavePrefixOp : DXILOpClass;
  def waveReadLaneAt : DXILOpClass;
  def waveReadLaneFirst : DXILOpClass;
  def worldRayDirection : DXILOpClass;
  def worldRayOrigin : DXILOpClass;
  def worldToObject : DXILOpClass;
  def writeSamplerFeedback : DXILOpClass;
  def writeSamplerFeedbackBias : DXILOpClass;
  def writeSamplerFeedbackGrad : DXILOpClass;
  def writeSamplerFeedbackLevel : DXILOpClass;

  // This is a sentinel definition. Hence placed at the end here and
  // not as part of the above alphabetically sorted valid definitions.
  // It is never used to construct the name of DXIL Op call name.
  // Additionally it is capitalized unlike all the others.
  def UnknownOpClass : DXILOpClass;
}

class DXILShaderStage;

def compute : DXILShaderStage;
def domain : DXILShaderStage;
def hull : DXILShaderStage;
def pixel : DXILShaderStage;
def vertex : DXILShaderStage;
def geometry : DXILShaderStage;
def library : DXILShaderStage;
def amplification : DXILShaderStage;
def mesh : DXILShaderStage;
def node : DXILShaderStage;
def raygeneration : DXILShaderStage;
def intersection : DXILShaderStage;
def anyhit : DXILShaderStage;
def closesthit : DXILShaderStage;
def callable : DXILShaderStage;
def miss : DXILShaderStage;

// Pseudo-stages
// Denote DXIL Op to be supported in all stages
def all_stages : DXILShaderStage;
// Denote support for DXIL Op to have been removed
def removed : DXILShaderStage;

// DXIL Op attributes

// A function attribute denotes that there is a corresponding LLVM function
// attribute that will be set when building the DXIL op. The mapping is defined
// by setDXILAttributes in DXILOpBuilder.cpp
class DXILAttribute;

def ReadNone : DXILAttribute;
def ReadOnly : DXILAttribute;
def NoDuplicate : DXILAttribute;
def NoReturn : DXILAttribute;

// A property is simply used to mark that a DXIL op belongs to a sub-group of
// DXIL ops, and it is used to query if a particular op holds this property.
// This is used for the static analysis of DXIL ops.
class DXILProperty;

def IsBarrier : DXILProperty;
def IsGradient : DXILProperty;
def IsFeedback : DXILProperty;
def IsWave : DXILProperty;
def RequiresUniformInputs : DXILProperty;

class Overloads<Version ver, list<DXILOpParamType> ols> {
  Version dxil_version = ver;
  list<DXILOpParamType> overload_types = ols;
}

class Stages<Version ver, list<DXILShaderStage> st> {
  Version dxil_version = ver;
  list<DXILShaderStage> shader_stages = st;
}

class Attributes<Version ver = DXIL1_0, list<DXILAttribute> attrs> {
  Version dxil_version = ver;
  list<DXILAttribute> fn_attrs = attrs;
}

defvar BarrierMode_DeviceMemoryBarrier = 2;
defvar BarrierMode_DeviceMemoryBarrierWithGroupSync = 3;
defvar BarrierMode_GroupMemoryBarrier = 8;
defvar BarrierMode_GroupMemoryBarrierWithGroupSync = 9;
defvar BarrierMode_AllMemoryBarrier = 10;
defvar BarrierMode_AllMemoryBarrierWithGroupSync = 11;

defvar WaveOpKind_Sum = 0;
defvar WaveOpKind_Product = 1;
defvar WaveOpKind_Min = 2;
defvar WaveOpKind_Max = 3;

defvar SignedOpKind_Signed = 0;
defvar SignedOpKind_Unsigned = 1;

// Intrinsic arg selection
class IntrinArgSelectType;
def IntrinArgSelect_Index : IntrinArgSelectType;
def IntrinArgSelect_I8 : IntrinArgSelectType;
def IntrinArgSelect_I32 : IntrinArgSelectType;

class IntrinArgSelect<IntrinArgSelectType type_, int value_> {
  IntrinArgSelectType type = type_;
  int value = value_;
}

class IntrinArgIndex<int index> : IntrinArgSelect<IntrinArgSelect_Index, index>;
class IntrinArgI8<int value> : IntrinArgSelect<IntrinArgSelect_I8, value>;
class IntrinArgI32<int value> : IntrinArgSelect<IntrinArgSelect_I32, value>;

// Select which intrinsic to lower from for a DXILOp.
// If the intrinsic is the only argument given to IntrinSelect, then the
// arguments of the intrinsic will be copied in the same order. Example:
//   let intrinsics = [
//     IntrinSelect<int_dx_my_intrinsic>,
//     IntrinSelect<int_dx_my_intrinsic2>,
//   ]
//=========================================================================================
// Using IntrinArgIndex<>, arguments of the intrinsic can be copied to the DXIL
// OP in specific order:
//   let intrinsics = [
//     IntrinSelect<int_dx_my_intrinsic,
//       [IntrinArgIndex<2>, IntrinArgIndex<1>, IntrinArgIndex<0>> ]
//     >,
//   ]
//=========================================================================================
// Using IntrinArgI8<> and IntrinArgI32<>, integer constants can be added
// directly to the dxil op. This can be used in conjunction with
// IntrinArgIndex:
//   let intrinsics = [
//     IntrinSelect<int_dx_wave_active_usum,
//       [ IntrinArgIndex<0>, IntrinArgI8<0>, IntrinArgI8<1> ]
//     >,
//     IntrinSelect<int_dx_wave_reduce_sum,
//       [ IntrinArgIndex<0>, IntrinArgI8<0>, IntrinArgI8<0> ]
//     >,
//   ]
//
class IntrinSelect<Intrinsic intrinsic_,
                   list<IntrinArgSelect> arg_selects_ = []> {
  Intrinsic intrinsic = intrinsic_;
  list<IntrinArgSelect> arg_selects = arg_selects_;
}

// Abstraction DXIL Operation
class DXILOp<int opcode, DXILOpClass opclass> {
  // A short description of the operation
  string Doc = "";

  // Opcode of DXIL Operation
  int OpCode = opcode;

  // Class of DXIL Operation.
  DXILOpClass OpClass = opclass;

  // LLVM Intrinsics DXIL Operation maps from
  list<IntrinSelect> intrinsics = [];

  // Result type of the op
  DXILOpParamType result;

  // List of argument types of the op. Default to 0 arguments.
  list<DXILOpParamType> arguments = [];

  // List of valid overload types predicated by DXIL version
  list<Overloads> overloads = [];

  // List of valid shader stages predicated by DXIL version
  list<Stages> stages;

  // Versioned attributes of operation
  list<Attributes> attributes = [];

  // List of properties. Default to no properties.
  list<DXILProperty> properties = [];
}

// Concrete definitions of DXIL Operations
//
// This are sorted by ascending value of the DXIL Opcodes

def Abs : DXILOp<6, unary> {
  let Doc = "Returns the absolute value of the input.";
  let intrinsics = [IntrinSelect<int_fabs>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Saturate : DXILOp<7, unary> {
  let Doc = "Clamps a single or double precision floating point value to "
            "[0.0f...1.0f].";
  let intrinsics = [IntrinSelect<int_dx_saturate>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def IsInf : DXILOp<9, isSpecialFloat> {
  let Doc = "Determines if the specified value is infinite.";
  let intrinsics = [IntrinSelect<int_dx_isinf>];
  let arguments = [OverloadTy];
  let result = Int1Ty;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Cos : DXILOp<12, unary> {
  let Doc = "Returns cosine(theta) for theta in radians.";
  let intrinsics = [IntrinSelect<int_cos>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Sin : DXILOp<13, unary> {
  let Doc = "Returns sine(theta) for theta in radians.";
  let intrinsics = [IntrinSelect<int_sin>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Tan : DXILOp<14, unary> {
  let Doc = "Returns tangent(theta) for theta in radians.";
  let intrinsics = [IntrinSelect<int_tan>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def ACos : DXILOp<15, unary> {
  let Doc = "Returns the arccosine of the specified value.";
  let intrinsics = [IntrinSelect<int_acos>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def ASin : DXILOp<16, unary> {
  let Doc = "Returns the arcsine of the specified value.";
  let intrinsics = [IntrinSelect<int_asin>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def ATan : DXILOp<17, unary> {
  let Doc = "Returns the arctangent of the specified value.";
  let intrinsics = [IntrinSelect<int_atan>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def HCos : DXILOp<18, unary> {
  let Doc = "Returns the hyperbolic cosine of the specified value.";
  let intrinsics = [IntrinSelect<int_cosh>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def HSin : DXILOp<19, unary> {
  let Doc = "Returns the hyperbolic sine of the specified value.";
  let intrinsics = [IntrinSelect<int_sinh>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def HTan : DXILOp<20, unary> {
  let Doc = "Returns the hyperbolic tan of the specified value.";
  let intrinsics = [IntrinSelect<int_tanh>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Exp2 : DXILOp<21, unary> {
  let Doc = "Returns the base 2 exponential, or 2**x, of the specified value. "
            "exp2(x) = 2**x.";
  let intrinsics = [IntrinSelect<int_exp2>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Frac : DXILOp<22, unary> {
  let Doc = "Returns a fraction from 0 to 1 that represents the decimal part "
            "of the input.";
  let intrinsics = [IntrinSelect<int_dx_frac>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Log2 : DXILOp<23, unary> {
  let Doc = "Returns the base-2 logarithm of the specified value.";
  let intrinsics = [IntrinSelect<int_log2>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Sqrt : DXILOp<24, unary> {
  let Doc = "Returns the square root of the specified floating-point value, "
            "per component.";
  let intrinsics = [IntrinSelect<int_sqrt>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def RSqrt : DXILOp<25, unary> {
  let Doc = "Returns the reciprocal of the square root of the specified value. "
            "rsqrt(x) = 1 / sqrt(x).";
  let intrinsics = [IntrinSelect<int_dx_rsqrt>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Round : DXILOp<26, unary> {
  let Doc = "Returns the input rounded to the nearest integer within a "
            "floating-point type.";
  let intrinsics = [IntrinSelect<int_roundeven>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Floor : DXILOp<27, unary> {
  let Doc =
      "Returns the largest integer that is less than or equal to the input.";
  let intrinsics = [IntrinSelect<int_floor>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Ceil : DXILOp<28, unary> {
  let Doc = "Returns the smallest integer that is greater than or equal to the "
            "input.";
  let intrinsics = [IntrinSelect<int_ceil>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Trunc : DXILOp<29, unary> {
  let Doc = "Returns the specified value truncated to the integer component.";
  let intrinsics = [IntrinSelect<int_trunc>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Rbits : DXILOp<30, unary> {
  let Doc = "Returns the specified value with its bits reversed.";
  let intrinsics = [IntrinSelect<int_bitreverse>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def CountBits : DXILOp<31, unaryBits> {
  let Doc = "Returns the number of 1 bits in the specified value.";
  let arguments = [OverloadTy];
  let result = Int32Ty;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FirstbitLo : DXILOp<32, unaryBits> {
  let Doc = "Returns the location of the first set bit starting from "
            "the lowest order bit and working upward.";
  let intrinsics = [IntrinSelect<int_dx_firstbitlow>];
  let arguments = [OverloadTy];
  let result = Int32Ty;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FirstbitHi : DXILOp<33, unaryBits> {
  let Doc = "Returns the location of the first set bit starting from "
            "the highest order bit and working downward.";
  let intrinsics = [IntrinSelect<int_dx_firstbituhigh>];
  let arguments = [OverloadTy];
  let result = Int32Ty;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FirstbitSHi : DXILOp<34, unaryBits> {
  let Doc = "Returns the location of the first set bit from "
            "the highest order bit based on the sign.";
  let intrinsics = [IntrinSelect<int_dx_firstbitshigh>];
  let arguments = [OverloadTy];
  let result = Int32Ty;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FMax : DXILOp<35, binary> {
  let Doc = "Float maximum. FMax(a,b) = a > b ? a : b";
  let intrinsics = [IntrinSelect<int_maxnum>];
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FMin : DXILOp<36, binary> {
  let Doc = "Float minimum. FMin(a,b) = a < b ? a : b";
  let intrinsics = [IntrinSelect<int_minnum>];
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def SMax : DXILOp<37, binary> {
  let Doc = "Signed integer maximum. SMax(a,b) = a > b ? a : b";
  let intrinsics = [IntrinSelect<int_smax>];
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def SMin : DXILOp<38, binary> {
  let Doc = "Signed integer minimum. SMin(a,b) = a < b ? a : b";
  let intrinsics = [IntrinSelect<int_smin>];
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def UMax : DXILOp<39, binary> {
  let Doc = "Unsigned integer maximum. UMax(a,b) = a > b ? a : b";
  let intrinsics = [IntrinSelect<int_umax>];
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def UMin : DXILOp<40, binary> {
  let Doc = "Unsigned integer minimum. UMin(a,b) = a < b ? a : b";
  let intrinsics = [IntrinSelect<int_umin>];
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FMad : DXILOp<46, tertiary> {
  let Doc = "Floating point arithmetic multiply/add operation. fmad(m,a,b) = m "
            "* a + b.";
  let intrinsics = [IntrinSelect<int_fmuladd>];
  let arguments = [OverloadTy, OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def IMad : DXILOp<48, tertiary> {
  let Doc = "Signed integer arithmetic multiply/add operation. imad(m,a,b) = m "
            "* a + b.";
  let intrinsics = [IntrinSelect<int_dx_imad>];
  let arguments = [OverloadTy, OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def UMad : DXILOp<49, tertiary> {
  let Doc = "Unsigned integer arithmetic multiply/add operation. umad(m,a, = m "
            "* a + b.";
  let intrinsics = [IntrinSelect<int_dx_umad>];
  let arguments = [OverloadTy, OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Dot2 : DXILOp<54, dot2> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is 0 to 1 inclusive";
  let intrinsics = [IntrinSelect<int_dx_dot2>];
  let arguments = !listsplat(OverloadTy, 4);
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Dot3 : DXILOp<55, dot3> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is 0 to 2 inclusive";
  let intrinsics = [IntrinSelect<int_dx_dot3>];
  let arguments = !listsplat(OverloadTy, 6);
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Dot4 : DXILOp<56, dot4> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is 0 to 3 inclusive";
  let intrinsics = [IntrinSelect<int_dx_dot4>];
  let arguments = !listsplat(OverloadTy, 8);
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def CreateHandle : DXILOp<57, createHandle> {
  let Doc = "creates the handle to a resource";
  // ResourceClass, RangeID, Index, NonUniform
  let arguments = [Int8Ty, Int32Ty, Int32Ty, Int1Ty];
  let result = HandleTy;
  let stages = [Stages<DXIL1_0, [all_stages]>, Stages<DXIL1_6, [removed]>];
  // NOTE: The ReadOnly attribute was set for consistency with DXC. However, it
  // seems like ReadNone may more appropiately describe it. So noted to
  // consider a change in the future
  let attributes = [Attributes<DXIL1_0, [ReadOnly]>];
}

def BufferLoad : DXILOp<68, bufferLoad> {
  let Doc = "reads from a TypedBuffer";
  // Handle, Coord0, Coord1
  let arguments = [HandleTy, Int32Ty, Int32Ty];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0,
                 [ResRetHalfTy, ResRetFloatTy, ResRetInt16Ty, ResRetInt32Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadOnly]>];
}

def BufferStore : DXILOp<69, bufferStore> {
  let Doc = "writes to an RWTypedBuffer";
  // Handle, Coord0, Coord1, Val0, Val1, Val2, Val3, Mask
  let arguments = [
    HandleTy, Int32Ty, Int32Ty, OverloadTy, OverloadTy, OverloadTy, OverloadTy,
    Int8Ty
  ];
  let result = VoidTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy, Int16Ty, Int32Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
}

def UpdateCounter : DXILOp<70, bufferUpdateCounter> {
  let Doc = "increments/decrements a buffer counter";
  let arguments = [HandleTy, Int8Ty];
  let result = Int32Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
}

def CheckAccessFullyMapped : DXILOp<71, checkAccessFullyMapped> {
  let Doc = "checks whether a Sample, Gather, or Load operation "
            "accessed mapped tiles in a tiled resource";
  let arguments = [OverloadTy];
  let result = Int1Ty;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadOnly]>];
}

def Barrier : DXILOp<80, barrier> {
  let Doc = "inserts a memory barrier in the shader";
  let intrinsics = [
    IntrinSelect<int_dx_group_memory_barrier_with_group_sync,
                 [IntrinArgI32<BarrierMode_GroupMemoryBarrierWithGroupSync>]>,
  ];

  let arguments = [Int32Ty];
  let result = VoidTy;
  let stages = [Stages<DXIL1_0, [compute, library]>];
  let attributes = [Attributes<DXIL1_0, []>];
  let properties = [IsBarrier];
}

def Discard : DXILOp<82, discard> {
  let Doc = "discard the current pixel";
  let intrinsics = [IntrinSelect<int_dx_discard>];
  let arguments = [Int1Ty];
  let result = VoidTy;
  let stages = [Stages<DXIL1_0, [pixel]>];
}

def ThreadId : DXILOp<93, threadId> {
  let Doc = "Reads the thread ID";
  let intrinsics = [IntrinSelect<int_dx_thread_id>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [compute, mesh, amplification, node]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def GroupId : DXILOp<94, groupId> {
  let Doc = "Reads the group ID (SV_GroupID)";
  let intrinsics = [IntrinSelect<int_dx_group_id>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [compute, mesh, amplification, node]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def ThreadIdInGroup : DXILOp<95, threadIdInGroup> {
  let Doc = "Reads the thread ID within the group  (SV_GroupThreadID)";
  let intrinsics = [IntrinSelect<int_dx_thread_id_in_group>];
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [compute, mesh, amplification, node]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FlattenedThreadIdInGroup : DXILOp<96, flattenedThreadIdInGroup> {
  let Doc = "Provides a flattened index for a given thread within a given "
            "group (SV_GroupIndex)";
  let intrinsics = [IntrinSelect<int_dx_flattened_thread_id_in_group>];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [compute, mesh, amplification, node]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def MakeDouble : DXILOp<101, makeDouble> {
  let Doc = "creates a double value";
  let intrinsics = [IntrinSelect<int_dx_asdouble>];
  let arguments = [Int32Ty, Int32Ty];
  let result = DoubleTy;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def SplitDouble : DXILOp<102, splitDouble> {
  let Doc = "Splits a double into 2 uints";
  let arguments = [OverloadTy];
  let result = SplitDoubleTy;
  let overloads = [Overloads<DXIL1_0, [DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def WaveIsFirstLane : DXILOp<110, waveIsFirstLane> {
  let Doc = "returns 1 for the first lane in the wave";
  let intrinsics = [IntrinSelect<int_dx_wave_is_first_lane>];
  let arguments = [];
  let result = Int1Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let properties = [IsWave];
}

def WaveGetLaneIndex : DXILOp<111, waveGetLaneIndex> {
  let Doc = "returns the index of the current lane in the wave";
  let intrinsics = [IntrinSelect<int_dx_wave_getlaneindex>];
  let arguments = [];
  let result = Int32Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadOnly]>];
  let properties = [IsWave];
}

def WaveActiveAnyTrue : DXILOp<113, waveAnyTrue> {
  let Doc = "returns true if the expression is true in any of the active lanes "
            "in the current wave";
  let intrinsics = [IntrinSelect<int_dx_wave_any>];
  let arguments = [Int1Ty];
  let result = Int1Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let properties = [IsWave];
}

def WaveActiveAllTrue : DXILOp<114, waveAllTrue> {
  let Doc = "returns true if the expression is true in all of the active lanes "
            "in the current wave";
  let intrinsics = [IntrinSelect<int_dx_wave_all>];
  let arguments = [Int1Ty];
  let result = Int1Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let properties = [IsWave];
}

def WaveReadLaneAt : DXILOp<117, waveReadLaneAt> {
  let Doc = "returns the value from the specified lane";
  let intrinsics = [IntrinSelect<int_dx_wave_readlane>];
  let arguments = [OverloadTy, Int32Ty];
  let result = OverloadTy;
  let overloads = [Overloads<
      DXIL1_0, [HalfTy, FloatTy, DoubleTy, Int1Ty, Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let properties = [IsWave];
}

def WaveActiveOp : DXILOp<119, waveActiveOp> {
  let Doc = "returns the result of the operation across waves";
  let intrinsics = [
    IntrinSelect<int_dx_wave_reduce_sum,
                 [
                   IntrinArgIndex<0>, IntrinArgI8<WaveOpKind_Sum>,
                   IntrinArgI8<SignedOpKind_Signed>
                 ]>,
    IntrinSelect<int_dx_wave_reduce_usum,
                 [
                   IntrinArgIndex<0>, IntrinArgI8<WaveOpKind_Sum>,
                   IntrinArgI8<SignedOpKind_Unsigned>
                 ]>,
    IntrinSelect<int_dx_wave_reduce_max,
                 [
                   IntrinArgIndex<0>, IntrinArgI8<WaveOpKind_Max>,
                   IntrinArgI8<SignedOpKind_Signed>
                 ]>,
    IntrinSelect<int_dx_wave_reduce_umax,
                 [
                   IntrinArgIndex<0>, IntrinArgI8<WaveOpKind_Max>,
                   IntrinArgI8<SignedOpKind_Unsigned>
                 ]>,
  ];

  let arguments = [OverloadTy, Int8Ty, Int8Ty];
  let result = OverloadTy;
  let overloads = [
    Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy, Int16Ty, Int32Ty, Int64Ty]>
  ];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, []>];
}

def WaveAllBitCount : DXILOp<135, waveAllOp> {
  let Doc = "returns the count of bits set to 1 across the wave";
  let intrinsics = [IntrinSelect<int_dx_wave_active_countbits>];
  let arguments = [Int1Ty];
  let result = Int32Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let properties = [IsWave];
}

def RawBufferLoad : DXILOp<139, rawBufferLoad> {
  let Doc = "reads from a raw buffer and structured buffer";
  // Handle, Coord0, Coord1, Mask, Alignment
  let arguments = [HandleTy, Int32Ty, Int32Ty, Int8Ty, Int32Ty];
  let result = OverloadTy;
  let overloads = [
    Overloads<DXIL1_2,
              [ResRetHalfTy, ResRetFloatTy, ResRetInt16Ty, ResRetInt32Ty]>,
    Overloads<DXIL1_3,
              [
                ResRetHalfTy, ResRetFloatTy, ResRetDoubleTy, ResRetInt16Ty,
                ResRetInt32Ty, ResRetInt64Ty
              ]>
  ];
  let stages = [Stages<DXIL1_2, [all_stages]>];
}

def RawBufferStore : DXILOp<140, rawBufferStore> {
  let Doc = "writes to a RWByteAddressBuffer or RWStructuredBuffer";
  // Handle, Coord0, Coord1, Val0, Val1, Val2, Val3, Mask, Alignment
  let arguments = [
    HandleTy, Int32Ty, Int32Ty, OverloadTy, OverloadTy, OverloadTy, OverloadTy,
    Int8Ty, Int32Ty
  ];
  let result = VoidTy;
  let overloads = [
    Overloads<DXIL1_2,
              [ResRetHalfTy, ResRetFloatTy, ResRetInt16Ty, ResRetInt32Ty]>,
    Overloads<DXIL1_3,
              [
                ResRetHalfTy, ResRetFloatTy, ResRetDoubleTy, ResRetInt16Ty,
                ResRetInt32Ty, ResRetInt64Ty
              ]>
  ];
  let stages = [Stages<DXIL1_2, [all_stages]>];
}

def Dot4AddI8Packed : DXILOp<163, dot4AddPacked> {
  let Doc = "signed dot product of 4 x i8 vectors packed into i32, with "
            "accumulate to i32";
  let intrinsics = [IntrinSelect<int_dx_dot4add_i8packed>];
  let arguments = [Int32Ty, Int32Ty, Int32Ty];
  let result = Int32Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Dot4AddU8Packed : DXILOp<164, dot4AddPacked> {
  let Doc = "unsigned dot product of 4 x i8 vectors packed into i32, with "
            "accumulate to i32";
  let intrinsics = [IntrinSelect<int_dx_dot4add_u8packed>];
  let arguments = [Int32Ty, Int32Ty, Int32Ty];
  let result = Int32Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def AnnotateHandle : DXILOp<216, annotateHandle> {
  let Doc = "annotate handle with resource properties";
  let arguments = [HandleTy, ResPropsTy];
  let result = HandleTy;
  let stages = [Stages<DXIL1_6, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def CreateHandleFromBinding : DXILOp<217, createHandleFromBinding> {
  let Doc = "create resource handle from binding";
  let arguments = [ResBindTy, Int32Ty, Int1Ty];
  let result = HandleTy;
  let stages = [Stages<DXIL1_6, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}
