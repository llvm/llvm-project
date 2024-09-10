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
def ResRetInt16Ty : DXILOpParamType;
def ResRetInt32Ty : DXILOpParamType;
def HandleTy : DXILOpParamType;
def ResBindTy : DXILOpParamType;
def ResPropsTy : DXILOpParamType;

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
  def writeSamplerFeedbackLevel: DXILOpClass;

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

class DXILAttribute;

def ReadOnly : DXILAttribute;
def ReadNone : DXILAttribute;
def IsDerivative : DXILAttribute;
def IsGradient : DXILAttribute;
def IsFeedback : DXILAttribute;
def IsWave : DXILAttribute;
def NeedsUniformInputs : DXILAttribute;
def IsBarrier : DXILAttribute;

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
  list<DXILAttribute> op_attrs = attrs;
}

// Abstraction DXIL Operation
class DXILOp<int opcode, DXILOpClass opclass> {
  // A short description of the operation
  string Doc = "";

  // Opcode of DXIL Operation
  int OpCode = opcode;

  // Class of DXIL Operation.
  DXILOpClass OpClass = opclass;

  // LLVM Intrinsic DXIL Operation maps to
  Intrinsic LLVMIntrinsic = ?;

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
}

// Concrete definitions of DXIL Operations

def Abs :  DXILOp<6, unary> {
  let Doc = "Returns the absolute value of the input.";
  let LLVMIntrinsic = int_fabs;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Saturate :  DXILOp<7, unary> {
  let Doc = "Clamps a single or double precision floating point value to [0.0f...1.0f].";
  let LLVMIntrinsic = int_dx_saturate;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def IsInf :  DXILOp<9, isSpecialFloat> {
  let Doc = "Determines if the specified value is infinite.";
  let LLVMIntrinsic = int_dx_isinf;
  let arguments = [OverloadTy];
  let result = Int1Ty;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Cos :  DXILOp<12, unary> {
  let Doc = "Returns cosine(theta) for theta in radians.";
  let LLVMIntrinsic = int_cos;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Sin :  DXILOp<13, unary> {
  let Doc = "Returns sine(theta) for theta in radians.";
  let LLVMIntrinsic = int_sin;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Tan :  DXILOp<14, unary> {
  let Doc = "Returns tangent(theta) for theta in radians.";
  let LLVMIntrinsic = int_tan;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def ACos :  DXILOp<15, unary> {
  let Doc = "Returns the arccosine of the specified value.";
  let LLVMIntrinsic = int_acos;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def ASin :  DXILOp<16, unary> {
  let Doc = "Returns the arcsine of the specified value.";
  let LLVMIntrinsic = int_asin;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def ATan :  DXILOp<17, unary> {
  let Doc = "Returns the arctangent of the specified value.";
  let LLVMIntrinsic = int_atan;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def HCos :  DXILOp<18, unary> {
  let Doc = "Returns the hyperbolic cosine of the specified value.";
  let LLVMIntrinsic = int_cosh;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def HSin :  DXILOp<19, unary> {
  let Doc = "Returns the hyperbolic sine of the specified value.";
  let LLVMIntrinsic = int_sinh;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def HTan :  DXILOp<20, unary> {
  let Doc = "Returns the hyperbolic tan of the specified value.";
  let LLVMIntrinsic = int_tanh;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Exp2 :  DXILOp<21, unary> {
  let Doc = "Returns the base 2 exponential, or 2**x, of the specified value. "
            "exp2(x) = 2**x.";
  let LLVMIntrinsic = int_exp2;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Frac :  DXILOp<22, unary> {
  let Doc = "Returns a fraction from 0 to 1 that represents the decimal part "
            "of the input.";
  let LLVMIntrinsic = int_dx_frac;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Log2 :  DXILOp<23, unary> {
  let Doc = "Returns the base-2 logarithm of the specified value.";
  let LLVMIntrinsic = int_log2;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Sqrt :  DXILOp<24, unary> {
  let Doc = "Returns the square root of the specified floating-point value, "
            "per component.";
  let LLVMIntrinsic = int_sqrt;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def RSqrt :  DXILOp<25, unary> {
  let Doc = "Returns the reciprocal of the square root of the specified value. "
            "rsqrt(x) = 1 / sqrt(x).";
  let LLVMIntrinsic = int_dx_rsqrt;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Round :  DXILOp<26, unary> {
  let Doc = "Returns the input rounded to the nearest integer within a "
            "floating-point type.";
  let LLVMIntrinsic = int_roundeven;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Floor :  DXILOp<27, unary> {
  let Doc =
      "Returns the largest integer that is less than or equal to the input.";
  let LLVMIntrinsic = int_floor;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Ceil :  DXILOp<28, unary> {
  let Doc = "Returns the smallest integer that is greater than or equal to the "
            "input.";
  let LLVMIntrinsic = int_ceil;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Trunc :  DXILOp<29, unary> {
  let Doc = "Returns the specified value truncated to the integer component.";
  let LLVMIntrinsic = int_trunc;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Rbits :  DXILOp<30, unary> {
  let Doc = "Returns the specified value with its bits reversed.";
  let LLVMIntrinsic = int_bitreverse;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FMax :  DXILOp<35, binary> {
  let Doc = "Float maximum. FMax(a,b) = a > b ? a : b";
  let LLVMIntrinsic = int_maxnum;
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FMin :  DXILOp<36, binary> {
  let Doc = "Float minimum. FMin(a,b) = a < b ? a : b";
  let LLVMIntrinsic = int_minnum;
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def SMax :  DXILOp<37, binary> {
  let Doc = "Signed integer maximum. SMax(a,b) = a > b ? a : b";
  let LLVMIntrinsic = int_smax;
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def SMin :  DXILOp<38, binary> {
  let Doc = "Signed integer minimum. SMin(a,b) = a < b ? a : b";
  let LLVMIntrinsic = int_smin;
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def UMax :  DXILOp<39, binary> {
  let Doc = "Unsigned integer maximum. UMax(a,b) = a > b ? a : b";
  let LLVMIntrinsic = int_umax;
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def UMin :  DXILOp<40, binary> {
  let Doc = "Unsigned integer minimum. UMin(a,b) = a < b ? a : b";
  let LLVMIntrinsic = int_umin;
  let arguments = [OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FMad :  DXILOp<46, tertiary> {
  let Doc = "Floating point arithmetic multiply/add operation. fmad(m,a,b) = m "
            "* a + b.";
  let LLVMIntrinsic = int_fmuladd;
  let arguments = [OverloadTy, OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [HalfTy, FloatTy, DoubleTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def IMad :  DXILOp<48, tertiary> {
  let Doc = "Signed integer arithmetic multiply/add operation. imad(m,a,b) = m "
            "* a + b.";
  let LLVMIntrinsic = int_dx_imad;
  let arguments = [OverloadTy, OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def UMad :  DXILOp<49, tertiary> {
  let Doc = "Unsigned integer arithmetic multiply/add operation. umad(m,a, = m "
            "* a + b.";
  let LLVMIntrinsic = int_dx_umad;
  let arguments = [OverloadTy, OverloadTy, OverloadTy];
  let result = OverloadTy;
  let overloads =
      [Overloads<DXIL1_0, [Int16Ty, Int32Ty, Int64Ty]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Dot2 :  DXILOp<54, dot2> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is 0 to 1 inclusive";
  let LLVMIntrinsic = int_dx_dot2;
  let arguments = !listsplat(OverloadTy, 4);
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Dot3 :  DXILOp<55, dot3> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is 0 to 2 inclusive";
  let LLVMIntrinsic = int_dx_dot3;
  let arguments = !listsplat(OverloadTy, 6);
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [HalfTy, FloatTy]>];
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def Dot4 :  DXILOp<56, dot4> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is 0 to 3 inclusive";
  let LLVMIntrinsic = int_dx_dot4;
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
}

def ThreadId :  DXILOp<93, threadId> {
  let Doc = "Reads the thread ID";
  let LLVMIntrinsic = int_dx_thread_id;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [compute, mesh, amplification, node]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def GroupId :  DXILOp<94, groupId> {
  let Doc = "Reads the group ID (SV_GroupID)";
  let LLVMIntrinsic = int_dx_group_id;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [compute, mesh, amplification, node]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def ThreadIdInGroup :  DXILOp<95, threadIdInGroup> {
  let Doc = "Reads the thread ID within the group  (SV_GroupThreadID)";
  let LLVMIntrinsic = int_dx_thread_id_in_group;
  let arguments = [OverloadTy];
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [compute, mesh, amplification, node]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def FlattenedThreadIdInGroup :  DXILOp<96, flattenedThreadIdInGroup> {
  let Doc = "Provides a flattened index for a given thread within a given "
            "group (SV_GroupIndex)";
  let LLVMIntrinsic = int_dx_flattened_thread_id_in_group;
  let result = OverloadTy;
  let overloads = [Overloads<DXIL1_0, [Int32Ty]>];
  let stages = [Stages<DXIL1_0, [compute, mesh, amplification, node]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}

def AnnotateHandle : DXILOp<217, annotateHandle> {
  let Doc = "annotate handle with resource properties";
  let arguments = [HandleTy, ResPropsTy];
  let result = HandleTy;
  let stages = [Stages<DXIL1_6, [all_stages]>];
}

def CreateHandleFromBinding : DXILOp<218, createHandleFromBinding> {
  let Doc = "create resource handle from binding";
  let arguments = [ResBindTy, Int32Ty, Int1Ty];
  let result = HandleTy;
  let stages = [Stages<DXIL1_6, [all_stages]>];
}

def WaveIsFirstLane :  DXILOp<110, waveIsFirstLane> {
  let Doc = "returns 1 for the first lane in the wave";
  let LLVMIntrinsic = int_dx_wave_is_first_lane;
  let arguments = [];
  let result = Int1Ty;
  let stages = [Stages<DXIL1_0, [all_stages]>];
  let attributes = [Attributes<DXIL1_0, [ReadNone]>];
}
