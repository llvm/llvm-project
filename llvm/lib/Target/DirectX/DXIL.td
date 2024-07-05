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

// Valid Shader model version records

// Shader Model 6.0 - 6.8 and DXIL Version 1.0 - 1.8
foreach i = 0...8 in {
  def SM6_#i : Version<6, i>;
  def DX1_#i : Version<1, i>;
}

// Resource ValueType - has no size or value
def resourceVT : ValueType<-1, -1>;

// Resource type
def dxil_resource_ty : LLVMType<resourceVT>;
// Overload type alias of llvm_any_ty
defvar dxil_overload_ty  = llvm_any_ty;

// DXIL Op attribute
class DXILOpAttr;

defset list<DXILOpAttr> OpAttributes = {
  def ReadOnly : DXILOpAttr;
  def ReadNone : DXILOpAttr;
  def IsDerivative : DXILOpAttr;
  def IsGradient : DXILOpAttr;
  def IsFeedback : DXILOpAttr;
  def IsWave : DXILOpAttr;
  def NeedsUniformInputs : DXILOpAttr;
  def IsBarrier : DXILOpAttr;
}

class DXILOpClass;

// Concrete definitions of DXIL Op Classes
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

  // This is a sentinel definition. Hence placed at the end of the list
  // and not as part of the above alphabetically sorted valid definitions.
  // Additionally it is capitalized unlike all the others.
  def UnknownOpClass: DXILOpClass;
}

// Shader stages
class ShaderStage;

defset list<ShaderStage> ShaderStages = {
  def compute : ShaderStage;
  def domain : ShaderStage;
  def hull : ShaderStage;
  def pixel : ShaderStage;
  def vertex : ShaderStage;
  def geometry : ShaderStage;
  def library : ShaderStage;
  def amplification : ShaderStage;
  def mesh : ShaderStage;
  def node : ShaderStage;
  def raygeneration : ShaderStage;
  def intersection : ShaderStage;
  def allKinds : ShaderStage;
}

// Primitive predicate
class Pred;

// Shader Model version predicate. This translates to
// a check for specified shader model version
class SMVersion<Version ver> : Pred {
  Version sm_version = ver;
}

// Class abstraction of constraints predicated on Shader Model version
class SMVersionConstraints<Version ver, dag oloads, dag stages> : SMVersion<ver> {
  dag overload_types = oloads;
  dag stage_kinds = stages;
}

// Marker used to identify argument list.
def ins;

// Marker used to identify result list.
def out;

// Marker used to identify list of shader model based attributes.
def sm_attrs;

// Marker used to identify overload types list.
def overloads;

// Marker used to identify stage kinds list.
def stages;

// Marker used to identify attribute list.
def attrs;

// Abstraction DXIL Operation
class DXILOp {
  // A short description of the operation
  string Doc = "";

  // Opcode of DXIL Operation
  int OpCode = 0;

  // Class of DXIL Operation.
  DXILOpClass OpClass = UnknownOpClass;

  // LLVM Intrinsic DXIL Operation maps to
  Intrinsic LLVMIntrinsic = ?;

  // Dag containing the arguments of the op. Default to 0 arguments.
  dag arguments = (ins);

  // Results of the op. Default to 0 results.
  dag result = (out);

  // List of constraints predicated on Shader Model version
  // This field is required to be specified. If a DXIL Op has no
  // overloads or stages predicated on Shader Model version, the
  // minimum Shader Model version the DXIL Op is supported it
  // should be specified as a single list item
  //       [SMVersionConstraints<SMX_Y, (overloads), (stages allKinds)]
  // If the DXIL Op is a DXIL Op that is not predicted on Shader
  // Model version,  it should be specified as an empty list.

  list<SMVersionConstraints> sm_constraints;

  // Non-predicated operation attributes
  dag attributes = (attrs);
  Version DXILVersion = ?;
}

// Concrete definitions of DXIL Operations

def IsInf : DXILOp {
  let Doc = "Determines if the specified value is infinite.";
  let OpCode = 9;
  let OpClass = isSpecialFloat;
  let LLVMIntrinsic = int_dx_isinf;
  let arguments = (ins llvm_anyfloat_ty);
  let result = (out llvm_i1_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Abs : DXILOp {
  let Doc = "Returns the absolute value of the input.";
  let OpCode = 6;
  let OpClass = unary;
  let LLVMIntrinsic = int_fabs;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty, llvm_double_ty),
                             (stages allKinds)>];
}

def Cos  : DXILOp {
  let Doc ="Returns cosine(theta) for theta in radians.";
  let OpCode = 12;
  let OpClass = unary;
  let LLVMIntrinsic = int_cos;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Sin  : DXILOp {
  let Doc ="Returns sine(theta) for theta in radians.";
  let OpCode = 13;
  let OpClass = unary;
  let LLVMIntrinsic = int_sin;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                          (overloads llvm_half_ty, llvm_float_ty),
                          (stages allKinds)>];
  let attributes = (attrs ReadNone);
  let DXILVersion = DX1_0;
}

def Tan  : DXILOp {
  let Doc = "Returns tangent(theta) for theta in radians.";
  let OpCode = 14;
  let OpClass = unary;
  let LLVMIntrinsic = int_tan;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def ACos  : DXILOp {
  let Doc = "Returns the arccosine of the specified value.";
  let OpCode = 15;
  let OpClass = unary;
  let LLVMIntrinsic = int_acos;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def ASin  : DXILOp {
  let Doc = "Returns the arcsine of the specified value.";
  let OpCode = 16;
  let OpClass = unary;
  let LLVMIntrinsic = int_asin;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def ATan  : DXILOp {
  let Doc = "Returns the arctangent of the specified value.";
  let OpCode = 17;
  let OpClass = unary;
  let LLVMIntrinsic = int_atan;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def HCos  : DXILOp {
  let Doc = "Returns the hyperbolic cosine of the specified value.";
  let OpCode = 18;
  let OpClass = unary;
  let LLVMIntrinsic = int_cosh;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def HSin  : DXILOp {
  let Doc = "Returns the hyperbolic sine of the specified value.";
  let OpCode = 19;
  let OpClass = unary;
  let LLVMIntrinsic = int_sinh;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def HTan  : DXILOp {
  let Doc = "Returns the hyperbolic tan of the specified value.";
  let OpCode = 20;
  let OpClass = unary;
  let LLVMIntrinsic = int_tanh;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Exp2 : DXILOp {
  let Doc = "Returns the base 2 exponential, or 2**x, of the specified value. exp2(x) = 2**x.";
  let OpCode = 21;
  let OpClass = unary;
  let LLVMIntrinsic = int_exp2;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Frac : DXILOp {
  let Doc = "Returns a fraction from 0 to 1 that represents the decimal part of the input.";
  let OpCode = 22;
  let OpClass = unary;
  let LLVMIntrinsic = int_dx_frac;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Log2 : DXILOp {
  let Doc = "Returns the base-2 logarithm of the specified value.";
  let OpCode = 23;
  let OpClass = unary;
  let LLVMIntrinsic = int_log2;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Sqrt : DXILOp {
  let Doc = "Returns the square root of the specified floating-point value, per component.";
  let OpCode = 24;
  let OpClass = unary;
  let LLVMIntrinsic = int_sqrt;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def RSqrt : DXILOp {
  let Doc = "Returns the reciprocal of the square root of the specified value. rsqrt(x) = 1 / sqrt(x).";
  let OpCode = 25;
  let OpClass = unary;
  let LLVMIntrinsic = int_dx_rsqrt;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Round : DXILOp {
  let Doc = "Returns the input rounded to the nearest integer within a floating-point type.";
  let OpCode = 26;
  let OpClass = unary;
  let LLVMIntrinsic = int_roundeven;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Floor : DXILOp {
  let Doc = "Returns the largest integer that is less than or equal to the input.";
  let OpCode = 27;
  let OpClass = unary;
  let LLVMIntrinsic = int_floor;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Ceil  : DXILOp {
  let Doc = "Returns the smallest integer that is greater than or equal to the input.";
  let OpCode = 28;
  let OpClass = unary;
  let LLVMIntrinsic = int_ceil;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Trunc : DXILOp {
  let Doc = "Returns the specified value truncated to the integer component.";
  let OpCode = 29;
  let OpClass = unary;
  let LLVMIntrinsic = int_trunc;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Rbits : DXILOp {
  let Doc = "Returns the specified value with its bits reversed.";
  let OpCode = 30;
  let OpClass = unary;
  let LLVMIntrinsic = int_bitreverse;
  let arguments = (ins LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_i16_ty, llvm_i32_ty, llvm_i64_ty),
                             (stages allKinds)>];
}

def FMax : DXILOp {
  let Doc = "Float maximum. FMax(a,b) = a > b ? a : b";
  let OpCode = 35;
  let OpClass = binary;
  let LLVMIntrinsic = int_maxnum;
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty, llvm_double_ty),
                             (stages allKinds)>];
}

def FMin : DXILOp {
  let Doc = "Float minimum. FMin(a,b) = a < b ? a : b";
  let OpCode = 36;
  let OpClass = binary;
  let LLVMIntrinsic = int_minnum;
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty, llvm_double_ty),
                             (stages allKinds)>];
}

def SMax : DXILOp {
  let Doc = "Signed integer maximum. SMax(a,b) = a > b ? a : b";
  let OpCode = 37;
  let OpClass = binary;
  let LLVMIntrinsic = int_smax;
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_i16_ty, llvm_i32_ty, llvm_i64_ty),
                             (stages allKinds)>];
}

def SMin : DXILOp {
  let Doc = "Signed integer minimum. SMin(a,b) = a < b ? a : b";
  let OpCode = 38;
  let OpClass = binary;
  let LLVMIntrinsic = int_smin;
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_i16_ty, llvm_i32_ty, llvm_i64_ty),
                             (stages allKinds)>];
}

def UMax : DXILOp {
  let Doc = "Unsigned integer maximum. UMax(a,b) = a > b ? a : b";
  let OpCode = 39;
  let OpClass = binary;
  let LLVMIntrinsic = int_umax;
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_i16_ty, llvm_i32_ty, llvm_i64_ty),
                             (stages allKinds)>];
}

def UMin : DXILOp {
  let Doc = "Unsigned integer minimum. UMin(a,b) = a < b ? a : b";
  let OpCode = 40;
  let OpClass = binary;
  let LLVMIntrinsic = int_umin;
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_i16_ty, llvm_i32_ty, llvm_i64_ty),
                             (stages allKinds)>];
}

def FMad : DXILOp {
  let Doc = "Floating point arithmetic multiply/add operation. fmad(m,a,b) = m * a + b.";
  let OpCode = 46;
  let OpClass = tertiary;
  let LLVMIntrinsic = int_fmuladd;
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty, llvm_double_ty),
                             (stages allKinds)>];
}

def IMad : DXILOp {
  let OpCode = 48;
  let OpClass = tertiary;
  let LLVMIntrinsic = int_dx_imad;
  let Doc = "Signed integer arithmetic multiply/add operation. imad(m,a,b) = m * a + b.";
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_i16_ty, llvm_i32_ty, llvm_i64_ty),
                             (stages allKinds)>];
}

def UMad : DXILOp {
  let OpCode = 49;
  let OpClass = tertiary;
  let LLVMIntrinsic = int_dx_umad;
  let Doc = "Unsigned integer arithmetic multiply/add operation. umad(m,a, = m * a + b.";
  let arguments = (ins LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>);
  let result = (out dxil_overload_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_i16_ty, llvm_i32_ty, llvm_i64_ty),
                             (stages allKinds)>];
}

def Dot2 : DXILOp {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + a[n]*b[n] where n is between 0 and 1";
  let OpCode = 54;
  let OpClass = dot2;
  let LLVMIntrinsic = int_dx_dot2;
  let arguments = !dag(ins, !listsplat(llvm_anyfloat_ty, 4), !listsplat("", 4));
  let result = (out llvm_anyfloat_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Dot3 : DXILOp {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + a[n]*b[n] where n is between 0 and 2";
  let OpCode = 55;
  let OpClass = dot3;
  let LLVMIntrinsic = int_dx_dot3;
  let arguments = !dag(ins, !listsplat(llvm_anyfloat_ty, 6), !listsplat("", 6));
  let result = (out llvm_anyfloat_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def Dot4 : DXILOp {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + a[n]*b[n] where n is between 0 and 3";
  let OpCode = 56;
  let OpClass = dot4;
  let LLVMIntrinsic = int_dx_dot4;
  let arguments = !dag(ins, !listsplat(llvm_anyfloat_ty, 8), !listsplat("", 8));
  let result = (out llvm_anyfloat_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0,
                             (overloads llvm_half_ty, llvm_float_ty),
                             (stages allKinds)>];
}

def ThreadId : DXILOp {
  let Doc = "Reads the thread ID";
  let OpCode = 93;
  let OpClass = threadId;
  let LLVMIntrinsic = int_dx_thread_id;
  let arguments = (ins llvm_i32_ty);
  let result = (out llvm_i32_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0, (overloads llvm_i32_ty),
                             (stages compute, mesh, amplification, node)>];
}

def GroupId  : DXILOp {
  let Doc = "Reads the group ID (SV_GroupID)";
  let OpCode = 94;
  let OpClass = groupId;
  let LLVMIntrinsic = int_dx_group_id;
  let arguments = (ins llvm_i32_ty);
  let result = (out llvm_i32_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0, (overloads llvm_i32_ty),
                             (stages compute, mesh, amplification, node)>];
}

def ThreadIdInGroup : DXILOp {
  let Doc = "Reads the thread ID within the group  (SV_GroupThreadID)";
  let OpCode = 95;
  let OpClass = threadIdInGroup;
  let LLVMIntrinsic = int_dx_thread_id_in_group;
  let arguments = (ins llvm_i32_ty);
  let result = (out llvm_i32_ty);
  let sm_constraints = [SMVersionConstraints<SM6_0, (overloads llvm_i32_ty),
                             (stages compute, mesh, amplification, node)>];
}

def FlattenedThreadIdInGroup : DXILOp {
  let Doc = "Provides a flattened index for a given thread within a given group (SV_GroupIndex)";
  let OpCode = 96;
  let OpClass = flattenedThreadIdInGroup;
  let LLVMIntrinsic = int_dx_flattened_thread_id_in_group;
  let arguments = (ins llvm_i32_ty);
  let result = (out);
  let sm_constraints = [SMVersionConstraints<SM6_0, (overloads llvm_i32_ty),
                             (stages compute, mesh, amplification, node)>];
}
