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
  def SM6_ #i : Version<6, i>;
  def DX1_ #i : Version<1, i>;
}

// Overload type alias of llvm_any_ty
defvar dxil_overload_ty = llvm_any_ty;

class DXILOpClass;

// Concrete definitions of DXIL Op Classes
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

// This is a sentinel definition. Hence placed at the end here
// and not as part of the above alphabetically sorted valid definitions.
// It is never used to construct the name of DXIL Op call name.
// Additionally it is capitalized unlike all the others.
def UnknownOpClass : DXILOpClass;

// Shader stages
class ShaderStage;

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
def anyhit : ShaderStage;
def closesthit : ShaderStage;
def callable : ShaderStage;
def miss : ShaderStage;

// Primitive predicate
class Pred;

// Following classes represent predicate that evaluate to the payload,
// if one xists, unconditionally.

// Predicate that evaluates to specified shader model version unconditionally.
class SMVersion<Version ver> : Pred {
  Version sm_version = ver;
}

// Predicate that evaluates to specified overload type list unconditionally.
class Overloads<list<LLVMType> tys> : Pred {
  list<LLVMType> overload_types = tys;
}

// Predicate that evaluates to specified list of shader stages unconditionally.
class Stages<list<ShaderStage> st> : Pred {
  list<ShaderStage> stage_kinds = st;
}

// Following class represents predicate that evaluates to the payload when
// specified predicate is true.

class Constraints<Pred p, list<Pred> l = []> : Pred {
  Pred pred = p;
  list<Pred> constraints = l;
}

// DXIL Op attributes defined as true predicates, if specified

class DXILAttribute;

def ReadOnly : DXILAttribute;
def ReadNone : DXILAttribute;
def IsDerivative : DXILAttribute;
def IsGradient : DXILAttribute;
def IsFeedback : DXILAttribute;
def IsWave : DXILAttribute;
def NeedsUniformInputs : DXILAttribute;
def IsBarrier : DXILAttribute;


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
  Intrinsic LLVMIntrinsic = ? ;

  // List of argument types of the op. Default to 0 arguments.
  list<LLVMType> arguments = [];

  // List of result types of the op. Default to 0 results.
  list<LLVMType> result = [];

  // List of constraints predicated on Shader Model version.
  // If no constraints are specified,
  // a) operation is assumed to be supported in Shader Model 6.0 and later.
  // b) has no overload types
  // c) is supported in all shader stage kinds
  // If a constraint is specified, Shader Model version on which the
  // constraint is predicated is mandatory. One or both of the Overload
  // type and shader kind constraints can be omitted when appropriate.

  list<Constraints> constraints = [];

  // Operation attributes
  list<DXILAttribute> attributes = [];

  Version DXILVersion = ? ;
}

// Convenience class for minimal DXIL Op definition

class DXILOpAndClass<int opcode, DXILOpClass opcalss> : DXILOp {
  int OpCode = opcode;
  DXILOpClass OpClass = opcalss;
}

// Concrete definitions of DXIL Operations

def Abs : DXILOpAndClass<6, unary> {
  let Doc = "Returns the absolute value of the input.";
  let LLVMIntrinsic = int_fabs;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_half_ty, llvm_float_ty, llvm_double_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def IsInf : DXILOpAndClass<9, isSpecialFloat> {
  let Doc = "Determines if the specified value is infinite.";
  let LLVMIntrinsic = int_dx_isinf;
  let arguments = [llvm_anyfloat_ty];
  let result = [llvm_i1_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Cos : DXILOpAndClass<12, unary> {
  let Doc = "Returns cosine(theta) for theta in radians.";
  let LLVMIntrinsic = int_cos;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Sin : DXILOpAndClass<13, unary> {
  let Doc = "Returns sine(theta) for theta in radians.";
  let LLVMIntrinsic = int_sin;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Tan : DXILOpAndClass<14, unary> {
  let Doc = "Returns tangent(theta) for theta in radians.";
  let LLVMIntrinsic = int_tan;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def ACos : DXILOpAndClass<15, unary> {
  let Doc = "Returns the arccosine of the specified value.";
  let LLVMIntrinsic = int_acos;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def ASin : DXILOpAndClass<16, unary> {
  let Doc = "Returns the arcsine of the specified value.";
  let LLVMIntrinsic = int_asin;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def ATan : DXILOpAndClass<17, unary> {
  let Doc = "Returns the arctangent of the specified value.";
  let LLVMIntrinsic = int_atan;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def HCos : DXILOpAndClass<18, unary> {
  let Doc = "Returns the hyperbolic cosine of the specified value.";
  let LLVMIntrinsic = int_cosh;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def HSin : DXILOpAndClass<19, unary> {
  let Doc = "Returns the hyperbolic sine of the specified value.";
  let LLVMIntrinsic = int_sinh;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def HTan : DXILOpAndClass<20, unary> {
  let Doc = "Returns the hyperbolic tan of the specified value.";
  let LLVMIntrinsic = int_tanh;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Exp2 : DXILOpAndClass<21, unary> {
  let Doc = "Returns the base 2 exponential, or 2**x, of the specified value. "
            "exp2(x) = 2**x.";
  let LLVMIntrinsic = int_exp2;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Frac : DXILOpAndClass<22, unary> {
  let Doc = "Returns a fraction from 0 to 1 that represents the decimal part "
            "of the input.";
  let LLVMIntrinsic = int_dx_frac;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Log2 : DXILOpAndClass<23, unary> {
  let Doc = "Returns the base-2 logarithm of the specified value.";
  let LLVMIntrinsic = int_log2;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Sqrt : DXILOpAndClass<24, unary> {
  let Doc = "Returns the square root of the specified floating-point value, "
            "per component.";
  let LLVMIntrinsic = int_sqrt;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def RSqrt : DXILOpAndClass<25, unary> {
  let Doc = "Returns the reciprocal of the square root of the specified value. "
            "rsqrt(x) = 1 / sqrt(x).";
  let LLVMIntrinsic = int_dx_rsqrt;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Round : DXILOpAndClass<26, unary> {
  let Doc = "Returns the input rounded to the nearest integer within a "
            "floating-point type.";
  let LLVMIntrinsic = int_roundeven;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Floor : DXILOpAndClass<27, unary> {
  let Doc =
      "Returns the largest integer that is less than or equal to the input.";
  let LLVMIntrinsic = int_floor;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Ceil : DXILOpAndClass<28, unary> {
  let Doc = "Returns the smallest integer that is greater than or equal to the "
            "input.";
  let LLVMIntrinsic = int_ceil;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Trunc : DXILOpAndClass<29, unary> {
  let Doc = "Returns the specified value truncated to the integer component.";
  let LLVMIntrinsic = int_trunc;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Rbits : DXILOpAndClass<30, unary> {
  let Doc = "Returns the specified value with its bits reversed.";
  let LLVMIntrinsic = int_bitreverse;
  let arguments = [LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def FMax : DXILOpAndClass<35, binary> {
  let Doc = "Float maximum. FMax(a,b) = a > b ? a : b";
  let LLVMIntrinsic = int_maxnum;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_half_ty, llvm_float_ty, llvm_double_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def FMin : DXILOpAndClass<36, binary> {
  let Doc = "Float minimum. FMin(a,b) = a < b ? a : b";
  let LLVMIntrinsic = int_minnum;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_half_ty, llvm_float_ty, llvm_double_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def SMax : DXILOpAndClass<37, binary> {
  let Doc = "Signed integer maximum. SMax(a,b) = a > b ? a : b";
  let LLVMIntrinsic = int_smax;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def SMin : DXILOpAndClass<38, binary> {
  let Doc = "Signed integer minimum. SMin(a,b) = a < b ? a : b";
  let LLVMIntrinsic = int_smin;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def UMax : DXILOpAndClass<39, binary> {
  let Doc = "Unsigned integer maximum. UMax(a,b) = a > b ? a : b";
  let LLVMIntrinsic = int_umax;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def UMin : DXILOpAndClass<40, binary> {
  let Doc = "Unsigned integer minimum. UMin(a,b) = a < b ? a : b";
  let LLVMIntrinsic = int_umin;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def FMad : DXILOpAndClass<46, tertiary> {
  let Doc = "Floating point arithmetic multiply/add operation. fmad(m,a,b) = m "
            "* a + b.";
  let LLVMIntrinsic = int_fmuladd;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_half_ty, llvm_float_ty, llvm_double_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def IMad : DXILOpAndClass<48, tertiary> {
  let Doc = "Signed integer arithmetic multiply/add operation. imad(m,a,b) = m "
            "* a + b.";
  let LLVMIntrinsic = int_dx_imad;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def UMad : DXILOpAndClass<49, tertiary> {
  let Doc = "Unsigned integer arithmetic multiply/add operation. umad(m,a, = m "
            "* a + b.";
  let LLVMIntrinsic = int_dx_umad;
  let arguments = [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>];
  let result = [dxil_overload_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Overloads<[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Dot2 : DXILOpAndClass<54, dot2> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is between 0 and 1";
  let LLVMIntrinsic = int_dx_dot2;
  let arguments = !listsplat(llvm_anyfloat_ty, 4);
  let result = [llvm_anyfloat_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Dot3 : DXILOpAndClass<55, dot3> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is between 0 and 2";
  let LLVMIntrinsic = int_dx_dot3;
  let arguments = !listsplat(llvm_anyfloat_ty, 6);
  let result = [llvm_anyfloat_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def Dot4 : DXILOpAndClass<56, dot4> {
  let Doc = "dot product of two float vectors Dot(a,b) = a[0]*b[0] + ... + "
            "a[n]*b[n] where n is between 0 and 3";
  let LLVMIntrinsic = int_dx_dot4;
  let arguments = !listsplat(llvm_anyfloat_ty, 8);
  let result = [llvm_anyfloat_ty];
  let constraints = [
    Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
  ];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def ThreadId : DXILOpAndClass<93, threadId> {
  let Doc = "Reads the thread ID";
  let LLVMIntrinsic = int_dx_thread_id;
  let arguments = [llvm_i32_ty];
  let result = [llvm_i32_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Stages<[compute, mesh, amplification, node]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def GroupId : DXILOpAndClass<94, groupId> {
  let Doc = "Reads the group ID (SV_GroupID)";
  let LLVMIntrinsic = int_dx_group_id;
  let arguments = [llvm_i32_ty];
  let result = [llvm_i32_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Stages<[compute, mesh, amplification, node]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def ThreadIdInGroup : DXILOpAndClass<95, threadIdInGroup> {
  let Doc = "Reads the thread ID within the group  (SV_GroupThreadID)";
  let LLVMIntrinsic = int_dx_thread_id_in_group;
  let arguments = [llvm_i32_ty];
  let result = [llvm_i32_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Stages<[compute, mesh, amplification, node]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}

def FlattenedThreadIdInGroup : DXILOpAndClass<96, flattenedThreadIdInGroup> {
  let Doc = "Provides a flattened index for a given thread within a given "
            "group (SV_GroupIndex)";
  let LLVMIntrinsic = int_dx_flattened_thread_id_in_group;
  let arguments = [llvm_i32_ty];
  let constraints =
      [Constraints<SMVersion<SM6_0>,
                   [Stages<[compute, mesh, amplification, node]>]>];
  let attributes = [ReadNone];
  let DXILVersion = DX1_0;
}
