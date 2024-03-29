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

// Abstract class to demarcate minimum Shader model version required
// to support DXIL Op
class DXILShaderModel<int major, int minor> {
  int MajorAndMinor = !add(!mul(major, 10), minor);
}

// Valid minimum Shader model version records

// Shader Mode 6.x
foreach i = 0...9 in {
  def SM6_#i : DXILShaderModel<6, i>;
}
// Shader Mode 7.x - for now 7.0 is defined. Extend as needed
foreach i = 0 in {
  def SM7_#i : DXILShaderModel<7, i>;
}

// Abstraction of class mapping valid DXIL Op overloads the minimum
// version of Shader Model they are supported
class DXILOpOverload<DXILShaderModel minsm, list<LLVMType> overloads> {
  DXILShaderModel ShaderModel = minsm;
  list<LLVMType> OpOverloads = overloads;
}

// Abstraction of DXIL Operation class.
// It encapsulates an associated function signature viz.,
// returnTy(param1Ty, param2Ty, ...) represented as a list of LLVMTypes.
// DXIL Ops that belong to a DXILOpClass record the signature of that
// DXILOpClass

class DXILOpClass<list<LLVMType> OpSig> {
  list<LLVMType> OpSignature = OpSig;
}

// Concrete definitions of DXIL Op Classes
// Note that these class name strings are specified as the third argument
// of add_dixil_op in utils/hct/hctdb.py and case converted in
// utils/hct/hctdb_instrhelp.py of DirectXShaderCompiler repo. The function
// name has the format "dx.op.<class-name>.<return-type>", in most cases.

defset list<DXILOpClass> OpClasses = {
def acceptHitAndEndSearch : DXILOpClass<[llvm_void_ty]>;
//  def allocateNodeOutputRecords : DXILOpClass;
def allocateRayQuery : DXILOpClass<[llvm_i32_ty, llvm_i32_ty]>;
//  def annotateHandle : DXILOpClass;
//  def annotateNodeHandle : DXILOpClass;
//  def annotateNodeRecordHandle : DXILOpClass;
//  def atomicBinOp : DXILOpClass;
//  def atomicCompareExchange : DXILOpClass;
def attributeAtVertex : DXILOpClass<[llvm_any_ty, llvm_i32_ty, llvm_i32_ty, llvm_i8_ty, llvm_i8_ty]>;
def barrier : DXILOpClass<[llvm_void_ty, llvm_i32_ty]>;
//  def barrierByMemoryHandle : DXILOpClass;
def barrierByMemoryType : DXILOpClass<[llvm_void_ty, llvm_i32_ty, llvm_i32_ty]>;
//  def barrierByNodeRecordHandle : DXILOpClass;
def binary : DXILOpClass<[llvm_any_ty, LLVMMatchType<0>, LLVMMatchType<0>]>;
def binaryWithCarryOrBorrow : DXILOpClass<[llvm_i32_ty, llvm_any_ty, LLVMMatchType<0>]>;
//  def binaryWithTwoOuts : DXILOpClass;
//  def bitcastF16toI16 : DXILOpClass;
//  def bitcastF32toI32 : DXILOpClass;
//  def bitcastF64toI64 : DXILOpClass;
//  def bitcastI16toF16 : DXILOpClass;
//  def bitcastI32toF32 : DXILOpClass;
//  def bitcastI64toF64 : DXILOpClass;
//  def bufferLoad : DXILOpClass;
//  def bufferStore : DXILOpClass;
//  def bufferUpdateCounter : DXILOpClass;
//  def calculateLOD : DXILOpClass;
//  def callShader : DXILOpClass;
//  def cbufferLoad : DXILOpClass;
//  def cbufferLoadLegacy : DXILOpClass;
//  def checkAccessFullyMapped : DXILOpClass;
//  def coverage : DXILOpClass;
//  def createHandle : DXILOpClass;
//  def createHandleForLib : DXILOpClass;
//  def createHandleFromBinding : DXILOpClass;
//  def createHandleFromHeap : DXILOpClass;
//  def createNodeInputRecordHandle : DXILOpClass;
//  def createNodeOutputHandle : DXILOpClass;
//  def cutStream : DXILOpClass;
//  def cycleCounterLegacy : DXILOpClass;
//  def discard : DXILOpClass;
//  def dispatchMesh : DXILOpClass;
//  def dispatchRaysDimensions : DXILOpClass;
//  def dispatchRaysIndex : DXILOpClass;
//  def domainLocation : DXILOpClass;
def dot2 : DXILOpClass<!listsplat(llvm_anyfloat_ty, 5)>;
//  def dot2AddHalf : DXILOpClass;
def dot3 : DXILOpClass<!listsplat(llvm_anyfloat_ty, 7)>;
def dot4 : DXILOpClass<!listsplat(llvm_anyfloat_ty, 9)>;
//  def dot4AddPacked : DXILOpClass;
//  def emitIndices : DXILOpClass;
//  def emitStream : DXILOpClass;
//  def emitThenCutStream : DXILOpClass;
//  def evalCentroid : DXILOpClass;
//  def evalSampleIndex : DXILOpClass;
//  def evalSnapped : DXILOpClass;
//  def finishedCrossGroupSharing : DXILOpClass;
def flattenedThreadIdInGroup : DXILOpClass<[llvm_i32_ty]>;
//  def geometryIndex : DXILOpClass;
//  def getDimensions : DXILOpClass;
//  def getInputRecordCount : DXILOpClass;
//  def getMeshPayload : DXILOpClass;
//  def getNodeRecordPtr : DXILOpClass;
//  def getRemainingRecursionLevels : DXILOpClass;
def groupId : DXILOpClass<[llvm_i32_ty, llvm_i32_ty]>;
//  def gsInstanceID : DXILOpClass;
//  def hitKind : DXILOpClass;
//  def ignoreHit : DXILOpClass;
//  def incrementOutputCount : DXILOpClass;
//  def indexNodeHandle : DXILOpClass;
//  def innerCoverage : DXILOpClass;
//  def instanceID : DXILOpClass;
//  def instanceIndex : DXILOpClass;
//  def isHelperLane : DXILOpClass;
def isSpecialFloat : DXILOpClass<[llvm_i1_ty, llvm_anyfloat_ty]>;
//  def legacyDoubleToFloat : DXILOpClass;
//  def legacyDoubleToSInt32 : DXILOpClass;
//  def legacyDoubleToUInt32 : DXILOpClass;
//  def legacyF16ToF32 : DXILOpClass;
//  def legacyF32ToF16 : DXILOpClass;
//  def loadInput : DXILOpClass;
//  def loadOutputControlPoint : DXILOpClass;
//  def loadPatchConstant : DXILOpClass;
//  def makeDouble : DXILOpClass;
//  def minPrecXRegLoad : DXILOpClass;
//  def minPrecXRegStore : DXILOpClass;
//  def nodeOutputIsValid : DXILOpClass;
//  def objectRayDirection : DXILOpClass;
//  def objectRayOrigin : DXILOpClass;
//  def objectToWorld : DXILOpClass;
//  def outputComplete : DXILOpClass;
//  def outputControlPointID : DXILOpClass;
//  def pack4x8 : DXILOpClass;
//  def primitiveID : DXILOpClass;
//  def primitiveIndex : DXILOpClass;
//  def quadOp : DXILOpClass;
//  def quadReadLaneAt : DXILOpClass;
//  def quadVote : DXILOpClass;
//  def quaternary : DXILOpClass;
//  def rawBufferLoad : DXILOpClass;
//  def rawBufferStore : DXILOpClass;
//  def rayFlags : DXILOpClass;
//  def rayQuery_Abort : DXILOpClass;
//  def rayQuery_CommitNonOpaqueTriangleHit : DXILOpClass;
//  def rayQuery_CommitProceduralPrimitiveHit : DXILOpClass;
//  def rayQuery_Proceed : DXILOpClass;
//  def rayQuery_StateMatrix : DXILOpClass;
//  def rayQuery_StateScalar : DXILOpClass;
//  def rayQuery_StateVector : DXILOpClass;
//  def rayQuery_TraceRayInline : DXILOpClass;
//  def rayTCurrent : DXILOpClass;
//  def rayTMin : DXILOpClass;
//  def renderTargetGetSampleCount : DXILOpClass;
//  def renderTargetGetSamplePosition : DXILOpClass;
//  def reportHit : DXILOpClass;
//  def sample : DXILOpClass;
//  def sampleBias : DXILOpClass;
//  def sampleCmp : DXILOpClass;
//  def sampleCmpBias : DXILOpClass;
//  def sampleCmpGrad : DXILOpClass;
//  def sampleCmpLevel : DXILOpClass;
//  def sampleCmpLevelZero : DXILOpClass;
//  def sampleGrad : DXILOpClass;
//  def sampleIndex : DXILOpClass;
//  def sampleLevel : DXILOpClass;
//  def setMeshOutputCounts : DXILOpClass;
//  def splitDouble : DXILOpClass;
//  def startInstanceLocation : DXILOpClass;
//  def startVertexLocation : DXILOpClass;
//  def storeOutput : DXILOpClass;
//  def storePatchConstant : DXILOpClass;
//  def storePrimitiveOutput : DXILOpClass;
//  def storeVertexOutput : DXILOpClass;
//  def tempRegLoad : DXILOpClass;
//  def tempRegStore : DXILOpClass;
def tertiary : DXILOpClass<[llvm_any_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>]>;
//  def texture2DMSGetSamplePosition : DXILOpClass;
//  def textureGather : DXILOpClass;
//  def textureGatherCmp : DXILOpClass;
//  def textureGatherRaw : DXILOpClass;
//  def textureLoad : DXILOpClass;
//  def textureStore : DXILOpClass;
//  def textureStoreSample : DXILOpClass;
def threadId : DXILOpClass<[llvm_i32_ty, llvm_i32_ty]>;
def threadIdInGroup : DXILOpClass<[llvm_i32_ty, llvm_i32_ty]>;
//  def traceRay : DXILOpClass;
def unary : DXILOpClass<[llvm_any_ty, LLVMMatchType<0>]>;
//  def unaryBits : DXILOpClass;
//  def unpack4x8 : DXILOpClass;
//  def viewID : DXILOpClass;
//  def waveActiveAllEqual : DXILOpClass;
//  def waveActiveBallot : DXILOpClass;
//  def waveActiveBit : DXILOpClass;
//  def waveActiveOp : DXILOpClass;
//  def waveAllOp : DXILOpClass;
//  def waveAllTrue : DXILOpClass;
//  def waveAnyTrue : DXILOpClass;
//  def waveGetLaneCount : DXILOpClass;
//  def waveGetLaneIndex : DXILOpClass;
//  def waveIsFirstLane : DXILOpClass;
//  def waveMatch : DXILOpClass;
//  def waveMatrix_Accumulate : DXILOpClass;
//  def waveMatrix_Annotate : DXILOpClass;
//  def waveMatrix_Depth : DXILOpClass;
//  def waveMatrix_Fill : DXILOpClass;
//  def waveMatrix_LoadGroupShared : DXILOpClass;
//  def waveMatrix_LoadRawBuf : DXILOpClass;
//  def waveMatrix_Multiply : DXILOpClass;
//  def waveMatrix_ScalarOp : DXILOpClass;
//  def waveMatrix_StoreGroupShared : DXILOpClass;
//  def waveMatrix_StoreRawBuf : DXILOpClass;
//  def waveMultiPrefixBitCount : DXILOpClass;
//  def waveMultiPrefixOp : DXILOpClass;
//  def wavePrefixOp : DXILOpClass;
//  def waveReadLaneAt : DXILOpClass;
//  def waveReadLaneFirst : DXILOpClass;
//  def worldRayDirection : DXILOpClass;
//  def worldRayOrigin : DXILOpClass;
//  def worldToObject : DXILOpClass;
//  def writeSamplerFeedback : DXILOpClass;
//  def writeSamplerFeedbackBias : DXILOpClass;
//  def writeSamplerFeedbackGrad : DXILOpClass;
//  def writeSamplerFeedbackLevel: DXILOpClass;

  // This is a sentinel definition. Hence placed at the end of the list
  // and not as part of the above alphabetically sorted valid definitions.
  // Additionally it is capitalized unlike all the others.
def UnknownOpClass: DXILOpClass<[]>;
}

// Abstraction DXIL Operation to LLVM intrinsic
class DXILOpMappingBase {
  int OpCode = 0;                      // Opcode of DXIL Operation
  DXILOpClass OpClass = UnknownOpClass;// Class of DXIL Operation.
  Intrinsic LLVMIntrinsic = ?;         // LLVM Intrinsic DXIL Operation maps to
  list<DXILOpOverload> OpOverloadTypes = ?; // Valid overload type
                                       // of DXIL Operation
  string Doc = "";                     // A short description of the operation
}

class DXILOpMapping<int opCode,
                    Intrinsic intrinsic,
                    list<DXILOpOverload> overloadTypes,
                    string doc> : DXILOpMappingBase {
  int OpCode = opCode;
  Intrinsic LLVMIntrinsic = intrinsic;
  list<DXILOpOverload> OpOverloadTypes = overloadTypes;
  string Doc = doc;
}

// Concrete definitions of DXIL Operation mapping to corresponding LLVM intrinsic

// IsSpecialFloat Class
let OpClass = isSpecialFloat in {
  def IsInf : DXILOpMapping<9,  int_dx_isinf, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                           "Determines if the specified value is infinite.">;
}

// Unary Class
let OpClass = unary in {
  def Abs : DXILOpMapping<6, int_fabs, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty, llvm_double_ty]>],
                          "Returns the absolute value of the input.">;

  def Cos  : DXILOpMapping<12, int_cos, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                          "Returns cosine(theta) for theta in radians.">;
  def Sin  : DXILOpMapping<13, int_sin, [DXILOpOverload<SM6_3, [llvm_half_ty, llvm_float_ty]>,
                                         DXILOpOverload<SM6_0, [llvm_float_ty]>],
                           "Returns sine(theta) for theta in radians.">;
  def Exp2 : DXILOpMapping<21, int_exp2, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                           "Returns the base 2 exponential, or 2**x, of the"
                           " specified value. exp2(x) = 2**x.">;
  def Frac : DXILOpMapping<22, int_dx_frac, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                            "Returns a fraction from 0 to 1 that represents the"
                            " decimal part of the input.">;
  def Log2 : DXILOpMapping<23, int_log2, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                           "Returns the base-2 logarithm of the specified value.">;
  def Sqrt : DXILOpMapping<24, int_sqrt, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                           "Returns the square root of the specified floating-point"
                           "value, per component.">;
  def RSqrt : DXILOpMapping<25, int_dx_rsqrt, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                            "Returns the reciprocal of the square root of the"
                            " specified value. rsqrt(x) = 1 / sqrt(x).">;
  def Round : DXILOpMapping<26, int_roundeven, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                            "Returns the input rounded to the nearest integer"
                            "within a floating-point type.">;
  def Floor : DXILOpMapping<27, int_floor, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                            "Returns the largest integer that is less than or equal to the input.">;
  def Ceil  : DXILOpMapping<28, int_ceil, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                            "Returns the smallest integer that is greater than or equal to the input.">;
  def Trunc : DXILOpMapping<29, int_trunc, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                            "Returns the specified value truncated to the integer component.">;
  def Rbits : DXILOpMapping<30, int_bitreverse, [DXILOpOverload<SM6_0, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>],
                            "Returns the specified value with its bits reversed.">;
}

// Binary Class
let OpClass = binary in {
// Float overloads
  def FMax : DXILOpMapping<35, int_maxnum, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty, llvm_double_ty]>],
                           "Float maximum. FMax(a,b) = a > b ? a : b">;
  def FMin : DXILOpMapping<36, int_minnum, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty, llvm_double_ty]>],
                           "Float minimum. FMin(a,b) = a < b ? a : b">;
// Int overloads
  def SMax : DXILOpMapping<37, int_smax, [DXILOpOverload<SM6_0,[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>],
                           "Signed integer maximum. SMax(a,b) = a > b ? a : b">;
  def SMin : DXILOpMapping<38, int_smin, [DXILOpOverload<SM6_0,[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>],
                           "Signed integer minimum. SMin(a,b) = a < b ? a : b">;
  def UMax : DXILOpMapping<39, int_umax, [DXILOpOverload<SM6_0,[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>],
                           "Unsigned integer maximum. UMax(a,b) = a > b ? a : b">;
  def UMin : DXILOpMapping<40, int_umin, [DXILOpOverload<SM6_0,[llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>],
                           "Unsigned integer minimum. UMin(a,b) = a < b ? a : b">;
}

// Tertiary Class
let OpClass = tertiary in {
// Float overloads
//   let OpOverloadTypes =  [llvm_half_ty, llvm_float_ty, llvm_double_ty] in {
  def FMad : DXILOpMapping<46, int_fmuladd, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty, llvm_double_ty]>],
                            "Floating point arithmetic multiply/add operation."
                            " fmad(m,a,b) = m * a + b.">;
// Int overloads
def IMad : DXILOpMapping<48, int_dx_imad, [DXILOpOverload<SM6_0, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>],
                         "Signed integer arithmetic multiply/add operation."
                          " imad(m,a,b) = m * a + b.">;
def UMad : DXILOpMapping<49, int_dx_umad, [DXILOpOverload<SM6_0, [llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>],
                        "Unsigned integer arithmetic multiply/add operation."
                        " umad(m,a, = m * a + b.">;
}

// Dot Operations
// let OpOverloadTypes =  [llvm_half_ty, llvm_float_ty] in
let OpClass = dot2 in
  def Dot2 : DXILOpMapping<54, int_dx_dot2, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                          "dot product of two float vectors Dot(a,b) = a[0]*b[0] +"
                          " ... + a[n]*b[n] where n is between 0 and 1">;
let OpClass = dot3 in
  def Dot3 : DXILOpMapping<55, int_dx_dot3, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                           "dot product of two float vectors Dot(a,b) = a[0]*b[0] +"
                           " ... + a[n]*b[n] where n is between 0 and 2">;
let OpClass = dot4 in
   def Dot4 : DXILOpMapping<56, int_dx_dot4, [DXILOpOverload<SM6_0, [llvm_half_ty, llvm_float_ty]>],
                            "dot product of two float vectors Dot(a,b) = a[0]*b[0] +"
                            " ... + a[n]*b[n] where n is between 0 and 3">;

// Thread Operations
let OpClass =  threadId in
  def ThreadId : DXILOpMapping<93, int_dx_thread_id, [DXILOpOverload<SM6_0, [llvm_i32_ty]>],
                              "Reads the thread ID">;
let OpClass =  groupId in
  def GroupId  : DXILOpMapping<94, int_dx_group_id, [DXILOpOverload<SM6_0, [llvm_i32_ty]>],
                               "Reads the group ID (SV_GroupID)">;
let OpClass =  threadIdInGroup in
  def ThreadIdInGroup : DXILOpMapping<95, int_dx_thread_id_in_group, [DXILOpOverload<SM6_0, [llvm_i32_ty]>],
                                     "Reads the thread ID within the group "
                                     "(SV_GroupThreadID)">;
let OpClass = flattenedThreadIdInGroup in
  def FlattenedThreadIdInGroup : DXILOpMapping<96, int_dx_flattened_thread_id_in_group,
                                               [DXILOpOverload<SM6_0, [llvm_i32_ty]>],
                                                "Provides a flattened index for a given"
                                                " thread within a given group (SV_GroupIndex)">;
