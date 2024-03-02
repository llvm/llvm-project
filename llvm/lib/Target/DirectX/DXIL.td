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

class DXILOpClass;

// Following is a set of DXIL Operation classes whose names appear to be
// arbitrary, yet need to be a substring of the function name used during
// lowering to DXIL Operation calls. These class name strings are specified
// as the third argument of add_dixil_op in utils/hct/hctdb.py and case converted
// in utils/hct/hctdb_instrhelp.py of DirectXShaderCompiler repo. The function
// name has the format "dx.op.<class-name>.<return-type>".

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
}

// Abstraction DXIL Operation to LLVM intrinsic
class DXILOpMapping<int opCode, DXILOpClass opClass, Intrinsic intrinsic, string doc> {
  int OpCode = opCode;                 // Opcode corresponding to DXIL Operation
  DXILOpClass OpClass = opClass;             // Class of DXIL Operation.
  Intrinsic LLVMIntrinsic = intrinsic; // LLVM Intrinsic the DXIL Operation maps
  string Doc = doc;                    // to a short description of the operation
}

// Concrete definition of DXIL Operation mapping to corresponding LLVM intrinsic
def Sin  : DXILOpMapping<13, unary, int_sin,
                         "Returns sine(theta) for theta in radians.">;
def Frac : DXILOpMapping<22, unary, int_dx_frac,
                         "Returns a fraction from 0 to 1 that represents the "
                         "decimal part of the input.">;
def Round : DXILOpMapping<26, unary, int_round,
                         "Returns the input rounded to the nearest integer"
                         "within a floating-point type.">;
def UMax : DXILOpMapping<39, binary, int_umax,
                         "Unsigned integer maximum. UMax(a,b) = a > b ? a : b">;
def ThreadId : DXILOpMapping<93, threadId, int_dx_thread_id,
                             "Reads the thread ID">;
def GroupId  : DXILOpMapping<94, groupId, int_dx_group_id,
                             "Reads the group ID (SV_GroupID)">;
def ThreadIdInGroup : DXILOpMapping<95, threadIdInGroup,
                                    int_dx_thread_id_in_group,
                                    "Reads the thread ID within the group "
                                    "(SV_GroupThreadID)">;
def FlattenedThreadIdInGroup : DXILOpMapping<96, flattenedThreadIdInGroup,
                                             int_dx_flattened_thread_id_in_group,
                                             "Provides a flattened index for a "
                                             "given thread within a given "
                                             "group (SV_GroupIndex)">;
