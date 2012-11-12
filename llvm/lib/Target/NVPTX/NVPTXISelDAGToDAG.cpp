//===-- NVPTXISelDAGToDAG.cpp - A dag to dag inst selector for NVPTX ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the NVPTX target.
//
//===----------------------------------------------------------------------===//


#include "llvm/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "NVPTXISelDAGToDAG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/GlobalValue.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "nvptx-isel"

using namespace llvm;


static cl::opt<bool>
UseFMADInstruction("nvptx-mad-enable",
                   cl::ZeroOrMore,
                cl::desc("NVPTX Specific: Enable generating FMAD instructions"),
                   cl::init(false));

static cl::opt<int>
FMAContractLevel("nvptx-fma-level",
                 cl::ZeroOrMore,
                 cl::desc("NVPTX Specific: FMA contraction (0: don't do it"
                     " 1: do it  2: do it aggressively"),
                     cl::init(2));


static cl::opt<int>
UsePrecDivF32("nvptx-prec-divf32",
              cl::ZeroOrMore,
             cl::desc("NVPTX Specifies: 0 use div.approx, 1 use div.full, 2 use"
                  " IEEE Compliant F32 div.rnd if avaiable."),
                  cl::init(2));

/// createNVPTXISelDag - This pass converts a legalized DAG into a
/// NVPTX-specific DAG, ready for instruction scheduling.
FunctionPass *llvm::createNVPTXISelDag(NVPTXTargetMachine &TM,
                                       llvm::CodeGenOpt::Level OptLevel) {
  return new NVPTXDAGToDAGISel(TM, OptLevel);
}


NVPTXDAGToDAGISel::NVPTXDAGToDAGISel(NVPTXTargetMachine &tm,
                                     CodeGenOpt::Level OptLevel)
: SelectionDAGISel(tm, OptLevel),
  Subtarget(tm.getSubtarget<NVPTXSubtarget>())
{
  // Always do fma.f32 fpcontract if the target supports the instruction.
  // Always do fma.f64 fpcontract if the target supports the instruction.
  // Do mad.f32 is nvptx-mad-enable is specified and the target does not
  // support fma.f32.

  doFMADF32 = (OptLevel > 0) && UseFMADInstruction && !Subtarget.hasFMAF32();
  doFMAF32 =  (OptLevel > 0) && Subtarget.hasFMAF32() &&
      (FMAContractLevel>=1);
  doFMAF64 =  (OptLevel > 0) && Subtarget.hasFMAF64() &&
      (FMAContractLevel>=1);
  doFMAF32AGG =  (OptLevel > 0) && Subtarget.hasFMAF32() &&
      (FMAContractLevel==2);
  doFMAF64AGG =  (OptLevel > 0) && Subtarget.hasFMAF64() &&
      (FMAContractLevel==2);

  allowFMA = (FMAContractLevel >= 1) || UseFMADInstruction;

  UseF32FTZ = false;

  doMulWide = (OptLevel > 0);

  // Decide how to translate f32 div
  do_DIVF32_PREC = UsePrecDivF32;
  // sm less than sm_20 does not support div.rnd. Use div.full.
  if (do_DIVF32_PREC == 2 && !Subtarget.reqPTX20())
    do_DIVF32_PREC = 1;

}

/// Select - Select instructions not customized! Used for
/// expanded, promoted and normal instructions.
SDNode* NVPTXDAGToDAGISel::Select(SDNode *N) {

  if (N->isMachineOpcode())
    return NULL;   // Already selected.

  SDNode *ResNode = NULL;
  switch (N->getOpcode()) {
  case ISD::LOAD:
    ResNode = SelectLoad(N);
    break;
  case ISD::STORE:
    ResNode = SelectStore(N);
    break;
  }
  if (ResNode)
    return ResNode;
  return SelectCode(N);
}


static unsigned int
getCodeAddrSpace(MemSDNode *N, const NVPTXSubtarget &Subtarget)
{
  const Value *Src = N->getSrcValue();
  if (!Src)
    return NVPTX::PTXLdStInstCode::LOCAL;

  if (const PointerType *PT = dyn_cast<PointerType>(Src->getType())) {
    switch (PT->getAddressSpace()) {
    case llvm::ADDRESS_SPACE_LOCAL: return NVPTX::PTXLdStInstCode::LOCAL;
    case llvm::ADDRESS_SPACE_GLOBAL: return NVPTX::PTXLdStInstCode::GLOBAL;
    case llvm::ADDRESS_SPACE_SHARED: return NVPTX::PTXLdStInstCode::SHARED;
    case llvm::ADDRESS_SPACE_CONST_NOT_GEN:
      return NVPTX::PTXLdStInstCode::CONSTANT;
    case llvm::ADDRESS_SPACE_GENERIC: return NVPTX::PTXLdStInstCode::GENERIC;
    case llvm::ADDRESS_SPACE_PARAM: return NVPTX::PTXLdStInstCode::PARAM;
    case llvm::ADDRESS_SPACE_CONST:
      // If the arch supports generic address space, translate it to GLOBAL
      // for correctness.
      // If the arch does not support generic address space, then the arch
      // does not really support ADDRESS_SPACE_CONST, translate it to
      // to CONSTANT for better performance.
      if (Subtarget.hasGenericLdSt())
        return NVPTX::PTXLdStInstCode::GLOBAL;
      else
        return NVPTX::PTXLdStInstCode::CONSTANT;
    default: break;
    }
  }
  return NVPTX::PTXLdStInstCode::LOCAL;
}


SDNode* NVPTXDAGToDAGISel::SelectLoad(SDNode *N) {
  DebugLoc dl = N->getDebugLoc();
  LoadSDNode *LD = cast<LoadSDNode>(N);
  EVT LoadedVT = LD->getMemoryVT();
  SDNode *NVPTXLD= NULL;

  // do not support pre/post inc/dec
  if (LD->isIndexed())
    return NULL;

  if (!LoadedVT.isSimple())
    return NULL;

  // Address Space Setting
  unsigned int codeAddrSpace = getCodeAddrSpace(LD, Subtarget);

  // Volatile Setting
  // - .volatile is only availalble for .global and .shared
  bool isVolatile = LD->isVolatile();
  if (codeAddrSpace != NVPTX::PTXLdStInstCode::GLOBAL &&
      codeAddrSpace != NVPTX::PTXLdStInstCode::SHARED &&
      codeAddrSpace != NVPTX::PTXLdStInstCode::GENERIC)
    isVolatile = false;

  // Vector Setting
  MVT SimpleVT = LoadedVT.getSimpleVT();
  unsigned vecType = NVPTX::PTXLdStInstCode::Scalar;
  if (SimpleVT.isVector()) {
    unsigned num = SimpleVT.getVectorNumElements();
    if (num == 2)
      vecType = NVPTX::PTXLdStInstCode::V2;
    else if (num == 4)
      vecType = NVPTX::PTXLdStInstCode::V4;
    else
      return NULL;
  }

  // Type Setting: fromType + fromTypeWidth
  //
  // Sign   : ISD::SEXTLOAD
  // Unsign : ISD::ZEXTLOAD, ISD::NON_EXTLOAD or ISD::EXTLOAD and the
  //          type is integer
  // Float  : ISD::NON_EXTLOAD or ISD::EXTLOAD and the type is float
  MVT ScalarVT = SimpleVT.getScalarType();
  unsigned fromTypeWidth =  ScalarVT.getSizeInBits();
  unsigned int fromType;
  if ((LD->getExtensionType() == ISD::SEXTLOAD))
    fromType = NVPTX::PTXLdStInstCode::Signed;
  else if (ScalarVT.isFloatingPoint())
    fromType = NVPTX::PTXLdStInstCode::Float;
  else
    fromType = NVPTX::PTXLdStInstCode::Unsigned;

  // Create the machine instruction DAG
  SDValue Chain = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue Addr;
  SDValue Offset, Base;
  unsigned Opcode;
  MVT::SimpleValueType TargetVT = LD->getValueType(0).getSimpleVT().SimpleTy;

  if (SelectDirectAddr(N1, Addr)) {
    switch (TargetVT) {
    case MVT::i8:    Opcode = NVPTX::LD_i8_avar; break;
    case MVT::i16:   Opcode = NVPTX::LD_i16_avar; break;
    case MVT::i32:   Opcode = NVPTX::LD_i32_avar; break;
    case MVT::i64:   Opcode = NVPTX::LD_i64_avar; break;
    case MVT::f32:   Opcode = NVPTX::LD_f32_avar; break;
    case MVT::f64:   Opcode = NVPTX::LD_f64_avar; break;
    case MVT::v2i8:  Opcode = NVPTX::LD_v2i8_avar; break;
    case MVT::v2i16: Opcode = NVPTX::LD_v2i16_avar; break;
    case MVT::v2i32: Opcode = NVPTX::LD_v2i32_avar; break;
    case MVT::v2i64: Opcode = NVPTX::LD_v2i64_avar; break;
    case MVT::v2f32: Opcode = NVPTX::LD_v2f32_avar; break;
    case MVT::v2f64: Opcode = NVPTX::LD_v2f64_avar; break;
    case MVT::v4i8:  Opcode = NVPTX::LD_v4i8_avar; break;
    case MVT::v4i16: Opcode = NVPTX::LD_v4i16_avar; break;
    case MVT::v4i32: Opcode = NVPTX::LD_v4i32_avar; break;
    case MVT::v4f32: Opcode = NVPTX::LD_v4f32_avar; break;
    default: return NULL;
    }
    SDValue Ops[] = { getI32Imm(isVolatile),
                      getI32Imm(codeAddrSpace),
                      getI32Imm(vecType),
                      getI32Imm(fromType),
                      getI32Imm(fromTypeWidth),
                      Addr, Chain };
    NVPTXLD = CurDAG->getMachineNode(Opcode, dl, TargetVT,
                                     MVT::Other, Ops, 7);
  } else if (Subtarget.is64Bit()?
      SelectADDRsi64(N1.getNode(), N1, Base, Offset):
      SelectADDRsi(N1.getNode(), N1, Base, Offset)) {
    switch (TargetVT) {
    case MVT::i8:    Opcode = NVPTX::LD_i8_asi; break;
    case MVT::i16:   Opcode = NVPTX::LD_i16_asi; break;
    case MVT::i32:   Opcode = NVPTX::LD_i32_asi; break;
    case MVT::i64:   Opcode = NVPTX::LD_i64_asi; break;
    case MVT::f32:   Opcode = NVPTX::LD_f32_asi; break;
    case MVT::f64:   Opcode = NVPTX::LD_f64_asi; break;
    case MVT::v2i8:  Opcode = NVPTX::LD_v2i8_asi; break;
    case MVT::v2i16: Opcode = NVPTX::LD_v2i16_asi; break;
    case MVT::v2i32: Opcode = NVPTX::LD_v2i32_asi; break;
    case MVT::v2i64: Opcode = NVPTX::LD_v2i64_asi; break;
    case MVT::v2f32: Opcode = NVPTX::LD_v2f32_asi; break;
    case MVT::v2f64: Opcode = NVPTX::LD_v2f64_asi; break;
    case MVT::v4i8:  Opcode = NVPTX::LD_v4i8_asi; break;
    case MVT::v4i16: Opcode = NVPTX::LD_v4i16_asi; break;
    case MVT::v4i32: Opcode = NVPTX::LD_v4i32_asi; break;
    case MVT::v4f32: Opcode = NVPTX::LD_v4f32_asi; break;
    default: return NULL;
    }
    SDValue Ops[] = { getI32Imm(isVolatile),
                      getI32Imm(codeAddrSpace),
                      getI32Imm(vecType),
                      getI32Imm(fromType),
                      getI32Imm(fromTypeWidth),
                      Base, Offset, Chain };
    NVPTXLD = CurDAG->getMachineNode(Opcode, dl, TargetVT,
                                     MVT::Other, Ops, 8);
  } else if (Subtarget.is64Bit()?
      SelectADDRri64(N1.getNode(), N1, Base, Offset):
      SelectADDRri(N1.getNode(), N1, Base, Offset)) {
    switch (TargetVT) {
    case MVT::i8:    Opcode = NVPTX::LD_i8_ari; break;
    case MVT::i16:   Opcode = NVPTX::LD_i16_ari; break;
    case MVT::i32:   Opcode = NVPTX::LD_i32_ari; break;
    case MVT::i64:   Opcode = NVPTX::LD_i64_ari; break;
    case MVT::f32:   Opcode = NVPTX::LD_f32_ari; break;
    case MVT::f64:   Opcode = NVPTX::LD_f64_ari; break;
    case MVT::v2i8:  Opcode = NVPTX::LD_v2i8_ari; break;
    case MVT::v2i16: Opcode = NVPTX::LD_v2i16_ari; break;
    case MVT::v2i32: Opcode = NVPTX::LD_v2i32_ari; break;
    case MVT::v2i64: Opcode = NVPTX::LD_v2i64_ari; break;
    case MVT::v2f32: Opcode = NVPTX::LD_v2f32_ari; break;
    case MVT::v2f64: Opcode = NVPTX::LD_v2f64_ari; break;
    case MVT::v4i8:  Opcode = NVPTX::LD_v4i8_ari; break;
    case MVT::v4i16: Opcode = NVPTX::LD_v4i16_ari; break;
    case MVT::v4i32: Opcode = NVPTX::LD_v4i32_ari; break;
    case MVT::v4f32: Opcode = NVPTX::LD_v4f32_ari; break;
    default: return NULL;
    }
    SDValue Ops[] = { getI32Imm(isVolatile),
                      getI32Imm(codeAddrSpace),
                      getI32Imm(vecType),
                      getI32Imm(fromType),
                      getI32Imm(fromTypeWidth),
                      Base, Offset, Chain };
    NVPTXLD = CurDAG->getMachineNode(Opcode, dl, TargetVT,
                                     MVT::Other, Ops, 8);
  }
  else {
    switch (TargetVT) {
    case MVT::i8:    Opcode = NVPTX::LD_i8_areg; break;
    case MVT::i16:   Opcode = NVPTX::LD_i16_areg; break;
    case MVT::i32:   Opcode = NVPTX::LD_i32_areg; break;
    case MVT::i64:   Opcode = NVPTX::LD_i64_areg; break;
    case MVT::f32:   Opcode = NVPTX::LD_f32_areg; break;
    case MVT::f64:   Opcode = NVPTX::LD_f64_areg; break;
    case MVT::v2i8:  Opcode = NVPTX::LD_v2i8_areg; break;
    case MVT::v2i16: Opcode = NVPTX::LD_v2i16_areg; break;
    case MVT::v2i32: Opcode = NVPTX::LD_v2i32_areg; break;
    case MVT::v2i64: Opcode = NVPTX::LD_v2i64_areg; break;
    case MVT::v2f32: Opcode = NVPTX::LD_v2f32_areg; break;
    case MVT::v2f64: Opcode = NVPTX::LD_v2f64_areg; break;
    case MVT::v4i8:  Opcode = NVPTX::LD_v4i8_areg; break;
    case MVT::v4i16: Opcode = NVPTX::LD_v4i16_areg; break;
    case MVT::v4i32: Opcode = NVPTX::LD_v4i32_areg; break;
    case MVT::v4f32: Opcode = NVPTX::LD_v4f32_areg; break;
    default: return NULL;
    }
    SDValue Ops[] = { getI32Imm(isVolatile),
                      getI32Imm(codeAddrSpace),
                      getI32Imm(vecType),
                      getI32Imm(fromType),
                      getI32Imm(fromTypeWidth),
                      N1, Chain };
    NVPTXLD = CurDAG->getMachineNode(Opcode, dl, TargetVT,
                                     MVT::Other, Ops, 7);
  }

  if (NVPTXLD != NULL) {
    MachineSDNode::mmo_iterator MemRefs0 = MF->allocateMemRefsArray(1);
    MemRefs0[0] = cast<MemSDNode>(N)->getMemOperand();
    cast<MachineSDNode>(NVPTXLD)->setMemRefs(MemRefs0, MemRefs0 + 1);
  }

  return NVPTXLD;
}

SDNode* NVPTXDAGToDAGISel::SelectStore(SDNode *N) {
  DebugLoc dl = N->getDebugLoc();
  StoreSDNode *ST = cast<StoreSDNode>(N);
  EVT StoreVT = ST->getMemoryVT();
  SDNode *NVPTXST = NULL;

  // do not support pre/post inc/dec
  if (ST->isIndexed())
    return NULL;

  if (!StoreVT.isSimple())
    return NULL;

  // Address Space Setting
  unsigned int codeAddrSpace = getCodeAddrSpace(ST, Subtarget);

  // Volatile Setting
  // - .volatile is only availalble for .global and .shared
  bool isVolatile = ST->isVolatile();
  if (codeAddrSpace != NVPTX::PTXLdStInstCode::GLOBAL &&
      codeAddrSpace != NVPTX::PTXLdStInstCode::SHARED &&
      codeAddrSpace != NVPTX::PTXLdStInstCode::GENERIC)
    isVolatile = false;

  // Vector Setting
  MVT SimpleVT = StoreVT.getSimpleVT();
  unsigned vecType = NVPTX::PTXLdStInstCode::Scalar;
  if (SimpleVT.isVector()) {
    unsigned num = SimpleVT.getVectorNumElements();
    if (num == 2)
      vecType = NVPTX::PTXLdStInstCode::V2;
    else if (num == 4)
      vecType = NVPTX::PTXLdStInstCode::V4;
    else
      return NULL;
  }

  // Type Setting: toType + toTypeWidth
  // - for integer type, always use 'u'
  //
  MVT ScalarVT = SimpleVT.getScalarType();
  unsigned toTypeWidth =  ScalarVT.getSizeInBits();
  unsigned int toType;
  if (ScalarVT.isFloatingPoint())
    toType = NVPTX::PTXLdStInstCode::Float;
  else
    toType = NVPTX::PTXLdStInstCode::Unsigned;

  // Create the machine instruction DAG
  SDValue Chain = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  SDValue Addr;
  SDValue Offset, Base;
  unsigned Opcode;
  MVT::SimpleValueType SourceVT =
      N1.getNode()->getValueType(0).getSimpleVT().SimpleTy;

  if (SelectDirectAddr(N2, Addr)) {
    switch (SourceVT) {
    case MVT::i8:    Opcode = NVPTX::ST_i8_avar; break;
    case MVT::i16:   Opcode = NVPTX::ST_i16_avar; break;
    case MVT::i32:   Opcode = NVPTX::ST_i32_avar; break;
    case MVT::i64:   Opcode = NVPTX::ST_i64_avar; break;
    case MVT::f32:   Opcode = NVPTX::ST_f32_avar; break;
    case MVT::f64:   Opcode = NVPTX::ST_f64_avar; break;
    case MVT::v2i8:  Opcode = NVPTX::ST_v2i8_avar; break;
    case MVT::v2i16: Opcode = NVPTX::ST_v2i16_avar; break;
    case MVT::v2i32: Opcode = NVPTX::ST_v2i32_avar; break;
    case MVT::v2i64: Opcode = NVPTX::ST_v2i64_avar; break;
    case MVT::v2f32: Opcode = NVPTX::ST_v2f32_avar; break;
    case MVT::v2f64: Opcode = NVPTX::ST_v2f64_avar; break;
    case MVT::v4i8:  Opcode = NVPTX::ST_v4i8_avar; break;
    case MVT::v4i16: Opcode = NVPTX::ST_v4i16_avar; break;
    case MVT::v4i32: Opcode = NVPTX::ST_v4i32_avar; break;
    case MVT::v4f32: Opcode = NVPTX::ST_v4f32_avar; break;
    default: return NULL;
    }
    SDValue Ops[] = { N1,
                      getI32Imm(isVolatile),
                      getI32Imm(codeAddrSpace),
                      getI32Imm(vecType),
                      getI32Imm(toType),
                      getI32Imm(toTypeWidth),
                      Addr, Chain };
    NVPTXST = CurDAG->getMachineNode(Opcode, dl,
                                     MVT::Other, Ops, 8);
  } else if (Subtarget.is64Bit()?
      SelectADDRsi64(N2.getNode(), N2, Base, Offset):
      SelectADDRsi(N2.getNode(), N2, Base, Offset)) {
    switch (SourceVT) {
    case MVT::i8:    Opcode = NVPTX::ST_i8_asi; break;
    case MVT::i16:   Opcode = NVPTX::ST_i16_asi; break;
    case MVT::i32:   Opcode = NVPTX::ST_i32_asi; break;
    case MVT::i64:   Opcode = NVPTX::ST_i64_asi; break;
    case MVT::f32:   Opcode = NVPTX::ST_f32_asi; break;
    case MVT::f64:   Opcode = NVPTX::ST_f64_asi; break;
    case MVT::v2i8:  Opcode = NVPTX::ST_v2i8_asi; break;
    case MVT::v2i16: Opcode = NVPTX::ST_v2i16_asi; break;
    case MVT::v2i32: Opcode = NVPTX::ST_v2i32_asi; break;
    case MVT::v2i64: Opcode = NVPTX::ST_v2i64_asi; break;
    case MVT::v2f32: Opcode = NVPTX::ST_v2f32_asi; break;
    case MVT::v2f64: Opcode = NVPTX::ST_v2f64_asi; break;
    case MVT::v4i8:  Opcode = NVPTX::ST_v4i8_asi; break;
    case MVT::v4i16: Opcode = NVPTX::ST_v4i16_asi; break;
    case MVT::v4i32: Opcode = NVPTX::ST_v4i32_asi; break;
    case MVT::v4f32: Opcode = NVPTX::ST_v4f32_asi; break;
    default: return NULL;
    }
    SDValue Ops[] = { N1,
                      getI32Imm(isVolatile),
                      getI32Imm(codeAddrSpace),
                      getI32Imm(vecType),
                      getI32Imm(toType),
                      getI32Imm(toTypeWidth),
                      Base, Offset, Chain };
    NVPTXST = CurDAG->getMachineNode(Opcode, dl,
                                     MVT::Other, Ops, 9);
  } else if (Subtarget.is64Bit()?
      SelectADDRri64(N2.getNode(), N2, Base, Offset):
      SelectADDRri(N2.getNode(), N2, Base, Offset)) {
    switch (SourceVT) {
    case MVT::i8:    Opcode = NVPTX::ST_i8_ari; break;
    case MVT::i16:   Opcode = NVPTX::ST_i16_ari; break;
    case MVT::i32:   Opcode = NVPTX::ST_i32_ari; break;
    case MVT::i64:   Opcode = NVPTX::ST_i64_ari; break;
    case MVT::f32:   Opcode = NVPTX::ST_f32_ari; break;
    case MVT::f64:   Opcode = NVPTX::ST_f64_ari; break;
    case MVT::v2i8:  Opcode = NVPTX::ST_v2i8_ari; break;
    case MVT::v2i16: Opcode = NVPTX::ST_v2i16_ari; break;
    case MVT::v2i32: Opcode = NVPTX::ST_v2i32_ari; break;
    case MVT::v2i64: Opcode = NVPTX::ST_v2i64_ari; break;
    case MVT::v2f32: Opcode = NVPTX::ST_v2f32_ari; break;
    case MVT::v2f64: Opcode = NVPTX::ST_v2f64_ari; break;
    case MVT::v4i8:  Opcode = NVPTX::ST_v4i8_ari; break;
    case MVT::v4i16: Opcode = NVPTX::ST_v4i16_ari; break;
    case MVT::v4i32: Opcode = NVPTX::ST_v4i32_ari; break;
    case MVT::v4f32: Opcode = NVPTX::ST_v4f32_ari; break;
    default: return NULL;
    }
    SDValue Ops[] = { N1,
                      getI32Imm(isVolatile),
                      getI32Imm(codeAddrSpace),
                      getI32Imm(vecType),
                      getI32Imm(toType),
                      getI32Imm(toTypeWidth),
                      Base, Offset, Chain };
    NVPTXST = CurDAG->getMachineNode(Opcode, dl,
                                     MVT::Other, Ops, 9);
  } else {
    switch (SourceVT) {
    case MVT::i8:    Opcode = NVPTX::ST_i8_areg; break;
    case MVT::i16:   Opcode = NVPTX::ST_i16_areg; break;
    case MVT::i32:   Opcode = NVPTX::ST_i32_areg; break;
    case MVT::i64:   Opcode = NVPTX::ST_i64_areg; break;
    case MVT::f32:   Opcode = NVPTX::ST_f32_areg; break;
    case MVT::f64:   Opcode = NVPTX::ST_f64_areg; break;
    case MVT::v2i8:  Opcode = NVPTX::ST_v2i8_areg; break;
    case MVT::v2i16: Opcode = NVPTX::ST_v2i16_areg; break;
    case MVT::v2i32: Opcode = NVPTX::ST_v2i32_areg; break;
    case MVT::v2i64: Opcode = NVPTX::ST_v2i64_areg; break;
    case MVT::v2f32: Opcode = NVPTX::ST_v2f32_areg; break;
    case MVT::v2f64: Opcode = NVPTX::ST_v2f64_areg; break;
    case MVT::v4i8:  Opcode = NVPTX::ST_v4i8_areg; break;
    case MVT::v4i16: Opcode = NVPTX::ST_v4i16_areg; break;
    case MVT::v4i32: Opcode = NVPTX::ST_v4i32_areg; break;
    case MVT::v4f32: Opcode = NVPTX::ST_v4f32_areg; break;
    default: return NULL;
    }
    SDValue Ops[] = { N1,
                      getI32Imm(isVolatile),
                      getI32Imm(codeAddrSpace),
                      getI32Imm(vecType),
                      getI32Imm(toType),
                      getI32Imm(toTypeWidth),
                      N2, Chain };
    NVPTXST = CurDAG->getMachineNode(Opcode, dl,
                                     MVT::Other, Ops, 8);
  }

  if (NVPTXST != NULL) {
    MachineSDNode::mmo_iterator MemRefs0 = MF->allocateMemRefsArray(1);
    MemRefs0[0] = cast<MemSDNode>(N)->getMemOperand();
    cast<MachineSDNode>(NVPTXST)->setMemRefs(MemRefs0, MemRefs0 + 1);
  }

  return NVPTXST;
}

// SelectDirectAddr - Match a direct address for DAG.
// A direct address could be a globaladdress or externalsymbol.
bool NVPTXDAGToDAGISel::SelectDirectAddr(SDValue N, SDValue &Address) {
  // Return true if TGA or ES.
  if (N.getOpcode() == ISD::TargetGlobalAddress
      || N.getOpcode() == ISD::TargetExternalSymbol) {
    Address = N;
    return true;
  }
  if (N.getOpcode() == NVPTXISD::Wrapper) {
    Address = N.getOperand(0);
    return true;
  }
  if (N.getOpcode() == ISD::INTRINSIC_WO_CHAIN) {
    unsigned IID = cast<ConstantSDNode>(N.getOperand(0))->getZExtValue();
    if (IID == Intrinsic::nvvm_ptr_gen_to_param)
      if (N.getOperand(1).getOpcode() == NVPTXISD::MoveParam)
        return (SelectDirectAddr(N.getOperand(1).getOperand(0), Address));
  }
  return false;
}

// symbol+offset
bool NVPTXDAGToDAGISel::SelectADDRsi_imp(SDNode *OpNode, SDValue Addr,
                                         SDValue &Base, SDValue &Offset,
                                         MVT mvt) {
  if (Addr.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      SDValue base=Addr.getOperand(0);
      if (SelectDirectAddr(base, Base)) {
        Offset = CurDAG->getTargetConstant(CN->getZExtValue(), mvt);
        return true;
      }
    }
  }
  return false;
}

// symbol+offset
bool NVPTXDAGToDAGISel::SelectADDRsi(SDNode *OpNode, SDValue Addr,
                                     SDValue &Base, SDValue &Offset) {
  return SelectADDRsi_imp(OpNode, Addr, Base, Offset, MVT::i32);
}

// symbol+offset
bool NVPTXDAGToDAGISel::SelectADDRsi64(SDNode *OpNode, SDValue Addr,
                                       SDValue &Base, SDValue &Offset) {
  return SelectADDRsi_imp(OpNode, Addr, Base, Offset, MVT::i64);
}

// register+offset
bool NVPTXDAGToDAGISel::SelectADDRri_imp(SDNode *OpNode, SDValue Addr,
                                         SDValue &Base, SDValue &Offset,
                                         MVT mvt) {
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), mvt);
    Offset = CurDAG->getTargetConstant(0, mvt);
    return true;
  }
  if (Addr.getOpcode() == ISD::TargetExternalSymbol ||
      Addr.getOpcode() == ISD::TargetGlobalAddress)
    return false;  // direct calls.

  if (Addr.getOpcode() == ISD::ADD) {
    if (SelectDirectAddr(Addr.getOperand(0), Addr)) {
      return false;
    }
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      if (FrameIndexSDNode *FIN =
          dyn_cast<FrameIndexSDNode>(Addr.getOperand(0)))
        // Constant offset from frame ref.
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), mvt);
      else
        Base = Addr.getOperand(0);
      Offset = CurDAG->getTargetConstant(CN->getZExtValue(), mvt);
      return true;
    }
  }
  return false;
}

// register+offset
bool NVPTXDAGToDAGISel::SelectADDRri(SDNode *OpNode, SDValue Addr,
                                     SDValue &Base, SDValue &Offset) {
  return SelectADDRri_imp(OpNode, Addr, Base, Offset, MVT::i32);
}

// register+offset
bool NVPTXDAGToDAGISel::SelectADDRri64(SDNode *OpNode, SDValue Addr,
                                       SDValue &Base, SDValue &Offset) {
  return SelectADDRri_imp(OpNode, Addr, Base, Offset, MVT::i64);
}

bool NVPTXDAGToDAGISel::ChkMemSDNodeAddressSpace(SDNode *N,
                                                 unsigned int spN) const {
  const Value *Src = NULL;
  // Even though MemIntrinsicSDNode is a subclas of MemSDNode,
  // the classof() for MemSDNode does not include MemIntrinsicSDNode
  // (See SelectionDAGNodes.h). So we need to check for both.
  if (MemSDNode *mN = dyn_cast<MemSDNode>(N)) {
    Src = mN->getSrcValue();
  }
  else if (MemSDNode *mN = dyn_cast<MemIntrinsicSDNode>(N)) {
    Src = mN->getSrcValue();
  }
  if (!Src)
    return false;
  if (const PointerType *PT = dyn_cast<PointerType>(Src->getType()))
    return (PT->getAddressSpace() == spN);
  return false;
}

/// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
/// inline asm expressions.
bool NVPTXDAGToDAGISel::SelectInlineAsmMemoryOperand(const SDValue &Op,
                                                     char ConstraintCode,
                                                 std::vector<SDValue> &OutOps) {
  SDValue Op0, Op1;
  switch (ConstraintCode) {
  default: return true;
  case 'm':   // memory
    if (SelectDirectAddr(Op, Op0)) {
      OutOps.push_back(Op0);
      OutOps.push_back(CurDAG->getTargetConstant(0, MVT::i32));
      return false;
    }
    if (SelectADDRri(Op.getNode(), Op, Op0, Op1)) {
      OutOps.push_back(Op0);
      OutOps.push_back(Op1);
      return false;
    }
    break;
  }
  return true;
}

// Return true if N is a undef or a constant.
// If N was undef, return a (i8imm 0) in Retval
// If N was imm, convert it to i8imm and return in Retval
// Note: The convert to i8imm is required, otherwise the
// pattern matcher inserts a bunch of IMOVi8rr to convert
// the imm to i8imm, and this causes instruction selection
// to fail.
bool NVPTXDAGToDAGISel::UndefOrImm(SDValue Op, SDValue N,
                                   SDValue &Retval) {
  if (!(N.getOpcode() == ISD::UNDEF) &&
      !(N.getOpcode() == ISD::Constant))
    return false;

  if (N.getOpcode() == ISD::UNDEF)
    Retval = CurDAG->getTargetConstant(0, MVT::i8);
  else {
    ConstantSDNode *cn = cast<ConstantSDNode>(N.getNode());
    unsigned retval = cn->getZExtValue();
    Retval = CurDAG->getTargetConstant(retval, MVT::i8);
  }
  return true;
}
