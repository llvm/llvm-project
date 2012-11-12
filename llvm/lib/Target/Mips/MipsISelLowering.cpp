//===-- MipsISelLowering.cpp - Mips DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Mips uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-lower"
#include "MipsISelLowering.h"
#include "MipsMachineFunction.h"
#include "MipsTargetMachine.h"
#include "MipsTargetObjectFile.h"
#include "MipsSubtarget.h"
#include "InstPrinter/MipsInstPrinter.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(NumTailCalls, "Number of tail calls");

static cl::opt<bool>
EnableMipsTailCalls("enable-mips-tail-calls", cl::Hidden,
                    cl::desc("MIPS: Enable tail calls."), cl::init(false));

static const uint16_t O32IntRegs[4] = {
  Mips::A0, Mips::A1, Mips::A2, Mips::A3
};

static const uint16_t Mips64IntRegs[8] = {
  Mips::A0_64, Mips::A1_64, Mips::A2_64, Mips::A3_64,
  Mips::T0_64, Mips::T1_64, Mips::T2_64, Mips::T3_64
};

static const uint16_t Mips64DPRegs[8] = {
  Mips::D12_64, Mips::D13_64, Mips::D14_64, Mips::D15_64,
  Mips::D16_64, Mips::D17_64, Mips::D18_64, Mips::D19_64
};

// If I is a shifted mask, set the size (Size) and the first bit of the
// mask (Pos), and return true.
// For example, if I is 0x003ff800, (Pos, Size) = (11, 11).
static bool IsShiftedMask(uint64_t I, uint64_t &Pos, uint64_t &Size) {
  if (!isShiftedMask_64(I))
     return false;

  Size = CountPopulation_64(I);
  Pos = CountTrailingZeros_64(I);
  return true;
}

static SDValue GetGlobalReg(SelectionDAG &DAG, EVT Ty) {
  MipsFunctionInfo *FI = DAG.getMachineFunction().getInfo<MipsFunctionInfo>();
  return DAG.getRegister(FI->getGlobalBaseReg(), Ty);
}

const char *MipsTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case MipsISD::JmpLink:           return "MipsISD::JmpLink";
  case MipsISD::TailCall:          return "MipsISD::TailCall";
  case MipsISD::Hi:                return "MipsISD::Hi";
  case MipsISD::Lo:                return "MipsISD::Lo";
  case MipsISD::GPRel:             return "MipsISD::GPRel";
  case MipsISD::ThreadPointer:     return "MipsISD::ThreadPointer";
  case MipsISD::Ret:               return "MipsISD::Ret";
  case MipsISD::FPBrcond:          return "MipsISD::FPBrcond";
  case MipsISD::FPCmp:             return "MipsISD::FPCmp";
  case MipsISD::CMovFP_T:          return "MipsISD::CMovFP_T";
  case MipsISD::CMovFP_F:          return "MipsISD::CMovFP_F";
  case MipsISD::FPRound:           return "MipsISD::FPRound";
  case MipsISD::MAdd:              return "MipsISD::MAdd";
  case MipsISD::MAddu:             return "MipsISD::MAddu";
  case MipsISD::MSub:              return "MipsISD::MSub";
  case MipsISD::MSubu:             return "MipsISD::MSubu";
  case MipsISD::DivRem:            return "MipsISD::DivRem";
  case MipsISD::DivRemU:           return "MipsISD::DivRemU";
  case MipsISD::BuildPairF64:      return "MipsISD::BuildPairF64";
  case MipsISD::ExtractElementF64: return "MipsISD::ExtractElementF64";
  case MipsISD::Wrapper:           return "MipsISD::Wrapper";
  case MipsISD::DynAlloc:          return "MipsISD::DynAlloc";
  case MipsISD::Sync:              return "MipsISD::Sync";
  case MipsISD::Ext:               return "MipsISD::Ext";
  case MipsISD::Ins:               return "MipsISD::Ins";
  case MipsISD::LWL:               return "MipsISD::LWL";
  case MipsISD::LWR:               return "MipsISD::LWR";
  case MipsISD::SWL:               return "MipsISD::SWL";
  case MipsISD::SWR:               return "MipsISD::SWR";
  case MipsISD::LDL:               return "MipsISD::LDL";
  case MipsISD::LDR:               return "MipsISD::LDR";
  case MipsISD::SDL:               return "MipsISD::SDL";
  case MipsISD::SDR:               return "MipsISD::SDR";
  case MipsISD::EXTP:              return "MipsISD::EXTP";
  case MipsISD::EXTPDP:            return "MipsISD::EXTPDP";
  case MipsISD::EXTR_S_H:          return "MipsISD::EXTR_S_H";
  case MipsISD::EXTR_W:            return "MipsISD::EXTR_W";
  case MipsISD::EXTR_R_W:          return "MipsISD::EXTR_R_W";
  case MipsISD::EXTR_RS_W:         return "MipsISD::EXTR_RS_W";
  case MipsISD::SHILO:             return "MipsISD::SHILO";
  case MipsISD::MTHLIP:            return "MipsISD::MTHLIP";
  case MipsISD::MULT:              return "MipsISD::MULT";
  case MipsISD::MULTU:             return "MipsISD::MULTU";
  case MipsISD::MADD_DSP:          return "MipsISD::MADD_DSPDSP";
  case MipsISD::MADDU_DSP:         return "MipsISD::MADDU_DSP";
  case MipsISD::MSUB_DSP:          return "MipsISD::MSUB_DSP";
  case MipsISD::MSUBU_DSP:         return "MipsISD::MSUBU_DSP";
  default:                         return NULL;
  }
}

MipsTargetLowering::
MipsTargetLowering(MipsTargetMachine &TM)
  : TargetLowering(TM, new MipsTargetObjectFile()),
    Subtarget(&TM.getSubtarget<MipsSubtarget>()),
    HasMips64(Subtarget->hasMips64()), IsN64(Subtarget->isABI_N64()),
    IsO32(Subtarget->isABI_O32()) {

  // Mips does not have i1 type, so use i32 for
  // setcc operations results (slt, sgt, ...).
  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent); // FIXME: Is this correct?

  // Set up the register classes
  addRegisterClass(MVT::i32, &Mips::CPURegsRegClass);

  if (HasMips64)
    addRegisterClass(MVT::i64, &Mips::CPU64RegsRegClass);

  if (Subtarget->inMips16Mode()) {
    addRegisterClass(MVT::i32, &Mips::CPU16RegsRegClass);
  }

  if (Subtarget->hasDSP()) {
    MVT::SimpleValueType VecTys[2] = {MVT::v2i16, MVT::v4i8};

    for (unsigned i = 0; i < array_lengthof(VecTys); ++i) {
      addRegisterClass(VecTys[i], &Mips::DSPRegsRegClass);

      // Expand all builtin opcodes.
      for (unsigned Opc = 0; Opc < ISD::BUILTIN_OP_END; ++Opc)
        setOperationAction(Opc, VecTys[i], Expand);

      setOperationAction(ISD::LOAD, VecTys[i], Legal);
      setOperationAction(ISD::STORE, VecTys[i], Legal);
      setOperationAction(ISD::BITCAST, VecTys[i], Legal);
    }
  }

  if (!TM.Options.UseSoftFloat) {
    addRegisterClass(MVT::f32, &Mips::FGR32RegClass);

    // When dealing with single precision only, use libcalls
    if (!Subtarget->isSingleFloat()) {
      if (HasMips64)
        addRegisterClass(MVT::f64, &Mips::FGR64RegClass);
      else
        addRegisterClass(MVT::f64, &Mips::AFGR64RegClass);
    }
  }

  // Load extented operations for i1 types must be promoted
  setLoadExtAction(ISD::EXTLOAD,  MVT::i1,  Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1,  Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1,  Promote);

  // MIPS doesn't have extending float->double load/store
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // Used by legalize types to correctly generate the setcc result.
  // Without this, every float setcc comes with a AND/OR with the result,
  // we don't want this, since the fpcmp result goes to a flag register,
  // which is used implicitly by brcond and select operations.
  AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);

  // Mips Custom Operations
  setOperationAction(ISD::GlobalAddress,      MVT::i32,   Custom);
  setOperationAction(ISD::BlockAddress,       MVT::i32,   Custom);
  setOperationAction(ISD::GlobalTLSAddress,   MVT::i32,   Custom);
  setOperationAction(ISD::JumpTable,          MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,       MVT::i32,   Custom);
  setOperationAction(ISD::SELECT,             MVT::f32,   Custom);
  setOperationAction(ISD::SELECT,             MVT::f64,   Custom);
  setOperationAction(ISD::SELECT,             MVT::i32,   Custom);
  setOperationAction(ISD::SELECT_CC,          MVT::f32,   Custom);
  setOperationAction(ISD::SELECT_CC,          MVT::f64,   Custom);
  setOperationAction(ISD::SETCC,              MVT::f32,   Custom);
  setOperationAction(ISD::SETCC,              MVT::f64,   Custom);
  setOperationAction(ISD::BRCOND,             MVT::Other, Custom);
  setOperationAction(ISD::VASTART,            MVT::Other, Custom);
  setOperationAction(ISD::FCOPYSIGN,          MVT::f32,   Custom);
  setOperationAction(ISD::FCOPYSIGN,          MVT::f64,   Custom);
  if (Subtarget->inMips16Mode()) {
    setOperationAction(ISD::MEMBARRIER,         MVT::Other, Expand);
    setOperationAction(ISD::ATOMIC_FENCE,       MVT::Other, Expand);
  }
  else {
    setOperationAction(ISD::MEMBARRIER,         MVT::Other, Custom);
    setOperationAction(ISD::ATOMIC_FENCE,       MVT::Other, Custom);
  }
  if (!Subtarget->inMips16Mode()) {
    setOperationAction(ISD::LOAD,               MVT::i32, Custom);
    setOperationAction(ISD::STORE,              MVT::i32, Custom);
  }

  if (!TM.Options.NoNaNsFPMath) {
    setOperationAction(ISD::FABS,             MVT::f32,   Custom);
    setOperationAction(ISD::FABS,             MVT::f64,   Custom);
  }

  if (HasMips64) {
    setOperationAction(ISD::GlobalAddress,      MVT::i64,   Custom);
    setOperationAction(ISD::BlockAddress,       MVT::i64,   Custom);
    setOperationAction(ISD::GlobalTLSAddress,   MVT::i64,   Custom);
    setOperationAction(ISD::JumpTable,          MVT::i64,   Custom);
    setOperationAction(ISD::ConstantPool,       MVT::i64,   Custom);
    setOperationAction(ISD::SELECT,             MVT::i64,   Custom);
    setOperationAction(ISD::LOAD,               MVT::i64,   Custom);
    setOperationAction(ISD::STORE,              MVT::i64,   Custom);
  }

  if (!HasMips64) {
    setOperationAction(ISD::SHL_PARTS,          MVT::i32,   Custom);
    setOperationAction(ISD::SRA_PARTS,          MVT::i32,   Custom);
    setOperationAction(ISD::SRL_PARTS,          MVT::i32,   Custom);
  }

  setOperationAction(ISD::ADD,                MVT::i32,   Custom);
  if (HasMips64)
    setOperationAction(ISD::ADD,                MVT::i64,   Custom);

  setOperationAction(ISD::SDIV, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIV, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIV, MVT::i64, Expand);
  setOperationAction(ISD::SREM, MVT::i64, Expand);
  setOperationAction(ISD::UDIV, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);

  // Operations not directly supported by Mips.
  setOperationAction(ISD::BR_JT,             MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,             MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC,         MVT::Other, Expand);
  setOperationAction(ISD::UINT_TO_FP,        MVT::i32,   Expand);
  setOperationAction(ISD::UINT_TO_FP,        MVT::i64,   Expand);
  setOperationAction(ISD::FP_TO_UINT,        MVT::i32,   Expand);
  setOperationAction(ISD::FP_TO_UINT,        MVT::i64,   Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1,    Expand);
  setOperationAction(ISD::CTPOP,             MVT::i32,   Expand);
  setOperationAction(ISD::CTPOP,             MVT::i64,   Expand);
  setOperationAction(ISD::CTTZ,              MVT::i32,   Expand);
  setOperationAction(ISD::CTTZ,              MVT::i64,   Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF,   MVT::i32,   Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF,   MVT::i64,   Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF,   MVT::i32,   Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF,   MVT::i64,   Expand);
  setOperationAction(ISD::ROTL,              MVT::i32,   Expand);
  setOperationAction(ISD::ROTL,              MVT::i64,   Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32,  Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64,  Expand);

  if (!Subtarget->hasMips32r2())
    setOperationAction(ISD::ROTR, MVT::i32,   Expand);

  if (!Subtarget->hasMips64r2())
    setOperationAction(ISD::ROTR, MVT::i64,   Expand);

  setOperationAction(ISD::FSIN,              MVT::f32,   Expand);
  setOperationAction(ISD::FSIN,              MVT::f64,   Expand);
  setOperationAction(ISD::FCOS,              MVT::f32,   Expand);
  setOperationAction(ISD::FCOS,              MVT::f64,   Expand);
  setOperationAction(ISD::FPOWI,             MVT::f32,   Expand);
  setOperationAction(ISD::FPOW,              MVT::f32,   Expand);
  setOperationAction(ISD::FPOW,              MVT::f64,   Expand);
  setOperationAction(ISD::FLOG,              MVT::f32,   Expand);
  setOperationAction(ISD::FLOG2,             MVT::f32,   Expand);
  setOperationAction(ISD::FLOG10,            MVT::f32,   Expand);
  setOperationAction(ISD::FEXP,              MVT::f32,   Expand);
  setOperationAction(ISD::FMA,               MVT::f32,   Expand);
  setOperationAction(ISD::FMA,               MVT::f64,   Expand);
  setOperationAction(ISD::FREM,              MVT::f32,   Expand);
  setOperationAction(ISD::FREM,              MVT::f64,   Expand);

  if (!TM.Options.NoNaNsFPMath) {
    setOperationAction(ISD::FNEG,             MVT::f32,   Expand);
    setOperationAction(ISD::FNEG,             MVT::f64,   Expand);
  }

  setOperationAction(ISD::EXCEPTIONADDR,     MVT::i32, Expand);
  setOperationAction(ISD::EXCEPTIONADDR,     MVT::i64, Expand);
  setOperationAction(ISD::EHSELECTION,       MVT::i32, Expand);
  setOperationAction(ISD::EHSELECTION,       MVT::i64, Expand);

  setOperationAction(ISD::VAARG,             MVT::Other, Expand);
  setOperationAction(ISD::VACOPY,            MVT::Other, Expand);
  setOperationAction(ISD::VAEND,             MVT::Other, Expand);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i64, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i64, Custom);

  // Use the default for now
  setOperationAction(ISD::STACKSAVE,         MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE,      MVT::Other, Expand);

  setOperationAction(ISD::ATOMIC_LOAD,       MVT::i32,    Expand);
  setOperationAction(ISD::ATOMIC_LOAD,       MVT::i64,    Expand);
  setOperationAction(ISD::ATOMIC_STORE,      MVT::i32,    Expand);
  setOperationAction(ISD::ATOMIC_STORE,      MVT::i64,    Expand);

  if (Subtarget->inMips16Mode()) {
    setOperationAction(ISD::ATOMIC_CMP_SWAP,       MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_SWAP,           MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_ADD,       MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_SUB,       MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_AND,       MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_OR,        MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_XOR,       MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_NAND,      MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_MIN,       MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_MAX,       MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_UMIN,      MVT::i32,    Expand);
    setOperationAction(ISD::ATOMIC_LOAD_UMAX,      MVT::i32,    Expand);
  }

  setInsertFencesForAtomic(true);

  if (!Subtarget->hasSEInReg()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8,  Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  }

  if (!Subtarget->hasBitCount()) {
    setOperationAction(ISD::CTLZ, MVT::i32, Expand);
    setOperationAction(ISD::CTLZ, MVT::i64, Expand);
  }

  if (!Subtarget->hasSwap()) {
    setOperationAction(ISD::BSWAP, MVT::i32, Expand);
    setOperationAction(ISD::BSWAP, MVT::i64, Expand);
  }

  if (HasMips64) {
    setLoadExtAction(ISD::SEXTLOAD, MVT::i32, Custom);
    setLoadExtAction(ISD::ZEXTLOAD, MVT::i32, Custom);
    setLoadExtAction(ISD::EXTLOAD, MVT::i32, Custom);
    setTruncStoreAction(MVT::i64, MVT::i32, Custom);
  }

  setTargetDAGCombine(ISD::ADDE);
  setTargetDAGCombine(ISD::SUBE);
  setTargetDAGCombine(ISD::SDIVREM);
  setTargetDAGCombine(ISD::UDIVREM);
  setTargetDAGCombine(ISD::SELECT);
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::ADD);

  setMinFunctionAlignment(HasMips64 ? 3 : 2);

  setStackPointerRegisterToSaveRestore(IsN64 ? Mips::SP_64 : Mips::SP);
  computeRegisterProperties();

  setExceptionPointerRegister(IsN64 ? Mips::A0_64 : Mips::A0);
  setExceptionSelectorRegister(IsN64 ? Mips::A1_64 : Mips::A1);

  maxStoresPerMemcpy = 16;
}

bool MipsTargetLowering::allowsUnalignedMemoryAccesses(EVT VT) const {
  MVT::SimpleValueType SVT = VT.getSimpleVT().SimpleTy;

  if (Subtarget->inMips16Mode())
    return false;

  switch (SVT) {
  case MVT::i64:
  case MVT::i32:
    return true;
  default:
    return false;
  }
}

EVT MipsTargetLowering::getSetCCResultType(EVT VT) const {
  return MVT::i32;
}

// SelectMadd -
// Transforms a subgraph in CurDAG if the following pattern is found:
//  (addc multLo, Lo0), (adde multHi, Hi0),
// where,
//  multHi/Lo: product of multiplication
//  Lo0: initial value of Lo register
//  Hi0: initial value of Hi register
// Return true if pattern matching was successful.
static bool SelectMadd(SDNode *ADDENode, SelectionDAG *CurDAG) {
  // ADDENode's second operand must be a flag output of an ADDC node in order
  // for the matching to be successful.
  SDNode *ADDCNode = ADDENode->getOperand(2).getNode();

  if (ADDCNode->getOpcode() != ISD::ADDC)
    return false;

  SDValue MultHi = ADDENode->getOperand(0);
  SDValue MultLo = ADDCNode->getOperand(0);
  SDNode *MultNode = MultHi.getNode();
  unsigned MultOpc = MultHi.getOpcode();

  // MultHi and MultLo must be generated by the same node,
  if (MultLo.getNode() != MultNode)
    return false;

  // and it must be a multiplication.
  if (MultOpc != ISD::SMUL_LOHI && MultOpc != ISD::UMUL_LOHI)
    return false;

  // MultLo amd MultHi must be the first and second output of MultNode
  // respectively.
  if (MultHi.getResNo() != 1 || MultLo.getResNo() != 0)
    return false;

  // Transform this to a MADD only if ADDENode and ADDCNode are the only users
  // of the values of MultNode, in which case MultNode will be removed in later
  // phases.
  // If there exist users other than ADDENode or ADDCNode, this function returns
  // here, which will result in MultNode being mapped to a single MULT
  // instruction node rather than a pair of MULT and MADD instructions being
  // produced.
  if (!MultHi.hasOneUse() || !MultLo.hasOneUse())
    return false;

  SDValue Chain = CurDAG->getEntryNode();
  DebugLoc dl = ADDENode->getDebugLoc();

  // create MipsMAdd(u) node
  MultOpc = MultOpc == ISD::UMUL_LOHI ? MipsISD::MAddu : MipsISD::MAdd;

  SDValue MAdd = CurDAG->getNode(MultOpc, dl, MVT::Glue,
                                 MultNode->getOperand(0),// Factor 0
                                 MultNode->getOperand(1),// Factor 1
                                 ADDCNode->getOperand(1),// Lo0
                                 ADDENode->getOperand(1));// Hi0

  // create CopyFromReg nodes
  SDValue CopyFromLo = CurDAG->getCopyFromReg(Chain, dl, Mips::LO, MVT::i32,
                                              MAdd);
  SDValue CopyFromHi = CurDAG->getCopyFromReg(CopyFromLo.getValue(1), dl,
                                              Mips::HI, MVT::i32,
                                              CopyFromLo.getValue(2));

  // replace uses of adde and addc here
  if (!SDValue(ADDCNode, 0).use_empty())
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(ADDCNode, 0), CopyFromLo);

  if (!SDValue(ADDENode, 0).use_empty())
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(ADDENode, 0), CopyFromHi);

  return true;
}

// SelectMsub -
// Transforms a subgraph in CurDAG if the following pattern is found:
//  (addc Lo0, multLo), (sube Hi0, multHi),
// where,
//  multHi/Lo: product of multiplication
//  Lo0: initial value of Lo register
//  Hi0: initial value of Hi register
// Return true if pattern matching was successful.
static bool SelectMsub(SDNode *SUBENode, SelectionDAG *CurDAG) {
  // SUBENode's second operand must be a flag output of an SUBC node in order
  // for the matching to be successful.
  SDNode *SUBCNode = SUBENode->getOperand(2).getNode();

  if (SUBCNode->getOpcode() != ISD::SUBC)
    return false;

  SDValue MultHi = SUBENode->getOperand(1);
  SDValue MultLo = SUBCNode->getOperand(1);
  SDNode *MultNode = MultHi.getNode();
  unsigned MultOpc = MultHi.getOpcode();

  // MultHi and MultLo must be generated by the same node,
  if (MultLo.getNode() != MultNode)
    return false;

  // and it must be a multiplication.
  if (MultOpc != ISD::SMUL_LOHI && MultOpc != ISD::UMUL_LOHI)
    return false;

  // MultLo amd MultHi must be the first and second output of MultNode
  // respectively.
  if (MultHi.getResNo() != 1 || MultLo.getResNo() != 0)
    return false;

  // Transform this to a MSUB only if SUBENode and SUBCNode are the only users
  // of the values of MultNode, in which case MultNode will be removed in later
  // phases.
  // If there exist users other than SUBENode or SUBCNode, this function returns
  // here, which will result in MultNode being mapped to a single MULT
  // instruction node rather than a pair of MULT and MSUB instructions being
  // produced.
  if (!MultHi.hasOneUse() || !MultLo.hasOneUse())
    return false;

  SDValue Chain = CurDAG->getEntryNode();
  DebugLoc dl = SUBENode->getDebugLoc();

  // create MipsSub(u) node
  MultOpc = MultOpc == ISD::UMUL_LOHI ? MipsISD::MSubu : MipsISD::MSub;

  SDValue MSub = CurDAG->getNode(MultOpc, dl, MVT::Glue,
                                 MultNode->getOperand(0),// Factor 0
                                 MultNode->getOperand(1),// Factor 1
                                 SUBCNode->getOperand(0),// Lo0
                                 SUBENode->getOperand(0));// Hi0

  // create CopyFromReg nodes
  SDValue CopyFromLo = CurDAG->getCopyFromReg(Chain, dl, Mips::LO, MVT::i32,
                                              MSub);
  SDValue CopyFromHi = CurDAG->getCopyFromReg(CopyFromLo.getValue(1), dl,
                                              Mips::HI, MVT::i32,
                                              CopyFromLo.getValue(2));

  // replace uses of sube and subc here
  if (!SDValue(SUBCNode, 0).use_empty())
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(SUBCNode, 0), CopyFromLo);

  if (!SDValue(SUBENode, 0).use_empty())
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(SUBENode, 0), CopyFromHi);

  return true;
}

static SDValue PerformADDECombine(SDNode *N, SelectionDAG &DAG,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const MipsSubtarget *Subtarget) {
  if (DCI.isBeforeLegalize())
    return SDValue();

  if (Subtarget->hasMips32() && N->getValueType(0) == MVT::i32 &&
      SelectMadd(N, &DAG))
    return SDValue(N, 0);

  return SDValue();
}

static SDValue PerformSUBECombine(SDNode *N, SelectionDAG &DAG,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const MipsSubtarget *Subtarget) {
  if (DCI.isBeforeLegalize())
    return SDValue();

  if (Subtarget->hasMips32() && N->getValueType(0) == MVT::i32 &&
      SelectMsub(N, &DAG))
    return SDValue(N, 0);

  return SDValue();
}

static SDValue PerformDivRemCombine(SDNode *N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const MipsSubtarget *Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  EVT Ty = N->getValueType(0);
  unsigned LO = (Ty == MVT::i32) ? Mips::LO : Mips::LO64;
  unsigned HI = (Ty == MVT::i32) ? Mips::HI : Mips::HI64;
  unsigned opc = N->getOpcode() == ISD::SDIVREM ? MipsISD::DivRem :
                                                  MipsISD::DivRemU;
  DebugLoc dl = N->getDebugLoc();

  SDValue DivRem = DAG.getNode(opc, dl, MVT::Glue,
                               N->getOperand(0), N->getOperand(1));
  SDValue InChain = DAG.getEntryNode();
  SDValue InGlue = DivRem;

  // insert MFLO
  if (N->hasAnyUseOfValue(0)) {
    SDValue CopyFromLo = DAG.getCopyFromReg(InChain, dl, LO, Ty,
                                            InGlue);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), CopyFromLo);
    InChain = CopyFromLo.getValue(1);
    InGlue = CopyFromLo.getValue(2);
  }

  // insert MFHI
  if (N->hasAnyUseOfValue(1)) {
    SDValue CopyFromHi = DAG.getCopyFromReg(InChain, dl,
                                            HI, Ty, InGlue);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), CopyFromHi);
  }

  return SDValue();
}

static Mips::CondCode FPCondCCodeToFCC(ISD::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Unknown fp condition code!");
  case ISD::SETEQ:
  case ISD::SETOEQ: return Mips::FCOND_OEQ;
  case ISD::SETUNE: return Mips::FCOND_UNE;
  case ISD::SETLT:
  case ISD::SETOLT: return Mips::FCOND_OLT;
  case ISD::SETGT:
  case ISD::SETOGT: return Mips::FCOND_OGT;
  case ISD::SETLE:
  case ISD::SETOLE: return Mips::FCOND_OLE;
  case ISD::SETGE:
  case ISD::SETOGE: return Mips::FCOND_OGE;
  case ISD::SETULT: return Mips::FCOND_ULT;
  case ISD::SETULE: return Mips::FCOND_ULE;
  case ISD::SETUGT: return Mips::FCOND_UGT;
  case ISD::SETUGE: return Mips::FCOND_UGE;
  case ISD::SETUO:  return Mips::FCOND_UN;
  case ISD::SETO:   return Mips::FCOND_OR;
  case ISD::SETNE:
  case ISD::SETONE: return Mips::FCOND_ONE;
  case ISD::SETUEQ: return Mips::FCOND_UEQ;
  }
}


// Returns true if condition code has to be inverted.
static bool InvertFPCondCode(Mips::CondCode CC) {
  if (CC >= Mips::FCOND_F && CC <= Mips::FCOND_NGT)
    return false;

  assert((CC >= Mips::FCOND_T && CC <= Mips::FCOND_GT) &&
         "Illegal Condition Code");

  return true;
}

// Creates and returns an FPCmp node from a setcc node.
// Returns Op if setcc is not a floating point comparison.
static SDValue CreateFPCmp(SelectionDAG &DAG, const SDValue &Op) {
  // must be a SETCC node
  if (Op.getOpcode() != ISD::SETCC)
    return Op;

  SDValue LHS = Op.getOperand(0);

  if (!LHS.getValueType().isFloatingPoint())
    return Op;

  SDValue RHS = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();

  // Assume the 3rd operand is a CondCodeSDNode. Add code to check the type of
  // node if necessary.
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();

  return DAG.getNode(MipsISD::FPCmp, dl, MVT::Glue, LHS, RHS,
                     DAG.getConstant(FPCondCCodeToFCC(CC), MVT::i32));
}

// Creates and returns a CMovFPT/F node.
static SDValue CreateCMovFP(SelectionDAG &DAG, SDValue Cond, SDValue True,
                            SDValue False, DebugLoc DL) {
  bool invert = InvertFPCondCode((Mips::CondCode)
                                 cast<ConstantSDNode>(Cond.getOperand(2))
                                 ->getSExtValue());

  return DAG.getNode((invert ? MipsISD::CMovFP_F : MipsISD::CMovFP_T), DL,
                     True.getValueType(), True, False, Cond);
}

static SDValue PerformSELECTCombine(SDNode *N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const MipsSubtarget *Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SDValue SetCC = N->getOperand(0);

  if ((SetCC.getOpcode() != ISD::SETCC) ||
      !SetCC.getOperand(0).getValueType().isInteger())
    return SDValue();

  SDValue False = N->getOperand(2);
  EVT FalseTy = False.getValueType();

  if (!FalseTy.isInteger())
    return SDValue();

  ConstantSDNode *CN = dyn_cast<ConstantSDNode>(False);

  if (!CN || CN->getZExtValue())
    return SDValue();

  const DebugLoc DL = N->getDebugLoc();
  ISD::CondCode CC = cast<CondCodeSDNode>(SetCC.getOperand(2))->get();
  SDValue True = N->getOperand(1);

  SetCC = DAG.getSetCC(DL, SetCC.getValueType(), SetCC.getOperand(0),
                       SetCC.getOperand(1), ISD::getSetCCInverse(CC, true));

  return DAG.getNode(ISD::SELECT, DL, FalseTy, SetCC, False, True);
}

static SDValue PerformANDCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const MipsSubtarget *Subtarget) {
  // Pattern match EXT.
  //  $dst = and ((sra or srl) $src , pos), (2**size - 1)
  //  => ext $dst, $src, size, pos
  if (DCI.isBeforeLegalizeOps() || !Subtarget->hasMips32r2())
    return SDValue();

  SDValue ShiftRight = N->getOperand(0), Mask = N->getOperand(1);
  unsigned ShiftRightOpc = ShiftRight.getOpcode();

  // Op's first operand must be a shift right.
  if (ShiftRightOpc != ISD::SRA && ShiftRightOpc != ISD::SRL)
    return SDValue();

  // The second operand of the shift must be an immediate.
  ConstantSDNode *CN;
  if (!(CN = dyn_cast<ConstantSDNode>(ShiftRight.getOperand(1))))
    return SDValue();

  uint64_t Pos = CN->getZExtValue();
  uint64_t SMPos, SMSize;

  // Op's second operand must be a shifted mask.
  if (!(CN = dyn_cast<ConstantSDNode>(Mask)) ||
      !IsShiftedMask(CN->getZExtValue(), SMPos, SMSize))
    return SDValue();

  // Return if the shifted mask does not start at bit 0 or the sum of its size
  // and Pos exceeds the word's size.
  EVT ValTy = N->getValueType(0);
  if (SMPos != 0 || Pos + SMSize > ValTy.getSizeInBits())
    return SDValue();

  return DAG.getNode(MipsISD::Ext, N->getDebugLoc(), ValTy,
                     ShiftRight.getOperand(0), DAG.getConstant(Pos, MVT::i32),
                     DAG.getConstant(SMSize, MVT::i32));
}

static SDValue PerformORCombine(SDNode *N, SelectionDAG &DAG,
                                TargetLowering::DAGCombinerInfo &DCI,
                                const MipsSubtarget *Subtarget) {
  // Pattern match INS.
  //  $dst = or (and $src1 , mask0), (and (shl $src, pos), mask1),
  //  where mask1 = (2**size - 1) << pos, mask0 = ~mask1
  //  => ins $dst, $src, size, pos, $src1
  if (DCI.isBeforeLegalizeOps() || !Subtarget->hasMips32r2())
    return SDValue();

  SDValue And0 = N->getOperand(0), And1 = N->getOperand(1);
  uint64_t SMPos0, SMSize0, SMPos1, SMSize1;
  ConstantSDNode *CN;

  // See if Op's first operand matches (and $src1 , mask0).
  if (And0.getOpcode() != ISD::AND)
    return SDValue();

  if (!(CN = dyn_cast<ConstantSDNode>(And0.getOperand(1))) ||
      !IsShiftedMask(~CN->getSExtValue(), SMPos0, SMSize0))
    return SDValue();

  // See if Op's second operand matches (and (shl $src, pos), mask1).
  if (And1.getOpcode() != ISD::AND)
    return SDValue();

  if (!(CN = dyn_cast<ConstantSDNode>(And1.getOperand(1))) ||
      !IsShiftedMask(CN->getZExtValue(), SMPos1, SMSize1))
    return SDValue();

  // The shift masks must have the same position and size.
  if (SMPos0 != SMPos1 || SMSize0 != SMSize1)
    return SDValue();

  SDValue Shl = And1.getOperand(0);
  if (Shl.getOpcode() != ISD::SHL)
    return SDValue();

  if (!(CN = dyn_cast<ConstantSDNode>(Shl.getOperand(1))))
    return SDValue();

  unsigned Shamt = CN->getZExtValue();

  // Return if the shift amount and the first bit position of mask are not the
  // same.
  EVT ValTy = N->getValueType(0);
  if ((Shamt != SMPos0) || (SMPos0 + SMSize0 > ValTy.getSizeInBits()))
    return SDValue();

  return DAG.getNode(MipsISD::Ins, N->getDebugLoc(), ValTy, Shl.getOperand(0),
                     DAG.getConstant(SMPos0, MVT::i32),
                     DAG.getConstant(SMSize0, MVT::i32), And0.getOperand(0));
}

static SDValue PerformADDCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const MipsSubtarget *Subtarget) {
  // (add v0, (add v1, abs_lo(tjt))) => (add (add v0, v1), abs_lo(tjt))

  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SDValue Add = N->getOperand(1);

  if (Add.getOpcode() != ISD::ADD)
    return SDValue();

  SDValue Lo = Add.getOperand(1);

  if ((Lo.getOpcode() != MipsISD::Lo) ||
      (Lo.getOperand(0).getOpcode() != ISD::TargetJumpTable))
    return SDValue();

  EVT ValTy = N->getValueType(0);
  DebugLoc DL = N->getDebugLoc();

  SDValue Add1 = DAG.getNode(ISD::ADD, DL, ValTy, N->getOperand(0),
                             Add.getOperand(0));
  return DAG.getNode(ISD::ADD, DL, ValTy, Add1, Lo);
}

SDValue  MipsTargetLowering::PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI)
  const {
  SelectionDAG &DAG = DCI.DAG;
  unsigned opc = N->getOpcode();

  switch (opc) {
  default: break;
  case ISD::ADDE:
    return PerformADDECombine(N, DAG, DCI, Subtarget);
  case ISD::SUBE:
    return PerformSUBECombine(N, DAG, DCI, Subtarget);
  case ISD::SDIVREM:
  case ISD::UDIVREM:
    return PerformDivRemCombine(N, DAG, DCI, Subtarget);
  case ISD::SELECT:
    return PerformSELECTCombine(N, DAG, DCI, Subtarget);
  case ISD::AND:
    return PerformANDCombine(N, DAG, DCI, Subtarget);
  case ISD::OR:
    return PerformORCombine(N, DAG, DCI, Subtarget);
  case ISD::ADD:
    return PerformADDCombine(N, DAG, DCI, Subtarget);
  }

  return SDValue();
}

void
MipsTargetLowering::LowerOperationWrapper(SDNode *N,
                                          SmallVectorImpl<SDValue> &Results,
                                          SelectionDAG &DAG) const {
  SDValue Res = LowerOperation(SDValue(N, 0), DAG);

  for (unsigned I = 0, E = Res->getNumValues(); I != E; ++I)
    Results.push_back(Res.getValue(I));
}

void
MipsTargetLowering::ReplaceNodeResults(SDNode *N,
                                       SmallVectorImpl<SDValue> &Results,
                                       SelectionDAG &DAG) const {
  SDValue Res = LowerOperation(SDValue(N, 0), DAG);

  for (unsigned I = 0, E = Res->getNumValues(); I != E; ++I)
    Results.push_back(Res.getValue(I));
}

SDValue MipsTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) const
{
  switch (Op.getOpcode())
  {
    case ISD::BRCOND:             return LowerBRCOND(Op, DAG);
    case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
    case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
    case ISD::BlockAddress:       return LowerBlockAddress(Op, DAG);
    case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
    case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
    case ISD::SELECT:             return LowerSELECT(Op, DAG);
    case ISD::SELECT_CC:          return LowerSELECT_CC(Op, DAG);
    case ISD::SETCC:              return LowerSETCC(Op, DAG);
    case ISD::VASTART:            return LowerVASTART(Op, DAG);
    case ISD::FCOPYSIGN:          return LowerFCOPYSIGN(Op, DAG);
    case ISD::FABS:               return LowerFABS(Op, DAG);
    case ISD::FRAMEADDR:          return LowerFRAMEADDR(Op, DAG);
    case ISD::RETURNADDR:         return LowerRETURNADDR(Op, DAG);
    case ISD::MEMBARRIER:         return LowerMEMBARRIER(Op, DAG);
    case ISD::ATOMIC_FENCE:       return LowerATOMIC_FENCE(Op, DAG);
    case ISD::SHL_PARTS:          return LowerShiftLeftParts(Op, DAG);
    case ISD::SRA_PARTS:          return LowerShiftRightParts(Op, DAG, true);
    case ISD::SRL_PARTS:          return LowerShiftRightParts(Op, DAG, false);
    case ISD::LOAD:               return LowerLOAD(Op, DAG);
    case ISD::STORE:              return LowerSTORE(Op, DAG);
    case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
    case ISD::INTRINSIC_W_CHAIN:  return LowerINTRINSIC_W_CHAIN(Op, DAG);
    case ISD::ADD:                return LowerADD(Op, DAG);
  }
  return SDValue();
}

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//

// AddLiveIn - This helper function adds the specified physical register to the
// MachineFunction as a live in value.  It also creates a corresponding
// virtual register for it.
static unsigned
AddLiveIn(MachineFunction &MF, unsigned PReg, const TargetRegisterClass *RC)
{
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getRegInfo().createVirtualRegister(RC);
  MF.getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}

// Get fp branch code (not opcode) from condition code.
static Mips::FPBranchCode GetFPBranchCodeFromCond(Mips::CondCode CC) {
  if (CC >= Mips::FCOND_F && CC <= Mips::FCOND_NGT)
    return Mips::BRANCH_T;

  assert((CC >= Mips::FCOND_T && CC <= Mips::FCOND_GT) &&
         "Invalid CondCode.");

  return Mips::BRANCH_F;
}

/*
static MachineBasicBlock* ExpandCondMov(MachineInstr *MI, MachineBasicBlock *BB,
                                        DebugLoc dl,
                                        const MipsSubtarget *Subtarget,
                                        const TargetInstrInfo *TII,
                                        bool isFPCmp, unsigned Opc) {
  // There is no need to expand CMov instructions if target has
  // conditional moves.
  if (Subtarget->hasCondMov())
    return BB;

  // To "insert" a SELECT_CC instruction, we actually have to insert the
  // diamond control-flow pattern.  The incoming instruction knows the
  // destination vreg to set, the condition code register to branch on, the
  // true/false values to select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = BB;
  ++It;

  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   setcc r1, r2, r3
  //   bNE   r1, r0, copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB  = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB  = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(It, copy0MBB);
  F->insert(It, sinkMBB);

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  // Emit the right instruction according to the type of the operands compared
  if (isFPCmp)
    BuildMI(BB, dl, TII->get(Opc)).addMBB(sinkMBB);
  else
    BuildMI(BB, dl, TII->get(Opc)).addReg(MI->getOperand(2).getReg())
      .addReg(Mips::ZERO).addMBB(sinkMBB);

  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  BB = copy0MBB;

  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %TrueValue, thisMBB ], [ %FalseValue, copy0MBB ]
  //  ...
  BB = sinkMBB;

  if (isFPCmp)
    BuildMI(*BB, BB->begin(), dl,
            TII->get(Mips::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB)
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB);
  else
    BuildMI(*BB, BB->begin(), dl,
            TII->get(Mips::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(3).getReg()).addMBB(thisMBB)
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB);

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return BB;
}
*/

MachineBasicBlock *
MipsTargetLowering::EmitBPOSGE32(MachineInstr *MI, MachineBasicBlock *BB) const{
  // $bb:
  //  bposge32_pseudo $vr0
  //  =>
  // $bb:
  //  bposge32 $tbb
  // $fbb:
  //  li $vr2, 0
  //  b $sink
  // $tbb:
  //  li $vr1, 1
  // $sink:
  //  $vr0 = phi($vr2, $fbb, $vr1, $tbb)

  MachineRegisterInfo &RegInfo = BB->getParent()->getRegInfo();
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  const TargetRegisterClass *RC = &Mips::CPURegsRegClass;
  DebugLoc DL = MI->getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = llvm::next(MachineFunction::iterator(BB));
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *FBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *TBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *Sink  = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(It, FBB);
  F->insert(It, TBB);
  F->insert(It, Sink);

  // Transfer the remainder of BB and its successor edges to Sink.
  Sink->splice(Sink->begin(), BB, llvm::next(MachineBasicBlock::iterator(MI)),
               BB->end());
  Sink->transferSuccessorsAndUpdatePHIs(BB);

  // Add successors.
  BB->addSuccessor(FBB);
  BB->addSuccessor(TBB);
  FBB->addSuccessor(Sink);
  TBB->addSuccessor(Sink);

  // Insert the real bposge32 instruction to $BB.
  BuildMI(BB, DL, TII->get(Mips::BPOSGE32)).addMBB(TBB);

  // Fill $FBB.
  unsigned VR2 = RegInfo.createVirtualRegister(RC);
  BuildMI(*FBB, FBB->end(), DL, TII->get(Mips::ADDiu), VR2)
    .addReg(Mips::ZERO).addImm(0);
  BuildMI(*FBB, FBB->end(), DL, TII->get(Mips::B)).addMBB(Sink);

  // Fill $TBB.
  unsigned VR1 = RegInfo.createVirtualRegister(RC);
  BuildMI(*TBB, TBB->end(), DL, TII->get(Mips::ADDiu), VR1)
    .addReg(Mips::ZERO).addImm(1);

  // Insert phi function to $Sink.
  BuildMI(*Sink, Sink->begin(), DL, TII->get(Mips::PHI),
          MI->getOperand(0).getReg())
    .addReg(VR2).addMBB(FBB).addReg(VR1).addMBB(TBB);

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return Sink;
}

MachineBasicBlock *
MipsTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                MachineBasicBlock *BB) const {
  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unexpected instr type to insert");
  case Mips::ATOMIC_LOAD_ADD_I8:
  case Mips::ATOMIC_LOAD_ADD_I8_P8:
    return EmitAtomicBinaryPartword(MI, BB, 1, Mips::ADDu);
  case Mips::ATOMIC_LOAD_ADD_I16:
  case Mips::ATOMIC_LOAD_ADD_I16_P8:
    return EmitAtomicBinaryPartword(MI, BB, 2, Mips::ADDu);
  case Mips::ATOMIC_LOAD_ADD_I32:
  case Mips::ATOMIC_LOAD_ADD_I32_P8:
    return EmitAtomicBinary(MI, BB, 4, Mips::ADDu);
  case Mips::ATOMIC_LOAD_ADD_I64:
  case Mips::ATOMIC_LOAD_ADD_I64_P8:
    return EmitAtomicBinary(MI, BB, 8, Mips::DADDu);

  case Mips::ATOMIC_LOAD_AND_I8:
  case Mips::ATOMIC_LOAD_AND_I8_P8:
    return EmitAtomicBinaryPartword(MI, BB, 1, Mips::AND);
  case Mips::ATOMIC_LOAD_AND_I16:
  case Mips::ATOMIC_LOAD_AND_I16_P8:
    return EmitAtomicBinaryPartword(MI, BB, 2, Mips::AND);
  case Mips::ATOMIC_LOAD_AND_I32:
  case Mips::ATOMIC_LOAD_AND_I32_P8:
    return EmitAtomicBinary(MI, BB, 4, Mips::AND);
  case Mips::ATOMIC_LOAD_AND_I64:
  case Mips::ATOMIC_LOAD_AND_I64_P8:
    return EmitAtomicBinary(MI, BB, 8, Mips::AND64);

  case Mips::ATOMIC_LOAD_OR_I8:
  case Mips::ATOMIC_LOAD_OR_I8_P8:
    return EmitAtomicBinaryPartword(MI, BB, 1, Mips::OR);
  case Mips::ATOMIC_LOAD_OR_I16:
  case Mips::ATOMIC_LOAD_OR_I16_P8:
    return EmitAtomicBinaryPartword(MI, BB, 2, Mips::OR);
  case Mips::ATOMIC_LOAD_OR_I32:
  case Mips::ATOMIC_LOAD_OR_I32_P8:
    return EmitAtomicBinary(MI, BB, 4, Mips::OR);
  case Mips::ATOMIC_LOAD_OR_I64:
  case Mips::ATOMIC_LOAD_OR_I64_P8:
    return EmitAtomicBinary(MI, BB, 8, Mips::OR64);

  case Mips::ATOMIC_LOAD_XOR_I8:
  case Mips::ATOMIC_LOAD_XOR_I8_P8:
    return EmitAtomicBinaryPartword(MI, BB, 1, Mips::XOR);
  case Mips::ATOMIC_LOAD_XOR_I16:
  case Mips::ATOMIC_LOAD_XOR_I16_P8:
    return EmitAtomicBinaryPartword(MI, BB, 2, Mips::XOR);
  case Mips::ATOMIC_LOAD_XOR_I32:
  case Mips::ATOMIC_LOAD_XOR_I32_P8:
    return EmitAtomicBinary(MI, BB, 4, Mips::XOR);
  case Mips::ATOMIC_LOAD_XOR_I64:
  case Mips::ATOMIC_LOAD_XOR_I64_P8:
    return EmitAtomicBinary(MI, BB, 8, Mips::XOR64);

  case Mips::ATOMIC_LOAD_NAND_I8:
  case Mips::ATOMIC_LOAD_NAND_I8_P8:
    return EmitAtomicBinaryPartword(MI, BB, 1, 0, true);
  case Mips::ATOMIC_LOAD_NAND_I16:
  case Mips::ATOMIC_LOAD_NAND_I16_P8:
    return EmitAtomicBinaryPartword(MI, BB, 2, 0, true);
  case Mips::ATOMIC_LOAD_NAND_I32:
  case Mips::ATOMIC_LOAD_NAND_I32_P8:
    return EmitAtomicBinary(MI, BB, 4, 0, true);
  case Mips::ATOMIC_LOAD_NAND_I64:
  case Mips::ATOMIC_LOAD_NAND_I64_P8:
    return EmitAtomicBinary(MI, BB, 8, 0, true);

  case Mips::ATOMIC_LOAD_SUB_I8:
  case Mips::ATOMIC_LOAD_SUB_I8_P8:
    return EmitAtomicBinaryPartword(MI, BB, 1, Mips::SUBu);
  case Mips::ATOMIC_LOAD_SUB_I16:
  case Mips::ATOMIC_LOAD_SUB_I16_P8:
    return EmitAtomicBinaryPartword(MI, BB, 2, Mips::SUBu);
  case Mips::ATOMIC_LOAD_SUB_I32:
  case Mips::ATOMIC_LOAD_SUB_I32_P8:
    return EmitAtomicBinary(MI, BB, 4, Mips::SUBu);
  case Mips::ATOMIC_LOAD_SUB_I64:
  case Mips::ATOMIC_LOAD_SUB_I64_P8:
    return EmitAtomicBinary(MI, BB, 8, Mips::DSUBu);

  case Mips::ATOMIC_SWAP_I8:
  case Mips::ATOMIC_SWAP_I8_P8:
    return EmitAtomicBinaryPartword(MI, BB, 1, 0);
  case Mips::ATOMIC_SWAP_I16:
  case Mips::ATOMIC_SWAP_I16_P8:
    return EmitAtomicBinaryPartword(MI, BB, 2, 0);
  case Mips::ATOMIC_SWAP_I32:
  case Mips::ATOMIC_SWAP_I32_P8:
    return EmitAtomicBinary(MI, BB, 4, 0);
  case Mips::ATOMIC_SWAP_I64:
  case Mips::ATOMIC_SWAP_I64_P8:
    return EmitAtomicBinary(MI, BB, 8, 0);

  case Mips::ATOMIC_CMP_SWAP_I8:
  case Mips::ATOMIC_CMP_SWAP_I8_P8:
    return EmitAtomicCmpSwapPartword(MI, BB, 1);
  case Mips::ATOMIC_CMP_SWAP_I16:
  case Mips::ATOMIC_CMP_SWAP_I16_P8:
    return EmitAtomicCmpSwapPartword(MI, BB, 2);
  case Mips::ATOMIC_CMP_SWAP_I32:
  case Mips::ATOMIC_CMP_SWAP_I32_P8:
    return EmitAtomicCmpSwap(MI, BB, 4);
  case Mips::ATOMIC_CMP_SWAP_I64:
  case Mips::ATOMIC_CMP_SWAP_I64_P8:
    return EmitAtomicCmpSwap(MI, BB, 8);
  case Mips::BPOSGE32_PSEUDO:
    return EmitBPOSGE32(MI, BB);
  }
}

// This function also handles Mips::ATOMIC_SWAP_I32 (when BinOpcode == 0), and
// Mips::ATOMIC_LOAD_NAND_I32 (when Nand == true)
MachineBasicBlock *
MipsTargetLowering::EmitAtomicBinary(MachineInstr *MI, MachineBasicBlock *BB,
                                     unsigned Size, unsigned BinOpcode,
                                     bool Nand) const {
  assert((Size == 4 || Size == 8) && "Unsupported size for EmitAtomicBinary.");

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::getIntegerVT(Size * 8));
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  unsigned LL, SC, AND, NOR, ZERO, BEQ;

  if (Size == 4) {
    LL = IsN64 ? Mips::LL_P8 : Mips::LL;
    SC = IsN64 ? Mips::SC_P8 : Mips::SC;
    AND = Mips::AND;
    NOR = Mips::NOR;
    ZERO = Mips::ZERO;
    BEQ = Mips::BEQ;
  }
  else {
    LL = IsN64 ? Mips::LLD_P8 : Mips::LLD;
    SC = IsN64 ? Mips::SCD_P8 : Mips::SCD;
    AND = Mips::AND64;
    NOR = Mips::NOR64;
    ZERO = Mips::ZERO_64;
    BEQ = Mips::BEQ64;
  }

  unsigned OldVal = MI->getOperand(0).getReg();
  unsigned Ptr = MI->getOperand(1).getReg();
  unsigned Incr = MI->getOperand(2).getReg();

  unsigned StoreVal = RegInfo.createVirtualRegister(RC);
  unsigned AndRes = RegInfo.createVirtualRegister(RC);
  unsigned Success = RegInfo.createVirtualRegister(RC);

  // insert new blocks after the current block
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineBasicBlock *loopMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = BB;
  ++It;
  MF->insert(It, loopMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  //  thisMBB:
  //    ...
  //    fallthrough --> loopMBB
  BB->addSuccessor(loopMBB);
  loopMBB->addSuccessor(loopMBB);
  loopMBB->addSuccessor(exitMBB);

  //  loopMBB:
  //    ll oldval, 0(ptr)
  //    <binop> storeval, oldval, incr
  //    sc success, storeval, 0(ptr)
  //    beq success, $0, loopMBB
  BB = loopMBB;
  BuildMI(BB, dl, TII->get(LL), OldVal).addReg(Ptr).addImm(0);
  if (Nand) {
    //  and andres, oldval, incr
    //  nor storeval, $0, andres
    BuildMI(BB, dl, TII->get(AND), AndRes).addReg(OldVal).addReg(Incr);
    BuildMI(BB, dl, TII->get(NOR), StoreVal).addReg(ZERO).addReg(AndRes);
  } else if (BinOpcode) {
    //  <binop> storeval, oldval, incr
    BuildMI(BB, dl, TII->get(BinOpcode), StoreVal).addReg(OldVal).addReg(Incr);
  } else {
    StoreVal = Incr;
  }
  BuildMI(BB, dl, TII->get(SC), Success).addReg(StoreVal).addReg(Ptr).addImm(0);
  BuildMI(BB, dl, TII->get(BEQ)).addReg(Success).addReg(ZERO).addMBB(loopMBB);

  MI->eraseFromParent();   // The instruction is gone now.

  return exitMBB;
}

MachineBasicBlock *
MipsTargetLowering::EmitAtomicBinaryPartword(MachineInstr *MI,
                                             MachineBasicBlock *BB,
                                             unsigned Size, unsigned BinOpcode,
                                             bool Nand) const {
  assert((Size == 1 || Size == 2) &&
      "Unsupported size for EmitAtomicBinaryPartial.");

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  unsigned LL = IsN64 ? Mips::LL_P8 : Mips::LL;
  unsigned SC = IsN64 ? Mips::SC_P8 : Mips::SC;

  unsigned Dest = MI->getOperand(0).getReg();
  unsigned Ptr = MI->getOperand(1).getReg();
  unsigned Incr = MI->getOperand(2).getReg();

  unsigned AlignedAddr = RegInfo.createVirtualRegister(RC);
  unsigned ShiftAmt = RegInfo.createVirtualRegister(RC);
  unsigned Mask = RegInfo.createVirtualRegister(RC);
  unsigned Mask2 = RegInfo.createVirtualRegister(RC);
  unsigned NewVal = RegInfo.createVirtualRegister(RC);
  unsigned OldVal = RegInfo.createVirtualRegister(RC);
  unsigned Incr2 = RegInfo.createVirtualRegister(RC);
  unsigned MaskLSB2 = RegInfo.createVirtualRegister(RC);
  unsigned PtrLSB2 = RegInfo.createVirtualRegister(RC);
  unsigned MaskUpper = RegInfo.createVirtualRegister(RC);
  unsigned AndRes = RegInfo.createVirtualRegister(RC);
  unsigned BinOpRes = RegInfo.createVirtualRegister(RC);
  unsigned MaskedOldVal0 = RegInfo.createVirtualRegister(RC);
  unsigned StoreVal = RegInfo.createVirtualRegister(RC);
  unsigned MaskedOldVal1 = RegInfo.createVirtualRegister(RC);
  unsigned SrlRes = RegInfo.createVirtualRegister(RC);
  unsigned SllRes = RegInfo.createVirtualRegister(RC);
  unsigned Success = RegInfo.createVirtualRegister(RC);

  // insert new blocks after the current block
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineBasicBlock *loopMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = BB;
  ++It;
  MF->insert(It, loopMBB);
  MF->insert(It, sinkMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)), BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(loopMBB);
  loopMBB->addSuccessor(loopMBB);
  loopMBB->addSuccessor(sinkMBB);
  sinkMBB->addSuccessor(exitMBB);

  //  thisMBB:
  //    addiu   masklsb2,$0,-4                # 0xfffffffc
  //    and     alignedaddr,ptr,masklsb2
  //    andi    ptrlsb2,ptr,3
  //    sll     shiftamt,ptrlsb2,3
  //    ori     maskupper,$0,255               # 0xff
  //    sll     mask,maskupper,shiftamt
  //    nor     mask2,$0,mask
  //    sll     incr2,incr,shiftamt

  int64_t MaskImm = (Size == 1) ? 255 : 65535;
  BuildMI(BB, dl, TII->get(Mips::ADDiu), MaskLSB2)
    .addReg(Mips::ZERO).addImm(-4);
  BuildMI(BB, dl, TII->get(Mips::AND), AlignedAddr)
    .addReg(Ptr).addReg(MaskLSB2);
  BuildMI(BB, dl, TII->get(Mips::ANDi), PtrLSB2).addReg(Ptr).addImm(3);
  BuildMI(BB, dl, TII->get(Mips::SLL), ShiftAmt).addReg(PtrLSB2).addImm(3);
  BuildMI(BB, dl, TII->get(Mips::ORi), MaskUpper)
    .addReg(Mips::ZERO).addImm(MaskImm);
  BuildMI(BB, dl, TII->get(Mips::SLLV), Mask)
    .addReg(ShiftAmt).addReg(MaskUpper);
  BuildMI(BB, dl, TII->get(Mips::NOR), Mask2).addReg(Mips::ZERO).addReg(Mask);
  BuildMI(BB, dl, TII->get(Mips::SLLV), Incr2).addReg(ShiftAmt).addReg(Incr);

  // atomic.load.binop
  // loopMBB:
  //   ll      oldval,0(alignedaddr)
  //   binop   binopres,oldval,incr2
  //   and     newval,binopres,mask
  //   and     maskedoldval0,oldval,mask2
  //   or      storeval,maskedoldval0,newval
  //   sc      success,storeval,0(alignedaddr)
  //   beq     success,$0,loopMBB

  // atomic.swap
  // loopMBB:
  //   ll      oldval,0(alignedaddr)
  //   and     newval,incr2,mask
  //   and     maskedoldval0,oldval,mask2
  //   or      storeval,maskedoldval0,newval
  //   sc      success,storeval,0(alignedaddr)
  //   beq     success,$0,loopMBB

  BB = loopMBB;
  BuildMI(BB, dl, TII->get(LL), OldVal).addReg(AlignedAddr).addImm(0);
  if (Nand) {
    //  and andres, oldval, incr2
    //  nor binopres, $0, andres
    //  and newval, binopres, mask
    BuildMI(BB, dl, TII->get(Mips::AND), AndRes).addReg(OldVal).addReg(Incr2);
    BuildMI(BB, dl, TII->get(Mips::NOR), BinOpRes)
      .addReg(Mips::ZERO).addReg(AndRes);
    BuildMI(BB, dl, TII->get(Mips::AND), NewVal).addReg(BinOpRes).addReg(Mask);
  } else if (BinOpcode) {
    //  <binop> binopres, oldval, incr2
    //  and newval, binopres, mask
    BuildMI(BB, dl, TII->get(BinOpcode), BinOpRes).addReg(OldVal).addReg(Incr2);
    BuildMI(BB, dl, TII->get(Mips::AND), NewVal).addReg(BinOpRes).addReg(Mask);
  } else {// atomic.swap
    //  and newval, incr2, mask
    BuildMI(BB, dl, TII->get(Mips::AND), NewVal).addReg(Incr2).addReg(Mask);
  }

  BuildMI(BB, dl, TII->get(Mips::AND), MaskedOldVal0)
    .addReg(OldVal).addReg(Mask2);
  BuildMI(BB, dl, TII->get(Mips::OR), StoreVal)
    .addReg(MaskedOldVal0).addReg(NewVal);
  BuildMI(BB, dl, TII->get(SC), Success)
    .addReg(StoreVal).addReg(AlignedAddr).addImm(0);
  BuildMI(BB, dl, TII->get(Mips::BEQ))
    .addReg(Success).addReg(Mips::ZERO).addMBB(loopMBB);

  //  sinkMBB:
  //    and     maskedoldval1,oldval,mask
  //    srl     srlres,maskedoldval1,shiftamt
  //    sll     sllres,srlres,24
  //    sra     dest,sllres,24
  BB = sinkMBB;
  int64_t ShiftImm = (Size == 1) ? 24 : 16;

  BuildMI(BB, dl, TII->get(Mips::AND), MaskedOldVal1)
    .addReg(OldVal).addReg(Mask);
  BuildMI(BB, dl, TII->get(Mips::SRLV), SrlRes)
      .addReg(ShiftAmt).addReg(MaskedOldVal1);
  BuildMI(BB, dl, TII->get(Mips::SLL), SllRes)
      .addReg(SrlRes).addImm(ShiftImm);
  BuildMI(BB, dl, TII->get(Mips::SRA), Dest)
      .addReg(SllRes).addImm(ShiftImm);

  MI->eraseFromParent();   // The instruction is gone now.

  return exitMBB;
}

MachineBasicBlock *
MipsTargetLowering::EmitAtomicCmpSwap(MachineInstr *MI,
                                      MachineBasicBlock *BB,
                                      unsigned Size) const {
  assert((Size == 4 || Size == 8) && "Unsupported size for EmitAtomicCmpSwap.");

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::getIntegerVT(Size * 8));
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  unsigned LL, SC, ZERO, BNE, BEQ;

  if (Size == 4) {
    LL = IsN64 ? Mips::LL_P8 : Mips::LL;
    SC = IsN64 ? Mips::SC_P8 : Mips::SC;
    ZERO = Mips::ZERO;
    BNE = Mips::BNE;
    BEQ = Mips::BEQ;
  }
  else {
    LL = IsN64 ? Mips::LLD_P8 : Mips::LLD;
    SC = IsN64 ? Mips::SCD_P8 : Mips::SCD;
    ZERO = Mips::ZERO_64;
    BNE = Mips::BNE64;
    BEQ = Mips::BEQ64;
  }

  unsigned Dest    = MI->getOperand(0).getReg();
  unsigned Ptr     = MI->getOperand(1).getReg();
  unsigned OldVal  = MI->getOperand(2).getReg();
  unsigned NewVal  = MI->getOperand(3).getReg();

  unsigned Success = RegInfo.createVirtualRegister(RC);

  // insert new blocks after the current block
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineBasicBlock *loop1MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *loop2MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = BB;
  ++It;
  MF->insert(It, loop1MBB);
  MF->insert(It, loop2MBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)), BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  //  thisMBB:
  //    ...
  //    fallthrough --> loop1MBB
  BB->addSuccessor(loop1MBB);
  loop1MBB->addSuccessor(exitMBB);
  loop1MBB->addSuccessor(loop2MBB);
  loop2MBB->addSuccessor(loop1MBB);
  loop2MBB->addSuccessor(exitMBB);

  // loop1MBB:
  //   ll dest, 0(ptr)
  //   bne dest, oldval, exitMBB
  BB = loop1MBB;
  BuildMI(BB, dl, TII->get(LL), Dest).addReg(Ptr).addImm(0);
  BuildMI(BB, dl, TII->get(BNE))
    .addReg(Dest).addReg(OldVal).addMBB(exitMBB);

  // loop2MBB:
  //   sc success, newval, 0(ptr)
  //   beq success, $0, loop1MBB
  BB = loop2MBB;
  BuildMI(BB, dl, TII->get(SC), Success)
    .addReg(NewVal).addReg(Ptr).addImm(0);
  BuildMI(BB, dl, TII->get(BEQ))
    .addReg(Success).addReg(ZERO).addMBB(loop1MBB);

  MI->eraseFromParent();   // The instruction is gone now.

  return exitMBB;
}

MachineBasicBlock *
MipsTargetLowering::EmitAtomicCmpSwapPartword(MachineInstr *MI,
                                              MachineBasicBlock *BB,
                                              unsigned Size) const {
  assert((Size == 1 || Size == 2) &&
      "Unsupported size for EmitAtomicCmpSwapPartial.");

  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  unsigned LL = IsN64 ? Mips::LL_P8 : Mips::LL;
  unsigned SC = IsN64 ? Mips::SC_P8 : Mips::SC;

  unsigned Dest    = MI->getOperand(0).getReg();
  unsigned Ptr     = MI->getOperand(1).getReg();
  unsigned CmpVal  = MI->getOperand(2).getReg();
  unsigned NewVal  = MI->getOperand(3).getReg();

  unsigned AlignedAddr = RegInfo.createVirtualRegister(RC);
  unsigned ShiftAmt = RegInfo.createVirtualRegister(RC);
  unsigned Mask = RegInfo.createVirtualRegister(RC);
  unsigned Mask2 = RegInfo.createVirtualRegister(RC);
  unsigned ShiftedCmpVal = RegInfo.createVirtualRegister(RC);
  unsigned OldVal = RegInfo.createVirtualRegister(RC);
  unsigned MaskedOldVal0 = RegInfo.createVirtualRegister(RC);
  unsigned ShiftedNewVal = RegInfo.createVirtualRegister(RC);
  unsigned MaskLSB2 = RegInfo.createVirtualRegister(RC);
  unsigned PtrLSB2 = RegInfo.createVirtualRegister(RC);
  unsigned MaskUpper = RegInfo.createVirtualRegister(RC);
  unsigned MaskedCmpVal = RegInfo.createVirtualRegister(RC);
  unsigned MaskedNewVal = RegInfo.createVirtualRegister(RC);
  unsigned MaskedOldVal1 = RegInfo.createVirtualRegister(RC);
  unsigned StoreVal = RegInfo.createVirtualRegister(RC);
  unsigned SrlRes = RegInfo.createVirtualRegister(RC);
  unsigned SllRes = RegInfo.createVirtualRegister(RC);
  unsigned Success = RegInfo.createVirtualRegister(RC);

  // insert new blocks after the current block
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineBasicBlock *loop1MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *loop2MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineFunction::iterator It = BB;
  ++It;
  MF->insert(It, loop1MBB);
  MF->insert(It, loop2MBB);
  MF->insert(It, sinkMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)), BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(loop1MBB);
  loop1MBB->addSuccessor(sinkMBB);
  loop1MBB->addSuccessor(loop2MBB);
  loop2MBB->addSuccessor(loop1MBB);
  loop2MBB->addSuccessor(sinkMBB);
  sinkMBB->addSuccessor(exitMBB);

  // FIXME: computation of newval2 can be moved to loop2MBB.
  //  thisMBB:
  //    addiu   masklsb2,$0,-4                # 0xfffffffc
  //    and     alignedaddr,ptr,masklsb2
  //    andi    ptrlsb2,ptr,3
  //    sll     shiftamt,ptrlsb2,3
  //    ori     maskupper,$0,255               # 0xff
  //    sll     mask,maskupper,shiftamt
  //    nor     mask2,$0,mask
  //    andi    maskedcmpval,cmpval,255
  //    sll     shiftedcmpval,maskedcmpval,shiftamt
  //    andi    maskednewval,newval,255
  //    sll     shiftednewval,maskednewval,shiftamt
  int64_t MaskImm = (Size == 1) ? 255 : 65535;
  BuildMI(BB, dl, TII->get(Mips::ADDiu), MaskLSB2)
    .addReg(Mips::ZERO).addImm(-4);
  BuildMI(BB, dl, TII->get(Mips::AND), AlignedAddr)
    .addReg(Ptr).addReg(MaskLSB2);
  BuildMI(BB, dl, TII->get(Mips::ANDi), PtrLSB2).addReg(Ptr).addImm(3);
  BuildMI(BB, dl, TII->get(Mips::SLL), ShiftAmt).addReg(PtrLSB2).addImm(3);
  BuildMI(BB, dl, TII->get(Mips::ORi), MaskUpper)
    .addReg(Mips::ZERO).addImm(MaskImm);
  BuildMI(BB, dl, TII->get(Mips::SLLV), Mask)
    .addReg(ShiftAmt).addReg(MaskUpper);
  BuildMI(BB, dl, TII->get(Mips::NOR), Mask2).addReg(Mips::ZERO).addReg(Mask);
  BuildMI(BB, dl, TII->get(Mips::ANDi), MaskedCmpVal)
    .addReg(CmpVal).addImm(MaskImm);
  BuildMI(BB, dl, TII->get(Mips::SLLV), ShiftedCmpVal)
    .addReg(ShiftAmt).addReg(MaskedCmpVal);
  BuildMI(BB, dl, TII->get(Mips::ANDi), MaskedNewVal)
    .addReg(NewVal).addImm(MaskImm);
  BuildMI(BB, dl, TII->get(Mips::SLLV), ShiftedNewVal)
    .addReg(ShiftAmt).addReg(MaskedNewVal);

  //  loop1MBB:
  //    ll      oldval,0(alginedaddr)
  //    and     maskedoldval0,oldval,mask
  //    bne     maskedoldval0,shiftedcmpval,sinkMBB
  BB = loop1MBB;
  BuildMI(BB, dl, TII->get(LL), OldVal).addReg(AlignedAddr).addImm(0);
  BuildMI(BB, dl, TII->get(Mips::AND), MaskedOldVal0)
    .addReg(OldVal).addReg(Mask);
  BuildMI(BB, dl, TII->get(Mips::BNE))
    .addReg(MaskedOldVal0).addReg(ShiftedCmpVal).addMBB(sinkMBB);

  //  loop2MBB:
  //    and     maskedoldval1,oldval,mask2
  //    or      storeval,maskedoldval1,shiftednewval
  //    sc      success,storeval,0(alignedaddr)
  //    beq     success,$0,loop1MBB
  BB = loop2MBB;
  BuildMI(BB, dl, TII->get(Mips::AND), MaskedOldVal1)
    .addReg(OldVal).addReg(Mask2);
  BuildMI(BB, dl, TII->get(Mips::OR), StoreVal)
    .addReg(MaskedOldVal1).addReg(ShiftedNewVal);
  BuildMI(BB, dl, TII->get(SC), Success)
      .addReg(StoreVal).addReg(AlignedAddr).addImm(0);
  BuildMI(BB, dl, TII->get(Mips::BEQ))
      .addReg(Success).addReg(Mips::ZERO).addMBB(loop1MBB);

  //  sinkMBB:
  //    srl     srlres,maskedoldval0,shiftamt
  //    sll     sllres,srlres,24
  //    sra     dest,sllres,24
  BB = sinkMBB;
  int64_t ShiftImm = (Size == 1) ? 24 : 16;

  BuildMI(BB, dl, TII->get(Mips::SRLV), SrlRes)
      .addReg(ShiftAmt).addReg(MaskedOldVal0);
  BuildMI(BB, dl, TII->get(Mips::SLL), SllRes)
      .addReg(SrlRes).addImm(ShiftImm);
  BuildMI(BB, dl, TII->get(Mips::SRA), Dest)
      .addReg(SllRes).addImm(ShiftImm);

  MI->eraseFromParent();   // The instruction is gone now.

  return exitMBB;
}

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//
SDValue MipsTargetLowering::
LowerBRCOND(SDValue Op, SelectionDAG &DAG) const
{
  // The first operand is the chain, the second is the condition, the third is
  // the block to branch to if the condition is true.
  SDValue Chain = Op.getOperand(0);
  SDValue Dest = Op.getOperand(2);
  DebugLoc dl = Op.getDebugLoc();

  SDValue CondRes = CreateFPCmp(DAG, Op.getOperand(1));

  // Return if flag is not set by a floating point comparison.
  if (CondRes.getOpcode() != MipsISD::FPCmp)
    return Op;

  SDValue CCNode  = CondRes.getOperand(2);
  Mips::CondCode CC =
    (Mips::CondCode)cast<ConstantSDNode>(CCNode)->getZExtValue();
  SDValue BrCode = DAG.getConstant(GetFPBranchCodeFromCond(CC), MVT::i32);

  return DAG.getNode(MipsISD::FPBrcond, dl, Op.getValueType(), Chain, BrCode,
                     Dest, CondRes);
}

SDValue MipsTargetLowering::
LowerSELECT(SDValue Op, SelectionDAG &DAG) const
{
  SDValue Cond = CreateFPCmp(DAG, Op.getOperand(0));

  // Return if flag is not set by a floating point comparison.
  if (Cond.getOpcode() != MipsISD::FPCmp)
    return Op;

  return CreateCMovFP(DAG, Cond, Op.getOperand(1), Op.getOperand(2),
                      Op.getDebugLoc());
}

SDValue MipsTargetLowering::
LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
  EVT Ty = Op.getOperand(0).getValueType();
  SDValue Cond = DAG.getNode(ISD::SETCC, DL, getSetCCResultType(Ty),
                             Op.getOperand(0), Op.getOperand(1),
                             Op.getOperand(4));

  return DAG.getNode(ISD::SELECT, DL, Op.getValueType(), Cond, Op.getOperand(2),
                     Op.getOperand(3));
}

SDValue MipsTargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Cond = CreateFPCmp(DAG, Op);

  assert(Cond.getOpcode() == MipsISD::FPCmp &&
         "Floating point operand expected.");

  SDValue True  = DAG.getConstant(1, MVT::i32);
  SDValue False = DAG.getConstant(0, MVT::i32);

  return CreateCMovFP(DAG, Cond, True, False, Op.getDebugLoc());
}

SDValue MipsTargetLowering::LowerGlobalAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  // FIXME there isn't actually debug info here
  DebugLoc dl = Op.getDebugLoc();
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();

  if (getTargetMachine().getRelocationModel() != Reloc::PIC_ && !IsN64) {
    SDVTList VTs = DAG.getVTList(MVT::i32);

    const MipsTargetObjectFile &TLOF =
      (const MipsTargetObjectFile&)getObjFileLowering();

    // %gp_rel relocation
    if (TLOF.IsGlobalInSmallSection(GV, getTargetMachine())) {
      SDValue GA = DAG.getTargetGlobalAddress(GV, dl, MVT::i32, 0,
                                              MipsII::MO_GPREL);
      SDValue GPRelNode = DAG.getNode(MipsISD::GPRel, dl, VTs, &GA, 1);
      SDValue GPReg = DAG.getRegister(Mips::GP, MVT::i32);
      return DAG.getNode(ISD::ADD, dl, MVT::i32, GPReg, GPRelNode);
    }
    // %hi/%lo relocation
    SDValue GAHi = DAG.getTargetGlobalAddress(GV, dl, MVT::i32, 0,
                                              MipsII::MO_ABS_HI);
    SDValue GALo = DAG.getTargetGlobalAddress(GV, dl, MVT::i32, 0,
                                              MipsII::MO_ABS_LO);
    SDValue HiPart = DAG.getNode(MipsISD::Hi, dl, VTs, &GAHi, 1);
    SDValue Lo = DAG.getNode(MipsISD::Lo, dl, MVT::i32, GALo);
    return DAG.getNode(ISD::ADD, dl, MVT::i32, HiPart, Lo);
  }

  EVT ValTy = Op.getValueType();
  bool HasGotOfst = (GV->hasInternalLinkage() ||
                     (GV->hasLocalLinkage() && !isa<Function>(GV)));
  unsigned GotFlag = HasMips64 ?
                     (HasGotOfst ? MipsII::MO_GOT_PAGE : MipsII::MO_GOT_DISP) :
                     (HasGotOfst ? MipsII::MO_GOT : MipsII::MO_GOT16);
  SDValue GA = DAG.getTargetGlobalAddress(GV, dl, ValTy, 0, GotFlag);
  GA = DAG.getNode(MipsISD::Wrapper, dl, ValTy, GetGlobalReg(DAG, ValTy), GA);
  SDValue ResNode = DAG.getLoad(ValTy, dl, DAG.getEntryNode(), GA,
                                MachinePointerInfo(), false, false, false, 0);
  // On functions and global targets not internal linked only
  // a load from got/GP is necessary for PIC to work.
  if (!HasGotOfst)
    return ResNode;
  SDValue GALo = DAG.getTargetGlobalAddress(GV, dl, ValTy, 0,
                                            HasMips64 ? MipsII::MO_GOT_OFST :
                                                        MipsII::MO_ABS_LO);
  SDValue Lo = DAG.getNode(MipsISD::Lo, dl, ValTy, GALo);
  return DAG.getNode(ISD::ADD, dl, ValTy, ResNode, Lo);
}

SDValue MipsTargetLowering::LowerBlockAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();
  // FIXME there isn't actually debug info here
  DebugLoc dl = Op.getDebugLoc();

  if (getTargetMachine().getRelocationModel() != Reloc::PIC_ && !IsN64) {
    // %hi/%lo relocation
    SDValue BAHi =
      DAG.getTargetBlockAddress(BA, MVT::i32, 0, MipsII::MO_ABS_HI);
    SDValue BALo =
      DAG.getTargetBlockAddress(BA, MVT::i32, 0, MipsII::MO_ABS_LO);
    SDValue Hi = DAG.getNode(MipsISD::Hi, dl, MVT::i32, BAHi);
    SDValue Lo = DAG.getNode(MipsISD::Lo, dl, MVT::i32, BALo);
    return DAG.getNode(ISD::ADD, dl, MVT::i32, Hi, Lo);
  }

  EVT ValTy = Op.getValueType();
  unsigned GOTFlag = HasMips64 ? MipsII::MO_GOT_PAGE : MipsII::MO_GOT;
  unsigned OFSTFlag = HasMips64 ? MipsII::MO_GOT_OFST : MipsII::MO_ABS_LO;
  SDValue BAGOTOffset = DAG.getTargetBlockAddress(BA, ValTy, 0, GOTFlag);
  BAGOTOffset = DAG.getNode(MipsISD::Wrapper, dl, ValTy,
                            GetGlobalReg(DAG, ValTy), BAGOTOffset);
  SDValue BALOOffset = DAG.getTargetBlockAddress(BA, ValTy, 0, OFSTFlag);
  SDValue Load = DAG.getLoad(ValTy, dl, DAG.getEntryNode(), BAGOTOffset,
                             MachinePointerInfo(), false, false, false, 0);
  SDValue Lo = DAG.getNode(MipsISD::Lo, dl, ValTy, BALOOffset);
  return DAG.getNode(ISD::ADD, dl, ValTy, Load, Lo);
}

SDValue MipsTargetLowering::
LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const
{
  // If the relocation model is PIC, use the General Dynamic TLS Model or
  // Local Dynamic TLS model, otherwise use the Initial Exec or
  // Local Exec TLS Model.

  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);
  DebugLoc dl = GA->getDebugLoc();
  const GlobalValue *GV = GA->getGlobal();
  EVT PtrVT = getPointerTy();

  TLSModel::Model model = getTargetMachine().getTLSModel(GV);

  if (model == TLSModel::GeneralDynamic || model == TLSModel::LocalDynamic) {
    // General Dynamic and Local Dynamic TLS Model.
    unsigned Flag = (model == TLSModel::LocalDynamic) ? MipsII::MO_TLSLDM
                                                      : MipsII::MO_TLSGD;

    SDValue TGA = DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0, Flag);
    SDValue Argument = DAG.getNode(MipsISD::Wrapper, dl, PtrVT,
                                   GetGlobalReg(DAG, PtrVT), TGA);
    unsigned PtrSize = PtrVT.getSizeInBits();
    IntegerType *PtrTy = Type::getIntNTy(*DAG.getContext(), PtrSize);

    SDValue TlsGetAddr = DAG.getExternalSymbol("__tls_get_addr", PtrVT);

    ArgListTy Args;
    ArgListEntry Entry;
    Entry.Node = Argument;
    Entry.Ty = PtrTy;
    Args.push_back(Entry);

    TargetLowering::CallLoweringInfo CLI(DAG.getEntryNode(), PtrTy,
                  false, false, false, false, 0, CallingConv::C,
                  /*isTailCall=*/false, /*doesNotRet=*/false,
                  /*isReturnValueUsed=*/true,
                  TlsGetAddr, Args, DAG, dl);
    std::pair<SDValue, SDValue> CallResult = LowerCallTo(CLI);

    SDValue Ret = CallResult.first;

    if (model != TLSModel::LocalDynamic)
      return Ret;

    SDValue TGAHi = DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                               MipsII::MO_DTPREL_HI);
    SDValue Hi = DAG.getNode(MipsISD::Hi, dl, PtrVT, TGAHi);
    SDValue TGALo = DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                               MipsII::MO_DTPREL_LO);
    SDValue Lo = DAG.getNode(MipsISD::Lo, dl, PtrVT, TGALo);
    SDValue Add = DAG.getNode(ISD::ADD, dl, PtrVT, Hi, Ret);
    return DAG.getNode(ISD::ADD, dl, PtrVT, Add, Lo);
  }

  SDValue Offset;
  if (model == TLSModel::InitialExec) {
    // Initial Exec TLS Model
    SDValue TGA = DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                             MipsII::MO_GOTTPREL);
    TGA = DAG.getNode(MipsISD::Wrapper, dl, PtrVT, GetGlobalReg(DAG, PtrVT),
                      TGA);
    Offset = DAG.getLoad(PtrVT, dl,
                         DAG.getEntryNode(), TGA, MachinePointerInfo(),
                         false, false, false, 0);
  } else {
    // Local Exec TLS Model
    assert(model == TLSModel::LocalExec);
    SDValue TGAHi = DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                               MipsII::MO_TPREL_HI);
    SDValue TGALo = DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                               MipsII::MO_TPREL_LO);
    SDValue Hi = DAG.getNode(MipsISD::Hi, dl, PtrVT, TGAHi);
    SDValue Lo = DAG.getNode(MipsISD::Lo, dl, PtrVT, TGALo);
    Offset = DAG.getNode(ISD::ADD, dl, PtrVT, Hi, Lo);
  }

  SDValue ThreadPointer = DAG.getNode(MipsISD::ThreadPointer, dl, PtrVT);
  return DAG.getNode(ISD::ADD, dl, PtrVT, ThreadPointer, Offset);
}

SDValue MipsTargetLowering::
LowerJumpTable(SDValue Op, SelectionDAG &DAG) const
{
  SDValue HiPart, JTI, JTILo;
  // FIXME there isn't actually debug info here
  DebugLoc dl = Op.getDebugLoc();
  bool IsPIC = getTargetMachine().getRelocationModel() == Reloc::PIC_;
  EVT PtrVT = Op.getValueType();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);

  if (!IsPIC && !IsN64) {
    JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT, MipsII::MO_ABS_HI);
    HiPart = DAG.getNode(MipsISD::Hi, dl, PtrVT, JTI);
    JTILo = DAG.getTargetJumpTable(JT->getIndex(), PtrVT, MipsII::MO_ABS_LO);
  } else {// Emit Load from Global Pointer
    unsigned GOTFlag = HasMips64 ? MipsII::MO_GOT_PAGE : MipsII::MO_GOT;
    unsigned OfstFlag = HasMips64 ? MipsII::MO_GOT_OFST : MipsII::MO_ABS_LO;
    JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT, GOTFlag);
    JTI = DAG.getNode(MipsISD::Wrapper, dl, PtrVT, GetGlobalReg(DAG, PtrVT),
                      JTI);
    HiPart = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), JTI,
                         MachinePointerInfo(), false, false, false, 0);
    JTILo = DAG.getTargetJumpTable(JT->getIndex(), PtrVT, OfstFlag);
  }

  SDValue Lo = DAG.getNode(MipsISD::Lo, dl, PtrVT, JTILo);
  return DAG.getNode(ISD::ADD, dl, PtrVT, HiPart, Lo);
}

SDValue MipsTargetLowering::
LowerConstantPool(SDValue Op, SelectionDAG &DAG) const
{
  SDValue ResNode;
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);
  const Constant *C = N->getConstVal();
  // FIXME there isn't actually debug info here
  DebugLoc dl = Op.getDebugLoc();

  // gp_rel relocation
  // FIXME: we should reference the constant pool using small data sections,
  // but the asm printer currently doesn't support this feature without
  // hacking it. This feature should come soon so we can uncomment the
  // stuff below.
  //if (IsInSmallSection(C->getType())) {
  //  SDValue GPRelNode = DAG.getNode(MipsISD::GPRel, MVT::i32, CP);
  //  SDValue GOT = DAG.getGLOBAL_OFFSET_TABLE(MVT::i32);
  //  ResNode = DAG.getNode(ISD::ADD, MVT::i32, GOT, GPRelNode);

  if (getTargetMachine().getRelocationModel() != Reloc::PIC_ && !IsN64) {
    SDValue CPHi = DAG.getTargetConstantPool(C, MVT::i32, N->getAlignment(),
                                             N->getOffset(), MipsII::MO_ABS_HI);
    SDValue CPLo = DAG.getTargetConstantPool(C, MVT::i32, N->getAlignment(),
                                             N->getOffset(), MipsII::MO_ABS_LO);
    SDValue HiPart = DAG.getNode(MipsISD::Hi, dl, MVT::i32, CPHi);
    SDValue Lo = DAG.getNode(MipsISD::Lo, dl, MVT::i32, CPLo);
    ResNode = DAG.getNode(ISD::ADD, dl, MVT::i32, HiPart, Lo);
  } else {
    EVT ValTy = Op.getValueType();
    unsigned GOTFlag = HasMips64 ? MipsII::MO_GOT_PAGE : MipsII::MO_GOT;
    unsigned OFSTFlag = HasMips64 ? MipsII::MO_GOT_OFST : MipsII::MO_ABS_LO;
    SDValue CP = DAG.getTargetConstantPool(C, ValTy, N->getAlignment(),
                                           N->getOffset(), GOTFlag);
    CP = DAG.getNode(MipsISD::Wrapper, dl, ValTy, GetGlobalReg(DAG, ValTy), CP);
    SDValue Load = DAG.getLoad(ValTy, dl, DAG.getEntryNode(), CP,
                               MachinePointerInfo::getConstantPool(), false,
                               false, false, 0);
    SDValue CPLo = DAG.getTargetConstantPool(C, ValTy, N->getAlignment(),
                                             N->getOffset(), OFSTFlag);
    SDValue Lo = DAG.getNode(MipsISD::Lo, dl, ValTy, CPLo);
    ResNode = DAG.getNode(ISD::ADD, dl, ValTy, Load, Lo);
  }

  return ResNode;
}

SDValue MipsTargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MipsFunctionInfo *FuncInfo = MF.getInfo<MipsFunctionInfo>();

  DebugLoc dl = Op.getDebugLoc();
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy());

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), dl, FI, Op.getOperand(1),
                      MachinePointerInfo(SV), false, false, 0);
}

static SDValue LowerFCOPYSIGN32(SDValue Op, SelectionDAG &DAG, bool HasR2) {
  EVT TyX = Op.getOperand(0).getValueType();
  EVT TyY = Op.getOperand(1).getValueType();
  SDValue Const1 = DAG.getConstant(1, MVT::i32);
  SDValue Const31 = DAG.getConstant(31, MVT::i32);
  DebugLoc DL = Op.getDebugLoc();
  SDValue Res;

  // If operand is of type f64, extract the upper 32-bit. Otherwise, bitcast it
  // to i32.
  SDValue X = (TyX == MVT::f32) ?
    DAG.getNode(ISD::BITCAST, DL, MVT::i32, Op.getOperand(0)) :
    DAG.getNode(MipsISD::ExtractElementF64, DL, MVT::i32, Op.getOperand(0),
                Const1);
  SDValue Y = (TyY == MVT::f32) ?
    DAG.getNode(ISD::BITCAST, DL, MVT::i32, Op.getOperand(1)) :
    DAG.getNode(MipsISD::ExtractElementF64, DL, MVT::i32, Op.getOperand(1),
                Const1);

  if (HasR2) {
    // ext  E, Y, 31, 1  ; extract bit31 of Y
    // ins  X, E, 31, 1  ; insert extracted bit at bit31 of X
    SDValue E = DAG.getNode(MipsISD::Ext, DL, MVT::i32, Y, Const31, Const1);
    Res = DAG.getNode(MipsISD::Ins, DL, MVT::i32, E, Const31, Const1, X);
  } else {
    // sll SllX, X, 1
    // srl SrlX, SllX, 1
    // srl SrlY, Y, 31
    // sll SllY, SrlX, 31
    // or  Or, SrlX, SllY
    SDValue SllX = DAG.getNode(ISD::SHL, DL, MVT::i32, X, Const1);
    SDValue SrlX = DAG.getNode(ISD::SRL, DL, MVT::i32, SllX, Const1);
    SDValue SrlY = DAG.getNode(ISD::SRL, DL, MVT::i32, Y, Const31);
    SDValue SllY = DAG.getNode(ISD::SHL, DL, MVT::i32, SrlY, Const31);
    Res = DAG.getNode(ISD::OR, DL, MVT::i32, SrlX, SllY);
  }

  if (TyX == MVT::f32)
    return DAG.getNode(ISD::BITCAST, DL, Op.getOperand(0).getValueType(), Res);

  SDValue LowX = DAG.getNode(MipsISD::ExtractElementF64, DL, MVT::i32,
                             Op.getOperand(0), DAG.getConstant(0, MVT::i32));
  return DAG.getNode(MipsISD::BuildPairF64, DL, MVT::f64, LowX, Res);
}

static SDValue LowerFCOPYSIGN64(SDValue Op, SelectionDAG &DAG, bool HasR2) {
  unsigned WidthX = Op.getOperand(0).getValueSizeInBits();
  unsigned WidthY = Op.getOperand(1).getValueSizeInBits();
  EVT TyX = MVT::getIntegerVT(WidthX), TyY = MVT::getIntegerVT(WidthY);
  SDValue Const1 = DAG.getConstant(1, MVT::i32);
  DebugLoc DL = Op.getDebugLoc();

  // Bitcast to integer nodes.
  SDValue X = DAG.getNode(ISD::BITCAST, DL, TyX, Op.getOperand(0));
  SDValue Y = DAG.getNode(ISD::BITCAST, DL, TyY, Op.getOperand(1));

  if (HasR2) {
    // ext  E, Y, width(Y) - 1, 1  ; extract bit width(Y)-1 of Y
    // ins  X, E, width(X) - 1, 1  ; insert extracted bit at bit width(X)-1 of X
    SDValue E = DAG.getNode(MipsISD::Ext, DL, TyY, Y,
                            DAG.getConstant(WidthY - 1, MVT::i32), Const1);

    if (WidthX > WidthY)
      E = DAG.getNode(ISD::ZERO_EXTEND, DL, TyX, E);
    else if (WidthY > WidthX)
      E = DAG.getNode(ISD::TRUNCATE, DL, TyX, E);

    SDValue I = DAG.getNode(MipsISD::Ins, DL, TyX, E,
                            DAG.getConstant(WidthX - 1, MVT::i32), Const1, X);
    return DAG.getNode(ISD::BITCAST, DL, Op.getOperand(0).getValueType(), I);
  }

  // (d)sll SllX, X, 1
  // (d)srl SrlX, SllX, 1
  // (d)srl SrlY, Y, width(Y)-1
  // (d)sll SllY, SrlX, width(Y)-1
  // or     Or, SrlX, SllY
  SDValue SllX = DAG.getNode(ISD::SHL, DL, TyX, X, Const1);
  SDValue SrlX = DAG.getNode(ISD::SRL, DL, TyX, SllX, Const1);
  SDValue SrlY = DAG.getNode(ISD::SRL, DL, TyY, Y,
                             DAG.getConstant(WidthY - 1, MVT::i32));

  if (WidthX > WidthY)
    SrlY = DAG.getNode(ISD::ZERO_EXTEND, DL, TyX, SrlY);
  else if (WidthY > WidthX)
    SrlY = DAG.getNode(ISD::TRUNCATE, DL, TyX, SrlY);

  SDValue SllY = DAG.getNode(ISD::SHL, DL, TyX, SrlY,
                             DAG.getConstant(WidthX - 1, MVT::i32));
  SDValue Or = DAG.getNode(ISD::OR, DL, TyX, SrlX, SllY);
  return DAG.getNode(ISD::BITCAST, DL, Op.getOperand(0).getValueType(), Or);
}

SDValue
MipsTargetLowering::LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const {
  if (Subtarget->hasMips64())
    return LowerFCOPYSIGN64(Op, DAG, Subtarget->hasMips32r2());

  return LowerFCOPYSIGN32(Op, DAG, Subtarget->hasMips32r2());
}

static SDValue LowerFABS32(SDValue Op, SelectionDAG &DAG, bool HasR2) {
  SDValue Res, Const1 = DAG.getConstant(1, MVT::i32);
  DebugLoc DL = Op.getDebugLoc();

  // If operand is of type f64, extract the upper 32-bit. Otherwise, bitcast it
  // to i32.
  SDValue X = (Op.getValueType() == MVT::f32) ?
    DAG.getNode(ISD::BITCAST, DL, MVT::i32, Op.getOperand(0)) :
    DAG.getNode(MipsISD::ExtractElementF64, DL, MVT::i32, Op.getOperand(0),
                Const1);

  // Clear MSB.
  if (HasR2)
    Res = DAG.getNode(MipsISD::Ins, DL, MVT::i32,
                      DAG.getRegister(Mips::ZERO, MVT::i32),
                      DAG.getConstant(31, MVT::i32), Const1, X);
  else {
    SDValue SllX = DAG.getNode(ISD::SHL, DL, MVT::i32, X, Const1);
    Res = DAG.getNode(ISD::SRL, DL, MVT::i32, SllX, Const1);
  }

  if (Op.getValueType() == MVT::f32)
    return DAG.getNode(ISD::BITCAST, DL, MVT::f32, Res);

  SDValue LowX = DAG.getNode(MipsISD::ExtractElementF64, DL, MVT::i32,
                             Op.getOperand(0), DAG.getConstant(0, MVT::i32));
  return DAG.getNode(MipsISD::BuildPairF64, DL, MVT::f64, LowX, Res);
}

static SDValue LowerFABS64(SDValue Op, SelectionDAG &DAG, bool HasR2) {
  SDValue Res, Const1 = DAG.getConstant(1, MVT::i32);
  DebugLoc DL = Op.getDebugLoc();

  // Bitcast to integer node.
  SDValue X = DAG.getNode(ISD::BITCAST, DL, MVT::i64, Op.getOperand(0));

  // Clear MSB.
  if (HasR2)
    Res = DAG.getNode(MipsISD::Ins, DL, MVT::i64,
                      DAG.getRegister(Mips::ZERO_64, MVT::i64),
                      DAG.getConstant(63, MVT::i32), Const1, X);
  else {
    SDValue SllX = DAG.getNode(ISD::SHL, DL, MVT::i64, X, Const1);
    Res = DAG.getNode(ISD::SRL, DL, MVT::i64, SllX, Const1);
  }

  return DAG.getNode(ISD::BITCAST, DL, MVT::f64, Res);
}

SDValue
MipsTargetLowering::LowerFABS(SDValue Op, SelectionDAG &DAG) const {
  if (Subtarget->hasMips64() && (Op.getValueType() == MVT::f64))
    return LowerFABS64(Op, DAG, Subtarget->hasMips32r2());

  return LowerFABS32(Op, DAG, Subtarget->hasMips32r2());
}

SDValue MipsTargetLowering::
LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const {
  // check the depth
  assert((cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() == 0) &&
         "Frame address can only be determined for current frame.");

  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setFrameAddressIsTaken(true);
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl,
                                         IsN64 ? Mips::FP_64 : Mips::FP, VT);
  return FrameAddr;
}

SDValue MipsTargetLowering::LowerRETURNADDR(SDValue Op,
                                            SelectionDAG &DAG) const {
  // check the depth
  assert((cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() == 0) &&
         "Return address can be determined only for current frame.");

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  EVT VT = Op.getValueType();
  unsigned RA = IsN64 ? Mips::RA_64 : Mips::RA;
  MFI->setReturnAddressIsTaken(true);

  // Return RA, which contains the return address. Mark it an implicit live-in.
  unsigned Reg = MF.addLiveIn(RA, getRegClassFor(VT));
  return DAG.getCopyFromReg(DAG.getEntryNode(), Op.getDebugLoc(), Reg, VT);
}

// TODO: set SType according to the desired memory barrier behavior.
SDValue
MipsTargetLowering::LowerMEMBARRIER(SDValue Op, SelectionDAG &DAG) const {
  unsigned SType = 0;
  DebugLoc dl = Op.getDebugLoc();
  return DAG.getNode(MipsISD::Sync, dl, MVT::Other, Op.getOperand(0),
                     DAG.getConstant(SType, MVT::i32));
}

SDValue MipsTargetLowering::LowerATOMIC_FENCE(SDValue Op,
                                              SelectionDAG &DAG) const {
  // FIXME: Need pseudo-fence for 'singlethread' fences
  // FIXME: Set SType for weaker fences where supported/appropriate.
  unsigned SType = 0;
  DebugLoc dl = Op.getDebugLoc();
  return DAG.getNode(MipsISD::Sync, dl, MVT::Other, Op.getOperand(0),
                     DAG.getConstant(SType, MVT::i32));
}

SDValue MipsTargetLowering::LowerShiftLeftParts(SDValue Op,
                                                SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  SDValue Lo = Op.getOperand(0), Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);

  // if shamt < 32:
  //  lo = (shl lo, shamt)
  //  hi = (or (shl hi, shamt) (srl (srl lo, 1), ~shamt))
  // else:
  //  lo = 0
  //  hi = (shl lo, shamt[4:0])
  SDValue Not = DAG.getNode(ISD::XOR, DL, MVT::i32, Shamt,
                            DAG.getConstant(-1, MVT::i32));
  SDValue ShiftRight1Lo = DAG.getNode(ISD::SRL, DL, MVT::i32, Lo,
                                      DAG.getConstant(1, MVT::i32));
  SDValue ShiftRightLo = DAG.getNode(ISD::SRL, DL, MVT::i32, ShiftRight1Lo,
                                     Not);
  SDValue ShiftLeftHi = DAG.getNode(ISD::SHL, DL, MVT::i32, Hi, Shamt);
  SDValue Or = DAG.getNode(ISD::OR, DL, MVT::i32, ShiftLeftHi, ShiftRightLo);
  SDValue ShiftLeftLo = DAG.getNode(ISD::SHL, DL, MVT::i32, Lo, Shamt);
  SDValue Cond = DAG.getNode(ISD::AND, DL, MVT::i32, Shamt,
                             DAG.getConstant(0x20, MVT::i32));
  Lo = DAG.getNode(ISD::SELECT, DL, MVT::i32, Cond,
                   DAG.getConstant(0, MVT::i32), ShiftLeftLo);
  Hi = DAG.getNode(ISD::SELECT, DL, MVT::i32, Cond, ShiftLeftLo, Or);

  SDValue Ops[2] = {Lo, Hi};
  return DAG.getMergeValues(Ops, 2, DL);
}

SDValue MipsTargetLowering::LowerShiftRightParts(SDValue Op, SelectionDAG &DAG,
                                                 bool IsSRA) const {
  DebugLoc DL = Op.getDebugLoc();
  SDValue Lo = Op.getOperand(0), Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);

  // if shamt < 32:
  //  lo = (or (shl (shl hi, 1), ~shamt) (srl lo, shamt))
  //  if isSRA:
  //    hi = (sra hi, shamt)
  //  else:
  //    hi = (srl hi, shamt)
  // else:
  //  if isSRA:
  //   lo = (sra hi, shamt[4:0])
  //   hi = (sra hi, 31)
  //  else:
  //   lo = (srl hi, shamt[4:0])
  //   hi = 0
  SDValue Not = DAG.getNode(ISD::XOR, DL, MVT::i32, Shamt,
                            DAG.getConstant(-1, MVT::i32));
  SDValue ShiftLeft1Hi = DAG.getNode(ISD::SHL, DL, MVT::i32, Hi,
                                     DAG.getConstant(1, MVT::i32));
  SDValue ShiftLeftHi = DAG.getNode(ISD::SHL, DL, MVT::i32, ShiftLeft1Hi, Not);
  SDValue ShiftRightLo = DAG.getNode(ISD::SRL, DL, MVT::i32, Lo, Shamt);
  SDValue Or = DAG.getNode(ISD::OR, DL, MVT::i32, ShiftLeftHi, ShiftRightLo);
  SDValue ShiftRightHi = DAG.getNode(IsSRA ? ISD::SRA : ISD::SRL, DL, MVT::i32,
                                     Hi, Shamt);
  SDValue Cond = DAG.getNode(ISD::AND, DL, MVT::i32, Shamt,
                             DAG.getConstant(0x20, MVT::i32));
  SDValue Shift31 = DAG.getNode(ISD::SRA, DL, MVT::i32, Hi,
                                DAG.getConstant(31, MVT::i32));
  Lo = DAG.getNode(ISD::SELECT, DL, MVT::i32, Cond, ShiftRightHi, Or);
  Hi = DAG.getNode(ISD::SELECT, DL, MVT::i32, Cond,
                   IsSRA ? Shift31 : DAG.getConstant(0, MVT::i32),
                   ShiftRightHi);

  SDValue Ops[2] = {Lo, Hi};
  return DAG.getMergeValues(Ops, 2, DL);
}

static SDValue CreateLoadLR(unsigned Opc, SelectionDAG &DAG, LoadSDNode *LD,
                            SDValue Chain, SDValue Src, unsigned Offset) {
  SDValue Ptr = LD->getBasePtr();
  EVT VT = LD->getValueType(0), MemVT = LD->getMemoryVT();
  EVT BasePtrVT = Ptr.getValueType();
  DebugLoc DL = LD->getDebugLoc();
  SDVTList VTList = DAG.getVTList(VT, MVT::Other);

  if (Offset)
    Ptr = DAG.getNode(ISD::ADD, DL, BasePtrVT, Ptr,
                      DAG.getConstant(Offset, BasePtrVT));

  SDValue Ops[] = { Chain, Ptr, Src };
  return DAG.getMemIntrinsicNode(Opc, DL, VTList, Ops, 3, MemVT,
                                 LD->getMemOperand());
}

// Expand an unaligned 32 or 64-bit integer load node.
SDValue MipsTargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  LoadSDNode *LD = cast<LoadSDNode>(Op);
  EVT MemVT = LD->getMemoryVT();

  // Return if load is aligned or if MemVT is neither i32 nor i64.
  if ((LD->getAlignment() >= MemVT.getSizeInBits() / 8) ||
      ((MemVT != MVT::i32) && (MemVT != MVT::i64)))
    return SDValue();

  bool IsLittle = Subtarget->isLittle();
  EVT VT = Op.getValueType();
  ISD::LoadExtType ExtType = LD->getExtensionType();
  SDValue Chain = LD->getChain(), Undef = DAG.getUNDEF(VT);

  assert((VT == MVT::i32) || (VT == MVT::i64));

  // Expand
  //  (set dst, (i64 (load baseptr)))
  // to
  //  (set tmp, (ldl (add baseptr, 7), undef))
  //  (set dst, (ldr baseptr, tmp))
  if ((VT == MVT::i64) && (ExtType == ISD::NON_EXTLOAD)) {
    SDValue LDL = CreateLoadLR(MipsISD::LDL, DAG, LD, Chain, Undef,
                               IsLittle ? 7 : 0);
    return CreateLoadLR(MipsISD::LDR, DAG, LD, LDL.getValue(1), LDL,
                        IsLittle ? 0 : 7);
  }

  SDValue LWL = CreateLoadLR(MipsISD::LWL, DAG, LD, Chain, Undef,
                             IsLittle ? 3 : 0);
  SDValue LWR = CreateLoadLR(MipsISD::LWR, DAG, LD, LWL.getValue(1), LWL,
                             IsLittle ? 0 : 3);

  // Expand
  //  (set dst, (i32 (load baseptr))) or
  //  (set dst, (i64 (sextload baseptr))) or
  //  (set dst, (i64 (extload baseptr)))
  // to
  //  (set tmp, (lwl (add baseptr, 3), undef))
  //  (set dst, (lwr baseptr, tmp))
  if ((VT == MVT::i32) || (ExtType == ISD::SEXTLOAD) ||
      (ExtType == ISD::EXTLOAD))
    return LWR;

  assert((VT == MVT::i64) && (ExtType == ISD::ZEXTLOAD));

  // Expand
  //  (set dst, (i64 (zextload baseptr)))
  // to
  //  (set tmp0, (lwl (add baseptr, 3), undef))
  //  (set tmp1, (lwr baseptr, tmp0))
  //  (set tmp2, (shl tmp1, 32))
  //  (set dst, (srl tmp2, 32))
  DebugLoc DL = LD->getDebugLoc();
  SDValue Const32 = DAG.getConstant(32, MVT::i32);
  SDValue SLL = DAG.getNode(ISD::SHL, DL, MVT::i64, LWR, Const32);
  SDValue SRL = DAG.getNode(ISD::SRL, DL, MVT::i64, SLL, Const32);
  SDValue Ops[] = { SRL, LWR.getValue(1) };
  return DAG.getMergeValues(Ops, 2, DL);
}

static SDValue CreateStoreLR(unsigned Opc, SelectionDAG &DAG, StoreSDNode *SD,
                             SDValue Chain, unsigned Offset) {
  SDValue Ptr = SD->getBasePtr(), Value = SD->getValue();
  EVT MemVT = SD->getMemoryVT(), BasePtrVT = Ptr.getValueType();
  DebugLoc DL = SD->getDebugLoc();
  SDVTList VTList = DAG.getVTList(MVT::Other);

  if (Offset)
    Ptr = DAG.getNode(ISD::ADD, DL, BasePtrVT, Ptr,
                      DAG.getConstant(Offset, BasePtrVT));

  SDValue Ops[] = { Chain, Value, Ptr };
  return DAG.getMemIntrinsicNode(Opc, DL, VTList, Ops, 3, MemVT,
                                 SD->getMemOperand());
}

// Expand an unaligned 32 or 64-bit integer store node.
SDValue MipsTargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  StoreSDNode *SD = cast<StoreSDNode>(Op);
  EVT MemVT = SD->getMemoryVT();

  // Return if store is aligned or if MemVT is neither i32 nor i64.
  if ((SD->getAlignment() >= MemVT.getSizeInBits() / 8) ||
      ((MemVT != MVT::i32) && (MemVT != MVT::i64)))
    return SDValue();

  bool IsLittle = Subtarget->isLittle();
  SDValue Value = SD->getValue(), Chain = SD->getChain();
  EVT VT = Value.getValueType();

  // Expand
  //  (store val, baseptr) or
  //  (truncstore val, baseptr)
  // to
  //  (swl val, (add baseptr, 3))
  //  (swr val, baseptr)
  if ((VT == MVT::i32) || SD->isTruncatingStore()) {
    SDValue SWL = CreateStoreLR(MipsISD::SWL, DAG, SD, Chain,
                                IsLittle ? 3 : 0);
    return CreateStoreLR(MipsISD::SWR, DAG, SD, SWL, IsLittle ? 0 : 3);
  }

  assert(VT == MVT::i64);

  // Expand
  //  (store val, baseptr)
  // to
  //  (sdl val, (add baseptr, 7))
  //  (sdr val, baseptr)
  SDValue SDL = CreateStoreLR(MipsISD::SDL, DAG, SD, Chain, IsLittle ? 7 : 0);
  return CreateStoreLR(MipsISD::SDR, DAG, SD, SDL, IsLittle ? 0 : 7);
}

// This function expands mips intrinsic nodes which have 64-bit input operands
// or output values.
//
// out64 = intrinsic-node in64
// =>
// lo = copy (extract-element (in64, 0))
// hi = copy (extract-element (in64, 1))
// mips-specific-node
// v0 = copy lo
// v1 = copy hi
// out64 = merge-values (v0, v1)
//
static SDValue LowerDSPIntr(SDValue Op, SelectionDAG &DAG,
                            unsigned Opc, bool HasI64In, bool HasI64Out) {
  DebugLoc DL = Op.getDebugLoc();
  bool HasChainIn = Op->getOperand(0).getValueType() == MVT::Other;
  SDValue Chain = HasChainIn ? Op->getOperand(0) : DAG.getEntryNode();
  SmallVector<SDValue, 3> Ops;

  if (HasI64In) {
    SDValue InLo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32,
                               Op->getOperand(1 + HasChainIn),
                               DAG.getConstant(0, MVT::i32));
    SDValue InHi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32,
                               Op->getOperand(1 + HasChainIn),
                               DAG.getConstant(1, MVT::i32));

    Chain = DAG.getCopyToReg(Chain, DL, Mips::LO, InLo, SDValue());
    Chain = DAG.getCopyToReg(Chain, DL, Mips::HI, InHi, Chain.getValue(1));

    Ops.push_back(Chain);
    Ops.append(Op->op_begin() + HasChainIn + 2, Op->op_end());
    Ops.push_back(Chain.getValue(1));
  } else {
    Ops.push_back(Chain);
    Ops.append(Op->op_begin() + HasChainIn + 1, Op->op_end());
  }

  if (!HasI64Out)
    return DAG.getNode(Opc, DL, Op->value_begin(), Op->getNumValues(),
                       Ops.begin(), Ops.size());

  SDValue Intr = DAG.getNode(Opc, DL, DAG.getVTList(MVT::Other, MVT::Glue),
                             Ops.begin(), Ops.size());
  SDValue OutLo = DAG.getCopyFromReg(Intr.getValue(0), DL, Mips::LO, MVT::i32,
                                     Intr.getValue(1));
  SDValue OutHi = DAG.getCopyFromReg(OutLo.getValue(1), DL, Mips::HI, MVT::i32,
                                     OutLo.getValue(2));
  SDValue Out = DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, OutLo, OutHi);

  if (!HasChainIn)
    return Out;

  SDValue Vals[] = { Out, OutHi.getValue(1) };
  return DAG.getMergeValues(Vals, 2, DL);
}

SDValue MipsTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                    SelectionDAG &DAG) const {
  switch (cast<ConstantSDNode>(Op->getOperand(0))->getZExtValue()) {
  default:
    return SDValue();
  case Intrinsic::mips_shilo:
    return LowerDSPIntr(Op, DAG, MipsISD::SHILO, true, true);
  case Intrinsic::mips_dpau_h_qbl:
    return LowerDSPIntr(Op, DAG, MipsISD::DPAU_H_QBL, true, true);
  case Intrinsic::mips_dpau_h_qbr:
    return LowerDSPIntr(Op, DAG, MipsISD::DPAU_H_QBR, true, true);
  case Intrinsic::mips_dpsu_h_qbl:
    return LowerDSPIntr(Op, DAG, MipsISD::DPSU_H_QBL, true, true);
  case Intrinsic::mips_dpsu_h_qbr:
    return LowerDSPIntr(Op, DAG, MipsISD::DPSU_H_QBR, true, true);
  case Intrinsic::mips_dpa_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPA_W_PH, true, true);
  case Intrinsic::mips_dps_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPS_W_PH, true, true);
  case Intrinsic::mips_dpax_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPAX_W_PH, true, true);
  case Intrinsic::mips_dpsx_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPSX_W_PH, true, true);
  case Intrinsic::mips_mulsa_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::MULSA_W_PH, true, true);
  case Intrinsic::mips_mult:
    return LowerDSPIntr(Op, DAG, MipsISD::MULT, false, true);
  case Intrinsic::mips_multu:
    return LowerDSPIntr(Op, DAG, MipsISD::MULTU, false, true);
  case Intrinsic::mips_madd:
    return LowerDSPIntr(Op, DAG, MipsISD::MADD_DSP, true, true);
  case Intrinsic::mips_maddu:
    return LowerDSPIntr(Op, DAG, MipsISD::MADDU_DSP, true, true);
  case Intrinsic::mips_msub:
    return LowerDSPIntr(Op, DAG, MipsISD::MSUB_DSP, true, true);
  case Intrinsic::mips_msubu:
    return LowerDSPIntr(Op, DAG, MipsISD::MSUBU_DSP, true, true);
  }
}

SDValue MipsTargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op,
                                                   SelectionDAG &DAG) const {
  switch (cast<ConstantSDNode>(Op->getOperand(1))->getZExtValue()) {
  default:
    return SDValue();
  case Intrinsic::mips_extp:
    return LowerDSPIntr(Op, DAG, MipsISD::EXTP, true, false);
  case Intrinsic::mips_extpdp:
    return LowerDSPIntr(Op, DAG, MipsISD::EXTPDP, true, false);
  case Intrinsic::mips_extr_w:
    return LowerDSPIntr(Op, DAG, MipsISD::EXTR_W, true, false);
  case Intrinsic::mips_extr_r_w:
    return LowerDSPIntr(Op, DAG, MipsISD::EXTR_R_W, true, false);
  case Intrinsic::mips_extr_rs_w:
    return LowerDSPIntr(Op, DAG, MipsISD::EXTR_RS_W, true, false);
  case Intrinsic::mips_extr_s_h:
    return LowerDSPIntr(Op, DAG, MipsISD::EXTR_S_H, true, false);
  case Intrinsic::mips_mthlip:
    return LowerDSPIntr(Op, DAG, MipsISD::MTHLIP, true, true);
  case Intrinsic::mips_mulsaq_s_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::MULSAQ_S_W_PH, true, true);
  case Intrinsic::mips_maq_s_w_phl:
    return LowerDSPIntr(Op, DAG, MipsISD::MAQ_S_W_PHL, true, true);
  case Intrinsic::mips_maq_s_w_phr:
    return LowerDSPIntr(Op, DAG, MipsISD::MAQ_S_W_PHR, true, true);
  case Intrinsic::mips_maq_sa_w_phl:
    return LowerDSPIntr(Op, DAG, MipsISD::MAQ_SA_W_PHL, true, true);
  case Intrinsic::mips_maq_sa_w_phr:
    return LowerDSPIntr(Op, DAG, MipsISD::MAQ_SA_W_PHR, true, true);
  case Intrinsic::mips_dpaq_s_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPAQ_S_W_PH, true, true);
  case Intrinsic::mips_dpsq_s_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPSQ_S_W_PH, true, true);
  case Intrinsic::mips_dpaq_sa_l_w:
    return LowerDSPIntr(Op, DAG, MipsISD::DPAQ_SA_L_W, true, true);
  case Intrinsic::mips_dpsq_sa_l_w:
    return LowerDSPIntr(Op, DAG, MipsISD::DPSQ_SA_L_W, true, true);
  case Intrinsic::mips_dpaqx_s_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPAQX_S_W_PH, true, true);
  case Intrinsic::mips_dpaqx_sa_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPAQX_SA_W_PH, true, true);
  case Intrinsic::mips_dpsqx_s_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPSQX_S_W_PH, true, true);
  case Intrinsic::mips_dpsqx_sa_w_ph:
    return LowerDSPIntr(Op, DAG, MipsISD::DPSQX_SA_W_PH, true, true);
  }
}

SDValue MipsTargetLowering::LowerADD(SDValue Op, SelectionDAG &DAG) const {
  if (Op->getOperand(0).getOpcode() != ISD::FRAMEADDR
      || cast<ConstantSDNode>
        (Op->getOperand(0).getOperand(0))->getZExtValue() != 0
      || Op->getOperand(1).getOpcode() != ISD::FRAME_TO_ARGS_OFFSET)
    return SDValue();

  // The pattern
  //   (add (frameaddr 0), (frame_to_args_offset))
  // results from lowering llvm.eh.dwarf.cfa intrinsic. Transform it to
  //   (add FrameObject, 0)
  // where FrameObject is a fixed StackObject with offset 0 which points to
  // the old stack pointer.
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  EVT ValTy = Op->getValueType(0);
  int FI = MFI->CreateFixedObject(Op.getValueSizeInBits() / 8, 0, false);
  SDValue InArgsAddr = DAG.getFrameIndex(FI, ValTy);
  return DAG.getNode(ISD::ADD, Op->getDebugLoc(), ValTy, InArgsAddr,
                     DAG.getConstant(0, ValTy));
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: Implement a generic logic using tblgen that can support this.
// Mips O32 ABI rules:
// ---
// i32 - Passed in A0, A1, A2, A3 and stack
// f32 - Only passed in f32 registers if no int reg has been used yet to hold
//       an argument. Otherwise, passed in A1, A2, A3 and stack.
// f64 - Only passed in two aliased f32 registers if no int reg has been used
//       yet to hold an argument. Otherwise, use A2, A3 and stack. If A1 is
//       not used, it must be shadowed. If only A3 is avaiable, shadow it and
//       go to stack.
//
//  For vararg functions, all arguments are passed in A0, A1, A2, A3 and stack.
//===----------------------------------------------------------------------===//

static bool CC_MipsO32(unsigned ValNo, MVT ValVT,
                       MVT LocVT, CCValAssign::LocInfo LocInfo,
                       ISD::ArgFlagsTy ArgFlags, CCState &State) {

  static const unsigned IntRegsSize=4, FloatRegsSize=2;

  static const uint16_t IntRegs[] = {
      Mips::A0, Mips::A1, Mips::A2, Mips::A3
  };
  static const uint16_t F32Regs[] = {
      Mips::F12, Mips::F14
  };
  static const uint16_t F64Regs[] = {
      Mips::D6, Mips::D7
  };

  // Do not process byval args here.
  if (ArgFlags.isByVal())
    return true;

  // Promote i8 and i16
  if (LocVT == MVT::i8 || LocVT == MVT::i16) {
    LocVT = MVT::i32;
    if (ArgFlags.isSExt())
      LocInfo = CCValAssign::SExt;
    else if (ArgFlags.isZExt())
      LocInfo = CCValAssign::ZExt;
    else
      LocInfo = CCValAssign::AExt;
  }

  unsigned Reg;

  // f32 and f64 are allocated in A0, A1, A2, A3 when either of the following
  // is true: function is vararg, argument is 3rd or higher, there is previous
  // argument which is not f32 or f64.
  bool AllocateFloatsInIntReg = State.isVarArg() || ValNo > 1
      || State.getFirstUnallocated(F32Regs, FloatRegsSize) != ValNo;
  unsigned OrigAlign = ArgFlags.getOrigAlign();
  bool isI64 = (ValVT == MVT::i32 && OrigAlign == 8);

  if (ValVT == MVT::i32 || (ValVT == MVT::f32 && AllocateFloatsInIntReg)) {
    Reg = State.AllocateReg(IntRegs, IntRegsSize);
    // If this is the first part of an i64 arg,
    // the allocated register must be either A0 or A2.
    if (isI64 && (Reg == Mips::A1 || Reg == Mips::A3))
      Reg = State.AllocateReg(IntRegs, IntRegsSize);
    LocVT = MVT::i32;
  } else if (ValVT == MVT::f64 && AllocateFloatsInIntReg) {
    // Allocate int register and shadow next int register. If first
    // available register is Mips::A1 or Mips::A3, shadow it too.
    Reg = State.AllocateReg(IntRegs, IntRegsSize);
    if (Reg == Mips::A1 || Reg == Mips::A3)
      Reg = State.AllocateReg(IntRegs, IntRegsSize);
    State.AllocateReg(IntRegs, IntRegsSize);
    LocVT = MVT::i32;
  } else if (ValVT.isFloatingPoint() && !AllocateFloatsInIntReg) {
    // we are guaranteed to find an available float register
    if (ValVT == MVT::f32) {
      Reg = State.AllocateReg(F32Regs, FloatRegsSize);
      // Shadow int register
      State.AllocateReg(IntRegs, IntRegsSize);
    } else {
      Reg = State.AllocateReg(F64Regs, FloatRegsSize);
      // Shadow int registers
      unsigned Reg2 = State.AllocateReg(IntRegs, IntRegsSize);
      if (Reg2 == Mips::A1 || Reg2 == Mips::A3)
        State.AllocateReg(IntRegs, IntRegsSize);
      State.AllocateReg(IntRegs, IntRegsSize);
    }
  } else
    llvm_unreachable("Cannot handle this ValVT.");

  if (!Reg) {
    unsigned Offset = State.AllocateStack(ValVT.getSizeInBits() >> 3,
                                          OrigAlign);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  } else
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));

  return false;
}

#include "MipsGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//                  Call Calling Convention Implementation
//===----------------------------------------------------------------------===//

static const unsigned O32IntRegsSize = 4;

// Return next O32 integer argument register.
static unsigned getNextIntArgReg(unsigned Reg) {
  assert((Reg == Mips::A0) || (Reg == Mips::A2));
  return (Reg == Mips::A0) ? Mips::A1 : Mips::A3;
}

/// IsEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization.
bool MipsTargetLowering::
IsEligibleForTailCallOptimization(const MipsCC &MipsCCInfo,
                                  unsigned NextStackOffset,
                                  const MipsFunctionInfo& FI) const {
  if (!EnableMipsTailCalls)
    return false;

  // No tail call optimization for mips16.
  if (Subtarget->inMips16Mode())
    return false;

  // Return false if either the callee or caller has a byval argument.
  if (MipsCCInfo.hasByValArg() || FI.hasByvalArg())
    return false;

  // Return true if the callee's argument area is no larger than the
  // caller's.
  return NextStackOffset <= FI.getIncomingArgSize();
}

SDValue
MipsTargetLowering::passArgOnStack(SDValue StackPtr, unsigned Offset,
                                   SDValue Chain, SDValue Arg, DebugLoc DL,
                                   bool IsTailCall, SelectionDAG &DAG) const {
  if (!IsTailCall) {
    SDValue PtrOff = DAG.getNode(ISD::ADD, DL, getPointerTy(), StackPtr,
                                 DAG.getIntPtrConstant(Offset));
    return DAG.getStore(Chain, DL, Arg, PtrOff, MachinePointerInfo(), false,
                        false, 0);
  }

  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  int FI = MFI->CreateFixedObject(Arg.getValueSizeInBits() / 8, Offset, false);
  SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
  return DAG.getStore(Chain, DL, Arg, FIN, MachinePointerInfo(),
                      /*isVolatile=*/ true, false, 0);
}

/// LowerCall - functions arguments are copied from virtual regs to
/// (physical regs)/(stack frame), CALLSEQ_START and CALLSEQ_END are emitted.
SDValue
MipsTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                              SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG                     = CLI.DAG;
  DebugLoc &dl                          = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals     = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins   = CLI.Ins;
  SDValue Chain                         = CLI.Chain;
  SDValue Callee                        = CLI.Callee;
  bool &isTailCall                      = CLI.IsTailCall;
  CallingConv::ID CallConv              = CLI.CallConv;
  bool isVarArg                         = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetFrameLowering *TFL = MF.getTarget().getFrameLowering();
  bool IsPIC = getTargetMachine().getRelocationModel() == Reloc::PIC_;

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());
  MipsCC MipsCCInfo(CallConv, isVarArg, IsO32, CCInfo);

  MipsCCInfo.analyzeCallOperands(Outs);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NextStackOffset = CCInfo.getNextStackOffset();

  // Check if it's really possible to do a tail call.
  if (isTailCall)
    isTailCall =
      IsEligibleForTailCallOptimization(MipsCCInfo, NextStackOffset,
                                        *MF.getInfo<MipsFunctionInfo>());

  if (isTailCall)
    ++NumTailCalls;

  // Chain is the output chain of the last Load/Store or CopyToReg node.
  // ByValChain is the output chain of the last Memcpy node created for copying
  // byval arguments to the stack.
  unsigned StackAlignment = TFL->getStackAlignment();
  NextStackOffset = RoundUpToAlignment(NextStackOffset, StackAlignment);
  SDValue NextStackOffsetVal = DAG.getIntPtrConstant(NextStackOffset, true);

  if (!isTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, NextStackOffsetVal);

  SDValue StackPtr = DAG.getCopyFromReg(Chain, dl,
                                        IsN64 ? Mips::SP_64 : Mips::SP,
                                        getPointerTy());

  // With EABI is it possible to have 16 args on registers.
  SmallVector<std::pair<unsigned, SDValue>, 16> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  MipsCC::byval_iterator ByValArg = MipsCCInfo.byval_begin();

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    SDValue Arg = OutVals[i];
    CCValAssign &VA = ArgLocs[i];
    MVT ValVT = VA.getValVT(), LocVT = VA.getLocVT();
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // ByVal Arg.
    if (Flags.isByVal()) {
      assert(Flags.getByValSize() &&
             "ByVal args of size 0 should have been ignored by front-end.");
      assert(ByValArg != MipsCCInfo.byval_end());
      assert(!isTailCall &&
             "Do not tail-call optimize if there is a byval argument.");
      passByValArg(Chain, dl, RegsToPass, MemOpChains, StackPtr, MFI, DAG, Arg,
                   MipsCCInfo, *ByValArg, Flags, Subtarget->isLittle());
      ++ByValArg;
      continue;
    }

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      if (VA.isRegLoc()) {
        if ((ValVT == MVT::f32 && LocVT == MVT::i32) ||
            (ValVT == MVT::f64 && LocVT == MVT::i64))
          Arg = DAG.getNode(ISD::BITCAST, dl, LocVT, Arg);
        else if (ValVT == MVT::f64 && LocVT == MVT::i32) {
          SDValue Lo = DAG.getNode(MipsISD::ExtractElementF64, dl, MVT::i32,
                                   Arg, DAG.getConstant(0, MVT::i32));
          SDValue Hi = DAG.getNode(MipsISD::ExtractElementF64, dl, MVT::i32,
                                   Arg, DAG.getConstant(1, MVT::i32));
          if (!Subtarget->isLittle())
            std::swap(Lo, Hi);
          unsigned LocRegLo = VA.getLocReg();
          unsigned LocRegHigh = getNextIntArgReg(LocRegLo);
          RegsToPass.push_back(std::make_pair(LocRegLo, Lo));
          RegsToPass.push_back(std::make_pair(LocRegHigh, Hi));
          continue;
        }
      }
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, dl, LocVT, Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, dl, LocVT, Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, dl, LocVT, Arg);
      break;
    }

    // Arguments that can be passed on register must be kept at
    // RegsToPass vector
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      continue;
    }

    // Register can't get to this point...
    assert(VA.isMemLoc());

    // emit ISD::STORE whichs stores the
    // parameter value to a stack Location
    MemOpChains.push_back(passArgOnStack(StackPtr, VA.getLocMemOffset(),
                                         Chain, Arg, dl, isTailCall, DAG));
  }

  // Transform all store nodes into one single node because all store
  // nodes are independent of each other.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  unsigned char OpFlag;
  bool IsPICCall = (IsN64 || IsPIC); // true if calls are translated to jalr $25
  bool GlobalOrExternal = false;
  SDValue CalleeLo;

  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    if (IsPICCall && G->getGlobal()->hasInternalLinkage()) {
      OpFlag = IsO32 ? MipsII::MO_GOT : MipsII::MO_GOT_PAGE;
      unsigned char LoFlag = IsO32 ? MipsII::MO_ABS_LO : MipsII::MO_GOT_OFST;
      Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl, getPointerTy(), 0,
                                          OpFlag);
      CalleeLo = DAG.getTargetGlobalAddress(G->getGlobal(), dl, getPointerTy(),
                                            0, LoFlag);
    } else {
      OpFlag = IsPICCall ? MipsII::MO_GOT_CALL : MipsII::MO_NO_FLAG;
      Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl,
                                          getPointerTy(), 0, OpFlag);
    }

    GlobalOrExternal = true;
  }
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    if (IsN64 || (!IsO32 && IsPIC))
      OpFlag = MipsII::MO_GOT_DISP;
    else if (!IsPIC) // !N64 && static
      OpFlag = MipsII::MO_NO_FLAG;
    else // O32 & PIC
      OpFlag = MipsII::MO_GOT_CALL;
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy(),
                                         OpFlag);
    GlobalOrExternal = true;
  }

  SDValue InFlag;

  // Create nodes that load address of callee and copy it to T9
  if (IsPICCall) {
    if (GlobalOrExternal) {
      // Load callee address
      Callee = DAG.getNode(MipsISD::Wrapper, dl, getPointerTy(),
                           GetGlobalReg(DAG, getPointerTy()), Callee);
      SDValue LoadValue = DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(),
                                      Callee, MachinePointerInfo::getGOT(),
                                      false, false, false, 0);

      // Use GOT+LO if callee has internal linkage.
      if (CalleeLo.getNode()) {
        SDValue Lo = DAG.getNode(MipsISD::Lo, dl, getPointerTy(), CalleeLo);
        Callee = DAG.getNode(ISD::ADD, dl, getPointerTy(), LoadValue, Lo);
      } else
        Callee = LoadValue;
    }
  }

  // T9 register operand.
  SDValue T9;

  // T9 should contain the address of the callee function if
  // -reloction-model=pic or it is an indirect call.
  if (IsPICCall || !GlobalOrExternal) {
    // copy to T9
    unsigned T9Reg = IsN64 ? Mips::T9_64 : Mips::T9;
    Chain = DAG.getCopyToReg(Chain, dl, T9Reg, Callee, SDValue(0, 0));
    InFlag = Chain.getValue(1);

    if (Subtarget->inMips16Mode())
      T9 = DAG.getRegister(T9Reg, getPointerTy());
    else
      Callee = DAG.getRegister(T9Reg, getPointerTy());
  }

  // Insert node "GP copy globalreg" before call to function.
  // Lazy-binding stubs require GP to point to the GOT.
  if (IsPICCall) {
    unsigned GPReg = IsN64 ? Mips::GP_64 : Mips::GP;
    EVT Ty = IsN64 ? MVT::i64 : MVT::i32;
    RegsToPass.push_back(std::make_pair(GPReg, GetGlobalReg(DAG, Ty)));
  }

  // Build a sequence of copy-to-reg nodes chained together with token
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emitted instructions must be
  // stuck together.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // MipsJmpLink = #chain, #target_address, #opt_in_flags...
  //             = Chain, Callee, Reg#1, Reg#2, ...
  //
  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  // Add T9 register operand.
  if (T9.getNode())
    Ops.push_back(T9);

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  if (isTailCall)
    return DAG.getNode(MipsISD::TailCall, dl, MVT::Other, &Ops[0], Ops.size());

  Chain  = DAG.getNode(MipsISD::JmpLink, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain, NextStackOffsetVal,
                             DAG.getIntPtrConstant(0, true), InFlag);
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg,
                         Ins, dl, DAG, InVals);
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
SDValue
MipsTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                    CallingConv::ID CallConv, bool isVarArg,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const {
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  CCInfo.AnalyzeCallResult(Ins, RetCC_Mips);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    Chain = DAG.getCopyFromReg(Chain, dl, RVLocs[i].getLocReg(),
                               RVLocs[i].getValVT(), InFlag).getValue(1);
    InFlag = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//             Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//
/// LowerFormalArguments - transform physical registers into virtual registers
/// and generate load operations for arguments places on the stack.
SDValue
MipsTargetLowering::LowerFormalArguments(SDValue Chain,
                                         CallingConv::ID CallConv,
                                         bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                         DebugLoc dl, SelectionDAG &DAG,
                                         SmallVectorImpl<SDValue> &InVals)
                                          const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  MipsFI->setVarArgsFrameIndex(0);

  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());
  MipsCC MipsCCInfo(CallConv, isVarArg, IsO32, CCInfo);

  MipsCCInfo.analyzeFormalArguments(Ins);
  MipsFI->setFormalArgInfo(CCInfo.getNextStackOffset(),
                           MipsCCInfo.hasByValArg());

  Function::const_arg_iterator FuncArg =
    DAG.getMachineFunction().getFunction()->arg_begin();
  unsigned CurArgIdx = 0;
  MipsCC::byval_iterator ByValArg = MipsCCInfo.byval_begin();

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    std::advance(FuncArg, Ins[i].OrigArgIndex - CurArgIdx);
    CurArgIdx = Ins[i].OrigArgIndex;
    EVT ValVT = VA.getValVT();
    ISD::ArgFlagsTy Flags = Ins[i].Flags;
    bool IsRegLoc = VA.isRegLoc();

    if (Flags.isByVal()) {
      assert(Flags.getByValSize() &&
             "ByVal args of size 0 should have been ignored by front-end.");
      assert(ByValArg != MipsCCInfo.byval_end());
      copyByValRegs(Chain, dl, OutChains, DAG, Flags, InVals, &*FuncArg,
                    MipsCCInfo, *ByValArg);
      ++ByValArg;
      continue;
    }

    // Arguments stored on registers
    if (IsRegLoc) {
      EVT RegVT = VA.getLocVT();
      unsigned ArgReg = VA.getLocReg();
      const TargetRegisterClass *RC;

      if (RegVT == MVT::i32)
        RC = &Mips::CPURegsRegClass;
      else if (RegVT == MVT::i64)
        RC = &Mips::CPU64RegsRegClass;
      else if (RegVT == MVT::f32)
        RC = &Mips::FGR32RegClass;
      else if (RegVT == MVT::f64)
        RC = HasMips64 ? &Mips::FGR64RegClass : &Mips::AFGR64RegClass;
      else
        llvm_unreachable("RegVT not supported by FormalArguments Lowering");

      // Transform the arguments stored on
      // physical registers into virtual ones
      unsigned Reg = AddLiveIn(DAG.getMachineFunction(), ArgReg, RC);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);

      // If this is an 8 or 16-bit value, it has been passed promoted
      // to 32 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      if (VA.getLocInfo() != CCValAssign::Full) {
        unsigned Opcode = 0;
        if (VA.getLocInfo() == CCValAssign::SExt)
          Opcode = ISD::AssertSext;
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          Opcode = ISD::AssertZext;
        if (Opcode)
          ArgValue = DAG.getNode(Opcode, dl, RegVT, ArgValue,
                                 DAG.getValueType(ValVT));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, ValVT, ArgValue);
      }

      // Handle floating point arguments passed in integer registers.
      if ((RegVT == MVT::i32 && ValVT == MVT::f32) ||
          (RegVT == MVT::i64 && ValVT == MVT::f64))
        ArgValue = DAG.getNode(ISD::BITCAST, dl, ValVT, ArgValue);
      else if (IsO32 && RegVT == MVT::i32 && ValVT == MVT::f64) {
        unsigned Reg2 = AddLiveIn(DAG.getMachineFunction(),
                                  getNextIntArgReg(ArgReg), RC);
        SDValue ArgValue2 = DAG.getCopyFromReg(Chain, dl, Reg2, RegVT);
        if (!Subtarget->isLittle())
          std::swap(ArgValue, ArgValue2);
        ArgValue = DAG.getNode(MipsISD::BuildPairF64, dl, MVT::f64,
                               ArgValue, ArgValue2);
      }

      InVals.push_back(ArgValue);
    } else { // VA.isRegLoc()

      // sanity check
      assert(VA.isMemLoc());

      // The stack pointer offset is relative to the caller stack frame.
      int FI = MFI->CreateFixedObject(ValVT.getSizeInBits()/8,
                                      VA.getLocMemOffset(), true);

      // Create load nodes to retrieve arguments from the stack
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
      InVals.push_back(DAG.getLoad(ValVT, dl, Chain, FIN,
                                   MachinePointerInfo::getFixedStack(FI),
                                   false, false, false, 0));
    }
  }

  // The mips ABIs for returning structs by value requires that we copy
  // the sret argument into $v0 for the return. Save the argument into
  // a virtual register so that we can access it from the return points.
  if (DAG.getMachineFunction().getFunction()->hasStructRetAttr()) {
    unsigned Reg = MipsFI->getSRetReturnReg();
    if (!Reg) {
      Reg = MF.getRegInfo().
        createVirtualRegister(getRegClassFor(IsN64 ? MVT::i64 : MVT::i32));
      MipsFI->setSRetReturnReg(Reg);
    }
    SDValue Copy = DAG.getCopyToReg(DAG.getEntryNode(), dl, Reg, InVals[0]);
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Copy, Chain);
  }

  if (isVarArg)
    writeVarArgRegs(OutChains, MipsCCInfo, Chain, dl, DAG);

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens when on varg functions
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &OutChains[0], OutChains.size());
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

bool
MipsTargetLowering::CanLowerReturn(CallingConv::ID CallConv,
                                   MachineFunction &MF, bool isVarArg,
                                   const SmallVectorImpl<ISD::OutputArg> &Outs,
                                   LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(),
                 RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_Mips);
}

SDValue
MipsTargetLowering::LowerReturn(SDValue Chain,
                                CallingConv::ID CallConv, bool isVarArg,
                                const SmallVectorImpl<ISD::OutputArg> &Outs,
                                const SmallVectorImpl<SDValue> &OutVals,
                                DebugLoc dl, SelectionDAG &DAG) const {

  // CCValAssign - represent the assignment of
  // the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  // Analize return values.
  CCInfo.AnalyzeReturn(Outs, RetCC_Mips);

  // If this is the first return lowered for this function, add
  // the regs to the liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), OutVals[i], Flag);

    // guarantee that all emitted copies are
    // stuck together, avoiding something bad
    Flag = Chain.getValue(1);
  }

  // The mips ABIs for returning structs by value requires that we copy
  // the sret argument into $v0 for the return. We saved the argument into
  // a virtual register in the entry block, so now we copy the value out
  // and into $v0.
  if (DAG.getMachineFunction().getFunction()->hasStructRetAttr()) {
    MachineFunction &MF      = DAG.getMachineFunction();
    MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
    unsigned Reg = MipsFI->getSRetReturnReg();

    if (!Reg)
      llvm_unreachable("sret virtual register not created in the entry block");
    SDValue Val = DAG.getCopyFromReg(Chain, dl, Reg, getPointerTy());
    unsigned V0 = IsN64 ? Mips::V0_64 : Mips::V0;

    Chain = DAG.getCopyToReg(Chain, dl, V0, Val, Flag);
    Flag = Chain.getValue(1);
    MF.getRegInfo().addLiveOut(V0);
  }

  // Return on Mips is always a "jr $ra"
  if (Flag.getNode())
    return DAG.getNode(MipsISD::Ret, dl, MVT::Other, Chain, Flag);

  // Return Void
  return DAG.getNode(MipsISD::Ret, dl, MVT::Other, Chain);
}

//===----------------------------------------------------------------------===//
//                           Mips Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
MipsTargetLowering::ConstraintType MipsTargetLowering::
getConstraintType(const std::string &Constraint) const
{
  // Mips specific constrainy
  // GCC config/mips/constraints.md
  //
  // 'd' : An address register. Equivalent to r
  //       unless generating MIPS16 code.
  // 'y' : Equivalent to r; retained for
  //       backwards compatibility.
  // 'c' : A register suitable for use in an indirect
  //       jump. This will always be $25 for -mabicalls.
  // 'l' : The lo register. 1 word storage.
  // 'x' : The hilo register pair. Double word storage.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
      default : break;
      case 'd':
      case 'y':
      case 'f':
      case 'c':
      case 'l':
      case 'x':
        return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
MipsTargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
    // If we don't have a value, we can't do a match,
    // but allow it at the lowest weight.
  if (CallOperandVal == NULL)
    return CW_Default;
  Type *type = CallOperandVal->getType();
  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;
  case 'd':
  case 'y':
    if (type->isIntegerTy())
      weight = CW_Register;
    break;
  case 'f':
    if (type->isFloatTy())
      weight = CW_Register;
    break;
  case 'c': // $25 for indirect jumps
  case 'l': // lo register
  case 'x': // hilo register pair
      if (type->isIntegerTy())
      weight = CW_SpecificReg;
      break;
  case 'I': // signed 16 bit immediate
  case 'J': // integer zero
  case 'K': // unsigned 16 bit immediate
  case 'L': // signed 32 bit immediate where lower 16 bits are 0
  case 'N': // immediate in the range of -65535 to -1 (inclusive)
  case 'O': // signed 15 bit immediate (+- 16383)
  case 'P': // immediate in the range of 65535 to 1 (inclusive)
    if (isa<ConstantInt>(CallOperandVal))
      weight = CW_Constant;
    break;
  }
  return weight;
}

/// Given a register class constraint, like 'r', if this corresponds directly
/// to an LLVM register class, return a register of 0 and the register class
/// pointer.
std::pair<unsigned, const TargetRegisterClass*> MipsTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint, EVT VT) const
{
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'd': // Address register. Same as 'r' unless generating MIPS16 code.
    case 'y': // Same as 'r'. Exists for compatibility.
    case 'r':
      if (VT == MVT::i32 || VT == MVT::i16 || VT == MVT::i8) {
        if (Subtarget->inMips16Mode())
          return std::make_pair(0U, &Mips::CPU16RegsRegClass);
        return std::make_pair(0U, &Mips::CPURegsRegClass);
      }
      if (VT == MVT::i64 && !HasMips64)
        return std::make_pair(0U, &Mips::CPURegsRegClass);
      if (VT == MVT::i64 && HasMips64)
        return std::make_pair(0U, &Mips::CPU64RegsRegClass);
      // This will generate an error message
      return std::make_pair(0u, static_cast<const TargetRegisterClass*>(0));
    case 'f':
      if (VT == MVT::f32)
        return std::make_pair(0U, &Mips::FGR32RegClass);
      if ((VT == MVT::f64) && (!Subtarget->isSingleFloat())) {
        if (Subtarget->isFP64bit())
          return std::make_pair(0U, &Mips::FGR64RegClass);
        return std::make_pair(0U, &Mips::AFGR64RegClass);
      }
      break;
    case 'c': // register suitable for indirect jump
      if (VT == MVT::i32)
        return std::make_pair((unsigned)Mips::T9, &Mips::CPURegsRegClass);
      assert(VT == MVT::i64 && "Unexpected type.");
      return std::make_pair((unsigned)Mips::T9_64, &Mips::CPU64RegsRegClass);
    case 'l': // register suitable for indirect jump
      if (VT == MVT::i32)
        return std::make_pair((unsigned)Mips::LO, &Mips::HILORegClass);
      return std::make_pair((unsigned)Mips::LO64, &Mips::HILO64RegClass);
    case 'x': // register suitable for indirect jump
      // Fixme: Not triggering the use of both hi and low
      // This will generate an error message
      return std::make_pair(0u, static_cast<const TargetRegisterClass*>(0));
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void MipsTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                     std::string &Constraint,
                                                     std::vector<SDValue>&Ops,
                                                     SelectionDAG &DAG) const {
  SDValue Result(0, 0);

  // Only support length 1 constraints for now.
  if (Constraint.length() > 1) return;

  char ConstraintLetter = Constraint[0];
  switch (ConstraintLetter) {
  default: break; // This will fall through to the generic implementation
  case 'I': // Signed 16 bit constant
    // If this fails, the parent routine will give an error
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      EVT Type = Op.getValueType();
      int64_t Val = C->getSExtValue();
      if (isInt<16>(Val)) {
        Result = DAG.getTargetConstant(Val, Type);
        break;
      }
    }
    return;
  case 'J': // integer zero
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      EVT Type = Op.getValueType();
      int64_t Val = C->getZExtValue();
      if (Val == 0) {
        Result = DAG.getTargetConstant(0, Type);
        break;
      }
    }
    return;
  case 'K': // unsigned 16 bit immediate
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      EVT Type = Op.getValueType();
      uint64_t Val = (uint64_t)C->getZExtValue();
      if (isUInt<16>(Val)) {
        Result = DAG.getTargetConstant(Val, Type);
        break;
      }
    }
    return;
  case 'L': // signed 32 bit immediate where lower 16 bits are 0
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      EVT Type = Op.getValueType();
      int64_t Val = C->getSExtValue();
      if ((isInt<32>(Val)) && ((Val & 0xffff) == 0)){
        Result = DAG.getTargetConstant(Val, Type);
        break;
      }
    }
    return;
  case 'N': // immediate in the range of -65535 to -1 (inclusive)
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      EVT Type = Op.getValueType();
      int64_t Val = C->getSExtValue();
      if ((Val >= -65535) && (Val <= -1)) {
        Result = DAG.getTargetConstant(Val, Type);
        break;
      }
    }
    return;
  case 'O': // signed 15 bit immediate
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      EVT Type = Op.getValueType();
      int64_t Val = C->getSExtValue();
      if ((isInt<15>(Val))) {
        Result = DAG.getTargetConstant(Val, Type);
        break;
      }
    }
    return;
  case 'P': // immediate in the range of 1 to 65535 (inclusive)
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      EVT Type = Op.getValueType();
      int64_t Val = C->getSExtValue();
      if ((Val <= 65535) && (Val >= 1)) {
        Result = DAG.getTargetConstant(Val, Type);
        break;
      }
    }
    return;
  }

  if (Result.getNode()) {
    Ops.push_back(Result);
    return;
  }

  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

bool
MipsTargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The Mips target isn't yet aware of offsets.
  return false;
}

EVT MipsTargetLowering::getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                                            unsigned SrcAlign, bool IsZeroVal,
                                            bool MemcpyStrSrc,
                                            MachineFunction &MF) const {
  if (Subtarget->hasMips64())
    return MVT::i64;

  return MVT::i32;
}

bool MipsTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  if (VT != MVT::f32 && VT != MVT::f64)
    return false;
  if (Imm.isNegZero())
    return false;
  return Imm.isZero();
}

unsigned MipsTargetLowering::getJumpTableEncoding() const {
  if (IsN64)
    return MachineJumpTableInfo::EK_GPRel64BlockAddress;

  return TargetLowering::getJumpTableEncoding();
}

MipsTargetLowering::MipsCC::MipsCC(CallingConv::ID CallConv, bool IsVarArg,
                                   bool IsO32, CCState &Info) : CCInfo(Info) {
  UseRegsForByval = true;

  if (IsO32) {
    RegSize = 4;
    NumIntArgRegs = array_lengthof(O32IntRegs);
    ReservedArgArea = 16;
    IntArgRegs = ShadowRegs = O32IntRegs;
    FixedFn = VarFn = CC_MipsO32;
  } else {
    RegSize = 8;
    NumIntArgRegs = array_lengthof(Mips64IntRegs);
    ReservedArgArea = 0;
    IntArgRegs = Mips64IntRegs;
    ShadowRegs = Mips64DPRegs;
    FixedFn = CC_MipsN;
    VarFn = CC_MipsN_VarArg;
  }

  if (CallConv == CallingConv::Fast) {
    assert(!IsVarArg);
    UseRegsForByval = false;
    ReservedArgArea = 0;
    FixedFn = VarFn = CC_Mips_FastCC;
  }

  // Pre-allocate reserved argument area.
  CCInfo.AllocateStack(ReservedArgArea, 1);
}

void MipsTargetLowering::MipsCC::
analyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Args) {
  unsigned NumOpnds = Args.size();

  for (unsigned I = 0; I != NumOpnds; ++I) {
    MVT ArgVT = Args[I].VT;
    ISD::ArgFlagsTy ArgFlags = Args[I].Flags;
    bool R;

    if (ArgFlags.isByVal()) {
      handleByValArg(I, ArgVT, ArgVT, CCValAssign::Full, ArgFlags);
      continue;
    }

    if (Args[I].IsFixed)
      R = FixedFn(I, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, CCInfo);
    else
      R = VarFn(I, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, CCInfo);

    if (R) {
#ifndef NDEBUG
      dbgs() << "Call operand #" << I << " has unhandled type "
             << EVT(ArgVT).getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

void MipsTargetLowering::MipsCC::
analyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Args) {
  unsigned NumArgs = Args.size();

  for (unsigned I = 0; I != NumArgs; ++I) {
    MVT ArgVT = Args[I].VT;
    ISD::ArgFlagsTy ArgFlags = Args[I].Flags;

    if (ArgFlags.isByVal()) {
      handleByValArg(I, ArgVT, ArgVT, CCValAssign::Full, ArgFlags);
      continue;
    }

    if (!FixedFn(I, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, CCInfo))
      continue;

#ifndef NDEBUG
    dbgs() << "Formal Arg #" << I << " has unhandled type "
           << EVT(ArgVT).getEVTString();
#endif
    llvm_unreachable(0);
  }
}

void
MipsTargetLowering::MipsCC::handleByValArg(unsigned ValNo, MVT ValVT,
                                           MVT LocVT,
                                           CCValAssign::LocInfo LocInfo,
                                           ISD::ArgFlagsTy ArgFlags) {
  assert(ArgFlags.getByValSize() && "Byval argument's size shouldn't be 0.");

  struct ByValArgInfo ByVal;
  unsigned ByValSize = RoundUpToAlignment(ArgFlags.getByValSize(), RegSize);
  unsigned Align = std::min(std::max(ArgFlags.getByValAlign(), RegSize),
                            RegSize * 2);

  if (UseRegsForByval)
    allocateRegs(ByVal, ByValSize, Align);

  // Allocate space on caller's stack.
  ByVal.Address = CCInfo.AllocateStack(ByValSize - RegSize * ByVal.NumRegs,
                                       Align);
  CCInfo.addLoc(CCValAssign::getMem(ValNo, ValVT, ByVal.Address, LocVT,
                                    LocInfo));
  ByValArgs.push_back(ByVal);
}

void MipsTargetLowering::MipsCC::allocateRegs(ByValArgInfo &ByVal,
                                              unsigned ByValSize,
                                              unsigned Align) {
  assert(!(ByValSize % RegSize) && !(Align % RegSize) &&
         "Byval argument's size and alignment should be a multiple of"
         "RegSize.");

  ByVal.FirstIdx = CCInfo.getFirstUnallocated(IntArgRegs, NumIntArgRegs);

  // If Align > RegSize, the first arg register must be even.
  if ((Align > RegSize) && (ByVal.FirstIdx % 2)) {
    CCInfo.AllocateReg(IntArgRegs[ByVal.FirstIdx], ShadowRegs[ByVal.FirstIdx]);
    ++ByVal.FirstIdx;
  }

  // Mark the registers allocated.
  for (unsigned I = ByVal.FirstIdx; ByValSize && (I < NumIntArgRegs);
       ByValSize -= RegSize, ++I, ++ByVal.NumRegs)
    CCInfo.AllocateReg(IntArgRegs[I], ShadowRegs[I]);
}

void MipsTargetLowering::
copyByValRegs(SDValue Chain, DebugLoc DL, std::vector<SDValue> &OutChains,
              SelectionDAG &DAG, const ISD::ArgFlagsTy &Flags,
              SmallVectorImpl<SDValue> &InVals, const Argument *FuncArg,
              const MipsCC &CC, const ByValArgInfo &ByVal) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned RegAreaSize = ByVal.NumRegs * CC.regSize();
  unsigned FrameObjSize = std::max(Flags.getByValSize(), RegAreaSize);
  int FrameObjOffset;

  if (RegAreaSize)
    FrameObjOffset = (int)CC.reservedArgArea() -
      (int)((CC.numIntArgRegs() - ByVal.FirstIdx) * CC.regSize());
  else
    FrameObjOffset = ByVal.Address;

  // Create frame object.
  EVT PtrTy = getPointerTy();
  int FI = MFI->CreateFixedObject(FrameObjSize, FrameObjOffset, true);
  SDValue FIN = DAG.getFrameIndex(FI, PtrTy);
  InVals.push_back(FIN);

  if (!ByVal.NumRegs)
    return;

  // Copy arg registers.
  EVT RegTy = MVT::getIntegerVT(CC.regSize() * 8);
  const TargetRegisterClass *RC = getRegClassFor(RegTy);

  for (unsigned I = 0; I < ByVal.NumRegs; ++I) {
    unsigned ArgReg = CC.intArgRegs()[ByVal.FirstIdx + I];
    unsigned VReg = AddLiveIn(MF, ArgReg, RC);
    unsigned Offset = I * CC.regSize();
    SDValue StorePtr = DAG.getNode(ISD::ADD, DL, PtrTy, FIN,
                                   DAG.getConstant(Offset, PtrTy));
    SDValue Store = DAG.getStore(Chain, DL, DAG.getRegister(VReg, RegTy),
                                 StorePtr, MachinePointerInfo(FuncArg, Offset),
                                 false, false, 0);
    OutChains.push_back(Store);
  }
}

// Copy byVal arg to registers and stack.
void MipsTargetLowering::
passByValArg(SDValue Chain, DebugLoc DL,
             SmallVector<std::pair<unsigned, SDValue>, 16> &RegsToPass,
             SmallVector<SDValue, 8> &MemOpChains, SDValue StackPtr,
             MachineFrameInfo *MFI, SelectionDAG &DAG, SDValue Arg,
             const MipsCC &CC, const ByValArgInfo &ByVal,
             const ISD::ArgFlagsTy &Flags, bool isLittle) const {
  unsigned ByValSize = Flags.getByValSize();
  unsigned Offset = 0; // Offset in # of bytes from the beginning of struct.
  unsigned RegSize = CC.regSize();
  unsigned Alignment = std::min(Flags.getByValAlign(), RegSize);
  EVT PtrTy = getPointerTy(), RegTy = MVT::getIntegerVT(RegSize * 8);

  if (ByVal.NumRegs) {
    const uint16_t *ArgRegs = CC.intArgRegs();
    bool LeftoverBytes = (ByVal.NumRegs * RegSize > ByValSize);
    unsigned I = 0;

    // Copy words to registers.
    for (; I < ByVal.NumRegs - LeftoverBytes; ++I, Offset += RegSize) {
      SDValue LoadPtr = DAG.getNode(ISD::ADD, DL, PtrTy, Arg,
                                    DAG.getConstant(Offset, PtrTy));
      SDValue LoadVal = DAG.getLoad(RegTy, DL, Chain, LoadPtr,
                                    MachinePointerInfo(), false, false, false,
                                    Alignment);
      MemOpChains.push_back(LoadVal.getValue(1));
      unsigned ArgReg = ArgRegs[ByVal.FirstIdx + I];
      RegsToPass.push_back(std::make_pair(ArgReg, LoadVal));
    }

    // Return if the struct has been fully copied.
    if (ByValSize == Offset)
      return;

    // Copy the remainder of the byval argument with sub-word loads and shifts.
    if (LeftoverBytes) {
      assert((ByValSize > Offset) && (ByValSize < Offset + RegSize) &&
             "Size of the remainder should be smaller than RegSize.");
      SDValue Val;

      for (unsigned LoadSize = RegSize / 2, TotalSizeLoaded = 0;
           Offset < ByValSize; LoadSize /= 2) {
        unsigned RemSize = ByValSize - Offset;

        if (RemSize < LoadSize)
          continue;

        // Load subword.
        SDValue LoadPtr = DAG.getNode(ISD::ADD, DL, PtrTy, Arg,
                                      DAG.getConstant(Offset, PtrTy));
        SDValue LoadVal =
          DAG.getExtLoad(ISD::ZEXTLOAD, DL, RegTy, Chain, LoadPtr,
                         MachinePointerInfo(), MVT::getIntegerVT(LoadSize * 8),
                         false, false, Alignment);
        MemOpChains.push_back(LoadVal.getValue(1));

        // Shift the loaded value.
        unsigned Shamt;

        if (isLittle)
          Shamt = TotalSizeLoaded;
        else
          Shamt = (RegSize - (TotalSizeLoaded + LoadSize)) * 8;

        SDValue Shift = DAG.getNode(ISD::SHL, DL, RegTy, LoadVal,
                                    DAG.getConstant(Shamt, MVT::i32));

        if (Val.getNode())
          Val = DAG.getNode(ISD::OR, DL, RegTy, Val, Shift);
        else
          Val = Shift;

        Offset += LoadSize;
        TotalSizeLoaded += LoadSize;
        Alignment = std::min(Alignment, LoadSize);
      }

      unsigned ArgReg = ArgRegs[ByVal.FirstIdx + I];
      RegsToPass.push_back(std::make_pair(ArgReg, Val));
      return;
    }
  }

  // Copy remainder of byval arg to it with memcpy.
  unsigned MemCpySize = ByValSize - Offset;
  SDValue Src = DAG.getNode(ISD::ADD, DL, PtrTy, Arg,
                            DAG.getConstant(Offset, PtrTy));
  SDValue Dst = DAG.getNode(ISD::ADD, DL, PtrTy, StackPtr,
                            DAG.getIntPtrConstant(ByVal.Address));
  Chain = DAG.getMemcpy(Chain, DL, Dst, Src,
                        DAG.getConstant(MemCpySize, PtrTy), Alignment,
                        /*isVolatile=*/false, /*AlwaysInline=*/false,
                        MachinePointerInfo(0), MachinePointerInfo(0));
  MemOpChains.push_back(Chain);
}

void
MipsTargetLowering::writeVarArgRegs(std::vector<SDValue> &OutChains,
                                    const MipsCC &CC, SDValue Chain,
                                    DebugLoc DL, SelectionDAG &DAG) const {
  unsigned NumRegs = CC.numIntArgRegs();
  const uint16_t *ArgRegs = CC.intArgRegs();
  const CCState &CCInfo = CC.getCCInfo();
  unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs, NumRegs);
  unsigned RegSize = CC.regSize();
  EVT RegTy = MVT::getIntegerVT(RegSize * 8);
  const TargetRegisterClass *RC = getRegClassFor(RegTy);
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  // Offset of the first variable argument from stack pointer.
  int VaArgOffset;

  if (NumRegs == Idx)
    VaArgOffset = RoundUpToAlignment(CCInfo.getNextStackOffset(), RegSize);
  else
    VaArgOffset =
      (int)CC.reservedArgArea() - (int)(RegSize * (NumRegs - Idx));

  // Record the frame index of the first variable argument
  // which is a value necessary to VASTART.
  int FI = MFI->CreateFixedObject(RegSize, VaArgOffset, true);
  MipsFI->setVarArgsFrameIndex(FI);

  // Copy the integer registers that have not been used for argument passing
  // to the argument register save area. For O32, the save area is allocated
  // in the caller's stack frame, while for N32/64, it is allocated in the
  // callee's stack frame.
  for (unsigned I = Idx; I < NumRegs; ++I, VaArgOffset += RegSize) {
    unsigned Reg = AddLiveIn(MF, ArgRegs[I], RC);
    SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegTy);
    FI = MFI->CreateFixedObject(RegSize, VaArgOffset, true);
    SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy());
    SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                 MachinePointerInfo(), false, false, 0);
    cast<StoreSDNode>(Store.getNode())->getMemOperand()->setValue(0);
    OutChains.push_back(Store);
  }
}
