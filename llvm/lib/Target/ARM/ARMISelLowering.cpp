//===-- ARMISelLowering.cpp - ARM DAG Lowering Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that ARM uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMConstantPoolValue.h"
#include "ARMISelLowering.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/Intrinsics.h"
#include "llvm/GlobalValue.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

ARMTargetLowering::ARMTargetLowering(TargetMachine &TM)
    : TargetLowering(TM), ARMPCLabelIndex(0) {
  Subtarget = &TM.getSubtarget<ARMSubtarget>();

  if (Subtarget->isTargetDarwin()) {
    // Don't have these.
    setLibcallName(RTLIB::UINTTOFP_I64_F32, NULL);
    setLibcallName(RTLIB::UINTTOFP_I64_F64, NULL);

    // Uses VFP for Thumb libfuncs if available.
    if (Subtarget->isThumb() && Subtarget->hasVFP2()) {
      // Single-precision floating-point arithmetic.
      setLibcallName(RTLIB::ADD_F32, "__addsf3vfp");
      setLibcallName(RTLIB::SUB_F32, "__subsf3vfp");
      setLibcallName(RTLIB::MUL_F32, "__mulsf3vfp");
      setLibcallName(RTLIB::DIV_F32, "__divsf3vfp");

      // Double-precision floating-point arithmetic.
      setLibcallName(RTLIB::ADD_F64, "__adddf3vfp");
      setLibcallName(RTLIB::SUB_F64, "__subdf3vfp");
      setLibcallName(RTLIB::MUL_F64, "__muldf3vfp");
      setLibcallName(RTLIB::DIV_F64, "__divdf3vfp");

      // Single-precision comparisons.
      setLibcallName(RTLIB::OEQ_F32, "__eqsf2vfp");
      setLibcallName(RTLIB::UNE_F32, "__nesf2vfp");
      setLibcallName(RTLIB::OLT_F32, "__ltsf2vfp");
      setLibcallName(RTLIB::OLE_F32, "__lesf2vfp");
      setLibcallName(RTLIB::OGE_F32, "__gesf2vfp");
      setLibcallName(RTLIB::OGT_F32, "__gtsf2vfp");
      setLibcallName(RTLIB::UO_F32,  "__unordsf2vfp");
      setLibcallName(RTLIB::O_F32,   "__unordsf2vfp");

      setCmpLibcallCC(RTLIB::OEQ_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::UNE_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OLT_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OLE_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OGE_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OGT_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::UO_F32,  ISD::SETNE);
      setCmpLibcallCC(RTLIB::O_F32,   ISD::SETEQ);

      // Double-precision comparisons.
      setLibcallName(RTLIB::OEQ_F64, "__eqdf2vfp");
      setLibcallName(RTLIB::UNE_F64, "__nedf2vfp");
      setLibcallName(RTLIB::OLT_F64, "__ltdf2vfp");
      setLibcallName(RTLIB::OLE_F64, "__ledf2vfp");
      setLibcallName(RTLIB::OGE_F64, "__gedf2vfp");
      setLibcallName(RTLIB::OGT_F64, "__gtdf2vfp");
      setLibcallName(RTLIB::UO_F64,  "__unorddf2vfp");
      setLibcallName(RTLIB::O_F64,   "__unorddf2vfp");

      setCmpLibcallCC(RTLIB::OEQ_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::UNE_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OLT_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OLE_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OGE_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OGT_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::UO_F64,  ISD::SETNE);
      setCmpLibcallCC(RTLIB::O_F64,   ISD::SETEQ);

      // Floating-point to integer conversions.
      // i64 conversions are done via library routines even when generating VFP
      // instructions, so use the same ones.
      setLibcallName(RTLIB::FPTOSINT_F64_I32, "__fixdfsivfp");
      setLibcallName(RTLIB::FPTOUINT_F64_I32, "__fixunsdfsivfp");
      setLibcallName(RTLIB::FPTOSINT_F32_I32, "__fixsfsivfp");
      setLibcallName(RTLIB::FPTOUINT_F32_I32, "__fixunssfsivfp");

      // Conversions between floating types.
      setLibcallName(RTLIB::FPROUND_F64_F32, "__truncdfsf2vfp");
      setLibcallName(RTLIB::FPEXT_F32_F64,   "__extendsfdf2vfp");

      // Integer to floating-point conversions.
      // i64 conversions are done via library routines even when generating VFP
      // instructions, so use the same ones.
      // FIXME: There appears to be some naming inconsistency in ARM libgcc: e.g.
      // __floatunsidf vs. __floatunssidfvfp.
      setLibcallName(RTLIB::SINTTOFP_I32_F64, "__floatsidfvfp");
      setLibcallName(RTLIB::UINTTOFP_I32_F64, "__floatunssidfvfp");
      setLibcallName(RTLIB::SINTTOFP_I32_F32, "__floatsisfvfp");
      setLibcallName(RTLIB::UINTTOFP_I32_F32, "__floatunssisfvfp");
    }
  }

  addRegisterClass(MVT::i32, ARM::GPRRegisterClass);
  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb()) {
    addRegisterClass(MVT::f32, ARM::SPRRegisterClass);
    addRegisterClass(MVT::f64, ARM::DPRRegisterClass);
    
    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  }
  computeRegisterProperties();

  // ARM does not have f32 extending load.
  setLoadXAction(ISD::EXTLOAD, MVT::f32, Expand);

  // ARM does not have i1 sign extending load.
  setLoadXAction(ISD::SEXTLOAD, MVT::i1, Promote);

  // ARM supports all 4 flavors of integer indexed load / store.
  for (unsigned im = (unsigned)ISD::PRE_INC;
       im != (unsigned)ISD::LAST_INDEXED_MODE; ++im) {
    setIndexedLoadAction(im,  MVT::i1,  Legal);
    setIndexedLoadAction(im,  MVT::i8,  Legal);
    setIndexedLoadAction(im,  MVT::i16, Legal);
    setIndexedLoadAction(im,  MVT::i32, Legal);
    setIndexedStoreAction(im, MVT::i1,  Legal);
    setIndexedStoreAction(im, MVT::i8,  Legal);
    setIndexedStoreAction(im, MVT::i16, Legal);
    setIndexedStoreAction(im, MVT::i32, Legal);
  }

  // i64 operation support.
  if (Subtarget->isThumb()) {
    setOperationAction(ISD::MUL,     MVT::i64, Expand);
    setOperationAction(ISD::MULHU,   MVT::i32, Expand);
    setOperationAction(ISD::MULHS,   MVT::i32, Expand);
    setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
    setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  } else {
    setOperationAction(ISD::MUL,     MVT::i64, Expand);
    setOperationAction(ISD::MULHU,   MVT::i32, Expand);
    if (!Subtarget->hasV6Ops())
      setOperationAction(ISD::MULHS, MVT::i32, Expand);
  }
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL,       MVT::i64, Custom);
  setOperationAction(ISD::SRA,       MVT::i64, Custom);

  // ARM does not have ROTL.
  setOperationAction(ISD::ROTL,  MVT::i32, Expand);
  setOperationAction(ISD::CTTZ , MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  if (!Subtarget->hasV5TOps() || Subtarget->isThumb())
    setOperationAction(ISD::CTLZ, MVT::i32, Expand);

  // Only ARMv6 has BSWAP.
  if (!Subtarget->hasV6Ops())
    setOperationAction(ISD::BSWAP, MVT::i32, Expand);

  // These are expanded into libcalls.
  setOperationAction(ISD::SDIV,  MVT::i32, Expand);
  setOperationAction(ISD::UDIV,  MVT::i32, Expand);
  setOperationAction(ISD::SREM,  MVT::i32, Expand);
  setOperationAction(ISD::UREM,  MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);
  
  // Support label based line numbers.
  setOperationAction(ISD::DBG_STOPPOINT, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);

  setOperationAction(ISD::RET,           MVT::Other, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32,   Custom);
  setOperationAction(ISD::GLOBAL_OFFSET_TABLE, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);

  // Use the default implementation.
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE,          MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE,       MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Expand);
  setOperationAction(ISD::MEMBARRIER        , MVT::Other, Expand);

  if (!Subtarget->hasV6Ops()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8,  Expand);
  }
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb())
    // Turn f64->i64 into FMRRD iff target supports vfp2.
    setOperationAction(ISD::BIT_CONVERT, MVT::i64, Custom);

  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  setOperationAction(ISD::SETCC    , MVT::i32, Expand);
  setOperationAction(ISD::SETCC    , MVT::f32, Expand);
  setOperationAction(ISD::SETCC    , MVT::f64, Expand);
  setOperationAction(ISD::SELECT   , MVT::i32, Expand);
  setOperationAction(ISD::SELECT   , MVT::f32, Expand);
  setOperationAction(ISD::SELECT   , MVT::f64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);

  setOperationAction(ISD::BRCOND   , MVT::Other, Expand);
  setOperationAction(ISD::BR_CC    , MVT::i32,   Custom);
  setOperationAction(ISD::BR_CC    , MVT::f32,   Custom);
  setOperationAction(ISD::BR_CC    , MVT::f64,   Custom);
  setOperationAction(ISD::BR_JT    , MVT::Other, Custom);

  // We don't support sin/cos/fmod/copysign/pow
  setOperationAction(ISD::FSIN     , MVT::f64, Expand);
  setOperationAction(ISD::FSIN     , MVT::f32, Expand);
  setOperationAction(ISD::FCOS     , MVT::f32, Expand);
  setOperationAction(ISD::FCOS     , MVT::f64, Expand);
  setOperationAction(ISD::FREM     , MVT::f64, Expand);
  setOperationAction(ISD::FREM     , MVT::f32, Expand);
  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb()) {
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);
  }
  setOperationAction(ISD::FPOW     , MVT::f64, Expand);
  setOperationAction(ISD::FPOW     , MVT::f32, Expand);
  
  // int <-> fp are custom expanded into bit_convert + ARMISD ops.
  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb()) {
    setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
    setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  }

  // We have target-specific dag combine patterns for the following nodes:
  // ARMISD::FMRRD  - No need to call setTargetDAGCombine
  
  setStackPointerRegisterToSaveRestore(ARM::SP);
  setSchedulingPreference(SchedulingForRegPressure);
  setIfCvtBlockSizeLimit(Subtarget->isThumb() ? 0 : 10);
  setIfCvtDupBlockSizeLimit(Subtarget->isThumb() ? 0 : 2);

  maxStoresPerMemcpy = 1;   //// temporary - rewrite interface to use type
}


const char *ARMTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case ARMISD::Wrapper:       return "ARMISD::Wrapper";
  case ARMISD::WrapperJT:     return "ARMISD::WrapperJT";
  case ARMISD::CALL:          return "ARMISD::CALL";
  case ARMISD::CALL_PRED:     return "ARMISD::CALL_PRED";
  case ARMISD::CALL_NOLINK:   return "ARMISD::CALL_NOLINK";
  case ARMISD::tCALL:         return "ARMISD::tCALL";
  case ARMISD::BRCOND:        return "ARMISD::BRCOND";
  case ARMISD::BR_JT:         return "ARMISD::BR_JT";
  case ARMISD::RET_FLAG:      return "ARMISD::RET_FLAG";
  case ARMISD::PIC_ADD:       return "ARMISD::PIC_ADD";
  case ARMISD::CMP:           return "ARMISD::CMP";
  case ARMISD::CMPNZ:         return "ARMISD::CMPNZ";
  case ARMISD::CMPFP:         return "ARMISD::CMPFP";
  case ARMISD::CMPFPw0:       return "ARMISD::CMPFPw0";
  case ARMISD::FMSTAT:        return "ARMISD::FMSTAT";
  case ARMISD::CMOV:          return "ARMISD::CMOV";
  case ARMISD::CNEG:          return "ARMISD::CNEG";
    
  case ARMISD::FTOSI:         return "ARMISD::FTOSI";
  case ARMISD::FTOUI:         return "ARMISD::FTOUI";
  case ARMISD::SITOF:         return "ARMISD::SITOF";
  case ARMISD::UITOF:         return "ARMISD::UITOF";

  case ARMISD::SRL_FLAG:      return "ARMISD::SRL_FLAG";
  case ARMISD::SRA_FLAG:      return "ARMISD::SRA_FLAG";
  case ARMISD::RRX:           return "ARMISD::RRX";
      
  case ARMISD::FMRRD:         return "ARMISD::FMRRD";
  case ARMISD::FMDRR:         return "ARMISD::FMDRR";

  case ARMISD::THREAD_POINTER:return "ARMISD::THREAD_POINTER";
  }
}

//===----------------------------------------------------------------------===//
// Lowering Code
//===----------------------------------------------------------------------===//


/// IntCCToARMCC - Convert a DAG integer condition code to an ARM CC
static ARMCC::CondCodes IntCCToARMCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition code!");
  case ISD::SETNE:  return ARMCC::NE;
  case ISD::SETEQ:  return ARMCC::EQ;
  case ISD::SETGT:  return ARMCC::GT;
  case ISD::SETGE:  return ARMCC::GE;
  case ISD::SETLT:  return ARMCC::LT;
  case ISD::SETLE:  return ARMCC::LE;
  case ISD::SETUGT: return ARMCC::HI;
  case ISD::SETUGE: return ARMCC::HS;
  case ISD::SETULT: return ARMCC::LO;
  case ISD::SETULE: return ARMCC::LS;
  }
}

/// FPCCToARMCC - Convert a DAG fp condition code to an ARM CC. It
/// returns true if the operands should be inverted to form the proper
/// comparison.
static bool FPCCToARMCC(ISD::CondCode CC, ARMCC::CondCodes &CondCode,
                        ARMCC::CondCodes &CondCode2) {
  bool Invert = false;
  CondCode2 = ARMCC::AL;
  switch (CC) {
  default: assert(0 && "Unknown FP condition!");
  case ISD::SETEQ:
  case ISD::SETOEQ: CondCode = ARMCC::EQ; break;
  case ISD::SETGT:
  case ISD::SETOGT: CondCode = ARMCC::GT; break;
  case ISD::SETGE:
  case ISD::SETOGE: CondCode = ARMCC::GE; break;
  case ISD::SETOLT: CondCode = ARMCC::MI; break;
  case ISD::SETOLE: CondCode = ARMCC::GT; Invert = true; break;
  case ISD::SETONE: CondCode = ARMCC::MI; CondCode2 = ARMCC::GT; break;
  case ISD::SETO:   CondCode = ARMCC::VC; break;
  case ISD::SETUO:  CondCode = ARMCC::VS; break;
  case ISD::SETUEQ: CondCode = ARMCC::EQ; CondCode2 = ARMCC::VS; break;
  case ISD::SETUGT: CondCode = ARMCC::HI; break;
  case ISD::SETUGE: CondCode = ARMCC::PL; break;
  case ISD::SETLT:
  case ISD::SETULT: CondCode = ARMCC::LT; break;
  case ISD::SETLE:
  case ISD::SETULE: CondCode = ARMCC::LE; break;
  case ISD::SETNE:
  case ISD::SETUNE: CondCode = ARMCC::NE; break;
  }
  return Invert;
}

static void
HowToPassArgument(MVT ObjectVT, unsigned NumGPRs,
                  unsigned StackOffset, unsigned &NeededGPRs,
                  unsigned &NeededStackSize, unsigned &GPRPad,
                  unsigned &StackPad, ISD::ArgFlagsTy Flags) {
  NeededStackSize = 0;
  NeededGPRs = 0;
  StackPad = 0;
  GPRPad = 0;
  unsigned align = Flags.getOrigAlign();
  GPRPad = NumGPRs % ((align + 3)/4);
  StackPad = StackOffset % align;
  unsigned firstGPR = NumGPRs + GPRPad;
  switch (ObjectVT.getSimpleVT()) {
  default: assert(0 && "Unhandled argument type!");
  case MVT::i32:
  case MVT::f32:
    if (firstGPR < 4)
      NeededGPRs = 1;
    else
      NeededStackSize = 4;
    break;
  case MVT::i64:
  case MVT::f64:
    if (firstGPR < 3)
      NeededGPRs = 2;
    else if (firstGPR == 3) {
      NeededGPRs = 1;
      NeededStackSize = 4;
    } else
      NeededStackSize = 8;
  }
}

/// LowerCALL - Lowering a ISD::CALL node into a callseq_start <-
/// ARMISD:CALL <- callseq_end chain. Also add input and output parameter
/// nodes.
SDValue ARMTargetLowering::LowerCALL(SDValue Op, SelectionDAG &DAG) {
  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  MVT RetVT = TheCall->getRetValType(0);
  SDValue Chain    = TheCall->getChain();
  unsigned CallConv  = TheCall->getCallingConv();
  assert((CallConv == CallingConv::C ||
          CallConv == CallingConv::Fast) && "unknown calling convention");
  SDValue Callee   = TheCall->getCallee();
  unsigned NumOps    = TheCall->getNumArgs();
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  unsigned NumGPRs = 0;     // GPRs used for parameter passing.

  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;

  // Add up all the space actually used.
  for (unsigned i = 0; i < NumOps; ++i) {
    unsigned ObjSize;
    unsigned ObjGPRs;
    unsigned StackPad;
    unsigned GPRPad;
    MVT ObjectVT = TheCall->getArg(i).getValueType();
    ISD::ArgFlagsTy Flags = TheCall->getArgFlags(i);
    HowToPassArgument(ObjectVT, NumGPRs, NumBytes, ObjGPRs, ObjSize,
                      GPRPad, StackPad, Flags);
    NumBytes += ObjSize + StackPad;
    NumGPRs += ObjGPRs + GPRPad;
  }

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));

  SDValue StackPtr = DAG.getRegister(ARM::SP, MVT::i32);

  static const unsigned GPRArgRegs[] = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3
  };

  NumGPRs = 0;
  std::vector<std::pair<unsigned, SDValue> > RegsToPass;
  std::vector<SDValue> MemOpChains;
  for (unsigned i = 0; i != NumOps; ++i) {
    SDValue Arg = TheCall->getArg(i);
    ISD::ArgFlagsTy Flags = TheCall->getArgFlags(i);
    MVT ArgVT = Arg.getValueType();

    unsigned ObjSize;
    unsigned ObjGPRs;
    unsigned GPRPad;
    unsigned StackPad;
    HowToPassArgument(ArgVT, NumGPRs, ArgOffset, ObjGPRs,
                      ObjSize, GPRPad, StackPad, Flags);
    NumGPRs += GPRPad;
    ArgOffset += StackPad;
    if (ObjGPRs > 0) {
      switch (ArgVT.getSimpleVT()) {
      default: assert(0 && "Unexpected ValueType for argument!");
      case MVT::i32:
        RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs], Arg));
        break;
      case MVT::f32:
        RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs],
                                 DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Arg)));
        break;
      case MVT::i64: {
        SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Arg,
                                   DAG.getConstant(0, getPointerTy()));
        SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Arg,
                                   DAG.getConstant(1, getPointerTy()));
        RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs], Lo));
        if (ObjGPRs == 2)
          RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs+1], Hi));
        else {
          SDValue PtrOff= DAG.getConstant(ArgOffset, StackPtr.getValueType());
          PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
          MemOpChains.push_back(DAG.getStore(Chain, Hi, PtrOff, NULL, 0));
        }
        break;
      }
      case MVT::f64: {
        SDValue Cvt = DAG.getNode(ARMISD::FMRRD,
                                    DAG.getVTList(MVT::i32, MVT::i32),
                                    &Arg, 1);
        RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs], Cvt));
        if (ObjGPRs == 2)
          RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs+1],
                                              Cvt.getValue(1)));
        else {
          SDValue PtrOff= DAG.getConstant(ArgOffset, StackPtr.getValueType());
          PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
          MemOpChains.push_back(DAG.getStore(Chain, Cvt.getValue(1), PtrOff,
                                             NULL, 0));
        }
        break;
      }
      }
    } else {
      assert(ObjSize != 0);
      SDValue PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
    }

    NumGPRs += ObjGPRs;
    ArgOffset += ObjSize;
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                             InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  bool isDirect = false;
  bool isARMFunc = false;
  bool isLocalARMFunc = false;
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    GlobalValue *GV = G->getGlobal();
    isDirect = true;
    bool isExt = (GV->isDeclaration() || GV->hasWeakLinkage() ||
                  GV->hasLinkOnceLinkage());
    bool isStub = (isExt && Subtarget->isTargetDarwin()) &&
                   getTargetMachine().getRelocationModel() != Reloc::Static;
    isARMFunc = !Subtarget->isThumb() || isStub;
    // ARM call to a local ARM function is predicable.
    isLocalARMFunc = !Subtarget->isThumb() && !isExt;
    // tBX takes a register source operand.
    if (isARMFunc && Subtarget->isThumb() && !Subtarget->hasV5TOps()) {
      ARMConstantPoolValue *CPV = new ARMConstantPoolValue(GV, ARMPCLabelIndex,
                                                           ARMCP::CPStub, 4);
      SDValue CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 2);
      CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), DAG.getEntryNode(), CPAddr, NULL, 0); 
      SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
      Callee = DAG.getNode(ARMISD::PIC_ADD, getPointerTy(), Callee, PICLabel);
   } else
      Callee = DAG.getTargetGlobalAddress(GV, getPointerTy());
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    isDirect = true;
    bool isStub = Subtarget->isTargetDarwin() &&
                  getTargetMachine().getRelocationModel() != Reloc::Static;
    isARMFunc = !Subtarget->isThumb() || isStub;
    // tBX takes a register source operand.
    const char *Sym = S->getSymbol();
    if (isARMFunc && Subtarget->isThumb() && !Subtarget->hasV5TOps()) {
      ARMConstantPoolValue *CPV = new ARMConstantPoolValue(Sym, ARMPCLabelIndex,
                                                           ARMCP::CPStub, 4);
      SDValue CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 2);
      CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), DAG.getEntryNode(), CPAddr, NULL, 0); 
      SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
      Callee = DAG.getNode(ARMISD::PIC_ADD, getPointerTy(), Callee, PICLabel);
    } else
      Callee = DAG.getTargetExternalSymbol(Sym, getPointerTy());
  }

  // FIXME: handle tail calls differently.
  unsigned CallOpc;
  if (Subtarget->isThumb()) {
    if (!Subtarget->hasV5TOps() && (!isDirect || isARMFunc))
      CallOpc = ARMISD::CALL_NOLINK;
    else
      CallOpc = isARMFunc ? ARMISD::CALL : ARMISD::tCALL;
  } else {
    CallOpc = (isDirect || Subtarget->hasV5TOps())
      ? (isLocalARMFunc ? ARMISD::CALL_PRED : ARMISD::CALL)
      : ARMISD::CALL_NOLINK;
  }
  if (CallOpc == ARMISD::CALL_NOLINK && !Subtarget->isThumb()) {
    // implicit def LR - LR mustn't be allocated as GRP:$dst of CALL_NOLINK
    Chain = DAG.getCopyToReg(Chain, ARM::LR,
                             DAG.getNode(ISD::UNDEF, MVT::i32), InFlag);
    InFlag = Chain.getValue(1);
  }

  std::vector<SDValue> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.getNode())
    Ops.push_back(InFlag);
  // Returns a chain and a flag for retval copy to use.
  Chain = DAG.getNode(CallOpc, DAG.getVTList(MVT::Other, MVT::Flag),
                      &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  if (RetVT != MVT::Other)
    InFlag = Chain.getValue(1);

  std::vector<SDValue> ResultVals;

  // If the call has results, copy the values out of the ret val registers.
  switch (RetVT.getSimpleVT()) {
  default: assert(0 && "Unexpected ret value!");
  case MVT::Other:
    break;
  case MVT::i32:
    Chain = DAG.getCopyFromReg(Chain, ARM::R0, MVT::i32, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    if (TheCall->getNumRetVals() > 1 &&
        TheCall->getRetValType(1) == MVT::i32) {
      // Returns a i64 value.
      Chain = DAG.getCopyFromReg(Chain, ARM::R1, MVT::i32,
                                 Chain.getValue(2)).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
    }
    break;
  case MVT::f32:
    Chain = DAG.getCopyFromReg(Chain, ARM::R0, MVT::i32, InFlag).getValue(1);
    ResultVals.push_back(DAG.getNode(ISD::BIT_CONVERT, MVT::f32,
                                     Chain.getValue(0)));
    break;
  case MVT::f64: {
    SDValue Lo = DAG.getCopyFromReg(Chain, ARM::R0, MVT::i32, InFlag);
    SDValue Hi = DAG.getCopyFromReg(Lo, ARM::R1, MVT::i32, Lo.getValue(2));
    ResultVals.push_back(DAG.getNode(ARMISD::FMDRR, MVT::f64, Lo, Hi));
    break;
  }
  }

  if (ResultVals.empty())
    return Chain;

  ResultVals.push_back(Chain);
  SDValue Res = DAG.getMergeValues(&ResultVals[0], ResultVals.size());
  return Res.getValue(Op.getResNo());
}

static SDValue LowerRET(SDValue Op, SelectionDAG &DAG) {
  SDValue Copy;
  SDValue Chain = Op.getOperand(0);
  switch(Op.getNumOperands()) {
  default:
    assert(0 && "Do not know how to return this many arguments!");
    abort();
  case 1: {
    SDValue LR = DAG.getRegister(ARM::LR, MVT::i32);
    return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Chain);
  }
  case 3:
    Op = Op.getOperand(1);
    if (Op.getValueType() == MVT::f32) {
      Op = DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Op);
    } else if (Op.getValueType() == MVT::f64) {
      // Legalize ret f64 -> ret 2 x i32.  We always have fmrrd if f64 is
      // available.
      Op = DAG.getNode(ARMISD::FMRRD, DAG.getVTList(MVT::i32, MVT::i32), &Op,1);
      SDValue Sign = DAG.getConstant(0, MVT::i32);
      return DAG.getNode(ISD::RET, MVT::Other, Chain, Op, Sign, 
                         Op.getValue(1), Sign);
    }
    Copy = DAG.getCopyToReg(Chain, ARM::R0, Op, SDValue());
    if (DAG.getMachineFunction().getRegInfo().liveout_empty())
      DAG.getMachineFunction().getRegInfo().addLiveOut(ARM::R0);
    break;
  case 5:
    Copy = DAG.getCopyToReg(Chain, ARM::R1, Op.getOperand(3), SDValue());
    Copy = DAG.getCopyToReg(Copy, ARM::R0, Op.getOperand(1), Copy.getValue(1));
    // If we haven't noted the R0+R1 are live out, do so now.
    if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
      DAG.getMachineFunction().getRegInfo().addLiveOut(ARM::R0);
      DAG.getMachineFunction().getRegInfo().addLiveOut(ARM::R1);
    }
    break;
  case 9:  // i128 -> 4 regs
    Copy = DAG.getCopyToReg(Chain, ARM::R3, Op.getOperand(7), SDValue());
    Copy = DAG.getCopyToReg(Copy , ARM::R2, Op.getOperand(5), Copy.getValue(1));
    Copy = DAG.getCopyToReg(Copy , ARM::R1, Op.getOperand(3), Copy.getValue(1));
    Copy = DAG.getCopyToReg(Copy , ARM::R0, Op.getOperand(1), Copy.getValue(1));
    // If we haven't noted the R0+R1 are live out, do so now.
    if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
      DAG.getMachineFunction().getRegInfo().addLiveOut(ARM::R0);
      DAG.getMachineFunction().getRegInfo().addLiveOut(ARM::R1);
      DAG.getMachineFunction().getRegInfo().addLiveOut(ARM::R2);
      DAG.getMachineFunction().getRegInfo().addLiveOut(ARM::R3);
    }
    break;
      
  }

  //We must use RET_FLAG instead of BRIND because BRIND doesn't have a flag
  return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
}

// ConstantPool, JumpTable, GlobalAddress, and ExternalSymbol are lowered as 
// their target countpart wrapped in the ARMISD::Wrapper node. Suppose N is
// one of the above mentioned nodes. It has to be wrapped because otherwise
// Select(N) returns N. So the raw TargetGlobalAddress nodes, etc. can only
// be used to form addressing mode. These wrapped nodes will be selected
// into MOVi.
static SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) {
  MVT PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  SDValue Res;
  if (CP->isMachineConstantPoolEntry())
    Res = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
                                    CP->getAlignment());
  else
    Res = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
                                    CP->getAlignment());
  return DAG.getNode(ARMISD::Wrapper, MVT::i32, Res);
}

// Lower ISD::GlobalTLSAddress using the "general dynamic" model
SDValue
ARMTargetLowering::LowerToTLSGeneralDynamicModel(GlobalAddressSDNode *GA,
                                                 SelectionDAG &DAG) {
  MVT PtrVT = getPointerTy();
  unsigned char PCAdj = Subtarget->isThumb() ? 4 : 8;
  ARMConstantPoolValue *CPV =
    new ARMConstantPoolValue(GA->getGlobal(), ARMPCLabelIndex, ARMCP::CPValue,
                             PCAdj, "tlsgd", true);
  SDValue Argument = DAG.getTargetConstantPool(CPV, PtrVT, 2);
  Argument = DAG.getNode(ARMISD::Wrapper, MVT::i32, Argument);
  Argument = DAG.getLoad(PtrVT, DAG.getEntryNode(), Argument, NULL, 0);
  SDValue Chain = Argument.getValue(1);

  SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
  Argument = DAG.getNode(ARMISD::PIC_ADD, PtrVT, Argument, PICLabel);

  // call __tls_get_addr.
  ArgListTy Args;
  ArgListEntry Entry;
  Entry.Node = Argument;
  Entry.Ty = (const Type *) Type::Int32Ty;
  Args.push_back(Entry);
  std::pair<SDValue, SDValue> CallResult =
    LowerCallTo(Chain, (const Type *) Type::Int32Ty, false, false, false, false,
                CallingConv::C, false,
                DAG.getExternalSymbol("__tls_get_addr", PtrVT), Args, DAG);
  return CallResult.first;
}

// Lower ISD::GlobalTLSAddress using the "initial exec" or
// "local exec" model.
SDValue
ARMTargetLowering::LowerToTLSExecModels(GlobalAddressSDNode *GA,
                                            SelectionDAG &DAG) {
  GlobalValue *GV = GA->getGlobal();
  SDValue Offset;
  SDValue Chain = DAG.getEntryNode();
  MVT PtrVT = getPointerTy();
  // Get the Thread Pointer
  SDValue ThreadPointer = DAG.getNode(ARMISD::THREAD_POINTER, PtrVT);

  if (GV->isDeclaration()){
    // initial exec model
    unsigned char PCAdj = Subtarget->isThumb() ? 4 : 8;
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GA->getGlobal(), ARMPCLabelIndex, ARMCP::CPValue,
                               PCAdj, "gottpoff", true);
    Offset = DAG.getTargetConstantPool(CPV, PtrVT, 2);
    Offset = DAG.getNode(ARMISD::Wrapper, MVT::i32, Offset);
    Offset = DAG.getLoad(PtrVT, Chain, Offset, NULL, 0);
    Chain = Offset.getValue(1);

    SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
    Offset = DAG.getNode(ARMISD::PIC_ADD, PtrVT, Offset, PICLabel);

    Offset = DAG.getLoad(PtrVT, Chain, Offset, NULL, 0);
  } else {
    // local exec model
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GV, ARMCP::CPValue, "tpoff");
    Offset = DAG.getTargetConstantPool(CPV, PtrVT, 2);
    Offset = DAG.getNode(ARMISD::Wrapper, MVT::i32, Offset);
    Offset = DAG.getLoad(PtrVT, Chain, Offset, NULL, 0);
  }

  // The address of the thread local variable is the add of the thread
  // pointer with the offset of the variable.
  return DAG.getNode(ISD::ADD, PtrVT, ThreadPointer, Offset);
}

SDValue
ARMTargetLowering::LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) {
  // TODO: implement the "local dynamic" model
  assert(Subtarget->isTargetELF() &&
         "TLS not implemented for non-ELF targets");
  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);
  // If the relocation model is PIC, use the "General Dynamic" TLS Model,
  // otherwise use the "Local Exec" TLS Model
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_)
    return LowerToTLSGeneralDynamicModel(GA, DAG);
  else
    return LowerToTLSExecModels(GA, DAG);
}

SDValue ARMTargetLowering::LowerGlobalAddressELF(SDValue Op,
                                                   SelectionDAG &DAG) {
  MVT PtrVT = getPointerTy();
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();
  if (RelocM == Reloc::PIC_) {
    bool UseGOTOFF = GV->hasInternalLinkage() || GV->hasHiddenVisibility();
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GV, ARMCP::CPValue, UseGOTOFF ? "GOTOFF":"GOT");
    SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 2);
    CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
    SDValue Result = DAG.getLoad(PtrVT, DAG.getEntryNode(), CPAddr, NULL, 0);
    SDValue Chain = Result.getValue(1);
    SDValue GOT = DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, PtrVT);
    Result = DAG.getNode(ISD::ADD, PtrVT, Result, GOT);
    if (!UseGOTOFF)
      Result = DAG.getLoad(PtrVT, Chain, Result, NULL, 0);
    return Result;
  } else {
    SDValue CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 2);
    CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
    return DAG.getLoad(PtrVT, DAG.getEntryNode(), CPAddr, NULL, 0);
  }
}

/// GVIsIndirectSymbol - true if the GV will be accessed via an indirect symbol
/// even in non-static mode.
static bool GVIsIndirectSymbol(GlobalValue *GV, Reloc::Model RelocM) {
  return RelocM != Reloc::Static &&
    (GV->hasWeakLinkage() || GV->hasLinkOnceLinkage() ||
     (GV->isDeclaration() && !GV->hasNotBeenReadFromBitcode()));
}

SDValue ARMTargetLowering::LowerGlobalAddressDarwin(SDValue Op,
                                                      SelectionDAG &DAG) {
  MVT PtrVT = getPointerTy();
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();
  bool IsIndirect = GVIsIndirectSymbol(GV, RelocM);
  SDValue CPAddr;
  if (RelocM == Reloc::Static)
    CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 2);
  else {
    unsigned PCAdj = (RelocM != Reloc::PIC_)
      ? 0 : (Subtarget->isThumb() ? 4 : 8);
    ARMCP::ARMCPKind Kind = IsIndirect ? ARMCP::CPNonLazyPtr
      : ARMCP::CPValue;
    ARMConstantPoolValue *CPV = new ARMConstantPoolValue(GV, ARMPCLabelIndex,
                                                         Kind, PCAdj);
    CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 2);
  }
  CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);

  SDValue Result = DAG.getLoad(PtrVT, DAG.getEntryNode(), CPAddr, NULL, 0);
  SDValue Chain = Result.getValue(1);

  if (RelocM == Reloc::PIC_) {
    SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
    Result = DAG.getNode(ARMISD::PIC_ADD, PtrVT, Result, PICLabel);
  }
  if (IsIndirect)
    Result = DAG.getLoad(PtrVT, Chain, Result, NULL, 0);

  return Result;
}

SDValue ARMTargetLowering::LowerGLOBAL_OFFSET_TABLE(SDValue Op,
                                                      SelectionDAG &DAG){
  assert(Subtarget->isTargetELF() &&
         "GLOBAL OFFSET TABLE not implemented for non-ELF targets");
  MVT PtrVT = getPointerTy();
  unsigned PCAdj = Subtarget->isThumb() ? 4 : 8;
  ARMConstantPoolValue *CPV = new ARMConstantPoolValue("_GLOBAL_OFFSET_TABLE_",
                                                       ARMPCLabelIndex,
                                                       ARMCP::CPValue, PCAdj);
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 2);
  CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
  SDValue Result = DAG.getLoad(PtrVT, DAG.getEntryNode(), CPAddr, NULL, 0);
  SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
  return DAG.getNode(ARMISD::PIC_ADD, PtrVT, Result, PICLabel);
}

static SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) {
  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  switch (IntNo) {
  default: return SDValue();    // Don't custom lower most intrinsics.
  case Intrinsic::arm_thread_pointer:
      return DAG.getNode(ARMISD::THREAD_POINTER, PtrVT);
  }
}

static SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG,
                              unsigned VarArgsFrameIndex) {
  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  SDValue FR = DAG.getFrameIndex(VarArgsFrameIndex, PtrVT);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), FR, Op.getOperand(1), SV, 0);
}

static SDValue LowerFORMAL_ARGUMENT(SDValue Op, SelectionDAG &DAG,
                                      unsigned ArgNo, unsigned &NumGPRs,
                                      unsigned &ArgOffset) {
  MachineFunction &MF = DAG.getMachineFunction();
  MVT ObjectVT = Op.getValue(ArgNo).getValueType();
  SDValue Root = Op.getOperand(0);
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  static const unsigned GPRArgRegs[] = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3
  };

  unsigned ObjSize;
  unsigned ObjGPRs;
  unsigned GPRPad;
  unsigned StackPad;
  ISD::ArgFlagsTy Flags =
    cast<ARG_FLAGSSDNode>(Op.getOperand(ArgNo + 3))->getArgFlags();
  HowToPassArgument(ObjectVT, NumGPRs, ArgOffset, ObjGPRs,
                    ObjSize, GPRPad, StackPad, Flags);
  NumGPRs += GPRPad;
  ArgOffset += StackPad;

  SDValue ArgValue;
  if (ObjGPRs == 1) {
    unsigned VReg = RegInfo.createVirtualRegister(&ARM::GPRRegClass);
    RegInfo.addLiveIn(GPRArgRegs[NumGPRs], VReg);
    ArgValue = DAG.getCopyFromReg(Root, VReg, MVT::i32);
    if (ObjectVT == MVT::f32)
      ArgValue = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, ArgValue);
  } else if (ObjGPRs == 2) {
    unsigned VReg = RegInfo.createVirtualRegister(&ARM::GPRRegClass);
    RegInfo.addLiveIn(GPRArgRegs[NumGPRs], VReg);
    ArgValue = DAG.getCopyFromReg(Root, VReg, MVT::i32);

    VReg = RegInfo.createVirtualRegister(&ARM::GPRRegClass);
    RegInfo.addLiveIn(GPRArgRegs[NumGPRs+1], VReg);
    SDValue ArgValue2 = DAG.getCopyFromReg(Root, VReg, MVT::i32);

    assert(ObjectVT != MVT::i64 && "i64 should already be lowered");
    ArgValue = DAG.getNode(ARMISD::FMDRR, MVT::f64, ArgValue, ArgValue2);
  }
  NumGPRs += ObjGPRs;

  if (ObjSize) {
    MachineFrameInfo *MFI = MF.getFrameInfo();
    int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    if (ObjGPRs == 0)
      ArgValue = DAG.getLoad(ObjectVT, Root, FIN, NULL, 0);
    else {
      SDValue ArgValue2 = DAG.getLoad(MVT::i32, Root, FIN, NULL, 0);
      assert(ObjectVT != MVT::i64 && "i64 should already be lowered");
      ArgValue = DAG.getNode(ARMISD::FMDRR, MVT::f64, ArgValue, ArgValue2);
    }

    ArgOffset += ObjSize;   // Move on to the next argument.
  }

  return ArgValue;
}

SDValue
ARMTargetLowering::LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG) {
  std::vector<SDValue> ArgValues;
  SDValue Root = Op.getOperand(0);
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  unsigned NumGPRs = 0;     // GPRs used for parameter passing.

  unsigned NumArgs = Op.getNode()->getNumValues()-1;
  for (unsigned ArgNo = 0; ArgNo < NumArgs; ++ArgNo)
    ArgValues.push_back(LowerFORMAL_ARGUMENT(Op, DAG, ArgNo,
                                             NumGPRs, ArgOffset));

  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue() != 0;
  if (isVarArg) {
    static const unsigned GPRArgRegs[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3
    };

    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    MachineFrameInfo *MFI = MF.getFrameInfo();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
    unsigned VARegSize = (4 - NumGPRs) * 4;
    unsigned VARegSaveSize = (VARegSize + Align - 1) & ~(Align - 1);
    if (VARegSaveSize) {
      // If this function is vararg, store any remaining integer argument regs
      // to their spots on the stack so that they may be loaded by deferencing
      // the result of va_next.
      AFI->setVarArgsRegSaveSize(VARegSaveSize);
      VarArgsFrameIndex = MFI->CreateFixedObject(VARegSaveSize, ArgOffset +
                                                 VARegSaveSize - VARegSize);
      SDValue FIN = DAG.getFrameIndex(VarArgsFrameIndex, getPointerTy());

      SmallVector<SDValue, 4> MemOps;
      for (; NumGPRs < 4; ++NumGPRs) {
        unsigned VReg = RegInfo.createVirtualRegister(&ARM::GPRRegClass);
        RegInfo.addLiveIn(GPRArgRegs[NumGPRs], VReg);
        SDValue Val = DAG.getCopyFromReg(Root, VReg, MVT::i32);
        SDValue Store = DAG.getStore(Val.getValue(1), Val, FIN, NULL, 0);
        MemOps.push_back(Store);
        FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                          DAG.getConstant(4, getPointerTy()));
      }
      if (!MemOps.empty())
        Root = DAG.getNode(ISD::TokenFactor, MVT::Other,
                           &MemOps[0], MemOps.size());
    } else
      // This will point to the next argument passed via stack.
      VarArgsFrameIndex = MFI->CreateFixedObject(4, ArgOffset);
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  return DAG.getMergeValues(Op.getNode()->getVTList(), &ArgValues[0],
                            ArgValues.size());
}

/// isFloatingPointZero - Return true if this is +0.0.
static bool isFloatingPointZero(SDValue Op) {
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op))
    return CFP->getValueAPF().isPosZero();
  else if (ISD::isEXTLoad(Op.getNode()) || ISD::isNON_EXTLoad(Op.getNode())) {
    // Maybe this has already been legalized into the constant pool?
    if (Op.getOperand(1).getOpcode() == ARMISD::Wrapper) {
      SDValue WrapperOp = Op.getOperand(1).getOperand(0);
      if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(WrapperOp))
        if (ConstantFP *CFP = dyn_cast<ConstantFP>(CP->getConstVal()))
          return CFP->getValueAPF().isPosZero();
    }
  }
  return false;
}

static bool isLegalCmpImmediate(unsigned C, bool isThumb) {
  return ( isThumb && (C & ~255U) == 0) ||
         (!isThumb && ARM_AM::getSOImmVal(C) != -1);
}

/// Returns appropriate ARM CMP (cmp) and corresponding condition code for
/// the given operands.
static SDValue getARMCmp(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                           SDValue &ARMCC, SelectionDAG &DAG, bool isThumb) {
  if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS.getNode())) {
    unsigned C = RHSC->getZExtValue();
    if (!isLegalCmpImmediate(C, isThumb)) {
      // Constant does not fit, try adjusting it by one?
      switch (CC) {
      default: break;
      case ISD::SETLT:
      case ISD::SETGE:
        if (isLegalCmpImmediate(C-1, isThumb)) {
          CC = (CC == ISD::SETLT) ? ISD::SETLE : ISD::SETGT;
          RHS = DAG.getConstant(C-1, MVT::i32);
        }
        break;
      case ISD::SETULT:
      case ISD::SETUGE:
        if (C > 0 && isLegalCmpImmediate(C-1, isThumb)) {
          CC = (CC == ISD::SETULT) ? ISD::SETULE : ISD::SETUGT;
          RHS = DAG.getConstant(C-1, MVT::i32);
        }
        break;
      case ISD::SETLE:
      case ISD::SETGT:
        if (isLegalCmpImmediate(C+1, isThumb)) {
          CC = (CC == ISD::SETLE) ? ISD::SETLT : ISD::SETGE;
          RHS = DAG.getConstant(C+1, MVT::i32);
        }
        break;
      case ISD::SETULE:
      case ISD::SETUGT:
        if (C < 0xffffffff && isLegalCmpImmediate(C+1, isThumb)) {
          CC = (CC == ISD::SETULE) ? ISD::SETULT : ISD::SETUGE;
          RHS = DAG.getConstant(C+1, MVT::i32);
        }
        break;
      }
    }
  }

  ARMCC::CondCodes CondCode = IntCCToARMCC(CC);
  ARMISD::NodeType CompareType;
  switch (CondCode) {
  default:
    CompareType = ARMISD::CMP;
    break;
  case ARMCC::EQ:
  case ARMCC::NE:
  case ARMCC::MI:
  case ARMCC::PL:
    // Uses only N and Z Flags
    CompareType = ARMISD::CMPNZ;
    break;
  }
  ARMCC = DAG.getConstant(CondCode, MVT::i32);
  return DAG.getNode(CompareType, MVT::Flag, LHS, RHS);
}

/// Returns a appropriate VFP CMP (fcmp{s|d}+fmstat) for the given operands.
static SDValue getVFPCmp(SDValue LHS, SDValue RHS, SelectionDAG &DAG) {
  SDValue Cmp;
  if (!isFloatingPointZero(RHS))
    Cmp = DAG.getNode(ARMISD::CMPFP, MVT::Flag, LHS, RHS);
  else
    Cmp = DAG.getNode(ARMISD::CMPFPw0, MVT::Flag, LHS);
  return DAG.getNode(ARMISD::FMSTAT, MVT::Flag, Cmp);
}

static SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG,
                                const ARMSubtarget *ST) {
  MVT VT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);

  if (LHS.getValueType() == MVT::i32) {
    SDValue ARMCC;
    SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
    SDValue Cmp = getARMCmp(LHS, RHS, CC, ARMCC, DAG, ST->isThumb());
    return DAG.getNode(ARMISD::CMOV, VT, FalseVal, TrueVal, ARMCC, CCR, Cmp);
  }

  ARMCC::CondCodes CondCode, CondCode2;
  if (FPCCToARMCC(CC, CondCode, CondCode2))
    std::swap(TrueVal, FalseVal);

  SDValue ARMCC = DAG.getConstant(CondCode, MVT::i32);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  SDValue Cmp = getVFPCmp(LHS, RHS, DAG);
  SDValue Result = DAG.getNode(ARMISD::CMOV, VT, FalseVal, TrueVal,
                                 ARMCC, CCR, Cmp);
  if (CondCode2 != ARMCC::AL) {
    SDValue ARMCC2 = DAG.getConstant(CondCode2, MVT::i32);
    // FIXME: Needs another CMP because flag can have but one use.
    SDValue Cmp2 = getVFPCmp(LHS, RHS, DAG);
    Result = DAG.getNode(ARMISD::CMOV, VT, Result, TrueVal, ARMCC2, CCR, Cmp2);
  }
  return Result;
}

static SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG,
                            const ARMSubtarget *ST) {
  SDValue  Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue    LHS = Op.getOperand(2);
  SDValue    RHS = Op.getOperand(3);
  SDValue   Dest = Op.getOperand(4);

  if (LHS.getValueType() == MVT::i32) {
    SDValue ARMCC;
    SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
    SDValue Cmp = getARMCmp(LHS, RHS, CC, ARMCC, DAG, ST->isThumb());
    return DAG.getNode(ARMISD::BRCOND, MVT::Other, Chain, Dest, ARMCC, CCR,Cmp);
  }

  assert(LHS.getValueType() == MVT::f32 || LHS.getValueType() == MVT::f64);
  ARMCC::CondCodes CondCode, CondCode2;
  if (FPCCToARMCC(CC, CondCode, CondCode2))
    // Swap the LHS/RHS of the comparison if needed.
    std::swap(LHS, RHS);
  
  SDValue Cmp = getVFPCmp(LHS, RHS, DAG);
  SDValue ARMCC = DAG.getConstant(CondCode, MVT::i32);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  SDVTList VTList = DAG.getVTList(MVT::Other, MVT::Flag);
  SDValue Ops[] = { Chain, Dest, ARMCC, CCR, Cmp };
  SDValue Res = DAG.getNode(ARMISD::BRCOND, VTList, Ops, 5);
  if (CondCode2 != ARMCC::AL) {
    ARMCC = DAG.getConstant(CondCode2, MVT::i32);
    SDValue Ops[] = { Res, Dest, ARMCC, CCR, Res.getValue(1) };
    Res = DAG.getNode(ARMISD::BRCOND, VTList, Ops, 5);
  }
  return Res;
}

SDValue ARMTargetLowering::LowerBR_JT(SDValue Op, SelectionDAG &DAG) {
  SDValue Chain = Op.getOperand(0);
  SDValue Table = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);

  MVT PTy = getPointerTy();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Table);
  ARMFunctionInfo *AFI = DAG.getMachineFunction().getInfo<ARMFunctionInfo>();
  SDValue UId =  DAG.getConstant(AFI->createJumpTableUId(), PTy);
  SDValue JTI = DAG.getTargetJumpTable(JT->getIndex(), PTy);
  Table = DAG.getNode(ARMISD::WrapperJT, MVT::i32, JTI, UId);
  Index = DAG.getNode(ISD::MUL, PTy, Index, DAG.getConstant(4, PTy));
  SDValue Addr = DAG.getNode(ISD::ADD, PTy, Index, Table);
  bool isPIC = getTargetMachine().getRelocationModel() == Reloc::PIC_;
  Addr = DAG.getLoad(isPIC ? (MVT)MVT::i32 : PTy,
                     Chain, Addr, NULL, 0);
  Chain = Addr.getValue(1);
  if (isPIC)
    Addr = DAG.getNode(ISD::ADD, PTy, Addr, Table);
  return DAG.getNode(ARMISD::BR_JT, MVT::Other, Chain, Addr, JTI, UId);
}

static SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG) {
  unsigned Opc =
    Op.getOpcode() == ISD::FP_TO_SINT ? ARMISD::FTOSI : ARMISD::FTOUI;
  Op = DAG.getNode(Opc, MVT::f32, Op.getOperand(0));
  return DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Op);
}

static SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getValueType();
  unsigned Opc =
    Op.getOpcode() == ISD::SINT_TO_FP ? ARMISD::SITOF : ARMISD::UITOF;

  Op = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Op.getOperand(0));
  return DAG.getNode(Opc, VT, Op);
}

static SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) {
  // Implement fcopysign with a fabs and a conditional fneg.
  SDValue Tmp0 = Op.getOperand(0);
  SDValue Tmp1 = Op.getOperand(1);
  MVT VT = Op.getValueType();
  MVT SrcVT = Tmp1.getValueType();
  SDValue AbsVal = DAG.getNode(ISD::FABS, VT, Tmp0);
  SDValue Cmp = getVFPCmp(Tmp1, DAG.getConstantFP(0.0, SrcVT), DAG);
  SDValue ARMCC = DAG.getConstant(ARMCC::LT, MVT::i32);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  return DAG.getNode(ARMISD::CNEG, VT, AbsVal, AbsVal, ARMCC, CCR, Cmp);
}

SDValue
ARMTargetLowering::EmitTargetCodeForMemcpy(SelectionDAG &DAG,
                                           SDValue Chain,
                                           SDValue Dst, SDValue Src,
                                           SDValue Size, unsigned Align,
                                           bool AlwaysInline,
                                         const Value *DstSV, uint64_t DstSVOff,
                                         const Value *SrcSV, uint64_t SrcSVOff){
  // Do repeated 4-byte loads and stores. To be improved.
  // This requires 4-byte alignment.
  if ((Align & 3) != 0)
    return SDValue();
  // This requires the copy size to be a constant, preferrably
  // within a subtarget-specific limit.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();
  uint64_t SizeVal = ConstantSize->getZExtValue();
  if (!AlwaysInline && SizeVal > getSubtarget()->getMaxInlineSizeThreshold())
    return SDValue();

  unsigned BytesLeft = SizeVal & 3;
  unsigned NumMemOps = SizeVal >> 2;
  unsigned EmittedNumMemOps = 0;
  MVT VT = MVT::i32;
  unsigned VTSize = 4;
  unsigned i = 0;
  const unsigned MAX_LOADS_IN_LDM = 6;
  SDValue TFOps[MAX_LOADS_IN_LDM];
  SDValue Loads[MAX_LOADS_IN_LDM];
  uint64_t SrcOff = 0, DstOff = 0;

  // Emit up to MAX_LOADS_IN_LDM loads, then a TokenFactor barrier, then the
  // same number of stores.  The loads and stores will get combined into
  // ldm/stm later on.
  while (EmittedNumMemOps < NumMemOps) {
    for (i = 0;
         i < MAX_LOADS_IN_LDM && EmittedNumMemOps + i < NumMemOps; ++i) {
      Loads[i] = DAG.getLoad(VT, Chain,
                             DAG.getNode(ISD::ADD, MVT::i32, Src,
                                         DAG.getConstant(SrcOff, MVT::i32)),
                             SrcSV, SrcSVOff + SrcOff);
      TFOps[i] = Loads[i].getValue(1);
      SrcOff += VTSize;
    }
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &TFOps[0], i);

    for (i = 0;
         i < MAX_LOADS_IN_LDM && EmittedNumMemOps + i < NumMemOps; ++i) {
      TFOps[i] = DAG.getStore(Chain, Loads[i],
                           DAG.getNode(ISD::ADD, MVT::i32, Dst, 
                                       DAG.getConstant(DstOff, MVT::i32)),
                           DstSV, DstSVOff + DstOff);
      DstOff += VTSize;
    }
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &TFOps[0], i);

    EmittedNumMemOps += i;
  }

  if (BytesLeft == 0) 
    return Chain;

  // Issue loads / stores for the trailing (1 - 3) bytes.
  unsigned BytesLeftSave = BytesLeft;
  i = 0;
  while (BytesLeft) {
    if (BytesLeft >= 2) {
      VT = MVT::i16;
      VTSize = 2;
    } else {
      VT = MVT::i8;
      VTSize = 1;
    }

    Loads[i] = DAG.getLoad(VT, Chain,
                           DAG.getNode(ISD::ADD, MVT::i32, Src,
                                       DAG.getConstant(SrcOff, MVT::i32)),
                           SrcSV, SrcSVOff + SrcOff);
    TFOps[i] = Loads[i].getValue(1);
    ++i;
    SrcOff += VTSize;
    BytesLeft -= VTSize;
  }
  Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &TFOps[0], i);

  i = 0;
  BytesLeft = BytesLeftSave;
  while (BytesLeft) {
    if (BytesLeft >= 2) {
      VT = MVT::i16;
      VTSize = 2;
    } else {
      VT = MVT::i8;
      VTSize = 1;
    }

    TFOps[i] = DAG.getStore(Chain, Loads[i],
                            DAG.getNode(ISD::ADD, MVT::i32, Dst, 
                                        DAG.getConstant(DstOff, MVT::i32)),
                            DstSV, DstSVOff + DstOff);
    ++i;
    DstOff += VTSize;
    BytesLeft -= VTSize;
  }
  return DAG.getNode(ISD::TokenFactor, MVT::Other, &TFOps[0], i);
}

static SDNode *ExpandBIT_CONVERT(SDNode *N, SelectionDAG &DAG) {
  // Turn f64->i64 into FMRRD.
  assert(N->getValueType(0) == MVT::i64 &&
         N->getOperand(0).getValueType() == MVT::f64);
  
  SDValue Op = N->getOperand(0);
  SDValue Cvt = DAG.getNode(ARMISD::FMRRD, DAG.getVTList(MVT::i32, MVT::i32),
                              &Op, 1);
  
  // Merge the pieces into a single i64 value.
  return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Cvt, Cvt.getValue(1)).getNode();
}

static SDNode *ExpandSRx(SDNode *N, SelectionDAG &DAG, const ARMSubtarget *ST) {
  assert(N->getValueType(0) == MVT::i64 &&
         (N->getOpcode() == ISD::SRL || N->getOpcode() == ISD::SRA) &&
         "Unknown shift to lower!");
  
  // We only lower SRA, SRL of 1 here, all others use generic lowering.
  if (!isa<ConstantSDNode>(N->getOperand(1)) ||
      cast<ConstantSDNode>(N->getOperand(1))->getZExtValue() != 1)
    return 0;
  
  // If we are in thumb mode, we don't have RRX.
  if (ST->isThumb()) return 0;
  
  // Okay, we have a 64-bit SRA or SRL of 1.  Lower this to an RRX expr.
  SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, N->getOperand(0),
                             DAG.getConstant(0, MVT::i32));
  SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, N->getOperand(0),
                             DAG.getConstant(1, MVT::i32));
  
  // First, build a SRA_FLAG/SRL_FLAG op, which shifts the top part by one and
  // captures the result into a carry flag.
  unsigned Opc = N->getOpcode() == ISD::SRL ? ARMISD::SRL_FLAG:ARMISD::SRA_FLAG;
  Hi = DAG.getNode(Opc, DAG.getVTList(MVT::i32, MVT::Flag), &Hi, 1);
  
  // The low part is an ARMISD::RRX operand, which shifts the carry in.
  Lo = DAG.getNode(ARMISD::RRX, MVT::i32, Lo, Hi.getValue(1));
  
  // Merge the pieces into a single i64 value.
 return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi).getNode();
}


SDValue ARMTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Don't know how to custom lower this!"); abort();
  case ISD::ConstantPool:  return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:
    return Subtarget->isTargetDarwin() ? LowerGlobalAddressDarwin(Op, DAG) :
      LowerGlobalAddressELF(Op, DAG);
  case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
  case ISD::CALL:          return LowerCALL(Op, DAG);
  case ISD::RET:           return LowerRET(Op, DAG);
  case ISD::SELECT_CC:     return LowerSELECT_CC(Op, DAG, Subtarget);
  case ISD::BR_CC:         return LowerBR_CC(Op, DAG, Subtarget);
  case ISD::BR_JT:         return LowerBR_JT(Op, DAG);
  case ISD::VASTART:       return LowerVASTART(Op, DAG, VarArgsFrameIndex);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:    return LowerINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:    return LowerFP_TO_INT(Op, DAG);
  case ISD::FCOPYSIGN:     return LowerFCOPYSIGN(Op, DAG);
  case ISD::FORMAL_ARGUMENTS: return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::RETURNADDR:    break;
  case ISD::FRAMEADDR:     break;
  case ISD::GLOBAL_OFFSET_TABLE: return LowerGLOBAL_OFFSET_TABLE(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
      
      
  // FIXME: Remove these when LegalizeDAGTypes lands.
  case ISD::BIT_CONVERT:   return SDValue(ExpandBIT_CONVERT(Op.getNode(), DAG), 0);
  case ISD::SRL:
  case ISD::SRA:           return SDValue(ExpandSRx(Op.getNode(), DAG,Subtarget),0);
  }
  return SDValue();
}


/// ReplaceNodeResults - Provide custom lowering hooks for nodes with illegal
/// result types.
SDNode *ARMTargetLowering::ReplaceNodeResults(SDNode *N, SelectionDAG &DAG) {
  switch (N->getOpcode()) {
  default: assert(0 && "Don't know how to custom expand this!"); abort();
  case ISD::BIT_CONVERT:   return ExpandBIT_CONVERT(N, DAG);
  case ISD::SRL:
  case ISD::SRA:           return ExpandSRx(N, DAG, Subtarget);
  }
}
  

//===----------------------------------------------------------------------===//
//                           ARM Scheduler Hooks
//===----------------------------------------------------------------------===//

MachineBasicBlock *
ARMTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                           MachineBasicBlock *BB) {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  switch (MI->getOpcode()) {
  default: assert(false && "Unexpected instr type to insert");
  case ARM::tMOVCCr: {
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
    //   cmpTY ccX, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *thisMBB  = BB;
    MachineFunction *F = BB->getParent();
    MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB  = F->CreateMachineBasicBlock(LLVM_BB);
    BuildMI(BB, TII->get(ARM::tBcc)).addMBB(sinkMBB)
      .addImm(MI->getOperand(3).getImm()).addReg(MI->getOperand(4).getReg());
    F->insert(It, copy0MBB);
    F->insert(It, sinkMBB);
    // Update machine-CFG edges by first adding all successors of the current
    // block to the new block which will contain the Phi node for the select.
    for(MachineBasicBlock::succ_iterator i = BB->succ_begin(),
        e = BB->succ_end(); i != e; ++i)
      sinkMBB->addSuccessor(*i);
    // Next, remove all successors of the current block, and add the true
    // and fallthrough blocks as its successors.
    while(!BB->succ_empty())
      BB->removeSuccessor(BB->succ_begin());
    BB->addSuccessor(copy0MBB);
    BB->addSuccessor(sinkMBB);

    //  copy0MBB:
    //   %FalseValue = ...
    //   # fallthrough to sinkMBB
    BB = copy0MBB;

    // Update machine-CFG edges
    BB->addSuccessor(sinkMBB);

    //  sinkMBB:
    //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
    //  ...
    BB = sinkMBB;
    BuildMI(BB, TII->get(ARM::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

    F->DeleteMachineInstr(MI);   // The pseudo instruction is gone now.
    return BB;
  }
  }
}

//===----------------------------------------------------------------------===//
//                           ARM Optimization Hooks
//===----------------------------------------------------------------------===//

/// PerformFMRRDCombine - Target-specific dag combine xforms for ARMISD::FMRRD.
static SDValue PerformFMRRDCombine(SDNode *N, 
                                     TargetLowering::DAGCombinerInfo &DCI) {
  // fmrrd(fmdrr x, y) -> x,y
  SDValue InDouble = N->getOperand(0);
  if (InDouble.getOpcode() == ARMISD::FMDRR)
    return DCI.CombineTo(N, InDouble.getOperand(0), InDouble.getOperand(1));
  return SDValue();
}

SDValue ARMTargetLowering::PerformDAGCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  default: break;
  case ARMISD::FMRRD: return PerformFMRRDCombine(N, DCI);
  }
  
  return SDValue();
}


/// isLegalAddressImmediate - Return true if the integer value can be used
/// as the offset of the target addressing mode for load / store of the
/// given type.
static bool isLegalAddressImmediate(int64_t V, MVT VT,
                                    const ARMSubtarget *Subtarget) {
  if (V == 0)
    return true;

  if (Subtarget->isThumb()) {
    if (V < 0)
      return false;

    unsigned Scale = 1;
    switch (VT.getSimpleVT()) {
    default: return false;
    case MVT::i1:
    case MVT::i8:
      // Scale == 1;
      break;
    case MVT::i16:
      // Scale == 2;
      Scale = 2;
      break;
    case MVT::i32:
      // Scale == 4;
      Scale = 4;
      break;
    }

    if ((V & (Scale - 1)) != 0)
      return false;
    V /= Scale;
    return V == (V & ((1LL << 5) - 1));
  }

  if (V < 0)
    V = - V;
  switch (VT.getSimpleVT()) {
  default: return false;
  case MVT::i1:
  case MVT::i8:
  case MVT::i32:
    // +- imm12
    return V == (V & ((1LL << 12) - 1));
  case MVT::i16:
    // +- imm8
    return V == (V & ((1LL << 8) - 1));
  case MVT::f32:
  case MVT::f64:
    if (!Subtarget->hasVFP2())
      return false;
    if ((V & 3) != 0)
      return false;
    V >>= 2;
    return V == (V & ((1LL << 8) - 1));
  }
}

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool ARMTargetLowering::isLegalAddressingMode(const AddrMode &AM, 
                                              const Type *Ty) const {
  if (!isLegalAddressImmediate(AM.BaseOffs, getValueType(Ty, true), Subtarget))
    return false;
  
  // Can never fold addr of global into load/store.
  if (AM.BaseGV) 
    return false;
  
  switch (AM.Scale) {
  case 0:  // no scale reg, must be "r+i" or "r", or "i".
    break;
  case 1:
    if (Subtarget->isThumb())
      return false;
    // FALL THROUGH.
  default:
    // ARM doesn't support any R+R*scale+imm addr modes.
    if (AM.BaseOffs)
      return false;
    
    int Scale = AM.Scale;
    switch (getValueType(Ty).getSimpleVT()) {
    default: return false;
    case MVT::i1:
    case MVT::i8:
    case MVT::i32:
    case MVT::i64:
      // This assumes i64 is legalized to a pair of i32. If not (i.e.
      // ldrd / strd are used, then its address mode is same as i16.
      // r + r
      if (Scale < 0) Scale = -Scale;
      if (Scale == 1)
        return true;
      // r + r << imm
      return isPowerOf2_32(Scale & ~1);
    case MVT::i16:
      // r + r
      if (((unsigned)AM.HasBaseReg + Scale) <= 2)
        return true;
      return false;
      
    case MVT::isVoid:
      // Note, we allow "void" uses (basically, uses that aren't loads or
      // stores), because arm allows folding a scale into many arithmetic
      // operations.  This should be made more precise and revisited later.
      
      // Allow r << imm, but the imm has to be a multiple of two.
      if (AM.Scale & 1) return false;
      return isPowerOf2_32(AM.Scale);
    }
    break;
  }
  return true;
}


static bool getIndexedAddressParts(SDNode *Ptr, MVT VT,
                                   bool isSEXTLoad, SDValue &Base,
                                   SDValue &Offset, bool &isInc,
                                   SelectionDAG &DAG) {
  if (Ptr->getOpcode() != ISD::ADD && Ptr->getOpcode() != ISD::SUB)
    return false;

  if (VT == MVT::i16 || ((VT == MVT::i8 || VT == MVT::i1) && isSEXTLoad)) {
    // AddressingMode 3
    Base = Ptr->getOperand(0);
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(Ptr->getOperand(1))) {
      int RHSC = (int)RHS->getZExtValue();
      if (RHSC < 0 && RHSC > -256) {
        isInc = false;
        Offset = DAG.getConstant(-RHSC, RHS->getValueType(0));
        return true;
      }
    }
    isInc = (Ptr->getOpcode() == ISD::ADD);
    Offset = Ptr->getOperand(1);
    return true;
  } else if (VT == MVT::i32 || VT == MVT::i8 || VT == MVT::i1) {
    // AddressingMode 2
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(Ptr->getOperand(1))) {
      int RHSC = (int)RHS->getZExtValue();
      if (RHSC < 0 && RHSC > -0x1000) {
        isInc = false;
        Offset = DAG.getConstant(-RHSC, RHS->getValueType(0));
        Base = Ptr->getOperand(0);
        return true;
      }
    }

    if (Ptr->getOpcode() == ISD::ADD) {
      isInc = true;
      ARM_AM::ShiftOpc ShOpcVal= ARM_AM::getShiftOpcForNode(Ptr->getOperand(0));
      if (ShOpcVal != ARM_AM::no_shift) {
        Base = Ptr->getOperand(1);
        Offset = Ptr->getOperand(0);
      } else {
        Base = Ptr->getOperand(0);
        Offset = Ptr->getOperand(1);
      }
      return true;
    }

    isInc = (Ptr->getOpcode() == ISD::ADD);
    Base = Ptr->getOperand(0);
    Offset = Ptr->getOperand(1);
    return true;
  }

  // FIXME: Use FLDM / FSTM to emulate indexed FP load / store.
  return false;
}

/// getPreIndexedAddressParts - returns true by value, base pointer and
/// offset pointer and addressing mode by reference if the node's address
/// can be legally represented as pre-indexed load / store address.
bool
ARMTargetLowering::getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                             SDValue &Offset,
                                             ISD::MemIndexedMode &AM,
                                             SelectionDAG &DAG) {
  if (Subtarget->isThumb())
    return false;

  MVT VT;
  SDValue Ptr;
  bool isSEXTLoad = false;
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    Ptr = LD->getBasePtr();
    VT  = LD->getMemoryVT();
    isSEXTLoad = LD->getExtensionType() == ISD::SEXTLOAD;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    Ptr = ST->getBasePtr();
    VT  = ST->getMemoryVT();
  } else
    return false;

  bool isInc;
  bool isLegal = getIndexedAddressParts(Ptr.getNode(), VT, isSEXTLoad, Base, Offset,
                                        isInc, DAG);
  if (isLegal) {
    AM = isInc ? ISD::PRE_INC : ISD::PRE_DEC;
    return true;
  }
  return false;
}

/// getPostIndexedAddressParts - returns true by value, base pointer and
/// offset pointer and addressing mode by reference if this node can be
/// combined with a load / store to form a post-indexed load / store.
bool ARMTargetLowering::getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                                   SDValue &Base,
                                                   SDValue &Offset,
                                                   ISD::MemIndexedMode &AM,
                                                   SelectionDAG &DAG) {
  if (Subtarget->isThumb())
    return false;

  MVT VT;
  SDValue Ptr;
  bool isSEXTLoad = false;
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    VT  = LD->getMemoryVT();
    isSEXTLoad = LD->getExtensionType() == ISD::SEXTLOAD;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    VT  = ST->getMemoryVT();
  } else
    return false;

  bool isInc;
  bool isLegal = getIndexedAddressParts(Op, VT, isSEXTLoad, Base, Offset,
                                        isInc, DAG);
  if (isLegal) {
    AM = isInc ? ISD::POST_INC : ISD::POST_DEC;
    return true;
  }
  return false;
}

void ARMTargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                       const APInt &Mask,
                                                       APInt &KnownZero, 
                                                       APInt &KnownOne,
                                                       const SelectionDAG &DAG,
                                                       unsigned Depth) const {
  KnownZero = KnownOne = APInt(Mask.getBitWidth(), 0);
  switch (Op.getOpcode()) {
  default: break;
  case ARMISD::CMOV: {
    // Bits are known zero/one if known on the LHS and RHS.
    DAG.ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
    if (KnownZero == 0 && KnownOne == 0) return;

    APInt KnownZeroRHS, KnownOneRHS;
    DAG.ComputeMaskedBits(Op.getOperand(1), Mask,
                          KnownZeroRHS, KnownOneRHS, Depth+1);
    KnownZero &= KnownZeroRHS;
    KnownOne  &= KnownOneRHS;
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
//                           ARM Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
ARMTargetLowering::ConstraintType
ARMTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:  break;
    case 'l': return C_RegisterClass;
    case 'w': return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass*> 
ARMTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                MVT VT) const {
  if (Constraint.size() == 1) {
    // GCC RS6000 Constraint Letters
    switch (Constraint[0]) {
    case 'l':
    // FIXME: in thumb mode, 'l' is only low-regs.
    // FALL THROUGH.
    case 'r':
      return std::make_pair(0U, ARM::GPRRegisterClass);
    case 'w':
      if (VT == MVT::f32)
        return std::make_pair(0U, ARM::SPRRegisterClass);
      if (VT == MVT::f64)
        return std::make_pair(0U, ARM::DPRRegisterClass);
      break;
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

std::vector<unsigned> ARMTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT VT) const {
  if (Constraint.size() != 1)
    return std::vector<unsigned>();

  switch (Constraint[0]) {      // GCC ARM Constraint Letters
  default: break;
  case 'l':
  case 'r':
    return make_vector<unsigned>(ARM::R0, ARM::R1, ARM::R2, ARM::R3,
                                 ARM::R4, ARM::R5, ARM::R6, ARM::R7,
                                 ARM::R8, ARM::R9, ARM::R10, ARM::R11,
                                 ARM::R12, ARM::LR, 0);
  case 'w':
    if (VT == MVT::f32)
      return make_vector<unsigned>(ARM::S0, ARM::S1, ARM::S2, ARM::S3,
                                   ARM::S4, ARM::S5, ARM::S6, ARM::S7,
                                   ARM::S8, ARM::S9, ARM::S10, ARM::S11,
                                   ARM::S12,ARM::S13,ARM::S14,ARM::S15,
                                   ARM::S16,ARM::S17,ARM::S18,ARM::S19,
                                   ARM::S20,ARM::S21,ARM::S22,ARM::S23,
                                   ARM::S24,ARM::S25,ARM::S26,ARM::S27,
                                   ARM::S28,ARM::S29,ARM::S30,ARM::S31, 0);
    if (VT == MVT::f64)
      return make_vector<unsigned>(ARM::D0, ARM::D1, ARM::D2, ARM::D3,
                                   ARM::D4, ARM::D5, ARM::D6, ARM::D7,
                                   ARM::D8, ARM::D9, ARM::D10,ARM::D11,
                                   ARM::D12,ARM::D13,ARM::D14,ARM::D15, 0);
      break;
  }

  return std::vector<unsigned>();
}
