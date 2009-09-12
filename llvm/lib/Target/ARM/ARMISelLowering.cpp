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
#include "ARMPerfectShuffle.h"
#include "ARMRegisterInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "ARMTargetObjectFile.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instruction.h"
#include "llvm/Intrinsics.h"
#include "llvm/GlobalValue.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

static bool CC_ARM_APCS_Custom_f64(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                                   CCValAssign::LocInfo &LocInfo,
                                   ISD::ArgFlagsTy &ArgFlags,
                                   CCState &State);
static bool CC_ARM_AAPCS_Custom_f64(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                                    CCValAssign::LocInfo &LocInfo,
                                    ISD::ArgFlagsTy &ArgFlags,
                                    CCState &State);
static bool RetCC_ARM_APCS_Custom_f64(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                                      CCValAssign::LocInfo &LocInfo,
                                      ISD::ArgFlagsTy &ArgFlags,
                                      CCState &State);
static bool RetCC_ARM_AAPCS_Custom_f64(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                                       CCValAssign::LocInfo &LocInfo,
                                       ISD::ArgFlagsTy &ArgFlags,
                                       CCState &State);

void ARMTargetLowering::addTypeForNEON(EVT VT, EVT PromotedLdStVT,
                                       EVT PromotedBitwiseVT) {
  if (VT != PromotedLdStVT) {
    setOperationAction(ISD::LOAD, VT.getSimpleVT(), Promote);
    AddPromotedToType (ISD::LOAD, VT.getSimpleVT(),
                       PromotedLdStVT.getSimpleVT());

    setOperationAction(ISD::STORE, VT.getSimpleVT(), Promote);
    AddPromotedToType (ISD::STORE, VT.getSimpleVT(),
                       PromotedLdStVT.getSimpleVT());
  }

  EVT ElemTy = VT.getVectorElementType();
  if (ElemTy != MVT::i64 && ElemTy != MVT::f64)
    setOperationAction(ISD::VSETCC, VT.getSimpleVT(), Custom);
  if (ElemTy == MVT::i8 || ElemTy == MVT::i16)
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::BUILD_VECTOR, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::VECTOR_SHUFFLE, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::SCALAR_TO_VECTOR, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::CONCAT_VECTORS, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, VT.getSimpleVT(), Expand);
  if (VT.isInteger()) {
    setOperationAction(ISD::SHL, VT.getSimpleVT(), Custom);
    setOperationAction(ISD::SRA, VT.getSimpleVT(), Custom);
    setOperationAction(ISD::SRL, VT.getSimpleVT(), Custom);
  }

  // Promote all bit-wise operations.
  if (VT.isInteger() && VT != PromotedBitwiseVT) {
    setOperationAction(ISD::AND, VT.getSimpleVT(), Promote);
    AddPromotedToType (ISD::AND, VT.getSimpleVT(),
                       PromotedBitwiseVT.getSimpleVT());
    setOperationAction(ISD::OR,  VT.getSimpleVT(), Promote);
    AddPromotedToType (ISD::OR,  VT.getSimpleVT(),
                       PromotedBitwiseVT.getSimpleVT());
    setOperationAction(ISD::XOR, VT.getSimpleVT(), Promote);
    AddPromotedToType (ISD::XOR, VT.getSimpleVT(),
                       PromotedBitwiseVT.getSimpleVT());
  }
}

void ARMTargetLowering::addDRTypeForNEON(EVT VT) {
  addRegisterClass(VT, ARM::DPRRegisterClass);
  addTypeForNEON(VT, MVT::f64, MVT::v2i32);
}

void ARMTargetLowering::addQRTypeForNEON(EVT VT) {
  addRegisterClass(VT, ARM::QPRRegisterClass);
  addTypeForNEON(VT, MVT::v2f64, MVT::v4i32);
}

static TargetLoweringObjectFile *createTLOF(TargetMachine &TM) {
  if (TM.getSubtarget<ARMSubtarget>().isTargetDarwin())
    return new TargetLoweringObjectFileMachO();
  return new ARMElfTargetObjectFile();
}

ARMTargetLowering::ARMTargetLowering(TargetMachine &TM)
    : TargetLowering(TM, createTLOF(TM)), ARMPCLabelIndex(0) {
  Subtarget = &TM.getSubtarget<ARMSubtarget>();

  if (Subtarget->isTargetDarwin()) {
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
      // FIXME: There appears to be some naming inconsistency in ARM libgcc:
      // e.g., __floatunsidf vs. __floatunssidfvfp.
      setLibcallName(RTLIB::SINTTOFP_I32_F64, "__floatsidfvfp");
      setLibcallName(RTLIB::UINTTOFP_I32_F64, "__floatunssidfvfp");
      setLibcallName(RTLIB::SINTTOFP_I32_F32, "__floatsisfvfp");
      setLibcallName(RTLIB::UINTTOFP_I32_F32, "__floatunssisfvfp");
    }
  }

  // These libcalls are not available in 32-bit.
  setLibcallName(RTLIB::SHL_I128, 0);
  setLibcallName(RTLIB::SRL_I128, 0);
  setLibcallName(RTLIB::SRA_I128, 0);

  // Libcalls should use the AAPCS base standard ABI, even if hard float
  // is in effect, as per the ARM RTABI specification, section 4.1.2.
  if (Subtarget->isAAPCS_ABI()) {
    for (int i = 0; i < RTLIB::UNKNOWN_LIBCALL; ++i) {
      setLibcallCallingConv(static_cast<RTLIB::Libcall>(i),
                            CallingConv::ARM_AAPCS);
    }
  }

  if (Subtarget->isThumb1Only())
    addRegisterClass(MVT::i32, ARM::tGPRRegisterClass);
  else
    addRegisterClass(MVT::i32, ARM::GPRRegisterClass);
  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb1Only()) {
    addRegisterClass(MVT::f32, ARM::SPRRegisterClass);
    addRegisterClass(MVT::f64, ARM::DPRRegisterClass);

    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  }

  if (Subtarget->hasNEON()) {
    addDRTypeForNEON(MVT::v2f32);
    addDRTypeForNEON(MVT::v8i8);
    addDRTypeForNEON(MVT::v4i16);
    addDRTypeForNEON(MVT::v2i32);
    addDRTypeForNEON(MVT::v1i64);

    addQRTypeForNEON(MVT::v4f32);
    addQRTypeForNEON(MVT::v2f64);
    addQRTypeForNEON(MVT::v16i8);
    addQRTypeForNEON(MVT::v8i16);
    addQRTypeForNEON(MVT::v4i32);
    addQRTypeForNEON(MVT::v2i64);

    setTargetDAGCombine(ISD::INTRINSIC_WO_CHAIN);
    setTargetDAGCombine(ISD::SHL);
    setTargetDAGCombine(ISD::SRL);
    setTargetDAGCombine(ISD::SRA);
    setTargetDAGCombine(ISD::SIGN_EXTEND);
    setTargetDAGCombine(ISD::ZERO_EXTEND);
    setTargetDAGCombine(ISD::ANY_EXTEND);
  }

  computeRegisterProperties();

  // ARM does not have f32 extending load.
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);

  // ARM does not have i1 sign extending load.
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);

  // ARM supports all 4 flavors of integer indexed load / store.
  if (!Subtarget->isThumb1Only()) {
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
  }

  // i64 operation support.
  if (Subtarget->isThumb1Only()) {
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
  setOperationAction(ISD::CTTZ,  MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  if (!Subtarget->hasV5TOps() || Subtarget->isThumb1Only())
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

  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32,   Custom);
  setOperationAction(ISD::GLOBAL_OFFSET_TABLE, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);

  // Use the default implementation.
  setOperationAction(ISD::VASTART,            MVT::Other, Custom);
  setOperationAction(ISD::VAARG,              MVT::Other, Expand);
  setOperationAction(ISD::VACOPY,             MVT::Other, Expand);
  setOperationAction(ISD::VAEND,              MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE,          MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE,       MVT::Other, Expand);
  setOperationAction(ISD::EHSELECTION,        MVT::i32,   Expand);
  // FIXME: Shouldn't need this, since no register is used, but the legalizer
  // doesn't yet know how to not do that for SjLj.
  setExceptionSelectorRegister(ARM::R0);
  if (Subtarget->isThumb())
    setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Custom);
  else
    setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);
  setOperationAction(ISD::MEMBARRIER,         MVT::Other, Expand);

  if (!Subtarget->hasV6Ops() && !Subtarget->isThumb2()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8,  Expand);
  }
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb1Only())
    // Turn f64->i64 into FMRRD, i64 -> f64 to FMDRR iff target supports vfp2.
    setOperationAction(ISD::BIT_CONVERT, MVT::i64, Custom);

  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

  setOperationAction(ISD::SETCC,     MVT::i32, Expand);
  setOperationAction(ISD::SETCC,     MVT::f32, Expand);
  setOperationAction(ISD::SETCC,     MVT::f64, Expand);
  setOperationAction(ISD::SELECT,    MVT::i32, Expand);
  setOperationAction(ISD::SELECT,    MVT::f32, Expand);
  setOperationAction(ISD::SELECT,    MVT::f64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);

  setOperationAction(ISD::BRCOND,    MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,     MVT::i32,   Custom);
  setOperationAction(ISD::BR_CC,     MVT::f32,   Custom);
  setOperationAction(ISD::BR_CC,     MVT::f64,   Custom);
  setOperationAction(ISD::BR_JT,     MVT::Other, Custom);

  // We don't support sin/cos/fmod/copysign/pow
  setOperationAction(ISD::FSIN,      MVT::f64, Expand);
  setOperationAction(ISD::FSIN,      MVT::f32, Expand);
  setOperationAction(ISD::FCOS,      MVT::f32, Expand);
  setOperationAction(ISD::FCOS,      MVT::f64, Expand);
  setOperationAction(ISD::FREM,      MVT::f64, Expand);
  setOperationAction(ISD::FREM,      MVT::f32, Expand);
  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb1Only()) {
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);
  }
  setOperationAction(ISD::FPOW,      MVT::f64, Expand);
  setOperationAction(ISD::FPOW,      MVT::f32, Expand);

  // int <-> fp are custom expanded into bit_convert + ARMISD ops.
  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb1Only()) {
    setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
    setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  }

  // We have target-specific dag combine patterns for the following nodes:
  // ARMISD::FMRRD  - No need to call setTargetDAGCombine
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::SUB);

  setStackPointerRegisterToSaveRestore(ARM::SP);
  setSchedulingPreference(SchedulingForRegPressure);

  // FIXME: If-converter should use instruction latency to determine
  // profitability rather than relying on fixed limits.
  if (Subtarget->getCPUString() == "generic") {
    // Generic (and overly aggressive) if-conversion limits.
    setIfCvtBlockSizeLimit(10);
    setIfCvtDupBlockSizeLimit(2);
  } else if (Subtarget->hasV6Ops()) {
    setIfCvtBlockSizeLimit(2);
    setIfCvtDupBlockSizeLimit(1);
  } else {
    setIfCvtBlockSizeLimit(3);
    setIfCvtDupBlockSizeLimit(2);
  }

  maxStoresPerMemcpy = 1;   //// temporary - rewrite interface to use type
  // Do not enable CodePlacementOpt for now: it currently runs after the
  // ARMConstantIslandPass and messes up branch relaxation and placement
  // of constant islands.
  // benefitFromCodePlacementOpt = true;
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
  case ARMISD::BR2_JT:        return "ARMISD::BR2_JT";
  case ARMISD::RET_FLAG:      return "ARMISD::RET_FLAG";
  case ARMISD::PIC_ADD:       return "ARMISD::PIC_ADD";
  case ARMISD::CMP:           return "ARMISD::CMP";
  case ARMISD::CMPZ:          return "ARMISD::CMPZ";
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

  case ARMISD::DYN_ALLOC:     return "ARMISD::DYN_ALLOC";

  case ARMISD::VCEQ:          return "ARMISD::VCEQ";
  case ARMISD::VCGE:          return "ARMISD::VCGE";
  case ARMISD::VCGEU:         return "ARMISD::VCGEU";
  case ARMISD::VCGT:          return "ARMISD::VCGT";
  case ARMISD::VCGTU:         return "ARMISD::VCGTU";
  case ARMISD::VTST:          return "ARMISD::VTST";

  case ARMISD::VSHL:          return "ARMISD::VSHL";
  case ARMISD::VSHRs:         return "ARMISD::VSHRs";
  case ARMISD::VSHRu:         return "ARMISD::VSHRu";
  case ARMISD::VSHLLs:        return "ARMISD::VSHLLs";
  case ARMISD::VSHLLu:        return "ARMISD::VSHLLu";
  case ARMISD::VSHLLi:        return "ARMISD::VSHLLi";
  case ARMISD::VSHRN:         return "ARMISD::VSHRN";
  case ARMISD::VRSHRs:        return "ARMISD::VRSHRs";
  case ARMISD::VRSHRu:        return "ARMISD::VRSHRu";
  case ARMISD::VRSHRN:        return "ARMISD::VRSHRN";
  case ARMISD::VQSHLs:        return "ARMISD::VQSHLs";
  case ARMISD::VQSHLu:        return "ARMISD::VQSHLu";
  case ARMISD::VQSHLsu:       return "ARMISD::VQSHLsu";
  case ARMISD::VQSHRNs:       return "ARMISD::VQSHRNs";
  case ARMISD::VQSHRNu:       return "ARMISD::VQSHRNu";
  case ARMISD::VQSHRNsu:      return "ARMISD::VQSHRNsu";
  case ARMISD::VQRSHRNs:      return "ARMISD::VQRSHRNs";
  case ARMISD::VQRSHRNu:      return "ARMISD::VQRSHRNu";
  case ARMISD::VQRSHRNsu:     return "ARMISD::VQRSHRNsu";
  case ARMISD::VGETLANEu:     return "ARMISD::VGETLANEu";
  case ARMISD::VGETLANEs:     return "ARMISD::VGETLANEs";
  case ARMISD::VDUP:          return "ARMISD::VDUP";
  case ARMISD::VDUPLANE:      return "ARMISD::VDUPLANE";
  case ARMISD::VLD2D:         return "ARMISD::VLD2D";
  case ARMISD::VLD3D:         return "ARMISD::VLD3D";
  case ARMISD::VLD4D:         return "ARMISD::VLD4D";
  case ARMISD::VST2D:         return "ARMISD::VST2D";
  case ARMISD::VST3D:         return "ARMISD::VST3D";
  case ARMISD::VST4D:         return "ARMISD::VST4D";
  case ARMISD::VEXT:          return "ARMISD::VEXT";
  case ARMISD::VREV64:        return "ARMISD::VREV64";
  case ARMISD::VREV32:        return "ARMISD::VREV32";
  case ARMISD::VREV16:        return "ARMISD::VREV16";
  case ARMISD::VZIP:          return "ARMISD::VZIP";
  case ARMISD::VUZP:          return "ARMISD::VUZP";
  case ARMISD::VTRN:          return "ARMISD::VTRN";
  }
}

/// getFunctionAlignment - Return the Log2 alignment of this function.
unsigned ARMTargetLowering::getFunctionAlignment(const Function *F) const {
  return getTargetMachine().getSubtarget<ARMSubtarget>().isThumb() ? 1 : 2;
}

//===----------------------------------------------------------------------===//
// Lowering Code
//===----------------------------------------------------------------------===//

/// IntCCToARMCC - Convert a DAG integer condition code to an ARM CC
static ARMCC::CondCodes IntCCToARMCC(ISD::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Unknown condition code!");
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
  default: llvm_unreachable("Unknown FP condition!");
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

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "ARMGenCallingConv.inc"

// APCS f64 is in register pairs, possibly split to stack
static bool f64AssignAPCS(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                          CCValAssign::LocInfo &LocInfo,
                          CCState &State, bool CanFail) {
  static const unsigned RegList[] = { ARM::R0, ARM::R1, ARM::R2, ARM::R3 };

  // Try to get the first register.
  if (unsigned Reg = State.AllocateReg(RegList, 4))
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  else {
    // For the 2nd half of a v2f64, do not fail.
    if (CanFail)
      return false;

    // Put the whole thing on the stack.
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                           State.AllocateStack(8, 4),
                                           LocVT, LocInfo));
    return true;
  }

  // Try to get the second register.
  if (unsigned Reg = State.AllocateReg(RegList, 4))
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  else
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                           State.AllocateStack(4, 4),
                                           LocVT, LocInfo));
  return true;
}

static bool CC_ARM_APCS_Custom_f64(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                                   CCValAssign::LocInfo &LocInfo,
                                   ISD::ArgFlagsTy &ArgFlags,
                                   CCState &State) {
  if (!f64AssignAPCS(ValNo, ValVT, LocVT, LocInfo, State, true))
    return false;
  if (LocVT == MVT::v2f64 &&
      !f64AssignAPCS(ValNo, ValVT, LocVT, LocInfo, State, false))
    return false;
  return true;  // we handled it
}

// AAPCS f64 is in aligned register pairs
static bool f64AssignAAPCS(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                           CCValAssign::LocInfo &LocInfo,
                           CCState &State, bool CanFail) {
  static const unsigned HiRegList[] = { ARM::R0, ARM::R2 };
  static const unsigned LoRegList[] = { ARM::R1, ARM::R3 };

  unsigned Reg = State.AllocateReg(HiRegList, LoRegList, 2);
  if (Reg == 0) {
    // For the 2nd half of a v2f64, do not just fail.
    if (CanFail)
      return false;

    // Put the whole thing on the stack.
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                           State.AllocateStack(8, 8),
                                           LocVT, LocInfo));
    return true;
  }

  unsigned i;
  for (i = 0; i < 2; ++i)
    if (HiRegList[i] == Reg)
      break;

  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, LoRegList[i],
                                         LocVT, LocInfo));
  return true;
}

static bool CC_ARM_AAPCS_Custom_f64(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                                    CCValAssign::LocInfo &LocInfo,
                                    ISD::ArgFlagsTy &ArgFlags,
                                    CCState &State) {
  if (!f64AssignAAPCS(ValNo, ValVT, LocVT, LocInfo, State, true))
    return false;
  if (LocVT == MVT::v2f64 &&
      !f64AssignAAPCS(ValNo, ValVT, LocVT, LocInfo, State, false))
    return false;
  return true;  // we handled it
}

static bool f64RetAssign(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                         CCValAssign::LocInfo &LocInfo, CCState &State) {
  static const unsigned HiRegList[] = { ARM::R0, ARM::R2 };
  static const unsigned LoRegList[] = { ARM::R1, ARM::R3 };

  unsigned Reg = State.AllocateReg(HiRegList, LoRegList, 2);
  if (Reg == 0)
    return false; // we didn't handle it

  unsigned i;
  for (i = 0; i < 2; ++i)
    if (HiRegList[i] == Reg)
      break;

  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, LoRegList[i],
                                         LocVT, LocInfo));
  return true;
}

static bool RetCC_ARM_APCS_Custom_f64(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                                      CCValAssign::LocInfo &LocInfo,
                                      ISD::ArgFlagsTy &ArgFlags,
                                      CCState &State) {
  if (!f64RetAssign(ValNo, ValVT, LocVT, LocInfo, State))
    return false;
  if (LocVT == MVT::v2f64 && !f64RetAssign(ValNo, ValVT, LocVT, LocInfo, State))
    return false;
  return true;  // we handled it
}

static bool RetCC_ARM_AAPCS_Custom_f64(unsigned &ValNo, EVT &ValVT, EVT &LocVT,
                                       CCValAssign::LocInfo &LocInfo,
                                       ISD::ArgFlagsTy &ArgFlags,
                                       CCState &State) {
  return RetCC_ARM_APCS_Custom_f64(ValNo, ValVT, LocVT, LocInfo, ArgFlags,
                                   State);
}

/// CCAssignFnForNode - Selects the correct CCAssignFn for a the
/// given CallingConvention value.
CCAssignFn *ARMTargetLowering::CCAssignFnForNode(unsigned CC,
                                                 bool Return,
                                                 bool isVarArg) const {
  switch (CC) {
  default:
    llvm_unreachable("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    // Use target triple & subtarget features to do actual dispatch.
    if (Subtarget->isAAPCS_ABI()) {
      if (Subtarget->hasVFP2() &&
          FloatABIType == FloatABI::Hard && !isVarArg)
        return (Return ? RetCC_ARM_AAPCS_VFP: CC_ARM_AAPCS_VFP);
      else
        return (Return ? RetCC_ARM_AAPCS: CC_ARM_AAPCS);
    } else
        return (Return ? RetCC_ARM_APCS: CC_ARM_APCS);
  case CallingConv::ARM_AAPCS_VFP:
    return (Return ? RetCC_ARM_AAPCS_VFP: CC_ARM_AAPCS_VFP);
  case CallingConv::ARM_AAPCS:
    return (Return ? RetCC_ARM_AAPCS: CC_ARM_AAPCS);
  case CallingConv::ARM_APCS:
    return (Return ? RetCC_ARM_APCS: CC_ARM_APCS);
  }
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
SDValue
ARMTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                   unsigned CallConv, bool isVarArg,
                                   const SmallVectorImpl<ISD::InputArg> &Ins,
                                   DebugLoc dl, SelectionDAG &DAG,
                                   SmallVectorImpl<SDValue> &InVals) {

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
                 RVLocs, *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins,
                           CCAssignFnForNode(CallConv, /* Return*/ true,
                                             isVarArg));

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];

    SDValue Val;
    if (VA.needsCustom()) {
      // Handle f64 or half of a v2f64.
      SDValue Lo = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), MVT::i32,
                                      InFlag);
      Chain = Lo.getValue(1);
      InFlag = Lo.getValue(2);
      VA = RVLocs[++i]; // skip ahead to next loc
      SDValue Hi = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), MVT::i32,
                                      InFlag);
      Chain = Hi.getValue(1);
      InFlag = Hi.getValue(2);
      Val = DAG.getNode(ARMISD::FMDRR, dl, MVT::f64, Lo, Hi);

      if (VA.getLocVT() == MVT::v2f64) {
        SDValue Vec = DAG.getNode(ISD::UNDEF, dl, MVT::v2f64);
        Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64, Vec, Val,
                          DAG.getConstant(0, MVT::i32));

        VA = RVLocs[++i]; // skip ahead to next loc
        Lo = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), MVT::i32, InFlag);
        Chain = Lo.getValue(1);
        InFlag = Lo.getValue(2);
        VA = RVLocs[++i]; // skip ahead to next loc
        Hi = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), MVT::i32, InFlag);
        Chain = Hi.getValue(1);
        InFlag = Hi.getValue(2);
        Val = DAG.getNode(ARMISD::FMDRR, dl, MVT::f64, Lo, Hi);
        Val = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64, Vec, Val,
                          DAG.getConstant(1, MVT::i32));
      }
    } else {
      Val = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), VA.getLocVT(),
                               InFlag);
      Chain = Val.getValue(1);
      InFlag = Val.getValue(2);
    }

    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BIT_CONVERT, dl, VA.getValVT(), Val);
      break;
    }

    InVals.push_back(Val);
  }

  return Chain;
}

/// CreateCopyOfByValArgument - Make a copy of an aggregate at address specified
/// by "Src" to address "Dst" of size "Size".  Alignment information is
/// specified by the specific parameter attribute.  The copy will be passed as
/// a byval function parameter.
/// Sometimes what we are copying is the end of a larger object, the part that
/// does not fit in registers.
static SDValue
CreateCopyOfByValArgument(SDValue Src, SDValue Dst, SDValue Chain,
                          ISD::ArgFlagsTy Flags, SelectionDAG &DAG,
                          DebugLoc dl) {
  SDValue SizeNode = DAG.getConstant(Flags.getByValSize(), MVT::i32);
  return DAG.getMemcpy(Chain, dl, Dst, Src, SizeNode, Flags.getByValAlign(),
                       /*AlwaysInline=*/false, NULL, 0, NULL, 0);
}

/// LowerMemOpCallTo - Store the argument to the stack.
SDValue
ARMTargetLowering::LowerMemOpCallTo(SDValue Chain,
                                    SDValue StackPtr, SDValue Arg,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    const CCValAssign &VA,
                                    ISD::ArgFlagsTy Flags) {
  unsigned LocMemOffset = VA.getLocMemOffset();
  SDValue PtrOff = DAG.getIntPtrConstant(LocMemOffset);
  PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, PtrOff);
  if (Flags.isByVal()) {
    return CreateCopyOfByValArgument(Arg, PtrOff, Chain, Flags, DAG, dl);
  }
  return DAG.getStore(Chain, dl, Arg, PtrOff,
                      PseudoSourceValue::getStack(), LocMemOffset);
}

void ARMTargetLowering::PassF64ArgInRegs(DebugLoc dl, SelectionDAG &DAG,
                                         SDValue Chain, SDValue &Arg,
                                         RegsToPassVector &RegsToPass,
                                         CCValAssign &VA, CCValAssign &NextVA,
                                         SDValue &StackPtr,
                                         SmallVector<SDValue, 8> &MemOpChains,
                                         ISD::ArgFlagsTy Flags) {

  SDValue fmrrd = DAG.getNode(ARMISD::FMRRD, dl,
                              DAG.getVTList(MVT::i32, MVT::i32), Arg);
  RegsToPass.push_back(std::make_pair(VA.getLocReg(), fmrrd));

  if (NextVA.isRegLoc())
    RegsToPass.push_back(std::make_pair(NextVA.getLocReg(), fmrrd.getValue(1)));
  else {
    assert(NextVA.isMemLoc());
    if (StackPtr.getNode() == 0)
      StackPtr = DAG.getCopyFromReg(Chain, dl, ARM::SP, getPointerTy());

    MemOpChains.push_back(LowerMemOpCallTo(Chain, StackPtr, fmrrd.getValue(1),
                                           dl, DAG, NextVA,
                                           Flags));
  }
}

/// LowerCall - Lowering a call into a callseq_start <-
/// ARMISD:CALL <- callseq_end chain. Also add input and output parameter
/// nodes.
SDValue
ARMTargetLowering::LowerCall(SDValue Chain, SDValue Callee,
                             unsigned CallConv, bool isVarArg,
                             bool isTailCall,
                             const SmallVectorImpl<ISD::OutputArg> &Outs,
                             const SmallVectorImpl<ISD::InputArg> &Ins,
                             DebugLoc dl, SelectionDAG &DAG,
                             SmallVectorImpl<SDValue> &InVals) {

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(), ArgLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallOperands(Outs,
                             CCAssignFnForNode(CallConv, /* Return*/ false,
                                               isVarArg));

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));

  SDValue StackPtr = DAG.getRegister(ARM::SP, MVT::i32);

  RegsToPassVector RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // Walk the register/memloc assignments, inserting copies/loads.  In the case
  // of tail call optimization, arguments are handled later.
  for (unsigned i = 0, realArgIdx = 0, e = ArgLocs.size();
       i != e;
       ++i, ++realArgIdx) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = Outs[realArgIdx].Val;
    ISD::ArgFlagsTy Flags = Outs[realArgIdx].Flags;

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BIT_CONVERT, dl, VA.getLocVT(), Arg);
      break;
    }

    // f64 and v2f64 might be passed in i32 pairs and must be split into pieces
    if (VA.needsCustom()) {
      if (VA.getLocVT() == MVT::v2f64) {
        SDValue Op0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64, Arg,
                                  DAG.getConstant(0, MVT::i32));
        SDValue Op1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64, Arg,
                                  DAG.getConstant(1, MVT::i32));

        PassF64ArgInRegs(dl, DAG, Chain, Op0, RegsToPass,
                         VA, ArgLocs[++i], StackPtr, MemOpChains, Flags);

        VA = ArgLocs[++i]; // skip ahead to next loc
        if (VA.isRegLoc()) {
          PassF64ArgInRegs(dl, DAG, Chain, Op1, RegsToPass,
                           VA, ArgLocs[++i], StackPtr, MemOpChains, Flags);
        } else {
          assert(VA.isMemLoc());
          if (StackPtr.getNode() == 0)
            StackPtr = DAG.getCopyFromReg(Chain, dl, ARM::SP, getPointerTy());

          MemOpChains.push_back(LowerMemOpCallTo(Chain, StackPtr, Op1,
                                                 dl, DAG, VA, Flags));
        }
      } else {
        PassF64ArgInRegs(dl, DAG, Chain, Arg, RegsToPass, VA, ArgLocs[++i],
                         StackPtr, MemOpChains, Flags);
      }
    } else if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());
      if (StackPtr.getNode() == 0)
        StackPtr = DAG.getCopyFromReg(Chain, dl, ARM::SP, getPointerTy());

      MemOpChains.push_back(LowerMemOpCallTo(Chain, StackPtr, Arg,
                                             dl, DAG, VA, Flags));
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
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
    bool isExt = GV->isDeclaration() || GV->isWeakForLinker();
    bool isStub = (isExt && Subtarget->isTargetDarwin()) &&
                   getTargetMachine().getRelocationModel() != Reloc::Static;
    isARMFunc = !Subtarget->isThumb() || isStub;
    // ARM call to a local ARM function is predicable.
    isLocalARMFunc = !Subtarget->isThumb() && !isExt;
    // tBX takes a register source operand.
    if (isARMFunc && Subtarget->isThumb1Only() && !Subtarget->hasV5TOps()) {
      ARMConstantPoolValue *CPV = new ARMConstantPoolValue(GV,
                                                           ARMPCLabelIndex, 4);
      SDValue CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 4);
      CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), dl,
                           DAG.getEntryNode(), CPAddr, NULL, 0);
      SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
      Callee = DAG.getNode(ARMISD::PIC_ADD, dl,
                           getPointerTy(), Callee, PICLabel);
   } else
      Callee = DAG.getTargetGlobalAddress(GV, getPointerTy());
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    isDirect = true;
    bool isStub = Subtarget->isTargetDarwin() &&
                  getTargetMachine().getRelocationModel() != Reloc::Static;
    isARMFunc = !Subtarget->isThumb() || isStub;
    // tBX takes a register source operand.
    const char *Sym = S->getSymbol();
    if (isARMFunc && Subtarget->isThumb1Only() && !Subtarget->hasV5TOps()) {
      ARMConstantPoolValue *CPV = new ARMConstantPoolValue(*DAG.getContext(),
                                                       Sym, ARMPCLabelIndex, 4);
      SDValue CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 4);
      CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), dl,
                           DAG.getEntryNode(), CPAddr, NULL, 0);
      SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
      Callee = DAG.getNode(ARMISD::PIC_ADD, dl,
                           getPointerTy(), Callee, PICLabel);
    } else
      Callee = DAG.getTargetExternalSymbol(Sym, getPointerTy());
  }

  // FIXME: handle tail calls differently.
  unsigned CallOpc;
  if (Subtarget->isThumb()) {
    if ((!isDirect || isARMFunc) && !Subtarget->hasV5TOps())
      CallOpc = ARMISD::CALL_NOLINK;
    else
      CallOpc = isARMFunc ? ARMISD::CALL : ARMISD::tCALL;
  } else {
    CallOpc = (isDirect || Subtarget->hasV5TOps())
      ? (isLocalARMFunc ? ARMISD::CALL_PRED : ARMISD::CALL)
      : ARMISD::CALL_NOLINK;
  }
  if (CallOpc == ARMISD::CALL_NOLINK && !Subtarget->isThumb1Only()) {
    // implicit def LR - LR mustn't be allocated as GRP:$dst of CALL_NOLINK
    Chain = DAG.getCopyToReg(Chain, dl, ARM::LR, DAG.getUNDEF(MVT::i32),InFlag);
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
  Chain = DAG.getNode(CallOpc, dl, DAG.getVTList(MVT::Other, MVT::Flag),
                      &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg, Ins,
                         dl, DAG, InVals);
}

SDValue
ARMTargetLowering::LowerReturn(SDValue Chain,
                               unsigned CallConv, bool isVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               DebugLoc dl, SelectionDAG &DAG) {

  // CCValAssign - represent the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slots.
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(), RVLocs,
                 *DAG.getContext());

  // Analyze outgoing return values.
  CCInfo.AnalyzeReturn(Outs, CCAssignFnForNode(CallConv, /* Return */ true,
                                               isVarArg));

  // If this is the first return lowered for this function, add
  // the regs to the liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0, realRVLocIdx = 0;
       i != RVLocs.size();
       ++i, ++realRVLocIdx) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    SDValue Arg = Outs[realRVLocIdx].Val;

    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BIT_CONVERT, dl, VA.getLocVT(), Arg);
      break;
    }

    if (VA.needsCustom()) {
      if (VA.getLocVT() == MVT::v2f64) {
        // Extract the first half and return it in two registers.
        SDValue Half = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64, Arg,
                                   DAG.getConstant(0, MVT::i32));
        SDValue HalfGPRs = DAG.getNode(ARMISD::FMRRD, dl,
                                       DAG.getVTList(MVT::i32, MVT::i32), Half);

        Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), HalfGPRs, Flag);
        Flag = Chain.getValue(1);
        VA = RVLocs[++i]; // skip ahead to next loc
        Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(),
                                 HalfGPRs.getValue(1), Flag);
        Flag = Chain.getValue(1);
        VA = RVLocs[++i]; // skip ahead to next loc

        // Extract the 2nd half and fall through to handle it as an f64 value.
        Arg = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64, Arg,
                          DAG.getConstant(1, MVT::i32));
      }
      // Legalize ret f64 -> ret 2 x i32.  We always have fmrrd if f64 is
      // available.
      SDValue fmrrd = DAG.getNode(ARMISD::FMRRD, dl,
                                  DAG.getVTList(MVT::i32, MVT::i32), &Arg, 1);
      Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), fmrrd, Flag);
      Flag = Chain.getValue(1);
      VA = RVLocs[++i]; // skip ahead to next loc
      Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), fmrrd.getValue(1),
                               Flag);
    } else
      Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), Arg, Flag);

    // Guarantee that all emitted copies are
    // stuck together, avoiding something bad.
    Flag = Chain.getValue(1);
  }

  SDValue result;
  if (Flag.getNode())
    result = DAG.getNode(ARMISD::RET_FLAG, dl, MVT::Other, Chain, Flag);
  else // Return Void
    result = DAG.getNode(ARMISD::RET_FLAG, dl, MVT::Other, Chain);

  return result;
}

// ConstantPool, JumpTable, GlobalAddress, and ExternalSymbol are lowered as
// their target counterpart wrapped in the ARMISD::Wrapper node. Suppose N is
// one of the above mentioned nodes. It has to be wrapped because otherwise
// Select(N) returns N. So the raw TargetGlobalAddress nodes, etc. can only
// be used to form addressing mode. These wrapped nodes will be selected
// into MOVi.
static SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) {
  EVT PtrVT = Op.getValueType();
  // FIXME there is no actual debug info here
  DebugLoc dl = Op.getDebugLoc();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  SDValue Res;
  if (CP->isMachineConstantPoolEntry())
    Res = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
                                    CP->getAlignment());
  else
    Res = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
                                    CP->getAlignment());
  return DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, Res);
}

// Lower ISD::GlobalTLSAddress using the "general dynamic" model
SDValue
ARMTargetLowering::LowerToTLSGeneralDynamicModel(GlobalAddressSDNode *GA,
                                                 SelectionDAG &DAG) {
  DebugLoc dl = GA->getDebugLoc();
  EVT PtrVT = getPointerTy();
  unsigned char PCAdj = Subtarget->isThumb() ? 4 : 8;
  ARMConstantPoolValue *CPV =
    new ARMConstantPoolValue(GA->getGlobal(), ARMPCLabelIndex,
                             PCAdj, "tlsgd", true);
  SDValue Argument = DAG.getTargetConstantPool(CPV, PtrVT, 4);
  Argument = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, Argument);
  Argument = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), Argument, NULL, 0);
  SDValue Chain = Argument.getValue(1);

  SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
  Argument = DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Argument, PICLabel);

  // call __tls_get_addr.
  ArgListTy Args;
  ArgListEntry Entry;
  Entry.Node = Argument;
  Entry.Ty = (const Type *) Type::getInt32Ty(*DAG.getContext());
  Args.push_back(Entry);
  // FIXME: is there useful debug info available here?
  std::pair<SDValue, SDValue> CallResult =
    LowerCallTo(Chain, (const Type *) Type::getInt32Ty(*DAG.getContext()),
                false, false, false, false,
                0, CallingConv::C, false, /*isReturnValueUsed=*/true,
                DAG.getExternalSymbol("__tls_get_addr", PtrVT), Args, DAG, dl);
  return CallResult.first;
}

// Lower ISD::GlobalTLSAddress using the "initial exec" or
// "local exec" model.
SDValue
ARMTargetLowering::LowerToTLSExecModels(GlobalAddressSDNode *GA,
                                        SelectionDAG &DAG) {
  GlobalValue *GV = GA->getGlobal();
  DebugLoc dl = GA->getDebugLoc();
  SDValue Offset;
  SDValue Chain = DAG.getEntryNode();
  EVT PtrVT = getPointerTy();
  // Get the Thread Pointer
  SDValue ThreadPointer = DAG.getNode(ARMISD::THREAD_POINTER, dl, PtrVT);

  if (GV->isDeclaration()) {
    // initial exec model
    unsigned char PCAdj = Subtarget->isThumb() ? 4 : 8;
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GA->getGlobal(), ARMPCLabelIndex,
                               PCAdj, "gottpoff", true);
    Offset = DAG.getTargetConstantPool(CPV, PtrVT, 4);
    Offset = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, Offset);
    Offset = DAG.getLoad(PtrVT, dl, Chain, Offset, NULL, 0);
    Chain = Offset.getValue(1);

    SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
    Offset = DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Offset, PICLabel);

    Offset = DAG.getLoad(PtrVT, dl, Chain, Offset, NULL, 0);
  } else {
    // local exec model
    ARMConstantPoolValue *CPV = new ARMConstantPoolValue(GV, "tpoff");
    Offset = DAG.getTargetConstantPool(CPV, PtrVT, 4);
    Offset = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, Offset);
    Offset = DAG.getLoad(PtrVT, dl, Chain, Offset, NULL, 0);
  }

  // The address of the thread local variable is the add of the thread
  // pointer with the offset of the variable.
  return DAG.getNode(ISD::ADD, dl, PtrVT, ThreadPointer, Offset);
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
  EVT PtrVT = getPointerTy();
  DebugLoc dl = Op.getDebugLoc();
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();
  if (RelocM == Reloc::PIC_) {
    bool UseGOTOFF = GV->hasLocalLinkage() || GV->hasHiddenVisibility();
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GV, UseGOTOFF ? "GOTOFF" : "GOT");
    SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
    CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
    SDValue Result = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(),
                                 CPAddr, NULL, 0);
    SDValue Chain = Result.getValue(1);
    SDValue GOT = DAG.getGLOBAL_OFFSET_TABLE(PtrVT);
    Result = DAG.getNode(ISD::ADD, dl, PtrVT, Result, GOT);
    if (!UseGOTOFF)
      Result = DAG.getLoad(PtrVT, dl, Chain, Result, NULL, 0);
    return Result;
  } else {
    SDValue CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 4);
    CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
    return DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), CPAddr, NULL, 0);
  }
}

SDValue ARMTargetLowering::LowerGlobalAddressDarwin(SDValue Op,
                                                    SelectionDAG &DAG) {
  EVT PtrVT = getPointerTy();
  DebugLoc dl = Op.getDebugLoc();
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();
  SDValue CPAddr;
  if (RelocM == Reloc::Static)
    CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 4);
  else {
    unsigned PCAdj = (RelocM != Reloc::PIC_) ? 0 : (Subtarget->isThumb()?4:8);
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GV, ARMPCLabelIndex, PCAdj);
    CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
  }
  CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);

  SDValue Result = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), CPAddr, NULL, 0);
  SDValue Chain = Result.getValue(1);

  if (RelocM == Reloc::PIC_) {
    SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
    Result = DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Result, PICLabel);
  }

  if (Subtarget->GVIsIndirectSymbol(GV, RelocM == Reloc::Static))
    Result = DAG.getLoad(PtrVT, dl, Chain, Result, NULL, 0);

  return Result;
}

SDValue ARMTargetLowering::LowerGLOBAL_OFFSET_TABLE(SDValue Op,
                                                    SelectionDAG &DAG){
  assert(Subtarget->isTargetELF() &&
         "GLOBAL OFFSET TABLE not implemented for non-ELF targets");
  EVT PtrVT = getPointerTy();
  DebugLoc dl = Op.getDebugLoc();
  unsigned PCAdj = Subtarget->isThumb() ? 4 : 8;
  ARMConstantPoolValue *CPV = new ARMConstantPoolValue(*DAG.getContext(),
                                                       "_GLOBAL_OFFSET_TABLE_",
                                                       ARMPCLabelIndex, PCAdj);
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
  CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
  SDValue Result = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), CPAddr, NULL, 0);
  SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
  return DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Result, PICLabel);
}

static SDValue LowerNeonVLDIntrinsic(SDValue Op, SelectionDAG &DAG,
                                     unsigned Opcode) {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getValueType(0);
  DebugLoc dl = Op.getDebugLoc();

  if (!VT.is64BitVector())
    return SDValue(); // unimplemented

  SDValue Ops[] = { Node->getOperand(0),
                    Node->getOperand(2) };
  return DAG.getNode(Opcode, dl, Node->getVTList(), Ops, 2);
}

static SDValue LowerNeonVSTIntrinsic(SDValue Op, SelectionDAG &DAG,
                                     unsigned Opcode, unsigned NumVecs) {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getOperand(3).getValueType();
  DebugLoc dl = Op.getDebugLoc();

  if (!VT.is64BitVector())
    return SDValue(); // unimplemented

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(Node->getOperand(0));
  Ops.push_back(Node->getOperand(2));
  for (unsigned N = 0; N < NumVecs; ++N)
    Ops.push_back(Node->getOperand(N + 3));
  return DAG.getNode(Opcode, dl, MVT::Other, Ops.data(), Ops.size());
}

SDValue
ARMTargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  switch (IntNo) {
  case Intrinsic::arm_neon_vld2:
    return LowerNeonVLDIntrinsic(Op, DAG, ARMISD::VLD2D);
  case Intrinsic::arm_neon_vld3:
    return LowerNeonVLDIntrinsic(Op, DAG, ARMISD::VLD3D);
  case Intrinsic::arm_neon_vld4:
    return LowerNeonVLDIntrinsic(Op, DAG, ARMISD::VLD4D);
  case Intrinsic::arm_neon_vst2:
    return LowerNeonVSTIntrinsic(Op, DAG, ARMISD::VST2D, 2);
  case Intrinsic::arm_neon_vst3:
    return LowerNeonVSTIntrinsic(Op, DAG, ARMISD::VST3D, 3);
  case Intrinsic::arm_neon_vst4:
    return LowerNeonVSTIntrinsic(Op, DAG, ARMISD::VST4D, 4);
  default: return SDValue();    // Don't custom lower most intrinsics.
  }
}

SDValue
ARMTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  DebugLoc dl = Op.getDebugLoc();
  switch (IntNo) {
  default: return SDValue();    // Don't custom lower most intrinsics.
  case Intrinsic::arm_thread_pointer: {
    EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
    return DAG.getNode(ARMISD::THREAD_POINTER, dl, PtrVT);
  }
  case Intrinsic::eh_sjlj_lsda: {
    // blah. horrible, horrible hack with the forced magic name.
    // really need to clean this up. It belongs in the target-independent
    // layer somehow that doesn't require the coupling with the asm
    // printer.
    MachineFunction &MF = DAG.getMachineFunction();
    EVT PtrVT = getPointerTy();
    DebugLoc dl = Op.getDebugLoc();
    Reloc::Model RelocM = getTargetMachine().getRelocationModel();
    SDValue CPAddr;
    unsigned PCAdj = (RelocM != Reloc::PIC_)
      ? 0 : (Subtarget->isThumb() ? 4 : 8);
    // Save off the LSDA name for the AsmPrinter to use when it's time
    // to emit the table
    std::string LSDAName = "L_lsda_";
    LSDAName += MF.getFunction()->getName();
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(*DAG.getContext(), LSDAName.c_str(), 
                               ARMPCLabelIndex, PCAdj);
    CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
    CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
    SDValue Result =
      DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), CPAddr, NULL, 0);
    SDValue Chain = Result.getValue(1);

    if (RelocM == Reloc::PIC_) {
      SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
      Result = DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Result, PICLabel);
    }
    return Result;
  }
  case Intrinsic::eh_sjlj_setjmp:
    return DAG.getNode(ARMISD::EH_SJLJ_SETJMP, dl, MVT::i32, Op.getOperand(1));
  }
}

static SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG,
                            unsigned VarArgsFrameIndex) {
  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  DebugLoc dl = Op.getDebugLoc();
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  SDValue FR = DAG.getFrameIndex(VarArgsFrameIndex, PtrVT);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), dl, FR, Op.getOperand(1), SV, 0);
}

SDValue
ARMTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) {
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  EVT VT = Node->getValueType(0);
  SDValue Chain = Op.getOperand(0);
  SDValue Size  = Op.getOperand(1);
  SDValue Align = Op.getOperand(2);

  // Chain the dynamic stack allocation so that it doesn't modify the stack
  // pointer when other instructions are using the stack.
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(0, true));

  unsigned AlignVal = cast<ConstantSDNode>(Align)->getZExtValue();
  unsigned StackAlign = getTargetMachine().getFrameInfo()->getStackAlignment();
  if (AlignVal > StackAlign)
    // Do this now since selection pass cannot introduce new target
    // independent node.
    Align = DAG.getConstant(-(uint64_t)AlignVal, VT);

  // In Thumb1 mode, there isn't a "sub r, sp, r" instruction, we will end up
  // using a "add r, sp, r" instead. Negate the size now so we don't have to
  // do even more horrible hack later.
  MachineFunction &MF = DAG.getMachineFunction();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  if (AFI->isThumb1OnlyFunction()) {
    bool Negate = true;
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Size);
    if (C) {
      uint32_t Val = C->getZExtValue();
      if (Val <= 508 && ((Val & 3) == 0))
        Negate = false;
    }
    if (Negate)
      Size = DAG.getNode(ISD::SUB, dl, VT, DAG.getConstant(0, VT), Size);
  }

  SDVTList VTList = DAG.getVTList(VT, MVT::Other);
  SDValue Ops1[] = { Chain, Size, Align };
  SDValue Res = DAG.getNode(ARMISD::DYN_ALLOC, dl, VTList, Ops1, 3);
  Chain = Res.getValue(1);
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(0, true),
                             DAG.getIntPtrConstant(0, true), SDValue());
  SDValue Ops2[] = { Res, Chain };
  return DAG.getMergeValues(Ops2, 2, dl);
}

SDValue
ARMTargetLowering::GetF64FormalArgument(CCValAssign &VA, CCValAssign &NextVA,
                                        SDValue &Root, SelectionDAG &DAG,
                                        DebugLoc dl) {
  MachineFunction &MF = DAG.getMachineFunction();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  TargetRegisterClass *RC;
  if (AFI->isThumb1OnlyFunction())
    RC = ARM::tGPRRegisterClass;
  else
    RC = ARM::GPRRegisterClass;

  // Transform the arguments stored in physical registers into virtual ones.
  unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
  SDValue ArgValue = DAG.getCopyFromReg(Root, dl, Reg, MVT::i32);

  SDValue ArgValue2;
  if (NextVA.isMemLoc()) {
    unsigned ArgSize = NextVA.getLocVT().getSizeInBits()/8;
    MachineFrameInfo *MFI = MF.getFrameInfo();
    int FI = MFI->CreateFixedObject(ArgSize, NextVA.getLocMemOffset());

    // Create load node to retrieve arguments from the stack.
    SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
    ArgValue2 = DAG.getLoad(MVT::i32, dl, Root, FIN, NULL, 0);
  } else {
    Reg = MF.addLiveIn(NextVA.getLocReg(), RC);
    ArgValue2 = DAG.getCopyFromReg(Root, dl, Reg, MVT::i32);
  }

  return DAG.getNode(ARMISD::FMDRR, dl, MVT::f64, ArgValue, ArgValue2);
}

SDValue
ARMTargetLowering::LowerFormalArguments(SDValue Chain,
                                        unsigned CallConv, bool isVarArg,
                                        const SmallVectorImpl<ISD::InputArg>
                                          &Ins,
                                        DebugLoc dl, SelectionDAG &DAG,
                                        SmallVectorImpl<SDValue> &InVals) {

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(), ArgLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins,
                                CCAssignFnForNode(CallConv, /* Return*/ false,
                                                  isVarArg));

  SmallVector<SDValue, 16> ArgValues;

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    // Arguments stored in registers.
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();

      SDValue ArgValue;
      if (VA.needsCustom()) {
        // f64 and vector types are split up into multiple registers or
        // combinations of registers and stack slots.
        RegVT = MVT::i32;

        if (VA.getLocVT() == MVT::v2f64) {
          SDValue ArgValue1 = GetF64FormalArgument(VA, ArgLocs[++i],
                                                   Chain, DAG, dl);
          VA = ArgLocs[++i]; // skip ahead to next loc
          SDValue ArgValue2 = GetF64FormalArgument(VA, ArgLocs[++i],
                                                   Chain, DAG, dl);
          ArgValue = DAG.getNode(ISD::UNDEF, dl, MVT::v2f64);
          ArgValue = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64,
                                 ArgValue, ArgValue1, DAG.getIntPtrConstant(0));
          ArgValue = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64,
                                 ArgValue, ArgValue2, DAG.getIntPtrConstant(1));
        } else
          ArgValue = GetF64FormalArgument(VA, ArgLocs[++i], Chain, DAG, dl);

      } else {
        TargetRegisterClass *RC;

        if (RegVT == MVT::f32)
          RC = ARM::SPRRegisterClass;
        else if (RegVT == MVT::f64)
          RC = ARM::DPRRegisterClass;
        else if (RegVT == MVT::v2f64)
          RC = ARM::QPRRegisterClass;
        else if (RegVT == MVT::i32)
          RC = (AFI->isThumb1OnlyFunction() ?
                ARM::tGPRRegisterClass : ARM::GPRRegisterClass);
        else
          llvm_unreachable("RegVT not supported by FORMAL_ARGUMENTS Lowering");

        // Transform the arguments in physical registers into virtual ones.
        unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
        ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);
      }

      // If this is an 8 or 16-bit value, it is really passed promoted
      // to 32 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      switch (VA.getLocInfo()) {
      default: llvm_unreachable("Unknown loc info!");
      case CCValAssign::Full: break;
      case CCValAssign::BCvt:
        ArgValue = DAG.getNode(ISD::BIT_CONVERT, dl, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::SExt:
        ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::ZExt:
        ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
        break;
      }

      InVals.push_back(ArgValue);

    } else { // VA.isRegLoc()

      // sanity check
      assert(VA.isMemLoc());
      assert(VA.getValVT() != MVT::i64 && "i64 should already be lowered");

      unsigned ArgSize = VA.getLocVT().getSizeInBits()/8;
      int FI = MFI->CreateFixedObject(ArgSize, VA.getLocMemOffset());

      // Create load nodes to retrieve arguments from the stack.
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
      InVals.push_back(DAG.getLoad(VA.getValVT(), dl, Chain, FIN, NULL, 0));
    }
  }

  // varargs
  if (isVarArg) {
    static const unsigned GPRArgRegs[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3
    };

    unsigned NumGPRs = CCInfo.getFirstUnallocated
      (GPRArgRegs, sizeof(GPRArgRegs) / sizeof(GPRArgRegs[0]));

    unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
    unsigned VARegSize = (4 - NumGPRs) * 4;
    unsigned VARegSaveSize = (VARegSize + Align - 1) & ~(Align - 1);
    unsigned ArgOffset = 0;
    if (VARegSaveSize) {
      // If this function is vararg, store any remaining integer argument regs
      // to their spots on the stack so that they may be loaded by deferencing
      // the result of va_next.
      AFI->setVarArgsRegSaveSize(VARegSaveSize);
      ArgOffset = CCInfo.getNextStackOffset();
      VarArgsFrameIndex = MFI->CreateFixedObject(VARegSaveSize, ArgOffset +
                                                 VARegSaveSize - VARegSize);
      SDValue FIN = DAG.getFrameIndex(VarArgsFrameIndex, getPointerTy());

      SmallVector<SDValue, 4> MemOps;
      for (; NumGPRs < 4; ++NumGPRs) {
        TargetRegisterClass *RC;
        if (AFI->isThumb1OnlyFunction())
          RC = ARM::tGPRRegisterClass;
        else
          RC = ARM::GPRRegisterClass;

        unsigned VReg = MF.addLiveIn(GPRArgRegs[NumGPRs], RC);
        SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i32);
        SDValue Store = DAG.getStore(Val.getValue(1), dl, Val, FIN, NULL, 0);
        MemOps.push_back(Store);
        FIN = DAG.getNode(ISD::ADD, dl, getPointerTy(), FIN,
                          DAG.getConstant(4, getPointerTy()));
      }
      if (!MemOps.empty())
        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                            &MemOps[0], MemOps.size());
    } else
      // This will point to the next argument passed via stack.
      VarArgsFrameIndex = MFI->CreateFixedObject(4, ArgOffset);
  }

  return Chain;
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

static bool isLegalCmpImmediate(unsigned C, bool isThumb1Only) {
  return ( isThumb1Only && (C & ~255U) == 0) ||
         (!isThumb1Only && ARM_AM::getSOImmVal(C) != -1);
}

/// Returns appropriate ARM CMP (cmp) and corresponding condition code for
/// the given operands.
static SDValue getARMCmp(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                         SDValue &ARMCC, SelectionDAG &DAG, bool isThumb1Only,
                         DebugLoc dl) {
  if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS.getNode())) {
    unsigned C = RHSC->getZExtValue();
    if (!isLegalCmpImmediate(C, isThumb1Only)) {
      // Constant does not fit, try adjusting it by one?
      switch (CC) {
      default: break;
      case ISD::SETLT:
      case ISD::SETGE:
        if (isLegalCmpImmediate(C-1, isThumb1Only)) {
          CC = (CC == ISD::SETLT) ? ISD::SETLE : ISD::SETGT;
          RHS = DAG.getConstant(C-1, MVT::i32);
        }
        break;
      case ISD::SETULT:
      case ISD::SETUGE:
        if (C > 0 && isLegalCmpImmediate(C-1, isThumb1Only)) {
          CC = (CC == ISD::SETULT) ? ISD::SETULE : ISD::SETUGT;
          RHS = DAG.getConstant(C-1, MVT::i32);
        }
        break;
      case ISD::SETLE:
      case ISD::SETGT:
        if (isLegalCmpImmediate(C+1, isThumb1Only)) {
          CC = (CC == ISD::SETLE) ? ISD::SETLT : ISD::SETGE;
          RHS = DAG.getConstant(C+1, MVT::i32);
        }
        break;
      case ISD::SETULE:
      case ISD::SETUGT:
        if (C < 0xffffffff && isLegalCmpImmediate(C+1, isThumb1Only)) {
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
    // Uses only Z Flag
    CompareType = ARMISD::CMPZ;
    break;
  }
  ARMCC = DAG.getConstant(CondCode, MVT::i32);
  return DAG.getNode(CompareType, dl, MVT::Flag, LHS, RHS);
}

/// Returns a appropriate VFP CMP (fcmp{s|d}+fmstat) for the given operands.
static SDValue getVFPCmp(SDValue LHS, SDValue RHS, SelectionDAG &DAG,
                         DebugLoc dl) {
  SDValue Cmp;
  if (!isFloatingPointZero(RHS))
    Cmp = DAG.getNode(ARMISD::CMPFP, dl, MVT::Flag, LHS, RHS);
  else
    Cmp = DAG.getNode(ARMISD::CMPFPw0, dl, MVT::Flag, LHS);
  return DAG.getNode(ARMISD::FMSTAT, dl, MVT::Flag, Cmp);
}

static SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG,
                              const ARMSubtarget *ST) {
  EVT VT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);
  DebugLoc dl = Op.getDebugLoc();

  if (LHS.getValueType() == MVT::i32) {
    SDValue ARMCC;
    SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
    SDValue Cmp = getARMCmp(LHS, RHS, CC, ARMCC, DAG, ST->isThumb1Only(), dl);
    return DAG.getNode(ARMISD::CMOV, dl, VT, FalseVal, TrueVal, ARMCC, CCR,Cmp);
  }

  ARMCC::CondCodes CondCode, CondCode2;
  if (FPCCToARMCC(CC, CondCode, CondCode2))
    std::swap(TrueVal, FalseVal);

  SDValue ARMCC = DAG.getConstant(CondCode, MVT::i32);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  SDValue Cmp = getVFPCmp(LHS, RHS, DAG, dl);
  SDValue Result = DAG.getNode(ARMISD::CMOV, dl, VT, FalseVal, TrueVal,
                                 ARMCC, CCR, Cmp);
  if (CondCode2 != ARMCC::AL) {
    SDValue ARMCC2 = DAG.getConstant(CondCode2, MVT::i32);
    // FIXME: Needs another CMP because flag can have but one use.
    SDValue Cmp2 = getVFPCmp(LHS, RHS, DAG, dl);
    Result = DAG.getNode(ARMISD::CMOV, dl, VT,
                         Result, TrueVal, ARMCC2, CCR, Cmp2);
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
  DebugLoc dl = Op.getDebugLoc();

  if (LHS.getValueType() == MVT::i32) {
    SDValue ARMCC;
    SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
    SDValue Cmp = getARMCmp(LHS, RHS, CC, ARMCC, DAG, ST->isThumb1Only(), dl);
    return DAG.getNode(ARMISD::BRCOND, dl, MVT::Other,
                       Chain, Dest, ARMCC, CCR,Cmp);
  }

  assert(LHS.getValueType() == MVT::f32 || LHS.getValueType() == MVT::f64);
  ARMCC::CondCodes CondCode, CondCode2;
  if (FPCCToARMCC(CC, CondCode, CondCode2))
    // Swap the LHS/RHS of the comparison if needed.
    std::swap(LHS, RHS);

  SDValue Cmp = getVFPCmp(LHS, RHS, DAG, dl);
  SDValue ARMCC = DAG.getConstant(CondCode, MVT::i32);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  SDVTList VTList = DAG.getVTList(MVT::Other, MVT::Flag);
  SDValue Ops[] = { Chain, Dest, ARMCC, CCR, Cmp };
  SDValue Res = DAG.getNode(ARMISD::BRCOND, dl, VTList, Ops, 5);
  if (CondCode2 != ARMCC::AL) {
    ARMCC = DAG.getConstant(CondCode2, MVT::i32);
    SDValue Ops[] = { Res, Dest, ARMCC, CCR, Res.getValue(1) };
    Res = DAG.getNode(ARMISD::BRCOND, dl, VTList, Ops, 5);
  }
  return Res;
}

SDValue ARMTargetLowering::LowerBR_JT(SDValue Op, SelectionDAG &DAG) {
  SDValue Chain = Op.getOperand(0);
  SDValue Table = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);
  DebugLoc dl = Op.getDebugLoc();

  EVT PTy = getPointerTy();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Table);
  ARMFunctionInfo *AFI = DAG.getMachineFunction().getInfo<ARMFunctionInfo>();
  SDValue UId = DAG.getConstant(AFI->createJumpTableUId(), PTy);
  SDValue JTI = DAG.getTargetJumpTable(JT->getIndex(), PTy);
  Table = DAG.getNode(ARMISD::WrapperJT, dl, MVT::i32, JTI, UId);
  Index = DAG.getNode(ISD::MUL, dl, PTy, Index, DAG.getConstant(4, PTy));
  SDValue Addr = DAG.getNode(ISD::ADD, dl, PTy, Index, Table);
  if (Subtarget->isThumb2()) {
    // Thumb2 uses a two-level jump. That is, it jumps into the jump table
    // which does another jump to the destination. This also makes it easier
    // to translate it to TBB / TBH later.
    // FIXME: This might not work if the function is extremely large.
    return DAG.getNode(ARMISD::BR2_JT, dl, MVT::Other, Chain,
                       Addr, Op.getOperand(2), JTI, UId);
  }
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_) {
    Addr = DAG.getLoad((EVT)MVT::i32, dl, Chain, Addr, NULL, 0);
    Chain = Addr.getValue(1);
    Addr = DAG.getNode(ISD::ADD, dl, PTy, Addr, Table);
    return DAG.getNode(ARMISD::BR_JT, dl, MVT::Other, Chain, Addr, JTI, UId);
  } else {
    Addr = DAG.getLoad(PTy, dl, Chain, Addr, NULL, 0);
    Chain = Addr.getValue(1);
    return DAG.getNode(ARMISD::BR_JT, dl, MVT::Other, Chain, Addr, JTI, UId);
  }
}

static SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  unsigned Opc =
    Op.getOpcode() == ISD::FP_TO_SINT ? ARMISD::FTOSI : ARMISD::FTOUI;
  Op = DAG.getNode(Opc, dl, MVT::f32, Op.getOperand(0));
  return DAG.getNode(ISD::BIT_CONVERT, dl, MVT::i32, Op);
}

static SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned Opc =
    Op.getOpcode() == ISD::SINT_TO_FP ? ARMISD::SITOF : ARMISD::UITOF;

  Op = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::f32, Op.getOperand(0));
  return DAG.getNode(Opc, dl, VT, Op);
}

static SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) {
  // Implement fcopysign with a fabs and a conditional fneg.
  SDValue Tmp0 = Op.getOperand(0);
  SDValue Tmp1 = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT SrcVT = Tmp1.getValueType();
  SDValue AbsVal = DAG.getNode(ISD::FABS, dl, VT, Tmp0);
  SDValue Cmp = getVFPCmp(Tmp1, DAG.getConstantFP(0.0, SrcVT), DAG, dl);
  SDValue ARMCC = DAG.getConstant(ARMCC::LT, MVT::i32);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  return DAG.getNode(ARMISD::CNEG, dl, VT, AbsVal, AbsVal, ARMCC, CCR, Cmp);
}

SDValue ARMTargetLowering::LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setFrameAddressIsTaken(true);
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();  // FIXME probably not meaningful
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  unsigned FrameReg = (Subtarget->isThumb() || Subtarget->isTargetDarwin())
    ? ARM::R7 : ARM::R11;
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl, FrameReg, VT);
  while (Depth--)
    FrameAddr = DAG.getLoad(VT, dl, DAG.getEntryNode(), FrameAddr, NULL, 0);
  return FrameAddr;
}

SDValue
ARMTargetLowering::EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
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
  EVT VT = MVT::i32;
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
      Loads[i] = DAG.getLoad(VT, dl, Chain,
                             DAG.getNode(ISD::ADD, dl, MVT::i32, Src,
                                         DAG.getConstant(SrcOff, MVT::i32)),
                             SrcSV, SrcSVOff + SrcOff);
      TFOps[i] = Loads[i].getValue(1);
      SrcOff += VTSize;
    }
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

    for (i = 0;
         i < MAX_LOADS_IN_LDM && EmittedNumMemOps + i < NumMemOps; ++i) {
      TFOps[i] = DAG.getStore(Chain, dl, Loads[i],
                           DAG.getNode(ISD::ADD, dl, MVT::i32, Dst,
                                       DAG.getConstant(DstOff, MVT::i32)),
                           DstSV, DstSVOff + DstOff);
      DstOff += VTSize;
    }
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

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

    Loads[i] = DAG.getLoad(VT, dl, Chain,
                           DAG.getNode(ISD::ADD, dl, MVT::i32, Src,
                                       DAG.getConstant(SrcOff, MVT::i32)),
                           SrcSV, SrcSVOff + SrcOff);
    TFOps[i] = Loads[i].getValue(1);
    ++i;
    SrcOff += VTSize;
    BytesLeft -= VTSize;
  }
  Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

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

    TFOps[i] = DAG.getStore(Chain, dl, Loads[i],
                            DAG.getNode(ISD::ADD, dl, MVT::i32, Dst,
                                        DAG.getConstant(DstOff, MVT::i32)),
                            DstSV, DstSVOff + DstOff);
    ++i;
    DstOff += VTSize;
    BytesLeft -= VTSize;
  }
  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);
}

static SDValue ExpandBIT_CONVERT(SDNode *N, SelectionDAG &DAG) {
  SDValue Op = N->getOperand(0);
  DebugLoc dl = N->getDebugLoc();
  if (N->getValueType(0) == MVT::f64) {
    // Turn i64->f64 into FMDRR.
    SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, Op,
                             DAG.getConstant(0, MVT::i32));
    SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, Op,
                             DAG.getConstant(1, MVT::i32));
    return DAG.getNode(ARMISD::FMDRR, dl, MVT::f64, Lo, Hi);
  }

  // Turn f64->i64 into FMRRD.
  SDValue Cvt = DAG.getNode(ARMISD::FMRRD, dl,
                            DAG.getVTList(MVT::i32, MVT::i32), &Op, 1);

  // Merge the pieces into a single i64 value.
  return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Cvt, Cvt.getValue(1));
}

/// getZeroVector - Returns a vector of specified type with all zero elements.
///
static SDValue getZeroVector(EVT VT, SelectionDAG &DAG, DebugLoc dl) {
  assert(VT.isVector() && "Expected a vector type");

  // Zero vectors are used to represent vector negation and in those cases
  // will be implemented with the NEON VNEG instruction.  However, VNEG does
  // not support i64 elements, so sometimes the zero vectors will need to be
  // explicitly constructed.  For those cases, and potentially other uses in
  // the future, always build zero vectors as <4 x i32> or <2 x i32> bitcasted
  // to their dest type.  This ensures they get CSE'd.
  SDValue Vec;
  SDValue Cst = DAG.getTargetConstant(0, MVT::i32);
  if (VT.getSizeInBits() == 64)
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v2i32, Cst, Cst);
  else
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Cst, Cst, Cst, Cst);

  return DAG.getNode(ISD::BIT_CONVERT, dl, VT, Vec);
}

/// getOnesVector - Returns a vector of specified type with all bits set.
///
static SDValue getOnesVector(EVT VT, SelectionDAG &DAG, DebugLoc dl) {
  assert(VT.isVector() && "Expected a vector type");

  // Always build ones vectors as <4 x i32> or <2 x i32> bitcasted to their dest
  // type.  This ensures they get CSE'd.
  SDValue Vec;
  SDValue Cst = DAG.getTargetConstant(~0U, MVT::i32);
  if (VT.getSizeInBits() == 64)
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v2i32, Cst, Cst);
  else
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Cst, Cst, Cst, Cst);

  return DAG.getNode(ISD::BIT_CONVERT, dl, VT, Vec);
}

static SDValue LowerShift(SDNode *N, SelectionDAG &DAG,
                          const ARMSubtarget *ST) {
  EVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  // Lower vector shifts on NEON to use VSHL.
  if (VT.isVector()) {
    assert(ST->hasNEON() && "unexpected vector shift");

    // Left shifts translate directly to the vshiftu intrinsic.
    if (N->getOpcode() == ISD::SHL)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                         DAG.getConstant(Intrinsic::arm_neon_vshiftu, MVT::i32),
                         N->getOperand(0), N->getOperand(1));

    assert((N->getOpcode() == ISD::SRA ||
            N->getOpcode() == ISD::SRL) && "unexpected vector shift opcode");

    // NEON uses the same intrinsics for both left and right shifts.  For
    // right shifts, the shift amounts are negative, so negate the vector of
    // shift amounts.
    EVT ShiftVT = N->getOperand(1).getValueType();
    SDValue NegatedCount = DAG.getNode(ISD::SUB, dl, ShiftVT,
                                       getZeroVector(ShiftVT, DAG, dl),
                                       N->getOperand(1));
    Intrinsic::ID vshiftInt = (N->getOpcode() == ISD::SRA ?
                               Intrinsic::arm_neon_vshifts :
                               Intrinsic::arm_neon_vshiftu);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(vshiftInt, MVT::i32),
                       N->getOperand(0), NegatedCount);
  }

  // We can get here for a node like i32 = ISD::SHL i32, i64
  if (VT != MVT::i64)
    return SDValue();

  assert((N->getOpcode() == ISD::SRL || N->getOpcode() == ISD::SRA) &&
         "Unknown shift to lower!");

  // We only lower SRA, SRL of 1 here, all others use generic lowering.
  if (!isa<ConstantSDNode>(N->getOperand(1)) ||
      cast<ConstantSDNode>(N->getOperand(1))->getZExtValue() != 1)
    return SDValue();

  // If we are in thumb mode, we don't have RRX.
  if (ST->isThumb1Only()) return SDValue();

  // Okay, we have a 64-bit SRA or SRL of 1.  Lower this to an RRX expr.
  SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, N->getOperand(0),
                             DAG.getConstant(0, MVT::i32));
  SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, N->getOperand(0),
                             DAG.getConstant(1, MVT::i32));

  // First, build a SRA_FLAG/SRL_FLAG op, which shifts the top part by one and
  // captures the result into a carry flag.
  unsigned Opc = N->getOpcode() == ISD::SRL ? ARMISD::SRL_FLAG:ARMISD::SRA_FLAG;
  Hi = DAG.getNode(Opc, dl, DAG.getVTList(MVT::i32, MVT::Flag), &Hi, 1);

  // The low part is an ARMISD::RRX operand, which shifts the carry in.
  Lo = DAG.getNode(ARMISD::RRX, dl, MVT::i32, Lo, Hi.getValue(1));

  // Merge the pieces into a single i64 value.
 return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Lo, Hi);
}

static SDValue LowerVSETCC(SDValue Op, SelectionDAG &DAG) {
  SDValue TmpOp0, TmpOp1;
  bool Invert = false;
  bool Swap = false;
  unsigned Opc = 0;

  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue CC = Op.getOperand(2);
  EVT VT = Op.getValueType();
  ISD::CondCode SetCCOpcode = cast<CondCodeSDNode>(CC)->get();
  DebugLoc dl = Op.getDebugLoc();

  if (Op.getOperand(1).getValueType().isFloatingPoint()) {
    switch (SetCCOpcode) {
    default: llvm_unreachable("Illegal FP comparison"); break;
    case ISD::SETUNE:
    case ISD::SETNE:  Invert = true; // Fallthrough
    case ISD::SETOEQ:
    case ISD::SETEQ:  Opc = ARMISD::VCEQ; break;
    case ISD::SETOLT:
    case ISD::SETLT: Swap = true; // Fallthrough
    case ISD::SETOGT:
    case ISD::SETGT:  Opc = ARMISD::VCGT; break;
    case ISD::SETOLE:
    case ISD::SETLE:  Swap = true; // Fallthrough
    case ISD::SETOGE:
    case ISD::SETGE: Opc = ARMISD::VCGE; break;
    case ISD::SETUGE: Swap = true; // Fallthrough
    case ISD::SETULE: Invert = true; Opc = ARMISD::VCGT; break;
    case ISD::SETUGT: Swap = true; // Fallthrough
    case ISD::SETULT: Invert = true; Opc = ARMISD::VCGE; break;
    case ISD::SETUEQ: Invert = true; // Fallthrough
    case ISD::SETONE:
      // Expand this to (OLT | OGT).
      TmpOp0 = Op0;
      TmpOp1 = Op1;
      Opc = ISD::OR;
      Op0 = DAG.getNode(ARMISD::VCGT, dl, VT, TmpOp1, TmpOp0);
      Op1 = DAG.getNode(ARMISD::VCGT, dl, VT, TmpOp0, TmpOp1);
      break;
    case ISD::SETUO: Invert = true; // Fallthrough
    case ISD::SETO:
      // Expand this to (OLT | OGE).
      TmpOp0 = Op0;
      TmpOp1 = Op1;
      Opc = ISD::OR;
      Op0 = DAG.getNode(ARMISD::VCGT, dl, VT, TmpOp1, TmpOp0);
      Op1 = DAG.getNode(ARMISD::VCGE, dl, VT, TmpOp0, TmpOp1);
      break;
    }
  } else {
    // Integer comparisons.
    switch (SetCCOpcode) {
    default: llvm_unreachable("Illegal integer comparison"); break;
    case ISD::SETNE:  Invert = true;
    case ISD::SETEQ:  Opc = ARMISD::VCEQ; break;
    case ISD::SETLT:  Swap = true;
    case ISD::SETGT:  Opc = ARMISD::VCGT; break;
    case ISD::SETLE:  Swap = true;
    case ISD::SETGE:  Opc = ARMISD::VCGE; break;
    case ISD::SETULT: Swap = true;
    case ISD::SETUGT: Opc = ARMISD::VCGTU; break;
    case ISD::SETULE: Swap = true;
    case ISD::SETUGE: Opc = ARMISD::VCGEU; break;
    }

    // Detect VTST (Vector Test Bits) = icmp ne (and (op0, op1), zero).
    if (Opc == ARMISD::VCEQ) {

      SDValue AndOp;
      if (ISD::isBuildVectorAllZeros(Op1.getNode()))
        AndOp = Op0;
      else if (ISD::isBuildVectorAllZeros(Op0.getNode()))
        AndOp = Op1;

      // Ignore bitconvert.
      if (AndOp.getNode() && AndOp.getOpcode() == ISD::BIT_CONVERT)
        AndOp = AndOp.getOperand(0);

      if (AndOp.getNode() && AndOp.getOpcode() == ISD::AND) {
        Opc = ARMISD::VTST;
        Op0 = DAG.getNode(ISD::BIT_CONVERT, dl, VT, AndOp.getOperand(0));
        Op1 = DAG.getNode(ISD::BIT_CONVERT, dl, VT, AndOp.getOperand(1));
        Invert = !Invert;
      }
    }
  }

  if (Swap)
    std::swap(Op0, Op1);

  SDValue Result = DAG.getNode(Opc, dl, VT, Op0, Op1);

  if (Invert)
    Result = DAG.getNOT(dl, Result, VT);

  return Result;
}

/// isVMOVSplat - Check if the specified splat value corresponds to an immediate
/// VMOV instruction, and if so, return the constant being splatted.
static SDValue isVMOVSplat(uint64_t SplatBits, uint64_t SplatUndef,
                           unsigned SplatBitSize, SelectionDAG &DAG) {
  switch (SplatBitSize) {
  case 8:
    // Any 1-byte value is OK.
    assert((SplatBits & ~0xff) == 0 && "one byte splat value is too big");
    return DAG.getTargetConstant(SplatBits, MVT::i8);

  case 16:
    // NEON's 16-bit VMOV supports splat values where only one byte is nonzero.
    if ((SplatBits & ~0xff) == 0 ||
        (SplatBits & ~0xff00) == 0)
      return DAG.getTargetConstant(SplatBits, MVT::i16);
    break;

  case 32:
    // NEON's 32-bit VMOV supports splat values where:
    // * only one byte is nonzero, or
    // * the least significant byte is 0xff and the second byte is nonzero, or
    // * the least significant 2 bytes are 0xff and the third is nonzero.
    if ((SplatBits & ~0xff) == 0 ||
        (SplatBits & ~0xff00) == 0 ||
        (SplatBits & ~0xff0000) == 0 ||
        (SplatBits & ~0xff000000) == 0)
      return DAG.getTargetConstant(SplatBits, MVT::i32);

    if ((SplatBits & ~0xffff) == 0 &&
        ((SplatBits | SplatUndef) & 0xff) == 0xff)
      return DAG.getTargetConstant(SplatBits | 0xff, MVT::i32);

    if ((SplatBits & ~0xffffff) == 0 &&
        ((SplatBits | SplatUndef) & 0xffff) == 0xffff)
      return DAG.getTargetConstant(SplatBits | 0xffff, MVT::i32);

    // Note: there are a few 32-bit splat values (specifically: 00ffff00,
    // ff000000, ff0000ff, and ffff00ff) that are valid for VMOV.I64 but not
    // VMOV.I32.  A (very) minor optimization would be to replicate the value
    // and fall through here to test for a valid 64-bit splat.  But, then the
    // caller would also need to check and handle the change in size.
    break;

  case 64: {
    // NEON has a 64-bit VMOV splat where each byte is either 0 or 0xff.
    uint64_t BitMask = 0xff;
    uint64_t Val = 0;
    for (int ByteNum = 0; ByteNum < 8; ++ByteNum) {
      if (((SplatBits | SplatUndef) & BitMask) == BitMask)
        Val |= BitMask;
      else if ((SplatBits & BitMask) != 0)
        return SDValue();
      BitMask <<= 8;
    }
    return DAG.getTargetConstant(Val, MVT::i64);
  }

  default:
    llvm_unreachable("unexpected size for isVMOVSplat");
    break;
  }

  return SDValue();
}

/// getVMOVImm - If this is a build_vector of constants which can be
/// formed by using a VMOV instruction of the specified element size,
/// return the constant being splatted.  The ByteSize field indicates the
/// number of bytes of each element [1248].
SDValue ARM::getVMOVImm(SDNode *N, unsigned ByteSize, SelectionDAG &DAG) {
  BuildVectorSDNode *BVN = dyn_cast<BuildVectorSDNode>(N);
  APInt SplatBits, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  if (! BVN || ! BVN->isConstantSplat(SplatBits, SplatUndef, SplatBitSize,
                                      HasAnyUndefs, ByteSize * 8))
    return SDValue();

  if (SplatBitSize > ByteSize * 8)
    return SDValue();

  return isVMOVSplat(SplatBits.getZExtValue(), SplatUndef.getZExtValue(),
                     SplatBitSize, DAG);
}

static bool isVEXTMask(const SmallVectorImpl<int> &M, EVT VT,
                       bool &ReverseVEXT, unsigned &Imm) {
  unsigned NumElts = VT.getVectorNumElements();
  ReverseVEXT = false;
  Imm = M[0];

  // If this is a VEXT shuffle, the immediate value is the index of the first
  // element.  The other shuffle indices must be the successive elements after
  // the first one.
  unsigned ExpectedElt = Imm;
  for (unsigned i = 1; i < NumElts; ++i) {
    // Increment the expected index.  If it wraps around, it may still be
    // a VEXT but the source vectors must be swapped.
    ExpectedElt += 1;
    if (ExpectedElt == NumElts * 2) {
      ExpectedElt = 0;
      ReverseVEXT = true;
    }

    if (ExpectedElt != static_cast<unsigned>(M[i]))
      return false;
  }

  // Adjust the index value if the source operands will be swapped.
  if (ReverseVEXT)
    Imm -= NumElts;

  return true;
}

/// isVREVMask - Check if a vector shuffle corresponds to a VREV
/// instruction with the specified blocksize.  (The order of the elements
/// within each block of the vector is reversed.)
static bool isVREVMask(const SmallVectorImpl<int> &M, EVT VT,
                       unsigned BlockSize) {
  assert((BlockSize==16 || BlockSize==32 || BlockSize==64) &&
         "Only possible block sizes for VREV are: 16, 32, 64");

  unsigned NumElts = VT.getVectorNumElements();
  unsigned EltSz = VT.getVectorElementType().getSizeInBits();
  unsigned BlockElts = M[0] + 1;

  if (BlockSize <= EltSz || BlockSize != BlockElts * EltSz)
    return false;

  for (unsigned i = 0; i < NumElts; ++i) {
    if ((unsigned) M[i] !=
        (i - i%BlockElts) + (BlockElts - 1 - i%BlockElts))
      return false;
  }

  return true;
}

static bool isVTRNMask(const SmallVectorImpl<int> &M, EVT VT,
                       unsigned &WhichResult) {
  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i < NumElts; i += 2) {
    if ((unsigned) M[i] != i + WhichResult ||
        (unsigned) M[i+1] != i + NumElts + WhichResult)
      return false;
  }
  return true;
}

static bool isVUZPMask(const SmallVectorImpl<int> &M, EVT VT,
                       unsigned &WhichResult) {
  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i != NumElts; ++i) {
    if ((unsigned) M[i] != 2 * i + WhichResult)
      return false;
  }

  // VUZP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && VT.getVectorElementType().getSizeInBits() == 32)
    return false;

  return true;
}

static bool isVZIPMask(const SmallVectorImpl<int> &M, EVT VT,
                       unsigned &WhichResult) {
  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  unsigned Idx = WhichResult * NumElts / 2;
  for (unsigned i = 0; i != NumElts; i += 2) {
    if ((unsigned) M[i] != Idx ||
        (unsigned) M[i+1] != Idx + NumElts)
      return false;
    Idx += 1;
  }

  // VZIP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && VT.getVectorElementType().getSizeInBits() == 32)
    return false;

  return true;
}

static SDValue BuildSplat(SDValue Val, EVT VT, SelectionDAG &DAG, DebugLoc dl) {
  // Canonicalize all-zeros and all-ones vectors.
  ConstantSDNode *ConstVal = cast<ConstantSDNode>(Val.getNode());
  if (ConstVal->isNullValue())
    return getZeroVector(VT, DAG, dl);
  if (ConstVal->isAllOnesValue())
    return getOnesVector(VT, DAG, dl);

  EVT CanonicalVT;
  if (VT.is64BitVector()) {
    switch (Val.getValueType().getSizeInBits()) {
    case 8:  CanonicalVT = MVT::v8i8; break;
    case 16: CanonicalVT = MVT::v4i16; break;
    case 32: CanonicalVT = MVT::v2i32; break;
    case 64: CanonicalVT = MVT::v1i64; break;
    default: llvm_unreachable("unexpected splat element type"); break;
    }
  } else {
    assert(VT.is128BitVector() && "unknown splat vector size");
    switch (Val.getValueType().getSizeInBits()) {
    case 8:  CanonicalVT = MVT::v16i8; break;
    case 16: CanonicalVT = MVT::v8i16; break;
    case 32: CanonicalVT = MVT::v4i32; break;
    case 64: CanonicalVT = MVT::v2i64; break;
    default: llvm_unreachable("unexpected splat element type"); break;
    }
  }

  // Build a canonical splat for this value.
  SmallVector<SDValue, 8> Ops;
  Ops.assign(CanonicalVT.getVectorNumElements(), Val);
  SDValue Res = DAG.getNode(ISD::BUILD_VECTOR, dl, CanonicalVT, &Ops[0],
                            Ops.size());
  return DAG.getNode(ISD::BIT_CONVERT, dl, VT, Res);
}

// If this is a case we can't handle, return null and let the default
// expansion code take care of it.
static SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) {
  BuildVectorSDNode *BVN = cast<BuildVectorSDNode>(Op.getNode());
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  APInt SplatBits, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  if (BVN->isConstantSplat(SplatBits, SplatUndef, SplatBitSize, HasAnyUndefs)) {
    SDValue Val = isVMOVSplat(SplatBits.getZExtValue(),
                              SplatUndef.getZExtValue(), SplatBitSize, DAG);
    if (Val.getNode())
      return BuildSplat(Val, VT, DAG, dl);
  }

  // If there are only 2 elements in a 128-bit vector, insert them into an
  // undef vector.  This handles the common case for 128-bit vector argument
  // passing, where the insertions should be translated to subreg accesses
  // with no real instructions.
  if (VT.is128BitVector() && Op.getNumOperands() == 2) {
    SDValue Val = DAG.getUNDEF(VT);
    SDValue Op0 = Op.getOperand(0);
    SDValue Op1 = Op.getOperand(1);
    if (Op0.getOpcode() != ISD::UNDEF)
      Val = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, Val, Op0,
                        DAG.getIntPtrConstant(0));
    if (Op1.getOpcode() != ISD::UNDEF)
      Val = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, Val, Op1,
                        DAG.getIntPtrConstant(1));
    return Val;
  }

  return SDValue();
}

/// isShuffleMaskLegal - Targets can use this to indicate that they only
/// support *some* VECTOR_SHUFFLE operations, those with specific masks.
/// By default, if a target supports the VECTOR_SHUFFLE node, all mask values
/// are assumed to be legal.
bool
ARMTargetLowering::isShuffleMaskLegal(const SmallVectorImpl<int> &M,
                                      EVT VT) const {
  if (VT.getVectorNumElements() == 4 &&
      (VT.is128BitVector() || VT.is64BitVector())) {
    unsigned PFIndexes[4];
    for (unsigned i = 0; i != 4; ++i) {
      if (M[i] < 0)
        PFIndexes[i] = 8;
      else
        PFIndexes[i] = M[i];
    }

    // Compute the index in the perfect shuffle table.
    unsigned PFTableIndex =
      PFIndexes[0]*9*9*9+PFIndexes[1]*9*9+PFIndexes[2]*9+PFIndexes[3];
    unsigned PFEntry = PerfectShuffleTable[PFTableIndex];
    unsigned Cost = (PFEntry >> 30);

    if (Cost <= 4)
      return true;
  }

  bool ReverseVEXT;
  unsigned Imm, WhichResult;

  return (ShuffleVectorSDNode::isSplatMask(&M[0], VT) ||
          isVREVMask(M, VT, 64) ||
          isVREVMask(M, VT, 32) ||
          isVREVMask(M, VT, 16) ||
          isVEXTMask(M, VT, ReverseVEXT, Imm) ||
          isVTRNMask(M, VT, WhichResult) ||
          isVUZPMask(M, VT, WhichResult) ||
          isVZIPMask(M, VT, WhichResult));
}

/// GeneratePerfectShuffle - Given an entry in the perfect-shuffle table, emit
/// the specified operations to build the shuffle.
static SDValue GeneratePerfectShuffle(unsigned PFEntry, SDValue LHS,
                                      SDValue RHS, SelectionDAG &DAG,
                                      DebugLoc dl) {
  unsigned OpNum = (PFEntry >> 26) & 0x0F;
  unsigned LHSID = (PFEntry >> 13) & ((1 << 13)-1);
  unsigned RHSID = (PFEntry >>  0) & ((1 << 13)-1);

  enum {
    OP_COPY = 0, // Copy, used for things like <u,u,u,3> to say it is <0,1,2,3>
    OP_VREV,
    OP_VDUP0,
    OP_VDUP1,
    OP_VDUP2,
    OP_VDUP3,
    OP_VEXT1,
    OP_VEXT2,
    OP_VEXT3,
    OP_VUZPL, // VUZP, left result
    OP_VUZPR, // VUZP, right result
    OP_VZIPL, // VZIP, left result
    OP_VZIPR, // VZIP, right result
    OP_VTRNL, // VTRN, left result
    OP_VTRNR  // VTRN, right result
  };

  if (OpNum == OP_COPY) {
    if (LHSID == (1*9+2)*9+3) return LHS;
    assert(LHSID == ((4*9+5)*9+6)*9+7 && "Illegal OP_COPY!");
    return RHS;
  }

  SDValue OpLHS, OpRHS;
  OpLHS = GeneratePerfectShuffle(PerfectShuffleTable[LHSID], LHS, RHS, DAG, dl);
  OpRHS = GeneratePerfectShuffle(PerfectShuffleTable[RHSID], LHS, RHS, DAG, dl);
  EVT VT = OpLHS.getValueType();

  switch (OpNum) {
  default: llvm_unreachable("Unknown shuffle opcode!");
  case OP_VREV:
    return DAG.getNode(ARMISD::VREV64, dl, VT, OpLHS);
  case OP_VDUP0:
  case OP_VDUP1:
  case OP_VDUP2:
  case OP_VDUP3:
    return DAG.getNode(ARMISD::VDUPLANE, dl, VT,
                       OpLHS, DAG.getConstant(OpNum-OP_VDUP0, MVT::i32));
  case OP_VEXT1:
  case OP_VEXT2:
  case OP_VEXT3:
    return DAG.getNode(ARMISD::VEXT, dl, VT,
                       OpLHS, OpRHS,
                       DAG.getConstant(OpNum-OP_VEXT1+1, MVT::i32));
  case OP_VUZPL:
  case OP_VUZPR:
    return DAG.getNode(ARMISD::VUZP, dl, DAG.getVTList(VT, VT),
                       OpLHS, OpRHS).getValue(OpNum-OP_VUZPL);
  case OP_VZIPL:
  case OP_VZIPR:
    return DAG.getNode(ARMISD::VZIP, dl, DAG.getVTList(VT, VT),
                       OpLHS, OpRHS).getValue(OpNum-OP_VZIPL);
  case OP_VTRNL:
  case OP_VTRNR:
    return DAG.getNode(ARMISD::VTRN, dl, DAG.getVTList(VT, VT),
                       OpLHS, OpRHS).getValue(OpNum-OP_VTRNL);
  }
}

static SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) {
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op.getNode());
  SmallVector<int, 8> ShuffleMask;

  // Convert shuffles that are directly supported on NEON to target-specific
  // DAG nodes, instead of keeping them as shuffles and matching them again
  // during code selection.  This is more efficient and avoids the possibility
  // of inconsistencies between legalization and selection.
  // FIXME: floating-point vectors should be canonicalized to integer vectors
  // of the same time so that they get CSEd properly.
  SVN->getMask(ShuffleMask);

  if (ShuffleVectorSDNode::isSplatMask(&ShuffleMask[0], VT)) {
    int Lane = SVN->getSplatIndex();
    if (Lane == 0 && V1.getOpcode() == ISD::SCALAR_TO_VECTOR) {
      return DAG.getNode(ARMISD::VDUP, dl, VT, V1.getOperand(0));
    }
    return DAG.getNode(ARMISD::VDUPLANE, dl, VT, V1,
                       DAG.getConstant(Lane, MVT::i32));
  }

  bool ReverseVEXT;
  unsigned Imm;
  if (isVEXTMask(ShuffleMask, VT, ReverseVEXT, Imm)) {
    if (ReverseVEXT)
      std::swap(V1, V2);
    return DAG.getNode(ARMISD::VEXT, dl, VT, V1, V2,
                       DAG.getConstant(Imm, MVT::i32));
  }

  if (isVREVMask(ShuffleMask, VT, 64))
    return DAG.getNode(ARMISD::VREV64, dl, VT, V1);
  if (isVREVMask(ShuffleMask, VT, 32))
    return DAG.getNode(ARMISD::VREV32, dl, VT, V1);
  if (isVREVMask(ShuffleMask, VT, 16))
    return DAG.getNode(ARMISD::VREV16, dl, VT, V1);

  // Check for Neon shuffles that modify both input vectors in place.
  // If both results are used, i.e., if there are two shuffles with the same
  // source operands and with masks corresponding to both results of one of
  // these operations, DAG memoization will ensure that a single node is
  // used for both shuffles.
  unsigned WhichResult;
  if (isVTRNMask(ShuffleMask, VT, WhichResult))
    return DAG.getNode(ARMISD::VTRN, dl, DAG.getVTList(VT, VT),
                       V1, V2).getValue(WhichResult);
  if (isVUZPMask(ShuffleMask, VT, WhichResult))
    return DAG.getNode(ARMISD::VUZP, dl, DAG.getVTList(VT, VT),
                       V1, V2).getValue(WhichResult);
  if (isVZIPMask(ShuffleMask, VT, WhichResult))
    return DAG.getNode(ARMISD::VZIP, dl, DAG.getVTList(VT, VT),
                       V1, V2).getValue(WhichResult);

  // If the shuffle is not directly supported and it has 4 elements, use
  // the PerfectShuffle-generated table to synthesize it from other shuffles.
  if (VT.getVectorNumElements() == 4 &&
      (VT.is128BitVector() || VT.is64BitVector())) {
    unsigned PFIndexes[4];
    for (unsigned i = 0; i != 4; ++i) {
      if (ShuffleMask[i] < 0)
        PFIndexes[i] = 8;
      else
        PFIndexes[i] = ShuffleMask[i];
    }

    // Compute the index in the perfect shuffle table.
    unsigned PFTableIndex =
      PFIndexes[0]*9*9*9+PFIndexes[1]*9*9+PFIndexes[2]*9+PFIndexes[3];

    unsigned PFEntry = PerfectShuffleTable[PFTableIndex];
    unsigned Cost = (PFEntry >> 30);

    if (Cost <= 4)
      return GeneratePerfectShuffle(PFEntry, V1, V2, DAG, dl);
  }

  return SDValue();
}

static SDValue LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) {
  return Op;
}

static SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  assert((VT == MVT::i8 || VT == MVT::i16) &&
         "unexpected type for custom-lowering vector extract");
  SDValue Vec = Op.getOperand(0);
  SDValue Lane = Op.getOperand(1);
  Op = DAG.getNode(ARMISD::VGETLANEu, dl, MVT::i32, Vec, Lane);
  Op = DAG.getNode(ISD::AssertZext, dl, MVT::i32, Op, DAG.getValueType(VT));
  return DAG.getNode(ISD::TRUNCATE, dl, VT, Op);
}

static SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) {
  // The only time a CONCAT_VECTORS operation can have legal types is when
  // two 64-bit vectors are concatenated to a 128-bit vector.
  assert(Op.getValueType().is128BitVector() && Op.getNumOperands() == 2 &&
         "unexpected CONCAT_VECTORS");
  DebugLoc dl = Op.getDebugLoc();
  SDValue Val = DAG.getUNDEF(MVT::v2f64);
  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  if (Op0.getOpcode() != ISD::UNDEF)
    Val = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64, Val,
                      DAG.getNode(ISD::BIT_CONVERT, dl, MVT::f64, Op0),
                      DAG.getIntPtrConstant(0));
  if (Op1.getOpcode() != ISD::UNDEF)
    Val = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64, Val,
                      DAG.getNode(ISD::BIT_CONVERT, dl, MVT::f64, Op1),
                      DAG.getIntPtrConstant(1));
  return DAG.getNode(ISD::BIT_CONVERT, dl, Op.getValueType(), Val);
}

SDValue ARMTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Don't know how to custom lower this!");
  case ISD::ConstantPool:  return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:
    return Subtarget->isTargetDarwin() ? LowerGlobalAddressDarwin(Op, DAG) :
      LowerGlobalAddressELF(Op, DAG);
  case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
  case ISD::SELECT_CC:     return LowerSELECT_CC(Op, DAG, Subtarget);
  case ISD::BR_CC:         return LowerBR_CC(Op, DAG, Subtarget);
  case ISD::BR_JT:         return LowerBR_JT(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC: return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::VASTART:       return LowerVASTART(Op, DAG, VarArgsFrameIndex);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:    return LowerINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:    return LowerFP_TO_INT(Op, DAG);
  case ISD::FCOPYSIGN:     return LowerFCOPYSIGN(Op, DAG);
  case ISD::RETURNADDR:    break;
  case ISD::FRAMEADDR:     return LowerFRAMEADDR(Op, DAG);
  case ISD::GLOBAL_OFFSET_TABLE: return LowerGLOBAL_OFFSET_TABLE(Op, DAG);
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN: return LowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::BIT_CONVERT:   return ExpandBIT_CONVERT(Op.getNode(), DAG);
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:           return LowerShift(Op.getNode(), DAG, Subtarget);
  case ISD::VSETCC:        return LowerVSETCC(Op, DAG);
  case ISD::BUILD_VECTOR:  return LowerBUILD_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE: return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::SCALAR_TO_VECTOR: return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT: return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::CONCAT_VECTORS: return LowerCONCAT_VECTORS(Op, DAG);
  }
  return SDValue();
}

/// ReplaceNodeResults - Replace the results of node with an illegal result
/// type with new values built out of custom code.
void ARMTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue>&Results,
                                           SelectionDAG &DAG) {
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom expand this!");
    return;
  case ISD::BIT_CONVERT:
    Results.push_back(ExpandBIT_CONVERT(N, DAG));
    return;
  case ISD::SRL:
  case ISD::SRA: {
    SDValue Res = LowerShift(N, DAG, Subtarget);
    if (Res.getNode())
      Results.push_back(Res);
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
//                           ARM Scheduler Hooks
//===----------------------------------------------------------------------===//

MachineBasicBlock *
ARMTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                               MachineBasicBlock *BB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  switch (MI->getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case ARM::tMOVCCr_pseudo: {
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
    BuildMI(BB, dl, TII->get(ARM::tBcc)).addMBB(sinkMBB)
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
    BuildMI(BB, dl, TII->get(ARM::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

    F->DeleteMachineInstr(MI);   // The pseudo instruction is gone now.
    return BB;
  }

  case ARM::tANDsp:
  case ARM::tADDspr_:
  case ARM::tSUBspi_:
  case ARM::t2SUBrSPi_:
  case ARM::t2SUBrSPi12_:
  case ARM::t2SUBrSPs_: {
    MachineFunction *MF = BB->getParent();
    unsigned DstReg = MI->getOperand(0).getReg();
    unsigned SrcReg = MI->getOperand(1).getReg();
    bool DstIsDead = MI->getOperand(0).isDead();
    bool SrcIsKill = MI->getOperand(1).isKill();

    if (SrcReg != ARM::SP) {
      // Copy the source to SP from virtual register.
      const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(SrcReg);
      unsigned CopyOpc = (RC == ARM::tGPRRegisterClass)
        ? ARM::tMOVtgpr2gpr : ARM::tMOVgpr2gpr;
      BuildMI(BB, dl, TII->get(CopyOpc), ARM::SP)
        .addReg(SrcReg, getKillRegState(SrcIsKill));
    }

    unsigned OpOpc = 0;
    bool NeedPred = false, NeedCC = false, NeedOp3 = false;
    switch (MI->getOpcode()) {
    default:
      llvm_unreachable("Unexpected pseudo instruction!");
    case ARM::tANDsp:
      OpOpc = ARM::tAND;
      NeedPred = true;
      break;
    case ARM::tADDspr_:
      OpOpc = ARM::tADDspr;
      break;
    case ARM::tSUBspi_:
      OpOpc = ARM::tSUBspi;
      break;
    case ARM::t2SUBrSPi_:
      OpOpc = ARM::t2SUBrSPi;
      NeedPred = true; NeedCC = true;
      break;
    case ARM::t2SUBrSPi12_:
      OpOpc = ARM::t2SUBrSPi12;
      NeedPred = true;
      break;
    case ARM::t2SUBrSPs_:
      OpOpc = ARM::t2SUBrSPs;
      NeedPred = true; NeedCC = true; NeedOp3 = true;
      break;
    }
    MachineInstrBuilder MIB = BuildMI(BB, dl, TII->get(OpOpc), ARM::SP);
    if (OpOpc == ARM::tAND)
      AddDefaultT1CC(MIB);
    MIB.addReg(ARM::SP);
    MIB.addOperand(MI->getOperand(2));
    if (NeedOp3)
      MIB.addOperand(MI->getOperand(3));
    if (NeedPred)
      AddDefaultPred(MIB);
    if (NeedCC)
      AddDefaultCC(MIB);

    // Copy the result from SP to virtual register.
    const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(DstReg);
    unsigned CopyOpc = (RC == ARM::tGPRRegisterClass)
      ? ARM::tMOVgpr2tgpr : ARM::tMOVgpr2gpr;
    BuildMI(BB, dl, TII->get(CopyOpc))
      .addReg(DstReg, getDefRegState(true) | getDeadRegState(DstIsDead))
      .addReg(ARM::SP);
    MF->DeleteMachineInstr(MI);   // The pseudo instruction is gone now.
    return BB;
  }
  }
}

//===----------------------------------------------------------------------===//
//                           ARM Optimization Hooks
//===----------------------------------------------------------------------===//

static
SDValue combineSelectAndUse(SDNode *N, SDValue Slct, SDValue OtherOp,
                            TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT = N->getValueType(0);
  unsigned Opc = N->getOpcode();
  bool isSlctCC = Slct.getOpcode() == ISD::SELECT_CC;
  SDValue LHS = isSlctCC ? Slct.getOperand(2) : Slct.getOperand(1);
  SDValue RHS = isSlctCC ? Slct.getOperand(3) : Slct.getOperand(2);
  ISD::CondCode CC = ISD::SETCC_INVALID;

  if (isSlctCC) {
    CC = cast<CondCodeSDNode>(Slct.getOperand(4))->get();
  } else {
    SDValue CCOp = Slct.getOperand(0);
    if (CCOp.getOpcode() == ISD::SETCC)
      CC = cast<CondCodeSDNode>(CCOp.getOperand(2))->get();
  }

  bool DoXform = false;
  bool InvCC = false;
  assert ((Opc == ISD::ADD || (Opc == ISD::SUB && Slct == N->getOperand(1))) &&
          "Bad input!");

  if (LHS.getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(LHS)->isNullValue()) {
    DoXform = true;
  } else if (CC != ISD::SETCC_INVALID &&
             RHS.getOpcode() == ISD::Constant &&
             cast<ConstantSDNode>(RHS)->isNullValue()) {
    std::swap(LHS, RHS);
    SDValue Op0 = Slct.getOperand(0);
    EVT OpVT = isSlctCC ? Op0.getValueType() :
                          Op0.getOperand(0).getValueType();
    bool isInt = OpVT.isInteger();
    CC = ISD::getSetCCInverse(CC, isInt);

    if (!TLI.isCondCodeLegal(CC, OpVT))
      return SDValue();         // Inverse operator isn't legal.

    DoXform = true;
    InvCC = true;
  }

  if (DoXform) {
    SDValue Result = DAG.getNode(Opc, RHS.getDebugLoc(), VT, OtherOp, RHS);
    if (isSlctCC)
      return DAG.getSelectCC(N->getDebugLoc(), OtherOp, Result,
                             Slct.getOperand(0), Slct.getOperand(1), CC);
    SDValue CCOp = Slct.getOperand(0);
    if (InvCC)
      CCOp = DAG.getSetCC(Slct.getDebugLoc(), CCOp.getValueType(),
                          CCOp.getOperand(0), CCOp.getOperand(1), CC);
    return DAG.getNode(ISD::SELECT, N->getDebugLoc(), VT,
                       CCOp, OtherOp, Result);
  }
  return SDValue();
}

/// PerformADDCombine - Target-specific dag combine xforms for ISD::ADD.
static SDValue PerformADDCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {
  // added by evan in r37685 with no testcase.
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);

  // fold (add (select cc, 0, c), x) -> (select cc, x, (add, x, c))
  if (N0.getOpcode() == ISD::SELECT && N0.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N0, N1, DCI);
    if (Result.getNode()) return Result;
  }
  if (N1.getOpcode() == ISD::SELECT && N1.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N1, N0, DCI);
    if (Result.getNode()) return Result;
  }

  return SDValue();
}

/// PerformSUBCombine - Target-specific dag combine xforms for ISD::SUB.
static SDValue PerformSUBCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {
  // added by evan in r37685 with no testcase.
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);

  // fold (sub x, (select cc, 0, c)) -> (select cc, x, (sub, x, c))
  if (N1.getOpcode() == ISD::SELECT && N1.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N1, N0, DCI);
    if (Result.getNode()) return Result;
  }

  return SDValue();
}


/// PerformFMRRDCombine - Target-specific dag combine xforms for ARMISD::FMRRD.
static SDValue PerformFMRRDCombine(SDNode *N,
                                   TargetLowering::DAGCombinerInfo &DCI) {
  // fmrrd(fmdrr x, y) -> x,y
  SDValue InDouble = N->getOperand(0);
  if (InDouble.getOpcode() == ARMISD::FMDRR)
    return DCI.CombineTo(N, InDouble.getOperand(0), InDouble.getOperand(1));
  return SDValue();
}

/// getVShiftImm - Check if this is a valid build_vector for the immediate
/// operand of a vector shift operation, where all the elements of the
/// build_vector must have the same constant integer value.
static bool getVShiftImm(SDValue Op, unsigned ElementBits, int64_t &Cnt) {
  // Ignore bit_converts.
  while (Op.getOpcode() == ISD::BIT_CONVERT)
    Op = Op.getOperand(0);
  BuildVectorSDNode *BVN = dyn_cast<BuildVectorSDNode>(Op.getNode());
  APInt SplatBits, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  if (! BVN || ! BVN->isConstantSplat(SplatBits, SplatUndef, SplatBitSize,
                                      HasAnyUndefs, ElementBits) ||
      SplatBitSize > ElementBits)
    return false;
  Cnt = SplatBits.getSExtValue();
  return true;
}

/// isVShiftLImm - Check if this is a valid build_vector for the immediate
/// operand of a vector shift left operation.  That value must be in the range:
///   0 <= Value < ElementBits for a left shift; or
///   0 <= Value <= ElementBits for a long left shift.
static bool isVShiftLImm(SDValue Op, EVT VT, bool isLong, int64_t &Cnt) {
  assert(VT.isVector() && "vector shift count is not a vector type");
  unsigned ElementBits = VT.getVectorElementType().getSizeInBits();
  if (! getVShiftImm(Op, ElementBits, Cnt))
    return false;
  return (Cnt >= 0 && (isLong ? Cnt-1 : Cnt) < ElementBits);
}

/// isVShiftRImm - Check if this is a valid build_vector for the immediate
/// operand of a vector shift right operation.  For a shift opcode, the value
/// is positive, but for an intrinsic the value count must be negative. The
/// absolute value must be in the range:
///   1 <= |Value| <= ElementBits for a right shift; or
///   1 <= |Value| <= ElementBits/2 for a narrow right shift.
static bool isVShiftRImm(SDValue Op, EVT VT, bool isNarrow, bool isIntrinsic,
                         int64_t &Cnt) {
  assert(VT.isVector() && "vector shift count is not a vector type");
  unsigned ElementBits = VT.getVectorElementType().getSizeInBits();
  if (! getVShiftImm(Op, ElementBits, Cnt))
    return false;
  if (isIntrinsic)
    Cnt = -Cnt;
  return (Cnt >= 1 && Cnt <= (isNarrow ? ElementBits/2 : ElementBits));
}

/// PerformIntrinsicCombine - ARM-specific DAG combining for intrinsics.
static SDValue PerformIntrinsicCombine(SDNode *N, SelectionDAG &DAG) {
  unsigned IntNo = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
  switch (IntNo) {
  default:
    // Don't do anything for most intrinsics.
    break;

  // Vector shifts: check for immediate versions and lower them.
  // Note: This is done during DAG combining instead of DAG legalizing because
  // the build_vectors for 64-bit vector element shift counts are generally
  // not legal, and it is hard to see their values after they get legalized to
  // loads from a constant pool.
  case Intrinsic::arm_neon_vshifts:
  case Intrinsic::arm_neon_vshiftu:
  case Intrinsic::arm_neon_vshiftls:
  case Intrinsic::arm_neon_vshiftlu:
  case Intrinsic::arm_neon_vshiftn:
  case Intrinsic::arm_neon_vrshifts:
  case Intrinsic::arm_neon_vrshiftu:
  case Intrinsic::arm_neon_vrshiftn:
  case Intrinsic::arm_neon_vqshifts:
  case Intrinsic::arm_neon_vqshiftu:
  case Intrinsic::arm_neon_vqshiftsu:
  case Intrinsic::arm_neon_vqshiftns:
  case Intrinsic::arm_neon_vqshiftnu:
  case Intrinsic::arm_neon_vqshiftnsu:
  case Intrinsic::arm_neon_vqrshiftns:
  case Intrinsic::arm_neon_vqrshiftnu:
  case Intrinsic::arm_neon_vqrshiftnsu: {
    EVT VT = N->getOperand(1).getValueType();
    int64_t Cnt;
    unsigned VShiftOpc = 0;

    switch (IntNo) {
    case Intrinsic::arm_neon_vshifts:
    case Intrinsic::arm_neon_vshiftu:
      if (isVShiftLImm(N->getOperand(2), VT, false, Cnt)) {
        VShiftOpc = ARMISD::VSHL;
        break;
      }
      if (isVShiftRImm(N->getOperand(2), VT, false, true, Cnt)) {
        VShiftOpc = (IntNo == Intrinsic::arm_neon_vshifts ?
                     ARMISD::VSHRs : ARMISD::VSHRu);
        break;
      }
      return SDValue();

    case Intrinsic::arm_neon_vshiftls:
    case Intrinsic::arm_neon_vshiftlu:
      if (isVShiftLImm(N->getOperand(2), VT, true, Cnt))
        break;
      llvm_unreachable("invalid shift count for vshll intrinsic");

    case Intrinsic::arm_neon_vrshifts:
    case Intrinsic::arm_neon_vrshiftu:
      if (isVShiftRImm(N->getOperand(2), VT, false, true, Cnt))
        break;
      return SDValue();

    case Intrinsic::arm_neon_vqshifts:
    case Intrinsic::arm_neon_vqshiftu:
      if (isVShiftLImm(N->getOperand(2), VT, false, Cnt))
        break;
      return SDValue();

    case Intrinsic::arm_neon_vqshiftsu:
      if (isVShiftLImm(N->getOperand(2), VT, false, Cnt))
        break;
      llvm_unreachable("invalid shift count for vqshlu intrinsic");

    case Intrinsic::arm_neon_vshiftn:
    case Intrinsic::arm_neon_vrshiftn:
    case Intrinsic::arm_neon_vqshiftns:
    case Intrinsic::arm_neon_vqshiftnu:
    case Intrinsic::arm_neon_vqshiftnsu:
    case Intrinsic::arm_neon_vqrshiftns:
    case Intrinsic::arm_neon_vqrshiftnu:
    case Intrinsic::arm_neon_vqrshiftnsu:
      // Narrowing shifts require an immediate right shift.
      if (isVShiftRImm(N->getOperand(2), VT, true, true, Cnt))
        break;
      llvm_unreachable("invalid shift count for narrowing vector shift intrinsic");

    default:
      llvm_unreachable("unhandled vector shift");
    }

    switch (IntNo) {
    case Intrinsic::arm_neon_vshifts:
    case Intrinsic::arm_neon_vshiftu:
      // Opcode already set above.
      break;
    case Intrinsic::arm_neon_vshiftls:
    case Intrinsic::arm_neon_vshiftlu:
      if (Cnt == VT.getVectorElementType().getSizeInBits())
        VShiftOpc = ARMISD::VSHLLi;
      else
        VShiftOpc = (IntNo == Intrinsic::arm_neon_vshiftls ?
                     ARMISD::VSHLLs : ARMISD::VSHLLu);
      break;
    case Intrinsic::arm_neon_vshiftn:
      VShiftOpc = ARMISD::VSHRN; break;
    case Intrinsic::arm_neon_vrshifts:
      VShiftOpc = ARMISD::VRSHRs; break;
    case Intrinsic::arm_neon_vrshiftu:
      VShiftOpc = ARMISD::VRSHRu; break;
    case Intrinsic::arm_neon_vrshiftn:
      VShiftOpc = ARMISD::VRSHRN; break;
    case Intrinsic::arm_neon_vqshifts:
      VShiftOpc = ARMISD::VQSHLs; break;
    case Intrinsic::arm_neon_vqshiftu:
      VShiftOpc = ARMISD::VQSHLu; break;
    case Intrinsic::arm_neon_vqshiftsu:
      VShiftOpc = ARMISD::VQSHLsu; break;
    case Intrinsic::arm_neon_vqshiftns:
      VShiftOpc = ARMISD::VQSHRNs; break;
    case Intrinsic::arm_neon_vqshiftnu:
      VShiftOpc = ARMISD::VQSHRNu; break;
    case Intrinsic::arm_neon_vqshiftnsu:
      VShiftOpc = ARMISD::VQSHRNsu; break;
    case Intrinsic::arm_neon_vqrshiftns:
      VShiftOpc = ARMISD::VQRSHRNs; break;
    case Intrinsic::arm_neon_vqrshiftnu:
      VShiftOpc = ARMISD::VQRSHRNu; break;
    case Intrinsic::arm_neon_vqrshiftnsu:
      VShiftOpc = ARMISD::VQRSHRNsu; break;
    }

    return DAG.getNode(VShiftOpc, N->getDebugLoc(), N->getValueType(0),
                       N->getOperand(1), DAG.getConstant(Cnt, MVT::i32));
  }

  case Intrinsic::arm_neon_vshiftins: {
    EVT VT = N->getOperand(1).getValueType();
    int64_t Cnt;
    unsigned VShiftOpc = 0;

    if (isVShiftLImm(N->getOperand(3), VT, false, Cnt))
      VShiftOpc = ARMISD::VSLI;
    else if (isVShiftRImm(N->getOperand(3), VT, false, true, Cnt))
      VShiftOpc = ARMISD::VSRI;
    else {
      llvm_unreachable("invalid shift count for vsli/vsri intrinsic");
    }

    return DAG.getNode(VShiftOpc, N->getDebugLoc(), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2),
                       DAG.getConstant(Cnt, MVT::i32));
  }

  case Intrinsic::arm_neon_vqrshifts:
  case Intrinsic::arm_neon_vqrshiftu:
    // No immediate versions of these to check for.
    break;
  }

  return SDValue();
}

/// PerformShiftCombine - Checks for immediate versions of vector shifts and
/// lowers them.  As with the vector shift intrinsics, this is done during DAG
/// combining instead of DAG legalizing because the build_vectors for 64-bit
/// vector element shift counts are generally not legal, and it is hard to see
/// their values after they get legalized to loads from a constant pool.
static SDValue PerformShiftCombine(SDNode *N, SelectionDAG &DAG,
                                   const ARMSubtarget *ST) {
  EVT VT = N->getValueType(0);

  // Nothing to be done for scalar shifts.
  if (! VT.isVector())
    return SDValue();

  assert(ST->hasNEON() && "unexpected vector shift");
  int64_t Cnt;

  switch (N->getOpcode()) {
  default: llvm_unreachable("unexpected shift opcode");

  case ISD::SHL:
    if (isVShiftLImm(N->getOperand(1), VT, false, Cnt))
      return DAG.getNode(ARMISD::VSHL, N->getDebugLoc(), VT, N->getOperand(0),
                         DAG.getConstant(Cnt, MVT::i32));
    break;

  case ISD::SRA:
  case ISD::SRL:
    if (isVShiftRImm(N->getOperand(1), VT, false, false, Cnt)) {
      unsigned VShiftOpc = (N->getOpcode() == ISD::SRA ?
                            ARMISD::VSHRs : ARMISD::VSHRu);
      return DAG.getNode(VShiftOpc, N->getDebugLoc(), VT, N->getOperand(0),
                         DAG.getConstant(Cnt, MVT::i32));
    }
  }
  return SDValue();
}

/// PerformExtendCombine - Target-specific DAG combining for ISD::SIGN_EXTEND,
/// ISD::ZERO_EXTEND, and ISD::ANY_EXTEND.
static SDValue PerformExtendCombine(SDNode *N, SelectionDAG &DAG,
                                    const ARMSubtarget *ST) {
  SDValue N0 = N->getOperand(0);

  // Check for sign- and zero-extensions of vector extract operations of 8-
  // and 16-bit vector elements.  NEON supports these directly.  They are
  // handled during DAG combining because type legalization will promote them
  // to 32-bit types and it is messy to recognize the operations after that.
  if (ST->hasNEON() && N0.getOpcode() == ISD::EXTRACT_VECTOR_ELT) {
    SDValue Vec = N0.getOperand(0);
    SDValue Lane = N0.getOperand(1);
    EVT VT = N->getValueType(0);
    EVT EltVT = N0.getValueType();
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();

    if (VT == MVT::i32 &&
        (EltVT == MVT::i8 || EltVT == MVT::i16) &&
        TLI.isTypeLegal(Vec.getValueType())) {

      unsigned Opc = 0;
      switch (N->getOpcode()) {
      default: llvm_unreachable("unexpected opcode");
      case ISD::SIGN_EXTEND:
        Opc = ARMISD::VGETLANEs;
        break;
      case ISD::ZERO_EXTEND:
      case ISD::ANY_EXTEND:
        Opc = ARMISD::VGETLANEu;
        break;
      }
      return DAG.getNode(Opc, N->getDebugLoc(), VT, Vec, Lane);
    }
  }

  return SDValue();
}

SDValue ARMTargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  default: break;
  case ISD::ADD:      return PerformADDCombine(N, DCI);
  case ISD::SUB:      return PerformSUBCombine(N, DCI);
  case ARMISD::FMRRD: return PerformFMRRDCombine(N, DCI);
  case ISD::INTRINSIC_WO_CHAIN:
    return PerformIntrinsicCombine(N, DCI.DAG);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    return PerformShiftCombine(N, DCI.DAG, Subtarget);
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
    return PerformExtendCombine(N, DCI.DAG, Subtarget);
  }
  return SDValue();
}

bool ARMTargetLowering::allowsUnalignedMemoryAccesses(EVT VT) const {
  if (!Subtarget->hasV6Ops())
    // Pre-v6 does not support unaligned mem access.
    return false;
  else if (!Subtarget->hasV6Ops()) {
    // v6 may or may not support unaligned mem access.
    if (!Subtarget->isTargetDarwin())
      return false;
  }

  switch (VT.getSimpleVT().SimpleTy) {
  default:
    return false;
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    return true;
  // FIXME: VLD1 etc with standard alignment is legal.
  }
}

static bool isLegalT1AddressImmediate(int64_t V, EVT VT) {
  if (V < 0)
    return false;

  unsigned Scale = 1;
  switch (VT.getSimpleVT().SimpleTy) {
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

static bool isLegalT2AddressImmediate(int64_t V, EVT VT,
                                      const ARMSubtarget *Subtarget) {
  bool isNeg = false;
  if (V < 0) {
    isNeg = true;
    V = - V;
  }

  switch (VT.getSimpleVT().SimpleTy) {
  default: return false;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    // + imm12 or - imm8
    if (isNeg)
      return V == (V & ((1LL << 8) - 1));
    return V == (V & ((1LL << 12) - 1));
  case MVT::f32:
  case MVT::f64:
    // Same as ARM mode. FIXME: NEON?
    if (!Subtarget->hasVFP2())
      return false;
    if ((V & 3) != 0)
      return false;
    V >>= 2;
    return V == (V & ((1LL << 8) - 1));
  }
}

/// isLegalAddressImmediate - Return true if the integer value can be used
/// as the offset of the target addressing mode for load / store of the
/// given type.
static bool isLegalAddressImmediate(int64_t V, EVT VT,
                                    const ARMSubtarget *Subtarget) {
  if (V == 0)
    return true;

  if (!VT.isSimple())
    return false;

  if (Subtarget->isThumb1Only())
    return isLegalT1AddressImmediate(V, VT);
  else if (Subtarget->isThumb2())
    return isLegalT2AddressImmediate(V, VT, Subtarget);

  // ARM mode.
  if (V < 0)
    V = - V;
  switch (VT.getSimpleVT().SimpleTy) {
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
    if (!Subtarget->hasVFP2()) // FIXME: NEON?
      return false;
    if ((V & 3) != 0)
      return false;
    V >>= 2;
    return V == (V & ((1LL << 8) - 1));
  }
}

bool ARMTargetLowering::isLegalT2ScaledAddressingMode(const AddrMode &AM,
                                                      EVT VT) const {
  int Scale = AM.Scale;
  if (Scale < 0)
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  default: return false;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    if (Scale == 1)
      return true;
    // r + r << imm
    Scale = Scale & ~1;
    return Scale == 2 || Scale == 4 || Scale == 8;
  case MVT::i64:
    // r + r
    if (((unsigned)AM.HasBaseReg + Scale) <= 2)
      return true;
    return false;
  case MVT::isVoid:
    // Note, we allow "void" uses (basically, uses that aren't loads or
    // stores), because arm allows folding a scale into many arithmetic
    // operations.  This should be made more precise and revisited later.

    // Allow r << imm, but the imm has to be a multiple of two.
    if (Scale & 1) return false;
    return isPowerOf2_32(Scale);
  }
}

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool ARMTargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                              const Type *Ty) const {
  EVT VT = getValueType(Ty, true);
  if (!isLegalAddressImmediate(AM.BaseOffs, VT, Subtarget))
    return false;

  // Can never fold addr of global into load/store.
  if (AM.BaseGV)
    return false;

  switch (AM.Scale) {
  case 0:  // no scale reg, must be "r+i" or "r", or "i".
    break;
  case 1:
    if (Subtarget->isThumb1Only())
      return false;
    // FALL THROUGH.
  default:
    // ARM doesn't support any R+R*scale+imm addr modes.
    if (AM.BaseOffs)
      return false;

    if (!VT.isSimple())
      return false;

    if (Subtarget->isThumb2())
      return isLegalT2ScaledAddressingMode(AM, VT);

    int Scale = AM.Scale;
    switch (VT.getSimpleVT().SimpleTy) {
    default: return false;
    case MVT::i1:
    case MVT::i8:
    case MVT::i32:
      if (Scale < 0) Scale = -Scale;
      if (Scale == 1)
        return true;
      // r + r << imm
      return isPowerOf2_32(Scale & ~1);
    case MVT::i16:
    case MVT::i64:
      // r + r
      if (((unsigned)AM.HasBaseReg + Scale) <= 2)
        return true;
      return false;

    case MVT::isVoid:
      // Note, we allow "void" uses (basically, uses that aren't loads or
      // stores), because arm allows folding a scale into many arithmetic
      // operations.  This should be made more precise and revisited later.

      // Allow r << imm, but the imm has to be a multiple of two.
      if (Scale & 1) return false;
      return isPowerOf2_32(Scale);
    }
    break;
  }
  return true;
}

static bool getARMIndexedAddressParts(SDNode *Ptr, EVT VT,
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
        assert(Ptr->getOpcode() == ISD::ADD);
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
        assert(Ptr->getOpcode() == ISD::ADD);
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

static bool getT2IndexedAddressParts(SDNode *Ptr, EVT VT,
                                     bool isSEXTLoad, SDValue &Base,
                                     SDValue &Offset, bool &isInc,
                                     SelectionDAG &DAG) {
  if (Ptr->getOpcode() != ISD::ADD && Ptr->getOpcode() != ISD::SUB)
    return false;

  Base = Ptr->getOperand(0);
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(Ptr->getOperand(1))) {
    int RHSC = (int)RHS->getZExtValue();
    if (RHSC < 0 && RHSC > -0x100) { // 8 bits.
      assert(Ptr->getOpcode() == ISD::ADD);
      isInc = false;
      Offset = DAG.getConstant(-RHSC, RHS->getValueType(0));
      return true;
    } else if (RHSC > 0 && RHSC < 0x100) { // 8 bit, no zero.
      isInc = Ptr->getOpcode() == ISD::ADD;
      Offset = DAG.getConstant(RHSC, RHS->getValueType(0));
      return true;
    }
  }

  return false;
}

/// getPreIndexedAddressParts - returns true by value, base pointer and
/// offset pointer and addressing mode by reference if the node's address
/// can be legally represented as pre-indexed load / store address.
bool
ARMTargetLowering::getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                             SDValue &Offset,
                                             ISD::MemIndexedMode &AM,
                                             SelectionDAG &DAG) const {
  if (Subtarget->isThumb1Only())
    return false;

  EVT VT;
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
  bool isLegal = false;
  if (Subtarget->isThumb2())
    isLegal = getT2IndexedAddressParts(Ptr.getNode(), VT, isSEXTLoad, Base,
                                       Offset, isInc, DAG);
  else
    isLegal = getARMIndexedAddressParts(Ptr.getNode(), VT, isSEXTLoad, Base,
                                        Offset, isInc, DAG);
  if (!isLegal)
    return false;

  AM = isInc ? ISD::PRE_INC : ISD::PRE_DEC;
  return true;
}

/// getPostIndexedAddressParts - returns true by value, base pointer and
/// offset pointer and addressing mode by reference if this node can be
/// combined with a load / store to form a post-indexed load / store.
bool ARMTargetLowering::getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                                   SDValue &Base,
                                                   SDValue &Offset,
                                                   ISD::MemIndexedMode &AM,
                                                   SelectionDAG &DAG) const {
  if (Subtarget->isThumb1Only())
    return false;

  EVT VT;
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
  bool isLegal = false;
  if (Subtarget->isThumb2())
    isLegal = getT2IndexedAddressParts(Op, VT, isSEXTLoad, Base, Offset,
                                        isInc, DAG);
  else
    isLegal = getARMIndexedAddressParts(Op, VT, isSEXTLoad, Base, Offset,
                                        isInc, DAG);
  if (!isLegal)
    return false;

  AM = isInc ? ISD::POST_INC : ISD::POST_DEC;
  return true;
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
                                                EVT VT) const {
  if (Constraint.size() == 1) {
    // GCC RS6000 Constraint Letters
    switch (Constraint[0]) {
    case 'l':
      if (Subtarget->isThumb1Only())
        return std::make_pair(0U, ARM::tGPRRegisterClass);
      else
        return std::make_pair(0U, ARM::GPRRegisterClass);
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
                                  EVT VT) const {
  if (Constraint.size() != 1)
    return std::vector<unsigned>();

  switch (Constraint[0]) {      // GCC ARM Constraint Letters
  default: break;
  case 'l':
    return make_vector<unsigned>(ARM::R0, ARM::R1, ARM::R2, ARM::R3,
                                 ARM::R4, ARM::R5, ARM::R6, ARM::R7,
                                 0);
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

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void ARMTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                     char Constraint,
                                                     bool hasMemory,
                                                     std::vector<SDValue>&Ops,
                                                     SelectionDAG &DAG) const {
  SDValue Result(0, 0);

  switch (Constraint) {
  default: break;
  case 'I': case 'J': case 'K': case 'L':
  case 'M': case 'N': case 'O':
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    if (!C)
      return;

    int64_t CVal64 = C->getSExtValue();
    int CVal = (int) CVal64;
    // None of these constraints allow values larger than 32 bits.  Check
    // that the value fits in an int.
    if (CVal != CVal64)
      return;

    switch (Constraint) {
      case 'I':
        if (Subtarget->isThumb1Only()) {
          // This must be a constant between 0 and 255, for ADD
          // immediates.
          if (CVal >= 0 && CVal <= 255)
            break;
        } else if (Subtarget->isThumb2()) {
          // A constant that can be used as an immediate value in a
          // data-processing instruction.
          if (ARM_AM::getT2SOImmVal(CVal) != -1)
            break;
        } else {
          // A constant that can be used as an immediate value in a
          // data-processing instruction.
          if (ARM_AM::getSOImmVal(CVal) != -1)
            break;
        }
        return;

      case 'J':
        if (Subtarget->isThumb()) {  // FIXME thumb2
          // This must be a constant between -255 and -1, for negated ADD
          // immediates. This can be used in GCC with an "n" modifier that
          // prints the negated value, for use with SUB instructions. It is
          // not useful otherwise but is implemented for compatibility.
          if (CVal >= -255 && CVal <= -1)
            break;
        } else {
          // This must be a constant between -4095 and 4095. It is not clear
          // what this constraint is intended for. Implemented for
          // compatibility with GCC.
          if (CVal >= -4095 && CVal <= 4095)
            break;
        }
        return;

      case 'K':
        if (Subtarget->isThumb1Only()) {
          // A 32-bit value where only one byte has a nonzero value. Exclude
          // zero to match GCC. This constraint is used by GCC internally for
          // constants that can be loaded with a move/shift combination.
          // It is not useful otherwise but is implemented for compatibility.
          if (CVal != 0 && ARM_AM::isThumbImmShiftedVal(CVal))
            break;
        } else if (Subtarget->isThumb2()) {
          // A constant whose bitwise inverse can be used as an immediate
          // value in a data-processing instruction. This can be used in GCC
          // with a "B" modifier that prints the inverted value, for use with
          // BIC and MVN instructions. It is not useful otherwise but is
          // implemented for compatibility.
          if (ARM_AM::getT2SOImmVal(~CVal) != -1)
            break;
        } else {
          // A constant whose bitwise inverse can be used as an immediate
          // value in a data-processing instruction. This can be used in GCC
          // with a "B" modifier that prints the inverted value, for use with
          // BIC and MVN instructions. It is not useful otherwise but is
          // implemented for compatibility.
          if (ARM_AM::getSOImmVal(~CVal) != -1)
            break;
        }
        return;

      case 'L':
        if (Subtarget->isThumb1Only()) {
          // This must be a constant between -7 and 7,
          // for 3-operand ADD/SUB immediate instructions.
          if (CVal >= -7 && CVal < 7)
            break;
        } else if (Subtarget->isThumb2()) {
          // A constant whose negation can be used as an immediate value in a
          // data-processing instruction. This can be used in GCC with an "n"
          // modifier that prints the negated value, for use with SUB
          // instructions. It is not useful otherwise but is implemented for
          // compatibility.
          if (ARM_AM::getT2SOImmVal(-CVal) != -1)
            break;
        } else {
          // A constant whose negation can be used as an immediate value in a
          // data-processing instruction. This can be used in GCC with an "n"
          // modifier that prints the negated value, for use with SUB
          // instructions. It is not useful otherwise but is implemented for
          // compatibility.
          if (ARM_AM::getSOImmVal(-CVal) != -1)
            break;
        }
        return;

      case 'M':
        if (Subtarget->isThumb()) { // FIXME thumb2
          // This must be a multiple of 4 between 0 and 1020, for
          // ADD sp + immediate.
          if ((CVal >= 0 && CVal <= 1020) && ((CVal & 3) == 0))
            break;
        } else {
          // A power of two or a constant between 0 and 32.  This is used in
          // GCC for the shift amount on shifted register operands, but it is
          // useful in general for any shift amounts.
          if ((CVal >= 0 && CVal <= 32) || ((CVal & (CVal - 1)) == 0))
            break;
        }
        return;

      case 'N':
        if (Subtarget->isThumb()) {  // FIXME thumb2
          // This must be a constant between 0 and 31, for shift amounts.
          if (CVal >= 0 && CVal <= 31)
            break;
        }
        return;

      case 'O':
        if (Subtarget->isThumb()) {  // FIXME thumb2
          // This must be a multiple of 4 between -508 and 508, for
          // ADD/SUB sp = sp + immediate.
          if ((CVal >= -508 && CVal <= 508) && ((CVal & 3) == 0))
            break;
        }
        return;
    }
    Result = DAG.getTargetConstant(CVal, Op.getValueType());
    break;
  }

  if (Result.getNode()) {
    Ops.push_back(Result);
    return;
  }
  return TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, hasMemory,
                                                      Ops, DAG);
}
