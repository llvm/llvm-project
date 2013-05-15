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

#define DEBUG_TYPE "arm-isel"
#include "ARMISelLowering.h"
#include "ARM.h"
#include "ARMCallingConv.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMPerfectShuffle.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "ARMTargetObjectFile.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

STATISTIC(NumTailCalls, "Number of tail calls");
STATISTIC(NumMovwMovt, "Number of GAs materialized with movw + movt");
STATISTIC(NumLoopByVals, "Number of loops generated for byval arguments");

// This option should go away when tail calls fully work.
static cl::opt<bool>
EnableARMTailCalls("arm-tail-calls", cl::Hidden,
  cl::desc("Generate tail calls (TEMPORARY OPTION)."),
  cl::init(false));

cl::opt<bool>
EnableARMLongCalls("arm-long-calls", cl::Hidden,
  cl::desc("Generate calls via indirect call instructions"),
  cl::init(false));

static cl::opt<bool>
ARMInterworking("arm-interworking", cl::Hidden,
  cl::desc("Enable / disable ARM interworking (for debugging only)"),
  cl::init(true));

namespace {
  class ARMCCState : public CCState {
  public:
    ARMCCState(CallingConv::ID CC, bool isVarArg, MachineFunction &MF,
               const TargetMachine &TM, SmallVector<CCValAssign, 16> &locs,
               LLVMContext &C, ParmContext PC)
        : CCState(CC, isVarArg, MF, TM, locs, C) {
      assert(((PC == Call) || (PC == Prologue)) &&
             "ARMCCState users must specify whether their context is call"
             "or prologue generation.");
      CallOrPrologue = PC;
    }
  };
}

// The APCS parameter registers.
static const uint16_t GPRArgRegs[] = {
  ARM::R0, ARM::R1, ARM::R2, ARM::R3
};

void ARMTargetLowering::addTypeForNEON(MVT VT, MVT PromotedLdStVT,
                                       MVT PromotedBitwiseVT) {
  if (VT != PromotedLdStVT) {
    setOperationAction(ISD::LOAD, VT, Promote);
    AddPromotedToType (ISD::LOAD, VT, PromotedLdStVT);

    setOperationAction(ISD::STORE, VT, Promote);
    AddPromotedToType (ISD::STORE, VT, PromotedLdStVT);
  }

  MVT ElemTy = VT.getVectorElementType();
  if (ElemTy != MVT::i64 && ElemTy != MVT::f64)
    setOperationAction(ISD::SETCC, VT, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
  if (ElemTy == MVT::i32) {
    setOperationAction(ISD::SINT_TO_FP, VT, Custom);
    setOperationAction(ISD::UINT_TO_FP, VT, Custom);
    setOperationAction(ISD::FP_TO_SINT, VT, Custom);
    setOperationAction(ISD::FP_TO_UINT, VT, Custom);
  } else {
    setOperationAction(ISD::SINT_TO_FP, VT, Expand);
    setOperationAction(ISD::UINT_TO_FP, VT, Expand);
    setOperationAction(ISD::FP_TO_SINT, VT, Expand);
    setOperationAction(ISD::FP_TO_UINT, VT, Expand);
  }
  setOperationAction(ISD::BUILD_VECTOR,      VT, Custom);
  setOperationAction(ISD::VECTOR_SHUFFLE,    VT, Custom);
  setOperationAction(ISD::CONCAT_VECTORS,    VT, Legal);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Legal);
  setOperationAction(ISD::SELECT,            VT, Expand);
  setOperationAction(ISD::SELECT_CC,         VT, Expand);
  setOperationAction(ISD::VSELECT,           VT, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Expand);
  if (VT.isInteger()) {
    setOperationAction(ISD::SHL, VT, Custom);
    setOperationAction(ISD::SRA, VT, Custom);
    setOperationAction(ISD::SRL, VT, Custom);
  }

  // Promote all bit-wise operations.
  if (VT.isInteger() && VT != PromotedBitwiseVT) {
    setOperationAction(ISD::AND, VT, Promote);
    AddPromotedToType (ISD::AND, VT, PromotedBitwiseVT);
    setOperationAction(ISD::OR,  VT, Promote);
    AddPromotedToType (ISD::OR,  VT, PromotedBitwiseVT);
    setOperationAction(ISD::XOR, VT, Promote);
    AddPromotedToType (ISD::XOR, VT, PromotedBitwiseVT);
  }

  // Neon does not support vector divide/remainder operations.
  setOperationAction(ISD::SDIV, VT, Expand);
  setOperationAction(ISD::UDIV, VT, Expand);
  setOperationAction(ISD::FDIV, VT, Expand);
  setOperationAction(ISD::SREM, VT, Expand);
  setOperationAction(ISD::UREM, VT, Expand);
  setOperationAction(ISD::FREM, VT, Expand);
}

void ARMTargetLowering::addDRTypeForNEON(MVT VT) {
  addRegisterClass(VT, &ARM::DPRRegClass);
  addTypeForNEON(VT, MVT::f64, MVT::v2i32);
}

void ARMTargetLowering::addQRTypeForNEON(MVT VT) {
  addRegisterClass(VT, &ARM::QPRRegClass);
  addTypeForNEON(VT, MVT::v2f64, MVT::v4i32);
}

static TargetLoweringObjectFile *createTLOF(TargetMachine &TM) {
  if (TM.getSubtarget<ARMSubtarget>().isTargetDarwin())
    return new TargetLoweringObjectFileMachO();

  return new ARMElfTargetObjectFile();
}

ARMTargetLowering::ARMTargetLowering(TargetMachine &TM)
    : TargetLowering(TM, createTLOF(TM)) {
  Subtarget = &TM.getSubtarget<ARMSubtarget>();
  RegInfo = TM.getRegisterInfo();
  Itins = TM.getInstrItineraryData();

  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);

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

  if (Subtarget->isAAPCS_ABI() && !Subtarget->isTargetDarwin()) {
    // Double-precision floating-point arithmetic helper functions
    // RTABI chapter 4.1.2, Table 2
    setLibcallName(RTLIB::ADD_F64, "__aeabi_dadd");
    setLibcallName(RTLIB::DIV_F64, "__aeabi_ddiv");
    setLibcallName(RTLIB::MUL_F64, "__aeabi_dmul");
    setLibcallName(RTLIB::SUB_F64, "__aeabi_dsub");
    setLibcallCallingConv(RTLIB::ADD_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::DIV_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::MUL_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SUB_F64, CallingConv::ARM_AAPCS);

    // Double-precision floating-point comparison helper functions
    // RTABI chapter 4.1.2, Table 3
    setLibcallName(RTLIB::OEQ_F64, "__aeabi_dcmpeq");
    setCmpLibcallCC(RTLIB::OEQ_F64, ISD::SETNE);
    setLibcallName(RTLIB::UNE_F64, "__aeabi_dcmpeq");
    setCmpLibcallCC(RTLIB::UNE_F64, ISD::SETEQ);
    setLibcallName(RTLIB::OLT_F64, "__aeabi_dcmplt");
    setCmpLibcallCC(RTLIB::OLT_F64, ISD::SETNE);
    setLibcallName(RTLIB::OLE_F64, "__aeabi_dcmple");
    setCmpLibcallCC(RTLIB::OLE_F64, ISD::SETNE);
    setLibcallName(RTLIB::OGE_F64, "__aeabi_dcmpge");
    setCmpLibcallCC(RTLIB::OGE_F64, ISD::SETNE);
    setLibcallName(RTLIB::OGT_F64, "__aeabi_dcmpgt");
    setCmpLibcallCC(RTLIB::OGT_F64, ISD::SETNE);
    setLibcallName(RTLIB::UO_F64,  "__aeabi_dcmpun");
    setCmpLibcallCC(RTLIB::UO_F64,  ISD::SETNE);
    setLibcallName(RTLIB::O_F64,   "__aeabi_dcmpun");
    setCmpLibcallCC(RTLIB::O_F64,   ISD::SETEQ);
    setLibcallCallingConv(RTLIB::OEQ_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UNE_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::OLT_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::OLE_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::OGE_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::OGT_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UO_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::O_F64, CallingConv::ARM_AAPCS);

    // Single-precision floating-point arithmetic helper functions
    // RTABI chapter 4.1.2, Table 4
    setLibcallName(RTLIB::ADD_F32, "__aeabi_fadd");
    setLibcallName(RTLIB::DIV_F32, "__aeabi_fdiv");
    setLibcallName(RTLIB::MUL_F32, "__aeabi_fmul");
    setLibcallName(RTLIB::SUB_F32, "__aeabi_fsub");
    setLibcallCallingConv(RTLIB::ADD_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::DIV_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::MUL_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SUB_F32, CallingConv::ARM_AAPCS);

    // Single-precision floating-point comparison helper functions
    // RTABI chapter 4.1.2, Table 5
    setLibcallName(RTLIB::OEQ_F32, "__aeabi_fcmpeq");
    setCmpLibcallCC(RTLIB::OEQ_F32, ISD::SETNE);
    setLibcallName(RTLIB::UNE_F32, "__aeabi_fcmpeq");
    setCmpLibcallCC(RTLIB::UNE_F32, ISD::SETEQ);
    setLibcallName(RTLIB::OLT_F32, "__aeabi_fcmplt");
    setCmpLibcallCC(RTLIB::OLT_F32, ISD::SETNE);
    setLibcallName(RTLIB::OLE_F32, "__aeabi_fcmple");
    setCmpLibcallCC(RTLIB::OLE_F32, ISD::SETNE);
    setLibcallName(RTLIB::OGE_F32, "__aeabi_fcmpge");
    setCmpLibcallCC(RTLIB::OGE_F32, ISD::SETNE);
    setLibcallName(RTLIB::OGT_F32, "__aeabi_fcmpgt");
    setCmpLibcallCC(RTLIB::OGT_F32, ISD::SETNE);
    setLibcallName(RTLIB::UO_F32,  "__aeabi_fcmpun");
    setCmpLibcallCC(RTLIB::UO_F32,  ISD::SETNE);
    setLibcallName(RTLIB::O_F32,   "__aeabi_fcmpun");
    setCmpLibcallCC(RTLIB::O_F32,   ISD::SETEQ);
    setLibcallCallingConv(RTLIB::OEQ_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UNE_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::OLT_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::OLE_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::OGE_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::OGT_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UO_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::O_F32, CallingConv::ARM_AAPCS);

    // Floating-point to integer conversions.
    // RTABI chapter 4.1.2, Table 6
    setLibcallName(RTLIB::FPTOSINT_F64_I32, "__aeabi_d2iz");
    setLibcallName(RTLIB::FPTOUINT_F64_I32, "__aeabi_d2uiz");
    setLibcallName(RTLIB::FPTOSINT_F64_I64, "__aeabi_d2lz");
    setLibcallName(RTLIB::FPTOUINT_F64_I64, "__aeabi_d2ulz");
    setLibcallName(RTLIB::FPTOSINT_F32_I32, "__aeabi_f2iz");
    setLibcallName(RTLIB::FPTOUINT_F32_I32, "__aeabi_f2uiz");
    setLibcallName(RTLIB::FPTOSINT_F32_I64, "__aeabi_f2lz");
    setLibcallName(RTLIB::FPTOUINT_F32_I64, "__aeabi_f2ulz");
    setLibcallCallingConv(RTLIB::FPTOSINT_F64_I32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::FPTOUINT_F64_I32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::FPTOSINT_F64_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::FPTOUINT_F64_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::FPTOSINT_F32_I32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::FPTOUINT_F32_I32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::FPTOSINT_F32_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::FPTOUINT_F32_I64, CallingConv::ARM_AAPCS);

    // Conversions between floating types.
    // RTABI chapter 4.1.2, Table 7
    setLibcallName(RTLIB::FPROUND_F64_F32, "__aeabi_d2f");
    setLibcallName(RTLIB::FPEXT_F32_F64,   "__aeabi_f2d");
    setLibcallCallingConv(RTLIB::FPROUND_F64_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::FPEXT_F32_F64, CallingConv::ARM_AAPCS);

    // Integer to floating-point conversions.
    // RTABI chapter 4.1.2, Table 8
    setLibcallName(RTLIB::SINTTOFP_I32_F64, "__aeabi_i2d");
    setLibcallName(RTLIB::UINTTOFP_I32_F64, "__aeabi_ui2d");
    setLibcallName(RTLIB::SINTTOFP_I64_F64, "__aeabi_l2d");
    setLibcallName(RTLIB::UINTTOFP_I64_F64, "__aeabi_ul2d");
    setLibcallName(RTLIB::SINTTOFP_I32_F32, "__aeabi_i2f");
    setLibcallName(RTLIB::UINTTOFP_I32_F32, "__aeabi_ui2f");
    setLibcallName(RTLIB::SINTTOFP_I64_F32, "__aeabi_l2f");
    setLibcallName(RTLIB::UINTTOFP_I64_F32, "__aeabi_ul2f");
    setLibcallCallingConv(RTLIB::SINTTOFP_I32_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UINTTOFP_I32_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SINTTOFP_I64_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UINTTOFP_I64_F64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SINTTOFP_I32_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UINTTOFP_I32_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SINTTOFP_I64_F32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UINTTOFP_I64_F32, CallingConv::ARM_AAPCS);

    // Long long helper functions
    // RTABI chapter 4.2, Table 9
    setLibcallName(RTLIB::MUL_I64,  "__aeabi_lmul");
    setLibcallName(RTLIB::SHL_I64, "__aeabi_llsl");
    setLibcallName(RTLIB::SRL_I64, "__aeabi_llsr");
    setLibcallName(RTLIB::SRA_I64, "__aeabi_lasr");
    setLibcallCallingConv(RTLIB::MUL_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SDIV_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UDIV_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SHL_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SRL_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SRA_I64, CallingConv::ARM_AAPCS);

    // Integer division functions
    // RTABI chapter 4.3.1
    setLibcallName(RTLIB::SDIV_I8,  "__aeabi_idiv");
    setLibcallName(RTLIB::SDIV_I16, "__aeabi_idiv");
    setLibcallName(RTLIB::SDIV_I32, "__aeabi_idiv");
    setLibcallName(RTLIB::SDIV_I64, "__aeabi_ldivmod");
    setLibcallName(RTLIB::UDIV_I8,  "__aeabi_uidiv");
    setLibcallName(RTLIB::UDIV_I16, "__aeabi_uidiv");
    setLibcallName(RTLIB::UDIV_I32, "__aeabi_uidiv");
    setLibcallName(RTLIB::UDIV_I64, "__aeabi_uldivmod");
    setLibcallCallingConv(RTLIB::SDIV_I8, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SDIV_I16, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SDIV_I32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::SDIV_I64, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UDIV_I8, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UDIV_I16, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UDIV_I32, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::UDIV_I64, CallingConv::ARM_AAPCS);

    // Memory operations
    // RTABI chapter 4.3.4
    setLibcallName(RTLIB::MEMCPY,  "__aeabi_memcpy");
    setLibcallName(RTLIB::MEMMOVE, "__aeabi_memmove");
    setLibcallName(RTLIB::MEMSET,  "__aeabi_memset");
    setLibcallCallingConv(RTLIB::MEMCPY, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::MEMMOVE, CallingConv::ARM_AAPCS);
    setLibcallCallingConv(RTLIB::MEMSET, CallingConv::ARM_AAPCS);
  }

  // Use divmod compiler-rt calls for iOS 5.0 and later.
  if (Subtarget->getTargetTriple().getOS() == Triple::IOS &&
      !Subtarget->getTargetTriple().isOSVersionLT(5, 0)) {
    setLibcallName(RTLIB::SDIVREM_I32, "__divmodsi4");
    setLibcallName(RTLIB::UDIVREM_I32, "__udivmodsi4");
  }

  if (Subtarget->isThumb1Only())
    addRegisterClass(MVT::i32, &ARM::tGPRRegClass);
  else
    addRegisterClass(MVT::i32, &ARM::GPRRegClass);
  if (!TM.Options.UseSoftFloat && Subtarget->hasVFP2() &&
      !Subtarget->isThumb1Only()) {
    addRegisterClass(MVT::f32, &ARM::SPRRegClass);
    if (!Subtarget->isFPOnlySP())
      addRegisterClass(MVT::f64, &ARM::DPRRegClass);

    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  }

  for (unsigned VT = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
       VT <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++VT) {
    for (unsigned InnerVT = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
         InnerVT <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++InnerVT)
      setTruncStoreAction((MVT::SimpleValueType)VT,
                          (MVT::SimpleValueType)InnerVT, Expand);
    setLoadExtAction(ISD::SEXTLOAD, (MVT::SimpleValueType)VT, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, (MVT::SimpleValueType)VT, Expand);
    setLoadExtAction(ISD::EXTLOAD, (MVT::SimpleValueType)VT, Expand);
  }

  setOperationAction(ISD::ConstantFP, MVT::f32, Custom);

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

    // v2f64 is legal so that QR subregs can be extracted as f64 elements, but
    // neither Neon nor VFP support any arithmetic operations on it.
    // The same with v4f32. But keep in mind that vadd, vsub, vmul are natively
    // supported for v4f32.
    setOperationAction(ISD::FADD, MVT::v2f64, Expand);
    setOperationAction(ISD::FSUB, MVT::v2f64, Expand);
    setOperationAction(ISD::FMUL, MVT::v2f64, Expand);
    // FIXME: Code duplication: FDIV and FREM are expanded always, see
    // ARMTargetLowering::addTypeForNEON method for details.
    setOperationAction(ISD::FDIV, MVT::v2f64, Expand);
    setOperationAction(ISD::FREM, MVT::v2f64, Expand);
    // FIXME: Create unittest.
    // In another words, find a way when "copysign" appears in DAG with vector
    // operands.
    setOperationAction(ISD::FCOPYSIGN, MVT::v2f64, Expand);
    // FIXME: Code duplication: SETCC has custom operation action, see
    // ARMTargetLowering::addTypeForNEON method for details.
    setOperationAction(ISD::SETCC, MVT::v2f64, Expand);
    // FIXME: Create unittest for FNEG and for FABS.
    setOperationAction(ISD::FNEG, MVT::v2f64, Expand);
    setOperationAction(ISD::FABS, MVT::v2f64, Expand);
    setOperationAction(ISD::FSQRT, MVT::v2f64, Expand);
    setOperationAction(ISD::FSIN, MVT::v2f64, Expand);
    setOperationAction(ISD::FCOS, MVT::v2f64, Expand);
    setOperationAction(ISD::FPOWI, MVT::v2f64, Expand);
    setOperationAction(ISD::FPOW, MVT::v2f64, Expand);
    setOperationAction(ISD::FLOG, MVT::v2f64, Expand);
    setOperationAction(ISD::FLOG2, MVT::v2f64, Expand);
    setOperationAction(ISD::FLOG10, MVT::v2f64, Expand);
    setOperationAction(ISD::FEXP, MVT::v2f64, Expand);
    setOperationAction(ISD::FEXP2, MVT::v2f64, Expand);
    // FIXME: Create unittest for FCEIL, FTRUNC, FRINT, FNEARBYINT, FFLOOR.
    setOperationAction(ISD::FCEIL, MVT::v2f64, Expand);
    setOperationAction(ISD::FTRUNC, MVT::v2f64, Expand);
    setOperationAction(ISD::FRINT, MVT::v2f64, Expand);
    setOperationAction(ISD::FNEARBYINT, MVT::v2f64, Expand);
    setOperationAction(ISD::FFLOOR, MVT::v2f64, Expand);
    setOperationAction(ISD::FMA, MVT::v2f64, Expand);

    setOperationAction(ISD::FSQRT, MVT::v4f32, Expand);
    setOperationAction(ISD::FSIN, MVT::v4f32, Expand);
    setOperationAction(ISD::FCOS, MVT::v4f32, Expand);
    setOperationAction(ISD::FPOWI, MVT::v4f32, Expand);
    setOperationAction(ISD::FPOW, MVT::v4f32, Expand);
    setOperationAction(ISD::FLOG, MVT::v4f32, Expand);
    setOperationAction(ISD::FLOG2, MVT::v4f32, Expand);
    setOperationAction(ISD::FLOG10, MVT::v4f32, Expand);
    setOperationAction(ISD::FEXP, MVT::v4f32, Expand);
    setOperationAction(ISD::FEXP2, MVT::v4f32, Expand);
    setOperationAction(ISD::FCEIL, MVT::v4f32, Expand);
    setOperationAction(ISD::FTRUNC, MVT::v4f32, Expand);
    setOperationAction(ISD::FRINT, MVT::v4f32, Expand);
    setOperationAction(ISD::FNEARBYINT, MVT::v4f32, Expand);
    setOperationAction(ISD::FFLOOR, MVT::v4f32, Expand);

    // Mark v2f32 intrinsics.
    setOperationAction(ISD::FSQRT, MVT::v2f32, Expand);
    setOperationAction(ISD::FSIN, MVT::v2f32, Expand);
    setOperationAction(ISD::FCOS, MVT::v2f32, Expand);
    setOperationAction(ISD::FPOWI, MVT::v2f32, Expand);
    setOperationAction(ISD::FPOW, MVT::v2f32, Expand);
    setOperationAction(ISD::FLOG, MVT::v2f32, Expand);
    setOperationAction(ISD::FLOG2, MVT::v2f32, Expand);
    setOperationAction(ISD::FLOG10, MVT::v2f32, Expand);
    setOperationAction(ISD::FEXP, MVT::v2f32, Expand);
    setOperationAction(ISD::FEXP2, MVT::v2f32, Expand);
    setOperationAction(ISD::FCEIL, MVT::v2f32, Expand);
    setOperationAction(ISD::FTRUNC, MVT::v2f32, Expand);
    setOperationAction(ISD::FRINT, MVT::v2f32, Expand);
    setOperationAction(ISD::FNEARBYINT, MVT::v2f32, Expand);
    setOperationAction(ISD::FFLOOR, MVT::v2f32, Expand);

    // Neon does not support some operations on v1i64 and v2i64 types.
    setOperationAction(ISD::MUL, MVT::v1i64, Expand);
    // Custom handling for some quad-vector types to detect VMULL.
    setOperationAction(ISD::MUL, MVT::v8i16, Custom);
    setOperationAction(ISD::MUL, MVT::v4i32, Custom);
    setOperationAction(ISD::MUL, MVT::v2i64, Custom);
    // Custom handling for some vector types to avoid expensive expansions
    setOperationAction(ISD::SDIV, MVT::v4i16, Custom);
    setOperationAction(ISD::SDIV, MVT::v8i8, Custom);
    setOperationAction(ISD::UDIV, MVT::v4i16, Custom);
    setOperationAction(ISD::UDIV, MVT::v8i8, Custom);
    setOperationAction(ISD::SETCC, MVT::v1i64, Expand);
    setOperationAction(ISD::SETCC, MVT::v2i64, Expand);
    // Neon does not have single instruction SINT_TO_FP and UINT_TO_FP with
    // a destination type that is wider than the source, and nor does
    // it have a FP_TO_[SU]INT instruction with a narrower destination than
    // source.
    setOperationAction(ISD::SINT_TO_FP, MVT::v4i16, Custom);
    setOperationAction(ISD::UINT_TO_FP, MVT::v4i16, Custom);
    setOperationAction(ISD::FP_TO_UINT, MVT::v4i16, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::v4i16, Custom);

    setOperationAction(ISD::FP_ROUND,   MVT::v2f32, Expand);
    setOperationAction(ISD::FP_EXTEND,  MVT::v2f64, Expand);

    // Custom expand long extensions to vectors.
    setOperationAction(ISD::SIGN_EXTEND, MVT::v8i32,  Custom);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v8i32,  Custom);
    setOperationAction(ISD::SIGN_EXTEND, MVT::v4i64,  Custom);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v4i64,  Custom);
    setOperationAction(ISD::SIGN_EXTEND, MVT::v16i32, Custom);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v16i32, Custom);
    setOperationAction(ISD::SIGN_EXTEND, MVT::v8i64,  Custom);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v8i64,  Custom);

    // NEON does not have single instruction CTPOP for vectors with element
    // types wider than 8-bits.  However, custom lowering can leverage the
    // v8i8/v16i8 vcnt instruction.
    setOperationAction(ISD::CTPOP,      MVT::v2i32, Custom);
    setOperationAction(ISD::CTPOP,      MVT::v4i32, Custom);
    setOperationAction(ISD::CTPOP,      MVT::v4i16, Custom);
    setOperationAction(ISD::CTPOP,      MVT::v8i16, Custom);

    // NEON only has FMA instructions as of VFP4.
    if (!Subtarget->hasVFP4()) {
      setOperationAction(ISD::FMA, MVT::v2f32, Expand);
      setOperationAction(ISD::FMA, MVT::v4f32, Expand);
    }

    setTargetDAGCombine(ISD::INTRINSIC_VOID);
    setTargetDAGCombine(ISD::INTRINSIC_W_CHAIN);
    setTargetDAGCombine(ISD::INTRINSIC_WO_CHAIN);
    setTargetDAGCombine(ISD::SHL);
    setTargetDAGCombine(ISD::SRL);
    setTargetDAGCombine(ISD::SRA);
    setTargetDAGCombine(ISD::SIGN_EXTEND);
    setTargetDAGCombine(ISD::ZERO_EXTEND);
    setTargetDAGCombine(ISD::ANY_EXTEND);
    setTargetDAGCombine(ISD::SELECT_CC);
    setTargetDAGCombine(ISD::BUILD_VECTOR);
    setTargetDAGCombine(ISD::VECTOR_SHUFFLE);
    setTargetDAGCombine(ISD::INSERT_VECTOR_ELT);
    setTargetDAGCombine(ISD::STORE);
    setTargetDAGCombine(ISD::FP_TO_SINT);
    setTargetDAGCombine(ISD::FP_TO_UINT);
    setTargetDAGCombine(ISD::FDIV);

    // It is legal to extload from v4i8 to v4i16 or v4i32.
    MVT Tys[6] = {MVT::v8i8, MVT::v4i8, MVT::v2i8,
                  MVT::v4i16, MVT::v2i16,
                  MVT::v2i32};
    for (unsigned i = 0; i < 6; ++i) {
      setLoadExtAction(ISD::EXTLOAD, Tys[i], Legal);
      setLoadExtAction(ISD::ZEXTLOAD, Tys[i], Legal);
      setLoadExtAction(ISD::SEXTLOAD, Tys[i], Legal);
    }
  }

  // ARM and Thumb2 support UMLAL/SMLAL.
  if (!Subtarget->isThumb1Only())
    setTargetDAGCombine(ISD::ADDC);


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
  setOperationAction(ISD::MUL,     MVT::i64, Expand);
  setOperationAction(ISD::MULHU,   MVT::i32, Expand);
  if (Subtarget->isThumb1Only()) {
    setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
    setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  }
  if (Subtarget->isThumb1Only() || !Subtarget->hasV6Ops()
      || (Subtarget->isThumb2() && !Subtarget->hasThumb2DSP()))
    setOperationAction(ISD::MULHS, MVT::i32, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRL,       MVT::i64, Custom);
  setOperationAction(ISD::SRA,       MVT::i64, Custom);

  if (!Subtarget->isThumb1Only()) {
    // FIXME: We should do this for Thumb1 as well.
    setOperationAction(ISD::ADDC,    MVT::i32, Custom);
    setOperationAction(ISD::ADDE,    MVT::i32, Custom);
    setOperationAction(ISD::SUBC,    MVT::i32, Custom);
    setOperationAction(ISD::SUBE,    MVT::i32, Custom);
  }

  // ARM does not have ROTL.
  setOperationAction(ISD::ROTL,  MVT::i32, Expand);
  setOperationAction(ISD::CTTZ,  MVT::i32, Custom);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  if (!Subtarget->hasV5TOps() || Subtarget->isThumb1Only())
    setOperationAction(ISD::CTLZ, MVT::i32, Expand);

  // These just redirect to CTTZ and CTLZ on ARM.
  setOperationAction(ISD::CTTZ_ZERO_UNDEF  , MVT::i32  , Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF  , MVT::i32  , Expand);

  // Only ARMv6 has BSWAP.
  if (!Subtarget->hasV6Ops())
    setOperationAction(ISD::BSWAP, MVT::i32, Expand);

  if (!(Subtarget->hasDivide() && Subtarget->isThumb2()) &&
      !(Subtarget->hasDivideInARMMode() && !Subtarget->isThumb())) {
    // These are expanded into libcalls if the cpu doesn't have HW divider.
    setOperationAction(ISD::SDIV,  MVT::i32, Expand);
    setOperationAction(ISD::UDIV,  MVT::i32, Expand);
  }
  setOperationAction(ISD::SREM,  MVT::i32, Expand);
  setOperationAction(ISD::UREM,  MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32,   Custom);
  setOperationAction(ISD::GLOBAL_OFFSET_TABLE, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  setOperationAction(ISD::BlockAddress, MVT::i32, Custom);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  // Use the default implementation.
  setOperationAction(ISD::VASTART,            MVT::Other, Custom);
  setOperationAction(ISD::VAARG,              MVT::Other, Expand);
  setOperationAction(ISD::VACOPY,             MVT::Other, Expand);
  setOperationAction(ISD::VAEND,              MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE,          MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE,       MVT::Other, Expand);

  if (!Subtarget->isTargetDarwin()) {
    // Non-Darwin platforms may return values in these registers via the
    // personality function.
    setOperationAction(ISD::EHSELECTION,      MVT::i32,   Expand);
    setOperationAction(ISD::EXCEPTIONADDR,    MVT::i32,   Expand);
    setExceptionPointerRegister(ARM::R0);
    setExceptionSelectorRegister(ARM::R1);
  }

  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);
  // ARMv6 Thumb1 (except for CPUs that support dmb / dsb) and earlier use
  // the default expansion.
  // FIXME: This should be checking for v6k, not just v6.
  if (Subtarget->hasDataBarrier() ||
      (Subtarget->hasV6Ops() && !Subtarget->isThumb())) {
    // membarrier needs custom lowering; the rest are legal and handled
    // normally.
    setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);
    // Custom lowering for 64-bit ops
    setOperationAction(ISD::ATOMIC_LOAD_ADD,  MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_SUB,  MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_AND,  MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_OR,   MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_XOR,  MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_SWAP,      MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_MIN,  MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_MAX,  MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_UMIN, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_UMAX, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_CMP_SWAP,  MVT::i64, Custom);
    // Automatically insert fences (dmb ist) around ATOMIC_SWAP etc.
    setInsertFencesForAtomic(true);
  } else {
    // Set them all for expansion, which will force libcalls.
    setOperationAction(ISD::ATOMIC_FENCE,   MVT::Other, Expand);
    setOperationAction(ISD::ATOMIC_CMP_SWAP,  MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_SWAP,      MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_ADD,  MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_SUB,  MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_AND,  MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_OR,   MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_XOR,  MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_NAND, MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_MIN, MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_MAX, MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_UMIN, MVT::i32, Expand);
    setOperationAction(ISD::ATOMIC_LOAD_UMAX, MVT::i32, Expand);
    // Mark ATOMIC_LOAD and ATOMIC_STORE custom so we can handle the
    // Unordered/Monotonic case.
    setOperationAction(ISD::ATOMIC_LOAD, MVT::i32, Custom);
    setOperationAction(ISD::ATOMIC_STORE, MVT::i32, Custom);
  }

  setOperationAction(ISD::PREFETCH,         MVT::Other, Custom);

  // Requires SXTB/SXTH, available on v6 and up in both ARM and Thumb modes.
  if (!Subtarget->hasV6Ops()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8,  Expand);
  }
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  if (!TM.Options.UseSoftFloat && Subtarget->hasVFP2() &&
      !Subtarget->isThumb1Only()) {
    // Turn f64->i64 into VMOVRRD, i64 -> f64 to VMOVDRR
    // iff target supports vfp2.
    setOperationAction(ISD::BITCAST, MVT::i64, Custom);
    setOperationAction(ISD::FLT_ROUNDS_, MVT::i32, Custom);
  }

  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  if (Subtarget->isTargetDarwin()) {
    setOperationAction(ISD::EH_SJLJ_SETJMP, MVT::i32, Custom);
    setOperationAction(ISD::EH_SJLJ_LONGJMP, MVT::Other, Custom);
    setLibcallName(RTLIB::UNWIND_RESUME, "_Unwind_SjLj_Resume");
  }

  setOperationAction(ISD::SETCC,     MVT::i32, Expand);
  setOperationAction(ISD::SETCC,     MVT::f32, Expand);
  setOperationAction(ISD::SETCC,     MVT::f64, Expand);
  setOperationAction(ISD::SELECT,    MVT::i32, Custom);
  setOperationAction(ISD::SELECT,    MVT::f32, Custom);
  setOperationAction(ISD::SELECT,    MVT::f64, Custom);
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
  setOperationAction(ISD::FSINCOS,   MVT::f64, Expand);
  setOperationAction(ISD::FSINCOS,   MVT::f32, Expand);
  setOperationAction(ISD::FREM,      MVT::f64, Expand);
  setOperationAction(ISD::FREM,      MVT::f32, Expand);
  if (!TM.Options.UseSoftFloat && Subtarget->hasVFP2() &&
      !Subtarget->isThumb1Only()) {
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);
  }
  setOperationAction(ISD::FPOW,      MVT::f64, Expand);
  setOperationAction(ISD::FPOW,      MVT::f32, Expand);

  if (!Subtarget->hasVFP4()) {
    setOperationAction(ISD::FMA, MVT::f64, Expand);
    setOperationAction(ISD::FMA, MVT::f32, Expand);
  }

  // Various VFP goodness
  if (!TM.Options.UseSoftFloat && !Subtarget->isThumb1Only()) {
    // int <-> fp are custom expanded into bit_convert + ARMISD ops.
    if (Subtarget->hasVFP2()) {
      setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
      setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);
      setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
      setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
    }
    // Special handling for half-precision FP.
    if (!Subtarget->hasFP16()) {
      setOperationAction(ISD::FP16_TO_FP32, MVT::f32, Expand);
      setOperationAction(ISD::FP32_TO_FP16, MVT::i32, Expand);
    }
  }

  // We have target-specific dag combine patterns for the following nodes:
  // ARMISD::VMOVRRD  - No need to call setTargetDAGCombine
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::SUB);
  setTargetDAGCombine(ISD::MUL);
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::XOR);

  if (Subtarget->hasV6Ops())
    setTargetDAGCombine(ISD::SRL);

  setStackPointerRegisterToSaveRestore(ARM::SP);

  if (TM.Options.UseSoftFloat || Subtarget->isThumb1Only() ||
      !Subtarget->hasVFP2())
    setSchedulingPreference(Sched::RegPressure);
  else
    setSchedulingPreference(Sched::Hybrid);

  //// temporary - rewrite interface to use type
  MaxStoresPerMemset = 8;
  MaxStoresPerMemsetOptSize = Subtarget->isTargetDarwin() ? 8 : 4;
  MaxStoresPerMemcpy = 4; // For @llvm.memcpy -> sequence of stores
  MaxStoresPerMemcpyOptSize = Subtarget->isTargetDarwin() ? 4 : 2;
  MaxStoresPerMemmove = 4; // For @llvm.memmove -> sequence of stores
  MaxStoresPerMemmoveOptSize = Subtarget->isTargetDarwin() ? 4 : 2;

  // On ARM arguments smaller than 4 bytes are extended, so all arguments
  // are at least 4 bytes aligned.
  setMinStackArgumentAlignment(4);

  // Prefer likely predicted branches to selects on out-of-order cores.
  PredictableSelectIsExpensive = Subtarget->isLikeA9();

  setMinFunctionAlignment(Subtarget->isThumb() ? 1 : 2);
}

// FIXME: It might make sense to define the representative register class as the
// nearest super-register that has a non-null superset. For example, DPR_VFP2 is
// a super-register of SPR, and DPR is a superset if DPR_VFP2. Consequently,
// SPR's representative would be DPR_VFP2. This should work well if register
// pressure tracking were modified such that a register use would increment the
// pressure of the register class's representative and all of it's super
// classes' representatives transitively. We have not implemented this because
// of the difficulty prior to coalescing of modeling operand register classes
// due to the common occurrence of cross class copies and subregister insertions
// and extractions.
std::pair<const TargetRegisterClass*, uint8_t>
ARMTargetLowering::findRepresentativeClass(MVT VT) const{
  const TargetRegisterClass *RRC = 0;
  uint8_t Cost = 1;
  switch (VT.SimpleTy) {
  default:
    return TargetLowering::findRepresentativeClass(VT);
  // Use DPR as representative register class for all floating point
  // and vector types. Since there are 32 SPR registers and 32 DPR registers so
  // the cost is 1 for both f32 and f64.
  case MVT::f32: case MVT::f64: case MVT::v8i8: case MVT::v4i16:
  case MVT::v2i32: case MVT::v1i64: case MVT::v2f32:
    RRC = &ARM::DPRRegClass;
    // When NEON is used for SP, only half of the register file is available
    // because operations that define both SP and DP results will be constrained
    // to the VFP2 class (D0-D15). We currently model this constraint prior to
    // coalescing by double-counting the SP regs. See the FIXME above.
    if (Subtarget->useNEONForSinglePrecisionFP())
      Cost = 2;
    break;
  case MVT::v16i8: case MVT::v8i16: case MVT::v4i32: case MVT::v2i64:
  case MVT::v4f32: case MVT::v2f64:
    RRC = &ARM::DPRRegClass;
    Cost = 2;
    break;
  case MVT::v4i64:
    RRC = &ARM::DPRRegClass;
    Cost = 4;
    break;
  case MVT::v8i64:
    RRC = &ARM::DPRRegClass;
    Cost = 8;
    break;
  }
  return std::make_pair(RRC, Cost);
}

const char *ARMTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case ARMISD::Wrapper:       return "ARMISD::Wrapper";
  case ARMISD::WrapperDYN:    return "ARMISD::WrapperDYN";
  case ARMISD::WrapperPIC:    return "ARMISD::WrapperPIC";
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
  case ARMISD::CMN:           return "ARMISD::CMN";
  case ARMISD::CMPZ:          return "ARMISD::CMPZ";
  case ARMISD::CMPFP:         return "ARMISD::CMPFP";
  case ARMISD::CMPFPw0:       return "ARMISD::CMPFPw0";
  case ARMISD::BCC_i64:       return "ARMISD::BCC_i64";
  case ARMISD::FMSTAT:        return "ARMISD::FMSTAT";

  case ARMISD::CMOV:          return "ARMISD::CMOV";

  case ARMISD::RBIT:          return "ARMISD::RBIT";

  case ARMISD::FTOSI:         return "ARMISD::FTOSI";
  case ARMISD::FTOUI:         return "ARMISD::FTOUI";
  case ARMISD::SITOF:         return "ARMISD::SITOF";
  case ARMISD::UITOF:         return "ARMISD::UITOF";

  case ARMISD::SRL_FLAG:      return "ARMISD::SRL_FLAG";
  case ARMISD::SRA_FLAG:      return "ARMISD::SRA_FLAG";
  case ARMISD::RRX:           return "ARMISD::RRX";

  case ARMISD::ADDC:          return "ARMISD::ADDC";
  case ARMISD::ADDE:          return "ARMISD::ADDE";
  case ARMISD::SUBC:          return "ARMISD::SUBC";
  case ARMISD::SUBE:          return "ARMISD::SUBE";

  case ARMISD::VMOVRRD:       return "ARMISD::VMOVRRD";
  case ARMISD::VMOVDRR:       return "ARMISD::VMOVDRR";

  case ARMISD::EH_SJLJ_SETJMP: return "ARMISD::EH_SJLJ_SETJMP";
  case ARMISD::EH_SJLJ_LONGJMP:return "ARMISD::EH_SJLJ_LONGJMP";

  case ARMISD::TC_RETURN:     return "ARMISD::TC_RETURN";

  case ARMISD::THREAD_POINTER:return "ARMISD::THREAD_POINTER";

  case ARMISD::DYN_ALLOC:     return "ARMISD::DYN_ALLOC";

  case ARMISD::MEMBARRIER:    return "ARMISD::MEMBARRIER";
  case ARMISD::MEMBARRIER_MCR: return "ARMISD::MEMBARRIER_MCR";

  case ARMISD::PRELOAD:       return "ARMISD::PRELOAD";

  case ARMISD::VCEQ:          return "ARMISD::VCEQ";
  case ARMISD::VCEQZ:         return "ARMISD::VCEQZ";
  case ARMISD::VCGE:          return "ARMISD::VCGE";
  case ARMISD::VCGEZ:         return "ARMISD::VCGEZ";
  case ARMISD::VCLEZ:         return "ARMISD::VCLEZ";
  case ARMISD::VCGEU:         return "ARMISD::VCGEU";
  case ARMISD::VCGT:          return "ARMISD::VCGT";
  case ARMISD::VCGTZ:         return "ARMISD::VCGTZ";
  case ARMISD::VCLTZ:         return "ARMISD::VCLTZ";
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
  case ARMISD::VMOVIMM:       return "ARMISD::VMOVIMM";
  case ARMISD::VMVNIMM:       return "ARMISD::VMVNIMM";
  case ARMISD::VMOVFPIMM:     return "ARMISD::VMOVFPIMM";
  case ARMISD::VDUP:          return "ARMISD::VDUP";
  case ARMISD::VDUPLANE:      return "ARMISD::VDUPLANE";
  case ARMISD::VEXT:          return "ARMISD::VEXT";
  case ARMISD::VREV64:        return "ARMISD::VREV64";
  case ARMISD::VREV32:        return "ARMISD::VREV32";
  case ARMISD::VREV16:        return "ARMISD::VREV16";
  case ARMISD::VZIP:          return "ARMISD::VZIP";
  case ARMISD::VUZP:          return "ARMISD::VUZP";
  case ARMISD::VTRN:          return "ARMISD::VTRN";
  case ARMISD::VTBL1:         return "ARMISD::VTBL1";
  case ARMISD::VTBL2:         return "ARMISD::VTBL2";
  case ARMISD::VMULLs:        return "ARMISD::VMULLs";
  case ARMISD::VMULLu:        return "ARMISD::VMULLu";
  case ARMISD::UMLAL:         return "ARMISD::UMLAL";
  case ARMISD::SMLAL:         return "ARMISD::SMLAL";
  case ARMISD::BUILD_VECTOR:  return "ARMISD::BUILD_VECTOR";
  case ARMISD::FMAX:          return "ARMISD::FMAX";
  case ARMISD::FMIN:          return "ARMISD::FMIN";
  case ARMISD::BFI:           return "ARMISD::BFI";
  case ARMISD::VORRIMM:       return "ARMISD::VORRIMM";
  case ARMISD::VBICIMM:       return "ARMISD::VBICIMM";
  case ARMISD::VBSL:          return "ARMISD::VBSL";
  case ARMISD::VLD2DUP:       return "ARMISD::VLD2DUP";
  case ARMISD::VLD3DUP:       return "ARMISD::VLD3DUP";
  case ARMISD::VLD4DUP:       return "ARMISD::VLD4DUP";
  case ARMISD::VLD1_UPD:      return "ARMISD::VLD1_UPD";
  case ARMISD::VLD2_UPD:      return "ARMISD::VLD2_UPD";
  case ARMISD::VLD3_UPD:      return "ARMISD::VLD3_UPD";
  case ARMISD::VLD4_UPD:      return "ARMISD::VLD4_UPD";
  case ARMISD::VLD2LN_UPD:    return "ARMISD::VLD2LN_UPD";
  case ARMISD::VLD3LN_UPD:    return "ARMISD::VLD3LN_UPD";
  case ARMISD::VLD4LN_UPD:    return "ARMISD::VLD4LN_UPD";
  case ARMISD::VLD2DUP_UPD:   return "ARMISD::VLD2DUP_UPD";
  case ARMISD::VLD3DUP_UPD:   return "ARMISD::VLD3DUP_UPD";
  case ARMISD::VLD4DUP_UPD:   return "ARMISD::VLD4DUP_UPD";
  case ARMISD::VST1_UPD:      return "ARMISD::VST1_UPD";
  case ARMISD::VST2_UPD:      return "ARMISD::VST2_UPD";
  case ARMISD::VST3_UPD:      return "ARMISD::VST3_UPD";
  case ARMISD::VST4_UPD:      return "ARMISD::VST4_UPD";
  case ARMISD::VST2LN_UPD:    return "ARMISD::VST2LN_UPD";
  case ARMISD::VST3LN_UPD:    return "ARMISD::VST3LN_UPD";
  case ARMISD::VST4LN_UPD:    return "ARMISD::VST4LN_UPD";
  }
}

EVT ARMTargetLowering::getSetCCResultType(EVT VT) const {
  if (!VT.isVector()) return getPointerTy();
  return VT.changeVectorElementTypeToInteger();
}

/// getRegClassFor - Return the register class that should be used for the
/// specified value type.
const TargetRegisterClass *ARMTargetLowering::getRegClassFor(MVT VT) const {
  // Map v4i64 to QQ registers but do not make the type legal. Similarly map
  // v8i64 to QQQQ registers. v4i64 and v8i64 are only used for REG_SEQUENCE to
  // load / store 4 to 8 consecutive D registers.
  if (Subtarget->hasNEON()) {
    if (VT == MVT::v4i64)
      return &ARM::QQPRRegClass;
    if (VT == MVT::v8i64)
      return &ARM::QQQQPRRegClass;
  }
  return TargetLowering::getRegClassFor(VT);
}

// Create a fast isel object.
FastISel *
ARMTargetLowering::createFastISel(FunctionLoweringInfo &funcInfo,
                                  const TargetLibraryInfo *libInfo) const {
  return ARM::createFastISel(funcInfo, libInfo);
}

/// getMaximalGlobalOffset - Returns the maximal possible offset which can
/// be used for loads / stores from the global.
unsigned ARMTargetLowering::getMaximalGlobalOffset() const {
  return (Subtarget->isThumb1Only() ? 127 : 4095);
}

Sched::Preference ARMTargetLowering::getSchedulingPreference(SDNode *N) const {
  unsigned NumVals = N->getNumValues();
  if (!NumVals)
    return Sched::RegPressure;

  for (unsigned i = 0; i != NumVals; ++i) {
    EVT VT = N->getValueType(i);
    if (VT == MVT::Glue || VT == MVT::Other)
      continue;
    if (VT.isFloatingPoint() || VT.isVector())
      return Sched::ILP;
  }

  if (!N->isMachineOpcode())
    return Sched::RegPressure;

  // Load are scheduled for latency even if there instruction itinerary
  // is not available.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  const MCInstrDesc &MCID = TII->get(N->getMachineOpcode());

  if (MCID.getNumDefs() == 0)
    return Sched::RegPressure;
  if (!Itins->isEmpty() &&
      Itins->getOperandCycle(MCID.getSchedClass(), 0) > 2)
    return Sched::ILP;

  return Sched::RegPressure;
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

/// FPCCToARMCC - Convert a DAG fp condition code to an ARM CC.
static void FPCCToARMCC(ISD::CondCode CC, ARMCC::CondCodes &CondCode,
                        ARMCC::CondCodes &CondCode2) {
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
  case ISD::SETOLE: CondCode = ARMCC::LS; break;
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
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "ARMGenCallingConv.inc"

/// CCAssignFnForNode - Selects the correct CCAssignFn for a the
/// given CallingConvention value.
CCAssignFn *ARMTargetLowering::CCAssignFnForNode(CallingConv::ID CC,
                                                 bool Return,
                                                 bool isVarArg) const {
  switch (CC) {
  default:
    llvm_unreachable("Unsupported calling convention");
  case CallingConv::Fast:
    if (Subtarget->hasVFP2() && !isVarArg) {
      if (!Subtarget->isAAPCS_ABI())
        return (Return ? RetFastCC_ARM_APCS : FastCC_ARM_APCS);
      // For AAPCS ABI targets, just use VFP variant of the calling convention.
      return (Return ? RetCC_ARM_AAPCS_VFP : CC_ARM_AAPCS_VFP);
    }
    // Fallthrough
  case CallingConv::C: {
    // Use target triple & subtarget features to do actual dispatch.
    if (!Subtarget->isAAPCS_ABI())
      return (Return ? RetCC_ARM_APCS : CC_ARM_APCS);
    else if (Subtarget->hasVFP2() &&
             getTargetMachine().Options.FloatABIType == FloatABI::Hard &&
             !isVarArg)
      return (Return ? RetCC_ARM_AAPCS_VFP : CC_ARM_AAPCS_VFP);
    return (Return ? RetCC_ARM_AAPCS : CC_ARM_AAPCS);
  }
  case CallingConv::ARM_AAPCS_VFP:
    if (!isVarArg)
      return (Return ? RetCC_ARM_AAPCS_VFP : CC_ARM_AAPCS_VFP);
    // Fallthrough
  case CallingConv::ARM_AAPCS:
    return (Return ? RetCC_ARM_AAPCS : CC_ARM_AAPCS);
  case CallingConv::ARM_APCS:
    return (Return ? RetCC_ARM_APCS : CC_ARM_APCS);
  case CallingConv::GHC:
    return (Return ? RetCC_ARM_APCS : CC_ARM_APCS_GHC);
  }
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
SDValue
ARMTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                   CallingConv::ID CallConv, bool isVarArg,
                                   const SmallVectorImpl<ISD::InputArg> &Ins,
                                   DebugLoc dl, SelectionDAG &DAG,
                                   SmallVectorImpl<SDValue> &InVals,
                                   bool isThisReturn, SDValue ThisVal) const {

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  ARMCCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                    getTargetMachine(), RVLocs, *DAG.getContext(), Call);
  CCInfo.AnalyzeCallResult(Ins,
                           CCAssignFnForNode(CallConv, /* Return*/ true,
                                             isVarArg));

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];

    // Pass 'this' value directly from the argument to return value, to avoid
    // reg unit interference
    if (i == 0 && isThisReturn) {
      assert(!VA.needsCustom() && VA.getLocVT() == MVT::i32 &&
             "unexpected return calling convention register assignment");
      InVals.push_back(ThisVal);
      continue;
    }

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
      Val = DAG.getNode(ARMISD::VMOVDRR, dl, MVT::f64, Lo, Hi);

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
        Val = DAG.getNode(ARMISD::VMOVDRR, dl, MVT::f64, Lo, Hi);
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
      Val = DAG.getNode(ISD::BITCAST, dl, VA.getValVT(), Val);
      break;
    }

    InVals.push_back(Val);
  }

  return Chain;
}

/// LowerMemOpCallTo - Store the argument to the stack.
SDValue
ARMTargetLowering::LowerMemOpCallTo(SDValue Chain,
                                    SDValue StackPtr, SDValue Arg,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    const CCValAssign &VA,
                                    ISD::ArgFlagsTy Flags) const {
  unsigned LocMemOffset = VA.getLocMemOffset();
  SDValue PtrOff = DAG.getIntPtrConstant(LocMemOffset);
  PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, PtrOff);
  return DAG.getStore(Chain, dl, Arg, PtrOff,
                      MachinePointerInfo::getStack(LocMemOffset),
                      false, false, 0);
}

void ARMTargetLowering::PassF64ArgInRegs(DebugLoc dl, SelectionDAG &DAG,
                                         SDValue Chain, SDValue &Arg,
                                         RegsToPassVector &RegsToPass,
                                         CCValAssign &VA, CCValAssign &NextVA,
                                         SDValue &StackPtr,
                                         SmallVector<SDValue, 8> &MemOpChains,
                                         ISD::ArgFlagsTy Flags) const {

  SDValue fmrrd = DAG.getNode(ARMISD::VMOVRRD, dl,
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
ARMTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
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
  bool doesNotRet                       = CLI.DoesNotReturn;
  bool isVarArg                         = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  bool isStructRet    = (Outs.empty()) ? false : Outs[0].Flags.isSRet();
  bool isThisReturn   = false;
  bool isSibCall      = false;
  // Disable tail calls if they're not supported.
  if (!EnableARMTailCalls && !Subtarget->supportsTailCall())
    isTailCall = false;
  if (isTailCall) {
    // Check if it's really possible to do a tail call.
    isTailCall = IsEligibleForTailCallOptimization(Callee, CallConv,
                    isVarArg, isStructRet, MF.getFunction()->hasStructRetAttr(),
                                                   Outs, OutVals, Ins, DAG);
    // We don't support GuaranteedTailCallOpt for ARM, only automatically
    // detected sibcalls.
    if (isTailCall) {
      ++NumTailCalls;
      isSibCall = true;
    }
  }

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  ARMCCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext(), Call);
  CCInfo.AnalyzeCallOperands(Outs,
                             CCAssignFnForNode(CallConv, /* Return*/ false,
                                               isVarArg));

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  // For tail calls, memory operands are available in our caller's stack.
  if (isSibCall)
    NumBytes = 0;

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  if (!isSibCall)
    Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));

  SDValue StackPtr = DAG.getCopyFromReg(Chain, dl, ARM::SP, getPointerTy());

  RegsToPassVector RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // Walk the register/memloc assignments, inserting copies/loads.  In the case
  // of tail call optimization, arguments are handled later.
  for (unsigned i = 0, realArgIdx = 0, e = ArgLocs.size();
       i != e;
       ++i, ++realArgIdx) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[realArgIdx];
    ISD::ArgFlagsTy Flags = Outs[realArgIdx].Flags;
    bool isByVal = Flags.isByVal();

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
      Arg = DAG.getNode(ISD::BITCAST, dl, VA.getLocVT(), Arg);
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

          MemOpChains.push_back(LowerMemOpCallTo(Chain, StackPtr, Op1,
                                                 dl, DAG, VA, Flags));
        }
      } else {
        PassF64ArgInRegs(dl, DAG, Chain, Arg, RegsToPass, VA, ArgLocs[++i],
                         StackPtr, MemOpChains, Flags);
      }
    } else if (VA.isRegLoc()) {
      if (realArgIdx == 0 && Flags.isReturned() && Outs[0].VT == MVT::i32) {
        assert(VA.getLocVT() == MVT::i32 &&
               "unexpected calling convention register assignment");
        assert(!Ins.empty() && Ins[0].VT == MVT::i32 &&
               "unexpected use of 'returned'");
        isThisReturn = true;
      }
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else if (isByVal) {
      assert(VA.isMemLoc());
      unsigned offset = 0;

      // True if this byval aggregate will be split between registers
      // and memory.
      unsigned ByValArgsCount = CCInfo.getInRegsParamsCount();
      unsigned CurByValIdx = CCInfo.getInRegsParamsProceed();

      if (CurByValIdx < ByValArgsCount) {

        unsigned RegBegin, RegEnd;
        CCInfo.getInRegsParamInfo(CurByValIdx, RegBegin, RegEnd);

        EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
        unsigned int i, j;
        for (i = 0, j = RegBegin; j < RegEnd; i++, j++) {
          SDValue Const = DAG.getConstant(4*i, MVT::i32);
          SDValue AddArg = DAG.getNode(ISD::ADD, dl, PtrVT, Arg, Const);
          SDValue Load = DAG.getLoad(PtrVT, dl, Chain, AddArg,
                                     MachinePointerInfo(),
                                     false, false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(j, Load));
        }

        // If parameter size outsides register area, "offset" value
        // helps us to calculate stack slot for remained part properly.
        offset = RegEnd - RegBegin;

        CCInfo.nextInRegsParam();
      }

      if (Flags.getByValSize() > 4*offset) {
        unsigned LocMemOffset = VA.getLocMemOffset();
        SDValue StkPtrOff = DAG.getIntPtrConstant(LocMemOffset);
        SDValue Dst = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr,
                                  StkPtrOff);
        SDValue SrcOffset = DAG.getIntPtrConstant(4*offset);
        SDValue Src = DAG.getNode(ISD::ADD, dl, getPointerTy(), Arg, SrcOffset);
        SDValue SizeNode = DAG.getConstant(Flags.getByValSize() - 4*offset,
                                           MVT::i32);
        SDValue AlignNode = DAG.getConstant(Flags.getByValAlign(), MVT::i32);

        SDVTList VTs = DAG.getVTList(MVT::Other, MVT::Glue);
        SDValue Ops[] = { Chain, Dst, Src, SizeNode, AlignNode};
        MemOpChains.push_back(DAG.getNode(ARMISD::COPY_STRUCT_BYVAL, dl, VTs,
                                          Ops, array_lengthof(Ops)));
      }
    } else if (!isSibCall) {
      assert(VA.isMemLoc());

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
  // Tail call byval lowering might overwrite argument registers so in case of
  // tail call optimization the copies to registers are lowered later.
  if (!isTailCall)
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
      Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                               RegsToPass[i].second, InFlag);
      InFlag = Chain.getValue(1);
    }

  // For tail calls lower the arguments to the 'real' stack slot.
  if (isTailCall) {
    // Force all the incoming stack arguments to be loaded from the stack
    // before any new outgoing arguments are stored to the stack, because the
    // outgoing stack slots may alias the incoming argument stack slots, and
    // the alias isn't otherwise explicit. This is slightly more conservative
    // than necessary, because it means that each store effectively depends
    // on every argument instead of just those arguments it would clobber.

    // Do not flag preceding copytoreg stuff together with the following stuff.
    InFlag = SDValue();
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
      Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                               RegsToPass[i].second, InFlag);
      InFlag = Chain.getValue(1);
    }
    InFlag = SDValue();
  }

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  bool isDirect = false;
  bool isARMFunc = false;
  bool isLocalARMFunc = false;
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  if (EnableARMLongCalls) {
    assert (getTargetMachine().getRelocationModel() == Reloc::Static
            && "long-calls with non-static relocation model!");
    // Handle a global address or an external symbol. If it's not one of
    // those, the target's already in a register, so we don't need to do
    // anything extra.
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
      const GlobalValue *GV = G->getGlobal();
      // Create a constant pool entry for the callee address
      unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
      ARMConstantPoolValue *CPV =
        ARMConstantPoolConstant::Create(GV, ARMPCLabelIndex, ARMCP::CPValue, 0);

      // Get the address of the callee into a register
      SDValue CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 4);
      CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), dl,
                           DAG.getEntryNode(), CPAddr,
                           MachinePointerInfo::getConstantPool(),
                           false, false, false, 0);
    } else if (ExternalSymbolSDNode *S=dyn_cast<ExternalSymbolSDNode>(Callee)) {
      const char *Sym = S->getSymbol();

      // Create a constant pool entry for the callee address
      unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
      ARMConstantPoolValue *CPV =
        ARMConstantPoolSymbol::Create(*DAG.getContext(), Sym,
                                      ARMPCLabelIndex, 0);
      // Get the address of the callee into a register
      SDValue CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 4);
      CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), dl,
                           DAG.getEntryNode(), CPAddr,
                           MachinePointerInfo::getConstantPool(),
                           false, false, false, 0);
    }
  } else if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    isDirect = true;
    bool isExt = GV->isDeclaration() || GV->isWeakForLinker();
    bool isStub = (isExt && Subtarget->isTargetDarwin()) &&
                   getTargetMachine().getRelocationModel() != Reloc::Static;
    isARMFunc = !Subtarget->isThumb() || isStub;
    // ARM call to a local ARM function is predicable.
    isLocalARMFunc = !Subtarget->isThumb() && (!isExt || !ARMInterworking);
    // tBX takes a register source operand.
    if (isARMFunc && Subtarget->isThumb1Only() && !Subtarget->hasV5TOps()) {
      unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
      ARMConstantPoolValue *CPV =
        ARMConstantPoolConstant::Create(GV, ARMPCLabelIndex, ARMCP::CPValue, 4);
      SDValue CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 4);
      CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), dl,
                           DAG.getEntryNode(), CPAddr,
                           MachinePointerInfo::getConstantPool(),
                           false, false, false, 0);
      SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex, MVT::i32);
      Callee = DAG.getNode(ARMISD::PIC_ADD, dl,
                           getPointerTy(), Callee, PICLabel);
    } else {
      // On ELF targets for PIC code, direct calls should go through the PLT
      unsigned OpFlags = 0;
      if (Subtarget->isTargetELF() &&
          getTargetMachine().getRelocationModel() == Reloc::PIC_)
        OpFlags = ARMII::MO_PLT;
      Callee = DAG.getTargetGlobalAddress(GV, dl, getPointerTy(), 0, OpFlags);
    }
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    isDirect = true;
    bool isStub = Subtarget->isTargetDarwin() &&
                  getTargetMachine().getRelocationModel() != Reloc::Static;
    isARMFunc = !Subtarget->isThumb() || isStub;
    // tBX takes a register source operand.
    const char *Sym = S->getSymbol();
    if (isARMFunc && Subtarget->isThumb1Only() && !Subtarget->hasV5TOps()) {
      unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
      ARMConstantPoolValue *CPV =
        ARMConstantPoolSymbol::Create(*DAG.getContext(), Sym,
                                      ARMPCLabelIndex, 4);
      SDValue CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 4);
      CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), dl,
                           DAG.getEntryNode(), CPAddr,
                           MachinePointerInfo::getConstantPool(),
                           false, false, false, 0);
      SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex, MVT::i32);
      Callee = DAG.getNode(ARMISD::PIC_ADD, dl,
                           getPointerTy(), Callee, PICLabel);
    } else {
      unsigned OpFlags = 0;
      // On ELF targets for PIC code, direct calls should go through the PLT
      if (Subtarget->isTargetELF() &&
                  getTargetMachine().getRelocationModel() == Reloc::PIC_)
        OpFlags = ARMII::MO_PLT;
      Callee = DAG.getTargetExternalSymbol(Sym, getPointerTy(), OpFlags);
    }
  }

  // FIXME: handle tail calls differently.
  unsigned CallOpc;
  bool HasMinSizeAttr = MF.getFunction()->getAttributes().
    hasAttribute(AttributeSet::FunctionIndex, Attribute::MinSize);
  if (Subtarget->isThumb()) {
    if ((!isDirect || isARMFunc) && !Subtarget->hasV5TOps())
      CallOpc = ARMISD::CALL_NOLINK;
    else
      CallOpc = isARMFunc ? ARMISD::CALL : ARMISD::tCALL;
  } else {
    if (!isDirect && !Subtarget->hasV5TOps())
      CallOpc = ARMISD::CALL_NOLINK;
    else if (doesNotRet && isDirect && Subtarget->hasRAS() &&
               // Emit regular call when code size is the priority
               !HasMinSizeAttr)
      // "mov lr, pc; b _foo" to avoid confusing the RSP
      CallOpc = ARMISD::CALL_NOLINK;
    else
      CallOpc = isLocalARMFunc ? ARMISD::CALL_PRED : ARMISD::CALL;
  }

  std::vector<SDValue> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  // Add a register mask operand representing the call-preserved registers.
  const uint32_t *Mask;
  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();
  const ARMBaseRegisterInfo *ARI = static_cast<const ARMBaseRegisterInfo*>(TRI);
  if (isThisReturn)
    // For 'this' returns, use the R0-preserving mask
    Mask = ARI->getThisReturnPreservedMask(CallConv);
  else
    Mask = ARI->getCallPreservedMask(CallConv);

  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  if (isTailCall)
    return DAG.getNode(ARMISD::TC_RETURN, dl, NodeTys, &Ops[0], Ops.size());

  // Returns a chain and a flag for retval copy to use.
  Chain = DAG.getNode(CallOpc, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg, Ins, dl, DAG,
                         InVals, isThisReturn,
                         isThisReturn ? OutVals[0] : SDValue());
}

/// HandleByVal - Every parameter *after* a byval parameter is passed
/// on the stack.  Remember the next parameter register to allocate,
/// and then confiscate the rest of the parameter registers to insure
/// this.
void
ARMTargetLowering::HandleByVal(
    CCState *State, unsigned &size, unsigned Align) const {
  unsigned reg = State->AllocateReg(GPRArgRegs, 4);
  assert((State->getCallOrPrologue() == Prologue ||
          State->getCallOrPrologue() == Call) &&
         "unhandled ParmContext");

  // For in-prologue parameters handling, we also introduce stack offset
  // for byval registers: see CallingConvLower.cpp, CCState::HandleByVal.
  // This behaviour outsides AAPCS rules (5.5 Parameters Passing) of how
  // NSAA should be evaluted (NSAA means "next stacked argument address").
  // So: NextStackOffset = NSAAOffset + SizeOfByValParamsStoredInRegs.
  // Then: NSAAOffset = NextStackOffset - SizeOfByValParamsStoredInRegs.
  unsigned NSAAOffset = State->getNextStackOffset();
  if (State->getCallOrPrologue() != Call) {
    for (unsigned i = 0, e = State->getInRegsParamsCount(); i != e; ++i) {
      unsigned RB, RE;
      State->getInRegsParamInfo(i, RB, RE);
      assert(NSAAOffset >= (RE-RB)*4 &&
             "Stack offset for byval regs doesn't introduced anymore?");
      NSAAOffset -= (RE-RB)*4;
    }
  }
  if ((ARM::R0 <= reg) && (reg <= ARM::R3)) {
    if (Subtarget->isAAPCS_ABI() && Align > 4) {
      unsigned AlignInRegs = Align / 4;
      unsigned Waste = (ARM::R4 - reg) % AlignInRegs;
      for (unsigned i = 0; i < Waste; ++i)
        reg = State->AllocateReg(GPRArgRegs, 4);
    }
    if (reg != 0) {
      unsigned excess = 4 * (ARM::R4 - reg);

      // Special case when NSAA != SP and parameter size greater than size of
      // all remained GPR regs. In that case we can't split parameter, we must
      // send it to stack. We also must set NCRN to R4, so waste all
      // remained registers.
      if (Subtarget->isAAPCS_ABI() && NSAAOffset != 0 && size > excess) {
        while (State->AllocateReg(GPRArgRegs, 4))
          ;
        return;
      }

      // First register for byval parameter is the first register that wasn't
      // allocated before this method call, so it would be "reg".
      // If parameter is small enough to be saved in range [reg, r4), then
      // the end (first after last) register would be reg + param-size-in-regs,
      // else parameter would be splitted between registers and stack,
      // end register would be r4 in this case.
      unsigned ByValRegBegin = reg;
      unsigned ByValRegEnd = (size < excess) ? reg + size/4 : ARM::R4;
      State->addInRegsParamInfo(ByValRegBegin, ByValRegEnd);
      // Note, first register is allocated in the beginning of function already,
      // allocate remained amount of registers we need.
      for (unsigned i = reg+1; i != ByValRegEnd; ++i)
        State->AllocateReg(GPRArgRegs, 4);
      // At a call site, a byval parameter that is split between
      // registers and memory needs its size truncated here.  In a
      // function prologue, such byval parameters are reassembled in
      // memory, and are not truncated.
      if (State->getCallOrPrologue() == Call) {
        // Make remained size equal to 0 in case, when
        // the whole structure may be stored into registers.
        if (size < excess)
          size = 0;
        else
          size -= excess;
      }
    }
  }
}

/// MatchingStackOffset - Return true if the given stack call argument is
/// already available in the same position (relatively) of the caller's
/// incoming argument stack.
static
bool MatchingStackOffset(SDValue Arg, unsigned Offset, ISD::ArgFlagsTy Flags,
                         MachineFrameInfo *MFI, const MachineRegisterInfo *MRI,
                         const TargetInstrInfo *TII) {
  unsigned Bytes = Arg.getValueType().getSizeInBits() / 8;
  int FI = INT_MAX;
  if (Arg.getOpcode() == ISD::CopyFromReg) {
    unsigned VR = cast<RegisterSDNode>(Arg.getOperand(1))->getReg();
    if (!TargetRegisterInfo::isVirtualRegister(VR))
      return false;
    MachineInstr *Def = MRI->getVRegDef(VR);
    if (!Def)
      return false;
    if (!Flags.isByVal()) {
      if (!TII->isLoadFromStackSlot(Def, FI))
        return false;
    } else {
      return false;
    }
  } else if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(Arg)) {
    if (Flags.isByVal())
      // ByVal argument is passed in as a pointer but it's now being
      // dereferenced. e.g.
      // define @foo(%struct.X* %A) {
      //   tail call @bar(%struct.X* byval %A)
      // }
      return false;
    SDValue Ptr = Ld->getBasePtr();
    FrameIndexSDNode *FINode = dyn_cast<FrameIndexSDNode>(Ptr);
    if (!FINode)
      return false;
    FI = FINode->getIndex();
  } else
    return false;

  assert(FI != INT_MAX);
  if (!MFI->isFixedObjectIndex(FI))
    return false;
  return Offset == MFI->getObjectOffset(FI) && Bytes == MFI->getObjectSize(FI);
}

/// IsEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization. Targets which want to do tail call
/// optimization should implement this function.
bool
ARMTargetLowering::IsEligibleForTailCallOptimization(SDValue Callee,
                                                     CallingConv::ID CalleeCC,
                                                     bool isVarArg,
                                                     bool isCalleeStructRet,
                                                     bool isCallerStructRet,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                                     SelectionDAG& DAG) const {
  const Function *CallerF = DAG.getMachineFunction().getFunction();
  CallingConv::ID CallerCC = CallerF->getCallingConv();
  bool CCMatch = CallerCC == CalleeCC;

  // Look for obvious safe cases to perform tail call optimization that do not
  // require ABI changes. This is what gcc calls sibcall.

  // Do not sibcall optimize vararg calls unless the call site is not passing
  // any arguments.
  if (isVarArg && !Outs.empty())
    return false;

  // Also avoid sibcall optimization if either caller or callee uses struct
  // return semantics.
  if (isCalleeStructRet || isCallerStructRet)
    return false;

  // FIXME: Completely disable sibcall for Thumb1 since Thumb1RegisterInfo::
  // emitEpilogue is not ready for them. Thumb tail calls also use t2B, as
  // the Thumb1 16-bit unconditional branch doesn't have sufficient relocation
  // support in the assembler and linker to be used. This would need to be
  // fixed to fully support tail calls in Thumb1.
  //
  // Doing this is tricky, since the LDM/POP instruction on Thumb doesn't take
  // LR.  This means if we need to reload LR, it takes an extra instructions,
  // which outweighs the value of the tail call; but here we don't know yet
  // whether LR is going to be used.  Probably the right approach is to
  // generate the tail call here and turn it back into CALL/RET in
  // emitEpilogue if LR is used.

  // Thumb1 PIC calls to external symbols use BX, so they can be tail calls,
  // but we need to make sure there are enough registers; the only valid
  // registers are the 4 used for parameters.  We don't currently do this
  // case.
  if (Subtarget->isThumb1Only())
    return false;

  // If the calling conventions do not match, then we'd better make sure the
  // results are returned in the same way as what the caller expects.
  if (!CCMatch) {
    SmallVector<CCValAssign, 16> RVLocs1;
    ARMCCState CCInfo1(CalleeCC, false, DAG.getMachineFunction(),
                       getTargetMachine(), RVLocs1, *DAG.getContext(), Call);
    CCInfo1.AnalyzeCallResult(Ins, CCAssignFnForNode(CalleeCC, true, isVarArg));

    SmallVector<CCValAssign, 16> RVLocs2;
    ARMCCState CCInfo2(CallerCC, false, DAG.getMachineFunction(),
                       getTargetMachine(), RVLocs2, *DAG.getContext(), Call);
    CCInfo2.AnalyzeCallResult(Ins, CCAssignFnForNode(CallerCC, true, isVarArg));

    if (RVLocs1.size() != RVLocs2.size())
      return false;
    for (unsigned i = 0, e = RVLocs1.size(); i != e; ++i) {
      if (RVLocs1[i].isRegLoc() != RVLocs2[i].isRegLoc())
        return false;
      if (RVLocs1[i].getLocInfo() != RVLocs2[i].getLocInfo())
        return false;
      if (RVLocs1[i].isRegLoc()) {
        if (RVLocs1[i].getLocReg() != RVLocs2[i].getLocReg())
          return false;
      } else {
        if (RVLocs1[i].getLocMemOffset() != RVLocs2[i].getLocMemOffset())
          return false;
      }
    }
  }

  // If Caller's vararg or byval argument has been split between registers and
  // stack, do not perform tail call, since part of the argument is in caller's
  // local frame.
  const ARMFunctionInfo *AFI_Caller = DAG.getMachineFunction().
                                      getInfo<ARMFunctionInfo>();
  if (AFI_Caller->getArgRegsSaveSize())
    return false;

  // If the callee takes no arguments then go on to check the results of the
  // call.
  if (!Outs.empty()) {
    // Check if stack adjustment is needed. For now, do not do this if any
    // argument is passed on the stack.
    SmallVector<CCValAssign, 16> ArgLocs;
    ARMCCState CCInfo(CalleeCC, isVarArg, DAG.getMachineFunction(),
                      getTargetMachine(), ArgLocs, *DAG.getContext(), Call);
    CCInfo.AnalyzeCallOperands(Outs,
                               CCAssignFnForNode(CalleeCC, false, isVarArg));
    if (CCInfo.getNextStackOffset()) {
      MachineFunction &MF = DAG.getMachineFunction();

      // Check if the arguments are already laid out in the right way as
      // the caller's fixed stack objects.
      MachineFrameInfo *MFI = MF.getFrameInfo();
      const MachineRegisterInfo *MRI = &MF.getRegInfo();
      const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
      for (unsigned i = 0, realArgIdx = 0, e = ArgLocs.size();
           i != e;
           ++i, ++realArgIdx) {
        CCValAssign &VA = ArgLocs[i];
        EVT RegVT = VA.getLocVT();
        SDValue Arg = OutVals[realArgIdx];
        ISD::ArgFlagsTy Flags = Outs[realArgIdx].Flags;
        if (VA.getLocInfo() == CCValAssign::Indirect)
          return false;
        if (VA.needsCustom()) {
          // f64 and vector types are split into multiple registers or
          // register/stack-slot combinations.  The types will not match
          // the registers; give up on memory f64 refs until we figure
          // out what to do about this.
          if (!VA.isRegLoc())
            return false;
          if (!ArgLocs[++i].isRegLoc())
            return false;
          if (RegVT == MVT::v2f64) {
            if (!ArgLocs[++i].isRegLoc())
              return false;
            if (!ArgLocs[++i].isRegLoc())
              return false;
          }
        } else if (!VA.isRegLoc()) {
          if (!MatchingStackOffset(Arg, VA.getLocMemOffset(), Flags,
                                   MFI, MRI, TII))
            return false;
        }
      }
    }
  }

  return true;
}

bool
ARMTargetLowering::CanLowerReturn(CallingConv::ID CallConv,
                                  MachineFunction &MF, bool isVarArg,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(), RVLocs, Context);
  return CCInfo.CheckReturn(Outs, CCAssignFnForNode(CallConv, /*Return=*/true,
                                                    isVarArg));
}

SDValue
ARMTargetLowering::LowerReturn(SDValue Chain,
                               CallingConv::ID CallConv, bool isVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               DebugLoc dl, SelectionDAG &DAG) const {

  // CCValAssign - represent the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slots.
  ARMCCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                    getTargetMachine(), RVLocs, *DAG.getContext(), Call);

  // Analyze outgoing return values.
  CCInfo.AnalyzeReturn(Outs, CCAssignFnForNode(CallConv, /* Return */ true,
                                               isVarArg));

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)

  // Copy the result values into the output registers.
  for (unsigned i = 0, realRVLocIdx = 0;
       i != RVLocs.size();
       ++i, ++realRVLocIdx) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    SDValue Arg = OutVals[realRVLocIdx];

    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, dl, VA.getLocVT(), Arg);
      break;
    }

    if (VA.needsCustom()) {
      if (VA.getLocVT() == MVT::v2f64) {
        // Extract the first half and return it in two registers.
        SDValue Half = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64, Arg,
                                   DAG.getConstant(0, MVT::i32));
        SDValue HalfGPRs = DAG.getNode(ARMISD::VMOVRRD, dl,
                                       DAG.getVTList(MVT::i32, MVT::i32), Half);

        Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), HalfGPRs, Flag);
        Flag = Chain.getValue(1);
        RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
        VA = RVLocs[++i]; // skip ahead to next loc
        Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(),
                                 HalfGPRs.getValue(1), Flag);
        Flag = Chain.getValue(1);
        RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
        VA = RVLocs[++i]; // skip ahead to next loc

        // Extract the 2nd half and fall through to handle it as an f64 value.
        Arg = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64, Arg,
                          DAG.getConstant(1, MVT::i32));
      }
      // Legalize ret f64 -> ret 2 x i32.  We always have fmrrd if f64 is
      // available.
      SDValue fmrrd = DAG.getNode(ARMISD::VMOVRRD, dl,
                                  DAG.getVTList(MVT::i32, MVT::i32), &Arg, 1);
      Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), fmrrd, Flag);
      Flag = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
      VA = RVLocs[++i]; // skip ahead to next loc
      Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), fmrrd.getValue(1),
                               Flag);
    } else
      Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), Arg, Flag);

    // Guarantee that all emitted copies are
    // stuck together, avoiding something bad.
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(ARMISD::RET_FLAG, dl, MVT::Other,
                     RetOps.data(), RetOps.size());
}

bool ARMTargetLowering::isUsedByReturnOnly(SDNode *N, SDValue &Chain) const {
  if (N->getNumValues() != 1)
    return false;
  if (!N->hasNUsesOfValue(1, 0))
    return false;

  SDValue TCChain = Chain;
  SDNode *Copy = *N->use_begin();
  if (Copy->getOpcode() == ISD::CopyToReg) {
    // If the copy has a glue operand, we conservatively assume it isn't safe to
    // perform a tail call.
    if (Copy->getOperand(Copy->getNumOperands()-1).getValueType() == MVT::Glue)
      return false;
    TCChain = Copy->getOperand(0);
  } else if (Copy->getOpcode() == ARMISD::VMOVRRD) {
    SDNode *VMov = Copy;
    // f64 returned in a pair of GPRs.
    SmallPtrSet<SDNode*, 2> Copies;
    for (SDNode::use_iterator UI = VMov->use_begin(), UE = VMov->use_end();
         UI != UE; ++UI) {
      if (UI->getOpcode() != ISD::CopyToReg)
        return false;
      Copies.insert(*UI);
    }
    if (Copies.size() > 2)
      return false;

    for (SDNode::use_iterator UI = VMov->use_begin(), UE = VMov->use_end();
         UI != UE; ++UI) {
      SDValue UseChain = UI->getOperand(0);
      if (Copies.count(UseChain.getNode()))
        // Second CopyToReg
        Copy = *UI;
      else
        // First CopyToReg
        TCChain = UseChain;
    }
  } else if (Copy->getOpcode() == ISD::BITCAST) {
    // f32 returned in a single GPR.
    if (!Copy->hasOneUse())
      return false;
    Copy = *Copy->use_begin();
    if (Copy->getOpcode() != ISD::CopyToReg || !Copy->hasNUsesOfValue(1, 0))
      return false;
    Chain = Copy->getOperand(0);
  } else {
    return false;
  }

  bool HasRet = false;
  for (SDNode::use_iterator UI = Copy->use_begin(), UE = Copy->use_end();
       UI != UE; ++UI) {
    if (UI->getOpcode() != ARMISD::RET_FLAG)
      return false;
    HasRet = true;
  }

  if (!HasRet)
    return false;

  Chain = TCChain;
  return true;
}

bool ARMTargetLowering::mayBeEmittedAsTailCall(CallInst *CI) const {
  if (!EnableARMTailCalls && !Subtarget->supportsTailCall())
    return false;

  if (!CI->isTailCall())
    return false;

  return !Subtarget->isThumb1Only();
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

unsigned ARMTargetLowering::getJumpTableEncoding() const {
  return MachineJumpTableInfo::EK_Inline;
}

SDValue ARMTargetLowering::LowerBlockAddress(SDValue Op,
                                             SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned ARMPCLabelIndex = 0;
  DebugLoc DL = Op.getDebugLoc();
  EVT PtrVT = getPointerTy();
  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();
  SDValue CPAddr;
  if (RelocM == Reloc::Static) {
    CPAddr = DAG.getTargetConstantPool(BA, PtrVT, 4);
  } else {
    unsigned PCAdj = Subtarget->isThumb() ? 4 : 8;
    ARMPCLabelIndex = AFI->createPICLabelUId();
    ARMConstantPoolValue *CPV =
      ARMConstantPoolConstant::Create(BA, ARMPCLabelIndex,
                                      ARMCP::CPBlockAddress, PCAdj);
    CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
  }
  CPAddr = DAG.getNode(ARMISD::Wrapper, DL, PtrVT, CPAddr);
  SDValue Result = DAG.getLoad(PtrVT, DL, DAG.getEntryNode(), CPAddr,
                               MachinePointerInfo::getConstantPool(),
                               false, false, false, 0);
  if (RelocM == Reloc::Static)
    return Result;
  SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex, MVT::i32);
  return DAG.getNode(ARMISD::PIC_ADD, DL, PtrVT, Result, PICLabel);
}

// Lower ISD::GlobalTLSAddress using the "general dynamic" model
SDValue
ARMTargetLowering::LowerToTLSGeneralDynamicModel(GlobalAddressSDNode *GA,
                                                 SelectionDAG &DAG) const {
  DebugLoc dl = GA->getDebugLoc();
  EVT PtrVT = getPointerTy();
  unsigned char PCAdj = Subtarget->isThumb() ? 4 : 8;
  MachineFunction &MF = DAG.getMachineFunction();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
  ARMConstantPoolValue *CPV =
    ARMConstantPoolConstant::Create(GA->getGlobal(), ARMPCLabelIndex,
                                    ARMCP::CPValue, PCAdj, ARMCP::TLSGD, true);
  SDValue Argument = DAG.getTargetConstantPool(CPV, PtrVT, 4);
  Argument = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, Argument);
  Argument = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), Argument,
                         MachinePointerInfo::getConstantPool(),
                         false, false, false, 0);
  SDValue Chain = Argument.getValue(1);

  SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex, MVT::i32);
  Argument = DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Argument, PICLabel);

  // call __tls_get_addr.
  ArgListTy Args;
  ArgListEntry Entry;
  Entry.Node = Argument;
  Entry.Ty = (Type *) Type::getInt32Ty(*DAG.getContext());
  Args.push_back(Entry);
  // FIXME: is there useful debug info available here?
  TargetLowering::CallLoweringInfo CLI(Chain,
                (Type *) Type::getInt32Ty(*DAG.getContext()),
                false, false, false, false,
                0, CallingConv::C, /*isTailCall=*/false,
                /*doesNotRet=*/false, /*isReturnValueUsed=*/true,
                DAG.getExternalSymbol("__tls_get_addr", PtrVT), Args, DAG, dl);
  std::pair<SDValue, SDValue> CallResult = LowerCallTo(CLI);
  return CallResult.first;
}

// Lower ISD::GlobalTLSAddress using the "initial exec" or
// "local exec" model.
SDValue
ARMTargetLowering::LowerToTLSExecModels(GlobalAddressSDNode *GA,
                                        SelectionDAG &DAG,
                                        TLSModel::Model model) const {
  const GlobalValue *GV = GA->getGlobal();
  DebugLoc dl = GA->getDebugLoc();
  SDValue Offset;
  SDValue Chain = DAG.getEntryNode();
  EVT PtrVT = getPointerTy();
  // Get the Thread Pointer
  SDValue ThreadPointer = DAG.getNode(ARMISD::THREAD_POINTER, dl, PtrVT);

  if (model == TLSModel::InitialExec) {
    MachineFunction &MF = DAG.getMachineFunction();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
    // Initial exec model.
    unsigned char PCAdj = Subtarget->isThumb() ? 4 : 8;
    ARMConstantPoolValue *CPV =
      ARMConstantPoolConstant::Create(GA->getGlobal(), ARMPCLabelIndex,
                                      ARMCP::CPValue, PCAdj, ARMCP::GOTTPOFF,
                                      true);
    Offset = DAG.getTargetConstantPool(CPV, PtrVT, 4);
    Offset = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, Offset);
    Offset = DAG.getLoad(PtrVT, dl, Chain, Offset,
                         MachinePointerInfo::getConstantPool(),
                         false, false, false, 0);
    Chain = Offset.getValue(1);

    SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex, MVT::i32);
    Offset = DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Offset, PICLabel);

    Offset = DAG.getLoad(PtrVT, dl, Chain, Offset,
                         MachinePointerInfo::getConstantPool(),
                         false, false, false, 0);
  } else {
    // local exec model
    assert(model == TLSModel::LocalExec);
    ARMConstantPoolValue *CPV =
      ARMConstantPoolConstant::Create(GV, ARMCP::TPOFF);
    Offset = DAG.getTargetConstantPool(CPV, PtrVT, 4);
    Offset = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, Offset);
    Offset = DAG.getLoad(PtrVT, dl, Chain, Offset,
                         MachinePointerInfo::getConstantPool(),
                         false, false, false, 0);
  }

  // The address of the thread local variable is the add of the thread
  // pointer with the offset of the variable.
  return DAG.getNode(ISD::ADD, dl, PtrVT, ThreadPointer, Offset);
}

SDValue
ARMTargetLowering::LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const {
  // TODO: implement the "local dynamic" model
  assert(Subtarget->isTargetELF() &&
         "TLS not implemented for non-ELF targets");
  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);

  TLSModel::Model model = getTargetMachine().getTLSModel(GA->getGlobal());

  switch (model) {
    case TLSModel::GeneralDynamic:
    case TLSModel::LocalDynamic:
      return LowerToTLSGeneralDynamicModel(GA, DAG);
    case TLSModel::InitialExec:
    case TLSModel::LocalExec:
      return LowerToTLSExecModels(GA, DAG, model);
  }
  llvm_unreachable("bogus TLS model");
}

SDValue ARMTargetLowering::LowerGlobalAddressELF(SDValue Op,
                                                 SelectionDAG &DAG) const {
  EVT PtrVT = getPointerTy();
  DebugLoc dl = Op.getDebugLoc();
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_) {
    bool UseGOTOFF = GV->hasLocalLinkage() || GV->hasHiddenVisibility();
    ARMConstantPoolValue *CPV =
      ARMConstantPoolConstant::Create(GV,
                                      UseGOTOFF ? ARMCP::GOTOFF : ARMCP::GOT);
    SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
    CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
    SDValue Result = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(),
                                 CPAddr,
                                 MachinePointerInfo::getConstantPool(),
                                 false, false, false, 0);
    SDValue Chain = Result.getValue(1);
    SDValue GOT = DAG.getGLOBAL_OFFSET_TABLE(PtrVT);
    Result = DAG.getNode(ISD::ADD, dl, PtrVT, Result, GOT);
    if (!UseGOTOFF)
      Result = DAG.getLoad(PtrVT, dl, Chain, Result,
                           MachinePointerInfo::getGOT(),
                           false, false, false, 0);
    return Result;
  }

  // If we have T2 ops, we can materialize the address directly via movt/movw
  // pair. This is always cheaper.
  if (Subtarget->useMovt()) {
    ++NumMovwMovt;
    // FIXME: Once remat is capable of dealing with instructions with register
    // operands, expand this into two nodes.
    return DAG.getNode(ARMISD::Wrapper, dl, PtrVT,
                       DAG.getTargetGlobalAddress(GV, dl, PtrVT));
  } else {
    SDValue CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 4);
    CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
    return DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), CPAddr,
                       MachinePointerInfo::getConstantPool(),
                       false, false, false, 0);
  }
}

SDValue ARMTargetLowering::LowerGlobalAddressDarwin(SDValue Op,
                                                    SelectionDAG &DAG) const {
  EVT PtrVT = getPointerTy();
  DebugLoc dl = Op.getDebugLoc();
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();

  // FIXME: Enable this for static codegen when tool issues are fixed.  Also
  // update ARMFastISel::ARMMaterializeGV.
  if (Subtarget->useMovt() && RelocM != Reloc::Static) {
    ++NumMovwMovt;
    // FIXME: Once remat is capable of dealing with instructions with register
    // operands, expand this into two nodes.
    if (RelocM == Reloc::Static)
      return DAG.getNode(ARMISD::Wrapper, dl, PtrVT,
                                 DAG.getTargetGlobalAddress(GV, dl, PtrVT));

    unsigned Wrapper = (RelocM == Reloc::PIC_)
      ? ARMISD::WrapperPIC : ARMISD::WrapperDYN;
    SDValue Result = DAG.getNode(Wrapper, dl, PtrVT,
                                 DAG.getTargetGlobalAddress(GV, dl, PtrVT));
    if (Subtarget->GVIsIndirectSymbol(GV, RelocM))
      Result = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), Result,
                           MachinePointerInfo::getGOT(),
                           false, false, false, 0);
    return Result;
  }

  unsigned ARMPCLabelIndex = 0;
  SDValue CPAddr;
  if (RelocM == Reloc::Static) {
    CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 4);
  } else {
    ARMFunctionInfo *AFI = DAG.getMachineFunction().getInfo<ARMFunctionInfo>();
    ARMPCLabelIndex = AFI->createPICLabelUId();
    unsigned PCAdj = (RelocM != Reloc::PIC_) ? 0 : (Subtarget->isThumb()?4:8);
    ARMConstantPoolValue *CPV =
      ARMConstantPoolConstant::Create(GV, ARMPCLabelIndex, ARMCP::CPValue,
                                      PCAdj);
    CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
  }
  CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);

  SDValue Result = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), CPAddr,
                               MachinePointerInfo::getConstantPool(),
                               false, false, false, 0);
  SDValue Chain = Result.getValue(1);

  if (RelocM == Reloc::PIC_) {
    SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex, MVT::i32);
    Result = DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Result, PICLabel);
  }

  if (Subtarget->GVIsIndirectSymbol(GV, RelocM))
    Result = DAG.getLoad(PtrVT, dl, Chain, Result, MachinePointerInfo::getGOT(),
                         false, false, false, 0);

  return Result;
}

SDValue ARMTargetLowering::LowerGLOBAL_OFFSET_TABLE(SDValue Op,
                                                    SelectionDAG &DAG) const {
  assert(Subtarget->isTargetELF() &&
         "GLOBAL OFFSET TABLE not implemented for non-ELF targets");
  MachineFunction &MF = DAG.getMachineFunction();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
  EVT PtrVT = getPointerTy();
  DebugLoc dl = Op.getDebugLoc();
  unsigned PCAdj = Subtarget->isThumb() ? 4 : 8;
  ARMConstantPoolValue *CPV =
    ARMConstantPoolSymbol::Create(*DAG.getContext(), "_GLOBAL_OFFSET_TABLE_",
                                  ARMPCLabelIndex, PCAdj);
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
  CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
  SDValue Result = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), CPAddr,
                               MachinePointerInfo::getConstantPool(),
                               false, false, false, 0);
  SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex, MVT::i32);
  return DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Result, PICLabel);
}

SDValue
ARMTargetLowering::LowerEH_SJLJ_SETJMP(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  SDValue Val = DAG.getConstant(0, MVT::i32);
  return DAG.getNode(ARMISD::EH_SJLJ_SETJMP, dl,
                     DAG.getVTList(MVT::i32, MVT::Other), Op.getOperand(0),
                     Op.getOperand(1), Val);
}

SDValue
ARMTargetLowering::LowerEH_SJLJ_LONGJMP(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  return DAG.getNode(ARMISD::EH_SJLJ_LONGJMP, dl, MVT::Other, Op.getOperand(0),
                     Op.getOperand(1), DAG.getConstant(0, MVT::i32));
}

SDValue
ARMTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG,
                                          const ARMSubtarget *Subtarget) const {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  DebugLoc dl = Op.getDebugLoc();
  switch (IntNo) {
  default: return SDValue();    // Don't custom lower most intrinsics.
  case Intrinsic::arm_thread_pointer: {
    EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
    return DAG.getNode(ARMISD::THREAD_POINTER, dl, PtrVT);
  }
  case Intrinsic::eh_sjlj_lsda: {
    MachineFunction &MF = DAG.getMachineFunction();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
    EVT PtrVT = getPointerTy();
    Reloc::Model RelocM = getTargetMachine().getRelocationModel();
    SDValue CPAddr;
    unsigned PCAdj = (RelocM != Reloc::PIC_)
      ? 0 : (Subtarget->isThumb() ? 4 : 8);
    ARMConstantPoolValue *CPV =
      ARMConstantPoolConstant::Create(MF.getFunction(), ARMPCLabelIndex,
                                      ARMCP::CPLSDA, PCAdj);
    CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 4);
    CPAddr = DAG.getNode(ARMISD::Wrapper, dl, MVT::i32, CPAddr);
    SDValue Result =
      DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), CPAddr,
                  MachinePointerInfo::getConstantPool(),
                  false, false, false, 0);

    if (RelocM == Reloc::PIC_) {
      SDValue PICLabel = DAG.getConstant(ARMPCLabelIndex, MVT::i32);
      Result = DAG.getNode(ARMISD::PIC_ADD, dl, PtrVT, Result, PICLabel);
    }
    return Result;
  }
  case Intrinsic::arm_neon_vmulls:
  case Intrinsic::arm_neon_vmullu: {
    unsigned NewOpc = (IntNo == Intrinsic::arm_neon_vmulls)
      ? ARMISD::VMULLs : ARMISD::VMULLu;
    return DAG.getNode(NewOpc, Op.getDebugLoc(), Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));
  }
  }
}

static SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG &DAG,
                                 const ARMSubtarget *Subtarget) {
  // FIXME: handle "fence singlethread" more efficiently.
  DebugLoc dl = Op.getDebugLoc();
  if (!Subtarget->hasDataBarrier()) {
    // Some ARMv6 cpus can support data barriers with an mcr instruction.
    // Thumb1 and pre-v6 ARM mode use a libcall instead and should never get
    // here.
    assert(Subtarget->hasV6Ops() && !Subtarget->isThumb() &&
           "Unexpected ISD::MEMBARRIER encountered. Should be libcall!");
    return DAG.getNode(ARMISD::MEMBARRIER_MCR, dl, MVT::Other, Op.getOperand(0),
                       DAG.getConstant(0, MVT::i32));
  }

  return DAG.getNode(ARMISD::MEMBARRIER, dl, MVT::Other, Op.getOperand(0),
                     DAG.getConstant(ARM_MB::ISH, MVT::i32));
}

static SDValue LowerPREFETCH(SDValue Op, SelectionDAG &DAG,
                             const ARMSubtarget *Subtarget) {
  // ARM pre v5TE and Thumb1 does not have preload instructions.
  if (!(Subtarget->isThumb2() ||
        (!Subtarget->isThumb1Only() && Subtarget->hasV5TEOps())))
    // Just preserve the chain.
    return Op.getOperand(0);

  DebugLoc dl = Op.getDebugLoc();
  unsigned isRead = ~cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue() & 1;
  if (!isRead &&
      (!Subtarget->hasV7Ops() || !Subtarget->hasMPExtension()))
    // ARMv7 with MP extension has PLDW.
    return Op.getOperand(0);

  unsigned isData = cast<ConstantSDNode>(Op.getOperand(4))->getZExtValue();
  if (Subtarget->isThumb()) {
    // Invert the bits.
    isRead = ~isRead & 1;
    isData = ~isData & 1;
  }

  return DAG.getNode(ARMISD::PRELOAD, dl, MVT::Other, Op.getOperand(0),
                     Op.getOperand(1), DAG.getConstant(isRead, MVT::i32),
                     DAG.getConstant(isData, MVT::i32));
}

static SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  ARMFunctionInfo *FuncInfo = MF.getInfo<ARMFunctionInfo>();

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  DebugLoc dl = Op.getDebugLoc();
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  SDValue FR = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(), PtrVT);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), dl, FR, Op.getOperand(1),
                      MachinePointerInfo(SV), false, false, 0);
}

SDValue
ARMTargetLowering::GetF64FormalArgument(CCValAssign &VA, CCValAssign &NextVA,
                                        SDValue &Root, SelectionDAG &DAG,
                                        DebugLoc dl) const {
  MachineFunction &MF = DAG.getMachineFunction();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  const TargetRegisterClass *RC;
  if (AFI->isThumb1OnlyFunction())
    RC = &ARM::tGPRRegClass;
  else
    RC = &ARM::GPRRegClass;

  // Transform the arguments stored in physical registers into virtual ones.
  unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
  SDValue ArgValue = DAG.getCopyFromReg(Root, dl, Reg, MVT::i32);

  SDValue ArgValue2;
  if (NextVA.isMemLoc()) {
    MachineFrameInfo *MFI = MF.getFrameInfo();
    int FI = MFI->CreateFixedObject(4, NextVA.getLocMemOffset(), true);

    // Create load node to retrieve arguments from the stack.
    SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
    ArgValue2 = DAG.getLoad(MVT::i32, dl, Root, FIN,
                            MachinePointerInfo::getFixedStack(FI),
                            false, false, false, 0);
  } else {
    Reg = MF.addLiveIn(NextVA.getLocReg(), RC);
    ArgValue2 = DAG.getCopyFromReg(Root, dl, Reg, MVT::i32);
  }

  return DAG.getNode(ARMISD::VMOVDRR, dl, MVT::f64, ArgValue, ArgValue2);
}

void
ARMTargetLowering::computeRegArea(CCState &CCInfo, MachineFunction &MF,
                                  unsigned InRegsParamRecordIdx,
                                  unsigned &ArgRegsSize,
                                  unsigned &ArgRegsSaveSize)
  const {
  unsigned NumGPRs;
  if (InRegsParamRecordIdx < CCInfo.getInRegsParamsCount()) {
    unsigned RBegin, REnd;
    CCInfo.getInRegsParamInfo(InRegsParamRecordIdx, RBegin, REnd);
    NumGPRs = REnd - RBegin;
  } else {
    unsigned int firstUnalloced;
    firstUnalloced = CCInfo.getFirstUnallocated(GPRArgRegs,
                                                sizeof(GPRArgRegs) /
                                                sizeof(GPRArgRegs[0]));
    NumGPRs = (firstUnalloced <= 3) ? (4 - firstUnalloced) : 0;
  }

  unsigned Align = MF.getTarget().getFrameLowering()->getStackAlignment();
  ArgRegsSize = NumGPRs * 4;
  ArgRegsSaveSize = (ArgRegsSize + Align - 1) & ~(Align - 1);
}

// The remaining GPRs hold either the beginning of variable-argument
// data, or the beginning of an aggregate passed by value (usually
// byval).  Either way, we allocate stack slots adjacent to the data
// provided by our caller, and store the unallocated registers there.
// If this is a variadic function, the va_list pointer will begin with
// these values; otherwise, this reassembles a (byval) structure that
// was split between registers and memory.
// Return: The frame index registers were stored into.
int
ARMTargetLowering::StoreByValRegs(CCState &CCInfo, SelectionDAG &DAG,
                                  DebugLoc dl, SDValue &Chain,
                                  const Value *OrigArg,
                                  unsigned InRegsParamRecordIdx,
                                  unsigned OffsetFromOrigArg,
                                  unsigned ArgOffset,
                                  bool ForceMutable) const {

  // Currently, two use-cases possible:
  // Case #1. Non var-args function, and we meet first byval parameter.
  //          Setup first unallocated register as first byval register;
  //          eat all remained registers
  //          (these two actions are performed by HandleByVal method).
  //          Then, here, we initialize stack frame with
  //          "store-reg" instructions.
  // Case #2. Var-args function, that doesn't contain byval parameters.
  //          The same: eat all remained unallocated registers,
  //          initialize stack frame.

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned firstRegToSaveIndex, lastRegToSaveIndex;
  unsigned RBegin, REnd;
  if (InRegsParamRecordIdx < CCInfo.getInRegsParamsCount()) {
    CCInfo.getInRegsParamInfo(InRegsParamRecordIdx, RBegin, REnd);
    firstRegToSaveIndex = RBegin - ARM::R0;
    lastRegToSaveIndex = REnd - ARM::R0;
  } else {
    firstRegToSaveIndex = CCInfo.getFirstUnallocated
      (GPRArgRegs, sizeof(GPRArgRegs) / sizeof(GPRArgRegs[0]));
    lastRegToSaveIndex = 4;
  }

  unsigned ArgRegsSize, ArgRegsSaveSize;
  computeRegArea(CCInfo, MF, InRegsParamRecordIdx, ArgRegsSize, ArgRegsSaveSize);

  // Store any by-val regs to their spots on the stack so that they may be
  // loaded by deferencing the result of formal parameter pointer or va_next.
  // Note: once stack area for byval/varargs registers
  // was initialized, it can't be initialized again.
  if (ArgRegsSaveSize) {

    int FrameIndex = MFI->CreateFixedObject(
                      ArgRegsSaveSize,
                      ArgOffset + ArgRegsSaveSize - ArgRegsSize,
                      false);
    SDValue FIN = DAG.getFrameIndex(FrameIndex, getPointerTy());

    SmallVector<SDValue, 4> MemOps;
    for (unsigned i = 0; firstRegToSaveIndex < lastRegToSaveIndex;
         ++firstRegToSaveIndex, ++i) {
      const TargetRegisterClass *RC;
      if (AFI->isThumb1OnlyFunction())
        RC = &ARM::tGPRRegClass;
      else
        RC = &ARM::GPRRegClass;

      unsigned VReg = MF.addLiveIn(GPRArgRegs[firstRegToSaveIndex], RC);
      SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i32);
      SDValue Store =
        DAG.getStore(Val.getValue(1), dl, Val, FIN,
                     MachinePointerInfo(OrigArg, OffsetFromOrigArg + 4*i),
                     false, false, 0);
      MemOps.push_back(Store);
      FIN = DAG.getNode(ISD::ADD, dl, getPointerTy(), FIN,
                        DAG.getConstant(4, getPointerTy()));
    }

    AFI->setArgRegsSaveSize(ArgRegsSaveSize + AFI->getArgRegsSaveSize());

    if (!MemOps.empty())
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                          &MemOps[0], MemOps.size());
    return FrameIndex;
  } else
    // This will point to the next argument passed via stack.
    return MFI->CreateFixedObject(4, ArgOffset, !ForceMutable);
}

// Setup stack frame, the va_list pointer will start from.
void
ARMTargetLowering::VarArgStyleRegisters(CCState &CCInfo, SelectionDAG &DAG,
                                        DebugLoc dl, SDValue &Chain,
                                        unsigned ArgOffset,
                                        bool ForceMutable) const {
  MachineFunction &MF = DAG.getMachineFunction();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  // Try to store any remaining integer argument regs
  // to their spots on the stack so that they may be loaded by deferencing
  // the result of va_next.
  // If there is no regs to be stored, just point address after last
  // argument passed via stack.
  int FrameIndex =
    StoreByValRegs(CCInfo, DAG, dl, Chain, 0, CCInfo.getInRegsParamsCount(),
                   0, ArgOffset, ForceMutable);

  AFI->setVarArgsFrameIndex(FrameIndex);
}

SDValue
ARMTargetLowering::LowerFormalArguments(SDValue Chain,
                                        CallingConv::ID CallConv, bool isVarArg,
                                        const SmallVectorImpl<ISD::InputArg>
                                          &Ins,
                                        DebugLoc dl, SelectionDAG &DAG,
                                        SmallVectorImpl<SDValue> &InVals)
                                          const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  ARMCCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                    getTargetMachine(), ArgLocs, *DAG.getContext(), Prologue);
  CCInfo.AnalyzeFormalArguments(Ins,
                                CCAssignFnForNode(CallConv, /* Return*/ false,
                                                  isVarArg));

  SmallVector<SDValue, 16> ArgValues;
  int lastInsIndex = -1;
  SDValue ArgValue;
  Function::const_arg_iterator CurOrigArg = MF.getFunction()->arg_begin();
  unsigned CurArgIdx = 0;

  // Initially ArgRegsSaveSize is zero.
  // Then we increase this value each time we meet byval parameter.
  // We also increase this value in case of varargs function.
  AFI->setArgRegsSaveSize(0);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    std::advance(CurOrigArg, Ins[VA.getValNo()].OrigArgIndex - CurArgIdx);
    CurArgIdx = Ins[VA.getValNo()].OrigArgIndex;
    // Arguments stored in registers.
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();

      if (VA.needsCustom()) {
        // f64 and vector types are split up into multiple registers or
        // combinations of registers and stack slots.
        if (VA.getLocVT() == MVT::v2f64) {
          SDValue ArgValue1 = GetF64FormalArgument(VA, ArgLocs[++i],
                                                   Chain, DAG, dl);
          VA = ArgLocs[++i]; // skip ahead to next loc
          SDValue ArgValue2;
          if (VA.isMemLoc()) {
            int FI = MFI->CreateFixedObject(8, VA.getLocMemOffset(), true);
            SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
            ArgValue2 = DAG.getLoad(MVT::f64, dl, Chain, FIN,
                                    MachinePointerInfo::getFixedStack(FI),
                                    false, false, false, 0);
          } else {
            ArgValue2 = GetF64FormalArgument(VA, ArgLocs[++i],
                                             Chain, DAG, dl);
          }
          ArgValue = DAG.getNode(ISD::UNDEF, dl, MVT::v2f64);
          ArgValue = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64,
                                 ArgValue, ArgValue1, DAG.getIntPtrConstant(0));
          ArgValue = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64,
                                 ArgValue, ArgValue2, DAG.getIntPtrConstant(1));
        } else
          ArgValue = GetF64FormalArgument(VA, ArgLocs[++i], Chain, DAG, dl);

      } else {
        const TargetRegisterClass *RC;

        if (RegVT == MVT::f32)
          RC = &ARM::SPRRegClass;
        else if (RegVT == MVT::f64)
          RC = &ARM::DPRRegClass;
        else if (RegVT == MVT::v2f64)
          RC = &ARM::QPRRegClass;
        else if (RegVT == MVT::i32)
          RC = AFI->isThumb1OnlyFunction() ?
            (const TargetRegisterClass*)&ARM::tGPRRegClass :
            (const TargetRegisterClass*)&ARM::GPRRegClass;
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
        ArgValue = DAG.getNode(ISD::BITCAST, dl, VA.getValVT(), ArgValue);
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

      int index = ArgLocs[i].getValNo();

      // Some Ins[] entries become multiple ArgLoc[] entries.
      // Process them only once.
      if (index != lastInsIndex)
        {
          ISD::ArgFlagsTy Flags = Ins[index].Flags;
          // FIXME: For now, all byval parameter objects are marked mutable.
          // This can be changed with more analysis.
          // In case of tail call optimization mark all arguments mutable.
          // Since they could be overwritten by lowering of arguments in case of
          // a tail call.
          if (Flags.isByVal()) {
            unsigned CurByValIndex = CCInfo.getInRegsParamsProceed();
            int FrameIndex = StoreByValRegs(
                CCInfo, DAG, dl, Chain, CurOrigArg,
                CurByValIndex,
                Ins[VA.getValNo()].PartOffset,
                VA.getLocMemOffset(),
                true /*force mutable frames*/);
            InVals.push_back(DAG.getFrameIndex(FrameIndex, getPointerTy()));
            CCInfo.nextInRegsParam();
          } else {
            int FI = MFI->CreateFixedObject(VA.getLocVT().getSizeInBits()/8,
                                            VA.getLocMemOffset(), true);

            // Create load nodes to retrieve arguments from the stack.
            SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
            InVals.push_back(DAG.getLoad(VA.getValVT(), dl, Chain, FIN,
                                         MachinePointerInfo::getFixedStack(FI),
                                         false, false, false, 0));
          }
          lastInsIndex = index;
        }
    }
  }

  // varargs
  if (isVarArg)
    VarArgStyleRegisters(CCInfo, DAG, dl, Chain,
                         CCInfo.getNextStackOffset());

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
        if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CP->getConstVal()))
          return CFP->getValueAPF().isPosZero();
    }
  }
  return false;
}

/// Returns appropriate ARM CMP (cmp) and corresponding condition code for
/// the given operands.
SDValue
ARMTargetLowering::getARMCmp(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                             SDValue &ARMcc, SelectionDAG &DAG,
                             DebugLoc dl) const {
  if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS.getNode())) {
    unsigned C = RHSC->getZExtValue();
    if (!isLegalICmpImmediate(C)) {
      // Constant does not fit, try adjusting it by one?
      switch (CC) {
      default: break;
      case ISD::SETLT:
      case ISD::SETGE:
        if (C != 0x80000000 && isLegalICmpImmediate(C-1)) {
          CC = (CC == ISD::SETLT) ? ISD::SETLE : ISD::SETGT;
          RHS = DAG.getConstant(C-1, MVT::i32);
        }
        break;
      case ISD::SETULT:
      case ISD::SETUGE:
        if (C != 0 && isLegalICmpImmediate(C-1)) {
          CC = (CC == ISD::SETULT) ? ISD::SETULE : ISD::SETUGT;
          RHS = DAG.getConstant(C-1, MVT::i32);
        }
        break;
      case ISD::SETLE:
      case ISD::SETGT:
        if (C != 0x7fffffff && isLegalICmpImmediate(C+1)) {
          CC = (CC == ISD::SETLE) ? ISD::SETLT : ISD::SETGE;
          RHS = DAG.getConstant(C+1, MVT::i32);
        }
        break;
      case ISD::SETULE:
      case ISD::SETUGT:
        if (C != 0xffffffff && isLegalICmpImmediate(C+1)) {
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
  ARMcc = DAG.getConstant(CondCode, MVT::i32);
  return DAG.getNode(CompareType, dl, MVT::Glue, LHS, RHS);
}

/// Returns a appropriate VFP CMP (fcmp{s|d}+fmstat) for the given operands.
SDValue
ARMTargetLowering::getVFPCmp(SDValue LHS, SDValue RHS, SelectionDAG &DAG,
                             DebugLoc dl) const {
  SDValue Cmp;
  if (!isFloatingPointZero(RHS))
    Cmp = DAG.getNode(ARMISD::CMPFP, dl, MVT::Glue, LHS, RHS);
  else
    Cmp = DAG.getNode(ARMISD::CMPFPw0, dl, MVT::Glue, LHS);
  return DAG.getNode(ARMISD::FMSTAT, dl, MVT::Glue, Cmp);
}

/// duplicateCmp - Glue values can have only one use, so this function
/// duplicates a comparison node.
SDValue
ARMTargetLowering::duplicateCmp(SDValue Cmp, SelectionDAG &DAG) const {
  unsigned Opc = Cmp.getOpcode();
  DebugLoc DL = Cmp.getDebugLoc();
  if (Opc == ARMISD::CMP || Opc == ARMISD::CMPZ)
    return DAG.getNode(Opc, DL, MVT::Glue, Cmp.getOperand(0),Cmp.getOperand(1));

  assert(Opc == ARMISD::FMSTAT && "unexpected comparison operation");
  Cmp = Cmp.getOperand(0);
  Opc = Cmp.getOpcode();
  if (Opc == ARMISD::CMPFP)
    Cmp = DAG.getNode(Opc, DL, MVT::Glue, Cmp.getOperand(0),Cmp.getOperand(1));
  else {
    assert(Opc == ARMISD::CMPFPw0 && "unexpected operand of FMSTAT");
    Cmp = DAG.getNode(Opc, DL, MVT::Glue, Cmp.getOperand(0));
  }
  return DAG.getNode(ARMISD::FMSTAT, DL, MVT::Glue, Cmp);
}

SDValue ARMTargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  SDValue Cond = Op.getOperand(0);
  SDValue SelectTrue = Op.getOperand(1);
  SDValue SelectFalse = Op.getOperand(2);
  DebugLoc dl = Op.getDebugLoc();

  // Convert:
  //
  //   (select (cmov 1, 0, cond), t, f) -> (cmov t, f, cond)
  //   (select (cmov 0, 1, cond), t, f) -> (cmov f, t, cond)
  //
  if (Cond.getOpcode() == ARMISD::CMOV && Cond.hasOneUse()) {
    const ConstantSDNode *CMOVTrue =
      dyn_cast<ConstantSDNode>(Cond.getOperand(0));
    const ConstantSDNode *CMOVFalse =
      dyn_cast<ConstantSDNode>(Cond.getOperand(1));

    if (CMOVTrue && CMOVFalse) {
      unsigned CMOVTrueVal = CMOVTrue->getZExtValue();
      unsigned CMOVFalseVal = CMOVFalse->getZExtValue();

      SDValue True;
      SDValue False;
      if (CMOVTrueVal == 1 && CMOVFalseVal == 0) {
        True = SelectTrue;
        False = SelectFalse;
      } else if (CMOVTrueVal == 0 && CMOVFalseVal == 1) {
        True = SelectFalse;
        False = SelectTrue;
      }

      if (True.getNode() && False.getNode()) {
        EVT VT = Op.getValueType();
        SDValue ARMcc = Cond.getOperand(2);
        SDValue CCR = Cond.getOperand(3);
        SDValue Cmp = duplicateCmp(Cond.getOperand(4), DAG);
        assert(True.getValueType() == VT);
        return DAG.getNode(ARMISD::CMOV, dl, VT, True, False, ARMcc, CCR, Cmp);
      }
    }
  }

  // ARM's BooleanContents value is UndefinedBooleanContent. Mask out the
  // undefined bits before doing a full-word comparison with zero.
  Cond = DAG.getNode(ISD::AND, dl, Cond.getValueType(), Cond,
                     DAG.getConstant(1, Cond.getValueType()));

  return DAG.getSelectCC(dl, Cond,
                         DAG.getConstant(0, Cond.getValueType()),
                         SelectTrue, SelectFalse, ISD::SETNE);
}

SDValue ARMTargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);
  DebugLoc dl = Op.getDebugLoc();

  if (LHS.getValueType() == MVT::i32) {
    SDValue ARMcc;
    SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
    SDValue Cmp = getARMCmp(LHS, RHS, CC, ARMcc, DAG, dl);
    return DAG.getNode(ARMISD::CMOV, dl, VT, FalseVal, TrueVal, ARMcc, CCR,Cmp);
  }

  ARMCC::CondCodes CondCode, CondCode2;
  FPCCToARMCC(CC, CondCode, CondCode2);

  SDValue ARMcc = DAG.getConstant(CondCode, MVT::i32);
  SDValue Cmp = getVFPCmp(LHS, RHS, DAG, dl);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  SDValue Result = DAG.getNode(ARMISD::CMOV, dl, VT, FalseVal, TrueVal,
                               ARMcc, CCR, Cmp);
  if (CondCode2 != ARMCC::AL) {
    SDValue ARMcc2 = DAG.getConstant(CondCode2, MVT::i32);
    // FIXME: Needs another CMP because flag can have but one use.
    SDValue Cmp2 = getVFPCmp(LHS, RHS, DAG, dl);
    Result = DAG.getNode(ARMISD::CMOV, dl, VT,
                         Result, TrueVal, ARMcc2, CCR, Cmp2);
  }
  return Result;
}

/// canChangeToInt - Given the fp compare operand, return true if it is suitable
/// to morph to an integer compare sequence.
static bool canChangeToInt(SDValue Op, bool &SeenZero,
                           const ARMSubtarget *Subtarget) {
  SDNode *N = Op.getNode();
  if (!N->hasOneUse())
    // Otherwise it requires moving the value from fp to integer registers.
    return false;
  if (!N->getNumValues())
    return false;
  EVT VT = Op.getValueType();
  if (VT != MVT::f32 && !Subtarget->isFPBrccSlow())
    // f32 case is generally profitable. f64 case only makes sense when vcmpe +
    // vmrs are very slow, e.g. cortex-a8.
    return false;

  if (isFloatingPointZero(Op)) {
    SeenZero = true;
    return true;
  }
  return ISD::isNormalLoad(N);
}

static SDValue bitcastf32Toi32(SDValue Op, SelectionDAG &DAG) {
  if (isFloatingPointZero(Op))
    return DAG.getConstant(0, MVT::i32);

  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(Op))
    return DAG.getLoad(MVT::i32, Op.getDebugLoc(),
                       Ld->getChain(), Ld->getBasePtr(), Ld->getPointerInfo(),
                       Ld->isVolatile(), Ld->isNonTemporal(),
                       Ld->isInvariant(), Ld->getAlignment());

  llvm_unreachable("Unknown VFP cmp argument!");
}

static void expandf64Toi32(SDValue Op, SelectionDAG &DAG,
                           SDValue &RetVal1, SDValue &RetVal2) {
  if (isFloatingPointZero(Op)) {
    RetVal1 = DAG.getConstant(0, MVT::i32);
    RetVal2 = DAG.getConstant(0, MVT::i32);
    return;
  }

  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(Op)) {
    SDValue Ptr = Ld->getBasePtr();
    RetVal1 = DAG.getLoad(MVT::i32, Op.getDebugLoc(),
                          Ld->getChain(), Ptr,
                          Ld->getPointerInfo(),
                          Ld->isVolatile(), Ld->isNonTemporal(),
                          Ld->isInvariant(), Ld->getAlignment());

    EVT PtrType = Ptr.getValueType();
    unsigned NewAlign = MinAlign(Ld->getAlignment(), 4);
    SDValue NewPtr = DAG.getNode(ISD::ADD, Op.getDebugLoc(),
                                 PtrType, Ptr, DAG.getConstant(4, PtrType));
    RetVal2 = DAG.getLoad(MVT::i32, Op.getDebugLoc(),
                          Ld->getChain(), NewPtr,
                          Ld->getPointerInfo().getWithOffset(4),
                          Ld->isVolatile(), Ld->isNonTemporal(),
                          Ld->isInvariant(), NewAlign);
    return;
  }

  llvm_unreachable("Unknown VFP cmp argument!");
}

/// OptimizeVFPBrcond - With -enable-unsafe-fp-math, it's legal to optimize some
/// f32 and even f64 comparisons to integer ones.
SDValue
ARMTargetLowering::OptimizeVFPBrcond(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  DebugLoc dl = Op.getDebugLoc();

  bool LHSSeenZero = false;
  bool LHSOk = canChangeToInt(LHS, LHSSeenZero, Subtarget);
  bool RHSSeenZero = false;
  bool RHSOk = canChangeToInt(RHS, RHSSeenZero, Subtarget);
  if (LHSOk && RHSOk && (LHSSeenZero || RHSSeenZero)) {
    // If unsafe fp math optimization is enabled and there are no other uses of
    // the CMP operands, and the condition code is EQ or NE, we can optimize it
    // to an integer comparison.
    if (CC == ISD::SETOEQ)
      CC = ISD::SETEQ;
    else if (CC == ISD::SETUNE)
      CC = ISD::SETNE;

    SDValue Mask = DAG.getConstant(0x7fffffff, MVT::i32);
    SDValue ARMcc;
    if (LHS.getValueType() == MVT::f32) {
      LHS = DAG.getNode(ISD::AND, dl, MVT::i32,
                        bitcastf32Toi32(LHS, DAG), Mask);
      RHS = DAG.getNode(ISD::AND, dl, MVT::i32,
                        bitcastf32Toi32(RHS, DAG), Mask);
      SDValue Cmp = getARMCmp(LHS, RHS, CC, ARMcc, DAG, dl);
      SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
      return DAG.getNode(ARMISD::BRCOND, dl, MVT::Other,
                         Chain, Dest, ARMcc, CCR, Cmp);
    }

    SDValue LHS1, LHS2;
    SDValue RHS1, RHS2;
    expandf64Toi32(LHS, DAG, LHS1, LHS2);
    expandf64Toi32(RHS, DAG, RHS1, RHS2);
    LHS2 = DAG.getNode(ISD::AND, dl, MVT::i32, LHS2, Mask);
    RHS2 = DAG.getNode(ISD::AND, dl, MVT::i32, RHS2, Mask);
    ARMCC::CondCodes CondCode = IntCCToARMCC(CC);
    ARMcc = DAG.getConstant(CondCode, MVT::i32);
    SDVTList VTList = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue Ops[] = { Chain, ARMcc, LHS1, LHS2, RHS1, RHS2, Dest };
    return DAG.getNode(ARMISD::BCC_i64, dl, VTList, Ops, 7);
  }

  return SDValue();
}

SDValue ARMTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  DebugLoc dl = Op.getDebugLoc();

  if (LHS.getValueType() == MVT::i32) {
    SDValue ARMcc;
    SDValue Cmp = getARMCmp(LHS, RHS, CC, ARMcc, DAG, dl);
    SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
    return DAG.getNode(ARMISD::BRCOND, dl, MVT::Other,
                       Chain, Dest, ARMcc, CCR, Cmp);
  }

  assert(LHS.getValueType() == MVT::f32 || LHS.getValueType() == MVT::f64);

  if (getTargetMachine().Options.UnsafeFPMath &&
      (CC == ISD::SETEQ || CC == ISD::SETOEQ ||
       CC == ISD::SETNE || CC == ISD::SETUNE)) {
    SDValue Result = OptimizeVFPBrcond(Op, DAG);
    if (Result.getNode())
      return Result;
  }

  ARMCC::CondCodes CondCode, CondCode2;
  FPCCToARMCC(CC, CondCode, CondCode2);

  SDValue ARMcc = DAG.getConstant(CondCode, MVT::i32);
  SDValue Cmp = getVFPCmp(LHS, RHS, DAG, dl);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  SDVTList VTList = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue Ops[] = { Chain, Dest, ARMcc, CCR, Cmp };
  SDValue Res = DAG.getNode(ARMISD::BRCOND, dl, VTList, Ops, 5);
  if (CondCode2 != ARMCC::AL) {
    ARMcc = DAG.getConstant(CondCode2, MVT::i32);
    SDValue Ops[] = { Res, Dest, ARMcc, CCR, Res.getValue(1) };
    Res = DAG.getNode(ARMISD::BRCOND, dl, VTList, Ops, 5);
  }
  return Res;
}

SDValue ARMTargetLowering::LowerBR_JT(SDValue Op, SelectionDAG &DAG) const {
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
    Addr = DAG.getLoad((EVT)MVT::i32, dl, Chain, Addr,
                       MachinePointerInfo::getJumpTable(),
                       false, false, false, 0);
    Chain = Addr.getValue(1);
    Addr = DAG.getNode(ISD::ADD, dl, PTy, Addr, Table);
    return DAG.getNode(ARMISD::BR_JT, dl, MVT::Other, Chain, Addr, JTI, UId);
  } else {
    Addr = DAG.getLoad(PTy, dl, Chain, Addr,
                       MachinePointerInfo::getJumpTable(),
                       false, false, false, 0);
    Chain = Addr.getValue(1);
    return DAG.getNode(ARMISD::BR_JT, dl, MVT::Other, Chain, Addr, JTI, UId);
  }
}

static SDValue LowerVectorFP_TO_INT(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();

  if (Op.getValueType().getVectorElementType() == MVT::i32) {
    if (Op.getOperand(0).getValueType().getVectorElementType() == MVT::f32)
      return Op;
    return DAG.UnrollVectorOp(Op.getNode());
  }

  assert(Op.getOperand(0).getValueType() == MVT::v4f32 &&
         "Invalid type for custom lowering!");
  if (VT != MVT::v4i16)
    return DAG.UnrollVectorOp(Op.getNode());

  Op = DAG.getNode(Op.getOpcode(), dl, MVT::v4i32, Op.getOperand(0));
  return DAG.getNode(ISD::TRUNCATE, dl, VT, Op);
}

static SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  if (VT.isVector())
    return LowerVectorFP_TO_INT(Op, DAG);

  DebugLoc dl = Op.getDebugLoc();
  unsigned Opc;

  switch (Op.getOpcode()) {
  default: llvm_unreachable("Invalid opcode!");
  case ISD::FP_TO_SINT:
    Opc = ARMISD::FTOSI;
    break;
  case ISD::FP_TO_UINT:
    Opc = ARMISD::FTOUI;
    break;
  }
  Op = DAG.getNode(Opc, dl, MVT::f32, Op.getOperand(0));
  return DAG.getNode(ISD::BITCAST, dl, MVT::i32, Op);
}

static SDValue LowerVectorINT_TO_FP(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();

  if (Op.getOperand(0).getValueType().getVectorElementType() == MVT::i32) {
    if (VT.getVectorElementType() == MVT::f32)
      return Op;
    return DAG.UnrollVectorOp(Op.getNode());
  }

  assert(Op.getOperand(0).getValueType() == MVT::v4i16 &&
         "Invalid type for custom lowering!");
  if (VT != MVT::v4f32)
    return DAG.UnrollVectorOp(Op.getNode());

  unsigned CastOpc;
  unsigned Opc;
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Invalid opcode!");
  case ISD::SINT_TO_FP:
    CastOpc = ISD::SIGN_EXTEND;
    Opc = ISD::SINT_TO_FP;
    break;
  case ISD::UINT_TO_FP:
    CastOpc = ISD::ZERO_EXTEND;
    Opc = ISD::UINT_TO_FP;
    break;
  }

  Op = DAG.getNode(CastOpc, dl, MVT::v4i32, Op.getOperand(0));
  return DAG.getNode(Opc, dl, VT, Op);
}

static SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  if (VT.isVector())
    return LowerVectorINT_TO_FP(Op, DAG);

  DebugLoc dl = Op.getDebugLoc();
  unsigned Opc;

  switch (Op.getOpcode()) {
  default: llvm_unreachable("Invalid opcode!");
  case ISD::SINT_TO_FP:
    Opc = ARMISD::SITOF;
    break;
  case ISD::UINT_TO_FP:
    Opc = ARMISD::UITOF;
    break;
  }

  Op = DAG.getNode(ISD::BITCAST, dl, MVT::f32, Op.getOperand(0));
  return DAG.getNode(Opc, dl, VT, Op);
}

SDValue ARMTargetLowering::LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const {
  // Implement fcopysign with a fabs and a conditional fneg.
  SDValue Tmp0 = Op.getOperand(0);
  SDValue Tmp1 = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT SrcVT = Tmp1.getValueType();
  bool InGPR = Tmp0.getOpcode() == ISD::BITCAST ||
    Tmp0.getOpcode() == ARMISD::VMOVDRR;
  bool UseNEON = !InGPR && Subtarget->hasNEON();

  if (UseNEON) {
    // Use VBSL to copy the sign bit.
    unsigned EncodedVal = ARM_AM::createNEONModImm(0x6, 0x80);
    SDValue Mask = DAG.getNode(ARMISD::VMOVIMM, dl, MVT::v2i32,
                               DAG.getTargetConstant(EncodedVal, MVT::i32));
    EVT OpVT = (VT == MVT::f32) ? MVT::v2i32 : MVT::v1i64;
    if (VT == MVT::f64)
      Mask = DAG.getNode(ARMISD::VSHL, dl, OpVT,
                         DAG.getNode(ISD::BITCAST, dl, OpVT, Mask),
                         DAG.getConstant(32, MVT::i32));
    else /*if (VT == MVT::f32)*/
      Tmp0 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2f32, Tmp0);
    if (SrcVT == MVT::f32) {
      Tmp1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2f32, Tmp1);
      if (VT == MVT::f64)
        Tmp1 = DAG.getNode(ARMISD::VSHL, dl, OpVT,
                           DAG.getNode(ISD::BITCAST, dl, OpVT, Tmp1),
                           DAG.getConstant(32, MVT::i32));
    } else if (VT == MVT::f32)
      Tmp1 = DAG.getNode(ARMISD::VSHRu, dl, MVT::v1i64,
                         DAG.getNode(ISD::BITCAST, dl, MVT::v1i64, Tmp1),
                         DAG.getConstant(32, MVT::i32));
    Tmp0 = DAG.getNode(ISD::BITCAST, dl, OpVT, Tmp0);
    Tmp1 = DAG.getNode(ISD::BITCAST, dl, OpVT, Tmp1);

    SDValue AllOnes = DAG.getTargetConstant(ARM_AM::createNEONModImm(0xe, 0xff),
                                            MVT::i32);
    AllOnes = DAG.getNode(ARMISD::VMOVIMM, dl, MVT::v8i8, AllOnes);
    SDValue MaskNot = DAG.getNode(ISD::XOR, dl, OpVT, Mask,
                                  DAG.getNode(ISD::BITCAST, dl, OpVT, AllOnes));

    SDValue Res = DAG.getNode(ISD::OR, dl, OpVT,
                              DAG.getNode(ISD::AND, dl, OpVT, Tmp1, Mask),
                              DAG.getNode(ISD::AND, dl, OpVT, Tmp0, MaskNot));
    if (VT == MVT::f32) {
      Res = DAG.getNode(ISD::BITCAST, dl, MVT::v2f32, Res);
      Res = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f32, Res,
                        DAG.getConstant(0, MVT::i32));
    } else {
      Res = DAG.getNode(ISD::BITCAST, dl, MVT::f64, Res);
    }

    return Res;
  }

  // Bitcast operand 1 to i32.
  if (SrcVT == MVT::f64)
    Tmp1 = DAG.getNode(ARMISD::VMOVRRD, dl, DAG.getVTList(MVT::i32, MVT::i32),
                       &Tmp1, 1).getValue(1);
  Tmp1 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, Tmp1);

  // Or in the signbit with integer operations.
  SDValue Mask1 = DAG.getConstant(0x80000000, MVT::i32);
  SDValue Mask2 = DAG.getConstant(0x7fffffff, MVT::i32);
  Tmp1 = DAG.getNode(ISD::AND, dl, MVT::i32, Tmp1, Mask1);
  if (VT == MVT::f32) {
    Tmp0 = DAG.getNode(ISD::AND, dl, MVT::i32,
                       DAG.getNode(ISD::BITCAST, dl, MVT::i32, Tmp0), Mask2);
    return DAG.getNode(ISD::BITCAST, dl, MVT::f32,
                       DAG.getNode(ISD::OR, dl, MVT::i32, Tmp0, Tmp1));
  }

  // f64: Or the high part with signbit and then combine two parts.
  Tmp0 = DAG.getNode(ARMISD::VMOVRRD, dl, DAG.getVTList(MVT::i32, MVT::i32),
                     &Tmp0, 1);
  SDValue Lo = Tmp0.getValue(0);
  SDValue Hi = DAG.getNode(ISD::AND, dl, MVT::i32, Tmp0.getValue(1), Mask2);
  Hi = DAG.getNode(ISD::OR, dl, MVT::i32, Hi, Tmp1);
  return DAG.getNode(ARMISD::VMOVDRR, dl, MVT::f64, Lo, Hi);
}

SDValue ARMTargetLowering::LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const{
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MFI->setReturnAddressIsTaken(true);

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (Depth) {
    SDValue FrameAddr = LowerFRAMEADDR(Op, DAG);
    SDValue Offset = DAG.getConstant(4, MVT::i32);
    return DAG.getLoad(VT, dl, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, dl, VT, FrameAddr, Offset),
                       MachinePointerInfo(), false, false, false, 0);
  }

  // Return LR, which contains the return address. Mark it an implicit live-in.
  unsigned Reg = MF.addLiveIn(ARM::LR, getRegClassFor(MVT::i32));
  return DAG.getCopyFromReg(DAG.getEntryNode(), dl, Reg, VT);
}

SDValue ARMTargetLowering::LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setFrameAddressIsTaken(true);

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();  // FIXME probably not meaningful
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  unsigned FrameReg = (Subtarget->isThumb() || Subtarget->isTargetDarwin())
    ? ARM::R7 : ARM::R11;
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl, FrameReg, VT);
  while (Depth--)
    FrameAddr = DAG.getLoad(VT, dl, DAG.getEntryNode(), FrameAddr,
                            MachinePointerInfo(),
                            false, false, false, 0);
  return FrameAddr;
}

/// Custom Expand long vector extensions, where size(DestVec) > 2*size(SrcVec),
/// and size(DestVec) > 128-bits.
/// This is achieved by doing the one extension from the SrcVec, splitting the
/// result, extending these parts, and then concatenating these into the
/// destination.
static SDValue ExpandVectorExtension(SDNode *N, SelectionDAG &DAG) {
  SDValue Op = N->getOperand(0);
  EVT SrcVT = Op.getValueType();
  EVT DestVT = N->getValueType(0);

  assert(DestVT.getSizeInBits() > 128 &&
         "Custom sext/zext expansion needs >128-bit vector.");
  // If this is a normal length extension, use the default expansion.
  if (SrcVT.getSizeInBits()*4 != DestVT.getSizeInBits() &&
      SrcVT.getSizeInBits()*8 != DestVT.getSizeInBits())
    return SDValue();

  DebugLoc dl = N->getDebugLoc();
  unsigned SrcEltSize = SrcVT.getVectorElementType().getSizeInBits();
  unsigned DestEltSize = DestVT.getVectorElementType().getSizeInBits();
  unsigned NumElts = SrcVT.getVectorNumElements();
  LLVMContext &Ctx = *DAG.getContext();
  SDValue Mid, SplitLo, SplitHi, ExtLo, ExtHi;

  EVT MidVT = EVT::getVectorVT(Ctx, EVT::getIntegerVT(Ctx, SrcEltSize*2),
                               NumElts);
  EVT SplitVT = EVT::getVectorVT(Ctx, EVT::getIntegerVT(Ctx, SrcEltSize*2),
                                 NumElts/2);
  EVT ExtVT = EVT::getVectorVT(Ctx, EVT::getIntegerVT(Ctx, DestEltSize),
                               NumElts/2);

  Mid = DAG.getNode(N->getOpcode(), dl, MidVT, Op);
  SplitLo = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, SplitVT, Mid,
                        DAG.getIntPtrConstant(0));
  SplitHi = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, SplitVT, Mid,
                        DAG.getIntPtrConstant(NumElts/2));
  ExtLo = DAG.getNode(N->getOpcode(), dl, ExtVT, SplitLo);
  ExtHi = DAG.getNode(N->getOpcode(), dl, ExtVT, SplitHi);
  return DAG.getNode(ISD::CONCAT_VECTORS, dl, DestVT, ExtLo, ExtHi);
}

/// ExpandBITCAST - If the target supports VFP, this function is called to
/// expand a bit convert where either the source or destination type is i64 to
/// use a VMOVDRR or VMOVRRD node.  This should not be done when the non-i64
/// operand type is illegal (e.g., v2f32 for a target that doesn't support
/// vectors), since the legalizer won't know what to do with that.
static SDValue ExpandBITCAST(SDNode *N, SelectionDAG &DAG) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  DebugLoc dl = N->getDebugLoc();
  SDValue Op = N->getOperand(0);

  // This function is only supposed to be called for i64 types, either as the
  // source or destination of the bit convert.
  EVT SrcVT = Op.getValueType();
  EVT DstVT = N->getValueType(0);
  assert((SrcVT == MVT::i64 || DstVT == MVT::i64) &&
         "ExpandBITCAST called for non-i64 type");

  // Turn i64->f64 into VMOVDRR.
  if (SrcVT == MVT::i64 && TLI.isTypeLegal(DstVT)) {
    SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, Op,
                             DAG.getConstant(0, MVT::i32));
    SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, Op,
                             DAG.getConstant(1, MVT::i32));
    return DAG.getNode(ISD::BITCAST, dl, DstVT,
                       DAG.getNode(ARMISD::VMOVDRR, dl, MVT::f64, Lo, Hi));
  }

  // Turn f64->i64 into VMOVRRD.
  if (DstVT == MVT::i64 && TLI.isTypeLegal(SrcVT)) {
    SDValue Cvt = DAG.getNode(ARMISD::VMOVRRD, dl,
                              DAG.getVTList(MVT::i32, MVT::i32), &Op, 1);
    // Merge the pieces into a single i64 value.
    return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Cvt, Cvt.getValue(1));
  }

  return SDValue();
}

/// getZeroVector - Returns a vector of specified type with all zero elements.
/// Zero vectors are used to represent vector negation and in those cases
/// will be implemented with the NEON VNEG instruction.  However, VNEG does
/// not support i64 elements, so sometimes the zero vectors will need to be
/// explicitly constructed.  Regardless, use a canonical VMOV to create the
/// zero vector.
static SDValue getZeroVector(EVT VT, SelectionDAG &DAG, DebugLoc dl) {
  assert(VT.isVector() && "Expected a vector type");
  // The canonical modified immediate encoding of a zero vector is....0!
  SDValue EncodedVal = DAG.getTargetConstant(0, MVT::i32);
  EVT VmovVT = VT.is128BitVector() ? MVT::v4i32 : MVT::v2i32;
  SDValue Vmov = DAG.getNode(ARMISD::VMOVIMM, dl, VmovVT, EncodedVal);
  return DAG.getNode(ISD::BITCAST, dl, VT, Vmov);
}

/// LowerShiftRightParts - Lower SRA_PARTS, which returns two
/// i32 values and take a 2 x i32 value to shift plus a shift amount.
SDValue ARMTargetLowering::LowerShiftRightParts(SDValue Op,
                                                SelectionDAG &DAG) const {
  assert(Op.getNumOperands() == 3 && "Not a double-shift!");
  EVT VT = Op.getValueType();
  unsigned VTBits = VT.getSizeInBits();
  DebugLoc dl = Op.getDebugLoc();
  SDValue ShOpLo = Op.getOperand(0);
  SDValue ShOpHi = Op.getOperand(1);
  SDValue ShAmt  = Op.getOperand(2);
  SDValue ARMcc;
  unsigned Opc = (Op.getOpcode() == ISD::SRA_PARTS) ? ISD::SRA : ISD::SRL;

  assert(Op.getOpcode() == ISD::SRA_PARTS || Op.getOpcode() == ISD::SRL_PARTS);

  SDValue RevShAmt = DAG.getNode(ISD::SUB, dl, MVT::i32,
                                 DAG.getConstant(VTBits, MVT::i32), ShAmt);
  SDValue Tmp1 = DAG.getNode(ISD::SRL, dl, VT, ShOpLo, ShAmt);
  SDValue ExtraShAmt = DAG.getNode(ISD::SUB, dl, MVT::i32, ShAmt,
                                   DAG.getConstant(VTBits, MVT::i32));
  SDValue Tmp2 = DAG.getNode(ISD::SHL, dl, VT, ShOpHi, RevShAmt);
  SDValue FalseVal = DAG.getNode(ISD::OR, dl, VT, Tmp1, Tmp2);
  SDValue TrueVal = DAG.getNode(Opc, dl, VT, ShOpHi, ExtraShAmt);

  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  SDValue Cmp = getARMCmp(ExtraShAmt, DAG.getConstant(0, MVT::i32), ISD::SETGE,
                          ARMcc, DAG, dl);
  SDValue Hi = DAG.getNode(Opc, dl, VT, ShOpHi, ShAmt);
  SDValue Lo = DAG.getNode(ARMISD::CMOV, dl, VT, FalseVal, TrueVal, ARMcc,
                           CCR, Cmp);

  SDValue Ops[2] = { Lo, Hi };
  return DAG.getMergeValues(Ops, 2, dl);
}

/// LowerShiftLeftParts - Lower SHL_PARTS, which returns two
/// i32 values and take a 2 x i32 value to shift plus a shift amount.
SDValue ARMTargetLowering::LowerShiftLeftParts(SDValue Op,
                                               SelectionDAG &DAG) const {
  assert(Op.getNumOperands() == 3 && "Not a double-shift!");
  EVT VT = Op.getValueType();
  unsigned VTBits = VT.getSizeInBits();
  DebugLoc dl = Op.getDebugLoc();
  SDValue ShOpLo = Op.getOperand(0);
  SDValue ShOpHi = Op.getOperand(1);
  SDValue ShAmt  = Op.getOperand(2);
  SDValue ARMcc;

  assert(Op.getOpcode() == ISD::SHL_PARTS);
  SDValue RevShAmt = DAG.getNode(ISD::SUB, dl, MVT::i32,
                                 DAG.getConstant(VTBits, MVT::i32), ShAmt);
  SDValue Tmp1 = DAG.getNode(ISD::SRL, dl, VT, ShOpLo, RevShAmt);
  SDValue ExtraShAmt = DAG.getNode(ISD::SUB, dl, MVT::i32, ShAmt,
                                   DAG.getConstant(VTBits, MVT::i32));
  SDValue Tmp2 = DAG.getNode(ISD::SHL, dl, VT, ShOpHi, ShAmt);
  SDValue Tmp3 = DAG.getNode(ISD::SHL, dl, VT, ShOpLo, ExtraShAmt);

  SDValue FalseVal = DAG.getNode(ISD::OR, dl, VT, Tmp1, Tmp2);
  SDValue CCR = DAG.getRegister(ARM::CPSR, MVT::i32);
  SDValue Cmp = getARMCmp(ExtraShAmt, DAG.getConstant(0, MVT::i32), ISD::SETGE,
                          ARMcc, DAG, dl);
  SDValue Lo = DAG.getNode(ISD::SHL, dl, VT, ShOpLo, ShAmt);
  SDValue Hi = DAG.getNode(ARMISD::CMOV, dl, VT, FalseVal, Tmp3, ARMcc,
                           CCR, Cmp);

  SDValue Ops[2] = { Lo, Hi };
  return DAG.getMergeValues(Ops, 2, dl);
}

SDValue ARMTargetLowering::LowerFLT_ROUNDS_(SDValue Op,
                                            SelectionDAG &DAG) const {
  // The rounding mode is in bits 23:22 of the FPSCR.
  // The ARM rounding mode value to FLT_ROUNDS mapping is 0->1, 1->2, 2->3, 3->0
  // The formula we use to implement this is (((FPSCR + 1 << 22) >> 22) & 3)
  // so that the shift + and get folded into a bitfield extract.
  DebugLoc dl = Op.getDebugLoc();
  SDValue FPSCR = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::i32,
                              DAG.getConstant(Intrinsic::arm_get_fpscr,
                                              MVT::i32));
  SDValue FltRounds = DAG.getNode(ISD::ADD, dl, MVT::i32, FPSCR,
                                  DAG.getConstant(1U << 22, MVT::i32));
  SDValue RMODE = DAG.getNode(ISD::SRL, dl, MVT::i32, FltRounds,
                              DAG.getConstant(22, MVT::i32));
  return DAG.getNode(ISD::AND, dl, MVT::i32, RMODE,
                     DAG.getConstant(3, MVT::i32));
}

static SDValue LowerCTTZ(SDNode *N, SelectionDAG &DAG,
                         const ARMSubtarget *ST) {
  EVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  if (!ST->hasV6T2Ops())
    return SDValue();

  SDValue rbit = DAG.getNode(ARMISD::RBIT, dl, VT, N->getOperand(0));
  return DAG.getNode(ISD::CTLZ, dl, VT, rbit);
}

/// getCTPOP16BitCounts - Returns a v8i8/v16i8 vector containing the bit-count
/// for each 16-bit element from operand, repeated.  The basic idea is to
/// leverage vcnt to get the 8-bit counts, gather and add the results.
///
/// Trace for v4i16:
/// input    = [v0    v1    v2    v3   ] (vi 16-bit element)
/// cast: N0 = [w0 w1 w2 w3 w4 w5 w6 w7] (v0 = [w0 w1], wi 8-bit element)
/// vcnt: N1 = [b0 b1 b2 b3 b4 b5 b6 b7] (bi = bit-count of 8-bit element wi)
/// vrev: N2 = [b1 b0 b3 b2 b5 b4 b7 b6]
///            [b0 b1 b2 b3 b4 b5 b6 b7]
///           +[b1 b0 b3 b2 b5 b4 b7 b6]
/// N3=N1+N2 = [k0 k0 k1 k1 k2 k2 k3 k3] (k0 = b0+b1 = bit-count of 16-bit v0,
/// vuzp:    = [k0 k1 k2 k3 k0 k1 k2 k3]  each ki is 8-bits)
static SDValue getCTPOP16BitCounts(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  DebugLoc DL = N->getDebugLoc();

  EVT VT8Bit = VT.is64BitVector() ? MVT::v8i8 : MVT::v16i8;
  SDValue N0 = DAG.getNode(ISD::BITCAST, DL, VT8Bit, N->getOperand(0));
  SDValue N1 = DAG.getNode(ISD::CTPOP, DL, VT8Bit, N0);
  SDValue N2 = DAG.getNode(ARMISD::VREV16, DL, VT8Bit, N1);
  SDValue N3 = DAG.getNode(ISD::ADD, DL, VT8Bit, N1, N2);
  return DAG.getNode(ARMISD::VUZP, DL, VT8Bit, N3, N3);
}

/// lowerCTPOP16BitElements - Returns a v4i16/v8i16 vector containing the
/// bit-count for each 16-bit element from the operand.  We need slightly
/// different sequencing for v4i16 and v8i16 to stay within NEON's available
/// 64/128-bit registers.
///
/// Trace for v4i16:
/// input           = [v0    v1    v2    v3    ] (vi 16-bit element)
/// v8i8: BitCounts = [k0 k1 k2 k3 k0 k1 k2 k3 ] (ki is the bit-count of vi)
/// v8i16:Extended  = [k0    k1    k2    k3    k0    k1    k2    k3    ]
/// v4i16:Extracted = [k0    k1    k2    k3    ]
static SDValue lowerCTPOP16BitElements(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  DebugLoc DL = N->getDebugLoc();

  SDValue BitCounts = getCTPOP16BitCounts(N, DAG);
  if (VT.is64BitVector()) {
    SDValue Extended = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::v8i16, BitCounts);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v4i16, Extended,
                       DAG.getIntPtrConstant(0));
  } else {
    SDValue Extracted = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v8i8,
                                    BitCounts, DAG.getIntPtrConstant(0));
    return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::v8i16, Extracted);
  }
}

/// lowerCTPOP32BitElements - Returns a v2i32/v4i32 vector containing the
/// bit-count for each 32-bit element from the operand.  The idea here is
/// to split the vector into 16-bit elements, leverage the 16-bit count
/// routine, and then combine the results.
///
/// Trace for v2i32 (v4i32 similar with Extracted/Extended exchanged):
/// input    = [v0    v1    ] (vi: 32-bit elements)
/// Bitcast  = [w0 w1 w2 w3 ] (wi: 16-bit elements, v0 = [w0 w1])
/// Counts16 = [k0 k1 k2 k3 ] (ki: 16-bit elements, bit-count of wi)
/// vrev: N0 = [k1 k0 k3 k2 ]
///            [k0 k1 k2 k3 ]
///       N1 =+[k1 k0 k3 k2 ]
///            [k0 k2 k1 k3 ]
///       N2 =+[k1 k3 k0 k2 ]
///            [k0    k2    k1    k3    ]
/// Extended =+[k1    k3    k0    k2    ]
///            [k0    k2    ]
/// Extracted=+[k1    k3    ]
///
static SDValue lowerCTPOP32BitElements(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  DebugLoc DL = N->getDebugLoc();

  EVT VT16Bit = VT.is64BitVector() ? MVT::v4i16 : MVT::v8i16;

  SDValue Bitcast = DAG.getNode(ISD::BITCAST, DL, VT16Bit, N->getOperand(0));
  SDValue Counts16 = lowerCTPOP16BitElements(Bitcast.getNode(), DAG);
  SDValue N0 = DAG.getNode(ARMISD::VREV32, DL, VT16Bit, Counts16);
  SDValue N1 = DAG.getNode(ISD::ADD, DL, VT16Bit, Counts16, N0);
  SDValue N2 = DAG.getNode(ARMISD::VUZP, DL, VT16Bit, N1, N1);

  if (VT.is64BitVector()) {
    SDValue Extended = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::v4i32, N2);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v2i32, Extended,
                       DAG.getIntPtrConstant(0));
  } else {
    SDValue Extracted = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v4i16, N2,
                                    DAG.getIntPtrConstant(0));
    return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::v4i32, Extracted);
  }
}

static SDValue LowerCTPOP(SDNode *N, SelectionDAG &DAG,
                          const ARMSubtarget *ST) {
  EVT VT = N->getValueType(0);

  assert(ST->hasNEON() && "Custom ctpop lowering requires NEON.");
  assert((VT == MVT::v2i32 || VT == MVT::v4i32 ||
          VT == MVT::v4i16 || VT == MVT::v8i16) &&
         "Unexpected type for custom ctpop lowering");

  if (VT.getVectorElementType() == MVT::i32)
    return lowerCTPOP32BitElements(N, DAG);
  else
    return lowerCTPOP16BitElements(N, DAG);
}

static SDValue LowerShift(SDNode *N, SelectionDAG &DAG,
                          const ARMSubtarget *ST) {
  EVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  if (!VT.isVector())
    return SDValue();

  // Lower vector shifts on NEON to use VSHL.
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

static SDValue Expand64BitShift(SDNode *N, SelectionDAG &DAG,
                                const ARMSubtarget *ST) {
  EVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

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
  Hi = DAG.getNode(Opc, dl, DAG.getVTList(MVT::i32, MVT::Glue), &Hi, 1);

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
    default: llvm_unreachable("Illegal FP comparison");
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
    default: llvm_unreachable("Illegal integer comparison");
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
      if (AndOp.getNode() && AndOp.getOpcode() == ISD::BITCAST)
        AndOp = AndOp.getOperand(0);

      if (AndOp.getNode() && AndOp.getOpcode() == ISD::AND) {
        Opc = ARMISD::VTST;
        Op0 = DAG.getNode(ISD::BITCAST, dl, VT, AndOp.getOperand(0));
        Op1 = DAG.getNode(ISD::BITCAST, dl, VT, AndOp.getOperand(1));
        Invert = !Invert;
      }
    }
  }

  if (Swap)
    std::swap(Op0, Op1);

  // If one of the operands is a constant vector zero, attempt to fold the
  // comparison to a specialized compare-against-zero form.
  SDValue SingleOp;
  if (ISD::isBuildVectorAllZeros(Op1.getNode()))
    SingleOp = Op0;
  else if (ISD::isBuildVectorAllZeros(Op0.getNode())) {
    if (Opc == ARMISD::VCGE)
      Opc = ARMISD::VCLEZ;
    else if (Opc == ARMISD::VCGT)
      Opc = ARMISD::VCLTZ;
    SingleOp = Op1;
  }

  SDValue Result;
  if (SingleOp.getNode()) {
    switch (Opc) {
    case ARMISD::VCEQ:
      Result = DAG.getNode(ARMISD::VCEQZ, dl, VT, SingleOp); break;
    case ARMISD::VCGE:
      Result = DAG.getNode(ARMISD::VCGEZ, dl, VT, SingleOp); break;
    case ARMISD::VCLEZ:
      Result = DAG.getNode(ARMISD::VCLEZ, dl, VT, SingleOp); break;
    case ARMISD::VCGT:
      Result = DAG.getNode(ARMISD::VCGTZ, dl, VT, SingleOp); break;
    case ARMISD::VCLTZ:
      Result = DAG.getNode(ARMISD::VCLTZ, dl, VT, SingleOp); break;
    default:
      Result = DAG.getNode(Opc, dl, VT, Op0, Op1);
    }
  } else {
     Result = DAG.getNode(Opc, dl, VT, Op0, Op1);
  }

  if (Invert)
    Result = DAG.getNOT(dl, Result, VT);

  return Result;
}

/// isNEONModifiedImm - Check if the specified splat value corresponds to a
/// valid vector constant for a NEON instruction with a "modified immediate"
/// operand (e.g., VMOV).  If so, return the encoded value.
static SDValue isNEONModifiedImm(uint64_t SplatBits, uint64_t SplatUndef,
                                 unsigned SplatBitSize, SelectionDAG &DAG,
                                 EVT &VT, bool is128Bits, NEONModImmType type) {
  unsigned OpCmode, Imm;

  // SplatBitSize is set to the smallest size that splats the vector, so a
  // zero vector will always have SplatBitSize == 8.  However, NEON modified
  // immediate instructions others than VMOV do not support the 8-bit encoding
  // of a zero vector, and the default encoding of zero is supposed to be the
  // 32-bit version.
  if (SplatBits == 0)
    SplatBitSize = 32;

  switch (SplatBitSize) {
  case 8:
    if (type != VMOVModImm)
      return SDValue();
    // Any 1-byte value is OK.  Op=0, Cmode=1110.
    assert((SplatBits & ~0xff) == 0 && "one byte splat value is too big");
    OpCmode = 0xe;
    Imm = SplatBits;
    VT = is128Bits ? MVT::v16i8 : MVT::v8i8;
    break;

  case 16:
    // NEON's 16-bit VMOV supports splat values where only one byte is nonzero.
    VT = is128Bits ? MVT::v8i16 : MVT::v4i16;
    if ((SplatBits & ~0xff) == 0) {
      // Value = 0x00nn: Op=x, Cmode=100x.
      OpCmode = 0x8;
      Imm = SplatBits;
      break;
    }
    if ((SplatBits & ~0xff00) == 0) {
      // Value = 0xnn00: Op=x, Cmode=101x.
      OpCmode = 0xa;
      Imm = SplatBits >> 8;
      break;
    }
    return SDValue();

  case 32:
    // NEON's 32-bit VMOV supports splat values where:
    // * only one byte is nonzero, or
    // * the least significant byte is 0xff and the second byte is nonzero, or
    // * the least significant 2 bytes are 0xff and the third is nonzero.
    VT = is128Bits ? MVT::v4i32 : MVT::v2i32;
    if ((SplatBits & ~0xff) == 0) {
      // Value = 0x000000nn: Op=x, Cmode=000x.
      OpCmode = 0;
      Imm = SplatBits;
      break;
    }
    if ((SplatBits & ~0xff00) == 0) {
      // Value = 0x0000nn00: Op=x, Cmode=001x.
      OpCmode = 0x2;
      Imm = SplatBits >> 8;
      break;
    }
    if ((SplatBits & ~0xff0000) == 0) {
      // Value = 0x00nn0000: Op=x, Cmode=010x.
      OpCmode = 0x4;
      Imm = SplatBits >> 16;
      break;
    }
    if ((SplatBits & ~0xff000000) == 0) {
      // Value = 0xnn000000: Op=x, Cmode=011x.
      OpCmode = 0x6;
      Imm = SplatBits >> 24;
      break;
    }

    // cmode == 0b1100 and cmode == 0b1101 are not supported for VORR or VBIC
    if (type == OtherModImm) return SDValue();

    if ((SplatBits & ~0xffff) == 0 &&
        ((SplatBits | SplatUndef) & 0xff) == 0xff) {
      // Value = 0x0000nnff: Op=x, Cmode=1100.
      OpCmode = 0xc;
      Imm = SplatBits >> 8;
      SplatBits |= 0xff;
      break;
    }

    if ((SplatBits & ~0xffffff) == 0 &&
        ((SplatBits | SplatUndef) & 0xffff) == 0xffff) {
      // Value = 0x00nnffff: Op=x, Cmode=1101.
      OpCmode = 0xd;
      Imm = SplatBits >> 16;
      SplatBits |= 0xffff;
      break;
    }

    // Note: there are a few 32-bit splat values (specifically: 00ffff00,
    // ff000000, ff0000ff, and ffff00ff) that are valid for VMOV.I64 but not
    // VMOV.I32.  A (very) minor optimization would be to replicate the value
    // and fall through here to test for a valid 64-bit splat.  But, then the
    // caller would also need to check and handle the change in size.
    return SDValue();

  case 64: {
    if (type != VMOVModImm)
      return SDValue();
    // NEON has a 64-bit VMOV splat where each byte is either 0 or 0xff.
    uint64_t BitMask = 0xff;
    uint64_t Val = 0;
    unsigned ImmMask = 1;
    Imm = 0;
    for (int ByteNum = 0; ByteNum < 8; ++ByteNum) {
      if (((SplatBits | SplatUndef) & BitMask) == BitMask) {
        Val |= BitMask;
        Imm |= ImmMask;
      } else if ((SplatBits & BitMask) != 0) {
        return SDValue();
      }
      BitMask <<= 8;
      ImmMask <<= 1;
    }
    // Op=1, Cmode=1110.
    OpCmode = 0x1e;
    SplatBits = Val;
    VT = is128Bits ? MVT::v2i64 : MVT::v1i64;
    break;
  }

  default:
    llvm_unreachable("unexpected size for isNEONModifiedImm");
  }

  unsigned EncodedVal = ARM_AM::createNEONModImm(OpCmode, Imm);
  return DAG.getTargetConstant(EncodedVal, MVT::i32);
}

SDValue ARMTargetLowering::LowerConstantFP(SDValue Op, SelectionDAG &DAG,
                                           const ARMSubtarget *ST) const {
  if (!ST->useNEONForSinglePrecisionFP() || !ST->hasVFP3() || ST->hasD16())
    return SDValue();

  ConstantFPSDNode *CFP = cast<ConstantFPSDNode>(Op);
  assert(Op.getValueType() == MVT::f32 &&
         "ConstantFP custom lowering should only occur for f32.");

  // Try splatting with a VMOV.f32...
  APFloat FPVal = CFP->getValueAPF();
  int ImmVal = ARM_AM::getFP32Imm(FPVal);
  if (ImmVal != -1) {
    DebugLoc DL = Op.getDebugLoc();
    SDValue NewVal = DAG.getTargetConstant(ImmVal, MVT::i32);
    SDValue VecConstant = DAG.getNode(ARMISD::VMOVFPIMM, DL, MVT::v2f32,
                                      NewVal);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, VecConstant,
                       DAG.getConstant(0, MVT::i32));
  }

  // If that fails, try a VMOV.i32
  EVT VMovVT;
  unsigned iVal = FPVal.bitcastToAPInt().getZExtValue();
  SDValue NewVal = isNEONModifiedImm(iVal, 0, 32, DAG, VMovVT, false,
                                     VMOVModImm);
  if (NewVal != SDValue()) {
    DebugLoc DL = Op.getDebugLoc();
    SDValue VecConstant = DAG.getNode(ARMISD::VMOVIMM, DL, VMovVT,
                                      NewVal);
    SDValue VecFConstant = DAG.getNode(ISD::BITCAST, DL, MVT::v2f32,
                                       VecConstant);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, VecFConstant,
                       DAG.getConstant(0, MVT::i32));
  }

  // Finally, try a VMVN.i32
  NewVal = isNEONModifiedImm(~iVal & 0xffffffff, 0, 32, DAG, VMovVT, false,
                             VMVNModImm);
  if (NewVal != SDValue()) {
    DebugLoc DL = Op.getDebugLoc();
    SDValue VecConstant = DAG.getNode(ARMISD::VMVNIMM, DL, VMovVT, NewVal);
    SDValue VecFConstant = DAG.getNode(ISD::BITCAST, DL, MVT::v2f32,
                                       VecConstant);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, VecFConstant,
                       DAG.getConstant(0, MVT::i32));
  }

  return SDValue();
}

// check if an VEXT instruction can handle the shuffle mask when the
// vector sources of the shuffle are the same.
static bool isSingletonVEXTMask(ArrayRef<int> M, EVT VT, unsigned &Imm) {
  unsigned NumElts = VT.getVectorNumElements();

  // Assume that the first shuffle index is not UNDEF.  Fail if it is.
  if (M[0] < 0)
    return false;

  Imm = M[0];

  // If this is a VEXT shuffle, the immediate value is the index of the first
  // element.  The other shuffle indices must be the successive elements after
  // the first one.
  unsigned ExpectedElt = Imm;
  for (unsigned i = 1; i < NumElts; ++i) {
    // Increment the expected index.  If it wraps around, just follow it
    // back to index zero and keep going.
    ++ExpectedElt;
    if (ExpectedElt == NumElts)
      ExpectedElt = 0;

    if (M[i] < 0) continue; // ignore UNDEF indices
    if (ExpectedElt != static_cast<unsigned>(M[i]))
      return false;
  }

  return true;
}


static bool isVEXTMask(ArrayRef<int> M, EVT VT,
                       bool &ReverseVEXT, unsigned &Imm) {
  unsigned NumElts = VT.getVectorNumElements();
  ReverseVEXT = false;

  // Assume that the first shuffle index is not UNDEF.  Fail if it is.
  if (M[0] < 0)
    return false;

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

    if (M[i] < 0) continue; // ignore UNDEF indices
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
static bool isVREVMask(ArrayRef<int> M, EVT VT, unsigned BlockSize) {
  assert((BlockSize==16 || BlockSize==32 || BlockSize==64) &&
         "Only possible block sizes for VREV are: 16, 32, 64");

  unsigned EltSz = VT.getVectorElementType().getSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  unsigned BlockElts = M[0] + 1;
  // If the first shuffle index is UNDEF, be optimistic.
  if (M[0] < 0)
    BlockElts = BlockSize / EltSz;

  if (BlockSize <= EltSz || BlockSize != BlockElts * EltSz)
    return false;

  for (unsigned i = 0; i < NumElts; ++i) {
    if (M[i] < 0) continue; // ignore UNDEF indices
    if ((unsigned) M[i] != (i - i%BlockElts) + (BlockElts - 1 - i%BlockElts))
      return false;
  }

  return true;
}

static bool isVTBLMask(ArrayRef<int> M, EVT VT) {
  // We can handle <8 x i8> vector shuffles. If the index in the mask is out of
  // range, then 0 is placed into the resulting vector. So pretty much any mask
  // of 8 elements can work here.
  return VT == MVT::v8i8 && M.size() == 8;
}

static bool isVTRNMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned EltSz = VT.getVectorElementType().getSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i < NumElts; i += 2) {
    if ((M[i] >= 0 && (unsigned) M[i] != i + WhichResult) ||
        (M[i+1] >= 0 && (unsigned) M[i+1] != i + NumElts + WhichResult))
      return false;
  }
  return true;
}

/// isVTRN_v_undef_Mask - Special case of isVTRNMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 0, 2, 2> instead of <0, 4, 2, 6>.
static bool isVTRN_v_undef_Mask(ArrayRef<int> M, EVT VT, unsigned &WhichResult){
  unsigned EltSz = VT.getVectorElementType().getSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i < NumElts; i += 2) {
    if ((M[i] >= 0 && (unsigned) M[i] != i + WhichResult) ||
        (M[i+1] >= 0 && (unsigned) M[i+1] != i + WhichResult))
      return false;
  }
  return true;
}

static bool isVUZPMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned EltSz = VT.getVectorElementType().getSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i != NumElts; ++i) {
    if (M[i] < 0) continue; // ignore UNDEF indices
    if ((unsigned) M[i] != 2 * i + WhichResult)
      return false;
  }

  // VUZP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && EltSz == 32)
    return false;

  return true;
}

/// isVUZP_v_undef_Mask - Special case of isVUZPMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 2, 0, 2> instead of <0, 2, 4, 6>,
static bool isVUZP_v_undef_Mask(ArrayRef<int> M, EVT VT, unsigned &WhichResult){
  unsigned EltSz = VT.getVectorElementType().getSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned Half = VT.getVectorNumElements() / 2;
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned j = 0; j != 2; ++j) {
    unsigned Idx = WhichResult;
    for (unsigned i = 0; i != Half; ++i) {
      int MIdx = M[i + j * Half];
      if (MIdx >= 0 && (unsigned) MIdx != Idx)
        return false;
      Idx += 2;
    }
  }

  // VUZP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && EltSz == 32)
    return false;

  return true;
}

static bool isVZIPMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned EltSz = VT.getVectorElementType().getSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  unsigned Idx = WhichResult * NumElts / 2;
  for (unsigned i = 0; i != NumElts; i += 2) {
    if ((M[i] >= 0 && (unsigned) M[i] != Idx) ||
        (M[i+1] >= 0 && (unsigned) M[i+1] != Idx + NumElts))
      return false;
    Idx += 1;
  }

  // VZIP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && EltSz == 32)
    return false;

  return true;
}

/// isVZIP_v_undef_Mask - Special case of isVZIPMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 0, 1, 1> instead of <0, 4, 1, 5>.
static bool isVZIP_v_undef_Mask(ArrayRef<int> M, EVT VT, unsigned &WhichResult){
  unsigned EltSz = VT.getVectorElementType().getSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  unsigned Idx = WhichResult * NumElts / 2;
  for (unsigned i = 0; i != NumElts; i += 2) {
    if ((M[i] >= 0 && (unsigned) M[i] != Idx) ||
        (M[i+1] >= 0 && (unsigned) M[i+1] != Idx))
      return false;
    Idx += 1;
  }

  // VZIP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && EltSz == 32)
    return false;

  return true;
}

/// \return true if this is a reverse operation on an vector.
static bool isReverseMask(ArrayRef<int> M, EVT VT) {
  unsigned NumElts = VT.getVectorNumElements();
  // Make sure the mask has the right size.
  if (NumElts != M.size())
      return false;

  // Look for <15, ..., 3, -1, 1, 0>.
  for (unsigned i = 0; i != NumElts; ++i)
    if (M[i] >= 0 && M[i] != (int) (NumElts - 1 - i))
      return false;

  return true;
}

// If N is an integer constant that can be moved into a register in one
// instruction, return an SDValue of such a constant (will become a MOV
// instruction).  Otherwise return null.
static SDValue IsSingleInstrConstant(SDValue N, SelectionDAG &DAG,
                                     const ARMSubtarget *ST, DebugLoc dl) {
  uint64_t Val;
  if (!isa<ConstantSDNode>(N))
    return SDValue();
  Val = cast<ConstantSDNode>(N)->getZExtValue();

  if (ST->isThumb1Only()) {
    if (Val <= 255 || ~Val <= 255)
      return DAG.getConstant(Val, MVT::i32);
  } else {
    if (ARM_AM::getSOImmVal(Val) != -1 || ARM_AM::getSOImmVal(~Val) != -1)
      return DAG.getConstant(Val, MVT::i32);
  }
  return SDValue();
}

// If this is a case we can't handle, return null and let the default
// expansion code take care of it.
SDValue ARMTargetLowering::LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG,
                                             const ARMSubtarget *ST) const {
  BuildVectorSDNode *BVN = cast<BuildVectorSDNode>(Op.getNode());
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  APInt SplatBits, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  if (BVN->isConstantSplat(SplatBits, SplatUndef, SplatBitSize, HasAnyUndefs)) {
    if (SplatBitSize <= 64) {
      // Check if an immediate VMOV works.
      EVT VmovVT;
      SDValue Val = isNEONModifiedImm(SplatBits.getZExtValue(),
                                      SplatUndef.getZExtValue(), SplatBitSize,
                                      DAG, VmovVT, VT.is128BitVector(),
                                      VMOVModImm);
      if (Val.getNode()) {
        SDValue Vmov = DAG.getNode(ARMISD::VMOVIMM, dl, VmovVT, Val);
        return DAG.getNode(ISD::BITCAST, dl, VT, Vmov);
      }

      // Try an immediate VMVN.
      uint64_t NegatedImm = (~SplatBits).getZExtValue();
      Val = isNEONModifiedImm(NegatedImm,
                                      SplatUndef.getZExtValue(), SplatBitSize,
                                      DAG, VmovVT, VT.is128BitVector(),
                                      VMVNModImm);
      if (Val.getNode()) {
        SDValue Vmov = DAG.getNode(ARMISD::VMVNIMM, dl, VmovVT, Val);
        return DAG.getNode(ISD::BITCAST, dl, VT, Vmov);
      }

      // Use vmov.f32 to materialize other v2f32 and v4f32 splats.
      if ((VT == MVT::v2f32 || VT == MVT::v4f32) && SplatBitSize == 32) {
        int ImmVal = ARM_AM::getFP32Imm(SplatBits);
        if (ImmVal != -1) {
          SDValue Val = DAG.getTargetConstant(ImmVal, MVT::i32);
          return DAG.getNode(ARMISD::VMOVFPIMM, dl, VT, Val);
        }
      }
    }
  }

  // Scan through the operands to see if only one value is used.
  //
  // As an optimisation, even if more than one value is used it may be more
  // profitable to splat with one value then change some lanes.
  //
  // Heuristically we decide to do this if the vector has a "dominant" value,
  // defined as splatted to more than half of the lanes.
  unsigned NumElts = VT.getVectorNumElements();
  bool isOnlyLowElement = true;
  bool usesOnlyOneValue = true;
  bool hasDominantValue = false;
  bool isConstant = true;

  // Map of the number of times a particular SDValue appears in the
  // element list.
  DenseMap<SDValue, unsigned> ValueCounts;
  SDValue Value;
  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue V = Op.getOperand(i);
    if (V.getOpcode() == ISD::UNDEF)
      continue;
    if (i > 0)
      isOnlyLowElement = false;
    if (!isa<ConstantFPSDNode>(V) && !isa<ConstantSDNode>(V))
      isConstant = false;

    ValueCounts.insert(std::make_pair(V, 0));
    unsigned &Count = ValueCounts[V];

    // Is this value dominant? (takes up more than half of the lanes)
    if (++Count > (NumElts / 2)) {
      hasDominantValue = true;
      Value = V;
    }
  }
  if (ValueCounts.size() != 1)
    usesOnlyOneValue = false;
  if (!Value.getNode() && ValueCounts.size() > 0)
    Value = ValueCounts.begin()->first;

  if (ValueCounts.size() == 0)
    return DAG.getUNDEF(VT);

  if (isOnlyLowElement)
    return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Value);

  unsigned EltSize = VT.getVectorElementType().getSizeInBits();

  // Use VDUP for non-constant splats.  For f32 constant splats, reduce to
  // i32 and try again.
  if (hasDominantValue && EltSize <= 32) {
    if (!isConstant) {
      SDValue N;

      // If we are VDUPing a value that comes directly from a vector, that will
      // cause an unnecessary move to and from a GPR, where instead we could
      // just use VDUPLANE. We can only do this if the lane being extracted
      // is at a constant index, as the VDUP from lane instructions only have
      // constant-index forms.
      if (Value->getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
          isa<ConstantSDNode>(Value->getOperand(1))) {
        // We need to create a new undef vector to use for the VDUPLANE if the
        // size of the vector from which we get the value is different than the
        // size of the vector that we need to create. We will insert the element
        // such that the register coalescer will remove unnecessary copies.
        if (VT != Value->getOperand(0).getValueType()) {
          ConstantSDNode *constIndex;
          constIndex = dyn_cast<ConstantSDNode>(Value->getOperand(1));
          assert(constIndex && "The index is not a constant!");
          unsigned index = constIndex->getAPIntValue().getLimitedValue() %
                             VT.getVectorNumElements();
          N =  DAG.getNode(ARMISD::VDUPLANE, dl, VT,
                 DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, DAG.getUNDEF(VT),
                        Value, DAG.getConstant(index, MVT::i32)),
                           DAG.getConstant(index, MVT::i32));
        } else
          N = DAG.getNode(ARMISD::VDUPLANE, dl, VT,
                        Value->getOperand(0), Value->getOperand(1));
      } else
        N = DAG.getNode(ARMISD::VDUP, dl, VT, Value);

      if (!usesOnlyOneValue) {
        // The dominant value was splatted as 'N', but we now have to insert
        // all differing elements.
        for (unsigned I = 0; I < NumElts; ++I) {
          if (Op.getOperand(I) == Value)
            continue;
          SmallVector<SDValue, 3> Ops;
          Ops.push_back(N);
          Ops.push_back(Op.getOperand(I));
          Ops.push_back(DAG.getConstant(I, MVT::i32));
          N = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, &Ops[0], 3);
        }
      }
      return N;
    }
    if (VT.getVectorElementType().isFloatingPoint()) {
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0; i < NumElts; ++i)
        Ops.push_back(DAG.getNode(ISD::BITCAST, dl, MVT::i32,
                                  Op.getOperand(i)));
      EVT VecVT = EVT::getVectorVT(*DAG.getContext(), MVT::i32, NumElts);
      SDValue Val = DAG.getNode(ISD::BUILD_VECTOR, dl, VecVT, &Ops[0], NumElts);
      Val = LowerBUILD_VECTOR(Val, DAG, ST);
      if (Val.getNode())
        return DAG.getNode(ISD::BITCAST, dl, VT, Val);
    }
    if (usesOnlyOneValue) {
      SDValue Val = IsSingleInstrConstant(Value, DAG, ST, dl);
      if (isConstant && Val.getNode())
        return DAG.getNode(ARMISD::VDUP, dl, VT, Val);
    }
  }

  // If all elements are constants and the case above didn't get hit, fall back
  // to the default expansion, which will generate a load from the constant
  // pool.
  if (isConstant)
    return SDValue();

  // Empirical tests suggest this is rarely worth it for vectors of length <= 2.
  if (NumElts >= 4) {
    SDValue shuffle = ReconstructShuffle(Op, DAG);
    if (shuffle != SDValue())
      return shuffle;
  }

  // Vectors with 32- or 64-bit elements can be built by directly assigning
  // the subregisters.  Lower it to an ARMISD::BUILD_VECTOR so the operands
  // will be legalized.
  if (EltSize >= 32) {
    // Do the expansion with floating-point types, since that is what the VFP
    // registers are defined to use, and since i64 is not legal.
    EVT EltVT = EVT::getFloatingPointVT(EltSize);
    EVT VecVT = EVT::getVectorVT(*DAG.getContext(), EltVT, NumElts);
    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0; i < NumElts; ++i)
      Ops.push_back(DAG.getNode(ISD::BITCAST, dl, EltVT, Op.getOperand(i)));
    SDValue Val = DAG.getNode(ARMISD::BUILD_VECTOR, dl, VecVT, &Ops[0],NumElts);
    return DAG.getNode(ISD::BITCAST, dl, VT, Val);
  }

  return SDValue();
}

// Gather data to see if the operation can be modelled as a
// shuffle in combination with VEXTs.
SDValue ARMTargetLowering::ReconstructShuffle(SDValue Op,
                                              SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  unsigned NumElts = VT.getVectorNumElements();

  SmallVector<SDValue, 2> SourceVecs;
  SmallVector<unsigned, 2> MinElts;
  SmallVector<unsigned, 2> MaxElts;

  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue V = Op.getOperand(i);
    if (V.getOpcode() == ISD::UNDEF)
      continue;
    else if (V.getOpcode() != ISD::EXTRACT_VECTOR_ELT) {
      // A shuffle can only come from building a vector from various
      // elements of other vectors.
      return SDValue();
    } else if (V.getOperand(0).getValueType().getVectorElementType() !=
               VT.getVectorElementType()) {
      // This code doesn't know how to handle shuffles where the vector
      // element types do not match (this happens because type legalization
      // promotes the return type of EXTRACT_VECTOR_ELT).
      // FIXME: It might be appropriate to extend this code to handle
      // mismatched types.
      return SDValue();
    }

    // Record this extraction against the appropriate vector if possible...
    SDValue SourceVec = V.getOperand(0);
    // If the element number isn't a constant, we can't effectively
    // analyze what's going on.
    if (!isa<ConstantSDNode>(V.getOperand(1)))
      return SDValue();
    unsigned EltNo = cast<ConstantSDNode>(V.getOperand(1))->getZExtValue();
    bool FoundSource = false;
    for (unsigned j = 0; j < SourceVecs.size(); ++j) {
      if (SourceVecs[j] == SourceVec) {
        if (MinElts[j] > EltNo)
          MinElts[j] = EltNo;
        if (MaxElts[j] < EltNo)
          MaxElts[j] = EltNo;
        FoundSource = true;
        break;
      }
    }

    // Or record a new source if not...
    if (!FoundSource) {
      SourceVecs.push_back(SourceVec);
      MinElts.push_back(EltNo);
      MaxElts.push_back(EltNo);
    }
  }

  // Currently only do something sane when at most two source vectors
  // involved.
  if (SourceVecs.size() > 2)
    return SDValue();

  SDValue ShuffleSrcs[2] = {DAG.getUNDEF(VT), DAG.getUNDEF(VT) };
  int VEXTOffsets[2] = {0, 0};

  // This loop extracts the usage patterns of the source vectors
  // and prepares appropriate SDValues for a shuffle if possible.
  for (unsigned i = 0; i < SourceVecs.size(); ++i) {
    if (SourceVecs[i].getValueType() == VT) {
      // No VEXT necessary
      ShuffleSrcs[i] = SourceVecs[i];
      VEXTOffsets[i] = 0;
      continue;
    } else if (SourceVecs[i].getValueType().getVectorNumElements() < NumElts) {
      // It probably isn't worth padding out a smaller vector just to
      // break it down again in a shuffle.
      return SDValue();
    }

    // Since only 64-bit and 128-bit vectors are legal on ARM and
    // we've eliminated the other cases...
    assert(SourceVecs[i].getValueType().getVectorNumElements() == 2*NumElts &&
           "unexpected vector sizes in ReconstructShuffle");

    if (MaxElts[i] - MinElts[i] >= NumElts) {
      // Span too large for a VEXT to cope
      return SDValue();
    }

    if (MinElts[i] >= NumElts) {
      // The extraction can just take the second half
      VEXTOffsets[i] = NumElts;
      ShuffleSrcs[i] = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT,
                                   SourceVecs[i],
                                   DAG.getIntPtrConstant(NumElts));
    } else if (MaxElts[i] < NumElts) {
      // The extraction can just take the first half
      VEXTOffsets[i] = 0;
      ShuffleSrcs[i] = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT,
                                   SourceVecs[i],
                                   DAG.getIntPtrConstant(0));
    } else {
      // An actual VEXT is needed
      VEXTOffsets[i] = MinElts[i];
      SDValue VEXTSrc1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT,
                                     SourceVecs[i],
                                     DAG.getIntPtrConstant(0));
      SDValue VEXTSrc2 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT,
                                     SourceVecs[i],
                                     DAG.getIntPtrConstant(NumElts));
      ShuffleSrcs[i] = DAG.getNode(ARMISD::VEXT, dl, VT, VEXTSrc1, VEXTSrc2,
                                   DAG.getConstant(VEXTOffsets[i], MVT::i32));
    }
  }

  SmallVector<int, 8> Mask;

  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue Entry = Op.getOperand(i);
    if (Entry.getOpcode() == ISD::UNDEF) {
      Mask.push_back(-1);
      continue;
    }

    SDValue ExtractVec = Entry.getOperand(0);
    int ExtractElt = cast<ConstantSDNode>(Op.getOperand(i)
                                          .getOperand(1))->getSExtValue();
    if (ExtractVec == SourceVecs[0]) {
      Mask.push_back(ExtractElt - VEXTOffsets[0]);
    } else {
      Mask.push_back(ExtractElt + NumElts - VEXTOffsets[1]);
    }
  }

  // Final check before we try to produce nonsense...
  if (isShuffleMaskLegal(Mask, VT))
    return DAG.getVectorShuffle(VT, dl, ShuffleSrcs[0], ShuffleSrcs[1],
                                &Mask[0]);

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

  unsigned EltSize = VT.getVectorElementType().getSizeInBits();
  return (EltSize >= 32 ||
          ShuffleVectorSDNode::isSplatMask(&M[0], VT) ||
          isVREVMask(M, VT, 64) ||
          isVREVMask(M, VT, 32) ||
          isVREVMask(M, VT, 16) ||
          isVEXTMask(M, VT, ReverseVEXT, Imm) ||
          isVTBLMask(M, VT) ||
          isVTRNMask(M, VT, WhichResult) ||
          isVUZPMask(M, VT, WhichResult) ||
          isVZIPMask(M, VT, WhichResult) ||
          isVTRN_v_undef_Mask(M, VT, WhichResult) ||
          isVUZP_v_undef_Mask(M, VT, WhichResult) ||
          isVZIP_v_undef_Mask(M, VT, WhichResult) ||
          ((VT == MVT::v8i16 || VT == MVT::v16i8) && isReverseMask(M, VT)));
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
    // VREV divides the vector in half and swaps within the half.
    if (VT.getVectorElementType() == MVT::i32 ||
        VT.getVectorElementType() == MVT::f32)
      return DAG.getNode(ARMISD::VREV64, dl, VT, OpLHS);
    // vrev <4 x i16> -> VREV32
    if (VT.getVectorElementType() == MVT::i16)
      return DAG.getNode(ARMISD::VREV32, dl, VT, OpLHS);
    // vrev <4 x i8> -> VREV16
    assert(VT.getVectorElementType() == MVT::i8);
    return DAG.getNode(ARMISD::VREV16, dl, VT, OpLHS);
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

static SDValue LowerVECTOR_SHUFFLEv8i8(SDValue Op,
                                       ArrayRef<int> ShuffleMask,
                                       SelectionDAG &DAG) {
  // Check to see if we can use the VTBL instruction.
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  DebugLoc DL = Op.getDebugLoc();

  SmallVector<SDValue, 8> VTBLMask;
  for (ArrayRef<int>::iterator
         I = ShuffleMask.begin(), E = ShuffleMask.end(); I != E; ++I)
    VTBLMask.push_back(DAG.getConstant(*I, MVT::i32));

  if (V2.getNode()->getOpcode() == ISD::UNDEF)
    return DAG.getNode(ARMISD::VTBL1, DL, MVT::v8i8, V1,
                       DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v8i8,
                                   &VTBLMask[0], 8));

  return DAG.getNode(ARMISD::VTBL2, DL, MVT::v8i8, V1, V2,
                     DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v8i8,
                                 &VTBLMask[0], 8));
}

static SDValue LowerReverse_VECTOR_SHUFFLEv16i8_v8i16(SDValue Op,
                                                      SelectionDAG &DAG) {
  DebugLoc DL = Op.getDebugLoc();
  SDValue OpLHS = Op.getOperand(0);
  EVT VT = OpLHS.getValueType();

  assert((VT == MVT::v8i16 || VT == MVT::v16i8) &&
         "Expect an v8i16/v16i8 type");
  OpLHS = DAG.getNode(ARMISD::VREV64, DL, VT, OpLHS);
  // For a v16i8 type: After the VREV, we have got <8, ...15, 8, ..., 0>. Now,
  // extract the first 8 bytes into the top double word and the last 8 bytes
  // into the bottom double word. The v8i16 case is similar.
  unsigned ExtractNum = (VT == MVT::v16i8) ? 8 : 4;
  return DAG.getNode(ARMISD::VEXT, DL, VT, OpLHS, OpLHS,
                     DAG.getConstant(ExtractNum, MVT::i32));
}

static SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) {
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op.getNode());

  // Convert shuffles that are directly supported on NEON to target-specific
  // DAG nodes, instead of keeping them as shuffles and matching them again
  // during code selection.  This is more efficient and avoids the possibility
  // of inconsistencies between legalization and selection.
  // FIXME: floating-point vectors should be canonicalized to integer vectors
  // of the same time so that they get CSEd properly.
  ArrayRef<int> ShuffleMask = SVN->getMask();

  unsigned EltSize = VT.getVectorElementType().getSizeInBits();
  if (EltSize <= 32) {
    if (ShuffleVectorSDNode::isSplatMask(&ShuffleMask[0], VT)) {
      int Lane = SVN->getSplatIndex();
      // If this is undef splat, generate it via "just" vdup, if possible.
      if (Lane == -1) Lane = 0;

      // Test if V1 is a SCALAR_TO_VECTOR.
      if (Lane == 0 && V1.getOpcode() == ISD::SCALAR_TO_VECTOR) {
        return DAG.getNode(ARMISD::VDUP, dl, VT, V1.getOperand(0));
      }
      // Test if V1 is a BUILD_VECTOR which is equivalent to a SCALAR_TO_VECTOR
      // (and probably will turn into a SCALAR_TO_VECTOR once legalization
      // reaches it).
      if (Lane == 0 && V1.getOpcode() == ISD::BUILD_VECTOR &&
          !isa<ConstantSDNode>(V1.getOperand(0))) {
        bool IsScalarToVector = true;
        for (unsigned i = 1, e = V1.getNumOperands(); i != e; ++i)
          if (V1.getOperand(i).getOpcode() != ISD::UNDEF) {
            IsScalarToVector = false;
            break;
          }
        if (IsScalarToVector)
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

    if (V2->getOpcode() == ISD::UNDEF &&
        isSingletonVEXTMask(ShuffleMask, VT, Imm)) {
      return DAG.getNode(ARMISD::VEXT, dl, VT, V1, V1,
                         DAG.getConstant(Imm, MVT::i32));
    }

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

    if (isVTRN_v_undef_Mask(ShuffleMask, VT, WhichResult))
      return DAG.getNode(ARMISD::VTRN, dl, DAG.getVTList(VT, VT),
                         V1, V1).getValue(WhichResult);
    if (isVUZP_v_undef_Mask(ShuffleMask, VT, WhichResult))
      return DAG.getNode(ARMISD::VUZP, dl, DAG.getVTList(VT, VT),
                         V1, V1).getValue(WhichResult);
    if (isVZIP_v_undef_Mask(ShuffleMask, VT, WhichResult))
      return DAG.getNode(ARMISD::VZIP, dl, DAG.getVTList(VT, VT),
                         V1, V1).getValue(WhichResult);
  }

  // If the shuffle is not directly supported and it has 4 elements, use
  // the PerfectShuffle-generated table to synthesize it from other shuffles.
  unsigned NumElts = VT.getVectorNumElements();
  if (NumElts == 4) {
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

  // Implement shuffles with 32- or 64-bit elements as ARMISD::BUILD_VECTORs.
  if (EltSize >= 32) {
    // Do the expansion with floating-point types, since that is what the VFP
    // registers are defined to use, and since i64 is not legal.
    EVT EltVT = EVT::getFloatingPointVT(EltSize);
    EVT VecVT = EVT::getVectorVT(*DAG.getContext(), EltVT, NumElts);
    V1 = DAG.getNode(ISD::BITCAST, dl, VecVT, V1);
    V2 = DAG.getNode(ISD::BITCAST, dl, VecVT, V2);
    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0; i < NumElts; ++i) {
      if (ShuffleMask[i] < 0)
        Ops.push_back(DAG.getUNDEF(EltVT));
      else
        Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT,
                                  ShuffleMask[i] < (int)NumElts ? V1 : V2,
                                  DAG.getConstant(ShuffleMask[i] & (NumElts-1),
                                                  MVT::i32)));
    }
    SDValue Val = DAG.getNode(ARMISD::BUILD_VECTOR, dl, VecVT, &Ops[0],NumElts);
    return DAG.getNode(ISD::BITCAST, dl, VT, Val);
  }

  if ((VT == MVT::v8i16 || VT == MVT::v16i8) && isReverseMask(ShuffleMask, VT))
    return LowerReverse_VECTOR_SHUFFLEv16i8_v8i16(Op, DAG);

  if (VT == MVT::v8i8) {
    SDValue NewOp = LowerVECTOR_SHUFFLEv8i8(Op, ShuffleMask, DAG);
    if (NewOp.getNode())
      return NewOp;
  }

  return SDValue();
}

static SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  // INSERT_VECTOR_ELT is legal only for immediate indexes.
  SDValue Lane = Op.getOperand(2);
  if (!isa<ConstantSDNode>(Lane))
    return SDValue();

  return Op;
}

static SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  // EXTRACT_VECTOR_ELT is legal only for immediate indexes.
  SDValue Lane = Op.getOperand(1);
  if (!isa<ConstantSDNode>(Lane))
    return SDValue();

  SDValue Vec = Op.getOperand(0);
  if (Op.getValueType() == MVT::i32 &&
      Vec.getValueType().getVectorElementType().getSizeInBits() < 32) {
    DebugLoc dl = Op.getDebugLoc();
    return DAG.getNode(ARMISD::VGETLANEu, dl, MVT::i32, Vec, Lane);
  }

  return Op;
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
                      DAG.getNode(ISD::BITCAST, dl, MVT::f64, Op0),
                      DAG.getIntPtrConstant(0));
  if (Op1.getOpcode() != ISD::UNDEF)
    Val = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v2f64, Val,
                      DAG.getNode(ISD::BITCAST, dl, MVT::f64, Op1),
                      DAG.getIntPtrConstant(1));
  return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Val);
}

/// isExtendedBUILD_VECTOR - Check if N is a constant BUILD_VECTOR where each
/// element has been zero/sign-extended, depending on the isSigned parameter,
/// from an integer type half its size.
static bool isExtendedBUILD_VECTOR(SDNode *N, SelectionDAG &DAG,
                                   bool isSigned) {
  // A v2i64 BUILD_VECTOR will have been legalized to a BITCAST from v4i32.
  EVT VT = N->getValueType(0);
  if (VT == MVT::v2i64 && N->getOpcode() == ISD::BITCAST) {
    SDNode *BVN = N->getOperand(0).getNode();
    if (BVN->getValueType(0) != MVT::v4i32 ||
        BVN->getOpcode() != ISD::BUILD_VECTOR)
      return false;
    unsigned LoElt = DAG.getTargetLoweringInfo().isBigEndian() ? 1 : 0;
    unsigned HiElt = 1 - LoElt;
    ConstantSDNode *Lo0 = dyn_cast<ConstantSDNode>(BVN->getOperand(LoElt));
    ConstantSDNode *Hi0 = dyn_cast<ConstantSDNode>(BVN->getOperand(HiElt));
    ConstantSDNode *Lo1 = dyn_cast<ConstantSDNode>(BVN->getOperand(LoElt+2));
    ConstantSDNode *Hi1 = dyn_cast<ConstantSDNode>(BVN->getOperand(HiElt+2));
    if (!Lo0 || !Hi0 || !Lo1 || !Hi1)
      return false;
    if (isSigned) {
      if (Hi0->getSExtValue() == Lo0->getSExtValue() >> 32 &&
          Hi1->getSExtValue() == Lo1->getSExtValue() >> 32)
        return true;
    } else {
      if (Hi0->isNullValue() && Hi1->isNullValue())
        return true;
    }
    return false;
  }

  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDNode *Elt = N->getOperand(i).getNode();
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Elt)) {
      unsigned EltSize = VT.getVectorElementType().getSizeInBits();
      unsigned HalfSize = EltSize / 2;
      if (isSigned) {
        if (!isIntN(HalfSize, C->getSExtValue()))
          return false;
      } else {
        if (!isUIntN(HalfSize, C->getZExtValue()))
          return false;
      }
      continue;
    }
    return false;
  }

  return true;
}

/// isSignExtended - Check if a node is a vector value that is sign-extended
/// or a constant BUILD_VECTOR with sign-extended elements.
static bool isSignExtended(SDNode *N, SelectionDAG &DAG) {
  if (N->getOpcode() == ISD::SIGN_EXTEND || ISD::isSEXTLoad(N))
    return true;
  if (isExtendedBUILD_VECTOR(N, DAG, true))
    return true;
  return false;
}

/// isZeroExtended - Check if a node is a vector value that is zero-extended
/// or a constant BUILD_VECTOR with zero-extended elements.
static bool isZeroExtended(SDNode *N, SelectionDAG &DAG) {
  if (N->getOpcode() == ISD::ZERO_EXTEND || ISD::isZEXTLoad(N))
    return true;
  if (isExtendedBUILD_VECTOR(N, DAG, false))
    return true;
  return false;
}

static EVT getExtensionTo64Bits(const EVT &OrigVT) {
  if (OrigVT.getSizeInBits() >= 64)
    return OrigVT;

  assert(OrigVT.isSimple() && "Expecting a simple value type");

  MVT::SimpleValueType OrigSimpleTy = OrigVT.getSimpleVT().SimpleTy;
  switch (OrigSimpleTy) {
  default: llvm_unreachable("Unexpected Vector Type");
  case MVT::v2i8:
  case MVT::v2i16:
     return MVT::v2i32;
  case MVT::v4i8:
    return  MVT::v4i16;
  }
}

/// AddRequiredExtensionForVMULL - Add a sign/zero extension to extend the total
/// value size to 64 bits. We need a 64-bit D register as an operand to VMULL.
/// We insert the required extension here to get the vector to fill a D register.
static SDValue AddRequiredExtensionForVMULL(SDValue N, SelectionDAG &DAG,
                                            const EVT &OrigTy,
                                            const EVT &ExtTy,
                                            unsigned ExtOpcode) {
  // The vector originally had a size of OrigTy. It was then extended to ExtTy.
  // We expect the ExtTy to be 128-bits total. If the OrigTy is less than
  // 64-bits we need to insert a new extension so that it will be 64-bits.
  assert(ExtTy.is128BitVector() && "Unexpected extension size");
  if (OrigTy.getSizeInBits() >= 64)
    return N;

  // Must extend size to at least 64 bits to be used as an operand for VMULL.
  EVT NewVT = getExtensionTo64Bits(OrigTy);

  return DAG.getNode(ExtOpcode, N->getDebugLoc(), NewVT, N);
}

/// SkipLoadExtensionForVMULL - return a load of the original vector size that
/// does not do any sign/zero extension. If the original vector is less
/// than 64 bits, an appropriate extension will be added after the load to
/// reach a total size of 64 bits. We have to add the extension separately
/// because ARM does not have a sign/zero extending load for vectors.
static SDValue SkipLoadExtensionForVMULL(LoadSDNode *LD, SelectionDAG& DAG) {
  EVT ExtendedTy = getExtensionTo64Bits(LD->getMemoryVT());

  // The load already has the right type.
  if (ExtendedTy == LD->getMemoryVT())
    return DAG.getLoad(LD->getMemoryVT(), LD->getDebugLoc(), LD->getChain(),
                LD->getBasePtr(), LD->getPointerInfo(), LD->isVolatile(),
                LD->isNonTemporal(), LD->isInvariant(),
                LD->getAlignment());

  // We need to create a zextload/sextload. We cannot just create a load
  // followed by a zext/zext node because LowerMUL is also run during normal
  // operation legalization where we can't create illegal types.
  return DAG.getExtLoad(LD->getExtensionType(), LD->getDebugLoc(), ExtendedTy,
                        LD->getChain(), LD->getBasePtr(), LD->getPointerInfo(),
                        LD->getMemoryVT(), LD->isVolatile(),
                        LD->isNonTemporal(), LD->getAlignment());
}

/// SkipExtensionForVMULL - For a node that is a SIGN_EXTEND, ZERO_EXTEND,
/// extending load, or BUILD_VECTOR with extended elements, return the
/// unextended value. The unextended vector should be 64 bits so that it can
/// be used as an operand to a VMULL instruction. If the original vector size
/// before extension is less than 64 bits we add a an extension to resize
/// the vector to 64 bits.
static SDValue SkipExtensionForVMULL(SDNode *N, SelectionDAG &DAG) {
  if (N->getOpcode() == ISD::SIGN_EXTEND || N->getOpcode() == ISD::ZERO_EXTEND)
    return AddRequiredExtensionForVMULL(N->getOperand(0), DAG,
                                        N->getOperand(0)->getValueType(0),
                                        N->getValueType(0),
                                        N->getOpcode());

  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N))
    return SkipLoadExtensionForVMULL(LD, DAG);

  // Otherwise, the value must be a BUILD_VECTOR.  For v2i64, it will
  // have been legalized as a BITCAST from v4i32.
  if (N->getOpcode() == ISD::BITCAST) {
    SDNode *BVN = N->getOperand(0).getNode();
    assert(BVN->getOpcode() == ISD::BUILD_VECTOR &&
           BVN->getValueType(0) == MVT::v4i32 && "expected v4i32 BUILD_VECTOR");
    unsigned LowElt = DAG.getTargetLoweringInfo().isBigEndian() ? 1 : 0;
    return DAG.getNode(ISD::BUILD_VECTOR, N->getDebugLoc(), MVT::v2i32,
                       BVN->getOperand(LowElt), BVN->getOperand(LowElt+2));
  }
  // Construct a new BUILD_VECTOR with elements truncated to half the size.
  assert(N->getOpcode() == ISD::BUILD_VECTOR && "expected BUILD_VECTOR");
  EVT VT = N->getValueType(0);
  unsigned EltSize = VT.getVectorElementType().getSizeInBits() / 2;
  unsigned NumElts = VT.getVectorNumElements();
  MVT TruncVT = MVT::getIntegerVT(EltSize);
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0; i != NumElts; ++i) {
    ConstantSDNode *C = cast<ConstantSDNode>(N->getOperand(i));
    const APInt &CInt = C->getAPIntValue();
    // Element types smaller than 32 bits are not legal, so use i32 elements.
    // The values are implicitly truncated so sext vs. zext doesn't matter.
    Ops.push_back(DAG.getConstant(CInt.zextOrTrunc(32), MVT::i32));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, N->getDebugLoc(),
                     MVT::getVectorVT(TruncVT, NumElts), Ops.data(), NumElts);
}

static bool isAddSubSExt(SDNode *N, SelectionDAG &DAG) {
  unsigned Opcode = N->getOpcode();
  if (Opcode == ISD::ADD || Opcode == ISD::SUB) {
    SDNode *N0 = N->getOperand(0).getNode();
    SDNode *N1 = N->getOperand(1).getNode();
    return N0->hasOneUse() && N1->hasOneUse() &&
      isSignExtended(N0, DAG) && isSignExtended(N1, DAG);
  }
  return false;
}

static bool isAddSubZExt(SDNode *N, SelectionDAG &DAG) {
  unsigned Opcode = N->getOpcode();
  if (Opcode == ISD::ADD || Opcode == ISD::SUB) {
    SDNode *N0 = N->getOperand(0).getNode();
    SDNode *N1 = N->getOperand(1).getNode();
    return N0->hasOneUse() && N1->hasOneUse() &&
      isZeroExtended(N0, DAG) && isZeroExtended(N1, DAG);
  }
  return false;
}

static SDValue LowerMUL(SDValue Op, SelectionDAG &DAG) {
  // Multiplications are only custom-lowered for 128-bit vectors so that
  // VMULL can be detected.  Otherwise v2i64 multiplications are not legal.
  EVT VT = Op.getValueType();
  assert(VT.is128BitVector() && VT.isInteger() &&
         "unexpected type for custom-lowering ISD::MUL");
  SDNode *N0 = Op.getOperand(0).getNode();
  SDNode *N1 = Op.getOperand(1).getNode();
  unsigned NewOpc = 0;
  bool isMLA = false;
  bool isN0SExt = isSignExtended(N0, DAG);
  bool isN1SExt = isSignExtended(N1, DAG);
  if (isN0SExt && isN1SExt)
    NewOpc = ARMISD::VMULLs;
  else {
    bool isN0ZExt = isZeroExtended(N0, DAG);
    bool isN1ZExt = isZeroExtended(N1, DAG);
    if (isN0ZExt && isN1ZExt)
      NewOpc = ARMISD::VMULLu;
    else if (isN1SExt || isN1ZExt) {
      // Look for (s/zext A + s/zext B) * (s/zext C). We want to turn these
      // into (s/zext A * s/zext C) + (s/zext B * s/zext C)
      if (isN1SExt && isAddSubSExt(N0, DAG)) {
        NewOpc = ARMISD::VMULLs;
        isMLA = true;
      } else if (isN1ZExt && isAddSubZExt(N0, DAG)) {
        NewOpc = ARMISD::VMULLu;
        isMLA = true;
      } else if (isN0ZExt && isAddSubZExt(N1, DAG)) {
        std::swap(N0, N1);
        NewOpc = ARMISD::VMULLu;
        isMLA = true;
      }
    }

    if (!NewOpc) {
      if (VT == MVT::v2i64)
        // Fall through to expand this.  It is not legal.
        return SDValue();
      else
        // Other vector multiplications are legal.
        return Op;
    }
  }

  // Legalize to a VMULL instruction.
  DebugLoc DL = Op.getDebugLoc();
  SDValue Op0;
  SDValue Op1 = SkipExtensionForVMULL(N1, DAG);
  if (!isMLA) {
    Op0 = SkipExtensionForVMULL(N0, DAG);
    assert(Op0.getValueType().is64BitVector() &&
           Op1.getValueType().is64BitVector() &&
           "unexpected types for extended operands to VMULL");
    return DAG.getNode(NewOpc, DL, VT, Op0, Op1);
  }

  // Optimizing (zext A + zext B) * C, to (VMULL A, C) + (VMULL B, C) during
  // isel lowering to take advantage of no-stall back to back vmul + vmla.
  //   vmull q0, d4, d6
  //   vmlal q0, d5, d6
  // is faster than
  //   vaddl q0, d4, d5
  //   vmovl q1, d6
  //   vmul  q0, q0, q1
  SDValue N00 = SkipExtensionForVMULL(N0->getOperand(0).getNode(), DAG);
  SDValue N01 = SkipExtensionForVMULL(N0->getOperand(1).getNode(), DAG);
  EVT Op1VT = Op1.getValueType();
  return DAG.getNode(N0->getOpcode(), DL, VT,
                     DAG.getNode(NewOpc, DL, VT,
                               DAG.getNode(ISD::BITCAST, DL, Op1VT, N00), Op1),
                     DAG.getNode(NewOpc, DL, VT,
                               DAG.getNode(ISD::BITCAST, DL, Op1VT, N01), Op1));
}

static SDValue
LowerSDIV_v4i8(SDValue X, SDValue Y, DebugLoc dl, SelectionDAG &DAG) {
  // Convert to float
  // float4 xf = vcvt_f32_s32(vmovl_s16(a.lo));
  // float4 yf = vcvt_f32_s32(vmovl_s16(b.lo));
  X = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::v4i32, X);
  Y = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::v4i32, Y);
  X = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::v4f32, X);
  Y = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::v4f32, Y);
  // Get reciprocal estimate.
  // float4 recip = vrecpeq_f32(yf);
  Y = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::v4f32,
                   DAG.getConstant(Intrinsic::arm_neon_vrecpe, MVT::i32), Y);
  // Because char has a smaller range than uchar, we can actually get away
  // without any newton steps.  This requires that we use a weird bias
  // of 0xb000, however (again, this has been exhaustively tested).
  // float4 result = as_float4(as_int4(xf*recip) + 0xb000);
  X = DAG.getNode(ISD::FMUL, dl, MVT::v4f32, X, Y);
  X = DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, X);
  Y = DAG.getConstant(0xb000, MVT::i32);
  Y = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Y, Y, Y, Y);
  X = DAG.getNode(ISD::ADD, dl, MVT::v4i32, X, Y);
  X = DAG.getNode(ISD::BITCAST, dl, MVT::v4f32, X);
  // Convert back to short.
  X = DAG.getNode(ISD::FP_TO_SINT, dl, MVT::v4i32, X);
  X = DAG.getNode(ISD::TRUNCATE, dl, MVT::v4i16, X);
  return X;
}

static SDValue
LowerSDIV_v4i16(SDValue N0, SDValue N1, DebugLoc dl, SelectionDAG &DAG) {
  SDValue N2;
  // Convert to float.
  // float4 yf = vcvt_f32_s32(vmovl_s16(y));
  // float4 xf = vcvt_f32_s32(vmovl_s16(x));
  N0 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::v4i32, N0);
  N1 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::v4i32, N1);
  N0 = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::v4f32, N0);
  N1 = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::v4f32, N1);

  // Use reciprocal estimate and one refinement step.
  // float4 recip = vrecpeq_f32(yf);
  // recip *= vrecpsq_f32(yf, recip);
  N2 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::v4f32,
                   DAG.getConstant(Intrinsic::arm_neon_vrecpe, MVT::i32), N1);
  N1 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::v4f32,
                   DAG.getConstant(Intrinsic::arm_neon_vrecps, MVT::i32),
                   N1, N2);
  N2 = DAG.getNode(ISD::FMUL, dl, MVT::v4f32, N1, N2);
  // Because short has a smaller range than ushort, we can actually get away
  // with only a single newton step.  This requires that we use a weird bias
  // of 89, however (again, this has been exhaustively tested).
  // float4 result = as_float4(as_int4(xf*recip) + 0x89);
  N0 = DAG.getNode(ISD::FMUL, dl, MVT::v4f32, N0, N2);
  N0 = DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, N0);
  N1 = DAG.getConstant(0x89, MVT::i32);
  N1 = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, N1, N1, N1, N1);
  N0 = DAG.getNode(ISD::ADD, dl, MVT::v4i32, N0, N1);
  N0 = DAG.getNode(ISD::BITCAST, dl, MVT::v4f32, N0);
  // Convert back to integer and return.
  // return vmovn_s32(vcvt_s32_f32(result));
  N0 = DAG.getNode(ISD::FP_TO_SINT, dl, MVT::v4i32, N0);
  N0 = DAG.getNode(ISD::TRUNCATE, dl, MVT::v4i16, N0);
  return N0;
}

static SDValue LowerSDIV(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  assert((VT == MVT::v4i16 || VT == MVT::v8i8) &&
         "unexpected type for custom-lowering ISD::SDIV");

  DebugLoc dl = Op.getDebugLoc();
  SDValue N0 = Op.getOperand(0);
  SDValue N1 = Op.getOperand(1);
  SDValue N2, N3;

  if (VT == MVT::v8i8) {
    N0 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::v8i16, N0);
    N1 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::v8i16, N1);

    N2 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i16, N0,
                     DAG.getIntPtrConstant(4));
    N3 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i16, N1,
                     DAG.getIntPtrConstant(4));
    N0 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i16, N0,
                     DAG.getIntPtrConstant(0));
    N1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i16, N1,
                     DAG.getIntPtrConstant(0));

    N0 = LowerSDIV_v4i8(N0, N1, dl, DAG); // v4i16
    N2 = LowerSDIV_v4i8(N2, N3, dl, DAG); // v4i16

    N0 = DAG.getNode(ISD::CONCAT_VECTORS, dl, MVT::v8i16, N0, N2);
    N0 = LowerCONCAT_VECTORS(N0, DAG);

    N0 = DAG.getNode(ISD::TRUNCATE, dl, MVT::v8i8, N0);
    return N0;
  }
  return LowerSDIV_v4i16(N0, N1, dl, DAG);
}

static SDValue LowerUDIV(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  assert((VT == MVT::v4i16 || VT == MVT::v8i8) &&
         "unexpected type for custom-lowering ISD::UDIV");

  DebugLoc dl = Op.getDebugLoc();
  SDValue N0 = Op.getOperand(0);
  SDValue N1 = Op.getOperand(1);
  SDValue N2, N3;

  if (VT == MVT::v8i8) {
    N0 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::v8i16, N0);
    N1 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::v8i16, N1);

    N2 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i16, N0,
                     DAG.getIntPtrConstant(4));
    N3 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i16, N1,
                     DAG.getIntPtrConstant(4));
    N0 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i16, N0,
                     DAG.getIntPtrConstant(0));
    N1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i16, N1,
                     DAG.getIntPtrConstant(0));

    N0 = LowerSDIV_v4i16(N0, N1, dl, DAG); // v4i16
    N2 = LowerSDIV_v4i16(N2, N3, dl, DAG); // v4i16

    N0 = DAG.getNode(ISD::CONCAT_VECTORS, dl, MVT::v8i16, N0, N2);
    N0 = LowerCONCAT_VECTORS(N0, DAG);

    N0 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::v8i8,
                     DAG.getConstant(Intrinsic::arm_neon_vqmovnsu, MVT::i32),
                     N0);
    return N0;
  }

  // v4i16 sdiv ... Convert to float.
  // float4 yf = vcvt_f32_s32(vmovl_u16(y));
  // float4 xf = vcvt_f32_s32(vmovl_u16(x));
  N0 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::v4i32, N0);
  N1 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::v4i32, N1);
  N0 = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::v4f32, N0);
  SDValue BN1 = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::v4f32, N1);

  // Use reciprocal estimate and two refinement steps.
  // float4 recip = vrecpeq_f32(yf);
  // recip *= vrecpsq_f32(yf, recip);
  // recip *= vrecpsq_f32(yf, recip);
  N2 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::v4f32,
                   DAG.getConstant(Intrinsic::arm_neon_vrecpe, MVT::i32), BN1);
  N1 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::v4f32,
                   DAG.getConstant(Intrinsic::arm_neon_vrecps, MVT::i32),
                   BN1, N2);
  N2 = DAG.getNode(ISD::FMUL, dl, MVT::v4f32, N1, N2);
  N1 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::v4f32,
                   DAG.getConstant(Intrinsic::arm_neon_vrecps, MVT::i32),
                   BN1, N2);
  N2 = DAG.getNode(ISD::FMUL, dl, MVT::v4f32, N1, N2);
  // Simply multiplying by the reciprocal estimate can leave us a few ulps
  // too low, so we add 2 ulps (exhaustive testing shows that this is enough,
  // and that it will never cause us to return an answer too large).
  // float4 result = as_float4(as_int4(xf*recip) + 2);
  N0 = DAG.getNode(ISD::FMUL, dl, MVT::v4f32, N0, N2);
  N0 = DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, N0);
  N1 = DAG.getConstant(2, MVT::i32);
  N1 = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, N1, N1, N1, N1);
  N0 = DAG.getNode(ISD::ADD, dl, MVT::v4i32, N0, N1);
  N0 = DAG.getNode(ISD::BITCAST, dl, MVT::v4f32, N0);
  // Convert back to integer and return.
  // return vmovn_u32(vcvt_s32_f32(result));
  N0 = DAG.getNode(ISD::FP_TO_SINT, dl, MVT::v4i32, N0);
  N0 = DAG.getNode(ISD::TRUNCATE, dl, MVT::v4i16, N0);
  return N0;
}

static SDValue LowerADDC_ADDE_SUBC_SUBE(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getNode()->getValueType(0);
  SDVTList VTs = DAG.getVTList(VT, MVT::i32);

  unsigned Opc;
  bool ExtraOp = false;
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Invalid code");
  case ISD::ADDC: Opc = ARMISD::ADDC; break;
  case ISD::ADDE: Opc = ARMISD::ADDE; ExtraOp = true; break;
  case ISD::SUBC: Opc = ARMISD::SUBC; break;
  case ISD::SUBE: Opc = ARMISD::SUBE; ExtraOp = true; break;
  }

  if (!ExtraOp)
    return DAG.getNode(Opc, Op->getDebugLoc(), VTs, Op.getOperand(0),
                       Op.getOperand(1));
  return DAG.getNode(Opc, Op->getDebugLoc(), VTs, Op.getOperand(0),
                     Op.getOperand(1), Op.getOperand(2));
}

static SDValue LowerAtomicLoadStore(SDValue Op, SelectionDAG &DAG) {
  // Monotonic load/store is legal for all targets
  if (cast<AtomicSDNode>(Op)->getOrdering() <= Monotonic)
    return Op;

  // Aquire/Release load/store is not legal for targets without a
  // dmb or equivalent available.
  return SDValue();
}


static void
ReplaceATOMIC_OP_64(SDNode *Node, SmallVectorImpl<SDValue>& Results,
                    SelectionDAG &DAG, unsigned NewOp) {
  DebugLoc dl = Node->getDebugLoc();
  assert (Node->getValueType(0) == MVT::i64 &&
          "Only know how to expand i64 atomics");

  SmallVector<SDValue, 6> Ops;
  Ops.push_back(Node->getOperand(0)); // Chain
  Ops.push_back(Node->getOperand(1)); // Ptr
  // Low part of Val1
  Ops.push_back(DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                            Node->getOperand(2), DAG.getIntPtrConstant(0)));
  // High part of Val1
  Ops.push_back(DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                            Node->getOperand(2), DAG.getIntPtrConstant(1)));
  if (NewOp == ARMISD::ATOMCMPXCHG64_DAG) {
    // High part of Val1
    Ops.push_back(DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                              Node->getOperand(3), DAG.getIntPtrConstant(0)));
    // High part of Val2
    Ops.push_back(DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                              Node->getOperand(3), DAG.getIntPtrConstant(1)));
  }
  SDVTList Tys = DAG.getVTList(MVT::i32, MVT::i32, MVT::Other);
  SDValue Result =
    DAG.getMemIntrinsicNode(NewOp, dl, Tys, Ops.data(), Ops.size(), MVT::i64,
                            cast<MemSDNode>(Node)->getMemOperand());
  SDValue OpsF[] = { Result.getValue(0), Result.getValue(1) };
  Results.push_back(DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, OpsF, 2));
  Results.push_back(Result.getValue(2));
}

SDValue ARMTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Don't know how to custom lower this!");
  case ISD::ConstantPool:  return LowerConstantPool(Op, DAG);
  case ISD::BlockAddress:  return LowerBlockAddress(Op, DAG);
  case ISD::GlobalAddress:
    return Subtarget->isTargetDarwin() ? LowerGlobalAddressDarwin(Op, DAG) :
      LowerGlobalAddressELF(Op, DAG);
  case ISD::GlobalTLSAddress: return LowerGlobalTLSAddress(Op, DAG);
  case ISD::SELECT:        return LowerSELECT(Op, DAG);
  case ISD::SELECT_CC:     return LowerSELECT_CC(Op, DAG);
  case ISD::BR_CC:         return LowerBR_CC(Op, DAG);
  case ISD::BR_JT:         return LowerBR_JT(Op, DAG);
  case ISD::VASTART:       return LowerVASTART(Op, DAG);
  case ISD::ATOMIC_FENCE:  return LowerATOMIC_FENCE(Op, DAG, Subtarget);
  case ISD::PREFETCH:      return LowerPREFETCH(Op, DAG, Subtarget);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:    return LowerINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:    return LowerFP_TO_INT(Op, DAG);
  case ISD::FCOPYSIGN:     return LowerFCOPYSIGN(Op, DAG);
  case ISD::RETURNADDR:    return LowerRETURNADDR(Op, DAG);
  case ISD::FRAMEADDR:     return LowerFRAMEADDR(Op, DAG);
  case ISD::GLOBAL_OFFSET_TABLE: return LowerGLOBAL_OFFSET_TABLE(Op, DAG);
  case ISD::EH_SJLJ_SETJMP: return LowerEH_SJLJ_SETJMP(Op, DAG);
  case ISD::EH_SJLJ_LONGJMP: return LowerEH_SJLJ_LONGJMP(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG,
                                                               Subtarget);
  case ISD::BITCAST:       return ExpandBITCAST(Op.getNode(), DAG);
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:           return LowerShift(Op.getNode(), DAG, Subtarget);
  case ISD::SHL_PARTS:     return LowerShiftLeftParts(Op, DAG);
  case ISD::SRL_PARTS:
  case ISD::SRA_PARTS:     return LowerShiftRightParts(Op, DAG);
  case ISD::CTTZ:          return LowerCTTZ(Op.getNode(), DAG, Subtarget);
  case ISD::CTPOP:         return LowerCTPOP(Op.getNode(), DAG, Subtarget);
  case ISD::SETCC:         return LowerVSETCC(Op, DAG);
  case ISD::ConstantFP:    return LowerConstantFP(Op, DAG, Subtarget);
  case ISD::BUILD_VECTOR:  return LowerBUILD_VECTOR(Op, DAG, Subtarget);
  case ISD::VECTOR_SHUFFLE: return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::INSERT_VECTOR_ELT: return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT: return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::CONCAT_VECTORS: return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::FLT_ROUNDS_:   return LowerFLT_ROUNDS_(Op, DAG);
  case ISD::MUL:           return LowerMUL(Op, DAG);
  case ISD::SDIV:          return LowerSDIV(Op, DAG);
  case ISD::UDIV:          return LowerUDIV(Op, DAG);
  case ISD::ADDC:
  case ISD::ADDE:
  case ISD::SUBC:
  case ISD::SUBE:          return LowerADDC_ADDE_SUBC_SUBE(Op, DAG);
  case ISD::ATOMIC_LOAD:
  case ISD::ATOMIC_STORE:  return LowerAtomicLoadStore(Op, DAG);
  }
}

/// ReplaceNodeResults - Replace the results of node with an illegal result
/// type with new values built out of custom code.
void ARMTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue>&Results,
                                           SelectionDAG &DAG) const {
  SDValue Res;
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom expand this!");
  case ISD::BITCAST:
    Res = ExpandBITCAST(N, DAG);
    break;
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
    Res = ExpandVectorExtension(N, DAG);
    break;
  case ISD::SRL:
  case ISD::SRA:
    Res = Expand64BitShift(N, DAG, Subtarget);
    break;
  case ISD::ATOMIC_LOAD_ADD:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMADD64_DAG);
    return;
  case ISD::ATOMIC_LOAD_AND:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMAND64_DAG);
    return;
  case ISD::ATOMIC_LOAD_NAND:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMNAND64_DAG);
    return;
  case ISD::ATOMIC_LOAD_OR:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMOR64_DAG);
    return;
  case ISD::ATOMIC_LOAD_SUB:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMSUB64_DAG);
    return;
  case ISD::ATOMIC_LOAD_XOR:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMXOR64_DAG);
    return;
  case ISD::ATOMIC_SWAP:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMSWAP64_DAG);
    return;
  case ISD::ATOMIC_CMP_SWAP:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMCMPXCHG64_DAG);
    return;
  case ISD::ATOMIC_LOAD_MIN:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMMIN64_DAG);
    return;
  case ISD::ATOMIC_LOAD_UMIN:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMUMIN64_DAG);
    return;
  case ISD::ATOMIC_LOAD_MAX:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMMAX64_DAG);
    return;
  case ISD::ATOMIC_LOAD_UMAX:
    ReplaceATOMIC_OP_64(N, Results, DAG, ARMISD::ATOMUMAX64_DAG);
    return;
  }
  if (Res.getNode())
    Results.push_back(Res);
}

//===----------------------------------------------------------------------===//
//                           ARM Scheduler Hooks
//===----------------------------------------------------------------------===//

MachineBasicBlock *
ARMTargetLowering::EmitAtomicCmpSwap(MachineInstr *MI,
                                     MachineBasicBlock *BB,
                                     unsigned Size) const {
  unsigned dest    = MI->getOperand(0).getReg();
  unsigned ptr     = MI->getOperand(1).getReg();
  unsigned oldval  = MI->getOperand(2).getReg();
  unsigned newval  = MI->getOperand(3).getReg();
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  bool isThumb2 = Subtarget->isThumb2();

  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  unsigned scratch = MRI.createVirtualRegister(isThumb2 ?
    (const TargetRegisterClass*)&ARM::rGPRRegClass :
    (const TargetRegisterClass*)&ARM::GPRRegClass);

  if (isThumb2) {
    MRI.constrainRegClass(dest, &ARM::rGPRRegClass);
    MRI.constrainRegClass(oldval, &ARM::rGPRRegClass);
    MRI.constrainRegClass(newval, &ARM::rGPRRegClass);
  }

  unsigned ldrOpc, strOpc;
  switch (Size) {
  default: llvm_unreachable("unsupported size for AtomicCmpSwap!");
  case 1:
    ldrOpc = isThumb2 ? ARM::t2LDREXB : ARM::LDREXB;
    strOpc = isThumb2 ? ARM::t2STREXB : ARM::STREXB;
    break;
  case 2:
    ldrOpc = isThumb2 ? ARM::t2LDREXH : ARM::LDREXH;
    strOpc = isThumb2 ? ARM::t2STREXH : ARM::STREXH;
    break;
  case 4:
    ldrOpc = isThumb2 ? ARM::t2LDREX : ARM::LDREX;
    strOpc = isThumb2 ? ARM::t2STREX : ARM::STREX;
    break;
  }

  MachineFunction *MF = BB->getParent();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = BB;
  ++It; // insert the new blocks after the current block

  MachineBasicBlock *loop1MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *loop2MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, loop1MBB);
  MF->insert(It, loop2MBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  //  thisMBB:
  //   ...
  //   fallthrough --> loop1MBB
  BB->addSuccessor(loop1MBB);

  // loop1MBB:
  //   ldrex dest, [ptr]
  //   cmp dest, oldval
  //   bne exitMBB
  BB = loop1MBB;
  MachineInstrBuilder MIB = BuildMI(BB, dl, TII->get(ldrOpc), dest).addReg(ptr);
  if (ldrOpc == ARM::t2LDREX)
    MIB.addImm(0);
  AddDefaultPred(MIB);
  AddDefaultPred(BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPrr : ARM::CMPrr))
                 .addReg(dest).addReg(oldval));
  BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
    .addMBB(exitMBB).addImm(ARMCC::NE).addReg(ARM::CPSR);
  BB->addSuccessor(loop2MBB);
  BB->addSuccessor(exitMBB);

  // loop2MBB:
  //   strex scratch, newval, [ptr]
  //   cmp scratch, #0
  //   bne loop1MBB
  BB = loop2MBB;
  MIB = BuildMI(BB, dl, TII->get(strOpc), scratch).addReg(newval).addReg(ptr);
  if (strOpc == ARM::t2STREX)
    MIB.addImm(0);
  AddDefaultPred(MIB);
  AddDefaultPred(BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPri : ARM::CMPri))
                 .addReg(scratch).addImm(0));
  BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
    .addMBB(loop1MBB).addImm(ARMCC::NE).addReg(ARM::CPSR);
  BB->addSuccessor(loop1MBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;

  MI->eraseFromParent();   // The instruction is gone now.

  return BB;
}

MachineBasicBlock *
ARMTargetLowering::EmitAtomicBinary(MachineInstr *MI, MachineBasicBlock *BB,
                                    unsigned Size, unsigned BinOpcode) const {
  // This also handles ATOMIC_SWAP, indicated by BinOpcode==0.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction *MF = BB->getParent();
  MachineFunction::iterator It = BB;
  ++It;

  unsigned dest = MI->getOperand(0).getReg();
  unsigned ptr = MI->getOperand(1).getReg();
  unsigned incr = MI->getOperand(2).getReg();
  DebugLoc dl = MI->getDebugLoc();
  bool isThumb2 = Subtarget->isThumb2();

  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  if (isThumb2) {
    MRI.constrainRegClass(dest, &ARM::rGPRRegClass);
    MRI.constrainRegClass(ptr, &ARM::rGPRRegClass);
  }

  unsigned ldrOpc, strOpc;
  switch (Size) {
  default: llvm_unreachable("unsupported size for AtomicCmpSwap!");
  case 1:
    ldrOpc = isThumb2 ? ARM::t2LDREXB : ARM::LDREXB;
    strOpc = isThumb2 ? ARM::t2STREXB : ARM::STREXB;
    break;
  case 2:
    ldrOpc = isThumb2 ? ARM::t2LDREXH : ARM::LDREXH;
    strOpc = isThumb2 ? ARM::t2STREXH : ARM::STREXH;
    break;
  case 4:
    ldrOpc = isThumb2 ? ARM::t2LDREX : ARM::LDREX;
    strOpc = isThumb2 ? ARM::t2STREX : ARM::STREX;
    break;
  }

  MachineBasicBlock *loopMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, loopMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  const TargetRegisterClass *TRC = isThumb2 ?
    (const TargetRegisterClass*)&ARM::rGPRRegClass :
    (const TargetRegisterClass*)&ARM::GPRRegClass;
  unsigned scratch = MRI.createVirtualRegister(TRC);
  unsigned scratch2 = (!BinOpcode) ? incr : MRI.createVirtualRegister(TRC);

  //  thisMBB:
  //   ...
  //   fallthrough --> loopMBB
  BB->addSuccessor(loopMBB);

  //  loopMBB:
  //   ldrex dest, ptr
  //   <binop> scratch2, dest, incr
  //   strex scratch, scratch2, ptr
  //   cmp scratch, #0
  //   bne- loopMBB
  //   fallthrough --> exitMBB
  BB = loopMBB;
  MachineInstrBuilder MIB = BuildMI(BB, dl, TII->get(ldrOpc), dest).addReg(ptr);
  if (ldrOpc == ARM::t2LDREX)
    MIB.addImm(0);
  AddDefaultPred(MIB);
  if (BinOpcode) {
    // operand order needs to go the other way for NAND
    if (BinOpcode == ARM::BICrr || BinOpcode == ARM::t2BICrr)
      AddDefaultPred(BuildMI(BB, dl, TII->get(BinOpcode), scratch2).
                     addReg(incr).addReg(dest)).addReg(0);
    else
      AddDefaultPred(BuildMI(BB, dl, TII->get(BinOpcode), scratch2).
                     addReg(dest).addReg(incr)).addReg(0);
  }

  MIB = BuildMI(BB, dl, TII->get(strOpc), scratch).addReg(scratch2).addReg(ptr);
  if (strOpc == ARM::t2STREX)
    MIB.addImm(0);
  AddDefaultPred(MIB);
  AddDefaultPred(BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPri : ARM::CMPri))
                 .addReg(scratch).addImm(0));
  BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
    .addMBB(loopMBB).addImm(ARMCC::NE).addReg(ARM::CPSR);

  BB->addSuccessor(loopMBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;

  MI->eraseFromParent();   // The instruction is gone now.

  return BB;
}

MachineBasicBlock *
ARMTargetLowering::EmitAtomicBinaryMinMax(MachineInstr *MI,
                                          MachineBasicBlock *BB,
                                          unsigned Size,
                                          bool signExtend,
                                          ARMCC::CondCodes Cond) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction *MF = BB->getParent();
  MachineFunction::iterator It = BB;
  ++It;

  unsigned dest = MI->getOperand(0).getReg();
  unsigned ptr = MI->getOperand(1).getReg();
  unsigned incr = MI->getOperand(2).getReg();
  unsigned oldval = dest;
  DebugLoc dl = MI->getDebugLoc();
  bool isThumb2 = Subtarget->isThumb2();

  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  if (isThumb2) {
    MRI.constrainRegClass(dest, &ARM::rGPRRegClass);
    MRI.constrainRegClass(ptr, &ARM::rGPRRegClass);
  }

  unsigned ldrOpc, strOpc, extendOpc;
  switch (Size) {
  default: llvm_unreachable("unsupported size for AtomicCmpSwap!");
  case 1:
    ldrOpc = isThumb2 ? ARM::t2LDREXB : ARM::LDREXB;
    strOpc = isThumb2 ? ARM::t2STREXB : ARM::STREXB;
    extendOpc = isThumb2 ? ARM::t2SXTB : ARM::SXTB;
    break;
  case 2:
    ldrOpc = isThumb2 ? ARM::t2LDREXH : ARM::LDREXH;
    strOpc = isThumb2 ? ARM::t2STREXH : ARM::STREXH;
    extendOpc = isThumb2 ? ARM::t2SXTH : ARM::SXTH;
    break;
  case 4:
    ldrOpc = isThumb2 ? ARM::t2LDREX : ARM::LDREX;
    strOpc = isThumb2 ? ARM::t2STREX : ARM::STREX;
    extendOpc = 0;
    break;
  }

  MachineBasicBlock *loopMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, loopMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  const TargetRegisterClass *TRC = isThumb2 ?
    (const TargetRegisterClass*)&ARM::rGPRRegClass :
    (const TargetRegisterClass*)&ARM::GPRRegClass;
  unsigned scratch = MRI.createVirtualRegister(TRC);
  unsigned scratch2 = MRI.createVirtualRegister(TRC);

  //  thisMBB:
  //   ...
  //   fallthrough --> loopMBB
  BB->addSuccessor(loopMBB);

  //  loopMBB:
  //   ldrex dest, ptr
  //   (sign extend dest, if required)
  //   cmp dest, incr
  //   cmov.cond scratch2, incr, dest
  //   strex scratch, scratch2, ptr
  //   cmp scratch, #0
  //   bne- loopMBB
  //   fallthrough --> exitMBB
  BB = loopMBB;
  MachineInstrBuilder MIB = BuildMI(BB, dl, TII->get(ldrOpc), dest).addReg(ptr);
  if (ldrOpc == ARM::t2LDREX)
    MIB.addImm(0);
  AddDefaultPred(MIB);

  // Sign extend the value, if necessary.
  if (signExtend && extendOpc) {
    oldval = MRI.createVirtualRegister(&ARM::GPRRegClass);
    AddDefaultPred(BuildMI(BB, dl, TII->get(extendOpc), oldval)
                     .addReg(dest)
                     .addImm(0));
  }

  // Build compare and cmov instructions.
  AddDefaultPred(BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPrr : ARM::CMPrr))
                 .addReg(oldval).addReg(incr));
  BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2MOVCCr : ARM::MOVCCr), scratch2)
         .addReg(incr).addReg(oldval).addImm(Cond).addReg(ARM::CPSR);

  MIB = BuildMI(BB, dl, TII->get(strOpc), scratch).addReg(scratch2).addReg(ptr);
  if (strOpc == ARM::t2STREX)
    MIB.addImm(0);
  AddDefaultPred(MIB);
  AddDefaultPred(BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPri : ARM::CMPri))
                 .addReg(scratch).addImm(0));
  BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
    .addMBB(loopMBB).addImm(ARMCC::NE).addReg(ARM::CPSR);

  BB->addSuccessor(loopMBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;

  MI->eraseFromParent();   // The instruction is gone now.

  return BB;
}

MachineBasicBlock *
ARMTargetLowering::EmitAtomicBinary64(MachineInstr *MI, MachineBasicBlock *BB,
                                      unsigned Op1, unsigned Op2,
                                      bool NeedsCarry, bool IsCmpxchg,
                                      bool IsMinMax, ARMCC::CondCodes CC) const {
  // This also handles ATOMIC_SWAP, indicated by Op1==0.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction *MF = BB->getParent();
  MachineFunction::iterator It = BB;
  ++It;

  unsigned destlo = MI->getOperand(0).getReg();
  unsigned desthi = MI->getOperand(1).getReg();
  unsigned ptr = MI->getOperand(2).getReg();
  unsigned vallo = MI->getOperand(3).getReg();
  unsigned valhi = MI->getOperand(4).getReg();
  DebugLoc dl = MI->getDebugLoc();
  bool isThumb2 = Subtarget->isThumb2();

  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  if (isThumb2) {
    MRI.constrainRegClass(destlo, &ARM::rGPRRegClass);
    MRI.constrainRegClass(desthi, &ARM::rGPRRegClass);
    MRI.constrainRegClass(ptr, &ARM::rGPRRegClass);
  }

  MachineBasicBlock *loopMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *contBB = 0, *cont2BB = 0;
  if (IsCmpxchg || IsMinMax)
    contBB = MF->CreateMachineBasicBlock(LLVM_BB);
  if (IsCmpxchg)
    cont2BB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);

  MF->insert(It, loopMBB);
  if (IsCmpxchg || IsMinMax) MF->insert(It, contBB);
  if (IsCmpxchg) MF->insert(It, cont2BB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  const TargetRegisterClass *TRC = isThumb2 ?
    (const TargetRegisterClass*)&ARM::tGPRRegClass :
    (const TargetRegisterClass*)&ARM::GPRRegClass;
  unsigned storesuccess = MRI.createVirtualRegister(TRC);

  //  thisMBB:
  //   ...
  //   fallthrough --> loopMBB
  BB->addSuccessor(loopMBB);

  //  loopMBB:
  //   ldrexd r2, r3, ptr
  //   <binopa> r0, r2, incr
  //   <binopb> r1, r3, incr
  //   strexd storesuccess, r0, r1, ptr
  //   cmp storesuccess, #0
  //   bne- loopMBB
  //   fallthrough --> exitMBB
  BB = loopMBB;

  // Load
  if (isThumb2) {
    AddDefaultPred(BuildMI(BB, dl, TII->get(ARM::t2LDREXD))
                   .addReg(destlo, RegState::Define)
                   .addReg(desthi, RegState::Define)
                   .addReg(ptr));
  } else {
    unsigned GPRPair0 = MRI.createVirtualRegister(&ARM::GPRPairRegClass);
    AddDefaultPred(BuildMI(BB, dl, TII->get(ARM::LDREXD))
                   .addReg(GPRPair0, RegState::Define).addReg(ptr));
    // Copy r2/r3 into dest.  (This copy will normally be coalesced.)
    BuildMI(BB, dl, TII->get(TargetOpcode::COPY), destlo)
      .addReg(GPRPair0, 0, ARM::gsub_0);
    BuildMI(BB, dl, TII->get(TargetOpcode::COPY), desthi)
      .addReg(GPRPair0, 0, ARM::gsub_1);
  }

  unsigned StoreLo, StoreHi;
  if (IsCmpxchg) {
    // Add early exit
    for (unsigned i = 0; i < 2; i++) {
      AddDefaultPred(BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPrr :
                                                         ARM::CMPrr))
                     .addReg(i == 0 ? destlo : desthi)
                     .addReg(i == 0 ? vallo : valhi));
      BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
        .addMBB(exitMBB).addImm(ARMCC::NE).addReg(ARM::CPSR);
      BB->addSuccessor(exitMBB);
      BB->addSuccessor(i == 0 ? contBB : cont2BB);
      BB = (i == 0 ? contBB : cont2BB);
    }

    // Copy to physregs for strexd
    StoreLo = MI->getOperand(5).getReg();
    StoreHi = MI->getOperand(6).getReg();
  } else if (Op1) {
    // Perform binary operation
    unsigned tmpRegLo = MRI.createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(BB, dl, TII->get(Op1), tmpRegLo)
                   .addReg(destlo).addReg(vallo))
        .addReg(NeedsCarry ? ARM::CPSR : 0, getDefRegState(NeedsCarry));
    unsigned tmpRegHi = MRI.createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(BB, dl, TII->get(Op2), tmpRegHi)
                   .addReg(desthi).addReg(valhi))
        .addReg(IsMinMax ? ARM::CPSR : 0, getDefRegState(IsMinMax));

    StoreLo = tmpRegLo;
    StoreHi = tmpRegHi;
  } else {
    // Copy to physregs for strexd
    StoreLo = vallo;
    StoreHi = valhi;
  }
  if (IsMinMax) {
    // Compare and branch to exit block.
    BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
      .addMBB(exitMBB).addImm(CC).addReg(ARM::CPSR);
    BB->addSuccessor(exitMBB);
    BB->addSuccessor(contBB);
    BB = contBB;
    StoreLo = vallo;
    StoreHi = valhi;
  }

  // Store
  if (isThumb2) {
    AddDefaultPred(BuildMI(BB, dl, TII->get(ARM::t2STREXD), storesuccess)
                   .addReg(StoreLo).addReg(StoreHi).addReg(ptr));
  } else {
    // Marshal a pair...
    unsigned StorePair = MRI.createVirtualRegister(&ARM::GPRPairRegClass);
    unsigned UndefPair = MRI.createVirtualRegister(&ARM::GPRPairRegClass);
    unsigned r1 = MRI.createVirtualRegister(&ARM::GPRPairRegClass);
    BuildMI(BB, dl, TII->get(TargetOpcode::IMPLICIT_DEF), UndefPair);
    BuildMI(BB, dl, TII->get(TargetOpcode::INSERT_SUBREG), r1)
      .addReg(UndefPair)
      .addReg(StoreLo)
      .addImm(ARM::gsub_0);
    BuildMI(BB, dl, TII->get(TargetOpcode::INSERT_SUBREG), StorePair)
      .addReg(r1)
      .addReg(StoreHi)
      .addImm(ARM::gsub_1);

    // ...and store it
    AddDefaultPred(BuildMI(BB, dl, TII->get(ARM::STREXD), storesuccess)
                   .addReg(StorePair).addReg(ptr));
  }
  // Cmp+jump
  AddDefaultPred(BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPri : ARM::CMPri))
                 .addReg(storesuccess).addImm(0));
  BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
    .addMBB(loopMBB).addImm(ARMCC::NE).addReg(ARM::CPSR);

  BB->addSuccessor(loopMBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;

  MI->eraseFromParent();   // The instruction is gone now.

  return BB;
}

/// SetupEntryBlockForSjLj - Insert code into the entry block that creates and
/// registers the function context.
void ARMTargetLowering::
SetupEntryBlockForSjLj(MachineInstr *MI, MachineBasicBlock *MBB,
                       MachineBasicBlock *DispatchBB, int FI) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo *MRI = &MF->getRegInfo();
  MachineConstantPool *MCP = MF->getConstantPool();
  ARMFunctionInfo *AFI = MF->getInfo<ARMFunctionInfo>();
  const Function *F = MF->getFunction();

  bool isThumb = Subtarget->isThumb();
  bool isThumb2 = Subtarget->isThumb2();

  unsigned PCLabelId = AFI->createPICLabelUId();
  unsigned PCAdj = (isThumb || isThumb2) ? 4 : 8;
  ARMConstantPoolValue *CPV =
    ARMConstantPoolMBB::Create(F->getContext(), DispatchBB, PCLabelId, PCAdj);
  unsigned CPI = MCP->getConstantPoolIndex(CPV, 4);

  const TargetRegisterClass *TRC = isThumb ?
    (const TargetRegisterClass*)&ARM::tGPRRegClass :
    (const TargetRegisterClass*)&ARM::GPRRegClass;

  // Grab constant pool and fixed stack memory operands.
  MachineMemOperand *CPMMO =
    MF->getMachineMemOperand(MachinePointerInfo::getConstantPool(),
                             MachineMemOperand::MOLoad, 4, 4);

  MachineMemOperand *FIMMOSt =
    MF->getMachineMemOperand(MachinePointerInfo::getFixedStack(FI),
                             MachineMemOperand::MOStore, 4, 4);

  // Load the address of the dispatch MBB into the jump buffer.
  if (isThumb2) {
    // Incoming value: jbuf
    //   ldr.n  r5, LCPI1_1
    //   orr    r5, r5, #1
    //   add    r5, pc
    //   str    r5, [$jbuf, #+4] ; &jbuf[1]
    unsigned NewVReg1 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::t2LDRpci), NewVReg1)
                   .addConstantPoolIndex(CPI)
                   .addMemOperand(CPMMO));
    // Set the low bit because of thumb mode.
    unsigned NewVReg2 = MRI->createVirtualRegister(TRC);
    AddDefaultCC(
      AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::t2ORRri), NewVReg2)
                     .addReg(NewVReg1, RegState::Kill)
                     .addImm(0x01)));
    unsigned NewVReg3 = MRI->createVirtualRegister(TRC);
    BuildMI(*MBB, MI, dl, TII->get(ARM::tPICADD), NewVReg3)
      .addReg(NewVReg2, RegState::Kill)
      .addImm(PCLabelId);
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::t2STRi12))
                   .addReg(NewVReg3, RegState::Kill)
                   .addFrameIndex(FI)
                   .addImm(36)  // &jbuf[1] :: pc
                   .addMemOperand(FIMMOSt));
  } else if (isThumb) {
    // Incoming value: jbuf
    //   ldr.n  r1, LCPI1_4
    //   add    r1, pc
    //   mov    r2, #1
    //   orrs   r1, r2
    //   add    r2, $jbuf, #+4 ; &jbuf[1]
    //   str    r1, [r2]
    unsigned NewVReg1 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::tLDRpci), NewVReg1)
                   .addConstantPoolIndex(CPI)
                   .addMemOperand(CPMMO));
    unsigned NewVReg2 = MRI->createVirtualRegister(TRC);
    BuildMI(*MBB, MI, dl, TII->get(ARM::tPICADD), NewVReg2)
      .addReg(NewVReg1, RegState::Kill)
      .addImm(PCLabelId);
    // Set the low bit because of thumb mode.
    unsigned NewVReg3 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::tMOVi8), NewVReg3)
                   .addReg(ARM::CPSR, RegState::Define)
                   .addImm(1));
    unsigned NewVReg4 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::tORR), NewVReg4)
                   .addReg(ARM::CPSR, RegState::Define)
                   .addReg(NewVReg2, RegState::Kill)
                   .addReg(NewVReg3, RegState::Kill));
    unsigned NewVReg5 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::tADDrSPi), NewVReg5)
                   .addFrameIndex(FI)
                   .addImm(36)); // &jbuf[1] :: pc
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::tSTRi))
                   .addReg(NewVReg4, RegState::Kill)
                   .addReg(NewVReg5, RegState::Kill)
                   .addImm(0)
                   .addMemOperand(FIMMOSt));
  } else {
    // Incoming value: jbuf
    //   ldr  r1, LCPI1_1
    //   add  r1, pc, r1
    //   str  r1, [$jbuf, #+4] ; &jbuf[1]
    unsigned NewVReg1 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::LDRi12),  NewVReg1)
                   .addConstantPoolIndex(CPI)
                   .addImm(0)
                   .addMemOperand(CPMMO));
    unsigned NewVReg2 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::PICADD), NewVReg2)
                   .addReg(NewVReg1, RegState::Kill)
                   .addImm(PCLabelId));
    AddDefaultPred(BuildMI(*MBB, MI, dl, TII->get(ARM::STRi12))
                   .addReg(NewVReg2, RegState::Kill)
                   .addFrameIndex(FI)
                   .addImm(36)  // &jbuf[1] :: pc
                   .addMemOperand(FIMMOSt));
  }
}

MachineBasicBlock *ARMTargetLowering::
EmitSjLjDispatchBlock(MachineInstr *MI, MachineBasicBlock *MBB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo *MRI = &MF->getRegInfo();
  ARMFunctionInfo *AFI = MF->getInfo<ARMFunctionInfo>();
  MachineFrameInfo *MFI = MF->getFrameInfo();
  int FI = MFI->getFunctionContextIndex();

  const TargetRegisterClass *TRC = Subtarget->isThumb() ?
    (const TargetRegisterClass*)&ARM::tGPRRegClass :
    (const TargetRegisterClass*)&ARM::GPRnopcRegClass;

  // Get a mapping of the call site numbers to all of the landing pads they're
  // associated with.
  DenseMap<unsigned, SmallVector<MachineBasicBlock*, 2> > CallSiteNumToLPad;
  unsigned MaxCSNum = 0;
  MachineModuleInfo &MMI = MF->getMMI();
  for (MachineFunction::iterator BB = MF->begin(), E = MF->end(); BB != E;
       ++BB) {
    if (!BB->isLandingPad()) continue;

    // FIXME: We should assert that the EH_LABEL is the first MI in the landing
    // pad.
    for (MachineBasicBlock::iterator
           II = BB->begin(), IE = BB->end(); II != IE; ++II) {
      if (!II->isEHLabel()) continue;

      MCSymbol *Sym = II->getOperand(0).getMCSymbol();
      if (!MMI.hasCallSiteLandingPad(Sym)) continue;

      SmallVectorImpl<unsigned> &CallSiteIdxs = MMI.getCallSiteLandingPad(Sym);
      for (SmallVectorImpl<unsigned>::iterator
             CSI = CallSiteIdxs.begin(), CSE = CallSiteIdxs.end();
           CSI != CSE; ++CSI) {
        CallSiteNumToLPad[*CSI].push_back(BB);
        MaxCSNum = std::max(MaxCSNum, *CSI);
      }
      break;
    }
  }

  // Get an ordered list of the machine basic blocks for the jump table.
  std::vector<MachineBasicBlock*> LPadList;
  SmallPtrSet<MachineBasicBlock*, 64> InvokeBBs;
  LPadList.reserve(CallSiteNumToLPad.size());
  for (unsigned I = 1; I <= MaxCSNum; ++I) {
    SmallVectorImpl<MachineBasicBlock*> &MBBList = CallSiteNumToLPad[I];
    for (SmallVectorImpl<MachineBasicBlock*>::iterator
           II = MBBList.begin(), IE = MBBList.end(); II != IE; ++II) {
      LPadList.push_back(*II);
      InvokeBBs.insert((*II)->pred_begin(), (*II)->pred_end());
    }
  }

  assert(!LPadList.empty() &&
         "No landing pad destinations for the dispatch jump table!");

  // Create the jump table and associated information.
  MachineJumpTableInfo *JTI =
    MF->getOrCreateJumpTableInfo(MachineJumpTableInfo::EK_Inline);
  unsigned MJTI = JTI->createJumpTableIndex(LPadList);
  unsigned UId = AFI->createJumpTableUId();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();

  // Create the MBBs for the dispatch code.

  // Shove the dispatch's address into the return slot in the function context.
  MachineBasicBlock *DispatchBB = MF->CreateMachineBasicBlock();
  DispatchBB->setIsLandingPad();

  MachineBasicBlock *TrapBB = MF->CreateMachineBasicBlock();
  unsigned trap_opcode;
  if (Subtarget->isThumb())
    trap_opcode = ARM::tTRAP;
  else
    trap_opcode = Subtarget->useNaClTrap() ? ARM::TRAPNaCl : ARM::TRAP;

  BuildMI(TrapBB, dl, TII->get(trap_opcode));
  DispatchBB->addSuccessor(TrapBB);

  MachineBasicBlock *DispContBB = MF->CreateMachineBasicBlock();
  DispatchBB->addSuccessor(DispContBB);

  // Insert and MBBs.
  MF->insert(MF->end(), DispatchBB);
  MF->insert(MF->end(), DispContBB);
  MF->insert(MF->end(), TrapBB);

  // Insert code into the entry block that creates and registers the function
  // context.
  SetupEntryBlockForSjLj(MI, MBB, DispatchBB, FI);

  MachineMemOperand *FIMMOLd =
    MF->getMachineMemOperand(MachinePointerInfo::getFixedStack(FI),
                             MachineMemOperand::MOLoad |
                             MachineMemOperand::MOVolatile, 4, 4);

  MachineInstrBuilder MIB;
  MIB = BuildMI(DispatchBB, dl, TII->get(ARM::Int_eh_sjlj_dispatchsetup));

  const ARMBaseInstrInfo *AII = static_cast<const ARMBaseInstrInfo*>(TII);
  const ARMBaseRegisterInfo &RI = AII->getRegisterInfo();

  // Add a register mask with no preserved registers.  This results in all
  // registers being marked as clobbered.
  MIB.addRegMask(RI.getNoPreservedMask());

  unsigned NumLPads = LPadList.size();
  if (Subtarget->isThumb2()) {
    unsigned NewVReg1 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::t2LDRi12), NewVReg1)
                   .addFrameIndex(FI)
                   .addImm(4)
                   .addMemOperand(FIMMOLd));

    if (NumLPads < 256) {
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::t2CMPri))
                     .addReg(NewVReg1)
                     .addImm(LPadList.size()));
    } else {
      unsigned VReg1 = MRI->createVirtualRegister(TRC);
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::t2MOVi16), VReg1)
                     .addImm(NumLPads & 0xFFFF));

      unsigned VReg2 = VReg1;
      if ((NumLPads & 0xFFFF0000) != 0) {
        VReg2 = MRI->createVirtualRegister(TRC);
        AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::t2MOVTi16), VReg2)
                       .addReg(VReg1)
                       .addImm(NumLPads >> 16));
      }

      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::t2CMPrr))
                     .addReg(NewVReg1)
                     .addReg(VReg2));
    }

    BuildMI(DispatchBB, dl, TII->get(ARM::t2Bcc))
      .addMBB(TrapBB)
      .addImm(ARMCC::HI)
      .addReg(ARM::CPSR);

    unsigned NewVReg3 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispContBB, dl, TII->get(ARM::t2LEApcrelJT),NewVReg3)
                   .addJumpTableIndex(MJTI)
                   .addImm(UId));

    unsigned NewVReg4 = MRI->createVirtualRegister(TRC);
    AddDefaultCC(
      AddDefaultPred(
        BuildMI(DispContBB, dl, TII->get(ARM::t2ADDrs), NewVReg4)
        .addReg(NewVReg3, RegState::Kill)
        .addReg(NewVReg1)
        .addImm(ARM_AM::getSORegOpc(ARM_AM::lsl, 2))));

    BuildMI(DispContBB, dl, TII->get(ARM::t2BR_JT))
      .addReg(NewVReg4, RegState::Kill)
      .addReg(NewVReg1)
      .addJumpTableIndex(MJTI)
      .addImm(UId);
  } else if (Subtarget->isThumb()) {
    unsigned NewVReg1 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::tLDRspi), NewVReg1)
                   .addFrameIndex(FI)
                   .addImm(1)
                   .addMemOperand(FIMMOLd));

    if (NumLPads < 256) {
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::tCMPi8))
                     .addReg(NewVReg1)
                     .addImm(NumLPads));
    } else {
      MachineConstantPool *ConstantPool = MF->getConstantPool();
      Type *Int32Ty = Type::getInt32Ty(MF->getFunction()->getContext());
      const Constant *C = ConstantInt::get(Int32Ty, NumLPads);

      // MachineConstantPool wants an explicit alignment.
      unsigned Align = getDataLayout()->getPrefTypeAlignment(Int32Ty);
      if (Align == 0)
        Align = getDataLayout()->getTypeAllocSize(C->getType());
      unsigned Idx = ConstantPool->getConstantPoolIndex(C, Align);

      unsigned VReg1 = MRI->createVirtualRegister(TRC);
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::tLDRpci))
                     .addReg(VReg1, RegState::Define)
                     .addConstantPoolIndex(Idx));
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::tCMPr))
                     .addReg(NewVReg1)
                     .addReg(VReg1));
    }

    BuildMI(DispatchBB, dl, TII->get(ARM::tBcc))
      .addMBB(TrapBB)
      .addImm(ARMCC::HI)
      .addReg(ARM::CPSR);

    unsigned NewVReg2 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispContBB, dl, TII->get(ARM::tLSLri), NewVReg2)
                   .addReg(ARM::CPSR, RegState::Define)
                   .addReg(NewVReg1)
                   .addImm(2));

    unsigned NewVReg3 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispContBB, dl, TII->get(ARM::tLEApcrelJT), NewVReg3)
                   .addJumpTableIndex(MJTI)
                   .addImm(UId));

    unsigned NewVReg4 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispContBB, dl, TII->get(ARM::tADDrr), NewVReg4)
                   .addReg(ARM::CPSR, RegState::Define)
                   .addReg(NewVReg2, RegState::Kill)
                   .addReg(NewVReg3));

    MachineMemOperand *JTMMOLd =
      MF->getMachineMemOperand(MachinePointerInfo::getJumpTable(),
                               MachineMemOperand::MOLoad, 4, 4);

    unsigned NewVReg5 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispContBB, dl, TII->get(ARM::tLDRi), NewVReg5)
                   .addReg(NewVReg4, RegState::Kill)
                   .addImm(0)
                   .addMemOperand(JTMMOLd));

    unsigned NewVReg6 = NewVReg5;
    if (RelocM == Reloc::PIC_) {
      NewVReg6 = MRI->createVirtualRegister(TRC);
      AddDefaultPred(BuildMI(DispContBB, dl, TII->get(ARM::tADDrr), NewVReg6)
                     .addReg(ARM::CPSR, RegState::Define)
                     .addReg(NewVReg5, RegState::Kill)
                     .addReg(NewVReg3));
    }

    BuildMI(DispContBB, dl, TII->get(ARM::tBR_JTr))
      .addReg(NewVReg6, RegState::Kill)
      .addJumpTableIndex(MJTI)
      .addImm(UId);
  } else {
    unsigned NewVReg1 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::LDRi12), NewVReg1)
                   .addFrameIndex(FI)
                   .addImm(4)
                   .addMemOperand(FIMMOLd));

    if (NumLPads < 256) {
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::CMPri))
                     .addReg(NewVReg1)
                     .addImm(NumLPads));
    } else if (Subtarget->hasV6T2Ops() && isUInt<16>(NumLPads)) {
      unsigned VReg1 = MRI->createVirtualRegister(TRC);
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::MOVi16), VReg1)
                     .addImm(NumLPads & 0xFFFF));

      unsigned VReg2 = VReg1;
      if ((NumLPads & 0xFFFF0000) != 0) {
        VReg2 = MRI->createVirtualRegister(TRC);
        AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::MOVTi16), VReg2)
                       .addReg(VReg1)
                       .addImm(NumLPads >> 16));
      }

      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::CMPrr))
                     .addReg(NewVReg1)
                     .addReg(VReg2));
    } else {
      MachineConstantPool *ConstantPool = MF->getConstantPool();
      Type *Int32Ty = Type::getInt32Ty(MF->getFunction()->getContext());
      const Constant *C = ConstantInt::get(Int32Ty, NumLPads);

      // MachineConstantPool wants an explicit alignment.
      unsigned Align = getDataLayout()->getPrefTypeAlignment(Int32Ty);
      if (Align == 0)
        Align = getDataLayout()->getTypeAllocSize(C->getType());
      unsigned Idx = ConstantPool->getConstantPoolIndex(C, Align);

      unsigned VReg1 = MRI->createVirtualRegister(TRC);
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::LDRcp))
                     .addReg(VReg1, RegState::Define)
                     .addConstantPoolIndex(Idx)
                     .addImm(0));
      AddDefaultPred(BuildMI(DispatchBB, dl, TII->get(ARM::CMPrr))
                     .addReg(NewVReg1)
                     .addReg(VReg1, RegState::Kill));
    }

    BuildMI(DispatchBB, dl, TII->get(ARM::Bcc))
      .addMBB(TrapBB)
      .addImm(ARMCC::HI)
      .addReg(ARM::CPSR);

    unsigned NewVReg3 = MRI->createVirtualRegister(TRC);
    AddDefaultCC(
      AddDefaultPred(BuildMI(DispContBB, dl, TII->get(ARM::MOVsi), NewVReg3)
                     .addReg(NewVReg1)
                     .addImm(ARM_AM::getSORegOpc(ARM_AM::lsl, 2))));
    unsigned NewVReg4 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(DispContBB, dl, TII->get(ARM::LEApcrelJT), NewVReg4)
                   .addJumpTableIndex(MJTI)
                   .addImm(UId));

    MachineMemOperand *JTMMOLd =
      MF->getMachineMemOperand(MachinePointerInfo::getJumpTable(),
                               MachineMemOperand::MOLoad, 4, 4);
    unsigned NewVReg5 = MRI->createVirtualRegister(TRC);
    AddDefaultPred(
      BuildMI(DispContBB, dl, TII->get(ARM::LDRrs), NewVReg5)
      .addReg(NewVReg3, RegState::Kill)
      .addReg(NewVReg4)
      .addImm(0)
      .addMemOperand(JTMMOLd));

    if (RelocM == Reloc::PIC_) {
      BuildMI(DispContBB, dl, TII->get(ARM::BR_JTadd))
        .addReg(NewVReg5, RegState::Kill)
        .addReg(NewVReg4)
        .addJumpTableIndex(MJTI)
        .addImm(UId);
    } else {
      BuildMI(DispContBB, dl, TII->get(ARM::BR_JTr))
        .addReg(NewVReg5, RegState::Kill)
        .addJumpTableIndex(MJTI)
        .addImm(UId);
    }
  }

  // Add the jump table entries as successors to the MBB.
  SmallPtrSet<MachineBasicBlock*, 8> SeenMBBs;
  for (std::vector<MachineBasicBlock*>::iterator
         I = LPadList.begin(), E = LPadList.end(); I != E; ++I) {
    MachineBasicBlock *CurMBB = *I;
    if (SeenMBBs.insert(CurMBB))
      DispContBB->addSuccessor(CurMBB);
  }

  // N.B. the order the invoke BBs are processed in doesn't matter here.
  const uint16_t *SavedRegs = RI.getCalleeSavedRegs(MF);
  SmallVector<MachineBasicBlock*, 64> MBBLPads;
  for (SmallPtrSet<MachineBasicBlock*, 64>::iterator
         I = InvokeBBs.begin(), E = InvokeBBs.end(); I != E; ++I) {
    MachineBasicBlock *BB = *I;

    // Remove the landing pad successor from the invoke block and replace it
    // with the new dispatch block.
    SmallVector<MachineBasicBlock*, 4> Successors(BB->succ_begin(),
                                                  BB->succ_end());
    while (!Successors.empty()) {
      MachineBasicBlock *SMBB = Successors.pop_back_val();
      if (SMBB->isLandingPad()) {
        BB->removeSuccessor(SMBB);
        MBBLPads.push_back(SMBB);
      }
    }

    BB->addSuccessor(DispatchBB);

    // Find the invoke call and mark all of the callee-saved registers as
    // 'implicit defined' so that they're spilled. This prevents code from
    // moving instructions to before the EH block, where they will never be
    // executed.
    for (MachineBasicBlock::reverse_iterator
           II = BB->rbegin(), IE = BB->rend(); II != IE; ++II) {
      if (!II->isCall()) continue;

      DenseMap<unsigned, bool> DefRegs;
      for (MachineInstr::mop_iterator
             OI = II->operands_begin(), OE = II->operands_end();
           OI != OE; ++OI) {
        if (!OI->isReg()) continue;
        DefRegs[OI->getReg()] = true;
      }

      MachineInstrBuilder MIB(*MF, &*II);

      for (unsigned i = 0; SavedRegs[i] != 0; ++i) {
        unsigned Reg = SavedRegs[i];
        if (Subtarget->isThumb2() &&
            !ARM::tGPRRegClass.contains(Reg) &&
            !ARM::hGPRRegClass.contains(Reg))
          continue;
        if (Subtarget->isThumb1Only() && !ARM::tGPRRegClass.contains(Reg))
          continue;
        if (!Subtarget->isThumb() && !ARM::GPRRegClass.contains(Reg))
          continue;
        if (!DefRegs[Reg])
          MIB.addReg(Reg, RegState::ImplicitDefine | RegState::Dead);
      }

      break;
    }
  }

  // Mark all former landing pads as non-landing pads. The dispatch is the only
  // landing pad now.
  for (SmallVectorImpl<MachineBasicBlock*>::iterator
         I = MBBLPads.begin(), E = MBBLPads.end(); I != E; ++I)
    (*I)->setIsLandingPad(false);

  // The instruction is gone now.
  MI->eraseFromParent();

  return MBB;
}

static
MachineBasicBlock *OtherSucc(MachineBasicBlock *MBB, MachineBasicBlock *Succ) {
  for (MachineBasicBlock::succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I)
    if (*I != Succ)
      return *I;
  llvm_unreachable("Expecting a BB with two successors!");
}

MachineBasicBlock *ARMTargetLowering::
EmitStructByval(MachineInstr *MI, MachineBasicBlock *BB) const {
  // This pseudo instruction has 3 operands: dst, src, size
  // We expand it to a loop if size > Subtarget->getMaxInlineSizeThreshold().
  // Otherwise, we will generate unrolled scalar copies.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = BB;
  ++It;

  unsigned dest = MI->getOperand(0).getReg();
  unsigned src = MI->getOperand(1).getReg();
  unsigned SizeVal = MI->getOperand(2).getImm();
  unsigned Align = MI->getOperand(3).getImm();
  DebugLoc dl = MI->getDebugLoc();

  bool isThumb2 = Subtarget->isThumb2();
  MachineFunction *MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  unsigned ldrOpc, strOpc, UnitSize = 0;

  const TargetRegisterClass *TRC = isThumb2 ?
    (const TargetRegisterClass*)&ARM::tGPRRegClass :
    (const TargetRegisterClass*)&ARM::GPRRegClass;
  const TargetRegisterClass *TRC_Vec = 0;

  if (Align & 1) {
    ldrOpc = isThumb2 ? ARM::t2LDRB_POST : ARM::LDRB_POST_IMM;
    strOpc = isThumb2 ? ARM::t2STRB_POST : ARM::STRB_POST_IMM;
    UnitSize = 1;
  } else if (Align & 2) {
    ldrOpc = isThumb2 ? ARM::t2LDRH_POST : ARM::LDRH_POST;
    strOpc = isThumb2 ? ARM::t2STRH_POST : ARM::STRH_POST;
    UnitSize = 2;
  } else {
    // Check whether we can use NEON instructions.
    if (!MF->getFunction()->getAttributes().
          hasAttribute(AttributeSet::FunctionIndex,
                       Attribute::NoImplicitFloat) &&
        Subtarget->hasNEON()) {
      if ((Align % 16 == 0) && SizeVal >= 16) {
        ldrOpc = ARM::VLD1q32wb_fixed;
        strOpc = ARM::VST1q32wb_fixed;
        UnitSize = 16;
        TRC_Vec = (const TargetRegisterClass*)&ARM::DPairRegClass;
      }
      else if ((Align % 8 == 0) && SizeVal >= 8) {
        ldrOpc = ARM::VLD1d32wb_fixed;
        strOpc = ARM::VST1d32wb_fixed;
        UnitSize = 8;
        TRC_Vec = (const TargetRegisterClass*)&ARM::DPRRegClass;
      }
    }
    // Can't use NEON instructions.
    if (UnitSize == 0) {
      ldrOpc = isThumb2 ? ARM::t2LDR_POST : ARM::LDR_POST_IMM;
      strOpc = isThumb2 ? ARM::t2STR_POST : ARM::STR_POST_IMM;
      UnitSize = 4;
    }
  }

  unsigned BytesLeft = SizeVal % UnitSize;
  unsigned LoopSize = SizeVal - BytesLeft;

  if (SizeVal <= Subtarget->getMaxInlineSizeThreshold()) {
    // Use LDR and STR to copy.
    // [scratch, srcOut] = LDR_POST(srcIn, UnitSize)
    // [destOut] = STR_POST(scratch, destIn, UnitSize)
    unsigned srcIn = src;
    unsigned destIn = dest;
    for (unsigned i = 0; i < LoopSize; i+=UnitSize) {
      unsigned scratch = MRI.createVirtualRegister(UnitSize >= 8 ? TRC_Vec:TRC);
      unsigned srcOut = MRI.createVirtualRegister(TRC);
      unsigned destOut = MRI.createVirtualRegister(TRC);
      if (UnitSize >= 8) {
        AddDefaultPred(BuildMI(*BB, MI, dl,
          TII->get(ldrOpc), scratch)
          .addReg(srcOut, RegState::Define).addReg(srcIn).addImm(0));

        AddDefaultPred(BuildMI(*BB, MI, dl, TII->get(strOpc), destOut)
          .addReg(destIn).addImm(0).addReg(scratch));
      } else if (isThumb2) {
        AddDefaultPred(BuildMI(*BB, MI, dl,
          TII->get(ldrOpc), scratch)
          .addReg(srcOut, RegState::Define).addReg(srcIn).addImm(UnitSize));

        AddDefaultPred(BuildMI(*BB, MI, dl, TII->get(strOpc), destOut)
          .addReg(scratch).addReg(destIn)
          .addImm(UnitSize));
      } else {
        AddDefaultPred(BuildMI(*BB, MI, dl,
          TII->get(ldrOpc), scratch)
          .addReg(srcOut, RegState::Define).addReg(srcIn).addReg(0)
          .addImm(UnitSize));

        AddDefaultPred(BuildMI(*BB, MI, dl, TII->get(strOpc), destOut)
          .addReg(scratch).addReg(destIn)
          .addReg(0).addImm(UnitSize));
      }
      srcIn = srcOut;
      destIn = destOut;
    }

    // Handle the leftover bytes with LDRB and STRB.
    // [scratch, srcOut] = LDRB_POST(srcIn, 1)
    // [destOut] = STRB_POST(scratch, destIn, 1)
    ldrOpc = isThumb2 ? ARM::t2LDRB_POST : ARM::LDRB_POST_IMM;
    strOpc = isThumb2 ? ARM::t2STRB_POST : ARM::STRB_POST_IMM;
    for (unsigned i = 0; i < BytesLeft; i++) {
      unsigned scratch = MRI.createVirtualRegister(TRC);
      unsigned srcOut = MRI.createVirtualRegister(TRC);
      unsigned destOut = MRI.createVirtualRegister(TRC);
      if (isThumb2) {
        AddDefaultPred(BuildMI(*BB, MI, dl,
          TII->get(ldrOpc),scratch)
          .addReg(srcOut, RegState::Define).addReg(srcIn).addImm(1));

        AddDefaultPred(BuildMI(*BB, MI, dl, TII->get(strOpc), destOut)
          .addReg(scratch).addReg(destIn)
          .addReg(0).addImm(1));
      } else {
        AddDefaultPred(BuildMI(*BB, MI, dl,
          TII->get(ldrOpc),scratch)
          .addReg(srcOut, RegState::Define).addReg(srcIn)
          .addReg(0).addImm(1));

        AddDefaultPred(BuildMI(*BB, MI, dl, TII->get(strOpc), destOut)
          .addReg(scratch).addReg(destIn)
          .addReg(0).addImm(1));
      }
      srcIn = srcOut;
      destIn = destOut;
    }
    MI->eraseFromParent();   // The instruction is gone now.
    return BB;
  }

  // Expand the pseudo op to a loop.
  // thisMBB:
  //   ...
  //   movw varEnd, # --> with thumb2
  //   movt varEnd, #
  //   ldrcp varEnd, idx --> without thumb2
  //   fallthrough --> loopMBB
  // loopMBB:
  //   PHI varPhi, varEnd, varLoop
  //   PHI srcPhi, src, srcLoop
  //   PHI destPhi, dst, destLoop
  //   [scratch, srcLoop] = LDR_POST(srcPhi, UnitSize)
  //   [destLoop] = STR_POST(scratch, destPhi, UnitSize)
  //   subs varLoop, varPhi, #UnitSize
  //   bne loopMBB
  //   fallthrough --> exitMBB
  // exitMBB:
  //   epilogue to handle left-over bytes
  //   [scratch, srcOut] = LDRB_POST(srcLoop, 1)
  //   [destOut] = STRB_POST(scratch, destLoop, 1)
  MachineBasicBlock *loopMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, loopMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Load an immediate to varEnd.
  unsigned varEnd = MRI.createVirtualRegister(TRC);
  if (isThumb2) {
    unsigned VReg1 = varEnd;
    if ((LoopSize & 0xFFFF0000) != 0)
      VReg1 = MRI.createVirtualRegister(TRC);
    AddDefaultPred(BuildMI(BB, dl, TII->get(ARM::t2MOVi16), VReg1)
                   .addImm(LoopSize & 0xFFFF));

    if ((LoopSize & 0xFFFF0000) != 0)
      AddDefaultPred(BuildMI(BB, dl, TII->get(ARM::t2MOVTi16), varEnd)
                     .addReg(VReg1)
                     .addImm(LoopSize >> 16));
  } else {
    MachineConstantPool *ConstantPool = MF->getConstantPool();
    Type *Int32Ty = Type::getInt32Ty(MF->getFunction()->getContext());
    const Constant *C = ConstantInt::get(Int32Ty, LoopSize);

    // MachineConstantPool wants an explicit alignment.
    unsigned Align = getDataLayout()->getPrefTypeAlignment(Int32Ty);
    if (Align == 0)
      Align = getDataLayout()->getTypeAllocSize(C->getType());
    unsigned Idx = ConstantPool->getConstantPoolIndex(C, Align);

    AddDefaultPred(BuildMI(BB, dl, TII->get(ARM::LDRcp))
                   .addReg(varEnd, RegState::Define)
                   .addConstantPoolIndex(Idx)
                   .addImm(0));
  }
  BB->addSuccessor(loopMBB);

  // Generate the loop body:
  //   varPhi = PHI(varLoop, varEnd)
  //   srcPhi = PHI(srcLoop, src)
  //   destPhi = PHI(destLoop, dst)
  MachineBasicBlock *entryBB = BB;
  BB = loopMBB;
  unsigned varLoop = MRI.createVirtualRegister(TRC);
  unsigned varPhi = MRI.createVirtualRegister(TRC);
  unsigned srcLoop = MRI.createVirtualRegister(TRC);
  unsigned srcPhi = MRI.createVirtualRegister(TRC);
  unsigned destLoop = MRI.createVirtualRegister(TRC);
  unsigned destPhi = MRI.createVirtualRegister(TRC);

  BuildMI(*BB, BB->begin(), dl, TII->get(ARM::PHI), varPhi)
    .addReg(varLoop).addMBB(loopMBB)
    .addReg(varEnd).addMBB(entryBB);
  BuildMI(BB, dl, TII->get(ARM::PHI), srcPhi)
    .addReg(srcLoop).addMBB(loopMBB)
    .addReg(src).addMBB(entryBB);
  BuildMI(BB, dl, TII->get(ARM::PHI), destPhi)
    .addReg(destLoop).addMBB(loopMBB)
    .addReg(dest).addMBB(entryBB);

  //   [scratch, srcLoop] = LDR_POST(srcPhi, UnitSize)
  //   [destLoop] = STR_POST(scratch, destPhi, UnitSiz)
  unsigned scratch = MRI.createVirtualRegister(UnitSize >= 8 ? TRC_Vec:TRC);
  if (UnitSize >= 8) {
    AddDefaultPred(BuildMI(BB, dl, TII->get(ldrOpc), scratch)
      .addReg(srcLoop, RegState::Define).addReg(srcPhi).addImm(0));

    AddDefaultPred(BuildMI(BB, dl, TII->get(strOpc), destLoop)
      .addReg(destPhi).addImm(0).addReg(scratch));
  } else if (isThumb2) {
    AddDefaultPred(BuildMI(BB, dl, TII->get(ldrOpc), scratch)
      .addReg(srcLoop, RegState::Define).addReg(srcPhi).addImm(UnitSize));

    AddDefaultPred(BuildMI(BB, dl, TII->get(strOpc), destLoop)
      .addReg(scratch).addReg(destPhi)
      .addImm(UnitSize));
  } else {
    AddDefaultPred(BuildMI(BB, dl, TII->get(ldrOpc), scratch)
      .addReg(srcLoop, RegState::Define).addReg(srcPhi).addReg(0)
      .addImm(UnitSize));

    AddDefaultPred(BuildMI(BB, dl, TII->get(strOpc), destLoop)
      .addReg(scratch).addReg(destPhi)
      .addReg(0).addImm(UnitSize));
  }

  // Decrement loop variable by UnitSize.
  MachineInstrBuilder MIB = BuildMI(BB, dl,
    TII->get(isThumb2 ? ARM::t2SUBri : ARM::SUBri), varLoop);
  AddDefaultCC(AddDefaultPred(MIB.addReg(varPhi).addImm(UnitSize)));
  MIB->getOperand(5).setReg(ARM::CPSR);
  MIB->getOperand(5).setIsDef(true);

  BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
    .addMBB(loopMBB).addImm(ARMCC::NE).addReg(ARM::CPSR);

  // loopMBB can loop back to loopMBB or fall through to exitMBB.
  BB->addSuccessor(loopMBB);
  BB->addSuccessor(exitMBB);

  // Add epilogue to handle BytesLeft.
  BB = exitMBB;
  MachineInstr *StartOfExit = exitMBB->begin();
  ldrOpc = isThumb2 ? ARM::t2LDRB_POST : ARM::LDRB_POST_IMM;
  strOpc = isThumb2 ? ARM::t2STRB_POST : ARM::STRB_POST_IMM;

  //   [scratch, srcOut] = LDRB_POST(srcLoop, 1)
  //   [destOut] = STRB_POST(scratch, destLoop, 1)
  unsigned srcIn = srcLoop;
  unsigned destIn = destLoop;
  for (unsigned i = 0; i < BytesLeft; i++) {
    unsigned scratch = MRI.createVirtualRegister(TRC);
    unsigned srcOut = MRI.createVirtualRegister(TRC);
    unsigned destOut = MRI.createVirtualRegister(TRC);
    if (isThumb2) {
      AddDefaultPred(BuildMI(*BB, StartOfExit, dl,
        TII->get(ldrOpc),scratch)
        .addReg(srcOut, RegState::Define).addReg(srcIn).addImm(1));

      AddDefaultPred(BuildMI(*BB, StartOfExit, dl, TII->get(strOpc), destOut)
        .addReg(scratch).addReg(destIn)
        .addImm(1));
    } else {
      AddDefaultPred(BuildMI(*BB, StartOfExit, dl,
        TII->get(ldrOpc),scratch)
        .addReg(srcOut, RegState::Define).addReg(srcIn).addReg(0).addImm(1));

      AddDefaultPred(BuildMI(*BB, StartOfExit, dl, TII->get(strOpc), destOut)
        .addReg(scratch).addReg(destIn)
        .addReg(0).addImm(1));
    }
    srcIn = srcOut;
    destIn = destOut;
  }

  MI->eraseFromParent();   // The instruction is gone now.
  return BB;
}

MachineBasicBlock *
ARMTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                               MachineBasicBlock *BB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  bool isThumb2 = Subtarget->isThumb2();
  switch (MI->getOpcode()) {
  default: {
    MI->dump();
    llvm_unreachable("Unexpected instr type to insert");
  }
  // The Thumb2 pre-indexed stores have the same MI operands, they just
  // define them differently in the .td files from the isel patterns, so
  // they need pseudos.
  case ARM::t2STR_preidx:
    MI->setDesc(TII->get(ARM::t2STR_PRE));
    return BB;
  case ARM::t2STRB_preidx:
    MI->setDesc(TII->get(ARM::t2STRB_PRE));
    return BB;
  case ARM::t2STRH_preidx:
    MI->setDesc(TII->get(ARM::t2STRH_PRE));
    return BB;

  case ARM::STRi_preidx:
  case ARM::STRBi_preidx: {
    unsigned NewOpc = MI->getOpcode() == ARM::STRi_preidx ?
      ARM::STR_PRE_IMM : ARM::STRB_PRE_IMM;
    // Decode the offset.
    unsigned Offset = MI->getOperand(4).getImm();
    bool isSub = ARM_AM::getAM2Op(Offset) == ARM_AM::sub;
    Offset = ARM_AM::getAM2Offset(Offset);
    if (isSub)
      Offset = -Offset;

    MachineMemOperand *MMO = *MI->memoperands_begin();
    BuildMI(*BB, MI, dl, TII->get(NewOpc))
      .addOperand(MI->getOperand(0))  // Rn_wb
      .addOperand(MI->getOperand(1))  // Rt
      .addOperand(MI->getOperand(2))  // Rn
      .addImm(Offset)                 // offset (skip GPR==zero_reg)
      .addOperand(MI->getOperand(5))  // pred
      .addOperand(MI->getOperand(6))
      .addMemOperand(MMO);
    MI->eraseFromParent();
    return BB;
  }
  case ARM::STRr_preidx:
  case ARM::STRBr_preidx:
  case ARM::STRH_preidx: {
    unsigned NewOpc;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("unexpected opcode!");
    case ARM::STRr_preidx: NewOpc = ARM::STR_PRE_REG; break;
    case ARM::STRBr_preidx: NewOpc = ARM::STRB_PRE_REG; break;
    case ARM::STRH_preidx: NewOpc = ARM::STRH_PRE; break;
    }
    MachineInstrBuilder MIB = BuildMI(*BB, MI, dl, TII->get(NewOpc));
    for (unsigned i = 0; i < MI->getNumOperands(); ++i)
      MIB.addOperand(MI->getOperand(i));
    MI->eraseFromParent();
    return BB;
  }
  case ARM::ATOMIC_LOAD_ADD_I8:
     return EmitAtomicBinary(MI, BB, 1, isThumb2 ? ARM::t2ADDrr : ARM::ADDrr);
  case ARM::ATOMIC_LOAD_ADD_I16:
     return EmitAtomicBinary(MI, BB, 2, isThumb2 ? ARM::t2ADDrr : ARM::ADDrr);
  case ARM::ATOMIC_LOAD_ADD_I32:
     return EmitAtomicBinary(MI, BB, 4, isThumb2 ? ARM::t2ADDrr : ARM::ADDrr);

  case ARM::ATOMIC_LOAD_AND_I8:
     return EmitAtomicBinary(MI, BB, 1, isThumb2 ? ARM::t2ANDrr : ARM::ANDrr);
  case ARM::ATOMIC_LOAD_AND_I16:
     return EmitAtomicBinary(MI, BB, 2, isThumb2 ? ARM::t2ANDrr : ARM::ANDrr);
  case ARM::ATOMIC_LOAD_AND_I32:
     return EmitAtomicBinary(MI, BB, 4, isThumb2 ? ARM::t2ANDrr : ARM::ANDrr);

  case ARM::ATOMIC_LOAD_OR_I8:
     return EmitAtomicBinary(MI, BB, 1, isThumb2 ? ARM::t2ORRrr : ARM::ORRrr);
  case ARM::ATOMIC_LOAD_OR_I16:
     return EmitAtomicBinary(MI, BB, 2, isThumb2 ? ARM::t2ORRrr : ARM::ORRrr);
  case ARM::ATOMIC_LOAD_OR_I32:
     return EmitAtomicBinary(MI, BB, 4, isThumb2 ? ARM::t2ORRrr : ARM::ORRrr);

  case ARM::ATOMIC_LOAD_XOR_I8:
     return EmitAtomicBinary(MI, BB, 1, isThumb2 ? ARM::t2EORrr : ARM::EORrr);
  case ARM::ATOMIC_LOAD_XOR_I16:
     return EmitAtomicBinary(MI, BB, 2, isThumb2 ? ARM::t2EORrr : ARM::EORrr);
  case ARM::ATOMIC_LOAD_XOR_I32:
     return EmitAtomicBinary(MI, BB, 4, isThumb2 ? ARM::t2EORrr : ARM::EORrr);

  case ARM::ATOMIC_LOAD_NAND_I8:
     return EmitAtomicBinary(MI, BB, 1, isThumb2 ? ARM::t2BICrr : ARM::BICrr);
  case ARM::ATOMIC_LOAD_NAND_I16:
     return EmitAtomicBinary(MI, BB, 2, isThumb2 ? ARM::t2BICrr : ARM::BICrr);
  case ARM::ATOMIC_LOAD_NAND_I32:
     return EmitAtomicBinary(MI, BB, 4, isThumb2 ? ARM::t2BICrr : ARM::BICrr);

  case ARM::ATOMIC_LOAD_SUB_I8:
     return EmitAtomicBinary(MI, BB, 1, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr);
  case ARM::ATOMIC_LOAD_SUB_I16:
     return EmitAtomicBinary(MI, BB, 2, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr);
  case ARM::ATOMIC_LOAD_SUB_I32:
     return EmitAtomicBinary(MI, BB, 4, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr);

  case ARM::ATOMIC_LOAD_MIN_I8:
     return EmitAtomicBinaryMinMax(MI, BB, 1, true, ARMCC::LT);
  case ARM::ATOMIC_LOAD_MIN_I16:
     return EmitAtomicBinaryMinMax(MI, BB, 2, true, ARMCC::LT);
  case ARM::ATOMIC_LOAD_MIN_I32:
     return EmitAtomicBinaryMinMax(MI, BB, 4, true, ARMCC::LT);

  case ARM::ATOMIC_LOAD_MAX_I8:
     return EmitAtomicBinaryMinMax(MI, BB, 1, true, ARMCC::GT);
  case ARM::ATOMIC_LOAD_MAX_I16:
     return EmitAtomicBinaryMinMax(MI, BB, 2, true, ARMCC::GT);
  case ARM::ATOMIC_LOAD_MAX_I32:
     return EmitAtomicBinaryMinMax(MI, BB, 4, true, ARMCC::GT);

  case ARM::ATOMIC_LOAD_UMIN_I8:
     return EmitAtomicBinaryMinMax(MI, BB, 1, false, ARMCC::LO);
  case ARM::ATOMIC_LOAD_UMIN_I16:
     return EmitAtomicBinaryMinMax(MI, BB, 2, false, ARMCC::LO);
  case ARM::ATOMIC_LOAD_UMIN_I32:
     return EmitAtomicBinaryMinMax(MI, BB, 4, false, ARMCC::LO);

  case ARM::ATOMIC_LOAD_UMAX_I8:
     return EmitAtomicBinaryMinMax(MI, BB, 1, false, ARMCC::HI);
  case ARM::ATOMIC_LOAD_UMAX_I16:
     return EmitAtomicBinaryMinMax(MI, BB, 2, false, ARMCC::HI);
  case ARM::ATOMIC_LOAD_UMAX_I32:
     return EmitAtomicBinaryMinMax(MI, BB, 4, false, ARMCC::HI);

  case ARM::ATOMIC_SWAP_I8:  return EmitAtomicBinary(MI, BB, 1, 0);
  case ARM::ATOMIC_SWAP_I16: return EmitAtomicBinary(MI, BB, 2, 0);
  case ARM::ATOMIC_SWAP_I32: return EmitAtomicBinary(MI, BB, 4, 0);

  case ARM::ATOMIC_CMP_SWAP_I8:  return EmitAtomicCmpSwap(MI, BB, 1);
  case ARM::ATOMIC_CMP_SWAP_I16: return EmitAtomicCmpSwap(MI, BB, 2);
  case ARM::ATOMIC_CMP_SWAP_I32: return EmitAtomicCmpSwap(MI, BB, 4);


  case ARM::ATOMADD6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2ADDrr : ARM::ADDrr,
                              isThumb2 ? ARM::t2ADCrr : ARM::ADCrr,
                              /*NeedsCarry*/ true);
  case ARM::ATOMSUB6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr,
                              isThumb2 ? ARM::t2SBCrr : ARM::SBCrr,
                              /*NeedsCarry*/ true);
  case ARM::ATOMOR6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2ORRrr : ARM::ORRrr,
                              isThumb2 ? ARM::t2ORRrr : ARM::ORRrr);
  case ARM::ATOMXOR6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2EORrr : ARM::EORrr,
                              isThumb2 ? ARM::t2EORrr : ARM::EORrr);
  case ARM::ATOMAND6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2ANDrr : ARM::ANDrr,
                              isThumb2 ? ARM::t2ANDrr : ARM::ANDrr);
  case ARM::ATOMSWAP6432:
    return EmitAtomicBinary64(MI, BB, 0, 0, false);
  case ARM::ATOMCMPXCHG6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr,
                              isThumb2 ? ARM::t2SBCrr : ARM::SBCrr,
                              /*NeedsCarry*/ false, /*IsCmpxchg*/true);
  case ARM::ATOMMIN6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr,
                              isThumb2 ? ARM::t2SBCrr : ARM::SBCrr,
                              /*NeedsCarry*/ true, /*IsCmpxchg*/false,
                              /*IsMinMax*/ true, ARMCC::LT);
  case ARM::ATOMMAX6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr,
                              isThumb2 ? ARM::t2SBCrr : ARM::SBCrr,
                              /*NeedsCarry*/ true, /*IsCmpxchg*/false,
                              /*IsMinMax*/ true, ARMCC::GE);
  case ARM::ATOMUMIN6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr,
                              isThumb2 ? ARM::t2SBCrr : ARM::SBCrr,
                              /*NeedsCarry*/ true, /*IsCmpxchg*/false,
                              /*IsMinMax*/ true, ARMCC::LO);
  case ARM::ATOMUMAX6432:
    return EmitAtomicBinary64(MI, BB, isThumb2 ? ARM::t2SUBrr : ARM::SUBrr,
                              isThumb2 ? ARM::t2SBCrr : ARM::SBCrr,
                              /*NeedsCarry*/ true, /*IsCmpxchg*/false,
                              /*IsMinMax*/ true, ARMCC::HS);

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
    F->insert(It, copy0MBB);
    F->insert(It, sinkMBB);

    // Transfer the remainder of BB and its successor edges to sinkMBB.
    sinkMBB->splice(sinkMBB->begin(), BB,
                    llvm::next(MachineBasicBlock::iterator(MI)),
                    BB->end());
    sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

    BB->addSuccessor(copy0MBB);
    BB->addSuccessor(sinkMBB);

    BuildMI(BB, dl, TII->get(ARM::tBcc)).addMBB(sinkMBB)
      .addImm(MI->getOperand(3).getImm()).addReg(MI->getOperand(4).getReg());

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
    BuildMI(*BB, BB->begin(), dl,
            TII->get(ARM::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

    MI->eraseFromParent();   // The pseudo instruction is gone now.
    return BB;
  }

  case ARM::BCCi64:
  case ARM::BCCZi64: {
    // If there is an unconditional branch to the other successor, remove it.
    BB->erase(llvm::next(MachineBasicBlock::iterator(MI)), BB->end());

    // Compare both parts that make up the double comparison separately for
    // equality.
    bool RHSisZero = MI->getOpcode() == ARM::BCCZi64;

    unsigned LHS1 = MI->getOperand(1).getReg();
    unsigned LHS2 = MI->getOperand(2).getReg();
    if (RHSisZero) {
      AddDefaultPred(BuildMI(BB, dl,
                             TII->get(isThumb2 ? ARM::t2CMPri : ARM::CMPri))
                     .addReg(LHS1).addImm(0));
      BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPri : ARM::CMPri))
        .addReg(LHS2).addImm(0)
        .addImm(ARMCC::EQ).addReg(ARM::CPSR);
    } else {
      unsigned RHS1 = MI->getOperand(3).getReg();
      unsigned RHS2 = MI->getOperand(4).getReg();
      AddDefaultPred(BuildMI(BB, dl,
                             TII->get(isThumb2 ? ARM::t2CMPrr : ARM::CMPrr))
                     .addReg(LHS1).addReg(RHS1));
      BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2CMPrr : ARM::CMPrr))
        .addReg(LHS2).addReg(RHS2)
        .addImm(ARMCC::EQ).addReg(ARM::CPSR);
    }

    MachineBasicBlock *destMBB = MI->getOperand(RHSisZero ? 3 : 5).getMBB();
    MachineBasicBlock *exitMBB = OtherSucc(BB, destMBB);
    if (MI->getOperand(0).getImm() == ARMCC::NE)
      std::swap(destMBB, exitMBB);

    BuildMI(BB, dl, TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc))
      .addMBB(destMBB).addImm(ARMCC::EQ).addReg(ARM::CPSR);
    if (isThumb2)
      AddDefaultPred(BuildMI(BB, dl, TII->get(ARM::t2B)).addMBB(exitMBB));
    else
      BuildMI(BB, dl, TII->get(ARM::B)) .addMBB(exitMBB);

    MI->eraseFromParent();   // The pseudo instruction is gone now.
    return BB;
  }

  case ARM::Int_eh_sjlj_setjmp:
  case ARM::Int_eh_sjlj_setjmp_nofp:
  case ARM::tInt_eh_sjlj_setjmp:
  case ARM::t2Int_eh_sjlj_setjmp:
  case ARM::t2Int_eh_sjlj_setjmp_nofp:
    EmitSjLjDispatchBlock(MI, BB);
    return BB;

  case ARM::ABS:
  case ARM::t2ABS: {
    // To insert an ABS instruction, we have to insert the
    // diamond control-flow pattern.  The incoming instruction knows the
    // source vreg to test against 0, the destination vreg to set,
    // the condition code register to branch on, the
    // true/false values to select between, and a branch opcode to use.
    // It transforms
    //     V1 = ABS V0
    // into
    //     V2 = MOVS V0
    //     BCC                      (branch to SinkBB if V0 >= 0)
    //     RSBBB: V3 = RSBri V2, 0  (compute ABS if V2 < 0)
    //     SinkBB: V1 = PHI(V2, V3)
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    MachineFunction::iterator BBI = BB;
    ++BBI;
    MachineFunction *Fn = BB->getParent();
    MachineBasicBlock *RSBBB = Fn->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *SinkBB  = Fn->CreateMachineBasicBlock(LLVM_BB);
    Fn->insert(BBI, RSBBB);
    Fn->insert(BBI, SinkBB);

    unsigned int ABSSrcReg = MI->getOperand(1).getReg();
    unsigned int ABSDstReg = MI->getOperand(0).getReg();
    bool isThumb2 = Subtarget->isThumb2();
    MachineRegisterInfo &MRI = Fn->getRegInfo();
    // In Thumb mode S must not be specified if source register is the SP or
    // PC and if destination register is the SP, so restrict register class
    unsigned NewRsbDstReg = MRI.createVirtualRegister(isThumb2 ?
      (const TargetRegisterClass*)&ARM::rGPRRegClass :
      (const TargetRegisterClass*)&ARM::GPRRegClass);

    // Transfer the remainder of BB and its successor edges to sinkMBB.
    SinkBB->splice(SinkBB->begin(), BB,
      llvm::next(MachineBasicBlock::iterator(MI)),
      BB->end());
    SinkBB->transferSuccessorsAndUpdatePHIs(BB);

    BB->addSuccessor(RSBBB);
    BB->addSuccessor(SinkBB);

    // fall through to SinkMBB
    RSBBB->addSuccessor(SinkBB);

    // insert a cmp at the end of BB
    AddDefaultPred(BuildMI(BB, dl,
                           TII->get(isThumb2 ? ARM::t2CMPri : ARM::CMPri))
                   .addReg(ABSSrcReg).addImm(0));

    // insert a bcc with opposite CC to ARMCC::MI at the end of BB
    BuildMI(BB, dl,
      TII->get(isThumb2 ? ARM::t2Bcc : ARM::Bcc)).addMBB(SinkBB)
      .addImm(ARMCC::getOppositeCondition(ARMCC::MI)).addReg(ARM::CPSR);

    // insert rsbri in RSBBB
    // Note: BCC and rsbri will be converted into predicated rsbmi
    // by if-conversion pass
    BuildMI(*RSBBB, RSBBB->begin(), dl,
      TII->get(isThumb2 ? ARM::t2RSBri : ARM::RSBri), NewRsbDstReg)
      .addReg(ABSSrcReg, RegState::Kill)
      .addImm(0).addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);

    // insert PHI in SinkBB,
    // reuse ABSDstReg to not change uses of ABS instruction
    BuildMI(*SinkBB, SinkBB->begin(), dl,
      TII->get(ARM::PHI), ABSDstReg)
      .addReg(NewRsbDstReg).addMBB(RSBBB)
      .addReg(ABSSrcReg).addMBB(BB);

    // remove ABS instruction
    MI->eraseFromParent();

    // return last added BB
    return SinkBB;
  }
  case ARM::COPY_STRUCT_BYVAL_I32:
    ++NumLoopByVals;
    return EmitStructByval(MI, BB);
  }
}

void ARMTargetLowering::AdjustInstrPostInstrSelection(MachineInstr *MI,
                                                      SDNode *Node) const {
  if (!MI->hasPostISelHook()) {
    assert(!convertAddSubFlagsOpcode(MI->getOpcode()) &&
           "Pseudo flag-setting opcodes must be marked with 'hasPostISelHook'");
    return;
  }

  const MCInstrDesc *MCID = &MI->getDesc();
  // Adjust potentially 's' setting instructions after isel, i.e. ADC, SBC, RSB,
  // RSC. Coming out of isel, they have an implicit CPSR def, but the optional
  // operand is still set to noreg. If needed, set the optional operand's
  // register to CPSR, and remove the redundant implicit def.
  //
  // e.g. ADCS (..., CPSR<imp-def>) -> ADC (... opt:CPSR<def>).

  // Rename pseudo opcodes.
  unsigned NewOpc = convertAddSubFlagsOpcode(MI->getOpcode());
  if (NewOpc) {
    const ARMBaseInstrInfo *TII =
      static_cast<const ARMBaseInstrInfo*>(getTargetMachine().getInstrInfo());
    MCID = &TII->get(NewOpc);

    assert(MCID->getNumOperands() == MI->getDesc().getNumOperands() + 1 &&
           "converted opcode should be the same except for cc_out");

    MI->setDesc(*MCID);

    // Add the optional cc_out operand
    MI->addOperand(MachineOperand::CreateReg(0, /*isDef=*/true));
  }
  unsigned ccOutIdx = MCID->getNumOperands() - 1;

  // Any ARM instruction that sets the 's' bit should specify an optional
  // "cc_out" operand in the last operand position.
  if (!MI->hasOptionalDef() || !MCID->OpInfo[ccOutIdx].isOptionalDef()) {
    assert(!NewOpc && "Optional cc_out operand required");
    return;
  }
  // Look for an implicit def of CPSR added by MachineInstr ctor. Remove it
  // since we already have an optional CPSR def.
  bool definesCPSR = false;
  bool deadCPSR = false;
  for (unsigned i = MCID->getNumOperands(), e = MI->getNumOperands();
       i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() && MO.getReg() == ARM::CPSR) {
      definesCPSR = true;
      if (MO.isDead())
        deadCPSR = true;
      MI->RemoveOperand(i);
      break;
    }
  }
  if (!definesCPSR) {
    assert(!NewOpc && "Optional cc_out operand required");
    return;
  }
  assert(deadCPSR == !Node->hasAnyUseOfValue(1) && "inconsistent dead flag");
  if (deadCPSR) {
    assert(!MI->getOperand(ccOutIdx).getReg() &&
           "expect uninitialized optional cc_out operand");
    return;
  }

  // If this instruction was defined with an optional CPSR def and its dag node
  // had a live implicit CPSR def, then activate the optional CPSR def.
  MachineOperand &MO = MI->getOperand(ccOutIdx);
  MO.setReg(ARM::CPSR);
  MO.setIsDef(true);
}

//===----------------------------------------------------------------------===//
//                           ARM Optimization Hooks
//===----------------------------------------------------------------------===//

// Helper function that checks if N is a null or all ones constant.
static inline bool isZeroOrAllOnes(SDValue N, bool AllOnes) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(N);
  if (!C)
    return false;
  return AllOnes ? C->isAllOnesValue() : C->isNullValue();
}

// Return true if N is conditionally 0 or all ones.
// Detects these expressions where cc is an i1 value:
//
//   (select cc 0, y)   [AllOnes=0]
//   (select cc y, 0)   [AllOnes=0]
//   (zext cc)          [AllOnes=0]
//   (sext cc)          [AllOnes=0/1]
//   (select cc -1, y)  [AllOnes=1]
//   (select cc y, -1)  [AllOnes=1]
//
// Invert is set when N is the null/all ones constant when CC is false.
// OtherOp is set to the alternative value of N.
static bool isConditionalZeroOrAllOnes(SDNode *N, bool AllOnes,
                                       SDValue &CC, bool &Invert,
                                       SDValue &OtherOp,
                                       SelectionDAG &DAG) {
  switch (N->getOpcode()) {
  default: return false;
  case ISD::SELECT: {
    CC = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue N2 = N->getOperand(2);
    if (isZeroOrAllOnes(N1, AllOnes)) {
      Invert = false;
      OtherOp = N2;
      return true;
    }
    if (isZeroOrAllOnes(N2, AllOnes)) {
      Invert = true;
      OtherOp = N1;
      return true;
    }
    return false;
  }
  case ISD::ZERO_EXTEND:
    // (zext cc) can never be the all ones value.
    if (AllOnes)
      return false;
    // Fall through.
  case ISD::SIGN_EXTEND: {
    EVT VT = N->getValueType(0);
    CC = N->getOperand(0);
    if (CC.getValueType() != MVT::i1)
      return false;
    Invert = !AllOnes;
    if (AllOnes)
      // When looking for an AllOnes constant, N is an sext, and the 'other'
      // value is 0.
      OtherOp = DAG.getConstant(0, VT);
    else if (N->getOpcode() == ISD::ZERO_EXTEND)
      // When looking for a 0 constant, N can be zext or sext.
      OtherOp = DAG.getConstant(1, VT);
    else
      OtherOp = DAG.getConstant(APInt::getAllOnesValue(VT.getSizeInBits()), VT);
    return true;
  }
  }
}

// Combine a constant select operand into its use:
//
//   (add (select cc, 0, c), x)  -> (select cc, x, (add, x, c))
//   (sub x, (select cc, 0, c))  -> (select cc, x, (sub, x, c))
//   (and (select cc, -1, c), x) -> (select cc, x, (and, x, c))  [AllOnes=1]
//   (or  (select cc, 0, c), x)  -> (select cc, x, (or, x, c))
//   (xor (select cc, 0, c), x)  -> (select cc, x, (xor, x, c))
//
// The transform is rejected if the select doesn't have a constant operand that
// is null, or all ones when AllOnes is set.
//
// Also recognize sext/zext from i1:
//
//   (add (zext cc), x) -> (select cc (add x, 1), x)
//   (add (sext cc), x) -> (select cc (add x, -1), x)
//
// These transformations eventually create predicated instructions.
//
// @param N       The node to transform.
// @param Slct    The N operand that is a select.
// @param OtherOp The other N operand (x above).
// @param DCI     Context.
// @param AllOnes Require the select constant to be all ones instead of null.
// @returns The new node, or SDValue() on failure.
static
SDValue combineSelectAndUse(SDNode *N, SDValue Slct, SDValue OtherOp,
                            TargetLowering::DAGCombinerInfo &DCI,
                            bool AllOnes = false) {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDValue NonConstantVal;
  SDValue CCOp;
  bool SwapSelectOps;
  if (!isConditionalZeroOrAllOnes(Slct.getNode(), AllOnes, CCOp, SwapSelectOps,
                                  NonConstantVal, DAG))
    return SDValue();

  // Slct is now know to be the desired identity constant when CC is true.
  SDValue TrueVal = OtherOp;
  SDValue FalseVal = DAG.getNode(N->getOpcode(), N->getDebugLoc(), VT,
                                 OtherOp, NonConstantVal);
  // Unless SwapSelectOps says CC should be false.
  if (SwapSelectOps)
    std::swap(TrueVal, FalseVal);

  return DAG.getNode(ISD::SELECT, N->getDebugLoc(), VT,
                     CCOp, TrueVal, FalseVal);
}

// Attempt combineSelectAndUse on each operand of a commutative operator N.
static
SDValue combineSelectAndUseCommutative(SDNode *N, bool AllOnes,
                                       TargetLowering::DAGCombinerInfo &DCI) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  if (N0.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N0, N1, DCI, AllOnes);
    if (Result.getNode())
      return Result;
  }
  if (N1.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N1, N0, DCI, AllOnes);
    if (Result.getNode())
      return Result;
  }
  return SDValue();
}

// AddCombineToVPADDL- For pair-wise add on neon, use the vpaddl instruction
// (only after legalization).
static SDValue AddCombineToVPADDL(SDNode *N, SDValue N0, SDValue N1,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const ARMSubtarget *Subtarget) {

  // Only perform optimization if after legalize, and if NEON is available. We
  // also expected both operands to be BUILD_VECTORs.
  if (DCI.isBeforeLegalize() || !Subtarget->hasNEON()
      || N0.getOpcode() != ISD::BUILD_VECTOR
      || N1.getOpcode() != ISD::BUILD_VECTOR)
    return SDValue();

  // Check output type since VPADDL operand elements can only be 8, 16, or 32.
  EVT VT = N->getValueType(0);
  if (!VT.isInteger() || VT.getVectorElementType() == MVT::i64)
    return SDValue();

  // Check that the vector operands are of the right form.
  // N0 and N1 are BUILD_VECTOR nodes with N number of EXTRACT_VECTOR
  // operands, where N is the size of the formed vector.
  // Each EXTRACT_VECTOR should have the same input vector and odd or even
  // index such that we have a pair wise add pattern.

  // Grab the vector that all EXTRACT_VECTOR nodes should be referencing.
  if (N0->getOperand(0)->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
    return SDValue();
  SDValue Vec = N0->getOperand(0)->getOperand(0);
  SDNode *V = Vec.getNode();
  unsigned nextIndex = 0;

  // For each operands to the ADD which are BUILD_VECTORs,
  // check to see if each of their operands are an EXTRACT_VECTOR with
  // the same vector and appropriate index.
  for (unsigned i = 0, e = N0->getNumOperands(); i != e; ++i) {
    if (N0->getOperand(i)->getOpcode() == ISD::EXTRACT_VECTOR_ELT
        && N1->getOperand(i)->getOpcode() == ISD::EXTRACT_VECTOR_ELT) {

      SDValue ExtVec0 = N0->getOperand(i);
      SDValue ExtVec1 = N1->getOperand(i);

      // First operand is the vector, verify its the same.
      if (V != ExtVec0->getOperand(0).getNode() ||
          V != ExtVec1->getOperand(0).getNode())
        return SDValue();

      // Second is the constant, verify its correct.
      ConstantSDNode *C0 = dyn_cast<ConstantSDNode>(ExtVec0->getOperand(1));
      ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(ExtVec1->getOperand(1));

      // For the constant, we want to see all the even or all the odd.
      if (!C0 || !C1 || C0->getZExtValue() != nextIndex
          || C1->getZExtValue() != nextIndex+1)
        return SDValue();

      // Increment index.
      nextIndex+=2;
    } else
      return SDValue();
  }

  // Create VPADDL node.
  SelectionDAG &DAG = DCI.DAG;
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Build operand list.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(DAG.getConstant(Intrinsic::arm_neon_vpaddls,
                                TLI.getPointerTy()));

  // Input is the vector.
  Ops.push_back(Vec);

  // Get widened type and narrowed type.
  MVT widenType;
  unsigned numElem = VT.getVectorNumElements();
  switch (VT.getVectorElementType().getSimpleVT().SimpleTy) {
    case MVT::i8: widenType = MVT::getVectorVT(MVT::i16, numElem); break;
    case MVT::i16: widenType = MVT::getVectorVT(MVT::i32, numElem); break;
    case MVT::i32: widenType = MVT::getVectorVT(MVT::i64, numElem); break;
    default:
      llvm_unreachable("Invalid vector element type for padd optimization.");
  }

  SDValue tmp = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, N->getDebugLoc(),
                            widenType, &Ops[0], Ops.size());
  return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), VT, tmp);
}

static SDValue findMUL_LOHI(SDValue V) {
  if (V->getOpcode() == ISD::UMUL_LOHI ||
      V->getOpcode() == ISD::SMUL_LOHI)
    return V;
  return SDValue();
}

static SDValue AddCombineTo64bitMLAL(SDNode *AddcNode,
                                     TargetLowering::DAGCombinerInfo &DCI,
                                     const ARMSubtarget *Subtarget) {

  if (Subtarget->isThumb1Only()) return SDValue();

  // Only perform the checks after legalize when the pattern is available.
  if (DCI.isBeforeLegalize()) return SDValue();

  // Look for multiply add opportunities.
  // The pattern is a ISD::UMUL_LOHI followed by two add nodes, where
  // each add nodes consumes a value from ISD::UMUL_LOHI and there is
  // a glue link from the first add to the second add.
  // If we find this pattern, we can replace the U/SMUL_LOHI, ADDC, and ADDE by
  // a S/UMLAL instruction.
  //          loAdd   UMUL_LOHI
  //            \    / :lo    \ :hi
  //             \  /          \          [no multiline comment]
  //              ADDC         |  hiAdd
  //                 \ :glue  /  /
  //                  \      /  /
  //                    ADDE
  //
  assert(AddcNode->getOpcode() == ISD::ADDC && "Expect an ADDC");
  SDValue AddcOp0 = AddcNode->getOperand(0);
  SDValue AddcOp1 = AddcNode->getOperand(1);

  // Check if the two operands are from the same mul_lohi node.
  if (AddcOp0.getNode() == AddcOp1.getNode())
    return SDValue();

  assert(AddcNode->getNumValues() == 2 &&
         AddcNode->getValueType(0) == MVT::i32 &&
         AddcNode->getValueType(1) == MVT::Glue &&
         "Expect ADDC with two result values: i32, glue");

  // Check that the ADDC adds the low result of the S/UMUL_LOHI.
  if (AddcOp0->getOpcode() != ISD::UMUL_LOHI &&
      AddcOp0->getOpcode() != ISD::SMUL_LOHI &&
      AddcOp1->getOpcode() != ISD::UMUL_LOHI &&
      AddcOp1->getOpcode() != ISD::SMUL_LOHI)
    return SDValue();

  // Look for the glued ADDE.
  SDNode* AddeNode = AddcNode->getGluedUser();
  if (AddeNode == NULL)
    return SDValue();

  // Make sure it is really an ADDE.
  if (AddeNode->getOpcode() != ISD::ADDE)
    return SDValue();

  assert(AddeNode->getNumOperands() == 3 &&
         AddeNode->getOperand(2).getValueType() == MVT::Glue &&
         "ADDE node has the wrong inputs");

  // Check for the triangle shape.
  SDValue AddeOp0 = AddeNode->getOperand(0);
  SDValue AddeOp1 = AddeNode->getOperand(1);

  // Make sure that the ADDE operands are not coming from the same node.
  if (AddeOp0.getNode() == AddeOp1.getNode())
    return SDValue();

  // Find the MUL_LOHI node walking up ADDE's operands.
  bool IsLeftOperandMUL = false;
  SDValue MULOp = findMUL_LOHI(AddeOp0);
  if (MULOp == SDValue())
   MULOp = findMUL_LOHI(AddeOp1);
  else
    IsLeftOperandMUL = true;
  if (MULOp == SDValue())
     return SDValue();

  // Figure out the right opcode.
  unsigned Opc = MULOp->getOpcode();
  unsigned FinalOpc = (Opc == ISD::SMUL_LOHI) ? ARMISD::SMLAL : ARMISD::UMLAL;

  // Figure out the high and low input values to the MLAL node.
  SDValue* HiMul = &MULOp;
  SDValue* HiAdd = NULL;
  SDValue* LoMul = NULL;
  SDValue* LowAdd = NULL;

  if (IsLeftOperandMUL)
    HiAdd = &AddeOp1;
  else
    HiAdd = &AddeOp0;


  if (AddcOp0->getOpcode() == Opc) {
    LoMul = &AddcOp0;
    LowAdd = &AddcOp1;
  }
  if (AddcOp1->getOpcode() == Opc) {
    LoMul = &AddcOp1;
    LowAdd = &AddcOp0;
  }

  if (LoMul == NULL)
    return SDValue();

  if (LoMul->getNode() != HiMul->getNode())
    return SDValue();

  // Create the merged node.
  SelectionDAG &DAG = DCI.DAG;

  // Build operand list.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(LoMul->getOperand(0));
  Ops.push_back(LoMul->getOperand(1));
  Ops.push_back(*LowAdd);
  Ops.push_back(*HiAdd);

  SDValue MLALNode =  DAG.getNode(FinalOpc, AddcNode->getDebugLoc(),
                                 DAG.getVTList(MVT::i32, MVT::i32),
                                 &Ops[0], Ops.size());

  // Replace the ADDs' nodes uses by the MLA node's values.
  SDValue HiMLALResult(MLALNode.getNode(), 1);
  DAG.ReplaceAllUsesOfValueWith(SDValue(AddeNode, 0), HiMLALResult);

  SDValue LoMLALResult(MLALNode.getNode(), 0);
  DAG.ReplaceAllUsesOfValueWith(SDValue(AddcNode, 0), LoMLALResult);

  // Return original node to notify the driver to stop replacing.
  SDValue resNode(AddcNode, 0);
  return resNode;
}

/// PerformADDCCombine - Target-specific dag combine transform from
/// ISD::ADDC, ISD::ADDE, and ISD::MUL_LOHI to MLAL.
static SDValue PerformADDCCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const ARMSubtarget *Subtarget) {

  return AddCombineTo64bitMLAL(N, DCI, Subtarget);

}

/// PerformADDCombineWithOperands - Try DAG combinations for an ADD with
/// operands N0 and N1.  This is a helper for PerformADDCombine that is
/// called with the default operands, and if that fails, with commuted
/// operands.
static SDValue PerformADDCombineWithOperands(SDNode *N, SDValue N0, SDValue N1,
                                          TargetLowering::DAGCombinerInfo &DCI,
                                          const ARMSubtarget *Subtarget){

  // Attempt to create vpaddl for this add.
  SDValue Result = AddCombineToVPADDL(N, N0, N1, DCI, Subtarget);
  if (Result.getNode())
    return Result;

  // fold (add (select cc, 0, c), x) -> (select cc, x, (add, x, c))
  if (N0.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N0, N1, DCI);
    if (Result.getNode()) return Result;
  }
  return SDValue();
}

/// PerformADDCombine - Target-specific dag combine xforms for ISD::ADD.
///
static SDValue PerformADDCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const ARMSubtarget *Subtarget) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // First try with the default operand order.
  SDValue Result = PerformADDCombineWithOperands(N, N0, N1, DCI, Subtarget);
  if (Result.getNode())
    return Result;

  // If that didn't work, try again with the operands commuted.
  return PerformADDCombineWithOperands(N, N1, N0, DCI, Subtarget);
}

/// PerformSUBCombine - Target-specific dag combine xforms for ISD::SUB.
///
static SDValue PerformSUBCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // fold (sub x, (select cc, 0, c)) -> (select cc, x, (sub, x, c))
  if (N1.getNode()->hasOneUse()) {
    SDValue Result = combineSelectAndUse(N, N1, N0, DCI);
    if (Result.getNode()) return Result;
  }

  return SDValue();
}

/// PerformVMULCombine
/// Distribute (A + B) * C to (A * C) + (B * C) to take advantage of the
/// special multiplier accumulator forwarding.
///   vmul d3, d0, d2
///   vmla d3, d1, d2
/// is faster than
///   vadd d3, d0, d1
///   vmul d3, d3, d2
static SDValue PerformVMULCombine(SDNode *N,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const ARMSubtarget *Subtarget) {
  if (!Subtarget->hasVMLxForwarding())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  unsigned Opcode = N0.getOpcode();
  if (Opcode != ISD::ADD && Opcode != ISD::SUB &&
      Opcode != ISD::FADD && Opcode != ISD::FSUB) {
    Opcode = N1.getOpcode();
    if (Opcode != ISD::ADD && Opcode != ISD::SUB &&
        Opcode != ISD::FADD && Opcode != ISD::FSUB)
      return SDValue();
    std::swap(N0, N1);
  }

  EVT VT = N->getValueType(0);
  DebugLoc DL = N->getDebugLoc();
  SDValue N00 = N0->getOperand(0);
  SDValue N01 = N0->getOperand(1);
  return DAG.getNode(Opcode, DL, VT,
                     DAG.getNode(ISD::MUL, DL, VT, N00, N1),
                     DAG.getNode(ISD::MUL, DL, VT, N01, N1));
}

static SDValue PerformMULCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const ARMSubtarget *Subtarget) {
  SelectionDAG &DAG = DCI.DAG;

  if (Subtarget->isThumb1Only())
    return SDValue();

  if (DCI.isBeforeLegalize() || DCI.isCalledByLegalizer())
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT.is64BitVector() || VT.is128BitVector())
    return PerformVMULCombine(N, DCI, Subtarget);
  if (VT != MVT::i32)
    return SDValue();

  ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!C)
    return SDValue();

  int64_t MulAmt = C->getSExtValue();
  unsigned ShiftAmt = CountTrailingZeros_64(MulAmt);

  ShiftAmt = ShiftAmt & (32 - 1);
  SDValue V = N->getOperand(0);
  DebugLoc DL = N->getDebugLoc();

  SDValue Res;
  MulAmt >>= ShiftAmt;

  if (MulAmt >= 0) {
    if (isPowerOf2_32(MulAmt - 1)) {
      // (mul x, 2^N + 1) => (add (shl x, N), x)
      Res = DAG.getNode(ISD::ADD, DL, VT,
                        V,
                        DAG.getNode(ISD::SHL, DL, VT,
                                    V,
                                    DAG.getConstant(Log2_32(MulAmt - 1),
                                                    MVT::i32)));
    } else if (isPowerOf2_32(MulAmt + 1)) {
      // (mul x, 2^N - 1) => (sub (shl x, N), x)
      Res = DAG.getNode(ISD::SUB, DL, VT,
                        DAG.getNode(ISD::SHL, DL, VT,
                                    V,
                                    DAG.getConstant(Log2_32(MulAmt + 1),
                                                    MVT::i32)),
                        V);
    } else
      return SDValue();
  } else {
    uint64_t MulAmtAbs = -MulAmt;
    if (isPowerOf2_32(MulAmtAbs + 1)) {
      // (mul x, -(2^N - 1)) => (sub x, (shl x, N))
      Res = DAG.getNode(ISD::SUB, DL, VT,
                        V,
                        DAG.getNode(ISD::SHL, DL, VT,
                                    V,
                                    DAG.getConstant(Log2_32(MulAmtAbs + 1),
                                                    MVT::i32)));
    } else if (isPowerOf2_32(MulAmtAbs - 1)) {
      // (mul x, -(2^N + 1)) => - (add (shl x, N), x)
      Res = DAG.getNode(ISD::ADD, DL, VT,
                        V,
                        DAG.getNode(ISD::SHL, DL, VT,
                                    V,
                                    DAG.getConstant(Log2_32(MulAmtAbs-1),
                                                    MVT::i32)));
      Res = DAG.getNode(ISD::SUB, DL, VT,
                        DAG.getConstant(0, MVT::i32),Res);

    } else
      return SDValue();
  }

  if (ShiftAmt != 0)
    Res = DAG.getNode(ISD::SHL, DL, VT,
                      Res, DAG.getConstant(ShiftAmt, MVT::i32));

  // Do not add new nodes to DAG combiner worklist.
  DCI.CombineTo(N, Res, false);
  return SDValue();
}

static SDValue PerformANDCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const ARMSubtarget *Subtarget) {

  // Attempt to use immediate-form VBIC
  BuildVectorSDNode *BVN = dyn_cast<BuildVectorSDNode>(N->getOperand(1));
  DebugLoc dl = N->getDebugLoc();
  EVT VT = N->getValueType(0);
  SelectionDAG &DAG = DCI.DAG;

  if(!DAG.getTargetLoweringInfo().isTypeLegal(VT))
    return SDValue();

  APInt SplatBits, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  if (BVN &&
      BVN->isConstantSplat(SplatBits, SplatUndef, SplatBitSize, HasAnyUndefs)) {
    if (SplatBitSize <= 64) {
      EVT VbicVT;
      SDValue Val = isNEONModifiedImm((~SplatBits).getZExtValue(),
                                      SplatUndef.getZExtValue(), SplatBitSize,
                                      DAG, VbicVT, VT.is128BitVector(),
                                      OtherModImm);
      if (Val.getNode()) {
        SDValue Input =
          DAG.getNode(ISD::BITCAST, dl, VbicVT, N->getOperand(0));
        SDValue Vbic = DAG.getNode(ARMISD::VBICIMM, dl, VbicVT, Input, Val);
        return DAG.getNode(ISD::BITCAST, dl, VT, Vbic);
      }
    }
  }

  if (!Subtarget->isThumb1Only()) {
    // fold (and (select cc, -1, c), x) -> (select cc, x, (and, x, c))
    SDValue Result = combineSelectAndUseCommutative(N, true, DCI);
    if (Result.getNode())
      return Result;
  }

  return SDValue();
}

/// PerformORCombine - Target-specific dag combine xforms for ISD::OR
static SDValue PerformORCombine(SDNode *N,
                                TargetLowering::DAGCombinerInfo &DCI,
                                const ARMSubtarget *Subtarget) {
  // Attempt to use immediate-form VORR
  BuildVectorSDNode *BVN = dyn_cast<BuildVectorSDNode>(N->getOperand(1));
  DebugLoc dl = N->getDebugLoc();
  EVT VT = N->getValueType(0);
  SelectionDAG &DAG = DCI.DAG;

  if(!DAG.getTargetLoweringInfo().isTypeLegal(VT))
    return SDValue();

  APInt SplatBits, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  if (BVN && Subtarget->hasNEON() &&
      BVN->isConstantSplat(SplatBits, SplatUndef, SplatBitSize, HasAnyUndefs)) {
    if (SplatBitSize <= 64) {
      EVT VorrVT;
      SDValue Val = isNEONModifiedImm(SplatBits.getZExtValue(),
                                      SplatUndef.getZExtValue(), SplatBitSize,
                                      DAG, VorrVT, VT.is128BitVector(),
                                      OtherModImm);
      if (Val.getNode()) {
        SDValue Input =
          DAG.getNode(ISD::BITCAST, dl, VorrVT, N->getOperand(0));
        SDValue Vorr = DAG.getNode(ARMISD::VORRIMM, dl, VorrVT, Input, Val);
        return DAG.getNode(ISD::BITCAST, dl, VT, Vorr);
      }
    }
  }

  if (!Subtarget->isThumb1Only()) {
    // fold (or (select cc, 0, c), x) -> (select cc, x, (or, x, c))
    SDValue Result = combineSelectAndUseCommutative(N, false, DCI);
    if (Result.getNode())
      return Result;
  }

  // The code below optimizes (or (and X, Y), Z).
  // The AND operand needs to have a single user to make these optimizations
  // profitable.
  SDValue N0 = N->getOperand(0);
  if (N0.getOpcode() != ISD::AND || !N0.hasOneUse())
    return SDValue();
  SDValue N1 = N->getOperand(1);

  // (or (and B, A), (and C, ~A)) => (VBSL A, B, C) when A is a constant.
  if (Subtarget->hasNEON() && N1.getOpcode() == ISD::AND && VT.isVector() &&
      DAG.getTargetLoweringInfo().isTypeLegal(VT)) {
    APInt SplatUndef;
    unsigned SplatBitSize;
    bool HasAnyUndefs;

    BuildVectorSDNode *BVN0 = dyn_cast<BuildVectorSDNode>(N0->getOperand(1));
    APInt SplatBits0;
    if (BVN0 && BVN0->isConstantSplat(SplatBits0, SplatUndef, SplatBitSize,
                                  HasAnyUndefs) && !HasAnyUndefs) {
      BuildVectorSDNode *BVN1 = dyn_cast<BuildVectorSDNode>(N1->getOperand(1));
      APInt SplatBits1;
      if (BVN1 && BVN1->isConstantSplat(SplatBits1, SplatUndef, SplatBitSize,
                                    HasAnyUndefs) && !HasAnyUndefs &&
          SplatBits0 == ~SplatBits1) {
        // Canonicalize the vector type to make instruction selection simpler.
        EVT CanonicalVT = VT.is128BitVector() ? MVT::v4i32 : MVT::v2i32;
        SDValue Result = DAG.getNode(ARMISD::VBSL, dl, CanonicalVT,
                                     N0->getOperand(1), N0->getOperand(0),
                                     N1->getOperand(0));
        return DAG.getNode(ISD::BITCAST, dl, VT, Result);
      }
    }
  }

  // Try to use the ARM/Thumb2 BFI (bitfield insert) instruction when
  // reasonable.

  // BFI is only available on V6T2+
  if (Subtarget->isThumb1Only() || !Subtarget->hasV6T2Ops())
    return SDValue();

  DebugLoc DL = N->getDebugLoc();
  // 1) or (and A, mask), val => ARMbfi A, val, mask
  //      iff (val & mask) == val
  //
  // 2) or (and A, mask), (and B, mask2) => ARMbfi A, (lsr B, amt), mask
  //  2a) iff isBitFieldInvertedMask(mask) && isBitFieldInvertedMask(~mask2)
  //          && mask == ~mask2
  //  2b) iff isBitFieldInvertedMask(~mask) && isBitFieldInvertedMask(mask2)
  //          && ~mask == mask2
  //  (i.e., copy a bitfield value into another bitfield of the same width)

  if (VT != MVT::i32)
    return SDValue();

  SDValue N00 = N0.getOperand(0);

  // The value and the mask need to be constants so we can verify this is
  // actually a bitfield set. If the mask is 0xffff, we can do better
  // via a movt instruction, so don't use BFI in that case.
  SDValue MaskOp = N0.getOperand(1);
  ConstantSDNode *MaskC = dyn_cast<ConstantSDNode>(MaskOp);
  if (!MaskC)
    return SDValue();
  unsigned Mask = MaskC->getZExtValue();
  if (Mask == 0xffff)
    return SDValue();
  SDValue Res;
  // Case (1): or (and A, mask), val => ARMbfi A, val, mask
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N1C) {
    unsigned Val = N1C->getZExtValue();
    if ((Val & ~Mask) != Val)
      return SDValue();

    if (ARM::isBitFieldInvertedMask(Mask)) {
      Val >>= CountTrailingZeros_32(~Mask);

      Res = DAG.getNode(ARMISD::BFI, DL, VT, N00,
                        DAG.getConstant(Val, MVT::i32),
                        DAG.getConstant(Mask, MVT::i32));

      // Do not add new nodes to DAG combiner worklist.
      DCI.CombineTo(N, Res, false);
      return SDValue();
    }
  } else if (N1.getOpcode() == ISD::AND) {
    // case (2) or (and A, mask), (and B, mask2) => ARMbfi A, (lsr B, amt), mask
    ConstantSDNode *N11C = dyn_cast<ConstantSDNode>(N1.getOperand(1));
    if (!N11C)
      return SDValue();
    unsigned Mask2 = N11C->getZExtValue();

    // Mask and ~Mask2 (or reverse) must be equivalent for the BFI pattern
    // as is to match.
    if (ARM::isBitFieldInvertedMask(Mask) &&
        (Mask == ~Mask2)) {
      // The pack halfword instruction works better for masks that fit it,
      // so use that when it's available.
      if (Subtarget->hasT2ExtractPack() &&
          (Mask == 0xffff || Mask == 0xffff0000))
        return SDValue();
      // 2a
      unsigned amt = CountTrailingZeros_32(Mask2);
      Res = DAG.getNode(ISD::SRL, DL, VT, N1.getOperand(0),
                        DAG.getConstant(amt, MVT::i32));
      Res = DAG.getNode(ARMISD::BFI, DL, VT, N00, Res,
                        DAG.getConstant(Mask, MVT::i32));
      // Do not add new nodes to DAG combiner worklist.
      DCI.CombineTo(N, Res, false);
      return SDValue();
    } else if (ARM::isBitFieldInvertedMask(~Mask) &&
               (~Mask == Mask2)) {
      // The pack halfword instruction works better for masks that fit it,
      // so use that when it's available.
      if (Subtarget->hasT2ExtractPack() &&
          (Mask2 == 0xffff || Mask2 == 0xffff0000))
        return SDValue();
      // 2b
      unsigned lsb = CountTrailingZeros_32(Mask);
      Res = DAG.getNode(ISD::SRL, DL, VT, N00,
                        DAG.getConstant(lsb, MVT::i32));
      Res = DAG.getNode(ARMISD::BFI, DL, VT, N1.getOperand(0), Res,
                        DAG.getConstant(Mask2, MVT::i32));
      // Do not add new nodes to DAG combiner worklist.
      DCI.CombineTo(N, Res, false);
      return SDValue();
    }
  }

  if (DAG.MaskedValueIsZero(N1, MaskC->getAPIntValue()) &&
      N00.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N00.getOperand(1)) &&
      ARM::isBitFieldInvertedMask(~Mask)) {
    // Case (3): or (and (shl A, #shamt), mask), B => ARMbfi B, A, ~mask
    // where lsb(mask) == #shamt and masked bits of B are known zero.
    SDValue ShAmt = N00.getOperand(1);
    unsigned ShAmtC = cast<ConstantSDNode>(ShAmt)->getZExtValue();
    unsigned LSB = CountTrailingZeros_32(Mask);
    if (ShAmtC != LSB)
      return SDValue();

    Res = DAG.getNode(ARMISD::BFI, DL, VT, N1, N00.getOperand(0),
                      DAG.getConstant(~Mask, MVT::i32));

    // Do not add new nodes to DAG combiner worklist.
    DCI.CombineTo(N, Res, false);
  }

  return SDValue();
}

static SDValue PerformXORCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const ARMSubtarget *Subtarget) {
  EVT VT = N->getValueType(0);
  SelectionDAG &DAG = DCI.DAG;

  if(!DAG.getTargetLoweringInfo().isTypeLegal(VT))
    return SDValue();

  if (!Subtarget->isThumb1Only()) {
    // fold (xor (select cc, 0, c), x) -> (select cc, x, (xor, x, c))
    SDValue Result = combineSelectAndUseCommutative(N, false, DCI);
    if (Result.getNode())
      return Result;
  }

  return SDValue();
}

/// PerformBFICombine - (bfi A, (and B, Mask1), Mask2) -> (bfi A, B, Mask2) iff
/// the bits being cleared by the AND are not demanded by the BFI.
static SDValue PerformBFICombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {
  SDValue N1 = N->getOperand(1);
  if (N1.getOpcode() == ISD::AND) {
    ConstantSDNode *N11C = dyn_cast<ConstantSDNode>(N1.getOperand(1));
    if (!N11C)
      return SDValue();
    unsigned InvMask = cast<ConstantSDNode>(N->getOperand(2))->getZExtValue();
    unsigned LSB = CountTrailingZeros_32(~InvMask);
    unsigned Width = (32 - CountLeadingZeros_32(~InvMask)) - LSB;
    unsigned Mask = (1 << Width)-1;
    unsigned Mask2 = N11C->getZExtValue();
    if ((Mask & (~Mask2)) == 0)
      return DCI.DAG.getNode(ARMISD::BFI, N->getDebugLoc(), N->getValueType(0),
                             N->getOperand(0), N1.getOperand(0),
                             N->getOperand(2));
  }
  return SDValue();
}

/// PerformVMOVRRDCombine - Target-specific dag combine xforms for
/// ARMISD::VMOVRRD.
static SDValue PerformVMOVRRDCombine(SDNode *N,
                                     TargetLowering::DAGCombinerInfo &DCI) {
  // vmovrrd(vmovdrr x, y) -> x,y
  SDValue InDouble = N->getOperand(0);
  if (InDouble.getOpcode() == ARMISD::VMOVDRR)
    return DCI.CombineTo(N, InDouble.getOperand(0), InDouble.getOperand(1));

  // vmovrrd(load f64) -> (load i32), (load i32)
  SDNode *InNode = InDouble.getNode();
  if (ISD::isNormalLoad(InNode) && InNode->hasOneUse() &&
      InNode->getValueType(0) == MVT::f64 &&
      InNode->getOperand(1).getOpcode() == ISD::FrameIndex &&
      !cast<LoadSDNode>(InNode)->isVolatile()) {
    // TODO: Should this be done for non-FrameIndex operands?
    LoadSDNode *LD = cast<LoadSDNode>(InNode);

    SelectionDAG &DAG = DCI.DAG;
    DebugLoc DL = LD->getDebugLoc();
    SDValue BasePtr = LD->getBasePtr();
    SDValue NewLD1 = DAG.getLoad(MVT::i32, DL, LD->getChain(), BasePtr,
                                 LD->getPointerInfo(), LD->isVolatile(),
                                 LD->isNonTemporal(), LD->isInvariant(),
                                 LD->getAlignment());

    SDValue OffsetPtr = DAG.getNode(ISD::ADD, DL, MVT::i32, BasePtr,
                                    DAG.getConstant(4, MVT::i32));
    SDValue NewLD2 = DAG.getLoad(MVT::i32, DL, NewLD1.getValue(1), OffsetPtr,
                                 LD->getPointerInfo(), LD->isVolatile(),
                                 LD->isNonTemporal(), LD->isInvariant(),
                                 std::min(4U, LD->getAlignment() / 2));

    DAG.ReplaceAllUsesOfValueWith(SDValue(LD, 1), NewLD2.getValue(1));
    SDValue Result = DCI.CombineTo(N, NewLD1, NewLD2);
    DCI.RemoveFromWorklist(LD);
    DAG.DeleteNode(LD);
    return Result;
  }

  return SDValue();
}

/// PerformVMOVDRRCombine - Target-specific dag combine xforms for
/// ARMISD::VMOVDRR.  This is also used for BUILD_VECTORs with 2 operands.
static SDValue PerformVMOVDRRCombine(SDNode *N, SelectionDAG &DAG) {
  // N=vmovrrd(X); vmovdrr(N:0, N:1) -> bit_convert(X)
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  if (Op0.getOpcode() == ISD::BITCAST)
    Op0 = Op0.getOperand(0);
  if (Op1.getOpcode() == ISD::BITCAST)
    Op1 = Op1.getOperand(0);
  if (Op0.getOpcode() == ARMISD::VMOVRRD &&
      Op0.getNode() == Op1.getNode() &&
      Op0.getResNo() == 0 && Op1.getResNo() == 1)
    return DAG.getNode(ISD::BITCAST, N->getDebugLoc(),
                       N->getValueType(0), Op0.getOperand(0));
  return SDValue();
}

/// PerformSTORECombine - Target-specific dag combine xforms for
/// ISD::STORE.
static SDValue PerformSTORECombine(SDNode *N,
                                   TargetLowering::DAGCombinerInfo &DCI) {
  StoreSDNode *St = cast<StoreSDNode>(N);
  if (St->isVolatile())
    return SDValue();

  // Optimize trunc store (of multiple scalars) to shuffle and store.  First,
  // pack all of the elements in one place.  Next, store to memory in fewer
  // chunks.
  SDValue StVal = St->getValue();
  EVT VT = StVal.getValueType();
  if (St->isTruncatingStore() && VT.isVector()) {
    SelectionDAG &DAG = DCI.DAG;
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    EVT StVT = St->getMemoryVT();
    unsigned NumElems = VT.getVectorNumElements();
    assert(StVT != VT && "Cannot truncate to the same type");
    unsigned FromEltSz = VT.getVectorElementType().getSizeInBits();
    unsigned ToEltSz = StVT.getVectorElementType().getSizeInBits();

    // From, To sizes and ElemCount must be pow of two
    if (!isPowerOf2_32(NumElems * FromEltSz * ToEltSz)) return SDValue();

    // We are going to use the original vector elt for storing.
    // Accumulated smaller vector elements must be a multiple of the store size.
    if (0 != (NumElems * FromEltSz) % ToEltSz) return SDValue();

    unsigned SizeRatio  = FromEltSz / ToEltSz;
    assert(SizeRatio * NumElems * ToEltSz == VT.getSizeInBits());

    // Create a type on which we perform the shuffle.
    EVT WideVecVT = EVT::getVectorVT(*DAG.getContext(), StVT.getScalarType(),
                                     NumElems*SizeRatio);
    assert(WideVecVT.getSizeInBits() == VT.getSizeInBits());

    DebugLoc DL = St->getDebugLoc();
    SDValue WideVec = DAG.getNode(ISD::BITCAST, DL, WideVecVT, StVal);
    SmallVector<int, 8> ShuffleVec(NumElems * SizeRatio, -1);
    for (unsigned i = 0; i < NumElems; ++i) ShuffleVec[i] = i * SizeRatio;

    // Can't shuffle using an illegal type.
    if (!TLI.isTypeLegal(WideVecVT)) return SDValue();

    SDValue Shuff = DAG.getVectorShuffle(WideVecVT, DL, WideVec,
                                DAG.getUNDEF(WideVec.getValueType()),
                                ShuffleVec.data());
    // At this point all of the data is stored at the bottom of the
    // register. We now need to save it to mem.

    // Find the largest store unit
    MVT StoreType = MVT::i8;
    for (unsigned tp = MVT::FIRST_INTEGER_VALUETYPE;
         tp < MVT::LAST_INTEGER_VALUETYPE; ++tp) {
      MVT Tp = (MVT::SimpleValueType)tp;
      if (TLI.isTypeLegal(Tp) && Tp.getSizeInBits() <= NumElems * ToEltSz)
        StoreType = Tp;
    }
    // Didn't find a legal store type.
    if (!TLI.isTypeLegal(StoreType))
      return SDValue();

    // Bitcast the original vector into a vector of store-size units
    EVT StoreVecVT = EVT::getVectorVT(*DAG.getContext(),
            StoreType, VT.getSizeInBits()/EVT(StoreType).getSizeInBits());
    assert(StoreVecVT.getSizeInBits() == VT.getSizeInBits());
    SDValue ShuffWide = DAG.getNode(ISD::BITCAST, DL, StoreVecVT, Shuff);
    SmallVector<SDValue, 8> Chains;
    SDValue Increment = DAG.getConstant(StoreType.getSizeInBits()/8,
                                        TLI.getPointerTy());
    SDValue BasePtr = St->getBasePtr();

    // Perform one or more big stores into memory.
    unsigned E = (ToEltSz*NumElems)/StoreType.getSizeInBits();
    for (unsigned I = 0; I < E; I++) {
      SDValue SubVec = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL,
                                   StoreType, ShuffWide,
                                   DAG.getIntPtrConstant(I));
      SDValue Ch = DAG.getStore(St->getChain(), DL, SubVec, BasePtr,
                                St->getPointerInfo(), St->isVolatile(),
                                St->isNonTemporal(), St->getAlignment());
      BasePtr = DAG.getNode(ISD::ADD, DL, BasePtr.getValueType(), BasePtr,
                            Increment);
      Chains.push_back(Ch);
    }
    return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, &Chains[0],
                       Chains.size());
  }

  if (!ISD::isNormalStore(St))
    return SDValue();

  // Split a store of a VMOVDRR into two integer stores to avoid mixing NEON and
  // ARM stores of arguments in the same cache line.
  if (StVal.getNode()->getOpcode() == ARMISD::VMOVDRR &&
      StVal.getNode()->hasOneUse()) {
    SelectionDAG  &DAG = DCI.DAG;
    DebugLoc DL = St->getDebugLoc();
    SDValue BasePtr = St->getBasePtr();
    SDValue NewST1 = DAG.getStore(St->getChain(), DL,
                                  StVal.getNode()->getOperand(0), BasePtr,
                                  St->getPointerInfo(), St->isVolatile(),
                                  St->isNonTemporal(), St->getAlignment());

    SDValue OffsetPtr = DAG.getNode(ISD::ADD, DL, MVT::i32, BasePtr,
                                    DAG.getConstant(4, MVT::i32));
    return DAG.getStore(NewST1.getValue(0), DL, StVal.getNode()->getOperand(1),
                        OffsetPtr, St->getPointerInfo(), St->isVolatile(),
                        St->isNonTemporal(),
                        std::min(4U, St->getAlignment() / 2));
  }

  if (StVal.getValueType() != MVT::i64 ||
      StVal.getNode()->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
    return SDValue();

  // Bitcast an i64 store extracted from a vector to f64.
  // Otherwise, the i64 value will be legalized to a pair of i32 values.
  SelectionDAG &DAG = DCI.DAG;
  DebugLoc dl = StVal.getDebugLoc();
  SDValue IntVec = StVal.getOperand(0);
  EVT FloatVT = EVT::getVectorVT(*DAG.getContext(), MVT::f64,
                                 IntVec.getValueType().getVectorNumElements());
  SDValue Vec = DAG.getNode(ISD::BITCAST, dl, FloatVT, IntVec);
  SDValue ExtElt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64,
                               Vec, StVal.getOperand(1));
  dl = N->getDebugLoc();
  SDValue V = DAG.getNode(ISD::BITCAST, dl, MVT::i64, ExtElt);
  // Make the DAGCombiner fold the bitcasts.
  DCI.AddToWorklist(Vec.getNode());
  DCI.AddToWorklist(ExtElt.getNode());
  DCI.AddToWorklist(V.getNode());
  return DAG.getStore(St->getChain(), dl, V, St->getBasePtr(),
                      St->getPointerInfo(), St->isVolatile(),
                      St->isNonTemporal(), St->getAlignment(),
                      St->getTBAAInfo());
}

/// hasNormalLoadOperand - Check if any of the operands of a BUILD_VECTOR node
/// are normal, non-volatile loads.  If so, it is profitable to bitcast an
/// i64 vector to have f64 elements, since the value can then be loaded
/// directly into a VFP register.
static bool hasNormalLoadOperand(SDNode *N) {
  unsigned NumElts = N->getValueType(0).getVectorNumElements();
  for (unsigned i = 0; i < NumElts; ++i) {
    SDNode *Elt = N->getOperand(i).getNode();
    if (ISD::isNormalLoad(Elt) && !cast<LoadSDNode>(Elt)->isVolatile())
      return true;
  }
  return false;
}

/// PerformBUILD_VECTORCombine - Target-specific dag combine xforms for
/// ISD::BUILD_VECTOR.
static SDValue PerformBUILD_VECTORCombine(SDNode *N,
                                          TargetLowering::DAGCombinerInfo &DCI){
  // build_vector(N=ARMISD::VMOVRRD(X), N:1) -> bit_convert(X):
  // VMOVRRD is introduced when legalizing i64 types.  It forces the i64 value
  // into a pair of GPRs, which is fine when the value is used as a scalar,
  // but if the i64 value is converted to a vector, we need to undo the VMOVRRD.
  SelectionDAG &DAG = DCI.DAG;
  if (N->getNumOperands() == 2) {
    SDValue RV = PerformVMOVDRRCombine(N, DAG);
    if (RV.getNode())
      return RV;
  }

  // Load i64 elements as f64 values so that type legalization does not split
  // them up into i32 values.
  EVT VT = N->getValueType(0);
  if (VT.getVectorElementType() != MVT::i64 || !hasNormalLoadOperand(N))
    return SDValue();
  DebugLoc dl = N->getDebugLoc();
  SmallVector<SDValue, 8> Ops;
  unsigned NumElts = VT.getVectorNumElements();
  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue V = DAG.getNode(ISD::BITCAST, dl, MVT::f64, N->getOperand(i));
    Ops.push_back(V);
    // Make the DAGCombiner fold the bitcast.
    DCI.AddToWorklist(V.getNode());
  }
  EVT FloatVT = EVT::getVectorVT(*DAG.getContext(), MVT::f64, NumElts);
  SDValue BV = DAG.getNode(ISD::BUILD_VECTOR, dl, FloatVT, Ops.data(), NumElts);
  return DAG.getNode(ISD::BITCAST, dl, VT, BV);
}

/// PerformInsertEltCombine - Target-specific dag combine xforms for
/// ISD::INSERT_VECTOR_ELT.
static SDValue PerformInsertEltCombine(SDNode *N,
                                       TargetLowering::DAGCombinerInfo &DCI) {
  // Bitcast an i64 load inserted into a vector to f64.
  // Otherwise, the i64 value will be legalized to a pair of i32 values.
  EVT VT = N->getValueType(0);
  SDNode *Elt = N->getOperand(1).getNode();
  if (VT.getVectorElementType() != MVT::i64 ||
      !ISD::isNormalLoad(Elt) || cast<LoadSDNode>(Elt)->isVolatile())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  DebugLoc dl = N->getDebugLoc();
  EVT FloatVT = EVT::getVectorVT(*DAG.getContext(), MVT::f64,
                                 VT.getVectorNumElements());
  SDValue Vec = DAG.getNode(ISD::BITCAST, dl, FloatVT, N->getOperand(0));
  SDValue V = DAG.getNode(ISD::BITCAST, dl, MVT::f64, N->getOperand(1));
  // Make the DAGCombiner fold the bitcasts.
  DCI.AddToWorklist(Vec.getNode());
  DCI.AddToWorklist(V.getNode());
  SDValue InsElt = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, FloatVT,
                               Vec, V, N->getOperand(2));
  return DAG.getNode(ISD::BITCAST, dl, VT, InsElt);
}

/// PerformVECTOR_SHUFFLECombine - Target-specific dag combine xforms for
/// ISD::VECTOR_SHUFFLE.
static SDValue PerformVECTOR_SHUFFLECombine(SDNode *N, SelectionDAG &DAG) {
  // The LLVM shufflevector instruction does not require the shuffle mask
  // length to match the operand vector length, but ISD::VECTOR_SHUFFLE does
  // have that requirement.  When translating to ISD::VECTOR_SHUFFLE, if the
  // operands do not match the mask length, they are extended by concatenating
  // them with undef vectors.  That is probably the right thing for other
  // targets, but for NEON it is better to concatenate two double-register
  // size vector operands into a single quad-register size vector.  Do that
  // transformation here:
  //   shuffle(concat(v1, undef), concat(v2, undef)) ->
  //   shuffle(concat(v1, v2), undef)
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  if (Op0.getOpcode() != ISD::CONCAT_VECTORS ||
      Op1.getOpcode() != ISD::CONCAT_VECTORS ||
      Op0.getNumOperands() != 2 ||
      Op1.getNumOperands() != 2)
    return SDValue();
  SDValue Concat0Op1 = Op0.getOperand(1);
  SDValue Concat1Op1 = Op1.getOperand(1);
  if (Concat0Op1.getOpcode() != ISD::UNDEF ||
      Concat1Op1.getOpcode() != ISD::UNDEF)
    return SDValue();
  // Skip the transformation if any of the types are illegal.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT = N->getValueType(0);
  if (!TLI.isTypeLegal(VT) ||
      !TLI.isTypeLegal(Concat0Op1.getValueType()) ||
      !TLI.isTypeLegal(Concat1Op1.getValueType()))
    return SDValue();

  SDValue NewConcat = DAG.getNode(ISD::CONCAT_VECTORS, N->getDebugLoc(), VT,
                                  Op0.getOperand(0), Op1.getOperand(0));
  // Translate the shuffle mask.
  SmallVector<int, 16> NewMask;
  unsigned NumElts = VT.getVectorNumElements();
  unsigned HalfElts = NumElts/2;
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);
  for (unsigned n = 0; n < NumElts; ++n) {
    int MaskElt = SVN->getMaskElt(n);
    int NewElt = -1;
    if (MaskElt < (int)HalfElts)
      NewElt = MaskElt;
    else if (MaskElt >= (int)NumElts && MaskElt < (int)(NumElts + HalfElts))
      NewElt = HalfElts + MaskElt - NumElts;
    NewMask.push_back(NewElt);
  }
  return DAG.getVectorShuffle(VT, N->getDebugLoc(), NewConcat,
                              DAG.getUNDEF(VT), NewMask.data());
}

/// CombineBaseUpdate - Target-specific DAG combine function for VLDDUP and
/// NEON load/store intrinsics to merge base address updates.
static SDValue CombineBaseUpdate(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {
  if (DCI.isBeforeLegalize() || DCI.isCalledByLegalizer())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  bool isIntrinsic = (N->getOpcode() == ISD::INTRINSIC_VOID ||
                      N->getOpcode() == ISD::INTRINSIC_W_CHAIN);
  unsigned AddrOpIdx = (isIntrinsic ? 2 : 1);
  SDValue Addr = N->getOperand(AddrOpIdx);

  // Search for a use of the address operand that is an increment.
  for (SDNode::use_iterator UI = Addr.getNode()->use_begin(),
         UE = Addr.getNode()->use_end(); UI != UE; ++UI) {
    SDNode *User = *UI;
    if (User->getOpcode() != ISD::ADD ||
        UI.getUse().getResNo() != Addr.getResNo())
      continue;

    // Check that the add is independent of the load/store.  Otherwise, folding
    // it would create a cycle.
    if (User->isPredecessorOf(N) || N->isPredecessorOf(User))
      continue;

    // Find the new opcode for the updating load/store.
    bool isLoad = true;
    bool isLaneOp = false;
    unsigned NewOpc = 0;
    unsigned NumVecs = 0;
    if (isIntrinsic) {
      unsigned IntNo = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
      switch (IntNo) {
      default: llvm_unreachable("unexpected intrinsic for Neon base update");
      case Intrinsic::arm_neon_vld1:     NewOpc = ARMISD::VLD1_UPD;
        NumVecs = 1; break;
      case Intrinsic::arm_neon_vld2:     NewOpc = ARMISD::VLD2_UPD;
        NumVecs = 2; break;
      case Intrinsic::arm_neon_vld3:     NewOpc = ARMISD::VLD3_UPD;
        NumVecs = 3; break;
      case Intrinsic::arm_neon_vld4:     NewOpc = ARMISD::VLD4_UPD;
        NumVecs = 4; break;
      case Intrinsic::arm_neon_vld2lane: NewOpc = ARMISD::VLD2LN_UPD;
        NumVecs = 2; isLaneOp = true; break;
      case Intrinsic::arm_neon_vld3lane: NewOpc = ARMISD::VLD3LN_UPD;
        NumVecs = 3; isLaneOp = true; break;
      case Intrinsic::arm_neon_vld4lane: NewOpc = ARMISD::VLD4LN_UPD;
        NumVecs = 4; isLaneOp = true; break;
      case Intrinsic::arm_neon_vst1:     NewOpc = ARMISD::VST1_UPD;
        NumVecs = 1; isLoad = false; break;
      case Intrinsic::arm_neon_vst2:     NewOpc = ARMISD::VST2_UPD;
        NumVecs = 2; isLoad = false; break;
      case Intrinsic::arm_neon_vst3:     NewOpc = ARMISD::VST3_UPD;
        NumVecs = 3; isLoad = false; break;
      case Intrinsic::arm_neon_vst4:     NewOpc = ARMISD::VST4_UPD;
        NumVecs = 4; isLoad = false; break;
      case Intrinsic::arm_neon_vst2lane: NewOpc = ARMISD::VST2LN_UPD;
        NumVecs = 2; isLoad = false; isLaneOp = true; break;
      case Intrinsic::arm_neon_vst3lane: NewOpc = ARMISD::VST3LN_UPD;
        NumVecs = 3; isLoad = false; isLaneOp = true; break;
      case Intrinsic::arm_neon_vst4lane: NewOpc = ARMISD::VST4LN_UPD;
        NumVecs = 4; isLoad = false; isLaneOp = true; break;
      }
    } else {
      isLaneOp = true;
      switch (N->getOpcode()) {
      default: llvm_unreachable("unexpected opcode for Neon base update");
      case ARMISD::VLD2DUP: NewOpc = ARMISD::VLD2DUP_UPD; NumVecs = 2; break;
      case ARMISD::VLD3DUP: NewOpc = ARMISD::VLD3DUP_UPD; NumVecs = 3; break;
      case ARMISD::VLD4DUP: NewOpc = ARMISD::VLD4DUP_UPD; NumVecs = 4; break;
      }
    }

    // Find the size of memory referenced by the load/store.
    EVT VecTy;
    if (isLoad)
      VecTy = N->getValueType(0);
    else
      VecTy = N->getOperand(AddrOpIdx+1).getValueType();
    unsigned NumBytes = NumVecs * VecTy.getSizeInBits() / 8;
    if (isLaneOp)
      NumBytes /= VecTy.getVectorNumElements();

    // If the increment is a constant, it must match the memory ref size.
    SDValue Inc = User->getOperand(User->getOperand(0) == Addr ? 1 : 0);
    if (ConstantSDNode *CInc = dyn_cast<ConstantSDNode>(Inc.getNode())) {
      uint64_t IncVal = CInc->getZExtValue();
      if (IncVal != NumBytes)
        continue;
    } else if (NumBytes >= 3 * 16) {
      // VLD3/4 and VST3/4 for 128-bit vectors are implemented with two
      // separate instructions that make it harder to use a non-constant update.
      continue;
    }

    // Create the new updating load/store node.
    EVT Tys[6];
    unsigned NumResultVecs = (isLoad ? NumVecs : 0);
    unsigned n;
    for (n = 0; n < NumResultVecs; ++n)
      Tys[n] = VecTy;
    Tys[n++] = MVT::i32;
    Tys[n] = MVT::Other;
    SDVTList SDTys = DAG.getVTList(Tys, NumResultVecs+2);
    SmallVector<SDValue, 8> Ops;
    Ops.push_back(N->getOperand(0)); // incoming chain
    Ops.push_back(N->getOperand(AddrOpIdx));
    Ops.push_back(Inc);
    for (unsigned i = AddrOpIdx + 1; i < N->getNumOperands(); ++i) {
      Ops.push_back(N->getOperand(i));
    }
    MemIntrinsicSDNode *MemInt = cast<MemIntrinsicSDNode>(N);
    SDValue UpdN = DAG.getMemIntrinsicNode(NewOpc, N->getDebugLoc(), SDTys,
                                           Ops.data(), Ops.size(),
                                           MemInt->getMemoryVT(),
                                           MemInt->getMemOperand());

    // Update the uses.
    std::vector<SDValue> NewResults;
    for (unsigned i = 0; i < NumResultVecs; ++i) {
      NewResults.push_back(SDValue(UpdN.getNode(), i));
    }
    NewResults.push_back(SDValue(UpdN.getNode(), NumResultVecs+1)); // chain
    DCI.CombineTo(N, NewResults);
    DCI.CombineTo(User, SDValue(UpdN.getNode(), NumResultVecs));

    break;
  }
  return SDValue();
}

/// CombineVLDDUP - For a VDUPLANE node N, check if its source operand is a
/// vldN-lane (N > 1) intrinsic, and if all the other uses of that intrinsic
/// are also VDUPLANEs.  If so, combine them to a vldN-dup operation and
/// return true.
static bool CombineVLDDUP(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  // vldN-dup instructions only support 64-bit vectors for N > 1.
  if (!VT.is64BitVector())
    return false;

  // Check if the VDUPLANE operand is a vldN-dup intrinsic.
  SDNode *VLD = N->getOperand(0).getNode();
  if (VLD->getOpcode() != ISD::INTRINSIC_W_CHAIN)
    return false;
  unsigned NumVecs = 0;
  unsigned NewOpc = 0;
  unsigned IntNo = cast<ConstantSDNode>(VLD->getOperand(1))->getZExtValue();
  if (IntNo == Intrinsic::arm_neon_vld2lane) {
    NumVecs = 2;
    NewOpc = ARMISD::VLD2DUP;
  } else if (IntNo == Intrinsic::arm_neon_vld3lane) {
    NumVecs = 3;
    NewOpc = ARMISD::VLD3DUP;
  } else if (IntNo == Intrinsic::arm_neon_vld4lane) {
    NumVecs = 4;
    NewOpc = ARMISD::VLD4DUP;
  } else {
    return false;
  }

  // First check that all the vldN-lane uses are VDUPLANEs and that the lane
  // numbers match the load.
  unsigned VLDLaneNo =
    cast<ConstantSDNode>(VLD->getOperand(NumVecs+3))->getZExtValue();
  for (SDNode::use_iterator UI = VLD->use_begin(), UE = VLD->use_end();
       UI != UE; ++UI) {
    // Ignore uses of the chain result.
    if (UI.getUse().getResNo() == NumVecs)
      continue;
    SDNode *User = *UI;
    if (User->getOpcode() != ARMISD::VDUPLANE ||
        VLDLaneNo != cast<ConstantSDNode>(User->getOperand(1))->getZExtValue())
      return false;
  }

  // Create the vldN-dup node.
  EVT Tys[5];
  unsigned n;
  for (n = 0; n < NumVecs; ++n)
    Tys[n] = VT;
  Tys[n] = MVT::Other;
  SDVTList SDTys = DAG.getVTList(Tys, NumVecs+1);
  SDValue Ops[] = { VLD->getOperand(0), VLD->getOperand(2) };
  MemIntrinsicSDNode *VLDMemInt = cast<MemIntrinsicSDNode>(VLD);
  SDValue VLDDup = DAG.getMemIntrinsicNode(NewOpc, VLD->getDebugLoc(), SDTys,
                                           Ops, 2, VLDMemInt->getMemoryVT(),
                                           VLDMemInt->getMemOperand());

  // Update the uses.
  for (SDNode::use_iterator UI = VLD->use_begin(), UE = VLD->use_end();
       UI != UE; ++UI) {
    unsigned ResNo = UI.getUse().getResNo();
    // Ignore uses of the chain result.
    if (ResNo == NumVecs)
      continue;
    SDNode *User = *UI;
    DCI.CombineTo(User, SDValue(VLDDup.getNode(), ResNo));
  }

  // Now the vldN-lane intrinsic is dead except for its chain result.
  // Update uses of the chain.
  std::vector<SDValue> VLDDupResults;
  for (unsigned n = 0; n < NumVecs; ++n)
    VLDDupResults.push_back(SDValue(VLDDup.getNode(), n));
  VLDDupResults.push_back(SDValue(VLDDup.getNode(), NumVecs));
  DCI.CombineTo(VLD, VLDDupResults);

  return true;
}

/// PerformVDUPLANECombine - Target-specific dag combine xforms for
/// ARMISD::VDUPLANE.
static SDValue PerformVDUPLANECombine(SDNode *N,
                                      TargetLowering::DAGCombinerInfo &DCI) {
  SDValue Op = N->getOperand(0);

  // If the source is a vldN-lane (N > 1) intrinsic, and all the other uses
  // of that intrinsic are also VDUPLANEs, combine them to a vldN-dup operation.
  if (CombineVLDDUP(N, DCI))
    return SDValue(N, 0);

  // If the source is already a VMOVIMM or VMVNIMM splat, the VDUPLANE is
  // redundant.  Ignore bit_converts for now; element sizes are checked below.
  while (Op.getOpcode() == ISD::BITCAST)
    Op = Op.getOperand(0);
  if (Op.getOpcode() != ARMISD::VMOVIMM && Op.getOpcode() != ARMISD::VMVNIMM)
    return SDValue();

  // Make sure the VMOV element size is not bigger than the VDUPLANE elements.
  unsigned EltSize = Op.getValueType().getVectorElementType().getSizeInBits();
  // The canonical VMOV for a zero vector uses a 32-bit element size.
  unsigned Imm = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  unsigned EltBits;
  if (ARM_AM::decodeNEONModImm(Imm, EltBits) == 0)
    EltSize = 8;
  EVT VT = N->getValueType(0);
  if (EltSize > VT.getVectorElementType().getSizeInBits())
    return SDValue();

  return DCI.DAG.getNode(ISD::BITCAST, N->getDebugLoc(), VT, Op);
}

// isConstVecPow2 - Return true if each vector element is a power of 2, all
// elements are the same constant, C, and Log2(C) ranges from 1 to 32.
static bool isConstVecPow2(SDValue ConstVec, bool isSigned, uint64_t &C)
{
  integerPart cN;
  integerPart c0 = 0;
  for (unsigned I = 0, E = ConstVec.getValueType().getVectorNumElements();
       I != E; I++) {
    ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(ConstVec.getOperand(I));
    if (!C)
      return false;

    bool isExact;
    APFloat APF = C->getValueAPF();
    if (APF.convertToInteger(&cN, 64, isSigned, APFloat::rmTowardZero, &isExact)
        != APFloat::opOK || !isExact)
      return false;

    c0 = (I == 0) ? cN : c0;
    if (!isPowerOf2_64(cN) || c0 != cN || Log2_64(c0) < 1 || Log2_64(c0) > 32)
      return false;
  }
  C = c0;
  return true;
}

/// PerformVCVTCombine - VCVT (floating-point to fixed-point, Advanced SIMD)
/// can replace combinations of VMUL and VCVT (floating-point to integer)
/// when the VMUL has a constant operand that is a power of 2.
///
/// Example (assume d17 = <float 8.000000e+00, float 8.000000e+00>):
///  vmul.f32        d16, d17, d16
///  vcvt.s32.f32    d16, d16
/// becomes:
///  vcvt.s32.f32    d16, d16, #3
static SDValue PerformVCVTCombine(SDNode *N,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const ARMSubtarget *Subtarget) {
  SelectionDAG &DAG = DCI.DAG;
  SDValue Op = N->getOperand(0);

  if (!Subtarget->hasNEON() || !Op.getValueType().isVector() ||
      Op.getOpcode() != ISD::FMUL)
    return SDValue();

  uint64_t C;
  SDValue N0 = Op->getOperand(0);
  SDValue ConstVec = Op->getOperand(1);
  bool isSigned = N->getOpcode() == ISD::FP_TO_SINT;

  if (ConstVec.getOpcode() != ISD::BUILD_VECTOR ||
      !isConstVecPow2(ConstVec, isSigned, C))
    return SDValue();

  unsigned IntrinsicOpcode = isSigned ? Intrinsic::arm_neon_vcvtfp2fxs :
    Intrinsic::arm_neon_vcvtfp2fxu;
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, N->getDebugLoc(),
                     N->getValueType(0),
                     DAG.getConstant(IntrinsicOpcode, MVT::i32), N0,
                     DAG.getConstant(Log2_64(C), MVT::i32));
}

/// PerformVDIVCombine - VCVT (fixed-point to floating-point, Advanced SIMD)
/// can replace combinations of VCVT (integer to floating-point) and VDIV
/// when the VDIV has a constant operand that is a power of 2.
///
/// Example (assume d17 = <float 8.000000e+00, float 8.000000e+00>):
///  vcvt.f32.s32    d16, d16
///  vdiv.f32        d16, d17, d16
/// becomes:
///  vcvt.f32.s32    d16, d16, #3
static SDValue PerformVDIVCombine(SDNode *N,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const ARMSubtarget *Subtarget) {
  SelectionDAG &DAG = DCI.DAG;
  SDValue Op = N->getOperand(0);
  unsigned OpOpcode = Op.getNode()->getOpcode();

  if (!Subtarget->hasNEON() || !N->getValueType(0).isVector() ||
      (OpOpcode != ISD::SINT_TO_FP && OpOpcode != ISD::UINT_TO_FP))
    return SDValue();

  uint64_t C;
  SDValue ConstVec = N->getOperand(1);
  bool isSigned = OpOpcode == ISD::SINT_TO_FP;

  if (ConstVec.getOpcode() != ISD::BUILD_VECTOR ||
      !isConstVecPow2(ConstVec, isSigned, C))
    return SDValue();

  unsigned IntrinsicOpcode = isSigned ? Intrinsic::arm_neon_vcvtfxs2fp :
    Intrinsic::arm_neon_vcvtfxu2fp;
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, N->getDebugLoc(),
                     Op.getValueType(),
                     DAG.getConstant(IntrinsicOpcode, MVT::i32),
                     Op.getOperand(0), DAG.getConstant(Log2_64(C), MVT::i32));
}

/// Getvshiftimm - Check if this is a valid build_vector for the immediate
/// operand of a vector shift operation, where all the elements of the
/// build_vector must have the same constant integer value.
static bool getVShiftImm(SDValue Op, unsigned ElementBits, int64_t &Cnt) {
  // Ignore bit_converts.
  while (Op.getOpcode() == ISD::BITCAST)
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
      llvm_unreachable("invalid shift count for narrowing vector shift "
                       "intrinsic");

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
  if (N->getOpcode() == ISD::SRL && VT == MVT::i32 && ST->hasV6Ops()) {
    // Canonicalize (srl (bswap x), 16) to (rotr (bswap x), 16) if the high
    // 16-bits of x is zero. This optimizes rev + lsr 16 to rev16.
    SDValue N1 = N->getOperand(1);
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N1)) {
      SDValue N0 = N->getOperand(0);
      if (C->getZExtValue() == 16 && N0.getOpcode() == ISD::BSWAP &&
          DAG.MaskedValueIsZero(N0.getOperand(0),
                                APInt::getHighBitsSet(32, 16)))
        return DAG.getNode(ISD::ROTR, N->getDebugLoc(), VT, N0, N1);
    }
  }

  // Nothing to be done for scalar shifts.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!VT.isVector() || !TLI.isTypeLegal(VT))
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
        TLI.isTypeLegal(Vec.getValueType()) &&
        isa<ConstantSDNode>(Lane)) {

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

/// PerformSELECT_CCCombine - Target-specific DAG combining for ISD::SELECT_CC
/// to match f32 max/min patterns to use NEON vmax/vmin instructions.
static SDValue PerformSELECT_CCCombine(SDNode *N, SelectionDAG &DAG,
                                       const ARMSubtarget *ST) {
  // If the target supports NEON, try to use vmax/vmin instructions for f32
  // selects like "x < y ? x : y".  Unless the NoNaNsFPMath option is set,
  // be careful about NaNs:  NEON's vmax/vmin return NaN if either operand is
  // a NaN; only do the transformation when it matches that behavior.

  // For now only do this when using NEON for FP operations; if using VFP, it
  // is not obvious that the benefit outweighs the cost of switching to the
  // NEON pipeline.
  if (!ST->hasNEON() || !ST->useNEONForSinglePrecisionFP() ||
      N->getValueType(0) != MVT::f32)
    return SDValue();

  SDValue CondLHS = N->getOperand(0);
  SDValue CondRHS = N->getOperand(1);
  SDValue LHS = N->getOperand(2);
  SDValue RHS = N->getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(4))->get();

  unsigned Opcode = 0;
  bool IsReversed;
  if (DAG.isEqualTo(LHS, CondLHS) && DAG.isEqualTo(RHS, CondRHS)) {
    IsReversed = false; // x CC y ? x : y
  } else if (DAG.isEqualTo(LHS, CondRHS) && DAG.isEqualTo(RHS, CondLHS)) {
    IsReversed = true ; // x CC y ? y : x
  } else {
    return SDValue();
  }

  bool IsUnordered;
  switch (CC) {
  default: break;
  case ISD::SETOLT:
  case ISD::SETOLE:
  case ISD::SETLT:
  case ISD::SETLE:
  case ISD::SETULT:
  case ISD::SETULE:
    // If LHS is NaN, an ordered comparison will be false and the result will
    // be the RHS, but vmin(NaN, RHS) = NaN.  Avoid this by checking that LHS
    // != NaN.  Likewise, for unordered comparisons, check for RHS != NaN.
    IsUnordered = (CC == ISD::SETULT || CC == ISD::SETULE);
    if (!DAG.isKnownNeverNaN(IsUnordered ? RHS : LHS))
      break;
    // For less-than-or-equal comparisons, "+0 <= -0" will be true but vmin
    // will return -0, so vmin can only be used for unsafe math or if one of
    // the operands is known to be nonzero.
    if ((CC == ISD::SETLE || CC == ISD::SETOLE || CC == ISD::SETULE) &&
        !DAG.getTarget().Options.UnsafeFPMath &&
        !(DAG.isKnownNeverZero(LHS) || DAG.isKnownNeverZero(RHS)))
      break;
    Opcode = IsReversed ? ARMISD::FMAX : ARMISD::FMIN;
    break;

  case ISD::SETOGT:
  case ISD::SETOGE:
  case ISD::SETGT:
  case ISD::SETGE:
  case ISD::SETUGT:
  case ISD::SETUGE:
    // If LHS is NaN, an ordered comparison will be false and the result will
    // be the RHS, but vmax(NaN, RHS) = NaN.  Avoid this by checking that LHS
    // != NaN.  Likewise, for unordered comparisons, check for RHS != NaN.
    IsUnordered = (CC == ISD::SETUGT || CC == ISD::SETUGE);
    if (!DAG.isKnownNeverNaN(IsUnordered ? RHS : LHS))
      break;
    // For greater-than-or-equal comparisons, "-0 >= +0" will be true but vmax
    // will return +0, so vmax can only be used for unsafe math or if one of
    // the operands is known to be nonzero.
    if ((CC == ISD::SETGE || CC == ISD::SETOGE || CC == ISD::SETUGE) &&
        !DAG.getTarget().Options.UnsafeFPMath &&
        !(DAG.isKnownNeverZero(LHS) || DAG.isKnownNeverZero(RHS)))
      break;
    Opcode = IsReversed ? ARMISD::FMIN : ARMISD::FMAX;
    break;
  }

  if (!Opcode)
    return SDValue();
  return DAG.getNode(Opcode, N->getDebugLoc(), N->getValueType(0), LHS, RHS);
}

/// PerformCMOVCombine - Target-specific DAG combining for ARMISD::CMOV.
SDValue
ARMTargetLowering::PerformCMOVCombine(SDNode *N, SelectionDAG &DAG) const {
  SDValue Cmp = N->getOperand(4);
  if (Cmp.getOpcode() != ARMISD::CMPZ)
    // Only looking at EQ and NE cases.
    return SDValue();

  EVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();
  SDValue LHS = Cmp.getOperand(0);
  SDValue RHS = Cmp.getOperand(1);
  SDValue FalseVal = N->getOperand(0);
  SDValue TrueVal = N->getOperand(1);
  SDValue ARMcc = N->getOperand(2);
  ARMCC::CondCodes CC =
    (ARMCC::CondCodes)cast<ConstantSDNode>(ARMcc)->getZExtValue();

  // Simplify
  //   mov     r1, r0
  //   cmp     r1, x
  //   mov     r0, y
  //   moveq   r0, x
  // to
  //   cmp     r0, x
  //   movne   r0, y
  //
  //   mov     r1, r0
  //   cmp     r1, x
  //   mov     r0, x
  //   movne   r0, y
  // to
  //   cmp     r0, x
  //   movne   r0, y
  /// FIXME: Turn this into a target neutral optimization?
  SDValue Res;
  if (CC == ARMCC::NE && FalseVal == RHS && FalseVal != LHS) {
    Res = DAG.getNode(ARMISD::CMOV, dl, VT, LHS, TrueVal, ARMcc,
                      N->getOperand(3), Cmp);
  } else if (CC == ARMCC::EQ && TrueVal == RHS) {
    SDValue ARMcc;
    SDValue NewCmp = getARMCmp(LHS, RHS, ISD::SETNE, ARMcc, DAG, dl);
    Res = DAG.getNode(ARMISD::CMOV, dl, VT, LHS, FalseVal, ARMcc,
                      N->getOperand(3), NewCmp);
  }

  if (Res.getNode()) {
    APInt KnownZero, KnownOne;
    DAG.ComputeMaskedBits(SDValue(N,0), KnownZero, KnownOne);
    // Capture demanded bits information that would be otherwise lost.
    if (KnownZero == 0xfffffffe)
      Res = DAG.getNode(ISD::AssertZext, dl, MVT::i32, Res,
                        DAG.getValueType(MVT::i1));
    else if (KnownZero == 0xffffff00)
      Res = DAG.getNode(ISD::AssertZext, dl, MVT::i32, Res,
                        DAG.getValueType(MVT::i8));
    else if (KnownZero == 0xffff0000)
      Res = DAG.getNode(ISD::AssertZext, dl, MVT::i32, Res,
                        DAG.getValueType(MVT::i16));
  }

  return Res;
}

SDValue ARMTargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  default: break;
  case ISD::ADDC:       return PerformADDCCombine(N, DCI, Subtarget);
  case ISD::ADD:        return PerformADDCombine(N, DCI, Subtarget);
  case ISD::SUB:        return PerformSUBCombine(N, DCI);
  case ISD::MUL:        return PerformMULCombine(N, DCI, Subtarget);
  case ISD::OR:         return PerformORCombine(N, DCI, Subtarget);
  case ISD::XOR:        return PerformXORCombine(N, DCI, Subtarget);
  case ISD::AND:        return PerformANDCombine(N, DCI, Subtarget);
  case ARMISD::BFI:     return PerformBFICombine(N, DCI);
  case ARMISD::VMOVRRD: return PerformVMOVRRDCombine(N, DCI);
  case ARMISD::VMOVDRR: return PerformVMOVDRRCombine(N, DCI.DAG);
  case ISD::STORE:      return PerformSTORECombine(N, DCI);
  case ISD::BUILD_VECTOR: return PerformBUILD_VECTORCombine(N, DCI);
  case ISD::INSERT_VECTOR_ELT: return PerformInsertEltCombine(N, DCI);
  case ISD::VECTOR_SHUFFLE: return PerformVECTOR_SHUFFLECombine(N, DCI.DAG);
  case ARMISD::VDUPLANE: return PerformVDUPLANECombine(N, DCI);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT: return PerformVCVTCombine(N, DCI, Subtarget);
  case ISD::FDIV:       return PerformVDIVCombine(N, DCI, Subtarget);
  case ISD::INTRINSIC_WO_CHAIN: return PerformIntrinsicCombine(N, DCI.DAG);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:        return PerformShiftCombine(N, DCI.DAG, Subtarget);
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND: return PerformExtendCombine(N, DCI.DAG, Subtarget);
  case ISD::SELECT_CC:  return PerformSELECT_CCCombine(N, DCI.DAG, Subtarget);
  case ARMISD::CMOV: return PerformCMOVCombine(N, DCI.DAG);
  case ARMISD::VLD2DUP:
  case ARMISD::VLD3DUP:
  case ARMISD::VLD4DUP:
    return CombineBaseUpdate(N, DCI);
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN:
    switch (cast<ConstantSDNode>(N->getOperand(1))->getZExtValue()) {
    case Intrinsic::arm_neon_vld1:
    case Intrinsic::arm_neon_vld2:
    case Intrinsic::arm_neon_vld3:
    case Intrinsic::arm_neon_vld4:
    case Intrinsic::arm_neon_vld2lane:
    case Intrinsic::arm_neon_vld3lane:
    case Intrinsic::arm_neon_vld4lane:
    case Intrinsic::arm_neon_vst1:
    case Intrinsic::arm_neon_vst2:
    case Intrinsic::arm_neon_vst3:
    case Intrinsic::arm_neon_vst4:
    case Intrinsic::arm_neon_vst2lane:
    case Intrinsic::arm_neon_vst3lane:
    case Intrinsic::arm_neon_vst4lane:
      return CombineBaseUpdate(N, DCI);
    default: break;
    }
    break;
  }
  return SDValue();
}

bool ARMTargetLowering::isDesirableToTransformToIntegerOp(unsigned Opc,
                                                          EVT VT) const {
  return (VT == MVT::f32) && (Opc == ISD::LOAD || Opc == ISD::STORE);
}

bool ARMTargetLowering::allowsUnalignedMemoryAccesses(EVT VT, bool *Fast) const {
  // The AllowsUnaliged flag models the SCTLR.A setting in ARM cpus
  bool AllowsUnaligned = Subtarget->allowsUnalignedMem();

  switch (VT.getSimpleVT().SimpleTy) {
  default:
    return false;
  case MVT::i8:
  case MVT::i16:
  case MVT::i32: {
    // Unaligned access can use (for example) LRDB, LRDH, LDR
    if (AllowsUnaligned) {
      if (Fast)
        *Fast = Subtarget->hasV7Ops();
      return true;
    }
    return false;
  }
  case MVT::f64:
  case MVT::v2f64: {
    // For any little-endian targets with neon, we can support unaligned ld/st
    // of D and Q (e.g. {D0,D1}) registers by using vld1.i8/vst1.i8.
    // A big-endian target may also explictly support unaligned accesses
    if (Subtarget->hasNEON() && (AllowsUnaligned || isLittleEndian())) {
      if (Fast)
        *Fast = true;
      return true;
    }
    return false;
  }
  }
}

static bool memOpAlign(unsigned DstAlign, unsigned SrcAlign,
                       unsigned AlignCheck) {
  return ((SrcAlign == 0 || SrcAlign % AlignCheck == 0) &&
          (DstAlign == 0 || DstAlign % AlignCheck == 0));
}

EVT ARMTargetLowering::getOptimalMemOpType(uint64_t Size,
                                           unsigned DstAlign, unsigned SrcAlign,
                                           bool IsMemset, bool ZeroMemset,
                                           bool MemcpyStrSrc,
                                           MachineFunction &MF) const {
  const Function *F = MF.getFunction();

  // See if we can use NEON instructions for this...
  if ((!IsMemset || ZeroMemset) &&
      Subtarget->hasNEON() &&
      !F->getAttributes().hasAttribute(AttributeSet::FunctionIndex,
                                       Attribute::NoImplicitFloat)) {
    bool Fast;
    if (Size >= 16 &&
        (memOpAlign(SrcAlign, DstAlign, 16) ||
         (allowsUnalignedMemoryAccesses(MVT::v2f64, &Fast) && Fast))) {
      return MVT::v2f64;
    } else if (Size >= 8 &&
               (memOpAlign(SrcAlign, DstAlign, 8) ||
                (allowsUnalignedMemoryAccesses(MVT::f64, &Fast) && Fast))) {
      return MVT::f64;
    }
  }

  // Lowering to i32/i16 if the size permits.
  if (Size >= 4)
    return MVT::i32;
  else if (Size >= 2)
    return MVT::i16;

  // Let the target-independent logic figure it out.
  return MVT::Other;
}

bool ARMTargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  if (Val.getOpcode() != ISD::LOAD)
    return false;

  EVT VT1 = Val.getValueType();
  if (!VT1.isSimple() || !VT1.isInteger() ||
      !VT2.isSimple() || !VT2.isInteger())
    return false;

  switch (VT1.getSimpleVT().SimpleTy) {
  default: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
    // 8-bit and 16-bit loads implicitly zero-extend to 32-bits.
    return true;
  }

  return false;
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
                                              Type *Ty) const {
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
  }
  return true;
}

/// isLegalICmpImmediate - Return true if the specified immediate is legal
/// icmp immediate, that is the target has icmp instructions which can compare
/// a register against the immediate without having to materialize the
/// immediate into a register.
bool ARMTargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  // Thumb2 and ARM modes can use cmn for negative immediates.
  if (!Subtarget->isThumb())
    return ARM_AM::getSOImmVal(llvm::abs64(Imm)) != -1;
  if (Subtarget->isThumb2())
    return ARM_AM::getT2SOImmVal(llvm::abs64(Imm)) != -1;
  // Thumb1 doesn't have cmn, and only 8-bit immediates.
  return Imm >= 0 && Imm <= 255;
}

/// isLegalAddImmediate - Return true if the specified immediate is a legal add
/// *or sub* immediate, that is the target has add or sub instructions which can
/// add a register with the immediate without having to materialize the
/// immediate into a register.
bool ARMTargetLowering::isLegalAddImmediate(int64_t Imm) const {
  // Same encoding for add/sub, just flip the sign.
  int64_t AbsImm = llvm::abs64(Imm);
  if (!Subtarget->isThumb())
    return ARM_AM::getSOImmVal(AbsImm) != -1;
  if (Subtarget->isThumb2())
    return ARM_AM::getT2SOImmVal(AbsImm) != -1;
  // Thumb1 only has 8-bit unsigned immediate.
  return AbsImm >= 0 && AbsImm <= 255;
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
      ARM_AM::ShiftOpc ShOpcVal=
        ARM_AM::getShiftOpcForNode(Ptr->getOperand(0).getOpcode());
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

  // FIXME: Use VLDM / VSTM to emulate indexed FP load / store.
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
    Ptr = LD->getBasePtr();
    isSEXTLoad = LD->getExtensionType() == ISD::SEXTLOAD;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    VT  = ST->getMemoryVT();
    Ptr = ST->getBasePtr();
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

  if (Ptr != Base) {
    // Swap base ptr and offset to catch more post-index load / store when
    // it's legal. In Thumb2 mode, offset must be an immediate.
    if (Ptr == Offset && Op->getOpcode() == ISD::ADD &&
        !Subtarget->isThumb2())
      std::swap(Base, Offset);

    // Post-indexed load / store update the base pointer.
    if (Ptr != Base)
      return false;
  }

  AM = isInc ? ISD::POST_INC : ISD::POST_DEC;
  return true;
}

void ARMTargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                       APInt &KnownZero,
                                                       APInt &KnownOne,
                                                       const SelectionDAG &DAG,
                                                       unsigned Depth) const {
  KnownZero = KnownOne = APInt(KnownOne.getBitWidth(), 0);
  switch (Op.getOpcode()) {
  default: break;
  case ARMISD::CMOV: {
    // Bits are known zero/one if known on the LHS and RHS.
    DAG.ComputeMaskedBits(Op.getOperand(0), KnownZero, KnownOne, Depth+1);
    if (KnownZero == 0 && KnownOne == 0) return;

    APInt KnownZeroRHS, KnownOneRHS;
    DAG.ComputeMaskedBits(Op.getOperand(1), KnownZeroRHS, KnownOneRHS, Depth+1);
    KnownZero &= KnownZeroRHS;
    KnownOne  &= KnownOneRHS;
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
//                           ARM Inline Assembly Support
//===----------------------------------------------------------------------===//

bool ARMTargetLowering::ExpandInlineAsm(CallInst *CI) const {
  // Looking for "rev" which is V6+.
  if (!Subtarget->hasV6Ops())
    return false;

  InlineAsm *IA = cast<InlineAsm>(CI->getCalledValue());
  std::string AsmStr = IA->getAsmString();
  SmallVector<StringRef, 4> AsmPieces;
  SplitString(AsmStr, AsmPieces, ";\n");

  switch (AsmPieces.size()) {
  default: return false;
  case 1:
    AsmStr = AsmPieces[0];
    AsmPieces.clear();
    SplitString(AsmStr, AsmPieces, " \t,");

    // rev $0, $1
    if (AsmPieces.size() == 3 &&
        AsmPieces[0] == "rev" && AsmPieces[1] == "$0" && AsmPieces[2] == "$1" &&
        IA->getConstraintString().compare(0, 4, "=l,l") == 0) {
      IntegerType *Ty = dyn_cast<IntegerType>(CI->getType());
      if (Ty && Ty->getBitWidth() == 32)
        return IntrinsicLowering::LowerToByteSwap(CI);
    }
    break;
  }

  return false;
}

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
ARMTargetLowering::ConstraintType
ARMTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:  break;
    case 'l': return C_RegisterClass;
    case 'w': return C_RegisterClass;
    case 'h': return C_RegisterClass;
    case 'x': return C_RegisterClass;
    case 't': return C_RegisterClass;
    case 'j': return C_Other; // Constant for movw.
      // An address with a single base register. Due to the way we
      // currently handle addresses it is the same as an 'r' memory constraint.
    case 'Q': return C_Memory;
    }
  } else if (Constraint.size() == 2) {
    switch (Constraint[0]) {
    default: break;
    // All 'U+' constraints are addresses.
    case 'U': return C_Memory;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
ARMTargetLowering::getSingleConstraintMatchWeight(
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
  case 'l':
    if (type->isIntegerTy()) {
      if (Subtarget->isThumb())
        weight = CW_SpecificReg;
      else
        weight = CW_Register;
    }
    break;
  case 'w':
    if (type->isFloatingPointTy())
      weight = CW_Register;
    break;
  }
  return weight;
}

typedef std::pair<unsigned, const TargetRegisterClass*> RCPair;
RCPair
ARMTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                EVT VT) const {
  if (Constraint.size() == 1) {
    // GCC ARM Constraint Letters
    switch (Constraint[0]) {
    case 'l': // Low regs or general regs.
      if (Subtarget->isThumb())
        return RCPair(0U, &ARM::tGPRRegClass);
      return RCPair(0U, &ARM::GPRRegClass);
    case 'h': // High regs or no regs.
      if (Subtarget->isThumb())
        return RCPair(0U, &ARM::hGPRRegClass);
      break;
    case 'r':
      return RCPair(0U, &ARM::GPRRegClass);
    case 'w':
      if (VT == MVT::f32)
        return RCPair(0U, &ARM::SPRRegClass);
      if (VT.getSizeInBits() == 64)
        return RCPair(0U, &ARM::DPRRegClass);
      if (VT.getSizeInBits() == 128)
        return RCPair(0U, &ARM::QPRRegClass);
      break;
    case 'x':
      if (VT == MVT::f32)
        return RCPair(0U, &ARM::SPR_8RegClass);
      if (VT.getSizeInBits() == 64)
        return RCPair(0U, &ARM::DPR_8RegClass);
      if (VT.getSizeInBits() == 128)
        return RCPair(0U, &ARM::QPR_8RegClass);
      break;
    case 't':
      if (VT == MVT::f32)
        return RCPair(0U, &ARM::SPRRegClass);
      break;
    }
  }
  if (StringRef("{cc}").equals_lower(Constraint))
    return std::make_pair(unsigned(ARM::CPSR), &ARM::CCRRegClass);

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void ARMTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                     std::string &Constraint,
                                                     std::vector<SDValue>&Ops,
                                                     SelectionDAG &DAG) const {
  SDValue Result(0, 0);

  // Currently only support length 1 constraints.
  if (Constraint.length() != 1) return;

  char ConstraintLetter = Constraint[0];
  switch (ConstraintLetter) {
  default: break;
  case 'j':
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

    switch (ConstraintLetter) {
      case 'j':
        // Constant suitable for movw, must be between 0 and
        // 65535.
        if (Subtarget->hasV6T2Ops())
          if (CVal >= 0 && CVal <= 65535)
            break;
        return;
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
  return TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

bool
ARMTargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The ARM target isn't yet aware of offsets.
  return false;
}

bool ARM::isBitFieldInvertedMask(unsigned v) {
  if (v == 0xffffffff)
    return 0;
  // there can be 1's on either or both "outsides", all the "inside"
  // bits must be 0's
  unsigned int lsb = 0, msb = 31;
  while (v & (1 << msb)) --msb;
  while (v & (1 << lsb)) ++lsb;
  for (unsigned int i = lsb; i <= msb; ++i) {
    if (v & (1 << i))
      return 0;
  }
  return 1;
}

/// isFPImmLegal - Returns true if the target can instruction select the
/// specified FP immediate natively. If false, the legalizer will
/// materialize the FP immediate as a load from a constant pool.
bool ARMTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  if (!Subtarget->hasVFP3())
    return false;
  if (VT == MVT::f32)
    return ARM_AM::getFP32Imm(Imm) != -1;
  if (VT == MVT::f64)
    return ARM_AM::getFP64Imm(Imm) != -1;
  return false;
}

/// getTgtMemIntrinsic - Represent NEON load and store intrinsics as
/// MemIntrinsicNodes.  The associated MachineMemOperands record the alignment
/// specified in the intrinsic calls.
bool ARMTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                           const CallInst &I,
                                           unsigned Intrinsic) const {
  switch (Intrinsic) {
  case Intrinsic::arm_neon_vld1:
  case Intrinsic::arm_neon_vld2:
  case Intrinsic::arm_neon_vld3:
  case Intrinsic::arm_neon_vld4:
  case Intrinsic::arm_neon_vld2lane:
  case Intrinsic::arm_neon_vld3lane:
  case Intrinsic::arm_neon_vld4lane: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    // Conservatively set memVT to the entire set of vectors loaded.
    uint64_t NumElts = getDataLayout()->getTypeAllocSize(I.getType()) / 8;
    Info.memVT = EVT::getVectorVT(I.getType()->getContext(), MVT::i64, NumElts);
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Value *AlignArg = I.getArgOperand(I.getNumArgOperands() - 1);
    Info.align = cast<ConstantInt>(AlignArg)->getZExtValue();
    Info.vol = false; // volatile loads with NEON intrinsics not supported
    Info.readMem = true;
    Info.writeMem = false;
    return true;
  }
  case Intrinsic::arm_neon_vst1:
  case Intrinsic::arm_neon_vst2:
  case Intrinsic::arm_neon_vst3:
  case Intrinsic::arm_neon_vst4:
  case Intrinsic::arm_neon_vst2lane:
  case Intrinsic::arm_neon_vst3lane:
  case Intrinsic::arm_neon_vst4lane: {
    Info.opc = ISD::INTRINSIC_VOID;
    // Conservatively set memVT to the entire set of vectors stored.
    unsigned NumElts = 0;
    for (unsigned ArgI = 1, ArgE = I.getNumArgOperands(); ArgI < ArgE; ++ArgI) {
      Type *ArgTy = I.getArgOperand(ArgI)->getType();
      if (!ArgTy->isVectorTy())
        break;
      NumElts += getDataLayout()->getTypeAllocSize(ArgTy) / 8;
    }
    Info.memVT = EVT::getVectorVT(I.getType()->getContext(), MVT::i64, NumElts);
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Value *AlignArg = I.getArgOperand(I.getNumArgOperands() - 1);
    Info.align = cast<ConstantInt>(AlignArg)->getZExtValue();
    Info.vol = false; // volatile stores with NEON intrinsics not supported
    Info.readMem = false;
    Info.writeMem = true;
    return true;
  }
  case Intrinsic::arm_strexd: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i64;
    Info.ptrVal = I.getArgOperand(2);
    Info.offset = 0;
    Info.align = 8;
    Info.vol = true;
    Info.readMem = false;
    Info.writeMem = true;
    return true;
  }
  case Intrinsic::arm_ldrexd: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i64;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = 8;
    Info.vol = true;
    Info.readMem = true;
    Info.writeMem = false;
    return true;
  }
  default:
    break;
  }

  return false;
}
