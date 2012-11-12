//===-- SPUISelLowering.cpp - Cell SPU DAG Lowering Implementation --------===//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPUTargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "SPUISelLowering.h"
#include "SPUTargetMachine.h"
#include "SPUFrameLowering.h"
#include "SPUMachineFunction.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  // Byte offset of the preferred slot (counted from the MSB)
  int prefslotOffset(EVT VT) {
    int retval=0;
    if (VT==MVT::i1) retval=3;
    if (VT==MVT::i8) retval=3;
    if (VT==MVT::i16) retval=2;

    return retval;
  }

  //! Expand a library call into an actual call DAG node
  /*!
   \note
   This code is taken from SelectionDAGLegalize, since it is not exposed as
   part of the LLVM SelectionDAG API.
   */

  SDValue
  ExpandLibCall(RTLIB::Libcall LC, SDValue Op, SelectionDAG &DAG,
                bool isSigned, SDValue &Hi, const SPUTargetLowering &TLI) {
    // The input chain to this libcall is the entry node of the function.
    // Legalizing the call will automatically add the previous call to the
    // dependence.
    SDValue InChain = DAG.getEntryNode();

    TargetLowering::ArgListTy Args;
    TargetLowering::ArgListEntry Entry;
    for (unsigned i = 0, e = Op.getNumOperands(); i != e; ++i) {
      EVT ArgVT = Op.getOperand(i).getValueType();
      Type *ArgTy = ArgVT.getTypeForEVT(*DAG.getContext());
      Entry.Node = Op.getOperand(i);
      Entry.Ty = ArgTy;
      Entry.isSExt = isSigned;
      Entry.isZExt = !isSigned;
      Args.push_back(Entry);
    }
    SDValue Callee = DAG.getExternalSymbol(TLI.getLibcallName(LC),
                                           TLI.getPointerTy());

    // Splice the libcall in wherever FindInputOutputChains tells us to.
    Type *RetTy =
                Op.getNode()->getValueType(0).getTypeForEVT(*DAG.getContext());
    TargetLowering::CallLoweringInfo CLI(InChain, RetTy, isSigned, !isSigned,
                                         false, false,
                            0, TLI.getLibcallCallingConv(LC),
                            /*isTailCall=*/false,
                                         /*doesNotRet=*/false,
                                         /*isReturnValueUsed=*/true,
                            Callee, Args, DAG, Op.getDebugLoc());
    std::pair<SDValue, SDValue> CallInfo = TLI.LowerCallTo(CLI);

    return CallInfo.first;
  }
}

SPUTargetLowering::SPUTargetLowering(SPUTargetMachine &TM)
  : TargetLowering(TM, new TargetLoweringObjectFileELF()),
    SPUTM(TM) {

  // Use _setjmp/_longjmp instead of setjmp/longjmp.
  setUseUnderscoreSetJmp(true);
  setUseUnderscoreLongJmp(true);

  // Set RTLIB libcall names as used by SPU:
  setLibcallName(RTLIB::DIV_F64, "__fast_divdf3");

  // Set up the SPU's register classes:
  addRegisterClass(MVT::i8,   &SPU::R8CRegClass);
  addRegisterClass(MVT::i16,  &SPU::R16CRegClass);
  addRegisterClass(MVT::i32,  &SPU::R32CRegClass);
  addRegisterClass(MVT::i64,  &SPU::R64CRegClass);
  addRegisterClass(MVT::f32,  &SPU::R32FPRegClass);
  addRegisterClass(MVT::f64,  &SPU::R64FPRegClass);
  addRegisterClass(MVT::i128, &SPU::GPRCRegClass);

  // SPU has no sign or zero extended loads for i1, i8, i16:
  setLoadExtAction(ISD::EXTLOAD,  MVT::i1, Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);

  setLoadExtAction(ISD::EXTLOAD,  MVT::f32, Expand);
  setLoadExtAction(ISD::EXTLOAD,  MVT::f64, Expand);

  setTruncStoreAction(MVT::i128, MVT::i64, Expand);
  setTruncStoreAction(MVT::i128, MVT::i32, Expand);
  setTruncStoreAction(MVT::i128, MVT::i16, Expand);
  setTruncStoreAction(MVT::i128, MVT::i8, Expand);

  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // SPU constant load actions are custom lowered:
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Custom);

  // SPU's loads and stores have to be custom lowered:
  for (unsigned sctype = (unsigned) MVT::i8; sctype < (unsigned) MVT::i128;
       ++sctype) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)sctype;

    setOperationAction(ISD::LOAD,   VT, Custom);
    setOperationAction(ISD::STORE,  VT, Custom);
    setLoadExtAction(ISD::EXTLOAD,  VT, Custom);
    setLoadExtAction(ISD::ZEXTLOAD, VT, Custom);
    setLoadExtAction(ISD::SEXTLOAD, VT, Custom);

    for (unsigned stype = sctype - 1; stype >= (unsigned) MVT::i8; --stype) {
      MVT::SimpleValueType StoreVT = (MVT::SimpleValueType) stype;
      setTruncStoreAction(VT, StoreVT, Expand);
    }
  }

  for (unsigned sctype = (unsigned) MVT::f32; sctype < (unsigned) MVT::f64;
       ++sctype) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType) sctype;

    setOperationAction(ISD::LOAD,   VT, Custom);
    setOperationAction(ISD::STORE,  VT, Custom);

    for (unsigned stype = sctype - 1; stype >= (unsigned) MVT::f32; --stype) {
      MVT::SimpleValueType StoreVT = (MVT::SimpleValueType) stype;
      setTruncStoreAction(VT, StoreVT, Expand);
    }
  }

  // Expand the jumptable branches
  setOperationAction(ISD::BR_JT,        MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,        MVT::Other, Expand);

  // Custom lower SELECT_CC for most cases, but expand by default
  setOperationAction(ISD::SELECT_CC,    MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC,    MVT::i8,    Custom);
  setOperationAction(ISD::SELECT_CC,    MVT::i16,   Custom);
  setOperationAction(ISD::SELECT_CC,    MVT::i32,   Custom);
  setOperationAction(ISD::SELECT_CC,    MVT::i64,   Custom);

  // SPU has no intrinsics for these particular operations:
  setOperationAction(ISD::MEMBARRIER, MVT::Other, Expand);
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Expand);

  // SPU has no division/remainder instructions
  setOperationAction(ISD::SREM,    MVT::i8,   Expand);
  setOperationAction(ISD::UREM,    MVT::i8,   Expand);
  setOperationAction(ISD::SDIV,    MVT::i8,   Expand);
  setOperationAction(ISD::UDIV,    MVT::i8,   Expand);
  setOperationAction(ISD::SDIVREM, MVT::i8,   Expand);
  setOperationAction(ISD::UDIVREM, MVT::i8,   Expand);
  setOperationAction(ISD::SREM,    MVT::i16,  Expand);
  setOperationAction(ISD::UREM,    MVT::i16,  Expand);
  setOperationAction(ISD::SDIV,    MVT::i16,  Expand);
  setOperationAction(ISD::UDIV,    MVT::i16,  Expand);
  setOperationAction(ISD::SDIVREM, MVT::i16,  Expand);
  setOperationAction(ISD::UDIVREM, MVT::i16,  Expand);
  setOperationAction(ISD::SREM,    MVT::i32,  Expand);
  setOperationAction(ISD::UREM,    MVT::i32,  Expand);
  setOperationAction(ISD::SDIV,    MVT::i32,  Expand);
  setOperationAction(ISD::UDIV,    MVT::i32,  Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32,  Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32,  Expand);
  setOperationAction(ISD::SREM,    MVT::i64,  Expand);
  setOperationAction(ISD::UREM,    MVT::i64,  Expand);
  setOperationAction(ISD::SDIV,    MVT::i64,  Expand);
  setOperationAction(ISD::UDIV,    MVT::i64,  Expand);
  setOperationAction(ISD::SDIVREM, MVT::i64,  Expand);
  setOperationAction(ISD::UDIVREM, MVT::i64,  Expand);
  setOperationAction(ISD::SREM,    MVT::i128, Expand);
  setOperationAction(ISD::UREM,    MVT::i128, Expand);
  setOperationAction(ISD::SDIV,    MVT::i128, Expand);
  setOperationAction(ISD::UDIV,    MVT::i128, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i128, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i128, Expand);

  // We don't support sin/cos/sqrt/fmod
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FREM , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::FREM , MVT::f32, Expand);

  // Expand fsqrt to the appropriate libcall (NOTE: should use h/w fsqrt
  // for f32!)
  setOperationAction(ISD::FSQRT, MVT::f64, Expand);
  setOperationAction(ISD::FSQRT, MVT::f32, Expand);

  setOperationAction(ISD::FMA, MVT::f64, Expand);
  setOperationAction(ISD::FMA, MVT::f32, Expand);

  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

  // SPU can do rotate right and left, so legalize it... but customize for i8
  // because instructions don't exist.

  // FIXME: Change from "expand" to appropriate type once ROTR is supported in
  //        .td files.
  setOperationAction(ISD::ROTR, MVT::i32,    Expand /*Legal*/);
  setOperationAction(ISD::ROTR, MVT::i16,    Expand /*Legal*/);
  setOperationAction(ISD::ROTR, MVT::i8,     Expand /*Custom*/);

  setOperationAction(ISD::ROTL, MVT::i32,    Legal);
  setOperationAction(ISD::ROTL, MVT::i16,    Legal);
  setOperationAction(ISD::ROTL, MVT::i8,     Custom);

  // SPU has no native version of shift left/right for i8
  setOperationAction(ISD::SHL,  MVT::i8,     Custom);
  setOperationAction(ISD::SRL,  MVT::i8,     Custom);
  setOperationAction(ISD::SRA,  MVT::i8,     Custom);

  // Make these operations legal and handle them during instruction selection:
  setOperationAction(ISD::SHL,  MVT::i64,    Legal);
  setOperationAction(ISD::SRL,  MVT::i64,    Legal);
  setOperationAction(ISD::SRA,  MVT::i64,    Legal);

  // Custom lower i8, i32 and i64 multiplications
  setOperationAction(ISD::MUL,  MVT::i8,     Custom);
  setOperationAction(ISD::MUL,  MVT::i32,    Legal);
  setOperationAction(ISD::MUL,  MVT::i64,    Legal);

  // Expand double-width multiplication
  // FIXME: It would probably be reasonable to support some of these operations
  setOperationAction(ISD::UMUL_LOHI, MVT::i8,  Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i8,  Expand);
  setOperationAction(ISD::MULHU,     MVT::i8,  Expand);
  setOperationAction(ISD::MULHS,     MVT::i8,  Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i16, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i16, Expand);
  setOperationAction(ISD::MULHU,     MVT::i16, Expand);
  setOperationAction(ISD::MULHS,     MVT::i16, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::MULHU,     MVT::i32, Expand);
  setOperationAction(ISD::MULHS,     MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::MULHU,     MVT::i64, Expand);
  setOperationAction(ISD::MULHS,     MVT::i64, Expand);

  // Need to custom handle (some) common i8, i64 math ops
  setOperationAction(ISD::ADD,  MVT::i8,     Custom);
  setOperationAction(ISD::ADD,  MVT::i64,    Legal);
  setOperationAction(ISD::SUB,  MVT::i8,     Custom);
  setOperationAction(ISD::SUB,  MVT::i64,    Legal);

  // SPU does not have BSWAP. It does have i32 support CTLZ.
  // CTPOP has to be custom lowered.
  setOperationAction(ISD::BSWAP, MVT::i32,   Expand);
  setOperationAction(ISD::BSWAP, MVT::i64,   Expand);

  setOperationAction(ISD::CTPOP, MVT::i8,    Custom);
  setOperationAction(ISD::CTPOP, MVT::i16,   Custom);
  setOperationAction(ISD::CTPOP, MVT::i32,   Custom);
  setOperationAction(ISD::CTPOP, MVT::i64,   Custom);
  setOperationAction(ISD::CTPOP, MVT::i128,  Expand);

  setOperationAction(ISD::CTTZ , MVT::i8,    Expand);
  setOperationAction(ISD::CTTZ , MVT::i16,   Expand);
  setOperationAction(ISD::CTTZ , MVT::i32,   Expand);
  setOperationAction(ISD::CTTZ , MVT::i64,   Expand);
  setOperationAction(ISD::CTTZ , MVT::i128,  Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i8,    Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i16,   Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32,   Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i64,   Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i128,  Expand);

  setOperationAction(ISD::CTLZ , MVT::i8,    Promote);
  setOperationAction(ISD::CTLZ , MVT::i16,   Promote);
  setOperationAction(ISD::CTLZ , MVT::i32,   Legal);
  setOperationAction(ISD::CTLZ , MVT::i64,   Expand);
  setOperationAction(ISD::CTLZ , MVT::i128,  Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i8,    Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i16,   Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32,   Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i64,   Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i128,  Expand);

  // SPU has a version of select that implements (a&~c)|(b&c), just like
  // select ought to work:
  setOperationAction(ISD::SELECT, MVT::i8,   Legal);
  setOperationAction(ISD::SELECT, MVT::i16,  Legal);
  setOperationAction(ISD::SELECT, MVT::i32,  Legal);
  setOperationAction(ISD::SELECT, MVT::i64,  Legal);

  setOperationAction(ISD::SETCC, MVT::i8,    Legal);
  setOperationAction(ISD::SETCC, MVT::i16,   Legal);
  setOperationAction(ISD::SETCC, MVT::i32,   Legal);
  setOperationAction(ISD::SETCC, MVT::i64,   Legal);
  setOperationAction(ISD::SETCC, MVT::f64,   Custom);

  // Custom lower i128 -> i64 truncates
  setOperationAction(ISD::TRUNCATE, MVT::i64, Custom);

  // Custom lower i32/i64 -> i128 sign extend
  setOperationAction(ISD::SIGN_EXTEND, MVT::i128, Custom);

  setOperationAction(ISD::FP_TO_SINT, MVT::i8, Promote);
  setOperationAction(ISD::FP_TO_UINT, MVT::i8, Promote);
  setOperationAction(ISD::FP_TO_SINT, MVT::i16, Promote);
  setOperationAction(ISD::FP_TO_UINT, MVT::i16, Promote);
  // SPU has a legal FP -> signed INT instruction for f32, but for f64, need
  // to expand to a libcall, hence the custom lowering:
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::i128, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::i128, Expand);

  // FDIV on SPU requires custom lowering
  setOperationAction(ISD::FDIV, MVT::f64, Expand);      // to libcall

  // SPU has [U|S]INT_TO_FP for f32->i32, but not for f64->i32, f64->i64:
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i16, Promote);
  setOperationAction(ISD::SINT_TO_FP, MVT::i8,  Promote);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i16, Promote);
  setOperationAction(ISD::UINT_TO_FP, MVT::i8,  Promote);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Custom);

  setOperationAction(ISD::BITCAST, MVT::i32, Legal);
  setOperationAction(ISD::BITCAST, MVT::f32, Legal);
  setOperationAction(ISD::BITCAST, MVT::i64, Legal);
  setOperationAction(ISD::BITCAST, MVT::f64, Legal);

  // We cannot sextinreg(i1).  Expand to shifts.
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  // We want to legalize GlobalAddress and ConstantPool nodes into the
  // appropriate instructions to materialize the address.
  for (unsigned sctype = (unsigned) MVT::i8; sctype < (unsigned) MVT::f128;
       ++sctype) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)sctype;

    setOperationAction(ISD::GlobalAddress,  VT, Custom);
    setOperationAction(ISD::ConstantPool,   VT, Custom);
    setOperationAction(ISD::JumpTable,      VT, Custom);
  }

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);

  // Use the default implementation.
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64  , Expand);

  // Cell SPU has instructions for converting between i64 and fp.
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);

  // To take advantage of the above i64 FP_TO_SINT, promote i32 FP_TO_UINT
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Promote);

  // BUILD_PAIR can't be handled natively, and should be expanded to shl/or
  setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand);

  // First set operation action for all vector types to expand. Then we
  // will selectively turn on ones that can be effectively codegen'd.
  addRegisterClass(MVT::v16i8, &SPU::VECREGRegClass);
  addRegisterClass(MVT::v8i16, &SPU::VECREGRegClass);
  addRegisterClass(MVT::v4i32, &SPU::VECREGRegClass);
  addRegisterClass(MVT::v2i64, &SPU::VECREGRegClass);
  addRegisterClass(MVT::v4f32, &SPU::VECREGRegClass);
  addRegisterClass(MVT::v2f64, &SPU::VECREGRegClass);

  for (unsigned i = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
       i <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++i) {
    MVT::SimpleValueType VT = (MVT::SimpleValueType)i;

    // Set operation actions to legal types only.
    if (!isTypeLegal(VT)) continue;

    // add/sub are legal for all supported vector VT's.
    setOperationAction(ISD::ADD,     VT, Legal);
    setOperationAction(ISD::SUB,     VT, Legal);
    // mul has to be custom lowered.
    setOperationAction(ISD::MUL,     VT, Legal);

    setOperationAction(ISD::AND,     VT, Legal);
    setOperationAction(ISD::OR,      VT, Legal);
    setOperationAction(ISD::XOR,     VT, Legal);
    setOperationAction(ISD::LOAD,    VT, Custom);
    setOperationAction(ISD::SELECT,  VT, Legal);
    setOperationAction(ISD::STORE,   VT, Custom);

    // These operations need to be expanded:
    setOperationAction(ISD::SDIV,    VT, Expand);
    setOperationAction(ISD::SREM,    VT, Expand);
    setOperationAction(ISD::UDIV,    VT, Expand);
    setOperationAction(ISD::UREM,    VT, Expand);

    // Expand all trunc stores
    for (unsigned j = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
         j <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++j) {
      MVT::SimpleValueType TargetVT = (MVT::SimpleValueType)j;
    setTruncStoreAction(VT, TargetVT, Expand);
    }

    // Custom lower build_vector, constant pool spills, insert and
    // extract vector elements:
    setOperationAction(ISD::BUILD_VECTOR, VT, Custom);
    setOperationAction(ISD::ConstantPool, VT, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR, VT, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
  }

  setOperationAction(ISD::SHL, MVT::v2i64, Expand);

  setOperationAction(ISD::AND, MVT::v16i8, Custom);
  setOperationAction(ISD::OR,  MVT::v16i8, Custom);
  setOperationAction(ISD::XOR, MVT::v16i8, Custom);
  setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4f32, Custom);

  setOperationAction(ISD::FDIV, MVT::v4f32, Legal);

  setBooleanContents(ZeroOrNegativeOneBooleanContent);
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent); // FIXME: Is this correct?

  setStackPointerRegisterToSaveRestore(SPU::R1);

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  setTargetDAGCombine(ISD::SIGN_EXTEND);
  setTargetDAGCombine(ISD::ANY_EXTEND);

  setMinFunctionAlignment(3);

  computeRegisterProperties();

  // Set pre-RA register scheduler default to BURR, which produces slightly
  // better code than the default (could also be TDRR, but TargetLowering.h
  // needs a mod to support that model):
  setSchedulingPreference(Sched::RegPressure);
}

const char *SPUTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case SPUISD::RET_FLAG: return "SPUISD::RET_FLAG";
  case SPUISD::Hi: return "SPUISD::Hi";
  case SPUISD::Lo: return "SPUISD::Lo";
  case SPUISD::PCRelAddr: return "SPUISD::PCRelAddr";
  case SPUISD::AFormAddr: return "SPUISD::AFormAddr";
  case SPUISD::IndirectAddr: return "SPUISD::IndirectAddr";
  case SPUISD::LDRESULT: return "SPUISD::LDRESULT";
  case SPUISD::CALL: return "SPUISD::CALL";
  case SPUISD::SHUFB: return "SPUISD::SHUFB";
  case SPUISD::SHUFFLE_MASK: return "SPUISD::SHUFFLE_MASK";
  case SPUISD::CNTB: return "SPUISD::CNTB";
  case SPUISD::PREFSLOT2VEC: return "SPUISD::PREFSLOT2VEC";
  case SPUISD::VEC2PREFSLOT: return "SPUISD::VEC2PREFSLOT";
  case SPUISD::SHL_BITS: return "SPUISD::SHL_BITS";
  case SPUISD::SHL_BYTES: return "SPUISD::SHL_BYTES";
  case SPUISD::VEC_ROTL: return "SPUISD::VEC_ROTL";
  case SPUISD::VEC_ROTR: return "SPUISD::VEC_ROTR";
  case SPUISD::ROTBYTES_LEFT: return "SPUISD::ROTBYTES_LEFT";
  case SPUISD::ROTBYTES_LEFT_BITS: return "SPUISD::ROTBYTES_LEFT_BITS";
  case SPUISD::SELECT_MASK: return "SPUISD::SELECT_MASK";
  case SPUISD::SELB: return "SPUISD::SELB";
  case SPUISD::ADD64_MARKER: return "SPUISD::ADD64_MARKER";
  case SPUISD::SUB64_MARKER: return "SPUISD::SUB64_MARKER";
  case SPUISD::MUL64_MARKER: return "SPUISD::MUL64_MARKER";
  }
}

//===----------------------------------------------------------------------===//
// Return the Cell SPU's SETCC result type
//===----------------------------------------------------------------------===//

EVT SPUTargetLowering::getSetCCResultType(EVT VT) const {
  // i8, i16 and i32 are valid SETCC result types
  MVT::SimpleValueType retval;

  switch(VT.getSimpleVT().SimpleTy){
    case MVT::i1:
    case MVT::i8:
      retval = MVT::i8; break;
    case MVT::i16:
      retval = MVT::i16; break;
    case MVT::i32:
    default:
      retval = MVT::i32;
  }
  return retval;
}

//===----------------------------------------------------------------------===//
// Calling convention code:
//===----------------------------------------------------------------------===//

#include "SPUGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//  LowerOperation implementation
//===----------------------------------------------------------------------===//

/// Custom lower loads for CellSPU
/*!
 All CellSPU loads and stores are aligned to 16-byte boundaries, so for elements
 within a 16-byte block, we have to rotate to extract the requested element.

 For extending loads, we also want to ensure that the following sequence is
 emitted, e.g. for MVT::f32 extending load to MVT::f64:

\verbatim
%1  v16i8,ch = load
%2  v16i8,ch = rotate %1
%3  v4f8, ch = bitconvert %2
%4  f32      = vec2perfslot %3
%5  f64      = fp_extend %4
\endverbatim
*/
static SDValue
LowerLOAD(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  LoadSDNode *LN = cast<LoadSDNode>(Op);
  SDValue the_chain = LN->getChain();
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  EVT InVT = LN->getMemoryVT();
  EVT OutVT = Op.getValueType();
  ISD::LoadExtType ExtType = LN->getExtensionType();
  unsigned alignment = LN->getAlignment();
  int pso = prefslotOffset(InVT);
  DebugLoc dl = Op.getDebugLoc();
  EVT vecVT = InVT.isVector()? InVT: EVT::getVectorVT(*DAG.getContext(), InVT,
                                                  (128 / InVT.getSizeInBits()));

  // two sanity checks
  assert( LN->getAddressingMode() == ISD::UNINDEXED
          && "we should get only UNINDEXED adresses");
  // clean aligned loads can be selected as-is
  if (InVT.getSizeInBits() == 128 && (alignment%16) == 0)
    return SDValue();

  // Get pointerinfos to the memory chunk(s) that contain the data to load
  uint64_t mpi_offset = LN->getPointerInfo().Offset;
  mpi_offset -= mpi_offset%16;
  MachinePointerInfo lowMemPtr(LN->getPointerInfo().V, mpi_offset);
  MachinePointerInfo highMemPtr(LN->getPointerInfo().V, mpi_offset+16);

  SDValue result;
  SDValue basePtr = LN->getBasePtr();
  SDValue rotate;

  if ((alignment%16) == 0) {
    ConstantSDNode *CN;

    // Special cases for a known aligned load to simplify the base pointer
    // and the rotation amount:
    if (basePtr.getOpcode() == ISD::ADD
        && (CN = dyn_cast<ConstantSDNode > (basePtr.getOperand(1))) != 0) {
      // Known offset into basePtr
      int64_t offset = CN->getSExtValue();
      int64_t rotamt = int64_t((offset & 0xf) - pso);

      if (rotamt < 0)
        rotamt += 16;

      rotate = DAG.getConstant(rotamt, MVT::i16);

      // Simplify the base pointer for this case:
      basePtr = basePtr.getOperand(0);
      if ((offset & ~0xf) > 0) {
        basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                              basePtr,
                              DAG.getConstant((offset & ~0xf), PtrVT));
      }
    } else if ((basePtr.getOpcode() == SPUISD::AFormAddr)
               || (basePtr.getOpcode() == SPUISD::IndirectAddr
                   && basePtr.getOperand(0).getOpcode() == SPUISD::Hi
                   && basePtr.getOperand(1).getOpcode() == SPUISD::Lo)) {
      // Plain aligned a-form address: rotate into preferred slot
      // Same for (SPUindirect (SPUhi ...), (SPUlo ...))
      int64_t rotamt = -pso;
      if (rotamt < 0)
        rotamt += 16;
      rotate = DAG.getConstant(rotamt, MVT::i16);
    } else {
      // Offset the rotate amount by the basePtr and the preferred slot
      // byte offset
      int64_t rotamt = -pso;
      if (rotamt < 0)
        rotamt += 16;
      rotate = DAG.getNode(ISD::ADD, dl, PtrVT,
                           basePtr,
                           DAG.getConstant(rotamt, PtrVT));
    }
  } else {
    // Unaligned load: must be more pessimistic about addressing modes:
    if (basePtr.getOpcode() == ISD::ADD) {
      MachineFunction &MF = DAG.getMachineFunction();
      MachineRegisterInfo &RegInfo = MF.getRegInfo();
      unsigned VReg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);
      SDValue Flag;

      SDValue Op0 = basePtr.getOperand(0);
      SDValue Op1 = basePtr.getOperand(1);

      if (isa<ConstantSDNode>(Op1)) {
        // Convert the (add <ptr>, <const>) to an indirect address contained
        // in a register. Note that this is done because we need to avoid
        // creating a 0(reg) d-form address due to the SPU's block loads.
        basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, Op0, Op1);
        the_chain = DAG.getCopyToReg(the_chain, dl, VReg, basePtr, Flag);
        basePtr = DAG.getCopyFromReg(the_chain, dl, VReg, PtrVT);
      } else {
        // Convert the (add <arg1>, <arg2>) to an indirect address, which
        // will likely be lowered as a reg(reg) x-form address.
        basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, Op0, Op1);
      }
    } else {
      basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                            basePtr,
                            DAG.getConstant(0, PtrVT));
   }

    // Offset the rotate amount by the basePtr and the preferred slot
    // byte offset
    rotate = DAG.getNode(ISD::ADD, dl, PtrVT,
                         basePtr,
                         DAG.getConstant(-pso, PtrVT));
  }

  // Do the load as a i128 to allow possible shifting
  SDValue low = DAG.getLoad(MVT::i128, dl, the_chain, basePtr,
                       lowMemPtr,
                       LN->isVolatile(), LN->isNonTemporal(), false, 16);

  // When the size is not greater than alignment we get all data with just
  // one load
  if (alignment >= InVT.getSizeInBits()/8) {
    // Update the chain
    the_chain = low.getValue(1);

    // Rotate into the preferred slot:
    result = DAG.getNode(SPUISD::ROTBYTES_LEFT, dl, MVT::i128,
                         low.getValue(0), rotate);

    // Convert the loaded v16i8 vector to the appropriate vector type
    // specified by the operand:
    EVT vecVT = EVT::getVectorVT(*DAG.getContext(),
                                 InVT, (128 / InVT.getSizeInBits()));
    result = DAG.getNode(SPUISD::VEC2PREFSLOT, dl, InVT,
                         DAG.getNode(ISD::BITCAST, dl, vecVT, result));
  }
  // When alignment is less than the size, we might need (known only at
  // run-time) two loads
  // TODO: if the memory address is composed only from constants, we have
  // extra kowledge, and might avoid the second load
  else {
    // storage position offset from lower 16 byte aligned memory chunk
    SDValue offset = DAG.getNode(ISD::AND, dl, MVT::i32,
                                  basePtr, DAG.getConstant( 0xf, MVT::i32 ) );
    // get a registerfull of ones. (this implementation is a workaround: LLVM
    // cannot handle 128 bit signed int constants)
    SDValue ones = DAG.getConstant(-1, MVT::v4i32 );
    ones = DAG.getNode(ISD::BITCAST, dl, MVT::i128, ones);

    SDValue high = DAG.getLoad(MVT::i128, dl, the_chain,
                               DAG.getNode(ISD::ADD, dl, PtrVT,
                                           basePtr,
                                           DAG.getConstant(16, PtrVT)),
                               highMemPtr,
                               LN->isVolatile(), LN->isNonTemporal(), false, 
                               16);

    the_chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, low.getValue(1),
                                                              high.getValue(1));

    // Shift the (possible) high part right to compensate the misalignemnt.
    // if there is no highpart (i.e. value is i64 and offset is 4), this
    // will zero out the high value.
    high = DAG.getNode(SPUISD::SRL_BYTES, dl, MVT::i128, high,
                                     DAG.getNode(ISD::SUB, dl, MVT::i32,
                                                 DAG.getConstant( 16, MVT::i32),
                                                 offset
                                                ));

    // Shift the low similarly
    // TODO: add SPUISD::SHL_BYTES
    low = DAG.getNode(SPUISD::SHL_BYTES, dl, MVT::i128, low, offset );

    // Merge the two parts
    result = DAG.getNode(ISD::BITCAST, dl, vecVT,
                          DAG.getNode(ISD::OR, dl, MVT::i128, low, high));

    if (!InVT.isVector()) {
      result = DAG.getNode(SPUISD::VEC2PREFSLOT, dl, InVT, result );
     }

  }
    // Handle extending loads by extending the scalar result:
    if (ExtType == ISD::SEXTLOAD) {
      result = DAG.getNode(ISD::SIGN_EXTEND, dl, OutVT, result);
    } else if (ExtType == ISD::ZEXTLOAD) {
      result = DAG.getNode(ISD::ZERO_EXTEND, dl, OutVT, result);
    } else if (ExtType == ISD::EXTLOAD) {
      unsigned NewOpc = ISD::ANY_EXTEND;

      if (OutVT.isFloatingPoint())
        NewOpc = ISD::FP_EXTEND;

      result = DAG.getNode(NewOpc, dl, OutVT, result);
    }

    SDVTList retvts = DAG.getVTList(OutVT, MVT::Other);
    SDValue retops[2] = {
      result,
      the_chain
    };

    result = DAG.getNode(SPUISD::LDRESULT, dl, retvts,
                         retops, sizeof(retops) / sizeof(retops[0]));
    return result;
}

/// Custom lower stores for CellSPU
/*!
 All CellSPU stores are aligned to 16-byte boundaries, so for elements
 within a 16-byte block, we have to generate a shuffle to insert the
 requested element into its place, then store the resulting block.
 */
static SDValue
LowerSTORE(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  StoreSDNode *SN = cast<StoreSDNode>(Op);
  SDValue Value = SN->getValue();
  EVT VT = Value.getValueType();
  EVT StVT = (!SN->isTruncatingStore() ? VT : SN->getMemoryVT());
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  DebugLoc dl = Op.getDebugLoc();
  unsigned alignment = SN->getAlignment();
  SDValue result;
  EVT vecVT = StVT.isVector()? StVT: EVT::getVectorVT(*DAG.getContext(), StVT,
                                                 (128 / StVT.getSizeInBits()));
  // Get pointerinfos to the memory chunk(s) that contain the data to load
  uint64_t mpi_offset = SN->getPointerInfo().Offset;
  mpi_offset -= mpi_offset%16;
  MachinePointerInfo lowMemPtr(SN->getPointerInfo().V, mpi_offset);
  MachinePointerInfo highMemPtr(SN->getPointerInfo().V, mpi_offset+16);


  // two sanity checks
  assert( SN->getAddressingMode() == ISD::UNINDEXED
          && "we should get only UNINDEXED adresses");
  // clean aligned loads can be selected as-is
  if (StVT.getSizeInBits() == 128 && (alignment%16) == 0)
    return SDValue();

  SDValue alignLoadVec;
  SDValue basePtr = SN->getBasePtr();
  SDValue the_chain = SN->getChain();
  SDValue insertEltOffs;

  if ((alignment%16) == 0) {
    ConstantSDNode *CN;
    // Special cases for a known aligned load to simplify the base pointer
    // and insertion byte:
    if (basePtr.getOpcode() == ISD::ADD
        && (CN = dyn_cast<ConstantSDNode>(basePtr.getOperand(1))) != 0) {
      // Known offset into basePtr
      int64_t offset = CN->getSExtValue();

      // Simplify the base pointer for this case:
      basePtr = basePtr.getOperand(0);
      insertEltOffs = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                                  basePtr,
                                  DAG.getConstant((offset & 0xf), PtrVT));

      if ((offset & ~0xf) > 0) {
        basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                              basePtr,
                              DAG.getConstant((offset & ~0xf), PtrVT));
      }
    } else {
      // Otherwise, assume it's at byte 0 of basePtr
      insertEltOffs = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                                  basePtr,
                                  DAG.getConstant(0, PtrVT));
      basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                                  basePtr,
                                  DAG.getConstant(0, PtrVT));
    }
  } else {
    // Unaligned load: must be more pessimistic about addressing modes:
    if (basePtr.getOpcode() == ISD::ADD) {
      MachineFunction &MF = DAG.getMachineFunction();
      MachineRegisterInfo &RegInfo = MF.getRegInfo();
      unsigned VReg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);
      SDValue Flag;

      SDValue Op0 = basePtr.getOperand(0);
      SDValue Op1 = basePtr.getOperand(1);

      if (isa<ConstantSDNode>(Op1)) {
        // Convert the (add <ptr>, <const>) to an indirect address contained
        // in a register. Note that this is done because we need to avoid
        // creating a 0(reg) d-form address due to the SPU's block loads.
        basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, Op0, Op1);
        the_chain = DAG.getCopyToReg(the_chain, dl, VReg, basePtr, Flag);
        basePtr = DAG.getCopyFromReg(the_chain, dl, VReg, PtrVT);
      } else {
        // Convert the (add <arg1>, <arg2>) to an indirect address, which
        // will likely be lowered as a reg(reg) x-form address.
        basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, Op0, Op1);
      }
    } else {
      basePtr = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                            basePtr,
                            DAG.getConstant(0, PtrVT));
    }

    // Insertion point is solely determined by basePtr's contents
    insertEltOffs = DAG.getNode(ISD::ADD, dl, PtrVT,
                                basePtr,
                                DAG.getConstant(0, PtrVT));
  }

  // Load the lower part of the memory to which to store.
  SDValue low = DAG.getLoad(vecVT, dl, the_chain, basePtr,
                          lowMemPtr, SN->isVolatile(), SN->isNonTemporal(),
                            false, 16);

  // if we don't need to store over the 16 byte boundary, one store suffices
  if (alignment >= StVT.getSizeInBits()/8) {
    // Update the chain
    the_chain = low.getValue(1);

    LoadSDNode *LN = cast<LoadSDNode>(low);
    SDValue theValue = SN->getValue();

    if (StVT != VT
        && (theValue.getOpcode() == ISD::AssertZext
            || theValue.getOpcode() == ISD::AssertSext)) {
      // Drill down and get the value for zero- and sign-extended
      // quantities
      theValue = theValue.getOperand(0);
    }

    // If the base pointer is already a D-form address, then just create
    // a new D-form address with a slot offset and the orignal base pointer.
    // Otherwise generate a D-form address with the slot offset relative
    // to the stack pointer, which is always aligned.
#if !defined(NDEBUG)
      if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
        errs() << "CellSPU LowerSTORE: basePtr = ";
        basePtr.getNode()->dump(&DAG);
        errs() << "\n";
      }
#endif

    SDValue insertEltOp = DAG.getNode(SPUISD::SHUFFLE_MASK, dl, vecVT,
                                      insertEltOffs);
    SDValue vectorizeOp = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, vecVT,
                                      theValue);

    result = DAG.getNode(SPUISD::SHUFB, dl, vecVT,
                         vectorizeOp, low,
                         DAG.getNode(ISD::BITCAST, dl,
                                     MVT::v4i32, insertEltOp));

    result = DAG.getStore(the_chain, dl, result, basePtr,
                          lowMemPtr,
                          LN->isVolatile(), LN->isNonTemporal(),
                          16);

  }
  // do the store when it might cross the 16 byte memory access boundary.
  else {
    // TODO issue a warning if SN->isVolatile()== true? This is likely not
    // what the user wanted.

    // address offset from nearest lower 16byte alinged address
    SDValue offset = DAG.getNode(ISD::AND, dl, MVT::i32,
                                    SN->getBasePtr(),
                                    DAG.getConstant(0xf, MVT::i32));
    // 16 - offset
    SDValue offset_compl = DAG.getNode(ISD::SUB, dl, MVT::i32,
                                           DAG.getConstant( 16, MVT::i32),
                                           offset);
    // 16 - sizeof(Value)
    SDValue surplus = DAG.getNode(ISD::SUB, dl, MVT::i32,
                                     DAG.getConstant( 16, MVT::i32),
                                     DAG.getConstant( VT.getSizeInBits()/8,
                                                      MVT::i32));
    // get a registerfull of ones
    SDValue ones = DAG.getConstant(-1, MVT::v4i32);
    ones = DAG.getNode(ISD::BITCAST, dl, MVT::i128, ones);

    // Create the 128 bit masks that have ones where the data to store is
    // located.
    SDValue lowmask, himask;
    // if the value to store don't fill up the an entire 128 bits, zero
    // out the last bits of the mask so that only the value we want to store
    // is masked.
    // this is e.g. in the case of store i32, align 2
    if (!VT.isVector()){
      Value = DAG.getNode(SPUISD::PREFSLOT2VEC, dl, vecVT, Value);
      lowmask = DAG.getNode(SPUISD::SRL_BYTES, dl, MVT::i128, ones, surplus);
      lowmask = DAG.getNode(SPUISD::SHL_BYTES, dl, MVT::i128, lowmask,
                                                               surplus);
      Value = DAG.getNode(ISD::BITCAST, dl, MVT::i128, Value);
      Value = DAG.getNode(ISD::AND, dl, MVT::i128, Value, lowmask);

    }
    else {
      lowmask = ones;
      Value = DAG.getNode(ISD::BITCAST, dl, MVT::i128, Value);
    }
    // this will zero, if there are no data that goes to the high quad
    himask = DAG.getNode(SPUISD::SHL_BYTES, dl, MVT::i128, lowmask,
                                                            offset_compl);
    lowmask = DAG.getNode(SPUISD::SRL_BYTES, dl, MVT::i128, lowmask,
                                                             offset);

    // Load in the old data and zero out the parts that will be overwritten with
    // the new data to store.
    SDValue hi = DAG.getLoad(MVT::i128, dl, the_chain,
                               DAG.getNode(ISD::ADD, dl, PtrVT, basePtr,
                                           DAG.getConstant( 16, PtrVT)),
                               highMemPtr,
                               SN->isVolatile(), SN->isNonTemporal(), 
                               false, 16);
    the_chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, low.getValue(1),
                                                              hi.getValue(1));

    low = DAG.getNode(ISD::AND, dl, MVT::i128,
                        DAG.getNode( ISD::BITCAST, dl, MVT::i128, low),
                        DAG.getNode( ISD::XOR, dl, MVT::i128, lowmask, ones));
    hi = DAG.getNode(ISD::AND, dl, MVT::i128,
                        DAG.getNode( ISD::BITCAST, dl, MVT::i128, hi),
                        DAG.getNode( ISD::XOR, dl, MVT::i128, himask, ones));

    // Shift the Value to store into place. rlow contains the parts that go to
    // the lower memory chunk, rhi has the parts that go to the upper one.
    SDValue rlow = DAG.getNode(SPUISD::SRL_BYTES, dl, MVT::i128, Value, offset);
    rlow = DAG.getNode(ISD::AND, dl, MVT::i128, rlow, lowmask);
    SDValue rhi = DAG.getNode(SPUISD::SHL_BYTES, dl, MVT::i128, Value,
                                                            offset_compl);

    // Merge the old data and the new data and store the results
    // Need to convert vectors here to integer as 'OR'ing floats assert
    rlow = DAG.getNode(ISD::OR, dl, MVT::i128,
                          DAG.getNode(ISD::BITCAST, dl, MVT::i128, low),
                          DAG.getNode(ISD::BITCAST, dl, MVT::i128, rlow));
    rhi = DAG.getNode(ISD::OR, dl, MVT::i128,
                         DAG.getNode(ISD::BITCAST, dl, MVT::i128, hi),
                         DAG.getNode(ISD::BITCAST, dl, MVT::i128, rhi));

    low = DAG.getStore(the_chain, dl, rlow, basePtr,
                          lowMemPtr,
                          SN->isVolatile(), SN->isNonTemporal(), 16);
    hi  = DAG.getStore(the_chain, dl, rhi,
                            DAG.getNode(ISD::ADD, dl, PtrVT, basePtr,
                                        DAG.getConstant( 16, PtrVT)),
                            highMemPtr,
                            SN->isVolatile(), SN->isNonTemporal(), 16);
    result = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, low.getValue(0),
                                                           hi.getValue(0));
  }

  return result;
}

//! Generate the address of a constant pool entry.
static SDValue
LowerConstantPool(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  EVT PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  const Constant *C = CP->getConstVal();
  SDValue CPI = DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment());
  SDValue Zero = DAG.getConstant(0, PtrVT);
  const TargetMachine &TM = DAG.getTarget();
  // FIXME there is no actual debug info here
  DebugLoc dl = Op.getDebugLoc();

  if (TM.getRelocationModel() == Reloc::Static) {
    if (!ST->usingLargeMem()) {
      // Just return the SDValue with the constant pool address in it.
      return DAG.getNode(SPUISD::AFormAddr, dl, PtrVT, CPI, Zero);
    } else {
      SDValue Hi = DAG.getNode(SPUISD::Hi, dl, PtrVT, CPI, Zero);
      SDValue Lo = DAG.getNode(SPUISD::Lo, dl, PtrVT, CPI, Zero);
      return DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, Hi, Lo);
    }
  }

  llvm_unreachable("LowerConstantPool: Relocation model other than static"
                   " not supported.");
}

//! Alternate entry point for generating the address of a constant pool entry
SDValue
SPU::LowerConstantPool(SDValue Op, SelectionDAG &DAG, const SPUTargetMachine &TM) {
  return ::LowerConstantPool(Op, DAG, TM.getSubtargetImpl());
}

static SDValue
LowerJumpTable(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  EVT PtrVT = Op.getValueType();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDValue JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT);
  SDValue Zero = DAG.getConstant(0, PtrVT);
  const TargetMachine &TM = DAG.getTarget();
  // FIXME there is no actual debug info here
  DebugLoc dl = Op.getDebugLoc();

  if (TM.getRelocationModel() == Reloc::Static) {
    if (!ST->usingLargeMem()) {
      return DAG.getNode(SPUISD::AFormAddr, dl, PtrVT, JTI, Zero);
    } else {
      SDValue Hi = DAG.getNode(SPUISD::Hi, dl, PtrVT, JTI, Zero);
      SDValue Lo = DAG.getNode(SPUISD::Lo, dl, PtrVT, JTI, Zero);
      return DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, Hi, Lo);
    }
  }

  llvm_unreachable("LowerJumpTable: Relocation model other than static"
                   " not supported.");
}

static SDValue
LowerGlobalAddress(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  EVT PtrVT = Op.getValueType();
  GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = GSDN->getGlobal();
  SDValue GA = DAG.getTargetGlobalAddress(GV, Op.getDebugLoc(),
                                          PtrVT, GSDN->getOffset());
  const TargetMachine &TM = DAG.getTarget();
  SDValue Zero = DAG.getConstant(0, PtrVT);
  // FIXME there is no actual debug info here
  DebugLoc dl = Op.getDebugLoc();

  if (TM.getRelocationModel() == Reloc::Static) {
    if (!ST->usingLargeMem()) {
      return DAG.getNode(SPUISD::AFormAddr, dl, PtrVT, GA, Zero);
    } else {
      SDValue Hi = DAG.getNode(SPUISD::Hi, dl, PtrVT, GA, Zero);
      SDValue Lo = DAG.getNode(SPUISD::Lo, dl, PtrVT, GA, Zero);
      return DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, Hi, Lo);
    }
  } else {
    report_fatal_error("LowerGlobalAddress: Relocation model other than static"
                      "not supported.");
    /*NOTREACHED*/
  }
}

//! Custom lower double precision floating point constants
static SDValue
LowerConstantFP(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  // FIXME there is no actual debug info here
  DebugLoc dl = Op.getDebugLoc();

  if (VT == MVT::f64) {
    ConstantFPSDNode *FP = cast<ConstantFPSDNode>(Op.getNode());

    assert((FP != 0) &&
           "LowerConstantFP: Node is not ConstantFPSDNode");

    uint64_t dbits = DoubleToBits(FP->getValueAPF().convertToDouble());
    SDValue T = DAG.getConstant(dbits, MVT::i64);
    SDValue Tvec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v2i64, T, T);
    return DAG.getNode(SPUISD::VEC2PREFSLOT, dl, VT,
                       DAG.getNode(ISD::BITCAST, dl, MVT::v2f64, Tvec));
  }

  return SDValue();
}

SDValue
SPUTargetLowering::LowerFormalArguments(SDValue Chain,
                                        CallingConv::ID CallConv, bool isVarArg,
                                        const SmallVectorImpl<ISD::InputArg>
                                          &Ins,
                                        DebugLoc dl, SelectionDAG &DAG,
                                        SmallVectorImpl<SDValue> &InVals)
                                          const {

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  SPUFunctionInfo *FuncInfo = MF.getInfo<SPUFunctionInfo>();

  unsigned ArgOffset = SPUFrameLowering::minStackSize();
  unsigned ArgRegIdx = 0;
  unsigned StackSlotSize = SPUFrameLowering::stackSlotSize();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());
  // FIXME: allow for other calling conventions
  CCInfo.AnalyzeFormalArguments(Ins, CCC_SPU);

  // Add DAG nodes to load the arguments or copy them out of registers.
  for (unsigned ArgNo = 0, e = Ins.size(); ArgNo != e; ++ArgNo) {
    EVT ObjectVT = Ins[ArgNo].VT;
    unsigned ObjSize = ObjectVT.getSizeInBits()/8;
    SDValue ArgVal;
    CCValAssign &VA = ArgLocs[ArgNo];

    if (VA.isRegLoc()) {
      const TargetRegisterClass *ArgRegClass;

      switch (ObjectVT.getSimpleVT().SimpleTy) {
      default:
        report_fatal_error("LowerFormalArguments Unhandled argument type: " +
                           Twine(ObjectVT.getEVTString()));
      case MVT::i8:
        ArgRegClass = &SPU::R8CRegClass;
        break;
      case MVT::i16:
        ArgRegClass = &SPU::R16CRegClass;
        break;
      case MVT::i32:
        ArgRegClass = &SPU::R32CRegClass;
        break;
      case MVT::i64:
        ArgRegClass = &SPU::R64CRegClass;
        break;
      case MVT::i128:
        ArgRegClass = &SPU::GPRCRegClass;
        break;
      case MVT::f32:
        ArgRegClass = &SPU::R32FPRegClass;
        break;
      case MVT::f64:
        ArgRegClass = &SPU::R64FPRegClass;
        break;
      case MVT::v2f64:
      case MVT::v4f32:
      case MVT::v2i64:
      case MVT::v4i32:
      case MVT::v8i16:
      case MVT::v16i8:
        ArgRegClass = &SPU::VECREGRegClass;
        break;
      }

      unsigned VReg = RegInfo.createVirtualRegister(ArgRegClass);
      RegInfo.addLiveIn(VA.getLocReg(), VReg);
      ArgVal = DAG.getCopyFromReg(Chain, dl, VReg, ObjectVT);
      ++ArgRegIdx;
    } else {
      // We need to load the argument to a virtual register if we determined
      // above that we ran out of physical registers of the appropriate type
      // or we're forced to do vararg
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset, true);
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      ArgVal = DAG.getLoad(ObjectVT, dl, Chain, FIN, MachinePointerInfo(),
                           false, false, false, 0);
      ArgOffset += StackSlotSize;
    }

    InVals.push_back(ArgVal);
    // Update the chain
    Chain = ArgVal.getOperand(0);
  }

  // vararg handling:
  if (isVarArg) {
    // FIXME: we should be able to query the argument registers from
    //        tablegen generated code.
    static const uint16_t ArgRegs[] = {
      SPU::R3,  SPU::R4,  SPU::R5,  SPU::R6,  SPU::R7,  SPU::R8,  SPU::R9,
      SPU::R10, SPU::R11, SPU::R12, SPU::R13, SPU::R14, SPU::R15, SPU::R16,
      SPU::R17, SPU::R18, SPU::R19, SPU::R20, SPU::R21, SPU::R22, SPU::R23,
      SPU::R24, SPU::R25, SPU::R26, SPU::R27, SPU::R28, SPU::R29, SPU::R30,
      SPU::R31, SPU::R32, SPU::R33, SPU::R34, SPU::R35, SPU::R36, SPU::R37,
      SPU::R38, SPU::R39, SPU::R40, SPU::R41, SPU::R42, SPU::R43, SPU::R44,
      SPU::R45, SPU::R46, SPU::R47, SPU::R48, SPU::R49, SPU::R50, SPU::R51,
      SPU::R52, SPU::R53, SPU::R54, SPU::R55, SPU::R56, SPU::R57, SPU::R58,
      SPU::R59, SPU::R60, SPU::R61, SPU::R62, SPU::R63, SPU::R64, SPU::R65,
      SPU::R66, SPU::R67, SPU::R68, SPU::R69, SPU::R70, SPU::R71, SPU::R72,
      SPU::R73, SPU::R74, SPU::R75, SPU::R76, SPU::R77, SPU::R78, SPU::R79
    };
    // size of ArgRegs array
    const unsigned NumArgRegs = 77;

    // We will spill (79-3)+1 registers to the stack
    SmallVector<SDValue, 79-3+1> MemOps;

    // Create the frame slot
    for (; ArgRegIdx != NumArgRegs; ++ArgRegIdx) {
      FuncInfo->setVarArgsFrameIndex(
        MFI->CreateFixedObject(StackSlotSize, ArgOffset, true));
      SDValue FIN = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(), PtrVT);
      unsigned VReg = MF.addLiveIn(ArgRegs[ArgRegIdx], &SPU::VECREGRegClass);
      SDValue ArgVal = DAG.getRegister(VReg, MVT::v16i8);
      SDValue Store = DAG.getStore(Chain, dl, ArgVal, FIN, MachinePointerInfo(),
                                   false, false, 0);
      Chain = Store.getOperand(0);
      MemOps.push_back(Store);

      // Increment address by stack slot size for the next stored argument
      ArgOffset += StackSlotSize;
    }
    if (!MemOps.empty())
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                          &MemOps[0], MemOps.size());
  }

  return Chain;
}

/// isLSAAddress - Return the immediate to use if the specified
/// value is representable as a LSA address.
static SDNode *isLSAAddress(SDValue Op, SelectionDAG &DAG) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
  if (!C) return 0;

  int Addr = C->getZExtValue();
  if ((Addr & 3) != 0 ||  // Low 2 bits are implicitly zero.
      (Addr << 14 >> 14) != Addr)
    return 0;  // Top 14 bits have to be sext of immediate.

  return DAG.getConstant((int)C->getZExtValue() >> 2, MVT::i32).getNode();
}

SDValue
SPUTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
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

  // CellSPU target does not yet support tail call optimization.
  isTailCall = false;

  const SPUSubtarget *ST = SPUTM.getSubtargetImpl();
  unsigned NumOps     = Outs.size();
  unsigned StackSlotSize = SPUFrameLowering::stackSlotSize();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());
  // FIXME: allow for other calling conventions
  CCInfo.AnalyzeCallOperands(Outs, CCC_SPU);

  const unsigned NumArgRegs = ArgLocs.size();


  // Handy pointer type
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

  // Set up a copy of the stack pointer for use loading and storing any
  // arguments that may not fit in the registers available for argument
  // passing.
  SDValue StackPtr = DAG.getRegister(SPU::R1, MVT::i32);

  // Figure out which arguments are going to go in registers, and which in
  // memory.
  unsigned ArgOffset = SPUFrameLowering::minStackSize(); // Just below [LR]
  unsigned ArgRegIdx = 0;

  // Keep track of registers passing arguments
  std::vector<std::pair<unsigned, SDValue> > RegsToPass;
  // And the arguments passed on the stack
  SmallVector<SDValue, 8> MemOpChains;

  for (; ArgRegIdx != NumOps; ++ArgRegIdx) {
    SDValue Arg = OutVals[ArgRegIdx];
    CCValAssign &VA = ArgLocs[ArgRegIdx];

    // PtrOff will be used to store the current argument to the stack if a
    // register cannot be found for it.
    SDValue PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());
    PtrOff = DAG.getNode(ISD::ADD, dl, PtrVT, StackPtr, PtrOff);

    switch (Arg.getValueType().getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unexpected ValueType for argument!");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
    case MVT::i128:
    case MVT::f32:
    case MVT::f64:
    case MVT::v2i64:
    case MVT::v2f64:
    case MVT::v4f32:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      if (ArgRegIdx != NumArgRegs) {
        RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      } else {
        MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                           MachinePointerInfo(),
                                           false, false, 0));
        ArgOffset += StackSlotSize;
      }
      break;
    }
  }

  // Accumulate how many bytes are to be pushed on the stack, including the
  // linkage area, and parameter passing area.  According to the SPU ABI,
  // we minimally need space for [LR] and [SP].
  unsigned NumStackBytes = ArgOffset - SPUFrameLowering::minStackSize();

  // Insert a call sequence start
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumStackBytes,
                                                            true));

  if (!MemOpChains.empty()) {
    // Adjust the stack pointer for the stack arguments.
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  SmallVector<SDValue, 8> Ops;
  unsigned CallOpc = SPUISD::CALL;

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    EVT CalleeVT = Callee.getValueType();
    SDValue Zero = DAG.getConstant(0, PtrVT);
    SDValue GA = DAG.getTargetGlobalAddress(GV, dl, CalleeVT);

    if (!ST->usingLargeMem()) {
      // Turn calls to targets that are defined (i.e., have bodies) into BRSL
      // style calls, otherwise, external symbols are BRASL calls. This assumes
      // that declared/defined symbols are in the same compilation unit and can
      // be reached through PC-relative jumps.
      //
      // NOTE:
      // This may be an unsafe assumption for JIT and really large compilation
      // units.
      if (GV->isDeclaration()) {
        Callee = DAG.getNode(SPUISD::AFormAddr, dl, CalleeVT, GA, Zero);
      } else {
        Callee = DAG.getNode(SPUISD::PCRelAddr, dl, CalleeVT, GA, Zero);
      }
    } else {
      // "Large memory" mode: Turn all calls into indirect calls with a X-form
      // address pairs:
      Callee = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, GA, Zero);
    }
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    EVT CalleeVT = Callee.getValueType();
    SDValue Zero = DAG.getConstant(0, PtrVT);
    SDValue ExtSym = DAG.getTargetExternalSymbol(S->getSymbol(),
        Callee.getValueType());

    if (!ST->usingLargeMem()) {
      Callee = DAG.getNode(SPUISD::AFormAddr, dl, CalleeVT, ExtSym, Zero);
    } else {
      Callee = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT, ExtSym, Zero);
    }
  } else if (SDNode *Dest = isLSAAddress(Callee, DAG)) {
    // If this is an absolute destination address that appears to be a legal
    // local store address, use the munged value.
    Callee = SDValue(Dest, 0);
  }

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
  Chain = DAG.getNode(CallOpc, dl, DAG.getVTList(MVT::Other, MVT::Glue),
                      &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumStackBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  // If the function returns void, just return the chain.
  if (Ins.empty())
    return Chain;

  // Now handle the return value(s)
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCRetInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                    getTargetMachine(), RVLocs, *DAG.getContext());
  CCRetInfo.AnalyzeCallResult(Ins, CCC_SPU);


  // If the call has results, copy the values out of the ret val registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];

    SDValue Val = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), VA.getLocVT(),
                                     InFlag);
    Chain = Val.getValue(1);
    InFlag = Val.getValue(2);
    InVals.push_back(Val);
   }

  return Chain;
}

SDValue
SPUTargetLowering::LowerReturn(SDValue Chain,
                               CallingConv::ID CallConv, bool isVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               DebugLoc dl, SelectionDAG &DAG) const {

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_SPU);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(),
                             OutVals[i], Flag);
    Flag = Chain.getValue(1);
  }

  if (Flag.getNode())
    return DAG.getNode(SPUISD::RET_FLAG, dl, MVT::Other, Chain, Flag);
  else
    return DAG.getNode(SPUISD::RET_FLAG, dl, MVT::Other, Chain);
}


//===----------------------------------------------------------------------===//
// Vector related lowering:
//===----------------------------------------------------------------------===//

static ConstantSDNode *
getVecImm(SDNode *N) {
  SDValue OpVal(0, 0);

  // Check to see if this buildvec has a single non-undef value in its elements.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    if (OpVal.getNode() == 0)
      OpVal = N->getOperand(i);
    else if (OpVal != N->getOperand(i))
      return 0;
  }

  if (OpVal.getNode() != 0) {
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal)) {
      return CN;
    }
  }

  return 0;
}

/// get_vec_i18imm - Test if this vector is a vector filled with the same value
/// and the value fits into an unsigned 18-bit constant, and if so, return the
/// constant
SDValue SPU::get_vec_u18imm(SDNode *N, SelectionDAG &DAG,
                              EVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    uint64_t Value = CN->getZExtValue();
    if (ValueType == MVT::i64) {
      uint64_t UValue = CN->getZExtValue();
      uint32_t upper = uint32_t(UValue >> 32);
      uint32_t lower = uint32_t(UValue);
      if (upper != lower)
        return SDValue();
      Value = Value >> 32;
    }
    if (Value <= 0x3ffff)
      return DAG.getTargetConstant(Value, ValueType);
  }

  return SDValue();
}

/// get_vec_i16imm - Test if this vector is a vector filled with the same value
/// and the value fits into a signed 16-bit constant, and if so, return the
/// constant
SDValue SPU::get_vec_i16imm(SDNode *N, SelectionDAG &DAG,
                              EVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    int64_t Value = CN->getSExtValue();
    if (ValueType == MVT::i64) {
      uint64_t UValue = CN->getZExtValue();
      uint32_t upper = uint32_t(UValue >> 32);
      uint32_t lower = uint32_t(UValue);
      if (upper != lower)
        return SDValue();
      Value = Value >> 32;
    }
    if (Value >= -(1 << 15) && Value <= ((1 << 15) - 1)) {
      return DAG.getTargetConstant(Value, ValueType);
    }
  }

  return SDValue();
}

/// get_vec_i10imm - Test if this vector is a vector filled with the same value
/// and the value fits into a signed 10-bit constant, and if so, return the
/// constant
SDValue SPU::get_vec_i10imm(SDNode *N, SelectionDAG &DAG,
                              EVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    int64_t Value = CN->getSExtValue();
    if (ValueType == MVT::i64) {
      uint64_t UValue = CN->getZExtValue();
      uint32_t upper = uint32_t(UValue >> 32);
      uint32_t lower = uint32_t(UValue);
      if (upper != lower)
        return SDValue();
      Value = Value >> 32;
    }
    if (isInt<10>(Value))
      return DAG.getTargetConstant(Value, ValueType);
  }

  return SDValue();
}

/// get_vec_i8imm - Test if this vector is a vector filled with the same value
/// and the value fits into a signed 8-bit constant, and if so, return the
/// constant.
///
/// @note: The incoming vector is v16i8 because that's the only way we can load
/// constant vectors. Thus, we test to see if the upper and lower bytes are the
/// same value.
SDValue SPU::get_vec_i8imm(SDNode *N, SelectionDAG &DAG,
                             EVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    int Value = (int) CN->getZExtValue();
    if (ValueType == MVT::i16
        && Value <= 0xffff                 /* truncated from uint64_t */
        && ((short) Value >> 8) == ((short) Value & 0xff))
      return DAG.getTargetConstant(Value & 0xff, ValueType);
    else if (ValueType == MVT::i8
             && (Value & 0xff) == Value)
      return DAG.getTargetConstant(Value, ValueType);
  }

  return SDValue();
}

/// get_ILHUvec_imm - Test if this vector is a vector filled with the same value
/// and the value fits into a signed 16-bit constant, and if so, return the
/// constant
SDValue SPU::get_ILHUvec_imm(SDNode *N, SelectionDAG &DAG,
                               EVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    uint64_t Value = CN->getZExtValue();
    if ((ValueType == MVT::i32
          && ((unsigned) Value & 0xffff0000) == (unsigned) Value)
        || (ValueType == MVT::i64 && (Value & 0xffff0000) == Value))
      return DAG.getTargetConstant(Value >> 16, ValueType);
  }

  return SDValue();
}

/// get_v4i32_imm - Catch-all for general 32-bit constant vectors
SDValue SPU::get_v4i32_imm(SDNode *N, SelectionDAG &DAG) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    return DAG.getTargetConstant((unsigned) CN->getZExtValue(), MVT::i32);
  }

  return SDValue();
}

/// get_v4i32_imm - Catch-all for general 64-bit constant vectors
SDValue SPU::get_v2i64_imm(SDNode *N, SelectionDAG &DAG) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    return DAG.getTargetConstant((unsigned) CN->getZExtValue(), MVT::i64);
  }

  return SDValue();
}

//! Lower a BUILD_VECTOR instruction creatively:
static SDValue
LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  EVT EltVT = VT.getVectorElementType();
  DebugLoc dl = Op.getDebugLoc();
  BuildVectorSDNode *BCN = dyn_cast<BuildVectorSDNode>(Op.getNode());
  assert(BCN != 0 && "Expected BuildVectorSDNode in SPU LowerBUILD_VECTOR");
  unsigned minSplatBits = EltVT.getSizeInBits();

  if (minSplatBits < 16)
    minSplatBits = 16;

  APInt APSplatBits, APSplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;

  if (!BCN->isConstantSplat(APSplatBits, APSplatUndef, SplatBitSize,
                            HasAnyUndefs, minSplatBits)
      || minSplatBits < SplatBitSize)
    return SDValue();   // Wasn't a constant vector or splat exceeded min

  uint64_t SplatBits = APSplatBits.getZExtValue();

  switch (VT.getSimpleVT().SimpleTy) {
  default:
    report_fatal_error("CellSPU: Unhandled VT in LowerBUILD_VECTOR, VT = " +
                       Twine(VT.getEVTString()));
    /*NOTREACHED*/
  case MVT::v4f32: {
    uint32_t Value32 = uint32_t(SplatBits);
    assert(SplatBitSize == 32
           && "LowerBUILD_VECTOR: Unexpected floating point vector element.");
    // NOTE: pretend the constant is an integer. LLVM won't load FP constants
    SDValue T = DAG.getConstant(Value32, MVT::i32);
    return DAG.getNode(ISD::BITCAST, dl, MVT::v4f32,
                       DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, T,T,T,T));
  }
  case MVT::v2f64: {
    uint64_t f64val = uint64_t(SplatBits);
    assert(SplatBitSize == 64
           && "LowerBUILD_VECTOR: 64-bit float vector size > 8 bytes.");
    // NOTE: pretend the constant is an integer. LLVM won't load FP constants
    SDValue T = DAG.getConstant(f64val, MVT::i64);
    return DAG.getNode(ISD::BITCAST, dl, MVT::v2f64,
                       DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v2i64, T, T));
  }
  case MVT::v16i8: {
   // 8-bit constants have to be expanded to 16-bits
   unsigned short Value16 = SplatBits /* | (SplatBits << 8) */;
   SmallVector<SDValue, 8> Ops;

   Ops.assign(8, DAG.getConstant(Value16, MVT::i16));
   return DAG.getNode(ISD::BITCAST, dl, VT,
                      DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v8i16, &Ops[0], Ops.size()));
  }
  case MVT::v8i16: {
    unsigned short Value16 = SplatBits;
    SDValue T = DAG.getConstant(Value16, EltVT);
    SmallVector<SDValue, 8> Ops;

    Ops.assign(8, T);
    return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Ops[0], Ops.size());
  }
  case MVT::v4i32: {
    SDValue T = DAG.getConstant(unsigned(SplatBits), VT.getVectorElementType());
    return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, T, T, T, T);
  }
  case MVT::v2i64: {
    return SPU::LowerV2I64Splat(VT, DAG, SplatBits, dl);
  }
  }
}

/*!
 */
SDValue
SPU::LowerV2I64Splat(EVT OpVT, SelectionDAG& DAG, uint64_t SplatVal,
                     DebugLoc dl) {
  uint32_t upper = uint32_t(SplatVal >> 32);
  uint32_t lower = uint32_t(SplatVal);

  if (upper == lower) {
    // Magic constant that can be matched by IL, ILA, et. al.
    SDValue Val = DAG.getTargetConstant(upper, MVT::i32);
    return DAG.getNode(ISD::BITCAST, dl, OpVT,
                       DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                   Val, Val, Val, Val));
  } else {
    bool upper_special, lower_special;

    // NOTE: This code creates common-case shuffle masks that can be easily
    // detected as common expressions. It is not attempting to create highly
    // specialized masks to replace any and all 0's, 0xff's and 0x80's.

    // Detect if the upper or lower half is a special shuffle mask pattern:
    upper_special = (upper == 0 || upper == 0xffffffff || upper == 0x80000000);
    lower_special = (lower == 0 || lower == 0xffffffff || lower == 0x80000000);

    // Both upper and lower are special, lower to a constant pool load:
    if (lower_special && upper_special) {
      SDValue UpperVal = DAG.getConstant(upper, MVT::i32);
      SDValue LowerVal = DAG.getConstant(lower, MVT::i32);
      SDValue BV = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                         UpperVal, LowerVal, UpperVal, LowerVal);
      return DAG.getNode(ISD::BITCAST, dl, OpVT, BV);
    }

    SDValue LO32;
    SDValue HI32;
    SmallVector<SDValue, 16> ShufBytes;
    SDValue Result;

    // Create lower vector if not a special pattern
    if (!lower_special) {
      SDValue LO32C = DAG.getConstant(lower, MVT::i32);
      LO32 = DAG.getNode(ISD::BITCAST, dl, OpVT,
                         DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                     LO32C, LO32C, LO32C, LO32C));
    }

    // Create upper vector if not a special pattern
    if (!upper_special) {
      SDValue HI32C = DAG.getConstant(upper, MVT::i32);
      HI32 = DAG.getNode(ISD::BITCAST, dl, OpVT,
                         DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                     HI32C, HI32C, HI32C, HI32C));
    }

    // If either upper or lower are special, then the two input operands are
    // the same (basically, one of them is a "don't care")
    if (lower_special)
      LO32 = HI32;
    if (upper_special)
      HI32 = LO32;

    for (int i = 0; i < 4; ++i) {
      uint64_t val = 0;
      for (int j = 0; j < 4; ++j) {
        SDValue V;
        bool process_upper, process_lower;
        val <<= 8;
        process_upper = (upper_special && (i & 1) == 0);
        process_lower = (lower_special && (i & 1) == 1);

        if (process_upper || process_lower) {
          if ((process_upper && upper == 0)
                  || (process_lower && lower == 0))
            val |= 0x80;
          else if ((process_upper && upper == 0xffffffff)
                  || (process_lower && lower == 0xffffffff))
            val |= 0xc0;
          else if ((process_upper && upper == 0x80000000)
                  || (process_lower && lower == 0x80000000))
            val |= (j == 0 ? 0xe0 : 0x80);
        } else
          val |= i * 4 + j + ((i & 1) * 16);
      }

      ShufBytes.push_back(DAG.getConstant(val, MVT::i32));
    }

    return DAG.getNode(SPUISD::SHUFB, dl, OpVT, HI32, LO32,
                       DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                   &ShufBytes[0], ShufBytes.size()));
  }
}

/// LowerVECTOR_SHUFFLE - Lower a vector shuffle (V1, V2, V3) to something on
/// which the Cell can operate. The code inspects V3 to ascertain whether the
/// permutation vector, V3, is monotonically increasing with one "exception"
/// element, e.g., (0, 1, _, 3). If this is the case, then generate a
/// SHUFFLE_MASK synthetic instruction. Otherwise, spill V3 to the constant pool.
/// In either case, the net result is going to eventually invoke SHUFB to
/// permute/shuffle the bytes from V1 and V2.
/// \note
/// SHUFFLE_MASK is eventually selected as one of the C*D instructions, generate
/// control word for byte/halfword/word insertion. This takes care of a single
/// element move from V2 into V1.
/// \note
/// SPUISD::SHUFB is eventually selected as Cell's <i>shufb</i> instructions.
static SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) {
  const ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op);
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();

  if (V2.getOpcode() == ISD::UNDEF) V2 = V1;

  // If we have a single element being moved from V1 to V2, this can be handled
  // using the C*[DX] compute mask instructions, but the vector elements have
  // to be monotonically increasing with one exception element, and the source
  // slot of the element to move must be the same as the destination.
  EVT VecVT = V1.getValueType();
  EVT EltVT = VecVT.getVectorElementType();
  unsigned EltsFromV2 = 0;
  unsigned V2EltOffset = 0;
  unsigned V2EltIdx0 = 0;
  unsigned CurrElt = 0;
  unsigned MaxElts = VecVT.getVectorNumElements();
  unsigned PrevElt = 0;
  bool monotonic = true;
  bool rotate = true;
  int rotamt=0;
  EVT maskVT;             // which of the c?d instructions to use

  if (EltVT == MVT::i8) {
    V2EltIdx0 = 16;
    maskVT = MVT::v16i8;
  } else if (EltVT == MVT::i16) {
    V2EltIdx0 = 8;
    maskVT = MVT::v8i16;
  } else if (EltVT == MVT::i32 || EltVT == MVT::f32) {
    V2EltIdx0 = 4;
    maskVT = MVT::v4i32;
  } else if (EltVT == MVT::i64 || EltVT == MVT::f64) {
    V2EltIdx0 = 2;
    maskVT = MVT::v2i64;
  } else
    llvm_unreachable("Unhandled vector type in LowerVECTOR_SHUFFLE");

  for (unsigned i = 0; i != MaxElts; ++i) {
    if (SVN->getMaskElt(i) < 0)
      continue;

    unsigned SrcElt = SVN->getMaskElt(i);

    if (monotonic) {
      if (SrcElt >= V2EltIdx0) {
        // TODO: optimize for the monotonic case when several consecutive
        // elements are taken form V2. Do we ever get such a case?
        if (EltsFromV2 == 0 && CurrElt == (SrcElt - V2EltIdx0))
          V2EltOffset = (SrcElt - V2EltIdx0) * (EltVT.getSizeInBits()/8);
        else
          monotonic = false;
        ++EltsFromV2;
      } else if (CurrElt != SrcElt) {
        monotonic = false;
      }

      ++CurrElt;
    }

    if (rotate) {
      if (PrevElt > 0 && SrcElt < MaxElts) {
        if ((PrevElt == SrcElt - 1)
            || (PrevElt == MaxElts - 1 && SrcElt == 0)) {
          PrevElt = SrcElt;
        } else {
          rotate = false;
        }
      } else if (i == 0 || (PrevElt==0 && SrcElt==1)) {
        // First time or after a "wrap around"
        rotamt = SrcElt-i;
        PrevElt = SrcElt;
      } else {
        // This isn't a rotation, takes elements from vector 2
        rotate = false;
      }
    }
  }

  if (EltsFromV2 == 1 && monotonic) {
    // Compute mask and shuffle
    EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

    // As SHUFFLE_MASK becomes a c?d instruction, feed it an address
    // R1 ($sp) is used here only as it is guaranteed to have last bits zero
    SDValue Pointer = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                                DAG.getRegister(SPU::R1, PtrVT),
                                DAG.getConstant(V2EltOffset, MVT::i32));
    SDValue ShufMaskOp = DAG.getNode(SPUISD::SHUFFLE_MASK, dl,
                                     maskVT, Pointer);

    // Use shuffle mask in SHUFB synthetic instruction:
    return DAG.getNode(SPUISD::SHUFB, dl, V1.getValueType(), V2, V1,
                       ShufMaskOp);
  } else if (rotate) {
    if (rotamt < 0)
      rotamt +=MaxElts;
    rotamt *= EltVT.getSizeInBits()/8;
    return DAG.getNode(SPUISD::ROTBYTES_LEFT, dl, V1.getValueType(),
                       V1, DAG.getConstant(rotamt, MVT::i16));
  } else {
   // Convert the SHUFFLE_VECTOR mask's input element units to the
   // actual bytes.
    unsigned BytesPerElement = EltVT.getSizeInBits()/8;

    SmallVector<SDValue, 16> ResultMask;
    for (unsigned i = 0, e = MaxElts; i != e; ++i) {
      unsigned SrcElt = SVN->getMaskElt(i) < 0 ? 0 : SVN->getMaskElt(i);

      for (unsigned j = 0; j < BytesPerElement; ++j)
        ResultMask.push_back(DAG.getConstant(SrcElt*BytesPerElement+j,MVT::i8));
    }
    SDValue VPermMask = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v16i8,
                                    &ResultMask[0], ResultMask.size());
    return DAG.getNode(SPUISD::SHUFB, dl, V1.getValueType(), V1, V2, VPermMask);
  }
}

static SDValue LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) {
  SDValue Op0 = Op.getOperand(0);                     // Op0 = the scalar
  DebugLoc dl = Op.getDebugLoc();

  if (Op0.getNode()->getOpcode() == ISD::Constant) {
    // For a constant, build the appropriate constant vector, which will
    // eventually simplify to a vector register load.

    ConstantSDNode *CN = cast<ConstantSDNode>(Op0.getNode());
    SmallVector<SDValue, 16> ConstVecValues;
    EVT VT;
    size_t n_copies;

    // Create a constant vector:
    switch (Op.getValueType().getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unexpected constant value type in "
                              "LowerSCALAR_TO_VECTOR");
    case MVT::v16i8: n_copies = 16; VT = MVT::i8; break;
    case MVT::v8i16: n_copies = 8; VT = MVT::i16; break;
    case MVT::v4i32: n_copies = 4; VT = MVT::i32; break;
    case MVT::v4f32: n_copies = 4; VT = MVT::f32; break;
    case MVT::v2i64: n_copies = 2; VT = MVT::i64; break;
    case MVT::v2f64: n_copies = 2; VT = MVT::f64; break;
    }

    SDValue CValue = DAG.getConstant(CN->getZExtValue(), VT);
    for (size_t j = 0; j < n_copies; ++j)
      ConstVecValues.push_back(CValue);

    return DAG.getNode(ISD::BUILD_VECTOR, dl, Op.getValueType(),
                       &ConstVecValues[0], ConstVecValues.size());
  } else {
    // Otherwise, copy the value from one register to another:
    switch (Op0.getValueType().getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unexpected value type in LowerSCALAR_TO_VECTOR");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
    case MVT::f32:
    case MVT::f64:
      return DAG.getNode(SPUISD::PREFSLOT2VEC, dl, Op.getValueType(), Op0, Op0);
    }
  }
}

static SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  SDValue N = Op.getOperand(0);
  SDValue Elt = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  SDValue retval;

  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Elt)) {
    // Constant argument:
    int EltNo = (int) C->getZExtValue();

    // sanity checks:
    if (VT == MVT::i8 && EltNo >= 16)
      llvm_unreachable("SPU LowerEXTRACT_VECTOR_ELT: i8 extraction slot > 15");
    else if (VT == MVT::i16 && EltNo >= 8)
      llvm_unreachable("SPU LowerEXTRACT_VECTOR_ELT: i16 extraction slot > 7");
    else if (VT == MVT::i32 && EltNo >= 4)
      llvm_unreachable("SPU LowerEXTRACT_VECTOR_ELT: i32 extraction slot > 4");
    else if (VT == MVT::i64 && EltNo >= 2)
      llvm_unreachable("SPU LowerEXTRACT_VECTOR_ELT: i64 extraction slot > 2");

    if (EltNo == 0 && (VT == MVT::i32 || VT == MVT::i64)) {
      // i32 and i64: Element 0 is the preferred slot
      return DAG.getNode(SPUISD::VEC2PREFSLOT, dl, VT, N);
    }

    // Need to generate shuffle mask and extract:
    int prefslot_begin = -1, prefslot_end = -1;
    int elt_byte = EltNo * VT.getSizeInBits() / 8;

    switch (VT.getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Invalid value type!");
    case MVT::i8: {
      prefslot_begin = prefslot_end = 3;
      break;
    }
    case MVT::i16: {
      prefslot_begin = 2; prefslot_end = 3;
      break;
    }
    case MVT::i32:
    case MVT::f32: {
      prefslot_begin = 0; prefslot_end = 3;
      break;
    }
    case MVT::i64:
    case MVT::f64: {
      prefslot_begin = 0; prefslot_end = 7;
      break;
    }
    }

    assert(prefslot_begin != -1 && prefslot_end != -1 &&
           "LowerEXTRACT_VECTOR_ELT: preferred slots uninitialized");

    unsigned int ShufBytes[16] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    for (int i = 0; i < 16; ++i) {
      // zero fill uppper part of preferred slot, don't care about the
      // other slots:
      unsigned int mask_val;
      if (i <= prefslot_end) {
        mask_val =
          ((i < prefslot_begin)
           ? 0x80
           : elt_byte + (i - prefslot_begin));

        ShufBytes[i] = mask_val;
      } else
        ShufBytes[i] = ShufBytes[i % (prefslot_end + 1)];
    }

    SDValue ShufMask[4];
    for (unsigned i = 0; i < sizeof(ShufMask)/sizeof(ShufMask[0]); ++i) {
      unsigned bidx = i * 4;
      unsigned int bits = ((ShufBytes[bidx] << 24) |
                           (ShufBytes[bidx+1] << 16) |
                           (ShufBytes[bidx+2] << 8) |
                           ShufBytes[bidx+3]);
      ShufMask[i] = DAG.getConstant(bits, MVT::i32);
    }

    SDValue ShufMaskVec =
      DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                  &ShufMask[0], sizeof(ShufMask)/sizeof(ShufMask[0]));

    retval = DAG.getNode(SPUISD::VEC2PREFSLOT, dl, VT,
                         DAG.getNode(SPUISD::SHUFB, dl, N.getValueType(),
                                     N, N, ShufMaskVec));
  } else {
    // Variable index: Rotate the requested element into slot 0, then replicate
    // slot 0 across the vector
    EVT VecVT = N.getValueType();
    if (!VecVT.isSimple() || !VecVT.isVector()) {
      report_fatal_error("LowerEXTRACT_VECTOR_ELT: Must have a simple, 128-bit"
                        "vector type!");
    }

    // Make life easier by making sure the index is zero-extended to i32
    if (Elt.getValueType() != MVT::i32)
      Elt = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Elt);

    // Scale the index to a bit/byte shift quantity
    APInt scaleFactor =
            APInt(32, uint64_t(16 / N.getValueType().getVectorNumElements()), false);
    unsigned scaleShift = scaleFactor.logBase2();
    SDValue vecShift;

    if (scaleShift > 0) {
      // Scale the shift factor:
      Elt = DAG.getNode(ISD::SHL, dl, MVT::i32, Elt,
                        DAG.getConstant(scaleShift, MVT::i32));
    }

    vecShift = DAG.getNode(SPUISD::SHL_BYTES, dl, VecVT, N, Elt);

    // Replicate the bytes starting at byte 0 across the entire vector (for
    // consistency with the notion of a unified register set)
    SDValue replicate;

    switch (VT.getSimpleVT().SimpleTy) {
    default:
      report_fatal_error("LowerEXTRACT_VECTOR_ELT(varable): Unhandled vector"
                        "type");
      /*NOTREACHED*/
    case MVT::i8: {
      SDValue factor = DAG.getConstant(0x00000000, MVT::i32);
      replicate = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                              factor, factor, factor, factor);
      break;
    }
    case MVT::i16: {
      SDValue factor = DAG.getConstant(0x00010001, MVT::i32);
      replicate = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                              factor, factor, factor, factor);
      break;
    }
    case MVT::i32:
    case MVT::f32: {
      SDValue factor = DAG.getConstant(0x00010203, MVT::i32);
      replicate = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                              factor, factor, factor, factor);
      break;
    }
    case MVT::i64:
    case MVT::f64: {
      SDValue loFactor = DAG.getConstant(0x00010203, MVT::i32);
      SDValue hiFactor = DAG.getConstant(0x04050607, MVT::i32);
      replicate = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                              loFactor, hiFactor, loFactor, hiFactor);
      break;
    }
    }

    retval = DAG.getNode(SPUISD::VEC2PREFSLOT, dl, VT,
                         DAG.getNode(SPUISD::SHUFB, dl, VecVT,
                                     vecShift, vecShift, replicate));
  }

  return retval;
}

static SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  SDValue VecOp = Op.getOperand(0);
  SDValue ValOp = Op.getOperand(1);
  SDValue IdxOp = Op.getOperand(2);
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT eltVT = ValOp.getValueType();

  // use 0 when the lane to insert to is 'undef'
  int64_t Offset=0;
  if (IdxOp.getOpcode() != ISD::UNDEF) {
    ConstantSDNode *CN = cast<ConstantSDNode>(IdxOp);
    assert(CN != 0 && "LowerINSERT_VECTOR_ELT: Index is not constant!");
    Offset = (CN->getSExtValue()) * eltVT.getSizeInBits()/8;
  }

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  // Use $sp ($1) because it's always 16-byte aligned and it's available:
  SDValue Pointer = DAG.getNode(SPUISD::IndirectAddr, dl, PtrVT,
                                DAG.getRegister(SPU::R1, PtrVT),
                                DAG.getConstant(Offset, PtrVT));
  // widen the mask when dealing with half vectors
  EVT maskVT = EVT::getVectorVT(*(DAG.getContext()), VT.getVectorElementType(),
                                128/ VT.getVectorElementType().getSizeInBits());
  SDValue ShufMask = DAG.getNode(SPUISD::SHUFFLE_MASK, dl, maskVT, Pointer);

  SDValue result =
    DAG.getNode(SPUISD::SHUFB, dl, VT,
                DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, ValOp),
                VecOp,
                DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, ShufMask));

  return result;
}

static SDValue LowerI8Math(SDValue Op, SelectionDAG &DAG, unsigned Opc,
                           const TargetLowering &TLI)
{
  SDValue N0 = Op.getOperand(0);      // Everything has at least one operand
  DebugLoc dl = Op.getDebugLoc();
  EVT ShiftVT = TLI.getShiftAmountTy(N0.getValueType());

  assert(Op.getValueType() == MVT::i8);
  switch (Opc) {
  default:
    llvm_unreachable("Unhandled i8 math operator");
  case ISD::ADD: {
    // 8-bit addition: Promote the arguments up to 16-bits and truncate
    // the result:
    SDValue N1 = Op.getOperand(1);
    N0 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i16, N0);
    N1 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i16, N1);
    return DAG.getNode(ISD::TRUNCATE, dl, MVT::i8,
                       DAG.getNode(Opc, dl, MVT::i16, N0, N1));

  }

  case ISD::SUB: {
    // 8-bit subtraction: Promote the arguments up to 16-bits and truncate
    // the result:
    SDValue N1 = Op.getOperand(1);
    N0 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i16, N0);
    N1 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i16, N1);
    return DAG.getNode(ISD::TRUNCATE, dl, MVT::i8,
                       DAG.getNode(Opc, dl, MVT::i16, N0, N1));
  }
  case ISD::ROTR:
  case ISD::ROTL: {
    SDValue N1 = Op.getOperand(1);
    EVT N1VT = N1.getValueType();

    N0 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, N0);
    if (!N1VT.bitsEq(ShiftVT)) {
      unsigned N1Opc = N1.getValueType().bitsLT(ShiftVT)
                       ? ISD::ZERO_EXTEND
                       : ISD::TRUNCATE;
      N1 = DAG.getNode(N1Opc, dl, ShiftVT, N1);
    }

    // Replicate lower 8-bits into upper 8:
    SDValue ExpandArg =
      DAG.getNode(ISD::OR, dl, MVT::i16, N0,
                  DAG.getNode(ISD::SHL, dl, MVT::i16,
                              N0, DAG.getConstant(8, MVT::i32)));

    // Truncate back down to i8
    return DAG.getNode(ISD::TRUNCATE, dl, MVT::i8,
                       DAG.getNode(Opc, dl, MVT::i16, ExpandArg, N1));
  }
  case ISD::SRL:
  case ISD::SHL: {
    SDValue N1 = Op.getOperand(1);
    EVT N1VT = N1.getValueType();

    N0 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, N0);
    if (!N1VT.bitsEq(ShiftVT)) {
      unsigned N1Opc = ISD::ZERO_EXTEND;

      if (N1.getValueType().bitsGT(ShiftVT))
        N1Opc = ISD::TRUNCATE;

      N1 = DAG.getNode(N1Opc, dl, ShiftVT, N1);
    }

    return DAG.getNode(ISD::TRUNCATE, dl, MVT::i8,
                       DAG.getNode(Opc, dl, MVT::i16, N0, N1));
  }
  case ISD::SRA: {
    SDValue N1 = Op.getOperand(1);
    EVT N1VT = N1.getValueType();

    N0 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i16, N0);
    if (!N1VT.bitsEq(ShiftVT)) {
      unsigned N1Opc = ISD::SIGN_EXTEND;

      if (N1VT.bitsGT(ShiftVT))
        N1Opc = ISD::TRUNCATE;
      N1 = DAG.getNode(N1Opc, dl, ShiftVT, N1);
    }

    return DAG.getNode(ISD::TRUNCATE, dl, MVT::i8,
                       DAG.getNode(Opc, dl, MVT::i16, N0, N1));
  }
  case ISD::MUL: {
    SDValue N1 = Op.getOperand(1);

    N0 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i16, N0);
    N1 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i16, N1);
    return DAG.getNode(ISD::TRUNCATE, dl, MVT::i8,
                       DAG.getNode(Opc, dl, MVT::i16, N0, N1));
  }
  }
}

//! Lower byte immediate operations for v16i8 vectors:
static SDValue
LowerByteImmed(SDValue Op, SelectionDAG &DAG) {
  SDValue ConstVec;
  SDValue Arg;
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();

  ConstVec = Op.getOperand(0);
  Arg = Op.getOperand(1);
  if (ConstVec.getNode()->getOpcode() != ISD::BUILD_VECTOR) {
    if (ConstVec.getNode()->getOpcode() == ISD::BITCAST) {
      ConstVec = ConstVec.getOperand(0);
    } else {
      ConstVec = Op.getOperand(1);
      Arg = Op.getOperand(0);
      if (ConstVec.getNode()->getOpcode() == ISD::BITCAST) {
        ConstVec = ConstVec.getOperand(0);
      }
    }
  }

  if (ConstVec.getNode()->getOpcode() == ISD::BUILD_VECTOR) {
    BuildVectorSDNode *BCN = dyn_cast<BuildVectorSDNode>(ConstVec.getNode());
    assert(BCN != 0 && "Expected BuildVectorSDNode in SPU LowerByteImmed");

    APInt APSplatBits, APSplatUndef;
    unsigned SplatBitSize;
    bool HasAnyUndefs;
    unsigned minSplatBits = VT.getVectorElementType().getSizeInBits();

    if (BCN->isConstantSplat(APSplatBits, APSplatUndef, SplatBitSize,
                              HasAnyUndefs, minSplatBits)
        && minSplatBits <= SplatBitSize) {
      uint64_t SplatBits = APSplatBits.getZExtValue();
      SDValue tc = DAG.getTargetConstant(SplatBits & 0xff, MVT::i8);

      SmallVector<SDValue, 16> tcVec;
      tcVec.assign(16, tc);
      return DAG.getNode(Op.getNode()->getOpcode(), dl, VT, Arg,
                         DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &tcVec[0], tcVec.size()));
    }
  }

  // These operations (AND, OR, XOR) are legal, they just couldn't be custom
  // lowered.  Return the operation, rather than a null SDValue.
  return Op;
}

//! Custom lowering for CTPOP (count population)
/*!
  Custom lowering code that counts the number ones in the input
  operand. SPU has such an instruction, but it counts the number of
  ones per byte, which then have to be accumulated.
*/
static SDValue LowerCTPOP(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  EVT vecVT = EVT::getVectorVT(*DAG.getContext(),
                               VT, (128 / VT.getSizeInBits()));
  DebugLoc dl = Op.getDebugLoc();

  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("Invalid value type!");
  case MVT::i8: {
    SDValue N = Op.getOperand(0);
    SDValue Elt0 = DAG.getConstant(0, MVT::i32);

    SDValue Promote = DAG.getNode(SPUISD::PREFSLOT2VEC, dl, vecVT, N, N);
    SDValue CNTB = DAG.getNode(SPUISD::CNTB, dl, vecVT, Promote);

    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i8, CNTB, Elt0);
  }

  case MVT::i16: {
    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();

    unsigned CNTB_reg = RegInfo.createVirtualRegister(&SPU::R16CRegClass);

    SDValue N = Op.getOperand(0);
    SDValue Elt0 = DAG.getConstant(0, MVT::i16);
    SDValue Mask0 = DAG.getConstant(0x0f, MVT::i16);
    SDValue Shift1 = DAG.getConstant(8, MVT::i32);

    SDValue Promote = DAG.getNode(SPUISD::PREFSLOT2VEC, dl, vecVT, N, N);
    SDValue CNTB = DAG.getNode(SPUISD::CNTB, dl, vecVT, Promote);

    // CNTB_result becomes the chain to which all of the virtual registers
    // CNTB_reg, SUM1_reg become associated:
    SDValue CNTB_result =
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16, CNTB, Elt0);

    SDValue CNTB_rescopy =
      DAG.getCopyToReg(CNTB_result, dl, CNTB_reg, CNTB_result);

    SDValue Tmp1 = DAG.getCopyFromReg(CNTB_rescopy, dl, CNTB_reg, MVT::i16);

    return DAG.getNode(ISD::AND, dl, MVT::i16,
                       DAG.getNode(ISD::ADD, dl, MVT::i16,
                                   DAG.getNode(ISD::SRL, dl, MVT::i16,
                                               Tmp1, Shift1),
                                   Tmp1),
                       Mask0);
  }

  case MVT::i32: {
    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();

    unsigned CNTB_reg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);
    unsigned SUM1_reg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);

    SDValue N = Op.getOperand(0);
    SDValue Elt0 = DAG.getConstant(0, MVT::i32);
    SDValue Mask0 = DAG.getConstant(0xff, MVT::i32);
    SDValue Shift1 = DAG.getConstant(16, MVT::i32);
    SDValue Shift2 = DAG.getConstant(8, MVT::i32);

    SDValue Promote = DAG.getNode(SPUISD::PREFSLOT2VEC, dl, vecVT, N, N);
    SDValue CNTB = DAG.getNode(SPUISD::CNTB, dl, vecVT, Promote);

    // CNTB_result becomes the chain to which all of the virtual registers
    // CNTB_reg, SUM1_reg become associated:
    SDValue CNTB_result =
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32, CNTB, Elt0);

    SDValue CNTB_rescopy =
      DAG.getCopyToReg(CNTB_result, dl, CNTB_reg, CNTB_result);

    SDValue Comp1 =
      DAG.getNode(ISD::SRL, dl, MVT::i32,
                  DAG.getCopyFromReg(CNTB_rescopy, dl, CNTB_reg, MVT::i32),
                  Shift1);

    SDValue Sum1 =
      DAG.getNode(ISD::ADD, dl, MVT::i32, Comp1,
                  DAG.getCopyFromReg(CNTB_rescopy, dl, CNTB_reg, MVT::i32));

    SDValue Sum1_rescopy =
      DAG.getCopyToReg(CNTB_result, dl, SUM1_reg, Sum1);

    SDValue Comp2 =
      DAG.getNode(ISD::SRL, dl, MVT::i32,
                  DAG.getCopyFromReg(Sum1_rescopy, dl, SUM1_reg, MVT::i32),
                  Shift2);
    SDValue Sum2 =
      DAG.getNode(ISD::ADD, dl, MVT::i32, Comp2,
                  DAG.getCopyFromReg(Sum1_rescopy, dl, SUM1_reg, MVT::i32));

    return DAG.getNode(ISD::AND, dl, MVT::i32, Sum2, Mask0);
  }

  case MVT::i64:
    break;
  }

  return SDValue();
}

//! Lower ISD::FP_TO_SINT, ISD::FP_TO_UINT for i32
/*!
 f32->i32 passes through unchanged, whereas f64->i32 expands to a libcall.
 All conversions to i64 are expanded to a libcall.
 */
static SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG,
                              const SPUTargetLowering &TLI) {
  EVT OpVT = Op.getValueType();
  SDValue Op0 = Op.getOperand(0);
  EVT Op0VT = Op0.getValueType();

  if ((OpVT == MVT::i32 && Op0VT == MVT::f64)
      || OpVT == MVT::i64) {
    // Convert f32 / f64 to i32 / i64 via libcall.
    RTLIB::Libcall LC =
            (Op.getOpcode() == ISD::FP_TO_SINT)
             ? RTLIB::getFPTOSINT(Op0VT, OpVT)
             : RTLIB::getFPTOUINT(Op0VT, OpVT);
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpectd fp-to-int conversion!");
    SDValue Dummy;
    return ExpandLibCall(LC, Op, DAG, false, Dummy, TLI);
  }

  return Op;
}

//! Lower ISD::SINT_TO_FP, ISD::UINT_TO_FP for i32
/*!
 i32->f32 passes through unchanged, whereas i32->f64 is expanded to a libcall.
 All conversions from i64 are expanded to a libcall.
 */
static SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG,
                              const SPUTargetLowering &TLI) {
  EVT OpVT = Op.getValueType();
  SDValue Op0 = Op.getOperand(0);
  EVT Op0VT = Op0.getValueType();

  if ((OpVT == MVT::f64 && Op0VT == MVT::i32)
      || Op0VT == MVT::i64) {
    // Convert i32, i64 to f64 via libcall:
    RTLIB::Libcall LC =
            (Op.getOpcode() == ISD::SINT_TO_FP)
             ? RTLIB::getSINTTOFP(Op0VT, OpVT)
             : RTLIB::getUINTTOFP(Op0VT, OpVT);
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpectd int-to-fp conversion!");
    SDValue Dummy;
    return ExpandLibCall(LC, Op, DAG, false, Dummy, TLI);
  }

  return Op;
}

//! Lower ISD::SETCC
/*!
 This handles MVT::f64 (double floating point) condition lowering
 */
static SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG,
                          const TargetLowering &TLI) {
  CondCodeSDNode *CC = dyn_cast<CondCodeSDNode>(Op.getOperand(2));
  DebugLoc dl = Op.getDebugLoc();
  assert(CC != 0 && "LowerSETCC: CondCodeSDNode should not be null here!\n");

  SDValue lhs = Op.getOperand(0);
  SDValue rhs = Op.getOperand(1);
  EVT lhsVT = lhs.getValueType();
  assert(lhsVT == MVT::f64 && "LowerSETCC: type other than MVT::64\n");

  EVT ccResultVT = TLI.getSetCCResultType(lhs.getValueType());
  APInt ccResultOnes = APInt::getAllOnesValue(ccResultVT.getSizeInBits());
  EVT IntVT(MVT::i64);

  // Take advantage of the fact that (truncate (sra arg, 32)) is efficiently
  // selected to a NOP:
  SDValue i64lhs = DAG.getNode(ISD::BITCAST, dl, IntVT, lhs);
  SDValue lhsHi32 =
          DAG.getNode(ISD::TRUNCATE, dl, MVT::i32,
                      DAG.getNode(ISD::SRL, dl, IntVT,
                                  i64lhs, DAG.getConstant(32, MVT::i32)));
  SDValue lhsHi32abs =
          DAG.getNode(ISD::AND, dl, MVT::i32,
                      lhsHi32, DAG.getConstant(0x7fffffff, MVT::i32));
  SDValue lhsLo32 =
          DAG.getNode(ISD::TRUNCATE, dl, MVT::i32, i64lhs);

  // SETO and SETUO only use the lhs operand:
  if (CC->get() == ISD::SETO) {
    // Evaluates to true if Op0 is not [SQ]NaN - lowers to the inverse of
    // SETUO
    APInt ccResultAllOnes = APInt::getAllOnesValue(ccResultVT.getSizeInBits());
    return DAG.getNode(ISD::XOR, dl, ccResultVT,
                       DAG.getSetCC(dl, ccResultVT,
                                    lhs, DAG.getConstantFP(0.0, lhsVT),
                                    ISD::SETUO),
                       DAG.getConstant(ccResultAllOnes, ccResultVT));
  } else if (CC->get() == ISD::SETUO) {
    // Evaluates to true if Op0 is [SQ]NaN
    return DAG.getNode(ISD::AND, dl, ccResultVT,
                       DAG.getSetCC(dl, ccResultVT,
                                    lhsHi32abs,
                                    DAG.getConstant(0x7ff00000, MVT::i32),
                                    ISD::SETGE),
                       DAG.getSetCC(dl, ccResultVT,
                                    lhsLo32,
                                    DAG.getConstant(0, MVT::i32),
                                    ISD::SETGT));
  }

  SDValue i64rhs = DAG.getNode(ISD::BITCAST, dl, IntVT, rhs);
  SDValue rhsHi32 =
          DAG.getNode(ISD::TRUNCATE, dl, MVT::i32,
                      DAG.getNode(ISD::SRL, dl, IntVT,
                                  i64rhs, DAG.getConstant(32, MVT::i32)));

  // If a value is negative, subtract from the sign magnitude constant:
  SDValue signMag2TC = DAG.getConstant(0x8000000000000000ULL, IntVT);

  // Convert the sign-magnitude representation into 2's complement:
  SDValue lhsSelectMask = DAG.getNode(ISD::SRA, dl, ccResultVT,
                                      lhsHi32, DAG.getConstant(31, MVT::i32));
  SDValue lhsSignMag2TC = DAG.getNode(ISD::SUB, dl, IntVT, signMag2TC, i64lhs);
  SDValue lhsSelect =
          DAG.getNode(ISD::SELECT, dl, IntVT,
                      lhsSelectMask, lhsSignMag2TC, i64lhs);

  SDValue rhsSelectMask = DAG.getNode(ISD::SRA, dl, ccResultVT,
                                      rhsHi32, DAG.getConstant(31, MVT::i32));
  SDValue rhsSignMag2TC = DAG.getNode(ISD::SUB, dl, IntVT, signMag2TC, i64rhs);
  SDValue rhsSelect =
          DAG.getNode(ISD::SELECT, dl, IntVT,
                      rhsSelectMask, rhsSignMag2TC, i64rhs);

  unsigned compareOp;

  switch (CC->get()) {
  case ISD::SETOEQ:
  case ISD::SETUEQ:
    compareOp = ISD::SETEQ; break;
  case ISD::SETOGT:
  case ISD::SETUGT:
    compareOp = ISD::SETGT; break;
  case ISD::SETOGE:
  case ISD::SETUGE:
    compareOp = ISD::SETGE; break;
  case ISD::SETOLT:
  case ISD::SETULT:
    compareOp = ISD::SETLT; break;
  case ISD::SETOLE:
  case ISD::SETULE:
    compareOp = ISD::SETLE; break;
  case ISD::SETUNE:
  case ISD::SETONE:
    compareOp = ISD::SETNE; break;
  default:
    report_fatal_error("CellSPU ISel Select: unimplemented f64 condition");
  }

  SDValue result =
          DAG.getSetCC(dl, ccResultVT, lhsSelect, rhsSelect,
                       (ISD::CondCode) compareOp);

  if ((CC->get() & 0x8) == 0) {
    // Ordered comparison:
    SDValue lhsNaN = DAG.getSetCC(dl, ccResultVT,
                                  lhs, DAG.getConstantFP(0.0, MVT::f64),
                                  ISD::SETO);
    SDValue rhsNaN = DAG.getSetCC(dl, ccResultVT,
                                  rhs, DAG.getConstantFP(0.0, MVT::f64),
                                  ISD::SETO);
    SDValue ordered = DAG.getNode(ISD::AND, dl, ccResultVT, lhsNaN, rhsNaN);

    result = DAG.getNode(ISD::AND, dl, ccResultVT, ordered, result);
  }

  return result;
}

//! Lower ISD::SELECT_CC
/*!
  ISD::SELECT_CC can (generally) be implemented directly on the SPU using the
  SELB instruction.

  \note Need to revisit this in the future: if the code path through the true
  and false value computations is longer than the latency of a branch (6
  cycles), then it would be more advantageous to branch and insert a new basic
  block and branch on the condition. However, this code does not make that
  assumption, given the simplisitc uses so far.
 */

static SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG,
                              const TargetLowering &TLI) {
  EVT VT = Op.getValueType();
  SDValue lhs = Op.getOperand(0);
  SDValue rhs = Op.getOperand(1);
  SDValue trueval = Op.getOperand(2);
  SDValue falseval = Op.getOperand(3);
  SDValue condition = Op.getOperand(4);
  DebugLoc dl = Op.getDebugLoc();

  // NOTE: SELB's arguments: $rA, $rB, $mask
  //
  // SELB selects bits from $rA where bits in $mask are 0, bits from $rB
  // where bits in $mask are 1. CCond will be inverted, having 1s where the
  // condition was true and 0s where the condition was false. Hence, the
  // arguments to SELB get reversed.

  // Note: Really should be ISD::SELECT instead of SPUISD::SELB, but LLVM's
  // legalizer insists on combining SETCC/SELECT into SELECT_CC, so we end up
  // with another "cannot select select_cc" assert:

  SDValue compare = DAG.getNode(ISD::SETCC, dl,
                                TLI.getSetCCResultType(Op.getValueType()),
                                lhs, rhs, condition);
  return DAG.getNode(SPUISD::SELB, dl, VT, falseval, trueval, compare);
}

//! Custom lower ISD::TRUNCATE
static SDValue LowerTRUNCATE(SDValue Op, SelectionDAG &DAG)
{
  // Type to truncate to
  EVT VT = Op.getValueType();
  MVT simpleVT = VT.getSimpleVT();
  EVT VecVT = EVT::getVectorVT(*DAG.getContext(),
                               VT, (128 / VT.getSizeInBits()));
  DebugLoc dl = Op.getDebugLoc();

  // Type to truncate from
  SDValue Op0 = Op.getOperand(0);
  EVT Op0VT = Op0.getValueType();

  if (Op0VT == MVT::i128 && simpleVT == MVT::i64) {
    // Create shuffle mask, least significant doubleword of quadword
    unsigned maskHigh = 0x08090a0b;
    unsigned maskLow = 0x0c0d0e0f;
    // Use a shuffle to perform the truncation
    SDValue shufMask = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                   DAG.getConstant(maskHigh, MVT::i32),
                                   DAG.getConstant(maskLow, MVT::i32),
                                   DAG.getConstant(maskHigh, MVT::i32),
                                   DAG.getConstant(maskLow, MVT::i32));

    SDValue truncShuffle = DAG.getNode(SPUISD::SHUFB, dl, VecVT,
                                       Op0, Op0, shufMask);

    return DAG.getNode(SPUISD::VEC2PREFSLOT, dl, VT, truncShuffle);
  }

  return SDValue();             // Leave the truncate unmolested
}

/*!
 * Emit the instruction sequence for i64/i32 -> i128 sign extend. The basic
 * algorithm is to duplicate the sign bit using rotmai to generate at
 * least one byte full of sign bits. Then propagate the "sign-byte" into
 * the leftmost words and the i64/i32 into the rightmost words using shufb.
 *
 * @param Op The sext operand
 * @param DAG The current DAG
 * @return The SDValue with the entire instruction sequence
 */
static SDValue LowerSIGN_EXTEND(SDValue Op, SelectionDAG &DAG)
{
  DebugLoc dl = Op.getDebugLoc();

  // Type to extend to
  MVT OpVT = Op.getValueType().getSimpleVT();

  // Type to extend from
  SDValue Op0 = Op.getOperand(0);
  MVT Op0VT = Op0.getValueType().getSimpleVT();

  // extend i8 & i16 via i32
  if (Op0VT == MVT::i8 || Op0VT == MVT::i16) {
    Op0 = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i32, Op0);
    Op0VT = MVT::i32;
  }

  // The type to extend to needs to be a i128 and
  // the type to extend from needs to be i64 or i32.
  assert((OpVT == MVT::i128 && (Op0VT == MVT::i64 || Op0VT == MVT::i32)) &&
          "LowerSIGN_EXTEND: input and/or output operand have wrong size");
  (void)OpVT;

  // Create shuffle mask
  unsigned mask1 = 0x10101010; // byte 0 - 3 and 4 - 7
  unsigned mask2 = Op0VT == MVT::i64 ? 0x00010203 : 0x10101010; // byte  8 - 11
  unsigned mask3 = Op0VT == MVT::i64 ? 0x04050607 : 0x00010203; // byte 12 - 15
  SDValue shufMask = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                 DAG.getConstant(mask1, MVT::i32),
                                 DAG.getConstant(mask1, MVT::i32),
                                 DAG.getConstant(mask2, MVT::i32),
                                 DAG.getConstant(mask3, MVT::i32));

  // Word wise arithmetic right shift to generate at least one byte
  // that contains sign bits.
  MVT mvt = Op0VT == MVT::i64 ? MVT::v2i64 : MVT::v4i32;
  SDValue sraVal = DAG.getNode(ISD::SRA,
                 dl,
                 mvt,
                 DAG.getNode(SPUISD::PREFSLOT2VEC, dl, mvt, Op0, Op0),
                 DAG.getConstant(31, MVT::i32));

  // reinterpret as a i128 (SHUFB requires it). This gets lowered away.
  SDValue extended = SDValue(DAG.getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                        dl, Op0VT, Op0,
                                        DAG.getTargetConstant(
                                                  SPU::GPRCRegClass.getID(),
                                                  MVT::i32)), 0);
  // Shuffle bytes - Copy the sign bits into the upper 64 bits
  // and the input value into the lower 64 bits.
  SDValue extShuffle = DAG.getNode(SPUISD::SHUFB, dl, mvt,
        extended, sraVal, shufMask);
  return DAG.getNode(ISD::BITCAST, dl, MVT::i128, extShuffle);
}

//! Custom (target-specific) lowering entry point
/*!
  This is where LLVM's DAG selection process calls to do target-specific
  lowering of nodes.
 */
SDValue
SPUTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const
{
  unsigned Opc = (unsigned) Op.getOpcode();
  EVT VT = Op.getValueType();

  switch (Opc) {
  default: {
#ifndef NDEBUG
    errs() << "SPUTargetLowering::LowerOperation(): need to lower this!\n";
    errs() << "Op.getOpcode() = " << Opc << "\n";
    errs() << "*Op.getNode():\n";
    Op.getNode()->dump();
#endif
    llvm_unreachable(0);
  }
  case ISD::LOAD:
  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
    return LowerLOAD(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::STORE:
    return LowerSTORE(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::ConstantFP:
    return LowerConstantFP(Op, DAG);

  // i8, i64 math ops:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::ROTR:
  case ISD::ROTL:
  case ISD::SRL:
  case ISD::SHL:
  case ISD::SRA: {
    if (VT == MVT::i8)
      return LowerI8Math(Op, DAG, Opc, *this);
    break;
  }

  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return LowerFP_TO_INT(Op, DAG, *this);

  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return LowerINT_TO_FP(Op, DAG, *this);

  // Vector-related lowering.
  case ISD::BUILD_VECTOR:
    return LowerBUILD_VECTOR(Op, DAG);
  case ISD::SCALAR_TO_VECTOR:
    return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:
    return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    return LowerINSERT_VECTOR_ELT(Op, DAG);

  // Look for ANDBI, ORBI and XORBI opportunities and lower appropriately:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    return LowerByteImmed(Op, DAG);

  // Vector and i8 multiply:
  case ISD::MUL:
    if (VT == MVT::i8)
      return LowerI8Math(Op, DAG, Opc, *this);

  case ISD::CTPOP:
    return LowerCTPOP(Op, DAG);

  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG, *this);

  case ISD::SETCC:
    return LowerSETCC(Op, DAG, *this);

  case ISD::TRUNCATE:
    return LowerTRUNCATE(Op, DAG);

  case ISD::SIGN_EXTEND:
    return LowerSIGN_EXTEND(Op, DAG);
  }

  return SDValue();
}

void SPUTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue>&Results,
                                           SelectionDAG &DAG) const
{
#if 0
  unsigned Opc = (unsigned) N->getOpcode();
  EVT OpVT = N->getValueType(0);

  switch (Opc) {
  default: {
    errs() << "SPUTargetLowering::ReplaceNodeResults(): need to fix this!\n";
    errs() << "Op.getOpcode() = " << Opc << "\n";
    errs() << "*Op.getNode():\n";
    N->dump();
    abort();
    /*NOTREACHED*/
  }
  }
#endif

  /* Otherwise, return unchanged */
}

//===----------------------------------------------------------------------===//
// Target Optimization Hooks
//===----------------------------------------------------------------------===//

SDValue
SPUTargetLowering::PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const
{
#if 0
  TargetMachine &TM = getTargetMachine();
#endif
  const SPUSubtarget *ST = SPUTM.getSubtargetImpl();
  SelectionDAG &DAG = DCI.DAG;
  SDValue Op0 = N->getOperand(0);       // everything has at least one operand
  EVT NodeVT = N->getValueType(0);      // The node's value type
  EVT Op0VT = Op0.getValueType();       // The first operand's result
  SDValue Result;                       // Initially, empty result
  DebugLoc dl = N->getDebugLoc();

  switch (N->getOpcode()) {
  default: break;
  case ISD::ADD: {
    SDValue Op1 = N->getOperand(1);

    if (Op0.getOpcode() == SPUISD::IndirectAddr
        || Op1.getOpcode() == SPUISD::IndirectAddr) {
      // Normalize the operands to reduce repeated code
      SDValue IndirectArg = Op0, AddArg = Op1;

      if (Op1.getOpcode() == SPUISD::IndirectAddr) {
        IndirectArg = Op1;
        AddArg = Op0;
      }

      if (isa<ConstantSDNode>(AddArg)) {
        ConstantSDNode *CN0 = cast<ConstantSDNode > (AddArg);
        SDValue IndOp1 = IndirectArg.getOperand(1);

        if (CN0->isNullValue()) {
          // (add (SPUindirect <arg>, <arg>), 0) ->
          // (SPUindirect <arg>, <arg>)

#if !defined(NDEBUG)
          if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
            errs() << "\n"
                 << "Replace: (add (SPUindirect <arg>, <arg>), 0)\n"
                 << "With:    (SPUindirect <arg>, <arg>)\n";
          }
#endif

          return IndirectArg;
        } else if (isa<ConstantSDNode>(IndOp1)) {
          // (add (SPUindirect <arg>, <const>), <const>) ->
          // (SPUindirect <arg>, <const + const>)
          ConstantSDNode *CN1 = cast<ConstantSDNode > (IndOp1);
          int64_t combinedConst = CN0->getSExtValue() + CN1->getSExtValue();
          SDValue combinedValue = DAG.getConstant(combinedConst, Op0VT);

#if !defined(NDEBUG)
          if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
            errs() << "\n"
                 << "Replace: (add (SPUindirect <arg>, " << CN1->getSExtValue()
                 << "), " << CN0->getSExtValue() << ")\n"
                 << "With:    (SPUindirect <arg>, "
                 << combinedConst << ")\n";
          }
#endif

          return DAG.getNode(SPUISD::IndirectAddr, dl, Op0VT,
                             IndirectArg, combinedValue);
        }
      }
    }
    break;
  }
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND: {
    if (Op0.getOpcode() == SPUISD::VEC2PREFSLOT && NodeVT == Op0VT) {
      // (any_extend (SPUextract_elt0 <arg>)) ->
      // (SPUextract_elt0 <arg>)
      // Types must match, however...
#if !defined(NDEBUG)
      if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
        errs() << "\nReplace: ";
        N->dump(&DAG);
        errs() << "\nWith:    ";
        Op0.getNode()->dump(&DAG);
        errs() << "\n";
      }
#endif

      return Op0;
    }
    break;
  }
  case SPUISD::IndirectAddr: {
    if (!ST->usingLargeMem() && Op0.getOpcode() == SPUISD::AFormAddr) {
      ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1));
      if (CN != 0 && CN->isNullValue()) {
        // (SPUindirect (SPUaform <addr>, 0), 0) ->
        // (SPUaform <addr>, 0)

        DEBUG(errs() << "Replace: ");
        DEBUG(N->dump(&DAG));
        DEBUG(errs() << "\nWith:    ");
        DEBUG(Op0.getNode()->dump(&DAG));
        DEBUG(errs() << "\n");

        return Op0;
      }
    } else if (Op0.getOpcode() == ISD::ADD) {
      SDValue Op1 = N->getOperand(1);
      if (ConstantSDNode *CN1 = dyn_cast<ConstantSDNode>(Op1)) {
        // (SPUindirect (add <arg>, <arg>), 0) ->
        // (SPUindirect <arg>, <arg>)
        if (CN1->isNullValue()) {

#if !defined(NDEBUG)
          if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
            errs() << "\n"
                 << "Replace: (SPUindirect (add <arg>, <arg>), 0)\n"
                 << "With:    (SPUindirect <arg>, <arg>)\n";
          }
#endif

          return DAG.getNode(SPUISD::IndirectAddr, dl, Op0VT,
                             Op0.getOperand(0), Op0.getOperand(1));
        }
      }
    }
    break;
  }
  case SPUISD::SHL_BITS:
  case SPUISD::SHL_BYTES:
  case SPUISD::ROTBYTES_LEFT: {
    SDValue Op1 = N->getOperand(1);

    // Kill degenerate vector shifts:
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op1)) {
      if (CN->isNullValue()) {
        Result = Op0;
      }
    }
    break;
  }
  case SPUISD::PREFSLOT2VEC: {
    switch (Op0.getOpcode()) {
    default:
      break;
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::SIGN_EXTEND: {
      // (SPUprefslot2vec (any|zero|sign_extend (SPUvec2prefslot <arg>))) ->
      // <arg>
      // but only if the SPUprefslot2vec and <arg> types match.
      SDValue Op00 = Op0.getOperand(0);
      if (Op00.getOpcode() == SPUISD::VEC2PREFSLOT) {
        SDValue Op000 = Op00.getOperand(0);
        if (Op000.getValueType() == NodeVT) {
          Result = Op000;
        }
      }
      break;
    }
    case SPUISD::VEC2PREFSLOT: {
      // (SPUprefslot2vec (SPUvec2prefslot <arg>)) ->
      // <arg>
      Result = Op0.getOperand(0);
      break;
    }
    }
    break;
  }
  }

  // Otherwise, return unchanged.
#ifndef NDEBUG
  if (Result.getNode()) {
    DEBUG(errs() << "\nReplace.SPU: ");
    DEBUG(N->dump(&DAG));
    DEBUG(errs() << "\nWith:        ");
    DEBUG(Result.getNode()->dump(&DAG));
    DEBUG(errs() << "\n");
  }
#endif

  return Result;
}

//===----------------------------------------------------------------------===//
// Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
SPUTargetLowering::ConstraintType
SPUTargetLowering::getConstraintType(const std::string &ConstraintLetter) const {
  if (ConstraintLetter.size() == 1) {
    switch (ConstraintLetter[0]) {
    default: break;
    case 'b':
    case 'r':
    case 'f':
    case 'v':
    case 'y':
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(ConstraintLetter);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
SPUTargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
    // If we don't have a value, we can't do a match,
    // but allow it at the lowest weight.
  if (CallOperandVal == NULL)
    return CW_Default;
  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;
    //FIXME: Seems like the supported constraint letters were just copied
    // from PPC, as the following doesn't correspond to the GCC docs.
    // I'm leaving it so until someone adds the corresponding lowering support.
  case 'b':
  case 'r':
  case 'f':
  case 'd':
  case 'v':
  case 'y':
    weight = CW_Register;
    break;
  }
  return weight;
}

std::pair<unsigned, const TargetRegisterClass*>
SPUTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                EVT VT) const
{
  if (Constraint.size() == 1) {
    // GCC RS6000 Constraint Letters
    switch (Constraint[0]) {
    case 'b':   // R1-R31
    case 'r':   // R0-R31
      if (VT == MVT::i64)
        return std::make_pair(0U, &SPU::R64CRegClass);
      return std::make_pair(0U, &SPU::R32CRegClass);
    case 'f':
      if (VT == MVT::f32)
        return std::make_pair(0U, &SPU::R32FPRegClass);
      if (VT == MVT::f64)
        return std::make_pair(0U, &SPU::R64FPRegClass);
      break;
    case 'v':
      return std::make_pair(0U, &SPU::GPRCRegClass);
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

//! Compute used/known bits for a SPU operand
void
SPUTargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                  APInt &KnownZero,
                                                  APInt &KnownOne,
                                                  const SelectionDAG &DAG,
                                                  unsigned Depth ) const {
#if 0
  const uint64_t uint64_sizebits = sizeof(uint64_t) * CHAR_BIT;

  switch (Op.getOpcode()) {
  default:
    // KnownZero = KnownOne = APInt(Mask.getBitWidth(), 0);
    break;
  case CALL:
  case SHUFB:
  case SHUFFLE_MASK:
  case CNTB:
  case SPUISD::PREFSLOT2VEC:
  case SPUISD::LDRESULT:
  case SPUISD::VEC2PREFSLOT:
  case SPUISD::SHLQUAD_L_BITS:
  case SPUISD::SHLQUAD_L_BYTES:
  case SPUISD::VEC_ROTL:
  case SPUISD::VEC_ROTR:
  case SPUISD::ROTBYTES_LEFT:
  case SPUISD::SELECT_MASK:
  case SPUISD::SELB:
  }
#endif
}

unsigned
SPUTargetLowering::ComputeNumSignBitsForTargetNode(SDValue Op,
                                                   unsigned Depth) const {
  switch (Op.getOpcode()) {
  default:
    return 1;

  case ISD::SETCC: {
    EVT VT = Op.getValueType();

    if (VT != MVT::i8 && VT != MVT::i16 && VT != MVT::i32) {
      VT = MVT::i32;
    }
    return VT.getSizeInBits();
  }
  }
}

// LowerAsmOperandForConstraint
void
SPUTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                std::string &Constraint,
                                                std::vector<SDValue> &Ops,
                                                SelectionDAG &DAG) const {
  // Default, for the time being, to the base class handler
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

/// isLegalAddressImmediate - Return true if the integer value can be used
/// as the offset of the target addressing mode.
bool SPUTargetLowering::isLegalAddressImmediate(int64_t V,
                                                Type *Ty) const {
  // SPU's addresses are 256K:
  return (V > -(1 << 18) && V < (1 << 18) - 1);
}

bool SPUTargetLowering::isLegalAddressImmediate(GlobalValue* GV) const {
  return false;
}

bool
SPUTargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The SPU target isn't yet aware of offsets.
  return false;
}

// can we compare to Imm without writing it into a register?
bool SPUTargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  //ceqi, cgti, etc. all take s10 operand
  return isInt<10>(Imm);
}

bool
SPUTargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                         Type * ) const{

  // A-form: 18bit absolute address.
  if (AM.BaseGV && !AM.HasBaseReg && AM.Scale == 0 && AM.BaseOffs == 0)
    return true;

  // D-form: reg + 14bit offset
  if (AM.BaseGV ==0 && AM.HasBaseReg && AM.Scale == 0 && isInt<14>(AM.BaseOffs))
    return true;

  // X-form: reg+reg
  if (AM.BaseGV == 0 && AM.HasBaseReg && AM.Scale == 1 && AM.BaseOffs ==0)
    return true;

  return false;
}
