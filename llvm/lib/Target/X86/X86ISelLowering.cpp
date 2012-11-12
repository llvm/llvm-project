//===-- X86ISelLowering.cpp - X86 DAG Lowering Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that X86 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-isel"
#include "X86ISelLowering.h"
#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86TargetMachine.h"
#include "X86TargetObjectFile.h"
#include "Utils/X86ShuffleDecode.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalAlias.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/VariadicFunction.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetOptions.h"
#include <bitset>
#include <cctype>
using namespace llvm;

STATISTIC(NumTailCalls, "Number of tail calls");

// Forward declarations.
static SDValue getMOVL(SelectionDAG &DAG, DebugLoc dl, EVT VT, SDValue V1,
                       SDValue V2);

/// Generate a DAG to grab 128-bits from a vector > 128 bits.  This
/// sets things up to match to an AVX VEXTRACTF128 instruction or a
/// simple subregister reference.  Idx is an index in the 128 bits we
/// want.  It need not be aligned to a 128-bit bounday.  That makes
/// lowering EXTRACT_VECTOR_ELT operations easier.
static SDValue Extract128BitVector(SDValue Vec, unsigned IdxVal,
                                   SelectionDAG &DAG, DebugLoc dl) {
  EVT VT = Vec.getValueType();
  assert(VT.is256BitVector() && "Unexpected vector size!");
  EVT ElVT = VT.getVectorElementType();
  unsigned Factor = VT.getSizeInBits()/128;
  EVT ResultVT = EVT::getVectorVT(*DAG.getContext(), ElVT,
                                  VT.getVectorNumElements()/Factor);

  // Extract from UNDEF is UNDEF.
  if (Vec.getOpcode() == ISD::UNDEF)
    return DAG.getUNDEF(ResultVT);

  // Extract the relevant 128 bits.  Generate an EXTRACT_SUBVECTOR
  // we can match to VEXTRACTF128.
  unsigned ElemsPerChunk = 128 / ElVT.getSizeInBits();

  // This is the index of the first element of the 128-bit chunk
  // we want.
  unsigned NormalizedIdxVal = (((IdxVal * ElVT.getSizeInBits()) / 128)
                               * ElemsPerChunk);

  SDValue VecIdx = DAG.getIntPtrConstant(NormalizedIdxVal);
  SDValue Result = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, ResultVT, Vec,
                               VecIdx);

  return Result;
}

/// Generate a DAG to put 128-bits into a vector > 128 bits.  This
/// sets things up to match to an AVX VINSERTF128 instruction or a
/// simple superregister reference.  Idx is an index in the 128 bits
/// we want.  It need not be aligned to a 128-bit bounday.  That makes
/// lowering INSERT_VECTOR_ELT operations easier.
static SDValue Insert128BitVector(SDValue Result, SDValue Vec,
                                  unsigned IdxVal, SelectionDAG &DAG,
                                  DebugLoc dl) {
  // Inserting UNDEF is Result
  if (Vec.getOpcode() == ISD::UNDEF)
    return Result;

  EVT VT = Vec.getValueType();
  assert(VT.is128BitVector() && "Unexpected vector size!");

  EVT ElVT = VT.getVectorElementType();
  EVT ResultVT = Result.getValueType();

  // Insert the relevant 128 bits.
  unsigned ElemsPerChunk = 128/ElVT.getSizeInBits();

  // This is the index of the first element of the 128-bit chunk
  // we want.
  unsigned NormalizedIdxVal = (((IdxVal * ElVT.getSizeInBits())/128)
                               * ElemsPerChunk);

  SDValue VecIdx = DAG.getIntPtrConstant(NormalizedIdxVal);
  return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, ResultVT, Result, Vec,
                     VecIdx);
}

/// Concat two 128-bit vectors into a 256 bit vector using VINSERTF128
/// instructions. This is used because creating CONCAT_VECTOR nodes of
/// BUILD_VECTORS returns a larger BUILD_VECTOR while we're trying to lower
/// large BUILD_VECTORS.
static SDValue Concat128BitVectors(SDValue V1, SDValue V2, EVT VT,
                                   unsigned NumElems, SelectionDAG &DAG,
                                   DebugLoc dl) {
  SDValue V = Insert128BitVector(DAG.getUNDEF(VT), V1, 0, DAG, dl);
  return Insert128BitVector(V, V2, NumElems/2, DAG, dl);
}

static TargetLoweringObjectFile *createTLOF(X86TargetMachine &TM) {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  bool is64Bit = Subtarget->is64Bit();

  if (Subtarget->isTargetEnvMacho()) {
    if (is64Bit)
      return new X86_64MachoTargetObjectFile();
    return new TargetLoweringObjectFileMachO();
  }

  if (Subtarget->isTargetLinux())
    return new X86LinuxTargetObjectFile();
  if (Subtarget->isTargetELF())
    return new TargetLoweringObjectFileELF();
  if (Subtarget->isTargetCOFF() && !Subtarget->isTargetEnvMacho())
    return new TargetLoweringObjectFileCOFF();
  llvm_unreachable("unknown subtarget type");
}

X86TargetLowering::X86TargetLowering(X86TargetMachine &TM)
  : TargetLowering(TM, createTLOF(TM)) {
  Subtarget = &TM.getSubtarget<X86Subtarget>();
  X86ScalarSSEf64 = Subtarget->hasSSE2();
  X86ScalarSSEf32 = Subtarget->hasSSE1();

  RegInfo = TM.getRegisterInfo();
  TD = getDataLayout();

  // Set up the TargetLowering object.
  static const MVT IntVTs[] = { MVT::i8, MVT::i16, MVT::i32, MVT::i64 };

  // X86 is weird, it always uses i8 for shift amounts and setcc results.
  setBooleanContents(ZeroOrOneBooleanContent);
  // X86-SSE is even stranger. It uses -1 or 0 for vector masks.
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);

  // For 64-bit since we have so many registers use the ILP scheduler, for
  // 32-bit code use the register pressure specific scheduling.
  // For Atom, always use ILP scheduling.
  if (Subtarget->isAtom())
    setSchedulingPreference(Sched::ILP);
  else if (Subtarget->is64Bit())
    setSchedulingPreference(Sched::ILP);
  else
    setSchedulingPreference(Sched::RegPressure);
  setStackPointerRegisterToSaveRestore(RegInfo->getStackRegister());

  // Bypass i32 with i8 on Atom when compiling with O2
  if (Subtarget->hasSlowDivide() && TM.getOptLevel() >= CodeGenOpt::Default)
    addBypassSlowDiv(32, 8);

  if (Subtarget->isTargetWindows() && !Subtarget->isTargetCygMing()) {
    // Setup Windows compiler runtime calls.
    setLibcallName(RTLIB::SDIV_I64, "_alldiv");
    setLibcallName(RTLIB::UDIV_I64, "_aulldiv");
    setLibcallName(RTLIB::SREM_I64, "_allrem");
    setLibcallName(RTLIB::UREM_I64, "_aullrem");
    setLibcallName(RTLIB::MUL_I64, "_allmul");
    setLibcallCallingConv(RTLIB::SDIV_I64, CallingConv::X86_StdCall);
    setLibcallCallingConv(RTLIB::UDIV_I64, CallingConv::X86_StdCall);
    setLibcallCallingConv(RTLIB::SREM_I64, CallingConv::X86_StdCall);
    setLibcallCallingConv(RTLIB::UREM_I64, CallingConv::X86_StdCall);
    setLibcallCallingConv(RTLIB::MUL_I64, CallingConv::X86_StdCall);

    // The _ftol2 runtime function has an unusual calling conv, which
    // is modeled by a special pseudo-instruction.
    setLibcallName(RTLIB::FPTOUINT_F64_I64, 0);
    setLibcallName(RTLIB::FPTOUINT_F32_I64, 0);
    setLibcallName(RTLIB::FPTOUINT_F64_I32, 0);
    setLibcallName(RTLIB::FPTOUINT_F32_I32, 0);
  }

  if (Subtarget->isTargetDarwin()) {
    // Darwin should use _setjmp/_longjmp instead of setjmp/longjmp.
    setUseUnderscoreSetJmp(false);
    setUseUnderscoreLongJmp(false);
  } else if (Subtarget->isTargetMingw()) {
    // MS runtime is weird: it exports _setjmp, but longjmp!
    setUseUnderscoreSetJmp(true);
    setUseUnderscoreLongJmp(false);
  } else {
    setUseUnderscoreSetJmp(true);
    setUseUnderscoreLongJmp(true);
  }

  // Set up the register classes.
  addRegisterClass(MVT::i8, &X86::GR8RegClass);
  addRegisterClass(MVT::i16, &X86::GR16RegClass);
  addRegisterClass(MVT::i32, &X86::GR32RegClass);
  if (Subtarget->is64Bit())
    addRegisterClass(MVT::i64, &X86::GR64RegClass);

  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);

  // We don't accept any truncstore of integer registers.
  setTruncStoreAction(MVT::i64, MVT::i32, Expand);
  setTruncStoreAction(MVT::i64, MVT::i16, Expand);
  setTruncStoreAction(MVT::i64, MVT::i8 , Expand);
  setTruncStoreAction(MVT::i32, MVT::i16, Expand);
  setTruncStoreAction(MVT::i32, MVT::i8 , Expand);
  setTruncStoreAction(MVT::i16, MVT::i8,  Expand);

  // SETOEQ and SETUNE require checking two conditions.
  setCondCodeAction(ISD::SETOEQ, MVT::f32, Expand);
  setCondCodeAction(ISD::SETOEQ, MVT::f64, Expand);
  setCondCodeAction(ISD::SETOEQ, MVT::f80, Expand);
  setCondCodeAction(ISD::SETUNE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUNE, MVT::f64, Expand);
  setCondCodeAction(ISD::SETUNE, MVT::f80, Expand);

  // Promote all UINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have this
  // operation.
  setOperationAction(ISD::UINT_TO_FP       , MVT::i1   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i8   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i16  , Promote);

  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::UINT_TO_FP     , MVT::i32  , Promote);
    setOperationAction(ISD::UINT_TO_FP     , MVT::i64  , Custom);
  } else if (!TM.Options.UseSoftFloat) {
    // We have an algorithm for SSE2->double, and we turn this into a
    // 64-bit FILD followed by conditional FADD for other targets.
    setOperationAction(ISD::UINT_TO_FP     , MVT::i64  , Custom);
    // We have an algorithm for SSE2, and we turn this into a 64-bit
    // FILD for other targets.
    setOperationAction(ISD::UINT_TO_FP     , MVT::i32  , Custom);
  }

  // Promote i1/i8 SINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have
  // this operation.
  setOperationAction(ISD::SINT_TO_FP       , MVT::i1   , Promote);
  setOperationAction(ISD::SINT_TO_FP       , MVT::i8   , Promote);

  if (!TM.Options.UseSoftFloat) {
    // SSE has no i16 to fp conversion, only i32
    if (X86ScalarSSEf32) {
      setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Promote);
      // f32 and f64 cases are Legal, f80 case is not
      setOperationAction(ISD::SINT_TO_FP     , MVT::i32  , Custom);
    } else {
      setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Custom);
      setOperationAction(ISD::SINT_TO_FP     , MVT::i32  , Custom);
    }
  } else {
    setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Promote);
    setOperationAction(ISD::SINT_TO_FP     , MVT::i32  , Promote);
  }

  // In 32-bit mode these are custom lowered.  In 64-bit mode F32 and F64
  // are Legal, f80 is custom lowered.
  setOperationAction(ISD::FP_TO_SINT     , MVT::i64  , Custom);
  setOperationAction(ISD::SINT_TO_FP     , MVT::i64  , Custom);

  // Promote i1/i8 FP_TO_SINT to larger FP_TO_SINTS's, as X86 doesn't have
  // this operation.
  setOperationAction(ISD::FP_TO_SINT       , MVT::i1   , Promote);
  setOperationAction(ISD::FP_TO_SINT       , MVT::i8   , Promote);

  if (X86ScalarSSEf32) {
    setOperationAction(ISD::FP_TO_SINT     , MVT::i16  , Promote);
    // f32 and f64 cases are Legal, f80 case is not
    setOperationAction(ISD::FP_TO_SINT     , MVT::i32  , Custom);
  } else {
    setOperationAction(ISD::FP_TO_SINT     , MVT::i16  , Custom);
    setOperationAction(ISD::FP_TO_SINT     , MVT::i32  , Custom);
  }

  // Handle FP_TO_UINT by promoting the destination to a larger signed
  // conversion.
  setOperationAction(ISD::FP_TO_UINT       , MVT::i1   , Promote);
  setOperationAction(ISD::FP_TO_UINT       , MVT::i8   , Promote);
  setOperationAction(ISD::FP_TO_UINT       , MVT::i16  , Promote);

  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::FP_TO_UINT     , MVT::i64  , Expand);
    setOperationAction(ISD::FP_TO_UINT     , MVT::i32  , Promote);
  } else if (!TM.Options.UseSoftFloat) {
    // Since AVX is a superset of SSE3, only check for SSE here.
    if (Subtarget->hasSSE1() && !Subtarget->hasSSE3())
      // Expand FP_TO_UINT into a select.
      // FIXME: We would like to use a Custom expander here eventually to do
      // the optimal thing for SSE vs. the default expansion in the legalizer.
      setOperationAction(ISD::FP_TO_UINT   , MVT::i32  , Expand);
    else
      // With SSE3 we can use fisttpll to convert to a signed i64; without
      // SSE, we're stuck with a fistpll.
      setOperationAction(ISD::FP_TO_UINT   , MVT::i32  , Custom);
  }

  if (isTargetFTOL()) {
    // Use the _ftol2 runtime function, which has a pseudo-instruction
    // to handle its weird calling convention.
    setOperationAction(ISD::FP_TO_UINT     , MVT::i64  , Custom);
  }

  // TODO: when we have SSE, these could be more efficient, by using movd/movq.
  if (!X86ScalarSSEf64) {
    setOperationAction(ISD::BITCAST        , MVT::f32  , Expand);
    setOperationAction(ISD::BITCAST        , MVT::i32  , Expand);
    if (Subtarget->is64Bit()) {
      setOperationAction(ISD::BITCAST      , MVT::f64  , Expand);
      // Without SSE, i64->f64 goes through memory.
      setOperationAction(ISD::BITCAST      , MVT::i64  , Expand);
    }
  }

  // Scalar integer divide and remainder are lowered to use operations that
  // produce two results, to match the available instructions. This exposes
  // the two-result form to trivial CSE, which is able to combine x/y and x%y
  // into a single instruction.
  //
  // Scalar integer multiply-high is also lowered to use two-result
  // operations, to match the available instructions. However, plain multiply
  // (low) operations are left as Legal, as there are single-result
  // instructions for this in x86. Using the two-result multiply instructions
  // when both high and low results are needed must be arranged by dagcombine.
  for (unsigned i = 0; i != array_lengthof(IntVTs); ++i) {
    MVT VT = IntVTs[i];
    setOperationAction(ISD::MULHS, VT, Expand);
    setOperationAction(ISD::MULHU, VT, Expand);
    setOperationAction(ISD::SDIV, VT, Expand);
    setOperationAction(ISD::UDIV, VT, Expand);
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::UREM, VT, Expand);

    // Add/Sub overflow ops with MVT::Glues are lowered to EFLAGS dependences.
    setOperationAction(ISD::ADDC, VT, Custom);
    setOperationAction(ISD::ADDE, VT, Custom);
    setOperationAction(ISD::SUBC, VT, Custom);
    setOperationAction(ISD::SUBE, VT, Custom);
  }

  setOperationAction(ISD::BR_JT            , MVT::Other, Expand);
  setOperationAction(ISD::BRCOND           , MVT::Other, Custom);
  setOperationAction(ISD::BR_CC            , MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC        , MVT::Other, Expand);
  if (Subtarget->is64Bit())
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16  , Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8   , Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1   , Expand);
  setOperationAction(ISD::FP_ROUND_INREG   , MVT::f32  , Expand);
  setOperationAction(ISD::FREM             , MVT::f32  , Expand);
  setOperationAction(ISD::FREM             , MVT::f64  , Expand);
  setOperationAction(ISD::FREM             , MVT::f80  , Expand);
  setOperationAction(ISD::FLT_ROUNDS_      , MVT::i32  , Custom);

  // Promote the i8 variants and force them on up to i32 which has a shorter
  // encoding.
  setOperationAction(ISD::CTTZ             , MVT::i8   , Promote);
  AddPromotedToType (ISD::CTTZ             , MVT::i8   , MVT::i32);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF  , MVT::i8   , Promote);
  AddPromotedToType (ISD::CTTZ_ZERO_UNDEF  , MVT::i8   , MVT::i32);
  if (Subtarget->hasBMI()) {
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i16  , Expand);
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32  , Expand);
    if (Subtarget->is64Bit())
      setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i64, Expand);
  } else {
    setOperationAction(ISD::CTTZ           , MVT::i16  , Custom);
    setOperationAction(ISD::CTTZ           , MVT::i32  , Custom);
    if (Subtarget->is64Bit())
      setOperationAction(ISD::CTTZ         , MVT::i64  , Custom);
  }

  if (Subtarget->hasLZCNT()) {
    // When promoting the i8 variants, force them to i32 for a shorter
    // encoding.
    setOperationAction(ISD::CTLZ           , MVT::i8   , Promote);
    AddPromotedToType (ISD::CTLZ           , MVT::i8   , MVT::i32);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i8   , Promote);
    AddPromotedToType (ISD::CTLZ_ZERO_UNDEF, MVT::i8   , MVT::i32);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i16  , Expand);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32  , Expand);
    if (Subtarget->is64Bit())
      setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i64, Expand);
  } else {
    setOperationAction(ISD::CTLZ           , MVT::i8   , Custom);
    setOperationAction(ISD::CTLZ           , MVT::i16  , Custom);
    setOperationAction(ISD::CTLZ           , MVT::i32  , Custom);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i8   , Custom);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i16  , Custom);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32  , Custom);
    if (Subtarget->is64Bit()) {
      setOperationAction(ISD::CTLZ         , MVT::i64  , Custom);
      setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i64, Custom);
    }
  }

  if (Subtarget->hasPOPCNT()) {
    setOperationAction(ISD::CTPOP          , MVT::i8   , Promote);
  } else {
    setOperationAction(ISD::CTPOP          , MVT::i8   , Expand);
    setOperationAction(ISD::CTPOP          , MVT::i16  , Expand);
    setOperationAction(ISD::CTPOP          , MVT::i32  , Expand);
    if (Subtarget->is64Bit())
      setOperationAction(ISD::CTPOP        , MVT::i64  , Expand);
  }

  setOperationAction(ISD::READCYCLECOUNTER , MVT::i64  , Custom);
  setOperationAction(ISD::BSWAP            , MVT::i16  , Expand);

  // These should be promoted to a larger select which is supported.
  setOperationAction(ISD::SELECT          , MVT::i1   , Promote);
  // X86 wants to expand cmov itself.
  setOperationAction(ISD::SELECT          , MVT::i8   , Custom);
  setOperationAction(ISD::SELECT          , MVT::i16  , Custom);
  setOperationAction(ISD::SELECT          , MVT::i32  , Custom);
  setOperationAction(ISD::SELECT          , MVT::f32  , Custom);
  setOperationAction(ISD::SELECT          , MVT::f64  , Custom);
  setOperationAction(ISD::SELECT          , MVT::f80  , Custom);
  setOperationAction(ISD::SETCC           , MVT::i8   , Custom);
  setOperationAction(ISD::SETCC           , MVT::i16  , Custom);
  setOperationAction(ISD::SETCC           , MVT::i32  , Custom);
  setOperationAction(ISD::SETCC           , MVT::f32  , Custom);
  setOperationAction(ISD::SETCC           , MVT::f64  , Custom);
  setOperationAction(ISD::SETCC           , MVT::f80  , Custom);
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::SELECT        , MVT::i64  , Custom);
    setOperationAction(ISD::SETCC         , MVT::i64  , Custom);
  }
  setOperationAction(ISD::EH_RETURN       , MVT::Other, Custom);
  // NOTE: EH_SJLJ_SETJMP/_LONGJMP supported here is NOT intened to support
  // SjLj exception handling but a light-weight setjmp/longjmp replacement to
  // support continuation, user-level threading, and etc.. As a result, no
  // other SjLj exception interfaces are implemented and please don't build
  // your own exception handling based on them.
  // LLVM/Clang supports zero-cost DWARF exception handling.
  setOperationAction(ISD::EH_SJLJ_SETJMP, MVT::i32, Custom);
  setOperationAction(ISD::EH_SJLJ_LONGJMP, MVT::Other, Custom);

  // Darwin ABI issue.
  setOperationAction(ISD::ConstantPool    , MVT::i32  , Custom);
  setOperationAction(ISD::JumpTable       , MVT::i32  , Custom);
  setOperationAction(ISD::GlobalAddress   , MVT::i32  , Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32  , Custom);
  if (Subtarget->is64Bit())
    setOperationAction(ISD::GlobalTLSAddress, MVT::i64, Custom);
  setOperationAction(ISD::ExternalSymbol  , MVT::i32  , Custom);
  setOperationAction(ISD::BlockAddress    , MVT::i32  , Custom);
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::ConstantPool  , MVT::i64  , Custom);
    setOperationAction(ISD::JumpTable     , MVT::i64  , Custom);
    setOperationAction(ISD::GlobalAddress , MVT::i64  , Custom);
    setOperationAction(ISD::ExternalSymbol, MVT::i64  , Custom);
    setOperationAction(ISD::BlockAddress  , MVT::i64  , Custom);
  }
  // 64-bit addm sub, shl, sra, srl (iff 32-bit x86)
  setOperationAction(ISD::SHL_PARTS       , MVT::i32  , Custom);
  setOperationAction(ISD::SRA_PARTS       , MVT::i32  , Custom);
  setOperationAction(ISD::SRL_PARTS       , MVT::i32  , Custom);
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::SHL_PARTS     , MVT::i64  , Custom);
    setOperationAction(ISD::SRA_PARTS     , MVT::i64  , Custom);
    setOperationAction(ISD::SRL_PARTS     , MVT::i64  , Custom);
  }

  if (Subtarget->hasSSE1())
    setOperationAction(ISD::PREFETCH      , MVT::Other, Legal);

  setOperationAction(ISD::MEMBARRIER    , MVT::Other, Custom);
  setOperationAction(ISD::ATOMIC_FENCE  , MVT::Other, Custom);

  // On X86 and X86-64, atomic operations are lowered to locked instructions.
  // Locked instructions, in turn, have implicit fence semantics (all memory
  // operations are flushed before issuing the locked instruction, and they
  // are not buffered), so we can fold away the common pattern of
  // fence-atomic-fence.
  setShouldFoldAtomicFences(true);

  // Expand certain atomics
  for (unsigned i = 0; i != array_lengthof(IntVTs); ++i) {
    MVT VT = IntVTs[i];
    setOperationAction(ISD::ATOMIC_CMP_SWAP, VT, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_SUB, VT, Custom);
    setOperationAction(ISD::ATOMIC_STORE, VT, Custom);
  }

  if (!Subtarget->is64Bit()) {
    setOperationAction(ISD::ATOMIC_LOAD, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_ADD, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_AND, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_OR, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_XOR, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_NAND, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_SWAP, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_MAX, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_MIN, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_UMAX, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_UMIN, MVT::i64, Custom);
  }

  if (Subtarget->hasCmpxchg16b()) {
    setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i128, Custom);
  }

  // FIXME - use subtarget debug flags
  if (!Subtarget->isTargetDarwin() &&
      !Subtarget->isTargetELF() &&
      !Subtarget->isTargetCygMing()) {
    setOperationAction(ISD::EH_LABEL, MVT::Other, Expand);
  }

  setOperationAction(ISD::EXCEPTIONADDR, MVT::i64, Expand);
  setOperationAction(ISD::EHSELECTION,   MVT::i64, Expand);
  setOperationAction(ISD::EXCEPTIONADDR, MVT::i32, Expand);
  setOperationAction(ISD::EHSELECTION,   MVT::i32, Expand);
  if (Subtarget->is64Bit()) {
    setExceptionPointerRegister(X86::RAX);
    setExceptionSelectorRegister(X86::RDX);
  } else {
    setExceptionPointerRegister(X86::EAX);
    setExceptionSelectorRegister(X86::EDX);
  }
  setOperationAction(ISD::FRAME_TO_ARGS_OFFSET, MVT::i32, Custom);
  setOperationAction(ISD::FRAME_TO_ARGS_OFFSET, MVT::i64, Custom);

  setOperationAction(ISD::INIT_TRAMPOLINE, MVT::Other, Custom);
  setOperationAction(ISD::ADJUST_TRAMPOLINE, MVT::Other, Custom);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::VAARG           , MVT::Other, Custom);
    setOperationAction(ISD::VACOPY          , MVT::Other, Custom);
  } else {
    setOperationAction(ISD::VAARG           , MVT::Other, Expand);
    setOperationAction(ISD::VACOPY          , MVT::Other, Expand);
  }

  setOperationAction(ISD::STACKSAVE,          MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE,       MVT::Other, Expand);

  if (Subtarget->isTargetCOFF() && !Subtarget->isTargetEnvMacho())
    setOperationAction(ISD::DYNAMIC_STACKALLOC, Subtarget->is64Bit() ?
                       MVT::i64 : MVT::i32, Custom);
  else if (TM.Options.EnableSegmentedStacks)
    setOperationAction(ISD::DYNAMIC_STACKALLOC, Subtarget->is64Bit() ?
                       MVT::i64 : MVT::i32, Custom);
  else
    setOperationAction(ISD::DYNAMIC_STACKALLOC, Subtarget->is64Bit() ?
                       MVT::i64 : MVT::i32, Expand);

  if (!TM.Options.UseSoftFloat && X86ScalarSSEf64) {
    // f32 and f64 use SSE.
    // Set up the FP register classes.
    addRegisterClass(MVT::f32, &X86::FR32RegClass);
    addRegisterClass(MVT::f64, &X86::FR64RegClass);

    // Use ANDPD to simulate FABS.
    setOperationAction(ISD::FABS , MVT::f64, Custom);
    setOperationAction(ISD::FABS , MVT::f32, Custom);

    // Use XORP to simulate FNEG.
    setOperationAction(ISD::FNEG , MVT::f64, Custom);
    setOperationAction(ISD::FNEG , MVT::f32, Custom);

    // Use ANDPD and ORPD to simulate FCOPYSIGN.
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);

    // Lower this to FGETSIGNx86 plus an AND.
    setOperationAction(ISD::FGETSIGN, MVT::i64, Custom);
    setOperationAction(ISD::FGETSIGN, MVT::i32, Custom);

    // We don't support sin/cos/fmod
    setOperationAction(ISD::FSIN , MVT::f64, Expand);
    setOperationAction(ISD::FCOS , MVT::f64, Expand);
    setOperationAction(ISD::FSIN , MVT::f32, Expand);
    setOperationAction(ISD::FCOS , MVT::f32, Expand);

    // Expand FP immediates into loads from the stack, except for the special
    // cases we handle.
    addLegalFPImmediate(APFloat(+0.0)); // xorpd
    addLegalFPImmediate(APFloat(+0.0f)); // xorps
  } else if (!TM.Options.UseSoftFloat && X86ScalarSSEf32) {
    // Use SSE for f32, x87 for f64.
    // Set up the FP register classes.
    addRegisterClass(MVT::f32, &X86::FR32RegClass);
    addRegisterClass(MVT::f64, &X86::RFP64RegClass);

    // Use ANDPS to simulate FABS.
    setOperationAction(ISD::FABS , MVT::f32, Custom);

    // Use XORP to simulate FNEG.
    setOperationAction(ISD::FNEG , MVT::f32, Custom);

    setOperationAction(ISD::UNDEF,     MVT::f64, Expand);

    // Use ANDPS and ORPS to simulate FCOPYSIGN.
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);

    // We don't support sin/cos/fmod
    setOperationAction(ISD::FSIN , MVT::f32, Expand);
    setOperationAction(ISD::FCOS , MVT::f32, Expand);

    // Special cases we handle for FP constants.
    addLegalFPImmediate(APFloat(+0.0f)); // xorps
    addLegalFPImmediate(APFloat(+0.0)); // FLD0
    addLegalFPImmediate(APFloat(+1.0)); // FLD1
    addLegalFPImmediate(APFloat(-0.0)); // FLD0/FCHS
    addLegalFPImmediate(APFloat(-1.0)); // FLD1/FCHS

    if (!TM.Options.UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f64  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f64  , Expand);
    }
  } else if (!TM.Options.UseSoftFloat) {
    // f32 and f64 in x87.
    // Set up the FP register classes.
    addRegisterClass(MVT::f64, &X86::RFP64RegClass);
    addRegisterClass(MVT::f32, &X86::RFP32RegClass);

    setOperationAction(ISD::UNDEF,     MVT::f64, Expand);
    setOperationAction(ISD::UNDEF,     MVT::f32, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

    if (!TM.Options.UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f32  , Expand);
      setOperationAction(ISD::FSIN           , MVT::f64  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f32  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f64  , Expand);
    }
    addLegalFPImmediate(APFloat(+0.0)); // FLD0
    addLegalFPImmediate(APFloat(+1.0)); // FLD1
    addLegalFPImmediate(APFloat(-0.0)); // FLD0/FCHS
    addLegalFPImmediate(APFloat(-1.0)); // FLD1/FCHS
    addLegalFPImmediate(APFloat(+0.0f)); // FLD0
    addLegalFPImmediate(APFloat(+1.0f)); // FLD1
    addLegalFPImmediate(APFloat(-0.0f)); // FLD0/FCHS
    addLegalFPImmediate(APFloat(-1.0f)); // FLD1/FCHS
  }

  // We don't support FMA.
  setOperationAction(ISD::FMA, MVT::f64, Expand);
  setOperationAction(ISD::FMA, MVT::f32, Expand);

  // Long double always uses X87.
  if (!TM.Options.UseSoftFloat) {
    addRegisterClass(MVT::f80, &X86::RFP80RegClass);
    setOperationAction(ISD::UNDEF,     MVT::f80, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f80, Expand);
    {
      APFloat TmpFlt = APFloat::getZero(APFloat::x87DoubleExtended);
      addLegalFPImmediate(TmpFlt);  // FLD0
      TmpFlt.changeSign();
      addLegalFPImmediate(TmpFlt);  // FLD0/FCHS

      bool ignored;
      APFloat TmpFlt2(+1.0);
      TmpFlt2.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
                      &ignored);
      addLegalFPImmediate(TmpFlt2);  // FLD1
      TmpFlt2.changeSign();
      addLegalFPImmediate(TmpFlt2);  // FLD1/FCHS
    }

    if (!TM.Options.UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f80  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f80  , Expand);
    }

    setOperationAction(ISD::FFLOOR, MVT::f80, Expand);
    setOperationAction(ISD::FCEIL,  MVT::f80, Expand);
    setOperationAction(ISD::FTRUNC, MVT::f80, Expand);
    setOperationAction(ISD::FRINT,  MVT::f80, Expand);
    setOperationAction(ISD::FNEARBYINT, MVT::f80, Expand);
    setOperationAction(ISD::FMA, MVT::f80, Expand);
  }

  // Always use a library call for pow.
  setOperationAction(ISD::FPOW             , MVT::f32  , Expand);
  setOperationAction(ISD::FPOW             , MVT::f64  , Expand);
  setOperationAction(ISD::FPOW             , MVT::f80  , Expand);

  setOperationAction(ISD::FLOG, MVT::f80, Expand);
  setOperationAction(ISD::FLOG2, MVT::f80, Expand);
  setOperationAction(ISD::FLOG10, MVT::f80, Expand);
  setOperationAction(ISD::FEXP, MVT::f80, Expand);
  setOperationAction(ISD::FEXP2, MVT::f80, Expand);

  // First set operation action for all vector types to either promote
  // (for widening) or expand (for scalarization). Then we will selectively
  // turn on ones that can be effectively codegen'd.
  for (int VT = MVT::FIRST_VECTOR_VALUETYPE;
           VT <= MVT::LAST_VECTOR_VALUETYPE; ++VT) {
    setOperationAction(ISD::ADD , (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SUB , (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FADD, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FNEG, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FSUB, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::MUL , (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FMUL, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SDIV, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::UDIV, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FDIV, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SREM, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::UREM, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::LOAD, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::VECTOR_SHUFFLE, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT,(MVT::SimpleValueType)VT,Expand);
    setOperationAction(ISD::INSERT_VECTOR_ELT,(MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::EXTRACT_SUBVECTOR,(MVT::SimpleValueType)VT,Expand);
    setOperationAction(ISD::INSERT_SUBVECTOR,(MVT::SimpleValueType)VT,Expand);
    setOperationAction(ISD::FABS, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FSIN, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FCOS, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FREM, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FMA,  (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FPOWI, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FSQRT, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FCOPYSIGN, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FFLOOR, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SDIVREM, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::UDIVREM, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FPOW, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CTPOP, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CTTZ, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CTLZ, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SHL, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SRA, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SRL, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::ROTL, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::ROTR, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::BSWAP, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SETCC, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FLOG, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FLOG2, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FLOG10, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FEXP, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FEXP2, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FP_TO_UINT, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FP_TO_SINT, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::UINT_TO_FP, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SINT_TO_FP, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, (MVT::SimpleValueType)VT,Expand);
    setOperationAction(ISD::TRUNCATE,  (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SIGN_EXTEND,  (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::ZERO_EXTEND,  (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::ANY_EXTEND,  (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::VSELECT,  (MVT::SimpleValueType)VT, Expand);
    for (int InnerVT = MVT::FIRST_VECTOR_VALUETYPE;
             InnerVT <= MVT::LAST_VECTOR_VALUETYPE; ++InnerVT)
      setTruncStoreAction((MVT::SimpleValueType)VT,
                          (MVT::SimpleValueType)InnerVT, Expand);
    setLoadExtAction(ISD::SEXTLOAD, (MVT::SimpleValueType)VT, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, (MVT::SimpleValueType)VT, Expand);
    setLoadExtAction(ISD::EXTLOAD, (MVT::SimpleValueType)VT, Expand);
  }

  // FIXME: In order to prevent SSE instructions being expanded to MMX ones
  // with -msoft-float, disable use of MMX as well.
  if (!TM.Options.UseSoftFloat && Subtarget->hasMMX()) {
    addRegisterClass(MVT::x86mmx, &X86::VR64RegClass);
    // No operations on x86mmx supported, everything uses intrinsics.
  }

  // MMX-sized vectors (other than x86mmx) are expected to be expanded
  // into smaller operations.
  setOperationAction(ISD::MULHS,              MVT::v8i8,  Expand);
  setOperationAction(ISD::MULHS,              MVT::v4i16, Expand);
  setOperationAction(ISD::MULHS,              MVT::v2i32, Expand);
  setOperationAction(ISD::MULHS,              MVT::v1i64, Expand);
  setOperationAction(ISD::AND,                MVT::v8i8,  Expand);
  setOperationAction(ISD::AND,                MVT::v4i16, Expand);
  setOperationAction(ISD::AND,                MVT::v2i32, Expand);
  setOperationAction(ISD::AND,                MVT::v1i64, Expand);
  setOperationAction(ISD::OR,                 MVT::v8i8,  Expand);
  setOperationAction(ISD::OR,                 MVT::v4i16, Expand);
  setOperationAction(ISD::OR,                 MVT::v2i32, Expand);
  setOperationAction(ISD::OR,                 MVT::v1i64, Expand);
  setOperationAction(ISD::XOR,                MVT::v8i8,  Expand);
  setOperationAction(ISD::XOR,                MVT::v4i16, Expand);
  setOperationAction(ISD::XOR,                MVT::v2i32, Expand);
  setOperationAction(ISD::XOR,                MVT::v1i64, Expand);
  setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v8i8,  Expand);
  setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v4i16, Expand);
  setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v2i32, Expand);
  setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v1i64, Expand);
  setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v1i64, Expand);
  setOperationAction(ISD::SELECT,             MVT::v8i8,  Expand);
  setOperationAction(ISD::SELECT,             MVT::v4i16, Expand);
  setOperationAction(ISD::SELECT,             MVT::v2i32, Expand);
  setOperationAction(ISD::SELECT,             MVT::v1i64, Expand);
  setOperationAction(ISD::BITCAST,            MVT::v8i8,  Expand);
  setOperationAction(ISD::BITCAST,            MVT::v4i16, Expand);
  setOperationAction(ISD::BITCAST,            MVT::v2i32, Expand);
  setOperationAction(ISD::BITCAST,            MVT::v1i64, Expand);

  if (!TM.Options.UseSoftFloat && Subtarget->hasSSE1()) {
    addRegisterClass(MVT::v4f32, &X86::VR128RegClass);

    setOperationAction(ISD::FADD,               MVT::v4f32, Legal);
    setOperationAction(ISD::FSUB,               MVT::v4f32, Legal);
    setOperationAction(ISD::FMUL,               MVT::v4f32, Legal);
    setOperationAction(ISD::FDIV,               MVT::v4f32, Legal);
    setOperationAction(ISD::FSQRT,              MVT::v4f32, Legal);
    setOperationAction(ISD::FNEG,               MVT::v4f32, Custom);
    setOperationAction(ISD::FABS,               MVT::v4f32, Custom);
    setOperationAction(ISD::LOAD,               MVT::v4f32, Legal);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v4f32, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v4f32, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4f32, Custom);
    setOperationAction(ISD::SELECT,             MVT::v4f32, Custom);
  }

  if (!TM.Options.UseSoftFloat && Subtarget->hasSSE2()) {
    addRegisterClass(MVT::v2f64, &X86::VR128RegClass);

    // FIXME: Unfortunately -soft-float and -no-implicit-float means XMM
    // registers cannot be used even for integer operations.
    addRegisterClass(MVT::v16i8, &X86::VR128RegClass);
    addRegisterClass(MVT::v8i16, &X86::VR128RegClass);
    addRegisterClass(MVT::v4i32, &X86::VR128RegClass);
    addRegisterClass(MVT::v2i64, &X86::VR128RegClass);

    setOperationAction(ISD::ADD,                MVT::v16i8, Legal);
    setOperationAction(ISD::ADD,                MVT::v8i16, Legal);
    setOperationAction(ISD::ADD,                MVT::v4i32, Legal);
    setOperationAction(ISD::ADD,                MVT::v2i64, Legal);
    setOperationAction(ISD::MUL,                MVT::v2i64, Custom);
    setOperationAction(ISD::SUB,                MVT::v16i8, Legal);
    setOperationAction(ISD::SUB,                MVT::v8i16, Legal);
    setOperationAction(ISD::SUB,                MVT::v4i32, Legal);
    setOperationAction(ISD::SUB,                MVT::v2i64, Legal);
    setOperationAction(ISD::MUL,                MVT::v8i16, Legal);
    setOperationAction(ISD::FADD,               MVT::v2f64, Legal);
    setOperationAction(ISD::FSUB,               MVT::v2f64, Legal);
    setOperationAction(ISD::FMUL,               MVT::v2f64, Legal);
    setOperationAction(ISD::FDIV,               MVT::v2f64, Legal);
    setOperationAction(ISD::FSQRT,              MVT::v2f64, Legal);
    setOperationAction(ISD::FNEG,               MVT::v2f64, Custom);
    setOperationAction(ISD::FABS,               MVT::v2f64, Custom);

    setOperationAction(ISD::SETCC,              MVT::v2i64, Custom);
    setOperationAction(ISD::SETCC,              MVT::v16i8, Custom);
    setOperationAction(ISD::SETCC,              MVT::v8i16, Custom);
    setOperationAction(ISD::SETCC,              MVT::v4i32, Custom);

    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v16i8, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4i32, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4f32, Custom);

    // Custom lower build_vector, vector_shuffle, and extract_vector_elt.
    for (int i = MVT::v16i8; i != MVT::v2i64; ++i) {
      MVT VT = (MVT::SimpleValueType)i;
      // Do not attempt to custom lower non-power-of-2 vectors
      if (!isPowerOf2_32(VT.getVectorNumElements()))
        continue;
      // Do not attempt to custom lower non-128-bit vectors
      if (!VT.is128BitVector())
        continue;
      setOperationAction(ISD::BUILD_VECTOR,       VT, Custom);
      setOperationAction(ISD::VECTOR_SHUFFLE,     VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
    }

    setOperationAction(ISD::BUILD_VECTOR,       MVT::v2f64, Custom);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v2i64, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v2f64, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v2i64, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v2f64, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f64, Custom);

    if (Subtarget->is64Bit()) {
      setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v2i64, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i64, Custom);
    }

    // Promote v16i8, v8i16, v4i32 load, select, and, or, xor to v2i64.
    for (int i = MVT::v16i8; i != MVT::v2i64; ++i) {
      MVT VT = (MVT::SimpleValueType)i;

      // Do not attempt to promote non-128-bit vectors
      if (!VT.is128BitVector())
        continue;

      setOperationAction(ISD::AND,    VT, Promote);
      AddPromotedToType (ISD::AND,    VT, MVT::v2i64);
      setOperationAction(ISD::OR,     VT, Promote);
      AddPromotedToType (ISD::OR,     VT, MVT::v2i64);
      setOperationAction(ISD::XOR,    VT, Promote);
      AddPromotedToType (ISD::XOR,    VT, MVT::v2i64);
      setOperationAction(ISD::LOAD,   VT, Promote);
      AddPromotedToType (ISD::LOAD,   VT, MVT::v2i64);
      setOperationAction(ISD::SELECT, VT, Promote);
      AddPromotedToType (ISD::SELECT, VT, MVT::v2i64);
    }

    setTruncStoreAction(MVT::f64, MVT::f32, Expand);

    // Custom lower v2i64 and v2f64 selects.
    setOperationAction(ISD::LOAD,               MVT::v2f64, Legal);
    setOperationAction(ISD::LOAD,               MVT::v2i64, Legal);
    setOperationAction(ISD::SELECT,             MVT::v2f64, Custom);
    setOperationAction(ISD::SELECT,             MVT::v2i64, Custom);

    setOperationAction(ISD::FP_TO_SINT,         MVT::v4i32, Legal);
    setOperationAction(ISD::SINT_TO_FP,         MVT::v4i32, Legal);

    setOperationAction(ISD::UINT_TO_FP,         MVT::v4i8,  Custom);
    setOperationAction(ISD::UINT_TO_FP,         MVT::v4i16, Custom);
    // As there is no 64-bit GPR available, we need build a special custom
    // sequence to convert from v2i32 to v2f32.
    if (!Subtarget->is64Bit())
      setOperationAction(ISD::UINT_TO_FP,       MVT::v2f32, Custom);

    setOperationAction(ISD::FP_EXTEND,          MVT::v2f32, Custom);
    setOperationAction(ISD::FP_ROUND,           MVT::v2f32, Custom);

    setLoadExtAction(ISD::EXTLOAD,              MVT::v2f32, Legal);
  }

  if (Subtarget->hasSSE41()) {
    setOperationAction(ISD::FFLOOR,             MVT::f32,   Legal);
    setOperationAction(ISD::FCEIL,              MVT::f32,   Legal);
    setOperationAction(ISD::FTRUNC,             MVT::f32,   Legal);
    setOperationAction(ISD::FRINT,              MVT::f32,   Legal);
    setOperationAction(ISD::FNEARBYINT,         MVT::f32,   Legal);
    setOperationAction(ISD::FFLOOR,             MVT::f64,   Legal);
    setOperationAction(ISD::FCEIL,              MVT::f64,   Legal);
    setOperationAction(ISD::FTRUNC,             MVT::f64,   Legal);
    setOperationAction(ISD::FRINT,              MVT::f64,   Legal);
    setOperationAction(ISD::FNEARBYINT,         MVT::f64,   Legal);

    setOperationAction(ISD::FFLOOR,             MVT::v4f32, Legal);
    setOperationAction(ISD::FFLOOR,             MVT::v2f64, Legal);

    // FIXME: Do we need to handle scalar-to-vector here?
    setOperationAction(ISD::MUL,                MVT::v4i32, Legal);

    setOperationAction(ISD::VSELECT,            MVT::v2f64, Legal);
    setOperationAction(ISD::VSELECT,            MVT::v2i64, Legal);
    setOperationAction(ISD::VSELECT,            MVT::v16i8, Legal);
    setOperationAction(ISD::VSELECT,            MVT::v4i32, Legal);
    setOperationAction(ISD::VSELECT,            MVT::v4f32, Legal);

    // i8 and i16 vectors are custom , because the source register and source
    // source memory operand types are not the same width.  f32 vectors are
    // custom since the immediate controlling the insert encodes additional
    // information.
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v16i8, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4i32, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4f32, Custom);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v16i8, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v8i16, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i32, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4f32, Custom);

    // FIXME: these should be Legal but thats only for the case where
    // the index is constant.  For now custom expand to deal with that.
    if (Subtarget->is64Bit()) {
      setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v2i64, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i64, Custom);
    }
  }

  if (Subtarget->hasSSE2()) {
    setOperationAction(ISD::SRL,               MVT::v8i16, Custom);
    setOperationAction(ISD::SRL,               MVT::v16i8, Custom);

    setOperationAction(ISD::SHL,               MVT::v8i16, Custom);
    setOperationAction(ISD::SHL,               MVT::v16i8, Custom);

    setOperationAction(ISD::SRA,               MVT::v8i16, Custom);
    setOperationAction(ISD::SRA,               MVT::v16i8, Custom);

    if (Subtarget->hasAVX2()) {
      setOperationAction(ISD::SRL,             MVT::v2i64, Legal);
      setOperationAction(ISD::SRL,             MVT::v4i32, Legal);

      setOperationAction(ISD::SHL,             MVT::v2i64, Legal);
      setOperationAction(ISD::SHL,             MVT::v4i32, Legal);

      setOperationAction(ISD::SRA,             MVT::v4i32, Legal);
    } else {
      setOperationAction(ISD::SRL,             MVT::v2i64, Custom);
      setOperationAction(ISD::SRL,             MVT::v4i32, Custom);

      setOperationAction(ISD::SHL,             MVT::v2i64, Custom);
      setOperationAction(ISD::SHL,             MVT::v4i32, Custom);

      setOperationAction(ISD::SRA,             MVT::v4i32, Custom);
    }
  }

  if (!TM.Options.UseSoftFloat && Subtarget->hasAVX()) {
    addRegisterClass(MVT::v32i8,  &X86::VR256RegClass);
    addRegisterClass(MVT::v16i16, &X86::VR256RegClass);
    addRegisterClass(MVT::v8i32,  &X86::VR256RegClass);
    addRegisterClass(MVT::v8f32,  &X86::VR256RegClass);
    addRegisterClass(MVT::v4i64,  &X86::VR256RegClass);
    addRegisterClass(MVT::v4f64,  &X86::VR256RegClass);

    setOperationAction(ISD::LOAD,               MVT::v8f32, Legal);
    setOperationAction(ISD::LOAD,               MVT::v4f64, Legal);
    setOperationAction(ISD::LOAD,               MVT::v4i64, Legal);

    setOperationAction(ISD::FADD,               MVT::v8f32, Legal);
    setOperationAction(ISD::FSUB,               MVT::v8f32, Legal);
    setOperationAction(ISD::FMUL,               MVT::v8f32, Legal);
    setOperationAction(ISD::FDIV,               MVT::v8f32, Legal);
    setOperationAction(ISD::FSQRT,              MVT::v8f32, Legal);
    setOperationAction(ISD::FFLOOR,             MVT::v8f32, Legal);
    setOperationAction(ISD::FNEG,               MVT::v8f32, Custom);
    setOperationAction(ISD::FABS,               MVT::v8f32, Custom);

    setOperationAction(ISD::FADD,               MVT::v4f64, Legal);
    setOperationAction(ISD::FSUB,               MVT::v4f64, Legal);
    setOperationAction(ISD::FMUL,               MVT::v4f64, Legal);
    setOperationAction(ISD::FDIV,               MVT::v4f64, Legal);
    setOperationAction(ISD::FSQRT,              MVT::v4f64, Legal);
    setOperationAction(ISD::FFLOOR,             MVT::v4f64, Legal);
    setOperationAction(ISD::FNEG,               MVT::v4f64, Custom);
    setOperationAction(ISD::FABS,               MVT::v4f64, Custom);

    setOperationAction(ISD::TRUNCATE,           MVT::v8i16, Custom);

    setOperationAction(ISD::FP_TO_SINT,         MVT::v8i16, Custom);

    setOperationAction(ISD::FP_TO_SINT,         MVT::v8i32, Legal);
    setOperationAction(ISD::SINT_TO_FP,         MVT::v8i32, Legal);
    setOperationAction(ISD::FP_ROUND,           MVT::v4f32, Legal);

    setOperationAction(ISD::ZERO_EXTEND,        MVT::v8i32, Custom);
    setOperationAction(ISD::UINT_TO_FP,         MVT::v8i8,  Custom);
    setOperationAction(ISD::UINT_TO_FP,         MVT::v8i16, Custom);

    setLoadExtAction(ISD::EXTLOAD,              MVT::v4f32, Legal);

    setOperationAction(ISD::SRL,               MVT::v16i16, Custom);
    setOperationAction(ISD::SRL,               MVT::v32i8, Custom);

    setOperationAction(ISD::SHL,               MVT::v16i16, Custom);
    setOperationAction(ISD::SHL,               MVT::v32i8, Custom);

    setOperationAction(ISD::SRA,               MVT::v16i16, Custom);
    setOperationAction(ISD::SRA,               MVT::v32i8, Custom);

    setOperationAction(ISD::SETCC,             MVT::v32i8, Custom);
    setOperationAction(ISD::SETCC,             MVT::v16i16, Custom);
    setOperationAction(ISD::SETCC,             MVT::v8i32, Custom);
    setOperationAction(ISD::SETCC,             MVT::v4i64, Custom);

    setOperationAction(ISD::SELECT,            MVT::v4f64, Custom);
    setOperationAction(ISD::SELECT,            MVT::v4i64, Custom);
    setOperationAction(ISD::SELECT,            MVT::v8f32, Custom);

    setOperationAction(ISD::VSELECT,           MVT::v4f64, Legal);
    setOperationAction(ISD::VSELECT,           MVT::v4i64, Legal);
    setOperationAction(ISD::VSELECT,           MVT::v8i32, Legal);
    setOperationAction(ISD::VSELECT,           MVT::v8f32, Legal);

    if (Subtarget->hasFMA() || Subtarget->hasFMA4()) {
      setOperationAction(ISD::FMA,             MVT::v8f32, Custom);
      setOperationAction(ISD::FMA,             MVT::v4f64, Custom);
      setOperationAction(ISD::FMA,             MVT::v4f32, Custom);
      setOperationAction(ISD::FMA,             MVT::v2f64, Custom);
      setOperationAction(ISD::FMA,             MVT::f32, Custom);
      setOperationAction(ISD::FMA,             MVT::f64, Custom);
    }

    if (Subtarget->hasAVX2()) {
      setOperationAction(ISD::ADD,             MVT::v4i64, Legal);
      setOperationAction(ISD::ADD,             MVT::v8i32, Legal);
      setOperationAction(ISD::ADD,             MVT::v16i16, Legal);
      setOperationAction(ISD::ADD,             MVT::v32i8, Legal);

      setOperationAction(ISD::SUB,             MVT::v4i64, Legal);
      setOperationAction(ISD::SUB,             MVT::v8i32, Legal);
      setOperationAction(ISD::SUB,             MVT::v16i16, Legal);
      setOperationAction(ISD::SUB,             MVT::v32i8, Legal);

      setOperationAction(ISD::MUL,             MVT::v4i64, Custom);
      setOperationAction(ISD::MUL,             MVT::v8i32, Legal);
      setOperationAction(ISD::MUL,             MVT::v16i16, Legal);
      // Don't lower v32i8 because there is no 128-bit byte mul

      setOperationAction(ISD::VSELECT,         MVT::v32i8, Legal);

      setOperationAction(ISD::SRL,             MVT::v4i64, Legal);
      setOperationAction(ISD::SRL,             MVT::v8i32, Legal);

      setOperationAction(ISD::SHL,             MVT::v4i64, Legal);
      setOperationAction(ISD::SHL,             MVT::v8i32, Legal);

      setOperationAction(ISD::SRA,             MVT::v8i32, Legal);
    } else {
      setOperationAction(ISD::ADD,             MVT::v4i64, Custom);
      setOperationAction(ISD::ADD,             MVT::v8i32, Custom);
      setOperationAction(ISD::ADD,             MVT::v16i16, Custom);
      setOperationAction(ISD::ADD,             MVT::v32i8, Custom);

      setOperationAction(ISD::SUB,             MVT::v4i64, Custom);
      setOperationAction(ISD::SUB,             MVT::v8i32, Custom);
      setOperationAction(ISD::SUB,             MVT::v16i16, Custom);
      setOperationAction(ISD::SUB,             MVT::v32i8, Custom);

      setOperationAction(ISD::MUL,             MVT::v4i64, Custom);
      setOperationAction(ISD::MUL,             MVT::v8i32, Custom);
      setOperationAction(ISD::MUL,             MVT::v16i16, Custom);
      // Don't lower v32i8 because there is no 128-bit byte mul

      setOperationAction(ISD::SRL,             MVT::v4i64, Custom);
      setOperationAction(ISD::SRL,             MVT::v8i32, Custom);

      setOperationAction(ISD::SHL,             MVT::v4i64, Custom);
      setOperationAction(ISD::SHL,             MVT::v8i32, Custom);

      setOperationAction(ISD::SRA,             MVT::v8i32, Custom);
    }

    // Custom lower several nodes for 256-bit types.
    for (int i = MVT::FIRST_VECTOR_VALUETYPE;
             i <= MVT::LAST_VECTOR_VALUETYPE; ++i) {
      MVT VT = (MVT::SimpleValueType)i;

      // Extract subvector is special because the value type
      // (result) is 128-bit but the source is 256-bit wide.
      if (VT.is128BitVector())
        setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

      // Do not attempt to custom lower other non-256-bit vectors
      if (!VT.is256BitVector())
        continue;

      setOperationAction(ISD::BUILD_VECTOR,       VT, Custom);
      setOperationAction(ISD::VECTOR_SHUFFLE,     VT, Custom);
      setOperationAction(ISD::INSERT_VECTOR_ELT,  VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
      setOperationAction(ISD::SCALAR_TO_VECTOR,   VT, Custom);
      setOperationAction(ISD::INSERT_SUBVECTOR,   VT, Custom);
      setOperationAction(ISD::CONCAT_VECTORS,     VT, Custom);
    }

    // Promote v32i8, v16i16, v8i32 select, and, or, xor to v4i64.
    for (int i = MVT::v32i8; i != MVT::v4i64; ++i) {
      MVT VT = (MVT::SimpleValueType)i;

      // Do not attempt to promote non-256-bit vectors
      if (!VT.is256BitVector())
        continue;

      setOperationAction(ISD::AND,    VT, Promote);
      AddPromotedToType (ISD::AND,    VT, MVT::v4i64);
      setOperationAction(ISD::OR,     VT, Promote);
      AddPromotedToType (ISD::OR,     VT, MVT::v4i64);
      setOperationAction(ISD::XOR,    VT, Promote);
      AddPromotedToType (ISD::XOR,    VT, MVT::v4i64);
      setOperationAction(ISD::LOAD,   VT, Promote);
      AddPromotedToType (ISD::LOAD,   VT, MVT::v4i64);
      setOperationAction(ISD::SELECT, VT, Promote);
      AddPromotedToType (ISD::SELECT, VT, MVT::v4i64);
    }
  }

  // SIGN_EXTEND_INREGs are evaluated by the extend type. Handle the expansion
  // of this type with custom code.
  for (int VT = MVT::FIRST_VECTOR_VALUETYPE;
           VT != MVT::LAST_VECTOR_VALUETYPE; VT++) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, (MVT::SimpleValueType)VT,
                       Custom);
  }

  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);


  // Only custom-lower 64-bit SADDO and friends on 64-bit because we don't
  // handle type legalization for these operations here.
  //
  // FIXME: We really should do custom legalization for addition and
  // subtraction on x86-32 once PR3203 is fixed.  We really can't do much better
  // than generic legalization for 64-bit multiplication-with-overflow, though.
  for (unsigned i = 0, e = 3+Subtarget->is64Bit(); i != e; ++i) {
    // Add/Sub/Mul with overflow operations are custom lowered.
    MVT VT = IntVTs[i];
    setOperationAction(ISD::SADDO, VT, Custom);
    setOperationAction(ISD::UADDO, VT, Custom);
    setOperationAction(ISD::SSUBO, VT, Custom);
    setOperationAction(ISD::USUBO, VT, Custom);
    setOperationAction(ISD::SMULO, VT, Custom);
    setOperationAction(ISD::UMULO, VT, Custom);
  }

  // There are no 8-bit 3-address imul/mul instructions
  setOperationAction(ISD::SMULO, MVT::i8, Expand);
  setOperationAction(ISD::UMULO, MVT::i8, Expand);

  if (!Subtarget->is64Bit()) {
    // These libcalls are not available in 32-bit.
    setLibcallName(RTLIB::SHL_I128, 0);
    setLibcallName(RTLIB::SRL_I128, 0);
    setLibcallName(RTLIB::SRA_I128, 0);
  }

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::VECTOR_SHUFFLE);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT);
  setTargetDAGCombine(ISD::VSELECT);
  setTargetDAGCombine(ISD::SELECT);
  setTargetDAGCombine(ISD::SHL);
  setTargetDAGCombine(ISD::SRA);
  setTargetDAGCombine(ISD::SRL);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::FADD);
  setTargetDAGCombine(ISD::FSUB);
  setTargetDAGCombine(ISD::FMA);
  setTargetDAGCombine(ISD::SUB);
  setTargetDAGCombine(ISD::LOAD);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  setTargetDAGCombine(ISD::ANY_EXTEND);
  setTargetDAGCombine(ISD::SIGN_EXTEND);
  setTargetDAGCombine(ISD::TRUNCATE);
  setTargetDAGCombine(ISD::SINT_TO_FP);
  setTargetDAGCombine(ISD::SETCC);
  if (Subtarget->is64Bit())
    setTargetDAGCombine(ISD::MUL);
  setTargetDAGCombine(ISD::XOR);

  computeRegisterProperties();

  // On Darwin, -Os means optimize for size without hurting performance,
  // do not reduce the limit.
  maxStoresPerMemset = 16; // For @llvm.memset -> sequence of stores
  maxStoresPerMemsetOptSize = Subtarget->isTargetDarwin() ? 16 : 8;
  maxStoresPerMemcpy = 8; // For @llvm.memcpy -> sequence of stores
  maxStoresPerMemcpyOptSize = Subtarget->isTargetDarwin() ? 8 : 4;
  maxStoresPerMemmove = 8; // For @llvm.memmove -> sequence of stores
  maxStoresPerMemmoveOptSize = Subtarget->isTargetDarwin() ? 8 : 4;
  setPrefLoopAlignment(4); // 2^4 bytes.
  benefitFromCodePlacementOpt = true;

  // Predictable cmov don't hurt on atom because it's in-order.
  predictableSelectIsExpensive = !Subtarget->isAtom();

  setPrefFunctionAlignment(4); // 2^4 bytes.
}


EVT X86TargetLowering::getSetCCResultType(EVT VT) const {
  if (!VT.isVector()) return MVT::i8;
  return VT.changeVectorElementTypeToInteger();
}


/// getMaxByValAlign - Helper for getByValTypeAlignment to determine
/// the desired ByVal argument alignment.
static void getMaxByValAlign(Type *Ty, unsigned &MaxAlign) {
  if (MaxAlign == 16)
    return;
  if (VectorType *VTy = dyn_cast<VectorType>(Ty)) {
    if (VTy->getBitWidth() == 128)
      MaxAlign = 16;
  } else if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    unsigned EltAlign = 0;
    getMaxByValAlign(ATy->getElementType(), EltAlign);
    if (EltAlign > MaxAlign)
      MaxAlign = EltAlign;
  } else if (StructType *STy = dyn_cast<StructType>(Ty)) {
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      unsigned EltAlign = 0;
      getMaxByValAlign(STy->getElementType(i), EltAlign);
      if (EltAlign > MaxAlign)
        MaxAlign = EltAlign;
      if (MaxAlign == 16)
        break;
    }
  }
}

/// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
/// function arguments in the caller parameter area. For X86, aggregates
/// that contain SSE vectors are placed at 16-byte boundaries while the rest
/// are at 4-byte boundaries.
unsigned X86TargetLowering::getByValTypeAlignment(Type *Ty) const {
  if (Subtarget->is64Bit()) {
    // Max of 8 and alignment of type.
    unsigned TyAlign = TD->getABITypeAlignment(Ty);
    if (TyAlign > 8)
      return TyAlign;
    return 8;
  }

  unsigned Align = 4;
  if (Subtarget->hasSSE1())
    getMaxByValAlign(Ty, Align);
  return Align;
}

/// getOptimalMemOpType - Returns the target specific optimal type for load
/// and store operations as a result of memset, memcpy, and memmove
/// lowering. If DstAlign is zero that means it's safe to destination
/// alignment can satisfy any constraint. Similarly if SrcAlign is zero it
/// means there isn't a need to check it against alignment requirement,
/// probably because the source does not need to be loaded. If
/// 'IsZeroVal' is true, that means it's safe to return a
/// non-scalar-integer type, e.g. empty string source, constant, or loaded
/// from memory. 'MemcpyStrSrc' indicates whether the memcpy source is
/// constant so it does not need to be loaded.
/// It returns EVT::Other if the type should be determined using generic
/// target-independent logic.
EVT
X86TargetLowering::getOptimalMemOpType(uint64_t Size,
                                       unsigned DstAlign, unsigned SrcAlign,
                                       bool IsZeroVal,
                                       bool MemcpyStrSrc,
                                       MachineFunction &MF) const {
  // FIXME: This turns off use of xmm stores for memset/memcpy on targets like
  // linux.  This is because the stack realignment code can't handle certain
  // cases like PR2962.  This should be removed when PR2962 is fixed.
  const Function *F = MF.getFunction();
  if (IsZeroVal &&
      !F->getFnAttributes().hasAttribute(Attributes::NoImplicitFloat)) {
    if (Size >= 16 &&
        (Subtarget->isUnalignedMemAccessFast() ||
         ((DstAlign == 0 || DstAlign >= 16) &&
          (SrcAlign == 0 || SrcAlign >= 16))) &&
        Subtarget->getStackAlignment() >= 16) {
      if (Subtarget->getStackAlignment() >= 32) {
        if (Subtarget->hasAVX2())
          return MVT::v8i32;
        if (Subtarget->hasAVX())
          return MVT::v8f32;
      }
      if (Subtarget->hasSSE2())
        return MVT::v4i32;
      if (Subtarget->hasSSE1())
        return MVT::v4f32;
    } else if (!MemcpyStrSrc && Size >= 8 &&
               !Subtarget->is64Bit() &&
               Subtarget->getStackAlignment() >= 8 &&
               Subtarget->hasSSE2()) {
      // Do not use f64 to lower memcpy if source is string constant. It's
      // better to use i32 to avoid the loads.
      return MVT::f64;
    }
  }
  if (Subtarget->is64Bit() && Size >= 8)
    return MVT::i64;
  return MVT::i32;
}

/// getJumpTableEncoding - Return the entry encoding for a jump table in the
/// current function.  The returned value is a member of the
/// MachineJumpTableInfo::JTEntryKind enum.
unsigned X86TargetLowering::getJumpTableEncoding() const {
  // In GOT pic mode, each entry in the jump table is emitted as a @GOTOFF
  // symbol.
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
      Subtarget->isPICStyleGOT())
    return MachineJumpTableInfo::EK_Custom32;

  // Otherwise, use the normal jump table encoding heuristics.
  return TargetLowering::getJumpTableEncoding();
}

const MCExpr *
X86TargetLowering::LowerCustomJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                             const MachineBasicBlock *MBB,
                                             unsigned uid,MCContext &Ctx) const{
  assert(getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
         Subtarget->isPICStyleGOT());
  // In 32-bit ELF systems, our jump table entries are formed with @GOTOFF
  // entries.
  return MCSymbolRefExpr::Create(MBB->getSymbol(),
                                 MCSymbolRefExpr::VK_GOTOFF, Ctx);
}

/// getPICJumpTableRelocaBase - Returns relocation base for the given PIC
/// jumptable.
SDValue X86TargetLowering::getPICJumpTableRelocBase(SDValue Table,
                                                    SelectionDAG &DAG) const {
  if (!Subtarget->is64Bit())
    // This doesn't have DebugLoc associated with it, but is not really the
    // same as a Register.
    return DAG.getNode(X86ISD::GlobalBaseReg, DebugLoc(), getPointerTy());
  return Table;
}

/// getPICJumpTableRelocBaseExpr - This returns the relocation base for the
/// given PIC jumptable, the same as getPICJumpTableRelocBase, but as an
/// MCExpr.
const MCExpr *X86TargetLowering::
getPICJumpTableRelocBaseExpr(const MachineFunction *MF, unsigned JTI,
                             MCContext &Ctx) const {
  // X86-64 uses RIP relative addressing based on the jump table label.
  if (Subtarget->isPICStyleRIPRel())
    return TargetLowering::getPICJumpTableRelocBaseExpr(MF, JTI, Ctx);

  // Otherwise, the reference is relative to the PIC base.
  return MCSymbolRefExpr::Create(MF->getPICBaseSymbol(), Ctx);
}

// FIXME: Why this routine is here? Move to RegInfo!
std::pair<const TargetRegisterClass*, uint8_t>
X86TargetLowering::findRepresentativeClass(EVT VT) const{
  const TargetRegisterClass *RRC = 0;
  uint8_t Cost = 1;
  switch (VT.getSimpleVT().SimpleTy) {
  default:
    return TargetLowering::findRepresentativeClass(VT);
  case MVT::i8: case MVT::i16: case MVT::i32: case MVT::i64:
    RRC = Subtarget->is64Bit() ?
      (const TargetRegisterClass*)&X86::GR64RegClass :
      (const TargetRegisterClass*)&X86::GR32RegClass;
    break;
  case MVT::x86mmx:
    RRC = &X86::VR64RegClass;
    break;
  case MVT::f32: case MVT::f64:
  case MVT::v16i8: case MVT::v8i16: case MVT::v4i32: case MVT::v2i64:
  case MVT::v4f32: case MVT::v2f64:
  case MVT::v32i8: case MVT::v8i32: case MVT::v4i64: case MVT::v8f32:
  case MVT::v4f64:
    RRC = &X86::VR128RegClass;
    break;
  }
  return std::make_pair(RRC, Cost);
}

bool X86TargetLowering::getStackCookieLocation(unsigned &AddressSpace,
                                               unsigned &Offset) const {
  if (!Subtarget->isTargetLinux())
    return false;

  if (Subtarget->is64Bit()) {
    // %fs:0x28, unless we're using a Kernel code model, in which case it's %gs:
    Offset = 0x28;
    if (getTargetMachine().getCodeModel() == CodeModel::Kernel)
      AddressSpace = 256;
    else
      AddressSpace = 257;
  } else {
    // %gs:0x14 on i386
    Offset = 0x14;
    AddressSpace = 256;
  }
  return true;
}


//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "X86GenCallingConv.inc"

bool
X86TargetLowering::CanLowerReturn(CallingConv::ID CallConv,
                                  MachineFunction &MF, bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(),
                 RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_X86);
}

SDValue
X86TargetLowering::LowerReturn(SDValue Chain,
                               CallingConv::ID CallConv, bool isVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               DebugLoc dl, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(),
                 RVLocs, *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_X86);

  // Add the regs to the liveout set for the function.
  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
  for (unsigned i = 0; i != RVLocs.size(); ++i)
    if (RVLocs[i].isRegLoc() && !MRI.isLiveOut(RVLocs[i].getLocReg()))
      MRI.addLiveOut(RVLocs[i].getLocReg());

  SDValue Flag;

  SmallVector<SDValue, 6> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)
  // Operand #1 = Bytes To Pop
  RetOps.push_back(DAG.getTargetConstant(FuncInfo->getBytesToPopOnReturn(),
                   MVT::i16));

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    SDValue ValToCopy = OutVals[i];
    EVT ValVT = ValToCopy.getValueType();

    // Promote values to the appropriate types
    if (VA.getLocInfo() == CCValAssign::SExt)
      ValToCopy = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), ValToCopy);
    else if (VA.getLocInfo() == CCValAssign::ZExt)
      ValToCopy = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), ValToCopy);
    else if (VA.getLocInfo() == CCValAssign::AExt)
      ValToCopy = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), ValToCopy);
    else if (VA.getLocInfo() == CCValAssign::BCvt)
      ValToCopy = DAG.getNode(ISD::BITCAST, dl, VA.getLocVT(), ValToCopy);

    // If this is x86-64, and we disabled SSE, we can't return FP values,
    // or SSE or MMX vectors.
    if ((ValVT == MVT::f32 || ValVT == MVT::f64 ||
         VA.getLocReg() == X86::XMM0 || VA.getLocReg() == X86::XMM1) &&
          (Subtarget->is64Bit() && !Subtarget->hasSSE1())) {
      report_fatal_error("SSE register return with SSE disabled");
    }
    // Likewise we can't return F64 values with SSE1 only.  gcc does so, but
    // llvm-gcc has never done it right and no one has noticed, so this
    // should be OK for now.
    if (ValVT == MVT::f64 &&
        (Subtarget->is64Bit() && !Subtarget->hasSSE2()))
      report_fatal_error("SSE2 register return with SSE2 disabled");

    // Returns in ST0/ST1 are handled specially: these are pushed as operands to
    // the RET instruction and handled by the FP Stackifier.
    if (VA.getLocReg() == X86::ST0 ||
        VA.getLocReg() == X86::ST1) {
      // If this is a copy from an xmm register to ST(0), use an FPExtend to
      // change the value to the FP stack register class.
      if (isScalarFPTypeInSSEReg(VA.getValVT()))
        ValToCopy = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f80, ValToCopy);
      RetOps.push_back(ValToCopy);
      // Don't emit a copytoreg.
      continue;
    }

    // 64-bit vector (MMX) values are returned in XMM0 / XMM1 except for v1i64
    // which is returned in RAX / RDX.
    if (Subtarget->is64Bit()) {
      if (ValVT == MVT::x86mmx) {
        if (VA.getLocReg() == X86::XMM0 || VA.getLocReg() == X86::XMM1) {
          ValToCopy = DAG.getNode(ISD::BITCAST, dl, MVT::i64, ValToCopy);
          ValToCopy = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2i64,
                                  ValToCopy);
          // If we don't have SSE2 available, convert to v4f32 so the generated
          // register is legal.
          if (!Subtarget->hasSSE2())
            ValToCopy = DAG.getNode(ISD::BITCAST, dl, MVT::v4f32,ValToCopy);
        }
      }
    }

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), ValToCopy, Flag);
    Flag = Chain.getValue(1);
  }

  // The x86-64 ABI for returning structs by value requires that we copy
  // the sret argument into %rax for the return. We saved the argument into
  // a virtual register in the entry block, so now we copy the value out
  // and into %rax.
  if (Subtarget->is64Bit() &&
      DAG.getMachineFunction().getFunction()->hasStructRetAttr()) {
    MachineFunction &MF = DAG.getMachineFunction();
    X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
    unsigned Reg = FuncInfo->getSRetReturnReg();
    assert(Reg &&
           "SRetReturnReg should have been set in LowerFormalArguments().");
    SDValue Val = DAG.getCopyFromReg(Chain, dl, Reg, getPointerTy());

    Chain = DAG.getCopyToReg(Chain, dl, X86::RAX, Val, Flag);
    Flag = Chain.getValue(1);

    // RAX now acts like a return value.
    MRI.addLiveOut(X86::RAX);
  }

  RetOps[0] = Chain;  // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(X86ISD::RET_FLAG, dl,
                     MVT::Other, &RetOps[0], RetOps.size());
}

bool X86TargetLowering::isUsedByReturnOnly(SDNode *N, SDValue &Chain) const {
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
  } else if (Copy->getOpcode() != ISD::FP_EXTEND)
    return false;

  bool HasRet = false;
  for (SDNode::use_iterator UI = Copy->use_begin(), UE = Copy->use_end();
       UI != UE; ++UI) {
    if (UI->getOpcode() != X86ISD::RET_FLAG)
      return false;
    HasRet = true;
  }

  if (!HasRet)
    return false;

  Chain = TCChain;
  return true;
}

EVT
X86TargetLowering::getTypeForExtArgOrReturn(LLVMContext &Context, EVT VT,
                                            ISD::NodeType ExtendKind) const {
  MVT ReturnMVT;
  // TODO: Is this also valid on 32-bit?
  if (Subtarget->is64Bit() && VT == MVT::i1 && ExtendKind == ISD::ZERO_EXTEND)
    ReturnMVT = MVT::i8;
  else
    ReturnMVT = MVT::i32;

  EVT MinVT = getRegisterType(Context, ReturnMVT);
  return VT.bitsLT(MinVT) ? MinVT : VT;
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
///
SDValue
X86TargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                   CallingConv::ID CallConv, bool isVarArg,
                                   const SmallVectorImpl<ISD::InputArg> &Ins,
                                   DebugLoc dl, SelectionDAG &DAG,
                                   SmallVectorImpl<SDValue> &InVals) const {

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  bool Is64Bit = Subtarget->is64Bit();
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC_X86);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    EVT CopyVT = VA.getValVT();

    // If this is x86-64, and we disabled SSE, we can't return FP values
    if ((CopyVT == MVT::f32 || CopyVT == MVT::f64) &&
        ((Is64Bit || Ins[i].Flags.isInReg()) && !Subtarget->hasSSE1())) {
      report_fatal_error("SSE register return with SSE disabled");
    }

    SDValue Val;

    // If this is a call to a function that returns an fp value on the floating
    // point stack, we must guarantee the value is popped from the stack, so
    // a CopyFromReg is not good enough - the copy instruction may be eliminated
    // if the return value is not used. We use the FpPOP_RETVAL instruction
    // instead.
    if (VA.getLocReg() == X86::ST0 || VA.getLocReg() == X86::ST1) {
      // If we prefer to use the value in xmm registers, copy it out as f80 and
      // use a truncate to move it from fp stack reg to xmm reg.
      if (isScalarFPTypeInSSEReg(VA.getValVT())) CopyVT = MVT::f80;
      SDValue Ops[] = { Chain, InFlag };
      Chain = SDValue(DAG.getMachineNode(X86::FpPOP_RETVAL, dl, CopyVT,
                                         MVT::Other, MVT::Glue, Ops, 2), 1);
      Val = Chain.getValue(0);

      // Round the f80 to the right size, which also moves it to the appropriate
      // xmm register.
      if (CopyVT != VA.getValVT())
        Val = DAG.getNode(ISD::FP_ROUND, dl, VA.getValVT(), Val,
                          // This truncation won't change the value.
                          DAG.getIntPtrConstant(1));
    } else {
      Chain = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(),
                                 CopyVT, InFlag).getValue(1);
      Val = Chain.getValue(0);
    }
    InFlag = Chain.getValue(2);
    InVals.push_back(Val);
  }

  return Chain;
}


//===----------------------------------------------------------------------===//
//                C & StdCall & Fast Calling Convention implementation
//===----------------------------------------------------------------------===//
//  StdCall calling convention seems to be standard for many Windows' API
//  routines and around. It differs from C calling convention just a little:
//  callee should clean up the stack, not caller. Symbols should be also
//  decorated in some fancy way :) It doesn't support any vector arguments.
//  For info on fast calling convention see Fast Calling Convention (tail call)
//  implementation LowerX86_32FastCCCallTo.

/// CallIsStructReturn - Determines whether a call uses struct return
/// semantics.
enum StructReturnType {
  NotStructReturn,
  RegStructReturn,
  StackStructReturn
};
static StructReturnType
callIsStructReturn(const SmallVectorImpl<ISD::OutputArg> &Outs) {
  if (Outs.empty())
    return NotStructReturn;

  const ISD::ArgFlagsTy &Flags = Outs[0].Flags;
  if (!Flags.isSRet())
    return NotStructReturn;
  if (Flags.isInReg())
    return RegStructReturn;
  return StackStructReturn;
}

/// ArgsAreStructReturn - Determines whether a function uses struct
/// return semantics.
static StructReturnType
argsAreStructReturn(const SmallVectorImpl<ISD::InputArg> &Ins) {
  if (Ins.empty())
    return NotStructReturn;

  const ISD::ArgFlagsTy &Flags = Ins[0].Flags;
  if (!Flags.isSRet())
    return NotStructReturn;
  if (Flags.isInReg())
    return RegStructReturn;
  return StackStructReturn;
}

/// CreateCopyOfByValArgument - Make a copy of an aggregate at address specified
/// by "Src" to address "Dst" with size and alignment information specified by
/// the specific parameter attribute. The copy will be passed as a byval
/// function parameter.
static SDValue
CreateCopyOfByValArgument(SDValue Src, SDValue Dst, SDValue Chain,
                          ISD::ArgFlagsTy Flags, SelectionDAG &DAG,
                          DebugLoc dl) {
  SDValue SizeNode = DAG.getConstant(Flags.getByValSize(), MVT::i32);

  return DAG.getMemcpy(Chain, dl, Dst, Src, SizeNode, Flags.getByValAlign(),
                       /*isVolatile*/false, /*AlwaysInline=*/true,
                       MachinePointerInfo(), MachinePointerInfo());
}

/// IsTailCallConvention - Return true if the calling convention is one that
/// supports tail call optimization.
static bool IsTailCallConvention(CallingConv::ID CC) {
  return (CC == CallingConv::Fast || CC == CallingConv::GHC);
}

bool X86TargetLowering::mayBeEmittedAsTailCall(CallInst *CI) const {
  if (!CI->isTailCall() || getTargetMachine().Options.DisableTailCalls)
    return false;

  CallSite CS(CI);
  CallingConv::ID CalleeCC = CS.getCallingConv();
  if (!IsTailCallConvention(CalleeCC) && CalleeCC != CallingConv::C)
    return false;

  return true;
}

/// FuncIsMadeTailCallSafe - Return true if the function is being made into
/// a tailcall target by changing its ABI.
static bool FuncIsMadeTailCallSafe(CallingConv::ID CC,
                                   bool GuaranteedTailCallOpt) {
  return GuaranteedTailCallOpt && IsTailCallConvention(CC);
}

SDValue
X86TargetLowering::LowerMemArgument(SDValue Chain,
                                    CallingConv::ID CallConv,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    const CCValAssign &VA,
                                    MachineFrameInfo *MFI,
                                    unsigned i) const {
  // Create the nodes corresponding to a load from this parameter slot.
  ISD::ArgFlagsTy Flags = Ins[i].Flags;
  bool AlwaysUseMutable = FuncIsMadeTailCallSafe(CallConv,
                              getTargetMachine().Options.GuaranteedTailCallOpt);
  bool isImmutable = !AlwaysUseMutable && !Flags.isByVal();
  EVT ValVT;

  // If value is passed by pointer we have address passed instead of the value
  // itself.
  if (VA.getLocInfo() == CCValAssign::Indirect)
    ValVT = VA.getLocVT();
  else
    ValVT = VA.getValVT();

  // FIXME: For now, all byval parameter objects are marked mutable. This can be
  // changed with more analysis.
  // In case of tail call optimization mark all arguments mutable. Since they
  // could be overwritten by lowering of arguments in case of a tail call.
  if (Flags.isByVal()) {
    unsigned Bytes = Flags.getByValSize();
    if (Bytes == 0) Bytes = 1; // Don't create zero-sized stack objects.
    int FI = MFI->CreateFixedObject(Bytes, VA.getLocMemOffset(), isImmutable);
    return DAG.getFrameIndex(FI, getPointerTy());
  } else {
    int FI = MFI->CreateFixedObject(ValVT.getSizeInBits()/8,
                                    VA.getLocMemOffset(), isImmutable);
    SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
    return DAG.getLoad(ValVT, dl, Chain, FIN,
                       MachinePointerInfo::getFixedStack(FI),
                       false, false, false, 0);
  }
}

SDValue
X86TargetLowering::LowerFormalArguments(SDValue Chain,
                                        CallingConv::ID CallConv,
                                        bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                        DebugLoc dl,
                                        SelectionDAG &DAG,
                                        SmallVectorImpl<SDValue> &InVals)
                                          const {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();

  const Function* Fn = MF.getFunction();
  if (Fn->hasExternalLinkage() &&
      Subtarget->isTargetCygMing() &&
      Fn->getName() == "main")
    FuncInfo->setForceFramePointer(true);

  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool Is64Bit = Subtarget->is64Bit();
  bool IsWindows = Subtarget->isTargetWindows();
  bool IsWin64 = Subtarget->isTargetWin64();

  assert(!(isVarArg && IsTailCallConvention(CallConv)) &&
         "Var args not supported with calling convention fastcc or ghc");

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(),
                 ArgLocs, *DAG.getContext());

  // Allocate shadow area for Win64
  if (IsWin64) {
    CCInfo.AllocateStack(32, 8);
  }

  CCInfo.AnalyzeFormalArguments(Ins, CC_X86);

  unsigned LastVal = ~0U;
  SDValue ArgValue;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    // TODO: If an arg is passed in two places (e.g. reg and stack), skip later
    // places.
    assert(VA.getValNo() != LastVal &&
           "Don't support value assigned to multiple locs yet");
    (void)LastVal;
    LastVal = VA.getValNo();

    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      const TargetRegisterClass *RC;
      if (RegVT == MVT::i32)
        RC = &X86::GR32RegClass;
      else if (Is64Bit && RegVT == MVT::i64)
        RC = &X86::GR64RegClass;
      else if (RegVT == MVT::f32)
        RC = &X86::FR32RegClass;
      else if (RegVT == MVT::f64)
        RC = &X86::FR64RegClass;
      else if (RegVT.is256BitVector())
        RC = &X86::VR256RegClass;
      else if (RegVT.is128BitVector())
        RC = &X86::VR128RegClass;
      else if (RegVT == MVT::x86mmx)
        RC = &X86::VR64RegClass;
      else
        llvm_unreachable("Unknown argument type!");

      unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
      ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);

      // If this is an 8 or 16-bit value, it is really passed promoted to 32
      // bits.  Insert an assert[sz]ext to capture this, then truncate to the
      // right size.
      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::BCvt)
        ArgValue = DAG.getNode(ISD::BITCAST, dl, VA.getValVT(), ArgValue);

      if (VA.isExtInLoc()) {
        // Handle MMX values passed in XMM regs.
        if (RegVT.isVector()) {
          ArgValue = DAG.getNode(X86ISD::MOVDQ2Q, dl, VA.getValVT(),
                                 ArgValue);
        } else
          ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
      }
    } else {
      assert(VA.isMemLoc());
      ArgValue = LowerMemArgument(Chain, CallConv, Ins, dl, DAG, VA, MFI, i);
    }

    // If value is passed via pointer - do a load.
    if (VA.getLocInfo() == CCValAssign::Indirect)
      ArgValue = DAG.getLoad(VA.getValVT(), dl, Chain, ArgValue,
                             MachinePointerInfo(), false, false, false, 0);

    InVals.push_back(ArgValue);
  }

  // The x86-64 ABI for returning structs by value requires that we copy
  // the sret argument into %rax for the return. Save the argument into
  // a virtual register so that we can access it from the return points.
  if (Is64Bit && MF.getFunction()->hasStructRetAttr()) {
    X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
    unsigned Reg = FuncInfo->getSRetReturnReg();
    if (!Reg) {
      Reg = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i64));
      FuncInfo->setSRetReturnReg(Reg);
    }
    SDValue Copy = DAG.getCopyToReg(DAG.getEntryNode(), dl, Reg, InVals[0]);
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Copy, Chain);
  }

  unsigned StackSize = CCInfo.getNextStackOffset();
  // Align stack specially for tail calls.
  if (FuncIsMadeTailCallSafe(CallConv,
                             MF.getTarget().Options.GuaranteedTailCallOpt))
    StackSize = GetAlignedArgumentStackSize(StackSize, DAG);

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (isVarArg) {
    if (Is64Bit || (CallConv != CallingConv::X86_FastCall &&
                    CallConv != CallingConv::X86_ThisCall)) {
      FuncInfo->setVarArgsFrameIndex(MFI->CreateFixedObject(1, StackSize,true));
    }
    if (Is64Bit) {
      unsigned TotalNumIntRegs = 0, TotalNumXMMRegs = 0;

      // FIXME: We should really autogenerate these arrays
      static const uint16_t GPR64ArgRegsWin64[] = {
        X86::RCX, X86::RDX, X86::R8,  X86::R9
      };
      static const uint16_t GPR64ArgRegs64Bit[] = {
        X86::RDI, X86::RSI, X86::RDX, X86::RCX, X86::R8, X86::R9
      };
      static const uint16_t XMMArgRegs64Bit[] = {
        X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
        X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
      };
      const uint16_t *GPR64ArgRegs;
      unsigned NumXMMRegs = 0;

      if (IsWin64) {
        // The XMM registers which might contain var arg parameters are shadowed
        // in their paired GPR.  So we only need to save the GPR to their home
        // slots.
        TotalNumIntRegs = 4;
        GPR64ArgRegs = GPR64ArgRegsWin64;
      } else {
        TotalNumIntRegs = 6; TotalNumXMMRegs = 8;
        GPR64ArgRegs = GPR64ArgRegs64Bit;

        NumXMMRegs = CCInfo.getFirstUnallocated(XMMArgRegs64Bit,
                                                TotalNumXMMRegs);
      }
      unsigned NumIntRegs = CCInfo.getFirstUnallocated(GPR64ArgRegs,
                                                       TotalNumIntRegs);

      bool NoImplicitFloatOps = Fn->getFnAttributes().
        hasAttribute(Attributes::NoImplicitFloat);
      assert(!(NumXMMRegs && !Subtarget->hasSSE1()) &&
             "SSE register cannot be used when SSE is disabled!");
      assert(!(NumXMMRegs && MF.getTarget().Options.UseSoftFloat &&
               NoImplicitFloatOps) &&
             "SSE register cannot be used when SSE is disabled!");
      if (MF.getTarget().Options.UseSoftFloat || NoImplicitFloatOps ||
          !Subtarget->hasSSE1())
        // Kernel mode asks for SSE to be disabled, so don't push them
        // on the stack.
        TotalNumXMMRegs = 0;

      if (IsWin64) {
        const TargetFrameLowering &TFI = *getTargetMachine().getFrameLowering();
        // Get to the caller-allocated home save location.  Add 8 to account
        // for the return address.
        int HomeOffset = TFI.getOffsetOfLocalArea() + 8;
        FuncInfo->setRegSaveFrameIndex(
          MFI->CreateFixedObject(1, NumIntRegs * 8 + HomeOffset, false));
        // Fixup to set vararg frame on shadow area (4 x i64).
        if (NumIntRegs < 4)
          FuncInfo->setVarArgsFrameIndex(FuncInfo->getRegSaveFrameIndex());
      } else {
        // For X86-64, if there are vararg parameters that are passed via
        // registers, then we must store them to their spots on the stack so
        // they may be loaded by deferencing the result of va_next.
        FuncInfo->setVarArgsGPOffset(NumIntRegs * 8);
        FuncInfo->setVarArgsFPOffset(TotalNumIntRegs * 8 + NumXMMRegs * 16);
        FuncInfo->setRegSaveFrameIndex(
          MFI->CreateStackObject(TotalNumIntRegs * 8 + TotalNumXMMRegs * 16, 16,
                               false));
      }

      // Store the integer parameter registers.
      SmallVector<SDValue, 8> MemOps;
      SDValue RSFIN = DAG.getFrameIndex(FuncInfo->getRegSaveFrameIndex(),
                                        getPointerTy());
      unsigned Offset = FuncInfo->getVarArgsGPOffset();
      for (; NumIntRegs != TotalNumIntRegs; ++NumIntRegs) {
        SDValue FIN = DAG.getNode(ISD::ADD, dl, getPointerTy(), RSFIN,
                                  DAG.getIntPtrConstant(Offset));
        unsigned VReg = MF.addLiveIn(GPR64ArgRegs[NumIntRegs],
                                     &X86::GR64RegClass);
        SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i64);
        SDValue Store =
          DAG.getStore(Val.getValue(1), dl, Val, FIN,
                       MachinePointerInfo::getFixedStack(
                         FuncInfo->getRegSaveFrameIndex(), Offset),
                       false, false, 0);
        MemOps.push_back(Store);
        Offset += 8;
      }

      if (TotalNumXMMRegs != 0 && NumXMMRegs != TotalNumXMMRegs) {
        // Now store the XMM (fp + vector) parameter registers.
        SmallVector<SDValue, 11> SaveXMMOps;
        SaveXMMOps.push_back(Chain);

        unsigned AL = MF.addLiveIn(X86::AL, &X86::GR8RegClass);
        SDValue ALVal = DAG.getCopyFromReg(DAG.getEntryNode(), dl, AL, MVT::i8);
        SaveXMMOps.push_back(ALVal);

        SaveXMMOps.push_back(DAG.getIntPtrConstant(
                               FuncInfo->getRegSaveFrameIndex()));
        SaveXMMOps.push_back(DAG.getIntPtrConstant(
                               FuncInfo->getVarArgsFPOffset()));

        for (; NumXMMRegs != TotalNumXMMRegs; ++NumXMMRegs) {
          unsigned VReg = MF.addLiveIn(XMMArgRegs64Bit[NumXMMRegs],
                                       &X86::VR128RegClass);
          SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, MVT::v4f32);
          SaveXMMOps.push_back(Val);
        }
        MemOps.push_back(DAG.getNode(X86ISD::VASTART_SAVE_XMM_REGS, dl,
                                     MVT::Other,
                                     &SaveXMMOps[0], SaveXMMOps.size()));
      }

      if (!MemOps.empty())
        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                            &MemOps[0], MemOps.size());
    }
  }

  // Some CCs need callee pop.
  if (X86::isCalleePop(CallConv, Is64Bit, isVarArg,
                       MF.getTarget().Options.GuaranteedTailCallOpt)) {
    FuncInfo->setBytesToPopOnReturn(StackSize); // Callee pops everything.
  } else {
    FuncInfo->setBytesToPopOnReturn(0); // Callee pops nothing.
    // If this is an sret function, the return should pop the hidden pointer.
    if (!Is64Bit && !IsTailCallConvention(CallConv) && !IsWindows &&
        argsAreStructReturn(Ins) == StackStructReturn)
      FuncInfo->setBytesToPopOnReturn(4);
  }

  if (!Is64Bit) {
    // RegSaveFrameIndex is X86-64 only.
    FuncInfo->setRegSaveFrameIndex(0xAAAAAAA);
    if (CallConv == CallingConv::X86_FastCall ||
        CallConv == CallingConv::X86_ThisCall)
      // fastcc functions can't have varargs.
      FuncInfo->setVarArgsFrameIndex(0xAAAAAAA);
  }

  FuncInfo->setArgumentStackSize(StackSize);

  return Chain;
}

SDValue
X86TargetLowering::LowerMemOpCallTo(SDValue Chain,
                                    SDValue StackPtr, SDValue Arg,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    const CCValAssign &VA,
                                    ISD::ArgFlagsTy Flags) const {
  unsigned LocMemOffset = VA.getLocMemOffset();
  SDValue PtrOff = DAG.getIntPtrConstant(LocMemOffset);
  PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, PtrOff);
  if (Flags.isByVal())
    return CreateCopyOfByValArgument(Arg, PtrOff, Chain, Flags, DAG, dl);

  return DAG.getStore(Chain, dl, Arg, PtrOff,
                      MachinePointerInfo::getStack(LocMemOffset),
                      false, false, 0);
}

/// EmitTailCallLoadRetAddr - Emit a load of return address if tail call
/// optimization is performed and it is required.
SDValue
X86TargetLowering::EmitTailCallLoadRetAddr(SelectionDAG &DAG,
                                           SDValue &OutRetAddr, SDValue Chain,
                                           bool IsTailCall, bool Is64Bit,
                                           int FPDiff, DebugLoc dl) const {
  // Adjust the Return address stack slot.
  EVT VT = getPointerTy();
  OutRetAddr = getReturnAddressFrameIndex(DAG);

  // Load the "old" Return address.
  OutRetAddr = DAG.getLoad(VT, dl, Chain, OutRetAddr, MachinePointerInfo(),
                           false, false, false, 0);
  return SDValue(OutRetAddr.getNode(), 1);
}

/// EmitTailCallStoreRetAddr - Emit a store of the return address if tail call
/// optimization is performed and it is required (FPDiff!=0).
static SDValue
EmitTailCallStoreRetAddr(SelectionDAG & DAG, MachineFunction &MF,
                         SDValue Chain, SDValue RetAddrFrIdx, EVT PtrVT,
                         unsigned SlotSize, int FPDiff, DebugLoc dl) {
  // Store the return address to the appropriate stack slot.
  if (!FPDiff) return Chain;
  // Calculate the new stack slot for the return address.
  int NewReturnAddrFI =
    MF.getFrameInfo()->CreateFixedObject(SlotSize, FPDiff-SlotSize, false);
  SDValue NewRetAddrFrIdx = DAG.getFrameIndex(NewReturnAddrFI, PtrVT);
  Chain = DAG.getStore(Chain, dl, RetAddrFrIdx, NewRetAddrFrIdx,
                       MachinePointerInfo::getFixedStack(NewReturnAddrFI),
                       false, false, 0);
  return Chain;
}

SDValue
X86TargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                             SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG                     = CLI.DAG;
  DebugLoc &dl                          = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals     = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins   = CLI.Ins;
  SDValue Chain                         = CLI.Chain;
  SDValue Callee                        = CLI.Callee;
  CallingConv::ID CallConv              = CLI.CallConv;
  bool &isTailCall                      = CLI.IsTailCall;
  bool isVarArg                         = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  bool Is64Bit        = Subtarget->is64Bit();
  bool IsWin64        = Subtarget->isTargetWin64();
  bool IsWindows      = Subtarget->isTargetWindows();
  StructReturnType SR = callIsStructReturn(Outs);
  bool IsSibcall      = false;

  if (MF.getTarget().Options.DisableTailCalls)
    isTailCall = false;

  if (isTailCall) {
    // Check if it's really possible to do a tail call.
    isTailCall = IsEligibleForTailCallOptimization(Callee, CallConv,
                    isVarArg, SR != NotStructReturn,
                    MF.getFunction()->hasStructRetAttr(), CLI.RetTy,
                    Outs, OutVals, Ins, DAG);

    // Sibcalls are automatically detected tailcalls which do not require
    // ABI changes.
    if (!MF.getTarget().Options.GuaranteedTailCallOpt && isTailCall)
      IsSibcall = true;

    if (isTailCall)
      ++NumTailCalls;
  }

  assert(!(isVarArg && IsTailCallConvention(CallConv)) &&
         "Var args not supported with calling convention fastcc or ghc");

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(),
                 ArgLocs, *DAG.getContext());

  // Allocate shadow area for Win64
  if (IsWin64) {
    CCInfo.AllocateStack(32, 8);
  }

  CCInfo.AnalyzeCallOperands(Outs, CC_X86);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();
  if (IsSibcall)
    // This is a sibcall. The memory operands are available in caller's
    // own caller's stack.
    NumBytes = 0;
  else if (getTargetMachine().Options.GuaranteedTailCallOpt &&
           IsTailCallConvention(CallConv))
    NumBytes = GetAlignedArgumentStackSize(NumBytes, DAG);

  int FPDiff = 0;
  if (isTailCall && !IsSibcall) {
    // Lower arguments at fp - stackoffset + fpdiff.
    X86MachineFunctionInfo *X86Info = MF.getInfo<X86MachineFunctionInfo>();
    unsigned NumBytesCallerPushed = X86Info->getBytesToPopOnReturn();

    FPDiff = NumBytesCallerPushed - NumBytes;

    // Set the delta of movement of the returnaddr stackslot.
    // But only set if delta is greater than previous delta.
    if (FPDiff < X86Info->getTCReturnAddrDelta())
      X86Info->setTCReturnAddrDelta(FPDiff);
  }

  if (!IsSibcall)
    Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));

  SDValue RetAddrFrIdx;
  // Load return address for tail calls.
  if (isTailCall && FPDiff)
    Chain = EmitTailCallLoadRetAddr(DAG, RetAddrFrIdx, Chain, isTailCall,
                                    Is64Bit, FPDiff, dl);

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;

  // Walk the register/memloc assignments, inserting copies/loads.  In the case
  // of tail call optimization arguments are handle later.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    EVT RegVT = VA.getLocVT();
    SDValue Arg = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    bool isByVal = Flags.isByVal();

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::AExt:
      if (RegVT.is128BitVector()) {
        // Special case: passing MMX values in XMM registers.
        Arg = DAG.getNode(ISD::BITCAST, dl, MVT::i64, Arg);
        Arg = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2i64, Arg);
        Arg = getMOVL(DAG, dl, MVT::v2i64, DAG.getUNDEF(MVT::v2i64), Arg);
      } else
        Arg = DAG.getNode(ISD::ANY_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, dl, RegVT, Arg);
      break;
    case CCValAssign::Indirect: {
      // Store the argument.
      SDValue SpillSlot = DAG.CreateStackTemporary(VA.getValVT());
      int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
      Chain = DAG.getStore(Chain, dl, Arg, SpillSlot,
                           MachinePointerInfo::getFixedStack(FI),
                           false, false, 0);
      Arg = SpillSlot;
      break;
    }
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      if (isVarArg && IsWin64) {
        // Win64 ABI requires argument XMM reg to be copied to the corresponding
        // shadow reg if callee is a varargs function.
        unsigned ShadowReg = 0;
        switch (VA.getLocReg()) {
        case X86::XMM0: ShadowReg = X86::RCX; break;
        case X86::XMM1: ShadowReg = X86::RDX; break;
        case X86::XMM2: ShadowReg = X86::R8; break;
        case X86::XMM3: ShadowReg = X86::R9; break;
        }
        if (ShadowReg)
          RegsToPass.push_back(std::make_pair(ShadowReg, Arg));
      }
    } else if (!IsSibcall && (!isTailCall || isByVal)) {
      assert(VA.isMemLoc());
      if (StackPtr.getNode() == 0)
        StackPtr = DAG.getCopyFromReg(Chain, dl, RegInfo->getStackRegister(),
                                      getPointerTy());
      MemOpChains.push_back(LowerMemOpCallTo(Chain, StackPtr, Arg,
                                             dl, DAG, VA, Flags));
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  if (Subtarget->isPICStyleGOT()) {
    // ELF / PIC requires GOT in the EBX register before function calls via PLT
    // GOT pointer.
    if (!isTailCall) {
      RegsToPass.push_back(std::make_pair(unsigned(X86::EBX),
               DAG.getNode(X86ISD::GlobalBaseReg, DebugLoc(), getPointerTy())));
    } else {
      // If we are tail calling and generating PIC/GOT style code load the
      // address of the callee into ECX. The value in ecx is used as target of
      // the tail jump. This is done to circumvent the ebx/callee-saved problem
      // for tail calls on PIC/GOT architectures. Normally we would just put the
      // address of GOT into ebx and then call target@PLT. But for tail calls
      // ebx would be restored (since ebx is callee saved) before jumping to the
      // target@PLT.

      // Note: The actual moving to ECX is done further down.
      GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee);
      if (G && !G->getGlobal()->hasHiddenVisibility() &&
          !G->getGlobal()->hasProtectedVisibility())
        Callee = LowerGlobalAddress(Callee, DAG);
      else if (isa<ExternalSymbolSDNode>(Callee))
        Callee = LowerExternalSymbol(Callee, DAG);
    }
  }

  if (Is64Bit && isVarArg && !IsWin64) {
    // From AMD64 ABI document:
    // For calls that may call functions that use varargs or stdargs
    // (prototype-less calls or calls to functions containing ellipsis (...) in
    // the declaration) %al is used as hidden argument to specify the number
    // of SSE registers used. The contents of %al do not need to match exactly
    // the number of registers, but must be an ubound on the number of SSE
    // registers used and is in the range 0 - 8 inclusive.

    // Count the number of XMM registers allocated.
    static const uint16_t XMMArgRegs[] = {
      X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
      X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
    };
    unsigned NumXMMRegs = CCInfo.getFirstUnallocated(XMMArgRegs, 8);
    assert((Subtarget->hasSSE1() || !NumXMMRegs)
           && "SSE registers cannot be used when SSE is disabled");

    RegsToPass.push_back(std::make_pair(unsigned(X86::AL),
                                        DAG.getConstant(NumXMMRegs, MVT::i8)));
  }

  // For tail calls lower the arguments to the 'real' stack slot.
  if (isTailCall) {
    // Force all the incoming stack arguments to be loaded from the stack
    // before any new outgoing arguments are stored to the stack, because the
    // outgoing stack slots may alias the incoming argument stack slots, and
    // the alias isn't otherwise explicit. This is slightly more conservative
    // than necessary, because it means that each store effectively depends
    // on every argument instead of just those arguments it would clobber.
    SDValue ArgChain = DAG.getStackArgumentTokenFactor(Chain);

    SmallVector<SDValue, 8> MemOpChains2;
    SDValue FIN;
    int FI = 0;
    if (getTargetMachine().Options.GuaranteedTailCallOpt) {
      for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
        CCValAssign &VA = ArgLocs[i];
        if (VA.isRegLoc())
          continue;
        assert(VA.isMemLoc());
        SDValue Arg = OutVals[i];
        ISD::ArgFlagsTy Flags = Outs[i].Flags;
        // Create frame index.
        int32_t Offset = VA.getLocMemOffset()+FPDiff;
        uint32_t OpSize = (VA.getLocVT().getSizeInBits()+7)/8;
        FI = MF.getFrameInfo()->CreateFixedObject(OpSize, Offset, true);
        FIN = DAG.getFrameIndex(FI, getPointerTy());

        if (Flags.isByVal()) {
          // Copy relative to framepointer.
          SDValue Source = DAG.getIntPtrConstant(VA.getLocMemOffset());
          if (StackPtr.getNode() == 0)
            StackPtr = DAG.getCopyFromReg(Chain, dl,
                                          RegInfo->getStackRegister(),
                                          getPointerTy());
          Source = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, Source);

          MemOpChains2.push_back(CreateCopyOfByValArgument(Source, FIN,
                                                           ArgChain,
                                                           Flags, DAG, dl));
        } else {
          // Store relative to framepointer.
          MemOpChains2.push_back(
            DAG.getStore(ArgChain, dl, Arg, FIN,
                         MachinePointerInfo::getFixedStack(FI),
                         false, false, 0));
        }
      }
    }

    if (!MemOpChains2.empty())
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                          &MemOpChains2[0], MemOpChains2.size());

    // Store the return address to the appropriate stack slot.
    Chain = EmitTailCallStoreRetAddr(DAG, MF, Chain, RetAddrFrIdx,
                                     getPointerTy(), RegInfo->getSlotSize(),
                                     FPDiff, dl);
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into registers.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  if (getTargetMachine().getCodeModel() == CodeModel::Large) {
    assert(Is64Bit && "Large code model is only legal in 64-bit mode.");
    // In the 64-bit large code model, we have to make all calls
    // through a register, since the call instruction's 32-bit
    // pc-relative offset may not be large enough to hold the whole
    // address.
  } else if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    // If the callee is a GlobalAddress node (quite common, every direct call
    // is) turn it into a TargetGlobalAddress node so that legalize doesn't hack
    // it.

    // We should use extra load for direct calls to dllimported functions in
    // non-JIT mode.
    const GlobalValue *GV = G->getGlobal();
    if (!GV->hasDLLImportLinkage()) {
      unsigned char OpFlags = 0;
      bool ExtraLoad = false;
      unsigned WrapperKind = ISD::DELETED_NODE;

      // On ELF targets, in both X86-64 and X86-32 mode, direct calls to
      // external symbols most go through the PLT in PIC mode.  If the symbol
      // has hidden or protected visibility, or if it is static or local, then
      // we don't need to use the PLT - we can directly call it.
      if (Subtarget->isTargetELF() &&
          getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
          GV->hasDefaultVisibility() && !GV->hasLocalLinkage()) {
        OpFlags = X86II::MO_PLT;
      } else if (Subtarget->isPICStyleStubAny() &&
                 (GV->isDeclaration() || GV->isWeakForLinker()) &&
                 (!Subtarget->getTargetTriple().isMacOSX() ||
                  Subtarget->getTargetTriple().isMacOSXVersionLT(10, 5))) {
        // PC-relative references to external symbols should go through $stub,
        // unless we're building with the leopard linker or later, which
        // automatically synthesizes these stubs.
        OpFlags = X86II::MO_DARWIN_STUB;
      } else if (Subtarget->isPICStyleRIPRel() &&
                 isa<Function>(GV) &&
                 cast<Function>(GV)->getFnAttributes().
                   hasAttribute(Attributes::NonLazyBind)) {
        // If the function is marked as non-lazy, generate an indirect call
        // which loads from the GOT directly. This avoids runtime overhead
        // at the cost of eager binding (and one extra byte of encoding).
        OpFlags = X86II::MO_GOTPCREL;
        WrapperKind = X86ISD::WrapperRIP;
        ExtraLoad = true;
      }

      Callee = DAG.getTargetGlobalAddress(GV, dl, getPointerTy(),
                                          G->getOffset(), OpFlags);

      // Add a wrapper if needed.
      if (WrapperKind != ISD::DELETED_NODE)
        Callee = DAG.getNode(X86ISD::WrapperRIP, dl, getPointerTy(), Callee);
      // Add extra indirection if needed.
      if (ExtraLoad)
        Callee = DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(), Callee,
                             MachinePointerInfo::getGOT(),
                             false, false, false, 0);
    }
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    unsigned char OpFlags = 0;

    // On ELF targets, in either X86-64 or X86-32 mode, direct calls to
    // external symbols should go through the PLT.
    if (Subtarget->isTargetELF() &&
        getTargetMachine().getRelocationModel() == Reloc::PIC_) {
      OpFlags = X86II::MO_PLT;
    } else if (Subtarget->isPICStyleStubAny() &&
               (!Subtarget->getTargetTriple().isMacOSX() ||
                Subtarget->getTargetTriple().isMacOSXVersionLT(10, 5))) {
      // PC-relative references to external symbols should go through $stub,
      // unless we're building with the leopard linker or later, which
      // automatically synthesizes these stubs.
      OpFlags = X86II::MO_DARWIN_STUB;
    }

    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy(),
                                         OpFlags);
  }

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;

  if (!IsSibcall && isTailCall) {
    Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                           DAG.getIntPtrConstant(0, true), InFlag);
    InFlag = Chain.getValue(1);
  }

  Ops.push_back(Chain);
  Ops.push_back(Callee);

  if (isTailCall)
    Ops.push_back(DAG.getConstant(FPDiff, MVT::i32));

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  if (isTailCall) {
    // We used to do:
    //// If this is the first return lowered for this function, add the regs
    //// to the liveout set for the function.
    // This isn't right, although it's probably harmless on x86; liveouts
    // should be computed from returns not tail calls.  Consider a void
    // function making a tail call to a function returning int.
    return DAG.getNode(X86ISD::TC_RETURN, dl,
                       NodeTys, &Ops[0], Ops.size());
  }

  Chain = DAG.getNode(X86ISD::CALL, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  unsigned NumBytesForCalleeToPush;
  if (X86::isCalleePop(CallConv, Is64Bit, isVarArg,
                       getTargetMachine().Options.GuaranteedTailCallOpt))
    NumBytesForCalleeToPush = NumBytes;    // Callee pops everything
  else if (!Is64Bit && !IsTailCallConvention(CallConv) && !IsWindows &&
           SR == StackStructReturn)
    // If this is a call to a struct-return function, the callee
    // pops the hidden struct pointer, so we have to push it back.
    // This is common for Darwin/X86, Linux & Mingw32 targets.
    // For MSVC Win32 targets, the caller pops the hidden struct pointer.
    NumBytesForCalleeToPush = 4;
  else
    NumBytesForCalleeToPush = 0;  // Callee pops nothing.

  // Returns a flag for retval copy to use.
  if (!IsSibcall) {
    Chain = DAG.getCALLSEQ_END(Chain,
                               DAG.getIntPtrConstant(NumBytes, true),
                               DAG.getIntPtrConstant(NumBytesForCalleeToPush,
                                                     true),
                               InFlag);
    InFlag = Chain.getValue(1);
  }

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg,
                         Ins, dl, DAG, InVals);
}


//===----------------------------------------------------------------------===//
//                Fast Calling Convention (tail call) implementation
//===----------------------------------------------------------------------===//

//  Like std call, callee cleans arguments, convention except that ECX is
//  reserved for storing the tail called function address. Only 2 registers are
//  free for argument passing (inreg). Tail call optimization is performed
//  provided:
//                * tailcallopt is enabled
//                * caller/callee are fastcc
//  On X86_64 architecture with GOT-style position independent code only local
//  (within module) calls are supported at the moment.
//  To keep the stack aligned according to platform abi the function
//  GetAlignedArgumentStackSize ensures that argument delta is always multiples
//  of stack alignment. (Dynamic linkers need this - darwin's dyld for example)
//  If a tail called function callee has more arguments than the caller the
//  caller needs to make sure that there is room to move the RETADDR to. This is
//  achieved by reserving an area the size of the argument delta right after the
//  original REtADDR, but before the saved framepointer or the spilled registers
//  e.g. caller(arg1, arg2) calls callee(arg1, arg2,arg3,arg4)
//  stack layout:
//    arg1
//    arg2
//    RETADDR
//    [ new RETADDR
//      move area ]
//    (possible EBP)
//    ESI
//    EDI
//    local1 ..

/// GetAlignedArgumentStackSize - Make the stack size align e.g 16n + 12 aligned
/// for a 16 byte align requirement.
unsigned
X86TargetLowering::GetAlignedArgumentStackSize(unsigned StackSize,
                                               SelectionDAG& DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  const TargetMachine &TM = MF.getTarget();
  const TargetFrameLowering &TFI = *TM.getFrameLowering();
  unsigned StackAlignment = TFI.getStackAlignment();
  uint64_t AlignMask = StackAlignment - 1;
  int64_t Offset = StackSize;
  unsigned SlotSize = RegInfo->getSlotSize();
  if ( (Offset & AlignMask) <= (StackAlignment - SlotSize) ) {
    // Number smaller than 12 so just add the difference.
    Offset += ((StackAlignment - SlotSize) - (Offset & AlignMask));
  } else {
    // Mask out lower bits, add stackalignment once plus the 12 bytes.
    Offset = ((~AlignMask) & Offset) + StackAlignment +
      (StackAlignment-SlotSize);
  }
  return Offset;
}

/// MatchingStackOffset - Return true if the given stack call argument is
/// already available in the same position (relatively) of the caller's
/// incoming argument stack.
static
bool MatchingStackOffset(SDValue Arg, unsigned Offset, ISD::ArgFlagsTy Flags,
                         MachineFrameInfo *MFI, const MachineRegisterInfo *MRI,
                         const X86InstrInfo *TII) {
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
      unsigned Opcode = Def->getOpcode();
      if ((Opcode == X86::LEA32r || Opcode == X86::LEA64r) &&
          Def->getOperand(1).isFI()) {
        FI = Def->getOperand(1).getIndex();
        Bytes = Flags.getByValSize();
      } else
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
  } else if (Arg.getOpcode() == ISD::FrameIndex && Flags.isByVal()) {
    FrameIndexSDNode *FINode = cast<FrameIndexSDNode>(Arg);
    FI = FINode->getIndex();
    Bytes = Flags.getByValSize();
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
X86TargetLowering::IsEligibleForTailCallOptimization(SDValue Callee,
                                                     CallingConv::ID CalleeCC,
                                                     bool isVarArg,
                                                     bool isCalleeStructRet,
                                                     bool isCallerStructRet,
                                                     Type *RetTy,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                                     SelectionDAG& DAG) const {
  if (!IsTailCallConvention(CalleeCC) &&
      CalleeCC != CallingConv::C)
    return false;

  // If -tailcallopt is specified, make fastcc functions tail-callable.
  const MachineFunction &MF = DAG.getMachineFunction();
  const Function *CallerF = DAG.getMachineFunction().getFunction();

  // If the function return type is x86_fp80 and the callee return type is not,
  // then the FP_EXTEND of the call result is not a nop. It's not safe to
  // perform a tailcall optimization here.
  if (CallerF->getReturnType()->isX86_FP80Ty() && !RetTy->isX86_FP80Ty())
    return false;

  CallingConv::ID CallerCC = CallerF->getCallingConv();
  bool CCMatch = CallerCC == CalleeCC;

  if (getTargetMachine().Options.GuaranteedTailCallOpt) {
    if (IsTailCallConvention(CalleeCC) && CCMatch)
      return true;
    return false;
  }

  // Look for obvious safe cases to perform tail call optimization that do not
  // require ABI changes. This is what gcc calls sibcall.

  // Can't do sibcall if stack needs to be dynamically re-aligned. PEI needs to
  // emit a special epilogue.
  if (RegInfo->needsStackRealignment(MF))
    return false;

  // Also avoid sibcall optimization if either caller or callee uses struct
  // return semantics.
  if (isCalleeStructRet || isCallerStructRet)
    return false;

  // An stdcall caller is expected to clean up its arguments; the callee
  // isn't going to do that.
  if (!CCMatch && CallerCC==CallingConv::X86_StdCall)
    return false;

  // Do not sibcall optimize vararg calls unless all arguments are passed via
  // registers.
  if (isVarArg && !Outs.empty()) {

    // Optimizing for varargs on Win64 is unlikely to be safe without
    // additional testing.
    if (Subtarget->isTargetWin64())
      return false;

    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CalleeCC, isVarArg, DAG.getMachineFunction(),
                   getTargetMachine(), ArgLocs, *DAG.getContext());

    CCInfo.AnalyzeCallOperands(Outs, CC_X86);
    for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i)
      if (!ArgLocs[i].isRegLoc())
        return false;
  }

  // If the call result is in ST0 / ST1, it needs to be popped off the x87
  // stack.  Therefore, if it's not used by the call it is not safe to optimize
  // this into a sibcall.
  bool Unused = false;
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    if (!Ins[i].Used) {
      Unused = true;
      break;
    }
  }
  if (Unused) {
    SmallVector<CCValAssign, 16> RVLocs;
    CCState CCInfo(CalleeCC, false, DAG.getMachineFunction(),
                   getTargetMachine(), RVLocs, *DAG.getContext());
    CCInfo.AnalyzeCallResult(Ins, RetCC_X86);
    for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
      CCValAssign &VA = RVLocs[i];
      if (VA.getLocReg() == X86::ST0 || VA.getLocReg() == X86::ST1)
        return false;
    }
  }

  // If the calling conventions do not match, then we'd better make sure the
  // results are returned in the same way as what the caller expects.
  if (!CCMatch) {
    SmallVector<CCValAssign, 16> RVLocs1;
    CCState CCInfo1(CalleeCC, false, DAG.getMachineFunction(),
                    getTargetMachine(), RVLocs1, *DAG.getContext());
    CCInfo1.AnalyzeCallResult(Ins, RetCC_X86);

    SmallVector<CCValAssign, 16> RVLocs2;
    CCState CCInfo2(CallerCC, false, DAG.getMachineFunction(),
                    getTargetMachine(), RVLocs2, *DAG.getContext());
    CCInfo2.AnalyzeCallResult(Ins, RetCC_X86);

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

  // If the callee takes no arguments then go on to check the results of the
  // call.
  if (!Outs.empty()) {
    // Check if stack adjustment is needed. For now, do not do this if any
    // argument is passed on the stack.
    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CalleeCC, isVarArg, DAG.getMachineFunction(),
                   getTargetMachine(), ArgLocs, *DAG.getContext());

    // Allocate shadow area for Win64
    if (Subtarget->isTargetWin64()) {
      CCInfo.AllocateStack(32, 8);
    }

    CCInfo.AnalyzeCallOperands(Outs, CC_X86);
    if (CCInfo.getNextStackOffset()) {
      MachineFunction &MF = DAG.getMachineFunction();
      if (MF.getInfo<X86MachineFunctionInfo>()->getBytesToPopOnReturn())
        return false;

      // Check if the arguments are already laid out in the right way as
      // the caller's fixed stack objects.
      MachineFrameInfo *MFI = MF.getFrameInfo();
      const MachineRegisterInfo *MRI = &MF.getRegInfo();
      const X86InstrInfo *TII =
        ((const X86TargetMachine&)getTargetMachine()).getInstrInfo();
      for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
        CCValAssign &VA = ArgLocs[i];
        SDValue Arg = OutVals[i];
        ISD::ArgFlagsTy Flags = Outs[i].Flags;
        if (VA.getLocInfo() == CCValAssign::Indirect)
          return false;
        if (!VA.isRegLoc()) {
          if (!MatchingStackOffset(Arg, VA.getLocMemOffset(), Flags,
                                   MFI, MRI, TII))
            return false;
        }
      }
    }

    // If the tailcall address may be in a register, then make sure it's
    // possible to register allocate for it. In 32-bit, the call address can
    // only target EAX, EDX, or ECX since the tail call must be scheduled after
    // callee-saved registers are restored. These happen to be the same
    // registers used to pass 'inreg' arguments so watch out for those.
    if (!Subtarget->is64Bit() &&
        !isa<GlobalAddressSDNode>(Callee) &&
        !isa<ExternalSymbolSDNode>(Callee)) {
      unsigned NumInRegs = 0;
      for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
        CCValAssign &VA = ArgLocs[i];
        if (!VA.isRegLoc())
          continue;
        unsigned Reg = VA.getLocReg();
        switch (Reg) {
        default: break;
        case X86::EAX: case X86::EDX: case X86::ECX:
          if (++NumInRegs == 3)
            return false;
          break;
        }
      }
    }
  }

  return true;
}

FastISel *
X86TargetLowering::createFastISel(FunctionLoweringInfo &funcInfo,
                                  const TargetLibraryInfo *libInfo) const {
  return X86::createFastISel(funcInfo, libInfo);
}


//===----------------------------------------------------------------------===//
//                           Other Lowering Hooks
//===----------------------------------------------------------------------===//

static bool MayFoldLoad(SDValue Op) {
  return Op.hasOneUse() && ISD::isNormalLoad(Op.getNode());
}

static bool MayFoldIntoStore(SDValue Op) {
  return Op.hasOneUse() && ISD::isNormalStore(*Op.getNode()->use_begin());
}

static bool isTargetShuffle(unsigned Opcode) {
  switch(Opcode) {
  default: return false;
  case X86ISD::PSHUFD:
  case X86ISD::PSHUFHW:
  case X86ISD::PSHUFLW:
  case X86ISD::SHUFP:
  case X86ISD::PALIGN:
  case X86ISD::MOVLHPS:
  case X86ISD::MOVLHPD:
  case X86ISD::MOVHLPS:
  case X86ISD::MOVLPS:
  case X86ISD::MOVLPD:
  case X86ISD::MOVSHDUP:
  case X86ISD::MOVSLDUP:
  case X86ISD::MOVDDUP:
  case X86ISD::MOVSS:
  case X86ISD::MOVSD:
  case X86ISD::UNPCKL:
  case X86ISD::UNPCKH:
  case X86ISD::VPERMILP:
  case X86ISD::VPERM2X128:
  case X86ISD::VPERMI:
    return true;
  }
}

static SDValue getTargetShuffleNode(unsigned Opc, DebugLoc dl, EVT VT,
                                    SDValue V1, SelectionDAG &DAG) {
  switch(Opc) {
  default: llvm_unreachable("Unknown x86 shuffle node");
  case X86ISD::MOVSHDUP:
  case X86ISD::MOVSLDUP:
  case X86ISD::MOVDDUP:
    return DAG.getNode(Opc, dl, VT, V1);
  }
}

static SDValue getTargetShuffleNode(unsigned Opc, DebugLoc dl, EVT VT,
                                    SDValue V1, unsigned TargetMask,
                                    SelectionDAG &DAG) {
  switch(Opc) {
  default: llvm_unreachable("Unknown x86 shuffle node");
  case X86ISD::PSHUFD:
  case X86ISD::PSHUFHW:
  case X86ISD::PSHUFLW:
  case X86ISD::VPERMILP:
  case X86ISD::VPERMI:
    return DAG.getNode(Opc, dl, VT, V1, DAG.getConstant(TargetMask, MVT::i8));
  }
}

static SDValue getTargetShuffleNode(unsigned Opc, DebugLoc dl, EVT VT,
                                    SDValue V1, SDValue V2, unsigned TargetMask,
                                    SelectionDAG &DAG) {
  switch(Opc) {
  default: llvm_unreachable("Unknown x86 shuffle node");
  case X86ISD::PALIGN:
  case X86ISD::SHUFP:
  case X86ISD::VPERM2X128:
    return DAG.getNode(Opc, dl, VT, V1, V2,
                       DAG.getConstant(TargetMask, MVT::i8));
  }
}

static SDValue getTargetShuffleNode(unsigned Opc, DebugLoc dl, EVT VT,
                                    SDValue V1, SDValue V2, SelectionDAG &DAG) {
  switch(Opc) {
  default: llvm_unreachable("Unknown x86 shuffle node");
  case X86ISD::MOVLHPS:
  case X86ISD::MOVLHPD:
  case X86ISD::MOVHLPS:
  case X86ISD::MOVLPS:
  case X86ISD::MOVLPD:
  case X86ISD::MOVSS:
  case X86ISD::MOVSD:
  case X86ISD::UNPCKL:
  case X86ISD::UNPCKH:
    return DAG.getNode(Opc, dl, VT, V1, V2);
  }
}

SDValue X86TargetLowering::getReturnAddressFrameIndex(SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
  int ReturnAddrIndex = FuncInfo->getRAIndex();

  if (ReturnAddrIndex == 0) {
    // Set up a frame object for the return address.
    unsigned SlotSize = RegInfo->getSlotSize();
    ReturnAddrIndex = MF.getFrameInfo()->CreateFixedObject(SlotSize, -SlotSize,
                                                           false);
    FuncInfo->setRAIndex(ReturnAddrIndex);
  }

  return DAG.getFrameIndex(ReturnAddrIndex, getPointerTy());
}


bool X86::isOffsetSuitableForCodeModel(int64_t Offset, CodeModel::Model M,
                                       bool hasSymbolicDisplacement) {
  // Offset should fit into 32 bit immediate field.
  if (!isInt<32>(Offset))
    return false;

  // If we don't have a symbolic displacement - we don't have any extra
  // restrictions.
  if (!hasSymbolicDisplacement)
    return true;

  // FIXME: Some tweaks might be needed for medium code model.
  if (M != CodeModel::Small && M != CodeModel::Kernel)
    return false;

  // For small code model we assume that latest object is 16MB before end of 31
  // bits boundary. We may also accept pretty large negative constants knowing
  // that all objects are in the positive half of address space.
  if (M == CodeModel::Small && Offset < 16*1024*1024)
    return true;

  // For kernel code model we know that all object resist in the negative half
  // of 32bits address space. We may not accept negative offsets, since they may
  // be just off and we may accept pretty large positive ones.
  if (M == CodeModel::Kernel && Offset > 0)
    return true;

  return false;
}

/// isCalleePop - Determines whether the callee is required to pop its
/// own arguments. Callee pop is necessary to support tail calls.
bool X86::isCalleePop(CallingConv::ID CallingConv,
                      bool is64Bit, bool IsVarArg, bool TailCallOpt) {
  if (IsVarArg)
    return false;

  switch (CallingConv) {
  default:
    return false;
  case CallingConv::X86_StdCall:
    return !is64Bit;
  case CallingConv::X86_FastCall:
    return !is64Bit;
  case CallingConv::X86_ThisCall:
    return !is64Bit;
  case CallingConv::Fast:
    return TailCallOpt;
  case CallingConv::GHC:
    return TailCallOpt;
  }
}

/// TranslateX86CC - do a one to one translation of a ISD::CondCode to the X86
/// specific condition code, returning the condition code and the LHS/RHS of the
/// comparison to make.
static unsigned TranslateX86CC(ISD::CondCode SetCCOpcode, bool isFP,
                               SDValue &LHS, SDValue &RHS, SelectionDAG &DAG) {
  if (!isFP) {
    if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS)) {
      if (SetCCOpcode == ISD::SETGT && RHSC->isAllOnesValue()) {
        // X > -1   -> X == 0, jump !sign.
        RHS = DAG.getConstant(0, RHS.getValueType());
        return X86::COND_NS;
      }
      if (SetCCOpcode == ISD::SETLT && RHSC->isNullValue()) {
        // X < 0   -> X == 0, jump on sign.
        return X86::COND_S;
      }
      if (SetCCOpcode == ISD::SETLT && RHSC->getZExtValue() == 1) {
        // X < 1   -> X <= 0
        RHS = DAG.getConstant(0, RHS.getValueType());
        return X86::COND_LE;
      }
    }

    switch (SetCCOpcode) {
    default: llvm_unreachable("Invalid integer condition!");
    case ISD::SETEQ:  return X86::COND_E;
    case ISD::SETGT:  return X86::COND_G;
    case ISD::SETGE:  return X86::COND_GE;
    case ISD::SETLT:  return X86::COND_L;
    case ISD::SETLE:  return X86::COND_LE;
    case ISD::SETNE:  return X86::COND_NE;
    case ISD::SETULT: return X86::COND_B;
    case ISD::SETUGT: return X86::COND_A;
    case ISD::SETULE: return X86::COND_BE;
    case ISD::SETUGE: return X86::COND_AE;
    }
  }

  // First determine if it is required or is profitable to flip the operands.

  // If LHS is a foldable load, but RHS is not, flip the condition.
  if (ISD::isNON_EXTLoad(LHS.getNode()) &&
      !ISD::isNON_EXTLoad(RHS.getNode())) {
    SetCCOpcode = getSetCCSwappedOperands(SetCCOpcode);
    std::swap(LHS, RHS);
  }

  switch (SetCCOpcode) {
  default: break;
  case ISD::SETOLT:
  case ISD::SETOLE:
  case ISD::SETUGT:
  case ISD::SETUGE:
    std::swap(LHS, RHS);
    break;
  }

  // On a floating point condition, the flags are set as follows:
  // ZF  PF  CF   op
  //  0 | 0 | 0 | X > Y
  //  0 | 0 | 1 | X < Y
  //  1 | 0 | 0 | X == Y
  //  1 | 1 | 1 | unordered
  switch (SetCCOpcode) {
  default: llvm_unreachable("Condcode should be pre-legalized away");
  case ISD::SETUEQ:
  case ISD::SETEQ:   return X86::COND_E;
  case ISD::SETOLT:              // flipped
  case ISD::SETOGT:
  case ISD::SETGT:   return X86::COND_A;
  case ISD::SETOLE:              // flipped
  case ISD::SETOGE:
  case ISD::SETGE:   return X86::COND_AE;
  case ISD::SETUGT:              // flipped
  case ISD::SETULT:
  case ISD::SETLT:   return X86::COND_B;
  case ISD::SETUGE:              // flipped
  case ISD::SETULE:
  case ISD::SETLE:   return X86::COND_BE;
  case ISD::SETONE:
  case ISD::SETNE:   return X86::COND_NE;
  case ISD::SETUO:   return X86::COND_P;
  case ISD::SETO:    return X86::COND_NP;
  case ISD::SETOEQ:
  case ISD::SETUNE:  return X86::COND_INVALID;
  }
}

/// hasFPCMov - is there a floating point cmov for the specific X86 condition
/// code. Current x86 isa includes the following FP cmov instructions:
/// fcmovb, fcomvbe, fcomve, fcmovu, fcmovae, fcmova, fcmovne, fcmovnu.
static bool hasFPCMov(unsigned X86CC) {
  switch (X86CC) {
  default:
    return false;
  case X86::COND_B:
  case X86::COND_BE:
  case X86::COND_E:
  case X86::COND_P:
  case X86::COND_A:
  case X86::COND_AE:
  case X86::COND_NE:
  case X86::COND_NP:
    return true;
  }
}

/// isFPImmLegal - Returns true if the target can instruction select the
/// specified FP immediate natively. If false, the legalizer will
/// materialize the FP immediate as a load from a constant pool.
bool X86TargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  for (unsigned i = 0, e = LegalFPImmediates.size(); i != e; ++i) {
    if (Imm.bitwiseIsEqual(LegalFPImmediates[i]))
      return true;
  }
  return false;
}

/// isUndefOrInRange - Return true if Val is undef or if its value falls within
/// the specified range (L, H].
static bool isUndefOrInRange(int Val, int Low, int Hi) {
  return (Val < 0) || (Val >= Low && Val < Hi);
}

/// isUndefOrEqual - Val is either less than zero (undef) or equal to the
/// specified value.
static bool isUndefOrEqual(int Val, int CmpVal) {
  if (Val < 0 || Val == CmpVal)
    return true;
  return false;
}

/// isSequentialOrUndefInRange - Return true if every element in Mask, beginning
/// from position Pos and ending in Pos+Size, falls within the specified
/// sequential range (L, L+Pos]. or is undef.
static bool isSequentialOrUndefInRange(ArrayRef<int> Mask,
                                       unsigned Pos, unsigned Size, int Low) {
  for (unsigned i = Pos, e = Pos+Size; i != e; ++i, ++Low)
    if (!isUndefOrEqual(Mask[i], Low))
      return false;
  return true;
}

/// isPSHUFDMask - Return true if the node specifies a shuffle of elements that
/// is suitable for input to PSHUFD or PSHUFW.  That is, it doesn't reference
/// the second operand.
static bool isPSHUFDMask(ArrayRef<int> Mask, EVT VT) {
  if (VT == MVT::v4f32 || VT == MVT::v4i32 )
    return (Mask[0] < 4 && Mask[1] < 4 && Mask[2] < 4 && Mask[3] < 4);
  if (VT == MVT::v2f64 || VT == MVT::v2i64)
    return (Mask[0] < 2 && Mask[1] < 2);
  return false;
}

/// isPSHUFHWMask - Return true if the node specifies a shuffle of elements that
/// is suitable for input to PSHUFHW.
static bool isPSHUFHWMask(ArrayRef<int> Mask, EVT VT, bool HasAVX2) {
  if (VT != MVT::v8i16 && (!HasAVX2 || VT != MVT::v16i16))
    return false;

  // Lower quadword copied in order or undef.
  if (!isSequentialOrUndefInRange(Mask, 0, 4, 0))
    return false;

  // Upper quadword shuffled.
  for (unsigned i = 4; i != 8; ++i)
    if (!isUndefOrInRange(Mask[i], 4, 8))
      return false;

  if (VT == MVT::v16i16) {
    // Lower quadword copied in order or undef.
    if (!isSequentialOrUndefInRange(Mask, 8, 4, 8))
      return false;

    // Upper quadword shuffled.
    for (unsigned i = 12; i != 16; ++i)
      if (!isUndefOrInRange(Mask[i], 12, 16))
        return false;
  }

  return true;
}

/// isPSHUFLWMask - Return true if the node specifies a shuffle of elements that
/// is suitable for input to PSHUFLW.
static bool isPSHUFLWMask(ArrayRef<int> Mask, EVT VT, bool HasAVX2) {
  if (VT != MVT::v8i16 && (!HasAVX2 || VT != MVT::v16i16))
    return false;

  // Upper quadword copied in order.
  if (!isSequentialOrUndefInRange(Mask, 4, 4, 4))
    return false;

  // Lower quadword shuffled.
  for (unsigned i = 0; i != 4; ++i)
    if (!isUndefOrInRange(Mask[i], 0, 4))
      return false;

  if (VT == MVT::v16i16) {
    // Upper quadword copied in order.
    if (!isSequentialOrUndefInRange(Mask, 12, 4, 12))
      return false;

    // Lower quadword shuffled.
    for (unsigned i = 8; i != 12; ++i)
      if (!isUndefOrInRange(Mask[i], 8, 12))
        return false;
  }

  return true;
}

/// isPALIGNRMask - Return true if the node specifies a shuffle of elements that
/// is suitable for input to PALIGNR.
static bool isPALIGNRMask(ArrayRef<int> Mask, EVT VT,
                          const X86Subtarget *Subtarget) {
  if ((VT.getSizeInBits() == 128 && !Subtarget->hasSSSE3()) ||
      (VT.getSizeInBits() == 256 && !Subtarget->hasAVX2()))
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElts = NumElts/NumLanes;

  // Do not handle 64-bit element shuffles with palignr.
  if (NumLaneElts == 2)
    return false;

  for (unsigned l = 0; l != NumElts; l+=NumLaneElts) {
    unsigned i;
    for (i = 0; i != NumLaneElts; ++i) {
      if (Mask[i+l] >= 0)
        break;
    }

    // Lane is all undef, go to next lane
    if (i == NumLaneElts)
      continue;

    int Start = Mask[i+l];

    // Make sure its in this lane in one of the sources
    if (!isUndefOrInRange(Start, l, l+NumLaneElts) &&
        !isUndefOrInRange(Start, l+NumElts, l+NumElts+NumLaneElts))
      return false;

    // If not lane 0, then we must match lane 0
    if (l != 0 && Mask[i] >= 0 && !isUndefOrEqual(Start, Mask[i]+l))
      return false;

    // Correct second source to be contiguous with first source
    if (Start >= (int)NumElts)
      Start -= NumElts - NumLaneElts;

    // Make sure we're shifting in the right direction.
    if (Start <= (int)(i+l))
      return false;

    Start -= i;

    // Check the rest of the elements to see if they are consecutive.
    for (++i; i != NumLaneElts; ++i) {
      int Idx = Mask[i+l];

      // Make sure its in this lane
      if (!isUndefOrInRange(Idx, l, l+NumLaneElts) &&
          !isUndefOrInRange(Idx, l+NumElts, l+NumElts+NumLaneElts))
        return false;

      // If not lane 0, then we must match lane 0
      if (l != 0 && Mask[i] >= 0 && !isUndefOrEqual(Idx, Mask[i]+l))
        return false;

      if (Idx >= (int)NumElts)
        Idx -= NumElts - NumLaneElts;

      if (!isUndefOrEqual(Idx, Start+i))
        return false;

    }
  }

  return true;
}

/// CommuteVectorShuffleMask - Change values in a shuffle permute mask assuming
/// the two vector operands have swapped position.
static void CommuteVectorShuffleMask(SmallVectorImpl<int> &Mask,
                                     unsigned NumElems) {
  for (unsigned i = 0; i != NumElems; ++i) {
    int idx = Mask[i];
    if (idx < 0)
      continue;
    else if (idx < (int)NumElems)
      Mask[i] = idx + NumElems;
    else
      Mask[i] = idx - NumElems;
  }
}

/// isSHUFPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to 128/256-bit
/// SHUFPS and SHUFPD. If Commuted is true, then it checks for sources to be
/// reverse of what x86 shuffles want.
static bool isSHUFPMask(ArrayRef<int> Mask, EVT VT, bool HasAVX,
                        bool Commuted = false) {
  if (!HasAVX && VT.getSizeInBits() == 256)
    return false;

  unsigned NumElems = VT.getVectorNumElements();
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElems = NumElems/NumLanes;

  if (NumLaneElems != 2 && NumLaneElems != 4)
    return false;

  // VSHUFPSY divides the resulting vector into 4 chunks.
  // The sources are also splitted into 4 chunks, and each destination
  // chunk must come from a different source chunk.
  //
  //  SRC1 =>   X7    X6    X5    X4    X3    X2    X1    X0
  //  SRC2 =>   Y7    Y6    Y5    Y4    Y3    Y2    Y1    Y9
  //
  //  DST  =>  Y7..Y4,   Y7..Y4,   X7..X4,   X7..X4,
  //           Y3..Y0,   Y3..Y0,   X3..X0,   X3..X0
  //
  // VSHUFPDY divides the resulting vector into 4 chunks.
  // The sources are also splitted into 4 chunks, and each destination
  // chunk must come from a different source chunk.
  //
  //  SRC1 =>      X3       X2       X1       X0
  //  SRC2 =>      Y3       Y2       Y1       Y0
  //
  //  DST  =>  Y3..Y2,  X3..X2,  Y1..Y0,  X1..X0
  //
  unsigned HalfLaneElems = NumLaneElems/2;
  for (unsigned l = 0; l != NumElems; l += NumLaneElems) {
    for (unsigned i = 0; i != NumLaneElems; ++i) {
      int Idx = Mask[i+l];
      unsigned RngStart = l + ((Commuted == (i<HalfLaneElems)) ? NumElems : 0);
      if (!isUndefOrInRange(Idx, RngStart, RngStart+NumLaneElems))
        return false;
      // For VSHUFPSY, the mask of the second half must be the same as the
      // first but with the appropriate offsets. This works in the same way as
      // VPERMILPS works with masks.
      if (NumElems != 8 || l == 0 || Mask[i] < 0)
        continue;
      if (!isUndefOrEqual(Idx, Mask[i]+l))
        return false;
    }
  }

  return true;
}

/// isMOVHLPSMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVHLPS.
static bool isMOVHLPSMask(ArrayRef<int> Mask, EVT VT) {
  if (!VT.is128BitVector())
    return false;

  unsigned NumElems = VT.getVectorNumElements();

  if (NumElems != 4)
    return false;

  // Expect bit0 == 6, bit1 == 7, bit2 == 2, bit3 == 3
  return isUndefOrEqual(Mask[0], 6) &&
         isUndefOrEqual(Mask[1], 7) &&
         isUndefOrEqual(Mask[2], 2) &&
         isUndefOrEqual(Mask[3], 3);
}

/// isMOVHLPS_v_undef_Mask - Special case of isMOVHLPSMask for canonical form
/// of vector_shuffle v, v, <2, 3, 2, 3>, i.e. vector_shuffle v, undef,
/// <2, 3, 2, 3>
static bool isMOVHLPS_v_undef_Mask(ArrayRef<int> Mask, EVT VT) {
  if (!VT.is128BitVector())
    return false;

  unsigned NumElems = VT.getVectorNumElements();

  if (NumElems != 4)
    return false;

  return isUndefOrEqual(Mask[0], 2) &&
         isUndefOrEqual(Mask[1], 3) &&
         isUndefOrEqual(Mask[2], 2) &&
         isUndefOrEqual(Mask[3], 3);
}

/// isMOVLPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVLP{S|D}.
static bool isMOVLPMask(ArrayRef<int> Mask, EVT VT) {
  if (!VT.is128BitVector())
    return false;

  unsigned NumElems = VT.getVectorNumElements();

  if (NumElems != 2 && NumElems != 4)
    return false;

  for (unsigned i = 0, e = NumElems/2; i != e; ++i)
    if (!isUndefOrEqual(Mask[i], i + NumElems))
      return false;

  for (unsigned i = NumElems/2, e = NumElems; i != e; ++i)
    if (!isUndefOrEqual(Mask[i], i))
      return false;

  return true;
}

/// isMOVLHPSMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVLHPS.
static bool isMOVLHPSMask(ArrayRef<int> Mask, EVT VT) {
  if (!VT.is128BitVector())
    return false;

  unsigned NumElems = VT.getVectorNumElements();

  if (NumElems != 2 && NumElems != 4)
    return false;

  for (unsigned i = 0, e = NumElems/2; i != e; ++i)
    if (!isUndefOrEqual(Mask[i], i))
      return false;

  for (unsigned i = 0, e = NumElems/2; i != e; ++i)
    if (!isUndefOrEqual(Mask[i + e], i + NumElems))
      return false;

  return true;
}

//
// Some special combinations that can be optimized.
//
static
SDValue Compact8x32ShuffleNode(ShuffleVectorSDNode *SVOp,
                               SelectionDAG &DAG) {
  EVT VT = SVOp->getValueType(0);
  DebugLoc dl = SVOp->getDebugLoc();

  if (VT != MVT::v8i32 && VT != MVT::v8f32)
    return SDValue();

  ArrayRef<int> Mask = SVOp->getMask();

  // These are the special masks that may be optimized.
  static const int MaskToOptimizeEven[] = {0, 8, 2, 10, 4, 12, 6, 14};
  static const int MaskToOptimizeOdd[]  = {1, 9, 3, 11, 5, 13, 7, 15};
  bool MatchEvenMask = true;
  bool MatchOddMask  = true;
  for (int i=0; i<8; ++i) {
    if (!isUndefOrEqual(Mask[i], MaskToOptimizeEven[i]))
      MatchEvenMask = false;
    if (!isUndefOrEqual(Mask[i], MaskToOptimizeOdd[i]))
      MatchOddMask = false;
  }

  if (!MatchEvenMask && !MatchOddMask)
    return SDValue();

  SDValue UndefNode = DAG.getNode(ISD::UNDEF, dl, VT);

  SDValue Op0 = SVOp->getOperand(0);
  SDValue Op1 = SVOp->getOperand(1);

  if (MatchEvenMask) {
    // Shift the second operand right to 32 bits.
    static const int ShiftRightMask[] = {-1, 0, -1, 2, -1, 4, -1, 6 };
    Op1 = DAG.getVectorShuffle(VT, dl, Op1, UndefNode, ShiftRightMask);
  } else {
    // Shift the first operand left to 32 bits.
    static const int ShiftLeftMask[] = {1, -1, 3, -1, 5, -1, 7, -1 };
    Op0 = DAG.getVectorShuffle(VT, dl, Op0, UndefNode, ShiftLeftMask);
  }
  static const int BlendMask[] = {0, 9, 2, 11, 4, 13, 6, 15};
  return DAG.getVectorShuffle(VT, dl, Op0, Op1, BlendMask);
}

/// isUNPCKLMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to UNPCKL.
static bool isUNPCKLMask(ArrayRef<int> Mask, EVT VT,
                         bool HasAVX2, bool V2IsSplat = false) {
  unsigned NumElts = VT.getVectorNumElements();

  assert((VT.is128BitVector() || VT.is256BitVector()) &&
         "Unsupported vector type for unpckh");

  if (VT.getSizeInBits() == 256 && NumElts != 4 && NumElts != 8 &&
      (!HasAVX2 || (NumElts != 16 && NumElts != 32)))
    return false;

  // Handle 128 and 256-bit vector lengths. AVX defines UNPCK* to operate
  // independently on 128-bit lanes.
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElts = NumElts/NumLanes;

  for (unsigned l = 0; l != NumLanes; ++l) {
    for (unsigned i = l*NumLaneElts, j = l*NumLaneElts;
         i != (l+1)*NumLaneElts;
         i += 2, ++j) {
      int BitI  = Mask[i];
      int BitI1 = Mask[i+1];
      if (!isUndefOrEqual(BitI, j))
        return false;
      if (V2IsSplat) {
        if (!isUndefOrEqual(BitI1, NumElts))
          return false;
      } else {
        if (!isUndefOrEqual(BitI1, j + NumElts))
          return false;
      }
    }
  }

  return true;
}

/// isUNPCKHMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to UNPCKH.
static bool isUNPCKHMask(ArrayRef<int> Mask, EVT VT,
                         bool HasAVX2, bool V2IsSplat = false) {
  unsigned NumElts = VT.getVectorNumElements();

  assert((VT.is128BitVector() || VT.is256BitVector()) &&
         "Unsupported vector type for unpckh");

  if (VT.getSizeInBits() == 256 && NumElts != 4 && NumElts != 8 &&
      (!HasAVX2 || (NumElts != 16 && NumElts != 32)))
    return false;

  // Handle 128 and 256-bit vector lengths. AVX defines UNPCK* to operate
  // independently on 128-bit lanes.
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElts = NumElts/NumLanes;

  for (unsigned l = 0; l != NumLanes; ++l) {
    for (unsigned i = l*NumLaneElts, j = (l*NumLaneElts)+NumLaneElts/2;
         i != (l+1)*NumLaneElts; i += 2, ++j) {
      int BitI  = Mask[i];
      int BitI1 = Mask[i+1];
      if (!isUndefOrEqual(BitI, j))
        return false;
      if (V2IsSplat) {
        if (isUndefOrEqual(BitI1, NumElts))
          return false;
      } else {
        if (!isUndefOrEqual(BitI1, j+NumElts))
          return false;
      }
    }
  }
  return true;
}

/// isUNPCKL_v_undef_Mask - Special case of isUNPCKLMask for canonical form
/// of vector_shuffle v, v, <0, 4, 1, 5>, i.e. vector_shuffle v, undef,
/// <0, 0, 1, 1>
static bool isUNPCKL_v_undef_Mask(ArrayRef<int> Mask, EVT VT,
                                  bool HasAVX2) {
  unsigned NumElts = VT.getVectorNumElements();

  assert((VT.is128BitVector() || VT.is256BitVector()) &&
         "Unsupported vector type for unpckh");

  if (VT.getSizeInBits() == 256 && NumElts != 4 && NumElts != 8 &&
      (!HasAVX2 || (NumElts != 16 && NumElts != 32)))
    return false;

  // For 256-bit i64/f64, use MOVDDUPY instead, so reject the matching pattern
  // FIXME: Need a better way to get rid of this, there's no latency difference
  // between UNPCKLPD and MOVDDUP, the later should always be checked first and
  // the former later. We should also remove the "_undef" special mask.
  if (NumElts == 4 && VT.getSizeInBits() == 256)
    return false;

  // Handle 128 and 256-bit vector lengths. AVX defines UNPCK* to operate
  // independently on 128-bit lanes.
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElts = NumElts/NumLanes;

  for (unsigned l = 0; l != NumLanes; ++l) {
    for (unsigned i = l*NumLaneElts, j = l*NumLaneElts;
         i != (l+1)*NumLaneElts;
         i += 2, ++j) {
      int BitI  = Mask[i];
      int BitI1 = Mask[i+1];

      if (!isUndefOrEqual(BitI, j))
        return false;
      if (!isUndefOrEqual(BitI1, j))
        return false;
    }
  }

  return true;
}

/// isUNPCKH_v_undef_Mask - Special case of isUNPCKHMask for canonical form
/// of vector_shuffle v, v, <2, 6, 3, 7>, i.e. vector_shuffle v, undef,
/// <2, 2, 3, 3>
static bool isUNPCKH_v_undef_Mask(ArrayRef<int> Mask, EVT VT, bool HasAVX2) {
  unsigned NumElts = VT.getVectorNumElements();

  assert((VT.is128BitVector() || VT.is256BitVector()) &&
         "Unsupported vector type for unpckh");

  if (VT.getSizeInBits() == 256 && NumElts != 4 && NumElts != 8 &&
      (!HasAVX2 || (NumElts != 16 && NumElts != 32)))
    return false;

  // Handle 128 and 256-bit vector lengths. AVX defines UNPCK* to operate
  // independently on 128-bit lanes.
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElts = NumElts/NumLanes;

  for (unsigned l = 0; l != NumLanes; ++l) {
    for (unsigned i = l*NumLaneElts, j = (l*NumLaneElts)+NumLaneElts/2;
         i != (l+1)*NumLaneElts; i += 2, ++j) {
      int BitI  = Mask[i];
      int BitI1 = Mask[i+1];
      if (!isUndefOrEqual(BitI, j))
        return false;
      if (!isUndefOrEqual(BitI1, j))
        return false;
    }
  }
  return true;
}

/// isMOVLMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSS,
/// MOVSD, and MOVD, i.e. setting the lowest element.
static bool isMOVLMask(ArrayRef<int> Mask, EVT VT) {
  if (VT.getVectorElementType().getSizeInBits() < 32)
    return false;
  if (!VT.is128BitVector())
    return false;

  unsigned NumElts = VT.getVectorNumElements();

  if (!isUndefOrEqual(Mask[0], NumElts))
    return false;

  for (unsigned i = 1; i != NumElts; ++i)
    if (!isUndefOrEqual(Mask[i], i))
      return false;

  return true;
}

/// isVPERM2X128Mask - Match 256-bit shuffles where the elements are considered
/// as permutations between 128-bit chunks or halves. As an example: this
/// shuffle bellow:
///   vector_shuffle <4, 5, 6, 7, 12, 13, 14, 15>
/// The first half comes from the second half of V1 and the second half from the
/// the second half of V2.
static bool isVPERM2X128Mask(ArrayRef<int> Mask, EVT VT, bool HasAVX) {
  if (!HasAVX || !VT.is256BitVector())
    return false;

  // The shuffle result is divided into half A and half B. In total the two
  // sources have 4 halves, namely: C, D, E, F. The final values of A and
  // B must come from C, D, E or F.
  unsigned HalfSize = VT.getVectorNumElements()/2;
  bool MatchA = false, MatchB = false;

  // Check if A comes from one of C, D, E, F.
  for (unsigned Half = 0; Half != 4; ++Half) {
    if (isSequentialOrUndefInRange(Mask, 0, HalfSize, Half*HalfSize)) {
      MatchA = true;
      break;
    }
  }

  // Check if B comes from one of C, D, E, F.
  for (unsigned Half = 0; Half != 4; ++Half) {
    if (isSequentialOrUndefInRange(Mask, HalfSize, HalfSize, Half*HalfSize)) {
      MatchB = true;
      break;
    }
  }

  return MatchA && MatchB;
}

/// getShuffleVPERM2X128Immediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_MASK mask with VPERM2F128/VPERM2I128 instructions.
static unsigned getShuffleVPERM2X128Immediate(ShuffleVectorSDNode *SVOp) {
  EVT VT = SVOp->getValueType(0);

  unsigned HalfSize = VT.getVectorNumElements()/2;

  unsigned FstHalf = 0, SndHalf = 0;
  for (unsigned i = 0; i < HalfSize; ++i) {
    if (SVOp->getMaskElt(i) > 0) {
      FstHalf = SVOp->getMaskElt(i)/HalfSize;
      break;
    }
  }
  for (unsigned i = HalfSize; i < HalfSize*2; ++i) {
    if (SVOp->getMaskElt(i) > 0) {
      SndHalf = SVOp->getMaskElt(i)/HalfSize;
      break;
    }
  }

  return (FstHalf | (SndHalf << 4));
}

/// isVPERMILPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to VPERMILPD*.
/// Note that VPERMIL mask matching is different depending whether theunderlying
/// type is 32 or 64. In the VPERMILPS the high half of the mask should point
/// to the same elements of the low, but to the higher half of the source.
/// In VPERMILPD the two lanes could be shuffled independently of each other
/// with the same restriction that lanes can't be crossed. Also handles PSHUFDY.
static bool isVPERMILPMask(ArrayRef<int> Mask, EVT VT, bool HasAVX) {
  if (!HasAVX)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  // Only match 256-bit with 32/64-bit types
  if (VT.getSizeInBits() != 256 || (NumElts != 4 && NumElts != 8))
    return false;

  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned LaneSize = NumElts/NumLanes;
  for (unsigned l = 0; l != NumElts; l += LaneSize) {
    for (unsigned i = 0; i != LaneSize; ++i) {
      if (!isUndefOrInRange(Mask[i+l], l, l+LaneSize))
        return false;
      if (NumElts != 8 || l == 0)
        continue;
      // VPERMILPS handling
      if (Mask[i] < 0)
        continue;
      if (!isUndefOrEqual(Mask[i+l], Mask[i]+l))
        return false;
    }
  }

  return true;
}

/// isCommutedMOVLMask - Returns true if the shuffle mask is except the reverse
/// of what x86 movss want. X86 movs requires the lowest  element to be lowest
/// element of vector 2 and the other elements to come from vector 1 in order.
static bool isCommutedMOVLMask(ArrayRef<int> Mask, EVT VT,
                               bool V2IsSplat = false, bool V2IsUndef = false) {
  if (!VT.is128BitVector())
    return false;

  unsigned NumOps = VT.getVectorNumElements();
  if (NumOps != 2 && NumOps != 4 && NumOps != 8 && NumOps != 16)
    return false;

  if (!isUndefOrEqual(Mask[0], 0))
    return false;

  for (unsigned i = 1; i != NumOps; ++i)
    if (!(isUndefOrEqual(Mask[i], i+NumOps) ||
          (V2IsUndef && isUndefOrInRange(Mask[i], NumOps, NumOps*2)) ||
          (V2IsSplat && isUndefOrEqual(Mask[i], NumOps))))
      return false;

  return true;
}

/// isMOVSHDUPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSHDUP.
/// Masks to match: <1, 1, 3, 3> or <1, 1, 3, 3, 5, 5, 7, 7>
static bool isMOVSHDUPMask(ArrayRef<int> Mask, EVT VT,
                           const X86Subtarget *Subtarget) {
  if (!Subtarget->hasSSE3())
    return false;

  unsigned NumElems = VT.getVectorNumElements();

  if ((VT.getSizeInBits() == 128 && NumElems != 4) ||
      (VT.getSizeInBits() == 256 && NumElems != 8))
    return false;

  // "i+1" is the value the indexed mask element must have
  for (unsigned i = 0; i != NumElems; i += 2)
    if (!isUndefOrEqual(Mask[i], i+1) ||
        !isUndefOrEqual(Mask[i+1], i+1))
      return false;

  return true;
}

/// isMOVSLDUPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSLDUP.
/// Masks to match: <0, 0, 2, 2> or <0, 0, 2, 2, 4, 4, 6, 6>
static bool isMOVSLDUPMask(ArrayRef<int> Mask, EVT VT,
                           const X86Subtarget *Subtarget) {
  if (!Subtarget->hasSSE3())
    return false;

  unsigned NumElems = VT.getVectorNumElements();

  if ((VT.getSizeInBits() == 128 && NumElems != 4) ||
      (VT.getSizeInBits() == 256 && NumElems != 8))
    return false;

  // "i" is the value the indexed mask element must have
  for (unsigned i = 0; i != NumElems; i += 2)
    if (!isUndefOrEqual(Mask[i], i) ||
        !isUndefOrEqual(Mask[i+1], i))
      return false;

  return true;
}

/// isMOVDDUPYMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to 256-bit
/// version of MOVDDUP.
static bool isMOVDDUPYMask(ArrayRef<int> Mask, EVT VT, bool HasAVX) {
  if (!HasAVX || !VT.is256BitVector())
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  if (NumElts != 4)
    return false;

  for (unsigned i = 0; i != NumElts/2; ++i)
    if (!isUndefOrEqual(Mask[i], 0))
      return false;
  for (unsigned i = NumElts/2; i != NumElts; ++i)
    if (!isUndefOrEqual(Mask[i], NumElts/2))
      return false;
  return true;
}

/// isMOVDDUPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to 128-bit
/// version of MOVDDUP.
static bool isMOVDDUPMask(ArrayRef<int> Mask, EVT VT) {
  if (!VT.is128BitVector())
    return false;

  unsigned e = VT.getVectorNumElements() / 2;
  for (unsigned i = 0; i != e; ++i)
    if (!isUndefOrEqual(Mask[i], i))
      return false;
  for (unsigned i = 0; i != e; ++i)
    if (!isUndefOrEqual(Mask[e+i], i))
      return false;
  return true;
}

/// isVEXTRACTF128Index - Return true if the specified
/// EXTRACT_SUBVECTOR operand specifies a vector extract that is
/// suitable for input to VEXTRACTF128.
bool X86::isVEXTRACTF128Index(SDNode *N) {
  if (!isa<ConstantSDNode>(N->getOperand(1).getNode()))
    return false;

  // The index should be aligned on a 128-bit boundary.
  uint64_t Index =
    cast<ConstantSDNode>(N->getOperand(1).getNode())->getZExtValue();

  unsigned VL = N->getValueType(0).getVectorNumElements();
  unsigned VBits = N->getValueType(0).getSizeInBits();
  unsigned ElSize = VBits / VL;
  bool Result = (Index * ElSize) % 128 == 0;

  return Result;
}

/// isVINSERTF128Index - Return true if the specified INSERT_SUBVECTOR
/// operand specifies a subvector insert that is suitable for input to
/// VINSERTF128.
bool X86::isVINSERTF128Index(SDNode *N) {
  if (!isa<ConstantSDNode>(N->getOperand(2).getNode()))
    return false;

  // The index should be aligned on a 128-bit boundary.
  uint64_t Index =
    cast<ConstantSDNode>(N->getOperand(2).getNode())->getZExtValue();

  unsigned VL = N->getValueType(0).getVectorNumElements();
  unsigned VBits = N->getValueType(0).getSizeInBits();
  unsigned ElSize = VBits / VL;
  bool Result = (Index * ElSize) % 128 == 0;

  return Result;
}

/// getShuffleSHUFImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with PSHUF* and SHUFP* instructions.
/// Handles 128-bit and 256-bit.
static unsigned getShuffleSHUFImmediate(ShuffleVectorSDNode *N) {
  EVT VT = N->getValueType(0);

  assert((VT.is128BitVector() || VT.is256BitVector()) &&
         "Unsupported vector type for PSHUF/SHUFP");

  // Handle 128 and 256-bit vector lengths. AVX defines PSHUF/SHUFP to operate
  // independently on 128-bit lanes.
  unsigned NumElts = VT.getVectorNumElements();
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElts = NumElts/NumLanes;

  assert((NumLaneElts == 2 || NumLaneElts == 4) &&
         "Only supports 2 or 4 elements per lane");

  unsigned Shift = (NumLaneElts == 4) ? 1 : 0;
  unsigned Mask = 0;
  for (unsigned i = 0; i != NumElts; ++i) {
    int Elt = N->getMaskElt(i);
    if (Elt < 0) continue;
    Elt &= NumLaneElts - 1;
    unsigned ShAmt = (i << Shift) % 8;
    Mask |= Elt << ShAmt;
  }

  return Mask;
}

/// getShufflePSHUFHWImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with the PSHUFHW instruction.
static unsigned getShufflePSHUFHWImmediate(ShuffleVectorSDNode *N) {
  EVT VT = N->getValueType(0);

  assert((VT == MVT::v8i16 || VT == MVT::v16i16) &&
         "Unsupported vector type for PSHUFHW");

  unsigned NumElts = VT.getVectorNumElements();

  unsigned Mask = 0;
  for (unsigned l = 0; l != NumElts; l += 8) {
    // 8 nodes per lane, but we only care about the last 4.
    for (unsigned i = 0; i < 4; ++i) {
      int Elt = N->getMaskElt(l+i+4);
      if (Elt < 0) continue;
      Elt &= 0x3; // only 2-bits.
      Mask |= Elt << (i * 2);
    }
  }

  return Mask;
}

/// getShufflePSHUFLWImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with the PSHUFLW instruction.
static unsigned getShufflePSHUFLWImmediate(ShuffleVectorSDNode *N) {
  EVT VT = N->getValueType(0);

  assert((VT == MVT::v8i16 || VT == MVT::v16i16) &&
         "Unsupported vector type for PSHUFHW");

  unsigned NumElts = VT.getVectorNumElements();

  unsigned Mask = 0;
  for (unsigned l = 0; l != NumElts; l += 8) {
    // 8 nodes per lane, but we only care about the first 4.
    for (unsigned i = 0; i < 4; ++i) {
      int Elt = N->getMaskElt(l+i);
      if (Elt < 0) continue;
      Elt &= 0x3; // only 2-bits
      Mask |= Elt << (i * 2);
    }
  }

  return Mask;
}

/// getShufflePALIGNRImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with the PALIGNR instruction.
static unsigned getShufflePALIGNRImmediate(ShuffleVectorSDNode *SVOp) {
  EVT VT = SVOp->getValueType(0);
  unsigned EltSize = VT.getVectorElementType().getSizeInBits() >> 3;

  unsigned NumElts = VT.getVectorNumElements();
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElts = NumElts/NumLanes;

  int Val = 0;
  unsigned i;
  for (i = 0; i != NumElts; ++i) {
    Val = SVOp->getMaskElt(i);
    if (Val >= 0)
      break;
  }
  if (Val >= (int)NumElts)
    Val -= NumElts - NumLaneElts;

  assert(Val - i > 0 && "PALIGNR imm should be positive");
  return (Val - i) * EltSize;
}

/// getExtractVEXTRACTF128Immediate - Return the appropriate immediate
/// to extract the specified EXTRACT_SUBVECTOR index with VEXTRACTF128
/// instructions.
unsigned X86::getExtractVEXTRACTF128Immediate(SDNode *N) {
  if (!isa<ConstantSDNode>(N->getOperand(1).getNode()))
    llvm_unreachable("Illegal extract subvector for VEXTRACTF128");

  uint64_t Index =
    cast<ConstantSDNode>(N->getOperand(1).getNode())->getZExtValue();

  EVT VecVT = N->getOperand(0).getValueType();
  EVT ElVT = VecVT.getVectorElementType();

  unsigned NumElemsPerChunk = 128 / ElVT.getSizeInBits();
  return Index / NumElemsPerChunk;
}

/// getInsertVINSERTF128Immediate - Return the appropriate immediate
/// to insert at the specified INSERT_SUBVECTOR index with VINSERTF128
/// instructions.
unsigned X86::getInsertVINSERTF128Immediate(SDNode *N) {
  if (!isa<ConstantSDNode>(N->getOperand(2).getNode()))
    llvm_unreachable("Illegal insert subvector for VINSERTF128");

  uint64_t Index =
    cast<ConstantSDNode>(N->getOperand(2).getNode())->getZExtValue();

  EVT VecVT = N->getValueType(0);
  EVT ElVT = VecVT.getVectorElementType();

  unsigned NumElemsPerChunk = 128 / ElVT.getSizeInBits();
  return Index / NumElemsPerChunk;
}

/// getShuffleCLImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with VPERMQ and VPERMPD instructions.
/// Handles 256-bit.
static unsigned getShuffleCLImmediate(ShuffleVectorSDNode *N) {
  EVT VT = N->getValueType(0);

  unsigned NumElts = VT.getVectorNumElements();

  assert((VT.is256BitVector() && NumElts == 4) &&
         "Unsupported vector type for VPERMQ/VPERMPD");

  unsigned Mask = 0;
  for (unsigned i = 0; i != NumElts; ++i) {
    int Elt = N->getMaskElt(i);
    if (Elt < 0)
      continue;
    Mask |= Elt << (i*2);
  }

  return Mask;
}
/// isZeroNode - Returns true if Elt is a constant zero or a floating point
/// constant +0.0.
bool X86::isZeroNode(SDValue Elt) {
  return ((isa<ConstantSDNode>(Elt) &&
           cast<ConstantSDNode>(Elt)->isNullValue()) ||
          (isa<ConstantFPSDNode>(Elt) &&
           cast<ConstantFPSDNode>(Elt)->getValueAPF().isPosZero()));
}

/// CommuteVectorShuffle - Swap vector_shuffle operands as well as values in
/// their permute mask.
static SDValue CommuteVectorShuffle(ShuffleVectorSDNode *SVOp,
                                    SelectionDAG &DAG) {
  EVT VT = SVOp->getValueType(0);
  unsigned NumElems = VT.getVectorNumElements();
  SmallVector<int, 8> MaskVec;

  for (unsigned i = 0; i != NumElems; ++i) {
    int Idx = SVOp->getMaskElt(i);
    if (Idx >= 0) {
      if (Idx < (int)NumElems)
        Idx += NumElems;
      else
        Idx -= NumElems;
    }
    MaskVec.push_back(Idx);
  }
  return DAG.getVectorShuffle(VT, SVOp->getDebugLoc(), SVOp->getOperand(1),
                              SVOp->getOperand(0), &MaskVec[0]);
}

/// ShouldXformToMOVHLPS - Return true if the node should be transformed to
/// match movhlps. The lower half elements should come from upper half of
/// V1 (and in order), and the upper half elements should come from the upper
/// half of V2 (and in order).
static bool ShouldXformToMOVHLPS(ArrayRef<int> Mask, EVT VT) {
  if (!VT.is128BitVector())
    return false;
  if (VT.getVectorNumElements() != 4)
    return false;
  for (unsigned i = 0, e = 2; i != e; ++i)
    if (!isUndefOrEqual(Mask[i], i+2))
      return false;
  for (unsigned i = 2; i != 4; ++i)
    if (!isUndefOrEqual(Mask[i], i+4))
      return false;
  return true;
}

/// isScalarLoadToVector - Returns true if the node is a scalar load that
/// is promoted to a vector. It also returns the LoadSDNode by reference if
/// required.
static bool isScalarLoadToVector(SDNode *N, LoadSDNode **LD = NULL) {
  if (N->getOpcode() != ISD::SCALAR_TO_VECTOR)
    return false;
  N = N->getOperand(0).getNode();
  if (!ISD::isNON_EXTLoad(N))
    return false;
  if (LD)
    *LD = cast<LoadSDNode>(N);
  return true;
}

// Test whether the given value is a vector value which will be legalized
// into a load.
static bool WillBeConstantPoolLoad(SDNode *N) {
  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  // Check for any non-constant elements.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
    switch (N->getOperand(i).getNode()->getOpcode()) {
    case ISD::UNDEF:
    case ISD::ConstantFP:
    case ISD::Constant:
      break;
    default:
      return false;
    }

  // Vectors of all-zeros and all-ones are materialized with special
  // instructions rather than being loaded.
  return !ISD::isBuildVectorAllZeros(N) &&
         !ISD::isBuildVectorAllOnes(N);
}

/// ShouldXformToMOVLP{S|D} - Return true if the node should be transformed to
/// match movlp{s|d}. The lower half elements should come from lower half of
/// V1 (and in order), and the upper half elements should come from the upper
/// half of V2 (and in order). And since V1 will become the source of the
/// MOVLP, it must be either a vector load or a scalar load to vector.
static bool ShouldXformToMOVLP(SDNode *V1, SDNode *V2,
                               ArrayRef<int> Mask, EVT VT) {
  if (!VT.is128BitVector())
    return false;

  if (!ISD::isNON_EXTLoad(V1) && !isScalarLoadToVector(V1))
    return false;
  // Is V2 is a vector load, don't do this transformation. We will try to use
  // load folding shufps op.
  if (ISD::isNON_EXTLoad(V2) || WillBeConstantPoolLoad(V2))
    return false;

  unsigned NumElems = VT.getVectorNumElements();

  if (NumElems != 2 && NumElems != 4)
    return false;
  for (unsigned i = 0, e = NumElems/2; i != e; ++i)
    if (!isUndefOrEqual(Mask[i], i))
      return false;
  for (unsigned i = NumElems/2, e = NumElems; i != e; ++i)
    if (!isUndefOrEqual(Mask[i], i+NumElems))
      return false;
  return true;
}

/// isSplatVector - Returns true if N is a BUILD_VECTOR node whose elements are
/// all the same.
static bool isSplatVector(SDNode *N) {
  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  SDValue SplatValue = N->getOperand(0);
  for (unsigned i = 1, e = N->getNumOperands(); i != e; ++i)
    if (N->getOperand(i) != SplatValue)
      return false;
  return true;
}

/// isZeroShuffle - Returns true if N is a VECTOR_SHUFFLE that can be resolved
/// to an zero vector.
/// FIXME: move to dag combiner / method on ShuffleVectorSDNode
static bool isZeroShuffle(ShuffleVectorSDNode *N) {
  SDValue V1 = N->getOperand(0);
  SDValue V2 = N->getOperand(1);
  unsigned NumElems = N->getValueType(0).getVectorNumElements();
  for (unsigned i = 0; i != NumElems; ++i) {
    int Idx = N->getMaskElt(i);
    if (Idx >= (int)NumElems) {
      unsigned Opc = V2.getOpcode();
      if (Opc == ISD::UNDEF || ISD::isBuildVectorAllZeros(V2.getNode()))
        continue;
      if (Opc != ISD::BUILD_VECTOR ||
          !X86::isZeroNode(V2.getOperand(Idx-NumElems)))
        return false;
    } else if (Idx >= 0) {
      unsigned Opc = V1.getOpcode();
      if (Opc == ISD::UNDEF || ISD::isBuildVectorAllZeros(V1.getNode()))
        continue;
      if (Opc != ISD::BUILD_VECTOR ||
          !X86::isZeroNode(V1.getOperand(Idx)))
        return false;
    }
  }
  return true;
}

/// getZeroVector - Returns a vector of specified type with all zero elements.
///
static SDValue getZeroVector(EVT VT, const X86Subtarget *Subtarget,
                             SelectionDAG &DAG, DebugLoc dl) {
  assert(VT.isVector() && "Expected a vector type");
  unsigned Size = VT.getSizeInBits();

  // Always build SSE zero vectors as <4 x i32> bitcasted
  // to their dest type. This ensures they get CSE'd.
  SDValue Vec;
  if (Size == 128) {  // SSE
    if (Subtarget->hasSSE2()) {  // SSE2
      SDValue Cst = DAG.getTargetConstant(0, MVT::i32);
      Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Cst, Cst, Cst, Cst);
    } else { // SSE1
      SDValue Cst = DAG.getTargetConstantFP(+0.0, MVT::f32);
      Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4f32, Cst, Cst, Cst, Cst);
    }
  } else if (Size == 256) { // AVX
    if (Subtarget->hasAVX2()) { // AVX2
      SDValue Cst = DAG.getTargetConstant(0, MVT::i32);
      SDValue Ops[] = { Cst, Cst, Cst, Cst, Cst, Cst, Cst, Cst };
      Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v8i32, Ops, 8);
    } else {
      // 256-bit logic and arithmetic instructions in AVX are all
      // floating-point, no support for integer ops. Emit fp zeroed vectors.
      SDValue Cst = DAG.getTargetConstantFP(+0.0, MVT::f32);
      SDValue Ops[] = { Cst, Cst, Cst, Cst, Cst, Cst, Cst, Cst };
      Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v8f32, Ops, 8);
    }
  } else
    llvm_unreachable("Unexpected vector type");

  return DAG.getNode(ISD::BITCAST, dl, VT, Vec);
}

/// getOnesVector - Returns a vector of specified type with all bits set.
/// Always build ones vectors as <4 x i32> or <8 x i32>. For 256-bit types with
/// no AVX2 supprt, use two <4 x i32> inserted in a <8 x i32> appropriately.
/// Then bitcast to their original type, ensuring they get CSE'd.
static SDValue getOnesVector(EVT VT, bool HasAVX2, SelectionDAG &DAG,
                             DebugLoc dl) {
  assert(VT.isVector() && "Expected a vector type");
  unsigned Size = VT.getSizeInBits();

  SDValue Cst = DAG.getTargetConstant(~0U, MVT::i32);
  SDValue Vec;
  if (Size == 256) {
    if (HasAVX2) { // AVX2
      SDValue Ops[] = { Cst, Cst, Cst, Cst, Cst, Cst, Cst, Cst };
      Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v8i32, Ops, 8);
    } else { // AVX
      Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Cst, Cst, Cst, Cst);
      Vec = Concat128BitVectors(Vec, Vec, MVT::v8i32, 8, DAG, dl);
    }
  } else if (Size == 128) {
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Cst, Cst, Cst, Cst);
  } else
    llvm_unreachable("Unexpected vector type");

  return DAG.getNode(ISD::BITCAST, dl, VT, Vec);
}

/// NormalizeMask - V2 is a splat, modify the mask (if needed) so all elements
/// that point to V2 points to its first element.
static void NormalizeMask(SmallVectorImpl<int> &Mask, unsigned NumElems) {
  for (unsigned i = 0; i != NumElems; ++i) {
    if (Mask[i] > (int)NumElems) {
      Mask[i] = NumElems;
    }
  }
}

/// getMOVLMask - Returns a vector_shuffle mask for an movs{s|d}, movd
/// operation of specified width.
static SDValue getMOVL(SelectionDAG &DAG, DebugLoc dl, EVT VT, SDValue V1,
                       SDValue V2) {
  unsigned NumElems = VT.getVectorNumElements();
  SmallVector<int, 8> Mask;
  Mask.push_back(NumElems);
  for (unsigned i = 1; i != NumElems; ++i)
    Mask.push_back(i);
  return DAG.getVectorShuffle(VT, dl, V1, V2, &Mask[0]);
}

/// getUnpackl - Returns a vector_shuffle node for an unpackl operation.
static SDValue getUnpackl(SelectionDAG &DAG, DebugLoc dl, EVT VT, SDValue V1,
                          SDValue V2) {
  unsigned NumElems = VT.getVectorNumElements();
  SmallVector<int, 8> Mask;
  for (unsigned i = 0, e = NumElems/2; i != e; ++i) {
    Mask.push_back(i);
    Mask.push_back(i + NumElems);
  }
  return DAG.getVectorShuffle(VT, dl, V1, V2, &Mask[0]);
}

/// getUnpackh - Returns a vector_shuffle node for an unpackh operation.
static SDValue getUnpackh(SelectionDAG &DAG, DebugLoc dl, EVT VT, SDValue V1,
                          SDValue V2) {
  unsigned NumElems = VT.getVectorNumElements();
  SmallVector<int, 8> Mask;
  for (unsigned i = 0, Half = NumElems/2; i != Half; ++i) {
    Mask.push_back(i + Half);
    Mask.push_back(i + NumElems + Half);
  }
  return DAG.getVectorShuffle(VT, dl, V1, V2, &Mask[0]);
}

// PromoteSplati8i16 - All i16 and i8 vector types can't be used directly by
// a generic shuffle instruction because the target has no such instructions.
// Generate shuffles which repeat i16 and i8 several times until they can be
// represented by v4f32 and then be manipulated by target suported shuffles.
static SDValue PromoteSplati8i16(SDValue V, SelectionDAG &DAG, int &EltNo) {
  EVT VT = V.getValueType();
  int NumElems = VT.getVectorNumElements();
  DebugLoc dl = V.getDebugLoc();

  while (NumElems > 4) {
    if (EltNo < NumElems/2) {
      V = getUnpackl(DAG, dl, VT, V, V);
    } else {
      V = getUnpackh(DAG, dl, VT, V, V);
      EltNo -= NumElems/2;
    }
    NumElems >>= 1;
  }
  return V;
}

/// getLegalSplat - Generate a legal splat with supported x86 shuffles
static SDValue getLegalSplat(SelectionDAG &DAG, SDValue V, int EltNo) {
  EVT VT = V.getValueType();
  DebugLoc dl = V.getDebugLoc();
  unsigned Size = VT.getSizeInBits();

  if (Size == 128) {
    V = DAG.getNode(ISD::BITCAST, dl, MVT::v4f32, V);
    int SplatMask[4] = { EltNo, EltNo, EltNo, EltNo };
    V = DAG.getVectorShuffle(MVT::v4f32, dl, V, DAG.getUNDEF(MVT::v4f32),
                             &SplatMask[0]);
  } else if (Size == 256) {
    // To use VPERMILPS to splat scalars, the second half of indicies must
    // refer to the higher part, which is a duplication of the lower one,
    // because VPERMILPS can only handle in-lane permutations.
    int SplatMask[8] = { EltNo, EltNo, EltNo, EltNo,
                         EltNo+4, EltNo+4, EltNo+4, EltNo+4 };

    V = DAG.getNode(ISD::BITCAST, dl, MVT::v8f32, V);
    V = DAG.getVectorShuffle(MVT::v8f32, dl, V, DAG.getUNDEF(MVT::v8f32),
                             &SplatMask[0]);
  } else
    llvm_unreachable("Vector size not supported");

  return DAG.getNode(ISD::BITCAST, dl, VT, V);
}

/// PromoteSplat - Splat is promoted to target supported vector shuffles.
static SDValue PromoteSplat(ShuffleVectorSDNode *SV, SelectionDAG &DAG) {
  EVT SrcVT = SV->getValueType(0);
  SDValue V1 = SV->getOperand(0);
  DebugLoc dl = SV->getDebugLoc();

  int EltNo = SV->getSplatIndex();
  int NumElems = SrcVT.getVectorNumElements();
  unsigned Size = SrcVT.getSizeInBits();

  assert(((Size == 128 && NumElems > 4) || Size == 256) &&
          "Unknown how to promote splat for type");

  // Extract the 128-bit part containing the splat element and update
  // the splat element index when it refers to the higher register.
  if (Size == 256) {
    V1 = Extract128BitVector(V1, EltNo, DAG, dl);
    if (EltNo >= NumElems/2)
      EltNo -= NumElems/2;
  }

  // All i16 and i8 vector types can't be used directly by a generic shuffle
  // instruction because the target has no such instruction. Generate shuffles
  // which repeat i16 and i8 several times until they fit in i32, and then can
  // be manipulated by target suported shuffles.
  EVT EltVT = SrcVT.getVectorElementType();
  if (EltVT == MVT::i8 || EltVT == MVT::i16)
    V1 = PromoteSplati8i16(V1, DAG, EltNo);

  // Recreate the 256-bit vector and place the same 128-bit vector
  // into the low and high part. This is necessary because we want
  // to use VPERM* to shuffle the vectors
  if (Size == 256) {
    V1 = DAG.getNode(ISD::CONCAT_VECTORS, dl, SrcVT, V1, V1);
  }

  return getLegalSplat(DAG, V1, EltNo);
}

/// getShuffleVectorZeroOrUndef - Return a vector_shuffle of the specified
/// vector of zero or undef vector.  This produces a shuffle where the low
/// element of V2 is swizzled into the zero/undef vector, landing at element
/// Idx.  This produces a shuffle mask like 4,1,2,3 (idx=0) or  0,1,2,4 (idx=3).
static SDValue getShuffleVectorZeroOrUndef(SDValue V2, unsigned Idx,
                                           bool IsZero,
                                           const X86Subtarget *Subtarget,
                                           SelectionDAG &DAG) {
  EVT VT = V2.getValueType();
  SDValue V1 = IsZero
    ? getZeroVector(VT, Subtarget, DAG, V2.getDebugLoc()) : DAG.getUNDEF(VT);
  unsigned NumElems = VT.getVectorNumElements();
  SmallVector<int, 16> MaskVec;
  for (unsigned i = 0; i != NumElems; ++i)
    // If this is the insertion idx, put the low elt of V2 here.
    MaskVec.push_back(i == Idx ? NumElems : i);
  return DAG.getVectorShuffle(VT, V2.getDebugLoc(), V1, V2, &MaskVec[0]);
}

/// getTargetShuffleMask - Calculates the shuffle mask corresponding to the
/// target specific opcode. Returns true if the Mask could be calculated.
/// Sets IsUnary to true if only uses one source.
static bool getTargetShuffleMask(SDNode *N, MVT VT,
                                 SmallVectorImpl<int> &Mask, bool &IsUnary) {
  unsigned NumElems = VT.getVectorNumElements();
  SDValue ImmN;

  IsUnary = false;
  switch(N->getOpcode()) {
  case X86ISD::SHUFP:
    ImmN = N->getOperand(N->getNumOperands()-1);
    DecodeSHUFPMask(VT, cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    break;
  case X86ISD::UNPCKH:
    DecodeUNPCKHMask(VT, Mask);
    break;
  case X86ISD::UNPCKL:
    DecodeUNPCKLMask(VT, Mask);
    break;
  case X86ISD::MOVHLPS:
    DecodeMOVHLPSMask(NumElems, Mask);
    break;
  case X86ISD::MOVLHPS:
    DecodeMOVLHPSMask(NumElems, Mask);
    break;
  case X86ISD::PSHUFD:
  case X86ISD::VPERMILP:
    ImmN = N->getOperand(N->getNumOperands()-1);
    DecodePSHUFMask(VT, cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = true;
    break;
  case X86ISD::PSHUFHW:
    ImmN = N->getOperand(N->getNumOperands()-1);
    DecodePSHUFHWMask(VT, cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = true;
    break;
  case X86ISD::PSHUFLW:
    ImmN = N->getOperand(N->getNumOperands()-1);
    DecodePSHUFLWMask(VT, cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = true;
    break;
  case X86ISD::VPERMI:
    ImmN = N->getOperand(N->getNumOperands()-1);
    DecodeVPERMMask(cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = true;
    break;
  case X86ISD::MOVSS:
  case X86ISD::MOVSD: {
    // The index 0 always comes from the first element of the second source,
    // this is why MOVSS and MOVSD are used in the first place. The other
    // elements come from the other positions of the first source vector
    Mask.push_back(NumElems);
    for (unsigned i = 1; i != NumElems; ++i) {
      Mask.push_back(i);
    }
    break;
  }
  case X86ISD::VPERM2X128:
    ImmN = N->getOperand(N->getNumOperands()-1);
    DecodeVPERM2X128Mask(VT, cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    if (Mask.empty()) return false;
    break;
  case X86ISD::MOVDDUP:
  case X86ISD::MOVLHPD:
  case X86ISD::MOVLPD:
  case X86ISD::MOVLPS:
  case X86ISD::MOVSHDUP:
  case X86ISD::MOVSLDUP:
  case X86ISD::PALIGN:
    // Not yet implemented
    return false;
  default: llvm_unreachable("unknown target shuffle node");
  }

  return true;
}

/// getShuffleScalarElt - Returns the scalar element that will make up the ith
/// element of the result of the vector shuffle.
static SDValue getShuffleScalarElt(SDNode *N, unsigned Index, SelectionDAG &DAG,
                                   unsigned Depth) {
  if (Depth == 6)
    return SDValue();  // Limit search depth.

  SDValue V = SDValue(N, 0);
  EVT VT = V.getValueType();
  unsigned Opcode = V.getOpcode();

  // Recurse into ISD::VECTOR_SHUFFLE node to find scalars.
  if (const ShuffleVectorSDNode *SV = dyn_cast<ShuffleVectorSDNode>(N)) {
    int Elt = SV->getMaskElt(Index);

    if (Elt < 0)
      return DAG.getUNDEF(VT.getVectorElementType());

    unsigned NumElems = VT.getVectorNumElements();
    SDValue NewV = (Elt < (int)NumElems) ? SV->getOperand(0)
                                         : SV->getOperand(1);
    return getShuffleScalarElt(NewV.getNode(), Elt % NumElems, DAG, Depth+1);
  }

  // Recurse into target specific vector shuffles to find scalars.
  if (isTargetShuffle(Opcode)) {
    MVT ShufVT = V.getValueType().getSimpleVT();
    unsigned NumElems = ShufVT.getVectorNumElements();
    SmallVector<int, 16> ShuffleMask;
    bool IsUnary;

    if (!getTargetShuffleMask(N, ShufVT, ShuffleMask, IsUnary))
      return SDValue();

    int Elt = ShuffleMask[Index];
    if (Elt < 0)
      return DAG.getUNDEF(ShufVT.getVectorElementType());

    SDValue NewV = (Elt < (int)NumElems) ? N->getOperand(0)
                                         : N->getOperand(1);
    return getShuffleScalarElt(NewV.getNode(), Elt % NumElems, DAG,
                               Depth+1);
  }

  // Actual nodes that may contain scalar elements
  if (Opcode == ISD::BITCAST) {
    V = V.getOperand(0);
    EVT SrcVT = V.getValueType();
    unsigned NumElems = VT.getVectorNumElements();

    if (!SrcVT.isVector() || SrcVT.getVectorNumElements() != NumElems)
      return SDValue();
  }

  if (V.getOpcode() == ISD::SCALAR_TO_VECTOR)
    return (Index == 0) ? V.getOperand(0)
                        : DAG.getUNDEF(VT.getVectorElementType());

  if (V.getOpcode() == ISD::BUILD_VECTOR)
    return V.getOperand(Index);

  return SDValue();
}

/// getNumOfConsecutiveZeros - Return the number of elements of a vector
/// shuffle operation which come from a consecutively from a zero. The
/// search can start in two different directions, from left or right.
static
unsigned getNumOfConsecutiveZeros(ShuffleVectorSDNode *SVOp, unsigned NumElems,
                                  bool ZerosFromLeft, SelectionDAG &DAG) {
  unsigned i;
  for (i = 0; i != NumElems; ++i) {
    unsigned Index = ZerosFromLeft ? i : NumElems-i-1;
    SDValue Elt = getShuffleScalarElt(SVOp, Index, DAG, 0);
    if (!(Elt.getNode() &&
         (Elt.getOpcode() == ISD::UNDEF || X86::isZeroNode(Elt))))
      break;
  }

  return i;
}

/// isShuffleMaskConsecutive - Check if the shuffle mask indicies [MaskI, MaskE)
/// correspond consecutively to elements from one of the vector operands,
/// starting from its index OpIdx. Also tell OpNum which source vector operand.
static
bool isShuffleMaskConsecutive(ShuffleVectorSDNode *SVOp,
                              unsigned MaskI, unsigned MaskE, unsigned OpIdx,
                              unsigned NumElems, unsigned &OpNum) {
  bool SeenV1 = false;
  bool SeenV2 = false;

  for (unsigned i = MaskI; i != MaskE; ++i, ++OpIdx) {
    int Idx = SVOp->getMaskElt(i);
    // Ignore undef indicies
    if (Idx < 0)
      continue;

    if (Idx < (int)NumElems)
      SeenV1 = true;
    else
      SeenV2 = true;

    // Only accept consecutive elements from the same vector
    if ((Idx % NumElems != OpIdx) || (SeenV1 && SeenV2))
      return false;
  }

  OpNum = SeenV1 ? 0 : 1;
  return true;
}

/// isVectorShiftRight - Returns true if the shuffle can be implemented as a
/// logical left shift of a vector.
static bool isVectorShiftRight(ShuffleVectorSDNode *SVOp, SelectionDAG &DAG,
                               bool &isLeft, SDValue &ShVal, unsigned &ShAmt) {
  unsigned NumElems = SVOp->getValueType(0).getVectorNumElements();
  unsigned NumZeros = getNumOfConsecutiveZeros(SVOp, NumElems,
              false /* check zeros from right */, DAG);
  unsigned OpSrc;

  if (!NumZeros)
    return false;

  // Considering the elements in the mask that are not consecutive zeros,
  // check if they consecutively come from only one of the source vectors.
  //
  //               V1 = {X, A, B, C}     0
  //                         \  \  \    /
  //   vector_shuffle V1, V2 <1, 2, 3, X>
  //
  if (!isShuffleMaskConsecutive(SVOp,
            0,                   // Mask Start Index
            NumElems-NumZeros,   // Mask End Index(exclusive)
            NumZeros,            // Where to start looking in the src vector
            NumElems,            // Number of elements in vector
            OpSrc))              // Which source operand ?
    return false;

  isLeft = false;
  ShAmt = NumZeros;
  ShVal = SVOp->getOperand(OpSrc);
  return true;
}

/// isVectorShiftLeft - Returns true if the shuffle can be implemented as a
/// logical left shift of a vector.
static bool isVectorShiftLeft(ShuffleVectorSDNode *SVOp, SelectionDAG &DAG,
                              bool &isLeft, SDValue &ShVal, unsigned &ShAmt) {
  unsigned NumElems = SVOp->getValueType(0).getVectorNumElements();
  unsigned NumZeros = getNumOfConsecutiveZeros(SVOp, NumElems,
              true /* check zeros from left */, DAG);
  unsigned OpSrc;

  if (!NumZeros)
    return false;

  // Considering the elements in the mask that are not consecutive zeros,
  // check if they consecutively come from only one of the source vectors.
  //
  //                           0    { A, B, X, X } = V2
  //                          / \    /  /
  //   vector_shuffle V1, V2 <X, X, 4, 5>
  //
  if (!isShuffleMaskConsecutive(SVOp,
            NumZeros,     // Mask Start Index
            NumElems,     // Mask End Index(exclusive)
            0,            // Where to start looking in the src vector
            NumElems,     // Number of elements in vector
            OpSrc))       // Which source operand ?
    return false;

  isLeft = true;
  ShAmt = NumZeros;
  ShVal = SVOp->getOperand(OpSrc);
  return true;
}

/// isVectorShift - Returns true if the shuffle can be implemented as a
/// logical left or right shift of a vector.
static bool isVectorShift(ShuffleVectorSDNode *SVOp, SelectionDAG &DAG,
                          bool &isLeft, SDValue &ShVal, unsigned &ShAmt) {
  // Although the logic below support any bitwidth size, there are no
  // shift instructions which handle more than 128-bit vectors.
  if (!SVOp->getValueType(0).is128BitVector())
    return false;

  if (isVectorShiftLeft(SVOp, DAG, isLeft, ShVal, ShAmt) ||
      isVectorShiftRight(SVOp, DAG, isLeft, ShVal, ShAmt))
    return true;

  return false;
}

/// LowerBuildVectorv16i8 - Custom lower build_vector of v16i8.
///
static SDValue LowerBuildVectorv16i8(SDValue Op, unsigned NonZeros,
                                       unsigned NumNonZero, unsigned NumZero,
                                       SelectionDAG &DAG,
                                       const X86Subtarget* Subtarget,
                                       const TargetLowering &TLI) {
  if (NumNonZero > 8)
    return SDValue();

  DebugLoc dl = Op.getDebugLoc();
  SDValue V(0, 0);
  bool First = true;
  for (unsigned i = 0; i < 16; ++i) {
    bool ThisIsNonZero = (NonZeros & (1 << i)) != 0;
    if (ThisIsNonZero && First) {
      if (NumZero)
        V = getZeroVector(MVT::v8i16, Subtarget, DAG, dl);
      else
        V = DAG.getUNDEF(MVT::v8i16);
      First = false;
    }

    if ((i & 1) != 0) {
      SDValue ThisElt(0, 0), LastElt(0, 0);
      bool LastIsNonZero = (NonZeros & (1 << (i-1))) != 0;
      if (LastIsNonZero) {
        LastElt = DAG.getNode(ISD::ZERO_EXTEND, dl,
                              MVT::i16, Op.getOperand(i-1));
      }
      if (ThisIsNonZero) {
        ThisElt = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, Op.getOperand(i));
        ThisElt = DAG.getNode(ISD::SHL, dl, MVT::i16,
                              ThisElt, DAG.getConstant(8, MVT::i8));
        if (LastIsNonZero)
          ThisElt = DAG.getNode(ISD::OR, dl, MVT::i16, ThisElt, LastElt);
      } else
        ThisElt = LastElt;

      if (ThisElt.getNode())
        V = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v8i16, V, ThisElt,
                        DAG.getIntPtrConstant(i/2));
    }
  }

  return DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, V);
}

/// LowerBuildVectorv8i16 - Custom lower build_vector of v8i16.
///
static SDValue LowerBuildVectorv8i16(SDValue Op, unsigned NonZeros,
                                     unsigned NumNonZero, unsigned NumZero,
                                     SelectionDAG &DAG,
                                     const X86Subtarget* Subtarget,
                                     const TargetLowering &TLI) {
  if (NumNonZero > 4)
    return SDValue();

  DebugLoc dl = Op.getDebugLoc();
  SDValue V(0, 0);
  bool First = true;
  for (unsigned i = 0; i < 8; ++i) {
    bool isNonZero = (NonZeros & (1 << i)) != 0;
    if (isNonZero) {
      if (First) {
        if (NumZero)
          V = getZeroVector(MVT::v8i16, Subtarget, DAG, dl);
        else
          V = DAG.getUNDEF(MVT::v8i16);
        First = false;
      }
      V = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl,
                      MVT::v8i16, V, Op.getOperand(i),
                      DAG.getIntPtrConstant(i));
    }
  }

  return V;
}

/// getVShift - Return a vector logical shift node.
///
static SDValue getVShift(bool isLeft, EVT VT, SDValue SrcOp,
                         unsigned NumBits, SelectionDAG &DAG,
                         const TargetLowering &TLI, DebugLoc dl) {
  assert(VT.is128BitVector() && "Unknown type for VShift");
  EVT ShVT = MVT::v2i64;
  unsigned Opc = isLeft ? X86ISD::VSHLDQ : X86ISD::VSRLDQ;
  SrcOp = DAG.getNode(ISD::BITCAST, dl, ShVT, SrcOp);
  return DAG.getNode(ISD::BITCAST, dl, VT,
                     DAG.getNode(Opc, dl, ShVT, SrcOp,
                             DAG.getConstant(NumBits,
                                  TLI.getShiftAmountTy(SrcOp.getValueType()))));
}

SDValue
X86TargetLowering::LowerAsSplatVectorLoad(SDValue SrcOp, EVT VT, DebugLoc dl,
                                          SelectionDAG &DAG) const {

  // Check if the scalar load can be widened into a vector load. And if
  // the address is "base + cst" see if the cst can be "absorbed" into
  // the shuffle mask.
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(SrcOp)) {
    SDValue Ptr = LD->getBasePtr();
    if (!ISD::isNormalLoad(LD) || LD->isVolatile())
      return SDValue();
    EVT PVT = LD->getValueType(0);
    if (PVT != MVT::i32 && PVT != MVT::f32)
      return SDValue();

    int FI = -1;
    int64_t Offset = 0;
    if (FrameIndexSDNode *FINode = dyn_cast<FrameIndexSDNode>(Ptr)) {
      FI = FINode->getIndex();
      Offset = 0;
    } else if (DAG.isBaseWithConstantOffset(Ptr) &&
               isa<FrameIndexSDNode>(Ptr.getOperand(0))) {
      FI = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
      Offset = Ptr.getConstantOperandVal(1);
      Ptr = Ptr.getOperand(0);
    } else {
      return SDValue();
    }

    // FIXME: 256-bit vector instructions don't require a strict alignment,
    // improve this code to support it better.
    unsigned RequiredAlign = VT.getSizeInBits()/8;
    SDValue Chain = LD->getChain();
    // Make sure the stack object alignment is at least 16 or 32.
    MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
    if (DAG.InferPtrAlignment(Ptr) < RequiredAlign) {
      if (MFI->isFixedObjectIndex(FI)) {
        // Can't change the alignment. FIXME: It's possible to compute
        // the exact stack offset and reference FI + adjust offset instead.
        // If someone *really* cares about this. That's the way to implement it.
        return SDValue();
      } else {
        MFI->setObjectAlignment(FI, RequiredAlign);
      }
    }

    // (Offset % 16 or 32) must be multiple of 4. Then address is then
    // Ptr + (Offset & ~15).
    if (Offset < 0)
      return SDValue();
    if ((Offset % RequiredAlign) & 3)
      return SDValue();
    int64_t StartOffset = Offset & ~(RequiredAlign-1);
    if (StartOffset)
      Ptr = DAG.getNode(ISD::ADD, Ptr.getDebugLoc(), Ptr.getValueType(),
                        Ptr,DAG.getConstant(StartOffset, Ptr.getValueType()));

    int EltNo = (Offset - StartOffset) >> 2;
    unsigned NumElems = VT.getVectorNumElements();

    EVT NVT = EVT::getVectorVT(*DAG.getContext(), PVT, NumElems);
    SDValue V1 = DAG.getLoad(NVT, dl, Chain, Ptr,
                             LD->getPointerInfo().getWithOffset(StartOffset),
                             false, false, false, 0);

    SmallVector<int, 8> Mask;
    for (unsigned i = 0; i != NumElems; ++i)
      Mask.push_back(EltNo);

    return DAG.getVectorShuffle(NVT, dl, V1, DAG.getUNDEF(NVT), &Mask[0]);
  }

  return SDValue();
}

/// EltsFromConsecutiveLoads - Given the initializing elements 'Elts' of a
/// vector of type 'VT', see if the elements can be replaced by a single large
/// load which has the same value as a build_vector whose operands are 'elts'.
///
/// Example: <load i32 *a, load i32 *a+4, undef, undef> -> zextload a
///
/// FIXME: we'd also like to handle the case where the last elements are zero
/// rather than undef via VZEXT_LOAD, but we do not detect that case today.
/// There's even a handy isZeroNode for that purpose.
static SDValue EltsFromConsecutiveLoads(EVT VT, SmallVectorImpl<SDValue> &Elts,
                                        DebugLoc &DL, SelectionDAG &DAG) {
  EVT EltVT = VT.getVectorElementType();
  unsigned NumElems = Elts.size();

  LoadSDNode *LDBase = NULL;
  unsigned LastLoadedElt = -1U;

  // For each element in the initializer, see if we've found a load or an undef.
  // If we don't find an initial load element, or later load elements are
  // non-consecutive, bail out.
  for (unsigned i = 0; i < NumElems; ++i) {
    SDValue Elt = Elts[i];

    if (!Elt.getNode() ||
        (Elt.getOpcode() != ISD::UNDEF && !ISD::isNON_EXTLoad(Elt.getNode())))
      return SDValue();
    if (!LDBase) {
      if (Elt.getNode()->getOpcode() == ISD::UNDEF)
        return SDValue();
      LDBase = cast<LoadSDNode>(Elt.getNode());
      LastLoadedElt = i;
      continue;
    }
    if (Elt.getOpcode() == ISD::UNDEF)
      continue;

    LoadSDNode *LD = cast<LoadSDNode>(Elt);
    if (!DAG.isConsecutiveLoad(LD, LDBase, EltVT.getSizeInBits()/8, i))
      return SDValue();
    LastLoadedElt = i;
  }

  // If we have found an entire vector of loads and undefs, then return a large
  // load of the entire vector width starting at the base pointer.  If we found
  // consecutive loads for the low half, generate a vzext_load node.
  if (LastLoadedElt == NumElems - 1) {
    if (DAG.InferPtrAlignment(LDBase->getBasePtr()) >= 16)
      return DAG.getLoad(VT, DL, LDBase->getChain(), LDBase->getBasePtr(),
                         LDBase->getPointerInfo(),
                         LDBase->isVolatile(), LDBase->isNonTemporal(),
                         LDBase->isInvariant(), 0);
    return DAG.getLoad(VT, DL, LDBase->getChain(), LDBase->getBasePtr(),
                       LDBase->getPointerInfo(),
                       LDBase->isVolatile(), LDBase->isNonTemporal(),
                       LDBase->isInvariant(), LDBase->getAlignment());
  }
  if (NumElems == 4 && LastLoadedElt == 1 &&
      DAG.getTargetLoweringInfo().isTypeLegal(MVT::v2i64)) {
    SDVTList Tys = DAG.getVTList(MVT::v2i64, MVT::Other);
    SDValue Ops[] = { LDBase->getChain(), LDBase->getBasePtr() };
    SDValue ResNode =
        DAG.getMemIntrinsicNode(X86ISD::VZEXT_LOAD, DL, Tys, Ops, 2, MVT::i64,
                                LDBase->getPointerInfo(),
                                LDBase->getAlignment(),
                                false/*isVolatile*/, true/*ReadMem*/,
                                false/*WriteMem*/);

    // Make sure the newly-created LOAD is in the same position as LDBase in
    // terms of dependency. We create a TokenFactor for LDBase and ResNode, and
    // update uses of LDBase's output chain to use the TokenFactor.
    if (LDBase->hasAnyUseOfValue(1)) {
      SDValue NewChain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other,
                             SDValue(LDBase, 1), SDValue(ResNode.getNode(), 1));
      DAG.ReplaceAllUsesOfValueWith(SDValue(LDBase, 1), NewChain);
      DAG.UpdateNodeOperands(NewChain.getNode(), SDValue(LDBase, 1),
                             SDValue(ResNode.getNode(), 1));
    }

    return DAG.getNode(ISD::BITCAST, DL, VT, ResNode);
  }
  return SDValue();
}

/// LowerVectorBroadcast - Attempt to use the vbroadcast instruction
/// to generate a splat value for the following cases:
/// 1. A splat BUILD_VECTOR which uses a single scalar load, or a constant.
/// 2. A splat shuffle which uses a scalar_to_vector node which comes from
/// a scalar load, or a constant.
/// The VBROADCAST node is returned when a pattern is found,
/// or SDValue() otherwise.
SDValue
X86TargetLowering::LowerVectorBroadcast(SDValue Op, SelectionDAG &DAG) const {
  if (!Subtarget->hasAVX())
    return SDValue();

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();

  assert((VT.is128BitVector() || VT.is256BitVector()) &&
         "Unsupported vector type for broadcast.");

  SDValue Ld;
  bool ConstSplatVal;

  switch (Op.getOpcode()) {
    default:
      // Unknown pattern found.
      return SDValue();

    case ISD::BUILD_VECTOR: {
      // The BUILD_VECTOR node must be a splat.
      if (!isSplatVector(Op.getNode()))
        return SDValue();

      Ld = Op.getOperand(0);
      ConstSplatVal = (Ld.getOpcode() == ISD::Constant ||
                     Ld.getOpcode() == ISD::ConstantFP);

      // The suspected load node has several users. Make sure that all
      // of its users are from the BUILD_VECTOR node.
      // Constants may have multiple users.
      if (!ConstSplatVal && !Ld->hasNUsesOfValue(VT.getVectorNumElements(), 0))
        return SDValue();
      break;
    }

    case ISD::VECTOR_SHUFFLE: {
      ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);

      // Shuffles must have a splat mask where the first element is
      // broadcasted.
      if ((!SVOp->isSplat()) || SVOp->getMaskElt(0) != 0)
        return SDValue();

      SDValue Sc = Op.getOperand(0);
      if (Sc.getOpcode() != ISD::SCALAR_TO_VECTOR &&
          Sc.getOpcode() != ISD::BUILD_VECTOR) {

        if (!Subtarget->hasAVX2())
          return SDValue();

        // Use the register form of the broadcast instruction available on AVX2.
        if (VT.is256BitVector())
          Sc = Extract128BitVector(Sc, 0, DAG, dl);
        return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Sc);
      }

      Ld = Sc.getOperand(0);
      ConstSplatVal = (Ld.getOpcode() == ISD::Constant ||
                       Ld.getOpcode() == ISD::ConstantFP);

      // The scalar_to_vector node and the suspected
      // load node must have exactly one user.
      // Constants may have multiple users.
      if (!ConstSplatVal && (!Sc.hasOneUse() || !Ld.hasOneUse()))
        return SDValue();
      break;
    }
  }

  bool Is256 = VT.is256BitVector();

  // Handle the broadcasting a single constant scalar from the constant pool
  // into a vector. On Sandybridge it is still better to load a constant vector
  // from the constant pool and not to broadcast it from a scalar.
  if (ConstSplatVal && Subtarget->hasAVX2()) {
    EVT CVT = Ld.getValueType();
    assert(!CVT.isVector() && "Must not broadcast a vector type");
    unsigned ScalarSize = CVT.getSizeInBits();

    if (ScalarSize == 32 || (Is256 && ScalarSize == 64)) {
      const Constant *C = 0;
      if (ConstantSDNode *CI = dyn_cast<ConstantSDNode>(Ld))
        C = CI->getConstantIntValue();
      else if (ConstantFPSDNode *CF = dyn_cast<ConstantFPSDNode>(Ld))
        C = CF->getConstantFPValue();

      assert(C && "Invalid constant type");

      SDValue CP = DAG.getConstantPool(C, getPointerTy());
      unsigned Alignment = cast<ConstantPoolSDNode>(CP)->getAlignment();
      Ld = DAG.getLoad(CVT, dl, DAG.getEntryNode(), CP,
                       MachinePointerInfo::getConstantPool(),
                       false, false, false, Alignment);

      return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Ld);
    }
  }

  bool IsLoad = ISD::isNormalLoad(Ld.getNode());
  unsigned ScalarSize = Ld.getValueType().getSizeInBits();

  // Handle AVX2 in-register broadcasts.
  if (!IsLoad && Subtarget->hasAVX2() &&
      (ScalarSize == 32 || (Is256 && ScalarSize == 64)))
    return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Ld);

  // The scalar source must be a normal load.
  if (!IsLoad)
    return SDValue();

  if (ScalarSize == 32 || (Is256 && ScalarSize == 64))
    return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Ld);

  // The integer check is needed for the 64-bit into 128-bit so it doesn't match
  // double since there is no vbroadcastsd xmm
  if (Subtarget->hasAVX2() && Ld.getValueType().isInteger()) {
    if (ScalarSize == 8 || ScalarSize == 16 || ScalarSize == 64)
      return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Ld);
  }

  // Unsupported broadcast.
  return SDValue();
}

SDValue
X86TargetLowering::buildFromShuffleMostly(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();

  // Skip if insert_vec_elt is not supported.
  if (!isOperationLegalOrCustom(ISD::INSERT_VECTOR_ELT, VT))
    return SDValue();

  DebugLoc DL = Op.getDebugLoc();
  unsigned NumElems = Op.getNumOperands();

  SDValue VecIn1;
  SDValue VecIn2;
  SmallVector<unsigned, 4> InsertIndices;
  SmallVector<int, 8> Mask(NumElems, -1);

  for (unsigned i = 0; i != NumElems; ++i) {
    unsigned Opc = Op.getOperand(i).getOpcode();

    if (Opc == ISD::UNDEF)
      continue;

    if (Opc != ISD::EXTRACT_VECTOR_ELT) {
      // Quit if more than 1 elements need inserting.
      if (InsertIndices.size() > 1)
        return SDValue();

      InsertIndices.push_back(i);
      continue;
    }

    SDValue ExtractedFromVec = Op.getOperand(i).getOperand(0);
    SDValue ExtIdx = Op.getOperand(i).getOperand(1);

    // Quit if extracted from vector of different type.
    if (ExtractedFromVec.getValueType() != VT)
      return SDValue();

    // Quit if non-constant index.
    if (!isa<ConstantSDNode>(ExtIdx))
      return SDValue();

    if (VecIn1.getNode() == 0)
      VecIn1 = ExtractedFromVec;
    else if (VecIn1 != ExtractedFromVec) {
      if (VecIn2.getNode() == 0)
        VecIn2 = ExtractedFromVec;
      else if (VecIn2 != ExtractedFromVec)
        // Quit if more than 2 vectors to shuffle
        return SDValue();
    }

    unsigned Idx = cast<ConstantSDNode>(ExtIdx)->getZExtValue();

    if (ExtractedFromVec == VecIn1)
      Mask[i] = Idx;
    else if (ExtractedFromVec == VecIn2)
      Mask[i] = Idx + NumElems;
  }

  if (VecIn1.getNode() == 0)
    return SDValue();

  VecIn2 = VecIn2.getNode() ? VecIn2 : DAG.getUNDEF(VT);
  SDValue NV = DAG.getVectorShuffle(VT, DL, VecIn1, VecIn2, &Mask[0]);
  for (unsigned i = 0, e = InsertIndices.size(); i != e; ++i) {
    unsigned Idx = InsertIndices[i];
    NV = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, VT, NV, Op.getOperand(Idx),
                     DAG.getIntPtrConstant(Idx));
  }

  return NV;
}

SDValue
X86TargetLowering::LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();

  EVT VT = Op.getValueType();
  EVT ExtVT = VT.getVectorElementType();
  unsigned NumElems = Op.getNumOperands();

  // Vectors containing all zeros can be matched by pxor and xorps later
  if (ISD::isBuildVectorAllZeros(Op.getNode())) {
    // Canonicalize this to <4 x i32> to 1) ensure the zero vectors are CSE'd
    // and 2) ensure that i64 scalars are eliminated on x86-32 hosts.
    if (VT == MVT::v4i32 || VT == MVT::v8i32)
      return Op;

    return getZeroVector(VT, Subtarget, DAG, dl);
  }

  // Vectors containing all ones can be matched by pcmpeqd on 128-bit width
  // vectors or broken into v4i32 operations on 256-bit vectors. AVX2 can use
  // vpcmpeqd on 256-bit vectors.
  if (ISD::isBuildVectorAllOnes(Op.getNode())) {
    if (VT == MVT::v4i32 || (VT == MVT::v8i32 && Subtarget->hasAVX2()))
      return Op;

    return getOnesVector(VT, Subtarget->hasAVX2(), DAG, dl);
  }

  SDValue Broadcast = LowerVectorBroadcast(Op, DAG);
  if (Broadcast.getNode())
    return Broadcast;

  unsigned EVTBits = ExtVT.getSizeInBits();

  unsigned NumZero  = 0;
  unsigned NumNonZero = 0;
  unsigned NonZeros = 0;
  bool IsAllConstants = true;
  SmallSet<SDValue, 8> Values;
  for (unsigned i = 0; i < NumElems; ++i) {
    SDValue Elt = Op.getOperand(i);
    if (Elt.getOpcode() == ISD::UNDEF)
      continue;
    Values.insert(Elt);
    if (Elt.getOpcode() != ISD::Constant &&
        Elt.getOpcode() != ISD::ConstantFP)
      IsAllConstants = false;
    if (X86::isZeroNode(Elt))
      NumZero++;
    else {
      NonZeros |= (1 << i);
      NumNonZero++;
    }
  }

  // All undef vector. Return an UNDEF.  All zero vectors were handled above.
  if (NumNonZero == 0)
    return DAG.getUNDEF(VT);

  // Special case for single non-zero, non-undef, element.
  if (NumNonZero == 1) {
    unsigned Idx = CountTrailingZeros_32(NonZeros);
    SDValue Item = Op.getOperand(Idx);

    // If this is an insertion of an i64 value on x86-32, and if the top bits of
    // the value are obviously zero, truncate the value to i32 and do the
    // insertion that way.  Only do this if the value is non-constant or if the
    // value is a constant being inserted into element 0.  It is cheaper to do
    // a constant pool load than it is to do a movd + shuffle.
    if (ExtVT == MVT::i64 && !Subtarget->is64Bit() &&
        (!IsAllConstants || Idx == 0)) {
      if (DAG.MaskedValueIsZero(Item, APInt::getBitsSet(64, 32, 64))) {
        // Handle SSE only.
        assert(VT == MVT::v2i64 && "Expected an SSE value type!");
        EVT VecVT = MVT::v4i32;
        unsigned VecElts = 4;

        // Truncate the value (which may itself be a constant) to i32, and
        // convert it to a vector with movd (S2V+shuffle to zero extend).
        Item = DAG.getNode(ISD::TRUNCATE, dl, MVT::i32, Item);
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VecVT, Item);
        Item = getShuffleVectorZeroOrUndef(Item, 0, true, Subtarget, DAG);

        // Now we have our 32-bit value zero extended in the low element of
        // a vector.  If Idx != 0, swizzle it into place.
        if (Idx != 0) {
          SmallVector<int, 4> Mask;
          Mask.push_back(Idx);
          for (unsigned i = 1; i != VecElts; ++i)
            Mask.push_back(i);
          Item = DAG.getVectorShuffle(VecVT, dl, Item, DAG.getUNDEF(VecVT),
                                      &Mask[0]);
        }
        return DAG.getNode(ISD::BITCAST, dl, VT, Item);
      }
    }

    // If we have a constant or non-constant insertion into the low element of
    // a vector, we can do this with SCALAR_TO_VECTOR + shuffle of zero into
    // the rest of the elements.  This will be matched as movd/movq/movss/movsd
    // depending on what the source datatype is.
    if (Idx == 0) {
      if (NumZero == 0)
        return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Item);

      if (ExtVT == MVT::i32 || ExtVT == MVT::f32 || ExtVT == MVT::f64 ||
          (ExtVT == MVT::i64 && Subtarget->is64Bit())) {
        if (VT.is256BitVector()) {
          SDValue ZeroVec = getZeroVector(VT, Subtarget, DAG, dl);
          return DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, ZeroVec,
                             Item, DAG.getIntPtrConstant(0));
        }
        assert(VT.is128BitVector() && "Expected an SSE value type!");
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Item);
        // Turn it into a MOVL (i.e. movss, movsd, or movd) to a zero vector.
        return getShuffleVectorZeroOrUndef(Item, 0, true, Subtarget, DAG);
      }

      if (ExtVT == MVT::i16 || ExtVT == MVT::i8) {
        Item = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Item);
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4i32, Item);
        if (VT.is256BitVector()) {
          SDValue ZeroVec = getZeroVector(MVT::v8i32, Subtarget, DAG, dl);
          Item = Insert128BitVector(ZeroVec, Item, 0, DAG, dl);
        } else {
          assert(VT.is128BitVector() && "Expected an SSE value type!");
          Item = getShuffleVectorZeroOrUndef(Item, 0, true, Subtarget, DAG);
        }
        return DAG.getNode(ISD::BITCAST, dl, VT, Item);
      }
    }

    // Is it a vector logical left shift?
    if (NumElems == 2 && Idx == 1 &&
        X86::isZeroNode(Op.getOperand(0)) &&
        !X86::isZeroNode(Op.getOperand(1))) {
      unsigned NumBits = VT.getSizeInBits();
      return getVShift(true, VT,
                       DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                   VT, Op.getOperand(1)),
                       NumBits/2, DAG, *this, dl);
    }

    if (IsAllConstants) // Otherwise, it's better to do a constpool load.
      return SDValue();

    // Otherwise, if this is a vector with i32 or f32 elements, and the element
    // is a non-constant being inserted into an element other than the low one,
    // we can't use a constant pool load.  Instead, use SCALAR_TO_VECTOR (aka
    // movd/movss) to move this into the low element, then shuffle it into
    // place.
    if (EVTBits == 32) {
      Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Item);

      // Turn it into a shuffle of zero and zero-extended scalar to vector.
      Item = getShuffleVectorZeroOrUndef(Item, 0, NumZero > 0, Subtarget, DAG);
      SmallVector<int, 8> MaskVec;
      for (unsigned i = 0; i != NumElems; ++i)
        MaskVec.push_back(i == Idx ? 0 : 1);
      return DAG.getVectorShuffle(VT, dl, Item, DAG.getUNDEF(VT), &MaskVec[0]);
    }
  }

  // Splat is obviously ok. Let legalizer expand it to a shuffle.
  if (Values.size() == 1) {
    if (EVTBits == 32) {
      // Instead of a shuffle like this:
      // shuffle (scalar_to_vector (load (ptr + 4))), undef, <0, 0, 0, 0>
      // Check if it's possible to issue this instead.
      // shuffle (vload ptr)), undef, <1, 1, 1, 1>
      unsigned Idx = CountTrailingZeros_32(NonZeros);
      SDValue Item = Op.getOperand(Idx);
      if (Op.getNode()->isOnlyUserOf(Item.getNode()))
        return LowerAsSplatVectorLoad(Item, VT, dl, DAG);
    }
    return SDValue();
  }

  // A vector full of immediates; various special cases are already
  // handled, so this is best done with a single constant-pool load.
  if (IsAllConstants)
    return SDValue();

  // For AVX-length vectors, build the individual 128-bit pieces and use
  // shuffles to put them in place.
  if (VT.is256BitVector()) {
    SmallVector<SDValue, 32> V;
    for (unsigned i = 0; i != NumElems; ++i)
      V.push_back(Op.getOperand(i));

    EVT HVT = EVT::getVectorVT(*DAG.getContext(), ExtVT, NumElems/2);

    // Build both the lower and upper subvector.
    SDValue Lower = DAG.getNode(ISD::BUILD_VECTOR, dl, HVT, &V[0], NumElems/2);
    SDValue Upper = DAG.getNode(ISD::BUILD_VECTOR, dl, HVT, &V[NumElems / 2],
                                NumElems/2);

    // Recreate the wider vector with the lower and upper part.
    return Concat128BitVectors(Lower, Upper, VT, NumElems, DAG, dl);
  }

  // Let legalizer expand 2-wide build_vectors.
  if (EVTBits == 64) {
    if (NumNonZero == 1) {
      // One half is zero or undef.
      unsigned Idx = CountTrailingZeros_32(NonZeros);
      SDValue V2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT,
                                 Op.getOperand(Idx));
      return getShuffleVectorZeroOrUndef(V2, Idx, true, Subtarget, DAG);
    }
    return SDValue();
  }

  // If element VT is < 32 bits, convert it to inserts into a zero vector.
  if (EVTBits == 8 && NumElems == 16) {
    SDValue V = LowerBuildVectorv16i8(Op, NonZeros,NumNonZero,NumZero, DAG,
                                        Subtarget, *this);
    if (V.getNode()) return V;
  }

  if (EVTBits == 16 && NumElems == 8) {
    SDValue V = LowerBuildVectorv8i16(Op, NonZeros,NumNonZero,NumZero, DAG,
                                      Subtarget, *this);
    if (V.getNode()) return V;
  }

  // If element VT is == 32 bits, turn it into a number of shuffles.
  SmallVector<SDValue, 8> V(NumElems);
  if (NumElems == 4 && NumZero > 0) {
    for (unsigned i = 0; i < 4; ++i) {
      bool isZero = !(NonZeros & (1 << i));
      if (isZero)
        V[i] = getZeroVector(VT, Subtarget, DAG, dl);
      else
        V[i] = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Op.getOperand(i));
    }

    for (unsigned i = 0; i < 2; ++i) {
      switch ((NonZeros & (0x3 << i*2)) >> (i*2)) {
        default: break;
        case 0:
          V[i] = V[i*2];  // Must be a zero vector.
          break;
        case 1:
          V[i] = getMOVL(DAG, dl, VT, V[i*2+1], V[i*2]);
          break;
        case 2:
          V[i] = getMOVL(DAG, dl, VT, V[i*2], V[i*2+1]);
          break;
        case 3:
          V[i] = getUnpackl(DAG, dl, VT, V[i*2], V[i*2+1]);
          break;
      }
    }

    bool Reverse1 = (NonZeros & 0x3) == 2;
    bool Reverse2 = ((NonZeros & (0x3 << 2)) >> 2) == 2;
    int MaskVec[] = {
      Reverse1 ? 1 : 0,
      Reverse1 ? 0 : 1,
      static_cast<int>(Reverse2 ? NumElems+1 : NumElems),
      static_cast<int>(Reverse2 ? NumElems   : NumElems+1)
    };
    return DAG.getVectorShuffle(VT, dl, V[0], V[1], &MaskVec[0]);
  }

  if (Values.size() > 1 && VT.is128BitVector()) {
    // Check for a build vector of consecutive loads.
    for (unsigned i = 0; i < NumElems; ++i)
      V[i] = Op.getOperand(i);

    // Check for elements which are consecutive loads.
    SDValue LD = EltsFromConsecutiveLoads(VT, V, dl, DAG);
    if (LD.getNode())
      return LD;

    // Check for a build vector from mostly shuffle plus few inserting.
    SDValue Sh = buildFromShuffleMostly(Op, DAG);
    if (Sh.getNode())
      return Sh;

    // For SSE 4.1, use insertps to put the high elements into the low element.
    if (getSubtarget()->hasSSE41()) {
      SDValue Result;
      if (Op.getOperand(0).getOpcode() != ISD::UNDEF)
        Result = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Op.getOperand(0));
      else
        Result = DAG.getUNDEF(VT);

      for (unsigned i = 1; i < NumElems; ++i) {
        if (Op.getOperand(i).getOpcode() == ISD::UNDEF) continue;
        Result = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, Result,
                             Op.getOperand(i), DAG.getIntPtrConstant(i));
      }
      return Result;
    }

    // Otherwise, expand into a number of unpckl*, start by extending each of
    // our (non-undef) elements to the full vector width with the element in the
    // bottom slot of the vector (which generates no code for SSE).
    for (unsigned i = 0; i < NumElems; ++i) {
      if (Op.getOperand(i).getOpcode() != ISD::UNDEF)
        V[i] = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Op.getOperand(i));
      else
        V[i] = DAG.getUNDEF(VT);
    }

    // Next, we iteratively mix elements, e.g. for v4f32:
    //   Step 1: unpcklps 0, 2 ==> X: <?, ?, 2, 0>
    //         : unpcklps 1, 3 ==> Y: <?, ?, 3, 1>
    //   Step 2: unpcklps X, Y ==>    <3, 2, 1, 0>
    unsigned EltStride = NumElems >> 1;
    while (EltStride != 0) {
      for (unsigned i = 0; i < EltStride; ++i) {
        // If V[i+EltStride] is undef and this is the first round of mixing,
        // then it is safe to just drop this shuffle: V[i] is already in the
        // right place, the one element (since it's the first round) being
        // inserted as undef can be dropped.  This isn't safe for successive
        // rounds because they will permute elements within both vectors.
        if (V[i+EltStride].getOpcode() == ISD::UNDEF &&
            EltStride == NumElems/2)
          continue;

        V[i] = getUnpackl(DAG, dl, VT, V[i], V[i + EltStride]);
      }
      EltStride >>= 1;
    }
    return V[0];
  }
  return SDValue();
}

// LowerAVXCONCAT_VECTORS - 256-bit AVX can use the vinsertf128 instruction
// to create 256-bit vectors from two other 128-bit ones.
static SDValue LowerAVXCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  EVT ResVT = Op.getValueType();

  assert(ResVT.is256BitVector() && "Value type must be 256-bit wide");

  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  unsigned NumElems = ResVT.getVectorNumElements();

  return Concat128BitVectors(V1, V2, ResVT, NumElems, DAG, dl);
}

static SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) {
  assert(Op.getNumOperands() == 2);

  // 256-bit AVX can use the vinsertf128 instruction to create 256-bit vectors
  // from two other 128-bit ones.
  return LowerAVXCONCAT_VECTORS(Op, DAG);
}

// Try to lower a shuffle node into a simple blend instruction.
static SDValue
LowerVECTOR_SHUFFLEtoBlend(ShuffleVectorSDNode *SVOp,
                           const X86Subtarget *Subtarget, SelectionDAG &DAG) {
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  DebugLoc dl = SVOp->getDebugLoc();
  MVT VT = SVOp->getValueType(0).getSimpleVT();
  unsigned NumElems = VT.getVectorNumElements();

  if (!Subtarget->hasSSE41())
    return SDValue();

  unsigned ISDNo = 0;
  MVT OpTy;

  switch (VT.SimpleTy) {
  default: return SDValue();
  case MVT::v8i16:
    ISDNo = X86ISD::BLENDPW;
    OpTy = MVT::v8i16;
    break;
  case MVT::v4i32:
  case MVT::v4f32:
    ISDNo = X86ISD::BLENDPS;
    OpTy = MVT::v4f32;
    break;
  case MVT::v2i64:
  case MVT::v2f64:
    ISDNo = X86ISD::BLENDPD;
    OpTy = MVT::v2f64;
    break;
  case MVT::v8i32:
  case MVT::v8f32:
    if (!Subtarget->hasAVX())
      return SDValue();
    ISDNo = X86ISD::BLENDPS;
    OpTy = MVT::v8f32;
    break;
  case MVT::v4i64:
  case MVT::v4f64:
    if (!Subtarget->hasAVX())
      return SDValue();
    ISDNo = X86ISD::BLENDPD;
    OpTy = MVT::v4f64;
    break;
  }
  assert(ISDNo && "Invalid Op Number");

  unsigned MaskVals = 0;

  for (unsigned i = 0; i != NumElems; ++i) {
    int EltIdx = SVOp->getMaskElt(i);
    if (EltIdx == (int)i || EltIdx < 0)
      MaskVals |= (1<<i);
    else if (EltIdx == (int)(i + NumElems))
      continue; // Bit is set to zero;
    else
      return SDValue();
  }

  V1 = DAG.getNode(ISD::BITCAST, dl, OpTy, V1);
  V2 = DAG.getNode(ISD::BITCAST, dl, OpTy, V2);
  SDValue Ret =  DAG.getNode(ISDNo, dl, OpTy, V1, V2,
                             DAG.getConstant(MaskVals, MVT::i32));
  return DAG.getNode(ISD::BITCAST, dl, VT, Ret);
}

// v8i16 shuffles - Prefer shuffles in the following order:
// 1. [all]   pshuflw, pshufhw, optional move
// 2. [ssse3] 1 x pshufb
// 3. [ssse3] 2 x pshufb + 1 x por
// 4. [all]   mov + pshuflw + pshufhw + N x (pextrw + pinsrw)
static SDValue
LowerVECTOR_SHUFFLEv8i16(SDValue Op, const X86Subtarget *Subtarget,
                         SelectionDAG &DAG) {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  DebugLoc dl = SVOp->getDebugLoc();
  SmallVector<int, 8> MaskVals;

  // Determine if more than 1 of the words in each of the low and high quadwords
  // of the result come from the same quadword of one of the two inputs.  Undef
  // mask values count as coming from any quadword, for better codegen.
  unsigned LoQuad[] = { 0, 0, 0, 0 };
  unsigned HiQuad[] = { 0, 0, 0, 0 };
  std::bitset<4> InputQuads;
  for (unsigned i = 0; i < 8; ++i) {
    unsigned *Quad = i < 4 ? LoQuad : HiQuad;
    int EltIdx = SVOp->getMaskElt(i);
    MaskVals.push_back(EltIdx);
    if (EltIdx < 0) {
      ++Quad[0];
      ++Quad[1];
      ++Quad[2];
      ++Quad[3];
      continue;
    }
    ++Quad[EltIdx / 4];
    InputQuads.set(EltIdx / 4);
  }

  int BestLoQuad = -1;
  unsigned MaxQuad = 1;
  for (unsigned i = 0; i < 4; ++i) {
    if (LoQuad[i] > MaxQuad) {
      BestLoQuad = i;
      MaxQuad = LoQuad[i];
    }
  }

  int BestHiQuad = -1;
  MaxQuad = 1;
  for (unsigned i = 0; i < 4; ++i) {
    if (HiQuad[i] > MaxQuad) {
      BestHiQuad = i;
      MaxQuad = HiQuad[i];
    }
  }

  // For SSSE3, If all 8 words of the result come from only 1 quadword of each
  // of the two input vectors, shuffle them into one input vector so only a
  // single pshufb instruction is necessary. If There are more than 2 input
  // quads, disable the next transformation since it does not help SSSE3.
  bool V1Used = InputQuads[0] || InputQuads[1];
  bool V2Used = InputQuads[2] || InputQuads[3];
  if (Subtarget->hasSSSE3()) {
    if (InputQuads.count() == 2 && V1Used && V2Used) {
      BestLoQuad = InputQuads[0] ? 0 : 1;
      BestHiQuad = InputQuads[2] ? 2 : 3;
    }
    if (InputQuads.count() > 2) {
      BestLoQuad = -1;
      BestHiQuad = -1;
    }
  }

  // If BestLoQuad or BestHiQuad are set, shuffle the quads together and update
  // the shuffle mask.  If a quad is scored as -1, that means that it contains
  // words from all 4 input quadwords.
  SDValue NewV;
  if (BestLoQuad >= 0 || BestHiQuad >= 0) {
    int MaskV[] = {
      BestLoQuad < 0 ? 0 : BestLoQuad,
      BestHiQuad < 0 ? 1 : BestHiQuad
    };
    NewV = DAG.getVectorShuffle(MVT::v2i64, dl,
                  DAG.getNode(ISD::BITCAST, dl, MVT::v2i64, V1),
                  DAG.getNode(ISD::BITCAST, dl, MVT::v2i64, V2), &MaskV[0]);
    NewV = DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, NewV);

    // Rewrite the MaskVals and assign NewV to V1 if NewV now contains all the
    // source words for the shuffle, to aid later transformations.
    bool AllWordsInNewV = true;
    bool InOrder[2] = { true, true };
    for (unsigned i = 0; i != 8; ++i) {
      int idx = MaskVals[i];
      if (idx != (int)i)
        InOrder[i/4] = false;
      if (idx < 0 || (idx/4) == BestLoQuad || (idx/4) == BestHiQuad)
        continue;
      AllWordsInNewV = false;
      break;
    }

    bool pshuflw = AllWordsInNewV, pshufhw = AllWordsInNewV;
    if (AllWordsInNewV) {
      for (int i = 0; i != 8; ++i) {
        int idx = MaskVals[i];
        if (idx < 0)
          continue;
        idx = MaskVals[i] = (idx / 4) == BestLoQuad ? (idx & 3) : (idx & 3) + 4;
        if ((idx != i) && idx < 4)
          pshufhw = false;
        if ((idx != i) && idx > 3)
          pshuflw = false;
      }
      V1 = NewV;
      V2Used = false;
      BestLoQuad = 0;
      BestHiQuad = 1;
    }

    // If we've eliminated the use of V2, and the new mask is a pshuflw or
    // pshufhw, that's as cheap as it gets.  Return the new shuffle.
    if ((pshufhw && InOrder[0]) || (pshuflw && InOrder[1])) {
      unsigned Opc = pshufhw ? X86ISD::PSHUFHW : X86ISD::PSHUFLW;
      unsigned TargetMask = 0;
      NewV = DAG.getVectorShuffle(MVT::v8i16, dl, NewV,
                                  DAG.getUNDEF(MVT::v8i16), &MaskVals[0]);
      ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(NewV.getNode());
      TargetMask = pshufhw ? getShufflePSHUFHWImmediate(SVOp):
                             getShufflePSHUFLWImmediate(SVOp);
      V1 = NewV.getOperand(0);
      return getTargetShuffleNode(Opc, dl, MVT::v8i16, V1, TargetMask, DAG);
    }
  }

  // If we have SSSE3, and all words of the result are from 1 input vector,
  // case 2 is generated, otherwise case 3 is generated.  If no SSSE3
  // is present, fall back to case 4.
  if (Subtarget->hasSSSE3()) {
    SmallVector<SDValue,16> pshufbMask;

    // If we have elements from both input vectors, set the high bit of the
    // shuffle mask element to zero out elements that come from V2 in the V1
    // mask, and elements that come from V1 in the V2 mask, so that the two
    // results can be OR'd together.
    bool TwoInputs = V1Used && V2Used;
    for (unsigned i = 0; i != 8; ++i) {
      int EltIdx = MaskVals[i] * 2;
      int Idx0 = (TwoInputs && (EltIdx >= 16)) ? 0x80 : EltIdx;
      int Idx1 = (TwoInputs && (EltIdx >= 16)) ? 0x80 : EltIdx+1;
      pshufbMask.push_back(DAG.getConstant(Idx0,   MVT::i8));
      pshufbMask.push_back(DAG.getConstant(Idx1, MVT::i8));
    }
    V1 = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, V1);
    V1 = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v16i8, V1,
                     DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::v16i8, &pshufbMask[0], 16));
    if (!TwoInputs)
      return DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, V1);

    // Calculate the shuffle mask for the second input, shuffle it, and
    // OR it with the first shuffled input.
    pshufbMask.clear();
    for (unsigned i = 0; i != 8; ++i) {
      int EltIdx = MaskVals[i] * 2;
      int Idx0 = (EltIdx < 16) ? 0x80 : EltIdx - 16;
      int Idx1 = (EltIdx < 16) ? 0x80 : EltIdx - 15;
      pshufbMask.push_back(DAG.getConstant(Idx0, MVT::i8));
      pshufbMask.push_back(DAG.getConstant(Idx1, MVT::i8));
    }
    V2 = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, V2);
    V2 = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v16i8, V2,
                     DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::v16i8, &pshufbMask[0], 16));
    V1 = DAG.getNode(ISD::OR, dl, MVT::v16i8, V1, V2);
    return DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, V1);
  }

  // If BestLoQuad >= 0, generate a pshuflw to put the low elements in order,
  // and update MaskVals with new element order.
  std::bitset<8> InOrder;
  if (BestLoQuad >= 0) {
    int MaskV[] = { -1, -1, -1, -1, 4, 5, 6, 7 };
    for (int i = 0; i != 4; ++i) {
      int idx = MaskVals[i];
      if (idx < 0) {
        InOrder.set(i);
      } else if ((idx / 4) == BestLoQuad) {
        MaskV[i] = idx & 3;
        InOrder.set(i);
      }
    }
    NewV = DAG.getVectorShuffle(MVT::v8i16, dl, NewV, DAG.getUNDEF(MVT::v8i16),
                                &MaskV[0]);

    if (NewV.getOpcode() == ISD::VECTOR_SHUFFLE && Subtarget->hasSSSE3()) {
      ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(NewV.getNode());
      NewV = getTargetShuffleNode(X86ISD::PSHUFLW, dl, MVT::v8i16,
                                  NewV.getOperand(0),
                                  getShufflePSHUFLWImmediate(SVOp), DAG);
    }
  }

  // If BestHi >= 0, generate a pshufhw to put the high elements in order,
  // and update MaskVals with the new element order.
  if (BestHiQuad >= 0) {
    int MaskV[] = { 0, 1, 2, 3, -1, -1, -1, -1 };
    for (unsigned i = 4; i != 8; ++i) {
      int idx = MaskVals[i];
      if (idx < 0) {
        InOrder.set(i);
      } else if ((idx / 4) == BestHiQuad) {
        MaskV[i] = (idx & 3) + 4;
        InOrder.set(i);
      }
    }
    NewV = DAG.getVectorShuffle(MVT::v8i16, dl, NewV, DAG.getUNDEF(MVT::v8i16),
                                &MaskV[0]);

    if (NewV.getOpcode() == ISD::VECTOR_SHUFFLE && Subtarget->hasSSSE3()) {
      ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(NewV.getNode());
      NewV = getTargetShuffleNode(X86ISD::PSHUFHW, dl, MVT::v8i16,
                                  NewV.getOperand(0),
                                  getShufflePSHUFHWImmediate(SVOp), DAG);
    }
  }

  // In case BestHi & BestLo were both -1, which means each quadword has a word
  // from each of the four input quadwords, calculate the InOrder bitvector now
  // before falling through to the insert/extract cleanup.
  if (BestLoQuad == -1 && BestHiQuad == -1) {
    NewV = V1;
    for (int i = 0; i != 8; ++i)
      if (MaskVals[i] < 0 || MaskVals[i] == i)
        InOrder.set(i);
  }

  // The other elements are put in the right place using pextrw and pinsrw.
  for (unsigned i = 0; i != 8; ++i) {
    if (InOrder[i])
      continue;
    int EltIdx = MaskVals[i];
    if (EltIdx < 0)
      continue;
    SDValue ExtOp = (EltIdx < 8) ?
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16, V1,
                  DAG.getIntPtrConstant(EltIdx)) :
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16, V2,
                  DAG.getIntPtrConstant(EltIdx - 8));
    NewV = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v8i16, NewV, ExtOp,
                       DAG.getIntPtrConstant(i));
  }
  return NewV;
}

// v16i8 shuffles - Prefer shuffles in the following order:
// 1. [ssse3] 1 x pshufb
// 2. [ssse3] 2 x pshufb + 1 x por
// 3. [all]   v8i16 shuffle + N x pextrw + rotate + pinsrw
static
SDValue LowerVECTOR_SHUFFLEv16i8(ShuffleVectorSDNode *SVOp,
                                 SelectionDAG &DAG,
                                 const X86TargetLowering &TLI) {
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  DebugLoc dl = SVOp->getDebugLoc();
  ArrayRef<int> MaskVals = SVOp->getMask();

  // If we have SSSE3, case 1 is generated when all result bytes come from
  // one of  the inputs.  Otherwise, case 2 is generated.  If no SSSE3 is
  // present, fall back to case 3.

  // If SSSE3, use 1 pshufb instruction per vector with elements in the result.
  if (TLI.getSubtarget()->hasSSSE3()) {
    SmallVector<SDValue,16> pshufbMask;

    // If all result elements are from one input vector, then only translate
    // undef mask values to 0x80 (zero out result) in the pshufb mask.
    //
    // Otherwise, we have elements from both input vectors, and must zero out
    // elements that come from V2 in the first mask, and V1 in the second mask
    // so that we can OR them together.
    for (unsigned i = 0; i != 16; ++i) {
      int EltIdx = MaskVals[i];
      if (EltIdx < 0 || EltIdx >= 16)
        EltIdx = 0x80;
      pshufbMask.push_back(DAG.getConstant(EltIdx, MVT::i8));
    }
    V1 = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v16i8, V1,
                     DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::v16i8, &pshufbMask[0], 16));

    // As PSHUFB will zero elements with negative indices, it's safe to ignore
    // the 2nd operand if it's undefined or zero.
    if (V2.getOpcode() == ISD::UNDEF ||
        ISD::isBuildVectorAllZeros(V2.getNode()))
      return V1;

    // Calculate the shuffle mask for the second input, shuffle it, and
    // OR it with the first shuffled input.
    pshufbMask.clear();
    for (unsigned i = 0; i != 16; ++i) {
      int EltIdx = MaskVals[i];
      EltIdx = (EltIdx < 16) ? 0x80 : EltIdx - 16;
      pshufbMask.push_back(DAG.getConstant(EltIdx, MVT::i8));
    }
    V2 = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v16i8, V2,
                     DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::v16i8, &pshufbMask[0], 16));
    return DAG.getNode(ISD::OR, dl, MVT::v16i8, V1, V2);
  }

  // No SSSE3 - Calculate in place words and then fix all out of place words
  // With 0-16 extracts & inserts.  Worst case is 16 bytes out of order from
  // the 16 different words that comprise the two doublequadword input vectors.
  V1 = DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, V1);
  V2 = DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, V2);
  SDValue NewV = V1;
  for (int i = 0; i != 8; ++i) {
    int Elt0 = MaskVals[i*2];
    int Elt1 = MaskVals[i*2+1];

    // This word of the result is all undef, skip it.
    if (Elt0 < 0 && Elt1 < 0)
      continue;

    // This word of the result is already in the correct place, skip it.
    if ((Elt0 == i*2) && (Elt1 == i*2+1))
      continue;

    SDValue Elt0Src = Elt0 < 16 ? V1 : V2;
    SDValue Elt1Src = Elt1 < 16 ? V1 : V2;
    SDValue InsElt;

    // If Elt0 and Elt1 are defined, are consecutive, and can be load
    // using a single extract together, load it and store it.
    if ((Elt0 >= 0) && ((Elt0 + 1) == Elt1) && ((Elt0 & 1) == 0)) {
      InsElt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16, Elt1Src,
                           DAG.getIntPtrConstant(Elt1 / 2));
      NewV = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v8i16, NewV, InsElt,
                        DAG.getIntPtrConstant(i));
      continue;
    }

    // If Elt1 is defined, extract it from the appropriate source.  If the
    // source byte is not also odd, shift the extracted word left 8 bits
    // otherwise clear the bottom 8 bits if we need to do an or.
    if (Elt1 >= 0) {
      InsElt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16, Elt1Src,
                           DAG.getIntPtrConstant(Elt1 / 2));
      if ((Elt1 & 1) == 0)
        InsElt = DAG.getNode(ISD::SHL, dl, MVT::i16, InsElt,
                             DAG.getConstant(8,
                                  TLI.getShiftAmountTy(InsElt.getValueType())));
      else if (Elt0 >= 0)
        InsElt = DAG.getNode(ISD::AND, dl, MVT::i16, InsElt,
                             DAG.getConstant(0xFF00, MVT::i16));
    }
    // If Elt0 is defined, extract it from the appropriate source.  If the
    // source byte is not also even, shift the extracted word right 8 bits. If
    // Elt1 was also defined, OR the extracted values together before
    // inserting them in the result.
    if (Elt0 >= 0) {
      SDValue InsElt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16,
                                    Elt0Src, DAG.getIntPtrConstant(Elt0 / 2));
      if ((Elt0 & 1) != 0)
        InsElt0 = DAG.getNode(ISD::SRL, dl, MVT::i16, InsElt0,
                              DAG.getConstant(8,
                                 TLI.getShiftAmountTy(InsElt0.getValueType())));
      else if (Elt1 >= 0)
        InsElt0 = DAG.getNode(ISD::AND, dl, MVT::i16, InsElt0,
                             DAG.getConstant(0x00FF, MVT::i16));
      InsElt = Elt1 >= 0 ? DAG.getNode(ISD::OR, dl, MVT::i16, InsElt, InsElt0)
                         : InsElt0;
    }
    NewV = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v8i16, NewV, InsElt,
                       DAG.getIntPtrConstant(i));
  }
  return DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, NewV);
}

// v32i8 shuffles - Translate to VPSHUFB if possible.
static
SDValue LowerVECTOR_SHUFFLEv32i8(ShuffleVectorSDNode *SVOp,
                                 const X86Subtarget *Subtarget,
                                 SelectionDAG &DAG) {
  EVT VT = SVOp->getValueType(0);
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  DebugLoc dl = SVOp->getDebugLoc();
  SmallVector<int, 32> MaskVals(SVOp->getMask().begin(), SVOp->getMask().end());

  bool V2IsUndef = V2.getOpcode() == ISD::UNDEF;
  bool V1IsAllZero = ISD::isBuildVectorAllZeros(V1.getNode());
  bool V2IsAllZero = ISD::isBuildVectorAllZeros(V2.getNode());

  // VPSHUFB may be generated if
  // (1) one of input vector is undefined or zeroinitializer.
  // The mask value 0x80 puts 0 in the corresponding slot of the vector.
  // And (2) the mask indexes don't cross the 128-bit lane.
  if (VT != MVT::v32i8 || !Subtarget->hasAVX2() ||
      (!V2IsUndef && !V2IsAllZero && !V1IsAllZero))
    return SDValue();

  if (V1IsAllZero && !V2IsAllZero) {
    CommuteVectorShuffleMask(MaskVals, 32);
    V1 = V2;
  }
  SmallVector<SDValue, 32> pshufbMask;
  for (unsigned i = 0; i != 32; i++) {
    int EltIdx = MaskVals[i];
    if (EltIdx < 0 || EltIdx >= 32)
      EltIdx = 0x80;
    else {
      if ((EltIdx >= 16 && i < 16) || (EltIdx < 16 && i >= 16))
        // Cross lane is not allowed.
        return SDValue();
      EltIdx &= 0xf;
    }
    pshufbMask.push_back(DAG.getConstant(EltIdx, MVT::i8));
  }
  return DAG.getNode(X86ISD::PSHUFB, dl, MVT::v32i8, V1,
                      DAG.getNode(ISD::BUILD_VECTOR, dl,
                                  MVT::v32i8, &pshufbMask[0], 32));
}

/// RewriteAsNarrowerShuffle - Try rewriting v8i16 and v16i8 shuffles as 4 wide
/// ones, or rewriting v4i32 / v4f32 as 2 wide ones if possible. This can be
/// done when every pair / quad of shuffle mask elements point to elements in
/// the right sequence. e.g.
/// vector_shuffle X, Y, <2, 3, | 10, 11, | 0, 1, | 14, 15>
static
SDValue RewriteAsNarrowerShuffle(ShuffleVectorSDNode *SVOp,
                                 SelectionDAG &DAG, DebugLoc dl) {
  MVT VT = SVOp->getValueType(0).getSimpleVT();
  unsigned NumElems = VT.getVectorNumElements();
  MVT NewVT;
  unsigned Scale;
  switch (VT.SimpleTy) {
  default: llvm_unreachable("Unexpected!");
  case MVT::v4f32:  NewVT = MVT::v2f64; Scale = 2; break;
  case MVT::v4i32:  NewVT = MVT::v2i64; Scale = 2; break;
  case MVT::v8i16:  NewVT = MVT::v4i32; Scale = 2; break;
  case MVT::v16i8:  NewVT = MVT::v4i32; Scale = 4; break;
  case MVT::v16i16: NewVT = MVT::v8i32; Scale = 2; break;
  case MVT::v32i8:  NewVT = MVT::v8i32; Scale = 4; break;
  }

  SmallVector<int, 8> MaskVec;
  for (unsigned i = 0; i != NumElems; i += Scale) {
    int StartIdx = -1;
    for (unsigned j = 0; j != Scale; ++j) {
      int EltIdx = SVOp->getMaskElt(i+j);
      if (EltIdx < 0)
        continue;
      if (StartIdx < 0)
        StartIdx = (EltIdx / Scale);
      if (EltIdx != (int)(StartIdx*Scale + j))
        return SDValue();
    }
    MaskVec.push_back(StartIdx);
  }

  SDValue V1 = DAG.getNode(ISD::BITCAST, dl, NewVT, SVOp->getOperand(0));
  SDValue V2 = DAG.getNode(ISD::BITCAST, dl, NewVT, SVOp->getOperand(1));
  return DAG.getVectorShuffle(NewVT, dl, V1, V2, &MaskVec[0]);
}

/// getVZextMovL - Return a zero-extending vector move low node.
///
static SDValue getVZextMovL(EVT VT, EVT OpVT,
                            SDValue SrcOp, SelectionDAG &DAG,
                            const X86Subtarget *Subtarget, DebugLoc dl) {
  if (VT == MVT::v2f64 || VT == MVT::v4f32) {
    LoadSDNode *LD = NULL;
    if (!isScalarLoadToVector(SrcOp.getNode(), &LD))
      LD = dyn_cast<LoadSDNode>(SrcOp);
    if (!LD) {
      // movssrr and movsdrr do not clear top bits. Try to use movd, movq
      // instead.
      MVT ExtVT = (OpVT == MVT::v2f64) ? MVT::i64 : MVT::i32;
      if ((ExtVT != MVT::i64 || Subtarget->is64Bit()) &&
          SrcOp.getOpcode() == ISD::SCALAR_TO_VECTOR &&
          SrcOp.getOperand(0).getOpcode() == ISD::BITCAST &&
          SrcOp.getOperand(0).getOperand(0).getValueType() == ExtVT) {
        // PR2108
        OpVT = (OpVT == MVT::v2f64) ? MVT::v2i64 : MVT::v4i32;
        return DAG.getNode(ISD::BITCAST, dl, VT,
                           DAG.getNode(X86ISD::VZEXT_MOVL, dl, OpVT,
                                       DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                                   OpVT,
                                                   SrcOp.getOperand(0)
                                                          .getOperand(0))));
      }
    }
  }

  return DAG.getNode(ISD::BITCAST, dl, VT,
                     DAG.getNode(X86ISD::VZEXT_MOVL, dl, OpVT,
                                 DAG.getNode(ISD::BITCAST, dl,
                                             OpVT, SrcOp)));
}

/// LowerVECTOR_SHUFFLE_256 - Handle all 256-bit wide vectors shuffles
/// which could not be matched by any known target speficic shuffle
static SDValue
LowerVECTOR_SHUFFLE_256(ShuffleVectorSDNode *SVOp, SelectionDAG &DAG) {

  SDValue NewOp = Compact8x32ShuffleNode(SVOp, DAG);
  if (NewOp.getNode())
    return NewOp;

  EVT VT = SVOp->getValueType(0);

  unsigned NumElems = VT.getVectorNumElements();
  unsigned NumLaneElems = NumElems / 2;

  DebugLoc dl = SVOp->getDebugLoc();
  MVT EltVT = VT.getVectorElementType().getSimpleVT();
  EVT NVT = MVT::getVectorVT(EltVT, NumLaneElems);
  SDValue Output[2];

  SmallVector<int, 16> Mask;
  for (unsigned l = 0; l < 2; ++l) {
    // Build a shuffle mask for the output, discovering on the fly which
    // input vectors to use as shuffle operands (recorded in InputUsed).
    // If building a suitable shuffle vector proves too hard, then bail
    // out with UseBuildVector set.
    bool UseBuildVector = false;
    int InputUsed[2] = { -1, -1 }; // Not yet discovered.
    unsigned LaneStart = l * NumLaneElems;
    for (unsigned i = 0; i != NumLaneElems; ++i) {
      // The mask element.  This indexes into the input.
      int Idx = SVOp->getMaskElt(i+LaneStart);
      if (Idx < 0) {
        // the mask element does not index into any input vector.
        Mask.push_back(-1);
        continue;
      }

      // The input vector this mask element indexes into.
      int Input = Idx / NumLaneElems;

      // Turn the index into an offset from the start of the input vector.
      Idx -= Input * NumLaneElems;

      // Find or create a shuffle vector operand to hold this input.
      unsigned OpNo;
      for (OpNo = 0; OpNo < array_lengthof(InputUsed); ++OpNo) {
        if (InputUsed[OpNo] == Input)
          // This input vector is already an operand.
          break;
        if (InputUsed[OpNo] < 0) {
          // Create a new operand for this input vector.
          InputUsed[OpNo] = Input;
          break;
        }
      }

      if (OpNo >= array_lengthof(InputUsed)) {
        // More than two input vectors used!  Give up on trying to create a
        // shuffle vector.  Insert all elements into a BUILD_VECTOR instead.
        UseBuildVector = true;
        break;
      }

      // Add the mask index for the new shuffle vector.
      Mask.push_back(Idx + OpNo * NumLaneElems);
    }

    if (UseBuildVector) {
      SmallVector<SDValue, 16> SVOps;
      for (unsigned i = 0; i != NumLaneElems; ++i) {
        // The mask element.  This indexes into the input.
        int Idx = SVOp->getMaskElt(i+LaneStart);
        if (Idx < 0) {
          SVOps.push_back(DAG.getUNDEF(EltVT));
          continue;
        }

        // The input vector this mask element indexes into.
        int Input = Idx / NumElems;

        // Turn the index into an offset from the start of the input vector.
        Idx -= Input * NumElems;

        // Extract the vector element by hand.
        SVOps.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT,
                                    SVOp->getOperand(Input),
                                    DAG.getIntPtrConstant(Idx)));
      }

      // Construct the output using a BUILD_VECTOR.
      Output[l] = DAG.getNode(ISD::BUILD_VECTOR, dl, NVT, &SVOps[0],
                              SVOps.size());
    } else if (InputUsed[0] < 0) {
      // No input vectors were used! The result is undefined.
      Output[l] = DAG.getUNDEF(NVT);
    } else {
      SDValue Op0 = Extract128BitVector(SVOp->getOperand(InputUsed[0] / 2),
                                        (InputUsed[0] % 2) * NumLaneElems,
                                        DAG, dl);
      // If only one input was used, use an undefined vector for the other.
      SDValue Op1 = (InputUsed[1] < 0) ? DAG.getUNDEF(NVT) :
        Extract128BitVector(SVOp->getOperand(InputUsed[1] / 2),
                            (InputUsed[1] % 2) * NumLaneElems, DAG, dl);
      // At least one input vector was used. Create a new shuffle vector.
      Output[l] = DAG.getVectorShuffle(NVT, dl, Op0, Op1, &Mask[0]);
    }

    Mask.clear();
  }

  // Concatenate the result back
  return DAG.getNode(ISD::CONCAT_VECTORS, dl, VT, Output[0], Output[1]);
}

/// LowerVECTOR_SHUFFLE_128v4 - Handle all 128-bit wide vectors with
/// 4 elements, and match them with several different shuffle types.
static SDValue
LowerVECTOR_SHUFFLE_128v4(ShuffleVectorSDNode *SVOp, SelectionDAG &DAG) {
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  DebugLoc dl = SVOp->getDebugLoc();
  EVT VT = SVOp->getValueType(0);

  assert(VT.is128BitVector() && "Unsupported vector size");

  std::pair<int, int> Locs[4];
  int Mask1[] = { -1, -1, -1, -1 };
  SmallVector<int, 8> PermMask(SVOp->getMask().begin(), SVOp->getMask().end());

  unsigned NumHi = 0;
  unsigned NumLo = 0;
  for (unsigned i = 0; i != 4; ++i) {
    int Idx = PermMask[i];
    if (Idx < 0) {
      Locs[i] = std::make_pair(-1, -1);
    } else {
      assert(Idx < 8 && "Invalid VECTOR_SHUFFLE index!");
      if (Idx < 4) {
        Locs[i] = std::make_pair(0, NumLo);
        Mask1[NumLo] = Idx;
        NumLo++;
      } else {
        Locs[i] = std::make_pair(1, NumHi);
        if (2+NumHi < 4)
          Mask1[2+NumHi] = Idx;
        NumHi++;
      }
    }
  }

  if (NumLo <= 2 && NumHi <= 2) {
    // If no more than two elements come from either vector. This can be
    // implemented with two shuffles. First shuffle gather the elements.
    // The second shuffle, which takes the first shuffle as both of its
    // vector operands, put the elements into the right order.
    V1 = DAG.getVectorShuffle(VT, dl, V1, V2, &Mask1[0]);

    int Mask2[] = { -1, -1, -1, -1 };

    for (unsigned i = 0; i != 4; ++i)
      if (Locs[i].first != -1) {
        unsigned Idx = (i < 2) ? 0 : 4;
        Idx += Locs[i].first * 2 + Locs[i].second;
        Mask2[i] = Idx;
      }

    return DAG.getVectorShuffle(VT, dl, V1, V1, &Mask2[0]);
  }

  if (NumLo == 3 || NumHi == 3) {
    // Otherwise, we must have three elements from one vector, call it X, and
    // one element from the other, call it Y.  First, use a shufps to build an
    // intermediate vector with the one element from Y and the element from X
    // that will be in the same half in the final destination (the indexes don't
    // matter). Then, use a shufps to build the final vector, taking the half
    // containing the element from Y from the intermediate, and the other half
    // from X.
    if (NumHi == 3) {
      // Normalize it so the 3 elements come from V1.
      CommuteVectorShuffleMask(PermMask, 4);
      std::swap(V1, V2);
    }

    // Find the element from V2.
    unsigned HiIndex;
    for (HiIndex = 0; HiIndex < 3; ++HiIndex) {
      int Val = PermMask[HiIndex];
      if (Val < 0)
        continue;
      if (Val >= 4)
        break;
    }

    Mask1[0] = PermMask[HiIndex];
    Mask1[1] = -1;
    Mask1[2] = PermMask[HiIndex^1];
    Mask1[3] = -1;
    V2 = DAG.getVectorShuffle(VT, dl, V1, V2, &Mask1[0]);

    if (HiIndex >= 2) {
      Mask1[0] = PermMask[0];
      Mask1[1] = PermMask[1];
      Mask1[2] = HiIndex & 1 ? 6 : 4;
      Mask1[3] = HiIndex & 1 ? 4 : 6;
      return DAG.getVectorShuffle(VT, dl, V1, V2, &Mask1[0]);
    }

    Mask1[0] = HiIndex & 1 ? 2 : 0;
    Mask1[1] = HiIndex & 1 ? 0 : 2;
    Mask1[2] = PermMask[2];
    Mask1[3] = PermMask[3];
    if (Mask1[2] >= 0)
      Mask1[2] += 4;
    if (Mask1[3] >= 0)
      Mask1[3] += 4;
    return DAG.getVectorShuffle(VT, dl, V2, V1, &Mask1[0]);
  }

  // Break it into (shuffle shuffle_hi, shuffle_lo).
  int LoMask[] = { -1, -1, -1, -1 };
  int HiMask[] = { -1, -1, -1, -1 };

  int *MaskPtr = LoMask;
  unsigned MaskIdx = 0;
  unsigned LoIdx = 0;
  unsigned HiIdx = 2;
  for (unsigned i = 0; i != 4; ++i) {
    if (i == 2) {
      MaskPtr = HiMask;
      MaskIdx = 1;
      LoIdx = 0;
      HiIdx = 2;
    }
    int Idx = PermMask[i];
    if (Idx < 0) {
      Locs[i] = std::make_pair(-1, -1);
    } else if (Idx < 4) {
      Locs[i] = std::make_pair(MaskIdx, LoIdx);
      MaskPtr[LoIdx] = Idx;
      LoIdx++;
    } else {
      Locs[i] = std::make_pair(MaskIdx, HiIdx);
      MaskPtr[HiIdx] = Idx;
      HiIdx++;
    }
  }

  SDValue LoShuffle = DAG.getVectorShuffle(VT, dl, V1, V2, &LoMask[0]);
  SDValue HiShuffle = DAG.getVectorShuffle(VT, dl, V1, V2, &HiMask[0]);
  int MaskOps[] = { -1, -1, -1, -1 };
  for (unsigned i = 0; i != 4; ++i)
    if (Locs[i].first != -1)
      MaskOps[i] = Locs[i].first * 4 + Locs[i].second;
  return DAG.getVectorShuffle(VT, dl, LoShuffle, HiShuffle, &MaskOps[0]);
}

static bool MayFoldVectorLoad(SDValue V) {
  while (V.hasOneUse() && V.getOpcode() == ISD::BITCAST)
    V = V.getOperand(0);

  if (V.hasOneUse() && V.getOpcode() == ISD::SCALAR_TO_VECTOR)
    V = V.getOperand(0);
  if (V.hasOneUse() && V.getOpcode() == ISD::BUILD_VECTOR &&
      V.getNumOperands() == 2 && V.getOperand(1).getOpcode() == ISD::UNDEF)
    // BUILD_VECTOR (load), undef
    V = V.getOperand(0);

  return MayFoldLoad(V);
}

// FIXME: the version above should always be used. Since there's
// a bug where several vector shuffles can't be folded because the
// DAG is not updated during lowering and a node claims to have two
// uses while it only has one, use this version, and let isel match
// another instruction if the load really happens to have more than
// one use. Remove this version after this bug get fixed.
// rdar://8434668, PR8156
static bool RelaxedMayFoldVectorLoad(SDValue V) {
  if (V.hasOneUse() && V.getOpcode() == ISD::BITCAST)
    V = V.getOperand(0);
  if (V.hasOneUse() && V.getOpcode() == ISD::SCALAR_TO_VECTOR)
    V = V.getOperand(0);
  if (ISD::isNormalLoad(V.getNode()))
    return true;
  return false;
}

static
SDValue getMOVDDup(SDValue &Op, DebugLoc &dl, SDValue V1, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();

  // Canonizalize to v2f64.
  V1 = DAG.getNode(ISD::BITCAST, dl, MVT::v2f64, V1);
  return DAG.getNode(ISD::BITCAST, dl, VT,
                     getTargetShuffleNode(X86ISD::MOVDDUP, dl, MVT::v2f64,
                                          V1, DAG));
}

static
SDValue getMOVLowToHigh(SDValue &Op, DebugLoc &dl, SelectionDAG &DAG,
                        bool HasSSE2) {
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  EVT VT = Op.getValueType();

  assert(VT != MVT::v2i64 && "unsupported shuffle type");

  if (HasSSE2 && VT == MVT::v2f64)
    return getTargetShuffleNode(X86ISD::MOVLHPD, dl, VT, V1, V2, DAG);

  // v4f32 or v4i32: canonizalized to v4f32 (which is legal for SSE1)
  return DAG.getNode(ISD::BITCAST, dl, VT,
                     getTargetShuffleNode(X86ISD::MOVLHPS, dl, MVT::v4f32,
                           DAG.getNode(ISD::BITCAST, dl, MVT::v4f32, V1),
                           DAG.getNode(ISD::BITCAST, dl, MVT::v4f32, V2), DAG));
}

static
SDValue getMOVHighToLow(SDValue &Op, DebugLoc &dl, SelectionDAG &DAG) {
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  EVT VT = Op.getValueType();

  assert((VT == MVT::v4i32 || VT == MVT::v4f32) &&
         "unsupported shuffle type");

  if (V2.getOpcode() == ISD::UNDEF)
    V2 = V1;

  // v4i32 or v4f32
  return getTargetShuffleNode(X86ISD::MOVHLPS, dl, VT, V1, V2, DAG);
}

static
SDValue getMOVLP(SDValue &Op, DebugLoc &dl, SelectionDAG &DAG, bool HasSSE2) {
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  EVT VT = Op.getValueType();
  unsigned NumElems = VT.getVectorNumElements();

  // Use MOVLPS and MOVLPD in case V1 or V2 are loads. During isel, the second
  // operand of these instructions is only memory, so check if there's a
  // potencial load folding here, otherwise use SHUFPS or MOVSD to match the
  // same masks.
  bool CanFoldLoad = false;

  // Trivial case, when V2 comes from a load.
  if (MayFoldVectorLoad(V2))
    CanFoldLoad = true;

  // When V1 is a load, it can be folded later into a store in isel, example:
  //  (store (v4f32 (X86Movlps (load addr:$src1), VR128:$src2)), addr:$src1)
  //    turns into:
  //  (MOVLPSmr addr:$src1, VR128:$src2)
  // So, recognize this potential and also use MOVLPS or MOVLPD
  else if (MayFoldVectorLoad(V1) && MayFoldIntoStore(Op))
    CanFoldLoad = true;

  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  if (CanFoldLoad) {
    if (HasSSE2 && NumElems == 2)
      return getTargetShuffleNode(X86ISD::MOVLPD, dl, VT, V1, V2, DAG);

    if (NumElems == 4)
      // If we don't care about the second element, proceed to use movss.
      if (SVOp->getMaskElt(1) != -1)
        return getTargetShuffleNode(X86ISD::MOVLPS, dl, VT, V1, V2, DAG);
  }

  // movl and movlp will both match v2i64, but v2i64 is never matched by
  // movl earlier because we make it strict to avoid messing with the movlp load
  // folding logic (see the code above getMOVLP call). Match it here then,
  // this is horrible, but will stay like this until we move all shuffle
  // matching to x86 specific nodes. Note that for the 1st condition all
  // types are matched with movsd.
  if (HasSSE2) {
    // FIXME: isMOVLMask should be checked and matched before getMOVLP,
    // as to remove this logic from here, as much as possible
    if (NumElems == 2 || !isMOVLMask(SVOp->getMask(), VT))
      return getTargetShuffleNode(X86ISD::MOVSD, dl, VT, V1, V2, DAG);
    return getTargetShuffleNode(X86ISD::MOVSS, dl, VT, V1, V2, DAG);
  }

  assert(VT != MVT::v4i32 && "unsupported shuffle type");

  // Invert the operand order and use SHUFPS to match it.
  return getTargetShuffleNode(X86ISD::SHUFP, dl, VT, V2, V1,
                              getShuffleSHUFImmediate(SVOp), DAG);
}

// Reduce a vector shuffle to zext.
SDValue
X86TargetLowering::lowerVectorIntExtend(SDValue Op, SelectionDAG &DAG) const {
  // PMOVZX is only available from SSE41.
  if (!Subtarget->hasSSE41())
    return SDValue();

  EVT VT = Op.getValueType();

  // Only AVX2 support 256-bit vector integer extending.
  if (!Subtarget->hasAVX2() && VT.is256BitVector())
    return SDValue();

  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  DebugLoc DL = Op.getDebugLoc();
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  unsigned NumElems = VT.getVectorNumElements();

  // Extending is an unary operation and the element type of the source vector
  // won't be equal to or larger than i64.
  if (V2.getOpcode() != ISD::UNDEF || !VT.isInteger() ||
      VT.getVectorElementType() == MVT::i64)
    return SDValue();

  // Find the expansion ratio, e.g. expanding from i8 to i32 has a ratio of 4.
  unsigned Shift = 1; // Start from 2, i.e. 1 << 1.
  while ((1U << Shift) < NumElems) {
    if (SVOp->getMaskElt(1U << Shift) == 1)
      break;
    Shift += 1;
    // The maximal ratio is 8, i.e. from i8 to i64.
    if (Shift > 3)
      return SDValue();
  }

  // Check the shuffle mask.
  unsigned Mask = (1U << Shift) - 1;
  for (unsigned i = 0; i != NumElems; ++i) {
    int EltIdx = SVOp->getMaskElt(i);
    if ((i & Mask) != 0 && EltIdx != -1)
      return SDValue();
    if ((i & Mask) == 0 && (unsigned)EltIdx != (i >> Shift))
      return SDValue();
  }

  unsigned NBits = VT.getVectorElementType().getSizeInBits() << Shift;
  EVT NeVT = EVT::getIntegerVT(*DAG.getContext(), NBits);
  EVT NVT = EVT::getVectorVT(*DAG.getContext(), NeVT, NumElems >> Shift);

  if (!isTypeLegal(NVT))
    return SDValue();

  // Simplify the operand as it's prepared to be fed into shuffle.
  unsigned SignificantBits = NVT.getSizeInBits() >> Shift;
  if (V1.getOpcode() == ISD::BITCAST &&
      V1.getOperand(0).getOpcode() == ISD::SCALAR_TO_VECTOR &&
      V1.getOperand(0).getOperand(0).getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
      V1.getOperand(0)
        .getOperand(0).getValueType().getSizeInBits() == SignificantBits) {
    // (bitcast (sclr2vec (ext_vec_elt x))) -> (bitcast x)
    SDValue V = V1.getOperand(0).getOperand(0).getOperand(0);
    ConstantSDNode *CIdx =
      dyn_cast<ConstantSDNode>(V1.getOperand(0).getOperand(0).getOperand(1));
    // If it's foldable, i.e. normal load with single use, we will let code
    // selection to fold it. Otherwise, we will short the conversion sequence.
    if (CIdx && CIdx->getZExtValue() == 0 &&
        (!ISD::isNormalLoad(V.getNode()) || !V.hasOneUse()))
      V1 = DAG.getNode(ISD::BITCAST, DL, V1.getValueType(), V);
  }

  return DAG.getNode(ISD::BITCAST, DL, VT,
                     DAG.getNode(X86ISD::VZEXT, DL, NVT, V1));
}

SDValue
X86TargetLowering::NormalizeVectorShuffle(SDValue Op, SelectionDAG &DAG) const {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);

  if (isZeroShuffle(SVOp))
    return getZeroVector(VT, Subtarget, DAG, dl);

  // Handle splat operations
  if (SVOp->isSplat()) {
    unsigned NumElem = VT.getVectorNumElements();
    int Size = VT.getSizeInBits();

    // Use vbroadcast whenever the splat comes from a foldable load
    SDValue Broadcast = LowerVectorBroadcast(Op, DAG);
    if (Broadcast.getNode())
      return Broadcast;

    // Handle splats by matching through known shuffle masks
    if ((Size == 128 && NumElem <= 4) ||
        (Size == 256 && NumElem < 8))
      return SDValue();

    // All remaning splats are promoted to target supported vector shuffles.
    return PromoteSplat(SVOp, DAG);
  }

  // Check integer expanding shuffles.
  SDValue NewOp = lowerVectorIntExtend(Op, DAG);
  if (NewOp.getNode())
    return NewOp;

  // If the shuffle can be profitably rewritten as a narrower shuffle, then
  // do it!
  if (VT == MVT::v8i16  || VT == MVT::v16i8 ||
      VT == MVT::v16i16 || VT == MVT::v32i8) {
    SDValue NewOp = RewriteAsNarrowerShuffle(SVOp, DAG, dl);
    if (NewOp.getNode())
      return DAG.getNode(ISD::BITCAST, dl, VT, NewOp);
  } else if ((VT == MVT::v4i32 ||
             (VT == MVT::v4f32 && Subtarget->hasSSE2()))) {
    // FIXME: Figure out a cleaner way to do this.
    // Try to make use of movq to zero out the top part.
    if (ISD::isBuildVectorAllZeros(V2.getNode())) {
      SDValue NewOp = RewriteAsNarrowerShuffle(SVOp, DAG, dl);
      if (NewOp.getNode()) {
        EVT NewVT = NewOp.getValueType();
        if (isCommutedMOVLMask(cast<ShuffleVectorSDNode>(NewOp)->getMask(),
                               NewVT, true, false))
          return getVZextMovL(VT, NewVT, NewOp.getOperand(0),
                              DAG, Subtarget, dl);
      }
    } else if (ISD::isBuildVectorAllZeros(V1.getNode())) {
      SDValue NewOp = RewriteAsNarrowerShuffle(SVOp, DAG, dl);
      if (NewOp.getNode()) {
        EVT NewVT = NewOp.getValueType();
        if (isMOVLMask(cast<ShuffleVectorSDNode>(NewOp)->getMask(), NewVT))
          return getVZextMovL(VT, NewVT, NewOp.getOperand(1),
                              DAG, Subtarget, dl);
      }
    }
  }
  return SDValue();
}

SDValue
X86TargetLowering::LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned NumElems = VT.getVectorNumElements();
  bool V1IsUndef = V1.getOpcode() == ISD::UNDEF;
  bool V2IsUndef = V2.getOpcode() == ISD::UNDEF;
  bool V1IsSplat = false;
  bool V2IsSplat = false;
  bool HasSSE2 = Subtarget->hasSSE2();
  bool HasAVX    = Subtarget->hasAVX();
  bool HasAVX2   = Subtarget->hasAVX2();
  MachineFunction &MF = DAG.getMachineFunction();
  bool OptForSize = MF.getFunction()->getFnAttributes().
    hasAttribute(Attributes::OptimizeForSize);

  assert(VT.getSizeInBits() != 64 && "Can't lower MMX shuffles");

  if (V1IsUndef && V2IsUndef)
    return DAG.getUNDEF(VT);

  assert(!V1IsUndef && "Op 1 of shuffle should not be undef");

  // Vector shuffle lowering takes 3 steps:
  //
  // 1) Normalize the input vectors. Here splats, zeroed vectors, profitable
  //    narrowing and commutation of operands should be handled.
  // 2) Matching of shuffles with known shuffle masks to x86 target specific
  //    shuffle nodes.
  // 3) Rewriting of unmatched masks into new generic shuffle operations,
  //    so the shuffle can be broken into other shuffles and the legalizer can
  //    try the lowering again.
  //
  // The general idea is that no vector_shuffle operation should be left to
  // be matched during isel, all of them must be converted to a target specific
  // node here.

  // Normalize the input vectors. Here splats, zeroed vectors, profitable
  // narrowing and commutation of operands should be handled. The actual code
  // doesn't include all of those, work in progress...
  SDValue NewOp = NormalizeVectorShuffle(Op, DAG);
  if (NewOp.getNode())
    return NewOp;

  SmallVector<int, 8> M(SVOp->getMask().begin(), SVOp->getMask().end());

  // NOTE: isPSHUFDMask can also match both masks below (unpckl_undef and
  // unpckh_undef). Only use pshufd if speed is more important than size.
  if (OptForSize && isUNPCKL_v_undef_Mask(M, VT, HasAVX2))
    return getTargetShuffleNode(X86ISD::UNPCKL, dl, VT, V1, V1, DAG);
  if (OptForSize && isUNPCKH_v_undef_Mask(M, VT, HasAVX2))
    return getTargetShuffleNode(X86ISD::UNPCKH, dl, VT, V1, V1, DAG);

  if (isMOVDDUPMask(M, VT) && Subtarget->hasSSE3() &&
      V2IsUndef && RelaxedMayFoldVectorLoad(V1))
    return getMOVDDup(Op, dl, V1, DAG);

  if (isMOVHLPS_v_undef_Mask(M, VT))
    return getMOVHighToLow(Op, dl, DAG);

  // Use to match splats
  if (HasSSE2 && isUNPCKHMask(M, VT, HasAVX2) && V2IsUndef &&
      (VT == MVT::v2f64 || VT == MVT::v2i64))
    return getTargetShuffleNode(X86ISD::UNPCKH, dl, VT, V1, V1, DAG);

  if (isPSHUFDMask(M, VT)) {
    // The actual implementation will match the mask in the if above and then
    // during isel it can match several different instructions, not only pshufd
    // as its name says, sad but true, emulate the behavior for now...
    if (isMOVDDUPMask(M, VT) && ((VT == MVT::v4f32 || VT == MVT::v2i64)))
      return getTargetShuffleNode(X86ISD::MOVLHPS, dl, VT, V1, V1, DAG);

    unsigned TargetMask = getShuffleSHUFImmediate(SVOp);

    if (HasAVX && (VT == MVT::v4f32 || VT == MVT::v2f64))
      return getTargetShuffleNode(X86ISD::VPERMILP, dl, VT, V1, TargetMask, DAG);

    if (HasSSE2 && (VT == MVT::v4f32 || VT == MVT::v4i32))
      return getTargetShuffleNode(X86ISD::PSHUFD, dl, VT, V1, TargetMask, DAG);

    return getTargetShuffleNode(X86ISD::SHUFP, dl, VT, V1, V1,
                                TargetMask, DAG);
  }

  // Check if this can be converted into a logical shift.
  bool isLeft = false;
  unsigned ShAmt = 0;
  SDValue ShVal;
  bool isShift = HasSSE2 && isVectorShift(SVOp, DAG, isLeft, ShVal, ShAmt);
  if (isShift && ShVal.hasOneUse()) {
    // If the shifted value has multiple uses, it may be cheaper to use
    // v_set0 + movlhps or movhlps, etc.
    EVT EltVT = VT.getVectorElementType();
    ShAmt *= EltVT.getSizeInBits();
    return getVShift(isLeft, VT, ShVal, ShAmt, DAG, *this, dl);
  }

  if (isMOVLMask(M, VT)) {
    if (ISD::isBuildVectorAllZeros(V1.getNode()))
      return getVZextMovL(VT, VT, V2, DAG, Subtarget, dl);
    if (!isMOVLPMask(M, VT)) {
      if (HasSSE2 && (VT == MVT::v2i64 || VT == MVT::v2f64))
        return getTargetShuffleNode(X86ISD::MOVSD, dl, VT, V1, V2, DAG);

      if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return getTargetShuffleNode(X86ISD::MOVSS, dl, VT, V1, V2, DAG);
    }
  }

  // FIXME: fold these into legal mask.
  if (isMOVLHPSMask(M, VT) && !isUNPCKLMask(M, VT, HasAVX2))
    return getMOVLowToHigh(Op, dl, DAG, HasSSE2);

  if (isMOVHLPSMask(M, VT))
    return getMOVHighToLow(Op, dl, DAG);

  if (V2IsUndef && isMOVSHDUPMask(M, VT, Subtarget))
    return getTargetShuffleNode(X86ISD::MOVSHDUP, dl, VT, V1, DAG);

  if (V2IsUndef && isMOVSLDUPMask(M, VT, Subtarget))
    return getTargetShuffleNode(X86ISD::MOVSLDUP, dl, VT, V1, DAG);

  if (isMOVLPMask(M, VT))
    return getMOVLP(Op, dl, DAG, HasSSE2);

  if (ShouldXformToMOVHLPS(M, VT) ||
      ShouldXformToMOVLP(V1.getNode(), V2.getNode(), M, VT))
    return CommuteVectorShuffle(SVOp, DAG);

  if (isShift) {
    // No better options. Use a vshldq / vsrldq.
    EVT EltVT = VT.getVectorElementType();
    ShAmt *= EltVT.getSizeInBits();
    return getVShift(isLeft, VT, ShVal, ShAmt, DAG, *this, dl);
  }

  bool Commuted = false;
  // FIXME: This should also accept a bitcast of a splat?  Be careful, not
  // 1,1,1,1 -> v8i16 though.
  V1IsSplat = isSplatVector(V1.getNode());
  V2IsSplat = isSplatVector(V2.getNode());

  // Canonicalize the splat or undef, if present, to be on the RHS.
  if (!V2IsUndef && V1IsSplat && !V2IsSplat) {
    CommuteVectorShuffleMask(M, NumElems);
    std::swap(V1, V2);
    std::swap(V1IsSplat, V2IsSplat);
    Commuted = true;
  }

  if (isCommutedMOVLMask(M, VT, V2IsSplat, V2IsUndef)) {
    // Shuffling low element of v1 into undef, just return v1.
    if (V2IsUndef)
      return V1;
    // If V2 is a splat, the mask may be malformed such as <4,3,3,3>, which
    // the instruction selector will not match, so get a canonical MOVL with
    // swapped operands to undo the commute.
    return getMOVL(DAG, dl, VT, V2, V1);
  }

  if (isUNPCKLMask(M, VT, HasAVX2))
    return getTargetShuffleNode(X86ISD::UNPCKL, dl, VT, V1, V2, DAG);

  if (isUNPCKHMask(M, VT, HasAVX2))
    return getTargetShuffleNode(X86ISD::UNPCKH, dl, VT, V1, V2, DAG);

  if (V2IsSplat) {
    // Normalize mask so all entries that point to V2 points to its first
    // element then try to match unpck{h|l} again. If match, return a
    // new vector_shuffle with the corrected mask.p
    SmallVector<int, 8> NewMask(M.begin(), M.end());
    NormalizeMask(NewMask, NumElems);
    if (isUNPCKLMask(NewMask, VT, HasAVX2, true))
      return getTargetShuffleNode(X86ISD::UNPCKL, dl, VT, V1, V2, DAG);
    if (isUNPCKHMask(NewMask, VT, HasAVX2, true))
      return getTargetShuffleNode(X86ISD::UNPCKH, dl, VT, V1, V2, DAG);
  }

  if (Commuted) {
    // Commute is back and try unpck* again.
    // FIXME: this seems wrong.
    CommuteVectorShuffleMask(M, NumElems);
    std::swap(V1, V2);
    std::swap(V1IsSplat, V2IsSplat);
    Commuted = false;

    if (isUNPCKLMask(M, VT, HasAVX2))
      return getTargetShuffleNode(X86ISD::UNPCKL, dl, VT, V1, V2, DAG);

    if (isUNPCKHMask(M, VT, HasAVX2))
      return getTargetShuffleNode(X86ISD::UNPCKH, dl, VT, V1, V2, DAG);
  }

  // Normalize the node to match x86 shuffle ops if needed
  if (!V2IsUndef && (isSHUFPMask(M, VT, HasAVX, /* Commuted */ true)))
    return CommuteVectorShuffle(SVOp, DAG);

  // The checks below are all present in isShuffleMaskLegal, but they are
  // inlined here right now to enable us to directly emit target specific
  // nodes, and remove one by one until they don't return Op anymore.

  if (isPALIGNRMask(M, VT, Subtarget))
    return getTargetShuffleNode(X86ISD::PALIGN, dl, VT, V1, V2,
                                getShufflePALIGNRImmediate(SVOp),
                                DAG);

  if (ShuffleVectorSDNode::isSplatMask(&M[0], VT) &&
      SVOp->getSplatIndex() == 0 && V2IsUndef) {
    if (VT == MVT::v2f64 || VT == MVT::v2i64)
      return getTargetShuffleNode(X86ISD::UNPCKL, dl, VT, V1, V1, DAG);
  }

  if (isPSHUFHWMask(M, VT, HasAVX2))
    return getTargetShuffleNode(X86ISD::PSHUFHW, dl, VT, V1,
                                getShufflePSHUFHWImmediate(SVOp),
                                DAG);

  if (isPSHUFLWMask(M, VT, HasAVX2))
    return getTargetShuffleNode(X86ISD::PSHUFLW, dl, VT, V1,
                                getShufflePSHUFLWImmediate(SVOp),
                                DAG);

  if (isSHUFPMask(M, VT, HasAVX))
    return getTargetShuffleNode(X86ISD::SHUFP, dl, VT, V1, V2,
                                getShuffleSHUFImmediate(SVOp), DAG);

  if (isUNPCKL_v_undef_Mask(M, VT, HasAVX2))
    return getTargetShuffleNode(X86ISD::UNPCKL, dl, VT, V1, V1, DAG);
  if (isUNPCKH_v_undef_Mask(M, VT, HasAVX2))
    return getTargetShuffleNode(X86ISD::UNPCKH, dl, VT, V1, V1, DAG);

  //===--------------------------------------------------------------------===//
  // Generate target specific nodes for 128 or 256-bit shuffles only
  // supported in the AVX instruction set.
  //

  // Handle VMOVDDUPY permutations
  if (V2IsUndef && isMOVDDUPYMask(M, VT, HasAVX))
    return getTargetShuffleNode(X86ISD::MOVDDUP, dl, VT, V1, DAG);

  // Handle VPERMILPS/D* permutations
  if (isVPERMILPMask(M, VT, HasAVX)) {
    if (HasAVX2 && VT == MVT::v8i32)
      return getTargetShuffleNode(X86ISD::PSHUFD, dl, VT, V1,
                                  getShuffleSHUFImmediate(SVOp), DAG);
    return getTargetShuffleNode(X86ISD::VPERMILP, dl, VT, V1,
                                getShuffleSHUFImmediate(SVOp), DAG);
  }

  // Handle VPERM2F128/VPERM2I128 permutations
  if (isVPERM2X128Mask(M, VT, HasAVX))
    return getTargetShuffleNode(X86ISD::VPERM2X128, dl, VT, V1,
                                V2, getShuffleVPERM2X128Immediate(SVOp), DAG);

  SDValue BlendOp = LowerVECTOR_SHUFFLEtoBlend(SVOp, Subtarget, DAG);
  if (BlendOp.getNode())
    return BlendOp;

  if (V2IsUndef && HasAVX2 && (VT == MVT::v8i32 || VT == MVT::v8f32)) {
    SmallVector<SDValue, 8> permclMask;
    for (unsigned i = 0; i != 8; ++i) {
      permclMask.push_back(DAG.getConstant((M[i]>=0) ? M[i] : 0, MVT::i32));
    }
    SDValue Mask = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v8i32,
                               &permclMask[0], 8);
    // Bitcast is for VPERMPS since mask is v8i32 but node takes v8f32
    return DAG.getNode(X86ISD::VPERMV, dl, VT,
                       DAG.getNode(ISD::BITCAST, dl, VT, Mask), V1);
  }

  if (V2IsUndef && HasAVX2 && (VT == MVT::v4i64 || VT == MVT::v4f64))
    return getTargetShuffleNode(X86ISD::VPERMI, dl, VT, V1,
                                getShuffleCLImmediate(SVOp), DAG);


  //===--------------------------------------------------------------------===//
  // Since no target specific shuffle was selected for this generic one,
  // lower it into other known shuffles. FIXME: this isn't true yet, but
  // this is the plan.
  //

  // Handle v8i16 specifically since SSE can do byte extraction and insertion.
  if (VT == MVT::v8i16) {
    SDValue NewOp = LowerVECTOR_SHUFFLEv8i16(Op, Subtarget, DAG);
    if (NewOp.getNode())
      return NewOp;
  }

  if (VT == MVT::v16i8) {
    SDValue NewOp = LowerVECTOR_SHUFFLEv16i8(SVOp, DAG, *this);
    if (NewOp.getNode())
      return NewOp;
  }

  if (VT == MVT::v32i8) {
    SDValue NewOp = LowerVECTOR_SHUFFLEv32i8(SVOp, Subtarget, DAG);
    if (NewOp.getNode())
      return NewOp;
  }

  // Handle all 128-bit wide vectors with 4 elements, and match them with
  // several different shuffle types.
  if (NumElems == 4 && VT.is128BitVector())
    return LowerVECTOR_SHUFFLE_128v4(SVOp, DAG);

  // Handle general 256-bit shuffles
  if (VT.is256BitVector())
    return LowerVECTOR_SHUFFLE_256(SVOp, DAG);

  return SDValue();
}

SDValue
X86TargetLowering::LowerEXTRACT_VECTOR_ELT_SSE4(SDValue Op,
                                                SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();

  if (!Op.getOperand(0).getValueType().is128BitVector())
    return SDValue();

  if (VT.getSizeInBits() == 8) {
    SDValue Extract = DAG.getNode(X86ISD::PEXTRB, dl, MVT::i32,
                                  Op.getOperand(0), Op.getOperand(1));
    SDValue Assert  = DAG.getNode(ISD::AssertZext, dl, MVT::i32, Extract,
                                  DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Assert);
  }

  if (VT.getSizeInBits() == 16) {
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    // If Idx is 0, it's cheaper to do a move instead of a pextrw.
    if (Idx == 0)
      return DAG.getNode(ISD::TRUNCATE, dl, MVT::i16,
                         DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                                     DAG.getNode(ISD::BITCAST, dl,
                                                 MVT::v4i32,
                                                 Op.getOperand(0)),
                                     Op.getOperand(1)));
    SDValue Extract = DAG.getNode(X86ISD::PEXTRW, dl, MVT::i32,
                                  Op.getOperand(0), Op.getOperand(1));
    SDValue Assert  = DAG.getNode(ISD::AssertZext, dl, MVT::i32, Extract,
                                  DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Assert);
  }

  if (VT == MVT::f32) {
    // EXTRACTPS outputs to a GPR32 register which will require a movd to copy
    // the result back to FR32 register. It's only worth matching if the
    // result has a single use which is a store or a bitcast to i32.  And in
    // the case of a store, it's not worth it if the index is a constant 0,
    // because a MOVSSmr can be used instead, which is smaller and faster.
    if (!Op.hasOneUse())
      return SDValue();
    SDNode *User = *Op.getNode()->use_begin();
    if ((User->getOpcode() != ISD::STORE ||
         (isa<ConstantSDNode>(Op.getOperand(1)) &&
          cast<ConstantSDNode>(Op.getOperand(1))->isNullValue())) &&
        (User->getOpcode() != ISD::BITCAST ||
         User->getValueType(0) != MVT::i32))
      return SDValue();
    SDValue Extract = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                                  DAG.getNode(ISD::BITCAST, dl, MVT::v4i32,
                                              Op.getOperand(0)),
                                              Op.getOperand(1));
    return DAG.getNode(ISD::BITCAST, dl, MVT::f32, Extract);
  }

  if (VT == MVT::i32 || VT == MVT::i64) {
    // ExtractPS/pextrq works with constant index.
    if (isa<ConstantSDNode>(Op.getOperand(1)))
      return Op;
  }
  return SDValue();
}


SDValue
X86TargetLowering::LowerEXTRACT_VECTOR_ELT(SDValue Op,
                                           SelectionDAG &DAG) const {
  if (!isa<ConstantSDNode>(Op.getOperand(1)))
    return SDValue();

  SDValue Vec = Op.getOperand(0);
  EVT VecVT = Vec.getValueType();

  // If this is a 256-bit vector result, first extract the 128-bit vector and
  // then extract the element from the 128-bit vector.
  if (VecVT.is256BitVector()) {
    DebugLoc dl = Op.getNode()->getDebugLoc();
    unsigned NumElems = VecVT.getVectorNumElements();
    SDValue Idx = Op.getOperand(1);
    unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();

    // Get the 128-bit vector.
    Vec = Extract128BitVector(Vec, IdxVal, DAG, dl);

    if (IdxVal >= NumElems/2)
      IdxVal -= NumElems/2;
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, Op.getValueType(), Vec,
                       DAG.getConstant(IdxVal, MVT::i32));
  }

  assert(VecVT.is128BitVector() && "Unexpected vector length");

  if (Subtarget->hasSSE41()) {
    SDValue Res = LowerEXTRACT_VECTOR_ELT_SSE4(Op, DAG);
    if (Res.getNode())
      return Res;
  }

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  // TODO: handle v16i8.
  if (VT.getSizeInBits() == 16) {
    SDValue Vec = Op.getOperand(0);
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    if (Idx == 0)
      return DAG.getNode(ISD::TRUNCATE, dl, MVT::i16,
                         DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                                     DAG.getNode(ISD::BITCAST, dl,
                                                 MVT::v4i32, Vec),
                                     Op.getOperand(1)));
    // Transform it so it match pextrw which produces a 32-bit result.
    EVT EltVT = MVT::i32;
    SDValue Extract = DAG.getNode(X86ISD::PEXTRW, dl, EltVT,
                                  Op.getOperand(0), Op.getOperand(1));
    SDValue Assert  = DAG.getNode(ISD::AssertZext, dl, EltVT, Extract,
                                  DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Assert);
  }

  if (VT.getSizeInBits() == 32) {
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    if (Idx == 0)
      return Op;

    // SHUFPS the element to the lowest double word, then movss.
    int Mask[4] = { static_cast<int>(Idx), -1, -1, -1 };
    EVT VVT = Op.getOperand(0).getValueType();
    SDValue Vec = DAG.getVectorShuffle(VVT, dl, Op.getOperand(0),
                                       DAG.getUNDEF(VVT), Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, VT, Vec,
                       DAG.getIntPtrConstant(0));
  }

  if (VT.getSizeInBits() == 64) {
    // FIXME: .td only matches this for <2 x f64>, not <2 x i64> on 32b
    // FIXME: seems like this should be unnecessary if mov{h,l}pd were taught
    //        to match extract_elt for f64.
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    if (Idx == 0)
      return Op;

    // UNPCKHPD the element to the lowest double word, then movsd.
    // Note if the lower 64 bits of the result of the UNPCKHPD is then stored
    // to a f64mem, the whole operation is folded into a single MOVHPDmr.
    int Mask[2] = { 1, -1 };
    EVT VVT = Op.getOperand(0).getValueType();
    SDValue Vec = DAG.getVectorShuffle(VVT, dl, Op.getOperand(0),
                                       DAG.getUNDEF(VVT), Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, VT, Vec,
                       DAG.getIntPtrConstant(0));
  }

  return SDValue();
}

SDValue
X86TargetLowering::LowerINSERT_VECTOR_ELT_SSE4(SDValue Op,
                                               SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  EVT EltVT = VT.getVectorElementType();
  DebugLoc dl = Op.getDebugLoc();

  SDValue N0 = Op.getOperand(0);
  SDValue N1 = Op.getOperand(1);
  SDValue N2 = Op.getOperand(2);

  if (!VT.is128BitVector())
    return SDValue();

  if ((EltVT.getSizeInBits() == 8 || EltVT.getSizeInBits() == 16) &&
      isa<ConstantSDNode>(N2)) {
    unsigned Opc;
    if (VT == MVT::v8i16)
      Opc = X86ISD::PINSRW;
    else if (VT == MVT::v16i8)
      Opc = X86ISD::PINSRB;
    else
      Opc = X86ISD::PINSRB;

    // Transform it so it match pinsr{b,w} which expects a GR32 as its second
    // argument.
    if (N1.getValueType() != MVT::i32)
      N1 = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, N1);
    if (N2.getValueType() != MVT::i32)
      N2 = DAG.getIntPtrConstant(cast<ConstantSDNode>(N2)->getZExtValue());
    return DAG.getNode(Opc, dl, VT, N0, N1, N2);
  }

  if (EltVT == MVT::f32 && isa<ConstantSDNode>(N2)) {
    // Bits [7:6] of the constant are the source select.  This will always be
    //  zero here.  The DAG Combiner may combine an extract_elt index into these
    //  bits.  For example (insert (extract, 3), 2) could be matched by putting
    //  the '3' into bits [7:6] of X86ISD::INSERTPS.
    // Bits [5:4] of the constant are the destination select.  This is the
    //  value of the incoming immediate.
    // Bits [3:0] of the constant are the zero mask.  The DAG Combiner may
    //   combine either bitwise AND or insert of float 0.0 to set these bits.
    N2 = DAG.getIntPtrConstant(cast<ConstantSDNode>(N2)->getZExtValue() << 4);
    // Create this as a scalar to vector..
    N1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4f32, N1);
    return DAG.getNode(X86ISD::INSERTPS, dl, VT, N0, N1, N2);
  }

  if ((EltVT == MVT::i32 || EltVT == MVT::i64) && isa<ConstantSDNode>(N2)) {
    // PINSR* works with constant index.
    return Op;
  }
  return SDValue();
}

SDValue
X86TargetLowering::LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  EVT EltVT = VT.getVectorElementType();

  DebugLoc dl = Op.getDebugLoc();
  SDValue N0 = Op.getOperand(0);
  SDValue N1 = Op.getOperand(1);
  SDValue N2 = Op.getOperand(2);

  // If this is a 256-bit vector result, first extract the 128-bit vector,
  // insert the element into the extracted half and then place it back.
  if (VT.is256BitVector()) {
    if (!isa<ConstantSDNode>(N2))
      return SDValue();

    // Get the desired 128-bit vector half.
    unsigned NumElems = VT.getVectorNumElements();
    unsigned IdxVal = cast<ConstantSDNode>(N2)->getZExtValue();
    SDValue V = Extract128BitVector(N0, IdxVal, DAG, dl);

    // Insert the element into the desired half.
    bool Upper = IdxVal >= NumElems/2;
    V = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, V.getValueType(), V, N1,
                 DAG.getConstant(Upper ? IdxVal-NumElems/2 : IdxVal, MVT::i32));

    // Insert the changed part back to the 256-bit vector
    return Insert128BitVector(N0, V, IdxVal, DAG, dl);
  }

  if (Subtarget->hasSSE41())
    return LowerINSERT_VECTOR_ELT_SSE4(Op, DAG);

  if (EltVT == MVT::i8)
    return SDValue();

  if (EltVT.getSizeInBits() == 16 && isa<ConstantSDNode>(N2)) {
    // Transform it so it match pinsrw which expects a 16-bit value in a GR32
    // as its second argument.
    if (N1.getValueType() != MVT::i32)
      N1 = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, N1);
    if (N2.getValueType() != MVT::i32)
      N2 = DAG.getIntPtrConstant(cast<ConstantSDNode>(N2)->getZExtValue());
    return DAG.getNode(X86ISD::PINSRW, dl, VT, N0, N1, N2);
  }
  return SDValue();
}

static SDValue LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) {
  LLVMContext *Context = DAG.getContext();
  DebugLoc dl = Op.getDebugLoc();
  EVT OpVT = Op.getValueType();

  // If this is a 256-bit vector result, first insert into a 128-bit
  // vector and then insert into the 256-bit vector.
  if (!OpVT.is128BitVector()) {
    // Insert into a 128-bit vector.
    EVT VT128 = EVT::getVectorVT(*Context,
                                 OpVT.getVectorElementType(),
                                 OpVT.getVectorNumElements() / 2);

    Op = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT128, Op.getOperand(0));

    // Insert the 128-bit vector.
    return Insert128BitVector(DAG.getUNDEF(OpVT), Op, 0, DAG, dl);
  }

  if (OpVT == MVT::v1i64 &&
      Op.getOperand(0).getValueType() == MVT::i64)
    return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v1i64, Op.getOperand(0));

  SDValue AnyExt = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, Op.getOperand(0));
  assert(OpVT.is128BitVector() && "Expected an SSE type!");
  return DAG.getNode(ISD::BITCAST, dl, OpVT,
                     DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4i32,AnyExt));
}

// Lower a node with an EXTRACT_SUBVECTOR opcode.  This may result in
// a simple subregister reference or explicit instructions to grab
// upper bits of a vector.
static SDValue LowerEXTRACT_SUBVECTOR(SDValue Op, const X86Subtarget *Subtarget,
                                      SelectionDAG &DAG) {
  if (Subtarget->hasAVX()) {
    DebugLoc dl = Op.getNode()->getDebugLoc();
    SDValue Vec = Op.getNode()->getOperand(0);
    SDValue Idx = Op.getNode()->getOperand(1);

    if (Op.getNode()->getValueType(0).is128BitVector() &&
        Vec.getNode()->getValueType(0).is256BitVector() &&
        isa<ConstantSDNode>(Idx)) {
      unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();
      return Extract128BitVector(Vec, IdxVal, DAG, dl);
    }
  }
  return SDValue();
}

// Lower a node with an INSERT_SUBVECTOR opcode.  This may result in a
// simple superregister reference or explicit instructions to insert
// the upper bits of a vector.
static SDValue LowerINSERT_SUBVECTOR(SDValue Op, const X86Subtarget *Subtarget,
                                     SelectionDAG &DAG) {
  if (Subtarget->hasAVX()) {
    DebugLoc dl = Op.getNode()->getDebugLoc();
    SDValue Vec = Op.getNode()->getOperand(0);
    SDValue SubVec = Op.getNode()->getOperand(1);
    SDValue Idx = Op.getNode()->getOperand(2);

    if (Op.getNode()->getValueType(0).is256BitVector() &&
        SubVec.getNode()->getValueType(0).is128BitVector() &&
        isa<ConstantSDNode>(Idx)) {
      unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();
      return Insert128BitVector(Vec, SubVec, IdxVal, DAG, dl);
    }
  }
  return SDValue();
}

// ConstantPool, JumpTable, GlobalAddress, and ExternalSymbol are lowered as
// their target countpart wrapped in the X86ISD::Wrapper node. Suppose N is
// one of the above mentioned nodes. It has to be wrapped because otherwise
// Select(N) returns N. So the raw TargetGlobalAddress nodes, etc. can only
// be used to form addressing mode. These wrapped nodes will be selected
// into MOV32ri.
SDValue
X86TargetLowering::LowerConstantPool(SDValue Op, SelectionDAG &DAG) const {
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);

  // In PIC mode (unless we're in RIPRel PIC mode) we add an offset to the
  // global base reg.
  unsigned char OpFlag = 0;
  unsigned WrapperKind = X86ISD::Wrapper;
  CodeModel::Model M = getTargetMachine().getCodeModel();

  if (Subtarget->isPICStyleRIPRel() &&
      (M == CodeModel::Small || M == CodeModel::Kernel))
    WrapperKind = X86ISD::WrapperRIP;
  else if (Subtarget->isPICStyleGOT())
    OpFlag = X86II::MO_GOTOFF;
  else if (Subtarget->isPICStyleStubPIC())
    OpFlag = X86II::MO_PIC_BASE_OFFSET;

  SDValue Result = DAG.getTargetConstantPool(CP->getConstVal(), getPointerTy(),
                                             CP->getAlignment(),
                                             CP->getOffset(), OpFlag);
  DebugLoc DL = CP->getDebugLoc();
  Result = DAG.getNode(WrapperKind, DL, getPointerTy(), Result);
  // With PIC, the address is actually $g + Offset.
  if (OpFlag) {
    Result = DAG.getNode(ISD::ADD, DL, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg,
                                     DebugLoc(), getPointerTy()),
                         Result);
  }

  return Result;
}

SDValue X86TargetLowering::LowerJumpTable(SDValue Op, SelectionDAG &DAG) const {
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);

  // In PIC mode (unless we're in RIPRel PIC mode) we add an offset to the
  // global base reg.
  unsigned char OpFlag = 0;
  unsigned WrapperKind = X86ISD::Wrapper;
  CodeModel::Model M = getTargetMachine().getCodeModel();

  if (Subtarget->isPICStyleRIPRel() &&
      (M == CodeModel::Small || M == CodeModel::Kernel))
    WrapperKind = X86ISD::WrapperRIP;
  else if (Subtarget->isPICStyleGOT())
    OpFlag = X86II::MO_GOTOFF;
  else if (Subtarget->isPICStyleStubPIC())
    OpFlag = X86II::MO_PIC_BASE_OFFSET;

  SDValue Result = DAG.getTargetJumpTable(JT->getIndex(), getPointerTy(),
                                          OpFlag);
  DebugLoc DL = JT->getDebugLoc();
  Result = DAG.getNode(WrapperKind, DL, getPointerTy(), Result);

  // With PIC, the address is actually $g + Offset.
  if (OpFlag)
    Result = DAG.getNode(ISD::ADD, DL, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg,
                                     DebugLoc(), getPointerTy()),
                         Result);

  return Result;
}

SDValue
X86TargetLowering::LowerExternalSymbol(SDValue Op, SelectionDAG &DAG) const {
  const char *Sym = cast<ExternalSymbolSDNode>(Op)->getSymbol();

  // In PIC mode (unless we're in RIPRel PIC mode) we add an offset to the
  // global base reg.
  unsigned char OpFlag = 0;
  unsigned WrapperKind = X86ISD::Wrapper;
  CodeModel::Model M = getTargetMachine().getCodeModel();

  if (Subtarget->isPICStyleRIPRel() &&
      (M == CodeModel::Small || M == CodeModel::Kernel)) {
    if (Subtarget->isTargetDarwin() || Subtarget->isTargetELF())
      OpFlag = X86II::MO_GOTPCREL;
    WrapperKind = X86ISD::WrapperRIP;
  } else if (Subtarget->isPICStyleGOT()) {
    OpFlag = X86II::MO_GOT;
  } else if (Subtarget->isPICStyleStubPIC()) {
    OpFlag = X86II::MO_DARWIN_NONLAZY_PIC_BASE;
  } else if (Subtarget->isPICStyleStubNoDynamic()) {
    OpFlag = X86II::MO_DARWIN_NONLAZY;
  }

  SDValue Result = DAG.getTargetExternalSymbol(Sym, getPointerTy(), OpFlag);

  DebugLoc DL = Op.getDebugLoc();
  Result = DAG.getNode(WrapperKind, DL, getPointerTy(), Result);


  // With PIC, the address is actually $g + Offset.
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
      !Subtarget->is64Bit()) {
    Result = DAG.getNode(ISD::ADD, DL, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg,
                                     DebugLoc(), getPointerTy()),
                         Result);
  }

  // For symbols that require a load from a stub to get the address, emit the
  // load.
  if (isGlobalStubReference(OpFlag))
    Result = DAG.getLoad(getPointerTy(), DL, DAG.getEntryNode(), Result,
                         MachinePointerInfo::getGOT(), false, false, false, 0);

  return Result;
}

SDValue
X86TargetLowering::LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const {
  // Create the TargetBlockAddressAddress node.
  unsigned char OpFlags =
    Subtarget->ClassifyBlockAddressReference();
  CodeModel::Model M = getTargetMachine().getCodeModel();
  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();
  int64_t Offset = cast<BlockAddressSDNode>(Op)->getOffset();
  DebugLoc dl = Op.getDebugLoc();
  SDValue Result = DAG.getTargetBlockAddress(BA, getPointerTy(), Offset,
                                             OpFlags);

  if (Subtarget->isPICStyleRIPRel() &&
      (M == CodeModel::Small || M == CodeModel::Kernel))
    Result = DAG.getNode(X86ISD::WrapperRIP, dl, getPointerTy(), Result);
  else
    Result = DAG.getNode(X86ISD::Wrapper, dl, getPointerTy(), Result);

  // With PIC, the address is actually $g + Offset.
  if (isGlobalRelativeToPICBase(OpFlags)) {
    Result = DAG.getNode(ISD::ADD, dl, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg, dl, getPointerTy()),
                         Result);
  }

  return Result;
}

SDValue
X86TargetLowering::LowerGlobalAddress(const GlobalValue *GV, DebugLoc dl,
                                      int64_t Offset,
                                      SelectionDAG &DAG) const {
  // Create the TargetGlobalAddress node, folding in the constant
  // offset if it is legal.
  unsigned char OpFlags =
    Subtarget->ClassifyGlobalReference(GV, getTargetMachine());
  CodeModel::Model M = getTargetMachine().getCodeModel();
  SDValue Result;
  if (OpFlags == X86II::MO_NO_FLAG &&
      X86::isOffsetSuitableForCodeModel(Offset, M)) {
    // A direct static reference to a global.
    Result = DAG.getTargetGlobalAddress(GV, dl, getPointerTy(), Offset);
    Offset = 0;
  } else {
    Result = DAG.getTargetGlobalAddress(GV, dl, getPointerTy(), 0, OpFlags);
  }

  if (Subtarget->isPICStyleRIPRel() &&
      (M == CodeModel::Small || M == CodeModel::Kernel))
    Result = DAG.getNode(X86ISD::WrapperRIP, dl, getPointerTy(), Result);
  else
    Result = DAG.getNode(X86ISD::Wrapper, dl, getPointerTy(), Result);

  // With PIC, the address is actually $g + Offset.
  if (isGlobalRelativeToPICBase(OpFlags)) {
    Result = DAG.getNode(ISD::ADD, dl, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg, dl, getPointerTy()),
                         Result);
  }

  // For globals that require a load from a stub to get the address, emit the
  // load.
  if (isGlobalStubReference(OpFlags))
    Result = DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(), Result,
                         MachinePointerInfo::getGOT(), false, false, false, 0);

  // If there was a non-zero offset that we didn't fold, create an explicit
  // addition for it.
  if (Offset != 0)
    Result = DAG.getNode(ISD::ADD, dl, getPointerTy(), Result,
                         DAG.getConstant(Offset, getPointerTy()));

  return Result;
}

SDValue
X86TargetLowering::LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const {
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  int64_t Offset = cast<GlobalAddressSDNode>(Op)->getOffset();
  return LowerGlobalAddress(GV, Op.getDebugLoc(), Offset, DAG);
}

static SDValue
GetTLSADDR(SelectionDAG &DAG, SDValue Chain, GlobalAddressSDNode *GA,
           SDValue *InFlag, const EVT PtrVT, unsigned ReturnReg,
           unsigned char OperandFlags, bool LocalDynamic = false) {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  DebugLoc dl = GA->getDebugLoc();
  SDValue TGA = DAG.getTargetGlobalAddress(GA->getGlobal(), dl,
                                           GA->getValueType(0),
                                           GA->getOffset(),
                                           OperandFlags);

  X86ISD::NodeType CallType = LocalDynamic ? X86ISD::TLSBASEADDR
                                           : X86ISD::TLSADDR;

  if (InFlag) {
    SDValue Ops[] = { Chain,  TGA, *InFlag };
    Chain = DAG.getNode(CallType, dl, NodeTys, Ops, 3);
  } else {
    SDValue Ops[]  = { Chain, TGA };
    Chain = DAG.getNode(CallType, dl, NodeTys, Ops, 2);
  }

  // TLSADDR will be codegen'ed as call. Inform MFI that function has calls.
  MFI->setAdjustsStack(true);

  SDValue Flag = Chain.getValue(1);
  return DAG.getCopyFromReg(Chain, dl, ReturnReg, PtrVT, Flag);
}

// Lower ISD::GlobalTLSAddress using the "general dynamic" model, 32 bit
static SDValue
LowerToTLSGeneralDynamicModel32(GlobalAddressSDNode *GA, SelectionDAG &DAG,
                                const EVT PtrVT) {
  SDValue InFlag;
  DebugLoc dl = GA->getDebugLoc();  // ? function entry point might be better
  SDValue Chain = DAG.getCopyToReg(DAG.getEntryNode(), dl, X86::EBX,
                                   DAG.getNode(X86ISD::GlobalBaseReg,
                                               DebugLoc(), PtrVT), InFlag);
  InFlag = Chain.getValue(1);

  return GetTLSADDR(DAG, Chain, GA, &InFlag, PtrVT, X86::EAX, X86II::MO_TLSGD);
}

// Lower ISD::GlobalTLSAddress using the "general dynamic" model, 64 bit
static SDValue
LowerToTLSGeneralDynamicModel64(GlobalAddressSDNode *GA, SelectionDAG &DAG,
                                const EVT PtrVT) {
  return GetTLSADDR(DAG, DAG.getEntryNode(), GA, NULL, PtrVT,
                    X86::RAX, X86II::MO_TLSGD);
}

static SDValue LowerToTLSLocalDynamicModel(GlobalAddressSDNode *GA,
                                           SelectionDAG &DAG,
                                           const EVT PtrVT,
                                           bool is64Bit) {
  DebugLoc dl = GA->getDebugLoc();

  // Get the start address of the TLS block for this module.
  X86MachineFunctionInfo* MFI = DAG.getMachineFunction()
      .getInfo<X86MachineFunctionInfo>();
  MFI->incNumLocalDynamicTLSAccesses();

  SDValue Base;
  if (is64Bit) {
    Base = GetTLSADDR(DAG, DAG.getEntryNode(), GA, NULL, PtrVT, X86::RAX,
                      X86II::MO_TLSLD, /*LocalDynamic=*/true);
  } else {
    SDValue InFlag;
    SDValue Chain = DAG.getCopyToReg(DAG.getEntryNode(), dl, X86::EBX,
        DAG.getNode(X86ISD::GlobalBaseReg, DebugLoc(), PtrVT), InFlag);
    InFlag = Chain.getValue(1);
    Base = GetTLSADDR(DAG, Chain, GA, &InFlag, PtrVT, X86::EAX,
                      X86II::MO_TLSLDM, /*LocalDynamic=*/true);
  }

  // Note: the CleanupLocalDynamicTLSPass will remove redundant computations
  // of Base.

  // Build x@dtpoff.
  unsigned char OperandFlags = X86II::MO_DTPOFF;
  unsigned WrapperKind = X86ISD::Wrapper;
  SDValue TGA = DAG.getTargetGlobalAddress(GA->getGlobal(), dl,
                                           GA->getValueType(0),
                                           GA->getOffset(), OperandFlags);
  SDValue Offset = DAG.getNode(WrapperKind, dl, PtrVT, TGA);

  // Add x@dtpoff with the base.
  return DAG.getNode(ISD::ADD, dl, PtrVT, Offset, Base);
}

// Lower ISD::GlobalTLSAddress using the "initial exec" or "local exec" model.
static SDValue LowerToTLSExecModel(GlobalAddressSDNode *GA, SelectionDAG &DAG,
                                   const EVT PtrVT, TLSModel::Model model,
                                   bool is64Bit, bool isPIC) {
  DebugLoc dl = GA->getDebugLoc();

  // Get the Thread Pointer, which is %gs:0 (32-bit) or %fs:0 (64-bit).
  Value *Ptr = Constant::getNullValue(Type::getInt8PtrTy(*DAG.getContext(),
                                                         is64Bit ? 257 : 256));

  SDValue ThreadPointer = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(),
                                      DAG.getIntPtrConstant(0),
                                      MachinePointerInfo(Ptr),
                                      false, false, false, 0);

  unsigned char OperandFlags = 0;
  // Most TLS accesses are not RIP relative, even on x86-64.  One exception is
  // initialexec.
  unsigned WrapperKind = X86ISD::Wrapper;
  if (model == TLSModel::LocalExec) {
    OperandFlags = is64Bit ? X86II::MO_TPOFF : X86II::MO_NTPOFF;
  } else if (model == TLSModel::InitialExec) {
    if (is64Bit) {
      OperandFlags = X86II::MO_GOTTPOFF;
      WrapperKind = X86ISD::WrapperRIP;
    } else {
      OperandFlags = isPIC ? X86II::MO_GOTNTPOFF : X86II::MO_INDNTPOFF;
    }
  } else {
    llvm_unreachable("Unexpected model");
  }

  // emit "addl x@ntpoff,%eax" (local exec)
  // or "addl x@indntpoff,%eax" (initial exec)
  // or "addl x@gotntpoff(%ebx) ,%eax" (initial exec, 32-bit pic)
  SDValue TGA = DAG.getTargetGlobalAddress(GA->getGlobal(), dl,
                                           GA->getValueType(0),
                                           GA->getOffset(), OperandFlags);
  SDValue Offset = DAG.getNode(WrapperKind, dl, PtrVT, TGA);

  if (model == TLSModel::InitialExec) {
    if (isPIC && !is64Bit) {
      Offset = DAG.getNode(ISD::ADD, dl, PtrVT,
                          DAG.getNode(X86ISD::GlobalBaseReg, DebugLoc(), PtrVT),
                           Offset);
    }

    Offset = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), Offset,
                         MachinePointerInfo::getGOT(), false, false, false,
                         0);
  }

  // The address of the thread local variable is the add of the thread
  // pointer with the offset of the variable.
  return DAG.getNode(ISD::ADD, dl, PtrVT, ThreadPointer, Offset);
}

SDValue
X86TargetLowering::LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const {

  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = GA->getGlobal();

  if (Subtarget->isTargetELF()) {
    TLSModel::Model model = getTargetMachine().getTLSModel(GV);

    switch (model) {
      case TLSModel::GeneralDynamic:
        if (Subtarget->is64Bit())
          return LowerToTLSGeneralDynamicModel64(GA, DAG, getPointerTy());
        return LowerToTLSGeneralDynamicModel32(GA, DAG, getPointerTy());
      case TLSModel::LocalDynamic:
        return LowerToTLSLocalDynamicModel(GA, DAG, getPointerTy(),
                                           Subtarget->is64Bit());
      case TLSModel::InitialExec:
      case TLSModel::LocalExec:
        return LowerToTLSExecModel(GA, DAG, getPointerTy(), model,
                                   Subtarget->is64Bit(),
                         getTargetMachine().getRelocationModel() == Reloc::PIC_);
    }
    llvm_unreachable("Unknown TLS model.");
  }

  if (Subtarget->isTargetDarwin()) {
    // Darwin only has one model of TLS.  Lower to that.
    unsigned char OpFlag = 0;
    unsigned WrapperKind = Subtarget->isPICStyleRIPRel() ?
                           X86ISD::WrapperRIP : X86ISD::Wrapper;

    // In PIC mode (unless we're in RIPRel PIC mode) we add an offset to the
    // global base reg.
    bool PIC32 = (getTargetMachine().getRelocationModel() == Reloc::PIC_) &&
                  !Subtarget->is64Bit();
    if (PIC32)
      OpFlag = X86II::MO_TLVP_PIC_BASE;
    else
      OpFlag = X86II::MO_TLVP;
    DebugLoc DL = Op.getDebugLoc();
    SDValue Result = DAG.getTargetGlobalAddress(GA->getGlobal(), DL,
                                                GA->getValueType(0),
                                                GA->getOffset(), OpFlag);
    SDValue Offset = DAG.getNode(WrapperKind, DL, getPointerTy(), Result);

    // With PIC32, the address is actually $g + Offset.
    if (PIC32)
      Offset = DAG.getNode(ISD::ADD, DL, getPointerTy(),
                           DAG.getNode(X86ISD::GlobalBaseReg,
                                       DebugLoc(), getPointerTy()),
                           Offset);

    // Lowering the machine isd will make sure everything is in the right
    // location.
    SDValue Chain = DAG.getEntryNode();
    SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue Args[] = { Chain, Offset };
    Chain = DAG.getNode(X86ISD::TLSCALL, DL, NodeTys, Args, 2);

    // TLSCALL will be codegen'ed as call. Inform MFI that function has calls.
    MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
    MFI->setAdjustsStack(true);

    // And our return value (tls address) is in the standard call return value
    // location.
    unsigned Reg = Subtarget->is64Bit() ? X86::RAX : X86::EAX;
    return DAG.getCopyFromReg(Chain, DL, Reg, getPointerTy(),
                              Chain.getValue(1));
  }

  if (Subtarget->isTargetWindows()) {
    // Just use the implicit TLS architecture
    // Need to generate someting similar to:
    //   mov     rdx, qword [gs:abs 58H]; Load pointer to ThreadLocalStorage
    //                                  ; from TEB
    //   mov     ecx, dword [rel _tls_index]: Load index (from C runtime)
    //   mov     rcx, qword [rdx+rcx*8]
    //   mov     eax, .tls$:tlsvar
    //   [rax+rcx] contains the address
    // Windows 64bit: gs:0x58
    // Windows 32bit: fs:__tls_array

    // If GV is an alias then use the aliasee for determining
    // thread-localness.
    if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
      GV = GA->resolveAliasedGlobal(false);
    DebugLoc dl = GA->getDebugLoc();
    SDValue Chain = DAG.getEntryNode();

    // Get the Thread Pointer, which is %fs:__tls_array (32-bit) or
    // %gs:0x58 (64-bit).
    Value *Ptr = Constant::getNullValue(Subtarget->is64Bit()
                                        ? Type::getInt8PtrTy(*DAG.getContext(),
                                                             256)
                                        : Type::getInt32PtrTy(*DAG.getContext(),
                                                              257));

    SDValue ThreadPointer = DAG.getLoad(getPointerTy(), dl, Chain,
                                        Subtarget->is64Bit()
                                        ? DAG.getIntPtrConstant(0x58)
                                        : DAG.getExternalSymbol("_tls_array",
                                                                getPointerTy()),
                                        MachinePointerInfo(Ptr),
                                        false, false, false, 0);

    // Load the _tls_index variable
    SDValue IDX = DAG.getExternalSymbol("_tls_index", getPointerTy());
    if (Subtarget->is64Bit())
      IDX = DAG.getExtLoad(ISD::ZEXTLOAD, dl, getPointerTy(), Chain,
                           IDX, MachinePointerInfo(), MVT::i32,
                           false, false, 0);
    else
      IDX = DAG.getLoad(getPointerTy(), dl, Chain, IDX, MachinePointerInfo(),
                        false, false, false, 0);

    SDValue Scale = DAG.getConstant(Log2_64_Ceil(TD->getPointerSize()),
                                    getPointerTy());
    IDX = DAG.getNode(ISD::SHL, dl, getPointerTy(), IDX, Scale);

    SDValue res = DAG.getNode(ISD::ADD, dl, getPointerTy(), ThreadPointer, IDX);
    res = DAG.getLoad(getPointerTy(), dl, Chain, res, MachinePointerInfo(),
                      false, false, false, 0);

    // Get the offset of start of .tls section
    SDValue TGA = DAG.getTargetGlobalAddress(GA->getGlobal(), dl,
                                             GA->getValueType(0),
                                             GA->getOffset(), X86II::MO_SECREL);
    SDValue Offset = DAG.getNode(X86ISD::Wrapper, dl, getPointerTy(), TGA);

    // The address of the thread local variable is the add of the thread
    // pointer with the offset of the variable.
    return DAG.getNode(ISD::ADD, dl, getPointerTy(), res, Offset);
  }

  llvm_unreachable("TLS not implemented for this target.");
}


/// LowerShiftParts - Lower SRA_PARTS and friends, which return two i32 values
/// and take a 2 x i32 value to shift plus a shift amount.
SDValue X86TargetLowering::LowerShiftParts(SDValue Op, SelectionDAG &DAG) const{
  assert(Op.getNumOperands() == 3 && "Not a double-shift!");
  EVT VT = Op.getValueType();
  unsigned VTBits = VT.getSizeInBits();
  DebugLoc dl = Op.getDebugLoc();
  bool isSRA = Op.getOpcode() == ISD::SRA_PARTS;
  SDValue ShOpLo = Op.getOperand(0);
  SDValue ShOpHi = Op.getOperand(1);
  SDValue ShAmt  = Op.getOperand(2);
  SDValue Tmp1 = isSRA ? DAG.getNode(ISD::SRA, dl, VT, ShOpHi,
                                     DAG.getConstant(VTBits - 1, MVT::i8))
                       : DAG.getConstant(0, VT);

  SDValue Tmp2, Tmp3;
  if (Op.getOpcode() == ISD::SHL_PARTS) {
    Tmp2 = DAG.getNode(X86ISD::SHLD, dl, VT, ShOpHi, ShOpLo, ShAmt);
    Tmp3 = DAG.getNode(ISD::SHL, dl, VT, ShOpLo, ShAmt);
  } else {
    Tmp2 = DAG.getNode(X86ISD::SHRD, dl, VT, ShOpLo, ShOpHi, ShAmt);
    Tmp3 = DAG.getNode(isSRA ? ISD::SRA : ISD::SRL, dl, VT, ShOpHi, ShAmt);
  }

  SDValue AndNode = DAG.getNode(ISD::AND, dl, MVT::i8, ShAmt,
                                DAG.getConstant(VTBits, MVT::i8));
  SDValue Cond = DAG.getNode(X86ISD::CMP, dl, MVT::i32,
                             AndNode, DAG.getConstant(0, MVT::i8));

  SDValue Hi, Lo;
  SDValue CC = DAG.getConstant(X86::COND_NE, MVT::i8);
  SDValue Ops0[4] = { Tmp2, Tmp3, CC, Cond };
  SDValue Ops1[4] = { Tmp3, Tmp1, CC, Cond };

  if (Op.getOpcode() == ISD::SHL_PARTS) {
    Hi = DAG.getNode(X86ISD::CMOV, dl, VT, Ops0, 4);
    Lo = DAG.getNode(X86ISD::CMOV, dl, VT, Ops1, 4);
  } else {
    Lo = DAG.getNode(X86ISD::CMOV, dl, VT, Ops0, 4);
    Hi = DAG.getNode(X86ISD::CMOV, dl, VT, Ops1, 4);
  }

  SDValue Ops[2] = { Lo, Hi };
  return DAG.getMergeValues(Ops, 2, dl);
}

SDValue X86TargetLowering::LowerSINT_TO_FP(SDValue Op,
                                           SelectionDAG &DAG) const {
  EVT SrcVT = Op.getOperand(0).getValueType();

  if (SrcVT.isVector())
    return SDValue();

  assert(SrcVT.getSimpleVT() <= MVT::i64 && SrcVT.getSimpleVT() >= MVT::i16 &&
         "Unknown SINT_TO_FP to lower!");

  // These are really Legal; return the operand so the caller accepts it as
  // Legal.
  if (SrcVT == MVT::i32 && isScalarFPTypeInSSEReg(Op.getValueType()))
    return Op;
  if (SrcVT == MVT::i64 && isScalarFPTypeInSSEReg(Op.getValueType()) &&
      Subtarget->is64Bit()) {
    return Op;
  }

  DebugLoc dl = Op.getDebugLoc();
  unsigned Size = SrcVT.getSizeInBits()/8;
  MachineFunction &MF = DAG.getMachineFunction();
  int SSFI = MF.getFrameInfo()->CreateStackObject(Size, Size, false);
  SDValue StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
  SDValue Chain = DAG.getStore(DAG.getEntryNode(), dl, Op.getOperand(0),
                               StackSlot,
                               MachinePointerInfo::getFixedStack(SSFI),
                               false, false, 0);
  return BuildFILD(Op, SrcVT, Chain, StackSlot, DAG);
}

SDValue X86TargetLowering::BuildFILD(SDValue Op, EVT SrcVT, SDValue Chain,
                                     SDValue StackSlot,
                                     SelectionDAG &DAG) const {
  // Build the FILD
  DebugLoc DL = Op.getDebugLoc();
  SDVTList Tys;
  bool useSSE = isScalarFPTypeInSSEReg(Op.getValueType());
  if (useSSE)
    Tys = DAG.getVTList(MVT::f64, MVT::Other, MVT::Glue);
  else
    Tys = DAG.getVTList(Op.getValueType(), MVT::Other);

  unsigned ByteSize = SrcVT.getSizeInBits()/8;

  FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(StackSlot);
  MachineMemOperand *MMO;
  if (FI) {
    int SSFI = FI->getIndex();
    MMO =
      DAG.getMachineFunction()
      .getMachineMemOperand(MachinePointerInfo::getFixedStack(SSFI),
                            MachineMemOperand::MOLoad, ByteSize, ByteSize);
  } else {
    MMO = cast<LoadSDNode>(StackSlot)->getMemOperand();
    StackSlot = StackSlot.getOperand(1);
  }
  SDValue Ops[] = { Chain, StackSlot, DAG.getValueType(SrcVT) };
  SDValue Result = DAG.getMemIntrinsicNode(useSSE ? X86ISD::FILD_FLAG :
                                           X86ISD::FILD, DL,
                                           Tys, Ops, array_lengthof(Ops),
                                           SrcVT, MMO);

  if (useSSE) {
    Chain = Result.getValue(1);
    SDValue InFlag = Result.getValue(2);

    // FIXME: Currently the FST is flagged to the FILD_FLAG. This
    // shouldn't be necessary except that RFP cannot be live across
    // multiple blocks. When stackifier is fixed, they can be uncoupled.
    MachineFunction &MF = DAG.getMachineFunction();
    unsigned SSFISize = Op.getValueType().getSizeInBits()/8;
    int SSFI = MF.getFrameInfo()->CreateStackObject(SSFISize, SSFISize, false);
    SDValue StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
    Tys = DAG.getVTList(MVT::Other);
    SDValue Ops[] = {
      Chain, Result, StackSlot, DAG.getValueType(Op.getValueType()), InFlag
    };
    MachineMemOperand *MMO =
      DAG.getMachineFunction()
      .getMachineMemOperand(MachinePointerInfo::getFixedStack(SSFI),
                            MachineMemOperand::MOStore, SSFISize, SSFISize);

    Chain = DAG.getMemIntrinsicNode(X86ISD::FST, DL, Tys,
                                    Ops, array_lengthof(Ops),
                                    Op.getValueType(), MMO);
    Result = DAG.getLoad(Op.getValueType(), DL, Chain, StackSlot,
                         MachinePointerInfo::getFixedStack(SSFI),
                         false, false, false, 0);
  }

  return Result;
}

// LowerUINT_TO_FP_i64 - 64-bit unsigned integer to double expansion.
SDValue X86TargetLowering::LowerUINT_TO_FP_i64(SDValue Op,
                                               SelectionDAG &DAG) const {
  // This algorithm is not obvious. Here it is what we're trying to output:
  /*
     movq       %rax,  %xmm0
     punpckldq  (c0),  %xmm0  // c0: (uint4){ 0x43300000U, 0x45300000U, 0U, 0U }
     subpd      (c1),  %xmm0  // c1: (double2){ 0x1.0p52, 0x1.0p52 * 0x1.0p32 }
     #ifdef __SSE3__
       haddpd   %xmm0, %xmm0
     #else
       pshufd   $0x4e, %xmm0, %xmm1
       addpd    %xmm1, %xmm0
     #endif
  */

  DebugLoc dl = Op.getDebugLoc();
  LLVMContext *Context = DAG.getContext();

  // Build some magic constants.
  const uint32_t CV0[] = { 0x43300000, 0x45300000, 0, 0 };
  Constant *C0 = ConstantDataVector::get(*Context, CV0);
  SDValue CPIdx0 = DAG.getConstantPool(C0, getPointerTy(), 16);

  SmallVector<Constant*,2> CV1;
  CV1.push_back(
        ConstantFP::get(*Context, APFloat(APInt(64, 0x4330000000000000ULL))));
  CV1.push_back(
        ConstantFP::get(*Context, APFloat(APInt(64, 0x4530000000000000ULL))));
  Constant *C1 = ConstantVector::get(CV1);
  SDValue CPIdx1 = DAG.getConstantPool(C1, getPointerTy(), 16);

  // Load the 64-bit value into an XMM register.
  SDValue XR1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2i64,
                            Op.getOperand(0));
  SDValue CLod0 = DAG.getLoad(MVT::v4i32, dl, DAG.getEntryNode(), CPIdx0,
                              MachinePointerInfo::getConstantPool(),
                              false, false, false, 16);
  SDValue Unpck1 = getUnpackl(DAG, dl, MVT::v4i32,
                              DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, XR1),
                              CLod0);

  SDValue CLod1 = DAG.getLoad(MVT::v2f64, dl, CLod0.getValue(1), CPIdx1,
                              MachinePointerInfo::getConstantPool(),
                              false, false, false, 16);
  SDValue XR2F = DAG.getNode(ISD::BITCAST, dl, MVT::v2f64, Unpck1);
  SDValue Sub = DAG.getNode(ISD::FSUB, dl, MVT::v2f64, XR2F, CLod1);
  SDValue Result;

  if (Subtarget->hasSSE3()) {
    // FIXME: The 'haddpd' instruction may be slower than 'movhlps + addsd'.
    Result = DAG.getNode(X86ISD::FHADD, dl, MVT::v2f64, Sub, Sub);
  } else {
    SDValue S2F = DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, Sub);
    SDValue Shuffle = getTargetShuffleNode(X86ISD::PSHUFD, dl, MVT::v4i32,
                                           S2F, 0x4E, DAG);
    Result = DAG.getNode(ISD::FADD, dl, MVT::v2f64,
                         DAG.getNode(ISD::BITCAST, dl, MVT::v2f64, Shuffle),
                         Sub);
  }

  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64, Result,
                     DAG.getIntPtrConstant(0));
}

// LowerUINT_TO_FP_i32 - 32-bit unsigned integer to float expansion.
SDValue X86TargetLowering::LowerUINT_TO_FP_i32(SDValue Op,
                                               SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  // FP constant to bias correct the final result.
  SDValue Bias = DAG.getConstantFP(BitsToDouble(0x4330000000000000ULL),
                                   MVT::f64);

  // Load the 32-bit value into an XMM register.
  SDValue Load = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4i32,
                             Op.getOperand(0));

  // Zero out the upper parts of the register.
  Load = getShuffleVectorZeroOrUndef(Load, 0, true, Subtarget, DAG);

  Load = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64,
                     DAG.getNode(ISD::BITCAST, dl, MVT::v2f64, Load),
                     DAG.getIntPtrConstant(0));

  // Or the load with the bias.
  SDValue Or = DAG.getNode(ISD::OR, dl, MVT::v2i64,
                           DAG.getNode(ISD::BITCAST, dl, MVT::v2i64,
                                       DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                                   MVT::v2f64, Load)),
                           DAG.getNode(ISD::BITCAST, dl, MVT::v2i64,
                                       DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                                   MVT::v2f64, Bias)));
  Or = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64,
                   DAG.getNode(ISD::BITCAST, dl, MVT::v2f64, Or),
                   DAG.getIntPtrConstant(0));

  // Subtract the bias.
  SDValue Sub = DAG.getNode(ISD::FSUB, dl, MVT::f64, Or, Bias);

  // Handle final rounding.
  EVT DestVT = Op.getValueType();

  if (DestVT.bitsLT(MVT::f64))
    return DAG.getNode(ISD::FP_ROUND, dl, DestVT, Sub,
                       DAG.getIntPtrConstant(0));
  if (DestVT.bitsGT(MVT::f64))
    return DAG.getNode(ISD::FP_EXTEND, dl, DestVT, Sub);

  // Handle final rounding.
  return Sub;
}

SDValue X86TargetLowering::lowerUINT_TO_FP_vec(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDValue N0 = Op.getOperand(0);
  EVT SVT = N0.getValueType();
  DebugLoc dl = Op.getDebugLoc();

  assert((SVT == MVT::v4i8 || SVT == MVT::v4i16 ||
          SVT == MVT::v8i8 || SVT == MVT::v8i16) &&
         "Custom UINT_TO_FP is not supported!");

  EVT NVT = EVT::getVectorVT(*DAG.getContext(), MVT::i32, SVT.getVectorNumElements());
  return DAG.getNode(ISD::SINT_TO_FP, dl, Op.getValueType(),
                     DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, N0));
}

SDValue X86TargetLowering::LowerUINT_TO_FP(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDValue N0 = Op.getOperand(0);
  DebugLoc dl = Op.getDebugLoc();

  if (Op.getValueType().isVector())
    return lowerUINT_TO_FP_vec(Op, DAG);

  // Since UINT_TO_FP is legal (it's marked custom), dag combiner won't
  // optimize it to a SINT_TO_FP when the sign bit is known zero. Perform
  // the optimization here.
  if (DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::SINT_TO_FP, dl, Op.getValueType(), N0);

  EVT SrcVT = N0.getValueType();
  EVT DstVT = Op.getValueType();
  if (SrcVT == MVT::i64 && DstVT == MVT::f64 && X86ScalarSSEf64)
    return LowerUINT_TO_FP_i64(Op, DAG);
  if (SrcVT == MVT::i32 && X86ScalarSSEf64)
    return LowerUINT_TO_FP_i32(Op, DAG);
  if (Subtarget->is64Bit() && SrcVT == MVT::i64 && DstVT == MVT::f32)
    return SDValue();

  // Make a 64-bit buffer, and use it to build an FILD.
  SDValue StackSlot = DAG.CreateStackTemporary(MVT::i64);
  if (SrcVT == MVT::i32) {
    SDValue WordOff = DAG.getConstant(4, getPointerTy());
    SDValue OffsetSlot = DAG.getNode(ISD::ADD, dl,
                                     getPointerTy(), StackSlot, WordOff);
    SDValue Store1 = DAG.getStore(DAG.getEntryNode(), dl, Op.getOperand(0),
                                  StackSlot, MachinePointerInfo(),
                                  false, false, 0);
    SDValue Store2 = DAG.getStore(Store1, dl, DAG.getConstant(0, MVT::i32),
                                  OffsetSlot, MachinePointerInfo(),
                                  false, false, 0);
    SDValue Fild = BuildFILD(Op, MVT::i64, Store2, StackSlot, DAG);
    return Fild;
  }

  assert(SrcVT == MVT::i64 && "Unexpected type in UINT_TO_FP");
  SDValue Store = DAG.getStore(DAG.getEntryNode(), dl, Op.getOperand(0),
                               StackSlot, MachinePointerInfo(),
                               false, false, 0);
  // For i64 source, we need to add the appropriate power of 2 if the input
  // was negative.  This is the same as the optimization in
  // DAGTypeLegalizer::ExpandIntOp_UNIT_TO_FP, and for it to be safe here,
  // we must be careful to do the computation in x87 extended precision, not
  // in SSE. (The generic code can't know it's OK to do this, or how to.)
  int SSFI = cast<FrameIndexSDNode>(StackSlot)->getIndex();
  MachineMemOperand *MMO =
    DAG.getMachineFunction()
    .getMachineMemOperand(MachinePointerInfo::getFixedStack(SSFI),
                          MachineMemOperand::MOLoad, 8, 8);

  SDVTList Tys = DAG.getVTList(MVT::f80, MVT::Other);
  SDValue Ops[] = { Store, StackSlot, DAG.getValueType(MVT::i64) };
  SDValue Fild = DAG.getMemIntrinsicNode(X86ISD::FILD, dl, Tys, Ops, 3,
                                         MVT::i64, MMO);

  APInt FF(32, 0x5F800000ULL);

  // Check whether the sign bit is set.
  SDValue SignSet = DAG.getSetCC(dl, getSetCCResultType(MVT::i64),
                                 Op.getOperand(0), DAG.getConstant(0, MVT::i64),
                                 ISD::SETLT);

  // Build a 64 bit pair (0, FF) in the constant pool, with FF in the lo bits.
  SDValue FudgePtr = DAG.getConstantPool(
                             ConstantInt::get(*DAG.getContext(), FF.zext(64)),
                                         getPointerTy());

  // Get a pointer to FF if the sign bit was set, or to 0 otherwise.
  SDValue Zero = DAG.getIntPtrConstant(0);
  SDValue Four = DAG.getIntPtrConstant(4);
  SDValue Offset = DAG.getNode(ISD::SELECT, dl, Zero.getValueType(), SignSet,
                               Zero, Four);
  FudgePtr = DAG.getNode(ISD::ADD, dl, getPointerTy(), FudgePtr, Offset);

  // Load the value out, extending it from f32 to f80.
  // FIXME: Avoid the extend by constructing the right constant pool?
  SDValue Fudge = DAG.getExtLoad(ISD::EXTLOAD, dl, MVT::f80, DAG.getEntryNode(),
                                 FudgePtr, MachinePointerInfo::getConstantPool(),
                                 MVT::f32, false, false, 4);
  // Extend everything to 80 bits to force it to be done on x87.
  SDValue Add = DAG.getNode(ISD::FADD, dl, MVT::f80, Fild, Fudge);
  return DAG.getNode(ISD::FP_ROUND, dl, DstVT, Add, DAG.getIntPtrConstant(0));
}

std::pair<SDValue,SDValue> X86TargetLowering::
FP_TO_INTHelper(SDValue Op, SelectionDAG &DAG, bool IsSigned, bool IsReplace) const {
  DebugLoc DL = Op.getDebugLoc();

  EVT DstTy = Op.getValueType();

  if (!IsSigned && !isIntegerTypeFTOL(DstTy)) {
    assert(DstTy == MVT::i32 && "Unexpected FP_TO_UINT");
    DstTy = MVT::i64;
  }

  assert(DstTy.getSimpleVT() <= MVT::i64 &&
         DstTy.getSimpleVT() >= MVT::i16 &&
         "Unknown FP_TO_INT to lower!");

  // These are really Legal.
  if (DstTy == MVT::i32 &&
      isScalarFPTypeInSSEReg(Op.getOperand(0).getValueType()))
    return std::make_pair(SDValue(), SDValue());
  if (Subtarget->is64Bit() &&
      DstTy == MVT::i64 &&
      isScalarFPTypeInSSEReg(Op.getOperand(0).getValueType()))
    return std::make_pair(SDValue(), SDValue());

  // We lower FP->int64 either into FISTP64 followed by a load from a temporary
  // stack slot, or into the FTOL runtime function.
  MachineFunction &MF = DAG.getMachineFunction();
  unsigned MemSize = DstTy.getSizeInBits()/8;
  int SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize, false);
  SDValue StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());

  unsigned Opc;
  if (!IsSigned && isIntegerTypeFTOL(DstTy))
    Opc = X86ISD::WIN_FTOL;
  else
    switch (DstTy.getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Invalid FP_TO_SINT to lower!");
    case MVT::i16: Opc = X86ISD::FP_TO_INT16_IN_MEM; break;
    case MVT::i32: Opc = X86ISD::FP_TO_INT32_IN_MEM; break;
    case MVT::i64: Opc = X86ISD::FP_TO_INT64_IN_MEM; break;
    }

  SDValue Chain = DAG.getEntryNode();
  SDValue Value = Op.getOperand(0);
  EVT TheVT = Op.getOperand(0).getValueType();
  // FIXME This causes a redundant load/store if the SSE-class value is already
  // in memory, such as if it is on the callstack.
  if (isScalarFPTypeInSSEReg(TheVT)) {
    assert(DstTy == MVT::i64 && "Invalid FP_TO_SINT to lower!");
    Chain = DAG.getStore(Chain, DL, Value, StackSlot,
                         MachinePointerInfo::getFixedStack(SSFI),
                         false, false, 0);
    SDVTList Tys = DAG.getVTList(Op.getOperand(0).getValueType(), MVT::Other);
    SDValue Ops[] = {
      Chain, StackSlot, DAG.getValueType(TheVT)
    };

    MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(SSFI),
                              MachineMemOperand::MOLoad, MemSize, MemSize);
    Value = DAG.getMemIntrinsicNode(X86ISD::FLD, DL, Tys, Ops, 3,
                                    DstTy, MMO);
    Chain = Value.getValue(1);
    SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize, false);
    StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
  }

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(SSFI),
                            MachineMemOperand::MOStore, MemSize, MemSize);

  if (Opc != X86ISD::WIN_FTOL) {
    // Build the FP_TO_INT*_IN_MEM
    SDValue Ops[] = { Chain, Value, StackSlot };
    SDValue FIST = DAG.getMemIntrinsicNode(Opc, DL, DAG.getVTList(MVT::Other),
                                           Ops, 3, DstTy, MMO);
    return std::make_pair(FIST, StackSlot);
  } else {
    SDValue ftol = DAG.getNode(X86ISD::WIN_FTOL, DL,
      DAG.getVTList(MVT::Other, MVT::Glue),
      Chain, Value);
    SDValue eax = DAG.getCopyFromReg(ftol, DL, X86::EAX,
      MVT::i32, ftol.getValue(1));
    SDValue edx = DAG.getCopyFromReg(eax.getValue(1), DL, X86::EDX,
      MVT::i32, eax.getValue(2));
    SDValue Ops[] = { eax, edx };
    SDValue pair = IsReplace
      ? DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, Ops, 2)
      : DAG.getMergeValues(Ops, 2, DL);
    return std::make_pair(pair, SDValue());
  }
}

SDValue X86TargetLowering::lowerZERO_EXTEND(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  SDValue In = Op.getOperand(0);
  EVT SVT = In.getValueType();

  if (!VT.is256BitVector() || !SVT.is128BitVector() ||
      VT.getVectorNumElements() != SVT.getVectorNumElements())
    return SDValue();

  assert(Subtarget->hasAVX() && "256-bit vector is observed without AVX!");

  // AVX2 has better support of integer extending.
  if (Subtarget->hasAVX2())
    return DAG.getNode(X86ISD::VZEXT, DL, VT, In);

  SDValue Lo = DAG.getNode(X86ISD::VZEXT, DL, MVT::v4i32, In);
  static const int Mask[] = {4, 5, 6, 7, -1, -1, -1, -1};
  SDValue Hi = DAG.getNode(X86ISD::VZEXT, DL, MVT::v4i32,
                           DAG.getVectorShuffle(MVT::v8i16, DL, In, DAG.getUNDEF(MVT::v8i16), &Mask[0]));

  return DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v8i32, Lo, Hi);
}

SDValue X86TargetLowering::lowerTRUNCATE(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT SVT = Op.getOperand(0).getValueType();

  if (!VT.is128BitVector() || !SVT.is256BitVector() ||
      VT.getVectorNumElements() != SVT.getVectorNumElements())
    return SDValue();

  assert(Subtarget->hasAVX() && "256-bit vector is observed without AVX!");

  unsigned NumElems = VT.getVectorNumElements();
  EVT NVT = EVT::getVectorVT(*DAG.getContext(), VT.getVectorElementType(),
                             NumElems * 2);

  SDValue In = Op.getOperand(0);
  SmallVector<int, 16> MaskVec(NumElems * 2, -1);
  // Prepare truncation shuffle mask
  for (unsigned i = 0; i != NumElems; ++i)
    MaskVec[i] = i * 2;
  SDValue V = DAG.getVectorShuffle(NVT, DL,
                                   DAG.getNode(ISD::BITCAST, DL, NVT, In),
                                   DAG.getUNDEF(NVT), &MaskVec[0]);
  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, V,
                     DAG.getIntPtrConstant(0));
}

SDValue X86TargetLowering::LowerFP_TO_SINT(SDValue Op,
                                           SelectionDAG &DAG) const {
  if (Op.getValueType().isVector()) {
    if (Op.getValueType() == MVT::v8i16)
      return DAG.getNode(ISD::TRUNCATE, Op.getDebugLoc(), Op.getValueType(),
                         DAG.getNode(ISD::FP_TO_SINT, Op.getDebugLoc(),
                                     MVT::v8i32, Op.getOperand(0)));
    return SDValue();
  }

  std::pair<SDValue,SDValue> Vals = FP_TO_INTHelper(Op, DAG,
    /*IsSigned=*/ true, /*IsReplace=*/ false);
  SDValue FIST = Vals.first, StackSlot = Vals.second;
  // If FP_TO_INTHelper failed, the node is actually supposed to be Legal.
  if (FIST.getNode() == 0) return Op;

  if (StackSlot.getNode())
    // Load the result.
    return DAG.getLoad(Op.getValueType(), Op.getDebugLoc(),
                       FIST, StackSlot, MachinePointerInfo(),
                       false, false, false, 0);

  // The node is the result.
  return FIST;
}

SDValue X86TargetLowering::LowerFP_TO_UINT(SDValue Op,
                                           SelectionDAG &DAG) const {
  std::pair<SDValue,SDValue> Vals = FP_TO_INTHelper(Op, DAG,
    /*IsSigned=*/ false, /*IsReplace=*/ false);
  SDValue FIST = Vals.first, StackSlot = Vals.second;
  assert(FIST.getNode() && "Unexpected failure");

  if (StackSlot.getNode())
    // Load the result.
    return DAG.getLoad(Op.getValueType(), Op.getDebugLoc(),
                       FIST, StackSlot, MachinePointerInfo(),
                       false, false, false, 0);

  // The node is the result.
  return FIST;
}

SDValue X86TargetLowering::lowerFP_EXTEND(SDValue Op,
                                          SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  SDValue In = Op.getOperand(0);
  EVT SVT = In.getValueType();

  assert(SVT == MVT::v2f32 && "Only customize MVT::v2f32 type legalization!");

  return DAG.getNode(X86ISD::VFPEXT, DL, VT,
                     DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v4f32,
                                 In, DAG.getUNDEF(SVT)));
}

SDValue X86TargetLowering::LowerFABS(SDValue Op, SelectionDAG &DAG) const {
  LLVMContext *Context = DAG.getContext();
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT EltVT = VT;
  unsigned NumElts = VT == MVT::f64 ? 2 : 4;
  if (VT.isVector()) {
    EltVT = VT.getVectorElementType();
    NumElts = VT.getVectorNumElements();
  }
  Constant *C;
  if (EltVT == MVT::f64)
    C = ConstantFP::get(*Context, APFloat(APInt(64, ~(1ULL << 63))));
  else
    C = ConstantFP::get(*Context, APFloat(APInt(32, ~(1U << 31))));
  C = ConstantVector::getSplat(NumElts, C);
  SDValue CPIdx = DAG.getConstantPool(C, getPointerTy());
  unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
  SDValue Mask = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                             MachinePointerInfo::getConstantPool(),
                             false, false, false, Alignment);
  if (VT.isVector()) {
    MVT ANDVT = VT.is128BitVector() ? MVT::v2i64 : MVT::v4i64;
    return DAG.getNode(ISD::BITCAST, dl, VT,
                       DAG.getNode(ISD::AND, dl, ANDVT,
                                   DAG.getNode(ISD::BITCAST, dl, ANDVT,
                                               Op.getOperand(0)),
                                   DAG.getNode(ISD::BITCAST, dl, ANDVT, Mask)));
  }
  return DAG.getNode(X86ISD::FAND, dl, VT, Op.getOperand(0), Mask);
}

SDValue X86TargetLowering::LowerFNEG(SDValue Op, SelectionDAG &DAG) const {
  LLVMContext *Context = DAG.getContext();
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT EltVT = VT;
  unsigned NumElts = VT == MVT::f64 ? 2 : 4;
  if (VT.isVector()) {
    EltVT = VT.getVectorElementType();
    NumElts = VT.getVectorNumElements();
  }
  Constant *C;
  if (EltVT == MVT::f64)
    C = ConstantFP::get(*Context, APFloat(APInt(64, 1ULL << 63)));
  else
    C = ConstantFP::get(*Context, APFloat(APInt(32, 1U << 31)));
  C = ConstantVector::getSplat(NumElts, C);
  SDValue CPIdx = DAG.getConstantPool(C, getPointerTy());
  unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
  SDValue Mask = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                             MachinePointerInfo::getConstantPool(),
                             false, false, false, Alignment);
  if (VT.isVector()) {
    MVT XORVT = VT.is128BitVector() ? MVT::v2i64 : MVT::v4i64;
    return DAG.getNode(ISD::BITCAST, dl, VT,
                       DAG.getNode(ISD::XOR, dl, XORVT,
                                   DAG.getNode(ISD::BITCAST, dl, XORVT,
                                               Op.getOperand(0)),
                                   DAG.getNode(ISD::BITCAST, dl, XORVT, Mask)));
  }

  return DAG.getNode(X86ISD::FXOR, dl, VT, Op.getOperand(0), Mask);
}

SDValue X86TargetLowering::LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const {
  LLVMContext *Context = DAG.getContext();
  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT SrcVT = Op1.getValueType();

  // If second operand is smaller, extend it first.
  if (SrcVT.bitsLT(VT)) {
    Op1 = DAG.getNode(ISD::FP_EXTEND, dl, VT, Op1);
    SrcVT = VT;
  }
  // And if it is bigger, shrink it first.
  if (SrcVT.bitsGT(VT)) {
    Op1 = DAG.getNode(ISD::FP_ROUND, dl, VT, Op1, DAG.getIntPtrConstant(1));
    SrcVT = VT;
  }

  // At this point the operands and the result should have the same
  // type, and that won't be f80 since that is not custom lowered.

  // First get the sign bit of second operand.
  SmallVector<Constant*,4> CV;
  if (SrcVT == MVT::f64) {
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(64, 1ULL << 63))));
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(64, 0))));
  } else {
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(32, 1U << 31))));
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(32, 0))));
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(32, 0))));
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(32, 0))));
  }
  Constant *C = ConstantVector::get(CV);
  SDValue CPIdx = DAG.getConstantPool(C, getPointerTy(), 16);
  SDValue Mask1 = DAG.getLoad(SrcVT, dl, DAG.getEntryNode(), CPIdx,
                              MachinePointerInfo::getConstantPool(),
                              false, false, false, 16);
  SDValue SignBit = DAG.getNode(X86ISD::FAND, dl, SrcVT, Op1, Mask1);

  // Shift sign bit right or left if the two operands have different types.
  if (SrcVT.bitsGT(VT)) {
    // Op0 is MVT::f32, Op1 is MVT::f64.
    SignBit = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2f64, SignBit);
    SignBit = DAG.getNode(X86ISD::FSRL, dl, MVT::v2f64, SignBit,
                          DAG.getConstant(32, MVT::i32));
    SignBit = DAG.getNode(ISD::BITCAST, dl, MVT::v4f32, SignBit);
    SignBit = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f32, SignBit,
                          DAG.getIntPtrConstant(0));
  }

  // Clear first operand sign bit.
  CV.clear();
  if (VT == MVT::f64) {
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(64, ~(1ULL << 63)))));
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(64, 0))));
  } else {
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(32, ~(1U << 31)))));
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(32, 0))));
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(32, 0))));
    CV.push_back(ConstantFP::get(*Context, APFloat(APInt(32, 0))));
  }
  C = ConstantVector::get(CV);
  CPIdx = DAG.getConstantPool(C, getPointerTy(), 16);
  SDValue Mask2 = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                              MachinePointerInfo::getConstantPool(),
                              false, false, false, 16);
  SDValue Val = DAG.getNode(X86ISD::FAND, dl, VT, Op0, Mask2);

  // Or the value with the sign bit.
  return DAG.getNode(X86ISD::FOR, dl, VT, Val, SignBit);
}

static SDValue LowerFGETSIGN(SDValue Op, SelectionDAG &DAG) {
  SDValue N0 = Op.getOperand(0);
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  // Lower ISD::FGETSIGN to (AND (X86ISD::FGETSIGNx86 ...) 1).
  SDValue xFGETSIGN = DAG.getNode(X86ISD::FGETSIGNx86, dl, VT, N0,
                                  DAG.getConstant(1, VT));
  return DAG.getNode(ISD::AND, dl, VT, xFGETSIGN, DAG.getConstant(1, VT));
}

// LowerVectorAllZeroTest - Check whether an OR'd tree is PTEST-able.
//
SDValue X86TargetLowering::LowerVectorAllZeroTest(SDValue Op, SelectionDAG &DAG) const {
  assert(Op.getOpcode() == ISD::OR && "Only check OR'd tree.");

  if (!Subtarget->hasSSE41())
    return SDValue();

  if (!Op->hasOneUse())
    return SDValue();

  SDNode *N = Op.getNode();
  DebugLoc DL = N->getDebugLoc();

  SmallVector<SDValue, 8> Opnds;
  DenseMap<SDValue, unsigned> VecInMap;
  EVT VT = MVT::Other;

  // Recognize a special case where a vector is casted into wide integer to
  // test all 0s.
  Opnds.push_back(N->getOperand(0));
  Opnds.push_back(N->getOperand(1));

  for (unsigned Slot = 0, e = Opnds.size(); Slot < e; ++Slot) {
    SmallVector<SDValue, 8>::const_iterator I = Opnds.begin() + Slot;
    // BFS traverse all OR'd operands.
    if (I->getOpcode() == ISD::OR) {
      Opnds.push_back(I->getOperand(0));
      Opnds.push_back(I->getOperand(1));
      // Re-evaluate the number of nodes to be traversed.
      e += 2; // 2 more nodes (LHS and RHS) are pushed.
      continue;
    }

    // Quit if a non-EXTRACT_VECTOR_ELT
    if (I->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return SDValue();

    // Quit if without a constant index.
    SDValue Idx = I->getOperand(1);
    if (!isa<ConstantSDNode>(Idx))
      return SDValue();

    SDValue ExtractedFromVec = I->getOperand(0);
    DenseMap<SDValue, unsigned>::iterator M = VecInMap.find(ExtractedFromVec);
    if (M == VecInMap.end()) {
      VT = ExtractedFromVec.getValueType();
      // Quit if not 128/256-bit vector.
      if (!VT.is128BitVector() && !VT.is256BitVector())
        return SDValue();
      // Quit if not the same type.
      if (VecInMap.begin() != VecInMap.end() &&
          VT != VecInMap.begin()->first.getValueType())
        return SDValue();
      M = VecInMap.insert(std::make_pair(ExtractedFromVec, 0)).first;
    }
    M->second |= 1U << cast<ConstantSDNode>(Idx)->getZExtValue();
  }

  assert((VT.is128BitVector() || VT.is256BitVector()) &&
         "Not extracted from 128-/256-bit vector.");

  unsigned FullMask = (1U << VT.getVectorNumElements()) - 1U;
  SmallVector<SDValue, 8> VecIns;

  for (DenseMap<SDValue, unsigned>::const_iterator
        I = VecInMap.begin(), E = VecInMap.end(); I != E; ++I) {
    // Quit if not all elements are used.
    if (I->second != FullMask)
      return SDValue();
    VecIns.push_back(I->first);
  }

  EVT TestVT = VT.is128BitVector() ? MVT::v2i64 : MVT::v4i64;

  // Cast all vectors into TestVT for PTEST.
  for (unsigned i = 0, e = VecIns.size(); i < e; ++i)
    VecIns[i] = DAG.getNode(ISD::BITCAST, DL, TestVT, VecIns[i]);

  // If more than one full vectors are evaluated, OR them first before PTEST.
  for (unsigned Slot = 0, e = VecIns.size(); e - Slot > 1; Slot += 2, e += 1) {
    // Each iteration will OR 2 nodes and append the result until there is only
    // 1 node left, i.e. the final OR'd value of all vectors.
    SDValue LHS = VecIns[Slot];
    SDValue RHS = VecIns[Slot + 1];
    VecIns.push_back(DAG.getNode(ISD::OR, DL, TestVT, LHS, RHS));
  }

  return DAG.getNode(X86ISD::PTEST, DL, MVT::i32,
                     VecIns.back(), VecIns.back());
}

/// Emit nodes that will be selected as "test Op0,Op0", or something
/// equivalent.
SDValue X86TargetLowering::EmitTest(SDValue Op, unsigned X86CC,
                                    SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();

  // CF and OF aren't always set the way we want. Determine which
  // of these we need.
  bool NeedCF = false;
  bool NeedOF = false;
  switch (X86CC) {
  default: break;
  case X86::COND_A: case X86::COND_AE:
  case X86::COND_B: case X86::COND_BE:
    NeedCF = true;
    break;
  case X86::COND_G: case X86::COND_GE:
  case X86::COND_L: case X86::COND_LE:
  case X86::COND_O: case X86::COND_NO:
    NeedOF = true;
    break;
  }

  // See if we can use the EFLAGS value from the operand instead of
  // doing a separate TEST. TEST always sets OF and CF to 0, so unless
  // we prove that the arithmetic won't overflow, we can't use OF or CF.
  if (Op.getResNo() != 0 || NeedOF || NeedCF)
    // Emit a CMP with 0, which is the TEST pattern.
    return DAG.getNode(X86ISD::CMP, dl, MVT::i32, Op,
                       DAG.getConstant(0, Op.getValueType()));

  unsigned Opcode = 0;
  unsigned NumOperands = 0;

  // Truncate operations may prevent the merge of the SETCC instruction
  // and the arithmetic intruction before it. Attempt to truncate the operands
  // of the arithmetic instruction and use a reduced bit-width instruction.
  bool NeedTruncation = false;
  SDValue ArithOp = Op;
  if (Op->getOpcode() == ISD::TRUNCATE && Op->hasOneUse()) {
    SDValue Arith = Op->getOperand(0);
    // Both the trunc and the arithmetic op need to have one user each.
    if (Arith->hasOneUse())
      switch (Arith.getOpcode()) {
        default: break;
        case ISD::ADD:
        case ISD::SUB:
        case ISD::AND:
        case ISD::OR:
        case ISD::XOR: {
          NeedTruncation = true;
          ArithOp = Arith;
        }
      }
  }

  // NOTICE: In the code below we use ArithOp to hold the arithmetic operation
  // which may be the result of a CAST.  We use the variable 'Op', which is the
  // non-casted variable when we check for possible users.
  switch (ArithOp.getOpcode()) {
  case ISD::ADD:
    // Due to an isel shortcoming, be conservative if this add is likely to be
    // selected as part of a load-modify-store instruction. When the root node
    // in a match is a store, isel doesn't know how to remap non-chain non-flag
    // uses of other nodes in the match, such as the ADD in this case. This
    // leads to the ADD being left around and reselected, with the result being
    // two adds in the output.  Alas, even if none our users are stores, that
    // doesn't prove we're O.K.  Ergo, if we have any parents that aren't
    // CopyToReg or SETCC, eschew INC/DEC.  A better fix seems to require
    // climbing the DAG back to the root, and it doesn't seem to be worth the
    // effort.
    for (SDNode::use_iterator UI = Op.getNode()->use_begin(),
         UE = Op.getNode()->use_end(); UI != UE; ++UI)
      if (UI->getOpcode() != ISD::CopyToReg &&
          UI->getOpcode() != ISD::SETCC &&
          UI->getOpcode() != ISD::STORE)
        goto default_case;

    if (ConstantSDNode *C =
        dyn_cast<ConstantSDNode>(ArithOp.getNode()->getOperand(1))) {
      // An add of one will be selected as an INC.
      if (C->getAPIntValue() == 1) {
        Opcode = X86ISD::INC;
        NumOperands = 1;
        break;
      }

      // An add of negative one (subtract of one) will be selected as a DEC.
      if (C->getAPIntValue().isAllOnesValue()) {
        Opcode = X86ISD::DEC;
        NumOperands = 1;
        break;
      }
    }

    // Otherwise use a regular EFLAGS-setting add.
    Opcode = X86ISD::ADD;
    NumOperands = 2;
    break;
  case ISD::AND: {
    // If the primary and result isn't used, don't bother using X86ISD::AND,
    // because a TEST instruction will be better.
    bool NonFlagUse = false;
    for (SDNode::use_iterator UI = Op.getNode()->use_begin(),
           UE = Op.getNode()->use_end(); UI != UE; ++UI) {
      SDNode *User = *UI;
      unsigned UOpNo = UI.getOperandNo();
      if (User->getOpcode() == ISD::TRUNCATE && User->hasOneUse()) {
        // Look pass truncate.
        UOpNo = User->use_begin().getOperandNo();
        User = *User->use_begin();
      }

      if (User->getOpcode() != ISD::BRCOND &&
          User->getOpcode() != ISD::SETCC &&
          !(User->getOpcode() == ISD::SELECT && UOpNo == 0)) {
        NonFlagUse = true;
        break;
      }
    }

    if (!NonFlagUse)
      break;
  }
    // FALL THROUGH
  case ISD::SUB:
  case ISD::OR:
  case ISD::XOR:
    // Due to the ISEL shortcoming noted above, be conservative if this op is
    // likely to be selected as part of a load-modify-store instruction.
    for (SDNode::use_iterator UI = Op.getNode()->use_begin(),
           UE = Op.getNode()->use_end(); UI != UE; ++UI)
      if (UI->getOpcode() == ISD::STORE)
        goto default_case;

    // Otherwise use a regular EFLAGS-setting instruction.
    switch (ArithOp.getOpcode()) {
    default: llvm_unreachable("unexpected operator!");
    case ISD::SUB: Opcode = X86ISD::SUB; break;
    case ISD::XOR: Opcode = X86ISD::XOR; break;
    case ISD::AND: Opcode = X86ISD::AND; break;
    case ISD::OR: {
      if (!NeedTruncation && (X86CC == X86::COND_E || X86CC == X86::COND_NE)) {
        SDValue EFLAGS = LowerVectorAllZeroTest(Op, DAG);
        if (EFLAGS.getNode())
          return EFLAGS;
      }
      Opcode = X86ISD::OR;
      break;
    }
    }

    NumOperands = 2;
    break;
  case X86ISD::ADD:
  case X86ISD::SUB:
  case X86ISD::INC:
  case X86ISD::DEC:
  case X86ISD::OR:
  case X86ISD::XOR:
  case X86ISD::AND:
    return SDValue(Op.getNode(), 1);
  default:
  default_case:
    break;
  }

  // If we found that truncation is beneficial, perform the truncation and
  // update 'Op'.
  if (NeedTruncation) {
    EVT VT = Op.getValueType();
    SDValue WideVal = Op->getOperand(0);
    EVT WideVT = WideVal.getValueType();
    unsigned ConvertedOp = 0;
    // Use a target machine opcode to prevent further DAGCombine
    // optimizations that may separate the arithmetic operations
    // from the setcc node.
    switch (WideVal.getOpcode()) {
      default: break;
      case ISD::ADD: ConvertedOp = X86ISD::ADD; break;
      case ISD::SUB: ConvertedOp = X86ISD::SUB; break;
      case ISD::AND: ConvertedOp = X86ISD::AND; break;
      case ISD::OR:  ConvertedOp = X86ISD::OR;  break;
      case ISD::XOR: ConvertedOp = X86ISD::XOR; break;
    }

    if (ConvertedOp) {
      const TargetLowering &TLI = DAG.getTargetLoweringInfo();
      if (TLI.isOperationLegal(WideVal.getOpcode(), WideVT)) {
        SDValue V0 = DAG.getNode(ISD::TRUNCATE, dl, VT, WideVal.getOperand(0));
        SDValue V1 = DAG.getNode(ISD::TRUNCATE, dl, VT, WideVal.getOperand(1));
        Op = DAG.getNode(ConvertedOp, dl, VT, V0, V1);
      }
    }
  }

  if (Opcode == 0)
    // Emit a CMP with 0, which is the TEST pattern.
    return DAG.getNode(X86ISD::CMP, dl, MVT::i32, Op,
                       DAG.getConstant(0, Op.getValueType()));

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::i32);
  SmallVector<SDValue, 4> Ops;
  for (unsigned i = 0; i != NumOperands; ++i)
    Ops.push_back(Op.getOperand(i));

  SDValue New = DAG.getNode(Opcode, dl, VTs, &Ops[0], NumOperands);
  DAG.ReplaceAllUsesWith(Op, New);
  return SDValue(New.getNode(), 1);
}

/// Emit nodes that will be selected as "cmp Op0,Op1", or something
/// equivalent.
SDValue X86TargetLowering::EmitCmp(SDValue Op0, SDValue Op1, unsigned X86CC,
                                   SelectionDAG &DAG) const {
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op1))
    if (C->getAPIntValue() == 0)
      return EmitTest(Op0, X86CC, DAG);

  DebugLoc dl = Op0.getDebugLoc();
  if ((Op0.getValueType() == MVT::i8 || Op0.getValueType() == MVT::i16 ||
       Op0.getValueType() == MVT::i32 || Op0.getValueType() == MVT::i64)) {
    // Use SUB instead of CMP to enable CSE between SUB and CMP.
    SDVTList VTs = DAG.getVTList(Op0.getValueType(), MVT::i32);
    SDValue Sub = DAG.getNode(X86ISD::SUB, dl, VTs,
                              Op0, Op1);
    return SDValue(Sub.getNode(), 1);
  }
  return DAG.getNode(X86ISD::CMP, dl, MVT::i32, Op0, Op1);
}

/// Convert a comparison if required by the subtarget.
SDValue X86TargetLowering::ConvertCmpIfNecessary(SDValue Cmp,
                                                 SelectionDAG &DAG) const {
  // If the subtarget does not support the FUCOMI instruction, floating-point
  // comparisons have to be converted.
  if (Subtarget->hasCMov() ||
      Cmp.getOpcode() != X86ISD::CMP ||
      !Cmp.getOperand(0).getValueType().isFloatingPoint() ||
      !Cmp.getOperand(1).getValueType().isFloatingPoint())
    return Cmp;

  // The instruction selector will select an FUCOM instruction instead of
  // FUCOMI, which writes the comparison result to FPSW instead of EFLAGS. Hence
  // build an SDNode sequence that transfers the result from FPSW into EFLAGS:
  // (X86sahf (trunc (srl (X86fp_stsw (trunc (X86cmp ...)), 8))))
  DebugLoc dl = Cmp.getDebugLoc();
  SDValue TruncFPSW = DAG.getNode(ISD::TRUNCATE, dl, MVT::i16, Cmp);
  SDValue FNStSW = DAG.getNode(X86ISD::FNSTSW16r, dl, MVT::i16, TruncFPSW);
  SDValue Srl = DAG.getNode(ISD::SRL, dl, MVT::i16, FNStSW,
                            DAG.getConstant(8, MVT::i8));
  SDValue TruncSrl = DAG.getNode(ISD::TRUNCATE, dl, MVT::i8, Srl);
  return DAG.getNode(X86ISD::SAHF, dl, MVT::i32, TruncSrl);
}

/// LowerToBT - Result of 'and' is compared against zero. Turn it into a BT node
/// if it's possible.
SDValue X86TargetLowering::LowerToBT(SDValue And, ISD::CondCode CC,
                                     DebugLoc dl, SelectionDAG &DAG) const {
  SDValue Op0 = And.getOperand(0);
  SDValue Op1 = And.getOperand(1);
  if (Op0.getOpcode() == ISD::TRUNCATE)
    Op0 = Op0.getOperand(0);
  if (Op1.getOpcode() == ISD::TRUNCATE)
    Op1 = Op1.getOperand(0);

  SDValue LHS, RHS;
  if (Op1.getOpcode() == ISD::SHL)
    std::swap(Op0, Op1);
  if (Op0.getOpcode() == ISD::SHL) {
    if (ConstantSDNode *And00C = dyn_cast<ConstantSDNode>(Op0.getOperand(0)))
      if (And00C->getZExtValue() == 1) {
        // If we looked past a truncate, check that it's only truncating away
        // known zeros.
        unsigned BitWidth = Op0.getValueSizeInBits();
        unsigned AndBitWidth = And.getValueSizeInBits();
        if (BitWidth > AndBitWidth) {
          APInt Zeros, Ones;
          DAG.ComputeMaskedBits(Op0, Zeros, Ones);
          if (Zeros.countLeadingOnes() < BitWidth - AndBitWidth)
            return SDValue();
        }
        LHS = Op1;
        RHS = Op0.getOperand(1);
      }
  } else if (Op1.getOpcode() == ISD::Constant) {
    ConstantSDNode *AndRHS = cast<ConstantSDNode>(Op1);
    uint64_t AndRHSVal = AndRHS->getZExtValue();
    SDValue AndLHS = Op0;

    if (AndRHSVal == 1 && AndLHS.getOpcode() == ISD::SRL) {
      LHS = AndLHS.getOperand(0);
      RHS = AndLHS.getOperand(1);
    }

    // Use BT if the immediate can't be encoded in a TEST instruction.
    if (!isUInt<32>(AndRHSVal) && isPowerOf2_64(AndRHSVal)) {
      LHS = AndLHS;
      RHS = DAG.getConstant(Log2_64_Ceil(AndRHSVal), LHS.getValueType());
    }
  }

  if (LHS.getNode()) {
    // If LHS is i8, promote it to i32 with any_extend.  There is no i8 BT
    // instruction.  Since the shift amount is in-range-or-undefined, we know
    // that doing a bittest on the i32 value is ok.  We extend to i32 because
    // the encoding for the i16 version is larger than the i32 version.
    // Also promote i16 to i32 for performance / code size reason.
    if (LHS.getValueType() == MVT::i8 ||
        LHS.getValueType() == MVT::i16)
      LHS = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, LHS);

    // If the operand types disagree, extend the shift amount to match.  Since
    // BT ignores high bits (like shifts) we can use anyextend.
    if (LHS.getValueType() != RHS.getValueType())
      RHS = DAG.getNode(ISD::ANY_EXTEND, dl, LHS.getValueType(), RHS);

    SDValue BT = DAG.getNode(X86ISD::BT, dl, MVT::i32, LHS, RHS);
    unsigned Cond = CC == ISD::SETEQ ? X86::COND_AE : X86::COND_B;
    return DAG.getNode(X86ISD::SETCC, dl, MVT::i8,
                       DAG.getConstant(Cond, MVT::i8), BT);
  }

  return SDValue();
}

SDValue X86TargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {

  if (Op.getValueType().isVector()) return LowerVSETCC(Op, DAG);

  assert(Op.getValueType() == MVT::i8 && "SetCC type must be 8-bit integer");
  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();

  // Optimize to BT if possible.
  // Lower (X & (1 << N)) == 0 to BT(X, N).
  // Lower ((X >>u N) & 1) != 0 to BT(X, N).
  // Lower ((X >>s N) & 1) != 0 to BT(X, N).
  if (Op0.getOpcode() == ISD::AND && Op0.hasOneUse() &&
      Op1.getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(Op1)->isNullValue() &&
      (CC == ISD::SETEQ || CC == ISD::SETNE)) {
    SDValue NewSetCC = LowerToBT(Op0, CC, dl, DAG);
    if (NewSetCC.getNode())
      return NewSetCC;
  }

  // Look for X == 0, X == 1, X != 0, or X != 1.  We can simplify some forms of
  // these.
  if (Op1.getOpcode() == ISD::Constant &&
      (cast<ConstantSDNode>(Op1)->getZExtValue() == 1 ||
       cast<ConstantSDNode>(Op1)->isNullValue()) &&
      (CC == ISD::SETEQ || CC == ISD::SETNE)) {

    // If the input is a setcc, then reuse the input setcc or use a new one with
    // the inverted condition.
    if (Op0.getOpcode() == X86ISD::SETCC) {
      X86::CondCode CCode = (X86::CondCode)Op0.getConstantOperandVal(0);
      bool Invert = (CC == ISD::SETNE) ^
        cast<ConstantSDNode>(Op1)->isNullValue();
      if (!Invert) return Op0;

      CCode = X86::GetOppositeBranchCondition(CCode);
      return DAG.getNode(X86ISD::SETCC, dl, MVT::i8,
                         DAG.getConstant(CCode, MVT::i8), Op0.getOperand(1));
    }
  }

  bool isFP = Op1.getValueType().isFloatingPoint();
  unsigned X86CC = TranslateX86CC(CC, isFP, Op0, Op1, DAG);
  if (X86CC == X86::COND_INVALID)
    return SDValue();

  SDValue EFLAGS = EmitCmp(Op0, Op1, X86CC, DAG);
  EFLAGS = ConvertCmpIfNecessary(EFLAGS, DAG);
  return DAG.getNode(X86ISD::SETCC, dl, MVT::i8,
                     DAG.getConstant(X86CC, MVT::i8), EFLAGS);
}

// Lower256IntVSETCC - Break a VSETCC 256-bit integer VSETCC into two new 128
// ones, and then concatenate the result back.
static SDValue Lower256IntVSETCC(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();

  assert(VT.is256BitVector() && Op.getOpcode() == ISD::SETCC &&
         "Unsupported value type for operation");

  unsigned NumElems = VT.getVectorNumElements();
  DebugLoc dl = Op.getDebugLoc();
  SDValue CC = Op.getOperand(2);

  // Extract the LHS vectors
  SDValue LHS = Op.getOperand(0);
  SDValue LHS1 = Extract128BitVector(LHS, 0, DAG, dl);
  SDValue LHS2 = Extract128BitVector(LHS, NumElems/2, DAG, dl);

  // Extract the RHS vectors
  SDValue RHS = Op.getOperand(1);
  SDValue RHS1 = Extract128BitVector(RHS, 0, DAG, dl);
  SDValue RHS2 = Extract128BitVector(RHS, NumElems/2, DAG, dl);

  // Issue the operation on the smaller types and concatenate the result back
  MVT EltVT = VT.getVectorElementType().getSimpleVT();
  EVT NewVT = MVT::getVectorVT(EltVT, NumElems/2);
  return DAG.getNode(ISD::CONCAT_VECTORS, dl, VT,
                     DAG.getNode(Op.getOpcode(), dl, NewVT, LHS1, RHS1, CC),
                     DAG.getNode(Op.getOpcode(), dl, NewVT, LHS2, RHS2, CC));
}


SDValue X86TargetLowering::LowerVSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Cond;
  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue CC = Op.getOperand(2);
  EVT VT = Op.getValueType();
  ISD::CondCode SetCCOpcode = cast<CondCodeSDNode>(CC)->get();
  bool isFP = Op.getOperand(1).getValueType().isFloatingPoint();
  DebugLoc dl = Op.getDebugLoc();

  if (isFP) {
#ifndef NDEBUG
    EVT EltVT = Op0.getValueType().getVectorElementType();
    assert(EltVT == MVT::f32 || EltVT == MVT::f64);
#endif

    unsigned SSECC;
    bool Swap = false;

    // SSE Condition code mapping:
    //  0 - EQ
    //  1 - LT
    //  2 - LE
    //  3 - UNORD
    //  4 - NEQ
    //  5 - NLT
    //  6 - NLE
    //  7 - ORD
    switch (SetCCOpcode) {
    default: llvm_unreachable("Unexpected SETCC condition");
    case ISD::SETOEQ:
    case ISD::SETEQ:  SSECC = 0; break;
    case ISD::SETOGT:
    case ISD::SETGT: Swap = true; // Fallthrough
    case ISD::SETLT:
    case ISD::SETOLT: SSECC = 1; break;
    case ISD::SETOGE:
    case ISD::SETGE: Swap = true; // Fallthrough
    case ISD::SETLE:
    case ISD::SETOLE: SSECC = 2; break;
    case ISD::SETUO:  SSECC = 3; break;
    case ISD::SETUNE:
    case ISD::SETNE:  SSECC = 4; break;
    case ISD::SETULE: Swap = true; // Fallthrough
    case ISD::SETUGE: SSECC = 5; break;
    case ISD::SETULT: Swap = true; // Fallthrough
    case ISD::SETUGT: SSECC = 6; break;
    case ISD::SETO:   SSECC = 7; break;
    case ISD::SETUEQ:
    case ISD::SETONE: SSECC = 8; break;
    }
    if (Swap)
      std::swap(Op0, Op1);

    // In the two special cases we can't handle, emit two comparisons.
    if (SSECC == 8) {
      unsigned CC0, CC1;
      unsigned CombineOpc;
      if (SetCCOpcode == ISD::SETUEQ) {
        CC0 = 3; CC1 = 0; CombineOpc = ISD::OR;
      } else {
        assert(SetCCOpcode == ISD::SETONE);
        CC0 = 7; CC1 = 4; CombineOpc = ISD::AND;
      }

      SDValue Cmp0 = DAG.getNode(X86ISD::CMPP, dl, VT, Op0, Op1,
                                 DAG.getConstant(CC0, MVT::i8));
      SDValue Cmp1 = DAG.getNode(X86ISD::CMPP, dl, VT, Op0, Op1,
                                 DAG.getConstant(CC1, MVT::i8));
      return DAG.getNode(CombineOpc, dl, VT, Cmp0, Cmp1);
    }
    // Handle all other FP comparisons here.
    return DAG.getNode(X86ISD::CMPP, dl, VT, Op0, Op1,
                       DAG.getConstant(SSECC, MVT::i8));
  }

  // Break 256-bit integer vector compare into smaller ones.
  if (VT.is256BitVector() && !Subtarget->hasAVX2())
    return Lower256IntVSETCC(Op, DAG);

  // We are handling one of the integer comparisons here.  Since SSE only has
  // GT and EQ comparisons for integer, swapping operands and multiple
  // operations may be required for some comparisons.
  unsigned Opc;
  bool Swap = false, Invert = false, FlipSigns = false;

  switch (SetCCOpcode) {
  default: llvm_unreachable("Unexpected SETCC condition");
  case ISD::SETNE:  Invert = true;
  case ISD::SETEQ:  Opc = X86ISD::PCMPEQ; break;
  case ISD::SETLT:  Swap = true;
  case ISD::SETGT:  Opc = X86ISD::PCMPGT; break;
  case ISD::SETGE:  Swap = true;
  case ISD::SETLE:  Opc = X86ISD::PCMPGT; Invert = true; break;
  case ISD::SETULT: Swap = true;
  case ISD::SETUGT: Opc = X86ISD::PCMPGT; FlipSigns = true; break;
  case ISD::SETUGE: Swap = true;
  case ISD::SETULE: Opc = X86ISD::PCMPGT; FlipSigns = true; Invert = true; break;
  }
  if (Swap)
    std::swap(Op0, Op1);

  // Check that the operation in question is available (most are plain SSE2,
  // but PCMPGTQ and PCMPEQQ have different requirements).
  if (VT == MVT::v2i64) {
    if (Opc == X86ISD::PCMPGT && !Subtarget->hasSSE42())
      return SDValue();
    if (Opc == X86ISD::PCMPEQ && !Subtarget->hasSSE41())
      return SDValue();
  }

  // Since SSE has no unsigned integer comparisons, we need to flip  the sign
  // bits of the inputs before performing those operations.
  if (FlipSigns) {
    EVT EltVT = VT.getVectorElementType();
    SDValue SignBit = DAG.getConstant(APInt::getSignBit(EltVT.getSizeInBits()),
                                      EltVT);
    std::vector<SDValue> SignBits(VT.getVectorNumElements(), SignBit);
    SDValue SignVec = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &SignBits[0],
                                    SignBits.size());
    Op0 = DAG.getNode(ISD::XOR, dl, VT, Op0, SignVec);
    Op1 = DAG.getNode(ISD::XOR, dl, VT, Op1, SignVec);
  }

  SDValue Result = DAG.getNode(Opc, dl, VT, Op0, Op1);

  // If the logical-not of the result is required, perform that now.
  if (Invert)
    Result = DAG.getNOT(dl, Result, VT);

  return Result;
}

// isX86LogicalCmp - Return true if opcode is a X86 logical comparison.
static bool isX86LogicalCmp(SDValue Op) {
  unsigned Opc = Op.getNode()->getOpcode();
  if (Opc == X86ISD::CMP || Opc == X86ISD::COMI || Opc == X86ISD::UCOMI ||
      Opc == X86ISD::SAHF)
    return true;
  if (Op.getResNo() == 1 &&
      (Opc == X86ISD::ADD ||
       Opc == X86ISD::SUB ||
       Opc == X86ISD::ADC ||
       Opc == X86ISD::SBB ||
       Opc == X86ISD::SMUL ||
       Opc == X86ISD::UMUL ||
       Opc == X86ISD::INC ||
       Opc == X86ISD::DEC ||
       Opc == X86ISD::OR ||
       Opc == X86ISD::XOR ||
       Opc == X86ISD::AND))
    return true;

  if (Op.getResNo() == 2 && Opc == X86ISD::UMUL)
    return true;

  return false;
}

static bool isZero(SDValue V) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(V);
  return C && C->isNullValue();
}

static bool isAllOnes(SDValue V) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(V);
  return C && C->isAllOnesValue();
}

static bool isTruncWithZeroHighBitsInput(SDValue V, SelectionDAG &DAG) {
  if (V.getOpcode() != ISD::TRUNCATE)
    return false;

  SDValue VOp0 = V.getOperand(0);
  unsigned InBits = VOp0.getValueSizeInBits();
  unsigned Bits = V.getValueSizeInBits();
  return DAG.MaskedValueIsZero(VOp0, APInt::getHighBitsSet(InBits,InBits-Bits));
}

SDValue X86TargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  bool addTest = true;
  SDValue Cond  = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue Op2 = Op.getOperand(2);
  DebugLoc DL = Op.getDebugLoc();
  SDValue CC;

  if (Cond.getOpcode() == ISD::SETCC) {
    SDValue NewCond = LowerSETCC(Cond, DAG);
    if (NewCond.getNode())
      Cond = NewCond;
  }

  // (select (x == 0), -1, y) -> (sign_bit (x - 1)) | y
  // (select (x == 0), y, -1) -> ~(sign_bit (x - 1)) | y
  // (select (x != 0), y, -1) -> (sign_bit (x - 1)) | y
  // (select (x != 0), -1, y) -> ~(sign_bit (x - 1)) | y
  if (Cond.getOpcode() == X86ISD::SETCC &&
      Cond.getOperand(1).getOpcode() == X86ISD::CMP &&
      isZero(Cond.getOperand(1).getOperand(1))) {
    SDValue Cmp = Cond.getOperand(1);

    unsigned CondCode =cast<ConstantSDNode>(Cond.getOperand(0))->getZExtValue();

    if ((isAllOnes(Op1) || isAllOnes(Op2)) &&
        (CondCode == X86::COND_E || CondCode == X86::COND_NE)) {
      SDValue Y = isAllOnes(Op2) ? Op1 : Op2;

      SDValue CmpOp0 = Cmp.getOperand(0);
      // Apply further optimizations for special cases
      // (select (x != 0), -1, 0) -> neg & sbb
      // (select (x == 0), 0, -1) -> neg & sbb
      if (ConstantSDNode *YC = dyn_cast<ConstantSDNode>(Y))
        if (YC->isNullValue() &&
            (isAllOnes(Op1) == (CondCode == X86::COND_NE))) {
          SDVTList VTs = DAG.getVTList(CmpOp0.getValueType(), MVT::i32);
          SDValue Neg = DAG.getNode(X86ISD::SUB, DL, VTs,
                                    DAG.getConstant(0, CmpOp0.getValueType()),
                                    CmpOp0);
          SDValue Res = DAG.getNode(X86ISD::SETCC_CARRY, DL, Op.getValueType(),
                                    DAG.getConstant(X86::COND_B, MVT::i8),
                                    SDValue(Neg.getNode(), 1));
          return Res;
        }

      Cmp = DAG.getNode(X86ISD::CMP, DL, MVT::i32,
                        CmpOp0, DAG.getConstant(1, CmpOp0.getValueType()));
      Cmp = ConvertCmpIfNecessary(Cmp, DAG);

      SDValue Res =   // Res = 0 or -1.
        DAG.getNode(X86ISD::SETCC_CARRY, DL, Op.getValueType(),
                    DAG.getConstant(X86::COND_B, MVT::i8), Cmp);

      if (isAllOnes(Op1) != (CondCode == X86::COND_E))
        Res = DAG.getNOT(DL, Res, Res.getValueType());

      ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(Op2);
      if (N2C == 0 || !N2C->isNullValue())
        Res = DAG.getNode(ISD::OR, DL, Res.getValueType(), Res, Y);
      return Res;
    }
  }

  // Look past (and (setcc_carry (cmp ...)), 1).
  if (Cond.getOpcode() == ISD::AND &&
      Cond.getOperand(0).getOpcode() == X86ISD::SETCC_CARRY) {
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Cond.getOperand(1));
    if (C && C->getAPIntValue() == 1)
      Cond = Cond.getOperand(0);
  }

  // If condition flag is set by a X86ISD::CMP, then use it as the condition
  // setting operand in place of the X86ISD::SETCC.
  unsigned CondOpcode = Cond.getOpcode();
  if (CondOpcode == X86ISD::SETCC ||
      CondOpcode == X86ISD::SETCC_CARRY) {
    CC = Cond.getOperand(0);

    SDValue Cmp = Cond.getOperand(1);
    unsigned Opc = Cmp.getOpcode();
    EVT VT = Op.getValueType();

    bool IllegalFPCMov = false;
    if (VT.isFloatingPoint() && !VT.isVector() &&
        !isScalarFPTypeInSSEReg(VT))  // FPStack?
      IllegalFPCMov = !hasFPCMov(cast<ConstantSDNode>(CC)->getSExtValue());

    if ((isX86LogicalCmp(Cmp) && !IllegalFPCMov) ||
        Opc == X86ISD::BT) { // FIXME
      Cond = Cmp;
      addTest = false;
    }
  } else if (CondOpcode == ISD::USUBO || CondOpcode == ISD::SSUBO ||
             CondOpcode == ISD::UADDO || CondOpcode == ISD::SADDO ||
             ((CondOpcode == ISD::UMULO || CondOpcode == ISD::SMULO) &&
              Cond.getOperand(0).getValueType() != MVT::i8)) {
    SDValue LHS = Cond.getOperand(0);
    SDValue RHS = Cond.getOperand(1);
    unsigned X86Opcode;
    unsigned X86Cond;
    SDVTList VTs;
    switch (CondOpcode) {
    case ISD::UADDO: X86Opcode = X86ISD::ADD; X86Cond = X86::COND_B; break;
    case ISD::SADDO: X86Opcode = X86ISD::ADD; X86Cond = X86::COND_O; break;
    case ISD::USUBO: X86Opcode = X86ISD::SUB; X86Cond = X86::COND_B; break;
    case ISD::SSUBO: X86Opcode = X86ISD::SUB; X86Cond = X86::COND_O; break;
    case ISD::UMULO: X86Opcode = X86ISD::UMUL; X86Cond = X86::COND_O; break;
    case ISD::SMULO: X86Opcode = X86ISD::SMUL; X86Cond = X86::COND_O; break;
    default: llvm_unreachable("unexpected overflowing operator");
    }
    if (CondOpcode == ISD::UMULO)
      VTs = DAG.getVTList(LHS.getValueType(), LHS.getValueType(),
                          MVT::i32);
    else
      VTs = DAG.getVTList(LHS.getValueType(), MVT::i32);

    SDValue X86Op = DAG.getNode(X86Opcode, DL, VTs, LHS, RHS);

    if (CondOpcode == ISD::UMULO)
      Cond = X86Op.getValue(2);
    else
      Cond = X86Op.getValue(1);

    CC = DAG.getConstant(X86Cond, MVT::i8);
    addTest = false;
  }

  if (addTest) {
    // Look pass the truncate if the high bits are known zero.
    if (isTruncWithZeroHighBitsInput(Cond, DAG))
        Cond = Cond.getOperand(0);

    // We know the result of AND is compared against zero. Try to match
    // it to BT.
    if (Cond.getOpcode() == ISD::AND && Cond.hasOneUse()) {
      SDValue NewSetCC = LowerToBT(Cond, ISD::SETNE, DL, DAG);
      if (NewSetCC.getNode()) {
        CC = NewSetCC.getOperand(0);
        Cond = NewSetCC.getOperand(1);
        addTest = false;
      }
    }
  }

  if (addTest) {
    CC = DAG.getConstant(X86::COND_NE, MVT::i8);
    Cond = EmitTest(Cond, X86::COND_NE, DAG);
  }

  // a <  b ? -1 :  0 -> RES = ~setcc_carry
  // a <  b ?  0 : -1 -> RES = setcc_carry
  // a >= b ? -1 :  0 -> RES = setcc_carry
  // a >= b ?  0 : -1 -> RES = ~setcc_carry
  if (Cond.getOpcode() == X86ISD::SUB) {
    Cond = ConvertCmpIfNecessary(Cond, DAG);
    unsigned CondCode = cast<ConstantSDNode>(CC)->getZExtValue();

    if ((CondCode == X86::COND_AE || CondCode == X86::COND_B) &&
        (isAllOnes(Op1) || isAllOnes(Op2)) && (isZero(Op1) || isZero(Op2))) {
      SDValue Res = DAG.getNode(X86ISD::SETCC_CARRY, DL, Op.getValueType(),
                                DAG.getConstant(X86::COND_B, MVT::i8), Cond);
      if (isAllOnes(Op1) != (CondCode == X86::COND_B))
        return DAG.getNOT(DL, Res, Res.getValueType());
      return Res;
    }
  }

  // X86 doesn't have an i8 cmov. If both operands are the result of a truncate
  // widen the cmov and push the truncate through. This avoids introducing a new
  // branch during isel and doesn't add any extensions.
  if (Op.getValueType() == MVT::i8 &&
      Op1.getOpcode() == ISD::TRUNCATE && Op2.getOpcode() == ISD::TRUNCATE) {
    SDValue T1 = Op1.getOperand(0), T2 = Op2.getOperand(0);
    if (T1.getValueType() == T2.getValueType() &&
        // Blacklist CopyFromReg to avoid partial register stalls.
        T1.getOpcode() != ISD::CopyFromReg && T2.getOpcode()!=ISD::CopyFromReg){
      SDVTList VTs = DAG.getVTList(T1.getValueType(), MVT::Glue);
      SDValue Cmov = DAG.getNode(X86ISD::CMOV, DL, VTs, T2, T1, CC, Cond);
      return DAG.getNode(ISD::TRUNCATE, DL, Op.getValueType(), Cmov);
    }
  }

  // X86ISD::CMOV means set the result (which is operand 1) to the RHS if
  // condition is true.
  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SDValue Ops[] = { Op2, Op1, CC, Cond };
  return DAG.getNode(X86ISD::CMOV, DL, VTs, Ops, array_lengthof(Ops));
}

// isAndOrOfSingleUseSetCCs - Return true if node is an ISD::AND or
// ISD::OR of two X86ISD::SETCC nodes each of which has no other use apart
// from the AND / OR.
static bool isAndOrOfSetCCs(SDValue Op, unsigned &Opc) {
  Opc = Op.getOpcode();
  if (Opc != ISD::OR && Opc != ISD::AND)
    return false;
  return (Op.getOperand(0).getOpcode() == X86ISD::SETCC &&
          Op.getOperand(0).hasOneUse() &&
          Op.getOperand(1).getOpcode() == X86ISD::SETCC &&
          Op.getOperand(1).hasOneUse());
}

// isXor1OfSetCC - Return true if node is an ISD::XOR of a X86ISD::SETCC and
// 1 and that the SETCC node has a single use.
static bool isXor1OfSetCC(SDValue Op) {
  if (Op.getOpcode() != ISD::XOR)
    return false;
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
  if (N1C && N1C->getAPIntValue() == 1) {
    return Op.getOperand(0).getOpcode() == X86ISD::SETCC &&
      Op.getOperand(0).hasOneUse();
  }
  return false;
}

SDValue X86TargetLowering::LowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  bool addTest = true;
  SDValue Chain = Op.getOperand(0);
  SDValue Cond  = Op.getOperand(1);
  SDValue Dest  = Op.getOperand(2);
  DebugLoc dl = Op.getDebugLoc();
  SDValue CC;
  bool Inverted = false;

  if (Cond.getOpcode() == ISD::SETCC) {
    // Check for setcc([su]{add,sub,mul}o == 0).
    if (cast<CondCodeSDNode>(Cond.getOperand(2))->get() == ISD::SETEQ &&
        isa<ConstantSDNode>(Cond.getOperand(1)) &&
        cast<ConstantSDNode>(Cond.getOperand(1))->isNullValue() &&
        Cond.getOperand(0).getResNo() == 1 &&
        (Cond.getOperand(0).getOpcode() == ISD::SADDO ||
         Cond.getOperand(0).getOpcode() == ISD::UADDO ||
         Cond.getOperand(0).getOpcode() == ISD::SSUBO ||
         Cond.getOperand(0).getOpcode() == ISD::USUBO ||
         Cond.getOperand(0).getOpcode() == ISD::SMULO ||
         Cond.getOperand(0).getOpcode() == ISD::UMULO)) {
      Inverted = true;
      Cond = Cond.getOperand(0);
    } else {
      SDValue NewCond = LowerSETCC(Cond, DAG);
      if (NewCond.getNode())
        Cond = NewCond;
    }
  }
#if 0
  // FIXME: LowerXALUO doesn't handle these!!
  else if (Cond.getOpcode() == X86ISD::ADD  ||
           Cond.getOpcode() == X86ISD::SUB  ||
           Cond.getOpcode() == X86ISD::SMUL ||
           Cond.getOpcode() == X86ISD::UMUL)
    Cond = LowerXALUO(Cond, DAG);
#endif

  // Look pass (and (setcc_carry (cmp ...)), 1).
  if (Cond.getOpcode() == ISD::AND &&
      Cond.getOperand(0).getOpcode() == X86ISD::SETCC_CARRY) {
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Cond.getOperand(1));
    if (C && C->getAPIntValue() == 1)
      Cond = Cond.getOperand(0);
  }

  // If condition flag is set by a X86ISD::CMP, then use it as the condition
  // setting operand in place of the X86ISD::SETCC.
  unsigned CondOpcode = Cond.getOpcode();
  if (CondOpcode == X86ISD::SETCC ||
      CondOpcode == X86ISD::SETCC_CARRY) {
    CC = Cond.getOperand(0);

    SDValue Cmp = Cond.getOperand(1);
    unsigned Opc = Cmp.getOpcode();
    // FIXME: WHY THE SPECIAL CASING OF LogicalCmp??
    if (isX86LogicalCmp(Cmp) || Opc == X86ISD::BT) {
      Cond = Cmp;
      addTest = false;
    } else {
      switch (cast<ConstantSDNode>(CC)->getZExtValue()) {
      default: break;
      case X86::COND_O:
      case X86::COND_B:
        // These can only come from an arithmetic instruction with overflow,
        // e.g. SADDO, UADDO.
        Cond = Cond.getNode()->getOperand(1);
        addTest = false;
        break;
      }
    }
  }
  CondOpcode = Cond.getOpcode();
  if (CondOpcode == ISD::UADDO || CondOpcode == ISD::SADDO ||
      CondOpcode == ISD::USUBO || CondOpcode == ISD::SSUBO ||
      ((CondOpcode == ISD::UMULO || CondOpcode == ISD::SMULO) &&
       Cond.getOperand(0).getValueType() != MVT::i8)) {
    SDValue LHS = Cond.getOperand(0);
    SDValue RHS = Cond.getOperand(1);
    unsigned X86Opcode;
    unsigned X86Cond;
    SDVTList VTs;
    switch (CondOpcode) {
    case ISD::UADDO: X86Opcode = X86ISD::ADD; X86Cond = X86::COND_B; break;
    case ISD::SADDO: X86Opcode = X86ISD::ADD; X86Cond = X86::COND_O; break;
    case ISD::USUBO: X86Opcode = X86ISD::SUB; X86Cond = X86::COND_B; break;
    case ISD::SSUBO: X86Opcode = X86ISD::SUB; X86Cond = X86::COND_O; break;
    case ISD::UMULO: X86Opcode = X86ISD::UMUL; X86Cond = X86::COND_O; break;
    case ISD::SMULO: X86Opcode = X86ISD::SMUL; X86Cond = X86::COND_O; break;
    default: llvm_unreachable("unexpected overflowing operator");
    }
    if (Inverted)
      X86Cond = X86::GetOppositeBranchCondition((X86::CondCode)X86Cond);
    if (CondOpcode == ISD::UMULO)
      VTs = DAG.getVTList(LHS.getValueType(), LHS.getValueType(),
                          MVT::i32);
    else
      VTs = DAG.getVTList(LHS.getValueType(), MVT::i32);

    SDValue X86Op = DAG.getNode(X86Opcode, dl, VTs, LHS, RHS);

    if (CondOpcode == ISD::UMULO)
      Cond = X86Op.getValue(2);
    else
      Cond = X86Op.getValue(1);

    CC = DAG.getConstant(X86Cond, MVT::i8);
    addTest = false;
  } else {
    unsigned CondOpc;
    if (Cond.hasOneUse() && isAndOrOfSetCCs(Cond, CondOpc)) {
      SDValue Cmp = Cond.getOperand(0).getOperand(1);
      if (CondOpc == ISD::OR) {
        // Also, recognize the pattern generated by an FCMP_UNE. We can emit
        // two branches instead of an explicit OR instruction with a
        // separate test.
        if (Cmp == Cond.getOperand(1).getOperand(1) &&
            isX86LogicalCmp(Cmp)) {
          CC = Cond.getOperand(0).getOperand(0);
          Chain = DAG.getNode(X86ISD::BRCOND, dl, Op.getValueType(),
                              Chain, Dest, CC, Cmp);
          CC = Cond.getOperand(1).getOperand(0);
          Cond = Cmp;
          addTest = false;
        }
      } else { // ISD::AND
        // Also, recognize the pattern generated by an FCMP_OEQ. We can emit
        // two branches instead of an explicit AND instruction with a
        // separate test. However, we only do this if this block doesn't
        // have a fall-through edge, because this requires an explicit
        // jmp when the condition is false.
        if (Cmp == Cond.getOperand(1).getOperand(1) &&
            isX86LogicalCmp(Cmp) &&
            Op.getNode()->hasOneUse()) {
          X86::CondCode CCode =
            (X86::CondCode)Cond.getOperand(0).getConstantOperandVal(0);
          CCode = X86::GetOppositeBranchCondition(CCode);
          CC = DAG.getConstant(CCode, MVT::i8);
          SDNode *User = *Op.getNode()->use_begin();
          // Look for an unconditional branch following this conditional branch.
          // We need this because we need to reverse the successors in order
          // to implement FCMP_OEQ.
          if (User->getOpcode() == ISD::BR) {
            SDValue FalseBB = User->getOperand(1);
            SDNode *NewBR =
              DAG.UpdateNodeOperands(User, User->getOperand(0), Dest);
            assert(NewBR == User);
            (void)NewBR;
            Dest = FalseBB;

            Chain = DAG.getNode(X86ISD::BRCOND, dl, Op.getValueType(),
                                Chain, Dest, CC, Cmp);
            X86::CondCode CCode =
              (X86::CondCode)Cond.getOperand(1).getConstantOperandVal(0);
            CCode = X86::GetOppositeBranchCondition(CCode);
            CC = DAG.getConstant(CCode, MVT::i8);
            Cond = Cmp;
            addTest = false;
          }
        }
      }
    } else if (Cond.hasOneUse() && isXor1OfSetCC(Cond)) {
      // Recognize for xorb (setcc), 1 patterns. The xor inverts the condition.
      // It should be transformed during dag combiner except when the condition
      // is set by a arithmetics with overflow node.
      X86::CondCode CCode =
        (X86::CondCode)Cond.getOperand(0).getConstantOperandVal(0);
      CCode = X86::GetOppositeBranchCondition(CCode);
      CC = DAG.getConstant(CCode, MVT::i8);
      Cond = Cond.getOperand(0).getOperand(1);
      addTest = false;
    } else if (Cond.getOpcode() == ISD::SETCC &&
               cast<CondCodeSDNode>(Cond.getOperand(2))->get() == ISD::SETOEQ) {
      // For FCMP_OEQ, we can emit
      // two branches instead of an explicit AND instruction with a
      // separate test. However, we only do this if this block doesn't
      // have a fall-through edge, because this requires an explicit
      // jmp when the condition is false.
      if (Op.getNode()->hasOneUse()) {
        SDNode *User = *Op.getNode()->use_begin();
        // Look for an unconditional branch following this conditional branch.
        // We need this because we need to reverse the successors in order
        // to implement FCMP_OEQ.
        if (User->getOpcode() == ISD::BR) {
          SDValue FalseBB = User->getOperand(1);
          SDNode *NewBR =
            DAG.UpdateNodeOperands(User, User->getOperand(0), Dest);
          assert(NewBR == User);
          (void)NewBR;
          Dest = FalseBB;

          SDValue Cmp = DAG.getNode(X86ISD::CMP, dl, MVT::i32,
                                    Cond.getOperand(0), Cond.getOperand(1));
          Cmp = ConvertCmpIfNecessary(Cmp, DAG);
          CC = DAG.getConstant(X86::COND_NE, MVT::i8);
          Chain = DAG.getNode(X86ISD::BRCOND, dl, Op.getValueType(),
                              Chain, Dest, CC, Cmp);
          CC = DAG.getConstant(X86::COND_P, MVT::i8);
          Cond = Cmp;
          addTest = false;
        }
      }
    } else if (Cond.getOpcode() == ISD::SETCC &&
               cast<CondCodeSDNode>(Cond.getOperand(2))->get() == ISD::SETUNE) {
      // For FCMP_UNE, we can emit
      // two branches instead of an explicit AND instruction with a
      // separate test. However, we only do this if this block doesn't
      // have a fall-through edge, because this requires an explicit
      // jmp when the condition is false.
      if (Op.getNode()->hasOneUse()) {
        SDNode *User = *Op.getNode()->use_begin();
        // Look for an unconditional branch following this conditional branch.
        // We need this because we need to reverse the successors in order
        // to implement FCMP_UNE.
        if (User->getOpcode() == ISD::BR) {
          SDValue FalseBB = User->getOperand(1);
          SDNode *NewBR =
            DAG.UpdateNodeOperands(User, User->getOperand(0), Dest);
          assert(NewBR == User);
          (void)NewBR;

          SDValue Cmp = DAG.getNode(X86ISD::CMP, dl, MVT::i32,
                                    Cond.getOperand(0), Cond.getOperand(1));
          Cmp = ConvertCmpIfNecessary(Cmp, DAG);
          CC = DAG.getConstant(X86::COND_NE, MVT::i8);
          Chain = DAG.getNode(X86ISD::BRCOND, dl, Op.getValueType(),
                              Chain, Dest, CC, Cmp);
          CC = DAG.getConstant(X86::COND_NP, MVT::i8);
          Cond = Cmp;
          addTest = false;
          Dest = FalseBB;
        }
      }
    }
  }

  if (addTest) {
    // Look pass the truncate if the high bits are known zero.
    if (isTruncWithZeroHighBitsInput(Cond, DAG))
        Cond = Cond.getOperand(0);

    // We know the result of AND is compared against zero. Try to match
    // it to BT.
    if (Cond.getOpcode() == ISD::AND && Cond.hasOneUse()) {
      SDValue NewSetCC = LowerToBT(Cond, ISD::SETNE, dl, DAG);
      if (NewSetCC.getNode()) {
        CC = NewSetCC.getOperand(0);
        Cond = NewSetCC.getOperand(1);
        addTest = false;
      }
    }
  }

  if (addTest) {
    CC = DAG.getConstant(X86::COND_NE, MVT::i8);
    Cond = EmitTest(Cond, X86::COND_NE, DAG);
  }
  Cond = ConvertCmpIfNecessary(Cond, DAG);
  return DAG.getNode(X86ISD::BRCOND, dl, Op.getValueType(),
                     Chain, Dest, CC, Cond);
}


// Lower dynamic stack allocation to _alloca call for Cygwin/Mingw targets.
// Calls to _alloca is needed to probe the stack when allocating more than 4k
// bytes in one go. Touching the stack at 4K increments is necessary to ensure
// that the guard pages used by the OS virtual memory manager are allocated in
// correct sequence.
SDValue
X86TargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                           SelectionDAG &DAG) const {
  assert((Subtarget->isTargetCygMing() || Subtarget->isTargetWindows() ||
          getTargetMachine().Options.EnableSegmentedStacks) &&
         "This should be used only on Windows targets or when segmented stacks "
         "are being used");
  assert(!Subtarget->isTargetEnvMacho() && "Not implemented");
  DebugLoc dl = Op.getDebugLoc();

  // Get the inputs.
  SDValue Chain = Op.getOperand(0);
  SDValue Size  = Op.getOperand(1);
  // FIXME: Ensure alignment here

  bool Is64Bit = Subtarget->is64Bit();
  EVT SPTy = Is64Bit ? MVT::i64 : MVT::i32;

  if (getTargetMachine().Options.EnableSegmentedStacks) {
    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &MRI = MF.getRegInfo();

    if (Is64Bit) {
      // The 64 bit implementation of segmented stacks needs to clobber both r10
      // r11. This makes it impossible to use it along with nested parameters.
      const Function *F = MF.getFunction();

      for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
           I != E; ++I)
        if (I->hasNestAttr())
          report_fatal_error("Cannot use segmented stacks with functions that "
                             "have nested arguments.");
    }

    const TargetRegisterClass *AddrRegClass =
      getRegClassFor(Subtarget->is64Bit() ? MVT::i64:MVT::i32);
    unsigned Vreg = MRI.createVirtualRegister(AddrRegClass);
    Chain = DAG.getCopyToReg(Chain, dl, Vreg, Size);
    SDValue Value = DAG.getNode(X86ISD::SEG_ALLOCA, dl, SPTy, Chain,
                                DAG.getRegister(Vreg, SPTy));
    SDValue Ops1[2] = { Value, Chain };
    return DAG.getMergeValues(Ops1, 2, dl);
  } else {
    SDValue Flag;
    unsigned Reg = (Subtarget->is64Bit() ? X86::RAX : X86::EAX);

    Chain = DAG.getCopyToReg(Chain, dl, Reg, Size, Flag);
    Flag = Chain.getValue(1);
    SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

    Chain = DAG.getNode(X86ISD::WIN_ALLOCA, dl, NodeTys, Chain, Flag);
    Flag = Chain.getValue(1);

    Chain = DAG.getCopyFromReg(Chain, dl, RegInfo->getStackRegister(),
                               SPTy).getValue(1);

    SDValue Ops1[2] = { Chain.getValue(0), Chain };
    return DAG.getMergeValues(Ops1, 2, dl);
  }
}

SDValue X86TargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();

  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  DebugLoc DL = Op.getDebugLoc();

  if (!Subtarget->is64Bit() || Subtarget->isTargetWin64()) {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    SDValue FR = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                   getPointerTy());
    return DAG.getStore(Op.getOperand(0), DL, FR, Op.getOperand(1),
                        MachinePointerInfo(SV), false, false, 0);
  }

  // __va_list_tag:
  //   gp_offset         (0 - 6 * 8)
  //   fp_offset         (48 - 48 + 8 * 16)
  //   overflow_arg_area (point to parameters coming in memory).
  //   reg_save_area
  SmallVector<SDValue, 8> MemOps;
  SDValue FIN = Op.getOperand(1);
  // Store gp_offset
  SDValue Store = DAG.getStore(Op.getOperand(0), DL,
                               DAG.getConstant(FuncInfo->getVarArgsGPOffset(),
                                               MVT::i32),
                               FIN, MachinePointerInfo(SV), false, false, 0);
  MemOps.push_back(Store);

  // Store fp_offset
  FIN = DAG.getNode(ISD::ADD, DL, getPointerTy(),
                    FIN, DAG.getIntPtrConstant(4));
  Store = DAG.getStore(Op.getOperand(0), DL,
                       DAG.getConstant(FuncInfo->getVarArgsFPOffset(),
                                       MVT::i32),
                       FIN, MachinePointerInfo(SV, 4), false, false, 0);
  MemOps.push_back(Store);

  // Store ptr to overflow_arg_area
  FIN = DAG.getNode(ISD::ADD, DL, getPointerTy(),
                    FIN, DAG.getIntPtrConstant(4));
  SDValue OVFIN = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                    getPointerTy());
  Store = DAG.getStore(Op.getOperand(0), DL, OVFIN, FIN,
                       MachinePointerInfo(SV, 8),
                       false, false, 0);
  MemOps.push_back(Store);

  // Store ptr to reg_save_area.
  FIN = DAG.getNode(ISD::ADD, DL, getPointerTy(),
                    FIN, DAG.getIntPtrConstant(8));
  SDValue RSFIN = DAG.getFrameIndex(FuncInfo->getRegSaveFrameIndex(),
                                    getPointerTy());
  Store = DAG.getStore(Op.getOperand(0), DL, RSFIN, FIN,
                       MachinePointerInfo(SV, 16), false, false, 0);
  MemOps.push_back(Store);
  return DAG.getNode(ISD::TokenFactor, DL, MVT::Other,
                     &MemOps[0], MemOps.size());
}

SDValue X86TargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  assert(Subtarget->is64Bit() &&
         "LowerVAARG only handles 64-bit va_arg!");
  assert((Subtarget->isTargetLinux() ||
          Subtarget->isTargetDarwin()) &&
          "Unhandled target in LowerVAARG");
  assert(Op.getNode()->getNumOperands() == 4);
  SDValue Chain = Op.getOperand(0);
  SDValue SrcPtr = Op.getOperand(1);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  unsigned Align = Op.getConstantOperandVal(3);
  DebugLoc dl = Op.getDebugLoc();

  EVT ArgVT = Op.getNode()->getValueType(0);
  Type *ArgTy = ArgVT.getTypeForEVT(*DAG.getContext());
  uint32_t ArgSize = getDataLayout()->getTypeAllocSize(ArgTy);
  uint8_t ArgMode;

  // Decide which area this value should be read from.
  // TODO: Implement the AMD64 ABI in its entirety. This simple
  // selection mechanism works only for the basic types.
  if (ArgVT == MVT::f80) {
    llvm_unreachable("va_arg for f80 not yet implemented");
  } else if (ArgVT.isFloatingPoint() && ArgSize <= 16 /*bytes*/) {
    ArgMode = 2;  // Argument passed in XMM register. Use fp_offset.
  } else if (ArgVT.isInteger() && ArgSize <= 32 /*bytes*/) {
    ArgMode = 1;  // Argument passed in GPR64 register(s). Use gp_offset.
  } else {
    llvm_unreachable("Unhandled argument type in LowerVAARG");
  }

  if (ArgMode == 2) {
    // Sanity Check: Make sure using fp_offset makes sense.
    assert(!getTargetMachine().Options.UseSoftFloat &&
           !(DAG.getMachineFunction()
                .getFunction()->getFnAttributes()
                .hasAttribute(Attributes::NoImplicitFloat)) &&
           Subtarget->hasSSE1());
  }

  // Insert VAARG_64 node into the DAG
  // VAARG_64 returns two values: Variable Argument Address, Chain
  SmallVector<SDValue, 11> InstOps;
  InstOps.push_back(Chain);
  InstOps.push_back(SrcPtr);
  InstOps.push_back(DAG.getConstant(ArgSize, MVT::i32));
  InstOps.push_back(DAG.getConstant(ArgMode, MVT::i8));
  InstOps.push_back(DAG.getConstant(Align, MVT::i32));
  SDVTList VTs = DAG.getVTList(getPointerTy(), MVT::Other);
  SDValue VAARG = DAG.getMemIntrinsicNode(X86ISD::VAARG_64, dl,
                                          VTs, &InstOps[0], InstOps.size(),
                                          MVT::i64,
                                          MachinePointerInfo(SV),
                                          /*Align=*/0,
                                          /*Volatile=*/false,
                                          /*ReadMem=*/true,
                                          /*WriteMem=*/true);
  Chain = VAARG.getValue(1);

  // Load the next argument and return it
  return DAG.getLoad(ArgVT, dl,
                     Chain,
                     VAARG,
                     MachinePointerInfo(),
                     false, false, false, 0);
}

static SDValue LowerVACOPY(SDValue Op, const X86Subtarget *Subtarget,
                           SelectionDAG &DAG) {
  // X86-64 va_list is a struct { i32, i32, i8*, i8* }.
  assert(Subtarget->is64Bit() && "This code only handles 64-bit va_copy!");
  SDValue Chain = Op.getOperand(0);
  SDValue DstPtr = Op.getOperand(1);
  SDValue SrcPtr = Op.getOperand(2);
  const Value *DstSV = cast<SrcValueSDNode>(Op.getOperand(3))->getValue();
  const Value *SrcSV = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();
  DebugLoc DL = Op.getDebugLoc();

  return DAG.getMemcpy(Chain, DL, DstPtr, SrcPtr,
                       DAG.getIntPtrConstant(24), 8, /*isVolatile*/false,
                       false,
                       MachinePointerInfo(DstSV), MachinePointerInfo(SrcSV));
}

// getTargetVShiftNOde - Handle vector element shifts where the shift amount
// may or may not be a constant. Takes immediate version of shift as input.
static SDValue getTargetVShiftNode(unsigned Opc, DebugLoc dl, EVT VT,
                                   SDValue SrcOp, SDValue ShAmt,
                                   SelectionDAG &DAG) {
  assert(ShAmt.getValueType() == MVT::i32 && "ShAmt is not i32");

  if (isa<ConstantSDNode>(ShAmt)) {
    // Constant may be a TargetConstant. Use a regular constant.
    uint32_t ShiftAmt = cast<ConstantSDNode>(ShAmt)->getZExtValue();
    switch (Opc) {
      default: llvm_unreachable("Unknown target vector shift node");
      case X86ISD::VSHLI:
      case X86ISD::VSRLI:
      case X86ISD::VSRAI:
        return DAG.getNode(Opc, dl, VT, SrcOp,
                           DAG.getConstant(ShiftAmt, MVT::i32));
    }
  }

  // Change opcode to non-immediate version
  switch (Opc) {
    default: llvm_unreachable("Unknown target vector shift node");
    case X86ISD::VSHLI: Opc = X86ISD::VSHL; break;
    case X86ISD::VSRLI: Opc = X86ISD::VSRL; break;
    case X86ISD::VSRAI: Opc = X86ISD::VSRA; break;
  }

  // Need to build a vector containing shift amount
  // Shift amount is 32-bits, but SSE instructions read 64-bit, so fill with 0
  SDValue ShOps[4];
  ShOps[0] = ShAmt;
  ShOps[1] = DAG.getConstant(0, MVT::i32);
  ShOps[2] = ShOps[3] = DAG.getUNDEF(MVT::i32);
  ShAmt = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, &ShOps[0], 4);

  // The return type has to be a 128-bit type with the same element
  // type as the input type.
  MVT EltVT = VT.getVectorElementType().getSimpleVT();
  EVT ShVT = MVT::getVectorVT(EltVT, 128/EltVT.getSizeInBits());

  ShAmt = DAG.getNode(ISD::BITCAST, dl, ShVT, ShAmt);
  return DAG.getNode(Opc, dl, VT, SrcOp, ShAmt);
}

static SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  switch (IntNo) {
  default: return SDValue();    // Don't custom lower most intrinsics.
  // Comparison intrinsics.
  case Intrinsic::x86_sse_comieq_ss:
  case Intrinsic::x86_sse_comilt_ss:
  case Intrinsic::x86_sse_comile_ss:
  case Intrinsic::x86_sse_comigt_ss:
  case Intrinsic::x86_sse_comige_ss:
  case Intrinsic::x86_sse_comineq_ss:
  case Intrinsic::x86_sse_ucomieq_ss:
  case Intrinsic::x86_sse_ucomilt_ss:
  case Intrinsic::x86_sse_ucomile_ss:
  case Intrinsic::x86_sse_ucomigt_ss:
  case Intrinsic::x86_sse_ucomige_ss:
  case Intrinsic::x86_sse_ucomineq_ss:
  case Intrinsic::x86_sse2_comieq_sd:
  case Intrinsic::x86_sse2_comilt_sd:
  case Intrinsic::x86_sse2_comile_sd:
  case Intrinsic::x86_sse2_comigt_sd:
  case Intrinsic::x86_sse2_comige_sd:
  case Intrinsic::x86_sse2_comineq_sd:
  case Intrinsic::x86_sse2_ucomieq_sd:
  case Intrinsic::x86_sse2_ucomilt_sd:
  case Intrinsic::x86_sse2_ucomile_sd:
  case Intrinsic::x86_sse2_ucomigt_sd:
  case Intrinsic::x86_sse2_ucomige_sd:
  case Intrinsic::x86_sse2_ucomineq_sd: {
    unsigned Opc;
    ISD::CondCode CC;
    switch (IntNo) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::x86_sse_comieq_ss:
    case Intrinsic::x86_sse2_comieq_sd:
      Opc = X86ISD::COMI;
      CC = ISD::SETEQ;
      break;
    case Intrinsic::x86_sse_comilt_ss:
    case Intrinsic::x86_sse2_comilt_sd:
      Opc = X86ISD::COMI;
      CC = ISD::SETLT;
      break;
    case Intrinsic::x86_sse_comile_ss:
    case Intrinsic::x86_sse2_comile_sd:
      Opc = X86ISD::COMI;
      CC = ISD::SETLE;
      break;
    case Intrinsic::x86_sse_comigt_ss:
    case Intrinsic::x86_sse2_comigt_sd:
      Opc = X86ISD::COMI;
      CC = ISD::SETGT;
      break;
    case Intrinsic::x86_sse_comige_ss:
    case Intrinsic::x86_sse2_comige_sd:
      Opc = X86ISD::COMI;
      CC = ISD::SETGE;
      break;
    case Intrinsic::x86_sse_comineq_ss:
    case Intrinsic::x86_sse2_comineq_sd:
      Opc = X86ISD::COMI;
      CC = ISD::SETNE;
      break;
    case Intrinsic::x86_sse_ucomieq_ss:
    case Intrinsic::x86_sse2_ucomieq_sd:
      Opc = X86ISD::UCOMI;
      CC = ISD::SETEQ;
      break;
    case Intrinsic::x86_sse_ucomilt_ss:
    case Intrinsic::x86_sse2_ucomilt_sd:
      Opc = X86ISD::UCOMI;
      CC = ISD::SETLT;
      break;
    case Intrinsic::x86_sse_ucomile_ss:
    case Intrinsic::x86_sse2_ucomile_sd:
      Opc = X86ISD::UCOMI;
      CC = ISD::SETLE;
      break;
    case Intrinsic::x86_sse_ucomigt_ss:
    case Intrinsic::x86_sse2_ucomigt_sd:
      Opc = X86ISD::UCOMI;
      CC = ISD::SETGT;
      break;
    case Intrinsic::x86_sse_ucomige_ss:
    case Intrinsic::x86_sse2_ucomige_sd:
      Opc = X86ISD::UCOMI;
      CC = ISD::SETGE;
      break;
    case Intrinsic::x86_sse_ucomineq_ss:
    case Intrinsic::x86_sse2_ucomineq_sd:
      Opc = X86ISD::UCOMI;
      CC = ISD::SETNE;
      break;
    }

    SDValue LHS = Op.getOperand(1);
    SDValue RHS = Op.getOperand(2);
    unsigned X86CC = TranslateX86CC(CC, true, LHS, RHS, DAG);
    assert(X86CC != X86::COND_INVALID && "Unexpected illegal condition!");
    SDValue Cond = DAG.getNode(Opc, dl, MVT::i32, LHS, RHS);
    SDValue SetCC = DAG.getNode(X86ISD::SETCC, dl, MVT::i8,
                                DAG.getConstant(X86CC, MVT::i8), Cond);
    return DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, SetCC);
  }

  // Arithmetic intrinsics.
  case Intrinsic::x86_sse2_pmulu_dq:
  case Intrinsic::x86_avx2_pmulu_dq:
    return DAG.getNode(X86ISD::PMULUDQ, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));

  // SSE3/AVX horizontal add/sub intrinsics
  case Intrinsic::x86_sse3_hadd_ps:
  case Intrinsic::x86_sse3_hadd_pd:
  case Intrinsic::x86_avx_hadd_ps_256:
  case Intrinsic::x86_avx_hadd_pd_256:
  case Intrinsic::x86_sse3_hsub_ps:
  case Intrinsic::x86_sse3_hsub_pd:
  case Intrinsic::x86_avx_hsub_ps_256:
  case Intrinsic::x86_avx_hsub_pd_256:
  case Intrinsic::x86_ssse3_phadd_w_128:
  case Intrinsic::x86_ssse3_phadd_d_128:
  case Intrinsic::x86_avx2_phadd_w:
  case Intrinsic::x86_avx2_phadd_d:
  case Intrinsic::x86_ssse3_phsub_w_128:
  case Intrinsic::x86_ssse3_phsub_d_128:
  case Intrinsic::x86_avx2_phsub_w:
  case Intrinsic::x86_avx2_phsub_d: {
    unsigned Opcode;
    switch (IntNo) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::x86_sse3_hadd_ps:
    case Intrinsic::x86_sse3_hadd_pd:
    case Intrinsic::x86_avx_hadd_ps_256:
    case Intrinsic::x86_avx_hadd_pd_256:
      Opcode = X86ISD::FHADD;
      break;
    case Intrinsic::x86_sse3_hsub_ps:
    case Intrinsic::x86_sse3_hsub_pd:
    case Intrinsic::x86_avx_hsub_ps_256:
    case Intrinsic::x86_avx_hsub_pd_256:
      Opcode = X86ISD::FHSUB;
      break;
    case Intrinsic::x86_ssse3_phadd_w_128:
    case Intrinsic::x86_ssse3_phadd_d_128:
    case Intrinsic::x86_avx2_phadd_w:
    case Intrinsic::x86_avx2_phadd_d:
      Opcode = X86ISD::HADD;
      break;
    case Intrinsic::x86_ssse3_phsub_w_128:
    case Intrinsic::x86_ssse3_phsub_d_128:
    case Intrinsic::x86_avx2_phsub_w:
    case Intrinsic::x86_avx2_phsub_d:
      Opcode = X86ISD::HSUB;
      break;
    }
    return DAG.getNode(Opcode, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));
  }

  // AVX2 variable shift intrinsics
  case Intrinsic::x86_avx2_psllv_d:
  case Intrinsic::x86_avx2_psllv_q:
  case Intrinsic::x86_avx2_psllv_d_256:
  case Intrinsic::x86_avx2_psllv_q_256:
  case Intrinsic::x86_avx2_psrlv_d:
  case Intrinsic::x86_avx2_psrlv_q:
  case Intrinsic::x86_avx2_psrlv_d_256:
  case Intrinsic::x86_avx2_psrlv_q_256:
  case Intrinsic::x86_avx2_psrav_d:
  case Intrinsic::x86_avx2_psrav_d_256: {
    unsigned Opcode;
    switch (IntNo) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::x86_avx2_psllv_d:
    case Intrinsic::x86_avx2_psllv_q:
    case Intrinsic::x86_avx2_psllv_d_256:
    case Intrinsic::x86_avx2_psllv_q_256:
      Opcode = ISD::SHL;
      break;
    case Intrinsic::x86_avx2_psrlv_d:
    case Intrinsic::x86_avx2_psrlv_q:
    case Intrinsic::x86_avx2_psrlv_d_256:
    case Intrinsic::x86_avx2_psrlv_q_256:
      Opcode = ISD::SRL;
      break;
    case Intrinsic::x86_avx2_psrav_d:
    case Intrinsic::x86_avx2_psrav_d_256:
      Opcode = ISD::SRA;
      break;
    }
    return DAG.getNode(Opcode, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));
  }

  case Intrinsic::x86_ssse3_pshuf_b_128:
  case Intrinsic::x86_avx2_pshuf_b:
    return DAG.getNode(X86ISD::PSHUFB, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));

  case Intrinsic::x86_ssse3_psign_b_128:
  case Intrinsic::x86_ssse3_psign_w_128:
  case Intrinsic::x86_ssse3_psign_d_128:
  case Intrinsic::x86_avx2_psign_b:
  case Intrinsic::x86_avx2_psign_w:
  case Intrinsic::x86_avx2_psign_d:
    return DAG.getNode(X86ISD::PSIGN, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));

  case Intrinsic::x86_sse41_insertps:
    return DAG.getNode(X86ISD::INSERTPS, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

  case Intrinsic::x86_avx_vperm2f128_ps_256:
  case Intrinsic::x86_avx_vperm2f128_pd_256:
  case Intrinsic::x86_avx_vperm2f128_si_256:
  case Intrinsic::x86_avx2_vperm2i128:
    return DAG.getNode(X86ISD::VPERM2X128, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

  case Intrinsic::x86_avx2_permd:
  case Intrinsic::x86_avx2_permps:
    // Operands intentionally swapped. Mask is last operand to intrinsic,
    // but second operand for node/intruction.
    return DAG.getNode(X86ISD::VPERMV, dl, Op.getValueType(),
                       Op.getOperand(2), Op.getOperand(1));

  // ptest and testp intrinsics. The intrinsic these come from are designed to
  // return an integer value, not just an instruction so lower it to the ptest
  // or testp pattern and a setcc for the result.
  case Intrinsic::x86_sse41_ptestz:
  case Intrinsic::x86_sse41_ptestc:
  case Intrinsic::x86_sse41_ptestnzc:
  case Intrinsic::x86_avx_ptestz_256:
  case Intrinsic::x86_avx_ptestc_256:
  case Intrinsic::x86_avx_ptestnzc_256:
  case Intrinsic::x86_avx_vtestz_ps:
  case Intrinsic::x86_avx_vtestc_ps:
  case Intrinsic::x86_avx_vtestnzc_ps:
  case Intrinsic::x86_avx_vtestz_pd:
  case Intrinsic::x86_avx_vtestc_pd:
  case Intrinsic::x86_avx_vtestnzc_pd:
  case Intrinsic::x86_avx_vtestz_ps_256:
  case Intrinsic::x86_avx_vtestc_ps_256:
  case Intrinsic::x86_avx_vtestnzc_ps_256:
  case Intrinsic::x86_avx_vtestz_pd_256:
  case Intrinsic::x86_avx_vtestc_pd_256:
  case Intrinsic::x86_avx_vtestnzc_pd_256: {
    bool IsTestPacked = false;
    unsigned X86CC;
    switch (IntNo) {
    default: llvm_unreachable("Bad fallthrough in Intrinsic lowering.");
    case Intrinsic::x86_avx_vtestz_ps:
    case Intrinsic::x86_avx_vtestz_pd:
    case Intrinsic::x86_avx_vtestz_ps_256:
    case Intrinsic::x86_avx_vtestz_pd_256:
      IsTestPacked = true; // Fallthrough
    case Intrinsic::x86_sse41_ptestz:
    case Intrinsic::x86_avx_ptestz_256:
      // ZF = 1
      X86CC = X86::COND_E;
      break;
    case Intrinsic::x86_avx_vtestc_ps:
    case Intrinsic::x86_avx_vtestc_pd:
    case Intrinsic::x86_avx_vtestc_ps_256:
    case Intrinsic::x86_avx_vtestc_pd_256:
      IsTestPacked = true; // Fallthrough
    case Intrinsic::x86_sse41_ptestc:
    case Intrinsic::x86_avx_ptestc_256:
      // CF = 1
      X86CC = X86::COND_B;
      break;
    case Intrinsic::x86_avx_vtestnzc_ps:
    case Intrinsic::x86_avx_vtestnzc_pd:
    case Intrinsic::x86_avx_vtestnzc_ps_256:
    case Intrinsic::x86_avx_vtestnzc_pd_256:
      IsTestPacked = true; // Fallthrough
    case Intrinsic::x86_sse41_ptestnzc:
    case Intrinsic::x86_avx_ptestnzc_256:
      // ZF and CF = 0
      X86CC = X86::COND_A;
      break;
    }

    SDValue LHS = Op.getOperand(1);
    SDValue RHS = Op.getOperand(2);
    unsigned TestOpc = IsTestPacked ? X86ISD::TESTP : X86ISD::PTEST;
    SDValue Test = DAG.getNode(TestOpc, dl, MVT::i32, LHS, RHS);
    SDValue CC = DAG.getConstant(X86CC, MVT::i8);
    SDValue SetCC = DAG.getNode(X86ISD::SETCC, dl, MVT::i8, CC, Test);
    return DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, SetCC);
  }

  // SSE/AVX shift intrinsics
  case Intrinsic::x86_sse2_psll_w:
  case Intrinsic::x86_sse2_psll_d:
  case Intrinsic::x86_sse2_psll_q:
  case Intrinsic::x86_avx2_psll_w:
  case Intrinsic::x86_avx2_psll_d:
  case Intrinsic::x86_avx2_psll_q:
  case Intrinsic::x86_sse2_psrl_w:
  case Intrinsic::x86_sse2_psrl_d:
  case Intrinsic::x86_sse2_psrl_q:
  case Intrinsic::x86_avx2_psrl_w:
  case Intrinsic::x86_avx2_psrl_d:
  case Intrinsic::x86_avx2_psrl_q:
  case Intrinsic::x86_sse2_psra_w:
  case Intrinsic::x86_sse2_psra_d:
  case Intrinsic::x86_avx2_psra_w:
  case Intrinsic::x86_avx2_psra_d: {
    unsigned Opcode;
    switch (IntNo) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::x86_sse2_psll_w:
    case Intrinsic::x86_sse2_psll_d:
    case Intrinsic::x86_sse2_psll_q:
    case Intrinsic::x86_avx2_psll_w:
    case Intrinsic::x86_avx2_psll_d:
    case Intrinsic::x86_avx2_psll_q:
      Opcode = X86ISD::VSHL;
      break;
    case Intrinsic::x86_sse2_psrl_w:
    case Intrinsic::x86_sse2_psrl_d:
    case Intrinsic::x86_sse2_psrl_q:
    case Intrinsic::x86_avx2_psrl_w:
    case Intrinsic::x86_avx2_psrl_d:
    case Intrinsic::x86_avx2_psrl_q:
      Opcode = X86ISD::VSRL;
      break;
    case Intrinsic::x86_sse2_psra_w:
    case Intrinsic::x86_sse2_psra_d:
    case Intrinsic::x86_avx2_psra_w:
    case Intrinsic::x86_avx2_psra_d:
      Opcode = X86ISD::VSRA;
      break;
    }
    return DAG.getNode(Opcode, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));
  }

  // SSE/AVX immediate shift intrinsics
  case Intrinsic::x86_sse2_pslli_w:
  case Intrinsic::x86_sse2_pslli_d:
  case Intrinsic::x86_sse2_pslli_q:
  case Intrinsic::x86_avx2_pslli_w:
  case Intrinsic::x86_avx2_pslli_d:
  case Intrinsic::x86_avx2_pslli_q:
  case Intrinsic::x86_sse2_psrli_w:
  case Intrinsic::x86_sse2_psrli_d:
  case Intrinsic::x86_sse2_psrli_q:
  case Intrinsic::x86_avx2_psrli_w:
  case Intrinsic::x86_avx2_psrli_d:
  case Intrinsic::x86_avx2_psrli_q:
  case Intrinsic::x86_sse2_psrai_w:
  case Intrinsic::x86_sse2_psrai_d:
  case Intrinsic::x86_avx2_psrai_w:
  case Intrinsic::x86_avx2_psrai_d: {
    unsigned Opcode;
    switch (IntNo) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::x86_sse2_pslli_w:
    case Intrinsic::x86_sse2_pslli_d:
    case Intrinsic::x86_sse2_pslli_q:
    case Intrinsic::x86_avx2_pslli_w:
    case Intrinsic::x86_avx2_pslli_d:
    case Intrinsic::x86_avx2_pslli_q:
      Opcode = X86ISD::VSHLI;
      break;
    case Intrinsic::x86_sse2_psrli_w:
    case Intrinsic::x86_sse2_psrli_d:
    case Intrinsic::x86_sse2_psrli_q:
    case Intrinsic::x86_avx2_psrli_w:
    case Intrinsic::x86_avx2_psrli_d:
    case Intrinsic::x86_avx2_psrli_q:
      Opcode = X86ISD::VSRLI;
      break;
    case Intrinsic::x86_sse2_psrai_w:
    case Intrinsic::x86_sse2_psrai_d:
    case Intrinsic::x86_avx2_psrai_w:
    case Intrinsic::x86_avx2_psrai_d:
      Opcode = X86ISD::VSRAI;
      break;
    }
    return getTargetVShiftNode(Opcode, dl, Op.getValueType(),
                               Op.getOperand(1), Op.getOperand(2), DAG);
  }

  case Intrinsic::x86_sse42_pcmpistria128:
  case Intrinsic::x86_sse42_pcmpestria128:
  case Intrinsic::x86_sse42_pcmpistric128:
  case Intrinsic::x86_sse42_pcmpestric128:
  case Intrinsic::x86_sse42_pcmpistrio128:
  case Intrinsic::x86_sse42_pcmpestrio128:
  case Intrinsic::x86_sse42_pcmpistris128:
  case Intrinsic::x86_sse42_pcmpestris128:
  case Intrinsic::x86_sse42_pcmpistriz128:
  case Intrinsic::x86_sse42_pcmpestriz128: {
    unsigned Opcode;
    unsigned X86CC;
    switch (IntNo) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::x86_sse42_pcmpistria128:
      Opcode = X86ISD::PCMPISTRI;
      X86CC = X86::COND_A;
      break;
    case Intrinsic::x86_sse42_pcmpestria128:
      Opcode = X86ISD::PCMPESTRI;
      X86CC = X86::COND_A;
      break;
    case Intrinsic::x86_sse42_pcmpistric128:
      Opcode = X86ISD::PCMPISTRI;
      X86CC = X86::COND_B;
      break;
    case Intrinsic::x86_sse42_pcmpestric128:
      Opcode = X86ISD::PCMPESTRI;
      X86CC = X86::COND_B;
      break;
    case Intrinsic::x86_sse42_pcmpistrio128:
      Opcode = X86ISD::PCMPISTRI;
      X86CC = X86::COND_O;
      break;
    case Intrinsic::x86_sse42_pcmpestrio128:
      Opcode = X86ISD::PCMPESTRI;
      X86CC = X86::COND_O;
      break;
    case Intrinsic::x86_sse42_pcmpistris128:
      Opcode = X86ISD::PCMPISTRI;
      X86CC = X86::COND_S;
      break;
    case Intrinsic::x86_sse42_pcmpestris128:
      Opcode = X86ISD::PCMPESTRI;
      X86CC = X86::COND_S;
      break;
    case Intrinsic::x86_sse42_pcmpistriz128:
      Opcode = X86ISD::PCMPISTRI;
      X86CC = X86::COND_E;
      break;
    case Intrinsic::x86_sse42_pcmpestriz128:
      Opcode = X86ISD::PCMPESTRI;
      X86CC = X86::COND_E;
      break;
    }
    SmallVector<SDValue, 5> NewOps;
    NewOps.append(Op->op_begin()+1, Op->op_end());
    SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::i32);
    SDValue PCMP = DAG.getNode(Opcode, dl, VTs, NewOps.data(), NewOps.size());
    SDValue SetCC = DAG.getNode(X86ISD::SETCC, dl, MVT::i8,
                                DAG.getConstant(X86CC, MVT::i8),
                                SDValue(PCMP.getNode(), 1));
    return DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, SetCC);
  }

  case Intrinsic::x86_sse42_pcmpistri128:
  case Intrinsic::x86_sse42_pcmpestri128: {
    unsigned Opcode;
    if (IntNo == Intrinsic::x86_sse42_pcmpistri128)
      Opcode = X86ISD::PCMPISTRI;
    else
      Opcode = X86ISD::PCMPESTRI;

    SmallVector<SDValue, 5> NewOps;
    NewOps.append(Op->op_begin()+1, Op->op_end());
    SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::i32);
    return DAG.getNode(Opcode, dl, VTs, NewOps.data(), NewOps.size());
  }
  case Intrinsic::x86_fma_vfmadd_ps:
  case Intrinsic::x86_fma_vfmadd_pd:
  case Intrinsic::x86_fma_vfmsub_ps:
  case Intrinsic::x86_fma_vfmsub_pd:
  case Intrinsic::x86_fma_vfnmadd_ps:
  case Intrinsic::x86_fma_vfnmadd_pd:
  case Intrinsic::x86_fma_vfnmsub_ps:
  case Intrinsic::x86_fma_vfnmsub_pd:
  case Intrinsic::x86_fma_vfmaddsub_ps:
  case Intrinsic::x86_fma_vfmaddsub_pd:
  case Intrinsic::x86_fma_vfmsubadd_ps:
  case Intrinsic::x86_fma_vfmsubadd_pd:
  case Intrinsic::x86_fma_vfmadd_ps_256:
  case Intrinsic::x86_fma_vfmadd_pd_256:
  case Intrinsic::x86_fma_vfmsub_ps_256:
  case Intrinsic::x86_fma_vfmsub_pd_256:
  case Intrinsic::x86_fma_vfnmadd_ps_256:
  case Intrinsic::x86_fma_vfnmadd_pd_256:
  case Intrinsic::x86_fma_vfnmsub_ps_256:
  case Intrinsic::x86_fma_vfnmsub_pd_256:
  case Intrinsic::x86_fma_vfmaddsub_ps_256:
  case Intrinsic::x86_fma_vfmaddsub_pd_256:
  case Intrinsic::x86_fma_vfmsubadd_ps_256:
  case Intrinsic::x86_fma_vfmsubadd_pd_256: {
    unsigned Opc;
    switch (IntNo) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::x86_fma_vfmadd_ps:
    case Intrinsic::x86_fma_vfmadd_pd:
    case Intrinsic::x86_fma_vfmadd_ps_256:
    case Intrinsic::x86_fma_vfmadd_pd_256:
      Opc = X86ISD::FMADD;
      break;
    case Intrinsic::x86_fma_vfmsub_ps:
    case Intrinsic::x86_fma_vfmsub_pd:
    case Intrinsic::x86_fma_vfmsub_ps_256:
    case Intrinsic::x86_fma_vfmsub_pd_256:
      Opc = X86ISD::FMSUB;
      break;
    case Intrinsic::x86_fma_vfnmadd_ps:
    case Intrinsic::x86_fma_vfnmadd_pd:
    case Intrinsic::x86_fma_vfnmadd_ps_256:
    case Intrinsic::x86_fma_vfnmadd_pd_256:
      Opc = X86ISD::FNMADD;
      break;
    case Intrinsic::x86_fma_vfnmsub_ps:
    case Intrinsic::x86_fma_vfnmsub_pd:
    case Intrinsic::x86_fma_vfnmsub_ps_256:
    case Intrinsic::x86_fma_vfnmsub_pd_256:
      Opc = X86ISD::FNMSUB;
      break;
    case Intrinsic::x86_fma_vfmaddsub_ps:
    case Intrinsic::x86_fma_vfmaddsub_pd:
    case Intrinsic::x86_fma_vfmaddsub_ps_256:
    case Intrinsic::x86_fma_vfmaddsub_pd_256:
      Opc = X86ISD::FMADDSUB;
      break;
    case Intrinsic::x86_fma_vfmsubadd_ps:
    case Intrinsic::x86_fma_vfmsubadd_pd:
    case Intrinsic::x86_fma_vfmsubadd_ps_256:
    case Intrinsic::x86_fma_vfmsubadd_pd_256:
      Opc = X86ISD::FMSUBADD;
      break;
    }

    return DAG.getNode(Opc, dl, Op.getValueType(), Op.getOperand(1),
                       Op.getOperand(2), Op.getOperand(3));
  }
  }
}

static SDValue LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  switch (IntNo) {
  default: return SDValue();    // Don't custom lower most intrinsics.

  // RDRAND intrinsics.
  case Intrinsic::x86_rdrand_16:
  case Intrinsic::x86_rdrand_32:
  case Intrinsic::x86_rdrand_64: {
    // Emit the node with the right value type.
    SDVTList VTs = DAG.getVTList(Op->getValueType(0), MVT::Glue, MVT::Other);
    SDValue Result = DAG.getNode(X86ISD::RDRAND, dl, VTs, Op.getOperand(0));

    // If the value returned by RDRAND was valid (CF=1), return 1. Otherwise
    // return the value from Rand, which is always 0, casted to i32.
    SDValue Ops[] = { DAG.getZExtOrTrunc(Result, dl, Op->getValueType(1)),
                      DAG.getConstant(1, Op->getValueType(1)),
                      DAG.getConstant(X86::COND_B, MVT::i32),
                      SDValue(Result.getNode(), 1) };
    SDValue isValid = DAG.getNode(X86ISD::CMOV, dl,
                                  DAG.getVTList(Op->getValueType(1), MVT::Glue),
                                  Ops, 4);

    // Return { result, isValid, chain }.
    return DAG.getNode(ISD::MERGE_VALUES, dl, Op->getVTList(), Result, isValid,
                       SDValue(Result.getNode(), 2));
  }
  }
}

SDValue X86TargetLowering::LowerRETURNADDR(SDValue Op,
                                           SelectionDAG &DAG) const {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setReturnAddressIsTaken(true);

  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  DebugLoc dl = Op.getDebugLoc();
  EVT PtrVT = getPointerTy();

  if (Depth > 0) {
    SDValue FrameAddr = LowerFRAMEADDR(Op, DAG);
    SDValue Offset =
      DAG.getConstant(RegInfo->getSlotSize(), PtrVT);
    return DAG.getLoad(PtrVT, dl, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, dl, PtrVT,
                                   FrameAddr, Offset),
                       MachinePointerInfo(), false, false, false, 0);
  }

  // Just load the return address.
  SDValue RetAddrFI = getReturnAddressFrameIndex(DAG);
  return DAG.getLoad(PtrVT, dl, DAG.getEntryNode(),
                     RetAddrFI, MachinePointerInfo(), false, false, false, 0);
}

SDValue X86TargetLowering::LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setFrameAddressIsTaken(true);

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();  // FIXME probably not meaningful
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  unsigned FrameReg = Subtarget->is64Bit() ? X86::RBP : X86::EBP;
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl, FrameReg, VT);
  while (Depth--)
    FrameAddr = DAG.getLoad(VT, dl, DAG.getEntryNode(), FrameAddr,
                            MachinePointerInfo(),
                            false, false, false, 0);
  return FrameAddr;
}

SDValue X86TargetLowering::LowerFRAME_TO_ARGS_OFFSET(SDValue Op,
                                                     SelectionDAG &DAG) const {
  return DAG.getIntPtrConstant(2 * RegInfo->getSlotSize());
}

SDValue X86TargetLowering::LowerEH_RETURN(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain     = Op.getOperand(0);
  SDValue Offset    = Op.getOperand(1);
  SDValue Handler   = Op.getOperand(2);
  DebugLoc dl       = Op.getDebugLoc();

  SDValue Frame = DAG.getCopyFromReg(DAG.getEntryNode(), dl,
                                     Subtarget->is64Bit() ? X86::RBP : X86::EBP,
                                     getPointerTy());
  unsigned StoreAddrReg = (Subtarget->is64Bit() ? X86::RCX : X86::ECX);

  SDValue StoreAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), Frame,
                                  DAG.getIntPtrConstant(RegInfo->getSlotSize()));
  StoreAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), StoreAddr, Offset);
  Chain = DAG.getStore(Chain, dl, Handler, StoreAddr, MachinePointerInfo(),
                       false, false, 0);
  Chain = DAG.getCopyToReg(Chain, dl, StoreAddrReg, StoreAddr);

  return DAG.getNode(X86ISD::EH_RETURN, dl,
                     MVT::Other,
                     Chain, DAG.getRegister(StoreAddrReg, getPointerTy()));
}

SDValue X86TargetLowering::lowerEH_SJLJ_SETJMP(SDValue Op,
                                               SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  return DAG.getNode(X86ISD::EH_SJLJ_SETJMP, DL,
                     DAG.getVTList(MVT::i32, MVT::Other),
                     Op.getOperand(0), Op.getOperand(1));
}

SDValue X86TargetLowering::lowerEH_SJLJ_LONGJMP(SDValue Op,
                                                SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  return DAG.getNode(X86ISD::EH_SJLJ_LONGJMP, DL, MVT::Other,
                     Op.getOperand(0), Op.getOperand(1));
}

static SDValue LowerADJUST_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) {
  return Op.getOperand(0);
}

SDValue X86TargetLowering::LowerINIT_TRAMPOLINE(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDValue Root = Op.getOperand(0);
  SDValue Trmp = Op.getOperand(1); // trampoline
  SDValue FPtr = Op.getOperand(2); // nested function
  SDValue Nest = Op.getOperand(3); // 'nest' parameter value
  DebugLoc dl  = Op.getDebugLoc();

  const Value *TrmpAddr = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();
  const TargetRegisterInfo* TRI = getTargetMachine().getRegisterInfo();

  if (Subtarget->is64Bit()) {
    SDValue OutChains[6];

    // Large code-model.
    const unsigned char JMP64r  = 0xFF; // 64-bit jmp through register opcode.
    const unsigned char MOV64ri = 0xB8; // X86::MOV64ri opcode.

    const unsigned char N86R10 = TRI->getEncodingValue(X86::R10) & 0x7;
    const unsigned char N86R11 = TRI->getEncodingValue(X86::R11) & 0x7;

    const unsigned char REX_WB = 0x40 | 0x08 | 0x01; // REX prefix

    // Load the pointer to the nested function into R11.
    unsigned OpCode = ((MOV64ri | N86R11) << 8) | REX_WB; // movabsq r11
    SDValue Addr = Trmp;
    OutChains[0] = DAG.getStore(Root, dl, DAG.getConstant(OpCode, MVT::i16),
                                Addr, MachinePointerInfo(TrmpAddr),
                                false, false, 0);

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(2, MVT::i64));
    OutChains[1] = DAG.getStore(Root, dl, FPtr, Addr,
                                MachinePointerInfo(TrmpAddr, 2),
                                false, false, 2);

    // Load the 'nest' parameter value into R10.
    // R10 is specified in X86CallingConv.td
    OpCode = ((MOV64ri | N86R10) << 8) | REX_WB; // movabsq r10
    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(10, MVT::i64));
    OutChains[2] = DAG.getStore(Root, dl, DAG.getConstant(OpCode, MVT::i16),
                                Addr, MachinePointerInfo(TrmpAddr, 10),
                                false, false, 0);

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(12, MVT::i64));
    OutChains[3] = DAG.getStore(Root, dl, Nest, Addr,
                                MachinePointerInfo(TrmpAddr, 12),
                                false, false, 2);

    // Jump to the nested function.
    OpCode = (JMP64r << 8) | REX_WB; // jmpq *...
    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(20, MVT::i64));
    OutChains[4] = DAG.getStore(Root, dl, DAG.getConstant(OpCode, MVT::i16),
                                Addr, MachinePointerInfo(TrmpAddr, 20),
                                false, false, 0);

    unsigned char ModRM = N86R11 | (4 << 3) | (3 << 6); // ...r11
    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(22, MVT::i64));
    OutChains[5] = DAG.getStore(Root, dl, DAG.getConstant(ModRM, MVT::i8), Addr,
                                MachinePointerInfo(TrmpAddr, 22),
                                false, false, 0);

    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains, 6);
  } else {
    const Function *Func =
      cast<Function>(cast<SrcValueSDNode>(Op.getOperand(5))->getValue());
    CallingConv::ID CC = Func->getCallingConv();
    unsigned NestReg;

    switch (CC) {
    default:
      llvm_unreachable("Unsupported calling convention");
    case CallingConv::C:
    case CallingConv::X86_StdCall: {
      // Pass 'nest' parameter in ECX.
      // Must be kept in sync with X86CallingConv.td
      NestReg = X86::ECX;

      // Check that ECX wasn't needed by an 'inreg' parameter.
      FunctionType *FTy = Func->getFunctionType();
      const AttrListPtr &Attrs = Func->getAttributes();

      if (!Attrs.isEmpty() && !Func->isVarArg()) {
        unsigned InRegCount = 0;
        unsigned Idx = 1;

        for (FunctionType::param_iterator I = FTy->param_begin(),
             E = FTy->param_end(); I != E; ++I, ++Idx)
          if (Attrs.getParamAttributes(Idx).hasAttribute(Attributes::InReg))
            // FIXME: should only count parameters that are lowered to integers.
            InRegCount += (TD->getTypeSizeInBits(*I) + 31) / 32;

        if (InRegCount > 2) {
          report_fatal_error("Nest register in use - reduce number of inreg"
                             " parameters!");
        }
      }
      break;
    }
    case CallingConv::X86_FastCall:
    case CallingConv::X86_ThisCall:
    case CallingConv::Fast:
      // Pass 'nest' parameter in EAX.
      // Must be kept in sync with X86CallingConv.td
      NestReg = X86::EAX;
      break;
    }

    SDValue OutChains[4];
    SDValue Addr, Disp;

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                       DAG.getConstant(10, MVT::i32));
    Disp = DAG.getNode(ISD::SUB, dl, MVT::i32, FPtr, Addr);

    // This is storing the opcode for MOV32ri.
    const unsigned char MOV32ri = 0xB8; // X86::MOV32ri's opcode byte.
    const unsigned char N86Reg = TRI->getEncodingValue(NestReg) & 0x7;
    OutChains[0] = DAG.getStore(Root, dl,
                                DAG.getConstant(MOV32ri|N86Reg, MVT::i8),
                                Trmp, MachinePointerInfo(TrmpAddr),
                                false, false, 0);

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                       DAG.getConstant(1, MVT::i32));
    OutChains[1] = DAG.getStore(Root, dl, Nest, Addr,
                                MachinePointerInfo(TrmpAddr, 1),
                                false, false, 1);

    const unsigned char JMP = 0xE9; // jmp <32bit dst> opcode.
    Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                       DAG.getConstant(5, MVT::i32));
    OutChains[2] = DAG.getStore(Root, dl, DAG.getConstant(JMP, MVT::i8), Addr,
                                MachinePointerInfo(TrmpAddr, 5),
                                false, false, 1);

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                       DAG.getConstant(6, MVT::i32));
    OutChains[3] = DAG.getStore(Root, dl, Disp, Addr,
                                MachinePointerInfo(TrmpAddr, 6),
                                false, false, 1);

    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains, 4);
  }
}

SDValue X86TargetLowering::LowerFLT_ROUNDS_(SDValue Op,
                                            SelectionDAG &DAG) const {
  /*
   The rounding mode is in bits 11:10 of FPSR, and has the following
   settings:
     00 Round to nearest
     01 Round to -inf
     10 Round to +inf
     11 Round to 0

  FLT_ROUNDS, on the other hand, expects the following:
    -1 Undefined
     0 Round to 0
     1 Round to nearest
     2 Round to +inf
     3 Round to -inf

  To perform the conversion, we do:
    (((((FPSR & 0x800) >> 11) | ((FPSR & 0x400) >> 9)) + 1) & 3)
  */

  MachineFunction &MF = DAG.getMachineFunction();
  const TargetMachine &TM = MF.getTarget();
  const TargetFrameLowering &TFI = *TM.getFrameLowering();
  unsigned StackAlignment = TFI.getStackAlignment();
  EVT VT = Op.getValueType();
  DebugLoc DL = Op.getDebugLoc();

  // Save FP Control Word to stack slot
  int SSFI = MF.getFrameInfo()->CreateStackObject(2, StackAlignment, false);
  SDValue StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());


  MachineMemOperand *MMO =
   MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(SSFI),
                           MachineMemOperand::MOStore, 2, 2);

  SDValue Ops[] = { DAG.getEntryNode(), StackSlot };
  SDValue Chain = DAG.getMemIntrinsicNode(X86ISD::FNSTCW16m, DL,
                                          DAG.getVTList(MVT::Other),
                                          Ops, 2, MVT::i16, MMO);

  // Load FP Control Word from stack slot
  SDValue CWD = DAG.getLoad(MVT::i16, DL, Chain, StackSlot,
                            MachinePointerInfo(), false, false, false, 0);

  // Transform as necessary
  SDValue CWD1 =
    DAG.getNode(ISD::SRL, DL, MVT::i16,
                DAG.getNode(ISD::AND, DL, MVT::i16,
                            CWD, DAG.getConstant(0x800, MVT::i16)),
                DAG.getConstant(11, MVT::i8));
  SDValue CWD2 =
    DAG.getNode(ISD::SRL, DL, MVT::i16,
                DAG.getNode(ISD::AND, DL, MVT::i16,
                            CWD, DAG.getConstant(0x400, MVT::i16)),
                DAG.getConstant(9, MVT::i8));

  SDValue RetVal =
    DAG.getNode(ISD::AND, DL, MVT::i16,
                DAG.getNode(ISD::ADD, DL, MVT::i16,
                            DAG.getNode(ISD::OR, DL, MVT::i16, CWD1, CWD2),
                            DAG.getConstant(1, MVT::i16)),
                DAG.getConstant(3, MVT::i16));


  return DAG.getNode((VT.getSizeInBits() < 16 ?
                      ISD::TRUNCATE : ISD::ZERO_EXTEND), DL, VT, RetVal);
}

static SDValue LowerCTLZ(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  EVT OpVT = VT;
  unsigned NumBits = VT.getSizeInBits();
  DebugLoc dl = Op.getDebugLoc();

  Op = Op.getOperand(0);
  if (VT == MVT::i8) {
    // Zero extend to i32 since there is not an i8 bsr.
    OpVT = MVT::i32;
    Op = DAG.getNode(ISD::ZERO_EXTEND, dl, OpVT, Op);
  }

  // Issue a bsr (scan bits in reverse) which also sets EFLAGS.
  SDVTList VTs = DAG.getVTList(OpVT, MVT::i32);
  Op = DAG.getNode(X86ISD::BSR, dl, VTs, Op);

  // If src is zero (i.e. bsr sets ZF), returns NumBits.
  SDValue Ops[] = {
    Op,
    DAG.getConstant(NumBits+NumBits-1, OpVT),
    DAG.getConstant(X86::COND_E, MVT::i8),
    Op.getValue(1)
  };
  Op = DAG.getNode(X86ISD::CMOV, dl, OpVT, Ops, array_lengthof(Ops));

  // Finally xor with NumBits-1.
  Op = DAG.getNode(ISD::XOR, dl, OpVT, Op, DAG.getConstant(NumBits-1, OpVT));

  if (VT == MVT::i8)
    Op = DAG.getNode(ISD::TRUNCATE, dl, MVT::i8, Op);
  return Op;
}

static SDValue LowerCTLZ_ZERO_UNDEF(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  EVT OpVT = VT;
  unsigned NumBits = VT.getSizeInBits();
  DebugLoc dl = Op.getDebugLoc();

  Op = Op.getOperand(0);
  if (VT == MVT::i8) {
    // Zero extend to i32 since there is not an i8 bsr.
    OpVT = MVT::i32;
    Op = DAG.getNode(ISD::ZERO_EXTEND, dl, OpVT, Op);
  }

  // Issue a bsr (scan bits in reverse).
  SDVTList VTs = DAG.getVTList(OpVT, MVT::i32);
  Op = DAG.getNode(X86ISD::BSR, dl, VTs, Op);

  // And xor with NumBits-1.
  Op = DAG.getNode(ISD::XOR, dl, OpVT, Op, DAG.getConstant(NumBits-1, OpVT));

  if (VT == MVT::i8)
    Op = DAG.getNode(ISD::TRUNCATE, dl, MVT::i8, Op);
  return Op;
}

static SDValue LowerCTTZ(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  unsigned NumBits = VT.getSizeInBits();
  DebugLoc dl = Op.getDebugLoc();
  Op = Op.getOperand(0);

  // Issue a bsf (scan bits forward) which also sets EFLAGS.
  SDVTList VTs = DAG.getVTList(VT, MVT::i32);
  Op = DAG.getNode(X86ISD::BSF, dl, VTs, Op);

  // If src is zero (i.e. bsf sets ZF), returns NumBits.
  SDValue Ops[] = {
    Op,
    DAG.getConstant(NumBits, VT),
    DAG.getConstant(X86::COND_E, MVT::i8),
    Op.getValue(1)
  };
  return DAG.getNode(X86ISD::CMOV, dl, VT, Ops, array_lengthof(Ops));
}

// Lower256IntArith - Break a 256-bit integer operation into two new 128-bit
// ones, and then concatenate the result back.
static SDValue Lower256IntArith(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();

  assert(VT.is256BitVector() && VT.isInteger() &&
         "Unsupported value type for operation");

  unsigned NumElems = VT.getVectorNumElements();
  DebugLoc dl = Op.getDebugLoc();

  // Extract the LHS vectors
  SDValue LHS = Op.getOperand(0);
  SDValue LHS1 = Extract128BitVector(LHS, 0, DAG, dl);
  SDValue LHS2 = Extract128BitVector(LHS, NumElems/2, DAG, dl);

  // Extract the RHS vectors
  SDValue RHS = Op.getOperand(1);
  SDValue RHS1 = Extract128BitVector(RHS, 0, DAG, dl);
  SDValue RHS2 = Extract128BitVector(RHS, NumElems/2, DAG, dl);

  MVT EltVT = VT.getVectorElementType().getSimpleVT();
  EVT NewVT = MVT::getVectorVT(EltVT, NumElems/2);

  return DAG.getNode(ISD::CONCAT_VECTORS, dl, VT,
                     DAG.getNode(Op.getOpcode(), dl, NewVT, LHS1, RHS1),
                     DAG.getNode(Op.getOpcode(), dl, NewVT, LHS2, RHS2));
}

static SDValue LowerADD(SDValue Op, SelectionDAG &DAG) {
  assert(Op.getValueType().is256BitVector() &&
         Op.getValueType().isInteger() &&
         "Only handle AVX 256-bit vector integer operation");
  return Lower256IntArith(Op, DAG);
}

static SDValue LowerSUB(SDValue Op, SelectionDAG &DAG) {
  assert(Op.getValueType().is256BitVector() &&
         Op.getValueType().isInteger() &&
         "Only handle AVX 256-bit vector integer operation");
  return Lower256IntArith(Op, DAG);
}

static SDValue LowerMUL(SDValue Op, const X86Subtarget *Subtarget,
                        SelectionDAG &DAG) {
  EVT VT = Op.getValueType();

  // Decompose 256-bit ops into smaller 128-bit ops.
  if (VT.is256BitVector() && !Subtarget->hasAVX2())
    return Lower256IntArith(Op, DAG);

  assert((VT == MVT::v2i64 || VT == MVT::v4i64) &&
         "Only know how to lower V2I64/V4I64 multiply");

  DebugLoc dl = Op.getDebugLoc();

  //  Ahi = psrlqi(a, 32);
  //  Bhi = psrlqi(b, 32);
  //
  //  AloBlo = pmuludq(a, b);
  //  AloBhi = pmuludq(a, Bhi);
  //  AhiBlo = pmuludq(Ahi, b);

  //  AloBhi = psllqi(AloBhi, 32);
  //  AhiBlo = psllqi(AhiBlo, 32);
  //  return AloBlo + AloBhi + AhiBlo;

  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);

  SDValue ShAmt = DAG.getConstant(32, MVT::i32);

  SDValue Ahi = DAG.getNode(X86ISD::VSRLI, dl, VT, A, ShAmt);
  SDValue Bhi = DAG.getNode(X86ISD::VSRLI, dl, VT, B, ShAmt);

  // Bit cast to 32-bit vectors for MULUDQ
  EVT MulVT = (VT == MVT::v2i64) ? MVT::v4i32 : MVT::v8i32;
  A = DAG.getNode(ISD::BITCAST, dl, MulVT, A);
  B = DAG.getNode(ISD::BITCAST, dl, MulVT, B);
  Ahi = DAG.getNode(ISD::BITCAST, dl, MulVT, Ahi);
  Bhi = DAG.getNode(ISD::BITCAST, dl, MulVT, Bhi);

  SDValue AloBlo = DAG.getNode(X86ISD::PMULUDQ, dl, VT, A, B);
  SDValue AloBhi = DAG.getNode(X86ISD::PMULUDQ, dl, VT, A, Bhi);
  SDValue AhiBlo = DAG.getNode(X86ISD::PMULUDQ, dl, VT, Ahi, B);

  AloBhi = DAG.getNode(X86ISD::VSHLI, dl, VT, AloBhi, ShAmt);
  AhiBlo = DAG.getNode(X86ISD::VSHLI, dl, VT, AhiBlo, ShAmt);

  SDValue Res = DAG.getNode(ISD::ADD, dl, VT, AloBlo, AloBhi);
  return DAG.getNode(ISD::ADD, dl, VT, Res, AhiBlo);
}

SDValue X86TargetLowering::LowerShift(SDValue Op, SelectionDAG &DAG) const {

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  SDValue R = Op.getOperand(0);
  SDValue Amt = Op.getOperand(1);
  LLVMContext *Context = DAG.getContext();

  if (!Subtarget->hasSSE2())
    return SDValue();

  // Optimize shl/srl/sra with constant shift amount.
  if (isSplatVector(Amt.getNode())) {
    SDValue SclrAmt = Amt->getOperand(0);
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(SclrAmt)) {
      uint64_t ShiftAmt = C->getZExtValue();

      if (VT == MVT::v2i64 || VT == MVT::v4i32 || VT == MVT::v8i16 ||
          (Subtarget->hasAVX2() &&
           (VT == MVT::v4i64 || VT == MVT::v8i32 || VT == MVT::v16i16))) {
        if (Op.getOpcode() == ISD::SHL)
          return DAG.getNode(X86ISD::VSHLI, dl, VT, R,
                             DAG.getConstant(ShiftAmt, MVT::i32));
        if (Op.getOpcode() == ISD::SRL)
          return DAG.getNode(X86ISD::VSRLI, dl, VT, R,
                             DAG.getConstant(ShiftAmt, MVT::i32));
        if (Op.getOpcode() == ISD::SRA && VT != MVT::v2i64 && VT != MVT::v4i64)
          return DAG.getNode(X86ISD::VSRAI, dl, VT, R,
                             DAG.getConstant(ShiftAmt, MVT::i32));
      }

      if (VT == MVT::v16i8) {
        if (Op.getOpcode() == ISD::SHL) {
          // Make a large shift.
          SDValue SHL = DAG.getNode(X86ISD::VSHLI, dl, MVT::v8i16, R,
                                    DAG.getConstant(ShiftAmt, MVT::i32));
          SHL = DAG.getNode(ISD::BITCAST, dl, VT, SHL);
          // Zero out the rightmost bits.
          SmallVector<SDValue, 16> V(16,
                                     DAG.getConstant(uint8_t(-1U << ShiftAmt),
                                                     MVT::i8));
          return DAG.getNode(ISD::AND, dl, VT, SHL,
                             DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &V[0], 16));
        }
        if (Op.getOpcode() == ISD::SRL) {
          // Make a large shift.
          SDValue SRL = DAG.getNode(X86ISD::VSRLI, dl, MVT::v8i16, R,
                                    DAG.getConstant(ShiftAmt, MVT::i32));
          SRL = DAG.getNode(ISD::BITCAST, dl, VT, SRL);
          // Zero out the leftmost bits.
          SmallVector<SDValue, 16> V(16,
                                     DAG.getConstant(uint8_t(-1U) >> ShiftAmt,
                                                     MVT::i8));
          return DAG.getNode(ISD::AND, dl, VT, SRL,
                             DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &V[0], 16));
        }
        if (Op.getOpcode() == ISD::SRA) {
          if (ShiftAmt == 7) {
            // R s>> 7  ===  R s< 0
            SDValue Zeros = getZeroVector(VT, Subtarget, DAG, dl);
            return DAG.getNode(X86ISD::PCMPGT, dl, VT, Zeros, R);
          }

          // R s>> a === ((R u>> a) ^ m) - m
          SDValue Res = DAG.getNode(ISD::SRL, dl, VT, R, Amt);
          SmallVector<SDValue, 16> V(16, DAG.getConstant(128 >> ShiftAmt,
                                                         MVT::i8));
          SDValue Mask = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &V[0], 16);
          Res = DAG.getNode(ISD::XOR, dl, VT, Res, Mask);
          Res = DAG.getNode(ISD::SUB, dl, VT, Res, Mask);
          return Res;
        }
        llvm_unreachable("Unknown shift opcode.");
      }

      if (Subtarget->hasAVX2() && VT == MVT::v32i8) {
        if (Op.getOpcode() == ISD::SHL) {
          // Make a large shift.
          SDValue SHL = DAG.getNode(X86ISD::VSHLI, dl, MVT::v16i16, R,
                                    DAG.getConstant(ShiftAmt, MVT::i32));
          SHL = DAG.getNode(ISD::BITCAST, dl, VT, SHL);
          // Zero out the rightmost bits.
          SmallVector<SDValue, 32> V(32,
                                     DAG.getConstant(uint8_t(-1U << ShiftAmt),
                                                     MVT::i8));
          return DAG.getNode(ISD::AND, dl, VT, SHL,
                             DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &V[0], 32));
        }
        if (Op.getOpcode() == ISD::SRL) {
          // Make a large shift.
          SDValue SRL = DAG.getNode(X86ISD::VSRLI, dl, MVT::v16i16, R,
                                    DAG.getConstant(ShiftAmt, MVT::i32));
          SRL = DAG.getNode(ISD::BITCAST, dl, VT, SRL);
          // Zero out the leftmost bits.
          SmallVector<SDValue, 32> V(32,
                                     DAG.getConstant(uint8_t(-1U) >> ShiftAmt,
                                                     MVT::i8));
          return DAG.getNode(ISD::AND, dl, VT, SRL,
                             DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &V[0], 32));
        }
        if (Op.getOpcode() == ISD::SRA) {
          if (ShiftAmt == 7) {
            // R s>> 7  ===  R s< 0
            SDValue Zeros = getZeroVector(VT, Subtarget, DAG, dl);
            return DAG.getNode(X86ISD::PCMPGT, dl, VT, Zeros, R);
          }

          // R s>> a === ((R u>> a) ^ m) - m
          SDValue Res = DAG.getNode(ISD::SRL, dl, VT, R, Amt);
          SmallVector<SDValue, 32> V(32, DAG.getConstant(128 >> ShiftAmt,
                                                         MVT::i8));
          SDValue Mask = DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &V[0], 32);
          Res = DAG.getNode(ISD::XOR, dl, VT, Res, Mask);
          Res = DAG.getNode(ISD::SUB, dl, VT, Res, Mask);
          return Res;
        }
        llvm_unreachable("Unknown shift opcode.");
      }
    }
  }

  // Lower SHL with variable shift amount.
  if (VT == MVT::v4i32 && Op->getOpcode() == ISD::SHL) {
    Op = DAG.getNode(X86ISD::VSHLI, dl, VT, Op.getOperand(1),
                     DAG.getConstant(23, MVT::i32));

    const uint32_t CV[] = { 0x3f800000U, 0x3f800000U, 0x3f800000U, 0x3f800000U};
    Constant *C = ConstantDataVector::get(*Context, CV);
    SDValue CPIdx = DAG.getConstantPool(C, getPointerTy(), 16);
    SDValue Addend = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                                 MachinePointerInfo::getConstantPool(),
                                 false, false, false, 16);

    Op = DAG.getNode(ISD::ADD, dl, VT, Op, Addend);
    Op = DAG.getNode(ISD::BITCAST, dl, MVT::v4f32, Op);
    Op = DAG.getNode(ISD::FP_TO_SINT, dl, VT, Op);
    return DAG.getNode(ISD::MUL, dl, VT, Op, R);
  }
  if (VT == MVT::v16i8 && Op->getOpcode() == ISD::SHL) {
    assert(Subtarget->hasSSE2() && "Need SSE2 for pslli/pcmpeq.");

    // a = a << 5;
    Op = DAG.getNode(X86ISD::VSHLI, dl, MVT::v8i16, Op.getOperand(1),
                     DAG.getConstant(5, MVT::i32));
    Op = DAG.getNode(ISD::BITCAST, dl, VT, Op);

    // Turn 'a' into a mask suitable for VSELECT
    SDValue VSelM = DAG.getConstant(0x80, VT);
    SDValue OpVSel = DAG.getNode(ISD::AND, dl, VT, VSelM, Op);
    OpVSel = DAG.getNode(X86ISD::PCMPEQ, dl, VT, OpVSel, VSelM);

    SDValue CM1 = DAG.getConstant(0x0f, VT);
    SDValue CM2 = DAG.getConstant(0x3f, VT);

    // r = VSELECT(r, psllw(r & (char16)15, 4), a);
    SDValue M = DAG.getNode(ISD::AND, dl, VT, R, CM1);
    M = getTargetVShiftNode(X86ISD::VSHLI, dl, MVT::v8i16, M,
                            DAG.getConstant(4, MVT::i32), DAG);
    M = DAG.getNode(ISD::BITCAST, dl, VT, M);
    R = DAG.getNode(ISD::VSELECT, dl, VT, OpVSel, M, R);

    // a += a
    Op = DAG.getNode(ISD::ADD, dl, VT, Op, Op);
    OpVSel = DAG.getNode(ISD::AND, dl, VT, VSelM, Op);
    OpVSel = DAG.getNode(X86ISD::PCMPEQ, dl, VT, OpVSel, VSelM);

    // r = VSELECT(r, psllw(r & (char16)63, 2), a);
    M = DAG.getNode(ISD::AND, dl, VT, R, CM2);
    M = getTargetVShiftNode(X86ISD::VSHLI, dl, MVT::v8i16, M,
                            DAG.getConstant(2, MVT::i32), DAG);
    M = DAG.getNode(ISD::BITCAST, dl, VT, M);
    R = DAG.getNode(ISD::VSELECT, dl, VT, OpVSel, M, R);

    // a += a
    Op = DAG.getNode(ISD::ADD, dl, VT, Op, Op);
    OpVSel = DAG.getNode(ISD::AND, dl, VT, VSelM, Op);
    OpVSel = DAG.getNode(X86ISD::PCMPEQ, dl, VT, OpVSel, VSelM);

    // return VSELECT(r, r+r, a);
    R = DAG.getNode(ISD::VSELECT, dl, VT, OpVSel,
                    DAG.getNode(ISD::ADD, dl, VT, R, R), R);
    return R;
  }

  // Decompose 256-bit shifts into smaller 128-bit shifts.
  if (VT.is256BitVector()) {
    unsigned NumElems = VT.getVectorNumElements();
    MVT EltVT = VT.getVectorElementType().getSimpleVT();
    EVT NewVT = MVT::getVectorVT(EltVT, NumElems/2);

    // Extract the two vectors
    SDValue V1 = Extract128BitVector(R, 0, DAG, dl);
    SDValue V2 = Extract128BitVector(R, NumElems/2, DAG, dl);

    // Recreate the shift amount vectors
    SDValue Amt1, Amt2;
    if (Amt.getOpcode() == ISD::BUILD_VECTOR) {
      // Constant shift amount
      SmallVector<SDValue, 4> Amt1Csts;
      SmallVector<SDValue, 4> Amt2Csts;
      for (unsigned i = 0; i != NumElems/2; ++i)
        Amt1Csts.push_back(Amt->getOperand(i));
      for (unsigned i = NumElems/2; i != NumElems; ++i)
        Amt2Csts.push_back(Amt->getOperand(i));

      Amt1 = DAG.getNode(ISD::BUILD_VECTOR, dl, NewVT,
                                 &Amt1Csts[0], NumElems/2);
      Amt2 = DAG.getNode(ISD::BUILD_VECTOR, dl, NewVT,
                                 &Amt2Csts[0], NumElems/2);
    } else {
      // Variable shift amount
      Amt1 = Extract128BitVector(Amt, 0, DAG, dl);
      Amt2 = Extract128BitVector(Amt, NumElems/2, DAG, dl);
    }

    // Issue new vector shifts for the smaller types
    V1 = DAG.getNode(Op.getOpcode(), dl, NewVT, V1, Amt1);
    V2 = DAG.getNode(Op.getOpcode(), dl, NewVT, V2, Amt2);

    // Concatenate the result back
    return DAG.getNode(ISD::CONCAT_VECTORS, dl, VT, V1, V2);
  }

  return SDValue();
}

static SDValue LowerXALUO(SDValue Op, SelectionDAG &DAG) {
  // Lower the "add/sub/mul with overflow" instruction into a regular ins plus
  // a "setcc" instruction that checks the overflow flag. The "brcond" lowering
  // looks for this combo and may remove the "setcc" instruction if the "setcc"
  // has only one use.
  SDNode *N = Op.getNode();
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  unsigned BaseOp = 0;
  unsigned Cond = 0;
  DebugLoc DL = Op.getDebugLoc();
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Unknown ovf instruction!");
  case ISD::SADDO:
    // A subtract of one will be selected as a INC. Note that INC doesn't
    // set CF, so we can't do this for UADDO.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(RHS))
      if (C->isOne()) {
        BaseOp = X86ISD::INC;
        Cond = X86::COND_O;
        break;
      }
    BaseOp = X86ISD::ADD;
    Cond = X86::COND_O;
    break;
  case ISD::UADDO:
    BaseOp = X86ISD::ADD;
    Cond = X86::COND_B;
    break;
  case ISD::SSUBO:
    // A subtract of one will be selected as a DEC. Note that DEC doesn't
    // set CF, so we can't do this for USUBO.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(RHS))
      if (C->isOne()) {
        BaseOp = X86ISD::DEC;
        Cond = X86::COND_O;
        break;
      }
    BaseOp = X86ISD::SUB;
    Cond = X86::COND_O;
    break;
  case ISD::USUBO:
    BaseOp = X86ISD::SUB;
    Cond = X86::COND_B;
    break;
  case ISD::SMULO:
    BaseOp = X86ISD::SMUL;
    Cond = X86::COND_O;
    break;
  case ISD::UMULO: { // i64, i8 = umulo lhs, rhs --> i64, i64, i32 umul lhs,rhs
    SDVTList VTs = DAG.getVTList(N->getValueType(0), N->getValueType(0),
                                 MVT::i32);
    SDValue Sum = DAG.getNode(X86ISD::UMUL, DL, VTs, LHS, RHS);

    SDValue SetCC =
      DAG.getNode(X86ISD::SETCC, DL, MVT::i8,
                  DAG.getConstant(X86::COND_O, MVT::i32),
                  SDValue(Sum.getNode(), 2));

    return DAG.getNode(ISD::MERGE_VALUES, DL, N->getVTList(), Sum, SetCC);
  }
  }

  // Also sets EFLAGS.
  SDVTList VTs = DAG.getVTList(N->getValueType(0), MVT::i32);
  SDValue Sum = DAG.getNode(BaseOp, DL, VTs, LHS, RHS);

  SDValue SetCC =
    DAG.getNode(X86ISD::SETCC, DL, N->getValueType(1),
                DAG.getConstant(Cond, MVT::i32),
                SDValue(Sum.getNode(), 1));

  return DAG.getNode(ISD::MERGE_VALUES, DL, N->getVTList(), Sum, SetCC);
}

SDValue X86TargetLowering::LowerSIGN_EXTEND_INREG(SDValue Op,
                                                  SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  EVT ExtraVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
  EVT VT = Op.getValueType();

  if (!Subtarget->hasSSE2() || !VT.isVector())
    return SDValue();

  unsigned BitsDiff = VT.getScalarType().getSizeInBits() -
                      ExtraVT.getScalarType().getSizeInBits();
  SDValue ShAmt = DAG.getConstant(BitsDiff, MVT::i32);

  switch (VT.getSimpleVT().SimpleTy) {
    default: return SDValue();
    case MVT::v8i32:
    case MVT::v16i16:
      if (!Subtarget->hasAVX())
        return SDValue();
      if (!Subtarget->hasAVX2()) {
        // needs to be split
        unsigned NumElems = VT.getVectorNumElements();

        // Extract the LHS vectors
        SDValue LHS = Op.getOperand(0);
        SDValue LHS1 = Extract128BitVector(LHS, 0, DAG, dl);
        SDValue LHS2 = Extract128BitVector(LHS, NumElems/2, DAG, dl);

        MVT EltVT = VT.getVectorElementType().getSimpleVT();
        EVT NewVT = MVT::getVectorVT(EltVT, NumElems/2);

        EVT ExtraEltVT = ExtraVT.getVectorElementType();
        unsigned ExtraNumElems = ExtraVT.getVectorNumElements();
        ExtraVT = EVT::getVectorVT(*DAG.getContext(), ExtraEltVT,
                                   ExtraNumElems/2);
        SDValue Extra = DAG.getValueType(ExtraVT);

        LHS1 = DAG.getNode(Op.getOpcode(), dl, NewVT, LHS1, Extra);
        LHS2 = DAG.getNode(Op.getOpcode(), dl, NewVT, LHS2, Extra);

        return DAG.getNode(ISD::CONCAT_VECTORS, dl, VT, LHS1, LHS2);
      }
      // fall through
    case MVT::v4i32:
    case MVT::v8i16: {
      SDValue Tmp1 = getTargetVShiftNode(X86ISD::VSHLI, dl, VT,
                                         Op.getOperand(0), ShAmt, DAG);
      return getTargetVShiftNode(X86ISD::VSRAI, dl, VT, Tmp1, ShAmt, DAG);
    }
  }
}


static SDValue LowerMEMBARRIER(SDValue Op, const X86Subtarget *Subtarget,
                              SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();

  // Go ahead and emit the fence on x86-64 even if we asked for no-sse2.
  // There isn't any reason to disable it if the target processor supports it.
  if (!Subtarget->hasSSE2() && !Subtarget->is64Bit()) {
    SDValue Chain = Op.getOperand(0);
    SDValue Zero = DAG.getConstant(0, MVT::i32);
    SDValue Ops[] = {
      DAG.getRegister(X86::ESP, MVT::i32), // Base
      DAG.getTargetConstant(1, MVT::i8),   // Scale
      DAG.getRegister(0, MVT::i32),        // Index
      DAG.getTargetConstant(0, MVT::i32),  // Disp
      DAG.getRegister(0, MVT::i32),        // Segment.
      Zero,
      Chain
    };
    SDNode *Res =
      DAG.getMachineNode(X86::OR32mrLocked, dl, MVT::Other, Ops,
                          array_lengthof(Ops));
    return SDValue(Res, 0);
  }

  unsigned isDev = cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue();
  if (!isDev)
    return DAG.getNode(X86ISD::MEMBARRIER, dl, MVT::Other, Op.getOperand(0));

  unsigned Op1 = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  unsigned Op2 = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue();
  unsigned Op3 = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
  unsigned Op4 = cast<ConstantSDNode>(Op.getOperand(4))->getZExtValue();

  // def : Pat<(membarrier (i8 0), (i8 0), (i8 0), (i8 1), (i8 1)), (SFENCE)>;
  if (!Op1 && !Op2 && !Op3 && Op4)
    return DAG.getNode(X86ISD::SFENCE, dl, MVT::Other, Op.getOperand(0));

  // def : Pat<(membarrier (i8 1), (i8 0), (i8 0), (i8 0), (i8 1)), (LFENCE)>;
  if (Op1 && !Op2 && !Op3 && !Op4)
    return DAG.getNode(X86ISD::LFENCE, dl, MVT::Other, Op.getOperand(0));

  // def : Pat<(membarrier (i8 imm), (i8 imm), (i8 imm), (i8 imm), (i8 1)),
  //           (MFENCE)>;
  return DAG.getNode(X86ISD::MFENCE, dl, MVT::Other, Op.getOperand(0));
}

static SDValue LowerATOMIC_FENCE(SDValue Op, const X86Subtarget *Subtarget,
                                 SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  AtomicOrdering FenceOrdering = static_cast<AtomicOrdering>(
    cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue());
  SynchronizationScope FenceScope = static_cast<SynchronizationScope>(
    cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue());

  // The only fence that needs an instruction is a sequentially-consistent
  // cross-thread fence.
  if (FenceOrdering == SequentiallyConsistent && FenceScope == CrossThread) {
    // Use mfence if we have SSE2 or we're on x86-64 (even if we asked for
    // no-sse2). There isn't any reason to disable it if the target processor
    // supports it.
    if (Subtarget->hasSSE2() || Subtarget->is64Bit())
      return DAG.getNode(X86ISD::MFENCE, dl, MVT::Other, Op.getOperand(0));

    SDValue Chain = Op.getOperand(0);
    SDValue Zero = DAG.getConstant(0, MVT::i32);
    SDValue Ops[] = {
      DAG.getRegister(X86::ESP, MVT::i32), // Base
      DAG.getTargetConstant(1, MVT::i8),   // Scale
      DAG.getRegister(0, MVT::i32),        // Index
      DAG.getTargetConstant(0, MVT::i32),  // Disp
      DAG.getRegister(0, MVT::i32),        // Segment.
      Zero,
      Chain
    };
    SDNode *Res =
      DAG.getMachineNode(X86::OR32mrLocked, dl, MVT::Other, Ops,
                         array_lengthof(Ops));
    return SDValue(Res, 0);
  }

  // MEMBARRIER is a compiler barrier; it codegens to a no-op.
  return DAG.getNode(X86ISD::MEMBARRIER, dl, MVT::Other, Op.getOperand(0));
}


static SDValue LowerCMP_SWAP(SDValue Op, const X86Subtarget *Subtarget,
                             SelectionDAG &DAG) {
  EVT T = Op.getValueType();
  DebugLoc DL = Op.getDebugLoc();
  unsigned Reg = 0;
  unsigned size = 0;
  switch(T.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("Invalid value type!");
  case MVT::i8:  Reg = X86::AL;  size = 1; break;
  case MVT::i16: Reg = X86::AX;  size = 2; break;
  case MVT::i32: Reg = X86::EAX; size = 4; break;
  case MVT::i64:
    assert(Subtarget->is64Bit() && "Node not type legal!");
    Reg = X86::RAX; size = 8;
    break;
  }
  SDValue cpIn = DAG.getCopyToReg(Op.getOperand(0), DL, Reg,
                                    Op.getOperand(2), SDValue());
  SDValue Ops[] = { cpIn.getValue(0),
                    Op.getOperand(1),
                    Op.getOperand(3),
                    DAG.getTargetConstant(size, MVT::i8),
                    cpIn.getValue(1) };
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Glue);
  MachineMemOperand *MMO = cast<AtomicSDNode>(Op)->getMemOperand();
  SDValue Result = DAG.getMemIntrinsicNode(X86ISD::LCMPXCHG_DAG, DL, Tys,
                                           Ops, 5, T, MMO);
  SDValue cpOut =
    DAG.getCopyFromReg(Result.getValue(0), DL, Reg, T, Result.getValue(1));
  return cpOut;
}

static SDValue LowerREADCYCLECOUNTER(SDValue Op, const X86Subtarget *Subtarget,
                                     SelectionDAG &DAG) {
  assert(Subtarget->is64Bit() && "Result not type legalized?");
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue TheChain = Op.getOperand(0);
  DebugLoc dl = Op.getDebugLoc();
  SDValue rd = DAG.getNode(X86ISD::RDTSC_DAG, dl, Tys, &TheChain, 1);
  SDValue rax = DAG.getCopyFromReg(rd, dl, X86::RAX, MVT::i64, rd.getValue(1));
  SDValue rdx = DAG.getCopyFromReg(rax.getValue(1), dl, X86::RDX, MVT::i64,
                                   rax.getValue(2));
  SDValue Tmp = DAG.getNode(ISD::SHL, dl, MVT::i64, rdx,
                            DAG.getConstant(32, MVT::i8));
  SDValue Ops[] = {
    DAG.getNode(ISD::OR, dl, MVT::i64, rax, Tmp),
    rdx.getValue(1)
  };
  return DAG.getMergeValues(Ops, 2, dl);
}

SDValue X86TargetLowering::LowerBITCAST(SDValue Op, SelectionDAG &DAG) const {
  EVT SrcVT = Op.getOperand(0).getValueType();
  EVT DstVT = Op.getValueType();
  assert(Subtarget->is64Bit() && !Subtarget->hasSSE2() &&
         Subtarget->hasMMX() && "Unexpected custom BITCAST");
  assert((DstVT == MVT::i64 ||
          (DstVT.isVector() && DstVT.getSizeInBits()==64)) &&
         "Unexpected custom BITCAST");
  // i64 <=> MMX conversions are Legal.
  if (SrcVT==MVT::i64 && DstVT.isVector())
    return Op;
  if (DstVT==MVT::i64 && SrcVT.isVector())
    return Op;
  // MMX <=> MMX conversions are Legal.
  if (SrcVT.isVector() && DstVT.isVector())
    return Op;
  // All other conversions need to be expanded.
  return SDValue();
}

static SDValue LowerLOAD_SUB(SDValue Op, SelectionDAG &DAG) {
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  EVT T = Node->getValueType(0);
  SDValue negOp = DAG.getNode(ISD::SUB, dl, T,
                              DAG.getConstant(0, T), Node->getOperand(2));
  return DAG.getAtomic(ISD::ATOMIC_LOAD_ADD, dl,
                       cast<AtomicSDNode>(Node)->getMemoryVT(),
                       Node->getOperand(0),
                       Node->getOperand(1), negOp,
                       cast<AtomicSDNode>(Node)->getSrcValue(),
                       cast<AtomicSDNode>(Node)->getAlignment(),
                       cast<AtomicSDNode>(Node)->getOrdering(),
                       cast<AtomicSDNode>(Node)->getSynchScope());
}

static SDValue LowerATOMIC_STORE(SDValue Op, SelectionDAG &DAG) {
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  EVT VT = cast<AtomicSDNode>(Node)->getMemoryVT();

  // Convert seq_cst store -> xchg
  // Convert wide store -> swap (-> cmpxchg8b/cmpxchg16b)
  // FIXME: On 32-bit, store -> fist or movq would be more efficient
  //        (The only way to get a 16-byte store is cmpxchg16b)
  // FIXME: 16-byte ATOMIC_SWAP isn't actually hooked up at the moment.
  if (cast<AtomicSDNode>(Node)->getOrdering() == SequentiallyConsistent ||
      !DAG.getTargetLoweringInfo().isTypeLegal(VT)) {
    SDValue Swap = DAG.getAtomic(ISD::ATOMIC_SWAP, dl,
                                 cast<AtomicSDNode>(Node)->getMemoryVT(),
                                 Node->getOperand(0),
                                 Node->getOperand(1), Node->getOperand(2),
                                 cast<AtomicSDNode>(Node)->getMemOperand(),
                                 cast<AtomicSDNode>(Node)->getOrdering(),
                                 cast<AtomicSDNode>(Node)->getSynchScope());
    return Swap.getValue(1);
  }
  // Other atomic stores have a simple pattern.
  return Op;
}

static SDValue LowerADDC_ADDE_SUBC_SUBE(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getNode()->getValueType(0);

  // Let legalize expand this if it isn't a legal type yet.
  if (!DAG.getTargetLoweringInfo().isTypeLegal(VT))
    return SDValue();

  SDVTList VTs = DAG.getVTList(VT, MVT::i32);

  unsigned Opc;
  bool ExtraOp = false;
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Invalid code");
  case ISD::ADDC: Opc = X86ISD::ADD; break;
  case ISD::ADDE: Opc = X86ISD::ADC; ExtraOp = true; break;
  case ISD::SUBC: Opc = X86ISD::SUB; break;
  case ISD::SUBE: Opc = X86ISD::SBB; ExtraOp = true; break;
  }

  if (!ExtraOp)
    return DAG.getNode(Opc, Op->getDebugLoc(), VTs, Op.getOperand(0),
                       Op.getOperand(1));
  return DAG.getNode(Opc, Op->getDebugLoc(), VTs, Op.getOperand(0),
                     Op.getOperand(1), Op.getOperand(2));
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDValue X86TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Should not custom lower this!");
  case ISD::SIGN_EXTEND_INREG:  return LowerSIGN_EXTEND_INREG(Op,DAG);
  case ISD::MEMBARRIER:         return LowerMEMBARRIER(Op, Subtarget, DAG);
  case ISD::ATOMIC_FENCE:       return LowerATOMIC_FENCE(Op, Subtarget, DAG);
  case ISD::ATOMIC_CMP_SWAP:    return LowerCMP_SWAP(Op, Subtarget, DAG);
  case ISD::ATOMIC_LOAD_SUB:    return LowerLOAD_SUB(Op,DAG);
  case ISD::ATOMIC_STORE:       return LowerATOMIC_STORE(Op,DAG);
  case ISD::BUILD_VECTOR:       return LowerBUILD_VECTOR(Op, DAG);
  case ISD::CONCAT_VECTORS:     return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::VECTOR_SHUFFLE:     return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT: return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:  return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_SUBVECTOR:  return LowerEXTRACT_SUBVECTOR(Op,Subtarget,DAG);
  case ISD::INSERT_SUBVECTOR:   return LowerINSERT_SUBVECTOR(Op, Subtarget,DAG);
  case ISD::SCALAR_TO_VECTOR:   return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
  case ISD::ExternalSymbol:     return LowerExternalSymbol(Op, DAG);
  case ISD::BlockAddress:       return LowerBlockAddress(Op, DAG);
  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:          return LowerShiftParts(Op, DAG);
  case ISD::SINT_TO_FP:         return LowerSINT_TO_FP(Op, DAG);
  case ISD::UINT_TO_FP:         return LowerUINT_TO_FP(Op, DAG);
  case ISD::TRUNCATE:           return lowerTRUNCATE(Op, DAG);
  case ISD::ZERO_EXTEND:        return lowerZERO_EXTEND(Op, DAG);
  case ISD::FP_TO_SINT:         return LowerFP_TO_SINT(Op, DAG);
  case ISD::FP_TO_UINT:         return LowerFP_TO_UINT(Op, DAG);
  case ISD::FP_EXTEND:          return lowerFP_EXTEND(Op, DAG);
  case ISD::FABS:               return LowerFABS(Op, DAG);
  case ISD::FNEG:               return LowerFNEG(Op, DAG);
  case ISD::FCOPYSIGN:          return LowerFCOPYSIGN(Op, DAG);
  case ISD::FGETSIGN:           return LowerFGETSIGN(Op, DAG);
  case ISD::SETCC:              return LowerSETCC(Op, DAG);
  case ISD::SELECT:             return LowerSELECT(Op, DAG);
  case ISD::BRCOND:             return LowerBRCOND(Op, DAG);
  case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
  case ISD::VASTART:            return LowerVASTART(Op, DAG);
  case ISD::VAARG:              return LowerVAARG(Op, DAG);
  case ISD::VACOPY:             return LowerVACOPY(Op, Subtarget, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN:  return LowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::RETURNADDR:         return LowerRETURNADDR(Op, DAG);
  case ISD::FRAMEADDR:          return LowerFRAMEADDR(Op, DAG);
  case ISD::FRAME_TO_ARGS_OFFSET:
                                return LowerFRAME_TO_ARGS_OFFSET(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC: return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::EH_RETURN:          return LowerEH_RETURN(Op, DAG);
  case ISD::EH_SJLJ_SETJMP:     return lowerEH_SJLJ_SETJMP(Op, DAG);
  case ISD::EH_SJLJ_LONGJMP:    return lowerEH_SJLJ_LONGJMP(Op, DAG);
  case ISD::INIT_TRAMPOLINE:    return LowerINIT_TRAMPOLINE(Op, DAG);
  case ISD::ADJUST_TRAMPOLINE:  return LowerADJUST_TRAMPOLINE(Op, DAG);
  case ISD::FLT_ROUNDS_:        return LowerFLT_ROUNDS_(Op, DAG);
  case ISD::CTLZ:               return LowerCTLZ(Op, DAG);
  case ISD::CTLZ_ZERO_UNDEF:    return LowerCTLZ_ZERO_UNDEF(Op, DAG);
  case ISD::CTTZ:               return LowerCTTZ(Op, DAG);
  case ISD::MUL:                return LowerMUL(Op, Subtarget, DAG);
  case ISD::SRA:
  case ISD::SRL:
  case ISD::SHL:                return LowerShift(Op, DAG);
  case ISD::SADDO:
  case ISD::UADDO:
  case ISD::SSUBO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:              return LowerXALUO(Op, DAG);
  case ISD::READCYCLECOUNTER:   return LowerREADCYCLECOUNTER(Op, Subtarget,DAG);
  case ISD::BITCAST:            return LowerBITCAST(Op, DAG);
  case ISD::ADDC:
  case ISD::ADDE:
  case ISD::SUBC:
  case ISD::SUBE:               return LowerADDC_ADDE_SUBC_SUBE(Op, DAG);
  case ISD::ADD:                return LowerADD(Op, DAG);
  case ISD::SUB:                return LowerSUB(Op, DAG);
  }
}

static void ReplaceATOMIC_LOAD(SDNode *Node,
                                  SmallVectorImpl<SDValue> &Results,
                                  SelectionDAG &DAG) {
  DebugLoc dl = Node->getDebugLoc();
  EVT VT = cast<AtomicSDNode>(Node)->getMemoryVT();

  // Convert wide load -> cmpxchg8b/cmpxchg16b
  // FIXME: On 32-bit, load -> fild or movq would be more efficient
  //        (The only way to get a 16-byte load is cmpxchg16b)
  // FIXME: 16-byte ATOMIC_CMP_SWAP isn't actually hooked up at the moment.
  SDValue Zero = DAG.getConstant(0, VT);
  SDValue Swap = DAG.getAtomic(ISD::ATOMIC_CMP_SWAP, dl, VT,
                               Node->getOperand(0),
                               Node->getOperand(1), Zero, Zero,
                               cast<AtomicSDNode>(Node)->getMemOperand(),
                               cast<AtomicSDNode>(Node)->getOrdering(),
                               cast<AtomicSDNode>(Node)->getSynchScope());
  Results.push_back(Swap.getValue(0));
  Results.push_back(Swap.getValue(1));
}

static void
ReplaceATOMIC_BINARY_64(SDNode *Node, SmallVectorImpl<SDValue>&Results,
                        SelectionDAG &DAG, unsigned NewOp) {
  DebugLoc dl = Node->getDebugLoc();
  assert (Node->getValueType(0) == MVT::i64 &&
          "Only know how to expand i64 atomics");

  SDValue Chain = Node->getOperand(0);
  SDValue In1 = Node->getOperand(1);
  SDValue In2L = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                             Node->getOperand(2), DAG.getIntPtrConstant(0));
  SDValue In2H = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                             Node->getOperand(2), DAG.getIntPtrConstant(1));
  SDValue Ops[] = { Chain, In1, In2L, In2H };
  SDVTList Tys = DAG.getVTList(MVT::i32, MVT::i32, MVT::Other);
  SDValue Result =
    DAG.getMemIntrinsicNode(NewOp, dl, Tys, Ops, 4, MVT::i64,
                            cast<MemSDNode>(Node)->getMemOperand());
  SDValue OpsF[] = { Result.getValue(0), Result.getValue(1)};
  Results.push_back(DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, OpsF, 2));
  Results.push_back(Result.getValue(2));
}

/// ReplaceNodeResults - Replace a node with an illegal result type
/// with a new node built out of custom code.
void X86TargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue>&Results,
                                           SelectionDAG &DAG) const {
  DebugLoc dl = N->getDebugLoc();
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Do not know how to custom type legalize this operation!");
  case ISD::SIGN_EXTEND_INREG:
  case ISD::ADDC:
  case ISD::ADDE:
  case ISD::SUBC:
  case ISD::SUBE:
    // We don't want to expand or promote these.
    return;
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT: {
    bool IsSigned = N->getOpcode() == ISD::FP_TO_SINT;

    if (!IsSigned && !isIntegerTypeFTOL(SDValue(N, 0).getValueType()))
      return;

    std::pair<SDValue,SDValue> Vals =
        FP_TO_INTHelper(SDValue(N, 0), DAG, IsSigned, /*IsReplace=*/ true);
    SDValue FIST = Vals.first, StackSlot = Vals.second;
    if (FIST.getNode() != 0) {
      EVT VT = N->getValueType(0);
      // Return a load from the stack slot.
      if (StackSlot.getNode() != 0)
        Results.push_back(DAG.getLoad(VT, dl, FIST, StackSlot,
                                      MachinePointerInfo(),
                                      false, false, false, 0));
      else
        Results.push_back(FIST);
    }
    return;
  }
  case ISD::UINT_TO_FP: {
    if (N->getOperand(0).getValueType() != MVT::v2i32 &&
        N->getValueType(0) != MVT::v2f32)
      return;
    SDValue ZExtIn = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::v2i64,
                                 N->getOperand(0));
    SDValue Bias = DAG.getConstantFP(BitsToDouble(0x4330000000000000ULL),
                                     MVT::f64);
    SDValue VBias = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v2f64, Bias, Bias);
    SDValue Or = DAG.getNode(ISD::OR, dl, MVT::v2i64, ZExtIn,
                             DAG.getNode(ISD::BITCAST, dl, MVT::v2i64, VBias));
    Or = DAG.getNode(ISD::BITCAST, dl, MVT::v2f64, Or);
    SDValue Sub = DAG.getNode(ISD::FSUB, dl, MVT::v2f64, Or, VBias);
    Results.push_back(DAG.getNode(X86ISD::VFPROUND, dl, MVT::v4f32, Sub));
    return;
  }
  case ISD::FP_ROUND: {
    SDValue V = DAG.getNode(X86ISD::VFPROUND, dl, MVT::v4f32, N->getOperand(0));
    Results.push_back(V);
    return;
  }
  case ISD::READCYCLECOUNTER: {
    SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue TheChain = N->getOperand(0);
    SDValue rd = DAG.getNode(X86ISD::RDTSC_DAG, dl, Tys, &TheChain, 1);
    SDValue eax = DAG.getCopyFromReg(rd, dl, X86::EAX, MVT::i32,
                                     rd.getValue(1));
    SDValue edx = DAG.getCopyFromReg(eax.getValue(1), dl, X86::EDX, MVT::i32,
                                     eax.getValue(2));
    // Use a buildpair to merge the two 32-bit values into a 64-bit one.
    SDValue Ops[] = { eax, edx };
    Results.push_back(DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Ops, 2));
    Results.push_back(edx.getValue(1));
    return;
  }
  case ISD::ATOMIC_CMP_SWAP: {
    EVT T = N->getValueType(0);
    assert((T == MVT::i64 || T == MVT::i128) && "can only expand cmpxchg pair");
    bool Regs64bit = T == MVT::i128;
    EVT HalfT = Regs64bit ? MVT::i64 : MVT::i32;
    SDValue cpInL, cpInH;
    cpInL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, HalfT, N->getOperand(2),
                        DAG.getConstant(0, HalfT));
    cpInH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, HalfT, N->getOperand(2),
                        DAG.getConstant(1, HalfT));
    cpInL = DAG.getCopyToReg(N->getOperand(0), dl,
                             Regs64bit ? X86::RAX : X86::EAX,
                             cpInL, SDValue());
    cpInH = DAG.getCopyToReg(cpInL.getValue(0), dl,
                             Regs64bit ? X86::RDX : X86::EDX,
                             cpInH, cpInL.getValue(1));
    SDValue swapInL, swapInH;
    swapInL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, HalfT, N->getOperand(3),
                          DAG.getConstant(0, HalfT));
    swapInH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, HalfT, N->getOperand(3),
                          DAG.getConstant(1, HalfT));
    swapInL = DAG.getCopyToReg(cpInH.getValue(0), dl,
                               Regs64bit ? X86::RBX : X86::EBX,
                               swapInL, cpInH.getValue(1));
    swapInH = DAG.getCopyToReg(swapInL.getValue(0), dl,
                               Regs64bit ? X86::RCX : X86::ECX,
                               swapInH, swapInL.getValue(1));
    SDValue Ops[] = { swapInH.getValue(0),
                      N->getOperand(1),
                      swapInH.getValue(1) };
    SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Glue);
    MachineMemOperand *MMO = cast<AtomicSDNode>(N)->getMemOperand();
    unsigned Opcode = Regs64bit ? X86ISD::LCMPXCHG16_DAG :
                                  X86ISD::LCMPXCHG8_DAG;
    SDValue Result = DAG.getMemIntrinsicNode(Opcode, dl, Tys,
                                             Ops, 3, T, MMO);
    SDValue cpOutL = DAG.getCopyFromReg(Result.getValue(0), dl,
                                        Regs64bit ? X86::RAX : X86::EAX,
                                        HalfT, Result.getValue(1));
    SDValue cpOutH = DAG.getCopyFromReg(cpOutL.getValue(1), dl,
                                        Regs64bit ? X86::RDX : X86::EDX,
                                        HalfT, cpOutL.getValue(2));
    SDValue OpsF[] = { cpOutL.getValue(0), cpOutH.getValue(0)};
    Results.push_back(DAG.getNode(ISD::BUILD_PAIR, dl, T, OpsF, 2));
    Results.push_back(cpOutH.getValue(1));
    return;
  }
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_SWAP: {
    unsigned Opc;
    switch (N->getOpcode()) {
    default: llvm_unreachable("Unexpected opcode");
    case ISD::ATOMIC_LOAD_ADD:
      Opc = X86ISD::ATOMADD64_DAG;
      break;
    case ISD::ATOMIC_LOAD_AND:
      Opc = X86ISD::ATOMAND64_DAG;
      break;
    case ISD::ATOMIC_LOAD_NAND:
      Opc = X86ISD::ATOMNAND64_DAG;
      break;
    case ISD::ATOMIC_LOAD_OR:
      Opc = X86ISD::ATOMOR64_DAG;
      break;
    case ISD::ATOMIC_LOAD_SUB:
      Opc = X86ISD::ATOMSUB64_DAG;
      break;
    case ISD::ATOMIC_LOAD_XOR:
      Opc = X86ISD::ATOMXOR64_DAG;
      break;
    case ISD::ATOMIC_LOAD_MAX:
      Opc = X86ISD::ATOMMAX64_DAG;
      break;
    case ISD::ATOMIC_LOAD_MIN:
      Opc = X86ISD::ATOMMIN64_DAG;
      break;
    case ISD::ATOMIC_LOAD_UMAX:
      Opc = X86ISD::ATOMUMAX64_DAG;
      break;
    case ISD::ATOMIC_LOAD_UMIN:
      Opc = X86ISD::ATOMUMIN64_DAG;
      break;
    case ISD::ATOMIC_SWAP:
      Opc = X86ISD::ATOMSWAP64_DAG;
      break;
    }
    ReplaceATOMIC_BINARY_64(N, Results, DAG, Opc);
    return;
  }
  case ISD::ATOMIC_LOAD:
    ReplaceATOMIC_LOAD(N, Results, DAG);
  }
}

const char *X86TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return NULL;
  case X86ISD::BSF:                return "X86ISD::BSF";
  case X86ISD::BSR:                return "X86ISD::BSR";
  case X86ISD::SHLD:               return "X86ISD::SHLD";
  case X86ISD::SHRD:               return "X86ISD::SHRD";
  case X86ISD::FAND:               return "X86ISD::FAND";
  case X86ISD::FOR:                return "X86ISD::FOR";
  case X86ISD::FXOR:               return "X86ISD::FXOR";
  case X86ISD::FSRL:               return "X86ISD::FSRL";
  case X86ISD::FILD:               return "X86ISD::FILD";
  case X86ISD::FILD_FLAG:          return "X86ISD::FILD_FLAG";
  case X86ISD::FP_TO_INT16_IN_MEM: return "X86ISD::FP_TO_INT16_IN_MEM";
  case X86ISD::FP_TO_INT32_IN_MEM: return "X86ISD::FP_TO_INT32_IN_MEM";
  case X86ISD::FP_TO_INT64_IN_MEM: return "X86ISD::FP_TO_INT64_IN_MEM";
  case X86ISD::FLD:                return "X86ISD::FLD";
  case X86ISD::FST:                return "X86ISD::FST";
  case X86ISD::CALL:               return "X86ISD::CALL";
  case X86ISD::RDTSC_DAG:          return "X86ISD::RDTSC_DAG";
  case X86ISD::BT:                 return "X86ISD::BT";
  case X86ISD::CMP:                return "X86ISD::CMP";
  case X86ISD::COMI:               return "X86ISD::COMI";
  case X86ISD::UCOMI:              return "X86ISD::UCOMI";
  case X86ISD::SETCC:              return "X86ISD::SETCC";
  case X86ISD::SETCC_CARRY:        return "X86ISD::SETCC_CARRY";
  case X86ISD::FSETCCsd:           return "X86ISD::FSETCCsd";
  case X86ISD::FSETCCss:           return "X86ISD::FSETCCss";
  case X86ISD::CMOV:               return "X86ISD::CMOV";
  case X86ISD::BRCOND:             return "X86ISD::BRCOND";
  case X86ISD::RET_FLAG:           return "X86ISD::RET_FLAG";
  case X86ISD::REP_STOS:           return "X86ISD::REP_STOS";
  case X86ISD::REP_MOVS:           return "X86ISD::REP_MOVS";
  case X86ISD::GlobalBaseReg:      return "X86ISD::GlobalBaseReg";
  case X86ISD::Wrapper:            return "X86ISD::Wrapper";
  case X86ISD::WrapperRIP:         return "X86ISD::WrapperRIP";
  case X86ISD::PEXTRB:             return "X86ISD::PEXTRB";
  case X86ISD::PEXTRW:             return "X86ISD::PEXTRW";
  case X86ISD::INSERTPS:           return "X86ISD::INSERTPS";
  case X86ISD::PINSRB:             return "X86ISD::PINSRB";
  case X86ISD::PINSRW:             return "X86ISD::PINSRW";
  case X86ISD::PSHUFB:             return "X86ISD::PSHUFB";
  case X86ISD::ANDNP:              return "X86ISD::ANDNP";
  case X86ISD::PSIGN:              return "X86ISD::PSIGN";
  case X86ISD::BLENDV:             return "X86ISD::BLENDV";
  case X86ISD::BLENDPW:            return "X86ISD::BLENDPW";
  case X86ISD::BLENDPS:            return "X86ISD::BLENDPS";
  case X86ISD::BLENDPD:            return "X86ISD::BLENDPD";
  case X86ISD::HADD:               return "X86ISD::HADD";
  case X86ISD::HSUB:               return "X86ISD::HSUB";
  case X86ISD::FHADD:              return "X86ISD::FHADD";
  case X86ISD::FHSUB:              return "X86ISD::FHSUB";
  case X86ISD::FMAX:               return "X86ISD::FMAX";
  case X86ISD::FMIN:               return "X86ISD::FMIN";
  case X86ISD::FMAXC:              return "X86ISD::FMAXC";
  case X86ISD::FMINC:              return "X86ISD::FMINC";
  case X86ISD::FRSQRT:             return "X86ISD::FRSQRT";
  case X86ISD::FRCP:               return "X86ISD::FRCP";
  case X86ISD::TLSADDR:            return "X86ISD::TLSADDR";
  case X86ISD::TLSBASEADDR:        return "X86ISD::TLSBASEADDR";
  case X86ISD::TLSCALL:            return "X86ISD::TLSCALL";
  case X86ISD::EH_SJLJ_SETJMP:     return "X86ISD::EH_SJLJ_SETJMP";
  case X86ISD::EH_SJLJ_LONGJMP:    return "X86ISD::EH_SJLJ_LONGJMP";
  case X86ISD::EH_RETURN:          return "X86ISD::EH_RETURN";
  case X86ISD::TC_RETURN:          return "X86ISD::TC_RETURN";
  case X86ISD::FNSTCW16m:          return "X86ISD::FNSTCW16m";
  case X86ISD::FNSTSW16r:          return "X86ISD::FNSTSW16r";
  case X86ISD::LCMPXCHG_DAG:       return "X86ISD::LCMPXCHG_DAG";
  case X86ISD::LCMPXCHG8_DAG:      return "X86ISD::LCMPXCHG8_DAG";
  case X86ISD::ATOMADD64_DAG:      return "X86ISD::ATOMADD64_DAG";
  case X86ISD::ATOMSUB64_DAG:      return "X86ISD::ATOMSUB64_DAG";
  case X86ISD::ATOMOR64_DAG:       return "X86ISD::ATOMOR64_DAG";
  case X86ISD::ATOMXOR64_DAG:      return "X86ISD::ATOMXOR64_DAG";
  case X86ISD::ATOMAND64_DAG:      return "X86ISD::ATOMAND64_DAG";
  case X86ISD::ATOMNAND64_DAG:     return "X86ISD::ATOMNAND64_DAG";
  case X86ISD::VZEXT_MOVL:         return "X86ISD::VZEXT_MOVL";
  case X86ISD::VSEXT_MOVL:         return "X86ISD::VSEXT_MOVL";
  case X86ISD::VZEXT_LOAD:         return "X86ISD::VZEXT_LOAD";
  case X86ISD::VZEXT:              return "X86ISD::VZEXT";
  case X86ISD::VSEXT:              return "X86ISD::VSEXT";
  case X86ISD::VFPEXT:             return "X86ISD::VFPEXT";
  case X86ISD::VFPROUND:           return "X86ISD::VFPROUND";
  case X86ISD::VSHLDQ:             return "X86ISD::VSHLDQ";
  case X86ISD::VSRLDQ:             return "X86ISD::VSRLDQ";
  case X86ISD::VSHL:               return "X86ISD::VSHL";
  case X86ISD::VSRL:               return "X86ISD::VSRL";
  case X86ISD::VSRA:               return "X86ISD::VSRA";
  case X86ISD::VSHLI:              return "X86ISD::VSHLI";
  case X86ISD::VSRLI:              return "X86ISD::VSRLI";
  case X86ISD::VSRAI:              return "X86ISD::VSRAI";
  case X86ISD::CMPP:               return "X86ISD::CMPP";
  case X86ISD::PCMPEQ:             return "X86ISD::PCMPEQ";
  case X86ISD::PCMPGT:             return "X86ISD::PCMPGT";
  case X86ISD::ADD:                return "X86ISD::ADD";
  case X86ISD::SUB:                return "X86ISD::SUB";
  case X86ISD::ADC:                return "X86ISD::ADC";
  case X86ISD::SBB:                return "X86ISD::SBB";
  case X86ISD::SMUL:               return "X86ISD::SMUL";
  case X86ISD::UMUL:               return "X86ISD::UMUL";
  case X86ISD::INC:                return "X86ISD::INC";
  case X86ISD::DEC:                return "X86ISD::DEC";
  case X86ISD::OR:                 return "X86ISD::OR";
  case X86ISD::XOR:                return "X86ISD::XOR";
  case X86ISD::AND:                return "X86ISD::AND";
  case X86ISD::ANDN:               return "X86ISD::ANDN";
  case X86ISD::BLSI:               return "X86ISD::BLSI";
  case X86ISD::BLSMSK:             return "X86ISD::BLSMSK";
  case X86ISD::BLSR:               return "X86ISD::BLSR";
  case X86ISD::MUL_IMM:            return "X86ISD::MUL_IMM";
  case X86ISD::PTEST:              return "X86ISD::PTEST";
  case X86ISD::TESTP:              return "X86ISD::TESTP";
  case X86ISD::PALIGN:             return "X86ISD::PALIGN";
  case X86ISD::PSHUFD:             return "X86ISD::PSHUFD";
  case X86ISD::PSHUFHW:            return "X86ISD::PSHUFHW";
  case X86ISD::PSHUFLW:            return "X86ISD::PSHUFLW";
  case X86ISD::SHUFP:              return "X86ISD::SHUFP";
  case X86ISD::MOVLHPS:            return "X86ISD::MOVLHPS";
  case X86ISD::MOVLHPD:            return "X86ISD::MOVLHPD";
  case X86ISD::MOVHLPS:            return "X86ISD::MOVHLPS";
  case X86ISD::MOVLPS:             return "X86ISD::MOVLPS";
  case X86ISD::MOVLPD:             return "X86ISD::MOVLPD";
  case X86ISD::MOVDDUP:            return "X86ISD::MOVDDUP";
  case X86ISD::MOVSHDUP:           return "X86ISD::MOVSHDUP";
  case X86ISD::MOVSLDUP:           return "X86ISD::MOVSLDUP";
  case X86ISD::MOVSD:              return "X86ISD::MOVSD";
  case X86ISD::MOVSS:              return "X86ISD::MOVSS";
  case X86ISD::UNPCKL:             return "X86ISD::UNPCKL";
  case X86ISD::UNPCKH:             return "X86ISD::UNPCKH";
  case X86ISD::VBROADCAST:         return "X86ISD::VBROADCAST";
  case X86ISD::VPERMILP:           return "X86ISD::VPERMILP";
  case X86ISD::VPERM2X128:         return "X86ISD::VPERM2X128";
  case X86ISD::VPERMV:             return "X86ISD::VPERMV";
  case X86ISD::VPERMI:             return "X86ISD::VPERMI";
  case X86ISD::PMULUDQ:            return "X86ISD::PMULUDQ";
  case X86ISD::VASTART_SAVE_XMM_REGS: return "X86ISD::VASTART_SAVE_XMM_REGS";
  case X86ISD::VAARG_64:           return "X86ISD::VAARG_64";
  case X86ISD::WIN_ALLOCA:         return "X86ISD::WIN_ALLOCA";
  case X86ISD::MEMBARRIER:         return "X86ISD::MEMBARRIER";
  case X86ISD::SEG_ALLOCA:         return "X86ISD::SEG_ALLOCA";
  case X86ISD::WIN_FTOL:           return "X86ISD::WIN_FTOL";
  case X86ISD::SAHF:               return "X86ISD::SAHF";
  case X86ISD::RDRAND:             return "X86ISD::RDRAND";
  case X86ISD::FMADD:              return "X86ISD::FMADD";
  case X86ISD::FMSUB:              return "X86ISD::FMSUB";
  case X86ISD::FNMADD:             return "X86ISD::FNMADD";
  case X86ISD::FNMSUB:             return "X86ISD::FNMSUB";
  case X86ISD::FMADDSUB:           return "X86ISD::FMADDSUB";
  case X86ISD::FMSUBADD:           return "X86ISD::FMSUBADD";
  case X86ISD::PCMPESTRI:          return "X86ISD::PCMPESTRI";
  case X86ISD::PCMPISTRI:          return "X86ISD::PCMPISTRI";
  }
}

// isLegalAddressingMode - Return true if the addressing mode represented
// by AM is legal for this target, for a load/store of the specified type.
bool X86TargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                              Type *Ty) const {
  // X86 supports extremely general addressing modes.
  CodeModel::Model M = getTargetMachine().getCodeModel();
  Reloc::Model R = getTargetMachine().getRelocationModel();

  // X86 allows a sign-extended 32-bit immediate field as a displacement.
  if (!X86::isOffsetSuitableForCodeModel(AM.BaseOffs, M, AM.BaseGV != NULL))
    return false;

  if (AM.BaseGV) {
    unsigned GVFlags =
      Subtarget->ClassifyGlobalReference(AM.BaseGV, getTargetMachine());

    // If a reference to this global requires an extra load, we can't fold it.
    if (isGlobalStubReference(GVFlags))
      return false;

    // If BaseGV requires a register for the PIC base, we cannot also have a
    // BaseReg specified.
    if (AM.HasBaseReg && isGlobalRelativeToPICBase(GVFlags))
      return false;

    // If lower 4G is not available, then we must use rip-relative addressing.
    if ((M != CodeModel::Small || R != Reloc::Static) &&
        Subtarget->is64Bit() && (AM.BaseOffs || AM.Scale > 1))
      return false;
  }

  switch (AM.Scale) {
  case 0:
  case 1:
  case 2:
  case 4:
  case 8:
    // These scales always work.
    break;
  case 3:
  case 5:
  case 9:
    // These scales are formed with basereg+scalereg.  Only accept if there is
    // no basereg yet.
    if (AM.HasBaseReg)
      return false;
    break;
  default:  // Other stuff never works.
    return false;
  }

  return true;
}


bool X86TargetLowering::isTruncateFree(Type *Ty1, Type *Ty2) const {
  if (!Ty1->isIntegerTy() || !Ty2->isIntegerTy())
    return false;
  unsigned NumBits1 = Ty1->getPrimitiveSizeInBits();
  unsigned NumBits2 = Ty2->getPrimitiveSizeInBits();
  if (NumBits1 <= NumBits2)
    return false;
  return true;
}

bool X86TargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  return Imm == (int32_t)Imm;
}

bool X86TargetLowering::isLegalAddImmediate(int64_t Imm) const {
  // Can also use sub to handle negated immediates.
  return Imm == (int32_t)Imm;
}

bool X86TargetLowering::isTruncateFree(EVT VT1, EVT VT2) const {
  if (!VT1.isInteger() || !VT2.isInteger())
    return false;
  unsigned NumBits1 = VT1.getSizeInBits();
  unsigned NumBits2 = VT2.getSizeInBits();
  if (NumBits1 <= NumBits2)
    return false;
  return true;
}

bool X86TargetLowering::isZExtFree(Type *Ty1, Type *Ty2) const {
  // x86-64 implicitly zero-extends 32-bit results in 64-bit registers.
  return Ty1->isIntegerTy(32) && Ty2->isIntegerTy(64) && Subtarget->is64Bit();
}

bool X86TargetLowering::isZExtFree(EVT VT1, EVT VT2) const {
  // x86-64 implicitly zero-extends 32-bit results in 64-bit registers.
  return VT1 == MVT::i32 && VT2 == MVT::i64 && Subtarget->is64Bit();
}

bool X86TargetLowering::isNarrowingProfitable(EVT VT1, EVT VT2) const {
  // i16 instructions are longer (0x66 prefix) and potentially slower.
  return !(VT1 == MVT::i32 && VT2 == MVT::i16);
}

/// isShuffleMaskLegal - Targets can use this to indicate that they only
/// support *some* VECTOR_SHUFFLE operations, those with specific masks.
/// By default, if a target supports the VECTOR_SHUFFLE node, all mask values
/// are assumed to be legal.
bool
X86TargetLowering::isShuffleMaskLegal(const SmallVectorImpl<int> &M,
                                      EVT VT) const {
  // Very little shuffling can be done for 64-bit vectors right now.
  if (VT.getSizeInBits() == 64)
    return false;

  // FIXME: pshufb, blends, shifts.
  return (VT.getVectorNumElements() == 2 ||
          ShuffleVectorSDNode::isSplatMask(&M[0], VT) ||
          isMOVLMask(M, VT) ||
          isSHUFPMask(M, VT, Subtarget->hasAVX()) ||
          isPSHUFDMask(M, VT) ||
          isPSHUFHWMask(M, VT, Subtarget->hasAVX2()) ||
          isPSHUFLWMask(M, VT, Subtarget->hasAVX2()) ||
          isPALIGNRMask(M, VT, Subtarget) ||
          isUNPCKLMask(M, VT, Subtarget->hasAVX2()) ||
          isUNPCKHMask(M, VT, Subtarget->hasAVX2()) ||
          isUNPCKL_v_undef_Mask(M, VT, Subtarget->hasAVX2()) ||
          isUNPCKH_v_undef_Mask(M, VT, Subtarget->hasAVX2()));
}

bool
X86TargetLowering::isVectorClearMaskLegal(const SmallVectorImpl<int> &Mask,
                                          EVT VT) const {
  unsigned NumElts = VT.getVectorNumElements();
  // FIXME: This collection of masks seems suspect.
  if (NumElts == 2)
    return true;
  if (NumElts == 4 && VT.is128BitVector()) {
    return (isMOVLMask(Mask, VT)  ||
            isCommutedMOVLMask(Mask, VT, true) ||
            isSHUFPMask(Mask, VT, Subtarget->hasAVX()) ||
            isSHUFPMask(Mask, VT, Subtarget->hasAVX(), /* Commuted */ true));
  }
  return false;
}

//===----------------------------------------------------------------------===//
//                           X86 Scheduler Hooks
//===----------------------------------------------------------------------===//

/// Utility function to emit xbegin specifying the start of an RTM region.
static MachineBasicBlock *EmitXBegin(MachineInstr *MI, MachineBasicBlock *MBB,
                                     const TargetInstrInfo *TII) {
  DebugLoc DL = MI->getDebugLoc();

  const BasicBlock *BB = MBB->getBasicBlock();
  MachineFunction::iterator I = MBB;
  ++I;

  // For the v = xbegin(), we generate
  //
  // thisMBB:
  //  xbegin sinkMBB
  //
  // mainMBB:
  //  eax = -1
  //
  // sinkMBB:
  //  v = eax

  MachineBasicBlock *thisMBB = MBB;
  MachineFunction *MF = MBB->getParent();
  MachineBasicBlock *mainMBB = MF->CreateMachineBasicBlock(BB);
  MachineBasicBlock *sinkMBB = MF->CreateMachineBasicBlock(BB);
  MF->insert(I, mainMBB);
  MF->insert(I, sinkMBB);

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), MBB,
                  llvm::next(MachineBasicBlock::iterator(MI)), MBB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  // thisMBB:
  //  xbegin sinkMBB
  //  # fallthrough to mainMBB
  //  # abortion to sinkMBB
  BuildMI(thisMBB, DL, TII->get(X86::XBEGIN_4)).addMBB(sinkMBB);
  thisMBB->addSuccessor(mainMBB);
  thisMBB->addSuccessor(sinkMBB);

  // mainMBB:
  //  EAX = -1
  BuildMI(mainMBB, DL, TII->get(X86::MOV32ri), X86::EAX).addImm(-1);
  mainMBB->addSuccessor(sinkMBB);

  // sinkMBB:
  // EAX is live into the sinkMBB
  sinkMBB->addLiveIn(X86::EAX);
  BuildMI(*sinkMBB, sinkMBB->begin(), DL,
          TII->get(TargetOpcode::COPY), MI->getOperand(0).getReg())
    .addReg(X86::EAX);

  MI->eraseFromParent();
  return sinkMBB;
}

// Get CMPXCHG opcode for the specified data type.
static unsigned getCmpXChgOpcode(EVT VT) {
  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::i8:  return X86::LCMPXCHG8;
  case MVT::i16: return X86::LCMPXCHG16;
  case MVT::i32: return X86::LCMPXCHG32;
  case MVT::i64: return X86::LCMPXCHG64;
  default:
    break;
  }
  llvm_unreachable("Invalid operand size!");
}

// Get LOAD opcode for the specified data type.
static unsigned getLoadOpcode(EVT VT) {
  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::i8:  return X86::MOV8rm;
  case MVT::i16: return X86::MOV16rm;
  case MVT::i32: return X86::MOV32rm;
  case MVT::i64: return X86::MOV64rm;
  default:
    break;
  }
  llvm_unreachable("Invalid operand size!");
}

// Get opcode of the non-atomic one from the specified atomic instruction.
static unsigned getNonAtomicOpcode(unsigned Opc) {
  switch (Opc) {
  case X86::ATOMAND8:  return X86::AND8rr;
  case X86::ATOMAND16: return X86::AND16rr;
  case X86::ATOMAND32: return X86::AND32rr;
  case X86::ATOMAND64: return X86::AND64rr;
  case X86::ATOMOR8:   return X86::OR8rr;
  case X86::ATOMOR16:  return X86::OR16rr;
  case X86::ATOMOR32:  return X86::OR32rr;
  case X86::ATOMOR64:  return X86::OR64rr;
  case X86::ATOMXOR8:  return X86::XOR8rr;
  case X86::ATOMXOR16: return X86::XOR16rr;
  case X86::ATOMXOR32: return X86::XOR32rr;
  case X86::ATOMXOR64: return X86::XOR64rr;
  }
  llvm_unreachable("Unhandled atomic-load-op opcode!");
}

// Get opcode of the non-atomic one from the specified atomic instruction with
// extra opcode.
static unsigned getNonAtomicOpcodeWithExtraOpc(unsigned Opc,
                                               unsigned &ExtraOpc) {
  switch (Opc) {
  case X86::ATOMNAND8:  ExtraOpc = X86::NOT8r;   return X86::AND8rr;
  case X86::ATOMNAND16: ExtraOpc = X86::NOT16r;  return X86::AND16rr;
  case X86::ATOMNAND32: ExtraOpc = X86::NOT32r;  return X86::AND32rr;
  case X86::ATOMNAND64: ExtraOpc = X86::NOT64r;  return X86::AND64rr;
  case X86::ATOMMAX8:   ExtraOpc = X86::CMP8rr;  return X86::CMOVL32rr;
  case X86::ATOMMAX16:  ExtraOpc = X86::CMP16rr; return X86::CMOVL16rr;
  case X86::ATOMMAX32:  ExtraOpc = X86::CMP32rr; return X86::CMOVL32rr;
  case X86::ATOMMAX64:  ExtraOpc = X86::CMP64rr; return X86::CMOVL64rr;
  case X86::ATOMMIN8:   ExtraOpc = X86::CMP8rr;  return X86::CMOVG32rr;
  case X86::ATOMMIN16:  ExtraOpc = X86::CMP16rr; return X86::CMOVG16rr;
  case X86::ATOMMIN32:  ExtraOpc = X86::CMP32rr; return X86::CMOVG32rr;
  case X86::ATOMMIN64:  ExtraOpc = X86::CMP64rr; return X86::CMOVG64rr;
  case X86::ATOMUMAX8:  ExtraOpc = X86::CMP8rr;  return X86::CMOVB32rr;
  case X86::ATOMUMAX16: ExtraOpc = X86::CMP16rr; return X86::CMOVB16rr;
  case X86::ATOMUMAX32: ExtraOpc = X86::CMP32rr; return X86::CMOVB32rr;
  case X86::ATOMUMAX64: ExtraOpc = X86::CMP64rr; return X86::CMOVB64rr;
  case X86::ATOMUMIN8:  ExtraOpc = X86::CMP8rr;  return X86::CMOVA32rr;
  case X86::ATOMUMIN16: ExtraOpc = X86::CMP16rr; return X86::CMOVA16rr;
  case X86::ATOMUMIN32: ExtraOpc = X86::CMP32rr; return X86::CMOVA32rr;
  case X86::ATOMUMIN64: ExtraOpc = X86::CMP64rr; return X86::CMOVA64rr;
  }
  llvm_unreachable("Unhandled atomic-load-op opcode!");
}

// Get opcode of the non-atomic one from the specified atomic instruction for
// 64-bit data type on 32-bit target.
static unsigned getNonAtomic6432Opcode(unsigned Opc, unsigned &HiOpc) {
  switch (Opc) {
  case X86::ATOMAND6432:  HiOpc = X86::AND32rr; return X86::AND32rr;
  case X86::ATOMOR6432:   HiOpc = X86::OR32rr;  return X86::OR32rr;
  case X86::ATOMXOR6432:  HiOpc = X86::XOR32rr; return X86::XOR32rr;
  case X86::ATOMADD6432:  HiOpc = X86::ADC32rr; return X86::ADD32rr;
  case X86::ATOMSUB6432:  HiOpc = X86::SBB32rr; return X86::SUB32rr;
  case X86::ATOMSWAP6432: HiOpc = X86::MOV32rr; return X86::MOV32rr;
  case X86::ATOMMAX6432:  HiOpc = X86::SETLr;   return X86::SETLr;
  case X86::ATOMMIN6432:  HiOpc = X86::SETGr;   return X86::SETGr;
  case X86::ATOMUMAX6432: HiOpc = X86::SETBr;   return X86::SETBr;
  case X86::ATOMUMIN6432: HiOpc = X86::SETAr;   return X86::SETAr;
  }
  llvm_unreachable("Unhandled atomic-load-op opcode!");
}

// Get opcode of the non-atomic one from the specified atomic instruction for
// 64-bit data type on 32-bit target with extra opcode.
static unsigned getNonAtomic6432OpcodeWithExtraOpc(unsigned Opc,
                                                   unsigned &HiOpc,
                                                   unsigned &ExtraOpc) {
  switch (Opc) {
  case X86::ATOMNAND6432:
    ExtraOpc = X86::NOT32r;
    HiOpc = X86::AND32rr;
    return X86::AND32rr;
  }
  llvm_unreachable("Unhandled atomic-load-op opcode!");
}

// Get pseudo CMOV opcode from the specified data type.
static unsigned getPseudoCMOVOpc(EVT VT) {
  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::i8:  return X86::CMOV_GR8;
  case MVT::i16: return X86::CMOV_GR16;
  case MVT::i32: return X86::CMOV_GR32;
  default:
    break;
  }
  llvm_unreachable("Unknown CMOV opcode!");
}

// EmitAtomicLoadArith - emit the code sequence for pseudo atomic instructions.
// They will be translated into a spin-loop or compare-exchange loop from
//
//    ...
//    dst = atomic-fetch-op MI.addr, MI.val
//    ...
//
// to
//
//    ...
//    EAX = LOAD MI.addr
// loop:
//    t1 = OP MI.val, EAX
//    LCMPXCHG [MI.addr], t1, [EAX is implicitly used & defined]
//    JNE loop
// sink:
//    dst = EAX
//    ...
MachineBasicBlock *
X86TargetLowering::EmitAtomicLoadArith(MachineInstr *MI,
                                       MachineBasicBlock *MBB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();

  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  const BasicBlock *BB = MBB->getBasicBlock();
  MachineFunction::iterator I = MBB;
  ++I;

  assert(MI->getNumOperands() <= X86::AddrNumOperands + 2 &&
         "Unexpected number of operands");

  assert(MI->hasOneMemOperand() &&
         "Expected atomic-load-op to have one memoperand");

  // Memory Reference
  MachineInstr::mmo_iterator MMOBegin = MI->memoperands_begin();
  MachineInstr::mmo_iterator MMOEnd = MI->memoperands_end();

  unsigned DstReg, SrcReg;
  unsigned MemOpndSlot;

  unsigned CurOp = 0;

  DstReg = MI->getOperand(CurOp++).getReg();
  MemOpndSlot = CurOp;
  CurOp += X86::AddrNumOperands;
  SrcReg = MI->getOperand(CurOp++).getReg();

  const TargetRegisterClass *RC = MRI.getRegClass(DstReg);
  MVT::SimpleValueType VT = *RC->vt_begin();
  unsigned AccPhyReg = getX86SubSuperRegister(X86::EAX, VT);

  unsigned LCMPXCHGOpc = getCmpXChgOpcode(VT);
  unsigned LOADOpc = getLoadOpcode(VT);

  // For the atomic load-arith operator, we generate
  //
  //  thisMBB:
  //    EAX = LOAD [MI.addr]
  //  mainMBB:
  //    t1 = OP MI.val, EAX
  //    LCMPXCHG [MI.addr], t1, [EAX is implicitly used & defined]
  //    JNE mainMBB
  //  sinkMBB:

  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *mainMBB = MF->CreateMachineBasicBlock(BB);
  MachineBasicBlock *sinkMBB = MF->CreateMachineBasicBlock(BB);
  MF->insert(I, mainMBB);
  MF->insert(I, sinkMBB);

  MachineInstrBuilder MIB;

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), MBB,
                  llvm::next(MachineBasicBlock::iterator(MI)), MBB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  // thisMBB:
  MIB = BuildMI(thisMBB, DL, TII->get(LOADOpc), AccPhyReg);
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i)
    MIB.addOperand(MI->getOperand(MemOpndSlot + i));
  MIB.setMemRefs(MMOBegin, MMOEnd);

  thisMBB->addSuccessor(mainMBB);

  // mainMBB:
  MachineBasicBlock *origMainMBB = mainMBB;
  mainMBB->addLiveIn(AccPhyReg);

  // Copy AccPhyReg as it is used more than once.
  unsigned AccReg = MRI.createVirtualRegister(RC);
  BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), AccReg)
    .addReg(AccPhyReg);

  unsigned t1 = MRI.createVirtualRegister(RC);
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default:
    llvm_unreachable("Unhandled atomic-load-op opcode!");
  case X86::ATOMAND8:
  case X86::ATOMAND16:
  case X86::ATOMAND32:
  case X86::ATOMAND64:
  case X86::ATOMOR8:
  case X86::ATOMOR16:
  case X86::ATOMOR32:
  case X86::ATOMOR64:
  case X86::ATOMXOR8:
  case X86::ATOMXOR16:
  case X86::ATOMXOR32:
  case X86::ATOMXOR64: {
    unsigned ARITHOpc = getNonAtomicOpcode(Opc);
    BuildMI(mainMBB, DL, TII->get(ARITHOpc), t1).addReg(SrcReg)
      .addReg(AccReg);
    break;
  }
  case X86::ATOMNAND8:
  case X86::ATOMNAND16:
  case X86::ATOMNAND32:
  case X86::ATOMNAND64: {
    unsigned t2 = MRI.createVirtualRegister(RC);
    unsigned NOTOpc;
    unsigned ANDOpc = getNonAtomicOpcodeWithExtraOpc(Opc, NOTOpc);
    BuildMI(mainMBB, DL, TII->get(ANDOpc), t2).addReg(SrcReg)
      .addReg(AccReg);
    BuildMI(mainMBB, DL, TII->get(NOTOpc), t1).addReg(t2);
    break;
  }
  case X86::ATOMMAX8:
  case X86::ATOMMAX16:
  case X86::ATOMMAX32:
  case X86::ATOMMAX64:
  case X86::ATOMMIN8:
  case X86::ATOMMIN16:
  case X86::ATOMMIN32:
  case X86::ATOMMIN64:
  case X86::ATOMUMAX8:
  case X86::ATOMUMAX16:
  case X86::ATOMUMAX32:
  case X86::ATOMUMAX64:
  case X86::ATOMUMIN8:
  case X86::ATOMUMIN16:
  case X86::ATOMUMIN32:
  case X86::ATOMUMIN64: {
    unsigned CMPOpc;
    unsigned CMOVOpc = getNonAtomicOpcodeWithExtraOpc(Opc, CMPOpc);

    BuildMI(mainMBB, DL, TII->get(CMPOpc))
      .addReg(SrcReg)
      .addReg(AccReg);

    if (Subtarget->hasCMov()) {
      if (VT != MVT::i8) {
        // Native support
        BuildMI(mainMBB, DL, TII->get(CMOVOpc), t1)
          .addReg(SrcReg)
          .addReg(AccReg);
      } else {
        // Promote i8 to i32 to use CMOV32
        const TargetRegisterClass *RC32 = getRegClassFor(MVT::i32);
        unsigned SrcReg32 = MRI.createVirtualRegister(RC32);
        unsigned AccReg32 = MRI.createVirtualRegister(RC32);
        unsigned t2 = MRI.createVirtualRegister(RC32);

        unsigned Undef = MRI.createVirtualRegister(RC32);
        BuildMI(mainMBB, DL, TII->get(TargetOpcode::IMPLICIT_DEF), Undef);

        BuildMI(mainMBB, DL, TII->get(TargetOpcode::INSERT_SUBREG), SrcReg32)
          .addReg(Undef)
          .addReg(SrcReg)
          .addImm(X86::sub_8bit);
        BuildMI(mainMBB, DL, TII->get(TargetOpcode::INSERT_SUBREG), AccReg32)
          .addReg(Undef)
          .addReg(AccReg)
          .addImm(X86::sub_8bit);

        BuildMI(mainMBB, DL, TII->get(CMOVOpc), t2)
          .addReg(SrcReg32)
          .addReg(AccReg32);

        BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), t1)
          .addReg(t2, 0, X86::sub_8bit);
      }
    } else {
      // Use pseudo select and lower them.
      assert((VT == MVT::i8 || VT == MVT::i16 || VT == MVT::i32) &&
             "Invalid atomic-load-op transformation!");
      unsigned SelOpc = getPseudoCMOVOpc(VT);
      X86::CondCode CC = X86::getCondFromCMovOpc(CMOVOpc);
      assert(CC != X86::COND_INVALID && "Invalid atomic-load-op transformation!");
      MIB = BuildMI(mainMBB, DL, TII->get(SelOpc), t1)
              .addReg(SrcReg).addReg(AccReg)
              .addImm(CC);
      mainMBB = EmitLoweredSelect(MIB, mainMBB);
    }
    break;
  }
  }

  // Copy AccPhyReg back from virtual register.
  BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), AccPhyReg)
    .addReg(AccReg);

  MIB = BuildMI(mainMBB, DL, TII->get(LCMPXCHGOpc));
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i)
    MIB.addOperand(MI->getOperand(MemOpndSlot + i));
  MIB.addReg(t1);
  MIB.setMemRefs(MMOBegin, MMOEnd);

  BuildMI(mainMBB, DL, TII->get(X86::JNE_4)).addMBB(origMainMBB);

  mainMBB->addSuccessor(origMainMBB);
  mainMBB->addSuccessor(sinkMBB);

  // sinkMBB:
  sinkMBB->addLiveIn(AccPhyReg);

  BuildMI(*sinkMBB, sinkMBB->begin(), DL,
          TII->get(TargetOpcode::COPY), DstReg)
    .addReg(AccPhyReg);

  MI->eraseFromParent();
  return sinkMBB;
}

// EmitAtomicLoadArith6432 - emit the code sequence for pseudo atomic
// instructions. They will be translated into a spin-loop or compare-exchange
// loop from
//
//    ...
//    dst = atomic-fetch-op MI.addr, MI.val
//    ...
//
// to
//
//    ...
//    EAX = LOAD [MI.addr + 0]
//    EDX = LOAD [MI.addr + 4]
// loop:
//    EBX = OP MI.val.lo, EAX
//    ECX = OP MI.val.hi, EDX
//    LCMPXCHG8B [MI.addr], [ECX:EBX & EDX:EAX are implicitly used and EDX:EAX is implicitly defined]
//    JNE loop
// sink:
//    dst = EDX:EAX
//    ...
MachineBasicBlock *
X86TargetLowering::EmitAtomicLoadArith6432(MachineInstr *MI,
                                           MachineBasicBlock *MBB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();

  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  const BasicBlock *BB = MBB->getBasicBlock();
  MachineFunction::iterator I = MBB;
  ++I;

  assert(MI->getNumOperands() <= X86::AddrNumOperands + 4 &&
         "Unexpected number of operands");

  assert(MI->hasOneMemOperand() &&
         "Expected atomic-load-op32 to have one memoperand");

  // Memory Reference
  MachineInstr::mmo_iterator MMOBegin = MI->memoperands_begin();
  MachineInstr::mmo_iterator MMOEnd = MI->memoperands_end();

  unsigned DstLoReg, DstHiReg;
  unsigned SrcLoReg, SrcHiReg;
  unsigned MemOpndSlot;

  unsigned CurOp = 0;

  DstLoReg = MI->getOperand(CurOp++).getReg();
  DstHiReg = MI->getOperand(CurOp++).getReg();
  MemOpndSlot = CurOp;
  CurOp += X86::AddrNumOperands;
  SrcLoReg = MI->getOperand(CurOp++).getReg();
  SrcHiReg = MI->getOperand(CurOp++).getReg();

  const TargetRegisterClass *RC = &X86::GR32RegClass;
  const TargetRegisterClass *RC8 = &X86::GR8RegClass;

  unsigned LCMPXCHGOpc = X86::LCMPXCHG8B;
  unsigned LOADOpc = X86::MOV32rm;

  // For the atomic load-arith operator, we generate
  //
  //  thisMBB:
  //    EAX = LOAD [MI.addr + 0]
  //    EDX = LOAD [MI.addr + 4]
  //  mainMBB:
  //    EBX = OP MI.vallo, EAX
  //    ECX = OP MI.valhi, EDX
  //    LCMPXCHG8B [MI.addr], [ECX:EBX & EDX:EAX are implicitly used and EDX:EAX is implicitly defined]
  //    JNE mainMBB
  //  sinkMBB:

  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *mainMBB = MF->CreateMachineBasicBlock(BB);
  MachineBasicBlock *sinkMBB = MF->CreateMachineBasicBlock(BB);
  MF->insert(I, mainMBB);
  MF->insert(I, sinkMBB);

  MachineInstrBuilder MIB;

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), MBB,
                  llvm::next(MachineBasicBlock::iterator(MI)), MBB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  // thisMBB:
  // Lo
  MIB = BuildMI(thisMBB, DL, TII->get(LOADOpc), X86::EAX);
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i)
    MIB.addOperand(MI->getOperand(MemOpndSlot + i));
  MIB.setMemRefs(MMOBegin, MMOEnd);
  // Hi
  MIB = BuildMI(thisMBB, DL, TII->get(LOADOpc), X86::EDX);
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i) {
    if (i == X86::AddrDisp)
      MIB.addDisp(MI->getOperand(MemOpndSlot + i), 4); // 4 == sizeof(i32)
    else
      MIB.addOperand(MI->getOperand(MemOpndSlot + i));
  }
  MIB.setMemRefs(MMOBegin, MMOEnd);

  thisMBB->addSuccessor(mainMBB);

  // mainMBB:
  MachineBasicBlock *origMainMBB = mainMBB;
  mainMBB->addLiveIn(X86::EAX);
  mainMBB->addLiveIn(X86::EDX);

  // Copy EDX:EAX as they are used more than once.
  unsigned LoReg = MRI.createVirtualRegister(RC);
  unsigned HiReg = MRI.createVirtualRegister(RC);
  BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), LoReg).addReg(X86::EAX);
  BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), HiReg).addReg(X86::EDX);

  unsigned t1L = MRI.createVirtualRegister(RC);
  unsigned t1H = MRI.createVirtualRegister(RC);

  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default:
    llvm_unreachable("Unhandled atomic-load-op6432 opcode!");
  case X86::ATOMAND6432:
  case X86::ATOMOR6432:
  case X86::ATOMXOR6432:
  case X86::ATOMADD6432:
  case X86::ATOMSUB6432: {
    unsigned HiOpc;
    unsigned LoOpc = getNonAtomic6432Opcode(Opc, HiOpc);
    BuildMI(mainMBB, DL, TII->get(LoOpc), t1L).addReg(SrcLoReg).addReg(LoReg);
    BuildMI(mainMBB, DL, TII->get(HiOpc), t1H).addReg(SrcHiReg).addReg(HiReg);
    break;
  }
  case X86::ATOMNAND6432: {
    unsigned HiOpc, NOTOpc;
    unsigned LoOpc = getNonAtomic6432OpcodeWithExtraOpc(Opc, HiOpc, NOTOpc);
    unsigned t2L = MRI.createVirtualRegister(RC);
    unsigned t2H = MRI.createVirtualRegister(RC);
    BuildMI(mainMBB, DL, TII->get(LoOpc), t2L).addReg(SrcLoReg).addReg(LoReg);
    BuildMI(mainMBB, DL, TII->get(HiOpc), t2H).addReg(SrcHiReg).addReg(HiReg);
    BuildMI(mainMBB, DL, TII->get(NOTOpc), t1L).addReg(t2L);
    BuildMI(mainMBB, DL, TII->get(NOTOpc), t1H).addReg(t2H);
    break;
  }
  case X86::ATOMMAX6432:
  case X86::ATOMMIN6432:
  case X86::ATOMUMAX6432:
  case X86::ATOMUMIN6432: {
    unsigned HiOpc;
    unsigned LoOpc = getNonAtomic6432Opcode(Opc, HiOpc);
    unsigned cL = MRI.createVirtualRegister(RC8);
    unsigned cH = MRI.createVirtualRegister(RC8);
    unsigned cL32 = MRI.createVirtualRegister(RC);
    unsigned cH32 = MRI.createVirtualRegister(RC);
    unsigned cc = MRI.createVirtualRegister(RC);
    // cl := cmp src_lo, lo
    BuildMI(mainMBB, DL, TII->get(X86::CMP32rr))
      .addReg(SrcLoReg).addReg(LoReg);
    BuildMI(mainMBB, DL, TII->get(LoOpc), cL);
    BuildMI(mainMBB, DL, TII->get(X86::MOVZX32rr8), cL32).addReg(cL);
    // ch := cmp src_hi, hi
    BuildMI(mainMBB, DL, TII->get(X86::CMP32rr))
      .addReg(SrcHiReg).addReg(HiReg);
    BuildMI(mainMBB, DL, TII->get(HiOpc), cH);
    BuildMI(mainMBB, DL, TII->get(X86::MOVZX32rr8), cH32).addReg(cH);
    // cc := if (src_hi == hi) ? cl : ch;
    if (Subtarget->hasCMov()) {
      BuildMI(mainMBB, DL, TII->get(X86::CMOVE32rr), cc)
        .addReg(cH32).addReg(cL32);
    } else {
      MIB = BuildMI(mainMBB, DL, TII->get(X86::CMOV_GR32), cc)
              .addReg(cH32).addReg(cL32)
              .addImm(X86::COND_E);
      mainMBB = EmitLoweredSelect(MIB, mainMBB);
    }
    BuildMI(mainMBB, DL, TII->get(X86::TEST32rr)).addReg(cc).addReg(cc);
    if (Subtarget->hasCMov()) {
      BuildMI(mainMBB, DL, TII->get(X86::CMOVNE32rr), t1L)
        .addReg(SrcLoReg).addReg(LoReg);
      BuildMI(mainMBB, DL, TII->get(X86::CMOVNE32rr), t1H)
        .addReg(SrcHiReg).addReg(HiReg);
    } else {
      MIB = BuildMI(mainMBB, DL, TII->get(X86::CMOV_GR32), t1L)
              .addReg(SrcLoReg).addReg(LoReg)
              .addImm(X86::COND_NE);
      mainMBB = EmitLoweredSelect(MIB, mainMBB);
      MIB = BuildMI(mainMBB, DL, TII->get(X86::CMOV_GR32), t1H)
              .addReg(SrcHiReg).addReg(HiReg)
              .addImm(X86::COND_NE);
      mainMBB = EmitLoweredSelect(MIB, mainMBB);
    }
    break;
  }
  case X86::ATOMSWAP6432: {
    unsigned HiOpc;
    unsigned LoOpc = getNonAtomic6432Opcode(Opc, HiOpc);
    BuildMI(mainMBB, DL, TII->get(LoOpc), t1L).addReg(SrcLoReg);
    BuildMI(mainMBB, DL, TII->get(HiOpc), t1H).addReg(SrcHiReg);
    break;
  }
  }

  // Copy EDX:EAX back from HiReg:LoReg
  BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), X86::EAX).addReg(LoReg);
  BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), X86::EDX).addReg(HiReg);
  // Copy ECX:EBX from t1H:t1L
  BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), X86::EBX).addReg(t1L);
  BuildMI(mainMBB, DL, TII->get(TargetOpcode::COPY), X86::ECX).addReg(t1H);

  MIB = BuildMI(mainMBB, DL, TII->get(LCMPXCHGOpc));
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i)
    MIB.addOperand(MI->getOperand(MemOpndSlot + i));
  MIB.setMemRefs(MMOBegin, MMOEnd);

  BuildMI(mainMBB, DL, TII->get(X86::JNE_4)).addMBB(origMainMBB);

  mainMBB->addSuccessor(origMainMBB);
  mainMBB->addSuccessor(sinkMBB);

  // sinkMBB:
  sinkMBB->addLiveIn(X86::EAX);
  sinkMBB->addLiveIn(X86::EDX);

  BuildMI(*sinkMBB, sinkMBB->begin(), DL,
          TII->get(TargetOpcode::COPY), DstLoReg)
    .addReg(X86::EAX);
  BuildMI(*sinkMBB, sinkMBB->begin(), DL,
          TII->get(TargetOpcode::COPY), DstHiReg)
    .addReg(X86::EDX);

  MI->eraseFromParent();
  return sinkMBB;
}

// FIXME: When we get size specific XMM0 registers, i.e. XMM0_V16I8
// or XMM0_V32I8 in AVX all of this code can be replaced with that
// in the .td file.
static MachineBasicBlock *EmitPCMPSTRM(MachineInstr *MI, MachineBasicBlock *BB,
                                       const TargetInstrInfo *TII) {
  unsigned Opc;
  switch (MI->getOpcode()) {
  default: llvm_unreachable("illegal opcode!");
  case X86::PCMPISTRM128REG:  Opc = X86::PCMPISTRM128rr;  break;
  case X86::VPCMPISTRM128REG: Opc = X86::VPCMPISTRM128rr; break;
  case X86::PCMPISTRM128MEM:  Opc = X86::PCMPISTRM128rm;  break;
  case X86::VPCMPISTRM128MEM: Opc = X86::VPCMPISTRM128rm; break;
  case X86::PCMPESTRM128REG:  Opc = X86::PCMPESTRM128rr;  break;
  case X86::VPCMPESTRM128REG: Opc = X86::VPCMPESTRM128rr; break;
  case X86::PCMPESTRM128MEM:  Opc = X86::PCMPESTRM128rm;  break;
  case X86::VPCMPESTRM128MEM: Opc = X86::VPCMPESTRM128rm; break;
  }

  DebugLoc dl = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(*BB, MI, dl, TII->get(Opc));

  unsigned NumArgs = MI->getNumOperands();
  for (unsigned i = 1; i < NumArgs; ++i) {
    MachineOperand &Op = MI->getOperand(i);
    if (!(Op.isReg() && Op.isImplicit()))
      MIB.addOperand(Op);
  }
  if (MI->hasOneMemOperand())
    MIB->setMemRefs(MI->memoperands_begin(), MI->memoperands_end());

  BuildMI(*BB, MI, dl,
    TII->get(TargetOpcode::COPY), MI->getOperand(0).getReg())
    .addReg(X86::XMM0);

  MI->eraseFromParent();
  return BB;
}

// FIXME: Custom handling because TableGen doesn't support multiple implicit
// defs in an instruction pattern
static MachineBasicBlock *EmitPCMPSTRI(MachineInstr *MI, MachineBasicBlock *BB,
                                       const TargetInstrInfo *TII) {
  unsigned Opc;
  switch (MI->getOpcode()) {
  default: llvm_unreachable("illegal opcode!");
  case X86::PCMPISTRIREG:  Opc = X86::PCMPISTRIrr;  break;
  case X86::VPCMPISTRIREG: Opc = X86::VPCMPISTRIrr; break;
  case X86::PCMPISTRIMEM:  Opc = X86::PCMPISTRIrm;  break;
  case X86::VPCMPISTRIMEM: Opc = X86::VPCMPISTRIrm; break;
  case X86::PCMPESTRIREG:  Opc = X86::PCMPESTRIrr;  break;
  case X86::VPCMPESTRIREG: Opc = X86::VPCMPESTRIrr; break;
  case X86::PCMPESTRIMEM:  Opc = X86::PCMPESTRIrm;  break;
  case X86::VPCMPESTRIMEM: Opc = X86::VPCMPESTRIrm; break;
  }

  DebugLoc dl = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(*BB, MI, dl, TII->get(Opc));

  unsigned NumArgs = MI->getNumOperands(); // remove the results
  for (unsigned i = 1; i < NumArgs; ++i) {
    MachineOperand &Op = MI->getOperand(i);
    if (!(Op.isReg() && Op.isImplicit()))
      MIB.addOperand(Op);
  }
  if (MI->hasOneMemOperand())
    MIB->setMemRefs(MI->memoperands_begin(), MI->memoperands_end());

  BuildMI(*BB, MI, dl,
    TII->get(TargetOpcode::COPY), MI->getOperand(0).getReg())
    .addReg(X86::ECX);

  MI->eraseFromParent();
  return BB;
}

static MachineBasicBlock * EmitMonitor(MachineInstr *MI, MachineBasicBlock *BB,
                                       const TargetInstrInfo *TII,
                                       const X86Subtarget* Subtarget) {
  DebugLoc dl = MI->getDebugLoc();

  // Address into RAX/EAX, other two args into ECX, EDX.
  unsigned MemOpc = Subtarget->is64Bit() ? X86::LEA64r : X86::LEA32r;
  unsigned MemReg = Subtarget->is64Bit() ? X86::RAX : X86::EAX;
  MachineInstrBuilder MIB = BuildMI(*BB, MI, dl, TII->get(MemOpc), MemReg);
  for (int i = 0; i < X86::AddrNumOperands; ++i)
    MIB.addOperand(MI->getOperand(i));

  unsigned ValOps = X86::AddrNumOperands;
  BuildMI(*BB, MI, dl, TII->get(TargetOpcode::COPY), X86::ECX)
    .addReg(MI->getOperand(ValOps).getReg());
  BuildMI(*BB, MI, dl, TII->get(TargetOpcode::COPY), X86::EDX)
    .addReg(MI->getOperand(ValOps+1).getReg());

  // The instruction doesn't actually take any operands though.
  BuildMI(*BB, MI, dl, TII->get(X86::MONITORrrr));

  MI->eraseFromParent(); // The pseudo is gone now.
  return BB;
}

MachineBasicBlock *
X86TargetLowering::EmitVAARG64WithCustomInserter(
                   MachineInstr *MI,
                   MachineBasicBlock *MBB) const {
  // Emit va_arg instruction on X86-64.

  // Operands to this pseudo-instruction:
  // 0  ) Output        : destination address (reg)
  // 1-5) Input         : va_list address (addr, i64mem)
  // 6  ) ArgSize       : Size (in bytes) of vararg type
  // 7  ) ArgMode       : 0=overflow only, 1=use gp_offset, 2=use fp_offset
  // 8  ) Align         : Alignment of type
  // 9  ) EFLAGS (implicit-def)

  assert(MI->getNumOperands() == 10 && "VAARG_64 should have 10 operands!");
  assert(X86::AddrNumOperands == 5 && "VAARG_64 assumes 5 address operands");

  unsigned DestReg = MI->getOperand(0).getReg();
  MachineOperand &Base = MI->getOperand(1);
  MachineOperand &Scale = MI->getOperand(2);
  MachineOperand &Index = MI->getOperand(3);
  MachineOperand &Disp = MI->getOperand(4);
  MachineOperand &Segment = MI->getOperand(5);
  unsigned ArgSize = MI->getOperand(6).getImm();
  unsigned ArgMode = MI->getOperand(7).getImm();
  unsigned Align = MI->getOperand(8).getImm();

  // Memory Reference
  assert(MI->hasOneMemOperand() && "Expected VAARG_64 to have one memoperand");
  MachineInstr::mmo_iterator MMOBegin = MI->memoperands_begin();
  MachineInstr::mmo_iterator MMOEnd = MI->memoperands_end();

  // Machine Information
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  const TargetRegisterClass *AddrRegClass = getRegClassFor(MVT::i64);
  const TargetRegisterClass *OffsetRegClass = getRegClassFor(MVT::i32);
  DebugLoc DL = MI->getDebugLoc();

  // struct va_list {
  //   i32   gp_offset
  //   i32   fp_offset
  //   i64   overflow_area (address)
  //   i64   reg_save_area (address)
  // }
  // sizeof(va_list) = 24
  // alignment(va_list) = 8

  unsigned TotalNumIntRegs = 6;
  unsigned TotalNumXMMRegs = 8;
  bool UseGPOffset = (ArgMode == 1);
  bool UseFPOffset = (ArgMode == 2);
  unsigned MaxOffset = TotalNumIntRegs * 8 +
                       (UseFPOffset ? TotalNumXMMRegs * 16 : 0);

  /* Align ArgSize to a multiple of 8 */
  unsigned ArgSizeA8 = (ArgSize + 7) & ~7;
  bool NeedsAlign = (Align > 8);

  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *overflowMBB;
  MachineBasicBlock *offsetMBB;
  MachineBasicBlock *endMBB;

  unsigned OffsetDestReg = 0;    // Argument address computed by offsetMBB
  unsigned OverflowDestReg = 0;  // Argument address computed by overflowMBB
  unsigned OffsetReg = 0;

  if (!UseGPOffset && !UseFPOffset) {
    // If we only pull from the overflow region, we don't create a branch.
    // We don't need to alter control flow.
    OffsetDestReg = 0; // unused
    OverflowDestReg = DestReg;

    offsetMBB = NULL;
    overflowMBB = thisMBB;
    endMBB = thisMBB;
  } else {
    // First emit code to check if gp_offset (or fp_offset) is below the bound.
    // If so, pull the argument from reg_save_area. (branch to offsetMBB)
    // If not, pull from overflow_area. (branch to overflowMBB)
    //
    //       thisMBB
    //         |     .
    //         |        .
    //     offsetMBB   overflowMBB
    //         |        .
    //         |     .
    //        endMBB

    // Registers for the PHI in endMBB
    OffsetDestReg = MRI.createVirtualRegister(AddrRegClass);
    OverflowDestReg = MRI.createVirtualRegister(AddrRegClass);

    const BasicBlock *LLVM_BB = MBB->getBasicBlock();
    MachineFunction *MF = MBB->getParent();
    overflowMBB = MF->CreateMachineBasicBlock(LLVM_BB);
    offsetMBB = MF->CreateMachineBasicBlock(LLVM_BB);
    endMBB = MF->CreateMachineBasicBlock(LLVM_BB);

    MachineFunction::iterator MBBIter = MBB;
    ++MBBIter;

    // Insert the new basic blocks
    MF->insert(MBBIter, offsetMBB);
    MF->insert(MBBIter, overflowMBB);
    MF->insert(MBBIter, endMBB);

    // Transfer the remainder of MBB and its successor edges to endMBB.
    endMBB->splice(endMBB->begin(), thisMBB,
                    llvm::next(MachineBasicBlock::iterator(MI)),
                    thisMBB->end());
    endMBB->transferSuccessorsAndUpdatePHIs(thisMBB);

    // Make offsetMBB and overflowMBB successors of thisMBB
    thisMBB->addSuccessor(offsetMBB);
    thisMBB->addSuccessor(overflowMBB);

    // endMBB is a successor of both offsetMBB and overflowMBB
    offsetMBB->addSuccessor(endMBB);
    overflowMBB->addSuccessor(endMBB);

    // Load the offset value into a register
    OffsetReg = MRI.createVirtualRegister(OffsetRegClass);
    BuildMI(thisMBB, DL, TII->get(X86::MOV32rm), OffsetReg)
      .addOperand(Base)
      .addOperand(Scale)
      .addOperand(Index)
      .addDisp(Disp, UseFPOffset ? 4 : 0)
      .addOperand(Segment)
      .setMemRefs(MMOBegin, MMOEnd);

    // Check if there is enough room left to pull this argument.
    BuildMI(thisMBB, DL, TII->get(X86::CMP32ri))
      .addReg(OffsetReg)
      .addImm(MaxOffset + 8 - ArgSizeA8);

    // Branch to "overflowMBB" if offset >= max
    // Fall through to "offsetMBB" otherwise
    BuildMI(thisMBB, DL, TII->get(X86::GetCondBranchFromCond(X86::COND_AE)))
      .addMBB(overflowMBB);
  }

  // In offsetMBB, emit code to use the reg_save_area.
  if (offsetMBB) {
    assert(OffsetReg != 0);

    // Read the reg_save_area address.
    unsigned RegSaveReg = MRI.createVirtualRegister(AddrRegClass);
    BuildMI(offsetMBB, DL, TII->get(X86::MOV64rm), RegSaveReg)
      .addOperand(Base)
      .addOperand(Scale)
      .addOperand(Index)
      .addDisp(Disp, 16)
      .addOperand(Segment)
      .setMemRefs(MMOBegin, MMOEnd);

    // Zero-extend the offset
    unsigned OffsetReg64 = MRI.createVirtualRegister(AddrRegClass);
      BuildMI(offsetMBB, DL, TII->get(X86::SUBREG_TO_REG), OffsetReg64)
        .addImm(0)
        .addReg(OffsetReg)
        .addImm(X86::sub_32bit);

    // Add the offset to the reg_save_area to get the final address.
    BuildMI(offsetMBB, DL, TII->get(X86::ADD64rr), OffsetDestReg)
      .addReg(OffsetReg64)
      .addReg(RegSaveReg);

    // Compute the offset for the next argument
    unsigned NextOffsetReg = MRI.createVirtualRegister(OffsetRegClass);
    BuildMI(offsetMBB, DL, TII->get(X86::ADD32ri), NextOffsetReg)
      .addReg(OffsetReg)
      .addImm(UseFPOffset ? 16 : 8);

    // Store it back into the va_list.
    BuildMI(offsetMBB, DL, TII->get(X86::MOV32mr))
      .addOperand(Base)
      .addOperand(Scale)
      .addOperand(Index)
      .addDisp(Disp, UseFPOffset ? 4 : 0)
      .addOperand(Segment)
      .addReg(NextOffsetReg)
      .setMemRefs(MMOBegin, MMOEnd);

    // Jump to endMBB
    BuildMI(offsetMBB, DL, TII->get(X86::JMP_4))
      .addMBB(endMBB);
  }

  //
  // Emit code to use overflow area
  //

  // Load the overflow_area address into a register.
  unsigned OverflowAddrReg = MRI.createVirtualRegister(AddrRegClass);
  BuildMI(overflowMBB, DL, TII->get(X86::MOV64rm), OverflowAddrReg)
    .addOperand(Base)
    .addOperand(Scale)
    .addOperand(Index)
    .addDisp(Disp, 8)
    .addOperand(Segment)
    .setMemRefs(MMOBegin, MMOEnd);

  // If we need to align it, do so. Otherwise, just copy the address
  // to OverflowDestReg.
  if (NeedsAlign) {
    // Align the overflow address
    assert((Align & (Align-1)) == 0 && "Alignment must be a power of 2");
    unsigned TmpReg = MRI.createVirtualRegister(AddrRegClass);

    // aligned_addr = (addr + (align-1)) & ~(align-1)
    BuildMI(overflowMBB, DL, TII->get(X86::ADD64ri32), TmpReg)
      .addReg(OverflowAddrReg)
      .addImm(Align-1);

    BuildMI(overflowMBB, DL, TII->get(X86::AND64ri32), OverflowDestReg)
      .addReg(TmpReg)
      .addImm(~(uint64_t)(Align-1));
  } else {
    BuildMI(overflowMBB, DL, TII->get(TargetOpcode::COPY), OverflowDestReg)
      .addReg(OverflowAddrReg);
  }

  // Compute the next overflow address after this argument.
  // (the overflow address should be kept 8-byte aligned)
  unsigned NextAddrReg = MRI.createVirtualRegister(AddrRegClass);
  BuildMI(overflowMBB, DL, TII->get(X86::ADD64ri32), NextAddrReg)
    .addReg(OverflowDestReg)
    .addImm(ArgSizeA8);

  // Store the new overflow address.
  BuildMI(overflowMBB, DL, TII->get(X86::MOV64mr))
    .addOperand(Base)
    .addOperand(Scale)
    .addOperand(Index)
    .addDisp(Disp, 8)
    .addOperand(Segment)
    .addReg(NextAddrReg)
    .setMemRefs(MMOBegin, MMOEnd);

  // If we branched, emit the PHI to the front of endMBB.
  if (offsetMBB) {
    BuildMI(*endMBB, endMBB->begin(), DL,
            TII->get(X86::PHI), DestReg)
      .addReg(OffsetDestReg).addMBB(offsetMBB)
      .addReg(OverflowDestReg).addMBB(overflowMBB);
  }

  // Erase the pseudo instruction
  MI->eraseFromParent();

  return endMBB;
}

MachineBasicBlock *
X86TargetLowering::EmitVAStartSaveXMMRegsWithCustomInserter(
                                                 MachineInstr *MI,
                                                 MachineBasicBlock *MBB) const {
  // Emit code to save XMM registers to the stack. The ABI says that the
  // number of registers to save is given in %al, so it's theoretically
  // possible to do an indirect jump trick to avoid saving all of them,
  // however this code takes a simpler approach and just executes all
  // of the stores if %al is non-zero. It's less code, and it's probably
  // easier on the hardware branch predictor, and stores aren't all that
  // expensive anyway.

  // Create the new basic blocks. One block contains all the XMM stores,
  // and one block is the final destination regardless of whether any
  // stores were performed.
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction *F = MBB->getParent();
  MachineFunction::iterator MBBIter = MBB;
  ++MBBIter;
  MachineBasicBlock *XMMSaveMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *EndMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(MBBIter, XMMSaveMBB);
  F->insert(MBBIter, EndMBB);

  // Transfer the remainder of MBB and its successor edges to EndMBB.
  EndMBB->splice(EndMBB->begin(), MBB,
                 llvm::next(MachineBasicBlock::iterator(MI)),
                 MBB->end());
  EndMBB->transferSuccessorsAndUpdatePHIs(MBB);

  // The original block will now fall through to the XMM save block.
  MBB->addSuccessor(XMMSaveMBB);
  // The XMMSaveMBB will fall through to the end block.
  XMMSaveMBB->addSuccessor(EndMBB);

  // Now add the instructions.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();

  unsigned CountReg = MI->getOperand(0).getReg();
  int64_t RegSaveFrameIndex = MI->getOperand(1).getImm();
  int64_t VarArgsFPOffset = MI->getOperand(2).getImm();

  if (!Subtarget->isTargetWin64()) {
    // If %al is 0, branch around the XMM save block.
    BuildMI(MBB, DL, TII->get(X86::TEST8rr)).addReg(CountReg).addReg(CountReg);
    BuildMI(MBB, DL, TII->get(X86::JE_4)).addMBB(EndMBB);
    MBB->addSuccessor(EndMBB);
  }

  unsigned MOVOpc = Subtarget->hasAVX() ? X86::VMOVAPSmr : X86::MOVAPSmr;
  // In the XMM save block, save all the XMM argument registers.
  for (int i = 3, e = MI->getNumOperands(); i != e; ++i) {
    int64_t Offset = (i - 3) * 16 + VarArgsFPOffset;
    MachineMemOperand *MMO =
      F->getMachineMemOperand(
          MachinePointerInfo::getFixedStack(RegSaveFrameIndex, Offset),
        MachineMemOperand::MOStore,
        /*Size=*/16, /*Align=*/16);
    BuildMI(XMMSaveMBB, DL, TII->get(MOVOpc))
      .addFrameIndex(RegSaveFrameIndex)
      .addImm(/*Scale=*/1)
      .addReg(/*IndexReg=*/0)
      .addImm(/*Disp=*/Offset)
      .addReg(/*Segment=*/0)
      .addReg(MI->getOperand(i).getReg())
      .addMemOperand(MMO);
  }

  MI->eraseFromParent();   // The pseudo instruction is gone now.

  return EndMBB;
}

// The EFLAGS operand of SelectItr might be missing a kill marker
// because there were multiple uses of EFLAGS, and ISel didn't know
// which to mark. Figure out whether SelectItr should have had a
// kill marker, and set it if it should. Returns the correct kill
// marker value.
static bool checkAndUpdateEFLAGSKill(MachineBasicBlock::iterator SelectItr,
                                     MachineBasicBlock* BB,
                                     const TargetRegisterInfo* TRI) {
  // Scan forward through BB for a use/def of EFLAGS.
  MachineBasicBlock::iterator miI(llvm::next(SelectItr));
  for (MachineBasicBlock::iterator miE = BB->end(); miI != miE; ++miI) {
    const MachineInstr& mi = *miI;
    if (mi.readsRegister(X86::EFLAGS))
      return false;
    if (mi.definesRegister(X86::EFLAGS))
      break; // Should have kill-flag - update below.
  }

  // If we hit the end of the block, check whether EFLAGS is live into a
  // successor.
  if (miI == BB->end()) {
    for (MachineBasicBlock::succ_iterator sItr = BB->succ_begin(),
                                          sEnd = BB->succ_end();
         sItr != sEnd; ++sItr) {
      MachineBasicBlock* succ = *sItr;
      if (succ->isLiveIn(X86::EFLAGS))
        return false;
    }
  }

  // We found a def, or hit the end of the basic block and EFLAGS wasn't live
  // out. SelectMI should have a kill flag on EFLAGS.
  SelectItr->addRegisterKilled(X86::EFLAGS, TRI);
  return true;
}

MachineBasicBlock *
X86TargetLowering::EmitLoweredSelect(MachineInstr *MI,
                                     MachineBasicBlock *BB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();

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
  MachineBasicBlock *thisMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(It, copy0MBB);
  F->insert(It, sinkMBB);

  // If the EFLAGS register isn't dead in the terminator, then claim that it's
  // live into the sink and copy blocks.
  const TargetRegisterInfo* TRI = getTargetMachine().getRegisterInfo();
  if (!MI->killsRegister(X86::EFLAGS) &&
      !checkAndUpdateEFLAGSKill(MI, BB, TRI)) {
    copy0MBB->addLiveIn(X86::EFLAGS);
    sinkMBB->addLiveIn(X86::EFLAGS);
  }

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  // Create the conditional branch instruction.
  unsigned Opc =
    X86::GetCondBranchFromCond((X86::CondCode)MI->getOperand(3).getImm());
  BuildMI(BB, DL, TII->get(Opc)).addMBB(sinkMBB);

  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  copy0MBB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BuildMI(*sinkMBB, sinkMBB->begin(), DL,
          TII->get(X86::PHI), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return sinkMBB;
}

MachineBasicBlock *
X86TargetLowering::EmitLoweredSegAlloca(MachineInstr *MI, MachineBasicBlock *BB,
                                        bool Is64Bit) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();
  MachineFunction *MF = BB->getParent();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();

  assert(getTargetMachine().Options.EnableSegmentedStacks);

  unsigned TlsReg = Is64Bit ? X86::FS : X86::GS;
  unsigned TlsOffset = Is64Bit ? 0x70 : 0x30;

  // BB:
  //  ... [Till the alloca]
  // If stacklet is not large enough, jump to mallocMBB
  //
  // bumpMBB:
  //  Allocate by subtracting from RSP
  //  Jump to continueMBB
  //
  // mallocMBB:
  //  Allocate by call to runtime
  //
  // continueMBB:
  //  ...
  //  [rest of original BB]
  //

  MachineBasicBlock *mallocMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *bumpMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *continueMBB = MF->CreateMachineBasicBlock(LLVM_BB);

  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *AddrRegClass =
    getRegClassFor(Is64Bit ? MVT::i64:MVT::i32);

  unsigned mallocPtrVReg = MRI.createVirtualRegister(AddrRegClass),
    bumpSPPtrVReg = MRI.createVirtualRegister(AddrRegClass),
    tmpSPVReg = MRI.createVirtualRegister(AddrRegClass),
    SPLimitVReg = MRI.createVirtualRegister(AddrRegClass),
    sizeVReg = MI->getOperand(1).getReg(),
    physSPReg = Is64Bit ? X86::RSP : X86::ESP;

  MachineFunction::iterator MBBIter = BB;
  ++MBBIter;

  MF->insert(MBBIter, bumpMBB);
  MF->insert(MBBIter, mallocMBB);
  MF->insert(MBBIter, continueMBB);

  continueMBB->splice(continueMBB->begin(), BB, llvm::next
                      (MachineBasicBlock::iterator(MI)), BB->end());
  continueMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Add code to the main basic block to check if the stack limit has been hit,
  // and if so, jump to mallocMBB otherwise to bumpMBB.
  BuildMI(BB, DL, TII->get(TargetOpcode::COPY), tmpSPVReg).addReg(physSPReg);
  BuildMI(BB, DL, TII->get(Is64Bit ? X86::SUB64rr:X86::SUB32rr), SPLimitVReg)
    .addReg(tmpSPVReg).addReg(sizeVReg);
  BuildMI(BB, DL, TII->get(Is64Bit ? X86::CMP64mr:X86::CMP32mr))
    .addReg(0).addImm(1).addReg(0).addImm(TlsOffset).addReg(TlsReg)
    .addReg(SPLimitVReg);
  BuildMI(BB, DL, TII->get(X86::JG_4)).addMBB(mallocMBB);

  // bumpMBB simply decreases the stack pointer, since we know the current
  // stacklet has enough space.
  BuildMI(bumpMBB, DL, TII->get(TargetOpcode::COPY), physSPReg)
    .addReg(SPLimitVReg);
  BuildMI(bumpMBB, DL, TII->get(TargetOpcode::COPY), bumpSPPtrVReg)
    .addReg(SPLimitVReg);
  BuildMI(bumpMBB, DL, TII->get(X86::JMP_4)).addMBB(continueMBB);

  // Calls into a routine in libgcc to allocate more space from the heap.
  const uint32_t *RegMask =
    getTargetMachine().getRegisterInfo()->getCallPreservedMask(CallingConv::C);
  if (Is64Bit) {
    BuildMI(mallocMBB, DL, TII->get(X86::MOV64rr), X86::RDI)
      .addReg(sizeVReg);
    BuildMI(mallocMBB, DL, TII->get(X86::CALL64pcrel32))
      .addExternalSymbol("__morestack_allocate_stack_space")
      .addRegMask(RegMask)
      .addReg(X86::RDI, RegState::Implicit)
      .addReg(X86::RAX, RegState::ImplicitDefine);
  } else {
    BuildMI(mallocMBB, DL, TII->get(X86::SUB32ri), physSPReg).addReg(physSPReg)
      .addImm(12);
    BuildMI(mallocMBB, DL, TII->get(X86::PUSH32r)).addReg(sizeVReg);
    BuildMI(mallocMBB, DL, TII->get(X86::CALLpcrel32))
      .addExternalSymbol("__morestack_allocate_stack_space")
      .addRegMask(RegMask)
      .addReg(X86::EAX, RegState::ImplicitDefine);
  }

  if (!Is64Bit)
    BuildMI(mallocMBB, DL, TII->get(X86::ADD32ri), physSPReg).addReg(physSPReg)
      .addImm(16);

  BuildMI(mallocMBB, DL, TII->get(TargetOpcode::COPY), mallocPtrVReg)
    .addReg(Is64Bit ? X86::RAX : X86::EAX);
  BuildMI(mallocMBB, DL, TII->get(X86::JMP_4)).addMBB(continueMBB);

  // Set up the CFG correctly.
  BB->addSuccessor(bumpMBB);
  BB->addSuccessor(mallocMBB);
  mallocMBB->addSuccessor(continueMBB);
  bumpMBB->addSuccessor(continueMBB);

  // Take care of the PHI nodes.
  BuildMI(*continueMBB, continueMBB->begin(), DL, TII->get(X86::PHI),
          MI->getOperand(0).getReg())
    .addReg(mallocPtrVReg).addMBB(mallocMBB)
    .addReg(bumpSPPtrVReg).addMBB(bumpMBB);

  // Delete the original pseudo instruction.
  MI->eraseFromParent();

  // And we're done.
  return continueMBB;
}

MachineBasicBlock *
X86TargetLowering::EmitLoweredWinAlloca(MachineInstr *MI,
                                          MachineBasicBlock *BB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();

  assert(!Subtarget->isTargetEnvMacho());

  // The lowering is pretty easy: we're just emitting the call to _alloca.  The
  // non-trivial part is impdef of ESP.

  if (Subtarget->isTargetWin64()) {
    if (Subtarget->isTargetCygMing()) {
      // ___chkstk(Mingw64):
      // Clobbers R10, R11, RAX and EFLAGS.
      // Updates RSP.
      BuildMI(*BB, MI, DL, TII->get(X86::W64ALLOCA))
        .addExternalSymbol("___chkstk")
        .addReg(X86::RAX, RegState::Implicit)
        .addReg(X86::RSP, RegState::Implicit)
        .addReg(X86::RAX, RegState::Define | RegState::Implicit)
        .addReg(X86::RSP, RegState::Define | RegState::Implicit)
        .addReg(X86::EFLAGS, RegState::Define | RegState::Implicit);
    } else {
      // __chkstk(MSVCRT): does not update stack pointer.
      // Clobbers R10, R11 and EFLAGS.
      // FIXME: RAX(allocated size) might be reused and not killed.
      BuildMI(*BB, MI, DL, TII->get(X86::W64ALLOCA))
        .addExternalSymbol("__chkstk")
        .addReg(X86::RAX, RegState::Implicit)
        .addReg(X86::EFLAGS, RegState::Define | RegState::Implicit);
      // RAX has the offset to subtracted from RSP.
      BuildMI(*BB, MI, DL, TII->get(X86::SUB64rr), X86::RSP)
        .addReg(X86::RSP)
        .addReg(X86::RAX);
    }
  } else {
    const char *StackProbeSymbol =
      Subtarget->isTargetWindows() ? "_chkstk" : "_alloca";

    BuildMI(*BB, MI, DL, TII->get(X86::CALLpcrel32))
      .addExternalSymbol(StackProbeSymbol)
      .addReg(X86::EAX, RegState::Implicit)
      .addReg(X86::ESP, RegState::Implicit)
      .addReg(X86::EAX, RegState::Define | RegState::Implicit)
      .addReg(X86::ESP, RegState::Define | RegState::Implicit)
      .addReg(X86::EFLAGS, RegState::Define | RegState::Implicit);
  }

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return BB;
}

MachineBasicBlock *
X86TargetLowering::EmitLoweredTLSCall(MachineInstr *MI,
                                      MachineBasicBlock *BB) const {
  // This is pretty easy.  We're taking the value that we received from
  // our load from the relocation, sticking it in either RDI (x86-64)
  // or EAX and doing an indirect call.  The return value will then
  // be in the normal return register.
  const X86InstrInfo *TII
    = static_cast<const X86InstrInfo*>(getTargetMachine().getInstrInfo());
  DebugLoc DL = MI->getDebugLoc();
  MachineFunction *F = BB->getParent();

  assert(Subtarget->isTargetDarwin() && "Darwin only instr emitted?");
  assert(MI->getOperand(3).isGlobal() && "This should be a global");

  // Get a register mask for the lowered call.
  // FIXME: The 32-bit calls have non-standard calling conventions. Use a
  // proper register mask.
  const uint32_t *RegMask =
    getTargetMachine().getRegisterInfo()->getCallPreservedMask(CallingConv::C);
  if (Subtarget->is64Bit()) {
    MachineInstrBuilder MIB = BuildMI(*BB, MI, DL,
                                      TII->get(X86::MOV64rm), X86::RDI)
    .addReg(X86::RIP)
    .addImm(0).addReg(0)
    .addGlobalAddress(MI->getOperand(3).getGlobal(), 0,
                      MI->getOperand(3).getTargetFlags())
    .addReg(0);
    MIB = BuildMI(*BB, MI, DL, TII->get(X86::CALL64m));
    addDirectMem(MIB, X86::RDI);
    MIB.addReg(X86::RAX, RegState::ImplicitDefine).addRegMask(RegMask);
  } else if (getTargetMachine().getRelocationModel() != Reloc::PIC_) {
    MachineInstrBuilder MIB = BuildMI(*BB, MI, DL,
                                      TII->get(X86::MOV32rm), X86::EAX)
    .addReg(0)
    .addImm(0).addReg(0)
    .addGlobalAddress(MI->getOperand(3).getGlobal(), 0,
                      MI->getOperand(3).getTargetFlags())
    .addReg(0);
    MIB = BuildMI(*BB, MI, DL, TII->get(X86::CALL32m));
    addDirectMem(MIB, X86::EAX);
    MIB.addReg(X86::EAX, RegState::ImplicitDefine).addRegMask(RegMask);
  } else {
    MachineInstrBuilder MIB = BuildMI(*BB, MI, DL,
                                      TII->get(X86::MOV32rm), X86::EAX)
    .addReg(TII->getGlobalBaseReg(F))
    .addImm(0).addReg(0)
    .addGlobalAddress(MI->getOperand(3).getGlobal(), 0,
                      MI->getOperand(3).getTargetFlags())
    .addReg(0);
    MIB = BuildMI(*BB, MI, DL, TII->get(X86::CALL32m));
    addDirectMem(MIB, X86::EAX);
    MIB.addReg(X86::EAX, RegState::ImplicitDefine).addRegMask(RegMask);
  }

  MI->eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

MachineBasicBlock *
X86TargetLowering::emitEHSjLjSetJmp(MachineInstr *MI,
                                    MachineBasicBlock *MBB) const {
  DebugLoc DL = MI->getDebugLoc();
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  const BasicBlock *BB = MBB->getBasicBlock();
  MachineFunction::iterator I = MBB;
  ++I;

  // Memory Reference
  MachineInstr::mmo_iterator MMOBegin = MI->memoperands_begin();
  MachineInstr::mmo_iterator MMOEnd = MI->memoperands_end();

  unsigned DstReg;
  unsigned MemOpndSlot = 0;

  unsigned CurOp = 0;

  DstReg = MI->getOperand(CurOp++).getReg();
  const TargetRegisterClass *RC = MRI.getRegClass(DstReg);
  assert(RC->hasType(MVT::i32) && "Invalid destination!");
  unsigned mainDstReg = MRI.createVirtualRegister(RC);
  unsigned restoreDstReg = MRI.createVirtualRegister(RC);

  MemOpndSlot = CurOp;

  MVT PVT = getPointerTy();
  assert((PVT == MVT::i64 || PVT == MVT::i32) &&
         "Invalid Pointer Size!");

  // For v = setjmp(buf), we generate
  //
  // thisMBB:
  //  buf[LabelOffset] = restoreMBB
  //  SjLjSetup restoreMBB
  //
  // mainMBB:
  //  v_main = 0
  //
  // sinkMBB:
  //  v = phi(main, restore)
  //
  // restoreMBB:
  //  v_restore = 1

  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *mainMBB = MF->CreateMachineBasicBlock(BB);
  MachineBasicBlock *sinkMBB = MF->CreateMachineBasicBlock(BB);
  MachineBasicBlock *restoreMBB = MF->CreateMachineBasicBlock(BB);
  MF->insert(I, mainMBB);
  MF->insert(I, sinkMBB);
  MF->push_back(restoreMBB);

  MachineInstrBuilder MIB;

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), MBB,
                  llvm::next(MachineBasicBlock::iterator(MI)), MBB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  // thisMBB:
  unsigned PtrStoreOpc = 0;
  unsigned LabelReg = 0;
  const int64_t LabelOffset = 1 * PVT.getStoreSize();
  Reloc::Model RM = getTargetMachine().getRelocationModel();
  bool UseImmLabel = (getTargetMachine().getCodeModel() == CodeModel::Small) &&
                     (RM == Reloc::Static || RM == Reloc::DynamicNoPIC);

  // Prepare IP either in reg or imm.
  if (!UseImmLabel) {
    PtrStoreOpc = (PVT == MVT::i64) ? X86::MOV64mr : X86::MOV32mr;
    const TargetRegisterClass *PtrRC = getRegClassFor(PVT);
    LabelReg = MRI.createVirtualRegister(PtrRC);
    if (Subtarget->is64Bit()) {
      MIB = BuildMI(*thisMBB, MI, DL, TII->get(X86::LEA64r), LabelReg)
              .addReg(X86::RIP)
              .addImm(0)
              .addReg(0)
              .addMBB(restoreMBB)
              .addReg(0);
    } else {
      const X86InstrInfo *XII = static_cast<const X86InstrInfo*>(TII);
      MIB = BuildMI(*thisMBB, MI, DL, TII->get(X86::LEA32r), LabelReg)
              .addReg(XII->getGlobalBaseReg(MF))
              .addImm(0)
              .addReg(0)
              .addMBB(restoreMBB, Subtarget->ClassifyBlockAddressReference())
              .addReg(0);
    }
  } else
    PtrStoreOpc = (PVT == MVT::i64) ? X86::MOV64mi32 : X86::MOV32mi;
  // Store IP
  MIB = BuildMI(*thisMBB, MI, DL, TII->get(PtrStoreOpc));
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i) {
    if (i == X86::AddrDisp)
      MIB.addDisp(MI->getOperand(MemOpndSlot + i), LabelOffset);
    else
      MIB.addOperand(MI->getOperand(MemOpndSlot + i));
  }
  if (!UseImmLabel)
    MIB.addReg(LabelReg);
  else
    MIB.addMBB(restoreMBB);
  MIB.setMemRefs(MMOBegin, MMOEnd);
  // Setup
  MIB = BuildMI(*thisMBB, MI, DL, TII->get(X86::EH_SjLj_Setup))
          .addMBB(restoreMBB);
  MIB.addRegMask(RegInfo->getNoPreservedMask());
  thisMBB->addSuccessor(mainMBB);
  thisMBB->addSuccessor(restoreMBB);

  // mainMBB:
  //  EAX = 0
  BuildMI(mainMBB, DL, TII->get(X86::MOV32r0), mainDstReg);
  mainMBB->addSuccessor(sinkMBB);

  // sinkMBB:
  BuildMI(*sinkMBB, sinkMBB->begin(), DL,
          TII->get(X86::PHI), DstReg)
    .addReg(mainDstReg).addMBB(mainMBB)
    .addReg(restoreDstReg).addMBB(restoreMBB);

  // restoreMBB:
  BuildMI(restoreMBB, DL, TII->get(X86::MOV32ri), restoreDstReg).addImm(1);
  BuildMI(restoreMBB, DL, TII->get(X86::JMP_4)).addMBB(sinkMBB);
  restoreMBB->addSuccessor(sinkMBB);

  MI->eraseFromParent();
  return sinkMBB;
}

MachineBasicBlock *
X86TargetLowering::emitEHSjLjLongJmp(MachineInstr *MI,
                                     MachineBasicBlock *MBB) const {
  DebugLoc DL = MI->getDebugLoc();
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  // Memory Reference
  MachineInstr::mmo_iterator MMOBegin = MI->memoperands_begin();
  MachineInstr::mmo_iterator MMOEnd = MI->memoperands_end();

  MVT PVT = getPointerTy();
  assert((PVT == MVT::i64 || PVT == MVT::i32) &&
         "Invalid Pointer Size!");

  const TargetRegisterClass *RC =
    (PVT == MVT::i64) ? &X86::GR64RegClass : &X86::GR32RegClass;
  unsigned Tmp = MRI.createVirtualRegister(RC);
  // Since FP is only updated here but NOT referenced, it's treated as GPR.
  unsigned FP = (PVT == MVT::i64) ? X86::RBP : X86::EBP;
  unsigned SP = RegInfo->getStackRegister();

  MachineInstrBuilder MIB;

  const int64_t LabelOffset = 1 * PVT.getStoreSize();
  const int64_t SPOffset = 2 * PVT.getStoreSize();

  unsigned PtrLoadOpc = (PVT == MVT::i64) ? X86::MOV64rm : X86::MOV32rm;
  unsigned IJmpOpc = (PVT == MVT::i64) ? X86::JMP64r : X86::JMP32r;

  // Reload FP
  MIB = BuildMI(*MBB, MI, DL, TII->get(PtrLoadOpc), FP);
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i)
    MIB.addOperand(MI->getOperand(i));
  MIB.setMemRefs(MMOBegin, MMOEnd);
  // Reload IP
  MIB = BuildMI(*MBB, MI, DL, TII->get(PtrLoadOpc), Tmp);
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i) {
    if (i == X86::AddrDisp)
      MIB.addDisp(MI->getOperand(i), LabelOffset);
    else
      MIB.addOperand(MI->getOperand(i));
  }
  MIB.setMemRefs(MMOBegin, MMOEnd);
  // Reload SP
  MIB = BuildMI(*MBB, MI, DL, TII->get(PtrLoadOpc), SP);
  for (unsigned i = 0; i < X86::AddrNumOperands; ++i) {
    if (i == X86::AddrDisp)
      MIB.addDisp(MI->getOperand(i), SPOffset);
    else
      MIB.addOperand(MI->getOperand(i));
  }
  MIB.setMemRefs(MMOBegin, MMOEnd);
  // Jump
  BuildMI(*MBB, MI, DL, TII->get(IJmpOpc)).addReg(Tmp);

  MI->eraseFromParent();
  return MBB;
}

MachineBasicBlock *
X86TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                               MachineBasicBlock *BB) const {
  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unexpected instr type to insert");
  case X86::TAILJMPd64:
  case X86::TAILJMPr64:
  case X86::TAILJMPm64:
    llvm_unreachable("TAILJMP64 would not be touched here.");
  case X86::TCRETURNdi64:
  case X86::TCRETURNri64:
  case X86::TCRETURNmi64:
    return BB;
  case X86::WIN_ALLOCA:
    return EmitLoweredWinAlloca(MI, BB);
  case X86::SEG_ALLOCA_32:
    return EmitLoweredSegAlloca(MI, BB, false);
  case X86::SEG_ALLOCA_64:
    return EmitLoweredSegAlloca(MI, BB, true);
  case X86::TLSCall_32:
  case X86::TLSCall_64:
    return EmitLoweredTLSCall(MI, BB);
  case X86::CMOV_GR8:
  case X86::CMOV_FR32:
  case X86::CMOV_FR64:
  case X86::CMOV_V4F32:
  case X86::CMOV_V2F64:
  case X86::CMOV_V2I64:
  case X86::CMOV_V8F32:
  case X86::CMOV_V4F64:
  case X86::CMOV_V4I64:
  case X86::CMOV_GR16:
  case X86::CMOV_GR32:
  case X86::CMOV_RFP32:
  case X86::CMOV_RFP64:
  case X86::CMOV_RFP80:
    return EmitLoweredSelect(MI, BB);

  case X86::FP32_TO_INT16_IN_MEM:
  case X86::FP32_TO_INT32_IN_MEM:
  case X86::FP32_TO_INT64_IN_MEM:
  case X86::FP64_TO_INT16_IN_MEM:
  case X86::FP64_TO_INT32_IN_MEM:
  case X86::FP64_TO_INT64_IN_MEM:
  case X86::FP80_TO_INT16_IN_MEM:
  case X86::FP80_TO_INT32_IN_MEM:
  case X86::FP80_TO_INT64_IN_MEM: {
    const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
    DebugLoc DL = MI->getDebugLoc();

    // Change the floating point control register to use "round towards zero"
    // mode when truncating to an integer value.
    MachineFunction *F = BB->getParent();
    int CWFrameIdx = F->getFrameInfo()->CreateStackObject(2, 2, false);
    addFrameReference(BuildMI(*BB, MI, DL,
                              TII->get(X86::FNSTCW16m)), CWFrameIdx);

    // Load the old value of the high byte of the control word...
    unsigned OldCW =
      F->getRegInfo().createVirtualRegister(&X86::GR16RegClass);
    addFrameReference(BuildMI(*BB, MI, DL, TII->get(X86::MOV16rm), OldCW),
                      CWFrameIdx);

    // Set the high part to be round to zero...
    addFrameReference(BuildMI(*BB, MI, DL, TII->get(X86::MOV16mi)), CWFrameIdx)
      .addImm(0xC7F);

    // Reload the modified control word now...
    addFrameReference(BuildMI(*BB, MI, DL,
                              TII->get(X86::FLDCW16m)), CWFrameIdx);

    // Restore the memory image of control word to original value
    addFrameReference(BuildMI(*BB, MI, DL, TII->get(X86::MOV16mr)), CWFrameIdx)
      .addReg(OldCW);

    // Get the X86 opcode to use.
    unsigned Opc;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("illegal opcode!");
    case X86::FP32_TO_INT16_IN_MEM: Opc = X86::IST_Fp16m32; break;
    case X86::FP32_TO_INT32_IN_MEM: Opc = X86::IST_Fp32m32; break;
    case X86::FP32_TO_INT64_IN_MEM: Opc = X86::IST_Fp64m32; break;
    case X86::FP64_TO_INT16_IN_MEM: Opc = X86::IST_Fp16m64; break;
    case X86::FP64_TO_INT32_IN_MEM: Opc = X86::IST_Fp32m64; break;
    case X86::FP64_TO_INT64_IN_MEM: Opc = X86::IST_Fp64m64; break;
    case X86::FP80_TO_INT16_IN_MEM: Opc = X86::IST_Fp16m80; break;
    case X86::FP80_TO_INT32_IN_MEM: Opc = X86::IST_Fp32m80; break;
    case X86::FP80_TO_INT64_IN_MEM: Opc = X86::IST_Fp64m80; break;
    }

    X86AddressMode AM;
    MachineOperand &Op = MI->getOperand(0);
    if (Op.isReg()) {
      AM.BaseType = X86AddressMode::RegBase;
      AM.Base.Reg = Op.getReg();
    } else {
      AM.BaseType = X86AddressMode::FrameIndexBase;
      AM.Base.FrameIndex = Op.getIndex();
    }
    Op = MI->getOperand(1);
    if (Op.isImm())
      AM.Scale = Op.getImm();
    Op = MI->getOperand(2);
    if (Op.isImm())
      AM.IndexReg = Op.getImm();
    Op = MI->getOperand(3);
    if (Op.isGlobal()) {
      AM.GV = Op.getGlobal();
    } else {
      AM.Disp = Op.getImm();
    }
    addFullAddress(BuildMI(*BB, MI, DL, TII->get(Opc)), AM)
                      .addReg(MI->getOperand(X86::AddrNumOperands).getReg());

    // Reload the original control word now.
    addFrameReference(BuildMI(*BB, MI, DL,
                              TII->get(X86::FLDCW16m)), CWFrameIdx);

    MI->eraseFromParent();   // The pseudo instruction is gone now.
    return BB;
  }
    // String/text processing lowering.
  case X86::PCMPISTRM128REG:
  case X86::VPCMPISTRM128REG:
  case X86::PCMPISTRM128MEM:
  case X86::VPCMPISTRM128MEM:
  case X86::PCMPESTRM128REG:
  case X86::VPCMPESTRM128REG:
  case X86::PCMPESTRM128MEM:
  case X86::VPCMPESTRM128MEM:
    assert(Subtarget->hasSSE42() &&
           "Target must have SSE4.2 or AVX features enabled");
    return EmitPCMPSTRM(MI, BB, getTargetMachine().getInstrInfo());

  // String/text processing lowering.
  case X86::PCMPISTRIREG:
  case X86::VPCMPISTRIREG:
  case X86::PCMPISTRIMEM:
  case X86::VPCMPISTRIMEM:
  case X86::PCMPESTRIREG:
  case X86::VPCMPESTRIREG:
  case X86::PCMPESTRIMEM:
  case X86::VPCMPESTRIMEM:
    assert(Subtarget->hasSSE42() &&
           "Target must have SSE4.2 or AVX features enabled");
    return EmitPCMPSTRI(MI, BB, getTargetMachine().getInstrInfo());

  // Thread synchronization.
  case X86::MONITOR:
    return EmitMonitor(MI, BB, getTargetMachine().getInstrInfo(), Subtarget);

  // xbegin
  case X86::XBEGIN:
    return EmitXBegin(MI, BB, getTargetMachine().getInstrInfo());

  // Atomic Lowering.
  case X86::ATOMAND8:
  case X86::ATOMAND16:
  case X86::ATOMAND32:
  case X86::ATOMAND64:
    // Fall through
  case X86::ATOMOR8:
  case X86::ATOMOR16:
  case X86::ATOMOR32:
  case X86::ATOMOR64:
    // Fall through
  case X86::ATOMXOR16:
  case X86::ATOMXOR8:
  case X86::ATOMXOR32:
  case X86::ATOMXOR64:
    // Fall through
  case X86::ATOMNAND8:
  case X86::ATOMNAND16:
  case X86::ATOMNAND32:
  case X86::ATOMNAND64:
    // Fall through
  case X86::ATOMMAX8:
  case X86::ATOMMAX16:
  case X86::ATOMMAX32:
  case X86::ATOMMAX64:
    // Fall through
  case X86::ATOMMIN8:
  case X86::ATOMMIN16:
  case X86::ATOMMIN32:
  case X86::ATOMMIN64:
    // Fall through
  case X86::ATOMUMAX8:
  case X86::ATOMUMAX16:
  case X86::ATOMUMAX32:
  case X86::ATOMUMAX64:
    // Fall through
  case X86::ATOMUMIN8:
  case X86::ATOMUMIN16:
  case X86::ATOMUMIN32:
  case X86::ATOMUMIN64:
    return EmitAtomicLoadArith(MI, BB);

  // This group does 64-bit operations on a 32-bit host.
  case X86::ATOMAND6432:
  case X86::ATOMOR6432:
  case X86::ATOMXOR6432:
  case X86::ATOMNAND6432:
  case X86::ATOMADD6432:
  case X86::ATOMSUB6432:
  case X86::ATOMMAX6432:
  case X86::ATOMMIN6432:
  case X86::ATOMUMAX6432:
  case X86::ATOMUMIN6432:
  case X86::ATOMSWAP6432:
    return EmitAtomicLoadArith6432(MI, BB);

  case X86::VASTART_SAVE_XMM_REGS:
    return EmitVAStartSaveXMMRegsWithCustomInserter(MI, BB);

  case X86::VAARG_64:
    return EmitVAARG64WithCustomInserter(MI, BB);

  case X86::EH_SjLj_SetJmp32:
  case X86::EH_SjLj_SetJmp64:
    return emitEHSjLjSetJmp(MI, BB);

  case X86::EH_SjLj_LongJmp32:
  case X86::EH_SjLj_LongJmp64:
    return emitEHSjLjLongJmp(MI, BB);
  }
}

//===----------------------------------------------------------------------===//
//                           X86 Optimization Hooks
//===----------------------------------------------------------------------===//

void X86TargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                       APInt &KnownZero,
                                                       APInt &KnownOne,
                                                       const SelectionDAG &DAG,
                                                       unsigned Depth) const {
  unsigned BitWidth = KnownZero.getBitWidth();
  unsigned Opc = Op.getOpcode();
  assert((Opc >= ISD::BUILTIN_OP_END ||
          Opc == ISD::INTRINSIC_WO_CHAIN ||
          Opc == ISD::INTRINSIC_W_CHAIN ||
          Opc == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");

  KnownZero = KnownOne = APInt(BitWidth, 0);   // Don't know anything.
  switch (Opc) {
  default: break;
  case X86ISD::ADD:
  case X86ISD::SUB:
  case X86ISD::ADC:
  case X86ISD::SBB:
  case X86ISD::SMUL:
  case X86ISD::UMUL:
  case X86ISD::INC:
  case X86ISD::DEC:
  case X86ISD::OR:
  case X86ISD::XOR:
  case X86ISD::AND:
    // These nodes' second result is a boolean.
    if (Op.getResNo() == 0)
      break;
    // Fallthrough
  case X86ISD::SETCC:
    KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - 1);
    break;
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntId = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
    unsigned NumLoBits = 0;
    switch (IntId) {
    default: break;
    case Intrinsic::x86_sse_movmsk_ps:
    case Intrinsic::x86_avx_movmsk_ps_256:
    case Intrinsic::x86_sse2_movmsk_pd:
    case Intrinsic::x86_avx_movmsk_pd_256:
    case Intrinsic::x86_mmx_pmovmskb:
    case Intrinsic::x86_sse2_pmovmskb_128:
    case Intrinsic::x86_avx2_pmovmskb: {
      // High bits of movmskp{s|d}, pmovmskb are known zero.
      switch (IntId) {
        default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
        case Intrinsic::x86_sse_movmsk_ps:      NumLoBits = 4; break;
        case Intrinsic::x86_avx_movmsk_ps_256:  NumLoBits = 8; break;
        case Intrinsic::x86_sse2_movmsk_pd:     NumLoBits = 2; break;
        case Intrinsic::x86_avx_movmsk_pd_256:  NumLoBits = 4; break;
        case Intrinsic::x86_mmx_pmovmskb:       NumLoBits = 8; break;
        case Intrinsic::x86_sse2_pmovmskb_128:  NumLoBits = 16; break;
        case Intrinsic::x86_avx2_pmovmskb:      NumLoBits = 32; break;
      }
      KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - NumLoBits);
      break;
    }
    }
    break;
  }
  }
}

unsigned X86TargetLowering::ComputeNumSignBitsForTargetNode(SDValue Op,
                                                         unsigned Depth) const {
  // SETCC_CARRY sets the dest to ~0 for true or 0 for false.
  if (Op.getOpcode() == X86ISD::SETCC_CARRY)
    return Op.getValueType().getScalarType().getSizeInBits();

  // Fallback case.
  return 1;
}

/// isGAPlusOffset - Returns true (and the GlobalValue and the offset) if the
/// node is a GlobalAddress + offset.
bool X86TargetLowering::isGAPlusOffset(SDNode *N,
                                       const GlobalValue* &GA,
                                       int64_t &Offset) const {
  if (N->getOpcode() == X86ISD::Wrapper) {
    if (isa<GlobalAddressSDNode>(N->getOperand(0))) {
      GA = cast<GlobalAddressSDNode>(N->getOperand(0))->getGlobal();
      Offset = cast<GlobalAddressSDNode>(N->getOperand(0))->getOffset();
      return true;
    }
  }
  return TargetLowering::isGAPlusOffset(N, GA, Offset);
}

/// isShuffleHigh128VectorInsertLow - Checks whether the shuffle node is the
/// same as extracting the high 128-bit part of 256-bit vector and then
/// inserting the result into the low part of a new 256-bit vector
static bool isShuffleHigh128VectorInsertLow(ShuffleVectorSDNode *SVOp) {
  EVT VT = SVOp->getValueType(0);
  unsigned NumElems = VT.getVectorNumElements();

  // vector_shuffle <4, 5, 6, 7, u, u, u, u> or <2, 3, u, u>
  for (unsigned i = 0, j = NumElems/2; i != NumElems/2; ++i, ++j)
    if (!isUndefOrEqual(SVOp->getMaskElt(i), j) ||
        SVOp->getMaskElt(j) >= 0)
      return false;

  return true;
}

/// isShuffleLow128VectorInsertHigh - Checks whether the shuffle node is the
/// same as extracting the low 128-bit part of 256-bit vector and then
/// inserting the result into the high part of a new 256-bit vector
static bool isShuffleLow128VectorInsertHigh(ShuffleVectorSDNode *SVOp) {
  EVT VT = SVOp->getValueType(0);
  unsigned NumElems = VT.getVectorNumElements();

  // vector_shuffle <u, u, u, u, 0, 1, 2, 3> or <u, u, 0, 1>
  for (unsigned i = NumElems/2, j = 0; i != NumElems; ++i, ++j)
    if (!isUndefOrEqual(SVOp->getMaskElt(i), j) ||
        SVOp->getMaskElt(j) >= 0)
      return false;

  return true;
}

/// PerformShuffleCombine256 - Performs shuffle combines for 256-bit vectors.
static SDValue PerformShuffleCombine256(SDNode *N, SelectionDAG &DAG,
                                        TargetLowering::DAGCombinerInfo &DCI,
                                        const X86Subtarget* Subtarget) {
  DebugLoc dl = N->getDebugLoc();
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  EVT VT = SVOp->getValueType(0);
  unsigned NumElems = VT.getVectorNumElements();

  if (V1.getOpcode() == ISD::CONCAT_VECTORS &&
      V2.getOpcode() == ISD::CONCAT_VECTORS) {
    //
    //                   0,0,0,...
    //                      |
    //    V      UNDEF    BUILD_VECTOR    UNDEF
    //     \      /           \           /
    //  CONCAT_VECTOR         CONCAT_VECTOR
    //         \                  /
    //          \                /
    //          RESULT: V + zero extended
    //
    if (V2.getOperand(0).getOpcode() != ISD::BUILD_VECTOR ||
        V2.getOperand(1).getOpcode() != ISD::UNDEF ||
        V1.getOperand(1).getOpcode() != ISD::UNDEF)
      return SDValue();

    if (!ISD::isBuildVectorAllZeros(V2.getOperand(0).getNode()))
      return SDValue();

    // To match the shuffle mask, the first half of the mask should
    // be exactly the first vector, and all the rest a splat with the
    // first element of the second one.
    for (unsigned i = 0; i != NumElems/2; ++i)
      if (!isUndefOrEqual(SVOp->getMaskElt(i), i) ||
          !isUndefOrEqual(SVOp->getMaskElt(i+NumElems/2), NumElems))
        return SDValue();

    // If V1 is coming from a vector load then just fold to a VZEXT_LOAD.
    if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(V1.getOperand(0))) {
      if (Ld->hasNUsesOfValue(1, 0)) {
        SDVTList Tys = DAG.getVTList(MVT::v4i64, MVT::Other);
        SDValue Ops[] = { Ld->getChain(), Ld->getBasePtr() };
        SDValue ResNode =
          DAG.getMemIntrinsicNode(X86ISD::VZEXT_LOAD, dl, Tys, Ops, 2,
                                  Ld->getMemoryVT(),
                                  Ld->getPointerInfo(),
                                  Ld->getAlignment(),
                                  false/*isVolatile*/, true/*ReadMem*/,
                                  false/*WriteMem*/);
        return DAG.getNode(ISD::BITCAST, dl, VT, ResNode);
      }
    }

    // Emit a zeroed vector and insert the desired subvector on its
    // first half.
    SDValue Zeros = getZeroVector(VT, Subtarget, DAG, dl);
    SDValue InsV = Insert128BitVector(Zeros, V1.getOperand(0), 0, DAG, dl);
    return DCI.CombineTo(N, InsV);
  }

  //===--------------------------------------------------------------------===//
  // Combine some shuffles into subvector extracts and inserts:
  //

  // vector_shuffle <4, 5, 6, 7, u, u, u, u> or <2, 3, u, u>
  if (isShuffleHigh128VectorInsertLow(SVOp)) {
    SDValue V = Extract128BitVector(V1, NumElems/2, DAG, dl);
    SDValue InsV = Insert128BitVector(DAG.getUNDEF(VT), V, 0, DAG, dl);
    return DCI.CombineTo(N, InsV);
  }

  // vector_shuffle <u, u, u, u, 0, 1, 2, 3> or <u, u, 0, 1>
  if (isShuffleLow128VectorInsertHigh(SVOp)) {
    SDValue V = Extract128BitVector(V1, 0, DAG, dl);
    SDValue InsV = Insert128BitVector(DAG.getUNDEF(VT), V, NumElems/2, DAG, dl);
    return DCI.CombineTo(N, InsV);
  }

  return SDValue();
}

/// PerformShuffleCombine - Performs several different shuffle combines.
static SDValue PerformShuffleCombine(SDNode *N, SelectionDAG &DAG,
                                     TargetLowering::DAGCombinerInfo &DCI,
                                     const X86Subtarget *Subtarget) {
  DebugLoc dl = N->getDebugLoc();
  EVT VT = N->getValueType(0);

  // Don't create instructions with illegal types after legalize types has run.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!DCI.isBeforeLegalize() && !TLI.isTypeLegal(VT.getVectorElementType()))
    return SDValue();

  // Combine 256-bit vector shuffles. This is only profitable when in AVX mode
  if (Subtarget->hasAVX() && VT.is256BitVector() &&
      N->getOpcode() == ISD::VECTOR_SHUFFLE)
    return PerformShuffleCombine256(N, DAG, DCI, Subtarget);

  // Only handle 128 wide vector from here on.
  if (!VT.is128BitVector())
    return SDValue();

  // Combine a vector_shuffle that is equal to build_vector load1, load2, load3,
  // load4, <0, 1, 2, 3> into a 128-bit load if the load addresses are
  // consecutive, non-overlapping, and in the right order.
  SmallVector<SDValue, 16> Elts;
  for (unsigned i = 0, e = VT.getVectorNumElements(); i != e; ++i)
    Elts.push_back(getShuffleScalarElt(N, i, DAG, 0));

  return EltsFromConsecutiveLoads(VT, Elts, dl, DAG);
}


/// PerformTruncateCombine - Converts truncate operation to
/// a sequence of vector shuffle operations.
/// It is possible when we truncate 256-bit vector to 128-bit vector
static SDValue PerformTruncateCombine(SDNode *N, SelectionDAG &DAG,
                                      TargetLowering::DAGCombinerInfo &DCI,
                                      const X86Subtarget *Subtarget)  {
  if (!DCI.isBeforeLegalizeOps())
    return SDValue();

  if (!Subtarget->hasAVX())
    return SDValue();

  EVT VT = N->getValueType(0);
  SDValue Op = N->getOperand(0);
  EVT OpVT = Op.getValueType();
  DebugLoc dl = N->getDebugLoc();

  if ((VT == MVT::v4i32) && (OpVT == MVT::v4i64)) {

    if (Subtarget->hasAVX2()) {
      // AVX2: v4i64 -> v4i32

      // VPERMD
      static const int ShufMask[] = {0, 2, 4, 6, -1, -1, -1, -1};

      Op = DAG.getNode(ISD::BITCAST, dl, MVT::v8i32, Op);
      Op = DAG.getVectorShuffle(MVT::v8i32, dl, Op, DAG.getUNDEF(MVT::v8i32),
                                ShufMask);

      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT, Op,
                         DAG.getIntPtrConstant(0));
    }

    // AVX: v4i64 -> v4i32
    SDValue OpLo = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v2i64, Op,
                               DAG.getIntPtrConstant(0));

    SDValue OpHi = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v2i64, Op,
                               DAG.getIntPtrConstant(2));

    OpLo = DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, OpLo);
    OpHi = DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, OpHi);

    // PSHUFD
    static const int ShufMask1[] = {0, 2, 0, 0};

    SDValue Undef = DAG.getUNDEF(VT);
    OpLo = DAG.getVectorShuffle(VT, dl, OpLo, Undef, ShufMask1);
    OpHi = DAG.getVectorShuffle(VT, dl, OpHi, Undef, ShufMask1);

    // MOVLHPS
    static const int ShufMask2[] = {0, 1, 4, 5};

    return DAG.getVectorShuffle(VT, dl, OpLo, OpHi, ShufMask2);
  }

  if ((VT == MVT::v8i16) && (OpVT == MVT::v8i32)) {

    if (Subtarget->hasAVX2()) {
      // AVX2: v8i32 -> v8i16

      Op = DAG.getNode(ISD::BITCAST, dl, MVT::v32i8, Op);

      // PSHUFB
      SmallVector<SDValue,32> pshufbMask;
      for (unsigned i = 0; i < 2; ++i) {
        pshufbMask.push_back(DAG.getConstant(0x0, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0x1, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0x4, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0x5, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0x8, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0x9, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0xc, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0xd, MVT::i8));
        for (unsigned j = 0; j < 8; ++j)
          pshufbMask.push_back(DAG.getConstant(0x80, MVT::i8));
      }
      SDValue BV = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v32i8,
                               &pshufbMask[0], 32);
      Op = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v32i8, Op, BV);

      Op = DAG.getNode(ISD::BITCAST, dl, MVT::v4i64, Op);

      static const int ShufMask[] = {0,  2,  -1,  -1};
      Op = DAG.getVectorShuffle(MVT::v4i64, dl,  Op, DAG.getUNDEF(MVT::v4i64),
                                &ShufMask[0]);

      Op = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v2i64, Op,
                       DAG.getIntPtrConstant(0));

      return DAG.getNode(ISD::BITCAST, dl, VT, Op);
    }

    SDValue OpLo = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i32, Op,
                               DAG.getIntPtrConstant(0));

    SDValue OpHi = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, MVT::v4i32, Op,
                               DAG.getIntPtrConstant(4));

    OpLo = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, OpLo);
    OpHi = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, OpHi);

    // PSHUFB
    static const int ShufMask1[] = {0,  1,  4,  5,  8,  9, 12, 13,
                                   -1, -1, -1, -1, -1, -1, -1, -1};

    SDValue Undef = DAG.getUNDEF(MVT::v16i8);
    OpLo = DAG.getVectorShuffle(MVT::v16i8, dl, OpLo, Undef, ShufMask1);
    OpHi = DAG.getVectorShuffle(MVT::v16i8, dl, OpHi, Undef, ShufMask1);

    OpLo = DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, OpLo);
    OpHi = DAG.getNode(ISD::BITCAST, dl, MVT::v4i32, OpHi);

    // MOVLHPS
    static const int ShufMask2[] = {0, 1, 4, 5};

    SDValue res = DAG.getVectorShuffle(MVT::v4i32, dl, OpLo, OpHi, ShufMask2);
    return DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, res);
  }

  return SDValue();
}

/// XFormVExtractWithShuffleIntoLoad - Check if a vector extract from a target
/// specific shuffle of a load can be folded into a single element load.
/// Similar handling for VECTOR_SHUFFLE is performed by DAGCombiner, but
/// shuffles have been customed lowered so we need to handle those here.
static SDValue XFormVExtractWithShuffleIntoLoad(SDNode *N, SelectionDAG &DAG,
                                         TargetLowering::DAGCombinerInfo &DCI) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SDValue InVec = N->getOperand(0);
  SDValue EltNo = N->getOperand(1);

  if (!isa<ConstantSDNode>(EltNo))
    return SDValue();

  EVT VT = InVec.getValueType();

  bool HasShuffleIntoBitcast = false;
  if (InVec.getOpcode() == ISD::BITCAST) {
    // Don't duplicate a load with other uses.
    if (!InVec.hasOneUse())
      return SDValue();
    EVT BCVT = InVec.getOperand(0).getValueType();
    if (BCVT.getVectorNumElements() != VT.getVectorNumElements())
      return SDValue();
    InVec = InVec.getOperand(0);
    HasShuffleIntoBitcast = true;
  }

  if (!isTargetShuffle(InVec.getOpcode()))
    return SDValue();

  // Don't duplicate a load with other uses.
  if (!InVec.hasOneUse())
    return SDValue();

  SmallVector<int, 16> ShuffleMask;
  bool UnaryShuffle;
  if (!getTargetShuffleMask(InVec.getNode(), VT.getSimpleVT(), ShuffleMask,
                            UnaryShuffle))
    return SDValue();

  // Select the input vector, guarding against out of range extract vector.
  unsigned NumElems = VT.getVectorNumElements();
  int Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
  int Idx = (Elt > (int)NumElems) ? -1 : ShuffleMask[Elt];
  SDValue LdNode = (Idx < (int)NumElems) ? InVec.getOperand(0)
                                         : InVec.getOperand(1);

  // If inputs to shuffle are the same for both ops, then allow 2 uses
  unsigned AllowedUses = InVec.getOperand(0) == InVec.getOperand(1) ? 2 : 1;

  if (LdNode.getOpcode() == ISD::BITCAST) {
    // Don't duplicate a load with other uses.
    if (!LdNode.getNode()->hasNUsesOfValue(AllowedUses, 0))
      return SDValue();

    AllowedUses = 1; // only allow 1 load use if we have a bitcast
    LdNode = LdNode.getOperand(0);
  }

  if (!ISD::isNormalLoad(LdNode.getNode()))
    return SDValue();

  LoadSDNode *LN0 = cast<LoadSDNode>(LdNode);

  if (!LN0 ||!LN0->hasNUsesOfValue(AllowedUses, 0) || LN0->isVolatile())
    return SDValue();

  if (HasShuffleIntoBitcast) {
    // If there's a bitcast before the shuffle, check if the load type and
    // alignment is valid.
    unsigned Align = LN0->getAlignment();
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    unsigned NewAlign = TLI.getDataLayout()->
      getABITypeAlignment(VT.getTypeForEVT(*DAG.getContext()));

    if (NewAlign > Align || !TLI.isOperationLegalOrCustom(ISD::LOAD, VT))
      return SDValue();
  }

  // All checks match so transform back to vector_shuffle so that DAG combiner
  // can finish the job
  DebugLoc dl = N->getDebugLoc();

  // Create shuffle node taking into account the case that its a unary shuffle
  SDValue Shuffle = (UnaryShuffle) ? DAG.getUNDEF(VT) : InVec.getOperand(1);
  Shuffle = DAG.getVectorShuffle(InVec.getValueType(), dl,
                                 InVec.getOperand(0), Shuffle,
                                 &ShuffleMask[0]);
  Shuffle = DAG.getNode(ISD::BITCAST, dl, VT, Shuffle);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, N->getValueType(0), Shuffle,
                     EltNo);
}

/// PerformEXTRACT_VECTOR_ELTCombine - Detect vector gather/scatter index
/// generation and convert it from being a bunch of shuffles and extracts
/// to a simple store and scalar loads to extract the elements.
static SDValue PerformEXTRACT_VECTOR_ELTCombine(SDNode *N, SelectionDAG &DAG,
                                         TargetLowering::DAGCombinerInfo &DCI) {
  SDValue NewOp = XFormVExtractWithShuffleIntoLoad(N, DAG, DCI);
  if (NewOp.getNode())
    return NewOp;

  SDValue InputVector = N->getOperand(0);
  // Detect whether we are trying to convert from mmx to i32 and the bitcast
  // from mmx to v2i32 has a single usage.
  if (InputVector.getNode()->getOpcode() == llvm::ISD::BITCAST &&
      InputVector.getNode()->getOperand(0).getValueType() == MVT::x86mmx &&
      InputVector.hasOneUse() && N->getValueType(0) == MVT::i32)
    return DAG.getNode(X86ISD::MMX_MOVD2W, InputVector.getDebugLoc(),
                       N->getValueType(0),
                       InputVector.getNode()->getOperand(0));

  // Only operate on vectors of 4 elements, where the alternative shuffling
  // gets to be more expensive.
  if (InputVector.getValueType() != MVT::v4i32)
    return SDValue();

  // Check whether every use of InputVector is an EXTRACT_VECTOR_ELT with a
  // single use which is a sign-extend or zero-extend, and all elements are
  // used.
  SmallVector<SDNode *, 4> Uses;
  unsigned ExtractedElements = 0;
  for (SDNode::use_iterator UI = InputVector.getNode()->use_begin(),
       UE = InputVector.getNode()->use_end(); UI != UE; ++UI) {
    if (UI.getUse().getResNo() != InputVector.getResNo())
      return SDValue();

    SDNode *Extract = *UI;
    if (Extract->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return SDValue();

    if (Extract->getValueType(0) != MVT::i32)
      return SDValue();
    if (!Extract->hasOneUse())
      return SDValue();
    if (Extract->use_begin()->getOpcode() != ISD::SIGN_EXTEND &&
        Extract->use_begin()->getOpcode() != ISD::ZERO_EXTEND)
      return SDValue();
    if (!isa<ConstantSDNode>(Extract->getOperand(1)))
      return SDValue();

    // Record which element was extracted.
    ExtractedElements |=
      1 << cast<ConstantSDNode>(Extract->getOperand(1))->getZExtValue();

    Uses.push_back(Extract);
  }

  // If not all the elements were used, this may not be worthwhile.
  if (ExtractedElements != 15)
    return SDValue();

  // Ok, we've now decided to do the transformation.
  DebugLoc dl = InputVector.getDebugLoc();

  // Store the value to a temporary stack slot.
  SDValue StackPtr = DAG.CreateStackTemporary(InputVector.getValueType());
  SDValue Ch = DAG.getStore(DAG.getEntryNode(), dl, InputVector, StackPtr,
                            MachinePointerInfo(), false, false, 0);

  // Replace each use (extract) with a load of the appropriate element.
  for (SmallVectorImpl<SDNode *>::iterator UI = Uses.begin(),
       UE = Uses.end(); UI != UE; ++UI) {
    SDNode *Extract = *UI;

    // cOMpute the element's address.
    SDValue Idx = Extract->getOperand(1);
    unsigned EltSize =
        InputVector.getValueType().getVectorElementType().getSizeInBits()/8;
    uint64_t Offset = EltSize * cast<ConstantSDNode>(Idx)->getZExtValue();
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    SDValue OffsetVal = DAG.getConstant(Offset, TLI.getPointerTy());

    SDValue ScalarAddr = DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(),
                                     StackPtr, OffsetVal);

    // Load the scalar.
    SDValue LoadScalar = DAG.getLoad(Extract->getValueType(0), dl, Ch,
                                     ScalarAddr, MachinePointerInfo(),
                                     false, false, false, 0);

    // Replace the exact with the load.
    DAG.ReplaceAllUsesOfValueWith(SDValue(Extract, 0), LoadScalar);
  }

  // The replacement was made in place; don't return anything.
  return SDValue();
}

/// PerformSELECTCombine - Do target-specific dag combines on SELECT and VSELECT
/// nodes.
static SDValue PerformSELECTCombine(SDNode *N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const X86Subtarget *Subtarget) {
  DebugLoc DL = N->getDebugLoc();
  SDValue Cond = N->getOperand(0);
  // Get the LHS/RHS of the select.
  SDValue LHS = N->getOperand(1);
  SDValue RHS = N->getOperand(2);
  EVT VT = LHS.getValueType();

  // If we have SSE[12] support, try to form min/max nodes. SSE min/max
  // instructions match the semantics of the common C idiom x<y?x:y but not
  // x<=y?x:y, because of how they handle negative zero (which can be
  // ignored in unsafe-math mode).
  if (Cond.getOpcode() == ISD::SETCC && VT.isFloatingPoint() &&
      VT != MVT::f80 && DAG.getTargetLoweringInfo().isTypeLegal(VT) &&
      (Subtarget->hasSSE2() ||
       (Subtarget->hasSSE1() && VT.getScalarType() == MVT::f32))) {
    ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();

    unsigned Opcode = 0;
    // Check for x CC y ? x : y.
    if (DAG.isEqualTo(LHS, Cond.getOperand(0)) &&
        DAG.isEqualTo(RHS, Cond.getOperand(1))) {
      switch (CC) {
      default: break;
      case ISD::SETULT:
        // Converting this to a min would handle NaNs incorrectly, and swapping
        // the operands would cause it to handle comparisons between positive
        // and negative zero incorrectly.
        if (!DAG.isKnownNeverNaN(LHS) || !DAG.isKnownNeverNaN(RHS)) {
          if (!DAG.getTarget().Options.UnsafeFPMath &&
              !(DAG.isKnownNeverZero(LHS) || DAG.isKnownNeverZero(RHS)))
            break;
          std::swap(LHS, RHS);
        }
        Opcode = X86ISD::FMIN;
        break;
      case ISD::SETOLE:
        // Converting this to a min would handle comparisons between positive
        // and negative zero incorrectly.
        if (!DAG.getTarget().Options.UnsafeFPMath &&
            !DAG.isKnownNeverZero(LHS) && !DAG.isKnownNeverZero(RHS))
          break;
        Opcode = X86ISD::FMIN;
        break;
      case ISD::SETULE:
        // Converting this to a min would handle both negative zeros and NaNs
        // incorrectly, but we can swap the operands to fix both.
        std::swap(LHS, RHS);
      case ISD::SETOLT:
      case ISD::SETLT:
      case ISD::SETLE:
        Opcode = X86ISD::FMIN;
        break;

      case ISD::SETOGE:
        // Converting this to a max would handle comparisons between positive
        // and negative zero incorrectly.
        if (!DAG.getTarget().Options.UnsafeFPMath &&
            !DAG.isKnownNeverZero(LHS) && !DAG.isKnownNeverZero(RHS))
          break;
        Opcode = X86ISD::FMAX;
        break;
      case ISD::SETUGT:
        // Converting this to a max would handle NaNs incorrectly, and swapping
        // the operands would cause it to handle comparisons between positive
        // and negative zero incorrectly.
        if (!DAG.isKnownNeverNaN(LHS) || !DAG.isKnownNeverNaN(RHS)) {
          if (!DAG.getTarget().Options.UnsafeFPMath &&
              !(DAG.isKnownNeverZero(LHS) || DAG.isKnownNeverZero(RHS)))
            break;
          std::swap(LHS, RHS);
        }
        Opcode = X86ISD::FMAX;
        break;
      case ISD::SETUGE:
        // Converting this to a max would handle both negative zeros and NaNs
        // incorrectly, but we can swap the operands to fix both.
        std::swap(LHS, RHS);
      case ISD::SETOGT:
      case ISD::SETGT:
      case ISD::SETGE:
        Opcode = X86ISD::FMAX;
        break;
      }
    // Check for x CC y ? y : x -- a min/max with reversed arms.
    } else if (DAG.isEqualTo(LHS, Cond.getOperand(1)) &&
               DAG.isEqualTo(RHS, Cond.getOperand(0))) {
      switch (CC) {
      default: break;
      case ISD::SETOGE:
        // Converting this to a min would handle comparisons between positive
        // and negative zero incorrectly, and swapping the operands would
        // cause it to handle NaNs incorrectly.
        if (!DAG.getTarget().Options.UnsafeFPMath &&
            !(DAG.isKnownNeverZero(LHS) || DAG.isKnownNeverZero(RHS))) {
          if (!DAG.isKnownNeverNaN(LHS) || !DAG.isKnownNeverNaN(RHS))
            break;
          std::swap(LHS, RHS);
        }
        Opcode = X86ISD::FMIN;
        break;
      case ISD::SETUGT:
        // Converting this to a min would handle NaNs incorrectly.
        if (!DAG.getTarget().Options.UnsafeFPMath &&
            (!DAG.isKnownNeverNaN(LHS) || !DAG.isKnownNeverNaN(RHS)))
          break;
        Opcode = X86ISD::FMIN;
        break;
      case ISD::SETUGE:
        // Converting this to a min would handle both negative zeros and NaNs
        // incorrectly, but we can swap the operands to fix both.
        std::swap(LHS, RHS);
      case ISD::SETOGT:
      case ISD::SETGT:
      case ISD::SETGE:
        Opcode = X86ISD::FMIN;
        break;

      case ISD::SETULT:
        // Converting this to a max would handle NaNs incorrectly.
        if (!DAG.isKnownNeverNaN(LHS) || !DAG.isKnownNeverNaN(RHS))
          break;
        Opcode = X86ISD::FMAX;
        break;
      case ISD::SETOLE:
        // Converting this to a max would handle comparisons between positive
        // and negative zero incorrectly, and swapping the operands would
        // cause it to handle NaNs incorrectly.
        if (!DAG.getTarget().Options.UnsafeFPMath &&
            !DAG.isKnownNeverZero(LHS) && !DAG.isKnownNeverZero(RHS)) {
          if (!DAG.isKnownNeverNaN(LHS) || !DAG.isKnownNeverNaN(RHS))
            break;
          std::swap(LHS, RHS);
        }
        Opcode = X86ISD::FMAX;
        break;
      case ISD::SETULE:
        // Converting this to a max would handle both negative zeros and NaNs
        // incorrectly, but we can swap the operands to fix both.
        std::swap(LHS, RHS);
      case ISD::SETOLT:
      case ISD::SETLT:
      case ISD::SETLE:
        Opcode = X86ISD::FMAX;
        break;
      }
    }

    if (Opcode)
      return DAG.getNode(Opcode, DL, N->getValueType(0), LHS, RHS);
  }

  // If this is a select between two integer constants, try to do some
  // optimizations.
  if (ConstantSDNode *TrueC = dyn_cast<ConstantSDNode>(LHS)) {
    if (ConstantSDNode *FalseC = dyn_cast<ConstantSDNode>(RHS))
      // Don't do this for crazy integer types.
      if (DAG.getTargetLoweringInfo().isTypeLegal(LHS.getValueType())) {
        // If this is efficiently invertible, canonicalize the LHSC/RHSC values
        // so that TrueC (the true value) is larger than FalseC.
        bool NeedsCondInvert = false;

        if (TrueC->getAPIntValue().ult(FalseC->getAPIntValue()) &&
            // Efficiently invertible.
            (Cond.getOpcode() == ISD::SETCC ||  // setcc -> invertible.
             (Cond.getOpcode() == ISD::XOR &&   // xor(X, C) -> invertible.
              isa<ConstantSDNode>(Cond.getOperand(1))))) {
          NeedsCondInvert = true;
          std::swap(TrueC, FalseC);
        }

        // Optimize C ? 8 : 0 -> zext(C) << 3.  Likewise for any pow2/0.
        if (FalseC->getAPIntValue() == 0 &&
            TrueC->getAPIntValue().isPowerOf2()) {
          if (NeedsCondInvert) // Invert the condition if needed.
            Cond = DAG.getNode(ISD::XOR, DL, Cond.getValueType(), Cond,
                               DAG.getConstant(1, Cond.getValueType()));

          // Zero extend the condition if needed.
          Cond = DAG.getNode(ISD::ZERO_EXTEND, DL, LHS.getValueType(), Cond);

          unsigned ShAmt = TrueC->getAPIntValue().logBase2();
          return DAG.getNode(ISD::SHL, DL, LHS.getValueType(), Cond,
                             DAG.getConstant(ShAmt, MVT::i8));
        }

        // Optimize Cond ? cst+1 : cst -> zext(setcc(C)+cst.
        if (FalseC->getAPIntValue()+1 == TrueC->getAPIntValue()) {
          if (NeedsCondInvert) // Invert the condition if needed.
            Cond = DAG.getNode(ISD::XOR, DL, Cond.getValueType(), Cond,
                               DAG.getConstant(1, Cond.getValueType()));

          // Zero extend the condition if needed.
          Cond = DAG.getNode(ISD::ZERO_EXTEND, DL,
                             FalseC->getValueType(0), Cond);
          return DAG.getNode(ISD::ADD, DL, Cond.getValueType(), Cond,
                             SDValue(FalseC, 0));
        }

        // Optimize cases that will turn into an LEA instruction.  This requires
        // an i32 or i64 and an efficient multiplier (1, 2, 3, 4, 5, 8, 9).
        if (N->getValueType(0) == MVT::i32 || N->getValueType(0) == MVT::i64) {
          uint64_t Diff = TrueC->getZExtValue()-FalseC->getZExtValue();
          if (N->getValueType(0) == MVT::i32) Diff = (unsigned)Diff;

          bool isFastMultiplier = false;
          if (Diff < 10) {
            switch ((unsigned char)Diff) {
              default: break;
              case 1:  // result = add base, cond
              case 2:  // result = lea base(    , cond*2)
              case 3:  // result = lea base(cond, cond*2)
              case 4:  // result = lea base(    , cond*4)
              case 5:  // result = lea base(cond, cond*4)
              case 8:  // result = lea base(    , cond*8)
              case 9:  // result = lea base(cond, cond*8)
                isFastMultiplier = true;
                break;
            }
          }

          if (isFastMultiplier) {
            APInt Diff = TrueC->getAPIntValue()-FalseC->getAPIntValue();
            if (NeedsCondInvert) // Invert the condition if needed.
              Cond = DAG.getNode(ISD::XOR, DL, Cond.getValueType(), Cond,
                                 DAG.getConstant(1, Cond.getValueType()));

            // Zero extend the condition if needed.
            Cond = DAG.getNode(ISD::ZERO_EXTEND, DL, FalseC->getValueType(0),
                               Cond);
            // Scale the condition by the difference.
            if (Diff != 1)
              Cond = DAG.getNode(ISD::MUL, DL, Cond.getValueType(), Cond,
                                 DAG.getConstant(Diff, Cond.getValueType()));

            // Add the base if non-zero.
            if (FalseC->getAPIntValue() != 0)
              Cond = DAG.getNode(ISD::ADD, DL, Cond.getValueType(), Cond,
                                 SDValue(FalseC, 0));
            return Cond;
          }
        }
      }
  }

  // Canonicalize max and min:
  // (x > y) ? x : y -> (x >= y) ? x : y
  // (x < y) ? x : y -> (x <= y) ? x : y
  // This allows use of COND_S / COND_NS (see TranslateX86CC) which eliminates
  // the need for an extra compare
  // against zero. e.g.
  // (x - y) > 0 : (x - y) ? 0 -> (x - y) >= 0 : (x - y) ? 0
  // subl   %esi, %edi
  // testl  %edi, %edi
  // movl   $0, %eax
  // cmovgl %edi, %eax
  // =>
  // xorl   %eax, %eax
  // subl   %esi, $edi
  // cmovsl %eax, %edi
  if (N->getOpcode() == ISD::SELECT && Cond.getOpcode() == ISD::SETCC &&
      DAG.isEqualTo(LHS, Cond.getOperand(0)) &&
      DAG.isEqualTo(RHS, Cond.getOperand(1))) {
    ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
    switch (CC) {
    default: break;
    case ISD::SETLT:
    case ISD::SETGT: {
      ISD::CondCode NewCC = (CC == ISD::SETLT) ? ISD::SETLE : ISD::SETGE;
      Cond = DAG.getSetCC(Cond.getDebugLoc(), Cond.getValueType(),
                          Cond.getOperand(0), Cond.getOperand(1), NewCC);
      return DAG.getNode(ISD::SELECT, DL, VT, Cond, LHS, RHS);
    }
    }
  }

  // If we know that this node is legal then we know that it is going to be
  // matched by one of the SSE/AVX BLEND instructions. These instructions only
  // depend on the highest bit in each word. Try to use SimplifyDemandedBits
  // to simplify previous instructions.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (N->getOpcode() == ISD::VSELECT && DCI.isBeforeLegalizeOps() &&
      !DCI.isBeforeLegalize() && TLI.isOperationLegal(ISD::VSELECT, VT)) {
    unsigned BitWidth = Cond.getValueType().getScalarType().getSizeInBits();

    // Don't optimize vector selects that map to mask-registers.
    if (BitWidth == 1)
      return SDValue();

    assert(BitWidth >= 8 && BitWidth <= 64 && "Invalid mask size");
    APInt DemandedMask = APInt::getHighBitsSet(BitWidth, 1);

    APInt KnownZero, KnownOne;
    TargetLowering::TargetLoweringOpt TLO(DAG, DCI.isBeforeLegalize(),
                                          DCI.isBeforeLegalizeOps());
    if (TLO.ShrinkDemandedConstant(Cond, DemandedMask) ||
        TLI.SimplifyDemandedBits(Cond, DemandedMask, KnownZero, KnownOne, TLO))
      DCI.CommitTargetLoweringOpt(TLO);
  }

  return SDValue();
}

// Check whether a boolean test is testing a boolean value generated by
// X86ISD::SETCC. If so, return the operand of that SETCC and proper condition
// code.
//
// Simplify the following patterns:
// (Op (CMP (SETCC Cond EFLAGS) 1) EQ) or
// (Op (CMP (SETCC Cond EFLAGS) 0) NEQ)
// to (Op EFLAGS Cond)
//
// (Op (CMP (SETCC Cond EFLAGS) 0) EQ) or
// (Op (CMP (SETCC Cond EFLAGS) 1) NEQ)
// to (Op EFLAGS !Cond)
//
// where Op could be BRCOND or CMOV.
//
static SDValue checkBoolTestSetCCCombine(SDValue Cmp, X86::CondCode &CC) {
  // Quit if not CMP and SUB with its value result used.
  if (Cmp.getOpcode() != X86ISD::CMP &&
      (Cmp.getOpcode() != X86ISD::SUB || Cmp.getNode()->hasAnyUseOfValue(0)))
      return SDValue();

  // Quit if not used as a boolean value.
  if (CC != X86::COND_E && CC != X86::COND_NE)
    return SDValue();

  // Check CMP operands. One of them should be 0 or 1 and the other should be
  // an SetCC or extended from it.
  SDValue Op1 = Cmp.getOperand(0);
  SDValue Op2 = Cmp.getOperand(1);

  SDValue SetCC;
  const ConstantSDNode* C = 0;
  bool needOppositeCond = (CC == X86::COND_E);

  if ((C = dyn_cast<ConstantSDNode>(Op1)))
    SetCC = Op2;
  else if ((C = dyn_cast<ConstantSDNode>(Op2)))
    SetCC = Op1;
  else // Quit if all operands are not constants.
    return SDValue();

  if (C->getZExtValue() == 1)
    needOppositeCond = !needOppositeCond;
  else if (C->getZExtValue() != 0)
    // Quit if the constant is neither 0 or 1.
    return SDValue();

  // Skip 'zext' node.
  if (SetCC.getOpcode() == ISD::ZERO_EXTEND)
    SetCC = SetCC.getOperand(0);

  switch (SetCC.getOpcode()) {
  case X86ISD::SETCC:
    // Set the condition code or opposite one if necessary.
    CC = X86::CondCode(SetCC.getConstantOperandVal(0));
    if (needOppositeCond)
      CC = X86::GetOppositeBranchCondition(CC);
    return SetCC.getOperand(1);
  case X86ISD::CMOV: {
    // Check whether false/true value has canonical one, i.e. 0 or 1.
    ConstantSDNode *FVal = dyn_cast<ConstantSDNode>(SetCC.getOperand(0));
    ConstantSDNode *TVal = dyn_cast<ConstantSDNode>(SetCC.getOperand(1));
    // Quit if true value is not a constant.
    if (!TVal)
      return SDValue();
    // Quit if false value is not a constant.
    if (!FVal) {
      // A special case for rdrand, where 0 is set if false cond is found.
      SDValue Op = SetCC.getOperand(0);
      if (Op.getOpcode() != X86ISD::RDRAND)
        return SDValue();
    }
    // Quit if false value is not the constant 0 or 1.
    bool FValIsFalse = true;
    if (FVal && FVal->getZExtValue() != 0) {
      if (FVal->getZExtValue() != 1)
        return SDValue();
      // If FVal is 1, opposite cond is needed.
      needOppositeCond = !needOppositeCond;
      FValIsFalse = false;
    }
    // Quit if TVal is not the constant opposite of FVal.
    if (FValIsFalse && TVal->getZExtValue() != 1)
      return SDValue();
    if (!FValIsFalse && TVal->getZExtValue() != 0)
      return SDValue();
    CC = X86::CondCode(SetCC.getConstantOperandVal(2));
    if (needOppositeCond)
      CC = X86::GetOppositeBranchCondition(CC);
    return SetCC.getOperand(3);
  }
  }

  return SDValue();
}

/// Optimize X86ISD::CMOV [LHS, RHS, CONDCODE (e.g. X86::COND_NE), CONDVAL]
static SDValue PerformCMOVCombine(SDNode *N, SelectionDAG &DAG,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const X86Subtarget *Subtarget) {
  DebugLoc DL = N->getDebugLoc();

  // If the flag operand isn't dead, don't touch this CMOV.
  if (N->getNumValues() == 2 && !SDValue(N, 1).use_empty())
    return SDValue();

  SDValue FalseOp = N->getOperand(0);
  SDValue TrueOp = N->getOperand(1);
  X86::CondCode CC = (X86::CondCode)N->getConstantOperandVal(2);
  SDValue Cond = N->getOperand(3);

  if (CC == X86::COND_E || CC == X86::COND_NE) {
    switch (Cond.getOpcode()) {
    default: break;
    case X86ISD::BSR:
    case X86ISD::BSF:
      // If operand of BSR / BSF are proven never zero, then ZF cannot be set.
      if (DAG.isKnownNeverZero(Cond.getOperand(0)))
        return (CC == X86::COND_E) ? FalseOp : TrueOp;
    }
  }

  SDValue Flags;

  Flags = checkBoolTestSetCCCombine(Cond, CC);
  if (Flags.getNode() &&
      // Extra check as FCMOV only supports a subset of X86 cond.
      (FalseOp.getValueType() != MVT::f80 || hasFPCMov(CC))) {
    SDValue Ops[] = { FalseOp, TrueOp,
                      DAG.getConstant(CC, MVT::i8), Flags };
    return DAG.getNode(X86ISD::CMOV, DL, N->getVTList(),
                       Ops, array_lengthof(Ops));
  }

  // If this is a select between two integer constants, try to do some
  // optimizations.  Note that the operands are ordered the opposite of SELECT
  // operands.
  if (ConstantSDNode *TrueC = dyn_cast<ConstantSDNode>(TrueOp)) {
    if (ConstantSDNode *FalseC = dyn_cast<ConstantSDNode>(FalseOp)) {
      // Canonicalize the TrueC/FalseC values so that TrueC (the true value) is
      // larger than FalseC (the false value).
      if (TrueC->getAPIntValue().ult(FalseC->getAPIntValue())) {
        CC = X86::GetOppositeBranchCondition(CC);
        std::swap(TrueC, FalseC);
        std::swap(TrueOp, FalseOp);
      }

      // Optimize C ? 8 : 0 -> zext(setcc(C)) << 3.  Likewise for any pow2/0.
      // This is efficient for any integer data type (including i8/i16) and
      // shift amount.
      if (FalseC->getAPIntValue() == 0 && TrueC->getAPIntValue().isPowerOf2()) {
        Cond = DAG.getNode(X86ISD::SETCC, DL, MVT::i8,
                           DAG.getConstant(CC, MVT::i8), Cond);

        // Zero extend the condition if needed.
        Cond = DAG.getNode(ISD::ZERO_EXTEND, DL, TrueC->getValueType(0), Cond);

        unsigned ShAmt = TrueC->getAPIntValue().logBase2();
        Cond = DAG.getNode(ISD::SHL, DL, Cond.getValueType(), Cond,
                           DAG.getConstant(ShAmt, MVT::i8));
        if (N->getNumValues() == 2)  // Dead flag value?
          return DCI.CombineTo(N, Cond, SDValue());
        return Cond;
      }

      // Optimize Cond ? cst+1 : cst -> zext(setcc(C)+cst.  This is efficient
      // for any integer data type, including i8/i16.
      if (FalseC->getAPIntValue()+1 == TrueC->getAPIntValue()) {
        Cond = DAG.getNode(X86ISD::SETCC, DL, MVT::i8,
                           DAG.getConstant(CC, MVT::i8), Cond);

        // Zero extend the condition if needed.
        Cond = DAG.getNode(ISD::ZERO_EXTEND, DL,
                           FalseC->getValueType(0), Cond);
        Cond = DAG.getNode(ISD::ADD, DL, Cond.getValueType(), Cond,
                           SDValue(FalseC, 0));

        if (N->getNumValues() == 2)  // Dead flag value?
          return DCI.CombineTo(N, Cond, SDValue());
        return Cond;
      }

      // Optimize cases that will turn into an LEA instruction.  This requires
      // an i32 or i64 and an efficient multiplier (1, 2, 3, 4, 5, 8, 9).
      if (N->getValueType(0) == MVT::i32 || N->getValueType(0) == MVT::i64) {
        uint64_t Diff = TrueC->getZExtValue()-FalseC->getZExtValue();
        if (N->getValueType(0) == MVT::i32) Diff = (unsigned)Diff;

        bool isFastMultiplier = false;
        if (Diff < 10) {
          switch ((unsigned char)Diff) {
          default: break;
          case 1:  // result = add base, cond
          case 2:  // result = lea base(    , cond*2)
          case 3:  // result = lea base(cond, cond*2)
          case 4:  // result = lea base(    , cond*4)
          case 5:  // result = lea base(cond, cond*4)
          case 8:  // result = lea base(    , cond*8)
          case 9:  // result = lea base(cond, cond*8)
            isFastMultiplier = true;
            break;
          }
        }

        if (isFastMultiplier) {
          APInt Diff = TrueC->getAPIntValue()-FalseC->getAPIntValue();
          Cond = DAG.getNode(X86ISD::SETCC, DL, MVT::i8,
                             DAG.getConstant(CC, MVT::i8), Cond);
          // Zero extend the condition if needed.
          Cond = DAG.getNode(ISD::ZERO_EXTEND, DL, FalseC->getValueType(0),
                             Cond);
          // Scale the condition by the difference.
          if (Diff != 1)
            Cond = DAG.getNode(ISD::MUL, DL, Cond.getValueType(), Cond,
                               DAG.getConstant(Diff, Cond.getValueType()));

          // Add the base if non-zero.
          if (FalseC->getAPIntValue() != 0)
            Cond = DAG.getNode(ISD::ADD, DL, Cond.getValueType(), Cond,
                               SDValue(FalseC, 0));
          if (N->getNumValues() == 2)  // Dead flag value?
            return DCI.CombineTo(N, Cond, SDValue());
          return Cond;
        }
      }
    }
  }

  // Handle these cases:
  //   (select (x != c), e, c) -> select (x != c), e, x),
  //   (select (x == c), c, e) -> select (x == c), x, e)
  // where the c is an integer constant, and the "select" is the combination
  // of CMOV and CMP.
  //
  // The rationale for this change is that the conditional-move from a constant
  // needs two instructions, however, conditional-move from a register needs
  // only one instruction.
  //
  // CAVEAT: By replacing a constant with a symbolic value, it may obscure
  //  some instruction-combining opportunities. This opt needs to be
  //  postponed as late as possible.
  //
  if (!DCI.isBeforeLegalize() && !DCI.isBeforeLegalizeOps()) {
    // the DCI.xxxx conditions are provided to postpone the optimization as
    // late as possible.

    ConstantSDNode *CmpAgainst = 0;
    if ((Cond.getOpcode() == X86ISD::CMP || Cond.getOpcode() == X86ISD::SUB) &&
        (CmpAgainst = dyn_cast<ConstantSDNode>(Cond.getOperand(1))) &&
        dyn_cast<ConstantSDNode>(Cond.getOperand(0)) == 0) {

      if (CC == X86::COND_NE &&
          CmpAgainst == dyn_cast<ConstantSDNode>(FalseOp)) {
        CC = X86::GetOppositeBranchCondition(CC);
        std::swap(TrueOp, FalseOp);
      }

      if (CC == X86::COND_E &&
          CmpAgainst == dyn_cast<ConstantSDNode>(TrueOp)) {
        SDValue Ops[] = { FalseOp, Cond.getOperand(0),
                          DAG.getConstant(CC, MVT::i8), Cond };
        return DAG.getNode(X86ISD::CMOV, DL, N->getVTList (), Ops,
                           array_lengthof(Ops));
      }
    }
  }

  return SDValue();
}


/// PerformMulCombine - Optimize a single multiply with constant into two
/// in order to implement it with two cheaper instructions, e.g.
/// LEA + SHL, LEA + LEA.
static SDValue PerformMulCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI) {
  if (DCI.isBeforeLegalize() || DCI.isCalledByLegalizer())
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i64)
    return SDValue();

  ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!C)
    return SDValue();
  uint64_t MulAmt = C->getZExtValue();
  if (isPowerOf2_64(MulAmt) || MulAmt == 3 || MulAmt == 5 || MulAmt == 9)
    return SDValue();

  uint64_t MulAmt1 = 0;
  uint64_t MulAmt2 = 0;
  if ((MulAmt % 9) == 0) {
    MulAmt1 = 9;
    MulAmt2 = MulAmt / 9;
  } else if ((MulAmt % 5) == 0) {
    MulAmt1 = 5;
    MulAmt2 = MulAmt / 5;
  } else if ((MulAmt % 3) == 0) {
    MulAmt1 = 3;
    MulAmt2 = MulAmt / 3;
  }
  if (MulAmt2 &&
      (isPowerOf2_64(MulAmt2) || MulAmt2 == 3 || MulAmt2 == 5 || MulAmt2 == 9)){
    DebugLoc DL = N->getDebugLoc();

    if (isPowerOf2_64(MulAmt2) &&
        !(N->hasOneUse() && N->use_begin()->getOpcode() == ISD::ADD))
      // If second multiplifer is pow2, issue it first. We want the multiply by
      // 3, 5, or 9 to be folded into the addressing mode unless the lone use
      // is an add.
      std::swap(MulAmt1, MulAmt2);

    SDValue NewMul;
    if (isPowerOf2_64(MulAmt1))
      NewMul = DAG.getNode(ISD::SHL, DL, VT, N->getOperand(0),
                           DAG.getConstant(Log2_64(MulAmt1), MVT::i8));
    else
      NewMul = DAG.getNode(X86ISD::MUL_IMM, DL, VT, N->getOperand(0),
                           DAG.getConstant(MulAmt1, VT));

    if (isPowerOf2_64(MulAmt2))
      NewMul = DAG.getNode(ISD::SHL, DL, VT, NewMul,
                           DAG.getConstant(Log2_64(MulAmt2), MVT::i8));
    else
      NewMul = DAG.getNode(X86ISD::MUL_IMM, DL, VT, NewMul,
                           DAG.getConstant(MulAmt2, VT));

    // Do not add new nodes to DAG combiner worklist.
    DCI.CombineTo(N, NewMul, false);
  }
  return SDValue();
}

static SDValue PerformSHLCombine(SDNode *N, SelectionDAG &DAG) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  EVT VT = N0.getValueType();

  // fold (shl (and (setcc_c), c1), c2) -> (and setcc_c, (c1 << c2))
  // since the result of setcc_c is all zero's or all ones.
  if (VT.isInteger() && !VT.isVector() &&
      N1C && N0.getOpcode() == ISD::AND &&
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getOpcode() == X86ISD::SETCC_CARRY ||
        ((N00.getOpcode() == ISD::ANY_EXTEND ||
          N00.getOpcode() == ISD::ZERO_EXTEND) &&
         N00.getOperand(0).getOpcode() == X86ISD::SETCC_CARRY)) {
      APInt Mask = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
      APInt ShAmt = N1C->getAPIntValue();
      Mask = Mask.shl(ShAmt);
      if (Mask != 0)
        return DAG.getNode(ISD::AND, N->getDebugLoc(), VT,
                           N00, DAG.getConstant(Mask, VT));
    }
  }


  // Hardware support for vector shifts is sparse which makes us scalarize the
  // vector operations in many cases. Also, on sandybridge ADD is faster than
  // shl.
  // (shl V, 1) -> add V,V
  if (isSplatVector(N1.getNode())) {
    assert(N0.getValueType().isVector() && "Invalid vector shift type");
    ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1->getOperand(0));
    // We shift all of the values by one. In many cases we do not have
    // hardware support for this operation. This is better expressed as an ADD
    // of two values.
    if (N1C && (1 == N1C->getZExtValue())) {
      return DAG.getNode(ISD::ADD, N->getDebugLoc(), VT, N0, N0);
    }
  }

  return SDValue();
}

/// PerformShiftCombine - Transforms vector shift nodes to use vector shifts
///                       when possible.
static SDValue PerformShiftCombine(SDNode* N, SelectionDAG &DAG,
                                   TargetLowering::DAGCombinerInfo &DCI,
                                   const X86Subtarget *Subtarget) {
  EVT VT = N->getValueType(0);
  if (N->getOpcode() == ISD::SHL) {
    SDValue V = PerformSHLCombine(N, DAG);
    if (V.getNode()) return V;
  }

  // On X86 with SSE2 support, we can transform this to a vector shift if
  // all elements are shifted by the same amount.  We can't do this in legalize
  // because the a constant vector is typically transformed to a constant pool
  // so we have no knowledge of the shift amount.
  if (!Subtarget->hasSSE2())
    return SDValue();

  if (VT != MVT::v2i64 && VT != MVT::v4i32 && VT != MVT::v8i16 &&
      (!Subtarget->hasAVX2() ||
       (VT != MVT::v4i64 && VT != MVT::v8i32 && VT != MVT::v16i16)))
    return SDValue();

  SDValue ShAmtOp = N->getOperand(1);
  EVT EltVT = VT.getVectorElementType();
  DebugLoc DL = N->getDebugLoc();
  SDValue BaseShAmt = SDValue();
  if (ShAmtOp.getOpcode() == ISD::BUILD_VECTOR) {
    unsigned NumElts = VT.getVectorNumElements();
    unsigned i = 0;
    for (; i != NumElts; ++i) {
      SDValue Arg = ShAmtOp.getOperand(i);
      if (Arg.getOpcode() == ISD::UNDEF) continue;
      BaseShAmt = Arg;
      break;
    }
    // Handle the case where the build_vector is all undef
    // FIXME: Should DAG allow this?
    if (i == NumElts)
      return SDValue();

    for (; i != NumElts; ++i) {
      SDValue Arg = ShAmtOp.getOperand(i);
      if (Arg.getOpcode() == ISD::UNDEF) continue;
      if (Arg != BaseShAmt) {
        return SDValue();
      }
    }
  } else if (ShAmtOp.getOpcode() == ISD::VECTOR_SHUFFLE &&
             cast<ShuffleVectorSDNode>(ShAmtOp)->isSplat()) {
    SDValue InVec = ShAmtOp.getOperand(0);
    if (InVec.getOpcode() == ISD::BUILD_VECTOR) {
      unsigned NumElts = InVec.getValueType().getVectorNumElements();
      unsigned i = 0;
      for (; i != NumElts; ++i) {
        SDValue Arg = InVec.getOperand(i);
        if (Arg.getOpcode() == ISD::UNDEF) continue;
        BaseShAmt = Arg;
        break;
      }
    } else if (InVec.getOpcode() == ISD::INSERT_VECTOR_ELT) {
       if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(InVec.getOperand(2))) {
         unsigned SplatIdx= cast<ShuffleVectorSDNode>(ShAmtOp)->getSplatIndex();
         if (C->getZExtValue() == SplatIdx)
           BaseShAmt = InVec.getOperand(1);
       }
    }
    if (BaseShAmt.getNode() == 0) {
      // Don't create instructions with illegal types after legalize
      // types has run.
      if (!DAG.getTargetLoweringInfo().isTypeLegal(EltVT) &&
          !DCI.isBeforeLegalize())
        return SDValue();

      BaseShAmt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, ShAmtOp,
                              DAG.getIntPtrConstant(0));
    }
  } else
    return SDValue();

  // The shift amount is an i32.
  if (EltVT.bitsGT(MVT::i32))
    BaseShAmt = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, BaseShAmt);
  else if (EltVT.bitsLT(MVT::i32))
    BaseShAmt = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, BaseShAmt);

  // The shift amount is identical so we can do a vector shift.
  SDValue  ValOp = N->getOperand(0);
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Unknown shift opcode!");
  case ISD::SHL:
    switch (VT.getSimpleVT().SimpleTy) {
    default: return SDValue();
    case MVT::v2i64:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v4i64:
    case MVT::v8i32:
    case MVT::v16i16:
      return getTargetVShiftNode(X86ISD::VSHLI, DL, VT, ValOp, BaseShAmt, DAG);
    }
  case ISD::SRA:
    switch (VT.getSimpleVT().SimpleTy) {
    default: return SDValue();
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v8i32:
    case MVT::v16i16:
      return getTargetVShiftNode(X86ISD::VSRAI, DL, VT, ValOp, BaseShAmt, DAG);
    }
  case ISD::SRL:
    switch (VT.getSimpleVT().SimpleTy) {
    default: return SDValue();
    case MVT::v2i64:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v4i64:
    case MVT::v8i32:
    case MVT::v16i16:
      return getTargetVShiftNode(X86ISD::VSRLI, DL, VT, ValOp, BaseShAmt, DAG);
    }
  }
}


// CMPEQCombine - Recognize the distinctive  (AND (setcc ...) (setcc ..))
// where both setccs reference the same FP CMP, and rewrite for CMPEQSS
// and friends.  Likewise for OR -> CMPNEQSS.
static SDValue CMPEQCombine(SDNode *N, SelectionDAG &DAG,
                            TargetLowering::DAGCombinerInfo &DCI,
                            const X86Subtarget *Subtarget) {
  unsigned opcode;

  // SSE1 supports CMP{eq|ne}SS, and SSE2 added CMP{eq|ne}SD, but
  // we're requiring SSE2 for both.
  if (Subtarget->hasSSE2() && isAndOrOfSetCCs(SDValue(N, 0U), opcode)) {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue CMP0 = N0->getOperand(1);
    SDValue CMP1 = N1->getOperand(1);
    DebugLoc DL = N->getDebugLoc();

    // The SETCCs should both refer to the same CMP.
    if (CMP0.getOpcode() != X86ISD::CMP || CMP0 != CMP1)
      return SDValue();

    SDValue CMP00 = CMP0->getOperand(0);
    SDValue CMP01 = CMP0->getOperand(1);
    EVT     VT    = CMP00.getValueType();

    if (VT == MVT::f32 || VT == MVT::f64) {
      bool ExpectingFlags = false;
      // Check for any users that want flags:
      for (SDNode::use_iterator UI = N->use_begin(),
             UE = N->use_end();
           !ExpectingFlags && UI != UE; ++UI)
        switch (UI->getOpcode()) {
        default:
        case ISD::BR_CC:
        case ISD::BRCOND:
        case ISD::SELECT:
          ExpectingFlags = true;
          break;
        case ISD::CopyToReg:
        case ISD::SIGN_EXTEND:
        case ISD::ZERO_EXTEND:
        case ISD::ANY_EXTEND:
          break;
        }

      if (!ExpectingFlags) {
        enum X86::CondCode cc0 = (enum X86::CondCode)N0.getConstantOperandVal(0);
        enum X86::CondCode cc1 = (enum X86::CondCode)N1.getConstantOperandVal(0);

        if (cc1 == X86::COND_E || cc1 == X86::COND_NE) {
          X86::CondCode tmp = cc0;
          cc0 = cc1;
          cc1 = tmp;
        }

        if ((cc0 == X86::COND_E  && cc1 == X86::COND_NP) ||
            (cc0 == X86::COND_NE && cc1 == X86::COND_P)) {
          bool is64BitFP = (CMP00.getValueType() == MVT::f64);
          X86ISD::NodeType NTOperator = is64BitFP ?
            X86ISD::FSETCCsd : X86ISD::FSETCCss;
          // FIXME: need symbolic constants for these magic numbers.
          // See X86ATTInstPrinter.cpp:printSSECC().
          unsigned x86cc = (cc0 == X86::COND_E) ? 0 : 4;
          SDValue OnesOrZeroesF = DAG.getNode(NTOperator, DL, MVT::f32, CMP00, CMP01,
                                              DAG.getConstant(x86cc, MVT::i8));
          SDValue OnesOrZeroesI = DAG.getNode(ISD::BITCAST, DL, MVT::i32,
                                              OnesOrZeroesF);
          SDValue ANDed = DAG.getNode(ISD::AND, DL, MVT::i32, OnesOrZeroesI,
                                      DAG.getConstant(1, MVT::i32));
          SDValue OneBitOfTruth = DAG.getNode(ISD::TRUNCATE, DL, MVT::i8, ANDed);
          return OneBitOfTruth;
        }
      }
    }
  }
  return SDValue();
}

/// CanFoldXORWithAllOnes - Test whether the XOR operand is a AllOnes vector
/// so it can be folded inside ANDNP.
static bool CanFoldXORWithAllOnes(const SDNode *N) {
  EVT VT = N->getValueType(0);

  // Match direct AllOnes for 128 and 256-bit vectors
  if (ISD::isBuildVectorAllOnes(N))
    return true;

  // Look through a bit convert.
  if (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0).getNode();

  // Sometimes the operand may come from a insert_subvector building a 256-bit
  // allones vector
  if (VT.is256BitVector() &&
      N->getOpcode() == ISD::INSERT_SUBVECTOR) {
    SDValue V1 = N->getOperand(0);
    SDValue V2 = N->getOperand(1);

    if (V1.getOpcode() == ISD::INSERT_SUBVECTOR &&
        V1.getOperand(0).getOpcode() == ISD::UNDEF &&
        ISD::isBuildVectorAllOnes(V1.getOperand(1).getNode()) &&
        ISD::isBuildVectorAllOnes(V2.getNode()))
      return true;
  }

  return false;
}

static SDValue PerformAndCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const X86Subtarget *Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SDValue R = CMPEQCombine(N, DAG, DCI, Subtarget);
  if (R.getNode())
    return R;

  EVT VT = N->getValueType(0);

  // Create ANDN, BLSI, and BLSR instructions
  // BLSI is X & (-X)
  // BLSR is X & (X-1)
  if (Subtarget->hasBMI() && (VT == MVT::i32 || VT == MVT::i64)) {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    DebugLoc DL = N->getDebugLoc();

    // Check LHS for not
    if (N0.getOpcode() == ISD::XOR && isAllOnes(N0.getOperand(1)))
      return DAG.getNode(X86ISD::ANDN, DL, VT, N0.getOperand(0), N1);
    // Check RHS for not
    if (N1.getOpcode() == ISD::XOR && isAllOnes(N1.getOperand(1)))
      return DAG.getNode(X86ISD::ANDN, DL, VT, N1.getOperand(0), N0);

    // Check LHS for neg
    if (N0.getOpcode() == ISD::SUB && N0.getOperand(1) == N1 &&
        isZero(N0.getOperand(0)))
      return DAG.getNode(X86ISD::BLSI, DL, VT, N1);

    // Check RHS for neg
    if (N1.getOpcode() == ISD::SUB && N1.getOperand(1) == N0 &&
        isZero(N1.getOperand(0)))
      return DAG.getNode(X86ISD::BLSI, DL, VT, N0);

    // Check LHS for X-1
    if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1 &&
        isAllOnes(N0.getOperand(1)))
      return DAG.getNode(X86ISD::BLSR, DL, VT, N1);

    // Check RHS for X-1
    if (N1.getOpcode() == ISD::ADD && N1.getOperand(0) == N0 &&
        isAllOnes(N1.getOperand(1)))
      return DAG.getNode(X86ISD::BLSR, DL, VT, N0);

    return SDValue();
  }

  // Want to form ANDNP nodes:
  // 1) In the hopes of then easily combining them with OR and AND nodes
  //    to form PBLEND/PSIGN.
  // 2) To match ANDN packed intrinsics
  if (VT != MVT::v2i64 && VT != MVT::v4i64)
    return SDValue();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  DebugLoc DL = N->getDebugLoc();

  // Check LHS for vnot
  if (N0.getOpcode() == ISD::XOR &&
      //ISD::isBuildVectorAllOnes(N0.getOperand(1).getNode()))
      CanFoldXORWithAllOnes(N0.getOperand(1).getNode()))
    return DAG.getNode(X86ISD::ANDNP, DL, VT, N0.getOperand(0), N1);

  // Check RHS for vnot
  if (N1.getOpcode() == ISD::XOR &&
      //ISD::isBuildVectorAllOnes(N1.getOperand(1).getNode()))
      CanFoldXORWithAllOnes(N1.getOperand(1).getNode()))
    return DAG.getNode(X86ISD::ANDNP, DL, VT, N1.getOperand(0), N0);

  return SDValue();
}

static SDValue PerformOrCombine(SDNode *N, SelectionDAG &DAG,
                                TargetLowering::DAGCombinerInfo &DCI,
                                const X86Subtarget *Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SDValue R = CMPEQCombine(N, DAG, DCI, Subtarget);
  if (R.getNode())
    return R;

  EVT VT = N->getValueType(0);

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // look for psign/blend
  if (VT == MVT::v2i64 || VT == MVT::v4i64) {
    if (!Subtarget->hasSSSE3() ||
        (VT == MVT::v4i64 && !Subtarget->hasAVX2()))
      return SDValue();

    // Canonicalize pandn to RHS
    if (N0.getOpcode() == X86ISD::ANDNP)
      std::swap(N0, N1);
    // or (and (m, y), (pandn m, x))
    if (N0.getOpcode() == ISD::AND && N1.getOpcode() == X86ISD::ANDNP) {
      SDValue Mask = N1.getOperand(0);
      SDValue X    = N1.getOperand(1);
      SDValue Y;
      if (N0.getOperand(0) == Mask)
        Y = N0.getOperand(1);
      if (N0.getOperand(1) == Mask)
        Y = N0.getOperand(0);

      // Check to see if the mask appeared in both the AND and ANDNP and
      if (!Y.getNode())
        return SDValue();

      // Validate that X, Y, and Mask are BIT_CONVERTS, and see through them.
      // Look through mask bitcast.
      if (Mask.getOpcode() == ISD::BITCAST)
        Mask = Mask.getOperand(0);
      if (X.getOpcode() == ISD::BITCAST)
        X = X.getOperand(0);
      if (Y.getOpcode() == ISD::BITCAST)
        Y = Y.getOperand(0);

      EVT MaskVT = Mask.getValueType();

      // Validate that the Mask operand is a vector sra node.
      // FIXME: what to do for bytes, since there is a psignb/pblendvb, but
      // there is no psrai.b
      if (Mask.getOpcode() != X86ISD::VSRAI)
        return SDValue();

      // Check that the SRA is all signbits.
      SDValue SraC = Mask.getOperand(1);
      unsigned SraAmt  = cast<ConstantSDNode>(SraC)->getZExtValue();
      unsigned EltBits = MaskVT.getVectorElementType().getSizeInBits();
      if ((SraAmt + 1) != EltBits)
        return SDValue();

      DebugLoc DL = N->getDebugLoc();

      // Now we know we at least have a plendvb with the mask val.  See if
      // we can form a psignb/w/d.
      // psign = x.type == y.type == mask.type && y = sub(0, x);
      if (Y.getOpcode() == ISD::SUB && Y.getOperand(1) == X &&
          ISD::isBuildVectorAllZeros(Y.getOperand(0).getNode()) &&
          X.getValueType() == MaskVT && Y.getValueType() == MaskVT) {
        assert((EltBits == 8 || EltBits == 16 || EltBits == 32) &&
               "Unsupported VT for PSIGN");
        Mask = DAG.getNode(X86ISD::PSIGN, DL, MaskVT, X, Mask.getOperand(0));
        return DAG.getNode(ISD::BITCAST, DL, VT, Mask);
      }
      // PBLENDVB only available on SSE 4.1
      if (!Subtarget->hasSSE41())
        return SDValue();

      EVT BlendVT = (VT == MVT::v4i64) ? MVT::v32i8 : MVT::v16i8;

      X = DAG.getNode(ISD::BITCAST, DL, BlendVT, X);
      Y = DAG.getNode(ISD::BITCAST, DL, BlendVT, Y);
      Mask = DAG.getNode(ISD::BITCAST, DL, BlendVT, Mask);
      Mask = DAG.getNode(ISD::VSELECT, DL, BlendVT, Mask, Y, X);
      return DAG.getNode(ISD::BITCAST, DL, VT, Mask);
    }
  }

  if (VT != MVT::i16 && VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  // fold (or (x << c) | (y >> (64 - c))) ==> (shld64 x, y, c)
  if (N0.getOpcode() == ISD::SRL && N1.getOpcode() == ISD::SHL)
    std::swap(N0, N1);
  if (N0.getOpcode() != ISD::SHL || N1.getOpcode() != ISD::SRL)
    return SDValue();
  if (!N0.hasOneUse() || !N1.hasOneUse())
    return SDValue();

  SDValue ShAmt0 = N0.getOperand(1);
  if (ShAmt0.getValueType() != MVT::i8)
    return SDValue();
  SDValue ShAmt1 = N1.getOperand(1);
  if (ShAmt1.getValueType() != MVT::i8)
    return SDValue();
  if (ShAmt0.getOpcode() == ISD::TRUNCATE)
    ShAmt0 = ShAmt0.getOperand(0);
  if (ShAmt1.getOpcode() == ISD::TRUNCATE)
    ShAmt1 = ShAmt1.getOperand(0);

  DebugLoc DL = N->getDebugLoc();
  unsigned Opc = X86ISD::SHLD;
  SDValue Op0 = N0.getOperand(0);
  SDValue Op1 = N1.getOperand(0);
  if (ShAmt0.getOpcode() == ISD::SUB) {
    Opc = X86ISD::SHRD;
    std::swap(Op0, Op1);
    std::swap(ShAmt0, ShAmt1);
  }

  unsigned Bits = VT.getSizeInBits();
  if (ShAmt1.getOpcode() == ISD::SUB) {
    SDValue Sum = ShAmt1.getOperand(0);
    if (ConstantSDNode *SumC = dyn_cast<ConstantSDNode>(Sum)) {
      SDValue ShAmt1Op1 = ShAmt1.getOperand(1);
      if (ShAmt1Op1.getNode()->getOpcode() == ISD::TRUNCATE)
        ShAmt1Op1 = ShAmt1Op1.getOperand(0);
      if (SumC->getSExtValue() == Bits && ShAmt1Op1 == ShAmt0)
        return DAG.getNode(Opc, DL, VT,
                           Op0, Op1,
                           DAG.getNode(ISD::TRUNCATE, DL,
                                       MVT::i8, ShAmt0));
    }
  } else if (ConstantSDNode *ShAmt1C = dyn_cast<ConstantSDNode>(ShAmt1)) {
    ConstantSDNode *ShAmt0C = dyn_cast<ConstantSDNode>(ShAmt0);
    if (ShAmt0C &&
        ShAmt0C->getSExtValue() + ShAmt1C->getSExtValue() == Bits)
      return DAG.getNode(Opc, DL, VT,
                         N0.getOperand(0), N1.getOperand(0),
                         DAG.getNode(ISD::TRUNCATE, DL,
                                       MVT::i8, ShAmt0));
  }

  return SDValue();
}

// Generate NEG and CMOV for integer abs.
static SDValue performIntegerAbsCombine(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);

  // Since X86 does not have CMOV for 8-bit integer, we don't convert
  // 8-bit integer abs to NEG and CMOV.
  if (VT.isInteger() && VT.getSizeInBits() == 8)
    return SDValue();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  DebugLoc DL = N->getDebugLoc();

  // Check pattern of XOR(ADD(X,Y), Y) where Y is SRA(X, size(X)-1)
  // and change it to SUB and CMOV.
  if (VT.isInteger() && N->getOpcode() == ISD::XOR &&
      N0.getOpcode() == ISD::ADD &&
      N0.getOperand(1) == N1 &&
      N1.getOpcode() == ISD::SRA &&
      N1.getOperand(0) == N0.getOperand(0))
    if (ConstantSDNode *Y1C = dyn_cast<ConstantSDNode>(N1.getOperand(1)))
      if (Y1C->getAPIntValue() == VT.getSizeInBits()-1) {
        // Generate SUB & CMOV.
        SDValue Neg = DAG.getNode(X86ISD::SUB, DL, DAG.getVTList(VT, MVT::i32),
                                  DAG.getConstant(0, VT), N0.getOperand(0));

        SDValue Ops[] = { N0.getOperand(0), Neg,
                          DAG.getConstant(X86::COND_GE, MVT::i8),
                          SDValue(Neg.getNode(), 1) };
        return DAG.getNode(X86ISD::CMOV, DL, DAG.getVTList(VT, MVT::Glue),
                           Ops, array_lengthof(Ops));
      }
  return SDValue();
}

// PerformXorCombine - Attempts to turn XOR nodes into BLSMSK nodes
static SDValue PerformXorCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const X86Subtarget *Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  if (Subtarget->hasCMov()) {
    SDValue RV = performIntegerAbsCombine(N, DAG);
    if (RV.getNode())
      return RV;
  }

  // Try forming BMI if it is available.
  if (!Subtarget->hasBMI())
    return SDValue();

  EVT VT = N->getValueType(0);

  if (VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  assert(Subtarget->hasBMI() && "Creating BLSMSK requires BMI instructions");

  // Create BLSMSK instructions by finding X ^ (X-1)
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  DebugLoc DL = N->getDebugLoc();

  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1 &&
      isAllOnes(N0.getOperand(1)))
    return DAG.getNode(X86ISD::BLSMSK, DL, VT, N1);

  if (N1.getOpcode() == ISD::ADD && N1.getOperand(0) == N0 &&
      isAllOnes(N1.getOperand(1)))
    return DAG.getNode(X86ISD::BLSMSK, DL, VT, N0);

  return SDValue();
}

/// PerformLOADCombine - Do target-specific dag combines on LOAD nodes.
static SDValue PerformLOADCombine(SDNode *N, SelectionDAG &DAG,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const X86Subtarget *Subtarget) {
  LoadSDNode *Ld = cast<LoadSDNode>(N);
  EVT RegVT = Ld->getValueType(0);
  EVT MemVT = Ld->getMemoryVT();
  DebugLoc dl = Ld->getDebugLoc();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  ISD::LoadExtType Ext = Ld->getExtensionType();

  // If this is a vector EXT Load then attempt to optimize it using a
  // shuffle. We need SSSE3 shuffles.
  // TODO: It is possible to support ZExt by zeroing the undef values
  // during the shuffle phase or after the shuffle.
  if (RegVT.isVector() && RegVT.isInteger() &&
      Ext == ISD::EXTLOAD && Subtarget->hasSSSE3()) {
    assert(MemVT != RegVT && "Cannot extend to the same type");
    assert(MemVT.isVector() && "Must load a vector from memory");

    unsigned NumElems = RegVT.getVectorNumElements();
    unsigned RegSz = RegVT.getSizeInBits();
    unsigned MemSz = MemVT.getSizeInBits();
    assert(RegSz > MemSz && "Register size must be greater than the mem size");

    // All sizes must be a power of two.
    if (!isPowerOf2_32(RegSz * MemSz * NumElems))
      return SDValue();

    // Attempt to load the original value using scalar loads.
    // Find the largest scalar type that divides the total loaded size.
    MVT SclrLoadTy = MVT::i8;
    for (unsigned tp = MVT::FIRST_INTEGER_VALUETYPE;
         tp < MVT::LAST_INTEGER_VALUETYPE; ++tp) {
      MVT Tp = (MVT::SimpleValueType)tp;
      if (TLI.isTypeLegal(Tp) && ((MemSz % Tp.getSizeInBits()) == 0)) {
        SclrLoadTy = Tp;
      }
    }

    // On 32bit systems, we can't save 64bit integers. Try bitcasting to F64.
    if (TLI.isTypeLegal(MVT::f64) && SclrLoadTy.getSizeInBits() < 64 &&
        (64 <= MemSz))
      SclrLoadTy = MVT::f64;

    // Calculate the number of scalar loads that we need to perform
    // in order to load our vector from memory.
    unsigned NumLoads = MemSz / SclrLoadTy.getSizeInBits();

    // Represent our vector as a sequence of elements which are the
    // largest scalar that we can load.
    EVT LoadUnitVecVT = EVT::getVectorVT(*DAG.getContext(), SclrLoadTy,
      RegSz/SclrLoadTy.getSizeInBits());

    // Represent the data using the same element type that is stored in
    // memory. In practice, we ''widen'' MemVT.
    EVT WideVecVT = EVT::getVectorVT(*DAG.getContext(), MemVT.getScalarType(),
                                  RegSz/MemVT.getScalarType().getSizeInBits());

    assert(WideVecVT.getSizeInBits() == LoadUnitVecVT.getSizeInBits() &&
      "Invalid vector type");

    // We can't shuffle using an illegal type.
    if (!TLI.isTypeLegal(WideVecVT))
      return SDValue();

    SmallVector<SDValue, 8> Chains;
    SDValue Ptr = Ld->getBasePtr();
    SDValue Increment = DAG.getConstant(SclrLoadTy.getSizeInBits()/8,
                                        TLI.getPointerTy());
    SDValue Res = DAG.getUNDEF(LoadUnitVecVT);

    for (unsigned i = 0; i < NumLoads; ++i) {
      // Perform a single load.
      SDValue ScalarLoad = DAG.getLoad(SclrLoadTy, dl, Ld->getChain(),
                                       Ptr, Ld->getPointerInfo(),
                                       Ld->isVolatile(), Ld->isNonTemporal(),
                                       Ld->isInvariant(), Ld->getAlignment());
      Chains.push_back(ScalarLoad.getValue(1));
      // Create the first element type using SCALAR_TO_VECTOR in order to avoid
      // another round of DAGCombining.
      if (i == 0)
        Res = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, LoadUnitVecVT, ScalarLoad);
      else
        Res = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, LoadUnitVecVT, Res,
                          ScalarLoad, DAG.getIntPtrConstant(i));

      Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr, Increment);
    }

    SDValue TF = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &Chains[0],
                               Chains.size());

    // Bitcast the loaded value to a vector of the original element type, in
    // the size of the target vector type.
    SDValue SlicedVec = DAG.getNode(ISD::BITCAST, dl, WideVecVT, Res);
    unsigned SizeRatio = RegSz/MemSz;

    // Redistribute the loaded elements into the different locations.
    SmallVector<int, 8> ShuffleVec(NumElems * SizeRatio, -1);
    for (unsigned i = 0; i != NumElems; ++i)
      ShuffleVec[i*SizeRatio] = i;

    SDValue Shuff = DAG.getVectorShuffle(WideVecVT, dl, SlicedVec,
                                         DAG.getUNDEF(WideVecVT),
                                         &ShuffleVec[0]);

    // Bitcast to the requested type.
    Shuff = DAG.getNode(ISD::BITCAST, dl, RegVT, Shuff);
    // Replace the original load with the new sequence
    // and return the new chain.
    return DCI.CombineTo(N, Shuff, TF, true);
  }

  return SDValue();
}

/// PerformSTORECombine - Do target-specific dag combines on STORE nodes.
static SDValue PerformSTORECombine(SDNode *N, SelectionDAG &DAG,
                                   const X86Subtarget *Subtarget) {
  StoreSDNode *St = cast<StoreSDNode>(N);
  EVT VT = St->getValue().getValueType();
  EVT StVT = St->getMemoryVT();
  DebugLoc dl = St->getDebugLoc();
  SDValue StoredVal = St->getOperand(1);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // If we are saving a concatenation of two XMM registers, perform two stores.
  // On Sandy Bridge, 256-bit memory operations are executed by two
  // 128-bit ports. However, on Haswell it is better to issue a single 256-bit
  // memory  operation.
  if (VT.is256BitVector() && !Subtarget->hasAVX2() &&
      StoredVal.getNode()->getOpcode() == ISD::CONCAT_VECTORS &&
      StoredVal.getNumOperands() == 2) {
    SDValue Value0 = StoredVal.getOperand(0);
    SDValue Value1 = StoredVal.getOperand(1);

    SDValue Stride = DAG.getConstant(16, TLI.getPointerTy());
    SDValue Ptr0 = St->getBasePtr();
    SDValue Ptr1 = DAG.getNode(ISD::ADD, dl, Ptr0.getValueType(), Ptr0, Stride);

    SDValue Ch0 = DAG.getStore(St->getChain(), dl, Value0, Ptr0,
                                St->getPointerInfo(), St->isVolatile(),
                                St->isNonTemporal(), St->getAlignment());
    SDValue Ch1 = DAG.getStore(St->getChain(), dl, Value1, Ptr1,
                                St->getPointerInfo(), St->isVolatile(),
                                St->isNonTemporal(), St->getAlignment());
    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Ch0, Ch1);
  }

  // Optimize trunc store (of multiple scalars) to shuffle and store.
  // First, pack all of the elements in one place. Next, store to memory
  // in fewer chunks.
  if (St->isTruncatingStore() && VT.isVector()) {
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    unsigned NumElems = VT.getVectorNumElements();
    assert(StVT != VT && "Cannot truncate to the same type");
    unsigned FromSz = VT.getVectorElementType().getSizeInBits();
    unsigned ToSz = StVT.getVectorElementType().getSizeInBits();

    // From, To sizes and ElemCount must be pow of two
    if (!isPowerOf2_32(NumElems * FromSz * ToSz)) return SDValue();
    // We are going to use the original vector elt for storing.
    // Accumulated smaller vector elements must be a multiple of the store size.
    if (0 != (NumElems * FromSz) % ToSz) return SDValue();

    unsigned SizeRatio  = FromSz / ToSz;

    assert(SizeRatio * NumElems * ToSz == VT.getSizeInBits());

    // Create a type on which we perform the shuffle
    EVT WideVecVT = EVT::getVectorVT(*DAG.getContext(),
            StVT.getScalarType(), NumElems*SizeRatio);

    assert(WideVecVT.getSizeInBits() == VT.getSizeInBits());

    SDValue WideVec = DAG.getNode(ISD::BITCAST, dl, WideVecVT, St->getValue());
    SmallVector<int, 8> ShuffleVec(NumElems * SizeRatio, -1);
    for (unsigned i = 0; i != NumElems; ++i)
      ShuffleVec[i] = i * SizeRatio;

    // Can't shuffle using an illegal type.
    if (!TLI.isTypeLegal(WideVecVT))
      return SDValue();

    SDValue Shuff = DAG.getVectorShuffle(WideVecVT, dl, WideVec,
                                         DAG.getUNDEF(WideVecVT),
                                         &ShuffleVec[0]);
    // At this point all of the data is stored at the bottom of the
    // register. We now need to save it to mem.

    // Find the largest store unit
    MVT StoreType = MVT::i8;
    for (unsigned tp = MVT::FIRST_INTEGER_VALUETYPE;
         tp < MVT::LAST_INTEGER_VALUETYPE; ++tp) {
      MVT Tp = (MVT::SimpleValueType)tp;
      if (TLI.isTypeLegal(Tp) && Tp.getSizeInBits() <= NumElems * ToSz)
        StoreType = Tp;
    }

    // On 32bit systems, we can't save 64bit integers. Try bitcasting to F64.
    if (TLI.isTypeLegal(MVT::f64) && StoreType.getSizeInBits() < 64 &&
        (64 <= NumElems * ToSz))
      StoreType = MVT::f64;

    // Bitcast the original vector into a vector of store-size units
    EVT StoreVecVT = EVT::getVectorVT(*DAG.getContext(),
            StoreType, VT.getSizeInBits()/StoreType.getSizeInBits());
    assert(StoreVecVT.getSizeInBits() == VT.getSizeInBits());
    SDValue ShuffWide = DAG.getNode(ISD::BITCAST, dl, StoreVecVT, Shuff);
    SmallVector<SDValue, 8> Chains;
    SDValue Increment = DAG.getConstant(StoreType.getSizeInBits()/8,
                                        TLI.getPointerTy());
    SDValue Ptr = St->getBasePtr();

    // Perform one or more big stores into memory.
    for (unsigned i=0, e=(ToSz*NumElems)/StoreType.getSizeInBits(); i!=e; ++i) {
      SDValue SubVec = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                                   StoreType, ShuffWide,
                                   DAG.getIntPtrConstant(i));
      SDValue Ch = DAG.getStore(St->getChain(), dl, SubVec, Ptr,
                                St->getPointerInfo(), St->isVolatile(),
                                St->isNonTemporal(), St->getAlignment());
      Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr, Increment);
      Chains.push_back(Ch);
    }

    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &Chains[0],
                               Chains.size());
  }


  // Turn load->store of MMX types into GPR load/stores.  This avoids clobbering
  // the FP state in cases where an emms may be missing.
  // A preferable solution to the general problem is to figure out the right
  // places to insert EMMS.  This qualifies as a quick hack.

  // Similarly, turn load->store of i64 into double load/stores in 32-bit mode.
  if (VT.getSizeInBits() != 64)
    return SDValue();

  const Function *F = DAG.getMachineFunction().getFunction();
  bool NoImplicitFloatOps = F->getFnAttributes().
    hasAttribute(Attributes::NoImplicitFloat);
  bool F64IsLegal = !DAG.getTarget().Options.UseSoftFloat && !NoImplicitFloatOps
                     && Subtarget->hasSSE2();
  if ((VT.isVector() ||
       (VT == MVT::i64 && F64IsLegal && !Subtarget->is64Bit())) &&
      isa<LoadSDNode>(St->getValue()) &&
      !cast<LoadSDNode>(St->getValue())->isVolatile() &&
      St->getChain().hasOneUse() && !St->isVolatile()) {
    SDNode* LdVal = St->getValue().getNode();
    LoadSDNode *Ld = 0;
    int TokenFactorIndex = -1;
    SmallVector<SDValue, 8> Ops;
    SDNode* ChainVal = St->getChain().getNode();
    // Must be a store of a load.  We currently handle two cases:  the load
    // is a direct child, and it's under an intervening TokenFactor.  It is
    // possible to dig deeper under nested TokenFactors.
    if (ChainVal == LdVal)
      Ld = cast<LoadSDNode>(St->getChain());
    else if (St->getValue().hasOneUse() &&
             ChainVal->getOpcode() == ISD::TokenFactor) {
      for (unsigned i = 0, e = ChainVal->getNumOperands(); i != e; ++i) {
        if (ChainVal->getOperand(i).getNode() == LdVal) {
          TokenFactorIndex = i;
          Ld = cast<LoadSDNode>(St->getValue());
        } else
          Ops.push_back(ChainVal->getOperand(i));
      }
    }

    if (!Ld || !ISD::isNormalLoad(Ld))
      return SDValue();

    // If this is not the MMX case, i.e. we are just turning i64 load/store
    // into f64 load/store, avoid the transformation if there are multiple
    // uses of the loaded value.
    if (!VT.isVector() && !Ld->hasNUsesOfValue(1, 0))
      return SDValue();

    DebugLoc LdDL = Ld->getDebugLoc();
    DebugLoc StDL = N->getDebugLoc();
    // If we are a 64-bit capable x86, lower to a single movq load/store pair.
    // Otherwise, if it's legal to use f64 SSE instructions, use f64 load/store
    // pair instead.
    if (Subtarget->is64Bit() || F64IsLegal) {
      EVT LdVT = Subtarget->is64Bit() ? MVT::i64 : MVT::f64;
      SDValue NewLd = DAG.getLoad(LdVT, LdDL, Ld->getChain(), Ld->getBasePtr(),
                                  Ld->getPointerInfo(), Ld->isVolatile(),
                                  Ld->isNonTemporal(), Ld->isInvariant(),
                                  Ld->getAlignment());
      SDValue NewChain = NewLd.getValue(1);
      if (TokenFactorIndex != -1) {
        Ops.push_back(NewChain);
        NewChain = DAG.getNode(ISD::TokenFactor, LdDL, MVT::Other, &Ops[0],
                               Ops.size());
      }
      return DAG.getStore(NewChain, StDL, NewLd, St->getBasePtr(),
                          St->getPointerInfo(),
                          St->isVolatile(), St->isNonTemporal(),
                          St->getAlignment());
    }

    // Otherwise, lower to two pairs of 32-bit loads / stores.
    SDValue LoAddr = Ld->getBasePtr();
    SDValue HiAddr = DAG.getNode(ISD::ADD, LdDL, MVT::i32, LoAddr,
                                 DAG.getConstant(4, MVT::i32));

    SDValue LoLd = DAG.getLoad(MVT::i32, LdDL, Ld->getChain(), LoAddr,
                               Ld->getPointerInfo(),
                               Ld->isVolatile(), Ld->isNonTemporal(),
                               Ld->isInvariant(), Ld->getAlignment());
    SDValue HiLd = DAG.getLoad(MVT::i32, LdDL, Ld->getChain(), HiAddr,
                               Ld->getPointerInfo().getWithOffset(4),
                               Ld->isVolatile(), Ld->isNonTemporal(),
                               Ld->isInvariant(),
                               MinAlign(Ld->getAlignment(), 4));

    SDValue NewChain = LoLd.getValue(1);
    if (TokenFactorIndex != -1) {
      Ops.push_back(LoLd);
      Ops.push_back(HiLd);
      NewChain = DAG.getNode(ISD::TokenFactor, LdDL, MVT::Other, &Ops[0],
                             Ops.size());
    }

    LoAddr = St->getBasePtr();
    HiAddr = DAG.getNode(ISD::ADD, StDL, MVT::i32, LoAddr,
                         DAG.getConstant(4, MVT::i32));

    SDValue LoSt = DAG.getStore(NewChain, StDL, LoLd, LoAddr,
                                St->getPointerInfo(),
                                St->isVolatile(), St->isNonTemporal(),
                                St->getAlignment());
    SDValue HiSt = DAG.getStore(NewChain, StDL, HiLd, HiAddr,
                                St->getPointerInfo().getWithOffset(4),
                                St->isVolatile(),
                                St->isNonTemporal(),
                                MinAlign(St->getAlignment(), 4));
    return DAG.getNode(ISD::TokenFactor, StDL, MVT::Other, LoSt, HiSt);
  }
  return SDValue();
}

/// isHorizontalBinOp - Return 'true' if this vector operation is "horizontal"
/// and return the operands for the horizontal operation in LHS and RHS.  A
/// horizontal operation performs the binary operation on successive elements
/// of its first operand, then on successive elements of its second operand,
/// returning the resulting values in a vector.  For example, if
///   A = < float a0, float a1, float a2, float a3 >
/// and
///   B = < float b0, float b1, float b2, float b3 >
/// then the result of doing a horizontal operation on A and B is
///   A horizontal-op B = < a0 op a1, a2 op a3, b0 op b1, b2 op b3 >.
/// In short, LHS and RHS are inspected to see if LHS op RHS is of the form
/// A horizontal-op B, for some already available A and B, and if so then LHS is
/// set to A, RHS to B, and the routine returns 'true'.
/// Note that the binary operation should have the property that if one of the
/// operands is UNDEF then the result is UNDEF.
static bool isHorizontalBinOp(SDValue &LHS, SDValue &RHS, bool IsCommutative) {
  // Look for the following pattern: if
  //   A = < float a0, float a1, float a2, float a3 >
  //   B = < float b0, float b1, float b2, float b3 >
  // and
  //   LHS = VECTOR_SHUFFLE A, B, <0, 2, 4, 6>
  //   RHS = VECTOR_SHUFFLE A, B, <1, 3, 5, 7>
  // then LHS op RHS = < a0 op a1, a2 op a3, b0 op b1, b2 op b3 >
  // which is A horizontal-op B.

  // At least one of the operands should be a vector shuffle.
  if (LHS.getOpcode() != ISD::VECTOR_SHUFFLE &&
      RHS.getOpcode() != ISD::VECTOR_SHUFFLE)
    return false;

  EVT VT = LHS.getValueType();

  assert((VT.is128BitVector() || VT.is256BitVector()) &&
         "Unsupported vector type for horizontal add/sub");

  // Handle 128 and 256-bit vector lengths. AVX defines horizontal add/sub to
  // operate independently on 128-bit lanes.
  unsigned NumElts = VT.getVectorNumElements();
  unsigned NumLanes = VT.getSizeInBits()/128;
  unsigned NumLaneElts = NumElts / NumLanes;
  assert((NumLaneElts % 2 == 0) &&
         "Vector type should have an even number of elements in each lane");
  unsigned HalfLaneElts = NumLaneElts/2;

  // View LHS in the form
  //   LHS = VECTOR_SHUFFLE A, B, LMask
  // If LHS is not a shuffle then pretend it is the shuffle
  //   LHS = VECTOR_SHUFFLE LHS, undef, <0, 1, ..., N-1>
  // NOTE: in what follows a default initialized SDValue represents an UNDEF of
  // type VT.
  SDValue A, B;
  SmallVector<int, 16> LMask(NumElts);
  if (LHS.getOpcode() == ISD::VECTOR_SHUFFLE) {
    if (LHS.getOperand(0).getOpcode() != ISD::UNDEF)
      A = LHS.getOperand(0);
    if (LHS.getOperand(1).getOpcode() != ISD::UNDEF)
      B = LHS.getOperand(1);
    ArrayRef<int> Mask = cast<ShuffleVectorSDNode>(LHS.getNode())->getMask();
    std::copy(Mask.begin(), Mask.end(), LMask.begin());
  } else {
    if (LHS.getOpcode() != ISD::UNDEF)
      A = LHS;
    for (unsigned i = 0; i != NumElts; ++i)
      LMask[i] = i;
  }

  // Likewise, view RHS in the form
  //   RHS = VECTOR_SHUFFLE C, D, RMask
  SDValue C, D;
  SmallVector<int, 16> RMask(NumElts);
  if (RHS.getOpcode() == ISD::VECTOR_SHUFFLE) {
    if (RHS.getOperand(0).getOpcode() != ISD::UNDEF)
      C = RHS.getOperand(0);
    if (RHS.getOperand(1).getOpcode() != ISD::UNDEF)
      D = RHS.getOperand(1);
    ArrayRef<int> Mask = cast<ShuffleVectorSDNode>(RHS.getNode())->getMask();
    std::copy(Mask.begin(), Mask.end(), RMask.begin());
  } else {
    if (RHS.getOpcode() != ISD::UNDEF)
      C = RHS;
    for (unsigned i = 0; i != NumElts; ++i)
      RMask[i] = i;
  }

  // Check that the shuffles are both shuffling the same vectors.
  if (!(A == C && B == D) && !(A == D && B == C))
    return false;

  // If everything is UNDEF then bail out: it would be better to fold to UNDEF.
  if (!A.getNode() && !B.getNode())
    return false;

  // If A and B occur in reverse order in RHS, then "swap" them (which means
  // rewriting the mask).
  if (A != C)
    CommuteVectorShuffleMask(RMask, NumElts);

  // At this point LHS and RHS are equivalent to
  //   LHS = VECTOR_SHUFFLE A, B, LMask
  //   RHS = VECTOR_SHUFFLE A, B, RMask
  // Check that the masks correspond to performing a horizontal operation.
  for (unsigned i = 0; i != NumElts; ++i) {
    int LIdx = LMask[i], RIdx = RMask[i];

    // Ignore any UNDEF components.
    if (LIdx < 0 || RIdx < 0 ||
        (!A.getNode() && (LIdx < (int)NumElts || RIdx < (int)NumElts)) ||
        (!B.getNode() && (LIdx >= (int)NumElts || RIdx >= (int)NumElts)))
      continue;

    // Check that successive elements are being operated on.  If not, this is
    // not a horizontal operation.
    unsigned Src = (i/HalfLaneElts) % 2; // each lane is split between srcs
    unsigned LaneStart = (i/NumLaneElts) * NumLaneElts;
    int Index = 2*(i%HalfLaneElts) + NumElts*Src + LaneStart;
    if (!(LIdx == Index && RIdx == Index + 1) &&
        !(IsCommutative && LIdx == Index + 1 && RIdx == Index))
      return false;
  }

  LHS = A.getNode() ? A : B; // If A is 'UNDEF', use B for it.
  RHS = B.getNode() ? B : A; // If B is 'UNDEF', use A for it.
  return true;
}

/// PerformFADDCombine - Do target-specific dag combines on floating point adds.
static SDValue PerformFADDCombine(SDNode *N, SelectionDAG &DAG,
                                  const X86Subtarget *Subtarget) {
  EVT VT = N->getValueType(0);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // Try to synthesize horizontal adds from adds of shuffles.
  if (((Subtarget->hasSSE3() && (VT == MVT::v4f32 || VT == MVT::v2f64)) ||
       (Subtarget->hasAVX() && (VT == MVT::v8f32 || VT == MVT::v4f64))) &&
      isHorizontalBinOp(LHS, RHS, true))
    return DAG.getNode(X86ISD::FHADD, N->getDebugLoc(), VT, LHS, RHS);
  return SDValue();
}

/// PerformFSUBCombine - Do target-specific dag combines on floating point subs.
static SDValue PerformFSUBCombine(SDNode *N, SelectionDAG &DAG,
                                  const X86Subtarget *Subtarget) {
  EVT VT = N->getValueType(0);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // Try to synthesize horizontal subs from subs of shuffles.
  if (((Subtarget->hasSSE3() && (VT == MVT::v4f32 || VT == MVT::v2f64)) ||
       (Subtarget->hasAVX() && (VT == MVT::v8f32 || VT == MVT::v4f64))) &&
      isHorizontalBinOp(LHS, RHS, false))
    return DAG.getNode(X86ISD::FHSUB, N->getDebugLoc(), VT, LHS, RHS);
  return SDValue();
}

/// PerformFORCombine - Do target-specific dag combines on X86ISD::FOR and
/// X86ISD::FXOR nodes.
static SDValue PerformFORCombine(SDNode *N, SelectionDAG &DAG) {
  assert(N->getOpcode() == X86ISD::FOR || N->getOpcode() == X86ISD::FXOR);
  // F[X]OR(0.0, x) -> x
  // F[X]OR(x, 0.0) -> x
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N->getOperand(0)))
    if (C->getValueAPF().isPosZero())
      return N->getOperand(1);
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N->getOperand(1)))
    if (C->getValueAPF().isPosZero())
      return N->getOperand(0);
  return SDValue();
}

/// PerformFMinFMaxCombine - Do target-specific dag combines on X86ISD::FMIN and
/// X86ISD::FMAX nodes.
static SDValue PerformFMinFMaxCombine(SDNode *N, SelectionDAG &DAG) {
  assert(N->getOpcode() == X86ISD::FMIN || N->getOpcode() == X86ISD::FMAX);

  // Only perform optimizations if UnsafeMath is used.
  if (!DAG.getTarget().Options.UnsafeFPMath)
    return SDValue();

  // If we run in unsafe-math mode, then convert the FMAX and FMIN nodes
  // into FMINC and FMAXC, which are Commutative operations.
  unsigned NewOp = 0;
  switch (N->getOpcode()) {
    default: llvm_unreachable("unknown opcode");
    case X86ISD::FMIN:  NewOp = X86ISD::FMINC; break;
    case X86ISD::FMAX:  NewOp = X86ISD::FMAXC; break;
  }

  return DAG.getNode(NewOp, N->getDebugLoc(), N->getValueType(0),
                     N->getOperand(0), N->getOperand(1));
}


/// PerformFANDCombine - Do target-specific dag combines on X86ISD::FAND nodes.
static SDValue PerformFANDCombine(SDNode *N, SelectionDAG &DAG) {
  // FAND(0.0, x) -> 0.0
  // FAND(x, 0.0) -> 0.0
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N->getOperand(0)))
    if (C->getValueAPF().isPosZero())
      return N->getOperand(0);
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N->getOperand(1)))
    if (C->getValueAPF().isPosZero())
      return N->getOperand(1);
  return SDValue();
}

static SDValue PerformBTCombine(SDNode *N,
                                SelectionDAG &DAG,
                                TargetLowering::DAGCombinerInfo &DCI) {
  // BT ignores high bits in the bit index operand.
  SDValue Op1 = N->getOperand(1);
  if (Op1.hasOneUse()) {
    unsigned BitWidth = Op1.getValueSizeInBits();
    APInt DemandedMask = APInt::getLowBitsSet(BitWidth, Log2_32(BitWidth));
    APInt KnownZero, KnownOne;
    TargetLowering::TargetLoweringOpt TLO(DAG, !DCI.isBeforeLegalize(),
                                          !DCI.isBeforeLegalizeOps());
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    if (TLO.ShrinkDemandedConstant(Op1, DemandedMask) ||
        TLI.SimplifyDemandedBits(Op1, DemandedMask, KnownZero, KnownOne, TLO))
      DCI.CommitTargetLoweringOpt(TLO);
  }
  return SDValue();
}

static SDValue PerformVZEXT_MOVLCombine(SDNode *N, SelectionDAG &DAG) {
  SDValue Op = N->getOperand(0);
  if (Op.getOpcode() == ISD::BITCAST)
    Op = Op.getOperand(0);
  EVT VT = N->getValueType(0), OpVT = Op.getValueType();
  if (Op.getOpcode() == X86ISD::VZEXT_LOAD &&
      VT.getVectorElementType().getSizeInBits() ==
      OpVT.getVectorElementType().getSizeInBits()) {
    return DAG.getNode(ISD::BITCAST, N->getDebugLoc(), VT, Op);
  }
  return SDValue();
}

static SDValue PerformSExtCombine(SDNode *N, SelectionDAG &DAG,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const X86Subtarget *Subtarget) {
  if (!DCI.isBeforeLegalizeOps())
    return SDValue();

  if (!Subtarget->hasAVX())
    return SDValue();

  EVT VT = N->getValueType(0);
  SDValue Op = N->getOperand(0);
  EVT OpVT = Op.getValueType();
  DebugLoc dl = N->getDebugLoc();

  if ((VT == MVT::v4i64 && OpVT == MVT::v4i32) ||
      (VT == MVT::v8i32 && OpVT == MVT::v8i16)) {

    if (Subtarget->hasAVX2())
      return DAG.getNode(X86ISD::VSEXT_MOVL, dl, VT, Op);

    // Optimize vectors in AVX mode
    // Sign extend  v8i16 to v8i32 and
    //              v4i32 to v4i64
    //
    // Divide input vector into two parts
    // for v4i32 the shuffle mask will be { 0, 1, -1, -1} {2, 3, -1, -1}
    // use vpmovsx instruction to extend v4i32 -> v2i64; v8i16 -> v4i32
    // concat the vectors to original VT

    unsigned NumElems = OpVT.getVectorNumElements();
    SDValue Undef = DAG.getUNDEF(OpVT);

    SmallVector<int,8> ShufMask1(NumElems, -1);
    for (unsigned i = 0; i != NumElems/2; ++i)
      ShufMask1[i] = i;

    SDValue OpLo = DAG.getVectorShuffle(OpVT, dl, Op, Undef, &ShufMask1[0]);

    SmallVector<int,8> ShufMask2(NumElems, -1);
    for (unsigned i = 0; i != NumElems/2; ++i)
      ShufMask2[i] = i + NumElems/2;

    SDValue OpHi = DAG.getVectorShuffle(OpVT, dl, Op, Undef, &ShufMask2[0]);

    EVT HalfVT = EVT::getVectorVT(*DAG.getContext(), VT.getScalarType(),
                                  VT.getVectorNumElements()/2);

    OpLo = DAG.getNode(X86ISD::VSEXT_MOVL, dl, HalfVT, OpLo);
    OpHi = DAG.getNode(X86ISD::VSEXT_MOVL, dl, HalfVT, OpHi);

    return DAG.getNode(ISD::CONCAT_VECTORS, dl, VT, OpLo, OpHi);
  }
  return SDValue();
}

static SDValue PerformFMACombine(SDNode *N, SelectionDAG &DAG,
                                 const X86Subtarget* Subtarget) {
  DebugLoc dl = N->getDebugLoc();
  EVT VT = N->getValueType(0);

  // Let legalize expand this if it isn't a legal type yet.
  if (!DAG.getTargetLoweringInfo().isTypeLegal(VT))
    return SDValue();

  EVT ScalarVT = VT.getScalarType();
  if ((ScalarVT != MVT::f32 && ScalarVT != MVT::f64) ||
      (!Subtarget->hasFMA() && !Subtarget->hasFMA4()))
    return SDValue();

  SDValue A = N->getOperand(0);
  SDValue B = N->getOperand(1);
  SDValue C = N->getOperand(2);

  bool NegA = (A.getOpcode() == ISD::FNEG);
  bool NegB = (B.getOpcode() == ISD::FNEG);
  bool NegC = (C.getOpcode() == ISD::FNEG);

  // Negative multiplication when NegA xor NegB
  bool NegMul = (NegA != NegB);
  if (NegA)
    A = A.getOperand(0);
  if (NegB)
    B = B.getOperand(0);
  if (NegC)
    C = C.getOperand(0);

  unsigned Opcode;
  if (!NegMul)
    Opcode = (!NegC) ? X86ISD::FMADD : X86ISD::FMSUB;
  else
    Opcode = (!NegC) ? X86ISD::FNMADD : X86ISD::FNMSUB;

  return DAG.getNode(Opcode, dl, VT, A, B, C);
}

static SDValue PerformZExtCombine(SDNode *N, SelectionDAG &DAG,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const X86Subtarget *Subtarget) {
  // (i32 zext (and (i8  x86isd::setcc_carry), 1)) ->
  //           (and (i32 x86isd::setcc_carry), 1)
  // This eliminates the zext. This transformation is necessary because
  // ISD::SETCC is always legalized to i8.
  DebugLoc dl = N->getDebugLoc();
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  if (N0.getOpcode() == ISD::AND &&
      N0.hasOneUse() &&
      N0.getOperand(0).hasOneUse()) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getOpcode() != X86ISD::SETCC_CARRY)
      return SDValue();
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!C || C->getZExtValue() != 1)
      return SDValue();
    return DAG.getNode(ISD::AND, dl, VT,
                       DAG.getNode(X86ISD::SETCC_CARRY, dl, VT,
                                   N00.getOperand(0), N00.getOperand(1)),
                       DAG.getConstant(1, VT));
  }

  // Optimize vectors in AVX mode:
  //
  //   v8i16 -> v8i32
  //   Use vpunpcklwd for 4 lower elements  v8i16 -> v4i32.
  //   Use vpunpckhwd for 4 upper elements  v8i16 -> v4i32.
  //   Concat upper and lower parts.
  //
  //   v4i32 -> v4i64
  //   Use vpunpckldq for 4 lower elements  v4i32 -> v2i64.
  //   Use vpunpckhdq for 4 upper elements  v4i32 -> v2i64.
  //   Concat upper and lower parts.
  //
  if (!DCI.isBeforeLegalizeOps())
    return SDValue();

  if (!Subtarget->hasAVX())
    return SDValue();

  if (((VT == MVT::v8i32) && (OpVT == MVT::v8i16)) ||
      ((VT == MVT::v4i64) && (OpVT == MVT::v4i32)))  {

    if (Subtarget->hasAVX2())
      return DAG.getNode(X86ISD::VZEXT_MOVL, dl, VT, N0);

    SDValue ZeroVec = getZeroVector(OpVT, Subtarget, DAG, dl);
    SDValue OpLo = getUnpackl(DAG, dl, OpVT, N0, ZeroVec);
    SDValue OpHi = getUnpackh(DAG, dl, OpVT, N0, ZeroVec);

    EVT HVT = EVT::getVectorVT(*DAG.getContext(), VT.getVectorElementType(),
                               VT.getVectorNumElements()/2);

    OpLo = DAG.getNode(ISD::BITCAST, dl, HVT, OpLo);
    OpHi = DAG.getNode(ISD::BITCAST, dl, HVT, OpHi);

    return DAG.getNode(ISD::CONCAT_VECTORS, dl, VT, OpLo, OpHi);
  }

  return SDValue();
}

// Optimize x == -y --> x+y == 0
//          x != -y --> x+y != 0
static SDValue PerformISDSETCCCombine(SDNode *N, SelectionDAG &DAG) {
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  if ((CC == ISD::SETNE || CC == ISD::SETEQ) && LHS.getOpcode() == ISD::SUB)
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(LHS.getOperand(0)))
      if (C->getAPIntValue() == 0 && LHS.hasOneUse()) {
        SDValue addV = DAG.getNode(ISD::ADD, N->getDebugLoc(),
                                   LHS.getValueType(), RHS, LHS.getOperand(1));
        return DAG.getSetCC(N->getDebugLoc(), N->getValueType(0),
                            addV, DAG.getConstant(0, addV.getValueType()), CC);
      }
  if ((CC == ISD::SETNE || CC == ISD::SETEQ) && RHS.getOpcode() == ISD::SUB)
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(RHS.getOperand(0)))
      if (C->getAPIntValue() == 0 && RHS.hasOneUse()) {
        SDValue addV = DAG.getNode(ISD::ADD, N->getDebugLoc(),
                                   RHS.getValueType(), LHS, RHS.getOperand(1));
        return DAG.getSetCC(N->getDebugLoc(), N->getValueType(0),
                            addV, DAG.getConstant(0, addV.getValueType()), CC);
      }
  return SDValue();
}

// Helper function of PerformSETCCCombine. It is to materialize "setb reg" 
// as "sbb reg,reg", since it can be extended without zext and produces 
// an all-ones bit which is more useful than 0/1 in some cases.
static SDValue MaterializeSETB(DebugLoc DL, SDValue EFLAGS, SelectionDAG &DAG) {
  return DAG.getNode(ISD::AND, DL, MVT::i8,
                     DAG.getNode(X86ISD::SETCC_CARRY, DL, MVT::i8,
                                 DAG.getConstant(X86::COND_B, MVT::i8), EFLAGS),
                     DAG.getConstant(1, MVT::i8));
}

// Optimize  RES = X86ISD::SETCC CONDCODE, EFLAG_INPUT
static SDValue PerformSETCCCombine(SDNode *N, SelectionDAG &DAG,
                                   TargetLowering::DAGCombinerInfo &DCI,
                                   const X86Subtarget *Subtarget) {
  DebugLoc DL = N->getDebugLoc();
  X86::CondCode CC = X86::CondCode(N->getConstantOperandVal(0));
  SDValue EFLAGS = N->getOperand(1);

  if (CC == X86::COND_A) {
    // Try to convert COND_A into COND_B in an attempt to facilitate 
    // materializing "setb reg".
    //
    // Do not flip "e > c", where "c" is a constant, because Cmp instruction
    // cannot take an immediate as its first operand.
    //
    if (EFLAGS.getOpcode() == X86ISD::SUB && EFLAGS.hasOneUse() && 
        EFLAGS.getValueType().isInteger() &&
        !isa<ConstantSDNode>(EFLAGS.getOperand(1))) {
      SDValue NewSub = DAG.getNode(X86ISD::SUB, EFLAGS.getDebugLoc(),
                                   EFLAGS.getNode()->getVTList(),
                                   EFLAGS.getOperand(1), EFLAGS.getOperand(0));
      SDValue NewEFLAGS = SDValue(NewSub.getNode(), EFLAGS.getResNo());
      return MaterializeSETB(DL, NewEFLAGS, DAG);
    }
  }

  // Materialize "setb reg" as "sbb reg,reg", since it can be extended without
  // a zext and produces an all-ones bit which is more useful than 0/1 in some
  // cases.
  if (CC == X86::COND_B)
    return MaterializeSETB(DL, EFLAGS, DAG);

  SDValue Flags;

  Flags = checkBoolTestSetCCCombine(EFLAGS, CC);
  if (Flags.getNode()) {
    SDValue Cond = DAG.getConstant(CC, MVT::i8);
    return DAG.getNode(X86ISD::SETCC, DL, N->getVTList(), Cond, Flags);
  }

  return SDValue();
}

// Optimize branch condition evaluation.
//
static SDValue PerformBrCondCombine(SDNode *N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const X86Subtarget *Subtarget) {
  DebugLoc DL = N->getDebugLoc();
  SDValue Chain = N->getOperand(0);
  SDValue Dest = N->getOperand(1);
  SDValue EFLAGS = N->getOperand(3);
  X86::CondCode CC = X86::CondCode(N->getConstantOperandVal(2));

  SDValue Flags;

  Flags = checkBoolTestSetCCCombine(EFLAGS, CC);
  if (Flags.getNode()) {
    SDValue Cond = DAG.getConstant(CC, MVT::i8);
    return DAG.getNode(X86ISD::BRCOND, DL, N->getVTList(), Chain, Dest, Cond,
                       Flags);
  }

  return SDValue();
}

static SDValue PerformSINT_TO_FPCombine(SDNode *N, SelectionDAG &DAG,
                                        const X86TargetLowering *XTLI) {
  SDValue Op0 = N->getOperand(0);
  EVT InVT = Op0->getValueType(0);

  // SINT_TO_FP(v4i8) -> SINT_TO_FP(SEXT(v4i8 to v4i32))
  if (InVT == MVT::v8i8 || InVT == MVT::v4i8) {
    DebugLoc dl = N->getDebugLoc();
    MVT DstVT = InVT == MVT::v4i8 ? MVT::v4i32 : MVT::v8i32;
    SDValue P = DAG.getNode(ISD::SIGN_EXTEND, dl, DstVT, Op0);
    return DAG.getNode(ISD::SINT_TO_FP, dl, N->getValueType(0), P);
  }

  // Transform (SINT_TO_FP (i64 ...)) into an x87 operation if we have
  // a 32-bit target where SSE doesn't support i64->FP operations.
  if (Op0.getOpcode() == ISD::LOAD) {
    LoadSDNode *Ld = cast<LoadSDNode>(Op0.getNode());
    EVT VT = Ld->getValueType(0);
    if (!Ld->isVolatile() && !N->getValueType(0).isVector() &&
        ISD::isNON_EXTLoad(Op0.getNode()) && Op0.hasOneUse() &&
        !XTLI->getSubtarget()->is64Bit() &&
        !DAG.getTargetLoweringInfo().isTypeLegal(VT)) {
      SDValue FILDChain = XTLI->BuildFILD(SDValue(N, 0), Ld->getValueType(0),
                                          Ld->getChain(), Op0, DAG);
      DAG.ReplaceAllUsesOfValueWith(Op0.getValue(1), FILDChain.getValue(1));
      return FILDChain;
    }
  }
  return SDValue();
}

// Optimize RES, EFLAGS = X86ISD::ADC LHS, RHS, EFLAGS
static SDValue PerformADCCombine(SDNode *N, SelectionDAG &DAG,
                                 X86TargetLowering::DAGCombinerInfo &DCI) {
  // If the LHS and RHS of the ADC node are zero, then it can't overflow and
  // the result is either zero or one (depending on the input carry bit).
  // Strength reduce this down to a "set on carry" aka SETCC_CARRY&1.
  if (X86::isZeroNode(N->getOperand(0)) &&
      X86::isZeroNode(N->getOperand(1)) &&
      // We don't have a good way to replace an EFLAGS use, so only do this when
      // dead right now.
      SDValue(N, 1).use_empty()) {
    DebugLoc DL = N->getDebugLoc();
    EVT VT = N->getValueType(0);
    SDValue CarryOut = DAG.getConstant(0, N->getValueType(1));
    SDValue Res1 = DAG.getNode(ISD::AND, DL, VT,
                               DAG.getNode(X86ISD::SETCC_CARRY, DL, VT,
                                           DAG.getConstant(X86::COND_B,MVT::i8),
                                           N->getOperand(2)),
                               DAG.getConstant(1, VT));
    return DCI.CombineTo(N, Res1, CarryOut);
  }

  return SDValue();
}

// fold (add Y, (sete  X, 0)) -> adc  0, Y
//      (add Y, (setne X, 0)) -> sbb -1, Y
//      (sub (sete  X, 0), Y) -> sbb  0, Y
//      (sub (setne X, 0), Y) -> adc -1, Y
static SDValue OptimizeConditionalInDecrement(SDNode *N, SelectionDAG &DAG) {
  DebugLoc DL = N->getDebugLoc();

  // Look through ZExts.
  SDValue Ext = N->getOperand(N->getOpcode() == ISD::SUB ? 1 : 0);
  if (Ext.getOpcode() != ISD::ZERO_EXTEND || !Ext.hasOneUse())
    return SDValue();

  SDValue SetCC = Ext.getOperand(0);
  if (SetCC.getOpcode() != X86ISD::SETCC || !SetCC.hasOneUse())
    return SDValue();

  X86::CondCode CC = (X86::CondCode)SetCC.getConstantOperandVal(0);
  if (CC != X86::COND_E && CC != X86::COND_NE)
    return SDValue();

  SDValue Cmp = SetCC.getOperand(1);
  if (Cmp.getOpcode() != X86ISD::CMP || !Cmp.hasOneUse() ||
      !X86::isZeroNode(Cmp.getOperand(1)) ||
      !Cmp.getOperand(0).getValueType().isInteger())
    return SDValue();

  SDValue CmpOp0 = Cmp.getOperand(0);
  SDValue NewCmp = DAG.getNode(X86ISD::CMP, DL, MVT::i32, CmpOp0,
                               DAG.getConstant(1, CmpOp0.getValueType()));

  SDValue OtherVal = N->getOperand(N->getOpcode() == ISD::SUB ? 0 : 1);
  if (CC == X86::COND_NE)
    return DAG.getNode(N->getOpcode() == ISD::SUB ? X86ISD::ADC : X86ISD::SBB,
                       DL, OtherVal.getValueType(), OtherVal,
                       DAG.getConstant(-1ULL, OtherVal.getValueType()), NewCmp);
  return DAG.getNode(N->getOpcode() == ISD::SUB ? X86ISD::SBB : X86ISD::ADC,
                     DL, OtherVal.getValueType(), OtherVal,
                     DAG.getConstant(0, OtherVal.getValueType()), NewCmp);
}

/// PerformADDCombine - Do target-specific dag combines on integer adds.
static SDValue PerformAddCombine(SDNode *N, SelectionDAG &DAG,
                                 const X86Subtarget *Subtarget) {
  EVT VT = N->getValueType(0);
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  // Try to synthesize horizontal adds from adds of shuffles.
  if (((Subtarget->hasSSSE3() && (VT == MVT::v8i16 || VT == MVT::v4i32)) ||
       (Subtarget->hasAVX2() && (VT == MVT::v16i16 || VT == MVT::v8i32))) &&
      isHorizontalBinOp(Op0, Op1, true))
    return DAG.getNode(X86ISD::HADD, N->getDebugLoc(), VT, Op0, Op1);

  return OptimizeConditionalInDecrement(N, DAG);
}

static SDValue PerformSubCombine(SDNode *N, SelectionDAG &DAG,
                                 const X86Subtarget *Subtarget) {
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  // X86 can't encode an immediate LHS of a sub. See if we can push the
  // negation into a preceding instruction.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op0)) {
    // If the RHS of the sub is a XOR with one use and a constant, invert the
    // immediate. Then add one to the LHS of the sub so we can turn
    // X-Y -> X+~Y+1, saving one register.
    if (Op1->hasOneUse() && Op1.getOpcode() == ISD::XOR &&
        isa<ConstantSDNode>(Op1.getOperand(1))) {
      APInt XorC = cast<ConstantSDNode>(Op1.getOperand(1))->getAPIntValue();
      EVT VT = Op0.getValueType();
      SDValue NewXor = DAG.getNode(ISD::XOR, Op1.getDebugLoc(), VT,
                                   Op1.getOperand(0),
                                   DAG.getConstant(~XorC, VT));
      return DAG.getNode(ISD::ADD, N->getDebugLoc(), VT, NewXor,
                         DAG.getConstant(C->getAPIntValue()+1, VT));
    }
  }

  // Try to synthesize horizontal adds from adds of shuffles.
  EVT VT = N->getValueType(0);
  if (((Subtarget->hasSSSE3() && (VT == MVT::v8i16 || VT == MVT::v4i32)) ||
       (Subtarget->hasAVX2() && (VT == MVT::v16i16 || VT == MVT::v8i32))) &&
      isHorizontalBinOp(Op0, Op1, true))
    return DAG.getNode(X86ISD::HSUB, N->getDebugLoc(), VT, Op0, Op1);

  return OptimizeConditionalInDecrement(N, DAG);
}

/// performVZEXTCombine - Performs build vector combines
static SDValue performVZEXTCombine(SDNode *N, SelectionDAG &DAG,
                                        TargetLowering::DAGCombinerInfo &DCI,
                                        const X86Subtarget *Subtarget) {
  // (vzext (bitcast (vzext (x)) -> (vzext x)
  SDValue In = N->getOperand(0);
  while (In.getOpcode() == ISD::BITCAST)
    In = In.getOperand(0);

  if (In.getOpcode() != X86ISD::VZEXT)
    return SDValue();

  return DAG.getNode(X86ISD::VZEXT, N->getDebugLoc(), N->getValueType(0), In.getOperand(0));
}

SDValue X86TargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default: break;
  case ISD::EXTRACT_VECTOR_ELT:
    return PerformEXTRACT_VECTOR_ELTCombine(N, DAG, DCI);
  case ISD::VSELECT:
  case ISD::SELECT:         return PerformSELECTCombine(N, DAG, DCI, Subtarget);
  case X86ISD::CMOV:        return PerformCMOVCombine(N, DAG, DCI, Subtarget);
  case ISD::ADD:            return PerformAddCombine(N, DAG, Subtarget);
  case ISD::SUB:            return PerformSubCombine(N, DAG, Subtarget);
  case X86ISD::ADC:         return PerformADCCombine(N, DAG, DCI);
  case ISD::MUL:            return PerformMulCombine(N, DAG, DCI);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:            return PerformShiftCombine(N, DAG, DCI, Subtarget);
  case ISD::AND:            return PerformAndCombine(N, DAG, DCI, Subtarget);
  case ISD::OR:             return PerformOrCombine(N, DAG, DCI, Subtarget);
  case ISD::XOR:            return PerformXorCombine(N, DAG, DCI, Subtarget);
  case ISD::LOAD:           return PerformLOADCombine(N, DAG, DCI, Subtarget);
  case ISD::STORE:          return PerformSTORECombine(N, DAG, Subtarget);
  case ISD::SINT_TO_FP:     return PerformSINT_TO_FPCombine(N, DAG, this);
  case ISD::FADD:           return PerformFADDCombine(N, DAG, Subtarget);
  case ISD::FSUB:           return PerformFSUBCombine(N, DAG, Subtarget);
  case X86ISD::FXOR:
  case X86ISD::FOR:         return PerformFORCombine(N, DAG);
  case X86ISD::FMIN:
  case X86ISD::FMAX:        return PerformFMinFMaxCombine(N, DAG);
  case X86ISD::FAND:        return PerformFANDCombine(N, DAG);
  case X86ISD::BT:          return PerformBTCombine(N, DAG, DCI);
  case X86ISD::VZEXT_MOVL:  return PerformVZEXT_MOVLCombine(N, DAG);
  case ISD::ANY_EXTEND:
  case ISD::ZERO_EXTEND:    return PerformZExtCombine(N, DAG, DCI, Subtarget);
  case ISD::SIGN_EXTEND:    return PerformSExtCombine(N, DAG, DCI, Subtarget);
  case ISD::TRUNCATE:       return PerformTruncateCombine(N, DAG,DCI,Subtarget);
  case ISD::SETCC:          return PerformISDSETCCCombine(N, DAG);
  case X86ISD::SETCC:       return PerformSETCCCombine(N, DAG, DCI, Subtarget);
  case X86ISD::BRCOND:      return PerformBrCondCombine(N, DAG, DCI, Subtarget);
  case X86ISD::VZEXT:       return performVZEXTCombine(N, DAG, DCI, Subtarget);
  case X86ISD::SHUFP:       // Handle all target specific shuffles
  case X86ISD::PALIGN:
  case X86ISD::UNPCKH:
  case X86ISD::UNPCKL:
  case X86ISD::MOVHLPS:
  case X86ISD::MOVLHPS:
  case X86ISD::PSHUFD:
  case X86ISD::PSHUFHW:
  case X86ISD::PSHUFLW:
  case X86ISD::MOVSS:
  case X86ISD::MOVSD:
  case X86ISD::VPERMILP:
  case X86ISD::VPERM2X128:
  case ISD::VECTOR_SHUFFLE: return PerformShuffleCombine(N, DAG, DCI,Subtarget);
  case ISD::FMA:            return PerformFMACombine(N, DAG, Subtarget);
  }

  return SDValue();
}

/// isTypeDesirableForOp - Return true if the target has native support for
/// the specified value type and it is 'desirable' to use the type for the
/// given node type. e.g. On x86 i16 is legal, but undesirable since i16
/// instruction encodings are longer and some i16 instructions are slow.
bool X86TargetLowering::isTypeDesirableForOp(unsigned Opc, EVT VT) const {
  if (!isTypeLegal(VT))
    return false;
  if (VT != MVT::i16)
    return true;

  switch (Opc) {
  default:
    return true;
  case ISD::LOAD:
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SUB:
  case ISD::ADD:
  case ISD::MUL:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    return false;
  }
}

/// IsDesirableToPromoteOp - This method query the target whether it is
/// beneficial for dag combiner to promote the specified node. If true, it
/// should return the desired promotion type by reference.
bool X86TargetLowering::IsDesirableToPromoteOp(SDValue Op, EVT &PVT) const {
  EVT VT = Op.getValueType();
  if (VT != MVT::i16)
    return false;

  bool Promote = false;
  bool Commute = false;
  switch (Op.getOpcode()) {
  default: break;
  case ISD::LOAD: {
    LoadSDNode *LD = cast<LoadSDNode>(Op);
    // If the non-extending load has a single use and it's not live out, then it
    // might be folded.
    if (LD->getExtensionType() == ISD::NON_EXTLOAD /*&&
                                                     Op.hasOneUse()*/) {
      for (SDNode::use_iterator UI = Op.getNode()->use_begin(),
             UE = Op.getNode()->use_end(); UI != UE; ++UI) {
        // The only case where we'd want to promote LOAD (rather then it being
        // promoted as an operand is when it's only use is liveout.
        if (UI->getOpcode() != ISD::CopyToReg)
          return false;
      }
    }
    Promote = true;
    break;
  }
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
    Promote = true;
    break;
  case ISD::SHL:
  case ISD::SRL: {
    SDValue N0 = Op.getOperand(0);
    // Look out for (store (shl (load), x)).
    if (MayFoldLoad(N0) && MayFoldIntoStore(Op))
      return false;
    Promote = true;
    break;
  }
  case ISD::ADD:
  case ISD::MUL:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    Commute = true;
    // fallthrough
  case ISD::SUB: {
    SDValue N0 = Op.getOperand(0);
    SDValue N1 = Op.getOperand(1);
    if (!Commute && MayFoldLoad(N1))
      return false;
    // Avoid disabling potential load folding opportunities.
    if (MayFoldLoad(N0) && (!isa<ConstantSDNode>(N1) || MayFoldIntoStore(Op)))
      return false;
    if (MayFoldLoad(N1) && (!isa<ConstantSDNode>(N0) || MayFoldIntoStore(Op)))
      return false;
    Promote = true;
  }
  }

  PVT = MVT::i32;
  return Promote;
}

//===----------------------------------------------------------------------===//
//                           X86 Inline Assembly Support
//===----------------------------------------------------------------------===//

namespace {
  // Helper to match a string separated by whitespace.
  bool matchAsmImpl(StringRef s, ArrayRef<const StringRef *> args) {
    s = s.substr(s.find_first_not_of(" \t")); // Skip leading whitespace.

    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      StringRef piece(*args[i]);
      if (!s.startswith(piece)) // Check if the piece matches.
        return false;

      s = s.substr(piece.size());
      StringRef::size_type pos = s.find_first_not_of(" \t");
      if (pos == 0) // We matched a prefix.
        return false;

      s = s.substr(pos);
    }

    return s.empty();
  }
  const VariadicFunction1<bool, StringRef, StringRef, matchAsmImpl> matchAsm={};
}

bool X86TargetLowering::ExpandInlineAsm(CallInst *CI) const {
  InlineAsm *IA = cast<InlineAsm>(CI->getCalledValue());

  std::string AsmStr = IA->getAsmString();

  IntegerType *Ty = dyn_cast<IntegerType>(CI->getType());
  if (!Ty || Ty->getBitWidth() % 16 != 0)
    return false;

  // TODO: should remove alternatives from the asmstring: "foo {a|b}" -> "foo a"
  SmallVector<StringRef, 4> AsmPieces;
  SplitString(AsmStr, AsmPieces, ";\n");

  switch (AsmPieces.size()) {
  default: return false;
  case 1:
    // FIXME: this should verify that we are targeting a 486 or better.  If not,
    // we will turn this bswap into something that will be lowered to logical
    // ops instead of emitting the bswap asm.  For now, we don't support 486 or
    // lower so don't worry about this.
    // bswap $0
    if (matchAsm(AsmPieces[0], "bswap", "$0") ||
        matchAsm(AsmPieces[0], "bswapl", "$0") ||
        matchAsm(AsmPieces[0], "bswapq", "$0") ||
        matchAsm(AsmPieces[0], "bswap", "${0:q}") ||
        matchAsm(AsmPieces[0], "bswapl", "${0:q}") ||
        matchAsm(AsmPieces[0], "bswapq", "${0:q}")) {
      // No need to check constraints, nothing other than the equivalent of
      // "=r,0" would be valid here.
      return IntrinsicLowering::LowerToByteSwap(CI);
    }

    // rorw $$8, ${0:w}  -->  llvm.bswap.i16
    if (CI->getType()->isIntegerTy(16) &&
        IA->getConstraintString().compare(0, 5, "=r,0,") == 0 &&
        (matchAsm(AsmPieces[0], "rorw", "$$8,", "${0:w}") ||
         matchAsm(AsmPieces[0], "rolw", "$$8,", "${0:w}"))) {
      AsmPieces.clear();
      const std::string &ConstraintsStr = IA->getConstraintString();
      SplitString(StringRef(ConstraintsStr).substr(5), AsmPieces, ",");
      std::sort(AsmPieces.begin(), AsmPieces.end());
      if (AsmPieces.size() == 4 &&
          AsmPieces[0] == "~{cc}" &&
          AsmPieces[1] == "~{dirflag}" &&
          AsmPieces[2] == "~{flags}" &&
          AsmPieces[3] == "~{fpsr}")
      return IntrinsicLowering::LowerToByteSwap(CI);
    }
    break;
  case 3:
    if (CI->getType()->isIntegerTy(32) &&
        IA->getConstraintString().compare(0, 5, "=r,0,") == 0 &&
        matchAsm(AsmPieces[0], "rorw", "$$8,", "${0:w}") &&
        matchAsm(AsmPieces[1], "rorl", "$$16,", "$0") &&
        matchAsm(AsmPieces[2], "rorw", "$$8,", "${0:w}")) {
      AsmPieces.clear();
      const std::string &ConstraintsStr = IA->getConstraintString();
      SplitString(StringRef(ConstraintsStr).substr(5), AsmPieces, ",");
      std::sort(AsmPieces.begin(), AsmPieces.end());
      if (AsmPieces.size() == 4 &&
          AsmPieces[0] == "~{cc}" &&
          AsmPieces[1] == "~{dirflag}" &&
          AsmPieces[2] == "~{flags}" &&
          AsmPieces[3] == "~{fpsr}")
        return IntrinsicLowering::LowerToByteSwap(CI);
    }

    if (CI->getType()->isIntegerTy(64)) {
      InlineAsm::ConstraintInfoVector Constraints = IA->ParseConstraints();
      if (Constraints.size() >= 2 &&
          Constraints[0].Codes.size() == 1 && Constraints[0].Codes[0] == "A" &&
          Constraints[1].Codes.size() == 1 && Constraints[1].Codes[0] == "0") {
        // bswap %eax / bswap %edx / xchgl %eax, %edx  -> llvm.bswap.i64
        if (matchAsm(AsmPieces[0], "bswap", "%eax") &&
            matchAsm(AsmPieces[1], "bswap", "%edx") &&
            matchAsm(AsmPieces[2], "xchgl", "%eax,", "%edx"))
          return IntrinsicLowering::LowerToByteSwap(CI);
      }
    }
    break;
  }
  return false;
}



/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
X86TargetLowering::ConstraintType
X86TargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'R':
    case 'q':
    case 'Q':
    case 'f':
    case 't':
    case 'u':
    case 'y':
    case 'x':
    case 'Y':
    case 'l':
      return C_RegisterClass;
    case 'a':
    case 'b':
    case 'c':
    case 'd':
    case 'S':
    case 'D':
    case 'A':
      return C_Register;
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'G':
    case 'C':
    case 'e':
    case 'Z':
      return C_Other;
    default:
      break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
  X86TargetLowering::getSingleConstraintMatchWeight(
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
  case 'R':
  case 'q':
  case 'Q':
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'S':
  case 'D':
  case 'A':
    if (CallOperandVal->getType()->isIntegerTy())
      weight = CW_SpecificReg;
    break;
  case 'f':
  case 't':
  case 'u':
      if (type->isFloatingPointTy())
        weight = CW_SpecificReg;
      break;
  case 'y':
      if (type->isX86_MMXTy() && Subtarget->hasMMX())
        weight = CW_SpecificReg;
      break;
  case 'x':
  case 'Y':
    if (((type->getPrimitiveSizeInBits() == 128) && Subtarget->hasSSE1()) ||
        ((type->getPrimitiveSizeInBits() == 256) && Subtarget->hasAVX()))
      weight = CW_Register;
    break;
  case 'I':
    if (ConstantInt *C = dyn_cast<ConstantInt>(info.CallOperandVal)) {
      if (C->getZExtValue() <= 31)
        weight = CW_Constant;
    }
    break;
  case 'J':
    if (ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (C->getZExtValue() <= 63)
        weight = CW_Constant;
    }
    break;
  case 'K':
    if (ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if ((C->getSExtValue() >= -0x80) && (C->getSExtValue() <= 0x7f))
        weight = CW_Constant;
    }
    break;
  case 'L':
    if (ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if ((C->getZExtValue() == 0xff) || (C->getZExtValue() == 0xffff))
        weight = CW_Constant;
    }
    break;
  case 'M':
    if (ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (C->getZExtValue() <= 3)
        weight = CW_Constant;
    }
    break;
  case 'N':
    if (ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (C->getZExtValue() <= 0xff)
        weight = CW_Constant;
    }
    break;
  case 'G':
  case 'C':
    if (dyn_cast<ConstantFP>(CallOperandVal)) {
      weight = CW_Constant;
    }
    break;
  case 'e':
    if (ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if ((C->getSExtValue() >= -0x80000000LL) &&
          (C->getSExtValue() <= 0x7fffffffLL))
        weight = CW_Constant;
    }
    break;
  case 'Z':
    if (ConstantInt *C = dyn_cast<ConstantInt>(CallOperandVal)) {
      if (C->getZExtValue() <= 0xffffffff)
        weight = CW_Constant;
    }
    break;
  }
  return weight;
}

/// LowerXConstraint - try to replace an X constraint, which matches anything,
/// with another that has more specific requirements based on the type of the
/// corresponding operand.
const char *X86TargetLowering::
LowerXConstraint(EVT ConstraintVT) const {
  // FP X constraints get lowered to SSE1/2 registers if available, otherwise
  // 'f' like normal targets.
  if (ConstraintVT.isFloatingPoint()) {
    if (Subtarget->hasSSE2())
      return "Y";
    if (Subtarget->hasSSE1())
      return "x";
  }

  return TargetLowering::LowerXConstraint(ConstraintVT);
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void X86TargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                     std::string &Constraint,
                                                     std::vector<SDValue>&Ops,
                                                     SelectionDAG &DAG) const {
  SDValue Result(0, 0);

  // Only support length 1 constraints for now.
  if (Constraint.length() > 1) return;

  char ConstraintLetter = Constraint[0];
  switch (ConstraintLetter) {
  default: break;
  case 'I':
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      if (C->getZExtValue() <= 31) {
        Result = DAG.getTargetConstant(C->getZExtValue(), Op.getValueType());
        break;
      }
    }
    return;
  case 'J':
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      if (C->getZExtValue() <= 63) {
        Result = DAG.getTargetConstant(C->getZExtValue(), Op.getValueType());
        break;
      }
    }
    return;
  case 'K':
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      if (isInt<8>(C->getSExtValue())) {
        Result = DAG.getTargetConstant(C->getZExtValue(), Op.getValueType());
        break;
      }
    }
    return;
  case 'N':
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      if (C->getZExtValue() <= 255) {
        Result = DAG.getTargetConstant(C->getZExtValue(), Op.getValueType());
        break;
      }
    }
    return;
  case 'e': {
    // 32-bit signed value
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      if (ConstantInt::isValueValidForType(Type::getInt32Ty(*DAG.getContext()),
                                           C->getSExtValue())) {
        // Widen to 64 bits here to get it sign extended.
        Result = DAG.getTargetConstant(C->getSExtValue(), MVT::i64);
        break;
      }
    // FIXME gcc accepts some relocatable values here too, but only in certain
    // memory models; it's complicated.
    }
    return;
  }
  case 'Z': {
    // 32-bit unsigned value
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      if (ConstantInt::isValueValidForType(Type::getInt32Ty(*DAG.getContext()),
                                           C->getZExtValue())) {
        Result = DAG.getTargetConstant(C->getZExtValue(), Op.getValueType());
        break;
      }
    }
    // FIXME gcc accepts some relocatable values here too, but only in certain
    // memory models; it's complicated.
    return;
  }
  case 'i': {
    // Literal immediates are always ok.
    if (ConstantSDNode *CST = dyn_cast<ConstantSDNode>(Op)) {
      // Widen to 64 bits here to get it sign extended.
      Result = DAG.getTargetConstant(CST->getSExtValue(), MVT::i64);
      break;
    }

    // In any sort of PIC mode addresses need to be computed at runtime by
    // adding in a register or some sort of table lookup.  These can't
    // be used as immediates.
    if (Subtarget->isPICStyleGOT() || Subtarget->isPICStyleStubPIC())
      return;

    // If we are in non-pic codegen mode, we allow the address of a global (with
    // an optional displacement) to be used with 'i'.
    GlobalAddressSDNode *GA = 0;
    int64_t Offset = 0;

    // Match either (GA), (GA+C), (GA+C1+C2), etc.
    while (1) {
      if ((GA = dyn_cast<GlobalAddressSDNode>(Op))) {
        Offset += GA->getOffset();
        break;
      } else if (Op.getOpcode() == ISD::ADD) {
        if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
          Offset += C->getZExtValue();
          Op = Op.getOperand(0);
          continue;
        }
      } else if (Op.getOpcode() == ISD::SUB) {
        if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
          Offset += -C->getZExtValue();
          Op = Op.getOperand(0);
          continue;
        }
      }

      // Otherwise, this isn't something we can handle, reject it.
      return;
    }

    const GlobalValue *GV = GA->getGlobal();
    // If we require an extra load to get this address, as in PIC mode, we
    // can't accept it.
    if (isGlobalStubReference(Subtarget->ClassifyGlobalReference(GV,
                                                        getTargetMachine())))
      return;

    Result = DAG.getTargetGlobalAddress(GV, Op.getDebugLoc(),
                                        GA->getValueType(0), Offset);
    break;
  }
  }

  if (Result.getNode()) {
    Ops.push_back(Result);
    return;
  }
  return TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

std::pair<unsigned, const TargetRegisterClass*>
X86TargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                EVT VT) const {
  // First, see if this is a constraint that directly corresponds to an LLVM
  // register class.
  if (Constraint.size() == 1) {
    // GCC Constraint Letters
    switch (Constraint[0]) {
    default: break;
      // TODO: Slight differences here in allocation order and leaving
      // RIP in the class. Do they matter any more here than they do
      // in the normal allocation?
    case 'q':   // GENERAL_REGS in 64-bit mode, Q_REGS in 32-bit mode.
      if (Subtarget->is64Bit()) {
        if (VT == MVT::i32 || VT == MVT::f32)
          return std::make_pair(0U, &X86::GR32RegClass);
        if (VT == MVT::i16)
          return std::make_pair(0U, &X86::GR16RegClass);
        if (VT == MVT::i8 || VT == MVT::i1)
          return std::make_pair(0U, &X86::GR8RegClass);
        if (VT == MVT::i64 || VT == MVT::f64)
          return std::make_pair(0U, &X86::GR64RegClass);
        break;
      }
      // 32-bit fallthrough
    case 'Q':   // Q_REGS
      if (VT == MVT::i32 || VT == MVT::f32)
        return std::make_pair(0U, &X86::GR32_ABCDRegClass);
      if (VT == MVT::i16)
        return std::make_pair(0U, &X86::GR16_ABCDRegClass);
      if (VT == MVT::i8 || VT == MVT::i1)
        return std::make_pair(0U, &X86::GR8_ABCD_LRegClass);
      if (VT == MVT::i64)
        return std::make_pair(0U, &X86::GR64_ABCDRegClass);
      break;
    case 'r':   // GENERAL_REGS
    case 'l':   // INDEX_REGS
      if (VT == MVT::i8 || VT == MVT::i1)
        return std::make_pair(0U, &X86::GR8RegClass);
      if (VT == MVT::i16)
        return std::make_pair(0U, &X86::GR16RegClass);
      if (VT == MVT::i32 || VT == MVT::f32 || !Subtarget->is64Bit())
        return std::make_pair(0U, &X86::GR32RegClass);
      return std::make_pair(0U, &X86::GR64RegClass);
    case 'R':   // LEGACY_REGS
      if (VT == MVT::i8 || VT == MVT::i1)
        return std::make_pair(0U, &X86::GR8_NOREXRegClass);
      if (VT == MVT::i16)
        return std::make_pair(0U, &X86::GR16_NOREXRegClass);
      if (VT == MVT::i32 || !Subtarget->is64Bit())
        return std::make_pair(0U, &X86::GR32_NOREXRegClass);
      return std::make_pair(0U, &X86::GR64_NOREXRegClass);
    case 'f':  // FP Stack registers.
      // If SSE is enabled for this VT, use f80 to ensure the isel moves the
      // value to the correct fpstack register class.
      if (VT == MVT::f32 && !isScalarFPTypeInSSEReg(VT))
        return std::make_pair(0U, &X86::RFP32RegClass);
      if (VT == MVT::f64 && !isScalarFPTypeInSSEReg(VT))
        return std::make_pair(0U, &X86::RFP64RegClass);
      return std::make_pair(0U, &X86::RFP80RegClass);
    case 'y':   // MMX_REGS if MMX allowed.
      if (!Subtarget->hasMMX()) break;
      return std::make_pair(0U, &X86::VR64RegClass);
    case 'Y':   // SSE_REGS if SSE2 allowed
      if (!Subtarget->hasSSE2()) break;
      // FALL THROUGH.
    case 'x':   // SSE_REGS if SSE1 allowed or AVX_REGS if AVX allowed
      if (!Subtarget->hasSSE1()) break;

      switch (VT.getSimpleVT().SimpleTy) {
      default: break;
      // Scalar SSE types.
      case MVT::f32:
      case MVT::i32:
        return std::make_pair(0U, &X86::FR32RegClass);
      case MVT::f64:
      case MVT::i64:
        return std::make_pair(0U, &X86::FR64RegClass);
      // Vector types.
      case MVT::v16i8:
      case MVT::v8i16:
      case MVT::v4i32:
      case MVT::v2i64:
      case MVT::v4f32:
      case MVT::v2f64:
        return std::make_pair(0U, &X86::VR128RegClass);
      // AVX types.
      case MVT::v32i8:
      case MVT::v16i16:
      case MVT::v8i32:
      case MVT::v4i64:
      case MVT::v8f32:
      case MVT::v4f64:
        return std::make_pair(0U, &X86::VR256RegClass);
      }
      break;
    }
  }

  // Use the default implementation in TargetLowering to convert the register
  // constraint into a member of a register class.
  std::pair<unsigned, const TargetRegisterClass*> Res;
  Res = TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);

  // Not found as a standard register?
  if (Res.second == 0) {
    // Map st(0) -> st(7) -> ST0
    if (Constraint.size() == 7 && Constraint[0] == '{' &&
        tolower(Constraint[1]) == 's' &&
        tolower(Constraint[2]) == 't' &&
        Constraint[3] == '(' &&
        (Constraint[4] >= '0' && Constraint[4] <= '7') &&
        Constraint[5] == ')' &&
        Constraint[6] == '}') {

      Res.first = X86::ST0+Constraint[4]-'0';
      Res.second = &X86::RFP80RegClass;
      return Res;
    }

    // GCC allows "st(0)" to be called just plain "st".
    if (StringRef("{st}").equals_lower(Constraint)) {
      Res.first = X86::ST0;
      Res.second = &X86::RFP80RegClass;
      return Res;
    }

    // flags -> EFLAGS
    if (StringRef("{flags}").equals_lower(Constraint)) {
      Res.first = X86::EFLAGS;
      Res.second = &X86::CCRRegClass;
      return Res;
    }

    // 'A' means EAX + EDX.
    if (Constraint == "A") {
      Res.first = X86::EAX;
      Res.second = &X86::GR32_ADRegClass;
      return Res;
    }
    return Res;
  }

  // Otherwise, check to see if this is a register class of the wrong value
  // type.  For example, we want to map "{ax},i32" -> {eax}, we don't want it to
  // turn into {ax},{dx}.
  if (Res.second->hasType(VT))
    return Res;   // Correct type already, nothing to do.

  // All of the single-register GCC register classes map their values onto
  // 16-bit register pieces "ax","dx","cx","bx","si","di","bp","sp".  If we
  // really want an 8-bit or 32-bit register, map to the appropriate register
  // class and return the appropriate register.
  if (Res.second == &X86::GR16RegClass) {
    if (VT == MVT::i8) {
      unsigned DestReg = 0;
      switch (Res.first) {
      default: break;
      case X86::AX: DestReg = X86::AL; break;
      case X86::DX: DestReg = X86::DL; break;
      case X86::CX: DestReg = X86::CL; break;
      case X86::BX: DestReg = X86::BL; break;
      }
      if (DestReg) {
        Res.first = DestReg;
        Res.second = &X86::GR8RegClass;
      }
    } else if (VT == MVT::i32) {
      unsigned DestReg = 0;
      switch (Res.first) {
      default: break;
      case X86::AX: DestReg = X86::EAX; break;
      case X86::DX: DestReg = X86::EDX; break;
      case X86::CX: DestReg = X86::ECX; break;
      case X86::BX: DestReg = X86::EBX; break;
      case X86::SI: DestReg = X86::ESI; break;
      case X86::DI: DestReg = X86::EDI; break;
      case X86::BP: DestReg = X86::EBP; break;
      case X86::SP: DestReg = X86::ESP; break;
      }
      if (DestReg) {
        Res.first = DestReg;
        Res.second = &X86::GR32RegClass;
      }
    } else if (VT == MVT::i64) {
      unsigned DestReg = 0;
      switch (Res.first) {
      default: break;
      case X86::AX: DestReg = X86::RAX; break;
      case X86::DX: DestReg = X86::RDX; break;
      case X86::CX: DestReg = X86::RCX; break;
      case X86::BX: DestReg = X86::RBX; break;
      case X86::SI: DestReg = X86::RSI; break;
      case X86::DI: DestReg = X86::RDI; break;
      case X86::BP: DestReg = X86::RBP; break;
      case X86::SP: DestReg = X86::RSP; break;
      }
      if (DestReg) {
        Res.first = DestReg;
        Res.second = &X86::GR64RegClass;
      }
    }
  } else if (Res.second == &X86::FR32RegClass ||
             Res.second == &X86::FR64RegClass ||
             Res.second == &X86::VR128RegClass) {
    // Handle references to XMM physical registers that got mapped into the
    // wrong class.  This can happen with constraints like {xmm0} where the
    // target independent register mapper will just pick the first match it can
    // find, ignoring the required type.

    if (VT == MVT::f32 || VT == MVT::i32)
      Res.second = &X86::FR32RegClass;
    else if (VT == MVT::f64 || VT == MVT::i64)
      Res.second = &X86::FR64RegClass;
    else if (X86::VR128RegClass.hasType(VT))
      Res.second = &X86::VR128RegClass;
    else if (X86::VR256RegClass.hasType(VT))
      Res.second = &X86::VR256RegClass;
  }

  return Res;
}

//===----------------------------------------------------------------------===//
//
// X86 cost model.
//
//===----------------------------------------------------------------------===//

struct X86CostTblEntry {
  int ISD;
  MVT Type;
  unsigned Cost;
};

static int
FindInTable(const X86CostTblEntry *Tbl, unsigned len, int ISD, MVT Ty) {
  for (unsigned int i = 0; i < len; ++i)
    if (Tbl[i].ISD == ISD && Tbl[i].Type == Ty)
      return i;

  // Could not find an entry.
  return -1;
}

struct X86TypeConversionCostTblEntry {
  int ISD;
  MVT Dst;
  MVT Src;
  unsigned Cost;
};

static int
FindInConvertTable(const X86TypeConversionCostTblEntry *Tbl, unsigned len,
                   int ISD, MVT Dst, MVT Src) {
  for (unsigned int i = 0; i < len; ++i)
    if (Tbl[i].ISD == ISD && Tbl[i].Src == Src && Tbl[i].Dst == Dst)
      return i;

  // Could not find an entry.
  return -1;
}

unsigned
X86VectorTargetTransformInfo::getArithmeticInstrCost(unsigned Opcode,
                                                     Type *Ty) const {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = getTypeLegalizationCost(Ty);

  int ISD = InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  const X86Subtarget &ST = TLI->getTargetMachine().getSubtarget<X86Subtarget>();

  static const X86CostTblEntry AVX1CostTable[] = {
    // We don't have to scalarize unsupported ops. We can issue two half-sized
    // operations and we only need to extract the upper YMM half.
    // Two ops + 1 extract + 1 insert = 4.
    { ISD::MUL,     MVT::v8i32,    4 },
    { ISD::SUB,     MVT::v8i32,    4 },
    { ISD::ADD,     MVT::v8i32,    4 },
    { ISD::MUL,     MVT::v4i64,    4 },
    { ISD::SUB,     MVT::v4i64,    4 },
    { ISD::ADD,     MVT::v4i64,    4 },
    };

  // Look for AVX1 lowering tricks.
  if (ST.hasAVX()) {
    int Idx = FindInTable(AVX1CostTable, array_lengthof(AVX1CostTable), ISD,
                          LT.second);
    if (Idx != -1)
      return LT.first * AVX1CostTable[Idx].Cost;
  }
  // Fallback to the default implementation.
  return VectorTargetTransformImpl::getArithmeticInstrCost(Opcode, Ty);
}

unsigned
X86VectorTargetTransformInfo::getVectorInstrCost(unsigned Opcode, Type *Val,
                                                 unsigned Index) const {
  assert(Val->isVectorTy() && "This must be a vector type");

  if (Index != -1U) {
    // Legalize the type.
    std::pair<unsigned, MVT> LT = getTypeLegalizationCost(Val);

    // This type is legalized to a scalar type.
    if (!LT.second.isVector())
      return 0;

    // The type may be split. Normalize the index to the new type.
    unsigned Width = LT.second.getVectorNumElements();
    Index = Index % Width;

    // Floating point scalars are already located in index #0.
    if (Val->getScalarType()->isFloatingPointTy() && Index == 0)
      return 0;
  }

  return VectorTargetTransformImpl::getVectorInstrCost(Opcode, Val, Index);
}

unsigned X86VectorTargetTransformInfo::getCmpSelInstrCost(unsigned Opcode,
                                                          Type *ValTy,
                                                          Type *CondTy) const {
  // Legalize the type.
  std::pair<unsigned, MVT> LT = getTypeLegalizationCost(ValTy);

  MVT MTy = LT.second;

  int ISD = InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  const X86Subtarget &ST =
  TLI->getTargetMachine().getSubtarget<X86Subtarget>();

  static const X86CostTblEntry SSE42CostTbl[] = {
    { ISD::SETCC,   MVT::v2f64,   1 },
    { ISD::SETCC,   MVT::v4f32,   1 },
    { ISD::SETCC,   MVT::v2i64,   1 },
    { ISD::SETCC,   MVT::v4i32,   1 },
    { ISD::SETCC,   MVT::v8i16,   1 },
    { ISD::SETCC,   MVT::v16i8,   1 },
  };

  static const X86CostTblEntry AVX1CostTbl[] = {
    { ISD::SETCC,   MVT::v4f64,   1 },
    { ISD::SETCC,   MVT::v8f32,   1 },
    // AVX1 does not support 8-wide integer compare.
    { ISD::SETCC,   MVT::v4i64,   4 },
    { ISD::SETCC,   MVT::v8i32,   4 },
    { ISD::SETCC,   MVT::v16i16,  4 },
    { ISD::SETCC,   MVT::v32i8,   4 },
  };

  static const X86CostTblEntry AVX2CostTbl[] = {
    { ISD::SETCC,   MVT::v4i64,   1 },
    { ISD::SETCC,   MVT::v8i32,   1 },
    { ISD::SETCC,   MVT::v16i16,  1 },
    { ISD::SETCC,   MVT::v32i8,   1 },
  };

  if (ST.hasSSE42()) {
    int Idx = FindInTable(SSE42CostTbl, array_lengthof(SSE42CostTbl), ISD, MTy);
    if (Idx != -1)
      return LT.first * SSE42CostTbl[Idx].Cost;
  }

  if (ST.hasAVX()) {
    int Idx = FindInTable(AVX1CostTbl, array_lengthof(AVX1CostTbl), ISD, MTy);
    if (Idx != -1)
      return LT.first * AVX1CostTbl[Idx].Cost;
  }

  if (ST.hasAVX2()) {
    int Idx = FindInTable(AVX2CostTbl, array_lengthof(AVX2CostTbl), ISD, MTy);
    if (Idx != -1)
      return LT.first * AVX2CostTbl[Idx].Cost;
  }

  return VectorTargetTransformImpl::getCmpSelInstrCost(Opcode, ValTy, CondTy);
}

unsigned X86VectorTargetTransformInfo::getCastInstrCost(unsigned Opcode,
                                                        Type *Dst,
                                                        Type *Src) const {
  int ISD = InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  EVT SrcTy = TLI->getValueType(Src);
  EVT DstTy = TLI->getValueType(Dst);

  if (!SrcTy.isSimple() || !DstTy.isSimple())
    return VectorTargetTransformImpl::getCastInstrCost(Opcode, Dst, Src);

  const X86Subtarget &ST = TLI->getTargetMachine().getSubtarget<X86Subtarget>();

  static const X86TypeConversionCostTblEntry AVXConversionTbl[] = {
    { ISD::SIGN_EXTEND, MVT::v8i32, MVT::v8i16, 1 },
    { ISD::ZERO_EXTEND, MVT::v8i32, MVT::v8i16, 1 },
    { ISD::SIGN_EXTEND, MVT::v4i64, MVT::v4i32, 1 },
    { ISD::ZERO_EXTEND, MVT::v4i64, MVT::v4i32, 1 },
    { ISD::TRUNCATE,    MVT::v4i32, MVT::v4i64, 1 },
    { ISD::TRUNCATE,    MVT::v8i16, MVT::v8i32, 1 },
    { ISD::SINT_TO_FP,  MVT::v8f32, MVT::v8i8,  1 },
    { ISD::SINT_TO_FP,  MVT::v4f32, MVT::v4i8,  1 },
    { ISD::UINT_TO_FP,  MVT::v8f32, MVT::v8i8,  1 },
    { ISD::UINT_TO_FP,  MVT::v4f32, MVT::v4i8,  1 },
    { ISD::FP_TO_SINT,  MVT::v8i8,  MVT::v8f32, 1 },
    { ISD::FP_TO_SINT,  MVT::v4i8,  MVT::v4f32, 1 },
    { ISD::ZERO_EXTEND, MVT::v8i32, MVT::v8i1,  6 },
    { ISD::SIGN_EXTEND, MVT::v8i32, MVT::v8i1,  9 },
    { ISD::TRUNCATE,    MVT::v8i32, MVT::v8i64, 3 },
  };

  if (ST.hasAVX()) {
    int Idx = FindInConvertTable(AVXConversionTbl,
                                 array_lengthof(AVXConversionTbl),
                                 ISD, DstTy.getSimpleVT(), SrcTy.getSimpleVT());
    if (Idx != -1)
      return AVXConversionTbl[Idx].Cost;
  }

  return VectorTargetTransformImpl::getCastInstrCost(Opcode, Dst, Src);
}

