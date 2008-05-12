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

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86ISelLowering.h"
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

// Forward declarations.
static SDOperand getMOVLMask(unsigned NumElems, SelectionDAG &DAG);

X86TargetLowering::X86TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  Subtarget = &TM.getSubtarget<X86Subtarget>();
  X86ScalarSSEf64 = Subtarget->hasSSE2();
  X86ScalarSSEf32 = Subtarget->hasSSE1();
  X86StackPtr = Subtarget->is64Bit() ? X86::RSP : X86::ESP;
  
  bool Fast = false;

  RegInfo = TM.getRegisterInfo();

  // Set up the TargetLowering object.

  // X86 is weird, it always uses i8 for shift amounts and setcc results.
  setShiftAmountType(MVT::i8);
  setSetCCResultContents(ZeroOrOneSetCCResult);
  setSchedulingPreference(SchedulingForRegPressure);
  setShiftAmountFlavor(Mask);   // shl X, 32 == shl X, 0
  setStackPointerRegisterToSaveRestore(X86StackPtr);

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
  addRegisterClass(MVT::i8, X86::GR8RegisterClass);
  addRegisterClass(MVT::i16, X86::GR16RegisterClass);
  addRegisterClass(MVT::i32, X86::GR32RegisterClass);
  if (Subtarget->is64Bit())
    addRegisterClass(MVT::i64, X86::GR64RegisterClass);

  setLoadXAction(ISD::SEXTLOAD, MVT::i1, Promote);

  // We don't accept any truncstore of integer registers.  
  setTruncStoreAction(MVT::i64, MVT::i32, Expand);
  setTruncStoreAction(MVT::i64, MVT::i16, Expand);
  setTruncStoreAction(MVT::i64, MVT::i8 , Expand);
  setTruncStoreAction(MVT::i32, MVT::i16, Expand);
  setTruncStoreAction(MVT::i32, MVT::i8 , Expand);
  setTruncStoreAction(MVT::i16, MVT::i8, Expand);

  // Promote all UINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have this
  // operation.
  setOperationAction(ISD::UINT_TO_FP       , MVT::i1   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i8   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i16  , Promote);

  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::UINT_TO_FP     , MVT::i64  , Expand);
    setOperationAction(ISD::UINT_TO_FP     , MVT::i32  , Promote);
  } else {
    if (X86ScalarSSEf64)
      // If SSE i64 SINT_TO_FP is not available, expand i32 UINT_TO_FP.
      setOperationAction(ISD::UINT_TO_FP   , MVT::i32  , Expand);
    else
      setOperationAction(ISD::UINT_TO_FP   , MVT::i32  , Promote);
  }

  // Promote i1/i8 SINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have
  // this operation.
  setOperationAction(ISD::SINT_TO_FP       , MVT::i1   , Promote);
  setOperationAction(ISD::SINT_TO_FP       , MVT::i8   , Promote);
  // SSE has no i16 to fp conversion, only i32
  if (X86ScalarSSEf32) {
    setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Promote);
    // f32 and f64 cases are Legal, f80 case is not
    setOperationAction(ISD::SINT_TO_FP     , MVT::i32  , Custom);
  } else {
    setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Custom);
    setOperationAction(ISD::SINT_TO_FP     , MVT::i32  , Custom);
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
  } else {
    if (X86ScalarSSEf32 && !Subtarget->hasSSE3())
      // Expand FP_TO_UINT into a select.
      // FIXME: We would like to use a Custom expander here eventually to do
      // the optimal thing for SSE vs. the default expansion in the legalizer.
      setOperationAction(ISD::FP_TO_UINT   , MVT::i32  , Expand);
    else
      // With SSE3 we can use fisttpll to convert to a signed i64.
      setOperationAction(ISD::FP_TO_UINT   , MVT::i32  , Promote);
  }

  // TODO: when we have SSE, these could be more efficient, by using movd/movq.
  if (!X86ScalarSSEf64) {
    setOperationAction(ISD::BIT_CONVERT      , MVT::f32  , Expand);
    setOperationAction(ISD::BIT_CONVERT      , MVT::i32  , Expand);
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
  setOperationAction(ISD::MULHS           , MVT::i8    , Expand);
  setOperationAction(ISD::MULHU           , MVT::i8    , Expand);
  setOperationAction(ISD::SDIV            , MVT::i8    , Expand);
  setOperationAction(ISD::UDIV            , MVT::i8    , Expand);
  setOperationAction(ISD::SREM            , MVT::i8    , Expand);
  setOperationAction(ISD::UREM            , MVT::i8    , Expand);
  setOperationAction(ISD::MULHS           , MVT::i16   , Expand);
  setOperationAction(ISD::MULHU           , MVT::i16   , Expand);
  setOperationAction(ISD::SDIV            , MVT::i16   , Expand);
  setOperationAction(ISD::UDIV            , MVT::i16   , Expand);
  setOperationAction(ISD::SREM            , MVT::i16   , Expand);
  setOperationAction(ISD::UREM            , MVT::i16   , Expand);
  setOperationAction(ISD::MULHS           , MVT::i32   , Expand);
  setOperationAction(ISD::MULHU           , MVT::i32   , Expand);
  setOperationAction(ISD::SDIV            , MVT::i32   , Expand);
  setOperationAction(ISD::UDIV            , MVT::i32   , Expand);
  setOperationAction(ISD::SREM            , MVT::i32   , Expand);
  setOperationAction(ISD::UREM            , MVT::i32   , Expand);
  setOperationAction(ISD::MULHS           , MVT::i64   , Expand);
  setOperationAction(ISD::MULHU           , MVT::i64   , Expand);
  setOperationAction(ISD::SDIV            , MVT::i64   , Expand);
  setOperationAction(ISD::UDIV            , MVT::i64   , Expand);
  setOperationAction(ISD::SREM            , MVT::i64   , Expand);
  setOperationAction(ISD::UREM            , MVT::i64   , Expand);

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
  
  setOperationAction(ISD::CTPOP            , MVT::i8   , Expand);
  setOperationAction(ISD::CTTZ             , MVT::i8   , Custom);
  setOperationAction(ISD::CTLZ             , MVT::i8   , Custom);
  setOperationAction(ISD::CTPOP            , MVT::i16  , Expand);
  setOperationAction(ISD::CTTZ             , MVT::i16  , Custom);
  setOperationAction(ISD::CTLZ             , MVT::i16  , Custom);
  setOperationAction(ISD::CTPOP            , MVT::i32  , Expand);
  setOperationAction(ISD::CTTZ             , MVT::i32  , Custom);
  setOperationAction(ISD::CTLZ             , MVT::i32  , Custom);
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::CTPOP          , MVT::i64  , Expand);
    setOperationAction(ISD::CTTZ           , MVT::i64  , Custom);
    setOperationAction(ISD::CTLZ           , MVT::i64  , Custom);
  }

  setOperationAction(ISD::READCYCLECOUNTER , MVT::i64  , Custom);
  setOperationAction(ISD::BSWAP            , MVT::i16  , Expand);

  // These should be promoted to a larger select which is supported.
  setOperationAction(ISD::SELECT           , MVT::i1   , Promote);
  setOperationAction(ISD::SELECT           , MVT::i8   , Promote);
  // X86 wants to expand cmov itself.
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
  // X86 ret instruction may pop stack.
  setOperationAction(ISD::RET             , MVT::Other, Custom);
  if (!Subtarget->is64Bit())
    setOperationAction(ISD::EH_RETURN       , MVT::Other, Custom);

  // Darwin ABI issue.
  setOperationAction(ISD::ConstantPool    , MVT::i32  , Custom);
  setOperationAction(ISD::JumpTable       , MVT::i32  , Custom);
  setOperationAction(ISD::GlobalAddress   , MVT::i32  , Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32  , Custom);
  if (Subtarget->is64Bit())
    setOperationAction(ISD::GlobalTLSAddress, MVT::i64, Custom);
  setOperationAction(ISD::ExternalSymbol  , MVT::i32  , Custom);
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::ConstantPool  , MVT::i64  , Custom);
    setOperationAction(ISD::JumpTable     , MVT::i64  , Custom);
    setOperationAction(ISD::GlobalAddress , MVT::i64  , Custom);
    setOperationAction(ISD::ExternalSymbol, MVT::i64  , Custom);
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

  if (!Subtarget->hasSSE2())
    setOperationAction(ISD::MEMBARRIER    , MVT::Other, Expand);

  // Expand certain atomics
  setOperationAction(ISD::ATOMIC_LCS     , MVT::i8, Custom);
  setOperationAction(ISD::ATOMIC_LCS     , MVT::i16, Custom);
  setOperationAction(ISD::ATOMIC_LCS     , MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LCS     , MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LSS     , MVT::i32, Expand);

  // Use the default ISD::LOCATION, ISD::DECLARE expansion.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  // FIXME - use subtarget debug flags
  if (!Subtarget->isTargetDarwin() &&
      !Subtarget->isTargetELF() &&
      !Subtarget->isTargetCygMing())
    setOperationAction(ISD::LABEL, MVT::Other, Expand);

  setOperationAction(ISD::EXCEPTIONADDR, MVT::i64, Expand);
  setOperationAction(ISD::EHSELECTION,   MVT::i64, Expand);
  setOperationAction(ISD::EXCEPTIONADDR, MVT::i32, Expand);
  setOperationAction(ISD::EHSELECTION,   MVT::i32, Expand);
  if (Subtarget->is64Bit()) {
    // FIXME: Verify
    setExceptionPointerRegister(X86::RAX);
    setExceptionSelectorRegister(X86::RDX);
  } else {
    setExceptionPointerRegister(X86::EAX);
    setExceptionSelectorRegister(X86::EDX);
  }
  setOperationAction(ISD::FRAME_TO_ARGS_OFFSET, MVT::i32, Custom);
  
  setOperationAction(ISD::TRAMPOLINE, MVT::Other, Custom);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);

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
  if (Subtarget->is64Bit())
    setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);
  if (Subtarget->isTargetCygMing())
    setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Custom);
  else
    setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);

  if (X86ScalarSSEf64) {
    // f32 and f64 use SSE.
    // Set up the FP register classes.
    addRegisterClass(MVT::f32, X86::FR32RegisterClass);
    addRegisterClass(MVT::f64, X86::FR64RegisterClass);

    // Use ANDPD to simulate FABS.
    setOperationAction(ISD::FABS , MVT::f64, Custom);
    setOperationAction(ISD::FABS , MVT::f32, Custom);

    // Use XORP to simulate FNEG.
    setOperationAction(ISD::FNEG , MVT::f64, Custom);
    setOperationAction(ISD::FNEG , MVT::f32, Custom);

    // Use ANDPD and ORPD to simulate FCOPYSIGN.
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);

    // We don't support sin/cos/fmod
    setOperationAction(ISD::FSIN , MVT::f64, Expand);
    setOperationAction(ISD::FCOS , MVT::f64, Expand);
    setOperationAction(ISD::FSIN , MVT::f32, Expand);
    setOperationAction(ISD::FCOS , MVT::f32, Expand);

    // Expand FP immediates into loads from the stack, except for the special
    // cases we handle.
    addLegalFPImmediate(APFloat(+0.0)); // xorpd
    addLegalFPImmediate(APFloat(+0.0f)); // xorps

    // Floating truncations from f80 and extensions to f80 go through memory.
    // If optimizing, we lie about this though and handle it in
    // InstructionSelectPreprocess so that dagcombine2 can hack on these.
    if (Fast) {
      setConvertAction(MVT::f32, MVT::f80, Expand);
      setConvertAction(MVT::f64, MVT::f80, Expand);
      setConvertAction(MVT::f80, MVT::f32, Expand);
      setConvertAction(MVT::f80, MVT::f64, Expand);
    }
  } else if (X86ScalarSSEf32) {
    // Use SSE for f32, x87 for f64.
    // Set up the FP register classes.
    addRegisterClass(MVT::f32, X86::FR32RegisterClass);
    addRegisterClass(MVT::f64, X86::RFP64RegisterClass);

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

    // SSE <-> X87 conversions go through memory.  If optimizing, we lie about
    // this though and handle it in InstructionSelectPreprocess so that
    // dagcombine2 can hack on these.
    if (Fast) {
      setConvertAction(MVT::f32, MVT::f64, Expand);
      setConvertAction(MVT::f32, MVT::f80, Expand);
      setConvertAction(MVT::f80, MVT::f32, Expand);    
      setConvertAction(MVT::f64, MVT::f32, Expand);
      // And x87->x87 truncations also.
      setConvertAction(MVT::f80, MVT::f64, Expand);
    }

    if (!UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f64  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f64  , Expand);
    }
  } else {
    // f32 and f64 in x87.
    // Set up the FP register classes.
    addRegisterClass(MVT::f64, X86::RFP64RegisterClass);
    addRegisterClass(MVT::f32, X86::RFP32RegisterClass);

    setOperationAction(ISD::UNDEF,     MVT::f64, Expand);
    setOperationAction(ISD::UNDEF,     MVT::f32, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

    // Floating truncations go through memory.  If optimizing, we lie about
    // this though and handle it in InstructionSelectPreprocess so that
    // dagcombine2 can hack on these.
    if (Fast) {
      setConvertAction(MVT::f80, MVT::f32, Expand);    
      setConvertAction(MVT::f64, MVT::f32, Expand);
      setConvertAction(MVT::f80, MVT::f64, Expand);
    }

    if (!UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f64  , Expand);
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

  // Long double always uses X87.
  addRegisterClass(MVT::f80, X86::RFP80RegisterClass);
  setOperationAction(ISD::UNDEF,     MVT::f80, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f80, Expand);
  {
    APFloat TmpFlt(+0.0);
    TmpFlt.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven);
    addLegalFPImmediate(TmpFlt);  // FLD0
    TmpFlt.changeSign();
    addLegalFPImmediate(TmpFlt);  // FLD0/FCHS
    APFloat TmpFlt2(+1.0);
    TmpFlt2.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven);
    addLegalFPImmediate(TmpFlt2);  // FLD1
    TmpFlt2.changeSign();
    addLegalFPImmediate(TmpFlt2);  // FLD1/FCHS
  }
    
  if (!UnsafeFPMath) {
    setOperationAction(ISD::FSIN           , MVT::f80  , Expand);
    setOperationAction(ISD::FCOS           , MVT::f80  , Expand);
  }

  // Always use a library call for pow.
  setOperationAction(ISD::FPOW             , MVT::f32  , Expand);
  setOperationAction(ISD::FPOW             , MVT::f64  , Expand);
  setOperationAction(ISD::FPOW             , MVT::f80  , Expand);

  // First set operation action for all vector types to expand. Then we
  // will selectively turn on ones that can be effectively codegen'd.
  for (unsigned VT = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
       VT <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++VT) {
    setOperationAction(ISD::ADD , (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SUB , (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FADD, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FNEG, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FSUB, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::MUL , (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FMUL, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SDIV, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::UDIV, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FDIV, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SREM, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::UREM, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::LOAD, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::VECTOR_SHUFFLE,     (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FABS, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FSIN, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FCOS, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FREM, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FPOWI, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FSQRT, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FCOPYSIGN, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SDIVREM, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::UDIVREM, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FPOW, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::CTPOP, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::CTTZ, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::CTLZ, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SHL, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SRA, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SRL, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::ROTL, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::ROTR, (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::BSWAP, (MVT::ValueType)VT, Expand);
  }

  if (Subtarget->hasMMX()) {
    addRegisterClass(MVT::v8i8,  X86::VR64RegisterClass);
    addRegisterClass(MVT::v4i16, X86::VR64RegisterClass);
    addRegisterClass(MVT::v2i32, X86::VR64RegisterClass);
    addRegisterClass(MVT::v1i64, X86::VR64RegisterClass);

    // FIXME: add MMX packed arithmetics

    setOperationAction(ISD::ADD,                MVT::v8i8,  Legal);
    setOperationAction(ISD::ADD,                MVT::v4i16, Legal);
    setOperationAction(ISD::ADD,                MVT::v2i32, Legal);
    setOperationAction(ISD::ADD,                MVT::v1i64, Legal);

    setOperationAction(ISD::SUB,                MVT::v8i8,  Legal);
    setOperationAction(ISD::SUB,                MVT::v4i16, Legal);
    setOperationAction(ISD::SUB,                MVT::v2i32, Legal);
    setOperationAction(ISD::SUB,                MVT::v1i64, Legal);

    setOperationAction(ISD::MULHS,              MVT::v4i16, Legal);
    setOperationAction(ISD::MUL,                MVT::v4i16, Legal);

    setOperationAction(ISD::AND,                MVT::v8i8,  Promote);
    AddPromotedToType (ISD::AND,                MVT::v8i8,  MVT::v1i64);
    setOperationAction(ISD::AND,                MVT::v4i16, Promote);
    AddPromotedToType (ISD::AND,                MVT::v4i16, MVT::v1i64);
    setOperationAction(ISD::AND,                MVT::v2i32, Promote);
    AddPromotedToType (ISD::AND,                MVT::v2i32, MVT::v1i64);
    setOperationAction(ISD::AND,                MVT::v1i64, Legal);

    setOperationAction(ISD::OR,                 MVT::v8i8,  Promote);
    AddPromotedToType (ISD::OR,                 MVT::v8i8,  MVT::v1i64);
    setOperationAction(ISD::OR,                 MVT::v4i16, Promote);
    AddPromotedToType (ISD::OR,                 MVT::v4i16, MVT::v1i64);
    setOperationAction(ISD::OR,                 MVT::v2i32, Promote);
    AddPromotedToType (ISD::OR,                 MVT::v2i32, MVT::v1i64);
    setOperationAction(ISD::OR,                 MVT::v1i64, Legal);

    setOperationAction(ISD::XOR,                MVT::v8i8,  Promote);
    AddPromotedToType (ISD::XOR,                MVT::v8i8,  MVT::v1i64);
    setOperationAction(ISD::XOR,                MVT::v4i16, Promote);
    AddPromotedToType (ISD::XOR,                MVT::v4i16, MVT::v1i64);
    setOperationAction(ISD::XOR,                MVT::v2i32, Promote);
    AddPromotedToType (ISD::XOR,                MVT::v2i32, MVT::v1i64);
    setOperationAction(ISD::XOR,                MVT::v1i64, Legal);

    setOperationAction(ISD::LOAD,               MVT::v8i8,  Promote);
    AddPromotedToType (ISD::LOAD,               MVT::v8i8,  MVT::v1i64);
    setOperationAction(ISD::LOAD,               MVT::v4i16, Promote);
    AddPromotedToType (ISD::LOAD,               MVT::v4i16, MVT::v1i64);
    setOperationAction(ISD::LOAD,               MVT::v2i32, Promote);
    AddPromotedToType (ISD::LOAD,               MVT::v2i32, MVT::v1i64);
    setOperationAction(ISD::LOAD,               MVT::v1i64, Legal);

    setOperationAction(ISD::BUILD_VECTOR,       MVT::v8i8,  Custom);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v4i16, Custom);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v2i32, Custom);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v1i64, Custom);

    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v8i8,  Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v4i16, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v2i32, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v1i64, Custom);

    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v8i8,  Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v4i16, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v1i64, Custom);
  }

  if (Subtarget->hasSSE1()) {
    addRegisterClass(MVT::v4f32, X86::VR128RegisterClass);

    setOperationAction(ISD::FADD,               MVT::v4f32, Legal);
    setOperationAction(ISD::FSUB,               MVT::v4f32, Legal);
    setOperationAction(ISD::FMUL,               MVT::v4f32, Legal);
    setOperationAction(ISD::FDIV,               MVT::v4f32, Legal);
    setOperationAction(ISD::FSQRT,              MVT::v4f32, Legal);
    setOperationAction(ISD::FNEG,               MVT::v4f32, Custom);
    setOperationAction(ISD::LOAD,               MVT::v4f32, Legal);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v4f32, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v4f32, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4f32, Custom);
    setOperationAction(ISD::SELECT,             MVT::v4f32, Custom);
  }

  if (Subtarget->hasSSE2()) {
    addRegisterClass(MVT::v2f64, X86::VR128RegisterClass);
    addRegisterClass(MVT::v16i8, X86::VR128RegisterClass);
    addRegisterClass(MVT::v8i16, X86::VR128RegisterClass);
    addRegisterClass(MVT::v4i32, X86::VR128RegisterClass);
    addRegisterClass(MVT::v2i64, X86::VR128RegisterClass);

    setOperationAction(ISD::ADD,                MVT::v16i8, Legal);
    setOperationAction(ISD::ADD,                MVT::v8i16, Legal);
    setOperationAction(ISD::ADD,                MVT::v4i32, Legal);
    setOperationAction(ISD::ADD,                MVT::v2i64, Legal);
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

    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v16i8, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4i32, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4f32, Custom);

    // Custom lower build_vector, vector_shuffle, and extract_vector_elt.
    for (unsigned VT = (unsigned)MVT::v16i8; VT != (unsigned)MVT::v2i64; VT++) {
      // Do not attempt to custom lower non-power-of-2 vectors
      if (!isPowerOf2_32(MVT::getVectorNumElements(VT)))
        continue;
      setOperationAction(ISD::BUILD_VECTOR,        (MVT::ValueType)VT, Custom);
      setOperationAction(ISD::VECTOR_SHUFFLE,      (MVT::ValueType)VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT,  (MVT::ValueType)VT, Custom);
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
    for (unsigned VT = (unsigned)MVT::v16i8; VT != (unsigned)MVT::v2i64; VT++) {
      setOperationAction(ISD::AND,    (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::AND,    (MVT::ValueType)VT, MVT::v2i64);
      setOperationAction(ISD::OR,     (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::OR,     (MVT::ValueType)VT, MVT::v2i64);
      setOperationAction(ISD::XOR,    (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::XOR,    (MVT::ValueType)VT, MVT::v2i64);
      setOperationAction(ISD::LOAD,   (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::LOAD,   (MVT::ValueType)VT, MVT::v2i64);
      setOperationAction(ISD::SELECT, (MVT::ValueType)VT, Promote);
      AddPromotedToType (ISD::SELECT, (MVT::ValueType)VT, MVT::v2i64);
    }

    setTruncStoreAction(MVT::f64, MVT::f32, Expand);

    // Custom lower v2i64 and v2f64 selects.
    setOperationAction(ISD::LOAD,               MVT::v2f64, Legal);
    setOperationAction(ISD::LOAD,               MVT::v2i64, Legal);
    setOperationAction(ISD::SELECT,             MVT::v2f64, Custom);
    setOperationAction(ISD::SELECT,             MVT::v2i64, Custom);
  }
  
  if (Subtarget->hasSSE41()) {
    // FIXME: Do we need to handle scalar-to-vector here?
    setOperationAction(ISD::MUL,                MVT::v4i32, Legal);

    // i8 and i16 vectors are custom , because the source register and source
    // source memory operand types are not the same width.  f32 vectors are
    // custom since the immediate controlling the insert encodes additional
    // information.
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v16i8, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4i32, Legal);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4f32, Custom);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v16i8, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v8i16, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i32, Legal);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4f32, Custom);

    if (Subtarget->is64Bit()) {
      setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v2i64, Legal);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i64, Legal);
    }
  }

  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::VECTOR_SHUFFLE);
  setTargetDAGCombine(ISD::BUILD_VECTOR);
  setTargetDAGCombine(ISD::SELECT);
  setTargetDAGCombine(ISD::STORE);

  computeRegisterProperties();

  // FIXME: These should be based on subtarget info. Plus, the values should
  // be smaller when we are in optimizing for size mode.
  maxStoresPerMemset = 16; // For %llvm.memset -> sequence of stores
  maxStoresPerMemcpy = 16; // For %llvm.memcpy -> sequence of stores
  maxStoresPerMemmove = 16; // For %llvm.memmove -> sequence of stores
  allowUnalignedMemoryAccesses = true; // x86 supports it!
  setPrefLoopAlignment(16);
}


MVT::ValueType
X86TargetLowering::getSetCCResultType(const SDOperand &) const {
  return MVT::i8;
}


/// getMaxByValAlign - Helper for getByValTypeAlignment to determine
/// the desired ByVal argument alignment.
static void getMaxByValAlign(const Type *Ty, unsigned &MaxAlign) {
  if (MaxAlign == 16)
    return;
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty)) {
    if (VTy->getBitWidth() == 128)
      MaxAlign = 16;
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    unsigned EltAlign = 0;
    getMaxByValAlign(ATy->getElementType(), EltAlign);
    if (EltAlign > MaxAlign)
      MaxAlign = EltAlign;
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      unsigned EltAlign = 0;
      getMaxByValAlign(STy->getElementType(i), EltAlign);
      if (EltAlign > MaxAlign)
        MaxAlign = EltAlign;
      if (MaxAlign == 16)
        break;
    }
  }
  return;
}

/// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
/// function arguments in the caller parameter area. For X86, aggregates
/// that contain SSE vectors are placed at 16-byte boundaries while the rest
/// are at 4-byte boundaries.
unsigned X86TargetLowering::getByValTypeAlignment(const Type *Ty) const {
  if (Subtarget->is64Bit())
    return getTargetData()->getABITypeAlignment(Ty);
  unsigned Align = 4;
  if (Subtarget->hasSSE1())
    getMaxByValAlign(Ty, Align);
  return Align;
}

/// getPICJumpTableRelocaBase - Returns relocation base for the given PIC
/// jumptable.
SDOperand X86TargetLowering::getPICJumpTableRelocBase(SDOperand Table,
                                                      SelectionDAG &DAG) const {
  if (usesGlobalOffsetTable())
    return DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, getPointerTy());
  if (!Subtarget->isPICStyleRIPRel())
    return DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy());
  return Table;
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "X86GenCallingConv.inc"

/// LowerRET - Lower an ISD::RET node.
SDOperand X86TargetLowering::LowerRET(SDOperand Op, SelectionDAG &DAG) {
  assert((Op.getNumOperands() & 1) == 1 && "ISD::RET should have odd # args");
  
  SmallVector<CCValAssign, 16> RVLocs;
  unsigned CC = DAG.getMachineFunction().getFunction()->getCallingConv();
  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();
  CCState CCInfo(CC, isVarArg, getTargetMachine(), RVLocs);
  CCInfo.AnalyzeReturn(Op.Val, RetCC_X86);
    
  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }
  SDOperand Chain = Op.getOperand(0);
  
  // Handle tail call return.
  Chain = GetPossiblePreceedingTailCall(Chain, X86ISD::TAILCALL);
  if (Chain.getOpcode() == X86ISD::TAILCALL) {
    SDOperand TailCall = Chain;
    SDOperand TargetAddress = TailCall.getOperand(1);
    SDOperand StackAdjustment = TailCall.getOperand(2);
    assert(((TargetAddress.getOpcode() == ISD::Register &&
               (cast<RegisterSDNode>(TargetAddress)->getReg() == X86::ECX ||
                cast<RegisterSDNode>(TargetAddress)->getReg() == X86::R9)) ||
              TargetAddress.getOpcode() == ISD::TargetExternalSymbol ||
              TargetAddress.getOpcode() == ISD::TargetGlobalAddress) && 
             "Expecting an global address, external symbol, or register");
    assert(StackAdjustment.getOpcode() == ISD::Constant &&
           "Expecting a const value");

    SmallVector<SDOperand,8> Operands;
    Operands.push_back(Chain.getOperand(0));
    Operands.push_back(TargetAddress);
    Operands.push_back(StackAdjustment);
    // Copy registers used by the call. Last operand is a flag so it is not
    // copied.
    for (unsigned i=3; i < TailCall.getNumOperands()-1; i++) {
      Operands.push_back(Chain.getOperand(i));
    }
    return DAG.getNode(X86ISD::TC_RETURN, MVT::Other, &Operands[0], 
                       Operands.size());
  }
  
  // Regular return.
  SDOperand Flag;

  SmallVector<SDOperand, 6> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)
  // Operand #1 = Bytes To Pop
  RetOps.push_back(DAG.getConstant(getBytesToPopOnReturn(), MVT::i16));
  
  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    SDOperand ValToCopy = Op.getOperand(i*2+1);
    
    // Returns in ST0/ST1 are handled specially: these are pushed as operands to
    // the RET instruction and handled by the FP Stackifier.
    if (RVLocs[i].getLocReg() == X86::ST0 ||
        RVLocs[i].getLocReg() == X86::ST1) {
      // If this is a copy from an xmm register to ST(0), use an FPExtend to
      // change the value to the FP stack register class.
      if (isScalarFPTypeInSSEReg(RVLocs[i].getValVT()))
        ValToCopy = DAG.getNode(ISD::FP_EXTEND, MVT::f80, ValToCopy);
      RetOps.push_back(ValToCopy);
      // Don't emit a copytoreg.
      continue;
    }
    
    Chain = DAG.getCopyToReg(Chain, VA.getLocReg(), ValToCopy, Flag);
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
    if (!Reg) {
      Reg = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i64));
      FuncInfo->setSRetReturnReg(Reg);
    }
    SDOperand Val = DAG.getCopyFromReg(Chain, Reg, getPointerTy());

    Chain = DAG.getCopyToReg(Chain, X86::RAX, Val, Flag);
    Flag = Chain.getValue(1);
  }
  
  RetOps[0] = Chain;  // Update chain.

  // Add the flag if we have it.
  if (Flag.Val)
    RetOps.push_back(Flag);
  
  return DAG.getNode(X86ISD::RET_FLAG, MVT::Other, &RetOps[0], RetOps.size());
}


/// LowerCallResult - Lower the result values of an ISD::CALL into the
/// appropriate copies out of appropriate physical registers.  This assumes that
/// Chain/InFlag are the input chain/flag to use, and that TheCall is the call
/// being lowered.  The returns a SDNode with the same number of values as the
/// ISD::CALL.
SDNode *X86TargetLowering::
LowerCallResult(SDOperand Chain, SDOperand InFlag, SDNode *TheCall, 
                unsigned CallingConv, SelectionDAG &DAG) {
  
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  bool isVarArg = cast<ConstantSDNode>(TheCall->getOperand(2))->getValue() != 0;
  CCState CCInfo(CallingConv, isVarArg, getTargetMachine(), RVLocs);
  CCInfo.AnalyzeCallResult(TheCall, RetCC_X86);

  SmallVector<SDOperand, 8> ResultVals;
  
  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    MVT::ValueType CopyVT = RVLocs[i].getValVT();
    
    // If this is a call to a function that returns an fp value on the floating
    // point stack, but where we prefer to use the value in xmm registers, copy
    // it out as F80 and use a truncate to move it from fp stack reg to xmm reg.
    if (RVLocs[i].getLocReg() == X86::ST0 &&
        isScalarFPTypeInSSEReg(RVLocs[i].getValVT())) {
      CopyVT = MVT::f80;
    }
    
    Chain = DAG.getCopyFromReg(Chain, RVLocs[i].getLocReg(),
                               CopyVT, InFlag).getValue(1);
    SDOperand Val = Chain.getValue(0);
    InFlag = Chain.getValue(2);

    if (CopyVT != RVLocs[i].getValVT()) {
      // Round the F80 the right size, which also moves to the appropriate xmm
      // register.
      Val = DAG.getNode(ISD::FP_ROUND, RVLocs[i].getValVT(), Val,
                        // This truncation won't change the value.
                        DAG.getIntPtrConstant(1));
    }
    
    ResultVals.push_back(Val);
  }
  
  // Merge everything together with a MERGE_VALUES node.
  ResultVals.push_back(Chain);
  return DAG.getNode(ISD::MERGE_VALUES, TheCall->getVTList(),
                     &ResultVals[0], ResultVals.size()).Val;
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

/// AddLiveIn - This helper function adds the specified physical register to the
/// MachineFunction as a live in value.  It also creates a corresponding virtual
/// register for it.
static unsigned AddLiveIn(MachineFunction &MF, unsigned PReg,
                          const TargetRegisterClass *RC) {
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getRegInfo().createVirtualRegister(RC);
  MF.getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}

/// CallIsStructReturn - Determines whether a CALL node uses struct return
/// semantics.
static bool CallIsStructReturn(SDOperand Op) {
  unsigned NumOps = (Op.getNumOperands() - 5) / 2;
  if (!NumOps)
    return false;

  return cast<ARG_FLAGSSDNode>(Op.getOperand(6))->getArgFlags().isSRet();
}

/// ArgsAreStructReturn - Determines whether a FORMAL_ARGUMENTS node uses struct
/// return semantics.
static bool ArgsAreStructReturn(SDOperand Op) {
  unsigned NumArgs = Op.Val->getNumValues() - 1;
  if (!NumArgs)
    return false;

  return cast<ARG_FLAGSSDNode>(Op.getOperand(3))->getArgFlags().isSRet();
}

/// IsCalleePop - Determines whether a CALL or FORMAL_ARGUMENTS node requires
/// the callee to pop its own arguments. Callee pop is necessary to support tail
/// calls.
bool X86TargetLowering::IsCalleePop(SDOperand Op) {
  bool IsVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  if (IsVarArg)
    return false;

  switch (cast<ConstantSDNode>(Op.getOperand(1))->getValue()) {
  default:
    return false;
  case CallingConv::X86_StdCall:
    return !Subtarget->is64Bit();
  case CallingConv::X86_FastCall:
    return !Subtarget->is64Bit();
  case CallingConv::Fast:
    return PerformTailCallOpt;
  }
}

/// CCAssignFnForNode - Selects the correct CCAssignFn for a CALL or
/// FORMAL_ARGUMENTS node.
CCAssignFn *X86TargetLowering::CCAssignFnForNode(SDOperand Op) const {
  unsigned CC = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  
  if (Subtarget->is64Bit()) {
    if (Subtarget->isTargetWin64())
      return CC_X86_Win64_C;
    else {
      if (CC == CallingConv::Fast && PerformTailCallOpt)
        return CC_X86_64_TailCall;
      else
        return CC_X86_64_C;
    }
  }

  if (CC == CallingConv::X86_FastCall)
    return CC_X86_32_FastCall;
  else if (CC == CallingConv::Fast && PerformTailCallOpt)
    return CC_X86_32_TailCall;
  else
    return CC_X86_32_C;
}

/// NameDecorationForFORMAL_ARGUMENTS - Selects the appropriate decoration to
/// apply to a MachineFunction containing a given FORMAL_ARGUMENTS node.
NameDecorationStyle
X86TargetLowering::NameDecorationForFORMAL_ARGUMENTS(SDOperand Op) {
  unsigned CC = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  if (CC == CallingConv::X86_FastCall)
    return FastCall;
  else if (CC == CallingConv::X86_StdCall)
    return StdCall;
  return None;
}


/// CallRequiresGOTInRegister - Check whether the call requires the GOT pointer
/// in a register before calling.
bool X86TargetLowering::CallRequiresGOTPtrInReg(bool Is64Bit, bool IsTailCall) {
  return !IsTailCall && !Is64Bit &&
    getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
    Subtarget->isPICStyleGOT();
}

/// CallRequiresFnAddressInReg - Check whether the call requires the function
/// address to be loaded in a register.
bool 
X86TargetLowering::CallRequiresFnAddressInReg(bool Is64Bit, bool IsTailCall) {
  return !Is64Bit && IsTailCall &&  
    getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
    Subtarget->isPICStyleGOT();
}

/// CreateCopyOfByValArgument - Make a copy of an aggregate at address specified
/// by "Src" to address "Dst" with size and alignment information specified by
/// the specific parameter attribute. The copy will be passed as a byval
/// function parameter.
static SDOperand 
CreateCopyOfByValArgument(SDOperand Src, SDOperand Dst, SDOperand Chain,
                          ISD::ArgFlagsTy Flags, SelectionDAG &DAG) {
  SDOperand SizeNode     = DAG.getConstant(Flags.getByValSize(), MVT::i32);
  return DAG.getMemcpy(Chain, Dst, Src, SizeNode, Flags.getByValAlign(),
                       /*AlwaysInline=*/true, NULL, 0, NULL, 0);
}

SDOperand X86TargetLowering::LowerMemArgument(SDOperand Op, SelectionDAG &DAG,
                                              const CCValAssign &VA,
                                              MachineFrameInfo *MFI,
                                              unsigned CC,
                                              SDOperand Root, unsigned i) {
  // Create the nodes corresponding to a load from this parameter slot.
  ISD::ArgFlagsTy Flags =
    cast<ARG_FLAGSSDNode>(Op.getOperand(3 + i))->getArgFlags();
  bool AlwaysUseMutable = (CC==CallingConv::Fast) && PerformTailCallOpt;
  bool isImmutable = !AlwaysUseMutable && !Flags.isByVal();

  // FIXME: For now, all byval parameter objects are marked mutable. This can be
  // changed with more analysis.  
  // In case of tail call optimization mark all arguments mutable. Since they
  // could be overwritten by lowering of arguments in case of a tail call.
  int FI = MFI->CreateFixedObject(MVT::getSizeInBits(VA.getValVT())/8,
                                  VA.getLocMemOffset(), isImmutable);
  SDOperand FIN = DAG.getFrameIndex(FI, getPointerTy());
  if (Flags.isByVal())
    return FIN;
  return DAG.getLoad(VA.getValVT(), Root, FIN,
                     PseudoSourceValue::getFixedStack(), FI);
}

SDOperand
X86TargetLowering::LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
  
  const Function* Fn = MF.getFunction();
  if (Fn->hasExternalLinkage() &&
      Subtarget->isTargetCygMing() &&
      Fn->getName() == "main")
    FuncInfo->setForceFramePointer(true);

  // Decorate the function name.
  FuncInfo->setDecorationStyle(NameDecorationForFORMAL_ARGUMENTS(Op));
  
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SDOperand Root = Op.getOperand(0);
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  unsigned CC = MF.getFunction()->getCallingConv();
  bool Is64Bit = Subtarget->is64Bit();
  bool IsWin64 = Subtarget->isTargetWin64();

  assert(!(isVarArg && CC == CallingConv::Fast) &&
         "Var args not supported with calling convention fastcc");

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);
  CCInfo.AnalyzeFormalArguments(Op.Val, CCAssignFnForNode(Op));
  
  SmallVector<SDOperand, 8> ArgValues;
  unsigned LastVal = ~0U;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    // TODO: If an arg is passed in two places (e.g. reg and stack), skip later
    // places.
    assert(VA.getValNo() != LastVal &&
           "Don't support value assigned to multiple locs yet");
    LastVal = VA.getValNo();
    
    if (VA.isRegLoc()) {
      MVT::ValueType RegVT = VA.getLocVT();
      TargetRegisterClass *RC;
      if (RegVT == MVT::i32)
        RC = X86::GR32RegisterClass;
      else if (Is64Bit && RegVT == MVT::i64)
        RC = X86::GR64RegisterClass;
      else if (RegVT == MVT::f32)
        RC = X86::FR32RegisterClass;
      else if (RegVT == MVT::f64)
        RC = X86::FR64RegisterClass;
      else if (MVT::isVector(RegVT) && MVT::getSizeInBits(RegVT) == 128)
        RC = X86::VR128RegisterClass;
      else if (MVT::isVector(RegVT)) {
        assert(MVT::getSizeInBits(RegVT) == 64);
        if (!Is64Bit)
          RC = X86::VR64RegisterClass;     // MMX values are passed in MMXs.
        else {
          // Darwin calling convention passes MMX values in either GPRs or
          // XMMs in x86-64. Other targets pass them in memory.
          if (RegVT != MVT::v1i64 && Subtarget->hasSSE2()) {
            RC = X86::VR128RegisterClass;  // MMX values are passed in XMMs.
            RegVT = MVT::v2i64;
          } else {
            RC = X86::GR64RegisterClass;   // v1i64 values are passed in GPRs.
            RegVT = MVT::i64;
          }
        }
      } else {
        assert(0 && "Unknown argument type!");
      }

      unsigned Reg = AddLiveIn(DAG.getMachineFunction(), VA.getLocReg(), RC);
      SDOperand ArgValue = DAG.getCopyFromReg(Root, Reg, RegVT);
      
      // If this is an 8 or 16-bit value, it is really passed promoted to 32
      // bits.  Insert an assert[sz]ext to capture this, then truncate to the
      // right size.
      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      
      if (VA.getLocInfo() != CCValAssign::Full)
        ArgValue = DAG.getNode(ISD::TRUNCATE, VA.getValVT(), ArgValue);
      
      // Handle MMX values passed in GPRs.
      if (Is64Bit && RegVT != VA.getLocVT()) {
        if (MVT::getSizeInBits(RegVT) == 64 && RC == X86::GR64RegisterClass)
          ArgValue = DAG.getNode(ISD::BIT_CONVERT, VA.getLocVT(), ArgValue);
        else if (RC == X86::VR128RegisterClass) {
          ArgValue = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i64, ArgValue,
                                 DAG.getConstant(0, MVT::i64));
          ArgValue = DAG.getNode(ISD::BIT_CONVERT, VA.getLocVT(), ArgValue);
        }
      }
      
      ArgValues.push_back(ArgValue);
    } else {
      assert(VA.isMemLoc());
      ArgValues.push_back(LowerMemArgument(Op, DAG, VA, MFI, CC, Root, i));
    }
  }

  // The x86-64 ABI for returning structs by value requires that we copy
  // the sret argument into %rax for the return. Save the argument into
  // a virtual register so that we can access it from the return points.
  if (Is64Bit && DAG.getMachineFunction().getFunction()->hasStructRetAttr()) {
    MachineFunction &MF = DAG.getMachineFunction();
    X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
    unsigned Reg = FuncInfo->getSRetReturnReg();
    if (!Reg) {
      Reg = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i64));
      FuncInfo->setSRetReturnReg(Reg);
    }
    SDOperand Copy = DAG.getCopyToReg(DAG.getEntryNode(), Reg, ArgValues[0]);
    Root = DAG.getNode(ISD::TokenFactor, MVT::Other, Copy, Root);
  }

  unsigned StackSize = CCInfo.getNextStackOffset();
  // align stack specially for tail calls
  if (CC == CallingConv::Fast)
    StackSize = GetAlignedArgumentStackSize(StackSize, DAG);

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (isVarArg) {
    if (Is64Bit || CC != CallingConv::X86_FastCall) {
      VarArgsFrameIndex = MFI->CreateFixedObject(1, StackSize);
    }
    if (Is64Bit) {
      unsigned TotalNumIntRegs = 0, TotalNumXMMRegs = 0;

      // FIXME: We should really autogenerate these arrays
      static const unsigned GPR64ArgRegsWin64[] = {
        X86::RCX, X86::RDX, X86::R8,  X86::R9
      };
      static const unsigned XMMArgRegsWin64[] = {
        X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3
      };
      static const unsigned GPR64ArgRegs64Bit[] = {
        X86::RDI, X86::RSI, X86::RDX, X86::RCX, X86::R8, X86::R9
      };
      static const unsigned XMMArgRegs64Bit[] = {
        X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
        X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
      };
      const unsigned *GPR64ArgRegs, *XMMArgRegs;

      if (IsWin64) {
        TotalNumIntRegs = 4; TotalNumXMMRegs = 4;
        GPR64ArgRegs = GPR64ArgRegsWin64;
        XMMArgRegs = XMMArgRegsWin64;
      } else {
        TotalNumIntRegs = 6; TotalNumXMMRegs = 8;
        GPR64ArgRegs = GPR64ArgRegs64Bit;
        XMMArgRegs = XMMArgRegs64Bit;
      }
      unsigned NumIntRegs = CCInfo.getFirstUnallocated(GPR64ArgRegs,
                                                       TotalNumIntRegs);
      unsigned NumXMMRegs = CCInfo.getFirstUnallocated(XMMArgRegs,
                                                       TotalNumXMMRegs);

      // For X86-64, if there are vararg parameters that are passed via
      // registers, then we must store them to their spots on the stack so they
      // may be loaded by deferencing the result of va_next.
      VarArgsGPOffset = NumIntRegs * 8;
      VarArgsFPOffset = TotalNumIntRegs * 8 + NumXMMRegs * 16;
      RegSaveFrameIndex = MFI->CreateStackObject(TotalNumIntRegs * 8 +
                                                 TotalNumXMMRegs * 16, 16);

      // Store the integer parameter registers.
      SmallVector<SDOperand, 8> MemOps;
      SDOperand RSFIN = DAG.getFrameIndex(RegSaveFrameIndex, getPointerTy());
      SDOperand FIN = DAG.getNode(ISD::ADD, getPointerTy(), RSFIN,
                                  DAG.getIntPtrConstant(VarArgsGPOffset));
      for (; NumIntRegs != TotalNumIntRegs; ++NumIntRegs) {
        unsigned VReg = AddLiveIn(MF, GPR64ArgRegs[NumIntRegs],
                                  X86::GR64RegisterClass);
        SDOperand Val = DAG.getCopyFromReg(Root, VReg, MVT::i64);
        SDOperand Store =
          DAG.getStore(Val.getValue(1), Val, FIN,
                       PseudoSourceValue::getFixedStack(),
                       RegSaveFrameIndex);
        MemOps.push_back(Store);
        FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                          DAG.getIntPtrConstant(8));
      }

      // Now store the XMM (fp + vector) parameter registers.
      FIN = DAG.getNode(ISD::ADD, getPointerTy(), RSFIN,
                        DAG.getIntPtrConstant(VarArgsFPOffset));
      for (; NumXMMRegs != TotalNumXMMRegs; ++NumXMMRegs) {
        unsigned VReg = AddLiveIn(MF, XMMArgRegs[NumXMMRegs],
                                  X86::VR128RegisterClass);
        SDOperand Val = DAG.getCopyFromReg(Root, VReg, MVT::v4f32);
        SDOperand Store =
          DAG.getStore(Val.getValue(1), Val, FIN,
                       PseudoSourceValue::getFixedStack(),
                       RegSaveFrameIndex);
        MemOps.push_back(Store);
        FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                          DAG.getIntPtrConstant(16));
      }
      if (!MemOps.empty())
          Root = DAG.getNode(ISD::TokenFactor, MVT::Other,
                             &MemOps[0], MemOps.size());
    }
  }
  
  // Make sure the instruction takes 8n+4 bytes to make sure the start of the
  // arguments and the arguments after the retaddr has been pushed are
  // aligned.
  if (!Is64Bit && CC == CallingConv::X86_FastCall &&
      !Subtarget->isTargetCygMing() && !Subtarget->isTargetWindows() &&
      (StackSize & 7) == 0)
    StackSize += 4;

  ArgValues.push_back(Root);

  // Some CCs need callee pop.
  if (IsCalleePop(Op)) {
    BytesToPopOnReturn  = StackSize; // Callee pops everything.
    BytesCallerReserves = 0;
  } else {
    BytesToPopOnReturn  = 0; // Callee pops nothing.
    // If this is an sret function, the return should pop the hidden pointer.
    if (!Is64Bit && ArgsAreStructReturn(Op))
      BytesToPopOnReturn = 4;  
    BytesCallerReserves = StackSize;
  }

  if (!Is64Bit) {
    RegSaveFrameIndex = 0xAAAAAAA;   // RegSaveFrameIndex is X86-64 only.
    if (CC == CallingConv::X86_FastCall)
      VarArgsFrameIndex = 0xAAAAAAA;   // fastcc functions can't have varargs.
  }

  FuncInfo->setBytesToPopOnReturn(BytesToPopOnReturn);

  // Return the new list of results.
  return DAG.getNode(ISD::MERGE_VALUES, Op.Val->getVTList(),
                     &ArgValues[0], ArgValues.size()).getValue(Op.ResNo);
}

SDOperand
X86TargetLowering::LowerMemOpCallTo(SDOperand Op, SelectionDAG &DAG,
                                    const SDOperand &StackPtr,
                                    const CCValAssign &VA,
                                    SDOperand Chain,
                                    SDOperand Arg) {
  unsigned LocMemOffset = VA.getLocMemOffset();
  SDOperand PtrOff = DAG.getIntPtrConstant(LocMemOffset);
  PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
  ISD::ArgFlagsTy Flags =
    cast<ARG_FLAGSSDNode>(Op.getOperand(6+2*VA.getValNo()))->getArgFlags();
  if (Flags.isByVal()) {
    return CreateCopyOfByValArgument(Arg, PtrOff, Chain, Flags, DAG);
  }
  return DAG.getStore(Chain, Arg, PtrOff,
                      PseudoSourceValue::getStack(), LocMemOffset);
}

/// EmitTailCallLoadRetAddr - Emit a load of return adress if tail call
/// optimization is performed and it is required.
SDOperand 
X86TargetLowering::EmitTailCallLoadRetAddr(SelectionDAG &DAG, 
                                           SDOperand &OutRetAddr,
                                           SDOperand Chain, 
                                           bool IsTailCall, 
                                           bool Is64Bit, 
                                           int FPDiff) {
  if (!IsTailCall || FPDiff==0) return Chain;

  // Adjust the Return address stack slot.
  MVT::ValueType VT = getPointerTy();
  OutRetAddr = getReturnAddressFrameIndex(DAG);
  // Load the "old" Return address.
  OutRetAddr = DAG.getLoad(VT, Chain,OutRetAddr, NULL, 0);
  return SDOperand(OutRetAddr.Val, 1);
}

/// EmitTailCallStoreRetAddr - Emit a store of the return adress if tail call
/// optimization is performed and it is required (FPDiff!=0).
static SDOperand 
EmitTailCallStoreRetAddr(SelectionDAG & DAG, MachineFunction &MF, 
                         SDOperand Chain, SDOperand RetAddrFrIdx,
                         bool Is64Bit, int FPDiff) {
  // Store the return address to the appropriate stack slot.
  if (!FPDiff) return Chain;
  // Calculate the new stack slot for the return address.
  int SlotSize = Is64Bit ? 8 : 4;
  int NewReturnAddrFI = 
    MF.getFrameInfo()->CreateFixedObject(SlotSize, FPDiff-SlotSize);
  MVT::ValueType VT = Is64Bit ? MVT::i64 : MVT::i32;
  SDOperand NewRetAddrFrIdx = DAG.getFrameIndex(NewReturnAddrFI, VT);
  Chain = DAG.getStore(Chain, RetAddrFrIdx, NewRetAddrFrIdx, 
                       PseudoSourceValue::getFixedStack(), NewReturnAddrFI);
  return Chain;
}

SDOperand X86TargetLowering::LowerCALL(SDOperand Op, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  SDOperand Chain     = Op.getOperand(0);
  unsigned CC         = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  bool isVarArg       = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  bool IsTailCall     = cast<ConstantSDNode>(Op.getOperand(3))->getValue() != 0
                        && CC == CallingConv::Fast && PerformTailCallOpt;
  SDOperand Callee    = Op.getOperand(4);
  bool Is64Bit        = Subtarget->is64Bit();
  bool IsStructRet    = CallIsStructReturn(Op);

  assert(!(isVarArg && CC == CallingConv::Fast) &&
         "Var args not supported with calling convention fastcc");

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);
  CCInfo.AnalyzeCallOperands(Op.Val, CCAssignFnForNode(Op));
  
  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();
  if (CC == CallingConv::Fast)
    NumBytes = GetAlignedArgumentStackSize(NumBytes, DAG);

  // Make sure the instruction takes 8n+4 bytes to make sure the start of the
  // arguments and the arguments after the retaddr has been pushed are aligned.
  if (!Is64Bit && CC == CallingConv::X86_FastCall &&
      !Subtarget->isTargetCygMing() && !Subtarget->isTargetWindows() &&
      (NumBytes & 7) == 0)
    NumBytes += 4;

  int FPDiff = 0;
  if (IsTailCall) {
    // Lower arguments at fp - stackoffset + fpdiff.
    unsigned NumBytesCallerPushed = 
      MF.getInfo<X86MachineFunctionInfo>()->getBytesToPopOnReturn();
    FPDiff = NumBytesCallerPushed - NumBytes;

    // Set the delta of movement of the returnaddr stackslot.
    // But only set if delta is greater than previous delta.
    if (FPDiff < (MF.getInfo<X86MachineFunctionInfo>()->getTCReturnAddrDelta()))
      MF.getInfo<X86MachineFunctionInfo>()->setTCReturnAddrDelta(FPDiff);
  }

  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes));

  SDOperand RetAddrFrIdx;
  // Load return adress for tail calls.
  Chain = EmitTailCallLoadRetAddr(DAG, RetAddrFrIdx, Chain, IsTailCall, Is64Bit,
                                  FPDiff);

  SmallVector<std::pair<unsigned, SDOperand>, 8> RegsToPass;
  SmallVector<SDOperand, 8> MemOpChains;
  SDOperand StackPtr;

  // Walk the register/memloc assignments, inserting copies/loads.  In the case
  // of tail call optimization arguments are handle later.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDOperand Arg = Op.getOperand(5+2*VA.getValNo());
    bool isByVal = cast<ARG_FLAGSSDNode>(Op.getOperand(6+2*VA.getValNo()))->
      getArgFlags().isByVal();
  
    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default: assert(0 && "Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, VA.getLocVT(), Arg);
      break;
    }
    
    if (VA.isRegLoc()) {
      if (Is64Bit) {
        MVT::ValueType RegVT = VA.getLocVT();
        if (MVT::isVector(RegVT) && MVT::getSizeInBits(RegVT) == 64)
          switch (VA.getLocReg()) {
          default:
            break;
          case X86::RDI: case X86::RSI: case X86::RDX: case X86::RCX:
          case X86::R8: {
            // Special case: passing MMX values in GPR registers.
            Arg = DAG.getNode(ISD::BIT_CONVERT, MVT::i64, Arg);
            break;
          }
          case X86::XMM0: case X86::XMM1: case X86::XMM2: case X86::XMM3:
          case X86::XMM4: case X86::XMM5: case X86::XMM6: case X86::XMM7: {
            // Special case: passing MMX values in XMM registers.
            Arg = DAG.getNode(ISD::BIT_CONVERT, MVT::i64, Arg);
            Arg = DAG.getNode(ISD::SCALAR_TO_VECTOR, MVT::v2i64, Arg);
            Arg = DAG.getNode(ISD::VECTOR_SHUFFLE, MVT::v2i64,
                              DAG.getNode(ISD::UNDEF, MVT::v2i64), Arg,
                              getMOVLMask(2, DAG));
            break;
          }
          }
      }
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      if (!IsTailCall || (IsTailCall && isByVal)) {
        assert(VA.isMemLoc());
        if (StackPtr.Val == 0)
          StackPtr = DAG.getCopyFromReg(Chain, X86StackPtr, getPointerTy());
        
        MemOpChains.push_back(LowerMemOpCallTo(Op, DAG, StackPtr, VA, Chain,
                                               Arg));
      }
    }
  }
  
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into registers.
  SDOperand InFlag;
  // Tail call byval lowering might overwrite argument registers so in case of
  // tail call optimization the copies to registers are lowered later.
  if (!IsTailCall)
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
      Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                               InFlag);
      InFlag = Chain.getValue(1);
    }

  // ELF / PIC requires GOT in the EBX register before function calls via PLT
  // GOT pointer.  
  if (CallRequiresGOTPtrInReg(Is64Bit, IsTailCall)) {
    Chain = DAG.getCopyToReg(Chain, X86::EBX,
                             DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()),
                             InFlag);
    InFlag = Chain.getValue(1);
  }
  // If we are tail calling and generating PIC/GOT style code load the address
  // of the callee into ecx. The value in ecx is used as target of the tail
  // jump. This is done to circumvent the ebx/callee-saved problem for tail
  // calls on PIC/GOT architectures. Normally we would just put the address of
  // GOT into ebx and then call target@PLT. But for tail callss ebx would be
  // restored (since ebx is callee saved) before jumping to the target@PLT.
  if (CallRequiresFnAddressInReg(Is64Bit, IsTailCall)) {
    // Note: The actual moving to ecx is done further down.
    GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee);
    if (G &&  !G->getGlobal()->hasHiddenVisibility() &&
        !G->getGlobal()->hasProtectedVisibility())
      Callee =  LowerGlobalAddress(Callee, DAG);
    else if (isa<ExternalSymbolSDNode>(Callee))
      Callee = LowerExternalSymbol(Callee,DAG);
  }

  if (Is64Bit && isVarArg) {
    // From AMD64 ABI document:
    // For calls that may call functions that use varargs or stdargs
    // (prototype-less calls or calls to functions containing ellipsis (...) in
    // the declaration) %al is used as hidden argument to specify the number
    // of SSE registers used. The contents of %al do not need to match exactly
    // the number of registers, but must be an ubound on the number of SSE
    // registers used and is in the range 0 - 8 inclusive.

    // FIXME: Verify this on Win64
    // Count the number of XMM registers allocated.
    static const unsigned XMMArgRegs[] = {
      X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
      X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
    };
    unsigned NumXMMRegs = CCInfo.getFirstUnallocated(XMMArgRegs, 8);
    
    Chain = DAG.getCopyToReg(Chain, X86::AL,
                             DAG.getConstant(NumXMMRegs, MVT::i8), InFlag);
    InFlag = Chain.getValue(1);
  }


  // For tail calls lower the arguments to the 'real' stack slot.
  if (IsTailCall) {
    SmallVector<SDOperand, 8> MemOpChains2;
    SDOperand FIN;
    int FI = 0;
    // Do not flag preceeding copytoreg stuff together with the following stuff.
    InFlag = SDOperand();
    for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
      CCValAssign &VA = ArgLocs[i];
      if (!VA.isRegLoc()) {
        assert(VA.isMemLoc());
        SDOperand Arg = Op.getOperand(5+2*VA.getValNo());
        SDOperand FlagsOp = Op.getOperand(6+2*VA.getValNo());
        ISD::ArgFlagsTy Flags =
          cast<ARG_FLAGSSDNode>(FlagsOp)->getArgFlags();
        // Create frame index.
        int32_t Offset = VA.getLocMemOffset()+FPDiff;
        uint32_t OpSize = (MVT::getSizeInBits(VA.getLocVT())+7)/8;
        FI = MF.getFrameInfo()->CreateFixedObject(OpSize, Offset);
        FIN = DAG.getFrameIndex(FI, getPointerTy());

        if (Flags.isByVal()) {
          // Copy relative to framepointer.
          SDOperand Source = DAG.getIntPtrConstant(VA.getLocMemOffset());
          if (StackPtr.Val == 0)
            StackPtr = DAG.getCopyFromReg(Chain, X86StackPtr, getPointerTy());
          Source = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, Source);

          MemOpChains2.push_back(CreateCopyOfByValArgument(Source, FIN, Chain,
                                                           Flags, DAG));
        } else {
          // Store relative to framepointer.
          MemOpChains2.push_back(
            DAG.getStore(Chain, Arg, FIN,
                         PseudoSourceValue::getFixedStack(), FI));
        }            
      }
    }

    if (!MemOpChains2.empty())
      Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                          &MemOpChains2[0], MemOpChains2.size());

    // Copy arguments to their registers.
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
      Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                               InFlag);
      InFlag = Chain.getValue(1);
    }
    InFlag =SDOperand();

    // Store the return address to the appropriate stack slot.
    Chain = EmitTailCallStoreRetAddr(DAG, MF, Chain, RetAddrFrIdx, Is64Bit,
                                     FPDiff);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    // We should use extra load for direct calls to dllimported functions in
    // non-JIT mode.
    if ((IsTailCall || !Is64Bit ||
         getTargetMachine().getCodeModel() != CodeModel::Large)
        && !Subtarget->GVRequiresExtraLoad(G->getGlobal(),
                                           getTargetMachine(), true))
      Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    if (IsTailCall || !Is64Bit ||
        getTargetMachine().getCodeModel() != CodeModel::Large)
      Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());
  } else if (IsTailCall) {
    unsigned Opc = Is64Bit ? X86::R9 : X86::ECX;

    Chain = DAG.getCopyToReg(Chain, 
                             DAG.getRegister(Opc, getPointerTy()), 
                             Callee,InFlag);
    Callee = DAG.getRegister(Opc, getPointerTy());
    // Add register as live out.
    DAG.getMachineFunction().getRegInfo().addLiveOut(Opc);
  }
 
  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
  SmallVector<SDOperand, 8> Ops;

  if (IsTailCall) {
    Ops.push_back(Chain);
    Ops.push_back(DAG.getIntPtrConstant(NumBytes));
    Ops.push_back(DAG.getIntPtrConstant(0));
    if (InFlag.Val)
      Ops.push_back(InFlag);
    Chain = DAG.getNode(ISD::CALLSEQ_END, NodeTys, &Ops[0], Ops.size());
    InFlag = Chain.getValue(1);
 
    // Returns a chain & a flag for retval copy to use.
    NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
    Ops.clear();
  }
  
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  if (IsTailCall)
    Ops.push_back(DAG.getConstant(FPDiff, MVT::i32));

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));
  
  // Add an implicit use GOT pointer in EBX.
  if (!IsTailCall && !Is64Bit &&
      getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
      Subtarget->isPICStyleGOT())
    Ops.push_back(DAG.getRegister(X86::EBX, getPointerTy()));

  // Add an implicit use of AL for x86 vararg functions.
  if (Is64Bit && isVarArg)
    Ops.push_back(DAG.getRegister(X86::AL, MVT::i8));

  if (InFlag.Val)
    Ops.push_back(InFlag);

  if (IsTailCall) {
    assert(InFlag.Val && 
           "Flag must be set. Depend on flag being set in LowerRET");
    Chain = DAG.getNode(X86ISD::TAILCALL,
                        Op.Val->getVTList(), &Ops[0], Ops.size());
      
    return SDOperand(Chain.Val, Op.ResNo);
  }

  Chain = DAG.getNode(X86ISD::CALL, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  unsigned NumBytesForCalleeToPush;
  if (IsCalleePop(Op))
    NumBytesForCalleeToPush = NumBytes;    // Callee pops everything
  else if (!Is64Bit && IsStructRet)
    // If this is is a call to a struct-return function, the callee
    // pops the hidden struct pointer, so we have to push it back.
    // This is common for Darwin/X86, Linux & Mingw32 targets.
    NumBytesForCalleeToPush = 4;
  else
    NumBytesForCalleeToPush = 0;  // Callee pops nothing.
  
  // Returns a flag for retval copy to use.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getIntPtrConstant(NumBytes),
                             DAG.getIntPtrConstant(NumBytesForCalleeToPush),
                             InFlag);
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return SDOperand(LowerCallResult(Chain, InFlag, Op.Val, CC, DAG), Op.ResNo);
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
unsigned X86TargetLowering::GetAlignedArgumentStackSize(unsigned StackSize, 
                                                        SelectionDAG& DAG) {
  if (PerformTailCallOpt) {
    MachineFunction &MF = DAG.getMachineFunction();
    const TargetMachine &TM = MF.getTarget();
    const TargetFrameInfo &TFI = *TM.getFrameInfo();
    unsigned StackAlignment = TFI.getStackAlignment();
    uint64_t AlignMask = StackAlignment - 1; 
    int64_t Offset = StackSize;
    unsigned SlotSize = Subtarget->is64Bit() ? 8 : 4;
    if ( (Offset & AlignMask) <= (StackAlignment - SlotSize) ) {
      // Number smaller than 12 so just add the difference.
      Offset += ((StackAlignment - SlotSize) - (Offset & AlignMask));
    } else {
      // Mask out lower bits, add stackalignment once plus the 12 bytes.
      Offset = ((~AlignMask) & Offset) + StackAlignment + 
        (StackAlignment-SlotSize);
    }
    StackSize = Offset;
  }
  return StackSize;
}

/// IsEligibleForTailCallElimination - Check to see whether the next instruction
/// following the call is a return. A function is eligible if caller/callee
/// calling conventions match, currently only fastcc supports tail calls, and
/// the function CALL is immediatly followed by a RET.
bool X86TargetLowering::IsEligibleForTailCallOptimization(SDOperand Call,
                                                      SDOperand Ret,
                                                      SelectionDAG& DAG) const {
  if (!PerformTailCallOpt)
    return false;

  if (CheckTailCallReturnConstraints(Call, Ret)) {
    MachineFunction &MF = DAG.getMachineFunction();
    unsigned CallerCC = MF.getFunction()->getCallingConv();
    unsigned CalleeCC = cast<ConstantSDNode>(Call.getOperand(1))->getValue();
    if (CalleeCC == CallingConv::Fast && CallerCC == CalleeCC) {
      SDOperand Callee = Call.getOperand(4);
      // On x86/32Bit PIC/GOT  tail calls are supported.
      if (getTargetMachine().getRelocationModel() != Reloc::PIC_ ||
          !Subtarget->isPICStyleGOT()|| !Subtarget->is64Bit())
        return true;

      // Can only do local tail calls (in same module, hidden or protected) on
      // x86_64 PIC/GOT at the moment.
      if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
        return G->getGlobal()->hasHiddenVisibility()
            || G->getGlobal()->hasProtectedVisibility();
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
//                           Other Lowering Hooks
//===----------------------------------------------------------------------===//


SDOperand X86TargetLowering::getReturnAddressFrameIndex(SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
  int ReturnAddrIndex = FuncInfo->getRAIndex();

  if (ReturnAddrIndex == 0) {
    // Set up a frame object for the return address.
    if (Subtarget->is64Bit())
      ReturnAddrIndex = MF.getFrameInfo()->CreateFixedObject(8, -8);
    else
      ReturnAddrIndex = MF.getFrameInfo()->CreateFixedObject(4, -4);

    FuncInfo->setRAIndex(ReturnAddrIndex);
  }

  return DAG.getFrameIndex(ReturnAddrIndex, getPointerTy());
}



/// translateX86CC - do a one to one translation of a ISD::CondCode to the X86
/// specific condition code. It returns a false if it cannot do a direct
/// translation. X86CC is the translated CondCode.  LHS/RHS are modified as
/// needed.
static bool translateX86CC(ISD::CondCode SetCCOpcode, bool isFP,
                           unsigned &X86CC, SDOperand &LHS, SDOperand &RHS,
                           SelectionDAG &DAG) {
  X86CC = X86::COND_INVALID;
  if (!isFP) {
    if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS)) {
      if (SetCCOpcode == ISD::SETGT && RHSC->isAllOnesValue()) {
        // X > -1   -> X == 0, jump !sign.
        RHS = DAG.getConstant(0, RHS.getValueType());
        X86CC = X86::COND_NS;
        return true;
      } else if (SetCCOpcode == ISD::SETLT && RHSC->isNullValue()) {
        // X < 0   -> X == 0, jump on sign.
        X86CC = X86::COND_S;
        return true;
      } else if (SetCCOpcode == ISD::SETLT && RHSC->getValue() == 1) {
        // X < 1   -> X <= 0
        RHS = DAG.getConstant(0, RHS.getValueType());
        X86CC = X86::COND_LE;
        return true;
      }
    }

    switch (SetCCOpcode) {
    default: break;
    case ISD::SETEQ:  X86CC = X86::COND_E;  break;
    case ISD::SETGT:  X86CC = X86::COND_G;  break;
    case ISD::SETGE:  X86CC = X86::COND_GE; break;
    case ISD::SETLT:  X86CC = X86::COND_L;  break;
    case ISD::SETLE:  X86CC = X86::COND_LE; break;
    case ISD::SETNE:  X86CC = X86::COND_NE; break;
    case ISD::SETULT: X86CC = X86::COND_B;  break;
    case ISD::SETUGT: X86CC = X86::COND_A;  break;
    case ISD::SETULE: X86CC = X86::COND_BE; break;
    case ISD::SETUGE: X86CC = X86::COND_AE; break;
    }
  } else {
    // On a floating point condition, the flags are set as follows:
    // ZF  PF  CF   op
    //  0 | 0 | 0 | X > Y
    //  0 | 0 | 1 | X < Y
    //  1 | 0 | 0 | X == Y
    //  1 | 1 | 1 | unordered
    bool Flip = false;
    switch (SetCCOpcode) {
    default: break;
    case ISD::SETUEQ:
    case ISD::SETEQ: X86CC = X86::COND_E;  break;
    case ISD::SETOLT: Flip = true; // Fallthrough
    case ISD::SETOGT:
    case ISD::SETGT: X86CC = X86::COND_A;  break;
    case ISD::SETOLE: Flip = true; // Fallthrough
    case ISD::SETOGE:
    case ISD::SETGE: X86CC = X86::COND_AE; break;
    case ISD::SETUGT: Flip = true; // Fallthrough
    case ISD::SETULT:
    case ISD::SETLT: X86CC = X86::COND_B;  break;
    case ISD::SETUGE: Flip = true; // Fallthrough
    case ISD::SETULE:
    case ISD::SETLE: X86CC = X86::COND_BE; break;
    case ISD::SETONE:
    case ISD::SETNE: X86CC = X86::COND_NE; break;
    case ISD::SETUO: X86CC = X86::COND_P;  break;
    case ISD::SETO:  X86CC = X86::COND_NP; break;
    }
    if (Flip)
      std::swap(LHS, RHS);
  }

  return X86CC != X86::COND_INVALID;
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

/// isUndefOrInRange - Op is either an undef node or a ConstantSDNode.  Return
/// true if Op is undef or if its value falls within the specified range (L, H].
static bool isUndefOrInRange(SDOperand Op, unsigned Low, unsigned Hi) {
  if (Op.getOpcode() == ISD::UNDEF)
    return true;

  unsigned Val = cast<ConstantSDNode>(Op)->getValue();
  return (Val >= Low && Val < Hi);
}

/// isUndefOrEqual - Op is either an undef node or a ConstantSDNode.  Return
/// true if Op is undef or if its value equal to the specified value.
static bool isUndefOrEqual(SDOperand Op, unsigned Val) {
  if (Op.getOpcode() == ISD::UNDEF)
    return true;
  return cast<ConstantSDNode>(Op)->getValue() == Val;
}

/// isPSHUFDMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to PSHUFD.
bool X86::isPSHUFDMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  if (N->getNumOperands() != 2 && N->getNumOperands() != 4)
    return false;

  // Check if the value doesn't reference the second vector.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    if (cast<ConstantSDNode>(Arg)->getValue() >= e)
      return false;
  }

  return true;
}

/// isPSHUFHWMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to PSHUFHW.
bool X86::isPSHUFHWMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  if (N->getNumOperands() != 8)
    return false;

  // Lower quadword copied in order.
  for (unsigned i = 0; i != 4; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    if (cast<ConstantSDNode>(Arg)->getValue() != i)
      return false;
  }

  // Upper quadword shuffled.
  for (unsigned i = 4; i != 8; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val < 4 || Val > 7)
      return false;
  }

  return true;
}

/// isPSHUFLWMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to PSHUFLW.
bool X86::isPSHUFLWMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  if (N->getNumOperands() != 8)
    return false;

  // Upper quadword copied in order.
  for (unsigned i = 4; i != 8; ++i)
    if (!isUndefOrEqual(N->getOperand(i), i))
      return false;

  // Lower quadword shuffled.
  for (unsigned i = 0; i != 4; ++i)
    if (!isUndefOrInRange(N->getOperand(i), 0, 4))
      return false;

  return true;
}

/// isSHUFPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to SHUFP*.
static bool isSHUFPMask(SDOperandPtr Elems, unsigned NumElems) {
  if (NumElems != 2 && NumElems != 4) return false;

  unsigned Half = NumElems / 2;
  for (unsigned i = 0; i < Half; ++i)
    if (!isUndefOrInRange(Elems[i], 0, NumElems))
      return false;
  for (unsigned i = Half; i < NumElems; ++i)
    if (!isUndefOrInRange(Elems[i], NumElems, NumElems*2))
      return false;

  return true;
}

bool X86::isSHUFPMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  return ::isSHUFPMask(N->op_begin(), N->getNumOperands());
}

/// isCommutedSHUFP - Returns true if the shuffle mask is exactly
/// the reverse of what x86 shuffles want. x86 shuffles requires the lower
/// half elements to come from vector 1 (which would equal the dest.) and
/// the upper half to come from vector 2.
static bool isCommutedSHUFP(SDOperandPtr Ops, unsigned NumOps) {
  if (NumOps != 2 && NumOps != 4) return false;

  unsigned Half = NumOps / 2;
  for (unsigned i = 0; i < Half; ++i)
    if (!isUndefOrInRange(Ops[i], NumOps, NumOps*2))
      return false;
  for (unsigned i = Half; i < NumOps; ++i)
    if (!isUndefOrInRange(Ops[i], 0, NumOps))
      return false;
  return true;
}

static bool isCommutedSHUFP(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  return isCommutedSHUFP(N->op_begin(), N->getNumOperands());
}

/// isMOVHLPSMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVHLPS.
bool X86::isMOVHLPSMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  if (N->getNumOperands() != 4)
    return false;

  // Expect bit0 == 6, bit1 == 7, bit2 == 2, bit3 == 3
  return isUndefOrEqual(N->getOperand(0), 6) &&
         isUndefOrEqual(N->getOperand(1), 7) &&
         isUndefOrEqual(N->getOperand(2), 2) &&
         isUndefOrEqual(N->getOperand(3), 3);
}

/// isMOVHLPS_v_undef_Mask - Special case of isMOVHLPSMask for canonical form
/// of vector_shuffle v, v, <2, 3, 2, 3>, i.e. vector_shuffle v, undef,
/// <2, 3, 2, 3>
bool X86::isMOVHLPS_v_undef_Mask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  if (N->getNumOperands() != 4)
    return false;

  // Expect bit0 == 2, bit1 == 3, bit2 == 2, bit3 == 3
  return isUndefOrEqual(N->getOperand(0), 2) &&
         isUndefOrEqual(N->getOperand(1), 3) &&
         isUndefOrEqual(N->getOperand(2), 2) &&
         isUndefOrEqual(N->getOperand(3), 3);
}

/// isMOVLPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVLP{S|D}.
bool X86::isMOVLPMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  unsigned NumElems = N->getNumOperands();
  if (NumElems != 2 && NumElems != 4)
    return false;

  for (unsigned i = 0; i < NumElems/2; ++i)
    if (!isUndefOrEqual(N->getOperand(i), i + NumElems))
      return false;

  for (unsigned i = NumElems/2; i < NumElems; ++i)
    if (!isUndefOrEqual(N->getOperand(i), i))
      return false;

  return true;
}

/// isMOVHPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVHP{S|D}
/// and MOVLHPS.
bool X86::isMOVHPMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  unsigned NumElems = N->getNumOperands();
  if (NumElems != 2 && NumElems != 4)
    return false;

  for (unsigned i = 0; i < NumElems/2; ++i)
    if (!isUndefOrEqual(N->getOperand(i), i))
      return false;

  for (unsigned i = 0; i < NumElems/2; ++i) {
    SDOperand Arg = N->getOperand(i + NumElems/2);
    if (!isUndefOrEqual(Arg, i + NumElems))
      return false;
  }

  return true;
}

/// isUNPCKLMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to UNPCKL.
bool static isUNPCKLMask(SDOperandPtr Elts, unsigned NumElts,
                         bool V2IsSplat = false) {
  if (NumElts != 2 && NumElts != 4 && NumElts != 8 && NumElts != 16)
    return false;

  for (unsigned i = 0, j = 0; i != NumElts; i += 2, ++j) {
    SDOperand BitI  = Elts[i];
    SDOperand BitI1 = Elts[i+1];
    if (!isUndefOrEqual(BitI, j))
      return false;
    if (V2IsSplat) {
      if (isUndefOrEqual(BitI1, NumElts))
        return false;
    } else {
      if (!isUndefOrEqual(BitI1, j + NumElts))
        return false;
    }
  }

  return true;
}

bool X86::isUNPCKLMask(SDNode *N, bool V2IsSplat) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  return ::isUNPCKLMask(N->op_begin(), N->getNumOperands(), V2IsSplat);
}

/// isUNPCKHMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to UNPCKH.
bool static isUNPCKHMask(SDOperandPtr Elts, unsigned NumElts,
                         bool V2IsSplat = false) {
  if (NumElts != 2 && NumElts != 4 && NumElts != 8 && NumElts != 16)
    return false;

  for (unsigned i = 0, j = 0; i != NumElts; i += 2, ++j) {
    SDOperand BitI  = Elts[i];
    SDOperand BitI1 = Elts[i+1];
    if (!isUndefOrEqual(BitI, j + NumElts/2))
      return false;
    if (V2IsSplat) {
      if (isUndefOrEqual(BitI1, NumElts))
        return false;
    } else {
      if (!isUndefOrEqual(BitI1, j + NumElts/2 + NumElts))
        return false;
    }
  }

  return true;
}

bool X86::isUNPCKHMask(SDNode *N, bool V2IsSplat) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  return ::isUNPCKHMask(N->op_begin(), N->getNumOperands(), V2IsSplat);
}

/// isUNPCKL_v_undef_Mask - Special case of isUNPCKLMask for canonical form
/// of vector_shuffle v, v, <0, 4, 1, 5>, i.e. vector_shuffle v, undef,
/// <0, 0, 1, 1>
bool X86::isUNPCKL_v_undef_Mask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  unsigned NumElems = N->getNumOperands();
  if (NumElems != 2 && NumElems != 4 && NumElems != 8 && NumElems != 16)
    return false;

  for (unsigned i = 0, j = 0; i != NumElems; i += 2, ++j) {
    SDOperand BitI  = N->getOperand(i);
    SDOperand BitI1 = N->getOperand(i+1);

    if (!isUndefOrEqual(BitI, j))
      return false;
    if (!isUndefOrEqual(BitI1, j))
      return false;
  }

  return true;
}

/// isUNPCKH_v_undef_Mask - Special case of isUNPCKHMask for canonical form
/// of vector_shuffle v, v, <2, 6, 3, 7>, i.e. vector_shuffle v, undef,
/// <2, 2, 3, 3>
bool X86::isUNPCKH_v_undef_Mask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  unsigned NumElems = N->getNumOperands();
  if (NumElems != 2 && NumElems != 4 && NumElems != 8 && NumElems != 16)
    return false;

  for (unsigned i = 0, j = NumElems / 2; i != NumElems; i += 2, ++j) {
    SDOperand BitI  = N->getOperand(i);
    SDOperand BitI1 = N->getOperand(i + 1);

    if (!isUndefOrEqual(BitI, j))
      return false;
    if (!isUndefOrEqual(BitI1, j))
      return false;
  }

  return true;
}

/// isMOVLMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSS,
/// MOVSD, and MOVD, i.e. setting the lowest element.
static bool isMOVLMask(SDOperandPtr Elts, unsigned NumElts) {
  if (NumElts != 2 && NumElts != 4)
    return false;

  if (!isUndefOrEqual(Elts[0], NumElts))
    return false;

  for (unsigned i = 1; i < NumElts; ++i) {
    if (!isUndefOrEqual(Elts[i], i))
      return false;
  }

  return true;
}

bool X86::isMOVLMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  return ::isMOVLMask(N->op_begin(), N->getNumOperands());
}

/// isCommutedMOVL - Returns true if the shuffle mask is except the reverse
/// of what x86 movss want. X86 movs requires the lowest  element to be lowest
/// element of vector 2 and the other elements to come from vector 1 in order.
static bool isCommutedMOVL(SDOperandPtr Ops, unsigned NumOps,
                           bool V2IsSplat = false,
                           bool V2IsUndef = false) {
  if (NumOps != 2 && NumOps != 4 && NumOps != 8 && NumOps != 16)
    return false;

  if (!isUndefOrEqual(Ops[0], 0))
    return false;

  for (unsigned i = 1; i < NumOps; ++i) {
    SDOperand Arg = Ops[i];
    if (!(isUndefOrEqual(Arg, i+NumOps) ||
          (V2IsUndef && isUndefOrInRange(Arg, NumOps, NumOps*2)) ||
          (V2IsSplat && isUndefOrEqual(Arg, NumOps))))
      return false;
  }

  return true;
}

static bool isCommutedMOVL(SDNode *N, bool V2IsSplat = false,
                           bool V2IsUndef = false) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  return isCommutedMOVL(N->op_begin(), N->getNumOperands(),
                        V2IsSplat, V2IsUndef);
}

/// isMOVSHDUPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSHDUP.
bool X86::isMOVSHDUPMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  if (N->getNumOperands() != 4)
    return false;

  // Expect 1, 1, 3, 3
  for (unsigned i = 0; i < 2; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val != 1) return false;
  }

  bool HasHi = false;
  for (unsigned i = 2; i < 4; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val != 3) return false;
    HasHi = true;
  }

  // Don't use movshdup if it can be done with a shufps.
  return HasHi;
}

/// isMOVSLDUPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSLDUP.
bool X86::isMOVSLDUPMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  if (N->getNumOperands() != 4)
    return false;

  // Expect 0, 0, 2, 2
  for (unsigned i = 0; i < 2; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val != 0) return false;
  }

  bool HasHi = false;
  for (unsigned i = 2; i < 4; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val != 2) return false;
    HasHi = true;
  }

  // Don't use movshdup if it can be done with a shufps.
  return HasHi;
}

/// isIdentityMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a identity operation on the LHS or RHS.
static bool isIdentityMask(SDNode *N, bool RHS = false) {
  unsigned NumElems = N->getNumOperands();
  for (unsigned i = 0; i < NumElems; ++i)
    if (!isUndefOrEqual(N->getOperand(i), i + (RHS ? NumElems : 0)))
      return false;
  return true;
}

/// isSplatMask - Return true if the specified VECTOR_SHUFFLE operand specifies
/// a splat of a single element.
static bool isSplatMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  // This is a splat operation if each element of the permute is the same, and
  // if the value doesn't reference the second vector.
  unsigned NumElems = N->getNumOperands();
  SDOperand ElementBase;
  unsigned i = 0;
  for (; i != NumElems; ++i) {
    SDOperand Elt = N->getOperand(i);
    if (isa<ConstantSDNode>(Elt)) {
      ElementBase = Elt;
      break;
    }
  }

  if (!ElementBase.Val)
    return false;

  for (; i != NumElems; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    if (Arg != ElementBase) return false;
  }

  // Make sure it is a splat of the first vector operand.
  return cast<ConstantSDNode>(ElementBase)->getValue() < NumElems;
}

/// isSplatMask - Return true if the specified VECTOR_SHUFFLE operand specifies
/// a splat of a single element and it's a 2 or 4 element mask.
bool X86::isSplatMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  // We can only splat 64-bit, and 32-bit quantities with a single instruction.
  if (N->getNumOperands() != 4 && N->getNumOperands() != 2)
    return false;
  return ::isSplatMask(N);
}

/// isSplatLoMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a splat of zero element.
bool X86::isSplatLoMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  for (unsigned i = 0, e = N->getNumOperands(); i < e; ++i)
    if (!isUndefOrEqual(N->getOperand(i), 0))
      return false;
  return true;
}

/// getShuffleSHUFImmediate - Return the appropriate immediate to shuffle
/// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUF* and SHUFP*
/// instructions.
unsigned X86::getShuffleSHUFImmediate(SDNode *N) {
  unsigned NumOperands = N->getNumOperands();
  unsigned Shift = (NumOperands == 4) ? 2 : 1;
  unsigned Mask = 0;
  for (unsigned i = 0; i < NumOperands; ++i) {
    unsigned Val = 0;
    SDOperand Arg = N->getOperand(NumOperands-i-1);
    if (Arg.getOpcode() != ISD::UNDEF)
      Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val >= NumOperands) Val -= NumOperands;
    Mask |= Val;
    if (i != NumOperands - 1)
      Mask <<= Shift;
  }

  return Mask;
}

/// getShufflePSHUFHWImmediate - Return the appropriate immediate to shuffle
/// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUFHW
/// instructions.
unsigned X86::getShufflePSHUFHWImmediate(SDNode *N) {
  unsigned Mask = 0;
  // 8 nodes, but we only care about the last 4.
  for (unsigned i = 7; i >= 4; --i) {
    unsigned Val = 0;
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() != ISD::UNDEF)
      Val = cast<ConstantSDNode>(Arg)->getValue();
    Mask |= (Val - 4);
    if (i != 4)
      Mask <<= 2;
  }

  return Mask;
}

/// getShufflePSHUFLWImmediate - Return the appropriate immediate to shuffle
/// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUFLW
/// instructions.
unsigned X86::getShufflePSHUFLWImmediate(SDNode *N) {
  unsigned Mask = 0;
  // 8 nodes, but we only care about the first 4.
  for (int i = 3; i >= 0; --i) {
    unsigned Val = 0;
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() != ISD::UNDEF)
      Val = cast<ConstantSDNode>(Arg)->getValue();
    Mask |= Val;
    if (i != 0)
      Mask <<= 2;
  }

  return Mask;
}

/// isPSHUFHW_PSHUFLWMask - true if the specified VECTOR_SHUFFLE operand
/// specifies a 8 element shuffle that can be broken into a pair of
/// PSHUFHW and PSHUFLW.
static bool isPSHUFHW_PSHUFLWMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  if (N->getNumOperands() != 8)
    return false;

  // Lower quadword shuffled.
  for (unsigned i = 0; i != 4; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val >= 4)
      return false;
  }

  // Upper quadword shuffled.
  for (unsigned i = 4; i != 8; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val < 4 || Val > 7)
      return false;
  }

  return true;
}

/// CommuteVectorShuffle - Swap vector_shuffle operands as well as
/// values in ther permute mask.
static SDOperand CommuteVectorShuffle(SDOperand Op, SDOperand &V1,
                                      SDOperand &V2, SDOperand &Mask,
                                      SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType MaskVT = Mask.getValueType();
  MVT::ValueType EltVT = MVT::getVectorElementType(MaskVT);
  unsigned NumElems = Mask.getNumOperands();
  SmallVector<SDOperand, 8> MaskVec;

  for (unsigned i = 0; i != NumElems; ++i) {
    SDOperand Arg = Mask.getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) {
      MaskVec.push_back(DAG.getNode(ISD::UNDEF, EltVT));
      continue;
    }
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val < NumElems)
      MaskVec.push_back(DAG.getConstant(Val + NumElems, EltVT));
    else
      MaskVec.push_back(DAG.getConstant(Val - NumElems, EltVT));
  }

  std::swap(V1, V2);
  Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], NumElems);
  return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2, Mask);
}

/// CommuteVectorShuffleMask - Change values in a shuffle permute mask assuming
/// the two vector operands have swapped position.
static
SDOperand CommuteVectorShuffleMask(SDOperand Mask, SelectionDAG &DAG) {
  MVT::ValueType MaskVT = Mask.getValueType();
  MVT::ValueType EltVT = MVT::getVectorElementType(MaskVT);
  unsigned NumElems = Mask.getNumOperands();
  SmallVector<SDOperand, 8> MaskVec;
  for (unsigned i = 0; i != NumElems; ++i) {
    SDOperand Arg = Mask.getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) {
      MaskVec.push_back(DAG.getNode(ISD::UNDEF, EltVT));
      continue;
    }
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
    if (Val < NumElems)
      MaskVec.push_back(DAG.getConstant(Val + NumElems, EltVT));
    else
      MaskVec.push_back(DAG.getConstant(Val - NumElems, EltVT));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], NumElems);
}


/// ShouldXformToMOVHLPS - Return true if the node should be transformed to
/// match movhlps. The lower half elements should come from upper half of
/// V1 (and in order), and the upper half elements should come from the upper
/// half of V2 (and in order).
static bool ShouldXformToMOVHLPS(SDNode *Mask) {
  unsigned NumElems = Mask->getNumOperands();
  if (NumElems != 4)
    return false;
  for (unsigned i = 0, e = 2; i != e; ++i)
    if (!isUndefOrEqual(Mask->getOperand(i), i+2))
      return false;
  for (unsigned i = 2; i != 4; ++i)
    if (!isUndefOrEqual(Mask->getOperand(i), i+4))
      return false;
  return true;
}

/// isScalarLoadToVector - Returns true if the node is a scalar load that
/// is promoted to a vector. It also returns the LoadSDNode by reference if
/// required.
static bool isScalarLoadToVector(SDNode *N, LoadSDNode **LD = NULL) {
  if (N->getOpcode() == ISD::SCALAR_TO_VECTOR) {
    N = N->getOperand(0).Val;
    if (ISD::isNON_EXTLoad(N)) {
      if (LD)
        *LD = cast<LoadSDNode>(N);
      return true;
    }
  }
  return false;
}

/// ShouldXformToMOVLP{S|D} - Return true if the node should be transformed to
/// match movlp{s|d}. The lower half elements should come from lower half of
/// V1 (and in order), and the upper half elements should come from the upper
/// half of V2 (and in order). And since V1 will become the source of the
/// MOVLP, it must be either a vector load or a scalar load to vector.
static bool ShouldXformToMOVLP(SDNode *V1, SDNode *V2, SDNode *Mask) {
  if (!ISD::isNON_EXTLoad(V1) && !isScalarLoadToVector(V1))
    return false;
  // Is V2 is a vector load, don't do this transformation. We will try to use
  // load folding shufps op.
  if (ISD::isNON_EXTLoad(V2))
    return false;

  unsigned NumElems = Mask->getNumOperands();
  if (NumElems != 2 && NumElems != 4)
    return false;
  for (unsigned i = 0, e = NumElems/2; i != e; ++i)
    if (!isUndefOrEqual(Mask->getOperand(i), i))
      return false;
  for (unsigned i = NumElems/2; i != NumElems; ++i)
    if (!isUndefOrEqual(Mask->getOperand(i), i+NumElems))
      return false;
  return true;
}

/// isSplatVector - Returns true if N is a BUILD_VECTOR node whose elements are
/// all the same.
static bool isSplatVector(SDNode *N) {
  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  SDOperand SplatValue = N->getOperand(0);
  for (unsigned i = 1, e = N->getNumOperands(); i != e; ++i)
    if (N->getOperand(i) != SplatValue)
      return false;
  return true;
}

/// isUndefShuffle - Returns true if N is a VECTOR_SHUFFLE that can be resolved
/// to an undef.
static bool isUndefShuffle(SDNode *N) {
  if (N->getOpcode() != ISD::VECTOR_SHUFFLE)
    return false;

  SDOperand V1 = N->getOperand(0);
  SDOperand V2 = N->getOperand(1);
  SDOperand Mask = N->getOperand(2);
  unsigned NumElems = Mask.getNumOperands();
  for (unsigned i = 0; i != NumElems; ++i) {
    SDOperand Arg = Mask.getOperand(i);
    if (Arg.getOpcode() != ISD::UNDEF) {
      unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
      if (Val < NumElems && V1.getOpcode() != ISD::UNDEF)
        return false;
      else if (Val >= NumElems && V2.getOpcode() != ISD::UNDEF)
        return false;
    }
  }
  return true;
}

/// isZeroNode - Returns true if Elt is a constant zero or a floating point
/// constant +0.0.
static inline bool isZeroNode(SDOperand Elt) {
  return ((isa<ConstantSDNode>(Elt) &&
           cast<ConstantSDNode>(Elt)->getValue() == 0) ||
          (isa<ConstantFPSDNode>(Elt) &&
           cast<ConstantFPSDNode>(Elt)->getValueAPF().isPosZero()));
}

/// isZeroShuffle - Returns true if N is a VECTOR_SHUFFLE that can be resolved
/// to an zero vector.
static bool isZeroShuffle(SDNode *N) {
  if (N->getOpcode() != ISD::VECTOR_SHUFFLE)
    return false;

  SDOperand V1 = N->getOperand(0);
  SDOperand V2 = N->getOperand(1);
  SDOperand Mask = N->getOperand(2);
  unsigned NumElems = Mask.getNumOperands();
  for (unsigned i = 0; i != NumElems; ++i) {
    SDOperand Arg = Mask.getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF)
      continue;
    
    unsigned Idx = cast<ConstantSDNode>(Arg)->getValue();
    if (Idx < NumElems) {
      unsigned Opc = V1.Val->getOpcode();
      if (Opc == ISD::UNDEF || ISD::isBuildVectorAllZeros(V1.Val))
        continue;
      if (Opc != ISD::BUILD_VECTOR ||
          !isZeroNode(V1.Val->getOperand(Idx)))
        return false;
    } else if (Idx >= NumElems) {
      unsigned Opc = V2.Val->getOpcode();
      if (Opc == ISD::UNDEF || ISD::isBuildVectorAllZeros(V2.Val))
        continue;
      if (Opc != ISD::BUILD_VECTOR ||
          !isZeroNode(V2.Val->getOperand(Idx - NumElems)))
        return false;
    }
  }
  return true;
}

/// getZeroVector - Returns a vector of specified type with all zero elements.
///
static SDOperand getZeroVector(MVT::ValueType VT, SelectionDAG &DAG) {
  assert(MVT::isVector(VT) && "Expected a vector type");
  
  // Always build zero vectors as <4 x i32> or <2 x i32> bitcasted to their dest
  // type.  This ensures they get CSE'd.
  SDOperand Cst = DAG.getTargetConstant(0, MVT::i32);
  SDOperand Vec;
  if (MVT::getSizeInBits(VT) == 64)  // MMX
    Vec = DAG.getNode(ISD::BUILD_VECTOR, MVT::v2i32, Cst, Cst);
  else                                              // SSE
    Vec = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, Cst, Cst, Cst, Cst);
  return DAG.getNode(ISD::BIT_CONVERT, VT, Vec);
}

/// getOnesVector - Returns a vector of specified type with all bits set.
///
static SDOperand getOnesVector(MVT::ValueType VT, SelectionDAG &DAG) {
  assert(MVT::isVector(VT) && "Expected a vector type");
  
  // Always build ones vectors as <4 x i32> or <2 x i32> bitcasted to their dest
  // type.  This ensures they get CSE'd.
  SDOperand Cst = DAG.getTargetConstant(~0U, MVT::i32);
  SDOperand Vec;
  if (MVT::getSizeInBits(VT) == 64)  // MMX
    Vec = DAG.getNode(ISD::BUILD_VECTOR, MVT::v2i32, Cst, Cst);
  else                                              // SSE
    Vec = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, Cst, Cst, Cst, Cst);
  return DAG.getNode(ISD::BIT_CONVERT, VT, Vec);
}


/// NormalizeMask - V2 is a splat, modify the mask (if needed) so all elements
/// that point to V2 points to its first element.
static SDOperand NormalizeMask(SDOperand Mask, SelectionDAG &DAG) {
  assert(Mask.getOpcode() == ISD::BUILD_VECTOR);

  bool Changed = false;
  SmallVector<SDOperand, 8> MaskVec;
  unsigned NumElems = Mask.getNumOperands();
  for (unsigned i = 0; i != NumElems; ++i) {
    SDOperand Arg = Mask.getOperand(i);
    if (Arg.getOpcode() != ISD::UNDEF) {
      unsigned Val = cast<ConstantSDNode>(Arg)->getValue();
      if (Val > NumElems) {
        Arg = DAG.getConstant(NumElems, Arg.getValueType());
        Changed = true;
      }
    }
    MaskVec.push_back(Arg);
  }

  if (Changed)
    Mask = DAG.getNode(ISD::BUILD_VECTOR, Mask.getValueType(),
                       &MaskVec[0], MaskVec.size());
  return Mask;
}

/// getMOVLMask - Returns a vector_shuffle mask for an movs{s|d}, movd
/// operation of specified width.
static SDOperand getMOVLMask(unsigned NumElems, SelectionDAG &DAG) {
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
  MVT::ValueType BaseVT = MVT::getVectorElementType(MaskVT);

  SmallVector<SDOperand, 8> MaskVec;
  MaskVec.push_back(DAG.getConstant(NumElems, BaseVT));
  for (unsigned i = 1; i != NumElems; ++i)
    MaskVec.push_back(DAG.getConstant(i, BaseVT));
  return DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], MaskVec.size());
}

/// getUnpacklMask - Returns a vector_shuffle mask for an unpackl operation
/// of specified width.
static SDOperand getUnpacklMask(unsigned NumElems, SelectionDAG &DAG) {
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
  MVT::ValueType BaseVT = MVT::getVectorElementType(MaskVT);
  SmallVector<SDOperand, 8> MaskVec;
  for (unsigned i = 0, e = NumElems/2; i != e; ++i) {
    MaskVec.push_back(DAG.getConstant(i,            BaseVT));
    MaskVec.push_back(DAG.getConstant(i + NumElems, BaseVT));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], MaskVec.size());
}

/// getUnpackhMask - Returns a vector_shuffle mask for an unpackh operation
/// of specified width.
static SDOperand getUnpackhMask(unsigned NumElems, SelectionDAG &DAG) {
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
  MVT::ValueType BaseVT = MVT::getVectorElementType(MaskVT);
  unsigned Half = NumElems/2;
  SmallVector<SDOperand, 8> MaskVec;
  for (unsigned i = 0; i != Half; ++i) {
    MaskVec.push_back(DAG.getConstant(i + Half,            BaseVT));
    MaskVec.push_back(DAG.getConstant(i + NumElems + Half, BaseVT));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], MaskVec.size());
}

/// getSwapEltZeroMask - Returns a vector_shuffle mask for a shuffle that swaps
/// element #0 of a vector with the specified index, leaving the rest of the
/// elements in place.
static SDOperand getSwapEltZeroMask(unsigned NumElems, unsigned DestElt,
                                   SelectionDAG &DAG) {
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
  MVT::ValueType BaseVT = MVT::getVectorElementType(MaskVT);
  SmallVector<SDOperand, 8> MaskVec;
  // Element #0 of the result gets the elt we are replacing.
  MaskVec.push_back(DAG.getConstant(DestElt, BaseVT));
  for (unsigned i = 1; i != NumElems; ++i)
    MaskVec.push_back(DAG.getConstant(i == DestElt ? 0 : i, BaseVT));
  return DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], MaskVec.size());
}

/// PromoteSplat - Promote a splat of v4f32, v8i16 or v16i8 to v4i32.
static SDOperand PromoteSplat(SDOperand Op, SelectionDAG &DAG, bool HasSSE2) {
  MVT::ValueType PVT = HasSSE2 ? MVT::v4i32 : MVT::v4f32;
  MVT::ValueType VT = Op.getValueType();
  if (PVT == VT)
    return Op;
  SDOperand V1 = Op.getOperand(0);
  SDOperand Mask = Op.getOperand(2);
  unsigned NumElems = Mask.getNumOperands();
  // Special handling of v4f32 -> v4i32.
  if (VT != MVT::v4f32) {
    Mask = getUnpacklMask(NumElems, DAG);
    while (NumElems > 4) {
      V1 = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V1, Mask);
      NumElems >>= 1;
    }
    Mask = getZeroVector(MVT::v4i32, DAG);
  }

  V1 = DAG.getNode(ISD::BIT_CONVERT, PVT, V1);
  SDOperand Shuffle = DAG.getNode(ISD::VECTOR_SHUFFLE, PVT, V1,
                                  DAG.getNode(ISD::UNDEF, PVT), Mask);
  return DAG.getNode(ISD::BIT_CONVERT, VT, Shuffle);
}

/// getShuffleVectorZeroOrUndef - Return a vector_shuffle of the specified
/// vector of zero or undef vector.  This produces a shuffle where the low
/// element of V2 is swizzled into the zero/undef vector, landing at element
/// Idx.  This produces a shuffle mask like 4,1,2,3 (idx=0) or  0,1,2,4 (idx=3).
static SDOperand getShuffleVectorZeroOrUndef(SDOperand V2, unsigned Idx,
                                             bool isZero, SelectionDAG &DAG) {
  MVT::ValueType VT = V2.getValueType();
  SDOperand V1 = isZero ? getZeroVector(VT, DAG) : DAG.getNode(ISD::UNDEF, VT);
  unsigned NumElems = MVT::getVectorNumElements(V2.getValueType());
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
  MVT::ValueType EVT = MVT::getVectorElementType(MaskVT);
  SmallVector<SDOperand, 16> MaskVec;
  for (unsigned i = 0; i != NumElems; ++i)
    if (i == Idx)  // If this is the insertion idx, put the low elt of V2 here.
      MaskVec.push_back(DAG.getConstant(NumElems, EVT));
    else
      MaskVec.push_back(DAG.getConstant(i, EVT));
  SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                               &MaskVec[0], MaskVec.size());
  return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2, Mask);
}

/// LowerBuildVectorv16i8 - Custom lower build_vector of v16i8.
///
static SDOperand LowerBuildVectorv16i8(SDOperand Op, unsigned NonZeros,
                                       unsigned NumNonZero, unsigned NumZero,
                                       SelectionDAG &DAG, TargetLowering &TLI) {
  if (NumNonZero > 8)
    return SDOperand();

  SDOperand V(0, 0);
  bool First = true;
  for (unsigned i = 0; i < 16; ++i) {
    bool ThisIsNonZero = (NonZeros & (1 << i)) != 0;
    if (ThisIsNonZero && First) {
      if (NumZero)
        V = getZeroVector(MVT::v8i16, DAG);
      else
        V = DAG.getNode(ISD::UNDEF, MVT::v8i16);
      First = false;
    }

    if ((i & 1) != 0) {
      SDOperand ThisElt(0, 0), LastElt(0, 0);
      bool LastIsNonZero = (NonZeros & (1 << (i-1))) != 0;
      if (LastIsNonZero) {
        LastElt = DAG.getNode(ISD::ZERO_EXTEND, MVT::i16, Op.getOperand(i-1));
      }
      if (ThisIsNonZero) {
        ThisElt = DAG.getNode(ISD::ZERO_EXTEND, MVT::i16, Op.getOperand(i));
        ThisElt = DAG.getNode(ISD::SHL, MVT::i16,
                              ThisElt, DAG.getConstant(8, MVT::i8));
        if (LastIsNonZero)
          ThisElt = DAG.getNode(ISD::OR, MVT::i16, ThisElt, LastElt);
      } else
        ThisElt = LastElt;

      if (ThisElt.Val)
        V = DAG.getNode(ISD::INSERT_VECTOR_ELT, MVT::v8i16, V, ThisElt,
                        DAG.getIntPtrConstant(i/2));
    }
  }

  return DAG.getNode(ISD::BIT_CONVERT, MVT::v16i8, V);
}

/// LowerBuildVectorv8i16 - Custom lower build_vector of v8i16.
///
static SDOperand LowerBuildVectorv8i16(SDOperand Op, unsigned NonZeros,
                                       unsigned NumNonZero, unsigned NumZero,
                                       SelectionDAG &DAG, TargetLowering &TLI) {
  if (NumNonZero > 4)
    return SDOperand();

  SDOperand V(0, 0);
  bool First = true;
  for (unsigned i = 0; i < 8; ++i) {
    bool isNonZero = (NonZeros & (1 << i)) != 0;
    if (isNonZero) {
      if (First) {
        if (NumZero)
          V = getZeroVector(MVT::v8i16, DAG);
        else
          V = DAG.getNode(ISD::UNDEF, MVT::v8i16);
        First = false;
      }
      V = DAG.getNode(ISD::INSERT_VECTOR_ELT, MVT::v8i16, V, Op.getOperand(i),
                      DAG.getIntPtrConstant(i));
    }
  }

  return V;
}

SDOperand
X86TargetLowering::LowerBUILD_VECTOR(SDOperand Op, SelectionDAG &DAG) {
  // All zero's are handled with pxor, all one's are handled with pcmpeqd.
  if (ISD::isBuildVectorAllZeros(Op.Val) || ISD::isBuildVectorAllOnes(Op.Val)) {
    // Canonicalize this to either <4 x i32> or <2 x i32> (SSE vs MMX) to
    // 1) ensure the zero vectors are CSE'd, and 2) ensure that i64 scalars are
    // eliminated on x86-32 hosts.
    if (Op.getValueType() == MVT::v4i32 || Op.getValueType() == MVT::v2i32)
      return Op;

    if (ISD::isBuildVectorAllOnes(Op.Val))
      return getOnesVector(Op.getValueType(), DAG);
    return getZeroVector(Op.getValueType(), DAG);
  }

  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType EVT = MVT::getVectorElementType(VT);
  unsigned EVTBits = MVT::getSizeInBits(EVT);

  unsigned NumElems = Op.getNumOperands();
  unsigned NumZero  = 0;
  unsigned NumNonZero = 0;
  unsigned NonZeros = 0;
  bool IsAllConstants = true;
  SmallSet<SDOperand, 8> Values;
  for (unsigned i = 0; i < NumElems; ++i) {
    SDOperand Elt = Op.getOperand(i);
    if (Elt.getOpcode() == ISD::UNDEF)
      continue;
    Values.insert(Elt);
    if (Elt.getOpcode() != ISD::Constant &&
        Elt.getOpcode() != ISD::ConstantFP)
      IsAllConstants = false;
    if (isZeroNode(Elt))
      NumZero++;
    else {
      NonZeros |= (1 << i);
      NumNonZero++;
    }
  }

  if (NumNonZero == 0) {
    // All undef vector. Return an UNDEF.  All zero vectors were handled above.
    return DAG.getNode(ISD::UNDEF, VT);
  }

  // Special case for single non-zero, non-undef, element.
  if (NumNonZero == 1 && NumElems <= 4) {
    unsigned Idx = CountTrailingZeros_32(NonZeros);
    SDOperand Item = Op.getOperand(Idx);
    
    // If this is an insertion of an i64 value on x86-32, and if the top bits of
    // the value are obviously zero, truncate the value to i32 and do the
    // insertion that way.  Only do this if the value is non-constant or if the
    // value is a constant being inserted into element 0.  It is cheaper to do
    // a constant pool load than it is to do a movd + shuffle.
    if (EVT == MVT::i64 && !Subtarget->is64Bit() &&
        (!IsAllConstants || Idx == 0)) {
      if (DAG.MaskedValueIsZero(Item, APInt::getBitsSet(64, 32, 64))) {
        // Handle MMX and SSE both.
        MVT::ValueType VecVT = VT == MVT::v2i64 ? MVT::v4i32 : MVT::v2i32;
        MVT::ValueType VecElts = VT == MVT::v2i64 ? 4 : 2;
        
        // Truncate the value (which may itself be a constant) to i32, and
        // convert it to a vector with movd (S2V+shuffle to zero extend).
        Item = DAG.getNode(ISD::TRUNCATE, MVT::i32, Item);
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, VecVT, Item);
        Item = getShuffleVectorZeroOrUndef(Item, 0, true, DAG);
        
        // Now we have our 32-bit value zero extended in the low element of
        // a vector.  If Idx != 0, swizzle it into place.
        if (Idx != 0) {
          SDOperand Ops[] = { 
            Item, DAG.getNode(ISD::UNDEF, Item.getValueType()),
            getSwapEltZeroMask(VecElts, Idx, DAG)
          };
          Item = DAG.getNode(ISD::VECTOR_SHUFFLE, VecVT, Ops, 3);
        }
        return DAG.getNode(ISD::BIT_CONVERT, Op.getValueType(), Item);
      }
    }
    
    // If we have a constant or non-constant insertion into the low element of
    // a vector, we can do this with SCALAR_TO_VECTOR + shuffle of zero into
    // the rest of the elements.  This will be matched as movd/movq/movss/movsd
    // depending on what the source datatype is.  Because we can only get here
    // when NumElems <= 4, this only needs to handle i32/f32/i64/f64.
    if (Idx == 0 &&
        // Don't do this for i64 values on x86-32.
        (EVT != MVT::i64 || Subtarget->is64Bit())) {
      Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, Item);
      // Turn it into a MOVL (i.e. movss, movsd, or movd) to a zero vector.
      return getShuffleVectorZeroOrUndef(Item, 0, NumZero > 0, DAG);
    }
    
    if (IsAllConstants) // Otherwise, it's better to do a constpool load.
      return SDOperand();

    // Otherwise, if this is a vector with i32 or f32 elements, and the element
    // is a non-constant being inserted into an element other than the low one,
    // we can't use a constant pool load.  Instead, use SCALAR_TO_VECTOR (aka
    // movd/movss) to move this into the low element, then shuffle it into
    // place.
    if (EVTBits == 32) {
      Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, Item);
      
      // Turn it into a shuffle of zero and zero-extended scalar to vector.
      Item = getShuffleVectorZeroOrUndef(Item, 0, NumZero > 0, DAG);
      MVT::ValueType MaskVT  = MVT::getIntVectorWithNumElements(NumElems);
      MVT::ValueType MaskEVT = MVT::getVectorElementType(MaskVT);
      SmallVector<SDOperand, 8> MaskVec;
      for (unsigned i = 0; i < NumElems; i++)
        MaskVec.push_back(DAG.getConstant((i == Idx) ? 0 : 1, MaskEVT));
      SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                   &MaskVec[0], MaskVec.size());
      return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, Item,
                         DAG.getNode(ISD::UNDEF, VT), Mask);
    }
  }

  // Splat is obviously ok. Let legalizer expand it to a shuffle.
  if (Values.size() == 1)
    return SDOperand();
  
  // A vector full of immediates; various special cases are already
  // handled, so this is best done with a single constant-pool load.
  if (IsAllConstants)
    return SDOperand();

  // Let legalizer expand 2-wide build_vectors.
  if (EVTBits == 64) {
    if (NumNonZero == 1) {
      // One half is zero or undef.
      unsigned Idx = CountTrailingZeros_32(NonZeros);
      SDOperand V2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, VT,
                                 Op.getOperand(Idx));
      return getShuffleVectorZeroOrUndef(V2, Idx, true, DAG);
    }
    return SDOperand();
  }

  // If element VT is < 32 bits, convert it to inserts into a zero vector.
  if (EVTBits == 8 && NumElems == 16) {
    SDOperand V = LowerBuildVectorv16i8(Op, NonZeros,NumNonZero,NumZero, DAG,
                                        *this);
    if (V.Val) return V;
  }

  if (EVTBits == 16 && NumElems == 8) {
    SDOperand V = LowerBuildVectorv8i16(Op, NonZeros,NumNonZero,NumZero, DAG,
                                        *this);
    if (V.Val) return V;
  }

  // If element VT is == 32 bits, turn it into a number of shuffles.
  SmallVector<SDOperand, 8> V;
  V.resize(NumElems);
  if (NumElems == 4 && NumZero > 0) {
    for (unsigned i = 0; i < 4; ++i) {
      bool isZero = !(NonZeros & (1 << i));
      if (isZero)
        V[i] = getZeroVector(VT, DAG);
      else
        V[i] = DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, Op.getOperand(i));
    }

    for (unsigned i = 0; i < 2; ++i) {
      switch ((NonZeros & (0x3 << i*2)) >> (i*2)) {
        default: break;
        case 0:
          V[i] = V[i*2];  // Must be a zero vector.
          break;
        case 1:
          V[i] = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V[i*2+1], V[i*2],
                             getMOVLMask(NumElems, DAG));
          break;
        case 2:
          V[i] = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V[i*2], V[i*2+1],
                             getMOVLMask(NumElems, DAG));
          break;
        case 3:
          V[i] = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V[i*2], V[i*2+1],
                             getUnpacklMask(NumElems, DAG));
          break;
      }
    }

    MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
    MVT::ValueType EVT = MVT::getVectorElementType(MaskVT);
    SmallVector<SDOperand, 8> MaskVec;
    bool Reverse = (NonZeros & 0x3) == 2;
    for (unsigned i = 0; i < 2; ++i)
      if (Reverse)
        MaskVec.push_back(DAG.getConstant(1-i, EVT));
      else
        MaskVec.push_back(DAG.getConstant(i, EVT));
    Reverse = ((NonZeros & (0x3 << 2)) >> 2) == 2;
    for (unsigned i = 0; i < 2; ++i)
      if (Reverse)
        MaskVec.push_back(DAG.getConstant(1-i+NumElems, EVT));
      else
        MaskVec.push_back(DAG.getConstant(i+NumElems, EVT));
    SDOperand ShufMask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                     &MaskVec[0], MaskVec.size());
    return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V[0], V[1], ShufMask);
  }

  if (Values.size() > 2) {
    // Expand into a number of unpckl*.
    // e.g. for v4f32
    //   Step 1: unpcklps 0, 2 ==> X: <?, ?, 2, 0>
    //         : unpcklps 1, 3 ==> Y: <?, ?, 3, 1>
    //   Step 2: unpcklps X, Y ==>    <3, 2, 1, 0>
    SDOperand UnpckMask = getUnpacklMask(NumElems, DAG);
    for (unsigned i = 0; i < NumElems; ++i)
      V[i] = DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, Op.getOperand(i));
    NumElems >>= 1;
    while (NumElems != 0) {
      for (unsigned i = 0; i < NumElems; ++i)
        V[i] = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V[i], V[i + NumElems],
                           UnpckMask);
      NumElems >>= 1;
    }
    return V[0];
  }

  return SDOperand();
}

static
SDOperand LowerVECTOR_SHUFFLEv8i16(SDOperand V1, SDOperand V2,
                                   SDOperand PermMask, SelectionDAG &DAG,
                                   TargetLowering &TLI) {
  SDOperand NewV;
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(8);
  MVT::ValueType MaskEVT = MVT::getVectorElementType(MaskVT);
  MVT::ValueType PtrVT = TLI.getPointerTy();
  SmallVector<SDOperand, 8> MaskElts(PermMask.Val->op_begin(),
                                     PermMask.Val->op_end());

  // First record which half of which vector the low elements come from.
  SmallVector<unsigned, 4> LowQuad(4);
  for (unsigned i = 0; i < 4; ++i) {
    SDOperand Elt = MaskElts[i];
    if (Elt.getOpcode() == ISD::UNDEF)
      continue;
    unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
    int QuadIdx = EltIdx / 4;
    ++LowQuad[QuadIdx];
  }
  int BestLowQuad = -1;
  unsigned MaxQuad = 1;
  for (unsigned i = 0; i < 4; ++i) {
    if (LowQuad[i] > MaxQuad) {
      BestLowQuad = i;
      MaxQuad = LowQuad[i];
    }
  }

  // Record which half of which vector the high elements come from.
  SmallVector<unsigned, 4> HighQuad(4);
  for (unsigned i = 4; i < 8; ++i) {
    SDOperand Elt = MaskElts[i];
    if (Elt.getOpcode() == ISD::UNDEF)
      continue;
    unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
    int QuadIdx = EltIdx / 4;
    ++HighQuad[QuadIdx];
  }
  int BestHighQuad = -1;
  MaxQuad = 1;
  for (unsigned i = 0; i < 4; ++i) {
    if (HighQuad[i] > MaxQuad) {
      BestHighQuad = i;
      MaxQuad = HighQuad[i];
    }
  }

  // If it's possible to sort parts of either half with PSHUF{H|L}W, then do it.
  if (BestLowQuad != -1 || BestHighQuad != -1) {
    // First sort the 4 chunks in order using shufpd.
    SmallVector<SDOperand, 8> MaskVec;
    if (BestLowQuad != -1)
      MaskVec.push_back(DAG.getConstant(BestLowQuad, MVT::i32));
    else
      MaskVec.push_back(DAG.getConstant(0, MVT::i32));
    if (BestHighQuad != -1)
      MaskVec.push_back(DAG.getConstant(BestHighQuad, MVT::i32));
    else
      MaskVec.push_back(DAG.getConstant(1, MVT::i32));
    SDOperand Mask= DAG.getNode(ISD::BUILD_VECTOR, MVT::v2i32, &MaskVec[0],2);
    NewV = DAG.getNode(ISD::VECTOR_SHUFFLE, MVT::v2i64,
                       DAG.getNode(ISD::BIT_CONVERT, MVT::v2i64, V1),
                       DAG.getNode(ISD::BIT_CONVERT, MVT::v2i64, V2), Mask);
    NewV = DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, NewV);

    // Now sort high and low parts separately.
    BitVector InOrder(8);
    if (BestLowQuad != -1) {
      // Sort lower half in order using PSHUFLW.
      MaskVec.clear();
      bool AnyOutOrder = false;
      for (unsigned i = 0; i != 4; ++i) {
        SDOperand Elt = MaskElts[i];
        if (Elt.getOpcode() == ISD::UNDEF) {
          MaskVec.push_back(Elt);
          InOrder.set(i);
        } else {
          unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
          if (EltIdx != i)
            AnyOutOrder = true;
          MaskVec.push_back(DAG.getConstant(EltIdx % 4, MaskEVT));
          // If this element is in the right place after this shuffle, then
          // remember it.
          if ((int)(EltIdx / 4) == BestLowQuad)
            InOrder.set(i);
        }
      }
      if (AnyOutOrder) {
        for (unsigned i = 4; i != 8; ++i)
          MaskVec.push_back(DAG.getConstant(i, MaskEVT));
        SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], 8);
        NewV = DAG.getNode(ISD::VECTOR_SHUFFLE, MVT::v8i16, NewV, NewV, Mask);
      }
    }

    if (BestHighQuad != -1) {
      // Sort high half in order using PSHUFHW if possible.
      MaskVec.clear();
      for (unsigned i = 0; i != 4; ++i)
        MaskVec.push_back(DAG.getConstant(i, MaskEVT));
      bool AnyOutOrder = false;
      for (unsigned i = 4; i != 8; ++i) {
        SDOperand Elt = MaskElts[i];
        if (Elt.getOpcode() == ISD::UNDEF) {
          MaskVec.push_back(Elt);
          InOrder.set(i);
        } else {
          unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
          if (EltIdx != i)
            AnyOutOrder = true;
          MaskVec.push_back(DAG.getConstant((EltIdx % 4) + 4, MaskEVT));
          // If this element is in the right place after this shuffle, then
          // remember it.
          if ((int)(EltIdx / 4) == BestHighQuad)
            InOrder.set(i);
        }
      }
      if (AnyOutOrder) {
        SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], 8);
        NewV = DAG.getNode(ISD::VECTOR_SHUFFLE, MVT::v8i16, NewV, NewV, Mask);
      }
    }

    // The other elements are put in the right place using pextrw and pinsrw.
    for (unsigned i = 0; i != 8; ++i) {
      if (InOrder[i])
        continue;
      SDOperand Elt = MaskElts[i];
      unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
      if (EltIdx == i)
        continue;
      SDOperand ExtOp = (EltIdx < 8)
        ? DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i16, V1,
                      DAG.getConstant(EltIdx, PtrVT))
        : DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i16, V2,
                      DAG.getConstant(EltIdx - 8, PtrVT));
      NewV = DAG.getNode(ISD::INSERT_VECTOR_ELT, MVT::v8i16, NewV, ExtOp,
                         DAG.getConstant(i, PtrVT));
    }
    return NewV;
  }

  // PSHUF{H|L}W are not used. Lower into extracts and inserts but try to use
  ///as few as possible.
  // First, let's find out how many elements are already in the right order.
  unsigned V1InOrder = 0;
  unsigned V1FromV1 = 0;
  unsigned V2InOrder = 0;
  unsigned V2FromV2 = 0;
  SmallVector<SDOperand, 8> V1Elts;
  SmallVector<SDOperand, 8> V2Elts;
  for (unsigned i = 0; i < 8; ++i) {
    SDOperand Elt = MaskElts[i];
    if (Elt.getOpcode() == ISD::UNDEF) {
      V1Elts.push_back(Elt);
      V2Elts.push_back(Elt);
      ++V1InOrder;
      ++V2InOrder;
      continue;
    }
    unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
    if (EltIdx == i) {
      V1Elts.push_back(Elt);
      V2Elts.push_back(DAG.getConstant(i+8, MaskEVT));
      ++V1InOrder;
    } else if (EltIdx == i+8) {
      V1Elts.push_back(Elt);
      V2Elts.push_back(DAG.getConstant(i, MaskEVT));
      ++V2InOrder;
    } else if (EltIdx < 8) {
      V1Elts.push_back(Elt);
      ++V1FromV1;
    } else {
      V2Elts.push_back(DAG.getConstant(EltIdx-8, MaskEVT));
      ++V2FromV2;
    }
  }

  if (V2InOrder > V1InOrder) {
    PermMask = CommuteVectorShuffleMask(PermMask, DAG);
    std::swap(V1, V2);
    std::swap(V1Elts, V2Elts);
    std::swap(V1FromV1, V2FromV2);
  }

  if ((V1FromV1 + V1InOrder) != 8) {
    // Some elements are from V2.
    if (V1FromV1) {
      // If there are elements that are from V1 but out of place,
      // then first sort them in place
      SmallVector<SDOperand, 8> MaskVec;
      for (unsigned i = 0; i < 8; ++i) {
        SDOperand Elt = V1Elts[i];
        if (Elt.getOpcode() == ISD::UNDEF) {
          MaskVec.push_back(DAG.getNode(ISD::UNDEF, MaskEVT));
          continue;
        }
        unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
        if (EltIdx >= 8)
          MaskVec.push_back(DAG.getNode(ISD::UNDEF, MaskEVT));
        else
          MaskVec.push_back(DAG.getConstant(EltIdx, MaskEVT));
      }
      SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], 8);
      V1 = DAG.getNode(ISD::VECTOR_SHUFFLE, MVT::v8i16, V1, V1, Mask);
    }

    NewV = V1;
    for (unsigned i = 0; i < 8; ++i) {
      SDOperand Elt = V1Elts[i];
      if (Elt.getOpcode() == ISD::UNDEF)
        continue;
      unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
      if (EltIdx < 8)
        continue;
      SDOperand ExtOp = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i16, V2,
                                    DAG.getConstant(EltIdx - 8, PtrVT));
      NewV = DAG.getNode(ISD::INSERT_VECTOR_ELT, MVT::v8i16, NewV, ExtOp,
                         DAG.getConstant(i, PtrVT));
    }
    return NewV;
  } else {
    // All elements are from V1.
    NewV = V1;
    for (unsigned i = 0; i < 8; ++i) {
      SDOperand Elt = V1Elts[i];
      if (Elt.getOpcode() == ISD::UNDEF)
        continue;
      unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
      SDOperand ExtOp = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i16, V1,
                                    DAG.getConstant(EltIdx, PtrVT));
      NewV = DAG.getNode(ISD::INSERT_VECTOR_ELT, MVT::v8i16, NewV, ExtOp,
                         DAG.getConstant(i, PtrVT));
    }
    return NewV;
  }
}

/// RewriteAsNarrowerShuffle - Try rewriting v8i16 and v16i8 shuffles as 4 wide
/// ones, or rewriting v4i32 / v2f32 as 2 wide ones if possible. This can be
/// done when every pair / quad of shuffle mask elements point to elements in
/// the right sequence. e.g.
/// vector_shuffle <>, <>, < 3, 4, | 10, 11, | 0, 1, | 14, 15>
static
SDOperand RewriteAsNarrowerShuffle(SDOperand V1, SDOperand V2,
                                MVT::ValueType VT,
                                SDOperand PermMask, SelectionDAG &DAG,
                                TargetLowering &TLI) {
  unsigned NumElems = PermMask.getNumOperands();
  unsigned NewWidth = (NumElems == 4) ? 2 : 4;
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NewWidth);
  MVT::ValueType NewVT = MaskVT;
  switch (VT) {
  case MVT::v4f32: NewVT = MVT::v2f64; break;
  case MVT::v4i32: NewVT = MVT::v2i64; break;
  case MVT::v8i16: NewVT = MVT::v4i32; break;
  case MVT::v16i8: NewVT = MVT::v4i32; break;
  default: assert(false && "Unexpected!");
  }

  if (NewWidth == 2) {
    if (MVT::isInteger(VT))
      NewVT = MVT::v2i64;
    else
      NewVT = MVT::v2f64;
  }
  unsigned Scale = NumElems / NewWidth;
  SmallVector<SDOperand, 8> MaskVec;
  for (unsigned i = 0; i < NumElems; i += Scale) {
    unsigned StartIdx = ~0U;
    for (unsigned j = 0; j < Scale; ++j) {
      SDOperand Elt = PermMask.getOperand(i+j);
      if (Elt.getOpcode() == ISD::UNDEF)
        continue;
      unsigned EltIdx = cast<ConstantSDNode>(Elt)->getValue();
      if (StartIdx == ~0U)
        StartIdx = EltIdx - (EltIdx % Scale);
      if (EltIdx != StartIdx + j)
        return SDOperand();
    }
    if (StartIdx == ~0U)
      MaskVec.push_back(DAG.getNode(ISD::UNDEF, MVT::i32));
    else
      MaskVec.push_back(DAG.getConstant(StartIdx / Scale, MVT::i32));
  }

  V1 = DAG.getNode(ISD::BIT_CONVERT, NewVT, V1);
  V2 = DAG.getNode(ISD::BIT_CONVERT, NewVT, V2);
  return DAG.getNode(ISD::VECTOR_SHUFFLE, NewVT, V1, V2,
                     DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                 &MaskVec[0], MaskVec.size()));
}

/// getVZextMovL - Return a zero-extending vector move low node.
///
static SDOperand getVZextMovL(MVT::ValueType VT, MVT::ValueType OpVT,
                               SDOperand SrcOp, SelectionDAG &DAG,
                               const X86Subtarget *Subtarget) {
  if (VT == MVT::v2f64 || VT == MVT::v4f32) {
    LoadSDNode *LD = NULL;
    if (!isScalarLoadToVector(SrcOp.Val, &LD))
      LD = dyn_cast<LoadSDNode>(SrcOp);
    if (!LD) {
      // movssrr and movsdrr do not clear top bits. Try to use movd, movq
      // instead.
      MVT::ValueType EVT = (OpVT == MVT::v2f64) ? MVT::i64 : MVT::i32;
      if ((EVT != MVT::i64 || Subtarget->is64Bit()) &&
          SrcOp.getOpcode() == ISD::SCALAR_TO_VECTOR &&
          SrcOp.getOperand(0).getOpcode() == ISD::BIT_CONVERT &&
          SrcOp.getOperand(0).getOperand(0).getValueType() == EVT) {
        // PR2108
        OpVT = (OpVT == MVT::v2f64) ? MVT::v2i64 : MVT::v4i32;
        return DAG.getNode(ISD::BIT_CONVERT, VT,
                           DAG.getNode(X86ISD::VZEXT_MOVL, OpVT,
                                       DAG.getNode(ISD::SCALAR_TO_VECTOR, OpVT,
                                                   SrcOp.getOperand(0).getOperand(0))));
      }
    }
  }

  return DAG.getNode(ISD::BIT_CONVERT, VT,
                     DAG.getNode(X86ISD::VZEXT_MOVL, OpVT,
                                 DAG.getNode(ISD::BIT_CONVERT, OpVT, SrcOp)));
}

SDOperand
X86TargetLowering::LowerVECTOR_SHUFFLE(SDOperand Op, SelectionDAG &DAG) {
  SDOperand V1 = Op.getOperand(0);
  SDOperand V2 = Op.getOperand(1);
  SDOperand PermMask = Op.getOperand(2);
  MVT::ValueType VT = Op.getValueType();
  unsigned NumElems = PermMask.getNumOperands();
  bool isMMX = MVT::getSizeInBits(VT) == 64;
  bool V1IsUndef = V1.getOpcode() == ISD::UNDEF;
  bool V2IsUndef = V2.getOpcode() == ISD::UNDEF;
  bool V1IsSplat = false;
  bool V2IsSplat = false;

  if (isUndefShuffle(Op.Val))
    return DAG.getNode(ISD::UNDEF, VT);

  if (isZeroShuffle(Op.Val))
    return getZeroVector(VT, DAG);

  if (isIdentityMask(PermMask.Val))
    return V1;
  else if (isIdentityMask(PermMask.Val, true))
    return V2;

  if (isSplatMask(PermMask.Val)) {
    if (isMMX || NumElems < 4) return Op;
    // Promote it to a v4{if}32 splat.
    return PromoteSplat(Op, DAG, Subtarget->hasSSE2());
  }

  // If the shuffle can be profitably rewritten as a narrower shuffle, then
  // do it!
  if (VT == MVT::v8i16 || VT == MVT::v16i8) {
    SDOperand NewOp= RewriteAsNarrowerShuffle(V1, V2, VT, PermMask, DAG, *this);
    if (NewOp.Val)
      return DAG.getNode(ISD::BIT_CONVERT, VT, LowerVECTOR_SHUFFLE(NewOp, DAG));
  } else if ((VT == MVT::v4i32 || (VT == MVT::v4f32 && Subtarget->hasSSE2()))) {
    // FIXME: Figure out a cleaner way to do this.
    // Try to make use of movq to zero out the top part.
    if (ISD::isBuildVectorAllZeros(V2.Val)) {
      SDOperand NewOp = RewriteAsNarrowerShuffle(V1, V2, VT, PermMask,
                                                 DAG, *this);
      if (NewOp.Val) {
        SDOperand NewV1 = NewOp.getOperand(0);
        SDOperand NewV2 = NewOp.getOperand(1);
        SDOperand NewMask = NewOp.getOperand(2);
        if (isCommutedMOVL(NewMask.Val, true, false)) {
          NewOp = CommuteVectorShuffle(NewOp, NewV1, NewV2, NewMask, DAG);
          return getVZextMovL(VT, NewOp.getValueType(), NewV2, DAG, Subtarget);
        }
      }
    } else if (ISD::isBuildVectorAllZeros(V1.Val)) {
      SDOperand NewOp= RewriteAsNarrowerShuffle(V1, V2, VT, PermMask,
                                                DAG, *this);
      if (NewOp.Val && X86::isMOVLMask(NewOp.getOperand(2).Val))
        return getVZextMovL(VT, NewOp.getValueType(), NewOp.getOperand(1),
                             DAG, Subtarget);
    }
  }

  if (X86::isMOVLMask(PermMask.Val)) {
    if (V1IsUndef)
      return V2;
    if (ISD::isBuildVectorAllZeros(V1.Val))
      return getVZextMovL(VT, VT, V2, DAG, Subtarget);
    return Op;
  }

  if (X86::isMOVSHDUPMask(PermMask.Val) ||
      X86::isMOVSLDUPMask(PermMask.Val) ||
      X86::isMOVHLPSMask(PermMask.Val) ||
      X86::isMOVHPMask(PermMask.Val) ||
      X86::isMOVLPMask(PermMask.Val))
    return Op;

  if (ShouldXformToMOVHLPS(PermMask.Val) ||
      ShouldXformToMOVLP(V1.Val, V2.Val, PermMask.Val))
    return CommuteVectorShuffle(Op, V1, V2, PermMask, DAG);

  bool Commuted = false;
  // FIXME: This should also accept a bitcast of a splat?  Be careful, not
  // 1,1,1,1 -> v8i16 though.
  V1IsSplat = isSplatVector(V1.Val);
  V2IsSplat = isSplatVector(V2.Val);
  
  // Canonicalize the splat or undef, if present, to be on the RHS.
  if ((V1IsSplat || V1IsUndef) && !(V2IsSplat || V2IsUndef)) {
    Op = CommuteVectorShuffle(Op, V1, V2, PermMask, DAG);
    std::swap(V1IsSplat, V2IsSplat);
    std::swap(V1IsUndef, V2IsUndef);
    Commuted = true;
  }

  // FIXME: Figure out a cleaner way to do this.
  if (isCommutedMOVL(PermMask.Val, V2IsSplat, V2IsUndef)) {
    if (V2IsUndef) return V1;
    Op = CommuteVectorShuffle(Op, V1, V2, PermMask, DAG);
    if (V2IsSplat) {
      // V2 is a splat, so the mask may be malformed. That is, it may point
      // to any V2 element. The instruction selectior won't like this. Get
      // a corrected mask and commute to form a proper MOVS{S|D}.
      SDOperand NewMask = getMOVLMask(NumElems, DAG);
      if (NewMask.Val != PermMask.Val)
        Op = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2, NewMask);
    }
    return Op;
  }

  if (X86::isUNPCKL_v_undef_Mask(PermMask.Val) ||
      X86::isUNPCKH_v_undef_Mask(PermMask.Val) ||
      X86::isUNPCKLMask(PermMask.Val) ||
      X86::isUNPCKHMask(PermMask.Val))
    return Op;

  if (V2IsSplat) {
    // Normalize mask so all entries that point to V2 points to its first
    // element then try to match unpck{h|l} again. If match, return a
    // new vector_shuffle with the corrected mask.
    SDOperand NewMask = NormalizeMask(PermMask, DAG);
    if (NewMask.Val != PermMask.Val) {
      if (X86::isUNPCKLMask(PermMask.Val, true)) {
        SDOperand NewMask = getUnpacklMask(NumElems, DAG);
        return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2, NewMask);
      } else if (X86::isUNPCKHMask(PermMask.Val, true)) {
        SDOperand NewMask = getUnpackhMask(NumElems, DAG);
        return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2, NewMask);
      }
    }
  }

  // Normalize the node to match x86 shuffle ops if needed
  if (V2.getOpcode() != ISD::UNDEF && isCommutedSHUFP(PermMask.Val))
      Op = CommuteVectorShuffle(Op, V1, V2, PermMask, DAG);

  if (Commuted) {
    // Commute is back and try unpck* again.
    Op = CommuteVectorShuffle(Op, V1, V2, PermMask, DAG);
    if (X86::isUNPCKL_v_undef_Mask(PermMask.Val) ||
        X86::isUNPCKH_v_undef_Mask(PermMask.Val) ||
        X86::isUNPCKLMask(PermMask.Val) ||
        X86::isUNPCKHMask(PermMask.Val))
      return Op;
  }

  // Try PSHUF* first, then SHUFP*.
  // MMX doesn't have PSHUFD but it does have PSHUFW. While it's theoretically
  // possible to shuffle a v2i32 using PSHUFW, that's not yet implemented.
  if (isMMX && NumElems == 4 && X86::isPSHUFDMask(PermMask.Val)) {
    if (V2.getOpcode() != ISD::UNDEF)
      return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1,
                         DAG.getNode(ISD::UNDEF, VT), PermMask);
    return Op;
  }

  if (!isMMX) {
    if (Subtarget->hasSSE2() &&
        (X86::isPSHUFDMask(PermMask.Val) ||
         X86::isPSHUFHWMask(PermMask.Val) ||
         X86::isPSHUFLWMask(PermMask.Val))) {
      MVT::ValueType RVT = VT;
      if (VT == MVT::v4f32) {
        RVT = MVT::v4i32;
        Op = DAG.getNode(ISD::VECTOR_SHUFFLE, RVT,
                         DAG.getNode(ISD::BIT_CONVERT, RVT, V1),
                         DAG.getNode(ISD::UNDEF, RVT), PermMask);
      } else if (V2.getOpcode() != ISD::UNDEF)
        Op = DAG.getNode(ISD::VECTOR_SHUFFLE, RVT, V1,
                         DAG.getNode(ISD::UNDEF, RVT), PermMask);
      if (RVT != VT)
        Op = DAG.getNode(ISD::BIT_CONVERT, VT, Op);
      return Op;
    }

    // Binary or unary shufps.
    if (X86::isSHUFPMask(PermMask.Val) ||
        (V2.getOpcode() == ISD::UNDEF && X86::isPSHUFDMask(PermMask.Val)))
      return Op;
  }

  // Handle v8i16 specifically since SSE can do byte extraction and insertion.
  if (VT == MVT::v8i16) {
    SDOperand NewOp = LowerVECTOR_SHUFFLEv8i16(V1, V2, PermMask, DAG, *this);
    if (NewOp.Val)
      return NewOp;
  }

  // Handle all 4 wide cases with a number of shuffles.
  if (NumElems == 4 && !isMMX) {
    // Don't do this for MMX.
    MVT::ValueType MaskVT = PermMask.getValueType();
    MVT::ValueType MaskEVT = MVT::getVectorElementType(MaskVT);
    SmallVector<std::pair<int, int>, 8> Locs;
    Locs.reserve(NumElems);
    SmallVector<SDOperand, 8> Mask1(NumElems,
                                    DAG.getNode(ISD::UNDEF, MaskEVT));
    SmallVector<SDOperand, 8> Mask2(NumElems,
                                    DAG.getNode(ISD::UNDEF, MaskEVT));
    unsigned NumHi = 0;
    unsigned NumLo = 0;
    // If no more than two elements come from either vector. This can be
    // implemented with two shuffles. First shuffle gather the elements.
    // The second shuffle, which takes the first shuffle as both of its
    // vector operands, put the elements into the right order.
    for (unsigned i = 0; i != NumElems; ++i) {
      SDOperand Elt = PermMask.getOperand(i);
      if (Elt.getOpcode() == ISD::UNDEF) {
        Locs[i] = std::make_pair(-1, -1);
      } else {
        unsigned Val = cast<ConstantSDNode>(Elt)->getValue();
        if (Val < NumElems) {
          Locs[i] = std::make_pair(0, NumLo);
          Mask1[NumLo] = Elt;
          NumLo++;
        } else {
          Locs[i] = std::make_pair(1, NumHi);
          if (2+NumHi < NumElems)
            Mask1[2+NumHi] = Elt;
          NumHi++;
        }
      }
    }
    if (NumLo <= 2 && NumHi <= 2) {
      V1 = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2,
                       DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                   &Mask1[0], Mask1.size()));
      for (unsigned i = 0; i != NumElems; ++i) {
        if (Locs[i].first == -1)
          continue;
        else {
          unsigned Idx = (i < NumElems/2) ? 0 : NumElems;
          Idx += Locs[i].first * (NumElems/2) + Locs[i].second;
          Mask2[i] = DAG.getConstant(Idx, MaskEVT);
        }
      }

      return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V1,
                         DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                     &Mask2[0], Mask2.size()));
    }

    // Break it into (shuffle shuffle_hi, shuffle_lo).
    Locs.clear();
    SmallVector<SDOperand,8> LoMask(NumElems, DAG.getNode(ISD::UNDEF, MaskEVT));
    SmallVector<SDOperand,8> HiMask(NumElems, DAG.getNode(ISD::UNDEF, MaskEVT));
    SmallVector<SDOperand,8> *MaskPtr = &LoMask;
    unsigned MaskIdx = 0;
    unsigned LoIdx = 0;
    unsigned HiIdx = NumElems/2;
    for (unsigned i = 0; i != NumElems; ++i) {
      if (i == NumElems/2) {
        MaskPtr = &HiMask;
        MaskIdx = 1;
        LoIdx = 0;
        HiIdx = NumElems/2;
      }
      SDOperand Elt = PermMask.getOperand(i);
      if (Elt.getOpcode() == ISD::UNDEF) {
        Locs[i] = std::make_pair(-1, -1);
      } else if (cast<ConstantSDNode>(Elt)->getValue() < NumElems) {
        Locs[i] = std::make_pair(MaskIdx, LoIdx);
        (*MaskPtr)[LoIdx] = Elt;
        LoIdx++;
      } else {
        Locs[i] = std::make_pair(MaskIdx, HiIdx);
        (*MaskPtr)[HiIdx] = Elt;
        HiIdx++;
      }
    }

    SDOperand LoShuffle =
      DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2,
                  DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                              &LoMask[0], LoMask.size()));
    SDOperand HiShuffle =
      DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2,
                  DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                              &HiMask[0], HiMask.size()));
    SmallVector<SDOperand, 8> MaskOps;
    for (unsigned i = 0; i != NumElems; ++i) {
      if (Locs[i].first == -1) {
        MaskOps.push_back(DAG.getNode(ISD::UNDEF, MaskEVT));
      } else {
        unsigned Idx = Locs[i].first * NumElems + Locs[i].second;
        MaskOps.push_back(DAG.getConstant(Idx, MaskEVT));
      }
    }
    return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, LoShuffle, HiShuffle,
                       DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                   &MaskOps[0], MaskOps.size()));
  }

  return SDOperand();
}

SDOperand
X86TargetLowering::LowerEXTRACT_VECTOR_ELT_SSE4(SDOperand Op,
                                                SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  if (MVT::getSizeInBits(VT) == 8) {
    SDOperand Extract = DAG.getNode(X86ISD::PEXTRB, MVT::i32,
                                    Op.getOperand(0), Op.getOperand(1));
    SDOperand Assert  = DAG.getNode(ISD::AssertZext, MVT::i32, Extract,
                                    DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, VT, Assert);
  } else if (MVT::getSizeInBits(VT) == 16) {
    SDOperand Extract = DAG.getNode(X86ISD::PEXTRW, MVT::i32,
                                    Op.getOperand(0), Op.getOperand(1));
    SDOperand Assert  = DAG.getNode(ISD::AssertZext, MVT::i32, Extract,
                                    DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, VT, Assert);
  } else if (VT == MVT::f32) {
    // EXTRACTPS outputs to a GPR32 register which will require a movd to copy
    // the result back to FR32 register. It's only worth matching if the
    // result has a single use which is a store or a bitcast to i32.
    if (!Op.hasOneUse())
      return SDOperand();
    SDNode *User = Op.Val->use_begin()->getUser();
    if (User->getOpcode() != ISD::STORE &&
        (User->getOpcode() != ISD::BIT_CONVERT ||
         User->getValueType(0) != MVT::i32))
      return SDOperand();
    SDOperand Extract = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i32,
                    DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32, Op.getOperand(0)),
                                    Op.getOperand(1));
    return DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Extract);
  }
  return SDOperand();
}


SDOperand
X86TargetLowering::LowerEXTRACT_VECTOR_ELT(SDOperand Op, SelectionDAG &DAG) {
  if (!isa<ConstantSDNode>(Op.getOperand(1)))
    return SDOperand();

  if (Subtarget->hasSSE41()) {
    SDOperand Res = LowerEXTRACT_VECTOR_ELT_SSE4(Op, DAG);
    if (Res.Val)
      return Res;
  }

  MVT::ValueType VT = Op.getValueType();
  // TODO: handle v16i8.
  if (MVT::getSizeInBits(VT) == 16) {
    SDOperand Vec = Op.getOperand(0);
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
    if (Idx == 0)
      return DAG.getNode(ISD::TRUNCATE, MVT::i16,
                         DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i32,
                                 DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32, Vec),
                                     Op.getOperand(1)));
    // Transform it so it match pextrw which produces a 32-bit result.
    MVT::ValueType EVT = (MVT::ValueType)(VT+1);
    SDOperand Extract = DAG.getNode(X86ISD::PEXTRW, EVT,
                                    Op.getOperand(0), Op.getOperand(1));
    SDOperand Assert  = DAG.getNode(ISD::AssertZext, EVT, Extract,
                                    DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, VT, Assert);
  } else if (MVT::getSizeInBits(VT) == 32) {
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
    if (Idx == 0)
      return Op;
    // SHUFPS the element to the lowest double word, then movss.
    MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(4);
    SmallVector<SDOperand, 8> IdxVec;
    IdxVec.
      push_back(DAG.getConstant(Idx, MVT::getVectorElementType(MaskVT)));
    IdxVec.
      push_back(DAG.getNode(ISD::UNDEF, MVT::getVectorElementType(MaskVT)));
    IdxVec.
      push_back(DAG.getNode(ISD::UNDEF, MVT::getVectorElementType(MaskVT)));
    IdxVec.
      push_back(DAG.getNode(ISD::UNDEF, MVT::getVectorElementType(MaskVT)));
    SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                 &IdxVec[0], IdxVec.size());
    SDOperand Vec = Op.getOperand(0);
    Vec = DAG.getNode(ISD::VECTOR_SHUFFLE, Vec.getValueType(),
                      Vec, DAG.getNode(ISD::UNDEF, Vec.getValueType()), Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, VT, Vec,
                       DAG.getIntPtrConstant(0));
  } else if (MVT::getSizeInBits(VT) == 64) {
    // FIXME: .td only matches this for <2 x f64>, not <2 x i64> on 32b
    // FIXME: seems like this should be unnecessary if mov{h,l}pd were taught
    //        to match extract_elt for f64.
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
    if (Idx == 0)
      return Op;

    // UNPCKHPD the element to the lowest double word, then movsd.
    // Note if the lower 64 bits of the result of the UNPCKHPD is then stored
    // to a f64mem, the whole operation is folded into a single MOVHPDmr.
    MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(4);
    SmallVector<SDOperand, 8> IdxVec;
    IdxVec.push_back(DAG.getConstant(1, MVT::getVectorElementType(MaskVT)));
    IdxVec.
      push_back(DAG.getNode(ISD::UNDEF, MVT::getVectorElementType(MaskVT)));
    SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                 &IdxVec[0], IdxVec.size());
    SDOperand Vec = Op.getOperand(0);
    Vec = DAG.getNode(ISD::VECTOR_SHUFFLE, Vec.getValueType(),
                      Vec, DAG.getNode(ISD::UNDEF, Vec.getValueType()), Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, VT, Vec,
                       DAG.getIntPtrConstant(0));
  }

  return SDOperand();
}

SDOperand
X86TargetLowering::LowerINSERT_VECTOR_ELT_SSE4(SDOperand Op, SelectionDAG &DAG){
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType EVT = MVT::getVectorElementType(VT);

  SDOperand N0 = Op.getOperand(0);
  SDOperand N1 = Op.getOperand(1);
  SDOperand N2 = Op.getOperand(2);

  if ((MVT::getSizeInBits(EVT) == 8) || (MVT::getSizeInBits(EVT) == 16)) {
    unsigned Opc = (MVT::getSizeInBits(EVT) == 8) ? X86ISD::PINSRB
                                                  : X86ISD::PINSRW;
    // Transform it so it match pinsr{b,w} which expects a GR32 as its second
    // argument.
    if (N1.getValueType() != MVT::i32)
      N1 = DAG.getNode(ISD::ANY_EXTEND, MVT::i32, N1);
    if (N2.getValueType() != MVT::i32)
      N2 = DAG.getIntPtrConstant(cast<ConstantSDNode>(N2)->getValue());
    return DAG.getNode(Opc, VT, N0, N1, N2);
  } else if (EVT == MVT::f32) {
    // Bits [7:6] of the constant are the source select.  This will always be
    //  zero here.  The DAG Combiner may combine an extract_elt index into these
    //  bits.  For example (insert (extract, 3), 2) could be matched by putting
    //  the '3' into bits [7:6] of X86ISD::INSERTPS.
    // Bits [5:4] of the constant are the destination select.  This is the 
    //  value of the incoming immediate.
    // Bits [3:0] of the constant are the zero mask.  The DAG Combiner may 
    //   combine either bitwise AND or insert of float 0.0 to set these bits.
    N2 = DAG.getIntPtrConstant(cast<ConstantSDNode>(N2)->getValue() << 4);
    return DAG.getNode(X86ISD::INSERTPS, VT, N0, N1, N2);
  }
  return SDOperand();
}

SDOperand
X86TargetLowering::LowerINSERT_VECTOR_ELT(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType EVT = MVT::getVectorElementType(VT);

  if (Subtarget->hasSSE41())
    return LowerINSERT_VECTOR_ELT_SSE4(Op, DAG);

  if (EVT == MVT::i8)
    return SDOperand();

  SDOperand N0 = Op.getOperand(0);
  SDOperand N1 = Op.getOperand(1);
  SDOperand N2 = Op.getOperand(2);

  if (MVT::getSizeInBits(EVT) == 16) {
    // Transform it so it match pinsrw which expects a 16-bit value in a GR32
    // as its second argument.
    if (N1.getValueType() != MVT::i32)
      N1 = DAG.getNode(ISD::ANY_EXTEND, MVT::i32, N1);
    if (N2.getValueType() != MVT::i32)
      N2 = DAG.getIntPtrConstant(cast<ConstantSDNode>(N2)->getValue());
    return DAG.getNode(X86ISD::PINSRW, VT, N0, N1, N2);
  }
  return SDOperand();
}

SDOperand
X86TargetLowering::LowerSCALAR_TO_VECTOR(SDOperand Op, SelectionDAG &DAG) {
  SDOperand AnyExt = DAG.getNode(ISD::ANY_EXTEND, MVT::i32, Op.getOperand(0));
  MVT::ValueType VT = MVT::v2i32;
  switch (Op.getValueType()) {
  default: break;
  case MVT::v16i8:
  case MVT::v8i16:
    VT = MVT::v4i32;
    break;
  }
  return DAG.getNode(ISD::BIT_CONVERT, Op.getValueType(),
                     DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, AnyExt));
}

// ConstantPool, JumpTable, GlobalAddress, and ExternalSymbol are lowered as
// their target countpart wrapped in the X86ISD::Wrapper node. Suppose N is
// one of the above mentioned nodes. It has to be wrapped because otherwise
// Select(N) returns N. So the raw TargetGlobalAddress nodes, etc. can only
// be used to form addressing mode. These wrapped nodes will be selected
// into MOV32ri.
SDOperand
X86TargetLowering::LowerConstantPool(SDOperand Op, SelectionDAG &DAG) {
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  SDOperand Result = DAG.getTargetConstantPool(CP->getConstVal(),
                                               getPointerTy(),
                                               CP->getAlignment());
  Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(), Result);
  // With PIC, the address is actually $g + Offset.
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
      !Subtarget->isPICStyleRIPRel()) {
    Result = DAG.getNode(ISD::ADD, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()),
                         Result);
  }

  return Result;
}

SDOperand
X86TargetLowering::LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG) {
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDOperand Result = DAG.getTargetGlobalAddress(GV, getPointerTy());
  // If it's a debug information descriptor, don't mess with it.
  if (DAG.isVerifiedDebugInfoDesc(Op))
    return Result;
  Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(), Result);
  // With PIC, the address is actually $g + Offset.
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
      !Subtarget->isPICStyleRIPRel()) {
    Result = DAG.getNode(ISD::ADD, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()),
                         Result);
  }
  
  // For Darwin & Mingw32, external and weak symbols are indirect, so we want to
  // load the value at address GV, not the value of GV itself. This means that
  // the GlobalAddress must be in the base or index register of the address, not
  // the GV offset field. Platform check is inside GVRequiresExtraLoad() call
  // The same applies for external symbols during PIC codegen
  if (Subtarget->GVRequiresExtraLoad(GV, getTargetMachine(), false))
    Result = DAG.getLoad(getPointerTy(), DAG.getEntryNode(), Result,
                         PseudoSourceValue::getGOT(), 0);

  return Result;
}

// Lower ISD::GlobalTLSAddress using the "general dynamic" model, 32 bit
static SDOperand
LowerToTLSGeneralDynamicModel32(GlobalAddressSDNode *GA, SelectionDAG &DAG,
                                const MVT::ValueType PtrVT) {
  SDOperand InFlag;
  SDOperand Chain = DAG.getCopyToReg(DAG.getEntryNode(), X86::EBX,
                                     DAG.getNode(X86ISD::GlobalBaseReg,
                                                 PtrVT), InFlag);
  InFlag = Chain.getValue(1);

  // emit leal symbol@TLSGD(,%ebx,1), %eax
  SDVTList NodeTys = DAG.getVTList(PtrVT, MVT::Other, MVT::Flag);
  SDOperand TGA = DAG.getTargetGlobalAddress(GA->getGlobal(),
                                             GA->getValueType(0),
                                             GA->getOffset());
  SDOperand Ops[] = { Chain,  TGA, InFlag };
  SDOperand Result = DAG.getNode(X86ISD::TLSADDR, NodeTys, Ops, 3);
  InFlag = Result.getValue(2);
  Chain = Result.getValue(1);

  // call ___tls_get_addr. This function receives its argument in
  // the register EAX.
  Chain = DAG.getCopyToReg(Chain, X86::EAX, Result, InFlag);
  InFlag = Chain.getValue(1);

  NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
  SDOperand Ops1[] = { Chain,
                      DAG.getTargetExternalSymbol("___tls_get_addr",
                                                  PtrVT),
                      DAG.getRegister(X86::EAX, PtrVT),
                      DAG.getRegister(X86::EBX, PtrVT),
                      InFlag };
  Chain = DAG.getNode(X86ISD::CALL, NodeTys, Ops1, 5);
  InFlag = Chain.getValue(1);

  return DAG.getCopyFromReg(Chain, X86::EAX, PtrVT, InFlag);
}

// Lower ISD::GlobalTLSAddress using the "general dynamic" model, 64 bit
static SDOperand
LowerToTLSGeneralDynamicModel64(GlobalAddressSDNode *GA, SelectionDAG &DAG,
                                const MVT::ValueType PtrVT) {
  SDOperand InFlag, Chain;

  // emit leaq symbol@TLSGD(%rip), %rdi
  SDVTList NodeTys = DAG.getVTList(PtrVT, MVT::Other, MVT::Flag);
  SDOperand TGA = DAG.getTargetGlobalAddress(GA->getGlobal(),
                                             GA->getValueType(0),
                                             GA->getOffset());
  SDOperand Ops[]  = { DAG.getEntryNode(), TGA};
  SDOperand Result = DAG.getNode(X86ISD::TLSADDR, NodeTys, Ops, 2);
  Chain  = Result.getValue(1);
  InFlag = Result.getValue(2);

  // call ___tls_get_addr. This function receives its argument in
  // the register RDI.
  Chain = DAG.getCopyToReg(Chain, X86::RDI, Result, InFlag);
  InFlag = Chain.getValue(1);

  NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
  SDOperand Ops1[] = { Chain,
                      DAG.getTargetExternalSymbol("___tls_get_addr",
                                                  PtrVT),
                      DAG.getRegister(X86::RDI, PtrVT),
                      InFlag };
  Chain = DAG.getNode(X86ISD::CALL, NodeTys, Ops1, 4);
  InFlag = Chain.getValue(1);

  return DAG.getCopyFromReg(Chain, X86::RAX, PtrVT, InFlag);
}

// Lower ISD::GlobalTLSAddress using the "initial exec" (for no-pic) or
// "local exec" model.
static SDOperand
LowerToTLSExecModel(GlobalAddressSDNode *GA, SelectionDAG &DAG,
                         const MVT::ValueType PtrVT) {
  // Get the Thread Pointer
  SDOperand ThreadPointer = DAG.getNode(X86ISD::THREAD_POINTER, PtrVT);
  // emit "addl x@ntpoff,%eax" (local exec) or "addl x@indntpoff,%eax" (initial
  // exec)
  SDOperand TGA = DAG.getTargetGlobalAddress(GA->getGlobal(),
                                             GA->getValueType(0),
                                             GA->getOffset());
  SDOperand Offset = DAG.getNode(X86ISD::Wrapper, PtrVT, TGA);

  if (GA->getGlobal()->isDeclaration()) // initial exec TLS model
    Offset = DAG.getLoad(PtrVT, DAG.getEntryNode(), Offset,
                         PseudoSourceValue::getGOT(), 0);

  // The address of the thread local variable is the add of the thread
  // pointer with the offset of the variable.
  return DAG.getNode(ISD::ADD, PtrVT, ThreadPointer, Offset);
}

SDOperand
X86TargetLowering::LowerGlobalTLSAddress(SDOperand Op, SelectionDAG &DAG) {
  // TODO: implement the "local dynamic" model
  // TODO: implement the "initial exec"model for pic executables
  assert(Subtarget->isTargetELF() &&
         "TLS not implemented for non-ELF targets");
  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);
  // If the relocation model is PIC, use the "General Dynamic" TLS Model,
  // otherwise use the "Local Exec"TLS Model
  if (Subtarget->is64Bit()) {
    return LowerToTLSGeneralDynamicModel64(GA, DAG, getPointerTy());
  } else {
    if (getTargetMachine().getRelocationModel() == Reloc::PIC_)
      return LowerToTLSGeneralDynamicModel32(GA, DAG, getPointerTy());
    else
      return LowerToTLSExecModel(GA, DAG, getPointerTy());
  }
}

SDOperand
X86TargetLowering::LowerExternalSymbol(SDOperand Op, SelectionDAG &DAG) {
  const char *Sym = cast<ExternalSymbolSDNode>(Op)->getSymbol();
  SDOperand Result = DAG.getTargetExternalSymbol(Sym, getPointerTy());
  Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(), Result);
  // With PIC, the address is actually $g + Offset.
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
      !Subtarget->isPICStyleRIPRel()) {
    Result = DAG.getNode(ISD::ADD, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()),
                         Result);
  }

  return Result;
}

SDOperand X86TargetLowering::LowerJumpTable(SDOperand Op, SelectionDAG &DAG) {
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDOperand Result = DAG.getTargetJumpTable(JT->getIndex(), getPointerTy());
  Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(), Result);
  // With PIC, the address is actually $g + Offset.
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_ &&
      !Subtarget->isPICStyleRIPRel()) {
    Result = DAG.getNode(ISD::ADD, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()),
                         Result);
  }

  return Result;
}

/// LowerShift - Lower SRA_PARTS and friends, which return two i32 values and
/// take a 2 x i32 value to shift plus a shift amount. 
SDOperand X86TargetLowering::LowerShift(SDOperand Op, SelectionDAG &DAG) {
  assert(Op.getNumOperands() == 3 && "Not a double-shift!");
  MVT::ValueType VT = Op.getValueType();
  unsigned VTBits = MVT::getSizeInBits(VT);
  bool isSRA = Op.getOpcode() == ISD::SRA_PARTS;
  SDOperand ShOpLo = Op.getOperand(0);
  SDOperand ShOpHi = Op.getOperand(1);
  SDOperand ShAmt  = Op.getOperand(2);
  SDOperand Tmp1 = isSRA ?
    DAG.getNode(ISD::SRA, VT, ShOpHi, DAG.getConstant(VTBits - 1, MVT::i8)) :
    DAG.getConstant(0, VT);

  SDOperand Tmp2, Tmp3;
  if (Op.getOpcode() == ISD::SHL_PARTS) {
    Tmp2 = DAG.getNode(X86ISD::SHLD, VT, ShOpHi, ShOpLo, ShAmt);
    Tmp3 = DAG.getNode(ISD::SHL, VT, ShOpLo, ShAmt);
  } else {
    Tmp2 = DAG.getNode(X86ISD::SHRD, VT, ShOpLo, ShOpHi, ShAmt);
    Tmp3 = DAG.getNode(isSRA ? ISD::SRA : ISD::SRL, VT, ShOpHi, ShAmt);
  }

  const MVT::ValueType *VTs = DAG.getNodeValueTypes(MVT::Other, MVT::Flag);
  SDOperand AndNode = DAG.getNode(ISD::AND, MVT::i8, ShAmt,
                                  DAG.getConstant(VTBits, MVT::i8));
  SDOperand Cond = DAG.getNode(X86ISD::CMP, VT,
                               AndNode, DAG.getConstant(0, MVT::i8));

  SDOperand Hi, Lo;
  SDOperand CC = DAG.getConstant(X86::COND_NE, MVT::i8);
  VTs = DAG.getNodeValueTypes(VT, MVT::Flag);
  SmallVector<SDOperand, 4> Ops;
  if (Op.getOpcode() == ISD::SHL_PARTS) {
    Ops.push_back(Tmp2);
    Ops.push_back(Tmp3);
    Ops.push_back(CC);
    Ops.push_back(Cond);
    Hi = DAG.getNode(X86ISD::CMOV, VT, &Ops[0], Ops.size());

    Ops.clear();
    Ops.push_back(Tmp3);
    Ops.push_back(Tmp1);
    Ops.push_back(CC);
    Ops.push_back(Cond);
    Lo = DAG.getNode(X86ISD::CMOV, VT, &Ops[0], Ops.size());
  } else {
    Ops.push_back(Tmp2);
    Ops.push_back(Tmp3);
    Ops.push_back(CC);
    Ops.push_back(Cond);
    Lo = DAG.getNode(X86ISD::CMOV, VT, &Ops[0], Ops.size());

    Ops.clear();
    Ops.push_back(Tmp3);
    Ops.push_back(Tmp1);
    Ops.push_back(CC);
    Ops.push_back(Cond);
    Hi = DAG.getNode(X86ISD::CMOV, VT, &Ops[0], Ops.size());
  }

  VTs = DAG.getNodeValueTypes(VT, VT);
  Ops.clear();
  Ops.push_back(Lo);
  Ops.push_back(Hi);
  return DAG.getNode(ISD::MERGE_VALUES, VTs, 2, &Ops[0], Ops.size());
}

SDOperand X86TargetLowering::LowerSINT_TO_FP(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType SrcVT = Op.getOperand(0).getValueType();
  assert(SrcVT <= MVT::i64 && SrcVT >= MVT::i16 &&
         "Unknown SINT_TO_FP to lower!");
  
  // These are really Legal; caller falls through into that case.
  if (SrcVT == MVT::i32 && isScalarFPTypeInSSEReg(Op.getValueType()))
    return SDOperand();
  if (SrcVT == MVT::i64 && Op.getValueType() != MVT::f80 && 
      Subtarget->is64Bit())
    return SDOperand();
  
  unsigned Size = MVT::getSizeInBits(SrcVT)/8;
  MachineFunction &MF = DAG.getMachineFunction();
  int SSFI = MF.getFrameInfo()->CreateStackObject(Size, Size);
  SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
  SDOperand Chain = DAG.getStore(DAG.getEntryNode(), Op.getOperand(0),
                                 StackSlot,
                                 PseudoSourceValue::getFixedStack(),
                                 SSFI);

  // Build the FILD
  SDVTList Tys;
  bool useSSE = isScalarFPTypeInSSEReg(Op.getValueType());
  if (useSSE)
    Tys = DAG.getVTList(MVT::f64, MVT::Other, MVT::Flag);
  else
    Tys = DAG.getVTList(Op.getValueType(), MVT::Other);
  SmallVector<SDOperand, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(StackSlot);
  Ops.push_back(DAG.getValueType(SrcVT));
  SDOperand Result = DAG.getNode(useSSE ? X86ISD::FILD_FLAG : X86ISD::FILD,
                                 Tys, &Ops[0], Ops.size());

  if (useSSE) {
    Chain = Result.getValue(1);
    SDOperand InFlag = Result.getValue(2);

    // FIXME: Currently the FST is flagged to the FILD_FLAG. This
    // shouldn't be necessary except that RFP cannot be live across
    // multiple blocks. When stackifier is fixed, they can be uncoupled.
    MachineFunction &MF = DAG.getMachineFunction();
    int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
    SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
    Tys = DAG.getVTList(MVT::Other);
    SmallVector<SDOperand, 8> Ops;
    Ops.push_back(Chain);
    Ops.push_back(Result);
    Ops.push_back(StackSlot);
    Ops.push_back(DAG.getValueType(Op.getValueType()));
    Ops.push_back(InFlag);
    Chain = DAG.getNode(X86ISD::FST, Tys, &Ops[0], Ops.size());
    Result = DAG.getLoad(Op.getValueType(), Chain, StackSlot,
                         PseudoSourceValue::getFixedStack(), SSFI);
  }

  return Result;
}

std::pair<SDOperand,SDOperand> X86TargetLowering::
FP_TO_SINTHelper(SDOperand Op, SelectionDAG &DAG) {
  assert(Op.getValueType() <= MVT::i64 && Op.getValueType() >= MVT::i16 &&
         "Unknown FP_TO_SINT to lower!");

  // These are really Legal.
  if (Op.getValueType() == MVT::i32 && 
      isScalarFPTypeInSSEReg(Op.getOperand(0).getValueType()))
    return std::make_pair(SDOperand(), SDOperand());
  if (Subtarget->is64Bit() &&
      Op.getValueType() == MVT::i64 &&
      Op.getOperand(0).getValueType() != MVT::f80)
    return std::make_pair(SDOperand(), SDOperand());

  // We lower FP->sint64 into FISTP64, followed by a load, all to a temporary
  // stack slot.
  MachineFunction &MF = DAG.getMachineFunction();
  unsigned MemSize = MVT::getSizeInBits(Op.getValueType())/8;
  int SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize);
  SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
  unsigned Opc;
  switch (Op.getValueType()) {
  default: assert(0 && "Invalid FP_TO_SINT to lower!");
  case MVT::i16: Opc = X86ISD::FP_TO_INT16_IN_MEM; break;
  case MVT::i32: Opc = X86ISD::FP_TO_INT32_IN_MEM; break;
  case MVT::i64: Opc = X86ISD::FP_TO_INT64_IN_MEM; break;
  }

  SDOperand Chain = DAG.getEntryNode();
  SDOperand Value = Op.getOperand(0);
  if (isScalarFPTypeInSSEReg(Op.getOperand(0).getValueType())) {
    assert(Op.getValueType() == MVT::i64 && "Invalid FP_TO_SINT to lower!");
    Chain = DAG.getStore(Chain, Value, StackSlot,
                         PseudoSourceValue::getFixedStack(), SSFI);
    SDVTList Tys = DAG.getVTList(Op.getOperand(0).getValueType(), MVT::Other);
    SDOperand Ops[] = {
      Chain, StackSlot, DAG.getValueType(Op.getOperand(0).getValueType())
    };
    Value = DAG.getNode(X86ISD::FLD, Tys, Ops, 3);
    Chain = Value.getValue(1);
    SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize);
    StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
  }

  // Build the FP_TO_INT*_IN_MEM
  SDOperand Ops[] = { Chain, Value, StackSlot };
  SDOperand FIST = DAG.getNode(Opc, MVT::Other, Ops, 3);

  return std::make_pair(FIST, StackSlot);
}

SDOperand X86TargetLowering::LowerFP_TO_SINT(SDOperand Op, SelectionDAG &DAG) {
  std::pair<SDOperand,SDOperand> Vals = FP_TO_SINTHelper(Op, DAG);
  SDOperand FIST = Vals.first, StackSlot = Vals.second;
  if (FIST.Val == 0) return SDOperand();
  
  // Load the result.
  return DAG.getLoad(Op.getValueType(), FIST, StackSlot, NULL, 0);
}

SDNode *X86TargetLowering::ExpandFP_TO_SINT(SDNode *N, SelectionDAG &DAG) {
  std::pair<SDOperand,SDOperand> Vals = FP_TO_SINTHelper(SDOperand(N, 0), DAG);
  SDOperand FIST = Vals.first, StackSlot = Vals.second;
  if (FIST.Val == 0) return 0;
  
  // Return an i64 load from the stack slot.
  SDOperand Res = DAG.getLoad(MVT::i64, FIST, StackSlot, NULL, 0);

  // Use a MERGE_VALUES node to drop the chain result value.
  return DAG.getNode(ISD::MERGE_VALUES, MVT::i64, Res).Val;
}  

SDOperand X86TargetLowering::LowerFABS(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType EltVT = VT;
  if (MVT::isVector(VT))
    EltVT = MVT::getVectorElementType(VT);
  std::vector<Constant*> CV;
  if (EltVT == MVT::f64) {
    Constant *C = ConstantFP::get(APFloat(APInt(64, ~(1ULL << 63))));
    CV.push_back(C);
    CV.push_back(C);
  } else {
    Constant *C = ConstantFP::get(APFloat(APInt(32, ~(1U << 31))));
    CV.push_back(C);
    CV.push_back(C);
    CV.push_back(C);
    CV.push_back(C);
  }
  Constant *C = ConstantVector::get(CV);
  SDOperand CPIdx = DAG.getConstantPool(C, getPointerTy(), 4);
  SDOperand Mask = DAG.getLoad(VT, DAG.getEntryNode(), CPIdx,
                               PseudoSourceValue::getConstantPool(), 0,
                               false, 16);
  return DAG.getNode(X86ISD::FAND, VT, Op.getOperand(0), Mask);
}

SDOperand X86TargetLowering::LowerFNEG(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType EltVT = VT;
  unsigned EltNum = 1;
  if (MVT::isVector(VT)) {
    EltVT = MVT::getVectorElementType(VT);
    EltNum = MVT::getVectorNumElements(VT);
  }
  std::vector<Constant*> CV;
  if (EltVT == MVT::f64) {
    Constant *C = ConstantFP::get(APFloat(APInt(64, 1ULL << 63)));
    CV.push_back(C);
    CV.push_back(C);
  } else {
    Constant *C = ConstantFP::get(APFloat(APInt(32, 1U << 31)));
    CV.push_back(C);
    CV.push_back(C);
    CV.push_back(C);
    CV.push_back(C);
  }
  Constant *C = ConstantVector::get(CV);
  SDOperand CPIdx = DAG.getConstantPool(C, getPointerTy(), 4);
  SDOperand Mask = DAG.getLoad(VT, DAG.getEntryNode(), CPIdx,
                               PseudoSourceValue::getConstantPool(), 0,
                               false, 16);
  if (MVT::isVector(VT)) {
    return DAG.getNode(ISD::BIT_CONVERT, VT,
                       DAG.getNode(ISD::XOR, MVT::v2i64,
                    DAG.getNode(ISD::BIT_CONVERT, MVT::v2i64, Op.getOperand(0)),
                    DAG.getNode(ISD::BIT_CONVERT, MVT::v2i64, Mask)));
  } else {
    return DAG.getNode(X86ISD::FXOR, VT, Op.getOperand(0), Mask);
  }
}

SDOperand X86TargetLowering::LowerFCOPYSIGN(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Op0 = Op.getOperand(0);
  SDOperand Op1 = Op.getOperand(1);
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType SrcVT = Op1.getValueType();

  // If second operand is smaller, extend it first.
  if (MVT::getSizeInBits(SrcVT) < MVT::getSizeInBits(VT)) {
    Op1 = DAG.getNode(ISD::FP_EXTEND, VT, Op1);
    SrcVT = VT;
  }
  // And if it is bigger, shrink it first.
  if (MVT::getSizeInBits(SrcVT) > MVT::getSizeInBits(VT)) {
    Op1 = DAG.getNode(ISD::FP_ROUND, VT, Op1, DAG.getIntPtrConstant(1));
    SrcVT = VT;
  }

  // At this point the operands and the result should have the same
  // type, and that won't be f80 since that is not custom lowered.

  // First get the sign bit of second operand.
  std::vector<Constant*> CV;
  if (SrcVT == MVT::f64) {
    CV.push_back(ConstantFP::get(APFloat(APInt(64, 1ULL << 63))));
    CV.push_back(ConstantFP::get(APFloat(APInt(64, 0))));
  } else {
    CV.push_back(ConstantFP::get(APFloat(APInt(32, 1U << 31))));
    CV.push_back(ConstantFP::get(APFloat(APInt(32, 0))));
    CV.push_back(ConstantFP::get(APFloat(APInt(32, 0))));
    CV.push_back(ConstantFP::get(APFloat(APInt(32, 0))));
  }
  Constant *C = ConstantVector::get(CV);
  SDOperand CPIdx = DAG.getConstantPool(C, getPointerTy(), 4);
  SDOperand Mask1 = DAG.getLoad(SrcVT, DAG.getEntryNode(), CPIdx,
                                PseudoSourceValue::getConstantPool(), 0,
                                false, 16);
  SDOperand SignBit = DAG.getNode(X86ISD::FAND, SrcVT, Op1, Mask1);

  // Shift sign bit right or left if the two operands have different types.
  if (MVT::getSizeInBits(SrcVT) > MVT::getSizeInBits(VT)) {
    // Op0 is MVT::f32, Op1 is MVT::f64.
    SignBit = DAG.getNode(ISD::SCALAR_TO_VECTOR, MVT::v2f64, SignBit);
    SignBit = DAG.getNode(X86ISD::FSRL, MVT::v2f64, SignBit,
                          DAG.getConstant(32, MVT::i32));
    SignBit = DAG.getNode(ISD::BIT_CONVERT, MVT::v4f32, SignBit);
    SignBit = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::f32, SignBit,
                          DAG.getIntPtrConstant(0));
  }

  // Clear first operand sign bit.
  CV.clear();
  if (VT == MVT::f64) {
    CV.push_back(ConstantFP::get(APFloat(APInt(64, ~(1ULL << 63)))));
    CV.push_back(ConstantFP::get(APFloat(APInt(64, 0))));
  } else {
    CV.push_back(ConstantFP::get(APFloat(APInt(32, ~(1U << 31)))));
    CV.push_back(ConstantFP::get(APFloat(APInt(32, 0))));
    CV.push_back(ConstantFP::get(APFloat(APInt(32, 0))));
    CV.push_back(ConstantFP::get(APFloat(APInt(32, 0))));
  }
  C = ConstantVector::get(CV);
  CPIdx = DAG.getConstantPool(C, getPointerTy(), 4);
  SDOperand Mask2 = DAG.getLoad(VT, DAG.getEntryNode(), CPIdx,
                                PseudoSourceValue::getConstantPool(), 0,
                                false, 16);
  SDOperand Val = DAG.getNode(X86ISD::FAND, VT, Op0, Mask2);

  // Or the value with the sign bit.
  return DAG.getNode(X86ISD::FOR, VT, Val, SignBit);
}

SDOperand X86TargetLowering::LowerSETCC(SDOperand Op, SelectionDAG &DAG) {
  assert(Op.getValueType() == MVT::i8 && "SetCC type must be 8-bit integer");
  SDOperand Cond;
  SDOperand Op0 = Op.getOperand(0);
  SDOperand Op1 = Op.getOperand(1);
  SDOperand CC = Op.getOperand(2);
  ISD::CondCode SetCCOpcode = cast<CondCodeSDNode>(CC)->get();
  bool isFP = MVT::isFloatingPoint(Op.getOperand(1).getValueType());
  unsigned X86CC;

  if (translateX86CC(cast<CondCodeSDNode>(CC)->get(), isFP, X86CC,
                     Op0, Op1, DAG)) {
    Cond = DAG.getNode(X86ISD::CMP, MVT::i32, Op0, Op1);
    return DAG.getNode(X86ISD::SETCC, MVT::i8,
                       DAG.getConstant(X86CC, MVT::i8), Cond);
  }

  assert(isFP && "Illegal integer SetCC!");

  Cond = DAG.getNode(X86ISD::CMP, MVT::i32, Op0, Op1);
  switch (SetCCOpcode) {
  default: assert(false && "Illegal floating point SetCC!");
  case ISD::SETOEQ: {  // !PF & ZF
    SDOperand Tmp1 = DAG.getNode(X86ISD::SETCC, MVT::i8,
                                 DAG.getConstant(X86::COND_NP, MVT::i8), Cond);
    SDOperand Tmp2 = DAG.getNode(X86ISD::SETCC, MVT::i8,
                                 DAG.getConstant(X86::COND_E, MVT::i8), Cond);
    return DAG.getNode(ISD::AND, MVT::i8, Tmp1, Tmp2);
  }
  case ISD::SETUNE: {  // PF | !ZF
    SDOperand Tmp1 = DAG.getNode(X86ISD::SETCC, MVT::i8,
                                 DAG.getConstant(X86::COND_P, MVT::i8), Cond);
    SDOperand Tmp2 = DAG.getNode(X86ISD::SETCC, MVT::i8,
                                 DAG.getConstant(X86::COND_NE, MVT::i8), Cond);
    return DAG.getNode(ISD::OR, MVT::i8, Tmp1, Tmp2);
  }
  }
}


SDOperand X86TargetLowering::LowerSELECT(SDOperand Op, SelectionDAG &DAG) {
  bool addTest = true;
  SDOperand Cond  = Op.getOperand(0);
  SDOperand CC;

  if (Cond.getOpcode() == ISD::SETCC)
    Cond = LowerSETCC(Cond, DAG);

  // If condition flag is set by a X86ISD::CMP, then use it as the condition
  // setting operand in place of the X86ISD::SETCC.
  if (Cond.getOpcode() == X86ISD::SETCC) {
    CC = Cond.getOperand(0);

    SDOperand Cmp = Cond.getOperand(1);
    unsigned Opc = Cmp.getOpcode();
    MVT::ValueType VT = Op.getValueType();
    
    bool IllegalFPCMov = false;
    if (MVT::isFloatingPoint(VT) && !MVT::isVector(VT) &&
        !isScalarFPTypeInSSEReg(VT))  // FPStack?
      IllegalFPCMov = !hasFPCMov(cast<ConstantSDNode>(CC)->getSignExtended());
    
    if ((Opc == X86ISD::CMP ||
         Opc == X86ISD::COMI ||
         Opc == X86ISD::UCOMI) && !IllegalFPCMov) {
      Cond = Cmp;
      addTest = false;
    }
  }

  if (addTest) {
    CC = DAG.getConstant(X86::COND_NE, MVT::i8);
    Cond= DAG.getNode(X86ISD::CMP, MVT::i32, Cond, DAG.getConstant(0, MVT::i8));
  }

  const MVT::ValueType *VTs = DAG.getNodeValueTypes(Op.getValueType(),
                                                    MVT::Flag);
  SmallVector<SDOperand, 4> Ops;
  // X86ISD::CMOV means set the result (which is operand 1) to the RHS if
  // condition is true.
  Ops.push_back(Op.getOperand(2));
  Ops.push_back(Op.getOperand(1));
  Ops.push_back(CC);
  Ops.push_back(Cond);
  return DAG.getNode(X86ISD::CMOV, VTs, 2, &Ops[0], Ops.size());
}

SDOperand X86TargetLowering::LowerBRCOND(SDOperand Op, SelectionDAG &DAG) {
  bool addTest = true;
  SDOperand Chain = Op.getOperand(0);
  SDOperand Cond  = Op.getOperand(1);
  SDOperand Dest  = Op.getOperand(2);
  SDOperand CC;

  if (Cond.getOpcode() == ISD::SETCC)
    Cond = LowerSETCC(Cond, DAG);

  // If condition flag is set by a X86ISD::CMP, then use it as the condition
  // setting operand in place of the X86ISD::SETCC.
  if (Cond.getOpcode() == X86ISD::SETCC) {
    CC = Cond.getOperand(0);

    SDOperand Cmp = Cond.getOperand(1);
    unsigned Opc = Cmp.getOpcode();
    if (Opc == X86ISD::CMP ||
        Opc == X86ISD::COMI ||
        Opc == X86ISD::UCOMI) {
      Cond = Cmp;
      addTest = false;
    }
  }

  if (addTest) {
    CC = DAG.getConstant(X86::COND_NE, MVT::i8);
    Cond= DAG.getNode(X86ISD::CMP, MVT::i32, Cond, DAG.getConstant(0, MVT::i8));
  }
  return DAG.getNode(X86ISD::BRCOND, Op.getValueType(),
                     Chain, Op.getOperand(2), CC, Cond);
}


// Lower dynamic stack allocation to _alloca call for Cygwin/Mingw targets.
// Calls to _alloca is needed to probe the stack when allocating more than 4k
// bytes in one go. Touching the stack at 4K increments is necessary to ensure
// that the guard pages used by the OS virtual memory manager are allocated in
// correct sequence.
SDOperand
X86TargetLowering::LowerDYNAMIC_STACKALLOC(SDOperand Op,
                                           SelectionDAG &DAG) {
  assert(Subtarget->isTargetCygMing() &&
         "This should be used only on Cygwin/Mingw targets");
  
  // Get the inputs.
  SDOperand Chain = Op.getOperand(0);
  SDOperand Size  = Op.getOperand(1);
  // FIXME: Ensure alignment here

  SDOperand Flag;
  
  MVT::ValueType IntPtr = getPointerTy();
  MVT::ValueType SPTy = Subtarget->is64Bit() ? MVT::i64 : MVT::i32;

  Chain = DAG.getCopyToReg(Chain, X86::EAX, Size, Flag);
  Flag = Chain.getValue(1);

  SDVTList  NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
  SDOperand Ops[] = { Chain,
                      DAG.getTargetExternalSymbol("_alloca", IntPtr),
                      DAG.getRegister(X86::EAX, IntPtr),
                      Flag };
  Chain = DAG.getNode(X86ISD::CALL, NodeTys, Ops, 4);
  Flag = Chain.getValue(1);

  Chain = DAG.getCopyFromReg(Chain, X86StackPtr, SPTy).getValue(1);
  
  std::vector<MVT::ValueType> Tys;
  Tys.push_back(SPTy);
  Tys.push_back(MVT::Other);
  SDOperand Ops1[2] = { Chain.getValue(0), Chain };
  return DAG.getNode(ISD::MERGE_VALUES, Tys, Ops1, 2);
}

SDOperand
X86TargetLowering::EmitTargetCodeForMemset(SelectionDAG &DAG,
                                           SDOperand Chain,
                                           SDOperand Dst, SDOperand Src,
                                           SDOperand Size, unsigned Align,
                                        const Value *DstSV, uint64_t DstSVOff) {
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);

  /// If not DWORD aligned or size is more than the threshold, call the library.
  /// The libc version is likely to be faster for these cases. It can use the
  /// address value and run time information about the CPU.
  if ((Align & 3) == 0 ||
      !ConstantSize ||
      ConstantSize->getValue() > getSubtarget()->getMaxInlineSizeThreshold()) {
    SDOperand InFlag(0, 0);

    // Check to see if there is a specialized entry-point for memory zeroing.
    ConstantSDNode *V = dyn_cast<ConstantSDNode>(Src);
    if (const char *bzeroEntry = 
          V && V->isNullValue() ? Subtarget->getBZeroEntry() : 0) {
      MVT::ValueType IntPtr = getPointerTy();
      const Type *IntPtrTy = getTargetData()->getIntPtrType();
      TargetLowering::ArgListTy Args; 
      TargetLowering::ArgListEntry Entry;
      Entry.Node = Dst;
      Entry.Ty = IntPtrTy;
      Args.push_back(Entry);
      Entry.Node = Size;
      Args.push_back(Entry);
      std::pair<SDOperand,SDOperand> CallResult =
        LowerCallTo(Chain, Type::VoidTy, false, false, false, CallingConv::C,
                    false, DAG.getExternalSymbol(bzeroEntry, IntPtr),
                    Args, DAG);
      return CallResult.second;
    }

    // Otherwise have the target-independent code call memset.
    return SDOperand();
  }

  uint64_t SizeVal = ConstantSize->getValue();
  SDOperand InFlag(0, 0);
  MVT::ValueType AVT;
  SDOperand Count;
  ConstantSDNode *ValC = dyn_cast<ConstantSDNode>(Src);
  unsigned BytesLeft = 0;
  bool TwoRepStos = false;
  if (ValC) {
    unsigned ValReg;
    uint64_t Val = ValC->getValue() & 255;

    // If the value is a constant, then we can potentially use larger sets.
    switch (Align & 3) {
      case 2:   // WORD aligned
        AVT = MVT::i16;
        ValReg = X86::AX;
        Val = (Val << 8) | Val;
        break;
      case 0:  // DWORD aligned
        AVT = MVT::i32;
        ValReg = X86::EAX;
        Val = (Val << 8)  | Val;
        Val = (Val << 16) | Val;
        if (Subtarget->is64Bit() && ((Align & 0x7) == 0)) {  // QWORD aligned
          AVT = MVT::i64;
          ValReg = X86::RAX;
          Val = (Val << 32) | Val;
        }
        break;
      default:  // Byte aligned
        AVT = MVT::i8;
        ValReg = X86::AL;
        Count = DAG.getIntPtrConstant(SizeVal);
        break;
    }

    if (AVT > MVT::i8) {
      unsigned UBytes = MVT::getSizeInBits(AVT) / 8;
      Count = DAG.getIntPtrConstant(SizeVal / UBytes);
      BytesLeft = SizeVal % UBytes;
    }

    Chain  = DAG.getCopyToReg(Chain, ValReg, DAG.getConstant(Val, AVT),
                              InFlag);
    InFlag = Chain.getValue(1);
  } else {
    AVT = MVT::i8;
    Count  = DAG.getIntPtrConstant(SizeVal);
    Chain  = DAG.getCopyToReg(Chain, X86::AL, Src, InFlag);
    InFlag = Chain.getValue(1);
  }

  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RCX : X86::ECX,
                            Count, InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RDI : X86::EDI,
                            Dst, InFlag);
  InFlag = Chain.getValue(1);

  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  SmallVector<SDOperand, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(DAG.getValueType(AVT));
  Ops.push_back(InFlag);
  Chain  = DAG.getNode(X86ISD::REP_STOS, Tys, &Ops[0], Ops.size());

  if (TwoRepStos) {
    InFlag = Chain.getValue(1);
    Count  = Size;
    MVT::ValueType CVT = Count.getValueType();
    SDOperand Left = DAG.getNode(ISD::AND, CVT, Count,
                               DAG.getConstant((AVT == MVT::i64) ? 7 : 3, CVT));
    Chain  = DAG.getCopyToReg(Chain, (CVT == MVT::i64) ? X86::RCX : X86::ECX,
                              Left, InFlag);
    InFlag = Chain.getValue(1);
    Tys = DAG.getVTList(MVT::Other, MVT::Flag);
    Ops.clear();
    Ops.push_back(Chain);
    Ops.push_back(DAG.getValueType(MVT::i8));
    Ops.push_back(InFlag);
    Chain  = DAG.getNode(X86ISD::REP_STOS, Tys, &Ops[0], Ops.size());
  } else if (BytesLeft) {
    // Handle the last 1 - 7 bytes.
    unsigned Offset = SizeVal - BytesLeft;
    MVT::ValueType AddrVT = Dst.getValueType();
    MVT::ValueType SizeVT = Size.getValueType();

    Chain = DAG.getMemset(Chain,
                          DAG.getNode(ISD::ADD, AddrVT, Dst,
                                      DAG.getConstant(Offset, AddrVT)),
                          Src,
                          DAG.getConstant(BytesLeft, SizeVT),
                          Align, DstSV, DstSVOff + Offset);
  }

  // TODO: Use a Tokenfactor, as in memcpy, instead of a single chain.
  return Chain;
}

SDOperand
X86TargetLowering::EmitTargetCodeForMemcpy(SelectionDAG &DAG,
                                           SDOperand Chain,
                                           SDOperand Dst, SDOperand Src,
                                           SDOperand Size, unsigned Align,
                                           bool AlwaysInline,
                                           const Value *DstSV, uint64_t DstSVOff,
                                           const Value *SrcSV, uint64_t SrcSVOff){
  
  // This requires the copy size to be a constant, preferrably
  // within a subtarget-specific limit.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDOperand();
  uint64_t SizeVal = ConstantSize->getValue();
  if (!AlwaysInline && SizeVal > getSubtarget()->getMaxInlineSizeThreshold())
    return SDOperand();

  MVT::ValueType AVT;
  unsigned BytesLeft = 0;
  if (Align >= 8 && Subtarget->is64Bit())
    AVT = MVT::i64;
  else if (Align >= 4)
    AVT = MVT::i32;
  else if (Align >= 2)
    AVT = MVT::i16;
  else
    AVT = MVT::i8;

  unsigned UBytes = MVT::getSizeInBits(AVT) / 8;
  unsigned CountVal = SizeVal / UBytes;
  SDOperand Count = DAG.getIntPtrConstant(CountVal);
  BytesLeft = SizeVal % UBytes;

  SDOperand InFlag(0, 0);
  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RCX : X86::ECX,
                            Count, InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RDI : X86::EDI,
                            Dst, InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RSI : X86::ESI,
                            Src, InFlag);
  InFlag = Chain.getValue(1);

  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  SmallVector<SDOperand, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(DAG.getValueType(AVT));
  Ops.push_back(InFlag);
  SDOperand RepMovs = DAG.getNode(X86ISD::REP_MOVS, Tys, &Ops[0], Ops.size());

  SmallVector<SDOperand, 4> Results;
  Results.push_back(RepMovs);
  if (BytesLeft) {
    // Handle the last 1 - 7 bytes.
    unsigned Offset = SizeVal - BytesLeft;
    MVT::ValueType DstVT = Dst.getValueType();
    MVT::ValueType SrcVT = Src.getValueType();
    MVT::ValueType SizeVT = Size.getValueType();
    Results.push_back(DAG.getMemcpy(Chain,
                                    DAG.getNode(ISD::ADD, DstVT, Dst,
                                                DAG.getConstant(Offset, DstVT)),
                                    DAG.getNode(ISD::ADD, SrcVT, Src,
                                                DAG.getConstant(Offset, SrcVT)),
                                    DAG.getConstant(BytesLeft, SizeVT),
                                    Align, AlwaysInline,
                                    DstSV, DstSVOff + Offset,
                                    SrcSV, SrcSVOff + Offset));
  }

  return DAG.getNode(ISD::TokenFactor, MVT::Other, &Results[0], Results.size());
}

/// Expand the result of: i64,outchain = READCYCLECOUNTER inchain
SDNode *X86TargetLowering::ExpandREADCYCLECOUNTER(SDNode *N, SelectionDAG &DAG){
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  SDOperand TheChain = N->getOperand(0);
  SDOperand rd = DAG.getNode(X86ISD::RDTSC_DAG, Tys, &TheChain, 1);
  if (Subtarget->is64Bit()) {
    SDOperand rax = DAG.getCopyFromReg(rd, X86::RAX, MVT::i64, rd.getValue(1));
    SDOperand rdx = DAG.getCopyFromReg(rax.getValue(1), X86::RDX,
                                       MVT::i64, rax.getValue(2));
    SDOperand Tmp = DAG.getNode(ISD::SHL, MVT::i64, rdx,
                                DAG.getConstant(32, MVT::i8));
    SDOperand Ops[] = {
      DAG.getNode(ISD::OR, MVT::i64, rax, Tmp), rdx.getValue(1)
    };
    
    Tys = DAG.getVTList(MVT::i64, MVT::Other);
    return DAG.getNode(ISD::MERGE_VALUES, Tys, Ops, 2).Val;
  }
  
  SDOperand eax = DAG.getCopyFromReg(rd, X86::EAX, MVT::i32, rd.getValue(1));
  SDOperand edx = DAG.getCopyFromReg(eax.getValue(1), X86::EDX,
                                       MVT::i32, eax.getValue(2));
  // Use a buildpair to merge the two 32-bit values into a 64-bit one. 
  SDOperand Ops[] = { eax, edx };
  Ops[0] = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Ops, 2);

  // Use a MERGE_VALUES to return the value and chain.
  Ops[1] = edx.getValue(1);
  Tys = DAG.getVTList(MVT::i64, MVT::Other);
  return DAG.getNode(ISD::MERGE_VALUES, Tys, Ops, 2).Val;
}

SDOperand X86TargetLowering::LowerVASTART(SDOperand Op, SelectionDAG &DAG) {
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  if (!Subtarget->is64Bit()) {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, getPointerTy());
    return DAG.getStore(Op.getOperand(0), FR,Op.getOperand(1), SV, 0);
  }

  // __va_list_tag:
  //   gp_offset         (0 - 6 * 8)
  //   fp_offset         (48 - 48 + 8 * 16)
  //   overflow_arg_area (point to parameters coming in memory).
  //   reg_save_area
  SmallVector<SDOperand, 8> MemOps;
  SDOperand FIN = Op.getOperand(1);
  // Store gp_offset
  SDOperand Store = DAG.getStore(Op.getOperand(0),
                                 DAG.getConstant(VarArgsGPOffset, MVT::i32),
                                 FIN, SV, 0);
  MemOps.push_back(Store);

  // Store fp_offset
  FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN, DAG.getIntPtrConstant(4));
  Store = DAG.getStore(Op.getOperand(0),
                       DAG.getConstant(VarArgsFPOffset, MVT::i32),
                       FIN, SV, 0);
  MemOps.push_back(Store);

  // Store ptr to overflow_arg_area
  FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN, DAG.getIntPtrConstant(4));
  SDOperand OVFIN = DAG.getFrameIndex(VarArgsFrameIndex, getPointerTy());
  Store = DAG.getStore(Op.getOperand(0), OVFIN, FIN, SV, 0);
  MemOps.push_back(Store);

  // Store ptr to reg_save_area.
  FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN, DAG.getIntPtrConstant(8));
  SDOperand RSFIN = DAG.getFrameIndex(RegSaveFrameIndex, getPointerTy());
  Store = DAG.getStore(Op.getOperand(0), RSFIN, FIN, SV, 0);
  MemOps.push_back(Store);
  return DAG.getNode(ISD::TokenFactor, MVT::Other, &MemOps[0], MemOps.size());
}

SDOperand X86TargetLowering::LowerVAARG(SDOperand Op, SelectionDAG &DAG) {
  // X86-64 va_list is a struct { i32, i32, i8*, i8* }.
  assert(Subtarget->is64Bit() && "This code only handles 64-bit va_arg!");
  SDOperand Chain = Op.getOperand(0);
  SDOperand SrcPtr = Op.getOperand(1);
  SDOperand SrcSV = Op.getOperand(2);

  assert(0 && "VAArgInst is not yet implemented for x86-64!");
  abort();
}

SDOperand X86TargetLowering::LowerVACOPY(SDOperand Op, SelectionDAG &DAG) {
  // X86-64 va_list is a struct { i32, i32, i8*, i8* }.
  assert(Subtarget->is64Bit() && "This code only handles 64-bit va_copy!");
  SDOperand Chain = Op.getOperand(0);
  SDOperand DstPtr = Op.getOperand(1);
  SDOperand SrcPtr = Op.getOperand(2);
  const Value *DstSV = cast<SrcValueSDNode>(Op.getOperand(3))->getValue();
  const Value *SrcSV = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();

  return DAG.getMemcpy(Chain, DstPtr, SrcPtr,
                       DAG.getIntPtrConstant(24), 8, false,
                       DstSV, 0, SrcSV, 0);
}

SDOperand
X86TargetLowering::LowerINTRINSIC_WO_CHAIN(SDOperand Op, SelectionDAG &DAG) {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getValue();
  switch (IntNo) {
  default: return SDOperand();    // Don't custom lower most intrinsics.
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
    unsigned Opc = 0;
    ISD::CondCode CC = ISD::SETCC_INVALID;
    switch (IntNo) {
    default: break;
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

    unsigned X86CC;
    SDOperand LHS = Op.getOperand(1);
    SDOperand RHS = Op.getOperand(2);
    translateX86CC(CC, true, X86CC, LHS, RHS, DAG);

    SDOperand Cond = DAG.getNode(Opc, MVT::i32, LHS, RHS);
    SDOperand SetCC = DAG.getNode(X86ISD::SETCC, MVT::i8,
                                  DAG.getConstant(X86CC, MVT::i8), Cond);
    return DAG.getNode(ISD::ANY_EXTEND, MVT::i32, SetCC);
  }

  // Fix vector shift instructions where the last operand is a non-immediate
  // i32 value.
  case Intrinsic::x86_sse2_pslli_w:
  case Intrinsic::x86_sse2_pslli_d:
  case Intrinsic::x86_sse2_pslli_q:
  case Intrinsic::x86_sse2_psrli_w:
  case Intrinsic::x86_sse2_psrli_d:
  case Intrinsic::x86_sse2_psrli_q:
  case Intrinsic::x86_sse2_psrai_w:
  case Intrinsic::x86_sse2_psrai_d:
  case Intrinsic::x86_mmx_pslli_w:
  case Intrinsic::x86_mmx_pslli_d:
  case Intrinsic::x86_mmx_pslli_q:
  case Intrinsic::x86_mmx_psrli_w:
  case Intrinsic::x86_mmx_psrli_d:
  case Intrinsic::x86_mmx_psrli_q:
  case Intrinsic::x86_mmx_psrai_w:
  case Intrinsic::x86_mmx_psrai_d: {
    SDOperand ShAmt = Op.getOperand(2);
    if (isa<ConstantSDNode>(ShAmt))
      return SDOperand();

    unsigned NewIntNo = 0;
    MVT::ValueType ShAmtVT = MVT::v4i32;
    switch (IntNo) {
    case Intrinsic::x86_sse2_pslli_w:
      NewIntNo = Intrinsic::x86_sse2_psll_w;
      break;
    case Intrinsic::x86_sse2_pslli_d:
      NewIntNo = Intrinsic::x86_sse2_psll_d;
      break;
    case Intrinsic::x86_sse2_pslli_q:
      NewIntNo = Intrinsic::x86_sse2_psll_q;
      break;
    case Intrinsic::x86_sse2_psrli_w:
      NewIntNo = Intrinsic::x86_sse2_psrl_w;
      break;
    case Intrinsic::x86_sse2_psrli_d:
      NewIntNo = Intrinsic::x86_sse2_psrl_d;
      break;
    case Intrinsic::x86_sse2_psrli_q:
      NewIntNo = Intrinsic::x86_sse2_psrl_q;
      break;
    case Intrinsic::x86_sse2_psrai_w:
      NewIntNo = Intrinsic::x86_sse2_psra_w;
      break;
    case Intrinsic::x86_sse2_psrai_d:
      NewIntNo = Intrinsic::x86_sse2_psra_d;
      break;
    default: {
      ShAmtVT = MVT::v2i32;
      switch (IntNo) {
      case Intrinsic::x86_mmx_pslli_w:
        NewIntNo = Intrinsic::x86_mmx_psll_w;
        break;
      case Intrinsic::x86_mmx_pslli_d:
        NewIntNo = Intrinsic::x86_mmx_psll_d;
        break;
      case Intrinsic::x86_mmx_pslli_q:
        NewIntNo = Intrinsic::x86_mmx_psll_q;
        break;
      case Intrinsic::x86_mmx_psrli_w:
        NewIntNo = Intrinsic::x86_mmx_psrl_w;
        break;
      case Intrinsic::x86_mmx_psrli_d:
        NewIntNo = Intrinsic::x86_mmx_psrl_d;
        break;
      case Intrinsic::x86_mmx_psrli_q:
        NewIntNo = Intrinsic::x86_mmx_psrl_q;
        break;
      case Intrinsic::x86_mmx_psrai_w:
        NewIntNo = Intrinsic::x86_mmx_psra_w;
        break;
      case Intrinsic::x86_mmx_psrai_d:
        NewIntNo = Intrinsic::x86_mmx_psra_d;
        break;
      default: abort();  // Can't reach here.
      }
      break;
    }
    }
    MVT::ValueType VT = Op.getValueType();
    ShAmt = DAG.getNode(ISD::BIT_CONVERT, VT,
                        DAG.getNode(ISD::SCALAR_TO_VECTOR, ShAmtVT, ShAmt));
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, VT,
                       DAG.getConstant(NewIntNo, MVT::i32),
                       Op.getOperand(1), ShAmt);
  }
  }
}

SDOperand X86TargetLowering::LowerRETURNADDR(SDOperand Op, SelectionDAG &DAG) {
  // Depths > 0 not supported yet!
  if (cast<ConstantSDNode>(Op.getOperand(0))->getValue() > 0)
    return SDOperand();
  
  // Just load the return address
  SDOperand RetAddrFI = getReturnAddressFrameIndex(DAG);
  return DAG.getLoad(getPointerTy(), DAG.getEntryNode(), RetAddrFI, NULL, 0);
}

SDOperand X86TargetLowering::LowerFRAMEADDR(SDOperand Op, SelectionDAG &DAG) {
  // Depths > 0 not supported yet!
  if (cast<ConstantSDNode>(Op.getOperand(0))->getValue() > 0)
    return SDOperand();
    
  SDOperand RetAddrFI = getReturnAddressFrameIndex(DAG);
  return DAG.getNode(ISD::SUB, getPointerTy(), RetAddrFI, 
                     DAG.getIntPtrConstant(4));
}

SDOperand X86TargetLowering::LowerFRAME_TO_ARGS_OFFSET(SDOperand Op,
                                                       SelectionDAG &DAG) {
  // Is not yet supported on x86-64
  if (Subtarget->is64Bit())
    return SDOperand();
  
  return DAG.getIntPtrConstant(8);
}

SDOperand X86TargetLowering::LowerEH_RETURN(SDOperand Op, SelectionDAG &DAG)
{
  assert(!Subtarget->is64Bit() &&
         "Lowering of eh_return builtin is not supported yet on x86-64");
    
  MachineFunction &MF = DAG.getMachineFunction();
  SDOperand Chain     = Op.getOperand(0);
  SDOperand Offset    = Op.getOperand(1);
  SDOperand Handler   = Op.getOperand(2);

  SDOperand Frame = DAG.getRegister(RegInfo->getFrameRegister(MF),
                                    getPointerTy());

  SDOperand StoreAddr = DAG.getNode(ISD::SUB, getPointerTy(), Frame,
                                    DAG.getIntPtrConstant(-4UL));
  StoreAddr = DAG.getNode(ISD::ADD, getPointerTy(), StoreAddr, Offset);
  Chain = DAG.getStore(Chain, Handler, StoreAddr, NULL, 0);
  Chain = DAG.getCopyToReg(Chain, X86::ECX, StoreAddr);
  MF.getRegInfo().addLiveOut(X86::ECX);

  return DAG.getNode(X86ISD::EH_RETURN, MVT::Other,
                     Chain, DAG.getRegister(X86::ECX, getPointerTy()));
}

SDOperand X86TargetLowering::LowerTRAMPOLINE(SDOperand Op,
                                             SelectionDAG &DAG) {
  SDOperand Root = Op.getOperand(0);
  SDOperand Trmp = Op.getOperand(1); // trampoline
  SDOperand FPtr = Op.getOperand(2); // nested function
  SDOperand Nest = Op.getOperand(3); // 'nest' parameter value

  const Value *TrmpAddr = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();

  const X86InstrInfo *TII =
    ((X86TargetMachine&)getTargetMachine()).getInstrInfo();

  if (Subtarget->is64Bit()) {
    SDOperand OutChains[6];

    // Large code-model.

    const unsigned char JMP64r  = TII->getBaseOpcodeFor(X86::JMP64r);
    const unsigned char MOV64ri = TII->getBaseOpcodeFor(X86::MOV64ri);

    const unsigned char N86R10 =
      ((const X86RegisterInfo*)RegInfo)->getX86RegNum(X86::R10);
    const unsigned char N86R11 =
      ((const X86RegisterInfo*)RegInfo)->getX86RegNum(X86::R11);

    const unsigned char REX_WB = 0x40 | 0x08 | 0x01; // REX prefix

    // Load the pointer to the nested function into R11.
    unsigned OpCode = ((MOV64ri | N86R11) << 8) | REX_WB; // movabsq r11
    SDOperand Addr = Trmp;
    OutChains[0] = DAG.getStore(Root, DAG.getConstant(OpCode, MVT::i16), Addr,
                                TrmpAddr, 0);

    Addr = DAG.getNode(ISD::ADD, MVT::i64, Trmp, DAG.getConstant(2, MVT::i64));
    OutChains[1] = DAG.getStore(Root, FPtr, Addr, TrmpAddr, 2, false, 2);

    // Load the 'nest' parameter value into R10.
    // R10 is specified in X86CallingConv.td
    OpCode = ((MOV64ri | N86R10) << 8) | REX_WB; // movabsq r10
    Addr = DAG.getNode(ISD::ADD, MVT::i64, Trmp, DAG.getConstant(10, MVT::i64));
    OutChains[2] = DAG.getStore(Root, DAG.getConstant(OpCode, MVT::i16), Addr,
                                TrmpAddr, 10);

    Addr = DAG.getNode(ISD::ADD, MVT::i64, Trmp, DAG.getConstant(12, MVT::i64));
    OutChains[3] = DAG.getStore(Root, Nest, Addr, TrmpAddr, 12, false, 2);

    // Jump to the nested function.
    OpCode = (JMP64r << 8) | REX_WB; // jmpq *...
    Addr = DAG.getNode(ISD::ADD, MVT::i64, Trmp, DAG.getConstant(20, MVT::i64));
    OutChains[4] = DAG.getStore(Root, DAG.getConstant(OpCode, MVT::i16), Addr,
                                TrmpAddr, 20);

    unsigned char ModRM = N86R11 | (4 << 3) | (3 << 6); // ...r11
    Addr = DAG.getNode(ISD::ADD, MVT::i64, Trmp, DAG.getConstant(22, MVT::i64));
    OutChains[5] = DAG.getStore(Root, DAG.getConstant(ModRM, MVT::i8), Addr,
                                TrmpAddr, 22);

    SDOperand Ops[] =
      { Trmp, DAG.getNode(ISD::TokenFactor, MVT::Other, OutChains, 6) };
    return DAG.getNode(ISD::MERGE_VALUES, Op.Val->getVTList(), Ops, 2);
  } else {
    const Function *Func =
      cast<Function>(cast<SrcValueSDNode>(Op.getOperand(5))->getValue());
    unsigned CC = Func->getCallingConv();
    unsigned NestReg;

    switch (CC) {
    default:
      assert(0 && "Unsupported calling convention");
    case CallingConv::C:
    case CallingConv::X86_StdCall: {
      // Pass 'nest' parameter in ECX.
      // Must be kept in sync with X86CallingConv.td
      NestReg = X86::ECX;

      // Check that ECX wasn't needed by an 'inreg' parameter.
      const FunctionType *FTy = Func->getFunctionType();
      const PAListPtr &Attrs = Func->getParamAttrs();

      if (!Attrs.isEmpty() && !Func->isVarArg()) {
        unsigned InRegCount = 0;
        unsigned Idx = 1;

        for (FunctionType::param_iterator I = FTy->param_begin(),
             E = FTy->param_end(); I != E; ++I, ++Idx)
          if (Attrs.paramHasAttr(Idx, ParamAttr::InReg))
            // FIXME: should only count parameters that are lowered to integers.
            InRegCount += (getTargetData()->getTypeSizeInBits(*I) + 31) / 32;

        if (InRegCount > 2) {
          cerr << "Nest register in use - reduce number of inreg parameters!\n";
          abort();
        }
      }
      break;
    }
    case CallingConv::X86_FastCall:
      // Pass 'nest' parameter in EAX.
      // Must be kept in sync with X86CallingConv.td
      NestReg = X86::EAX;
      break;
    }

    SDOperand OutChains[4];
    SDOperand Addr, Disp;

    Addr = DAG.getNode(ISD::ADD, MVT::i32, Trmp, DAG.getConstant(10, MVT::i32));
    Disp = DAG.getNode(ISD::SUB, MVT::i32, FPtr, Addr);

    const unsigned char MOV32ri = TII->getBaseOpcodeFor(X86::MOV32ri);
    const unsigned char N86Reg =
      ((const X86RegisterInfo*)RegInfo)->getX86RegNum(NestReg);
    OutChains[0] = DAG.getStore(Root, DAG.getConstant(MOV32ri|N86Reg, MVT::i8),
                                Trmp, TrmpAddr, 0);

    Addr = DAG.getNode(ISD::ADD, MVT::i32, Trmp, DAG.getConstant(1, MVT::i32));
    OutChains[1] = DAG.getStore(Root, Nest, Addr, TrmpAddr, 1, false, 1);

    const unsigned char JMP = TII->getBaseOpcodeFor(X86::JMP);
    Addr = DAG.getNode(ISD::ADD, MVT::i32, Trmp, DAG.getConstant(5, MVT::i32));
    OutChains[2] = DAG.getStore(Root, DAG.getConstant(JMP, MVT::i8), Addr,
                                TrmpAddr, 5, false, 1);

    Addr = DAG.getNode(ISD::ADD, MVT::i32, Trmp, DAG.getConstant(6, MVT::i32));
    OutChains[3] = DAG.getStore(Root, Disp, Addr, TrmpAddr, 6, false, 1);

    SDOperand Ops[] =
      { Trmp, DAG.getNode(ISD::TokenFactor, MVT::Other, OutChains, 4) };
    return DAG.getNode(ISD::MERGE_VALUES, Op.Val->getVTList(), Ops, 2);
  }
}

SDOperand X86TargetLowering::LowerFLT_ROUNDS_(SDOperand Op, SelectionDAG &DAG) {
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
  const TargetFrameInfo &TFI = *TM.getFrameInfo();
  unsigned StackAlignment = TFI.getStackAlignment();
  MVT::ValueType VT = Op.getValueType();

  // Save FP Control Word to stack slot
  int SSFI = MF.getFrameInfo()->CreateStackObject(2, StackAlignment);
  SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());

  SDOperand Chain = DAG.getNode(X86ISD::FNSTCW16m, MVT::Other,
                                DAG.getEntryNode(), StackSlot);

  // Load FP Control Word from stack slot
  SDOperand CWD = DAG.getLoad(MVT::i16, Chain, StackSlot, NULL, 0);

  // Transform as necessary
  SDOperand CWD1 =
    DAG.getNode(ISD::SRL, MVT::i16,
                DAG.getNode(ISD::AND, MVT::i16,
                            CWD, DAG.getConstant(0x800, MVT::i16)),
                DAG.getConstant(11, MVT::i8));
  SDOperand CWD2 =
    DAG.getNode(ISD::SRL, MVT::i16,
                DAG.getNode(ISD::AND, MVT::i16,
                            CWD, DAG.getConstant(0x400, MVT::i16)),
                DAG.getConstant(9, MVT::i8));

  SDOperand RetVal =
    DAG.getNode(ISD::AND, MVT::i16,
                DAG.getNode(ISD::ADD, MVT::i16,
                            DAG.getNode(ISD::OR, MVT::i16, CWD1, CWD2),
                            DAG.getConstant(1, MVT::i16)),
                DAG.getConstant(3, MVT::i16));


  return DAG.getNode((MVT::getSizeInBits(VT) < 16 ?
                      ISD::TRUNCATE : ISD::ZERO_EXTEND), VT, RetVal);
}

SDOperand X86TargetLowering::LowerCTLZ(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType OpVT = VT;
  unsigned NumBits = MVT::getSizeInBits(VT);

  Op = Op.getOperand(0);
  if (VT == MVT::i8) {
    // Zero extend to i32 since there is not an i8 bsr.
    OpVT = MVT::i32;
    Op = DAG.getNode(ISD::ZERO_EXTEND, OpVT, Op);
  }

  // Issue a bsr (scan bits in reverse) which also sets EFLAGS.
  SDVTList VTs = DAG.getVTList(OpVT, MVT::i32);
  Op = DAG.getNode(X86ISD::BSR, VTs, Op);

  // If src is zero (i.e. bsr sets ZF), returns NumBits.
  SmallVector<SDOperand, 4> Ops;
  Ops.push_back(Op);
  Ops.push_back(DAG.getConstant(NumBits+NumBits-1, OpVT));
  Ops.push_back(DAG.getConstant(X86::COND_E, MVT::i8));
  Ops.push_back(Op.getValue(1));
  Op = DAG.getNode(X86ISD::CMOV, OpVT, &Ops[0], 4);

  // Finally xor with NumBits-1.
  Op = DAG.getNode(ISD::XOR, OpVT, Op, DAG.getConstant(NumBits-1, OpVT));

  if (VT == MVT::i8)
    Op = DAG.getNode(ISD::TRUNCATE, MVT::i8, Op);
  return Op;
}

SDOperand X86TargetLowering::LowerCTTZ(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType OpVT = VT;
  unsigned NumBits = MVT::getSizeInBits(VT);

  Op = Op.getOperand(0);
  if (VT == MVT::i8) {
    OpVT = MVT::i32;
    Op = DAG.getNode(ISD::ZERO_EXTEND, OpVT, Op);
  }

  // Issue a bsf (scan bits forward) which also sets EFLAGS.
  SDVTList VTs = DAG.getVTList(OpVT, MVT::i32);
  Op = DAG.getNode(X86ISD::BSF, VTs, Op);

  // If src is zero (i.e. bsf sets ZF), returns NumBits.
  SmallVector<SDOperand, 4> Ops;
  Ops.push_back(Op);
  Ops.push_back(DAG.getConstant(NumBits, OpVT));
  Ops.push_back(DAG.getConstant(X86::COND_E, MVT::i8));
  Ops.push_back(Op.getValue(1));
  Op = DAG.getNode(X86ISD::CMOV, OpVT, &Ops[0], 4);

  if (VT == MVT::i8)
    Op = DAG.getNode(ISD::TRUNCATE, MVT::i8, Op);
  return Op;
}

SDOperand X86TargetLowering::LowerLCS(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType T = cast<AtomicSDNode>(Op.Val)->getVT();
  unsigned Reg = 0;
  unsigned size = 0;
  switch(T) {
  case MVT::i8:  Reg = X86::AL;  size = 1; break;
  case MVT::i16: Reg = X86::AX;  size = 2; break;
  case MVT::i32: Reg = X86::EAX; size = 4; break;
  case MVT::i64: 
    if (Subtarget->is64Bit()) {
      Reg = X86::RAX; size = 8;
    } else //Should go away when LowerType stuff lands
      return SDOperand(ExpandATOMIC_LCS(Op.Val, DAG), 0);
    break;
  };
  SDOperand cpIn = DAG.getCopyToReg(Op.getOperand(0), Reg,
                                    Op.getOperand(3), SDOperand());
  SDOperand Ops[] = { cpIn.getValue(0),
                      Op.getOperand(1),
                      Op.getOperand(2),
                      DAG.getTargetConstant(size, MVT::i8),
                      cpIn.getValue(1) };
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  SDOperand Result = DAG.getNode(X86ISD::LCMPXCHG_DAG, Tys, Ops, 5);
  SDOperand cpOut = 
    DAG.getCopyFromReg(Result.getValue(0), Reg, T, Result.getValue(1));
  return cpOut;
}

SDNode* X86TargetLowering::ExpandATOMIC_LCS(SDNode* Op, SelectionDAG &DAG) {
  MVT::ValueType T = cast<AtomicSDNode>(Op)->getVT();
  assert (T == MVT::i64 && "Only know how to expand i64 CAS");
  SDOperand cpInL, cpInH;
  cpInL = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op->getOperand(3),
                      DAG.getConstant(0, MVT::i32));
  cpInH = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op->getOperand(3),
                      DAG.getConstant(1, MVT::i32));
  cpInL = DAG.getCopyToReg(Op->getOperand(0), X86::EAX,
                           cpInL, SDOperand());
  cpInH = DAG.getCopyToReg(cpInL.getValue(0), X86::EDX,
                           cpInH, cpInL.getValue(1));
  SDOperand swapInL, swapInH;
  swapInL = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op->getOperand(2),
                        DAG.getConstant(0, MVT::i32));
  swapInH = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op->getOperand(2),
                        DAG.getConstant(1, MVT::i32));
  swapInL = DAG.getCopyToReg(cpInH.getValue(0), X86::EBX,
                             swapInL, cpInH.getValue(1));
  swapInH = DAG.getCopyToReg(swapInL.getValue(0), X86::ECX,
                             swapInH, swapInL.getValue(1));
  SDOperand Ops[] = { swapInH.getValue(0),
                      Op->getOperand(1),
                      swapInH.getValue(1)};
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  SDOperand Result = DAG.getNode(X86ISD::LCMPXCHG8_DAG, Tys, Ops, 3);
  SDOperand cpOutL = DAG.getCopyFromReg(Result.getValue(0), X86::EAX, MVT::i32, 
                                        Result.getValue(1));
  SDOperand cpOutH = DAG.getCopyFromReg(cpOutL.getValue(1), X86::EDX, MVT::i32, 
                                        cpOutL.getValue(2));
  SDOperand OpsF[] = { cpOutL.getValue(0), cpOutH.getValue(0)};
  SDOperand ResultVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, OpsF, 2);
  Tys = DAG.getVTList(MVT::i64, MVT::Other);
  return DAG.getNode(ISD::MERGE_VALUES, Tys, ResultVal, cpOutH.getValue(1)).Val;
}

SDNode* X86TargetLowering::ExpandATOMIC_LSS(SDNode* Op, SelectionDAG &DAG) {
  MVT::ValueType T = cast<AtomicSDNode>(Op)->getVT();
  assert (T == MVT::i32 && "Only know how to expand i32 LSS");
  SDOperand negOp = DAG.getNode(ISD::SUB, T,
                                DAG.getConstant(0, T), Op->getOperand(2));
  return DAG.getAtomic(ISD::ATOMIC_LAS, Op->getOperand(0),
                       Op->getOperand(1), negOp, T).Val;
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand X86TargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  case ISD::ATOMIC_LCS:         return LowerLCS(Op,DAG);
  case ISD::BUILD_VECTOR:       return LowerBUILD_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:     return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT: return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:  return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::SCALAR_TO_VECTOR:   return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
  case ISD::ExternalSymbol:     return LowerExternalSymbol(Op, DAG);
  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:          return LowerShift(Op, DAG);
  case ISD::SINT_TO_FP:         return LowerSINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:         return LowerFP_TO_SINT(Op, DAG);
  case ISD::FABS:               return LowerFABS(Op, DAG);
  case ISD::FNEG:               return LowerFNEG(Op, DAG);
  case ISD::FCOPYSIGN:          return LowerFCOPYSIGN(Op, DAG);
  case ISD::SETCC:              return LowerSETCC(Op, DAG);
  case ISD::SELECT:             return LowerSELECT(Op, DAG);
  case ISD::BRCOND:             return LowerBRCOND(Op, DAG);
  case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
  case ISD::CALL:               return LowerCALL(Op, DAG);
  case ISD::RET:                return LowerRET(Op, DAG);
  case ISD::FORMAL_ARGUMENTS:   return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::VASTART:            return LowerVASTART(Op, DAG);
  case ISD::VAARG:              return LowerVAARG(Op, DAG);
  case ISD::VACOPY:             return LowerVACOPY(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::RETURNADDR:         return LowerRETURNADDR(Op, DAG);
  case ISD::FRAMEADDR:          return LowerFRAMEADDR(Op, DAG);
  case ISD::FRAME_TO_ARGS_OFFSET:
                                return LowerFRAME_TO_ARGS_OFFSET(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC: return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::EH_RETURN:          return LowerEH_RETURN(Op, DAG);
  case ISD::TRAMPOLINE:         return LowerTRAMPOLINE(Op, DAG);
  case ISD::FLT_ROUNDS_:        return LowerFLT_ROUNDS_(Op, DAG);
  case ISD::CTLZ:               return LowerCTLZ(Op, DAG);
  case ISD::CTTZ:               return LowerCTTZ(Op, DAG);
      
  // FIXME: REMOVE THIS WHEN LegalizeDAGTypes lands.
  case ISD::READCYCLECOUNTER:
    return SDOperand(ExpandREADCYCLECOUNTER(Op.Val, DAG), 0);
  }
}

/// ExpandOperation - Provide custom lowering hooks for expanding operations.
SDNode *X86TargetLowering::ExpandOperationResult(SDNode *N, SelectionDAG &DAG) {
  switch (N->getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  case ISD::FP_TO_SINT:         return ExpandFP_TO_SINT(N, DAG);
  case ISD::READCYCLECOUNTER:   return ExpandREADCYCLECOUNTER(N, DAG);
  case ISD::ATOMIC_LCS:         return ExpandATOMIC_LCS(N, DAG);
  case ISD::ATOMIC_LSS:         return ExpandATOMIC_LSS(N,DAG);
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
  case X86ISD::TAILCALL:           return "X86ISD::TAILCALL";
  case X86ISD::RDTSC_DAG:          return "X86ISD::RDTSC_DAG";
  case X86ISD::CMP:                return "X86ISD::CMP";
  case X86ISD::COMI:               return "X86ISD::COMI";
  case X86ISD::UCOMI:              return "X86ISD::UCOMI";
  case X86ISD::SETCC:              return "X86ISD::SETCC";
  case X86ISD::CMOV:               return "X86ISD::CMOV";
  case X86ISD::BRCOND:             return "X86ISD::BRCOND";
  case X86ISD::RET_FLAG:           return "X86ISD::RET_FLAG";
  case X86ISD::REP_STOS:           return "X86ISD::REP_STOS";
  case X86ISD::REP_MOVS:           return "X86ISD::REP_MOVS";
  case X86ISD::GlobalBaseReg:      return "X86ISD::GlobalBaseReg";
  case X86ISD::Wrapper:            return "X86ISD::Wrapper";
  case X86ISD::PEXTRB:             return "X86ISD::PEXTRB";
  case X86ISD::PEXTRW:             return "X86ISD::PEXTRW";
  case X86ISD::INSERTPS:           return "X86ISD::INSERTPS";
  case X86ISD::PINSRB:             return "X86ISD::PINSRB";
  case X86ISD::PINSRW:             return "X86ISD::PINSRW";
  case X86ISD::FMAX:               return "X86ISD::FMAX";
  case X86ISD::FMIN:               return "X86ISD::FMIN";
  case X86ISD::FRSQRT:             return "X86ISD::FRSQRT";
  case X86ISD::FRCP:               return "X86ISD::FRCP";
  case X86ISD::TLSADDR:            return "X86ISD::TLSADDR";
  case X86ISD::THREAD_POINTER:     return "X86ISD::THREAD_POINTER";
  case X86ISD::EH_RETURN:          return "X86ISD::EH_RETURN";
  case X86ISD::TC_RETURN:          return "X86ISD::TC_RETURN";
  case X86ISD::FNSTCW16m:          return "X86ISD::FNSTCW16m";
  case X86ISD::LCMPXCHG_DAG:       return "X86ISD::LCMPXCHG_DAG";
  case X86ISD::LCMPXCHG8_DAG:      return "X86ISD::LCMPXCHG8_DAG";
  case X86ISD::VZEXT_MOVL:         return "X86ISD::VZEXT_MOVL";
  case X86ISD::VZEXT_LOAD:         return "X86ISD::VZEXT_LOAD";
  }
}

// isLegalAddressingMode - Return true if the addressing mode represented
// by AM is legal for this target, for a load/store of the specified type.
bool X86TargetLowering::isLegalAddressingMode(const AddrMode &AM, 
                                              const Type *Ty) const {
  // X86 supports extremely general addressing modes.
  
  // X86 allows a sign-extended 32-bit immediate field as a displacement.
  if (AM.BaseOffs <= -(1LL << 32) || AM.BaseOffs >= (1LL << 32)-1)
    return false;
  
  if (AM.BaseGV) {
    // We can only fold this if we don't need an extra load.
    if (Subtarget->GVRequiresExtraLoad(AM.BaseGV, getTargetMachine(), false))
      return false;

    // X86-64 only supports addr of globals in small code model.
    if (Subtarget->is64Bit()) {
      if (getTargetMachine().getCodeModel() != CodeModel::Small)
        return false;
      // If lower 4G is not available, then we must use rip-relative addressing.
      if (AM.BaseOffs || AM.Scale > 1)
        return false;
    }
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


bool X86TargetLowering::isTruncateFree(const Type *Ty1, const Type *Ty2) const {
  if (!Ty1->isInteger() || !Ty2->isInteger())
    return false;
  unsigned NumBits1 = Ty1->getPrimitiveSizeInBits();
  unsigned NumBits2 = Ty2->getPrimitiveSizeInBits();
  if (NumBits1 <= NumBits2)
    return false;
  return Subtarget->is64Bit() || NumBits1 < 64;
}

bool X86TargetLowering::isTruncateFree(MVT::ValueType VT1,
                                       MVT::ValueType VT2) const {
  if (!MVT::isInteger(VT1) || !MVT::isInteger(VT2))
    return false;
  unsigned NumBits1 = MVT::getSizeInBits(VT1);
  unsigned NumBits2 = MVT::getSizeInBits(VT2);
  if (NumBits1 <= NumBits2)
    return false;
  return Subtarget->is64Bit() || NumBits1 < 64;
}

/// isShuffleMaskLegal - Targets can use this to indicate that they only
/// support *some* VECTOR_SHUFFLE operations, those with specific masks.
/// By default, if a target supports the VECTOR_SHUFFLE node, all mask values
/// are assumed to be legal.
bool
X86TargetLowering::isShuffleMaskLegal(SDOperand Mask, MVT::ValueType VT) const {
  // Only do shuffles on 128-bit vector types for now.
  if (MVT::getSizeInBits(VT) == 64) return false;
  return (Mask.Val->getNumOperands() <= 4 ||
          isIdentityMask(Mask.Val) ||
          isIdentityMask(Mask.Val, true) ||
          isSplatMask(Mask.Val)  ||
          isPSHUFHW_PSHUFLWMask(Mask.Val) ||
          X86::isUNPCKLMask(Mask.Val) ||
          X86::isUNPCKHMask(Mask.Val) ||
          X86::isUNPCKL_v_undef_Mask(Mask.Val) ||
          X86::isUNPCKH_v_undef_Mask(Mask.Val));
}

bool
X86TargetLowering::isVectorClearMaskLegal(const std::vector<SDOperand> &BVOps,
                                          MVT::ValueType EVT,
                                          SelectionDAG &DAG) const {
  unsigned NumElts = BVOps.size();
  // Only do shuffles on 128-bit vector types for now.
  if (MVT::getSizeInBits(EVT) * NumElts == 64) return false;
  if (NumElts == 2) return true;
  if (NumElts == 4) {
    return (isMOVLMask(&BVOps[0], 4)  ||
            isCommutedMOVL(&BVOps[0], 4, true) ||
            isSHUFPMask(&BVOps[0], 4) || 
            isCommutedSHUFP(&BVOps[0], 4));
  }
  return false;
}

//===----------------------------------------------------------------------===//
//                           X86 Scheduler Hooks
//===----------------------------------------------------------------------===//

// private utility function
MachineBasicBlock *
X86TargetLowering::EmitAtomicBitwiseWithCustomInserter(MachineInstr *bInstr,
                                                       MachineBasicBlock *MBB,
                                                       unsigned regOpc,
                                                       unsigned immOpc) {
  // For the atomic bitwise operator, we generate
  //   thisMBB:
  //   newMBB:
  //     ld  t1 = [bitinstr.addr]
  //     op  t2 = t1, [bitinstr.val]
  //     mov EAX = t1
  //     lcs dest = [bitinstr.addr], t2  [EAX is implicit]
  //     bz  newMBB
  //     fallthrough -->nextMBB
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator MBBIter = MBB;
  ++MBBIter;
  
  /// First build the CFG
  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *newMBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *nextMBB = new MachineBasicBlock(LLVM_BB);
  F->getBasicBlockList().insert(MBBIter, newMBB);
  F->getBasicBlockList().insert(MBBIter, nextMBB);
  
  // Move all successors to thisMBB to nextMBB
  nextMBB->transferSuccessors(thisMBB);
    
  // Update thisMBB to fall through to newMBB
  thisMBB->addSuccessor(newMBB);
  
  // newMBB jumps to itself and fall through to nextMBB
  newMBB->addSuccessor(nextMBB);
  newMBB->addSuccessor(newMBB);
  
  // Insert instructions into newMBB based on incoming instruction
  assert(bInstr->getNumOperands() < 8 && "unexpected number of operands");
  MachineOperand& destOper = bInstr->getOperand(0);
  MachineOperand* argOpers[6];
  int numArgs = bInstr->getNumOperands() - 1;
  for (int i=0; i < numArgs; ++i)
    argOpers[i] = &bInstr->getOperand(i+1);

  // x86 address has 4 operands: base, index, scale, and displacement
  int lastAddrIndx = 3; // [0,3]
  int valArgIndx = 4;
  
  unsigned t1 = F->getRegInfo().createVirtualRegister(X86::GR32RegisterClass);
  MachineInstrBuilder MIB = BuildMI(newMBB, TII->get(X86::MOV32rm), t1);
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);
  
  unsigned t2 = F->getRegInfo().createVirtualRegister(X86::GR32RegisterClass);
  assert(   (argOpers[valArgIndx]->isReg() || argOpers[valArgIndx]->isImm())
         && "invalid operand");
  if (argOpers[valArgIndx]->isReg())
    MIB = BuildMI(newMBB, TII->get(regOpc), t2);
  else
    MIB = BuildMI(newMBB, TII->get(immOpc), t2);
  MIB.addReg(t1);
  (*MIB).addOperand(*argOpers[valArgIndx]);
  
  MIB = BuildMI(newMBB, TII->get(X86::MOV32rr), X86::EAX);
  MIB.addReg(t1);
  
  MIB = BuildMI(newMBB, TII->get(X86::LCMPXCHG32));
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);
  MIB.addReg(t2);
  
  MIB = BuildMI(newMBB, TII->get(X86::MOV32rr), destOper.getReg());
  MIB.addReg(X86::EAX);
  
  // insert branch
  BuildMI(newMBB, TII->get(X86::JNE)).addMBB(newMBB);

  delete bInstr;   // The pseudo instruction is gone now.
  return nextMBB;
}

// private utility function
MachineBasicBlock *
X86TargetLowering::EmitAtomicMinMaxWithCustomInserter(MachineInstr *mInstr,
                                                      MachineBasicBlock *MBB,
                                                      unsigned cmovOpc) {
  // For the atomic min/max operator, we generate
  //   thisMBB:
  //   newMBB:
  //     ld t1 = [min/max.addr]
  //     mov t2 = [min/max.val] 
  //     cmp  t1, t2
  //     cmov[cond] t2 = t1
  //     mov EAX = t1
  //     lcs dest = [bitinstr.addr], t2  [EAX is implicit]
  //     bz   newMBB
  //     fallthrough -->nextMBB
  //
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator MBBIter = MBB;
  ++MBBIter;
  
  /// First build the CFG
  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *newMBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *nextMBB = new MachineBasicBlock(LLVM_BB);
  F->getBasicBlockList().insert(MBBIter, newMBB);
  F->getBasicBlockList().insert(MBBIter, nextMBB);
  
  // Move all successors to thisMBB to nextMBB
  nextMBB->transferSuccessors(thisMBB);
  
  // Update thisMBB to fall through to newMBB
  thisMBB->addSuccessor(newMBB);
  
  // newMBB jumps to newMBB and fall through to nextMBB
  newMBB->addSuccessor(nextMBB);
  newMBB->addSuccessor(newMBB);
  
  // Insert instructions into newMBB based on incoming instruction
  assert(mInstr->getNumOperands() < 8 && "unexpected number of operands");
  MachineOperand& destOper = mInstr->getOperand(0);
  MachineOperand* argOpers[6];
  int numArgs = mInstr->getNumOperands() - 1;
  for (int i=0; i < numArgs; ++i)
    argOpers[i] = &mInstr->getOperand(i+1);
  
  // x86 address has 4 operands: base, index, scale, and displacement
  int lastAddrIndx = 3; // [0,3]
  int valArgIndx = 4;
  
  unsigned t1 = F->getRegInfo().createVirtualRegister(X86::GR32RegisterClass);
  MachineInstrBuilder MIB = BuildMI(newMBB, TII->get(X86::MOV32rm), t1);
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);

  // We only support register and immediate values
  assert(   (argOpers[valArgIndx]->isReg() || argOpers[valArgIndx]->isImm())
         && "invalid operand");
  
  unsigned t2 = F->getRegInfo().createVirtualRegister(X86::GR32RegisterClass);  
  if (argOpers[valArgIndx]->isReg())
    MIB = BuildMI(newMBB, TII->get(X86::MOV32rr), t2);
  else 
    MIB = BuildMI(newMBB, TII->get(X86::MOV32rr), t2);
  (*MIB).addOperand(*argOpers[valArgIndx]);

  MIB = BuildMI(newMBB, TII->get(X86::MOV32rr), X86::EAX);
  MIB.addReg(t1);

  MIB = BuildMI(newMBB, TII->get(X86::CMP32rr));
  MIB.addReg(t1);
  MIB.addReg(t2);

  // Generate movc
  unsigned t3 = F->getRegInfo().createVirtualRegister(X86::GR32RegisterClass);
  MIB = BuildMI(newMBB, TII->get(cmovOpc),t3);
  MIB.addReg(t2);
  MIB.addReg(t1);

  // Cmp and exchange if none has modified the memory location
  MIB = BuildMI(newMBB, TII->get(X86::LCMPXCHG32));
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);
  MIB.addReg(t3);
  
  MIB = BuildMI(newMBB, TII->get(X86::MOV32rr), destOper.getReg());
  MIB.addReg(X86::EAX);
  
  // insert branch
  BuildMI(newMBB, TII->get(X86::JNE)).addMBB(newMBB);

  delete mInstr;   // The pseudo instruction is gone now.
  return nextMBB;
}


MachineBasicBlock *
X86TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                               MachineBasicBlock *BB) {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  switch (MI->getOpcode()) {
  default: assert(false && "Unexpected instr type to insert");
  case X86::CMOV_FR32:
  case X86::CMOV_FR64:
  case X86::CMOV_V4F32:
  case X86::CMOV_V2F64:
  case X86::CMOV_V2I64: {
    // To "insert" a SELECT_CC instruction, we actually have to insert the
    // diamond control-flow pattern.  The incoming instruction knows the
    // destination vreg to set, the condition code register to branch on, the
    // true/false values to select between, and a branch opcode to use.
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY ccX, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *thisMBB = BB;
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
    unsigned Opc =
      X86::GetCondBranchFromCond((X86::CondCode)MI->getOperand(3).getImm());
    BuildMI(BB, TII->get(Opc)).addMBB(sinkMBB);
    MachineFunction *F = BB->getParent();
    F->getBasicBlockList().insert(It, copy0MBB);
    F->getBasicBlockList().insert(It, sinkMBB);
    // Update machine-CFG edges by transferring all successors of the current
    // block to the new block which will contain the Phi node for the select.
    sinkMBB->transferSuccessors(BB);

    // Add the true and fallthrough blocks as its successors.
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
    BuildMI(BB, TII->get(X86::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

    delete MI;   // The pseudo instruction is gone now.
    return BB;
  }

  case X86::FP32_TO_INT16_IN_MEM:
  case X86::FP32_TO_INT32_IN_MEM:
  case X86::FP32_TO_INT64_IN_MEM:
  case X86::FP64_TO_INT16_IN_MEM:
  case X86::FP64_TO_INT32_IN_MEM:
  case X86::FP64_TO_INT64_IN_MEM:
  case X86::FP80_TO_INT16_IN_MEM:
  case X86::FP80_TO_INT32_IN_MEM:
  case X86::FP80_TO_INT64_IN_MEM: {
    // Change the floating point control register to use "round towards zero"
    // mode when truncating to an integer value.
    MachineFunction *F = BB->getParent();
    int CWFrameIdx = F->getFrameInfo()->CreateStackObject(2, 2);
    addFrameReference(BuildMI(BB, TII->get(X86::FNSTCW16m)), CWFrameIdx);

    // Load the old value of the high byte of the control word...
    unsigned OldCW =
      F->getRegInfo().createVirtualRegister(X86::GR16RegisterClass);
    addFrameReference(BuildMI(BB, TII->get(X86::MOV16rm), OldCW), CWFrameIdx);

    // Set the high part to be round to zero...
    addFrameReference(BuildMI(BB, TII->get(X86::MOV16mi)), CWFrameIdx)
      .addImm(0xC7F);

    // Reload the modified control word now...
    addFrameReference(BuildMI(BB, TII->get(X86::FLDCW16m)), CWFrameIdx);

    // Restore the memory image of control word to original value
    addFrameReference(BuildMI(BB, TII->get(X86::MOV16mr)), CWFrameIdx)
      .addReg(OldCW);

    // Get the X86 opcode to use.
    unsigned Opc;
    switch (MI->getOpcode()) {
    default: assert(0 && "illegal opcode!");
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
    if (Op.isRegister()) {
      AM.BaseType = X86AddressMode::RegBase;
      AM.Base.Reg = Op.getReg();
    } else {
      AM.BaseType = X86AddressMode::FrameIndexBase;
      AM.Base.FrameIndex = Op.getIndex();
    }
    Op = MI->getOperand(1);
    if (Op.isImmediate())
      AM.Scale = Op.getImm();
    Op = MI->getOperand(2);
    if (Op.isImmediate())
      AM.IndexReg = Op.getImm();
    Op = MI->getOperand(3);
    if (Op.isGlobalAddress()) {
      AM.GV = Op.getGlobal();
    } else {
      AM.Disp = Op.getImm();
    }
    addFullAddress(BuildMI(BB, TII->get(Opc)), AM)
                      .addReg(MI->getOperand(4).getReg());

    // Reload the original control word now.
    addFrameReference(BuildMI(BB, TII->get(X86::FLDCW16m)), CWFrameIdx);

    delete MI;   // The pseudo instruction is gone now.
    return BB;
  }
  case X86::ATOMAND32:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND32rr,
                                                       X86::AND32ri);
  case X86::ATOMOR32:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::OR32rr, 
                                                       X86::OR32ri);
  case X86::ATOMXOR32:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::XOR32rr,
                                                       X86::XOR32ri);
  case X86::ATOMMIN32:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVL32rr);
  case X86::ATOMMAX32:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVG32rr);
  case X86::ATOMUMIN32:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVB32rr);
  case X86::ATOMUMAX32:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVA32rr);
  }
}

//===----------------------------------------------------------------------===//
//                           X86 Optimization Hooks
//===----------------------------------------------------------------------===//

void X86TargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op,
                                                       const APInt &Mask,
                                                       APInt &KnownZero,
                                                       APInt &KnownOne,
                                                       const SelectionDAG &DAG,
                                                       unsigned Depth) const {
  unsigned Opc = Op.getOpcode();
  assert((Opc >= ISD::BUILTIN_OP_END ||
          Opc == ISD::INTRINSIC_WO_CHAIN ||
          Opc == ISD::INTRINSIC_W_CHAIN ||
          Opc == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");

  KnownZero = KnownOne = APInt(Mask.getBitWidth(), 0);   // Don't know anything.
  switch (Opc) {
  default: break;
  case X86ISD::SETCC:
    KnownZero |= APInt::getHighBitsSet(Mask.getBitWidth(),
                                       Mask.getBitWidth() - 1);
    break;
  }
}

/// getShuffleScalarElt - Returns the scalar element that will make up the ith
/// element of the result of the vector shuffle.
static SDOperand getShuffleScalarElt(SDNode *N, unsigned i, SelectionDAG &DAG) {
  MVT::ValueType VT = N->getValueType(0);
  SDOperand PermMask = N->getOperand(2);
  unsigned NumElems = PermMask.getNumOperands();
  SDOperand V = (i < NumElems) ? N->getOperand(0) : N->getOperand(1);
  i %= NumElems;
  if (V.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    return (i == 0)
     ? V.getOperand(0) : DAG.getNode(ISD::UNDEF, MVT::getVectorElementType(VT));
  } else if (V.getOpcode() == ISD::VECTOR_SHUFFLE) {
    SDOperand Idx = PermMask.getOperand(i);
    if (Idx.getOpcode() == ISD::UNDEF)
      return DAG.getNode(ISD::UNDEF, MVT::getVectorElementType(VT));
    return getShuffleScalarElt(V.Val,cast<ConstantSDNode>(Idx)->getValue(),DAG);
  }
  return SDOperand();
}

/// isGAPlusOffset - Returns true (and the GlobalValue and the offset) if the
/// node is a GlobalAddress + an offset.
static bool isGAPlusOffset(SDNode *N, GlobalValue* &GA, int64_t &Offset) {
  unsigned Opc = N->getOpcode();
  if (Opc == X86ISD::Wrapper) {
    if (dyn_cast<GlobalAddressSDNode>(N->getOperand(0))) {
      GA = cast<GlobalAddressSDNode>(N->getOperand(0))->getGlobal();
      return true;
    }
  } else if (Opc == ISD::ADD) {
    SDOperand N1 = N->getOperand(0);
    SDOperand N2 = N->getOperand(1);
    if (isGAPlusOffset(N1.Val, GA, Offset)) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode>(N2);
      if (V) {
        Offset += V->getSignExtended();
        return true;
      }
    } else if (isGAPlusOffset(N2.Val, GA, Offset)) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode>(N1);
      if (V) {
        Offset += V->getSignExtended();
        return true;
      }
    }
  }
  return false;
}

/// isConsecutiveLoad - Returns true if N is loading from an address of Base
/// + Dist * Size.
static bool isConsecutiveLoad(SDNode *N, SDNode *Base, int Dist, int Size,
                              MachineFrameInfo *MFI) {
  if (N->getOperand(0).Val != Base->getOperand(0).Val)
    return false;

  SDOperand Loc = N->getOperand(1);
  SDOperand BaseLoc = Base->getOperand(1);
  if (Loc.getOpcode() == ISD::FrameIndex) {
    if (BaseLoc.getOpcode() != ISD::FrameIndex)
      return false;
    int FI  = cast<FrameIndexSDNode>(Loc)->getIndex();
    int BFI = cast<FrameIndexSDNode>(BaseLoc)->getIndex();
    int FS  = MFI->getObjectSize(FI);
    int BFS = MFI->getObjectSize(BFI);
    if (FS != BFS || FS != Size) return false;
    return MFI->getObjectOffset(FI) == (MFI->getObjectOffset(BFI) + Dist*Size);
  } else {
    GlobalValue *GV1 = NULL;
    GlobalValue *GV2 = NULL;
    int64_t Offset1 = 0;
    int64_t Offset2 = 0;
    bool isGA1 = isGAPlusOffset(Loc.Val, GV1, Offset1);
    bool isGA2 = isGAPlusOffset(BaseLoc.Val, GV2, Offset2);
    if (isGA1 && isGA2 && GV1 == GV2)
      return Offset1 == (Offset2 + Dist*Size);
  }

  return false;
}

static bool isBaseAlignmentOfN(unsigned N, SDNode *Base, MachineFrameInfo *MFI,
                               const X86Subtarget *Subtarget) {
  GlobalValue *GV;
  int64_t Offset = 0;
  if (isGAPlusOffset(Base, GV, Offset))
    return (GV->getAlignment() >= N && (Offset % N) == 0);
  // DAG combine handles the stack object case.
  return false;
}

static bool EltsFromConsecutiveLoads(SDNode *N, SDOperand PermMask,
                                     unsigned NumElems, MVT::ValueType EVT,
                                     MachineFrameInfo *MFI,
                                     SelectionDAG &DAG, SDNode *&Base) {
  Base = NULL;
  for (unsigned i = 0; i < NumElems; ++i) {
    SDOperand Idx = PermMask.getOperand(i);
    if (Idx.getOpcode() == ISD::UNDEF) {
      if (!Base)
        return false;
      continue;
    }

    unsigned Index = cast<ConstantSDNode>(Idx)->getValue();
    SDOperand Elt = getShuffleScalarElt(N, Index, DAG);
    if (!Elt.Val ||
        (Elt.getOpcode() != ISD::UNDEF && !ISD::isNON_EXTLoad(Elt.Val)))
      return false;
    if (!Base) {
      Base = Elt.Val;
      if (Base->getOpcode() == ISD::UNDEF)
        return false;
      continue;
    }
    if (Elt.getOpcode() == ISD::UNDEF)
      continue;

    if (!isConsecutiveLoad(Elt.Val, Base, i, MVT::getSizeInBits(EVT)/8,MFI))
      return false;
  }
  return true;
}

/// PerformShuffleCombine - Combine a vector_shuffle that is equal to
/// build_vector load1, load2, load3, load4, <0, 1, 2, 3> into a 128-bit load
/// if the load addresses are consecutive, non-overlapping, and in the right
/// order.
static SDOperand PerformShuffleCombine(SDNode *N, SelectionDAG &DAG,
                                       const X86Subtarget *Subtarget) {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = MVT::getVectorElementType(VT);
  SDOperand PermMask = N->getOperand(2);
  unsigned NumElems = PermMask.getNumOperands();
  SDNode *Base = NULL;
  if (!EltsFromConsecutiveLoads(N, PermMask, NumElems, EVT, MFI, DAG, Base))
    return SDOperand();

  LoadSDNode *LD = cast<LoadSDNode>(Base);
  if (isBaseAlignmentOfN(16, Base->getOperand(1).Val, MFI, Subtarget))
    return DAG.getLoad(VT, LD->getChain(), LD->getBasePtr(), LD->getSrcValue(),
                       LD->getSrcValueOffset(), LD->isVolatile());
  return DAG.getLoad(VT, LD->getChain(), LD->getBasePtr(), LD->getSrcValue(),
                     LD->getSrcValueOffset(), LD->isVolatile(),
                     LD->getAlignment());
}

static SDNode *getBuildPairElt(SDNode *N, unsigned i) {
  SDOperand Elt = N->getOperand(i);
  if (Elt.getOpcode() != ISD::MERGE_VALUES)
    return Elt.Val;
  return Elt.getOperand(Elt.ResNo).Val;
}

static SDOperand PerformBuildVectorCombine(SDNode *N, SelectionDAG &DAG,
                                       const X86Subtarget *Subtarget) {
  // Ignore single operand BUILD_VECTOR.
  if (N->getNumOperands() == 1)
    return SDOperand();

  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = MVT::getVectorElementType(VT);
  if ((EVT != MVT::i64 && EVT != MVT::f64) || Subtarget->is64Bit())
    // We are looking for load i64 and zero extend. We want to transform
    // it before legalizer has a chance to expand it. Also look for i64
    // BUILD_PAIR bit casted to f64.
    return SDOperand();
  // This must be an insertion into a zero vector.
  SDOperand HighElt = N->getOperand(1);
  if (!isZeroNode(HighElt))
    return SDOperand();

  // Value must be a load.
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  SDNode *Base = N->getOperand(0).Val;
  if (!isa<LoadSDNode>(Base)) {
    if (Base->getOpcode() == ISD::BIT_CONVERT)
      Base = Base->getOperand(0).Val;
    if (Base->getOpcode() != ISD::BUILD_PAIR)
      return SDOperand();
    SDNode *Pair = Base;
    Base = getBuildPairElt(Pair, 0);
    if (!ISD::isNON_EXTLoad(Base))
      return SDOperand();
    SDNode *NextLD = getBuildPairElt(Pair, 1);
    if (!ISD::isNON_EXTLoad(NextLD) ||
        !isConsecutiveLoad(NextLD, Base, 1, 4/*32 bits*/, MFI))
      return SDOperand();
  }
  LoadSDNode *LD = cast<LoadSDNode>(Base);

  // Transform it into VZEXT_LOAD addr.
  return DAG.getNode(X86ISD::VZEXT_LOAD, VT, LD->getChain(), LD->getBasePtr());
}                                           

/// PerformSELECTCombine - Do target-specific dag combines on SELECT nodes.
static SDOperand PerformSELECTCombine(SDNode *N, SelectionDAG &DAG,
                                      const X86Subtarget *Subtarget) {
  SDOperand Cond = N->getOperand(0);

  // If we have SSE[12] support, try to form min/max nodes.
  if (Subtarget->hasSSE2() &&
      (N->getValueType(0) == MVT::f32 || N->getValueType(0) == MVT::f64)) {
    if (Cond.getOpcode() == ISD::SETCC) {
      // Get the LHS/RHS of the select.
      SDOperand LHS = N->getOperand(1);
      SDOperand RHS = N->getOperand(2);
      ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();

      unsigned Opcode = 0;
      if (LHS == Cond.getOperand(0) && RHS == Cond.getOperand(1)) {
        switch (CC) {
        default: break;
        case ISD::SETOLE: // (X <= Y) ? X : Y -> min
        case ISD::SETULE:
        case ISD::SETLE:
          if (!UnsafeFPMath) break;
          // FALL THROUGH.
        case ISD::SETOLT:  // (X olt/lt Y) ? X : Y -> min
        case ISD::SETLT:
          Opcode = X86ISD::FMIN;
          break;

        case ISD::SETOGT: // (X > Y) ? X : Y -> max
        case ISD::SETUGT:
        case ISD::SETGT:
          if (!UnsafeFPMath) break;
          // FALL THROUGH.
        case ISD::SETUGE:  // (X uge/ge Y) ? X : Y -> max
        case ISD::SETGE:
          Opcode = X86ISD::FMAX;
          break;
        }
      } else if (LHS == Cond.getOperand(1) && RHS == Cond.getOperand(0)) {
        switch (CC) {
        default: break;
        case ISD::SETOGT: // (X > Y) ? Y : X -> min
        case ISD::SETUGT:
        case ISD::SETGT:
          if (!UnsafeFPMath) break;
          // FALL THROUGH.
        case ISD::SETUGE:  // (X uge/ge Y) ? Y : X -> min
        case ISD::SETGE:
          Opcode = X86ISD::FMIN;
          break;

        case ISD::SETOLE:   // (X <= Y) ? Y : X -> max
        case ISD::SETULE:
        case ISD::SETLE:
          if (!UnsafeFPMath) break;
          // FALL THROUGH.
        case ISD::SETOLT:   // (X olt/lt Y) ? Y : X -> max
        case ISD::SETLT:
          Opcode = X86ISD::FMAX;
          break;
        }
      }

      if (Opcode)
        return DAG.getNode(Opcode, N->getValueType(0), LHS, RHS);
    }

  }

  return SDOperand();
}

/// PerformSTORECombine - Do target-specific dag combines on STORE nodes.
static SDOperand PerformSTORECombine(SDNode *N, SelectionDAG &DAG,
                                     const X86Subtarget *Subtarget) {
  // Turn load->store of MMX types into GPR load/stores.  This avoids clobbering
  // the FP state in cases where an emms may be missing.
  // A preferable solution to the general problem is to figure out the right
  // places to insert EMMS.  This qualifies as a quick hack.
  StoreSDNode *St = cast<StoreSDNode>(N);
  if (MVT::isVector(St->getValue().getValueType()) && 
      MVT::getSizeInBits(St->getValue().getValueType()) == 64 &&
      isa<LoadSDNode>(St->getValue()) &&
      !cast<LoadSDNode>(St->getValue())->isVolatile() &&
      St->getChain().hasOneUse() && !St->isVolatile()) {
    SDNode* LdVal = St->getValue().Val;
    LoadSDNode *Ld = 0;
    int TokenFactorIndex = -1;
    SmallVector<SDOperand, 8> Ops;
    SDNode* ChainVal = St->getChain().Val;
    // Must be a store of a load.  We currently handle two cases:  the load
    // is a direct child, and it's under an intervening TokenFactor.  It is
    // possible to dig deeper under nested TokenFactors.
    if (ChainVal == LdVal)
      Ld = cast<LoadSDNode>(St->getChain());
    else if (St->getValue().hasOneUse() &&
             ChainVal->getOpcode() == ISD::TokenFactor) {
      for (unsigned i=0, e = ChainVal->getNumOperands(); i != e; ++i) {
        if (ChainVal->getOperand(i).Val == LdVal) {
          TokenFactorIndex = i;
          Ld = cast<LoadSDNode>(St->getValue());
        } else
          Ops.push_back(ChainVal->getOperand(i));
      }
    }
    if (Ld) {
      // If we are a 64-bit capable x86, lower to a single movq load/store pair.
      if (Subtarget->is64Bit()) {
        SDOperand NewLd = DAG.getLoad(MVT::i64, Ld->getChain(), 
                                      Ld->getBasePtr(), Ld->getSrcValue(), 
                                      Ld->getSrcValueOffset(), Ld->isVolatile(),
                                      Ld->getAlignment());
        SDOperand NewChain = NewLd.getValue(1);
        if (TokenFactorIndex != -1) {
          Ops.push_back(NewChain);
          NewChain = DAG.getNode(ISD::TokenFactor, MVT::Other, &Ops[0], 
                                 Ops.size());
        }
        return DAG.getStore(NewChain, NewLd, St->getBasePtr(),
                            St->getSrcValue(), St->getSrcValueOffset(),
                            St->isVolatile(), St->getAlignment());
      }

      // Otherwise, lower to two 32-bit copies.
      SDOperand LoAddr = Ld->getBasePtr();
      SDOperand HiAddr = DAG.getNode(ISD::ADD, MVT::i32, LoAddr,
                                     DAG.getConstant(MVT::i32, 4));

      SDOperand LoLd = DAG.getLoad(MVT::i32, Ld->getChain(), LoAddr,
                                   Ld->getSrcValue(), Ld->getSrcValueOffset(),
                                   Ld->isVolatile(), Ld->getAlignment());
      SDOperand HiLd = DAG.getLoad(MVT::i32, Ld->getChain(), HiAddr,
                                   Ld->getSrcValue(), Ld->getSrcValueOffset()+4,
                                   Ld->isVolatile(), 
                                   MinAlign(Ld->getAlignment(), 4));

      SDOperand NewChain = LoLd.getValue(1);
      if (TokenFactorIndex != -1) {
        Ops.push_back(LoLd);
        Ops.push_back(HiLd);
        NewChain = DAG.getNode(ISD::TokenFactor, MVT::Other, &Ops[0], 
                               Ops.size());
      }

      LoAddr = St->getBasePtr();
      HiAddr = DAG.getNode(ISD::ADD, MVT::i32, LoAddr,
                           DAG.getConstant(MVT::i32, 4));

      SDOperand LoSt = DAG.getStore(NewChain, LoLd, LoAddr,
                          St->getSrcValue(), St->getSrcValueOffset(),
                          St->isVolatile(), St->getAlignment());
      SDOperand HiSt = DAG.getStore(NewChain, HiLd, HiAddr,
                                    St->getSrcValue(), St->getSrcValueOffset()+4,
                                    St->isVolatile(), 
                                    MinAlign(St->getAlignment(), 4));
      return DAG.getNode(ISD::TokenFactor, MVT::Other, LoSt, HiSt);
    }
  }
  return SDOperand();
}

/// PerformFORCombine - Do target-specific dag combines on X86ISD::FOR and
/// X86ISD::FXOR nodes.
static SDOperand PerformFORCombine(SDNode *N, SelectionDAG &DAG) {
  assert(N->getOpcode() == X86ISD::FOR || N->getOpcode() == X86ISD::FXOR);
  // F[X]OR(0.0, x) -> x
  // F[X]OR(x, 0.0) -> x
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N->getOperand(0)))
    if (C->getValueAPF().isPosZero())
      return N->getOperand(1);
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N->getOperand(1)))
    if (C->getValueAPF().isPosZero())
      return N->getOperand(0);
  return SDOperand();
}

/// PerformFANDCombine - Do target-specific dag combines on X86ISD::FAND nodes.
static SDOperand PerformFANDCombine(SDNode *N, SelectionDAG &DAG) {
  // FAND(0.0, x) -> 0.0
  // FAND(x, 0.0) -> 0.0
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N->getOperand(0)))
    if (C->getValueAPF().isPosZero())
      return N->getOperand(0);
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N->getOperand(1)))
    if (C->getValueAPF().isPosZero())
      return N->getOperand(1);
  return SDOperand();
}


SDOperand X86TargetLowering::PerformDAGCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default: break;
  case ISD::VECTOR_SHUFFLE: return PerformShuffleCombine(N, DAG, Subtarget);
  case ISD::BUILD_VECTOR:   return PerformBuildVectorCombine(N, DAG, Subtarget);
  case ISD::SELECT:         return PerformSELECTCombine(N, DAG, Subtarget);
  case ISD::STORE:          return PerformSTORECombine(N, DAG, Subtarget);
  case X86ISD::FXOR:
  case X86ISD::FOR:         return PerformFORCombine(N, DAG);
  case X86ISD::FAND:        return PerformFANDCombine(N, DAG);
  }

  return SDOperand();
}

//===----------------------------------------------------------------------===//
//                           X86 Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
X86TargetLowering::ConstraintType
X86TargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'A':
    case 'f':
    case 'r':
    case 'R':
    case 'l':
    case 'q':
    case 'Q':
    case 'x':
    case 'y':
    case 'Y':
      return C_RegisterClass;
    default:
      break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

/// LowerXConstraint - try to replace an X constraint, which matches anything,
/// with another that has more specific requirements based on the type of the
/// corresponding operand.
const char *X86TargetLowering::
LowerXConstraint(MVT::ValueType ConstraintVT) const {
  // FP X constraints get lowered to SSE1/2 registers if available, otherwise
  // 'f' like normal targets.
  if (MVT::isFloatingPoint(ConstraintVT)) {
    if (Subtarget->hasSSE2())
      return "Y";
    if (Subtarget->hasSSE1())
      return "x";
  }
  
  return TargetLowering::LowerXConstraint(ConstraintVT);
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void X86TargetLowering::LowerAsmOperandForConstraint(SDOperand Op,
                                                     char Constraint,
                                                     std::vector<SDOperand>&Ops,
                                                     SelectionDAG &DAG) const {
  SDOperand Result(0, 0);
  
  switch (Constraint) {
  default: break;
  case 'I':
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      if (C->getValue() <= 31) {
        Result = DAG.getTargetConstant(C->getValue(), Op.getValueType());
        break;
      }
    }
    return;
  case 'N':
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
      if (C->getValue() <= 255) {
        Result = DAG.getTargetConstant(C->getValue(), Op.getValueType());
        break;
      }
    }
    return;
  case 'i': {
    // Literal immediates are always ok.
    if (ConstantSDNode *CST = dyn_cast<ConstantSDNode>(Op)) {
      Result = DAG.getTargetConstant(CST->getValue(), Op.getValueType());
      break;
    }

    // If we are in non-pic codegen mode, we allow the address of a global (with
    // an optional displacement) to be used with 'i'.
    GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op);
    int64_t Offset = 0;
    
    // Match either (GA) or (GA+C)
    if (GA) {
      Offset = GA->getOffset();
    } else if (Op.getOpcode() == ISD::ADD) {
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
      GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(0));
      if (C && GA) {
        Offset = GA->getOffset()+C->getValue();
      } else {
        C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
        GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(0));
        if (C && GA)
          Offset = GA->getOffset()+C->getValue();
        else
          C = 0, GA = 0;
      }
    }
    
    if (GA) {
      // If addressing this global requires a load (e.g. in PIC mode), we can't
      // match.
      if (Subtarget->GVRequiresExtraLoad(GA->getGlobal(), getTargetMachine(),
                                         false))
        return;

      Op = DAG.getTargetGlobalAddress(GA->getGlobal(), GA->getValueType(0),
                                      Offset);
      Result = Op;
      break;
    }

    // Otherwise, not valid for this mode.
    return;
  }
  }
  
  if (Result.Val) {
    Ops.push_back(Result);
    return;
  }
  return TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

std::vector<unsigned> X86TargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    // FIXME: not handling fp-stack yet!
    switch (Constraint[0]) {      // GCC X86 Constraint Letters
    default: break;  // Unknown constraint letter
    case 'A':   // EAX/EDX
      if (VT == MVT::i32 || VT == MVT::i64)
        return make_vector<unsigned>(X86::EAX, X86::EDX, 0);
      break;
    case 'q':   // Q_REGS (GENERAL_REGS in 64-bit mode)
    case 'Q':   // Q_REGS
      if (VT == MVT::i32)
        return make_vector<unsigned>(X86::EAX, X86::EDX, X86::ECX, X86::EBX, 0);
      else if (VT == MVT::i16)
        return make_vector<unsigned>(X86::AX, X86::DX, X86::CX, X86::BX, 0);
      else if (VT == MVT::i8)
        return make_vector<unsigned>(X86::AL, X86::DL, X86::CL, X86::BL, 0);
      else if (VT == MVT::i64)
        return make_vector<unsigned>(X86::RAX, X86::RDX, X86::RCX, X86::RBX, 0);
      break;
    }
  }

  return std::vector<unsigned>();
}

std::pair<unsigned, const TargetRegisterClass*>
X86TargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                MVT::ValueType VT) const {
  // First, see if this is a constraint that directly corresponds to an LLVM
  // register class.
  if (Constraint.size() == 1) {
    // GCC Constraint Letters
    switch (Constraint[0]) {
    default: break;
    case 'r':   // GENERAL_REGS
    case 'R':   // LEGACY_REGS
    case 'l':   // INDEX_REGS
      if (VT == MVT::i64 && Subtarget->is64Bit())
        return std::make_pair(0U, X86::GR64RegisterClass);
      if (VT == MVT::i32)
        return std::make_pair(0U, X86::GR32RegisterClass);
      else if (VT == MVT::i16)
        return std::make_pair(0U, X86::GR16RegisterClass);
      else if (VT == MVT::i8)
        return std::make_pair(0U, X86::GR8RegisterClass);
      break;
    case 'f':  // FP Stack registers.
      // If SSE is enabled for this VT, use f80 to ensure the isel moves the
      // value to the correct fpstack register class.
      if (VT == MVT::f32 && !isScalarFPTypeInSSEReg(VT))
        return std::make_pair(0U, X86::RFP32RegisterClass);
      if (VT == MVT::f64 && !isScalarFPTypeInSSEReg(VT))
        return std::make_pair(0U, X86::RFP64RegisterClass);
      return std::make_pair(0U, X86::RFP80RegisterClass);
    case 'y':   // MMX_REGS if MMX allowed.
      if (!Subtarget->hasMMX()) break;
      return std::make_pair(0U, X86::VR64RegisterClass);
      break;
    case 'Y':   // SSE_REGS if SSE2 allowed
      if (!Subtarget->hasSSE2()) break;
      // FALL THROUGH.
    case 'x':   // SSE_REGS if SSE1 allowed
      if (!Subtarget->hasSSE1()) break;
      
      switch (VT) {
      default: break;
      // Scalar SSE types.
      case MVT::f32:
      case MVT::i32:
        return std::make_pair(0U, X86::FR32RegisterClass);
      case MVT::f64:
      case MVT::i64:
        return std::make_pair(0U, X86::FR64RegisterClass);
      // Vector types.
      case MVT::v16i8:
      case MVT::v8i16:
      case MVT::v4i32:
      case MVT::v2i64:
      case MVT::v4f32:
      case MVT::v2f64:
        return std::make_pair(0U, X86::VR128RegisterClass);
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
    // GCC calls "st(0)" just plain "st".
    if (StringsEqualNoCase("{st}", Constraint)) {
      Res.first = X86::ST0;
      Res.second = X86::RFP80RegisterClass;
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
  if (Res.second != X86::GR16RegisterClass)
    return Res;

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
      Res.second = Res.second = X86::GR8RegisterClass;
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
      Res.second = Res.second = X86::GR32RegisterClass;
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
      Res.second = Res.second = X86::GR64RegisterClass;
    }
  }

  return Res;
}
