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
#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86ISelLowering.h"
#include "X86ShuffleDecode.h"
#include "X86TargetMachine.h"
#include "X86TargetObjectFile.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalAlias.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace dwarf;

STATISTIC(NumTailCalls, "Number of tail calls");

static cl::opt<bool>
DisableMMX("disable-mmx", cl::Hidden, cl::desc("Disable use of MMX"));

// Forward declarations.
static SDValue getMOVL(SelectionDAG &DAG, DebugLoc dl, EVT VT, SDValue V1,
                       SDValue V2);

static TargetLoweringObjectFile *createTLOF(X86TargetMachine &TM) {
  
  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();
  
  if (TM.getSubtarget<X86Subtarget>().isTargetDarwin()) {
    if (is64Bit) return new X8664_MachoTargetObjectFile();
    return new TargetLoweringObjectFileMachO();
  } else if (TM.getSubtarget<X86Subtarget>().isTargetELF() ){
    if (is64Bit) return new X8664_ELFTargetObjectFile(TM);
    return new X8632_ELFTargetObjectFile(TM);
  } else if (TM.getSubtarget<X86Subtarget>().isTargetCOFF()) {
    return new TargetLoweringObjectFileCOFF();
  }  
  llvm_unreachable("unknown subtarget type");
}

X86TargetLowering::X86TargetLowering(X86TargetMachine &TM)
  : TargetLowering(TM, createTLOF(TM)) {
  Subtarget = &TM.getSubtarget<X86Subtarget>();
  X86ScalarSSEf64 = Subtarget->hasSSE2();
  X86ScalarSSEf32 = Subtarget->hasSSE1();
  X86StackPtr = Subtarget->is64Bit() ? X86::RSP : X86::ESP;

  RegInfo = TM.getRegisterInfo();
  TD = getTargetData();

  // Set up the TargetLowering object.

  // X86 is weird, it always uses i8 for shift amounts and setcc results.
  setShiftAmountType(MVT::i8);
  setBooleanContents(ZeroOrOneBooleanContent);
  setSchedulingPreference(Sched::RegPressure);
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
    setOperationAction(ISD::UINT_TO_FP     , MVT::i64  , Expand);
  } else if (!UseSoftFloat) {
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

  if (!UseSoftFloat) {
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
  } else if (!UseSoftFloat) {
    if (X86ScalarSSEf32 && !Subtarget->hasSSE3())
      // Expand FP_TO_UINT into a select.
      // FIXME: We would like to use a Custom expander here eventually to do
      // the optimal thing for SSE vs. the default expansion in the legalizer.
      setOperationAction(ISD::FP_TO_UINT   , MVT::i32  , Expand);
    else
      // With SSE3 we can use fisttpll to convert to a signed i64; without
      // SSE, we're stuck with a fistpll.
      setOperationAction(ISD::FP_TO_UINT   , MVT::i32  , Custom);
  }

  // TODO: when we have SSE, these could be more efficient, by using movd/movq.
  if (!X86ScalarSSEf64) { 
    setOperationAction(ISD::BIT_CONVERT      , MVT::f32  , Expand);
    setOperationAction(ISD::BIT_CONVERT      , MVT::i32  , Expand);
    if (Subtarget->is64Bit()) {
      setOperationAction(ISD::BIT_CONVERT    , MVT::f64  , Expand);
      // Without SSE, i64->f64 goes through memory; i64->MMX is Legal.
      if (Subtarget->hasMMX() && !DisableMMX)
        setOperationAction(ISD::BIT_CONVERT    , MVT::i64  , Custom);
      else 
        setOperationAction(ISD::BIT_CONVERT    , MVT::i64  , Expand);
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
  setOperationAction(ISD::SELECT          , MVT::i1   , Promote);
  // X86 wants to expand cmov itself.
  setOperationAction(ISD::SELECT          , MVT::i8   , Custom);
  setOperationAction(ISD::SELECT        , MVT::i16  , Custom);
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

  // We may not have a libcall for MEMBARRIER so we should lower this.
  setOperationAction(ISD::MEMBARRIER    , MVT::Other, Custom);
  
  // On X86 and X86-64, atomic operations are lowered to locked instructions.
  // Locked instructions, in turn, have implicit fence semantics (all memory
  // operations are flushed before issuing the locked instruction, and they
  // are not buffered), so we can fold away the common pattern of
  // fence-atomic-fence.
  setShouldFoldAtomicFences(true);

  // Expand certain atomics
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i8, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i16, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i64, Custom);

  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i8, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i16, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i64, Custom);

  if (!Subtarget->is64Bit()) {
    setOperationAction(ISD::ATOMIC_LOAD_ADD, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_AND, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_OR, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_XOR, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_LOAD_NAND, MVT::i64, Custom);
    setOperationAction(ISD::ATOMIC_SWAP, MVT::i64, Custom);
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

  if (!UseSoftFloat && X86ScalarSSEf64) {
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
  } else if (!UseSoftFloat && X86ScalarSSEf32) {
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

    if (!UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f64  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f64  , Expand);
    }
  } else if (!UseSoftFloat) {
    // f32 and f64 in x87.
    // Set up the FP register classes.
    addRegisterClass(MVT::f64, X86::RFP64RegisterClass);
    addRegisterClass(MVT::f32, X86::RFP32RegisterClass);

    setOperationAction(ISD::UNDEF,     MVT::f64, Expand);
    setOperationAction(ISD::UNDEF,     MVT::f32, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

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
  if (!UseSoftFloat) {
    addRegisterClass(MVT::f80, X86::RFP80RegisterClass);
    setOperationAction(ISD::UNDEF,     MVT::f80, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f80, Expand);
    {
      bool ignored;
      APFloat TmpFlt(+0.0);
      TmpFlt.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
                     &ignored);
      addLegalFPImmediate(TmpFlt);  // FLD0
      TmpFlt.changeSign();
      addLegalFPImmediate(TmpFlt);  // FLD0/FCHS
      APFloat TmpFlt2(+1.0);
      TmpFlt2.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
                      &ignored);
      addLegalFPImmediate(TmpFlt2);  // FLD1
      TmpFlt2.changeSign();
      addLegalFPImmediate(TmpFlt2);  // FLD1/FCHS
    }

    if (!UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f80  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f80  , Expand);
    }
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
  for (unsigned VT = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
       VT <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++VT) {
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
    setOperationAction(ISD::EXTRACT_SUBVECTOR,(MVT::SimpleValueType)VT,Expand);
    setOperationAction(ISD::INSERT_VECTOR_ELT,(MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FABS, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FSIN, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FCOS, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FREM, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FPOWI, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FSQRT, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FCOPYSIGN, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SDIVREM, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::UDIVREM, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::FPOW, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CTPOP, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CTTZ, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CTLZ, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SHL, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SRA, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::SRL, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::ROTL, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::ROTR, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::BSWAP, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::VSETCC, (MVT::SimpleValueType)VT, Expand);
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
    for (unsigned InnerVT = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
         InnerVT <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++InnerVT)
      setTruncStoreAction((MVT::SimpleValueType)VT,
                          (MVT::SimpleValueType)InnerVT, Expand);
    setLoadExtAction(ISD::SEXTLOAD, (MVT::SimpleValueType)VT, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, (MVT::SimpleValueType)VT, Expand);
    setLoadExtAction(ISD::EXTLOAD, (MVT::SimpleValueType)VT, Expand);
  }

  // FIXME: In order to prevent SSE instructions being expanded to MMX ones
  // with -msoft-float, disable use of MMX as well.
  if (!UseSoftFloat && !DisableMMX && Subtarget->hasMMX()) {
    addRegisterClass(MVT::v8i8,  X86::VR64RegisterClass, false);
    addRegisterClass(MVT::v4i16, X86::VR64RegisterClass, false);
    addRegisterClass(MVT::v2i32, X86::VR64RegisterClass, false);
    
    addRegisterClass(MVT::v1i64, X86::VR64RegisterClass, false);

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

    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4i16, Custom);

    setOperationAction(ISD::SELECT,             MVT::v8i8, Promote);
    setOperationAction(ISD::SELECT,             MVT::v4i16, Promote);
    setOperationAction(ISD::SELECT,             MVT::v2i32, Promote);
    setOperationAction(ISD::SELECT,             MVT::v1i64, Custom);
    setOperationAction(ISD::VSETCC,             MVT::v8i8, Custom);
    setOperationAction(ISD::VSETCC,             MVT::v4i16, Custom);
    setOperationAction(ISD::VSETCC,             MVT::v2i32, Custom);

    if (!X86ScalarSSEf64 && Subtarget->is64Bit()) {
      setOperationAction(ISD::BIT_CONVERT,        MVT::v8i8,  Custom);
      setOperationAction(ISD::BIT_CONVERT,        MVT::v4i16, Custom);
      setOperationAction(ISD::BIT_CONVERT,        MVT::v2i32, Custom);
      setOperationAction(ISD::BIT_CONVERT,        MVT::v1i64, Custom);
    }
  }

  if (!UseSoftFloat && Subtarget->hasSSE1()) {
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
    setOperationAction(ISD::VSETCC,             MVT::v4f32, Custom);
  }

  if (!UseSoftFloat && Subtarget->hasSSE2()) {
    addRegisterClass(MVT::v2f64, X86::VR128RegisterClass);

    // FIXME: Unfortunately -soft-float and -no-implicit-float means XMM
    // registers cannot be used even for integer operations.
    addRegisterClass(MVT::v16i8, X86::VR128RegisterClass);
    addRegisterClass(MVT::v8i16, X86::VR128RegisterClass);
    addRegisterClass(MVT::v4i32, X86::VR128RegisterClass);
    addRegisterClass(MVT::v2i64, X86::VR128RegisterClass);

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

    setOperationAction(ISD::VSETCC,             MVT::v2f64, Custom);
    setOperationAction(ISD::VSETCC,             MVT::v16i8, Custom);
    setOperationAction(ISD::VSETCC,             MVT::v8i16, Custom);
    setOperationAction(ISD::VSETCC,             MVT::v4i32, Custom);

    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v16i8, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4i32, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4f32, Custom);

    setOperationAction(ISD::CONCAT_VECTORS,     MVT::v2f64, Custom);
    setOperationAction(ISD::CONCAT_VECTORS,     MVT::v2i64, Custom);
    setOperationAction(ISD::CONCAT_VECTORS,     MVT::v16i8, Custom);
    setOperationAction(ISD::CONCAT_VECTORS,     MVT::v8i16, Custom);
    setOperationAction(ISD::CONCAT_VECTORS,     MVT::v4i32, Custom);

    // Custom lower build_vector, vector_shuffle, and extract_vector_elt.
    for (unsigned i = (unsigned)MVT::v16i8; i != (unsigned)MVT::v2i64; ++i) {
      EVT VT = (MVT::SimpleValueType)i;
      // Do not attempt to custom lower non-power-of-2 vectors
      if (!isPowerOf2_32(VT.getVectorNumElements()))
        continue;
      // Do not attempt to custom lower non-128-bit vectors
      if (!VT.is128BitVector())
        continue;
      setOperationAction(ISD::BUILD_VECTOR,
                         VT.getSimpleVT().SimpleTy, Custom);
      setOperationAction(ISD::VECTOR_SHUFFLE,
                         VT.getSimpleVT().SimpleTy, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT,
                         VT.getSimpleVT().SimpleTy, Custom);
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
    for (unsigned i = (unsigned)MVT::v16i8; i != (unsigned)MVT::v2i64; i++) {
      MVT::SimpleValueType SVT = (MVT::SimpleValueType)i;
      EVT VT = SVT;

      // Do not attempt to promote non-128-bit vectors
      if (!VT.is128BitVector())
        continue;
      
      setOperationAction(ISD::AND,    SVT, Promote);
      AddPromotedToType (ISD::AND,    SVT, MVT::v2i64);
      setOperationAction(ISD::OR,     SVT, Promote);
      AddPromotedToType (ISD::OR,     SVT, MVT::v2i64);
      setOperationAction(ISD::XOR,    SVT, Promote);
      AddPromotedToType (ISD::XOR,    SVT, MVT::v2i64);
      setOperationAction(ISD::LOAD,   SVT, Promote);
      AddPromotedToType (ISD::LOAD,   SVT, MVT::v2i64);
      setOperationAction(ISD::SELECT, SVT, Promote);
      AddPromotedToType (ISD::SELECT, SVT, MVT::v2i64);
    }

    setTruncStoreAction(MVT::f64, MVT::f32, Expand);

    // Custom lower v2i64 and v2f64 selects.
    setOperationAction(ISD::LOAD,               MVT::v2f64, Legal);
    setOperationAction(ISD::LOAD,               MVT::v2i64, Legal);
    setOperationAction(ISD::SELECT,             MVT::v2f64, Custom);
    setOperationAction(ISD::SELECT,             MVT::v2i64, Custom);

    setOperationAction(ISD::FP_TO_SINT,         MVT::v4i32, Legal);
    setOperationAction(ISD::SINT_TO_FP,         MVT::v4i32, Legal);
    if (!DisableMMX && Subtarget->hasMMX()) {
      setOperationAction(ISD::FP_TO_SINT,         MVT::v2i32, Custom);
      setOperationAction(ISD::SINT_TO_FP,         MVT::v2i32, Custom);
    }
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

    // FIXME: Do we need to handle scalar-to-vector here?
    setOperationAction(ISD::MUL,                MVT::v4i32, Legal);

    // Can turn SHL into an integer multiply.
    setOperationAction(ISD::SHL,                MVT::v4i32, Custom);
    setOperationAction(ISD::SHL,                MVT::v16i8, Custom);

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

    if (Subtarget->is64Bit()) {
      setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v2i64, Legal);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i64, Legal);
    }
  }

  if (Subtarget->hasSSE42()) {
    setOperationAction(ISD::VSETCC,             MVT::v2i64, Custom);
  }

  if (!UseSoftFloat && Subtarget->hasAVX()) {
    addRegisterClass(MVT::v8f32, X86::VR256RegisterClass);
    addRegisterClass(MVT::v4f64, X86::VR256RegisterClass);
    addRegisterClass(MVT::v8i32, X86::VR256RegisterClass);
    addRegisterClass(MVT::v4i64, X86::VR256RegisterClass);
    addRegisterClass(MVT::v32i8, X86::VR256RegisterClass);

    setOperationAction(ISD::LOAD,               MVT::v8f32, Legal);
    setOperationAction(ISD::LOAD,               MVT::v8i32, Legal);
    setOperationAction(ISD::LOAD,               MVT::v4f64, Legal);
    setOperationAction(ISD::LOAD,               MVT::v4i64, Legal);
    setOperationAction(ISD::FADD,               MVT::v8f32, Legal);
    setOperationAction(ISD::FSUB,               MVT::v8f32, Legal);
    setOperationAction(ISD::FMUL,               MVT::v8f32, Legal);
    setOperationAction(ISD::FDIV,               MVT::v8f32, Legal);
    setOperationAction(ISD::FSQRT,              MVT::v8f32, Legal);
    setOperationAction(ISD::FNEG,               MVT::v8f32, Custom);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v8f32, Custom);
    //setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v8f32, Custom);
    //setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v8f32, Custom);
    //setOperationAction(ISD::SELECT,             MVT::v8f32, Custom);
    //setOperationAction(ISD::VSETCC,             MVT::v8f32, Custom);

    // Operations to consider commented out -v16i16 v32i8
    //setOperationAction(ISD::ADD,                MVT::v16i16, Legal);
    setOperationAction(ISD::ADD,                MVT::v8i32, Custom);
    setOperationAction(ISD::ADD,                MVT::v4i64, Custom);
    //setOperationAction(ISD::SUB,                MVT::v32i8, Legal);
    //setOperationAction(ISD::SUB,                MVT::v16i16, Legal);
    setOperationAction(ISD::SUB,                MVT::v8i32, Custom);
    setOperationAction(ISD::SUB,                MVT::v4i64, Custom);
    //setOperationAction(ISD::MUL,                MVT::v16i16, Legal);
    setOperationAction(ISD::FADD,               MVT::v4f64, Legal);
    setOperationAction(ISD::FSUB,               MVT::v4f64, Legal);
    setOperationAction(ISD::FMUL,               MVT::v4f64, Legal);
    setOperationAction(ISD::FDIV,               MVT::v4f64, Legal);
    setOperationAction(ISD::FSQRT,              MVT::v4f64, Legal);
    setOperationAction(ISD::FNEG,               MVT::v4f64, Custom);

    setOperationAction(ISD::VSETCC,             MVT::v4f64, Custom);
    // setOperationAction(ISD::VSETCC,             MVT::v32i8, Custom);
    // setOperationAction(ISD::VSETCC,             MVT::v16i16, Custom);
    setOperationAction(ISD::VSETCC,             MVT::v8i32, Custom);

    // setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v32i8, Custom);
    // setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v16i16, Custom);
    // setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v16i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v8i32, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v8f32, Custom);

    setOperationAction(ISD::BUILD_VECTOR,       MVT::v4f64, Custom);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v4i64, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v4f64, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v4i64, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4f64, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4f64, Custom);

#if 0
    // Not sure we want to do this since there are no 256-bit integer
    // operations in AVX

    // Custom lower build_vector, vector_shuffle, and extract_vector_elt.
    // This includes 256-bit vectors
    for (unsigned i = (unsigned)MVT::v16i8; i != (unsigned)MVT::v4i64; ++i) {
      EVT VT = (MVT::SimpleValueType)i;

      // Do not attempt to custom lower non-power-of-2 vectors
      if (!isPowerOf2_32(VT.getVectorNumElements()))
        continue;

      setOperationAction(ISD::BUILD_VECTOR,       VT, Custom);
      setOperationAction(ISD::VECTOR_SHUFFLE,     VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
    }

    if (Subtarget->is64Bit()) {
      setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4i64, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i64, Custom);
    }
#endif

#if 0
    // Not sure we want to do this since there are no 256-bit integer
    // operations in AVX

    // Promote v32i8, v16i16, v8i32 load, select, and, or, xor to v4i64.
    // Including 256-bit vectors
    for (unsigned i = (unsigned)MVT::v16i8; i != (unsigned)MVT::v4i64; i++) {
      EVT VT = (MVT::SimpleValueType)i;

      if (!VT.is256BitVector()) {
        continue;
      }
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

    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
#endif
  }

  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  // Add/Sub/Mul with overflow operations are custom lowered.
  setOperationAction(ISD::SADDO, MVT::i32, Custom);
  setOperationAction(ISD::UADDO, MVT::i32, Custom);
  setOperationAction(ISD::SSUBO, MVT::i32, Custom);
  setOperationAction(ISD::USUBO, MVT::i32, Custom);
  setOperationAction(ISD::SMULO, MVT::i32, Custom);

  // Only custom-lower 64-bit SADDO and friends on 64-bit because we don't
  // handle type legalization for these operations here.
  //
  // FIXME: We really should do custom legalization for addition and
  // subtraction on x86-32 once PR3203 is fixed.  We really can't do much better
  // than generic legalization for 64-bit multiplication-with-overflow, though.
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::SADDO, MVT::i64, Custom);
    setOperationAction(ISD::UADDO, MVT::i64, Custom);
    setOperationAction(ISD::SSUBO, MVT::i64, Custom);
    setOperationAction(ISD::USUBO, MVT::i64, Custom);
    setOperationAction(ISD::SMULO, MVT::i64, Custom);
  }

  if (!Subtarget->is64Bit()) {
    // These libcalls are not available in 32-bit.
    setLibcallName(RTLIB::SHL_I128, 0);
    setLibcallName(RTLIB::SRL_I128, 0);
    setLibcallName(RTLIB::SRA_I128, 0);
  }

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::VECTOR_SHUFFLE);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT);
  setTargetDAGCombine(ISD::BUILD_VECTOR);
  setTargetDAGCombine(ISD::SELECT);
  setTargetDAGCombine(ISD::SHL);
  setTargetDAGCombine(ISD::SRA);
  setTargetDAGCombine(ISD::SRL);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  if (Subtarget->is64Bit())
    setTargetDAGCombine(ISD::MUL);

  computeRegisterProperties();

  // FIXME: These should be based on subtarget info. Plus, the values should
  // be smaller when we are in optimizing for size mode.
  maxStoresPerMemset = 16; // For @llvm.memset -> sequence of stores
  maxStoresPerMemcpy = 8; // For @llvm.memcpy -> sequence of stores
  maxStoresPerMemmove = 3; // For @llvm.memmove -> sequence of stores
  setPrefLoopAlignment(16);
  benefitFromCodePlacementOpt = true;
}


MVT::SimpleValueType X86TargetLowering::getSetCCResultType(EVT VT) const {
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
/// 'NonScalarIntSafe' is true, that means it's safe to return a
/// non-scalar-integer type, e.g. empty string source, constant, or loaded
/// from memory. 'MemcpyStrSrc' indicates whether the memcpy source is
/// constant so it does not need to be loaded.
/// It returns EVT::Other if the type should be determined using generic
/// target-independent logic.
EVT
X86TargetLowering::getOptimalMemOpType(uint64_t Size,
                                       unsigned DstAlign, unsigned SrcAlign,
                                       bool NonScalarIntSafe,
                                       bool MemcpyStrSrc,
                                       MachineFunction &MF) const {
  // FIXME: This turns off use of xmm stores for memset/memcpy on targets like
  // linux.  This is because the stack realignment code can't handle certain
  // cases like PR2962.  This should be removed when PR2962 is fixed.
  const Function *F = MF.getFunction();
  if (NonScalarIntSafe &&
      !F->hasFnAttr(Attribute::NoImplicitFloat)) {
    if (Size >= 16 &&
        (Subtarget->isUnalignedMemAccessFast() ||
         ((DstAlign == 0 || DstAlign >= 16) &&
          (SrcAlign == 0 || SrcAlign >= 16))) &&
        Subtarget->getStackAlignment() >= 16) {
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

/// getPICBaseSymbol - Return the X86-32 PIC base.
MCSymbol *
X86TargetLowering::getPICBaseSymbol(const MachineFunction *MF,
                                    MCContext &Ctx) const {
  const MCAsmInfo &MAI = *getTargetMachine().getMCAsmInfo();
  return Ctx.GetOrCreateSymbol(Twine(MAI.getPrivateGlobalPrefix())+
                               Twine(MF->getFunctionNumber())+"$pb");
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
  return MCSymbolRefExpr::Create(getPICBaseSymbol(MF, Ctx), Ctx);
}

/// getFunctionAlignment - Return the Log2 alignment of this function.
unsigned X86TargetLowering::getFunctionAlignment(const Function *F) const {
  return F->hasFnAttr(Attribute::OptimizeForSize) ? 0 : 4;
}

std::pair<const TargetRegisterClass*, uint8_t>
X86TargetLowering::findRepresentativeClass(EVT VT) const{
  const TargetRegisterClass *RRC = 0;
  uint8_t Cost = 1;
  switch (VT.getSimpleVT().SimpleTy) {
  default:
    return TargetLowering::findRepresentativeClass(VT);
  case MVT::i8: case MVT::i16: case MVT::i32: case MVT::i64:
    RRC = (Subtarget->is64Bit()
           ? X86::GR64RegisterClass : X86::GR32RegisterClass);
    break;
  case MVT::v8i8: case MVT::v4i16:
  case MVT::v2i32: case MVT::v1i64: 
    RRC = X86::VR64RegisterClass;
    break;
  case MVT::f32: case MVT::f64:
  case MVT::v16i8: case MVT::v8i16: case MVT::v4i32: case MVT::v2i64:
  case MVT::v4f32: case MVT::v2f64:
  case MVT::v32i8: case MVT::v8i32: case MVT::v4i64: case MVT::v8f32:
  case MVT::v4f64:
    RRC = X86::VR128RegisterClass;
    break;
  }
  return std::make_pair(RRC, Cost);
}

unsigned
X86TargetLowering::getRegPressureLimit(const TargetRegisterClass *RC,
                                       MachineFunction &MF) const {
  unsigned FPDiff = RegInfo->hasFP(MF) ? 1 : 0;
  switch (RC->getID()) {
  default:
    return 0;
  case X86::GR32RegClassID:
    return 4 - FPDiff;
  case X86::GR64RegClassID:
    return 8 - FPDiff;
  case X86::VR128RegClassID:
    return Subtarget->is64Bit() ? 10 : 4;
  case X86::VR64RegClassID:
    return 4;
  }
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
X86TargetLowering::CanLowerReturn(CallingConv::ID CallConv, bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
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
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
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

    // If this is x86-64, and we disabled SSE, we can't return FP values
    if ((ValVT == MVT::f32 || ValVT == MVT::f64) &&
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
      if (ValVT.isVector() && ValVT.getSizeInBits() == 64) {
        ValToCopy = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::i64, ValToCopy);
        if (VA.getLocReg() == X86::XMM0 || VA.getLocReg() == X86::XMM1) {
          ValToCopy = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2i64,
                                  ValToCopy);
          
          // If we don't have SSE2 available, convert to v4f32 so the generated
          // register is legal.
          if (!Subtarget->hasSSE2())
            ValToCopy = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v4f32,ValToCopy);
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
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
                 RVLocs, *DAG.getContext());
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
    // point stack, we must guarantee the the value is popped from the stack, so
    // a CopyFromReg is not good enough - the copy instruction may be eliminated
    // if the return value is not used. We use the FpGET_ST0 instructions
    // instead.
    if (VA.getLocReg() == X86::ST0 || VA.getLocReg() == X86::ST1) {
      // If we prefer to use the value in xmm registers, copy it out as f80 and
      // use a truncate to move it from fp stack reg to xmm reg.
      if (isScalarFPTypeInSSEReg(VA.getValVT())) CopyVT = MVT::f80;
      bool isST0 = VA.getLocReg() == X86::ST0;
      unsigned Opc = 0;
      if (CopyVT == MVT::f32) Opc = isST0 ? X86::FpGET_ST0_32:X86::FpGET_ST1_32;
      if (CopyVT == MVT::f64) Opc = isST0 ? X86::FpGET_ST0_64:X86::FpGET_ST1_64;
      if (CopyVT == MVT::f80) Opc = isST0 ? X86::FpGET_ST0_80:X86::FpGET_ST1_80;
      SDValue Ops[] = { Chain, InFlag };
      Chain = SDValue(DAG.getMachineNode(Opc, dl, CopyVT, MVT::Other, MVT::Flag,
                                         Ops, 2), 1);
      Val = Chain.getValue(0);

      // Round the f80 to the right size, which also moves it to the appropriate
      // xmm register.
      if (CopyVT != VA.getValVT())
        Val = DAG.getNode(ISD::FP_ROUND, dl, VA.getValVT(), Val,
                          // This truncation won't change the value.
                          DAG.getIntPtrConstant(1));
    } else if (Is64Bit && CopyVT.isVector() && CopyVT.getSizeInBits() == 64) {
      // For x86-64, MMX values are returned in XMM0 / XMM1 except for v1i64.
      if (VA.getLocReg() == X86::XMM0 || VA.getLocReg() == X86::XMM1) {
        Chain = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(),
                                   MVT::v2i64, InFlag).getValue(1);
        Val = Chain.getValue(0);
        Val = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i64,
                          Val, DAG.getConstant(0, MVT::i64));
      } else {
        Chain = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(),
                                   MVT::i64, InFlag).getValue(1);
        Val = Chain.getValue(0);
      }
      Val = DAG.getNode(ISD::BIT_CONVERT, dl, CopyVT, Val);
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
static bool CallIsStructReturn(const SmallVectorImpl<ISD::OutputArg> &Outs) {
  if (Outs.empty())
    return false;

  return Outs[0].Flags.isSRet();
}

/// ArgsAreStructReturn - Determines whether a function uses struct
/// return semantics.
static bool
ArgsAreStructReturn(const SmallVectorImpl<ISD::InputArg> &Ins) {
  if (Ins.empty())
    return false;

  return Ins[0].Flags.isSRet();
}

/// CCAssignFnForNode - Selects the correct CCAssignFn for a the
/// given CallingConvention value.
CCAssignFn *X86TargetLowering::CCAssignFnForNode(CallingConv::ID CC) const {
  if (Subtarget->is64Bit()) {
    if (CC == CallingConv::GHC)
      return CC_X86_64_GHC;
    else if (Subtarget->isTargetWin64())
      return CC_X86_Win64_C;
    else
      return CC_X86_64_C;
  }

  if (CC == CallingConv::X86_FastCall)
    return CC_X86_32_FastCall;
  else if (CC == CallingConv::X86_ThisCall)
    return CC_X86_32_ThisCall;
  else if (CC == CallingConv::Fast)
    return CC_X86_32_FastCC;
  else if (CC == CallingConv::GHC)
    return CC_X86_32_GHC;
  else
    return CC_X86_32_C;
}

/// CreateCopyOfByValArgument - Make a copy of an aggregate at address specified
/// by "Src" to address "Dst" with size and alignment information specified by
/// the specific parameter attribute. The copy will be passed as a byval
/// function parameter.
static SDValue
CreateCopyOfByValArgument(SDValue Src, SDValue Dst, SDValue Chain,
                          ISD::ArgFlagsTy Flags, SelectionDAG &DAG,
                          DebugLoc dl) {
  SDValue SizeNode     = DAG.getConstant(Flags.getByValSize(), MVT::i32);
  return DAG.getMemcpy(Chain, dl, Dst, Src, SizeNode, Flags.getByValAlign(),
                       /*isVolatile*/false, /*AlwaysInline=*/true,
                       NULL, 0, NULL, 0);
}

/// IsTailCallConvention - Return true if the calling convention is one that
/// supports tail call optimization.
static bool IsTailCallConvention(CallingConv::ID CC) {
  return (CC == CallingConv::Fast || CC == CallingConv::GHC);
}

/// FuncIsMadeTailCallSafe - Return true if the function is being made into
/// a tailcall target by changing its ABI.
static bool FuncIsMadeTailCallSafe(CallingConv::ID CC) {
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
  bool AlwaysUseMutable = FuncIsMadeTailCallSafe(CallConv);
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
    int FI = MFI->CreateFixedObject(Flags.getByValSize(),
                                    VA.getLocMemOffset(), isImmutable);
    return DAG.getFrameIndex(FI, getPointerTy());
  } else {
    int FI = MFI->CreateFixedObject(ValVT.getSizeInBits()/8,
                                    VA.getLocMemOffset(), isImmutable);
    SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
    return DAG.getLoad(ValVT, dl, Chain, FIN,
                       PseudoSourceValue::getFixedStack(FI), 0,
                       false, false, 0);
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
  bool IsWin64 = Subtarget->isTargetWin64();

  assert(!(isVarArg && IsTailCallConvention(CallConv)) &&
         "Var args not supported with calling convention fastcc or ghc");

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
                 ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CCAssignFnForNode(CallConv));

  unsigned LastVal = ~0U;
  SDValue ArgValue;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    // TODO: If an arg is passed in two places (e.g. reg and stack), skip later
    // places.
    assert(VA.getValNo() != LastVal &&
           "Don't support value assigned to multiple locs yet");
    LastVal = VA.getValNo();

    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      TargetRegisterClass *RC = NULL;
      if (RegVT == MVT::i32)
        RC = X86::GR32RegisterClass;
      else if (Is64Bit && RegVT == MVT::i64)
        RC = X86::GR64RegisterClass;
      else if (RegVT == MVT::f32)
        RC = X86::FR32RegisterClass;
      else if (RegVT == MVT::f64)
        RC = X86::FR64RegisterClass;
      else if (RegVT.isVector() && RegVT.getSizeInBits() == 256)
        RC = X86::VR256RegisterClass;
      else if (RegVT.isVector() && RegVT.getSizeInBits() == 128)
        RC = X86::VR128RegisterClass;
      else if (RegVT.isVector() && RegVT.getSizeInBits() == 64)
        RC = X86::VR64RegisterClass;
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
        ArgValue = DAG.getNode(ISD::BIT_CONVERT, dl, VA.getValVT(), ArgValue);

      if (VA.isExtInLoc()) {
        // Handle MMX values passed in XMM regs.
        if (RegVT.isVector()) {
          ArgValue = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i64,
                                 ArgValue, DAG.getConstant(0, MVT::i64));
          ArgValue = DAG.getNode(ISD::BIT_CONVERT, dl, VA.getValVT(), ArgValue);
        } else
          ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
      }
    } else {
      assert(VA.isMemLoc());
      ArgValue = LowerMemArgument(Chain, CallConv, Ins, dl, DAG, VA, MFI, i);
    }

    // If value is passed via pointer - do a load.
    if (VA.getLocInfo() == CCValAssign::Indirect)
      ArgValue = DAG.getLoad(VA.getValVT(), dl, Chain, ArgValue, NULL, 0,
                             false, false, 0);

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
  if (FuncIsMadeTailCallSafe(CallConv))
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

      bool NoImplicitFloatOps = Fn->hasFnAttr(Attribute::NoImplicitFloat);
      assert(!(NumXMMRegs && !Subtarget->hasSSE1()) &&
             "SSE register cannot be used when SSE is disabled!");
      assert(!(NumXMMRegs && UseSoftFloat && NoImplicitFloatOps) &&
             "SSE register cannot be used when SSE is disabled!");
      if (UseSoftFloat || NoImplicitFloatOps || !Subtarget->hasSSE1())
        // Kernel mode asks for SSE to be disabled, so don't push them
        // on the stack.
        TotalNumXMMRegs = 0;

      // For X86-64, if there are vararg parameters that are passed via
      // registers, then we must store them to their spots on the stack so they
      // may be loaded by deferencing the result of va_next.
      FuncInfo->setVarArgsGPOffset(NumIntRegs * 8);
      FuncInfo->setVarArgsFPOffset(TotalNumIntRegs * 8 + NumXMMRegs * 16);
      FuncInfo->setRegSaveFrameIndex(
        MFI->CreateStackObject(TotalNumIntRegs * 8 + TotalNumXMMRegs * 16, 16,
                               false));

      // Store the integer parameter registers.
      SmallVector<SDValue, 8> MemOps;
      SDValue RSFIN = DAG.getFrameIndex(FuncInfo->getRegSaveFrameIndex(),
                                        getPointerTy());
      unsigned Offset = FuncInfo->getVarArgsGPOffset();
      for (; NumIntRegs != TotalNumIntRegs; ++NumIntRegs) {
        SDValue FIN = DAG.getNode(ISD::ADD, dl, getPointerTy(), RSFIN,
                                  DAG.getIntPtrConstant(Offset));
        unsigned VReg = MF.addLiveIn(GPR64ArgRegs[NumIntRegs],
                                     X86::GR64RegisterClass);
        SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i64);
        SDValue Store =
          DAG.getStore(Val.getValue(1), dl, Val, FIN,
                       PseudoSourceValue::getFixedStack(
                         FuncInfo->getRegSaveFrameIndex()),
                       Offset, false, false, 0);
        MemOps.push_back(Store);
        Offset += 8;
      }

      if (TotalNumXMMRegs != 0 && NumXMMRegs != TotalNumXMMRegs) {
        // Now store the XMM (fp + vector) parameter registers.
        SmallVector<SDValue, 11> SaveXMMOps;
        SaveXMMOps.push_back(Chain);

        unsigned AL = MF.addLiveIn(X86::AL, X86::GR8RegisterClass);
        SDValue ALVal = DAG.getCopyFromReg(DAG.getEntryNode(), dl, AL, MVT::i8);
        SaveXMMOps.push_back(ALVal);

        SaveXMMOps.push_back(DAG.getIntPtrConstant(
                               FuncInfo->getRegSaveFrameIndex()));
        SaveXMMOps.push_back(DAG.getIntPtrConstant(
                               FuncInfo->getVarArgsFPOffset()));

        for (; NumXMMRegs != TotalNumXMMRegs; ++NumXMMRegs) {
          unsigned VReg = MF.addLiveIn(XMMArgRegs[NumXMMRegs],
                                       X86::VR128RegisterClass);
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
  if (Subtarget->IsCalleePop(isVarArg, CallConv)) {
    FuncInfo->setBytesToPopOnReturn(StackSize); // Callee pops everything.
  } else {
    FuncInfo->setBytesToPopOnReturn(0); // Callee pops nothing.
    // If this is an sret function, the return should pop the hidden pointer.
    if (!Is64Bit && !IsTailCallConvention(CallConv) && ArgsAreStructReturn(Ins))
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

  return Chain;
}

SDValue
X86TargetLowering::LowerMemOpCallTo(SDValue Chain,
                                    SDValue StackPtr, SDValue Arg,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    const CCValAssign &VA,
                                    ISD::ArgFlagsTy Flags) const {
  const unsigned FirstStackArgOffset = (Subtarget->isTargetWin64() ? 32 : 0);
  unsigned LocMemOffset = FirstStackArgOffset + VA.getLocMemOffset();
  SDValue PtrOff = DAG.getIntPtrConstant(LocMemOffset);
  PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, PtrOff);
  if (Flags.isByVal()) {
    return CreateCopyOfByValArgument(Arg, PtrOff, Chain, Flags, DAG, dl);
  }
  return DAG.getStore(Chain, dl, Arg, PtrOff,
                      PseudoSourceValue::getStack(), LocMemOffset,
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
  OutRetAddr = DAG.getLoad(VT, dl, Chain, OutRetAddr, NULL, 0, false, false, 0);
  return SDValue(OutRetAddr.getNode(), 1);
}

/// EmitTailCallStoreRetAddr - Emit a store of the return adress if tail call
/// optimization is performed and it is required (FPDiff!=0).
static SDValue
EmitTailCallStoreRetAddr(SelectionDAG & DAG, MachineFunction &MF,
                         SDValue Chain, SDValue RetAddrFrIdx,
                         bool Is64Bit, int FPDiff, DebugLoc dl) {
  // Store the return address to the appropriate stack slot.
  if (!FPDiff) return Chain;
  // Calculate the new stack slot for the return address.
  int SlotSize = Is64Bit ? 8 : 4;
  int NewReturnAddrFI =
    MF.getFrameInfo()->CreateFixedObject(SlotSize, FPDiff-SlotSize, false);
  EVT VT = Is64Bit ? MVT::i64 : MVT::i32;
  SDValue NewRetAddrFrIdx = DAG.getFrameIndex(NewReturnAddrFI, VT);
  Chain = DAG.getStore(Chain, dl, RetAddrFrIdx, NewRetAddrFrIdx,
                       PseudoSourceValue::getFixedStack(NewReturnAddrFI), 0,
                       false, false, 0);
  return Chain;
}

SDValue
X86TargetLowering::LowerCall(SDValue Chain, SDValue Callee,
                             CallingConv::ID CallConv, bool isVarArg,
                             bool &isTailCall,
                             const SmallVectorImpl<ISD::OutputArg> &Outs,
                             const SmallVectorImpl<SDValue> &OutVals,
                             const SmallVectorImpl<ISD::InputArg> &Ins,
                             DebugLoc dl, SelectionDAG &DAG,
                             SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  bool Is64Bit        = Subtarget->is64Bit();
  bool IsStructRet    = CallIsStructReturn(Outs);
  bool IsSibcall      = false;

  if (isTailCall) {
    // Check if it's really possible to do a tail call.
    isTailCall = IsEligibleForTailCallOptimization(Callee, CallConv,
                    isVarArg, IsStructRet, MF.getFunction()->hasStructRetAttr(),
                                                   Outs, OutVals, Ins, DAG);

    // Sibcalls are automatically detected tailcalls which do not require
    // ABI changes.
    if (!GuaranteedTailCallOpt && isTailCall)
      IsSibcall = true;

    if (isTailCall)
      ++NumTailCalls;
  }

  assert(!(isVarArg && IsTailCallConvention(CallConv)) &&
         "Var args not supported with calling convention fastcc or ghc");

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
                 ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForNode(CallConv));

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();
  if (IsSibcall)
    // This is a sibcall. The memory operands are available in caller's
    // own caller's stack.
    NumBytes = 0;
  else if (GuaranteedTailCallOpt && IsTailCallConvention(CallConv))
    NumBytes = GetAlignedArgumentStackSize(NumBytes, DAG);

  int FPDiff = 0;
  if (isTailCall && !IsSibcall) {
    // Lower arguments at fp - stackoffset + fpdiff.
    unsigned NumBytesCallerPushed =
      MF.getInfo<X86MachineFunctionInfo>()->getBytesToPopOnReturn();
    FPDiff = NumBytesCallerPushed - NumBytes;

    // Set the delta of movement of the returnaddr stackslot.
    // But only set if delta is greater than previous delta.
    if (FPDiff < (MF.getInfo<X86MachineFunctionInfo>()->getTCReturnAddrDelta()))
      MF.getInfo<X86MachineFunctionInfo>()->setTCReturnAddrDelta(FPDiff);
  }

  if (!IsSibcall)
    Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));

  SDValue RetAddrFrIdx;
  // Load return adress for tail calls.
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
      if (RegVT.isVector() && RegVT.getSizeInBits() == 128) {
        // Special case: passing MMX values in XMM registers.
        Arg = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::i64, Arg);
        Arg = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2i64, Arg);
        Arg = getMOVL(DAG, dl, MVT::v2i64, DAG.getUNDEF(MVT::v2i64), Arg);
      } else
        Arg = DAG.getNode(ISD::ANY_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BIT_CONVERT, dl, RegVT, Arg);
      break;
    case CCValAssign::Indirect: {
      // Store the argument.
      SDValue SpillSlot = DAG.CreateStackTemporary(VA.getValVT());
      int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
      Chain = DAG.getStore(Chain, dl, Arg, SpillSlot,
                           PseudoSourceValue::getFixedStack(FI), 0,
                           false, false, 0);
      Arg = SpillSlot;
      break;
    }
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      if (isVarArg && Subtarget->isTargetWin64()) {
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
        StackPtr = DAG.getCopyFromReg(Chain, dl, X86StackPtr, getPointerTy());
      MemOpChains.push_back(LowerMemOpCallTo(Chain, StackPtr, Arg,
                                             dl, DAG, VA, Flags));
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into registers.
  SDValue InFlag;
  // Tail call byval lowering might overwrite argument registers so in case of
  // tail call optimization the copies to registers are lowered later.
  if (!isTailCall)
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
      Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                               RegsToPass[i].second, InFlag);
      InFlag = Chain.getValue(1);
    }

  if (Subtarget->isPICStyleGOT()) {
    // ELF / PIC requires GOT in the EBX register before function calls via PLT
    // GOT pointer.
    if (!isTailCall) {
      Chain = DAG.getCopyToReg(Chain, dl, X86::EBX,
                               DAG.getNode(X86ISD::GlobalBaseReg,
                                           DebugLoc(), getPointerTy()),
                               InFlag);
      InFlag = Chain.getValue(1);
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

  if (Is64Bit && isVarArg && !Subtarget->isTargetWin64()) {
    // From AMD64 ABI document:
    // For calls that may call functions that use varargs or stdargs
    // (prototype-less calls or calls to functions containing ellipsis (...) in
    // the declaration) %al is used as hidden argument to specify the number
    // of SSE registers used. The contents of %al do not need to match exactly
    // the number of registers, but must be an ubound on the number of SSE
    // registers used and is in the range 0 - 8 inclusive.

    // Count the number of XMM registers allocated.
    static const unsigned XMMArgRegs[] = {
      X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
      X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
    };
    unsigned NumXMMRegs = CCInfo.getFirstUnallocated(XMMArgRegs, 8);
    assert((Subtarget->hasSSE1() || !NumXMMRegs)
           && "SSE registers cannot be used when SSE is disabled");

    Chain = DAG.getCopyToReg(Chain, dl, X86::AL,
                             DAG.getConstant(NumXMMRegs, MVT::i8), InFlag);
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
    SDValue ArgChain = DAG.getStackArgumentTokenFactor(Chain);

    SmallVector<SDValue, 8> MemOpChains2;
    SDValue FIN;
    int FI = 0;
    // Do not flag preceeding copytoreg stuff together with the following stuff.
    InFlag = SDValue();
    if (GuaranteedTailCallOpt) {
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
            StackPtr = DAG.getCopyFromReg(Chain, dl, X86StackPtr,
                                          getPointerTy());
          Source = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, Source);

          MemOpChains2.push_back(CreateCopyOfByValArgument(Source, FIN,
                                                           ArgChain,
                                                           Flags, DAG, dl));
        } else {
          // Store relative to framepointer.
          MemOpChains2.push_back(
            DAG.getStore(ArgChain, dl, Arg, FIN,
                         PseudoSourceValue::getFixedStack(FI), 0,
                         false, false, 0));
        }
      }
    }

    if (!MemOpChains2.empty())
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                          &MemOpChains2[0], MemOpChains2.size());

    // Copy arguments to their registers.
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
      Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                               RegsToPass[i].second, InFlag);
      InFlag = Chain.getValue(1);
    }
    InFlag =SDValue();

    // Store the return address to the appropriate stack slot.
    Chain = EmitTailCallStoreRetAddr(DAG, MF, Chain, RetAddrFrIdx, Is64Bit,
                                     FPDiff, dl);
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
               Subtarget->getDarwinVers() < 9) {
        // PC-relative references to external symbols should go through $stub,
        // unless we're building with the leopard linker or later, which
        // automatically synthesizes these stubs.
        OpFlags = X86II::MO_DARWIN_STUB;
      }

      Callee = DAG.getTargetGlobalAddress(GV, dl, getPointerTy(),
                                          G->getOffset(), OpFlags);
    }
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    unsigned char OpFlags = 0;

    // On ELF targets, in either X86-64 or X86-32 mode, direct calls to external
    // symbols should go through the PLT.
    if (Subtarget->isTargetELF() &&
        getTargetMachine().getRelocationModel() == Reloc::PIC_) {
      OpFlags = X86II::MO_PLT;
    } else if (Subtarget->isPICStyleStubAny() &&
             Subtarget->getDarwinVers() < 9) {
      // PC-relative references to external symbols should go through $stub,
      // unless we're building with the leopard linker or later, which
      // automatically synthesizes these stubs.
      OpFlags = X86II::MO_DARWIN_STUB;
    }

    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy(),
                                         OpFlags);
  }

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
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

  // Add an implicit use GOT pointer in EBX.
  if (!isTailCall && Subtarget->isPICStyleGOT())
    Ops.push_back(DAG.getRegister(X86::EBX, getPointerTy()));

  // Add an implicit use of AL for non-Windows x86 64-bit vararg functions.
  if (Is64Bit && isVarArg && !Subtarget->isTargetWin64())
    Ops.push_back(DAG.getRegister(X86::AL, MVT::i8));

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
  if (Subtarget->IsCalleePop(isVarArg, CallConv))
    NumBytesForCalleeToPush = NumBytes;    // Callee pops everything
  else if (!Is64Bit && !IsTailCallConvention(CallConv) && IsStructRet)
    // If this is a call to a struct-return function, the callee
    // pops the hidden struct pointer, so we have to push it back.
    // This is common for Darwin/X86, Linux & Mingw32 targets.
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
  const TargetFrameInfo &TFI = *TM.getFrameInfo();
  unsigned StackAlignment = TFI.getStackAlignment();
  uint64_t AlignMask = StackAlignment - 1;
  int64_t Offset = StackSize;
  uint64_t SlotSize = TD->getPointerSize();
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
    if (!VR || TargetRegisterInfo::isPhysicalRegister(VR))
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
  CallingConv::ID CallerCC = CallerF->getCallingConv();
  bool CCMatch = CallerCC == CalleeCC;

  if (GuaranteedTailCallOpt) {
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

  // Do not sibcall optimize vararg calls unless the call site is not passing
  // any arguments.
  if (isVarArg && !Outs.empty())
    return false;

  // Also avoid sibcall optimization if either caller or callee uses struct
  // return semantics.
  if (isCalleeStructRet || isCallerStructRet)
    return false;

  // If the call result is in ST0 / ST1, it needs to be popped off the x87 stack.
  // Therefore if it's not used by the call it is not safe to optimize this into
  // a sibcall.
  bool Unused = false;
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    if (!Ins[i].Used) {
      Unused = true;
      break;
    }
  }
  if (Unused) {
    SmallVector<CCValAssign, 16> RVLocs;
    CCState CCInfo(CalleeCC, false, getTargetMachine(),
                   RVLocs, *DAG.getContext());
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
    CCState CCInfo1(CalleeCC, false, getTargetMachine(),
                    RVLocs1, *DAG.getContext());
    CCInfo1.AnalyzeCallResult(Ins, RetCC_X86);

    SmallVector<CCValAssign, 16> RVLocs2;
    CCState CCInfo2(CallerCC, false, getTargetMachine(),
                    RVLocs2, *DAG.getContext());
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
    CCState CCInfo(CalleeCC, isVarArg, getTargetMachine(),
                   ArgLocs, *DAG.getContext());
    CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForNode(CalleeCC));
    if (CCInfo.getNextStackOffset()) {
      MachineFunction &MF = DAG.getMachineFunction();
      if (MF.getInfo<X86MachineFunctionInfo>()->getBytesToPopOnReturn())
        return false;
      if (Subtarget->isTargetWin64())
        // Win64 ABI has additional complications.
        return false;

      // Check if the arguments are already laid out in the right way as
      // the caller's fixed stack objects.
      MachineFrameInfo *MFI = MF.getFrameInfo();
      const MachineRegisterInfo *MRI = &MF.getRegInfo();
      const X86InstrInfo *TII =
        ((X86TargetMachine&)getTargetMachine()).getInstrInfo();
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
X86TargetLowering::createFastISel(FunctionLoweringInfo &funcInfo) const {
  return X86::createFastISel(funcInfo);
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
  case X86ISD::SHUFPD:
  case X86ISD::SHUFPS:
  case X86ISD::MOVLHPS:
  case X86ISD::MOVLHPD:
  case X86ISD::MOVHLPS:
  case X86ISD::MOVLPS:
  case X86ISD::MOVLPD:
  case X86ISD::MOVSHDUP:
  case X86ISD::MOVSLDUP:
  case X86ISD::MOVSS:
  case X86ISD::MOVSD:
  case X86ISD::UNPCKLPS:
  case X86ISD::UNPCKLPD:
  case X86ISD::PUNPCKLWD:
  case X86ISD::PUNPCKLBW:
  case X86ISD::PUNPCKLDQ:
  case X86ISD::PUNPCKLQDQ:
  case X86ISD::UNPCKHPS:
  case X86ISD::UNPCKHPD:
  case X86ISD::PUNPCKHWD:
  case X86ISD::PUNPCKHBW:
  case X86ISD::PUNPCKHDQ:
  case X86ISD::PUNPCKHQDQ:
    return true;
  }
  return false;
}

static SDValue getTargetShuffleNode(unsigned Opc, DebugLoc dl, EVT VT,
                                               SDValue V1, SelectionDAG &DAG) {
  switch(Opc) {
  default: llvm_unreachable("Unknown x86 shuffle node");
  case X86ISD::MOVSHDUP:
  case X86ISD::MOVSLDUP:
    return DAG.getNode(Opc, dl, VT, V1);
  }

  return SDValue();
}

static SDValue getTargetShuffleNode(unsigned Opc, DebugLoc dl, EVT VT,
                          SDValue V1, unsigned TargetMask, SelectionDAG &DAG) {
  switch(Opc) {
  default: llvm_unreachable("Unknown x86 shuffle node");
  case X86ISD::PSHUFD:
  case X86ISD::PSHUFHW:
  case X86ISD::PSHUFLW:
    return DAG.getNode(Opc, dl, VT, V1, DAG.getConstant(TargetMask, MVT::i8));
  }

  return SDValue();
}

static SDValue getTargetShuffleNode(unsigned Opc, DebugLoc dl, EVT VT,
               SDValue V1, SDValue V2, unsigned TargetMask, SelectionDAG &DAG) {
  switch(Opc) {
  default: llvm_unreachable("Unknown x86 shuffle node");
  case X86ISD::SHUFPD:
  case X86ISD::SHUFPS:
    return DAG.getNode(Opc, dl, VT, V1, V2,
                       DAG.getConstant(TargetMask, MVT::i8));
  }
  return SDValue();
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
  case X86ISD::UNPCKLPS:
  case X86ISD::UNPCKLPD:
  case X86ISD::PUNPCKLWD:
  case X86ISD::PUNPCKLBW:
  case X86ISD::PUNPCKLDQ:
  case X86ISD::PUNPCKLQDQ:
  case X86ISD::UNPCKHPS:
  case X86ISD::UNPCKHPD:
  case X86ISD::PUNPCKHWD:
  case X86ISD::PUNPCKHBW:
  case X86ISD::PUNPCKHDQ:
  case X86ISD::PUNPCKHQDQ:
    return DAG.getNode(Opc, dl, VT, V1, V2);
  }
  return SDValue();
}

SDValue X86TargetLowering::getReturnAddressFrameIndex(SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
  int ReturnAddrIndex = FuncInfo->getRAIndex();

  if (ReturnAddrIndex == 0) {
    // Set up a frame object for the return address.
    uint64_t SlotSize = TD->getPointerSize();
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
      } else if (SetCCOpcode == ISD::SETLT && RHSC->isNullValue()) {
        // X < 0   -> X == 0, jump on sign.
        return X86::COND_S;
      } else if (SetCCOpcode == ISD::SETLT && RHSC->getZExtValue() == 1) {
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
  if ((ISD::isNON_EXTLoad(LHS.getNode()) && LHS.hasOneUse()) &&
      !(ISD::isNON_EXTLoad(RHS.getNode()) && RHS.hasOneUse())) {
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

/// isPSHUFDMask - Return true if the node specifies a shuffle of elements that
/// is suitable for input to PSHUFD or PSHUFW.  That is, it doesn't reference
/// the second operand.
static bool isPSHUFDMask(const SmallVectorImpl<int> &Mask, EVT VT) {
  if (VT == MVT::v4f32 || VT == MVT::v4i32 || VT == MVT::v4i16)
    return (Mask[0] < 4 && Mask[1] < 4 && Mask[2] < 4 && Mask[3] < 4);
  if (VT == MVT::v2f64 || VT == MVT::v2i64)
    return (Mask[0] < 2 && Mask[1] < 2);
  return false;
}

bool X86::isPSHUFDMask(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isPSHUFDMask(M, N->getValueType(0));
}

/// isPSHUFHWMask - Return true if the node specifies a shuffle of elements that
/// is suitable for input to PSHUFHW.
static bool isPSHUFHWMask(const SmallVectorImpl<int> &Mask, EVT VT) {
  if (VT != MVT::v8i16)
    return false;

  // Lower quadword copied in order or undef.
  for (int i = 0; i != 4; ++i)
    if (Mask[i] >= 0 && Mask[i] != i)
      return false;

  // Upper quadword shuffled.
  for (int i = 4; i != 8; ++i)
    if (Mask[i] >= 0 && (Mask[i] < 4 || Mask[i] > 7))
      return false;

  return true;
}

bool X86::isPSHUFHWMask(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isPSHUFHWMask(M, N->getValueType(0));
}

/// isPSHUFLWMask - Return true if the node specifies a shuffle of elements that
/// is suitable for input to PSHUFLW.
static bool isPSHUFLWMask(const SmallVectorImpl<int> &Mask, EVT VT) {
  if (VT != MVT::v8i16)
    return false;

  // Upper quadword copied in order.
  for (int i = 4; i != 8; ++i)
    if (Mask[i] >= 0 && Mask[i] != i)
      return false;

  // Lower quadword shuffled.
  for (int i = 0; i != 4; ++i)
    if (Mask[i] >= 4)
      return false;

  return true;
}

bool X86::isPSHUFLWMask(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isPSHUFLWMask(M, N->getValueType(0));
}

/// isPALIGNRMask - Return true if the node specifies a shuffle of elements that
/// is suitable for input to PALIGNR.
static bool isPALIGNRMask(const SmallVectorImpl<int> &Mask, EVT VT,
                          bool hasSSSE3) {
  int i, e = VT.getVectorNumElements();
  
  // Do not handle v2i64 / v2f64 shuffles with palignr.
  if (e < 4 || !hasSSSE3)
    return false;
  
  for (i = 0; i != e; ++i)
    if (Mask[i] >= 0)
      break;
  
  // All undef, not a palignr.
  if (i == e)
    return false;

  // Determine if it's ok to perform a palignr with only the LHS, since we
  // don't have access to the actual shuffle elements to see if RHS is undef.
  bool Unary = Mask[i] < (int)e;
  bool NeedsUnary = false;

  int s = Mask[i] - i;
  
  // Check the rest of the elements to see if they are consecutive.
  for (++i; i != e; ++i) {
    int m = Mask[i];
    if (m < 0) 
      continue;
    
    Unary = Unary && (m < (int)e);
    NeedsUnary = NeedsUnary || (m < s);

    if (NeedsUnary && !Unary)
      return false;
    if (Unary && m != ((s+i) & (e-1)))
      return false;
    if (!Unary && m != (s+i))
      return false;
  }
  return true;
}

bool X86::isPALIGNRMask(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isPALIGNRMask(M, N->getValueType(0), true);
}

/// isSHUFPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to SHUFP*.
static bool isSHUFPMask(const SmallVectorImpl<int> &Mask, EVT VT) {
  int NumElems = VT.getVectorNumElements();
  if (NumElems != 2 && NumElems != 4)
    return false;

  int Half = NumElems / 2;
  for (int i = 0; i < Half; ++i)
    if (!isUndefOrInRange(Mask[i], 0, NumElems))
      return false;
  for (int i = Half; i < NumElems; ++i)
    if (!isUndefOrInRange(Mask[i], NumElems, NumElems*2))
      return false;

  return true;
}

bool X86::isSHUFPMask(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isSHUFPMask(M, N->getValueType(0));
}

/// isCommutedSHUFP - Returns true if the shuffle mask is exactly
/// the reverse of what x86 shuffles want. x86 shuffles requires the lower
/// half elements to come from vector 1 (which would equal the dest.) and
/// the upper half to come from vector 2.
static bool isCommutedSHUFPMask(const SmallVectorImpl<int> &Mask, EVT VT) {
  int NumElems = VT.getVectorNumElements();

  if (NumElems != 2 && NumElems != 4)
    return false;

  int Half = NumElems / 2;
  for (int i = 0; i < Half; ++i)
    if (!isUndefOrInRange(Mask[i], NumElems, NumElems*2))
      return false;
  for (int i = Half; i < NumElems; ++i)
    if (!isUndefOrInRange(Mask[i], 0, NumElems))
      return false;
  return true;
}

static bool isCommutedSHUFP(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return isCommutedSHUFPMask(M, N->getValueType(0));
}

/// isMOVHLPSMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVHLPS.
bool X86::isMOVHLPSMask(ShuffleVectorSDNode *N) {
  if (N->getValueType(0).getVectorNumElements() != 4)
    return false;

  // Expect bit0 == 6, bit1 == 7, bit2 == 2, bit3 == 3
  return isUndefOrEqual(N->getMaskElt(0), 6) &&
         isUndefOrEqual(N->getMaskElt(1), 7) &&
         isUndefOrEqual(N->getMaskElt(2), 2) &&
         isUndefOrEqual(N->getMaskElt(3), 3);
}

/// isMOVHLPS_v_undef_Mask - Special case of isMOVHLPSMask for canonical form
/// of vector_shuffle v, v, <2, 3, 2, 3>, i.e. vector_shuffle v, undef,
/// <2, 3, 2, 3>
bool X86::isMOVHLPS_v_undef_Mask(ShuffleVectorSDNode *N) {
  unsigned NumElems = N->getValueType(0).getVectorNumElements();
  
  if (NumElems != 4)
    return false;
  
  return isUndefOrEqual(N->getMaskElt(0), 2) &&
  isUndefOrEqual(N->getMaskElt(1), 3) &&
  isUndefOrEqual(N->getMaskElt(2), 2) &&
  isUndefOrEqual(N->getMaskElt(3), 3);
}

/// isMOVLPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVLP{S|D}.
bool X86::isMOVLPMask(ShuffleVectorSDNode *N) {
  unsigned NumElems = N->getValueType(0).getVectorNumElements();

  if (NumElems != 2 && NumElems != 4)
    return false;

  for (unsigned i = 0; i < NumElems/2; ++i)
    if (!isUndefOrEqual(N->getMaskElt(i), i + NumElems))
      return false;

  for (unsigned i = NumElems/2; i < NumElems; ++i)
    if (!isUndefOrEqual(N->getMaskElt(i), i))
      return false;

  return true;
}

/// isMOVLHPSMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVLHPS.
bool X86::isMOVLHPSMask(ShuffleVectorSDNode *N) {
  unsigned NumElems = N->getValueType(0).getVectorNumElements();

  if (NumElems != 2 && NumElems != 4)
    return false;

  for (unsigned i = 0; i < NumElems/2; ++i)
    if (!isUndefOrEqual(N->getMaskElt(i), i))
      return false;

  for (unsigned i = 0; i < NumElems/2; ++i)
    if (!isUndefOrEqual(N->getMaskElt(i + NumElems/2), i + NumElems))
      return false;

  return true;
}

/// isUNPCKLMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to UNPCKL.
static bool isUNPCKLMask(const SmallVectorImpl<int> &Mask, EVT VT,
                         bool V2IsSplat = false) {
  int NumElts = VT.getVectorNumElements();
  if (NumElts != 2 && NumElts != 4 && NumElts != 8 && NumElts != 16)
    return false;

  for (int i = 0, j = 0; i != NumElts; i += 2, ++j) {
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
  return true;
}

bool X86::isUNPCKLMask(ShuffleVectorSDNode *N, bool V2IsSplat) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isUNPCKLMask(M, N->getValueType(0), V2IsSplat);
}

/// isUNPCKHMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to UNPCKH.
static bool isUNPCKHMask(const SmallVectorImpl<int> &Mask, EVT VT,
                         bool V2IsSplat = false) {
  int NumElts = VT.getVectorNumElements();
  if (NumElts != 2 && NumElts != 4 && NumElts != 8 && NumElts != 16)
    return false;

  for (int i = 0, j = 0; i != NumElts; i += 2, ++j) {
    int BitI  = Mask[i];
    int BitI1 = Mask[i+1];
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

bool X86::isUNPCKHMask(ShuffleVectorSDNode *N, bool V2IsSplat) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isUNPCKHMask(M, N->getValueType(0), V2IsSplat);
}

/// isUNPCKL_v_undef_Mask - Special case of isUNPCKLMask for canonical form
/// of vector_shuffle v, v, <0, 4, 1, 5>, i.e. vector_shuffle v, undef,
/// <0, 0, 1, 1>
static bool isUNPCKL_v_undef_Mask(const SmallVectorImpl<int> &Mask, EVT VT) {
  int NumElems = VT.getVectorNumElements();
  if (NumElems != 2 && NumElems != 4 && NumElems != 8 && NumElems != 16)
    return false;

  for (int i = 0, j = 0; i != NumElems; i += 2, ++j) {
    int BitI  = Mask[i];
    int BitI1 = Mask[i+1];
    if (!isUndefOrEqual(BitI, j))
      return false;
    if (!isUndefOrEqual(BitI1, j))
      return false;
  }
  return true;
}

bool X86::isUNPCKL_v_undef_Mask(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isUNPCKL_v_undef_Mask(M, N->getValueType(0));
}

/// isUNPCKH_v_undef_Mask - Special case of isUNPCKHMask for canonical form
/// of vector_shuffle v, v, <2, 6, 3, 7>, i.e. vector_shuffle v, undef,
/// <2, 2, 3, 3>
static bool isUNPCKH_v_undef_Mask(const SmallVectorImpl<int> &Mask, EVT VT) {
  int NumElems = VT.getVectorNumElements();
  if (NumElems != 2 && NumElems != 4 && NumElems != 8 && NumElems != 16)
    return false;

  for (int i = 0, j = NumElems / 2; i != NumElems; i += 2, ++j) {
    int BitI  = Mask[i];
    int BitI1 = Mask[i+1];
    if (!isUndefOrEqual(BitI, j))
      return false;
    if (!isUndefOrEqual(BitI1, j))
      return false;
  }
  return true;
}

bool X86::isUNPCKH_v_undef_Mask(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isUNPCKH_v_undef_Mask(M, N->getValueType(0));
}

/// isMOVLMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSS,
/// MOVSD, and MOVD, i.e. setting the lowest element.
static bool isMOVLMask(const SmallVectorImpl<int> &Mask, EVT VT) {
  if (VT.getVectorElementType().getSizeInBits() < 32)
    return false;

  int NumElts = VT.getVectorNumElements();

  if (!isUndefOrEqual(Mask[0], NumElts))
    return false;

  for (int i = 1; i < NumElts; ++i)
    if (!isUndefOrEqual(Mask[i], i))
      return false;

  return true;
}

bool X86::isMOVLMask(ShuffleVectorSDNode *N) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return ::isMOVLMask(M, N->getValueType(0));
}

/// isCommutedMOVL - Returns true if the shuffle mask is except the reverse
/// of what x86 movss want. X86 movs requires the lowest  element to be lowest
/// element of vector 2 and the other elements to come from vector 1 in order.
static bool isCommutedMOVLMask(const SmallVectorImpl<int> &Mask, EVT VT,
                               bool V2IsSplat = false, bool V2IsUndef = false) {
  int NumOps = VT.getVectorNumElements();
  if (NumOps != 2 && NumOps != 4 && NumOps != 8 && NumOps != 16)
    return false;

  if (!isUndefOrEqual(Mask[0], 0))
    return false;

  for (int i = 1; i < NumOps; ++i)
    if (!(isUndefOrEqual(Mask[i], i+NumOps) ||
          (V2IsUndef && isUndefOrInRange(Mask[i], NumOps, NumOps*2)) ||
          (V2IsSplat && isUndefOrEqual(Mask[i], NumOps))))
      return false;

  return true;
}

static bool isCommutedMOVL(ShuffleVectorSDNode *N, bool V2IsSplat = false,
                           bool V2IsUndef = false) {
  SmallVector<int, 8> M;
  N->getMask(M);
  return isCommutedMOVLMask(M, N->getValueType(0), V2IsSplat, V2IsUndef);
}

/// isMOVSHDUPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSHDUP.
bool X86::isMOVSHDUPMask(ShuffleVectorSDNode *N) {
  if (N->getValueType(0).getVectorNumElements() != 4)
    return false;

  // Expect 1, 1, 3, 3
  for (unsigned i = 0; i < 2; ++i) {
    int Elt = N->getMaskElt(i);
    if (Elt >= 0 && Elt != 1)
      return false;
  }

  bool HasHi = false;
  for (unsigned i = 2; i < 4; ++i) {
    int Elt = N->getMaskElt(i);
    if (Elt >= 0 && Elt != 3)
      return false;
    if (Elt == 3)
      HasHi = true;
  }
  // Don't use movshdup if it can be done with a shufps.
  // FIXME: verify that matching u, u, 3, 3 is what we want.
  return HasHi;
}

/// isMOVSLDUPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSLDUP.
bool X86::isMOVSLDUPMask(ShuffleVectorSDNode *N) {
  if (N->getValueType(0).getVectorNumElements() != 4)
    return false;

  // Expect 0, 0, 2, 2
  for (unsigned i = 0; i < 2; ++i)
    if (N->getMaskElt(i) > 0)
      return false;

  bool HasHi = false;
  for (unsigned i = 2; i < 4; ++i) {
    int Elt = N->getMaskElt(i);
    if (Elt >= 0 && Elt != 2)
      return false;
    if (Elt == 2)
      HasHi = true;
  }
  // Don't use movsldup if it can be done with a shufps.
  return HasHi;
}

/// isMOVDDUPMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVDDUP.
bool X86::isMOVDDUPMask(ShuffleVectorSDNode *N) {
  int e = N->getValueType(0).getVectorNumElements() / 2;

  for (int i = 0; i < e; ++i)
    if (!isUndefOrEqual(N->getMaskElt(i), i))
      return false;
  for (int i = 0; i < e; ++i)
    if (!isUndefOrEqual(N->getMaskElt(e+i), i))
      return false;
  return true;
}

/// getShuffleSHUFImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with PSHUF* and SHUFP* instructions.
unsigned X86::getShuffleSHUFImmediate(SDNode *N) {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  int NumOperands = SVOp->getValueType(0).getVectorNumElements();

  unsigned Shift = (NumOperands == 4) ? 2 : 1;
  unsigned Mask = 0;
  for (int i = 0; i < NumOperands; ++i) {
    int Val = SVOp->getMaskElt(NumOperands-i-1);
    if (Val < 0) Val = 0;
    if (Val >= NumOperands) Val -= NumOperands;
    Mask |= Val;
    if (i != NumOperands - 1)
      Mask <<= Shift;
  }
  return Mask;
}

/// getShufflePSHUFHWImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with the PSHUFHW instruction.
unsigned X86::getShufflePSHUFHWImmediate(SDNode *N) {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  unsigned Mask = 0;
  // 8 nodes, but we only care about the last 4.
  for (unsigned i = 7; i >= 4; --i) {
    int Val = SVOp->getMaskElt(i);
    if (Val >= 0)
      Mask |= (Val - 4);
    if (i != 4)
      Mask <<= 2;
  }
  return Mask;
}

/// getShufflePSHUFLWImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with the PSHUFLW instruction.
unsigned X86::getShufflePSHUFLWImmediate(SDNode *N) {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  unsigned Mask = 0;
  // 8 nodes, but we only care about the first 4.
  for (int i = 3; i >= 0; --i) {
    int Val = SVOp->getMaskElt(i);
    if (Val >= 0)
      Mask |= Val;
    if (i != 0)
      Mask <<= 2;
  }
  return Mask;
}

/// getShufflePALIGNRImmediate - Return the appropriate immediate to shuffle
/// the specified VECTOR_SHUFFLE mask with the PALIGNR instruction.
unsigned X86::getShufflePALIGNRImmediate(SDNode *N) {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  EVT VVT = N->getValueType(0);
  unsigned EltSize = VVT.getVectorElementType().getSizeInBits() >> 3;
  int Val = 0;

  unsigned i, e;
  for (i = 0, e = VVT.getVectorNumElements(); i != e; ++i) {
    Val = SVOp->getMaskElt(i);
    if (Val >= 0)
      break;
  }
  return (Val - i) * EltSize;
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
    int idx = SVOp->getMaskElt(i);
    if (idx < 0)
      MaskVec.push_back(idx);
    else if (idx < (int)NumElems)
      MaskVec.push_back(idx + NumElems);
    else
      MaskVec.push_back(idx - NumElems);
  }
  return DAG.getVectorShuffle(VT, SVOp->getDebugLoc(), SVOp->getOperand(1),
                              SVOp->getOperand(0), &MaskVec[0]);
}

/// CommuteVectorShuffleMask - Change values in a shuffle permute mask assuming
/// the two vector operands have swapped position.
static void CommuteVectorShuffleMask(SmallVectorImpl<int> &Mask, EVT VT) {
  unsigned NumElems = VT.getVectorNumElements();
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

/// ShouldXformToMOVHLPS - Return true if the node should be transformed to
/// match movhlps. The lower half elements should come from upper half of
/// V1 (and in order), and the upper half elements should come from the upper
/// half of V2 (and in order).
static bool ShouldXformToMOVHLPS(ShuffleVectorSDNode *Op) {
  if (Op->getValueType(0).getVectorNumElements() != 4)
    return false;
  for (unsigned i = 0, e = 2; i != e; ++i)
    if (!isUndefOrEqual(Op->getMaskElt(i), i+2))
      return false;
  for (unsigned i = 2; i != 4; ++i)
    if (!isUndefOrEqual(Op->getMaskElt(i), i+4))
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

/// ShouldXformToMOVLP{S|D} - Return true if the node should be transformed to
/// match movlp{s|d}. The lower half elements should come from lower half of
/// V1 (and in order), and the upper half elements should come from the upper
/// half of V2 (and in order). And since V1 will become the source of the
/// MOVLP, it must be either a vector load or a scalar load to vector.
static bool ShouldXformToMOVLP(SDNode *V1, SDNode *V2,
                               ShuffleVectorSDNode *Op) {
  if (!ISD::isNON_EXTLoad(V1) && !isScalarLoadToVector(V1))
    return false;
  // Is V2 is a vector load, don't do this transformation. We will try to use
  // load folding shufps op.
  if (ISD::isNON_EXTLoad(V2))
    return false;

  unsigned NumElems = Op->getValueType(0).getVectorNumElements();

  if (NumElems != 2 && NumElems != 4)
    return false;
  for (unsigned i = 0, e = NumElems/2; i != e; ++i)
    if (!isUndefOrEqual(Op->getMaskElt(i), i))
      return false;
  for (unsigned i = NumElems/2; i != NumElems; ++i)
    if (!isUndefOrEqual(Op->getMaskElt(i), i+NumElems))
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
static SDValue getZeroVector(EVT VT, bool HasSSE2, SelectionDAG &DAG,
                             DebugLoc dl) {
  assert(VT.isVector() && "Expected a vector type");

  // Always build zero vectors as <4 x i32> or <2 x i32> bitcasted
  // to their dest type. This ensures they get CSE'd.
  SDValue Vec;
  if (VT.getSizeInBits() == 64) { // MMX
    SDValue Cst = DAG.getTargetConstant(0, MVT::i32);
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v2i32, Cst, Cst);
  } else if (VT.getSizeInBits() == 128) {
    if (HasSSE2) {  // SSE2
      SDValue Cst = DAG.getTargetConstant(0, MVT::i32);
      Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Cst, Cst, Cst, Cst);
    } else { // SSE1
      SDValue Cst = DAG.getTargetConstantFP(+0.0, MVT::f32);
      Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4f32, Cst, Cst, Cst, Cst);
    }
  } else if (VT.getSizeInBits() == 256) { // AVX
    // 256-bit logic and arithmetic instructions in AVX are
    // all floating-point, no support for integer ops. Default
    // to emitting fp zeroed vectors then.
    SDValue Cst = DAG.getTargetConstantFP(+0.0, MVT::f32);
    SDValue Ops[] = { Cst, Cst, Cst, Cst, Cst, Cst, Cst, Cst };
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v8f32, Ops, 8);
  }
  return DAG.getNode(ISD::BIT_CONVERT, dl, VT, Vec);
}

/// getOnesVector - Returns a vector of specified type with all bits set.
///
static SDValue getOnesVector(EVT VT, SelectionDAG &DAG, DebugLoc dl) {
  assert(VT.isVector() && "Expected a vector type");

  // Always build ones vectors as <4 x i32> or <2 x i32> bitcasted to their dest
  // type.  This ensures they get CSE'd.
  SDValue Cst = DAG.getTargetConstant(~0U, MVT::i32);
  SDValue Vec;
  if (VT.getSizeInBits() == 64) // MMX
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v2i32, Cst, Cst);
  else // SSE
    Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Cst, Cst, Cst, Cst);
  return DAG.getNode(ISD::BIT_CONVERT, dl, VT, Vec);
}


/// NormalizeMask - V2 is a splat, modify the mask (if needed) so all elements
/// that point to V2 points to its first element.
static SDValue NormalizeMask(ShuffleVectorSDNode *SVOp, SelectionDAG &DAG) {
  EVT VT = SVOp->getValueType(0);
  unsigned NumElems = VT.getVectorNumElements();

  bool Changed = false;
  SmallVector<int, 8> MaskVec;
  SVOp->getMask(MaskVec);

  for (unsigned i = 0; i != NumElems; ++i) {
    if (MaskVec[i] > (int)NumElems) {
      MaskVec[i] = NumElems;
      Changed = true;
    }
  }
  if (Changed)
    return DAG.getVectorShuffle(VT, SVOp->getDebugLoc(), SVOp->getOperand(0),
                                SVOp->getOperand(1), &MaskVec[0]);
  return SDValue(SVOp, 0);
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

/// getUnpackhMask - Returns a vector_shuffle node for an unpackh operation.
static SDValue getUnpackh(SelectionDAG &DAG, DebugLoc dl, EVT VT, SDValue V1,
                          SDValue V2) {
  unsigned NumElems = VT.getVectorNumElements();
  unsigned Half = NumElems/2;
  SmallVector<int, 8> Mask;
  for (unsigned i = 0; i != Half; ++i) {
    Mask.push_back(i + Half);
    Mask.push_back(i + NumElems + Half);
  }
  return DAG.getVectorShuffle(VT, dl, V1, V2, &Mask[0]);
}

/// PromoteSplat - Promote a splat of v4i32, v8i16 or v16i8 to v4f32.
static SDValue PromoteSplat(ShuffleVectorSDNode *SV, SelectionDAG &DAG) {
  if (SV->getValueType(0).getVectorNumElements() <= 4)
    return SDValue(SV, 0);

  EVT PVT = MVT::v4f32;
  EVT VT = SV->getValueType(0);
  DebugLoc dl = SV->getDebugLoc();
  SDValue V1 = SV->getOperand(0);
  int NumElems = VT.getVectorNumElements();
  int EltNo = SV->getSplatIndex();

  // unpack elements to the correct location
  while (NumElems > 4) {
    if (EltNo < NumElems/2) {
      V1 = getUnpackl(DAG, dl, VT, V1, V1);
    } else {
      V1 = getUnpackh(DAG, dl, VT, V1, V1);
      EltNo -= NumElems/2;
    }
    NumElems >>= 1;
  }

  // Perform the splat.
  int SplatMask[4] = { EltNo, EltNo, EltNo, EltNo };
  V1 = DAG.getNode(ISD::BIT_CONVERT, dl, PVT, V1);
  V1 = DAG.getVectorShuffle(PVT, dl, V1, DAG.getUNDEF(PVT), &SplatMask[0]);
  return DAG.getNode(ISD::BIT_CONVERT, dl, VT, V1);
}

/// getShuffleVectorZeroOrUndef - Return a vector_shuffle of the specified
/// vector of zero or undef vector.  This produces a shuffle where the low
/// element of V2 is swizzled into the zero/undef vector, landing at element
/// Idx.  This produces a shuffle mask like 4,1,2,3 (idx=0) or  0,1,2,4 (idx=3).
static SDValue getShuffleVectorZeroOrUndef(SDValue V2, unsigned Idx,
                                             bool isZero, bool HasSSE2,
                                             SelectionDAG &DAG) {
  EVT VT = V2.getValueType();
  SDValue V1 = isZero
    ? getZeroVector(VT, HasSSE2, DAG, V2.getDebugLoc()) : DAG.getUNDEF(VT);
  unsigned NumElems = VT.getVectorNumElements();
  SmallVector<int, 16> MaskVec;
  for (unsigned i = 0; i != NumElems; ++i)
    // If this is the insertion idx, put the low elt of V2 here.
    MaskVec.push_back(i == Idx ? NumElems : i);
  return DAG.getVectorShuffle(VT, V2.getDebugLoc(), V1, V2, &MaskVec[0]);
}

/// getShuffleScalarElt - Returns the scalar element that will make up the ith
/// element of the result of the vector shuffle.
SDValue getShuffleScalarElt(SDNode *N, int Index, SelectionDAG &DAG,
                            unsigned Depth) {
  if (Depth == 6)
    return SDValue();  // Limit search depth.

  SDValue V = SDValue(N, 0);
  EVT VT = V.getValueType();
  unsigned Opcode = V.getOpcode();

  // Recurse into ISD::VECTOR_SHUFFLE node to find scalars.
  if (const ShuffleVectorSDNode *SV = dyn_cast<ShuffleVectorSDNode>(N)) {
    Index = SV->getMaskElt(Index);

    if (Index < 0)
      return DAG.getUNDEF(VT.getVectorElementType());

    int NumElems = VT.getVectorNumElements();
    SDValue NewV = (Index < NumElems) ? SV->getOperand(0) : SV->getOperand(1);
    return getShuffleScalarElt(NewV.getNode(), Index % NumElems, DAG, Depth+1);
  }

  // Recurse into target specific vector shuffles to find scalars.
  if (isTargetShuffle(Opcode)) {
    int NumElems = VT.getVectorNumElements();
    SmallVector<unsigned, 16> ShuffleMask;
    SDValue ImmN;

    switch(Opcode) {
    case X86ISD::SHUFPS:
    case X86ISD::SHUFPD:
      ImmN = N->getOperand(N->getNumOperands()-1);
      DecodeSHUFPSMask(NumElems,
                       cast<ConstantSDNode>(ImmN)->getZExtValue(),
                       ShuffleMask);
      break;
    case X86ISD::PUNPCKHBW:
    case X86ISD::PUNPCKHWD:
    case X86ISD::PUNPCKHDQ:
    case X86ISD::PUNPCKHQDQ:
      DecodePUNPCKHMask(NumElems, ShuffleMask);
      break;
    case X86ISD::UNPCKHPS:
    case X86ISD::UNPCKHPD:
      DecodeUNPCKHPMask(NumElems, ShuffleMask);
      break;
    case X86ISD::PUNPCKLBW:
    case X86ISD::PUNPCKLWD:
    case X86ISD::PUNPCKLDQ:
    case X86ISD::PUNPCKLQDQ:
      DecodePUNPCKLMask(NumElems, ShuffleMask);
      break;
    case X86ISD::UNPCKLPS:
    case X86ISD::UNPCKLPD:
      DecodeUNPCKLPMask(NumElems, ShuffleMask);
      break;
    case X86ISD::MOVHLPS:
      DecodeMOVHLPSMask(NumElems, ShuffleMask);
      break;
    case X86ISD::MOVLHPS:
      DecodeMOVLHPSMask(NumElems, ShuffleMask);
      break;
    case X86ISD::PSHUFD:
      ImmN = N->getOperand(N->getNumOperands()-1);
      DecodePSHUFMask(NumElems,
                      cast<ConstantSDNode>(ImmN)->getZExtValue(),
                      ShuffleMask);
      break;
    case X86ISD::PSHUFHW:
      ImmN = N->getOperand(N->getNumOperands()-1);
      DecodePSHUFHWMask(cast<ConstantSDNode>(ImmN)->getZExtValue(),
                        ShuffleMask);
      break;
    case X86ISD::PSHUFLW:
      ImmN = N->getOperand(N->getNumOperands()-1);
      DecodePSHUFLWMask(cast<ConstantSDNode>(ImmN)->getZExtValue(),
                        ShuffleMask);
      break;
    case X86ISD::MOVSS:
    case X86ISD::MOVSD: {
      // The index 0 always comes from the first element of the second source,
      // this is why MOVSS and MOVSD are used in the first place. The other
      // elements come from the other positions of the first source vector.
      unsigned OpNum = (Index == 0) ? 1 : 0;
      return getShuffleScalarElt(V.getOperand(OpNum).getNode(), Index, DAG,
                                 Depth+1);
    }
    default:
      assert("not implemented for target shuffle node");
      return SDValue();
    }

    Index = ShuffleMask[Index];
    if (Index < 0)
      return DAG.getUNDEF(VT.getVectorElementType());

    SDValue NewV = (Index < NumElems) ? N->getOperand(0) : N->getOperand(1);
    return getShuffleScalarElt(NewV.getNode(), Index % NumElems, DAG,
                               Depth+1);
  }

  // Actual nodes that may contain scalar elements
  if (Opcode == ISD::BIT_CONVERT) {
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
/// search can start in two diferent directions, from left or right.
static
unsigned getNumOfConsecutiveZeros(SDNode *N, int NumElems,
                                  bool ZerosFromLeft, SelectionDAG &DAG) {
  int i = 0;

  while (i < NumElems) {
    unsigned Index = ZerosFromLeft ? i : NumElems-i-1;
    SDValue Elt = getShuffleScalarElt(N, Index, DAG, 0);
    if (!(Elt.getNode() &&
         (Elt.getOpcode() == ISD::UNDEF || X86::isZeroNode(Elt))))
      break;
    ++i;
  }

  return i;
}

/// isShuffleMaskConsecutive - Check if the shuffle mask indicies from MaskI to
/// MaskE correspond consecutively to elements from one of the vector operands,
/// starting from its index OpIdx. Also tell OpNum which source vector operand.
static
bool isShuffleMaskConsecutive(ShuffleVectorSDNode *SVOp, int MaskI, int MaskE,
                              int OpIdx, int NumElems, unsigned &OpNum) {
  bool SeenV1 = false;
  bool SeenV2 = false;

  for (int i = MaskI; i <= MaskE; ++i, ++OpIdx) {
    int Idx = SVOp->getMaskElt(i);
    // Ignore undef indicies
    if (Idx < 0)
      continue;

    if (Idx < NumElems)
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
            NumElems-NumZeros-1, // Mask End Index
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
            NumElems-1,   // Mask End Index
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
        V = getZeroVector(MVT::v8i16, true, DAG, dl);
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

  return DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v16i8, V);
}

/// LowerBuildVectorv8i16 - Custom lower build_vector of v8i16.
///
static SDValue LowerBuildVectorv8i16(SDValue Op, unsigned NonZeros,
                                     unsigned NumNonZero, unsigned NumZero,
                                     SelectionDAG &DAG,
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
          V = getZeroVector(MVT::v8i16, true, DAG, dl);
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
  bool isMMX = VT.getSizeInBits() == 64;
  EVT ShVT = isMMX ? MVT::v1i64 : MVT::v2i64;
  unsigned Opc = isLeft ? X86ISD::VSHL : X86ISD::VSRL;
  SrcOp = DAG.getNode(ISD::BIT_CONVERT, dl, ShVT, SrcOp);
  return DAG.getNode(ISD::BIT_CONVERT, dl, VT,
                     DAG.getNode(Opc, dl, ShVT, SrcOp,
                             DAG.getConstant(NumBits, TLI.getShiftAmountTy())));
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
    } else if (Ptr.getOpcode() == ISD::ADD &&
               isa<ConstantSDNode>(Ptr.getOperand(1)) &&
               isa<FrameIndexSDNode>(Ptr.getOperand(0))) {
      FI = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
      Offset = Ptr.getConstantOperandVal(1);
      Ptr = Ptr.getOperand(0);
    } else {
      return SDValue();
    }

    SDValue Chain = LD->getChain();
    // Make sure the stack object alignment is at least 16.
    MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
    if (DAG.InferPtrAlignment(Ptr) < 16) {
      if (MFI->isFixedObjectIndex(FI)) {
        // Can't change the alignment. FIXME: It's possible to compute
        // the exact stack offset and reference FI + adjust offset instead.
        // If someone *really* cares about this. That's the way to implement it.
        return SDValue();
      } else {
        MFI->setObjectAlignment(FI, 16);
      }
    }

    // (Offset % 16) must be multiple of 4. Then address is then
    // Ptr + (Offset & ~15).
    if (Offset < 0)
      return SDValue();
    if ((Offset % 16) & 3)
      return SDValue();
    int64_t StartOffset = Offset & ~15;
    if (StartOffset)
      Ptr = DAG.getNode(ISD::ADD, Ptr.getDebugLoc(), Ptr.getValueType(),
                        Ptr,DAG.getConstant(StartOffset, Ptr.getValueType()));

    int EltNo = (Offset - StartOffset) >> 2;
    int Mask[4] = { EltNo, EltNo, EltNo, EltNo };
    EVT VT = (PVT == MVT::i32) ? MVT::v4i32 : MVT::v4f32;
    SDValue V1 = DAG.getLoad(VT, dl, Chain, Ptr,LD->getSrcValue(),0,
                             false, false, 0);
    // Canonicalize it to a v4i32 shuffle.
    V1 = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v4i32, V1);
    return DAG.getNode(ISD::BIT_CONVERT, dl, VT,
                       DAG.getVectorShuffle(MVT::v4i32, dl, V1,
                                            DAG.getUNDEF(MVT::v4i32), &Mask[0]));
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
                                        DebugLoc &dl, SelectionDAG &DAG) {
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
      return DAG.getLoad(VT, dl, LDBase->getChain(), LDBase->getBasePtr(),
                         LDBase->getSrcValue(), LDBase->getSrcValueOffset(),
                         LDBase->isVolatile(), LDBase->isNonTemporal(), 0);
    return DAG.getLoad(VT, dl, LDBase->getChain(), LDBase->getBasePtr(),
                       LDBase->getSrcValue(), LDBase->getSrcValueOffset(),
                       LDBase->isVolatile(), LDBase->isNonTemporal(),
                       LDBase->getAlignment());
  } else if (NumElems == 4 && LastLoadedElt == 1) {
    SDVTList Tys = DAG.getVTList(MVT::v2i64, MVT::Other);
    SDValue Ops[] = { LDBase->getChain(), LDBase->getBasePtr() };
    SDValue ResNode = DAG.getNode(X86ISD::VZEXT_LOAD, dl, Tys, Ops, 2);
    return DAG.getNode(ISD::BIT_CONVERT, dl, VT, ResNode);
  }
  return SDValue();
}

SDValue
X86TargetLowering::LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  // All zero's are handled with pxor in SSE2 and above, xorps in SSE1.
  // All one's are handled with pcmpeqd. In AVX, zero's are handled with
  // vpxor in 128-bit and xor{pd,ps} in 256-bit, but no 256 version of pcmpeqd
  // is present, so AllOnes is ignored.
  if (ISD::isBuildVectorAllZeros(Op.getNode()) ||
      (Op.getValueType().getSizeInBits() != 256 &&
       ISD::isBuildVectorAllOnes(Op.getNode()))) {
    // Canonicalize this to either <4 x i32> or <2 x i32> (SSE vs MMX) to
    // 1) ensure the zero vectors are CSE'd, and 2) ensure that i64 scalars are
    // eliminated on x86-32 hosts.
    if (Op.getValueType() == MVT::v4i32 || Op.getValueType() == MVT::v2i32)
      return Op;

    if (ISD::isBuildVectorAllOnes(Op.getNode()))
      return getOnesVector(Op.getValueType(), DAG, dl);
    return getZeroVector(Op.getValueType(), Subtarget->hasSSE2(), DAG, dl);
  }

  EVT VT = Op.getValueType();
  EVT ExtVT = VT.getVectorElementType();
  unsigned EVTBits = ExtVT.getSizeInBits();

  unsigned NumElems = Op.getNumOperands();
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
        // Handle MMX and SSE both.
        EVT VecVT = VT == MVT::v2i64 ? MVT::v4i32 : MVT::v2i32;
        unsigned VecElts = VT == MVT::v2i64 ? 4 : 2;

        // Truncate the value (which may itself be a constant) to i32, and
        // convert it to a vector with movd (S2V+shuffle to zero extend).
        Item = DAG.getNode(ISD::TRUNCATE, dl, MVT::i32, Item);
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VecVT, Item);
        Item = getShuffleVectorZeroOrUndef(Item, 0, true,
                                           Subtarget->hasSSE2(), DAG);

        // Now we have our 32-bit value zero extended in the low element of
        // a vector.  If Idx != 0, swizzle it into place.
        if (Idx != 0) {
          SmallVector<int, 4> Mask;
          Mask.push_back(Idx);
          for (unsigned i = 1; i != VecElts; ++i)
            Mask.push_back(i);
          Item = DAG.getVectorShuffle(VecVT, dl, Item,
                                      DAG.getUNDEF(Item.getValueType()),
                                      &Mask[0]);
        }
        return DAG.getNode(ISD::BIT_CONVERT, dl, Op.getValueType(), Item);
      }
    }

    // If we have a constant or non-constant insertion into the low element of
    // a vector, we can do this with SCALAR_TO_VECTOR + shuffle of zero into
    // the rest of the elements.  This will be matched as movd/movq/movss/movsd
    // depending on what the source datatype is.
    if (Idx == 0) {
      if (NumZero == 0) {
        return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Item);
      } else if (ExtVT == MVT::i32 || ExtVT == MVT::f32 || ExtVT == MVT::f64 ||
          (ExtVT == MVT::i64 && Subtarget->is64Bit())) {
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Item);
        // Turn it into a MOVL (i.e. movss, movsd, or movd) to a zero vector.
        return getShuffleVectorZeroOrUndef(Item, 0, true, Subtarget->hasSSE2(),
                                           DAG);
      } else if (ExtVT == MVT::i16 || ExtVT == MVT::i8) {
        Item = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Item);
        EVT MiddleVT = VT.getSizeInBits() == 64 ? MVT::v2i32 : MVT::v4i32;
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MiddleVT, Item);
        Item = getShuffleVectorZeroOrUndef(Item, 0, true,
                                           Subtarget->hasSSE2(), DAG);
        return DAG.getNode(ISD::BIT_CONVERT, dl, VT, Item);
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
      Item = getShuffleVectorZeroOrUndef(Item, 0, NumZero > 0,
                                         Subtarget->hasSSE2(), DAG);
      SmallVector<int, 8> MaskVec;
      for (unsigned i = 0; i < NumElems; i++)
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

  // Let legalizer expand 2-wide build_vectors.
  if (EVTBits == 64) {
    if (NumNonZero == 1) {
      // One half is zero or undef.
      unsigned Idx = CountTrailingZeros_32(NonZeros);
      SDValue V2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT,
                                 Op.getOperand(Idx));
      return getShuffleVectorZeroOrUndef(V2, Idx, true,
                                         Subtarget->hasSSE2(), DAG);
    }
    return SDValue();
  }

  // If element VT is < 32 bits, convert it to inserts into a zero vector.
  if (EVTBits == 8 && NumElems == 16) {
    SDValue V = LowerBuildVectorv16i8(Op, NonZeros,NumNonZero,NumZero, DAG,
                                        *this);
    if (V.getNode()) return V;
  }

  if (EVTBits == 16 && NumElems == 8) {
    SDValue V = LowerBuildVectorv8i16(Op, NonZeros,NumNonZero,NumZero, DAG,
                                      *this);
    if (V.getNode()) return V;
  }

  // If element VT is == 32 bits, turn it into a number of shuffles.
  SmallVector<SDValue, 8> V;
  V.resize(NumElems);
  if (NumElems == 4 && NumZero > 0) {
    for (unsigned i = 0; i < 4; ++i) {
      bool isZero = !(NonZeros & (1 << i));
      if (isZero)
        V[i] = getZeroVector(VT, Subtarget->hasSSE2(), DAG, dl);
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

    SmallVector<int, 8> MaskVec;
    bool Reverse = (NonZeros & 0x3) == 2;
    for (unsigned i = 0; i < 2; ++i)
      MaskVec.push_back(Reverse ? 1-i : i);
    Reverse = ((NonZeros & (0x3 << 2)) >> 2) == 2;
    for (unsigned i = 0; i < 2; ++i)
      MaskVec.push_back(Reverse ? 1-i+NumElems : i+NumElems);
    return DAG.getVectorShuffle(VT, dl, V[0], V[1], &MaskVec[0]);
  }

  if (Values.size() > 1 && VT.getSizeInBits() == 128) {
    // Check for a build vector of consecutive loads.
    for (unsigned i = 0; i < NumElems; ++i)
      V[i] = Op.getOperand(i);
    
    // Check for elements which are consecutive loads.
    SDValue LD = EltsFromConsecutiveLoads(VT, V, dl, DAG);
    if (LD.getNode())
      return LD;
    
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

SDValue
X86TargetLowering::LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const {
  // We support concatenate two MMX registers and place them in a MMX
  // register.  This is better than doing a stack convert.
  DebugLoc dl = Op.getDebugLoc();
  EVT ResVT = Op.getValueType();
  assert(Op.getNumOperands() == 2);
  assert(ResVT == MVT::v2i64 || ResVT == MVT::v4i32 ||
         ResVT == MVT::v8i16 || ResVT == MVT::v16i8);
  int Mask[2];
  SDValue InVec = DAG.getNode(ISD::BIT_CONVERT,dl, MVT::v1i64, Op.getOperand(0));
  SDValue VecOp = DAG.getNode(X86ISD::MOVQ2DQ, dl, MVT::v2i64, InVec);
  InVec = Op.getOperand(1);
  if (InVec.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    unsigned NumElts = ResVT.getVectorNumElements();
    VecOp = DAG.getNode(ISD::BIT_CONVERT, dl, ResVT, VecOp);
    VecOp = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, ResVT, VecOp,
                       InVec.getOperand(0), DAG.getIntPtrConstant(NumElts/2+1));
  } else {
    InVec = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v1i64, InVec);
    SDValue VecOp2 = DAG.getNode(X86ISD::MOVQ2DQ, dl, MVT::v2i64, InVec);
    Mask[0] = 0; Mask[1] = 2;
    VecOp = DAG.getVectorShuffle(MVT::v2i64, dl, VecOp, VecOp2, Mask);
  }
  return DAG.getNode(ISD::BIT_CONVERT, dl, ResVT, VecOp);
}

// v8i16 shuffles - Prefer shuffles in the following order:
// 1. [all]   pshuflw, pshufhw, optional move
// 2. [ssse3] 1 x pshufb
// 3. [ssse3] 2 x pshufb + 1 x por
// 4. [all]   mov + pshuflw + pshufhw + N x (pextrw + pinsrw)
SDValue
X86TargetLowering::LowerVECTOR_SHUFFLEv8i16(SDValue Op,
                                            SelectionDAG &DAG) const {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  DebugLoc dl = SVOp->getDebugLoc();
  SmallVector<int, 8> MaskVals;

  // Determine if more than 1 of the words in each of the low and high quadwords
  // of the result come from the same quadword of one of the two inputs.  Undef
  // mask values count as coming from any quadword, for better codegen.
  SmallVector<unsigned, 4> LoQuad(4);
  SmallVector<unsigned, 4> HiQuad(4);
  BitVector InputQuads(4);
  for (unsigned i = 0; i < 8; ++i) {
    SmallVectorImpl<unsigned> &Quad = i < 4 ? LoQuad : HiQuad;
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
      BestLoQuad = InputQuads.find_first();
      BestHiQuad = InputQuads.find_next(BestLoQuad);
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
    SmallVector<int, 8> MaskV;
    MaskV.push_back(BestLoQuad < 0 ? 0 : BestLoQuad);
    MaskV.push_back(BestHiQuad < 0 ? 1 : BestHiQuad);
    NewV = DAG.getVectorShuffle(MVT::v2i64, dl,
                  DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2i64, V1),
                  DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2i64, V2), &MaskV[0]);
    NewV = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v8i16, NewV);

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
      TargetMask = pshufhw ? X86::getShufflePSHUFHWImmediate(NewV.getNode()):
                             X86::getShufflePSHUFLWImmediate(NewV.getNode());
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
      if (TwoInputs && (EltIdx >= 16)) {
        pshufbMask.push_back(DAG.getConstant(0x80, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0x80, MVT::i8));
        continue;
      }
      pshufbMask.push_back(DAG.getConstant(EltIdx,   MVT::i8));
      pshufbMask.push_back(DAG.getConstant(EltIdx+1, MVT::i8));
    }
    V1 = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v16i8, V1);
    V1 = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v16i8, V1,
                     DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::v16i8, &pshufbMask[0], 16));
    if (!TwoInputs)
      return DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v8i16, V1);

    // Calculate the shuffle mask for the second input, shuffle it, and
    // OR it with the first shuffled input.
    pshufbMask.clear();
    for (unsigned i = 0; i != 8; ++i) {
      int EltIdx = MaskVals[i] * 2;
      if (EltIdx < 16) {
        pshufbMask.push_back(DAG.getConstant(0x80, MVT::i8));
        pshufbMask.push_back(DAG.getConstant(0x80, MVT::i8));
        continue;
      }
      pshufbMask.push_back(DAG.getConstant(EltIdx - 16, MVT::i8));
      pshufbMask.push_back(DAG.getConstant(EltIdx - 15, MVT::i8));
    }
    V2 = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v16i8, V2);
    V2 = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v16i8, V2,
                     DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::v16i8, &pshufbMask[0], 16));
    V1 = DAG.getNode(ISD::OR, dl, MVT::v16i8, V1, V2);
    return DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v8i16, V1);
  }

  // If BestLoQuad >= 0, generate a pshuflw to put the low elements in order,
  // and update MaskVals with new element order.
  BitVector InOrder(8);
  if (BestLoQuad >= 0) {
    SmallVector<int, 8> MaskV;
    for (int i = 0; i != 4; ++i) {
      int idx = MaskVals[i];
      if (idx < 0) {
        MaskV.push_back(-1);
        InOrder.set(i);
      } else if ((idx / 4) == BestLoQuad) {
        MaskV.push_back(idx & 3);
        InOrder.set(i);
      } else {
        MaskV.push_back(-1);
      }
    }
    for (unsigned i = 4; i != 8; ++i)
      MaskV.push_back(i);
    NewV = DAG.getVectorShuffle(MVT::v8i16, dl, NewV, DAG.getUNDEF(MVT::v8i16),
                                &MaskV[0]);

    if (NewV.getOpcode() == ISD::VECTOR_SHUFFLE && Subtarget->hasSSSE3())
      NewV = getTargetShuffleNode(X86ISD::PSHUFLW, dl, MVT::v8i16,
                               NewV.getOperand(0),
                               X86::getShufflePSHUFLWImmediate(NewV.getNode()),
                               DAG);
  }

  // If BestHi >= 0, generate a pshufhw to put the high elements in order,
  // and update MaskVals with the new element order.
  if (BestHiQuad >= 0) {
    SmallVector<int, 8> MaskV;
    for (unsigned i = 0; i != 4; ++i)
      MaskV.push_back(i);
    for (unsigned i = 4; i != 8; ++i) {
      int idx = MaskVals[i];
      if (idx < 0) {
        MaskV.push_back(-1);
        InOrder.set(i);
      } else if ((idx / 4) == BestHiQuad) {
        MaskV.push_back((idx & 3) + 4);
        InOrder.set(i);
      } else {
        MaskV.push_back(-1);
      }
    }
    NewV = DAG.getVectorShuffle(MVT::v8i16, dl, NewV, DAG.getUNDEF(MVT::v8i16),
                                &MaskV[0]);

    if (NewV.getOpcode() == ISD::VECTOR_SHUFFLE && Subtarget->hasSSSE3())
      NewV = getTargetShuffleNode(X86ISD::PSHUFHW, dl, MVT::v8i16,
                              NewV.getOperand(0),
                              X86::getShufflePSHUFHWImmediate(NewV.getNode()),
                              DAG);
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
    SDValue ExtOp = (EltIdx < 8)
    ? DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16, V1,
                  DAG.getIntPtrConstant(EltIdx))
    : DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16, V2,
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
  SmallVector<int, 16> MaskVals;
  SVOp->getMask(MaskVals);

  // If we have SSSE3, case 1 is generated when all result bytes come from
  // one of  the inputs.  Otherwise, case 2 is generated.  If no SSSE3 is
  // present, fall back to case 3.
  // FIXME: kill V2Only once shuffles are canonizalized by getNode.
  bool V1Only = true;
  bool V2Only = true;
  for (unsigned i = 0; i < 16; ++i) {
    int EltIdx = MaskVals[i];
    if (EltIdx < 0)
      continue;
    if (EltIdx < 16)
      V2Only = false;
    else
      V1Only = false;
  }

  // If SSSE3, use 1 pshufb instruction per vector with elements in the result.
  if (TLI.getSubtarget()->hasSSSE3()) {
    SmallVector<SDValue,16> pshufbMask;

    // If all result elements are from one input vector, then only translate
    // undef mask values to 0x80 (zero out result) in the pshufb mask.
    //
    // Otherwise, we have elements from both input vectors, and must zero out
    // elements that come from V2 in the first mask, and V1 in the second mask
    // so that we can OR them together.
    bool TwoInputs = !(V1Only || V2Only);
    for (unsigned i = 0; i != 16; ++i) {
      int EltIdx = MaskVals[i];
      if (EltIdx < 0 || (TwoInputs && EltIdx >= 16)) {
        pshufbMask.push_back(DAG.getConstant(0x80, MVT::i8));
        continue;
      }
      pshufbMask.push_back(DAG.getConstant(EltIdx, MVT::i8));
    }
    // If all the elements are from V2, assign it to V1 and return after
    // building the first pshufb.
    if (V2Only)
      V1 = V2;
    V1 = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v16i8, V1,
                     DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::v16i8, &pshufbMask[0], 16));
    if (!TwoInputs)
      return V1;

    // Calculate the shuffle mask for the second input, shuffle it, and
    // OR it with the first shuffled input.
    pshufbMask.clear();
    for (unsigned i = 0; i != 16; ++i) {
      int EltIdx = MaskVals[i];
      if (EltIdx < 16) {
        pshufbMask.push_back(DAG.getConstant(0x80, MVT::i8));
        continue;
      }
      pshufbMask.push_back(DAG.getConstant(EltIdx - 16, MVT::i8));
    }
    V2 = DAG.getNode(X86ISD::PSHUFB, dl, MVT::v16i8, V2,
                     DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::v16i8, &pshufbMask[0], 16));
    return DAG.getNode(ISD::OR, dl, MVT::v16i8, V1, V2);
  }

  // No SSSE3 - Calculate in place words and then fix all out of place words
  // With 0-16 extracts & inserts.  Worst case is 16 bytes out of order from
  // the 16 different words that comprise the two doublequadword input vectors.
  V1 = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v8i16, V1);
  V2 = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v8i16, V2);
  SDValue NewV = V2Only ? V2 : V1;
  for (int i = 0; i != 8; ++i) {
    int Elt0 = MaskVals[i*2];
    int Elt1 = MaskVals[i*2+1];

    // This word of the result is all undef, skip it.
    if (Elt0 < 0 && Elt1 < 0)
      continue;

    // This word of the result is already in the correct place, skip it.
    if (V1Only && (Elt0 == i*2) && (Elt1 == i*2+1))
      continue;
    if (V2Only && (Elt0 == i*2+16) && (Elt1 == i*2+17))
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
                             DAG.getConstant(8, TLI.getShiftAmountTy()));
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
                              DAG.getConstant(8, TLI.getShiftAmountTy()));
      else if (Elt1 >= 0)
        InsElt0 = DAG.getNode(ISD::AND, dl, MVT::i16, InsElt0,
                             DAG.getConstant(0x00FF, MVT::i16));
      InsElt = Elt1 >= 0 ? DAG.getNode(ISD::OR, dl, MVT::i16, InsElt, InsElt0)
                         : InsElt0;
    }
    NewV = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v8i16, NewV, InsElt,
                       DAG.getIntPtrConstant(i));
  }
  return DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v16i8, NewV);
}

/// RewriteAsNarrowerShuffle - Try rewriting v8i16 and v16i8 shuffles as 4 wide
/// ones, or rewriting v4i32 / v2i32 as 2 wide ones if possible. This can be
/// done when every pair / quad of shuffle mask elements point to elements in
/// the right sequence. e.g.
/// vector_shuffle <>, <>, < 3, 4, | 10, 11, | 0, 1, | 14, 15>
static
SDValue RewriteAsNarrowerShuffle(ShuffleVectorSDNode *SVOp,
                                 SelectionDAG &DAG,
                                 const TargetLowering &TLI, DebugLoc dl) {
  EVT VT = SVOp->getValueType(0);
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  unsigned NumElems = VT.getVectorNumElements();
  unsigned NewWidth = (NumElems == 4) ? 2 : 4;
  EVT MaskVT = (NewWidth == 4) ? MVT::v4i16 : MVT::v2i32;
  EVT NewVT = MaskVT;
  switch (VT.getSimpleVT().SimpleTy) {
  default: assert(false && "Unexpected!");
  case MVT::v4f32: NewVT = MVT::v2f64; break;
  case MVT::v4i32: NewVT = MVT::v2i64; break;
  case MVT::v8i16: NewVT = MVT::v4i32; break;
  case MVT::v16i8: NewVT = MVT::v4i32; break;
  }

  if (NewWidth == 2) {
    if (VT.isInteger())
      NewVT = MVT::v2i64;
    else
      NewVT = MVT::v2f64;
  }
  int Scale = NumElems / NewWidth;
  SmallVector<int, 8> MaskVec;
  for (unsigned i = 0; i < NumElems; i += Scale) {
    int StartIdx = -1;
    for (int j = 0; j < Scale; ++j) {
      int EltIdx = SVOp->getMaskElt(i+j);
      if (EltIdx < 0)
        continue;
      if (StartIdx == -1)
        StartIdx = EltIdx - (EltIdx % Scale);
      if (EltIdx != StartIdx + j)
        return SDValue();
    }
    if (StartIdx == -1)
      MaskVec.push_back(-1);
    else
      MaskVec.push_back(StartIdx / Scale);
  }

  V1 = DAG.getNode(ISD::BIT_CONVERT, dl, NewVT, V1);
  V2 = DAG.getNode(ISD::BIT_CONVERT, dl, NewVT, V2);
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
      if ((ExtVT.SimpleTy != MVT::i64 || Subtarget->is64Bit()) &&
          SrcOp.getOpcode() == ISD::SCALAR_TO_VECTOR &&
          SrcOp.getOperand(0).getOpcode() == ISD::BIT_CONVERT &&
          SrcOp.getOperand(0).getOperand(0).getValueType() == ExtVT) {
        // PR2108
        OpVT = (OpVT == MVT::v2f64) ? MVT::v2i64 : MVT::v4i32;
        return DAG.getNode(ISD::BIT_CONVERT, dl, VT,
                           DAG.getNode(X86ISD::VZEXT_MOVL, dl, OpVT,
                                       DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                                   OpVT,
                                                   SrcOp.getOperand(0)
                                                          .getOperand(0))));
      }
    }
  }

  return DAG.getNode(ISD::BIT_CONVERT, dl, VT,
                     DAG.getNode(X86ISD::VZEXT_MOVL, dl, OpVT,
                                 DAG.getNode(ISD::BIT_CONVERT, dl,
                                             OpVT, SrcOp)));
}

/// LowerVECTOR_SHUFFLE_4wide - Handle all 4 wide cases with a number of
/// shuffles.
static SDValue
LowerVECTOR_SHUFFLE_4wide(ShuffleVectorSDNode *SVOp, SelectionDAG &DAG) {
  SDValue V1 = SVOp->getOperand(0);
  SDValue V2 = SVOp->getOperand(1);
  DebugLoc dl = SVOp->getDebugLoc();
  EVT VT = SVOp->getValueType(0);

  SmallVector<std::pair<int, int>, 8> Locs;
  Locs.resize(4);
  SmallVector<int, 8> Mask1(4U, -1);
  SmallVector<int, 8> PermMask;
  SVOp->getMask(PermMask);

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

    SmallVector<int, 8> Mask2(4U, -1);

    for (unsigned i = 0; i != 4; ++i) {
      if (Locs[i].first == -1)
        continue;
      else {
        unsigned Idx = (i < 2) ? 0 : 4;
        Idx += Locs[i].first * 2 + Locs[i].second;
        Mask2[i] = Idx;
      }
    }

    return DAG.getVectorShuffle(VT, dl, V1, V1, &Mask2[0]);
  } else if (NumLo == 3 || NumHi == 3) {
    // Otherwise, we must have three elements from one vector, call it X, and
    // one element from the other, call it Y.  First, use a shufps to build an
    // intermediate vector with the one element from Y and the element from X
    // that will be in the same half in the final destination (the indexes don't
    // matter). Then, use a shufps to build the final vector, taking the half
    // containing the element from Y from the intermediate, and the other half
    // from X.
    if (NumHi == 3) {
      // Normalize it so the 3 elements come from V1.
      CommuteVectorShuffleMask(PermMask, VT);
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
    } else {
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
  }

  // Break it into (shuffle shuffle_hi, shuffle_lo).
  Locs.clear();
  SmallVector<int,8> LoMask(4U, -1);
  SmallVector<int,8> HiMask(4U, -1);

  SmallVector<int,8> *MaskPtr = &LoMask;
  unsigned MaskIdx = 0;
  unsigned LoIdx = 0;
  unsigned HiIdx = 2;
  for (unsigned i = 0; i != 4; ++i) {
    if (i == 2) {
      MaskPtr = &HiMask;
      MaskIdx = 1;
      LoIdx = 0;
      HiIdx = 2;
    }
    int Idx = PermMask[i];
    if (Idx < 0) {
      Locs[i] = std::make_pair(-1, -1);
    } else if (Idx < 4) {
      Locs[i] = std::make_pair(MaskIdx, LoIdx);
      (*MaskPtr)[LoIdx] = Idx;
      LoIdx++;
    } else {
      Locs[i] = std::make_pair(MaskIdx, HiIdx);
      (*MaskPtr)[HiIdx] = Idx;
      HiIdx++;
    }
  }

  SDValue LoShuffle = DAG.getVectorShuffle(VT, dl, V1, V2, &LoMask[0]);
  SDValue HiShuffle = DAG.getVectorShuffle(VT, dl, V1, V2, &HiMask[0]);
  SmallVector<int, 8> MaskOps;
  for (unsigned i = 0; i != 4; ++i) {
    if (Locs[i].first == -1) {
      MaskOps.push_back(-1);
    } else {
      unsigned Idx = Locs[i].first * 4 + Locs[i].second;
      MaskOps.push_back(Idx);
    }
  }
  return DAG.getVectorShuffle(VT, dl, LoShuffle, HiShuffle, &MaskOps[0]);
}

static bool MayFoldVectorLoad(SDValue V) {
  if (V.hasOneUse() && V.getOpcode() == ISD::BIT_CONVERT)
    V = V.getOperand(0);
  if (V.hasOneUse() && V.getOpcode() == ISD::SCALAR_TO_VECTOR)
    V = V.getOperand(0);
  if (MayFoldLoad(V))
    return true;
  return false;
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

  // v4f32 or v4i32
  return getTargetShuffleNode(X86ISD::MOVLHPS, dl, VT, V1, V2, DAG);
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
  if (MayFoldVectorLoad(V1) && MayFoldIntoStore(Op))
    CanFoldLoad = true;

  if (CanFoldLoad) {
    if (HasSSE2 && NumElems == 2)
      return getTargetShuffleNode(X86ISD::MOVLPD, dl, VT, V1, V2, DAG);

    if (NumElems == 4)
      return getTargetShuffleNode(X86ISD::MOVLPS, dl, VT, V1, V2, DAG);
  }

  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  // movl and movlp will both match v2i64, but v2i64 is never matched by
  // movl earlier because we make it strict to avoid messing with the movlp load
  // folding logic (see the code above getMOVLP call). Match it here then,
  // this is horrible, but will stay like this until we move all shuffle
  // matching to x86 specific nodes. Note that for the 1st condition all
  // types are matched with movsd.
  if ((HasSSE2 && NumElems == 2) || !X86::isMOVLMask(SVOp))
    return getTargetShuffleNode(X86ISD::MOVSD, dl, VT, V1, V2, DAG);
  else if (HasSSE2)
    return getTargetShuffleNode(X86ISD::MOVSS, dl, VT, V1, V2, DAG);


  assert(VT != MVT::v4i32 && "unsupported shuffle type");

  // Invert the operand order and use SHUFPS to match it.
  return getTargetShuffleNode(X86ISD::SHUFPS, dl, VT, V2, V1,
                              X86::getShuffleSHUFImmediate(SVOp), DAG);
}

static inline unsigned getUNPCKLOpcode(EVT VT) {
  switch(VT.getSimpleVT().SimpleTy) {
  case MVT::v4i32: return X86ISD::PUNPCKLDQ;
  case MVT::v2i64: return X86ISD::PUNPCKLQDQ;
  case MVT::v4f32: return X86ISD::UNPCKLPS;
  case MVT::v2f64: return X86ISD::UNPCKLPD;
  case MVT::v16i8: return X86ISD::PUNPCKLBW;
  case MVT::v8i16: return X86ISD::PUNPCKLWD;
  default:
    llvm_unreachable("Unknow type for unpckl");
  }
  return 0;
}

static inline unsigned getUNPCKHOpcode(EVT VT) {
  switch(VT.getSimpleVT().SimpleTy) {
  case MVT::v4i32: return X86ISD::PUNPCKHDQ;
  case MVT::v2i64: return X86ISD::PUNPCKHQDQ;
  case MVT::v4f32: return X86ISD::UNPCKHPS;
  case MVT::v2f64: return X86ISD::UNPCKHPD;
  case MVT::v16i8: return X86ISD::PUNPCKHBW;
  case MVT::v8i16: return X86ISD::PUNPCKHWD;
  default:
    llvm_unreachable("Unknow type for unpckh");
  }
  return 0;
}

SDValue
X86TargetLowering::LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned NumElems = VT.getVectorNumElements();
  bool isMMX = VT.getSizeInBits() == 64;
  bool V1IsUndef = V1.getOpcode() == ISD::UNDEF;
  bool V2IsUndef = V2.getOpcode() == ISD::UNDEF;
  bool V1IsSplat = false;
  bool V2IsSplat = false;
  bool HasSSE2 = Subtarget->hasSSE2() || Subtarget->hasAVX();
  bool HasSSE3 = Subtarget->hasSSE3() || Subtarget->hasAVX();
  MachineFunction &MF = DAG.getMachineFunction();
  bool OptForSize = MF.getFunction()->hasFnAttr(Attribute::OptimizeForSize);

  if (isZeroShuffle(SVOp))
    return getZeroVector(VT, Subtarget->hasSSE2(), DAG, dl);

  // Promote splats to v4f32.
  if (SVOp->isSplat()) {
    if (isMMX || NumElems < 4)
      return Op;
    return PromoteSplat(SVOp, DAG);
  }

  // If the shuffle can be profitably rewritten as a narrower shuffle, then
  // do it!
  if (VT == MVT::v8i16 || VT == MVT::v16i8) {
    SDValue NewOp = RewriteAsNarrowerShuffle(SVOp, DAG, *this, dl);
    if (NewOp.getNode())
      return DAG.getNode(ISD::BIT_CONVERT, dl, VT,
                         LowerVECTOR_SHUFFLE(NewOp, DAG));
  } else if ((VT == MVT::v4i32 || (VT == MVT::v4f32 && Subtarget->hasSSE2()))) {
    // FIXME: Figure out a cleaner way to do this.
    // Try to make use of movq to zero out the top part.
    if (ISD::isBuildVectorAllZeros(V2.getNode())) {
      SDValue NewOp = RewriteAsNarrowerShuffle(SVOp, DAG, *this, dl);
      if (NewOp.getNode()) {
        if (isCommutedMOVL(cast<ShuffleVectorSDNode>(NewOp), true, false))
          return getVZextMovL(VT, NewOp.getValueType(), NewOp.getOperand(0),
                              DAG, Subtarget, dl);
      }
    } else if (ISD::isBuildVectorAllZeros(V1.getNode())) {
      SDValue NewOp = RewriteAsNarrowerShuffle(SVOp, DAG, *this, dl);
      if (NewOp.getNode() && X86::isMOVLMask(cast<ShuffleVectorSDNode>(NewOp)))
        return getVZextMovL(VT, NewOp.getValueType(), NewOp.getOperand(1),
                            DAG, Subtarget, dl);
    }
  }

  // NOTE: isPSHUFDMask can also match both masks below (unpckl_undef and
  // unpckh_undef). Only use pshufd if speed is more important than size.
  if (OptForSize && X86::isUNPCKL_v_undef_Mask(SVOp))
    if (VT != MVT::v2i64 && VT != MVT::v2f64)
      return getTargetShuffleNode(getUNPCKLOpcode(VT), dl, VT, V1, V1, DAG);
  if (OptForSize && X86::isUNPCKH_v_undef_Mask(SVOp))
    if (VT != MVT::v2i64 && VT != MVT::v2f64)
      return getTargetShuffleNode(getUNPCKHOpcode(VT), dl, VT, V1, V1, DAG);

  if (X86::isPSHUFDMask(SVOp)) {
    // The actual implementation will match the mask in the if above and then
    // during isel it can match several different instructions, not only pshufd
    // as its name says, sad but true, emulate the behavior for now...
    if (X86::isMOVDDUPMask(SVOp) && ((VT == MVT::v4f32 || VT == MVT::v2i64)))
        return getTargetShuffleNode(X86ISD::MOVLHPS, dl, VT, V1, V1, DAG);

    unsigned TargetMask = X86::getShuffleSHUFImmediate(SVOp);

    if (HasSSE2 && (VT == MVT::v4f32 || VT == MVT::v4i32))
      return getTargetShuffleNode(X86ISD::PSHUFD, dl, VT, V1, TargetMask, DAG);

    if (HasSSE2 && (VT == MVT::v2i64 || VT == MVT::v2f64))
      return getTargetShuffleNode(X86ISD::SHUFPD, dl, VT, V1, V1,
                                  TargetMask, DAG);

    if (VT == MVT::v4f32)
      return getTargetShuffleNode(X86ISD::SHUFPS, dl, VT, V1, V1,
                                  TargetMask, DAG);
  }

  // Check if this can be converted into a logical shift.
  bool isLeft = false;
  unsigned ShAmt = 0;
  SDValue ShVal;
  bool isShift = getSubtarget()->hasSSE2() &&
    isVectorShift(SVOp, DAG, isLeft, ShVal, ShAmt);
  if (isShift && ShVal.hasOneUse()) {
    // If the shifted value has multiple uses, it may be cheaper to use
    // v_set0 + movlhps or movhlps, etc.
    EVT EltVT = VT.getVectorElementType();
    ShAmt *= EltVT.getSizeInBits();
    return getVShift(isLeft, VT, ShVal, ShAmt, DAG, *this, dl);
  }

  if (X86::isMOVLMask(SVOp)) {
    if (V1IsUndef)
      return V2;
    if (ISD::isBuildVectorAllZeros(V1.getNode()))
      return getVZextMovL(VT, VT, V2, DAG, Subtarget, dl);
    if (!isMMX && !X86::isMOVLPMask(SVOp)) {
      if (HasSSE2 && (VT == MVT::v2i64 || VT == MVT::v2f64))
        return getTargetShuffleNode(X86ISD::MOVSD, dl, VT, V1, V2, DAG);

      if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return getTargetShuffleNode(X86ISD::MOVSS, dl, VT, V1, V2, DAG);
    }
  }

  // FIXME: fold these into legal mask.
  if (!isMMX) {
    if (X86::isMOVLHPSMask(SVOp) && !X86::isUNPCKLMask(SVOp))
      return getMOVLowToHigh(Op, dl, DAG, HasSSE2);

    if (X86::isMOVHLPSMask(SVOp))
      return getMOVHighToLow(Op, dl, DAG);

    if (X86::isMOVSHDUPMask(SVOp) && HasSSE3 && V2IsUndef && NumElems == 4)
      return getTargetShuffleNode(X86ISD::MOVSHDUP, dl, VT, V1, DAG);

    if (X86::isMOVSLDUPMask(SVOp) && HasSSE3 && V2IsUndef && NumElems == 4)
      return getTargetShuffleNode(X86ISD::MOVSLDUP, dl, VT, V1, DAG);

    if (X86::isMOVLPMask(SVOp))
      return getMOVLP(Op, dl, DAG, HasSSE2);
  }

  if (ShouldXformToMOVHLPS(SVOp) ||
      ShouldXformToMOVLP(V1.getNode(), V2.getNode(), SVOp))
    return CommuteVectorShuffle(SVOp, DAG);

  if (isShift) {
    // No better options. Use a vshl / vsrl.
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
  if ((V1IsSplat || V1IsUndef) && !(V2IsSplat || V2IsUndef)) {
    Op = CommuteVectorShuffle(SVOp, DAG);
    SVOp = cast<ShuffleVectorSDNode>(Op);
    V1 = SVOp->getOperand(0);
    V2 = SVOp->getOperand(1);
    std::swap(V1IsSplat, V2IsSplat);
    std::swap(V1IsUndef, V2IsUndef);
    Commuted = true;
  }

  if (isCommutedMOVL(SVOp, V2IsSplat, V2IsUndef)) {
    // Shuffling low element of v1 into undef, just return v1.
    if (V2IsUndef)
      return V1;
    // If V2 is a splat, the mask may be malformed such as <4,3,3,3>, which
    // the instruction selector will not match, so get a canonical MOVL with
    // swapped operands to undo the commute.
    return getMOVL(DAG, dl, VT, V2, V1);
  }

  if (X86::isUNPCKL_v_undef_Mask(SVOp) || X86::isUNPCKLMask(SVOp))
    return (isMMX) ?
      Op : getTargetShuffleNode(getUNPCKLOpcode(VT), dl, VT, V1, V2, DAG);

  if (X86::isUNPCKH_v_undef_Mask(SVOp) || X86::isUNPCKHMask(SVOp))
    return (isMMX) ?
      Op : getTargetShuffleNode(getUNPCKHOpcode(VT), dl, VT, V1, V2, DAG);

  if (V2IsSplat) {
    // Normalize mask so all entries that point to V2 points to its first
    // element then try to match unpck{h|l} again. If match, return a
    // new vector_shuffle with the corrected mask.
    SDValue NewMask = NormalizeMask(SVOp, DAG);
    ShuffleVectorSDNode *NSVOp = cast<ShuffleVectorSDNode>(NewMask);
    if (NSVOp != SVOp) {
      if (X86::isUNPCKLMask(NSVOp, true)) {
        return NewMask;
      } else if (X86::isUNPCKHMask(NSVOp, true)) {
        return NewMask;
      }
    }
  }

  if (Commuted) {
    // Commute is back and try unpck* again.
    // FIXME: this seems wrong.
    SDValue NewOp = CommuteVectorShuffle(SVOp, DAG);
    ShuffleVectorSDNode *NewSVOp = cast<ShuffleVectorSDNode>(NewOp);

    if (X86::isUNPCKL_v_undef_Mask(NewSVOp) || X86::isUNPCKLMask(NewSVOp))
      return (isMMX) ?
        NewOp : getTargetShuffleNode(getUNPCKLOpcode(VT), dl, VT, V2, V1, DAG);

    if (X86::isUNPCKH_v_undef_Mask(NewSVOp) || X86::isUNPCKHMask(NewSVOp))
      return (isMMX) ?
        NewOp : getTargetShuffleNode(getUNPCKHOpcode(VT), dl, VT, V2, V1, DAG);
  }

  // FIXME: for mmx, bitcast v2i32 to v4i16 for shuffle.

  // Normalize the node to match x86 shuffle ops if needed
  if (!isMMX && V2.getOpcode() != ISD::UNDEF && isCommutedSHUFP(SVOp))
    return CommuteVectorShuffle(SVOp, DAG);

  // The checks below are all present in isShuffleMaskLegal, but they are
  // inlined here right now to enable us to directly emit target specific
  // nodes, and remove one by one until they don't return Op anymore.
  SmallVector<int, 16> M;
  SVOp->getMask(M);

  // Very little shuffling can be done for 64-bit vectors right now.
  if (VT.getSizeInBits() == 64)
    return isPALIGNRMask(M, VT, Subtarget->hasSSSE3()) ? Op : SDValue();

  // FIXME: pshufb, blends, shifts.
  if (VT.getVectorNumElements() == 2 ||
      ShuffleVectorSDNode::isSplatMask(&M[0], VT) ||
      isPALIGNRMask(M, VT, Subtarget->hasSSSE3()))
    return Op;

  if (isPSHUFHWMask(M, VT))
    return getTargetShuffleNode(X86ISD::PSHUFHW, dl, VT, V1,
                                X86::getShufflePSHUFHWImmediate(SVOp),
                                DAG);

  if (isPSHUFLWMask(M, VT))
    return getTargetShuffleNode(X86ISD::PSHUFLW, dl, VT, V1,
                                X86::getShufflePSHUFLWImmediate(SVOp),
                                DAG);

  if (isSHUFPMask(M, VT)) {
    unsigned TargetMask = X86::getShuffleSHUFImmediate(SVOp);
    if (VT == MVT::v4f32 || VT == MVT::v4i32)
      return getTargetShuffleNode(X86ISD::SHUFPS, dl, VT, V1, V2,
                                  TargetMask, DAG);
    if (VT == MVT::v2f64 || VT == MVT::v2i64)
      return getTargetShuffleNode(X86ISD::SHUFPD, dl, VT, V1, V2,
                                  TargetMask, DAG);
  }

  // Handle v8i16 specifically since SSE can do byte extraction and insertion.
  if (VT == MVT::v8i16) {
    SDValue NewOp = LowerVECTOR_SHUFFLEv8i16(Op, DAG);
    if (NewOp.getNode())
      return NewOp;
  }

  if (VT == MVT::v16i8) {
    SDValue NewOp = LowerVECTOR_SHUFFLEv16i8(SVOp, DAG, *this);
    if (NewOp.getNode())
      return NewOp;
  }

  // Handle all 4 wide cases with a number of shuffles except for MMX.
  if (NumElems == 4 && !isMMX)
    return LowerVECTOR_SHUFFLE_4wide(SVOp, DAG);

  return SDValue();
}

SDValue
X86TargetLowering::LowerEXTRACT_VECTOR_ELT_SSE4(SDValue Op,
                                                SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  if (VT.getSizeInBits() == 8) {
    SDValue Extract = DAG.getNode(X86ISD::PEXTRB, dl, MVT::i32,
                                    Op.getOperand(0), Op.getOperand(1));
    SDValue Assert  = DAG.getNode(ISD::AssertZext, dl, MVT::i32, Extract,
                                    DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Assert);
  } else if (VT.getSizeInBits() == 16) {
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    // If Idx is 0, it's cheaper to do a move instead of a pextrw.
    if (Idx == 0)
      return DAG.getNode(ISD::TRUNCATE, dl, MVT::i16,
                         DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                                     DAG.getNode(ISD::BIT_CONVERT, dl,
                                                 MVT::v4i32,
                                                 Op.getOperand(0)),
                                     Op.getOperand(1)));
    SDValue Extract = DAG.getNode(X86ISD::PEXTRW, dl, MVT::i32,
                                    Op.getOperand(0), Op.getOperand(1));
    SDValue Assert  = DAG.getNode(ISD::AssertZext, dl, MVT::i32, Extract,
                                    DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Assert);
  } else if (VT == MVT::f32) {
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
        (User->getOpcode() != ISD::BIT_CONVERT ||
         User->getValueType(0) != MVT::i32))
      return SDValue();
    SDValue Extract = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                                  DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v4i32,
                                              Op.getOperand(0)),
                                              Op.getOperand(1));
    return DAG.getNode(ISD::BIT_CONVERT, dl, MVT::f32, Extract);
  } else if (VT == MVT::i32) {
    // ExtractPS works with constant index.
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
                                     DAG.getNode(ISD::BIT_CONVERT, dl,
                                                 MVT::v4i32, Vec),
                                     Op.getOperand(1)));
    // Transform it so it match pextrw which produces a 32-bit result.
    EVT EltVT = MVT::i32;
    SDValue Extract = DAG.getNode(X86ISD::PEXTRW, dl, EltVT,
                                    Op.getOperand(0), Op.getOperand(1));
    SDValue Assert  = DAG.getNode(ISD::AssertZext, dl, EltVT, Extract,
                                    DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Assert);
  } else if (VT.getSizeInBits() == 32) {
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    if (Idx == 0)
      return Op;

    // SHUFPS the element to the lowest double word, then movss.
    int Mask[4] = { Idx, -1, -1, -1 };
    EVT VVT = Op.getOperand(0).getValueType();
    SDValue Vec = DAG.getVectorShuffle(VVT, dl, Op.getOperand(0),
                                       DAG.getUNDEF(VVT), Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, VT, Vec,
                       DAG.getIntPtrConstant(0));
  } else if (VT.getSizeInBits() == 64) {
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

  if ((EltVT.getSizeInBits() == 8 || EltVT.getSizeInBits() == 16) &&
      isa<ConstantSDNode>(N2)) {
    unsigned Opc;
    if (VT == MVT::v8i16)
      Opc = X86ISD::PINSRW;
    else if (VT == MVT::v4i16)
      Opc = X86ISD::MMX_PINSRW;
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
  } else if (EltVT == MVT::f32 && isa<ConstantSDNode>(N2)) {
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
  } else if (EltVT == MVT::i32 && isa<ConstantSDNode>(N2)) {
    // PINSR* works with constant index.
    return Op;
  }
  return SDValue();
}

SDValue
X86TargetLowering::LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  EVT EltVT = VT.getVectorElementType();

  if (Subtarget->hasSSE41())
    return LowerINSERT_VECTOR_ELT_SSE4(Op, DAG);

  if (EltVT == MVT::i8)
    return SDValue();

  DebugLoc dl = Op.getDebugLoc();
  SDValue N0 = Op.getOperand(0);
  SDValue N1 = Op.getOperand(1);
  SDValue N2 = Op.getOperand(2);

  if (EltVT.getSizeInBits() == 16 && isa<ConstantSDNode>(N2)) {
    // Transform it so it match pinsrw which expects a 16-bit value in a GR32
    // as its second argument.
    if (N1.getValueType() != MVT::i32)
      N1 = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, N1);
    if (N2.getValueType() != MVT::i32)
      N2 = DAG.getIntPtrConstant(cast<ConstantSDNode>(N2)->getZExtValue());
    return DAG.getNode(VT == MVT::v8i16 ? X86ISD::PINSRW : X86ISD::MMX_PINSRW,
                       dl, VT, N0, N1, N2);
  }
  return SDValue();
}

SDValue
X86TargetLowering::LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  
  if (Op.getValueType() == MVT::v1i64 &&
      Op.getOperand(0).getValueType() == MVT::i64)
    return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v1i64, Op.getOperand(0));

  SDValue AnyExt = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, Op.getOperand(0));
  EVT VT = MVT::v2i32;
  switch (Op.getValueType().getSimpleVT().SimpleTy) {
  default: break;
  case MVT::v16i8:
  case MVT::v8i16:
    VT = MVT::v4i32;
    break;
  }
  return DAG.getNode(ISD::BIT_CONVERT, dl, Op.getValueType(),
                     DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, AnyExt));
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
  if (OpFlag) {
    Result = DAG.getNode(ISD::ADD, DL, getPointerTy(),
                         DAG.getNode(X86ISD::GlobalBaseReg,
                                     DebugLoc(), getPointerTy()),
                         Result);
  }

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
      (M == CodeModel::Small || M == CodeModel::Kernel))
    WrapperKind = X86ISD::WrapperRIP;
  else if (Subtarget->isPICStyleGOT())
    OpFlag = X86II::MO_GOTOFF;
  else if (Subtarget->isPICStyleStubPIC())
    OpFlag = X86II::MO_PIC_BASE_OFFSET;

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

  return Result;
}

SDValue
X86TargetLowering::LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const {
  // Create the TargetBlockAddressAddress node.
  unsigned char OpFlags =
    Subtarget->ClassifyBlockAddressReference();
  CodeModel::Model M = getTargetMachine().getCodeModel();
  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();
  DebugLoc dl = Op.getDebugLoc();
  SDValue Result = DAG.getBlockAddress(BA, getPointerTy(),
                                       /*isTarget=*/true, OpFlags);

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
                         PseudoSourceValue::getGOT(), 0, false, false, 0);

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
           unsigned char OperandFlags) {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
  DebugLoc dl = GA->getDebugLoc();
  SDValue TGA = DAG.getTargetGlobalAddress(GA->getGlobal(), dl,
                                           GA->getValueType(0),
                                           GA->getOffset(),
                                           OperandFlags);
  if (InFlag) {
    SDValue Ops[] = { Chain,  TGA, *InFlag };
    Chain = DAG.getNode(X86ISD::TLSADDR, dl, NodeTys, Ops, 3);
  } else {
    SDValue Ops[]  = { Chain, TGA };
    Chain = DAG.getNode(X86ISD::TLSADDR, dl, NodeTys, Ops, 2);
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

// Lower ISD::GlobalTLSAddress using the "initial exec" (for no-pic) or
// "local exec" model.
static SDValue LowerToTLSExecModel(GlobalAddressSDNode *GA, SelectionDAG &DAG,
                                   const EVT PtrVT, TLSModel::Model model,
                                   bool is64Bit) {
  DebugLoc dl = GA->getDebugLoc();
  // Get the Thread Pointer
  SDValue Base = DAG.getNode(X86ISD::SegmentBaseAddress,
                             DebugLoc(), PtrVT,
                             DAG.getRegister(is64Bit? X86::FS : X86::GS,
                                             MVT::i32));

  SDValue ThreadPointer = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), Base,
                                      NULL, 0, false, false, 0);

  unsigned char OperandFlags = 0;
  // Most TLS accesses are not RIP relative, even on x86-64.  One exception is
  // initialexec.
  unsigned WrapperKind = X86ISD::Wrapper;
  if (model == TLSModel::LocalExec) {
    OperandFlags = is64Bit ? X86II::MO_TPOFF : X86II::MO_NTPOFF;
  } else if (is64Bit) {
    assert(model == TLSModel::InitialExec);
    OperandFlags = X86II::MO_GOTTPOFF;
    WrapperKind = X86ISD::WrapperRIP;
  } else {
    assert(model == TLSModel::InitialExec);
    OperandFlags = X86II::MO_INDNTPOFF;
  }

  // emit "addl x@ntpoff,%eax" (local exec) or "addl x@indntpoff,%eax" (initial
  // exec)
  SDValue TGA = DAG.getTargetGlobalAddress(GA->getGlobal(), dl, 
                                           GA->getValueType(0),
                                           GA->getOffset(), OperandFlags);
  SDValue Offset = DAG.getNode(WrapperKind, dl, PtrVT, TGA);

  if (model == TLSModel::InitialExec)
    Offset = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), Offset,
                         PseudoSourceValue::getGOT(), 0, false, false, 0);

  // The address of the thread local variable is the add of the thread
  // pointer with the offset of the variable.
  return DAG.getNode(ISD::ADD, dl, PtrVT, ThreadPointer, Offset);
}

SDValue
X86TargetLowering::LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const {
  
  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = GA->getGlobal();

  if (Subtarget->isTargetELF()) {
    // TODO: implement the "local dynamic" model
    // TODO: implement the "initial exec"model for pic executables
    
    // If GV is an alias then use the aliasee for determining
    // thread-localness.
    if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
      GV = GA->resolveAliasedGlobal(false);
    
    TLSModel::Model model 
      = getTLSModel(GV, getTargetMachine().getRelocationModel());
    
    switch (model) {
      case TLSModel::GeneralDynamic:
      case TLSModel::LocalDynamic: // not implemented
        if (Subtarget->is64Bit())
          return LowerToTLSGeneralDynamicModel64(GA, DAG, getPointerTy());
        return LowerToTLSGeneralDynamicModel32(GA, DAG, getPointerTy());
        
      case TLSModel::InitialExec:
      case TLSModel::LocalExec:
        return LowerToTLSExecModel(GA, DAG, getPointerTy(), model,
                                   Subtarget->is64Bit());
    }
  } else if (Subtarget->isTargetDarwin()) {
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
                                                getPointerTy(),
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
    SDValue Args[] = { Offset };
    SDValue Chain = DAG.getNode(X86ISD::TLSCALL, DL, MVT::Other, Args, 1);
    
    // TLSCALL will be codegen'ed as call. Inform MFI that function has calls.
    MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
    MFI->setAdjustsStack(true);

    // And our return value (tls address) is in the standard call return value
    // location.
    unsigned Reg = Subtarget->is64Bit() ? X86::RAX : X86::EAX;
    return DAG.getCopyFromReg(Chain, DL, Reg, getPointerTy());
  }
  
  assert(false &&
         "TLS not implemented for this target.");

  llvm_unreachable("Unreachable");
  return SDValue();
}


/// LowerShift - Lower SRA_PARTS and friends, which return two i32 values and
/// take a 2 x i32 value to shift plus a shift amount.
SDValue X86TargetLowering::LowerShift(SDValue Op, SelectionDAG &DAG) const {
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

  if (SrcVT.isVector()) {
    if (SrcVT == MVT::v2i32 && Op.getValueType() == MVT::v2f64) {
      return Op;
    }
    return SDValue();
  }

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
                               PseudoSourceValue::getFixedStack(SSFI), 0,
                               false, false, 0);
  return BuildFILD(Op, SrcVT, Chain, StackSlot, DAG);
}

SDValue X86TargetLowering::BuildFILD(SDValue Op, EVT SrcVT, SDValue Chain,
                                     SDValue StackSlot, 
                                     SelectionDAG &DAG) const {
  // Build the FILD
  DebugLoc dl = Op.getDebugLoc();
  SDVTList Tys;
  bool useSSE = isScalarFPTypeInSSEReg(Op.getValueType());
  if (useSSE)
    Tys = DAG.getVTList(MVT::f64, MVT::Other, MVT::Flag);
  else
    Tys = DAG.getVTList(Op.getValueType(), MVT::Other);
  SDValue Ops[] = { Chain, StackSlot, DAG.getValueType(SrcVT) };
  SDValue Result = DAG.getNode(useSSE ? X86ISD::FILD_FLAG : X86ISD::FILD, dl,
                               Tys, Ops, array_lengthof(Ops));

  if (useSSE) {
    Chain = Result.getValue(1);
    SDValue InFlag = Result.getValue(2);

    // FIXME: Currently the FST is flagged to the FILD_FLAG. This
    // shouldn't be necessary except that RFP cannot be live across
    // multiple blocks. When stackifier is fixed, they can be uncoupled.
    MachineFunction &MF = DAG.getMachineFunction();
    int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8, false);
    SDValue StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
    Tys = DAG.getVTList(MVT::Other);
    SDValue Ops[] = {
      Chain, Result, StackSlot, DAG.getValueType(Op.getValueType()), InFlag
    };
    Chain = DAG.getNode(X86ISD::FST, dl, Tys, Ops, array_lengthof(Ops));
    Result = DAG.getLoad(Op.getValueType(), dl, Chain, StackSlot,
                         PseudoSourceValue::getFixedStack(SSFI), 0,
                         false, false, 0);
  }

  return Result;
}

// LowerUINT_TO_FP_i64 - 64-bit unsigned integer to double expansion.
SDValue X86TargetLowering::LowerUINT_TO_FP_i64(SDValue Op,
                                               SelectionDAG &DAG) const {
  // This algorithm is not obvious. Here it is in C code, more or less:
  /*
    double uint64_to_double( uint32_t hi, uint32_t lo ) {
      static const __m128i exp = { 0x4330000045300000ULL, 0 };
      static const __m128d bias = { 0x1.0p84, 0x1.0p52 };

      // Copy ints to xmm registers.
      __m128i xh = _mm_cvtsi32_si128( hi );
      __m128i xl = _mm_cvtsi32_si128( lo );

      // Combine into low half of a single xmm register.
      __m128i x = _mm_unpacklo_epi32( xh, xl );
      __m128d d;
      double sd;

      // Merge in appropriate exponents to give the integer bits the right
      // magnitude.
      x = _mm_unpacklo_epi32( x, exp );

      // Subtract away the biases to deal with the IEEE-754 double precision
      // implicit 1.
      d = _mm_sub_pd( (__m128d) x, bias );

      // All conversions up to here are exact. The correctly rounded result is
      // calculated using the current rounding mode using the following
      // horizontal add.
      d = _mm_add_sd( d, _mm_unpackhi_pd( d, d ) );
      _mm_store_sd( &sd, d );   // Because we are returning doubles in XMM, this
                                // store doesn't really need to be here (except
                                // maybe to zero the other double)
      return sd;
    }
  */

  DebugLoc dl = Op.getDebugLoc();
  LLVMContext *Context = DAG.getContext();

  // Build some magic constants.
  std::vector<Constant*> CV0;
  CV0.push_back(ConstantInt::get(*Context, APInt(32, 0x45300000)));
  CV0.push_back(ConstantInt::get(*Context, APInt(32, 0x43300000)));
  CV0.push_back(ConstantInt::get(*Context, APInt(32, 0)));
  CV0.push_back(ConstantInt::get(*Context, APInt(32, 0)));
  Constant *C0 = ConstantVector::get(CV0);
  SDValue CPIdx0 = DAG.getConstantPool(C0, getPointerTy(), 16);

  std::vector<Constant*> CV1;
  CV1.push_back(
    ConstantFP::get(*Context, APFloat(APInt(64, 0x4530000000000000ULL))));
  CV1.push_back(
    ConstantFP::get(*Context, APFloat(APInt(64, 0x4330000000000000ULL))));
  Constant *C1 = ConstantVector::get(CV1);
  SDValue CPIdx1 = DAG.getConstantPool(C1, getPointerTy(), 16);

  SDValue XR1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4i32,
                            DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                                        Op.getOperand(0),
                                        DAG.getIntPtrConstant(1)));
  SDValue XR2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4i32,
                            DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                                        Op.getOperand(0),
                                        DAG.getIntPtrConstant(0)));
  SDValue Unpck1 = getUnpackl(DAG, dl, MVT::v4i32, XR1, XR2);
  SDValue CLod0 = DAG.getLoad(MVT::v4i32, dl, DAG.getEntryNode(), CPIdx0,
                              PseudoSourceValue::getConstantPool(), 0,
                              false, false, 16);
  SDValue Unpck2 = getUnpackl(DAG, dl, MVT::v4i32, Unpck1, CLod0);
  SDValue XR2F = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2f64, Unpck2);
  SDValue CLod1 = DAG.getLoad(MVT::v2f64, dl, CLod0.getValue(1), CPIdx1,
                              PseudoSourceValue::getConstantPool(), 0,
                              false, false, 16);
  SDValue Sub = DAG.getNode(ISD::FSUB, dl, MVT::v2f64, XR2F, CLod1);

  // Add the halves; easiest way is to swap them into another reg first.
  int ShufMask[2] = { 1, -1 };
  SDValue Shuf = DAG.getVectorShuffle(MVT::v2f64, dl, Sub,
                                      DAG.getUNDEF(MVT::v2f64), ShufMask);
  SDValue Add = DAG.getNode(ISD::FADD, dl, MVT::v2f64, Shuf, Sub);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64, Add,
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
                             DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                                         Op.getOperand(0),
                                         DAG.getIntPtrConstant(0)));

  Load = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64,
                     DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2f64, Load),
                     DAG.getIntPtrConstant(0));

  // Or the load with the bias.
  SDValue Or = DAG.getNode(ISD::OR, dl, MVT::v2i64,
                           DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2i64,
                                       DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                                   MVT::v2f64, Load)),
                           DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2i64,
                                       DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                                   MVT::v2f64, Bias)));
  Or = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::f64,
                   DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2f64, Or),
                   DAG.getIntPtrConstant(0));

  // Subtract the bias.
  SDValue Sub = DAG.getNode(ISD::FSUB, dl, MVT::f64, Or, Bias);

  // Handle final rounding.
  EVT DestVT = Op.getValueType();

  if (DestVT.bitsLT(MVT::f64)) {
    return DAG.getNode(ISD::FP_ROUND, dl, DestVT, Sub,
                       DAG.getIntPtrConstant(0));
  } else if (DestVT.bitsGT(MVT::f64)) {
    return DAG.getNode(ISD::FP_EXTEND, dl, DestVT, Sub);
  }

  // Handle final rounding.
  return Sub;
}

SDValue X86TargetLowering::LowerUINT_TO_FP(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDValue N0 = Op.getOperand(0);
  DebugLoc dl = Op.getDebugLoc();

  // Since UINT_TO_FP is legal (it's marked custom), dag combiner won't
  // optimize it to a SINT_TO_FP when the sign bit is known zero. Perform
  // the optimization here.
  if (DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::SINT_TO_FP, dl, Op.getValueType(), N0);

  EVT SrcVT = N0.getValueType();
  EVT DstVT = Op.getValueType();
  if (SrcVT == MVT::i64 && DstVT == MVT::f64 && X86ScalarSSEf64)
    return LowerUINT_TO_FP_i64(Op, DAG);
  else if (SrcVT == MVT::i32 && X86ScalarSSEf64)
    return LowerUINT_TO_FP_i32(Op, DAG);

  // Make a 64-bit buffer, and use it to build an FILD.
  SDValue StackSlot = DAG.CreateStackTemporary(MVT::i64);
  if (SrcVT == MVT::i32) {
    SDValue WordOff = DAG.getConstant(4, getPointerTy());
    SDValue OffsetSlot = DAG.getNode(ISD::ADD, dl,
                                     getPointerTy(), StackSlot, WordOff);
    SDValue Store1 = DAG.getStore(DAG.getEntryNode(), dl, Op.getOperand(0),
                                  StackSlot, NULL, 0, false, false, 0);
    SDValue Store2 = DAG.getStore(Store1, dl, DAG.getConstant(0, MVT::i32),
                                  OffsetSlot, NULL, 0, false, false, 0);
    SDValue Fild = BuildFILD(Op, MVT::i64, Store2, StackSlot, DAG);
    return Fild;
  }

  assert(SrcVT == MVT::i64 && "Unexpected type in UINT_TO_FP");
  SDValue Store = DAG.getStore(DAG.getEntryNode(), dl, Op.getOperand(0),
                                StackSlot, NULL, 0, false, false, 0);
  // For i64 source, we need to add the appropriate power of 2 if the input
  // was negative.  This is the same as the optimization in
  // DAGTypeLegalizer::ExpandIntOp_UNIT_TO_FP, and for it to be safe here,
  // we must be careful to do the computation in x87 extended precision, not
  // in SSE. (The generic code can't know it's OK to do this, or how to.)
  SDVTList Tys = DAG.getVTList(MVT::f80, MVT::Other);
  SDValue Ops[] = { Store, StackSlot, DAG.getValueType(MVT::i64) };
  SDValue Fild = DAG.getNode(X86ISD::FILD, dl, Tys, Ops, 3);

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
  SDValue Fudge = DAG.getExtLoad(ISD::EXTLOAD, MVT::f80, dl, DAG.getEntryNode(),
                                 FudgePtr, PseudoSourceValue::getConstantPool(),
                                 0, MVT::f32, false, false, 4);
  // Extend everything to 80 bits to force it to be done on x87.
  SDValue Add = DAG.getNode(ISD::FADD, dl, MVT::f80, Fild, Fudge);
  return DAG.getNode(ISD::FP_ROUND, dl, DstVT, Add, DAG.getIntPtrConstant(0));
}

std::pair<SDValue,SDValue> X86TargetLowering::
FP_TO_INTHelper(SDValue Op, SelectionDAG &DAG, bool IsSigned) const {
  DebugLoc dl = Op.getDebugLoc();

  EVT DstTy = Op.getValueType();

  if (!IsSigned) {
    assert(DstTy == MVT::i32 && "Unexpected FP_TO_UINT");
    DstTy = MVT::i64;
  }

  assert(DstTy.getSimpleVT() <= MVT::i64 &&
         DstTy.getSimpleVT() >= MVT::i16 &&
         "Unknown FP_TO_SINT to lower!");

  // These are really Legal.
  if (DstTy == MVT::i32 &&
      isScalarFPTypeInSSEReg(Op.getOperand(0).getValueType()))
    return std::make_pair(SDValue(), SDValue());
  if (Subtarget->is64Bit() &&
      DstTy == MVT::i64 &&
      isScalarFPTypeInSSEReg(Op.getOperand(0).getValueType()))
    return std::make_pair(SDValue(), SDValue());

  // We lower FP->sint64 into FISTP64, followed by a load, all to a temporary
  // stack slot.
  MachineFunction &MF = DAG.getMachineFunction();
  unsigned MemSize = DstTy.getSizeInBits()/8;
  int SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize, false);
  SDValue StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());

  unsigned Opc;
  switch (DstTy.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("Invalid FP_TO_SINT to lower!");
  case MVT::i16: Opc = X86ISD::FP_TO_INT16_IN_MEM; break;
  case MVT::i32: Opc = X86ISD::FP_TO_INT32_IN_MEM; break;
  case MVT::i64: Opc = X86ISD::FP_TO_INT64_IN_MEM; break;
  }

  SDValue Chain = DAG.getEntryNode();
  SDValue Value = Op.getOperand(0);
  if (isScalarFPTypeInSSEReg(Op.getOperand(0).getValueType())) {
    assert(DstTy == MVT::i64 && "Invalid FP_TO_SINT to lower!");
    Chain = DAG.getStore(Chain, dl, Value, StackSlot,
                         PseudoSourceValue::getFixedStack(SSFI), 0,
                         false, false, 0);
    SDVTList Tys = DAG.getVTList(Op.getOperand(0).getValueType(), MVT::Other);
    SDValue Ops[] = {
      Chain, StackSlot, DAG.getValueType(Op.getOperand(0).getValueType())
    };
    Value = DAG.getNode(X86ISD::FLD, dl, Tys, Ops, 3);
    Chain = Value.getValue(1);
    SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize, false);
    StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
  }

  // Build the FP_TO_INT*_IN_MEM
  SDValue Ops[] = { Chain, Value, StackSlot };
  SDValue FIST = DAG.getNode(Opc, dl, MVT::Other, Ops, 3);

  return std::make_pair(FIST, StackSlot);
}

SDValue X86TargetLowering::LowerFP_TO_SINT(SDValue Op,
                                           SelectionDAG &DAG) const {
  if (Op.getValueType().isVector()) {
    if (Op.getValueType() == MVT::v2i32 &&
        Op.getOperand(0).getValueType() == MVT::v2f64) {
      return Op;
    }
    return SDValue();
  }

  std::pair<SDValue,SDValue> Vals = FP_TO_INTHelper(Op, DAG, true);
  SDValue FIST = Vals.first, StackSlot = Vals.second;
  // If FP_TO_INTHelper failed, the node is actually supposed to be Legal.
  if (FIST.getNode() == 0) return Op;

  // Load the result.
  return DAG.getLoad(Op.getValueType(), Op.getDebugLoc(),
                     FIST, StackSlot, NULL, 0, false, false, 0);
}

SDValue X86TargetLowering::LowerFP_TO_UINT(SDValue Op,
                                           SelectionDAG &DAG) const {
  std::pair<SDValue,SDValue> Vals = FP_TO_INTHelper(Op, DAG, false);
  SDValue FIST = Vals.first, StackSlot = Vals.second;
  assert(FIST.getNode() && "Unexpected failure");

  // Load the result.
  return DAG.getLoad(Op.getValueType(), Op.getDebugLoc(),
                     FIST, StackSlot, NULL, 0, false, false, 0);
}

SDValue X86TargetLowering::LowerFABS(SDValue Op,
                                     SelectionDAG &DAG) const {
  LLVMContext *Context = DAG.getContext();
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT EltVT = VT;
  if (VT.isVector())
    EltVT = VT.getVectorElementType();
  std::vector<Constant*> CV;
  if (EltVT == MVT::f64) {
    Constant *C = ConstantFP::get(*Context, APFloat(APInt(64, ~(1ULL << 63))));
    CV.push_back(C);
    CV.push_back(C);
  } else {
    Constant *C = ConstantFP::get(*Context, APFloat(APInt(32, ~(1U << 31))));
    CV.push_back(C);
    CV.push_back(C);
    CV.push_back(C);
    CV.push_back(C);
  }
  Constant *C = ConstantVector::get(CV);
  SDValue CPIdx = DAG.getConstantPool(C, getPointerTy(), 16);
  SDValue Mask = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                             PseudoSourceValue::getConstantPool(), 0,
                             false, false, 16);
  return DAG.getNode(X86ISD::FAND, dl, VT, Op.getOperand(0), Mask);
}

SDValue X86TargetLowering::LowerFNEG(SDValue Op, SelectionDAG &DAG) const {
  LLVMContext *Context = DAG.getContext();
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  EVT EltVT = VT;
  if (VT.isVector())
    EltVT = VT.getVectorElementType();
  std::vector<Constant*> CV;
  if (EltVT == MVT::f64) {
    Constant *C = ConstantFP::get(*Context, APFloat(APInt(64, 1ULL << 63)));
    CV.push_back(C);
    CV.push_back(C);
  } else {
    Constant *C = ConstantFP::get(*Context, APFloat(APInt(32, 1U << 31)));
    CV.push_back(C);
    CV.push_back(C);
    CV.push_back(C);
    CV.push_back(C);
  }
  Constant *C = ConstantVector::get(CV);
  SDValue CPIdx = DAG.getConstantPool(C, getPointerTy(), 16);
  SDValue Mask = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                             PseudoSourceValue::getConstantPool(), 0,
                             false, false, 16);
  if (VT.isVector()) {
    return DAG.getNode(ISD::BIT_CONVERT, dl, VT,
                       DAG.getNode(ISD::XOR, dl, MVT::v2i64,
                    DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2i64,
                                Op.getOperand(0)),
                    DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v2i64, Mask)));
  } else {
    return DAG.getNode(X86ISD::FXOR, dl, VT, Op.getOperand(0), Mask);
  }
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
  std::vector<Constant*> CV;
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
                              PseudoSourceValue::getConstantPool(), 0,
                              false, false, 16);
  SDValue SignBit = DAG.getNode(X86ISD::FAND, dl, SrcVT, Op1, Mask1);

  // Shift sign bit right or left if the two operands have different types.
  if (SrcVT.bitsGT(VT)) {
    // Op0 is MVT::f32, Op1 is MVT::f64.
    SignBit = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2f64, SignBit);
    SignBit = DAG.getNode(X86ISD::FSRL, dl, MVT::v2f64, SignBit,
                          DAG.getConstant(32, MVT::i32));
    SignBit = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v4f32, SignBit);
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
                              PseudoSourceValue::getConstantPool(), 0,
                              false, false, 16);
  SDValue Val = DAG.getNode(X86ISD::FAND, dl, VT, Op0, Mask2);

  // Or the value with the sign bit.
  return DAG.getNode(X86ISD::FOR, dl, VT, Val, SignBit);
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
  switch (Op.getNode()->getOpcode()) {
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
      if (UI->getOpcode() != ISD::CopyToReg && UI->getOpcode() != ISD::SETCC)
        goto default_case;

    if (ConstantSDNode *C =
        dyn_cast<ConstantSDNode>(Op.getNode()->getOperand(1))) {
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
          (User->getOpcode() != ISD::SELECT || UOpNo != 0)) {
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
    switch (Op.getNode()->getOpcode()) {
    default: llvm_unreachable("unexpected operator!");
    case ISD::SUB: Opcode = X86ISD::SUB; break;
    case ISD::OR:  Opcode = X86ISD::OR;  break;
    case ISD::XOR: Opcode = X86ISD::XOR; break;
    case ISD::AND: Opcode = X86ISD::AND; break;
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
  return DAG.getNode(X86ISD::CMP, dl, MVT::i32, Op0, Op1);
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
          APInt Mask = APInt::getAllOnesValue(BitWidth), Zeros, Ones;
          DAG.ComputeMaskedBits(Op0, Mask, Zeros, Ones);
          if (Zeros.countLeadingOnes() < BitWidth - AndBitWidth)
            return SDValue();
        }
        LHS = Op1;
        RHS = Op0.getOperand(1);
      }
  } else if (Op1.getOpcode() == ISD::Constant) {
    ConstantSDNode *AndRHS = cast<ConstantSDNode>(Op1);
    SDValue AndLHS = Op0;
    if (AndRHS->getZExtValue() == 1 && AndLHS.getOpcode() == ISD::SRL) {
      LHS = AndLHS.getOperand(0);
      RHS = AndLHS.getOperand(1);
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
  assert(Op.getValueType() == MVT::i8 && "SetCC type must be 8-bit integer");
  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();

  // Optimize to BT if possible.
  // Lower (X & (1 << N)) == 0 to BT(X, N).
  // Lower ((X >>u N) & 1) != 0 to BT(X, N).
  // Lower ((X >>s N) & 1) != 0 to BT(X, N).
  if (Op0.getOpcode() == ISD::AND &&
      Op0.hasOneUse() &&
      Op1.getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(Op1)->isNullValue() &&
      (CC == ISD::SETEQ || CC == ISD::SETNE)) {
    SDValue NewSetCC = LowerToBT(Op0, CC, dl, DAG);
    if (NewSetCC.getNode())
      return NewSetCC;
  }

  // Look for "(setcc) == / != 1" to avoid unncessary setcc.
  if (Op0.getOpcode() == X86ISD::SETCC &&
      Op1.getOpcode() == ISD::Constant &&
      (cast<ConstantSDNode>(Op1)->getZExtValue() == 1 ||
       cast<ConstantSDNode>(Op1)->isNullValue()) &&
      (CC == ISD::SETEQ || CC == ISD::SETNE)) {
    X86::CondCode CCode = (X86::CondCode)Op0.getConstantOperandVal(0);
    bool Invert = (CC == ISD::SETNE) ^
      cast<ConstantSDNode>(Op1)->isNullValue();
    if (Invert)
      CCode = X86::GetOppositeBranchCondition(CCode);
    return DAG.getNode(X86ISD::SETCC, dl, MVT::i8,
                       DAG.getConstant(CCode, MVT::i8), Op0.getOperand(1));
  }

  bool isFP = Op1.getValueType().isFloatingPoint();
  unsigned X86CC = TranslateX86CC(CC, isFP, Op0, Op1, DAG);
  if (X86CC == X86::COND_INVALID)
    return SDValue();

  SDValue Cond = EmitCmp(Op0, Op1, X86CC, DAG);

  // Use sbb x, x to materialize carry bit into a GPR.
  if (X86CC == X86::COND_B)
    return DAG.getNode(ISD::AND, dl, MVT::i8,
                       DAG.getNode(X86ISD::SETCC_CARRY, dl, MVT::i8,
                                   DAG.getConstant(X86CC, MVT::i8), Cond),
                       DAG.getConstant(1, MVT::i8));

  return DAG.getNode(X86ISD::SETCC, dl, MVT::i8,
                     DAG.getConstant(X86CC, MVT::i8), Cond);
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
    unsigned SSECC = 8;
    EVT VT0 = Op0.getValueType();
    assert(VT0 == MVT::v4f32 || VT0 == MVT::v2f64);
    unsigned Opc = VT0 == MVT::v4f32 ? X86ISD::CMPPS : X86ISD::CMPPD;
    bool Swap = false;

    switch (SetCCOpcode) {
    default: break;
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
    case ISD::SETULE: Swap = true;
    case ISD::SETUGE: SSECC = 5; break;
    case ISD::SETULT: Swap = true;
    case ISD::SETUGT: SSECC = 6; break;
    case ISD::SETO:   SSECC = 7; break;
    }
    if (Swap)
      std::swap(Op0, Op1);

    // In the two special cases we can't handle, emit two comparisons.
    if (SSECC == 8) {
      if (SetCCOpcode == ISD::SETUEQ) {
        SDValue UNORD, EQ;
        UNORD = DAG.getNode(Opc, dl, VT, Op0, Op1, DAG.getConstant(3, MVT::i8));
        EQ = DAG.getNode(Opc, dl, VT, Op0, Op1, DAG.getConstant(0, MVT::i8));
        return DAG.getNode(ISD::OR, dl, VT, UNORD, EQ);
      }
      else if (SetCCOpcode == ISD::SETONE) {
        SDValue ORD, NEQ;
        ORD = DAG.getNode(Opc, dl, VT, Op0, Op1, DAG.getConstant(7, MVT::i8));
        NEQ = DAG.getNode(Opc, dl, VT, Op0, Op1, DAG.getConstant(4, MVT::i8));
        return DAG.getNode(ISD::AND, dl, VT, ORD, NEQ);
      }
      llvm_unreachable("Illegal FP comparison");
    }
    // Handle all other FP comparisons here.
    return DAG.getNode(Opc, dl, VT, Op0, Op1, DAG.getConstant(SSECC, MVT::i8));
  }

  // We are handling one of the integer comparisons here.  Since SSE only has
  // GT and EQ comparisons for integer, swapping operands and multiple
  // operations may be required for some comparisons.
  unsigned Opc = 0, EQOpc = 0, GTOpc = 0;
  bool Swap = false, Invert = false, FlipSigns = false;

  switch (VT.getSimpleVT().SimpleTy) {
  default: break;
  case MVT::v8i8:
  case MVT::v16i8: EQOpc = X86ISD::PCMPEQB; GTOpc = X86ISD::PCMPGTB; break;
  case MVT::v4i16:
  case MVT::v8i16: EQOpc = X86ISD::PCMPEQW; GTOpc = X86ISD::PCMPGTW; break;
  case MVT::v2i32:
  case MVT::v4i32: EQOpc = X86ISD::PCMPEQD; GTOpc = X86ISD::PCMPGTD; break;
  case MVT::v2i64: EQOpc = X86ISD::PCMPEQQ; GTOpc = X86ISD::PCMPGTQ; break;
  }

  switch (SetCCOpcode) {
  default: break;
  case ISD::SETNE:  Invert = true;
  case ISD::SETEQ:  Opc = EQOpc; break;
  case ISD::SETLT:  Swap = true;
  case ISD::SETGT:  Opc = GTOpc; break;
  case ISD::SETGE:  Swap = true;
  case ISD::SETLE:  Opc = GTOpc; Invert = true; break;
  case ISD::SETULT: Swap = true;
  case ISD::SETUGT: Opc = GTOpc; FlipSigns = true; break;
  case ISD::SETUGE: Swap = true;
  case ISD::SETULE: Opc = GTOpc; FlipSigns = true; Invert = true; break;
  }
  if (Swap)
    std::swap(Op0, Op1);

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
  if (Opc == X86ISD::CMP || Opc == X86ISD::COMI || Opc == X86ISD::UCOMI)
    return true;
  if (Op.getResNo() == 1 &&
      (Opc == X86ISD::ADD ||
       Opc == X86ISD::SUB ||
       Opc == X86ISD::SMUL ||
       Opc == X86ISD::UMUL ||
       Opc == X86ISD::INC ||
       Opc == X86ISD::DEC ||
       Opc == X86ISD::OR ||
       Opc == X86ISD::XOR ||
       Opc == X86ISD::AND))
    return true;

  return false;
}

SDValue X86TargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  bool addTest = true;
  SDValue Cond  = Op.getOperand(0);
  DebugLoc dl = Op.getDebugLoc();
  SDValue CC;

  if (Cond.getOpcode() == ISD::SETCC) {
    SDValue NewCond = LowerSETCC(Cond, DAG);
    if (NewCond.getNode())
      Cond = NewCond;
  }

  // (select (x == 0), -1, 0) -> (sign_bit (x - 1))
  SDValue Op1 = Op.getOperand(1);
  SDValue Op2 = Op.getOperand(2);
  if (Cond.getOpcode() == X86ISD::SETCC &&
      cast<ConstantSDNode>(Cond.getOperand(0))->getZExtValue() == X86::COND_E) {
    SDValue Cmp = Cond.getOperand(1);
    if (Cmp.getOpcode() == X86ISD::CMP) {
      ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(Op1);
      ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(Op2);
      ConstantSDNode *RHSC =
        dyn_cast<ConstantSDNode>(Cmp.getOperand(1).getNode());
      if (N1C && N1C->isAllOnesValue() &&
          N2C && N2C->isNullValue() &&
          RHSC && RHSC->isNullValue()) {
        SDValue CmpOp0 = Cmp.getOperand(0);
        Cmp = DAG.getNode(X86ISD::CMP, dl, MVT::i32,
                          CmpOp0, DAG.getConstant(1, CmpOp0.getValueType()));
        return DAG.getNode(X86ISD::SETCC_CARRY, dl, Op.getValueType(),
                           DAG.getConstant(X86::COND_B, MVT::i8), Cmp);
      }
    }
  }

  // Look pass (and (setcc_carry (cmp ...)), 1).
  if (Cond.getOpcode() == ISD::AND &&
      Cond.getOperand(0).getOpcode() == X86ISD::SETCC_CARRY) {
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Cond.getOperand(1));
    if (C && C->getAPIntValue() == 1) 
      Cond = Cond.getOperand(0);
  }

  // If condition flag is set by a X86ISD::CMP, then use it as the condition
  // setting operand in place of the X86ISD::SETCC.
  if (Cond.getOpcode() == X86ISD::SETCC ||
      Cond.getOpcode() == X86ISD::SETCC_CARRY) {
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
  }

  if (addTest) {
    // Look pass the truncate.
    if (Cond.getOpcode() == ISD::TRUNCATE)
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

  // X86ISD::CMOV means set the result (which is operand 1) to the RHS if
  // condition is true.
  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Flag);
  SDValue Ops[] = { Op2, Op1, CC, Cond };
  return DAG.getNode(X86ISD::CMOV, dl, VTs, Ops, array_lengthof(Ops));
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

  if (Cond.getOpcode() == ISD::SETCC) {
    SDValue NewCond = LowerSETCC(Cond, DAG);
    if (NewCond.getNode())
      Cond = NewCond;
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
  if (Cond.getOpcode() == X86ISD::SETCC ||
      Cond.getOpcode() == X86ISD::SETCC_CARRY) {
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
    }
  }

  if (addTest) {
    // Look pass the truncate.
    if (Cond.getOpcode() == ISD::TRUNCATE)
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
  assert(Subtarget->isTargetCygMing() &&
         "This should be used only on Cygwin/Mingw targets");
  DebugLoc dl = Op.getDebugLoc();

  // Get the inputs.
  SDValue Chain = Op.getOperand(0);
  SDValue Size  = Op.getOperand(1);
  // FIXME: Ensure alignment here

  SDValue Flag;

  EVT SPTy = Subtarget->is64Bit() ? MVT::i64 : MVT::i32;

  Chain = DAG.getCopyToReg(Chain, dl, X86::EAX, Size, Flag);
  Flag = Chain.getValue(1);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);

  Chain = DAG.getNode(X86ISD::MINGW_ALLOCA, dl, NodeTys, Chain, Flag);
  Flag = Chain.getValue(1);

  Chain = DAG.getCopyFromReg(Chain, dl, X86StackPtr, SPTy).getValue(1);

  SDValue Ops1[2] = { Chain.getValue(0), Chain };
  return DAG.getMergeValues(Ops1, 2, dl);
}

SDValue X86TargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();

  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  DebugLoc dl = Op.getDebugLoc();

  if (!Subtarget->is64Bit()) {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    SDValue FR = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                   getPointerTy());
    return DAG.getStore(Op.getOperand(0), dl, FR, Op.getOperand(1), SV, 0,
                        false, false, 0);
  }

  // __va_list_tag:
  //   gp_offset         (0 - 6 * 8)
  //   fp_offset         (48 - 48 + 8 * 16)
  //   overflow_arg_area (point to parameters coming in memory).
  //   reg_save_area
  SmallVector<SDValue, 8> MemOps;
  SDValue FIN = Op.getOperand(1);
  // Store gp_offset
  SDValue Store = DAG.getStore(Op.getOperand(0), dl,
                               DAG.getConstant(FuncInfo->getVarArgsGPOffset(),
                                               MVT::i32),
                               FIN, SV, 0, false, false, 0);
  MemOps.push_back(Store);

  // Store fp_offset
  FIN = DAG.getNode(ISD::ADD, dl, getPointerTy(),
                    FIN, DAG.getIntPtrConstant(4));
  Store = DAG.getStore(Op.getOperand(0), dl,
                       DAG.getConstant(FuncInfo->getVarArgsFPOffset(),
                                       MVT::i32),
                       FIN, SV, 4, false, false, 0);
  MemOps.push_back(Store);

  // Store ptr to overflow_arg_area
  FIN = DAG.getNode(ISD::ADD, dl, getPointerTy(),
                    FIN, DAG.getIntPtrConstant(4));
  SDValue OVFIN = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                    getPointerTy());
  Store = DAG.getStore(Op.getOperand(0), dl, OVFIN, FIN, SV, 8,
                       false, false, 0);
  MemOps.push_back(Store);

  // Store ptr to reg_save_area.
  FIN = DAG.getNode(ISD::ADD, dl, getPointerTy(),
                    FIN, DAG.getIntPtrConstant(8));
  SDValue RSFIN = DAG.getFrameIndex(FuncInfo->getRegSaveFrameIndex(),
                                    getPointerTy());
  Store = DAG.getStore(Op.getOperand(0), dl, RSFIN, FIN, SV, 16,
                       false, false, 0);
  MemOps.push_back(Store);
  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                     &MemOps[0], MemOps.size());
}

SDValue X86TargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  // X86-64 va_list is a struct { i32, i32, i8*, i8* }.
  assert(Subtarget->is64Bit() && "This code only handles 64-bit va_arg!");

  report_fatal_error("VAArgInst is not yet implemented for x86-64!");
  return SDValue();
}

SDValue X86TargetLowering::LowerVACOPY(SDValue Op, SelectionDAG &DAG) const {
  // X86-64 va_list is a struct { i32, i32, i8*, i8* }.
  assert(Subtarget->is64Bit() && "This code only handles 64-bit va_copy!");
  SDValue Chain = Op.getOperand(0);
  SDValue DstPtr = Op.getOperand(1);
  SDValue SrcPtr = Op.getOperand(2);
  const Value *DstSV = cast<SrcValueSDNode>(Op.getOperand(3))->getValue();
  const Value *SrcSV = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();
  DebugLoc dl = Op.getDebugLoc();

  return DAG.getMemcpy(Chain, dl, DstPtr, SrcPtr,
                       DAG.getIntPtrConstant(24), 8, /*isVolatile*/false,
                       false, DstSV, 0, SrcSV, 0);
}

SDValue
X86TargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const {
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

    SDValue LHS = Op.getOperand(1);
    SDValue RHS = Op.getOperand(2);
    unsigned X86CC = TranslateX86CC(CC, true, LHS, RHS, DAG);
    assert(X86CC != X86::COND_INVALID && "Unexpected illegal condition!");
    SDValue Cond = DAG.getNode(Opc, dl, MVT::i32, LHS, RHS);
    SDValue SetCC = DAG.getNode(X86ISD::SETCC, dl, MVT::i8,
                                DAG.getConstant(X86CC, MVT::i8), Cond);
    return DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, SetCC);
  }
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
    unsigned X86CC = 0;
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
    SDValue ShAmt = Op.getOperand(2);
    if (isa<ConstantSDNode>(ShAmt))
      return SDValue();

    unsigned NewIntNo = 0;
    EVT ShAmtVT = MVT::v4i32;
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
      default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
      }
      break;
    }
    }

    // The vector shift intrinsics with scalars uses 32b shift amounts but
    // the sse2/mmx shift instructions reads 64 bits. Set the upper 32 bits
    // to be zero.
    SDValue ShOps[4];
    ShOps[0] = ShAmt;
    ShOps[1] = DAG.getConstant(0, MVT::i32);
    if (ShAmtVT == MVT::v4i32) {
      ShOps[2] = DAG.getUNDEF(MVT::i32);
      ShOps[3] = DAG.getUNDEF(MVT::i32);
      ShAmt =  DAG.getNode(ISD::BUILD_VECTOR, dl, ShAmtVT, &ShOps[0], 4);
    } else {
      ShAmt =  DAG.getNode(ISD::BUILD_VECTOR, dl, ShAmtVT, &ShOps[0], 2);
    }

    EVT VT = Op.getValueType();
    ShAmt = DAG.getNode(ISD::BIT_CONVERT, dl, VT, ShAmt);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(NewIntNo, MVT::i32),
                       Op.getOperand(1), ShAmt);
  }
  }
}

SDValue X86TargetLowering::LowerRETURNADDR(SDValue Op,
                                           SelectionDAG &DAG) const {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setReturnAddressIsTaken(true);

  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  DebugLoc dl = Op.getDebugLoc();

  if (Depth > 0) {
    SDValue FrameAddr = LowerFRAMEADDR(Op, DAG);
    SDValue Offset =
      DAG.getConstant(TD->getPointerSize(),
                      Subtarget->is64Bit() ? MVT::i64 : MVT::i32);
    return DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, dl, getPointerTy(),
                                   FrameAddr, Offset),
                       NULL, 0, false, false, 0);
  }

  // Just load the return address.
  SDValue RetAddrFI = getReturnAddressFrameIndex(DAG);
  return DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(),
                     RetAddrFI, NULL, 0, false, false, 0);
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
    FrameAddr = DAG.getLoad(VT, dl, DAG.getEntryNode(), FrameAddr, NULL, 0,
                            false, false, 0);
  return FrameAddr;
}

SDValue X86TargetLowering::LowerFRAME_TO_ARGS_OFFSET(SDValue Op,
                                                     SelectionDAG &DAG) const {
  return DAG.getIntPtrConstant(2*TD->getPointerSize());
}

SDValue X86TargetLowering::LowerEH_RETURN(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SDValue Chain     = Op.getOperand(0);
  SDValue Offset    = Op.getOperand(1);
  SDValue Handler   = Op.getOperand(2);
  DebugLoc dl       = Op.getDebugLoc();

  SDValue Frame = DAG.getCopyFromReg(DAG.getEntryNode(), dl,
                                     Subtarget->is64Bit() ? X86::RBP : X86::EBP,
                                     getPointerTy());
  unsigned StoreAddrReg = (Subtarget->is64Bit() ? X86::RCX : X86::ECX);

  SDValue StoreAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), Frame,
                                  DAG.getIntPtrConstant(TD->getPointerSize()));
  StoreAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), StoreAddr, Offset);
  Chain = DAG.getStore(Chain, dl, Handler, StoreAddr, NULL, 0, false, false, 0);
  Chain = DAG.getCopyToReg(Chain, dl, StoreAddrReg, StoreAddr);
  MF.getRegInfo().addLiveOut(StoreAddrReg);

  return DAG.getNode(X86ISD::EH_RETURN, dl,
                     MVT::Other,
                     Chain, DAG.getRegister(StoreAddrReg, getPointerTy()));
}

SDValue X86TargetLowering::LowerTRAMPOLINE(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDValue Root = Op.getOperand(0);
  SDValue Trmp = Op.getOperand(1); // trampoline
  SDValue FPtr = Op.getOperand(2); // nested function
  SDValue Nest = Op.getOperand(3); // 'nest' parameter value
  DebugLoc dl  = Op.getDebugLoc();

  const Value *TrmpAddr = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();

  if (Subtarget->is64Bit()) {
    SDValue OutChains[6];

    // Large code-model.
    const unsigned char JMP64r  = 0xFF; // 64-bit jmp through register opcode.
    const unsigned char MOV64ri = 0xB8; // X86::MOV64ri opcode.

    const unsigned char N86R10 = RegInfo->getX86RegNum(X86::R10);
    const unsigned char N86R11 = RegInfo->getX86RegNum(X86::R11);

    const unsigned char REX_WB = 0x40 | 0x08 | 0x01; // REX prefix

    // Load the pointer to the nested function into R11.
    unsigned OpCode = ((MOV64ri | N86R11) << 8) | REX_WB; // movabsq r11
    SDValue Addr = Trmp;
    OutChains[0] = DAG.getStore(Root, dl, DAG.getConstant(OpCode, MVT::i16),
                                Addr, TrmpAddr, 0, false, false, 0);

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(2, MVT::i64));
    OutChains[1] = DAG.getStore(Root, dl, FPtr, Addr, TrmpAddr, 2,
                                false, false, 2);

    // Load the 'nest' parameter value into R10.
    // R10 is specified in X86CallingConv.td
    OpCode = ((MOV64ri | N86R10) << 8) | REX_WB; // movabsq r10
    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(10, MVT::i64));
    OutChains[2] = DAG.getStore(Root, dl, DAG.getConstant(OpCode, MVT::i16),
                                Addr, TrmpAddr, 10, false, false, 0);

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(12, MVT::i64));
    OutChains[3] = DAG.getStore(Root, dl, Nest, Addr, TrmpAddr, 12,
                                false, false, 2);

    // Jump to the nested function.
    OpCode = (JMP64r << 8) | REX_WB; // jmpq *...
    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(20, MVT::i64));
    OutChains[4] = DAG.getStore(Root, dl, DAG.getConstant(OpCode, MVT::i16),
                                Addr, TrmpAddr, 20, false, false, 0);

    unsigned char ModRM = N86R11 | (4 << 3) | (3 << 6); // ...r11
    Addr = DAG.getNode(ISD::ADD, dl, MVT::i64, Trmp,
                       DAG.getConstant(22, MVT::i64));
    OutChains[5] = DAG.getStore(Root, dl, DAG.getConstant(ModRM, MVT::i8), Addr,
                                TrmpAddr, 22, false, false, 0);

    SDValue Ops[] =
      { Trmp, DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains, 6) };
    return DAG.getMergeValues(Ops, 2, dl);
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
      const FunctionType *FTy = Func->getFunctionType();
      const AttrListPtr &Attrs = Func->getAttributes();

      if (!Attrs.isEmpty() && !Func->isVarArg()) {
        unsigned InRegCount = 0;
        unsigned Idx = 1;

        for (FunctionType::param_iterator I = FTy->param_begin(),
             E = FTy->param_end(); I != E; ++I, ++Idx)
          if (Attrs.paramHasAttr(Idx, Attribute::InReg))
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
    const unsigned char N86Reg = RegInfo->getX86RegNum(NestReg);
    OutChains[0] = DAG.getStore(Root, dl,
                                DAG.getConstant(MOV32ri|N86Reg, MVT::i8),
                                Trmp, TrmpAddr, 0, false, false, 0);

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                       DAG.getConstant(1, MVT::i32));
    OutChains[1] = DAG.getStore(Root, dl, Nest, Addr, TrmpAddr, 1,
                                false, false, 1);

    const unsigned char JMP = 0xE9; // jmp <32bit dst> opcode.
    Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                       DAG.getConstant(5, MVT::i32));
    OutChains[2] = DAG.getStore(Root, dl, DAG.getConstant(JMP, MVT::i8), Addr,
                                TrmpAddr, 5, false, false, 1);

    Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                       DAG.getConstant(6, MVT::i32));
    OutChains[3] = DAG.getStore(Root, dl, Disp, Addr, TrmpAddr, 6,
                                false, false, 1);

    SDValue Ops[] =
      { Trmp, DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains, 4) };
    return DAG.getMergeValues(Ops, 2, dl);
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
  const TargetFrameInfo &TFI = *TM.getFrameInfo();
  unsigned StackAlignment = TFI.getStackAlignment();
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();

  // Save FP Control Word to stack slot
  int SSFI = MF.getFrameInfo()->CreateStackObject(2, StackAlignment, false);
  SDValue StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());

  SDValue Chain = DAG.getNode(X86ISD::FNSTCW16m, dl, MVT::Other,
                              DAG.getEntryNode(), StackSlot);

  // Load FP Control Word from stack slot
  SDValue CWD = DAG.getLoad(MVT::i16, dl, Chain, StackSlot, NULL, 0,
                            false, false, 0);

  // Transform as necessary
  SDValue CWD1 =
    DAG.getNode(ISD::SRL, dl, MVT::i16,
                DAG.getNode(ISD::AND, dl, MVT::i16,
                            CWD, DAG.getConstant(0x800, MVT::i16)),
                DAG.getConstant(11, MVT::i8));
  SDValue CWD2 =
    DAG.getNode(ISD::SRL, dl, MVT::i16,
                DAG.getNode(ISD::AND, dl, MVT::i16,
                            CWD, DAG.getConstant(0x400, MVT::i16)),
                DAG.getConstant(9, MVT::i8));

  SDValue RetVal =
    DAG.getNode(ISD::AND, dl, MVT::i16,
                DAG.getNode(ISD::ADD, dl, MVT::i16,
                            DAG.getNode(ISD::OR, dl, MVT::i16, CWD1, CWD2),
                            DAG.getConstant(1, MVT::i16)),
                DAG.getConstant(3, MVT::i16));


  return DAG.getNode((VT.getSizeInBits() < 16 ?
                      ISD::TRUNCATE : ISD::ZERO_EXTEND), dl, VT, RetVal);
}

SDValue X86TargetLowering::LowerCTLZ(SDValue Op, SelectionDAG &DAG) const {
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

SDValue X86TargetLowering::LowerCTTZ(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  EVT OpVT = VT;
  unsigned NumBits = VT.getSizeInBits();
  DebugLoc dl = Op.getDebugLoc();

  Op = Op.getOperand(0);
  if (VT == MVT::i8) {
    OpVT = MVT::i32;
    Op = DAG.getNode(ISD::ZERO_EXTEND, dl, OpVT, Op);
  }

  // Issue a bsf (scan bits forward) which also sets EFLAGS.
  SDVTList VTs = DAG.getVTList(OpVT, MVT::i32);
  Op = DAG.getNode(X86ISD::BSF, dl, VTs, Op);

  // If src is zero (i.e. bsf sets ZF), returns NumBits.
  SDValue Ops[] = {
    Op,
    DAG.getConstant(NumBits, OpVT),
    DAG.getConstant(X86::COND_E, MVT::i8),
    Op.getValue(1)
  };
  Op = DAG.getNode(X86ISD::CMOV, dl, OpVT, Ops, array_lengthof(Ops));

  if (VT == MVT::i8)
    Op = DAG.getNode(ISD::TRUNCATE, dl, MVT::i8, Op);
  return Op;
}

SDValue X86TargetLowering::LowerMUL_V2I64(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  assert(VT == MVT::v2i64 && "Only know how to lower V2I64 multiply");
  DebugLoc dl = Op.getDebugLoc();

  //  ulong2 Ahi = __builtin_ia32_psrlqi128( a, 32);
  //  ulong2 Bhi = __builtin_ia32_psrlqi128( b, 32);
  //  ulong2 AloBlo = __builtin_ia32_pmuludq128( a, b );
  //  ulong2 AloBhi = __builtin_ia32_pmuludq128( a, Bhi );
  //  ulong2 AhiBlo = __builtin_ia32_pmuludq128( Ahi, b );
  //
  //  AloBhi = __builtin_ia32_psllqi128( AloBhi, 32 );
  //  AhiBlo = __builtin_ia32_psllqi128( AhiBlo, 32 );
  //  return AloBlo + AloBhi + AhiBlo;

  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);

  SDValue Ahi = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(Intrinsic::x86_sse2_psrli_q, MVT::i32),
                       A, DAG.getConstant(32, MVT::i32));
  SDValue Bhi = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(Intrinsic::x86_sse2_psrli_q, MVT::i32),
                       B, DAG.getConstant(32, MVT::i32));
  SDValue AloBlo = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(Intrinsic::x86_sse2_pmulu_dq, MVT::i32),
                       A, B);
  SDValue AloBhi = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(Intrinsic::x86_sse2_pmulu_dq, MVT::i32),
                       A, Bhi);
  SDValue AhiBlo = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(Intrinsic::x86_sse2_pmulu_dq, MVT::i32),
                       Ahi, B);
  AloBhi = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(Intrinsic::x86_sse2_pslli_q, MVT::i32),
                       AloBhi, DAG.getConstant(32, MVT::i32));
  AhiBlo = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                       DAG.getConstant(Intrinsic::x86_sse2_pslli_q, MVT::i32),
                       AhiBlo, DAG.getConstant(32, MVT::i32));
  SDValue Res = DAG.getNode(ISD::ADD, dl, VT, AloBlo, AloBhi);
  Res = DAG.getNode(ISD::ADD, dl, VT, Res, AhiBlo);
  return Res;
}

SDValue X86TargetLowering::LowerSHL(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  SDValue R = Op.getOperand(0);

  LLVMContext *Context = DAG.getContext();

  assert(Subtarget->hasSSE41() && "Cannot lower SHL without SSE4.1 or later");

  if (VT == MVT::v4i32) {
    Op = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                     DAG.getConstant(Intrinsic::x86_sse2_pslli_d, MVT::i32),
                     Op.getOperand(1), DAG.getConstant(23, MVT::i32));

    ConstantInt *CI = ConstantInt::get(*Context, APInt(32, 0x3f800000U));
    
    std::vector<Constant*> CV(4, CI);
    Constant *C = ConstantVector::get(CV);
    SDValue CPIdx = DAG.getConstantPool(C, getPointerTy(), 16);
    SDValue Addend = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                                 PseudoSourceValue::getConstantPool(), 0,
                                 false, false, 16);

    Op = DAG.getNode(ISD::ADD, dl, VT, Op, Addend);
    Op = DAG.getNode(ISD::BIT_CONVERT, dl, MVT::v4f32, Op);
    Op = DAG.getNode(ISD::FP_TO_SINT, dl, VT, Op);
    return DAG.getNode(ISD::MUL, dl, VT, Op, R);
  }
  if (VT == MVT::v16i8) {
    // a = a << 5;
    Op = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                     DAG.getConstant(Intrinsic::x86_sse2_pslli_w, MVT::i32),
                     Op.getOperand(1), DAG.getConstant(5, MVT::i32));

    ConstantInt *CM1 = ConstantInt::get(*Context, APInt(8, 15));
    ConstantInt *CM2 = ConstantInt::get(*Context, APInt(8, 63));

    std::vector<Constant*> CVM1(16, CM1);
    std::vector<Constant*> CVM2(16, CM2);
    Constant *C = ConstantVector::get(CVM1);
    SDValue CPIdx = DAG.getConstantPool(C, getPointerTy(), 16);
    SDValue M = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                            PseudoSourceValue::getConstantPool(), 0,
                            false, false, 16);

    // r = pblendv(r, psllw(r & (char16)15, 4), a);
    M = DAG.getNode(ISD::AND, dl, VT, R, M);
    M = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                    DAG.getConstant(Intrinsic::x86_sse2_pslli_w, MVT::i32), M,
                    DAG.getConstant(4, MVT::i32));
    R = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                    DAG.getConstant(Intrinsic::x86_sse41_pblendvb, MVT::i32),
                    R, M, Op);
    // a += a
    Op = DAG.getNode(ISD::ADD, dl, VT, Op, Op);
    
    C = ConstantVector::get(CVM2);
    CPIdx = DAG.getConstantPool(C, getPointerTy(), 16);
    M = DAG.getLoad(VT, dl, DAG.getEntryNode(), CPIdx,
                    PseudoSourceValue::getConstantPool(), 0, false, false, 16);
    
    // r = pblendv(r, psllw(r & (char16)63, 2), a);
    M = DAG.getNode(ISD::AND, dl, VT, R, M);
    M = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                    DAG.getConstant(Intrinsic::x86_sse2_pslli_w, MVT::i32), M,
                    DAG.getConstant(2, MVT::i32));
    R = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                    DAG.getConstant(Intrinsic::x86_sse41_pblendvb, MVT::i32),
                    R, M, Op);
    // a += a
    Op = DAG.getNode(ISD::ADD, dl, VT, Op, Op);
    
    // return pblendv(r, r+r, a);
    R = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, VT,
                    DAG.getConstant(Intrinsic::x86_sse41_pblendvb, MVT::i32),
                    R, DAG.getNode(ISD::ADD, dl, VT, R, R), Op);
    return R;
  }
  return SDValue();
}

SDValue X86TargetLowering::LowerXALUO(SDValue Op, SelectionDAG &DAG) const {
  // Lower the "add/sub/mul with overflow" instruction into a regular ins plus
  // a "setcc" instruction that checks the overflow flag. The "brcond" lowering
  // looks for this combo and may remove the "setcc" instruction if the "setcc"
  // has only one use.
  SDNode *N = Op.getNode();
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  unsigned BaseOp = 0;
  unsigned Cond = 0;
  DebugLoc dl = Op.getDebugLoc();

  switch (Op.getOpcode()) {
  default: llvm_unreachable("Unknown ovf instruction!");
  case ISD::SADDO:
    // A subtract of one will be selected as a INC. Note that INC doesn't
    // set CF, so we can't do this for UADDO.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op))
      if (C->getAPIntValue() == 1) {
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
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op))
      if (C->getAPIntValue() == 1) {
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
  case ISD::UMULO:
    BaseOp = X86ISD::UMUL;
    Cond = X86::COND_B;
    break;
  }

  // Also sets EFLAGS.
  SDVTList VTs = DAG.getVTList(N->getValueType(0), MVT::i32);
  SDValue Sum = DAG.getNode(BaseOp, dl, VTs, LHS, RHS);

  SDValue SetCC =
    DAG.getNode(X86ISD::SETCC, dl, N->getValueType(1),
                DAG.getConstant(Cond, MVT::i32), SDValue(Sum.getNode(), 1));

  DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), SetCC);
  return Sum;
}

SDValue X86TargetLowering::LowerMEMBARRIER(SDValue Op, SelectionDAG &DAG) const{
  DebugLoc dl = Op.getDebugLoc();
  
  if (!Subtarget->hasSSE2()) {
    SDValue Chain = Op.getOperand(0);
    SDValue Zero = DAG.getConstant(0, 
                                   Subtarget->is64Bit() ? MVT::i64 : MVT::i32);
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

SDValue X86TargetLowering::LowerCMP_SWAP(SDValue Op, SelectionDAG &DAG) const {
  EVT T = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned Reg = 0;
  unsigned size = 0;
  switch(T.getSimpleVT().SimpleTy) {
  default:
    assert(false && "Invalid value type!");
  case MVT::i8:  Reg = X86::AL;  size = 1; break;
  case MVT::i16: Reg = X86::AX;  size = 2; break;
  case MVT::i32: Reg = X86::EAX; size = 4; break;
  case MVT::i64:
    assert(Subtarget->is64Bit() && "Node not type legal!");
    Reg = X86::RAX; size = 8;
    break;
  }
  SDValue cpIn = DAG.getCopyToReg(Op.getOperand(0), dl, Reg,
                                    Op.getOperand(2), SDValue());
  SDValue Ops[] = { cpIn.getValue(0),
                    Op.getOperand(1),
                    Op.getOperand(3),
                    DAG.getTargetConstant(size, MVT::i8),
                    cpIn.getValue(1) };
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  SDValue Result = DAG.getNode(X86ISD::LCMPXCHG_DAG, dl, Tys, Ops, 5);
  SDValue cpOut =
    DAG.getCopyFromReg(Result.getValue(0), dl, Reg, T, Result.getValue(1));
  return cpOut;
}

SDValue X86TargetLowering::LowerREADCYCLECOUNTER(SDValue Op,
                                                 SelectionDAG &DAG) const {
  assert(Subtarget->is64Bit() && "Result not type legalized?");
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
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

SDValue X86TargetLowering::LowerBIT_CONVERT(SDValue Op,
                                            SelectionDAG &DAG) const {
  EVT SrcVT = Op.getOperand(0).getValueType();
  EVT DstVT = Op.getValueType();
  assert((Subtarget->is64Bit() && !Subtarget->hasSSE2() && 
          Subtarget->hasMMX() && !DisableMMX) &&
         "Unexpected custom BIT_CONVERT");
  assert((DstVT == MVT::i64 || 
          (DstVT.isVector() && DstVT.getSizeInBits()==64)) &&
         "Unexpected custom BIT_CONVERT");
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
SDValue X86TargetLowering::LowerLOAD_SUB(SDValue Op, SelectionDAG &DAG) const {
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
                       cast<AtomicSDNode>(Node)->getAlignment());
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDValue X86TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Should not custom lower this!");
  case ISD::MEMBARRIER:         return LowerMEMBARRIER(Op,DAG);
  case ISD::ATOMIC_CMP_SWAP:    return LowerCMP_SWAP(Op,DAG);
  case ISD::ATOMIC_LOAD_SUB:    return LowerLOAD_SUB(Op,DAG);
  case ISD::BUILD_VECTOR:       return LowerBUILD_VECTOR(Op, DAG);
  case ISD::CONCAT_VECTORS:     return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::VECTOR_SHUFFLE:     return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT: return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:  return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::SCALAR_TO_VECTOR:   return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
  case ISD::ExternalSymbol:     return LowerExternalSymbol(Op, DAG);
  case ISD::BlockAddress:       return LowerBlockAddress(Op, DAG);
  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:          return LowerShift(Op, DAG);
  case ISD::SINT_TO_FP:         return LowerSINT_TO_FP(Op, DAG);
  case ISD::UINT_TO_FP:         return LowerUINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:         return LowerFP_TO_SINT(Op, DAG);
  case ISD::FP_TO_UINT:         return LowerFP_TO_UINT(Op, DAG);
  case ISD::FABS:               return LowerFABS(Op, DAG);
  case ISD::FNEG:               return LowerFNEG(Op, DAG);
  case ISD::FCOPYSIGN:          return LowerFCOPYSIGN(Op, DAG);
  case ISD::SETCC:              return LowerSETCC(Op, DAG);
  case ISD::VSETCC:             return LowerVSETCC(Op, DAG);
  case ISD::SELECT:             return LowerSELECT(Op, DAG);
  case ISD::BRCOND:             return LowerBRCOND(Op, DAG);
  case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
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
  case ISD::MUL:                return LowerMUL_V2I64(Op, DAG);
  case ISD::SHL:                return LowerSHL(Op, DAG);
  case ISD::SADDO:
  case ISD::UADDO:
  case ISD::SSUBO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:              return LowerXALUO(Op, DAG);
  case ISD::READCYCLECOUNTER:   return LowerREADCYCLECOUNTER(Op, DAG);
  case ISD::BIT_CONVERT:        return LowerBIT_CONVERT(Op, DAG);
  }
}

void X86TargetLowering::
ReplaceATOMIC_BINARY_64(SDNode *Node, SmallVectorImpl<SDValue>&Results,
                        SelectionDAG &DAG, unsigned NewOp) const {
  EVT T = Node->getValueType(0);
  DebugLoc dl = Node->getDebugLoc();
  assert (T == MVT::i64 && "Only know how to expand i64 atomics");

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
    assert(false && "Do not know how to custom type legalize this operation!");
    return;
  case ISD::FP_TO_SINT: {
    std::pair<SDValue,SDValue> Vals =
        FP_TO_INTHelper(SDValue(N, 0), DAG, true);
    SDValue FIST = Vals.first, StackSlot = Vals.second;
    if (FIST.getNode() != 0) {
      EVT VT = N->getValueType(0);
      // Return a load from the stack slot.
      Results.push_back(DAG.getLoad(VT, dl, FIST, StackSlot, NULL, 0,
                                    false, false, 0));
    }
    return;
  }
  case ISD::READCYCLECOUNTER: {
    SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
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
    assert (T == MVT::i64 && "Only know how to expand i64 Cmp and Swap");
    SDValue cpInL, cpInH;
    cpInL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, N->getOperand(2),
                        DAG.getConstant(0, MVT::i32));
    cpInH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, N->getOperand(2),
                        DAG.getConstant(1, MVT::i32));
    cpInL = DAG.getCopyToReg(N->getOperand(0), dl, X86::EAX, cpInL, SDValue());
    cpInH = DAG.getCopyToReg(cpInL.getValue(0), dl, X86::EDX, cpInH,
                             cpInL.getValue(1));
    SDValue swapInL, swapInH;
    swapInL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, N->getOperand(3),
                          DAG.getConstant(0, MVT::i32));
    swapInH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32, N->getOperand(3),
                          DAG.getConstant(1, MVT::i32));
    swapInL = DAG.getCopyToReg(cpInH.getValue(0), dl, X86::EBX, swapInL,
                               cpInH.getValue(1));
    swapInH = DAG.getCopyToReg(swapInL.getValue(0), dl, X86::ECX, swapInH,
                               swapInL.getValue(1));
    SDValue Ops[] = { swapInH.getValue(0),
                      N->getOperand(1),
                      swapInH.getValue(1) };
    SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
    SDValue Result = DAG.getNode(X86ISD::LCMPXCHG8_DAG, dl, Tys, Ops, 3);
    SDValue cpOutL = DAG.getCopyFromReg(Result.getValue(0), dl, X86::EAX,
                                        MVT::i32, Result.getValue(1));
    SDValue cpOutH = DAG.getCopyFromReg(cpOutL.getValue(1), dl, X86::EDX,
                                        MVT::i32, cpOutL.getValue(2));
    SDValue OpsF[] = { cpOutL.getValue(0), cpOutH.getValue(0)};
    Results.push_back(DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, OpsF, 2));
    Results.push_back(cpOutH.getValue(1));
    return;
  }
  case ISD::ATOMIC_LOAD_ADD:
    ReplaceATOMIC_BINARY_64(N, Results, DAG, X86ISD::ATOMADD64_DAG);
    return;
  case ISD::ATOMIC_LOAD_AND:
    ReplaceATOMIC_BINARY_64(N, Results, DAG, X86ISD::ATOMAND64_DAG);
    return;
  case ISD::ATOMIC_LOAD_NAND:
    ReplaceATOMIC_BINARY_64(N, Results, DAG, X86ISD::ATOMNAND64_DAG);
    return;
  case ISD::ATOMIC_LOAD_OR:
    ReplaceATOMIC_BINARY_64(N, Results, DAG, X86ISD::ATOMOR64_DAG);
    return;
  case ISD::ATOMIC_LOAD_SUB:
    ReplaceATOMIC_BINARY_64(N, Results, DAG, X86ISD::ATOMSUB64_DAG);
    return;
  case ISD::ATOMIC_LOAD_XOR:
    ReplaceATOMIC_BINARY_64(N, Results, DAG, X86ISD::ATOMXOR64_DAG);
    return;
  case ISD::ATOMIC_SWAP:
    ReplaceATOMIC_BINARY_64(N, Results, DAG, X86ISD::ATOMSWAP64_DAG);
    return;
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
  case X86ISD::MMX_PINSRW:         return "X86ISD::MMX_PINSRW";
  case X86ISD::PSHUFB:             return "X86ISD::PSHUFB";
  case X86ISD::FMAX:               return "X86ISD::FMAX";
  case X86ISD::FMIN:               return "X86ISD::FMIN";
  case X86ISD::FRSQRT:             return "X86ISD::FRSQRT";
  case X86ISD::FRCP:               return "X86ISD::FRCP";
  case X86ISD::TLSADDR:            return "X86ISD::TLSADDR";
  case X86ISD::TLSCALL:            return "X86ISD::TLSCALL";
  case X86ISD::SegmentBaseAddress: return "X86ISD::SegmentBaseAddress";
  case X86ISD::EH_RETURN:          return "X86ISD::EH_RETURN";
  case X86ISD::TC_RETURN:          return "X86ISD::TC_RETURN";
  case X86ISD::FNSTCW16m:          return "X86ISD::FNSTCW16m";
  case X86ISD::LCMPXCHG_DAG:       return "X86ISD::LCMPXCHG_DAG";
  case X86ISD::LCMPXCHG8_DAG:      return "X86ISD::LCMPXCHG8_DAG";
  case X86ISD::ATOMADD64_DAG:      return "X86ISD::ATOMADD64_DAG";
  case X86ISD::ATOMSUB64_DAG:      return "X86ISD::ATOMSUB64_DAG";
  case X86ISD::ATOMOR64_DAG:       return "X86ISD::ATOMOR64_DAG";
  case X86ISD::ATOMXOR64_DAG:      return "X86ISD::ATOMXOR64_DAG";
  case X86ISD::ATOMAND64_DAG:      return "X86ISD::ATOMAND64_DAG";
  case X86ISD::ATOMNAND64_DAG:     return "X86ISD::ATOMNAND64_DAG";
  case X86ISD::VZEXT_MOVL:         return "X86ISD::VZEXT_MOVL";
  case X86ISD::VZEXT_LOAD:         return "X86ISD::VZEXT_LOAD";
  case X86ISD::VSHL:               return "X86ISD::VSHL";
  case X86ISD::VSRL:               return "X86ISD::VSRL";
  case X86ISD::CMPPD:              return "X86ISD::CMPPD";
  case X86ISD::CMPPS:              return "X86ISD::CMPPS";
  case X86ISD::PCMPEQB:            return "X86ISD::PCMPEQB";
  case X86ISD::PCMPEQW:            return "X86ISD::PCMPEQW";
  case X86ISD::PCMPEQD:            return "X86ISD::PCMPEQD";
  case X86ISD::PCMPEQQ:            return "X86ISD::PCMPEQQ";
  case X86ISD::PCMPGTB:            return "X86ISD::PCMPGTB";
  case X86ISD::PCMPGTW:            return "X86ISD::PCMPGTW";
  case X86ISD::PCMPGTD:            return "X86ISD::PCMPGTD";
  case X86ISD::PCMPGTQ:            return "X86ISD::PCMPGTQ";
  case X86ISD::ADD:                return "X86ISD::ADD";
  case X86ISD::SUB:                return "X86ISD::SUB";
  case X86ISD::SMUL:               return "X86ISD::SMUL";
  case X86ISD::UMUL:               return "X86ISD::UMUL";
  case X86ISD::INC:                return "X86ISD::INC";
  case X86ISD::DEC:                return "X86ISD::DEC";
  case X86ISD::OR:                 return "X86ISD::OR";
  case X86ISD::XOR:                return "X86ISD::XOR";
  case X86ISD::AND:                return "X86ISD::AND";
  case X86ISD::MUL_IMM:            return "X86ISD::MUL_IMM";
  case X86ISD::PTEST:              return "X86ISD::PTEST";
  case X86ISD::TESTP:              return "X86ISD::TESTP";
  case X86ISD::PALIGN:             return "X86ISD::PALIGN";
  case X86ISD::PSHUFD:             return "X86ISD::PSHUFD";
  case X86ISD::PSHUFHW:            return "X86ISD::PSHUFHW";
  case X86ISD::PSHUFHW_LD:         return "X86ISD::PSHUFHW_LD";
  case X86ISD::PSHUFLW:            return "X86ISD::PSHUFLW";
  case X86ISD::PSHUFLW_LD:         return "X86ISD::PSHUFLW_LD";
  case X86ISD::SHUFPS:             return "X86ISD::SHUFPS";
  case X86ISD::SHUFPD:             return "X86ISD::SHUFPD";
  case X86ISD::MOVLHPS:            return "X86ISD::MOVLHPS";
  case X86ISD::MOVLHPD:            return "X86ISD::MOVLHPD";
  case X86ISD::MOVHLPS:            return "X86ISD::MOVHLPS";
  case X86ISD::MOVHLPD:            return "X86ISD::MOVHLPD";
  case X86ISD::MOVLPS:             return "X86ISD::MOVLPS";
  case X86ISD::MOVLPD:             return "X86ISD::MOVLPD";
  case X86ISD::MOVDDUP:            return "X86ISD::MOVDDUP";
  case X86ISD::MOVSHDUP:           return "X86ISD::MOVSHDUP";
  case X86ISD::MOVSLDUP:           return "X86ISD::MOVSLDUP";
  case X86ISD::MOVSHDUP_LD:        return "X86ISD::MOVSHDUP_LD";
  case X86ISD::MOVSLDUP_LD:        return "X86ISD::MOVSLDUP_LD";
  case X86ISD::MOVSD:              return "X86ISD::MOVSD";
  case X86ISD::MOVSS:              return "X86ISD::MOVSS";
  case X86ISD::UNPCKLPS:           return "X86ISD::UNPCKLPS";
  case X86ISD::UNPCKLPD:           return "X86ISD::UNPCKLPD";
  case X86ISD::UNPCKHPS:           return "X86ISD::UNPCKHPS";
  case X86ISD::UNPCKHPD:           return "X86ISD::UNPCKHPD";
  case X86ISD::PUNPCKLBW:          return "X86ISD::PUNPCKLBW";
  case X86ISD::PUNPCKLWD:          return "X86ISD::PUNPCKLWD";
  case X86ISD::PUNPCKLDQ:          return "X86ISD::PUNPCKLDQ";
  case X86ISD::PUNPCKLQDQ:         return "X86ISD::PUNPCKLQDQ";
  case X86ISD::PUNPCKHBW:          return "X86ISD::PUNPCKHBW";
  case X86ISD::PUNPCKHWD:          return "X86ISD::PUNPCKHWD";
  case X86ISD::PUNPCKHDQ:          return "X86ISD::PUNPCKHDQ";
  case X86ISD::PUNPCKHQDQ:         return "X86ISD::PUNPCKHQDQ";
  case X86ISD::VASTART_SAVE_XMM_REGS: return "X86ISD::VASTART_SAVE_XMM_REGS";
  case X86ISD::MINGW_ALLOCA:       return "X86ISD::MINGW_ALLOCA";
  }
}

// isLegalAddressingMode - Return true if the addressing mode represented
// by AM is legal for this target, for a load/store of the specified type.
bool X86TargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                              const Type *Ty) const {
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


bool X86TargetLowering::isTruncateFree(const Type *Ty1, const Type *Ty2) const {
  if (!Ty1->isIntegerTy() || !Ty2->isIntegerTy())
    return false;
  unsigned NumBits1 = Ty1->getPrimitiveSizeInBits();
  unsigned NumBits2 = Ty2->getPrimitiveSizeInBits();
  if (NumBits1 <= NumBits2)
    return false;
  return true;
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

bool X86TargetLowering::isZExtFree(const Type *Ty1, const Type *Ty2) const {
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
    return isPALIGNRMask(M, VT, Subtarget->hasSSSE3());

  // FIXME: pshufb, blends, shifts.
  return (VT.getVectorNumElements() == 2 ||
          ShuffleVectorSDNode::isSplatMask(&M[0], VT) ||
          isMOVLMask(M, VT) ||
          isSHUFPMask(M, VT) ||
          isPSHUFDMask(M, VT) ||
          isPSHUFHWMask(M, VT) ||
          isPSHUFLWMask(M, VT) ||
          isPALIGNRMask(M, VT, Subtarget->hasSSSE3()) ||
          isUNPCKLMask(M, VT) ||
          isUNPCKHMask(M, VT) ||
          isUNPCKL_v_undef_Mask(M, VT) ||
          isUNPCKH_v_undef_Mask(M, VT));
}

bool
X86TargetLowering::isVectorClearMaskLegal(const SmallVectorImpl<int> &Mask,
                                          EVT VT) const {
  unsigned NumElts = VT.getVectorNumElements();
  // FIXME: This collection of masks seems suspect.
  if (NumElts == 2)
    return true;
  if (NumElts == 4 && VT.getSizeInBits() == 128) {
    return (isMOVLMask(Mask, VT)  ||
            isCommutedMOVLMask(Mask, VT, true) ||
            isSHUFPMask(Mask, VT) ||
            isCommutedSHUFPMask(Mask, VT));
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
                                                       unsigned immOpc,
                                                       unsigned LoadOpc,
                                                       unsigned CXchgOpc,
                                                       unsigned notOpc,
                                                       unsigned EAXreg,
                                                       TargetRegisterClass *RC,
                                                       bool invSrc) const {
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
  MachineFunction::iterator MBBIter = MBB;
  ++MBBIter;

  /// First build the CFG
  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *newMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *nextMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(MBBIter, newMBB);
  F->insert(MBBIter, nextMBB);

  // Transfer the remainder of thisMBB and its successor edges to nextMBB.
  nextMBB->splice(nextMBB->begin(), thisMBB,
                  llvm::next(MachineBasicBlock::iterator(bInstr)),
                  thisMBB->end());
  nextMBB->transferSuccessorsAndUpdatePHIs(thisMBB);

  // Update thisMBB to fall through to newMBB
  thisMBB->addSuccessor(newMBB);

  // newMBB jumps to itself and fall through to nextMBB
  newMBB->addSuccessor(nextMBB);
  newMBB->addSuccessor(newMBB);

  // Insert instructions into newMBB based on incoming instruction
  assert(bInstr->getNumOperands() < X86::AddrNumOperands + 4 &&
         "unexpected number of operands");
  DebugLoc dl = bInstr->getDebugLoc();
  MachineOperand& destOper = bInstr->getOperand(0);
  MachineOperand* argOpers[2 + X86::AddrNumOperands];
  int numArgs = bInstr->getNumOperands() - 1;
  for (int i=0; i < numArgs; ++i)
    argOpers[i] = &bInstr->getOperand(i+1);

  // x86 address has 4 operands: base, index, scale, and displacement
  int lastAddrIndx = X86::AddrNumOperands - 1; // [0,3]
  int valArgIndx = lastAddrIndx + 1;

  unsigned t1 = F->getRegInfo().createVirtualRegister(RC);
  MachineInstrBuilder MIB = BuildMI(newMBB, dl, TII->get(LoadOpc), t1);
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);

  unsigned tt = F->getRegInfo().createVirtualRegister(RC);
  if (invSrc) {
    MIB = BuildMI(newMBB, dl, TII->get(notOpc), tt).addReg(t1);
  }
  else
    tt = t1;

  unsigned t2 = F->getRegInfo().createVirtualRegister(RC);
  assert((argOpers[valArgIndx]->isReg() ||
          argOpers[valArgIndx]->isImm()) &&
         "invalid operand");
  if (argOpers[valArgIndx]->isReg())
    MIB = BuildMI(newMBB, dl, TII->get(regOpc), t2);
  else
    MIB = BuildMI(newMBB, dl, TII->get(immOpc), t2);
  MIB.addReg(tt);
  (*MIB).addOperand(*argOpers[valArgIndx]);

  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), EAXreg);
  MIB.addReg(t1);

  MIB = BuildMI(newMBB, dl, TII->get(CXchgOpc));
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);
  MIB.addReg(t2);
  assert(bInstr->hasOneMemOperand() && "Unexpected number of memoperand");
  (*MIB).setMemRefs(bInstr->memoperands_begin(),
                    bInstr->memoperands_end());

  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), destOper.getReg());
  MIB.addReg(EAXreg);

  // insert branch
  BuildMI(newMBB, dl, TII->get(X86::JNE_4)).addMBB(newMBB);

  bInstr->eraseFromParent();   // The pseudo instruction is gone now.
  return nextMBB;
}

// private utility function:  64 bit atomics on 32 bit host.
MachineBasicBlock *
X86TargetLowering::EmitAtomicBit6432WithCustomInserter(MachineInstr *bInstr,
                                                       MachineBasicBlock *MBB,
                                                       unsigned regOpcL,
                                                       unsigned regOpcH,
                                                       unsigned immOpcL,
                                                       unsigned immOpcH,
                                                       bool invSrc) const {
  // For the atomic bitwise operator, we generate
  //   thisMBB (instructions are in pairs, except cmpxchg8b)
  //     ld t1,t2 = [bitinstr.addr]
  //   newMBB:
  //     out1, out2 = phi (thisMBB, t1/t2) (newMBB, t3/t4)
  //     op  t5, t6 <- out1, out2, [bitinstr.val]
  //      (for SWAP, substitute:  mov t5, t6 <- [bitinstr.val])
  //     mov ECX, EBX <- t5, t6
  //     mov EAX, EDX <- t1, t2
  //     cmpxchg8b [bitinstr.addr]  [EAX, EDX, EBX, ECX implicit]
  //     mov t3, t4 <- EAX, EDX
  //     bz  newMBB
  //     result in out1, out2
  //     fallthrough -->nextMBB

  const TargetRegisterClass *RC = X86::GR32RegisterClass;
  const unsigned LoadOpc = X86::MOV32rm;
  const unsigned NotOpc = X86::NOT32r;
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator MBBIter = MBB;
  ++MBBIter;

  /// First build the CFG
  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *newMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *nextMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(MBBIter, newMBB);
  F->insert(MBBIter, nextMBB);

  // Transfer the remainder of thisMBB and its successor edges to nextMBB.
  nextMBB->splice(nextMBB->begin(), thisMBB,
                  llvm::next(MachineBasicBlock::iterator(bInstr)),
                  thisMBB->end());
  nextMBB->transferSuccessorsAndUpdatePHIs(thisMBB);

  // Update thisMBB to fall through to newMBB
  thisMBB->addSuccessor(newMBB);

  // newMBB jumps to itself and fall through to nextMBB
  newMBB->addSuccessor(nextMBB);
  newMBB->addSuccessor(newMBB);

  DebugLoc dl = bInstr->getDebugLoc();
  // Insert instructions into newMBB based on incoming instruction
  // There are 8 "real" operands plus 9 implicit def/uses, ignored here.
  assert(bInstr->getNumOperands() < X86::AddrNumOperands + 14 &&
         "unexpected number of operands");
  MachineOperand& dest1Oper = bInstr->getOperand(0);
  MachineOperand& dest2Oper = bInstr->getOperand(1);
  MachineOperand* argOpers[2 + X86::AddrNumOperands];
  for (int i=0; i < 2 + X86::AddrNumOperands; ++i) {
    argOpers[i] = &bInstr->getOperand(i+2);

    // We use some of the operands multiple times, so conservatively just
    // clear any kill flags that might be present.
    if (argOpers[i]->isReg() && argOpers[i]->isUse())
      argOpers[i]->setIsKill(false);
  }

  // x86 address has 5 operands: base, index, scale, displacement, and segment.
  int lastAddrIndx = X86::AddrNumOperands - 1; // [0,3]

  unsigned t1 = F->getRegInfo().createVirtualRegister(RC);
  MachineInstrBuilder MIB = BuildMI(thisMBB, dl, TII->get(LoadOpc), t1);
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);
  unsigned t2 = F->getRegInfo().createVirtualRegister(RC);
  MIB = BuildMI(thisMBB, dl, TII->get(LoadOpc), t2);
  // add 4 to displacement.
  for (int i=0; i <= lastAddrIndx-2; ++i)
    (*MIB).addOperand(*argOpers[i]);
  MachineOperand newOp3 = *(argOpers[3]);
  if (newOp3.isImm())
    newOp3.setImm(newOp3.getImm()+4);
  else
    newOp3.setOffset(newOp3.getOffset()+4);
  (*MIB).addOperand(newOp3);
  (*MIB).addOperand(*argOpers[lastAddrIndx]);

  // t3/4 are defined later, at the bottom of the loop
  unsigned t3 = F->getRegInfo().createVirtualRegister(RC);
  unsigned t4 = F->getRegInfo().createVirtualRegister(RC);
  BuildMI(newMBB, dl, TII->get(X86::PHI), dest1Oper.getReg())
    .addReg(t1).addMBB(thisMBB).addReg(t3).addMBB(newMBB);
  BuildMI(newMBB, dl, TII->get(X86::PHI), dest2Oper.getReg())
    .addReg(t2).addMBB(thisMBB).addReg(t4).addMBB(newMBB);

  // The subsequent operations should be using the destination registers of
  //the PHI instructions.
  if (invSrc) {
    t1 = F->getRegInfo().createVirtualRegister(RC);
    t2 = F->getRegInfo().createVirtualRegister(RC);
    MIB = BuildMI(newMBB, dl, TII->get(NotOpc), t1).addReg(dest1Oper.getReg());
    MIB = BuildMI(newMBB, dl, TII->get(NotOpc), t2).addReg(dest2Oper.getReg());
  } else {
    t1 = dest1Oper.getReg();
    t2 = dest2Oper.getReg();
  }

  int valArgIndx = lastAddrIndx + 1;
  assert((argOpers[valArgIndx]->isReg() ||
          argOpers[valArgIndx]->isImm()) &&
         "invalid operand");
  unsigned t5 = F->getRegInfo().createVirtualRegister(RC);
  unsigned t6 = F->getRegInfo().createVirtualRegister(RC);
  if (argOpers[valArgIndx]->isReg())
    MIB = BuildMI(newMBB, dl, TII->get(regOpcL), t5);
  else
    MIB = BuildMI(newMBB, dl, TII->get(immOpcL), t5);
  if (regOpcL != X86::MOV32rr)
    MIB.addReg(t1);
  (*MIB).addOperand(*argOpers[valArgIndx]);
  assert(argOpers[valArgIndx + 1]->isReg() ==
         argOpers[valArgIndx]->isReg());
  assert(argOpers[valArgIndx + 1]->isImm() ==
         argOpers[valArgIndx]->isImm());
  if (argOpers[valArgIndx + 1]->isReg())
    MIB = BuildMI(newMBB, dl, TII->get(regOpcH), t6);
  else
    MIB = BuildMI(newMBB, dl, TII->get(immOpcH), t6);
  if (regOpcH != X86::MOV32rr)
    MIB.addReg(t2);
  (*MIB).addOperand(*argOpers[valArgIndx + 1]);

  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), X86::EAX);
  MIB.addReg(t1);
  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), X86::EDX);
  MIB.addReg(t2);

  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), X86::EBX);
  MIB.addReg(t5);
  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), X86::ECX);
  MIB.addReg(t6);

  MIB = BuildMI(newMBB, dl, TII->get(X86::LCMPXCHG8B));
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);

  assert(bInstr->hasOneMemOperand() && "Unexpected number of memoperand");
  (*MIB).setMemRefs(bInstr->memoperands_begin(),
                    bInstr->memoperands_end());

  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), t3);
  MIB.addReg(X86::EAX);
  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), t4);
  MIB.addReg(X86::EDX);

  // insert branch
  BuildMI(newMBB, dl, TII->get(X86::JNE_4)).addMBB(newMBB);

  bInstr->eraseFromParent();   // The pseudo instruction is gone now.
  return nextMBB;
}

// private utility function
MachineBasicBlock *
X86TargetLowering::EmitAtomicMinMaxWithCustomInserter(MachineInstr *mInstr,
                                                      MachineBasicBlock *MBB,
                                                      unsigned cmovOpc) const {
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
  MachineFunction::iterator MBBIter = MBB;
  ++MBBIter;

  /// First build the CFG
  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *thisMBB = MBB;
  MachineBasicBlock *newMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *nextMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(MBBIter, newMBB);
  F->insert(MBBIter, nextMBB);

  // Transfer the remainder of thisMBB and its successor edges to nextMBB.
  nextMBB->splice(nextMBB->begin(), thisMBB,
                  llvm::next(MachineBasicBlock::iterator(mInstr)),
                  thisMBB->end());
  nextMBB->transferSuccessorsAndUpdatePHIs(thisMBB);

  // Update thisMBB to fall through to newMBB
  thisMBB->addSuccessor(newMBB);

  // newMBB jumps to newMBB and fall through to nextMBB
  newMBB->addSuccessor(nextMBB);
  newMBB->addSuccessor(newMBB);

  DebugLoc dl = mInstr->getDebugLoc();
  // Insert instructions into newMBB based on incoming instruction
  assert(mInstr->getNumOperands() < X86::AddrNumOperands + 4 &&
         "unexpected number of operands");
  MachineOperand& destOper = mInstr->getOperand(0);
  MachineOperand* argOpers[2 + X86::AddrNumOperands];
  int numArgs = mInstr->getNumOperands() - 1;
  for (int i=0; i < numArgs; ++i)
    argOpers[i] = &mInstr->getOperand(i+1);

  // x86 address has 4 operands: base, index, scale, and displacement
  int lastAddrIndx = X86::AddrNumOperands - 1; // [0,3]
  int valArgIndx = lastAddrIndx + 1;

  unsigned t1 = F->getRegInfo().createVirtualRegister(X86::GR32RegisterClass);
  MachineInstrBuilder MIB = BuildMI(newMBB, dl, TII->get(X86::MOV32rm), t1);
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);

  // We only support register and immediate values
  assert((argOpers[valArgIndx]->isReg() ||
          argOpers[valArgIndx]->isImm()) &&
         "invalid operand");

  unsigned t2 = F->getRegInfo().createVirtualRegister(X86::GR32RegisterClass);
  if (argOpers[valArgIndx]->isReg())
    MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), t2);
  else
    MIB = BuildMI(newMBB, dl, TII->get(X86::MOV32rr), t2);
  (*MIB).addOperand(*argOpers[valArgIndx]);

  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), X86::EAX);
  MIB.addReg(t1);

  MIB = BuildMI(newMBB, dl, TII->get(X86::CMP32rr));
  MIB.addReg(t1);
  MIB.addReg(t2);

  // Generate movc
  unsigned t3 = F->getRegInfo().createVirtualRegister(X86::GR32RegisterClass);
  MIB = BuildMI(newMBB, dl, TII->get(cmovOpc),t3);
  MIB.addReg(t2);
  MIB.addReg(t1);

  // Cmp and exchange if none has modified the memory location
  MIB = BuildMI(newMBB, dl, TII->get(X86::LCMPXCHG32));
  for (int i=0; i <= lastAddrIndx; ++i)
    (*MIB).addOperand(*argOpers[i]);
  MIB.addReg(t3);
  assert(mInstr->hasOneMemOperand() && "Unexpected number of memoperand");
  (*MIB).setMemRefs(mInstr->memoperands_begin(),
                    mInstr->memoperands_end());

  MIB = BuildMI(newMBB, dl, TII->get(TargetOpcode::COPY), destOper.getReg());
  MIB.addReg(X86::EAX);

  // insert branch
  BuildMI(newMBB, dl, TII->get(X86::JNE_4)).addMBB(newMBB);

  mInstr->eraseFromParent();   // The pseudo instruction is gone now.
  return nextMBB;
}

// FIXME: When we get size specific XMM0 registers, i.e. XMM0_V16I8
// or XMM0_V32I8 in AVX all of this code can be replaced with that
// in the .td file.
MachineBasicBlock *
X86TargetLowering::EmitPCMP(MachineInstr *MI, MachineBasicBlock *BB,
                            unsigned numArgs, bool memArg) const {

  assert((Subtarget->hasSSE42() || Subtarget->hasAVX()) &&
         "Target must have SSE4.2 or AVX features enabled");

  DebugLoc dl = MI->getDebugLoc();
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  unsigned Opc;

  if (!Subtarget->hasAVX()) {
    if (memArg)
      Opc = numArgs == 3 ? X86::PCMPISTRM128rm : X86::PCMPESTRM128rm;
    else
      Opc = numArgs == 3 ? X86::PCMPISTRM128rr : X86::PCMPESTRM128rr;
  } else {
    if (memArg)
      Opc = numArgs == 3 ? X86::VPCMPISTRM128rm : X86::VPCMPESTRM128rm;
    else
      Opc = numArgs == 3 ? X86::VPCMPISTRM128rr : X86::VPCMPESTRM128rr;
  }

  MachineInstrBuilder MIB = BuildMI(BB, dl, TII->get(Opc));

  for (unsigned i = 0; i < numArgs; ++i) {
    MachineOperand &Op = MI->getOperand(i+1);

    if (!(Op.isReg() && Op.isImplicit()))
      MIB.addOperand(Op);
  }

  BuildMI(BB, dl, TII->get(X86::MOVAPSrr), MI->getOperand(0).getReg())
    .addReg(X86::XMM0);

  MI->eraseFromParent();

  return BB;
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

  // In the XMM save block, save all the XMM argument registers.
  for (int i = 3, e = MI->getNumOperands(); i != e; ++i) {
    int64_t Offset = (i - 3) * 16 + VarArgsFPOffset;
    MachineMemOperand *MMO =
      F->getMachineMemOperand(
        PseudoSourceValue::getFixedStack(RegSaveFrameIndex),
        MachineMemOperand::MOStore, Offset,
        /*Size=*/16, /*Align=*/16);
    BuildMI(XMMSaveMBB, DL, TII->get(X86::MOVAPSmr))
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
  const MachineFunction *MF = BB->getParent();
  const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();
  BitVector ReservedRegs = TRI->getReservedRegs(*MF);

  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    if (!MO.isReg() || !MO.isUse() || MO.isKill()) continue;
    unsigned Reg = MO.getReg();
    if (Reg != X86::EFLAGS) continue;
    copy0MBB->addLiveIn(Reg);
    sinkMBB->addLiveIn(Reg);
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
X86TargetLowering::EmitLoweredMingwAlloca(MachineInstr *MI,
                                          MachineBasicBlock *BB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();

  // The lowering is pretty easy: we're just emitting the call to _alloca.  The
  // non-trivial part is impdef of ESP.
  // FIXME: The code should be tweaked as soon as we'll try to do codegen for
  // mingw-w64.

  BuildMI(*BB, MI, DL, TII->get(X86::CALLpcrel32))
    .addExternalSymbol("_alloca")
    .addReg(X86::EAX, RegState::Implicit)
    .addReg(X86::ESP, RegState::Implicit)
    .addReg(X86::EAX, RegState::Define | RegState::Implicit)
    .addReg(X86::ESP, RegState::Define | RegState::Implicit)
    .addReg(X86::EFLAGS, RegState::Define | RegState::Implicit);

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
  bool IsWin64 = Subtarget->isTargetWin64();
  
  assert(MI->getOperand(3).isGlobal() && "This should be a global");
  
  if (Subtarget->is64Bit()) {
    MachineInstrBuilder MIB = BuildMI(*BB, MI, DL,
                                      TII->get(X86::MOV64rm), X86::RDI)
    .addReg(X86::RIP)
    .addImm(0).addReg(0)
    .addGlobalAddress(MI->getOperand(3).getGlobal(), 0, 
                      MI->getOperand(3).getTargetFlags())
    .addReg(0);
    MIB = BuildMI(*BB, MI, DL, TII->get(IsWin64 ? X86::WINCALL64m : X86::CALL64m));
    addDirectMem(MIB, X86::RDI);
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
  }
  
  MI->eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

MachineBasicBlock *
X86TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                               MachineBasicBlock *BB) const {
  switch (MI->getOpcode()) {
  default: assert(false && "Unexpected instr type to insert");
  case X86::MINGW_ALLOCA:
    return EmitLoweredMingwAlloca(MI, BB);
  case X86::TLSCall_32:
  case X86::TLSCall_64:
    return EmitLoweredTLSCall(MI, BB);
  case X86::CMOV_GR8:
  case X86::CMOV_V1I64:
  case X86::CMOV_FR32:
  case X86::CMOV_FR64:
  case X86::CMOV_V4F32:
  case X86::CMOV_V2F64:
  case X86::CMOV_V2I64:
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
      F->getRegInfo().createVirtualRegister(X86::GR16RegisterClass);
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
    return EmitPCMP(MI, BB, 3, false /* in-mem */);
  case X86::PCMPISTRM128MEM:
  case X86::VPCMPISTRM128MEM:
    return EmitPCMP(MI, BB, 3, true /* in-mem */);
  case X86::PCMPESTRM128REG:
  case X86::VPCMPESTRM128REG:
    return EmitPCMP(MI, BB, 5, false /* in mem */);
  case X86::PCMPESTRM128MEM:
  case X86::VPCMPESTRM128MEM:
    return EmitPCMP(MI, BB, 5, true /* in mem */);

    // Atomic Lowering.
  case X86::ATOMAND32:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND32rr,
                                               X86::AND32ri, X86::MOV32rm,
                                               X86::LCMPXCHG32,
                                               X86::NOT32r, X86::EAX,
                                               X86::GR32RegisterClass);
  case X86::ATOMOR32:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::OR32rr,
                                               X86::OR32ri, X86::MOV32rm,
                                               X86::LCMPXCHG32,
                                               X86::NOT32r, X86::EAX,
                                               X86::GR32RegisterClass);
  case X86::ATOMXOR32:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::XOR32rr,
                                               X86::XOR32ri, X86::MOV32rm,
                                               X86::LCMPXCHG32,
                                               X86::NOT32r, X86::EAX,
                                               X86::GR32RegisterClass);
  case X86::ATOMNAND32:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND32rr,
                                               X86::AND32ri, X86::MOV32rm,
                                               X86::LCMPXCHG32,
                                               X86::NOT32r, X86::EAX,
                                               X86::GR32RegisterClass, true);
  case X86::ATOMMIN32:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVL32rr);
  case X86::ATOMMAX32:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVG32rr);
  case X86::ATOMUMIN32:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVB32rr);
  case X86::ATOMUMAX32:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVA32rr);

  case X86::ATOMAND16:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND16rr,
                                               X86::AND16ri, X86::MOV16rm,
                                               X86::LCMPXCHG16,
                                               X86::NOT16r, X86::AX,
                                               X86::GR16RegisterClass);
  case X86::ATOMOR16:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::OR16rr,
                                               X86::OR16ri, X86::MOV16rm,
                                               X86::LCMPXCHG16,
                                               X86::NOT16r, X86::AX,
                                               X86::GR16RegisterClass);
  case X86::ATOMXOR16:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::XOR16rr,
                                               X86::XOR16ri, X86::MOV16rm,
                                               X86::LCMPXCHG16,
                                               X86::NOT16r, X86::AX,
                                               X86::GR16RegisterClass);
  case X86::ATOMNAND16:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND16rr,
                                               X86::AND16ri, X86::MOV16rm,
                                               X86::LCMPXCHG16,
                                               X86::NOT16r, X86::AX,
                                               X86::GR16RegisterClass, true);
  case X86::ATOMMIN16:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVL16rr);
  case X86::ATOMMAX16:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVG16rr);
  case X86::ATOMUMIN16:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVB16rr);
  case X86::ATOMUMAX16:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVA16rr);

  case X86::ATOMAND8:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND8rr,
                                               X86::AND8ri, X86::MOV8rm,
                                               X86::LCMPXCHG8,
                                               X86::NOT8r, X86::AL,
                                               X86::GR8RegisterClass);
  case X86::ATOMOR8:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::OR8rr,
                                               X86::OR8ri, X86::MOV8rm,
                                               X86::LCMPXCHG8,
                                               X86::NOT8r, X86::AL,
                                               X86::GR8RegisterClass);
  case X86::ATOMXOR8:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::XOR8rr,
                                               X86::XOR8ri, X86::MOV8rm,
                                               X86::LCMPXCHG8,
                                               X86::NOT8r, X86::AL,
                                               X86::GR8RegisterClass);
  case X86::ATOMNAND8:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND8rr,
                                               X86::AND8ri, X86::MOV8rm,
                                               X86::LCMPXCHG8,
                                               X86::NOT8r, X86::AL,
                                               X86::GR8RegisterClass, true);
  // FIXME: There are no CMOV8 instructions; MIN/MAX need some other way.
  // This group is for 64-bit host.
  case X86::ATOMAND64:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND64rr,
                                               X86::AND64ri32, X86::MOV64rm,
                                               X86::LCMPXCHG64,
                                               X86::NOT64r, X86::RAX,
                                               X86::GR64RegisterClass);
  case X86::ATOMOR64:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::OR64rr,
                                               X86::OR64ri32, X86::MOV64rm,
                                               X86::LCMPXCHG64,
                                               X86::NOT64r, X86::RAX,
                                               X86::GR64RegisterClass);
  case X86::ATOMXOR64:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::XOR64rr,
                                               X86::XOR64ri32, X86::MOV64rm,
                                               X86::LCMPXCHG64,
                                               X86::NOT64r, X86::RAX,
                                               X86::GR64RegisterClass);
  case X86::ATOMNAND64:
    return EmitAtomicBitwiseWithCustomInserter(MI, BB, X86::AND64rr,
                                               X86::AND64ri32, X86::MOV64rm,
                                               X86::LCMPXCHG64,
                                               X86::NOT64r, X86::RAX,
                                               X86::GR64RegisterClass, true);
  case X86::ATOMMIN64:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVL64rr);
  case X86::ATOMMAX64:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVG64rr);
  case X86::ATOMUMIN64:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVB64rr);
  case X86::ATOMUMAX64:
    return EmitAtomicMinMaxWithCustomInserter(MI, BB, X86::CMOVA64rr);

  // This group does 64-bit operations on a 32-bit host.
  case X86::ATOMAND6432:
    return EmitAtomicBit6432WithCustomInserter(MI, BB,
                                               X86::AND32rr, X86::AND32rr,
                                               X86::AND32ri, X86::AND32ri,
                                               false);
  case X86::ATOMOR6432:
    return EmitAtomicBit6432WithCustomInserter(MI, BB,
                                               X86::OR32rr, X86::OR32rr,
                                               X86::OR32ri, X86::OR32ri,
                                               false);
  case X86::ATOMXOR6432:
    return EmitAtomicBit6432WithCustomInserter(MI, BB,
                                               X86::XOR32rr, X86::XOR32rr,
                                               X86::XOR32ri, X86::XOR32ri,
                                               false);
  case X86::ATOMNAND6432:
    return EmitAtomicBit6432WithCustomInserter(MI, BB,
                                               X86::AND32rr, X86::AND32rr,
                                               X86::AND32ri, X86::AND32ri,
                                               true);
  case X86::ATOMADD6432:
    return EmitAtomicBit6432WithCustomInserter(MI, BB,
                                               X86::ADD32rr, X86::ADC32rr,
                                               X86::ADD32ri, X86::ADC32ri,
                                               false);
  case X86::ATOMSUB6432:
    return EmitAtomicBit6432WithCustomInserter(MI, BB,
                                               X86::SUB32rr, X86::SBB32rr,
                                               X86::SUB32ri, X86::SBB32ri,
                                               false);
  case X86::ATOMSWAP6432:
    return EmitAtomicBit6432WithCustomInserter(MI, BB,
                                               X86::MOV32rr, X86::MOV32rr,
                                               X86::MOV32ri, X86::MOV32ri,
                                               false);
  case X86::VASTART_SAVE_XMM_REGS:
    return EmitVAStartSaveXMMRegsWithCustomInserter(MI, BB);
  }
}

//===----------------------------------------------------------------------===//
//                           X86 Optimization Hooks
//===----------------------------------------------------------------------===//

void X86TargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
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
  case X86ISD::ADD:
  case X86ISD::SUB:
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
    KnownZero |= APInt::getHighBitsSet(Mask.getBitWidth(),
                                       Mask.getBitWidth() - 1);
    break;
  }
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

/// PerformShuffleCombine - Combine a vector_shuffle that is equal to
/// build_vector load1, load2, load3, load4, <0, 1, 2, 3> into a 128-bit load
/// if the load addresses are consecutive, non-overlapping, and in the right
/// order.
static SDValue PerformShuffleCombine(SDNode *N, SelectionDAG &DAG,
                                     const TargetLowering &TLI) {
  DebugLoc dl = N->getDebugLoc();
  EVT VT = N->getValueType(0);

  if (VT.getSizeInBits() != 128)
    return SDValue();

  SmallVector<SDValue, 16> Elts;
  for (unsigned i = 0, e = VT.getVectorNumElements(); i != e; ++i)
    Elts.push_back(getShuffleScalarElt(N, i, DAG, 0));

  return EltsFromConsecutiveLoads(VT, Elts, dl, DAG);
}

/// PerformEXTRACT_VECTOR_ELTCombine - Detect vector gather/scatter index
/// generation and convert it from being a bunch of shuffles and extracts
/// to a simple store and scalar loads to extract the elements.
static SDValue PerformEXTRACT_VECTOR_ELTCombine(SDNode *N, SelectionDAG &DAG,
                                                const TargetLowering &TLI) {
  SDValue InputVector = N->getOperand(0);

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
  SDValue Ch = DAG.getStore(DAG.getEntryNode(), dl, InputVector, StackPtr, NULL,
                            0, false, false, 0);

  // Replace each use (extract) with a load of the appropriate element.
  for (SmallVectorImpl<SDNode *>::iterator UI = Uses.begin(),
       UE = Uses.end(); UI != UE; ++UI) {
    SDNode *Extract = *UI;

    // Compute the element's address.
    SDValue Idx = Extract->getOperand(1);
    unsigned EltSize =
        InputVector.getValueType().getVectorElementType().getSizeInBits()/8;
    uint64_t Offset = EltSize * cast<ConstantSDNode>(Idx)->getZExtValue();
    SDValue OffsetVal = DAG.getConstant(Offset, TLI.getPointerTy());

    SDValue ScalarAddr = DAG.getNode(ISD::ADD, dl, Idx.getValueType(),
                                     OffsetVal, StackPtr);

    // Load the scalar.
    SDValue LoadScalar = DAG.getLoad(Extract->getValueType(0), dl, Ch,
                                     ScalarAddr, NULL, 0, false, false, 0);

    // Replace the exact with the load.
    DAG.ReplaceAllUsesOfValueWith(SDValue(Extract, 0), LoadScalar);
  }

  // The replacement was made in place; don't return anything.
  return SDValue();
}

/// PerformSELECTCombine - Do target-specific dag combines on SELECT nodes.
static SDValue PerformSELECTCombine(SDNode *N, SelectionDAG &DAG,
                                    const X86Subtarget *Subtarget) {
  DebugLoc DL = N->getDebugLoc();
  SDValue Cond = N->getOperand(0);
  // Get the LHS/RHS of the select.
  SDValue LHS = N->getOperand(1);
  SDValue RHS = N->getOperand(2);

  // If we have SSE[12] support, try to form min/max nodes. SSE min/max
  // instructions match the semantics of the common C idiom x<y?x:y but not
  // x<=y?x:y, because of how they handle negative zero (which can be
  // ignored in unsafe-math mode).
  if (Subtarget->hasSSE2() &&
      (LHS.getValueType() == MVT::f32 || LHS.getValueType() == MVT::f64) &&
      Cond.getOpcode() == ISD::SETCC) {
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
          if (!UnsafeFPMath &&
              !(DAG.isKnownNeverZero(LHS) || DAG.isKnownNeverZero(RHS)))
            break;
          std::swap(LHS, RHS);
        }
        Opcode = X86ISD::FMIN;
        break;
      case ISD::SETOLE:
        // Converting this to a min would handle comparisons between positive
        // and negative zero incorrectly.
        if (!UnsafeFPMath &&
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
        if (!UnsafeFPMath &&
            !DAG.isKnownNeverZero(LHS) && !DAG.isKnownNeverZero(LHS))
          break;
        Opcode = X86ISD::FMAX;
        break;
      case ISD::SETUGT:
        // Converting this to a max would handle NaNs incorrectly, and swapping
        // the operands would cause it to handle comparisons between positive
        // and negative zero incorrectly.
        if (!DAG.isKnownNeverNaN(LHS) || !DAG.isKnownNeverNaN(RHS)) {
          if (!UnsafeFPMath &&
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
        if (!UnsafeFPMath &&
            !(DAG.isKnownNeverZero(LHS) || DAG.isKnownNeverZero(RHS))) {
          if (!DAG.isKnownNeverNaN(LHS) || !DAG.isKnownNeverNaN(RHS))
            break;
          std::swap(LHS, RHS);
        }
        Opcode = X86ISD::FMIN;
        break;
      case ISD::SETUGT:
        // Converting this to a min would handle NaNs incorrectly.
        if (!UnsafeFPMath &&
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
        if (!UnsafeFPMath &&
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

  return SDValue();
}

/// Optimize X86ISD::CMOV [LHS, RHS, CONDCODE (e.g. X86::COND_NE), CONDVAL]
static SDValue PerformCMOVCombine(SDNode *N, SelectionDAG &DAG,
                                  TargetLowering::DAGCombinerInfo &DCI) {
  DebugLoc DL = N->getDebugLoc();

  // If the flag operand isn't dead, don't touch this CMOV.
  if (N->getNumValues() == 2 && !SDValue(N, 1).use_empty())
    return SDValue();

  // If this is a select between two integer constants, try to do some
  // optimizations.  Note that the operands are ordered the opposite of SELECT
  // operands.
  if (ConstantSDNode *TrueC = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
    if (ConstantSDNode *FalseC = dyn_cast<ConstantSDNode>(N->getOperand(0))) {
      // Canonicalize the TrueC/FalseC values so that TrueC (the true value) is
      // larger than FalseC (the false value).
      X86::CondCode CC = (X86::CondCode)N->getConstantOperandVal(2);

      if (TrueC->getAPIntValue().ult(FalseC->getAPIntValue())) {
        CC = X86::GetOppositeBranchCondition(CC);
        std::swap(TrueC, FalseC);
      }

      // Optimize C ? 8 : 0 -> zext(setcc(C)) << 3.  Likewise for any pow2/0.
      // This is efficient for any integer data type (including i8/i16) and
      // shift amount.
      if (FalseC->getAPIntValue() == 0 && TrueC->getAPIntValue().isPowerOf2()) {
        SDValue Cond = N->getOperand(3);
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
        SDValue Cond = N->getOperand(3);
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
          SDValue Cond = N->getOperand(3);
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
  if (N1C && N0.getOpcode() == ISD::AND &&
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

  return SDValue();
}

/// PerformShiftCombine - Transforms vector shift nodes to use vector shifts
///                       when possible.
static SDValue PerformShiftCombine(SDNode* N, SelectionDAG &DAG,
                                   const X86Subtarget *Subtarget) {
  EVT VT = N->getValueType(0);
  if (!VT.isVector() && VT.isInteger() &&
      N->getOpcode() == ISD::SHL)
    return PerformSHLCombine(N, DAG);

  // On X86 with SSE2 support, we can transform this to a vector shift if
  // all elements are shifted by the same amount.  We can't do this in legalize
  // because the a constant vector is typically transformed to a constant pool
  // so we have no knowledge of the shift amount.
  if (!Subtarget->hasSSE2())
    return SDValue();

  if (VT != MVT::v2i64 && VT != MVT::v4i32 && VT != MVT::v8i16)
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
    if (BaseShAmt.getNode() == 0)
      BaseShAmt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, ShAmtOp,
                              DAG.getIntPtrConstant(0));
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
    break;
  case ISD::SHL:
    if (VT == MVT::v2i64)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                         DAG.getConstant(Intrinsic::x86_sse2_pslli_q, MVT::i32),
                         ValOp, BaseShAmt);
    if (VT == MVT::v4i32)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                         DAG.getConstant(Intrinsic::x86_sse2_pslli_d, MVT::i32),
                         ValOp, BaseShAmt);
    if (VT == MVT::v8i16)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                         DAG.getConstant(Intrinsic::x86_sse2_pslli_w, MVT::i32),
                         ValOp, BaseShAmt);
    break;
  case ISD::SRA:
    if (VT == MVT::v4i32)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                         DAG.getConstant(Intrinsic::x86_sse2_psrai_d, MVT::i32),
                         ValOp, BaseShAmt);
    if (VT == MVT::v8i16)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                         DAG.getConstant(Intrinsic::x86_sse2_psrai_w, MVT::i32),
                         ValOp, BaseShAmt);
    break;
  case ISD::SRL:
    if (VT == MVT::v2i64)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                         DAG.getConstant(Intrinsic::x86_sse2_psrli_q, MVT::i32),
                         ValOp, BaseShAmt);
    if (VT == MVT::v4i32)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                         DAG.getConstant(Intrinsic::x86_sse2_psrli_d, MVT::i32),
                         ValOp, BaseShAmt);
    if (VT ==  MVT::v8i16)
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                         DAG.getConstant(Intrinsic::x86_sse2_psrli_w, MVT::i32),
                         ValOp, BaseShAmt);
    break;
  }
  return SDValue();
}

static SDValue PerformOrCombine(SDNode *N, SelectionDAG &DAG,
                                TargetLowering::DAGCombinerInfo &DCI,
                                const X86Subtarget *Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i16 && VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  // fold (or (x << c) | (y >> (64 - c))) ==> (shld64 x, y, c)
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
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

/// PerformSTORECombine - Do target-specific dag combines on STORE nodes.
static SDValue PerformSTORECombine(SDNode *N, SelectionDAG &DAG,
                                   const X86Subtarget *Subtarget) {
  // Turn load->store of MMX types into GPR load/stores.  This avoids clobbering
  // the FP state in cases where an emms may be missing.
  // A preferable solution to the general problem is to figure out the right
  // places to insert EMMS.  This qualifies as a quick hack.

  // Similarly, turn load->store of i64 into double load/stores in 32-bit mode.
  StoreSDNode *St = cast<StoreSDNode>(N);
  EVT VT = St->getValue().getValueType();
  if (VT.getSizeInBits() != 64)
    return SDValue();

  const Function *F = DAG.getMachineFunction().getFunction();
  bool NoImplicitFloatOps = F->hasFnAttr(Attribute::NoImplicitFloat);
  bool F64IsLegal = !UseSoftFloat && !NoImplicitFloatOps
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
      for (unsigned i=0, e = ChainVal->getNumOperands(); i != e; ++i) {
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
      SDValue NewLd = DAG.getLoad(LdVT, LdDL, Ld->getChain(),
                                  Ld->getBasePtr(), Ld->getSrcValue(),
                                  Ld->getSrcValueOffset(), Ld->isVolatile(),
                                  Ld->isNonTemporal(), Ld->getAlignment());
      SDValue NewChain = NewLd.getValue(1);
      if (TokenFactorIndex != -1) {
        Ops.push_back(NewChain);
        NewChain = DAG.getNode(ISD::TokenFactor, LdDL, MVT::Other, &Ops[0],
                               Ops.size());
      }
      return DAG.getStore(NewChain, StDL, NewLd, St->getBasePtr(),
                          St->getSrcValue(), St->getSrcValueOffset(),
                          St->isVolatile(), St->isNonTemporal(),
                          St->getAlignment());
    }

    // Otherwise, lower to two pairs of 32-bit loads / stores.
    SDValue LoAddr = Ld->getBasePtr();
    SDValue HiAddr = DAG.getNode(ISD::ADD, LdDL, MVT::i32, LoAddr,
                                 DAG.getConstant(4, MVT::i32));

    SDValue LoLd = DAG.getLoad(MVT::i32, LdDL, Ld->getChain(), LoAddr,
                               Ld->getSrcValue(), Ld->getSrcValueOffset(),
                               Ld->isVolatile(), Ld->isNonTemporal(),
                               Ld->getAlignment());
    SDValue HiLd = DAG.getLoad(MVT::i32, LdDL, Ld->getChain(), HiAddr,
                               Ld->getSrcValue(), Ld->getSrcValueOffset()+4,
                               Ld->isVolatile(), Ld->isNonTemporal(),
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
                                St->getSrcValue(), St->getSrcValueOffset(),
                                St->isVolatile(), St->isNonTemporal(),
                                St->getAlignment());
    SDValue HiSt = DAG.getStore(NewChain, StDL, HiLd, HiAddr,
                                St->getSrcValue(),
                                St->getSrcValueOffset() + 4,
                                St->isVolatile(),
                                St->isNonTemporal(),
                                MinAlign(St->getAlignment(), 4));
    return DAG.getNode(ISD::TokenFactor, StDL, MVT::Other, LoSt, HiSt);
  }
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
  if (Op.getOpcode() == ISD::BIT_CONVERT)
    Op = Op.getOperand(0);
  EVT VT = N->getValueType(0), OpVT = Op.getValueType();
  if (Op.getOpcode() == X86ISD::VZEXT_LOAD &&
      VT.getVectorElementType().getSizeInBits() ==
      OpVT.getVectorElementType().getSizeInBits()) {
    return DAG.getNode(ISD::BIT_CONVERT, N->getDebugLoc(), VT, Op);
  }
  return SDValue();
}

static SDValue PerformZExtCombine(SDNode *N, SelectionDAG &DAG) {
  // (i32 zext (and (i8  x86isd::setcc_carry), 1)) ->
  //           (and (i32 x86isd::setcc_carry), 1)
  // This eliminates the zext. This transformation is necessary because
  // ISD::SETCC is always legalized to i8.
  DebugLoc dl = N->getDebugLoc();
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
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

  return SDValue();
}

SDValue X86TargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default: break;
  case ISD::EXTRACT_VECTOR_ELT:
                        return PerformEXTRACT_VECTOR_ELTCombine(N, DAG, *this);
  case ISD::SELECT:         return PerformSELECTCombine(N, DAG, Subtarget);
  case X86ISD::CMOV:        return PerformCMOVCombine(N, DAG, DCI);
  case ISD::MUL:            return PerformMulCombine(N, DAG, DCI);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:            return PerformShiftCombine(N, DAG, Subtarget);
  case ISD::OR:             return PerformOrCombine(N, DAG, DCI, Subtarget);
  case ISD::STORE:          return PerformSTORECombine(N, DAG, Subtarget);
  case X86ISD::FXOR:
  case X86ISD::FOR:         return PerformFORCombine(N, DAG);
  case X86ISD::FAND:        return PerformFANDCombine(N, DAG);
  case X86ISD::BT:          return PerformBTCombine(N, DAG, DCI);
  case X86ISD::VZEXT_MOVL:  return PerformVZEXT_MOVLCombine(N, DAG);
  case ISD::ZERO_EXTEND:    return PerformZExtCombine(N, DAG);
  case X86ISD::SHUFPS:      // Handle all target specific shuffles
  case X86ISD::SHUFPD:
  case X86ISD::PUNPCKHBW:
  case X86ISD::PUNPCKHWD:
  case X86ISD::PUNPCKHDQ:
  case X86ISD::PUNPCKHQDQ:
  case X86ISD::UNPCKHPS:
  case X86ISD::UNPCKHPD:
  case X86ISD::PUNPCKLBW:
  case X86ISD::PUNPCKLWD:
  case X86ISD::PUNPCKLDQ:
  case X86ISD::PUNPCKLQDQ:
  case X86ISD::UNPCKLPS:
  case X86ISD::UNPCKLPD:
  case X86ISD::MOVHLPS:
  case X86ISD::MOVLHPS:
  case X86ISD::PSHUFD:
  case X86ISD::PSHUFHW:
  case X86ISD::PSHUFLW:
  case X86ISD::MOVSS:
  case X86ISD::MOVSD:
  case ISD::VECTOR_SHUFFLE: return PerformShuffleCombine(N, DAG, *this);
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

static bool LowerToBSwap(CallInst *CI) {
  // FIXME: this should verify that we are targetting a 486 or better.  If not,
  // we will turn this bswap into something that will be lowered to logical ops
  // instead of emitting the bswap asm.  For now, we don't support 486 or lower
  // so don't worry about this.

  // Verify this is a simple bswap.
  if (CI->getNumArgOperands() != 1 ||
      CI->getType() != CI->getArgOperand(0)->getType() ||
      !CI->getType()->isIntegerTy())
    return false;

  const IntegerType *Ty = dyn_cast<IntegerType>(CI->getType());
  if (!Ty || Ty->getBitWidth() % 16 != 0)
    return false;

  // Okay, we can do this xform, do so now.
  const Type *Tys[] = { Ty };
  Module *M = CI->getParent()->getParent()->getParent();
  Constant *Int = Intrinsic::getDeclaration(M, Intrinsic::bswap, Tys, 1);

  Value *Op = CI->getArgOperand(0);
  Op = CallInst::Create(Int, Op, CI->getName(), CI);

  CI->replaceAllUsesWith(Op);
  CI->eraseFromParent();
  return true;
}

bool X86TargetLowering::ExpandInlineAsm(CallInst *CI) const {
  InlineAsm *IA = cast<InlineAsm>(CI->getCalledValue());
  std::vector<InlineAsm::ConstraintInfo> Constraints = IA->ParseConstraints();

  std::string AsmStr = IA->getAsmString();

  // TODO: should remove alternatives from the asmstring: "foo {a|b}" -> "foo a"
  SmallVector<StringRef, 4> AsmPieces;
  SplitString(AsmStr, AsmPieces, "\n");  // ; as separator?

  switch (AsmPieces.size()) {
  default: return false;
  case 1:
    AsmStr = AsmPieces[0];
    AsmPieces.clear();
    SplitString(AsmStr, AsmPieces, " \t");  // Split with whitespace.

    // bswap $0
    if (AsmPieces.size() == 2 &&
        (AsmPieces[0] == "bswap" ||
         AsmPieces[0] == "bswapq" ||
         AsmPieces[0] == "bswapl") &&
        (AsmPieces[1] == "$0" ||
         AsmPieces[1] == "${0:q}")) {
      // No need to check constraints, nothing other than the equivalent of
      // "=r,0" would be valid here.
      return LowerToBSwap(CI);
    }
    // rorw $$8, ${0:w}  -->  llvm.bswap.i16
    if (CI->getType()->isIntegerTy(16) &&
        AsmPieces.size() == 3 &&
        (AsmPieces[0] == "rorw" || AsmPieces[0] == "rolw") &&
        AsmPieces[1] == "$$8," &&
        AsmPieces[2] == "${0:w}" &&
        IA->getConstraintString().compare(0, 5, "=r,0,") == 0) {
      AsmPieces.clear();
      const std::string &Constraints = IA->getConstraintString();
      SplitString(StringRef(Constraints).substr(5), AsmPieces, ",");
      std::sort(AsmPieces.begin(), AsmPieces.end());
      if (AsmPieces.size() == 4 &&
          AsmPieces[0] == "~{cc}" &&
          AsmPieces[1] == "~{dirflag}" &&
          AsmPieces[2] == "~{flags}" &&
          AsmPieces[3] == "~{fpsr}") {
        return LowerToBSwap(CI);
      }
    }
    break;
  case 3:
    if (CI->getType()->isIntegerTy(64) &&
        Constraints.size() >= 2 &&
        Constraints[0].Codes.size() == 1 && Constraints[0].Codes[0] == "A" &&
        Constraints[1].Codes.size() == 1 && Constraints[1].Codes[0] == "0") {
      // bswap %eax / bswap %edx / xchgl %eax, %edx  -> llvm.bswap.i64
      SmallVector<StringRef, 4> Words;
      SplitString(AsmPieces[0], Words, " \t");
      if (Words.size() == 2 && Words[0] == "bswap" && Words[1] == "%eax") {
        Words.clear();
        SplitString(AsmPieces[1], Words, " \t");
        if (Words.size() == 2 && Words[0] == "bswap" && Words[1] == "%edx") {
          Words.clear();
          SplitString(AsmPieces[2], Words, " \t,");
          if (Words.size() == 3 && Words[0] == "xchgl" && Words[1] == "%eax" &&
              Words[2] == "%edx") {
            return LowerToBSwap(CI);
          }
        }
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
    case 'A':
      return C_Register;
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
    case 'e':
    case 'Z':
      return C_Other;
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
                                                     char Constraint,
                                                     std::vector<SDValue>&Ops,
                                                     SelectionDAG &DAG) const {
  SDValue Result(0, 0);

  switch (Constraint) {
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
      if ((int8_t)C->getSExtValue() == C->getSExtValue()) {
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

std::vector<unsigned> X86TargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  EVT VT) const {
  if (Constraint.size() == 1) {
    // FIXME: not handling fp-stack yet!
    switch (Constraint[0]) {      // GCC X86 Constraint Letters
    default: break;  // Unknown constraint letter
    case 'q':   // GENERAL_REGS in 64-bit mode, Q_REGS in 32-bit mode.
      if (Subtarget->is64Bit()) {
        if (VT == MVT::i32)
          return make_vector<unsigned>(X86::EAX, X86::EDX, X86::ECX, X86::EBX,
                                       X86::ESI, X86::EDI, X86::R8D, X86::R9D,
                                       X86::R10D,X86::R11D,X86::R12D,
                                       X86::R13D,X86::R14D,X86::R15D,
                                       X86::EBP, X86::ESP, 0);
        else if (VT == MVT::i16)
          return make_vector<unsigned>(X86::AX,  X86::DX,  X86::CX, X86::BX,
                                       X86::SI,  X86::DI,  X86::R8W,X86::R9W,
                                       X86::R10W,X86::R11W,X86::R12W,
                                       X86::R13W,X86::R14W,X86::R15W,
                                       X86::BP,  X86::SP, 0);
        else if (VT == MVT::i8)
          return make_vector<unsigned>(X86::AL,  X86::DL,  X86::CL, X86::BL,
                                       X86::SIL, X86::DIL, X86::R8B,X86::R9B,
                                       X86::R10B,X86::R11B,X86::R12B,
                                       X86::R13B,X86::R14B,X86::R15B,
                                       X86::BPL, X86::SPL, 0);

        else if (VT == MVT::i64)
          return make_vector<unsigned>(X86::RAX, X86::RDX, X86::RCX, X86::RBX,
                                       X86::RSI, X86::RDI, X86::R8,  X86::R9,
                                       X86::R10, X86::R11, X86::R12,
                                       X86::R13, X86::R14, X86::R15,
                                       X86::RBP, X86::RSP, 0);

        break;
      }
      // 32-bit fallthrough
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
                                                EVT VT) const {
  // First, see if this is a constraint that directly corresponds to an LLVM
  // register class.
  if (Constraint.size() == 1) {
    // GCC Constraint Letters
    switch (Constraint[0]) {
    default: break;
    case 'r':   // GENERAL_REGS
    case 'l':   // INDEX_REGS
      if (VT == MVT::i8)
        return std::make_pair(0U, X86::GR8RegisterClass);
      if (VT == MVT::i16)
        return std::make_pair(0U, X86::GR16RegisterClass);
      if (VT == MVT::i32 || !Subtarget->is64Bit())
        return std::make_pair(0U, X86::GR32RegisterClass);
      return std::make_pair(0U, X86::GR64RegisterClass);
    case 'R':   // LEGACY_REGS
      if (VT == MVT::i8)
        return std::make_pair(0U, X86::GR8_NOREXRegisterClass);
      if (VT == MVT::i16)
        return std::make_pair(0U, X86::GR16_NOREXRegisterClass);
      if (VT == MVT::i32 || !Subtarget->is64Bit())
        return std::make_pair(0U, X86::GR32_NOREXRegisterClass);
      return std::make_pair(0U, X86::GR64_NOREXRegisterClass);
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
    case 'Y':   // SSE_REGS if SSE2 allowed
      if (!Subtarget->hasSSE2()) break;
      // FALL THROUGH.
    case 'x':   // SSE_REGS if SSE1 allowed
      if (!Subtarget->hasSSE1()) break;

      switch (VT.getSimpleVT().SimpleTy) {
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
    // Map st(0) -> st(7) -> ST0
    if (Constraint.size() == 7 && Constraint[0] == '{' &&
        tolower(Constraint[1]) == 's' &&
        tolower(Constraint[2]) == 't' &&
        Constraint[3] == '(' &&
        (Constraint[4] >= '0' && Constraint[4] <= '7') &&
        Constraint[5] == ')' &&
        Constraint[6] == '}') {

      Res.first = X86::ST0+Constraint[4]-'0';
      Res.second = X86::RFP80RegisterClass;
      return Res;
    }

    // GCC allows "st(0)" to be called just plain "st".
    if (StringRef("{st}").equals_lower(Constraint)) {
      Res.first = X86::ST0;
      Res.second = X86::RFP80RegisterClass;
      return Res;
    }

    // flags -> EFLAGS
    if (StringRef("{flags}").equals_lower(Constraint)) {
      Res.first = X86::EFLAGS;
      Res.second = X86::CCRRegisterClass;
      return Res;
    }

    // 'A' means EAX + EDX.
    if (Constraint == "A") {
      Res.first = X86::EAX;
      Res.second = X86::GR32_ADRegisterClass;
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
  if (Res.second == X86::GR16RegisterClass) {
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
        Res.second = X86::GR8RegisterClass;
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
        Res.second = X86::GR32RegisterClass;
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
        Res.second = X86::GR64RegisterClass;
      }
    }
  } else if (Res.second == X86::FR32RegisterClass ||
             Res.second == X86::FR64RegisterClass ||
             Res.second == X86::VR128RegisterClass) {
    // Handle references to XMM physical registers that got mapped into the
    // wrong class.  This can happen with constraints like {xmm0} where the
    // target independent register mapper will just pick the first match it can
    // find, ignoring the required type.
    if (VT == MVT::f32)
      Res.second = X86::FR32RegisterClass;
    else if (VT == MVT::f64)
      Res.second = X86::FR64RegisterClass;
    else if (X86::VR128RegisterClass->hasType(VT))
      Res.second = X86::VR128RegisterClass;
  }

  return Res;
}
