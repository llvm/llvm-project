//===-- X86ISelLowering.cpp - X86 DAG Lowering Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

// FIXME: temporary.
static cl::opt<bool> EnableFastCC("enable-x86-fastcc", cl::Hidden,
                                  cl::desc("Enable fastcc on X86"));
X86TargetLowering::X86TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  Subtarget = &TM.getSubtarget<X86Subtarget>();
  X86ScalarSSE = Subtarget->hasSSE2();
  X86StackPtr = Subtarget->is64Bit() ? X86::RSP : X86::ESP;

  // Set up the TargetLowering object.

  // X86 is weird, it always uses i8 for shift amounts and setcc results.
  setShiftAmountType(MVT::i8);
  setSetCCResultType(MVT::i8);
  setSetCCResultContents(ZeroOrOneSetCCResult);
  setSchedulingPreference(SchedulingForRegPressure);
  setShiftAmountFlavor(Mask);   // shl X, 32 == shl X, 0
  setStackPointerRegisterToSaveRestore(X86StackPtr);

  if (!Subtarget->isTargetDarwin())
    // Darwin should use _setjmp/_longjmp instead of setjmp/longjmp.
    setUseUnderscoreSetJmpLongJmp(true);
    
  // Add legal addressing mode scale values.
  addLegalAddressScale(8);
  addLegalAddressScale(4);
  addLegalAddressScale(2);
  // Enter the ones which require both scale + index last. These are more
  // expensive.
  addLegalAddressScale(9);
  addLegalAddressScale(5);
  addLegalAddressScale(3);
  
  // Set up the register classes.
  addRegisterClass(MVT::i8, X86::GR8RegisterClass);
  addRegisterClass(MVT::i16, X86::GR16RegisterClass);
  addRegisterClass(MVT::i32, X86::GR32RegisterClass);
  if (Subtarget->is64Bit())
    addRegisterClass(MVT::i64, X86::GR64RegisterClass);

  setLoadXAction(ISD::SEXTLOAD, MVT::i1, Expand);

  // Promote all UINT_TO_FP to larger SINT_TO_FP's, as X86 doesn't have this
  // operation.
  setOperationAction(ISD::UINT_TO_FP       , MVT::i1   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i8   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i16  , Promote);

  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::UINT_TO_FP     , MVT::i64  , Expand);
    setOperationAction(ISD::UINT_TO_FP     , MVT::i32  , Promote);
  } else {
    if (X86ScalarSSE)
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
  if (X86ScalarSSE)
    setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Promote);
  else {
    setOperationAction(ISD::SINT_TO_FP     , MVT::i16  , Custom);
    setOperationAction(ISD::SINT_TO_FP     , MVT::i32  , Custom);
  }

  if (!Subtarget->is64Bit()) {
    // Custom lower SINT_TO_FP and FP_TO_SINT from/to i64 in 32-bit mode.
    setOperationAction(ISD::SINT_TO_FP     , MVT::i64  , Custom);
    setOperationAction(ISD::FP_TO_SINT     , MVT::i64  , Custom);
  }

  // Promote i1/i8 FP_TO_SINT to larger FP_TO_SINTS's, as X86 doesn't have
  // this operation.
  setOperationAction(ISD::FP_TO_SINT       , MVT::i1   , Promote);
  setOperationAction(ISD::FP_TO_SINT       , MVT::i8   , Promote);

  if (X86ScalarSSE) {
    setOperationAction(ISD::FP_TO_SINT     , MVT::i16  , Promote);
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
    if (X86ScalarSSE && !Subtarget->hasSSE3())
      // Expand FP_TO_UINT into a select.
      // FIXME: We would like to use a Custom expander here eventually to do
      // the optimal thing for SSE vs. the default expansion in the legalizer.
      setOperationAction(ISD::FP_TO_UINT   , MVT::i32  , Expand);
    else
      // With SSE3 we can use fisttpll to convert to a signed i64.
      setOperationAction(ISD::FP_TO_UINT   , MVT::i32  , Promote);
  }

  setOperationAction(ISD::BIT_CONVERT      , MVT::f32  , Expand);
  setOperationAction(ISD::BIT_CONVERT      , MVT::i32  , Expand);

  setOperationAction(ISD::BR_JT            , MVT::Other, Expand);
  setOperationAction(ISD::BRCOND           , MVT::Other, Custom);
  setOperationAction(ISD::BR_CC            , MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC        , MVT::Other, Expand);
  setOperationAction(ISD::MEMMOVE          , MVT::Other, Expand);
  if (Subtarget->is64Bit())
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16  , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8   , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1   , Expand);
  setOperationAction(ISD::FP_ROUND_INREG   , MVT::f32  , Expand);
  setOperationAction(ISD::FREM             , MVT::f64  , Expand);

  setOperationAction(ISD::CTPOP            , MVT::i8   , Expand);
  setOperationAction(ISD::CTTZ             , MVT::i8   , Expand);
  setOperationAction(ISD::CTLZ             , MVT::i8   , Expand);
  setOperationAction(ISD::CTPOP            , MVT::i16  , Expand);
  setOperationAction(ISD::CTTZ             , MVT::i16  , Expand);
  setOperationAction(ISD::CTLZ             , MVT::i16  , Expand);
  setOperationAction(ISD::CTPOP            , MVT::i32  , Expand);
  setOperationAction(ISD::CTTZ             , MVT::i32  , Expand);
  setOperationAction(ISD::CTLZ             , MVT::i32  , Expand);
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::CTPOP          , MVT::i64  , Expand);
    setOperationAction(ISD::CTTZ           , MVT::i64  , Expand);
    setOperationAction(ISD::CTLZ           , MVT::i64  , Expand);
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
  setOperationAction(ISD::SETCC           , MVT::i8   , Custom);
  setOperationAction(ISD::SETCC           , MVT::i16  , Custom);
  setOperationAction(ISD::SETCC           , MVT::i32  , Custom);
  setOperationAction(ISD::SETCC           , MVT::f32  , Custom);
  setOperationAction(ISD::SETCC           , MVT::f64  , Custom);
  if (Subtarget->is64Bit()) {
    setOperationAction(ISD::SELECT        , MVT::i64  , Custom);
    setOperationAction(ISD::SETCC         , MVT::i64  , Custom);
  }
  // X86 ret instruction may pop stack.
  setOperationAction(ISD::RET             , MVT::Other, Custom);
  // Darwin ABI issue.
  setOperationAction(ISD::ConstantPool    , MVT::i32  , Custom);
  setOperationAction(ISD::JumpTable       , MVT::i32  , Custom);
  setOperationAction(ISD::GlobalAddress   , MVT::i32  , Custom);
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
  // X86 wants to expand memset / memcpy itself.
  setOperationAction(ISD::MEMSET          , MVT::Other, Custom);
  setOperationAction(ISD::MEMCPY          , MVT::Other, Custom);

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  // FIXME - use subtarget debug flags
  if (!Subtarget->isTargetDarwin() &&
      !Subtarget->isTargetELF() &&
      !Subtarget->isTargetCygwin())
    setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  
  // Use the default implementation.
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE,          MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE,       MVT::Other, Expand);
  if (Subtarget->is64Bit())
    setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Expand);

  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

  if (X86ScalarSSE) {
    // Set up the FP register classes.
    addRegisterClass(MVT::f32, X86::FR32RegisterClass);
    addRegisterClass(MVT::f64, X86::FR64RegisterClass);

    // Use ANDPD to simulate FABS.
    setOperationAction(ISD::FABS , MVT::f64, Custom);
    setOperationAction(ISD::FABS , MVT::f32, Custom);

    // Use XORP to simulate FNEG.
    setOperationAction(ISD::FNEG , MVT::f64, Custom);
    setOperationAction(ISD::FNEG , MVT::f32, Custom);

    // We don't support sin/cos/fmod
    setOperationAction(ISD::FSIN , MVT::f64, Expand);
    setOperationAction(ISD::FCOS , MVT::f64, Expand);
    setOperationAction(ISD::FREM , MVT::f64, Expand);
    setOperationAction(ISD::FSIN , MVT::f32, Expand);
    setOperationAction(ISD::FCOS , MVT::f32, Expand);
    setOperationAction(ISD::FREM , MVT::f32, Expand);

    // Expand FP immediates into loads from the stack, except for the special
    // cases we handle.
    setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
    setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
    addLegalFPImmediate(+0.0); // xorps / xorpd
  } else {
    // Set up the FP register classes.
    addRegisterClass(MVT::f64, X86::RFPRegisterClass);
    
    setOperationAction(ISD::UNDEF, MVT::f64, Expand);
    
    if (!UnsafeFPMath) {
      setOperationAction(ISD::FSIN           , MVT::f64  , Expand);
      setOperationAction(ISD::FCOS           , MVT::f64  , Expand);
    }

    setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
    addLegalFPImmediate(+0.0); // FLD0
    addLegalFPImmediate(+1.0); // FLD1
    addLegalFPImmediate(-0.0); // FLD0/FCHS
    addLegalFPImmediate(-1.0); // FLD1/FCHS
  }

  // First set operation action for all vector types to expand. Then we
  // will selectively turn on ones that can be effectively codegen'd.
  for (unsigned VT = (unsigned)MVT::Vector + 1;
       VT != (unsigned)MVT::LAST_VALUETYPE; VT++) {
    setOperationAction(ISD::ADD , (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::SUB , (MVT::ValueType)VT, Expand);
    setOperationAction(ISD::FADD, (MVT::ValueType)VT, Expand);
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
  }

  if (Subtarget->hasMMX()) {
    addRegisterClass(MVT::v8i8,  X86::VR64RegisterClass);
    addRegisterClass(MVT::v4i16, X86::VR64RegisterClass);
    addRegisterClass(MVT::v2i32, X86::VR64RegisterClass);

    // FIXME: add MMX packed arithmetics
    setOperationAction(ISD::BUILD_VECTOR,     MVT::v8i8,  Expand);
    setOperationAction(ISD::BUILD_VECTOR,     MVT::v4i16, Expand);
    setOperationAction(ISD::BUILD_VECTOR,     MVT::v2i32, Expand);
  }

  if (Subtarget->hasSSE1()) {
    addRegisterClass(MVT::v4f32, X86::VR128RegisterClass);

    setOperationAction(ISD::FADD,               MVT::v4f32, Legal);
    setOperationAction(ISD::FSUB,               MVT::v4f32, Legal);
    setOperationAction(ISD::FMUL,               MVT::v4f32, Legal);
    setOperationAction(ISD::FDIV,               MVT::v4f32, Legal);
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
    setOperationAction(ISD::SUB,                MVT::v16i8, Legal);
    setOperationAction(ISD::SUB,                MVT::v8i16, Legal);
    setOperationAction(ISD::SUB,                MVT::v4i32, Legal);
    setOperationAction(ISD::MUL,                MVT::v8i16, Legal);
    setOperationAction(ISD::FADD,               MVT::v2f64, Legal);
    setOperationAction(ISD::FSUB,               MVT::v2f64, Legal);
    setOperationAction(ISD::FMUL,               MVT::v2f64, Legal);
    setOperationAction(ISD::FDIV,               MVT::v2f64, Legal);

    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v16i8, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR,   MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v8i16, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4i32, Custom);
    // Implement v4f32 insert_vector_elt in terms of SSE2 v8i16 ones.
    setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v4f32, Custom);

    // Custom lower build_vector, vector_shuffle, and extract_vector_elt.
    for (unsigned VT = (unsigned)MVT::v16i8; VT != (unsigned)MVT::v2i64; VT++) {
      setOperationAction(ISD::BUILD_VECTOR,        (MVT::ValueType)VT, Custom);
      setOperationAction(ISD::VECTOR_SHUFFLE,      (MVT::ValueType)VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT,  (MVT::ValueType)VT, Custom);
    }
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v2f64, Custom);
    setOperationAction(ISD::BUILD_VECTOR,       MVT::v2i64, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v2f64, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v2i64, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f64, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i64, Custom);

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

    // Custom lower v2i64 and v2f64 selects.
    setOperationAction(ISD::LOAD,               MVT::v2f64, Legal);
    setOperationAction(ISD::LOAD,               MVT::v2i64, Legal);
    setOperationAction(ISD::SELECT,             MVT::v2f64, Custom);
    setOperationAction(ISD::SELECT,             MVT::v2i64, Custom);
  }

  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::VECTOR_SHUFFLE);
  setTargetDAGCombine(ISD::SELECT);

  computeRegisterProperties();

  // FIXME: These should be based on subtarget info. Plus, the values should
  // be smaller when we are in optimizing for size mode.
  maxStoresPerMemset = 16; // For %llvm.memset -> sequence of stores
  maxStoresPerMemcpy = 16; // For %llvm.memcpy -> sequence of stores
  maxStoresPerMemmove = 16; // For %llvm.memmove -> sequence of stores
  allowUnalignedMemoryAccesses = true; // x86 supports it!
}

//===----------------------------------------------------------------------===//
//                    C Calling Convention implementation
//===----------------------------------------------------------------------===//

/// AddLiveIn - This helper function adds the specified physical register to the
/// MachineFunction as a live in value.  It also creates a corresponding virtual
/// register for it.
static unsigned AddLiveIn(MachineFunction &MF, unsigned PReg,
                          TargetRegisterClass *RC) {
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getSSARegMap()->createVirtualRegister(RC);
  MF.addLiveIn(PReg, VReg);
  return VReg;
}

/// HowToPassCCCArgument - Returns how an formal argument of the specified type
/// should be passed. If it is through stack, returns the size of the stack
/// slot; if it is through XMM register, returns the number of XMM registers
/// are needed.
static void
HowToPassCCCArgument(MVT::ValueType ObjectVT, unsigned NumXMMRegs,
                     unsigned &ObjSize, unsigned &ObjXMMRegs) {
  ObjXMMRegs = 0;

  switch (ObjectVT) {
  default: assert(0 && "Unhandled argument type!");
  case MVT::i8:  ObjSize = 1; break;
  case MVT::i16: ObjSize = 2; break;
  case MVT::i32: ObjSize = 4; break;
  case MVT::i64: ObjSize = 8; break;
  case MVT::f32: ObjSize = 4; break;
  case MVT::f64: ObjSize = 8; break;
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v2f64:
    if (NumXMMRegs < 4)
      ObjXMMRegs = 1;
    else
      ObjSize = 16;
    break;
  }
}

SDOperand X86TargetLowering::LowerCCCArguments(SDOperand Op, SelectionDAG &DAG) {
  unsigned NumArgs = Op.Val->getNumValues() - 1;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SDOperand Root = Op.getOperand(0);
  std::vector<SDOperand> ArgValues;

  // Add DAG nodes to load the arguments...  On entry to a function on the X86,
  // the stack frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first argument (leftmost lexically)
  // [ESP + 8] -- second argument, if first argument is <= 4 bytes in size
  //    ...
  //
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  unsigned NumXMMRegs = 0;  // XMM regs used for parameter passing.
  static const unsigned XMMArgRegs[] = {
    X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3
  };
  for (unsigned i = 0; i < NumArgs; ++i) {
    MVT::ValueType ObjectVT = Op.getValue(i).getValueType();
    unsigned ArgIncrement = 4;
    unsigned ObjSize = 0;
    unsigned ObjXMMRegs = 0;
    HowToPassCCCArgument(ObjectVT, NumXMMRegs, ObjSize, ObjXMMRegs);
    if (ObjSize > 4)
      ArgIncrement = ObjSize;

    SDOperand ArgValue;
    if (ObjXMMRegs) {
      // Passed in a XMM register.
      unsigned Reg = AddLiveIn(MF, XMMArgRegs[NumXMMRegs],
                               X86::VR128RegisterClass);
      ArgValue= DAG.getCopyFromReg(Root, Reg, ObjectVT);
      ArgValues.push_back(ArgValue);
      NumXMMRegs += ObjXMMRegs;
    } else {
      // XMM arguments have to be aligned on 16-byte boundary.
      if (ObjSize == 16)
        ArgOffset = ((ArgOffset + 15) / 16) * 16;
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, getPointerTy());
      ArgValue = DAG.getLoad(Op.Val->getValueType(i), Root, FIN, NULL, 0);
      ArgValues.push_back(ArgValue);
      ArgOffset += ArgIncrement;   // Move on to the next argument...
    }
  }

  ArgValues.push_back(Root);

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  if (isVarArg)
    VarArgsFrameIndex = MFI->CreateFixedObject(1, ArgOffset);
  RegSaveFrameIndex = 0xAAAAAAA;  // X86-64 only.
  ReturnAddrIndex = 0;            // No return address slot generated yet.
  BytesToPopOnReturn = 0;         // Callee pops nothing.
  BytesCallerReserves = ArgOffset;

  // If this is a struct return on, the callee pops the hidden struct
  // pointer. This is common for Darwin/X86, Linux & Mingw32 targets.
  if (MF.getFunction()->getCallingConv() == CallingConv::CSRet)
    BytesToPopOnReturn = 4;

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVTs(Op.Val->value_begin(),
                                     Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVTs, &ArgValues[0],ArgValues.size());
}


SDOperand X86TargetLowering::LowerCCCCallTo(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain     = Op.getOperand(0);
  unsigned CallingConv= cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  bool isTailCall     = cast<ConstantSDNode>(Op.getOperand(3))->getValue() != 0;
  SDOperand Callee    = Op.getOperand(4);
  MVT::ValueType RetVT= Op.Val->getValueType(0);
  unsigned NumOps     = (Op.getNumOperands() - 5) / 2;

  // Keep track of the number of XMM regs passed so far.
  unsigned NumXMMRegs = 0;
  static const unsigned XMMArgRegs[] = {
    X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3
  };

  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;
  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);

    switch (Arg.getValueType()) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::f32:
      NumBytes += 4;
      break;
    case MVT::i64:
    case MVT::f64:
      NumBytes += 8;
      break;
    case MVT::v16i8:
    case MVT::v8i16:
    case MVT::v4i32:
    case MVT::v2i64:
    case MVT::v4f32:
    case MVT::v2f64:
      if (NumXMMRegs < 4)
        ++NumXMMRegs;
      else {
        // XMM arguments have to be aligned on 16-byte boundary.
        NumBytes = ((NumBytes + 15) / 16) * 16;
        NumBytes += 16;
      }
      break;
    }
  }

  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(NumBytes, getPointerTy()));

  // Arguments go on the stack in reverse order, as specified by the ABI.
  unsigned ArgOffset = 0;
  NumXMMRegs = 0;
  std::vector<std::pair<unsigned, SDOperand> > RegsToPass;
  std::vector<SDOperand> MemOpChains;
  SDOperand StackPtr = DAG.getRegister(X86StackPtr, getPointerTy());
  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);

    switch (Arg.getValueType()) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i8:
    case MVT::i16: {
      // Promote the integer to 32 bits.  If the input type is signed use a
      // sign extend, otherwise use a zero extend.
      unsigned ExtOp =
        dyn_cast<ConstantSDNode>(Op.getOperand(5+2*i+1))->getValue() ?
        ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
      Arg = DAG.getNode(ExtOp, MVT::i32, Arg);
    }
    // Fallthrough

    case MVT::i32:
    case MVT::f32: {
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
      ArgOffset += 4;
      break;
    }
    case MVT::i64:
    case MVT::f64: {
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
      ArgOffset += 8;
      break;
    }
    case MVT::v16i8:
    case MVT::v8i16:
    case MVT::v4i32:
    case MVT::v2i64:
    case MVT::v4f32:
    case MVT::v2f64:
      if (NumXMMRegs < 4) {
        RegsToPass.push_back(std::make_pair(XMMArgRegs[NumXMMRegs], Arg));
        NumXMMRegs++;
      } else {
        // XMM arguments have to be aligned on 16-byte boundary.
        ArgOffset = ((ArgOffset + 15) / 16) * 16;
        SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
        PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
        MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
        ArgOffset += 16;
      }
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into registers.
  SDOperand InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                             InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first, 
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.Val)
    Ops.push_back(InFlag);

  Chain = DAG.getNode(isTailCall ? X86ISD::TAILCALL : X86ISD::CALL,
                      NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  unsigned NumBytesForCalleeToPush = 0;

  // If this is is a call to a struct-return function, the callee
  // pops the hidden struct pointer, so we have to push it back.
  // This is common for Darwin/X86, Linux & Mingw32 targets.
  if (CallingConv == CallingConv::CSRet)
    NumBytesForCalleeToPush = 4;
  
  NodeTys.clear();
  NodeTys.push_back(MVT::Other);   // Returns a chain
  if (RetVT != MVT::Other)
    NodeTys.push_back(MVT::Flag);  // Returns a flag for retval copy to use.
  Ops.clear();
  Ops.push_back(Chain);
  Ops.push_back(DAG.getConstant(NumBytes, getPointerTy()));
  Ops.push_back(DAG.getConstant(NumBytesForCalleeToPush, getPointerTy()));
  Ops.push_back(InFlag);
  Chain = DAG.getNode(ISD::CALLSEQ_END, NodeTys, &Ops[0], Ops.size());
  if (RetVT != MVT::Other)
    InFlag = Chain.getValue(1);
  
  std::vector<SDOperand> ResultVals;
  NodeTys.clear();
  switch (RetVT) {
  default: assert(0 && "Unknown value type to return!");
  case MVT::Other: break;
  case MVT::i8:
    Chain = DAG.getCopyFromReg(Chain, X86::AL, MVT::i8, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i8);
    break;
  case MVT::i16:
    Chain = DAG.getCopyFromReg(Chain, X86::AX, MVT::i16, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i16);
    break;
  case MVT::i32:
    if (Op.Val->getValueType(1) == MVT::i32) {
      Chain = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      Chain = DAG.getCopyFromReg(Chain, X86::EDX, MVT::i32,
                                 Chain.getValue(2)).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      NodeTys.push_back(MVT::i32);
    } else {
      Chain = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
    }
    NodeTys.push_back(MVT::i32);
    break;
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v2f64:
    Chain = DAG.getCopyFromReg(Chain, X86::XMM0, RetVT, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(RetVT);
    break;
  case MVT::f32:
  case MVT::f64: {
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::f64);
    Tys.push_back(MVT::Other);
    Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(InFlag);
    SDOperand RetVal = DAG.getNode(X86ISD::FP_GET_RESULT, Tys, 
                                   &Ops[0], Ops.size());
    Chain  = RetVal.getValue(1);
    InFlag = RetVal.getValue(2);
    if (X86ScalarSSE) {
      // FIXME: Currently the FST is flagged to the FP_GET_RESULT. This
      // shouldn't be necessary except that RFP cannot be live across
      // multiple blocks. When stackifier is fixed, they can be uncoupled.
      MachineFunction &MF = DAG.getMachineFunction();
      int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
      SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
      Tys.clear();
      Tys.push_back(MVT::Other);
      Ops.clear();
      Ops.push_back(Chain);
      Ops.push_back(RetVal);
      Ops.push_back(StackSlot);
      Ops.push_back(DAG.getValueType(RetVT));
      Ops.push_back(InFlag);
      Chain = DAG.getNode(X86ISD::FST, Tys, &Ops[0], Ops.size());
      RetVal = DAG.getLoad(RetVT, Chain, StackSlot, NULL, 0);
      Chain = RetVal.getValue(1);
    }

    if (RetVT == MVT::f32 && !X86ScalarSSE)
      // FIXME: we would really like to remember that this FP_ROUND
      // operation is okay to eliminate if we allow excess FP precision.
      RetVal = DAG.getNode(ISD::FP_ROUND, MVT::f32, RetVal);
    ResultVals.push_back(RetVal);
    NodeTys.push_back(RetVT);
    break;
  }
  }

  // If the function returns void, just return the chain.
  if (ResultVals.empty())
    return Chain;
  
  // Otherwise, merge everything together with a MERGE_VALUES node.
  NodeTys.push_back(MVT::Other);
  ResultVals.push_back(Chain);
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys,
                              &ResultVals[0], ResultVals.size());
  return Res.getValue(Op.ResNo);
}


//===----------------------------------------------------------------------===//
//                 X86-64 C Calling Convention implementation
//===----------------------------------------------------------------------===//

/// HowToPassX86_64CCCArgument - Returns how an formal argument of the specified
/// type should be passed. If it is through stack, returns the size of the stack
/// slot; if it is through integer or XMM register, returns the number of
/// integer or XMM registers are needed.
static void
HowToPassX86_64CCCArgument(MVT::ValueType ObjectVT,
                           unsigned NumIntRegs, unsigned NumXMMRegs,
                           unsigned &ObjSize, unsigned &ObjIntRegs,
                           unsigned &ObjXMMRegs) {
  ObjSize = 0;
  ObjIntRegs = 0;
  ObjXMMRegs = 0;

  switch (ObjectVT) {
  default: assert(0 && "Unhandled argument type!");
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
    if (NumIntRegs < 6)
      ObjIntRegs = 1;
    else {
      switch (ObjectVT) {
      default: break;
      case MVT::i8:  ObjSize = 1; break;
      case MVT::i16: ObjSize = 2; break;
      case MVT::i32: ObjSize = 4; break;
      case MVT::i64: ObjSize = 8; break;
      }
    }
    break;
  case MVT::f32:
  case MVT::f64:
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v2f64:
    if (NumXMMRegs < 8)
      ObjXMMRegs = 1;
    else {
      switch (ObjectVT) {
      default: break;
      case MVT::f32:  ObjSize = 4; break;
      case MVT::f64:  ObjSize = 8; break;
      case MVT::v16i8:
      case MVT::v8i16:
      case MVT::v4i32:
      case MVT::v2i64:
      case MVT::v4f32:
      case MVT::v2f64: ObjSize = 16; break;
    }
    break;
  }
  }
}

SDOperand
X86TargetLowering::LowerX86_64CCCArguments(SDOperand Op, SelectionDAG &DAG) {
  unsigned NumArgs = Op.Val->getNumValues() - 1;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SDOperand Root = Op.getOperand(0);
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  std::vector<SDOperand> ArgValues;

  // Add DAG nodes to load the arguments...  On entry to a function on the X86,
  // the stack frame looks like this:
  //
  // [RSP] -- return address
  // [RSP + 8] -- first nonreg argument (leftmost lexically)
  // [RSP +16] -- second nonreg argument, if 1st argument is <= 8 bytes in size
  //    ...
  //
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  unsigned NumIntRegs = 0;  // Int regs used for parameter passing.
  unsigned NumXMMRegs = 0;  // XMM regs used for parameter passing.

  static const unsigned GPR8ArgRegs[] = {
    X86::DIL, X86::SIL, X86::DL,  X86::CL,  X86::R8B, X86::R9B
  };
  static const unsigned GPR16ArgRegs[] = {
    X86::DI,  X86::SI,  X86::DX,  X86::CX,  X86::R8W, X86::R9W
  };
  static const unsigned GPR32ArgRegs[] = {
    X86::EDI, X86::ESI, X86::EDX, X86::ECX, X86::R8D, X86::R9D
  };
  static const unsigned GPR64ArgRegs[] = {
    X86::RDI, X86::RSI, X86::RDX, X86::RCX, X86::R8,  X86::R9
  };
  static const unsigned XMMArgRegs[] = {
    X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
    X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
  };

  for (unsigned i = 0; i < NumArgs; ++i) {
    MVT::ValueType ObjectVT = Op.getValue(i).getValueType();
    unsigned ArgIncrement = 8;
    unsigned ObjSize = 0;
    unsigned ObjIntRegs = 0;
    unsigned ObjXMMRegs = 0;

    // FIXME: __int128 and long double support?
    HowToPassX86_64CCCArgument(ObjectVT, NumIntRegs, NumXMMRegs,
                               ObjSize, ObjIntRegs, ObjXMMRegs);
    if (ObjSize > 8)
      ArgIncrement = ObjSize;

    unsigned Reg = 0;
    SDOperand ArgValue;
    if (ObjIntRegs || ObjXMMRegs) {
      switch (ObjectVT) {
      default: assert(0 && "Unhandled argument type!");
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      case MVT::i64: {
        TargetRegisterClass *RC = NULL;
        switch (ObjectVT) {
        default: break;
        case MVT::i8: 
          RC = X86::GR8RegisterClass;
          Reg = GPR8ArgRegs[NumIntRegs];
          break;
        case MVT::i16:
          RC = X86::GR16RegisterClass;
          Reg = GPR16ArgRegs[NumIntRegs];
          break;
        case MVT::i32:
          RC = X86::GR32RegisterClass;
          Reg = GPR32ArgRegs[NumIntRegs];
          break;
        case MVT::i64:
          RC = X86::GR64RegisterClass;
          Reg = GPR64ArgRegs[NumIntRegs];
          break;
        }
        Reg = AddLiveIn(MF, Reg, RC);
        ArgValue = DAG.getCopyFromReg(Root, Reg, ObjectVT);
        break;
      }
      case MVT::f32:
      case MVT::f64:
      case MVT::v16i8:
      case MVT::v8i16:
      case MVT::v4i32:
      case MVT::v2i64:
      case MVT::v4f32:
      case MVT::v2f64: {
        TargetRegisterClass *RC= (ObjectVT == MVT::f32) ?
          X86::FR32RegisterClass : ((ObjectVT == MVT::f64) ?
                              X86::FR64RegisterClass : X86::VR128RegisterClass);
        Reg = AddLiveIn(MF, XMMArgRegs[NumXMMRegs], RC);
        ArgValue = DAG.getCopyFromReg(Root, Reg, ObjectVT);
        break;
      }
      }
      NumIntRegs += ObjIntRegs;
      NumXMMRegs += ObjXMMRegs;
    } else if (ObjSize) {
      // XMM arguments have to be aligned on 16-byte boundary.
      if (ObjSize == 16)
        ArgOffset = ((ArgOffset + 15) / 16) * 16;
      // Create the SelectionDAG nodes corresponding to a load from this
      // parameter.
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, getPointerTy());
      ArgValue = DAG.getLoad(Op.Val->getValueType(i), Root, FIN, NULL, 0);
      ArgOffset += ArgIncrement;   // Move on to the next argument.
    }

    ArgValues.push_back(ArgValue);
  }

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (isVarArg) {
    // For X86-64, if there are vararg parameters that are passed via
    // registers, then we must store them to their spots on the stack so they
    // may be loaded by deferencing the result of va_next.
    VarArgsGPOffset = NumIntRegs * 8;
    VarArgsFPOffset = 6 * 8 + NumXMMRegs * 16;
    VarArgsFrameIndex = MFI->CreateFixedObject(1, ArgOffset);
    RegSaveFrameIndex = MFI->CreateStackObject(6 * 8 + 8 * 16, 16);

    // Store the integer parameter registers.
    std::vector<SDOperand> MemOps;
    SDOperand RSFIN = DAG.getFrameIndex(RegSaveFrameIndex, getPointerTy());
    SDOperand FIN = DAG.getNode(ISD::ADD, getPointerTy(), RSFIN,
                              DAG.getConstant(VarArgsGPOffset, getPointerTy()));
    for (; NumIntRegs != 6; ++NumIntRegs) {
      unsigned VReg = AddLiveIn(MF, GPR64ArgRegs[NumIntRegs],
                                X86::GR64RegisterClass);
      SDOperand Val = DAG.getCopyFromReg(Root, VReg, MVT::i64);
      SDOperand Store = DAG.getStore(Val.getValue(1), Val, FIN, NULL, 0);
      MemOps.push_back(Store);
      FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                        DAG.getConstant(8, getPointerTy()));
    }

    // Now store the XMM (fp + vector) parameter registers.
    FIN = DAG.getNode(ISD::ADD, getPointerTy(), RSFIN,
                      DAG.getConstant(VarArgsFPOffset, getPointerTy()));
    for (; NumXMMRegs != 8; ++NumXMMRegs) {
      unsigned VReg = AddLiveIn(MF, XMMArgRegs[NumXMMRegs],
                                X86::VR128RegisterClass);
      SDOperand Val = DAG.getCopyFromReg(Root, VReg, MVT::v4f32);
      SDOperand Store = DAG.getStore(Val.getValue(1), Val, FIN, NULL, 0);
      MemOps.push_back(Store);
      FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                        DAG.getConstant(16, getPointerTy()));
    }
    if (!MemOps.empty())
        Root = DAG.getNode(ISD::TokenFactor, MVT::Other,
                           &MemOps[0], MemOps.size());
  }

  ArgValues.push_back(Root);

  ReturnAddrIndex = 0;     // No return address slot generated yet.
  BytesToPopOnReturn = 0;  // Callee pops nothing.
  BytesCallerReserves = ArgOffset;

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVTs(Op.Val->value_begin(),
                                     Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVTs, &ArgValues[0],ArgValues.size());
}

SDOperand
X86TargetLowering::LowerX86_64CCCCallTo(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain     = Op.getOperand(0);
  bool isVarArg       = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  bool isTailCall     = cast<ConstantSDNode>(Op.getOperand(3))->getValue() != 0;
  SDOperand Callee    = Op.getOperand(4);
  MVT::ValueType RetVT= Op.Val->getValueType(0);
  unsigned NumOps     = (Op.getNumOperands() - 5) / 2;

  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;
  unsigned NumIntRegs = 0;  // Int regs used for parameter passing.
  unsigned NumXMMRegs = 0;  // XMM regs used for parameter passing.

  static const unsigned GPR8ArgRegs[] = {
    X86::DIL, X86::SIL, X86::DL,  X86::CL,  X86::R8B, X86::R9B
  };
  static const unsigned GPR16ArgRegs[] = {
    X86::DI,  X86::SI,  X86::DX,  X86::CX,  X86::R8W, X86::R9W
  };
  static const unsigned GPR32ArgRegs[] = {
    X86::EDI, X86::ESI, X86::EDX, X86::ECX, X86::R8D, X86::R9D
  };
  static const unsigned GPR64ArgRegs[] = {
    X86::RDI, X86::RSI, X86::RDX, X86::RCX, X86::R8,  X86::R9
  };
  static const unsigned XMMArgRegs[] = {
    X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
    X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
  };

  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);
    MVT::ValueType ArgVT = Arg.getValueType();

    switch (ArgVT) {
    default: assert(0 && "Unknown value type!");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
      if (NumIntRegs < 6)
        ++NumIntRegs;
      else
        NumBytes += 8;
      break;
    case MVT::f32:
    case MVT::f64:
    case MVT::v16i8:
    case MVT::v8i16:
    case MVT::v4i32:
    case MVT::v2i64:
    case MVT::v4f32:
    case MVT::v2f64:
      if (NumXMMRegs < 8)
        NumXMMRegs++;
      else if (ArgVT == MVT::f32 || ArgVT == MVT::f64)
        NumBytes += 8;
      else {
        // XMM arguments have to be aligned on 16-byte boundary.
        NumBytes = ((NumBytes + 15) / 16) * 16;
        NumBytes += 16;
      }
      break;
    }
  }

  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(NumBytes, getPointerTy()));

  // Arguments go on the stack in reverse order, as specified by the ABI.
  unsigned ArgOffset = 0;
  NumIntRegs = 0;
  NumXMMRegs = 0;
  std::vector<std::pair<unsigned, SDOperand> > RegsToPass;
  std::vector<SDOperand> MemOpChains;
  SDOperand StackPtr = DAG.getRegister(X86StackPtr, getPointerTy());
  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);
    MVT::ValueType ArgVT = Arg.getValueType();

    switch (ArgVT) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
      if (NumIntRegs < 6) {
        unsigned Reg = 0;
        switch (ArgVT) {
        default: break;
        case MVT::i8:  Reg = GPR8ArgRegs[NumIntRegs];  break;
        case MVT::i16: Reg = GPR16ArgRegs[NumIntRegs]; break;
        case MVT::i32: Reg = GPR32ArgRegs[NumIntRegs]; break;
        case MVT::i64: Reg = GPR64ArgRegs[NumIntRegs]; break;
        }
        RegsToPass.push_back(std::make_pair(Reg, Arg));
        ++NumIntRegs;
      } else {
        SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
        PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
        MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
        ArgOffset += 8;
      }
      break;
    case MVT::f32:
    case MVT::f64:
    case MVT::v16i8:
    case MVT::v8i16:
    case MVT::v4i32:
    case MVT::v2i64:
    case MVT::v4f32:
    case MVT::v2f64:
      if (NumXMMRegs < 8) {
        RegsToPass.push_back(std::make_pair(XMMArgRegs[NumXMMRegs], Arg));
        NumXMMRegs++;
      } else {
        if (ArgVT != MVT::f32 && ArgVT != MVT::f64) {
          // XMM arguments have to be aligned on 16-byte boundary.
          ArgOffset = ((ArgOffset + 15) / 16) * 16;
        }
        SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
        PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
        MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
        if (ArgVT == MVT::f32 || ArgVT == MVT::f64)
          ArgOffset += 8;
        else
          ArgOffset += 16;
      }
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into registers.
  SDOperand InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                             InFlag);
    InFlag = Chain.getValue(1);
  }

  if (isVarArg) {
    // From AMD64 ABI document:
    // For calls that may call functions that use varargs or stdargs
    // (prototype-less calls or calls to functions containing ellipsis (...) in
    // the declaration) %al is used as hidden argument to specify the number
    // of SSE registers used. The contents of %al do not need to match exactly
    // the number of registers, but must be an ubound on the number of SSE
    // registers used and is in the range 0 - 8 inclusive.
    Chain = DAG.getCopyToReg(Chain, X86::AL,
                             DAG.getConstant(NumXMMRegs, MVT::i8), InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first, 
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.Val)
    Ops.push_back(InFlag);

  // FIXME: Do not generate X86ISD::TAILCALL for now.
  Chain = DAG.getNode(isTailCall ? X86ISD::TAILCALL : X86ISD::CALL,
                      NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  NodeTys.clear();
  NodeTys.push_back(MVT::Other);   // Returns a chain
  if (RetVT != MVT::Other)
    NodeTys.push_back(MVT::Flag);  // Returns a flag for retval copy to use.
  Ops.clear();
  Ops.push_back(Chain);
  Ops.push_back(DAG.getConstant(NumBytes, getPointerTy()));
  Ops.push_back(DAG.getConstant(0, getPointerTy()));
  Ops.push_back(InFlag);
  Chain = DAG.getNode(ISD::CALLSEQ_END, NodeTys, &Ops[0], Ops.size());
  if (RetVT != MVT::Other)
    InFlag = Chain.getValue(1);
  
  std::vector<SDOperand> ResultVals;
  NodeTys.clear();
  switch (RetVT) {
  default: assert(0 && "Unknown value type to return!");
  case MVT::Other: break;
  case MVT::i8:
    Chain = DAG.getCopyFromReg(Chain, X86::AL, MVT::i8, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i8);
    break;
  case MVT::i16:
    Chain = DAG.getCopyFromReg(Chain, X86::AX, MVT::i16, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i16);
    break;
  case MVT::i32:
    Chain = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i32);
    break;
  case MVT::i64:
    if (Op.Val->getValueType(1) == MVT::i64) {
      // FIXME: __int128 support?
      Chain = DAG.getCopyFromReg(Chain, X86::RAX, MVT::i64, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      Chain = DAG.getCopyFromReg(Chain, X86::RDX, MVT::i64,
                                 Chain.getValue(2)).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      NodeTys.push_back(MVT::i64);
    } else {
      Chain = DAG.getCopyFromReg(Chain, X86::RAX, MVT::i64, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
    }
    NodeTys.push_back(MVT::i64);
    break;
  case MVT::f32:
  case MVT::f64:
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v2f64:
    // FIXME: long double support?
    Chain = DAG.getCopyFromReg(Chain, X86::XMM0, RetVT, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(RetVT);
    break;
  }

  // If the function returns void, just return the chain.
  if (ResultVals.empty())
    return Chain;
  
  // Otherwise, merge everything together with a MERGE_VALUES node.
  NodeTys.push_back(MVT::Other);
  ResultVals.push_back(Chain);
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys,
                              &ResultVals[0], ResultVals.size());
  return Res.getValue(Op.ResNo);
}

//===----------------------------------------------------------------------===//
//                    Fast Calling Convention implementation
//===----------------------------------------------------------------------===//
//
// The X86 'fast' calling convention passes up to two integer arguments in
// registers (an appropriate portion of EAX/EDX), passes arguments in C order,
// and requires that the callee pop its arguments off the stack (allowing proper
// tail calls), and has the same return value conventions as C calling convs.
//
// This calling convention always arranges for the callee pop value to be 8n+4
// bytes, which is needed for tail recursion elimination and stack alignment
// reasons.
//
// Note that this can be enhanced in the future to pass fp vals in registers
// (when we have a global fp allocator) and do other tricks.
//

/// HowToPassFastCCArgument - Returns how an formal argument of the specified
/// type should be passed. If it is through stack, returns the size of the stack
/// slot; if it is through integer or XMM register, returns the number of
/// integer or XMM registers are needed.
static void
HowToPassFastCCArgument(MVT::ValueType ObjectVT,
                        unsigned NumIntRegs, unsigned NumXMMRegs,
                        unsigned &ObjSize, unsigned &ObjIntRegs,
                        unsigned &ObjXMMRegs) {
  ObjSize = 0;
  ObjIntRegs = 0;
  ObjXMMRegs = 0;

  switch (ObjectVT) {
  default: assert(0 && "Unhandled argument type!");
  case MVT::i8:
#if FASTCC_NUM_INT_ARGS_INREGS > 0
    if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS)
      ObjIntRegs = 1;
    else
#endif
      ObjSize = 1;
    break;
  case MVT::i16:
#if FASTCC_NUM_INT_ARGS_INREGS > 0
    if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS)
      ObjIntRegs = 1;
    else
#endif
      ObjSize = 2;
    break;
  case MVT::i32:
#if FASTCC_NUM_INT_ARGS_INREGS > 0
    if (NumIntRegs < FASTCC_NUM_INT_ARGS_INREGS)
      ObjIntRegs = 1;
    else
#endif
      ObjSize = 4;
    break;
  case MVT::i64:
#if FASTCC_NUM_INT_ARGS_INREGS > 0
    if (NumIntRegs+2 <= FASTCC_NUM_INT_ARGS_INREGS) {
      ObjIntRegs = 2;
    } else if (NumIntRegs+1 <= FASTCC_NUM_INT_ARGS_INREGS) {
      ObjIntRegs = 1;
      ObjSize = 4;
    } else
#endif
      ObjSize = 8;
  case MVT::f32:
    ObjSize = 4;
    break;
  case MVT::f64:
    ObjSize = 8;
    break;
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v2f64:
    if (NumXMMRegs < 4)
      ObjXMMRegs = 1;
    else
      ObjSize = 16;
    break;
  }
}

SDOperand
X86TargetLowering::LowerFastCCArguments(SDOperand Op, SelectionDAG &DAG) {
  unsigned NumArgs = Op.Val->getNumValues()-1;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SDOperand Root = Op.getOperand(0);
  std::vector<SDOperand> ArgValues;

  // Add DAG nodes to load the arguments...  On entry to a function the stack
  // frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first nonreg argument (leftmost lexically)
  // [ESP + 8] -- second nonreg argument, if 1st argument is <= 4 bytes in size
  //    ...
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot

  // Keep track of the number of integer regs passed so far.  This can be either
  // 0 (neither EAX or EDX used), 1 (EAX is used) or 2 (EAX and EDX are both
  // used).
  unsigned NumIntRegs = 0;
  unsigned NumXMMRegs = 0;  // XMM regs used for parameter passing.

  static const unsigned XMMArgRegs[] = {
    X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3
  };
  
  for (unsigned i = 0; i < NumArgs; ++i) {
    MVT::ValueType ObjectVT = Op.getValue(i).getValueType();
    unsigned ArgIncrement = 4;
    unsigned ObjSize = 0;
    unsigned ObjIntRegs = 0;
    unsigned ObjXMMRegs = 0;

    HowToPassFastCCArgument(ObjectVT, NumIntRegs, NumXMMRegs,
                            ObjSize, ObjIntRegs, ObjXMMRegs);
    if (ObjSize > 4)
      ArgIncrement = ObjSize;

    unsigned Reg = 0;
    SDOperand ArgValue;
    if (ObjIntRegs || ObjXMMRegs) {
      switch (ObjectVT) {
      default: assert(0 && "Unhandled argument type!");
      case MVT::i8:
        Reg = AddLiveIn(MF, NumIntRegs ? X86::DL : X86::AL,
                        X86::GR8RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, MVT::i8);
        break;
      case MVT::i16:
        Reg = AddLiveIn(MF, NumIntRegs ? X86::DX : X86::AX,
                        X86::GR16RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, MVT::i16);
        break;
      case MVT::i32:
        Reg = AddLiveIn(MF, NumIntRegs ? X86::EDX : X86::EAX,
                        X86::GR32RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, MVT::i32);
        break;
      case MVT::i64:
        Reg = AddLiveIn(MF, NumIntRegs ? X86::EDX : X86::EAX,
                        X86::GR32RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, MVT::i32);
        if (ObjIntRegs == 2) {
          Reg = AddLiveIn(MF, X86::EDX, X86::GR32RegisterClass);
          SDOperand ArgValue2 = DAG.getCopyFromReg(Root, Reg, MVT::i32);
          ArgValue= DAG.getNode(ISD::BUILD_PAIR, MVT::i64, ArgValue, ArgValue2);
        }
        break;
      case MVT::v16i8:
      case MVT::v8i16:
      case MVT::v4i32:
      case MVT::v2i64:
      case MVT::v4f32:
      case MVT::v2f64:
        Reg = AddLiveIn(MF, XMMArgRegs[NumXMMRegs], X86::VR128RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, ObjectVT);
        break;
      }
      NumIntRegs += ObjIntRegs;
      NumXMMRegs += ObjXMMRegs;
    }

    if (ObjSize) {
      // XMM arguments have to be aligned on 16-byte boundary.
      if (ObjSize == 16)
        ArgOffset = ((ArgOffset + 15) / 16) * 16;
      // Create the SelectionDAG nodes corresponding to a load from this
      // parameter.
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, getPointerTy());
      if (ObjectVT == MVT::i64 && ObjIntRegs) {
        SDOperand ArgValue2 = DAG.getLoad(Op.Val->getValueType(i), Root, FIN,
                                          NULL, 0);
        ArgValue = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, ArgValue, ArgValue2);
      } else
        ArgValue = DAG.getLoad(Op.Val->getValueType(i), Root, FIN, NULL, 0);
      ArgOffset += ArgIncrement;   // Move on to the next argument.
    }

    ArgValues.push_back(ArgValue);
  }

  ArgValues.push_back(Root);

  // Make sure the instruction takes 8n+4 bytes to make sure the start of the
  // arguments and the arguments after the retaddr has been pushed are aligned.
  if ((ArgOffset & 7) == 0)
    ArgOffset += 4;

  VarArgsFrameIndex = 0xAAAAAAA;   // fastcc functions can't have varargs.
  RegSaveFrameIndex = 0xAAAAAAA;   // X86-64 only.
  ReturnAddrIndex = 0;             // No return address slot generated yet.
  BytesToPopOnReturn = ArgOffset;  // Callee pops all stack arguments.
  BytesCallerReserves = 0;

  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(MF.getFunction()->getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    MF.addLiveOut(X86::EAX);
    break;
  case MVT::i64:
    MF.addLiveOut(X86::EAX);
    MF.addLiveOut(X86::EDX);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.addLiveOut(X86::ST0);
    break;
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v2f64:
    MF.addLiveOut(X86::XMM0);
    break;
  }

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVTs(Op.Val->value_begin(),
                                     Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVTs, &ArgValues[0],ArgValues.size());
}

SDOperand X86TargetLowering::LowerFastCCCallTo(SDOperand Op, SelectionDAG &DAG,
                                               bool isFastCall) {
  SDOperand Chain     = Op.getOperand(0);
  bool isTailCall     = cast<ConstantSDNode>(Op.getOperand(3))->getValue() != 0;
  SDOperand Callee    = Op.getOperand(4);
  MVT::ValueType RetVT= Op.Val->getValueType(0);
  unsigned NumOps     = (Op.getNumOperands() - 5) / 2;

  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;

  // Keep track of the number of integer regs passed so far.  This can be either
  // 0 (neither EAX or EDX used), 1 (EAX is used) or 2 (EAX and EDX are both
  // used).
  unsigned NumIntRegs = 0;
  unsigned NumXMMRegs = 0;  // XMM regs used for parameter passing.

  static const unsigned GPRArgRegs[][2] = {
    { X86::AL,  X86::DL },
    { X86::AX,  X86::DX },
    { X86::EAX, X86::EDX }
  };
#if 0
  static const unsigned FastCallGPRArgRegs[][2] = {
    { X86::CL,  X86::DL },
    { X86::CX,  X86::DX },
    { X86::ECX, X86::EDX }
  };  
#endif
  static const unsigned XMMArgRegs[] = {
    X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3
  };

  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);

    switch (Arg.getValueType()) {
    default: assert(0 && "Unknown value type!");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32: {
     unsigned MaxNumIntRegs = (isFastCall ? 2 : FASTCC_NUM_INT_ARGS_INREGS);
     if (NumIntRegs < MaxNumIntRegs) {
       ++NumIntRegs;
       break;
     }
     } // Fall through
    case MVT::f32:
      NumBytes += 4;
      break;
    case MVT::f64:
      NumBytes += 8;
      break;
    case MVT::v16i8:
    case MVT::v8i16:
    case MVT::v4i32:
    case MVT::v2i64:
    case MVT::v4f32:
    case MVT::v2f64:
     if (isFastCall) {
      assert(0 && "Unknown value type!");
     } else {
       if (NumXMMRegs < 4)
         NumXMMRegs++;
       else {
         // XMM arguments have to be aligned on 16-byte boundary.
         NumBytes = ((NumBytes + 15) / 16) * 16;
         NumBytes += 16;
       }
     }
     break;
    }
  }

  // Make sure the instruction takes 8n+4 bytes to make sure the start of the
  // arguments and the arguments after the retaddr has been pushed are aligned.
  if ((NumBytes & 7) == 0)
    NumBytes += 4;

  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(NumBytes, getPointerTy()));

  // Arguments go on the stack in reverse order, as specified by the ABI.
  unsigned ArgOffset = 0;
  NumIntRegs = 0;
  std::vector<std::pair<unsigned, SDOperand> > RegsToPass;
  std::vector<SDOperand> MemOpChains;
  SDOperand StackPtr = DAG.getRegister(X86StackPtr, getPointerTy());
  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);

    switch (Arg.getValueType()) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32: {
     unsigned MaxNumIntRegs = (isFastCall ? 2 : FASTCC_NUM_INT_ARGS_INREGS);
     if (NumIntRegs < MaxNumIntRegs) {
       RegsToPass.push_back(
         std::make_pair(GPRArgRegs[Arg.getValueType()-MVT::i8][NumIntRegs],
                        Arg));
       ++NumIntRegs;
       break;
     }
     } // Fall through
    case MVT::f32: {
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
      ArgOffset += 4;
      break;
    }
    case MVT::f64: {
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
      ArgOffset += 8;
      break;
    }
    case MVT::v16i8:
    case MVT::v8i16:
    case MVT::v4i32:
    case MVT::v2i64:
    case MVT::v4f32:
    case MVT::v2f64:
     if (isFastCall) {
       assert(0 && "Unexpected ValueType for argument!");
     } else {
       if (NumXMMRegs < 4) {
         RegsToPass.push_back(std::make_pair(XMMArgRegs[NumXMMRegs], Arg));
         NumXMMRegs++;
       } else {
         // XMM arguments have to be aligned on 16-byte boundary.
         ArgOffset = ((ArgOffset + 15) / 16) * 16;
         SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
         PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
         MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
         ArgOffset += 16;
       }       
     }
     break;
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into registers.
  SDOperand InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                             InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first, 
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.Val)
    Ops.push_back(InFlag);

  // FIXME: Do not generate X86ISD::TAILCALL for now.
  Chain = DAG.getNode(isTailCall ? X86ISD::TAILCALL : X86ISD::CALL,
                      NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  NodeTys.clear();
  NodeTys.push_back(MVT::Other);   // Returns a chain
  if (RetVT != MVT::Other)
    NodeTys.push_back(MVT::Flag);  // Returns a flag for retval copy to use.
  Ops.clear();
  Ops.push_back(Chain);
  Ops.push_back(DAG.getConstant(NumBytes, getPointerTy()));
  Ops.push_back(DAG.getConstant(NumBytes, getPointerTy()));
  Ops.push_back(InFlag);
  Chain = DAG.getNode(ISD::CALLSEQ_END, NodeTys, &Ops[0], Ops.size());
  if (RetVT != MVT::Other)
    InFlag = Chain.getValue(1);
  
  std::vector<SDOperand> ResultVals;
  NodeTys.clear();
  switch (RetVT) {
  default: assert(0 && "Unknown value type to return!");
  case MVT::Other: break;
  case MVT::i8:
    Chain = DAG.getCopyFromReg(Chain, X86::AL, MVT::i8, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i8);
    break;
  case MVT::i16:
    Chain = DAG.getCopyFromReg(Chain, X86::AX, MVT::i16, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i16);
    break;
  case MVT::i32:
    if (Op.Val->getValueType(1) == MVT::i32) {
      Chain = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      Chain = DAG.getCopyFromReg(Chain, X86::EDX, MVT::i32,
                                 Chain.getValue(2)).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      NodeTys.push_back(MVT::i32);
    } else {
      Chain = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
    }
    NodeTys.push_back(MVT::i32);
    break;
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v2f64:
   if (isFastCall) {
     assert(0 && "Unknown value type to return!");
   } else {
     Chain = DAG.getCopyFromReg(Chain, X86::XMM0, RetVT, InFlag).getValue(1);
     ResultVals.push_back(Chain.getValue(0));
     NodeTys.push_back(RetVT);
   }
   break;
  case MVT::f32:
  case MVT::f64: {
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::f64);
    Tys.push_back(MVT::Other);
    Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(InFlag);
    SDOperand RetVal = DAG.getNode(X86ISD::FP_GET_RESULT, Tys,
                                   &Ops[0], Ops.size());
    Chain  = RetVal.getValue(1);
    InFlag = RetVal.getValue(2);
    if (X86ScalarSSE) {
      // FIXME: Currently the FST is flagged to the FP_GET_RESULT. This
      // shouldn't be necessary except that RFP cannot be live across
      // multiple blocks. When stackifier is fixed, they can be uncoupled.
      MachineFunction &MF = DAG.getMachineFunction();
      int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
      SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
      Tys.clear();
      Tys.push_back(MVT::Other);
      Ops.clear();
      Ops.push_back(Chain);
      Ops.push_back(RetVal);
      Ops.push_back(StackSlot);
      Ops.push_back(DAG.getValueType(RetVT));
      Ops.push_back(InFlag);
      Chain = DAG.getNode(X86ISD::FST, Tys, &Ops[0], Ops.size());
      RetVal = DAG.getLoad(RetVT, Chain, StackSlot, NULL, 0);
      Chain = RetVal.getValue(1);
    }

    if (RetVT == MVT::f32 && !X86ScalarSSE)
      // FIXME: we would really like to remember that this FP_ROUND
      // operation is okay to eliminate if we allow excess FP precision.
      RetVal = DAG.getNode(ISD::FP_ROUND, MVT::f32, RetVal);
    ResultVals.push_back(RetVal);
    NodeTys.push_back(RetVT);
    break;
  }
  }


  // If the function returns void, just return the chain.
  if (ResultVals.empty())
    return Chain;
  
  // Otherwise, merge everything together with a MERGE_VALUES node.
  NodeTys.push_back(MVT::Other);
  ResultVals.push_back(Chain);
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys,
                              &ResultVals[0], ResultVals.size());
  return Res.getValue(Op.ResNo);
}

//===----------------------------------------------------------------------===//
//                  StdCall Calling Convention implementation
//===----------------------------------------------------------------------===//
//  StdCall calling convention seems to be standard for many Windows' API
//  routines and around. It differs from C calling convention just a little:
//  callee should clean up the stack, not caller. Symbols should be also
//  decorated in some fancy way :) It doesn't support any vector arguments.

/// HowToPassStdCallCCArgument - Returns how an formal argument of the specified
/// type should be passed. Returns the size of the stack slot
static void
HowToPassStdCallCCArgument(MVT::ValueType ObjectVT, unsigned &ObjSize) {
  switch (ObjectVT) {
  default: assert(0 && "Unhandled argument type!");
  case MVT::i8:  ObjSize = 1; break;
  case MVT::i16: ObjSize = 2; break;
  case MVT::i32: ObjSize = 4; break;
  case MVT::i64: ObjSize = 8; break;
  case MVT::f32: ObjSize = 4; break;
  case MVT::f64: ObjSize = 8; break;
  }
}

SDOperand X86TargetLowering::LowerStdCallCCArguments(SDOperand Op,
                                                     SelectionDAG &DAG) {
  unsigned NumArgs = Op.Val->getNumValues() - 1;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SDOperand Root = Op.getOperand(0);
  std::vector<SDOperand> ArgValues;

  // Add DAG nodes to load the arguments...  On entry to a function on the X86,
  // the stack frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first argument (leftmost lexically)
  // [ESP + 8] -- second argument, if first argument is <= 4 bytes in size
  //    ...
  //
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  for (unsigned i = 0; i < NumArgs; ++i) {
    MVT::ValueType ObjectVT = Op.getValue(i).getValueType();
    unsigned ArgIncrement = 4;
    unsigned ObjSize = 0;
    HowToPassStdCallCCArgument(ObjectVT, ObjSize);
    if (ObjSize > 4)
      ArgIncrement = ObjSize;

    SDOperand ArgValue;
    // Create the frame index object for this incoming parameter...
    int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
    SDOperand FIN = DAG.getFrameIndex(FI, getPointerTy());
    ArgValue = DAG.getLoad(Op.Val->getValueType(i), Root, FIN, NULL, 0);
    ArgValues.push_back(ArgValue);
    ArgOffset += ArgIncrement;   // Move on to the next argument...
  }

  ArgValues.push_back(Root);
  
  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  if (isVarArg) {
    BytesToPopOnReturn = 0;         // Callee pops nothing.
    BytesCallerReserves = ArgOffset;
    VarArgsFrameIndex = MFI->CreateFixedObject(1, ArgOffset);
  } else {
    BytesToPopOnReturn = ArgOffset; // Callee pops everything..
    BytesCallerReserves = 0;
  }
  RegSaveFrameIndex = 0xAAAAAAA;    // X86-64 only.
  ReturnAddrIndex = 0;              // No return address slot generated yet.

  MF.getInfo<X86FunctionInfo>()->setBytesToPopOnReturn(BytesToPopOnReturn);
  
  // Return the new list of results.
  std::vector<MVT::ValueType> RetVTs(Op.Val->value_begin(),
                                     Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVTs, &ArgValues[0],ArgValues.size());
}


SDOperand X86TargetLowering::LowerStdCallCCCallTo(SDOperand Op,
                                                  SelectionDAG &DAG) {
  SDOperand Chain     = Op.getOperand(0);
  bool isVarArg       = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  bool isTailCall     = cast<ConstantSDNode>(Op.getOperand(3))->getValue() != 0;
  SDOperand Callee    = Op.getOperand(4);
  MVT::ValueType RetVT= Op.Val->getValueType(0);
  unsigned NumOps     = (Op.getNumOperands() - 5) / 2;  
  
  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;
  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);

    switch (Arg.getValueType()) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::f32:
      NumBytes += 4;
      break;
    case MVT::i64:
    case MVT::f64:
      NumBytes += 8;
      break;
    }
  }
  
  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(NumBytes, getPointerTy()));

  // Arguments go on the stack in reverse order, as specified by the ABI.
  unsigned ArgOffset = 0;
  std::vector<SDOperand> MemOpChains;
  SDOperand StackPtr = DAG.getRegister(X86StackPtr, getPointerTy());
  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);

    switch (Arg.getValueType()) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i8:
    case MVT::i16: {
      // Promote the integer to 32 bits.  If the input type is signed use a
      // sign extend, otherwise use a zero extend.
      unsigned ExtOp =
        dyn_cast<ConstantSDNode>(Op.getOperand(5+2*i+1))->getValue() ?
        ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
      Arg = DAG.getNode(ExtOp, MVT::i32, Arg);
    }
    // Fallthrough

    case MVT::i32:
    case MVT::f32: {
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
      ArgOffset += 4;
      break;
    }
    case MVT::i64:
    case MVT::f64: {
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, getPointerTy(), StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
      ArgOffset += 8;
      break;
    }
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  Chain = DAG.getNode(isTailCall ? X86ISD::TAILCALL : X86ISD::CALL,
                      NodeTys, &Ops[0], Ops.size());
  SDOperand InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  unsigned NumBytesForCalleeToPush;
  
  if (isVarArg) {
    NumBytesForCalleeToPush = 0;
  } else {
    NumBytesForCalleeToPush = NumBytes;
  }

  NodeTys.clear();
  NodeTys.push_back(MVT::Other);   // Returns a chain
  if (RetVT != MVT::Other)
    NodeTys.push_back(MVT::Flag);  // Returns a flag for retval copy to use.
  Ops.clear();
  Ops.push_back(Chain);
  Ops.push_back(DAG.getConstant(NumBytes, getPointerTy()));
  Ops.push_back(DAG.getConstant(NumBytesForCalleeToPush, getPointerTy()));
  Ops.push_back(InFlag);
  Chain = DAG.getNode(ISD::CALLSEQ_END, NodeTys, &Ops[0], Ops.size());
  if (RetVT != MVT::Other)
    InFlag = Chain.getValue(1);
  
  std::vector<SDOperand> ResultVals;
  NodeTys.clear();
  switch (RetVT) {
  default: assert(0 && "Unknown value type to return!");
  case MVT::Other: break;
  case MVT::i8:
    Chain = DAG.getCopyFromReg(Chain, X86::AL, MVT::i8, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i8);
    break;
  case MVT::i16:
    Chain = DAG.getCopyFromReg(Chain, X86::AX, MVT::i16, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i16);
    break;
  case MVT::i32:
    if (Op.Val->getValueType(1) == MVT::i32) {
      Chain = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      Chain = DAG.getCopyFromReg(Chain, X86::EDX, MVT::i32,
                                 Chain.getValue(2)).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      NodeTys.push_back(MVT::i32);
    } else {
      Chain = DAG.getCopyFromReg(Chain, X86::EAX, MVT::i32, InFlag).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
    }
    NodeTys.push_back(MVT::i32);
    break;
  case MVT::f32:
  case MVT::f64: {
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::f64);
    Tys.push_back(MVT::Other);
    Tys.push_back(MVT::Flag);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(InFlag);
    SDOperand RetVal = DAG.getNode(X86ISD::FP_GET_RESULT, Tys, 
                                   &Ops[0], Ops.size());
    Chain  = RetVal.getValue(1);
    InFlag = RetVal.getValue(2);
    if (X86ScalarSSE) {
      // FIXME: Currently the FST is flagged to the FP_GET_RESULT. This
      // shouldn't be necessary except that RFP cannot be live across
      // multiple blocks. When stackifier is fixed, they can be uncoupled.
      MachineFunction &MF = DAG.getMachineFunction();
      int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
      SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
      Tys.clear();
      Tys.push_back(MVT::Other);
      Ops.clear();
      Ops.push_back(Chain);
      Ops.push_back(RetVal);
      Ops.push_back(StackSlot);
      Ops.push_back(DAG.getValueType(RetVT));
      Ops.push_back(InFlag);
      Chain = DAG.getNode(X86ISD::FST, Tys, &Ops[0], Ops.size());
      RetVal = DAG.getLoad(RetVT, Chain, StackSlot, NULL, 0);
      Chain = RetVal.getValue(1);
    }

    if (RetVT == MVT::f32 && !X86ScalarSSE)
      // FIXME: we would really like to remember that this FP_ROUND
      // operation is okay to eliminate if we allow excess FP precision.
      RetVal = DAG.getNode(ISD::FP_ROUND, MVT::f32, RetVal);
    ResultVals.push_back(RetVal);
    NodeTys.push_back(RetVT);
    break;
  }
  }

  // If the function returns void, just return the chain.
  if (ResultVals.empty())
    return Chain;
  
  // Otherwise, merge everything together with a MERGE_VALUES node.
  NodeTys.push_back(MVT::Other);
  ResultVals.push_back(Chain);
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys,
                              &ResultVals[0], ResultVals.size());
  return Res.getValue(Op.ResNo);
}

//===----------------------------------------------------------------------===//
//                  FastCall Calling Convention implementation
//===----------------------------------------------------------------------===//
//
// The X86 'fastcall' calling convention passes up to two integer arguments in
// registers (an appropriate portion of ECX/EDX), passes arguments in C order,
// and requires that the callee pop its arguments off the stack (allowing proper
// tail calls), and has the same return value conventions as C calling convs.
//
// This calling convention always arranges for the callee pop value to be 8n+4
// bytes, which is needed for tail recursion elimination and stack alignment
// reasons.
//

/// HowToPassFastCallCCArgument - Returns how an formal argument of the
/// specified type should be passed. If it is through stack, returns the size of
/// the stack slot; if it is through integer register, returns the number of
/// integer registers are needed.
static void
HowToPassFastCallCCArgument(MVT::ValueType ObjectVT,
                            unsigned NumIntRegs,
                            unsigned &ObjSize,
                            unsigned &ObjIntRegs)
{
  ObjSize = 0;
  ObjIntRegs = 0;

  switch (ObjectVT) {
  default: assert(0 && "Unhandled argument type!");
  case MVT::i8:
   if (NumIntRegs < 2)
     ObjIntRegs = 1;
   else
     ObjSize = 1;
   break;
  case MVT::i16:
   if (NumIntRegs < 2)
     ObjIntRegs = 1;
   else
     ObjSize = 2;
   break;
  case MVT::i32:
   if (NumIntRegs < 2)
     ObjIntRegs = 1;
   else
     ObjSize = 4;
    break;
  case MVT::i64:
   if (NumIntRegs+2 <= 2) {
     ObjIntRegs = 2;
   } else if (NumIntRegs+1 <= 2) {
     ObjIntRegs = 1;
     ObjSize = 4;
   } else
     ObjSize = 8;
   case MVT::f32:
    ObjSize = 4;
    break;
   case MVT::f64:
    ObjSize = 8;
    break;
  }
}

SDOperand
X86TargetLowering::LowerFastCallCCArguments(SDOperand Op, SelectionDAG &DAG) {
  unsigned NumArgs = Op.Val->getNumValues()-1;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SDOperand Root = Op.getOperand(0);
  std::vector<SDOperand> ArgValues;

  // Add DAG nodes to load the arguments...  On entry to a function the stack
  // frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first nonreg argument (leftmost lexically)
  // [ESP + 8] -- second nonreg argument, if 1st argument is <= 4 bytes in size
  //    ...
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot

  // Keep track of the number of integer regs passed so far.  This can be either
  // 0 (neither ECX or EDX used), 1 (ECX is used) or 2 (ECX and EDX are both
  // used).
  unsigned NumIntRegs = 0;

  for (unsigned i = 0; i < NumArgs; ++i) {
    MVT::ValueType ObjectVT = Op.getValue(i).getValueType();
    unsigned ArgIncrement = 4;
    unsigned ObjSize = 0;
    unsigned ObjIntRegs = 0;

    HowToPassFastCallCCArgument(ObjectVT, NumIntRegs, ObjSize, ObjIntRegs);
    if (ObjSize > 4)
      ArgIncrement = ObjSize;

    unsigned Reg = 0;
    SDOperand ArgValue;
    if (ObjIntRegs) {
      switch (ObjectVT) {
      default: assert(0 && "Unhandled argument type!");
      case MVT::i8:
        Reg = AddLiveIn(MF, NumIntRegs ? X86::DL : X86::CL,
                        X86::GR8RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, MVT::i8);
        break;
      case MVT::i16:
        Reg = AddLiveIn(MF, NumIntRegs ? X86::DX : X86::CX,
                        X86::GR16RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, MVT::i16);
        break;
      case MVT::i32:
        Reg = AddLiveIn(MF, NumIntRegs ? X86::EDX : X86::ECX,
                        X86::GR32RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, MVT::i32);
        break;
      case MVT::i64:
        Reg = AddLiveIn(MF, NumIntRegs ? X86::EDX : X86::ECX,
                        X86::GR32RegisterClass);
        ArgValue = DAG.getCopyFromReg(Root, Reg, MVT::i32);
        if (ObjIntRegs == 2) {
          Reg = AddLiveIn(MF, X86::EDX, X86::GR32RegisterClass);
          SDOperand ArgValue2 = DAG.getCopyFromReg(Root, Reg, MVT::i32);
          ArgValue= DAG.getNode(ISD::BUILD_PAIR, MVT::i64, ArgValue, ArgValue2);
        }
        break;
      }
      
      NumIntRegs += ObjIntRegs;
    }

    if (ObjSize) {
      // Create the SelectionDAG nodes corresponding to a load from this
      // parameter.
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, getPointerTy());
      if (ObjectVT == MVT::i64 && ObjIntRegs) {
        SDOperand ArgValue2 = DAG.getLoad(Op.Val->getValueType(i), Root, FIN,
                                          NULL, 0);
        ArgValue = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, ArgValue, ArgValue2);
      } else
        ArgValue = DAG.getLoad(Op.Val->getValueType(i), Root, FIN, NULL, 0);
      ArgOffset += ArgIncrement;   // Move on to the next argument.
    }

    ArgValues.push_back(ArgValue);
  }

  ArgValues.push_back(Root);

  // Make sure the instruction takes 8n+4 bytes to make sure the start of the
  // arguments and the arguments after the retaddr has been pushed are aligned.
  if ((ArgOffset & 7) == 0)
    ArgOffset += 4;

  VarArgsFrameIndex = 0xAAAAAAA;   // fastcc functions can't have varargs.
  RegSaveFrameIndex = 0xAAAAAAA;   // X86-64 only.
  ReturnAddrIndex = 0;             // No return address slot generated yet.
  BytesToPopOnReturn = ArgOffset;  // Callee pops all stack arguments.
  BytesCallerReserves = 0;

  MF.getInfo<X86FunctionInfo>()->setBytesToPopOnReturn(BytesToPopOnReturn);

  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(MF.getFunction()->getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    MF.addLiveOut(X86::ECX);
    break;
  case MVT::i64:
    MF.addLiveOut(X86::ECX);
    MF.addLiveOut(X86::EDX);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.addLiveOut(X86::ST0);
    break;
  }

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVTs(Op.Val->value_begin(),
                                     Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVTs, &ArgValues[0],ArgValues.size());
}

SDOperand X86TargetLowering::getReturnAddressFrameIndex(SelectionDAG &DAG) {
  if (ReturnAddrIndex == 0) {
    // Set up a frame object for the return address.
    MachineFunction &MF = DAG.getMachineFunction();
    if (Subtarget->is64Bit())
      ReturnAddrIndex = MF.getFrameInfo()->CreateFixedObject(8, -8);
    else
      ReturnAddrIndex = MF.getFrameInfo()->CreateFixedObject(4, -4);
  }

  return DAG.getFrameIndex(ReturnAddrIndex, getPointerTy());
}



std::pair<SDOperand, SDOperand> X86TargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  SDOperand Result;
  if (Depth)        // Depths > 0 not supported yet!
    Result = DAG.getConstant(0, getPointerTy());
  else {
    SDOperand RetAddrFI = getReturnAddressFrameIndex(DAG);
    if (!isFrameAddress)
      // Just load the return address
      Result = DAG.getLoad(getPointerTy(), DAG.getEntryNode(), RetAddrFI,
                           NULL, 0);
    else
      Result = DAG.getNode(ISD::SUB, getPointerTy(), RetAddrFI,
                           DAG.getConstant(4, getPointerTy()));
  }
  return std::make_pair(Result, Chain);
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

/// DarwinGVRequiresExtraLoad - true if accessing the GV requires an extra
/// load. For Darwin, external and weak symbols are indirect, loading the value
/// at address GV rather then the value of GV itself. This means that the
/// GlobalAddress must be in the base or index register of the address, not the
/// GV offset field.
static bool DarwinGVRequiresExtraLoad(GlobalValue *GV) {
  return (GV->hasWeakLinkage() || GV->hasLinkOnceLinkage() ||
          (GV->isExternal() && !GV->hasNotBeenReadFromBytecode()));
}

/// WindowsGVRequiresExtraLoad - true if accessing the GV requires an extra
/// load. For Windows, dllimported symbols are indirect, loading the value at
/// address GV rather then the value of GV itself. This means that the
/// GlobalAddress must be in the base or index register of the address, not the
/// GV offset field.
static bool WindowsGVRequiresExtraLoad(GlobalValue *GV) {
  return (GV->hasDLLImportLinkage());  
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

  if (N->getNumOperands() != 4)
    return false;

  // Check if the value doesn't reference the second vector.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDOperand Arg = N->getOperand(i);
    if (Arg.getOpcode() == ISD::UNDEF) continue;
    assert(isa<ConstantSDNode>(Arg) && "Invalid VECTOR_SHUFFLE mask!");
    if (cast<ConstantSDNode>(Arg)->getValue() >= 4)
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
static bool isSHUFPMask(std::vector<SDOperand> &N) {
  unsigned NumElems = N.size();
  if (NumElems != 2 && NumElems != 4) return false;

  unsigned Half = NumElems / 2;
  for (unsigned i = 0; i < Half; ++i)
    if (!isUndefOrInRange(N[i], 0, NumElems))
      return false;
  for (unsigned i = Half; i < NumElems; ++i)
    if (!isUndefOrInRange(N[i], NumElems, NumElems*2))
      return false;

  return true;
}

bool X86::isSHUFPMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
  return ::isSHUFPMask(Ops);
}

/// isCommutedSHUFP - Returns true if the shuffle mask is except
/// the reverse of what x86 shuffles want. x86 shuffles requires the lower
/// half elements to come from vector 1 (which would equal the dest.) and
/// the upper half to come from vector 2.
static bool isCommutedSHUFP(std::vector<SDOperand> &Ops) {
  unsigned NumElems = Ops.size();
  if (NumElems != 2 && NumElems != 4) return false;

  unsigned Half = NumElems / 2;
  for (unsigned i = 0; i < Half; ++i)
    if (!isUndefOrInRange(Ops[i], NumElems, NumElems*2))
      return false;
  for (unsigned i = Half; i < NumElems; ++i)
    if (!isUndefOrInRange(Ops[i], 0, NumElems))
      return false;
  return true;
}

static bool isCommutedSHUFP(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
  return isCommutedSHUFP(Ops);
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
bool static isUNPCKLMask(std::vector<SDOperand> &N, bool V2IsSplat = false) {
  unsigned NumElems = N.size();
  if (NumElems != 2 && NumElems != 4 && NumElems != 8 && NumElems != 16)
    return false;

  for (unsigned i = 0, j = 0; i != NumElems; i += 2, ++j) {
    SDOperand BitI  = N[i];
    SDOperand BitI1 = N[i+1];
    if (!isUndefOrEqual(BitI, j))
      return false;
    if (V2IsSplat) {
      if (isUndefOrEqual(BitI1, NumElems))
        return false;
    } else {
      if (!isUndefOrEqual(BitI1, j + NumElems))
        return false;
    }
  }

  return true;
}

bool X86::isUNPCKLMask(SDNode *N, bool V2IsSplat) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
  return ::isUNPCKLMask(Ops, V2IsSplat);
}

/// isUNPCKHMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to UNPCKH.
bool static isUNPCKHMask(std::vector<SDOperand> &N, bool V2IsSplat = false) {
  unsigned NumElems = N.size();
  if (NumElems != 2 && NumElems != 4 && NumElems != 8 && NumElems != 16)
    return false;

  for (unsigned i = 0, j = 0; i != NumElems; i += 2, ++j) {
    SDOperand BitI  = N[i];
    SDOperand BitI1 = N[i+1];
    if (!isUndefOrEqual(BitI, j + NumElems/2))
      return false;
    if (V2IsSplat) {
      if (isUndefOrEqual(BitI1, NumElems))
        return false;
    } else {
      if (!isUndefOrEqual(BitI1, j + NumElems/2 + NumElems))
        return false;
    }
  }

  return true;
}

bool X86::isUNPCKHMask(SDNode *N, bool V2IsSplat) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
  return ::isUNPCKHMask(Ops, V2IsSplat);
}

/// isUNPCKL_v_undef_Mask - Special case of isUNPCKLMask for canonical form
/// of vector_shuffle v, v, <0, 4, 1, 5>, i.e. vector_shuffle v, undef,
/// <0, 0, 1, 1>
bool X86::isUNPCKL_v_undef_Mask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);

  unsigned NumElems = N->getNumOperands();
  if (NumElems != 4 && NumElems != 8 && NumElems != 16)
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

/// isMOVLMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a shuffle of elements that is suitable for input to MOVSS,
/// MOVSD, and MOVD, i.e. setting the lowest element.
static bool isMOVLMask(std::vector<SDOperand> &N) {
  unsigned NumElems = N.size();
  if (NumElems != 2 && NumElems != 4 && NumElems != 8 && NumElems != 16)
    return false;

  if (!isUndefOrEqual(N[0], NumElems))
    return false;

  for (unsigned i = 1; i < NumElems; ++i) {
    SDOperand Arg = N[i];
    if (!isUndefOrEqual(Arg, i))
      return false;
  }

  return true;
}

bool X86::isMOVLMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
  return ::isMOVLMask(Ops);
}

/// isCommutedMOVL - Returns true if the shuffle mask is except the reverse
/// of what x86 movss want. X86 movs requires the lowest  element to be lowest
/// element of vector 2 and the other elements to come from vector 1 in order.
static bool isCommutedMOVL(std::vector<SDOperand> &Ops, bool V2IsSplat = false,
                           bool V2IsUndef = false) {
  unsigned NumElems = Ops.size();
  if (NumElems != 2 && NumElems != 4 && NumElems != 8 && NumElems != 16)
    return false;

  if (!isUndefOrEqual(Ops[0], 0))
    return false;

  for (unsigned i = 1; i < NumElems; ++i) {
    SDOperand Arg = Ops[i];
    if (!(isUndefOrEqual(Arg, i+NumElems) ||
          (V2IsUndef && isUndefOrInRange(Arg, NumElems, NumElems*2)) ||
          (V2IsSplat && isUndefOrEqual(Arg, NumElems))))
      return false;
  }

  return true;
}

static bool isCommutedMOVL(SDNode *N, bool V2IsSplat = false,
                           bool V2IsUndef = false) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  std::vector<SDOperand> Ops(N->op_begin(), N->op_end());
  return isCommutedMOVL(Ops, V2IsSplat, V2IsUndef);
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
    if (Val > 4)
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

/// CommuteVectorShuffle - Swap vector_shuffle operandsas well as
/// values in ther permute mask.
static SDOperand CommuteVectorShuffle(SDOperand Op, SDOperand &V1,
                                      SDOperand &V2, SDOperand &Mask,
                                      SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType MaskVT = Mask.getValueType();
  MVT::ValueType EltVT = MVT::getVectorBaseType(MaskVT);
  unsigned NumElems = Mask.getNumOperands();
  std::vector<SDOperand> MaskVec;

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
  Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], MaskVec.size());
  return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2, Mask);
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
/// is promoted to a vector.
static inline bool isScalarLoadToVector(SDNode *N) {
  if (N->getOpcode() == ISD::SCALAR_TO_VECTOR) {
    N = N->getOperand(0).Val;
    return ISD::isNON_EXTLoad(N);
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
  if (N->getOpcode() != ISD::BUILD_VECTOR)
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

/// NormalizeMask - V2 is a splat, modify the mask (if needed) so all elements
/// that point to V2 points to its first element.
static SDOperand NormalizeMask(SDOperand Mask, SelectionDAG &DAG) {
  assert(Mask.getOpcode() == ISD::BUILD_VECTOR);

  bool Changed = false;
  std::vector<SDOperand> MaskVec;
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
  MVT::ValueType BaseVT = MVT::getVectorBaseType(MaskVT);

  std::vector<SDOperand> MaskVec;
  MaskVec.push_back(DAG.getConstant(NumElems, BaseVT));
  for (unsigned i = 1; i != NumElems; ++i)
    MaskVec.push_back(DAG.getConstant(i, BaseVT));
  return DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], MaskVec.size());
}

/// getUnpacklMask - Returns a vector_shuffle mask for an unpackl operation
/// of specified width.
static SDOperand getUnpacklMask(unsigned NumElems, SelectionDAG &DAG) {
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
  MVT::ValueType BaseVT = MVT::getVectorBaseType(MaskVT);
  std::vector<SDOperand> MaskVec;
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
  MVT::ValueType BaseVT = MVT::getVectorBaseType(MaskVT);
  unsigned Half = NumElems/2;
  std::vector<SDOperand> MaskVec;
  for (unsigned i = 0; i != Half; ++i) {
    MaskVec.push_back(DAG.getConstant(i + Half,            BaseVT));
    MaskVec.push_back(DAG.getConstant(i + NumElems + Half, BaseVT));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0], MaskVec.size());
}

/// getZeroVector - Returns a vector of specified type with all zero elements.
///
static SDOperand getZeroVector(MVT::ValueType VT, SelectionDAG &DAG) {
  assert(MVT::isVector(VT) && "Expected a vector type");
  unsigned NumElems = getVectorNumElements(VT);
  MVT::ValueType EVT = MVT::getVectorBaseType(VT);
  bool isFP = MVT::isFloatingPoint(EVT);
  SDOperand Zero = isFP ? DAG.getConstantFP(0.0, EVT) : DAG.getConstant(0, EVT);
  std::vector<SDOperand> ZeroVec(NumElems, Zero);
  return DAG.getNode(ISD::BUILD_VECTOR, VT, &ZeroVec[0], ZeroVec.size());
}

/// PromoteSplat - Promote a splat of v8i16 or v16i8 to v4i32.
///
static SDOperand PromoteSplat(SDOperand Op, SelectionDAG &DAG) {
  SDOperand V1 = Op.getOperand(0);
  SDOperand Mask = Op.getOperand(2);
  MVT::ValueType VT = Op.getValueType();
  unsigned NumElems = Mask.getNumOperands();
  Mask = getUnpacklMask(NumElems, DAG);
  while (NumElems != 4) {
    V1 = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V1, Mask);
    NumElems >>= 1;
  }
  V1 = DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32, V1);

  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(4);
  Mask = getZeroVector(MaskVT, DAG);
  SDOperand Shuffle = DAG.getNode(ISD::VECTOR_SHUFFLE, MVT::v4i32, V1,
                                  DAG.getNode(ISD::UNDEF, MVT::v4i32), Mask);
  return DAG.getNode(ISD::BIT_CONVERT, VT, Shuffle);
}

/// isZeroNode - Returns true if Elt is a constant zero or a floating point
/// constant +0.0.
static inline bool isZeroNode(SDOperand Elt) {
  return ((isa<ConstantSDNode>(Elt) &&
           cast<ConstantSDNode>(Elt)->getValue() == 0) ||
          (isa<ConstantFPSDNode>(Elt) &&
           cast<ConstantFPSDNode>(Elt)->isExactlyValue(0.0)));
}

/// getShuffleVectorZeroOrUndef - Return a vector_shuffle of the specified
/// vector and zero or undef vector.
static SDOperand getShuffleVectorZeroOrUndef(SDOperand V2, MVT::ValueType VT,
                                             unsigned NumElems, unsigned Idx,
                                             bool isZero, SelectionDAG &DAG) {
  SDOperand V1 = isZero ? getZeroVector(VT, DAG) : DAG.getNode(ISD::UNDEF, VT);
  MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
  MVT::ValueType EVT = MVT::getVectorBaseType(MaskVT);
  SDOperand Zero = DAG.getConstant(0, EVT);
  std::vector<SDOperand> MaskVec(NumElems, Zero);
  MaskVec[Idx] = DAG.getConstant(NumElems, EVT);
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
                        DAG.getConstant(i/2, TLI.getPointerTy()));
    }
  }

  return DAG.getNode(ISD::BIT_CONVERT, MVT::v16i8, V);
}

/// LowerBuildVectorv16i8 - Custom lower build_vector of v8i16.
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
                      DAG.getConstant(i, TLI.getPointerTy()));
    }
  }

  return V;
}

SDOperand
X86TargetLowering::LowerBUILD_VECTOR(SDOperand Op, SelectionDAG &DAG) {
  // All zero's are handled with pxor.
  if (ISD::isBuildVectorAllZeros(Op.Val))
    return Op;

  // All one's are handled with pcmpeqd.
  if (ISD::isBuildVectorAllOnes(Op.Val))
    return Op;

  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType EVT = MVT::getVectorBaseType(VT);
  unsigned EVTBits = MVT::getSizeInBits(EVT);

  unsigned NumElems = Op.getNumOperands();
  unsigned NumZero  = 0;
  unsigned NumNonZero = 0;
  unsigned NonZeros = 0;
  std::set<SDOperand> Values;
  for (unsigned i = 0; i < NumElems; ++i) {
    SDOperand Elt = Op.getOperand(i);
    if (Elt.getOpcode() != ISD::UNDEF) {
      Values.insert(Elt);
      if (isZeroNode(Elt))
        NumZero++;
      else {
        NonZeros |= (1 << i);
        NumNonZero++;
      }
    }
  }

  if (NumNonZero == 0)
    // Must be a mix of zero and undef. Return a zero vector.
    return getZeroVector(VT, DAG);

  // Splat is obviously ok. Let legalizer expand it to a shuffle.
  if (Values.size() == 1)
    return SDOperand();

  // Special case for single non-zero element.
  if (NumNonZero == 1) {
    unsigned Idx = CountTrailingZeros_32(NonZeros);
    SDOperand Item = Op.getOperand(Idx);
    Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, Item);
    if (Idx == 0)
      // Turn it into a MOVL (i.e. movss, movsd, or movd) to a zero vector.
      return getShuffleVectorZeroOrUndef(Item, VT, NumElems, Idx,
                                         NumZero > 0, DAG);

    if (EVTBits == 32) {
      // Turn it into a shuffle of zero and zero-extended scalar to vector.
      Item = getShuffleVectorZeroOrUndef(Item, VT, NumElems, 0, NumZero > 0,
                                         DAG);
      MVT::ValueType MaskVT  = MVT::getIntVectorWithNumElements(NumElems);
      MVT::ValueType MaskEVT = MVT::getVectorBaseType(MaskVT);
      std::vector<SDOperand> MaskVec;
      for (unsigned i = 0; i < NumElems; i++)
        MaskVec.push_back(DAG.getConstant((i == Idx) ? 0 : 1, MaskEVT));
      SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                   &MaskVec[0], MaskVec.size());
      return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, Item,
                         DAG.getNode(ISD::UNDEF, VT), Mask);
    }
  }

  // Let legalizer expand 2-wide build_vector's.
  if (EVTBits == 64)
    return SDOperand();

  // If element VT is < 32 bits, convert it to inserts into a zero vector.
  if (EVTBits == 8) {
    SDOperand V = LowerBuildVectorv16i8(Op, NonZeros,NumNonZero,NumZero, DAG,
                                        *this);
    if (V.Val) return V;
  }

  if (EVTBits == 16) {
    SDOperand V = LowerBuildVectorv8i16(Op, NonZeros,NumNonZero,NumZero, DAG,
                                        *this);
    if (V.Val) return V;
  }

  // If element VT is == 32 bits, turn it into a number of shuffles.
  std::vector<SDOperand> V(NumElems);
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

    // Take advantage of the fact GR32 to VR128 scalar_to_vector (i.e. movd)
    // clears the upper bits. 
    // FIXME: we can do the same for v4f32 case when we know both parts of
    // the lower half come from scalar_to_vector (loadf32). We should do
    // that in post legalizer dag combiner with target specific hooks.
    if (MVT::isInteger(EVT) && (NonZeros & (0x3 << 2)) == 0)
      return V[0];
    MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
    MVT::ValueType EVT = MVT::getVectorBaseType(MaskVT);
    std::vector<SDOperand> MaskVec;
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

SDOperand
X86TargetLowering::LowerVECTOR_SHUFFLE(SDOperand Op, SelectionDAG &DAG) {
  SDOperand V1 = Op.getOperand(0);
  SDOperand V2 = Op.getOperand(1);
  SDOperand PermMask = Op.getOperand(2);
  MVT::ValueType VT = Op.getValueType();
  unsigned NumElems = PermMask.getNumOperands();
  bool V1IsUndef = V1.getOpcode() == ISD::UNDEF;
  bool V2IsUndef = V2.getOpcode() == ISD::UNDEF;
  bool V1IsSplat = false;
  bool V2IsSplat = false;

  if (isUndefShuffle(Op.Val))
    return DAG.getNode(ISD::UNDEF, VT);

  if (isSplatMask(PermMask.Val)) {
    if (NumElems <= 4) return Op;
    // Promote it to a v4i32 splat.
    return PromoteSplat(Op, DAG);
  }

  if (X86::isMOVLMask(PermMask.Val))
    return (V1IsUndef) ? V2 : Op;
      
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
  V1IsSplat = isSplatVector(V1.Val);
  V2IsSplat = isSplatVector(V2.Val);
  if ((V1IsSplat || V1IsUndef) && !(V2IsSplat || V2IsUndef)) {
    Op = CommuteVectorShuffle(Op, V1, V2, PermMask, DAG);
    std::swap(V1IsSplat, V2IsSplat);
    std::swap(V1IsUndef, V2IsUndef);
    Commuted = true;
  }

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
        X86::isUNPCKLMask(PermMask.Val) ||
        X86::isUNPCKHMask(PermMask.Val))
      return Op;
  }

  // If VT is integer, try PSHUF* first, then SHUFP*.
  if (MVT::isInteger(VT)) {
    if (X86::isPSHUFDMask(PermMask.Val) ||
        X86::isPSHUFHWMask(PermMask.Val) ||
        X86::isPSHUFLWMask(PermMask.Val)) {
      if (V2.getOpcode() != ISD::UNDEF)
        return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1,
                           DAG.getNode(ISD::UNDEF, V1.getValueType()),PermMask);
      return Op;
    }

    if (X86::isSHUFPMask(PermMask.Val))
      return Op;

    // Handle v8i16 shuffle high / low shuffle node pair.
    if (VT == MVT::v8i16 && isPSHUFHW_PSHUFLWMask(PermMask.Val)) {
      MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(NumElems);
      MVT::ValueType BaseVT = MVT::getVectorBaseType(MaskVT);
      std::vector<SDOperand> MaskVec;
      for (unsigned i = 0; i != 4; ++i)
        MaskVec.push_back(PermMask.getOperand(i));
      for (unsigned i = 4; i != 8; ++i)
        MaskVec.push_back(DAG.getConstant(i, BaseVT));
      SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                   &MaskVec[0], MaskVec.size());
      V1 = DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2, Mask);
      MaskVec.clear();
      for (unsigned i = 0; i != 4; ++i)
        MaskVec.push_back(DAG.getConstant(i, BaseVT));
      for (unsigned i = 4; i != 8; ++i)
        MaskVec.push_back(PermMask.getOperand(i));
      Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT, &MaskVec[0],MaskVec.size());
      return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1, V2, Mask);
    }
  } else {
    // Floating point cases in the other order.
    if (X86::isSHUFPMask(PermMask.Val))
      return Op;
    if (X86::isPSHUFDMask(PermMask.Val) ||
        X86::isPSHUFHWMask(PermMask.Val) ||
        X86::isPSHUFLWMask(PermMask.Val)) {
      if (V2.getOpcode() != ISD::UNDEF)
        return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, V1,
                           DAG.getNode(ISD::UNDEF, V1.getValueType()),PermMask);
      return Op;
    }
  }

  if (NumElems == 4) {
    MVT::ValueType MaskVT = PermMask.getValueType();
    MVT::ValueType MaskEVT = MVT::getVectorBaseType(MaskVT);
    std::vector<std::pair<int, int> > Locs;
    Locs.reserve(NumElems);
    std::vector<SDOperand> Mask1(NumElems, DAG.getNode(ISD::UNDEF, MaskEVT));
    std::vector<SDOperand> Mask2(NumElems, DAG.getNode(ISD::UNDEF, MaskEVT));
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
    std::vector<SDOperand> LoMask(NumElems, DAG.getNode(ISD::UNDEF, MaskEVT));
    std::vector<SDOperand> HiMask(NumElems, DAG.getNode(ISD::UNDEF, MaskEVT));
    std::vector<SDOperand> *MaskPtr = &LoMask;
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
    std::vector<SDOperand> MaskOps;
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
X86TargetLowering::LowerEXTRACT_VECTOR_ELT(SDOperand Op, SelectionDAG &DAG) {
  if (!isa<ConstantSDNode>(Op.getOperand(1)))
    return SDOperand();

  MVT::ValueType VT = Op.getValueType();
  // TODO: handle v16i8.
  if (MVT::getSizeInBits(VT) == 16) {
    // Transform it so it match pextrw which produces a 32-bit result.
    MVT::ValueType EVT = (MVT::ValueType)(VT+1);
    SDOperand Extract = DAG.getNode(X86ISD::PEXTRW, EVT,
                                    Op.getOperand(0), Op.getOperand(1));
    SDOperand Assert  = DAG.getNode(ISD::AssertZext, EVT, Extract,
                                    DAG.getValueType(VT));
    return DAG.getNode(ISD::TRUNCATE, VT, Assert);
  } else if (MVT::getSizeInBits(VT) == 32) {
    SDOperand Vec = Op.getOperand(0);
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
    if (Idx == 0)
      return Op;
    // SHUFPS the element to the lowest double word, then movss.
    MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(4);
    std::vector<SDOperand> IdxVec;
    IdxVec.push_back(DAG.getConstant(Idx, MVT::getVectorBaseType(MaskVT)));
    IdxVec.push_back(DAG.getNode(ISD::UNDEF, MVT::getVectorBaseType(MaskVT)));
    IdxVec.push_back(DAG.getNode(ISD::UNDEF, MVT::getVectorBaseType(MaskVT)));
    IdxVec.push_back(DAG.getNode(ISD::UNDEF, MVT::getVectorBaseType(MaskVT)));
    SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                 &IdxVec[0], IdxVec.size());
    Vec = DAG.getNode(ISD::VECTOR_SHUFFLE, Vec.getValueType(),
                      Vec, Vec, Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, VT, Vec,
                       DAG.getConstant(0, getPointerTy()));
  } else if (MVT::getSizeInBits(VT) == 64) {
    SDOperand Vec = Op.getOperand(0);
    unsigned Idx = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
    if (Idx == 0)
      return Op;

    // UNPCKHPD the element to the lowest double word, then movsd.
    // Note if the lower 64 bits of the result of the UNPCKHPD is then stored
    // to a f64mem, the whole operation is folded into a single MOVHPDmr.
    MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(4);
    std::vector<SDOperand> IdxVec;
    IdxVec.push_back(DAG.getConstant(1, MVT::getVectorBaseType(MaskVT)));
    IdxVec.push_back(DAG.getNode(ISD::UNDEF, MVT::getVectorBaseType(MaskVT)));
    SDOperand Mask = DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                 &IdxVec[0], IdxVec.size());
    Vec = DAG.getNode(ISD::VECTOR_SHUFFLE, Vec.getValueType(),
                      Vec, DAG.getNode(ISD::UNDEF, Vec.getValueType()), Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, VT, Vec,
                       DAG.getConstant(0, getPointerTy()));
  }

  return SDOperand();
}

SDOperand
X86TargetLowering::LowerINSERT_VECTOR_ELT(SDOperand Op, SelectionDAG &DAG) {
  // Transform it so it match pinsrw which expects a 16-bit value in a GR32
  // as its second argument.
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType BaseVT = MVT::getVectorBaseType(VT);
  SDOperand N0 = Op.getOperand(0);
  SDOperand N1 = Op.getOperand(1);
  SDOperand N2 = Op.getOperand(2);
  if (MVT::getSizeInBits(BaseVT) == 16) {
    if (N1.getValueType() != MVT::i32)
      N1 = DAG.getNode(ISD::ANY_EXTEND, MVT::i32, N1);
    if (N2.getValueType() != MVT::i32)
      N2 = DAG.getConstant(cast<ConstantSDNode>(N2)->getValue(), MVT::i32);
    return DAG.getNode(X86ISD::PINSRW, VT, N0, N1, N2);
  } else if (MVT::getSizeInBits(BaseVT) == 32) {
    unsigned Idx = cast<ConstantSDNode>(N2)->getValue();
    if (Idx == 0) {
      // Use a movss.
      N1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, N1);
      MVT::ValueType MaskVT = MVT::getIntVectorWithNumElements(4);
      MVT::ValueType BaseVT = MVT::getVectorBaseType(MaskVT);
      std::vector<SDOperand> MaskVec;
      MaskVec.push_back(DAG.getConstant(4, BaseVT));
      for (unsigned i = 1; i <= 3; ++i)
        MaskVec.push_back(DAG.getConstant(i, BaseVT));
      return DAG.getNode(ISD::VECTOR_SHUFFLE, VT, N0, N1,
                         DAG.getNode(ISD::BUILD_VECTOR, MaskVT,
                                     &MaskVec[0], MaskVec.size()));
    } else {
      // Use two pinsrw instructions to insert a 32 bit value.
      Idx <<= 1;
      if (MVT::isFloatingPoint(N1.getValueType())) {
        if (ISD::isNON_EXTLoad(N1.Val)) {
          // Just load directly from f32mem to GR32.
          LoadSDNode *LD = cast<LoadSDNode>(N1);
          N1 = DAG.getLoad(MVT::i32, LD->getChain(), LD->getBasePtr(),
                           LD->getSrcValue(), LD->getSrcValueOffset());
        } else {
          N1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, MVT::v4f32, N1);
          N1 = DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32, N1);
          N1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i32, N1,
                           DAG.getConstant(0, getPointerTy()));
        }
      }
      N0 = DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, N0);
      N0 = DAG.getNode(X86ISD::PINSRW, MVT::v8i16, N0, N1,
                       DAG.getConstant(Idx, getPointerTy()));
      N1 = DAG.getNode(ISD::SRL, MVT::i32, N1, DAG.getConstant(16, MVT::i8));
      N0 = DAG.getNode(X86ISD::PINSRW, MVT::v8i16, N0, N1,
                       DAG.getConstant(Idx+1, getPointerTy()));
      return DAG.getNode(ISD::BIT_CONVERT, VT, N0);
    }
  }

  return SDOperand();
}

SDOperand
X86TargetLowering::LowerSCALAR_TO_VECTOR(SDOperand Op, SelectionDAG &DAG) {
  SDOperand AnyExt = DAG.getNode(ISD::ANY_EXTEND, MVT::i32, Op.getOperand(0));
  return DAG.getNode(X86ISD::S2VEC, Op.getValueType(), AnyExt);
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
  SDOperand Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(),
                                 DAG.getTargetConstantPool(CP->getConstVal(),
                                                           getPointerTy(),
                                                           CP->getAlignment()));
  if (Subtarget->isTargetDarwin()) {
    // With PIC, the address is actually $g + Offset.
    if (!Subtarget->is64Bit() &&
        getTargetMachine().getRelocationModel() == Reloc::PIC_)
      Result = DAG.getNode(ISD::ADD, getPointerTy(),
                    DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()), Result);
  }

  return Result;
}

SDOperand
X86TargetLowering::LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG) {
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDOperand Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(),
                                 DAG.getTargetGlobalAddress(GV,
                                                            getPointerTy()));
  if (Subtarget->isTargetDarwin()) {
    // With PIC, the address is actually $g + Offset.
    if (!Subtarget->is64Bit() &&
        getTargetMachine().getRelocationModel() == Reloc::PIC_)
      Result = DAG.getNode(ISD::ADD, getPointerTy(),
                           DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()),
                           Result);

    // For Darwin, external and weak symbols are indirect, so we want to load
    // the value at address GV, not the value of GV itself. This means that
    // the GlobalAddress must be in the base or index register of the address,
    // not the GV offset field.
    if (getTargetMachine().getRelocationModel() != Reloc::Static &&
        DarwinGVRequiresExtraLoad(GV))
      Result = DAG.getLoad(getPointerTy(), DAG.getEntryNode(), Result, NULL, 0);
  } else if (Subtarget->isTargetCygwin() || Subtarget->isTargetWindows()) {
    // FIXME: What about PIC?
    if (WindowsGVRequiresExtraLoad(GV))
      Result = DAG.getLoad(getPointerTy(), DAG.getEntryNode(), Result, NULL, 0);
  }
  

  return Result;
}

SDOperand
X86TargetLowering::LowerExternalSymbol(SDOperand Op, SelectionDAG &DAG) {
  const char *Sym = cast<ExternalSymbolSDNode>(Op)->getSymbol();
  SDOperand Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(),
                                 DAG.getTargetExternalSymbol(Sym,
                                                             getPointerTy()));
  if (Subtarget->isTargetDarwin()) {
    // With PIC, the address is actually $g + Offset.
    if (!Subtarget->is64Bit() &&
        getTargetMachine().getRelocationModel() == Reloc::PIC_)
      Result = DAG.getNode(ISD::ADD, getPointerTy(),
                           DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()),
                           Result);
  }

  return Result;
}

SDOperand X86TargetLowering::LowerShift(SDOperand Op, SelectionDAG &DAG) {
    assert(Op.getNumOperands() == 3 && Op.getValueType() == MVT::i32 &&
           "Not an i64 shift!");
    bool isSRA = Op.getOpcode() == ISD::SRA_PARTS;
    SDOperand ShOpLo = Op.getOperand(0);
    SDOperand ShOpHi = Op.getOperand(1);
    SDOperand ShAmt  = Op.getOperand(2);
    SDOperand Tmp1 = isSRA ?
      DAG.getNode(ISD::SRA, MVT::i32, ShOpHi, DAG.getConstant(31, MVT::i8)) :
      DAG.getConstant(0, MVT::i32);

    SDOperand Tmp2, Tmp3;
    if (Op.getOpcode() == ISD::SHL_PARTS) {
      Tmp2 = DAG.getNode(X86ISD::SHLD, MVT::i32, ShOpHi, ShOpLo, ShAmt);
      Tmp3 = DAG.getNode(ISD::SHL, MVT::i32, ShOpLo, ShAmt);
    } else {
      Tmp2 = DAG.getNode(X86ISD::SHRD, MVT::i32, ShOpLo, ShOpHi, ShAmt);
      Tmp3 = DAG.getNode(isSRA ? ISD::SRA : ISD::SRL, MVT::i32, ShOpHi, ShAmt);
    }

    const MVT::ValueType *VTs = DAG.getNodeValueTypes(MVT::Other, MVT::Flag);
    SDOperand AndNode = DAG.getNode(ISD::AND, MVT::i8, ShAmt,
                                    DAG.getConstant(32, MVT::i8));
    SDOperand COps[]={DAG.getEntryNode(), AndNode, DAG.getConstant(0, MVT::i8)};
    SDOperand InFlag = DAG.getNode(X86ISD::CMP, VTs, 2, COps, 3).getValue(1);

    SDOperand Hi, Lo;
    SDOperand CC = DAG.getConstant(X86::COND_NE, MVT::i8);

    VTs = DAG.getNodeValueTypes(MVT::i32, MVT::Flag);
    SmallVector<SDOperand, 4> Ops;
    if (Op.getOpcode() == ISD::SHL_PARTS) {
      Ops.push_back(Tmp2);
      Ops.push_back(Tmp3);
      Ops.push_back(CC);
      Ops.push_back(InFlag);
      Hi = DAG.getNode(X86ISD::CMOV, VTs, 2, &Ops[0], Ops.size());
      InFlag = Hi.getValue(1);

      Ops.clear();
      Ops.push_back(Tmp3);
      Ops.push_back(Tmp1);
      Ops.push_back(CC);
      Ops.push_back(InFlag);
      Lo = DAG.getNode(X86ISD::CMOV, VTs, 2, &Ops[0], Ops.size());
    } else {
      Ops.push_back(Tmp2);
      Ops.push_back(Tmp3);
      Ops.push_back(CC);
      Ops.push_back(InFlag);
      Lo = DAG.getNode(X86ISD::CMOV, VTs, 2, &Ops[0], Ops.size());
      InFlag = Lo.getValue(1);

      Ops.clear();
      Ops.push_back(Tmp3);
      Ops.push_back(Tmp1);
      Ops.push_back(CC);
      Ops.push_back(InFlag);
      Hi = DAG.getNode(X86ISD::CMOV, VTs, 2, &Ops[0], Ops.size());
    }

    VTs = DAG.getNodeValueTypes(MVT::i32, MVT::i32);
    Ops.clear();
    Ops.push_back(Lo);
    Ops.push_back(Hi);
    return DAG.getNode(ISD::MERGE_VALUES, VTs, 2, &Ops[0], Ops.size());
}

SDOperand X86TargetLowering::LowerSINT_TO_FP(SDOperand Op, SelectionDAG &DAG) {
  assert(Op.getOperand(0).getValueType() <= MVT::i64 &&
         Op.getOperand(0).getValueType() >= MVT::i16 &&
         "Unknown SINT_TO_FP to lower!");

  SDOperand Result;
  MVT::ValueType SrcVT = Op.getOperand(0).getValueType();
  unsigned Size = MVT::getSizeInBits(SrcVT)/8;
  MachineFunction &MF = DAG.getMachineFunction();
  int SSFI = MF.getFrameInfo()->CreateStackObject(Size, Size);
  SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
  SDOperand Chain = DAG.getStore(DAG.getEntryNode(), Op.getOperand(0),
                                 StackSlot, NULL, 0);

  // Build the FILD
  std::vector<MVT::ValueType> Tys;
  Tys.push_back(MVT::f64);
  Tys.push_back(MVT::Other);
  if (X86ScalarSSE) Tys.push_back(MVT::Flag);
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(StackSlot);
  Ops.push_back(DAG.getValueType(SrcVT));
  Result = DAG.getNode(X86ScalarSSE ? X86ISD::FILD_FLAG :X86ISD::FILD,
                       Tys, &Ops[0], Ops.size());

  if (X86ScalarSSE) {
    Chain = Result.getValue(1);
    SDOperand InFlag = Result.getValue(2);

    // FIXME: Currently the FST is flagged to the FILD_FLAG. This
    // shouldn't be necessary except that RFP cannot be live across
    // multiple blocks. When stackifier is fixed, they can be uncoupled.
    MachineFunction &MF = DAG.getMachineFunction();
    int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8);
    SDOperand StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::Other);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(Result);
    Ops.push_back(StackSlot);
    Ops.push_back(DAG.getValueType(Op.getValueType()));
    Ops.push_back(InFlag);
    Chain = DAG.getNode(X86ISD::FST, Tys, &Ops[0], Ops.size());
    Result = DAG.getLoad(Op.getValueType(), Chain, StackSlot, NULL, 0);
  }

  return Result;
}

SDOperand X86TargetLowering::LowerFP_TO_SINT(SDOperand Op, SelectionDAG &DAG) {
  assert(Op.getValueType() <= MVT::i64 && Op.getValueType() >= MVT::i16 &&
         "Unknown FP_TO_SINT to lower!");
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
  if (X86ScalarSSE) {
    assert(Op.getValueType() == MVT::i64 && "Invalid FP_TO_SINT to lower!");
    Chain = DAG.getStore(Chain, Value, StackSlot, NULL, 0);
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::f64);
    Tys.push_back(MVT::Other);
    std::vector<SDOperand> Ops;
    Ops.push_back(Chain);
    Ops.push_back(StackSlot);
    Ops.push_back(DAG.getValueType(Op.getOperand(0).getValueType()));
    Value = DAG.getNode(X86ISD::FLD, Tys, &Ops[0], Ops.size());
    Chain = Value.getValue(1);
    SSFI = MF.getFrameInfo()->CreateStackObject(MemSize, MemSize);
    StackSlot = DAG.getFrameIndex(SSFI, getPointerTy());
  }

  // Build the FP_TO_INT*_IN_MEM
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Value);
  Ops.push_back(StackSlot);
  SDOperand FIST = DAG.getNode(Opc, MVT::Other, &Ops[0], Ops.size());

  // Load the result.
  return DAG.getLoad(Op.getValueType(), FIST, StackSlot, NULL, 0);
}

SDOperand X86TargetLowering::LowerFABS(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  const Type *OpNTy =  MVT::getTypeForValueType(VT);
  std::vector<Constant*> CV;
  if (VT == MVT::f64) {
    CV.push_back(ConstantFP::get(OpNTy, BitsToDouble(~(1ULL << 63))));
    CV.push_back(ConstantFP::get(OpNTy, 0.0));
  } else {
    CV.push_back(ConstantFP::get(OpNTy, BitsToFloat(~(1U << 31))));
    CV.push_back(ConstantFP::get(OpNTy, 0.0));
    CV.push_back(ConstantFP::get(OpNTy, 0.0));
    CV.push_back(ConstantFP::get(OpNTy, 0.0));
  }
  Constant *CS = ConstantStruct::get(CV);
  SDOperand CPIdx = DAG.getConstantPool(CS, getPointerTy(), 4);
  std::vector<MVT::ValueType> Tys;
  Tys.push_back(VT);
  Tys.push_back(MVT::Other);
  SmallVector<SDOperand, 3> Ops;
  Ops.push_back(DAG.getEntryNode());
  Ops.push_back(CPIdx);
  Ops.push_back(DAG.getSrcValue(NULL));
  SDOperand Mask = DAG.getNode(X86ISD::LOAD_PACK, Tys, &Ops[0], Ops.size());
  return DAG.getNode(X86ISD::FAND, VT, Op.getOperand(0), Mask);
}

SDOperand X86TargetLowering::LowerFNEG(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  const Type *OpNTy =  MVT::getTypeForValueType(VT);
  std::vector<Constant*> CV;
  if (VT == MVT::f64) {
    CV.push_back(ConstantFP::get(OpNTy, BitsToDouble(1ULL << 63)));
    CV.push_back(ConstantFP::get(OpNTy, 0.0));
  } else {
    CV.push_back(ConstantFP::get(OpNTy, BitsToFloat(1U << 31)));
    CV.push_back(ConstantFP::get(OpNTy, 0.0));
    CV.push_back(ConstantFP::get(OpNTy, 0.0));
    CV.push_back(ConstantFP::get(OpNTy, 0.0));
  }
  Constant *CS = ConstantStruct::get(CV);
  SDOperand CPIdx = DAG.getConstantPool(CS, getPointerTy(), 4);
  std::vector<MVT::ValueType> Tys;
  Tys.push_back(VT);
  Tys.push_back(MVT::Other);
  SmallVector<SDOperand, 3> Ops;
  Ops.push_back(DAG.getEntryNode());
  Ops.push_back(CPIdx);
  Ops.push_back(DAG.getSrcValue(NULL));
  SDOperand Mask = DAG.getNode(X86ISD::LOAD_PACK, Tys, &Ops[0], Ops.size());
  return DAG.getNode(X86ISD::FXOR, VT, Op.getOperand(0), Mask);
}

SDOperand X86TargetLowering::LowerSETCC(SDOperand Op, SelectionDAG &DAG,
                                        SDOperand Chain) {
  assert(Op.getValueType() == MVT::i8 && "SetCC type must be 8-bit integer");
  SDOperand Cond;
  SDOperand Op0 = Op.getOperand(0);
  SDOperand Op1 = Op.getOperand(1);
  SDOperand CC = Op.getOperand(2);
  ISD::CondCode SetCCOpcode = cast<CondCodeSDNode>(CC)->get();
  const MVT::ValueType *VTs1 = DAG.getNodeValueTypes(MVT::Other, MVT::Flag);
  const MVT::ValueType *VTs2 = DAG.getNodeValueTypes(MVT::i8, MVT::Flag);
  bool isFP = MVT::isFloatingPoint(Op.getOperand(1).getValueType());
  unsigned X86CC;

  if (translateX86CC(cast<CondCodeSDNode>(CC)->get(), isFP, X86CC, 
                     Op0, Op1, DAG)) {
    SDOperand Ops1[] = { Chain, Op0, Op1 };
    Cond = DAG.getNode(X86ISD::CMP, VTs1, 2, Ops1, 3).getValue(1);
    SDOperand Ops2[] = { DAG.getConstant(X86CC, MVT::i8), Cond };
    return DAG.getNode(X86ISD::SETCC, VTs2, 2, Ops2, 2);
  }

  assert(isFP && "Illegal integer SetCC!");

  SDOperand COps[] = { Chain, Op0, Op1 };
  Cond = DAG.getNode(X86ISD::CMP, VTs1, 2, COps, 3).getValue(1);

  switch (SetCCOpcode) {
  default: assert(false && "Illegal floating point SetCC!");
  case ISD::SETOEQ: {  // !PF & ZF
    SDOperand Ops1[] = { DAG.getConstant(X86::COND_NP, MVT::i8), Cond };
    SDOperand Tmp1 = DAG.getNode(X86ISD::SETCC, VTs2, 2, Ops1, 2);
    SDOperand Ops2[] = { DAG.getConstant(X86::COND_E, MVT::i8),
                         Tmp1.getValue(1) };
    SDOperand Tmp2 = DAG.getNode(X86ISD::SETCC, VTs2, 2, Ops2, 2);
    return DAG.getNode(ISD::AND, MVT::i8, Tmp1, Tmp2);
  }
  case ISD::SETUNE: {  // PF | !ZF
    SDOperand Ops1[] = { DAG.getConstant(X86::COND_P, MVT::i8), Cond };
    SDOperand Tmp1 = DAG.getNode(X86ISD::SETCC, VTs2, 2, Ops1, 2);
    SDOperand Ops2[] = { DAG.getConstant(X86::COND_NE, MVT::i8),
                         Tmp1.getValue(1) };
    SDOperand Tmp2 = DAG.getNode(X86ISD::SETCC, VTs2, 2, Ops2, 2);
    return DAG.getNode(ISD::OR, MVT::i8, Tmp1, Tmp2);
  }
  }
}

SDOperand X86TargetLowering::LowerSELECT(SDOperand Op, SelectionDAG &DAG) {
  bool addTest = true;
  SDOperand Chain = DAG.getEntryNode();
  SDOperand Cond  = Op.getOperand(0);
  SDOperand CC;
  const MVT::ValueType *VTs = DAG.getNodeValueTypes(MVT::Other, MVT::Flag);

  if (Cond.getOpcode() == ISD::SETCC)
    Cond = LowerSETCC(Cond, DAG, Chain);

  if (Cond.getOpcode() == X86ISD::SETCC) {
    CC = Cond.getOperand(0);

    // If condition flag is set by a X86ISD::CMP, then make a copy of it
    // (since flag operand cannot be shared). Use it as the condition setting
    // operand in place of the X86ISD::SETCC.
    // If the X86ISD::SETCC has more than one use, then perhaps it's better
    // to use a test instead of duplicating the X86ISD::CMP (for register
    // pressure reason)?
    SDOperand Cmp = Cond.getOperand(1);
    unsigned Opc = Cmp.getOpcode();
    bool IllegalFPCMov = !X86ScalarSSE &&
      MVT::isFloatingPoint(Op.getValueType()) &&
      !hasFPCMov(cast<ConstantSDNode>(CC)->getSignExtended());
    if ((Opc == X86ISD::CMP || Opc == X86ISD::COMI || Opc == X86ISD::UCOMI) &&
        !IllegalFPCMov) {
      SDOperand Ops[] = { Chain, Cmp.getOperand(1), Cmp.getOperand(2) };
      Cond = DAG.getNode(Opc, VTs, 2, Ops, 3);
      addTest = false;
    }
  }

  if (addTest) {
    CC = DAG.getConstant(X86::COND_NE, MVT::i8);
    SDOperand Ops[] = { Chain, Cond, DAG.getConstant(0, MVT::i8) };
    Cond = DAG.getNode(X86ISD::CMP, VTs, 2, Ops, 3);
  }

  VTs = DAG.getNodeValueTypes(Op.getValueType(), MVT::Flag);
  SmallVector<SDOperand, 4> Ops;
  // X86ISD::CMOV means set the result (which is operand 1) to the RHS if
  // condition is true.
  Ops.push_back(Op.getOperand(2));
  Ops.push_back(Op.getOperand(1));
  Ops.push_back(CC);
  Ops.push_back(Cond.getValue(1));
  return DAG.getNode(X86ISD::CMOV, VTs, 2, &Ops[0], Ops.size());
}

SDOperand X86TargetLowering::LowerBRCOND(SDOperand Op, SelectionDAG &DAG) {
  bool addTest = true;
  SDOperand Chain = Op.getOperand(0);
  SDOperand Cond  = Op.getOperand(1);
  SDOperand Dest  = Op.getOperand(2);
  SDOperand CC;
  const MVT::ValueType *VTs = DAG.getNodeValueTypes(MVT::Other, MVT::Flag);

  if (Cond.getOpcode() == ISD::SETCC)
    Cond = LowerSETCC(Cond, DAG, Chain);

  if (Cond.getOpcode() == X86ISD::SETCC) {
    CC = Cond.getOperand(0);

    // If condition flag is set by a X86ISD::CMP, then make a copy of it
    // (since flag operand cannot be shared). Use it as the condition setting
    // operand in place of the X86ISD::SETCC.
    // If the X86ISD::SETCC has more than one use, then perhaps it's better
    // to use a test instead of duplicating the X86ISD::CMP (for register
    // pressure reason)?
    SDOperand Cmp = Cond.getOperand(1);
    unsigned Opc = Cmp.getOpcode();
    if (Opc == X86ISD::CMP || Opc == X86ISD::COMI || Opc == X86ISD::UCOMI) {
      SDOperand Ops[] = { Chain, Cmp.getOperand(1), Cmp.getOperand(2) };
      Cond = DAG.getNode(Opc, VTs, 2, Ops, 3);
      addTest = false;
    }
  }

  if (addTest) {
    CC = DAG.getConstant(X86::COND_NE, MVT::i8);
    SDOperand Ops[] = { Chain, Cond, DAG.getConstant(0, MVT::i8) };
    Cond = DAG.getNode(X86ISD::CMP, VTs, 2, Ops, 3);
  }
  return DAG.getNode(X86ISD::BRCOND, Op.getValueType(),
                     Cond, Op.getOperand(2), CC, Cond.getValue(1));
}

SDOperand X86TargetLowering::LowerJumpTable(SDOperand Op, SelectionDAG &DAG) {
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDOperand Result = DAG.getNode(X86ISD::Wrapper, getPointerTy(),
                                 DAG.getTargetJumpTable(JT->getIndex(),
                                                        getPointerTy()));
  if (Subtarget->isTargetDarwin()) {
    // With PIC, the address is actually $g + Offset.
    if (!Subtarget->is64Bit() &&
        getTargetMachine().getRelocationModel() == Reloc::PIC_)
      Result = DAG.getNode(ISD::ADD, getPointerTy(),
                           DAG.getNode(X86ISD::GlobalBaseReg, getPointerTy()),
                           Result);    
  }

  return Result;
}

SDOperand X86TargetLowering::LowerCALL(SDOperand Op, SelectionDAG &DAG) {
  unsigned CallingConv= cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  
  if (Subtarget->is64Bit())
    return LowerX86_64CCCCallTo(Op, DAG);
  else
    switch (CallingConv) {
    default:
      assert(0 && "Unsupported calling convention");      
    case CallingConv::Fast:
      if (EnableFastCC) {
        return LowerFastCCCallTo(Op, DAG, false);
      }
      // Falls through
    case CallingConv::C:
    case CallingConv::CSRet:
      return LowerCCCCallTo(Op, DAG);
    case CallingConv::X86_StdCall: 
      return LowerStdCallCCCallTo(Op, DAG);
    case CallingConv::X86_FastCall:
      return LowerFastCCCallTo(Op, DAG, true);
    }
}

SDOperand X86TargetLowering::LowerRET(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Copy;
    
  switch(Op.getNumOperands()) {
    default:
      assert(0 && "Do not know how to return this many arguments!");
      abort();
    case 1:    // ret void.
      return DAG.getNode(X86ISD::RET_FLAG, MVT::Other, Op.getOperand(0),
                        DAG.getConstant(getBytesToPopOnReturn(), MVT::i16));
    case 3: {
      MVT::ValueType ArgVT = Op.getOperand(1).getValueType();
      
      if (MVT::isVector(ArgVT) ||
          (Subtarget->is64Bit() && MVT::isFloatingPoint(ArgVT))) {
        // Integer or FP vector result -> XMM0.
        if (DAG.getMachineFunction().liveout_empty())
          DAG.getMachineFunction().addLiveOut(X86::XMM0);
        Copy = DAG.getCopyToReg(Op.getOperand(0), X86::XMM0, Op.getOperand(1),
                                SDOperand());
      } else if (MVT::isInteger(ArgVT)) {
        // Integer result -> EAX / RAX.
        // The C calling convention guarantees the return value has been
        // promoted to at least MVT::i32. The X86-64 ABI doesn't require the
        // value to be promoted MVT::i64. So we don't have to extend it to
        // 64-bit. Return the value in EAX, but mark RAX as liveout.
        unsigned Reg = Subtarget->is64Bit() ? X86::RAX : X86::EAX;
        if (DAG.getMachineFunction().liveout_empty())
          DAG.getMachineFunction().addLiveOut(Reg);

        Reg = (ArgVT == MVT::i64) ? X86::RAX : X86::EAX;
        Copy = DAG.getCopyToReg(Op.getOperand(0), Reg, Op.getOperand(1),
                                SDOperand());
      } else if (!X86ScalarSSE) {
        // FP return with fp-stack value.
        if (DAG.getMachineFunction().liveout_empty())
          DAG.getMachineFunction().addLiveOut(X86::ST0);

        std::vector<MVT::ValueType> Tys;
        Tys.push_back(MVT::Other);
        Tys.push_back(MVT::Flag);
        std::vector<SDOperand> Ops;
        Ops.push_back(Op.getOperand(0));
        Ops.push_back(Op.getOperand(1));
        Copy = DAG.getNode(X86ISD::FP_SET_RESULT, Tys, &Ops[0], Ops.size());
      } else {
        // FP return with ScalarSSE (return on fp-stack).
        if (DAG.getMachineFunction().liveout_empty())
          DAG.getMachineFunction().addLiveOut(X86::ST0);

        SDOperand MemLoc;
        SDOperand Chain = Op.getOperand(0);
        SDOperand Value = Op.getOperand(1);

        if (ISD::isNON_EXTLoad(Value.Val) &&
            (Chain == Value.getValue(1) || Chain == Value.getOperand(0))) {
          Chain  = Value.getOperand(0);
          MemLoc = Value.getOperand(1);
        } else {
          // Spill the value to memory and reload it into top of stack.
          unsigned Size = MVT::getSizeInBits(ArgVT)/8;
          MachineFunction &MF = DAG.getMachineFunction();
          int SSFI = MF.getFrameInfo()->CreateStackObject(Size, Size);
          MemLoc = DAG.getFrameIndex(SSFI, getPointerTy());
          Chain = DAG.getStore(Op.getOperand(0), Value, MemLoc, NULL, 0);
        }
        std::vector<MVT::ValueType> Tys;
        Tys.push_back(MVT::f64);
        Tys.push_back(MVT::Other);
        std::vector<SDOperand> Ops;
        Ops.push_back(Chain);
        Ops.push_back(MemLoc);
        Ops.push_back(DAG.getValueType(ArgVT));
        Copy = DAG.getNode(X86ISD::FLD, Tys, &Ops[0], Ops.size());
        Tys.clear();
        Tys.push_back(MVT::Other);
        Tys.push_back(MVT::Flag);
        Ops.clear();
        Ops.push_back(Copy.getValue(1));
        Ops.push_back(Copy);
        Copy = DAG.getNode(X86ISD::FP_SET_RESULT, Tys, &Ops[0], Ops.size());
      }
      break;
    }
    case 5: {
      unsigned Reg1 = Subtarget->is64Bit() ? X86::RAX : X86::EAX;
      unsigned Reg2 = Subtarget->is64Bit() ? X86::RDX : X86::EDX;
      if (DAG.getMachineFunction().liveout_empty()) {
        DAG.getMachineFunction().addLiveOut(Reg1);
        DAG.getMachineFunction().addLiveOut(Reg2);
      }

      Copy = DAG.getCopyToReg(Op.getOperand(0), Reg2, Op.getOperand(3), 
                              SDOperand());
      Copy = DAG.getCopyToReg(Copy, Reg1, Op.getOperand(1), Copy.getValue(1));
      break;
    }
  }
  return DAG.getNode(X86ISD::RET_FLAG, MVT::Other,
                     Copy, DAG.getConstant(getBytesToPopOnReturn(), MVT::i16),
                     Copy.getValue(1));
}

SDOperand
X86TargetLowering::LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  const Function* Fn = MF.getFunction();
  if (Fn->hasExternalLinkage() &&
      Subtarget->isTargetCygwin() &&
      Fn->getName() == "main")
    MF.getInfo<X86FunctionInfo>()->setForceFramePointer(true);

  unsigned CC = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  if (Subtarget->is64Bit())
    return LowerX86_64CCCArguments(Op, DAG);
  else
    switch(CC) {
    default:
      assert(0 && "Unsupported calling convention");
    case CallingConv::Fast:
      if (EnableFastCC) {
        return LowerFastCCArguments(Op, DAG);
      }
      // Falls through
    case CallingConv::C:
    case CallingConv::CSRet:
      return LowerCCCArguments(Op, DAG);
    case CallingConv::X86_StdCall:
      MF.getInfo<X86FunctionInfo>()->setDecorationStyle(StdCall);
      return LowerStdCallCCArguments(Op, DAG);
    case CallingConv::X86_FastCall:
      MF.getInfo<X86FunctionInfo>()->setDecorationStyle(FastCall);
      return LowerFastCallCCArguments(Op, DAG);
    }
}

SDOperand X86TargetLowering::LowerMEMSET(SDOperand Op, SelectionDAG &DAG) {
  SDOperand InFlag(0, 0);
  SDOperand Chain = Op.getOperand(0);
  unsigned Align =
    (unsigned)cast<ConstantSDNode>(Op.getOperand(4))->getValue();
  if (Align == 0) Align = 1;

  ConstantSDNode *I = dyn_cast<ConstantSDNode>(Op.getOperand(3));
  // If not DWORD aligned, call memset if size is less than the threshold.
  // It knows how to align to the right boundary first.
  if ((Align & 3) != 0 ||
      (I && I->getValue() < Subtarget->getMinRepStrSizeThreshold())) {
    MVT::ValueType IntPtr = getPointerTy();
    const Type *IntPtrTy = getTargetData()->getIntPtrType();
    std::vector<std::pair<SDOperand, const Type*> > Args;
    Args.push_back(std::make_pair(Op.getOperand(1), IntPtrTy));
    // Extend the ubyte argument to be an int value for the call.
    SDOperand Val = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Op.getOperand(2));
    Args.push_back(std::make_pair(Val, IntPtrTy));
    Args.push_back(std::make_pair(Op.getOperand(3), IntPtrTy));
    std::pair<SDOperand,SDOperand> CallResult =
      LowerCallTo(Chain, Type::VoidTy, false, CallingConv::C, false,
                  DAG.getExternalSymbol("memset", IntPtr), Args, DAG);
    return CallResult.second;
  }

  MVT::ValueType AVT;
  SDOperand Count;
  ConstantSDNode *ValC = dyn_cast<ConstantSDNode>(Op.getOperand(2));
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
        if (Subtarget->is64Bit() && ((Align & 0xF) == 0)) {  // QWORD aligned
          AVT = MVT::i64;
          ValReg = X86::RAX;
          Val = (Val << 32) | Val;
        }
        break;
      default:  // Byte aligned
        AVT = MVT::i8;
        ValReg = X86::AL;
        Count = Op.getOperand(3);
        break;
    }

    if (AVT > MVT::i8) {
      if (I) {
        unsigned UBytes = MVT::getSizeInBits(AVT) / 8;
        Count = DAG.getConstant(I->getValue() / UBytes, getPointerTy());
        BytesLeft = I->getValue() % UBytes;
      } else {
        assert(AVT >= MVT::i32 &&
               "Do not use rep;stos if not at least DWORD aligned");
        Count = DAG.getNode(ISD::SRL, Op.getOperand(3).getValueType(),
                            Op.getOperand(3), DAG.getConstant(2, MVT::i8));
        TwoRepStos = true;
      }
    }

    Chain  = DAG.getCopyToReg(Chain, ValReg, DAG.getConstant(Val, AVT),
                              InFlag);
    InFlag = Chain.getValue(1);
  } else {
    AVT = MVT::i8;
    Count  = Op.getOperand(3);
    Chain  = DAG.getCopyToReg(Chain, X86::AL, Op.getOperand(2), InFlag);
    InFlag = Chain.getValue(1);
  }

  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RCX : X86::ECX,
                            Count, InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RDI : X86::EDI,
                            Op.getOperand(1), InFlag);
  InFlag = Chain.getValue(1);

  std::vector<MVT::ValueType> Tys;
  Tys.push_back(MVT::Other);
  Tys.push_back(MVT::Flag);
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(DAG.getValueType(AVT));
  Ops.push_back(InFlag);
  Chain  = DAG.getNode(X86ISD::REP_STOS, Tys, &Ops[0], Ops.size());

  if (TwoRepStos) {
    InFlag = Chain.getValue(1);
    Count = Op.getOperand(3);
    MVT::ValueType CVT = Count.getValueType();
    SDOperand Left = DAG.getNode(ISD::AND, CVT, Count,
                               DAG.getConstant((AVT == MVT::i64) ? 7 : 3, CVT));
    Chain  = DAG.getCopyToReg(Chain, (CVT == MVT::i64) ? X86::RCX : X86::ECX,
                              Left, InFlag);
    InFlag = Chain.getValue(1);
    Tys.clear();
    Tys.push_back(MVT::Other);
    Tys.push_back(MVT::Flag);
    Ops.clear();
    Ops.push_back(Chain);
    Ops.push_back(DAG.getValueType(MVT::i8));
    Ops.push_back(InFlag);
    Chain  = DAG.getNode(X86ISD::REP_STOS, Tys, &Ops[0], Ops.size());
  } else if (BytesLeft) {
    // Issue stores for the last 1 - 7 bytes.
    SDOperand Value;
    unsigned Val = ValC->getValue() & 255;
    unsigned Offset = I->getValue() - BytesLeft;
    SDOperand DstAddr = Op.getOperand(1);
    MVT::ValueType AddrVT = DstAddr.getValueType();
    if (BytesLeft >= 4) {
      Val = (Val << 8)  | Val;
      Val = (Val << 16) | Val;
      Value = DAG.getConstant(Val, MVT::i32);
      Chain = DAG.getStore(Chain, Value,
                           DAG.getNode(ISD::ADD, AddrVT, DstAddr,
                                       DAG.getConstant(Offset, AddrVT)),
                           NULL, 0);
      BytesLeft -= 4;
      Offset += 4;
    }
    if (BytesLeft >= 2) {
      Value = DAG.getConstant((Val << 8) | Val, MVT::i16);
      Chain = DAG.getStore(Chain, Value,
                           DAG.getNode(ISD::ADD, AddrVT, DstAddr,
                                       DAG.getConstant(Offset, AddrVT)),
                           NULL, 0);
      BytesLeft -= 2;
      Offset += 2;
    }
    if (BytesLeft == 1) {
      Value = DAG.getConstant(Val, MVT::i8);
      Chain = DAG.getStore(Chain, Value,
                           DAG.getNode(ISD::ADD, AddrVT, DstAddr,
                                       DAG.getConstant(Offset, AddrVT)),
                           NULL, 0);
    }
  }

  return Chain;
}

SDOperand X86TargetLowering::LowerMEMCPY(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain = Op.getOperand(0);
  unsigned Align =
    (unsigned)cast<ConstantSDNode>(Op.getOperand(4))->getValue();
  if (Align == 0) Align = 1;

  ConstantSDNode *I = dyn_cast<ConstantSDNode>(Op.getOperand(3));
  // If not DWORD aligned, call memcpy if size is less than the threshold.
  // It knows how to align to the right boundary first.
  if ((Align & 3) != 0 ||
      (I && I->getValue() < Subtarget->getMinRepStrSizeThreshold())) {
    MVT::ValueType IntPtr = getPointerTy();
    const Type *IntPtrTy = getTargetData()->getIntPtrType();
    std::vector<std::pair<SDOperand, const Type*> > Args;
    Args.push_back(std::make_pair(Op.getOperand(1), IntPtrTy));
    Args.push_back(std::make_pair(Op.getOperand(2), IntPtrTy));
    Args.push_back(std::make_pair(Op.getOperand(3), IntPtrTy));
    std::pair<SDOperand,SDOperand> CallResult =
      LowerCallTo(Chain, Type::VoidTy, false, CallingConv::C, false,
                  DAG.getExternalSymbol("memcpy", IntPtr), Args, DAG);
    return CallResult.second;
  }

  MVT::ValueType AVT;
  SDOperand Count;
  unsigned BytesLeft = 0;
  bool TwoRepMovs = false;
  switch (Align & 3) {
    case 2:   // WORD aligned
      AVT = MVT::i16;
      break;
    case 0:  // DWORD aligned
      AVT = MVT::i32;
      if (Subtarget->is64Bit() && ((Align & 0xF) == 0))  // QWORD aligned
        AVT = MVT::i64;
      break;
    default:  // Byte aligned
      AVT = MVT::i8;
      Count = Op.getOperand(3);
      break;
  }

  if (AVT > MVT::i8) {
    if (I) {
      unsigned UBytes = MVT::getSizeInBits(AVT) / 8;
      Count = DAG.getConstant(I->getValue() / UBytes, getPointerTy());
      BytesLeft = I->getValue() % UBytes;
    } else {
      assert(AVT >= MVT::i32 &&
             "Do not use rep;movs if not at least DWORD aligned");
      Count = DAG.getNode(ISD::SRL, Op.getOperand(3).getValueType(),
                          Op.getOperand(3), DAG.getConstant(2, MVT::i8));
      TwoRepMovs = true;
    }
  }

  SDOperand InFlag(0, 0);
  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RCX : X86::ECX,
                            Count, InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RDI : X86::EDI,
                            Op.getOperand(1), InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, Subtarget->is64Bit() ? X86::RSI : X86::ESI,
                            Op.getOperand(2), InFlag);
  InFlag = Chain.getValue(1);

  std::vector<MVT::ValueType> Tys;
  Tys.push_back(MVT::Other);
  Tys.push_back(MVT::Flag);
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(DAG.getValueType(AVT));
  Ops.push_back(InFlag);
  Chain = DAG.getNode(X86ISD::REP_MOVS, Tys, &Ops[0], Ops.size());

  if (TwoRepMovs) {
    InFlag = Chain.getValue(1);
    Count = Op.getOperand(3);
    MVT::ValueType CVT = Count.getValueType();
    SDOperand Left = DAG.getNode(ISD::AND, CVT, Count,
                               DAG.getConstant((AVT == MVT::i64) ? 7 : 3, CVT));
    Chain  = DAG.getCopyToReg(Chain, (CVT == MVT::i64) ? X86::RCX : X86::ECX,
                              Left, InFlag);
    InFlag = Chain.getValue(1);
    Tys.clear();
    Tys.push_back(MVT::Other);
    Tys.push_back(MVT::Flag);
    Ops.clear();
    Ops.push_back(Chain);
    Ops.push_back(DAG.getValueType(MVT::i8));
    Ops.push_back(InFlag);
    Chain = DAG.getNode(X86ISD::REP_MOVS, Tys, &Ops[0], Ops.size());
  } else if (BytesLeft) {
    // Issue loads and stores for the last 1 - 7 bytes.
    unsigned Offset = I->getValue() - BytesLeft;
    SDOperand DstAddr = Op.getOperand(1);
    MVT::ValueType DstVT = DstAddr.getValueType();
    SDOperand SrcAddr = Op.getOperand(2);
    MVT::ValueType SrcVT = SrcAddr.getValueType();
    SDOperand Value;
    if (BytesLeft >= 4) {
      Value = DAG.getLoad(MVT::i32, Chain,
                          DAG.getNode(ISD::ADD, SrcVT, SrcAddr,
                                      DAG.getConstant(Offset, SrcVT)),
                          NULL, 0);
      Chain = Value.getValue(1);
      Chain = DAG.getStore(Chain, Value,
                           DAG.getNode(ISD::ADD, DstVT, DstAddr,
                                       DAG.getConstant(Offset, DstVT)),
                           NULL, 0);
      BytesLeft -= 4;
      Offset += 4;
    }
    if (BytesLeft >= 2) {
      Value = DAG.getLoad(MVT::i16, Chain,
                          DAG.getNode(ISD::ADD, SrcVT, SrcAddr,
                                      DAG.getConstant(Offset, SrcVT)),
                          NULL, 0);
      Chain = Value.getValue(1);
      Chain = DAG.getStore(Chain, Value,
                           DAG.getNode(ISD::ADD, DstVT, DstAddr,
                                       DAG.getConstant(Offset, DstVT)),
                           NULL, 0);
      BytesLeft -= 2;
      Offset += 2;
    }

    if (BytesLeft == 1) {
      Value = DAG.getLoad(MVT::i8, Chain,
                          DAG.getNode(ISD::ADD, SrcVT, SrcAddr,
                                      DAG.getConstant(Offset, SrcVT)),
                          NULL, 0);
      Chain = Value.getValue(1);
      Chain = DAG.getStore(Chain, Value,
                           DAG.getNode(ISD::ADD, DstVT, DstAddr,
                                       DAG.getConstant(Offset, DstVT)),
                           NULL, 0);
    }
  }

  return Chain;
}

SDOperand
X86TargetLowering::LowerREADCYCLCECOUNTER(SDOperand Op, SelectionDAG &DAG) {
  std::vector<MVT::ValueType> Tys;
  Tys.push_back(MVT::Other);
  Tys.push_back(MVT::Flag);
  std::vector<SDOperand> Ops;
  Ops.push_back(Op.getOperand(0));
  SDOperand rd = DAG.getNode(X86ISD::RDTSC_DAG, Tys, &Ops[0], Ops.size());
  Ops.clear();
  Ops.push_back(DAG.getCopyFromReg(rd, X86::EAX, MVT::i32, rd.getValue(1)));
  Ops.push_back(DAG.getCopyFromReg(Ops[0].getValue(1), X86::EDX, 
                                   MVT::i32, Ops[0].getValue(2)));
  Ops.push_back(Ops[1].getValue(1));
  Tys[0] = Tys[1] = MVT::i32;
  Tys.push_back(MVT::Other);
  return DAG.getNode(ISD::MERGE_VALUES, Tys, &Ops[0], Ops.size());
}

SDOperand X86TargetLowering::LowerVASTART(SDOperand Op, SelectionDAG &DAG) {
  SrcValueSDNode *SV = cast<SrcValueSDNode>(Op.getOperand(2));

  if (!Subtarget->is64Bit()) {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, getPointerTy());
    return DAG.getStore(Op.getOperand(0), FR,Op.getOperand(1), SV->getValue(),
                        SV->getOffset());
  }

  // __va_list_tag:
  //   gp_offset         (0 - 6 * 8)
  //   fp_offset         (48 - 48 + 8 * 16)
  //   overflow_arg_area (point to parameters coming in memory).
  //   reg_save_area
  std::vector<SDOperand> MemOps;
  SDOperand FIN = Op.getOperand(1);
  // Store gp_offset
  SDOperand Store = DAG.getStore(Op.getOperand(0),
                                 DAG.getConstant(VarArgsGPOffset, MVT::i32),
                                 FIN, SV->getValue(), SV->getOffset());
  MemOps.push_back(Store);

  // Store fp_offset
  FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                    DAG.getConstant(4, getPointerTy()));
  Store = DAG.getStore(Op.getOperand(0),
                       DAG.getConstant(VarArgsFPOffset, MVT::i32),
                       FIN, SV->getValue(), SV->getOffset());
  MemOps.push_back(Store);

  // Store ptr to overflow_arg_area
  FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                    DAG.getConstant(4, getPointerTy()));
  SDOperand OVFIN = DAG.getFrameIndex(VarArgsFrameIndex, getPointerTy());
  Store = DAG.getStore(Op.getOperand(0), OVFIN, FIN, SV->getValue(),
                       SV->getOffset());
  MemOps.push_back(Store);

  // Store ptr to reg_save_area.
  FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                    DAG.getConstant(8, getPointerTy()));
  SDOperand RSFIN = DAG.getFrameIndex(RegSaveFrameIndex, getPointerTy());
  Store = DAG.getStore(Op.getOperand(0), RSFIN, FIN, SV->getValue(),
                       SV->getOffset());
  MemOps.push_back(Store);
  return DAG.getNode(ISD::TokenFactor, MVT::Other, &MemOps[0], MemOps.size());
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

    const MVT::ValueType *VTs = DAG.getNodeValueTypes(MVT::Other, MVT::Flag);
    SDOperand Ops1[] = { DAG.getEntryNode(), LHS, RHS };
    SDOperand Cond = DAG.getNode(Opc, VTs, 2, Ops1, 3);
    VTs = DAG.getNodeValueTypes(MVT::i8, MVT::Flag);
    SDOperand Ops2[] = { DAG.getConstant(X86CC, MVT::i8), Cond };
    SDOperand SetCC = DAG.getNode(X86ISD::SETCC, VTs, 2, Ops2, 2);
    return DAG.getNode(ISD::ANY_EXTEND, MVT::i32, SetCC);
  }
  }
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand X86TargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  case ISD::BUILD_VECTOR:       return LowerBUILD_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:     return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT: return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:  return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::SCALAR_TO_VECTOR:   return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::ExternalSymbol:     return LowerExternalSymbol(Op, DAG);
  case ISD::SHL_PARTS:
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:          return LowerShift(Op, DAG);
  case ISD::SINT_TO_FP:         return LowerSINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:         return LowerFP_TO_SINT(Op, DAG);
  case ISD::FABS:               return LowerFABS(Op, DAG);
  case ISD::FNEG:               return LowerFNEG(Op, DAG);
  case ISD::SETCC:              return LowerSETCC(Op, DAG, DAG.getEntryNode());
  case ISD::SELECT:             return LowerSELECT(Op, DAG);
  case ISD::BRCOND:             return LowerBRCOND(Op, DAG);
  case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
  case ISD::CALL:               return LowerCALL(Op, DAG);
  case ISD::RET:                return LowerRET(Op, DAG);
  case ISD::FORMAL_ARGUMENTS:   return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::MEMSET:             return LowerMEMSET(Op, DAG);
  case ISD::MEMCPY:             return LowerMEMCPY(Op, DAG);
  case ISD::READCYCLECOUNTER:   return LowerREADCYCLCECOUNTER(Op, DAG);
  case ISD::VASTART:            return LowerVASTART(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  }
}

const char *X86TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return NULL;
  case X86ISD::SHLD:               return "X86ISD::SHLD";
  case X86ISD::SHRD:               return "X86ISD::SHRD";
  case X86ISD::FAND:               return "X86ISD::FAND";
  case X86ISD::FXOR:               return "X86ISD::FXOR";
  case X86ISD::FILD:               return "X86ISD::FILD";
  case X86ISD::FILD_FLAG:          return "X86ISD::FILD_FLAG";
  case X86ISD::FP_TO_INT16_IN_MEM: return "X86ISD::FP_TO_INT16_IN_MEM";
  case X86ISD::FP_TO_INT32_IN_MEM: return "X86ISD::FP_TO_INT32_IN_MEM";
  case X86ISD::FP_TO_INT64_IN_MEM: return "X86ISD::FP_TO_INT64_IN_MEM";
  case X86ISD::FLD:                return "X86ISD::FLD";
  case X86ISD::FST:                return "X86ISD::FST";
  case X86ISD::FP_GET_RESULT:      return "X86ISD::FP_GET_RESULT";
  case X86ISD::FP_SET_RESULT:      return "X86ISD::FP_SET_RESULT";
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
  case X86ISD::LOAD_PACK:          return "X86ISD::LOAD_PACK";
  case X86ISD::LOAD_UA:            return "X86ISD::LOAD_UA";
  case X86ISD::GlobalBaseReg:      return "X86ISD::GlobalBaseReg";
  case X86ISD::Wrapper:            return "X86ISD::Wrapper";
  case X86ISD::S2VEC:              return "X86ISD::S2VEC";
  case X86ISD::PEXTRW:             return "X86ISD::PEXTRW";
  case X86ISD::PINSRW:             return "X86ISD::PINSRW";
  case X86ISD::FMAX:               return "X86ISD::FMAX";
  case X86ISD::FMIN:               return "X86ISD::FMIN";
  }
}

/// isLegalAddressImmediate - Return true if the integer value or
/// GlobalValue can be used as the offset of the target addressing mode.
bool X86TargetLowering::isLegalAddressImmediate(int64_t V) const {
  // X86 allows a sign-extended 32-bit immediate field.
  return (V > -(1LL << 32) && V < (1LL << 32)-1);
}

bool X86TargetLowering::isLegalAddressImmediate(GlobalValue *GV) const {
  // GV is 64-bit but displacement field is 32-bit unless we are in small code
  // model. Mac OS X happens to support only small PIC code model.
  // FIXME: better support for other OS's.
  if (Subtarget->is64Bit() && !Subtarget->isTargetDarwin())
    return false;
  if (Subtarget->isTargetDarwin()) {
    Reloc::Model RModel = getTargetMachine().getRelocationModel();
    if (RModel == Reloc::Static)
      return true;
    else if (RModel == Reloc::DynamicNoPIC)
      return !DarwinGVRequiresExtraLoad(GV);
    else
      return false;
  } else
    return true;
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
          isSplatMask(Mask.Val)  ||
          isPSHUFHW_PSHUFLWMask(Mask.Val) ||
          X86::isUNPCKLMask(Mask.Val) ||
          X86::isUNPCKL_v_undef_Mask(Mask.Val) ||
          X86::isUNPCKHMask(Mask.Val));
}

bool X86TargetLowering::isVectorClearMaskLegal(std::vector<SDOperand> &BVOps,
                                               MVT::ValueType EVT,
                                               SelectionDAG &DAG) const {
  unsigned NumElts = BVOps.size();
  // Only do shuffles on 128-bit vector types for now.
  if (MVT::getSizeInBits(EVT) * NumElts == 64) return false;
  if (NumElts == 2) return true;
  if (NumElts == 4) {
    return (isMOVLMask(BVOps)  || isCommutedMOVL(BVOps, true) ||
            isSHUFPMask(BVOps) || isCommutedSHUFP(BVOps));
  }
  return false;
}

//===----------------------------------------------------------------------===//
//                           X86 Scheduler Hooks
//===----------------------------------------------------------------------===//

MachineBasicBlock *
X86TargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                           MachineBasicBlock *BB) {
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
    BuildMI(BB, Opc, 1).addMBB(sinkMBB);
    MachineFunction *F = BB->getParent();
    F->getBasicBlockList().insert(It, copy0MBB);
    F->getBasicBlockList().insert(It, sinkMBB);
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
    BuildMI(BB, X86::PHI, 4, MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

    delete MI;   // The pseudo instruction is gone now.
    return BB;
  }

  case X86::FP_TO_INT16_IN_MEM:
  case X86::FP_TO_INT32_IN_MEM:
  case X86::FP_TO_INT64_IN_MEM: {
    // Change the floating point control register to use "round towards zero"
    // mode when truncating to an integer value.
    MachineFunction *F = BB->getParent();
    int CWFrameIdx = F->getFrameInfo()->CreateStackObject(2, 2);
    addFrameReference(BuildMI(BB, X86::FNSTCW16m, 4), CWFrameIdx);

    // Load the old value of the high byte of the control word...
    unsigned OldCW =
      F->getSSARegMap()->createVirtualRegister(X86::GR16RegisterClass);
    addFrameReference(BuildMI(BB, X86::MOV16rm, 4, OldCW), CWFrameIdx);

    // Set the high part to be round to zero...
    addFrameReference(BuildMI(BB, X86::MOV16mi, 5), CWFrameIdx).addImm(0xC7F);

    // Reload the modified control word now...
    addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);

    // Restore the memory image of control word to original value
    addFrameReference(BuildMI(BB, X86::MOV16mr, 5), CWFrameIdx).addReg(OldCW);

    // Get the X86 opcode to use.
    unsigned Opc;
    switch (MI->getOpcode()) {
    default: assert(0 && "illegal opcode!");
    case X86::FP_TO_INT16_IN_MEM: Opc = X86::FpIST16m; break;
    case X86::FP_TO_INT32_IN_MEM: Opc = X86::FpIST32m; break;
    case X86::FP_TO_INT64_IN_MEM: Opc = X86::FpIST64m; break;
    }

    X86AddressMode AM;
    MachineOperand &Op = MI->getOperand(0);
    if (Op.isRegister()) {
      AM.BaseType = X86AddressMode::RegBase;
      AM.Base.Reg = Op.getReg();
    } else {
      AM.BaseType = X86AddressMode::FrameIndexBase;
      AM.Base.FrameIndex = Op.getFrameIndex();
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
    addFullAddress(BuildMI(BB, Opc, 5), AM).addReg(MI->getOperand(4).getReg());

    // Reload the original control word now.
    addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);

    delete MI;   // The pseudo instruction is gone now.
    return BB;
  }
  }
}

//===----------------------------------------------------------------------===//
//                           X86 Optimization Hooks
//===----------------------------------------------------------------------===//

void X86TargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op,
                                                       uint64_t Mask,
                                                       uint64_t &KnownZero, 
                                                       uint64_t &KnownOne,
                                                       unsigned Depth) const {
  unsigned Opc = Op.getOpcode();
  assert((Opc >= ISD::BUILTIN_OP_END ||
          Opc == ISD::INTRINSIC_WO_CHAIN ||
          Opc == ISD::INTRINSIC_W_CHAIN ||
          Opc == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");

  KnownZero = KnownOne = 0;   // Don't know anything.
  switch (Opc) {
  default: break;
  case X86ISD::SETCC: 
    KnownZero |= (MVT::getIntVTBitMask(Op.getValueType()) ^ 1ULL);
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
      ? V.getOperand(0) : DAG.getNode(ISD::UNDEF, MVT::getVectorBaseType(VT));
  } else if (V.getOpcode() == ISD::VECTOR_SHUFFLE) {
    SDOperand Idx = PermMask.getOperand(i);
    if (Idx.getOpcode() == ISD::UNDEF)
      return DAG.getNode(ISD::UNDEF, MVT::getVectorBaseType(VT));
    return getShuffleScalarElt(V.Val,cast<ConstantSDNode>(Idx)->getValue(),DAG);
  }
  return SDOperand();
}

/// isGAPlusOffset - Returns true (and the GlobalValue and the offset) if the
/// node is a GlobalAddress + an offset.
static bool isGAPlusOffset(SDNode *N, GlobalValue* &GA, int64_t &Offset) {
  if (N->getOpcode() == X86ISD::Wrapper) {
    if (dyn_cast<GlobalAddressSDNode>(N->getOperand(0))) {
      GA = cast<GlobalAddressSDNode>(N->getOperand(0))->getGlobal();
      return true;
    }
  } else if (N->getOpcode() == ISD::ADD) {
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
    int FI  = dyn_cast<FrameIndexSDNode>(Loc)->getIndex();
    int BFI = dyn_cast<FrameIndexSDNode>(BaseLoc)->getIndex();
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

static bool isBaseAlignment16(SDNode *Base, MachineFrameInfo *MFI,
                              const X86Subtarget *Subtarget) {
  GlobalValue *GV;
  int64_t Offset;
  if (isGAPlusOffset(Base, GV, Offset))
    return (GV->getAlignment() >= 16 && (Offset % 16) == 0);
  else {
    assert(Base->getOpcode() == ISD::FrameIndex && "Unexpected base node!");
    int BFI = dyn_cast<FrameIndexSDNode>(Base)->getIndex();
    if (BFI < 0)
      // Fixed objects do not specify alignment, however the offsets are known.
      return ((Subtarget->getStackAlignment() % 16) == 0 &&
              (MFI->getObjectOffset(BFI) % 16) == 0);
    else
      return MFI->getObjectAlignment(BFI) >= 16;
  }
  return false;
}


/// PerformShuffleCombine - Combine a vector_shuffle that is equal to
/// build_vector load1, load2, load3, load4, <0, 1, 2, 3> into a 128-bit load
/// if the load addresses are consecutive, non-overlapping, and in the right
/// order.
static SDOperand PerformShuffleCombine(SDNode *N, SelectionDAG &DAG,
                                       const X86Subtarget *Subtarget) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = MVT::getVectorBaseType(VT);
  SDOperand PermMask = N->getOperand(2);
  int NumElems = (int)PermMask.getNumOperands();
  SDNode *Base = NULL;
  for (int i = 0; i < NumElems; ++i) {
    SDOperand Idx = PermMask.getOperand(i);
    if (Idx.getOpcode() == ISD::UNDEF) {
      if (!Base) return SDOperand();
    } else {
      SDOperand Arg =
        getShuffleScalarElt(N, cast<ConstantSDNode>(Idx)->getValue(), DAG);
      if (!Arg.Val || !ISD::isNON_EXTLoad(Arg.Val))
        return SDOperand();
      if (!Base)
        Base = Arg.Val;
      else if (!isConsecutiveLoad(Arg.Val, Base,
                                  i, MVT::getSizeInBits(EVT)/8,MFI))
        return SDOperand();
    }
  }

  bool isAlign16 = isBaseAlignment16(Base->getOperand(1).Val, MFI, Subtarget);
  if (isAlign16) {
    LoadSDNode *LD = cast<LoadSDNode>(Base);
    return DAG.getLoad(VT, LD->getChain(), LD->getBasePtr(), LD->getSrcValue(),
                       LD->getSrcValueOffset());
  } else {
    // Just use movups, it's shorter.
    std::vector<MVT::ValueType> Tys;
    Tys.push_back(MVT::v4f32);
    Tys.push_back(MVT::Other);
    SmallVector<SDOperand, 3> Ops;
    Ops.push_back(Base->getOperand(0));
    Ops.push_back(Base->getOperand(1));
    Ops.push_back(Base->getOperand(2));
    return DAG.getNode(ISD::BIT_CONVERT, VT,
                       DAG.getNode(X86ISD::LOAD_UA, Tys, &Ops[0], Ops.size()));
  }
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


SDOperand X86TargetLowering::PerformDAGCombine(SDNode *N, 
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default: break;
  case ISD::VECTOR_SHUFFLE:
    return PerformShuffleCombine(N, DAG, Subtarget);
  case ISD::SELECT:
    return PerformSELECTCombine(N, DAG, Subtarget);
  }

  return SDOperand();
}

//===----------------------------------------------------------------------===//
//                           X86 Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
X86TargetLowering::ConstraintType
X86TargetLowering::getConstraintType(char ConstraintLetter) const {
  switch (ConstraintLetter) {
  case 'A':
  case 'r':
  case 'R':
  case 'l':
  case 'q':
  case 'Q':
  case 'x':
  case 'Y':
    return C_RegisterClass;
  default: return TargetLowering::getConstraintType(ConstraintLetter);
  }
}

/// isOperandValidForConstraint - Return the specified operand (possibly
/// modified) if the specified SDOperand is valid for the specified target
/// constraint letter, otherwise return null.
SDOperand X86TargetLowering::
isOperandValidForConstraint(SDOperand Op, char Constraint, SelectionDAG &DAG) {
  switch (Constraint) {
  default: break;
  case 'i':
    // Literal immediates are always ok.
    if (isa<ConstantSDNode>(Op)) return Op;
    
    // If we are in non-pic codegen mode, we allow the address of a global to
    // be used with 'i'.
    if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op)) {
      if (getTargetMachine().getRelocationModel() == Reloc::PIC_)
        return SDOperand(0, 0);
      
      if (GA->getOpcode() != ISD::TargetGlobalAddress)
        Op = DAG.getTargetGlobalAddress(GA->getGlobal(), GA->getValueType(0),
                                        GA->getOffset());
      return Op;
    }
    
    // Otherwise, not valid for this mode.
    return SDOperand(0, 0);
  }
  return TargetLowering::isOperandValidForConstraint(Op, Constraint, DAG);
}


std::vector<unsigned> X86TargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    // FIXME: not handling fp-stack yet!
    // FIXME: not handling MMX registers yet ('y' constraint).
    switch (Constraint[0]) {      // GCC X86 Constraint Letters
    default: break;  // Unknown constraint letter
    case 'A':   // EAX/EDX
      if (VT == MVT::i32 || VT == MVT::i64)
        return make_vector<unsigned>(X86::EAX, X86::EDX, 0);
      break;
    case 'r':   // GENERAL_REGS
    case 'R':   // LEGACY_REGS
      if (VT == MVT::i32)
        return make_vector<unsigned>(X86::EAX, X86::EDX, X86::ECX, X86::EBX,
                                     X86::ESI, X86::EDI, X86::EBP, X86::ESP, 0);
      else if (VT == MVT::i16)
        return make_vector<unsigned>(X86::AX, X86::DX, X86::CX, X86::BX, 
                                     X86::SI, X86::DI, X86::BP, X86::SP, 0);
      else if (VT == MVT::i8)
        return make_vector<unsigned>(X86::AL, X86::DL, X86::CL, X86::DL, 0);
      break;
    case 'l':   // INDEX_REGS
      if (VT == MVT::i32)
        return make_vector<unsigned>(X86::EAX, X86::EDX, X86::ECX, X86::EBX,
                                     X86::ESI, X86::EDI, X86::EBP, 0);
      else if (VT == MVT::i16)
        return make_vector<unsigned>(X86::AX, X86::DX, X86::CX, X86::BX, 
                                     X86::SI, X86::DI, X86::BP, 0);
      else if (VT == MVT::i8)
        return make_vector<unsigned>(X86::AL, X86::DL, X86::CL, X86::DL, 0);
      break;
    case 'q':   // Q_REGS (GENERAL_REGS in 64-bit mode)
    case 'Q':   // Q_REGS
      if (VT == MVT::i32)
        return make_vector<unsigned>(X86::EAX, X86::EDX, X86::ECX, X86::EBX, 0);
      else if (VT == MVT::i16)
        return make_vector<unsigned>(X86::AX, X86::DX, X86::CX, X86::BX, 0);
      else if (VT == MVT::i8)
        return make_vector<unsigned>(X86::AL, X86::DL, X86::CL, X86::DL, 0);
        break;
    case 'x':   // SSE_REGS if SSE1 allowed
      if (Subtarget->hasSSE1())
        return make_vector<unsigned>(X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
                                     X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7,
                                     0);
      return std::vector<unsigned>();
    case 'Y':   // SSE_REGS if SSE2 allowed
      if (Subtarget->hasSSE2())
        return make_vector<unsigned>(X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
                                     X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7,
                                     0);
      return std::vector<unsigned>();
    }
  }
  
  return std::vector<unsigned>();
}

std::pair<unsigned, const TargetRegisterClass*> 
X86TargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                MVT::ValueType VT) const {
  // Use the default implementation in TargetLowering to convert the register
  // constraint into a member of a register class.
  std::pair<unsigned, const TargetRegisterClass*> Res;
  Res = TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);

  // Not found as a standard register?
  if (Res.second == 0) {
    // GCC calls "st(0)" just plain "st".
    if (StringsEqualNoCase("{st}", Constraint)) {
      Res.first = X86::ST0;
      Res.second = X86::RSTRegisterClass;
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

