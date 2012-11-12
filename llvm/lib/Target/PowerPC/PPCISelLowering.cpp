//===-- PPCISelLowering.cpp - PPC DAG Lowering Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPCISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "PPCISelLowering.h"
#include "PPCMachineFunctionInfo.h"
#include "PPCPerfectShuffle.h"
#include "PPCTargetMachine.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

static bool CC_PPC_SVR4_Custom_Dummy(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                     CCValAssign::LocInfo &LocInfo,
                                     ISD::ArgFlagsTy &ArgFlags,
                                     CCState &State);
static bool CC_PPC_SVR4_Custom_AlignArgRegs(unsigned &ValNo, MVT &ValVT,
                                            MVT &LocVT,
                                            CCValAssign::LocInfo &LocInfo,
                                            ISD::ArgFlagsTy &ArgFlags,
                                            CCState &State);
static bool CC_PPC_SVR4_Custom_AlignFPArgRegs(unsigned &ValNo, MVT &ValVT,
                                              MVT &LocVT,
                                              CCValAssign::LocInfo &LocInfo,
                                              ISD::ArgFlagsTy &ArgFlags,
                                              CCState &State);

static cl::opt<bool> DisablePPCPreinc("disable-ppc-preinc",
cl::desc("disable preincrement load/store generation on PPC"), cl::Hidden);

static cl::opt<bool> DisableILPPref("disable-ppc-ilp-pref",
cl::desc("disable setting the node scheduling preference to ILP on PPC"), cl::Hidden);

static TargetLoweringObjectFile *CreateTLOF(const PPCTargetMachine &TM) {
  if (TM.getSubtargetImpl()->isDarwin())
    return new TargetLoweringObjectFileMachO();

  return new TargetLoweringObjectFileELF();
}

PPCTargetLowering::PPCTargetLowering(PPCTargetMachine &TM)
  : TargetLowering(TM, CreateTLOF(TM)), PPCSubTarget(*TM.getSubtargetImpl()) {
  const PPCSubtarget *Subtarget = &TM.getSubtarget<PPCSubtarget>();

  setPow2DivIsCheap();

  // Use _setjmp/_longjmp instead of setjmp/longjmp.
  setUseUnderscoreSetJmp(true);
  setUseUnderscoreLongJmp(true);

  // On PPC32/64, arguments smaller than 4/8 bytes are extended, so all
  // arguments are at least 4/8 bytes aligned.
  bool isPPC64 = Subtarget->isPPC64();
  setMinStackArgumentAlignment(isPPC64 ? 8:4);

  // Set up the register classes.
  addRegisterClass(MVT::i32, &PPC::GPRCRegClass);
  addRegisterClass(MVT::f32, &PPC::F4RCRegClass);
  addRegisterClass(MVT::f64, &PPC::F8RCRegClass);

  // PowerPC has an i16 but no i8 (or i1) SEXTLOAD
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i8, Expand);

  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // PowerPC has pre-inc load and store's.
  setIndexedLoadAction(ISD::PRE_INC, MVT::i1, Legal);
  setIndexedLoadAction(ISD::PRE_INC, MVT::i8, Legal);
  setIndexedLoadAction(ISD::PRE_INC, MVT::i16, Legal);
  setIndexedLoadAction(ISD::PRE_INC, MVT::i32, Legal);
  setIndexedLoadAction(ISD::PRE_INC, MVT::i64, Legal);
  setIndexedStoreAction(ISD::PRE_INC, MVT::i1, Legal);
  setIndexedStoreAction(ISD::PRE_INC, MVT::i8, Legal);
  setIndexedStoreAction(ISD::PRE_INC, MVT::i16, Legal);
  setIndexedStoreAction(ISD::PRE_INC, MVT::i32, Legal);
  setIndexedStoreAction(ISD::PRE_INC, MVT::i64, Legal);

  // This is used in the ppcf128->int sequence.  Note it has different semantics
  // from FP_ROUND:  that rounds to nearest, this rounds to zero.
  setOperationAction(ISD::FP_ROUND_INREG, MVT::ppcf128, Custom);

  // We do not currently implement these libm ops for PowerPC.
  setOperationAction(ISD::FFLOOR, MVT::ppcf128, Expand);
  setOperationAction(ISD::FCEIL,  MVT::ppcf128, Expand);
  setOperationAction(ISD::FTRUNC, MVT::ppcf128, Expand);
  setOperationAction(ISD::FRINT,  MVT::ppcf128, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::ppcf128, Expand);

  // PowerPC has no SREM/UREM instructions
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);

  // Don't use SMUL_LOHI/UMUL_LOHI or SDIVREM/UDIVREM to lower SREM/UREM.
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i64, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i64, Expand);

  // We don't support sin/cos/sqrt/fmod/pow
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FREM , MVT::f64, Expand);
  setOperationAction(ISD::FPOW , MVT::f64, Expand);
  setOperationAction(ISD::FMA  , MVT::f64, Legal);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::FREM , MVT::f32, Expand);
  setOperationAction(ISD::FPOW , MVT::f32, Expand);
  setOperationAction(ISD::FMA  , MVT::f32, Legal);

  setOperationAction(ISD::FLT_ROUNDS_, MVT::i32, Custom);

  // If we're enabling GP optimizations, use hardware square root
  if (!Subtarget->hasFSQRT()) {
    setOperationAction(ISD::FSQRT, MVT::f64, Expand);
    setOperationAction(ISD::FSQRT, MVT::f32, Expand);
  }

  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

  // PowerPC does not have BSWAP, CTPOP or CTTZ
  setOperationAction(ISD::BSWAP, MVT::i32  , Expand);
  setOperationAction(ISD::CTPOP, MVT::i32  , Expand);
  setOperationAction(ISD::CTTZ , MVT::i32  , Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i64  , Expand);
  setOperationAction(ISD::CTPOP, MVT::i64  , Expand);
  setOperationAction(ISD::CTTZ , MVT::i64  , Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i64, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i64, Expand);

  // PowerPC does not have ROTR
  setOperationAction(ISD::ROTR, MVT::i32   , Expand);
  setOperationAction(ISD::ROTR, MVT::i64   , Expand);

  // PowerPC does not have Select
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::i64, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::f64, Expand);

  // PowerPC wants to turn select_cc of FP into fsel when possible.
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);

  // PowerPC wants to optimize integer setcc a bit
  setOperationAction(ISD::SETCC, MVT::i32, Custom);

  // PowerPC does not have BRCOND which requires SetCC
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);

  setOperationAction(ISD::BR_JT,  MVT::Other, Expand);

  // PowerPC turns FP_TO_SINT into FCTIWZ and some load/stores.
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);

  // PowerPC does not have [U|S]INT_TO_FP
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);

  setOperationAction(ISD::BITCAST, MVT::f32, Expand);
  setOperationAction(ISD::BITCAST, MVT::i32, Expand);
  setOperationAction(ISD::BITCAST, MVT::i64, Expand);
  setOperationAction(ISD::BITCAST, MVT::f64, Expand);

  // We cannot sextinreg(i1).  Expand to shifts.
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  setOperationAction(ISD::EXCEPTIONADDR, MVT::i64, Expand);
  setOperationAction(ISD::EHSELECTION,   MVT::i64, Expand);
  setOperationAction(ISD::EXCEPTIONADDR, MVT::i32, Expand);
  setOperationAction(ISD::EHSELECTION,   MVT::i32, Expand);


  // We want to legalize GlobalAddress and ConstantPool nodes into the
  // appropriate instructions to materialize the address.
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  setOperationAction(ISD::BlockAddress,  MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32, Custom);
  setOperationAction(ISD::JumpTable,     MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i64, Custom);
  setOperationAction(ISD::BlockAddress,  MVT::i64, Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i64, Custom);
  setOperationAction(ISD::JumpTable,     MVT::i64, Custom);

  // TRAP is legal.
  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  // TRAMPOLINE is custom lowered.
  setOperationAction(ISD::INIT_TRAMPOLINE, MVT::Other, Custom);
  setOperationAction(ISD::ADJUST_TRAMPOLINE, MVT::Other, Custom);

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);

  if (Subtarget->isSVR4ABI()) {
    if (isPPC64) {
      // VAARG always uses double-word chunks, so promote anything smaller.
      setOperationAction(ISD::VAARG, MVT::i1, Promote);
      AddPromotedToType (ISD::VAARG, MVT::i1, MVT::i64);
      setOperationAction(ISD::VAARG, MVT::i8, Promote);
      AddPromotedToType (ISD::VAARG, MVT::i8, MVT::i64);
      setOperationAction(ISD::VAARG, MVT::i16, Promote);
      AddPromotedToType (ISD::VAARG, MVT::i16, MVT::i64);
      setOperationAction(ISD::VAARG, MVT::i32, Promote);
      AddPromotedToType (ISD::VAARG, MVT::i32, MVT::i64);
      setOperationAction(ISD::VAARG, MVT::Other, Expand);
    } else {
      // VAARG is custom lowered with the 32-bit SVR4 ABI.
      setOperationAction(ISD::VAARG, MVT::Other, Custom);
      setOperationAction(ISD::VAARG, MVT::i64, Custom);
    }
  } else
    setOperationAction(ISD::VAARG, MVT::Other, Expand);

  // Use the default implementation.
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Custom);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Custom);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64  , Custom);

  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  // Comparisons that require checking two conditions.
  setCondCodeAction(ISD::SETULT, MVT::f32, Expand);
  setCondCodeAction(ISD::SETULT, MVT::f64, Expand);
  setCondCodeAction(ISD::SETUGT, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUGT, MVT::f64, Expand);
  setCondCodeAction(ISD::SETUEQ, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUEQ, MVT::f64, Expand);
  setCondCodeAction(ISD::SETOGE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETOGE, MVT::f64, Expand);
  setCondCodeAction(ISD::SETOLE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETOLE, MVT::f64, Expand);
  setCondCodeAction(ISD::SETONE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETONE, MVT::f64, Expand);

  if (Subtarget->has64BitSupport()) {
    // They also have instructions for converting between i64 and fp.
    setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
    setOperationAction(ISD::FP_TO_UINT, MVT::i64, Expand);
    setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
    setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);
    // This is just the low 32 bits of a (signed) fp->i64 conversion.
    // We cannot do this with Promote because i64 is not a legal type.
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);

    // FIXME: disable this lowered code.  This generates 64-bit register values,
    // and we don't model the fact that the top part is clobbered by calls.  We
    // need to flag these together so that the value isn't live across a call.
    //setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
  } else {
    // PowerPC does not have FP_TO_UINT on 32-bit implementations.
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  }

  if (Subtarget->use64BitRegs()) {
    // 64-bit PowerPC implementations can support i64 types directly
    addRegisterClass(MVT::i64, &PPC::G8RCRegClass);
    // BUILD_PAIR can't be handled natively, and should be expanded to shl/or
    setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand);
    // 64-bit PowerPC wants to expand i128 shifts itself.
    setOperationAction(ISD::SHL_PARTS, MVT::i64, Custom);
    setOperationAction(ISD::SRA_PARTS, MVT::i64, Custom);
    setOperationAction(ISD::SRL_PARTS, MVT::i64, Custom);
  } else {
    // 32-bit PowerPC wants to expand i64 shifts itself.
    setOperationAction(ISD::SHL_PARTS, MVT::i32, Custom);
    setOperationAction(ISD::SRA_PARTS, MVT::i32, Custom);
    setOperationAction(ISD::SRL_PARTS, MVT::i32, Custom);
  }

  if (Subtarget->hasAltivec()) {
    // First set operation action for all vector types to expand. Then we
    // will selectively turn on ones that can be effectively codegen'd.
    for (unsigned i = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
         i <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++i) {
      MVT::SimpleValueType VT = (MVT::SimpleValueType)i;

      // add/sub are legal for all supported vector VT's.
      setOperationAction(ISD::ADD , VT, Legal);
      setOperationAction(ISD::SUB , VT, Legal);

      // We promote all shuffles to v16i8.
      setOperationAction(ISD::VECTOR_SHUFFLE, VT, Promote);
      AddPromotedToType (ISD::VECTOR_SHUFFLE, VT, MVT::v16i8);

      // We promote all non-typed operations to v4i32.
      setOperationAction(ISD::AND   , VT, Promote);
      AddPromotedToType (ISD::AND   , VT, MVT::v4i32);
      setOperationAction(ISD::OR    , VT, Promote);
      AddPromotedToType (ISD::OR    , VT, MVT::v4i32);
      setOperationAction(ISD::XOR   , VT, Promote);
      AddPromotedToType (ISD::XOR   , VT, MVT::v4i32);
      setOperationAction(ISD::LOAD  , VT, Promote);
      AddPromotedToType (ISD::LOAD  , VT, MVT::v4i32);
      setOperationAction(ISD::SELECT, VT, Promote);
      AddPromotedToType (ISD::SELECT, VT, MVT::v4i32);
      setOperationAction(ISD::STORE, VT, Promote);
      AddPromotedToType (ISD::STORE, VT, MVT::v4i32);

      // No other operations are legal.
      setOperationAction(ISD::MUL , VT, Expand);
      setOperationAction(ISD::SDIV, VT, Expand);
      setOperationAction(ISD::SREM, VT, Expand);
      setOperationAction(ISD::UDIV, VT, Expand);
      setOperationAction(ISD::UREM, VT, Expand);
      setOperationAction(ISD::FDIV, VT, Expand);
      setOperationAction(ISD::FNEG, VT, Expand);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Expand);
      setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Expand);
      setOperationAction(ISD::BUILD_VECTOR, VT, Expand);
      setOperationAction(ISD::UMUL_LOHI, VT, Expand);
      setOperationAction(ISD::SMUL_LOHI, VT, Expand);
      setOperationAction(ISD::UDIVREM, VT, Expand);
      setOperationAction(ISD::SDIVREM, VT, Expand);
      setOperationAction(ISD::SCALAR_TO_VECTOR, VT, Expand);
      setOperationAction(ISD::FPOW, VT, Expand);
      setOperationAction(ISD::CTPOP, VT, Expand);
      setOperationAction(ISD::CTLZ, VT, Expand);
      setOperationAction(ISD::CTLZ_ZERO_UNDEF, VT, Expand);
      setOperationAction(ISD::CTTZ, VT, Expand);
      setOperationAction(ISD::CTTZ_ZERO_UNDEF, VT, Expand);
      setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Expand);

      for (unsigned j = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
           j <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++j) {
        MVT::SimpleValueType InnerVT = (MVT::SimpleValueType)j;
        setTruncStoreAction(VT, InnerVT, Expand);
      }
      setLoadExtAction(ISD::SEXTLOAD, VT, Expand);
      setLoadExtAction(ISD::ZEXTLOAD, VT, Expand);
      setLoadExtAction(ISD::EXTLOAD, VT, Expand);
    }

    for (unsigned i = (unsigned)MVT::FIRST_FP_VECTOR_VALUETYPE;
         i <= (unsigned)MVT::LAST_FP_VECTOR_VALUETYPE; ++i) {
      MVT::SimpleValueType VT = (MVT::SimpleValueType)i;
      setOperationAction(ISD::FSQRT, VT, Expand);
    }

    // We can custom expand all VECTOR_SHUFFLEs to VPERM, others we can handle
    // with merges, splats, etc.
    setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16i8, Custom);

    setOperationAction(ISD::AND   , MVT::v4i32, Legal);
    setOperationAction(ISD::OR    , MVT::v4i32, Legal);
    setOperationAction(ISD::XOR   , MVT::v4i32, Legal);
    setOperationAction(ISD::LOAD  , MVT::v4i32, Legal);
    setOperationAction(ISD::SELECT, MVT::v4i32, Expand);
    setOperationAction(ISD::STORE , MVT::v4i32, Legal);
    setOperationAction(ISD::FP_TO_SINT, MVT::v4i32, Legal);
    setOperationAction(ISD::FP_TO_UINT, MVT::v4i32, Legal);
    setOperationAction(ISD::SINT_TO_FP, MVT::v4i32, Legal);
    setOperationAction(ISD::UINT_TO_FP, MVT::v4i32, Legal);

    addRegisterClass(MVT::v4f32, &PPC::VRRCRegClass);
    addRegisterClass(MVT::v4i32, &PPC::VRRCRegClass);
    addRegisterClass(MVT::v8i16, &PPC::VRRCRegClass);
    addRegisterClass(MVT::v16i8, &PPC::VRRCRegClass);

    setOperationAction(ISD::MUL, MVT::v4f32, Legal);
    setOperationAction(ISD::FMA, MVT::v4f32, Legal);
    setOperationAction(ISD::MUL, MVT::v4i32, Custom);
    setOperationAction(ISD::MUL, MVT::v8i16, Custom);
    setOperationAction(ISD::MUL, MVT::v16i8, Custom);

    setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4f32, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4i32, Custom);

    setOperationAction(ISD::BUILD_VECTOR, MVT::v16i8, Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v8i16, Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4i32, Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4f32, Custom);

    // Altivec does not contain unordered floating-point compare instructions
    setCondCodeAction(ISD::SETUO, MVT::v4f32, Expand);
    setCondCodeAction(ISD::SETUEQ, MVT::v4f32, Expand);
    setCondCodeAction(ISD::SETUGT, MVT::v4f32, Expand);
    setCondCodeAction(ISD::SETUGE, MVT::v4f32, Expand);
    setCondCodeAction(ISD::SETULT, MVT::v4f32, Expand);
    setCondCodeAction(ISD::SETULE, MVT::v4f32, Expand);
  }

  if (Subtarget->has64BitSupport()) {
    setOperationAction(ISD::PREFETCH, MVT::Other, Legal);
    setOperationAction(ISD::READCYCLECOUNTER, MVT::i64, Legal);
  }

  setOperationAction(ISD::ATOMIC_LOAD,  MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_STORE, MVT::i32, Expand);

  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent); // FIXME: Is this correct?

  if (isPPC64) {
    setStackPointerRegisterToSaveRestore(PPC::X1);
    setExceptionPointerRegister(PPC::X3);
    setExceptionSelectorRegister(PPC::X4);
  } else {
    setStackPointerRegisterToSaveRestore(PPC::R1);
    setExceptionPointerRegister(PPC::R3);
    setExceptionSelectorRegister(PPC::R4);
  }

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::SINT_TO_FP);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::BR_CC);
  setTargetDAGCombine(ISD::BSWAP);

  // Darwin long double math library functions have $LDBL128 appended.
  if (Subtarget->isDarwin()) {
    setLibcallName(RTLIB::COS_PPCF128, "cosl$LDBL128");
    setLibcallName(RTLIB::POW_PPCF128, "powl$LDBL128");
    setLibcallName(RTLIB::REM_PPCF128, "fmodl$LDBL128");
    setLibcallName(RTLIB::SIN_PPCF128, "sinl$LDBL128");
    setLibcallName(RTLIB::SQRT_PPCF128, "sqrtl$LDBL128");
    setLibcallName(RTLIB::LOG_PPCF128, "logl$LDBL128");
    setLibcallName(RTLIB::LOG2_PPCF128, "log2l$LDBL128");
    setLibcallName(RTLIB::LOG10_PPCF128, "log10l$LDBL128");
    setLibcallName(RTLIB::EXP_PPCF128, "expl$LDBL128");
    setLibcallName(RTLIB::EXP2_PPCF128, "exp2l$LDBL128");
  }

  setMinFunctionAlignment(2);
  if (PPCSubTarget.isDarwin())
    setPrefFunctionAlignment(4);

  if (isPPC64 && Subtarget->isJITCodeModel())
    // Temporary workaround for the inability of PPC64 JIT to handle jump
    // tables.
    setSupportJumpTables(false);

  setInsertFencesForAtomic(true);

  setSchedulingPreference(Sched::Hybrid);

  computeRegisterProperties();

  // The Freescale cores does better with aggressive inlining of memcpy and
  // friends. Gcc uses same threshold of 128 bytes (= 32 word stores).
  if (Subtarget->getDarwinDirective() == PPC::DIR_E500mc ||
      Subtarget->getDarwinDirective() == PPC::DIR_E5500) {
    maxStoresPerMemset = 32;
    maxStoresPerMemsetOptSize = 16;
    maxStoresPerMemcpy = 32;
    maxStoresPerMemcpyOptSize = 8;
    maxStoresPerMemmove = 32;
    maxStoresPerMemmoveOptSize = 8;

    setPrefFunctionAlignment(4);
    benefitFromCodePlacementOpt = true;
  }
}

/// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
/// function arguments in the caller parameter area.
unsigned PPCTargetLowering::getByValTypeAlignment(Type *Ty) const {
  const TargetMachine &TM = getTargetMachine();
  // Darwin passes everything on 4 byte boundary.
  if (TM.getSubtarget<PPCSubtarget>().isDarwin())
    return 4;

  // 16byte and wider vectors are passed on 16byte boundary.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    if (VTy->getBitWidth() >= 128)
      return 16;

  // The rest is 8 on PPC64 and 4 on PPC32 boundary.
   if (PPCSubTarget.isPPC64())
     return 8;

  return 4;
}

const char *PPCTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case PPCISD::FSEL:            return "PPCISD::FSEL";
  case PPCISD::FCFID:           return "PPCISD::FCFID";
  case PPCISD::FCTIDZ:          return "PPCISD::FCTIDZ";
  case PPCISD::FCTIWZ:          return "PPCISD::FCTIWZ";
  case PPCISD::STFIWX:          return "PPCISD::STFIWX";
  case PPCISD::VMADDFP:         return "PPCISD::VMADDFP";
  case PPCISD::VNMSUBFP:        return "PPCISD::VNMSUBFP";
  case PPCISD::VPERM:           return "PPCISD::VPERM";
  case PPCISD::Hi:              return "PPCISD::Hi";
  case PPCISD::Lo:              return "PPCISD::Lo";
  case PPCISD::TOC_ENTRY:       return "PPCISD::TOC_ENTRY";
  case PPCISD::TOC_RESTORE:     return "PPCISD::TOC_RESTORE";
  case PPCISD::LOAD:            return "PPCISD::LOAD";
  case PPCISD::LOAD_TOC:        return "PPCISD::LOAD_TOC";
  case PPCISD::DYNALLOC:        return "PPCISD::DYNALLOC";
  case PPCISD::GlobalBaseReg:   return "PPCISD::GlobalBaseReg";
  case PPCISD::SRL:             return "PPCISD::SRL";
  case PPCISD::SRA:             return "PPCISD::SRA";
  case PPCISD::SHL:             return "PPCISD::SHL";
  case PPCISD::EXTSW_32:        return "PPCISD::EXTSW_32";
  case PPCISD::STD_32:          return "PPCISD::STD_32";
  case PPCISD::CALL_SVR4:       return "PPCISD::CALL_SVR4";
  case PPCISD::CALL_NOP_SVR4:   return "PPCISD::CALL_NOP_SVR4";
  case PPCISD::CALL_Darwin:     return "PPCISD::CALL_Darwin";
  case PPCISD::NOP:             return "PPCISD::NOP";
  case PPCISD::MTCTR:           return "PPCISD::MTCTR";
  case PPCISD::BCTRL_Darwin:    return "PPCISD::BCTRL_Darwin";
  case PPCISD::BCTRL_SVR4:      return "PPCISD::BCTRL_SVR4";
  case PPCISD::RET_FLAG:        return "PPCISD::RET_FLAG";
  case PPCISD::MFCR:            return "PPCISD::MFCR";
  case PPCISD::VCMP:            return "PPCISD::VCMP";
  case PPCISD::VCMPo:           return "PPCISD::VCMPo";
  case PPCISD::LBRX:            return "PPCISD::LBRX";
  case PPCISD::STBRX:           return "PPCISD::STBRX";
  case PPCISD::LARX:            return "PPCISD::LARX";
  case PPCISD::STCX:            return "PPCISD::STCX";
  case PPCISD::COND_BRANCH:     return "PPCISD::COND_BRANCH";
  case PPCISD::MFFS:            return "PPCISD::MFFS";
  case PPCISD::MTFSB0:          return "PPCISD::MTFSB0";
  case PPCISD::MTFSB1:          return "PPCISD::MTFSB1";
  case PPCISD::FADDRTZ:         return "PPCISD::FADDRTZ";
  case PPCISD::MTFSF:           return "PPCISD::MTFSF";
  case PPCISD::TC_RETURN:       return "PPCISD::TC_RETURN";
  case PPCISD::CR6SET:          return "PPCISD::CR6SET";
  case PPCISD::CR6UNSET:        return "PPCISD::CR6UNSET";
  }
}

EVT PPCTargetLowering::getSetCCResultType(EVT VT) const {
  if (!VT.isVector())
    return MVT::i32;
  return VT.changeVectorElementTypeToInteger();
}

//===----------------------------------------------------------------------===//
// Node matching predicates, for use by the tblgen matching code.
//===----------------------------------------------------------------------===//

/// isFloatingPointZero - Return true if this is 0.0 or -0.0.
static bool isFloatingPointZero(SDValue Op) {
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op))
    return CFP->getValueAPF().isZero();
  else if (ISD::isEXTLoad(Op.getNode()) || ISD::isNON_EXTLoad(Op.getNode())) {
    // Maybe this has already been legalized into the constant pool?
    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Op.getOperand(1)))
      if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CP->getConstVal()))
        return CFP->getValueAPF().isZero();
  }
  return false;
}

/// isConstantOrUndef - Op is either an undef node or a ConstantSDNode.  Return
/// true if Op is undef or if it matches the specified value.
static bool isConstantOrUndef(int Op, int Val) {
  return Op < 0 || Op == Val;
}

/// isVPKUHUMShuffleMask - Return true if this is the shuffle mask for a
/// VPKUHUM instruction.
bool PPC::isVPKUHUMShuffleMask(ShuffleVectorSDNode *N, bool isUnary) {
  if (!isUnary) {
    for (unsigned i = 0; i != 16; ++i)
      if (!isConstantOrUndef(N->getMaskElt(i),  i*2+1))
        return false;
  } else {
    for (unsigned i = 0; i != 8; ++i)
      if (!isConstantOrUndef(N->getMaskElt(i),    i*2+1) ||
          !isConstantOrUndef(N->getMaskElt(i+8),  i*2+1))
        return false;
  }
  return true;
}

/// isVPKUWUMShuffleMask - Return true if this is the shuffle mask for a
/// VPKUWUM instruction.
bool PPC::isVPKUWUMShuffleMask(ShuffleVectorSDNode *N, bool isUnary) {
  if (!isUnary) {
    for (unsigned i = 0; i != 16; i += 2)
      if (!isConstantOrUndef(N->getMaskElt(i  ),  i*2+2) ||
          !isConstantOrUndef(N->getMaskElt(i+1),  i*2+3))
        return false;
  } else {
    for (unsigned i = 0; i != 8; i += 2)
      if (!isConstantOrUndef(N->getMaskElt(i  ),  i*2+2) ||
          !isConstantOrUndef(N->getMaskElt(i+1),  i*2+3) ||
          !isConstantOrUndef(N->getMaskElt(i+8),  i*2+2) ||
          !isConstantOrUndef(N->getMaskElt(i+9),  i*2+3))
        return false;
  }
  return true;
}

/// isVMerge - Common function, used to match vmrg* shuffles.
///
static bool isVMerge(ShuffleVectorSDNode *N, unsigned UnitSize,
                     unsigned LHSStart, unsigned RHSStart) {
  assert(N->getValueType(0) == MVT::v16i8 &&
         "PPC only supports shuffles by bytes!");
  assert((UnitSize == 1 || UnitSize == 2 || UnitSize == 4) &&
         "Unsupported merge size!");

  for (unsigned i = 0; i != 8/UnitSize; ++i)     // Step over units
    for (unsigned j = 0; j != UnitSize; ++j) {   // Step over bytes within unit
      if (!isConstantOrUndef(N->getMaskElt(i*UnitSize*2+j),
                             LHSStart+j+i*UnitSize) ||
          !isConstantOrUndef(N->getMaskElt(i*UnitSize*2+UnitSize+j),
                             RHSStart+j+i*UnitSize))
        return false;
    }
  return true;
}

/// isVMRGLShuffleMask - Return true if this is a shuffle mask suitable for
/// a VRGL* instruction with the specified unit size (1,2 or 4 bytes).
bool PPC::isVMRGLShuffleMask(ShuffleVectorSDNode *N, unsigned UnitSize,
                             bool isUnary) {
  if (!isUnary)
    return isVMerge(N, UnitSize, 8, 24);
  return isVMerge(N, UnitSize, 8, 8);
}

/// isVMRGHShuffleMask - Return true if this is a shuffle mask suitable for
/// a VRGH* instruction with the specified unit size (1,2 or 4 bytes).
bool PPC::isVMRGHShuffleMask(ShuffleVectorSDNode *N, unsigned UnitSize,
                             bool isUnary) {
  if (!isUnary)
    return isVMerge(N, UnitSize, 0, 16);
  return isVMerge(N, UnitSize, 0, 0);
}


/// isVSLDOIShuffleMask - If this is a vsldoi shuffle mask, return the shift
/// amount, otherwise return -1.
int PPC::isVSLDOIShuffleMask(SDNode *N, bool isUnary) {
  assert(N->getValueType(0) == MVT::v16i8 &&
         "PPC only supports shuffles by bytes!");

  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);

  // Find the first non-undef value in the shuffle mask.
  unsigned i;
  for (i = 0; i != 16 && SVOp->getMaskElt(i) < 0; ++i)
    /*search*/;

  if (i == 16) return -1;  // all undef.

  // Otherwise, check to see if the rest of the elements are consecutively
  // numbered from this value.
  unsigned ShiftAmt = SVOp->getMaskElt(i);
  if (ShiftAmt < i) return -1;
  ShiftAmt -= i;

  if (!isUnary) {
    // Check the rest of the elements to see if they are consecutive.
    for (++i; i != 16; ++i)
      if (!isConstantOrUndef(SVOp->getMaskElt(i), ShiftAmt+i))
        return -1;
  } else {
    // Check the rest of the elements to see if they are consecutive.
    for (++i; i != 16; ++i)
      if (!isConstantOrUndef(SVOp->getMaskElt(i), (ShiftAmt+i) & 15))
        return -1;
  }
  return ShiftAmt;
}

/// isSplatShuffleMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a splat of a single element that is suitable for input to
/// VSPLTB/VSPLTH/VSPLTW.
bool PPC::isSplatShuffleMask(ShuffleVectorSDNode *N, unsigned EltSize) {
  assert(N->getValueType(0) == MVT::v16i8 &&
         (EltSize == 1 || EltSize == 2 || EltSize == 4));

  // This is a splat operation if each element of the permute is the same, and
  // if the value doesn't reference the second vector.
  unsigned ElementBase = N->getMaskElt(0);

  // FIXME: Handle UNDEF elements too!
  if (ElementBase >= 16)
    return false;

  // Check that the indices are consecutive, in the case of a multi-byte element
  // splatted with a v16i8 mask.
  for (unsigned i = 1; i != EltSize; ++i)
    if (N->getMaskElt(i) < 0 || N->getMaskElt(i) != (int)(i+ElementBase))
      return false;

  for (unsigned i = EltSize, e = 16; i != e; i += EltSize) {
    if (N->getMaskElt(i) < 0) continue;
    for (unsigned j = 0; j != EltSize; ++j)
      if (N->getMaskElt(i+j) != N->getMaskElt(j))
        return false;
  }
  return true;
}

/// isAllNegativeZeroVector - Returns true if all elements of build_vector
/// are -0.0.
bool PPC::isAllNegativeZeroVector(SDNode *N) {
  BuildVectorSDNode *BV = cast<BuildVectorSDNode>(N);

  APInt APVal, APUndef;
  unsigned BitSize;
  bool HasAnyUndefs;

  if (BV->isConstantSplat(APVal, APUndef, BitSize, HasAnyUndefs, 32, true))
    if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0)))
      return CFP->getValueAPF().isNegZero();

  return false;
}

/// getVSPLTImmediate - Return the appropriate VSPLT* immediate to splat the
/// specified isSplatShuffleMask VECTOR_SHUFFLE mask.
unsigned PPC::getVSPLTImmediate(SDNode *N, unsigned EltSize) {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  assert(isSplatShuffleMask(SVOp, EltSize));
  return SVOp->getMaskElt(0) / EltSize;
}

/// get_VSPLTI_elt - If this is a build_vector of constants which can be formed
/// by using a vspltis[bhw] instruction of the specified element size, return
/// the constant being splatted.  The ByteSize field indicates the number of
/// bytes of each element [124] -> [bhw].
SDValue PPC::get_VSPLTI_elt(SDNode *N, unsigned ByteSize, SelectionDAG &DAG) {
  SDValue OpVal(0, 0);

  // If ByteSize of the splat is bigger than the element size of the
  // build_vector, then we have a case where we are checking for a splat where
  // multiple elements of the buildvector are folded together into a single
  // logical element of the splat (e.g. "vsplish 1" to splat {0,1}*8).
  unsigned EltSize = 16/N->getNumOperands();
  if (EltSize < ByteSize) {
    unsigned Multiple = ByteSize/EltSize;   // Number of BV entries per spltval.
    SDValue UniquedVals[4];
    assert(Multiple > 1 && Multiple <= 4 && "How can this happen?");

    // See if all of the elements in the buildvector agree across.
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
      if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
      // If the element isn't a constant, bail fully out.
      if (!isa<ConstantSDNode>(N->getOperand(i))) return SDValue();


      if (UniquedVals[i&(Multiple-1)].getNode() == 0)
        UniquedVals[i&(Multiple-1)] = N->getOperand(i);
      else if (UniquedVals[i&(Multiple-1)] != N->getOperand(i))
        return SDValue();  // no match.
    }

    // Okay, if we reached this point, UniquedVals[0..Multiple-1] contains
    // either constant or undef values that are identical for each chunk.  See
    // if these chunks can form into a larger vspltis*.

    // Check to see if all of the leading entries are either 0 or -1.  If
    // neither, then this won't fit into the immediate field.
    bool LeadingZero = true;
    bool LeadingOnes = true;
    for (unsigned i = 0; i != Multiple-1; ++i) {
      if (UniquedVals[i].getNode() == 0) continue;  // Must have been undefs.

      LeadingZero &= cast<ConstantSDNode>(UniquedVals[i])->isNullValue();
      LeadingOnes &= cast<ConstantSDNode>(UniquedVals[i])->isAllOnesValue();
    }
    // Finally, check the least significant entry.
    if (LeadingZero) {
      if (UniquedVals[Multiple-1].getNode() == 0)
        return DAG.getTargetConstant(0, MVT::i32);  // 0,0,0,undef
      int Val = cast<ConstantSDNode>(UniquedVals[Multiple-1])->getZExtValue();
      if (Val < 16)
        return DAG.getTargetConstant(Val, MVT::i32);  // 0,0,0,4 -> vspltisw(4)
    }
    if (LeadingOnes) {
      if (UniquedVals[Multiple-1].getNode() == 0)
        return DAG.getTargetConstant(~0U, MVT::i32);  // -1,-1,-1,undef
      int Val =cast<ConstantSDNode>(UniquedVals[Multiple-1])->getSExtValue();
      if (Val >= -16)                            // -1,-1,-1,-2 -> vspltisw(-2)
        return DAG.getTargetConstant(Val, MVT::i32);
    }

    return SDValue();
  }

  // Check to see if this buildvec has a single non-undef value in its elements.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    if (OpVal.getNode() == 0)
      OpVal = N->getOperand(i);
    else if (OpVal != N->getOperand(i))
      return SDValue();
  }

  if (OpVal.getNode() == 0) return SDValue();  // All UNDEF: use implicit def.

  unsigned ValSizeInBytes = EltSize;
  uint64_t Value = 0;
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal)) {
    Value = CN->getZExtValue();
  } else if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(OpVal)) {
    assert(CN->getValueType(0) == MVT::f32 && "Only one legal FP vector type!");
    Value = FloatToBits(CN->getValueAPF().convertToFloat());
  }

  // If the splat value is larger than the element value, then we can never do
  // this splat.  The only case that we could fit the replicated bits into our
  // immediate field for would be zero, and we prefer to use vxor for it.
  if (ValSizeInBytes < ByteSize) return SDValue();

  // If the element value is larger than the splat value, cut it in half and
  // check to see if the two halves are equal.  Continue doing this until we
  // get to ByteSize.  This allows us to handle 0x01010101 as 0x01.
  while (ValSizeInBytes > ByteSize) {
    ValSizeInBytes >>= 1;

    // If the top half equals the bottom half, we're still ok.
    if (((Value >> (ValSizeInBytes*8)) & ((1 << (8*ValSizeInBytes))-1)) !=
         (Value                        & ((1 << (8*ValSizeInBytes))-1)))
      return SDValue();
  }

  // Properly sign extend the value.
  int MaskVal = SignExtend32(Value, ByteSize * 8);

  // If this is zero, don't match, zero matches ISD::isBuildVectorAllZeros.
  if (MaskVal == 0) return SDValue();

  // Finally, if this value fits in a 5 bit sext field, return it
  if (SignExtend32<5>(MaskVal) == MaskVal)
    return DAG.getTargetConstant(MaskVal, MVT::i32);
  return SDValue();
}

//===----------------------------------------------------------------------===//
//  Addressing Mode Selection
//===----------------------------------------------------------------------===//

/// isIntS16Immediate - This method tests to see if the node is either a 32-bit
/// or 64-bit immediate, and if the value can be accurately represented as a
/// sign extension from a 16-bit value.  If so, this returns true and the
/// immediate.
static bool isIntS16Immediate(SDNode *N, short &Imm) {
  if (N->getOpcode() != ISD::Constant)
    return false;

  Imm = (short)cast<ConstantSDNode>(N)->getZExtValue();
  if (N->getValueType(0) == MVT::i32)
    return Imm == (int32_t)cast<ConstantSDNode>(N)->getZExtValue();
  else
    return Imm == (int64_t)cast<ConstantSDNode>(N)->getZExtValue();
}
static bool isIntS16Immediate(SDValue Op, short &Imm) {
  return isIntS16Immediate(Op.getNode(), Imm);
}


/// SelectAddressRegReg - Given the specified addressed, check to see if it
/// can be represented as an indexed [r+r] operation.  Returns false if it
/// can be more efficiently represented with [r+imm].
bool PPCTargetLowering::SelectAddressRegReg(SDValue N, SDValue &Base,
                                            SDValue &Index,
                                            SelectionDAG &DAG) const {
  short imm = 0;
  if (N.getOpcode() == ISD::ADD) {
    if (isIntS16Immediate(N.getOperand(1), imm))
      return false;    // r+i
    if (N.getOperand(1).getOpcode() == PPCISD::Lo)
      return false;    // r+i

    Base = N.getOperand(0);
    Index = N.getOperand(1);
    return true;
  } else if (N.getOpcode() == ISD::OR) {
    if (isIntS16Immediate(N.getOperand(1), imm))
      return false;    // r+i can fold it if we can.

    // If this is an or of disjoint bitfields, we can codegen this as an add
    // (for better address arithmetic) if the LHS and RHS of the OR are provably
    // disjoint.
    APInt LHSKnownZero, LHSKnownOne;
    APInt RHSKnownZero, RHSKnownOne;
    DAG.ComputeMaskedBits(N.getOperand(0),
                          LHSKnownZero, LHSKnownOne);

    if (LHSKnownZero.getBoolValue()) {
      DAG.ComputeMaskedBits(N.getOperand(1),
                            RHSKnownZero, RHSKnownOne);
      // If all of the bits are known zero on the LHS or RHS, the add won't
      // carry.
      if (~(LHSKnownZero | RHSKnownZero) == 0) {
        Base = N.getOperand(0);
        Index = N.getOperand(1);
        return true;
      }
    }
  }

  return false;
}

/// Returns true if the address N can be represented by a base register plus
/// a signed 16-bit displacement [r+imm], and if it is not better
/// represented as reg+reg.
bool PPCTargetLowering::SelectAddressRegImm(SDValue N, SDValue &Disp,
                                            SDValue &Base,
                                            SelectionDAG &DAG) const {
  // FIXME dl should come from parent load or store, not from address
  DebugLoc dl = N.getDebugLoc();
  // If this can be more profitably realized as r+r, fail.
  if (SelectAddressRegReg(N, Disp, Base, DAG))
    return false;

  if (N.getOpcode() == ISD::ADD) {
    short imm = 0;
    if (isIntS16Immediate(N.getOperand(1), imm)) {
      Disp = DAG.getTargetConstant((int)imm & 0xFFFF, MVT::i32);
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N.getOperand(0))) {
        Base = DAG.getTargetFrameIndex(FI->getIndex(), N.getValueType());
      } else {
        Base = N.getOperand(0);
      }
      return true; // [r+i]
    } else if (N.getOperand(1).getOpcode() == PPCISD::Lo) {
      // Match LOAD (ADD (X, Lo(G))).
      assert(!cast<ConstantSDNode>(N.getOperand(1).getOperand(1))->getZExtValue()
             && "Cannot handle constant offsets yet!");
      Disp = N.getOperand(1).getOperand(0);  // The global address.
      assert(Disp.getOpcode() == ISD::TargetGlobalAddress ||
             Disp.getOpcode() == ISD::TargetGlobalTLSAddress ||
             Disp.getOpcode() == ISD::TargetConstantPool ||
             Disp.getOpcode() == ISD::TargetJumpTable);
      Base = N.getOperand(0);
      return true;  // [&g+r]
    }
  } else if (N.getOpcode() == ISD::OR) {
    short imm = 0;
    if (isIntS16Immediate(N.getOperand(1), imm)) {
      // If this is an or of disjoint bitfields, we can codegen this as an add
      // (for better address arithmetic) if the LHS and RHS of the OR are
      // provably disjoint.
      APInt LHSKnownZero, LHSKnownOne;
      DAG.ComputeMaskedBits(N.getOperand(0), LHSKnownZero, LHSKnownOne);

      if ((LHSKnownZero.getZExtValue()|~(uint64_t)imm) == ~0ULL) {
        // If all of the bits are known zero on the LHS or RHS, the add won't
        // carry.
        Base = N.getOperand(0);
        Disp = DAG.getTargetConstant((int)imm & 0xFFFF, MVT::i32);
        return true;
      }
    }
  } else if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    // Loading from a constant address.

    // If this address fits entirely in a 16-bit sext immediate field, codegen
    // this as "d, 0"
    short Imm;
    if (isIntS16Immediate(CN, Imm)) {
      Disp = DAG.getTargetConstant(Imm, CN->getValueType(0));
      Base = DAG.getRegister(PPCSubTarget.isPPC64() ? PPC::X0 : PPC::R0,
                             CN->getValueType(0));
      return true;
    }

    // Handle 32-bit sext immediates with LIS + addr mode.
    if (CN->getValueType(0) == MVT::i32 ||
        (int64_t)CN->getZExtValue() == (int)CN->getZExtValue()) {
      int Addr = (int)CN->getZExtValue();

      // Otherwise, break this down into an LIS + disp.
      Disp = DAG.getTargetConstant((short)Addr, MVT::i32);

      Base = DAG.getTargetConstant((Addr - (signed short)Addr) >> 16, MVT::i32);
      unsigned Opc = CN->getValueType(0) == MVT::i32 ? PPC::LIS : PPC::LIS8;
      Base = SDValue(DAG.getMachineNode(Opc, dl, CN->getValueType(0), Base), 0);
      return true;
    }
  }

  Disp = DAG.getTargetConstant(0, getPointerTy());
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N))
    Base = DAG.getTargetFrameIndex(FI->getIndex(), N.getValueType());
  else
    Base = N;
  return true;      // [r+0]
}

/// SelectAddressRegRegOnly - Given the specified addressed, force it to be
/// represented as an indexed [r+r] operation.
bool PPCTargetLowering::SelectAddressRegRegOnly(SDValue N, SDValue &Base,
                                                SDValue &Index,
                                                SelectionDAG &DAG) const {
  // Check to see if we can easily represent this as an [r+r] address.  This
  // will fail if it thinks that the address is more profitably represented as
  // reg+imm, e.g. where imm = 0.
  if (SelectAddressRegReg(N, Base, Index, DAG))
    return true;

  // If the operand is an addition, always emit this as [r+r], since this is
  // better (for code size, and execution, as the memop does the add for free)
  // than emitting an explicit add.
  if (N.getOpcode() == ISD::ADD) {
    Base = N.getOperand(0);
    Index = N.getOperand(1);
    return true;
  }

  // Otherwise, do it the hard way, using R0 as the base register.
  Base = DAG.getRegister(PPCSubTarget.isPPC64() ? PPC::X0 : PPC::R0,
                         N.getValueType());
  Index = N;
  return true;
}

/// SelectAddressRegImmShift - Returns true if the address N can be
/// represented by a base register plus a signed 14-bit displacement
/// [r+imm*4].  Suitable for use by STD and friends.
bool PPCTargetLowering::SelectAddressRegImmShift(SDValue N, SDValue &Disp,
                                                 SDValue &Base,
                                                 SelectionDAG &DAG) const {
  // FIXME dl should come from the parent load or store, not the address
  DebugLoc dl = N.getDebugLoc();
  // If this can be more profitably realized as r+r, fail.
  if (SelectAddressRegReg(N, Disp, Base, DAG))
    return false;

  if (N.getOpcode() == ISD::ADD) {
    short imm = 0;
    if (isIntS16Immediate(N.getOperand(1), imm) && (imm & 3) == 0) {
      Disp = DAG.getTargetConstant(((int)imm & 0xFFFF) >> 2, MVT::i32);
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N.getOperand(0))) {
        Base = DAG.getTargetFrameIndex(FI->getIndex(), N.getValueType());
      } else {
        Base = N.getOperand(0);
      }
      return true; // [r+i]
    } else if (N.getOperand(1).getOpcode() == PPCISD::Lo) {
      // Match LOAD (ADD (X, Lo(G))).
      assert(!cast<ConstantSDNode>(N.getOperand(1).getOperand(1))->getZExtValue()
             && "Cannot handle constant offsets yet!");
      Disp = N.getOperand(1).getOperand(0);  // The global address.
      assert(Disp.getOpcode() == ISD::TargetGlobalAddress ||
             Disp.getOpcode() == ISD::TargetConstantPool ||
             Disp.getOpcode() == ISD::TargetJumpTable);
      Base = N.getOperand(0);
      return true;  // [&g+r]
    }
  } else if (N.getOpcode() == ISD::OR) {
    short imm = 0;
    if (isIntS16Immediate(N.getOperand(1), imm) && (imm & 3) == 0) {
      // If this is an or of disjoint bitfields, we can codegen this as an add
      // (for better address arithmetic) if the LHS and RHS of the OR are
      // provably disjoint.
      APInt LHSKnownZero, LHSKnownOne;
      DAG.ComputeMaskedBits(N.getOperand(0), LHSKnownZero, LHSKnownOne);
      if ((LHSKnownZero.getZExtValue()|~(uint64_t)imm) == ~0ULL) {
        // If all of the bits are known zero on the LHS or RHS, the add won't
        // carry.
        Base = N.getOperand(0);
        Disp = DAG.getTargetConstant(((int)imm & 0xFFFF) >> 2, MVT::i32);
        return true;
      }
    }
  } else if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    // Loading from a constant address.  Verify low two bits are clear.
    if ((CN->getZExtValue() & 3) == 0) {
      // If this address fits entirely in a 14-bit sext immediate field, codegen
      // this as "d, 0"
      short Imm;
      if (isIntS16Immediate(CN, Imm)) {
        Disp = DAG.getTargetConstant((unsigned short)Imm >> 2, getPointerTy());
        Base = DAG.getRegister(PPCSubTarget.isPPC64() ? PPC::X0 : PPC::R0,
                               CN->getValueType(0));
        return true;
      }

      // Fold the low-part of 32-bit absolute addresses into addr mode.
      if (CN->getValueType(0) == MVT::i32 ||
          (int64_t)CN->getZExtValue() == (int)CN->getZExtValue()) {
        int Addr = (int)CN->getZExtValue();

        // Otherwise, break this down into an LIS + disp.
        Disp = DAG.getTargetConstant((short)Addr >> 2, MVT::i32);
        Base = DAG.getTargetConstant((Addr-(signed short)Addr) >> 16, MVT::i32);
        unsigned Opc = CN->getValueType(0) == MVT::i32 ? PPC::LIS : PPC::LIS8;
        Base = SDValue(DAG.getMachineNode(Opc, dl, CN->getValueType(0), Base),0);
        return true;
      }
    }
  }

  Disp = DAG.getTargetConstant(0, getPointerTy());
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N))
    Base = DAG.getTargetFrameIndex(FI->getIndex(), N.getValueType());
  else
    Base = N;
  return true;      // [r+0]
}


/// getPreIndexedAddressParts - returns true by value, base pointer and
/// offset pointer and addressing mode by reference if the node's address
/// can be legally represented as pre-indexed load / store address.
bool PPCTargetLowering::getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                                  SDValue &Offset,
                                                  ISD::MemIndexedMode &AM,
                                                  SelectionDAG &DAG) const {
  if (DisablePPCPreinc) return false;

  SDValue Ptr;
  EVT VT;
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    Ptr = LD->getBasePtr();
    VT = LD->getMemoryVT();

  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    Ptr = ST->getBasePtr();
    VT  = ST->getMemoryVT();
  } else
    return false;

  // PowerPC doesn't have preinc load/store instructions for vectors.
  if (VT.isVector())
    return false;

  if (SelectAddressRegReg(Ptr, Offset, Base, DAG)) {
    AM = ISD::PRE_INC;
    return true;
  }

  // LDU/STU use reg+imm*4, others use reg+imm.
  if (VT != MVT::i64) {
    // reg + imm
    if (!SelectAddressRegImm(Ptr, Offset, Base, DAG))
      return false;
  } else {
    // reg + imm * 4.
    if (!SelectAddressRegImmShift(Ptr, Offset, Base, DAG))
      return false;
  }

  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    // PPC64 doesn't have lwau, but it does have lwaux.  Reject preinc load of
    // sext i32 to i64 when addr mode is r+i.
    if (LD->getValueType(0) == MVT::i64 && LD->getMemoryVT() == MVT::i32 &&
        LD->getExtensionType() == ISD::SEXTLOAD &&
        isa<ConstantSDNode>(Offset))
      return false;
  }

  AM = ISD::PRE_INC;
  return true;
}

//===----------------------------------------------------------------------===//
//  LowerOperation implementation
//===----------------------------------------------------------------------===//

/// GetLabelAccessInfo - Return true if we should reference labels using a
/// PICBase, set the HiOpFlags and LoOpFlags to the target MO flags.
static bool GetLabelAccessInfo(const TargetMachine &TM, unsigned &HiOpFlags,
                               unsigned &LoOpFlags, const GlobalValue *GV = 0) {
  HiOpFlags = PPCII::MO_HA16;
  LoOpFlags = PPCII::MO_LO16;

  // Don't use the pic base if not in PIC relocation model.  Or if we are on a
  // non-darwin platform.  We don't support PIC on other platforms yet.
  bool isPIC = TM.getRelocationModel() == Reloc::PIC_ &&
               TM.getSubtarget<PPCSubtarget>().isDarwin();
  if (isPIC) {
    HiOpFlags |= PPCII::MO_PIC_FLAG;
    LoOpFlags |= PPCII::MO_PIC_FLAG;
  }

  // If this is a reference to a global value that requires a non-lazy-ptr, make
  // sure that instruction lowering adds it.
  if (GV && TM.getSubtarget<PPCSubtarget>().hasLazyResolverStub(GV, TM)) {
    HiOpFlags |= PPCII::MO_NLP_FLAG;
    LoOpFlags |= PPCII::MO_NLP_FLAG;

    if (GV->hasHiddenVisibility()) {
      HiOpFlags |= PPCII::MO_NLP_HIDDEN_FLAG;
      LoOpFlags |= PPCII::MO_NLP_HIDDEN_FLAG;
    }
  }

  return isPIC;
}

static SDValue LowerLabelRef(SDValue HiPart, SDValue LoPart, bool isPIC,
                             SelectionDAG &DAG) {
  EVT PtrVT = HiPart.getValueType();
  SDValue Zero = DAG.getConstant(0, PtrVT);
  DebugLoc DL = HiPart.getDebugLoc();

  SDValue Hi = DAG.getNode(PPCISD::Hi, DL, PtrVT, HiPart, Zero);
  SDValue Lo = DAG.getNode(PPCISD::Lo, DL, PtrVT, LoPart, Zero);

  // With PIC, the first instruction is actually "GR+hi(&G)".
  if (isPIC)
    Hi = DAG.getNode(ISD::ADD, DL, PtrVT,
                     DAG.getNode(PPCISD::GlobalBaseReg, DL, PtrVT), Hi);

  // Generate non-pic code that has direct accesses to the constant pool.
  // The address of the global is just (hi(&g)+lo(&g)).
  return DAG.getNode(ISD::ADD, DL, PtrVT, Hi, Lo);
}

SDValue PPCTargetLowering::LowerConstantPool(SDValue Op,
                                             SelectionDAG &DAG) const {
  EVT PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  const Constant *C = CP->getConstVal();

  // 64-bit SVR4 ABI code is always position-independent.
  // The actual address of the GlobalValue is stored in the TOC.
  if (PPCSubTarget.isSVR4ABI() && PPCSubTarget.isPPC64()) {
    SDValue GA = DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment(), 0);
    return DAG.getNode(PPCISD::TOC_ENTRY, CP->getDebugLoc(), MVT::i64, GA,
                       DAG.getRegister(PPC::X2, MVT::i64));
  }

  unsigned MOHiFlag, MOLoFlag;
  bool isPIC = GetLabelAccessInfo(DAG.getTarget(), MOHiFlag, MOLoFlag);
  SDValue CPIHi =
    DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment(), 0, MOHiFlag);
  SDValue CPILo =
    DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment(), 0, MOLoFlag);
  return LowerLabelRef(CPIHi, CPILo, isPIC, DAG);
}

SDValue PPCTargetLowering::LowerJumpTable(SDValue Op, SelectionDAG &DAG) const {
  EVT PtrVT = Op.getValueType();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);

  // 64-bit SVR4 ABI code is always position-independent.
  // The actual address of the GlobalValue is stored in the TOC.
  if (PPCSubTarget.isSVR4ABI() && PPCSubTarget.isPPC64()) {
    SDValue GA = DAG.getTargetJumpTable(JT->getIndex(), PtrVT);
    return DAG.getNode(PPCISD::TOC_ENTRY, JT->getDebugLoc(), MVT::i64, GA,
                       DAG.getRegister(PPC::X2, MVT::i64));
  }

  unsigned MOHiFlag, MOLoFlag;
  bool isPIC = GetLabelAccessInfo(DAG.getTarget(), MOHiFlag, MOLoFlag);
  SDValue JTIHi = DAG.getTargetJumpTable(JT->getIndex(), PtrVT, MOHiFlag);
  SDValue JTILo = DAG.getTargetJumpTable(JT->getIndex(), PtrVT, MOLoFlag);
  return LowerLabelRef(JTIHi, JTILo, isPIC, DAG);
}

SDValue PPCTargetLowering::LowerBlockAddress(SDValue Op,
                                             SelectionDAG &DAG) const {
  EVT PtrVT = Op.getValueType();

  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();

  unsigned MOHiFlag, MOLoFlag;
  bool isPIC = GetLabelAccessInfo(DAG.getTarget(), MOHiFlag, MOLoFlag);
  SDValue TgtBAHi = DAG.getTargetBlockAddress(BA, PtrVT, 0, MOHiFlag);
  SDValue TgtBALo = DAG.getTargetBlockAddress(BA, PtrVT, 0, MOLoFlag);
  return LowerLabelRef(TgtBAHi, TgtBALo, isPIC, DAG);
}

SDValue PPCTargetLowering::LowerGlobalTLSAddress(SDValue Op,
                                              SelectionDAG &DAG) const {

  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);
  DebugLoc dl = GA->getDebugLoc();
  const GlobalValue *GV = GA->getGlobal();
  EVT PtrVT = getPointerTy();
  bool is64bit = PPCSubTarget.isPPC64();

  TLSModel::Model model = getTargetMachine().getTLSModel(GV);

  SDValue TGAHi = DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                             PPCII::MO_TPREL16_HA);
  SDValue TGALo = DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                             PPCII::MO_TPREL16_LO);

  if (model != TLSModel::LocalExec)
    llvm_unreachable("only local-exec TLS mode supported");
  SDValue TLSReg = DAG.getRegister(is64bit ? PPC::X13 : PPC::R2,
                                   is64bit ? MVT::i64 : MVT::i32);
  SDValue Hi = DAG.getNode(PPCISD::Hi, dl, PtrVT, TGAHi, TLSReg);
  return DAG.getNode(PPCISD::Lo, dl, PtrVT, TGALo, Hi);
}

SDValue PPCTargetLowering::LowerGlobalAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  EVT PtrVT = Op.getValueType();
  GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
  DebugLoc DL = GSDN->getDebugLoc();
  const GlobalValue *GV = GSDN->getGlobal();

  // 64-bit SVR4 ABI code is always position-independent.
  // The actual address of the GlobalValue is stored in the TOC.
  if (PPCSubTarget.isSVR4ABI() && PPCSubTarget.isPPC64()) {
    SDValue GA = DAG.getTargetGlobalAddress(GV, DL, PtrVT, GSDN->getOffset());
    return DAG.getNode(PPCISD::TOC_ENTRY, DL, MVT::i64, GA,
                       DAG.getRegister(PPC::X2, MVT::i64));
  }

  unsigned MOHiFlag, MOLoFlag;
  bool isPIC = GetLabelAccessInfo(DAG.getTarget(), MOHiFlag, MOLoFlag, GV);

  SDValue GAHi =
    DAG.getTargetGlobalAddress(GV, DL, PtrVT, GSDN->getOffset(), MOHiFlag);
  SDValue GALo =
    DAG.getTargetGlobalAddress(GV, DL, PtrVT, GSDN->getOffset(), MOLoFlag);

  SDValue Ptr = LowerLabelRef(GAHi, GALo, isPIC, DAG);

  // If the global reference is actually to a non-lazy-pointer, we have to do an
  // extra load to get the address of the global.
  if (MOHiFlag & PPCII::MO_NLP_FLAG)
    Ptr = DAG.getLoad(PtrVT, DL, DAG.getEntryNode(), Ptr, MachinePointerInfo(),
                      false, false, false, 0);
  return Ptr;
}

SDValue PPCTargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  DebugLoc dl = Op.getDebugLoc();

  // If we're comparing for equality to zero, expose the fact that this is
  // implented as a ctlz/srl pair on ppc, so that the dag combiner can
  // fold the new nodes.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
    if (C->isNullValue() && CC == ISD::SETEQ) {
      EVT VT = Op.getOperand(0).getValueType();
      SDValue Zext = Op.getOperand(0);
      if (VT.bitsLT(MVT::i32)) {
        VT = MVT::i32;
        Zext = DAG.getNode(ISD::ZERO_EXTEND, dl, VT, Op.getOperand(0));
      }
      unsigned Log2b = Log2_32(VT.getSizeInBits());
      SDValue Clz = DAG.getNode(ISD::CTLZ, dl, VT, Zext);
      SDValue Scc = DAG.getNode(ISD::SRL, dl, VT, Clz,
                                DAG.getConstant(Log2b, MVT::i32));
      return DAG.getNode(ISD::TRUNCATE, dl, MVT::i32, Scc);
    }
    // Leave comparisons against 0 and -1 alone for now, since they're usually
    // optimized.  FIXME: revisit this when we can custom lower all setcc
    // optimizations.
    if (C->isAllOnesValue() || C->isNullValue())
      return SDValue();
  }

  // If we have an integer seteq/setne, turn it into a compare against zero
  // by xor'ing the rhs with the lhs, which is faster than setting a
  // condition register, reading it back out, and masking the correct bit.  The
  // normal approach here uses sub to do this instead of xor.  Using xor exposes
  // the result to other bit-twiddling opportunities.
  EVT LHSVT = Op.getOperand(0).getValueType();
  if (LHSVT.isInteger() && (CC == ISD::SETEQ || CC == ISD::SETNE)) {
    EVT VT = Op.getValueType();
    SDValue Sub = DAG.getNode(ISD::XOR, dl, LHSVT, Op.getOperand(0),
                                Op.getOperand(1));
    return DAG.getSetCC(dl, VT, Sub, DAG.getConstant(0, LHSVT), CC);
  }
  return SDValue();
}

SDValue PPCTargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG,
                                      const PPCSubtarget &Subtarget) const {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getValueType(0);
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  SDValue InChain = Node->getOperand(0);
  SDValue VAListPtr = Node->getOperand(1);
  const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  DebugLoc dl = Node->getDebugLoc();

  assert(!Subtarget.isPPC64() && "LowerVAARG is PPC32 only");

  // gpr_index
  SDValue GprIndex = DAG.getExtLoad(ISD::ZEXTLOAD, dl, MVT::i32, InChain,
                                    VAListPtr, MachinePointerInfo(SV), MVT::i8,
                                    false, false, 0);
  InChain = GprIndex.getValue(1);

  if (VT == MVT::i64) {
    // Check if GprIndex is even
    SDValue GprAnd = DAG.getNode(ISD::AND, dl, MVT::i32, GprIndex,
                                 DAG.getConstant(1, MVT::i32));
    SDValue CC64 = DAG.getSetCC(dl, MVT::i32, GprAnd,
                                DAG.getConstant(0, MVT::i32), ISD::SETNE);
    SDValue GprIndexPlusOne = DAG.getNode(ISD::ADD, dl, MVT::i32, GprIndex,
                                          DAG.getConstant(1, MVT::i32));
    // Align GprIndex to be even if it isn't
    GprIndex = DAG.getNode(ISD::SELECT, dl, MVT::i32, CC64, GprIndexPlusOne,
                           GprIndex);
  }

  // fpr index is 1 byte after gpr
  SDValue FprPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                               DAG.getConstant(1, MVT::i32));

  // fpr
  SDValue FprIndex = DAG.getExtLoad(ISD::ZEXTLOAD, dl, MVT::i32, InChain,
                                    FprPtr, MachinePointerInfo(SV), MVT::i8,
                                    false, false, 0);
  InChain = FprIndex.getValue(1);

  SDValue RegSaveAreaPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                                       DAG.getConstant(8, MVT::i32));

  SDValue OverflowAreaPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                                        DAG.getConstant(4, MVT::i32));

  // areas
  SDValue OverflowArea = DAG.getLoad(MVT::i32, dl, InChain, OverflowAreaPtr,
                                     MachinePointerInfo(), false, false,
                                     false, 0);
  InChain = OverflowArea.getValue(1);

  SDValue RegSaveArea = DAG.getLoad(MVT::i32, dl, InChain, RegSaveAreaPtr,
                                    MachinePointerInfo(), false, false,
                                    false, 0);
  InChain = RegSaveArea.getValue(1);

  // select overflow_area if index > 8
  SDValue CC = DAG.getSetCC(dl, MVT::i32, VT.isInteger() ? GprIndex : FprIndex,
                            DAG.getConstant(8, MVT::i32), ISD::SETLT);

  // adjustment constant gpr_index * 4/8
  SDValue RegConstant = DAG.getNode(ISD::MUL, dl, MVT::i32,
                                    VT.isInteger() ? GprIndex : FprIndex,
                                    DAG.getConstant(VT.isInteger() ? 4 : 8,
                                                    MVT::i32));

  // OurReg = RegSaveArea + RegConstant
  SDValue OurReg = DAG.getNode(ISD::ADD, dl, PtrVT, RegSaveArea,
                               RegConstant);

  // Floating types are 32 bytes into RegSaveArea
  if (VT.isFloatingPoint())
    OurReg = DAG.getNode(ISD::ADD, dl, PtrVT, OurReg,
                         DAG.getConstant(32, MVT::i32));

  // increase {f,g}pr_index by 1 (or 2 if VT is i64)
  SDValue IndexPlus1 = DAG.getNode(ISD::ADD, dl, MVT::i32,
                                   VT.isInteger() ? GprIndex : FprIndex,
                                   DAG.getConstant(VT == MVT::i64 ? 2 : 1,
                                                   MVT::i32));

  InChain = DAG.getTruncStore(InChain, dl, IndexPlus1,
                              VT.isInteger() ? VAListPtr : FprPtr,
                              MachinePointerInfo(SV),
                              MVT::i8, false, false, 0);

  // determine if we should load from reg_save_area or overflow_area
  SDValue Result = DAG.getNode(ISD::SELECT, dl, PtrVT, CC, OurReg, OverflowArea);

  // increase overflow_area by 4/8 if gpr/fpr > 8
  SDValue OverflowAreaPlusN = DAG.getNode(ISD::ADD, dl, PtrVT, OverflowArea,
                                          DAG.getConstant(VT.isInteger() ? 4 : 8,
                                          MVT::i32));

  OverflowArea = DAG.getNode(ISD::SELECT, dl, MVT::i32, CC, OverflowArea,
                             OverflowAreaPlusN);

  InChain = DAG.getTruncStore(InChain, dl, OverflowArea,
                              OverflowAreaPtr,
                              MachinePointerInfo(),
                              MVT::i32, false, false, 0);

  return DAG.getLoad(VT, dl, InChain, Result, MachinePointerInfo(),
                     false, false, false, 0);
}

SDValue PPCTargetLowering::LowerADJUST_TRAMPOLINE(SDValue Op,
                                                  SelectionDAG &DAG) const {
  return Op.getOperand(0);
}

SDValue PPCTargetLowering::LowerINIT_TRAMPOLINE(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Trmp = Op.getOperand(1); // trampoline
  SDValue FPtr = Op.getOperand(2); // nested function
  SDValue Nest = Op.getOperand(3); // 'nest' parameter value
  DebugLoc dl = Op.getDebugLoc();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  bool isPPC64 = (PtrVT == MVT::i64);
  Type *IntPtrTy =
    DAG.getTargetLoweringInfo().getDataLayout()->getIntPtrType(
                                                             *DAG.getContext());

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;

  Entry.Ty = IntPtrTy;
  Entry.Node = Trmp; Args.push_back(Entry);

  // TrampSize == (isPPC64 ? 48 : 40);
  Entry.Node = DAG.getConstant(isPPC64 ? 48 : 40,
                               isPPC64 ? MVT::i64 : MVT::i32);
  Args.push_back(Entry);

  Entry.Node = FPtr; Args.push_back(Entry);
  Entry.Node = Nest; Args.push_back(Entry);

  // Lower to a call to __trampoline_setup(Trmp, TrampSize, FPtr, ctx_reg)
  TargetLowering::CallLoweringInfo CLI(Chain,
                                       Type::getVoidTy(*DAG.getContext()),
                                       false, false, false, false, 0,
                                       CallingConv::C,
                /*isTailCall=*/false,
                                       /*doesNotRet=*/false,
                                       /*isReturnValueUsed=*/true,
                DAG.getExternalSymbol("__trampoline_setup", PtrVT),
                Args, DAG, dl);
  std::pair<SDValue, SDValue> CallResult = LowerCallTo(CLI);

  return CallResult.second;
}

SDValue PPCTargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG,
                                        const PPCSubtarget &Subtarget) const {
  MachineFunction &MF = DAG.getMachineFunction();
  PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();

  DebugLoc dl = Op.getDebugLoc();

  if (Subtarget.isDarwinABI() || Subtarget.isPPC64()) {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
    SDValue FR = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(), PtrVT);
    const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
    return DAG.getStore(Op.getOperand(0), dl, FR, Op.getOperand(1),
                        MachinePointerInfo(SV),
                        false, false, 0);
  }

  // For the 32-bit SVR4 ABI we follow the layout of the va_list struct.
  // We suppose the given va_list is already allocated.
  //
  // typedef struct {
  //  char gpr;     /* index into the array of 8 GPRs
  //                 * stored in the register save area
  //                 * gpr=0 corresponds to r3,
  //                 * gpr=1 to r4, etc.
  //                 */
  //  char fpr;     /* index into the array of 8 FPRs
  //                 * stored in the register save area
  //                 * fpr=0 corresponds to f1,
  //                 * fpr=1 to f2, etc.
  //                 */
  //  char *overflow_arg_area;
  //                /* location on stack that holds
  //                 * the next overflow argument
  //                 */
  //  char *reg_save_area;
  //               /* where r3:r10 and f1:f8 (if saved)
  //                * are stored
  //                */
  // } va_list[1];


  SDValue ArgGPR = DAG.getConstant(FuncInfo->getVarArgsNumGPR(), MVT::i32);
  SDValue ArgFPR = DAG.getConstant(FuncInfo->getVarArgsNumFPR(), MVT::i32);


  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

  SDValue StackOffsetFI = DAG.getFrameIndex(FuncInfo->getVarArgsStackOffset(),
                                            PtrVT);
  SDValue FR = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 PtrVT);

  uint64_t FrameOffset = PtrVT.getSizeInBits()/8;
  SDValue ConstFrameOffset = DAG.getConstant(FrameOffset, PtrVT);

  uint64_t StackOffset = PtrVT.getSizeInBits()/8 - 1;
  SDValue ConstStackOffset = DAG.getConstant(StackOffset, PtrVT);

  uint64_t FPROffset = 1;
  SDValue ConstFPROffset = DAG.getConstant(FPROffset, PtrVT);

  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  // Store first byte : number of int regs
  SDValue firstStore = DAG.getTruncStore(Op.getOperand(0), dl, ArgGPR,
                                         Op.getOperand(1),
                                         MachinePointerInfo(SV),
                                         MVT::i8, false, false, 0);
  uint64_t nextOffset = FPROffset;
  SDValue nextPtr = DAG.getNode(ISD::ADD, dl, PtrVT, Op.getOperand(1),
                                  ConstFPROffset);

  // Store second byte : number of float regs
  SDValue secondStore =
    DAG.getTruncStore(firstStore, dl, ArgFPR, nextPtr,
                      MachinePointerInfo(SV, nextOffset), MVT::i8,
                      false, false, 0);
  nextOffset += StackOffset;
  nextPtr = DAG.getNode(ISD::ADD, dl, PtrVT, nextPtr, ConstStackOffset);

  // Store second word : arguments given on stack
  SDValue thirdStore =
    DAG.getStore(secondStore, dl, StackOffsetFI, nextPtr,
                 MachinePointerInfo(SV, nextOffset),
                 false, false, 0);
  nextOffset += FrameOffset;
  nextPtr = DAG.getNode(ISD::ADD, dl, PtrVT, nextPtr, ConstFrameOffset);

  // Store third word : arguments given in registers
  return DAG.getStore(thirdStore, dl, FR, nextPtr,
                      MachinePointerInfo(SV, nextOffset),
                      false, false, 0);

}

#include "PPCGenCallingConv.inc"

static bool CC_PPC_SVR4_Custom_Dummy(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                     CCValAssign::LocInfo &LocInfo,
                                     ISD::ArgFlagsTy &ArgFlags,
                                     CCState &State) {
  return true;
}

static bool CC_PPC_SVR4_Custom_AlignArgRegs(unsigned &ValNo, MVT &ValVT,
                                            MVT &LocVT,
                                            CCValAssign::LocInfo &LocInfo,
                                            ISD::ArgFlagsTy &ArgFlags,
                                            CCState &State) {
  static const uint16_t ArgRegs[] = {
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  const unsigned NumArgRegs = array_lengthof(ArgRegs);

  unsigned RegNum = State.getFirstUnallocated(ArgRegs, NumArgRegs);

  // Skip one register if the first unallocated register has an even register
  // number and there are still argument registers available which have not been
  // allocated yet. RegNum is actually an index into ArgRegs, which means we
  // need to skip a register if RegNum is odd.
  if (RegNum != NumArgRegs && RegNum % 2 == 1) {
    State.AllocateReg(ArgRegs[RegNum]);
  }

  // Always return false here, as this function only makes sure that the first
  // unallocated register has an odd register number and does not actually
  // allocate a register for the current argument.
  return false;
}

static bool CC_PPC_SVR4_Custom_AlignFPArgRegs(unsigned &ValNo, MVT &ValVT,
                                              MVT &LocVT,
                                              CCValAssign::LocInfo &LocInfo,
                                              ISD::ArgFlagsTy &ArgFlags,
                                              CCState &State) {
  static const uint16_t ArgRegs[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8
  };

  const unsigned NumArgRegs = array_lengthof(ArgRegs);

  unsigned RegNum = State.getFirstUnallocated(ArgRegs, NumArgRegs);

  // If there is only one Floating-point register left we need to put both f64
  // values of a split ppc_fp128 value on the stack.
  if (RegNum != NumArgRegs && ArgRegs[RegNum] == PPC::F8) {
    State.AllocateReg(ArgRegs[RegNum]);
  }

  // Always return false here, as this function only makes sure that the two f64
  // values a ppc_fp128 value is split into are both passed in registers or both
  // passed on the stack and does not actually allocate a register for the
  // current argument.
  return false;
}

/// GetFPR - Get the set of FP registers that should be allocated for arguments,
/// on Darwin.
static const uint16_t *GetFPR() {
  static const uint16_t FPR[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
  };

  return FPR;
}

/// CalculateStackSlotSize - Calculates the size reserved for this argument on
/// the stack.
static unsigned CalculateStackSlotSize(EVT ArgVT, ISD::ArgFlagsTy Flags,
                                       unsigned PtrByteSize) {
  unsigned ArgSize = ArgVT.getSizeInBits()/8;
  if (Flags.isByVal())
    ArgSize = Flags.getByValSize();
  ArgSize = ((ArgSize + PtrByteSize - 1)/PtrByteSize) * PtrByteSize;

  return ArgSize;
}

SDValue
PPCTargetLowering::LowerFormalArguments(SDValue Chain,
                                        CallingConv::ID CallConv, bool isVarArg,
                                        const SmallVectorImpl<ISD::InputArg>
                                          &Ins,
                                        DebugLoc dl, SelectionDAG &DAG,
                                        SmallVectorImpl<SDValue> &InVals)
                                          const {
  if (PPCSubTarget.isSVR4ABI()) {
    if (PPCSubTarget.isPPC64())
      return LowerFormalArguments_64SVR4(Chain, CallConv, isVarArg, Ins,
                                         dl, DAG, InVals);
    else
      return LowerFormalArguments_32SVR4(Chain, CallConv, isVarArg, Ins,
                                         dl, DAG, InVals);
  } else {
    return LowerFormalArguments_Darwin(Chain, CallConv, isVarArg, Ins,
                                       dl, DAG, InVals);
  }
}

SDValue
PPCTargetLowering::LowerFormalArguments_32SVR4(
                                      SDValue Chain,
                                      CallingConv::ID CallConv, bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg>
                                        &Ins,
                                      DebugLoc dl, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {

  // 32-bit SVR4 ABI Stack Frame Layout:
  //              +-----------------------------------+
  //        +-->  |            Back chain             |
  //        |     +-----------------------------------+
  //        |     | Floating-point register save area |
  //        |     +-----------------------------------+
  //        |     |    General register save area     |
  //        |     +-----------------------------------+
  //        |     |          CR save word             |
  //        |     +-----------------------------------+
  //        |     |         VRSAVE save word          |
  //        |     +-----------------------------------+
  //        |     |         Alignment padding         |
  //        |     +-----------------------------------+
  //        |     |     Vector register save area     |
  //        |     +-----------------------------------+
  //        |     |       Local variable space        |
  //        |     +-----------------------------------+
  //        |     |        Parameter list area        |
  //        |     +-----------------------------------+
  //        |     |           LR save word            |
  //        |     +-----------------------------------+
  // SP-->  +---  |            Back chain             |
  //              +-----------------------------------+
  //
  // Specifications:
  //   System V Application Binary Interface PowerPC Processor Supplement
  //   AltiVec Technology Programming Interface Manual

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  // Potential tail calls could cause overwriting of argument stack slots.
  bool isImmutable = !(getTargetMachine().Options.GuaranteedTailCallOpt &&
                       (CallConv == CallingConv::Fast));
  unsigned PtrByteSize = 4;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  // Reserve space for the linkage area on the stack.
  CCInfo.AllocateStack(PPCFrameLowering::getLinkageSize(false, false), PtrByteSize);

  CCInfo.AnalyzeFormalArguments(Ins, CC_PPC_SVR4);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    // Arguments stored in registers.
    if (VA.isRegLoc()) {
      const TargetRegisterClass *RC;
      EVT ValVT = VA.getValVT();

      switch (ValVT.getSimpleVT().SimpleTy) {
        default:
          llvm_unreachable("ValVT not supported by formal arguments Lowering");
        case MVT::i32:
          RC = &PPC::GPRCRegClass;
          break;
        case MVT::f32:
          RC = &PPC::F4RCRegClass;
          break;
        case MVT::f64:
          RC = &PPC::F8RCRegClass;
          break;
        case MVT::v16i8:
        case MVT::v8i16:
        case MVT::v4i32:
        case MVT::v4f32:
          RC = &PPC::VRRCRegClass;
          break;
      }

      // Transform the arguments stored in physical registers into virtual ones.
      unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, ValVT);

      InVals.push_back(ArgValue);
    } else {
      // Argument stored in memory.
      assert(VA.isMemLoc());

      unsigned ArgSize = VA.getLocVT().getSizeInBits() / 8;
      int FI = MFI->CreateFixedObject(ArgSize, VA.getLocMemOffset(),
                                      isImmutable);

      // Create load nodes to retrieve arguments from the stack.
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      InVals.push_back(DAG.getLoad(VA.getValVT(), dl, Chain, FIN,
                                   MachinePointerInfo(),
                                   false, false, false, 0));
    }
  }

  // Assign locations to all of the incoming aggregate by value arguments.
  // Aggregates passed by value are stored in the local variable space of the
  // caller's stack frame, right above the parameter list area.
  SmallVector<CCValAssign, 16> ByValArgLocs;
  CCState CCByValInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                      getTargetMachine(), ByValArgLocs, *DAG.getContext());

  // Reserve stack space for the allocations in CCInfo.
  CCByValInfo.AllocateStack(CCInfo.getNextStackOffset(), PtrByteSize);

  CCByValInfo.AnalyzeFormalArguments(Ins, CC_PPC_SVR4_ByVal);

  // Area that is at least reserved in the caller of this function.
  unsigned MinReservedArea = CCByValInfo.getNextStackOffset();

  // Set the size that is at least reserved in caller of this function.  Tail
  // call optimized function's reserved stack space needs to be aligned so that
  // taking the difference between two stack areas will result in an aligned
  // stack.
  PPCFunctionInfo *FI = MF.getInfo<PPCFunctionInfo>();

  MinReservedArea =
    std::max(MinReservedArea,
             PPCFrameLowering::getMinCallFrameSize(false, false));

  unsigned TargetAlign = DAG.getMachineFunction().getTarget().getFrameLowering()->
    getStackAlignment();
  unsigned AlignMask = TargetAlign-1;
  MinReservedArea = (MinReservedArea + AlignMask) & ~AlignMask;

  FI->setMinReservedArea(MinReservedArea);

  SmallVector<SDValue, 8> MemOps;

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (isVarArg) {
    static const uint16_t GPArgRegs[] = {
      PPC::R3, PPC::R4, PPC::R5, PPC::R6,
      PPC::R7, PPC::R8, PPC::R9, PPC::R10,
    };
    const unsigned NumGPArgRegs = array_lengthof(GPArgRegs);

    static const uint16_t FPArgRegs[] = {
      PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
      PPC::F8
    };
    const unsigned NumFPArgRegs = array_lengthof(FPArgRegs);

    FuncInfo->setVarArgsNumGPR(CCInfo.getFirstUnallocated(GPArgRegs,
                                                          NumGPArgRegs));
    FuncInfo->setVarArgsNumFPR(CCInfo.getFirstUnallocated(FPArgRegs,
                                                          NumFPArgRegs));

    // Make room for NumGPArgRegs and NumFPArgRegs.
    int Depth = NumGPArgRegs * PtrVT.getSizeInBits()/8 +
                NumFPArgRegs * EVT(MVT::f64).getSizeInBits()/8;

    FuncInfo->setVarArgsStackOffset(
      MFI->CreateFixedObject(PtrVT.getSizeInBits()/8,
                             CCInfo.getNextStackOffset(), true));

    FuncInfo->setVarArgsFrameIndex(MFI->CreateStackObject(Depth, 8, false));
    SDValue FIN = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(), PtrVT);

    // The fixed integer arguments of a variadic function are stored to the
    // VarArgsFrameIndex on the stack so that they may be loaded by deferencing
    // the result of va_next.
    for (unsigned GPRIndex = 0; GPRIndex != NumGPArgRegs; ++GPRIndex) {
      // Get an existing live-in vreg, or add a new one.
      unsigned VReg = MF.getRegInfo().getLiveInVirtReg(GPArgRegs[GPRIndex]);
      if (!VReg)
        VReg = MF.addLiveIn(GPArgRegs[GPRIndex], &PPC::GPRCRegClass);

      SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, PtrVT);
      SDValue Store = DAG.getStore(Val.getValue(1), dl, Val, FIN,
                                   MachinePointerInfo(), false, false, 0);
      MemOps.push_back(Store);
      // Increment the address by four for the next argument to store
      SDValue PtrOff = DAG.getConstant(PtrVT.getSizeInBits()/8, PtrVT);
      FIN = DAG.getNode(ISD::ADD, dl, PtrOff.getValueType(), FIN, PtrOff);
    }

    // FIXME 32-bit SVR4: We only need to save FP argument registers if CR bit 6
    // is set.
    // The double arguments are stored to the VarArgsFrameIndex
    // on the stack.
    for (unsigned FPRIndex = 0; FPRIndex != NumFPArgRegs; ++FPRIndex) {
      // Get an existing live-in vreg, or add a new one.
      unsigned VReg = MF.getRegInfo().getLiveInVirtReg(FPArgRegs[FPRIndex]);
      if (!VReg)
        VReg = MF.addLiveIn(FPArgRegs[FPRIndex], &PPC::F8RCRegClass);

      SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, MVT::f64);
      SDValue Store = DAG.getStore(Val.getValue(1), dl, Val, FIN,
                                   MachinePointerInfo(), false, false, 0);
      MemOps.push_back(Store);
      // Increment the address by eight for the next argument to store
      SDValue PtrOff = DAG.getConstant(EVT(MVT::f64).getSizeInBits()/8,
                                         PtrVT);
      FIN = DAG.getNode(ISD::ADD, dl, PtrOff.getValueType(), FIN, PtrOff);
    }
  }

  if (!MemOps.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl,
                        MVT::Other, &MemOps[0], MemOps.size());

  return Chain;
}

// PPC64 passes i8, i16, and i32 values in i64 registers. Promote
// value to MVT::i64 and then truncate to the correct register size.
SDValue
PPCTargetLowering::extendArgForPPC64(ISD::ArgFlagsTy Flags, EVT ObjectVT,
                                     SelectionDAG &DAG, SDValue ArgVal,
                                     DebugLoc dl) const {
  if (Flags.isSExt())
    ArgVal = DAG.getNode(ISD::AssertSext, dl, MVT::i64, ArgVal,
                         DAG.getValueType(ObjectVT));
  else if (Flags.isZExt())
    ArgVal = DAG.getNode(ISD::AssertZext, dl, MVT::i64, ArgVal,
                         DAG.getValueType(ObjectVT));
  
  return DAG.getNode(ISD::TRUNCATE, dl, MVT::i32, ArgVal);
}

// Set the size that is at least reserved in caller of this function.  Tail
// call optimized functions' reserved stack space needs to be aligned so that
// taking the difference between two stack areas will result in an aligned
// stack.
void
PPCTargetLowering::setMinReservedArea(MachineFunction &MF, SelectionDAG &DAG,
                                      unsigned nAltivecParamsAtEnd,
                                      unsigned MinReservedArea,
                                      bool isPPC64) const {
  PPCFunctionInfo *FI = MF.getInfo<PPCFunctionInfo>();
  // Add the Altivec parameters at the end, if needed.
  if (nAltivecParamsAtEnd) {
    MinReservedArea = ((MinReservedArea+15)/16)*16;
    MinReservedArea += 16*nAltivecParamsAtEnd;
  }
  MinReservedArea =
    std::max(MinReservedArea,
             PPCFrameLowering::getMinCallFrameSize(isPPC64, true));
  unsigned TargetAlign
    = DAG.getMachineFunction().getTarget().getFrameLowering()->
        getStackAlignment();
  unsigned AlignMask = TargetAlign-1;
  MinReservedArea = (MinReservedArea + AlignMask) & ~AlignMask;
  FI->setMinReservedArea(MinReservedArea);
}

SDValue
PPCTargetLowering::LowerFormalArguments_64SVR4(
                                      SDValue Chain,
                                      CallingConv::ID CallConv, bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg>
                                        &Ins,
                                      DebugLoc dl, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {
  // TODO: add description of PPC stack frame format, or at least some docs.
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  // Potential tail calls could cause overwriting of argument stack slots.
  bool isImmutable = !(getTargetMachine().Options.GuaranteedTailCallOpt &&
                       (CallConv == CallingConv::Fast));
  unsigned PtrByteSize = 8;

  unsigned ArgOffset = PPCFrameLowering::getLinkageSize(true, true);
  // Area that is at least reserved in caller of this function.
  unsigned MinReservedArea = ArgOffset;

  static const uint16_t GPR[] = {
    PPC::X3, PPC::X4, PPC::X5, PPC::X6,
    PPC::X7, PPC::X8, PPC::X9, PPC::X10,
  };

  static const uint16_t *FPR = GetFPR();

  static const uint16_t VR[] = {
    PPC::V2, PPC::V3, PPC::V4, PPC::V5, PPC::V6, PPC::V7, PPC::V8,
    PPC::V9, PPC::V10, PPC::V11, PPC::V12, PPC::V13
  };

  const unsigned Num_GPR_Regs = array_lengthof(GPR);
  const unsigned Num_FPR_Regs = 13;
  const unsigned Num_VR_Regs  = array_lengthof(VR);

  unsigned GPR_idx = 0, FPR_idx = 0, VR_idx = 0;

  // Add DAG nodes to load the arguments or copy them out of registers.  On
  // entry to a function on PPC, the arguments start after the linkage area,
  // although the first ones are often in registers.

  SmallVector<SDValue, 8> MemOps;
  unsigned nAltivecParamsAtEnd = 0;
  Function::const_arg_iterator FuncArg = MF.getFunction()->arg_begin();
  for (unsigned ArgNo = 0, e = Ins.size(); ArgNo != e; ++ArgNo, ++FuncArg) {
    SDValue ArgVal;
    bool needsLoad = false;
    EVT ObjectVT = Ins[ArgNo].VT;
    unsigned ObjSize = ObjectVT.getSizeInBits()/8;
    unsigned ArgSize = ObjSize;
    ISD::ArgFlagsTy Flags = Ins[ArgNo].Flags;

    unsigned CurArgOffset = ArgOffset;

    // Varargs or 64 bit Altivec parameters are padded to a 16 byte boundary.
    if (ObjectVT==MVT::v4f32 || ObjectVT==MVT::v4i32 ||
        ObjectVT==MVT::v8i16 || ObjectVT==MVT::v16i8) {
      if (isVarArg) {
        MinReservedArea = ((MinReservedArea+15)/16)*16;
        MinReservedArea += CalculateStackSlotSize(ObjectVT,
                                                  Flags,
                                                  PtrByteSize);
      } else
        nAltivecParamsAtEnd++;
    } else
      // Calculate min reserved area.
      MinReservedArea += CalculateStackSlotSize(Ins[ArgNo].VT,
                                                Flags,
                                                PtrByteSize);

    // FIXME the codegen can be much improved in some cases.
    // We do not have to keep everything in memory.
    if (Flags.isByVal()) {
      // ObjSize is the true size, ArgSize rounded up to multiple of registers.
      ObjSize = Flags.getByValSize();
      ArgSize = ((ObjSize + PtrByteSize - 1)/PtrByteSize) * PtrByteSize;
      // Empty aggregate parameters do not take up registers.  Examples:
      //   struct { } a;
      //   union  { } b;
      //   int c[0];
      // etc.  However, we have to provide a place-holder in InVals, so
      // pretend we have an 8-byte item at the current address for that
      // purpose.
      if (!ObjSize) {
        int FI = MFI->CreateFixedObject(PtrByteSize, ArgOffset, true);
        SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
        InVals.push_back(FIN);
        continue;
      }
      // All aggregates smaller than 8 bytes must be passed right-justified.
      if (ObjSize < PtrByteSize)
        CurArgOffset = CurArgOffset + (PtrByteSize - ObjSize);
      // The value of the object is its address.
      int FI = MFI->CreateFixedObject(ObjSize, CurArgOffset, true);
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      InVals.push_back(FIN);

      if (ObjSize < 8) {
        if (GPR_idx != Num_GPR_Regs) {
          unsigned VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::G8RCRegClass);
          SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, PtrVT);
          SDValue Store;

          if (ObjSize==1 || ObjSize==2 || ObjSize==4) {
            EVT ObjType = (ObjSize == 1 ? MVT::i8 :
                           (ObjSize == 2 ? MVT::i16 : MVT::i32));
            Store = DAG.getTruncStore(Val.getValue(1), dl, Val, FIN,
                                      MachinePointerInfo(FuncArg, CurArgOffset),
                                      ObjType, false, false, 0);
          } else {
            // For sizes that don't fit a truncating store (3, 5, 6, 7),
            // store the whole register as-is to the parameter save area
            // slot.  The address of the parameter was already calculated
            // above (InVals.push_back(FIN)) to be the right-justified
            // offset within the slot.  For this store, we need a new
            // frame index that points at the beginning of the slot.
            int FI = MFI->CreateFixedObject(PtrByteSize, ArgOffset, true);
            SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
            Store = DAG.getStore(Val.getValue(1), dl, Val, FIN,
                                 MachinePointerInfo(FuncArg, ArgOffset),
                                 false, false, 0);
          }

          MemOps.push_back(Store);
          ++GPR_idx;
        }
        // Whether we copied from a register or not, advance the offset
        // into the parameter save area by a full doubleword.
        ArgOffset += PtrByteSize;
        continue;
      }

      for (unsigned j = 0; j < ArgSize; j += PtrByteSize) {
        // Store whatever pieces of the object are in registers
        // to memory.  ArgOffset will be the address of the beginning
        // of the object.
        if (GPR_idx != Num_GPR_Regs) {
          unsigned VReg;
          VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::G8RCRegClass);
          int FI = MFI->CreateFixedObject(PtrByteSize, ArgOffset, true);
          SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
          SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, PtrVT);
          SDValue Store = DAG.getStore(Val.getValue(1), dl, Val, FIN,
                                       MachinePointerInfo(FuncArg, ArgOffset),
                                       false, false, 0);
          MemOps.push_back(Store);
          ++GPR_idx;
          ArgOffset += PtrByteSize;
        } else {
          ArgOffset += ArgSize - j;
          break;
        }
      }
      continue;
    }

    switch (ObjectVT.getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unhandled argument type!");
    case MVT::i32:
    case MVT::i64:
      if (GPR_idx != Num_GPR_Regs) {
        unsigned VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::G8RCRegClass);
        ArgVal = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i64);

        if (ObjectVT == MVT::i32)
          // PPC64 passes i8, i16, and i32 values in i64 registers. Promote
          // value to MVT::i64 and then truncate to the correct register size.
          ArgVal = extendArgForPPC64(Flags, ObjectVT, DAG, ArgVal, dl);

        ++GPR_idx;
      } else {
        needsLoad = true;
        ArgSize = PtrByteSize;
      }
      ArgOffset += 8;
      break;

    case MVT::f32:
    case MVT::f64:
      // Every 8 bytes of argument space consumes one of the GPRs available for
      // argument passing.
      if (GPR_idx != Num_GPR_Regs) {
        ++GPR_idx;
      }
      if (FPR_idx != Num_FPR_Regs) {
        unsigned VReg;

        if (ObjectVT == MVT::f32)
          VReg = MF.addLiveIn(FPR[FPR_idx], &PPC::F4RCRegClass);
        else
          VReg = MF.addLiveIn(FPR[FPR_idx], &PPC::F8RCRegClass);

        ArgVal = DAG.getCopyFromReg(Chain, dl, VReg, ObjectVT);
        ++FPR_idx;
      } else {
        needsLoad = true;
        ArgSize = PtrByteSize;
      }

      ArgOffset += 8;
      break;
    case MVT::v4f32:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      // Note that vector arguments in registers don't reserve stack space,
      // except in varargs functions.
      if (VR_idx != Num_VR_Regs) {
        unsigned VReg = MF.addLiveIn(VR[VR_idx], &PPC::VRRCRegClass);
        ArgVal = DAG.getCopyFromReg(Chain, dl, VReg, ObjectVT);
        if (isVarArg) {
          while ((ArgOffset % 16) != 0) {
            ArgOffset += PtrByteSize;
            if (GPR_idx != Num_GPR_Regs)
              GPR_idx++;
          }
          ArgOffset += 16;
          GPR_idx = std::min(GPR_idx+4, Num_GPR_Regs); // FIXME correct for ppc64?
        }
        ++VR_idx;
      } else {
        // Vectors are aligned.
        ArgOffset = ((ArgOffset+15)/16)*16;
        CurArgOffset = ArgOffset;
        ArgOffset += 16;
        needsLoad = true;
      }
      break;
    }

    // We need to load the argument to a virtual register if we determined
    // above that we ran out of physical registers of the appropriate type.
    if (needsLoad) {
      int FI = MFI->CreateFixedObject(ObjSize,
                                      CurArgOffset + (ArgSize - ObjSize),
                                      isImmutable);
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      ArgVal = DAG.getLoad(ObjectVT, dl, Chain, FIN, MachinePointerInfo(),
                           false, false, false, 0);
    }

    InVals.push_back(ArgVal);
  }

  // Set the size that is at least reserved in caller of this function.  Tail
  // call optimized functions' reserved stack space needs to be aligned so that
  // taking the difference between two stack areas will result in an aligned
  // stack.
  setMinReservedArea(MF, DAG, nAltivecParamsAtEnd, MinReservedArea, true);

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (isVarArg) {
    int Depth = ArgOffset;

    FuncInfo->setVarArgsFrameIndex(
      MFI->CreateFixedObject(PtrByteSize, Depth, true));
    SDValue FIN = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(), PtrVT);

    // If this function is vararg, store any remaining integer argument regs
    // to their spots on the stack so that they may be loaded by deferencing the
    // result of va_next.
    for (; GPR_idx != Num_GPR_Regs; ++GPR_idx) {
      unsigned VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::G8RCRegClass);
      SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, PtrVT);
      SDValue Store = DAG.getStore(Val.getValue(1), dl, Val, FIN,
                                   MachinePointerInfo(), false, false, 0);
      MemOps.push_back(Store);
      // Increment the address by four for the next argument to store
      SDValue PtrOff = DAG.getConstant(PtrByteSize, PtrVT);
      FIN = DAG.getNode(ISD::ADD, dl, PtrOff.getValueType(), FIN, PtrOff);
    }
  }

  if (!MemOps.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl,
                        MVT::Other, &MemOps[0], MemOps.size());

  return Chain;
}

SDValue
PPCTargetLowering::LowerFormalArguments_Darwin(
                                      SDValue Chain,
                                      CallingConv::ID CallConv, bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg>
                                        &Ins,
                                      DebugLoc dl, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {
  // TODO: add description of PPC stack frame format, or at least some docs.
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  bool isPPC64 = PtrVT == MVT::i64;
  // Potential tail calls could cause overwriting of argument stack slots.
  bool isImmutable = !(getTargetMachine().Options.GuaranteedTailCallOpt &&
                       (CallConv == CallingConv::Fast));
  unsigned PtrByteSize = isPPC64 ? 8 : 4;

  unsigned ArgOffset = PPCFrameLowering::getLinkageSize(isPPC64, true);
  // Area that is at least reserved in caller of this function.
  unsigned MinReservedArea = ArgOffset;

  static const uint16_t GPR_32[] = {           // 32-bit registers.
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const uint16_t GPR_64[] = {           // 64-bit registers.
    PPC::X3, PPC::X4, PPC::X5, PPC::X6,
    PPC::X7, PPC::X8, PPC::X9, PPC::X10,
  };

  static const uint16_t *FPR = GetFPR();

  static const uint16_t VR[] = {
    PPC::V2, PPC::V3, PPC::V4, PPC::V5, PPC::V6, PPC::V7, PPC::V8,
    PPC::V9, PPC::V10, PPC::V11, PPC::V12, PPC::V13
  };

  const unsigned Num_GPR_Regs = array_lengthof(GPR_32);
  const unsigned Num_FPR_Regs = 13;
  const unsigned Num_VR_Regs  = array_lengthof( VR);

  unsigned GPR_idx = 0, FPR_idx = 0, VR_idx = 0;

  const uint16_t *GPR = isPPC64 ? GPR_64 : GPR_32;

  // In 32-bit non-varargs functions, the stack space for vectors is after the
  // stack space for non-vectors.  We do not use this space unless we have
  // too many vectors to fit in registers, something that only occurs in
  // constructed examples:), but we have to walk the arglist to figure
  // that out...for the pathological case, compute VecArgOffset as the
  // start of the vector parameter area.  Computing VecArgOffset is the
  // entire point of the following loop.
  unsigned VecArgOffset = ArgOffset;
  if (!isVarArg && !isPPC64) {
    for (unsigned ArgNo = 0, e = Ins.size(); ArgNo != e;
         ++ArgNo) {
      EVT ObjectVT = Ins[ArgNo].VT;
      ISD::ArgFlagsTy Flags = Ins[ArgNo].Flags;

      if (Flags.isByVal()) {
        // ObjSize is the true size, ArgSize rounded up to multiple of regs.
        unsigned ObjSize = Flags.getByValSize();
        unsigned ArgSize =
                ((ObjSize + PtrByteSize - 1)/PtrByteSize) * PtrByteSize;
        VecArgOffset += ArgSize;
        continue;
      }

      switch(ObjectVT.getSimpleVT().SimpleTy) {
      default: llvm_unreachable("Unhandled argument type!");
      case MVT::i32:
      case MVT::f32:
        VecArgOffset += 4;
        break;
      case MVT::i64:  // PPC64
      case MVT::f64:
        // FIXME: We are guaranteed to be !isPPC64 at this point.
        // Does MVT::i64 apply?
        VecArgOffset += 8;
        break;
      case MVT::v4f32:
      case MVT::v4i32:
      case MVT::v8i16:
      case MVT::v16i8:
        // Nothing to do, we're only looking at Nonvector args here.
        break;
      }
    }
  }
  // We've found where the vector parameter area in memory is.  Skip the
  // first 12 parameters; these don't use that memory.
  VecArgOffset = ((VecArgOffset+15)/16)*16;
  VecArgOffset += 12*16;

  // Add DAG nodes to load the arguments or copy them out of registers.  On
  // entry to a function on PPC, the arguments start after the linkage area,
  // although the first ones are often in registers.

  SmallVector<SDValue, 8> MemOps;
  unsigned nAltivecParamsAtEnd = 0;
  Function::const_arg_iterator FuncArg = MF.getFunction()->arg_begin();
  for (unsigned ArgNo = 0, e = Ins.size(); ArgNo != e; ++ArgNo, ++FuncArg) {
    SDValue ArgVal;
    bool needsLoad = false;
    EVT ObjectVT = Ins[ArgNo].VT;
    unsigned ObjSize = ObjectVT.getSizeInBits()/8;
    unsigned ArgSize = ObjSize;
    ISD::ArgFlagsTy Flags = Ins[ArgNo].Flags;

    unsigned CurArgOffset = ArgOffset;

    // Varargs or 64 bit Altivec parameters are padded to a 16 byte boundary.
    if (ObjectVT==MVT::v4f32 || ObjectVT==MVT::v4i32 ||
        ObjectVT==MVT::v8i16 || ObjectVT==MVT::v16i8) {
      if (isVarArg || isPPC64) {
        MinReservedArea = ((MinReservedArea+15)/16)*16;
        MinReservedArea += CalculateStackSlotSize(ObjectVT,
                                                  Flags,
                                                  PtrByteSize);
      } else  nAltivecParamsAtEnd++;
    } else
      // Calculate min reserved area.
      MinReservedArea += CalculateStackSlotSize(Ins[ArgNo].VT,
                                                Flags,
                                                PtrByteSize);

    // FIXME the codegen can be much improved in some cases.
    // We do not have to keep everything in memory.
    if (Flags.isByVal()) {
      // ObjSize is the true size, ArgSize rounded up to multiple of registers.
      ObjSize = Flags.getByValSize();
      ArgSize = ((ObjSize + PtrByteSize - 1)/PtrByteSize) * PtrByteSize;
      // Objects of size 1 and 2 are right justified, everything else is
      // left justified.  This means the memory address is adjusted forwards.
      if (ObjSize==1 || ObjSize==2) {
        CurArgOffset = CurArgOffset + (4 - ObjSize);
      }
      // The value of the object is its address.
      int FI = MFI->CreateFixedObject(ObjSize, CurArgOffset, true);
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      InVals.push_back(FIN);
      if (ObjSize==1 || ObjSize==2) {
        if (GPR_idx != Num_GPR_Regs) {
          unsigned VReg;
          if (isPPC64)
            VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::G8RCRegClass);
          else
            VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::GPRCRegClass);
          SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, PtrVT);
          EVT ObjType = ObjSize == 1 ? MVT::i8 : MVT::i16;
          SDValue Store = DAG.getTruncStore(Val.getValue(1), dl, Val, FIN,
                                            MachinePointerInfo(FuncArg,
                                              CurArgOffset),
                                            ObjType, false, false, 0);
          MemOps.push_back(Store);
          ++GPR_idx;
        }

        ArgOffset += PtrByteSize;

        continue;
      }
      for (unsigned j = 0; j < ArgSize; j += PtrByteSize) {
        // Store whatever pieces of the object are in registers
        // to memory.  ArgOffset will be the address of the beginning
        // of the object.
        if (GPR_idx != Num_GPR_Regs) {
          unsigned VReg;
          if (isPPC64)
            VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::G8RCRegClass);
          else
            VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::GPRCRegClass);
          int FI = MFI->CreateFixedObject(PtrByteSize, ArgOffset, true);
          SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
          SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, PtrVT);
          SDValue Store = DAG.getStore(Val.getValue(1), dl, Val, FIN,
                                       MachinePointerInfo(FuncArg, ArgOffset),
                                       false, false, 0);
          MemOps.push_back(Store);
          ++GPR_idx;
          ArgOffset += PtrByteSize;
        } else {
          ArgOffset += ArgSize - (ArgOffset-CurArgOffset);
          break;
        }
      }
      continue;
    }

    switch (ObjectVT.getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unhandled argument type!");
    case MVT::i32:
      if (!isPPC64) {
        if (GPR_idx != Num_GPR_Regs) {
          unsigned VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::GPRCRegClass);
          ArgVal = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i32);
          ++GPR_idx;
        } else {
          needsLoad = true;
          ArgSize = PtrByteSize;
        }
        // All int arguments reserve stack space in the Darwin ABI.
        ArgOffset += PtrByteSize;
        break;
      }
      // FALLTHROUGH
    case MVT::i64:  // PPC64
      if (GPR_idx != Num_GPR_Regs) {
        unsigned VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::G8RCRegClass);
        ArgVal = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i64);

        if (ObjectVT == MVT::i32)
          // PPC64 passes i8, i16, and i32 values in i64 registers. Promote
          // value to MVT::i64 and then truncate to the correct register size.
          ArgVal = extendArgForPPC64(Flags, ObjectVT, DAG, ArgVal, dl);

        ++GPR_idx;
      } else {
        needsLoad = true;
        ArgSize = PtrByteSize;
      }
      // All int arguments reserve stack space in the Darwin ABI.
      ArgOffset += 8;
      break;

    case MVT::f32:
    case MVT::f64:
      // Every 4 bytes of argument space consumes one of the GPRs available for
      // argument passing.
      if (GPR_idx != Num_GPR_Regs) {
        ++GPR_idx;
        if (ObjSize == 8 && GPR_idx != Num_GPR_Regs && !isPPC64)
          ++GPR_idx;
      }
      if (FPR_idx != Num_FPR_Regs) {
        unsigned VReg;

        if (ObjectVT == MVT::f32)
          VReg = MF.addLiveIn(FPR[FPR_idx], &PPC::F4RCRegClass);
        else
          VReg = MF.addLiveIn(FPR[FPR_idx], &PPC::F8RCRegClass);

        ArgVal = DAG.getCopyFromReg(Chain, dl, VReg, ObjectVT);
        ++FPR_idx;
      } else {
        needsLoad = true;
      }

      // All FP arguments reserve stack space in the Darwin ABI.
      ArgOffset += isPPC64 ? 8 : ObjSize;
      break;
    case MVT::v4f32:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      // Note that vector arguments in registers don't reserve stack space,
      // except in varargs functions.
      if (VR_idx != Num_VR_Regs) {
        unsigned VReg = MF.addLiveIn(VR[VR_idx], &PPC::VRRCRegClass);
        ArgVal = DAG.getCopyFromReg(Chain, dl, VReg, ObjectVT);
        if (isVarArg) {
          while ((ArgOffset % 16) != 0) {
            ArgOffset += PtrByteSize;
            if (GPR_idx != Num_GPR_Regs)
              GPR_idx++;
          }
          ArgOffset += 16;
          GPR_idx = std::min(GPR_idx+4, Num_GPR_Regs); // FIXME correct for ppc64?
        }
        ++VR_idx;
      } else {
        if (!isVarArg && !isPPC64) {
          // Vectors go after all the nonvectors.
          CurArgOffset = VecArgOffset;
          VecArgOffset += 16;
        } else {
          // Vectors are aligned.
          ArgOffset = ((ArgOffset+15)/16)*16;
          CurArgOffset = ArgOffset;
          ArgOffset += 16;
        }
        needsLoad = true;
      }
      break;
    }

    // We need to load the argument to a virtual register if we determined above
    // that we ran out of physical registers of the appropriate type.
    if (needsLoad) {
      int FI = MFI->CreateFixedObject(ObjSize,
                                      CurArgOffset + (ArgSize - ObjSize),
                                      isImmutable);
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      ArgVal = DAG.getLoad(ObjectVT, dl, Chain, FIN, MachinePointerInfo(),
                           false, false, false, 0);
    }

    InVals.push_back(ArgVal);
  }

  // Set the size that is at least reserved in caller of this function.  Tail
  // call optimized functions' reserved stack space needs to be aligned so that
  // taking the difference between two stack areas will result in an aligned
  // stack.
  setMinReservedArea(MF, DAG, nAltivecParamsAtEnd, MinReservedArea, isPPC64);

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (isVarArg) {
    int Depth = ArgOffset;

    FuncInfo->setVarArgsFrameIndex(
      MFI->CreateFixedObject(PtrVT.getSizeInBits()/8,
                             Depth, true));
    SDValue FIN = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(), PtrVT);

    // If this function is vararg, store any remaining integer argument regs
    // to their spots on the stack so that they may be loaded by deferencing the
    // result of va_next.
    for (; GPR_idx != Num_GPR_Regs; ++GPR_idx) {
      unsigned VReg;

      if (isPPC64)
        VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::G8RCRegClass);
      else
        VReg = MF.addLiveIn(GPR[GPR_idx], &PPC::GPRCRegClass);

      SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, PtrVT);
      SDValue Store = DAG.getStore(Val.getValue(1), dl, Val, FIN,
                                   MachinePointerInfo(), false, false, 0);
      MemOps.push_back(Store);
      // Increment the address by four for the next argument to store
      SDValue PtrOff = DAG.getConstant(PtrVT.getSizeInBits()/8, PtrVT);
      FIN = DAG.getNode(ISD::ADD, dl, PtrOff.getValueType(), FIN, PtrOff);
    }
  }

  if (!MemOps.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl,
                        MVT::Other, &MemOps[0], MemOps.size());

  return Chain;
}

/// CalculateParameterAndLinkageAreaSize - Get the size of the parameter plus
/// linkage area for the Darwin ABI, or the 64-bit SVR4 ABI.
static unsigned
CalculateParameterAndLinkageAreaSize(SelectionDAG &DAG,
                                     bool isPPC64,
                                     bool isVarArg,
                                     unsigned CC,
                                     const SmallVectorImpl<ISD::OutputArg>
                                       &Outs,
                                     const SmallVectorImpl<SDValue> &OutVals,
                                     unsigned &nAltivecParamsAtEnd) {
  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.  We start with 24/48 bytes, which is
  // prereserved space for [SP][CR][LR][3 x unused].
  unsigned NumBytes = PPCFrameLowering::getLinkageSize(isPPC64, true);
  unsigned NumOps = Outs.size();
  unsigned PtrByteSize = isPPC64 ? 8 : 4;

  // Add up all the space actually used.
  // In 32-bit non-varargs calls, Altivec parameters all go at the end; usually
  // they all go in registers, but we must reserve stack space for them for
  // possible use by the caller.  In varargs or 64-bit calls, parameters are
  // assigned stack space in order, with padding so Altivec parameters are
  // 16-byte aligned.
  nAltivecParamsAtEnd = 0;
  for (unsigned i = 0; i != NumOps; ++i) {
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    EVT ArgVT = Outs[i].VT;
    // Varargs Altivec parameters are padded to a 16 byte boundary.
    if (ArgVT==MVT::v4f32 || ArgVT==MVT::v4i32 ||
        ArgVT==MVT::v8i16 || ArgVT==MVT::v16i8) {
      if (!isVarArg && !isPPC64) {
        // Non-varargs Altivec parameters go after all the non-Altivec
        // parameters; handle those later so we know how much padding we need.
        nAltivecParamsAtEnd++;
        continue;
      }
      // Varargs and 64-bit Altivec parameters are padded to 16 byte boundary.
      NumBytes = ((NumBytes+15)/16)*16;
    }
    NumBytes += CalculateStackSlotSize(ArgVT, Flags, PtrByteSize);
  }

   // Allow for Altivec parameters at the end, if needed.
  if (nAltivecParamsAtEnd) {
    NumBytes = ((NumBytes+15)/16)*16;
    NumBytes += 16*nAltivecParamsAtEnd;
  }

  // The prolog code of the callee may store up to 8 GPR argument registers to
  // the stack, allowing va_start to index over them in memory if its varargs.
  // Because we cannot tell if this is needed on the caller side, we have to
  // conservatively assume that it is needed.  As such, make sure we have at
  // least enough stack space for the caller to store the 8 GPRs.
  NumBytes = std::max(NumBytes,
                      PPCFrameLowering::getMinCallFrameSize(isPPC64, true));

  // Tail call needs the stack to be aligned.
  if (CC == CallingConv::Fast && DAG.getTarget().Options.GuaranteedTailCallOpt){
    unsigned TargetAlign = DAG.getMachineFunction().getTarget().
      getFrameLowering()->getStackAlignment();
    unsigned AlignMask = TargetAlign-1;
    NumBytes = (NumBytes + AlignMask) & ~AlignMask;
  }

  return NumBytes;
}

/// CalculateTailCallSPDiff - Get the amount the stack pointer has to be
/// adjusted to accommodate the arguments for the tailcall.
static int CalculateTailCallSPDiff(SelectionDAG& DAG, bool isTailCall,
                                   unsigned ParamSize) {

  if (!isTailCall) return 0;

  PPCFunctionInfo *FI = DAG.getMachineFunction().getInfo<PPCFunctionInfo>();
  unsigned CallerMinReservedArea = FI->getMinReservedArea();
  int SPDiff = (int)CallerMinReservedArea - (int)ParamSize;
  // Remember only if the new adjustement is bigger.
  if (SPDiff < FI->getTailCallSPDelta())
    FI->setTailCallSPDelta(SPDiff);

  return SPDiff;
}

/// IsEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization. Targets which want to do tail call
/// optimization should implement this function.
bool
PPCTargetLowering::IsEligibleForTailCallOptimization(SDValue Callee,
                                                     CallingConv::ID CalleeCC,
                                                     bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                                     SelectionDAG& DAG) const {
  if (!getTargetMachine().Options.GuaranteedTailCallOpt)
    return false;

  // Variable argument functions are not supported.
  if (isVarArg)
    return false;

  MachineFunction &MF = DAG.getMachineFunction();
  CallingConv::ID CallerCC = MF.getFunction()->getCallingConv();
  if (CalleeCC == CallingConv::Fast && CallerCC == CalleeCC) {
    // Functions containing by val parameters are not supported.
    for (unsigned i = 0; i != Ins.size(); i++) {
       ISD::ArgFlagsTy Flags = Ins[i].Flags;
       if (Flags.isByVal()) return false;
    }

    // Non PIC/GOT  tail calls are supported.
    if (getTargetMachine().getRelocationModel() != Reloc::PIC_)
      return true;

    // At the moment we can only do local tail calls (in same module, hidden
    // or protected) if we are generating PIC.
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
      return G->getGlobal()->hasHiddenVisibility()
          || G->getGlobal()->hasProtectedVisibility();
  }

  return false;
}

/// isCallCompatibleAddress - Return the immediate to use if the specified
/// 32-bit value is representable in the immediate field of a BxA instruction.
static SDNode *isBLACompatibleAddress(SDValue Op, SelectionDAG &DAG) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
  if (!C) return 0;

  int Addr = C->getZExtValue();
  if ((Addr & 3) != 0 ||  // Low 2 bits are implicitly zero.
      SignExtend32<26>(Addr) != Addr)
    return 0;  // Top 6 bits have to be sext of immediate.

  return DAG.getConstant((int)C->getZExtValue() >> 2,
                         DAG.getTargetLoweringInfo().getPointerTy()).getNode();
}

namespace {

struct TailCallArgumentInfo {
  SDValue Arg;
  SDValue FrameIdxOp;
  int       FrameIdx;

  TailCallArgumentInfo() : FrameIdx(0) {}
};

}

/// StoreTailCallArgumentsToStackSlot - Stores arguments to their stack slot.
static void
StoreTailCallArgumentsToStackSlot(SelectionDAG &DAG,
                                           SDValue Chain,
                   const SmallVector<TailCallArgumentInfo, 8> &TailCallArgs,
                   SmallVector<SDValue, 8> &MemOpChains,
                   DebugLoc dl) {
  for (unsigned i = 0, e = TailCallArgs.size(); i != e; ++i) {
    SDValue Arg = TailCallArgs[i].Arg;
    SDValue FIN = TailCallArgs[i].FrameIdxOp;
    int FI = TailCallArgs[i].FrameIdx;
    // Store relative to framepointer.
    MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, FIN,
                                       MachinePointerInfo::getFixedStack(FI),
                                       false, false, 0));
  }
}

/// EmitTailCallStoreFPAndRetAddr - Move the frame pointer and return address to
/// the appropriate stack slot for the tail call optimized function call.
static SDValue EmitTailCallStoreFPAndRetAddr(SelectionDAG &DAG,
                                               MachineFunction &MF,
                                               SDValue Chain,
                                               SDValue OldRetAddr,
                                               SDValue OldFP,
                                               int SPDiff,
                                               bool isPPC64,
                                               bool isDarwinABI,
                                               DebugLoc dl) {
  if (SPDiff) {
    // Calculate the new stack slot for the return address.
    int SlotSize = isPPC64 ? 8 : 4;
    int NewRetAddrLoc = SPDiff + PPCFrameLowering::getReturnSaveOffset(isPPC64,
                                                                   isDarwinABI);
    int NewRetAddr = MF.getFrameInfo()->CreateFixedObject(SlotSize,
                                                          NewRetAddrLoc, true);
    EVT VT = isPPC64 ? MVT::i64 : MVT::i32;
    SDValue NewRetAddrFrIdx = DAG.getFrameIndex(NewRetAddr, VT);
    Chain = DAG.getStore(Chain, dl, OldRetAddr, NewRetAddrFrIdx,
                         MachinePointerInfo::getFixedStack(NewRetAddr),
                         false, false, 0);

    // When using the 32/64-bit SVR4 ABI there is no need to move the FP stack
    // slot as the FP is never overwritten.
    if (isDarwinABI) {
      int NewFPLoc =
        SPDiff + PPCFrameLowering::getFramePointerSaveOffset(isPPC64, isDarwinABI);
      int NewFPIdx = MF.getFrameInfo()->CreateFixedObject(SlotSize, NewFPLoc,
                                                          true);
      SDValue NewFramePtrIdx = DAG.getFrameIndex(NewFPIdx, VT);
      Chain = DAG.getStore(Chain, dl, OldFP, NewFramePtrIdx,
                           MachinePointerInfo::getFixedStack(NewFPIdx),
                           false, false, 0);
    }
  }
  return Chain;
}

/// CalculateTailCallArgDest - Remember Argument for later processing. Calculate
/// the position of the argument.
static void
CalculateTailCallArgDest(SelectionDAG &DAG, MachineFunction &MF, bool isPPC64,
                         SDValue Arg, int SPDiff, unsigned ArgOffset,
                      SmallVector<TailCallArgumentInfo, 8>& TailCallArguments) {
  int Offset = ArgOffset + SPDiff;
  uint32_t OpSize = (Arg.getValueType().getSizeInBits()+7)/8;
  int FI = MF.getFrameInfo()->CreateFixedObject(OpSize, Offset, true);
  EVT VT = isPPC64 ? MVT::i64 : MVT::i32;
  SDValue FIN = DAG.getFrameIndex(FI, VT);
  TailCallArgumentInfo Info;
  Info.Arg = Arg;
  Info.FrameIdxOp = FIN;
  Info.FrameIdx = FI;
  TailCallArguments.push_back(Info);
}

/// EmitTCFPAndRetAddrLoad - Emit load from frame pointer and return address
/// stack slot. Returns the chain as result and the loaded frame pointers in
/// LROpOut/FPOpout. Used when tail calling.
SDValue PPCTargetLowering::EmitTailCallLoadFPAndRetAddr(SelectionDAG & DAG,
                                                        int SPDiff,
                                                        SDValue Chain,
                                                        SDValue &LROpOut,
                                                        SDValue &FPOpOut,
                                                        bool isDarwinABI,
                                                        DebugLoc dl) const {
  if (SPDiff) {
    // Load the LR and FP stack slot for later adjusting.
    EVT VT = PPCSubTarget.isPPC64() ? MVT::i64 : MVT::i32;
    LROpOut = getReturnAddrFrameIndex(DAG);
    LROpOut = DAG.getLoad(VT, dl, Chain, LROpOut, MachinePointerInfo(),
                          false, false, false, 0);
    Chain = SDValue(LROpOut.getNode(), 1);

    // When using the 32/64-bit SVR4 ABI there is no need to load the FP stack
    // slot as the FP is never overwritten.
    if (isDarwinABI) {
      FPOpOut = getFramePointerFrameIndex(DAG);
      FPOpOut = DAG.getLoad(VT, dl, Chain, FPOpOut, MachinePointerInfo(),
                            false, false, false, 0);
      Chain = SDValue(FPOpOut.getNode(), 1);
    }
  }
  return Chain;
}

/// CreateCopyOfByValArgument - Make a copy of an aggregate at address specified
/// by "Src" to address "Dst" of size "Size".  Alignment information is
/// specified by the specific parameter attribute. The copy will be passed as
/// a byval function parameter.
/// Sometimes what we are copying is the end of a larger object, the part that
/// does not fit in registers.
static SDValue
CreateCopyOfByValArgument(SDValue Src, SDValue Dst, SDValue Chain,
                          ISD::ArgFlagsTy Flags, SelectionDAG &DAG,
                          DebugLoc dl) {
  SDValue SizeNode = DAG.getConstant(Flags.getByValSize(), MVT::i32);
  return DAG.getMemcpy(Chain, dl, Dst, Src, SizeNode, Flags.getByValAlign(),
                       false, false, MachinePointerInfo(0),
                       MachinePointerInfo(0));
}

/// LowerMemOpCallTo - Store the argument to the stack or remember it in case of
/// tail calls.
static void
LowerMemOpCallTo(SelectionDAG &DAG, MachineFunction &MF, SDValue Chain,
                 SDValue Arg, SDValue PtrOff, int SPDiff,
                 unsigned ArgOffset, bool isPPC64, bool isTailCall,
                 bool isVector, SmallVector<SDValue, 8> &MemOpChains,
                 SmallVector<TailCallArgumentInfo, 8> &TailCallArguments,
                 DebugLoc dl) {
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  if (!isTailCall) {
    if (isVector) {
      SDValue StackPtr;
      if (isPPC64)
        StackPtr = DAG.getRegister(PPC::X1, MVT::i64);
      else
        StackPtr = DAG.getRegister(PPC::R1, MVT::i32);
      PtrOff = DAG.getNode(ISD::ADD, dl, PtrVT, StackPtr,
                           DAG.getConstant(ArgOffset, PtrVT));
    }
    MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                       MachinePointerInfo(), false, false, 0));
  // Calculate and remember argument location.
  } else CalculateTailCallArgDest(DAG, MF, isPPC64, Arg, SPDiff, ArgOffset,
                                  TailCallArguments);
}

static
void PrepareTailCall(SelectionDAG &DAG, SDValue &InFlag, SDValue &Chain,
                     DebugLoc dl, bool isPPC64, int SPDiff, unsigned NumBytes,
                     SDValue LROp, SDValue FPOp, bool isDarwinABI,
                     SmallVector<TailCallArgumentInfo, 8> &TailCallArguments) {
  MachineFunction &MF = DAG.getMachineFunction();

  // Emit a sequence of copyto/copyfrom virtual registers for arguments that
  // might overwrite each other in case of tail call optimization.
  SmallVector<SDValue, 8> MemOpChains2;
  // Do not flag preceding copytoreg stuff together with the following stuff.
  InFlag = SDValue();
  StoreTailCallArgumentsToStackSlot(DAG, Chain, TailCallArguments,
                                    MemOpChains2, dl);
  if (!MemOpChains2.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains2[0], MemOpChains2.size());

  // Store the return address to the appropriate stack slot.
  Chain = EmitTailCallStoreFPAndRetAddr(DAG, MF, Chain, LROp, FPOp, SPDiff,
                                        isPPC64, isDarwinABI, dl);

  // Emit callseq_end just before tailcall node.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  InFlag = Chain.getValue(1);
}

static
unsigned PrepareCall(SelectionDAG &DAG, SDValue &Callee, SDValue &InFlag,
                     SDValue &Chain, DebugLoc dl, int SPDiff, bool isTailCall,
                     SmallVector<std::pair<unsigned, SDValue>, 8> &RegsToPass,
                     SmallVector<SDValue, 8> &Ops, std::vector<EVT> &NodeTys,
                     const PPCSubtarget &PPCSubTarget) {

  bool isPPC64 = PPCSubTarget.isPPC64();
  bool isSVR4ABI = PPCSubTarget.isSVR4ABI();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Glue);    // Returns a flag for retval copy to use.

  unsigned CallOpc = isSVR4ABI ? PPCISD::CALL_SVR4 : PPCISD::CALL_Darwin;

  bool needIndirectCall = true;
  if (SDNode *Dest = isBLACompatibleAddress(Callee, DAG)) {
    // If this is an absolute destination address, use the munged value.
    Callee = SDValue(Dest, 0);
    needIndirectCall = false;
  }

  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    // XXX Work around for http://llvm.org/bugs/show_bug.cgi?id=5201
    // Use indirect calls for ALL functions calls in JIT mode, since the
    // far-call stubs may be outside relocation limits for a BL instruction.
    if (!DAG.getTarget().getSubtarget<PPCSubtarget>().isJITCodeModel()) {
      unsigned OpFlags = 0;
      if (DAG.getTarget().getRelocationModel() != Reloc::Static &&
          (PPCSubTarget.getTargetTriple().isMacOSX() &&
           PPCSubTarget.getTargetTriple().isMacOSXVersionLT(10, 5)) &&
          (G->getGlobal()->isDeclaration() ||
           G->getGlobal()->isWeakForLinker())) {
        // PC-relative references to external symbols should go through $stub,
        // unless we're building with the leopard linker or later, which
        // automatically synthesizes these stubs.
        OpFlags = PPCII::MO_DARWIN_STUB;
      }

      // If the callee is a GlobalAddress/ExternalSymbol node (quite common,
      // every direct call is) turn it into a TargetGlobalAddress /
      // TargetExternalSymbol node so that legalize doesn't hack it.
      Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl,
                                          Callee.getValueType(),
                                          0, OpFlags);
      needIndirectCall = false;
    }
  }

  if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    unsigned char OpFlags = 0;

    if (DAG.getTarget().getRelocationModel() != Reloc::Static &&
        (PPCSubTarget.getTargetTriple().isMacOSX() &&
         PPCSubTarget.getTargetTriple().isMacOSXVersionLT(10, 5))) {
      // PC-relative references to external symbols should go through $stub,
      // unless we're building with the leopard linker or later, which
      // automatically synthesizes these stubs.
      OpFlags = PPCII::MO_DARWIN_STUB;
    }

    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), Callee.getValueType(),
                                         OpFlags);
    needIndirectCall = false;
  }

  if (needIndirectCall) {
    // Otherwise, this is an indirect call.  We have to use a MTCTR/BCTRL pair
    // to do the call, we can't use PPCISD::CALL.
    SDValue MTCTROps[] = {Chain, Callee, InFlag};

    if (isSVR4ABI && isPPC64) {
      // Function pointers in the 64-bit SVR4 ABI do not point to the function
      // entry point, but to the function descriptor (the function entry point
      // address is part of the function descriptor though).
      // The function descriptor is a three doubleword structure with the
      // following fields: function entry point, TOC base address and
      // environment pointer.
      // Thus for a call through a function pointer, the following actions need
      // to be performed:
      //   1. Save the TOC of the caller in the TOC save area of its stack
      //      frame (this is done in LowerCall_Darwin() or LowerCall_64SVR4()).
      //   2. Load the address of the function entry point from the function
      //      descriptor.
      //   3. Load the TOC of the callee from the function descriptor into r2.
      //   4. Load the environment pointer from the function descriptor into
      //      r11.
      //   5. Branch to the function entry point address.
      //   6. On return of the callee, the TOC of the caller needs to be
      //      restored (this is done in FinishCall()).
      //
      // All those operations are flagged together to ensure that no other
      // operations can be scheduled in between. E.g. without flagging the
      // operations together, a TOC access in the caller could be scheduled
      // between the load of the callee TOC and the branch to the callee, which
      // results in the TOC access going through the TOC of the callee instead
      // of going through the TOC of the caller, which leads to incorrect code.

      // Load the address of the function entry point from the function
      // descriptor.
      SDVTList VTs = DAG.getVTList(MVT::i64, MVT::Other, MVT::Glue);
      SDValue LoadFuncPtr = DAG.getNode(PPCISD::LOAD, dl, VTs, MTCTROps,
                                        InFlag.getNode() ? 3 : 2);
      Chain = LoadFuncPtr.getValue(1);
      InFlag = LoadFuncPtr.getValue(2);

      // Load environment pointer into r11.
      // Offset of the environment pointer within the function descriptor.
      SDValue PtrOff = DAG.getIntPtrConstant(16);

      SDValue AddPtr = DAG.getNode(ISD::ADD, dl, MVT::i64, Callee, PtrOff);
      SDValue LoadEnvPtr = DAG.getNode(PPCISD::LOAD, dl, VTs, Chain, AddPtr,
                                       InFlag);
      Chain = LoadEnvPtr.getValue(1);
      InFlag = LoadEnvPtr.getValue(2);

      SDValue EnvVal = DAG.getCopyToReg(Chain, dl, PPC::X11, LoadEnvPtr,
                                        InFlag);
      Chain = EnvVal.getValue(0);
      InFlag = EnvVal.getValue(1);

      // Load TOC of the callee into r2. We are using a target-specific load
      // with r2 hard coded, because the result of a target-independent load
      // would never go directly into r2, since r2 is a reserved register (which
      // prevents the register allocator from allocating it), resulting in an
      // additional register being allocated and an unnecessary move instruction
      // being generated.
      VTs = DAG.getVTList(MVT::Other, MVT::Glue);
      SDValue LoadTOCPtr = DAG.getNode(PPCISD::LOAD_TOC, dl, VTs, Chain,
                                       Callee, InFlag);
      Chain = LoadTOCPtr.getValue(0);
      InFlag = LoadTOCPtr.getValue(1);

      MTCTROps[0] = Chain;
      MTCTROps[1] = LoadFuncPtr;
      MTCTROps[2] = InFlag;
    }

    Chain = DAG.getNode(PPCISD::MTCTR, dl, NodeTys, MTCTROps,
                        2 + (InFlag.getNode() != 0));
    InFlag = Chain.getValue(1);

    NodeTys.clear();
    NodeTys.push_back(MVT::Other);
    NodeTys.push_back(MVT::Glue);
    Ops.push_back(Chain);
    CallOpc = isSVR4ABI ? PPCISD::BCTRL_SVR4 : PPCISD::BCTRL_Darwin;
    Callee.setNode(0);
    // Add CTR register as callee so a bctr can be emitted later.
    if (isTailCall)
      Ops.push_back(DAG.getRegister(isPPC64 ? PPC::CTR8 : PPC::CTR, PtrVT));
  }

  // If this is a direct call, pass the chain and the callee.
  if (Callee.getNode()) {
    Ops.push_back(Chain);
    Ops.push_back(Callee);
  }
  // If this is a tail call add stack pointer delta.
  if (isTailCall)
    Ops.push_back(DAG.getConstant(SPDiff, MVT::i32));

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  return CallOpc;
}

static
bool isLocalCall(const SDValue &Callee)
{
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    return !G->getGlobal()->isDeclaration() &&
           !G->getGlobal()->isWeakForLinker();
  return false;
}

SDValue
PPCTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                   CallingConv::ID CallConv, bool isVarArg,
                                   const SmallVectorImpl<ISD::InputArg> &Ins,
                                   DebugLoc dl, SelectionDAG &DAG,
                                   SmallVectorImpl<SDValue> &InVals) const {

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCRetInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                    getTargetMachine(), RVLocs, *DAG.getContext());
  CCRetInfo.AnalyzeCallResult(Ins, RetCC_PPC);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    SDValue Val = DAG.getCopyFromReg(Chain, dl,
                                     VA.getLocReg(), VA.getLocVT(), InFlag);
    Chain = Val.getValue(1);
    InFlag = Val.getValue(2);

    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::AssertZext, dl, VA.getLocVT(), Val,
                        DAG.getValueType(VA.getValVT()));
      Val = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), Val);
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::AssertSext, dl, VA.getLocVT(), Val,
                        DAG.getValueType(VA.getValVT()));
      Val = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), Val);
      break;
    }

    InVals.push_back(Val);
  }

  return Chain;
}

SDValue
PPCTargetLowering::FinishCall(CallingConv::ID CallConv, DebugLoc dl,
                              bool isTailCall, bool isVarArg,
                              SelectionDAG &DAG,
                              SmallVector<std::pair<unsigned, SDValue>, 8>
                                &RegsToPass,
                              SDValue InFlag, SDValue Chain,
                              SDValue &Callee,
                              int SPDiff, unsigned NumBytes,
                              const SmallVectorImpl<ISD::InputArg> &Ins,
                              SmallVectorImpl<SDValue> &InVals) const {
  std::vector<EVT> NodeTys;
  SmallVector<SDValue, 8> Ops;
  unsigned CallOpc = PrepareCall(DAG, Callee, InFlag, Chain, dl, SPDiff,
                                 isTailCall, RegsToPass, Ops, NodeTys,
                                 PPCSubTarget);

  // Add implicit use of CR bit 6 for 32-bit SVR4 vararg calls
  if (isVarArg && PPCSubTarget.isSVR4ABI() && !PPCSubTarget.isPPC64())
    Ops.push_back(DAG.getRegister(PPC::CR1EQ, MVT::i32));

  // When performing tail call optimization the callee pops its arguments off
  // the stack. Account for this here so these bytes can be pushed back on in
  // PPCRegisterInfo::eliminateCallFramePseudoInstr.
  int BytesCalleePops =
    (CallConv == CallingConv::Fast &&
     getTargetMachine().Options.GuaranteedTailCallOpt) ? NumBytes : 0;

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  // Emit tail call.
  if (isTailCall) {
    // If this is the first return lowered for this function, add the regs
    // to the liveout set for the function.
    if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
      SmallVector<CCValAssign, 16> RVLocs;
      CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                     getTargetMachine(), RVLocs, *DAG.getContext());
      CCInfo.AnalyzeCallResult(Ins, RetCC_PPC);
      for (unsigned i = 0; i != RVLocs.size(); ++i)
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
    }

    assert(((Callee.getOpcode() == ISD::Register &&
             cast<RegisterSDNode>(Callee)->getReg() == PPC::CTR) ||
            Callee.getOpcode() == ISD::TargetExternalSymbol ||
            Callee.getOpcode() == ISD::TargetGlobalAddress ||
            isa<ConstantSDNode>(Callee)) &&
    "Expecting an global address, external symbol, absolute value or register");

    return DAG.getNode(PPCISD::TC_RETURN, dl, MVT::Other, &Ops[0], Ops.size());
  }

  // Add a NOP immediately after the branch instruction when using the 64-bit
  // SVR4 ABI. At link time, if caller and callee are in a different module and
  // thus have a different TOC, the call will be replaced with a call to a stub
  // function which saves the current TOC, loads the TOC of the callee and
  // branches to the callee. The NOP will be replaced with a load instruction
  // which restores the TOC of the caller from the TOC save slot of the current
  // stack frame. If caller and callee belong to the same module (and have the
  // same TOC), the NOP will remain unchanged.

  bool needsTOCRestore = false;
  if (!isTailCall && PPCSubTarget.isSVR4ABI()&& PPCSubTarget.isPPC64()) {
    if (CallOpc == PPCISD::BCTRL_SVR4) {
      // This is a call through a function pointer.
      // Restore the caller TOC from the save area into R2.
      // See PrepareCall() for more information about calls through function
      // pointers in the 64-bit SVR4 ABI.
      // We are using a target-specific load with r2 hard coded, because the
      // result of a target-independent load would never go directly into r2,
      // since r2 is a reserved register (which prevents the register allocator
      // from allocating it), resulting in an additional register being
      // allocated and an unnecessary move instruction being generated.
      needsTOCRestore = true;
    } else if ((CallOpc == PPCISD::CALL_SVR4) && !isLocalCall(Callee)) {
      // Otherwise insert NOP for non-local calls.
      CallOpc = PPCISD::CALL_NOP_SVR4;
    }
  }

  Chain = DAG.getNode(CallOpc, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  if (needsTOCRestore) {
    SDVTList VTs = DAG.getVTList(MVT::Other, MVT::Glue);
    Chain = DAG.getNode(PPCISD::TOC_RESTORE, dl, VTs, Chain, InFlag);
    InFlag = Chain.getValue(1);
  }

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(BytesCalleePops, true),
                             InFlag);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  return LowerCallResult(Chain, InFlag, CallConv, isVarArg,
                         Ins, dl, DAG, InVals);
}

SDValue
PPCTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
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

  if (isTailCall)
    isTailCall = IsEligibleForTailCallOptimization(Callee, CallConv, isVarArg,
                                                   Ins, DAG);

  if (PPCSubTarget.isSVR4ABI()) {
    if (PPCSubTarget.isPPC64())
      return LowerCall_64SVR4(Chain, Callee, CallConv, isVarArg,
                              isTailCall, Outs, OutVals, Ins,
                              dl, DAG, InVals);
    else
      return LowerCall_32SVR4(Chain, Callee, CallConv, isVarArg,
                              isTailCall, Outs, OutVals, Ins,
                              dl, DAG, InVals);
  }

  return LowerCall_Darwin(Chain, Callee, CallConv, isVarArg,
                          isTailCall, Outs, OutVals, Ins,
                          dl, DAG, InVals);
}

SDValue
PPCTargetLowering::LowerCall_32SVR4(SDValue Chain, SDValue Callee,
                                    CallingConv::ID CallConv, bool isVarArg,
                                    bool isTailCall,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const {
  // See PPCTargetLowering::LowerFormalArguments_32SVR4() for a description
  // of the 32-bit SVR4 ABI stack frame layout.

  assert((CallConv == CallingConv::C ||
          CallConv == CallingConv::Fast) && "Unknown calling convention!");

  unsigned PtrByteSize = 4;

  MachineFunction &MF = DAG.getMachineFunction();

  // Mark this function as potentially containing a function that contains a
  // tail call. As a consequence the frame pointer will be used for dynamicalloc
  // and restoring the callers stack pointer in this functions epilog. This is
  // done because by tail calling the called function might overwrite the value
  // in this function's (MF) stack pointer stack slot 0(SP).
  if (getTargetMachine().Options.GuaranteedTailCallOpt &&
      CallConv == CallingConv::Fast)
    MF.getInfo<PPCFunctionInfo>()->setHasFastCall();

  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, parameter list area and the part of the local variable space which
  // contains copies of aggregates which are passed by value.

  // Assign locations to all of the outgoing arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  // Reserve space for the linkage area on the stack.
  CCInfo.AllocateStack(PPCFrameLowering::getLinkageSize(false, false), PtrByteSize);

  if (isVarArg) {
    // Handle fixed and variable vector arguments differently.
    // Fixed vector arguments go into registers as long as registers are
    // available. Variable vector arguments always go into memory.
    unsigned NumArgs = Outs.size();

    for (unsigned i = 0; i != NumArgs; ++i) {
      MVT ArgVT = Outs[i].VT;
      ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
      bool Result;

      if (Outs[i].IsFixed) {
        Result = CC_PPC_SVR4(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags,
                             CCInfo);
      } else {
        Result = CC_PPC_SVR4_VarArg(i, ArgVT, ArgVT, CCValAssign::Full,
                                    ArgFlags, CCInfo);
      }

      if (Result) {
#ifndef NDEBUG
        errs() << "Call operand #" << i << " has unhandled type "
             << EVT(ArgVT).getEVTString() << "\n";
#endif
        llvm_unreachable(0);
      }
    }
  } else {
    // All arguments are treated the same.
    CCInfo.AnalyzeCallOperands(Outs, CC_PPC_SVR4);
  }

  // Assign locations to all of the outgoing aggregate by value arguments.
  SmallVector<CCValAssign, 16> ByValArgLocs;
  CCState CCByValInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                      getTargetMachine(), ByValArgLocs, *DAG.getContext());

  // Reserve stack space for the allocations in CCInfo.
  CCByValInfo.AllocateStack(CCInfo.getNextStackOffset(), PtrByteSize);

  CCByValInfo.AnalyzeCallOperands(Outs, CC_PPC_SVR4_ByVal);

  // Size of the linkage area, parameter list area and the part of the local
  // space variable where copies of aggregates which are passed by value are
  // stored.
  unsigned NumBytes = CCByValInfo.getNextStackOffset();

  // Calculate by how many bytes the stack has to be adjusted in case of tail
  // call optimization.
  int SPDiff = CalculateTailCallSPDiff(DAG, isTailCall, NumBytes);

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));
  SDValue CallSeqStart = Chain;

  // Load the return address and frame pointer so it can be moved somewhere else
  // later.
  SDValue LROp, FPOp;
  Chain = EmitTailCallLoadFPAndRetAddr(DAG, SPDiff, Chain, LROp, FPOp, false,
                                       dl);

  // Set up a copy of the stack pointer for use loading and storing any
  // arguments that may not fit in the registers available for argument
  // passing.
  SDValue StackPtr = DAG.getRegister(PPC::R1, MVT::i32);

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<TailCallArgumentInfo, 8> TailCallArguments;
  SmallVector<SDValue, 8> MemOpChains;

  bool seenFloatArg = false;
  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, j = 0, e = ArgLocs.size();
       i != e;
       ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    if (Flags.isByVal()) {
      // Argument is an aggregate which is passed by value, thus we need to
      // create a copy of it in the local variable space of the current stack
      // frame (which is the stack frame of the caller) and pass the address of
      // this copy to the callee.
      assert((j < ByValArgLocs.size()) && "Index out of bounds!");
      CCValAssign &ByValVA = ByValArgLocs[j++];
      assert((VA.getValNo() == ByValVA.getValNo()) && "ValNo mismatch!");

      // Memory reserved in the local variable space of the callers stack frame.
      unsigned LocMemOffset = ByValVA.getLocMemOffset();

      SDValue PtrOff = DAG.getIntPtrConstant(LocMemOffset);
      PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, PtrOff);

      // Create a copy of the argument in the local area of the current
      // stack frame.
      SDValue MemcpyCall =
        CreateCopyOfByValArgument(Arg, PtrOff,
                                  CallSeqStart.getNode()->getOperand(0),
                                  Flags, DAG, dl);

      // This must go outside the CALLSEQ_START..END.
      SDValue NewCallSeqStart = DAG.getCALLSEQ_START(MemcpyCall,
                           CallSeqStart.getNode()->getOperand(1));
      DAG.ReplaceAllUsesWith(CallSeqStart.getNode(),
                             NewCallSeqStart.getNode());
      Chain = CallSeqStart = NewCallSeqStart;

      // Pass the address of the aggregate copy on the stack either in a
      // physical register or in the parameter list area of the current stack
      // frame to the callee.
      Arg = PtrOff;
    }

    if (VA.isRegLoc()) {
      seenFloatArg |= VA.getLocVT().isFloatingPoint();
      // Put argument in a physical register.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      // Put argument in the parameter list area of the current stack frame.
      assert(VA.isMemLoc());
      unsigned LocMemOffset = VA.getLocMemOffset();

      if (!isTailCall) {
        SDValue PtrOff = DAG.getIntPtrConstant(LocMemOffset);
        PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, PtrOff);

        MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                           MachinePointerInfo(),
                                           false, false, 0));
      } else {
        // Calculate and remember argument location.
        CalculateTailCallArgDest(DAG, MF, false, Arg, SPDiff, LocMemOffset,
                                 TailCallArguments);
      }
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

  // Set CR bit 6 to true if this is a vararg call with floating args passed in
  // registers.
  if (isVarArg) {
    SDVTList VTs = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue Ops[] = { Chain, InFlag };

    Chain = DAG.getNode(seenFloatArg ? PPCISD::CR6SET : PPCISD::CR6UNSET,
                        dl, VTs, Ops, InFlag.getNode() ? 2 : 1);

    InFlag = Chain.getValue(1);
  }

  if (isTailCall)
    PrepareTailCall(DAG, InFlag, Chain, dl, false, SPDiff, NumBytes, LROp, FPOp,
                    false, TailCallArguments);

  return FinishCall(CallConv, dl, isTailCall, isVarArg, DAG,
                    RegsToPass, InFlag, Chain, Callee, SPDiff, NumBytes,
                    Ins, InVals);
}

// Copy an argument into memory, being careful to do this outside the
// call sequence for the call to which the argument belongs.
SDValue
PPCTargetLowering::createMemcpyOutsideCallSeq(SDValue Arg, SDValue PtrOff,
                                              SDValue CallSeqStart,
                                              ISD::ArgFlagsTy Flags,
                                              SelectionDAG &DAG,
                                              DebugLoc dl) const {
  SDValue MemcpyCall = CreateCopyOfByValArgument(Arg, PtrOff,
                        CallSeqStart.getNode()->getOperand(0),
                        Flags, DAG, dl);
  // The MEMCPY must go outside the CALLSEQ_START..END.
  SDValue NewCallSeqStart = DAG.getCALLSEQ_START(MemcpyCall,
                             CallSeqStart.getNode()->getOperand(1));
  DAG.ReplaceAllUsesWith(CallSeqStart.getNode(),
                         NewCallSeqStart.getNode());
  return NewCallSeqStart;
}

SDValue
PPCTargetLowering::LowerCall_64SVR4(SDValue Chain, SDValue Callee,
                                    CallingConv::ID CallConv, bool isVarArg,
                                    bool isTailCall,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const {

  unsigned NumOps = Outs.size();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  unsigned PtrByteSize = 8;

  MachineFunction &MF = DAG.getMachineFunction();

  // Mark this function as potentially containing a function that contains a
  // tail call. As a consequence the frame pointer will be used for dynamicalloc
  // and restoring the callers stack pointer in this functions epilog. This is
  // done because by tail calling the called function might overwrite the value
  // in this function's (MF) stack pointer stack slot 0(SP).
  if (getTargetMachine().Options.GuaranteedTailCallOpt &&
      CallConv == CallingConv::Fast)
    MF.getInfo<PPCFunctionInfo>()->setHasFastCall();

  unsigned nAltivecParamsAtEnd = 0;

  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.  We start with at least 48 bytes, which
  // is reserved space for [SP][CR][LR][3 x unused].
  // NOTE: For PPC64, nAltivecParamsAtEnd always remains zero as a result
  // of this call.
  unsigned NumBytes =
    CalculateParameterAndLinkageAreaSize(DAG, true, isVarArg, CallConv,
                                         Outs, OutVals, nAltivecParamsAtEnd);

  // Calculate by how many bytes the stack has to be adjusted in case of tail
  // call optimization.
  int SPDiff = CalculateTailCallSPDiff(DAG, isTailCall, NumBytes);

  // To protect arguments on the stack from being clobbered in a tail call,
  // force all the loads to happen before doing any other lowering.
  if (isTailCall)
    Chain = DAG.getStackArgumentTokenFactor(Chain);

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));
  SDValue CallSeqStart = Chain;

  // Load the return address and frame pointer so it can be move somewhere else
  // later.
  SDValue LROp, FPOp;
  Chain = EmitTailCallLoadFPAndRetAddr(DAG, SPDiff, Chain, LROp, FPOp, true,
                                       dl);

  // Set up a copy of the stack pointer for use loading and storing any
  // arguments that may not fit in the registers available for argument
  // passing.
  SDValue StackPtr = DAG.getRegister(PPC::X1, MVT::i64);

  // Figure out which arguments are going to go in registers, and which in
  // memory.  Also, if this is a vararg function, floating point operations
  // must be stored to our stack, and loaded into integer regs as well, if
  // any integer regs are available for argument passing.
  unsigned ArgOffset = PPCFrameLowering::getLinkageSize(true, true);
  unsigned GPR_idx = 0, FPR_idx = 0, VR_idx = 0;

  static const uint16_t GPR[] = {
    PPC::X3, PPC::X4, PPC::X5, PPC::X6,
    PPC::X7, PPC::X8, PPC::X9, PPC::X10,
  };
  static const uint16_t *FPR = GetFPR();

  static const uint16_t VR[] = {
    PPC::V2, PPC::V3, PPC::V4, PPC::V5, PPC::V6, PPC::V7, PPC::V8,
    PPC::V9, PPC::V10, PPC::V11, PPC::V12, PPC::V13
  };
  const unsigned NumGPRs = array_lengthof(GPR);
  const unsigned NumFPRs = 13;
  const unsigned NumVRs  = array_lengthof(VR);

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<TailCallArgumentInfo, 8> TailCallArguments;

  SmallVector<SDValue, 8> MemOpChains;
  for (unsigned i = 0; i != NumOps; ++i) {
    SDValue Arg = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // PtrOff will be used to store the current argument to the stack if a
    // register cannot be found for it.
    SDValue PtrOff;

    PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());

    PtrOff = DAG.getNode(ISD::ADD, dl, PtrVT, StackPtr, PtrOff);

    // Promote integers to 64-bit values.
    if (Arg.getValueType() == MVT::i32) {
      // FIXME: Should this use ANY_EXTEND if neither sext nor zext?
      unsigned ExtOp = Flags.isSExt() ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
      Arg = DAG.getNode(ExtOp, dl, MVT::i64, Arg);
    }

    // FIXME memcpy is used way more than necessary.  Correctness first.
    // Note: "by value" is code for passing a structure by value, not
    // basic types.
    if (Flags.isByVal()) {
      // Note: Size includes alignment padding, so
      //   struct x { short a; char b; }
      // will have Size = 4.  With #pragma pack(1), it will have Size = 3.
      // These are the proper values we need for right-justifying the
      // aggregate in a parameter register.
      unsigned Size = Flags.getByValSize();

      // An empty aggregate parameter takes up no storage and no
      // registers.
      if (Size == 0)
        continue;

      // All aggregates smaller than 8 bytes must be passed right-justified.
      if (Size==1 || Size==2 || Size==4) {
        EVT VT = (Size==1) ? MVT::i8 : ((Size==2) ? MVT::i16 : MVT::i32);
        if (GPR_idx != NumGPRs) {
          SDValue Load = DAG.getExtLoad(ISD::EXTLOAD, dl, PtrVT, Chain, Arg,
                                        MachinePointerInfo(), VT,
                                        false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));

          ArgOffset += PtrByteSize;
          continue;
        }
      }

      if (GPR_idx == NumGPRs && Size < 8) {
        SDValue Const = DAG.getConstant(PtrByteSize - Size,
                                        PtrOff.getValueType());
        SDValue AddPtr = DAG.getNode(ISD::ADD, dl, PtrVT, PtrOff, Const);
        Chain = CallSeqStart = createMemcpyOutsideCallSeq(Arg, AddPtr,
                                                          CallSeqStart,
                                                          Flags, DAG, dl);
        ArgOffset += PtrByteSize;
        continue;
      }
      // Copy entire object into memory.  There are cases where gcc-generated
      // code assumes it is there, even if it could be put entirely into
      // registers.  (This is not what the doc says.)

      // FIXME: The above statement is likely due to a misunderstanding of the
      // documents.  All arguments must be copied into the parameter area BY
      // THE CALLEE in the event that the callee takes the address of any
      // formal argument.  That has not yet been implemented.  However, it is
      // reasonable to use the stack area as a staging area for the register
      // load.

      // Skip this for small aggregates, as we will use the same slot for a
      // right-justified copy, below.
      if (Size >= 8)
        Chain = CallSeqStart = createMemcpyOutsideCallSeq(Arg, PtrOff,
                                                          CallSeqStart,
                                                          Flags, DAG, dl);

      // When a register is available, pass a small aggregate right-justified.
      if (Size < 8 && GPR_idx != NumGPRs) {
        // The easiest way to get this right-justified in a register
        // is to copy the structure into the rightmost portion of a
        // local variable slot, then load the whole slot into the
        // register.
        // FIXME: The memcpy seems to produce pretty awful code for
        // small aggregates, particularly for packed ones.
        // FIXME: It would be preferable to use the slot in the 
        // parameter save area instead of a new local variable.
        SDValue Const = DAG.getConstant(8 - Size, PtrOff.getValueType());
        SDValue AddPtr = DAG.getNode(ISD::ADD, dl, PtrVT, PtrOff, Const);
        Chain = CallSeqStart = createMemcpyOutsideCallSeq(Arg, AddPtr,
                                                          CallSeqStart,
                                                          Flags, DAG, dl);

        // Load the slot into the register.
        SDValue Load = DAG.getLoad(PtrVT, dl, Chain, PtrOff,
                                   MachinePointerInfo(),
                                   false, false, false, 0);
        MemOpChains.push_back(Load.getValue(1));
        RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));

        // Done with this argument.
        ArgOffset += PtrByteSize;
        continue;
      }

      // For aggregates larger than PtrByteSize, copy the pieces of the
      // object that fit into registers from the parameter save area.
      for (unsigned j=0; j<Size; j+=PtrByteSize) {
        SDValue Const = DAG.getConstant(j, PtrOff.getValueType());
        SDValue AddArg = DAG.getNode(ISD::ADD, dl, PtrVT, Arg, Const);
        if (GPR_idx != NumGPRs) {
          SDValue Load = DAG.getLoad(PtrVT, dl, Chain, AddArg,
                                     MachinePointerInfo(),
                                     false, false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
          ArgOffset += PtrByteSize;
        } else {
          ArgOffset += ((Size - j + PtrByteSize-1)/PtrByteSize)*PtrByteSize;
          break;
        }
      }
      continue;
    }

    switch (Arg.getValueType().getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unexpected ValueType for argument!");
    case MVT::i32:
    case MVT::i64:
      if (GPR_idx != NumGPRs) {
        RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Arg));
      } else {
        LowerMemOpCallTo(DAG, MF, Chain, Arg, PtrOff, SPDiff, ArgOffset,
                         true, isTailCall, false, MemOpChains,
                         TailCallArguments, dl);
      }
      ArgOffset += PtrByteSize;
      break;
    case MVT::f32:
    case MVT::f64:
      if (FPR_idx != NumFPRs) {
        RegsToPass.push_back(std::make_pair(FPR[FPR_idx++], Arg));

        if (isVarArg) {
          // A single float or an aggregate containing only a single float
          // must be passed right-justified in the stack doubleword, and
          // in the GPR, if one is available.
          SDValue StoreOff;
          if (Arg.getValueType().getSimpleVT().SimpleTy == MVT::f32) {
            SDValue ConstFour = DAG.getConstant(4, PtrOff.getValueType());
            StoreOff = DAG.getNode(ISD::ADD, dl, PtrVT, PtrOff, ConstFour);
          } else
            StoreOff = PtrOff;

          SDValue Store = DAG.getStore(Chain, dl, Arg, StoreOff,
                                       MachinePointerInfo(), false, false, 0);
          MemOpChains.push_back(Store);

          // Float varargs are always shadowed in available integer registers
          if (GPR_idx != NumGPRs) {
            SDValue Load = DAG.getLoad(PtrVT, dl, Store, PtrOff,
                                       MachinePointerInfo(), false, false,
                                       false, 0);
            MemOpChains.push_back(Load.getValue(1));
            RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
          }
        } else if (GPR_idx != NumGPRs)
          // If we have any FPRs remaining, we may also have GPRs remaining.
          ++GPR_idx;
      } else {
        // Single-precision floating-point values are mapped to the
        // second (rightmost) word of the stack doubleword.
        if (Arg.getValueType() == MVT::f32) {
          SDValue ConstFour = DAG.getConstant(4, PtrOff.getValueType());
          PtrOff = DAG.getNode(ISD::ADD, dl, PtrVT, PtrOff, ConstFour);
        }

        LowerMemOpCallTo(DAG, MF, Chain, Arg, PtrOff, SPDiff, ArgOffset,
                         true, isTailCall, false, MemOpChains,
                         TailCallArguments, dl);
      }
      ArgOffset += 8;
      break;
    case MVT::v4f32:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      if (isVarArg) {
        // These go aligned on the stack, or in the corresponding R registers
        // when within range.  The Darwin PPC ABI doc claims they also go in
        // V registers; in fact gcc does this only for arguments that are
        // prototyped, not for those that match the ...  We do it for all
        // arguments, seems to work.
        while (ArgOffset % 16 !=0) {
          ArgOffset += PtrByteSize;
          if (GPR_idx != NumGPRs)
            GPR_idx++;
        }
        // We could elide this store in the case where the object fits
        // entirely in R registers.  Maybe later.
        PtrOff = DAG.getNode(ISD::ADD, dl, PtrVT, StackPtr,
                            DAG.getConstant(ArgOffset, PtrVT));
        SDValue Store = DAG.getStore(Chain, dl, Arg, PtrOff,
                                     MachinePointerInfo(), false, false, 0);
        MemOpChains.push_back(Store);
        if (VR_idx != NumVRs) {
          SDValue Load = DAG.getLoad(MVT::v4f32, dl, Store, PtrOff,
                                     MachinePointerInfo(),
                                     false, false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(VR[VR_idx++], Load));
        }
        ArgOffset += 16;
        for (unsigned i=0; i<16; i+=PtrByteSize) {
          if (GPR_idx == NumGPRs)
            break;
          SDValue Ix = DAG.getNode(ISD::ADD, dl, PtrVT, PtrOff,
                                  DAG.getConstant(i, PtrVT));
          SDValue Load = DAG.getLoad(PtrVT, dl, Store, Ix, MachinePointerInfo(),
                                     false, false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
        }
        break;
      }

      // Non-varargs Altivec params generally go in registers, but have
      // stack space allocated at the end.
      if (VR_idx != NumVRs) {
        // Doesn't have GPR space allocated.
        RegsToPass.push_back(std::make_pair(VR[VR_idx++], Arg));
      } else {
        LowerMemOpCallTo(DAG, MF, Chain, Arg, PtrOff, SPDiff, ArgOffset,
                         true, isTailCall, true, MemOpChains,
                         TailCallArguments, dl);
        ArgOffset += 16;
      }
      break;
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Check if this is an indirect call (MTCTR/BCTRL).
  // See PrepareCall() for more information about calls through function
  // pointers in the 64-bit SVR4 ABI.
  if (!isTailCall &&
      !dyn_cast<GlobalAddressSDNode>(Callee) &&
      !dyn_cast<ExternalSymbolSDNode>(Callee) &&
      !isBLACompatibleAddress(Callee, DAG)) {
    // Load r2 into a virtual register and store it to the TOC save area.
    SDValue Val = DAG.getCopyFromReg(Chain, dl, PPC::X2, MVT::i64);
    // TOC save area offset.
    SDValue PtrOff = DAG.getIntPtrConstant(40);
    SDValue AddPtr = DAG.getNode(ISD::ADD, dl, PtrVT, StackPtr, PtrOff);
    Chain = DAG.getStore(Val.getValue(1), dl, Val, AddPtr, MachinePointerInfo(),
                         false, false, 0);
    // R12 must contain the address of an indirect callee.  This does not
    // mean the MTCTR instruction must use R12; it's easier to model this
    // as an extra parameter, so do that.
    RegsToPass.push_back(std::make_pair((unsigned)PPC::X12, Callee));
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  if (isTailCall)
    PrepareTailCall(DAG, InFlag, Chain, dl, true, SPDiff, NumBytes, LROp,
                    FPOp, true, TailCallArguments);

  return FinishCall(CallConv, dl, isTailCall, isVarArg, DAG,
                    RegsToPass, InFlag, Chain, Callee, SPDiff, NumBytes,
                    Ins, InVals);
}

SDValue
PPCTargetLowering::LowerCall_Darwin(SDValue Chain, SDValue Callee,
                                    CallingConv::ID CallConv, bool isVarArg,
                                    bool isTailCall,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const {

  unsigned NumOps = Outs.size();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  bool isPPC64 = PtrVT == MVT::i64;
  unsigned PtrByteSize = isPPC64 ? 8 : 4;

  MachineFunction &MF = DAG.getMachineFunction();

  // Mark this function as potentially containing a function that contains a
  // tail call. As a consequence the frame pointer will be used for dynamicalloc
  // and restoring the callers stack pointer in this functions epilog. This is
  // done because by tail calling the called function might overwrite the value
  // in this function's (MF) stack pointer stack slot 0(SP).
  if (getTargetMachine().Options.GuaranteedTailCallOpt &&
      CallConv == CallingConv::Fast)
    MF.getInfo<PPCFunctionInfo>()->setHasFastCall();

  unsigned nAltivecParamsAtEnd = 0;

  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.  We start with 24/48 bytes, which is
  // prereserved space for [SP][CR][LR][3 x unused].
  unsigned NumBytes =
    CalculateParameterAndLinkageAreaSize(DAG, isPPC64, isVarArg, CallConv,
                                         Outs, OutVals,
                                         nAltivecParamsAtEnd);

  // Calculate by how many bytes the stack has to be adjusted in case of tail
  // call optimization.
  int SPDiff = CalculateTailCallSPDiff(DAG, isTailCall, NumBytes);

  // To protect arguments on the stack from being clobbered in a tail call,
  // force all the loads to happen before doing any other lowering.
  if (isTailCall)
    Chain = DAG.getStackArgumentTokenFactor(Chain);

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));
  SDValue CallSeqStart = Chain;

  // Load the return address and frame pointer so it can be move somewhere else
  // later.
  SDValue LROp, FPOp;
  Chain = EmitTailCallLoadFPAndRetAddr(DAG, SPDiff, Chain, LROp, FPOp, true,
                                       dl);

  // Set up a copy of the stack pointer for use loading and storing any
  // arguments that may not fit in the registers available for argument
  // passing.
  SDValue StackPtr;
  if (isPPC64)
    StackPtr = DAG.getRegister(PPC::X1, MVT::i64);
  else
    StackPtr = DAG.getRegister(PPC::R1, MVT::i32);

  // Figure out which arguments are going to go in registers, and which in
  // memory.  Also, if this is a vararg function, floating point operations
  // must be stored to our stack, and loaded into integer regs as well, if
  // any integer regs are available for argument passing.
  unsigned ArgOffset = PPCFrameLowering::getLinkageSize(isPPC64, true);
  unsigned GPR_idx = 0, FPR_idx = 0, VR_idx = 0;

  static const uint16_t GPR_32[] = {           // 32-bit registers.
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const uint16_t GPR_64[] = {           // 64-bit registers.
    PPC::X3, PPC::X4, PPC::X5, PPC::X6,
    PPC::X7, PPC::X8, PPC::X9, PPC::X10,
  };
  static const uint16_t *FPR = GetFPR();

  static const uint16_t VR[] = {
    PPC::V2, PPC::V3, PPC::V4, PPC::V5, PPC::V6, PPC::V7, PPC::V8,
    PPC::V9, PPC::V10, PPC::V11, PPC::V12, PPC::V13
  };
  const unsigned NumGPRs = array_lengthof(GPR_32);
  const unsigned NumFPRs = 13;
  const unsigned NumVRs  = array_lengthof(VR);

  const uint16_t *GPR = isPPC64 ? GPR_64 : GPR_32;

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<TailCallArgumentInfo, 8> TailCallArguments;

  SmallVector<SDValue, 8> MemOpChains;
  for (unsigned i = 0; i != NumOps; ++i) {
    SDValue Arg = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // PtrOff will be used to store the current argument to the stack if a
    // register cannot be found for it.
    SDValue PtrOff;

    PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());

    PtrOff = DAG.getNode(ISD::ADD, dl, PtrVT, StackPtr, PtrOff);

    // On PPC64, promote integers to 64-bit values.
    if (isPPC64 && Arg.getValueType() == MVT::i32) {
      // FIXME: Should this use ANY_EXTEND if neither sext nor zext?
      unsigned ExtOp = Flags.isSExt() ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
      Arg = DAG.getNode(ExtOp, dl, MVT::i64, Arg);
    }

    // FIXME memcpy is used way more than necessary.  Correctness first.
    // Note: "by value" is code for passing a structure by value, not
    // basic types.
    if (Flags.isByVal()) {
      unsigned Size = Flags.getByValSize();
      // Very small objects are passed right-justified.  Everything else is
      // passed left-justified.
      if (Size==1 || Size==2) {
        EVT VT = (Size==1) ? MVT::i8 : MVT::i16;
        if (GPR_idx != NumGPRs) {
          SDValue Load = DAG.getExtLoad(ISD::EXTLOAD, dl, PtrVT, Chain, Arg,
                                        MachinePointerInfo(), VT,
                                        false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));

          ArgOffset += PtrByteSize;
        } else {
          SDValue Const = DAG.getConstant(PtrByteSize - Size,
                                          PtrOff.getValueType());
          SDValue AddPtr = DAG.getNode(ISD::ADD, dl, PtrVT, PtrOff, Const);
          Chain = CallSeqStart = createMemcpyOutsideCallSeq(Arg, AddPtr,
                                                            CallSeqStart,
                                                            Flags, DAG, dl);
          ArgOffset += PtrByteSize;
        }
        continue;
      }
      // Copy entire object into memory.  There are cases where gcc-generated
      // code assumes it is there, even if it could be put entirely into
      // registers.  (This is not what the doc says.)
      Chain = CallSeqStart = createMemcpyOutsideCallSeq(Arg, PtrOff,
                                                        CallSeqStart,
                                                        Flags, DAG, dl);

      // For small aggregates (Darwin only) and aggregates >= PtrByteSize,
      // copy the pieces of the object that fit into registers from the
      // parameter save area.
      for (unsigned j=0; j<Size; j+=PtrByteSize) {
        SDValue Const = DAG.getConstant(j, PtrOff.getValueType());
        SDValue AddArg = DAG.getNode(ISD::ADD, dl, PtrVT, Arg, Const);
        if (GPR_idx != NumGPRs) {
          SDValue Load = DAG.getLoad(PtrVT, dl, Chain, AddArg,
                                     MachinePointerInfo(),
                                     false, false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
          ArgOffset += PtrByteSize;
        } else {
          ArgOffset += ((Size - j + PtrByteSize-1)/PtrByteSize)*PtrByteSize;
          break;
        }
      }
      continue;
    }

    switch (Arg.getValueType().getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unexpected ValueType for argument!");
    case MVT::i32:
    case MVT::i64:
      if (GPR_idx != NumGPRs) {
        RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Arg));
      } else {
        LowerMemOpCallTo(DAG, MF, Chain, Arg, PtrOff, SPDiff, ArgOffset,
                         isPPC64, isTailCall, false, MemOpChains,
                         TailCallArguments, dl);
      }
      ArgOffset += PtrByteSize;
      break;
    case MVT::f32:
    case MVT::f64:
      if (FPR_idx != NumFPRs) {
        RegsToPass.push_back(std::make_pair(FPR[FPR_idx++], Arg));

        if (isVarArg) {
          SDValue Store = DAG.getStore(Chain, dl, Arg, PtrOff,
                                       MachinePointerInfo(), false, false, 0);
          MemOpChains.push_back(Store);

          // Float varargs are always shadowed in available integer registers
          if (GPR_idx != NumGPRs) {
            SDValue Load = DAG.getLoad(PtrVT, dl, Store, PtrOff,
                                       MachinePointerInfo(), false, false,
                                       false, 0);
            MemOpChains.push_back(Load.getValue(1));
            RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
          }
          if (GPR_idx != NumGPRs && Arg.getValueType() == MVT::f64 && !isPPC64){
            SDValue ConstFour = DAG.getConstant(4, PtrOff.getValueType());
            PtrOff = DAG.getNode(ISD::ADD, dl, PtrVT, PtrOff, ConstFour);
            SDValue Load = DAG.getLoad(PtrVT, dl, Store, PtrOff,
                                       MachinePointerInfo(),
                                       false, false, false, 0);
            MemOpChains.push_back(Load.getValue(1));
            RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
          }
        } else {
          // If we have any FPRs remaining, we may also have GPRs remaining.
          // Args passed in FPRs consume either 1 (f32) or 2 (f64) available
          // GPRs.
          if (GPR_idx != NumGPRs)
            ++GPR_idx;
          if (GPR_idx != NumGPRs && Arg.getValueType() == MVT::f64 &&
              !isPPC64)  // PPC64 has 64-bit GPR's obviously :)
            ++GPR_idx;
        }
      } else
        LowerMemOpCallTo(DAG, MF, Chain, Arg, PtrOff, SPDiff, ArgOffset,
                         isPPC64, isTailCall, false, MemOpChains,
                         TailCallArguments, dl);
      if (isPPC64)
        ArgOffset += 8;
      else
        ArgOffset += Arg.getValueType() == MVT::f32 ? 4 : 8;
      break;
    case MVT::v4f32:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      if (isVarArg) {
        // These go aligned on the stack, or in the corresponding R registers
        // when within range.  The Darwin PPC ABI doc claims they also go in
        // V registers; in fact gcc does this only for arguments that are
        // prototyped, not for those that match the ...  We do it for all
        // arguments, seems to work.
        while (ArgOffset % 16 !=0) {
          ArgOffset += PtrByteSize;
          if (GPR_idx != NumGPRs)
            GPR_idx++;
        }
        // We could elide this store in the case where the object fits
        // entirely in R registers.  Maybe later.
        PtrOff = DAG.getNode(ISD::ADD, dl, PtrVT, StackPtr,
                            DAG.getConstant(ArgOffset, PtrVT));
        SDValue Store = DAG.getStore(Chain, dl, Arg, PtrOff,
                                     MachinePointerInfo(), false, false, 0);
        MemOpChains.push_back(Store);
        if (VR_idx != NumVRs) {
          SDValue Load = DAG.getLoad(MVT::v4f32, dl, Store, PtrOff,
                                     MachinePointerInfo(),
                                     false, false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(VR[VR_idx++], Load));
        }
        ArgOffset += 16;
        for (unsigned i=0; i<16; i+=PtrByteSize) {
          if (GPR_idx == NumGPRs)
            break;
          SDValue Ix = DAG.getNode(ISD::ADD, dl, PtrVT, PtrOff,
                                  DAG.getConstant(i, PtrVT));
          SDValue Load = DAG.getLoad(PtrVT, dl, Store, Ix, MachinePointerInfo(),
                                     false, false, false, 0);
          MemOpChains.push_back(Load.getValue(1));
          RegsToPass.push_back(std::make_pair(GPR[GPR_idx++], Load));
        }
        break;
      }

      // Non-varargs Altivec params generally go in registers, but have
      // stack space allocated at the end.
      if (VR_idx != NumVRs) {
        // Doesn't have GPR space allocated.
        RegsToPass.push_back(std::make_pair(VR[VR_idx++], Arg));
      } else if (nAltivecParamsAtEnd==0) {
        // We are emitting Altivec params in order.
        LowerMemOpCallTo(DAG, MF, Chain, Arg, PtrOff, SPDiff, ArgOffset,
                         isPPC64, isTailCall, true, MemOpChains,
                         TailCallArguments, dl);
        ArgOffset += 16;
      }
      break;
    }
  }
  // If all Altivec parameters fit in registers, as they usually do,
  // they get stack space following the non-Altivec parameters.  We
  // don't track this here because nobody below needs it.
  // If there are more Altivec parameters than fit in registers emit
  // the stores here.
  if (!isVarArg && nAltivecParamsAtEnd > NumVRs) {
    unsigned j = 0;
    // Offset is aligned; skip 1st 12 params which go in V registers.
    ArgOffset = ((ArgOffset+15)/16)*16;
    ArgOffset += 12*16;
    for (unsigned i = 0; i != NumOps; ++i) {
      SDValue Arg = OutVals[i];
      EVT ArgType = Outs[i].VT;
      if (ArgType==MVT::v4f32 || ArgType==MVT::v4i32 ||
          ArgType==MVT::v8i16 || ArgType==MVT::v16i8) {
        if (++j > NumVRs) {
          SDValue PtrOff;
          // We are emitting Altivec params in order.
          LowerMemOpCallTo(DAG, MF, Chain, Arg, PtrOff, SPDiff, ArgOffset,
                           isPPC64, isTailCall, true, MemOpChains,
                           TailCallArguments, dl);
          ArgOffset += 16;
        }
      }
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // On Darwin, R12 must contain the address of an indirect callee.  This does
  // not mean the MTCTR instruction must use R12; it's easier to model this as
  // an extra parameter, so do that.
  if (!isTailCall &&
      !dyn_cast<GlobalAddressSDNode>(Callee) &&
      !dyn_cast<ExternalSymbolSDNode>(Callee) &&
      !isBLACompatibleAddress(Callee, DAG))
    RegsToPass.push_back(std::make_pair((unsigned)(isPPC64 ? PPC::X12 :
                                                   PPC::R12), Callee));

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  if (isTailCall)
    PrepareTailCall(DAG, InFlag, Chain, dl, isPPC64, SPDiff, NumBytes, LROp,
                    FPOp, true, TailCallArguments);

  return FinishCall(CallConv, dl, isTailCall, isVarArg, DAG,
                    RegsToPass, InFlag, Chain, Callee, SPDiff, NumBytes,
                    Ins, InVals);
}

bool
PPCTargetLowering::CanLowerReturn(CallingConv::ID CallConv,
                                  MachineFunction &MF, bool isVarArg,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(),
                 RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_PPC);
}

SDValue
PPCTargetLowering::LowerReturn(SDValue Chain,
                               CallingConv::ID CallConv, bool isVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               DebugLoc dl, SelectionDAG &DAG) const {

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_PPC);

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

    SDValue Arg = OutVals[i];

    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    }

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), Arg, Flag);
    Flag = Chain.getValue(1);
  }

  if (Flag.getNode())
    return DAG.getNode(PPCISD::RET_FLAG, dl, MVT::Other, Chain, Flag);
  else
    return DAG.getNode(PPCISD::RET_FLAG, dl, MVT::Other, Chain);
}

SDValue PPCTargetLowering::LowerSTACKRESTORE(SDValue Op, SelectionDAG &DAG,
                                   const PPCSubtarget &Subtarget) const {
  // When we pop the dynamic allocation we need to restore the SP link.
  DebugLoc dl = Op.getDebugLoc();

  // Get the corect type for pointers.
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

  // Construct the stack pointer operand.
  bool isPPC64 = Subtarget.isPPC64();
  unsigned SP = isPPC64 ? PPC::X1 : PPC::R1;
  SDValue StackPtr = DAG.getRegister(SP, PtrVT);

  // Get the operands for the STACKRESTORE.
  SDValue Chain = Op.getOperand(0);
  SDValue SaveSP = Op.getOperand(1);

  // Load the old link SP.
  SDValue LoadLinkSP = DAG.getLoad(PtrVT, dl, Chain, StackPtr,
                                   MachinePointerInfo(),
                                   false, false, false, 0);

  // Restore the stack pointer.
  Chain = DAG.getCopyToReg(LoadLinkSP.getValue(1), dl, SP, SaveSP);

  // Store the old link SP.
  return DAG.getStore(Chain, dl, LoadLinkSP, StackPtr, MachinePointerInfo(),
                      false, false, 0);
}



SDValue
PPCTargetLowering::getReturnAddrFrameIndex(SelectionDAG & DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  bool isPPC64 = PPCSubTarget.isPPC64();
  bool isDarwinABI = PPCSubTarget.isDarwinABI();
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

  // Get current frame pointer save index.  The users of this index will be
  // primarily DYNALLOC instructions.
  PPCFunctionInfo *FI = MF.getInfo<PPCFunctionInfo>();
  int RASI = FI->getReturnAddrSaveIndex();

  // If the frame pointer save index hasn't been defined yet.
  if (!RASI) {
    // Find out what the fix offset of the frame pointer save area.
    int LROffset = PPCFrameLowering::getReturnSaveOffset(isPPC64, isDarwinABI);
    // Allocate the frame index for frame pointer save area.
    RASI = MF.getFrameInfo()->CreateFixedObject(isPPC64? 8 : 4, LROffset, true);
    // Save the result.
    FI->setReturnAddrSaveIndex(RASI);
  }
  return DAG.getFrameIndex(RASI, PtrVT);
}

SDValue
PPCTargetLowering::getFramePointerFrameIndex(SelectionDAG & DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  bool isPPC64 = PPCSubTarget.isPPC64();
  bool isDarwinABI = PPCSubTarget.isDarwinABI();
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

  // Get current frame pointer save index.  The users of this index will be
  // primarily DYNALLOC instructions.
  PPCFunctionInfo *FI = MF.getInfo<PPCFunctionInfo>();
  int FPSI = FI->getFramePointerSaveIndex();

  // If the frame pointer save index hasn't been defined yet.
  if (!FPSI) {
    // Find out what the fix offset of the frame pointer save area.
    int FPOffset = PPCFrameLowering::getFramePointerSaveOffset(isPPC64,
                                                           isDarwinABI);

    // Allocate the frame index for frame pointer save area.
    FPSI = MF.getFrameInfo()->CreateFixedObject(isPPC64? 8 : 4, FPOffset, true);
    // Save the result.
    FI->setFramePointerSaveIndex(FPSI);
  }
  return DAG.getFrameIndex(FPSI, PtrVT);
}

SDValue PPCTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                         SelectionDAG &DAG,
                                         const PPCSubtarget &Subtarget) const {
  // Get the inputs.
  SDValue Chain = Op.getOperand(0);
  SDValue Size  = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();

  // Get the corect type for pointers.
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  // Negate the size.
  SDValue NegSize = DAG.getNode(ISD::SUB, dl, PtrVT,
                                  DAG.getConstant(0, PtrVT), Size);
  // Construct a node for the frame pointer save index.
  SDValue FPSIdx = getFramePointerFrameIndex(DAG);
  // Build a DYNALLOC node.
  SDValue Ops[3] = { Chain, NegSize, FPSIdx };
  SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);
  return DAG.getNode(PPCISD::DYNALLOC, dl, VTs, Ops, 3);
}

/// LowerSELECT_CC - Lower floating point select_cc's into fsel instruction when
/// possible.
SDValue PPCTargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  // Not FP? Not a fsel.
  if (!Op.getOperand(0).getValueType().isFloatingPoint() ||
      !Op.getOperand(2).getValueType().isFloatingPoint())
    return Op;

  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();

  // Cannot handle SETEQ/SETNE.
  if (CC == ISD::SETEQ || CC == ISD::SETNE) return Op;

  EVT ResVT = Op.getValueType();
  EVT CmpVT = Op.getOperand(0).getValueType();
  SDValue LHS = Op.getOperand(0), RHS = Op.getOperand(1);
  SDValue TV  = Op.getOperand(2), FV  = Op.getOperand(3);
  DebugLoc dl = Op.getDebugLoc();

  // If the RHS of the comparison is a 0.0, we don't need to do the
  // subtraction at all.
  if (isFloatingPointZero(RHS))
    switch (CC) {
    default: break;       // SETUO etc aren't handled by fsel.
    case ISD::SETULT:
    case ISD::SETLT:
      std::swap(TV, FV);  // fsel is natively setge, swap operands for setlt
    case ISD::SETOGE:
    case ISD::SETGE:
      if (LHS.getValueType() == MVT::f32)   // Comparison is always 64-bits
        LHS = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, LHS);
      return DAG.getNode(PPCISD::FSEL, dl, ResVT, LHS, TV, FV);
    case ISD::SETUGT:
    case ISD::SETGT:
      std::swap(TV, FV);  // fsel is natively setge, swap operands for setlt
    case ISD::SETOLE:
    case ISD::SETLE:
      if (LHS.getValueType() == MVT::f32)   // Comparison is always 64-bits
        LHS = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, LHS);
      return DAG.getNode(PPCISD::FSEL, dl, ResVT,
                         DAG.getNode(ISD::FNEG, dl, MVT::f64, LHS), TV, FV);
    }

  SDValue Cmp;
  switch (CC) {
  default: break;       // SETUO etc aren't handled by fsel.
  case ISD::SETULT:
  case ISD::SETLT:
    Cmp = DAG.getNode(ISD::FSUB, dl, CmpVT, LHS, RHS);
    if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
      Cmp = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, dl, ResVT, Cmp, FV, TV);
  case ISD::SETOGE:
  case ISD::SETGE:
    Cmp = DAG.getNode(ISD::FSUB, dl, CmpVT, LHS, RHS);
    if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
      Cmp = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, dl, ResVT, Cmp, TV, FV);
  case ISD::SETUGT:
  case ISD::SETGT:
    Cmp = DAG.getNode(ISD::FSUB, dl, CmpVT, RHS, LHS);
    if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
      Cmp = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, dl, ResVT, Cmp, FV, TV);
  case ISD::SETOLE:
  case ISD::SETLE:
    Cmp = DAG.getNode(ISD::FSUB, dl, CmpVT, RHS, LHS);
    if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
      Cmp = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, dl, ResVT, Cmp, TV, FV);
  }
  return Op;
}

// FIXME: Split this code up when LegalizeDAGTypes lands.
SDValue PPCTargetLowering::LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG,
                                           DebugLoc dl) const {
  assert(Op.getOperand(0).getValueType().isFloatingPoint());
  SDValue Src = Op.getOperand(0);
  if (Src.getValueType() == MVT::f32)
    Src = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, Src);

  SDValue Tmp;
  switch (Op.getValueType().getSimpleVT().SimpleTy) {
  default: llvm_unreachable("Unhandled FP_TO_INT type in custom expander!");
  case MVT::i32:
    Tmp = DAG.getNode(Op.getOpcode()==ISD::FP_TO_SINT ? PPCISD::FCTIWZ :
                                                         PPCISD::FCTIDZ,
                      dl, MVT::f64, Src);
    break;
  case MVT::i64:
    Tmp = DAG.getNode(PPCISD::FCTIDZ, dl, MVT::f64, Src);
    break;
  }

  // Convert the FP value to an int value through memory.
  SDValue FIPtr = DAG.CreateStackTemporary(MVT::f64);

  // Emit a store to the stack slot.
  SDValue Chain = DAG.getStore(DAG.getEntryNode(), dl, Tmp, FIPtr,
                               MachinePointerInfo(), false, false, 0);

  // Result is a load from the stack slot.  If loading 4 bytes, make sure to
  // add in a bias.
  if (Op.getValueType() == MVT::i32)
    FIPtr = DAG.getNode(ISD::ADD, dl, FIPtr.getValueType(), FIPtr,
                        DAG.getConstant(4, FIPtr.getValueType()));
  return DAG.getLoad(Op.getValueType(), dl, Chain, FIPtr, MachinePointerInfo(),
                     false, false, false, 0);
}

SDValue PPCTargetLowering::LowerSINT_TO_FP(SDValue Op,
                                           SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  // Don't handle ppc_fp128 here; let it be lowered to a libcall.
  if (Op.getValueType() != MVT::f32 && Op.getValueType() != MVT::f64)
    return SDValue();

  if (Op.getOperand(0).getValueType() == MVT::i64) {
    SDValue SINT = Op.getOperand(0);
    // When converting to single-precision, we actually need to convert
    // to double-precision first and then round to single-precision.
    // To avoid double-rounding effects during that operation, we have
    // to prepare the input operand.  Bits that might be truncated when
    // converting to double-precision are replaced by a bit that won't
    // be lost at this stage, but is below the single-precision rounding
    // position.
    //
    // However, if -enable-unsafe-fp-math is in effect, accept double
    // rounding to avoid the extra overhead.
    if (Op.getValueType() == MVT::f32 &&
        !DAG.getTarget().Options.UnsafeFPMath) {

      // Twiddle input to make sure the low 11 bits are zero.  (If this
      // is the case, we are guaranteed the value will fit into the 53 bit
      // mantissa of an IEEE double-precision value without rounding.)
      // If any of those low 11 bits were not zero originally, make sure
      // bit 12 (value 2048) is set instead, so that the final rounding
      // to single-precision gets the correct result.
      SDValue Round = DAG.getNode(ISD::AND, dl, MVT::i64,
                                  SINT, DAG.getConstant(2047, MVT::i64));
      Round = DAG.getNode(ISD::ADD, dl, MVT::i64,
                          Round, DAG.getConstant(2047, MVT::i64));
      Round = DAG.getNode(ISD::OR, dl, MVT::i64, Round, SINT);
      Round = DAG.getNode(ISD::AND, dl, MVT::i64,
                          Round, DAG.getConstant(-2048, MVT::i64));

      // However, we cannot use that value unconditionally: if the magnitude
      // of the input value is small, the bit-twiddling we did above might
      // end up visibly changing the output.  Fortunately, in that case, we
      // don't need to twiddle bits since the original input will convert
      // exactly to double-precision floating-point already.  Therefore,
      // construct a conditional to use the original value if the top 11
      // bits are all sign-bit copies, and use the rounded value computed
      // above otherwise.
      SDValue Cond = DAG.getNode(ISD::SRA, dl, MVT::i64,
                                 SINT, DAG.getConstant(53, MVT::i32));
      Cond = DAG.getNode(ISD::ADD, dl, MVT::i64,
                         Cond, DAG.getConstant(1, MVT::i64));
      Cond = DAG.getSetCC(dl, MVT::i32,
                          Cond, DAG.getConstant(1, MVT::i64), ISD::SETUGT);

      SINT = DAG.getNode(ISD::SELECT, dl, MVT::i64, Cond, Round, SINT);
    }
    SDValue Bits = DAG.getNode(ISD::BITCAST, dl, MVT::f64, SINT);
    SDValue FP = DAG.getNode(PPCISD::FCFID, dl, MVT::f64, Bits);
    if (Op.getValueType() == MVT::f32)
      FP = DAG.getNode(ISD::FP_ROUND, dl,
                       MVT::f32, FP, DAG.getIntPtrConstant(0));
    return FP;
  }

  assert(Op.getOperand(0).getValueType() == MVT::i32 &&
         "Unhandled SINT_TO_FP type in custom expander!");
  // Since we only generate this in 64-bit mode, we can take advantage of
  // 64-bit registers.  In particular, sign extend the input value into the
  // 64-bit register with extsw, store the WHOLE 64-bit value into the stack
  // then lfd it and fcfid it.
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *FrameInfo = MF.getFrameInfo();
  int FrameIdx = FrameInfo->CreateStackObject(8, 8, false);
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  SDValue FIdx = DAG.getFrameIndex(FrameIdx, PtrVT);

  SDValue Ext64 = DAG.getNode(PPCISD::EXTSW_32, dl, MVT::i32,
                                Op.getOperand(0));

  // STD the extended value into the stack slot.
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FrameIdx),
                            MachineMemOperand::MOStore, 8, 8);
  SDValue Ops[] = { DAG.getEntryNode(), Ext64, FIdx };
  SDValue Store =
    DAG.getMemIntrinsicNode(PPCISD::STD_32, dl, DAG.getVTList(MVT::Other),
                            Ops, 4, MVT::i64, MMO);
  // Load the value as a double.
  SDValue Ld = DAG.getLoad(MVT::f64, dl, Store, FIdx, MachinePointerInfo(),
                           false, false, false, 0);

  // FCFID it and return it.
  SDValue FP = DAG.getNode(PPCISD::FCFID, dl, MVT::f64, Ld);
  if (Op.getValueType() == MVT::f32)
    FP = DAG.getNode(ISD::FP_ROUND, dl, MVT::f32, FP, DAG.getIntPtrConstant(0));
  return FP;
}

SDValue PPCTargetLowering::LowerFLT_ROUNDS_(SDValue Op,
                                            SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  /*
   The rounding mode is in bits 30:31 of FPSR, and has the following
   settings:
     00 Round to nearest
     01 Round to 0
     10 Round to +inf
     11 Round to -inf

  FLT_ROUNDS, on the other hand, expects the following:
    -1 Undefined
     0 Round to 0
     1 Round to nearest
     2 Round to +inf
     3 Round to -inf

  To perform the conversion, we do:
    ((FPSCR & 0x3) ^ ((~FPSCR & 0x3) >> 1))
  */

  MachineFunction &MF = DAG.getMachineFunction();
  EVT VT = Op.getValueType();
  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  std::vector<EVT> NodeTys;
  SDValue MFFSreg, InFlag;

  // Save FP Control Word to register
  NodeTys.push_back(MVT::f64);    // return register
  NodeTys.push_back(MVT::Glue);   // unused in this context
  SDValue Chain = DAG.getNode(PPCISD::MFFS, dl, NodeTys, &InFlag, 0);

  // Save FP register to stack slot
  int SSFI = MF.getFrameInfo()->CreateStackObject(8, 8, false);
  SDValue StackSlot = DAG.getFrameIndex(SSFI, PtrVT);
  SDValue Store = DAG.getStore(DAG.getEntryNode(), dl, Chain,
                               StackSlot, MachinePointerInfo(), false, false,0);

  // Load FP Control Word from low 32 bits of stack slot.
  SDValue Four = DAG.getConstant(4, PtrVT);
  SDValue Addr = DAG.getNode(ISD::ADD, dl, PtrVT, StackSlot, Four);
  SDValue CWD = DAG.getLoad(MVT::i32, dl, Store, Addr, MachinePointerInfo(),
                            false, false, false, 0);

  // Transform as necessary
  SDValue CWD1 =
    DAG.getNode(ISD::AND, dl, MVT::i32,
                CWD, DAG.getConstant(3, MVT::i32));
  SDValue CWD2 =
    DAG.getNode(ISD::SRL, dl, MVT::i32,
                DAG.getNode(ISD::AND, dl, MVT::i32,
                            DAG.getNode(ISD::XOR, dl, MVT::i32,
                                        CWD, DAG.getConstant(3, MVT::i32)),
                            DAG.getConstant(3, MVT::i32)),
                DAG.getConstant(1, MVT::i32));

  SDValue RetVal =
    DAG.getNode(ISD::XOR, dl, MVT::i32, CWD1, CWD2);

  return DAG.getNode((VT.getSizeInBits() < 16 ?
                      ISD::TRUNCATE : ISD::ZERO_EXTEND), dl, VT, RetVal);
}

SDValue PPCTargetLowering::LowerSHL_PARTS(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  unsigned BitWidth = VT.getSizeInBits();
  DebugLoc dl = Op.getDebugLoc();
  assert(Op.getNumOperands() == 3 &&
         VT == Op.getOperand(1).getValueType() &&
         "Unexpected SHL!");

  // Expand into a bunch of logical ops.  Note that these ops
  // depend on the PPC behavior for oversized shift amounts.
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Amt = Op.getOperand(2);
  EVT AmtVT = Amt.getValueType();

  SDValue Tmp1 = DAG.getNode(ISD::SUB, dl, AmtVT,
                             DAG.getConstant(BitWidth, AmtVT), Amt);
  SDValue Tmp2 = DAG.getNode(PPCISD::SHL, dl, VT, Hi, Amt);
  SDValue Tmp3 = DAG.getNode(PPCISD::SRL, dl, VT, Lo, Tmp1);
  SDValue Tmp4 = DAG.getNode(ISD::OR , dl, VT, Tmp2, Tmp3);
  SDValue Tmp5 = DAG.getNode(ISD::ADD, dl, AmtVT, Amt,
                             DAG.getConstant(-BitWidth, AmtVT));
  SDValue Tmp6 = DAG.getNode(PPCISD::SHL, dl, VT, Lo, Tmp5);
  SDValue OutHi = DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp6);
  SDValue OutLo = DAG.getNode(PPCISD::SHL, dl, VT, Lo, Amt);
  SDValue OutOps[] = { OutLo, OutHi };
  return DAG.getMergeValues(OutOps, 2, dl);
}

SDValue PPCTargetLowering::LowerSRL_PARTS(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned BitWidth = VT.getSizeInBits();
  assert(Op.getNumOperands() == 3 &&
         VT == Op.getOperand(1).getValueType() &&
         "Unexpected SRL!");

  // Expand into a bunch of logical ops.  Note that these ops
  // depend on the PPC behavior for oversized shift amounts.
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Amt = Op.getOperand(2);
  EVT AmtVT = Amt.getValueType();

  SDValue Tmp1 = DAG.getNode(ISD::SUB, dl, AmtVT,
                             DAG.getConstant(BitWidth, AmtVT), Amt);
  SDValue Tmp2 = DAG.getNode(PPCISD::SRL, dl, VT, Lo, Amt);
  SDValue Tmp3 = DAG.getNode(PPCISD::SHL, dl, VT, Hi, Tmp1);
  SDValue Tmp4 = DAG.getNode(ISD::OR, dl, VT, Tmp2, Tmp3);
  SDValue Tmp5 = DAG.getNode(ISD::ADD, dl, AmtVT, Amt,
                             DAG.getConstant(-BitWidth, AmtVT));
  SDValue Tmp6 = DAG.getNode(PPCISD::SRL, dl, VT, Hi, Tmp5);
  SDValue OutLo = DAG.getNode(ISD::OR, dl, VT, Tmp4, Tmp6);
  SDValue OutHi = DAG.getNode(PPCISD::SRL, dl, VT, Hi, Amt);
  SDValue OutOps[] = { OutLo, OutHi };
  return DAG.getMergeValues(OutOps, 2, dl);
}

SDValue PPCTargetLowering::LowerSRA_PARTS(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  unsigned BitWidth = VT.getSizeInBits();
  assert(Op.getNumOperands() == 3 &&
         VT == Op.getOperand(1).getValueType() &&
         "Unexpected SRA!");

  // Expand into a bunch of logical ops, followed by a select_cc.
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Amt = Op.getOperand(2);
  EVT AmtVT = Amt.getValueType();

  SDValue Tmp1 = DAG.getNode(ISD::SUB, dl, AmtVT,
                             DAG.getConstant(BitWidth, AmtVT), Amt);
  SDValue Tmp2 = DAG.getNode(PPCISD::SRL, dl, VT, Lo, Amt);
  SDValue Tmp3 = DAG.getNode(PPCISD::SHL, dl, VT, Hi, Tmp1);
  SDValue Tmp4 = DAG.getNode(ISD::OR, dl, VT, Tmp2, Tmp3);
  SDValue Tmp5 = DAG.getNode(ISD::ADD, dl, AmtVT, Amt,
                             DAG.getConstant(-BitWidth, AmtVT));
  SDValue Tmp6 = DAG.getNode(PPCISD::SRA, dl, VT, Hi, Tmp5);
  SDValue OutHi = DAG.getNode(PPCISD::SRA, dl, VT, Hi, Amt);
  SDValue OutLo = DAG.getSelectCC(dl, Tmp5, DAG.getConstant(0, AmtVT),
                                  Tmp4, Tmp6, ISD::SETLE);
  SDValue OutOps[] = { OutLo, OutHi };
  return DAG.getMergeValues(OutOps, 2, dl);
}

//===----------------------------------------------------------------------===//
// Vector related lowering.
//

/// BuildSplatI - Build a canonical splati of Val with an element size of
/// SplatSize.  Cast the result to VT.
static SDValue BuildSplatI(int Val, unsigned SplatSize, EVT VT,
                             SelectionDAG &DAG, DebugLoc dl) {
  assert(Val >= -16 && Val <= 15 && "vsplti is out of range!");

  static const EVT VTys[] = { // canonical VT to use for each size.
    MVT::v16i8, MVT::v8i16, MVT::Other, MVT::v4i32
  };

  EVT ReqVT = VT != MVT::Other ? VT : VTys[SplatSize-1];

  // Force vspltis[hw] -1 to vspltisb -1 to canonicalize.
  if (Val == -1)
    SplatSize = 1;

  EVT CanonicalVT = VTys[SplatSize-1];

  // Build a canonical splat for this value.
  SDValue Elt = DAG.getConstant(Val, MVT::i32);
  SmallVector<SDValue, 8> Ops;
  Ops.assign(CanonicalVT.getVectorNumElements(), Elt);
  SDValue Res = DAG.getNode(ISD::BUILD_VECTOR, dl, CanonicalVT,
                              &Ops[0], Ops.size());
  return DAG.getNode(ISD::BITCAST, dl, ReqVT, Res);
}

/// BuildIntrinsicOp - Return a binary operator intrinsic node with the
/// specified intrinsic ID.
static SDValue BuildIntrinsicOp(unsigned IID, SDValue LHS, SDValue RHS,
                                SelectionDAG &DAG, DebugLoc dl,
                                EVT DestVT = MVT::Other) {
  if (DestVT == MVT::Other) DestVT = LHS.getValueType();
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, DestVT,
                     DAG.getConstant(IID, MVT::i32), LHS, RHS);
}

/// BuildIntrinsicOp - Return a ternary operator intrinsic node with the
/// specified intrinsic ID.
static SDValue BuildIntrinsicOp(unsigned IID, SDValue Op0, SDValue Op1,
                                SDValue Op2, SelectionDAG &DAG,
                                DebugLoc dl, EVT DestVT = MVT::Other) {
  if (DestVT == MVT::Other) DestVT = Op0.getValueType();
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, DestVT,
                     DAG.getConstant(IID, MVT::i32), Op0, Op1, Op2);
}


/// BuildVSLDOI - Return a VECTOR_SHUFFLE that is a vsldoi of the specified
/// amount.  The result has the specified value type.
static SDValue BuildVSLDOI(SDValue LHS, SDValue RHS, unsigned Amt,
                             EVT VT, SelectionDAG &DAG, DebugLoc dl) {
  // Force LHS/RHS to be the right type.
  LHS = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, LHS);
  RHS = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, RHS);

  int Ops[16];
  for (unsigned i = 0; i != 16; ++i)
    Ops[i] = i + Amt;
  SDValue T = DAG.getVectorShuffle(MVT::v16i8, dl, LHS, RHS, Ops);
  return DAG.getNode(ISD::BITCAST, dl, VT, T);
}

// If this is a case we can't handle, return null and let the default
// expansion code take care of it.  If we CAN select this case, and if it
// selects to a single instruction, return Op.  Otherwise, if we can codegen
// this case more efficiently than a constant pool load, lower it to the
// sequence of ops that should be used.
SDValue PPCTargetLowering::LowerBUILD_VECTOR(SDValue Op,
                                             SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  BuildVectorSDNode *BVN = dyn_cast<BuildVectorSDNode>(Op.getNode());
  assert(BVN != 0 && "Expected a BuildVectorSDNode in LowerBUILD_VECTOR");

  // Check if this is a splat of a constant value.
  APInt APSplatBits, APSplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  if (! BVN->isConstantSplat(APSplatBits, APSplatUndef, SplatBitSize,
                             HasAnyUndefs, 0, true) || SplatBitSize > 32)
    return SDValue();

  unsigned SplatBits = APSplatBits.getZExtValue();
  unsigned SplatUndef = APSplatUndef.getZExtValue();
  unsigned SplatSize = SplatBitSize / 8;

  // First, handle single instruction cases.

  // All zeros?
  if (SplatBits == 0) {
    // Canonicalize all zero vectors to be v4i32.
    if (Op.getValueType() != MVT::v4i32 || HasAnyUndefs) {
      SDValue Z = DAG.getConstant(0, MVT::i32);
      Z = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32, Z, Z, Z, Z);
      Op = DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Z);
    }
    return Op;
  }

  // If the sign extended value is in the range [-16,15], use VSPLTI[bhw].
  int32_t SextVal= (int32_t(SplatBits << (32-SplatBitSize)) >>
                    (32-SplatBitSize));
  if (SextVal >= -16 && SextVal <= 15)
    return BuildSplatI(SextVal, SplatSize, Op.getValueType(), DAG, dl);


  // Two instruction sequences.

  // If this value is in the range [-32,30] and is even, use:
  //    tmp = VSPLTI[bhw], result = add tmp, tmp
  if (SextVal >= -32 && SextVal <= 30 && (SextVal & 1) == 0) {
    SDValue Res = BuildSplatI(SextVal >> 1, SplatSize, MVT::Other, DAG, dl);
    Res = DAG.getNode(ISD::ADD, dl, Res.getValueType(), Res, Res);
    return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Res);
  }

  // If this is 0x8000_0000 x 4, turn into vspltisw + vslw.  If it is
  // 0x7FFF_FFFF x 4, turn it into not(0x8000_0000).  This is important
  // for fneg/fabs.
  if (SplatSize == 4 && SplatBits == (0x7FFFFFFF&~SplatUndef)) {
    // Make -1 and vspltisw -1:
    SDValue OnesV = BuildSplatI(-1, 4, MVT::v4i32, DAG, dl);

    // Make the VSLW intrinsic, computing 0x8000_0000.
    SDValue Res = BuildIntrinsicOp(Intrinsic::ppc_altivec_vslw, OnesV,
                                   OnesV, DAG, dl);

    // xor by OnesV to invert it.
    Res = DAG.getNode(ISD::XOR, dl, MVT::v4i32, Res, OnesV);
    return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Res);
  }

  // Check to see if this is a wide variety of vsplti*, binop self cases.
  static const signed char SplatCsts[] = {
    -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7,
    -8, 8, -9, 9, -10, 10, -11, 11, -12, 12, -13, 13, 14, -14, 15, -15, -16
  };

  for (unsigned idx = 0; idx < array_lengthof(SplatCsts); ++idx) {
    // Indirect through the SplatCsts array so that we favor 'vsplti -1' for
    // cases which are ambiguous (e.g. formation of 0x8000_0000).  'vsplti -1'
    int i = SplatCsts[idx];

    // Figure out what shift amount will be used by altivec if shifted by i in
    // this splat size.
    unsigned TypeShiftAmt = i & (SplatBitSize-1);

    // vsplti + shl self.
    if (SextVal == (int)((unsigned)i << TypeShiftAmt)) {
      SDValue Res = BuildSplatI(i, SplatSize, MVT::Other, DAG, dl);
      static const unsigned IIDs[] = { // Intrinsic to use for each size.
        Intrinsic::ppc_altivec_vslb, Intrinsic::ppc_altivec_vslh, 0,
        Intrinsic::ppc_altivec_vslw
      };
      Res = BuildIntrinsicOp(IIDs[SplatSize-1], Res, Res, DAG, dl);
      return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Res);
    }

    // vsplti + srl self.
    if (SextVal == (int)((unsigned)i >> TypeShiftAmt)) {
      SDValue Res = BuildSplatI(i, SplatSize, MVT::Other, DAG, dl);
      static const unsigned IIDs[] = { // Intrinsic to use for each size.
        Intrinsic::ppc_altivec_vsrb, Intrinsic::ppc_altivec_vsrh, 0,
        Intrinsic::ppc_altivec_vsrw
      };
      Res = BuildIntrinsicOp(IIDs[SplatSize-1], Res, Res, DAG, dl);
      return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Res);
    }

    // vsplti + sra self.
    if (SextVal == (int)((unsigned)i >> TypeShiftAmt)) {
      SDValue Res = BuildSplatI(i, SplatSize, MVT::Other, DAG, dl);
      static const unsigned IIDs[] = { // Intrinsic to use for each size.
        Intrinsic::ppc_altivec_vsrab, Intrinsic::ppc_altivec_vsrah, 0,
        Intrinsic::ppc_altivec_vsraw
      };
      Res = BuildIntrinsicOp(IIDs[SplatSize-1], Res, Res, DAG, dl);
      return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Res);
    }

    // vsplti + rol self.
    if (SextVal == (int)(((unsigned)i << TypeShiftAmt) |
                         ((unsigned)i >> (SplatBitSize-TypeShiftAmt)))) {
      SDValue Res = BuildSplatI(i, SplatSize, MVT::Other, DAG, dl);
      static const unsigned IIDs[] = { // Intrinsic to use for each size.
        Intrinsic::ppc_altivec_vrlb, Intrinsic::ppc_altivec_vrlh, 0,
        Intrinsic::ppc_altivec_vrlw
      };
      Res = BuildIntrinsicOp(IIDs[SplatSize-1], Res, Res, DAG, dl);
      return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Res);
    }

    // t = vsplti c, result = vsldoi t, t, 1
    if (SextVal == (int)(((unsigned)i << 8) | (i < 0 ? 0xFF : 0))) {
      SDValue T = BuildSplatI(i, SplatSize, MVT::v16i8, DAG, dl);
      return BuildVSLDOI(T, T, 1, Op.getValueType(), DAG, dl);
    }
    // t = vsplti c, result = vsldoi t, t, 2
    if (SextVal == (int)(((unsigned)i << 16) | (i < 0 ? 0xFFFF : 0))) {
      SDValue T = BuildSplatI(i, SplatSize, MVT::v16i8, DAG, dl);
      return BuildVSLDOI(T, T, 2, Op.getValueType(), DAG, dl);
    }
    // t = vsplti c, result = vsldoi t, t, 3
    if (SextVal == (int)(((unsigned)i << 24) | (i < 0 ? 0xFFFFFF : 0))) {
      SDValue T = BuildSplatI(i, SplatSize, MVT::v16i8, DAG, dl);
      return BuildVSLDOI(T, T, 3, Op.getValueType(), DAG, dl);
    }
  }

  // Three instruction sequences.

  // Odd, in range [17,31]:  (vsplti C)-(vsplti -16).
  if (SextVal >= 0 && SextVal <= 31) {
    SDValue LHS = BuildSplatI(SextVal-16, SplatSize, MVT::Other, DAG, dl);
    SDValue RHS = BuildSplatI(-16, SplatSize, MVT::Other, DAG, dl);
    LHS = DAG.getNode(ISD::SUB, dl, LHS.getValueType(), LHS, RHS);
    return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), LHS);
  }
  // Odd, in range [-31,-17]:  (vsplti C)+(vsplti -16).
  if (SextVal >= -31 && SextVal <= 0) {
    SDValue LHS = BuildSplatI(SextVal+16, SplatSize, MVT::Other, DAG, dl);
    SDValue RHS = BuildSplatI(-16, SplatSize, MVT::Other, DAG, dl);
    LHS = DAG.getNode(ISD::ADD, dl, LHS.getValueType(), LHS, RHS);
    return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), LHS);
  }

  return SDValue();
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
    OP_COPY = 0,  // Copy, used for things like <u,u,u,3> to say it is <0,1,2,3>
    OP_VMRGHW,
    OP_VMRGLW,
    OP_VSPLTISW0,
    OP_VSPLTISW1,
    OP_VSPLTISW2,
    OP_VSPLTISW3,
    OP_VSLDOI4,
    OP_VSLDOI8,
    OP_VSLDOI12
  };

  if (OpNum == OP_COPY) {
    if (LHSID == (1*9+2)*9+3) return LHS;
    assert(LHSID == ((4*9+5)*9+6)*9+7 && "Illegal OP_COPY!");
    return RHS;
  }

  SDValue OpLHS, OpRHS;
  OpLHS = GeneratePerfectShuffle(PerfectShuffleTable[LHSID], LHS, RHS, DAG, dl);
  OpRHS = GeneratePerfectShuffle(PerfectShuffleTable[RHSID], LHS, RHS, DAG, dl);

  int ShufIdxs[16];
  switch (OpNum) {
  default: llvm_unreachable("Unknown i32 permute!");
  case OP_VMRGHW:
    ShufIdxs[ 0] =  0; ShufIdxs[ 1] =  1; ShufIdxs[ 2] =  2; ShufIdxs[ 3] =  3;
    ShufIdxs[ 4] = 16; ShufIdxs[ 5] = 17; ShufIdxs[ 6] = 18; ShufIdxs[ 7] = 19;
    ShufIdxs[ 8] =  4; ShufIdxs[ 9] =  5; ShufIdxs[10] =  6; ShufIdxs[11] =  7;
    ShufIdxs[12] = 20; ShufIdxs[13] = 21; ShufIdxs[14] = 22; ShufIdxs[15] = 23;
    break;
  case OP_VMRGLW:
    ShufIdxs[ 0] =  8; ShufIdxs[ 1] =  9; ShufIdxs[ 2] = 10; ShufIdxs[ 3] = 11;
    ShufIdxs[ 4] = 24; ShufIdxs[ 5] = 25; ShufIdxs[ 6] = 26; ShufIdxs[ 7] = 27;
    ShufIdxs[ 8] = 12; ShufIdxs[ 9] = 13; ShufIdxs[10] = 14; ShufIdxs[11] = 15;
    ShufIdxs[12] = 28; ShufIdxs[13] = 29; ShufIdxs[14] = 30; ShufIdxs[15] = 31;
    break;
  case OP_VSPLTISW0:
    for (unsigned i = 0; i != 16; ++i)
      ShufIdxs[i] = (i&3)+0;
    break;
  case OP_VSPLTISW1:
    for (unsigned i = 0; i != 16; ++i)
      ShufIdxs[i] = (i&3)+4;
    break;
  case OP_VSPLTISW2:
    for (unsigned i = 0; i != 16; ++i)
      ShufIdxs[i] = (i&3)+8;
    break;
  case OP_VSPLTISW3:
    for (unsigned i = 0; i != 16; ++i)
      ShufIdxs[i] = (i&3)+12;
    break;
  case OP_VSLDOI4:
    return BuildVSLDOI(OpLHS, OpRHS, 4, OpLHS.getValueType(), DAG, dl);
  case OP_VSLDOI8:
    return BuildVSLDOI(OpLHS, OpRHS, 8, OpLHS.getValueType(), DAG, dl);
  case OP_VSLDOI12:
    return BuildVSLDOI(OpLHS, OpRHS, 12, OpLHS.getValueType(), DAG, dl);
  }
  EVT VT = OpLHS.getValueType();
  OpLHS = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, OpLHS);
  OpRHS = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, OpRHS);
  SDValue T = DAG.getVectorShuffle(MVT::v16i8, dl, OpLHS, OpRHS, ShufIdxs);
  return DAG.getNode(ISD::BITCAST, dl, VT, T);
}

/// LowerVECTOR_SHUFFLE - Return the code we lower for VECTOR_SHUFFLE.  If this
/// is a shuffle we can handle in a single instruction, return it.  Otherwise,
/// return the code it can be lowered into.  Worst case, it can always be
/// lowered into a vperm.
SDValue PPCTargetLowering::LowerVECTOR_SHUFFLE(SDValue Op,
                                               SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  EVT VT = Op.getValueType();

  // Cases that are handled by instructions that take permute immediates
  // (such as vsplt*) should be left as VECTOR_SHUFFLE nodes so they can be
  // selected by the instruction selector.
  if (V2.getOpcode() == ISD::UNDEF) {
    if (PPC::isSplatShuffleMask(SVOp, 1) ||
        PPC::isSplatShuffleMask(SVOp, 2) ||
        PPC::isSplatShuffleMask(SVOp, 4) ||
        PPC::isVPKUWUMShuffleMask(SVOp, true) ||
        PPC::isVPKUHUMShuffleMask(SVOp, true) ||
        PPC::isVSLDOIShuffleMask(SVOp, true) != -1 ||
        PPC::isVMRGLShuffleMask(SVOp, 1, true) ||
        PPC::isVMRGLShuffleMask(SVOp, 2, true) ||
        PPC::isVMRGLShuffleMask(SVOp, 4, true) ||
        PPC::isVMRGHShuffleMask(SVOp, 1, true) ||
        PPC::isVMRGHShuffleMask(SVOp, 2, true) ||
        PPC::isVMRGHShuffleMask(SVOp, 4, true)) {
      return Op;
    }
  }

  // Altivec has a variety of "shuffle immediates" that take two vector inputs
  // and produce a fixed permutation.  If any of these match, do not lower to
  // VPERM.
  if (PPC::isVPKUWUMShuffleMask(SVOp, false) ||
      PPC::isVPKUHUMShuffleMask(SVOp, false) ||
      PPC::isVSLDOIShuffleMask(SVOp, false) != -1 ||
      PPC::isVMRGLShuffleMask(SVOp, 1, false) ||
      PPC::isVMRGLShuffleMask(SVOp, 2, false) ||
      PPC::isVMRGLShuffleMask(SVOp, 4, false) ||
      PPC::isVMRGHShuffleMask(SVOp, 1, false) ||
      PPC::isVMRGHShuffleMask(SVOp, 2, false) ||
      PPC::isVMRGHShuffleMask(SVOp, 4, false))
    return Op;

  // Check to see if this is a shuffle of 4-byte values.  If so, we can use our
  // perfect shuffle table to emit an optimal matching sequence.
  ArrayRef<int> PermMask = SVOp->getMask();

  unsigned PFIndexes[4];
  bool isFourElementShuffle = true;
  for (unsigned i = 0; i != 4 && isFourElementShuffle; ++i) { // Element number
    unsigned EltNo = 8;   // Start out undef.
    for (unsigned j = 0; j != 4; ++j) {  // Intra-element byte.
      if (PermMask[i*4+j] < 0)
        continue;   // Undef, ignore it.

      unsigned ByteSource = PermMask[i*4+j];
      if ((ByteSource & 3) != j) {
        isFourElementShuffle = false;
        break;
      }

      if (EltNo == 8) {
        EltNo = ByteSource/4;
      } else if (EltNo != ByteSource/4) {
        isFourElementShuffle = false;
        break;
      }
    }
    PFIndexes[i] = EltNo;
  }

  // If this shuffle can be expressed as a shuffle of 4-byte elements, use the
  // perfect shuffle vector to determine if it is cost effective to do this as
  // discrete instructions, or whether we should use a vperm.
  if (isFourElementShuffle) {
    // Compute the index in the perfect shuffle table.
    unsigned PFTableIndex =
      PFIndexes[0]*9*9*9+PFIndexes[1]*9*9+PFIndexes[2]*9+PFIndexes[3];

    unsigned PFEntry = PerfectShuffleTable[PFTableIndex];
    unsigned Cost  = (PFEntry >> 30);

    // Determining when to avoid vperm is tricky.  Many things affect the cost
    // of vperm, particularly how many times the perm mask needs to be computed.
    // For example, if the perm mask can be hoisted out of a loop or is already
    // used (perhaps because there are multiple permutes with the same shuffle
    // mask?) the vperm has a cost of 1.  OTOH, hoisting the permute mask out of
    // the loop requires an extra register.
    //
    // As a compromise, we only emit discrete instructions if the shuffle can be
    // generated in 3 or fewer operations.  When we have loop information
    // available, if this block is within a loop, we should avoid using vperm
    // for 3-operation perms and use a constant pool load instead.
    if (Cost < 3)
      return GeneratePerfectShuffle(PFEntry, V1, V2, DAG, dl);
  }

  // Lower this to a VPERM(V1, V2, V3) expression, where V3 is a constant
  // vector that will get spilled to the constant pool.
  if (V2.getOpcode() == ISD::UNDEF) V2 = V1;

  // The SHUFFLE_VECTOR mask is almost exactly what we want for vperm, except
  // that it is in input element units, not in bytes.  Convert now.
  EVT EltVT = V1.getValueType().getVectorElementType();
  unsigned BytesPerElement = EltVT.getSizeInBits()/8;

  SmallVector<SDValue, 16> ResultMask;
  for (unsigned i = 0, e = VT.getVectorNumElements(); i != e; ++i) {
    unsigned SrcElt = PermMask[i] < 0 ? 0 : PermMask[i];

    for (unsigned j = 0; j != BytesPerElement; ++j)
      ResultMask.push_back(DAG.getConstant(SrcElt*BytesPerElement+j,
                                           MVT::i32));
  }

  SDValue VPermMask = DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v16i8,
                                    &ResultMask[0], ResultMask.size());
  return DAG.getNode(PPCISD::VPERM, dl, V1.getValueType(), V1, V2, VPermMask);
}

/// getAltivecCompareInfo - Given an intrinsic, return false if it is not an
/// altivec comparison.  If it is, return true and fill in Opc/isDot with
/// information about the intrinsic.
static bool getAltivecCompareInfo(SDValue Intrin, int &CompareOpc,
                                  bool &isDot) {
  unsigned IntrinsicID =
    cast<ConstantSDNode>(Intrin.getOperand(0))->getZExtValue();
  CompareOpc = -1;
  isDot = false;
  switch (IntrinsicID) {
  default: return false;
    // Comparison predicates.
  case Intrinsic::ppc_altivec_vcmpbfp_p:  CompareOpc = 966; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpeqfp_p: CompareOpc = 198; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpequb_p: CompareOpc =   6; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpequh_p: CompareOpc =  70; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpequw_p: CompareOpc = 134; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgefp_p: CompareOpc = 454; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtfp_p: CompareOpc = 710; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtsb_p: CompareOpc = 774; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtsh_p: CompareOpc = 838; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtsw_p: CompareOpc = 902; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtub_p: CompareOpc = 518; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtuh_p: CompareOpc = 582; isDot = 1; break;
  case Intrinsic::ppc_altivec_vcmpgtuw_p: CompareOpc = 646; isDot = 1; break;

    // Normal Comparisons.
  case Intrinsic::ppc_altivec_vcmpbfp:    CompareOpc = 966; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpeqfp:   CompareOpc = 198; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpequb:   CompareOpc =   6; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpequh:   CompareOpc =  70; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpequw:   CompareOpc = 134; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgefp:   CompareOpc = 454; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtfp:   CompareOpc = 710; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtsb:   CompareOpc = 774; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtsh:   CompareOpc = 838; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtsw:   CompareOpc = 902; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtub:   CompareOpc = 518; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtuh:   CompareOpc = 582; isDot = 0; break;
  case Intrinsic::ppc_altivec_vcmpgtuw:   CompareOpc = 646; isDot = 0; break;
  }
  return true;
}

/// LowerINTRINSIC_WO_CHAIN - If this is an intrinsic that we want to custom
/// lower, do it, otherwise return null.
SDValue PPCTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                   SelectionDAG &DAG) const {
  // If this is a lowered altivec predicate compare, CompareOpc is set to the
  // opcode number of the comparison.
  DebugLoc dl = Op.getDebugLoc();
  int CompareOpc;
  bool isDot;
  if (!getAltivecCompareInfo(Op, CompareOpc, isDot))
    return SDValue();    // Don't custom lower most intrinsics.

  // If this is a non-dot comparison, make the VCMP node and we are done.
  if (!isDot) {
    SDValue Tmp = DAG.getNode(PPCISD::VCMP, dl, Op.getOperand(2).getValueType(),
                              Op.getOperand(1), Op.getOperand(2),
                              DAG.getConstant(CompareOpc, MVT::i32));
    return DAG.getNode(ISD::BITCAST, dl, Op.getValueType(), Tmp);
  }

  // Create the PPCISD altivec 'dot' comparison node.
  SDValue Ops[] = {
    Op.getOperand(2),  // LHS
    Op.getOperand(3),  // RHS
    DAG.getConstant(CompareOpc, MVT::i32)
  };
  std::vector<EVT> VTs;
  VTs.push_back(Op.getOperand(2).getValueType());
  VTs.push_back(MVT::Glue);
  SDValue CompNode = DAG.getNode(PPCISD::VCMPo, dl, VTs, Ops, 3);

  // Now that we have the comparison, emit a copy from the CR to a GPR.
  // This is flagged to the above dot comparison.
  SDValue Flags = DAG.getNode(PPCISD::MFCR, dl, MVT::i32,
                                DAG.getRegister(PPC::CR6, MVT::i32),
                                CompNode.getValue(1));

  // Unpack the result based on how the target uses it.
  unsigned BitNo;   // Bit # of CR6.
  bool InvertBit;   // Invert result?
  switch (cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue()) {
  default:  // Can't happen, don't crash on invalid number though.
  case 0:   // Return the value of the EQ bit of CR6.
    BitNo = 0; InvertBit = false;
    break;
  case 1:   // Return the inverted value of the EQ bit of CR6.
    BitNo = 0; InvertBit = true;
    break;
  case 2:   // Return the value of the LT bit of CR6.
    BitNo = 2; InvertBit = false;
    break;
  case 3:   // Return the inverted value of the LT bit of CR6.
    BitNo = 2; InvertBit = true;
    break;
  }

  // Shift the bit into the low position.
  Flags = DAG.getNode(ISD::SRL, dl, MVT::i32, Flags,
                      DAG.getConstant(8-(3-BitNo), MVT::i32));
  // Isolate the bit.
  Flags = DAG.getNode(ISD::AND, dl, MVT::i32, Flags,
                      DAG.getConstant(1, MVT::i32));

  // If we are supposed to, toggle the bit.
  if (InvertBit)
    Flags = DAG.getNode(ISD::XOR, dl, MVT::i32, Flags,
                        DAG.getConstant(1, MVT::i32));
  return Flags;
}

SDValue PPCTargetLowering::LowerSCALAR_TO_VECTOR(SDValue Op,
                                                   SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  // Create a stack slot that is 16-byte aligned.
  MachineFrameInfo *FrameInfo = DAG.getMachineFunction().getFrameInfo();
  int FrameIdx = FrameInfo->CreateStackObject(16, 16, false);
  EVT PtrVT = getPointerTy();
  SDValue FIdx = DAG.getFrameIndex(FrameIdx, PtrVT);

  // Store the input value into Value#0 of the stack slot.
  SDValue Store = DAG.getStore(DAG.getEntryNode(), dl,
                               Op.getOperand(0), FIdx, MachinePointerInfo(),
                               false, false, 0);
  // Load it out.
  return DAG.getLoad(Op.getValueType(), dl, Store, FIdx, MachinePointerInfo(),
                     false, false, false, 0);
}

SDValue PPCTargetLowering::LowerMUL(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  if (Op.getValueType() == MVT::v4i32) {
    SDValue LHS = Op.getOperand(0), RHS = Op.getOperand(1);

    SDValue Zero  = BuildSplatI(  0, 1, MVT::v4i32, DAG, dl);
    SDValue Neg16 = BuildSplatI(-16, 4, MVT::v4i32, DAG, dl);//+16 as shift amt.

    SDValue RHSSwap =   // = vrlw RHS, 16
      BuildIntrinsicOp(Intrinsic::ppc_altivec_vrlw, RHS, Neg16, DAG, dl);

    // Shrinkify inputs to v8i16.
    LHS = DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, LHS);
    RHS = DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, RHS);
    RHSSwap = DAG.getNode(ISD::BITCAST, dl, MVT::v8i16, RHSSwap);

    // Low parts multiplied together, generating 32-bit results (we ignore the
    // top parts).
    SDValue LoProd = BuildIntrinsicOp(Intrinsic::ppc_altivec_vmulouh,
                                        LHS, RHS, DAG, dl, MVT::v4i32);

    SDValue HiProd = BuildIntrinsicOp(Intrinsic::ppc_altivec_vmsumuhm,
                                      LHS, RHSSwap, Zero, DAG, dl, MVT::v4i32);
    // Shift the high parts up 16 bits.
    HiProd = BuildIntrinsicOp(Intrinsic::ppc_altivec_vslw, HiProd,
                              Neg16, DAG, dl);
    return DAG.getNode(ISD::ADD, dl, MVT::v4i32, LoProd, HiProd);
  } else if (Op.getValueType() == MVT::v8i16) {
    SDValue LHS = Op.getOperand(0), RHS = Op.getOperand(1);

    SDValue Zero = BuildSplatI(0, 1, MVT::v8i16, DAG, dl);

    return BuildIntrinsicOp(Intrinsic::ppc_altivec_vmladduhm,
                            LHS, RHS, Zero, DAG, dl);
  } else if (Op.getValueType() == MVT::v16i8) {
    SDValue LHS = Op.getOperand(0), RHS = Op.getOperand(1);

    // Multiply the even 8-bit parts, producing 16-bit sums.
    SDValue EvenParts = BuildIntrinsicOp(Intrinsic::ppc_altivec_vmuleub,
                                           LHS, RHS, DAG, dl, MVT::v8i16);
    EvenParts = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, EvenParts);

    // Multiply the odd 8-bit parts, producing 16-bit sums.
    SDValue OddParts = BuildIntrinsicOp(Intrinsic::ppc_altivec_vmuloub,
                                          LHS, RHS, DAG, dl, MVT::v8i16);
    OddParts = DAG.getNode(ISD::BITCAST, dl, MVT::v16i8, OddParts);

    // Merge the results together.
    int Ops[16];
    for (unsigned i = 0; i != 8; ++i) {
      Ops[i*2  ] = 2*i+1;
      Ops[i*2+1] = 2*i+1+16;
    }
    return DAG.getVectorShuffle(MVT::v16i8, dl, EvenParts, OddParts, Ops);
  } else {
    llvm_unreachable("Unknown mul to lower!");
  }
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDValue PPCTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Wasn't expecting to be able to lower this!");
  case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
  case ISD::BlockAddress:       return LowerBlockAddress(Op, DAG);
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
  case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
  case ISD::SETCC:              return LowerSETCC(Op, DAG);
  case ISD::INIT_TRAMPOLINE:    return LowerINIT_TRAMPOLINE(Op, DAG);
  case ISD::ADJUST_TRAMPOLINE:  return LowerADJUST_TRAMPOLINE(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG, PPCSubTarget);

  case ISD::VAARG:
    return LowerVAARG(Op, DAG, PPCSubTarget);

  case ISD::STACKRESTORE:       return LowerSTACKRESTORE(Op, DAG, PPCSubTarget);
  case ISD::DYNAMIC_STACKALLOC:
    return LowerDYNAMIC_STACKALLOC(Op, DAG, PPCSubTarget);

  case ISD::SELECT_CC:          return LowerSELECT_CC(Op, DAG);
  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT:         return LowerFP_TO_INT(Op, DAG,
                                                       Op.getDebugLoc());
  case ISD::SINT_TO_FP:         return LowerSINT_TO_FP(Op, DAG);
  case ISD::FLT_ROUNDS_:        return LowerFLT_ROUNDS_(Op, DAG);

  // Lower 64-bit shifts.
  case ISD::SHL_PARTS:          return LowerSHL_PARTS(Op, DAG);
  case ISD::SRL_PARTS:          return LowerSRL_PARTS(Op, DAG);
  case ISD::SRA_PARTS:          return LowerSRA_PARTS(Op, DAG);

  // Vector-related lowering.
  case ISD::BUILD_VECTOR:       return LowerBUILD_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:     return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::SCALAR_TO_VECTOR:   return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::MUL:                return LowerMUL(Op, DAG);

  // Frame & Return address.
  case ISD::RETURNADDR:         return LowerRETURNADDR(Op, DAG);
  case ISD::FRAMEADDR:          return LowerFRAMEADDR(Op, DAG);
  }
}

void PPCTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue>&Results,
                                           SelectionDAG &DAG) const {
  const TargetMachine &TM = getTargetMachine();
  DebugLoc dl = N->getDebugLoc();
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Do not know how to custom type legalize this operation!");
  case ISD::VAARG: {
    if (!TM.getSubtarget<PPCSubtarget>().isSVR4ABI()
        || TM.getSubtarget<PPCSubtarget>().isPPC64())
      return;

    EVT VT = N->getValueType(0);

    if (VT == MVT::i64) {
      SDValue NewNode = LowerVAARG(SDValue(N, 1), DAG, PPCSubTarget);

      Results.push_back(NewNode);
      Results.push_back(NewNode.getValue(1));
    }
    return;
  }
  case ISD::FP_ROUND_INREG: {
    assert(N->getValueType(0) == MVT::ppcf128);
    assert(N->getOperand(0).getValueType() == MVT::ppcf128);
    SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, dl,
                             MVT::f64, N->getOperand(0),
                             DAG.getIntPtrConstant(0));
    SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, dl,
                             MVT::f64, N->getOperand(0),
                             DAG.getIntPtrConstant(1));

    // This sequence changes FPSCR to do round-to-zero, adds the two halves
    // of the long double, and puts FPSCR back the way it was.  We do not
    // actually model FPSCR.
    std::vector<EVT> NodeTys;
    SDValue Ops[4], Result, MFFSreg, InFlag, FPreg;

    NodeTys.push_back(MVT::f64);   // Return register
    NodeTys.push_back(MVT::Glue);    // Returns a flag for later insns
    Result = DAG.getNode(PPCISD::MFFS, dl, NodeTys, &InFlag, 0);
    MFFSreg = Result.getValue(0);
    InFlag = Result.getValue(1);

    NodeTys.clear();
    NodeTys.push_back(MVT::Glue);   // Returns a flag
    Ops[0] = DAG.getConstant(31, MVT::i32);
    Ops[1] = InFlag;
    Result = DAG.getNode(PPCISD::MTFSB1, dl, NodeTys, Ops, 2);
    InFlag = Result.getValue(0);

    NodeTys.clear();
    NodeTys.push_back(MVT::Glue);   // Returns a flag
    Ops[0] = DAG.getConstant(30, MVT::i32);
    Ops[1] = InFlag;
    Result = DAG.getNode(PPCISD::MTFSB0, dl, NodeTys, Ops, 2);
    InFlag = Result.getValue(0);

    NodeTys.clear();
    NodeTys.push_back(MVT::f64);    // result of add
    NodeTys.push_back(MVT::Glue);   // Returns a flag
    Ops[0] = Lo;
    Ops[1] = Hi;
    Ops[2] = InFlag;
    Result = DAG.getNode(PPCISD::FADDRTZ, dl, NodeTys, Ops, 3);
    FPreg = Result.getValue(0);
    InFlag = Result.getValue(1);

    NodeTys.clear();
    NodeTys.push_back(MVT::f64);
    Ops[0] = DAG.getConstant(1, MVT::i32);
    Ops[1] = MFFSreg;
    Ops[2] = FPreg;
    Ops[3] = InFlag;
    Result = DAG.getNode(PPCISD::MTFSF, dl, NodeTys, Ops, 4);
    FPreg = Result.getValue(0);

    // We know the low half is about to be thrown away, so just use something
    // convenient.
    Results.push_back(DAG.getNode(ISD::BUILD_PAIR, dl, MVT::ppcf128,
                                FPreg, FPreg));
    return;
  }
  case ISD::FP_TO_SINT:
    Results.push_back(LowerFP_TO_INT(SDValue(N, 0), DAG, dl));
    return;
  }
}


//===----------------------------------------------------------------------===//
//  Other Lowering Code
//===----------------------------------------------------------------------===//

MachineBasicBlock *
PPCTargetLowering::EmitAtomicBinary(MachineInstr *MI, MachineBasicBlock *BB,
                                    bool is64bit, unsigned BinOpcode) const {
  // This also handles ATOMIC_SWAP, indicated by BinOpcode==0.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction *F = BB->getParent();
  MachineFunction::iterator It = BB;
  ++It;

  unsigned dest = MI->getOperand(0).getReg();
  unsigned ptrA = MI->getOperand(1).getReg();
  unsigned ptrB = MI->getOperand(2).getReg();
  unsigned incr = MI->getOperand(3).getReg();
  DebugLoc dl = MI->getDebugLoc();

  MachineBasicBlock *loopMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(It, loopMBB);
  F->insert(It, exitMBB);
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  MachineRegisterInfo &RegInfo = F->getRegInfo();
  unsigned TmpReg = (!BinOpcode) ? incr :
    RegInfo.createVirtualRegister(
       is64bit ? (const TargetRegisterClass *) &PPC::G8RCRegClass :
                 (const TargetRegisterClass *) &PPC::GPRCRegClass);

  //  thisMBB:
  //   ...
  //   fallthrough --> loopMBB
  BB->addSuccessor(loopMBB);

  //  loopMBB:
  //   l[wd]arx dest, ptr
  //   add r0, dest, incr
  //   st[wd]cx. r0, ptr
  //   bne- loopMBB
  //   fallthrough --> exitMBB
  BB = loopMBB;
  BuildMI(BB, dl, TII->get(is64bit ? PPC::LDARX : PPC::LWARX), dest)
    .addReg(ptrA).addReg(ptrB);
  if (BinOpcode)
    BuildMI(BB, dl, TII->get(BinOpcode), TmpReg).addReg(incr).addReg(dest);
  BuildMI(BB, dl, TII->get(is64bit ? PPC::STDCX : PPC::STWCX))
    .addReg(TmpReg).addReg(ptrA).addReg(ptrB);
  BuildMI(BB, dl, TII->get(PPC::BCC))
    .addImm(PPC::PRED_NE).addReg(PPC::CR0).addMBB(loopMBB);
  BB->addSuccessor(loopMBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;
  return BB;
}

MachineBasicBlock *
PPCTargetLowering::EmitPartwordAtomicBinary(MachineInstr *MI,
                                            MachineBasicBlock *BB,
                                            bool is8bit,    // operation
                                            unsigned BinOpcode) const {
  // This also handles ATOMIC_SWAP, indicated by BinOpcode==0.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  // In 64 bit mode we have to use 64 bits for addresses, even though the
  // lwarx/stwcx are 32 bits.  With the 32-bit atomics we can use address
  // registers without caring whether they're 32 or 64, but here we're
  // doing actual arithmetic on the addresses.
  bool is64bit = PPCSubTarget.isPPC64();
  unsigned ZeroReg = is64bit ? PPC::X0 : PPC::R0;

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction *F = BB->getParent();
  MachineFunction::iterator It = BB;
  ++It;

  unsigned dest = MI->getOperand(0).getReg();
  unsigned ptrA = MI->getOperand(1).getReg();
  unsigned ptrB = MI->getOperand(2).getReg();
  unsigned incr = MI->getOperand(3).getReg();
  DebugLoc dl = MI->getDebugLoc();

  MachineBasicBlock *loopMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(It, loopMBB);
  F->insert(It, exitMBB);
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  MachineRegisterInfo &RegInfo = F->getRegInfo();
  const TargetRegisterClass *RC =
    is64bit ? (const TargetRegisterClass *) &PPC::G8RCRegClass :
              (const TargetRegisterClass *) &PPC::GPRCRegClass;
  unsigned PtrReg = RegInfo.createVirtualRegister(RC);
  unsigned Shift1Reg = RegInfo.createVirtualRegister(RC);
  unsigned ShiftReg = RegInfo.createVirtualRegister(RC);
  unsigned Incr2Reg = RegInfo.createVirtualRegister(RC);
  unsigned MaskReg = RegInfo.createVirtualRegister(RC);
  unsigned Mask2Reg = RegInfo.createVirtualRegister(RC);
  unsigned Mask3Reg = RegInfo.createVirtualRegister(RC);
  unsigned Tmp2Reg = RegInfo.createVirtualRegister(RC);
  unsigned Tmp3Reg = RegInfo.createVirtualRegister(RC);
  unsigned Tmp4Reg = RegInfo.createVirtualRegister(RC);
  unsigned TmpDestReg = RegInfo.createVirtualRegister(RC);
  unsigned Ptr1Reg;
  unsigned TmpReg = (!BinOpcode) ? Incr2Reg : RegInfo.createVirtualRegister(RC);

  //  thisMBB:
  //   ...
  //   fallthrough --> loopMBB
  BB->addSuccessor(loopMBB);

  // The 4-byte load must be aligned, while a char or short may be
  // anywhere in the word.  Hence all this nasty bookkeeping code.
  //   add ptr1, ptrA, ptrB [copy if ptrA==0]
  //   rlwinm shift1, ptr1, 3, 27, 28 [3, 27, 27]
  //   xori shift, shift1, 24 [16]
  //   rlwinm ptr, ptr1, 0, 0, 29
  //   slw incr2, incr, shift
  //   li mask2, 255 [li mask3, 0; ori mask2, mask3, 65535]
  //   slw mask, mask2, shift
  //  loopMBB:
  //   lwarx tmpDest, ptr
  //   add tmp, tmpDest, incr2
  //   andc tmp2, tmpDest, mask
  //   and tmp3, tmp, mask
  //   or tmp4, tmp3, tmp2
  //   stwcx. tmp4, ptr
  //   bne- loopMBB
  //   fallthrough --> exitMBB
  //   srw dest, tmpDest, shift
  if (ptrA != ZeroReg) {
    Ptr1Reg = RegInfo.createVirtualRegister(RC);
    BuildMI(BB, dl, TII->get(is64bit ? PPC::ADD8 : PPC::ADD4), Ptr1Reg)
      .addReg(ptrA).addReg(ptrB);
  } else {
    Ptr1Reg = ptrB;
  }
  BuildMI(BB, dl, TII->get(PPC::RLWINM), Shift1Reg).addReg(Ptr1Reg)
      .addImm(3).addImm(27).addImm(is8bit ? 28 : 27);
  BuildMI(BB, dl, TII->get(is64bit ? PPC::XORI8 : PPC::XORI), ShiftReg)
      .addReg(Shift1Reg).addImm(is8bit ? 24 : 16);
  if (is64bit)
    BuildMI(BB, dl, TII->get(PPC::RLDICR), PtrReg)
      .addReg(Ptr1Reg).addImm(0).addImm(61);
  else
    BuildMI(BB, dl, TII->get(PPC::RLWINM), PtrReg)
      .addReg(Ptr1Reg).addImm(0).addImm(0).addImm(29);
  BuildMI(BB, dl, TII->get(PPC::SLW), Incr2Reg)
      .addReg(incr).addReg(ShiftReg);
  if (is8bit)
    BuildMI(BB, dl, TII->get(PPC::LI), Mask2Reg).addImm(255);
  else {
    BuildMI(BB, dl, TII->get(PPC::LI), Mask3Reg).addImm(0);
    BuildMI(BB, dl, TII->get(PPC::ORI),Mask2Reg).addReg(Mask3Reg).addImm(65535);
  }
  BuildMI(BB, dl, TII->get(PPC::SLW), MaskReg)
      .addReg(Mask2Reg).addReg(ShiftReg);

  BB = loopMBB;
  BuildMI(BB, dl, TII->get(PPC::LWARX), TmpDestReg)
    .addReg(ZeroReg).addReg(PtrReg);
  if (BinOpcode)
    BuildMI(BB, dl, TII->get(BinOpcode), TmpReg)
      .addReg(Incr2Reg).addReg(TmpDestReg);
  BuildMI(BB, dl, TII->get(is64bit ? PPC::ANDC8 : PPC::ANDC), Tmp2Reg)
    .addReg(TmpDestReg).addReg(MaskReg);
  BuildMI(BB, dl, TII->get(is64bit ? PPC::AND8 : PPC::AND), Tmp3Reg)
    .addReg(TmpReg).addReg(MaskReg);
  BuildMI(BB, dl, TII->get(is64bit ? PPC::OR8 : PPC::OR), Tmp4Reg)
    .addReg(Tmp3Reg).addReg(Tmp2Reg);
  BuildMI(BB, dl, TII->get(is64bit ? PPC::STDCX : PPC::STWCX))
    .addReg(Tmp4Reg).addReg(ZeroReg).addReg(PtrReg);
  BuildMI(BB, dl, TII->get(PPC::BCC))
    .addImm(PPC::PRED_NE).addReg(PPC::CR0).addMBB(loopMBB);
  BB->addSuccessor(loopMBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;
  BuildMI(*BB, BB->begin(), dl, TII->get(PPC::SRW), dest).addReg(TmpDestReg)
    .addReg(ShiftReg);
  return BB;
}

MachineBasicBlock *
PPCTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                               MachineBasicBlock *BB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  // To "insert" these instructions we actually have to insert their
  // control-flow patterns.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = BB;
  ++It;

  MachineFunction *F = BB->getParent();

  if (PPCSubTarget.hasISEL() && (MI->getOpcode() == PPC::SELECT_CC_I4 ||
                                 MI->getOpcode() == PPC::SELECT_CC_I8)) {
    unsigned OpCode = MI->getOpcode() == PPC::SELECT_CC_I8 ?
                                         PPC::ISEL8 : PPC::ISEL;
    unsigned SelectPred = MI->getOperand(4).getImm();
    DebugLoc dl = MI->getDebugLoc();

    // The SelectPred is ((BI << 5) | BO) for a BCC
    unsigned BO = SelectPred & 0xF;
    assert((BO == 12 || BO == 4) && "invalid predicate BO field for isel");

    unsigned TrueOpNo, FalseOpNo;
    if (BO == 12) {
      TrueOpNo = 2;
      FalseOpNo = 3;
    } else {
      TrueOpNo = 3;
      FalseOpNo = 2;
      SelectPred = PPC::InvertPredicate((PPC::Predicate)SelectPred);
    }

    BuildMI(*BB, MI, dl, TII->get(OpCode), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(TrueOpNo).getReg())
      .addReg(MI->getOperand(FalseOpNo).getReg())
      .addImm(SelectPred).addReg(MI->getOperand(1).getReg());
  } else if (MI->getOpcode() == PPC::SELECT_CC_I4 ||
             MI->getOpcode() == PPC::SELECT_CC_I8 ||
             MI->getOpcode() == PPC::SELECT_CC_F4 ||
             MI->getOpcode() == PPC::SELECT_CC_F8 ||
             MI->getOpcode() == PPC::SELECT_CC_VRRC) {


    // The incoming instruction knows the destination vreg to set, the
    // condition code register to branch on, the true/false values to
    // select between, and a branch opcode to use.

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY ccX, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *thisMBB = BB;
    MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB = F->CreateMachineBasicBlock(LLVM_BB);
    unsigned SelectPred = MI->getOperand(4).getImm();
    DebugLoc dl = MI->getDebugLoc();
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

    BuildMI(BB, dl, TII->get(PPC::BCC))
      .addImm(SelectPred).addReg(MI->getOperand(1).getReg()).addMBB(sinkMBB);

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
            TII->get(PPC::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(3).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);
  }
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_ADD_I8)
    BB = EmitPartwordAtomicBinary(MI, BB, true, PPC::ADD4);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_ADD_I16)
    BB = EmitPartwordAtomicBinary(MI, BB, false, PPC::ADD4);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_ADD_I32)
    BB = EmitAtomicBinary(MI, BB, false, PPC::ADD4);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_ADD_I64)
    BB = EmitAtomicBinary(MI, BB, true, PPC::ADD8);

  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_AND_I8)
    BB = EmitPartwordAtomicBinary(MI, BB, true, PPC::AND);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_AND_I16)
    BB = EmitPartwordAtomicBinary(MI, BB, false, PPC::AND);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_AND_I32)
    BB = EmitAtomicBinary(MI, BB, false, PPC::AND);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_AND_I64)
    BB = EmitAtomicBinary(MI, BB, true, PPC::AND8);

  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_OR_I8)
    BB = EmitPartwordAtomicBinary(MI, BB, true, PPC::OR);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_OR_I16)
    BB = EmitPartwordAtomicBinary(MI, BB, false, PPC::OR);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_OR_I32)
    BB = EmitAtomicBinary(MI, BB, false, PPC::OR);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_OR_I64)
    BB = EmitAtomicBinary(MI, BB, true, PPC::OR8);

  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_XOR_I8)
    BB = EmitPartwordAtomicBinary(MI, BB, true, PPC::XOR);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_XOR_I16)
    BB = EmitPartwordAtomicBinary(MI, BB, false, PPC::XOR);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_XOR_I32)
    BB = EmitAtomicBinary(MI, BB, false, PPC::XOR);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_XOR_I64)
    BB = EmitAtomicBinary(MI, BB, true, PPC::XOR8);

  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_NAND_I8)
    BB = EmitPartwordAtomicBinary(MI, BB, true, PPC::ANDC);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_NAND_I16)
    BB = EmitPartwordAtomicBinary(MI, BB, false, PPC::ANDC);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_NAND_I32)
    BB = EmitAtomicBinary(MI, BB, false, PPC::ANDC);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_NAND_I64)
    BB = EmitAtomicBinary(MI, BB, true, PPC::ANDC8);

  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_SUB_I8)
    BB = EmitPartwordAtomicBinary(MI, BB, true, PPC::SUBF);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_SUB_I16)
    BB = EmitPartwordAtomicBinary(MI, BB, false, PPC::SUBF);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_SUB_I32)
    BB = EmitAtomicBinary(MI, BB, false, PPC::SUBF);
  else if (MI->getOpcode() == PPC::ATOMIC_LOAD_SUB_I64)
    BB = EmitAtomicBinary(MI, BB, true, PPC::SUBF8);

  else if (MI->getOpcode() == PPC::ATOMIC_SWAP_I8)
    BB = EmitPartwordAtomicBinary(MI, BB, true, 0);
  else if (MI->getOpcode() == PPC::ATOMIC_SWAP_I16)
    BB = EmitPartwordAtomicBinary(MI, BB, false, 0);
  else if (MI->getOpcode() == PPC::ATOMIC_SWAP_I32)
    BB = EmitAtomicBinary(MI, BB, false, 0);
  else if (MI->getOpcode() == PPC::ATOMIC_SWAP_I64)
    BB = EmitAtomicBinary(MI, BB, true, 0);

  else if (MI->getOpcode() == PPC::ATOMIC_CMP_SWAP_I32 ||
           MI->getOpcode() == PPC::ATOMIC_CMP_SWAP_I64) {
    bool is64bit = MI->getOpcode() == PPC::ATOMIC_CMP_SWAP_I64;

    unsigned dest   = MI->getOperand(0).getReg();
    unsigned ptrA   = MI->getOperand(1).getReg();
    unsigned ptrB   = MI->getOperand(2).getReg();
    unsigned oldval = MI->getOperand(3).getReg();
    unsigned newval = MI->getOperand(4).getReg();
    DebugLoc dl     = MI->getDebugLoc();

    MachineBasicBlock *loop1MBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *loop2MBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *midMBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *exitMBB = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, loop1MBB);
    F->insert(It, loop2MBB);
    F->insert(It, midMBB);
    F->insert(It, exitMBB);
    exitMBB->splice(exitMBB->begin(), BB,
                    llvm::next(MachineBasicBlock::iterator(MI)),
                    BB->end());
    exitMBB->transferSuccessorsAndUpdatePHIs(BB);

    //  thisMBB:
    //   ...
    //   fallthrough --> loopMBB
    BB->addSuccessor(loop1MBB);

    // loop1MBB:
    //   l[wd]arx dest, ptr
    //   cmp[wd] dest, oldval
    //   bne- midMBB
    // loop2MBB:
    //   st[wd]cx. newval, ptr
    //   bne- loopMBB
    //   b exitBB
    // midMBB:
    //   st[wd]cx. dest, ptr
    // exitBB:
    BB = loop1MBB;
    BuildMI(BB, dl, TII->get(is64bit ? PPC::LDARX : PPC::LWARX), dest)
      .addReg(ptrA).addReg(ptrB);
    BuildMI(BB, dl, TII->get(is64bit ? PPC::CMPD : PPC::CMPW), PPC::CR0)
      .addReg(oldval).addReg(dest);
    BuildMI(BB, dl, TII->get(PPC::BCC))
      .addImm(PPC::PRED_NE).addReg(PPC::CR0).addMBB(midMBB);
    BB->addSuccessor(loop2MBB);
    BB->addSuccessor(midMBB);

    BB = loop2MBB;
    BuildMI(BB, dl, TII->get(is64bit ? PPC::STDCX : PPC::STWCX))
      .addReg(newval).addReg(ptrA).addReg(ptrB);
    BuildMI(BB, dl, TII->get(PPC::BCC))
      .addImm(PPC::PRED_NE).addReg(PPC::CR0).addMBB(loop1MBB);
    BuildMI(BB, dl, TII->get(PPC::B)).addMBB(exitMBB);
    BB->addSuccessor(loop1MBB);
    BB->addSuccessor(exitMBB);

    BB = midMBB;
    BuildMI(BB, dl, TII->get(is64bit ? PPC::STDCX : PPC::STWCX))
      .addReg(dest).addReg(ptrA).addReg(ptrB);
    BB->addSuccessor(exitMBB);

    //  exitMBB:
    //   ...
    BB = exitMBB;
  } else if (MI->getOpcode() == PPC::ATOMIC_CMP_SWAP_I8 ||
             MI->getOpcode() == PPC::ATOMIC_CMP_SWAP_I16) {
    // We must use 64-bit registers for addresses when targeting 64-bit,
    // since we're actually doing arithmetic on them.  Other registers
    // can be 32-bit.
    bool is64bit = PPCSubTarget.isPPC64();
    bool is8bit = MI->getOpcode() == PPC::ATOMIC_CMP_SWAP_I8;

    unsigned dest   = MI->getOperand(0).getReg();
    unsigned ptrA   = MI->getOperand(1).getReg();
    unsigned ptrB   = MI->getOperand(2).getReg();
    unsigned oldval = MI->getOperand(3).getReg();
    unsigned newval = MI->getOperand(4).getReg();
    DebugLoc dl     = MI->getDebugLoc();

    MachineBasicBlock *loop1MBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *loop2MBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *midMBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *exitMBB = F->CreateMachineBasicBlock(LLVM_BB);
    F->insert(It, loop1MBB);
    F->insert(It, loop2MBB);
    F->insert(It, midMBB);
    F->insert(It, exitMBB);
    exitMBB->splice(exitMBB->begin(), BB,
                    llvm::next(MachineBasicBlock::iterator(MI)),
                    BB->end());
    exitMBB->transferSuccessorsAndUpdatePHIs(BB);

    MachineRegisterInfo &RegInfo = F->getRegInfo();
    const TargetRegisterClass *RC =
      is64bit ? (const TargetRegisterClass *) &PPC::G8RCRegClass :
                (const TargetRegisterClass *) &PPC::GPRCRegClass;
    unsigned PtrReg = RegInfo.createVirtualRegister(RC);
    unsigned Shift1Reg = RegInfo.createVirtualRegister(RC);
    unsigned ShiftReg = RegInfo.createVirtualRegister(RC);
    unsigned NewVal2Reg = RegInfo.createVirtualRegister(RC);
    unsigned NewVal3Reg = RegInfo.createVirtualRegister(RC);
    unsigned OldVal2Reg = RegInfo.createVirtualRegister(RC);
    unsigned OldVal3Reg = RegInfo.createVirtualRegister(RC);
    unsigned MaskReg = RegInfo.createVirtualRegister(RC);
    unsigned Mask2Reg = RegInfo.createVirtualRegister(RC);
    unsigned Mask3Reg = RegInfo.createVirtualRegister(RC);
    unsigned Tmp2Reg = RegInfo.createVirtualRegister(RC);
    unsigned Tmp4Reg = RegInfo.createVirtualRegister(RC);
    unsigned TmpDestReg = RegInfo.createVirtualRegister(RC);
    unsigned Ptr1Reg;
    unsigned TmpReg = RegInfo.createVirtualRegister(RC);
    unsigned ZeroReg = is64bit ? PPC::X0 : PPC::R0;
    //  thisMBB:
    //   ...
    //   fallthrough --> loopMBB
    BB->addSuccessor(loop1MBB);

    // The 4-byte load must be aligned, while a char or short may be
    // anywhere in the word.  Hence all this nasty bookkeeping code.
    //   add ptr1, ptrA, ptrB [copy if ptrA==0]
    //   rlwinm shift1, ptr1, 3, 27, 28 [3, 27, 27]
    //   xori shift, shift1, 24 [16]
    //   rlwinm ptr, ptr1, 0, 0, 29
    //   slw newval2, newval, shift
    //   slw oldval2, oldval,shift
    //   li mask2, 255 [li mask3, 0; ori mask2, mask3, 65535]
    //   slw mask, mask2, shift
    //   and newval3, newval2, mask
    //   and oldval3, oldval2, mask
    // loop1MBB:
    //   lwarx tmpDest, ptr
    //   and tmp, tmpDest, mask
    //   cmpw tmp, oldval3
    //   bne- midMBB
    // loop2MBB:
    //   andc tmp2, tmpDest, mask
    //   or tmp4, tmp2, newval3
    //   stwcx. tmp4, ptr
    //   bne- loop1MBB
    //   b exitBB
    // midMBB:
    //   stwcx. tmpDest, ptr
    // exitBB:
    //   srw dest, tmpDest, shift
    if (ptrA != ZeroReg) {
      Ptr1Reg = RegInfo.createVirtualRegister(RC);
      BuildMI(BB, dl, TII->get(is64bit ? PPC::ADD8 : PPC::ADD4), Ptr1Reg)
        .addReg(ptrA).addReg(ptrB);
    } else {
      Ptr1Reg = ptrB;
    }
    BuildMI(BB, dl, TII->get(PPC::RLWINM), Shift1Reg).addReg(Ptr1Reg)
        .addImm(3).addImm(27).addImm(is8bit ? 28 : 27);
    BuildMI(BB, dl, TII->get(is64bit ? PPC::XORI8 : PPC::XORI), ShiftReg)
        .addReg(Shift1Reg).addImm(is8bit ? 24 : 16);
    if (is64bit)
      BuildMI(BB, dl, TII->get(PPC::RLDICR), PtrReg)
        .addReg(Ptr1Reg).addImm(0).addImm(61);
    else
      BuildMI(BB, dl, TII->get(PPC::RLWINM), PtrReg)
        .addReg(Ptr1Reg).addImm(0).addImm(0).addImm(29);
    BuildMI(BB, dl, TII->get(PPC::SLW), NewVal2Reg)
        .addReg(newval).addReg(ShiftReg);
    BuildMI(BB, dl, TII->get(PPC::SLW), OldVal2Reg)
        .addReg(oldval).addReg(ShiftReg);
    if (is8bit)
      BuildMI(BB, dl, TII->get(PPC::LI), Mask2Reg).addImm(255);
    else {
      BuildMI(BB, dl, TII->get(PPC::LI), Mask3Reg).addImm(0);
      BuildMI(BB, dl, TII->get(PPC::ORI), Mask2Reg)
        .addReg(Mask3Reg).addImm(65535);
    }
    BuildMI(BB, dl, TII->get(PPC::SLW), MaskReg)
        .addReg(Mask2Reg).addReg(ShiftReg);
    BuildMI(BB, dl, TII->get(PPC::AND), NewVal3Reg)
        .addReg(NewVal2Reg).addReg(MaskReg);
    BuildMI(BB, dl, TII->get(PPC::AND), OldVal3Reg)
        .addReg(OldVal2Reg).addReg(MaskReg);

    BB = loop1MBB;
    BuildMI(BB, dl, TII->get(PPC::LWARX), TmpDestReg)
        .addReg(ZeroReg).addReg(PtrReg);
    BuildMI(BB, dl, TII->get(PPC::AND),TmpReg)
        .addReg(TmpDestReg).addReg(MaskReg);
    BuildMI(BB, dl, TII->get(PPC::CMPW), PPC::CR0)
        .addReg(TmpReg).addReg(OldVal3Reg);
    BuildMI(BB, dl, TII->get(PPC::BCC))
        .addImm(PPC::PRED_NE).addReg(PPC::CR0).addMBB(midMBB);
    BB->addSuccessor(loop2MBB);
    BB->addSuccessor(midMBB);

    BB = loop2MBB;
    BuildMI(BB, dl, TII->get(PPC::ANDC),Tmp2Reg)
        .addReg(TmpDestReg).addReg(MaskReg);
    BuildMI(BB, dl, TII->get(PPC::OR),Tmp4Reg)
        .addReg(Tmp2Reg).addReg(NewVal3Reg);
    BuildMI(BB, dl, TII->get(PPC::STWCX)).addReg(Tmp4Reg)
        .addReg(ZeroReg).addReg(PtrReg);
    BuildMI(BB, dl, TII->get(PPC::BCC))
      .addImm(PPC::PRED_NE).addReg(PPC::CR0).addMBB(loop1MBB);
    BuildMI(BB, dl, TII->get(PPC::B)).addMBB(exitMBB);
    BB->addSuccessor(loop1MBB);
    BB->addSuccessor(exitMBB);

    BB = midMBB;
    BuildMI(BB, dl, TII->get(PPC::STWCX)).addReg(TmpDestReg)
      .addReg(ZeroReg).addReg(PtrReg);
    BB->addSuccessor(exitMBB);

    //  exitMBB:
    //   ...
    BB = exitMBB;
    BuildMI(*BB, BB->begin(), dl, TII->get(PPC::SRW),dest).addReg(TmpReg)
      .addReg(ShiftReg);
  } else {
    llvm_unreachable("Unexpected instr type to insert");
  }

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return BB;
}

//===----------------------------------------------------------------------===//
// Target Optimization Hooks
//===----------------------------------------------------------------------===//

SDValue PPCTargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  const TargetMachine &TM = getTargetMachine();
  SelectionDAG &DAG = DCI.DAG;
  DebugLoc dl = N->getDebugLoc();
  switch (N->getOpcode()) {
  default: break;
  case PPCISD::SHL:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(0))) {
      if (C->isNullValue())   // 0 << V -> 0.
        return N->getOperand(0);
    }
    break;
  case PPCISD::SRL:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(0))) {
      if (C->isNullValue())   // 0 >>u V -> 0.
        return N->getOperand(0);
    }
    break;
  case PPCISD::SRA:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(0))) {
      if (C->isNullValue() ||   //  0 >>s V -> 0.
          C->isAllOnesValue())    // -1 >>s V -> -1.
        return N->getOperand(0);
    }
    break;

  case ISD::SINT_TO_FP:
    if (TM.getSubtarget<PPCSubtarget>().has64BitSupport()) {
      if (N->getOperand(0).getOpcode() == ISD::FP_TO_SINT) {
        // Turn (sint_to_fp (fp_to_sint X)) -> fctidz/fcfid without load/stores.
        // We allow the src/dst to be either f32/f64, but the intermediate
        // type must be i64.
        if (N->getOperand(0).getValueType() == MVT::i64 &&
            N->getOperand(0).getOperand(0).getValueType() != MVT::ppcf128) {
          SDValue Val = N->getOperand(0).getOperand(0);
          if (Val.getValueType() == MVT::f32) {
            Val = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, Val);
            DCI.AddToWorklist(Val.getNode());
          }

          Val = DAG.getNode(PPCISD::FCTIDZ, dl, MVT::f64, Val);
          DCI.AddToWorklist(Val.getNode());
          Val = DAG.getNode(PPCISD::FCFID, dl, MVT::f64, Val);
          DCI.AddToWorklist(Val.getNode());
          if (N->getValueType(0) == MVT::f32) {
            Val = DAG.getNode(ISD::FP_ROUND, dl, MVT::f32, Val,
                              DAG.getIntPtrConstant(0));
            DCI.AddToWorklist(Val.getNode());
          }
          return Val;
        } else if (N->getOperand(0).getValueType() == MVT::i32) {
          // If the intermediate type is i32, we can avoid the load/store here
          // too.
        }
      }
    }
    break;
  case ISD::STORE:
    // Turn STORE (FP_TO_SINT F) -> STFIWX(FCTIWZ(F)).
    if (TM.getSubtarget<PPCSubtarget>().hasSTFIWX() &&
        !cast<StoreSDNode>(N)->isTruncatingStore() &&
        N->getOperand(1).getOpcode() == ISD::FP_TO_SINT &&
        N->getOperand(1).getValueType() == MVT::i32 &&
        N->getOperand(1).getOperand(0).getValueType() != MVT::ppcf128) {
      SDValue Val = N->getOperand(1).getOperand(0);
      if (Val.getValueType() == MVT::f32) {
        Val = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f64, Val);
        DCI.AddToWorklist(Val.getNode());
      }
      Val = DAG.getNode(PPCISD::FCTIWZ, dl, MVT::f64, Val);
      DCI.AddToWorklist(Val.getNode());

      Val = DAG.getNode(PPCISD::STFIWX, dl, MVT::Other, N->getOperand(0), Val,
                        N->getOperand(2), N->getOperand(3));
      DCI.AddToWorklist(Val.getNode());
      return Val;
    }

    // Turn STORE (BSWAP) -> sthbrx/stwbrx.
    if (cast<StoreSDNode>(N)->isUnindexed() &&
        N->getOperand(1).getOpcode() == ISD::BSWAP &&
        N->getOperand(1).getNode()->hasOneUse() &&
        (N->getOperand(1).getValueType() == MVT::i32 ||
         N->getOperand(1).getValueType() == MVT::i16)) {
      SDValue BSwapOp = N->getOperand(1).getOperand(0);
      // Do an any-extend to 32-bits if this is a half-word input.
      if (BSwapOp.getValueType() == MVT::i16)
        BSwapOp = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, BSwapOp);

      SDValue Ops[] = {
        N->getOperand(0), BSwapOp, N->getOperand(2),
        DAG.getValueType(N->getOperand(1).getValueType())
      };
      return
        DAG.getMemIntrinsicNode(PPCISD::STBRX, dl, DAG.getVTList(MVT::Other),
                                Ops, array_lengthof(Ops),
                                cast<StoreSDNode>(N)->getMemoryVT(),
                                cast<StoreSDNode>(N)->getMemOperand());
    }
    break;
  case ISD::BSWAP:
    // Turn BSWAP (LOAD) -> lhbrx/lwbrx.
    if (ISD::isNON_EXTLoad(N->getOperand(0).getNode()) &&
        N->getOperand(0).hasOneUse() &&
        (N->getValueType(0) == MVT::i32 || N->getValueType(0) == MVT::i16)) {
      SDValue Load = N->getOperand(0);
      LoadSDNode *LD = cast<LoadSDNode>(Load);
      // Create the byte-swapping load.
      SDValue Ops[] = {
        LD->getChain(),    // Chain
        LD->getBasePtr(),  // Ptr
        DAG.getValueType(N->getValueType(0)) // VT
      };
      SDValue BSLoad =
        DAG.getMemIntrinsicNode(PPCISD::LBRX, dl,
                                DAG.getVTList(MVT::i32, MVT::Other), Ops, 3,
                                LD->getMemoryVT(), LD->getMemOperand());

      // If this is an i16 load, insert the truncate.
      SDValue ResVal = BSLoad;
      if (N->getValueType(0) == MVT::i16)
        ResVal = DAG.getNode(ISD::TRUNCATE, dl, MVT::i16, BSLoad);

      // First, combine the bswap away.  This makes the value produced by the
      // load dead.
      DCI.CombineTo(N, ResVal);

      // Next, combine the load away, we give it a bogus result value but a real
      // chain result.  The result value is dead because the bswap is dead.
      DCI.CombineTo(Load.getNode(), ResVal, BSLoad.getValue(1));

      // Return N so it doesn't get rechecked!
      return SDValue(N, 0);
    }

    break;
  case PPCISD::VCMP: {
    // If a VCMPo node already exists with exactly the same operands as this
    // node, use its result instead of this node (VCMPo computes both a CR6 and
    // a normal output).
    //
    if (!N->getOperand(0).hasOneUse() &&
        !N->getOperand(1).hasOneUse() &&
        !N->getOperand(2).hasOneUse()) {

      // Scan all of the users of the LHS, looking for VCMPo's that match.
      SDNode *VCMPoNode = 0;

      SDNode *LHSN = N->getOperand(0).getNode();
      for (SDNode::use_iterator UI = LHSN->use_begin(), E = LHSN->use_end();
           UI != E; ++UI)
        if (UI->getOpcode() == PPCISD::VCMPo &&
            UI->getOperand(1) == N->getOperand(1) &&
            UI->getOperand(2) == N->getOperand(2) &&
            UI->getOperand(0) == N->getOperand(0)) {
          VCMPoNode = *UI;
          break;
        }

      // If there is no VCMPo node, or if the flag value has a single use, don't
      // transform this.
      if (!VCMPoNode || VCMPoNode->hasNUsesOfValue(0, 1))
        break;

      // Look at the (necessarily single) use of the flag value.  If it has a
      // chain, this transformation is more complex.  Note that multiple things
      // could use the value result, which we should ignore.
      SDNode *FlagUser = 0;
      for (SDNode::use_iterator UI = VCMPoNode->use_begin();
           FlagUser == 0; ++UI) {
        assert(UI != VCMPoNode->use_end() && "Didn't find user!");
        SDNode *User = *UI;
        for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i) {
          if (User->getOperand(i) == SDValue(VCMPoNode, 1)) {
            FlagUser = User;
            break;
          }
        }
      }

      // If the user is a MFCR instruction, we know this is safe.  Otherwise we
      // give up for right now.
      if (FlagUser->getOpcode() == PPCISD::MFCR)
        return SDValue(VCMPoNode, 0);
    }
    break;
  }
  case ISD::BR_CC: {
    // If this is a branch on an altivec predicate comparison, lower this so
    // that we don't have to do a MFCR: instead, branch directly on CR6.  This
    // lowering is done pre-legalize, because the legalizer lowers the predicate
    // compare down to code that is difficult to reassemble.
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(1))->get();
    SDValue LHS = N->getOperand(2), RHS = N->getOperand(3);
    int CompareOpc;
    bool isDot;

    if (LHS.getOpcode() == ISD::INTRINSIC_WO_CHAIN &&
        isa<ConstantSDNode>(RHS) && (CC == ISD::SETEQ || CC == ISD::SETNE) &&
        getAltivecCompareInfo(LHS, CompareOpc, isDot)) {
      assert(isDot && "Can't compare against a vector result!");

      // If this is a comparison against something other than 0/1, then we know
      // that the condition is never/always true.
      unsigned Val = cast<ConstantSDNode>(RHS)->getZExtValue();
      if (Val != 0 && Val != 1) {
        if (CC == ISD::SETEQ)      // Cond never true, remove branch.
          return N->getOperand(0);
        // Always !=, turn it into an unconditional branch.
        return DAG.getNode(ISD::BR, dl, MVT::Other,
                           N->getOperand(0), N->getOperand(4));
      }

      bool BranchOnWhenPredTrue = (CC == ISD::SETEQ) ^ (Val == 0);

      // Create the PPCISD altivec 'dot' comparison node.
      std::vector<EVT> VTs;
      SDValue Ops[] = {
        LHS.getOperand(2),  // LHS of compare
        LHS.getOperand(3),  // RHS of compare
        DAG.getConstant(CompareOpc, MVT::i32)
      };
      VTs.push_back(LHS.getOperand(2).getValueType());
      VTs.push_back(MVT::Glue);
      SDValue CompNode = DAG.getNode(PPCISD::VCMPo, dl, VTs, Ops, 3);

      // Unpack the result based on how the target uses it.
      PPC::Predicate CompOpc;
      switch (cast<ConstantSDNode>(LHS.getOperand(1))->getZExtValue()) {
      default:  // Can't happen, don't crash on invalid number though.
      case 0:   // Branch on the value of the EQ bit of CR6.
        CompOpc = BranchOnWhenPredTrue ? PPC::PRED_EQ : PPC::PRED_NE;
        break;
      case 1:   // Branch on the inverted value of the EQ bit of CR6.
        CompOpc = BranchOnWhenPredTrue ? PPC::PRED_NE : PPC::PRED_EQ;
        break;
      case 2:   // Branch on the value of the LT bit of CR6.
        CompOpc = BranchOnWhenPredTrue ? PPC::PRED_LT : PPC::PRED_GE;
        break;
      case 3:   // Branch on the inverted value of the LT bit of CR6.
        CompOpc = BranchOnWhenPredTrue ? PPC::PRED_GE : PPC::PRED_LT;
        break;
      }

      return DAG.getNode(PPCISD::COND_BRANCH, dl, MVT::Other, N->getOperand(0),
                         DAG.getConstant(CompOpc, MVT::i32),
                         DAG.getRegister(PPC::CR6, MVT::i32),
                         N->getOperand(4), CompNode.getValue(1));
    }
    break;
  }
  }

  return SDValue();
}

//===----------------------------------------------------------------------===//
// Inline Assembly Support
//===----------------------------------------------------------------------===//

void PPCTargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                       APInt &KnownZero,
                                                       APInt &KnownOne,
                                                       const SelectionDAG &DAG,
                                                       unsigned Depth) const {
  KnownZero = KnownOne = APInt(KnownZero.getBitWidth(), 0);
  switch (Op.getOpcode()) {
  default: break;
  case PPCISD::LBRX: {
    // lhbrx is known to have the top bits cleared out.
    if (cast<VTSDNode>(Op.getOperand(2))->getVT() == MVT::i16)
      KnownZero = 0xFFFF0000;
    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    switch (cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue()) {
    default: break;
    case Intrinsic::ppc_altivec_vcmpbfp_p:
    case Intrinsic::ppc_altivec_vcmpeqfp_p:
    case Intrinsic::ppc_altivec_vcmpequb_p:
    case Intrinsic::ppc_altivec_vcmpequh_p:
    case Intrinsic::ppc_altivec_vcmpequw_p:
    case Intrinsic::ppc_altivec_vcmpgefp_p:
    case Intrinsic::ppc_altivec_vcmpgtfp_p:
    case Intrinsic::ppc_altivec_vcmpgtsb_p:
    case Intrinsic::ppc_altivec_vcmpgtsh_p:
    case Intrinsic::ppc_altivec_vcmpgtsw_p:
    case Intrinsic::ppc_altivec_vcmpgtub_p:
    case Intrinsic::ppc_altivec_vcmpgtuh_p:
    case Intrinsic::ppc_altivec_vcmpgtuw_p:
      KnownZero = ~1U;  // All bits but the low one are known to be zero.
      break;
    }
  }
  }
}


/// getConstraintType - Given a constraint, return the type of
/// constraint it is for this target.
PPCTargetLowering::ConstraintType
PPCTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 'b':
    case 'r':
    case 'f':
    case 'v':
    case 'y':
      return C_RegisterClass;
    case 'Z':
      // FIXME: While Z does indicate a memory constraint, it specifically
      // indicates an r+r address (used in conjunction with the 'y' modifier
      // in the replacement string). Currently, we're forcing the base
      // register to be r0 in the asm printer (which is interpreted as zero)
      // and forming the complete address in the second register. This is
      // suboptimal.
      return C_Memory;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
PPCTargetLowering::getSingleConstraintMatchWeight(
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
  case 'b':
    if (type->isIntegerTy())
      weight = CW_Register;
    break;
  case 'f':
    if (type->isFloatTy())
      weight = CW_Register;
    break;
  case 'd':
    if (type->isDoubleTy())
      weight = CW_Register;
    break;
  case 'v':
    if (type->isVectorTy())
      weight = CW_Register;
    break;
  case 'y':
    weight = CW_Register;
    break;
  case 'Z':
    weight = CW_Memory;
    break;
  }
  return weight;
}

std::pair<unsigned, const TargetRegisterClass*>
PPCTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                EVT VT) const {
  if (Constraint.size() == 1) {
    // GCC RS6000 Constraint Letters
    switch (Constraint[0]) {
    case 'b':   // R1-R31
    case 'r':   // R0-R31
      if (VT == MVT::i64 && PPCSubTarget.isPPC64())
        return std::make_pair(0U, &PPC::G8RCRegClass);
      return std::make_pair(0U, &PPC::GPRCRegClass);
    case 'f':
      if (VT == MVT::f32 || VT == MVT::i32)
        return std::make_pair(0U, &PPC::F4RCRegClass);
      if (VT == MVT::f64 || VT == MVT::i64)
        return std::make_pair(0U, &PPC::F8RCRegClass);
      break;
    case 'v':
      return std::make_pair(0U, &PPC::VRRCRegClass);
    case 'y':   // crrc
      return std::make_pair(0U, &PPC::CRRCRegClass);
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}


/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void PPCTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                     std::string &Constraint,
                                                     std::vector<SDValue>&Ops,
                                                     SelectionDAG &DAG) const {
  SDValue Result(0,0);

  // Only support length 1 constraints.
  if (Constraint.length() > 1) return;

  char Letter = Constraint[0];
  switch (Letter) {
  default: break;
  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P': {
    ConstantSDNode *CST = dyn_cast<ConstantSDNode>(Op);
    if (!CST) return; // Must be an immediate to match.
    unsigned Value = CST->getZExtValue();
    switch (Letter) {
    default: llvm_unreachable("Unknown constraint letter!");
    case 'I':  // "I" is a signed 16-bit constant.
      if ((short)Value == (int)Value)
        Result = DAG.getTargetConstant(Value, Op.getValueType());
      break;
    case 'J':  // "J" is a constant with only the high-order 16 bits nonzero.
    case 'L':  // "L" is a signed 16-bit constant shifted left 16 bits.
      if ((short)Value == 0)
        Result = DAG.getTargetConstant(Value, Op.getValueType());
      break;
    case 'K':  // "K" is a constant with only the low-order 16 bits nonzero.
      if ((Value >> 16) == 0)
        Result = DAG.getTargetConstant(Value, Op.getValueType());
      break;
    case 'M':  // "M" is a constant that is greater than 31.
      if (Value > 31)
        Result = DAG.getTargetConstant(Value, Op.getValueType());
      break;
    case 'N':  // "N" is a positive constant that is an exact power of two.
      if ((int)Value > 0 && isPowerOf2_32(Value))
        Result = DAG.getTargetConstant(Value, Op.getValueType());
      break;
    case 'O':  // "O" is the constant zero.
      if (Value == 0)
        Result = DAG.getTargetConstant(Value, Op.getValueType());
      break;
    case 'P':  // "P" is a constant whose negation is a signed 16-bit constant.
      if ((short)-Value == (int)-Value)
        Result = DAG.getTargetConstant(Value, Op.getValueType());
      break;
    }
    break;
  }
  }

  if (Result.getNode()) {
    Ops.push_back(Result);
    return;
  }

  // Handle standard constraint letters.
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

// isLegalAddressingMode - Return true if the addressing mode represented
// by AM is legal for this target, for a load/store of the specified type.
bool PPCTargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                              Type *Ty) const {
  // FIXME: PPC does not allow r+i addressing modes for vectors!

  // PPC allows a sign-extended 16-bit immediate field.
  if (AM.BaseOffs <= -(1LL << 16) || AM.BaseOffs >= (1LL << 16)-1)
    return false;

  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // PPC only support r+r,
  switch (AM.Scale) {
  case 0:  // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (AM.HasBaseReg && AM.BaseOffs)  // "r+r+i" is not allowed.
      return false;
    // Otherwise we have r+r or r+i.
    break;
  case 2:
    if (AM.HasBaseReg || AM.BaseOffs)  // 2*r+r  or  2*r+i is not allowed.
      return false;
    // Allow 2*r as r+r.
    break;
  default:
    // No other scales are supported.
    return false;
  }

  return true;
}

/// isLegalAddressImmediate - Return true if the integer value can be used
/// as the offset of the target addressing mode for load / store of the
/// given type.
bool PPCTargetLowering::isLegalAddressImmediate(int64_t V,Type *Ty) const{
  // PPC allows a sign-extended 16-bit immediate field.
  return (V > -(1 << 16) && V < (1 << 16)-1);
}

bool PPCTargetLowering::isLegalAddressImmediate(GlobalValue* GV) const {
  return false;
}

SDValue PPCTargetLowering::LowerRETURNADDR(SDValue Op,
                                           SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MFI->setReturnAddressIsTaken(true);

  DebugLoc dl = Op.getDebugLoc();
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();

  // Make sure the function does not optimize away the store of the RA to
  // the stack.
  PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();
  FuncInfo->setLRStoreRequired();
  bool isPPC64 = PPCSubTarget.isPPC64();
  bool isDarwinABI = PPCSubTarget.isDarwinABI();

  if (Depth > 0) {
    SDValue FrameAddr = LowerFRAMEADDR(Op, DAG);
    SDValue Offset =

      DAG.getConstant(PPCFrameLowering::getReturnSaveOffset(isPPC64, isDarwinABI),
                      isPPC64? MVT::i64 : MVT::i32);
    return DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, dl, getPointerTy(),
                                   FrameAddr, Offset),
                       MachinePointerInfo(), false, false, false, 0);
  }

  // Just load the return address off the stack.
  SDValue RetAddrFI = getReturnAddrFrameIndex(DAG);
  return DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(),
                     RetAddrFI, MachinePointerInfo(), false, false, false, 0);
}

SDValue PPCTargetLowering::LowerFRAMEADDR(SDValue Op,
                                          SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();

  EVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  bool isPPC64 = PtrVT == MVT::i64;

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MFI->setFrameAddressIsTaken(true);
  bool is31 = (getTargetMachine().Options.DisableFramePointerElim(MF) ||
               MFI->hasVarSizedObjects()) &&
                  MFI->getStackSize() &&
                  !MF.getFunction()->getFnAttributes().
                    hasAttribute(Attributes::Naked);
  unsigned FrameReg = isPPC64 ? (is31 ? PPC::X31 : PPC::X1) :
                                (is31 ? PPC::R31 : PPC::R1);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl, FrameReg,
                                         PtrVT);
  while (Depth--)
    FrameAddr = DAG.getLoad(Op.getValueType(), dl, DAG.getEntryNode(),
                            FrameAddr, MachinePointerInfo(), false, false,
                            false, 0);
  return FrameAddr;
}

bool
PPCTargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The PowerPC target isn't yet aware of offsets.
  return false;
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
EVT PPCTargetLowering::getOptimalMemOpType(uint64_t Size,
                                           unsigned DstAlign, unsigned SrcAlign,
                                           bool IsZeroVal,
                                           bool MemcpyStrSrc,
                                           MachineFunction &MF) const {
  if (this->PPCSubTarget.isPPC64()) {
    return MVT::i64;
  } else {
    return MVT::i32;
  }
}

/// isFMAFasterThanMulAndAdd - Return true if an FMA operation is faster than
/// a pair of mul and add instructions. fmuladd intrinsics will be expanded to
/// FMAs when this method returns true (and FMAs are legal), otherwise fmuladd
/// is expanded to mul + add.
bool PPCTargetLowering::isFMAFasterThanMulAndAdd(EVT VT) const {
  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
  case MVT::f64:
  case MVT::v4f32:
    return true;
  default:
    break;
  }

  return false;
}

Sched::Preference PPCTargetLowering::getSchedulingPreference(SDNode *N) const {
  if (DisableILPPref)
    return TargetLowering::getSchedulingPreference(N);

  return Sched::ILP;
}

