//===-- RISCVISelLowering.cpp - RISCV DAG Lowering Implementation  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that RISCV uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "RISCVISelLowering.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-lower"

STATISTIC(NumTailCalls, "Number of tail calls");

RISCVTargetLowering::RISCVTargetLowering(const TargetMachine &TM,
                                         const RISCVSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {

  if (Subtarget.isRV32E())
    report_fatal_error("Codegen not yet implemented for RV32E");

  RISCVABI::ABI ABI = Subtarget.getTargetABI();
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialised target ABI");

  if ((ABI == RISCVABI::ABI_ILP32F || ABI == RISCVABI::ABI_LP64F) &&
      !Subtarget.hasStdExtF()) {
    errs() << "Hard-float 'f' ABI can't be used for a target that "
                "doesn't support the F instruction set extension (ignoring "
                          "target-abi)\n";
    ABI = Subtarget.is64Bit() ? RISCVABI::ABI_LP64 : RISCVABI::ABI_ILP32;
  } else if ((ABI == RISCVABI::ABI_ILP32D || ABI == RISCVABI::ABI_LP64D) &&
             !Subtarget.hasStdExtD()) {
    errs() << "Hard-float 'd' ABI can't be used for a target that "
              "doesn't support the D instruction set extension (ignoring "
              "target-abi)\n";
    ABI = Subtarget.is64Bit() ? RISCVABI::ABI_LP64 : RISCVABI::ABI_ILP32;
  }

  switch (ABI) {
  default:
    report_fatal_error("Don't know how to lower this ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64:
  case RISCVABI::ABI_LP64F:
  case RISCVABI::ABI_LP64D:
    break;
  }

  MVT XLenVT = Subtarget.getXLenVT();

  // Set up the register classes.
  addRegisterClass(XLenVT, &RISCV::GPRRegClass);

  if (Subtarget.hasStdExtZfh())
    addRegisterClass(MVT::f16, &RISCV::FPR16RegClass);
  if (Subtarget.hasStdExtF())
    addRegisterClass(MVT::f32, &RISCV::FPR32RegClass);
  if (Subtarget.hasStdExtD())
    addRegisterClass(MVT::f64, &RISCV::FPR64RegClass);

  static const MVT::SimpleValueType BoolVecVTs[] = {
      MVT::nxv1i1,  MVT::nxv2i1,  MVT::nxv4i1, MVT::nxv8i1,
      MVT::nxv16i1, MVT::nxv32i1, MVT::nxv64i1};
  static const MVT::SimpleValueType IntVecVTs[] = {
      MVT::nxv1i8,  MVT::nxv2i8,   MVT::nxv4i8,   MVT::nxv8i8,  MVT::nxv16i8,
      MVT::nxv32i8, MVT::nxv64i8,  MVT::nxv1i16,  MVT::nxv2i16, MVT::nxv4i16,
      MVT::nxv8i16, MVT::nxv16i16, MVT::nxv32i16, MVT::nxv1i32, MVT::nxv2i32,
      MVT::nxv4i32, MVT::nxv8i32,  MVT::nxv16i32, MVT::nxv1i64, MVT::nxv2i64,
      MVT::nxv4i64, MVT::nxv8i64};
  static const MVT::SimpleValueType F16VecVTs[] = {
      MVT::nxv1f16, MVT::nxv2f16,  MVT::nxv4f16,
      MVT::nxv8f16, MVT::nxv16f16, MVT::nxv32f16};
  static const MVT::SimpleValueType F32VecVTs[] = {
      MVT::nxv1f32, MVT::nxv2f32, MVT::nxv4f32, MVT::nxv8f32, MVT::nxv16f32};
  static const MVT::SimpleValueType F64VecVTs[] = {
      MVT::nxv1f64, MVT::nxv2f64, MVT::nxv4f64, MVT::nxv8f64};

  if (Subtarget.hasVInstructions()) {
    auto addRegClassForRVV = [this](MVT VT) {
      unsigned Size = VT.getSizeInBits().getKnownMinValue();
      assert(Size <= 512 && isPowerOf2_32(Size));
      const TargetRegisterClass *RC;
      if (Size <= 64)
        RC = &RISCV::VRRegClass;
      else if (Size == 128)
        RC = &RISCV::VRM2RegClass;
      else if (Size == 256)
        RC = &RISCV::VRM4RegClass;
      else
        RC = &RISCV::VRM8RegClass;

      addRegisterClass(VT, RC);
    };

    for (MVT VT : BoolVecVTs)
      addRegClassForRVV(VT);
    for (MVT VT : IntVecVTs) {
      if (VT.getVectorElementType() == MVT::i64 &&
          !Subtarget.hasVInstructionsI64())
        continue;
      addRegClassForRVV(VT);
    }

    if (Subtarget.hasVInstructionsF16())
      for (MVT VT : F16VecVTs)
        addRegClassForRVV(VT);

    if (Subtarget.hasVInstructionsF32())
      for (MVT VT : F32VecVTs)
        addRegClassForRVV(VT);

    if (Subtarget.hasVInstructionsF64())
      for (MVT VT : F64VecVTs)
        addRegClassForRVV(VT);

    if (Subtarget.useRVVForFixedLengthVectors()) {
      auto addRegClassForFixedVectors = [this](MVT VT) {
        MVT ContainerVT = getContainerForFixedLengthVector(VT);
        unsigned RCID = getRegClassIDForVecVT(ContainerVT);
        const RISCVRegisterInfo &TRI = *Subtarget.getRegisterInfo();
        addRegisterClass(VT, TRI.getRegClass(RCID));
      };
      for (MVT VT : MVT::integer_fixedlen_vector_valuetypes())
        if (useRVVForFixedLengthVectorVT(VT))
          addRegClassForFixedVectors(VT);

      for (MVT VT : MVT::fp_fixedlen_vector_valuetypes())
        if (useRVVForFixedLengthVectorVT(VT))
          addRegClassForFixedVectors(VT);
    }
  }

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(RISCV::X2);

  for (auto N : {ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD})
    setLoadExtAction(N, XLenVT, MVT::i1, Promote);

  // TODO: add all necessary setOperationAction calls.
  setOperationAction(ISD::DYNAMIC_STACKALLOC, XLenVT, Expand);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, XLenVT, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::SELECT_CC, XLenVT, Expand);

  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  if (!Subtarget.hasStdExtZbb()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  }

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::ADD, MVT::i32, Custom);
    setOperationAction(ISD::SUB, MVT::i32, Custom);
    setOperationAction(ISD::SHL, MVT::i32, Custom);
    setOperationAction(ISD::SRA, MVT::i32, Custom);
    setOperationAction(ISD::SRL, MVT::i32, Custom);

    setOperationAction(ISD::UADDO, MVT::i32, Custom);
    setOperationAction(ISD::USUBO, MVT::i32, Custom);
    setOperationAction(ISD::UADDSAT, MVT::i32, Custom);
    setOperationAction(ISD::USUBSAT, MVT::i32, Custom);
  } else {
    setLibcallName(RTLIB::SHL_I128, nullptr);
    setLibcallName(RTLIB::SRL_I128, nullptr);
    setLibcallName(RTLIB::SRA_I128, nullptr);
    setLibcallName(RTLIB::MUL_I128, nullptr);
    setLibcallName(RTLIB::MULO_I64, nullptr);
  }

  if (!Subtarget.hasStdExtM()) {
    setOperationAction(ISD::MUL, XLenVT, Expand);
    setOperationAction(ISD::MULHS, XLenVT, Expand);
    setOperationAction(ISD::MULHU, XLenVT, Expand);
    setOperationAction(ISD::SDIV, XLenVT, Expand);
    setOperationAction(ISD::UDIV, XLenVT, Expand);
    setOperationAction(ISD::SREM, XLenVT, Expand);
    setOperationAction(ISD::UREM, XLenVT, Expand);
  } else {
    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::MUL, MVT::i32, Custom);
      setOperationAction(ISD::MUL, MVT::i128, Custom);

      setOperationAction(ISD::SDIV, MVT::i8, Custom);
      setOperationAction(ISD::UDIV, MVT::i8, Custom);
      setOperationAction(ISD::UREM, MVT::i8, Custom);
      setOperationAction(ISD::SDIV, MVT::i16, Custom);
      setOperationAction(ISD::UDIV, MVT::i16, Custom);
      setOperationAction(ISD::UREM, MVT::i16, Custom);
      setOperationAction(ISD::SDIV, MVT::i32, Custom);
      setOperationAction(ISD::UDIV, MVT::i32, Custom);
      setOperationAction(ISD::UREM, MVT::i32, Custom);
    } else {
      setOperationAction(ISD::MUL, MVT::i64, Custom);
    }
  }

  setOperationAction(ISD::SDIVREM, XLenVT, Expand);
  setOperationAction(ISD::UDIVREM, XLenVT, Expand);
  setOperationAction(ISD::SMUL_LOHI, XLenVT, Expand);
  setOperationAction(ISD::UMUL_LOHI, XLenVT, Expand);

  setOperationAction(ISD::SHL_PARTS, XLenVT, Custom);
  setOperationAction(ISD::SRL_PARTS, XLenVT, Custom);
  setOperationAction(ISD::SRA_PARTS, XLenVT, Custom);

  if (Subtarget.hasStdExtZbb() || Subtarget.hasStdExtZbp() ||
      Subtarget.hasStdExtZbkb()) {
    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::ROTL, MVT::i32, Custom);
      setOperationAction(ISD::ROTR, MVT::i32, Custom);
    }
  } else {
    setOperationAction(ISD::ROTL, XLenVT, Expand);
    setOperationAction(ISD::ROTR, XLenVT, Expand);
  }

  if (Subtarget.hasStdExtZbp()) {
    // Custom lower bswap/bitreverse so we can convert them to GREVI to enable
    // more combining.
    setOperationAction(ISD::BITREVERSE, XLenVT,   Custom);
    setOperationAction(ISD::BSWAP,      XLenVT,   Custom);
    setOperationAction(ISD::BITREVERSE, MVT::i8,  Custom);
    // BSWAP i8 doesn't exist.
    setOperationAction(ISD::BITREVERSE, MVT::i16, Custom);
    setOperationAction(ISD::BSWAP,      MVT::i16, Custom);

    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::BITREVERSE, MVT::i32, Custom);
      setOperationAction(ISD::BSWAP,      MVT::i32, Custom);
    }
  } else {
    // With Zbb we have an XLen rev8 instruction, but not GREVI. So we'll
    // pattern match it directly in isel.
    setOperationAction(ISD::BSWAP, XLenVT,
                       (Subtarget.hasStdExtZbb() || Subtarget.hasStdExtZbkb())
                           ? Legal
                           : Expand);
    // Zbkb can use rev8+brev8 to implement bitreverse.
    setOperationAction(ISD::BITREVERSE, XLenVT,
                       Subtarget.hasStdExtZbkb() ? Custom : Expand);
  }

  if (Subtarget.hasStdExtZbb()) {
    setOperationAction(ISD::SMIN, XLenVT, Legal);
    setOperationAction(ISD::SMAX, XLenVT, Legal);
    setOperationAction(ISD::UMIN, XLenVT, Legal);
    setOperationAction(ISD::UMAX, XLenVT, Legal);

    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::CTTZ, MVT::i32, Custom);
      setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Custom);
      setOperationAction(ISD::CTLZ, MVT::i32, Custom);
      setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Custom);
    }
  } else {
    setOperationAction(ISD::CTTZ, XLenVT, Expand);
    setOperationAction(ISD::CTLZ, XLenVT, Expand);
    setOperationAction(ISD::CTPOP, XLenVT, Expand);

    if (Subtarget.is64Bit())
      setOperationAction(ISD::ABS, MVT::i32, Custom);
  }

  if (Subtarget.hasStdExtZbt()) {
    setOperationAction(ISD::FSHL, XLenVT, Custom);
    setOperationAction(ISD::FSHR, XLenVT, Custom);
    setOperationAction(ISD::SELECT, XLenVT, Legal);

    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::FSHL, MVT::i32, Custom);
      setOperationAction(ISD::FSHR, MVT::i32, Custom);
    }
  } else {
    setOperationAction(ISD::SELECT, XLenVT, Custom);
  }

  static constexpr ISD::NodeType FPLegalNodeTypes[] = {
      ISD::FMINNUM,        ISD::FMAXNUM,       ISD::LRINT,
      ISD::LLRINT,         ISD::LROUND,        ISD::LLROUND,
      ISD::STRICT_LRINT,   ISD::STRICT_LLRINT, ISD::STRICT_LROUND,
      ISD::STRICT_LLROUND, ISD::STRICT_FMA,    ISD::STRICT_FADD,
      ISD::STRICT_FSUB,    ISD::STRICT_FMUL,   ISD::STRICT_FDIV,
      ISD::STRICT_FSQRT,   ISD::STRICT_FSETCC, ISD::STRICT_FSETCCS};

  static const ISD::CondCode FPCCToExpand[] = {
      ISD::SETOGT, ISD::SETOGE, ISD::SETONE, ISD::SETUEQ, ISD::SETUGT,
      ISD::SETUGE, ISD::SETULT, ISD::SETULE, ISD::SETUNE, ISD::SETGT,
      ISD::SETGE,  ISD::SETNE,  ISD::SETO,   ISD::SETUO};

  static const ISD::NodeType FPOpToExpand[] = {
      ISD::FSIN, ISD::FCOS,       ISD::FSINCOS,   ISD::FPOW,
      ISD::FREM, ISD::FP16_TO_FP, ISD::FP_TO_FP16};

  if (Subtarget.hasStdExtZfh())
    setOperationAction(ISD::BITCAST, MVT::i16, Custom);

  if (Subtarget.hasStdExtZfh()) {
    for (auto NT : FPLegalNodeTypes)
      setOperationAction(NT, MVT::f16, Legal);
    setOperationAction(ISD::STRICT_FP_ROUND, MVT::f16, Legal);
    setOperationAction(ISD::STRICT_FP_EXTEND, MVT::f32, Legal);
    for (auto CC : FPCCToExpand)
      setCondCodeAction(CC, MVT::f16, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f16, Expand);
    setOperationAction(ISD::SELECT, MVT::f16, Custom);
    setOperationAction(ISD::BR_CC, MVT::f16, Expand);

    setOperationAction(ISD::FREM,       MVT::f16, Promote);
    setOperationAction(ISD::FCEIL,      MVT::f16, Promote);
    setOperationAction(ISD::FFLOOR,     MVT::f16, Promote);
    setOperationAction(ISD::FNEARBYINT, MVT::f16, Promote);
    setOperationAction(ISD::FRINT,      MVT::f16, Promote);
    setOperationAction(ISD::FROUND,     MVT::f16, Promote);
    setOperationAction(ISD::FROUNDEVEN, MVT::f16, Promote);
    setOperationAction(ISD::FTRUNC,     MVT::f16, Promote);
    setOperationAction(ISD::FPOW,       MVT::f16, Promote);
    setOperationAction(ISD::FPOWI,      MVT::f16, Promote);
    setOperationAction(ISD::FCOS,       MVT::f16, Promote);
    setOperationAction(ISD::FSIN,       MVT::f16, Promote);
    setOperationAction(ISD::FSINCOS,    MVT::f16, Promote);
    setOperationAction(ISD::FEXP,       MVT::f16, Promote);
    setOperationAction(ISD::FEXP2,      MVT::f16, Promote);
    setOperationAction(ISD::FLOG,       MVT::f16, Promote);
    setOperationAction(ISD::FLOG2,      MVT::f16, Promote);
    setOperationAction(ISD::FLOG10,     MVT::f16, Promote);

    // FIXME: Need to promote f16 STRICT_* to f32 libcalls, but we don't have
    // complete support for all operations in LegalizeDAG.

    // We need to custom promote this.
    if (Subtarget.is64Bit())
      setOperationAction(ISD::FPOWI, MVT::i32, Custom);
  }

  if (Subtarget.hasStdExtF()) {
    for (auto NT : FPLegalNodeTypes)
      setOperationAction(NT, MVT::f32, Legal);
    for (auto CC : FPCCToExpand)
      setCondCodeAction(CC, MVT::f32, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
    setOperationAction(ISD::SELECT, MVT::f32, Custom);
    setOperationAction(ISD::BR_CC, MVT::f32, Expand);
    for (auto Op : FPOpToExpand)
      setOperationAction(Op, MVT::f32, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f16, Expand);
    setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  }

  if (Subtarget.hasStdExtF() && Subtarget.is64Bit())
    setOperationAction(ISD::BITCAST, MVT::i32, Custom);

  if (Subtarget.hasStdExtD()) {
    for (auto NT : FPLegalNodeTypes)
      setOperationAction(NT, MVT::f64, Legal);
    setOperationAction(ISD::STRICT_FP_ROUND, MVT::f32, Legal);
    setOperationAction(ISD::STRICT_FP_EXTEND, MVT::f64, Legal);
    for (auto CC : FPCCToExpand)
      setCondCodeAction(CC, MVT::f64, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
    setOperationAction(ISD::SELECT, MVT::f64, Custom);
    setOperationAction(ISD::BR_CC, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
    for (auto Op : FPOpToExpand)
      setOperationAction(Op, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f16, Expand);
    setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  }

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
    setOperationAction(ISD::STRICT_FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::STRICT_FP_TO_SINT, MVT::i32, Custom);
  }

  if (Subtarget.hasStdExtF()) {
    setOperationAction(ISD::FP_TO_UINT_SAT, XLenVT, Custom);
    setOperationAction(ISD::FP_TO_SINT_SAT, XLenVT, Custom);

    setOperationAction(ISD::STRICT_FP_TO_UINT, XLenVT, Legal);
    setOperationAction(ISD::STRICT_FP_TO_SINT, XLenVT, Legal);
    setOperationAction(ISD::STRICT_UINT_TO_FP, XLenVT, Legal);
    setOperationAction(ISD::STRICT_SINT_TO_FP, XLenVT, Legal);

    setOperationAction(ISD::FLT_ROUNDS_, XLenVT, Custom);
    setOperationAction(ISD::SET_ROUNDING, MVT::Other, Custom);
  }

  setOperationAction(ISD::GlobalAddress, XLenVT, Custom);
  setOperationAction(ISD::BlockAddress, XLenVT, Custom);
  setOperationAction(ISD::ConstantPool, XLenVT, Custom);
  setOperationAction(ISD::JumpTable, XLenVT, Custom);

  setOperationAction(ISD::GlobalTLSAddress, XLenVT, Custom);

  // TODO: On M-mode only targets, the cycle[h] CSR may not be present.
  // Unfortunately this can't be determined just from the ISA naming string.
  setOperationAction(ISD::READCYCLECOUNTER, MVT::i64,
                     Subtarget.is64Bit() ? Legal : Custom);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  if (Subtarget.is64Bit())
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i32, Custom);

  if (Subtarget.hasStdExtA()) {
    setMaxAtomicSizeInBitsSupported(Subtarget.getXLen());
    setMinCmpXchgSizeInBits(32);
  } else {
    setMaxAtomicSizeInBitsSupported(0);
  }

  setBooleanContents(ZeroOrOneBooleanContent);

  if (Subtarget.hasVInstructions()) {
    setBooleanVectorContents(ZeroOrOneBooleanContent);

    setOperationAction(ISD::VSCALE, XLenVT, Custom);

    // RVV intrinsics may have illegal operands.
    // We also need to custom legalize vmv.x.s.
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i8, Custom);
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i16, Custom);
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i8, Custom);
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i16, Custom);
    if (Subtarget.is64Bit()) {
      setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i32, Custom);
    } else {
      setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i64, Custom);
      setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i64, Custom);
    }

    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);
    setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

    static const unsigned IntegerVPOps[] = {
        ISD::VP_ADD,         ISD::VP_SUB,         ISD::VP_MUL,
        ISD::VP_SDIV,        ISD::VP_UDIV,        ISD::VP_SREM,
        ISD::VP_UREM,        ISD::VP_AND,         ISD::VP_OR,
        ISD::VP_XOR,         ISD::VP_ASHR,        ISD::VP_LSHR,
        ISD::VP_SHL,         ISD::VP_REDUCE_ADD,  ISD::VP_REDUCE_AND,
        ISD::VP_REDUCE_OR,   ISD::VP_REDUCE_XOR,  ISD::VP_REDUCE_SMAX,
        ISD::VP_REDUCE_SMIN, ISD::VP_REDUCE_UMAX, ISD::VP_REDUCE_UMIN,
        ISD::VP_MERGE,       ISD::VP_SELECT};

    static const unsigned FloatingPointVPOps[] = {
        ISD::VP_FADD,        ISD::VP_FSUB,        ISD::VP_FMUL,
        ISD::VP_FDIV,        ISD::VP_FNEG,        ISD::VP_FMA,
        ISD::VP_REDUCE_FADD, ISD::VP_REDUCE_SEQ_FADD, ISD::VP_REDUCE_FMIN,
        ISD::VP_REDUCE_FMAX, ISD::VP_MERGE,       ISD::VP_SELECT};

    if (!Subtarget.is64Bit()) {
      // We must custom-lower certain vXi64 operations on RV32 due to the vector
      // element type being illegal.
      setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::i64, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::i64, Custom);

      setOperationAction(ISD::VECREDUCE_ADD, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_AND, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_OR, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_XOR, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_SMAX, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_SMIN, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_UMAX, MVT::i64, Custom);
      setOperationAction(ISD::VECREDUCE_UMIN, MVT::i64, Custom);

      setOperationAction(ISD::VP_REDUCE_ADD, MVT::i64, Custom);
      setOperationAction(ISD::VP_REDUCE_AND, MVT::i64, Custom);
      setOperationAction(ISD::VP_REDUCE_OR, MVT::i64, Custom);
      setOperationAction(ISD::VP_REDUCE_XOR, MVT::i64, Custom);
      setOperationAction(ISD::VP_REDUCE_SMAX, MVT::i64, Custom);
      setOperationAction(ISD::VP_REDUCE_SMIN, MVT::i64, Custom);
      setOperationAction(ISD::VP_REDUCE_UMAX, MVT::i64, Custom);
      setOperationAction(ISD::VP_REDUCE_UMIN, MVT::i64, Custom);
    }

    for (MVT VT : BoolVecVTs) {
      setOperationAction(ISD::SPLAT_VECTOR, VT, Custom);

      // Mask VTs are custom-expanded into a series of standard nodes
      setOperationAction(ISD::TRUNCATE, VT, Custom);
      setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);
      setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
      setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

      setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);

      setOperationAction(ISD::SELECT, VT, Custom);
      setOperationAction(ISD::SELECT_CC, VT, Expand);
      setOperationAction(ISD::VSELECT, VT, Expand);
      setOperationAction(ISD::VP_MERGE, VT, Expand);
      setOperationAction(ISD::VP_SELECT, VT, Expand);

      setOperationAction(ISD::VP_AND, VT, Custom);
      setOperationAction(ISD::VP_OR, VT, Custom);
      setOperationAction(ISD::VP_XOR, VT, Custom);

      setOperationAction(ISD::VECREDUCE_AND, VT, Custom);
      setOperationAction(ISD::VECREDUCE_OR, VT, Custom);
      setOperationAction(ISD::VECREDUCE_XOR, VT, Custom);

      setOperationAction(ISD::VP_REDUCE_AND, VT, Custom);
      setOperationAction(ISD::VP_REDUCE_OR, VT, Custom);
      setOperationAction(ISD::VP_REDUCE_XOR, VT, Custom);

      // RVV has native int->float & float->int conversions where the
      // element type sizes are within one power-of-two of each other. Any
      // wider distances between type sizes have to be lowered as sequences
      // which progressively narrow the gap in stages.
      setOperationAction(ISD::SINT_TO_FP, VT, Custom);
      setOperationAction(ISD::UINT_TO_FP, VT, Custom);
      setOperationAction(ISD::FP_TO_SINT, VT, Custom);
      setOperationAction(ISD::FP_TO_UINT, VT, Custom);

      // Expand all extending loads to types larger than this, and truncating
      // stores from types larger than this.
      for (MVT OtherVT : MVT::integer_scalable_vector_valuetypes()) {
        setTruncStoreAction(OtherVT, VT, Expand);
        setLoadExtAction(ISD::EXTLOAD, OtherVT, VT, Expand);
        setLoadExtAction(ISD::SEXTLOAD, OtherVT, VT, Expand);
        setLoadExtAction(ISD::ZEXTLOAD, OtherVT, VT, Expand);
      }
    }

    for (MVT VT : IntVecVTs) {
      if (VT.getVectorElementType() == MVT::i64 &&
          !Subtarget.hasVInstructionsI64())
        continue;

      setOperationAction(ISD::SPLAT_VECTOR, VT, Legal);
      setOperationAction(ISD::SPLAT_VECTOR_PARTS, VT, Custom);

      // Vectors implement MULHS/MULHU.
      setOperationAction(ISD::SMUL_LOHI, VT, Expand);
      setOperationAction(ISD::UMUL_LOHI, VT, Expand);

      // nxvXi64 MULHS/MULHU requires the V extension instead of Zve64*.
      if (VT.getVectorElementType() == MVT::i64 && !Subtarget.hasStdExtV()) {
        setOperationAction(ISD::MULHU, VT, Expand);
        setOperationAction(ISD::MULHS, VT, Expand);
      }

      setOperationAction(ISD::SMIN, VT, Legal);
      setOperationAction(ISD::SMAX, VT, Legal);
      setOperationAction(ISD::UMIN, VT, Legal);
      setOperationAction(ISD::UMAX, VT, Legal);

      setOperationAction(ISD::ROTL, VT, Expand);
      setOperationAction(ISD::ROTR, VT, Expand);

      setOperationAction(ISD::CTTZ, VT, Expand);
      setOperationAction(ISD::CTLZ, VT, Expand);
      setOperationAction(ISD::CTPOP, VT, Expand);

      setOperationAction(ISD::BSWAP, VT, Expand);

      // Custom-lower extensions and truncations from/to mask types.
      setOperationAction(ISD::ANY_EXTEND, VT, Custom);
      setOperationAction(ISD::SIGN_EXTEND, VT, Custom);
      setOperationAction(ISD::ZERO_EXTEND, VT, Custom);

      // RVV has native int->float & float->int conversions where the
      // element type sizes are within one power-of-two of each other. Any
      // wider distances between type sizes have to be lowered as sequences
      // which progressively narrow the gap in stages.
      setOperationAction(ISD::SINT_TO_FP, VT, Custom);
      setOperationAction(ISD::UINT_TO_FP, VT, Custom);
      setOperationAction(ISD::FP_TO_SINT, VT, Custom);
      setOperationAction(ISD::FP_TO_UINT, VT, Custom);

      setOperationAction(ISD::SADDSAT, VT, Legal);
      setOperationAction(ISD::UADDSAT, VT, Legal);
      setOperationAction(ISD::SSUBSAT, VT, Legal);
      setOperationAction(ISD::USUBSAT, VT, Legal);

      // Integer VTs are lowered as a series of "RISCVISD::TRUNCATE_VECTOR_VL"
      // nodes which truncate by one power of two at a time.
      setOperationAction(ISD::TRUNCATE, VT, Custom);

      // Custom-lower insert/extract operations to simplify patterns.
      setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);

      // Custom-lower reduction operations to set up the corresponding custom
      // nodes' operands.
      setOperationAction(ISD::VECREDUCE_ADD, VT, Custom);
      setOperationAction(ISD::VECREDUCE_AND, VT, Custom);
      setOperationAction(ISD::VECREDUCE_OR, VT, Custom);
      setOperationAction(ISD::VECREDUCE_XOR, VT, Custom);
      setOperationAction(ISD::VECREDUCE_SMAX, VT, Custom);
      setOperationAction(ISD::VECREDUCE_SMIN, VT, Custom);
      setOperationAction(ISD::VECREDUCE_UMAX, VT, Custom);
      setOperationAction(ISD::VECREDUCE_UMIN, VT, Custom);

      for (unsigned VPOpc : IntegerVPOps)
        setOperationAction(VPOpc, VT, Custom);

      setOperationAction(ISD::LOAD, VT, Custom);
      setOperationAction(ISD::STORE, VT, Custom);

      setOperationAction(ISD::MLOAD, VT, Custom);
      setOperationAction(ISD::MSTORE, VT, Custom);
      setOperationAction(ISD::MGATHER, VT, Custom);
      setOperationAction(ISD::MSCATTER, VT, Custom);

      setOperationAction(ISD::VP_LOAD, VT, Custom);
      setOperationAction(ISD::VP_STORE, VT, Custom);
      setOperationAction(ISD::VP_GATHER, VT, Custom);
      setOperationAction(ISD::VP_SCATTER, VT, Custom);

      setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);
      setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
      setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

      setOperationAction(ISD::SELECT, VT, Custom);
      setOperationAction(ISD::SELECT_CC, VT, Expand);

      setOperationAction(ISD::STEP_VECTOR, VT, Custom);
      setOperationAction(ISD::VECTOR_REVERSE, VT, Custom);

      for (MVT OtherVT : MVT::integer_scalable_vector_valuetypes()) {
        setTruncStoreAction(VT, OtherVT, Expand);
        setLoadExtAction(ISD::EXTLOAD, OtherVT, VT, Expand);
        setLoadExtAction(ISD::SEXTLOAD, OtherVT, VT, Expand);
        setLoadExtAction(ISD::ZEXTLOAD, OtherVT, VT, Expand);
      }

      // Splice
      setOperationAction(ISD::VECTOR_SPLICE, VT, Custom);

      // Lower CTLZ_ZERO_UNDEF and CTTZ_ZERO_UNDEF if we have a floating point
      // type that can represent the value exactly.
      if (VT.getVectorElementType() != MVT::i64) {
        MVT FloatEltVT =
            VT.getVectorElementType() == MVT::i32 ? MVT::f64 : MVT::f32;
        EVT FloatVT = MVT::getVectorVT(FloatEltVT, VT.getVectorElementCount());
        if (isTypeLegal(FloatVT)) {
          setOperationAction(ISD::CTLZ_ZERO_UNDEF, VT, Custom);
          setOperationAction(ISD::CTTZ_ZERO_UNDEF, VT, Custom);
        }
      }
    }

    // Expand various CCs to best match the RVV ISA, which natively supports UNE
    // but no other unordered comparisons, and supports all ordered comparisons
    // except ONE. Additionally, we expand GT,OGT,GE,OGE for optimization
    // purposes; they are expanded to their swapped-operand CCs (LT,OLT,LE,OLE),
    // and we pattern-match those back to the "original", swapping operands once
    // more. This way we catch both operations and both "vf" and "fv" forms with
    // fewer patterns.
    static const ISD::CondCode VFPCCToExpand[] = {
        ISD::SETO,   ISD::SETONE, ISD::SETUEQ, ISD::SETUGT,
        ISD::SETUGE, ISD::SETULT, ISD::SETULE, ISD::SETUO,
        ISD::SETGT,  ISD::SETOGT, ISD::SETGE,  ISD::SETOGE,
    };

    // Sets common operation actions on RVV floating-point vector types.
    const auto SetCommonVFPActions = [&](MVT VT) {
      setOperationAction(ISD::SPLAT_VECTOR, VT, Legal);
      // RVV has native FP_ROUND & FP_EXTEND conversions where the element type
      // sizes are within one power-of-two of each other. Therefore conversions
      // between vXf16 and vXf64 must be lowered as sequences which convert via
      // vXf32.
      setOperationAction(ISD::FP_ROUND, VT, Custom);
      setOperationAction(ISD::FP_EXTEND, VT, Custom);
      // Custom-lower insert/extract operations to simplify patterns.
      setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
      // Expand various condition codes (explained above).
      for (auto CC : VFPCCToExpand)
        setCondCodeAction(CC, VT, Expand);

      setOperationAction(ISD::FMINNUM, VT, Legal);
      setOperationAction(ISD::FMAXNUM, VT, Legal);

      setOperationAction(ISD::FTRUNC, VT, Custom);
      setOperationAction(ISD::FCEIL, VT, Custom);
      setOperationAction(ISD::FFLOOR, VT, Custom);
      setOperationAction(ISD::FROUND, VT, Custom);

      setOperationAction(ISD::VECREDUCE_FADD, VT, Custom);
      setOperationAction(ISD::VECREDUCE_SEQ_FADD, VT, Custom);
      setOperationAction(ISD::VECREDUCE_FMIN, VT, Custom);
      setOperationAction(ISD::VECREDUCE_FMAX, VT, Custom);

      setOperationAction(ISD::FCOPYSIGN, VT, Legal);

      setOperationAction(ISD::LOAD, VT, Custom);
      setOperationAction(ISD::STORE, VT, Custom);

      setOperationAction(ISD::MLOAD, VT, Custom);
      setOperationAction(ISD::MSTORE, VT, Custom);
      setOperationAction(ISD::MGATHER, VT, Custom);
      setOperationAction(ISD::MSCATTER, VT, Custom);

      setOperationAction(ISD::VP_LOAD, VT, Custom);
      setOperationAction(ISD::VP_STORE, VT, Custom);
      setOperationAction(ISD::VP_GATHER, VT, Custom);
      setOperationAction(ISD::VP_SCATTER, VT, Custom);

      setOperationAction(ISD::SELECT, VT, Custom);
      setOperationAction(ISD::SELECT_CC, VT, Expand);

      setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);
      setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
      setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

      setOperationAction(ISD::VECTOR_REVERSE, VT, Custom);
      setOperationAction(ISD::VECTOR_SPLICE, VT, Custom);

      for (unsigned VPOpc : FloatingPointVPOps)
        setOperationAction(VPOpc, VT, Custom);
    };

    // Sets common extload/truncstore actions on RVV floating-point vector
    // types.
    const auto SetCommonVFPExtLoadTruncStoreActions =
        [&](MVT VT, ArrayRef<MVT::SimpleValueType> SmallerVTs) {
          for (auto SmallVT : SmallerVTs) {
            setTruncStoreAction(VT, SmallVT, Expand);
            setLoadExtAction(ISD::EXTLOAD, VT, SmallVT, Expand);
          }
        };

    if (Subtarget.hasVInstructionsF16())
      for (MVT VT : F16VecVTs)
        SetCommonVFPActions(VT);

    for (MVT VT : F32VecVTs) {
      if (Subtarget.hasVInstructionsF32())
        SetCommonVFPActions(VT);
      SetCommonVFPExtLoadTruncStoreActions(VT, F16VecVTs);
    }

    for (MVT VT : F64VecVTs) {
      if (Subtarget.hasVInstructionsF64())
        SetCommonVFPActions(VT);
      SetCommonVFPExtLoadTruncStoreActions(VT, F16VecVTs);
      SetCommonVFPExtLoadTruncStoreActions(VT, F32VecVTs);
    }

    if (Subtarget.useRVVForFixedLengthVectors()) {
      for (MVT VT : MVT::integer_fixedlen_vector_valuetypes()) {
        if (!useRVVForFixedLengthVectorVT(VT))
          continue;

        // By default everything must be expanded.
        for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op)
          setOperationAction(Op, VT, Expand);
        for (MVT OtherVT : MVT::integer_fixedlen_vector_valuetypes()) {
          setTruncStoreAction(VT, OtherVT, Expand);
          setLoadExtAction(ISD::EXTLOAD, OtherVT, VT, Expand);
          setLoadExtAction(ISD::SEXTLOAD, OtherVT, VT, Expand);
          setLoadExtAction(ISD::ZEXTLOAD, OtherVT, VT, Expand);
        }

        // We use EXTRACT_SUBVECTOR as a "cast" from scalable to fixed.
        setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
        setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

        setOperationAction(ISD::BUILD_VECTOR, VT, Custom);
        setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);

        setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
        setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);

        setOperationAction(ISD::LOAD, VT, Custom);
        setOperationAction(ISD::STORE, VT, Custom);

        setOperationAction(ISD::SETCC, VT, Custom);

        setOperationAction(ISD::SELECT, VT, Custom);

        setOperationAction(ISD::TRUNCATE, VT, Custom);

        setOperationAction(ISD::BITCAST, VT, Custom);

        setOperationAction(ISD::VECREDUCE_AND, VT, Custom);
        setOperationAction(ISD::VECREDUCE_OR, VT, Custom);
        setOperationAction(ISD::VECREDUCE_XOR, VT, Custom);

        setOperationAction(ISD::VP_REDUCE_AND, VT, Custom);
        setOperationAction(ISD::VP_REDUCE_OR, VT, Custom);
        setOperationAction(ISD::VP_REDUCE_XOR, VT, Custom);

        setOperationAction(ISD::SINT_TO_FP, VT, Custom);
        setOperationAction(ISD::UINT_TO_FP, VT, Custom);
        setOperationAction(ISD::FP_TO_SINT, VT, Custom);
        setOperationAction(ISD::FP_TO_UINT, VT, Custom);

        // Operations below are different for between masks and other vectors.
        if (VT.getVectorElementType() == MVT::i1) {
          setOperationAction(ISD::VP_AND, VT, Custom);
          setOperationAction(ISD::VP_OR, VT, Custom);
          setOperationAction(ISD::VP_XOR, VT, Custom);
          setOperationAction(ISD::AND, VT, Custom);
          setOperationAction(ISD::OR, VT, Custom);
          setOperationAction(ISD::XOR, VT, Custom);
          continue;
        }

        // Use SPLAT_VECTOR to prevent type legalization from destroying the
        // splats when type legalizing i64 scalar on RV32.
        // FIXME: Use SPLAT_VECTOR for all types? DAGCombine probably needs
        // improvements first.
        if (!Subtarget.is64Bit() && VT.getVectorElementType() == MVT::i64) {
          setOperationAction(ISD::SPLAT_VECTOR, VT, Custom);
          setOperationAction(ISD::SPLAT_VECTOR_PARTS, VT, Custom);
        }

        setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
        setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);

        setOperationAction(ISD::MLOAD, VT, Custom);
        setOperationAction(ISD::MSTORE, VT, Custom);
        setOperationAction(ISD::MGATHER, VT, Custom);
        setOperationAction(ISD::MSCATTER, VT, Custom);

        setOperationAction(ISD::VP_LOAD, VT, Custom);
        setOperationAction(ISD::VP_STORE, VT, Custom);
        setOperationAction(ISD::VP_GATHER, VT, Custom);
        setOperationAction(ISD::VP_SCATTER, VT, Custom);

        setOperationAction(ISD::ADD, VT, Custom);
        setOperationAction(ISD::MUL, VT, Custom);
        setOperationAction(ISD::SUB, VT, Custom);
        setOperationAction(ISD::AND, VT, Custom);
        setOperationAction(ISD::OR, VT, Custom);
        setOperationAction(ISD::XOR, VT, Custom);
        setOperationAction(ISD::SDIV, VT, Custom);
        setOperationAction(ISD::SREM, VT, Custom);
        setOperationAction(ISD::UDIV, VT, Custom);
        setOperationAction(ISD::UREM, VT, Custom);
        setOperationAction(ISD::SHL, VT, Custom);
        setOperationAction(ISD::SRA, VT, Custom);
        setOperationAction(ISD::SRL, VT, Custom);

        setOperationAction(ISD::SMIN, VT, Custom);
        setOperationAction(ISD::SMAX, VT, Custom);
        setOperationAction(ISD::UMIN, VT, Custom);
        setOperationAction(ISD::UMAX, VT, Custom);
        setOperationAction(ISD::ABS,  VT, Custom);

        // vXi64 MULHS/MULHU requires the V extension instead of Zve64*.
        if (VT.getVectorElementType() != MVT::i64 || Subtarget.hasStdExtV()) {
          setOperationAction(ISD::MULHS, VT, Custom);
          setOperationAction(ISD::MULHU, VT, Custom);
        }

        setOperationAction(ISD::SADDSAT, VT, Custom);
        setOperationAction(ISD::UADDSAT, VT, Custom);
        setOperationAction(ISD::SSUBSAT, VT, Custom);
        setOperationAction(ISD::USUBSAT, VT, Custom);

        setOperationAction(ISD::VSELECT, VT, Custom);
        setOperationAction(ISD::SELECT_CC, VT, Expand);

        setOperationAction(ISD::ANY_EXTEND, VT, Custom);
        setOperationAction(ISD::SIGN_EXTEND, VT, Custom);
        setOperationAction(ISD::ZERO_EXTEND, VT, Custom);

        // Custom-lower reduction operations to set up the corresponding custom
        // nodes' operands.
        setOperationAction(ISD::VECREDUCE_ADD, VT, Custom);
        setOperationAction(ISD::VECREDUCE_SMAX, VT, Custom);
        setOperationAction(ISD::VECREDUCE_SMIN, VT, Custom);
        setOperationAction(ISD::VECREDUCE_UMAX, VT, Custom);
        setOperationAction(ISD::VECREDUCE_UMIN, VT, Custom);

        for (unsigned VPOpc : IntegerVPOps)
          setOperationAction(VPOpc, VT, Custom);

        // Lower CTLZ_ZERO_UNDEF and CTTZ_ZERO_UNDEF if we have a floating point
        // type that can represent the value exactly.
        if (VT.getVectorElementType() != MVT::i64) {
          MVT FloatEltVT =
              VT.getVectorElementType() == MVT::i32 ? MVT::f64 : MVT::f32;
          EVT FloatVT =
              MVT::getVectorVT(FloatEltVT, VT.getVectorElementCount());
          if (isTypeLegal(FloatVT)) {
            setOperationAction(ISD::CTLZ_ZERO_UNDEF, VT, Custom);
            setOperationAction(ISD::CTTZ_ZERO_UNDEF, VT, Custom);
          }
        }
      }

      for (MVT VT : MVT::fp_fixedlen_vector_valuetypes()) {
        if (!useRVVForFixedLengthVectorVT(VT))
          continue;

        // By default everything must be expanded.
        for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op)
          setOperationAction(Op, VT, Expand);
        for (MVT OtherVT : MVT::fp_fixedlen_vector_valuetypes()) {
          setLoadExtAction(ISD::EXTLOAD, OtherVT, VT, Expand);
          setTruncStoreAction(VT, OtherVT, Expand);
        }

        // We use EXTRACT_SUBVECTOR as a "cast" from scalable to fixed.
        setOperationAction(ISD::INSERT_SUBVECTOR, VT, Custom);
        setOperationAction(ISD::EXTRACT_SUBVECTOR, VT, Custom);

        setOperationAction(ISD::BUILD_VECTOR, VT, Custom);
        setOperationAction(ISD::CONCAT_VECTORS, VT, Custom);
        setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
        setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
        setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);

        setOperationAction(ISD::LOAD, VT, Custom);
        setOperationAction(ISD::STORE, VT, Custom);
        setOperationAction(ISD::MLOAD, VT, Custom);
        setOperationAction(ISD::MSTORE, VT, Custom);
        setOperationAction(ISD::MGATHER, VT, Custom);
        setOperationAction(ISD::MSCATTER, VT, Custom);

        setOperationAction(ISD::VP_LOAD, VT, Custom);
        setOperationAction(ISD::VP_STORE, VT, Custom);
        setOperationAction(ISD::VP_GATHER, VT, Custom);
        setOperationAction(ISD::VP_SCATTER, VT, Custom);

        setOperationAction(ISD::FADD, VT, Custom);
        setOperationAction(ISD::FSUB, VT, Custom);
        setOperationAction(ISD::FMUL, VT, Custom);
        setOperationAction(ISD::FDIV, VT, Custom);
        setOperationAction(ISD::FNEG, VT, Custom);
        setOperationAction(ISD::FABS, VT, Custom);
        setOperationAction(ISD::FCOPYSIGN, VT, Custom);
        setOperationAction(ISD::FSQRT, VT, Custom);
        setOperationAction(ISD::FMA, VT, Custom);
        setOperationAction(ISD::FMINNUM, VT, Custom);
        setOperationAction(ISD::FMAXNUM, VT, Custom);

        setOperationAction(ISD::FP_ROUND, VT, Custom);
        setOperationAction(ISD::FP_EXTEND, VT, Custom);

        setOperationAction(ISD::FTRUNC, VT, Custom);
        setOperationAction(ISD::FCEIL, VT, Custom);
        setOperationAction(ISD::FFLOOR, VT, Custom);
        setOperationAction(ISD::FROUND, VT, Custom);

        for (auto CC : VFPCCToExpand)
          setCondCodeAction(CC, VT, Expand);

        setOperationAction(ISD::VSELECT, VT, Custom);
        setOperationAction(ISD::SELECT, VT, Custom);
        setOperationAction(ISD::SELECT_CC, VT, Expand);

        setOperationAction(ISD::BITCAST, VT, Custom);

        setOperationAction(ISD::VECREDUCE_FADD, VT, Custom);
        setOperationAction(ISD::VECREDUCE_SEQ_FADD, VT, Custom);
        setOperationAction(ISD::VECREDUCE_FMIN, VT, Custom);
        setOperationAction(ISD::VECREDUCE_FMAX, VT, Custom);

        for (unsigned VPOpc : FloatingPointVPOps)
          setOperationAction(VPOpc, VT, Custom);
      }

      // Custom-legalize bitcasts from fixed-length vectors to scalar types.
      setOperationAction(ISD::BITCAST, MVT::i8, Custom);
      setOperationAction(ISD::BITCAST, MVT::i16, Custom);
      setOperationAction(ISD::BITCAST, MVT::i32, Custom);
      setOperationAction(ISD::BITCAST, MVT::i64, Custom);
      if (Subtarget.hasStdExtZfh())
        setOperationAction(ISD::BITCAST, MVT::f16, Custom);
      if (Subtarget.hasStdExtF())
        setOperationAction(ISD::BITCAST, MVT::f32, Custom);
      if (Subtarget.hasStdExtD())
        setOperationAction(ISD::BITCAST, MVT::f64, Custom);
    }
  }

  // Function alignments.
  const Align FunctionAlignment(Subtarget.hasStdExtC() ? 2 : 4);
  setMinFunctionAlignment(FunctionAlignment);
  setPrefFunctionAlignment(FunctionAlignment);

  setMinimumJumpTableEntries(5);

  // Jumps are expensive, compared to logic
  setJumpIsExpensive();

  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::SUB);
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::XOR);
  if (Subtarget.hasStdExtZbp()) {
    setTargetDAGCombine(ISD::ROTL);
    setTargetDAGCombine(ISD::ROTR);
  }
  if (Subtarget.hasStdExtZbkb())
    setTargetDAGCombine(ISD::BITREVERSE);
  setTargetDAGCombine(ISD::INTRINSIC_WO_CHAIN);
  if (Subtarget.hasStdExtZfh() || Subtarget.hasStdExtZbb())
    setTargetDAGCombine(ISD::SIGN_EXTEND_INREG);
  if (Subtarget.hasStdExtF()) {
    setTargetDAGCombine(ISD::ZERO_EXTEND);
    setTargetDAGCombine(ISD::FP_TO_SINT);
    setTargetDAGCombine(ISD::FP_TO_UINT);
    setTargetDAGCombine(ISD::FP_TO_SINT_SAT);
    setTargetDAGCombine(ISD::FP_TO_UINT_SAT);
  }
  if (Subtarget.hasVInstructions()) {
    setTargetDAGCombine(ISD::FCOPYSIGN);
    setTargetDAGCombine(ISD::MGATHER);
    setTargetDAGCombine(ISD::MSCATTER);
    setTargetDAGCombine(ISD::VP_GATHER);
    setTargetDAGCombine(ISD::VP_SCATTER);
    setTargetDAGCombine(ISD::SRA);
    setTargetDAGCombine(ISD::SRL);
    setTargetDAGCombine(ISD::SHL);
    setTargetDAGCombine(ISD::STORE);
    setTargetDAGCombine(ISD::SPLAT_VECTOR);
  }

  setLibcallName(RTLIB::FPEXT_F16_F32, "__extendhfsf2");
  setLibcallName(RTLIB::FPROUND_F32_F16, "__truncsfhf2");
}

EVT RISCVTargetLowering::getSetCCResultType(const DataLayout &DL,
                                            LLVMContext &Context,
                                            EVT VT) const {
  if (!VT.isVector())
    return getPointerTy(DL);
  if (Subtarget.hasVInstructions() &&
      (VT.isScalableVector() || Subtarget.useRVVForFixedLengthVectors()))
    return EVT::getVectorVT(Context, MVT::i1, VT.getVectorElementCount());
  return VT.changeVectorElementTypeToInteger();
}

MVT RISCVTargetLowering::getVPExplicitVectorLengthTy() const {
  return Subtarget.getXLenVT();
}

bool RISCVTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                             const CallInst &I,
                                             MachineFunction &MF,
                                             unsigned Intrinsic) const {
  auto &DL = I.getModule()->getDataLayout();
  switch (Intrinsic) {
  default:
    return false;
  case Intrinsic::riscv_masked_atomicrmw_xchg_i32:
  case Intrinsic::riscv_masked_atomicrmw_add_i32:
  case Intrinsic::riscv_masked_atomicrmw_sub_i32:
  case Intrinsic::riscv_masked_atomicrmw_nand_i32:
  case Intrinsic::riscv_masked_atomicrmw_max_i32:
  case Intrinsic::riscv_masked_atomicrmw_min_i32:
  case Intrinsic::riscv_masked_atomicrmw_umax_i32:
  case Intrinsic::riscv_masked_atomicrmw_umin_i32:
  case Intrinsic::riscv_masked_cmpxchg_i32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Align(4);
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore |
                 MachineMemOperand::MOVolatile;
    return true;
  case Intrinsic::riscv_masked_strided_load:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = getValueType(DL, I.getType()->getScalarType());
    Info.align = Align(DL.getTypeSizeInBits(I.getType()->getScalarType()) / 8);
    Info.size = MemoryLocation::UnknownSize;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  case Intrinsic::riscv_masked_strided_store:
    Info.opc = ISD::INTRINSIC_VOID;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT =
        getValueType(DL, I.getArgOperand(0)->getType()->getScalarType());
    Info.align = Align(
        DL.getTypeSizeInBits(I.getArgOperand(0)->getType()->getScalarType()) /
        8);
    Info.size = MemoryLocation::UnknownSize;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  case Intrinsic::riscv_seg2_load:
  case Intrinsic::riscv_seg3_load:
  case Intrinsic::riscv_seg4_load:
  case Intrinsic::riscv_seg5_load:
  case Intrinsic::riscv_seg6_load:
  case Intrinsic::riscv_seg7_load:
  case Intrinsic::riscv_seg8_load:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT =
        getValueType(DL, I.getType()->getStructElementType(0)->getScalarType());
    Info.align =
        Align(DL.getTypeSizeInBits(
                  I.getType()->getStructElementType(0)->getScalarType()) /
              8);
    Info.size = MemoryLocation::UnknownSize;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
}

bool RISCVTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                const AddrMode &AM, Type *Ty,
                                                unsigned AS,
                                                Instruction *I) const {
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // Require a 12-bit signed offset.
  if (!isInt<12>(AM.BaseOffs))
    return false;

  switch (AM.Scale) {
  case 0: // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (!AM.HasBaseReg) // allow "r+i".
      break;
    return false; // disallow "r+r" or "r+r+i".
  default:
    return false;
  }

  return true;
}

bool RISCVTargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

bool RISCVTargetLowering::isLegalAddImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

// On RV32, 64-bit integers are split into their high and low parts and held
// in two different registers, so the trunc is free since the low register can
// just be used.
bool RISCVTargetLowering::isTruncateFree(Type *SrcTy, Type *DstTy) const {
  if (Subtarget.is64Bit() || !SrcTy->isIntegerTy() || !DstTy->isIntegerTy())
    return false;
  unsigned SrcBits = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBits = DstTy->getPrimitiveSizeInBits();
  return (SrcBits == 64 && DestBits == 32);
}

bool RISCVTargetLowering::isTruncateFree(EVT SrcVT, EVT DstVT) const {
  if (Subtarget.is64Bit() || SrcVT.isVector() || DstVT.isVector() ||
      !SrcVT.isInteger() || !DstVT.isInteger())
    return false;
  unsigned SrcBits = SrcVT.getSizeInBits();
  unsigned DestBits = DstVT.getSizeInBits();
  return (SrcBits == 64 && DestBits == 32);
}

bool RISCVTargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  // Zexts are free if they can be combined with a load.
  // Don't advertise i32->i64 zextload as being free for RV64. It interacts
  // poorly with type legalization of compares preferring sext.
  if (auto *LD = dyn_cast<LoadSDNode>(Val)) {
    EVT MemVT = LD->getMemoryVT();
    if ((MemVT == MVT::i8 || MemVT == MVT::i16) &&
        (LD->getExtensionType() == ISD::NON_EXTLOAD ||
         LD->getExtensionType() == ISD::ZEXTLOAD))
      return true;
  }

  return TargetLowering::isZExtFree(Val, VT2);
}

bool RISCVTargetLowering::isSExtCheaperThanZExt(EVT SrcVT, EVT DstVT) const {
  return Subtarget.is64Bit() && SrcVT == MVT::i32 && DstVT == MVT::i64;
}

bool RISCVTargetLowering::isCheapToSpeculateCttz() const {
  return Subtarget.hasStdExtZbb();
}

bool RISCVTargetLowering::isCheapToSpeculateCtlz() const {
  return Subtarget.hasStdExtZbb();
}

bool RISCVTargetLowering::hasAndNotCompare(SDValue Y) const {
  EVT VT = Y.getValueType();

  // FIXME: Support vectors once we have tests.
  if (VT.isVector())
    return false;

  return (Subtarget.hasStdExtZbb() || Subtarget.hasStdExtZbp() ||
          Subtarget.hasStdExtZbkb()) &&
         !isa<ConstantSDNode>(Y);
}

/// Check if sinking \p I's operands to I's basic block is profitable, because
/// the operands can be folded into a target instruction, e.g.
/// splats of scalars can fold into vector instructions.
bool RISCVTargetLowering::shouldSinkOperands(
    Instruction *I, SmallVectorImpl<Use *> &Ops) const {
  using namespace llvm::PatternMatch;

  if (!I->getType()->isVectorTy() || !Subtarget.hasVInstructions())
    return false;

  auto IsSinker = [&](Instruction *I, int Operand) {
    switch (I->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
    case Instruction::FDiv:
    case Instruction::ICmp:
    case Instruction::FCmp:
      return true;
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      return Operand == 1;
    case Instruction::Call:
      if (auto *II = dyn_cast<IntrinsicInst>(I)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::fma:
        case Intrinsic::vp_fma:
          return Operand == 0 || Operand == 1;
        // FIXME: Our patterns can only match vx/vf instructions when the splat
        // it on the RHS, because TableGen doesn't recognize our VP operations
        // as commutative.
        case Intrinsic::vp_add:
        case Intrinsic::vp_mul:
        case Intrinsic::vp_and:
        case Intrinsic::vp_or:
        case Intrinsic::vp_xor:
        case Intrinsic::vp_fadd:
        case Intrinsic::vp_fmul:
        case Intrinsic::vp_shl:
        case Intrinsic::vp_lshr:
        case Intrinsic::vp_ashr:
        case Intrinsic::vp_udiv:
        case Intrinsic::vp_sdiv:
        case Intrinsic::vp_urem:
        case Intrinsic::vp_srem:
          return Operand == 1;
        // ... with the exception of vp.sub/vp.fsub/vp.fdiv, which have
        // explicit patterns for both LHS and RHS (as 'vr' versions).
        case Intrinsic::vp_sub:
        case Intrinsic::vp_fsub:
        case Intrinsic::vp_fdiv:
          return Operand == 0 || Operand == 1;
        default:
          return false;
        }
      }
      return false;
    default:
      return false;
    }
  };

  for (auto OpIdx : enumerate(I->operands())) {
    if (!IsSinker(I, OpIdx.index()))
      continue;

    Instruction *Op = dyn_cast<Instruction>(OpIdx.value().get());
    // Make sure we are not already sinking this operand
    if (!Op || any_of(Ops, [&](Use *U) { return U->get() == Op; }))
      continue;

    // We are looking for a splat that can be sunk.
    if (!match(Op, m_Shuffle(m_InsertElt(m_Undef(), m_Value(), m_ZeroInt()),
                             m_Undef(), m_ZeroMask())))
      continue;

    // All uses of the shuffle should be sunk to avoid duplicating it across gpr
    // and vector registers
    for (Use &U : Op->uses()) {
      Instruction *Insn = cast<Instruction>(U.getUser());
      if (!IsSinker(Insn, U.getOperandNo()))
        return false;
    }

    Ops.push_back(&Op->getOperandUse(0));
    Ops.push_back(&OpIdx.value());
  }
  return true;
}

bool RISCVTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                       bool ForCodeSize) const {
  // FIXME: Change to Zfhmin once f16 becomes a legal type with Zfhmin.
  if (VT == MVT::f16 && !Subtarget.hasStdExtZfh())
    return false;
  if (VT == MVT::f32 && !Subtarget.hasStdExtF())
    return false;
  if (VT == MVT::f64 && !Subtarget.hasStdExtD())
    return false;
  return Imm.isZero();
}

bool RISCVTargetLowering::hasBitPreservingFPLogic(EVT VT) const {
  return (VT == MVT::f16 && Subtarget.hasStdExtZfh()) ||
         (VT == MVT::f32 && Subtarget.hasStdExtF()) ||
         (VT == MVT::f64 && Subtarget.hasStdExtD());
}

MVT RISCVTargetLowering::getRegisterTypeForCallingConv(LLVMContext &Context,
                                                      CallingConv::ID CC,
                                                      EVT VT) const {
  // Use f32 to pass f16 if it is legal and Zfh is not enabled.
  // We might still end up using a GPR but that will be decided based on ABI.
  // FIXME: Change to Zfhmin once f16 becomes a legal type with Zfhmin.
  if (VT == MVT::f16 && Subtarget.hasStdExtF() && !Subtarget.hasStdExtZfh())
    return MVT::f32;

  return TargetLowering::getRegisterTypeForCallingConv(Context, CC, VT);
}

unsigned RISCVTargetLowering::getNumRegistersForCallingConv(LLVMContext &Context,
                                                           CallingConv::ID CC,
                                                           EVT VT) const {
  // Use f32 to pass f16 if it is legal and Zfh is not enabled.
  // We might still end up using a GPR but that will be decided based on ABI.
  // FIXME: Change to Zfhmin once f16 becomes a legal type with Zfhmin.
  if (VT == MVT::f16 && Subtarget.hasStdExtF() && !Subtarget.hasStdExtZfh())
    return 1;

  return TargetLowering::getNumRegistersForCallingConv(Context, CC, VT);
}

// Changes the condition code and swaps operands if necessary, so the SetCC
// operation matches one of the comparisons supported directly by branches
// in the RISC-V ISA. May adjust compares to favor compare with 0 over compare
// with 1/-1.
static void translateSetCCForBranch(const SDLoc &DL, SDValue &LHS, SDValue &RHS,
                                    ISD::CondCode &CC, SelectionDAG &DAG) {
  // Convert X > -1 to X >= 0.
  if (CC == ISD::SETGT && isAllOnesConstant(RHS)) {
    RHS = DAG.getConstant(0, DL, RHS.getValueType());
    CC = ISD::SETGE;
    return;
  }
  // Convert X < 1 to 0 >= X.
  if (CC == ISD::SETLT && isOneConstant(RHS)) {
    RHS = LHS;
    LHS = DAG.getConstant(0, DL, RHS.getValueType());
    CC = ISD::SETGE;
    return;
  }

  switch (CC) {
  default:
    break;
  case ISD::SETGT:
  case ISD::SETLE:
  case ISD::SETUGT:
  case ISD::SETULE:
    CC = ISD::getSetCCSwappedOperands(CC);
    std::swap(LHS, RHS);
    break;
  }
}

RISCVII::VLMUL RISCVTargetLowering::getLMUL(MVT VT) {
  assert(VT.isScalableVector() && "Expecting a scalable vector type");
  unsigned KnownSize = VT.getSizeInBits().getKnownMinValue();
  if (VT.getVectorElementType() == MVT::i1)
    KnownSize *= 8;

  switch (KnownSize) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case 8:
    return RISCVII::VLMUL::LMUL_F8;
  case 16:
    return RISCVII::VLMUL::LMUL_F4;
  case 32:
    return RISCVII::VLMUL::LMUL_F2;
  case 64:
    return RISCVII::VLMUL::LMUL_1;
  case 128:
    return RISCVII::VLMUL::LMUL_2;
  case 256:
    return RISCVII::VLMUL::LMUL_4;
  case 512:
    return RISCVII::VLMUL::LMUL_8;
  }
}

unsigned RISCVTargetLowering::getRegClassIDForLMUL(RISCVII::VLMUL LMul) {
  switch (LMul) {
  default:
    llvm_unreachable("Invalid LMUL.");
  case RISCVII::VLMUL::LMUL_F8:
  case RISCVII::VLMUL::LMUL_F4:
  case RISCVII::VLMUL::LMUL_F2:
  case RISCVII::VLMUL::LMUL_1:
    return RISCV::VRRegClassID;
  case RISCVII::VLMUL::LMUL_2:
    return RISCV::VRM2RegClassID;
  case RISCVII::VLMUL::LMUL_4:
    return RISCV::VRM4RegClassID;
  case RISCVII::VLMUL::LMUL_8:
    return RISCV::VRM8RegClassID;
  }
}

unsigned RISCVTargetLowering::getSubregIndexByMVT(MVT VT, unsigned Index) {
  RISCVII::VLMUL LMUL = getLMUL(VT);
  if (LMUL == RISCVII::VLMUL::LMUL_F8 ||
      LMUL == RISCVII::VLMUL::LMUL_F4 ||
      LMUL == RISCVII::VLMUL::LMUL_F2 ||
      LMUL == RISCVII::VLMUL::LMUL_1) {
    static_assert(RISCV::sub_vrm1_7 == RISCV::sub_vrm1_0 + 7,
                  "Unexpected subreg numbering");
    return RISCV::sub_vrm1_0 + Index;
  }
  if (LMUL == RISCVII::VLMUL::LMUL_2) {
    static_assert(RISCV::sub_vrm2_3 == RISCV::sub_vrm2_0 + 3,
                  "Unexpected subreg numbering");
    return RISCV::sub_vrm2_0 + Index;
  }
  if (LMUL == RISCVII::VLMUL::LMUL_4) {
    static_assert(RISCV::sub_vrm4_1 == RISCV::sub_vrm4_0 + 1,
                  "Unexpected subreg numbering");
    return RISCV::sub_vrm4_0 + Index;
  }
  llvm_unreachable("Invalid vector type.");
}

unsigned RISCVTargetLowering::getRegClassIDForVecVT(MVT VT) {
  if (VT.getVectorElementType() == MVT::i1)
    return RISCV::VRRegClassID;
  return getRegClassIDForLMUL(getLMUL(VT));
}

// Attempt to decompose a subvector insert/extract between VecVT and
// SubVecVT via subregister indices. Returns the subregister index that
// can perform the subvector insert/extract with the given element index, as
// well as the index corresponding to any leftover subvectors that must be
// further inserted/extracted within the register class for SubVecVT.
std::pair<unsigned, unsigned>
RISCVTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
    MVT VecVT, MVT SubVecVT, unsigned InsertExtractIdx,
    const RISCVRegisterInfo *TRI) {
  static_assert((RISCV::VRM8RegClassID > RISCV::VRM4RegClassID &&
                 RISCV::VRM4RegClassID > RISCV::VRM2RegClassID &&
                 RISCV::VRM2RegClassID > RISCV::VRRegClassID),
                "Register classes not ordered");
  unsigned VecRegClassID = getRegClassIDForVecVT(VecVT);
  unsigned SubRegClassID = getRegClassIDForVecVT(SubVecVT);
  // Try to compose a subregister index that takes us from the incoming
  // LMUL>1 register class down to the outgoing one. At each step we half
  // the LMUL:
  //   nxv16i32@12 -> nxv2i32: sub_vrm4_1_then_sub_vrm2_1_then_sub_vrm1_0
  // Note that this is not guaranteed to find a subregister index, such as
  // when we are extracting from one VR type to another.
  unsigned SubRegIdx = RISCV::NoSubRegister;
  for (const unsigned RCID :
       {RISCV::VRM4RegClassID, RISCV::VRM2RegClassID, RISCV::VRRegClassID})
    if (VecRegClassID > RCID && SubRegClassID <= RCID) {
      VecVT = VecVT.getHalfNumVectorElementsVT();
      bool IsHi =
          InsertExtractIdx >= VecVT.getVectorElementCount().getKnownMinValue();
      SubRegIdx = TRI->composeSubRegIndices(SubRegIdx,
                                            getSubregIndexByMVT(VecVT, IsHi));
      if (IsHi)
        InsertExtractIdx -= VecVT.getVectorElementCount().getKnownMinValue();
    }
  return {SubRegIdx, InsertExtractIdx};
}

// Permit combining of mask vectors as BUILD_VECTOR never expands to scalar
// stores for those types.
bool RISCVTargetLowering::mergeStoresAfterLegalization(EVT VT) const {
  return !Subtarget.useRVVForFixedLengthVectors() ||
         (VT.isFixedLengthVector() && VT.getVectorElementType() == MVT::i1);
}

bool RISCVTargetLowering::isLegalElementTypeForRVV(Type *ScalarTy) const {
  if (ScalarTy->isPointerTy())
    return true;

  if (ScalarTy->isIntegerTy(8) || ScalarTy->isIntegerTy(16) ||
      ScalarTy->isIntegerTy(32))
    return true;

  if (ScalarTy->isIntegerTy(64))
    return Subtarget.hasVInstructionsI64();

  if (ScalarTy->isHalfTy())
    return Subtarget.hasVInstructionsF16();
  if (ScalarTy->isFloatTy())
    return Subtarget.hasVInstructionsF32();
  if (ScalarTy->isDoubleTy())
    return Subtarget.hasVInstructionsF64();

  return false;
}

static SDValue getVLOperand(SDValue Op) {
  assert((Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN) &&
         "Unexpected opcode");
  bool HasChain = Op.getOpcode() == ISD::INTRINSIC_W_CHAIN;
  unsigned IntNo = Op.getConstantOperandVal(HasChain ? 1 : 0);
  const RISCVVIntrinsicsTable::RISCVVIntrinsicInfo *II =
      RISCVVIntrinsicsTable::getRISCVVIntrinsicInfo(IntNo);
  if (!II)
    return SDValue();
  return Op.getOperand(II->VLOperand + 1 + HasChain);
}

static bool useRVVForFixedLengthVectorVT(MVT VT,
                                         const RISCVSubtarget &Subtarget) {
  assert(VT.isFixedLengthVector() && "Expected a fixed length vector type!");
  if (!Subtarget.useRVVForFixedLengthVectors())
    return false;

  // We only support a set of vector types with a consistent maximum fixed size
  // across all supported vector element types to avoid legalization issues.
  // Therefore -- since the largest is v1024i8/v512i16/etc -- the largest
  // fixed-length vector type we support is 1024 bytes.
  if (VT.getFixedSizeInBits() > 1024 * 8)
    return false;

  unsigned MinVLen = Subtarget.getMinRVVVectorSizeInBits();

  MVT EltVT = VT.getVectorElementType();

  // Don't use RVV for vectors we cannot scalarize if required.
  switch (EltVT.SimpleTy) {
  // i1 is supported but has different rules.
  default:
    return false;
  case MVT::i1:
    // Masks can only use a single register.
    if (VT.getVectorNumElements() > MinVLen)
      return false;
    MinVLen /= 8;
    break;
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    break;
  case MVT::i64:
    if (!Subtarget.hasVInstructionsI64())
      return false;
    break;
  case MVT::f16:
    if (!Subtarget.hasVInstructionsF16())
      return false;
    break;
  case MVT::f32:
    if (!Subtarget.hasVInstructionsF32())
      return false;
    break;
  case MVT::f64:
    if (!Subtarget.hasVInstructionsF64())
      return false;
    break;
  }

  // Reject elements larger than ELEN.
  if (EltVT.getSizeInBits() > Subtarget.getMaxELENForFixedLengthVectors())
    return false;

  unsigned LMul = divideCeil(VT.getSizeInBits(), MinVLen);
  // Don't use RVV for types that don't fit.
  if (LMul > Subtarget.getMaxLMULForFixedLengthVectors())
    return false;

  // TODO: Perhaps an artificial restriction, but worth having whilst getting
  // the base fixed length RVV support in place.
  if (!VT.isPow2VectorType())
    return false;

  return true;
}

bool RISCVTargetLowering::useRVVForFixedLengthVectorVT(MVT VT) const {
  return ::useRVVForFixedLengthVectorVT(VT, Subtarget);
}

// Return the largest legal scalable vector type that matches VT's element type.
static MVT getContainerForFixedLengthVector(const TargetLowering &TLI, MVT VT,
                                            const RISCVSubtarget &Subtarget) {
  // This may be called before legal types are setup.
  assert(((VT.isFixedLengthVector() && TLI.isTypeLegal(VT)) ||
          useRVVForFixedLengthVectorVT(VT, Subtarget)) &&
         "Expected legal fixed length vector!");

  unsigned MinVLen = Subtarget.getMinRVVVectorSizeInBits();
  unsigned MaxELen = Subtarget.getMaxELENForFixedLengthVectors();

  MVT EltVT = VT.getVectorElementType();
  switch (EltVT.SimpleTy) {
  default:
    llvm_unreachable("unexpected element type for RVV container");
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
  case MVT::f16:
  case MVT::f32:
  case MVT::f64: {
    // We prefer to use LMUL=1 for VLEN sized types. Use fractional lmuls for
    // narrower types. The smallest fractional LMUL we support is 8/ELEN. Within
    // each fractional LMUL we support SEW between 8 and LMUL*ELEN.
    unsigned NumElts =
        (VT.getVectorNumElements() * RISCV::RVVBitsPerBlock) / MinVLen;
    NumElts = std::max(NumElts, RISCV::RVVBitsPerBlock / MaxELen);
    assert(isPowerOf2_32(NumElts) && "Expected power of 2 NumElts");
    return MVT::getScalableVectorVT(EltVT, NumElts);
  }
  }
}

static MVT getContainerForFixedLengthVector(SelectionDAG &DAG, MVT VT,
                                            const RISCVSubtarget &Subtarget) {
  return getContainerForFixedLengthVector(DAG.getTargetLoweringInfo(), VT,
                                          Subtarget);
}

MVT RISCVTargetLowering::getContainerForFixedLengthVector(MVT VT) const {
  return ::getContainerForFixedLengthVector(*this, VT, getSubtarget());
}

// Grow V to consume an entire RVV register.
static SDValue convertToScalableVector(EVT VT, SDValue V, SelectionDAG &DAG,
                                       const RISCVSubtarget &Subtarget) {
  assert(VT.isScalableVector() &&
         "Expected to convert into a scalable vector!");
  assert(V.getValueType().isFixedLengthVector() &&
         "Expected a fixed length vector operand!");
  SDLoc DL(V);
  SDValue Zero = DAG.getConstant(0, DL, Subtarget.getXLenVT());
  return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, DAG.getUNDEF(VT), V, Zero);
}

// Shrink V so it's just big enough to maintain a VT's worth of data.
static SDValue convertFromScalableVector(EVT VT, SDValue V, SelectionDAG &DAG,
                                         const RISCVSubtarget &Subtarget) {
  assert(VT.isFixedLengthVector() &&
         "Expected to convert into a fixed length vector!");
  assert(V.getValueType().isScalableVector() &&
         "Expected a scalable vector operand!");
  SDLoc DL(V);
  SDValue Zero = DAG.getConstant(0, DL, Subtarget.getXLenVT());
  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, V, Zero);
}

// Gets the two common "VL" operands: an all-ones mask and the vector length.
// VecVT is a vector type, either fixed-length or scalable, and ContainerVT is
// the vector type that it is contained in.
static std::pair<SDValue, SDValue>
getDefaultVLOps(MVT VecVT, MVT ContainerVT, SDLoc DL, SelectionDAG &DAG,
                const RISCVSubtarget &Subtarget) {
  assert(ContainerVT.isScalableVector() && "Expecting scalable container type");
  MVT XLenVT = Subtarget.getXLenVT();
  SDValue VL = VecVT.isFixedLengthVector()
                   ? DAG.getConstant(VecVT.getVectorNumElements(), DL, XLenVT)
                   : DAG.getRegister(RISCV::X0, XLenVT);
  MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
  SDValue Mask = DAG.getNode(RISCVISD::VMSET_VL, DL, MaskVT, VL);
  return {Mask, VL};
}

// As above but assuming the given type is a scalable vector type.
static std::pair<SDValue, SDValue>
getDefaultScalableVLOps(MVT VecVT, SDLoc DL, SelectionDAG &DAG,
                        const RISCVSubtarget &Subtarget) {
  assert(VecVT.isScalableVector() && "Expecting a scalable vector");
  return getDefaultVLOps(VecVT, VecVT, DL, DAG, Subtarget);
}

// The state of RVV BUILD_VECTOR and VECTOR_SHUFFLE lowering is that very few
// of either is (currently) supported. This can get us into an infinite loop
// where we try to lower a BUILD_VECTOR as a VECTOR_SHUFFLE as a BUILD_VECTOR
// as a ..., etc.
// Until either (or both) of these can reliably lower any node, reporting that
// we don't want to expand BUILD_VECTORs via VECTOR_SHUFFLEs at least breaks
// the infinite loop. Note that this lowers BUILD_VECTOR through the stack,
// which is not desirable.
bool RISCVTargetLowering::shouldExpandBuildVectorWithShuffles(
    EVT VT, unsigned DefinedValues) const {
  return false;
}

static SDValue lowerFP_TO_INT_SAT(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget) {
  // RISCV FP-to-int conversions saturate to the destination register size, but
  // don't produce 0 for nan. We can use a conversion instruction and fix the
  // nan case with a compare and a select.
  SDValue Src = Op.getOperand(0);

  EVT DstVT = Op.getValueType();
  EVT SatVT = cast<VTSDNode>(Op.getOperand(1))->getVT();

  bool IsSigned = Op.getOpcode() == ISD::FP_TO_SINT_SAT;
  unsigned Opc;
  if (SatVT == DstVT)
    Opc = IsSigned ? RISCVISD::FCVT_X : RISCVISD::FCVT_XU;
  else if (DstVT == MVT::i64 && SatVT == MVT::i32)
    Opc = IsSigned ? RISCVISD::FCVT_W_RV64 : RISCVISD::FCVT_WU_RV64;
  else
    return SDValue();
  // FIXME: Support other SatVTs by clamping before or after the conversion.

  SDLoc DL(Op);
  SDValue FpToInt = DAG.getNode(
      Opc, DL, DstVT, Src,
      DAG.getTargetConstant(RISCVFPRndMode::RTZ, DL, Subtarget.getXLenVT()));

  SDValue ZeroInt = DAG.getConstant(0, DL, DstVT);
  return DAG.getSelectCC(DL, Src, Src, ZeroInt, FpToInt, ISD::CondCode::SETUO);
}

// Expand vector FTRUNC, FCEIL, and FFLOOR by converting to the integer domain
// and back. Taking care to avoid converting values that are nan or already
// correct.
// TODO: Floor and ceil could be shorter by changing rounding mode, but we don't
// have FRM dependencies modeled yet.
static SDValue lowerFTRUNC_FCEIL_FFLOOR(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getSimpleValueType();
  assert(VT.isVector() && "Unexpected type");

  SDLoc DL(Op);

  // Freeze the source since we are increasing the number of uses.
  SDValue Src = DAG.getFreeze(Op.getOperand(0));

  // Truncate to integer and convert back to FP.
  MVT IntVT = VT.changeVectorElementTypeToInteger();
  SDValue Truncated = DAG.getNode(ISD::FP_TO_SINT, DL, IntVT, Src);
  Truncated = DAG.getNode(ISD::SINT_TO_FP, DL, VT, Truncated);

  MVT SetccVT = MVT::getVectorVT(MVT::i1, VT.getVectorElementCount());

  if (Op.getOpcode() == ISD::FCEIL) {
    // If the truncated value is the greater than or equal to the original
    // value, we've computed the ceil. Otherwise, we went the wrong way and
    // need to increase by 1.
    // FIXME: This should use a masked operation. Handle here or in isel?
    SDValue Adjust = DAG.getNode(ISD::FADD, DL, VT, Truncated,
                                 DAG.getConstantFP(1.0, DL, VT));
    SDValue NeedAdjust = DAG.getSetCC(DL, SetccVT, Truncated, Src, ISD::SETOLT);
    Truncated = DAG.getSelect(DL, VT, NeedAdjust, Adjust, Truncated);
  } else if (Op.getOpcode() == ISD::FFLOOR) {
    // If the truncated value is the less than or equal to the original value,
    // we've computed the floor. Otherwise, we went the wrong way and need to
    // decrease by 1.
    // FIXME: This should use a masked operation. Handle here or in isel?
    SDValue Adjust = DAG.getNode(ISD::FSUB, DL, VT, Truncated,
                                 DAG.getConstantFP(1.0, DL, VT));
    SDValue NeedAdjust = DAG.getSetCC(DL, SetccVT, Truncated, Src, ISD::SETOGT);
    Truncated = DAG.getSelect(DL, VT, NeedAdjust, Adjust, Truncated);
  }

  // Restore the original sign so that -0.0 is preserved.
  Truncated = DAG.getNode(ISD::FCOPYSIGN, DL, VT, Truncated, Src);

  // Determine the largest integer that can be represented exactly. This and
  // values larger than it don't have any fractional bits so don't need to
  // be converted.
  const fltSemantics &FltSem = DAG.EVTToAPFloatSemantics(VT);
  unsigned Precision = APFloat::semanticsPrecision(FltSem);
  APFloat MaxVal = APFloat(FltSem);
  MaxVal.convertFromAPInt(APInt::getOneBitSet(Precision, Precision - 1),
                          /*IsSigned*/ false, APFloat::rmNearestTiesToEven);
  SDValue MaxValNode = DAG.getConstantFP(MaxVal, DL, VT);

  // If abs(Src) was larger than MaxVal or nan, keep it.
  SDValue Abs = DAG.getNode(ISD::FABS, DL, VT, Src);
  SDValue Setcc = DAG.getSetCC(DL, SetccVT, Abs, MaxValNode, ISD::SETOLT);
  return DAG.getSelect(DL, VT, Setcc, Truncated, Src);
}

// ISD::FROUND is defined to round to nearest with ties rounding away from 0.
// This mode isn't supported in vector hardware on RISCV. But as long as we
// aren't compiling with trapping math, we can emulate this with
// floor(X + copysign(nextafter(0.5, 0.0), X)).
// FIXME: Could be shorter by changing rounding mode, but we don't have FRM
// dependencies modeled yet.
// FIXME: Use masked operations to avoid final merge.
static SDValue lowerFROUND(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getSimpleValueType();
  assert(VT.isVector() && "Unexpected type");

  SDLoc DL(Op);

  // Freeze the source since we are increasing the number of uses.
  SDValue Src = DAG.getFreeze(Op.getOperand(0));

  // We do the conversion on the absolute value and fix the sign at the end.
  SDValue Abs = DAG.getNode(ISD::FABS, DL, VT, Src);

  const fltSemantics &FltSem = DAG.EVTToAPFloatSemantics(VT);
  bool Ignored;
  APFloat Point5Pred = APFloat(0.5f);
  Point5Pred.convert(FltSem, APFloat::rmNearestTiesToEven, &Ignored);
  Point5Pred.next(/*nextDown*/ true);

  // Add the adjustment.
  SDValue Adjust = DAG.getNode(ISD::FADD, DL, VT, Abs,
                               DAG.getConstantFP(Point5Pred, DL, VT));

  // Truncate to integer and convert back to fp.
  MVT IntVT = VT.changeVectorElementTypeToInteger();
  SDValue Truncated = DAG.getNode(ISD::FP_TO_SINT, DL, IntVT, Adjust);
  Truncated = DAG.getNode(ISD::SINT_TO_FP, DL, VT, Truncated);

  // Restore the original sign.
  Truncated = DAG.getNode(ISD::FCOPYSIGN, DL, VT, Truncated, Src);

  // Determine the largest integer that can be represented exactly. This and
  // values larger than it don't have any fractional bits so don't need to
  // be converted.
  unsigned Precision = APFloat::semanticsPrecision(FltSem);
  APFloat MaxVal = APFloat(FltSem);
  MaxVal.convertFromAPInt(APInt::getOneBitSet(Precision, Precision - 1),
                          /*IsSigned*/ false, APFloat::rmNearestTiesToEven);
  SDValue MaxValNode = DAG.getConstantFP(MaxVal, DL, VT);

  // If abs(Src) was larger than MaxVal or nan, keep it.
  MVT SetccVT = MVT::getVectorVT(MVT::i1, VT.getVectorElementCount());
  SDValue Setcc = DAG.getSetCC(DL, SetccVT, Abs, MaxValNode, ISD::SETOLT);
  return DAG.getSelect(DL, VT, Setcc, Truncated, Src);
}

static SDValue lowerSPLAT_VECTOR(SDValue Op, SelectionDAG &DAG,
                                 const RISCVSubtarget &Subtarget) {
  MVT VT = Op.getSimpleValueType();
  assert(VT.isFixedLengthVector() && "Unexpected vector!");

  MVT ContainerVT = getContainerForFixedLengthVector(DAG, VT, Subtarget);

  SDLoc DL(Op);
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  unsigned Opc =
      VT.isFloatingPoint() ? RISCVISD::VFMV_V_F_VL : RISCVISD::VMV_V_X_VL;
  SDValue Splat = DAG.getNode(Opc, DL, ContainerVT, DAG.getUNDEF(ContainerVT),
                              Op.getOperand(0), VL);
  return convertFromScalableVector(VT, Splat, DAG, Subtarget);
}

struct VIDSequence {
  int64_t StepNumerator;
  unsigned StepDenominator;
  int64_t Addend;
};

// Try to match an arithmetic-sequence BUILD_VECTOR [X,X+S,X+2*S,...,X+(N-1)*S]
// to the (non-zero) step S and start value X. This can be then lowered as the
// RVV sequence (VID * S) + X, for example.
// The step S is represented as an integer numerator divided by a positive
// denominator. Note that the implementation currently only identifies
// sequences in which either the numerator is +/- 1 or the denominator is 1. It
// cannot detect 2/3, for example.
// Note that this method will also match potentially unappealing index
// sequences, like <i32 0, i32 50939494>, however it is left to the caller to
// determine whether this is worth generating code for.
static Optional<VIDSequence> isSimpleVIDSequence(SDValue Op) {
  unsigned NumElts = Op.getNumOperands();
  assert(Op.getOpcode() == ISD::BUILD_VECTOR && "Unexpected BUILD_VECTOR");
  if (!Op.getValueType().isInteger())
    return None;

  Optional<unsigned> SeqStepDenom;
  Optional<int64_t> SeqStepNum, SeqAddend;
  Optional<std::pair<uint64_t, unsigned>> PrevElt;
  unsigned EltSizeInBits = Op.getValueType().getScalarSizeInBits();
  for (unsigned Idx = 0; Idx < NumElts; Idx++) {
    // Assume undef elements match the sequence; we just have to be careful
    // when interpolating across them.
    if (Op.getOperand(Idx).isUndef())
      continue;
    // The BUILD_VECTOR must be all constants.
    if (!isa<ConstantSDNode>(Op.getOperand(Idx)))
      return None;

    uint64_t Val = Op.getConstantOperandVal(Idx) &
                   maskTrailingOnes<uint64_t>(EltSizeInBits);

    if (PrevElt) {
      // Calculate the step since the last non-undef element, and ensure
      // it's consistent across the entire sequence.
      unsigned IdxDiff = Idx - PrevElt->second;
      int64_t ValDiff = SignExtend64(Val - PrevElt->first, EltSizeInBits);

      // A zero-value value difference means that we're somewhere in the middle
      // of a fractional step, e.g. <0,0,0*,0,1,1,1,1>. Wait until we notice a
      // step change before evaluating the sequence.
      if (ValDiff != 0) {
        int64_t Remainder = ValDiff % IdxDiff;
        // Normalize the step if it's greater than 1.
        if (Remainder != ValDiff) {
          // The difference must cleanly divide the element span.
          if (Remainder != 0)
            return None;
          ValDiff /= IdxDiff;
          IdxDiff = 1;
        }

        if (!SeqStepNum)
          SeqStepNum = ValDiff;
        else if (ValDiff != SeqStepNum)
          return None;

        if (!SeqStepDenom)
          SeqStepDenom = IdxDiff;
        else if (IdxDiff != *SeqStepDenom)
          return None;
      }
    }

    // Record and/or check any addend.
    if (SeqStepNum && SeqStepDenom) {
      uint64_t ExpectedVal =
          (int64_t)(Idx * (uint64_t)*SeqStepNum) / *SeqStepDenom;
      int64_t Addend = SignExtend64(Val - ExpectedVal, EltSizeInBits);
      if (!SeqAddend)
        SeqAddend = Addend;
      else if (SeqAddend != Addend)
        return None;
    }

    // Record this non-undef element for later.
    if (!PrevElt || PrevElt->first != Val)
      PrevElt = std::make_pair(Val, Idx);
  }
  // We need to have logged both a step and an addend for this to count as
  // a legal index sequence.
  if (!SeqStepNum || !SeqStepDenom || !SeqAddend)
    return None;

  return VIDSequence{*SeqStepNum, *SeqStepDenom, *SeqAddend};
}

// Match a splatted value (SPLAT_VECTOR/BUILD_VECTOR) of an EXTRACT_VECTOR_ELT
// and lower it as a VRGATHER_VX_VL from the source vector.
static SDValue matchSplatAsGather(SDValue SplatVal, MVT VT, const SDLoc &DL,
                                  SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget) {
  if (SplatVal.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
    return SDValue();
  SDValue Vec = SplatVal.getOperand(0);
  // Only perform this optimization on vectors of the same size for simplicity.
  if (Vec.getValueType() != VT)
    return SDValue();
  SDValue Idx = SplatVal.getOperand(1);
  // The index must be a legal type.
  if (Idx.getValueType() != Subtarget.getXLenVT())
    return SDValue();

  MVT ContainerVT = VT;
  if (VT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(DAG, VT, Subtarget);
    Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  }

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  SDValue Gather = DAG.getNode(RISCVISD::VRGATHER_VX_VL, DL, ContainerVT, Vec,
                               Idx, Mask, VL);

  if (!VT.isFixedLengthVector())
    return Gather;

  return convertFromScalableVector(VT, Gather, DAG, Subtarget);
}

static SDValue lowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG,
                                 const RISCVSubtarget &Subtarget) {
  MVT VT = Op.getSimpleValueType();
  assert(VT.isFixedLengthVector() && "Unexpected vector!");

  MVT ContainerVT = getContainerForFixedLengthVector(DAG, VT, Subtarget);

  SDLoc DL(Op);
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  MVT XLenVT = Subtarget.getXLenVT();
  unsigned NumElts = Op.getNumOperands();

  if (VT.getVectorElementType() == MVT::i1) {
    if (ISD::isBuildVectorAllZeros(Op.getNode())) {
      SDValue VMClr = DAG.getNode(RISCVISD::VMCLR_VL, DL, ContainerVT, VL);
      return convertFromScalableVector(VT, VMClr, DAG, Subtarget);
    }

    if (ISD::isBuildVectorAllOnes(Op.getNode())) {
      SDValue VMSet = DAG.getNode(RISCVISD::VMSET_VL, DL, ContainerVT, VL);
      return convertFromScalableVector(VT, VMSet, DAG, Subtarget);
    }

    // Lower constant mask BUILD_VECTORs via an integer vector type, in
    // scalar integer chunks whose bit-width depends on the number of mask
    // bits and XLEN.
    // First, determine the most appropriate scalar integer type to use. This
    // is at most XLenVT, but may be shrunk to a smaller vector element type
    // according to the size of the final vector - use i8 chunks rather than
    // XLenVT if we're producing a v8i1. This results in more consistent
    // codegen across RV32 and RV64.
    unsigned NumViaIntegerBits =
        std::min(std::max(NumElts, 8u), Subtarget.getXLen());
    NumViaIntegerBits = std::min(NumViaIntegerBits,
                                 Subtarget.getMaxELENForFixedLengthVectors());
    if (ISD::isBuildVectorOfConstantSDNodes(Op.getNode())) {
      // If we have to use more than one INSERT_VECTOR_ELT then this
      // optimization is likely to increase code size; avoid peforming it in
      // such a case. We can use a load from a constant pool in this case.
      if (DAG.shouldOptForSize() && NumElts > NumViaIntegerBits)
        return SDValue();
      // Now we can create our integer vector type. Note that it may be larger
      // than the resulting mask type: v4i1 would use v1i8 as its integer type.
      MVT IntegerViaVecVT =
          MVT::getVectorVT(MVT::getIntegerVT(NumViaIntegerBits),
                           divideCeil(NumElts, NumViaIntegerBits));

      uint64_t Bits = 0;
      unsigned BitPos = 0, IntegerEltIdx = 0;
      SDValue Vec = DAG.getUNDEF(IntegerViaVecVT);

      for (unsigned I = 0; I < NumElts; I++, BitPos++) {
        // Once we accumulate enough bits to fill our scalar type, insert into
        // our vector and clear our accumulated data.
        if (I != 0 && I % NumViaIntegerBits == 0) {
          if (NumViaIntegerBits <= 32)
            Bits = SignExtend64(Bits, 32);
          SDValue Elt = DAG.getConstant(Bits, DL, XLenVT);
          Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, IntegerViaVecVT, Vec,
                            Elt, DAG.getConstant(IntegerEltIdx, DL, XLenVT));
          Bits = 0;
          BitPos = 0;
          IntegerEltIdx++;
        }
        SDValue V = Op.getOperand(I);
        bool BitValue = !V.isUndef() && cast<ConstantSDNode>(V)->getZExtValue();
        Bits |= ((uint64_t)BitValue << BitPos);
      }

      // Insert the (remaining) scalar value into position in our integer
      // vector type.
      if (NumViaIntegerBits <= 32)
        Bits = SignExtend64(Bits, 32);
      SDValue Elt = DAG.getConstant(Bits, DL, XLenVT);
      Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, IntegerViaVecVT, Vec, Elt,
                        DAG.getConstant(IntegerEltIdx, DL, XLenVT));

      if (NumElts < NumViaIntegerBits) {
        // If we're producing a smaller vector than our minimum legal integer
        // type, bitcast to the equivalent (known-legal) mask type, and extract
        // our final mask.
        assert(IntegerViaVecVT == MVT::v1i8 && "Unexpected mask vector type");
        Vec = DAG.getBitcast(MVT::v8i1, Vec);
        Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, Vec,
                          DAG.getConstant(0, DL, XLenVT));
      } else {
        // Else we must have produced an integer type with the same size as the
        // mask type; bitcast for the final result.
        assert(VT.getSizeInBits() == IntegerViaVecVT.getSizeInBits());
        Vec = DAG.getBitcast(VT, Vec);
      }

      return Vec;
    }

    // A BUILD_VECTOR can be lowered as a SETCC. For each fixed-length mask
    // vector type, we have a legal equivalently-sized i8 type, so we can use
    // that.
    MVT WideVecVT = VT.changeVectorElementType(MVT::i8);
    SDValue VecZero = DAG.getConstant(0, DL, WideVecVT);

    SDValue WideVec;
    if (SDValue Splat = cast<BuildVectorSDNode>(Op)->getSplatValue()) {
      // For a splat, perform a scalar truncate before creating the wider
      // vector.
      assert(Splat.getValueType() == XLenVT &&
             "Unexpected type for i1 splat value");
      Splat = DAG.getNode(ISD::AND, DL, XLenVT, Splat,
                          DAG.getConstant(1, DL, XLenVT));
      WideVec = DAG.getSplatBuildVector(WideVecVT, DL, Splat);
    } else {
      SmallVector<SDValue, 8> Ops(Op->op_values());
      WideVec = DAG.getBuildVector(WideVecVT, DL, Ops);
      SDValue VecOne = DAG.getConstant(1, DL, WideVecVT);
      WideVec = DAG.getNode(ISD::AND, DL, WideVecVT, WideVec, VecOne);
    }

    return DAG.getSetCC(DL, VT, WideVec, VecZero, ISD::SETNE);
  }

  if (SDValue Splat = cast<BuildVectorSDNode>(Op)->getSplatValue()) {
    if (auto Gather = matchSplatAsGather(Splat, VT, DL, DAG, Subtarget))
      return Gather;
    unsigned Opc = VT.isFloatingPoint() ? RISCVISD::VFMV_V_F_VL
                                        : RISCVISD::VMV_V_X_VL;
    Splat =
        DAG.getNode(Opc, DL, ContainerVT, DAG.getUNDEF(ContainerVT), Splat, VL);
    return convertFromScalableVector(VT, Splat, DAG, Subtarget);
  }

  // Try and match index sequences, which we can lower to the vid instruction
  // with optional modifications. An all-undef vector is matched by
  // getSplatValue, above.
  if (auto SimpleVID = isSimpleVIDSequence(Op)) {
    int64_t StepNumerator = SimpleVID->StepNumerator;
    unsigned StepDenominator = SimpleVID->StepDenominator;
    int64_t Addend = SimpleVID->Addend;

    assert(StepNumerator != 0 && "Invalid step");
    bool Negate = false;
    int64_t SplatStepVal = StepNumerator;
    unsigned StepOpcode = ISD::MUL;
    if (StepNumerator != 1) {
      if (isPowerOf2_64(std::abs(StepNumerator))) {
        Negate = StepNumerator < 0;
        StepOpcode = ISD::SHL;
        SplatStepVal = Log2_64(std::abs(StepNumerator));
      }
    }

    // Only emit VIDs with suitably-small steps/addends. We use imm5 is a
    // threshold since it's the immediate value many RVV instructions accept.
    // There is no vmul.vi instruction so ensure multiply constant can fit in
    // a single addi instruction.
    if (((StepOpcode == ISD::MUL && isInt<12>(SplatStepVal)) ||
         (StepOpcode == ISD::SHL && isUInt<5>(SplatStepVal))) &&
        isPowerOf2_32(StepDenominator) && isInt<5>(Addend)) {
      SDValue VID = DAG.getNode(RISCVISD::VID_VL, DL, ContainerVT, Mask, VL);
      // Convert right out of the scalable type so we can use standard ISD
      // nodes for the rest of the computation. If we used scalable types with
      // these, we'd lose the fixed-length vector info and generate worse
      // vsetvli code.
      VID = convertFromScalableVector(VT, VID, DAG, Subtarget);
      if ((StepOpcode == ISD::MUL && SplatStepVal != 1) ||
          (StepOpcode == ISD::SHL && SplatStepVal != 0)) {
        SDValue SplatStep = DAG.getSplatVector(
            VT, DL, DAG.getConstant(SplatStepVal, DL, XLenVT));
        VID = DAG.getNode(StepOpcode, DL, VT, VID, SplatStep);
      }
      if (StepDenominator != 1) {
        SDValue SplatStep = DAG.getSplatVector(
            VT, DL, DAG.getConstant(Log2_64(StepDenominator), DL, XLenVT));
        VID = DAG.getNode(ISD::SRL, DL, VT, VID, SplatStep);
      }
      if (Addend != 0 || Negate) {
        SDValue SplatAddend =
            DAG.getSplatVector(VT, DL, DAG.getConstant(Addend, DL, XLenVT));
        VID = DAG.getNode(Negate ? ISD::SUB : ISD::ADD, DL, VT, SplatAddend, VID);
      }
      return VID;
    }
  }

  // Attempt to detect "hidden" splats, which only reveal themselves as splats
  // when re-interpreted as a vector with a larger element type. For example,
  //   v4i16 = build_vector i16 0, i16 1, i16 0, i16 1
  // could be instead splat as
  //   v2i32 = build_vector i32 0x00010000, i32 0x00010000
  // TODO: This optimization could also work on non-constant splats, but it
  // would require bit-manipulation instructions to construct the splat value.
  SmallVector<SDValue> Sequence;
  unsigned EltBitSize = VT.getScalarSizeInBits();
  const auto *BV = cast<BuildVectorSDNode>(Op);
  if (VT.isInteger() && EltBitSize < 64 &&
      ISD::isBuildVectorOfConstantSDNodes(Op.getNode()) &&
      BV->getRepeatedSequence(Sequence) &&
      (Sequence.size() * EltBitSize) <= 64) {
    unsigned SeqLen = Sequence.size();
    MVT ViaIntVT = MVT::getIntegerVT(EltBitSize * SeqLen);
    MVT ViaVecVT = MVT::getVectorVT(ViaIntVT, NumElts / SeqLen);
    assert((ViaIntVT == MVT::i16 || ViaIntVT == MVT::i32 ||
            ViaIntVT == MVT::i64) &&
           "Unexpected sequence type");

    unsigned EltIdx = 0;
    uint64_t EltMask = maskTrailingOnes<uint64_t>(EltBitSize);
    uint64_t SplatValue = 0;
    // Construct the amalgamated value which can be splatted as this larger
    // vector type.
    for (const auto &SeqV : Sequence) {
      if (!SeqV.isUndef())
        SplatValue |= ((cast<ConstantSDNode>(SeqV)->getZExtValue() & EltMask)
                       << (EltIdx * EltBitSize));
      EltIdx++;
    }

    // On RV64, sign-extend from 32 to 64 bits where possible in order to
    // achieve better constant materializion.
    if (Subtarget.is64Bit() && ViaIntVT == MVT::i32)
      SplatValue = SignExtend64(SplatValue, 32);

    // Since we can't introduce illegal i64 types at this stage, we can only
    // perform an i64 splat on RV32 if it is its own sign-extended value. That
    // way we can use RVV instructions to splat.
    assert((ViaIntVT.bitsLE(XLenVT) ||
            (!Subtarget.is64Bit() && ViaIntVT == MVT::i64)) &&
           "Unexpected bitcast sequence");
    if (ViaIntVT.bitsLE(XLenVT) || isInt<32>(SplatValue)) {
      SDValue ViaVL =
          DAG.getConstant(ViaVecVT.getVectorNumElements(), DL, XLenVT);
      MVT ViaContainerVT =
          getContainerForFixedLengthVector(DAG, ViaVecVT, Subtarget);
      SDValue Splat =
          DAG.getNode(RISCVISD::VMV_V_X_VL, DL, ViaContainerVT,
                      DAG.getUNDEF(ViaContainerVT),
                      DAG.getConstant(SplatValue, DL, XLenVT), ViaVL);
      Splat = convertFromScalableVector(ViaVecVT, Splat, DAG, Subtarget);
      return DAG.getBitcast(VT, Splat);
    }
  }

  // Try and optimize BUILD_VECTORs with "dominant values" - these are values
  // which constitute a large proportion of the elements. In such cases we can
  // splat a vector with the dominant element and make up the shortfall with
  // INSERT_VECTOR_ELTs.
  // Note that this includes vectors of 2 elements by association. The
  // upper-most element is the "dominant" one, allowing us to use a splat to
  // "insert" the upper element, and an insert of the lower element at position
  // 0, which improves codegen.
  SDValue DominantValue;
  unsigned MostCommonCount = 0;
  DenseMap<SDValue, unsigned> ValueCounts;
  unsigned NumUndefElts =
      count_if(Op->op_values(), [](const SDValue &V) { return V.isUndef(); });

  // Track the number of scalar loads we know we'd be inserting, estimated as
  // any non-zero floating-point constant. Other kinds of element are either
  // already in registers or are materialized on demand. The threshold at which
  // a vector load is more desirable than several scalar materializion and
  // vector-insertion instructions is not known.
  unsigned NumScalarLoads = 0;

  for (SDValue V : Op->op_values()) {
    if (V.isUndef())
      continue;

    ValueCounts.insert(std::make_pair(V, 0));
    unsigned &Count = ValueCounts[V];

    if (auto *CFP = dyn_cast<ConstantFPSDNode>(V))
      NumScalarLoads += !CFP->isExactlyValue(+0.0);

    // Is this value dominant? In case of a tie, prefer the highest element as
    // it's cheaper to insert near the beginning of a vector than it is at the
    // end.
    if (++Count >= MostCommonCount) {
      DominantValue = V;
      MostCommonCount = Count;
    }
  }

  assert(DominantValue && "Not expecting an all-undef BUILD_VECTOR");
  unsigned NumDefElts = NumElts - NumUndefElts;
  unsigned DominantValueCountThreshold = NumDefElts <= 2 ? 0 : NumDefElts - 2;

  // Don't perform this optimization when optimizing for size, since
  // materializing elements and inserting them tends to cause code bloat.
  if (!DAG.shouldOptForSize() && NumScalarLoads < NumElts &&
      ((MostCommonCount > DominantValueCountThreshold) ||
       (ValueCounts.size() <= Log2_32(NumDefElts)))) {
    // Start by splatting the most common element.
    SDValue Vec = DAG.getSplatBuildVector(VT, DL, DominantValue);

    DenseSet<SDValue> Processed{DominantValue};
    MVT SelMaskTy = VT.changeVectorElementType(MVT::i1);
    for (const auto &OpIdx : enumerate(Op->ops())) {
      const SDValue &V = OpIdx.value();
      if (V.isUndef() || !Processed.insert(V).second)
        continue;
      if (ValueCounts[V] == 1) {
        Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, VT, Vec, V,
                          DAG.getConstant(OpIdx.index(), DL, XLenVT));
      } else {
        // Blend in all instances of this value using a VSELECT, using a
        // mask where each bit signals whether that element is the one
        // we're after.
        SmallVector<SDValue> Ops;
        transform(Op->op_values(), std::back_inserter(Ops), [&](SDValue V1) {
          return DAG.getConstant(V == V1, DL, XLenVT);
        });
        Vec = DAG.getNode(ISD::VSELECT, DL, VT,
                          DAG.getBuildVector(SelMaskTy, DL, Ops),
                          DAG.getSplatBuildVector(VT, DL, V), Vec);
      }
    }

    return Vec;
  }

  return SDValue();
}

static SDValue splatPartsI64WithVL(const SDLoc &DL, MVT VT, SDValue Passthru,
                                   SDValue Lo, SDValue Hi, SDValue VL,
                                   SelectionDAG &DAG) {
  bool HasPassthru = Passthru && !Passthru.isUndef();
  if (!HasPassthru && !Passthru)
    Passthru = DAG.getUNDEF(VT);
  if (isa<ConstantSDNode>(Lo) && isa<ConstantSDNode>(Hi)) {
    int32_t LoC = cast<ConstantSDNode>(Lo)->getSExtValue();
    int32_t HiC = cast<ConstantSDNode>(Hi)->getSExtValue();
    // If Hi constant is all the same sign bit as Lo, lower this as a custom
    // node in order to try and match RVV vector/scalar instructions.
    if ((LoC >> 31) == HiC)
      return DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VT, Passthru, Lo, VL);

    // If vl is equal to XLEN_MAX and Hi constant is equal to Lo, we could use
    // vmv.v.x whose EEW = 32 to lower it.
    auto *Const = dyn_cast<ConstantSDNode>(VL);
    if (LoC == HiC && Const && Const->isAllOnesValue()) {
      MVT InterVT = MVT::getVectorVT(MVT::i32, VT.getVectorElementCount() * 2);
      // TODO: if vl <= min(VLMAX), we can also do this. But we could not
      // access the subtarget here now.
      auto InterVec = DAG.getNode(
          RISCVISD::VMV_V_X_VL, DL, InterVT, DAG.getUNDEF(InterVT), Lo,
                                  DAG.getRegister(RISCV::X0, MVT::i32));
      return DAG.getNode(ISD::BITCAST, DL, VT, InterVec);
    }
  }

  // Fall back to a stack store and stride x0 vector load.
  return DAG.getNode(RISCVISD::SPLAT_VECTOR_SPLIT_I64_VL, DL, VT, Passthru, Lo,
                     Hi, VL);
}

// Called by type legalization to handle splat of i64 on RV32.
// FIXME: We can optimize this when the type has sign or zero bits in one
// of the halves.
static SDValue splatSplitI64WithVL(const SDLoc &DL, MVT VT, SDValue Passthru,
                                   SDValue Scalar, SDValue VL,
                                   SelectionDAG &DAG) {
  assert(Scalar.getValueType() == MVT::i64 && "Unexpected VT!");
  SDValue Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Scalar,
                           DAG.getConstant(0, DL, MVT::i32));
  SDValue Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Scalar,
                           DAG.getConstant(1, DL, MVT::i32));
  return splatPartsI64WithVL(DL, VT, Passthru, Lo, Hi, VL, DAG);
}

// This function lowers a splat of a scalar operand Splat with the vector
// length VL. It ensures the final sequence is type legal, which is useful when
// lowering a splat after type legalization.
static SDValue lowerScalarSplat(SDValue Passthru, SDValue Scalar, SDValue VL,
                                MVT VT, SDLoc DL, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget) {
  bool HasPassthru = Passthru && !Passthru.isUndef();
  if (!HasPassthru && !Passthru)
    Passthru = DAG.getUNDEF(VT);
  if (VT.isFloatingPoint()) {
    // If VL is 1, we could use vfmv.s.f.
    if (isOneConstant(VL))
      return DAG.getNode(RISCVISD::VFMV_S_F_VL, DL, VT, Passthru, Scalar, VL);
    return DAG.getNode(RISCVISD::VFMV_V_F_VL, DL, VT, Passthru, Scalar, VL);
  }

  MVT XLenVT = Subtarget.getXLenVT();

  // Simplest case is that the operand needs to be promoted to XLenVT.
  if (Scalar.getValueType().bitsLE(XLenVT)) {
    // If the operand is a constant, sign extend to increase our chances
    // of being able to use a .vi instruction. ANY_EXTEND would become a
    // a zero extend and the simm5 check in isel would fail.
    // FIXME: Should we ignore the upper bits in isel instead?
    unsigned ExtOpc =
        isa<ConstantSDNode>(Scalar) ? ISD::SIGN_EXTEND : ISD::ANY_EXTEND;
    Scalar = DAG.getNode(ExtOpc, DL, XLenVT, Scalar);
    ConstantSDNode *Const = dyn_cast<ConstantSDNode>(Scalar);
    // If VL is 1 and the scalar value won't benefit from immediate, we could
    // use vmv.s.x.
    if (isOneConstant(VL) &&
        (!Const || isNullConstant(Scalar) || !isInt<5>(Const->getSExtValue())))
      return DAG.getNode(RISCVISD::VMV_S_X_VL, DL, VT, Passthru, Scalar, VL);
    return DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VT, Passthru, Scalar, VL);
  }

  assert(XLenVT == MVT::i32 && Scalar.getValueType() == MVT::i64 &&
         "Unexpected scalar for splat lowering!");

  if (isOneConstant(VL) && isNullConstant(Scalar))
    return DAG.getNode(RISCVISD::VMV_S_X_VL, DL, VT, Passthru,
                       DAG.getConstant(0, DL, XLenVT), VL);

  // Otherwise use the more complicated splatting algorithm.
  return splatSplitI64WithVL(DL, VT, Passthru, Scalar, VL, DAG);
}

static bool isInterleaveShuffle(ArrayRef<int> Mask, MVT VT, bool &SwapSources,
                                const RISCVSubtarget &Subtarget) {
  // We need to be able to widen elements to the next larger integer type.
  if (VT.getScalarSizeInBits() >= Subtarget.getMaxELENForFixedLengthVectors())
    return false;

  int Size = Mask.size();
  assert(Size == (int)VT.getVectorNumElements() && "Unexpected mask size");

  int Srcs[] = {-1, -1};
  for (int i = 0; i != Size; ++i) {
    // Ignore undef elements.
    if (Mask[i] < 0)
      continue;

    // Is this an even or odd element.
    int Pol = i % 2;

    // Ensure we consistently use the same source for this element polarity.
    int Src = Mask[i] / Size;
    if (Srcs[Pol] < 0)
      Srcs[Pol] = Src;
    if (Srcs[Pol] != Src)
      return false;

    // Make sure the element within the source is appropriate for this element
    // in the destination.
    int Elt = Mask[i] % Size;
    if (Elt != i / 2)
      return false;
  }

  // We need to find a source for each polarity and they can't be the same.
  if (Srcs[0] < 0 || Srcs[1] < 0 || Srcs[0] == Srcs[1])
    return false;

  // Swap the sources if the second source was in the even polarity.
  SwapSources = Srcs[0] > Srcs[1];

  return true;
}

/// Match shuffles that concatenate two vectors, rotate the concatenation,
/// and then extract the original number of elements from the rotated result.
/// This is equivalent to vector.splice or X86's PALIGNR instruction. The
/// returned rotation amount is for a rotate right, where elements move from
/// higher elements to lower elements. \p LoSrc indicates the first source
/// vector of the rotate or -1 for undef. \p HiSrc indicates the second vector
/// of the rotate or -1 for undef. At least one of \p LoSrc and \p HiSrc will be
/// 0 or 1 if a rotation is found.
///
/// NOTE: We talk about rotate to the right which matches how bit shift and
/// rotate instructions are described where LSBs are on the right, but LLVM IR
/// and the table below write vectors with the lowest elements on the left.
static int isElementRotate(int &LoSrc, int &HiSrc, ArrayRef<int> Mask) {
  int Size = Mask.size();

  // We need to detect various ways of spelling a rotation:
  //   [11, 12, 13, 14, 15,  0,  1,  2]
  //   [-1, 12, 13, 14, -1, -1,  1, -1]
  //   [-1, -1, -1, -1, -1, -1,  1,  2]
  //   [ 3,  4,  5,  6,  7,  8,  9, 10]
  //   [-1,  4,  5,  6, -1, -1,  9, -1]
  //   [-1,  4,  5,  6, -1, -1, -1, -1]
  int Rotation = 0;
  LoSrc = -1;
  HiSrc = -1;
  for (int i = 0; i != Size; ++i) {
    int M = Mask[i];
    if (M < 0)
      continue;

    // Determine where a rotate vector would have started.
    int StartIdx = i - (M % Size);
    // The identity rotation isn't interesting, stop.
    if (StartIdx == 0)
      return -1;

    // If we found the tail of a vector the rotation must be the missing
    // front. If we found the head of a vector, it must be how much of the
    // head.
    int CandidateRotation = StartIdx < 0 ? -StartIdx : Size - StartIdx;

    if (Rotation == 0)
      Rotation = CandidateRotation;
    else if (Rotation != CandidateRotation)
      // The rotations don't match, so we can't match this mask.
      return -1;

    // Compute which value this mask is pointing at.
    int MaskSrc = M < Size ? 0 : 1;

    // Compute which of the two target values this index should be assigned to.
    // This reflects whether the high elements are remaining or the low elemnts
    // are remaining.
    int &TargetSrc = StartIdx < 0 ? HiSrc : LoSrc;

    // Either set up this value if we've not encountered it before, or check
    // that it remains consistent.
    if (TargetSrc < 0)
      TargetSrc = MaskSrc;
    else if (TargetSrc != MaskSrc)
      // This may be a rotation, but it pulls from the inputs in some
      // unsupported interleaving.
      return -1;
  }

  // Check that we successfully analyzed the mask, and normalize the results.
  assert(Rotation != 0 && "Failed to locate a viable rotation!");
  assert((LoSrc >= 0 || HiSrc >= 0) &&
         "Failed to find a rotated input vector!");

  return Rotation;
}

static SDValue lowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget) {
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  SDLoc DL(Op);
  MVT XLenVT = Subtarget.getXLenVT();
  MVT VT = Op.getSimpleValueType();
  unsigned NumElts = VT.getVectorNumElements();
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op.getNode());

  MVT ContainerVT = getContainerForFixedLengthVector(DAG, VT, Subtarget);

  SDValue TrueMask, VL;
  std::tie(TrueMask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  if (SVN->isSplat()) {
    const int Lane = SVN->getSplatIndex();
    if (Lane >= 0) {
      MVT SVT = VT.getVectorElementType();

      // Turn splatted vector load into a strided load with an X0 stride.
      SDValue V = V1;
      // Peek through CONCAT_VECTORS as VectorCombine can concat a vector
      // with undef.
      // FIXME: Peek through INSERT_SUBVECTOR, EXTRACT_SUBVECTOR, bitcasts?
      int Offset = Lane;
      if (V.getOpcode() == ISD::CONCAT_VECTORS) {
        int OpElements =
            V.getOperand(0).getSimpleValueType().getVectorNumElements();
        V = V.getOperand(Offset / OpElements);
        Offset %= OpElements;
      }

      // We need to ensure the load isn't atomic or volatile.
      if (ISD::isNormalLoad(V.getNode()) && cast<LoadSDNode>(V)->isSimple()) {
        auto *Ld = cast<LoadSDNode>(V);
        Offset *= SVT.getStoreSize();
        SDValue NewAddr = DAG.getMemBasePlusOffset(Ld->getBasePtr(),
                                                   TypeSize::Fixed(Offset), DL);

        // If this is SEW=64 on RV32, use a strided load with a stride of x0.
        if (SVT.isInteger() && SVT.bitsGT(XLenVT)) {
          SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});
          SDValue IntID =
              DAG.getTargetConstant(Intrinsic::riscv_vlse, DL, XLenVT);
          SDValue Ops[] = {Ld->getChain(),
                           IntID,
                           DAG.getUNDEF(ContainerVT),
                           NewAddr,
                           DAG.getRegister(RISCV::X0, XLenVT),
                           VL};
          SDValue NewLoad = DAG.getMemIntrinsicNode(
              ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops, SVT,
              DAG.getMachineFunction().getMachineMemOperand(
                  Ld->getMemOperand(), Offset, SVT.getStoreSize()));
          DAG.makeEquivalentMemoryOrdering(Ld, NewLoad);
          return convertFromScalableVector(VT, NewLoad, DAG, Subtarget);
        }

        // Otherwise use a scalar load and splat. This will give the best
        // opportunity to fold a splat into the operation. ISel can turn it into
        // the x0 strided load if we aren't able to fold away the select.
        if (SVT.isFloatingPoint())
          V = DAG.getLoad(SVT, DL, Ld->getChain(), NewAddr,
                          Ld->getPointerInfo().getWithOffset(Offset),
                          Ld->getOriginalAlign(),
                          Ld->getMemOperand()->getFlags());
        else
          V = DAG.getExtLoad(ISD::SEXTLOAD, DL, XLenVT, Ld->getChain(), NewAddr,
                             Ld->getPointerInfo().getWithOffset(Offset), SVT,
                             Ld->getOriginalAlign(),
                             Ld->getMemOperand()->getFlags());
        DAG.makeEquivalentMemoryOrdering(Ld, V);

        unsigned Opc =
            VT.isFloatingPoint() ? RISCVISD::VFMV_V_F_VL : RISCVISD::VMV_V_X_VL;
        SDValue Splat =
            DAG.getNode(Opc, DL, ContainerVT, DAG.getUNDEF(ContainerVT), V, VL);
        return convertFromScalableVector(VT, Splat, DAG, Subtarget);
      }

      V1 = convertToScalableVector(ContainerVT, V1, DAG, Subtarget);
      assert(Lane < (int)NumElts && "Unexpected lane!");
      SDValue Gather =
          DAG.getNode(RISCVISD::VRGATHER_VX_VL, DL, ContainerVT, V1,
                      DAG.getConstant(Lane, DL, XLenVT), TrueMask, VL);
      return convertFromScalableVector(VT, Gather, DAG, Subtarget);
    }
  }

  ArrayRef<int> Mask = SVN->getMask();

  // Lower rotations to a SLIDEDOWN and a SLIDEUP. One of the source vectors may
  // be undef which can be handled with a single SLIDEDOWN/UP.
  int LoSrc, HiSrc;
  int Rotation = isElementRotate(LoSrc, HiSrc, Mask);
  if (Rotation > 0) {
    SDValue LoV, HiV;
    if (LoSrc >= 0) {
      LoV = LoSrc == 0 ? V1 : V2;
      LoV = convertToScalableVector(ContainerVT, LoV, DAG, Subtarget);
    }
    if (HiSrc >= 0) {
      HiV = HiSrc == 0 ? V1 : V2;
      HiV = convertToScalableVector(ContainerVT, HiV, DAG, Subtarget);
    }

    // We found a rotation. We need to slide HiV down by Rotation. Then we need
    // to slide LoV up by (NumElts - Rotation).
    unsigned InvRotate = NumElts - Rotation;

    SDValue Res = DAG.getUNDEF(ContainerVT);
    if (HiV) {
      // If we are doing a SLIDEDOWN+SLIDEUP, reduce the VL for the SLIDEDOWN.
      // FIXME: If we are only doing a SLIDEDOWN, don't reduce the VL as it
      // causes multiple vsetvlis in some test cases such as lowering
      // reduce.mul
      SDValue DownVL = VL;
      if (LoV)
        DownVL = DAG.getConstant(InvRotate, DL, XLenVT);
      Res =
          DAG.getNode(RISCVISD::VSLIDEDOWN_VL, DL, ContainerVT, Res, HiV,
                      DAG.getConstant(Rotation, DL, XLenVT), TrueMask, DownVL);
    }
    if (LoV)
      Res = DAG.getNode(RISCVISD::VSLIDEUP_VL, DL, ContainerVT, Res, LoV,
                        DAG.getConstant(InvRotate, DL, XLenVT), TrueMask, VL);

    return convertFromScalableVector(VT, Res, DAG, Subtarget);
  }

  // Detect an interleave shuffle and lower to
  // (vmaccu.vx (vwaddu.vx lohalf(V1), lohalf(V2)), lohalf(V2), (2^eltbits - 1))
  bool SwapSources;
  if (isInterleaveShuffle(Mask, VT, SwapSources, Subtarget)) {
    // Swap sources if needed.
    if (SwapSources)
      std::swap(V1, V2);

    // Extract the lower half of the vectors.
    MVT HalfVT = VT.getHalfNumVectorElementsVT();
    V1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, HalfVT, V1,
                     DAG.getConstant(0, DL, XLenVT));
    V2 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, HalfVT, V2,
                     DAG.getConstant(0, DL, XLenVT));

    // Double the element width and halve the number of elements in an int type.
    unsigned EltBits = VT.getScalarSizeInBits();
    MVT WideIntEltVT = MVT::getIntegerVT(EltBits * 2);
    MVT WideIntVT =
        MVT::getVectorVT(WideIntEltVT, VT.getVectorNumElements() / 2);
    // Convert this to a scalable vector. We need to base this on the
    // destination size to ensure there's always a type with a smaller LMUL.
    MVT WideIntContainerVT =
        getContainerForFixedLengthVector(DAG, WideIntVT, Subtarget);

    // Convert sources to scalable vectors with the same element count as the
    // larger type.
    MVT HalfContainerVT = MVT::getVectorVT(
        VT.getVectorElementType(), WideIntContainerVT.getVectorElementCount());
    V1 = convertToScalableVector(HalfContainerVT, V1, DAG, Subtarget);
    V2 = convertToScalableVector(HalfContainerVT, V2, DAG, Subtarget);

    // Cast sources to integer.
    MVT IntEltVT = MVT::getIntegerVT(EltBits);
    MVT IntHalfVT =
        MVT::getVectorVT(IntEltVT, HalfContainerVT.getVectorElementCount());
    V1 = DAG.getBitcast(IntHalfVT, V1);
    V2 = DAG.getBitcast(IntHalfVT, V2);

    // Freeze V2 since we use it twice and we need to be sure that the add and
    // multiply see the same value.
    V2 = DAG.getFreeze(V2);

    // Recreate TrueMask using the widened type's element count.
    MVT MaskVT =
        MVT::getVectorVT(MVT::i1, HalfContainerVT.getVectorElementCount());
    TrueMask = DAG.getNode(RISCVISD::VMSET_VL, DL, MaskVT, VL);

    // Widen V1 and V2 with 0s and add one copy of V2 to V1.
    SDValue Add = DAG.getNode(RISCVISD::VWADDU_VL, DL, WideIntContainerVT, V1,
                              V2, TrueMask, VL);
    // Create 2^eltbits - 1 copies of V2 by multiplying by the largest integer.
    SDValue Multiplier = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, IntHalfVT,
                                     DAG.getUNDEF(IntHalfVT),
                                     DAG.getAllOnesConstant(DL, XLenVT));
    SDValue WidenMul = DAG.getNode(RISCVISD::VWMULU_VL, DL, WideIntContainerVT,
                                   V2, Multiplier, TrueMask, VL);
    // Add the new copies to our previous addition giving us 2^eltbits copies of
    // V2. This is equivalent to shifting V2 left by eltbits. This should
    // combine with the vwmulu.vv above to form vwmaccu.vv.
    Add = DAG.getNode(RISCVISD::ADD_VL, DL, WideIntContainerVT, Add, WidenMul,
                      TrueMask, VL);
    // Cast back to ContainerVT. We need to re-create a new ContainerVT in case
    // WideIntContainerVT is a larger fractional LMUL than implied by the fixed
    // vector VT.
    ContainerVT =
        MVT::getVectorVT(VT.getVectorElementType(),
                         WideIntContainerVT.getVectorElementCount() * 2);
    Add = DAG.getBitcast(ContainerVT, Add);
    return convertFromScalableVector(VT, Add, DAG, Subtarget);
  }

  // Detect shuffles which can be re-expressed as vector selects; these are
  // shuffles in which each element in the destination is taken from an element
  // at the corresponding index in either source vectors.
  bool IsSelect = all_of(enumerate(Mask), [&](const auto &MaskIdx) {
    int MaskIndex = MaskIdx.value();
    return MaskIndex < 0 || MaskIdx.index() == (unsigned)MaskIndex % NumElts;
  });

  assert(!V1.isUndef() && "Unexpected shuffle canonicalization");

  SmallVector<SDValue> MaskVals;
  // As a backup, shuffles can be lowered via a vrgather instruction, possibly
  // merged with a second vrgather.
  SmallVector<SDValue> GatherIndicesLHS, GatherIndicesRHS;

  // By default we preserve the original operand order, and use a mask to
  // select LHS as true and RHS as false. However, since RVV vector selects may
  // feature splats but only on the LHS, we may choose to invert our mask and
  // instead select between RHS and LHS.
  bool SwapOps = DAG.isSplatValue(V2) && !DAG.isSplatValue(V1);
  bool InvertMask = IsSelect == SwapOps;

  // Keep a track of which non-undef indices are used by each LHS/RHS shuffle
  // half.
  DenseMap<int, unsigned> LHSIndexCounts, RHSIndexCounts;

  // Now construct the mask that will be used by the vselect or blended
  // vrgather operation. For vrgathers, construct the appropriate indices into
  // each vector.
  for (int MaskIndex : Mask) {
    bool SelectMaskVal = (MaskIndex < (int)NumElts) ^ InvertMask;
    MaskVals.push_back(DAG.getConstant(SelectMaskVal, DL, XLenVT));
    if (!IsSelect) {
      bool IsLHSOrUndefIndex = MaskIndex < (int)NumElts;
      GatherIndicesLHS.push_back(IsLHSOrUndefIndex && MaskIndex >= 0
                                     ? DAG.getConstant(MaskIndex, DL, XLenVT)
                                     : DAG.getUNDEF(XLenVT));
      GatherIndicesRHS.push_back(
          IsLHSOrUndefIndex ? DAG.getUNDEF(XLenVT)
                            : DAG.getConstant(MaskIndex - NumElts, DL, XLenVT));
      if (IsLHSOrUndefIndex && MaskIndex >= 0)
        ++LHSIndexCounts[MaskIndex];
      if (!IsLHSOrUndefIndex)
        ++RHSIndexCounts[MaskIndex - NumElts];
    }
  }

  if (SwapOps) {
    std::swap(V1, V2);
    std::swap(GatherIndicesLHS, GatherIndicesRHS);
  }

  assert(MaskVals.size() == NumElts && "Unexpected select-like shuffle");
  MVT MaskVT = MVT::getVectorVT(MVT::i1, NumElts);
  SDValue SelectMask = DAG.getBuildVector(MaskVT, DL, MaskVals);

  if (IsSelect)
    return DAG.getNode(ISD::VSELECT, DL, VT, SelectMask, V1, V2);

  if (VT.getScalarSizeInBits() == 8 && VT.getVectorNumElements() > 256) {
    // On such a large vector we're unable to use i8 as the index type.
    // FIXME: We could promote the index to i16 and use vrgatherei16, but that
    // may involve vector splitting if we're already at LMUL=8, or our
    // user-supplied maximum fixed-length LMUL.
    return SDValue();
  }

  unsigned GatherVXOpc = RISCVISD::VRGATHER_VX_VL;
  unsigned GatherVVOpc = RISCVISD::VRGATHER_VV_VL;
  MVT IndexVT = VT.changeTypeToInteger();
  // Since we can't introduce illegal index types at this stage, use i16 and
  // vrgatherei16 if the corresponding index type for plain vrgather is greater
  // than XLenVT.
  if (IndexVT.getScalarType().bitsGT(XLenVT)) {
    GatherVVOpc = RISCVISD::VRGATHEREI16_VV_VL;
    IndexVT = IndexVT.changeVectorElementType(MVT::i16);
  }

  MVT IndexContainerVT =
      ContainerVT.changeVectorElementType(IndexVT.getScalarType());

  SDValue Gather;
  // TODO: This doesn't trigger for i64 vectors on RV32, since there we
  // encounter a bitcasted BUILD_VECTOR with low/high i32 values.
  if (SDValue SplatValue = DAG.getSplatValue(V1, /*LegalTypes*/ true)) {
    Gather = lowerScalarSplat(SDValue(), SplatValue, VL, ContainerVT, DL, DAG,
                              Subtarget);
  } else {
    V1 = convertToScalableVector(ContainerVT, V1, DAG, Subtarget);
    // If only one index is used, we can use a "splat" vrgather.
    // TODO: We can splat the most-common index and fix-up any stragglers, if
    // that's beneficial.
    if (LHSIndexCounts.size() == 1) {
      int SplatIndex = LHSIndexCounts.begin()->getFirst();
      Gather =
          DAG.getNode(GatherVXOpc, DL, ContainerVT, V1,
                      DAG.getConstant(SplatIndex, DL, XLenVT), TrueMask, VL);
    } else {
      SDValue LHSIndices = DAG.getBuildVector(IndexVT, DL, GatherIndicesLHS);
      LHSIndices =
          convertToScalableVector(IndexContainerVT, LHSIndices, DAG, Subtarget);

      Gather = DAG.getNode(GatherVVOpc, DL, ContainerVT, V1, LHSIndices,
                           TrueMask, VL);
    }
  }

  // If a second vector operand is used by this shuffle, blend it in with an
  // additional vrgather.
  if (!V2.isUndef()) {
    V2 = convertToScalableVector(ContainerVT, V2, DAG, Subtarget);
    // If only one index is used, we can use a "splat" vrgather.
    // TODO: We can splat the most-common index and fix-up any stragglers, if
    // that's beneficial.
    if (RHSIndexCounts.size() == 1) {
      int SplatIndex = RHSIndexCounts.begin()->getFirst();
      V2 = DAG.getNode(GatherVXOpc, DL, ContainerVT, V2,
                       DAG.getConstant(SplatIndex, DL, XLenVT), TrueMask, VL);
    } else {
      SDValue RHSIndices = DAG.getBuildVector(IndexVT, DL, GatherIndicesRHS);
      RHSIndices =
          convertToScalableVector(IndexContainerVT, RHSIndices, DAG, Subtarget);
      V2 = DAG.getNode(GatherVVOpc, DL, ContainerVT, V2, RHSIndices, TrueMask,
                       VL);
    }

    MVT MaskContainerVT = ContainerVT.changeVectorElementType(MVT::i1);
    SelectMask =
        convertToScalableVector(MaskContainerVT, SelectMask, DAG, Subtarget);

    Gather = DAG.getNode(RISCVISD::VSELECT_VL, DL, ContainerVT, SelectMask, V2,
                         Gather, VL);
  }

  return convertFromScalableVector(VT, Gather, DAG, Subtarget);
}

bool RISCVTargetLowering::isShuffleMaskLegal(ArrayRef<int> M, EVT VT) const {
  // Support splats for any type. These should type legalize well.
  if (ShuffleVectorSDNode::isSplatMask(M.data(), VT))
    return true;

  // Only support legal VTs for other shuffles for now.
  if (!isTypeLegal(VT))
    return false;

  MVT SVT = VT.getSimpleVT();

  bool SwapSources;
  int LoSrc, HiSrc;
  return (isElementRotate(LoSrc, HiSrc, M) > 0) ||
         isInterleaveShuffle(M, SVT, SwapSources, Subtarget);
}

static SDValue getRVVFPExtendOrRound(SDValue Op, MVT VT, MVT ContainerVT,
                                     SDLoc DL, SelectionDAG &DAG,
                                     const RISCVSubtarget &Subtarget) {
  if (VT.isScalableVector())
    return DAG.getFPExtendOrRound(Op, DL, VT);
  assert(VT.isFixedLengthVector() &&
         "Unexpected value type for RVV FP extend/round lowering");
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);
  unsigned RVVOpc = ContainerVT.bitsGT(Op.getSimpleValueType())
                        ? RISCVISD::FP_EXTEND_VL
                        : RISCVISD::FP_ROUND_VL;
  return DAG.getNode(RVVOpc, DL, ContainerVT, Op, Mask, VL);
}

// Lower CTLZ_ZERO_UNDEF or CTTZ_ZERO_UNDEF by converting to FP and extracting
// the exponent.
static SDValue lowerCTLZ_CTTZ_ZERO_UNDEF(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getSimpleValueType();
  unsigned EltSize = VT.getScalarSizeInBits();
  SDValue Src = Op.getOperand(0);
  SDLoc DL(Op);

  // We need a FP type that can represent the value.
  // TODO: Use f16 for i8 when possible?
  MVT FloatEltVT = EltSize == 32 ? MVT::f64 : MVT::f32;
  MVT FloatVT = MVT::getVectorVT(FloatEltVT, VT.getVectorElementCount());

  // Legal types should have been checked in the RISCVTargetLowering
  // constructor.
  // TODO: Splitting may make sense in some cases.
  assert(DAG.getTargetLoweringInfo().isTypeLegal(FloatVT) &&
         "Expected legal float type!");

  // For CTTZ_ZERO_UNDEF, we need to extract the lowest set bit using X & -X.
  // The trailing zero count is equal to log2 of this single bit value.
  if (Op.getOpcode() == ISD::CTTZ_ZERO_UNDEF) {
    SDValue Neg =
        DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT), Src);
    Src = DAG.getNode(ISD::AND, DL, VT, Src, Neg);
  }

  // We have a legal FP type, convert to it.
  SDValue FloatVal = DAG.getNode(ISD::UINT_TO_FP, DL, FloatVT, Src);
  // Bitcast to integer and shift the exponent to the LSB.
  EVT IntVT = FloatVT.changeVectorElementTypeToInteger();
  SDValue Bitcast = DAG.getBitcast(IntVT, FloatVal);
  unsigned ShiftAmt = FloatEltVT == MVT::f64 ? 52 : 23;
  SDValue Shift = DAG.getNode(ISD::SRL, DL, IntVT, Bitcast,
                              DAG.getConstant(ShiftAmt, DL, IntVT));
  // Truncate back to original type to allow vnsrl.
  SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, VT, Shift);
  // The exponent contains log2 of the value in biased form.
  unsigned ExponentBias = FloatEltVT == MVT::f64 ? 1023 : 127;

  // For trailing zeros, we just need to subtract the bias.
  if (Op.getOpcode() == ISD::CTTZ_ZERO_UNDEF)
    return DAG.getNode(ISD::SUB, DL, VT, Trunc,
                       DAG.getConstant(ExponentBias, DL, VT));

  // For leading zeros, we need to remove the bias and convert from log2 to
  // leading zeros. We can do this by subtracting from (Bias + (EltSize - 1)).
  unsigned Adjust = ExponentBias + (EltSize - 1);
  return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(Adjust, DL, VT), Trunc);
}

// While RVV has alignment restrictions, we should always be able to load as a
// legal equivalently-sized byte-typed vector instead. This method is
// responsible for re-expressing a ISD::LOAD via a correctly-aligned type. If
// the load is already correctly-aligned, it returns SDValue().
SDValue RISCVTargetLowering::expandUnalignedRVVLoad(SDValue Op,
                                                    SelectionDAG &DAG) const {
  auto *Load = cast<LoadSDNode>(Op);
  assert(Load && Load->getMemoryVT().isVector() && "Expected vector load");

  if (allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                     Load->getMemoryVT(),
                                     *Load->getMemOperand()))
    return SDValue();

  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  unsigned EltSizeBits = VT.getScalarSizeInBits();
  assert((EltSizeBits == 16 || EltSizeBits == 32 || EltSizeBits == 64) &&
         "Unexpected unaligned RVV load type");
  MVT NewVT =
      MVT::getVectorVT(MVT::i8, VT.getVectorElementCount() * (EltSizeBits / 8));
  assert(NewVT.isValid() &&
         "Expecting equally-sized RVV vector types to be legal");
  SDValue L = DAG.getLoad(NewVT, DL, Load->getChain(), Load->getBasePtr(),
                          Load->getPointerInfo(), Load->getOriginalAlign(),
                          Load->getMemOperand()->getFlags());
  return DAG.getMergeValues({DAG.getBitcast(VT, L), L.getValue(1)}, DL);
}

// While RVV has alignment restrictions, we should always be able to store as a
// legal equivalently-sized byte-typed vector instead. This method is
// responsible for re-expressing a ISD::STORE via a correctly-aligned type. It
// returns SDValue() if the store is already correctly aligned.
SDValue RISCVTargetLowering::expandUnalignedRVVStore(SDValue Op,
                                                     SelectionDAG &DAG) const {
  auto *Store = cast<StoreSDNode>(Op);
  assert(Store && Store->getValue().getValueType().isVector() &&
         "Expected vector store");

  if (allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                     Store->getMemoryVT(),
                                     *Store->getMemOperand()))
    return SDValue();

  SDLoc DL(Op);
  SDValue StoredVal = Store->getValue();
  MVT VT = StoredVal.getSimpleValueType();
  unsigned EltSizeBits = VT.getScalarSizeInBits();
  assert((EltSizeBits == 16 || EltSizeBits == 32 || EltSizeBits == 64) &&
         "Unexpected unaligned RVV store type");
  MVT NewVT =
      MVT::getVectorVT(MVT::i8, VT.getVectorElementCount() * (EltSizeBits / 8));
  assert(NewVT.isValid() &&
         "Expecting equally-sized RVV vector types to be legal");
  StoredVal = DAG.getBitcast(NewVT, StoredVal);
  return DAG.getStore(Store->getChain(), DL, StoredVal, Store->getBasePtr(),
                      Store->getPointerInfo(), Store->getOriginalAlign(),
                      Store->getMemOperand()->getFlags());
}

SDValue RISCVTargetLowering::LowerOperation(SDValue Op,
                                            SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    report_fatal_error("unimplemented operand");
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::BlockAddress:
    return lowerBlockAddress(Op, DAG);
  case ISD::ConstantPool:
    return lowerConstantPool(Op, DAG);
  case ISD::JumpTable:
    return lowerJumpTable(Op, DAG);
  case ISD::GlobalTLSAddress:
    return lowerGlobalTLSAddress(Op, DAG);
  case ISD::SELECT:
    return lowerSELECT(Op, DAG);
  case ISD::BRCOND:
    return lowerBRCOND(Op, DAG);
  case ISD::VASTART:
    return lowerVASTART(Op, DAG);
  case ISD::FRAMEADDR:
    return lowerFRAMEADDR(Op, DAG);
  case ISD::RETURNADDR:
    return lowerRETURNADDR(Op, DAG);
  case ISD::SHL_PARTS:
    return lowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return lowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return lowerShiftRightParts(Op, DAG, false);
  case ISD::BITCAST: {
    SDLoc DL(Op);
    EVT VT = Op.getValueType();
    SDValue Op0 = Op.getOperand(0);
    EVT Op0VT = Op0.getValueType();
    MVT XLenVT = Subtarget.getXLenVT();
    if (VT.isFixedLengthVector()) {
      // We can handle fixed length vector bitcasts with a simple replacement
      // in isel.
      if (Op0VT.isFixedLengthVector())
        return Op;
      // When bitcasting from scalar to fixed-length vector, insert the scalar
      // into a one-element vector of the result type, and perform a vector
      // bitcast.
      if (!Op0VT.isVector()) {
        EVT BVT = EVT::getVectorVT(*DAG.getContext(), Op0VT, 1);
        if (!isTypeLegal(BVT))
          return SDValue();
        return DAG.getBitcast(VT, DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, BVT,
                                              DAG.getUNDEF(BVT), Op0,
                                              DAG.getConstant(0, DL, XLenVT)));
      }
      return SDValue();
    }
    // Custom-legalize bitcasts from fixed-length vector types to scalar types
    // thus: bitcast the vector to a one-element vector type whose element type
    // is the same as the result type, and extract the first element.
    if (!VT.isVector() && Op0VT.isFixedLengthVector()) {
      EVT BVT = EVT::getVectorVT(*DAG.getContext(), VT, 1);
      if (!isTypeLegal(BVT))
        return SDValue();
      SDValue BVec = DAG.getBitcast(BVT, Op0);
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, BVec,
                         DAG.getConstant(0, DL, XLenVT));
    }
    if (VT == MVT::f16 && Op0VT == MVT::i16 && Subtarget.hasStdExtZfh()) {
      SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, XLenVT, Op0);
      SDValue FPConv = DAG.getNode(RISCVISD::FMV_H_X, DL, MVT::f16, NewOp0);
      return FPConv;
    }
    if (VT == MVT::f32 && Op0VT == MVT::i32 && Subtarget.is64Bit() &&
        Subtarget.hasStdExtF()) {
      SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op0);
      SDValue FPConv =
          DAG.getNode(RISCVISD::FMV_W_X_RV64, DL, MVT::f32, NewOp0);
      return FPConv;
    }
    return SDValue();
  }
  case ISD::INTRINSIC_WO_CHAIN:
    return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN:
    return LowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID:
    return LowerINTRINSIC_VOID(Op, DAG);
  case ISD::BSWAP:
  case ISD::BITREVERSE: {
    MVT VT = Op.getSimpleValueType();
    SDLoc DL(Op);
    if (Subtarget.hasStdExtZbp()) {
      // Convert BSWAP/BITREVERSE to GREVI to enable GREVI combinining.
      // Start with the maximum immediate value which is the bitwidth - 1.
      unsigned Imm = VT.getSizeInBits() - 1;
      // If this is BSWAP rather than BITREVERSE, clear the lower 3 bits.
      if (Op.getOpcode() == ISD::BSWAP)
        Imm &= ~0x7U;
      return DAG.getNode(RISCVISD::GREV, DL, VT, Op.getOperand(0),
                         DAG.getConstant(Imm, DL, VT));
    }
    assert(Subtarget.hasStdExtZbkb() && "Unexpected custom legalization");
    assert(Op.getOpcode() == ISD::BITREVERSE && "Unexpected opcode");
    // Expand bitreverse to a bswap(rev8) followed by brev8.
    SDValue BSwap = DAG.getNode(ISD::BSWAP, DL, VT, Op.getOperand(0));
    // We use the Zbp grevi encoding for rev.b/brev8 which will be recognized
    // as brev8 by an isel pattern.
    return DAG.getNode(RISCVISD::GREV, DL, VT, BSwap,
                       DAG.getConstant(7, DL, VT));
  }
  case ISD::FSHL:
  case ISD::FSHR: {
    MVT VT = Op.getSimpleValueType();
    assert(VT == Subtarget.getXLenVT() && "Unexpected custom legalization");
    SDLoc DL(Op);
    // FSL/FSR take a log2(XLen)+1 bit shift amount but XLenVT FSHL/FSHR only
    // use log(XLen) bits. Mask the shift amount accordingly to prevent
    // accidentally setting the extra bit.
    unsigned ShAmtWidth = Subtarget.getXLen() - 1;
    SDValue ShAmt = DAG.getNode(ISD::AND, DL, VT, Op.getOperand(2),
                                DAG.getConstant(ShAmtWidth, DL, VT));
    // fshl and fshr concatenate their operands in the same order. fsr and fsl
    // instruction use different orders. fshl will return its first operand for
    // shift of zero, fshr will return its second operand. fsl and fsr both
    // return rs1 so the ISD nodes need to have different operand orders.
    // Shift amount is in rs2.
    SDValue Op0 = Op.getOperand(0);
    SDValue Op1 = Op.getOperand(1);
    unsigned Opc = RISCVISD::FSL;
    if (Op.getOpcode() == ISD::FSHR) {
      std::swap(Op0, Op1);
      Opc = RISCVISD::FSR;
    }
    return DAG.getNode(Opc, DL, VT, Op0, Op1, ShAmt);
  }
  case ISD::TRUNCATE: {
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    // Only custom-lower vector truncates
    if (!VT.isVector())
      return Op;

    // Truncates to mask types are handled differently
    if (VT.getVectorElementType() == MVT::i1)
      return lowerVectorMaskTrunc(Op, DAG);

    // RVV only has truncates which operate from SEW*2->SEW, so lower arbitrary
    // truncates as a series of "RISCVISD::TRUNCATE_VECTOR_VL" nodes which
    // truncate by one power of two at a time.
    MVT DstEltVT = VT.getVectorElementType();

    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();
    MVT SrcEltVT = SrcVT.getVectorElementType();

    assert(DstEltVT.bitsLT(SrcEltVT) &&
           isPowerOf2_64(DstEltVT.getSizeInBits()) &&
           isPowerOf2_64(SrcEltVT.getSizeInBits()) &&
           "Unexpected vector truncate lowering");

    MVT ContainerVT = SrcVT;
    if (SrcVT.isFixedLengthVector()) {
      ContainerVT = getContainerForFixedLengthVector(SrcVT);
      Src = convertToScalableVector(ContainerVT, Src, DAG, Subtarget);
    }

    SDValue Result = Src;
    SDValue Mask, VL;
    std::tie(Mask, VL) =
        getDefaultVLOps(SrcVT, ContainerVT, DL, DAG, Subtarget);
    LLVMContext &Context = *DAG.getContext();
    const ElementCount Count = ContainerVT.getVectorElementCount();
    do {
      SrcEltVT = MVT::getIntegerVT(SrcEltVT.getSizeInBits() / 2);
      EVT ResultVT = EVT::getVectorVT(Context, SrcEltVT, Count);
      Result = DAG.getNode(RISCVISD::TRUNCATE_VECTOR_VL, DL, ResultVT, Result,
                           Mask, VL);
    } while (SrcEltVT != DstEltVT);

    if (SrcVT.isFixedLengthVector())
      Result = convertFromScalableVector(VT, Result, DAG, Subtarget);

    return Result;
  }
  case ISD::ANY_EXTEND:
  case ISD::ZERO_EXTEND:
    if (Op.getOperand(0).getValueType().isVector() &&
        Op.getOperand(0).getValueType().getVectorElementType() == MVT::i1)
      return lowerVectorMaskExt(Op, DAG, /*ExtVal*/ 1);
    return lowerFixedLengthVectorExtendToRVV(Op, DAG, RISCVISD::VZEXT_VL);
  case ISD::SIGN_EXTEND:
    if (Op.getOperand(0).getValueType().isVector() &&
        Op.getOperand(0).getValueType().getVectorElementType() == MVT::i1)
      return lowerVectorMaskExt(Op, DAG, /*ExtVal*/ -1);
    return lowerFixedLengthVectorExtendToRVV(Op, DAG, RISCVISD::VSEXT_VL);
  case ISD::SPLAT_VECTOR_PARTS:
    return lowerSPLAT_VECTOR_PARTS(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    return lowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return lowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::VSCALE: {
    MVT VT = Op.getSimpleValueType();
    SDLoc DL(Op);
    SDValue VLENB = DAG.getNode(RISCVISD::READ_VLENB, DL, VT);
    // We define our scalable vector types for lmul=1 to use a 64 bit known
    // minimum size. e.g. <vscale x 2 x i32>. VLENB is in bytes so we calculate
    // vscale as VLENB / 8.
    static_assert(RISCV::RVVBitsPerBlock == 64, "Unexpected bits per block!");
    if (Subtarget.getMinVLen() < RISCV::RVVBitsPerBlock)
      report_fatal_error("Support for VLEN==32 is incomplete.");
    if (isa<ConstantSDNode>(Op.getOperand(0))) {
      // We assume VLENB is a multiple of 8. We manually choose the best shift
      // here because SimplifyDemandedBits isn't always able to simplify it.
      uint64_t Val = Op.getConstantOperandVal(0);
      if (isPowerOf2_64(Val)) {
        uint64_t Log2 = Log2_64(Val);
        if (Log2 < 3)
          return DAG.getNode(ISD::SRL, DL, VT, VLENB,
                             DAG.getConstant(3 - Log2, DL, VT));
        if (Log2 > 3)
          return DAG.getNode(ISD::SHL, DL, VT, VLENB,
                             DAG.getConstant(Log2 - 3, DL, VT));
        return VLENB;
      }
      // If the multiplier is a multiple of 8, scale it down to avoid needing
      // to shift the VLENB value.
      if ((Val % 8) == 0)
        return DAG.getNode(ISD::MUL, DL, VT, VLENB,
                           DAG.getConstant(Val / 8, DL, VT));
    }

    SDValue VScale = DAG.getNode(ISD::SRL, DL, VT, VLENB,
                                 DAG.getConstant(3, DL, VT));
    return DAG.getNode(ISD::MUL, DL, VT, VScale, Op.getOperand(0));
  }
  case ISD::FPOWI: {
    // Custom promote f16 powi with illegal i32 integer type on RV64. Once
    // promoted this will be legalized into a libcall by LegalizeIntegerTypes.
    if (Op.getValueType() == MVT::f16 && Subtarget.is64Bit() &&
        Op.getOperand(1).getValueType() == MVT::i32) {
      SDLoc DL(Op);
      SDValue Op0 = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f32, Op.getOperand(0));
      SDValue Powi =
          DAG.getNode(ISD::FPOWI, DL, MVT::f32, Op0, Op.getOperand(1));
      return DAG.getNode(ISD::FP_ROUND, DL, MVT::f16, Powi,
                         DAG.getIntPtrConstant(0, DL));
    }
    return SDValue();
  }
  case ISD::FP_EXTEND: {
    // RVV can only do fp_extend to types double the size as the source. We
    // custom-lower f16->f64 extensions to two hops of ISD::FP_EXTEND, going
    // via f32.
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();

    // Prepare any fixed-length vector operands.
    MVT ContainerVT = VT;
    if (SrcVT.isFixedLengthVector()) {
      ContainerVT = getContainerForFixedLengthVector(VT);
      MVT SrcContainerVT =
          ContainerVT.changeVectorElementType(SrcVT.getVectorElementType());
      Src = convertToScalableVector(SrcContainerVT, Src, DAG, Subtarget);
    }

    if (!VT.isVector() || VT.getVectorElementType() != MVT::f64 ||
        SrcVT.getVectorElementType() != MVT::f16) {
      // For scalable vectors, we only need to close the gap between
      // vXf16->vXf64.
      if (!VT.isFixedLengthVector())
        return Op;
      // For fixed-length vectors, lower the FP_EXTEND to a custom "VL" version.
      Src = getRVVFPExtendOrRound(Src, VT, ContainerVT, DL, DAG, Subtarget);
      return convertFromScalableVector(VT, Src, DAG, Subtarget);
    }

    MVT InterVT = VT.changeVectorElementType(MVT::f32);
    MVT InterContainerVT = ContainerVT.changeVectorElementType(MVT::f32);
    SDValue IntermediateExtend = getRVVFPExtendOrRound(
        Src, InterVT, InterContainerVT, DL, DAG, Subtarget);

    SDValue Extend = getRVVFPExtendOrRound(IntermediateExtend, VT, ContainerVT,
                                           DL, DAG, Subtarget);
    if (VT.isFixedLengthVector())
      return convertFromScalableVector(VT, Extend, DAG, Subtarget);
    return Extend;
  }
  case ISD::FP_ROUND: {
    // RVV can only do fp_round to types half the size as the source. We
    // custom-lower f64->f16 rounds via RVV's round-to-odd float
    // conversion instruction.
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();

    // Prepare any fixed-length vector operands.
    MVT ContainerVT = VT;
    if (VT.isFixedLengthVector()) {
      MVT SrcContainerVT = getContainerForFixedLengthVector(SrcVT);
      ContainerVT =
          SrcContainerVT.changeVectorElementType(VT.getVectorElementType());
      Src = convertToScalableVector(SrcContainerVT, Src, DAG, Subtarget);
    }

    if (!VT.isVector() || VT.getVectorElementType() != MVT::f16 ||
        SrcVT.getVectorElementType() != MVT::f64) {
      // For scalable vectors, we only need to close the gap between
      // vXf64<->vXf16.
      if (!VT.isFixedLengthVector())
        return Op;
      // For fixed-length vectors, lower the FP_ROUND to a custom "VL" version.
      Src = getRVVFPExtendOrRound(Src, VT, ContainerVT, DL, DAG, Subtarget);
      return convertFromScalableVector(VT, Src, DAG, Subtarget);
    }

    SDValue Mask, VL;
    std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

    MVT InterVT = ContainerVT.changeVectorElementType(MVT::f32);
    SDValue IntermediateRound =
        DAG.getNode(RISCVISD::VFNCVT_ROD_VL, DL, InterVT, Src, Mask, VL);
    SDValue Round = getRVVFPExtendOrRound(IntermediateRound, VT, ContainerVT,
                                          DL, DAG, Subtarget);

    if (VT.isFixedLengthVector())
      return convertFromScalableVector(VT, Round, DAG, Subtarget);
    return Round;
  }
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: {
    // RVV can only do fp<->int conversions to types half/double the size as
    // the source. We custom-lower any conversions that do two hops into
    // sequences.
    MVT VT = Op.getSimpleValueType();
    if (!VT.isVector())
      return Op;
    SDLoc DL(Op);
    SDValue Src = Op.getOperand(0);
    MVT EltVT = VT.getVectorElementType();
    MVT SrcVT = Src.getSimpleValueType();
    MVT SrcEltVT = SrcVT.getVectorElementType();
    unsigned EltSize = EltVT.getSizeInBits();
    unsigned SrcEltSize = SrcEltVT.getSizeInBits();
    assert(isPowerOf2_32(EltSize) && isPowerOf2_32(SrcEltSize) &&
           "Unexpected vector element types");

    bool IsInt2FP = SrcEltVT.isInteger();
    // Widening conversions
    if (EltSize > SrcEltSize && (EltSize / SrcEltSize >= 4)) {
      if (IsInt2FP) {
        // Do a regular integer sign/zero extension then convert to float.
        MVT IVecVT = MVT::getVectorVT(MVT::getIntegerVT(EltVT.getSizeInBits()),
                                      VT.getVectorElementCount());
        unsigned ExtOpcode = Op.getOpcode() == ISD::UINT_TO_FP
                                 ? ISD::ZERO_EXTEND
                                 : ISD::SIGN_EXTEND;
        SDValue Ext = DAG.getNode(ExtOpcode, DL, IVecVT, Src);
        return DAG.getNode(Op.getOpcode(), DL, VT, Ext);
      }
      // FP2Int
      assert(SrcEltVT == MVT::f16 && "Unexpected FP_TO_[US]INT lowering");
      // Do one doubling fp_extend then complete the operation by converting
      // to int.
      MVT InterimFVT = MVT::getVectorVT(MVT::f32, VT.getVectorElementCount());
      SDValue FExt = DAG.getFPExtendOrRound(Src, DL, InterimFVT);
      return DAG.getNode(Op.getOpcode(), DL, VT, FExt);
    }

    // Narrowing conversions
    if (SrcEltSize > EltSize && (SrcEltSize / EltSize >= 4)) {
      if (IsInt2FP) {
        // One narrowing int_to_fp, then an fp_round.
        assert(EltVT == MVT::f16 && "Unexpected [US]_TO_FP lowering");
        MVT InterimFVT = MVT::getVectorVT(MVT::f32, VT.getVectorElementCount());
        SDValue Int2FP = DAG.getNode(Op.getOpcode(), DL, InterimFVT, Src);
        return DAG.getFPExtendOrRound(Int2FP, DL, VT);
      }
      // FP2Int
      // One narrowing fp_to_int, then truncate the integer. If the float isn't
      // representable by the integer, the result is poison.
      MVT IVecVT =
          MVT::getVectorVT(MVT::getIntegerVT(SrcEltVT.getSizeInBits() / 2),
                           VT.getVectorElementCount());
      SDValue FP2Int = DAG.getNode(Op.getOpcode(), DL, IVecVT, Src);
      return DAG.getNode(ISD::TRUNCATE, DL, VT, FP2Int);
    }

    // Scalable vectors can exit here. Patterns will handle equally-sized
    // conversions halving/doubling ones.
    if (!VT.isFixedLengthVector())
      return Op;

    // For fixed-length vectors we lower to a custom "VL" node.
    unsigned RVVOpc = 0;
    switch (Op.getOpcode()) {
    default:
      llvm_unreachable("Impossible opcode");
    case ISD::FP_TO_SINT:
      RVVOpc = RISCVISD::FP_TO_SINT_VL;
      break;
    case ISD::FP_TO_UINT:
      RVVOpc = RISCVISD::FP_TO_UINT_VL;
      break;
    case ISD::SINT_TO_FP:
      RVVOpc = RISCVISD::SINT_TO_FP_VL;
      break;
    case ISD::UINT_TO_FP:
      RVVOpc = RISCVISD::UINT_TO_FP_VL;
      break;
    }

    MVT ContainerVT, SrcContainerVT;
    // Derive the reference container type from the larger vector type.
    if (SrcEltSize > EltSize) {
      SrcContainerVT = getContainerForFixedLengthVector(SrcVT);
      ContainerVT =
          SrcContainerVT.changeVectorElementType(VT.getVectorElementType());
    } else {
      ContainerVT = getContainerForFixedLengthVector(VT);
      SrcContainerVT = ContainerVT.changeVectorElementType(SrcEltVT);
    }

    SDValue Mask, VL;
    std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

    Src = convertToScalableVector(SrcContainerVT, Src, DAG, Subtarget);
    Src = DAG.getNode(RVVOpc, DL, ContainerVT, Src, Mask, VL);
    return convertFromScalableVector(VT, Src, DAG, Subtarget);
  }
  case ISD::FP_TO_SINT_SAT:
  case ISD::FP_TO_UINT_SAT:
    return lowerFP_TO_INT_SAT(Op, DAG, Subtarget);
  case ISD::FTRUNC:
  case ISD::FCEIL:
  case ISD::FFLOOR:
    return lowerFTRUNC_FCEIL_FFLOOR(Op, DAG);
  case ISD::FROUND:
    return lowerFROUND(Op, DAG);
  case ISD::VECREDUCE_ADD:
  case ISD::VECREDUCE_UMAX:
  case ISD::VECREDUCE_SMAX:
  case ISD::VECREDUCE_UMIN:
  case ISD::VECREDUCE_SMIN:
    return lowerVECREDUCE(Op, DAG);
  case ISD::VECREDUCE_AND:
  case ISD::VECREDUCE_OR:
  case ISD::VECREDUCE_XOR:
    if (Op.getOperand(0).getValueType().getVectorElementType() == MVT::i1)
      return lowerVectorMaskVecReduction(Op, DAG, /*IsVP*/ false);
    return lowerVECREDUCE(Op, DAG);
  case ISD::VECREDUCE_FADD:
  case ISD::VECREDUCE_SEQ_FADD:
  case ISD::VECREDUCE_FMIN:
  case ISD::VECREDUCE_FMAX:
    return lowerFPVECREDUCE(Op, DAG);
  case ISD::VP_REDUCE_ADD:
  case ISD::VP_REDUCE_UMAX:
  case ISD::VP_REDUCE_SMAX:
  case ISD::VP_REDUCE_UMIN:
  case ISD::VP_REDUCE_SMIN:
  case ISD::VP_REDUCE_FADD:
  case ISD::VP_REDUCE_SEQ_FADD:
  case ISD::VP_REDUCE_FMIN:
  case ISD::VP_REDUCE_FMAX:
    return lowerVPREDUCE(Op, DAG);
  case ISD::VP_REDUCE_AND:
  case ISD::VP_REDUCE_OR:
  case ISD::VP_REDUCE_XOR:
    if (Op.getOperand(1).getValueType().getVectorElementType() == MVT::i1)
      return lowerVectorMaskVecReduction(Op, DAG, /*IsVP*/ true);
    return lowerVPREDUCE(Op, DAG);
  case ISD::INSERT_SUBVECTOR:
    return lowerINSERT_SUBVECTOR(Op, DAG);
  case ISD::EXTRACT_SUBVECTOR:
    return lowerEXTRACT_SUBVECTOR(Op, DAG);
  case ISD::STEP_VECTOR:
    return lowerSTEP_VECTOR(Op, DAG);
  case ISD::VECTOR_REVERSE:
    return lowerVECTOR_REVERSE(Op, DAG);
  case ISD::VECTOR_SPLICE:
    return lowerVECTOR_SPLICE(Op, DAG);
  case ISD::BUILD_VECTOR:
    return lowerBUILD_VECTOR(Op, DAG, Subtarget);
  case ISD::SPLAT_VECTOR:
    if (Op.getValueType().getVectorElementType() == MVT::i1)
      return lowerVectorMaskSplat(Op, DAG);
    return lowerSPLAT_VECTOR(Op, DAG, Subtarget);
  case ISD::VECTOR_SHUFFLE:
    return lowerVECTOR_SHUFFLE(Op, DAG, Subtarget);
  case ISD::CONCAT_VECTORS: {
    // Split CONCAT_VECTORS into a series of INSERT_SUBVECTOR nodes. This is
    // better than going through the stack, as the default expansion does.
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    unsigned NumOpElts =
        Op.getOperand(0).getSimpleValueType().getVectorMinNumElements();
    SDValue Vec = DAG.getUNDEF(VT);
    for (const auto &OpIdx : enumerate(Op->ops())) {
      SDValue SubVec = OpIdx.value();
      // Don't insert undef subvectors.
      if (SubVec.isUndef())
        continue;
      Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, Vec, SubVec,
                        DAG.getIntPtrConstant(OpIdx.index() * NumOpElts, DL));
    }
    return Vec;
  }
  case ISD::LOAD:
    if (auto V = expandUnalignedRVVLoad(Op, DAG))
      return V;
    if (Op.getValueType().isFixedLengthVector())
      return lowerFixedLengthVectorLoadToRVV(Op, DAG);
    return Op;
  case ISD::STORE:
    if (auto V = expandUnalignedRVVStore(Op, DAG))
      return V;
    if (Op.getOperand(1).getValueType().isFixedLengthVector())
      return lowerFixedLengthVectorStoreToRVV(Op, DAG);
    return Op;
  case ISD::MLOAD:
  case ISD::VP_LOAD:
    return lowerMaskedLoad(Op, DAG);
  case ISD::MSTORE:
  case ISD::VP_STORE:
    return lowerMaskedStore(Op, DAG);
  case ISD::SETCC:
    return lowerFixedLengthVectorSetccToRVV(Op, DAG);
  case ISD::ADD:
    return lowerToScalableOp(Op, DAG, RISCVISD::ADD_VL);
  case ISD::SUB:
    return lowerToScalableOp(Op, DAG, RISCVISD::SUB_VL);
  case ISD::MUL:
    return lowerToScalableOp(Op, DAG, RISCVISD::MUL_VL);
  case ISD::MULHS:
    return lowerToScalableOp(Op, DAG, RISCVISD::MULHS_VL);
  case ISD::MULHU:
    return lowerToScalableOp(Op, DAG, RISCVISD::MULHU_VL);
  case ISD::AND:
    return lowerFixedLengthVectorLogicOpToRVV(Op, DAG, RISCVISD::VMAND_VL,
                                              RISCVISD::AND_VL);
  case ISD::OR:
    return lowerFixedLengthVectorLogicOpToRVV(Op, DAG, RISCVISD::VMOR_VL,
                                              RISCVISD::OR_VL);
  case ISD::XOR:
    return lowerFixedLengthVectorLogicOpToRVV(Op, DAG, RISCVISD::VMXOR_VL,
                                              RISCVISD::XOR_VL);
  case ISD::SDIV:
    return lowerToScalableOp(Op, DAG, RISCVISD::SDIV_VL);
  case ISD::SREM:
    return lowerToScalableOp(Op, DAG, RISCVISD::SREM_VL);
  case ISD::UDIV:
    return lowerToScalableOp(Op, DAG, RISCVISD::UDIV_VL);
  case ISD::UREM:
    return lowerToScalableOp(Op, DAG, RISCVISD::UREM_VL);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    if (Op.getSimpleValueType().isFixedLengthVector())
      return lowerFixedLengthVectorShiftToRVV(Op, DAG);
    // This can be called for an i32 shift amount that needs to be promoted.
    assert(Op.getOperand(1).getValueType() == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    return SDValue();
  case ISD::SADDSAT:
    return lowerToScalableOp(Op, DAG, RISCVISD::SADDSAT_VL);
  case ISD::UADDSAT:
    return lowerToScalableOp(Op, DAG, RISCVISD::UADDSAT_VL);
  case ISD::SSUBSAT:
    return lowerToScalableOp(Op, DAG, RISCVISD::SSUBSAT_VL);
  case ISD::USUBSAT:
    return lowerToScalableOp(Op, DAG, RISCVISD::USUBSAT_VL);
  case ISD::FADD:
    return lowerToScalableOp(Op, DAG, RISCVISD::FADD_VL);
  case ISD::FSUB:
    return lowerToScalableOp(Op, DAG, RISCVISD::FSUB_VL);
  case ISD::FMUL:
    return lowerToScalableOp(Op, DAG, RISCVISD::FMUL_VL);
  case ISD::FDIV:
    return lowerToScalableOp(Op, DAG, RISCVISD::FDIV_VL);
  case ISD::FNEG:
    return lowerToScalableOp(Op, DAG, RISCVISD::FNEG_VL);
  case ISD::FABS:
    return lowerToScalableOp(Op, DAG, RISCVISD::FABS_VL);
  case ISD::FSQRT:
    return lowerToScalableOp(Op, DAG, RISCVISD::FSQRT_VL);
  case ISD::FMA:
    return lowerToScalableOp(Op, DAG, RISCVISD::FMA_VL);
  case ISD::SMIN:
    return lowerToScalableOp(Op, DAG, RISCVISD::SMIN_VL);
  case ISD::SMAX:
    return lowerToScalableOp(Op, DAG, RISCVISD::SMAX_VL);
  case ISD::UMIN:
    return lowerToScalableOp(Op, DAG, RISCVISD::UMIN_VL);
  case ISD::UMAX:
    return lowerToScalableOp(Op, DAG, RISCVISD::UMAX_VL);
  case ISD::FMINNUM:
    return lowerToScalableOp(Op, DAG, RISCVISD::FMINNUM_VL);
  case ISD::FMAXNUM:
    return lowerToScalableOp(Op, DAG, RISCVISD::FMAXNUM_VL);
  case ISD::ABS:
    return lowerABS(Op, DAG);
  case ISD::CTLZ_ZERO_UNDEF:
  case ISD::CTTZ_ZERO_UNDEF:
    return lowerCTLZ_CTTZ_ZERO_UNDEF(Op, DAG);
  case ISD::VSELECT:
    return lowerFixedLengthVectorSelectToRVV(Op, DAG);
  case ISD::FCOPYSIGN:
    return lowerFixedLengthVectorFCOPYSIGNToRVV(Op, DAG);
  case ISD::MGATHER:
  case ISD::VP_GATHER:
    return lowerMaskedGather(Op, DAG);
  case ISD::MSCATTER:
  case ISD::VP_SCATTER:
    return lowerMaskedScatter(Op, DAG);
  case ISD::FLT_ROUNDS_:
    return lowerGET_ROUNDING(Op, DAG);
  case ISD::SET_ROUNDING:
    return lowerSET_ROUNDING(Op, DAG);
  case ISD::VP_SELECT:
    return lowerVPOp(Op, DAG, RISCVISD::VSELECT_VL);
  case ISD::VP_MERGE:
    return lowerVPOp(Op, DAG, RISCVISD::VP_MERGE_VL);
  case ISD::VP_ADD:
    return lowerVPOp(Op, DAG, RISCVISD::ADD_VL);
  case ISD::VP_SUB:
    return lowerVPOp(Op, DAG, RISCVISD::SUB_VL);
  case ISD::VP_MUL:
    return lowerVPOp(Op, DAG, RISCVISD::MUL_VL);
  case ISD::VP_SDIV:
    return lowerVPOp(Op, DAG, RISCVISD::SDIV_VL);
  case ISD::VP_UDIV:
    return lowerVPOp(Op, DAG, RISCVISD::UDIV_VL);
  case ISD::VP_SREM:
    return lowerVPOp(Op, DAG, RISCVISD::SREM_VL);
  case ISD::VP_UREM:
    return lowerVPOp(Op, DAG, RISCVISD::UREM_VL);
  case ISD::VP_AND:
    return lowerLogicVPOp(Op, DAG, RISCVISD::VMAND_VL, RISCVISD::AND_VL);
  case ISD::VP_OR:
    return lowerLogicVPOp(Op, DAG, RISCVISD::VMOR_VL, RISCVISD::OR_VL);
  case ISD::VP_XOR:
    return lowerLogicVPOp(Op, DAG, RISCVISD::VMXOR_VL, RISCVISD::XOR_VL);
  case ISD::VP_ASHR:
    return lowerVPOp(Op, DAG, RISCVISD::SRA_VL);
  case ISD::VP_LSHR:
    return lowerVPOp(Op, DAG, RISCVISD::SRL_VL);
  case ISD::VP_SHL:
    return lowerVPOp(Op, DAG, RISCVISD::SHL_VL);
  case ISD::VP_FADD:
    return lowerVPOp(Op, DAG, RISCVISD::FADD_VL);
  case ISD::VP_FSUB:
    return lowerVPOp(Op, DAG, RISCVISD::FSUB_VL);
  case ISD::VP_FMUL:
    return lowerVPOp(Op, DAG, RISCVISD::FMUL_VL);
  case ISD::VP_FDIV:
    return lowerVPOp(Op, DAG, RISCVISD::FDIV_VL);
  case ISD::VP_FNEG:
    return lowerVPOp(Op, DAG, RISCVISD::FNEG_VL);
  case ISD::VP_FMA:
    return lowerVPOp(Op, DAG, RISCVISD::FMA_VL);
  }
}

static SDValue getTargetNode(GlobalAddressSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, Flags);
}

static SDValue getTargetNode(BlockAddressSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetBlockAddress(N->getBlockAddress(), Ty, N->getOffset(),
                                   Flags);
}

static SDValue getTargetNode(ConstantPoolSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetConstantPool(N->getConstVal(), Ty, N->getAlign(),
                                   N->getOffset(), Flags);
}

static SDValue getTargetNode(JumpTableSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetJumpTable(N->getIndex(), Ty, Flags);
}

template <class NodeTy>
SDValue RISCVTargetLowering::getAddr(NodeTy *N, SelectionDAG &DAG,
                                     bool IsLocal) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());

  if (isPositionIndependent()) {
    SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);
    if (IsLocal)
      // Use PC-relative addressing to access the symbol. This generates the
      // pattern (PseudoLLA sym), which expands to (addi (auipc %pcrel_hi(sym))
      // %pcrel_lo(auipc)).
      return SDValue(DAG.getMachineNode(RISCV::PseudoLLA, DL, Ty, Addr), 0);

    // Use PC-relative addressing to access the GOT for this symbol, then load
    // the address from the GOT. This generates the pattern (PseudoLA sym),
    // which expands to (ld (addi (auipc %got_pcrel_hi(sym)) %pcrel_lo(auipc))).
    return SDValue(DAG.getMachineNode(RISCV::PseudoLA, DL, Ty, Addr), 0);
  }

  switch (getTargetMachine().getCodeModel()) {
  default:
    report_fatal_error("Unsupported code model for lowering");
  case CodeModel::Small: {
    // Generate a sequence for accessing addresses within the first 2 GiB of
    // address space. This generates the pattern (addi (lui %hi(sym)) %lo(sym)).
    SDValue AddrHi = getTargetNode(N, DL, Ty, DAG, RISCVII::MO_HI);
    SDValue AddrLo = getTargetNode(N, DL, Ty, DAG, RISCVII::MO_LO);
    SDValue MNHi = SDValue(DAG.getMachineNode(RISCV::LUI, DL, Ty, AddrHi), 0);
    return SDValue(DAG.getMachineNode(RISCV::ADDI, DL, Ty, MNHi, AddrLo), 0);
  }
  case CodeModel::Medium: {
    // Generate a sequence for accessing addresses within any 2GiB range within
    // the address space. This generates the pattern (PseudoLLA sym), which
    // expands to (addi (auipc %pcrel_hi(sym)) %pcrel_lo(auipc)).
    SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);
    return SDValue(DAG.getMachineNode(RISCV::PseudoLLA, DL, Ty, Addr), 0);
  }
  }
}

SDValue RISCVTargetLowering::lowerGlobalAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  const GlobalValue *GV = N->getGlobal();
  bool IsLocal = getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);
  SDValue Addr = getAddr(N, DAG, IsLocal);

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, XLenVT));
  return Addr;
}

SDValue RISCVTargetLowering::lowerBlockAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  BlockAddressSDNode *N = cast<BlockAddressSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue RISCVTargetLowering::lowerConstantPool(SDValue Op,
                                               SelectionDAG &DAG) const {
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue RISCVTargetLowering::lowerJumpTable(SDValue Op,
                                            SelectionDAG &DAG) const {
  JumpTableSDNode *N = cast<JumpTableSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue RISCVTargetLowering::getStaticTLSAddr(GlobalAddressSDNode *N,
                                              SelectionDAG &DAG,
                                              bool UseGOT) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  const GlobalValue *GV = N->getGlobal();
  MVT XLenVT = Subtarget.getXLenVT();

  if (UseGOT) {
    // Use PC-relative addressing to access the GOT for this TLS symbol, then
    // load the address from the GOT and add the thread pointer. This generates
    // the pattern (PseudoLA_TLS_IE sym), which expands to
    // (ld (auipc %tls_ie_pcrel_hi(sym)) %pcrel_lo(auipc)).
    SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, Ty, 0, 0);
    SDValue Load =
        SDValue(DAG.getMachineNode(RISCV::PseudoLA_TLS_IE, DL, Ty, Addr), 0);

    // Add the thread pointer.
    SDValue TPReg = DAG.getRegister(RISCV::X4, XLenVT);
    return DAG.getNode(ISD::ADD, DL, Ty, Load, TPReg);
  }

  // Generate a sequence for accessing the address relative to the thread
  // pointer, with the appropriate adjustment for the thread pointer offset.
  // This generates the pattern
  // (add (add_tprel (lui %tprel_hi(sym)) tp %tprel_add(sym)) %tprel_lo(sym))
  SDValue AddrHi =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_HI);
  SDValue AddrAdd =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_ADD);
  SDValue AddrLo =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_LO);

  SDValue MNHi = SDValue(DAG.getMachineNode(RISCV::LUI, DL, Ty, AddrHi), 0);
  SDValue TPReg = DAG.getRegister(RISCV::X4, XLenVT);
  SDValue MNAdd = SDValue(
      DAG.getMachineNode(RISCV::PseudoAddTPRel, DL, Ty, MNHi, TPReg, AddrAdd),
      0);
  return SDValue(DAG.getMachineNode(RISCV::ADDI, DL, Ty, MNAdd, AddrLo), 0);
}

SDValue RISCVTargetLowering::getDynamicTLSAddr(GlobalAddressSDNode *N,
                                               SelectionDAG &DAG) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  IntegerType *CallTy = Type::getIntNTy(*DAG.getContext(), Ty.getSizeInBits());
  const GlobalValue *GV = N->getGlobal();

  // Use a PC-relative addressing mode to access the global dynamic GOT address.
  // This generates the pattern (PseudoLA_TLS_GD sym), which expands to
  // (addi (auipc %tls_gd_pcrel_hi(sym)) %pcrel_lo(auipc)).
  SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, Ty, 0, 0);
  SDValue Load =
      SDValue(DAG.getMachineNode(RISCV::PseudoLA_TLS_GD, DL, Ty, Addr), 0);

  // Prepare argument list to generate call.
  ArgListTy Args;
  ArgListEntry Entry;
  Entry.Node = Load;
  Entry.Ty = CallTy;
  Args.push_back(Entry);

  // Setup call to __tls_get_addr.
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(DL)
      .setChain(DAG.getEntryNode())
      .setLibCallee(CallingConv::C, CallTy,
                    DAG.getExternalSymbol("__tls_get_addr", Ty),
                    std::move(Args));

  return LowerCallTo(CLI).first;
}

SDValue RISCVTargetLowering::lowerGlobalTLSAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  TLSModel::Model Model = getTargetMachine().getTLSModel(N->getGlobal());

  if (DAG.getMachineFunction().getFunction().getCallingConv() ==
      CallingConv::GHC)
    report_fatal_error("In GHC calling convention TLS is not supported");

  SDValue Addr;
  switch (Model) {
  case TLSModel::LocalExec:
    Addr = getStaticTLSAddr(N, DAG, /*UseGOT=*/false);
    break;
  case TLSModel::InitialExec:
    Addr = getStaticTLSAddr(N, DAG, /*UseGOT=*/true);
    break;
  case TLSModel::LocalDynamic:
  case TLSModel::GeneralDynamic:
    Addr = getDynamicTLSAddr(N, DAG);
    break;
  }

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, XLenVT));
  return Addr;
}

SDValue RISCVTargetLowering::lowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  SDValue CondV = Op.getOperand(0);
  SDValue TrueV = Op.getOperand(1);
  SDValue FalseV = Op.getOperand(2);
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  // Lower vector SELECTs to VSELECTs by splatting the condition.
  if (VT.isVector()) {
    MVT SplatCondVT = VT.changeVectorElementType(MVT::i1);
    SDValue CondSplat = VT.isScalableVector()
                            ? DAG.getSplatVector(SplatCondVT, DL, CondV)
                            : DAG.getSplatBuildVector(SplatCondVT, DL, CondV);
    return DAG.getNode(ISD::VSELECT, DL, VT, CondSplat, TrueV, FalseV);
  }

  // If the result type is XLenVT and CondV is the output of a SETCC node
  // which also operated on XLenVT inputs, then merge the SETCC node into the
  // lowered RISCVISD::SELECT_CC to take advantage of the integer
  // compare+branch instructions. i.e.:
  // (select (setcc lhs, rhs, cc), truev, falsev)
  // -> (riscvisd::select_cc lhs, rhs, cc, truev, falsev)
  if (VT == XLenVT && CondV.getOpcode() == ISD::SETCC &&
      CondV.getOperand(0).getSimpleValueType() == XLenVT) {
    SDValue LHS = CondV.getOperand(0);
    SDValue RHS = CondV.getOperand(1);
    const auto *CC = cast<CondCodeSDNode>(CondV.getOperand(2));
    ISD::CondCode CCVal = CC->get();

    // Special case for a select of 2 constants that have a diffence of 1.
    // Normally this is done by DAGCombine, but if the select is introduced by
    // type legalization or op legalization, we miss it. Restricting to SETLT
    // case for now because that is what signed saturating add/sub need.
    // FIXME: We don't need the condition to be SETLT or even a SETCC,
    // but we would probably want to swap the true/false values if the condition
    // is SETGE/SETLE to avoid an XORI.
    if (isa<ConstantSDNode>(TrueV) && isa<ConstantSDNode>(FalseV) &&
        CCVal == ISD::SETLT) {
      const APInt &TrueVal = cast<ConstantSDNode>(TrueV)->getAPIntValue();
      const APInt &FalseVal = cast<ConstantSDNode>(FalseV)->getAPIntValue();
      if (TrueVal - 1 == FalseVal)
        return DAG.getNode(ISD::ADD, DL, Op.getValueType(), CondV, FalseV);
      if (TrueVal + 1 == FalseVal)
        return DAG.getNode(ISD::SUB, DL, Op.getValueType(), FalseV, CondV);
    }

    translateSetCCForBranch(DL, LHS, RHS, CCVal, DAG);

    SDValue TargetCC = DAG.getCondCode(CCVal);
    SDValue Ops[] = {LHS, RHS, TargetCC, TrueV, FalseV};
    return DAG.getNode(RISCVISD::SELECT_CC, DL, Op.getValueType(), Ops);
  }

  // Otherwise:
  // (select condv, truev, falsev)
  // -> (riscvisd::select_cc condv, zero, setne, truev, falsev)
  SDValue Zero = DAG.getConstant(0, DL, XLenVT);
  SDValue SetNE = DAG.getCondCode(ISD::SETNE);

  SDValue Ops[] = {CondV, Zero, SetNE, TrueV, FalseV};

  return DAG.getNode(RISCVISD::SELECT_CC, DL, Op.getValueType(), Ops);
}

SDValue RISCVTargetLowering::lowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  SDValue CondV = Op.getOperand(1);
  SDLoc DL(Op);
  MVT XLenVT = Subtarget.getXLenVT();

  if (CondV.getOpcode() == ISD::SETCC &&
      CondV.getOperand(0).getValueType() == XLenVT) {
    SDValue LHS = CondV.getOperand(0);
    SDValue RHS = CondV.getOperand(1);
    ISD::CondCode CCVal = cast<CondCodeSDNode>(CondV.getOperand(2))->get();

    translateSetCCForBranch(DL, LHS, RHS, CCVal, DAG);

    SDValue TargetCC = DAG.getCondCode(CCVal);
    return DAG.getNode(RISCVISD::BR_CC, DL, Op.getValueType(), Op.getOperand(0),
                       LHS, RHS, TargetCC, Op.getOperand(2));
  }

  return DAG.getNode(RISCVISD::BR_CC, DL, Op.getValueType(), Op.getOperand(0),
                     CondV, DAG.getConstant(0, DL, XLenVT),
                     DAG.getCondCode(ISD::SETNE), Op.getOperand(2));
}

SDValue RISCVTargetLowering::lowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  RISCVMachineFunctionInfo *FuncInfo = MF.getInfo<RISCVMachineFunctionInfo>();

  SDLoc DL(Op);
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy(MF.getDataLayout()));

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, FI, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue RISCVTargetLowering::lowerFRAMEADDR(SDValue Op,
                                            SelectionDAG &DAG) const {
  const RISCVRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setFrameAddressIsTaken(true);
  Register FrameReg = RI.getFrameRegister(MF);
  int XLenInBytes = Subtarget.getXLen() / 8;

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), DL, FrameReg, VT);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  while (Depth--) {
    int Offset = -(XLenInBytes * 2);
    SDValue Ptr = DAG.getNode(ISD::ADD, DL, VT, FrameAddr,
                              DAG.getIntPtrConstant(Offset, DL));
    FrameAddr =
        DAG.getLoad(VT, DL, DAG.getEntryNode(), Ptr, MachinePointerInfo());
  }
  return FrameAddr;
}

SDValue RISCVTargetLowering::lowerRETURNADDR(SDValue Op,
                                             SelectionDAG &DAG) const {
  const RISCVRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setReturnAddressIsTaken(true);
  MVT XLenVT = Subtarget.getXLenVT();
  int XLenInBytes = Subtarget.getXLen() / 8;

  if (verifyReturnAddressArgumentIsConstant(Op, DAG))
    return SDValue();

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (Depth) {
    int Off = -XLenInBytes;
    SDValue FrameAddr = lowerFRAMEADDR(Op, DAG);
    SDValue Offset = DAG.getConstant(Off, DL, VT);
    return DAG.getLoad(VT, DL, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, DL, VT, FrameAddr, Offset),
                       MachinePointerInfo());
  }

  // Return the value of the return address register, marking it an implicit
  // live-in.
  Register Reg = MF.addLiveIn(RI.getRARegister(), getRegClassFor(XLenVT));
  return DAG.getCopyFromReg(DAG.getEntryNode(), DL, Reg, XLenVT);
}

SDValue RISCVTargetLowering::lowerShiftLeftParts(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // if Shamt-XLEN < 0: // Shamt < XLEN
  //   Lo = Lo << Shamt
  //   Hi = (Hi << Shamt) | ((Lo >>u 1) >>u (XLEN-1 ^ Shamt))
  // else:
  //   Lo = 0
  //   Hi = Lo << (Shamt-XLEN)

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusXLen = DAG.getConstant(-(int)Subtarget.getXLen(), DL, VT);
  SDValue XLenMinus1 = DAG.getConstant(Subtarget.getXLen() - 1, DL, VT);
  SDValue ShamtMinusXLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusXLen);
  SDValue XLenMinus1Shamt = DAG.getNode(ISD::XOR, DL, VT, Shamt, XLenMinus1);

  SDValue LoTrue = DAG.getNode(ISD::SHL, DL, VT, Lo, Shamt);
  SDValue ShiftRight1Lo = DAG.getNode(ISD::SRL, DL, VT, Lo, One);
  SDValue ShiftRightLo =
      DAG.getNode(ISD::SRL, DL, VT, ShiftRight1Lo, XLenMinus1Shamt);
  SDValue ShiftLeftHi = DAG.getNode(ISD::SHL, DL, VT, Hi, Shamt);
  SDValue HiTrue = DAG.getNode(ISD::OR, DL, VT, ShiftLeftHi, ShiftRightLo);
  SDValue HiFalse = DAG.getNode(ISD::SHL, DL, VT, Lo, ShamtMinusXLen);

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusXLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, Zero);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

SDValue RISCVTargetLowering::lowerShiftRightParts(SDValue Op, SelectionDAG &DAG,
                                                  bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // SRA expansion:
  //   if Shamt-XLEN < 0: // Shamt < XLEN
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (ShAmt ^ XLEN-1))
  //     Hi = Hi >>s Shamt
  //   else:
  //     Lo = Hi >>s (Shamt-XLEN);
  //     Hi = Hi >>s (XLEN-1)
  //
  // SRL expansion:
  //   if Shamt-XLEN < 0: // Shamt < XLEN
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (ShAmt ^ XLEN-1))
  //     Hi = Hi >>u Shamt
  //   else:
  //     Lo = Hi >>u (Shamt-XLEN);
  //     Hi = 0;

  unsigned ShiftRightOp = IsSRA ? ISD::SRA : ISD::SRL;

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusXLen = DAG.getConstant(-(int)Subtarget.getXLen(), DL, VT);
  SDValue XLenMinus1 = DAG.getConstant(Subtarget.getXLen() - 1, DL, VT);
  SDValue ShamtMinusXLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusXLen);
  SDValue XLenMinus1Shamt = DAG.getNode(ISD::XOR, DL, VT, Shamt, XLenMinus1);

  SDValue ShiftRightLo = DAG.getNode(ISD::SRL, DL, VT, Lo, Shamt);
  SDValue ShiftLeftHi1 = DAG.getNode(ISD::SHL, DL, VT, Hi, One);
  SDValue ShiftLeftHi =
      DAG.getNode(ISD::SHL, DL, VT, ShiftLeftHi1, XLenMinus1Shamt);
  SDValue LoTrue = DAG.getNode(ISD::OR, DL, VT, ShiftRightLo, ShiftLeftHi);
  SDValue HiTrue = DAG.getNode(ShiftRightOp, DL, VT, Hi, Shamt);
  SDValue LoFalse = DAG.getNode(ShiftRightOp, DL, VT, Hi, ShamtMinusXLen);
  SDValue HiFalse =
      IsSRA ? DAG.getNode(ISD::SRA, DL, VT, Hi, XLenMinus1) : Zero;

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusXLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, LoFalse);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

// Lower splats of i1 types to SETCC. For each mask vector type, we have a
// legal equivalently-sized i8 type, so we can use that as a go-between.
SDValue RISCVTargetLowering::lowerVectorMaskSplat(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  SDValue SplatVal = Op.getOperand(0);
  // All-zeros or all-ones splats are handled specially.
  if (ISD::isConstantSplatVectorAllOnes(Op.getNode())) {
    SDValue VL = getDefaultScalableVLOps(VT, DL, DAG, Subtarget).second;
    return DAG.getNode(RISCVISD::VMSET_VL, DL, VT, VL);
  }
  if (ISD::isConstantSplatVectorAllZeros(Op.getNode())) {
    SDValue VL = getDefaultScalableVLOps(VT, DL, DAG, Subtarget).second;
    return DAG.getNode(RISCVISD::VMCLR_VL, DL, VT, VL);
  }
  MVT XLenVT = Subtarget.getXLenVT();
  assert(SplatVal.getValueType() == XLenVT &&
         "Unexpected type for i1 splat value");
  MVT InterVT = VT.changeVectorElementType(MVT::i8);
  SplatVal = DAG.getNode(ISD::AND, DL, XLenVT, SplatVal,
                         DAG.getConstant(1, DL, XLenVT));
  SDValue LHS = DAG.getSplatVector(InterVT, DL, SplatVal);
  SDValue Zero = DAG.getConstant(0, DL, InterVT);
  return DAG.getSetCC(DL, VT, LHS, Zero, ISD::SETNE);
}

// Custom-lower a SPLAT_VECTOR_PARTS where XLEN<SEW, as the SEW element type is
// illegal (currently only vXi64 RV32).
// FIXME: We could also catch non-constant sign-extended i32 values and lower
// them to VMV_V_X_VL.
SDValue RISCVTargetLowering::lowerSPLAT_VECTOR_PARTS(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VecVT = Op.getSimpleValueType();
  assert(!Subtarget.is64Bit() && VecVT.getVectorElementType() == MVT::i64 &&
         "Unexpected SPLAT_VECTOR_PARTS lowering");

  assert(Op.getNumOperands() == 2 && "Unexpected number of operands!");
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);

  if (VecVT.isFixedLengthVector()) {
    MVT ContainerVT = getContainerForFixedLengthVector(VecVT);
    SDLoc DL(Op);
    SDValue Mask, VL;
    std::tie(Mask, VL) =
        getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

    SDValue Res =
        splatPartsI64WithVL(DL, ContainerVT, SDValue(), Lo, Hi, VL, DAG);
    return convertFromScalableVector(VecVT, Res, DAG, Subtarget);
  }

  if (isa<ConstantSDNode>(Lo) && isa<ConstantSDNode>(Hi)) {
    int32_t LoC = cast<ConstantSDNode>(Lo)->getSExtValue();
    int32_t HiC = cast<ConstantSDNode>(Hi)->getSExtValue();
    // If Hi constant is all the same sign bit as Lo, lower this as a custom
    // node in order to try and match RVV vector/scalar instructions.
    if ((LoC >> 31) == HiC)
      return DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VecVT, DAG.getUNDEF(VecVT),
                         Lo, DAG.getRegister(RISCV::X0, MVT::i32));
  }

  // Detect cases where Hi is (SRA Lo, 31) which means Hi is Lo sign extended.
  if (Hi.getOpcode() == ISD::SRA && Hi.getOperand(0) == Lo &&
      isa<ConstantSDNode>(Hi.getOperand(1)) &&
      Hi.getConstantOperandVal(1) == 31)
    return DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VecVT, DAG.getUNDEF(VecVT), Lo,
                       DAG.getRegister(RISCV::X0, MVT::i32));

  // Fall back to use a stack store and stride x0 vector load. Use X0 as VL.
  return DAG.getNode(RISCVISD::SPLAT_VECTOR_SPLIT_I64_VL, DL, VecVT,
                     DAG.getUNDEF(VecVT), Lo, Hi,
                     DAG.getRegister(RISCV::X0, MVT::i32));
}

// Custom-lower extensions from mask vectors by using a vselect either with 1
// for zero/any-extension or -1 for sign-extension:
//   (vXiN = (s|z)ext vXi1:vmask) -> (vXiN = vselect vmask, (-1 or 1), 0)
// Note that any-extension is lowered identically to zero-extension.
SDValue RISCVTargetLowering::lowerVectorMaskExt(SDValue Op, SelectionDAG &DAG,
                                                int64_t ExtTrueVal) const {
  SDLoc DL(Op);
  MVT VecVT = Op.getSimpleValueType();
  SDValue Src = Op.getOperand(0);
  // Only custom-lower extensions from mask types
  assert(Src.getValueType().isVector() &&
         Src.getValueType().getVectorElementType() == MVT::i1);

  MVT XLenVT = Subtarget.getXLenVT();
  SDValue SplatZero = DAG.getConstant(0, DL, XLenVT);
  SDValue SplatTrueVal = DAG.getConstant(ExtTrueVal, DL, XLenVT);

  if (VecVT.isScalableVector()) {
    // Be careful not to introduce illegal scalar types at this stage, and be
    // careful also about splatting constants as on RV32, vXi64 SPLAT_VECTOR is
    // illegal and must be expanded. Since we know that the constants are
    // sign-extended 32-bit values, we use VMV_V_X_VL directly.
    bool IsRV32E64 =
        !Subtarget.is64Bit() && VecVT.getVectorElementType() == MVT::i64;

    if (!IsRV32E64) {
      SplatZero = DAG.getSplatVector(VecVT, DL, SplatZero);
      SplatTrueVal = DAG.getSplatVector(VecVT, DL, SplatTrueVal);
    } else {
      SplatZero =
          DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VecVT, DAG.getUNDEF(VecVT),
                      SplatZero, DAG.getRegister(RISCV::X0, XLenVT));
      SplatTrueVal =
          DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VecVT, DAG.getUNDEF(VecVT),
                      SplatTrueVal, DAG.getRegister(RISCV::X0, XLenVT));
    }

    return DAG.getNode(ISD::VSELECT, DL, VecVT, Src, SplatTrueVal, SplatZero);
  }

  MVT ContainerVT = getContainerForFixedLengthVector(VecVT);
  MVT I1ContainerVT =
      MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());

  SDValue CC = convertToScalableVector(I1ContainerVT, Src, DAG, Subtarget);

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  SplatZero = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, ContainerVT,
                          DAG.getUNDEF(ContainerVT), SplatZero, VL);
  SplatTrueVal = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, ContainerVT,
                             DAG.getUNDEF(ContainerVT), SplatTrueVal, VL);
  SDValue Select = DAG.getNode(RISCVISD::VSELECT_VL, DL, ContainerVT, CC,
                               SplatTrueVal, SplatZero, VL);

  return convertFromScalableVector(VecVT, Select, DAG, Subtarget);
}

SDValue RISCVTargetLowering::lowerFixedLengthVectorExtendToRVV(
    SDValue Op, SelectionDAG &DAG, unsigned ExtendOpc) const {
  MVT ExtVT = Op.getSimpleValueType();
  // Only custom-lower extensions from fixed-length vector types.
  if (!ExtVT.isFixedLengthVector())
    return Op;
  MVT VT = Op.getOperand(0).getSimpleValueType();
  // Grab the canonical container type for the extended type. Infer the smaller
  // type from that to ensure the same number of vector elements, as we know
  // the LMUL will be sufficient to hold the smaller type.
  MVT ContainerExtVT = getContainerForFixedLengthVector(ExtVT);
  // Get the extended container type manually to ensure the same number of
  // vector elements between source and dest.
  MVT ContainerVT = MVT::getVectorVT(VT.getVectorElementType(),
                                     ContainerExtVT.getVectorElementCount());

  SDValue Op1 =
      convertToScalableVector(ContainerVT, Op.getOperand(0), DAG, Subtarget);

  SDLoc DL(Op);
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  SDValue Ext = DAG.getNode(ExtendOpc, DL, ContainerExtVT, Op1, Mask, VL);

  return convertFromScalableVector(ExtVT, Ext, DAG, Subtarget);
}

// Custom-lower truncations from vectors to mask vectors by using a mask and a
// setcc operation:
//   (vXi1 = trunc vXiN vec) -> (vXi1 = setcc (and vec, 1), 0, ne)
SDValue RISCVTargetLowering::lowerVectorMaskTrunc(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT MaskVT = Op.getValueType();
  // Only expect to custom-lower truncations to mask types
  assert(MaskVT.isVector() && MaskVT.getVectorElementType() == MVT::i1 &&
         "Unexpected type for vector mask lowering");
  SDValue Src = Op.getOperand(0);
  MVT VecVT = Src.getSimpleValueType();

  // If this is a fixed vector, we need to convert it to a scalable vector.
  MVT ContainerVT = VecVT;
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    Src = convertToScalableVector(ContainerVT, Src, DAG, Subtarget);
  }

  SDValue SplatOne = DAG.getConstant(1, DL, Subtarget.getXLenVT());
  SDValue SplatZero = DAG.getConstant(0, DL, Subtarget.getXLenVT());

  SplatOne = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, ContainerVT,
                         DAG.getUNDEF(ContainerVT), SplatOne);
  SplatZero = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, ContainerVT,
                          DAG.getUNDEF(ContainerVT), SplatZero);

  if (VecVT.isScalableVector()) {
    SDValue Trunc = DAG.getNode(ISD::AND, DL, VecVT, Src, SplatOne);
    return DAG.getSetCC(DL, MaskVT, Trunc, SplatZero, ISD::SETNE);
  }

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  MVT MaskContainerVT = ContainerVT.changeVectorElementType(MVT::i1);
  SDValue Trunc =
      DAG.getNode(RISCVISD::AND_VL, DL, ContainerVT, Src, SplatOne, Mask, VL);
  Trunc = DAG.getNode(RISCVISD::SETCC_VL, DL, MaskContainerVT, Trunc, SplatZero,
                      DAG.getCondCode(ISD::SETNE), Mask, VL);
  return convertFromScalableVector(MaskVT, Trunc, DAG, Subtarget);
}

// Custom-legalize INSERT_VECTOR_ELT so that the value is inserted into the
// first position of a vector, and that vector is slid up to the insert index.
// By limiting the active vector length to index+1 and merging with the
// original vector (with an undisturbed tail policy for elements >= VL), we
// achieve the desired result of leaving all elements untouched except the one
// at VL-1, which is replaced with the desired value.
SDValue RISCVTargetLowering::lowerINSERT_VECTOR_ELT(SDValue Op,
                                                    SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VecVT = Op.getSimpleValueType();
  SDValue Vec = Op.getOperand(0);
  SDValue Val = Op.getOperand(1);
  SDValue Idx = Op.getOperand(2);

  if (VecVT.getVectorElementType() == MVT::i1) {
    // FIXME: For now we just promote to an i8 vector and insert into that,
    // but this is probably not optimal.
    MVT WideVT = MVT::getVectorVT(MVT::i8, VecVT.getVectorElementCount());
    Vec = DAG.getNode(ISD::ZERO_EXTEND, DL, WideVT, Vec);
    Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, WideVT, Vec, Val, Idx);
    return DAG.getNode(ISD::TRUNCATE, DL, VecVT, Vec);
  }

  MVT ContainerVT = VecVT;
  // If the operand is a fixed-length vector, convert to a scalable one.
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  }

  MVT XLenVT = Subtarget.getXLenVT();

  SDValue Zero = DAG.getConstant(0, DL, XLenVT);
  bool IsLegalInsert = Subtarget.is64Bit() || Val.getValueType() != MVT::i64;
  // Even i64-element vectors on RV32 can be lowered without scalar
  // legalization if the most-significant 32 bits of the value are not affected
  // by the sign-extension of the lower 32 bits.
  // TODO: We could also catch sign extensions of a 32-bit value.
  if (!IsLegalInsert && isa<ConstantSDNode>(Val)) {
    const auto *CVal = cast<ConstantSDNode>(Val);
    if (isInt<32>(CVal->getSExtValue())) {
      IsLegalInsert = true;
      Val = DAG.getConstant(CVal->getSExtValue(), DL, MVT::i32);
    }
  }

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  SDValue ValInVec;

  if (IsLegalInsert) {
    unsigned Opc =
        VecVT.isFloatingPoint() ? RISCVISD::VFMV_S_F_VL : RISCVISD::VMV_S_X_VL;
    if (isNullConstant(Idx)) {
      Vec = DAG.getNode(Opc, DL, ContainerVT, Vec, Val, VL);
      if (!VecVT.isFixedLengthVector())
        return Vec;
      return convertFromScalableVector(VecVT, Vec, DAG, Subtarget);
    }
    ValInVec =
        DAG.getNode(Opc, DL, ContainerVT, DAG.getUNDEF(ContainerVT), Val, VL);
  } else {
    // On RV32, i64-element vectors must be specially handled to place the
    // value at element 0, by using two vslide1up instructions in sequence on
    // the i32 split lo/hi value. Use an equivalently-sized i32 vector for
    // this.
    SDValue One = DAG.getConstant(1, DL, XLenVT);
    SDValue ValLo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Val, Zero);
    SDValue ValHi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Val, One);
    MVT I32ContainerVT =
        MVT::getVectorVT(MVT::i32, ContainerVT.getVectorElementCount() * 2);
    SDValue I32Mask =
        getDefaultScalableVLOps(I32ContainerVT, DL, DAG, Subtarget).first;
    // Limit the active VL to two.
    SDValue InsertI64VL = DAG.getConstant(2, DL, XLenVT);
    // Note: We can't pass a UNDEF to the first VSLIDE1UP_VL since an untied
    // undef doesn't obey the earlyclobber constraint. Just splat a zero value.
    ValInVec = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, I32ContainerVT,
                           DAG.getUNDEF(I32ContainerVT), Zero, InsertI64VL);
    // First slide in the hi value, then the lo in underneath it.
    ValInVec = DAG.getNode(RISCVISD::VSLIDE1UP_VL, DL, I32ContainerVT,
                           DAG.getUNDEF(I32ContainerVT), ValInVec, ValHi,
                           I32Mask, InsertI64VL);
    ValInVec = DAG.getNode(RISCVISD::VSLIDE1UP_VL, DL, I32ContainerVT,
                           DAG.getUNDEF(I32ContainerVT), ValInVec, ValLo,
                           I32Mask, InsertI64VL);
    // Bitcast back to the right container type.
    ValInVec = DAG.getBitcast(ContainerVT, ValInVec);
  }

  // Now that the value is in a vector, slide it into position.
  SDValue InsertVL =
      DAG.getNode(ISD::ADD, DL, XLenVT, Idx, DAG.getConstant(1, DL, XLenVT));
  SDValue Slideup = DAG.getNode(RISCVISD::VSLIDEUP_VL, DL, ContainerVT, Vec,
                                ValInVec, Idx, Mask, InsertVL);
  if (!VecVT.isFixedLengthVector())
    return Slideup;
  return convertFromScalableVector(VecVT, Slideup, DAG, Subtarget);
}

// Custom-lower EXTRACT_VECTOR_ELT operations to slide the vector down, then
// extract the first element: (extractelt (slidedown vec, idx), 0). For integer
// types this is done using VMV_X_S to allow us to glean information about the
// sign bits of the result.
SDValue RISCVTargetLowering::lowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Idx = Op.getOperand(1);
  SDValue Vec = Op.getOperand(0);
  EVT EltVT = Op.getValueType();
  MVT VecVT = Vec.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  if (VecVT.getVectorElementType() == MVT::i1) {
    if (VecVT.isFixedLengthVector()) {
      unsigned NumElts = VecVT.getVectorNumElements();
      if (NumElts >= 8) {
        MVT WideEltVT;
        unsigned WidenVecLen;
        SDValue ExtractElementIdx;
        SDValue ExtractBitIdx;
        unsigned MaxEEW = Subtarget.getMaxELENForFixedLengthVectors();
        MVT LargestEltVT = MVT::getIntegerVT(
            std::min(MaxEEW, unsigned(XLenVT.getSizeInBits())));
        if (NumElts <= LargestEltVT.getSizeInBits()) {
          assert(isPowerOf2_32(NumElts) &&
                 "the number of elements should be power of 2");
          WideEltVT = MVT::getIntegerVT(NumElts);
          WidenVecLen = 1;
          ExtractElementIdx = DAG.getConstant(0, DL, XLenVT);
          ExtractBitIdx = Idx;
        } else {
          WideEltVT = LargestEltVT;
          WidenVecLen = NumElts / WideEltVT.getSizeInBits();
          // extract element index = index / element width
          ExtractElementIdx = DAG.getNode(
              ISD::SRL, DL, XLenVT, Idx,
              DAG.getConstant(Log2_64(WideEltVT.getSizeInBits()), DL, XLenVT));
          // mask bit index = index % element width
          ExtractBitIdx = DAG.getNode(
              ISD::AND, DL, XLenVT, Idx,
              DAG.getConstant(WideEltVT.getSizeInBits() - 1, DL, XLenVT));
        }
        MVT WideVT = MVT::getVectorVT(WideEltVT, WidenVecLen);
        Vec = DAG.getNode(ISD::BITCAST, DL, WideVT, Vec);
        SDValue ExtractElt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, XLenVT,
                                         Vec, ExtractElementIdx);
        // Extract the bit from GPR.
        SDValue ShiftRight =
            DAG.getNode(ISD::SRL, DL, XLenVT, ExtractElt, ExtractBitIdx);
        return DAG.getNode(ISD::AND, DL, XLenVT, ShiftRight,
                           DAG.getConstant(1, DL, XLenVT));
      }
    }
    // Otherwise, promote to an i8 vector and extract from that.
    MVT WideVT = MVT::getVectorVT(MVT::i8, VecVT.getVectorElementCount());
    Vec = DAG.getNode(ISD::ZERO_EXTEND, DL, WideVT, Vec);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Vec, Idx);
  }

  // If this is a fixed vector, we need to convert it to a scalable vector.
  MVT ContainerVT = VecVT;
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  }

  // If the index is 0, the vector is already in the right position.
  if (!isNullConstant(Idx)) {
    // Use a VL of 1 to avoid processing more elements than we need.
    SDValue VL = DAG.getConstant(1, DL, XLenVT);
    MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
    SDValue Mask = DAG.getNode(RISCVISD::VMSET_VL, DL, MaskVT, VL);
    Vec = DAG.getNode(RISCVISD::VSLIDEDOWN_VL, DL, ContainerVT,
                      DAG.getUNDEF(ContainerVT), Vec, Idx, Mask, VL);
  }

  if (!EltVT.isInteger()) {
    // Floating-point extracts are handled in TableGen.
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Vec,
                       DAG.getConstant(0, DL, XLenVT));
  }

  SDValue Elt0 = DAG.getNode(RISCVISD::VMV_X_S, DL, XLenVT, Vec);
  return DAG.getNode(ISD::TRUNCATE, DL, EltVT, Elt0);
}

// Some RVV intrinsics may claim that they want an integer operand to be
// promoted or expanded.
static SDValue lowerVectorIntrinsicScalars(SDValue Op, SelectionDAG &DAG,
                                           const RISCVSubtarget &Subtarget) {
  assert((Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN) &&
         "Unexpected opcode");

  if (!Subtarget.hasVInstructions())
    return SDValue();

  bool HasChain = Op.getOpcode() == ISD::INTRINSIC_W_CHAIN;
  unsigned IntNo = Op.getConstantOperandVal(HasChain ? 1 : 0);
  SDLoc DL(Op);

  const RISCVVIntrinsicsTable::RISCVVIntrinsicInfo *II =
      RISCVVIntrinsicsTable::getRISCVVIntrinsicInfo(IntNo);
  if (!II || !II->hasScalarOperand())
    return SDValue();

  unsigned SplatOp = II->ScalarOperand + 1 + HasChain;
  assert(SplatOp < Op.getNumOperands());

  SmallVector<SDValue, 8> Operands(Op->op_begin(), Op->op_end());
  SDValue &ScalarOp = Operands[SplatOp];
  MVT OpVT = ScalarOp.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  // If this isn't a scalar, or its type is XLenVT we're done.
  if (!OpVT.isScalarInteger() || OpVT == XLenVT)
    return SDValue();

  // Simplest case is that the operand needs to be promoted to XLenVT.
  if (OpVT.bitsLT(XLenVT)) {
    // If the operand is a constant, sign extend to increase our chances
    // of being able to use a .vi instruction. ANY_EXTEND would become a
    // a zero extend and the simm5 check in isel would fail.
    // FIXME: Should we ignore the upper bits in isel instead?
    unsigned ExtOpc =
        isa<ConstantSDNode>(ScalarOp) ? ISD::SIGN_EXTEND : ISD::ANY_EXTEND;
    ScalarOp = DAG.getNode(ExtOpc, DL, XLenVT, ScalarOp);
    return DAG.getNode(Op->getOpcode(), DL, Op->getVTList(), Operands);
  }

  // Use the previous operand to get the vXi64 VT. The result might be a mask
  // VT for compares. Using the previous operand assumes that the previous
  // operand will never have a smaller element size than a scalar operand and
  // that a widening operation never uses SEW=64.
  // NOTE: If this fails the below assert, we can probably just find the
  // element count from any operand or result and use it to construct the VT.
  assert(II->ScalarOperand > 0 && "Unexpected splat operand!");
  MVT VT = Op.getOperand(SplatOp - 1).getSimpleValueType();

  // The more complex case is when the scalar is larger than XLenVT.
  assert(XLenVT == MVT::i32 && OpVT == MVT::i64 &&
         VT.getVectorElementType() == MVT::i64 && "Unexpected VTs!");

  // If this is a sign-extended 32-bit constant, we can truncate it and rely
  // on the instruction to sign-extend since SEW>XLEN.
  if (auto *CVal = dyn_cast<ConstantSDNode>(ScalarOp)) {
    if (isInt<32>(CVal->getSExtValue())) {
      ScalarOp = DAG.getConstant(CVal->getSExtValue(), DL, MVT::i32);
      return DAG.getNode(Op->getOpcode(), DL, Op->getVTList(), Operands);
    }
  }

  switch (IntNo) {
  case Intrinsic::riscv_vslide1up:
  case Intrinsic::riscv_vslide1down:
  case Intrinsic::riscv_vslide1up_mask:
  case Intrinsic::riscv_vslide1down_mask: {
    // We need to special case these when the scalar is larger than XLen.
    unsigned NumOps = Op.getNumOperands();
    bool IsMasked = NumOps == 7;

    // Convert the vector source to the equivalent nxvXi32 vector.
    MVT I32VT = MVT::getVectorVT(MVT::i32, VT.getVectorElementCount() * 2);
    SDValue Vec = DAG.getBitcast(I32VT, Operands[2]);

    SDValue ScalarLo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, ScalarOp,
                                   DAG.getConstant(0, DL, XLenVT));
    SDValue ScalarHi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, ScalarOp,
                                   DAG.getConstant(1, DL, XLenVT));

    // Double the VL since we halved SEW.
    SDValue AVL = getVLOperand(Op);
    SDValue I32VL;

    // Optimize for constant AVL
    if (isa<ConstantSDNode>(AVL)) {
      unsigned EltSize = VT.getScalarSizeInBits();
      unsigned MinSize = VT.getSizeInBits().getKnownMinValue();

      unsigned VectorBitsMax = Subtarget.getRealMaxVLen();
      unsigned MaxVLMAX =
          RISCVTargetLowering::computeVLMAX(VectorBitsMax, EltSize, MinSize);

      unsigned VectorBitsMin = Subtarget.getRealMinVLen();
      unsigned MinVLMAX =
          RISCVTargetLowering::computeVLMAX(VectorBitsMin, EltSize, MinSize);

      uint64_t AVLInt = cast<ConstantSDNode>(AVL)->getZExtValue();
      if (AVLInt <= MinVLMAX) {
        I32VL = DAG.getConstant(2 * AVLInt, DL, XLenVT);
      } else if (AVLInt >= 2 * MaxVLMAX) {
        // Just set vl to VLMAX in this situation
        RISCVII::VLMUL Lmul = RISCVTargetLowering::getLMUL(I32VT);
        SDValue LMUL = DAG.getConstant(Lmul, DL, XLenVT);
        unsigned Sew = RISCVVType::encodeSEW(I32VT.getScalarSizeInBits());
        SDValue SEW = DAG.getConstant(Sew, DL, XLenVT);
        SDValue SETVLMAX = DAG.getTargetConstant(
            Intrinsic::riscv_vsetvlimax_opt, DL, MVT::i32);
        I32VL = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, XLenVT, SETVLMAX, SEW,
                            LMUL);
      } else {
        // For AVL between (MinVLMAX, 2 * MaxVLMAX), the actual working vl
        // is related to the hardware implementation.
        // So let the following code handle
      }
    }
    if (!I32VL) {
      RISCVII::VLMUL Lmul = RISCVTargetLowering::getLMUL(VT);
      SDValue LMUL = DAG.getConstant(Lmul, DL, XLenVT);
      unsigned Sew = RISCVVType::encodeSEW(VT.getScalarSizeInBits());
      SDValue SEW = DAG.getConstant(Sew, DL, XLenVT);
      SDValue SETVL =
          DAG.getTargetConstant(Intrinsic::riscv_vsetvli_opt, DL, MVT::i32);
      // Using vsetvli instruction to get actually used length which related to
      // the hardware implementation
      SDValue VL = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, XLenVT, SETVL, AVL,
                               SEW, LMUL);
      I32VL =
          DAG.getNode(ISD::SHL, DL, XLenVT, VL, DAG.getConstant(1, DL, XLenVT));
    }

    MVT I32MaskVT = MVT::getVectorVT(MVT::i1, I32VT.getVectorElementCount());
    SDValue I32Mask = DAG.getNode(RISCVISD::VMSET_VL, DL, I32MaskVT, I32VL);

    // Shift the two scalar parts in using SEW=32 slide1up/slide1down
    // instructions.
    SDValue Passthru;
    if (IsMasked)
      Passthru = DAG.getUNDEF(I32VT);
    else
      Passthru = DAG.getBitcast(I32VT, Operands[1]);

    if (IntNo == Intrinsic::riscv_vslide1up ||
        IntNo == Intrinsic::riscv_vslide1up_mask) {
      Vec = DAG.getNode(RISCVISD::VSLIDE1UP_VL, DL, I32VT, Passthru, Vec,
                        ScalarHi, I32Mask, I32VL);
      Vec = DAG.getNode(RISCVISD::VSLIDE1UP_VL, DL, I32VT, Passthru, Vec,
                        ScalarLo, I32Mask, I32VL);
    } else {
      Vec = DAG.getNode(RISCVISD::VSLIDE1DOWN_VL, DL, I32VT, Passthru, Vec,
                        ScalarLo, I32Mask, I32VL);
      Vec = DAG.getNode(RISCVISD::VSLIDE1DOWN_VL, DL, I32VT, Passthru, Vec,
                        ScalarHi, I32Mask, I32VL);
    }

    // Convert back to nxvXi64.
    Vec = DAG.getBitcast(VT, Vec);

    if (!IsMasked)
      return Vec;
    // Apply mask after the operation.
    SDValue Mask = Operands[NumOps - 3];
    SDValue MaskedOff = Operands[1];
    // Assume Policy operand is the last operand.
    uint64_t Policy =
        cast<ConstantSDNode>(Operands[NumOps - 1])->getZExtValue();
    // We don't need to select maskedoff if it's undef.
    if (MaskedOff.isUndef())
      return Vec;
    // TAMU
    if (Policy == RISCVII::TAIL_AGNOSTIC)
      return DAG.getNode(RISCVISD::VSELECT_VL, DL, VT, Mask, Vec, MaskedOff,
                         AVL);
    // TUMA or TUMU: Currently we always emit tumu policy regardless of tuma.
    // It's fine because vmerge does not care mask policy.
    return DAG.getNode(RISCVISD::VP_MERGE_VL, DL, VT, Mask, Vec, MaskedOff,
                       AVL);
  }
  }

  // We need to convert the scalar to a splat vector.
  // FIXME: Can we implicitly truncate the scalar if it is known to
  // be sign extended?
  SDValue VL = getVLOperand(Op);
  assert(VL.getValueType() == XLenVT);
  ScalarOp = splatSplitI64WithVL(DL, VT, SDValue(), ScalarOp, VL, DAG);
  return DAG.getNode(Op->getOpcode(), DL, Op->getVTList(), Operands);
}

SDValue RISCVTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                     SelectionDAG &DAG) const {
  unsigned IntNo = Op.getConstantOperandVal(0);
  SDLoc DL(Op);
  MVT XLenVT = Subtarget.getXLenVT();

  switch (IntNo) {
  default:
    break; // Don't custom lower most intrinsics.
  case Intrinsic::thread_pointer: {
    EVT PtrVT = getPointerTy(DAG.getDataLayout());
    return DAG.getRegister(RISCV::X4, PtrVT);
  }
  case Intrinsic::riscv_orc_b:
  case Intrinsic::riscv_brev8: {
    // Lower to the GORCI encoding for orc.b or the GREVI encoding for brev8.
    unsigned Opc =
        IntNo == Intrinsic::riscv_brev8 ? RISCVISD::GREV : RISCVISD::GORC;
    return DAG.getNode(Opc, DL, XLenVT, Op.getOperand(1),
                       DAG.getConstant(7, DL, XLenVT));
  }
  case Intrinsic::riscv_grev:
  case Intrinsic::riscv_gorc: {
    unsigned Opc =
        IntNo == Intrinsic::riscv_grev ? RISCVISD::GREV : RISCVISD::GORC;
    return DAG.getNode(Opc, DL, XLenVT, Op.getOperand(1), Op.getOperand(2));
  }
  case Intrinsic::riscv_zip:
  case Intrinsic::riscv_unzip: {
    // Lower to the SHFLI encoding for zip or the UNSHFLI encoding for unzip.
    // For i32 the immediate is 15. For i64 the immediate is 31.
    unsigned Opc =
        IntNo == Intrinsic::riscv_zip ? RISCVISD::SHFL : RISCVISD::UNSHFL;
    unsigned BitWidth = Op.getValueSizeInBits();
    assert(isPowerOf2_32(BitWidth) && BitWidth >= 2 && "Unexpected bit width");
    return DAG.getNode(Opc, DL, XLenVT, Op.getOperand(1),
                       DAG.getConstant((BitWidth / 2) - 1, DL, XLenVT));
  }
  case Intrinsic::riscv_shfl:
  case Intrinsic::riscv_unshfl: {
    unsigned Opc =
        IntNo == Intrinsic::riscv_shfl ? RISCVISD::SHFL : RISCVISD::UNSHFL;
    return DAG.getNode(Opc, DL, XLenVT, Op.getOperand(1), Op.getOperand(2));
  }
  case Intrinsic::riscv_bcompress:
  case Intrinsic::riscv_bdecompress: {
    unsigned Opc = IntNo == Intrinsic::riscv_bcompress ? RISCVISD::BCOMPRESS
                                                       : RISCVISD::BDECOMPRESS;
    return DAG.getNode(Opc, DL, XLenVT, Op.getOperand(1), Op.getOperand(2));
  }
  case Intrinsic::riscv_bfp:
    return DAG.getNode(RISCVISD::BFP, DL, XLenVT, Op.getOperand(1),
                       Op.getOperand(2));
  case Intrinsic::riscv_fsl:
    return DAG.getNode(RISCVISD::FSL, DL, XLenVT, Op.getOperand(1),
                       Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::riscv_fsr:
    return DAG.getNode(RISCVISD::FSR, DL, XLenVT, Op.getOperand(1),
                       Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::riscv_vmv_x_s:
    assert(Op.getValueType() == XLenVT && "Unexpected VT!");
    return DAG.getNode(RISCVISD::VMV_X_S, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_vmv_v_x:
    return lowerScalarSplat(Op.getOperand(1), Op.getOperand(2),
                            Op.getOperand(3), Op.getSimpleValueType(), DL, DAG,
                            Subtarget);
  case Intrinsic::riscv_vfmv_v_f:
    return DAG.getNode(RISCVISD::VFMV_V_F_VL, DL, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::riscv_vmv_s_x: {
    SDValue Scalar = Op.getOperand(2);

    if (Scalar.getValueType().bitsLE(XLenVT)) {
      Scalar = DAG.getNode(ISD::ANY_EXTEND, DL, XLenVT, Scalar);
      return DAG.getNode(RISCVISD::VMV_S_X_VL, DL, Op.getValueType(),
                         Op.getOperand(1), Scalar, Op.getOperand(3));
    }

    assert(Scalar.getValueType() == MVT::i64 && "Unexpected scalar VT!");

    // This is an i64 value that lives in two scalar registers. We have to
    // insert this in a convoluted way. First we build vXi64 splat containing
    // the/ two values that we assemble using some bit math. Next we'll use
    // vid.v and vmseq to build a mask with bit 0 set. Then we'll use that mask
    // to merge element 0 from our splat into the source vector.
    // FIXME: This is probably not the best way to do this, but it is
    // consistent with INSERT_VECTOR_ELT lowering so it is a good starting
    // point.
    //   sw lo, (a0)
    //   sw hi, 4(a0)
    //   vlse vX, (a0)
    //
    //   vid.v      vVid
    //   vmseq.vx   mMask, vVid, 0
    //   vmerge.vvm vDest, vSrc, vVal, mMask
    MVT VT = Op.getSimpleValueType();
    SDValue Vec = Op.getOperand(1);
    SDValue VL = getVLOperand(Op);

    SDValue SplattedVal = splatSplitI64WithVL(DL, VT, SDValue(), Scalar, VL, DAG);
    if (Op.getOperand(1).isUndef())
      return SplattedVal;
    SDValue SplattedIdx =
        DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VT, DAG.getUNDEF(VT),
                    DAG.getConstant(0, DL, MVT::i32), VL);

    MVT MaskVT = MVT::getVectorVT(MVT::i1, VT.getVectorElementCount());
    SDValue Mask = DAG.getNode(RISCVISD::VMSET_VL, DL, MaskVT, VL);
    SDValue VID = DAG.getNode(RISCVISD::VID_VL, DL, VT, Mask, VL);
    SDValue SelectCond =
        DAG.getNode(RISCVISD::SETCC_VL, DL, MaskVT, VID, SplattedIdx,
                    DAG.getCondCode(ISD::SETEQ), Mask, VL);
    return DAG.getNode(RISCVISD::VSELECT_VL, DL, VT, SelectCond, SplattedVal,
                       Vec, VL);
  }
  }

  return lowerVectorIntrinsicScalars(Op, DAG, Subtarget);
}

SDValue RISCVTargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op,
                                                    SelectionDAG &DAG) const {
  unsigned IntNo = Op.getConstantOperandVal(1);
  switch (IntNo) {
  default:
    break;
  case Intrinsic::riscv_masked_strided_load: {
    SDLoc DL(Op);
    MVT XLenVT = Subtarget.getXLenVT();

    // If the mask is known to be all ones, optimize to an unmasked intrinsic;
    // the selection of the masked intrinsics doesn't do this for us.
    SDValue Mask = Op.getOperand(5);
    bool IsUnmasked = ISD::isConstantSplatVectorAllOnes(Mask.getNode());

    MVT VT = Op->getSimpleValueType(0);
    MVT ContainerVT = getContainerForFixedLengthVector(VT);

    SDValue PassThru = Op.getOperand(2);
    if (!IsUnmasked) {
      MVT MaskVT =
          MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
      Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
      PassThru = convertToScalableVector(ContainerVT, PassThru, DAG, Subtarget);
    }

    SDValue VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);

    SDValue IntID = DAG.getTargetConstant(
        IsUnmasked ? Intrinsic::riscv_vlse : Intrinsic::riscv_vlse_mask, DL,
        XLenVT);

    auto *Load = cast<MemIntrinsicSDNode>(Op);
    SmallVector<SDValue, 8> Ops{Load->getChain(), IntID};
    if (IsUnmasked)
      Ops.push_back(DAG.getUNDEF(ContainerVT));
    else
      Ops.push_back(PassThru);
    Ops.push_back(Op.getOperand(3)); // Ptr
    Ops.push_back(Op.getOperand(4)); // Stride
    if (!IsUnmasked)
      Ops.push_back(Mask);
    Ops.push_back(VL);
    if (!IsUnmasked) {
      SDValue Policy = DAG.getTargetConstant(RISCVII::TAIL_AGNOSTIC, DL, XLenVT);
      Ops.push_back(Policy);
    }

    SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});
    SDValue Result =
        DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops,
                                Load->getMemoryVT(), Load->getMemOperand());
    SDValue Chain = Result.getValue(1);
    Result = convertFromScalableVector(VT, Result, DAG, Subtarget);
    return DAG.getMergeValues({Result, Chain}, DL);
  }
  case Intrinsic::riscv_seg2_load:
  case Intrinsic::riscv_seg3_load:
  case Intrinsic::riscv_seg4_load:
  case Intrinsic::riscv_seg5_load:
  case Intrinsic::riscv_seg6_load:
  case Intrinsic::riscv_seg7_load:
  case Intrinsic::riscv_seg8_load: {
    SDLoc DL(Op);
    static const Intrinsic::ID VlsegInts[7] = {
        Intrinsic::riscv_vlseg2, Intrinsic::riscv_vlseg3,
        Intrinsic::riscv_vlseg4, Intrinsic::riscv_vlseg5,
        Intrinsic::riscv_vlseg6, Intrinsic::riscv_vlseg7,
        Intrinsic::riscv_vlseg8};
    unsigned NF = Op->getNumValues() - 1;
    assert(NF >= 2 && NF <= 8 && "Unexpected seg number");
    MVT XLenVT = Subtarget.getXLenVT();
    MVT VT = Op->getSimpleValueType(0);
    MVT ContainerVT = getContainerForFixedLengthVector(VT);

    SDValue VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);
    SDValue IntID = DAG.getTargetConstant(VlsegInts[NF - 2], DL, XLenVT);
    auto *Load = cast<MemIntrinsicSDNode>(Op);
    SmallVector<EVT, 9> ContainerVTs(NF, ContainerVT);
    ContainerVTs.push_back(MVT::Other);
    SDVTList VTs = DAG.getVTList(ContainerVTs);
    SDValue Result =
        DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs,
                                {Load->getChain(), IntID, Op.getOperand(2), VL},
                                Load->getMemoryVT(), Load->getMemOperand());
    SmallVector<SDValue, 9> Results;
    for (unsigned int RetIdx = 0; RetIdx < NF; RetIdx++)
      Results.push_back(convertFromScalableVector(VT, Result.getValue(RetIdx),
                                                  DAG, Subtarget));
    Results.push_back(Result.getValue(NF));
    return DAG.getMergeValues(Results, DL);
  }
  }

  return lowerVectorIntrinsicScalars(Op, DAG, Subtarget);
}

SDValue RISCVTargetLowering::LowerINTRINSIC_VOID(SDValue Op,
                                                 SelectionDAG &DAG) const {
  unsigned IntNo = Op.getConstantOperandVal(1);
  switch (IntNo) {
  default:
    break;
  case Intrinsic::riscv_masked_strided_store: {
    SDLoc DL(Op);
    MVT XLenVT = Subtarget.getXLenVT();

    // If the mask is known to be all ones, optimize to an unmasked intrinsic;
    // the selection of the masked intrinsics doesn't do this for us.
    SDValue Mask = Op.getOperand(5);
    bool IsUnmasked = ISD::isConstantSplatVectorAllOnes(Mask.getNode());

    SDValue Val = Op.getOperand(2);
    MVT VT = Val.getSimpleValueType();
    MVT ContainerVT = getContainerForFixedLengthVector(VT);

    Val = convertToScalableVector(ContainerVT, Val, DAG, Subtarget);
    if (!IsUnmasked) {
      MVT MaskVT =
          MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
      Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
    }

    SDValue VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);

    SDValue IntID = DAG.getTargetConstant(
        IsUnmasked ? Intrinsic::riscv_vsse : Intrinsic::riscv_vsse_mask, DL,
        XLenVT);

    auto *Store = cast<MemIntrinsicSDNode>(Op);
    SmallVector<SDValue, 8> Ops{Store->getChain(), IntID};
    Ops.push_back(Val);
    Ops.push_back(Op.getOperand(3)); // Ptr
    Ops.push_back(Op.getOperand(4)); // Stride
    if (!IsUnmasked)
      Ops.push_back(Mask);
    Ops.push_back(VL);

    return DAG.getMemIntrinsicNode(ISD::INTRINSIC_VOID, DL, Store->getVTList(),
                                   Ops, Store->getMemoryVT(),
                                   Store->getMemOperand());
  }
  }

  return SDValue();
}

static MVT getLMUL1VT(MVT VT) {
  assert(VT.getVectorElementType().getSizeInBits() <= 64 &&
         "Unexpected vector MVT");
  return MVT::getScalableVectorVT(
      VT.getVectorElementType(),
      RISCV::RVVBitsPerBlock / VT.getVectorElementType().getSizeInBits());
}

static unsigned getRVVReductionOp(unsigned ISDOpcode) {
  switch (ISDOpcode) {
  default:
    llvm_unreachable("Unhandled reduction");
  case ISD::VECREDUCE_ADD:
    return RISCVISD::VECREDUCE_ADD_VL;
  case ISD::VECREDUCE_UMAX:
    return RISCVISD::VECREDUCE_UMAX_VL;
  case ISD::VECREDUCE_SMAX:
    return RISCVISD::VECREDUCE_SMAX_VL;
  case ISD::VECREDUCE_UMIN:
    return RISCVISD::VECREDUCE_UMIN_VL;
  case ISD::VECREDUCE_SMIN:
    return RISCVISD::VECREDUCE_SMIN_VL;
  case ISD::VECREDUCE_AND:
    return RISCVISD::VECREDUCE_AND_VL;
  case ISD::VECREDUCE_OR:
    return RISCVISD::VECREDUCE_OR_VL;
  case ISD::VECREDUCE_XOR:
    return RISCVISD::VECREDUCE_XOR_VL;
  }
}

SDValue RISCVTargetLowering::lowerVectorMaskVecReduction(SDValue Op,
                                                         SelectionDAG &DAG,
                                                         bool IsVP) const {
  SDLoc DL(Op);
  SDValue Vec = Op.getOperand(IsVP ? 1 : 0);
  MVT VecVT = Vec.getSimpleValueType();
  assert((Op.getOpcode() == ISD::VECREDUCE_AND ||
          Op.getOpcode() == ISD::VECREDUCE_OR ||
          Op.getOpcode() == ISD::VECREDUCE_XOR ||
          Op.getOpcode() == ISD::VP_REDUCE_AND ||
          Op.getOpcode() == ISD::VP_REDUCE_OR ||
          Op.getOpcode() == ISD::VP_REDUCE_XOR) &&
         "Unexpected reduction lowering");

  MVT XLenVT = Subtarget.getXLenVT();
  assert(Op.getValueType() == XLenVT &&
         "Expected reduction output to be legalized to XLenVT");

  MVT ContainerVT = VecVT;
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  }

  SDValue Mask, VL;
  if (IsVP) {
    Mask = Op.getOperand(2);
    VL = Op.getOperand(3);
  } else {
    std::tie(Mask, VL) =
        getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);
  }

  unsigned BaseOpc;
  ISD::CondCode CC;
  SDValue Zero = DAG.getConstant(0, DL, XLenVT);

  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("Unhandled reduction");
  case ISD::VECREDUCE_AND:
  case ISD::VP_REDUCE_AND: {
    // vcpop ~x == 0
    SDValue TrueMask = DAG.getNode(RISCVISD::VMSET_VL, DL, ContainerVT, VL);
    Vec = DAG.getNode(RISCVISD::VMXOR_VL, DL, ContainerVT, Vec, TrueMask, VL);
    Vec = DAG.getNode(RISCVISD::VCPOP_VL, DL, XLenVT, Vec, Mask, VL);
    CC = ISD::SETEQ;
    BaseOpc = ISD::AND;
    break;
  }
  case ISD::VECREDUCE_OR:
  case ISD::VP_REDUCE_OR:
    // vcpop x != 0
    Vec = DAG.getNode(RISCVISD::VCPOP_VL, DL, XLenVT, Vec, Mask, VL);
    CC = ISD::SETNE;
    BaseOpc = ISD::OR;
    break;
  case ISD::VECREDUCE_XOR:
  case ISD::VP_REDUCE_XOR: {
    // ((vcpop x) & 1) != 0
    SDValue One = DAG.getConstant(1, DL, XLenVT);
    Vec = DAG.getNode(RISCVISD::VCPOP_VL, DL, XLenVT, Vec, Mask, VL);
    Vec = DAG.getNode(ISD::AND, DL, XLenVT, Vec, One);
    CC = ISD::SETNE;
    BaseOpc = ISD::XOR;
    break;
  }
  }

  SDValue SetCC = DAG.getSetCC(DL, XLenVT, Vec, Zero, CC);

  if (!IsVP)
    return SetCC;

  // Now include the start value in the operation.
  // Note that we must return the start value when no elements are operated
  // upon. The vcpop instructions we've emitted in each case above will return
  // 0 for an inactive vector, and so we've already received the neutral value:
  // AND gives us (0 == 0) -> 1 and OR/XOR give us (0 != 0) -> 0. Therefore we
  // can simply include the start value.
  return DAG.getNode(BaseOpc, DL, XLenVT, SetCC, Op.getOperand(0));
}

SDValue RISCVTargetLowering::lowerVECREDUCE(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Vec = Op.getOperand(0);
  EVT VecEVT = Vec.getValueType();

  unsigned BaseOpc = ISD::getVecReduceBaseOpcode(Op.getOpcode());

  // Due to ordering in legalize types we may have a vector type that needs to
  // be split. Do that manually so we can get down to a legal type.
  while (getTypeAction(*DAG.getContext(), VecEVT) ==
         TargetLowering::TypeSplitVector) {
    SDValue Lo, Hi;
    std::tie(Lo, Hi) = DAG.SplitVector(Vec, DL);
    VecEVT = Lo.getValueType();
    Vec = DAG.getNode(BaseOpc, DL, VecEVT, Lo, Hi);
  }

  // TODO: The type may need to be widened rather than split. Or widened before
  // it can be split.
  if (!isTypeLegal(VecEVT))
    return SDValue();

  MVT VecVT = VecEVT.getSimpleVT();
  MVT VecEltVT = VecVT.getVectorElementType();
  unsigned RVVOpcode = getRVVReductionOp(Op.getOpcode());

  MVT ContainerVT = VecVT;
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  }

  MVT M1VT = getLMUL1VT(ContainerVT);
  MVT XLenVT = Subtarget.getXLenVT();

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  SDValue NeutralElem =
      DAG.getNeutralElement(BaseOpc, DL, VecEltVT, SDNodeFlags());
  SDValue IdentitySplat =
      lowerScalarSplat(SDValue(), NeutralElem, DAG.getConstant(1, DL, XLenVT),
                       M1VT, DL, DAG, Subtarget);
  SDValue Reduction = DAG.getNode(RVVOpcode, DL, M1VT, DAG.getUNDEF(M1VT), Vec,
                                  IdentitySplat, Mask, VL);
  SDValue Elt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VecEltVT, Reduction,
                             DAG.getConstant(0, DL, XLenVT));
  return DAG.getSExtOrTrunc(Elt0, DL, Op.getValueType());
}

// Given a reduction op, this function returns the matching reduction opcode,
// the vector SDValue and the scalar SDValue required to lower this to a
// RISCVISD node.
static std::tuple<unsigned, SDValue, SDValue>
getRVVFPReductionOpAndOperands(SDValue Op, SelectionDAG &DAG, EVT EltVT) {
  SDLoc DL(Op);
  auto Flags = Op->getFlags();
  unsigned Opcode = Op.getOpcode();
  unsigned BaseOpcode = ISD::getVecReduceBaseOpcode(Opcode);
  switch (Opcode) {
  default:
    llvm_unreachable("Unhandled reduction");
  case ISD::VECREDUCE_FADD: {
    // Use positive zero if we can. It is cheaper to materialize.
    SDValue Zero =
        DAG.getConstantFP(Flags.hasNoSignedZeros() ? 0.0 : -0.0, DL, EltVT);
    return std::make_tuple(RISCVISD::VECREDUCE_FADD_VL, Op.getOperand(0), Zero);
  }
  case ISD::VECREDUCE_SEQ_FADD:
    return std::make_tuple(RISCVISD::VECREDUCE_SEQ_FADD_VL, Op.getOperand(1),
                           Op.getOperand(0));
  case ISD::VECREDUCE_FMIN:
    return std::make_tuple(RISCVISD::VECREDUCE_FMIN_VL, Op.getOperand(0),
                           DAG.getNeutralElement(BaseOpcode, DL, EltVT, Flags));
  case ISD::VECREDUCE_FMAX:
    return std::make_tuple(RISCVISD::VECREDUCE_FMAX_VL, Op.getOperand(0),
                           DAG.getNeutralElement(BaseOpcode, DL, EltVT, Flags));
  }
}

SDValue RISCVTargetLowering::lowerFPVECREDUCE(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VecEltVT = Op.getSimpleValueType();

  unsigned RVVOpcode;
  SDValue VectorVal, ScalarVal;
  std::tie(RVVOpcode, VectorVal, ScalarVal) =
      getRVVFPReductionOpAndOperands(Op, DAG, VecEltVT);
  MVT VecVT = VectorVal.getSimpleValueType();

  MVT ContainerVT = VecVT;
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    VectorVal = convertToScalableVector(ContainerVT, VectorVal, DAG, Subtarget);
  }

  MVT M1VT = getLMUL1VT(VectorVal.getSimpleValueType());
  MVT XLenVT = Subtarget.getXLenVT();

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget);

  SDValue ScalarSplat =
      lowerScalarSplat(SDValue(), ScalarVal, DAG.getConstant(1, DL, XLenVT),
                       M1VT, DL, DAG, Subtarget);
  SDValue Reduction = DAG.getNode(RVVOpcode, DL, M1VT, DAG.getUNDEF(M1VT),
                                  VectorVal, ScalarSplat, Mask, VL);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VecEltVT, Reduction,
                     DAG.getConstant(0, DL, XLenVT));
}

static unsigned getRVVVPReductionOp(unsigned ISDOpcode) {
  switch (ISDOpcode) {
  default:
    llvm_unreachable("Unhandled reduction");
  case ISD::VP_REDUCE_ADD:
    return RISCVISD::VECREDUCE_ADD_VL;
  case ISD::VP_REDUCE_UMAX:
    return RISCVISD::VECREDUCE_UMAX_VL;
  case ISD::VP_REDUCE_SMAX:
    return RISCVISD::VECREDUCE_SMAX_VL;
  case ISD::VP_REDUCE_UMIN:
    return RISCVISD::VECREDUCE_UMIN_VL;
  case ISD::VP_REDUCE_SMIN:
    return RISCVISD::VECREDUCE_SMIN_VL;
  case ISD::VP_REDUCE_AND:
    return RISCVISD::VECREDUCE_AND_VL;
  case ISD::VP_REDUCE_OR:
    return RISCVISD::VECREDUCE_OR_VL;
  case ISD::VP_REDUCE_XOR:
    return RISCVISD::VECREDUCE_XOR_VL;
  case ISD::VP_REDUCE_FADD:
    return RISCVISD::VECREDUCE_FADD_VL;
  case ISD::VP_REDUCE_SEQ_FADD:
    return RISCVISD::VECREDUCE_SEQ_FADD_VL;
  case ISD::VP_REDUCE_FMAX:
    return RISCVISD::VECREDUCE_FMAX_VL;
  case ISD::VP_REDUCE_FMIN:
    return RISCVISD::VECREDUCE_FMIN_VL;
  }
}

SDValue RISCVTargetLowering::lowerVPREDUCE(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Vec = Op.getOperand(1);
  EVT VecEVT = Vec.getValueType();

  // TODO: The type may need to be widened rather than split. Or widened before
  // it can be split.
  if (!isTypeLegal(VecEVT))
    return SDValue();

  MVT VecVT = VecEVT.getSimpleVT();
  MVT VecEltVT = VecVT.getVectorElementType();
  unsigned RVVOpcode = getRVVVPReductionOp(Op.getOpcode());

  MVT ContainerVT = VecVT;
  if (VecVT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VecVT);
    Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
  }

  SDValue VL = Op.getOperand(3);
  SDValue Mask = Op.getOperand(2);

  MVT M1VT = getLMUL1VT(ContainerVT);
  MVT XLenVT = Subtarget.getXLenVT();
  MVT ResVT = !VecVT.isInteger() || VecEltVT.bitsGE(XLenVT) ? VecEltVT : XLenVT;

  SDValue StartSplat = lowerScalarSplat(SDValue(), Op.getOperand(0),
                                        DAG.getConstant(1, DL, XLenVT), M1VT,
                                        DL, DAG, Subtarget);
  SDValue Reduction =
      DAG.getNode(RVVOpcode, DL, M1VT, StartSplat, Vec, StartSplat, Mask, VL);
  SDValue Elt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, ResVT, Reduction,
                             DAG.getConstant(0, DL, XLenVT));
  if (!VecVT.isInteger())
    return Elt0;
  return DAG.getSExtOrTrunc(Elt0, DL, Op.getValueType());
}

SDValue RISCVTargetLowering::lowerINSERT_SUBVECTOR(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDValue Vec = Op.getOperand(0);
  SDValue SubVec = Op.getOperand(1);
  MVT VecVT = Vec.getSimpleValueType();
  MVT SubVecVT = SubVec.getSimpleValueType();

  SDLoc DL(Op);
  MVT XLenVT = Subtarget.getXLenVT();
  unsigned OrigIdx = Op.getConstantOperandVal(2);
  const RISCVRegisterInfo *TRI = Subtarget.getRegisterInfo();

  // We don't have the ability to slide mask vectors up indexed by their i1
  // elements; the smallest we can do is i8. Often we are able to bitcast to
  // equivalent i8 vectors. Note that when inserting a fixed-length vector
  // into a scalable one, we might not necessarily have enough scalable
  // elements to safely divide by 8: nxv1i1 = insert nxv1i1, v4i1 is valid.
  if (SubVecVT.getVectorElementType() == MVT::i1 &&
      (OrigIdx != 0 || !Vec.isUndef())) {
    if (VecVT.getVectorMinNumElements() >= 8 &&
        SubVecVT.getVectorMinNumElements() >= 8) {
      assert(OrigIdx % 8 == 0 && "Invalid index");
      assert(VecVT.getVectorMinNumElements() % 8 == 0 &&
             SubVecVT.getVectorMinNumElements() % 8 == 0 &&
             "Unexpected mask vector lowering");
      OrigIdx /= 8;
      SubVecVT =
          MVT::getVectorVT(MVT::i8, SubVecVT.getVectorMinNumElements() / 8,
                           SubVecVT.isScalableVector());
      VecVT = MVT::getVectorVT(MVT::i8, VecVT.getVectorMinNumElements() / 8,
                               VecVT.isScalableVector());
      Vec = DAG.getBitcast(VecVT, Vec);
      SubVec = DAG.getBitcast(SubVecVT, SubVec);
    } else {
      // We can't slide this mask vector up indexed by its i1 elements.
      // This poses a problem when we wish to insert a scalable vector which
      // can't be re-expressed as a larger type. Just choose the slow path and
      // extend to a larger type, then truncate back down.
      MVT ExtVecVT = VecVT.changeVectorElementType(MVT::i8);
      MVT ExtSubVecVT = SubVecVT.changeVectorElementType(MVT::i8);
      Vec = DAG.getNode(ISD::ZERO_EXTEND, DL, ExtVecVT, Vec);
      SubVec = DAG.getNode(ISD::ZERO_EXTEND, DL, ExtSubVecVT, SubVec);
      Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, ExtVecVT, Vec, SubVec,
                        Op.getOperand(2));
      SDValue SplatZero = DAG.getConstant(0, DL, ExtVecVT);
      return DAG.getSetCC(DL, VecVT, Vec, SplatZero, ISD::SETNE);
    }
  }

  // If the subvector vector is a fixed-length type, we cannot use subregister
  // manipulation to simplify the codegen; we don't know which register of a
  // LMUL group contains the specific subvector as we only know the minimum
  // register size. Therefore we must slide the vector group up the full
  // amount.
  if (SubVecVT.isFixedLengthVector()) {
    if (OrigIdx == 0 && Vec.isUndef() && !VecVT.isFixedLengthVector())
      return Op;
    MVT ContainerVT = VecVT;
    if (VecVT.isFixedLengthVector()) {
      ContainerVT = getContainerForFixedLengthVector(VecVT);
      Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
    }
    SubVec = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, ContainerVT,
                         DAG.getUNDEF(ContainerVT), SubVec,
                         DAG.getConstant(0, DL, XLenVT));
    if (OrigIdx == 0 && Vec.isUndef() && VecVT.isFixedLengthVector()) {
      SubVec = convertFromScalableVector(VecVT, SubVec, DAG, Subtarget);
      return DAG.getBitcast(Op.getValueType(), SubVec);
    }
    SDValue Mask =
        getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget).first;
    // Set the vector length to only the number of elements we care about. Note
    // that for slideup this includes the offset.
    SDValue VL =
        DAG.getConstant(OrigIdx + SubVecVT.getVectorNumElements(), DL, XLenVT);
    SDValue SlideupAmt = DAG.getConstant(OrigIdx, DL, XLenVT);
    SDValue Slideup = DAG.getNode(RISCVISD::VSLIDEUP_VL, DL, ContainerVT, Vec,
                                  SubVec, SlideupAmt, Mask, VL);
    if (VecVT.isFixedLengthVector())
      Slideup = convertFromScalableVector(VecVT, Slideup, DAG, Subtarget);
    return DAG.getBitcast(Op.getValueType(), Slideup);
  }

  unsigned SubRegIdx, RemIdx;
  std::tie(SubRegIdx, RemIdx) =
      RISCVTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
          VecVT, SubVecVT, OrigIdx, TRI);

  RISCVII::VLMUL SubVecLMUL = RISCVTargetLowering::getLMUL(SubVecVT);
  bool IsSubVecPartReg = SubVecLMUL == RISCVII::VLMUL::LMUL_F2 ||
                         SubVecLMUL == RISCVII::VLMUL::LMUL_F4 ||
                         SubVecLMUL == RISCVII::VLMUL::LMUL_F8;

  // 1. If the Idx has been completely eliminated and this subvector's size is
  // a vector register or a multiple thereof, or the surrounding elements are
  // undef, then this is a subvector insert which naturally aligns to a vector
  // register. These can easily be handled using subregister manipulation.
  // 2. If the subvector is smaller than a vector register, then the insertion
  // must preserve the undisturbed elements of the register. We do this by
  // lowering to an EXTRACT_SUBVECTOR grabbing the nearest LMUL=1 vector type
  // (which resolves to a subregister copy), performing a VSLIDEUP to place the
  // subvector within the vector register, and an INSERT_SUBVECTOR of that
  // LMUL=1 type back into the larger vector (resolving to another subregister
  // operation). See below for how our VSLIDEUP works. We go via a LMUL=1 type
  // to avoid allocating a large register group to hold our subvector.
  if (RemIdx == 0 && (!IsSubVecPartReg || Vec.isUndef()))
    return Op;

  // VSLIDEUP works by leaving elements 0<i<OFFSET undisturbed, elements
  // OFFSET<=i<VL set to the "subvector" and vl<=i<VLMAX set to the tail policy
  // (in our case undisturbed). This means we can set up a subvector insertion
  // where OFFSET is the insertion offset, and the VL is the OFFSET plus the
  // size of the subvector.
  MVT InterSubVT = VecVT;
  SDValue AlignedExtract = Vec;
  unsigned AlignedIdx = OrigIdx - RemIdx;
  if (VecVT.bitsGT(getLMUL1VT(VecVT))) {
    InterSubVT = getLMUL1VT(VecVT);
    // Extract a subvector equal to the nearest full vector register type. This
    // should resolve to a EXTRACT_SUBREG instruction.
    AlignedExtract = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, InterSubVT, Vec,
                                 DAG.getConstant(AlignedIdx, DL, XLenVT));
  }

  SDValue SlideupAmt = DAG.getConstant(RemIdx, DL, XLenVT);
  // For scalable vectors this must be further multiplied by vscale.
  SlideupAmt = DAG.getNode(ISD::VSCALE, DL, XLenVT, SlideupAmt);

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultScalableVLOps(VecVT, DL, DAG, Subtarget);

  // Construct the vector length corresponding to RemIdx + length(SubVecVT).
  VL = DAG.getConstant(SubVecVT.getVectorMinNumElements(), DL, XLenVT);
  VL = DAG.getNode(ISD::VSCALE, DL, XLenVT, VL);
  VL = DAG.getNode(ISD::ADD, DL, XLenVT, SlideupAmt, VL);

  SubVec = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, InterSubVT,
                       DAG.getUNDEF(InterSubVT), SubVec,
                       DAG.getConstant(0, DL, XLenVT));

  SDValue Slideup = DAG.getNode(RISCVISD::VSLIDEUP_VL, DL, InterSubVT,
                                AlignedExtract, SubVec, SlideupAmt, Mask, VL);

  // If required, insert this subvector back into the correct vector register.
  // This should resolve to an INSERT_SUBREG instruction.
  if (VecVT.bitsGT(InterSubVT))
    Slideup = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VecVT, Vec, Slideup,
                          DAG.getConstant(AlignedIdx, DL, XLenVT));

  // We might have bitcast from a mask type: cast back to the original type if
  // required.
  return DAG.getBitcast(Op.getSimpleValueType(), Slideup);
}

SDValue RISCVTargetLowering::lowerEXTRACT_SUBVECTOR(SDValue Op,
                                                    SelectionDAG &DAG) const {
  SDValue Vec = Op.getOperand(0);
  MVT SubVecVT = Op.getSimpleValueType();
  MVT VecVT = Vec.getSimpleValueType();

  SDLoc DL(Op);
  MVT XLenVT = Subtarget.getXLenVT();
  unsigned OrigIdx = Op.getConstantOperandVal(1);
  const RISCVRegisterInfo *TRI = Subtarget.getRegisterInfo();

  // We don't have the ability to slide mask vectors down indexed by their i1
  // elements; the smallest we can do is i8. Often we are able to bitcast to
  // equivalent i8 vectors. Note that when extracting a fixed-length vector
  // from a scalable one, we might not necessarily have enough scalable
  // elements to safely divide by 8: v8i1 = extract nxv1i1 is valid.
  if (SubVecVT.getVectorElementType() == MVT::i1 && OrigIdx != 0) {
    if (VecVT.getVectorMinNumElements() >= 8 &&
        SubVecVT.getVectorMinNumElements() >= 8) {
      assert(OrigIdx % 8 == 0 && "Invalid index");
      assert(VecVT.getVectorMinNumElements() % 8 == 0 &&
             SubVecVT.getVectorMinNumElements() % 8 == 0 &&
             "Unexpected mask vector lowering");
      OrigIdx /= 8;
      SubVecVT =
          MVT::getVectorVT(MVT::i8, SubVecVT.getVectorMinNumElements() / 8,
                           SubVecVT.isScalableVector());
      VecVT = MVT::getVectorVT(MVT::i8, VecVT.getVectorMinNumElements() / 8,
                               VecVT.isScalableVector());
      Vec = DAG.getBitcast(VecVT, Vec);
    } else {
      // We can't slide this mask vector down, indexed by its i1 elements.
      // This poses a problem when we wish to extract a scalable vector which
      // can't be re-expressed as a larger type. Just choose the slow path and
      // extend to a larger type, then truncate back down.
      // TODO: We could probably improve this when extracting certain fixed
      // from fixed, where we can extract as i8 and shift the correct element
      // right to reach the desired subvector?
      MVT ExtVecVT = VecVT.changeVectorElementType(MVT::i8);
      MVT ExtSubVecVT = SubVecVT.changeVectorElementType(MVT::i8);
      Vec = DAG.getNode(ISD::ZERO_EXTEND, DL, ExtVecVT, Vec);
      Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, ExtSubVecVT, Vec,
                        Op.getOperand(1));
      SDValue SplatZero = DAG.getConstant(0, DL, ExtSubVecVT);
      return DAG.getSetCC(DL, SubVecVT, Vec, SplatZero, ISD::SETNE);
    }
  }

  // If the subvector vector is a fixed-length type, we cannot use subregister
  // manipulation to simplify the codegen; we don't know which register of a
  // LMUL group contains the specific subvector as we only know the minimum
  // register size. Therefore we must slide the vector group down the full
  // amount.
  if (SubVecVT.isFixedLengthVector()) {
    // With an index of 0 this is a cast-like subvector, which can be performed
    // with subregister operations.
    if (OrigIdx == 0)
      return Op;
    MVT ContainerVT = VecVT;
    if (VecVT.isFixedLengthVector()) {
      ContainerVT = getContainerForFixedLengthVector(VecVT);
      Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
    }
    SDValue Mask =
        getDefaultVLOps(VecVT, ContainerVT, DL, DAG, Subtarget).first;
    // Set the vector length to only the number of elements we care about. This
    // avoids sliding down elements we're going to discard straight away.
    SDValue VL = DAG.getConstant(SubVecVT.getVectorNumElements(), DL, XLenVT);
    SDValue SlidedownAmt = DAG.getConstant(OrigIdx, DL, XLenVT);
    SDValue Slidedown =
        DAG.getNode(RISCVISD::VSLIDEDOWN_VL, DL, ContainerVT,
                    DAG.getUNDEF(ContainerVT), Vec, SlidedownAmt, Mask, VL);
    // Now we can use a cast-like subvector extract to get the result.
    Slidedown = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVecVT, Slidedown,
                            DAG.getConstant(0, DL, XLenVT));
    return DAG.getBitcast(Op.getValueType(), Slidedown);
  }

  unsigned SubRegIdx, RemIdx;
  std::tie(SubRegIdx, RemIdx) =
      RISCVTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
          VecVT, SubVecVT, OrigIdx, TRI);

  // If the Idx has been completely eliminated then this is a subvector extract
  // which naturally aligns to a vector register. These can easily be handled
  // using subregister manipulation.
  if (RemIdx == 0)
    return Op;

  // Else we must shift our vector register directly to extract the subvector.
  // Do this using VSLIDEDOWN.

  // If the vector type is an LMUL-group type, extract a subvector equal to the
  // nearest full vector register type. This should resolve to a EXTRACT_SUBREG
  // instruction.
  MVT InterSubVT = VecVT;
  if (VecVT.bitsGT(getLMUL1VT(VecVT))) {
    InterSubVT = getLMUL1VT(VecVT);
    Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, InterSubVT, Vec,
                      DAG.getConstant(OrigIdx - RemIdx, DL, XLenVT));
  }

  // Slide this vector register down by the desired number of elements in order
  // to place the desired subvector starting at element 0.
  SDValue SlidedownAmt = DAG.getConstant(RemIdx, DL, XLenVT);
  // For scalable vectors this must be further multiplied by vscale.
  SlidedownAmt = DAG.getNode(ISD::VSCALE, DL, XLenVT, SlidedownAmt);

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultScalableVLOps(InterSubVT, DL, DAG, Subtarget);
  SDValue Slidedown =
      DAG.getNode(RISCVISD::VSLIDEDOWN_VL, DL, InterSubVT,
                  DAG.getUNDEF(InterSubVT), Vec, SlidedownAmt, Mask, VL);

  // Now the vector is in the right position, extract our final subvector. This
  // should resolve to a COPY.
  Slidedown = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVecVT, Slidedown,
                          DAG.getConstant(0, DL, XLenVT));

  // We might have bitcast from a mask type: cast back to the original type if
  // required.
  return DAG.getBitcast(Op.getSimpleValueType(), Slidedown);
}

// Lower step_vector to the vid instruction. Any non-identity step value must
// be accounted for my manual expansion.
SDValue RISCVTargetLowering::lowerSTEP_VECTOR(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultScalableVLOps(VT, DL, DAG, Subtarget);
  SDValue StepVec = DAG.getNode(RISCVISD::VID_VL, DL, VT, Mask, VL);
  uint64_t StepValImm = Op.getConstantOperandVal(0);
  if (StepValImm != 1) {
    if (isPowerOf2_64(StepValImm)) {
      SDValue StepVal =
          DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VT, DAG.getUNDEF(VT),
                      DAG.getConstant(Log2_64(StepValImm), DL, XLenVT));
      StepVec = DAG.getNode(ISD::SHL, DL, VT, StepVec, StepVal);
    } else {
      SDValue StepVal = lowerScalarSplat(
          SDValue(), DAG.getConstant(StepValImm, DL, VT.getVectorElementType()),
          VL, VT, DL, DAG, Subtarget);
      StepVec = DAG.getNode(ISD::MUL, DL, VT, StepVec, StepVal);
    }
  }
  return StepVec;
}

// Implement vector_reverse using vrgather.vv with indices determined by
// subtracting the id of each element from (VLMAX-1). This will convert
// the indices like so:
// (0, 1,..., VLMAX-2, VLMAX-1) -> (VLMAX-1, VLMAX-2,..., 1, 0).
// TODO: This code assumes VLMAX <= 65536 for LMUL=8 SEW=16.
SDValue RISCVTargetLowering::lowerVECTOR_REVERSE(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VecVT = Op.getSimpleValueType();
  unsigned EltSize = VecVT.getScalarSizeInBits();
  unsigned MinSize = VecVT.getSizeInBits().getKnownMinValue();

  unsigned MaxVLMAX = 0;
  unsigned VectorBitsMax = Subtarget.getMaxRVVVectorSizeInBits();
  if (VectorBitsMax != 0)
    MaxVLMAX =
        RISCVTargetLowering::computeVLMAX(VectorBitsMax, EltSize, MinSize);

  unsigned GatherOpc = RISCVISD::VRGATHER_VV_VL;
  MVT IntVT = VecVT.changeVectorElementTypeToInteger();

  // If this is SEW=8 and VLMAX is unknown or more than 256, we need
  // to use vrgatherei16.vv.
  // TODO: It's also possible to use vrgatherei16.vv for other types to
  // decrease register width for the index calculation.
  if ((MaxVLMAX == 0 || MaxVLMAX > 256) && EltSize == 8) {
    // If this is LMUL=8, we have to split before can use vrgatherei16.vv.
    // Reverse each half, then reassemble them in reverse order.
    // NOTE: It's also possible that after splitting that VLMAX no longer
    // requires vrgatherei16.vv.
    if (MinSize == (8 * RISCV::RVVBitsPerBlock)) {
      SDValue Lo, Hi;
      std::tie(Lo, Hi) = DAG.SplitVectorOperand(Op.getNode(), 0);
      EVT LoVT, HiVT;
      std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(VecVT);
      Lo = DAG.getNode(ISD::VECTOR_REVERSE, DL, LoVT, Lo);
      Hi = DAG.getNode(ISD::VECTOR_REVERSE, DL, HiVT, Hi);
      // Reassemble the low and high pieces reversed.
      // FIXME: This is a CONCAT_VECTORS.
      SDValue Res =
          DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VecVT, DAG.getUNDEF(VecVT), Hi,
                      DAG.getIntPtrConstant(0, DL));
      return DAG.getNode(
          ISD::INSERT_SUBVECTOR, DL, VecVT, Res, Lo,
          DAG.getIntPtrConstant(LoVT.getVectorMinNumElements(), DL));
    }

    // Just promote the int type to i16 which will double the LMUL.
    IntVT = MVT::getVectorVT(MVT::i16, VecVT.getVectorElementCount());
    GatherOpc = RISCVISD::VRGATHEREI16_VV_VL;
  }

  MVT XLenVT = Subtarget.getXLenVT();
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultScalableVLOps(VecVT, DL, DAG, Subtarget);

  // Calculate VLMAX-1 for the desired SEW.
  unsigned MinElts = VecVT.getVectorMinNumElements();
  SDValue VLMax = DAG.getNode(ISD::VSCALE, DL, XLenVT,
                              DAG.getConstant(MinElts, DL, XLenVT));
  SDValue VLMinus1 =
      DAG.getNode(ISD::SUB, DL, XLenVT, VLMax, DAG.getConstant(1, DL, XLenVT));

  // Splat VLMAX-1 taking care to handle SEW==64 on RV32.
  bool IsRV32E64 =
      !Subtarget.is64Bit() && IntVT.getVectorElementType() == MVT::i64;
  SDValue SplatVL;
  if (!IsRV32E64)
    SplatVL = DAG.getSplatVector(IntVT, DL, VLMinus1);
  else
    SplatVL = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, IntVT, DAG.getUNDEF(IntVT),
                          VLMinus1, DAG.getRegister(RISCV::X0, XLenVT));

  SDValue VID = DAG.getNode(RISCVISD::VID_VL, DL, IntVT, Mask, VL);
  SDValue Indices =
      DAG.getNode(RISCVISD::SUB_VL, DL, IntVT, SplatVL, VID, Mask, VL);

  return DAG.getNode(GatherOpc, DL, VecVT, Op.getOperand(0), Indices, Mask, VL);
}

SDValue RISCVTargetLowering::lowerVECTOR_SPLICE(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  MVT XLenVT = Subtarget.getXLenVT();
  MVT VecVT = Op.getSimpleValueType();

  unsigned MinElts = VecVT.getVectorMinNumElements();
  SDValue VLMax = DAG.getNode(ISD::VSCALE, DL, XLenVT,
                              DAG.getConstant(MinElts, DL, XLenVT));

  int64_t ImmValue = cast<ConstantSDNode>(Op.getOperand(2))->getSExtValue();
  SDValue DownOffset, UpOffset;
  if (ImmValue >= 0) {
    // The operand is a TargetConstant, we need to rebuild it as a regular
    // constant.
    DownOffset = DAG.getConstant(ImmValue, DL, XLenVT);
    UpOffset = DAG.getNode(ISD::SUB, DL, XLenVT, VLMax, DownOffset);
  } else {
    // The operand is a TargetConstant, we need to rebuild it as a regular
    // constant rather than negating the original operand.
    UpOffset = DAG.getConstant(-ImmValue, DL, XLenVT);
    DownOffset = DAG.getNode(ISD::SUB, DL, XLenVT, VLMax, UpOffset);
  }

  MVT MaskVT = MVT::getVectorVT(MVT::i1, VecVT.getVectorElementCount());
  SDValue TrueMask = DAG.getNode(RISCVISD::VMSET_VL, DL, MaskVT, VLMax);

  SDValue SlideDown =
      DAG.getNode(RISCVISD::VSLIDEDOWN_VL, DL, VecVT, DAG.getUNDEF(VecVT), V1,
                  DownOffset, TrueMask, UpOffset);
  return DAG.getNode(RISCVISD::VSLIDEUP_VL, DL, VecVT, SlideDown, V2, UpOffset,
                     TrueMask,
                     DAG.getTargetConstant(RISCV::VLMaxSentinel, DL, XLenVT));
}

SDValue
RISCVTargetLowering::lowerFixedLengthVectorLoadToRVV(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *Load = cast<LoadSDNode>(Op);

  assert(allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                        Load->getMemoryVT(),
                                        *Load->getMemOperand()) &&
         "Expecting a correctly-aligned load");

  MVT VT = Op.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();
  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  SDValue VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);

  bool IsMaskOp = VT.getVectorElementType() == MVT::i1;
  SDValue IntID = DAG.getTargetConstant(
      IsMaskOp ? Intrinsic::riscv_vlm : Intrinsic::riscv_vle, DL, XLenVT);
  SmallVector<SDValue, 4> Ops{Load->getChain(), IntID};
  if (!IsMaskOp)
    Ops.push_back(DAG.getUNDEF(ContainerVT));
  Ops.push_back(Load->getBasePtr());
  Ops.push_back(VL);
  SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});
  SDValue NewLoad =
      DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops,
                              Load->getMemoryVT(), Load->getMemOperand());

  SDValue Result = convertFromScalableVector(VT, NewLoad, DAG, Subtarget);
  return DAG.getMergeValues({Result, Load->getChain()}, DL);
}

SDValue
RISCVTargetLowering::lowerFixedLengthVectorStoreToRVV(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *Store = cast<StoreSDNode>(Op);

  assert(allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                        Store->getMemoryVT(),
                                        *Store->getMemOperand()) &&
         "Expecting a correctly-aligned store");

  SDValue StoreVal = Store->getValue();
  MVT VT = StoreVal.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  // If the size less than a byte, we need to pad with zeros to make a byte.
  if (VT.getVectorElementType() == MVT::i1 && VT.getVectorNumElements() < 8) {
    VT = MVT::v8i1;
    StoreVal = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT,
                           DAG.getConstant(0, DL, VT), StoreVal,
                           DAG.getIntPtrConstant(0, DL));
  }

  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  SDValue VL = DAG.getConstant(VT.getVectorNumElements(), DL, XLenVT);

  SDValue NewValue =
      convertToScalableVector(ContainerVT, StoreVal, DAG, Subtarget);

  bool IsMaskOp = VT.getVectorElementType() == MVT::i1;
  SDValue IntID = DAG.getTargetConstant(
      IsMaskOp ? Intrinsic::riscv_vsm : Intrinsic::riscv_vse, DL, XLenVT);
  return DAG.getMemIntrinsicNode(
      ISD::INTRINSIC_VOID, DL, DAG.getVTList(MVT::Other),
      {Store->getChain(), IntID, NewValue, Store->getBasePtr(), VL},
      Store->getMemoryVT(), Store->getMemOperand());
}

SDValue RISCVTargetLowering::lowerMaskedLoad(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();

  const auto *MemSD = cast<MemSDNode>(Op);
  EVT MemVT = MemSD->getMemoryVT();
  MachineMemOperand *MMO = MemSD->getMemOperand();
  SDValue Chain = MemSD->getChain();
  SDValue BasePtr = MemSD->getBasePtr();

  SDValue Mask, PassThru, VL;
  if (const auto *VPLoad = dyn_cast<VPLoadSDNode>(Op)) {
    Mask = VPLoad->getMask();
    PassThru = DAG.getUNDEF(VT);
    VL = VPLoad->getVectorLength();
  } else {
    const auto *MLoad = cast<MaskedLoadSDNode>(Op);
    Mask = MLoad->getMask();
    PassThru = MLoad->getPassThru();
  }

  bool IsUnmasked = ISD::isConstantSplatVectorAllOnes(Mask.getNode());

  MVT XLenVT = Subtarget.getXLenVT();

  MVT ContainerVT = VT;
  if (VT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VT);
    PassThru = convertToScalableVector(ContainerVT, PassThru, DAG, Subtarget);
    if (!IsUnmasked) {
      MVT MaskVT =
          MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
      Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
    }
  }

  if (!VL)
    VL = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget).second;

  unsigned IntID =
      IsUnmasked ? Intrinsic::riscv_vle : Intrinsic::riscv_vle_mask;
  SmallVector<SDValue, 8> Ops{Chain, DAG.getTargetConstant(IntID, DL, XLenVT)};
  if (IsUnmasked)
    Ops.push_back(DAG.getUNDEF(ContainerVT));
  else
    Ops.push_back(PassThru);
  Ops.push_back(BasePtr);
  if (!IsUnmasked)
    Ops.push_back(Mask);
  Ops.push_back(VL);
  if (!IsUnmasked)
    Ops.push_back(DAG.getTargetConstant(RISCVII::TAIL_AGNOSTIC, DL, XLenVT));

  SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});

  SDValue Result =
      DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops, MemVT, MMO);
  Chain = Result.getValue(1);

  if (VT.isFixedLengthVector())
    Result = convertFromScalableVector(VT, Result, DAG, Subtarget);

  return DAG.getMergeValues({Result, Chain}, DL);
}

SDValue RISCVTargetLowering::lowerMaskedStore(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);

  const auto *MemSD = cast<MemSDNode>(Op);
  EVT MemVT = MemSD->getMemoryVT();
  MachineMemOperand *MMO = MemSD->getMemOperand();
  SDValue Chain = MemSD->getChain();
  SDValue BasePtr = MemSD->getBasePtr();
  SDValue Val, Mask, VL;

  if (const auto *VPStore = dyn_cast<VPStoreSDNode>(Op)) {
    Val = VPStore->getValue();
    Mask = VPStore->getMask();
    VL = VPStore->getVectorLength();
  } else {
    const auto *MStore = cast<MaskedStoreSDNode>(Op);
    Val = MStore->getValue();
    Mask = MStore->getMask();
  }

  bool IsUnmasked = ISD::isConstantSplatVectorAllOnes(Mask.getNode());

  MVT VT = Val.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  MVT ContainerVT = VT;
  if (VT.isFixedLengthVector()) {
    ContainerVT = getContainerForFixedLengthVector(VT);

    Val = convertToScalableVector(ContainerVT, Val, DAG, Subtarget);
    if (!IsUnmasked) {
      MVT MaskVT =
          MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
      Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
    }
  }

  if (!VL)
    VL = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget).second;

  unsigned IntID =
      IsUnmasked ? Intrinsic::riscv_vse : Intrinsic::riscv_vse_mask;
  SmallVector<SDValue, 8> Ops{Chain, DAG.getTargetConstant(IntID, DL, XLenVT)};
  Ops.push_back(Val);
  Ops.push_back(BasePtr);
  if (!IsUnmasked)
    Ops.push_back(Mask);
  Ops.push_back(VL);

  return DAG.getMemIntrinsicNode(ISD::INTRINSIC_VOID, DL,
                                 DAG.getVTList(MVT::Other), Ops, MemVT, MMO);
}

SDValue
RISCVTargetLowering::lowerFixedLengthVectorSetccToRVV(SDValue Op,
                                                      SelectionDAG &DAG) const {
  MVT InVT = Op.getOperand(0).getSimpleValueType();
  MVT ContainerVT = getContainerForFixedLengthVector(InVT);

  MVT VT = Op.getSimpleValueType();

  SDValue Op1 =
      convertToScalableVector(ContainerVT, Op.getOperand(0), DAG, Subtarget);
  SDValue Op2 =
      convertToScalableVector(ContainerVT, Op.getOperand(1), DAG, Subtarget);

  SDLoc DL(Op);
  SDValue VL =
      DAG.getConstant(VT.getVectorNumElements(), DL, Subtarget.getXLenVT());

  MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
  SDValue Mask = DAG.getNode(RISCVISD::VMSET_VL, DL, MaskVT, VL);

  SDValue Cmp = DAG.getNode(RISCVISD::SETCC_VL, DL, MaskVT, Op1, Op2,
                            Op.getOperand(2), Mask, VL);

  return convertFromScalableVector(VT, Cmp, DAG, Subtarget);
}

SDValue RISCVTargetLowering::lowerFixedLengthVectorLogicOpToRVV(
    SDValue Op, SelectionDAG &DAG, unsigned MaskOpc, unsigned VecOpc) const {
  MVT VT = Op.getSimpleValueType();

  if (VT.getVectorElementType() == MVT::i1)
    return lowerToScalableOp(Op, DAG, MaskOpc, /*HasMask*/ false);

  return lowerToScalableOp(Op, DAG, VecOpc, /*HasMask*/ true);
}

SDValue
RISCVTargetLowering::lowerFixedLengthVectorShiftToRVV(SDValue Op,
                                                      SelectionDAG &DAG) const {
  unsigned Opc;
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Unexpected opcode!");
  case ISD::SHL: Opc = RISCVISD::SHL_VL; break;
  case ISD::SRA: Opc = RISCVISD::SRA_VL; break;
  case ISD::SRL: Opc = RISCVISD::SRL_VL; break;
  }

  return lowerToScalableOp(Op, DAG, Opc);
}

// Lower vector ABS to smax(X, sub(0, X)).
SDValue RISCVTargetLowering::lowerABS(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  SDValue X = Op.getOperand(0);

  assert(VT.isFixedLengthVector() && "Unexpected type");

  MVT ContainerVT = getContainerForFixedLengthVector(VT);
  X = convertToScalableVector(ContainerVT, X, DAG, Subtarget);

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  SDValue SplatZero = DAG.getNode(
      RISCVISD::VMV_V_X_VL, DL, ContainerVT, DAG.getUNDEF(ContainerVT),
      DAG.getConstant(0, DL, Subtarget.getXLenVT()));
  SDValue NegX =
      DAG.getNode(RISCVISD::SUB_VL, DL, ContainerVT, SplatZero, X, Mask, VL);
  SDValue Max =
      DAG.getNode(RISCVISD::SMAX_VL, DL, ContainerVT, X, NegX, Mask, VL);

  return convertFromScalableVector(VT, Max, DAG, Subtarget);
}

SDValue RISCVTargetLowering::lowerFixedLengthVectorFCOPYSIGNToRVV(
    SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  SDValue Mag = Op.getOperand(0);
  SDValue Sign = Op.getOperand(1);
  assert(Mag.getValueType() == Sign.getValueType() &&
         "Can only handle COPYSIGN with matching types.");

  MVT ContainerVT = getContainerForFixedLengthVector(VT);
  Mag = convertToScalableVector(ContainerVT, Mag, DAG, Subtarget);
  Sign = convertToScalableVector(ContainerVT, Sign, DAG, Subtarget);

  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  SDValue CopySign =
      DAG.getNode(RISCVISD::FCOPYSIGN_VL, DL, ContainerVT, Mag, Sign, Mask, VL);

  return convertFromScalableVector(VT, CopySign, DAG, Subtarget);
}

SDValue RISCVTargetLowering::lowerFixedLengthVectorSelectToRVV(
    SDValue Op, SelectionDAG &DAG) const {
  MVT VT = Op.getSimpleValueType();
  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  MVT I1ContainerVT =
      MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());

  SDValue CC =
      convertToScalableVector(I1ContainerVT, Op.getOperand(0), DAG, Subtarget);
  SDValue Op1 =
      convertToScalableVector(ContainerVT, Op.getOperand(1), DAG, Subtarget);
  SDValue Op2 =
      convertToScalableVector(ContainerVT, Op.getOperand(2), DAG, Subtarget);

  SDLoc DL(Op);
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);

  SDValue Select =
      DAG.getNode(RISCVISD::VSELECT_VL, DL, ContainerVT, CC, Op1, Op2, VL);

  return convertFromScalableVector(VT, Select, DAG, Subtarget);
}

SDValue RISCVTargetLowering::lowerToScalableOp(SDValue Op, SelectionDAG &DAG,
                                               unsigned NewOpc,
                                               bool HasMask) const {
  MVT VT = Op.getSimpleValueType();
  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  // Create list of operands by converting existing ones to scalable types.
  SmallVector<SDValue, 6> Ops;
  for (const SDValue &V : Op->op_values()) {
    assert(!isa<VTSDNode>(V) && "Unexpected VTSDNode node!");

    // Pass through non-vector operands.
    if (!V.getValueType().isVector()) {
      Ops.push_back(V);
      continue;
    }

    // "cast" fixed length vector to a scalable vector.
    assert(useRVVForFixedLengthVectorVT(V.getSimpleValueType()) &&
           "Only fixed length vectors are supported!");
    Ops.push_back(convertToScalableVector(ContainerVT, V, DAG, Subtarget));
  }

  SDLoc DL(Op);
  SDValue Mask, VL;
  std::tie(Mask, VL) = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget);
  if (HasMask)
    Ops.push_back(Mask);
  Ops.push_back(VL);

  SDValue ScalableRes = DAG.getNode(NewOpc, DL, ContainerVT, Ops);
  return convertFromScalableVector(VT, ScalableRes, DAG, Subtarget);
}

// Lower a VP_* ISD node to the corresponding RISCVISD::*_VL node:
// * Operands of each node are assumed to be in the same order.
// * The EVL operand is promoted from i32 to i64 on RV64.
// * Fixed-length vectors are converted to their scalable-vector container
//   types.
SDValue RISCVTargetLowering::lowerVPOp(SDValue Op, SelectionDAG &DAG,
                                       unsigned RISCVISDOpc) const {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  SmallVector<SDValue, 4> Ops;

  for (const auto &OpIdx : enumerate(Op->ops())) {
    SDValue V = OpIdx.value();
    assert(!isa<VTSDNode>(V) && "Unexpected VTSDNode node!");
    // Pass through operands which aren't fixed-length vectors.
    if (!V.getValueType().isFixedLengthVector()) {
      Ops.push_back(V);
      continue;
    }
    // "cast" fixed length vector to a scalable vector.
    MVT OpVT = V.getSimpleValueType();
    MVT ContainerVT = getContainerForFixedLengthVector(OpVT);
    assert(useRVVForFixedLengthVectorVT(OpVT) &&
           "Only fixed length vectors are supported!");
    Ops.push_back(convertToScalableVector(ContainerVT, V, DAG, Subtarget));
  }

  if (!VT.isFixedLengthVector())
    return DAG.getNode(RISCVISDOpc, DL, VT, Ops);

  MVT ContainerVT = getContainerForFixedLengthVector(VT);

  SDValue VPOp = DAG.getNode(RISCVISDOpc, DL, ContainerVT, Ops);

  return convertFromScalableVector(VT, VPOp, DAG, Subtarget);
}

SDValue RISCVTargetLowering::lowerLogicVPOp(SDValue Op, SelectionDAG &DAG,
                                            unsigned MaskOpc,
                                            unsigned VecOpc) const {
  MVT VT = Op.getSimpleValueType();
  if (VT.getVectorElementType() != MVT::i1)
    return lowerVPOp(Op, DAG, VecOpc);

  // It is safe to drop mask parameter as masked-off elements are undef.
  SDValue Op1 = Op->getOperand(0);
  SDValue Op2 = Op->getOperand(1);
  SDValue VL = Op->getOperand(3);

  MVT ContainerVT = VT;
  const bool IsFixed = VT.isFixedLengthVector();
  if (IsFixed) {
    ContainerVT = getContainerForFixedLengthVector(VT);
    Op1 = convertToScalableVector(ContainerVT, Op1, DAG, Subtarget);
    Op2 = convertToScalableVector(ContainerVT, Op2, DAG, Subtarget);
  }

  SDLoc DL(Op);
  SDValue Val = DAG.getNode(MaskOpc, DL, ContainerVT, Op1, Op2, VL);
  if (!IsFixed)
    return Val;
  return convertFromScalableVector(VT, Val, DAG, Subtarget);
}

// Custom lower MGATHER/VP_GATHER to a legalized form for RVV. It will then be
// matched to a RVV indexed load. The RVV indexed load instructions only
// support the "unsigned unscaled" addressing mode; indices are implicitly
// zero-extended or truncated to XLEN and are treated as byte offsets. Any
// signed or scaled indexing is extended to the XLEN value type and scaled
// accordingly.
SDValue RISCVTargetLowering::lowerMaskedGather(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();

  const auto *MemSD = cast<MemSDNode>(Op.getNode());
  EVT MemVT = MemSD->getMemoryVT();
  MachineMemOperand *MMO = MemSD->getMemOperand();
  SDValue Chain = MemSD->getChain();
  SDValue BasePtr = MemSD->getBasePtr();

  ISD::LoadExtType LoadExtType;
  SDValue Index, Mask, PassThru, VL;

  if (auto *VPGN = dyn_cast<VPGatherSDNode>(Op.getNode())) {
    Index = VPGN->getIndex();
    Mask = VPGN->getMask();
    PassThru = DAG.getUNDEF(VT);
    VL = VPGN->getVectorLength();
    // VP doesn't support extending loads.
    LoadExtType = ISD::NON_EXTLOAD;
  } else {
    // Else it must be a MGATHER.
    auto *MGN = cast<MaskedGatherSDNode>(Op.getNode());
    Index = MGN->getIndex();
    Mask = MGN->getMask();
    PassThru = MGN->getPassThru();
    LoadExtType = MGN->getExtensionType();
  }

  MVT IndexVT = Index.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Unexpected VTs!");
  assert(BasePtr.getSimpleValueType() == XLenVT && "Unexpected pointer type");
  // Targets have to explicitly opt-in for extending vector loads.
  assert(LoadExtType == ISD::NON_EXTLOAD &&
         "Unexpected extending MGATHER/VP_GATHER");
  (void)LoadExtType;

  // If the mask is known to be all ones, optimize to an unmasked intrinsic;
  // the selection of the masked intrinsics doesn't do this for us.
  bool IsUnmasked = ISD::isConstantSplatVectorAllOnes(Mask.getNode());

  MVT ContainerVT = VT;
  if (VT.isFixedLengthVector()) {
    // We need to use the larger of the result and index type to determine the
    // scalable type to use so we don't increase LMUL for any operand/result.
    if (VT.bitsGE(IndexVT)) {
      ContainerVT = getContainerForFixedLengthVector(VT);
      IndexVT = MVT::getVectorVT(IndexVT.getVectorElementType(),
                                 ContainerVT.getVectorElementCount());
    } else {
      IndexVT = getContainerForFixedLengthVector(IndexVT);
      ContainerVT = MVT::getVectorVT(ContainerVT.getVectorElementType(),
                                     IndexVT.getVectorElementCount());
    }

    Index = convertToScalableVector(IndexVT, Index, DAG, Subtarget);

    if (!IsUnmasked) {
      MVT MaskVT =
          MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
      Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
      PassThru = convertToScalableVector(ContainerVT, PassThru, DAG, Subtarget);
    }
  }

  if (!VL)
    VL = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget).second;

  if (XLenVT == MVT::i32 && IndexVT.getVectorElementType().bitsGT(XLenVT)) {
    IndexVT = IndexVT.changeVectorElementType(XLenVT);
    SDValue TrueMask = DAG.getNode(RISCVISD::VMSET_VL, DL, Mask.getValueType(),
                                   VL);
    Index = DAG.getNode(RISCVISD::TRUNCATE_VECTOR_VL, DL, IndexVT, Index,
                        TrueMask, VL);
  }

  unsigned IntID =
      IsUnmasked ? Intrinsic::riscv_vluxei : Intrinsic::riscv_vluxei_mask;
  SmallVector<SDValue, 8> Ops{Chain, DAG.getTargetConstant(IntID, DL, XLenVT)};
  if (IsUnmasked)
    Ops.push_back(DAG.getUNDEF(ContainerVT));
  else
    Ops.push_back(PassThru);
  Ops.push_back(BasePtr);
  Ops.push_back(Index);
  if (!IsUnmasked)
    Ops.push_back(Mask);
  Ops.push_back(VL);
  if (!IsUnmasked)
    Ops.push_back(DAG.getTargetConstant(RISCVII::TAIL_AGNOSTIC, DL, XLenVT));

  SDVTList VTs = DAG.getVTList({ContainerVT, MVT::Other});
  SDValue Result =
      DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops, MemVT, MMO);
  Chain = Result.getValue(1);

  if (VT.isFixedLengthVector())
    Result = convertFromScalableVector(VT, Result, DAG, Subtarget);

  return DAG.getMergeValues({Result, Chain}, DL);
}

// Custom lower MSCATTER/VP_SCATTER to a legalized form for RVV. It will then be
// matched to a RVV indexed store. The RVV indexed store instructions only
// support the "unsigned unscaled" addressing mode; indices are implicitly
// zero-extended or truncated to XLEN and are treated as byte offsets. Any
// signed or scaled indexing is extended to the XLEN value type and scaled
// accordingly.
SDValue RISCVTargetLowering::lowerMaskedScatter(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const auto *MemSD = cast<MemSDNode>(Op.getNode());
  EVT MemVT = MemSD->getMemoryVT();
  MachineMemOperand *MMO = MemSD->getMemOperand();
  SDValue Chain = MemSD->getChain();
  SDValue BasePtr = MemSD->getBasePtr();

  bool IsTruncatingStore = false;
  SDValue Index, Mask, Val, VL;

  if (auto *VPSN = dyn_cast<VPScatterSDNode>(Op.getNode())) {
    Index = VPSN->getIndex();
    Mask = VPSN->getMask();
    Val = VPSN->getValue();
    VL = VPSN->getVectorLength();
    // VP doesn't support truncating stores.
    IsTruncatingStore = false;
  } else {
    // Else it must be a MSCATTER.
    auto *MSN = cast<MaskedScatterSDNode>(Op.getNode());
    Index = MSN->getIndex();
    Mask = MSN->getMask();
    Val = MSN->getValue();
    IsTruncatingStore = MSN->isTruncatingStore();
  }

  MVT VT = Val.getSimpleValueType();
  MVT IndexVT = Index.getSimpleValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
         "Unexpected VTs!");
  assert(BasePtr.getSimpleValueType() == XLenVT && "Unexpected pointer type");
  // Targets have to explicitly opt-in for extending vector loads and
  // truncating vector stores.
  assert(!IsTruncatingStore && "Unexpected truncating MSCATTER/VP_SCATTER");
  (void)IsTruncatingStore;

  // If the mask is known to be all ones, optimize to an unmasked intrinsic;
  // the selection of the masked intrinsics doesn't do this for us.
  bool IsUnmasked = ISD::isConstantSplatVectorAllOnes(Mask.getNode());

  MVT ContainerVT = VT;
  if (VT.isFixedLengthVector()) {
    // We need to use the larger of the value and index type to determine the
    // scalable type to use so we don't increase LMUL for any operand/result.
    if (VT.bitsGE(IndexVT)) {
      ContainerVT = getContainerForFixedLengthVector(VT);
      IndexVT = MVT::getVectorVT(IndexVT.getVectorElementType(),
                                 ContainerVT.getVectorElementCount());
    } else {
      IndexVT = getContainerForFixedLengthVector(IndexVT);
      ContainerVT = MVT::getVectorVT(VT.getVectorElementType(),
                                     IndexVT.getVectorElementCount());
    }

    Index = convertToScalableVector(IndexVT, Index, DAG, Subtarget);
    Val = convertToScalableVector(ContainerVT, Val, DAG, Subtarget);

    if (!IsUnmasked) {
      MVT MaskVT =
          MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
      Mask = convertToScalableVector(MaskVT, Mask, DAG, Subtarget);
    }
  }

  if (!VL)
    VL = getDefaultVLOps(VT, ContainerVT, DL, DAG, Subtarget).second;

  if (XLenVT == MVT::i32 && IndexVT.getVectorElementType().bitsGT(XLenVT)) {
    IndexVT = IndexVT.changeVectorElementType(XLenVT);
    SDValue TrueMask = DAG.getNode(RISCVISD::VMSET_VL, DL, Mask.getValueType(),
                                   VL);
    Index = DAG.getNode(RISCVISD::TRUNCATE_VECTOR_VL, DL, IndexVT, Index,
                        TrueMask, VL);
  }

  unsigned IntID =
      IsUnmasked ? Intrinsic::riscv_vsoxei : Intrinsic::riscv_vsoxei_mask;
  SmallVector<SDValue, 8> Ops{Chain, DAG.getTargetConstant(IntID, DL, XLenVT)};
  Ops.push_back(Val);
  Ops.push_back(BasePtr);
  Ops.push_back(Index);
  if (!IsUnmasked)
    Ops.push_back(Mask);
  Ops.push_back(VL);

  return DAG.getMemIntrinsicNode(ISD::INTRINSIC_VOID, DL,
                                 DAG.getVTList(MVT::Other), Ops, MemVT, MMO);
}

SDValue RISCVTargetLowering::lowerGET_ROUNDING(SDValue Op,
                                               SelectionDAG &DAG) const {
  const MVT XLenVT = Subtarget.getXLenVT();
  SDLoc DL(Op);
  SDValue Chain = Op->getOperand(0);
  SDValue SysRegNo = DAG.getTargetConstant(
      RISCVSysReg::lookupSysRegByName("FRM")->Encoding, DL, XLenVT);
  SDVTList VTs = DAG.getVTList(XLenVT, MVT::Other);
  SDValue RM = DAG.getNode(RISCVISD::READ_CSR, DL, VTs, Chain, SysRegNo);

  // Encoding used for rounding mode in RISCV differs from that used in
  // FLT_ROUNDS. To convert it the RISCV rounding mode is used as an index in a
  // table, which consists of a sequence of 4-bit fields, each representing
  // corresponding FLT_ROUNDS mode.
  static const int Table =
      (int(RoundingMode::NearestTiesToEven) << 4 * RISCVFPRndMode::RNE) |
      (int(RoundingMode::TowardZero) << 4 * RISCVFPRndMode::RTZ) |
      (int(RoundingMode::TowardNegative) << 4 * RISCVFPRndMode::RDN) |
      (int(RoundingMode::TowardPositive) << 4 * RISCVFPRndMode::RUP) |
      (int(RoundingMode::NearestTiesToAway) << 4 * RISCVFPRndMode::RMM);

  SDValue Shift =
      DAG.getNode(ISD::SHL, DL, XLenVT, RM, DAG.getConstant(2, DL, XLenVT));
  SDValue Shifted = DAG.getNode(ISD::SRL, DL, XLenVT,
                                DAG.getConstant(Table, DL, XLenVT), Shift);
  SDValue Masked = DAG.getNode(ISD::AND, DL, XLenVT, Shifted,
                               DAG.getConstant(7, DL, XLenVT));

  return DAG.getMergeValues({Masked, Chain}, DL);
}

SDValue RISCVTargetLowering::lowerSET_ROUNDING(SDValue Op,
                                               SelectionDAG &DAG) const {
  const MVT XLenVT = Subtarget.getXLenVT();
  SDLoc DL(Op);
  SDValue Chain = Op->getOperand(0);
  SDValue RMValue = Op->getOperand(1);
  SDValue SysRegNo = DAG.getTargetConstant(
      RISCVSysReg::lookupSysRegByName("FRM")->Encoding, DL, XLenVT);

  // Encoding used for rounding mode in RISCV differs from that used in
  // FLT_ROUNDS. To convert it the C rounding mode is used as an index in
  // a table, which consists of a sequence of 4-bit fields, each representing
  // corresponding RISCV mode.
  static const unsigned Table =
      (RISCVFPRndMode::RNE << 4 * int(RoundingMode::NearestTiesToEven)) |
      (RISCVFPRndMode::RTZ << 4 * int(RoundingMode::TowardZero)) |
      (RISCVFPRndMode::RDN << 4 * int(RoundingMode::TowardNegative)) |
      (RISCVFPRndMode::RUP << 4 * int(RoundingMode::TowardPositive)) |
      (RISCVFPRndMode::RMM << 4 * int(RoundingMode::NearestTiesToAway));

  SDValue Shift = DAG.getNode(ISD::SHL, DL, XLenVT, RMValue,
                              DAG.getConstant(2, DL, XLenVT));
  SDValue Shifted = DAG.getNode(ISD::SRL, DL, XLenVT,
                                DAG.getConstant(Table, DL, XLenVT), Shift);
  RMValue = DAG.getNode(ISD::AND, DL, XLenVT, Shifted,
                        DAG.getConstant(0x7, DL, XLenVT));
  return DAG.getNode(RISCVISD::WRITE_CSR, DL, MVT::Other, Chain, SysRegNo,
                     RMValue);
}

static RISCVISD::NodeType getRISCVWOpcodeByIntr(unsigned IntNo) {
  switch (IntNo) {
  default:
    llvm_unreachable("Unexpected Intrinsic");
  case Intrinsic::riscv_bcompress:
    return RISCVISD::BCOMPRESSW;
  case Intrinsic::riscv_bdecompress:
    return RISCVISD::BDECOMPRESSW;
  case Intrinsic::riscv_bfp:
    return RISCVISD::BFPW;
  case Intrinsic::riscv_fsl:
    return RISCVISD::FSLW;
  case Intrinsic::riscv_fsr:
    return RISCVISD::FSRW;
  }
}

// Converts the given intrinsic to a i64 operation with any extension.
static SDValue customLegalizeToWOpByIntr(SDNode *N, SelectionDAG &DAG,
                                         unsigned IntNo) {
  SDLoc DL(N);
  RISCVISD::NodeType WOpcode = getRISCVWOpcodeByIntr(IntNo);
  SDValue NewOp1 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
  SDValue NewOp2 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
  SDValue NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp1, NewOp2);
  // ReplaceNodeResults requires we maintain the same type for the return value.
  return DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), NewRes);
}

// Returns the opcode of the target-specific SDNode that implements the 32-bit
// form of the given Opcode.
static RISCVISD::NodeType getRISCVWOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::SHL:
    return RISCVISD::SLLW;
  case ISD::SRA:
    return RISCVISD::SRAW;
  case ISD::SRL:
    return RISCVISD::SRLW;
  case ISD::SDIV:
    return RISCVISD::DIVW;
  case ISD::UDIV:
    return RISCVISD::DIVUW;
  case ISD::UREM:
    return RISCVISD::REMUW;
  case ISD::ROTL:
    return RISCVISD::ROLW;
  case ISD::ROTR:
    return RISCVISD::RORW;
  }
}

// Converts the given i8/i16/i32 operation to a target-specific SelectionDAG
// node. Because i8/i16/i32 isn't a legal type for RV64, these operations would
// otherwise be promoted to i64, making it difficult to select the
// SLLW/DIVUW/.../*W later one because the fact the operation was originally of
// type i8/i16/i32 is lost.
static SDValue customLegalizeToWOp(SDNode *N, SelectionDAG &DAG,
                                   unsigned ExtOpc = ISD::ANY_EXTEND) {
  SDLoc DL(N);
  RISCVISD::NodeType WOpcode = getRISCVWOpcode(N->getOpcode());
  SDValue NewOp0 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(1));
  SDValue NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp0, NewOp1);
  // ReplaceNodeResults requires we maintain the same type for the return value.
  return DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), NewRes);
}

// Converts the given 32-bit operation to a i64 operation with signed extension
// semantic to reduce the signed extension instructions.
static SDValue customLegalizeToWOpWithSExt(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
  SDValue NewWOp = DAG.getNode(N->getOpcode(), DL, MVT::i64, NewOp0, NewOp1);
  SDValue NewRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, NewWOp,
                               DAG.getValueType(MVT::i32));
  return DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes);
}

void RISCVTargetLowering::ReplaceNodeResults(SDNode *N,
                                             SmallVectorImpl<SDValue> &Results,
                                             SelectionDAG &DAG) const {
  SDLoc DL(N);
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom type legalize this operation!");
  case ISD::STRICT_FP_TO_SINT:
  case ISD::STRICT_FP_TO_UINT:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    bool IsStrict = N->isStrictFPOpcode();
    bool IsSigned = N->getOpcode() == ISD::FP_TO_SINT ||
                    N->getOpcode() == ISD::STRICT_FP_TO_SINT;
    SDValue Op0 = IsStrict ? N->getOperand(1) : N->getOperand(0);
    if (getTypeAction(*DAG.getContext(), Op0.getValueType()) !=
        TargetLowering::TypeSoftenFloat) {
      if (!isTypeLegal(Op0.getValueType()))
        return;
      if (IsStrict) {
        unsigned Opc = IsSigned ? RISCVISD::STRICT_FCVT_W_RV64
                                : RISCVISD::STRICT_FCVT_WU_RV64;
        SDVTList VTs = DAG.getVTList(MVT::i64, MVT::Other);
        SDValue Res = DAG.getNode(
            Opc, DL, VTs, N->getOperand(0), Op0,
            DAG.getTargetConstant(RISCVFPRndMode::RTZ, DL, MVT::i64));
        Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
        Results.push_back(Res.getValue(1));
        return;
      }
      unsigned Opc = IsSigned ? RISCVISD::FCVT_W_RV64 : RISCVISD::FCVT_WU_RV64;
      SDValue Res =
          DAG.getNode(Opc, DL, MVT::i64, Op0,
                      DAG.getTargetConstant(RISCVFPRndMode::RTZ, DL, MVT::i64));
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      return;
    }
    // If the FP type needs to be softened, emit a library call using the 'si'
    // version. If we left it to default legalization we'd end up with 'di'. If
    // the FP type doesn't need to be softened just let generic type
    // legalization promote the result type.
    RTLIB::Libcall LC;
    if (IsSigned)
      LC = RTLIB::getFPTOSINT(Op0.getValueType(), N->getValueType(0));
    else
      LC = RTLIB::getFPTOUINT(Op0.getValueType(), N->getValueType(0));
    MakeLibCallOptions CallOptions;
    EVT OpVT = Op0.getValueType();
    CallOptions.setTypeListBeforeSoften(OpVT, N->getValueType(0), true);
    SDValue Chain = IsStrict ? N->getOperand(0) : SDValue();
    SDValue Result;
    std::tie(Result, Chain) =
        makeLibCall(DAG, LC, N->getValueType(0), Op0, CallOptions, DL, Chain);
    Results.push_back(Result);
    if (IsStrict)
      Results.push_back(Chain);
    break;
  }
  case ISD::READCYCLECOUNTER: {
    assert(!Subtarget.is64Bit() &&
           "READCYCLECOUNTER only has custom type legalization on riscv32");

    SDVTList VTs = DAG.getVTList(MVT::i32, MVT::i32, MVT::Other);
    SDValue RCW =
        DAG.getNode(RISCVISD::READ_CYCLE_WIDE, DL, VTs, N->getOperand(0));

    Results.push_back(
        DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, RCW, RCW.getValue(1)));
    Results.push_back(RCW.getValue(2));
    break;
  }
  case ISD::MUL: {
    unsigned Size = N->getSimpleValueType(0).getSizeInBits();
    unsigned XLen = Subtarget.getXLen();
    // This multiply needs to be expanded, try to use MULHSU+MUL if possible.
    if (Size > XLen) {
      assert(Size == (XLen * 2) && "Unexpected custom legalisation");
      SDValue LHS = N->getOperand(0);
      SDValue RHS = N->getOperand(1);
      APInt HighMask = APInt::getHighBitsSet(Size, XLen);

      bool LHSIsU = DAG.MaskedValueIsZero(LHS, HighMask);
      bool RHSIsU = DAG.MaskedValueIsZero(RHS, HighMask);
      // We need exactly one side to be unsigned.
      if (LHSIsU == RHSIsU)
        return;

      auto MakeMULPair = [&](SDValue S, SDValue U) {
        MVT XLenVT = Subtarget.getXLenVT();
        S = DAG.getNode(ISD::TRUNCATE, DL, XLenVT, S);
        U = DAG.getNode(ISD::TRUNCATE, DL, XLenVT, U);
        SDValue Lo = DAG.getNode(ISD::MUL, DL, XLenVT, S, U);
        SDValue Hi = DAG.getNode(RISCVISD::MULHSU, DL, XLenVT, S, U);
        return DAG.getNode(ISD::BUILD_PAIR, DL, N->getValueType(0), Lo, Hi);
      };

      bool LHSIsS = DAG.ComputeNumSignBits(LHS) > XLen;
      bool RHSIsS = DAG.ComputeNumSignBits(RHS) > XLen;

      // The other operand should be signed, but still prefer MULH when
      // possible.
      if (RHSIsU && LHSIsS && !RHSIsS)
        Results.push_back(MakeMULPair(LHS, RHS));
      else if (LHSIsU && RHSIsS && !LHSIsS)
        Results.push_back(MakeMULPair(RHS, LHS));

      return;
    }
    LLVM_FALLTHROUGH;
  }
  case ISD::ADD:
  case ISD::SUB:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    Results.push_back(customLegalizeToWOpWithSExt(N, DAG));
    break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() != ISD::Constant) {
      Results.push_back(customLegalizeToWOp(N, DAG));
      break;
    }

    // Custom legalize ISD::SHL by placing a SIGN_EXTEND_INREG after. This is
    // similar to customLegalizeToWOpWithSExt, but we must zero_extend the
    // shift amount.
    if (N->getOpcode() == ISD::SHL) {
      SDLoc DL(N);
      SDValue NewOp0 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
      SDValue NewOp1 =
          DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue NewWOp = DAG.getNode(ISD::SHL, DL, MVT::i64, NewOp0, NewOp1);
      SDValue NewRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, NewWOp,
                                   DAG.getValueType(MVT::i32));
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes));
    }

    break;
  case ISD::ROTL:
  case ISD::ROTR:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    Results.push_back(customLegalizeToWOp(N, DAG));
    break;
  case ISD::CTTZ:
  case ISD::CTTZ_ZERO_UNDEF:
  case ISD::CTLZ:
  case ISD::CTLZ_ZERO_UNDEF: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");

    SDValue NewOp0 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
    bool IsCTZ =
        N->getOpcode() == ISD::CTTZ || N->getOpcode() == ISD::CTTZ_ZERO_UNDEF;
    unsigned Opc = IsCTZ ? RISCVISD::CTZW : RISCVISD::CLZW;
    SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp0);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
    return;
  }
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::UREM: {
    MVT VT = N->getSimpleValueType(0);
    assert((VT == MVT::i8 || VT == MVT::i16 || VT == MVT::i32) &&
           Subtarget.is64Bit() && Subtarget.hasStdExtM() &&
           "Unexpected custom legalisation");
    // Don't promote division/remainder by constant since we should expand those
    // to multiply by magic constant.
    // FIXME: What if the expansion is disabled for minsize.
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      return;

    // If the input is i32, use ANY_EXTEND since the W instructions don't read
    // the upper 32 bits. For other types we need to sign or zero extend
    // based on the opcode.
    unsigned ExtOpc = ISD::ANY_EXTEND;
    if (VT != MVT::i32)
      ExtOpc = N->getOpcode() == ISD::SDIV ? ISD::SIGN_EXTEND
                                           : ISD::ZERO_EXTEND;

    Results.push_back(customLegalizeToWOp(N, DAG, ExtOpc));
    break;
  }
  case ISD::UADDO:
  case ISD::USUBO: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    bool IsAdd = N->getOpcode() == ISD::UADDO;
    // Create an ADDW or SUBW.
    SDValue LHS = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
    SDValue RHS = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
    SDValue Res =
        DAG.getNode(IsAdd ? ISD::ADD : ISD::SUB, DL, MVT::i64, LHS, RHS);
    Res = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, Res,
                      DAG.getValueType(MVT::i32));

    // Sign extend the LHS and perform an unsigned compare with the ADDW result.
    // Since the inputs are sign extended from i32, this is equivalent to
    // comparing the lower 32 bits.
    LHS = DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, N->getOperand(0));
    SDValue Overflow = DAG.getSetCC(DL, N->getValueType(1), Res, LHS,
                                    IsAdd ? ISD::SETULT : ISD::SETUGT);

    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
    Results.push_back(Overflow);
    return;
  }
  case ISD::UADDSAT:
  case ISD::USUBSAT: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (Subtarget.hasStdExtZbb()) {
      // With Zbb we can sign extend and let LegalizeDAG use minu/maxu. Using
      // sign extend allows overflow of the lower 32 bits to be detected on
      // the promoted size.
      SDValue LHS =
          DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, N->getOperand(0));
      SDValue RHS =
          DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue Res = DAG.getNode(N->getOpcode(), DL, MVT::i64, LHS, RHS);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      return;
    }

    // Without Zbb, expand to UADDO/USUBO+select which will trigger our custom
    // promotion for UADDO/USUBO.
    Results.push_back(expandAddSubSat(N, DAG));
    return;
  }
  case ISD::ABS: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
          DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, N->getOperand(0));

    // Expand abs to Y = (sraiw X, 31); subw(xor(X, Y), Y)

    SDValue Src = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));

    // Freeze the source so we can increase it's use count.
    Src = DAG.getFreeze(Src);

    // Copy sign bit to all bits using the sraiw pattern.
    SDValue SignFill = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, Src,
                                   DAG.getValueType(MVT::i32));
    SignFill = DAG.getNode(ISD::SRA, DL, MVT::i64, SignFill,
                           DAG.getConstant(31, DL, MVT::i64));

    SDValue NewRes = DAG.getNode(ISD::XOR, DL, MVT::i64, Src, SignFill);
    NewRes = DAG.getNode(ISD::SUB, DL, MVT::i64, NewRes, SignFill);

    // NOTE: The result is only required to be anyextended, but sext is
    // consistent with type legalization of sub.
    NewRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, NewRes,
                         DAG.getValueType(MVT::i32));
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes));
    return;
  }
  case ISD::BITCAST: {
    EVT VT = N->getValueType(0);
    assert(VT.isInteger() && !VT.isVector() && "Unexpected VT!");
    SDValue Op0 = N->getOperand(0);
    EVT Op0VT = Op0.getValueType();
    MVT XLenVT = Subtarget.getXLenVT();
    if (VT == MVT::i16 && Op0VT == MVT::f16 && Subtarget.hasStdExtZfh()) {
      SDValue FPConv = DAG.getNode(RISCVISD::FMV_X_ANYEXTH, DL, XLenVT, Op0);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, FPConv));
    } else if (VT == MVT::i32 && Op0VT == MVT::f32 && Subtarget.is64Bit() &&
               Subtarget.hasStdExtF()) {
      SDValue FPConv =
          DAG.getNode(RISCVISD::FMV_X_ANYEXTW_RV64, DL, MVT::i64, Op0);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, FPConv));
    } else if (!VT.isVector() && Op0VT.isFixedLengthVector() &&
               isTypeLegal(Op0VT)) {
      // Custom-legalize bitcasts from fixed-length vector types to illegal
      // scalar types in order to improve codegen. Bitcast the vector to a
      // one-element vector type whose element type is the same as the result
      // type, and extract the first element.
      EVT BVT = EVT::getVectorVT(*DAG.getContext(), VT, 1);
      if (isTypeLegal(BVT)) {
        SDValue BVec = DAG.getBitcast(BVT, Op0);
        Results.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, BVec,
                                      DAG.getConstant(0, DL, XLenVT)));
      }
    }
    break;
  }
  case RISCVISD::GREV:
  case RISCVISD::GORC:
  case RISCVISD::SHFL: {
    MVT VT = N->getSimpleValueType(0);
    MVT XLenVT = Subtarget.getXLenVT();
    assert((VT == MVT::i16 || (VT == MVT::i32 && Subtarget.is64Bit())) &&
           "Unexpected custom legalisation");
    assert(isa<ConstantSDNode>(N->getOperand(1)) && "Expected constant");
    assert((Subtarget.hasStdExtZbp() ||
            (Subtarget.hasStdExtZbkb() && N->getOpcode() == RISCVISD::GREV &&
             N->getConstantOperandVal(1) == 7)) &&
           "Unexpected extension");
    SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, XLenVT, N->getOperand(0));
    SDValue NewOp1 =
        DAG.getNode(ISD::ZERO_EXTEND, DL, XLenVT, N->getOperand(1));
    SDValue NewRes = DAG.getNode(N->getOpcode(), DL, XLenVT, NewOp0, NewOp1);
    // ReplaceNodeResults requires we maintain the same type for the return
    // value.
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, NewRes));
    break;
  }
  case ISD::BSWAP:
  case ISD::BITREVERSE: {
    MVT VT = N->getSimpleValueType(0);
    MVT XLenVT = Subtarget.getXLenVT();
    assert((VT == MVT::i8 || VT == MVT::i16 ||
            (VT == MVT::i32 && Subtarget.is64Bit())) &&
           Subtarget.hasStdExtZbp() && "Unexpected custom legalisation");
    SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, XLenVT, N->getOperand(0));
    unsigned Imm = VT.getSizeInBits() - 1;
    // If this is BSWAP rather than BITREVERSE, clear the lower 3 bits.
    if (N->getOpcode() == ISD::BSWAP)
      Imm &= ~0x7U;
    SDValue GREVI = DAG.getNode(RISCVISD::GREV, DL, XLenVT, NewOp0,
                                DAG.getConstant(Imm, DL, XLenVT));
    // ReplaceNodeResults requires we maintain the same type for the return
    // value.
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, GREVI));
    break;
  }
  case ISD::FSHL:
  case ISD::FSHR: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           Subtarget.hasStdExtZbt() && "Unexpected custom legalisation");
    SDValue NewOp0 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
    SDValue NewOp1 =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
    SDValue NewShAmt =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
    // FSLW/FSRW take a 6 bit shift amount but i32 FSHL/FSHR only use 5 bits.
    // Mask the shift amount to 5 bits to prevent accidentally setting bit 5.
    NewShAmt = DAG.getNode(ISD::AND, DL, MVT::i64, NewShAmt,
                           DAG.getConstant(0x1f, DL, MVT::i64));
    // fshl and fshr concatenate their operands in the same order. fsrw and fslw
    // instruction use different orders. fshl will return its first operand for
    // shift of zero, fshr will return its second operand. fsl and fsr both
    // return rs1 so the ISD nodes need to have different operand orders.
    // Shift amount is in rs2.
    unsigned Opc = RISCVISD::FSLW;
    if (N->getOpcode() == ISD::FSHR) {
      std::swap(NewOp0, NewOp1);
      Opc = RISCVISD::FSRW;
    }
    SDValue NewOp = DAG.getNode(Opc, DL, MVT::i64, NewOp0, NewOp1, NewShAmt);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewOp));
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    // Custom-legalize an EXTRACT_VECTOR_ELT where XLEN<SEW, as the SEW element
    // type is illegal (currently only vXi64 RV32).
    // With vmv.x.s, when SEW > XLEN, only the least-significant XLEN bits are
    // transferred to the destination register. We issue two of these from the
    // upper- and lower- halves of the SEW-bit vector element, slid down to the
    // first element.
    SDValue Vec = N->getOperand(0);
    SDValue Idx = N->getOperand(1);

    // The vector type hasn't been legalized yet so we can't issue target
    // specific nodes if it needs legalization.
    // FIXME: We would manually legalize if it's important.
    if (!isTypeLegal(Vec.getValueType()))
      return;

    MVT VecVT = Vec.getSimpleValueType();

    assert(!Subtarget.is64Bit() && N->getValueType(0) == MVT::i64 &&
           VecVT.getVectorElementType() == MVT::i64 &&
           "Unexpected EXTRACT_VECTOR_ELT legalization");

    // If this is a fixed vector, we need to convert it to a scalable vector.
    MVT ContainerVT = VecVT;
    if (VecVT.isFixedLengthVector()) {
      ContainerVT = getContainerForFixedLengthVector(VecVT);
      Vec = convertToScalableVector(ContainerVT, Vec, DAG, Subtarget);
    }

    MVT XLenVT = Subtarget.getXLenVT();

    // Use a VL of 1 to avoid processing more elements than we need.
    MVT MaskVT = MVT::getVectorVT(MVT::i1, ContainerVT.getVectorElementCount());
    SDValue VL = DAG.getConstant(1, DL, XLenVT);
    SDValue Mask = DAG.getNode(RISCVISD::VMSET_VL, DL, MaskVT, VL);

    // Unless the index is known to be 0, we must slide the vector down to get
    // the desired element into index 0.
    if (!isNullConstant(Idx)) {
      Vec = DAG.getNode(RISCVISD::VSLIDEDOWN_VL, DL, ContainerVT,
                        DAG.getUNDEF(ContainerVT), Vec, Idx, Mask, VL);
    }

    // Extract the lower XLEN bits of the correct vector element.
    SDValue EltLo = DAG.getNode(RISCVISD::VMV_X_S, DL, XLenVT, Vec);

    // To extract the upper XLEN bits of the vector element, shift the first
    // element right by 32 bits and re-extract the lower XLEN bits.
    SDValue ThirtyTwoV = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, ContainerVT,
                                     DAG.getUNDEF(ContainerVT),
                                     DAG.getConstant(32, DL, XLenVT), VL);
    SDValue LShr32 = DAG.getNode(RISCVISD::SRL_VL, DL, ContainerVT, Vec,
                                 ThirtyTwoV, Mask, VL);

    SDValue EltHi = DAG.getNode(RISCVISD::VMV_X_S, DL, XLenVT, LShr32);

    Results.push_back(DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, EltLo, EltHi));
    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    switch (IntNo) {
    default:
      llvm_unreachable(
          "Don't know how to custom type legalize this intrinsic!");
    case Intrinsic::riscv_grev:
    case Intrinsic::riscv_gorc: {
      assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
             "Unexpected custom legalisation");
      SDValue NewOp1 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue NewOp2 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
      unsigned Opc =
          IntNo == Intrinsic::riscv_grev ? RISCVISD::GREVW : RISCVISD::GORCW;
      // If the control is a constant, promote the node by clearing any extra
      // bits bits in the control. isel will form greviw/gorciw if the result is
      // sign extended.
      if (isa<ConstantSDNode>(NewOp2)) {
        NewOp2 = DAG.getNode(ISD::AND, DL, MVT::i64, NewOp2,
                             DAG.getConstant(0x1f, DL, MVT::i64));
        Opc = IntNo == Intrinsic::riscv_grev ? RISCVISD::GREV : RISCVISD::GORC;
      }
      SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp1, NewOp2);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      break;
    }
    case Intrinsic::riscv_bcompress:
    case Intrinsic::riscv_bdecompress:
    case Intrinsic::riscv_bfp: {
      assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
             "Unexpected custom legalisation");
      Results.push_back(customLegalizeToWOpByIntr(N, DAG, IntNo));
      break;
    }
    case Intrinsic::riscv_fsl:
    case Intrinsic::riscv_fsr: {
      assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
             "Unexpected custom legalisation");
      SDValue NewOp1 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue NewOp2 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
      SDValue NewOp3 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(3));
      unsigned Opc = getRISCVWOpcodeByIntr(IntNo);
      SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp1, NewOp2, NewOp3);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      break;
    }
    case Intrinsic::riscv_orc_b: {
      // Lower to the GORCI encoding for orc.b with the operand extended.
      SDValue NewOp =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue Res = DAG.getNode(RISCVISD::GORC, DL, MVT::i64, NewOp,
                                DAG.getConstant(7, DL, MVT::i64));
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      return;
    }
    case Intrinsic::riscv_shfl:
    case Intrinsic::riscv_unshfl: {
      assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
             "Unexpected custom legalisation");
      SDValue NewOp1 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
      SDValue NewOp2 =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(2));
      unsigned Opc =
          IntNo == Intrinsic::riscv_shfl ? RISCVISD::SHFLW : RISCVISD::UNSHFLW;
      // There is no (UN)SHFLIW. If the control word is a constant, we can use
      // (UN)SHFLI with bit 4 of the control word cleared. The upper 32 bit half
      // will be shuffled the same way as the lower 32 bit half, but the two
      // halves won't cross.
      if (isa<ConstantSDNode>(NewOp2)) {
        NewOp2 = DAG.getNode(ISD::AND, DL, MVT::i64, NewOp2,
                             DAG.getConstant(0xf, DL, MVT::i64));
        Opc =
            IntNo == Intrinsic::riscv_shfl ? RISCVISD::SHFL : RISCVISD::UNSHFL;
      }
      SDValue Res = DAG.getNode(Opc, DL, MVT::i64, NewOp1, NewOp2);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Res));
      break;
    }
    case Intrinsic::riscv_vmv_x_s: {
      EVT VT = N->getValueType(0);
      MVT XLenVT = Subtarget.getXLenVT();
      if (VT.bitsLT(XLenVT)) {
        // Simple case just extract using vmv.x.s and truncate.
        SDValue Extract = DAG.getNode(RISCVISD::VMV_X_S, DL,
                                      Subtarget.getXLenVT(), N->getOperand(1));
        Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, Extract));
        return;
      }

      assert(VT == MVT::i64 && !Subtarget.is64Bit() &&
             "Unexpected custom legalization");

      // We need to do the move in two steps.
      SDValue Vec = N->getOperand(1);
      MVT VecVT = Vec.getSimpleValueType();

      // First extract the lower XLEN bits of the element.
      SDValue EltLo = DAG.getNode(RISCVISD::VMV_X_S, DL, XLenVT, Vec);

      // To extract the upper XLEN bits of the vector element, shift the first
      // element right by 32 bits and re-extract the lower XLEN bits.
      SDValue VL = DAG.getConstant(1, DL, XLenVT);
      MVT MaskVT = MVT::getVectorVT(MVT::i1, VecVT.getVectorElementCount());
      SDValue Mask = DAG.getNode(RISCVISD::VMSET_VL, DL, MaskVT, VL);
      SDValue ThirtyTwoV =
          DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VecVT, DAG.getUNDEF(VecVT),
                      DAG.getConstant(32, DL, XLenVT), VL);
      SDValue LShr32 =
          DAG.getNode(RISCVISD::SRL_VL, DL, VecVT, Vec, ThirtyTwoV, Mask, VL);
      SDValue EltHi = DAG.getNode(RISCVISD::VMV_X_S, DL, XLenVT, LShr32);

      Results.push_back(
          DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, EltLo, EltHi));
      break;
    }
    }
    break;
  }
  case ISD::VECREDUCE_ADD:
  case ISD::VECREDUCE_AND:
  case ISD::VECREDUCE_OR:
  case ISD::VECREDUCE_XOR:
  case ISD::VECREDUCE_SMAX:
  case ISD::VECREDUCE_UMAX:
  case ISD::VECREDUCE_SMIN:
  case ISD::VECREDUCE_UMIN:
    if (SDValue V = lowerVECREDUCE(SDValue(N, 0), DAG))
      Results.push_back(V);
    break;
  case ISD::VP_REDUCE_ADD:
  case ISD::VP_REDUCE_AND:
  case ISD::VP_REDUCE_OR:
  case ISD::VP_REDUCE_XOR:
  case ISD::VP_REDUCE_SMAX:
  case ISD::VP_REDUCE_UMAX:
  case ISD::VP_REDUCE_SMIN:
  case ISD::VP_REDUCE_UMIN:
    if (SDValue V = lowerVPREDUCE(SDValue(N, 0), DAG))
      Results.push_back(V);
    break;
  case ISD::FLT_ROUNDS_: {
    SDVTList VTs = DAG.getVTList(Subtarget.getXLenVT(), MVT::Other);
    SDValue Res = DAG.getNode(ISD::FLT_ROUNDS_, DL, VTs, N->getOperand(0));
    Results.push_back(Res.getValue(0));
    Results.push_back(Res.getValue(1));
    break;
  }
  }
}

// A structure to hold one of the bit-manipulation patterns below. Together, a
// SHL and non-SHL pattern may form a bit-manipulation pair on a single source:
//   (or (and (shl x, 1), 0xAAAAAAAA),
//       (and (srl x, 1), 0x55555555))
struct RISCVBitmanipPat {
  SDValue Op;
  unsigned ShAmt;
  bool IsSHL;

  bool formsPairWith(const RISCVBitmanipPat &Other) const {
    return Op == Other.Op && ShAmt == Other.ShAmt && IsSHL != Other.IsSHL;
  }
};

// Matches patterns of the form
//   (and (shl x, C2), (C1 << C2))
//   (and (srl x, C2), C1)
//   (shl (and x, C1), C2)
//   (srl (and x, (C1 << C2)), C2)
// Where C2 is a power of 2 and C1 has at least that many leading zeroes.
// The expected masks for each shift amount are specified in BitmanipMasks where
// BitmanipMasks[log2(C2)] specifies the expected C1 value.
// The max allowed shift amount is either XLen/2 or XLen/4 determined by whether
// BitmanipMasks contains 6 or 5 entries assuming that the maximum possible
// XLen is 64.
static Optional<RISCVBitmanipPat>
matchRISCVBitmanipPat(SDValue Op, ArrayRef<uint64_t> BitmanipMasks) {
  assert((BitmanipMasks.size() == 5 || BitmanipMasks.size() == 6) &&
         "Unexpected number of masks");
  Optional<uint64_t> Mask;
  // Optionally consume a mask around the shift operation.
  if (Op.getOpcode() == ISD::AND && isa<ConstantSDNode>(Op.getOperand(1))) {
    Mask = Op.getConstantOperandVal(1);
    Op = Op.getOperand(0);
  }
  if (Op.getOpcode() != ISD::SHL && Op.getOpcode() != ISD::SRL)
    return None;
  bool IsSHL = Op.getOpcode() == ISD::SHL;

  if (!isa<ConstantSDNode>(Op.getOperand(1)))
    return None;
  uint64_t ShAmt = Op.getConstantOperandVal(1);

  unsigned Width = Op.getValueType() == MVT::i64 ? 64 : 32;
  if (ShAmt >= Width || !isPowerOf2_64(ShAmt))
    return None;
  // If we don't have enough masks for 64 bit, then we must be trying to
  // match SHFL so we're only allowed to shift 1/4 of the width.
  if (BitmanipMasks.size() == 5 && ShAmt >= (Width / 2))
    return None;

  SDValue Src = Op.getOperand(0);

  // The expected mask is shifted left when the AND is found around SHL
  // patterns.
  //   ((x >> 1) & 0x55555555)
  //   ((x << 1) & 0xAAAAAAAA)
  bool SHLExpMask = IsSHL;

  if (!Mask) {
    // Sometimes LLVM keeps the mask as an operand of the shift, typically when
    // the mask is all ones: consume that now.
    if (Src.getOpcode() == ISD::AND && isa<ConstantSDNode>(Src.getOperand(1))) {
      Mask = Src.getConstantOperandVal(1);
      Src = Src.getOperand(0);
      // The expected mask is now in fact shifted left for SRL, so reverse the
      // decision.
      //   ((x & 0xAAAAAAAA) >> 1)
      //   ((x & 0x55555555) << 1)
      SHLExpMask = !SHLExpMask;
    } else {
      // Use a default shifted mask of all-ones if there's no AND, truncated
      // down to the expected width. This simplifies the logic later on.
      Mask = maskTrailingOnes<uint64_t>(Width);
      *Mask &= (IsSHL ? *Mask << ShAmt : *Mask >> ShAmt);
    }
  }

  unsigned MaskIdx = Log2_32(ShAmt);
  uint64_t ExpMask = BitmanipMasks[MaskIdx] & maskTrailingOnes<uint64_t>(Width);

  if (SHLExpMask)
    ExpMask <<= ShAmt;

  if (Mask != ExpMask)
    return None;

  return RISCVBitmanipPat{Src, (unsigned)ShAmt, IsSHL};
}

// Matches any of the following bit-manipulation patterns:
//   (and (shl x, 1), (0x55555555 << 1))
//   (and (srl x, 1), 0x55555555)
//   (shl (and x, 0x55555555), 1)
//   (srl (and x, (0x55555555 << 1)), 1)
// where the shift amount and mask may vary thus:
//   [1]  = 0x55555555 / 0xAAAAAAAA
//   [2]  = 0x33333333 / 0xCCCCCCCC
//   [4]  = 0x0F0F0F0F / 0xF0F0F0F0
//   [8]  = 0x00FF00FF / 0xFF00FF00
//   [16] = 0x0000FFFF / 0xFFFFFFFF
//   [32] = 0x00000000FFFFFFFF / 0xFFFFFFFF00000000 (for RV64)
static Optional<RISCVBitmanipPat> matchGREVIPat(SDValue Op) {
  // These are the unshifted masks which we use to match bit-manipulation
  // patterns. They may be shifted left in certain circumstances.
  static const uint64_t BitmanipMasks[] = {
      0x5555555555555555ULL, 0x3333333333333333ULL, 0x0F0F0F0F0F0F0F0FULL,
      0x00FF00FF00FF00FFULL, 0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL};

  return matchRISCVBitmanipPat(Op, BitmanipMasks);
}

// Match the following pattern as a GREVI(W) operation
//   (or (BITMANIP_SHL x), (BITMANIP_SRL x))
static SDValue combineORToGREV(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget) {
  assert(Subtarget.hasStdExtZbp() && "Expected Zbp extenson");
  EVT VT = Op.getValueType();

  if (VT == Subtarget.getXLenVT() || (Subtarget.is64Bit() && VT == MVT::i32)) {
    auto LHS = matchGREVIPat(Op.getOperand(0));
    auto RHS = matchGREVIPat(Op.getOperand(1));
    if (LHS && RHS && LHS->formsPairWith(*RHS)) {
      SDLoc DL(Op);
      return DAG.getNode(RISCVISD::GREV, DL, VT, LHS->Op,
                         DAG.getConstant(LHS->ShAmt, DL, VT));
    }
  }
  return SDValue();
}

// Matches any the following pattern as a GORCI(W) operation
// 1.  (or (GREVI x, shamt), x) if shamt is a power of 2
// 2.  (or x, (GREVI x, shamt)) if shamt is a power of 2
// 3.  (or (or (BITMANIP_SHL x), x), (BITMANIP_SRL x))
// Note that with the variant of 3.,
//     (or (or (BITMANIP_SHL x), (BITMANIP_SRL x)), x)
// the inner pattern will first be matched as GREVI and then the outer
// pattern will be matched to GORC via the first rule above.
// 4.  (or (rotl/rotr x, bitwidth/2), x)
static SDValue combineORToGORC(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget) {
  assert(Subtarget.hasStdExtZbp() && "Expected Zbp extenson");
  EVT VT = Op.getValueType();

  if (VT == Subtarget.getXLenVT() || (Subtarget.is64Bit() && VT == MVT::i32)) {
    SDLoc DL(Op);
    SDValue Op0 = Op.getOperand(0);
    SDValue Op1 = Op.getOperand(1);

    auto MatchOROfReverse = [&](SDValue Reverse, SDValue X) {
      if (Reverse.getOpcode() == RISCVISD::GREV && Reverse.getOperand(0) == X &&
          isa<ConstantSDNode>(Reverse.getOperand(1)) &&
          isPowerOf2_32(Reverse.getConstantOperandVal(1)))
        return DAG.getNode(RISCVISD::GORC, DL, VT, X, Reverse.getOperand(1));
      // We can also form GORCI from ROTL/ROTR by half the bitwidth.
      if ((Reverse.getOpcode() == ISD::ROTL ||
           Reverse.getOpcode() == ISD::ROTR) &&
          Reverse.getOperand(0) == X &&
          isa<ConstantSDNode>(Reverse.getOperand(1))) {
        uint64_t RotAmt = Reverse.getConstantOperandVal(1);
        if (RotAmt == (VT.getSizeInBits() / 2))
          return DAG.getNode(RISCVISD::GORC, DL, VT, X,
                             DAG.getConstant(RotAmt, DL, VT));
      }
      return SDValue();
    };

    // Check for either commutable permutation of (or (GREVI x, shamt), x)
    if (SDValue V = MatchOROfReverse(Op0, Op1))
      return V;
    if (SDValue V = MatchOROfReverse(Op1, Op0))
      return V;

    // OR is commutable so canonicalize its OR operand to the left
    if (Op0.getOpcode() != ISD::OR && Op1.getOpcode() == ISD::OR)
      std::swap(Op0, Op1);
    if (Op0.getOpcode() != ISD::OR)
      return SDValue();
    SDValue OrOp0 = Op0.getOperand(0);
    SDValue OrOp1 = Op0.getOperand(1);
    auto LHS = matchGREVIPat(OrOp0);
    // OR is commutable so swap the operands and try again: x might have been
    // on the left
    if (!LHS) {
      std::swap(OrOp0, OrOp1);
      LHS = matchGREVIPat(OrOp0);
    }
    auto RHS = matchGREVIPat(Op1);
    if (LHS && RHS && LHS->formsPairWith(*RHS) && LHS->Op == OrOp1) {
      return DAG.getNode(RISCVISD::GORC, DL, VT, LHS->Op,
                         DAG.getConstant(LHS->ShAmt, DL, VT));
    }
  }
  return SDValue();
}

// Matches any of the following bit-manipulation patterns:
//   (and (shl x, 1), (0x22222222 << 1))
//   (and (srl x, 1), 0x22222222)
//   (shl (and x, 0x22222222), 1)
//   (srl (and x, (0x22222222 << 1)), 1)
// where the shift amount and mask may vary thus:
//   [1]  = 0x22222222 / 0x44444444
//   [2]  = 0x0C0C0C0C / 0x3C3C3C3C
//   [4]  = 0x00F000F0 / 0x0F000F00
//   [8]  = 0x0000FF00 / 0x00FF0000
//   [16] = 0x00000000FFFF0000 / 0x0000FFFF00000000 (for RV64)
static Optional<RISCVBitmanipPat> matchSHFLPat(SDValue Op) {
  // These are the unshifted masks which we use to match bit-manipulation
  // patterns. They may be shifted left in certain circumstances.
  static const uint64_t BitmanipMasks[] = {
      0x2222222222222222ULL, 0x0C0C0C0C0C0C0C0CULL, 0x00F000F000F000F0ULL,
      0x0000FF000000FF00ULL, 0x00000000FFFF0000ULL};

  return matchRISCVBitmanipPat(Op, BitmanipMasks);
}

// Match (or (or (SHFL_SHL x), (SHFL_SHR x)), (SHFL_AND x)
static SDValue combineORToSHFL(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget) {
  assert(Subtarget.hasStdExtZbp() && "Expected Zbp extenson");
  EVT VT = Op.getValueType();

  if (VT != MVT::i32 && VT != Subtarget.getXLenVT())
    return SDValue();

  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);

  // Or is commutable so canonicalize the second OR to the LHS.
  if (Op0.getOpcode() != ISD::OR)
    std::swap(Op0, Op1);
  if (Op0.getOpcode() != ISD::OR)
    return SDValue();

  // We found an inner OR, so our operands are the operands of the inner OR
  // and the other operand of the outer OR.
  SDValue A = Op0.getOperand(0);
  SDValue B = Op0.getOperand(1);
  SDValue C = Op1;

  auto Match1 = matchSHFLPat(A);
  auto Match2 = matchSHFLPat(B);

  // If neither matched, we failed.
  if (!Match1 && !Match2)
    return SDValue();

  // We had at least one match. if one failed, try the remaining C operand.
  if (!Match1) {
    std::swap(A, C);
    Match1 = matchSHFLPat(A);
    if (!Match1)
      return SDValue();
  } else if (!Match2) {
    std::swap(B, C);
    Match2 = matchSHFLPat(B);
    if (!Match2)
      return SDValue();
  }
  assert(Match1 && Match2);

  // Make sure our matches pair up.
  if (!Match1->formsPairWith(*Match2))
    return SDValue();

  // All the remains is to make sure C is an AND with the same input, that masks
  // out the bits that are being shuffled.
  if (C.getOpcode() != ISD::AND || !isa<ConstantSDNode>(C.getOperand(1)) ||
      C.getOperand(0) != Match1->Op)
    return SDValue();

  uint64_t Mask = C.getConstantOperandVal(1);

  static const uint64_t BitmanipMasks[] = {
      0x9999999999999999ULL, 0xC3C3C3C3C3C3C3C3ULL, 0xF00FF00FF00FF00FULL,
      0xFF0000FFFF0000FFULL, 0xFFFF00000000FFFFULL,
  };

  unsigned Width = Op.getValueType() == MVT::i64 ? 64 : 32;
  unsigned MaskIdx = Log2_32(Match1->ShAmt);
  uint64_t ExpMask = BitmanipMasks[MaskIdx] & maskTrailingOnes<uint64_t>(Width);

  if (Mask != ExpMask)
    return SDValue();

  SDLoc DL(Op);
  return DAG.getNode(RISCVISD::SHFL, DL, VT, Match1->Op,
                     DAG.getConstant(Match1->ShAmt, DL, VT));
}

// Optimize (add (shl x, c0), (shl y, c1)) ->
//          (SLLI (SH*ADD x, y), c0), if c1-c0 equals to [1|2|3].
static SDValue transformAddShlImm(SDNode *N, SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget) {
  // Perform this optimization only in the zba extension.
  if (!Subtarget.hasStdExtZba())
    return SDValue();

  // Skip for vector types and larger types.
  EVT VT = N->getValueType(0);
  if (VT.isVector() || VT.getSizeInBits() > Subtarget.getXLen())
    return SDValue();

  // The two operand nodes must be SHL and have no other use.
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  if (N0->getOpcode() != ISD::SHL || N1->getOpcode() != ISD::SHL ||
      !N0->hasOneUse() || !N1->hasOneUse())
    return SDValue();

  // Check c0 and c1.
  auto *N0C = dyn_cast<ConstantSDNode>(N0->getOperand(1));
  auto *N1C = dyn_cast<ConstantSDNode>(N1->getOperand(1));
  if (!N0C || !N1C)
    return SDValue();
  int64_t C0 = N0C->getSExtValue();
  int64_t C1 = N1C->getSExtValue();
  if (C0 <= 0 || C1 <= 0)
    return SDValue();

  // Skip if SH1ADD/SH2ADD/SH3ADD are not applicable.
  int64_t Bits = std::min(C0, C1);
  int64_t Diff = std::abs(C0 - C1);
  if (Diff != 1 && Diff != 2 && Diff != 3)
    return SDValue();

  // Build nodes.
  SDLoc DL(N);
  SDValue NS = (C0 < C1) ? N0->getOperand(0) : N1->getOperand(0);
  SDValue NL = (C0 > C1) ? N0->getOperand(0) : N1->getOperand(0);
  SDValue NA0 =
      DAG.getNode(ISD::SHL, DL, VT, NL, DAG.getConstant(Diff, DL, VT));
  SDValue NA1 = DAG.getNode(ISD::ADD, DL, VT, NA0, NS);
  return DAG.getNode(ISD::SHL, DL, VT, NA1, DAG.getConstant(Bits, DL, VT));
}

// Combine
// ROTR ((GREVI x, 24), 16) -> (GREVI x, 8) for RV32
// ROTL ((GREVI x, 24), 16) -> (GREVI x, 8) for RV32
// ROTR ((GREVI x, 56), 32) -> (GREVI x, 24) for RV64
// ROTL ((GREVI x, 56), 32) -> (GREVI x, 24) for RV64
// RORW ((GREVI x, 24), 16) -> (GREVIW x, 8) for RV64
// ROLW ((GREVI x, 24), 16) -> (GREVIW x, 8) for RV64
// The grev patterns represents BSWAP.
// FIXME: This can be generalized to any GREV. We just need to toggle the MSB
// off the grev.
static SDValue combineROTR_ROTL_RORW_ROLW(SDNode *N, SelectionDAG &DAG,
                                          const RISCVSubtarget &Subtarget) {
  bool IsWInstruction =
      N->getOpcode() == RISCVISD::RORW || N->getOpcode() == RISCVISD::ROLW;
  assert((N->getOpcode() == ISD::ROTR || N->getOpcode() == ISD::ROTL ||
          IsWInstruction) &&
         "Unexpected opcode!");
  SDValue Src = N->getOperand(0);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  if (!Subtarget.hasStdExtZbp() || Src.getOpcode() != RISCVISD::GREV)
    return SDValue();

  if (!isa<ConstantSDNode>(N->getOperand(1)) ||
      !isa<ConstantSDNode>(Src.getOperand(1)))
    return SDValue();

  unsigned BitWidth = IsWInstruction ? 32 : VT.getSizeInBits();
  assert(isPowerOf2_32(BitWidth) && "Expected a power of 2");

  // Needs to be a rotate by half the bitwidth for ROTR/ROTL or by 16 for
  // RORW/ROLW. And the grev should be the encoding for bswap for this width.
  unsigned ShAmt1 = N->getConstantOperandVal(1);
  unsigned ShAmt2 = Src.getConstantOperandVal(1);
  if (BitWidth < 32 || ShAmt1 != (BitWidth / 2) || ShAmt2 != (BitWidth - 8))
    return SDValue();

  Src = Src.getOperand(0);

  // Toggle bit the MSB of the shift.
  unsigned CombinedShAmt = ShAmt1 ^ ShAmt2;
  if (CombinedShAmt == 0)
    return Src;

  SDValue Res = DAG.getNode(
      RISCVISD::GREV, DL, VT, Src,
      DAG.getConstant(CombinedShAmt, DL, N->getOperand(1).getValueType()));
  if (!IsWInstruction)
    return Res;

  // Sign extend the result to match the behavior of the rotate. This will be
  // selected to GREVIW in isel.
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, VT, Res,
                     DAG.getValueType(MVT::i32));
}

// Combine (GREVI (GREVI x, C2), C1) -> (GREVI x, C1^C2) when C1^C2 is
// non-zero, and to x when it is. Any repeated GREVI stage undoes itself.
// Combine (GORCI (GORCI x, C2), C1) -> (GORCI x, C1|C2). Repeated stage does
// not undo itself, but they are redundant.
static SDValue combineGREVI_GORCI(SDNode *N, SelectionDAG &DAG) {
  bool IsGORC = N->getOpcode() == RISCVISD::GORC;
  assert((IsGORC || N->getOpcode() == RISCVISD::GREV) && "Unexpected opcode");
  SDValue Src = N->getOperand(0);

  if (Src.getOpcode() != N->getOpcode())
    return SDValue();

  if (!isa<ConstantSDNode>(N->getOperand(1)) ||
      !isa<ConstantSDNode>(Src.getOperand(1)))
    return SDValue();

  unsigned ShAmt1 = N->getConstantOperandVal(1);
  unsigned ShAmt2 = Src.getConstantOperandVal(1);
  Src = Src.getOperand(0);

  unsigned CombinedShAmt;
  if (IsGORC)
    CombinedShAmt = ShAmt1 | ShAmt2;
  else
    CombinedShAmt = ShAmt1 ^ ShAmt2;

  if (CombinedShAmt == 0)
    return Src;

  SDLoc DL(N);
  return DAG.getNode(
      N->getOpcode(), DL, N->getValueType(0), Src,
      DAG.getConstant(CombinedShAmt, DL, N->getOperand(1).getValueType()));
}

// Combine a constant select operand into its use:
//
// (and (select cond, -1, c), x)
//   -> (select cond, x, (and x, c))  [AllOnes=1]
// (or  (select cond, 0, c), x)
//   -> (select cond, x, (or x, c))  [AllOnes=0]
// (xor (select cond, 0, c), x)
//   -> (select cond, x, (xor x, c))  [AllOnes=0]
// (add (select cond, 0, c), x)
//   -> (select cond, x, (add x, c))  [AllOnes=0]
// (sub x, (select cond, 0, c))
//   -> (select cond, x, (sub x, c))  [AllOnes=0]
static SDValue combineSelectAndUse(SDNode *N, SDValue Slct, SDValue OtherOp,
                                   SelectionDAG &DAG, bool AllOnes) {
  EVT VT = N->getValueType(0);

  // Skip vectors.
  if (VT.isVector())
    return SDValue();

  if ((Slct.getOpcode() != ISD::SELECT &&
       Slct.getOpcode() != RISCVISD::SELECT_CC) ||
      !Slct.hasOneUse())
    return SDValue();

  auto isZeroOrAllOnes = [](SDValue N, bool AllOnes) {
    return AllOnes ? isAllOnesConstant(N) : isNullConstant(N);
  };

  bool SwapSelectOps;
  unsigned OpOffset = Slct.getOpcode() == RISCVISD::SELECT_CC ? 2 : 0;
  SDValue TrueVal = Slct.getOperand(1 + OpOffset);
  SDValue FalseVal = Slct.getOperand(2 + OpOffset);
  SDValue NonConstantVal;
  if (isZeroOrAllOnes(TrueVal, AllOnes)) {
    SwapSelectOps = false;
    NonConstantVal = FalseVal;
  } else if (isZeroOrAllOnes(FalseVal, AllOnes)) {
    SwapSelectOps = true;
    NonConstantVal = TrueVal;
  } else
    return SDValue();

  // Slct is now know to be the desired identity constant when CC is true.
  TrueVal = OtherOp;
  FalseVal = DAG.getNode(N->getOpcode(), SDLoc(N), VT, OtherOp, NonConstantVal);
  // Unless SwapSelectOps says the condition should be false.
  if (SwapSelectOps)
    std::swap(TrueVal, FalseVal);

  if (Slct.getOpcode() == RISCVISD::SELECT_CC)
    return DAG.getNode(RISCVISD::SELECT_CC, SDLoc(N), VT,
                       {Slct.getOperand(0), Slct.getOperand(1),
                        Slct.getOperand(2), TrueVal, FalseVal});

  return DAG.getNode(ISD::SELECT, SDLoc(N), VT,
                     {Slct.getOperand(0), TrueVal, FalseVal});
}

// Attempt combineSelectAndUse on each operand of a commutative operator N.
static SDValue combineSelectAndUseCommutative(SDNode *N, SelectionDAG &DAG,
                                              bool AllOnes) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  if (SDValue Result = combineSelectAndUse(N, N0, N1, DAG, AllOnes))
    return Result;
  if (SDValue Result = combineSelectAndUse(N, N1, N0, DAG, AllOnes))
    return Result;
  return SDValue();
}

// Transform (add (mul x, c0), c1) ->
//           (add (mul (add x, c1/c0), c0), c1%c0).
// if c1/c0 and c1%c0 are simm12, while c1 is not. A special corner case
// that should be excluded is when c0*(c1/c0) is simm12, which will lead
// to an infinite loop in DAGCombine if transformed.
// Or transform (add (mul x, c0), c1) ->
//              (add (mul (add x, c1/c0+1), c0), c1%c0-c0),
// if c1/c0+1 and c1%c0-c0 are simm12, while c1 is not. A special corner
// case that should be excluded is when c0*(c1/c0+1) is simm12, which will
// lead to an infinite loop in DAGCombine if transformed.
// Or transform (add (mul x, c0), c1) ->
//              (add (mul (add x, c1/c0-1), c0), c1%c0+c0),
// if c1/c0-1 and c1%c0+c0 are simm12, while c1 is not. A special corner
// case that should be excluded is when c0*(c1/c0-1) is simm12, which will
// lead to an infinite loop in DAGCombine if transformed.
// Or transform (add (mul x, c0), c1) ->
//              (mul (add x, c1/c0), c0).
// if c1%c0 is zero, and c1/c0 is simm12 while c1 is not.
static SDValue transformAddImmMulImm(SDNode *N, SelectionDAG &DAG,
                                     const RISCVSubtarget &Subtarget) {
  // Skip for vector types and larger types.
  EVT VT = N->getValueType(0);
  if (VT.isVector() || VT.getSizeInBits() > Subtarget.getXLen())
    return SDValue();
  // The first operand node must be a MUL and has no other use.
  SDValue N0 = N->getOperand(0);
  if (!N0->hasOneUse() || N0->getOpcode() != ISD::MUL)
    return SDValue();
  // Check if c0 and c1 match above conditions.
  auto *N0C = dyn_cast<ConstantSDNode>(N0->getOperand(1));
  auto *N1C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!N0C || !N1C)
    return SDValue();
  // If N0C has multiple uses it's possible one of the cases in
  // DAGCombiner::isMulAddWithConstProfitable will be true, which would result
  // in an infinite loop.
  if (!N0C->hasOneUse())
    return SDValue();
  int64_t C0 = N0C->getSExtValue();
  int64_t C1 = N1C->getSExtValue();
  int64_t CA, CB;
  if (C0 == -1 || C0 == 0 || C0 == 1 || isInt<12>(C1))
    return SDValue();
  // Search for proper CA (non-zero) and CB that both are simm12.
  if ((C1 / C0) != 0 && isInt<12>(C1 / C0) && isInt<12>(C1 % C0) &&
      !isInt<12>(C0 * (C1 / C0))) {
    CA = C1 / C0;
    CB = C1 % C0;
  } else if ((C1 / C0 + 1) != 0 && isInt<12>(C1 / C0 + 1) &&
             isInt<12>(C1 % C0 - C0) && !isInt<12>(C0 * (C1 / C0 + 1))) {
    CA = C1 / C0 + 1;
    CB = C1 % C0 - C0;
  } else if ((C1 / C0 - 1) != 0 && isInt<12>(C1 / C0 - 1) &&
             isInt<12>(C1 % C0 + C0) && !isInt<12>(C0 * (C1 / C0 - 1))) {
    CA = C1 / C0 - 1;
    CB = C1 % C0 + C0;
  } else
    return SDValue();
  // Build new nodes (add (mul (add x, c1/c0), c0), c1%c0).
  SDLoc DL(N);
  SDValue New0 = DAG.getNode(ISD::ADD, DL, VT, N0->getOperand(0),
                             DAG.getConstant(CA, DL, VT));
  SDValue New1 =
      DAG.getNode(ISD::MUL, DL, VT, New0, DAG.getConstant(C0, DL, VT));
  return DAG.getNode(ISD::ADD, DL, VT, New1, DAG.getConstant(CB, DL, VT));
}

static SDValue performADDCombine(SDNode *N, SelectionDAG &DAG,
                                 const RISCVSubtarget &Subtarget) {
  if (SDValue V = transformAddImmMulImm(N, DAG, Subtarget))
    return V;
  if (SDValue V = transformAddShlImm(N, DAG, Subtarget))
    return V;
  // fold (add (select lhs, rhs, cc, 0, y), x) ->
  //      (select lhs, rhs, cc, x, (add x, y))
  return combineSelectAndUseCommutative(N, DAG, /*AllOnes*/ false);
}

static SDValue performSUBCombine(SDNode *N, SelectionDAG &DAG) {
  // fold (sub x, (select lhs, rhs, cc, 0, y)) ->
  //      (select lhs, rhs, cc, x, (sub x, y))
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  return combineSelectAndUse(N, N1, N0, DAG, /*AllOnes*/ false);
}

static SDValue performANDCombine(SDNode *N, SelectionDAG &DAG) {
  // fold (and (select lhs, rhs, cc, -1, y), x) ->
  //      (select lhs, rhs, cc, x, (and x, y))
  return combineSelectAndUseCommutative(N, DAG, /*AllOnes*/ true);
}

static SDValue performORCombine(SDNode *N, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget) {
  if (Subtarget.hasStdExtZbp()) {
    if (auto GREV = combineORToGREV(SDValue(N, 0), DAG, Subtarget))
      return GREV;
    if (auto GORC = combineORToGORC(SDValue(N, 0), DAG, Subtarget))
      return GORC;
    if (auto SHFL = combineORToSHFL(SDValue(N, 0), DAG, Subtarget))
      return SHFL;
  }

  // fold (or (select cond, 0, y), x) ->
  //      (select cond, x, (or x, y))
  return combineSelectAndUseCommutative(N, DAG, /*AllOnes*/ false);
}

static SDValue performXORCombine(SDNode *N, SelectionDAG &DAG) {
  // fold (xor (select cond, 0, y), x) ->
  //      (select cond, x, (xor x, y))
  return combineSelectAndUseCommutative(N, DAG, /*AllOnes*/ false);
}

static SDValue
performSIGN_EXTEND_INREGCombine(SDNode *N, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget) {
  SDValue Src = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // Fold (sext_inreg (fmv_x_anyexth X), i16) -> (fmv_x_signexth X)
  if (Src.getOpcode() == RISCVISD::FMV_X_ANYEXTH &&
      cast<VTSDNode>(N->getOperand(1))->getVT().bitsGE(MVT::i16))
    return DAG.getNode(RISCVISD::FMV_X_SIGNEXTH, SDLoc(N), VT,
                       Src.getOperand(0));

  // Fold (i64 (sext_inreg (abs X), i32)) ->
  // (i64 (smax (sext_inreg (neg X), i32), X)) if X has more than 32 sign bits.
  // The (sext_inreg (neg X), i32) will be selected to negw by isel. This
  // pattern occurs after type legalization of (i32 (abs X)) on RV64 if the user
  // of the (i32 (abs X)) is a sext or setcc or something else that causes type
  // legalization to add a sext_inreg after the abs. The (i32 (abs X)) will have
  // been type legalized to (i64 (abs (sext_inreg X, i32))), but the sext_inreg
  // may get combined into an earlier operation so we need to use
  // ComputeNumSignBits.
  // NOTE: (i64 (sext_inreg (abs X), i32)) can also be created for
  // (i64 (ashr (shl (abs X), 32), 32)) without any type legalization so
  // we can't assume that X has 33 sign bits. We must check.
  if (Subtarget.hasStdExtZbb() && Subtarget.is64Bit() &&
      Src.getOpcode() == ISD::ABS && Src.hasOneUse() && VT == MVT::i64 &&
      cast<VTSDNode>(N->getOperand(1))->getVT() == MVT::i32 &&
      DAG.ComputeNumSignBits(Src.getOperand(0)) > 32) {
    SDLoc DL(N);
    SDValue Freeze = DAG.getFreeze(Src.getOperand(0));
    SDValue Neg =
        DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, MVT::i64), Freeze);
    Neg = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, Neg,
                      DAG.getValueType(MVT::i32));
    return DAG.getNode(ISD::SMAX, DL, MVT::i64, Freeze, Neg);
  }

  return SDValue();
}

// Try to form vwadd(u).wv/wx or vwsub(u).wv/wx. It might later be optimized to
// vwadd(u).vv/vx or vwsub(u).vv/vx.
static SDValue combineADDSUB_VLToVWADDSUB_VL(SDNode *N, SelectionDAG &DAG,
                                             bool Commute = false) {
  assert((N->getOpcode() == RISCVISD::ADD_VL ||
          N->getOpcode() == RISCVISD::SUB_VL) &&
         "Unexpected opcode");
  bool IsAdd = N->getOpcode() == RISCVISD::ADD_VL;
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  if (Commute)
    std::swap(Op0, Op1);

  MVT VT = N->getSimpleValueType(0);

  // Determine the narrow size for a widening add/sub.
  unsigned NarrowSize = VT.getScalarSizeInBits() / 2;
  MVT NarrowVT = MVT::getVectorVT(MVT::getIntegerVT(NarrowSize),
                                  VT.getVectorElementCount());

  SDValue Mask = N->getOperand(2);
  SDValue VL = N->getOperand(3);

  SDLoc DL(N);

  // If the RHS is a sext or zext, we can form a widening op.
  if ((Op1.getOpcode() == RISCVISD::VZEXT_VL ||
       Op1.getOpcode() == RISCVISD::VSEXT_VL) &&
      Op1.hasOneUse() && Op1.getOperand(1) == Mask && Op1.getOperand(2) == VL) {
    unsigned ExtOpc = Op1.getOpcode();
    Op1 = Op1.getOperand(0);
    // Re-introduce narrower extends if needed.
    if (Op1.getValueType() != NarrowVT)
      Op1 = DAG.getNode(ExtOpc, DL, NarrowVT, Op1, Mask, VL);

    unsigned WOpc;
    if (ExtOpc == RISCVISD::VSEXT_VL)
      WOpc = IsAdd ? RISCVISD::VWADD_W_VL : RISCVISD::VWSUB_W_VL;
    else
      WOpc = IsAdd ? RISCVISD::VWADDU_W_VL : RISCVISD::VWSUBU_W_VL;

    return DAG.getNode(WOpc, DL, VT, Op0, Op1, Mask, VL);
  }

  // FIXME: Is it useful to form a vwadd.wx or vwsub.wx if it removes a scalar
  // sext/zext?

  return SDValue();
}

// Try to convert vwadd(u).wv/wx or vwsub(u).wv/wx to vwadd(u).vv/vx or
// vwsub(u).vv/vx.
static SDValue combineVWADD_W_VL_VWSUB_W_VL(SDNode *N, SelectionDAG &DAG) {
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  SDValue Mask = N->getOperand(2);
  SDValue VL = N->getOperand(3);

  MVT VT = N->getSimpleValueType(0);
  MVT NarrowVT = Op1.getSimpleValueType();
  unsigned NarrowSize = NarrowVT.getScalarSizeInBits();

  unsigned VOpc;
  switch (N->getOpcode()) {
  default: llvm_unreachable("Unexpected opcode");
  case RISCVISD::VWADD_W_VL:  VOpc = RISCVISD::VWADD_VL;  break;
  case RISCVISD::VWSUB_W_VL:  VOpc = RISCVISD::VWSUB_VL;  break;
  case RISCVISD::VWADDU_W_VL: VOpc = RISCVISD::VWADDU_VL; break;
  case RISCVISD::VWSUBU_W_VL: VOpc = RISCVISD::VWSUBU_VL; break;
  }

  bool IsSigned = N->getOpcode() == RISCVISD::VWADD_W_VL ||
                  N->getOpcode() == RISCVISD::VWSUB_W_VL;

  SDLoc DL(N);

  // If the LHS is a sext or zext, we can narrow this op to the same size as
  // the RHS.
  if (((Op0.getOpcode() == RISCVISD::VZEXT_VL && !IsSigned) ||
       (Op0.getOpcode() == RISCVISD::VSEXT_VL && IsSigned)) &&
      Op0.hasOneUse() && Op0.getOperand(1) == Mask && Op0.getOperand(2) == VL) {
    unsigned ExtOpc = Op0.getOpcode();
    Op0 = Op0.getOperand(0);
    // Re-introduce narrower extends if needed.
    if (Op0.getValueType() != NarrowVT)
      Op0 = DAG.getNode(ExtOpc, DL, NarrowVT, Op0, Mask, VL);
    return DAG.getNode(VOpc, DL, VT, Op0, Op1, Mask, VL);
  }

  bool IsAdd = N->getOpcode() == RISCVISD::VWADD_W_VL ||
               N->getOpcode() == RISCVISD::VWADDU_W_VL;

  // Look for splats on the left hand side of a vwadd(u).wv. We might be able
  // to commute and use a vwadd(u).vx instead.
  if (IsAdd && Op0.getOpcode() == RISCVISD::VMV_V_X_VL &&
      Op0.getOperand(0).isUndef() && Op0.getOperand(2) == VL) {
    Op0 = Op0.getOperand(1);

    // See if have enough sign bits or zero bits in the scalar to use a
    // widening add/sub by splatting to smaller element size.
    unsigned EltBits = VT.getScalarSizeInBits();
    unsigned ScalarBits = Op0.getValueSizeInBits();
    // Make sure we're getting all element bits from the scalar register.
    // FIXME: Support implicit sign extension of vmv.v.x?
    if (ScalarBits < EltBits)
      return SDValue();

    if (IsSigned) {
      if (DAG.ComputeNumSignBits(Op0) <= (ScalarBits - NarrowSize))
        return SDValue();
    } else {
      APInt Mask = APInt::getBitsSetFrom(ScalarBits, NarrowSize);
      if (!DAG.MaskedValueIsZero(Op0, Mask))
        return SDValue();
    }

    Op0 = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, NarrowVT,
                      DAG.getUNDEF(NarrowVT), Op0, VL);
    return DAG.getNode(VOpc, DL, VT, Op1, Op0, Mask, VL);
  }

  return SDValue();
}

// Try to form VWMUL, VWMULU or VWMULSU.
// TODO: Support VWMULSU.vx with a sign extend Op and a splat of scalar Op.
static SDValue combineMUL_VLToVWMUL_VL(SDNode *N, SelectionDAG &DAG,
                                       bool Commute) {
  assert(N->getOpcode() == RISCVISD::MUL_VL && "Unexpected opcode");
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  if (Commute)
    std::swap(Op0, Op1);

  bool IsSignExt = Op0.getOpcode() == RISCVISD::VSEXT_VL;
  bool IsZeroExt = Op0.getOpcode() == RISCVISD::VZEXT_VL;
  bool IsVWMULSU = IsSignExt && Op1.getOpcode() == RISCVISD::VZEXT_VL;
  if ((!IsSignExt && !IsZeroExt) || !Op0.hasOneUse())
    return SDValue();

  SDValue Mask = N->getOperand(2);
  SDValue VL = N->getOperand(3);

  // Make sure the mask and VL match.
  if (Op0.getOperand(1) != Mask || Op0.getOperand(2) != VL)
    return SDValue();

  MVT VT = N->getSimpleValueType(0);

  // Determine the narrow size for a widening multiply.
  unsigned NarrowSize = VT.getScalarSizeInBits() / 2;
  MVT NarrowVT = MVT::getVectorVT(MVT::getIntegerVT(NarrowSize),
                                  VT.getVectorElementCount());

  SDLoc DL(N);

  // See if the other operand is the same opcode.
  if (IsVWMULSU || Op0.getOpcode() == Op1.getOpcode()) {
    if (!Op1.hasOneUse())
      return SDValue();

    // Make sure the mask and VL match.
    if (Op1.getOperand(1) != Mask || Op1.getOperand(2) != VL)
      return SDValue();

    Op1 = Op1.getOperand(0);
  } else if (Op1.getOpcode() == RISCVISD::VMV_V_X_VL) {
    // The operand is a splat of a scalar.

    // The pasthru must be undef for tail agnostic
    if (!Op1.getOperand(0).isUndef())
      return SDValue();
    // The VL must be the same.
    if (Op1.getOperand(2) != VL)
      return SDValue();

    // Get the scalar value.
    Op1 = Op1.getOperand(1);

    // See if have enough sign bits or zero bits in the scalar to use a
    // widening multiply by splatting to smaller element size.
    unsigned EltBits = VT.getScalarSizeInBits();
    unsigned ScalarBits = Op1.getValueSizeInBits();
    // Make sure we're getting all element bits from the scalar register.
    // FIXME: Support implicit sign extension of vmv.v.x?
    if (ScalarBits < EltBits)
      return SDValue();

    // If the LHS is a sign extend, try to use vwmul.
    if (IsSignExt && DAG.ComputeNumSignBits(Op1) > (ScalarBits - NarrowSize)) {
      // Can use vwmul.
    } else {
      // Otherwise try to use vwmulu or vwmulsu.
      APInt Mask = APInt::getBitsSetFrom(ScalarBits, NarrowSize);
      if (DAG.MaskedValueIsZero(Op1, Mask))
        IsVWMULSU = IsSignExt;
      else
        return SDValue();
    }

    Op1 = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, NarrowVT,
                      DAG.getUNDEF(NarrowVT), Op1, VL);
  } else
    return SDValue();

  Op0 = Op0.getOperand(0);

  // Re-introduce narrower extends if needed.
  unsigned ExtOpc = IsSignExt ? RISCVISD::VSEXT_VL : RISCVISD::VZEXT_VL;
  if (Op0.getValueType() != NarrowVT)
    Op0 = DAG.getNode(ExtOpc, DL, NarrowVT, Op0, Mask, VL);
  // vwmulsu requires second operand to be zero extended.
  ExtOpc = IsVWMULSU ? RISCVISD::VZEXT_VL : ExtOpc;
  if (Op1.getValueType() != NarrowVT)
    Op1 = DAG.getNode(ExtOpc, DL, NarrowVT, Op1, Mask, VL);

  unsigned WMulOpc = RISCVISD::VWMULSU_VL;
  if (!IsVWMULSU)
    WMulOpc = IsSignExt ? RISCVISD::VWMUL_VL : RISCVISD::VWMULU_VL;
  return DAG.getNode(WMulOpc, DL, VT, Op0, Op1, Mask, VL);
}

static RISCVFPRndMode::RoundingMode matchRoundingOp(SDValue Op) {
  switch (Op.getOpcode()) {
  case ISD::FROUNDEVEN: return RISCVFPRndMode::RNE;
  case ISD::FTRUNC:     return RISCVFPRndMode::RTZ;
  case ISD::FFLOOR:     return RISCVFPRndMode::RDN;
  case ISD::FCEIL:      return RISCVFPRndMode::RUP;
  case ISD::FROUND:     return RISCVFPRndMode::RMM;
  }

  return RISCVFPRndMode::Invalid;
}

// Fold
//   (fp_to_int (froundeven X)) -> fcvt X, rne
//   (fp_to_int (ftrunc X))     -> fcvt X, rtz
//   (fp_to_int (ffloor X))     -> fcvt X, rdn
//   (fp_to_int (fceil X))      -> fcvt X, rup
//   (fp_to_int (fround X))     -> fcvt X, rmm
static SDValue performFP_TO_INTCombine(SDNode *N,
                                       TargetLowering::DAGCombinerInfo &DCI,
                                       const RISCVSubtarget &Subtarget) {
  SelectionDAG &DAG = DCI.DAG;
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  MVT XLenVT = Subtarget.getXLenVT();

  // Only handle XLen or i32 types. Other types narrower than XLen will
  // eventually be legalized to XLenVT.
  EVT VT = N->getValueType(0);
  if (VT != MVT::i32 && VT != XLenVT)
    return SDValue();

  SDValue Src = N->getOperand(0);

  // Ensure the FP type is also legal.
  if (!TLI.isTypeLegal(Src.getValueType()))
    return SDValue();

  // Don't do this for f16 with Zfhmin and not Zfh.
  if (Src.getValueType() == MVT::f16 && !Subtarget.hasStdExtZfh())
    return SDValue();

  RISCVFPRndMode::RoundingMode FRM = matchRoundingOp(Src);
  if (FRM == RISCVFPRndMode::Invalid)
    return SDValue();

  bool IsSigned = N->getOpcode() == ISD::FP_TO_SINT;

  unsigned Opc;
  if (VT == XLenVT)
    Opc = IsSigned ? RISCVISD::FCVT_X : RISCVISD::FCVT_XU;
  else
    Opc = IsSigned ? RISCVISD::FCVT_W_RV64 : RISCVISD::FCVT_WU_RV64;

  SDLoc DL(N);
  SDValue FpToInt = DAG.getNode(Opc, DL, XLenVT, Src.getOperand(0),
                                DAG.getTargetConstant(FRM, DL, XLenVT));
  return DAG.getNode(ISD::TRUNCATE, DL, VT, FpToInt);
}

// Fold
//   (fp_to_int_sat (froundeven X)) -> (select X == nan, 0, (fcvt X, rne))
//   (fp_to_int_sat (ftrunc X))     -> (select X == nan, 0, (fcvt X, rtz))
//   (fp_to_int_sat (ffloor X))     -> (select X == nan, 0, (fcvt X, rdn))
//   (fp_to_int_sat (fceil X))      -> (select X == nan, 0, (fcvt X, rup))
//   (fp_to_int_sat (fround X))     -> (select X == nan, 0, (fcvt X, rmm))
static SDValue performFP_TO_INT_SATCombine(SDNode *N,
                                       TargetLowering::DAGCombinerInfo &DCI,
                                       const RISCVSubtarget &Subtarget) {
  SelectionDAG &DAG = DCI.DAG;
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  MVT XLenVT = Subtarget.getXLenVT();

  // Only handle XLen types. Other types narrower than XLen will eventually be
  // legalized to XLenVT.
  EVT DstVT = N->getValueType(0);
  if (DstVT != XLenVT)
    return SDValue();

  SDValue Src = N->getOperand(0);

  // Ensure the FP type is also legal.
  if (!TLI.isTypeLegal(Src.getValueType()))
    return SDValue();

  // Don't do this for f16 with Zfhmin and not Zfh.
  if (Src.getValueType() == MVT::f16 && !Subtarget.hasStdExtZfh())
    return SDValue();

  EVT SatVT = cast<VTSDNode>(N->getOperand(1))->getVT();

  RISCVFPRndMode::RoundingMode FRM = matchRoundingOp(Src);
  if (FRM == RISCVFPRndMode::Invalid)
    return SDValue();

  bool IsSigned = N->getOpcode() == ISD::FP_TO_SINT_SAT;

  unsigned Opc;
  if (SatVT == DstVT)
    Opc = IsSigned ? RISCVISD::FCVT_X : RISCVISD::FCVT_XU;
  else if (DstVT == MVT::i64 && SatVT == MVT::i32)
    Opc = IsSigned ? RISCVISD::FCVT_W_RV64 : RISCVISD::FCVT_WU_RV64;
  else
    return SDValue();
  // FIXME: Support other SatVTs by clamping before or after the conversion.

  Src = Src.getOperand(0);

  SDLoc DL(N);
  SDValue FpToInt = DAG.getNode(Opc, DL, XLenVT, Src,
                                DAG.getTargetConstant(FRM, DL, XLenVT));

  // RISCV FP-to-int conversions saturate to the destination register size, but
  // don't produce 0 for nan.
  SDValue ZeroInt = DAG.getConstant(0, DL, DstVT);
  return DAG.getSelectCC(DL, Src, Src, ZeroInt, FpToInt, ISD::CondCode::SETUO);
}

// Combine (bitreverse (bswap X)) to the BREV8 GREVI encoding if the type is
// smaller than XLenVT.
static SDValue performBITREVERSECombine(SDNode *N, SelectionDAG &DAG,
                                        const RISCVSubtarget &Subtarget) {
  assert(Subtarget.hasStdExtZbkb() && "Unexpected extension");

  SDValue Src = N->getOperand(0);
  if (Src.getOpcode() != ISD::BSWAP)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (!VT.isScalarInteger() || VT.getSizeInBits() >= Subtarget.getXLen() ||
      !isPowerOf2_32(VT.getSizeInBits()))
    return SDValue();

  SDLoc DL(N);
  return DAG.getNode(RISCVISD::GREV, DL, VT, Src.getOperand(0),
                     DAG.getConstant(7, DL, VT));
}

SDValue RISCVTargetLowering::PerformDAGCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  // Helper to call SimplifyDemandedBits on an operand of N where only some low
  // bits are demanded. N will be added to the Worklist if it was not deleted.
  // Caller should return SDValue(N, 0) if this returns true.
  auto SimplifyDemandedLowBitsHelper = [&](unsigned OpNo, unsigned LowBits) {
    SDValue Op = N->getOperand(OpNo);
    APInt Mask = APInt::getLowBitsSet(Op.getValueSizeInBits(), LowBits);
    if (!SimplifyDemandedBits(Op, Mask, DCI))
      return false;

    if (N->getOpcode() != ISD::DELETED_NODE)
      DCI.AddToWorklist(N);
    return true;
  };

  switch (N->getOpcode()) {
  default:
    break;
  case RISCVISD::SplitF64: {
    SDValue Op0 = N->getOperand(0);
    // If the input to SplitF64 is just BuildPairF64 then the operation is
    // redundant. Instead, use BuildPairF64's operands directly.
    if (Op0->getOpcode() == RISCVISD::BuildPairF64)
      return DCI.CombineTo(N, Op0.getOperand(0), Op0.getOperand(1));

    if (Op0->isUndef()) {
      SDValue Lo = DAG.getUNDEF(MVT::i32);
      SDValue Hi = DAG.getUNDEF(MVT::i32);
      return DCI.CombineTo(N, Lo, Hi);
    }

    SDLoc DL(N);

    // It's cheaper to materialise two 32-bit integers than to load a double
    // from the constant pool and transfer it to integer registers through the
    // stack.
    if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op0)) {
      APInt V = C->getValueAPF().bitcastToAPInt();
      SDValue Lo = DAG.getConstant(V.trunc(32), DL, MVT::i32);
      SDValue Hi = DAG.getConstant(V.lshr(32).trunc(32), DL, MVT::i32);
      return DCI.CombineTo(N, Lo, Hi);
    }

    // This is a target-specific version of a DAGCombine performed in
    // DAGCombiner::visitBITCAST. It performs the equivalent of:
    // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
    // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
    if (!(Op0.getOpcode() == ISD::FNEG || Op0.getOpcode() == ISD::FABS) ||
        !Op0.getNode()->hasOneUse())
      break;
    SDValue NewSplitF64 =
        DAG.getNode(RISCVISD::SplitF64, DL, DAG.getVTList(MVT::i32, MVT::i32),
                    Op0.getOperand(0));
    SDValue Lo = NewSplitF64.getValue(0);
    SDValue Hi = NewSplitF64.getValue(1);
    APInt SignBit = APInt::getSignMask(32);
    if (Op0.getOpcode() == ISD::FNEG) {
      SDValue NewHi = DAG.getNode(ISD::XOR, DL, MVT::i32, Hi,
                                  DAG.getConstant(SignBit, DL, MVT::i32));
      return DCI.CombineTo(N, Lo, NewHi);
    }
    assert(Op0.getOpcode() == ISD::FABS);
    SDValue NewHi = DAG.getNode(ISD::AND, DL, MVT::i32, Hi,
                                DAG.getConstant(~SignBit, DL, MVT::i32));
    return DCI.CombineTo(N, Lo, NewHi);
  }
  case RISCVISD::SLLW:
  case RISCVISD::SRAW:
  case RISCVISD::SRLW: {
    // Only the lower 32 bits of LHS and lower 5 bits of RHS are read.
    if (SimplifyDemandedLowBitsHelper(0, 32) ||
        SimplifyDemandedLowBitsHelper(1, 5))
      return SDValue(N, 0);

    break;
  }
  case ISD::ROTR:
  case ISD::ROTL:
  case RISCVISD::RORW:
  case RISCVISD::ROLW: {
    if (N->getOpcode() == RISCVISD::RORW || N->getOpcode() == RISCVISD::ROLW) {
      // Only the lower 32 bits of LHS and lower 5 bits of RHS are read.
      if (SimplifyDemandedLowBitsHelper(0, 32) ||
          SimplifyDemandedLowBitsHelper(1, 5))
        return SDValue(N, 0);
    }

    return combineROTR_ROTL_RORW_ROLW(N, DAG, Subtarget);
  }
  case RISCVISD::CLZW:
  case RISCVISD::CTZW: {
    // Only the lower 32 bits of the first operand are read
    if (SimplifyDemandedLowBitsHelper(0, 32))
      return SDValue(N, 0);
    break;
  }
  case RISCVISD::GREV:
  case RISCVISD::GORC: {
    // Only the lower log2(Bitwidth) bits of the the shift amount are read.
    unsigned BitWidth = N->getOperand(1).getValueSizeInBits();
    assert(isPowerOf2_32(BitWidth) && "Unexpected bit width");
    if (SimplifyDemandedLowBitsHelper(1, Log2_32(BitWidth)))
      return SDValue(N, 0);

    return combineGREVI_GORCI(N, DAG);
  }
  case RISCVISD::GREVW:
  case RISCVISD::GORCW: {
    // Only the lower 32 bits of LHS and lower 5 bits of RHS are read.
    if (SimplifyDemandedLowBitsHelper(0, 32) ||
        SimplifyDemandedLowBitsHelper(1, 5))
      return SDValue(N, 0);

    break;
  }
  case RISCVISD::SHFL:
  case RISCVISD::UNSHFL: {
    // Only the lower log2(Bitwidth)-1 bits of the the shift amount are read.
    unsigned BitWidth = N->getOperand(1).getValueSizeInBits();
    assert(isPowerOf2_32(BitWidth) && "Unexpected bit width");
    if (SimplifyDemandedLowBitsHelper(1, Log2_32(BitWidth) - 1))
      return SDValue(N, 0);

    break;
  }
  case RISCVISD::SHFLW:
  case RISCVISD::UNSHFLW: {
    // Only the lower 32 bits of LHS and lower 4 bits of RHS are read.
    if (SimplifyDemandedLowBitsHelper(0, 32) ||
        SimplifyDemandedLowBitsHelper(1, 4))
      return SDValue(N, 0);

    break;
  }
  case RISCVISD::BCOMPRESSW:
  case RISCVISD::BDECOMPRESSW: {
    // Only the lower 32 bits of LHS and RHS are read.
    if (SimplifyDemandedLowBitsHelper(0, 32) ||
        SimplifyDemandedLowBitsHelper(1, 32))
      return SDValue(N, 0);

    break;
  }
  case RISCVISD::FSR:
  case RISCVISD::FSL:
  case RISCVISD::FSRW:
  case RISCVISD::FSLW: {
    bool IsWInstruction =
        N->getOpcode() == RISCVISD::FSRW || N->getOpcode() == RISCVISD::FSLW;
    unsigned BitWidth =
        IsWInstruction ? 32 : N->getSimpleValueType(0).getSizeInBits();
    assert(isPowerOf2_32(BitWidth) && "Unexpected bit width");
    // Only the lower log2(Bitwidth)+1 bits of the the shift amount are read.
    if (SimplifyDemandedLowBitsHelper(1, Log2_32(BitWidth) + 1))
      return SDValue(N, 0);

    break;
  }
  case RISCVISD::FMV_X_ANYEXTH:
  case RISCVISD::FMV_X_ANYEXTW_RV64: {
    SDLoc DL(N);
    SDValue Op0 = N->getOperand(0);
    MVT VT = N->getSimpleValueType(0);
    // If the input to FMV_X_ANYEXTW_RV64 is just FMV_W_X_RV64 then the
    // conversion is unnecessary and can be replaced with the FMV_W_X_RV64
    // operand. Similar for FMV_X_ANYEXTH and FMV_H_X.
    if ((N->getOpcode() == RISCVISD::FMV_X_ANYEXTW_RV64 &&
         Op0->getOpcode() == RISCVISD::FMV_W_X_RV64) ||
        (N->getOpcode() == RISCVISD::FMV_X_ANYEXTH &&
         Op0->getOpcode() == RISCVISD::FMV_H_X)) {
      assert(Op0.getOperand(0).getValueType() == VT &&
             "Unexpected value type!");
      return Op0.getOperand(0);
    }

    // This is a target-specific version of a DAGCombine performed in
    // DAGCombiner::visitBITCAST. It performs the equivalent of:
    // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
    // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
    if (!(Op0.getOpcode() == ISD::FNEG || Op0.getOpcode() == ISD::FABS) ||
        !Op0.getNode()->hasOneUse())
      break;
    SDValue NewFMV = DAG.getNode(N->getOpcode(), DL, VT, Op0.getOperand(0));
    unsigned FPBits = N->getOpcode() == RISCVISD::FMV_X_ANYEXTW_RV64 ? 32 : 16;
    APInt SignBit = APInt::getSignMask(FPBits).sextOrSelf(VT.getSizeInBits());
    if (Op0.getOpcode() == ISD::FNEG)
      return DAG.getNode(ISD::XOR, DL, VT, NewFMV,
                         DAG.getConstant(SignBit, DL, VT));

    assert(Op0.getOpcode() == ISD::FABS);
    return DAG.getNode(ISD::AND, DL, VT, NewFMV,
                       DAG.getConstant(~SignBit, DL, VT));
  }
  case ISD::ADD:
    return performADDCombine(N, DAG, Subtarget);
  case ISD::SUB:
    return performSUBCombine(N, DAG);
  case ISD::AND:
    return performANDCombine(N, DAG);
  case ISD::OR:
    return performORCombine(N, DAG, Subtarget);
  case ISD::XOR:
    return performXORCombine(N, DAG);
  case ISD::SIGN_EXTEND_INREG:
    return performSIGN_EXTEND_INREGCombine(N, DAG, Subtarget);
  case ISD::ZERO_EXTEND:
    // Fold (zero_extend (fp_to_uint X)) to prevent forming fcvt+zexti32 during
    // type legalization. This is safe because fp_to_uint produces poison if
    // it overflows.
    if (N->getValueType(0) == MVT::i64 && Subtarget.is64Bit()) {
      SDValue Src = N->getOperand(0);
      if (Src.getOpcode() == ISD::FP_TO_UINT &&
          isTypeLegal(Src.getOperand(0).getValueType()))
        return DAG.getNode(ISD::FP_TO_UINT, SDLoc(N), MVT::i64,
                           Src.getOperand(0));
      if (Src.getOpcode() == ISD::STRICT_FP_TO_UINT && Src.hasOneUse() &&
          isTypeLegal(Src.getOperand(1).getValueType())) {
        SDVTList VTs = DAG.getVTList(MVT::i64, MVT::Other);
        SDValue Res = DAG.getNode(ISD::STRICT_FP_TO_UINT, SDLoc(N), VTs,
                                  Src.getOperand(0), Src.getOperand(1));
        DCI.CombineTo(N, Res);
        DAG.ReplaceAllUsesOfValueWith(Src.getValue(1), Res.getValue(1));
        DCI.recursivelyDeleteUnusedNodes(Src.getNode());
        return SDValue(N, 0); // Return N so it doesn't get rechecked.
      }
    }
    return SDValue();
  case RISCVISD::SELECT_CC: {
    // Transform
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    SDValue TrueV = N->getOperand(3);
    SDValue FalseV = N->getOperand(4);

    // If the True and False values are the same, we don't need a select_cc.
    if (TrueV == FalseV)
      return TrueV;

    ISD::CondCode CCVal = cast<CondCodeSDNode>(N->getOperand(2))->get();
    if (!ISD::isIntEqualitySetCC(CCVal))
      break;

    // Fold (select_cc (setlt X, Y), 0, ne, trueV, falseV) ->
    //      (select_cc X, Y, lt, trueV, falseV)
    // Sometimes the setcc is introduced after select_cc has been formed.
    if (LHS.getOpcode() == ISD::SETCC && isNullConstant(RHS) &&
        LHS.getOperand(0).getValueType() == Subtarget.getXLenVT()) {
      // If we're looking for eq 0 instead of ne 0, we need to invert the
      // condition.
      bool Invert = CCVal == ISD::SETEQ;
      CCVal = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
      if (Invert)
        CCVal = ISD::getSetCCInverse(CCVal, LHS.getValueType());

      SDLoc DL(N);
      RHS = LHS.getOperand(1);
      LHS = LHS.getOperand(0);
      translateSetCCForBranch(DL, LHS, RHS, CCVal, DAG);

      SDValue TargetCC = DAG.getCondCode(CCVal);
      return DAG.getNode(RISCVISD::SELECT_CC, DL, N->getValueType(0),
                         {LHS, RHS, TargetCC, TrueV, FalseV});
    }

    // Fold (select_cc (xor X, Y), 0, eq/ne, trueV, falseV) ->
    //      (select_cc X, Y, eq/ne, trueV, falseV)
    if (LHS.getOpcode() == ISD::XOR && isNullConstant(RHS))
      return DAG.getNode(RISCVISD::SELECT_CC, SDLoc(N), N->getValueType(0),
                         {LHS.getOperand(0), LHS.getOperand(1),
                          N->getOperand(2), TrueV, FalseV});
    // (select_cc X, 1, setne, trueV, falseV) ->
    // (select_cc X, 0, seteq, trueV, falseV) if we can prove X is 0/1.
    // This can occur when legalizing some floating point comparisons.
    APInt Mask = APInt::getBitsSetFrom(LHS.getValueSizeInBits(), 1);
    if (isOneConstant(RHS) && DAG.MaskedValueIsZero(LHS, Mask)) {
      SDLoc DL(N);
      CCVal = ISD::getSetCCInverse(CCVal, LHS.getValueType());
      SDValue TargetCC = DAG.getCondCode(CCVal);
      RHS = DAG.getConstant(0, DL, LHS.getValueType());
      return DAG.getNode(RISCVISD::SELECT_CC, DL, N->getValueType(0),
                         {LHS, RHS, TargetCC, TrueV, FalseV});
    }

    break;
  }
  case RISCVISD::BR_CC: {
    SDValue LHS = N->getOperand(1);
    SDValue RHS = N->getOperand(2);
    ISD::CondCode CCVal = cast<CondCodeSDNode>(N->getOperand(3))->get();
    if (!ISD::isIntEqualitySetCC(CCVal))
      break;

    // Fold (br_cc (setlt X, Y), 0, ne, dest) ->
    //      (br_cc X, Y, lt, dest)
    // Sometimes the setcc is introduced after br_cc has been formed.
    if (LHS.getOpcode() == ISD::SETCC && isNullConstant(RHS) &&
        LHS.getOperand(0).getValueType() == Subtarget.getXLenVT()) {
      // If we're looking for eq 0 instead of ne 0, we need to invert the
      // condition.
      bool Invert = CCVal == ISD::SETEQ;
      CCVal = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
      if (Invert)
        CCVal = ISD::getSetCCInverse(CCVal, LHS.getValueType());

      SDLoc DL(N);
      RHS = LHS.getOperand(1);
      LHS = LHS.getOperand(0);
      translateSetCCForBranch(DL, LHS, RHS, CCVal, DAG);

      return DAG.getNode(RISCVISD::BR_CC, DL, N->getValueType(0),
                         N->getOperand(0), LHS, RHS, DAG.getCondCode(CCVal),
                         N->getOperand(4));
    }

    // Fold (br_cc (xor X, Y), 0, eq/ne, dest) ->
    //      (br_cc X, Y, eq/ne, trueV, falseV)
    if (LHS.getOpcode() == ISD::XOR && isNullConstant(RHS))
      return DAG.getNode(RISCVISD::BR_CC, SDLoc(N), N->getValueType(0),
                         N->getOperand(0), LHS.getOperand(0), LHS.getOperand(1),
                         N->getOperand(3), N->getOperand(4));

    // (br_cc X, 1, setne, br_cc) ->
    // (br_cc X, 0, seteq, br_cc) if we can prove X is 0/1.
    // This can occur when legalizing some floating point comparisons.
    APInt Mask = APInt::getBitsSetFrom(LHS.getValueSizeInBits(), 1);
    if (isOneConstant(RHS) && DAG.MaskedValueIsZero(LHS, Mask)) {
      SDLoc DL(N);
      CCVal = ISD::getSetCCInverse(CCVal, LHS.getValueType());
      SDValue TargetCC = DAG.getCondCode(CCVal);
      RHS = DAG.getConstant(0, DL, LHS.getValueType());
      return DAG.getNode(RISCVISD::BR_CC, DL, N->getValueType(0),
                         N->getOperand(0), LHS, RHS, TargetCC,
                         N->getOperand(4));
    }
    break;
  }
  case ISD::BITREVERSE:
    return performBITREVERSECombine(N, DAG, Subtarget);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return performFP_TO_INTCombine(N, DCI, Subtarget);
  case ISD::FP_TO_SINT_SAT:
  case ISD::FP_TO_UINT_SAT:
    return performFP_TO_INT_SATCombine(N, DCI, Subtarget);
  case ISD::FCOPYSIGN: {
    EVT VT = N->getValueType(0);
    if (!VT.isVector())
      break;
    // There is a form of VFSGNJ which injects the negated sign of its second
    // operand. Try and bubble any FNEG up after the extend/round to produce
    // this optimized pattern. Avoid modifying cases where FP_ROUND and
    // TRUNC=1.
    SDValue In2 = N->getOperand(1);
    // Avoid cases where the extend/round has multiple uses, as duplicating
    // those is typically more expensive than removing a fneg.
    if (!In2.hasOneUse())
      break;
    if (In2.getOpcode() != ISD::FP_EXTEND &&
        (In2.getOpcode() != ISD::FP_ROUND || In2.getConstantOperandVal(1) != 0))
      break;
    In2 = In2.getOperand(0);
    if (In2.getOpcode() != ISD::FNEG)
      break;
    SDLoc DL(N);
    SDValue NewFPExtRound = DAG.getFPExtendOrRound(In2.getOperand(0), DL, VT);
    return DAG.getNode(ISD::FCOPYSIGN, DL, VT, N->getOperand(0),
                       DAG.getNode(ISD::FNEG, DL, VT, NewFPExtRound));
  }
  case ISD::MGATHER:
  case ISD::MSCATTER:
  case ISD::VP_GATHER:
  case ISD::VP_SCATTER: {
    if (!DCI.isBeforeLegalize())
      break;
    SDValue Index, ScaleOp;
    bool IsIndexScaled = false;
    bool IsIndexSigned = false;
    if (const auto *VPGSN = dyn_cast<VPGatherScatterSDNode>(N)) {
      Index = VPGSN->getIndex();
      ScaleOp = VPGSN->getScale();
      IsIndexScaled = VPGSN->isIndexScaled();
      IsIndexSigned = VPGSN->isIndexSigned();
    } else {
      const auto *MGSN = cast<MaskedGatherScatterSDNode>(N);
      Index = MGSN->getIndex();
      ScaleOp = MGSN->getScale();
      IsIndexScaled = MGSN->isIndexScaled();
      IsIndexSigned = MGSN->isIndexSigned();
    }
    EVT IndexVT = Index.getValueType();
    MVT XLenVT = Subtarget.getXLenVT();
    // RISCV indexed loads only support the "unsigned unscaled" addressing
    // mode, so anything else must be manually legalized.
    bool NeedsIdxLegalization =
        IsIndexScaled ||
        (IsIndexSigned && IndexVT.getVectorElementType().bitsLT(XLenVT));
    if (!NeedsIdxLegalization)
      break;

    SDLoc DL(N);

    // Any index legalization should first promote to XLenVT, so we don't lose
    // bits when scaling. This may create an illegal index type so we let
    // LLVM's legalization take care of the splitting.
    // FIXME: LLVM can't split VP_GATHER or VP_SCATTER yet.
    if (IndexVT.getVectorElementType().bitsLT(XLenVT)) {
      IndexVT = IndexVT.changeVectorElementType(XLenVT);
      Index = DAG.getNode(IsIndexSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND,
                          DL, IndexVT, Index);
    }

    unsigned Scale = cast<ConstantSDNode>(ScaleOp)->getZExtValue();
    if (IsIndexScaled && Scale != 1) {
      // Manually scale the indices by the element size.
      // TODO: Sanitize the scale operand here?
      // TODO: For VP nodes, should we use VP_SHL here?
      assert(isPowerOf2_32(Scale) && "Expecting power-of-two types");
      SDValue SplatScale = DAG.getConstant(Log2_32(Scale), DL, IndexVT);
      Index = DAG.getNode(ISD::SHL, DL, IndexVT, Index, SplatScale);
    }

    ISD::MemIndexType NewIndexTy = ISD::UNSIGNED_UNSCALED;
    if (const auto *VPGN = dyn_cast<VPGatherSDNode>(N))
      return DAG.getGatherVP(N->getVTList(), VPGN->getMemoryVT(), DL,
                             {VPGN->getChain(), VPGN->getBasePtr(), Index,
                              VPGN->getScale(), VPGN->getMask(),
                              VPGN->getVectorLength()},
                             VPGN->getMemOperand(), NewIndexTy);
    if (const auto *VPSN = dyn_cast<VPScatterSDNode>(N))
      return DAG.getScatterVP(N->getVTList(), VPSN->getMemoryVT(), DL,
                              {VPSN->getChain(), VPSN->getValue(),
                               VPSN->getBasePtr(), Index, VPSN->getScale(),
                               VPSN->getMask(), VPSN->getVectorLength()},
                              VPSN->getMemOperand(), NewIndexTy);
    if (const auto *MGN = dyn_cast<MaskedGatherSDNode>(N))
      return DAG.getMaskedGather(
          N->getVTList(), MGN->getMemoryVT(), DL,
          {MGN->getChain(), MGN->getPassThru(), MGN->getMask(),
           MGN->getBasePtr(), Index, MGN->getScale()},
          MGN->getMemOperand(), NewIndexTy, MGN->getExtensionType());
    const auto *MSN = cast<MaskedScatterSDNode>(N);
    return DAG.getMaskedScatter(
        N->getVTList(), MSN->getMemoryVT(), DL,
        {MSN->getChain(), MSN->getValue(), MSN->getMask(), MSN->getBasePtr(),
         Index, MSN->getScale()},
        MSN->getMemOperand(), NewIndexTy, MSN->isTruncatingStore());
  }
  case RISCVISD::SRA_VL:
  case RISCVISD::SRL_VL:
  case RISCVISD::SHL_VL: {
    SDValue ShAmt = N->getOperand(1);
    if (ShAmt.getOpcode() == RISCVISD::SPLAT_VECTOR_SPLIT_I64_VL) {
      // We don't need the upper 32 bits of a 64-bit element for a shift amount.
      SDLoc DL(N);
      SDValue VL = N->getOperand(3);
      EVT VT = N->getValueType(0);
      ShAmt = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VT, DAG.getUNDEF(VT),
                          ShAmt.getOperand(1), VL);
      return DAG.getNode(N->getOpcode(), DL, VT, N->getOperand(0), ShAmt,
                         N->getOperand(2), N->getOperand(3));
    }
    break;
  }
  case ISD::SRA:
  case ISD::SRL:
  case ISD::SHL: {
    SDValue ShAmt = N->getOperand(1);
    if (ShAmt.getOpcode() == RISCVISD::SPLAT_VECTOR_SPLIT_I64_VL) {
      // We don't need the upper 32 bits of a 64-bit element for a shift amount.
      SDLoc DL(N);
      EVT VT = N->getValueType(0);
      ShAmt = DAG.getNode(RISCVISD::VMV_V_X_VL, DL, VT, DAG.getUNDEF(VT),
                          ShAmt.getOperand(1),
                          DAG.getRegister(RISCV::X0, Subtarget.getXLenVT()));
      return DAG.getNode(N->getOpcode(), DL, VT, N->getOperand(0), ShAmt);
    }
    break;
  }
  case RISCVISD::ADD_VL:
    if (SDValue V = combineADDSUB_VLToVWADDSUB_VL(N, DAG, /*Commute*/ false))
      return V;
    return combineADDSUB_VLToVWADDSUB_VL(N, DAG, /*Commute*/ true);
  case RISCVISD::SUB_VL:
    return combineADDSUB_VLToVWADDSUB_VL(N, DAG);
  case RISCVISD::VWADD_W_VL:
  case RISCVISD::VWADDU_W_VL:
  case RISCVISD::VWSUB_W_VL:
  case RISCVISD::VWSUBU_W_VL:
    return combineVWADD_W_VL_VWSUB_W_VL(N, DAG);
  case RISCVISD::MUL_VL:
    if (SDValue V = combineMUL_VLToVWMUL_VL(N, DAG, /*Commute*/ false))
      return V;
    // Mul is commutative.
    return combineMUL_VLToVWMUL_VL(N, DAG, /*Commute*/ true);
  case ISD::STORE: {
    auto *Store = cast<StoreSDNode>(N);
    SDValue Val = Store->getValue();
    // Combine store of vmv.x.s to vse with VL of 1.
    // FIXME: Support FP.
    if (Val.getOpcode() == RISCVISD::VMV_X_S) {
      SDValue Src = Val.getOperand(0);
      EVT VecVT = Src.getValueType();
      EVT MemVT = Store->getMemoryVT();
      // The memory VT and the element type must match.
      if (VecVT.getVectorElementType() == MemVT) {
        SDLoc DL(N);
        MVT MaskVT = MVT::getVectorVT(MVT::i1, VecVT.getVectorElementCount());
        return DAG.getStoreVP(
            Store->getChain(), DL, Src, Store->getBasePtr(), Store->getOffset(),
            DAG.getConstant(1, DL, MaskVT),
            DAG.getConstant(1, DL, Subtarget.getXLenVT()), MemVT,
            Store->getMemOperand(), Store->getAddressingMode(),
            Store->isTruncatingStore(), /*IsCompress*/ false);
      }
    }

    break;
  }
  case ISD::SPLAT_VECTOR: {
    EVT VT = N->getValueType(0);
    // Only perform this combine on legal MVT types.
    if (!isTypeLegal(VT))
      break;
    if (auto Gather = matchSplatAsGather(N->getOperand(0), VT.getSimpleVT(), N,
                                         DAG, Subtarget))
      return Gather;
    break;
  }
  case RISCVISD::VMV_V_X_VL: {
    // Tail agnostic VMV.V.X only demands the vector element bitwidth from the
    // scalar input.
    unsigned ScalarSize = N->getOperand(1).getValueSizeInBits();
    unsigned EltWidth = N->getValueType(0).getScalarSizeInBits();
    if (ScalarSize > EltWidth && N->getOperand(0).isUndef())
      if (SimplifyDemandedLowBitsHelper(1, EltWidth))
        return SDValue(N, 0);

    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = N->getConstantOperandVal(0);
    switch (IntNo) {
      // By default we do not combine any intrinsic.
    default:
      return SDValue();
    case Intrinsic::riscv_vcpop:
    case Intrinsic::riscv_vcpop_mask:
    case Intrinsic::riscv_vfirst:
    case Intrinsic::riscv_vfirst_mask: {
      SDValue VL = N->getOperand(2);
      if (IntNo == Intrinsic::riscv_vcpop_mask ||
          IntNo == Intrinsic::riscv_vfirst_mask)
        VL = N->getOperand(3);
      if (!isNullConstant(VL))
        return SDValue();
      // If VL is 0, vcpop -> li 0, vfirst -> li -1.
      SDLoc DL(N);
      EVT VT = N->getValueType(0);
      if (IntNo == Intrinsic::riscv_vfirst ||
          IntNo == Intrinsic::riscv_vfirst_mask)
        return DAG.getConstant(-1, DL, VT);
      return DAG.getConstant(0, DL, VT);
    }
    }
  }
  }

  return SDValue();
}

bool RISCVTargetLowering::isDesirableToCommuteWithShift(
    const SDNode *N, CombineLevel Level) const {
  // The following folds are only desirable if `(OP _, c1 << c2)` can be
  // materialised in fewer instructions than `(OP _, c1)`:
  //
  //   (shl (add x, c1), c2) -> (add (shl x, c2), c1 << c2)
  //   (shl (or x, c1), c2) -> (or (shl x, c2), c1 << c2)
  SDValue N0 = N->getOperand(0);
  EVT Ty = N0.getValueType();
  if (Ty.isScalarInteger() &&
      (N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::OR)) {
    auto *C1 = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    auto *C2 = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (C1 && C2) {
      const APInt &C1Int = C1->getAPIntValue();
      APInt ShiftedC1Int = C1Int << C2->getAPIntValue();

      // We can materialise `c1 << c2` into an add immediate, so it's "free",
      // and the combine should happen, to potentially allow further combines
      // later.
      if (ShiftedC1Int.getMinSignedBits() <= 64 &&
          isLegalAddImmediate(ShiftedC1Int.getSExtValue()))
        return true;

      // We can materialise `c1` in an add immediate, so it's "free", and the
      // combine should be prevented.
      if (C1Int.getMinSignedBits() <= 64 &&
          isLegalAddImmediate(C1Int.getSExtValue()))
        return false;

      // Neither constant will fit into an immediate, so find materialisation
      // costs.
      int C1Cost = RISCVMatInt::getIntMatCost(C1Int, Ty.getSizeInBits(),
                                              Subtarget.getFeatureBits(),
                                              /*CompressionCost*/true);
      int ShiftedC1Cost = RISCVMatInt::getIntMatCost(
          ShiftedC1Int, Ty.getSizeInBits(), Subtarget.getFeatureBits(),
          /*CompressionCost*/true);

      // Materialising `c1` is cheaper than materialising `c1 << c2`, so the
      // combine should be prevented.
      if (C1Cost < ShiftedC1Cost)
        return false;
    }
  }
  return true;
}

bool RISCVTargetLowering::targetShrinkDemandedConstant(
    SDValue Op, const APInt &DemandedBits, const APInt &DemandedElts,
    TargetLoweringOpt &TLO) const {
  // Delay this optimization as late as possible.
  if (!TLO.LegalOps)
    return false;

  EVT VT = Op.getValueType();
  if (VT.isVector())
    return false;

  // Only handle AND for now.
  if (Op.getOpcode() != ISD::AND)
    return false;

  ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
  if (!C)
    return false;

  const APInt &Mask = C->getAPIntValue();

  // Clear all non-demanded bits initially.
  APInt ShrunkMask = Mask & DemandedBits;

  // Try to make a smaller immediate by setting undemanded bits.

  APInt ExpandedMask = Mask | ~DemandedBits;

  auto IsLegalMask = [ShrunkMask, ExpandedMask](const APInt &Mask) -> bool {
    return ShrunkMask.isSubsetOf(Mask) && Mask.isSubsetOf(ExpandedMask);
  };
  auto UseMask = [Mask, Op, VT, &TLO](const APInt &NewMask) -> bool {
    if (NewMask == Mask)
      return true;
    SDLoc DL(Op);
    SDValue NewC = TLO.DAG.getConstant(NewMask, DL, VT);
    SDValue NewOp = TLO.DAG.getNode(ISD::AND, DL, VT, Op.getOperand(0), NewC);
    return TLO.CombineTo(Op, NewOp);
  };

  // If the shrunk mask fits in sign extended 12 bits, let the target
  // independent code apply it.
  if (ShrunkMask.isSignedIntN(12))
    return false;

  // Preserve (and X, 0xffff) when zext.h is supported.
  if (Subtarget.hasStdExtZbb() || Subtarget.hasStdExtZbp()) {
    APInt NewMask = APInt(Mask.getBitWidth(), 0xffff);
    if (IsLegalMask(NewMask))
      return UseMask(NewMask);
  }

  // Try to preserve (and X, 0xffffffff), the (zext_inreg X, i32) pattern.
  if (VT == MVT::i64) {
    APInt NewMask = APInt(64, 0xffffffff);
    if (IsLegalMask(NewMask))
      return UseMask(NewMask);
  }

  // For the remaining optimizations, we need to be able to make a negative
  // number through a combination of mask and undemanded bits.
  if (!ExpandedMask.isNegative())
    return false;

  // What is the fewest number of bits we need to represent the negative number.
  unsigned MinSignedBits = ExpandedMask.getMinSignedBits();

  // Try to make a 12 bit negative immediate. If that fails try to make a 32
  // bit negative immediate unless the shrunk immediate already fits in 32 bits.
  APInt NewMask = ShrunkMask;
  if (MinSignedBits <= 12)
    NewMask.setBitsFrom(11);
  else if (MinSignedBits <= 32 && !ShrunkMask.isSignedIntN(32))
    NewMask.setBitsFrom(31);
  else
    return false;

  // Check that our new mask is a subset of the demanded mask.
  assert(IsLegalMask(NewMask));
  return UseMask(NewMask);
}

static void computeGREV(APInt &Src, unsigned ShAmt) {
  ShAmt &= Src.getBitWidth() - 1;
  uint64_t x = Src.getZExtValue();
  if (ShAmt & 1)
    x = ((x & 0x5555555555555555LL) << 1) | ((x & 0xAAAAAAAAAAAAAAAALL) >> 1);
  if (ShAmt & 2)
    x = ((x & 0x3333333333333333LL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCLL) >> 2);
  if (ShAmt & 4)
    x = ((x & 0x0F0F0F0F0F0F0F0FLL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0LL) >> 4);
  if (ShAmt & 8)
    x = ((x & 0x00FF00FF00FF00FFLL) << 8) | ((x & 0xFF00FF00FF00FF00LL) >> 8);
  if (ShAmt & 16)
    x = ((x & 0x0000FFFF0000FFFFLL) << 16) | ((x & 0xFFFF0000FFFF0000LL) >> 16);
  if (ShAmt & 32)
    x = ((x & 0x00000000FFFFFFFFLL) << 32) | ((x & 0xFFFFFFFF00000000LL) >> 32);
  Src = x;
}

void RISCVTargetLowering::computeKnownBitsForTargetNode(const SDValue Op,
                                                        KnownBits &Known,
                                                        const APInt &DemandedElts,
                                                        const SelectionDAG &DAG,
                                                        unsigned Depth) const {
  unsigned BitWidth = Known.getBitWidth();
  unsigned Opc = Op.getOpcode();
  assert((Opc >= ISD::BUILTIN_OP_END ||
          Opc == ISD::INTRINSIC_WO_CHAIN ||
          Opc == ISD::INTRINSIC_W_CHAIN ||
          Opc == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");

  Known.resetAll();
  switch (Opc) {
  default: break;
  case RISCVISD::SELECT_CC: {
    Known = DAG.computeKnownBits(Op.getOperand(4), Depth + 1);
    // If we don't know any bits, early out.
    if (Known.isUnknown())
      break;
    KnownBits Known2 = DAG.computeKnownBits(Op.getOperand(3), Depth + 1);

    // Only known if known in both the LHS and RHS.
    Known = KnownBits::commonBits(Known, Known2);
    break;
  }
  case RISCVISD::REMUW: {
    KnownBits Known2;
    Known = DAG.computeKnownBits(Op.getOperand(0), DemandedElts, Depth + 1);
    Known2 = DAG.computeKnownBits(Op.getOperand(1), DemandedElts, Depth + 1);
    // We only care about the lower 32 bits.
    Known = KnownBits::urem(Known.trunc(32), Known2.trunc(32));
    // Restore the original width by sign extending.
    Known = Known.sext(BitWidth);
    break;
  }
  case RISCVISD::DIVUW: {
    KnownBits Known2;
    Known = DAG.computeKnownBits(Op.getOperand(0), DemandedElts, Depth + 1);
    Known2 = DAG.computeKnownBits(Op.getOperand(1), DemandedElts, Depth + 1);
    // We only care about the lower 32 bits.
    Known = KnownBits::udiv(Known.trunc(32), Known2.trunc(32));
    // Restore the original width by sign extending.
    Known = Known.sext(BitWidth);
    break;
  }
  case RISCVISD::CTZW: {
    KnownBits Known2 = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    unsigned PossibleTZ = Known2.trunc(32).countMaxTrailingZeros();
    unsigned LowBits = Log2_32(PossibleTZ) + 1;
    Known.Zero.setBitsFrom(LowBits);
    break;
  }
  case RISCVISD::CLZW: {
    KnownBits Known2 = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    unsigned PossibleLZ = Known2.trunc(32).countMaxLeadingZeros();
    unsigned LowBits = Log2_32(PossibleLZ) + 1;
    Known.Zero.setBitsFrom(LowBits);
    break;
  }
  case RISCVISD::GREV: {
    if (auto *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      Known = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
      unsigned ShAmt = C->getZExtValue();
      computeGREV(Known.Zero, ShAmt);
      computeGREV(Known.One, ShAmt);
    }
    break;
  }
  case RISCVISD::READ_VLENB: {
    // If we know the minimum VLen from Zvl extensions, we can use that to
    // determine the trailing zeros of VLENB.
    // FIXME: Limit to 128 bit vectors until we have more testing.
    unsigned MinVLenB = std::min(128U, Subtarget.getMinVLen()) / 8;
    if (MinVLenB > 0)
      Known.Zero.setLowBits(Log2_32(MinVLenB));
    // We assume VLENB is no more than 65536 / 8 bytes.
    Known.Zero.setBitsFrom(14);
    break;
  }
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo =
        Op.getConstantOperandVal(Opc == ISD::INTRINSIC_WO_CHAIN ? 0 : 1);
    switch (IntNo) {
    default:
      // We can't do anything for most intrinsics.
      break;
    case Intrinsic::riscv_vsetvli:
    case Intrinsic::riscv_vsetvlimax:
    case Intrinsic::riscv_vsetvli_opt:
    case Intrinsic::riscv_vsetvlimax_opt:
      // Assume that VL output is positive and would fit in an int32_t.
      // TODO: VLEN might be capped at 16 bits in a future V spec update.
      if (BitWidth >= 32)
        Known.Zero.setBitsFrom(31);
      break;
    }
    break;
  }
  }
}

unsigned RISCVTargetLowering::ComputeNumSignBitsForTargetNode(
    SDValue Op, const APInt &DemandedElts, const SelectionDAG &DAG,
    unsigned Depth) const {
  switch (Op.getOpcode()) {
  default:
    break;
  case RISCVISD::SELECT_CC: {
    unsigned Tmp =
        DAG.ComputeNumSignBits(Op.getOperand(3), DemandedElts, Depth + 1);
    if (Tmp == 1) return 1;  // Early out.
    unsigned Tmp2 =
        DAG.ComputeNumSignBits(Op.getOperand(4), DemandedElts, Depth + 1);
    return std::min(Tmp, Tmp2);
  }
  case RISCVISD::SLLW:
  case RISCVISD::SRAW:
  case RISCVISD::SRLW:
  case RISCVISD::DIVW:
  case RISCVISD::DIVUW:
  case RISCVISD::REMUW:
  case RISCVISD::ROLW:
  case RISCVISD::RORW:
  case RISCVISD::GREVW:
  case RISCVISD::GORCW:
  case RISCVISD::FSLW:
  case RISCVISD::FSRW:
  case RISCVISD::SHFLW:
  case RISCVISD::UNSHFLW:
  case RISCVISD::BCOMPRESSW:
  case RISCVISD::BDECOMPRESSW:
  case RISCVISD::BFPW:
  case RISCVISD::FCVT_W_RV64:
  case RISCVISD::FCVT_WU_RV64:
  case RISCVISD::STRICT_FCVT_W_RV64:
  case RISCVISD::STRICT_FCVT_WU_RV64:
    // TODO: As the result is sign-extended, this is conservatively correct. A
    // more precise answer could be calculated for SRAW depending on known
    // bits in the shift amount.
    return 33;
  case RISCVISD::SHFL:
  case RISCVISD::UNSHFL: {
    // There is no SHFLIW, but a i64 SHFLI with bit 4 of the control word
    // cleared doesn't affect bit 31. The upper 32 bits will be shuffled, but
    // will stay within the upper 32 bits. If there were more than 32 sign bits
    // before there will be at least 33 sign bits after.
    if (Op.getValueType() == MVT::i64 &&
        isa<ConstantSDNode>(Op.getOperand(1)) &&
        (Op.getConstantOperandVal(1) & 0x10) == 0) {
      unsigned Tmp = DAG.ComputeNumSignBits(Op.getOperand(0), Depth + 1);
      if (Tmp > 32)
        return 33;
    }
    break;
  }
  case RISCVISD::VMV_X_S: {
    // The number of sign bits of the scalar result is computed by obtaining the
    // element type of the input vector operand, subtracting its width from the
    // XLEN, and then adding one (sign bit within the element type). If the
    // element type is wider than XLen, the least-significant XLEN bits are
    // taken.
    unsigned XLen = Subtarget.getXLen();
    unsigned EltBits = Op.getOperand(0).getScalarValueSizeInBits();
    if (EltBits <= XLen)
      return XLen - EltBits + 1;
    break;
  }
  }

  return 1;
}

static MachineBasicBlock *emitReadCycleWidePseudo(MachineInstr &MI,
                                                  MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::ReadCycleWide && "Unexpected instruction");

  // To read the 64-bit cycle CSR on a 32-bit target, we read the two halves.
  // Should the count have wrapped while it was being read, we need to try
  // again.
  // ...
  // read:
  // rdcycleh x3 # load high word of cycle
  // rdcycle  x2 # load low word of cycle
  // rdcycleh x4 # load high word of cycle
  // bne x3, x4, read # check if high word reads match, otherwise try again
  // ...

  MachineFunction &MF = *BB->getParent();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *LoopMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MF.insert(It, LoopMBB);

  MachineBasicBlock *DoneMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MF.insert(It, DoneMBB);

  // Transfer the remainder of BB and its successor edges to DoneMBB.
  DoneMBB->splice(DoneMBB->begin(), BB,
                  std::next(MachineBasicBlock::iterator(MI)), BB->end());
  DoneMBB->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(LoopMBB);

  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  Register ReadAgainReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
  Register LoReg = MI.getOperand(0).getReg();
  Register HiReg = MI.getOperand(1).getReg();
  DebugLoc DL = MI.getDebugLoc();

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), HiReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLEH")->Encoding)
      .addReg(RISCV::X0);
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), LoReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLE")->Encoding)
      .addReg(RISCV::X0);
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), ReadAgainReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLEH")->Encoding)
      .addReg(RISCV::X0);

  BuildMI(LoopMBB, DL, TII->get(RISCV::BNE))
      .addReg(HiReg)
      .addReg(ReadAgainReg)
      .addMBB(LoopMBB);

  LoopMBB->addSuccessor(LoopMBB);
  LoopMBB->addSuccessor(DoneMBB);

  MI.eraseFromParent();

  return DoneMBB;
}

static MachineBasicBlock *emitSplitF64Pseudo(MachineInstr &MI,
                                             MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::SplitF64Pseudo && "Unexpected instruction");

  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  Register LoReg = MI.getOperand(0).getReg();
  Register HiReg = MI.getOperand(1).getReg();
  Register SrcReg = MI.getOperand(2).getReg();
  const TargetRegisterClass *SrcRC = &RISCV::FPR64RegClass;
  int FI = MF.getInfo<RISCVMachineFunctionInfo>()->getMoveF64FrameIndex(MF);

  TII.storeRegToStackSlot(*BB, MI, SrcReg, MI.getOperand(2).isKill(), FI, SrcRC,
                          RI);
  MachinePointerInfo MPI = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMOLo =
      MF.getMachineMemOperand(MPI, MachineMemOperand::MOLoad, 4, Align(8));
  MachineMemOperand *MMOHi = MF.getMachineMemOperand(
      MPI.getWithOffset(4), MachineMemOperand::MOLoad, 4, Align(8));
  BuildMI(*BB, MI, DL, TII.get(RISCV::LW), LoReg)
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMOLo);
  BuildMI(*BB, MI, DL, TII.get(RISCV::LW), HiReg)
      .addFrameIndex(FI)
      .addImm(4)
      .addMemOperand(MMOHi);
  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *emitBuildPairF64Pseudo(MachineInstr &MI,
                                                 MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::BuildPairF64Pseudo &&
         "Unexpected instruction");

  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  Register DstReg = MI.getOperand(0).getReg();
  Register LoReg = MI.getOperand(1).getReg();
  Register HiReg = MI.getOperand(2).getReg();
  const TargetRegisterClass *DstRC = &RISCV::FPR64RegClass;
  int FI = MF.getInfo<RISCVMachineFunctionInfo>()->getMoveF64FrameIndex(MF);

  MachinePointerInfo MPI = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMOLo =
      MF.getMachineMemOperand(MPI, MachineMemOperand::MOStore, 4, Align(8));
  MachineMemOperand *MMOHi = MF.getMachineMemOperand(
      MPI.getWithOffset(4), MachineMemOperand::MOStore, 4, Align(8));
  BuildMI(*BB, MI, DL, TII.get(RISCV::SW))
      .addReg(LoReg, getKillRegState(MI.getOperand(1).isKill()))
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMOLo);
  BuildMI(*BB, MI, DL, TII.get(RISCV::SW))
      .addReg(HiReg, getKillRegState(MI.getOperand(2).isKill()))
      .addFrameIndex(FI)
      .addImm(4)
      .addMemOperand(MMOHi);
  TII.loadRegFromStackSlot(*BB, MI, DstReg, FI, DstRC, RI);
  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static bool isSelectPseudo(MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case RISCV::Select_GPR_Using_CC_GPR:
  case RISCV::Select_FPR16_Using_CC_GPR:
  case RISCV::Select_FPR32_Using_CC_GPR:
  case RISCV::Select_FPR64_Using_CC_GPR:
    return true;
  }
}

static MachineBasicBlock *emitQuietFCMP(MachineInstr &MI, MachineBasicBlock *BB,
                                        unsigned RelOpcode, unsigned EqOpcode,
                                        const RISCVSubtarget &Subtarget) {
  DebugLoc DL = MI.getDebugLoc();
  Register DstReg = MI.getOperand(0).getReg();
  Register Src1Reg = MI.getOperand(1).getReg();
  Register Src2Reg = MI.getOperand(2).getReg();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  Register SavedFFlags = MRI.createVirtualRegister(&RISCV::GPRRegClass);
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();

  // Save the current FFLAGS.
  BuildMI(*BB, MI, DL, TII.get(RISCV::ReadFFLAGS), SavedFFlags);

  auto MIB = BuildMI(*BB, MI, DL, TII.get(RelOpcode), DstReg)
                 .addReg(Src1Reg)
                 .addReg(Src2Reg);
  if (MI.getFlag(MachineInstr::MIFlag::NoFPExcept))
    MIB->setFlag(MachineInstr::MIFlag::NoFPExcept);

  // Restore the FFLAGS.
  BuildMI(*BB, MI, DL, TII.get(RISCV::WriteFFLAGS))
      .addReg(SavedFFlags, RegState::Kill);

  // Issue a dummy FEQ opcode to raise exception for signaling NaNs.
  auto MIB2 = BuildMI(*BB, MI, DL, TII.get(EqOpcode), RISCV::X0)
                  .addReg(Src1Reg, getKillRegState(MI.getOperand(1).isKill()))
                  .addReg(Src2Reg, getKillRegState(MI.getOperand(2).isKill()));
  if (MI.getFlag(MachineInstr::MIFlag::NoFPExcept))
    MIB2->setFlag(MachineInstr::MIFlag::NoFPExcept);

  // Erase the pseudoinstruction.
  MI.eraseFromParent();
  return BB;
}

static MachineBasicBlock *emitSelectPseudo(MachineInstr &MI,
                                           MachineBasicBlock *BB,
                                           const RISCVSubtarget &Subtarget) {
  // To "insert" Select_* instructions, we actually have to insert the triangle
  // control-flow pattern.  The incoming instructions know the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and the condcode to use to select the appropriate branch.
  //
  // We produce the following control flow:
  //     HeadMBB
  //     |  \
  //     |  IfFalseMBB
  //     | /
  //    TailMBB
  //
  // When we find a sequence of selects we attempt to optimize their emission
  // by sharing the control flow. Currently we only handle cases where we have
  // multiple selects with the exact same condition (same LHS, RHS and CC).
  // The selects may be interleaved with other instructions if the other
  // instructions meet some requirements we deem safe:
  // - They are debug instructions. Otherwise,
  // - They do not have side-effects, do not access memory and their inputs do
  //   not depend on the results of the select pseudo-instructions.
  // The TrueV/FalseV operands of the selects cannot depend on the result of
  // previous selects in the sequence.
  // These conditions could be further relaxed. See the X86 target for a
  // related approach and more information.
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  auto CC = static_cast<RISCVCC::CondCode>(MI.getOperand(3).getImm());

  SmallVector<MachineInstr *, 4> SelectDebugValues;
  SmallSet<Register, 4> SelectDests;
  SelectDests.insert(MI.getOperand(0).getReg());

  MachineInstr *LastSelectPseudo = &MI;

  for (auto E = BB->end(), SequenceMBBI = MachineBasicBlock::iterator(MI);
       SequenceMBBI != E; ++SequenceMBBI) {
    if (SequenceMBBI->isDebugInstr())
      continue;
    else if (isSelectPseudo(*SequenceMBBI)) {
      if (SequenceMBBI->getOperand(1).getReg() != LHS ||
          SequenceMBBI->getOperand(2).getReg() != RHS ||
          SequenceMBBI->getOperand(3).getImm() != CC ||
          SelectDests.count(SequenceMBBI->getOperand(4).getReg()) ||
          SelectDests.count(SequenceMBBI->getOperand(5).getReg()))
        break;
      LastSelectPseudo = &*SequenceMBBI;
      SequenceMBBI->collectDebugValues(SelectDebugValues);
      SelectDests.insert(SequenceMBBI->getOperand(0).getReg());
    } else {
      if (SequenceMBBI->hasUnmodeledSideEffects() ||
          SequenceMBBI->mayLoadOrStore())
        break;
      if (llvm::any_of(SequenceMBBI->operands(), [&](MachineOperand &MO) {
            return MO.isReg() && MO.isUse() && SelectDests.count(MO.getReg());
          }))
        break;
    }
  }

  const RISCVInstrInfo &TII = *Subtarget.getInstrInfo();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction::iterator I = ++BB->getIterator();

  MachineBasicBlock *HeadMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *TailMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *IfFalseMBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(I, IfFalseMBB);
  F->insert(I, TailMBB);

  // Transfer debug instructions associated with the selects to TailMBB.
  for (MachineInstr *DebugInstr : SelectDebugValues) {
    TailMBB->push_back(DebugInstr->removeFromParent());
  }

  // Move all instructions after the sequence to TailMBB.
  TailMBB->splice(TailMBB->end(), HeadMBB,
                  std::next(LastSelectPseudo->getIterator()), HeadMBB->end());
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi nodes for the selects.
  TailMBB->transferSuccessorsAndUpdatePHIs(HeadMBB);
  // Set the successors for HeadMBB.
  HeadMBB->addSuccessor(IfFalseMBB);
  HeadMBB->addSuccessor(TailMBB);

  // Insert appropriate branch.
  BuildMI(HeadMBB, DL, TII.getBrCond(CC))
    .addReg(LHS)
    .addReg(RHS)
    .addMBB(TailMBB);

  // IfFalseMBB just falls through to TailMBB.
  IfFalseMBB->addSuccessor(TailMBB);

  // Create PHIs for all of the select pseudo-instructions.
  auto SelectMBBI = MI.getIterator();
  auto SelectEnd = std::next(LastSelectPseudo->getIterator());
  auto InsertionPoint = TailMBB->begin();
  while (SelectMBBI != SelectEnd) {
    auto Next = std::next(SelectMBBI);
    if (isSelectPseudo(*SelectMBBI)) {
      // %Result = phi [ %TrueValue, HeadMBB ], [ %FalseValue, IfFalseMBB ]
      BuildMI(*TailMBB, InsertionPoint, SelectMBBI->getDebugLoc(),
              TII.get(RISCV::PHI), SelectMBBI->getOperand(0).getReg())
          .addReg(SelectMBBI->getOperand(4).getReg())
          .addMBB(HeadMBB)
          .addReg(SelectMBBI->getOperand(5).getReg())
          .addMBB(IfFalseMBB);
      SelectMBBI->eraseFromParent();
    }
    SelectMBBI = Next;
  }

  F->getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
  return TailMBB;
}

MachineBasicBlock *
RISCVTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                 MachineBasicBlock *BB) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case RISCV::ReadCycleWide:
    assert(!Subtarget.is64Bit() &&
           "ReadCycleWrite is only to be used on riscv32");
    return emitReadCycleWidePseudo(MI, BB);
  case RISCV::Select_GPR_Using_CC_GPR:
  case RISCV::Select_FPR16_Using_CC_GPR:
  case RISCV::Select_FPR32_Using_CC_GPR:
  case RISCV::Select_FPR64_Using_CC_GPR:
    return emitSelectPseudo(MI, BB, Subtarget);
  case RISCV::BuildPairF64Pseudo:
    return emitBuildPairF64Pseudo(MI, BB);
  case RISCV::SplitF64Pseudo:
    return emitSplitF64Pseudo(MI, BB);
  case RISCV::PseudoQuietFLE_H:
    return emitQuietFCMP(MI, BB, RISCV::FLE_H, RISCV::FEQ_H, Subtarget);
  case RISCV::PseudoQuietFLT_H:
    return emitQuietFCMP(MI, BB, RISCV::FLT_H, RISCV::FEQ_H, Subtarget);
  case RISCV::PseudoQuietFLE_S:
    return emitQuietFCMP(MI, BB, RISCV::FLE_S, RISCV::FEQ_S, Subtarget);
  case RISCV::PseudoQuietFLT_S:
    return emitQuietFCMP(MI, BB, RISCV::FLT_S, RISCV::FEQ_S, Subtarget);
  case RISCV::PseudoQuietFLE_D:
    return emitQuietFCMP(MI, BB, RISCV::FLE_D, RISCV::FEQ_D, Subtarget);
  case RISCV::PseudoQuietFLT_D:
    return emitQuietFCMP(MI, BB, RISCV::FLT_D, RISCV::FEQ_D, Subtarget);
  }
}

void RISCVTargetLowering::AdjustInstrPostInstrSelection(MachineInstr &MI,
                                                        SDNode *Node) const {
  // Add FRM dependency to any instructions with dynamic rounding mode.
  unsigned Opc = MI.getOpcode();
  auto Idx = RISCV::getNamedOperandIdx(Opc, RISCV::OpName::frm);
  if (Idx < 0)
    return;
  if (MI.getOperand(Idx).getImm() != RISCVFPRndMode::DYN)
    return;
  // If the instruction already reads FRM, don't add another read.
  if (MI.readsRegister(RISCV::FRM))
    return;
  MI.addOperand(
      MachineOperand::CreateReg(RISCV::FRM, /*isDef*/ false, /*isImp*/ true));
}

// Calling Convention Implementation.
// The expectations for frontend ABI lowering vary from target to target.
// Ideally, an LLVM frontend would be able to avoid worrying about many ABI
// details, but this is a longer term goal. For now, we simply try to keep the
// role of the frontend as simple and well-defined as possible. The rules can
// be summarised as:
// * Never split up large scalar arguments. We handle them here.
// * If a hardfloat calling convention is being used, and the struct may be
// passed in a pair of registers (fp+fp, int+fp), and both registers are
// available, then pass as two separate arguments. If either the GPRs or FPRs
// are exhausted, then pass according to the rule below.
// * If a struct could never be passed in registers or directly in a stack
// slot (as it is larger than 2*XLEN and the floating point rules don't
// apply), then pass it using a pointer with the byval attribute.
// * If a struct is less than 2*XLEN, then coerce to either a two-element
// word-sized array or a 2*XLEN scalar (depending on alignment).
// * The frontend can determine whether a struct is returned by reference or
// not based on its size and fields. If it will be returned by reference, the
// frontend must modify the prototype so a pointer with the sret annotation is
// passed as the first argument. This is not necessary for large scalar
// returns.
// * Struct return values and varargs should be coerced to structs containing
// register-size fields in the same situations they would be for fixed
// arguments.

static const MCPhysReg ArgGPRs[] = {
  RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13,
  RISCV::X14, RISCV::X15, RISCV::X16, RISCV::X17
};
static const MCPhysReg ArgFPR16s[] = {
  RISCV::F10_H, RISCV::F11_H, RISCV::F12_H, RISCV::F13_H,
  RISCV::F14_H, RISCV::F15_H, RISCV::F16_H, RISCV::F17_H
};
static const MCPhysReg ArgFPR32s[] = {
  RISCV::F10_F, RISCV::F11_F, RISCV::F12_F, RISCV::F13_F,
  RISCV::F14_F, RISCV::F15_F, RISCV::F16_F, RISCV::F17_F
};
static const MCPhysReg ArgFPR64s[] = {
  RISCV::F10_D, RISCV::F11_D, RISCV::F12_D, RISCV::F13_D,
  RISCV::F14_D, RISCV::F15_D, RISCV::F16_D, RISCV::F17_D
};
// This is an interim calling convention and it may be changed in the future.
static const MCPhysReg ArgVRs[] = {
    RISCV::V8,  RISCV::V9,  RISCV::V10, RISCV::V11, RISCV::V12, RISCV::V13,
    RISCV::V14, RISCV::V15, RISCV::V16, RISCV::V17, RISCV::V18, RISCV::V19,
    RISCV::V20, RISCV::V21, RISCV::V22, RISCV::V23};
static const MCPhysReg ArgVRM2s[] = {RISCV::V8M2,  RISCV::V10M2, RISCV::V12M2,
                                     RISCV::V14M2, RISCV::V16M2, RISCV::V18M2,
                                     RISCV::V20M2, RISCV::V22M2};
static const MCPhysReg ArgVRM4s[] = {RISCV::V8M4, RISCV::V12M4, RISCV::V16M4,
                                     RISCV::V20M4};
static const MCPhysReg ArgVRM8s[] = {RISCV::V8M8, RISCV::V16M8};

// Pass a 2*XLEN argument that has been split into two XLEN values through
// registers or the stack as necessary.
static bool CC_RISCVAssign2XLen(unsigned XLen, CCState &State, CCValAssign VA1,
                                ISD::ArgFlagsTy ArgFlags1, unsigned ValNo2,
                                MVT ValVT2, MVT LocVT2,
                                ISD::ArgFlagsTy ArgFlags2) {
  unsigned XLenInBytes = XLen / 8;
  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // At least one half can be passed via register.
    State.addLoc(CCValAssign::getReg(VA1.getValNo(), VA1.getValVT(), Reg,
                                     VA1.getLocVT(), CCValAssign::Full));
  } else {
    // Both halves must be passed on the stack, with proper alignment.
    Align StackAlign =
        std::max(Align(XLenInBytes), ArgFlags1.getNonZeroOrigAlign());
    State.addLoc(
        CCValAssign::getMem(VA1.getValNo(), VA1.getValVT(),
                            State.AllocateStack(XLenInBytes, StackAlign),
                            VA1.getLocVT(), CCValAssign::Full));
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
    return false;
  }

  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // The second half can also be passed via register.
    State.addLoc(
        CCValAssign::getReg(ValNo2, ValVT2, Reg, LocVT2, CCValAssign::Full));
  } else {
    // The second half is passed via the stack, without additional alignment.
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
  }

  return false;
}

static unsigned allocateRVVReg(MVT ValVT, unsigned ValNo,
                               Optional<unsigned> FirstMaskArgument,
                               CCState &State, const RISCVTargetLowering &TLI) {
  const TargetRegisterClass *RC = TLI.getRegClassFor(ValVT);
  if (RC == &RISCV::VRRegClass) {
    // Assign the first mask argument to V0.
    // This is an interim calling convention and it may be changed in the
    // future.
    if (FirstMaskArgument.hasValue() && ValNo == FirstMaskArgument.getValue())
      return State.AllocateReg(RISCV::V0);
    return State.AllocateReg(ArgVRs);
  }
  if (RC == &RISCV::VRM2RegClass)
    return State.AllocateReg(ArgVRM2s);
  if (RC == &RISCV::VRM4RegClass)
    return State.AllocateReg(ArgVRM4s);
  if (RC == &RISCV::VRM8RegClass)
    return State.AllocateReg(ArgVRM8s);
  llvm_unreachable("Unhandled register class for ValueType");
}

// Implements the RISC-V calling convention. Returns true upon failure.
static bool CC_RISCV(const DataLayout &DL, RISCVABI::ABI ABI, unsigned ValNo,
                     MVT ValVT, MVT LocVT, CCValAssign::LocInfo LocInfo,
                     ISD::ArgFlagsTy ArgFlags, CCState &State, bool IsFixed,
                     bool IsRet, Type *OrigTy, const RISCVTargetLowering &TLI,
                     Optional<unsigned> FirstMaskArgument) {
  unsigned XLen = DL.getLargestLegalIntTypeSizeInBits();
  assert(XLen == 32 || XLen == 64);
  MVT XLenVT = XLen == 32 ? MVT::i32 : MVT::i64;

  // Any return value split in to more than two values can't be returned
  // directly. Vectors are returned via the available vector registers.
  if (!LocVT.isVector() && IsRet && ValNo > 1)
    return true;

  // UseGPRForF16_F32 if targeting one of the soft-float ABIs, if passing a
  // variadic argument, or if no F16/F32 argument registers are available.
  bool UseGPRForF16_F32 = true;
  // UseGPRForF64 if targeting soft-float ABIs or an FLEN=32 ABI, if passing a
  // variadic argument, or if no F64 argument registers are available.
  bool UseGPRForF64 = true;

  switch (ABI) {
  default:
    llvm_unreachable("Unexpected ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    break;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    UseGPRForF16_F32 = !IsFixed;
    break;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    UseGPRForF16_F32 = !IsFixed;
    UseGPRForF64 = !IsFixed;
    break;
  }

  // FPR16, FPR32, and FPR64 alias each other.
  if (State.getFirstUnallocated(ArgFPR32s) == array_lengthof(ArgFPR32s)) {
    UseGPRForF16_F32 = true;
    UseGPRForF64 = true;
  }

  // From this point on, rely on UseGPRForF16_F32, UseGPRForF64 and
  // similar local variables rather than directly checking against the target
  // ABI.

  if (UseGPRForF16_F32 && (ValVT == MVT::f16 || ValVT == MVT::f32)) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::BCvt;
  } else if (UseGPRForF64 && XLen == 64 && ValVT == MVT::f64) {
    LocVT = MVT::i64;
    LocInfo = CCValAssign::BCvt;
  }

  // If this is a variadic argument, the RISC-V calling convention requires
  // that it is assigned an 'even' or 'aligned' register if it has 8-byte
  // alignment (RV32) or 16-byte alignment (RV64). An aligned register should
  // be used regardless of whether the original argument was split during
  // legalisation or not. The argument will not be passed by registers if the
  // original type is larger than 2*XLEN, so the register alignment rule does
  // not apply.
  unsigned TwoXLenInBytes = (2 * XLen) / 8;
  if (!IsFixed && ArgFlags.getNonZeroOrigAlign() == TwoXLenInBytes &&
      DL.getTypeAllocSize(OrigTy) == TwoXLenInBytes) {
    unsigned RegIdx = State.getFirstUnallocated(ArgGPRs);
    // Skip 'odd' register if necessary.
    if (RegIdx != array_lengthof(ArgGPRs) && RegIdx % 2 == 1)
      State.AllocateReg(ArgGPRs);
  }

  SmallVectorImpl<CCValAssign> &PendingLocs = State.getPendingLocs();
  SmallVectorImpl<ISD::ArgFlagsTy> &PendingArgFlags =
      State.getPendingArgFlags();

  assert(PendingLocs.size() == PendingArgFlags.size() &&
         "PendingLocs and PendingArgFlags out of sync");

  // Handle passing f64 on RV32D with a soft float ABI or when floating point
  // registers are exhausted.
  if (UseGPRForF64 && XLen == 32 && ValVT == MVT::f64) {
    assert(!ArgFlags.isSplit() && PendingLocs.empty() &&
           "Can't lower f64 if it is split");
    // Depending on available argument GPRS, f64 may be passed in a pair of
    // GPRs, split between a GPR and the stack, or passed completely on the
    // stack. LowerCall/LowerFormalArguments/LowerReturn must recognise these
    // cases.
    Register Reg = State.AllocateReg(ArgGPRs);
    LocVT = MVT::i32;
    if (!Reg) {
      unsigned StackOffset = State.AllocateStack(8, Align(8));
      State.addLoc(
          CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      return false;
    }
    if (!State.AllocateReg(ArgGPRs))
      State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // Fixed-length vectors are located in the corresponding scalable-vector
  // container types.
  if (ValVT.isFixedLengthVector())
    LocVT = TLI.getContainerForFixedLengthVector(LocVT);

  // Split arguments might be passed indirectly, so keep track of the pending
  // values. Split vectors are passed via a mix of registers and indirectly, so
  // treat them as we would any other argument.
  if (ValVT.isScalarInteger() && (ArgFlags.isSplit() || !PendingLocs.empty())) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::Indirect;
    PendingLocs.push_back(
        CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));
    PendingArgFlags.push_back(ArgFlags);
    if (!ArgFlags.isSplitEnd()) {
      return false;
    }
  }

  // If the split argument only had two elements, it should be passed directly
  // in registers or on the stack.
  if (ValVT.isScalarInteger() && ArgFlags.isSplitEnd() &&
      PendingLocs.size() <= 2) {
    assert(PendingLocs.size() == 2 && "Unexpected PendingLocs.size()");
    // Apply the normal calling convention rules to the first half of the
    // split argument.
    CCValAssign VA = PendingLocs[0];
    ISD::ArgFlagsTy AF = PendingArgFlags[0];
    PendingLocs.clear();
    PendingArgFlags.clear();
    return CC_RISCVAssign2XLen(XLen, State, VA, AF, ValNo, ValVT, LocVT,
                               ArgFlags);
  }

  // Allocate to a register if possible, or else a stack slot.
  Register Reg;
  unsigned StoreSizeBytes = XLen / 8;
  Align StackAlign = Align(XLen / 8);

  if (ValVT == MVT::f16 && !UseGPRForF16_F32)
    Reg = State.AllocateReg(ArgFPR16s);
  else if (ValVT == MVT::f32 && !UseGPRForF16_F32)
    Reg = State.AllocateReg(ArgFPR32s);
  else if (ValVT == MVT::f64 && !UseGPRForF64)
    Reg = State.AllocateReg(ArgFPR64s);
  else if (ValVT.isVector()) {
    Reg = allocateRVVReg(ValVT, ValNo, FirstMaskArgument, State, TLI);
    if (!Reg) {
      // For return values, the vector must be passed fully via registers or
      // via the stack.
      // FIXME: The proposed vector ABI only mandates v8-v15 for return values,
      // but we're using all of them.
      if (IsRet)
        return true;
      // Try using a GPR to pass the address
      if ((Reg = State.AllocateReg(ArgGPRs))) {
        LocVT = XLenVT;
        LocInfo = CCValAssign::Indirect;
      } else if (ValVT.isScalableVector()) {
        LocVT = XLenVT;
        LocInfo = CCValAssign::Indirect;
      } else {
        // Pass fixed-length vectors on the stack.
        LocVT = ValVT;
        StoreSizeBytes = ValVT.getStoreSize();
        // Align vectors to their element sizes, being careful for vXi1
        // vectors.
        StackAlign = MaybeAlign(ValVT.getScalarSizeInBits() / 8).valueOrOne();
      }
    }
  } else {
    Reg = State.AllocateReg(ArgGPRs);
  }

  unsigned StackOffset =
      Reg ? 0 : State.AllocateStack(StoreSizeBytes, StackAlign);

  // If we reach this point and PendingLocs is non-empty, we must be at the
  // end of a split argument that must be passed indirectly.
  if (!PendingLocs.empty()) {
    assert(ArgFlags.isSplitEnd() && "Expected ArgFlags.isSplitEnd()");
    assert(PendingLocs.size() > 2 && "Unexpected PendingLocs.size()");

    for (auto &It : PendingLocs) {
      if (Reg)
        It.convertToReg(Reg);
      else
        It.convertToMem(StackOffset);
      State.addLoc(It);
    }
    PendingLocs.clear();
    PendingArgFlags.clear();
    return false;
  }

  assert((!UseGPRForF16_F32 || !UseGPRForF64 || LocVT == XLenVT ||
          (TLI.getSubtarget().hasVInstructions() && ValVT.isVector())) &&
         "Expected an XLenVT or vector types at this stage");

  if (Reg) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // When a floating-point value is passed on the stack, no bit-conversion is
  // needed.
  if (ValVT.isFloatingPoint()) {
    LocVT = ValVT;
    LocInfo = CCValAssign::Full;
  }
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
  return false;
}

template <typename ArgTy>
static Optional<unsigned> preAssignMask(const ArgTy &Args) {
  for (const auto &ArgIdx : enumerate(Args)) {
    MVT ArgVT = ArgIdx.value().VT;
    if (ArgVT.isVector() && ArgVT.getVectorElementType() == MVT::i1)
      return ArgIdx.index();
  }
  return None;
}

void RISCVTargetLowering::analyzeInputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet,
    RISCVCCAssignFn Fn) const {
  unsigned NumArgs = Ins.size();
  FunctionType *FType = MF.getFunction().getFunctionType();

  Optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasVInstructions())
    FirstMaskArgument = preAssignMask(Ins);

  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ArgVT = Ins[i].VT;
    ISD::ArgFlagsTy ArgFlags = Ins[i].Flags;

    Type *ArgTy = nullptr;
    if (IsRet)
      ArgTy = FType->getReturnType();
    else if (Ins[i].isOrigArg())
      ArgTy = FType->getParamType(Ins[i].getOrigArgIndex());

    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (Fn(MF.getDataLayout(), ABI, i, ArgVT, ArgVT, CCValAssign::Full,
           ArgFlags, CCInfo, /*IsFixed=*/true, IsRet, ArgTy, *this,
           FirstMaskArgument)) {
      LLVM_DEBUG(dbgs() << "InputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << '\n');
      llvm_unreachable(nullptr);
    }
  }
}

void RISCVTargetLowering::analyzeOutputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::OutputArg> &Outs, bool IsRet,
    CallLoweringInfo *CLI, RISCVCCAssignFn Fn) const {
  unsigned NumArgs = Outs.size();

  Optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasVInstructions())
    FirstMaskArgument = preAssignMask(Outs);

  for (unsigned i = 0; i != NumArgs; i++) {
    MVT ArgVT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    Type *OrigTy = CLI ? CLI->getArgs()[Outs[i].OrigArgIndex].Ty : nullptr;

    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (Fn(MF.getDataLayout(), ABI, i, ArgVT, ArgVT, CCValAssign::Full,
           ArgFlags, CCInfo, Outs[i].IsFixed, IsRet, OrigTy, *this,
           FirstMaskArgument)) {
      LLVM_DEBUG(dbgs() << "OutputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << "\n");
      llvm_unreachable(nullptr);
    }
  }
}

// Convert Val to a ValVT. Should not be called for CCValAssign::Indirect
// values.
static SDValue convertLocVTToValVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL,
                                   const RISCVSubtarget &Subtarget) {
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    if (VA.getValVT().isFixedLengthVector() && VA.getLocVT().isScalableVector())
      Val = convertFromScalableVector(VA.getValVT(), Val, DAG, Subtarget);
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT().isInteger() && VA.getValVT() == MVT::f16)
      Val = DAG.getNode(RISCVISD::FMV_H_X, DL, MVT::f16, Val);
    else if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32)
      Val = DAG.getNode(RISCVISD::FMV_W_X_RV64, DL, MVT::f32, Val);
    else
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromRegLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL,
                                const RISCVTargetLowering &TLI) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  EVT LocVT = VA.getLocVT();
  SDValue Val;
  const TargetRegisterClass *RC = TLI.getRegClassFor(LocVT.getSimpleVT());
  Register VReg = RegInfo.createVirtualRegister(RC);
  RegInfo.addLiveIn(VA.getLocReg(), VReg);
  Val = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);

  if (VA.getLocInfo() == CCValAssign::Indirect)
    return Val;

  return convertLocVTToValVT(DAG, Val, VA, DL, TLI.getSubtarget());
}

static SDValue convertValVTToLocVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL,
                                   const RISCVSubtarget &Subtarget) {
  EVT LocVT = VA.getLocVT();

  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    if (VA.getValVT().isFixedLengthVector() && LocVT.isScalableVector())
      Val = convertToScalableVector(LocVT, Val, DAG, Subtarget);
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT().isInteger() && VA.getValVT() == MVT::f16)
      Val = DAG.getNode(RISCVISD::FMV_X_ANYEXTH, DL, VA.getLocVT(), Val);
    else if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32)
      Val = DAG.getNode(RISCVISD::FMV_X_ANYEXTW_RV64, DL, MVT::i64, Val);
    else
      Val = DAG.getNode(ISD::BITCAST, DL, LocVT, Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromMemLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  EVT LocVT = VA.getLocVT();
  EVT ValVT = VA.getValVT();
  EVT PtrVT = MVT::getIntegerVT(DAG.getDataLayout().getPointerSizeInBits(0));
  if (ValVT.isScalableVector()) {
    // When the value is a scalable vector, we save the pointer which points to
    // the scalable vector value in the stack. The ValVT will be the pointer
    // type, instead of the scalable vector type.
    ValVT = LocVT;
  }
  int FI = MFI.CreateFixedObject(ValVT.getStoreSize(), VA.getLocMemOffset(),
                                 /*IsImmutable=*/true);
  SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
  SDValue Val;

  ISD::LoadExtType ExtType;
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
  case CCValAssign::Indirect:
  case CCValAssign::BCvt:
    ExtType = ISD::NON_EXTLOAD;
    break;
  }
  Val = DAG.getExtLoad(
      ExtType, DL, LocVT, Chain, FIN,
      MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI), ValVT);
  return Val;
}

static SDValue unpackF64OnRV32DSoftABI(SelectionDAG &DAG, SDValue Chain,
                                       const CCValAssign &VA, const SDLoc &DL) {
  assert(VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64 &&
         "Unexpected VA");
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  if (VA.isMemLoc()) {
    // f64 is passed on the stack.
    int FI =
        MFI.CreateFixedObject(8, VA.getLocMemOffset(), /*IsImmutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    return DAG.getLoad(MVT::f64, DL, Chain, FIN,
                       MachinePointerInfo::getFixedStack(MF, FI));
  }

  assert(VA.isRegLoc() && "Expected register VA assignment");

  Register LoVReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
  RegInfo.addLiveIn(VA.getLocReg(), LoVReg);
  SDValue Lo = DAG.getCopyFromReg(Chain, DL, LoVReg, MVT::i32);
  SDValue Hi;
  if (VA.getLocReg() == RISCV::X17) {
    // Second half of f64 is passed on the stack.
    int FI = MFI.CreateFixedObject(4, 0, /*IsImmutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    Hi = DAG.getLoad(MVT::i32, DL, Chain, FIN,
                     MachinePointerInfo::getFixedStack(MF, FI));
  } else {
    // Second half of f64 is passed in another GPR.
    Register HiVReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
    RegInfo.addLiveIn(VA.getLocReg() + 1, HiVReg);
    Hi = DAG.getCopyFromReg(Chain, DL, HiVReg, MVT::i32);
  }
  return DAG.getNode(RISCVISD::BuildPairF64, DL, MVT::f64, Lo, Hi);
}

// FastCC has less than 1% performance improvement for some particular
// benchmark. But theoretically, it may has benenfit for some cases.
static bool CC_RISCV_FastCC(const DataLayout &DL, RISCVABI::ABI ABI,
                            unsigned ValNo, MVT ValVT, MVT LocVT,
                            CCValAssign::LocInfo LocInfo,
                            ISD::ArgFlagsTy ArgFlags, CCState &State,
                            bool IsFixed, bool IsRet, Type *OrigTy,
                            const RISCVTargetLowering &TLI,
                            Optional<unsigned> FirstMaskArgument) {

  // X5 and X6 might be used for save-restore libcall.
  static const MCPhysReg GPRList[] = {
      RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13, RISCV::X14,
      RISCV::X15, RISCV::X16, RISCV::X17, RISCV::X7,  RISCV::X28,
      RISCV::X29, RISCV::X30, RISCV::X31};

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    if (unsigned Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f16) {
    static const MCPhysReg FPR16List[] = {
        RISCV::F10_H, RISCV::F11_H, RISCV::F12_H, RISCV::F13_H, RISCV::F14_H,
        RISCV::F15_H, RISCV::F16_H, RISCV::F17_H, RISCV::F0_H,  RISCV::F1_H,
        RISCV::F2_H,  RISCV::F3_H,  RISCV::F4_H,  RISCV::F5_H,  RISCV::F6_H,
        RISCV::F7_H,  RISCV::F28_H, RISCV::F29_H, RISCV::F30_H, RISCV::F31_H};
    if (unsigned Reg = State.AllocateReg(FPR16List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32) {
    static const MCPhysReg FPR32List[] = {
        RISCV::F10_F, RISCV::F11_F, RISCV::F12_F, RISCV::F13_F, RISCV::F14_F,
        RISCV::F15_F, RISCV::F16_F, RISCV::F17_F, RISCV::F0_F,  RISCV::F1_F,
        RISCV::F2_F,  RISCV::F3_F,  RISCV::F4_F,  RISCV::F5_F,  RISCV::F6_F,
        RISCV::F7_F,  RISCV::F28_F, RISCV::F29_F, RISCV::F30_F, RISCV::F31_F};
    if (unsigned Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64) {
    static const MCPhysReg FPR64List[] = {
        RISCV::F10_D, RISCV::F11_D, RISCV::F12_D, RISCV::F13_D, RISCV::F14_D,
        RISCV::F15_D, RISCV::F16_D, RISCV::F17_D, RISCV::F0_D,  RISCV::F1_D,
        RISCV::F2_D,  RISCV::F3_D,  RISCV::F4_D,  RISCV::F5_D,  RISCV::F6_D,
        RISCV::F7_D,  RISCV::F28_D, RISCV::F29_D, RISCV::F30_D, RISCV::F31_D};
    if (unsigned Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    unsigned Offset4 = State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset4, LocVT, LocInfo));
    return false;
  }

  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    unsigned Offset5 = State.AllocateStack(8, Align(8));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset5, LocVT, LocInfo));
    return false;
  }

  if (LocVT.isVector()) {
    if (unsigned Reg =
            allocateRVVReg(ValVT, ValNo, FirstMaskArgument, State, TLI)) {
      // Fixed-length vectors are located in the corresponding scalable-vector
      // container types.
      if (ValVT.isFixedLengthVector())
        LocVT = TLI.getContainerForFixedLengthVector(LocVT);
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    } else {
      // Try and pass the address via a "fast" GPR.
      if (unsigned GPRReg = State.AllocateReg(GPRList)) {
        LocInfo = CCValAssign::Indirect;
        LocVT = TLI.getSubtarget().getXLenVT();
        State.addLoc(CCValAssign::getReg(ValNo, ValVT, GPRReg, LocVT, LocInfo));
      } else if (ValVT.isFixedLengthVector()) {
        auto StackAlign =
            MaybeAlign(ValVT.getScalarSizeInBits() / 8).valueOrOne();
        unsigned StackOffset =
            State.AllocateStack(ValVT.getStoreSize(), StackAlign);
        State.addLoc(
            CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      } else {
        // Can't pass scalable vectors on the stack.
        return true;
      }
    }

    return false;
  }

  return true; // CC didn't match.
}

static bool CC_RISCV_GHC(unsigned ValNo, MVT ValVT, MVT LocVT,
                         CCValAssign::LocInfo LocInfo,
                         ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    // Pass in STG registers: Base, Sp, Hp, R1, R2, R3, R4, R5, R6, R7, SpLim
    //                        s1    s2  s3  s4  s5  s6  s7  s8  s9  s10 s11
    static const MCPhysReg GPRList[] = {
        RISCV::X9, RISCV::X18, RISCV::X19, RISCV::X20, RISCV::X21, RISCV::X22,
        RISCV::X23, RISCV::X24, RISCV::X25, RISCV::X26, RISCV::X27};
    if (unsigned Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32) {
    // Pass in STG registers: F1, ..., F6
    //                        fs0 ... fs5
    static const MCPhysReg FPR32List[] = {RISCV::F8_F, RISCV::F9_F,
                                          RISCV::F18_F, RISCV::F19_F,
                                          RISCV::F20_F, RISCV::F21_F};
    if (unsigned Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64) {
    // Pass in STG registers: D1, ..., D6
    //                        fs6 ... fs11
    static const MCPhysReg FPR64List[] = {RISCV::F22_D, RISCV::F23_D,
                                          RISCV::F24_D, RISCV::F25_D,
                                          RISCV::F26_D, RISCV::F27_D};
    if (unsigned Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  report_fatal_error("No registers left in GHC calling convention");
  return true;
}

// Transform physical registers into virtual registers.
SDValue RISCVTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  MachineFunction &MF = DAG.getMachineFunction();

  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  case CallingConv::GHC:
    if (!MF.getSubtarget().getFeatureBits()[RISCV::FeatureStdExtF] ||
        !MF.getSubtarget().getFeatureBits()[RISCV::FeatureStdExtD])
      report_fatal_error(
        "GHC calling convention requires the F and D instruction set extensions");
  }

  const Function &Func = MF.getFunction();
  if (Func.hasFnAttribute("interrupt")) {
    if (!Func.arg_empty())
      report_fatal_error(
        "Functions with the interrupt attribute cannot have arguments!");

    StringRef Kind =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

    if (!(Kind == "user" || Kind == "supervisor" || Kind == "machine"))
      report_fatal_error(
        "Function interrupt attribute argument not supported!");
  }

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = Subtarget.getXLenVT();
  unsigned XLenInBytes = Subtarget.getXLen() / 8;
  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (CallConv == CallingConv::GHC)
    CCInfo.AnalyzeFormalArguments(Ins, CC_RISCV_GHC);
  else
    analyzeInputArgs(MF, CCInfo, Ins, /*IsRet=*/false,
                     CallConv == CallingConv::Fast ? CC_RISCV_FastCC
                                                   : CC_RISCV);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue;
    // Passing f64 on RV32D with a soft float ABI must be handled as a special
    // case.
    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64)
      ArgValue = unpackF64OnRV32DSoftABI(DAG, Chain, VA, DL);
    else if (VA.isRegLoc())
      ArgValue = unpackFromRegLoc(DAG, Chain, VA, DL, *this);
    else
      ArgValue = unpackFromMemLoc(DAG, Chain, VA, DL);

    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // If the original argument was split and passed by reference (e.g. i128
      // on RV32), we need to load all parts of it here (using the same
      // address). Vectors may be partly split to registers and partly to the
      // stack, in which case the base address is partly offset and subsequent
      // stores are relative to that.
      InVals.push_back(DAG.getLoad(VA.getValVT(), DL, Chain, ArgValue,
                                   MachinePointerInfo()));
      unsigned ArgIndex = Ins[i].OrigArgIndex;
      unsigned ArgPartOffset = Ins[i].PartOffset;
      assert(VA.getValVT().isVector() || ArgPartOffset == 0);
      while (i + 1 != e && Ins[i + 1].OrigArgIndex == ArgIndex) {
        CCValAssign &PartVA = ArgLocs[i + 1];
        unsigned PartOffset = Ins[i + 1].PartOffset - ArgPartOffset;
        SDValue Offset = DAG.getIntPtrConstant(PartOffset, DL);
        if (PartVA.getValVT().isScalableVector())
          Offset = DAG.getNode(ISD::VSCALE, DL, XLenVT, Offset);
        SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, ArgValue, Offset);
        InVals.push_back(DAG.getLoad(PartVA.getValVT(), DL, Chain, Address,
                                     MachinePointerInfo()));
        ++i;
      }
      continue;
    }
    InVals.push_back(ArgValue);
  }

  if (IsVarArg) {
    ArrayRef<MCPhysReg> ArgRegs = makeArrayRef(ArgGPRs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &RISCV::GPRRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    RISCVMachineFunctionInfo *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a7.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getNextStackOffset();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = XLenInBytes * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
    RVFI->setVarArgsFrameIndex(FI);

    // If saving an odd number of registers then create an extra stack slot to
    // ensure that the frame pointer is 2*XLEN-aligned, which in turn ensures
    // offsets to even-numbered registered remain 2*XLEN-aligned.
    if (Idx % 2) {
      MFI.CreateFixedObject(XLenInBytes, VaArgOffset - (int)XLenInBytes, true);
      VarArgsSaveSize += XLenInBytes;
    }

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size();
         ++I, VaArgOffset += XLenInBytes) {
      const Register Reg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(ArgRegs[I], Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, XLenVT);
      FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
    RVFI->setVarArgsSaveSize(VarArgsSaveSize);
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens for vararg functions.
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

/// isEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization.
/// Note: This is modelled after ARM's IsEligibleForTailCallOptimization.
bool RISCVTargetLowering::isEligibleForTailCallOptimization(
    CCState &CCInfo, CallLoweringInfo &CLI, MachineFunction &MF,
    const SmallVector<CCValAssign, 16> &ArgLocs) const {

  auto &Callee = CLI.Callee;
  auto CalleeCC = CLI.CallConv;
  auto &Outs = CLI.Outs;
  auto &Caller = MF.getFunction();
  auto CallerCC = Caller.getCallingConv();

  // Exception-handling functions need a special set of instructions to
  // indicate a return to the hardware. Tail-calling another function would
  // probably break this.
  // TODO: The "interrupt" attribute isn't currently defined by RISC-V. This
  // should be expanded as new function attributes are introduced.
  if (Caller.hasFnAttribute("interrupt"))
    return false;

  // Do not tail call opt if the stack is used to pass parameters.
  if (CCInfo.getNextStackOffset() != 0)
    return false;

  // Do not tail call opt if any parameters need to be passed indirectly.
  // Since long doubles (fp128) and i128 are larger than 2*XLEN, they are
  // passed indirectly. So the address of the value will be passed in a
  // register, or if not available, then the address is put on the stack. In
  // order to pass indirectly, space on the stack often needs to be allocated
  // in order to store the value. In this case the CCInfo.getNextStackOffset()
  // != 0 check is not enough and we need to check if any CCValAssign ArgsLocs
  // are passed CCValAssign::Indirect.
  for (auto &VA : ArgLocs)
    if (VA.getLocInfo() == CCValAssign::Indirect)
      return false;

  // Do not tail call opt if either caller or callee uses struct return
  // semantics.
  auto IsCallerStructRet = Caller.hasStructRetAttr();
  auto IsCalleeStructRet = Outs.empty() ? false : Outs[0].Flags.isSRet();
  if (IsCallerStructRet || IsCalleeStructRet)
    return false;

  // Externally-defined functions with weak linkage should not be
  // tail-called. The behaviour of branch instructions in this situation (as
  // used for tail calls) is implementation-defined, so we cannot rely on the
  // linker replacing the tail call with a return.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    if (GV->hasExternalWeakLinkage())
      return false;
  }

  // The callee has to preserve all registers the caller needs to preserve.
  const RISCVRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);
  if (CalleeCC != CallerCC) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }

  // Byval parameters hand the function a pointer directly into the stack area
  // we want to reuse during a tail call. Working around this *is* possible
  // but less efficient and uglier in LowerCall.
  for (auto &Arg : Outs)
    if (Arg.Flags.isByVal())
      return false;

  return true;
}

static Align getPrefTypeAlign(EVT VT, SelectionDAG &DAG) {
  return DAG.getDataLayout().getPrefTypeAlign(
      VT.getTypeForEVT(*DAG.getContext()));
}

// Lower a call to a callseq_start + CALL + callseq_end chain, and add input
// and output parameter nodes.
SDValue RISCVTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                       SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = Subtarget.getXLenVT();

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (CallConv == CallingConv::GHC)
    ArgCCInfo.AnalyzeCallOperands(Outs, CC_RISCV_GHC);
  else
    analyzeOutputArgs(MF, ArgCCInfo, Outs, /*IsRet=*/false, &CLI,
                      CallConv == CallingConv::Fast ? CC_RISCV_FastCC
                                                    : CC_RISCV);

  // Check if it's really possible to do a tail call.
  if (IsTailCall)
    IsTailCall = isEligibleForTailCallOptimization(ArgCCInfo, CLI, MF, ArgLocs);

  if (IsTailCall)
    ++NumTailCalls;
  else if (CLI.CB && CLI.CB->isMustTailCall())
    report_fatal_error("failed to perform tail call elimination on a call "
                       "site marked musttail");

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

  // Create local copies for byval args
  SmallVector<SDValue, 8> ByValArgs;
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    if (!Flags.isByVal())
      continue;

    SDValue Arg = OutVals[i];
    unsigned Size = Flags.getByValSize();
    Align Alignment = Flags.getNonZeroByValAlign();

    int FI =
        MF.getFrameInfo().CreateStackObject(Size, Alignment, /*isSS=*/false);
    SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
    SDValue SizeNode = DAG.getConstant(Size, DL, XLenVT);

    Chain = DAG.getMemcpy(Chain, DL, FIPtr, Arg, SizeNode, Alignment,
                          /*IsVolatile=*/false,
                          /*AlwaysInline=*/false, IsTailCall,
                          MachinePointerInfo(), MachinePointerInfo());
    ByValArgs.push_back(FIPtr);
  }

  if (!IsTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<Register, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned i = 0, j = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // Handle passing f64 on RV32D with a soft float ABI as a special case.
    bool IsF64OnRV32DSoftABI =
        VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64;
    if (IsF64OnRV32DSoftABI && VA.isRegLoc()) {
      SDValue SplitF64 = DAG.getNode(
          RISCVISD::SplitF64, DL, DAG.getVTList(MVT::i32, MVT::i32), ArgValue);
      SDValue Lo = SplitF64.getValue(0);
      SDValue Hi = SplitF64.getValue(1);

      Register RegLo = VA.getLocReg();
      RegsToPass.push_back(std::make_pair(RegLo, Lo));

      if (RegLo == RISCV::X17) {
        // Second half of f64 is passed on the stack.
        // Work out the address of the stack slot.
        if (!StackPtr.getNode())
          StackPtr = DAG.getCopyFromReg(Chain, DL, RISCV::X2, PtrVT);
        // Emit the store.
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, Hi, StackPtr, MachinePointerInfo()));
      } else {
        // Second half of f64 is passed in another GPR.
        assert(RegLo < RISCV::X31 && "Invalid register pair");
        Register RegHigh = RegLo + 1;
        RegsToPass.push_back(std::make_pair(RegHigh, Hi));
      }
      continue;
    }

    // IsF64OnRV32DSoftABI && VA.isMemLoc() is handled below in the same way
    // as any other MemLoc.

    // Promote the value if needed.
    // For now, only handle fully promoted and indirect arguments.
    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // Store the argument in a stack slot and pass its address.
      Align StackAlign =
          std::max(getPrefTypeAlign(Outs[i].ArgVT, DAG),
                   getPrefTypeAlign(ArgValue.getValueType(), DAG));
      TypeSize StoredSize = ArgValue.getValueType().getStoreSize();
      // If the original argument was split (e.g. i128), we need
      // to store the required parts of it here (and pass just one address).
      // Vectors may be partly split to registers and partly to the stack, in
      // which case the base address is partly offset and subsequent stores are
      // relative to that.
      unsigned ArgIndex = Outs[i].OrigArgIndex;
      unsigned ArgPartOffset = Outs[i].PartOffset;
      assert(VA.getValVT().isVector() || ArgPartOffset == 0);
      // Calculate the total size to store. We don't have access to what we're
      // actually storing other than performing the loop and collecting the
      // info.
      SmallVector<std::pair<SDValue, SDValue>> Parts;
      while (i + 1 != e && Outs[i + 1].OrigArgIndex == ArgIndex) {
        SDValue PartValue = OutVals[i + 1];
        unsigned PartOffset = Outs[i + 1].PartOffset - ArgPartOffset;
        SDValue Offset = DAG.getIntPtrConstant(PartOffset, DL);
        EVT PartVT = PartValue.getValueType();
        if (PartVT.isScalableVector())
          Offset = DAG.getNode(ISD::VSCALE, DL, XLenVT, Offset);
        StoredSize += PartVT.getStoreSize();
        StackAlign = std::max(StackAlign, getPrefTypeAlign(PartVT, DAG));
        Parts.push_back(std::make_pair(PartValue, Offset));
        ++i;
      }
      SDValue SpillSlot = DAG.CreateStackTemporary(StoredSize, StackAlign);
      int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, SpillSlot,
                       MachinePointerInfo::getFixedStack(MF, FI)));
      for (const auto &Part : Parts) {
        SDValue PartValue = Part.first;
        SDValue PartOffset = Part.second;
        SDValue Address =
            DAG.getNode(ISD::ADD, DL, PtrVT, SpillSlot, PartOffset);
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, PartValue, Address,
                         MachinePointerInfo::getFixedStack(MF, FI)));
      }
      ArgValue = SpillSlot;
    } else {
      ArgValue = convertValVTToLocVT(DAG, ArgValue, VA, DL, Subtarget);
    }

    // Use local copy if it is a byval arg.
    if (Flags.isByVal())
      ArgValue = ByValArgs[j++];

    if (VA.isRegLoc()) {
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");
      assert(!IsTailCall && "Tail call not allowed if stack is used "
                            "for passing parameters");

      // Work out the address of the stack slot.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, RISCV::X2, PtrVT);
      SDValue Address =
          DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                      DAG.getIntPtrConstant(VA.getLocMemOffset(), DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  SDValue Glue;

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, Glue);
    Glue = Chain.getValue(1);
  }

  // Validate that none of the argument registers have been marked as
  // reserved, if so report an error. Do the same for the return address if this
  // is not a tailcall.
  validateCCReservedRegs(RegsToPass, MF);
  if (!IsTailCall &&
      MF.getSubtarget<RISCVSubtarget>().isRegisterReservedByUser(RISCV::X1))
    MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(),
        "Return address register required, but has been reserved."});

  // If the callee is a GlobalAddress/ExternalSymbol node, turn it into a
  // TargetGlobalAddress/TargetExternalSymbol node so that legalize won't
  // split it and then direct call can be matched by PseudoCALL.
  if (GlobalAddressSDNode *S = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = S->getGlobal();

    unsigned OpFlags = RISCVII::MO_CALL;
    if (!getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV))
      OpFlags = RISCVII::MO_PLT;

    Callee = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, OpFlags);
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    unsigned OpFlags = RISCVII::MO_CALL;

    if (!getTargetMachine().shouldAssumeDSOLocal(*MF.getFunction().getParent(),
                                                 nullptr))
      OpFlags = RISCVII::MO_PLT;

    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), PtrVT, OpFlags);
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  if (!IsTailCall) {
    // Add a register mask operand representing the call-preserved registers.
    const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
    const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
    assert(Mask && "Missing call preserved mask for calling convention");
    Ops.push_back(DAG.getRegisterMask(Mask));
  }

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  if (IsTailCall) {
    MF.getFrameInfo().setHasTailCall();
    return DAG.getNode(RISCVISD::TAIL, DL, NodeTys, Ops);
  }

  Chain = DAG.getNode(RISCVISD::CALL, DL, NodeTys, Ops);
  DAG.addNoMergeSiteInfo(Chain.getNode(), CLI.NoMerge);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true),
                             Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());
  analyzeInputArgs(MF, RetCCInfo, Ins, /*IsRet=*/true, CC_RISCV);

  // Copy all of the result registers out of their specified physreg.
  for (auto &VA : RVLocs) {
    // Copy the value out
    SDValue RetValue =
        DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), Glue);
    // Glue the RetValue to the end of the call sequence
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64) {
      assert(VA.getLocReg() == ArgGPRs[0] && "Unexpected reg assignment");
      SDValue RetValue2 =
          DAG.getCopyFromReg(Chain, DL, ArgGPRs[1], MVT::i32, Glue);
      Chain = RetValue2.getValue(1);
      Glue = RetValue2.getValue(2);
      RetValue = DAG.getNode(RISCVISD::BuildPairF64, DL, MVT::f64, RetValue,
                             RetValue2);
    }

    RetValue = convertLocVTToValVT(DAG, RetValue, VA, DL, Subtarget);

    InVals.push_back(RetValue);
  }

  return Chain;
}

bool RISCVTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);

  Optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasVInstructions())
    FirstMaskArgument = preAssignMask(Outs);

  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    MVT VT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (CC_RISCV(MF.getDataLayout(), ABI, i, VT, VT, CCValAssign::Full,
                 ArgFlags, CCInfo, /*IsFixed=*/true, /*IsRet=*/true, nullptr,
                 *this, FirstMaskArgument))
      return false;
  }
  return true;
}

SDValue
RISCVTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                 bool IsVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 const SDLoc &DL, SelectionDAG &DAG) const {
  const MachineFunction &MF = DAG.getMachineFunction();
  const RISCVSubtarget &STI = MF.getSubtarget<RISCVSubtarget>();

  // Stores the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> RVLocs;

  // Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  analyzeOutputArgs(DAG.getMachineFunction(), CCInfo, Outs, /*IsRet=*/true,
                    nullptr, CC_RISCV);

  if (CallConv == CallingConv::GHC && !RVLocs.empty())
    report_fatal_error("GHC functions return void only");

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = RVLocs.size(); i < e; ++i) {
    SDValue Val = OutVals[i];
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64) {
      // Handle returning f64 on RV32D with a soft float ABI.
      assert(VA.isRegLoc() && "Expected return via registers");
      SDValue SplitF64 = DAG.getNode(RISCVISD::SplitF64, DL,
                                     DAG.getVTList(MVT::i32, MVT::i32), Val);
      SDValue Lo = SplitF64.getValue(0);
      SDValue Hi = SplitF64.getValue(1);
      Register RegLo = VA.getLocReg();
      assert(RegLo < RISCV::X31 && "Invalid register pair");
      Register RegHi = RegLo + 1;

      if (STI.isRegisterReservedByUser(RegLo) ||
          STI.isRegisterReservedByUser(RegHi))
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
            MF.getFunction(),
            "Return value register required, but has been reserved."});

      Chain = DAG.getCopyToReg(Chain, DL, RegLo, Lo, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegLo, MVT::i32));
      Chain = DAG.getCopyToReg(Chain, DL, RegHi, Hi, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegHi, MVT::i32));
    } else {
      // Handle a 'normal' return.
      Val = convertValVTToLocVT(DAG, Val, VA, DL, Subtarget);
      Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Glue);

      if (STI.isRegisterReservedByUser(VA.getLocReg()))
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
            MF.getFunction(),
            "Return value register required, but has been reserved."});

      // Guarantee that all emitted copies are stuck together.
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
    }
  }

  RetOps[0] = Chain; // Update chain.

  // Add the glue node if we have it.
  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }

  unsigned RetOpc = RISCVISD::RET_FLAG;
  // Interrupt service routines use different return instructions.
  const Function &Func = DAG.getMachineFunction().getFunction();
  if (Func.hasFnAttribute("interrupt")) {
    if (!Func.getReturnType()->isVoidTy())
      report_fatal_error(
          "Functions with the interrupt attribute must have void return type!");

    MachineFunction &MF = DAG.getMachineFunction();
    StringRef Kind =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

    if (Kind == "user")
      RetOpc = RISCVISD::URET_FLAG;
    else if (Kind == "supervisor")
      RetOpc = RISCVISD::SRET_FLAG;
    else
      RetOpc = RISCVISD::MRET_FLAG;
  }

  return DAG.getNode(RetOpc, DL, MVT::Other, RetOps);
}

void RISCVTargetLowering::validateCCReservedRegs(
    const SmallVectorImpl<std::pair<llvm::Register, llvm::SDValue>> &Regs,
    MachineFunction &MF) const {
  const Function &F = MF.getFunction();
  const RISCVSubtarget &STI = MF.getSubtarget<RISCVSubtarget>();

  if (llvm::any_of(Regs, [&STI](auto Reg) {
        return STI.isRegisterReservedByUser(Reg.first);
      }))
    F.getContext().diagnose(DiagnosticInfoUnsupported{
        F, "Argument register required, but has been reserved."});
}

bool RISCVTargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  return CI->isTailCall();
}

const char *RISCVTargetLowering::getTargetNodeName(unsigned Opcode) const {
#define NODE_NAME_CASE(NODE)                                                   \
  case RISCVISD::NODE:                                                         \
    return "RISCVISD::" #NODE;
  // clang-format off
  switch ((RISCVISD::NodeType)Opcode) {
  case RISCVISD::FIRST_NUMBER:
    break;
  NODE_NAME_CASE(RET_FLAG)
  NODE_NAME_CASE(URET_FLAG)
  NODE_NAME_CASE(SRET_FLAG)
  NODE_NAME_CASE(MRET_FLAG)
  NODE_NAME_CASE(CALL)
  NODE_NAME_CASE(SELECT_CC)
  NODE_NAME_CASE(BR_CC)
  NODE_NAME_CASE(BuildPairF64)
  NODE_NAME_CASE(SplitF64)
  NODE_NAME_CASE(TAIL)
  NODE_NAME_CASE(MULHSU)
  NODE_NAME_CASE(SLLW)
  NODE_NAME_CASE(SRAW)
  NODE_NAME_CASE(SRLW)
  NODE_NAME_CASE(DIVW)
  NODE_NAME_CASE(DIVUW)
  NODE_NAME_CASE(REMUW)
  NODE_NAME_CASE(ROLW)
  NODE_NAME_CASE(RORW)
  NODE_NAME_CASE(CLZW)
  NODE_NAME_CASE(CTZW)
  NODE_NAME_CASE(FSLW)
  NODE_NAME_CASE(FSRW)
  NODE_NAME_CASE(FSL)
  NODE_NAME_CASE(FSR)
  NODE_NAME_CASE(FMV_H_X)
  NODE_NAME_CASE(FMV_X_ANYEXTH)
  NODE_NAME_CASE(FMV_X_SIGNEXTH)
  NODE_NAME_CASE(FMV_W_X_RV64)
  NODE_NAME_CASE(FMV_X_ANYEXTW_RV64)
  NODE_NAME_CASE(FCVT_X)
  NODE_NAME_CASE(FCVT_XU)
  NODE_NAME_CASE(FCVT_W_RV64)
  NODE_NAME_CASE(FCVT_WU_RV64)
  NODE_NAME_CASE(STRICT_FCVT_W_RV64)
  NODE_NAME_CASE(STRICT_FCVT_WU_RV64)
  NODE_NAME_CASE(READ_CYCLE_WIDE)
  NODE_NAME_CASE(GREV)
  NODE_NAME_CASE(GREVW)
  NODE_NAME_CASE(GORC)
  NODE_NAME_CASE(GORCW)
  NODE_NAME_CASE(SHFL)
  NODE_NAME_CASE(SHFLW)
  NODE_NAME_CASE(UNSHFL)
  NODE_NAME_CASE(UNSHFLW)
  NODE_NAME_CASE(BFP)
  NODE_NAME_CASE(BFPW)
  NODE_NAME_CASE(BCOMPRESS)
  NODE_NAME_CASE(BCOMPRESSW)
  NODE_NAME_CASE(BDECOMPRESS)
  NODE_NAME_CASE(BDECOMPRESSW)
  NODE_NAME_CASE(VMV_V_X_VL)
  NODE_NAME_CASE(VFMV_V_F_VL)
  NODE_NAME_CASE(VMV_X_S)
  NODE_NAME_CASE(VMV_S_X_VL)
  NODE_NAME_CASE(VFMV_S_F_VL)
  NODE_NAME_CASE(SPLAT_VECTOR_SPLIT_I64_VL)
  NODE_NAME_CASE(READ_VLENB)
  NODE_NAME_CASE(TRUNCATE_VECTOR_VL)
  NODE_NAME_CASE(VSLIDEUP_VL)
  NODE_NAME_CASE(VSLIDE1UP_VL)
  NODE_NAME_CASE(VSLIDEDOWN_VL)
  NODE_NAME_CASE(VSLIDE1DOWN_VL)
  NODE_NAME_CASE(VID_VL)
  NODE_NAME_CASE(VFNCVT_ROD_VL)
  NODE_NAME_CASE(VECREDUCE_ADD_VL)
  NODE_NAME_CASE(VECREDUCE_UMAX_VL)
  NODE_NAME_CASE(VECREDUCE_SMAX_VL)
  NODE_NAME_CASE(VECREDUCE_UMIN_VL)
  NODE_NAME_CASE(VECREDUCE_SMIN_VL)
  NODE_NAME_CASE(VECREDUCE_AND_VL)
  NODE_NAME_CASE(VECREDUCE_OR_VL)
  NODE_NAME_CASE(VECREDUCE_XOR_VL)
  NODE_NAME_CASE(VECREDUCE_FADD_VL)
  NODE_NAME_CASE(VECREDUCE_SEQ_FADD_VL)
  NODE_NAME_CASE(VECREDUCE_FMIN_VL)
  NODE_NAME_CASE(VECREDUCE_FMAX_VL)
  NODE_NAME_CASE(ADD_VL)
  NODE_NAME_CASE(AND_VL)
  NODE_NAME_CASE(MUL_VL)
  NODE_NAME_CASE(OR_VL)
  NODE_NAME_CASE(SDIV_VL)
  NODE_NAME_CASE(SHL_VL)
  NODE_NAME_CASE(SREM_VL)
  NODE_NAME_CASE(SRA_VL)
  NODE_NAME_CASE(SRL_VL)
  NODE_NAME_CASE(SUB_VL)
  NODE_NAME_CASE(UDIV_VL)
  NODE_NAME_CASE(UREM_VL)
  NODE_NAME_CASE(XOR_VL)
  NODE_NAME_CASE(SADDSAT_VL)
  NODE_NAME_CASE(UADDSAT_VL)
  NODE_NAME_CASE(SSUBSAT_VL)
  NODE_NAME_CASE(USUBSAT_VL)
  NODE_NAME_CASE(FADD_VL)
  NODE_NAME_CASE(FSUB_VL)
  NODE_NAME_CASE(FMUL_VL)
  NODE_NAME_CASE(FDIV_VL)
  NODE_NAME_CASE(FNEG_VL)
  NODE_NAME_CASE(FABS_VL)
  NODE_NAME_CASE(FSQRT_VL)
  NODE_NAME_CASE(FMA_VL)
  NODE_NAME_CASE(FCOPYSIGN_VL)
  NODE_NAME_CASE(SMIN_VL)
  NODE_NAME_CASE(SMAX_VL)
  NODE_NAME_CASE(UMIN_VL)
  NODE_NAME_CASE(UMAX_VL)
  NODE_NAME_CASE(FMINNUM_VL)
  NODE_NAME_CASE(FMAXNUM_VL)
  NODE_NAME_CASE(MULHS_VL)
  NODE_NAME_CASE(MULHU_VL)
  NODE_NAME_CASE(FP_TO_SINT_VL)
  NODE_NAME_CASE(FP_TO_UINT_VL)
  NODE_NAME_CASE(SINT_TO_FP_VL)
  NODE_NAME_CASE(UINT_TO_FP_VL)
  NODE_NAME_CASE(FP_EXTEND_VL)
  NODE_NAME_CASE(FP_ROUND_VL)
  NODE_NAME_CASE(VWMUL_VL)
  NODE_NAME_CASE(VWMULU_VL)
  NODE_NAME_CASE(VWMULSU_VL)
  NODE_NAME_CASE(VWADD_VL)
  NODE_NAME_CASE(VWADDU_VL)
  NODE_NAME_CASE(VWSUB_VL)
  NODE_NAME_CASE(VWSUBU_VL)
  NODE_NAME_CASE(VWADD_W_VL)
  NODE_NAME_CASE(VWADDU_W_VL)
  NODE_NAME_CASE(VWSUB_W_VL)
  NODE_NAME_CASE(VWSUBU_W_VL)
  NODE_NAME_CASE(SETCC_VL)
  NODE_NAME_CASE(VSELECT_VL)
  NODE_NAME_CASE(VP_MERGE_VL)
  NODE_NAME_CASE(VMAND_VL)
  NODE_NAME_CASE(VMOR_VL)
  NODE_NAME_CASE(VMXOR_VL)
  NODE_NAME_CASE(VMCLR_VL)
  NODE_NAME_CASE(VMSET_VL)
  NODE_NAME_CASE(VRGATHER_VX_VL)
  NODE_NAME_CASE(VRGATHER_VV_VL)
  NODE_NAME_CASE(VRGATHEREI16_VV_VL)
  NODE_NAME_CASE(VSEXT_VL)
  NODE_NAME_CASE(VZEXT_VL)
  NODE_NAME_CASE(VCPOP_VL)
  NODE_NAME_CASE(READ_CSR)
  NODE_NAME_CASE(WRITE_CSR)
  NODE_NAME_CASE(SWAP_CSR)
  }
  // clang-format on
  return nullptr;
#undef NODE_NAME_CASE
}

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
RISCVTargetLowering::ConstraintType
RISCVTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'f':
      return C_RegisterClass;
    case 'I':
    case 'J':
    case 'K':
      return C_Immediate;
    case 'A':
      return C_Memory;
    case 'S': // A symbolic address
      return C_Other;
    }
  } else {
    if (Constraint == "vr" || Constraint == "vm")
      return C_RegisterClass;
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
RISCVTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                  StringRef Constraint,
                                                  MVT VT) const {
  // First, see if this is a constraint that directly corresponds to a
  // RISCV register class.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      // TODO: Support fixed vectors up to XLen for P extension?
      if (VT.isVector())
        break;
      return std::make_pair(0U, &RISCV::GPRRegClass);
    case 'f':
      if (Subtarget.hasStdExtZfh() && VT == MVT::f16)
        return std::make_pair(0U, &RISCV::FPR16RegClass);
      if (Subtarget.hasStdExtF() && VT == MVT::f32)
        return std::make_pair(0U, &RISCV::FPR32RegClass);
      if (Subtarget.hasStdExtD() && VT == MVT::f64)
        return std::make_pair(0U, &RISCV::FPR64RegClass);
      break;
    default:
      break;
    }
  } else if (Constraint == "vr") {
    for (const auto *RC : {&RISCV::VRRegClass, &RISCV::VRM2RegClass,
                           &RISCV::VRM4RegClass, &RISCV::VRM8RegClass}) {
      if (TRI->isTypeLegalForClass(*RC, VT.SimpleTy))
        return std::make_pair(0U, RC);
    }
  } else if (Constraint == "vm") {
    if (TRI->isTypeLegalForClass(RISCV::VMV0RegClass, VT.SimpleTy))
      return std::make_pair(0U, &RISCV::VMV0RegClass);
  }

  // Clang will correctly decode the usage of register name aliases into their
  // official names. However, other frontends like `rustc` do not. This allows
  // users of these frontends to use the ABI names for registers in LLVM-style
  // register constraints.
  unsigned XRegFromAlias = StringSwitch<unsigned>(Constraint.lower())
                               .Case("{zero}", RISCV::X0)
                               .Case("{ra}", RISCV::X1)
                               .Case("{sp}", RISCV::X2)
                               .Case("{gp}", RISCV::X3)
                               .Case("{tp}", RISCV::X4)
                               .Case("{t0}", RISCV::X5)
                               .Case("{t1}", RISCV::X6)
                               .Case("{t2}", RISCV::X7)
                               .Cases("{s0}", "{fp}", RISCV::X8)
                               .Case("{s1}", RISCV::X9)
                               .Case("{a0}", RISCV::X10)
                               .Case("{a1}", RISCV::X11)
                               .Case("{a2}", RISCV::X12)
                               .Case("{a3}", RISCV::X13)
                               .Case("{a4}", RISCV::X14)
                               .Case("{a5}", RISCV::X15)
                               .Case("{a6}", RISCV::X16)
                               .Case("{a7}", RISCV::X17)
                               .Case("{s2}", RISCV::X18)
                               .Case("{s3}", RISCV::X19)
                               .Case("{s4}", RISCV::X20)
                               .Case("{s5}", RISCV::X21)
                               .Case("{s6}", RISCV::X22)
                               .Case("{s7}", RISCV::X23)
                               .Case("{s8}", RISCV::X24)
                               .Case("{s9}", RISCV::X25)
                               .Case("{s10}", RISCV::X26)
                               .Case("{s11}", RISCV::X27)
                               .Case("{t3}", RISCV::X28)
                               .Case("{t4}", RISCV::X29)
                               .Case("{t5}", RISCV::X30)
                               .Case("{t6}", RISCV::X31)
                               .Default(RISCV::NoRegister);
  if (XRegFromAlias != RISCV::NoRegister)
    return std::make_pair(XRegFromAlias, &RISCV::GPRRegClass);

  // Since TargetLowering::getRegForInlineAsmConstraint uses the name of the
  // TableGen record rather than the AsmName to choose registers for InlineAsm
  // constraints, plus we want to match those names to the widest floating point
  // register type available, manually select floating point registers here.
  //
  // The second case is the ABI name of the register, so that frontends can also
  // use the ABI names in register constraint lists.
  if (Subtarget.hasStdExtF()) {
    unsigned FReg = StringSwitch<unsigned>(Constraint.lower())
                        .Cases("{f0}", "{ft0}", RISCV::F0_F)
                        .Cases("{f1}", "{ft1}", RISCV::F1_F)
                        .Cases("{f2}", "{ft2}", RISCV::F2_F)
                        .Cases("{f3}", "{ft3}", RISCV::F3_F)
                        .Cases("{f4}", "{ft4}", RISCV::F4_F)
                        .Cases("{f5}", "{ft5}", RISCV::F5_F)
                        .Cases("{f6}", "{ft6}", RISCV::F6_F)
                        .Cases("{f7}", "{ft7}", RISCV::F7_F)
                        .Cases("{f8}", "{fs0}", RISCV::F8_F)
                        .Cases("{f9}", "{fs1}", RISCV::F9_F)
                        .Cases("{f10}", "{fa0}", RISCV::F10_F)
                        .Cases("{f11}", "{fa1}", RISCV::F11_F)
                        .Cases("{f12}", "{fa2}", RISCV::F12_F)
                        .Cases("{f13}", "{fa3}", RISCV::F13_F)
                        .Cases("{f14}", "{fa4}", RISCV::F14_F)
                        .Cases("{f15}", "{fa5}", RISCV::F15_F)
                        .Cases("{f16}", "{fa6}", RISCV::F16_F)
                        .Cases("{f17}", "{fa7}", RISCV::F17_F)
                        .Cases("{f18}", "{fs2}", RISCV::F18_F)
                        .Cases("{f19}", "{fs3}", RISCV::F19_F)
                        .Cases("{f20}", "{fs4}", RISCV::F20_F)
                        .Cases("{f21}", "{fs5}", RISCV::F21_F)
                        .Cases("{f22}", "{fs6}", RISCV::F22_F)
                        .Cases("{f23}", "{fs7}", RISCV::F23_F)
                        .Cases("{f24}", "{fs8}", RISCV::F24_F)
                        .Cases("{f25}", "{fs9}", RISCV::F25_F)
                        .Cases("{f26}", "{fs10}", RISCV::F26_F)
                        .Cases("{f27}", "{fs11}", RISCV::F27_F)
                        .Cases("{f28}", "{ft8}", RISCV::F28_F)
                        .Cases("{f29}", "{ft9}", RISCV::F29_F)
                        .Cases("{f30}", "{ft10}", RISCV::F30_F)
                        .Cases("{f31}", "{ft11}", RISCV::F31_F)
                        .Default(RISCV::NoRegister);
    if (FReg != RISCV::NoRegister) {
      assert(RISCV::F0_F <= FReg && FReg <= RISCV::F31_F && "Unknown fp-reg");
      if (Subtarget.hasStdExtD() && (VT == MVT::f64 || VT == MVT::Other)) {
        unsigned RegNo = FReg - RISCV::F0_F;
        unsigned DReg = RISCV::F0_D + RegNo;
        return std::make_pair(DReg, &RISCV::FPR64RegClass);
      }
      if (VT == MVT::f32 || VT == MVT::Other)
        return std::make_pair(FReg, &RISCV::FPR32RegClass);
      if (Subtarget.hasStdExtZfh() && VT == MVT::f16) {
        unsigned RegNo = FReg - RISCV::F0_F;
        unsigned HReg = RISCV::F0_H + RegNo;
        return std::make_pair(HReg, &RISCV::FPR16RegClass);
      }
    }
  }

  if (Subtarget.hasVInstructions()) {
    Register VReg = StringSwitch<Register>(Constraint.lower())
                        .Case("{v0}", RISCV::V0)
                        .Case("{v1}", RISCV::V1)
                        .Case("{v2}", RISCV::V2)
                        .Case("{v3}", RISCV::V3)
                        .Case("{v4}", RISCV::V4)
                        .Case("{v5}", RISCV::V5)
                        .Case("{v6}", RISCV::V6)
                        .Case("{v7}", RISCV::V7)
                        .Case("{v8}", RISCV::V8)
                        .Case("{v9}", RISCV::V9)
                        .Case("{v10}", RISCV::V10)
                        .Case("{v11}", RISCV::V11)
                        .Case("{v12}", RISCV::V12)
                        .Case("{v13}", RISCV::V13)
                        .Case("{v14}", RISCV::V14)
                        .Case("{v15}", RISCV::V15)
                        .Case("{v16}", RISCV::V16)
                        .Case("{v17}", RISCV::V17)
                        .Case("{v18}", RISCV::V18)
                        .Case("{v19}", RISCV::V19)
                        .Case("{v20}", RISCV::V20)
                        .Case("{v21}", RISCV::V21)
                        .Case("{v22}", RISCV::V22)
                        .Case("{v23}", RISCV::V23)
                        .Case("{v24}", RISCV::V24)
                        .Case("{v25}", RISCV::V25)
                        .Case("{v26}", RISCV::V26)
                        .Case("{v27}", RISCV::V27)
                        .Case("{v28}", RISCV::V28)
                        .Case("{v29}", RISCV::V29)
                        .Case("{v30}", RISCV::V30)
                        .Case("{v31}", RISCV::V31)
                        .Default(RISCV::NoRegister);
    if (VReg != RISCV::NoRegister) {
      if (TRI->isTypeLegalForClass(RISCV::VMRegClass, VT.SimpleTy))
        return std::make_pair(VReg, &RISCV::VMRegClass);
      if (TRI->isTypeLegalForClass(RISCV::VRRegClass, VT.SimpleTy))
        return std::make_pair(VReg, &RISCV::VRRegClass);
      for (const auto *RC :
           {&RISCV::VRM2RegClass, &RISCV::VRM4RegClass, &RISCV::VRM8RegClass}) {
        if (TRI->isTypeLegalForClass(*RC, VT.SimpleTy)) {
          VReg = TRI->getMatchingSuperReg(VReg, RISCV::sub_vrm1_0, RC);
          return std::make_pair(VReg, RC);
        }
      }
    }
  }

  std::pair<Register, const TargetRegisterClass *> Res =
      TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);

  // If we picked one of the Zfinx register classes, remap it to the GPR class.
  // FIXME: When Zfinx is supported in CodeGen this will need to take the
  // Subtarget into account.
  if (Res.second == &RISCV::GPRF16RegClass ||
      Res.second == &RISCV::GPRF32RegClass ||
      Res.second == &RISCV::GPRF64RegClass)
    return std::make_pair(Res.first, &RISCV::GPRRegClass);

  return Res;
}

unsigned
RISCVTargetLowering::getInlineAsmMemConstraint(StringRef ConstraintCode) const {
  // Currently only support length 1 constraints.
  if (ConstraintCode.size() == 1) {
    switch (ConstraintCode[0]) {
    case 'A':
      return InlineAsm::Constraint_A;
    default:
      break;
    }
  }

  return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
}

void RISCVTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, std::string &Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  // Currently only support length 1 constraints.
  if (Constraint.length() == 1) {
    switch (Constraint[0]) {
    case 'I':
      // Validate & create a 12-bit signed immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getSExtValue();
        if (isInt<12>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    case 'J':
      // Validate & create an integer zero operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (C->getZExtValue() == 0)
          Ops.push_back(
              DAG.getTargetConstant(0, SDLoc(Op), Subtarget.getXLenVT()));
      return;
    case 'K':
      // Validate & create a 5-bit unsigned immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getZExtValue();
        if (isUInt<5>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    case 'S':
      if (const auto *GA = dyn_cast<GlobalAddressSDNode>(Op)) {
        Ops.push_back(DAG.getTargetGlobalAddress(GA->getGlobal(), SDLoc(Op),
                                                 GA->getValueType(0)));
      } else if (const auto *BA = dyn_cast<BlockAddressSDNode>(Op)) {
        Ops.push_back(DAG.getTargetBlockAddress(BA->getBlockAddress(),
                                                BA->getValueType(0)));
      }
      return;
    default:
      break;
    }
  }
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

Instruction *RISCVTargetLowering::emitLeadingFence(IRBuilderBase &Builder,
                                                   Instruction *Inst,
                                                   AtomicOrdering Ord) const {
  if (isa<LoadInst>(Inst) && Ord == AtomicOrdering::SequentiallyConsistent)
    return Builder.CreateFence(Ord);
  if (isa<StoreInst>(Inst) && isReleaseOrStronger(Ord))
    return Builder.CreateFence(AtomicOrdering::Release);
  return nullptr;
}

Instruction *RISCVTargetLowering::emitTrailingFence(IRBuilderBase &Builder,
                                                    Instruction *Inst,
                                                    AtomicOrdering Ord) const {
  if (isa<LoadInst>(Inst) && isAcquireOrStronger(Ord))
    return Builder.CreateFence(AtomicOrdering::Acquire);
  return nullptr;
}

TargetLowering::AtomicExpansionKind
RISCVTargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const {
  // atomicrmw {fadd,fsub} must be expanded to use compare-exchange, as floating
  // point operations can't be used in an lr/sc sequence without breaking the
  // forward-progress guarantee.
  if (AI->isFloatingPointOperation())
    return AtomicExpansionKind::CmpXChg;

  unsigned Size = AI->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

static Intrinsic::ID
getIntrinsicForMaskedAtomicRMWBinOp(unsigned XLen, AtomicRMWInst::BinOp BinOp) {
  if (XLen == 32) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::riscv_masked_atomicrmw_xchg_i32;
    case AtomicRMWInst::Add:
      return Intrinsic::riscv_masked_atomicrmw_add_i32;
    case AtomicRMWInst::Sub:
      return Intrinsic::riscv_masked_atomicrmw_sub_i32;
    case AtomicRMWInst::Nand:
      return Intrinsic::riscv_masked_atomicrmw_nand_i32;
    case AtomicRMWInst::Max:
      return Intrinsic::riscv_masked_atomicrmw_max_i32;
    case AtomicRMWInst::Min:
      return Intrinsic::riscv_masked_atomicrmw_min_i32;
    case AtomicRMWInst::UMax:
      return Intrinsic::riscv_masked_atomicrmw_umax_i32;
    case AtomicRMWInst::UMin:
      return Intrinsic::riscv_masked_atomicrmw_umin_i32;
    }
  }

  if (XLen == 64) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::riscv_masked_atomicrmw_xchg_i64;
    case AtomicRMWInst::Add:
      return Intrinsic::riscv_masked_atomicrmw_add_i64;
    case AtomicRMWInst::Sub:
      return Intrinsic::riscv_masked_atomicrmw_sub_i64;
    case AtomicRMWInst::Nand:
      return Intrinsic::riscv_masked_atomicrmw_nand_i64;
    case AtomicRMWInst::Max:
      return Intrinsic::riscv_masked_atomicrmw_max_i64;
    case AtomicRMWInst::Min:
      return Intrinsic::riscv_masked_atomicrmw_min_i64;
    case AtomicRMWInst::UMax:
      return Intrinsic::riscv_masked_atomicrmw_umax_i64;
    case AtomicRMWInst::UMin:
      return Intrinsic::riscv_masked_atomicrmw_umin_i64;
    }
  }

  llvm_unreachable("Unexpected XLen\n");
}

Value *RISCVTargetLowering::emitMaskedAtomicRMWIntrinsic(
    IRBuilderBase &Builder, AtomicRMWInst *AI, Value *AlignedAddr, Value *Incr,
    Value *Mask, Value *ShiftAmt, AtomicOrdering Ord) const {
  unsigned XLen = Subtarget.getXLen();
  Value *Ordering =
      Builder.getIntN(XLen, static_cast<uint64_t>(AI->getOrdering()));
  Type *Tys[] = {AlignedAddr->getType()};
  Function *LrwOpScwLoop = Intrinsic::getDeclaration(
      AI->getModule(),
      getIntrinsicForMaskedAtomicRMWBinOp(XLen, AI->getOperation()), Tys);

  if (XLen == 64) {
    Incr = Builder.CreateSExt(Incr, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    ShiftAmt = Builder.CreateSExt(ShiftAmt, Builder.getInt64Ty());
  }

  Value *Result;

  // Must pass the shift amount needed to sign extend the loaded value prior
  // to performing a signed comparison for min/max. ShiftAmt is the number of
  // bits to shift the value into position. Pass XLen-ShiftAmt-ValWidth, which
  // is the number of bits to left+right shift the value in order to
  // sign-extend.
  if (AI->getOperation() == AtomicRMWInst::Min ||
      AI->getOperation() == AtomicRMWInst::Max) {
    const DataLayout &DL = AI->getModule()->getDataLayout();
    unsigned ValWidth =
        DL.getTypeStoreSizeInBits(AI->getValOperand()->getType());
    Value *SextShamt =
        Builder.CreateSub(Builder.getIntN(XLen, XLen - ValWidth), ShiftAmt);
    Result = Builder.CreateCall(LrwOpScwLoop,
                                {AlignedAddr, Incr, Mask, SextShamt, Ordering});
  } else {
    Result =
        Builder.CreateCall(LrwOpScwLoop, {AlignedAddr, Incr, Mask, Ordering});
  }

  if (XLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

TargetLowering::AtomicExpansionKind
RISCVTargetLowering::shouldExpandAtomicCmpXchgInIR(
    AtomicCmpXchgInst *CI) const {
  unsigned Size = CI->getCompareOperand()->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

Value *RISCVTargetLowering::emitMaskedAtomicCmpXchgIntrinsic(
    IRBuilderBase &Builder, AtomicCmpXchgInst *CI, Value *AlignedAddr,
    Value *CmpVal, Value *NewVal, Value *Mask, AtomicOrdering Ord) const {
  unsigned XLen = Subtarget.getXLen();
  Value *Ordering = Builder.getIntN(XLen, static_cast<uint64_t>(Ord));
  Intrinsic::ID CmpXchgIntrID = Intrinsic::riscv_masked_cmpxchg_i32;
  if (XLen == 64) {
    CmpVal = Builder.CreateSExt(CmpVal, Builder.getInt64Ty());
    NewVal = Builder.CreateSExt(NewVal, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    CmpXchgIntrID = Intrinsic::riscv_masked_cmpxchg_i64;
  }
  Type *Tys[] = {AlignedAddr->getType()};
  Function *MaskedCmpXchg =
      Intrinsic::getDeclaration(CI->getModule(), CmpXchgIntrID, Tys);
  Value *Result = Builder.CreateCall(
      MaskedCmpXchg, {AlignedAddr, CmpVal, NewVal, Mask, Ordering});
  if (XLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

bool RISCVTargetLowering::shouldRemoveExtendFromGSIndex(EVT VT) const {
  return false;
}

bool RISCVTargetLowering::shouldConvertFpToSat(unsigned Op, EVT FPVT,
                                               EVT VT) const {
  if (!isOperationLegalOrCustom(Op, VT) || !FPVT.isSimple())
    return false;

  switch (FPVT.getSimpleVT().SimpleTy) {
  case MVT::f16:
    return Subtarget.hasStdExtZfh();
  case MVT::f32:
    return Subtarget.hasStdExtF();
  case MVT::f64:
    return Subtarget.hasStdExtD();
  default:
    return false;
  }
}

unsigned RISCVTargetLowering::getJumpTableEncoding() const {
  // If we are using the small code model, we can reduce size of jump table
  // entry to 4 bytes.
  if (Subtarget.is64Bit() && !isPositionIndependent() &&
      getTargetMachine().getCodeModel() == CodeModel::Small) {
    return MachineJumpTableInfo::EK_Custom32;
  }
  return TargetLowering::getJumpTableEncoding();
}

const MCExpr *RISCVTargetLowering::LowerCustomJumpTableEntry(
    const MachineJumpTableInfo *MJTI, const MachineBasicBlock *MBB,
    unsigned uid, MCContext &Ctx) const {
  assert(Subtarget.is64Bit() && !isPositionIndependent() &&
         getTargetMachine().getCodeModel() == CodeModel::Small);
  return MCSymbolRefExpr::create(MBB->getSymbol(), Ctx);
}

bool RISCVTargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                     EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f16:
    return Subtarget.hasStdExtZfh();
  case MVT::f32:
    return Subtarget.hasStdExtF();
  case MVT::f64:
    return Subtarget.hasStdExtD();
  default:
    break;
  }

  return false;
}

Register RISCVTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return RISCV::X10;
}

Register RISCVTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return RISCV::X11;
}

bool RISCVTargetLowering::shouldExtendTypeInLibCall(EVT Type) const {
  // Return false to suppress the unnecessary extensions if the LibCall
  // arguments or return value is f32 type for LP64 ABI.
  RISCVABI::ABI ABI = Subtarget.getTargetABI();
  if (ABI == RISCVABI::ABI_LP64 && (Type == MVT::f32))
    return false;

  return true;
}

bool RISCVTargetLowering::shouldSignExtendTypeInLibCall(EVT Type, bool IsSigned) const {
  if (Subtarget.is64Bit() && Type == MVT::i32)
    return true;

  return IsSigned;
}

bool RISCVTargetLowering::decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                                 SDValue C) const {
  // Check integral scalar types.
  if (VT.isScalarInteger()) {
    // Omit the optimization if the sub target has the M extension and the data
    // size exceeds XLen.
    if (Subtarget.hasStdExtM() && VT.getSizeInBits() > Subtarget.getXLen())
      return false;
    if (auto *ConstNode = dyn_cast<ConstantSDNode>(C.getNode())) {
      // Break the MUL to a SLLI and an ADD/SUB.
      const APInt &Imm = ConstNode->getAPIntValue();
      if ((Imm + 1).isPowerOf2() || (Imm - 1).isPowerOf2() ||
          (1 - Imm).isPowerOf2() || (-1 - Imm).isPowerOf2())
        return true;
      // Optimize the MUL to (SH*ADD x, (SLLI x, bits)) if Imm is not simm12.
      if (Subtarget.hasStdExtZba() && !Imm.isSignedIntN(12) &&
          ((Imm - 2).isPowerOf2() || (Imm - 4).isPowerOf2() ||
           (Imm - 8).isPowerOf2()))
        return true;
      // Omit the following optimization if the sub target has the M extension
      // and the data size >= XLen.
      if (Subtarget.hasStdExtM() && VT.getSizeInBits() >= Subtarget.getXLen())
        return false;
      // Break the MUL to two SLLI instructions and an ADD/SUB, if Imm needs
      // a pair of LUI/ADDI.
      if (!Imm.isSignedIntN(12) && Imm.countTrailingZeros() < 12) {
        APInt ImmS = Imm.ashr(Imm.countTrailingZeros());
        if ((ImmS + 1).isPowerOf2() || (ImmS - 1).isPowerOf2() ||
            (1 - ImmS).isPowerOf2())
        return true;
      }
    }
  }

  return false;
}

bool RISCVTargetLowering::isMulAddWithConstProfitable(SDValue AddNode,
                                                      SDValue ConstNode) const {
  // Let the DAGCombiner decide for vectors.
  EVT VT = AddNode.getValueType();
  if (VT.isVector())
    return true;

  // Let the DAGCombiner decide for larger types.
  if (VT.getScalarSizeInBits() > Subtarget.getXLen())
    return true;

  // It is worse if c1 is simm12 while c1*c2 is not.
  ConstantSDNode *C1Node = cast<ConstantSDNode>(AddNode.getOperand(1));
  ConstantSDNode *C2Node = cast<ConstantSDNode>(ConstNode);
  const APInt &C1 = C1Node->getAPIntValue();
  const APInt &C2 = C2Node->getAPIntValue();
  if (C1.isSignedIntN(12) && !(C1 * C2).isSignedIntN(12))
    return false;

  // Default to true and let the DAGCombiner decide.
  return true;
}

bool RISCVTargetLowering::allowsMisalignedMemoryAccesses(
    EVT VT, unsigned AddrSpace, Align Alignment, MachineMemOperand::Flags Flags,
    bool *Fast) const {
  if (!VT.isVector())
    return false;

  EVT ElemVT = VT.getVectorElementType();
  if (Alignment >= ElemVT.getStoreSize()) {
    if (Fast)
      *Fast = true;
    return true;
  }

  return false;
}

bool RISCVTargetLowering::splitValueIntoRegisterParts(
    SelectionDAG &DAG, const SDLoc &DL, SDValue Val, SDValue *Parts,
    unsigned NumParts, MVT PartVT, Optional<CallingConv::ID> CC) const {
  bool IsABIRegCopy = CC.hasValue();
  EVT ValueVT = Val.getValueType();
  if (IsABIRegCopy && ValueVT == MVT::f16 && PartVT == MVT::f32) {
    // Cast the f16 to i16, extend to i32, pad with ones to make a float nan,
    // and cast to f32.
    Val = DAG.getNode(ISD::BITCAST, DL, MVT::i16, Val);
    Val = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Val);
    Val = DAG.getNode(ISD::OR, DL, MVT::i32, Val,
                      DAG.getConstant(0xFFFF0000, DL, MVT::i32));
    Val = DAG.getNode(ISD::BITCAST, DL, MVT::f32, Val);
    Parts[0] = Val;
    return true;
  }

  if (ValueVT.isScalableVector() && PartVT.isScalableVector()) {
    LLVMContext &Context = *DAG.getContext();
    EVT ValueEltVT = ValueVT.getVectorElementType();
    EVT PartEltVT = PartVT.getVectorElementType();
    unsigned ValueVTBitSize = ValueVT.getSizeInBits().getKnownMinSize();
    unsigned PartVTBitSize = PartVT.getSizeInBits().getKnownMinSize();
    if (PartVTBitSize % ValueVTBitSize == 0) {
      assert(PartVTBitSize >= ValueVTBitSize);
      // If the element types are different, bitcast to the same element type of
      // PartVT first.
      // Give an example here, we want copy a <vscale x 1 x i8> value to
      // <vscale x 4 x i16>.
      // We need to convert <vscale x 1 x i8> to <vscale x 8 x i8> by insert
      // subvector, then we can bitcast to <vscale x 4 x i16>.
      if (ValueEltVT != PartEltVT) {
        if (PartVTBitSize > ValueVTBitSize) {
          unsigned Count = PartVTBitSize / ValueEltVT.getFixedSizeInBits();
          assert(Count != 0 && "The number of element should not be zero.");
          EVT SameEltTypeVT =
              EVT::getVectorVT(Context, ValueEltVT, Count, /*IsScalable=*/true);
          Val = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, SameEltTypeVT,
                            DAG.getUNDEF(SameEltTypeVT), Val,
                            DAG.getVectorIdxConstant(0, DL));
        }
        Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
      } else {
        Val =
            DAG.getNode(ISD::INSERT_SUBVECTOR, DL, PartVT, DAG.getUNDEF(PartVT),
                        Val, DAG.getVectorIdxConstant(0, DL));
      }
      Parts[0] = Val;
      return true;
    }
  }
  return false;
}

SDValue RISCVTargetLowering::joinRegisterPartsIntoValue(
    SelectionDAG &DAG, const SDLoc &DL, const SDValue *Parts, unsigned NumParts,
    MVT PartVT, EVT ValueVT, Optional<CallingConv::ID> CC) const {
  bool IsABIRegCopy = CC.hasValue();
  if (IsABIRegCopy && ValueVT == MVT::f16 && PartVT == MVT::f32) {
    SDValue Val = Parts[0];

    // Cast the f32 to i32, truncate to i16, and cast back to f16.
    Val = DAG.getNode(ISD::BITCAST, DL, MVT::i32, Val);
    Val = DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, Val);
    Val = DAG.getNode(ISD::BITCAST, DL, MVT::f16, Val);
    return Val;
  }

  if (ValueVT.isScalableVector() && PartVT.isScalableVector()) {
    LLVMContext &Context = *DAG.getContext();
    SDValue Val = Parts[0];
    EVT ValueEltVT = ValueVT.getVectorElementType();
    EVT PartEltVT = PartVT.getVectorElementType();
    unsigned ValueVTBitSize = ValueVT.getSizeInBits().getKnownMinSize();
    unsigned PartVTBitSize = PartVT.getSizeInBits().getKnownMinSize();
    if (PartVTBitSize % ValueVTBitSize == 0) {
      assert(PartVTBitSize >= ValueVTBitSize);
      EVT SameEltTypeVT = ValueVT;
      // If the element types are different, convert it to the same element type
      // of PartVT.
      // Give an example here, we want copy a <vscale x 1 x i8> value from
      // <vscale x 4 x i16>.
      // We need to convert <vscale x 4 x i16> to <vscale x 8 x i8> first,
      // then we can extract <vscale x 1 x i8>.
      if (ValueEltVT != PartEltVT) {
        unsigned Count = PartVTBitSize / ValueEltVT.getFixedSizeInBits();
        assert(Count != 0 && "The number of element should not be zero.");
        SameEltTypeVT =
            EVT::getVectorVT(Context, ValueEltVT, Count, /*IsScalable=*/true);
        Val = DAG.getNode(ISD::BITCAST, DL, SameEltTypeVT, Val);
      }
      Val = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, ValueVT, Val,
                        DAG.getVectorIdxConstant(0, DL));
      return Val;
    }
  }
  return SDValue();
}

SDValue
RISCVTargetLowering::BuildSDIVPow2(SDNode *N, const APInt &Divisor,
                                   SelectionDAG &DAG,
                                   SmallVectorImpl<SDNode *> &Created) const {
  AttributeList Attr = DAG.getMachineFunction().getFunction().getAttributes();
  if (isIntDivCheap(N->getValueType(0), Attr))
    return SDValue(N, 0); // Lower SDIV as SDIV

  assert((Divisor.isPowerOf2() || Divisor.isNegatedPowerOf2()) &&
         "Unexpected divisor!");

  // Conditional move is needed, so do the transformation iff Zbt is enabled.
  if (!Subtarget.hasStdExtZbt())
    return SDValue();

  // When |Divisor| >= 2 ^ 12, it isn't profitable to do such transformation.
  // Besides, more critical path instructions will be generated when dividing
  // by 2. So we keep using the original DAGs for these cases.
  unsigned Lg2 = Divisor.countTrailingZeros();
  if (Lg2 == 1 || Lg2 >= 12)
    return SDValue();

  // fold (sdiv X, pow2)
  EVT VT = N->getValueType(0);
  if (VT != MVT::i32 && !(Subtarget.is64Bit() && VT == MVT::i64))
    return SDValue();

  SDLoc DL(N);
  SDValue N0 = N->getOperand(0);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue Pow2MinusOne = DAG.getConstant((1ULL << Lg2) - 1, DL, VT);

  // Add (N0 < 0) ? Pow2 - 1 : 0;
  SDValue Cmp = DAG.getSetCC(DL, VT, N0, Zero, ISD::SETLT);
  SDValue Add = DAG.getNode(ISD::ADD, DL, VT, N0, Pow2MinusOne);
  SDValue Sel = DAG.getNode(ISD::SELECT, DL, VT, Cmp, Add, N0);

  Created.push_back(Cmp.getNode());
  Created.push_back(Add.getNode());
  Created.push_back(Sel.getNode());

  // Divide by pow2.
  SDValue SRA =
      DAG.getNode(ISD::SRA, DL, VT, Sel, DAG.getConstant(Lg2, DL, VT));

  // If we're dividing by a positive value, we're done.  Otherwise, we must
  // negate the result.
  if (Divisor.isNonNegative())
    return SRA;

  Created.push_back(SRA.getNode());
  return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT), SRA);
}

#define GET_REGISTER_MATCHER
#include "RISCVGenAsmMatcher.inc"

Register
RISCVTargetLowering::getRegisterByName(const char *RegName, LLT VT,
                                       const MachineFunction &MF) const {
  Register Reg = MatchRegisterAltName(RegName);
  if (Reg == RISCV::NoRegister)
    Reg = MatchRegisterName(RegName);
  if (Reg == RISCV::NoRegister)
    report_fatal_error(
        Twine("Invalid register name \"" + StringRef(RegName) + "\"."));
  BitVector ReservedRegs = Subtarget.getRegisterInfo()->getReservedRegs(MF);
  if (!ReservedRegs.test(Reg) && !Subtarget.isRegisterReservedByUser(Reg))
    report_fatal_error(Twine("Trying to obtain non-reserved register \"" +
                             StringRef(RegName) + "\"."));
  return Reg;
}

namespace llvm {
namespace RISCVVIntrinsicsTable {

#define GET_RISCVVIntrinsicsTable_IMPL
#include "RISCVGenSearchableTables.inc"

} // namespace RISCVVIntrinsicsTable

} // namespace llvm
