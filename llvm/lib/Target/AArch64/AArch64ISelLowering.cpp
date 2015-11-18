//===-- AArch64ISelLowering.cpp - AArch64 DAG Lowering Implementation  ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64TargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "AArch64ISelLowering.h"
#include "AArch64CallingConvention.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64PerfectShuffle.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "AArch64TargetObjectFile.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

#define DEBUG_TYPE "aarch64-lower"

STATISTIC(NumTailCalls, "Number of tail calls");
STATISTIC(NumShiftInserts, "Number of vector shift inserts");

// Place holder until extr generation is tested fully.
static cl::opt<bool>
EnableAArch64ExtrGeneration("aarch64-extr-generation", cl::Hidden,
                          cl::desc("Allow AArch64 (or (shift)(shift))->extract"),
                          cl::init(true));

static cl::opt<bool>
EnableAArch64SlrGeneration("aarch64-shift-insert-generation", cl::Hidden,
                           cl::desc("Allow AArch64 SLI/SRI formation"),
                           cl::init(false));

// FIXME: The necessary dtprel relocations don't seem to be supported
// well in the GNU bfd and gold linkers at the moment. Therefore, by
// default, for now, fall back to GeneralDynamic code generation.
cl::opt<bool> EnableAArch64ELFLocalDynamicTLSGeneration(
    "aarch64-elf-ldtls-generation", cl::Hidden,
    cl::desc("Allow AArch64 Local Dynamic TLS code generation"),
    cl::init(false));

/// Value type used for condition codes.
static const MVT MVT_CC = MVT::i32;

AArch64TargetLowering::AArch64TargetLowering(const TargetMachine &TM,
                                             const AArch64Subtarget &STI)
    : TargetLowering(TM), Subtarget(&STI) {

  // AArch64 doesn't have comparisons which set GPRs or setcc instructions, so
  // we have to make something up. Arbitrarily, choose ZeroOrOne.
  setBooleanContents(ZeroOrOneBooleanContent);
  // When comparing vectors the result sets the different elements in the
  // vector to all-one or all-zero.
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);

  // Set up the register classes.
  addRegisterClass(MVT::i32, &AArch64::GPR32allRegClass);
  addRegisterClass(MVT::i64, &AArch64::GPR64allRegClass);

  if (Subtarget->hasFPARMv8()) {
    addRegisterClass(MVT::f16, &AArch64::FPR16RegClass);
    addRegisterClass(MVT::f32, &AArch64::FPR32RegClass);
    addRegisterClass(MVT::f64, &AArch64::FPR64RegClass);
    addRegisterClass(MVT::f128, &AArch64::FPR128RegClass);
  }

  if (Subtarget->hasNEON()) {
    addRegisterClass(MVT::v16i8, &AArch64::FPR8RegClass);
    addRegisterClass(MVT::v8i16, &AArch64::FPR16RegClass);
    // Someone set us up the NEON.
    addDRTypeForNEON(MVT::v2f32);
    addDRTypeForNEON(MVT::v8i8);
    addDRTypeForNEON(MVT::v4i16);
    addDRTypeForNEON(MVT::v2i32);
    addDRTypeForNEON(MVT::v1i64);
    addDRTypeForNEON(MVT::v1f64);
    addDRTypeForNEON(MVT::v4f16);

    addQRTypeForNEON(MVT::v4f32);
    addQRTypeForNEON(MVT::v2f64);
    addQRTypeForNEON(MVT::v16i8);
    addQRTypeForNEON(MVT::v8i16);
    addQRTypeForNEON(MVT::v4i32);
    addQRTypeForNEON(MVT::v2i64);
    addQRTypeForNEON(MVT::v8f16);
  }

  // Compute derived properties from the register classes
  computeRegisterProperties(Subtarget->getRegisterInfo());

  // Provide all sorts of operation actions
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i64, Custom);
  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  setOperationAction(ISD::SETCC, MVT::i64, Custom);
  setOperationAction(ISD::SETCC, MVT::f32, Custom);
  setOperationAction(ISD::SETCC, MVT::f64, Custom);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::i64, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f64, Custom);
  setOperationAction(ISD::SELECT, MVT::i32, Custom);
  setOperationAction(ISD::SELECT, MVT::i64, Custom);
  setOperationAction(ISD::SELECT, MVT::f32, Custom);
  setOperationAction(ISD::SELECT, MVT::f64, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::JumpTable, MVT::i64, Custom);

  setOperationAction(ISD::SHL_PARTS, MVT::i64, Custom);
  setOperationAction(ISD::SRA_PARTS, MVT::i64, Custom);
  setOperationAction(ISD::SRL_PARTS, MVT::i64, Custom);

  setOperationAction(ISD::FREM, MVT::f32, Expand);
  setOperationAction(ISD::FREM, MVT::f64, Expand);
  setOperationAction(ISD::FREM, MVT::f80, Expand);

  // Custom lowering hooks are needed for XOR
  // to fold it into CSINC/CSINV.
  setOperationAction(ISD::XOR, MVT::i32, Custom);
  setOperationAction(ISD::XOR, MVT::i64, Custom);

  // Virtually no operation on f128 is legal, but LLVM can't expand them when
  // there's a valid register class, so we need custom operations in most cases.
  setOperationAction(ISD::FABS, MVT::f128, Expand);
  setOperationAction(ISD::FADD, MVT::f128, Custom);
  setOperationAction(ISD::FCOPYSIGN, MVT::f128, Expand);
  setOperationAction(ISD::FCOS, MVT::f128, Expand);
  setOperationAction(ISD::FDIV, MVT::f128, Custom);
  setOperationAction(ISD::FMA, MVT::f128, Expand);
  setOperationAction(ISD::FMUL, MVT::f128, Custom);
  setOperationAction(ISD::FNEG, MVT::f128, Expand);
  setOperationAction(ISD::FPOW, MVT::f128, Expand);
  setOperationAction(ISD::FREM, MVT::f128, Expand);
  setOperationAction(ISD::FRINT, MVT::f128, Expand);
  setOperationAction(ISD::FSIN, MVT::f128, Expand);
  setOperationAction(ISD::FSINCOS, MVT::f128, Expand);
  setOperationAction(ISD::FSQRT, MVT::f128, Expand);
  setOperationAction(ISD::FSUB, MVT::f128, Custom);
  setOperationAction(ISD::FTRUNC, MVT::f128, Expand);
  setOperationAction(ISD::SETCC, MVT::f128, Custom);
  setOperationAction(ISD::BR_CC, MVT::f128, Custom);
  setOperationAction(ISD::SELECT, MVT::f128, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f128, Custom);
  setOperationAction(ISD::FP_EXTEND, MVT::f128, Custom);

  // Lowering for many of the conversions is actually specified by the non-f128
  // type. The LowerXXX function will be trivial when f128 isn't involved.
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i128, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i128, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i128, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i128, Custom);
  setOperationAction(ISD::FP_ROUND, MVT::f32, Custom);
  setOperationAction(ISD::FP_ROUND, MVT::f64, Custom);

  // Variable arguments.
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  setOperationAction(ISD::VACOPY, MVT::Other, Custom);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  // Variable-sized objects.
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);

  // Constant pool entries
  setOperationAction(ISD::ConstantPool, MVT::i64, Custom);

  // BlockAddress
  setOperationAction(ISD::BlockAddress, MVT::i64, Custom);

  // Add/Sub overflow ops with MVT::Glues are lowered to NZCV dependences.
  setOperationAction(ISD::ADDC, MVT::i32, Custom);
  setOperationAction(ISD::ADDE, MVT::i32, Custom);
  setOperationAction(ISD::SUBC, MVT::i32, Custom);
  setOperationAction(ISD::SUBE, MVT::i32, Custom);
  setOperationAction(ISD::ADDC, MVT::i64, Custom);
  setOperationAction(ISD::ADDE, MVT::i64, Custom);
  setOperationAction(ISD::SUBC, MVT::i64, Custom);
  setOperationAction(ISD::SUBE, MVT::i64, Custom);

  // AArch64 lacks both left-rotate and popcount instructions.
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i64, Expand);
  for (MVT VT : MVT::vector_valuetypes()) {
    setOperationAction(ISD::ROTL, VT, Expand);
    setOperationAction(ISD::ROTR, VT, Expand);
  }

  // AArch64 doesn't have {U|S}MUL_LOHI.
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);


  // Expand the undefined-at-zero variants to cttz/ctlz to their defined-at-zero
  // counterparts, which AArch64 supports directly.
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i64, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i64, Expand);

  setOperationAction(ISD::CTPOP, MVT::i32, Custom);
  setOperationAction(ISD::CTPOP, MVT::i64, Custom);

  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i64, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i64, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);

  // Custom lower Add/Sub/Mul with overflow.
  setOperationAction(ISD::SADDO, MVT::i32, Custom);
  setOperationAction(ISD::SADDO, MVT::i64, Custom);
  setOperationAction(ISD::UADDO, MVT::i32, Custom);
  setOperationAction(ISD::UADDO, MVT::i64, Custom);
  setOperationAction(ISD::SSUBO, MVT::i32, Custom);
  setOperationAction(ISD::SSUBO, MVT::i64, Custom);
  setOperationAction(ISD::USUBO, MVT::i32, Custom);
  setOperationAction(ISD::USUBO, MVT::i64, Custom);
  setOperationAction(ISD::SMULO, MVT::i32, Custom);
  setOperationAction(ISD::SMULO, MVT::i64, Custom);
  setOperationAction(ISD::UMULO, MVT::i32, Custom);
  setOperationAction(ISD::UMULO, MVT::i64, Custom);

  setOperationAction(ISD::FSIN, MVT::f32, Expand);
  setOperationAction(ISD::FSIN, MVT::f64, Expand);
  setOperationAction(ISD::FCOS, MVT::f32, Expand);
  setOperationAction(ISD::FCOS, MVT::f64, Expand);
  setOperationAction(ISD::FPOW, MVT::f32, Expand);
  setOperationAction(ISD::FPOW, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);

  // f16 is a storage-only type, always promote it to f32.
  setOperationAction(ISD::SETCC,       MVT::f16,  Promote);
  setOperationAction(ISD::BR_CC,       MVT::f16,  Promote);
  setOperationAction(ISD::SELECT_CC,   MVT::f16,  Promote);
  setOperationAction(ISD::SELECT,      MVT::f16,  Promote);
  setOperationAction(ISD::FADD,        MVT::f16,  Promote);
  setOperationAction(ISD::FSUB,        MVT::f16,  Promote);
  setOperationAction(ISD::FMUL,        MVT::f16,  Promote);
  setOperationAction(ISD::FDIV,        MVT::f16,  Promote);
  setOperationAction(ISD::FREM,        MVT::f16,  Promote);
  setOperationAction(ISD::FMA,         MVT::f16,  Promote);
  setOperationAction(ISD::FNEG,        MVT::f16,  Promote);
  setOperationAction(ISD::FABS,        MVT::f16,  Promote);
  setOperationAction(ISD::FCEIL,       MVT::f16,  Promote);
  setOperationAction(ISD::FCOPYSIGN,   MVT::f16,  Promote);
  setOperationAction(ISD::FCOS,        MVT::f16,  Promote);
  setOperationAction(ISD::FFLOOR,      MVT::f16,  Promote);
  setOperationAction(ISD::FNEARBYINT,  MVT::f16,  Promote);
  setOperationAction(ISD::FPOW,        MVT::f16,  Promote);
  setOperationAction(ISD::FPOWI,       MVT::f16,  Promote);
  setOperationAction(ISD::FRINT,       MVT::f16,  Promote);
  setOperationAction(ISD::FSIN,        MVT::f16,  Promote);
  setOperationAction(ISD::FSINCOS,     MVT::f16,  Promote);
  setOperationAction(ISD::FSQRT,       MVT::f16,  Promote);
  setOperationAction(ISD::FEXP,        MVT::f16,  Promote);
  setOperationAction(ISD::FEXP2,       MVT::f16,  Promote);
  setOperationAction(ISD::FLOG,        MVT::f16,  Promote);
  setOperationAction(ISD::FLOG2,       MVT::f16,  Promote);
  setOperationAction(ISD::FLOG10,      MVT::f16,  Promote);
  setOperationAction(ISD::FROUND,      MVT::f16,  Promote);
  setOperationAction(ISD::FTRUNC,      MVT::f16,  Promote);
  setOperationAction(ISD::FMINNUM,     MVT::f16,  Promote);
  setOperationAction(ISD::FMAXNUM,     MVT::f16,  Promote);
  setOperationAction(ISD::FMINNAN,     MVT::f16,  Promote);
  setOperationAction(ISD::FMAXNAN,     MVT::f16,  Promote);

  // v4f16 is also a storage-only type, so promote it to v4f32 when that is
  // known to be safe.
  setOperationAction(ISD::FADD, MVT::v4f16, Promote);
  setOperationAction(ISD::FSUB, MVT::v4f16, Promote);
  setOperationAction(ISD::FMUL, MVT::v4f16, Promote);
  setOperationAction(ISD::FDIV, MVT::v4f16, Promote);
  setOperationAction(ISD::FP_EXTEND, MVT::v4f16, Promote);
  setOperationAction(ISD::FP_ROUND, MVT::v4f16, Promote);
  AddPromotedToType(ISD::FADD, MVT::v4f16, MVT::v4f32);
  AddPromotedToType(ISD::FSUB, MVT::v4f16, MVT::v4f32);
  AddPromotedToType(ISD::FMUL, MVT::v4f16, MVT::v4f32);
  AddPromotedToType(ISD::FDIV, MVT::v4f16, MVT::v4f32);
  AddPromotedToType(ISD::FP_EXTEND, MVT::v4f16, MVT::v4f32);
  AddPromotedToType(ISD::FP_ROUND, MVT::v4f16, MVT::v4f32);

  // Expand all other v4f16 operations.
  // FIXME: We could generate better code by promoting some operations to
  // a pair of v4f32s
  setOperationAction(ISD::FABS, MVT::v4f16, Expand);
  setOperationAction(ISD::FCEIL, MVT::v4f16, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::v4f16, Expand);
  setOperationAction(ISD::FCOS, MVT::v4f16, Expand);
  setOperationAction(ISD::FFLOOR, MVT::v4f16, Expand);
  setOperationAction(ISD::FMA, MVT::v4f16, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::v4f16, Expand);
  setOperationAction(ISD::FNEG, MVT::v4f16, Expand);
  setOperationAction(ISD::FPOW, MVT::v4f16, Expand);
  setOperationAction(ISD::FPOWI, MVT::v4f16, Expand);
  setOperationAction(ISD::FREM, MVT::v4f16, Expand);
  setOperationAction(ISD::FROUND, MVT::v4f16, Expand);
  setOperationAction(ISD::FRINT, MVT::v4f16, Expand);
  setOperationAction(ISD::FSIN, MVT::v4f16, Expand);
  setOperationAction(ISD::FSINCOS, MVT::v4f16, Expand);
  setOperationAction(ISD::FSQRT, MVT::v4f16, Expand);
  setOperationAction(ISD::FTRUNC, MVT::v4f16, Expand);
  setOperationAction(ISD::SETCC, MVT::v4f16, Expand);
  setOperationAction(ISD::BR_CC, MVT::v4f16, Expand);
  setOperationAction(ISD::SELECT, MVT::v4f16, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::v4f16, Expand);
  setOperationAction(ISD::FEXP, MVT::v4f16, Expand);
  setOperationAction(ISD::FEXP2, MVT::v4f16, Expand);
  setOperationAction(ISD::FLOG, MVT::v4f16, Expand);
  setOperationAction(ISD::FLOG2, MVT::v4f16, Expand);
  setOperationAction(ISD::FLOG10, MVT::v4f16, Expand);


  // v8f16 is also a storage-only type, so expand it.
  setOperationAction(ISD::FABS, MVT::v8f16, Expand);
  setOperationAction(ISD::FADD, MVT::v8f16, Expand);
  setOperationAction(ISD::FCEIL, MVT::v8f16, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::v8f16, Expand);
  setOperationAction(ISD::FCOS, MVT::v8f16, Expand);
  setOperationAction(ISD::FDIV, MVT::v8f16, Expand);
  setOperationAction(ISD::FFLOOR, MVT::v8f16, Expand);
  setOperationAction(ISD::FMA, MVT::v8f16, Expand);
  setOperationAction(ISD::FMUL, MVT::v8f16, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::v8f16, Expand);
  setOperationAction(ISD::FNEG, MVT::v8f16, Expand);
  setOperationAction(ISD::FPOW, MVT::v8f16, Expand);
  setOperationAction(ISD::FPOWI, MVT::v8f16, Expand);
  setOperationAction(ISD::FREM, MVT::v8f16, Expand);
  setOperationAction(ISD::FROUND, MVT::v8f16, Expand);
  setOperationAction(ISD::FRINT, MVT::v8f16, Expand);
  setOperationAction(ISD::FSIN, MVT::v8f16, Expand);
  setOperationAction(ISD::FSINCOS, MVT::v8f16, Expand);
  setOperationAction(ISD::FSQRT, MVT::v8f16, Expand);
  setOperationAction(ISD::FSUB, MVT::v8f16, Expand);
  setOperationAction(ISD::FTRUNC, MVT::v8f16, Expand);
  setOperationAction(ISD::SETCC, MVT::v8f16, Expand);
  setOperationAction(ISD::BR_CC, MVT::v8f16, Expand);
  setOperationAction(ISD::SELECT, MVT::v8f16, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::v8f16, Expand);
  setOperationAction(ISD::FP_EXTEND, MVT::v8f16, Expand);
  setOperationAction(ISD::FEXP, MVT::v8f16, Expand);
  setOperationAction(ISD::FEXP2, MVT::v8f16, Expand);
  setOperationAction(ISD::FLOG, MVT::v8f16, Expand);
  setOperationAction(ISD::FLOG2, MVT::v8f16, Expand);
  setOperationAction(ISD::FLOG10, MVT::v8f16, Expand);

  // AArch64 has implementations of a lot of rounding-like FP operations.
  for (MVT Ty : {MVT::f32, MVT::f64}) {
    setOperationAction(ISD::FFLOOR, Ty, Legal);
    setOperationAction(ISD::FNEARBYINT, Ty, Legal);
    setOperationAction(ISD::FCEIL, Ty, Legal);
    setOperationAction(ISD::FRINT, Ty, Legal);
    setOperationAction(ISD::FTRUNC, Ty, Legal);
    setOperationAction(ISD::FROUND, Ty, Legal);
    setOperationAction(ISD::FMINNUM, Ty, Legal);
    setOperationAction(ISD::FMAXNUM, Ty, Legal);
    setOperationAction(ISD::FMINNAN, Ty, Legal);
    setOperationAction(ISD::FMAXNAN, Ty, Legal);
  }

  setOperationAction(ISD::PREFETCH, MVT::Other, Custom);

  // Lower READCYCLECOUNTER using an mrs from PMCCNTR_EL0.
  // This requires the Performance Monitors extension.
  if (Subtarget->hasPerfMon())
    setOperationAction(ISD::READCYCLECOUNTER, MVT::i64, Legal);

  if (Subtarget->isTargetMachO()) {
    // For iOS, we don't want to the normal expansion of a libcall to
    // sincos. We want to issue a libcall to __sincos_stret to avoid memory
    // traffic.
    setOperationAction(ISD::FSINCOS, MVT::f64, Custom);
    setOperationAction(ISD::FSINCOS, MVT::f32, Custom);
  } else {
    setOperationAction(ISD::FSINCOS, MVT::f64, Expand);
    setOperationAction(ISD::FSINCOS, MVT::f32, Expand);
  }

  // Make floating-point constants legal for the large code model, so they don't
  // become loads from the constant pool.
  if (Subtarget->isTargetMachO() && TM.getCodeModel() == CodeModel::Large) {
    setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
    setOperationAction(ISD::ConstantFP, MVT::f64, Legal);
  }

  // AArch64 does not have floating-point extending loads, i1 sign-extending
  // load, floating-point truncating stores, or v2i32->v2i16 truncating store.
  for (MVT VT : MVT::fp_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f16, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f32, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f80, Expand);
  }
  for (MVT VT : MVT::integer_valuetypes())
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Expand);

  setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  setTruncStoreAction(MVT::f128, MVT::f80, Expand);
  setTruncStoreAction(MVT::f128, MVT::f64, Expand);
  setTruncStoreAction(MVT::f128, MVT::f32, Expand);
  setTruncStoreAction(MVT::f128, MVT::f16, Expand);

  setOperationAction(ISD::BITCAST, MVT::i16, Custom);
  setOperationAction(ISD::BITCAST, MVT::f16, Custom);

  // Indexed loads and stores are supported.
  for (unsigned im = (unsigned)ISD::PRE_INC;
       im != (unsigned)ISD::LAST_INDEXED_MODE; ++im) {
    setIndexedLoadAction(im, MVT::i8, Legal);
    setIndexedLoadAction(im, MVT::i16, Legal);
    setIndexedLoadAction(im, MVT::i32, Legal);
    setIndexedLoadAction(im, MVT::i64, Legal);
    setIndexedLoadAction(im, MVT::f64, Legal);
    setIndexedLoadAction(im, MVT::f32, Legal);
    setIndexedLoadAction(im, MVT::f16, Legal);
    setIndexedStoreAction(im, MVT::i8, Legal);
    setIndexedStoreAction(im, MVT::i16, Legal);
    setIndexedStoreAction(im, MVT::i32, Legal);
    setIndexedStoreAction(im, MVT::i64, Legal);
    setIndexedStoreAction(im, MVT::f64, Legal);
    setIndexedStoreAction(im, MVT::f32, Legal);
    setIndexedStoreAction(im, MVT::f16, Legal);
  }

  // Trap.
  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  // We combine OR nodes for bitfield operations.
  setTargetDAGCombine(ISD::OR);

  // Vector add and sub nodes may conceal a high-half opportunity.
  // Also, try to fold ADD into CSINC/CSINV..
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::SUB);

  setTargetDAGCombine(ISD::XOR);
  setTargetDAGCombine(ISD::SINT_TO_FP);
  setTargetDAGCombine(ISD::UINT_TO_FP);

  setTargetDAGCombine(ISD::FP_TO_SINT);
  setTargetDAGCombine(ISD::FP_TO_UINT);
  setTargetDAGCombine(ISD::FDIV);

  setTargetDAGCombine(ISD::INTRINSIC_WO_CHAIN);

  setTargetDAGCombine(ISD::ANY_EXTEND);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  setTargetDAGCombine(ISD::SIGN_EXTEND);
  setTargetDAGCombine(ISD::BITCAST);
  setTargetDAGCombine(ISD::CONCAT_VECTORS);
  setTargetDAGCombine(ISD::STORE);
  if (Subtarget->supportsAddressTopByteIgnored())
    setTargetDAGCombine(ISD::LOAD);

  setTargetDAGCombine(ISD::MUL);

  setTargetDAGCombine(ISD::SELECT);
  setTargetDAGCombine(ISD::VSELECT);

  setTargetDAGCombine(ISD::INTRINSIC_VOID);
  setTargetDAGCombine(ISD::INTRINSIC_W_CHAIN);
  setTargetDAGCombine(ISD::INSERT_VECTOR_ELT);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT);

  MaxStoresPerMemset = MaxStoresPerMemsetOptSize = 8;
  MaxStoresPerMemcpy = MaxStoresPerMemcpyOptSize = 4;
  MaxStoresPerMemmove = MaxStoresPerMemmoveOptSize = 4;

  setStackPointerRegisterToSaveRestore(AArch64::SP);

  setSchedulingPreference(Sched::Hybrid);

  // Enable TBZ/TBNZ
  MaskAndBranchFoldingIsLegal = true;
  EnableExtLdPromotion = true;

  setMinFunctionAlignment(2);

  setHasExtractBitsInsn(true);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  if (Subtarget->hasNEON()) {
    // FIXME: v1f64 shouldn't be legal if we can avoid it, because it leads to
    // silliness like this:
    setOperationAction(ISD::FABS, MVT::v1f64, Expand);
    setOperationAction(ISD::FADD, MVT::v1f64, Expand);
    setOperationAction(ISD::FCEIL, MVT::v1f64, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::v1f64, Expand);
    setOperationAction(ISD::FCOS, MVT::v1f64, Expand);
    setOperationAction(ISD::FDIV, MVT::v1f64, Expand);
    setOperationAction(ISD::FFLOOR, MVT::v1f64, Expand);
    setOperationAction(ISD::FMA, MVT::v1f64, Expand);
    setOperationAction(ISD::FMUL, MVT::v1f64, Expand);
    setOperationAction(ISD::FNEARBYINT, MVT::v1f64, Expand);
    setOperationAction(ISD::FNEG, MVT::v1f64, Expand);
    setOperationAction(ISD::FPOW, MVT::v1f64, Expand);
    setOperationAction(ISD::FREM, MVT::v1f64, Expand);
    setOperationAction(ISD::FROUND, MVT::v1f64, Expand);
    setOperationAction(ISD::FRINT, MVT::v1f64, Expand);
    setOperationAction(ISD::FSIN, MVT::v1f64, Expand);
    setOperationAction(ISD::FSINCOS, MVT::v1f64, Expand);
    setOperationAction(ISD::FSQRT, MVT::v1f64, Expand);
    setOperationAction(ISD::FSUB, MVT::v1f64, Expand);
    setOperationAction(ISD::FTRUNC, MVT::v1f64, Expand);
    setOperationAction(ISD::SETCC, MVT::v1f64, Expand);
    setOperationAction(ISD::BR_CC, MVT::v1f64, Expand);
    setOperationAction(ISD::SELECT, MVT::v1f64, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::v1f64, Expand);
    setOperationAction(ISD::FP_EXTEND, MVT::v1f64, Expand);

    setOperationAction(ISD::FP_TO_SINT, MVT::v1i64, Expand);
    setOperationAction(ISD::FP_TO_UINT, MVT::v1i64, Expand);
    setOperationAction(ISD::SINT_TO_FP, MVT::v1i64, Expand);
    setOperationAction(ISD::UINT_TO_FP, MVT::v1i64, Expand);
    setOperationAction(ISD::FP_ROUND, MVT::v1f64, Expand);

    setOperationAction(ISD::MUL, MVT::v1i64, Expand);

    // AArch64 doesn't have a direct vector ->f32 conversion instructions for
    // elements smaller than i32, so promote the input to i32 first.
    setOperationAction(ISD::UINT_TO_FP, MVT::v4i8, Promote);
    setOperationAction(ISD::SINT_TO_FP, MVT::v4i8, Promote);
    setOperationAction(ISD::UINT_TO_FP, MVT::v4i16, Promote);
    setOperationAction(ISD::SINT_TO_FP, MVT::v4i16, Promote);
    // i8 and i16 vector elements also need promotion to i32 for v8i8 or v8i16
    // -> v8f16 conversions.
    setOperationAction(ISD::SINT_TO_FP, MVT::v8i8, Promote);
    setOperationAction(ISD::UINT_TO_FP, MVT::v8i8, Promote);
    setOperationAction(ISD::SINT_TO_FP, MVT::v8i16, Promote);
    setOperationAction(ISD::UINT_TO_FP, MVT::v8i16, Promote);
    // Similarly, there is no direct i32 -> f64 vector conversion instruction.
    setOperationAction(ISD::SINT_TO_FP, MVT::v2i32, Custom);
    setOperationAction(ISD::UINT_TO_FP, MVT::v2i32, Custom);
    setOperationAction(ISD::SINT_TO_FP, MVT::v2i64, Custom);
    setOperationAction(ISD::UINT_TO_FP, MVT::v2i64, Custom);
    // Or, direct i32 -> f16 vector conversion.  Set it so custom, so the
    // conversion happens in two steps: v4i32 -> v4f32 -> v4f16
    setOperationAction(ISD::SINT_TO_FP, MVT::v4i32, Custom);
    setOperationAction(ISD::UINT_TO_FP, MVT::v4i32, Custom);

    // AArch64 doesn't have MUL.2d:
    setOperationAction(ISD::MUL, MVT::v2i64, Expand);
    // Custom handling for some quad-vector types to detect MULL.
    setOperationAction(ISD::MUL, MVT::v8i16, Custom);
    setOperationAction(ISD::MUL, MVT::v4i32, Custom);
    setOperationAction(ISD::MUL, MVT::v2i64, Custom);

    setOperationAction(ISD::ANY_EXTEND, MVT::v4i32, Legal);
    setTruncStoreAction(MVT::v2i32, MVT::v2i16, Expand);
    // Likewise, narrowing and extending vector loads/stores aren't handled
    // directly.
    for (MVT VT : MVT::vector_valuetypes()) {
      setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Expand);

      setOperationAction(ISD::MULHS, VT, Expand);
      setOperationAction(ISD::SMUL_LOHI, VT, Expand);
      setOperationAction(ISD::MULHU, VT, Expand);
      setOperationAction(ISD::UMUL_LOHI, VT, Expand);

      setOperationAction(ISD::BSWAP, VT, Expand);

      for (MVT InnerVT : MVT::vector_valuetypes()) {
        setTruncStoreAction(VT, InnerVT, Expand);
        setLoadExtAction(ISD::SEXTLOAD, VT, InnerVT, Expand);
        setLoadExtAction(ISD::ZEXTLOAD, VT, InnerVT, Expand);
        setLoadExtAction(ISD::EXTLOAD, VT, InnerVT, Expand);
      }
    }

    // AArch64 has implementations of a lot of rounding-like FP operations.
    for (MVT Ty : {MVT::v2f32, MVT::v4f32, MVT::v2f64}) {
      setOperationAction(ISD::FFLOOR, Ty, Legal);
      setOperationAction(ISD::FNEARBYINT, Ty, Legal);
      setOperationAction(ISD::FCEIL, Ty, Legal);
      setOperationAction(ISD::FRINT, Ty, Legal);
      setOperationAction(ISD::FTRUNC, Ty, Legal);
      setOperationAction(ISD::FROUND, Ty, Legal);
    }
  }

  // Prefer likely predicted branches to selects on out-of-order cores.
  if (Subtarget->isCortexA57())
    PredictableSelectIsExpensive = true;
}

void AArch64TargetLowering::addTypeForNEON(EVT VT, EVT PromotedBitwiseVT) {
  if (VT == MVT::v2f32 || VT == MVT::v4f16) {
    setOperationAction(ISD::LOAD, VT.getSimpleVT(), Promote);
    AddPromotedToType(ISD::LOAD, VT.getSimpleVT(), MVT::v2i32);

    setOperationAction(ISD::STORE, VT.getSimpleVT(), Promote);
    AddPromotedToType(ISD::STORE, VT.getSimpleVT(), MVT::v2i32);
  } else if (VT == MVT::v2f64 || VT == MVT::v4f32 || VT == MVT::v8f16) {
    setOperationAction(ISD::LOAD, VT.getSimpleVT(), Promote);
    AddPromotedToType(ISD::LOAD, VT.getSimpleVT(), MVT::v2i64);

    setOperationAction(ISD::STORE, VT.getSimpleVT(), Promote);
    AddPromotedToType(ISD::STORE, VT.getSimpleVT(), MVT::v2i64);
  }

  // Mark vector float intrinsics as expand.
  if (VT == MVT::v2f32 || VT == MVT::v4f32 || VT == MVT::v2f64) {
    setOperationAction(ISD::FSIN, VT.getSimpleVT(), Expand);
    setOperationAction(ISD::FCOS, VT.getSimpleVT(), Expand);
    setOperationAction(ISD::FPOWI, VT.getSimpleVT(), Expand);
    setOperationAction(ISD::FPOW, VT.getSimpleVT(), Expand);
    setOperationAction(ISD::FLOG, VT.getSimpleVT(), Expand);
    setOperationAction(ISD::FLOG2, VT.getSimpleVT(), Expand);
    setOperationAction(ISD::FLOG10, VT.getSimpleVT(), Expand);
    setOperationAction(ISD::FEXP, VT.getSimpleVT(), Expand);
    setOperationAction(ISD::FEXP2, VT.getSimpleVT(), Expand);

    // But we do support custom-lowering for FCOPYSIGN.
    setOperationAction(ISD::FCOPYSIGN, VT.getSimpleVT(), Custom);
  }

  setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::BUILD_VECTOR, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::VECTOR_SHUFFLE, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::SRA, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::SRL, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::SHL, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::AND, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::OR, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::SETCC, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::CONCAT_VECTORS, VT.getSimpleVT(), Legal);

  setOperationAction(ISD::SELECT, VT.getSimpleVT(), Expand);
  setOperationAction(ISD::SELECT_CC, VT.getSimpleVT(), Expand);
  setOperationAction(ISD::VSELECT, VT.getSimpleVT(), Expand);
  for (MVT InnerVT : MVT::all_valuetypes())
    setLoadExtAction(ISD::EXTLOAD, InnerVT, VT.getSimpleVT(), Expand);

  // CNT supports only B element sizes.
  if (VT != MVT::v8i8 && VT != MVT::v16i8)
    setOperationAction(ISD::CTPOP, VT.getSimpleVT(), Expand);

  setOperationAction(ISD::UDIV, VT.getSimpleVT(), Expand);
  setOperationAction(ISD::SDIV, VT.getSimpleVT(), Expand);
  setOperationAction(ISD::UREM, VT.getSimpleVT(), Expand);
  setOperationAction(ISD::SREM, VT.getSimpleVT(), Expand);
  setOperationAction(ISD::FREM, VT.getSimpleVT(), Expand);

  setOperationAction(ISD::FP_TO_SINT, VT.getSimpleVT(), Custom);
  setOperationAction(ISD::FP_TO_UINT, VT.getSimpleVT(), Custom);

  // [SU][MIN|MAX] and [SU]ABSDIFF are available for all NEON types apart from
  // i64.
  if (!VT.isFloatingPoint() &&
      VT.getSimpleVT() != MVT::v2i64 && VT.getSimpleVT() != MVT::v1i64)
    for (unsigned Opcode : {ISD::SMIN, ISD::SMAX, ISD::UMIN, ISD::UMAX,
                            ISD::SABSDIFF, ISD::UABSDIFF})
      setOperationAction(Opcode, VT.getSimpleVT(), Legal);

  // F[MIN|MAX][NUM|NAN] are available for all FP NEON types (not f16 though!).
  if (VT.isFloatingPoint() && VT.getVectorElementType() != MVT::f16)
    for (unsigned Opcode : {ISD::FMINNAN, ISD::FMAXNAN,
                            ISD::FMINNUM, ISD::FMAXNUM})
      setOperationAction(Opcode, VT.getSimpleVT(), Legal);

  if (Subtarget->isLittleEndian()) {
    for (unsigned im = (unsigned)ISD::PRE_INC;
         im != (unsigned)ISD::LAST_INDEXED_MODE; ++im) {
      setIndexedLoadAction(im, VT.getSimpleVT(), Legal);
      setIndexedStoreAction(im, VT.getSimpleVT(), Legal);
    }
  }
}

void AArch64TargetLowering::addDRTypeForNEON(MVT VT) {
  addRegisterClass(VT, &AArch64::FPR64RegClass);
  addTypeForNEON(VT, MVT::v2i32);
}

void AArch64TargetLowering::addQRTypeForNEON(MVT VT) {
  addRegisterClass(VT, &AArch64::FPR128RegClass);
  addTypeForNEON(VT, MVT::v4i32);
}

EVT AArch64TargetLowering::getSetCCResultType(const DataLayout &, LLVMContext &,
                                              EVT VT) const {
  if (!VT.isVector())
    return MVT::i32;
  return VT.changeVectorElementTypeToInteger();
}

/// computeKnownBitsForTargetNode - Determine which of the bits specified in
/// Mask are known to be either zero or one and return them in the
/// KnownZero/KnownOne bitsets.
void AArch64TargetLowering::computeKnownBitsForTargetNode(
    const SDValue Op, APInt &KnownZero, APInt &KnownOne,
    const SelectionDAG &DAG, unsigned Depth) const {
  switch (Op.getOpcode()) {
  default:
    break;
  case AArch64ISD::CSEL: {
    APInt KnownZero2, KnownOne2;
    DAG.computeKnownBits(Op->getOperand(0), KnownZero, KnownOne, Depth + 1);
    DAG.computeKnownBits(Op->getOperand(1), KnownZero2, KnownOne2, Depth + 1);
    KnownZero &= KnownZero2;
    KnownOne &= KnownOne2;
    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    ConstantSDNode *CN = cast<ConstantSDNode>(Op->getOperand(1));
    Intrinsic::ID IntID = static_cast<Intrinsic::ID>(CN->getZExtValue());
    switch (IntID) {
    default: return;
    case Intrinsic::aarch64_ldaxr:
    case Intrinsic::aarch64_ldxr: {
      unsigned BitWidth = KnownOne.getBitWidth();
      EVT VT = cast<MemIntrinsicSDNode>(Op)->getMemoryVT();
      unsigned MemBits = VT.getScalarType().getSizeInBits();
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - MemBits);
      return;
    }
    }
    break;
  }
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_VOID: {
    unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
    switch (IntNo) {
    default:
      break;
    case Intrinsic::aarch64_neon_umaxv:
    case Intrinsic::aarch64_neon_uminv: {
      // Figure out the datatype of the vector operand. The UMINV instruction
      // will zero extend the result, so we can mark as known zero all the
      // bits larger than the element datatype. 32-bit or larget doesn't need
      // this as those are legal types and will be handled by isel directly.
      MVT VT = Op.getOperand(1).getValueType().getSimpleVT();
      unsigned BitWidth = KnownZero.getBitWidth();
      if (VT == MVT::v8i8 || VT == MVT::v16i8) {
        assert(BitWidth >= 8 && "Unexpected width!");
        APInt Mask = APInt::getHighBitsSet(BitWidth, BitWidth - 8);
        KnownZero |= Mask;
      } else if (VT == MVT::v4i16 || VT == MVT::v8i16) {
        assert(BitWidth >= 16 && "Unexpected width!");
        APInt Mask = APInt::getHighBitsSet(BitWidth, BitWidth - 16);
        KnownZero |= Mask;
      }
      break;
    } break;
    }
  }
  }
}

MVT AArch64TargetLowering::getScalarShiftAmountTy(const DataLayout &DL,
                                                  EVT) const {
  return MVT::i64;
}

bool AArch64TargetLowering::allowsMisalignedMemoryAccesses(EVT VT,
                                                           unsigned AddrSpace,
                                                           unsigned Align,
                                                           bool *Fast) const {
  if (Subtarget->requiresStrictAlign())
    return false;

  // FIXME: This is mostly true for Cyclone, but not necessarily others.
  if (Fast) {
    // FIXME: Define an attribute for slow unaligned accesses instead of
    // relying on the CPU type as a proxy.
    // On Cyclone, unaligned 128-bit stores are slow.
    *Fast = !Subtarget->isCyclone() || VT.getStoreSize() != 16 ||
            // See comments in performSTORECombine() for more details about
            // these conditions.

            // Code that uses clang vector extensions can mark that it
            // wants unaligned accesses to be treated as fast by
            // underspecifying alignment to be 1 or 2.
            Align <= 2 ||

            // Disregard v2i64. Memcpy lowering produces those and splitting
            // them regresses performance on micro-benchmarks and olden/bh.
            VT == MVT::v2i64;
  }
  return true;
}

FastISel *
AArch64TargetLowering::createFastISel(FunctionLoweringInfo &funcInfo,
                                      const TargetLibraryInfo *libInfo) const {
  return AArch64::createFastISel(funcInfo, libInfo);
}

const char *AArch64TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((AArch64ISD::NodeType)Opcode) {
  case AArch64ISD::FIRST_NUMBER:      break;
  case AArch64ISD::CALL:              return "AArch64ISD::CALL";
  case AArch64ISD::ADRP:              return "AArch64ISD::ADRP";
  case AArch64ISD::ADDlow:            return "AArch64ISD::ADDlow";
  case AArch64ISD::LOADgot:           return "AArch64ISD::LOADgot";
  case AArch64ISD::RET_FLAG:          return "AArch64ISD::RET_FLAG";
  case AArch64ISD::BRCOND:            return "AArch64ISD::BRCOND";
  case AArch64ISD::CSEL:              return "AArch64ISD::CSEL";
  case AArch64ISD::FCSEL:             return "AArch64ISD::FCSEL";
  case AArch64ISD::CSINV:             return "AArch64ISD::CSINV";
  case AArch64ISD::CSNEG:             return "AArch64ISD::CSNEG";
  case AArch64ISD::CSINC:             return "AArch64ISD::CSINC";
  case AArch64ISD::THREAD_POINTER:    return "AArch64ISD::THREAD_POINTER";
  case AArch64ISD::TLSDESC_CALLSEQ:   return "AArch64ISD::TLSDESC_CALLSEQ";
  case AArch64ISD::ADC:               return "AArch64ISD::ADC";
  case AArch64ISD::SBC:               return "AArch64ISD::SBC";
  case AArch64ISD::ADDS:              return "AArch64ISD::ADDS";
  case AArch64ISD::SUBS:              return "AArch64ISD::SUBS";
  case AArch64ISD::ADCS:              return "AArch64ISD::ADCS";
  case AArch64ISD::SBCS:              return "AArch64ISD::SBCS";
  case AArch64ISD::ANDS:              return "AArch64ISD::ANDS";
  case AArch64ISD::CCMP:              return "AArch64ISD::CCMP";
  case AArch64ISD::CCMN:              return "AArch64ISD::CCMN";
  case AArch64ISD::FCCMP:             return "AArch64ISD::FCCMP";
  case AArch64ISD::FCMP:              return "AArch64ISD::FCMP";
  case AArch64ISD::DUP:               return "AArch64ISD::DUP";
  case AArch64ISD::DUPLANE8:          return "AArch64ISD::DUPLANE8";
  case AArch64ISD::DUPLANE16:         return "AArch64ISD::DUPLANE16";
  case AArch64ISD::DUPLANE32:         return "AArch64ISD::DUPLANE32";
  case AArch64ISD::DUPLANE64:         return "AArch64ISD::DUPLANE64";
  case AArch64ISD::MOVI:              return "AArch64ISD::MOVI";
  case AArch64ISD::MOVIshift:         return "AArch64ISD::MOVIshift";
  case AArch64ISD::MOVIedit:          return "AArch64ISD::MOVIedit";
  case AArch64ISD::MOVImsl:           return "AArch64ISD::MOVImsl";
  case AArch64ISD::FMOV:              return "AArch64ISD::FMOV";
  case AArch64ISD::MVNIshift:         return "AArch64ISD::MVNIshift";
  case AArch64ISD::MVNImsl:           return "AArch64ISD::MVNImsl";
  case AArch64ISD::BICi:              return "AArch64ISD::BICi";
  case AArch64ISD::ORRi:              return "AArch64ISD::ORRi";
  case AArch64ISD::BSL:               return "AArch64ISD::BSL";
  case AArch64ISD::NEG:               return "AArch64ISD::NEG";
  case AArch64ISD::EXTR:              return "AArch64ISD::EXTR";
  case AArch64ISD::ZIP1:              return "AArch64ISD::ZIP1";
  case AArch64ISD::ZIP2:              return "AArch64ISD::ZIP2";
  case AArch64ISD::UZP1:              return "AArch64ISD::UZP1";
  case AArch64ISD::UZP2:              return "AArch64ISD::UZP2";
  case AArch64ISD::TRN1:              return "AArch64ISD::TRN1";
  case AArch64ISD::TRN2:              return "AArch64ISD::TRN2";
  case AArch64ISD::REV16:             return "AArch64ISD::REV16";
  case AArch64ISD::REV32:             return "AArch64ISD::REV32";
  case AArch64ISD::REV64:             return "AArch64ISD::REV64";
  case AArch64ISD::EXT:               return "AArch64ISD::EXT";
  case AArch64ISD::VSHL:              return "AArch64ISD::VSHL";
  case AArch64ISD::VLSHR:             return "AArch64ISD::VLSHR";
  case AArch64ISD::VASHR:             return "AArch64ISD::VASHR";
  case AArch64ISD::CMEQ:              return "AArch64ISD::CMEQ";
  case AArch64ISD::CMGE:              return "AArch64ISD::CMGE";
  case AArch64ISD::CMGT:              return "AArch64ISD::CMGT";
  case AArch64ISD::CMHI:              return "AArch64ISD::CMHI";
  case AArch64ISD::CMHS:              return "AArch64ISD::CMHS";
  case AArch64ISD::FCMEQ:             return "AArch64ISD::FCMEQ";
  case AArch64ISD::FCMGE:             return "AArch64ISD::FCMGE";
  case AArch64ISD::FCMGT:             return "AArch64ISD::FCMGT";
  case AArch64ISD::CMEQz:             return "AArch64ISD::CMEQz";
  case AArch64ISD::CMGEz:             return "AArch64ISD::CMGEz";
  case AArch64ISD::CMGTz:             return "AArch64ISD::CMGTz";
  case AArch64ISD::CMLEz:             return "AArch64ISD::CMLEz";
  case AArch64ISD::CMLTz:             return "AArch64ISD::CMLTz";
  case AArch64ISD::FCMEQz:            return "AArch64ISD::FCMEQz";
  case AArch64ISD::FCMGEz:            return "AArch64ISD::FCMGEz";
  case AArch64ISD::FCMGTz:            return "AArch64ISD::FCMGTz";
  case AArch64ISD::FCMLEz:            return "AArch64ISD::FCMLEz";
  case AArch64ISD::FCMLTz:            return "AArch64ISD::FCMLTz";
  case AArch64ISD::SADDV:             return "AArch64ISD::SADDV";
  case AArch64ISD::UADDV:             return "AArch64ISD::UADDV";
  case AArch64ISD::SMINV:             return "AArch64ISD::SMINV";
  case AArch64ISD::UMINV:             return "AArch64ISD::UMINV";
  case AArch64ISD::SMAXV:             return "AArch64ISD::SMAXV";
  case AArch64ISD::UMAXV:             return "AArch64ISD::UMAXV";
  case AArch64ISD::NOT:               return "AArch64ISD::NOT";
  case AArch64ISD::BIT:               return "AArch64ISD::BIT";
  case AArch64ISD::CBZ:               return "AArch64ISD::CBZ";
  case AArch64ISD::CBNZ:              return "AArch64ISD::CBNZ";
  case AArch64ISD::TBZ:               return "AArch64ISD::TBZ";
  case AArch64ISD::TBNZ:              return "AArch64ISD::TBNZ";
  case AArch64ISD::TC_RETURN:         return "AArch64ISD::TC_RETURN";
  case AArch64ISD::PREFETCH:          return "AArch64ISD::PREFETCH";
  case AArch64ISD::SITOF:             return "AArch64ISD::SITOF";
  case AArch64ISD::UITOF:             return "AArch64ISD::UITOF";
  case AArch64ISD::NVCAST:            return "AArch64ISD::NVCAST";
  case AArch64ISD::SQSHL_I:           return "AArch64ISD::SQSHL_I";
  case AArch64ISD::UQSHL_I:           return "AArch64ISD::UQSHL_I";
  case AArch64ISD::SRSHR_I:           return "AArch64ISD::SRSHR_I";
  case AArch64ISD::URSHR_I:           return "AArch64ISD::URSHR_I";
  case AArch64ISD::SQSHLU_I:          return "AArch64ISD::SQSHLU_I";
  case AArch64ISD::WrapperLarge:      return "AArch64ISD::WrapperLarge";
  case AArch64ISD::LD2post:           return "AArch64ISD::LD2post";
  case AArch64ISD::LD3post:           return "AArch64ISD::LD3post";
  case AArch64ISD::LD4post:           return "AArch64ISD::LD4post";
  case AArch64ISD::ST2post:           return "AArch64ISD::ST2post";
  case AArch64ISD::ST3post:           return "AArch64ISD::ST3post";
  case AArch64ISD::ST4post:           return "AArch64ISD::ST4post";
  case AArch64ISD::LD1x2post:         return "AArch64ISD::LD1x2post";
  case AArch64ISD::LD1x3post:         return "AArch64ISD::LD1x3post";
  case AArch64ISD::LD1x4post:         return "AArch64ISD::LD1x4post";
  case AArch64ISD::ST1x2post:         return "AArch64ISD::ST1x2post";
  case AArch64ISD::ST1x3post:         return "AArch64ISD::ST1x3post";
  case AArch64ISD::ST1x4post:         return "AArch64ISD::ST1x4post";
  case AArch64ISD::LD1DUPpost:        return "AArch64ISD::LD1DUPpost";
  case AArch64ISD::LD2DUPpost:        return "AArch64ISD::LD2DUPpost";
  case AArch64ISD::LD3DUPpost:        return "AArch64ISD::LD3DUPpost";
  case AArch64ISD::LD4DUPpost:        return "AArch64ISD::LD4DUPpost";
  case AArch64ISD::LD1LANEpost:       return "AArch64ISD::LD1LANEpost";
  case AArch64ISD::LD2LANEpost:       return "AArch64ISD::LD2LANEpost";
  case AArch64ISD::LD3LANEpost:       return "AArch64ISD::LD3LANEpost";
  case AArch64ISD::LD4LANEpost:       return "AArch64ISD::LD4LANEpost";
  case AArch64ISD::ST2LANEpost:       return "AArch64ISD::ST2LANEpost";
  case AArch64ISD::ST3LANEpost:       return "AArch64ISD::ST3LANEpost";
  case AArch64ISD::ST4LANEpost:       return "AArch64ISD::ST4LANEpost";
  case AArch64ISD::SMULL:             return "AArch64ISD::SMULL";
  case AArch64ISD::UMULL:             return "AArch64ISD::UMULL";
  }
  return nullptr;
}

MachineBasicBlock *
AArch64TargetLowering::EmitF128CSEL(MachineInstr *MI,
                                    MachineBasicBlock *MBB) const {
  // We materialise the F128CSEL pseudo-instruction as some control flow and a
  // phi node:

  // OrigBB:
  //     [... previous instrs leading to comparison ...]
  //     b.ne TrueBB
  //     b EndBB
  // TrueBB:
  //     ; Fallthrough
  // EndBB:
  //     Dest = PHI [IfTrue, TrueBB], [IfFalse, OrigBB]

  MachineFunction *MF = MBB->getParent();
  const TargetInstrInfo *TII = Subtarget->getInstrInfo();
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  DebugLoc DL = MI->getDebugLoc();
  MachineFunction::iterator It = ++MBB->getIterator();

  unsigned DestReg = MI->getOperand(0).getReg();
  unsigned IfTrueReg = MI->getOperand(1).getReg();
  unsigned IfFalseReg = MI->getOperand(2).getReg();
  unsigned CondCode = MI->getOperand(3).getImm();
  bool NZCVKilled = MI->getOperand(4).isKill();

  MachineBasicBlock *TrueBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *EndBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, TrueBB);
  MF->insert(It, EndBB);

  // Transfer rest of current basic-block to EndBB
  EndBB->splice(EndBB->begin(), MBB, std::next(MachineBasicBlock::iterator(MI)),
                MBB->end());
  EndBB->transferSuccessorsAndUpdatePHIs(MBB);

  BuildMI(MBB, DL, TII->get(AArch64::Bcc)).addImm(CondCode).addMBB(TrueBB);
  BuildMI(MBB, DL, TII->get(AArch64::B)).addMBB(EndBB);
  MBB->addSuccessor(TrueBB);
  MBB->addSuccessor(EndBB);

  // TrueBB falls through to the end.
  TrueBB->addSuccessor(EndBB);

  if (!NZCVKilled) {
    TrueBB->addLiveIn(AArch64::NZCV);
    EndBB->addLiveIn(AArch64::NZCV);
  }

  BuildMI(*EndBB, EndBB->begin(), DL, TII->get(AArch64::PHI), DestReg)
      .addReg(IfTrueReg)
      .addMBB(TrueBB)
      .addReg(IfFalseReg)
      .addMBB(MBB);

  MI->eraseFromParent();
  return EndBB;
}

MachineBasicBlock *
AArch64TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                 MachineBasicBlock *BB) const {
  switch (MI->getOpcode()) {
  default:
#ifndef NDEBUG
    MI->dump();
#endif
    llvm_unreachable("Unexpected instruction for custom inserter!");

  case AArch64::F128CSEL:
    return EmitF128CSEL(MI, BB);

  case TargetOpcode::STACKMAP:
  case TargetOpcode::PATCHPOINT:
    return emitPatchPoint(MI, BB);
  }
}

//===----------------------------------------------------------------------===//
// AArch64 Lowering private implementation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lowering Code
//===----------------------------------------------------------------------===//

/// changeIntCCToAArch64CC - Convert a DAG integer condition code to an AArch64
/// CC
static AArch64CC::CondCode changeIntCCToAArch64CC(ISD::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case ISD::SETNE:
    return AArch64CC::NE;
  case ISD::SETEQ:
    return AArch64CC::EQ;
  case ISD::SETGT:
    return AArch64CC::GT;
  case ISD::SETGE:
    return AArch64CC::GE;
  case ISD::SETLT:
    return AArch64CC::LT;
  case ISD::SETLE:
    return AArch64CC::LE;
  case ISD::SETUGT:
    return AArch64CC::HI;
  case ISD::SETUGE:
    return AArch64CC::HS;
  case ISD::SETULT:
    return AArch64CC::LO;
  case ISD::SETULE:
    return AArch64CC::LS;
  }
}

/// changeFPCCToAArch64CC - Convert a DAG fp condition code to an AArch64 CC.
static void changeFPCCToAArch64CC(ISD::CondCode CC,
                                  AArch64CC::CondCode &CondCode,
                                  AArch64CC::CondCode &CondCode2) {
  CondCode2 = AArch64CC::AL;
  switch (CC) {
  default:
    llvm_unreachable("Unknown FP condition!");
  case ISD::SETEQ:
  case ISD::SETOEQ:
    CondCode = AArch64CC::EQ;
    break;
  case ISD::SETGT:
  case ISD::SETOGT:
    CondCode = AArch64CC::GT;
    break;
  case ISD::SETGE:
  case ISD::SETOGE:
    CondCode = AArch64CC::GE;
    break;
  case ISD::SETOLT:
    CondCode = AArch64CC::MI;
    break;
  case ISD::SETOLE:
    CondCode = AArch64CC::LS;
    break;
  case ISD::SETONE:
    CondCode = AArch64CC::MI;
    CondCode2 = AArch64CC::GT;
    break;
  case ISD::SETO:
    CondCode = AArch64CC::VC;
    break;
  case ISD::SETUO:
    CondCode = AArch64CC::VS;
    break;
  case ISD::SETUEQ:
    CondCode = AArch64CC::EQ;
    CondCode2 = AArch64CC::VS;
    break;
  case ISD::SETUGT:
    CondCode = AArch64CC::HI;
    break;
  case ISD::SETUGE:
    CondCode = AArch64CC::PL;
    break;
  case ISD::SETLT:
  case ISD::SETULT:
    CondCode = AArch64CC::LT;
    break;
  case ISD::SETLE:
  case ISD::SETULE:
    CondCode = AArch64CC::LE;
    break;
  case ISD::SETNE:
  case ISD::SETUNE:
    CondCode = AArch64CC::NE;
    break;
  }
}

/// changeVectorFPCCToAArch64CC - Convert a DAG fp condition code to an AArch64
/// CC usable with the vector instructions. Fewer operations are available
/// without a real NZCV register, so we have to use less efficient combinations
/// to get the same effect.
static void changeVectorFPCCToAArch64CC(ISD::CondCode CC,
                                        AArch64CC::CondCode &CondCode,
                                        AArch64CC::CondCode &CondCode2,
                                        bool &Invert) {
  Invert = false;
  switch (CC) {
  default:
    // Mostly the scalar mappings work fine.
    changeFPCCToAArch64CC(CC, CondCode, CondCode2);
    break;
  case ISD::SETUO:
    Invert = true; // Fallthrough
  case ISD::SETO:
    CondCode = AArch64CC::MI;
    CondCode2 = AArch64CC::GE;
    break;
  case ISD::SETUEQ:
  case ISD::SETULT:
  case ISD::SETULE:
  case ISD::SETUGT:
  case ISD::SETUGE:
    // All of the compare-mask comparisons are ordered, but we can switch
    // between the two by a double inversion. E.g. ULE == !OGT.
    Invert = true;
    changeFPCCToAArch64CC(getSetCCInverse(CC, false), CondCode, CondCode2);
    break;
  }
}

static bool isLegalArithImmed(uint64_t C) {
  // Matches AArch64DAGToDAGISel::SelectArithImmed().
  return (C >> 12 == 0) || ((C & 0xFFFULL) == 0 && C >> 24 == 0);
}

static SDValue emitComparison(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                              SDLoc dl, SelectionDAG &DAG) {
  EVT VT = LHS.getValueType();

  if (VT.isFloatingPoint())
    return DAG.getNode(AArch64ISD::FCMP, dl, VT, LHS, RHS);

  // The CMP instruction is just an alias for SUBS, and representing it as
  // SUBS means that it's possible to get CSE with subtract operations.
  // A later phase can perform the optimization of setting the destination
  // register to WZR/XZR if it ends up being unused.
  unsigned Opcode = AArch64ISD::SUBS;

  if (RHS.getOpcode() == ISD::SUB && isa<ConstantSDNode>(RHS.getOperand(0)) &&
      cast<ConstantSDNode>(RHS.getOperand(0))->getZExtValue() == 0 &&
      (CC == ISD::SETEQ || CC == ISD::SETNE)) {
    // We'd like to combine a (CMP op1, (sub 0, op2) into a CMN instruction on
    // the grounds that "op1 - (-op2) == op1 + op2". However, the C and V flags
    // can be set differently by this operation. It comes down to whether
    // "SInt(~op2)+1 == SInt(~op2+1)" (and the same for UInt). If they are then
    // everything is fine. If not then the optimization is wrong. Thus general
    // comparisons are only valid if op2 != 0.

    // So, finally, the only LLVM-native comparisons that don't mention C and V
    // are SETEQ and SETNE. They're the only ones we can safely use CMN for in
    // the absence of information about op2.
    Opcode = AArch64ISD::ADDS;
    RHS = RHS.getOperand(1);
  } else if (LHS.getOpcode() == ISD::AND && isa<ConstantSDNode>(RHS) &&
             cast<ConstantSDNode>(RHS)->getZExtValue() == 0 &&
             !isUnsignedIntSetCC(CC)) {
    // Similarly, (CMP (and X, Y), 0) can be implemented with a TST
    // (a.k.a. ANDS) except that the flags are only guaranteed to work for one
    // of the signed comparisons.
    Opcode = AArch64ISD::ANDS;
    RHS = LHS.getOperand(1);
    LHS = LHS.getOperand(0);
  }

  return DAG.getNode(Opcode, dl, DAG.getVTList(VT, MVT_CC), LHS, RHS)
      .getValue(1);
}

/// \defgroup AArch64CCMP CMP;CCMP matching
///
/// These functions deal with the formation of CMP;CCMP;... sequences.
/// The CCMP/CCMN/FCCMP/FCCMPE instructions allow the conditional execution of
/// a comparison. They set the NZCV flags to a predefined value if their
/// predicate is false. This allows to express arbitrary conjunctions, for
/// example "cmp 0 (and (setCA (cmp A)) (setCB (cmp B))))"
/// expressed as:
///   cmp A
///   ccmp B, inv(CB), CA
///   check for CB flags
///
/// In general we can create code for arbitrary "... (and (and A B) C)"
/// sequences. We can also implement some "or" expressions, because "(or A B)"
/// is equivalent to "not (and (not A) (not B))" and we can implement some
/// negation operations:
/// We can negate the results of a single comparison by inverting the flags
/// used when the predicate fails and inverting the flags tested in the next
/// instruction; We can also negate the results of the whole previous
/// conditional compare sequence by inverting the flags tested in the next
/// instruction. However there is no way to negate the result of a partial
/// sequence.
///
/// Therefore on encountering an "or" expression we can negate the subtree on
/// one side and have to be able to push the negate to the leafs of the subtree
/// on the other side (see also the comments in code). As complete example:
/// "or (or (setCA (cmp A)) (setCB (cmp B)))
///     (and (setCC (cmp C)) (setCD (cmp D)))"
/// is transformed to
/// "not (and (not (and (setCC (cmp C)) (setCC (cmp D))))
///           (and (not (setCA (cmp A)) (not (setCB (cmp B))))))"
/// and implemented as:
///   cmp C
///   ccmp D, inv(CD), CC
///   ccmp A, CA, inv(CD)
///   ccmp B, CB, inv(CA)
///   check for CB flags
/// A counterexample is "or (and A B) (and C D)" which cannot be implemented
/// by conditional compare sequences.
/// @{

/// Create a conditional comparison; Use CCMP, CCMN or FCCMP as appropriate.
static SDValue emitConditionalComparison(SDValue LHS, SDValue RHS,
                                         ISD::CondCode CC, SDValue CCOp,
                                         SDValue Condition, unsigned NZCV,
                                         SDLoc DL, SelectionDAG &DAG) {
  unsigned Opcode = 0;
  if (LHS.getValueType().isFloatingPoint())
    Opcode = AArch64ISD::FCCMP;
  else if (RHS.getOpcode() == ISD::SUB) {
    SDValue SubOp0 = RHS.getOperand(0);
    if (const ConstantSDNode *SubOp0C = dyn_cast<ConstantSDNode>(SubOp0))
      if (SubOp0C->isNullValue() && (CC == ISD::SETEQ || CC == ISD::SETNE)) {
        // See emitComparison() on why we can only do this for SETEQ and SETNE.
        Opcode = AArch64ISD::CCMN;
        RHS = RHS.getOperand(1);
      }
  }
  if (Opcode == 0)
    Opcode = AArch64ISD::CCMP;

  SDValue NZCVOp = DAG.getConstant(NZCV, DL, MVT::i32);
  return DAG.getNode(Opcode, DL, MVT_CC, LHS, RHS, NZCVOp, Condition, CCOp);
}

/// Returns true if @p Val is a tree of AND/OR/SETCC operations.
/// CanPushNegate is set to true if we can push a negate operation through
/// the tree in a was that we are left with AND operations and negate operations
/// at the leafs only. i.e. "not (or (or x y) z)" can be changed to
/// "and (and (not x) (not y)) (not z)"; "not (or (and x y) z)" cannot be
/// brought into such a form.
static bool isConjunctionDisjunctionTree(const SDValue Val, bool &CanPushNegate,
                                         unsigned Depth = 0) {
  if (!Val.hasOneUse())
    return false;
  unsigned Opcode = Val->getOpcode();
  if (Opcode == ISD::SETCC) {
    CanPushNegate = true;
    return true;
  }
  // Protect against stack overflow.
  if (Depth > 15)
    return false;
  if (Opcode == ISD::AND || Opcode == ISD::OR) {
    SDValue O0 = Val->getOperand(0);
    SDValue O1 = Val->getOperand(1);
    bool CanPushNegateL;
    if (!isConjunctionDisjunctionTree(O0, CanPushNegateL, Depth+1))
      return false;
    bool CanPushNegateR;
    if (!isConjunctionDisjunctionTree(O1, CanPushNegateR, Depth+1))
      return false;
    // We cannot push a negate through an AND operation (it would become an OR),
    // we can however change a (not (or x y)) to (and (not x) (not y)) if we can
    // push the negate through the x/y subtrees.
    CanPushNegate = (Opcode == ISD::OR) && CanPushNegateL && CanPushNegateR;
    return true;
  }
  return false;
}

/// Emit conjunction or disjunction tree with the CMP/FCMP followed by a chain
/// of CCMP/CFCMP ops. See @ref AArch64CCMP.
/// Tries to transform the given i1 producing node @p Val to a series compare
/// and conditional compare operations. @returns an NZCV flags producing node
/// and sets @p OutCC to the flags that should be tested or returns SDValue() if
/// transformation was not possible.
/// On recursive invocations @p PushNegate may be set to true to have negation
/// effects pushed to the tree leafs; @p Predicate is an NZCV flag predicate
/// for the comparisons in the current subtree; @p Depth limits the search
/// depth to avoid stack overflow.
static SDValue emitConjunctionDisjunctionTree(SelectionDAG &DAG, SDValue Val,
    AArch64CC::CondCode &OutCC, bool PushNegate = false,
    SDValue CCOp = SDValue(), AArch64CC::CondCode Predicate = AArch64CC::AL,
    unsigned Depth = 0) {
  // We're at a tree leaf, produce a conditional comparison operation.
  unsigned Opcode = Val->getOpcode();
  if (Opcode == ISD::SETCC) {
    SDValue LHS = Val->getOperand(0);
    SDValue RHS = Val->getOperand(1);
    ISD::CondCode CC = cast<CondCodeSDNode>(Val->getOperand(2))->get();
    bool isInteger = LHS.getValueType().isInteger();
    if (PushNegate)
      CC = getSetCCInverse(CC, isInteger);
    SDLoc DL(Val);
    // Determine OutCC and handle FP special case.
    if (isInteger) {
      OutCC = changeIntCCToAArch64CC(CC);
    } else {
      assert(LHS.getValueType().isFloatingPoint());
      AArch64CC::CondCode ExtraCC;
      changeFPCCToAArch64CC(CC, OutCC, ExtraCC);
      // Surpisingly some floating point conditions can't be tested with a
      // single condition code. Construct an additional comparison in this case.
      // See comment below on how we deal with OR conditions.
      if (ExtraCC != AArch64CC::AL) {
        SDValue ExtraCmp;
        if (!CCOp.getNode())
          ExtraCmp = emitComparison(LHS, RHS, CC, DL, DAG);
        else {
          SDValue ConditionOp = DAG.getConstant(Predicate, DL, MVT_CC);
          // Note that we want the inverse of ExtraCC, so NZCV is not inversed.
          unsigned NZCV = AArch64CC::getNZCVToSatisfyCondCode(ExtraCC);
          ExtraCmp = emitConditionalComparison(LHS, RHS, CC, CCOp, ConditionOp,
                                               NZCV, DL, DAG);
        }
        CCOp = ExtraCmp;
        Predicate = AArch64CC::getInvertedCondCode(ExtraCC);
        OutCC = AArch64CC::getInvertedCondCode(OutCC);
      }
    }

    // Produce a normal comparison if we are first in the chain
    if (!CCOp.getNode())
      return emitComparison(LHS, RHS, CC, DL, DAG);
    // Otherwise produce a ccmp.
    SDValue ConditionOp = DAG.getConstant(Predicate, DL, MVT_CC);
    AArch64CC::CondCode InvOutCC = AArch64CC::getInvertedCondCode(OutCC);
    unsigned NZCV = AArch64CC::getNZCVToSatisfyCondCode(InvOutCC);
    return emitConditionalComparison(LHS, RHS, CC, CCOp, ConditionOp, NZCV, DL,
                                     DAG);
  } else if ((Opcode != ISD::AND && Opcode != ISD::OR) || !Val->hasOneUse())
    return SDValue();

  assert((Opcode == ISD::OR || !PushNegate)
         && "Can only push negate through OR operation");

  // Check if both sides can be transformed.
  SDValue LHS = Val->getOperand(0);
  SDValue RHS = Val->getOperand(1);
  bool CanPushNegateL;
  if (!isConjunctionDisjunctionTree(LHS, CanPushNegateL, Depth+1))
    return SDValue();
  bool CanPushNegateR;
  if (!isConjunctionDisjunctionTree(RHS, CanPushNegateR, Depth+1))
    return SDValue();

  // Do we need to negate our operands?
  bool NegateOperands = Opcode == ISD::OR;
  // We can negate the results of all previous operations by inverting the
  // predicate flags giving us a free negation for one side. For the other side
  // we need to be able to push the negation to the leafs of the tree.
  if (NegateOperands) {
    if (!CanPushNegateL && !CanPushNegateR)
      return SDValue();
    // Order the side where we can push the negate through to LHS.
    if (!CanPushNegateL && CanPushNegateR)
      std::swap(LHS, RHS);
  } else {
    bool NeedsNegOutL = LHS->getOpcode() == ISD::OR;
    bool NeedsNegOutR = RHS->getOpcode() == ISD::OR;
    if (NeedsNegOutL && NeedsNegOutR)
      return SDValue();
    // Order the side where we need to negate the output flags to RHS so it
    // gets emitted first.
    if (NeedsNegOutL)
      std::swap(LHS, RHS);
  }

  // Emit RHS. If we want to negate the tree we only need to push a negate
  // through if we are already in a PushNegate case, otherwise we can negate
  // the "flags to test" afterwards.
  AArch64CC::CondCode RHSCC;
  SDValue CmpR = emitConjunctionDisjunctionTree(DAG, RHS, RHSCC, PushNegate,
                                                CCOp, Predicate, Depth+1);
  if (NegateOperands && !PushNegate)
    RHSCC = AArch64CC::getInvertedCondCode(RHSCC);
  // Emit LHS. We must push the negate through if we need to negate it.
  SDValue CmpL = emitConjunctionDisjunctionTree(DAG, LHS, OutCC, NegateOperands,
                                                CmpR, RHSCC, Depth+1);
  // If we transformed an OR to and AND then we have to negate the result
  // (or absorb a PushNegate resulting in a double negation).
  if (Opcode == ISD::OR && !PushNegate)
    OutCC = AArch64CC::getInvertedCondCode(OutCC);
  return CmpL;
}

/// @}

static SDValue getAArch64Cmp(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                             SDValue &AArch64cc, SelectionDAG &DAG, SDLoc dl) {
  if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS.getNode())) {
    EVT VT = RHS.getValueType();
    uint64_t C = RHSC->getZExtValue();
    if (!isLegalArithImmed(C)) {
      // Constant does not fit, try adjusting it by one?
      switch (CC) {
      default:
        break;
      case ISD::SETLT:
      case ISD::SETGE:
        if ((VT == MVT::i32 && C != 0x80000000 &&
             isLegalArithImmed((uint32_t)(C - 1))) ||
            (VT == MVT::i64 && C != 0x80000000ULL &&
             isLegalArithImmed(C - 1ULL))) {
          CC = (CC == ISD::SETLT) ? ISD::SETLE : ISD::SETGT;
          C = (VT == MVT::i32) ? (uint32_t)(C - 1) : C - 1;
          RHS = DAG.getConstant(C, dl, VT);
        }
        break;
      case ISD::SETULT:
      case ISD::SETUGE:
        if ((VT == MVT::i32 && C != 0 &&
             isLegalArithImmed((uint32_t)(C - 1))) ||
            (VT == MVT::i64 && C != 0ULL && isLegalArithImmed(C - 1ULL))) {
          CC = (CC == ISD::SETULT) ? ISD::SETULE : ISD::SETUGT;
          C = (VT == MVT::i32) ? (uint32_t)(C - 1) : C - 1;
          RHS = DAG.getConstant(C, dl, VT);
        }
        break;
      case ISD::SETLE:
      case ISD::SETGT:
        if ((VT == MVT::i32 && C != INT32_MAX &&
             isLegalArithImmed((uint32_t)(C + 1))) ||
            (VT == MVT::i64 && C != INT64_MAX &&
             isLegalArithImmed(C + 1ULL))) {
          CC = (CC == ISD::SETLE) ? ISD::SETLT : ISD::SETGE;
          C = (VT == MVT::i32) ? (uint32_t)(C + 1) : C + 1;
          RHS = DAG.getConstant(C, dl, VT);
        }
        break;
      case ISD::SETULE:
      case ISD::SETUGT:
        if ((VT == MVT::i32 && C != UINT32_MAX &&
             isLegalArithImmed((uint32_t)(C + 1))) ||
            (VT == MVT::i64 && C != UINT64_MAX &&
             isLegalArithImmed(C + 1ULL))) {
          CC = (CC == ISD::SETULE) ? ISD::SETULT : ISD::SETUGE;
          C = (VT == MVT::i32) ? (uint32_t)(C + 1) : C + 1;
          RHS = DAG.getConstant(C, dl, VT);
        }
        break;
      }
    }
  }
  SDValue Cmp;
  AArch64CC::CondCode AArch64CC;
  if ((CC == ISD::SETEQ || CC == ISD::SETNE) && isa<ConstantSDNode>(RHS)) {
    const ConstantSDNode *RHSC = cast<ConstantSDNode>(RHS);

    // The imm operand of ADDS is an unsigned immediate, in the range 0 to 4095.
    // For the i8 operand, the largest immediate is 255, so this can be easily
    // encoded in the compare instruction. For the i16 operand, however, the
    // largest immediate cannot be encoded in the compare.
    // Therefore, use a sign extending load and cmn to avoid materializing the
    // -1 constant. For example,
    // movz w1, #65535
    // ldrh w0, [x0, #0]
    // cmp w0, w1
    // >
    // ldrsh w0, [x0, #0]
    // cmn w0, #1
    // Fundamental, we're relying on the property that (zext LHS) == (zext RHS)
    // if and only if (sext LHS) == (sext RHS). The checks are in place to
    // ensure both the LHS and RHS are truly zero extended and to make sure the
    // transformation is profitable.
    if ((RHSC->getZExtValue() >> 16 == 0) && isa<LoadSDNode>(LHS) &&
        cast<LoadSDNode>(LHS)->getExtensionType() == ISD::ZEXTLOAD &&
        cast<LoadSDNode>(LHS)->getMemoryVT() == MVT::i16 &&
        LHS.getNode()->hasNUsesOfValue(1, 0)) {
      int16_t ValueofRHS = cast<ConstantSDNode>(RHS)->getZExtValue();
      if (ValueofRHS < 0 && isLegalArithImmed(-ValueofRHS)) {
        SDValue SExt =
            DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, LHS.getValueType(), LHS,
                        DAG.getValueType(MVT::i16));
        Cmp = emitComparison(SExt, DAG.getConstant(ValueofRHS, dl,
                                                   RHS.getValueType()),
                             CC, dl, DAG);
        AArch64CC = changeIntCCToAArch64CC(CC);
      }
    }

    if (!Cmp && (RHSC->isNullValue() || RHSC->isOne())) {
      if ((Cmp = emitConjunctionDisjunctionTree(DAG, LHS, AArch64CC))) {
        if ((CC == ISD::SETNE) ^ RHSC->isNullValue())
          AArch64CC = AArch64CC::getInvertedCondCode(AArch64CC);
      }
    }
  }

  if (!Cmp) {
    Cmp = emitComparison(LHS, RHS, CC, dl, DAG);
    AArch64CC = changeIntCCToAArch64CC(CC);
  }
  AArch64cc = DAG.getConstant(AArch64CC, dl, MVT_CC);
  return Cmp;
}

static std::pair<SDValue, SDValue>
getAArch64XALUOOp(AArch64CC::CondCode &CC, SDValue Op, SelectionDAG &DAG) {
  assert((Op.getValueType() == MVT::i32 || Op.getValueType() == MVT::i64) &&
         "Unsupported value type");
  SDValue Value, Overflow;
  SDLoc DL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  unsigned Opc = 0;
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("Unknown overflow instruction!");
  case ISD::SADDO:
    Opc = AArch64ISD::ADDS;
    CC = AArch64CC::VS;
    break;
  case ISD::UADDO:
    Opc = AArch64ISD::ADDS;
    CC = AArch64CC::HS;
    break;
  case ISD::SSUBO:
    Opc = AArch64ISD::SUBS;
    CC = AArch64CC::VS;
    break;
  case ISD::USUBO:
    Opc = AArch64ISD::SUBS;
    CC = AArch64CC::LO;
    break;
  // Multiply needs a little bit extra work.
  case ISD::SMULO:
  case ISD::UMULO: {
    CC = AArch64CC::NE;
    bool IsSigned = Op.getOpcode() == ISD::SMULO;
    if (Op.getValueType() == MVT::i32) {
      unsigned ExtendOpc = IsSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
      // For a 32 bit multiply with overflow check we want the instruction
      // selector to generate a widening multiply (SMADDL/UMADDL). For that we
      // need to generate the following pattern:
      // (i64 add 0, (i64 mul (i64 sext|zext i32 %a), (i64 sext|zext i32 %b))
      LHS = DAG.getNode(ExtendOpc, DL, MVT::i64, LHS);
      RHS = DAG.getNode(ExtendOpc, DL, MVT::i64, RHS);
      SDValue Mul = DAG.getNode(ISD::MUL, DL, MVT::i64, LHS, RHS);
      SDValue Add = DAG.getNode(ISD::ADD, DL, MVT::i64, Mul,
                                DAG.getConstant(0, DL, MVT::i64));
      // On AArch64 the upper 32 bits are always zero extended for a 32 bit
      // operation. We need to clear out the upper 32 bits, because we used a
      // widening multiply that wrote all 64 bits. In the end this should be a
      // noop.
      Value = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Add);
      if (IsSigned) {
        // The signed overflow check requires more than just a simple check for
        // any bit set in the upper 32 bits of the result. These bits could be
        // just the sign bits of a negative number. To perform the overflow
        // check we have to arithmetic shift right the 32nd bit of the result by
        // 31 bits. Then we compare the result to the upper 32 bits.
        SDValue UpperBits = DAG.getNode(ISD::SRL, DL, MVT::i64, Add,
                                        DAG.getConstant(32, DL, MVT::i64));
        UpperBits = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, UpperBits);
        SDValue LowerBits = DAG.getNode(ISD::SRA, DL, MVT::i32, Value,
                                        DAG.getConstant(31, DL, MVT::i64));
        // It is important that LowerBits is last, otherwise the arithmetic
        // shift will not be folded into the compare (SUBS).
        SDVTList VTs = DAG.getVTList(MVT::i32, MVT::i32);
        Overflow = DAG.getNode(AArch64ISD::SUBS, DL, VTs, UpperBits, LowerBits)
                       .getValue(1);
      } else {
        // The overflow check for unsigned multiply is easy. We only need to
        // check if any of the upper 32 bits are set. This can be done with a
        // CMP (shifted register). For that we need to generate the following
        // pattern:
        // (i64 AArch64ISD::SUBS i64 0, (i64 srl i64 %Mul, i64 32)
        SDValue UpperBits = DAG.getNode(ISD::SRL, DL, MVT::i64, Mul,
                                        DAG.getConstant(32, DL, MVT::i64));
        SDVTList VTs = DAG.getVTList(MVT::i64, MVT::i32);
        Overflow =
            DAG.getNode(AArch64ISD::SUBS, DL, VTs,
                        DAG.getConstant(0, DL, MVT::i64),
                        UpperBits).getValue(1);
      }
      break;
    }
    assert(Op.getValueType() == MVT::i64 && "Expected an i64 value type");
    // For the 64 bit multiply
    Value = DAG.getNode(ISD::MUL, DL, MVT::i64, LHS, RHS);
    if (IsSigned) {
      SDValue UpperBits = DAG.getNode(ISD::MULHS, DL, MVT::i64, LHS, RHS);
      SDValue LowerBits = DAG.getNode(ISD::SRA, DL, MVT::i64, Value,
                                      DAG.getConstant(63, DL, MVT::i64));
      // It is important that LowerBits is last, otherwise the arithmetic
      // shift will not be folded into the compare (SUBS).
      SDVTList VTs = DAG.getVTList(MVT::i64, MVT::i32);
      Overflow = DAG.getNode(AArch64ISD::SUBS, DL, VTs, UpperBits, LowerBits)
                     .getValue(1);
    } else {
      SDValue UpperBits = DAG.getNode(ISD::MULHU, DL, MVT::i64, LHS, RHS);
      SDVTList VTs = DAG.getVTList(MVT::i64, MVT::i32);
      Overflow =
          DAG.getNode(AArch64ISD::SUBS, DL, VTs,
                      DAG.getConstant(0, DL, MVT::i64),
                      UpperBits).getValue(1);
    }
    break;
  }
  } // switch (...)

  if (Opc) {
    SDVTList VTs = DAG.getVTList(Op->getValueType(0), MVT::i32);

    // Emit the AArch64 operation with overflow check.
    Value = DAG.getNode(Opc, DL, VTs, LHS, RHS);
    Overflow = Value.getValue(1);
  }
  return std::make_pair(Value, Overflow);
}

SDValue AArch64TargetLowering::LowerF128Call(SDValue Op, SelectionDAG &DAG,
                                             RTLIB::Libcall Call) const {
  SmallVector<SDValue, 2> Ops(Op->op_begin(), Op->op_end());
  return makeLibCall(DAG, Call, MVT::f128, Ops, false, SDLoc(Op)).first;
}

static SDValue LowerXOR(SDValue Op, SelectionDAG &DAG) {
  SDValue Sel = Op.getOperand(0);
  SDValue Other = Op.getOperand(1);

  // If neither operand is a SELECT_CC, give up.
  if (Sel.getOpcode() != ISD::SELECT_CC)
    std::swap(Sel, Other);
  if (Sel.getOpcode() != ISD::SELECT_CC)
    return Op;

  // The folding we want to perform is:
  // (xor x, (select_cc a, b, cc, 0, -1) )
  //   -->
  // (csel x, (xor x, -1), cc ...)
  //
  // The latter will get matched to a CSINV instruction.

  ISD::CondCode CC = cast<CondCodeSDNode>(Sel.getOperand(4))->get();
  SDValue LHS = Sel.getOperand(0);
  SDValue RHS = Sel.getOperand(1);
  SDValue TVal = Sel.getOperand(2);
  SDValue FVal = Sel.getOperand(3);
  SDLoc dl(Sel);

  // FIXME: This could be generalized to non-integer comparisons.
  if (LHS.getValueType() != MVT::i32 && LHS.getValueType() != MVT::i64)
    return Op;

  ConstantSDNode *CFVal = dyn_cast<ConstantSDNode>(FVal);
  ConstantSDNode *CTVal = dyn_cast<ConstantSDNode>(TVal);

  // The values aren't constants, this isn't the pattern we're looking for.
  if (!CFVal || !CTVal)
    return Op;

  // We can commute the SELECT_CC by inverting the condition.  This
  // might be needed to make this fit into a CSINV pattern.
  if (CTVal->isAllOnesValue() && CFVal->isNullValue()) {
    std::swap(TVal, FVal);
    std::swap(CTVal, CFVal);
    CC = ISD::getSetCCInverse(CC, true);
  }

  // If the constants line up, perform the transform!
  if (CTVal->isNullValue() && CFVal->isAllOnesValue()) {
    SDValue CCVal;
    SDValue Cmp = getAArch64Cmp(LHS, RHS, CC, CCVal, DAG, dl);

    FVal = Other;
    TVal = DAG.getNode(ISD::XOR, dl, Other.getValueType(), Other,
                       DAG.getConstant(-1ULL, dl, Other.getValueType()));

    return DAG.getNode(AArch64ISD::CSEL, dl, Sel.getValueType(), FVal, TVal,
                       CCVal, Cmp);
  }

  return Op;
}

static SDValue LowerADDC_ADDE_SUBC_SUBE(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();

  // Let legalize expand this if it isn't a legal type yet.
  if (!DAG.getTargetLoweringInfo().isTypeLegal(VT))
    return SDValue();

  SDVTList VTs = DAG.getVTList(VT, MVT::i32);

  unsigned Opc;
  bool ExtraOp = false;
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("Invalid code");
  case ISD::ADDC:
    Opc = AArch64ISD::ADDS;
    break;
  case ISD::SUBC:
    Opc = AArch64ISD::SUBS;
    break;
  case ISD::ADDE:
    Opc = AArch64ISD::ADCS;
    ExtraOp = true;
    break;
  case ISD::SUBE:
    Opc = AArch64ISD::SBCS;
    ExtraOp = true;
    break;
  }

  if (!ExtraOp)
    return DAG.getNode(Opc, SDLoc(Op), VTs, Op.getOperand(0), Op.getOperand(1));
  return DAG.getNode(Opc, SDLoc(Op), VTs, Op.getOperand(0), Op.getOperand(1),
                     Op.getOperand(2));
}

static SDValue LowerXALUO(SDValue Op, SelectionDAG &DAG) {
  // Let legalize expand this if it isn't a legal type yet.
  if (!DAG.getTargetLoweringInfo().isTypeLegal(Op.getValueType()))
    return SDValue();

  SDLoc dl(Op);
  AArch64CC::CondCode CC;
  // The actual operation that sets the overflow or carry flag.
  SDValue Value, Overflow;
  std::tie(Value, Overflow) = getAArch64XALUOOp(CC, Op, DAG);

  // We use 0 and 1 as false and true values.
  SDValue TVal = DAG.getConstant(1, dl, MVT::i32);
  SDValue FVal = DAG.getConstant(0, dl, MVT::i32);

  // We use an inverted condition, because the conditional select is inverted
  // too. This will allow it to be selected to a single instruction:
  // CSINC Wd, WZR, WZR, invert(cond).
  SDValue CCVal = DAG.getConstant(getInvertedCondCode(CC), dl, MVT::i32);
  Overflow = DAG.getNode(AArch64ISD::CSEL, dl, MVT::i32, FVal, TVal,
                         CCVal, Overflow);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::i32);
  return DAG.getNode(ISD::MERGE_VALUES, dl, VTs, Value, Overflow);
}

// Prefetch operands are:
// 1: Address to prefetch
// 2: bool isWrite
// 3: int locality (0 = no locality ... 3 = extreme locality)
// 4: bool isDataCache
static SDValue LowerPREFETCH(SDValue Op, SelectionDAG &DAG) {
  SDLoc DL(Op);
  unsigned IsWrite = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue();
  unsigned Locality = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
  unsigned IsData = cast<ConstantSDNode>(Op.getOperand(4))->getZExtValue();

  bool IsStream = !Locality;
  // When the locality number is set
  if (Locality) {
    // The front-end should have filtered out the out-of-range values
    assert(Locality <= 3 && "Prefetch locality out-of-range");
    // The locality degree is the opposite of the cache speed.
    // Put the number the other way around.
    // The encoding starts at 0 for level 1
    Locality = 3 - Locality;
  }

  // built the mask value encoding the expected behavior.
  unsigned PrfOp = (IsWrite << 4) |     // Load/Store bit
                   (!IsData << 3) |     // IsDataCache bit
                   (Locality << 1) |    // Cache level bits
                   (unsigned)IsStream;  // Stream bit
  return DAG.getNode(AArch64ISD::PREFETCH, DL, MVT::Other, Op.getOperand(0),
                     DAG.getConstant(PrfOp, DL, MVT::i32), Op.getOperand(1));
}

SDValue AArch64TargetLowering::LowerFP_EXTEND(SDValue Op,
                                              SelectionDAG &DAG) const {
  assert(Op.getValueType() == MVT::f128 && "Unexpected lowering");

  RTLIB::Libcall LC;
  LC = RTLIB::getFPEXT(Op.getOperand(0).getValueType(), Op.getValueType());

  return LowerF128Call(Op, DAG, LC);
}

SDValue AArch64TargetLowering::LowerFP_ROUND(SDValue Op,
                                             SelectionDAG &DAG) const {
  if (Op.getOperand(0).getValueType() != MVT::f128) {
    // It's legal except when f128 is involved
    return Op;
  }

  RTLIB::Libcall LC;
  LC = RTLIB::getFPROUND(Op.getOperand(0).getValueType(), Op.getValueType());

  // FP_ROUND node has a second operand indicating whether it is known to be
  // precise. That doesn't take part in the LibCall so we can't directly use
  // LowerF128Call.
  SDValue SrcVal = Op.getOperand(0);
  return makeLibCall(DAG, LC, Op.getValueType(), SrcVal, /*isSigned*/ false,
                     SDLoc(Op)).first;
}

static SDValue LowerVectorFP_TO_INT(SDValue Op, SelectionDAG &DAG) {
  // Warning: We maintain cost tables in AArch64TargetTransformInfo.cpp.
  // Any additional optimization in this function should be recorded
  // in the cost tables.
  EVT InVT = Op.getOperand(0).getValueType();
  EVT VT = Op.getValueType();

  if (VT.getSizeInBits() < InVT.getSizeInBits()) {
    SDLoc dl(Op);
    SDValue Cv =
        DAG.getNode(Op.getOpcode(), dl, InVT.changeVectorElementTypeToInteger(),
                    Op.getOperand(0));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Cv);
  }

  if (VT.getSizeInBits() > InVT.getSizeInBits()) {
    SDLoc dl(Op);
    MVT ExtVT =
        MVT::getVectorVT(MVT::getFloatingPointVT(VT.getScalarSizeInBits()),
                         VT.getVectorNumElements());
    SDValue Ext = DAG.getNode(ISD::FP_EXTEND, dl, ExtVT, Op.getOperand(0));
    return DAG.getNode(Op.getOpcode(), dl, VT, Ext);
  }

  // Type changing conversions are illegal.
  return Op;
}

SDValue AArch64TargetLowering::LowerFP_TO_INT(SDValue Op,
                                              SelectionDAG &DAG) const {
  if (Op.getOperand(0).getValueType().isVector())
    return LowerVectorFP_TO_INT(Op, DAG);

  // f16 conversions are promoted to f32.
  if (Op.getOperand(0).getValueType() == MVT::f16) {
    SDLoc dl(Op);
    return DAG.getNode(
        Op.getOpcode(), dl, Op.getValueType(),
        DAG.getNode(ISD::FP_EXTEND, dl, MVT::f32, Op.getOperand(0)));
  }

  if (Op.getOperand(0).getValueType() != MVT::f128) {
    // It's legal except when f128 is involved
    return Op;
  }

  RTLIB::Libcall LC;
  if (Op.getOpcode() == ISD::FP_TO_SINT)
    LC = RTLIB::getFPTOSINT(Op.getOperand(0).getValueType(), Op.getValueType());
  else
    LC = RTLIB::getFPTOUINT(Op.getOperand(0).getValueType(), Op.getValueType());

  SmallVector<SDValue, 2> Ops(Op->op_begin(), Op->op_end());
  return makeLibCall(DAG, LC, Op.getValueType(), Ops, false, SDLoc(Op)).first;
}

static SDValue LowerVectorINT_TO_FP(SDValue Op, SelectionDAG &DAG) {
  // Warning: We maintain cost tables in AArch64TargetTransformInfo.cpp.
  // Any additional optimization in this function should be recorded
  // in the cost tables.
  EVT VT = Op.getValueType();
  SDLoc dl(Op);
  SDValue In = Op.getOperand(0);
  EVT InVT = In.getValueType();

  if (VT.getSizeInBits() < InVT.getSizeInBits()) {
    MVT CastVT =
        MVT::getVectorVT(MVT::getFloatingPointVT(InVT.getScalarSizeInBits()),
                         InVT.getVectorNumElements());
    In = DAG.getNode(Op.getOpcode(), dl, CastVT, In);
    return DAG.getNode(ISD::FP_ROUND, dl, VT, In, DAG.getIntPtrConstant(0, dl));
  }

  if (VT.getSizeInBits() > InVT.getSizeInBits()) {
    unsigned CastOpc =
        Op.getOpcode() == ISD::SINT_TO_FP ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
    EVT CastVT = VT.changeVectorElementTypeToInteger();
    In = DAG.getNode(CastOpc, dl, CastVT, In);
    return DAG.getNode(Op.getOpcode(), dl, VT, In);
  }

  return Op;
}

SDValue AArch64TargetLowering::LowerINT_TO_FP(SDValue Op,
                                            SelectionDAG &DAG) const {
  if (Op.getValueType().isVector())
    return LowerVectorINT_TO_FP(Op, DAG);

  // f16 conversions are promoted to f32.
  if (Op.getValueType() == MVT::f16) {
    SDLoc dl(Op);
    return DAG.getNode(
        ISD::FP_ROUND, dl, MVT::f16,
        DAG.getNode(Op.getOpcode(), dl, MVT::f32, Op.getOperand(0)),
        DAG.getIntPtrConstant(0, dl));
  }

  // i128 conversions are libcalls.
  if (Op.getOperand(0).getValueType() == MVT::i128)
    return SDValue();

  // Other conversions are legal, unless it's to the completely software-based
  // fp128.
  if (Op.getValueType() != MVT::f128)
    return Op;

  RTLIB::Libcall LC;
  if (Op.getOpcode() == ISD::SINT_TO_FP)
    LC = RTLIB::getSINTTOFP(Op.getOperand(0).getValueType(), Op.getValueType());
  else
    LC = RTLIB::getUINTTOFP(Op.getOperand(0).getValueType(), Op.getValueType());

  return LowerF128Call(Op, DAG, LC);
}

SDValue AArch64TargetLowering::LowerFSINCOS(SDValue Op,
                                            SelectionDAG &DAG) const {
  // For iOS, we want to call an alternative entry point: __sincos_stret,
  // which returns the values in two S / D registers.
  SDLoc dl(Op);
  SDValue Arg = Op.getOperand(0);
  EVT ArgVT = Arg.getValueType();
  Type *ArgTy = ArgVT.getTypeForEVT(*DAG.getContext());

  ArgListTy Args;
  ArgListEntry Entry;

  Entry.Node = Arg;
  Entry.Ty = ArgTy;
  Entry.isSExt = false;
  Entry.isZExt = false;
  Args.push_back(Entry);

  const char *LibcallName =
      (ArgVT == MVT::f64) ? "__sincos_stret" : "__sincosf_stret";
  SDValue Callee =
      DAG.getExternalSymbol(LibcallName, getPointerTy(DAG.getDataLayout()));

  StructType *RetTy = StructType::get(ArgTy, ArgTy, nullptr);
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl).setChain(DAG.getEntryNode())
    .setCallee(CallingConv::Fast, RetTy, Callee, std::move(Args), 0);

  std::pair<SDValue, SDValue> CallResult = LowerCallTo(CLI);
  return CallResult.first;
}

static SDValue LowerBITCAST(SDValue Op, SelectionDAG &DAG) {
  if (Op.getValueType() != MVT::f16)
    return SDValue();

  assert(Op.getOperand(0).getValueType() == MVT::i16);
  SDLoc DL(Op);

  Op = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Op.getOperand(0));
  Op = DAG.getNode(ISD::BITCAST, DL, MVT::f32, Op);
  return SDValue(
      DAG.getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL, MVT::f16, Op,
                         DAG.getTargetConstant(AArch64::hsub, DL, MVT::i32)),
      0);
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

static SDValue addRequiredExtensionForVectorMULL(SDValue N, SelectionDAG &DAG,
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

  return DAG.getNode(ExtOpcode, SDLoc(N), NewVT, N);
}

static bool isExtendedBUILD_VECTOR(SDNode *N, SelectionDAG &DAG,
                                   bool isSigned) {
  EVT VT = N->getValueType(0);

  if (N->getOpcode() != ISD::BUILD_VECTOR)
    return false;

  for (const SDValue &Elt : N->op_values()) {
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

static SDValue skipExtensionForVectorMULL(SDNode *N, SelectionDAG &DAG) {
  if (N->getOpcode() == ISD::SIGN_EXTEND || N->getOpcode() == ISD::ZERO_EXTEND)
    return addRequiredExtensionForVectorMULL(N->getOperand(0), DAG,
                                             N->getOperand(0)->getValueType(0),
                                             N->getValueType(0),
                                             N->getOpcode());

  assert(N->getOpcode() == ISD::BUILD_VECTOR && "expected BUILD_VECTOR");
  EVT VT = N->getValueType(0);
  SDLoc dl(N);
  unsigned EltSize = VT.getVectorElementType().getSizeInBits() / 2;
  unsigned NumElts = VT.getVectorNumElements();
  MVT TruncVT = MVT::getIntegerVT(EltSize);
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0; i != NumElts; ++i) {
    ConstantSDNode *C = cast<ConstantSDNode>(N->getOperand(i));
    const APInt &CInt = C->getAPIntValue();
    // Element types smaller than 32 bits are not legal, so use i32 elements.
    // The values are implicitly truncated so sext vs. zext doesn't matter.
    Ops.push_back(DAG.getConstant(CInt.zextOrTrunc(32), dl, MVT::i32));
  }
  return DAG.getNode(ISD::BUILD_VECTOR, dl,
                     MVT::getVectorVT(TruncVT, NumElts), Ops);
}

static bool isSignExtended(SDNode *N, SelectionDAG &DAG) {
  if (N->getOpcode() == ISD::SIGN_EXTEND)
    return true;
  if (isExtendedBUILD_VECTOR(N, DAG, true))
    return true;
  return false;
}

static bool isZeroExtended(SDNode *N, SelectionDAG &DAG) {
  if (N->getOpcode() == ISD::ZERO_EXTEND)
    return true;
  if (isExtendedBUILD_VECTOR(N, DAG, false))
    return true;
  return false;
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
    NewOpc = AArch64ISD::SMULL;
  else {
    bool isN0ZExt = isZeroExtended(N0, DAG);
    bool isN1ZExt = isZeroExtended(N1, DAG);
    if (isN0ZExt && isN1ZExt)
      NewOpc = AArch64ISD::UMULL;
    else if (isN1SExt || isN1ZExt) {
      // Look for (s/zext A + s/zext B) * (s/zext C). We want to turn these
      // into (s/zext A * s/zext C) + (s/zext B * s/zext C)
      if (isN1SExt && isAddSubSExt(N0, DAG)) {
        NewOpc = AArch64ISD::SMULL;
        isMLA = true;
      } else if (isN1ZExt && isAddSubZExt(N0, DAG)) {
        NewOpc =  AArch64ISD::UMULL;
        isMLA = true;
      } else if (isN0ZExt && isAddSubZExt(N1, DAG)) {
        std::swap(N0, N1);
        NewOpc =  AArch64ISD::UMULL;
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

  // Legalize to a S/UMULL instruction
  SDLoc DL(Op);
  SDValue Op0;
  SDValue Op1 = skipExtensionForVectorMULL(N1, DAG);
  if (!isMLA) {
    Op0 = skipExtensionForVectorMULL(N0, DAG);
    assert(Op0.getValueType().is64BitVector() &&
           Op1.getValueType().is64BitVector() &&
           "unexpected types for extended operands to VMULL");
    return DAG.getNode(NewOpc, DL, VT, Op0, Op1);
  }
  // Optimizing (zext A + zext B) * C, to (S/UMULL A, C) + (S/UMULL B, C) during
  // isel lowering to take advantage of no-stall back to back s/umul + s/umla.
  // This is true for CPUs with accumulate forwarding such as Cortex-A53/A57
  SDValue N00 = skipExtensionForVectorMULL(N0->getOperand(0).getNode(), DAG);
  SDValue N01 = skipExtensionForVectorMULL(N0->getOperand(1).getNode(), DAG);
  EVT Op1VT = Op1.getValueType();
  return DAG.getNode(N0->getOpcode(), DL, VT,
                     DAG.getNode(NewOpc, DL, VT,
                               DAG.getNode(ISD::BITCAST, DL, Op1VT, N00), Op1),
                     DAG.getNode(NewOpc, DL, VT,
                               DAG.getNode(ISD::BITCAST, DL, Op1VT, N01), Op1));
}

SDValue AArch64TargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                     SelectionDAG &DAG) const {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDLoc dl(Op);
  switch (IntNo) {
  default: return SDValue();    // Don't custom lower most intrinsics.
  case Intrinsic::aarch64_thread_pointer: {
    EVT PtrVT = getPointerTy(DAG.getDataLayout());
    return DAG.getNode(AArch64ISD::THREAD_POINTER, dl, PtrVT);
  }
  case Intrinsic::aarch64_neon_smax:
    return DAG.getNode(ISD::SMAX, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::aarch64_neon_umax:
    return DAG.getNode(ISD::UMAX, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::aarch64_neon_smin:
    return DAG.getNode(ISD::SMIN, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::aarch64_neon_umin:
    return DAG.getNode(ISD::UMIN, dl, Op.getValueType(),
                       Op.getOperand(1), Op.getOperand(2));
  }
}

SDValue AArch64TargetLowering::LowerOperation(SDValue Op,
                                              SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("unimplemented operand");
    return SDValue();
  case ISD::BITCAST:
    return LowerBITCAST(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:
    return LowerGlobalTLSAddress(Op, DAG);
  case ISD::SETCC:
    return LowerSETCC(Op, DAG);
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::SELECT:
    return LowerSELECT(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG);
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG);
  case ISD::BlockAddress:
    return LowerBlockAddress(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::VACOPY:
    return LowerVACOPY(Op, DAG);
  case ISD::VAARG:
    return LowerVAARG(Op, DAG);
  case ISD::ADDC:
  case ISD::ADDE:
  case ISD::SUBC:
  case ISD::SUBE:
    return LowerADDC_ADDE_SUBC_SUBE(Op, DAG);
  case ISD::SADDO:
  case ISD::UADDO:
  case ISD::SSUBO:
  case ISD::USUBO:
  case ISD::SMULO:
  case ISD::UMULO:
    return LowerXALUO(Op, DAG);
  case ISD::FADD:
    return LowerF128Call(Op, DAG, RTLIB::ADD_F128);
  case ISD::FSUB:
    return LowerF128Call(Op, DAG, RTLIB::SUB_F128);
  case ISD::FMUL:
    return LowerF128Call(Op, DAG, RTLIB::MUL_F128);
  case ISD::FDIV:
    return LowerF128Call(Op, DAG, RTLIB::DIV_F128);
  case ISD::FP_ROUND:
    return LowerFP_ROUND(Op, DAG);
  case ISD::FP_EXTEND:
    return LowerFP_EXTEND(Op, DAG);
  case ISD::FRAMEADDR:
    return LowerFRAMEADDR(Op, DAG);
  case ISD::RETURNADDR:
    return LowerRETURNADDR(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::BUILD_VECTOR:
    return LowerBUILD_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:
    return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::EXTRACT_SUBVECTOR:
    return LowerEXTRACT_SUBVECTOR(Op, DAG);
  case ISD::SRA:
  case ISD::SRL:
  case ISD::SHL:
    return LowerVectorSRA_SRL_SHL(Op, DAG);
  case ISD::SHL_PARTS:
    return LowerShiftLeftParts(Op, DAG);
  case ISD::SRL_PARTS:
  case ISD::SRA_PARTS:
    return LowerShiftRightParts(Op, DAG);
  case ISD::CTPOP:
    return LowerCTPOP(Op, DAG);
  case ISD::FCOPYSIGN:
    return LowerFCOPYSIGN(Op, DAG);
  case ISD::AND:
    return LowerVectorAND(Op, DAG);
  case ISD::OR:
    return LowerVectorOR(Op, DAG);
  case ISD::XOR:
    return LowerXOR(Op, DAG);
  case ISD::PREFETCH:
    return LowerPREFETCH(Op, DAG);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return LowerINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return LowerFP_TO_INT(Op, DAG);
  case ISD::FSINCOS:
    return LowerFSINCOS(Op, DAG);
  case ISD::MUL:
    return LowerMUL(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN:
    return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  }
}

/// getFunctionAlignment - Return the Log2 alignment of this function.
unsigned AArch64TargetLowering::getFunctionAlignment(const Function *F) const {
  return 2;
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "AArch64GenCallingConv.inc"

/// Selects the correct CCAssignFn for a given CallingConvention value.
CCAssignFn *AArch64TargetLowering::CCAssignFnForCall(CallingConv::ID CC,
                                                     bool IsVarArg) const {
  switch (CC) {
  default:
    llvm_unreachable("Unsupported calling convention.");
  case CallingConv::WebKit_JS:
    return CC_AArch64_WebKit_JS;
  case CallingConv::GHC:
    return CC_AArch64_GHC;
  case CallingConv::C:
  case CallingConv::Fast:
    if (!Subtarget->isTargetDarwin())
      return CC_AArch64_AAPCS;
    return IsVarArg ? CC_AArch64_DarwinPCS_VarArg : CC_AArch64_DarwinPCS;
  }
}

SDValue AArch64TargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc DL, SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  // At this point, Ins[].VT may already be promoted to i32. To correctly
  // handle passing i8 as i8 instead of i32 on stack, we pass in both i32 and
  // i8 to CC_AArch64_AAPCS with i32 being ValVT and i8 being LocVT.
  // Since AnalyzeFormalArguments uses Ins[].VT for both ValVT and LocVT, here
  // we use a special version of AnalyzeFormalArguments to pass in ValVT and
  // LocVT.
  unsigned NumArgs = Ins.size();
  Function::const_arg_iterator CurOrigArg = MF.getFunction()->arg_begin();
  unsigned CurArgIdx = 0;
  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ValVT = Ins[i].VT;
    if (Ins[i].isOrigArg()) {
      std::advance(CurOrigArg, Ins[i].getOrigArgIndex() - CurArgIdx);
      CurArgIdx = Ins[i].getOrigArgIndex();

      // Get type of the original argument.
      EVT ActualVT = getValueType(DAG.getDataLayout(), CurOrigArg->getType(),
                                  /*AllowUnknown*/ true);
      MVT ActualMVT = ActualVT.isSimple() ? ActualVT.getSimpleVT() : MVT::Other;
      // If ActualMVT is i1/i8/i16, we should set LocVT to i8/i8/i16.
      if (ActualMVT == MVT::i1 || ActualMVT == MVT::i8)
        ValVT = MVT::i8;
      else if (ActualMVT == MVT::i16)
        ValVT = MVT::i16;
    }
    CCAssignFn *AssignFn = CCAssignFnForCall(CallConv, /*IsVarArg=*/false);
    bool Res =
        AssignFn(i, ValVT, ValVT, CCValAssign::Full, Ins[i].Flags, CCInfo);
    assert(!Res && "Call operand has unhandled type");
    (void)Res;
  }
  assert(ArgLocs.size() == Ins.size());
  SmallVector<SDValue, 16> ArgValues;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    if (Ins[i].Flags.isByVal()) {
      // Byval is used for HFAs in the PCS, but the system should work in a
      // non-compliant manner for larger structs.
      EVT PtrVT = getPointerTy(DAG.getDataLayout());
      int Size = Ins[i].Flags.getByValSize();
      unsigned NumRegs = (Size + 7) / 8;

      // FIXME: This works on big-endian for composite byvals, which are the common
      // case. It should also work for fundamental types too.
      unsigned FrameIdx =
        MFI->CreateFixedObject(8 * NumRegs, VA.getLocMemOffset(), false);
      SDValue FrameIdxN = DAG.getFrameIndex(FrameIdx, PtrVT);
      InVals.push_back(FrameIdxN);

      continue;
    }
    
    if (VA.isRegLoc()) {
      // Arguments stored in registers.
      EVT RegVT = VA.getLocVT();

      SDValue ArgValue;
      const TargetRegisterClass *RC;

      if (RegVT == MVT::i32)
        RC = &AArch64::GPR32RegClass;
      else if (RegVT == MVT::i64)
        RC = &AArch64::GPR64RegClass;
      else if (RegVT == MVT::f16)
        RC = &AArch64::FPR16RegClass;
      else if (RegVT == MVT::f32)
        RC = &AArch64::FPR32RegClass;
      else if (RegVT == MVT::f64 || RegVT.is64BitVector())
        RC = &AArch64::FPR64RegClass;
      else if (RegVT == MVT::f128 || RegVT.is128BitVector())
        RC = &AArch64::FPR128RegClass;
      else
        llvm_unreachable("RegVT not supported by FORMAL_ARGUMENTS Lowering");

      // Transform the arguments in physical registers into virtual ones.
      unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
      ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegVT);

      // If this is an 8, 16 or 32-bit value, it is really passed promoted
      // to 64 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      switch (VA.getLocInfo()) {
      default:
        llvm_unreachable("Unknown loc info!");
      case CCValAssign::Full:
        break;
      case CCValAssign::BCvt:
        ArgValue = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::AExt:
      case CCValAssign::SExt:
      case CCValAssign::ZExt:
        // SelectionDAGBuilder will insert appropriate AssertZExt & AssertSExt
        // nodes after our lowering.
        assert(RegVT == Ins[i].VT && "incorrect register location selected");
        break;
      }

      InVals.push_back(ArgValue);

    } else { // VA.isRegLoc()
      assert(VA.isMemLoc() && "CCValAssign is neither reg nor mem");
      unsigned ArgOffset = VA.getLocMemOffset();
      unsigned ArgSize = VA.getValVT().getSizeInBits() / 8;

      uint32_t BEAlign = 0;
      if (!Subtarget->isLittleEndian() && ArgSize < 8 &&
          !Ins[i].Flags.isInConsecutiveRegs())
        BEAlign = 8 - ArgSize;

      int FI = MFI->CreateFixedObject(ArgSize, ArgOffset + BEAlign, true);

      // Create load nodes to retrieve arguments from the stack.
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue ArgValue;

      // For NON_EXTLOAD, generic code in getLoad assert(ValVT == MemVT)
      ISD::LoadExtType ExtType = ISD::NON_EXTLOAD;
      MVT MemVT = VA.getValVT();

      switch (VA.getLocInfo()) {
      default:
        break;
      case CCValAssign::BCvt:
        MemVT = VA.getLocVT();
        break;
      case CCValAssign::SExt:
        ExtType = ISD::SEXTLOAD;
        break;
      case CCValAssign::ZExt:
        ExtType = ISD::ZEXTLOAD;
        break;
      case CCValAssign::AExt:
        ExtType = ISD::EXTLOAD;
        break;
      }

      ArgValue = DAG.getExtLoad(
          ExtType, DL, VA.getLocVT(), Chain, FIN,
          MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI),
          MemVT, false, false, false, 0);

      InVals.push_back(ArgValue);
    }
  }

  // varargs
  if (isVarArg) {
    if (!Subtarget->isTargetDarwin()) {
      // The AAPCS variadic function ABI is identical to the non-variadic
      // one. As a result there may be more arguments in registers and we should
      // save them for future reference.
      saveVarArgRegisters(CCInfo, DAG, DL, Chain);
    }

    AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
    // This will point to the next argument passed via stack.
    unsigned StackOffset = CCInfo.getNextStackOffset();
    // We currently pass all varargs at 8-byte alignment.
    StackOffset = ((StackOffset + 7) & ~7);
    AFI->setVarArgsStackIndex(MFI->CreateFixedObject(4, StackOffset, true));
  }

  AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
  unsigned StackArgSize = CCInfo.getNextStackOffset();
  bool TailCallOpt = MF.getTarget().Options.GuaranteedTailCallOpt;
  if (DoesCalleeRestoreStack(CallConv, TailCallOpt)) {
    // This is a non-standard ABI so by fiat I say we're allowed to make full
    // use of the stack area to be popped, which must be aligned to 16 bytes in
    // any case:
    StackArgSize = RoundUpToAlignment(StackArgSize, 16);

    // If we're expected to restore the stack (e.g. fastcc) then we'll be adding
    // a multiple of 16.
    FuncInfo->setArgumentStackToRestore(StackArgSize);

    // This realignment carries over to the available bytes below. Our own
    // callers will guarantee the space is free by giving an aligned value to
    // CALLSEQ_START.
  }
  // Even if we're not expected to free up the space, it's useful to know how
  // much is there while considering tail calls (because we can reuse it).
  FuncInfo->setBytesInStackArgArea(StackArgSize);

  return Chain;
}

void AArch64TargetLowering::saveVarArgRegisters(CCState &CCInfo,
                                                SelectionDAG &DAG, SDLoc DL,
                                                SDValue &Chain) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
  auto PtrVT = getPointerTy(DAG.getDataLayout());

  SmallVector<SDValue, 8> MemOps;

  static const MCPhysReg GPRArgRegs[] = { AArch64::X0, AArch64::X1, AArch64::X2,
                                          AArch64::X3, AArch64::X4, AArch64::X5,
                                          AArch64::X6, AArch64::X7 };
  static const unsigned NumGPRArgRegs = array_lengthof(GPRArgRegs);
  unsigned FirstVariadicGPR = CCInfo.getFirstUnallocated(GPRArgRegs);

  unsigned GPRSaveSize = 8 * (NumGPRArgRegs - FirstVariadicGPR);
  int GPRIdx = 0;
  if (GPRSaveSize != 0) {
    GPRIdx = MFI->CreateStackObject(GPRSaveSize, 8, false);

    SDValue FIN = DAG.getFrameIndex(GPRIdx, PtrVT);

    for (unsigned i = FirstVariadicGPR; i < NumGPRArgRegs; ++i) {
      unsigned VReg = MF.addLiveIn(GPRArgRegs[i], &AArch64::GPR64RegClass);
      SDValue Val = DAG.getCopyFromReg(Chain, DL, VReg, MVT::i64);
      SDValue Store = DAG.getStore(
          Val.getValue(1), DL, Val, FIN,
          MachinePointerInfo::getStack(DAG.getMachineFunction(), i * 8), false,
          false, 0);
      MemOps.push_back(Store);
      FIN =
          DAG.getNode(ISD::ADD, DL, PtrVT, FIN, DAG.getConstant(8, DL, PtrVT));
    }
  }
  FuncInfo->setVarArgsGPRIndex(GPRIdx);
  FuncInfo->setVarArgsGPRSize(GPRSaveSize);

  if (Subtarget->hasFPARMv8()) {
    static const MCPhysReg FPRArgRegs[] = {
        AArch64::Q0, AArch64::Q1, AArch64::Q2, AArch64::Q3,
        AArch64::Q4, AArch64::Q5, AArch64::Q6, AArch64::Q7};
    static const unsigned NumFPRArgRegs = array_lengthof(FPRArgRegs);
    unsigned FirstVariadicFPR = CCInfo.getFirstUnallocated(FPRArgRegs);

    unsigned FPRSaveSize = 16 * (NumFPRArgRegs - FirstVariadicFPR);
    int FPRIdx = 0;
    if (FPRSaveSize != 0) {
      FPRIdx = MFI->CreateStackObject(FPRSaveSize, 16, false);

      SDValue FIN = DAG.getFrameIndex(FPRIdx, PtrVT);

      for (unsigned i = FirstVariadicFPR; i < NumFPRArgRegs; ++i) {
        unsigned VReg = MF.addLiveIn(FPRArgRegs[i], &AArch64::FPR128RegClass);
        SDValue Val = DAG.getCopyFromReg(Chain, DL, VReg, MVT::f128);

        SDValue Store = DAG.getStore(
            Val.getValue(1), DL, Val, FIN,
            MachinePointerInfo::getStack(DAG.getMachineFunction(), i * 16),
            false, false, 0);
        MemOps.push_back(Store);
        FIN = DAG.getNode(ISD::ADD, DL, PtrVT, FIN,
                          DAG.getConstant(16, DL, PtrVT));
      }
    }
    FuncInfo->setVarArgsFPRIndex(FPRIdx);
    FuncInfo->setVarArgsFPRSize(FPRSaveSize);
  }

  if (!MemOps.empty()) {
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOps);
  }
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
SDValue AArch64TargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc DL, SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals, bool isThisReturn,
    SDValue ThisVal) const {
  CCAssignFn *RetCC = CallConv == CallingConv::WebKit_JS
                          ? RetCC_AArch64_WebKit_JS
                          : RetCC_AArch64_AAPCS;
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];

    // Pass 'this' value directly from the argument to return value, to avoid
    // reg unit interference
    if (i == 0 && isThisReturn) {
      assert(!VA.needsCustom() && VA.getLocVT() == MVT::i64 &&
             "unexpected return calling convention register assignment");
      InVals.push_back(ThisVal);
      continue;
    }

    SDValue Val =
        DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), InFlag);
    Chain = Val.getValue(1);
    InFlag = Val.getValue(2);

    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
      break;
    }

    InVals.push_back(Val);
  }

  return Chain;
}

bool AArch64TargetLowering::isEligibleForTailCallOptimization(
    SDValue Callee, CallingConv::ID CalleeCC, bool isVarArg,
    bool isCalleeStructRet, bool isCallerStructRet,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG &DAG) const {
  // For CallingConv::C this function knows whether the ABI needs
  // changing. That's not true for other conventions so they will have to opt in
  // manually.
  if (!IsTailCallConvention(CalleeCC) && CalleeCC != CallingConv::C)
    return false;

  const MachineFunction &MF = DAG.getMachineFunction();
  const Function *CallerF = MF.getFunction();
  CallingConv::ID CallerCC = CallerF->getCallingConv();
  bool CCMatch = CallerCC == CalleeCC;

  // Byval parameters hand the function a pointer directly into the stack area
  // we want to reuse during a tail call. Working around this *is* possible (see
  // X86) but less efficient and uglier in LowerCall.
  for (Function::const_arg_iterator i = CallerF->arg_begin(),
                                    e = CallerF->arg_end();
       i != e; ++i)
    if (i->hasByValAttr())
      return false;

  if (getTargetMachine().Options.GuaranteedTailCallOpt) {
    if (IsTailCallConvention(CalleeCC) && CCMatch)
      return true;
    return false;
  }

  // Externally-defined functions with weak linkage should not be
  // tail-called on AArch64 when the OS does not support dynamic
  // pre-emption of symbols, as the AAELF spec requires normal calls
  // to undefined weak functions to be replaced with a NOP or jump to the
  // next instruction. The behaviour of branch instructions in this
  // situation (as used for tail calls) is implementation-defined, so we
  // cannot rely on the linker replacing the tail call with a return.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    const Triple &TT = getTargetMachine().getTargetTriple();
    if (GV->hasExternalWeakLinkage() &&
        (!TT.isOSWindows() || TT.isOSBinFormatELF() || TT.isOSBinFormatMachO()))
      return false;
  }

  // Now we search for cases where we can use a tail call without changing the
  // ABI. Sibcall is used in some places (particularly gcc) to refer to this
  // concept.

  // I want anyone implementing a new calling convention to think long and hard
  // about this assert.
  assert((!isVarArg || CalleeCC == CallingConv::C) &&
         "Unexpected variadic calling convention");

  if (isVarArg && !Outs.empty()) {
    // At least two cases here: if caller is fastcc then we can't have any
    // memory arguments (we'd be expected to clean up the stack afterwards). If
    // caller is C then we could potentially use its argument area.

    // FIXME: for now we take the most conservative of these in both cases:
    // disallow all variadic memory operands.
    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CalleeCC, isVarArg, DAG.getMachineFunction(), ArgLocs,
                   *DAG.getContext());

    CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForCall(CalleeCC, true));
    for (const CCValAssign &ArgLoc : ArgLocs)
      if (!ArgLoc.isRegLoc())
        return false;
  }

  // If the calling conventions do not match, then we'd better make sure the
  // results are returned in the same way as what the caller expects.
  if (!CCMatch) {
    SmallVector<CCValAssign, 16> RVLocs1;
    CCState CCInfo1(CalleeCC, false, DAG.getMachineFunction(), RVLocs1,
                    *DAG.getContext());
    CCInfo1.AnalyzeCallResult(Ins, CCAssignFnForCall(CalleeCC, isVarArg));

    SmallVector<CCValAssign, 16> RVLocs2;
    CCState CCInfo2(CallerCC, false, DAG.getMachineFunction(), RVLocs2,
                    *DAG.getContext());
    CCInfo2.AnalyzeCallResult(Ins, CCAssignFnForCall(CallerCC, isVarArg));

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

  // Nothing more to check if the callee is taking no arguments
  if (Outs.empty())
    return true;

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CalleeCC, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForCall(CalleeCC, isVarArg));

  const AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();

  // If the stack arguments for this call would fit into our own save area then
  // the call can be made tail.
  return CCInfo.getNextStackOffset() <= FuncInfo->getBytesInStackArgArea();
}

SDValue AArch64TargetLowering::addTokenForArgument(SDValue Chain,
                                                   SelectionDAG &DAG,
                                                   MachineFrameInfo *MFI,
                                                   int ClobberedFI) const {
  SmallVector<SDValue, 8> ArgChains;
  int64_t FirstByte = MFI->getObjectOffset(ClobberedFI);
  int64_t LastByte = FirstByte + MFI->getObjectSize(ClobberedFI) - 1;

  // Include the original chain at the beginning of the list. When this is
  // used by target LowerCall hooks, this helps legalize find the
  // CALLSEQ_BEGIN node.
  ArgChains.push_back(Chain);

  // Add a chain value for each stack argument corresponding
  for (SDNode::use_iterator U = DAG.getEntryNode().getNode()->use_begin(),
                            UE = DAG.getEntryNode().getNode()->use_end();
       U != UE; ++U)
    if (LoadSDNode *L = dyn_cast<LoadSDNode>(*U))
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(L->getBasePtr()))
        if (FI->getIndex() < 0) {
          int64_t InFirstByte = MFI->getObjectOffset(FI->getIndex());
          int64_t InLastByte = InFirstByte;
          InLastByte += MFI->getObjectSize(FI->getIndex()) - 1;

          if ((InFirstByte <= FirstByte && FirstByte <= InLastByte) ||
              (FirstByte <= InFirstByte && InFirstByte <= LastByte))
            ArgChains.push_back(SDValue(L, 1));
        }

  // Build a tokenfactor for all the chains.
  return DAG.getNode(ISD::TokenFactor, SDLoc(Chain), MVT::Other, ArgChains);
}

bool AArch64TargetLowering::DoesCalleeRestoreStack(CallingConv::ID CallCC,
                                                   bool TailCallOpt) const {
  return CallCC == CallingConv::Fast && TailCallOpt;
}

bool AArch64TargetLowering::IsTailCallConvention(CallingConv::ID CallCC) const {
  return CallCC == CallingConv::Fast;
}

/// LowerCall - Lower a call to a callseq_start + CALL + callseq_end chain,
/// and add input and output parameter nodes.
SDValue
AArch64TargetLowering::LowerCall(CallLoweringInfo &CLI,
                                 SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  bool IsStructRet = (Outs.empty()) ? false : Outs[0].Flags.isSRet();
  bool IsThisReturn = false;

  AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
  bool TailCallOpt = MF.getTarget().Options.GuaranteedTailCallOpt;
  bool IsSibCall = false;

  if (IsTailCall) {
    // Check if it's really possible to do a tail call.
    IsTailCall = isEligibleForTailCallOptimization(
        Callee, CallConv, IsVarArg, IsStructRet,
        MF.getFunction()->hasStructRetAttr(), Outs, OutVals, Ins, DAG);
    if (!IsTailCall && CLI.CS && CLI.CS->isMustTailCall())
      report_fatal_error("failed to perform tail call elimination on a call "
                         "site marked musttail");

    // A sibling call is one where we're under the usual C ABI and not planning
    // to change that but can still do a tail call:
    if (!TailCallOpt && IsTailCall)
      IsSibCall = true;

    if (IsTailCall)
      ++NumTailCalls;
  }

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  if (IsVarArg) {
    // Handle fixed and variable vector arguments differently.
    // Variable vector arguments always go into memory.
    unsigned NumArgs = Outs.size();

    for (unsigned i = 0; i != NumArgs; ++i) {
      MVT ArgVT = Outs[i].VT;
      ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
      CCAssignFn *AssignFn = CCAssignFnForCall(CallConv,
                                               /*IsVarArg=*/ !Outs[i].IsFixed);
      bool Res = AssignFn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, CCInfo);
      assert(!Res && "Call operand has unhandled type");
      (void)Res;
    }
  } else {
    // At this point, Outs[].VT may already be promoted to i32. To correctly
    // handle passing i8 as i8 instead of i32 on stack, we pass in both i32 and
    // i8 to CC_AArch64_AAPCS with i32 being ValVT and i8 being LocVT.
    // Since AnalyzeCallOperands uses Ins[].VT for both ValVT and LocVT, here
    // we use a special version of AnalyzeCallOperands to pass in ValVT and
    // LocVT.
    unsigned NumArgs = Outs.size();
    for (unsigned i = 0; i != NumArgs; ++i) {
      MVT ValVT = Outs[i].VT;
      // Get type of the original argument.
      EVT ActualVT = getValueType(DAG.getDataLayout(),
                                  CLI.getArgs()[Outs[i].OrigArgIndex].Ty,
                                  /*AllowUnknown*/ true);
      MVT ActualMVT = ActualVT.isSimple() ? ActualVT.getSimpleVT() : ValVT;
      ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
      // If ActualMVT is i1/i8/i16, we should set LocVT to i8/i8/i16.
      if (ActualMVT == MVT::i1 || ActualMVT == MVT::i8)
        ValVT = MVT::i8;
      else if (ActualMVT == MVT::i16)
        ValVT = MVT::i16;

      CCAssignFn *AssignFn = CCAssignFnForCall(CallConv, /*IsVarArg=*/false);
      bool Res = AssignFn(i, ValVT, ValVT, CCValAssign::Full, ArgFlags, CCInfo);
      assert(!Res && "Call operand has unhandled type");
      (void)Res;
    }
  }

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  if (IsSibCall) {
    // Since we're not changing the ABI to make this a tail call, the memory
    // operands are already available in the caller's incoming argument space.
    NumBytes = 0;
  }

  // FPDiff is the byte offset of the call's argument area from the callee's.
  // Stores to callee stack arguments will be placed in FixedStackSlots offset
  // by this amount for a tail call. In a sibling call it must be 0 because the
  // caller will deallocate the entire stack and the callee still expects its
  // arguments to begin at SP+0. Completely unused for non-tail calls.
  int FPDiff = 0;

  if (IsTailCall && !IsSibCall) {
    unsigned NumReusableBytes = FuncInfo->getBytesInStackArgArea();

    // Since callee will pop argument stack as a tail call, we must keep the
    // popped size 16-byte aligned.
    NumBytes = RoundUpToAlignment(NumBytes, 16);

    // FPDiff will be negative if this tail call requires more space than we
    // would automatically have in our incoming argument space. Positive if we
    // can actually shrink the stack.
    FPDiff = NumReusableBytes - NumBytes;

    // The stack pointer must be 16-byte aligned at all times it's used for a
    // memory operation, which in practice means at *all* times and in
    // particular across call boundaries. Therefore our own arguments started at
    // a 16-byte aligned SP and the delta applied for the tail call should
    // satisfy the same constraint.
    assert(FPDiff % 16 == 0 && "unaligned stack on tail call");
  }

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  if (!IsSibCall)
    Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, DL,
                                                              true),
                                 DL);

  SDValue StackPtr = DAG.getCopyFromReg(Chain, DL, AArch64::SP,
                                        getPointerTy(DAG.getDataLayout()));

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  auto PtrVT = getPointerTy(DAG.getDataLayout());

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, realArgIdx = 0, e = ArgLocs.size(); i != e;
       ++i, ++realArgIdx) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[realArgIdx];
    ISD::ArgFlagsTy Flags = Outs[realArgIdx].Flags;

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      if (Outs[realArgIdx].ArgVT == MVT::i1) {
        // AAPCS requires i1 to be zero-extended to 8-bits by the caller.
        Arg = DAG.getNode(ISD::TRUNCATE, DL, MVT::i1, Arg);
        Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i8, Arg);
      }
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::FPExt:
      Arg = DAG.getNode(ISD::FP_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    }

    if (VA.isRegLoc()) {
      if (realArgIdx == 0 && Flags.isReturned() && Outs[0].VT == MVT::i64) {
        assert(VA.getLocVT() == MVT::i64 &&
               "unexpected calling convention register assignment");
        assert(!Ins.empty() && Ins[0].VT == MVT::i64 &&
               "unexpected use of 'returned'");
        IsThisReturn = true;
      }
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());

      SDValue DstAddr;
      MachinePointerInfo DstInfo;

      // FIXME: This works on big-endian for composite byvals, which are the
      // common case. It should also work for fundamental types too.
      uint32_t BEAlign = 0;
      unsigned OpSize = Flags.isByVal() ? Flags.getByValSize() * 8
                                        : VA.getValVT().getSizeInBits();
      OpSize = (OpSize + 7) / 8;
      if (!Subtarget->isLittleEndian() && !Flags.isByVal() &&
          !Flags.isInConsecutiveRegs()) {
        if (OpSize < 8)
          BEAlign = 8 - OpSize;
      }
      unsigned LocMemOffset = VA.getLocMemOffset();
      int32_t Offset = LocMemOffset + BEAlign;
      SDValue PtrOff = DAG.getIntPtrConstant(Offset, DL);
      PtrOff = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr, PtrOff);

      if (IsTailCall) {
        Offset = Offset + FPDiff;
        int FI = MF.getFrameInfo()->CreateFixedObject(OpSize, Offset, true);

        DstAddr = DAG.getFrameIndex(FI, PtrVT);
        DstInfo =
            MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI);

        // Make sure any stack arguments overlapping with where we're storing
        // are loaded before this eventual operation. Otherwise they'll be
        // clobbered.
        Chain = addTokenForArgument(Chain, DAG, MF.getFrameInfo(), FI);
      } else {
        SDValue PtrOff = DAG.getIntPtrConstant(Offset, DL);

        DstAddr = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr, PtrOff);
        DstInfo = MachinePointerInfo::getStack(DAG.getMachineFunction(),
                                               LocMemOffset);
      }

      if (Outs[i].Flags.isByVal()) {
        SDValue SizeNode =
            DAG.getConstant(Outs[i].Flags.getByValSize(), DL, MVT::i64);
        SDValue Cpy = DAG.getMemcpy(
            Chain, DL, DstAddr, Arg, SizeNode, Outs[i].Flags.getByValAlign(),
            /*isVol = */ false, /*AlwaysInline = */ false,
            /*isTailCall = */ false,
            DstInfo, MachinePointerInfo());

        MemOpChains.push_back(Cpy);
      } else {
        // Since we pass i1/i8/i16 as i1/i8/i16 on stack and Arg is already
        // promoted to a legal register type i32, we should truncate Arg back to
        // i1/i8/i16.
        if (VA.getValVT() == MVT::i1 || VA.getValVT() == MVT::i8 ||
            VA.getValVT() == MVT::i16)
          Arg = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Arg);

        SDValue Store =
            DAG.getStore(Chain, DL, Arg, DstAddr, DstInfo, false, false, 0);
        MemOpChains.push_back(Store);
      }
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (auto &RegToPass : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, RegToPass.first,
                             RegToPass.second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  if (getTargetMachine().getCodeModel() == CodeModel::Large &&
      Subtarget->isTargetMachO()) {
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
      const GlobalValue *GV = G->getGlobal();
      bool InternalLinkage = GV->hasInternalLinkage();
      if (InternalLinkage)
        Callee = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, 0);
      else {
        Callee =
            DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, AArch64II::MO_GOT);
        Callee = DAG.getNode(AArch64ISD::LOADgot, DL, PtrVT, Callee);
      }
    } else if (ExternalSymbolSDNode *S =
                   dyn_cast<ExternalSymbolSDNode>(Callee)) {
      const char *Sym = S->getSymbol();
      Callee = DAG.getTargetExternalSymbol(Sym, PtrVT, AArch64II::MO_GOT);
      Callee = DAG.getNode(AArch64ISD::LOADgot, DL, PtrVT, Callee);
    }
  } else if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    Callee = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, 0);
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    const char *Sym = S->getSymbol();
    Callee = DAG.getTargetExternalSymbol(Sym, PtrVT, 0);
  }

  // We don't usually want to end the call-sequence here because we would tidy
  // the frame up *after* the call, however in the ABI-changing tail-call case
  // we've carefully laid out the parameters so that when sp is reset they'll be
  // in the correct location.
  if (IsTailCall && !IsSibCall) {
    Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, DL, true),
                               DAG.getIntPtrConstant(0, DL, true), InFlag, DL);
    InFlag = Chain.getValue(1);
  }

  std::vector<SDValue> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  if (IsTailCall) {
    // Each tail call may have to adjust the stack by a different amount, so
    // this information must travel along with the operation for eventual
    // consumption by emitEpilogue.
    Ops.push_back(DAG.getTargetConstant(FPDiff, DL, MVT::i32));
  }

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (auto &RegToPass : RegsToPass)
    Ops.push_back(DAG.getRegister(RegToPass.first,
                                  RegToPass.second.getValueType()));

  // Add a register mask operand representing the call-preserved registers.
  const uint32_t *Mask;
  const AArch64RegisterInfo *TRI = Subtarget->getRegisterInfo();
  if (IsThisReturn) {
    // For 'this' returns, use the X0-preserving mask if applicable
    Mask = TRI->getThisReturnPreservedMask(MF, CallConv);
    if (!Mask) {
      IsThisReturn = false;
      Mask = TRI->getCallPreservedMask(MF, CallConv);
    }
  } else
    Mask = TRI->getCallPreservedMask(MF, CallConv);

  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  // If we're doing a tall call, use a TC_RETURN here rather than an
  // actual call instruction.
  if (IsTailCall) {
    MF.getFrameInfo()->setHasTailCall();
    return DAG.getNode(AArch64ISD::TC_RETURN, DL, NodeTys, Ops);
  }

  // Returns a chain and a flag for retval copy to use.
  Chain = DAG.getNode(AArch64ISD::CALL, DL, NodeTys, Ops);
  InFlag = Chain.getValue(1);

  uint64_t CalleePopBytes = DoesCalleeRestoreStack(CallConv, TailCallOpt)
                                ? RoundUpToAlignment(NumBytes, 16)
                                : 0;

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, DL, true),
                             DAG.getIntPtrConstant(CalleePopBytes, DL, true),
                             InFlag, DL);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, IsVarArg, Ins, DL, DAG,
                         InVals, IsThisReturn,
                         IsThisReturn ? OutVals[0] : SDValue());
}

bool AArch64TargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  CCAssignFn *RetCC = CallConv == CallingConv::WebKit_JS
                          ? RetCC_AArch64_WebKit_JS
                          : RetCC_AArch64_AAPCS;
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC);
}

SDValue
AArch64TargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                   bool isVarArg,
                                   const SmallVectorImpl<ISD::OutputArg> &Outs,
                                   const SmallVectorImpl<SDValue> &OutVals,
                                   SDLoc DL, SelectionDAG &DAG) const {
  CCAssignFn *RetCC = CallConv == CallingConv::WebKit_JS
                          ? RetCC_AArch64_WebKit_JS
                          : RetCC_AArch64_AAPCS;
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC);

  // Copy the result values into the output registers.
  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);
  for (unsigned i = 0, realRVLocIdx = 0; i != RVLocs.size();
       ++i, ++realRVLocIdx) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    SDValue Arg = OutVals[realRVLocIdx];

    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      if (Outs[i].ArgVT == MVT::i1) {
        // AAPCS requires i1 to be zero-extended to i8 by the producer of the
        // value. This is strictly redundant on Darwin (which uses "zeroext
        // i1"), but will be optimised out before ISel.
        Arg = DAG.getNode(ISD::TRUNCATE, DL, MVT::i1, Arg);
        Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      }
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    }

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Arg, Flag);
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain; // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(AArch64ISD::RET_FLAG, DL, MVT::Other, RetOps);
}

//===----------------------------------------------------------------------===//
//  Other Lowering Code
//===----------------------------------------------------------------------===//

SDValue AArch64TargetLowering::LowerGlobalAddress(SDValue Op,
                                                  SelectionDAG &DAG) const {
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);
  const GlobalAddressSDNode *GN = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = GN->getGlobal();
  unsigned char OpFlags =
      Subtarget->ClassifyGlobalReference(GV, getTargetMachine());

  assert(cast<GlobalAddressSDNode>(Op)->getOffset() == 0 &&
         "unexpected offset in global node");

  // This also catched the large code model case for Darwin.
  if ((OpFlags & AArch64II::MO_GOT) != 0) {
    SDValue GotAddr = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, OpFlags);
    // FIXME: Once remat is capable of dealing with instructions with register
    // operands, expand this into two nodes instead of using a wrapper node.
    return DAG.getNode(AArch64ISD::LOADgot, DL, PtrVT, GotAddr);
  }

  if ((OpFlags & AArch64II::MO_CONSTPOOL) != 0) {
    assert(getTargetMachine().getCodeModel() == CodeModel::Small &&
           "use of MO_CONSTPOOL only supported on small model");
    SDValue Hi = DAG.getTargetConstantPool(GV, PtrVT, 0, 0, AArch64II::MO_PAGE);
    SDValue ADRP = DAG.getNode(AArch64ISD::ADRP, DL, PtrVT, Hi);
    unsigned char LoFlags = AArch64II::MO_PAGEOFF | AArch64II::MO_NC;
    SDValue Lo = DAG.getTargetConstantPool(GV, PtrVT, 0, 0, LoFlags);
    SDValue PoolAddr = DAG.getNode(AArch64ISD::ADDlow, DL, PtrVT, ADRP, Lo);
    SDValue GlobalAddr = DAG.getLoad(
        PtrVT, DL, DAG.getEntryNode(), PoolAddr,
        MachinePointerInfo::getConstantPool(DAG.getMachineFunction()),
        /*isVolatile=*/false,
        /*isNonTemporal=*/true,
        /*isInvariant=*/true, 8);
    if (GN->getOffset() != 0)
      return DAG.getNode(ISD::ADD, DL, PtrVT, GlobalAddr,
                         DAG.getConstant(GN->getOffset(), DL, PtrVT));
    return GlobalAddr;
  }

  if (getTargetMachine().getCodeModel() == CodeModel::Large) {
    const unsigned char MO_NC = AArch64II::MO_NC;
    return DAG.getNode(
        AArch64ISD::WrapperLarge, DL, PtrVT,
        DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, AArch64II::MO_G3),
        DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, AArch64II::MO_G2 | MO_NC),
        DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, AArch64II::MO_G1 | MO_NC),
        DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, AArch64II::MO_G0 | MO_NC));
  } else {
    // Use ADRP/ADD or ADRP/LDR for everything else: the small model on ELF and
    // the only correct model on Darwin.
    SDValue Hi = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0,
                                            OpFlags | AArch64II::MO_PAGE);
    unsigned char LoFlags = OpFlags | AArch64II::MO_PAGEOFF | AArch64II::MO_NC;
    SDValue Lo = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, LoFlags);

    SDValue ADRP = DAG.getNode(AArch64ISD::ADRP, DL, PtrVT, Hi);
    return DAG.getNode(AArch64ISD::ADDlow, DL, PtrVT, ADRP, Lo);
  }
}

/// \brief Convert a TLS address reference into the correct sequence of loads
/// and calls to compute the variable's address (for Darwin, currently) and
/// return an SDValue containing the final node.

/// Darwin only has one TLS scheme which must be capable of dealing with the
/// fully general situation, in the worst case. This means:
///     + "extern __thread" declaration.
///     + Defined in a possibly unknown dynamic library.
///
/// The general system is that each __thread variable has a [3 x i64] descriptor
/// which contains information used by the runtime to calculate the address. The
/// only part of this the compiler needs to know about is the first xword, which
/// contains a function pointer that must be called with the address of the
/// entire descriptor in "x0".
///
/// Since this descriptor may be in a different unit, in general even the
/// descriptor must be accessed via an indirect load. The "ideal" code sequence
/// is:
///     adrp x0, _var@TLVPPAGE
///     ldr x0, [x0, _var@TLVPPAGEOFF]   ; x0 now contains address of descriptor
///     ldr x1, [x0]                     ; x1 contains 1st entry of descriptor,
///                                      ; the function pointer
///     blr x1                           ; Uses descriptor address in x0
///     ; Address of _var is now in x0.
///
/// If the address of _var's descriptor *is* known to the linker, then it can
/// change the first "ldr" instruction to an appropriate "add x0, x0, #imm" for
/// a slight efficiency gain.
SDValue
AArch64TargetLowering::LowerDarwinGlobalTLSAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  assert(Subtarget->isTargetDarwin() && "TLS only supported on Darwin");

  SDLoc DL(Op);
  MVT PtrVT = getPointerTy(DAG.getDataLayout());
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();

  SDValue TLVPAddr =
      DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, AArch64II::MO_TLS);
  SDValue DescAddr = DAG.getNode(AArch64ISD::LOADgot, DL, PtrVT, TLVPAddr);

  // The first entry in the descriptor is a function pointer that we must call
  // to obtain the address of the variable.
  SDValue Chain = DAG.getEntryNode();
  SDValue FuncTLVGet =
      DAG.getLoad(MVT::i64, DL, Chain, DescAddr,
                  MachinePointerInfo::getGOT(DAG.getMachineFunction()), false,
                  true, true, 8);
  Chain = FuncTLVGet.getValue(1);

  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setAdjustsStack(true);

  // TLS calls preserve all registers except those that absolutely must be
  // trashed: X0 (it takes an argument), LR (it's a call) and NZCV (let's not be
  // silly).
  const uint32_t *Mask =
      Subtarget->getRegisterInfo()->getTLSCallPreservedMask();

  // Finally, we can make the call. This is just a degenerate version of a
  // normal AArch64 call node: x0 takes the address of the descriptor, and
  // returns the address of the variable in this thread.
  Chain = DAG.getCopyToReg(Chain, DL, AArch64::X0, DescAddr, SDValue());
  Chain =
      DAG.getNode(AArch64ISD::CALL, DL, DAG.getVTList(MVT::Other, MVT::Glue),
                  Chain, FuncTLVGet, DAG.getRegister(AArch64::X0, MVT::i64),
                  DAG.getRegisterMask(Mask), Chain.getValue(1));
  return DAG.getCopyFromReg(Chain, DL, AArch64::X0, PtrVT, Chain.getValue(1));
}

/// When accessing thread-local variables under either the general-dynamic or
/// local-dynamic system, we make a "TLS-descriptor" call. The variable will
/// have a descriptor, accessible via a PC-relative ADRP, and whose first entry
/// is a function pointer to carry out the resolution.
///
/// The sequence is:
///    adrp  x0, :tlsdesc:var
///    ldr   x1, [x0, #:tlsdesc_lo12:var]
///    add   x0, x0, #:tlsdesc_lo12:var
///    .tlsdesccall var
///    blr   x1
///    (TPIDR_EL0 offset now in x0)
///
///  The above sequence must be produced unscheduled, to enable the linker to
///  optimize/relax this sequence.
///  Therefore, a pseudo-instruction (TLSDESC_CALLSEQ) is used to represent the
///  above sequence, and expanded really late in the compilation flow, to ensure
///  the sequence is produced as per above.
SDValue AArch64TargetLowering::LowerELFTLSDescCallSeq(SDValue SymAddr, SDLoc DL,
                                                      SelectionDAG &DAG) const {
  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  SDValue Chain = DAG.getEntryNode();
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  SmallVector<SDValue, 2> Ops;
  Ops.push_back(Chain);
  Ops.push_back(SymAddr);

  Chain = DAG.getNode(AArch64ISD::TLSDESC_CALLSEQ, DL, NodeTys, Ops);
  SDValue Glue = Chain.getValue(1);

  return DAG.getCopyFromReg(Chain, DL, AArch64::X0, PtrVT, Glue);
}

SDValue
AArch64TargetLowering::LowerELFGlobalTLSAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  assert(Subtarget->isTargetELF() && "This function expects an ELF target");
  assert(getTargetMachine().getCodeModel() == CodeModel::Small &&
         "ELF TLS only supported in small memory model");
  // Different choices can be made for the maximum size of the TLS area for a
  // module. For the small address model, the default TLS size is 16MiB and the
  // maximum TLS size is 4GiB.
  // FIXME: add -mtls-size command line option and make it control the 16MiB
  // vs. 4GiB code sequence generation.
  const GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);

  TLSModel::Model Model = getTargetMachine().getTLSModel(GA->getGlobal());

  if (DAG.getTarget().Options.EmulatedTLS)
    return LowerToTLSEmulatedModel(GA, DAG);

  if (!EnableAArch64ELFLocalDynamicTLSGeneration) {
    if (Model == TLSModel::LocalDynamic)
      Model = TLSModel::GeneralDynamic;
  }

  SDValue TPOff;
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);
  const GlobalValue *GV = GA->getGlobal();

  SDValue ThreadBase = DAG.getNode(AArch64ISD::THREAD_POINTER, DL, PtrVT);

  if (Model == TLSModel::LocalExec) {
    SDValue HiVar = DAG.getTargetGlobalAddress(
        GV, DL, PtrVT, 0, AArch64II::MO_TLS | AArch64II::MO_HI12);
    SDValue LoVar = DAG.getTargetGlobalAddress(
        GV, DL, PtrVT, 0,
        AArch64II::MO_TLS | AArch64II::MO_PAGEOFF | AArch64II::MO_NC);

    SDValue TPWithOff_lo =
        SDValue(DAG.getMachineNode(AArch64::ADDXri, DL, PtrVT, ThreadBase,
                                   HiVar,
                                   DAG.getTargetConstant(0, DL, MVT::i32)),
                0);
    SDValue TPWithOff =
        SDValue(DAG.getMachineNode(AArch64::ADDXri, DL, PtrVT, TPWithOff_lo,
                                   LoVar,
                                   DAG.getTargetConstant(0, DL, MVT::i32)),
                0);
    return TPWithOff;
  } else if (Model == TLSModel::InitialExec) {
    TPOff = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, AArch64II::MO_TLS);
    TPOff = DAG.getNode(AArch64ISD::LOADgot, DL, PtrVT, TPOff);
  } else if (Model == TLSModel::LocalDynamic) {
    // Local-dynamic accesses proceed in two phases. A general-dynamic TLS
    // descriptor call against the special symbol _TLS_MODULE_BASE_ to calculate
    // the beginning of the module's TLS region, followed by a DTPREL offset
    // calculation.

    // These accesses will need deduplicating if there's more than one.
    AArch64FunctionInfo *MFI =
        DAG.getMachineFunction().getInfo<AArch64FunctionInfo>();
    MFI->incNumLocalDynamicTLSAccesses();

    // The call needs a relocation too for linker relaxation. It doesn't make
    // sense to call it MO_PAGE or MO_PAGEOFF though so we need another copy of
    // the address.
    SDValue SymAddr = DAG.getTargetExternalSymbol("_TLS_MODULE_BASE_", PtrVT,
                                                  AArch64II::MO_TLS);

    // Now we can calculate the offset from TPIDR_EL0 to this module's
    // thread-local area.
    TPOff = LowerELFTLSDescCallSeq(SymAddr, DL, DAG);

    // Now use :dtprel_whatever: operations to calculate this variable's offset
    // in its thread-storage area.
    SDValue HiVar = DAG.getTargetGlobalAddress(
        GV, DL, MVT::i64, 0, AArch64II::MO_TLS | AArch64II::MO_HI12);
    SDValue LoVar = DAG.getTargetGlobalAddress(
        GV, DL, MVT::i64, 0,
        AArch64II::MO_TLS | AArch64II::MO_PAGEOFF | AArch64II::MO_NC);

    TPOff = SDValue(DAG.getMachineNode(AArch64::ADDXri, DL, PtrVT, TPOff, HiVar,
                                       DAG.getTargetConstant(0, DL, MVT::i32)),
                    0);
    TPOff = SDValue(DAG.getMachineNode(AArch64::ADDXri, DL, PtrVT, TPOff, LoVar,
                                       DAG.getTargetConstant(0, DL, MVT::i32)),
                    0);
  } else if (Model == TLSModel::GeneralDynamic) {
    // The call needs a relocation too for linker relaxation. It doesn't make
    // sense to call it MO_PAGE or MO_PAGEOFF though so we need another copy of
    // the address.
    SDValue SymAddr =
        DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, AArch64II::MO_TLS);

    // Finally we can make a call to calculate the offset from tpidr_el0.
    TPOff = LowerELFTLSDescCallSeq(SymAddr, DL, DAG);
  } else
    llvm_unreachable("Unsupported ELF TLS access model");

  return DAG.getNode(ISD::ADD, DL, PtrVT, ThreadBase, TPOff);
}

SDValue AArch64TargetLowering::LowerGlobalTLSAddress(SDValue Op,
                                                     SelectionDAG &DAG) const {
  if (Subtarget->isTargetDarwin())
    return LowerDarwinGlobalTLSAddress(Op, DAG);
  else if (Subtarget->isTargetELF())
    return LowerELFGlobalTLSAddress(Op, DAG);

  llvm_unreachable("Unexpected platform trying to use TLS");
}
SDValue AArch64TargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  SDLoc dl(Op);

  // Handle f128 first, since lowering it will result in comparing the return
  // value of a libcall against zero, which is just what the rest of LowerBR_CC
  // is expecting to deal with.
  if (LHS.getValueType() == MVT::f128) {
    softenSetCCOperands(DAG, MVT::f128, LHS, RHS, CC, dl);

    // If softenSetCCOperands returned a scalar, we need to compare the result
    // against zero to select between true and false values.
    if (!RHS.getNode()) {
      RHS = DAG.getConstant(0, dl, LHS.getValueType());
      CC = ISD::SETNE;
    }
  }

  // Optimize {s|u}{add|sub|mul}.with.overflow feeding into a branch
  // instruction.
  unsigned Opc = LHS.getOpcode();
  if (LHS.getResNo() == 1 && isa<ConstantSDNode>(RHS) &&
      cast<ConstantSDNode>(RHS)->isOne() &&
      (Opc == ISD::SADDO || Opc == ISD::UADDO || Opc == ISD::SSUBO ||
       Opc == ISD::USUBO || Opc == ISD::SMULO || Opc == ISD::UMULO)) {
    assert((CC == ISD::SETEQ || CC == ISD::SETNE) &&
           "Unexpected condition code.");
    // Only lower legal XALUO ops.
    if (!DAG.getTargetLoweringInfo().isTypeLegal(LHS->getValueType(0)))
      return SDValue();

    // The actual operation with overflow check.
    AArch64CC::CondCode OFCC;
    SDValue Value, Overflow;
    std::tie(Value, Overflow) = getAArch64XALUOOp(OFCC, LHS.getValue(0), DAG);

    if (CC == ISD::SETNE)
      OFCC = getInvertedCondCode(OFCC);
    SDValue CCVal = DAG.getConstant(OFCC, dl, MVT::i32);

    return DAG.getNode(AArch64ISD::BRCOND, dl, MVT::Other, Chain, Dest, CCVal,
                       Overflow);
  }

  if (LHS.getValueType().isInteger()) {
    assert((LHS.getValueType() == RHS.getValueType()) &&
           (LHS.getValueType() == MVT::i32 || LHS.getValueType() == MVT::i64));

    // If the RHS of the comparison is zero, we can potentially fold this
    // to a specialized branch.
    const ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS);
    if (RHSC && RHSC->getZExtValue() == 0) {
      if (CC == ISD::SETEQ) {
        // See if we can use a TBZ to fold in an AND as well.
        // TBZ has a smaller branch displacement than CBZ.  If the offset is
        // out of bounds, a late MI-layer pass rewrites branches.
        // 403.gcc is an example that hits this case.
        if (LHS.getOpcode() == ISD::AND &&
            isa<ConstantSDNode>(LHS.getOperand(1)) &&
            isPowerOf2_64(LHS.getConstantOperandVal(1))) {
          SDValue Test = LHS.getOperand(0);
          uint64_t Mask = LHS.getConstantOperandVal(1);
          return DAG.getNode(AArch64ISD::TBZ, dl, MVT::Other, Chain, Test,
                             DAG.getConstant(Log2_64(Mask), dl, MVT::i64),
                             Dest);
        }

        return DAG.getNode(AArch64ISD::CBZ, dl, MVT::Other, Chain, LHS, Dest);
      } else if (CC == ISD::SETNE) {
        // See if we can use a TBZ to fold in an AND as well.
        // TBZ has a smaller branch displacement than CBZ.  If the offset is
        // out of bounds, a late MI-layer pass rewrites branches.
        // 403.gcc is an example that hits this case.
        if (LHS.getOpcode() == ISD::AND &&
            isa<ConstantSDNode>(LHS.getOperand(1)) &&
            isPowerOf2_64(LHS.getConstantOperandVal(1))) {
          SDValue Test = LHS.getOperand(0);
          uint64_t Mask = LHS.getConstantOperandVal(1);
          return DAG.getNode(AArch64ISD::TBNZ, dl, MVT::Other, Chain, Test,
                             DAG.getConstant(Log2_64(Mask), dl, MVT::i64),
                             Dest);
        }

        return DAG.getNode(AArch64ISD::CBNZ, dl, MVT::Other, Chain, LHS, Dest);
      } else if (CC == ISD::SETLT && LHS.getOpcode() != ISD::AND) {
        // Don't combine AND since emitComparison converts the AND to an ANDS
        // (a.k.a. TST) and the test in the test bit and branch instruction
        // becomes redundant.  This would also increase register pressure.
        uint64_t Mask = LHS.getValueType().getSizeInBits() - 1;
        return DAG.getNode(AArch64ISD::TBNZ, dl, MVT::Other, Chain, LHS,
                           DAG.getConstant(Mask, dl, MVT::i64), Dest);
      }
    }
    if (RHSC && RHSC->getSExtValue() == -1 && CC == ISD::SETGT &&
        LHS.getOpcode() != ISD::AND) {
      // Don't combine AND since emitComparison converts the AND to an ANDS
      // (a.k.a. TST) and the test in the test bit and branch instruction
      // becomes redundant.  This would also increase register pressure.
      uint64_t Mask = LHS.getValueType().getSizeInBits() - 1;
      return DAG.getNode(AArch64ISD::TBZ, dl, MVT::Other, Chain, LHS,
                         DAG.getConstant(Mask, dl, MVT::i64), Dest);
    }

    SDValue CCVal;
    SDValue Cmp = getAArch64Cmp(LHS, RHS, CC, CCVal, DAG, dl);
    return DAG.getNode(AArch64ISD::BRCOND, dl, MVT::Other, Chain, Dest, CCVal,
                       Cmp);
  }

  assert(LHS.getValueType() == MVT::f32 || LHS.getValueType() == MVT::f64);

  // Unfortunately, the mapping of LLVM FP CC's onto AArch64 CC's isn't totally
  // clean.  Some of them require two branches to implement.
  SDValue Cmp = emitComparison(LHS, RHS, CC, dl, DAG);
  AArch64CC::CondCode CC1, CC2;
  changeFPCCToAArch64CC(CC, CC1, CC2);
  SDValue CC1Val = DAG.getConstant(CC1, dl, MVT::i32);
  SDValue BR1 =
      DAG.getNode(AArch64ISD::BRCOND, dl, MVT::Other, Chain, Dest, CC1Val, Cmp);
  if (CC2 != AArch64CC::AL) {
    SDValue CC2Val = DAG.getConstant(CC2, dl, MVT::i32);
    return DAG.getNode(AArch64ISD::BRCOND, dl, MVT::Other, BR1, Dest, CC2Val,
                       Cmp);
  }

  return BR1;
}

SDValue AArch64TargetLowering::LowerFCOPYSIGN(SDValue Op,
                                              SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  SDValue In1 = Op.getOperand(0);
  SDValue In2 = Op.getOperand(1);
  EVT SrcVT = In2.getValueType();

  if (SrcVT.bitsLT(VT))
    In2 = DAG.getNode(ISD::FP_EXTEND, DL, VT, In2);
  else if (SrcVT.bitsGT(VT))
    In2 = DAG.getNode(ISD::FP_ROUND, DL, VT, In2, DAG.getIntPtrConstant(0, DL));

  EVT VecVT;
  EVT EltVT;
  uint64_t EltMask;
  SDValue VecVal1, VecVal2;
  if (VT == MVT::f32 || VT == MVT::v2f32 || VT == MVT::v4f32) {
    EltVT = MVT::i32;
    VecVT = (VT == MVT::v2f32 ? MVT::v2i32 : MVT::v4i32);
    EltMask = 0x80000000ULL;

    if (!VT.isVector()) {
      VecVal1 = DAG.getTargetInsertSubreg(AArch64::ssub, DL, VecVT,
                                          DAG.getUNDEF(VecVT), In1);
      VecVal2 = DAG.getTargetInsertSubreg(AArch64::ssub, DL, VecVT,
                                          DAG.getUNDEF(VecVT), In2);
    } else {
      VecVal1 = DAG.getNode(ISD::BITCAST, DL, VecVT, In1);
      VecVal2 = DAG.getNode(ISD::BITCAST, DL, VecVT, In2);
    }
  } else if (VT == MVT::f64 || VT == MVT::v2f64) {
    EltVT = MVT::i64;
    VecVT = MVT::v2i64;

    // We want to materialize a mask with the high bit set, but the AdvSIMD
    // immediate moves cannot materialize that in a single instruction for
    // 64-bit elements. Instead, materialize zero and then negate it.
    EltMask = 0;

    if (!VT.isVector()) {
      VecVal1 = DAG.getTargetInsertSubreg(AArch64::dsub, DL, VecVT,
                                          DAG.getUNDEF(VecVT), In1);
      VecVal2 = DAG.getTargetInsertSubreg(AArch64::dsub, DL, VecVT,
                                          DAG.getUNDEF(VecVT), In2);
    } else {
      VecVal1 = DAG.getNode(ISD::BITCAST, DL, VecVT, In1);
      VecVal2 = DAG.getNode(ISD::BITCAST, DL, VecVT, In2);
    }
  } else {
    llvm_unreachable("Invalid type for copysign!");
  }

  SDValue BuildVec = DAG.getConstant(EltMask, DL, VecVT);

  // If we couldn't materialize the mask above, then the mask vector will be
  // the zero vector, and we need to negate it here.
  if (VT == MVT::f64 || VT == MVT::v2f64) {
    BuildVec = DAG.getNode(ISD::BITCAST, DL, MVT::v2f64, BuildVec);
    BuildVec = DAG.getNode(ISD::FNEG, DL, MVT::v2f64, BuildVec);
    BuildVec = DAG.getNode(ISD::BITCAST, DL, MVT::v2i64, BuildVec);
  }

  SDValue Sel =
      DAG.getNode(AArch64ISD::BIT, DL, VecVT, VecVal1, VecVal2, BuildVec);

  if (VT == MVT::f32)
    return DAG.getTargetExtractSubreg(AArch64::ssub, DL, VT, Sel);
  else if (VT == MVT::f64)
    return DAG.getTargetExtractSubreg(AArch64::dsub, DL, VT, Sel);
  else
    return DAG.getNode(ISD::BITCAST, DL, VT, Sel);
}

SDValue AArch64TargetLowering::LowerCTPOP(SDValue Op, SelectionDAG &DAG) const {
  if (DAG.getMachineFunction().getFunction()->hasFnAttribute(
          Attribute::NoImplicitFloat))
    return SDValue();

  if (!Subtarget->hasNEON())
    return SDValue();

  // While there is no integer popcount instruction, it can
  // be more efficiently lowered to the following sequence that uses
  // AdvSIMD registers/instructions as long as the copies to/from
  // the AdvSIMD registers are cheap.
  //  FMOV    D0, X0        // copy 64-bit int to vector, high bits zero'd
  //  CNT     V0.8B, V0.8B  // 8xbyte pop-counts
  //  ADDV    B0, V0.8B     // sum 8xbyte pop-counts
  //  UMOV    X0, V0.B[0]   // copy byte result back to integer reg
  SDValue Val = Op.getOperand(0);
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  if (VT == MVT::i32)
    Val = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, Val);
  Val = DAG.getNode(ISD::BITCAST, DL, MVT::v8i8, Val);

  SDValue CtPop = DAG.getNode(ISD::CTPOP, DL, MVT::v8i8, Val);
  SDValue UaddLV = DAG.getNode(
      ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
      DAG.getConstant(Intrinsic::aarch64_neon_uaddlv, DL, MVT::i32), CtPop);

  if (VT == MVT::i64)
    UaddLV = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, UaddLV);
  return UaddLV;
}

SDValue AArch64TargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {

  if (Op.getValueType().isVector())
    return LowerVSETCC(Op, DAG);

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  SDLoc dl(Op);

  // We chose ZeroOrOneBooleanContents, so use zero and one.
  EVT VT = Op.getValueType();
  SDValue TVal = DAG.getConstant(1, dl, VT);
  SDValue FVal = DAG.getConstant(0, dl, VT);

  // Handle f128 first, since one possible outcome is a normal integer
  // comparison which gets picked up by the next if statement.
  if (LHS.getValueType() == MVT::f128) {
    softenSetCCOperands(DAG, MVT::f128, LHS, RHS, CC, dl);

    // If softenSetCCOperands returned a scalar, use it.
    if (!RHS.getNode()) {
      assert(LHS.getValueType() == Op.getValueType() &&
             "Unexpected setcc expansion!");
      return LHS;
    }
  }

  if (LHS.getValueType().isInteger()) {
    SDValue CCVal;
    SDValue Cmp =
        getAArch64Cmp(LHS, RHS, ISD::getSetCCInverse(CC, true), CCVal, DAG, dl);

    // Note that we inverted the condition above, so we reverse the order of
    // the true and false operands here.  This will allow the setcc to be
    // matched to a single CSINC instruction.
    return DAG.getNode(AArch64ISD::CSEL, dl, VT, FVal, TVal, CCVal, Cmp);
  }

  // Now we know we're dealing with FP values.
  assert(LHS.getValueType() == MVT::f32 || LHS.getValueType() == MVT::f64);

  // If that fails, we'll need to perform an FCMP + CSEL sequence.  Go ahead
  // and do the comparison.
  SDValue Cmp = emitComparison(LHS, RHS, CC, dl, DAG);

  AArch64CC::CondCode CC1, CC2;
  changeFPCCToAArch64CC(CC, CC1, CC2);
  if (CC2 == AArch64CC::AL) {
    changeFPCCToAArch64CC(ISD::getSetCCInverse(CC, false), CC1, CC2);
    SDValue CC1Val = DAG.getConstant(CC1, dl, MVT::i32);

    // Note that we inverted the condition above, so we reverse the order of
    // the true and false operands here.  This will allow the setcc to be
    // matched to a single CSINC instruction.
    return DAG.getNode(AArch64ISD::CSEL, dl, VT, FVal, TVal, CC1Val, Cmp);
  } else {
    // Unfortunately, the mapping of LLVM FP CC's onto AArch64 CC's isn't
    // totally clean.  Some of them require two CSELs to implement.  As is in
    // this case, we emit the first CSEL and then emit a second using the output
    // of the first as the RHS.  We're effectively OR'ing the two CC's together.

    // FIXME: It would be nice if we could match the two CSELs to two CSINCs.
    SDValue CC1Val = DAG.getConstant(CC1, dl, MVT::i32);
    SDValue CS1 =
        DAG.getNode(AArch64ISD::CSEL, dl, VT, TVal, FVal, CC1Val, Cmp);

    SDValue CC2Val = DAG.getConstant(CC2, dl, MVT::i32);
    return DAG.getNode(AArch64ISD::CSEL, dl, VT, TVal, CS1, CC2Val, Cmp);
  }
}

SDValue AArch64TargetLowering::LowerSELECT_CC(ISD::CondCode CC, SDValue LHS,
                                              SDValue RHS, SDValue TVal,
                                              SDValue FVal, SDLoc dl,
                                              SelectionDAG &DAG) const {
  // Handle f128 first, because it will result in a comparison of some RTLIB
  // call result against zero.
  if (LHS.getValueType() == MVT::f128) {
    softenSetCCOperands(DAG, MVT::f128, LHS, RHS, CC, dl);

    // If softenSetCCOperands returned a scalar, we need to compare the result
    // against zero to select between true and false values.
    if (!RHS.getNode()) {
      RHS = DAG.getConstant(0, dl, LHS.getValueType());
      CC = ISD::SETNE;
    }
  }

  // Also handle f16, for which we need to do a f32 comparison.
  if (LHS.getValueType() == MVT::f16) {
    LHS = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f32, LHS);
    RHS = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f32, RHS);
  }

  // Next, handle integers.
  if (LHS.getValueType().isInteger()) {
    assert((LHS.getValueType() == RHS.getValueType()) &&
           (LHS.getValueType() == MVT::i32 || LHS.getValueType() == MVT::i64));

    unsigned Opcode = AArch64ISD::CSEL;

    // If both the TVal and the FVal are constants, see if we can swap them in
    // order to for a CSINV or CSINC out of them.
    ConstantSDNode *CFVal = dyn_cast<ConstantSDNode>(FVal);
    ConstantSDNode *CTVal = dyn_cast<ConstantSDNode>(TVal);

    if (CTVal && CFVal && CTVal->isAllOnesValue() && CFVal->isNullValue()) {
      std::swap(TVal, FVal);
      std::swap(CTVal, CFVal);
      CC = ISD::getSetCCInverse(CC, true);
    } else if (CTVal && CFVal && CTVal->isOne() && CFVal->isNullValue()) {
      std::swap(TVal, FVal);
      std::swap(CTVal, CFVal);
      CC = ISD::getSetCCInverse(CC, true);
    } else if (TVal.getOpcode() == ISD::XOR) {
      // If TVal is a NOT we want to swap TVal and FVal so that we can match
      // with a CSINV rather than a CSEL.
      ConstantSDNode *CVal = dyn_cast<ConstantSDNode>(TVal.getOperand(1));

      if (CVal && CVal->isAllOnesValue()) {
        std::swap(TVal, FVal);
        std::swap(CTVal, CFVal);
        CC = ISD::getSetCCInverse(CC, true);
      }
    } else if (TVal.getOpcode() == ISD::SUB) {
      // If TVal is a negation (SUB from 0) we want to swap TVal and FVal so
      // that we can match with a CSNEG rather than a CSEL.
      ConstantSDNode *CVal = dyn_cast<ConstantSDNode>(TVal.getOperand(0));

      if (CVal && CVal->isNullValue()) {
        std::swap(TVal, FVal);
        std::swap(CTVal, CFVal);
        CC = ISD::getSetCCInverse(CC, true);
      }
    } else if (CTVal && CFVal) {
      const int64_t TrueVal = CTVal->getSExtValue();
      const int64_t FalseVal = CFVal->getSExtValue();
      bool Swap = false;

      // If both TVal and FVal are constants, see if FVal is the
      // inverse/negation/increment of TVal and generate a CSINV/CSNEG/CSINC
      // instead of a CSEL in that case.
      if (TrueVal == ~FalseVal) {
        Opcode = AArch64ISD::CSINV;
      } else if (TrueVal == -FalseVal) {
        Opcode = AArch64ISD::CSNEG;
      } else if (TVal.getValueType() == MVT::i32) {
        // If our operands are only 32-bit wide, make sure we use 32-bit
        // arithmetic for the check whether we can use CSINC. This ensures that
        // the addition in the check will wrap around properly in case there is
        // an overflow (which would not be the case if we do the check with
        // 64-bit arithmetic).
        const uint32_t TrueVal32 = CTVal->getZExtValue();
        const uint32_t FalseVal32 = CFVal->getZExtValue();

        if ((TrueVal32 == FalseVal32 + 1) || (TrueVal32 + 1 == FalseVal32)) {
          Opcode = AArch64ISD::CSINC;

          if (TrueVal32 > FalseVal32) {
            Swap = true;
          }
        }
        // 64-bit check whether we can use CSINC.
      } else if ((TrueVal == FalseVal + 1) || (TrueVal + 1 == FalseVal)) {
        Opcode = AArch64ISD::CSINC;

        if (TrueVal > FalseVal) {
          Swap = true;
        }
      }

      // Swap TVal and FVal if necessary.
      if (Swap) {
        std::swap(TVal, FVal);
        std::swap(CTVal, CFVal);
        CC = ISD::getSetCCInverse(CC, true);
      }

      if (Opcode != AArch64ISD::CSEL) {
        // Drop FVal since we can get its value by simply inverting/negating
        // TVal.
        FVal = TVal;
      }
    }

    SDValue CCVal;
    SDValue Cmp = getAArch64Cmp(LHS, RHS, CC, CCVal, DAG, dl);

    EVT VT = TVal.getValueType();
    return DAG.getNode(Opcode, dl, VT, TVal, FVal, CCVal, Cmp);
  }

  // Now we know we're dealing with FP values.
  assert(LHS.getValueType() == MVT::f32 || LHS.getValueType() == MVT::f64);
  assert(LHS.getValueType() == RHS.getValueType());
  EVT VT = TVal.getValueType();
  SDValue Cmp = emitComparison(LHS, RHS, CC, dl, DAG);

  // Unfortunately, the mapping of LLVM FP CC's onto AArch64 CC's isn't totally
  // clean.  Some of them require two CSELs to implement.
  AArch64CC::CondCode CC1, CC2;
  changeFPCCToAArch64CC(CC, CC1, CC2);
  SDValue CC1Val = DAG.getConstant(CC1, dl, MVT::i32);
  SDValue CS1 = DAG.getNode(AArch64ISD::CSEL, dl, VT, TVal, FVal, CC1Val, Cmp);

  // If we need a second CSEL, emit it, using the output of the first as the
  // RHS.  We're effectively OR'ing the two CC's together.
  if (CC2 != AArch64CC::AL) {
    SDValue CC2Val = DAG.getConstant(CC2, dl, MVT::i32);
    return DAG.getNode(AArch64ISD::CSEL, dl, VT, TVal, CS1, CC2Val, Cmp);
  }

  // Otherwise, return the output of the first CSEL.
  return CS1;
}

SDValue AArch64TargetLowering::LowerSELECT_CC(SDValue Op,
                                              SelectionDAG &DAG) const {
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TVal = Op.getOperand(2);
  SDValue FVal = Op.getOperand(3);
  SDLoc DL(Op);
  return LowerSELECT_CC(CC, LHS, RHS, TVal, FVal, DL, DAG);
}

SDValue AArch64TargetLowering::LowerSELECT(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDValue CCVal = Op->getOperand(0);
  SDValue TVal = Op->getOperand(1);
  SDValue FVal = Op->getOperand(2);
  SDLoc DL(Op);

  unsigned Opc = CCVal.getOpcode();
  // Optimize {s|u}{add|sub|mul}.with.overflow feeding into a select
  // instruction.
  if (CCVal.getResNo() == 1 &&
      (Opc == ISD::SADDO || Opc == ISD::UADDO || Opc == ISD::SSUBO ||
       Opc == ISD::USUBO || Opc == ISD::SMULO || Opc == ISD::UMULO)) {
    // Only lower legal XALUO ops.
    if (!DAG.getTargetLoweringInfo().isTypeLegal(CCVal->getValueType(0)))
      return SDValue();

    AArch64CC::CondCode OFCC;
    SDValue Value, Overflow;
    std::tie(Value, Overflow) = getAArch64XALUOOp(OFCC, CCVal.getValue(0), DAG);
    SDValue CCVal = DAG.getConstant(OFCC, DL, MVT::i32);

    return DAG.getNode(AArch64ISD::CSEL, DL, Op.getValueType(), TVal, FVal,
                       CCVal, Overflow);
  }

  // Lower it the same way as we would lower a SELECT_CC node.
  ISD::CondCode CC;
  SDValue LHS, RHS;
  if (CCVal.getOpcode() == ISD::SETCC) {
    LHS = CCVal.getOperand(0);
    RHS = CCVal.getOperand(1);
    CC = cast<CondCodeSDNode>(CCVal->getOperand(2))->get();
  } else {
    LHS = CCVal;
    RHS = DAG.getConstant(0, DL, CCVal.getValueType());
    CC = ISD::SETNE;
  }
  return LowerSELECT_CC(CC, LHS, RHS, TVal, FVal, DL, DAG);
}

SDValue AArch64TargetLowering::LowerJumpTable(SDValue Op,
                                              SelectionDAG &DAG) const {
  // Jump table entries as PC relative offsets. No additional tweaking
  // is necessary here. Just get the address of the jump table.
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);

  if (getTargetMachine().getCodeModel() == CodeModel::Large &&
      !Subtarget->isTargetMachO()) {
    const unsigned char MO_NC = AArch64II::MO_NC;
    return DAG.getNode(
        AArch64ISD::WrapperLarge, DL, PtrVT,
        DAG.getTargetJumpTable(JT->getIndex(), PtrVT, AArch64II::MO_G3),
        DAG.getTargetJumpTable(JT->getIndex(), PtrVT, AArch64II::MO_G2 | MO_NC),
        DAG.getTargetJumpTable(JT->getIndex(), PtrVT, AArch64II::MO_G1 | MO_NC),
        DAG.getTargetJumpTable(JT->getIndex(), PtrVT,
                               AArch64II::MO_G0 | MO_NC));
  }

  SDValue Hi =
      DAG.getTargetJumpTable(JT->getIndex(), PtrVT, AArch64II::MO_PAGE);
  SDValue Lo = DAG.getTargetJumpTable(JT->getIndex(), PtrVT,
                                      AArch64II::MO_PAGEOFF | AArch64II::MO_NC);
  SDValue ADRP = DAG.getNode(AArch64ISD::ADRP, DL, PtrVT, Hi);
  return DAG.getNode(AArch64ISD::ADDlow, DL, PtrVT, ADRP, Lo);
}

SDValue AArch64TargetLowering::LowerConstantPool(SDValue Op,
                                                 SelectionDAG &DAG) const {
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);

  if (getTargetMachine().getCodeModel() == CodeModel::Large) {
    // Use the GOT for the large code model on iOS.
    if (Subtarget->isTargetMachO()) {
      SDValue GotAddr = DAG.getTargetConstantPool(
          CP->getConstVal(), PtrVT, CP->getAlignment(), CP->getOffset(),
          AArch64II::MO_GOT);
      return DAG.getNode(AArch64ISD::LOADgot, DL, PtrVT, GotAddr);
    }

    const unsigned char MO_NC = AArch64II::MO_NC;
    return DAG.getNode(
        AArch64ISD::WrapperLarge, DL, PtrVT,
        DAG.getTargetConstantPool(CP->getConstVal(), PtrVT, CP->getAlignment(),
                                  CP->getOffset(), AArch64II::MO_G3),
        DAG.getTargetConstantPool(CP->getConstVal(), PtrVT, CP->getAlignment(),
                                  CP->getOffset(), AArch64II::MO_G2 | MO_NC),
        DAG.getTargetConstantPool(CP->getConstVal(), PtrVT, CP->getAlignment(),
                                  CP->getOffset(), AArch64II::MO_G1 | MO_NC),
        DAG.getTargetConstantPool(CP->getConstVal(), PtrVT, CP->getAlignment(),
                                  CP->getOffset(), AArch64II::MO_G0 | MO_NC));
  } else {
    // Use ADRP/ADD or ADRP/LDR for everything else: the small memory model on
    // ELF, the only valid one on Darwin.
    SDValue Hi =
        DAG.getTargetConstantPool(CP->getConstVal(), PtrVT, CP->getAlignment(),
                                  CP->getOffset(), AArch64II::MO_PAGE);
    SDValue Lo = DAG.getTargetConstantPool(
        CP->getConstVal(), PtrVT, CP->getAlignment(), CP->getOffset(),
        AArch64II::MO_PAGEOFF | AArch64II::MO_NC);

    SDValue ADRP = DAG.getNode(AArch64ISD::ADRP, DL, PtrVT, Hi);
    return DAG.getNode(AArch64ISD::ADDlow, DL, PtrVT, ADRP, Lo);
  }
}

SDValue AArch64TargetLowering::LowerBlockAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);
  if (getTargetMachine().getCodeModel() == CodeModel::Large &&
      !Subtarget->isTargetMachO()) {
    const unsigned char MO_NC = AArch64II::MO_NC;
    return DAG.getNode(
        AArch64ISD::WrapperLarge, DL, PtrVT,
        DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_G3),
        DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_G2 | MO_NC),
        DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_G1 | MO_NC),
        DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_G0 | MO_NC));
  } else {
    SDValue Hi = DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_PAGE);
    SDValue Lo = DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_PAGEOFF |
                                                             AArch64II::MO_NC);
    SDValue ADRP = DAG.getNode(AArch64ISD::ADRP, DL, PtrVT, Hi);
    return DAG.getNode(AArch64ISD::ADDlow, DL, PtrVT, ADRP, Lo);
  }
}

SDValue AArch64TargetLowering::LowerDarwin_VASTART(SDValue Op,
                                                 SelectionDAG &DAG) const {
  AArch64FunctionInfo *FuncInfo =
      DAG.getMachineFunction().getInfo<AArch64FunctionInfo>();

  SDLoc DL(Op);
  SDValue FR = DAG.getFrameIndex(FuncInfo->getVarArgsStackIndex(),
                                 getPointerTy(DAG.getDataLayout()));
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, FR, Op.getOperand(1),
                      MachinePointerInfo(SV), false, false, 0);
}

SDValue AArch64TargetLowering::LowerAAPCS_VASTART(SDValue Op,
                                                SelectionDAG &DAG) const {
  // The layout of the va_list struct is specified in the AArch64 Procedure Call
  // Standard, section B.3.
  MachineFunction &MF = DAG.getMachineFunction();
  AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
  auto PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);

  SDValue Chain = Op.getOperand(0);
  SDValue VAList = Op.getOperand(1);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  SmallVector<SDValue, 4> MemOps;

  // void *__stack at offset 0
  SDValue Stack = DAG.getFrameIndex(FuncInfo->getVarArgsStackIndex(), PtrVT);
  MemOps.push_back(DAG.getStore(Chain, DL, Stack, VAList,
                                MachinePointerInfo(SV), false, false, 8));

  // void *__gr_top at offset 8
  int GPRSize = FuncInfo->getVarArgsGPRSize();
  if (GPRSize > 0) {
    SDValue GRTop, GRTopAddr;

    GRTopAddr =
        DAG.getNode(ISD::ADD, DL, PtrVT, VAList, DAG.getConstant(8, DL, PtrVT));

    GRTop = DAG.getFrameIndex(FuncInfo->getVarArgsGPRIndex(), PtrVT);
    GRTop = DAG.getNode(ISD::ADD, DL, PtrVT, GRTop,
                        DAG.getConstant(GPRSize, DL, PtrVT));

    MemOps.push_back(DAG.getStore(Chain, DL, GRTop, GRTopAddr,
                                  MachinePointerInfo(SV, 8), false, false, 8));
  }

  // void *__vr_top at offset 16
  int FPRSize = FuncInfo->getVarArgsFPRSize();
  if (FPRSize > 0) {
    SDValue VRTop, VRTopAddr;
    VRTopAddr = DAG.getNode(ISD::ADD, DL, PtrVT, VAList,
                            DAG.getConstant(16, DL, PtrVT));

    VRTop = DAG.getFrameIndex(FuncInfo->getVarArgsFPRIndex(), PtrVT);
    VRTop = DAG.getNode(ISD::ADD, DL, PtrVT, VRTop,
                        DAG.getConstant(FPRSize, DL, PtrVT));

    MemOps.push_back(DAG.getStore(Chain, DL, VRTop, VRTopAddr,
                                  MachinePointerInfo(SV, 16), false, false, 8));
  }

  // int __gr_offs at offset 24
  SDValue GROffsAddr =
      DAG.getNode(ISD::ADD, DL, PtrVT, VAList, DAG.getConstant(24, DL, PtrVT));
  MemOps.push_back(DAG.getStore(Chain, DL,
                                DAG.getConstant(-GPRSize, DL, MVT::i32),
                                GROffsAddr, MachinePointerInfo(SV, 24), false,
                                false, 4));

  // int __vr_offs at offset 28
  SDValue VROffsAddr =
      DAG.getNode(ISD::ADD, DL, PtrVT, VAList, DAG.getConstant(28, DL, PtrVT));
  MemOps.push_back(DAG.getStore(Chain, DL,
                                DAG.getConstant(-FPRSize, DL, MVT::i32),
                                VROffsAddr, MachinePointerInfo(SV, 28), false,
                                false, 4));

  return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOps);
}

SDValue AArch64TargetLowering::LowerVASTART(SDValue Op,
                                            SelectionDAG &DAG) const {
  return Subtarget->isTargetDarwin() ? LowerDarwin_VASTART(Op, DAG)
                                     : LowerAAPCS_VASTART(Op, DAG);
}

SDValue AArch64TargetLowering::LowerVACOPY(SDValue Op,
                                           SelectionDAG &DAG) const {
  // AAPCS has three pointers and two ints (= 32 bytes), Darwin has single
  // pointer.
  SDLoc DL(Op);
  unsigned VaListSize = Subtarget->isTargetDarwin() ? 8 : 32;
  const Value *DestSV = cast<SrcValueSDNode>(Op.getOperand(3))->getValue();
  const Value *SrcSV = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();

  return DAG.getMemcpy(Op.getOperand(0), DL, Op.getOperand(1),
                       Op.getOperand(2),
                       DAG.getConstant(VaListSize, DL, MVT::i32),
                       8, false, false, false, MachinePointerInfo(DestSV),
                       MachinePointerInfo(SrcSV));
}

SDValue AArch64TargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  assert(Subtarget->isTargetDarwin() &&
         "automatic va_arg instruction only works on Darwin");

  const Value *V = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Addr = Op.getOperand(1);
  unsigned Align = Op.getConstantOperandVal(3);
  auto PtrVT = getPointerTy(DAG.getDataLayout());

  SDValue VAList = DAG.getLoad(PtrVT, DL, Chain, Addr, MachinePointerInfo(V),
                               false, false, false, 0);
  Chain = VAList.getValue(1);

  if (Align > 8) {
    assert(((Align & (Align - 1)) == 0) && "Expected Align to be a power of 2");
    VAList = DAG.getNode(ISD::ADD, DL, PtrVT, VAList,
                         DAG.getConstant(Align - 1, DL, PtrVT));
    VAList = DAG.getNode(ISD::AND, DL, PtrVT, VAList,
                         DAG.getConstant(-(int64_t)Align, DL, PtrVT));
  }

  Type *ArgTy = VT.getTypeForEVT(*DAG.getContext());
  uint64_t ArgSize = DAG.getDataLayout().getTypeAllocSize(ArgTy);

  // Scalar integer and FP values smaller than 64 bits are implicitly extended
  // up to 64 bits.  At the very least, we have to increase the striding of the
  // vaargs list to match this, and for FP values we need to introduce
  // FP_ROUND nodes as well.
  if (VT.isInteger() && !VT.isVector())
    ArgSize = 8;
  bool NeedFPTrunc = false;
  if (VT.isFloatingPoint() && !VT.isVector() && VT != MVT::f64) {
    ArgSize = 8;
    NeedFPTrunc = true;
  }

  // Increment the pointer, VAList, to the next vaarg
  SDValue VANext = DAG.getNode(ISD::ADD, DL, PtrVT, VAList,
                               DAG.getConstant(ArgSize, DL, PtrVT));
  // Store the incremented VAList to the legalized pointer
  SDValue APStore = DAG.getStore(Chain, DL, VANext, Addr, MachinePointerInfo(V),
                                 false, false, 0);

  // Load the actual argument out of the pointer VAList
  if (NeedFPTrunc) {
    // Load the value as an f64.
    SDValue WideFP = DAG.getLoad(MVT::f64, DL, APStore, VAList,
                                 MachinePointerInfo(), false, false, false, 0);
    // Round the value down to an f32.
    SDValue NarrowFP = DAG.getNode(ISD::FP_ROUND, DL, VT, WideFP.getValue(0),
                                   DAG.getIntPtrConstant(1, DL));
    SDValue Ops[] = { NarrowFP, WideFP.getValue(1) };
    // Merge the rounded value with the chain output of the load.
    return DAG.getMergeValues(Ops, DL);
  }

  return DAG.getLoad(VT, DL, APStore, VAList, MachinePointerInfo(), false,
                     false, false, 0);
}

SDValue AArch64TargetLowering::LowerFRAMEADDR(SDValue Op,
                                              SelectionDAG &DAG) const {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setFrameAddressIsTaken(true);

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDValue FrameAddr =
      DAG.getCopyFromReg(DAG.getEntryNode(), DL, AArch64::FP, VT);
  while (Depth--)
    FrameAddr = DAG.getLoad(VT, DL, DAG.getEntryNode(), FrameAddr,
                            MachinePointerInfo(), false, false, false, 0);
  return FrameAddr;
}

// FIXME? Maybe this could be a TableGen attribute on some registers and
// this table could be generated automatically from RegInfo.
unsigned AArch64TargetLowering::getRegisterByName(const char* RegName, EVT VT,
                                                  SelectionDAG &DAG) const {
  unsigned Reg = StringSwitch<unsigned>(RegName)
                       .Case("sp", AArch64::SP)
                       .Default(0);
  if (Reg)
    return Reg;
  report_fatal_error(Twine("Invalid register name \""
                              + StringRef(RegName)  + "\"."));
}

SDValue AArch64TargetLowering::LowerRETURNADDR(SDValue Op,
                                               SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MFI->setReturnAddressIsTaken(true);

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (Depth) {
    SDValue FrameAddr = LowerFRAMEADDR(Op, DAG);
    SDValue Offset = DAG.getConstant(8, DL, getPointerTy(DAG.getDataLayout()));
    return DAG.getLoad(VT, DL, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, DL, VT, FrameAddr, Offset),
                       MachinePointerInfo(), false, false, false, 0);
  }

  // Return LR, which contains the return address. Mark it an implicit live-in.
  unsigned Reg = MF.addLiveIn(AArch64::LR, &AArch64::GPR64RegClass);
  return DAG.getCopyFromReg(DAG.getEntryNode(), DL, Reg, VT);
}

/// LowerShiftRightParts - Lower SRA_PARTS, which returns two
/// i64 values and take a 2 x i64 value to shift plus a shift amount.
SDValue AArch64TargetLowering::LowerShiftRightParts(SDValue Op,
                                                    SelectionDAG &DAG) const {
  assert(Op.getNumOperands() == 3 && "Not a double-shift!");
  EVT VT = Op.getValueType();
  unsigned VTBits = VT.getSizeInBits();
  SDLoc dl(Op);
  SDValue ShOpLo = Op.getOperand(0);
  SDValue ShOpHi = Op.getOperand(1);
  SDValue ShAmt = Op.getOperand(2);
  SDValue ARMcc;
  unsigned Opc = (Op.getOpcode() == ISD::SRA_PARTS) ? ISD::SRA : ISD::SRL;

  assert(Op.getOpcode() == ISD::SRA_PARTS || Op.getOpcode() == ISD::SRL_PARTS);

  SDValue RevShAmt = DAG.getNode(ISD::SUB, dl, MVT::i64,
                                 DAG.getConstant(VTBits, dl, MVT::i64), ShAmt);
  SDValue Tmp1 = DAG.getNode(ISD::SRL, dl, VT, ShOpLo, ShAmt);
  SDValue ExtraShAmt = DAG.getNode(ISD::SUB, dl, MVT::i64, ShAmt,
                                   DAG.getConstant(VTBits, dl, MVT::i64));
  SDValue Tmp2 = DAG.getNode(ISD::SHL, dl, VT, ShOpHi, RevShAmt);

  SDValue Cmp = emitComparison(ExtraShAmt, DAG.getConstant(0, dl, MVT::i64),
                               ISD::SETGE, dl, DAG);
  SDValue CCVal = DAG.getConstant(AArch64CC::GE, dl, MVT::i32);

  SDValue FalseValLo = DAG.getNode(ISD::OR, dl, VT, Tmp1, Tmp2);
  SDValue TrueValLo = DAG.getNode(Opc, dl, VT, ShOpHi, ExtraShAmt);
  SDValue Lo =
      DAG.getNode(AArch64ISD::CSEL, dl, VT, TrueValLo, FalseValLo, CCVal, Cmp);

  // AArch64 shifts larger than the register width are wrapped rather than
  // clamped, so we can't just emit "hi >> x".
  SDValue FalseValHi = DAG.getNode(Opc, dl, VT, ShOpHi, ShAmt);
  SDValue TrueValHi = Opc == ISD::SRA
                          ? DAG.getNode(Opc, dl, VT, ShOpHi,
                                        DAG.getConstant(VTBits - 1, dl,
                                                        MVT::i64))
                          : DAG.getConstant(0, dl, VT);
  SDValue Hi =
      DAG.getNode(AArch64ISD::CSEL, dl, VT, TrueValHi, FalseValHi, CCVal, Cmp);

  SDValue Ops[2] = { Lo, Hi };
  return DAG.getMergeValues(Ops, dl);
}

/// LowerShiftLeftParts - Lower SHL_PARTS, which returns two
/// i64 values and take a 2 x i64 value to shift plus a shift amount.
SDValue AArch64TargetLowering::LowerShiftLeftParts(SDValue Op,
                                                 SelectionDAG &DAG) const {
  assert(Op.getNumOperands() == 3 && "Not a double-shift!");
  EVT VT = Op.getValueType();
  unsigned VTBits = VT.getSizeInBits();
  SDLoc dl(Op);
  SDValue ShOpLo = Op.getOperand(0);
  SDValue ShOpHi = Op.getOperand(1);
  SDValue ShAmt = Op.getOperand(2);
  SDValue ARMcc;

  assert(Op.getOpcode() == ISD::SHL_PARTS);
  SDValue RevShAmt = DAG.getNode(ISD::SUB, dl, MVT::i64,
                                 DAG.getConstant(VTBits, dl, MVT::i64), ShAmt);
  SDValue Tmp1 = DAG.getNode(ISD::SRL, dl, VT, ShOpLo, RevShAmt);
  SDValue ExtraShAmt = DAG.getNode(ISD::SUB, dl, MVT::i64, ShAmt,
                                   DAG.getConstant(VTBits, dl, MVT::i64));
  SDValue Tmp2 = DAG.getNode(ISD::SHL, dl, VT, ShOpHi, ShAmt);
  SDValue Tmp3 = DAG.getNode(ISD::SHL, dl, VT, ShOpLo, ExtraShAmt);

  SDValue FalseVal = DAG.getNode(ISD::OR, dl, VT, Tmp1, Tmp2);

  SDValue Cmp = emitComparison(ExtraShAmt, DAG.getConstant(0, dl, MVT::i64),
                               ISD::SETGE, dl, DAG);
  SDValue CCVal = DAG.getConstant(AArch64CC::GE, dl, MVT::i32);
  SDValue Hi =
      DAG.getNode(AArch64ISD::CSEL, dl, VT, Tmp3, FalseVal, CCVal, Cmp);

  // AArch64 shifts of larger than register sizes are wrapped rather than
  // clamped, so we can't just emit "lo << a" if a is too big.
  SDValue TrueValLo = DAG.getConstant(0, dl, VT);
  SDValue FalseValLo = DAG.getNode(ISD::SHL, dl, VT, ShOpLo, ShAmt);
  SDValue Lo =
      DAG.getNode(AArch64ISD::CSEL, dl, VT, TrueValLo, FalseValLo, CCVal, Cmp);

  SDValue Ops[2] = { Lo, Hi };
  return DAG.getMergeValues(Ops, dl);
}

bool AArch64TargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  // The AArch64 target doesn't support folding offsets into global addresses.
  return false;
}

bool AArch64TargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  // We can materialize #0.0 as fmov $Rd, XZR for 64-bit and 32-bit cases.
  // FIXME: We should be able to handle f128 as well with a clever lowering.
  if (Imm.isPosZero() && (VT == MVT::f64 || VT == MVT::f32))
    return true;

  if (VT == MVT::f64)
    return AArch64_AM::getFP64Imm(Imm) != -1;
  else if (VT == MVT::f32)
    return AArch64_AM::getFP32Imm(Imm) != -1;
  return false;
}

//===----------------------------------------------------------------------===//
//                          AArch64 Optimization Hooks
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                          AArch64 Inline Assembly Support
//===----------------------------------------------------------------------===//

// Table of Constraints
// TODO: This is the current set of constraints supported by ARM for the
// compiler, not all of them may make sense, e.g. S may be difficult to support.
//
// r - A general register
// w - An FP/SIMD register of some size in the range v0-v31
// x - An FP/SIMD register of some size in the range v0-v15
// I - Constant that can be used with an ADD instruction
// J - Constant that can be used with a SUB instruction
// K - Constant that can be used with a 32-bit logical instruction
// L - Constant that can be used with a 64-bit logical instruction
// M - Constant that can be used as a 32-bit MOV immediate
// N - Constant that can be used as a 64-bit MOV immediate
// Q - A memory reference with base register and no offset
// S - A symbolic address
// Y - Floating point constant zero
// Z - Integer constant zero
//
//   Note that general register operands will be output using their 64-bit x
// register name, whatever the size of the variable, unless the asm operand
// is prefixed by the %w modifier. Floating-point and SIMD register operands
// will be output with the v prefix unless prefixed by the %b, %h, %s, %d or
// %q modifier.

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
AArch64TargetLowering::ConstraintType
AArch64TargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'z':
      return C_Other;
    case 'x':
    case 'w':
      return C_RegisterClass;
    // An address with a single base register. Due to the way we
    // currently handle addresses it is the same as 'r'.
    case 'Q':
      return C_Memory;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
AArch64TargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
  // If we don't have a value, we can't do a match,
  // but allow it at the lowest weight.
  if (!CallOperandVal)
    return CW_Default;
  Type *type = CallOperandVal->getType();
  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;
  case 'x':
  case 'w':
    if (type->isFloatingPointTy() || type->isVectorTy())
      weight = CW_Register;
    break;
  case 'z':
    weight = CW_Constant;
    break;
  }
  return weight;
}

std::pair<unsigned, const TargetRegisterClass *>
AArch64TargetLowering::getRegForInlineAsmConstraint(
    const TargetRegisterInfo *TRI, StringRef Constraint, MVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      if (VT.getSizeInBits() == 64)
        return std::make_pair(0U, &AArch64::GPR64commonRegClass);
      return std::make_pair(0U, &AArch64::GPR32commonRegClass);
    case 'w':
      if (VT == MVT::f32)
        return std::make_pair(0U, &AArch64::FPR32RegClass);
      if (VT.getSizeInBits() == 64)
        return std::make_pair(0U, &AArch64::FPR64RegClass);
      if (VT.getSizeInBits() == 128)
        return std::make_pair(0U, &AArch64::FPR128RegClass);
      break;
    // The instructions that this constraint is designed for can
    // only take 128-bit registers so just use that regclass.
    case 'x':
      if (VT.getSizeInBits() == 128)
        return std::make_pair(0U, &AArch64::FPR128_loRegClass);
      break;
    }
  }
  if (StringRef("{cc}").equals_lower(Constraint))
    return std::make_pair(unsigned(AArch64::NZCV), &AArch64::CCRRegClass);

  // Use the default implementation in TargetLowering to convert the register
  // constraint into a member of a register class.
  std::pair<unsigned, const TargetRegisterClass *> Res;
  Res = TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);

  // Not found as a standard register?
  if (!Res.second) {
    unsigned Size = Constraint.size();
    if ((Size == 4 || Size == 5) && Constraint[0] == '{' &&
        tolower(Constraint[1]) == 'v' && Constraint[Size - 1] == '}') {
      int RegNo;
      bool Failed = Constraint.slice(2, Size - 1).getAsInteger(10, RegNo);
      if (!Failed && RegNo >= 0 && RegNo <= 31) {
        // v0 - v31 are aliases of q0 - q31.
        // By default we'll emit v0-v31 for this unless there's a modifier where
        // we'll emit the correct register as well.
        Res.first = AArch64::FPR128RegClass.getRegister(RegNo);
        Res.second = &AArch64::FPR128RegClass;
      }
    }
  }

  return Res;
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void AArch64TargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, std::string &Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  SDValue Result;

  // Currently only support length 1 constraints.
  if (Constraint.length() != 1)
    return;

  char ConstraintLetter = Constraint[0];
  switch (ConstraintLetter) {
  default:
    break;

  // This set of constraints deal with valid constants for various instructions.
  // Validate and return a target constant for them if we can.
  case 'z': {
    // 'z' maps to xzr or wzr so it needs an input of 0.
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    if (!C || C->getZExtValue() != 0)
      return;

    if (Op.getValueType() == MVT::i64)
      Result = DAG.getRegister(AArch64::XZR, MVT::i64);
    else
      Result = DAG.getRegister(AArch64::WZR, MVT::i32);
    break;
  }

  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    if (!C)
      return;

    // Grab the value and do some validation.
    uint64_t CVal = C->getZExtValue();
    switch (ConstraintLetter) {
    // The I constraint applies only to simple ADD or SUB immediate operands:
    // i.e. 0 to 4095 with optional shift by 12
    // The J constraint applies only to ADD or SUB immediates that would be
    // valid when negated, i.e. if [an add pattern] were to be output as a SUB
    // instruction [or vice versa], in other words -1 to -4095 with optional
    // left shift by 12.
    case 'I':
      if (isUInt<12>(CVal) || isShiftedUInt<12, 12>(CVal))
        break;
      return;
    case 'J': {
      uint64_t NVal = -C->getSExtValue();
      if (isUInt<12>(NVal) || isShiftedUInt<12, 12>(NVal)) {
        CVal = C->getSExtValue();
        break;
      }
      return;
    }
    // The K and L constraints apply *only* to logical immediates, including
    // what used to be the MOVI alias for ORR (though the MOVI alias has now
    // been removed and MOV should be used). So these constraints have to
    // distinguish between bit patterns that are valid 32-bit or 64-bit
    // "bitmask immediates": for example 0xaaaaaaaa is a valid bimm32 (K), but
    // not a valid bimm64 (L) where 0xaaaaaaaaaaaaaaaa would be valid, and vice
    // versa.
    case 'K':
      if (AArch64_AM::isLogicalImmediate(CVal, 32))
        break;
      return;
    case 'L':
      if (AArch64_AM::isLogicalImmediate(CVal, 64))
        break;
      return;
    // The M and N constraints are a superset of K and L respectively, for use
    // with the MOV (immediate) alias. As well as the logical immediates they
    // also match 32 or 64-bit immediates that can be loaded either using a
    // *single* MOVZ or MOVN , such as 32-bit 0x12340000, 0x00001234, 0xffffedca
    // (M) or 64-bit 0x1234000000000000 (N) etc.
    // As a note some of this code is liberally stolen from the asm parser.
    case 'M': {
      if (!isUInt<32>(CVal))
        return;
      if (AArch64_AM::isLogicalImmediate(CVal, 32))
        break;
      if ((CVal & 0xFFFF) == CVal)
        break;
      if ((CVal & 0xFFFF0000ULL) == CVal)
        break;
      uint64_t NCVal = ~(uint32_t)CVal;
      if ((NCVal & 0xFFFFULL) == NCVal)
        break;
      if ((NCVal & 0xFFFF0000ULL) == NCVal)
        break;
      return;
    }
    case 'N': {
      if (AArch64_AM::isLogicalImmediate(CVal, 64))
        break;
      if ((CVal & 0xFFFFULL) == CVal)
        break;
      if ((CVal & 0xFFFF0000ULL) == CVal)
        break;
      if ((CVal & 0xFFFF00000000ULL) == CVal)
        break;
      if ((CVal & 0xFFFF000000000000ULL) == CVal)
        break;
      uint64_t NCVal = ~CVal;
      if ((NCVal & 0xFFFFULL) == NCVal)
        break;
      if ((NCVal & 0xFFFF0000ULL) == NCVal)
        break;
      if ((NCVal & 0xFFFF00000000ULL) == NCVal)
        break;
      if ((NCVal & 0xFFFF000000000000ULL) == NCVal)
        break;
      return;
    }
    default:
      return;
    }

    // All assembler immediates are 64-bit integers.
    Result = DAG.getTargetConstant(CVal, SDLoc(Op), MVT::i64);
    break;
  }

  if (Result.getNode()) {
    Ops.push_back(Result);
    return;
  }

  return TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

//===----------------------------------------------------------------------===//
//                     AArch64 Advanced SIMD Support
//===----------------------------------------------------------------------===//

/// WidenVector - Given a value in the V64 register class, produce the
/// equivalent value in the V128 register class.
static SDValue WidenVector(SDValue V64Reg, SelectionDAG &DAG) {
  EVT VT = V64Reg.getValueType();
  unsigned NarrowSize = VT.getVectorNumElements();
  MVT EltTy = VT.getVectorElementType().getSimpleVT();
  MVT WideTy = MVT::getVectorVT(EltTy, 2 * NarrowSize);
  SDLoc DL(V64Reg);

  return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, WideTy, DAG.getUNDEF(WideTy),
                     V64Reg, DAG.getConstant(0, DL, MVT::i32));
}

/// getExtFactor - Determine the adjustment factor for the position when
/// generating an "extract from vector registers" instruction.
static unsigned getExtFactor(SDValue &V) {
  EVT EltType = V.getValueType().getVectorElementType();
  return EltType.getSizeInBits() / 8;
}

/// NarrowVector - Given a value in the V128 register class, produce the
/// equivalent value in the V64 register class.
static SDValue NarrowVector(SDValue V128Reg, SelectionDAG &DAG) {
  EVT VT = V128Reg.getValueType();
  unsigned WideSize = VT.getVectorNumElements();
  MVT EltTy = VT.getVectorElementType().getSimpleVT();
  MVT NarrowTy = MVT::getVectorVT(EltTy, WideSize / 2);
  SDLoc DL(V128Reg);

  return DAG.getTargetExtractSubreg(AArch64::dsub, DL, NarrowTy, V128Reg);
}

// Gather data to see if the operation can be modelled as a
// shuffle in combination with VEXTs.
SDValue AArch64TargetLowering::ReconstructShuffle(SDValue Op,
                                                  SelectionDAG &DAG) const {
  assert(Op.getOpcode() == ISD::BUILD_VECTOR && "Unknown opcode!");
  SDLoc dl(Op);
  EVT VT = Op.getValueType();
  unsigned NumElts = VT.getVectorNumElements();

  struct ShuffleSourceInfo {
    SDValue Vec;
    unsigned MinElt;
    unsigned MaxElt;

    // We may insert some combination of BITCASTs and VEXT nodes to force Vec to
    // be compatible with the shuffle we intend to construct. As a result
    // ShuffleVec will be some sliding window into the original Vec.
    SDValue ShuffleVec;

    // Code should guarantee that element i in Vec starts at element "WindowBase
    // + i * WindowScale in ShuffleVec".
    int WindowBase;
    int WindowScale;

    bool operator ==(SDValue OtherVec) { return Vec == OtherVec; }
    ShuffleSourceInfo(SDValue Vec)
        : Vec(Vec), MinElt(UINT_MAX), MaxElt(0), ShuffleVec(Vec), WindowBase(0),
          WindowScale(1) {}
  };

  // First gather all vectors used as an immediate source for this BUILD_VECTOR
  // node.
  SmallVector<ShuffleSourceInfo, 2> Sources;
  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue V = Op.getOperand(i);
    if (V.getOpcode() == ISD::UNDEF)
      continue;
    else if (V.getOpcode() != ISD::EXTRACT_VECTOR_ELT) {
      // A shuffle can only come from building a vector from various
      // elements of other vectors.
      return SDValue();
    }

    // Add this element source to the list if it's not already there.
    SDValue SourceVec = V.getOperand(0);
    auto Source = std::find(Sources.begin(), Sources.end(), SourceVec);
    if (Source == Sources.end())
      Source = Sources.insert(Sources.end(), ShuffleSourceInfo(SourceVec));

    // Update the minimum and maximum lane number seen.
    unsigned EltNo = cast<ConstantSDNode>(V.getOperand(1))->getZExtValue();
    Source->MinElt = std::min(Source->MinElt, EltNo);
    Source->MaxElt = std::max(Source->MaxElt, EltNo);
  }

  // Currently only do something sane when at most two source vectors
  // are involved.
  if (Sources.size() > 2)
    return SDValue();

  // Find out the smallest element size among result and two sources, and use
  // it as element size to build the shuffle_vector.
  EVT SmallestEltTy = VT.getVectorElementType();
  for (auto &Source : Sources) {
    EVT SrcEltTy = Source.Vec.getValueType().getVectorElementType();
    if (SrcEltTy.bitsLT(SmallestEltTy)) {
      SmallestEltTy = SrcEltTy;
    }
  }
  unsigned ResMultiplier =
      VT.getVectorElementType().getSizeInBits() / SmallestEltTy.getSizeInBits();
  NumElts = VT.getSizeInBits() / SmallestEltTy.getSizeInBits();
  EVT ShuffleVT = EVT::getVectorVT(*DAG.getContext(), SmallestEltTy, NumElts);

  // If the source vector is too wide or too narrow, we may nevertheless be able
  // to construct a compatible shuffle either by concatenating it with UNDEF or
  // extracting a suitable range of elements.
  for (auto &Src : Sources) {
    EVT SrcVT = Src.ShuffleVec.getValueType();

    if (SrcVT.getSizeInBits() == VT.getSizeInBits())
      continue;

    // This stage of the search produces a source with the same element type as
    // the original, but with a total width matching the BUILD_VECTOR output.
    EVT EltVT = SrcVT.getVectorElementType();
    unsigned NumSrcElts = VT.getSizeInBits() / EltVT.getSizeInBits();
    EVT DestVT = EVT::getVectorVT(*DAG.getContext(), EltVT, NumSrcElts);

    if (SrcVT.getSizeInBits() < VT.getSizeInBits()) {
      assert(2 * SrcVT.getSizeInBits() == VT.getSizeInBits());
      // We can pad out the smaller vector for free, so if it's part of a
      // shuffle...
      Src.ShuffleVec =
          DAG.getNode(ISD::CONCAT_VECTORS, dl, DestVT, Src.ShuffleVec,
                      DAG.getUNDEF(Src.ShuffleVec.getValueType()));
      continue;
    }

    assert(SrcVT.getSizeInBits() == 2 * VT.getSizeInBits());

    if (Src.MaxElt - Src.MinElt >= NumSrcElts) {
      // Span too large for a VEXT to cope
      return SDValue();
    }

    if (Src.MinElt >= NumSrcElts) {
      // The extraction can just take the second half
      Src.ShuffleVec =
          DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, DestVT, Src.ShuffleVec,
                      DAG.getConstant(NumSrcElts, dl, MVT::i64));
      Src.WindowBase = -NumSrcElts;
    } else if (Src.MaxElt < NumSrcElts) {
      // The extraction can just take the first half
      Src.ShuffleVec =
          DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, DestVT, Src.ShuffleVec,
                      DAG.getConstant(0, dl, MVT::i64));
    } else {
      // An actual VEXT is needed
      SDValue VEXTSrc1 =
          DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, DestVT, Src.ShuffleVec,
                      DAG.getConstant(0, dl, MVT::i64));
      SDValue VEXTSrc2 =
          DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, DestVT, Src.ShuffleVec,
                      DAG.getConstant(NumSrcElts, dl, MVT::i64));
      unsigned Imm = Src.MinElt * getExtFactor(VEXTSrc1);

      Src.ShuffleVec = DAG.getNode(AArch64ISD::EXT, dl, DestVT, VEXTSrc1,
                                   VEXTSrc2,
                                   DAG.getConstant(Imm, dl, MVT::i32));
      Src.WindowBase = -Src.MinElt;
    }
  }

  // Another possible incompatibility occurs from the vector element types. We
  // can fix this by bitcasting the source vectors to the same type we intend
  // for the shuffle.
  for (auto &Src : Sources) {
    EVT SrcEltTy = Src.ShuffleVec.getValueType().getVectorElementType();
    if (SrcEltTy == SmallestEltTy)
      continue;
    assert(ShuffleVT.getVectorElementType() == SmallestEltTy);
    Src.ShuffleVec = DAG.getNode(ISD::BITCAST, dl, ShuffleVT, Src.ShuffleVec);
    Src.WindowScale = SrcEltTy.getSizeInBits() / SmallestEltTy.getSizeInBits();
    Src.WindowBase *= Src.WindowScale;
  }

  // Final sanity check before we try to actually produce a shuffle.
  DEBUG(
    for (auto Src : Sources)
      assert(Src.ShuffleVec.getValueType() == ShuffleVT);
  );

  // The stars all align, our next step is to produce the mask for the shuffle.
  SmallVector<int, 8> Mask(ShuffleVT.getVectorNumElements(), -1);
  int BitsPerShuffleLane = ShuffleVT.getVectorElementType().getSizeInBits();
  for (unsigned i = 0; i < VT.getVectorNumElements(); ++i) {
    SDValue Entry = Op.getOperand(i);
    if (Entry.getOpcode() == ISD::UNDEF)
      continue;

    auto Src = std::find(Sources.begin(), Sources.end(), Entry.getOperand(0));
    int EltNo = cast<ConstantSDNode>(Entry.getOperand(1))->getSExtValue();

    // EXTRACT_VECTOR_ELT performs an implicit any_ext; BUILD_VECTOR an implicit
    // trunc. So only std::min(SrcBits, DestBits) actually get defined in this
    // segment.
    EVT OrigEltTy = Entry.getOperand(0).getValueType().getVectorElementType();
    int BitsDefined = std::min(OrigEltTy.getSizeInBits(),
                               VT.getVectorElementType().getSizeInBits());
    int LanesDefined = BitsDefined / BitsPerShuffleLane;

    // This source is expected to fill ResMultiplier lanes of the final shuffle,
    // starting at the appropriate offset.
    int *LaneMask = &Mask[i * ResMultiplier];

    int ExtractBase = EltNo * Src->WindowScale + Src->WindowBase;
    ExtractBase += NumElts * (Src - Sources.begin());
    for (int j = 0; j < LanesDefined; ++j)
      LaneMask[j] = ExtractBase + j;
  }

  // Final check before we try to produce nonsense...
  if (!isShuffleMaskLegal(Mask, ShuffleVT))
    return SDValue();

  SDValue ShuffleOps[] = { DAG.getUNDEF(ShuffleVT), DAG.getUNDEF(ShuffleVT) };
  for (unsigned i = 0; i < Sources.size(); ++i)
    ShuffleOps[i] = Sources[i].ShuffleVec;

  SDValue Shuffle = DAG.getVectorShuffle(ShuffleVT, dl, ShuffleOps[0],
                                         ShuffleOps[1], &Mask[0]);
  return DAG.getNode(ISD::BITCAST, dl, VT, Shuffle);
}

// check if an EXT instruction can handle the shuffle mask when the
// vector sources of the shuffle are the same.
static bool isSingletonEXTMask(ArrayRef<int> M, EVT VT, unsigned &Imm) {
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

    if (M[i] < 0)
      continue; // ignore UNDEF indices
    if (ExpectedElt != static_cast<unsigned>(M[i]))
      return false;
  }

  return true;
}

// check if an EXT instruction can handle the shuffle mask when the
// vector sources of the shuffle are different.
static bool isEXTMask(ArrayRef<int> M, EVT VT, bool &ReverseEXT,
                      unsigned &Imm) {
  // Look for the first non-undef element.
  const int *FirstRealElt = std::find_if(M.begin(), M.end(),
      [](int Elt) {return Elt >= 0;});

  // Benefit form APInt to handle overflow when calculating expected element.
  unsigned NumElts = VT.getVectorNumElements();
  unsigned MaskBits = APInt(32, NumElts * 2).logBase2();
  APInt ExpectedElt = APInt(MaskBits, *FirstRealElt + 1);
  // The following shuffle indices must be the successive elements after the
  // first real element.
  const int *FirstWrongElt = std::find_if(FirstRealElt + 1, M.end(),
      [&](int Elt) {return Elt != ExpectedElt++ && Elt != -1;});
  if (FirstWrongElt != M.end())
    return false;

  // The index of an EXT is the first element if it is not UNDEF.
  // Watch out for the beginning UNDEFs. The EXT index should be the expected
  // value of the first element.  E.g. 
  // <-1, -1, 3, ...> is treated as <1, 2, 3, ...>.
  // <-1, -1, 0, 1, ...> is treated as <2*NumElts-2, 2*NumElts-1, 0, 1, ...>.
  // ExpectedElt is the last mask index plus 1.
  Imm = ExpectedElt.getZExtValue();

  // There are two difference cases requiring to reverse input vectors.
  // For example, for vector <4 x i32> we have the following cases,
  // Case 1: shufflevector(<4 x i32>,<4 x i32>,<-1, -1, -1, 0>)
  // Case 2: shufflevector(<4 x i32>,<4 x i32>,<-1, -1, 7, 0>)
  // For both cases, we finally use mask <5, 6, 7, 0>, which requires
  // to reverse two input vectors.
  if (Imm < NumElts)
    ReverseEXT = true;
  else
    Imm -= NumElts;

  return true;
}

/// isREVMask - Check if a vector shuffle corresponds to a REV
/// instruction with the specified blocksize.  (The order of the elements
/// within each block of the vector is reversed.)
static bool isREVMask(ArrayRef<int> M, EVT VT, unsigned BlockSize) {
  assert((BlockSize == 16 || BlockSize == 32 || BlockSize == 64) &&
         "Only possible block sizes for REV are: 16, 32, 64");

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
    if (M[i] < 0)
      continue; // ignore UNDEF indices
    if ((unsigned)M[i] != (i - i % BlockElts) + (BlockElts - 1 - i % BlockElts))
      return false;
  }

  return true;
}

static bool isZIPMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  unsigned Idx = WhichResult * NumElts / 2;
  for (unsigned i = 0; i != NumElts; i += 2) {
    if ((M[i] >= 0 && (unsigned)M[i] != Idx) ||
        (M[i + 1] >= 0 && (unsigned)M[i + 1] != Idx + NumElts))
      return false;
    Idx += 1;
  }

  return true;
}

static bool isUZPMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i != NumElts; ++i) {
    if (M[i] < 0)
      continue; // ignore UNDEF indices
    if ((unsigned)M[i] != 2 * i + WhichResult)
      return false;
  }

  return true;
}

static bool isTRNMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i < NumElts; i += 2) {
    if ((M[i] >= 0 && (unsigned)M[i] != i + WhichResult) ||
        (M[i + 1] >= 0 && (unsigned)M[i + 1] != i + NumElts + WhichResult))
      return false;
  }
  return true;
}

/// isZIP_v_undef_Mask - Special case of isZIPMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 0, 1, 1> instead of <0, 4, 1, 5>.
static bool isZIP_v_undef_Mask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  unsigned Idx = WhichResult * NumElts / 2;
  for (unsigned i = 0; i != NumElts; i += 2) {
    if ((M[i] >= 0 && (unsigned)M[i] != Idx) ||
        (M[i + 1] >= 0 && (unsigned)M[i + 1] != Idx))
      return false;
    Idx += 1;
  }

  return true;
}

/// isUZP_v_undef_Mask - Special case of isUZPMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 2, 0, 2> instead of <0, 2, 4, 6>,
static bool isUZP_v_undef_Mask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned Half = VT.getVectorNumElements() / 2;
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned j = 0; j != 2; ++j) {
    unsigned Idx = WhichResult;
    for (unsigned i = 0; i != Half; ++i) {
      int MIdx = M[i + j * Half];
      if (MIdx >= 0 && (unsigned)MIdx != Idx)
        return false;
      Idx += 2;
    }
  }

  return true;
}

/// isTRN_v_undef_Mask - Special case of isTRNMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 0, 2, 2> instead of <0, 4, 2, 6>.
static bool isTRN_v_undef_Mask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned NumElts = VT.getVectorNumElements();
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i < NumElts; i += 2) {
    if ((M[i] >= 0 && (unsigned)M[i] != i + WhichResult) ||
        (M[i + 1] >= 0 && (unsigned)M[i + 1] != i + WhichResult))
      return false;
  }
  return true;
}

static bool isINSMask(ArrayRef<int> M, int NumInputElements,
                      bool &DstIsLeft, int &Anomaly) {
  if (M.size() != static_cast<size_t>(NumInputElements))
    return false;

  int NumLHSMatch = 0, NumRHSMatch = 0;
  int LastLHSMismatch = -1, LastRHSMismatch = -1;

  for (int i = 0; i < NumInputElements; ++i) {
    if (M[i] == -1) {
      ++NumLHSMatch;
      ++NumRHSMatch;
      continue;
    }

    if (M[i] == i)
      ++NumLHSMatch;
    else
      LastLHSMismatch = i;

    if (M[i] == i + NumInputElements)
      ++NumRHSMatch;
    else
      LastRHSMismatch = i;
  }

  if (NumLHSMatch == NumInputElements - 1) {
    DstIsLeft = true;
    Anomaly = LastLHSMismatch;
    return true;
  } else if (NumRHSMatch == NumInputElements - 1) {
    DstIsLeft = false;
    Anomaly = LastRHSMismatch;
    return true;
  }

  return false;
}

static bool isConcatMask(ArrayRef<int> Mask, EVT VT, bool SplitLHS) {
  if (VT.getSizeInBits() != 128)
    return false;

  unsigned NumElts = VT.getVectorNumElements();

  for (int I = 0, E = NumElts / 2; I != E; I++) {
    if (Mask[I] != I)
      return false;
  }

  int Offset = NumElts / 2;
  for (int I = NumElts / 2, E = NumElts; I != E; I++) {
    if (Mask[I] != I + SplitLHS * Offset)
      return false;
  }

  return true;
}

static SDValue tryFormConcatFromShuffle(SDValue Op, SelectionDAG &DAG) {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue V0 = Op.getOperand(0);
  SDValue V1 = Op.getOperand(1);
  ArrayRef<int> Mask = cast<ShuffleVectorSDNode>(Op)->getMask();

  if (VT.getVectorElementType() != V0.getValueType().getVectorElementType() ||
      VT.getVectorElementType() != V1.getValueType().getVectorElementType())
    return SDValue();

  bool SplitV0 = V0.getValueType().getSizeInBits() == 128;

  if (!isConcatMask(Mask, VT, SplitV0))
    return SDValue();

  EVT CastVT = EVT::getVectorVT(*DAG.getContext(), VT.getVectorElementType(),
                                VT.getVectorNumElements() / 2);
  if (SplitV0) {
    V0 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, CastVT, V0,
                     DAG.getConstant(0, DL, MVT::i64));
  }
  if (V1.getValueType().getSizeInBits() == 128) {
    V1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, CastVT, V1,
                     DAG.getConstant(0, DL, MVT::i64));
  }
  return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, V0, V1);
}

/// GeneratePerfectShuffle - Given an entry in the perfect-shuffle table, emit
/// the specified operations to build the shuffle.
static SDValue GeneratePerfectShuffle(unsigned PFEntry, SDValue LHS,
                                      SDValue RHS, SelectionDAG &DAG,
                                      SDLoc dl) {
  unsigned OpNum = (PFEntry >> 26) & 0x0F;
  unsigned LHSID = (PFEntry >> 13) & ((1 << 13) - 1);
  unsigned RHSID = (PFEntry >> 0) & ((1 << 13) - 1);

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
    if (LHSID == (1 * 9 + 2) * 9 + 3)
      return LHS;
    assert(LHSID == ((4 * 9 + 5) * 9 + 6) * 9 + 7 && "Illegal OP_COPY!");
    return RHS;
  }

  SDValue OpLHS, OpRHS;
  OpLHS = GeneratePerfectShuffle(PerfectShuffleTable[LHSID], LHS, RHS, DAG, dl);
  OpRHS = GeneratePerfectShuffle(PerfectShuffleTable[RHSID], LHS, RHS, DAG, dl);
  EVT VT = OpLHS.getValueType();

  switch (OpNum) {
  default:
    llvm_unreachable("Unknown shuffle opcode!");
  case OP_VREV:
    // VREV divides the vector in half and swaps within the half.
    if (VT.getVectorElementType() == MVT::i32 ||
        VT.getVectorElementType() == MVT::f32)
      return DAG.getNode(AArch64ISD::REV64, dl, VT, OpLHS);
    // vrev <4 x i16> -> REV32
    if (VT.getVectorElementType() == MVT::i16 ||
        VT.getVectorElementType() == MVT::f16)
      return DAG.getNode(AArch64ISD::REV32, dl, VT, OpLHS);
    // vrev <4 x i8> -> REV16
    assert(VT.getVectorElementType() == MVT::i8);
    return DAG.getNode(AArch64ISD::REV16, dl, VT, OpLHS);
  case OP_VDUP0:
  case OP_VDUP1:
  case OP_VDUP2:
  case OP_VDUP3: {
    EVT EltTy = VT.getVectorElementType();
    unsigned Opcode;
    if (EltTy == MVT::i8)
      Opcode = AArch64ISD::DUPLANE8;
    else if (EltTy == MVT::i16 || EltTy == MVT::f16)
      Opcode = AArch64ISD::DUPLANE16;
    else if (EltTy == MVT::i32 || EltTy == MVT::f32)
      Opcode = AArch64ISD::DUPLANE32;
    else if (EltTy == MVT::i64 || EltTy == MVT::f64)
      Opcode = AArch64ISD::DUPLANE64;
    else
      llvm_unreachable("Invalid vector element type?");

    if (VT.getSizeInBits() == 64)
      OpLHS = WidenVector(OpLHS, DAG);
    SDValue Lane = DAG.getConstant(OpNum - OP_VDUP0, dl, MVT::i64);
    return DAG.getNode(Opcode, dl, VT, OpLHS, Lane);
  }
  case OP_VEXT1:
  case OP_VEXT2:
  case OP_VEXT3: {
    unsigned Imm = (OpNum - OP_VEXT1 + 1) * getExtFactor(OpLHS);
    return DAG.getNode(AArch64ISD::EXT, dl, VT, OpLHS, OpRHS,
                       DAG.getConstant(Imm, dl, MVT::i32));
  }
  case OP_VUZPL:
    return DAG.getNode(AArch64ISD::UZP1, dl, DAG.getVTList(VT, VT), OpLHS,
                       OpRHS);
  case OP_VUZPR:
    return DAG.getNode(AArch64ISD::UZP2, dl, DAG.getVTList(VT, VT), OpLHS,
                       OpRHS);
  case OP_VZIPL:
    return DAG.getNode(AArch64ISD::ZIP1, dl, DAG.getVTList(VT, VT), OpLHS,
                       OpRHS);
  case OP_VZIPR:
    return DAG.getNode(AArch64ISD::ZIP2, dl, DAG.getVTList(VT, VT), OpLHS,
                       OpRHS);
  case OP_VTRNL:
    return DAG.getNode(AArch64ISD::TRN1, dl, DAG.getVTList(VT, VT), OpLHS,
                       OpRHS);
  case OP_VTRNR:
    return DAG.getNode(AArch64ISD::TRN2, dl, DAG.getVTList(VT, VT), OpLHS,
                       OpRHS);
  }
}

static SDValue GenerateTBL(SDValue Op, ArrayRef<int> ShuffleMask,
                           SelectionDAG &DAG) {
  // Check to see if we can use the TBL instruction.
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  SDLoc DL(Op);

  EVT EltVT = Op.getValueType().getVectorElementType();
  unsigned BytesPerElt = EltVT.getSizeInBits() / 8;

  SmallVector<SDValue, 8> TBLMask;
  for (int Val : ShuffleMask) {
    for (unsigned Byte = 0; Byte < BytesPerElt; ++Byte) {
      unsigned Offset = Byte + Val * BytesPerElt;
      TBLMask.push_back(DAG.getConstant(Offset, DL, MVT::i32));
    }
  }

  MVT IndexVT = MVT::v8i8;
  unsigned IndexLen = 8;
  if (Op.getValueType().getSizeInBits() == 128) {
    IndexVT = MVT::v16i8;
    IndexLen = 16;
  }

  SDValue V1Cst = DAG.getNode(ISD::BITCAST, DL, IndexVT, V1);
  SDValue V2Cst = DAG.getNode(ISD::BITCAST, DL, IndexVT, V2);

  SDValue Shuffle;
  if (V2.getNode()->getOpcode() == ISD::UNDEF) {
    if (IndexLen == 8)
      V1Cst = DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v16i8, V1Cst, V1Cst);
    Shuffle = DAG.getNode(
        ISD::INTRINSIC_WO_CHAIN, DL, IndexVT,
        DAG.getConstant(Intrinsic::aarch64_neon_tbl1, DL, MVT::i32), V1Cst,
        DAG.getNode(ISD::BUILD_VECTOR, DL, IndexVT,
                    makeArrayRef(TBLMask.data(), IndexLen)));
  } else {
    if (IndexLen == 8) {
      V1Cst = DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v16i8, V1Cst, V2Cst);
      Shuffle = DAG.getNode(
          ISD::INTRINSIC_WO_CHAIN, DL, IndexVT,
          DAG.getConstant(Intrinsic::aarch64_neon_tbl1, DL, MVT::i32), V1Cst,
          DAG.getNode(ISD::BUILD_VECTOR, DL, IndexVT,
                      makeArrayRef(TBLMask.data(), IndexLen)));
    } else {
      // FIXME: We cannot, for the moment, emit a TBL2 instruction because we
      // cannot currently represent the register constraints on the input
      // table registers.
      //  Shuffle = DAG.getNode(AArch64ISD::TBL2, DL, IndexVT, V1Cst, V2Cst,
      //                   DAG.getNode(ISD::BUILD_VECTOR, DL, IndexVT,
      //                               &TBLMask[0], IndexLen));
      Shuffle = DAG.getNode(
          ISD::INTRINSIC_WO_CHAIN, DL, IndexVT,
          DAG.getConstant(Intrinsic::aarch64_neon_tbl2, DL, MVT::i32),
          V1Cst, V2Cst,
          DAG.getNode(ISD::BUILD_VECTOR, DL, IndexVT,
                      makeArrayRef(TBLMask.data(), IndexLen)));
    }
  }
  return DAG.getNode(ISD::BITCAST, DL, Op.getValueType(), Shuffle);
}

static unsigned getDUPLANEOp(EVT EltType) {
  if (EltType == MVT::i8)
    return AArch64ISD::DUPLANE8;
  if (EltType == MVT::i16 || EltType == MVT::f16)
    return AArch64ISD::DUPLANE16;
  if (EltType == MVT::i32 || EltType == MVT::f32)
    return AArch64ISD::DUPLANE32;
  if (EltType == MVT::i64 || EltType == MVT::f64)
    return AArch64ISD::DUPLANE64;

  llvm_unreachable("Invalid vector element type?");
}

SDValue AArch64TargetLowering::LowerVECTOR_SHUFFLE(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc dl(Op);
  EVT VT = Op.getValueType();

  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op.getNode());

  // Convert shuffles that are directly supported on NEON to target-specific
  // DAG nodes, instead of keeping them as shuffles and matching them again
  // during code selection.  This is more efficient and avoids the possibility
  // of inconsistencies between legalization and selection.
  ArrayRef<int> ShuffleMask = SVN->getMask();

  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);

  if (ShuffleVectorSDNode::isSplatMask(&ShuffleMask[0],
                                       V1.getValueType().getSimpleVT())) {
    int Lane = SVN->getSplatIndex();
    // If this is undef splat, generate it via "just" vdup, if possible.
    if (Lane == -1)
      Lane = 0;

    if (Lane == 0 && V1.getOpcode() == ISD::SCALAR_TO_VECTOR)
      return DAG.getNode(AArch64ISD::DUP, dl, V1.getValueType(),
                         V1.getOperand(0));
    // Test if V1 is a BUILD_VECTOR and the lane being referenced is a non-
    // constant. If so, we can just reference the lane's definition directly.
    if (V1.getOpcode() == ISD::BUILD_VECTOR &&
        !isa<ConstantSDNode>(V1.getOperand(Lane)))
      return DAG.getNode(AArch64ISD::DUP, dl, VT, V1.getOperand(Lane));

    // Otherwise, duplicate from the lane of the input vector.
    unsigned Opcode = getDUPLANEOp(V1.getValueType().getVectorElementType());

    // SelectionDAGBuilder may have "helpfully" already extracted or conatenated
    // to make a vector of the same size as this SHUFFLE. We can ignore the
    // extract entirely, and canonicalise the concat using WidenVector.
    if (V1.getOpcode() == ISD::EXTRACT_SUBVECTOR) {
      Lane += cast<ConstantSDNode>(V1.getOperand(1))->getZExtValue();
      V1 = V1.getOperand(0);
    } else if (V1.getOpcode() == ISD::CONCAT_VECTORS) {
      unsigned Idx = Lane >= (int)VT.getVectorNumElements() / 2;
      Lane -= Idx * VT.getVectorNumElements() / 2;
      V1 = WidenVector(V1.getOperand(Idx), DAG);
    } else if (VT.getSizeInBits() == 64)
      V1 = WidenVector(V1, DAG);

    return DAG.getNode(Opcode, dl, VT, V1, DAG.getConstant(Lane, dl, MVT::i64));
  }

  if (isREVMask(ShuffleMask, VT, 64))
    return DAG.getNode(AArch64ISD::REV64, dl, V1.getValueType(), V1, V2);
  if (isREVMask(ShuffleMask, VT, 32))
    return DAG.getNode(AArch64ISD::REV32, dl, V1.getValueType(), V1, V2);
  if (isREVMask(ShuffleMask, VT, 16))
    return DAG.getNode(AArch64ISD::REV16, dl, V1.getValueType(), V1, V2);

  bool ReverseEXT = false;
  unsigned Imm;
  if (isEXTMask(ShuffleMask, VT, ReverseEXT, Imm)) {
    if (ReverseEXT)
      std::swap(V1, V2);
    Imm *= getExtFactor(V1);
    return DAG.getNode(AArch64ISD::EXT, dl, V1.getValueType(), V1, V2,
                       DAG.getConstant(Imm, dl, MVT::i32));
  } else if (V2->getOpcode() == ISD::UNDEF &&
             isSingletonEXTMask(ShuffleMask, VT, Imm)) {
    Imm *= getExtFactor(V1);
    return DAG.getNode(AArch64ISD::EXT, dl, V1.getValueType(), V1, V1,
                       DAG.getConstant(Imm, dl, MVT::i32));
  }

  unsigned WhichResult;
  if (isZIPMask(ShuffleMask, VT, WhichResult)) {
    unsigned Opc = (WhichResult == 0) ? AArch64ISD::ZIP1 : AArch64ISD::ZIP2;
    return DAG.getNode(Opc, dl, V1.getValueType(), V1, V2);
  }
  if (isUZPMask(ShuffleMask, VT, WhichResult)) {
    unsigned Opc = (WhichResult == 0) ? AArch64ISD::UZP1 : AArch64ISD::UZP2;
    return DAG.getNode(Opc, dl, V1.getValueType(), V1, V2);
  }
  if (isTRNMask(ShuffleMask, VT, WhichResult)) {
    unsigned Opc = (WhichResult == 0) ? AArch64ISD::TRN1 : AArch64ISD::TRN2;
    return DAG.getNode(Opc, dl, V1.getValueType(), V1, V2);
  }

  if (isZIP_v_undef_Mask(ShuffleMask, VT, WhichResult)) {
    unsigned Opc = (WhichResult == 0) ? AArch64ISD::ZIP1 : AArch64ISD::ZIP2;
    return DAG.getNode(Opc, dl, V1.getValueType(), V1, V1);
  }
  if (isUZP_v_undef_Mask(ShuffleMask, VT, WhichResult)) {
    unsigned Opc = (WhichResult == 0) ? AArch64ISD::UZP1 : AArch64ISD::UZP2;
    return DAG.getNode(Opc, dl, V1.getValueType(), V1, V1);
  }
  if (isTRN_v_undef_Mask(ShuffleMask, VT, WhichResult)) {
    unsigned Opc = (WhichResult == 0) ? AArch64ISD::TRN1 : AArch64ISD::TRN2;
    return DAG.getNode(Opc, dl, V1.getValueType(), V1, V1);
  }

  SDValue Concat = tryFormConcatFromShuffle(Op, DAG);
  if (Concat.getNode())
    return Concat;

  bool DstIsLeft;
  int Anomaly;
  int NumInputElements = V1.getValueType().getVectorNumElements();
  if (isINSMask(ShuffleMask, NumInputElements, DstIsLeft, Anomaly)) {
    SDValue DstVec = DstIsLeft ? V1 : V2;
    SDValue DstLaneV = DAG.getConstant(Anomaly, dl, MVT::i64);

    SDValue SrcVec = V1;
    int SrcLane = ShuffleMask[Anomaly];
    if (SrcLane >= NumInputElements) {
      SrcVec = V2;
      SrcLane -= VT.getVectorNumElements();
    }
    SDValue SrcLaneV = DAG.getConstant(SrcLane, dl, MVT::i64);

    EVT ScalarVT = VT.getVectorElementType();

    if (ScalarVT.getSizeInBits() < 32 && ScalarVT.isInteger())
      ScalarVT = MVT::i32;

    return DAG.getNode(
        ISD::INSERT_VECTOR_ELT, dl, VT, DstVec,
        DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, ScalarVT, SrcVec, SrcLaneV),
        DstLaneV);
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
    unsigned PFTableIndex = PFIndexes[0] * 9 * 9 * 9 + PFIndexes[1] * 9 * 9 +
                            PFIndexes[2] * 9 + PFIndexes[3];
    unsigned PFEntry = PerfectShuffleTable[PFTableIndex];
    unsigned Cost = (PFEntry >> 30);

    if (Cost <= 4)
      return GeneratePerfectShuffle(PFEntry, V1, V2, DAG, dl);
  }

  return GenerateTBL(Op, ShuffleMask, DAG);
}

static bool resolveBuildVector(BuildVectorSDNode *BVN, APInt &CnstBits,
                               APInt &UndefBits) {
  EVT VT = BVN->getValueType(0);
  APInt SplatBits, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;
  if (BVN->isConstantSplat(SplatBits, SplatUndef, SplatBitSize, HasAnyUndefs)) {
    unsigned NumSplats = VT.getSizeInBits() / SplatBitSize;

    for (unsigned i = 0; i < NumSplats; ++i) {
      CnstBits <<= SplatBitSize;
      UndefBits <<= SplatBitSize;
      CnstBits |= SplatBits.zextOrTrunc(VT.getSizeInBits());
      UndefBits |= (SplatBits ^ SplatUndef).zextOrTrunc(VT.getSizeInBits());
    }

    return true;
  }

  return false;
}

SDValue AArch64TargetLowering::LowerVectorAND(SDValue Op,
                                              SelectionDAG &DAG) const {
  BuildVectorSDNode *BVN =
      dyn_cast<BuildVectorSDNode>(Op.getOperand(1).getNode());
  SDValue LHS = Op.getOperand(0);
  SDLoc dl(Op);
  EVT VT = Op.getValueType();

  if (!BVN)
    return Op;

  APInt CnstBits(VT.getSizeInBits(), 0);
  APInt UndefBits(VT.getSizeInBits(), 0);
  if (resolveBuildVector(BVN, CnstBits, UndefBits)) {
    // We only have BIC vector immediate instruction, which is and-not.
    CnstBits = ~CnstBits;

    // We make use of a little bit of goto ickiness in order to avoid having to
    // duplicate the immediate matching logic for the undef toggled case.
    bool SecondTry = false;
  AttemptModImm:

    if (CnstBits.getHiBits(64) == CnstBits.getLoBits(64)) {
      CnstBits = CnstBits.zextOrTrunc(64);
      uint64_t CnstVal = CnstBits.getZExtValue();

      if (AArch64_AM::isAdvSIMDModImmType1(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType1(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::BICi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(0, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType2(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType2(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::BICi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(8, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType3(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType3(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::BICi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(16, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType4(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType4(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::BICi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(24, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType5(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType5(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v8i16 : MVT::v4i16;
        SDValue Mov = DAG.getNode(AArch64ISD::BICi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(0, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType6(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType6(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v8i16 : MVT::v4i16;
        SDValue Mov = DAG.getNode(AArch64ISD::BICi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(8, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }
    }

    if (SecondTry)
      goto FailedModImm;
    SecondTry = true;
    CnstBits = ~UndefBits;
    goto AttemptModImm;
  }

// We can always fall back to a non-immediate AND.
FailedModImm:
  return Op;
}

// Specialized code to quickly find if PotentialBVec is a BuildVector that
// consists of only the same constant int value, returned in reference arg
// ConstVal
static bool isAllConstantBuildVector(const SDValue &PotentialBVec,
                                     uint64_t &ConstVal) {
  BuildVectorSDNode *Bvec = dyn_cast<BuildVectorSDNode>(PotentialBVec);
  if (!Bvec)
    return false;
  ConstantSDNode *FirstElt = dyn_cast<ConstantSDNode>(Bvec->getOperand(0));
  if (!FirstElt)
    return false;
  EVT VT = Bvec->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();
  for (unsigned i = 1; i < NumElts; ++i)
    if (dyn_cast<ConstantSDNode>(Bvec->getOperand(i)) != FirstElt)
      return false;
  ConstVal = FirstElt->getZExtValue();
  return true;
}

static unsigned getIntrinsicID(const SDNode *N) {
  unsigned Opcode = N->getOpcode();
  switch (Opcode) {
  default:
    return Intrinsic::not_intrinsic;
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    if (IID < Intrinsic::num_intrinsics)
      return IID;
    return Intrinsic::not_intrinsic;
  }
  }
}

// Attempt to form a vector S[LR]I from (or (and X, BvecC1), (lsl Y, C2)),
// to (SLI X, Y, C2), where X and Y have matching vector types, BvecC1 is a
// BUILD_VECTORs with constant element C1, C2 is a constant, and C1 == ~C2.
// Also, logical shift right -> sri, with the same structure.
static SDValue tryLowerToSLI(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);

  if (!VT.isVector())
    return SDValue();

  SDLoc DL(N);

  // Is the first op an AND?
  const SDValue And = N->getOperand(0);
  if (And.getOpcode() != ISD::AND)
    return SDValue();

  // Is the second op an shl or lshr?
  SDValue Shift = N->getOperand(1);
  // This will have been turned into: AArch64ISD::VSHL vector, #shift
  // or AArch64ISD::VLSHR vector, #shift
  unsigned ShiftOpc = Shift.getOpcode();
  if ((ShiftOpc != AArch64ISD::VSHL && ShiftOpc != AArch64ISD::VLSHR))
    return SDValue();
  bool IsShiftRight = ShiftOpc == AArch64ISD::VLSHR;

  // Is the shift amount constant?
  ConstantSDNode *C2node = dyn_cast<ConstantSDNode>(Shift.getOperand(1));
  if (!C2node)
    return SDValue();

  // Is the and mask vector all constant?
  uint64_t C1;
  if (!isAllConstantBuildVector(And.getOperand(1), C1))
    return SDValue();

  // Is C1 == ~C2, taking into account how much one can shift elements of a
  // particular size?
  uint64_t C2 = C2node->getZExtValue();
  unsigned ElemSizeInBits = VT.getVectorElementType().getSizeInBits();
  if (C2 > ElemSizeInBits)
    return SDValue();
  unsigned ElemMask = (1 << ElemSizeInBits) - 1;
  if ((C1 & ElemMask) != (~C2 & ElemMask))
    return SDValue();

  SDValue X = And.getOperand(0);
  SDValue Y = Shift.getOperand(0);

  unsigned Intrin =
      IsShiftRight ? Intrinsic::aarch64_neon_vsri : Intrinsic::aarch64_neon_vsli;
  SDValue ResultSLI =
      DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                  DAG.getConstant(Intrin, DL, MVT::i32), X, Y,
                  Shift.getOperand(1));

  DEBUG(dbgs() << "aarch64-lower: transformed: \n");
  DEBUG(N->dump(&DAG));
  DEBUG(dbgs() << "into: \n");
  DEBUG(ResultSLI->dump(&DAG));

  ++NumShiftInserts;
  return ResultSLI;
}

SDValue AArch64TargetLowering::LowerVectorOR(SDValue Op,
                                             SelectionDAG &DAG) const {
  // Attempt to form a vector S[LR]I from (or (and X, C1), (lsl Y, C2))
  if (EnableAArch64SlrGeneration) {
    SDValue Res = tryLowerToSLI(Op.getNode(), DAG);
    if (Res.getNode())
      return Res;
  }

  BuildVectorSDNode *BVN =
      dyn_cast<BuildVectorSDNode>(Op.getOperand(0).getNode());
  SDValue LHS = Op.getOperand(1);
  SDLoc dl(Op);
  EVT VT = Op.getValueType();

  // OR commutes, so try swapping the operands.
  if (!BVN) {
    LHS = Op.getOperand(0);
    BVN = dyn_cast<BuildVectorSDNode>(Op.getOperand(1).getNode());
  }
  if (!BVN)
    return Op;

  APInt CnstBits(VT.getSizeInBits(), 0);
  APInt UndefBits(VT.getSizeInBits(), 0);
  if (resolveBuildVector(BVN, CnstBits, UndefBits)) {
    // We make use of a little bit of goto ickiness in order to avoid having to
    // duplicate the immediate matching logic for the undef toggled case.
    bool SecondTry = false;
  AttemptModImm:

    if (CnstBits.getHiBits(64) == CnstBits.getLoBits(64)) {
      CnstBits = CnstBits.zextOrTrunc(64);
      uint64_t CnstVal = CnstBits.getZExtValue();

      if (AArch64_AM::isAdvSIMDModImmType1(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType1(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::ORRi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(0, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType2(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType2(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::ORRi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(8, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType3(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType3(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::ORRi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(16, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType4(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType4(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::ORRi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(24, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType5(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType5(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v8i16 : MVT::v4i16;
        SDValue Mov = DAG.getNode(AArch64ISD::ORRi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(0, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType6(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType6(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v8i16 : MVT::v4i16;
        SDValue Mov = DAG.getNode(AArch64ISD::ORRi, dl, MovTy, LHS,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(8, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }
    }

    if (SecondTry)
      goto FailedModImm;
    SecondTry = true;
    CnstBits = UndefBits;
    goto AttemptModImm;
  }

// We can always fall back to a non-immediate OR.
FailedModImm:
  return Op;
}

// Normalize the operands of BUILD_VECTOR. The value of constant operands will
// be truncated to fit element width.
static SDValue NormalizeBuildVector(SDValue Op,
                                    SelectionDAG &DAG) {
  assert(Op.getOpcode() == ISD::BUILD_VECTOR && "Unknown opcode!");
  SDLoc dl(Op);
  EVT VT = Op.getValueType();
  EVT EltTy= VT.getVectorElementType();

  if (EltTy.isFloatingPoint() || EltTy.getSizeInBits() > 16)
    return Op;

  SmallVector<SDValue, 16> Ops;
  for (SDValue Lane : Op->ops()) {
    if (auto *CstLane = dyn_cast<ConstantSDNode>(Lane)) {
      APInt LowBits(EltTy.getSizeInBits(),
                    CstLane->getZExtValue());
      Lane = DAG.getConstant(LowBits.getZExtValue(), dl, MVT::i32);
    }
    Ops.push_back(Lane);
  }
  return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, Ops);
}

SDValue AArch64TargetLowering::LowerBUILD_VECTOR(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc dl(Op);
  EVT VT = Op.getValueType();
  Op = NormalizeBuildVector(Op, DAG);
  BuildVectorSDNode *BVN = cast<BuildVectorSDNode>(Op.getNode());

  APInt CnstBits(VT.getSizeInBits(), 0);
  APInt UndefBits(VT.getSizeInBits(), 0);
  if (resolveBuildVector(BVN, CnstBits, UndefBits)) {
    // We make use of a little bit of goto ickiness in order to avoid having to
    // duplicate the immediate matching logic for the undef toggled case.
    bool SecondTry = false;
  AttemptModImm:

    if (CnstBits.getHiBits(64) == CnstBits.getLoBits(64)) {
      CnstBits = CnstBits.zextOrTrunc(64);
      uint64_t CnstVal = CnstBits.getZExtValue();

      // Certain magic vector constants (used to express things like NOT
      // and NEG) are passed through unmodified.  This allows codegen patterns
      // for these operations to match.  Special-purpose patterns will lower
      // these immediates to MOVIs if it proves necessary.
      if (VT.isInteger() && (CnstVal == 0 || CnstVal == ~0ULL))
        return Op;

      // The many faces of MOVI...
      if (AArch64_AM::isAdvSIMDModImmType10(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType10(CnstVal);
        if (VT.getSizeInBits() == 128) {
          SDValue Mov = DAG.getNode(AArch64ISD::MOVIedit, dl, MVT::v2i64,
                                    DAG.getConstant(CnstVal, dl, MVT::i32));
          return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
        }

        // Support the V64 version via subregister insertion.
        SDValue Mov = DAG.getNode(AArch64ISD::MOVIedit, dl, MVT::f64,
                                  DAG.getConstant(CnstVal, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType1(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType1(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(0, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType2(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType2(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(8, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType3(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType3(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(16, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType4(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType4(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(24, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType5(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType5(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v8i16 : MVT::v4i16;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(0, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType6(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType6(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v8i16 : MVT::v4i16;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(8, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType7(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType7(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVImsl, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(264, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType8(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType8(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVImsl, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(272, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType9(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType9(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v16i8 : MVT::v8i8;
        SDValue Mov = DAG.getNode(AArch64ISD::MOVI, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      // The few faces of FMOV...
      if (AArch64_AM::isAdvSIMDModImmType11(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType11(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4f32 : MVT::v2f32;
        SDValue Mov = DAG.getNode(AArch64ISD::FMOV, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType12(CnstVal) &&
          VT.getSizeInBits() == 128) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType12(CnstVal);
        SDValue Mov = DAG.getNode(AArch64ISD::FMOV, dl, MVT::v2f64,
                                  DAG.getConstant(CnstVal, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      // The many faces of MVNI...
      CnstVal = ~CnstVal;
      if (AArch64_AM::isAdvSIMDModImmType1(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType1(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MVNIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(0, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType2(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType2(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MVNIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(8, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType3(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType3(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MVNIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(16, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType4(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType4(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MVNIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(24, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType5(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType5(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v8i16 : MVT::v4i16;
        SDValue Mov = DAG.getNode(AArch64ISD::MVNIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(0, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType6(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType6(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v8i16 : MVT::v4i16;
        SDValue Mov = DAG.getNode(AArch64ISD::MVNIshift, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(8, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType7(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType7(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MVNImsl, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(264, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }

      if (AArch64_AM::isAdvSIMDModImmType8(CnstVal)) {
        CnstVal = AArch64_AM::encodeAdvSIMDModImmType8(CnstVal);
        MVT MovTy = (VT.getSizeInBits() == 128) ? MVT::v4i32 : MVT::v2i32;
        SDValue Mov = DAG.getNode(AArch64ISD::MVNImsl, dl, MovTy,
                                  DAG.getConstant(CnstVal, dl, MVT::i32),
                                  DAG.getConstant(272, dl, MVT::i32));
        return DAG.getNode(AArch64ISD::NVCAST, dl, VT, Mov);
      }
    }

    if (SecondTry)
      goto FailedModImm;
    SecondTry = true;
    CnstBits = UndefBits;
    goto AttemptModImm;
  }
FailedModImm:

  // Scan through the operands to find some interesting properties we can
  // exploit:
  //   1) If only one value is used, we can use a DUP, or
  //   2) if only the low element is not undef, we can just insert that, or
  //   3) if only one constant value is used (w/ some non-constant lanes),
  //      we can splat the constant value into the whole vector then fill
  //      in the non-constant lanes.
  //   4) FIXME: If different constant values are used, but we can intelligently
  //             select the values we'll be overwriting for the non-constant
  //             lanes such that we can directly materialize the vector
  //             some other way (MOVI, e.g.), we can be sneaky.
  unsigned NumElts = VT.getVectorNumElements();
  bool isOnlyLowElement = true;
  bool usesOnlyOneValue = true;
  bool usesOnlyOneConstantValue = true;
  bool isConstant = true;
  unsigned NumConstantLanes = 0;
  SDValue Value;
  SDValue ConstantValue;
  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue V = Op.getOperand(i);
    if (V.getOpcode() == ISD::UNDEF)
      continue;
    if (i > 0)
      isOnlyLowElement = false;
    if (!isa<ConstantFPSDNode>(V) && !isa<ConstantSDNode>(V))
      isConstant = false;

    if (isa<ConstantSDNode>(V) || isa<ConstantFPSDNode>(V)) {
      ++NumConstantLanes;
      if (!ConstantValue.getNode())
        ConstantValue = V;
      else if (ConstantValue != V)
        usesOnlyOneConstantValue = false;
    }

    if (!Value.getNode())
      Value = V;
    else if (V != Value)
      usesOnlyOneValue = false;
  }

  if (!Value.getNode())
    return DAG.getUNDEF(VT);

  if (isOnlyLowElement)
    return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Value);

  // Use DUP for non-constant splats.  For f32 constant splats, reduce to
  // i32 and try again.
  if (usesOnlyOneValue) {
    if (!isConstant) {
      if (Value.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
          Value.getValueType() != VT)
        return DAG.getNode(AArch64ISD::DUP, dl, VT, Value);

      // This is actually a DUPLANExx operation, which keeps everything vectory.

      // DUPLANE works on 128-bit vectors, widen it if necessary.
      SDValue Lane = Value.getOperand(1);
      Value = Value.getOperand(0);
      if (Value.getValueType().getSizeInBits() == 64)
        Value = WidenVector(Value, DAG);

      unsigned Opcode = getDUPLANEOp(VT.getVectorElementType());
      return DAG.getNode(Opcode, dl, VT, Value, Lane);
    }

    if (VT.getVectorElementType().isFloatingPoint()) {
      SmallVector<SDValue, 8> Ops;
      EVT EltTy = VT.getVectorElementType();
      assert ((EltTy == MVT::f16 || EltTy == MVT::f32 || EltTy == MVT::f64) &&
              "Unsupported floating-point vector type");
      MVT NewType = MVT::getIntegerVT(EltTy.getSizeInBits());
      for (unsigned i = 0; i < NumElts; ++i)
        Ops.push_back(DAG.getNode(ISD::BITCAST, dl, NewType, Op.getOperand(i)));
      EVT VecVT = EVT::getVectorVT(*DAG.getContext(), NewType, NumElts);
      SDValue Val = DAG.getNode(ISD::BUILD_VECTOR, dl, VecVT, Ops);
      Val = LowerBUILD_VECTOR(Val, DAG);
      if (Val.getNode())
        return DAG.getNode(ISD::BITCAST, dl, VT, Val);
    }
  }

  // If there was only one constant value used and for more than one lane,
  // start by splatting that value, then replace the non-constant lanes. This
  // is better than the default, which will perform a separate initialization
  // for each lane.
  if (NumConstantLanes > 0 && usesOnlyOneConstantValue) {
    SDValue Val = DAG.getNode(AArch64ISD::DUP, dl, VT, ConstantValue);
    // Now insert the non-constant lanes.
    for (unsigned i = 0; i < NumElts; ++i) {
      SDValue V = Op.getOperand(i);
      SDValue LaneIdx = DAG.getConstant(i, dl, MVT::i64);
      if (!isa<ConstantSDNode>(V) && !isa<ConstantFPSDNode>(V)) {
        // Note that type legalization likely mucked about with the VT of the
        // source operand, so we may have to convert it here before inserting.
        Val = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, Val, V, LaneIdx);
      }
    }
    return Val;
  }

  // If all elements are constants and the case above didn't get hit, fall back
  // to the default expansion, which will generate a load from the constant
  // pool.
  if (isConstant)
    return SDValue();

  // Empirical tests suggest this is rarely worth it for vectors of length <= 2.
  if (NumElts >= 4) {
    if (SDValue shuffle = ReconstructShuffle(Op, DAG))
      return shuffle;
  }

  // If all else fails, just use a sequence of INSERT_VECTOR_ELT when we
  // know the default expansion would otherwise fall back on something even
  // worse. For a vector with one or two non-undef values, that's
  // scalar_to_vector for the elements followed by a shuffle (provided the
  // shuffle is valid for the target) and materialization element by element
  // on the stack followed by a load for everything else.
  if (!isConstant && !usesOnlyOneValue) {
    SDValue Vec = DAG.getUNDEF(VT);
    SDValue Op0 = Op.getOperand(0);
    unsigned ElemSize = VT.getVectorElementType().getSizeInBits();
    unsigned i = 0;
    // For 32 and 64 bit types, use INSERT_SUBREG for lane zero to
    // a) Avoid a RMW dependency on the full vector register, and
    // b) Allow the register coalescer to fold away the copy if the
    //    value is already in an S or D register.
    // Do not do this for UNDEF/LOAD nodes because we have better patterns
    // for those avoiding the SCALAR_TO_VECTOR/BUILD_VECTOR.
    if (Op0.getOpcode() != ISD::UNDEF && Op0.getOpcode() != ISD::LOAD &&
        (ElemSize == 32 || ElemSize == 64)) {
      unsigned SubIdx = ElemSize == 32 ? AArch64::ssub : AArch64::dsub;
      MachineSDNode *N =
          DAG.getMachineNode(TargetOpcode::INSERT_SUBREG, dl, VT, Vec, Op0,
                             DAG.getTargetConstant(SubIdx, dl, MVT::i32));
      Vec = SDValue(N, 0);
      ++i;
    }
    for (; i < NumElts; ++i) {
      SDValue V = Op.getOperand(i);
      if (V.getOpcode() == ISD::UNDEF)
        continue;
      SDValue LaneIdx = DAG.getConstant(i, dl, MVT::i64);
      Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, Vec, V, LaneIdx);
    }
    return Vec;
  }

  // Just use the default expansion. We failed to find a better alternative.
  return SDValue();
}

SDValue AArch64TargetLowering::LowerINSERT_VECTOR_ELT(SDValue Op,
                                                      SelectionDAG &DAG) const {
  assert(Op.getOpcode() == ISD::INSERT_VECTOR_ELT && "Unknown opcode!");

  // Check for non-constant or out of range lane.
  EVT VT = Op.getOperand(0).getValueType();
  ConstantSDNode *CI = dyn_cast<ConstantSDNode>(Op.getOperand(2));
  if (!CI || CI->getZExtValue() >= VT.getVectorNumElements())
    return SDValue();


  // Insertion/extraction are legal for V128 types.
  if (VT == MVT::v16i8 || VT == MVT::v8i16 || VT == MVT::v4i32 ||
      VT == MVT::v2i64 || VT == MVT::v4f32 || VT == MVT::v2f64 ||
      VT == MVT::v8f16)
    return Op;

  if (VT != MVT::v8i8 && VT != MVT::v4i16 && VT != MVT::v2i32 &&
      VT != MVT::v1i64 && VT != MVT::v2f32 && VT != MVT::v4f16)
    return SDValue();

  // For V64 types, we perform insertion by expanding the value
  // to a V128 type and perform the insertion on that.
  SDLoc DL(Op);
  SDValue WideVec = WidenVector(Op.getOperand(0), DAG);
  EVT WideTy = WideVec.getValueType();

  SDValue Node = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, WideTy, WideVec,
                             Op.getOperand(1), Op.getOperand(2));
  // Re-narrow the resultant vector.
  return NarrowVector(Node, DAG);
}

SDValue
AArch64TargetLowering::LowerEXTRACT_VECTOR_ELT(SDValue Op,
                                               SelectionDAG &DAG) const {
  assert(Op.getOpcode() == ISD::EXTRACT_VECTOR_ELT && "Unknown opcode!");

  // Check for non-constant or out of range lane.
  EVT VT = Op.getOperand(0).getValueType();
  ConstantSDNode *CI = dyn_cast<ConstantSDNode>(Op.getOperand(1));
  if (!CI || CI->getZExtValue() >= VT.getVectorNumElements())
    return SDValue();


  // Insertion/extraction are legal for V128 types.
  if (VT == MVT::v16i8 || VT == MVT::v8i16 || VT == MVT::v4i32 ||
      VT == MVT::v2i64 || VT == MVT::v4f32 || VT == MVT::v2f64 ||
      VT == MVT::v8f16)
    return Op;

  if (VT != MVT::v8i8 && VT != MVT::v4i16 && VT != MVT::v2i32 &&
      VT != MVT::v1i64 && VT != MVT::v2f32 && VT != MVT::v4f16)
    return SDValue();

  // For V64 types, we perform extraction by expanding the value
  // to a V128 type and perform the extraction on that.
  SDLoc DL(Op);
  SDValue WideVec = WidenVector(Op.getOperand(0), DAG);
  EVT WideTy = WideVec.getValueType();

  EVT ExtrTy = WideTy.getVectorElementType();
  if (ExtrTy == MVT::i16 || ExtrTy == MVT::i8)
    ExtrTy = MVT::i32;

  // For extractions, we just return the result directly.
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, ExtrTy, WideVec,
                     Op.getOperand(1));
}

SDValue AArch64TargetLowering::LowerEXTRACT_SUBVECTOR(SDValue Op,
                                                      SelectionDAG &DAG) const {
  EVT VT = Op.getOperand(0).getValueType();
  SDLoc dl(Op);
  // Just in case...
  if (!VT.isVector())
    return SDValue();

  ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(Op.getOperand(1));
  if (!Cst)
    return SDValue();
  unsigned Val = Cst->getZExtValue();

  unsigned Size = Op.getValueType().getSizeInBits();

  // This will get lowered to an appropriate EXTRACT_SUBREG in ISel.
  if (Val == 0)
    return Op;

  // If this is extracting the upper 64-bits of a 128-bit vector, we match
  // that directly.
  if (Size == 64 && Val * VT.getVectorElementType().getSizeInBits() == 64)
    return Op;

  return SDValue();
}

bool AArch64TargetLowering::isShuffleMaskLegal(const SmallVectorImpl<int> &M,
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
    unsigned PFTableIndex = PFIndexes[0] * 9 * 9 * 9 + PFIndexes[1] * 9 * 9 +
                            PFIndexes[2] * 9 + PFIndexes[3];
    unsigned PFEntry = PerfectShuffleTable[PFTableIndex];
    unsigned Cost = (PFEntry >> 30);

    if (Cost <= 4)
      return true;
  }

  bool DummyBool;
  int DummyInt;
  unsigned DummyUnsigned;

  return (ShuffleVectorSDNode::isSplatMask(&M[0], VT) || isREVMask(M, VT, 64) ||
          isREVMask(M, VT, 32) || isREVMask(M, VT, 16) ||
          isEXTMask(M, VT, DummyBool, DummyUnsigned) ||
          // isTBLMask(M, VT) || // FIXME: Port TBL support from ARM.
          isTRNMask(M, VT, DummyUnsigned) || isUZPMask(M, VT, DummyUnsigned) ||
          isZIPMask(M, VT, DummyUnsigned) ||
          isTRN_v_undef_Mask(M, VT, DummyUnsigned) ||
          isUZP_v_undef_Mask(M, VT, DummyUnsigned) ||
          isZIP_v_undef_Mask(M, VT, DummyUnsigned) ||
          isINSMask(M, VT.getVectorNumElements(), DummyBool, DummyInt) ||
          isConcatMask(M, VT, VT.getSizeInBits() == 128));
}

/// getVShiftImm - Check if this is a valid build_vector for the immediate
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
  if (!BVN || !BVN->isConstantSplat(SplatBits, SplatUndef, SplatBitSize,
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
  int64_t ElementBits = VT.getVectorElementType().getSizeInBits();
  if (!getVShiftImm(Op, ElementBits, Cnt))
    return false;
  return (Cnt >= 0 && (isLong ? Cnt - 1 : Cnt) < ElementBits);
}

/// isVShiftRImm - Check if this is a valid build_vector for the immediate
/// operand of a vector shift right operation. The value must be in the range:
///   1 <= Value <= ElementBits for a right shift; or
static bool isVShiftRImm(SDValue Op, EVT VT, bool isNarrow, int64_t &Cnt) {
  assert(VT.isVector() && "vector shift count is not a vector type");
  int64_t ElementBits = VT.getVectorElementType().getSizeInBits();
  if (!getVShiftImm(Op, ElementBits, Cnt))
    return false;
  return (Cnt >= 1 && Cnt <= (isNarrow ? ElementBits / 2 : ElementBits));
}

SDValue AArch64TargetLowering::LowerVectorSRA_SRL_SHL(SDValue Op,
                                                      SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  int64_t Cnt;

  if (!Op.getOperand(1).getValueType().isVector())
    return Op;
  unsigned EltSize = VT.getVectorElementType().getSizeInBits();

  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("unexpected shift opcode");

  case ISD::SHL:
    if (isVShiftLImm(Op.getOperand(1), VT, false, Cnt) && Cnt < EltSize)
      return DAG.getNode(AArch64ISD::VSHL, DL, VT, Op.getOperand(0),
                         DAG.getConstant(Cnt, DL, MVT::i32));
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                       DAG.getConstant(Intrinsic::aarch64_neon_ushl, DL,
                                       MVT::i32),
                       Op.getOperand(0), Op.getOperand(1));
  case ISD::SRA:
  case ISD::SRL:
    // Right shift immediate
    if (isVShiftRImm(Op.getOperand(1), VT, false, Cnt) && Cnt < EltSize) {
      unsigned Opc =
          (Op.getOpcode() == ISD::SRA) ? AArch64ISD::VASHR : AArch64ISD::VLSHR;
      return DAG.getNode(Opc, DL, VT, Op.getOperand(0),
                         DAG.getConstant(Cnt, DL, MVT::i32));
    }

    // Right shift register.  Note, there is not a shift right register
    // instruction, but the shift left register instruction takes a signed
    // value, where negative numbers specify a right shift.
    unsigned Opc = (Op.getOpcode() == ISD::SRA) ? Intrinsic::aarch64_neon_sshl
                                                : Intrinsic::aarch64_neon_ushl;
    // negate the shift amount
    SDValue NegShift = DAG.getNode(AArch64ISD::NEG, DL, VT, Op.getOperand(1));
    SDValue NegShiftLeft =
        DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                    DAG.getConstant(Opc, DL, MVT::i32), Op.getOperand(0),
                    NegShift);
    return NegShiftLeft;
  }

  return SDValue();
}

static SDValue EmitVectorComparison(SDValue LHS, SDValue RHS,
                                    AArch64CC::CondCode CC, bool NoNans, EVT VT,
                                    SDLoc dl, SelectionDAG &DAG) {
  EVT SrcVT = LHS.getValueType();
  assert(VT.getSizeInBits() == SrcVT.getSizeInBits() &&
         "function only supposed to emit natural comparisons");

  BuildVectorSDNode *BVN = dyn_cast<BuildVectorSDNode>(RHS.getNode());
  APInt CnstBits(VT.getSizeInBits(), 0);
  APInt UndefBits(VT.getSizeInBits(), 0);
  bool IsCnst = BVN && resolveBuildVector(BVN, CnstBits, UndefBits);
  bool IsZero = IsCnst && (CnstBits == 0);

  if (SrcVT.getVectorElementType().isFloatingPoint()) {
    switch (CC) {
    default:
      return SDValue();
    case AArch64CC::NE: {
      SDValue Fcmeq;
      if (IsZero)
        Fcmeq = DAG.getNode(AArch64ISD::FCMEQz, dl, VT, LHS);
      else
        Fcmeq = DAG.getNode(AArch64ISD::FCMEQ, dl, VT, LHS, RHS);
      return DAG.getNode(AArch64ISD::NOT, dl, VT, Fcmeq);
    }
    case AArch64CC::EQ:
      if (IsZero)
        return DAG.getNode(AArch64ISD::FCMEQz, dl, VT, LHS);
      return DAG.getNode(AArch64ISD::FCMEQ, dl, VT, LHS, RHS);
    case AArch64CC::GE:
      if (IsZero)
        return DAG.getNode(AArch64ISD::FCMGEz, dl, VT, LHS);
      return DAG.getNode(AArch64ISD::FCMGE, dl, VT, LHS, RHS);
    case AArch64CC::GT:
      if (IsZero)
        return DAG.getNode(AArch64ISD::FCMGTz, dl, VT, LHS);
      return DAG.getNode(AArch64ISD::FCMGT, dl, VT, LHS, RHS);
    case AArch64CC::LS:
      if (IsZero)
        return DAG.getNode(AArch64ISD::FCMLEz, dl, VT, LHS);
      return DAG.getNode(AArch64ISD::FCMGE, dl, VT, RHS, LHS);
    case AArch64CC::LT:
      if (!NoNans)
        return SDValue();
    // If we ignore NaNs then we can use to the MI implementation.
    // Fallthrough.
    case AArch64CC::MI:
      if (IsZero)
        return DAG.getNode(AArch64ISD::FCMLTz, dl, VT, LHS);
      return DAG.getNode(AArch64ISD::FCMGT, dl, VT, RHS, LHS);
    }
  }

  switch (CC) {
  default:
    return SDValue();
  case AArch64CC::NE: {
    SDValue Cmeq;
    if (IsZero)
      Cmeq = DAG.getNode(AArch64ISD::CMEQz, dl, VT, LHS);
    else
      Cmeq = DAG.getNode(AArch64ISD::CMEQ, dl, VT, LHS, RHS);
    return DAG.getNode(AArch64ISD::NOT, dl, VT, Cmeq);
  }
  case AArch64CC::EQ:
    if (IsZero)
      return DAG.getNode(AArch64ISD::CMEQz, dl, VT, LHS);
    return DAG.getNode(AArch64ISD::CMEQ, dl, VT, LHS, RHS);
  case AArch64CC::GE:
    if (IsZero)
      return DAG.getNode(AArch64ISD::CMGEz, dl, VT, LHS);
    return DAG.getNode(AArch64ISD::CMGE, dl, VT, LHS, RHS);
  case AArch64CC::GT:
    if (IsZero)
      return DAG.getNode(AArch64ISD::CMGTz, dl, VT, LHS);
    return DAG.getNode(AArch64ISD::CMGT, dl, VT, LHS, RHS);
  case AArch64CC::LE:
    if (IsZero)
      return DAG.getNode(AArch64ISD::CMLEz, dl, VT, LHS);
    return DAG.getNode(AArch64ISD::CMGE, dl, VT, RHS, LHS);
  case AArch64CC::LS:
    return DAG.getNode(AArch64ISD::CMHS, dl, VT, RHS, LHS);
  case AArch64CC::LO:
    return DAG.getNode(AArch64ISD::CMHI, dl, VT, RHS, LHS);
  case AArch64CC::LT:
    if (IsZero)
      return DAG.getNode(AArch64ISD::CMLTz, dl, VT, LHS);
    return DAG.getNode(AArch64ISD::CMGT, dl, VT, RHS, LHS);
  case AArch64CC::HI:
    return DAG.getNode(AArch64ISD::CMHI, dl, VT, LHS, RHS);
  case AArch64CC::HS:
    return DAG.getNode(AArch64ISD::CMHS, dl, VT, LHS, RHS);
  }
}

SDValue AArch64TargetLowering::LowerVSETCC(SDValue Op,
                                           SelectionDAG &DAG) const {
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  EVT CmpVT = LHS.getValueType().changeVectorElementTypeToInteger();
  SDLoc dl(Op);

  if (LHS.getValueType().getVectorElementType().isInteger()) {
    assert(LHS.getValueType() == RHS.getValueType());
    AArch64CC::CondCode AArch64CC = changeIntCCToAArch64CC(CC);
    SDValue Cmp =
        EmitVectorComparison(LHS, RHS, AArch64CC, false, CmpVT, dl, DAG);
    return DAG.getSExtOrTrunc(Cmp, dl, Op.getValueType());
  }

  assert(LHS.getValueType().getVectorElementType() == MVT::f32 ||
         LHS.getValueType().getVectorElementType() == MVT::f64);

  // Unfortunately, the mapping of LLVM FP CC's onto AArch64 CC's isn't totally
  // clean.  Some of them require two branches to implement.
  AArch64CC::CondCode CC1, CC2;
  bool ShouldInvert;
  changeVectorFPCCToAArch64CC(CC, CC1, CC2, ShouldInvert);

  bool NoNaNs = getTargetMachine().Options.NoNaNsFPMath;
  SDValue Cmp =
      EmitVectorComparison(LHS, RHS, CC1, NoNaNs, CmpVT, dl, DAG);
  if (!Cmp.getNode())
    return SDValue();

  if (CC2 != AArch64CC::AL) {
    SDValue Cmp2 =
        EmitVectorComparison(LHS, RHS, CC2, NoNaNs, CmpVT, dl, DAG);
    if (!Cmp2.getNode())
      return SDValue();

    Cmp = DAG.getNode(ISD::OR, dl, CmpVT, Cmp, Cmp2);
  }

  Cmp = DAG.getSExtOrTrunc(Cmp, dl, Op.getValueType());

  if (ShouldInvert)
    return Cmp = DAG.getNOT(dl, Cmp, Cmp.getValueType());

  return Cmp;
}

/// getTgtMemIntrinsic - Represent NEON load and store intrinsics as
/// MemIntrinsicNodes.  The associated MachineMemOperands record the alignment
/// specified in the intrinsic calls.
bool AArch64TargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                               const CallInst &I,
                                               unsigned Intrinsic) const {
  auto &DL = I.getModule()->getDataLayout();
  switch (Intrinsic) {
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_ld4:
  case Intrinsic::aarch64_neon_ld1x2:
  case Intrinsic::aarch64_neon_ld1x3:
  case Intrinsic::aarch64_neon_ld1x4:
  case Intrinsic::aarch64_neon_ld2lane:
  case Intrinsic::aarch64_neon_ld3lane:
  case Intrinsic::aarch64_neon_ld4lane:
  case Intrinsic::aarch64_neon_ld2r:
  case Intrinsic::aarch64_neon_ld3r:
  case Intrinsic::aarch64_neon_ld4r: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    // Conservatively set memVT to the entire set of vectors loaded.
    uint64_t NumElts = DL.getTypeAllocSize(I.getType()) / 8;
    Info.memVT = EVT::getVectorVT(I.getType()->getContext(), MVT::i64, NumElts);
    Info.ptrVal = I.getArgOperand(I.getNumArgOperands() - 1);
    Info.offset = 0;
    Info.align = 0;
    Info.vol = false; // volatile loads with NEON intrinsics not supported
    Info.readMem = true;
    Info.writeMem = false;
    return true;
  }
  case Intrinsic::aarch64_neon_st2:
  case Intrinsic::aarch64_neon_st3:
  case Intrinsic::aarch64_neon_st4:
  case Intrinsic::aarch64_neon_st1x2:
  case Intrinsic::aarch64_neon_st1x3:
  case Intrinsic::aarch64_neon_st1x4:
  case Intrinsic::aarch64_neon_st2lane:
  case Intrinsic::aarch64_neon_st3lane:
  case Intrinsic::aarch64_neon_st4lane: {
    Info.opc = ISD::INTRINSIC_VOID;
    // Conservatively set memVT to the entire set of vectors stored.
    unsigned NumElts = 0;
    for (unsigned ArgI = 1, ArgE = I.getNumArgOperands(); ArgI < ArgE; ++ArgI) {
      Type *ArgTy = I.getArgOperand(ArgI)->getType();
      if (!ArgTy->isVectorTy())
        break;
      NumElts += DL.getTypeAllocSize(ArgTy) / 8;
    }
    Info.memVT = EVT::getVectorVT(I.getType()->getContext(), MVT::i64, NumElts);
    Info.ptrVal = I.getArgOperand(I.getNumArgOperands() - 1);
    Info.offset = 0;
    Info.align = 0;
    Info.vol = false; // volatile stores with NEON intrinsics not supported
    Info.readMem = false;
    Info.writeMem = true;
    return true;
  }
  case Intrinsic::aarch64_ldaxr:
  case Intrinsic::aarch64_ldxr: {
    PointerType *PtrTy = cast<PointerType>(I.getArgOperand(0)->getType());
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = DL.getABITypeAlignment(PtrTy->getElementType());
    Info.vol = true;
    Info.readMem = true;
    Info.writeMem = false;
    return true;
  }
  case Intrinsic::aarch64_stlxr:
  case Intrinsic::aarch64_stxr: {
    PointerType *PtrTy = cast<PointerType>(I.getArgOperand(1)->getType());
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = I.getArgOperand(1);
    Info.offset = 0;
    Info.align = DL.getABITypeAlignment(PtrTy->getElementType());
    Info.vol = true;
    Info.readMem = false;
    Info.writeMem = true;
    return true;
  }
  case Intrinsic::aarch64_ldaxp:
  case Intrinsic::aarch64_ldxp: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i128;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = 16;
    Info.vol = true;
    Info.readMem = true;
    Info.writeMem = false;
    return true;
  }
  case Intrinsic::aarch64_stlxp:
  case Intrinsic::aarch64_stxp: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i128;
    Info.ptrVal = I.getArgOperand(2);
    Info.offset = 0;
    Info.align = 16;
    Info.vol = true;
    Info.readMem = false;
    Info.writeMem = true;
    return true;
  }
  default:
    break;
  }

  return false;
}

// Truncations from 64-bit GPR to 32-bit GPR is free.
bool AArch64TargetLowering::isTruncateFree(Type *Ty1, Type *Ty2) const {
  if (!Ty1->isIntegerTy() || !Ty2->isIntegerTy())
    return false;
  unsigned NumBits1 = Ty1->getPrimitiveSizeInBits();
  unsigned NumBits2 = Ty2->getPrimitiveSizeInBits();
  return NumBits1 > NumBits2;
}
bool AArch64TargetLowering::isTruncateFree(EVT VT1, EVT VT2) const {
  if (VT1.isVector() || VT2.isVector() || !VT1.isInteger() || !VT2.isInteger())
    return false;
  unsigned NumBits1 = VT1.getSizeInBits();
  unsigned NumBits2 = VT2.getSizeInBits();
  return NumBits1 > NumBits2;
}

/// Check if it is profitable to hoist instruction in then/else to if.
/// Not profitable if I and it's user can form a FMA instruction
/// because we prefer FMSUB/FMADD.
bool AArch64TargetLowering::isProfitableToHoist(Instruction *I) const {
  if (I->getOpcode() != Instruction::FMul)
    return true;

  if (I->getNumUses() != 1)
    return true;

  Instruction *User = I->user_back();

  if (User &&
      !(User->getOpcode() == Instruction::FSub ||
        User->getOpcode() == Instruction::FAdd))
    return true;

  const TargetOptions &Options = getTargetMachine().Options;
  const DataLayout &DL = I->getModule()->getDataLayout();
  EVT VT = getValueType(DL, User->getOperand(0)->getType());

  if (isFMAFasterThanFMulAndFAdd(VT) &&
      isOperationLegalOrCustom(ISD::FMA, VT) &&
      (Options.AllowFPOpFusion == FPOpFusion::Fast || Options.UnsafeFPMath))
    return false;

  return true;
}

// All 32-bit GPR operations implicitly zero the high-half of the corresponding
// 64-bit GPR.
bool AArch64TargetLowering::isZExtFree(Type *Ty1, Type *Ty2) const {
  if (!Ty1->isIntegerTy() || !Ty2->isIntegerTy())
    return false;
  unsigned NumBits1 = Ty1->getPrimitiveSizeInBits();
  unsigned NumBits2 = Ty2->getPrimitiveSizeInBits();
  return NumBits1 == 32 && NumBits2 == 64;
}
bool AArch64TargetLowering::isZExtFree(EVT VT1, EVT VT2) const {
  if (VT1.isVector() || VT2.isVector() || !VT1.isInteger() || !VT2.isInteger())
    return false;
  unsigned NumBits1 = VT1.getSizeInBits();
  unsigned NumBits2 = VT2.getSizeInBits();
  return NumBits1 == 32 && NumBits2 == 64;
}

bool AArch64TargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  EVT VT1 = Val.getValueType();
  if (isZExtFree(VT1, VT2)) {
    return true;
  }

  if (Val.getOpcode() != ISD::LOAD)
    return false;

  // 8-, 16-, and 32-bit integer loads all implicitly zero-extend.
  return (VT1.isSimple() && !VT1.isVector() && VT1.isInteger() &&
          VT2.isSimple() && !VT2.isVector() && VT2.isInteger() &&
          VT1.getSizeInBits() <= 32);
}

bool AArch64TargetLowering::isExtFreeImpl(const Instruction *Ext) const {
  if (isa<FPExtInst>(Ext))
    return false;

  // Vector types are next free.
  if (Ext->getType()->isVectorTy())
    return false;

  for (const Use &U : Ext->uses()) {
    // The extension is free if we can fold it with a left shift in an
    // addressing mode or an arithmetic operation: add, sub, and cmp.

    // Is there a shift?
    const Instruction *Instr = cast<Instruction>(U.getUser());

    // Is this a constant shift?
    switch (Instr->getOpcode()) {
    case Instruction::Shl:
      if (!isa<ConstantInt>(Instr->getOperand(1)))
        return false;
      break;
    case Instruction::GetElementPtr: {
      gep_type_iterator GTI = gep_type_begin(Instr);
      auto &DL = Ext->getModule()->getDataLayout();
      std::advance(GTI, U.getOperandNo());
      Type *IdxTy = *GTI;
      // This extension will end up with a shift because of the scaling factor.
      // 8-bit sized types have a scaling factor of 1, thus a shift amount of 0.
      // Get the shift amount based on the scaling factor:
      // log2(sizeof(IdxTy)) - log2(8).
      uint64_t ShiftAmt =
          countTrailingZeros(DL.getTypeStoreSizeInBits(IdxTy)) - 3;
      // Is the constant foldable in the shift of the addressing mode?
      // I.e., shift amount is between 1 and 4 inclusive.
      if (ShiftAmt == 0 || ShiftAmt > 4)
        return false;
      break;
    }
    case Instruction::Trunc:
      // Check if this is a noop.
      // trunc(sext ty1 to ty2) to ty1.
      if (Instr->getType() == Ext->getOperand(0)->getType())
        continue;
    // FALL THROUGH.
    default:
      return false;
    }

    // At this point we can use the bfm family, so this extension is free
    // for that use.
  }
  return true;
}

bool AArch64TargetLowering::hasPairedLoad(Type *LoadedType,
                                          unsigned &RequiredAligment) const {
  if (!LoadedType->isIntegerTy() && !LoadedType->isFloatTy())
    return false;
  // Cyclone supports unaligned accesses.
  RequiredAligment = 0;
  unsigned NumBits = LoadedType->getPrimitiveSizeInBits();
  return NumBits == 32 || NumBits == 64;
}

bool AArch64TargetLowering::hasPairedLoad(EVT LoadedType,
                                          unsigned &RequiredAligment) const {
  if (!LoadedType.isSimple() ||
      (!LoadedType.isInteger() && !LoadedType.isFloatingPoint()))
    return false;
  // Cyclone supports unaligned accesses.
  RequiredAligment = 0;
  unsigned NumBits = LoadedType.getSizeInBits();
  return NumBits == 32 || NumBits == 64;
}

/// \brief Lower an interleaved load into a ldN intrinsic.
///
/// E.g. Lower an interleaved load (Factor = 2):
///        %wide.vec = load <8 x i32>, <8 x i32>* %ptr
///        %v0 = shuffle %wide.vec, undef, <0, 2, 4, 6>  ; Extract even elements
///        %v1 = shuffle %wide.vec, undef, <1, 3, 5, 7>  ; Extract odd elements
///
///      Into:
///        %ld2 = { <4 x i32>, <4 x i32> } call llvm.aarch64.neon.ld2(%ptr)
///        %vec0 = extractelement { <4 x i32>, <4 x i32> } %ld2, i32 0
///        %vec1 = extractelement { <4 x i32>, <4 x i32> } %ld2, i32 1
bool AArch64TargetLowering::lowerInterleavedLoad(
    LoadInst *LI, ArrayRef<ShuffleVectorInst *> Shuffles,
    ArrayRef<unsigned> Indices, unsigned Factor) const {
  assert(Factor >= 2 && Factor <= getMaxSupportedInterleaveFactor() &&
         "Invalid interleave factor");
  assert(!Shuffles.empty() && "Empty shufflevector input");
  assert(Shuffles.size() == Indices.size() &&
         "Unmatched number of shufflevectors and indices");

  const DataLayout &DL = LI->getModule()->getDataLayout();

  VectorType *VecTy = Shuffles[0]->getType();
  unsigned VecSize = DL.getTypeAllocSizeInBits(VecTy);

  // Skip if we do not have NEON and skip illegal vector types.
  if (!Subtarget->hasNEON() || (VecSize != 64 && VecSize != 128))
    return false;

  // A pointer vector can not be the return type of the ldN intrinsics. Need to
  // load integer vectors first and then convert to pointer vectors.
  Type *EltTy = VecTy->getVectorElementType();
  if (EltTy->isPointerTy())
    VecTy =
        VectorType::get(DL.getIntPtrType(EltTy), VecTy->getVectorNumElements());

  Type *PtrTy = VecTy->getPointerTo(LI->getPointerAddressSpace());
  Type *Tys[2] = {VecTy, PtrTy};
  static const Intrinsic::ID LoadInts[3] = {Intrinsic::aarch64_neon_ld2,
                                            Intrinsic::aarch64_neon_ld3,
                                            Intrinsic::aarch64_neon_ld4};
  Function *LdNFunc =
      Intrinsic::getDeclaration(LI->getModule(), LoadInts[Factor - 2], Tys);

  IRBuilder<> Builder(LI);
  Value *Ptr = Builder.CreateBitCast(LI->getPointerOperand(), PtrTy);

  CallInst *LdN = Builder.CreateCall(LdNFunc, Ptr, "ldN");

  // Replace uses of each shufflevector with the corresponding vector loaded
  // by ldN.
  for (unsigned i = 0; i < Shuffles.size(); i++) {
    ShuffleVectorInst *SVI = Shuffles[i];
    unsigned Index = Indices[i];

    Value *SubVec = Builder.CreateExtractValue(LdN, Index);

    // Convert the integer vector to pointer vector if the element is pointer.
    if (EltTy->isPointerTy())
      SubVec = Builder.CreateIntToPtr(SubVec, SVI->getType());

    SVI->replaceAllUsesWith(SubVec);
  }

  return true;
}

/// \brief Get a mask consisting of sequential integers starting from \p Start.
///
/// I.e. <Start, Start + 1, ..., Start + NumElts - 1>
static Constant *getSequentialMask(IRBuilder<> &Builder, unsigned Start,
                                   unsigned NumElts) {
  SmallVector<Constant *, 16> Mask;
  for (unsigned i = 0; i < NumElts; i++)
    Mask.push_back(Builder.getInt32(Start + i));

  return ConstantVector::get(Mask);
}

/// \brief Lower an interleaved store into a stN intrinsic.
///
/// E.g. Lower an interleaved store (Factor = 3):
///        %i.vec = shuffle <8 x i32> %v0, <8 x i32> %v1,
///                                  <0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11>
///        store <12 x i32> %i.vec, <12 x i32>* %ptr
///
///      Into:
///        %sub.v0 = shuffle <8 x i32> %v0, <8 x i32> v1, <0, 1, 2, 3>
///        %sub.v1 = shuffle <8 x i32> %v0, <8 x i32> v1, <4, 5, 6, 7>
///        %sub.v2 = shuffle <8 x i32> %v0, <8 x i32> v1, <8, 9, 10, 11>
///        call void llvm.aarch64.neon.st3(%sub.v0, %sub.v1, %sub.v2, %ptr)
///
/// Note that the new shufflevectors will be removed and we'll only generate one
/// st3 instruction in CodeGen.
bool AArch64TargetLowering::lowerInterleavedStore(StoreInst *SI,
                                                  ShuffleVectorInst *SVI,
                                                  unsigned Factor) const {
  assert(Factor >= 2 && Factor <= getMaxSupportedInterleaveFactor() &&
         "Invalid interleave factor");

  VectorType *VecTy = SVI->getType();
  assert(VecTy->getVectorNumElements() % Factor == 0 &&
         "Invalid interleaved store");

  unsigned NumSubElts = VecTy->getVectorNumElements() / Factor;
  Type *EltTy = VecTy->getVectorElementType();
  VectorType *SubVecTy = VectorType::get(EltTy, NumSubElts);

  const DataLayout &DL = SI->getModule()->getDataLayout();
  unsigned SubVecSize = DL.getTypeAllocSizeInBits(SubVecTy);

  // Skip if we do not have NEON and skip illegal vector types.
  if (!Subtarget->hasNEON() || (SubVecSize != 64 && SubVecSize != 128))
    return false;

  Value *Op0 = SVI->getOperand(0);
  Value *Op1 = SVI->getOperand(1);
  IRBuilder<> Builder(SI);

  // StN intrinsics don't support pointer vectors as arguments. Convert pointer
  // vectors to integer vectors.
  if (EltTy->isPointerTy()) {
    Type *IntTy = DL.getIntPtrType(EltTy);
    unsigned NumOpElts =
        dyn_cast<VectorType>(Op0->getType())->getVectorNumElements();

    // Convert to the corresponding integer vector.
    Type *IntVecTy = VectorType::get(IntTy, NumOpElts);
    Op0 = Builder.CreatePtrToInt(Op0, IntVecTy);
    Op1 = Builder.CreatePtrToInt(Op1, IntVecTy);

    SubVecTy = VectorType::get(IntTy, NumSubElts);
  }

  Type *PtrTy = SubVecTy->getPointerTo(SI->getPointerAddressSpace());
  Type *Tys[2] = {SubVecTy, PtrTy};
  static const Intrinsic::ID StoreInts[3] = {Intrinsic::aarch64_neon_st2,
                                             Intrinsic::aarch64_neon_st3,
                                             Intrinsic::aarch64_neon_st4};
  Function *StNFunc =
      Intrinsic::getDeclaration(SI->getModule(), StoreInts[Factor - 2], Tys);

  SmallVector<Value *, 5> Ops;

  // Split the shufflevector operands into sub vectors for the new stN call.
  for (unsigned i = 0; i < Factor; i++)
    Ops.push_back(Builder.CreateShuffleVector(
        Op0, Op1, getSequentialMask(Builder, NumSubElts * i, NumSubElts)));

  Ops.push_back(Builder.CreateBitCast(SI->getPointerOperand(), PtrTy));
  Builder.CreateCall(StNFunc, Ops);
  return true;
}

static bool memOpAlign(unsigned DstAlign, unsigned SrcAlign,
                       unsigned AlignCheck) {
  return ((SrcAlign == 0 || SrcAlign % AlignCheck == 0) &&
          (DstAlign == 0 || DstAlign % AlignCheck == 0));
}

EVT AArch64TargetLowering::getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                                               unsigned SrcAlign, bool IsMemset,
                                               bool ZeroMemset,
                                               bool MemcpyStrSrc,
                                               MachineFunction &MF) const {
  // Don't use AdvSIMD to implement 16-byte memset. It would have taken one
  // instruction to materialize the v2i64 zero and one store (with restrictive
  // addressing mode). Just do two i64 store of zero-registers.
  bool Fast;
  const Function *F = MF.getFunction();
  if (Subtarget->hasFPARMv8() && !IsMemset && Size >= 16 &&
      !F->hasFnAttribute(Attribute::NoImplicitFloat) &&
      (memOpAlign(SrcAlign, DstAlign, 16) ||
       (allowsMisalignedMemoryAccesses(MVT::f128, 0, 1, &Fast) && Fast)))
    return MVT::f128;

  if (Size >= 8 &&
      (memOpAlign(SrcAlign, DstAlign, 8) ||
       (allowsMisalignedMemoryAccesses(MVT::i64, 0, 1, &Fast) && Fast)))
    return MVT::i64;

  if (Size >= 4 &&
      (memOpAlign(SrcAlign, DstAlign, 4) ||
       (allowsMisalignedMemoryAccesses(MVT::i32, 0, 1, &Fast) && Fast)))
    return MVT::i32;

  return MVT::Other;
}

// 12-bit optionally shifted immediates are legal for adds.
bool AArch64TargetLowering::isLegalAddImmediate(int64_t Immed) const {
  if ((Immed >> 12) == 0 || ((Immed & 0xfff) == 0 && Immed >> 24 == 0))
    return true;
  return false;
}

// Integer comparisons are implemented with ADDS/SUBS, so the range of valid
// immediates is the same as for an add or a sub.
bool AArch64TargetLowering::isLegalICmpImmediate(int64_t Immed) const {
  if (Immed < 0)
    Immed *= -1;
  return isLegalAddImmediate(Immed);
}

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool AArch64TargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                  const AddrMode &AM, Type *Ty,
                                                  unsigned AS) const {
  // AArch64 has five basic addressing modes:
  //  reg
  //  reg + 9-bit signed offset
  //  reg + SIZE_IN_BYTES * 12-bit unsigned offset
  //  reg1 + reg2
  //  reg + SIZE_IN_BYTES * reg

  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // No reg+reg+imm addressing.
  if (AM.HasBaseReg && AM.BaseOffs && AM.Scale)
    return false;

  // check reg + imm case:
  // i.e., reg + 0, reg + imm9, reg + SIZE_IN_BYTES * uimm12
  uint64_t NumBytes = 0;
  if (Ty->isSized()) {
    uint64_t NumBits = DL.getTypeSizeInBits(Ty);
    NumBytes = NumBits / 8;
    if (!isPowerOf2_64(NumBits))
      NumBytes = 0;
  }

  if (!AM.Scale) {
    int64_t Offset = AM.BaseOffs;

    // 9-bit signed offset
    if (Offset >= -(1LL << 9) && Offset <= (1LL << 9) - 1)
      return true;

    // 12-bit unsigned offset
    unsigned shift = Log2_64(NumBytes);
    if (NumBytes && Offset > 0 && (Offset / NumBytes) <= (1LL << 12) - 1 &&
        // Must be a multiple of NumBytes (NumBytes is a power of 2)
        (Offset >> shift) << shift == Offset)
      return true;
    return false;
  }

  // Check reg1 + SIZE_IN_BYTES * reg2 and reg1 + reg2

  if (!AM.Scale || AM.Scale == 1 ||
      (AM.Scale > 0 && (uint64_t)AM.Scale == NumBytes))
    return true;
  return false;
}

int AArch64TargetLowering::getScalingFactorCost(const DataLayout &DL,
                                                const AddrMode &AM, Type *Ty,
                                                unsigned AS) const {
  // Scaling factors are not free at all.
  // Operands                     | Rt Latency
  // -------------------------------------------
  // Rt, [Xn, Xm]                 | 4
  // -------------------------------------------
  // Rt, [Xn, Xm, lsl #imm]       | Rn: 4 Rm: 5
  // Rt, [Xn, Wm, <extend> #imm]  |
  if (isLegalAddressingMode(DL, AM, Ty, AS))
    // Scale represents reg2 * scale, thus account for 1 if
    // it is not equal to 0 or 1.
    return AM.Scale != 0 && AM.Scale != 1;
  return -1;
}

bool AArch64TargetLowering::isFMAFasterThanFMulAndFAdd(EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
  case MVT::f64:
    return true;
  default:
    break;
  }

  return false;
}

const MCPhysReg *
AArch64TargetLowering::getScratchRegisters(CallingConv::ID) const {
  // LR is a callee-save register, but we must treat it as clobbered by any call
  // site. Hence we include LR in the scratch registers, which are in turn added
  // as implicit-defs for stackmaps and patchpoints.
  static const MCPhysReg ScratchRegs[] = {
    AArch64::X16, AArch64::X17, AArch64::LR, 0
  };
  return ScratchRegs;
}

bool
AArch64TargetLowering::isDesirableToCommuteWithShift(const SDNode *N) const {
  EVT VT = N->getValueType(0);
    // If N is unsigned bit extraction: ((x >> C) & mask), then do not combine
    // it with shift to let it be lowered to UBFX.
  if (N->getOpcode() == ISD::AND && (VT == MVT::i32 || VT == MVT::i64) &&
      isa<ConstantSDNode>(N->getOperand(1))) {
    uint64_t TruncMask = N->getConstantOperandVal(1);
    if (isMask_64(TruncMask) &&
      N->getOperand(0).getOpcode() == ISD::SRL &&
      isa<ConstantSDNode>(N->getOperand(0)->getOperand(1)))
      return false;
  }
  return true;
}

bool AArch64TargetLowering::shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                                              Type *Ty) const {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return false;

  int64_t Val = Imm.getSExtValue();
  if (Val == 0 || AArch64_AM::isLogicalImmediate(Val, BitSize))
    return true;

  if ((int64_t)Val < 0)
    Val = ~Val;
  if (BitSize == 32)
    Val &= (1LL << 32) - 1;

  unsigned LZ = countLeadingZeros((uint64_t)Val);
  unsigned Shift = (63 - LZ) / 16;
  // MOVZ is free so return true for one or fewer MOVK.
  return Shift < 3;
}

// Generate SUBS and CSEL for integer abs.
static SDValue performIntegerAbsCombine(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDLoc DL(N);

  // Check pattern of XOR(ADD(X,Y), Y) where Y is SRA(X, size(X)-1)
  // and change it to SUB and CSEL.
  if (VT.isInteger() && N->getOpcode() == ISD::XOR &&
      N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1 &&
      N1.getOpcode() == ISD::SRA && N1.getOperand(0) == N0.getOperand(0))
    if (ConstantSDNode *Y1C = dyn_cast<ConstantSDNode>(N1.getOperand(1)))
      if (Y1C->getAPIntValue() == VT.getSizeInBits() - 1) {
        SDValue Neg = DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT),
                                  N0.getOperand(0));
        // Generate SUBS & CSEL.
        SDValue Cmp =
            DAG.getNode(AArch64ISD::SUBS, DL, DAG.getVTList(VT, MVT::i32),
                        N0.getOperand(0), DAG.getConstant(0, DL, VT));
        return DAG.getNode(AArch64ISD::CSEL, DL, VT, N0.getOperand(0), Neg,
                           DAG.getConstant(AArch64CC::PL, DL, MVT::i32),
                           SDValue(Cmp.getNode(), 1));
      }
  return SDValue();
}

// performXorCombine - Attempts to handle integer ABS.
static SDValue performXorCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const AArch64Subtarget *Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  return performIntegerAbsCombine(N, DAG);
}

SDValue
AArch64TargetLowering::BuildSDIVPow2(SDNode *N, const APInt &Divisor,
                                     SelectionDAG &DAG,
                                     std::vector<SDNode *> *Created) const {
  // fold (sdiv X, pow2)
  EVT VT = N->getValueType(0);
  if ((VT != MVT::i32 && VT != MVT::i64) ||
      !(Divisor.isPowerOf2() || (-Divisor).isPowerOf2()))
    return SDValue();

  SDLoc DL(N);
  SDValue N0 = N->getOperand(0);
  unsigned Lg2 = Divisor.countTrailingZeros();
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue Pow2MinusOne = DAG.getConstant((1ULL << Lg2) - 1, DL, VT);

  // Add (N0 < 0) ? Pow2 - 1 : 0;
  SDValue CCVal;
  SDValue Cmp = getAArch64Cmp(N0, Zero, ISD::SETLT, CCVal, DAG, DL);
  SDValue Add = DAG.getNode(ISD::ADD, DL, VT, N0, Pow2MinusOne);
  SDValue CSel = DAG.getNode(AArch64ISD::CSEL, DL, VT, Add, N0, CCVal, Cmp);

  if (Created) {
    Created->push_back(Cmp.getNode());
    Created->push_back(Add.getNode());
    Created->push_back(CSel.getNode());
  }

  // Divide by pow2.
  SDValue SRA =
      DAG.getNode(ISD::SRA, DL, VT, CSel, DAG.getConstant(Lg2, DL, MVT::i64));

  // If we're dividing by a positive value, we're done.  Otherwise, we must
  // negate the result.
  if (Divisor.isNonNegative())
    return SRA;

  if (Created)
    Created->push_back(SRA.getNode());
  return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT), SRA);
}

static SDValue performMulCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const AArch64Subtarget *Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  // Multiplication of a power of two plus/minus one can be done more
  // cheaply as as shift+add/sub. For now, this is true unilaterally. If
  // future CPUs have a cheaper MADD instruction, this may need to be
  // gated on a subtarget feature. For Cyclone, 32-bit MADD is 4 cycles and
  // 64-bit is 5 cycles, so this is always a win.
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
    APInt Value = C->getAPIntValue();
    EVT VT = N->getValueType(0);
    SDLoc DL(N);
    if (Value.isNonNegative()) {
      // (mul x, 2^N + 1) => (add (shl x, N), x)
      APInt VM1 = Value - 1;
      if (VM1.isPowerOf2()) {
        SDValue ShiftedVal =
            DAG.getNode(ISD::SHL, DL, VT, N->getOperand(0),
                        DAG.getConstant(VM1.logBase2(), DL, MVT::i64));
        return DAG.getNode(ISD::ADD, DL, VT, ShiftedVal,
                           N->getOperand(0));
      }
      // (mul x, 2^N - 1) => (sub (shl x, N), x)
      APInt VP1 = Value + 1;
      if (VP1.isPowerOf2()) {
        SDValue ShiftedVal =
            DAG.getNode(ISD::SHL, DL, VT, N->getOperand(0),
                        DAG.getConstant(VP1.logBase2(), DL, MVT::i64));
        return DAG.getNode(ISD::SUB, DL, VT, ShiftedVal,
                           N->getOperand(0));
      }
    } else {
      // (mul x, -(2^N - 1)) => (sub x, (shl x, N))
      APInt VNP1 = -Value + 1;
      if (VNP1.isPowerOf2()) {
        SDValue ShiftedVal =
            DAG.getNode(ISD::SHL, DL, VT, N->getOperand(0),
                        DAG.getConstant(VNP1.logBase2(), DL, MVT::i64));
        return DAG.getNode(ISD::SUB, DL, VT, N->getOperand(0),
                           ShiftedVal);
      }
      // (mul x, -(2^N + 1)) => - (add (shl x, N), x)
      APInt VNM1 = -Value - 1;
      if (VNM1.isPowerOf2()) {
        SDValue ShiftedVal =
            DAG.getNode(ISD::SHL, DL, VT, N->getOperand(0),
                        DAG.getConstant(VNM1.logBase2(), DL, MVT::i64));
        SDValue Add =
            DAG.getNode(ISD::ADD, DL, VT, ShiftedVal, N->getOperand(0));
        return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT), Add);
      }
    }
  }
  return SDValue();
}

static SDValue performVectorCompareAndMaskUnaryOpCombine(SDNode *N,
                                                         SelectionDAG &DAG) {
  // Take advantage of vector comparisons producing 0 or -1 in each lane to
  // optimize away operation when it's from a constant.
  //
  // The general transformation is:
  //    UNARYOP(AND(VECTOR_CMP(x,y), constant)) -->
  //       AND(VECTOR_CMP(x,y), constant2)
  //    constant2 = UNARYOP(constant)

  // Early exit if this isn't a vector operation, the operand of the
  // unary operation isn't a bitwise AND, or if the sizes of the operations
  // aren't the same.
  EVT VT = N->getValueType(0);
  if (!VT.isVector() || N->getOperand(0)->getOpcode() != ISD::AND ||
      N->getOperand(0)->getOperand(0)->getOpcode() != ISD::SETCC ||
      VT.getSizeInBits() != N->getOperand(0)->getValueType(0).getSizeInBits())
    return SDValue();

  // Now check that the other operand of the AND is a constant. We could
  // make the transformation for non-constant splats as well, but it's unclear
  // that would be a benefit as it would not eliminate any operations, just
  // perform one more step in scalar code before moving to the vector unit.
  if (BuildVectorSDNode *BV =
          dyn_cast<BuildVectorSDNode>(N->getOperand(0)->getOperand(1))) {
    // Bail out if the vector isn't a constant.
    if (!BV->isConstant())
      return SDValue();

    // Everything checks out. Build up the new and improved node.
    SDLoc DL(N);
    EVT IntVT = BV->getValueType(0);
    // Create a new constant of the appropriate type for the transformed
    // DAG.
    SDValue SourceConst = DAG.getNode(N->getOpcode(), DL, VT, SDValue(BV, 0));
    // The AND node needs bitcasts to/from an integer vector type around it.
    SDValue MaskConst = DAG.getNode(ISD::BITCAST, DL, IntVT, SourceConst);
    SDValue NewAnd = DAG.getNode(ISD::AND, DL, IntVT,
                                 N->getOperand(0)->getOperand(0), MaskConst);
    SDValue Res = DAG.getNode(ISD::BITCAST, DL, VT, NewAnd);
    return Res;
  }

  return SDValue();
}

static SDValue performIntToFpCombine(SDNode *N, SelectionDAG &DAG,
                                     const AArch64Subtarget *Subtarget) {
  // First try to optimize away the conversion when it's conditionally from
  // a constant. Vectors only.
  if (SDValue Res = performVectorCompareAndMaskUnaryOpCombine(N, DAG))
    return Res;

  EVT VT = N->getValueType(0);
  if (VT != MVT::f32 && VT != MVT::f64)
    return SDValue();

  // Only optimize when the source and destination types have the same width.
  if (VT.getSizeInBits() != N->getOperand(0).getValueType().getSizeInBits())
    return SDValue();

  // If the result of an integer load is only used by an integer-to-float
  // conversion, use a fp load instead and a AdvSIMD scalar {S|U}CVTF instead.
  // This eliminates an "integer-to-vector-move" UOP and improves throughput.
  SDValue N0 = N->getOperand(0);
  if (Subtarget->hasNEON() && ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      // Do not change the width of a volatile load.
      !cast<LoadSDNode>(N0)->isVolatile()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue Load = DAG.getLoad(VT, SDLoc(N), LN0->getChain(), LN0->getBasePtr(),
                               LN0->getPointerInfo(), LN0->isVolatile(),
                               LN0->isNonTemporal(), LN0->isInvariant(),
                               LN0->getAlignment());

    // Make sure successors of the original load stay after it by updating them
    // to use the new Chain.
    DAG.ReplaceAllUsesOfValueWith(SDValue(LN0, 1), Load.getValue(1));

    unsigned Opcode =
        (N->getOpcode() == ISD::SINT_TO_FP) ? AArch64ISD::SITOF : AArch64ISD::UITOF;
    return DAG.getNode(Opcode, SDLoc(N), VT, Load);
  }

  return SDValue();
}

/// Fold a floating-point multiply by power of two into floating-point to
/// fixed-point conversion.
static SDValue performFpToIntCombine(SDNode *N, SelectionDAG &DAG,
                                     const AArch64Subtarget *Subtarget) {
  if (!Subtarget->hasNEON())
    return SDValue();

  SDValue Op = N->getOperand(0);
  if (!Op.getValueType().isVector() || Op.getOpcode() != ISD::FMUL)
    return SDValue();

  SDValue ConstVec = Op->getOperand(1);
  if (!isa<BuildVectorSDNode>(ConstVec))
    return SDValue();

  MVT FloatTy = Op.getSimpleValueType().getVectorElementType();
  uint32_t FloatBits = FloatTy.getSizeInBits();
  if (FloatBits != 32 && FloatBits != 64)
    return SDValue();

  MVT IntTy = N->getSimpleValueType(0).getVectorElementType();
  uint32_t IntBits = IntTy.getSizeInBits();
  if (IntBits != 16 && IntBits != 32 && IntBits != 64)
    return SDValue();

  // Avoid conversions where iN is larger than the float (e.g., float -> i64).
  if (IntBits > FloatBits)
    return SDValue();

  BitVector UndefElements;
  BuildVectorSDNode *BV = cast<BuildVectorSDNode>(ConstVec);
  int32_t Bits = IntBits == 64 ? 64 : 32;
  int32_t C = BV->getConstantFPSplatPow2ToLog2Int(&UndefElements, Bits + 1);
  if (C == -1 || C == 0 || C > Bits)
    return SDValue();

  MVT ResTy;
  unsigned NumLanes = Op.getValueType().getVectorNumElements();
  switch (NumLanes) {
  default:
    return SDValue();
  case 2:
    ResTy = FloatBits == 32 ? MVT::v2i32 : MVT::v2i64;
    break;
  case 4:
    ResTy = MVT::v4i32;
    break;
  }

  SDLoc DL(N);
  bool IsSigned = N->getOpcode() == ISD::FP_TO_SINT;
  unsigned IntrinsicOpcode = IsSigned ? Intrinsic::aarch64_neon_vcvtfp2fxs
                                      : Intrinsic::aarch64_neon_vcvtfp2fxu;
  SDValue FixConv =
      DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, ResTy,
                  DAG.getConstant(IntrinsicOpcode, DL, MVT::i32),
                  Op->getOperand(0), DAG.getConstant(C, DL, MVT::i32));
  // We can handle smaller integers by generating an extra trunc.
  if (IntBits < FloatBits)
    FixConv = DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), FixConv);

  return FixConv;
}

/// Fold a floating-point divide by power of two into fixed-point to
/// floating-point conversion.
static SDValue performFDivCombine(SDNode *N, SelectionDAG &DAG,
                                  const AArch64Subtarget *Subtarget) {
  if (!Subtarget->hasNEON())
    return SDValue();

  SDValue Op = N->getOperand(0);
  unsigned Opc = Op->getOpcode();
  if (!Op.getValueType().isVector() ||
      (Opc != ISD::SINT_TO_FP && Opc != ISD::UINT_TO_FP))
    return SDValue();

  SDValue ConstVec = N->getOperand(1);
  if (!isa<BuildVectorSDNode>(ConstVec))
    return SDValue();

  MVT IntTy = Op.getOperand(0).getSimpleValueType().getVectorElementType();
  int32_t IntBits = IntTy.getSizeInBits();
  if (IntBits != 16 && IntBits != 32 && IntBits != 64)
    return SDValue();

  MVT FloatTy = N->getSimpleValueType(0).getVectorElementType();
  int32_t FloatBits = FloatTy.getSizeInBits();
  if (FloatBits != 32 && FloatBits != 64)
    return SDValue();

  // Avoid conversions where iN is larger than the float (e.g., i64 -> float).
  if (IntBits > FloatBits)
    return SDValue();

  BitVector UndefElements;
  BuildVectorSDNode *BV = cast<BuildVectorSDNode>(ConstVec);
  int32_t C = BV->getConstantFPSplatPow2ToLog2Int(&UndefElements, FloatBits + 1);
  if (C == -1 || C == 0 || C > FloatBits)
    return SDValue();

  MVT ResTy;
  unsigned NumLanes = Op.getValueType().getVectorNumElements();
  switch (NumLanes) {
  default:
    return SDValue();
  case 2:
    ResTy = FloatBits == 32 ? MVT::v2i32 : MVT::v2i64;
    break;
  case 4:
    ResTy = MVT::v4i32;
    break;
  }

  SDLoc DL(N);
  SDValue ConvInput = Op.getOperand(0);
  bool IsSigned = Opc == ISD::SINT_TO_FP;
  if (IntBits < FloatBits)
    ConvInput = DAG.getNode(IsSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND, DL,
                            ResTy, ConvInput);

  unsigned IntrinsicOpcode = IsSigned ? Intrinsic::aarch64_neon_vcvtfxs2fp
                                      : Intrinsic::aarch64_neon_vcvtfxu2fp;
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, Op.getValueType(),
                     DAG.getConstant(IntrinsicOpcode, DL, MVT::i32), ConvInput,
                     DAG.getConstant(C, DL, MVT::i32));
}

/// An EXTR instruction is made up of two shifts, ORed together. This helper
/// searches for and classifies those shifts.
static bool findEXTRHalf(SDValue N, SDValue &Src, uint32_t &ShiftAmount,
                         bool &FromHi) {
  if (N.getOpcode() == ISD::SHL)
    FromHi = false;
  else if (N.getOpcode() == ISD::SRL)
    FromHi = true;
  else
    return false;

  if (!isa<ConstantSDNode>(N.getOperand(1)))
    return false;

  ShiftAmount = N->getConstantOperandVal(1);
  Src = N->getOperand(0);
  return true;
}

/// EXTR instruction extracts a contiguous chunk of bits from two existing
/// registers viewed as a high/low pair. This function looks for the pattern:
/// (or (shl VAL1, #N), (srl VAL2, #RegWidth-N)) and replaces it with an
/// EXTR. Can't quite be done in TableGen because the two immediates aren't
/// independent.
static SDValue tryCombineToEXTR(SDNode *N,
                                TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  assert(N->getOpcode() == ISD::OR && "Unexpected root");

  if (VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  SDValue LHS;
  uint32_t ShiftLHS = 0;
  bool LHSFromHi = 0;
  if (!findEXTRHalf(N->getOperand(0), LHS, ShiftLHS, LHSFromHi))
    return SDValue();

  SDValue RHS;
  uint32_t ShiftRHS = 0;
  bool RHSFromHi = 0;
  if (!findEXTRHalf(N->getOperand(1), RHS, ShiftRHS, RHSFromHi))
    return SDValue();

  // If they're both trying to come from the high part of the register, they're
  // not really an EXTR.
  if (LHSFromHi == RHSFromHi)
    return SDValue();

  if (ShiftLHS + ShiftRHS != VT.getSizeInBits())
    return SDValue();

  if (LHSFromHi) {
    std::swap(LHS, RHS);
    std::swap(ShiftLHS, ShiftRHS);
  }

  return DAG.getNode(AArch64ISD::EXTR, DL, VT, LHS, RHS,
                     DAG.getConstant(ShiftRHS, DL, MVT::i64));
}

static SDValue tryCombineToBSL(SDNode *N,
                                TargetLowering::DAGCombinerInfo &DCI) {
  EVT VT = N->getValueType(0);
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  if (!VT.isVector())
    return SDValue();

  SDValue N0 = N->getOperand(0);
  if (N0.getOpcode() != ISD::AND)
    return SDValue();

  SDValue N1 = N->getOperand(1);
  if (N1.getOpcode() != ISD::AND)
    return SDValue();

  // We only have to look for constant vectors here since the general, variable
  // case can be handled in TableGen.
  unsigned Bits = VT.getVectorElementType().getSizeInBits();
  uint64_t BitMask = Bits == 64 ? -1ULL : ((1ULL << Bits) - 1);
  for (int i = 1; i >= 0; --i)
    for (int j = 1; j >= 0; --j) {
      BuildVectorSDNode *BVN0 = dyn_cast<BuildVectorSDNode>(N0->getOperand(i));
      BuildVectorSDNode *BVN1 = dyn_cast<BuildVectorSDNode>(N1->getOperand(j));
      if (!BVN0 || !BVN1)
        continue;

      bool FoundMatch = true;
      for (unsigned k = 0; k < VT.getVectorNumElements(); ++k) {
        ConstantSDNode *CN0 = dyn_cast<ConstantSDNode>(BVN0->getOperand(k));
        ConstantSDNode *CN1 = dyn_cast<ConstantSDNode>(BVN1->getOperand(k));
        if (!CN0 || !CN1 ||
            CN0->getZExtValue() != (BitMask & ~CN1->getZExtValue())) {
          FoundMatch = false;
          break;
        }
      }

      if (FoundMatch)
        return DAG.getNode(AArch64ISD::BSL, DL, VT, SDValue(BVN0, 0),
                           N0->getOperand(1 - i), N1->getOperand(1 - j));
    }

  return SDValue();
}

static SDValue performORCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI,
                                const AArch64Subtarget *Subtarget) {
  // Attempt to form an EXTR from (or (shl VAL1, #N), (srl VAL2, #RegWidth-N))
  if (!EnableAArch64ExtrGeneration)
    return SDValue();
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  if (!DAG.getTargetLoweringInfo().isTypeLegal(VT))
    return SDValue();

  SDValue Res = tryCombineToEXTR(N, DCI);
  if (Res.getNode())
    return Res;

  Res = tryCombineToBSL(N, DCI);
  if (Res.getNode())
    return Res;

  return SDValue();
}

static SDValue performBitcastCombine(SDNode *N,
                                     TargetLowering::DAGCombinerInfo &DCI,
                                     SelectionDAG &DAG) {
  // Wait 'til after everything is legalized to try this. That way we have
  // legal vector types and such.
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  // Remove extraneous bitcasts around an extract_subvector.
  // For example,
  //    (v4i16 (bitconvert
  //             (extract_subvector (v2i64 (bitconvert (v8i16 ...)), (i64 1)))))
  //  becomes
  //    (extract_subvector ((v8i16 ...), (i64 4)))

  // Only interested in 64-bit vectors as the ultimate result.
  EVT VT = N->getValueType(0);
  if (!VT.isVector())
    return SDValue();
  if (VT.getSimpleVT().getSizeInBits() != 64)
    return SDValue();
  // Is the operand an extract_subvector starting at the beginning or halfway
  // point of the vector? A low half may also come through as an
  // EXTRACT_SUBREG, so look for that, too.
  SDValue Op0 = N->getOperand(0);
  if (Op0->getOpcode() != ISD::EXTRACT_SUBVECTOR &&
      !(Op0->isMachineOpcode() &&
        Op0->getMachineOpcode() == AArch64::EXTRACT_SUBREG))
    return SDValue();
  uint64_t idx = cast<ConstantSDNode>(Op0->getOperand(1))->getZExtValue();
  if (Op0->getOpcode() == ISD::EXTRACT_SUBVECTOR) {
    if (Op0->getValueType(0).getVectorNumElements() != idx && idx != 0)
      return SDValue();
  } else if (Op0->getMachineOpcode() == AArch64::EXTRACT_SUBREG) {
    if (idx != AArch64::dsub)
      return SDValue();
    // The dsub reference is equivalent to a lane zero subvector reference.
    idx = 0;
  }
  // Look through the bitcast of the input to the extract.
  if (Op0->getOperand(0)->getOpcode() != ISD::BITCAST)
    return SDValue();
  SDValue Source = Op0->getOperand(0)->getOperand(0);
  // If the source type has twice the number of elements as our destination
  // type, we know this is an extract of the high or low half of the vector.
  EVT SVT = Source->getValueType(0);
  if (SVT.getVectorNumElements() != VT.getVectorNumElements() * 2)
    return SDValue();

  DEBUG(dbgs() << "aarch64-lower: bitcast extract_subvector simplification\n");

  // Create the simplified form to just extract the low or high half of the
  // vector directly rather than bothering with the bitcasts.
  SDLoc dl(N);
  unsigned NumElements = VT.getVectorNumElements();
  if (idx) {
    SDValue HalfIdx = DAG.getConstant(NumElements, dl, MVT::i64);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT, Source, HalfIdx);
  } else {
    SDValue SubReg = DAG.getTargetConstant(AArch64::dsub, dl, MVT::i32);
    return SDValue(DAG.getMachineNode(TargetOpcode::EXTRACT_SUBREG, dl, VT,
                                      Source, SubReg),
                   0);
  }
}

static SDValue performConcatVectorsCombine(SDNode *N,
                                           TargetLowering::DAGCombinerInfo &DCI,
                                           SelectionDAG &DAG) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);

  // Optimize concat_vectors of truncated vectors, where the intermediate
  // type is illegal, to avoid said illegality,  e.g.,
  //   (v4i16 (concat_vectors (v2i16 (truncate (v2i64))),
  //                          (v2i16 (truncate (v2i64)))))
  // ->
  //   (v4i16 (truncate (vector_shuffle (v4i32 (bitcast (v2i64))),
  //                                    (v4i32 (bitcast (v2i64))),
  //                                    <0, 2, 4, 6>)))
  // This isn't really target-specific, but ISD::TRUNCATE legality isn't keyed
  // on both input and result type, so we might generate worse code.
  // On AArch64 we know it's fine for v2i64->v4i16 and v4i32->v8i8.
  if (N->getNumOperands() == 2 &&
      N0->getOpcode() == ISD::TRUNCATE &&
      N1->getOpcode() == ISD::TRUNCATE) {
    SDValue N00 = N0->getOperand(0);
    SDValue N10 = N1->getOperand(0);
    EVT N00VT = N00.getValueType();

    if (N00VT == N10.getValueType() &&
        (N00VT == MVT::v2i64 || N00VT == MVT::v4i32) &&
        N00VT.getScalarSizeInBits() == 4 * VT.getScalarSizeInBits()) {
      MVT MidVT = (N00VT == MVT::v2i64 ? MVT::v4i32 : MVT::v8i16);
      SmallVector<int, 8> Mask(MidVT.getVectorNumElements());
      for (size_t i = 0; i < Mask.size(); ++i)
        Mask[i] = i * 2;
      return DAG.getNode(ISD::TRUNCATE, dl, VT,
                         DAG.getVectorShuffle(
                             MidVT, dl,
                             DAG.getNode(ISD::BITCAST, dl, MidVT, N00),
                             DAG.getNode(ISD::BITCAST, dl, MidVT, N10), Mask));
    }
  }

  // Wait 'til after everything is legalized to try this. That way we have
  // legal vector types and such.
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  // If we see a (concat_vectors (v1x64 A), (v1x64 A)) it's really a vector
  // splat. The indexed instructions are going to be expecting a DUPLANE64, so
  // canonicalise to that.
  if (N0 == N1 && VT.getVectorNumElements() == 2) {
    assert(VT.getVectorElementType().getSizeInBits() == 64);
    return DAG.getNode(AArch64ISD::DUPLANE64, dl, VT, WidenVector(N0, DAG),
                       DAG.getConstant(0, dl, MVT::i64));
  }

  // Canonicalise concat_vectors so that the right-hand vector has as few
  // bit-casts as possible before its real operation. The primary matching
  // destination for these operations will be the narrowing "2" instructions,
  // which depend on the operation being performed on this right-hand vector.
  // For example,
  //    (concat_vectors LHS,  (v1i64 (bitconvert (v4i16 RHS))))
  // becomes
  //    (bitconvert (concat_vectors (v4i16 (bitconvert LHS)), RHS))

  if (N1->getOpcode() != ISD::BITCAST)
    return SDValue();
  SDValue RHS = N1->getOperand(0);
  MVT RHSTy = RHS.getValueType().getSimpleVT();
  // If the RHS is not a vector, this is not the pattern we're looking for.
  if (!RHSTy.isVector())
    return SDValue();

  DEBUG(dbgs() << "aarch64-lower: concat_vectors bitcast simplification\n");

  MVT ConcatTy = MVT::getVectorVT(RHSTy.getVectorElementType(),
                                  RHSTy.getVectorNumElements() * 2);
  return DAG.getNode(ISD::BITCAST, dl, VT,
                     DAG.getNode(ISD::CONCAT_VECTORS, dl, ConcatTy,
                                 DAG.getNode(ISD::BITCAST, dl, RHSTy, N0),
                                 RHS));
}

static SDValue tryCombineFixedPointConvert(SDNode *N,
                                           TargetLowering::DAGCombinerInfo &DCI,
                                           SelectionDAG &DAG) {
  // Wait 'til after everything is legalized to try this. That way we have
  // legal vector types and such.
  if (DCI.isBeforeLegalizeOps())
    return SDValue();
  // Transform a scalar conversion of a value from a lane extract into a
  // lane extract of a vector conversion. E.g., from foo1 to foo2:
  // double foo1(int64x2_t a) { return vcvtd_n_f64_s64(a[1], 9); }
  // double foo2(int64x2_t a) { return vcvtq_n_f64_s64(a, 9)[1]; }
  //
  // The second form interacts better with instruction selection and the
  // register allocator to avoid cross-class register copies that aren't
  // coalescable due to a lane reference.

  // Check the operand and see if it originates from a lane extract.
  SDValue Op1 = N->getOperand(1);
  if (Op1.getOpcode() == ISD::EXTRACT_VECTOR_ELT) {
    // Yep, no additional predication needed. Perform the transform.
    SDValue IID = N->getOperand(0);
    SDValue Shift = N->getOperand(2);
    SDValue Vec = Op1.getOperand(0);
    SDValue Lane = Op1.getOperand(1);
    EVT ResTy = N->getValueType(0);
    EVT VecResTy;
    SDLoc DL(N);

    // The vector width should be 128 bits by the time we get here, even
    // if it started as 64 bits (the extract_vector handling will have
    // done so).
    assert(Vec.getValueType().getSizeInBits() == 128 &&
           "unexpected vector size on extract_vector_elt!");
    if (Vec.getValueType() == MVT::v4i32)
      VecResTy = MVT::v4f32;
    else if (Vec.getValueType() == MVT::v2i64)
      VecResTy = MVT::v2f64;
    else
      llvm_unreachable("unexpected vector type!");

    SDValue Convert =
        DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VecResTy, IID, Vec, Shift);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, ResTy, Convert, Lane);
  }
  return SDValue();
}

// AArch64 high-vector "long" operations are formed by performing the non-high
// version on an extract_subvector of each operand which gets the high half:
//
//  (longop2 LHS, RHS) == (longop (extract_high LHS), (extract_high RHS))
//
// However, there are cases which don't have an extract_high explicitly, but
// have another operation that can be made compatible with one for free. For
// example:
//
//  (dupv64 scalar) --> (extract_high (dup128 scalar))
//
// This routine does the actual conversion of such DUPs, once outer routines
// have determined that everything else is in order.
// It also supports immediate DUP-like nodes (MOVI/MVNi), which we can fold
// similarly here.
static SDValue tryExtendDUPToExtractHigh(SDValue N, SelectionDAG &DAG) {
  switch (N.getOpcode()) {
  case AArch64ISD::DUP:
  case AArch64ISD::DUPLANE8:
  case AArch64ISD::DUPLANE16:
  case AArch64ISD::DUPLANE32:
  case AArch64ISD::DUPLANE64:
  case AArch64ISD::MOVI:
  case AArch64ISD::MOVIshift:
  case AArch64ISD::MOVIedit:
  case AArch64ISD::MOVImsl:
  case AArch64ISD::MVNIshift:
  case AArch64ISD::MVNImsl:
    break;
  default:
    // FMOV could be supported, but isn't very useful, as it would only occur
    // if you passed a bitcast' floating point immediate to an eligible long
    // integer op (addl, smull, ...).
    return SDValue();
  }

  MVT NarrowTy = N.getSimpleValueType();
  if (!NarrowTy.is64BitVector())
    return SDValue();

  MVT ElementTy = NarrowTy.getVectorElementType();
  unsigned NumElems = NarrowTy.getVectorNumElements();
  MVT NewVT = MVT::getVectorVT(ElementTy, NumElems * 2);

  SDLoc dl(N);
  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, NarrowTy,
                     DAG.getNode(N->getOpcode(), dl, NewVT, N->ops()),
                     DAG.getConstant(NumElems, dl, MVT::i64));
}

static bool isEssentiallyExtractSubvector(SDValue N) {
  if (N.getOpcode() == ISD::EXTRACT_SUBVECTOR)
    return true;

  return N.getOpcode() == ISD::BITCAST &&
         N.getOperand(0).getOpcode() == ISD::EXTRACT_SUBVECTOR;
}

/// \brief Helper structure to keep track of ISD::SET_CC operands.
struct GenericSetCCInfo {
  const SDValue *Opnd0;
  const SDValue *Opnd1;
  ISD::CondCode CC;
};

/// \brief Helper structure to keep track of a SET_CC lowered into AArch64 code.
struct AArch64SetCCInfo {
  const SDValue *Cmp;
  AArch64CC::CondCode CC;
};

/// \brief Helper structure to keep track of SetCC information.
union SetCCInfo {
  GenericSetCCInfo Generic;
  AArch64SetCCInfo AArch64;
};

/// \brief Helper structure to be able to read SetCC information.  If set to
/// true, IsAArch64 field, Info is a AArch64SetCCInfo, otherwise Info is a
/// GenericSetCCInfo.
struct SetCCInfoAndKind {
  SetCCInfo Info;
  bool IsAArch64;
};

/// \brief Check whether or not \p Op is a SET_CC operation, either a generic or
/// an
/// AArch64 lowered one.
/// \p SetCCInfo is filled accordingly.
/// \post SetCCInfo is meanginfull only when this function returns true.
/// \return True when Op is a kind of SET_CC operation.
static bool isSetCC(SDValue Op, SetCCInfoAndKind &SetCCInfo) {
  // If this is a setcc, this is straight forward.
  if (Op.getOpcode() == ISD::SETCC) {
    SetCCInfo.Info.Generic.Opnd0 = &Op.getOperand(0);
    SetCCInfo.Info.Generic.Opnd1 = &Op.getOperand(1);
    SetCCInfo.Info.Generic.CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
    SetCCInfo.IsAArch64 = false;
    return true;
  }
  // Otherwise, check if this is a matching csel instruction.
  // In other words:
  // - csel 1, 0, cc
  // - csel 0, 1, !cc
  if (Op.getOpcode() != AArch64ISD::CSEL)
    return false;
  // Set the information about the operands.
  // TODO: we want the operands of the Cmp not the csel
  SetCCInfo.Info.AArch64.Cmp = &Op.getOperand(3);
  SetCCInfo.IsAArch64 = true;
  SetCCInfo.Info.AArch64.CC = static_cast<AArch64CC::CondCode>(
      cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue());

  // Check that the operands matches the constraints:
  // (1) Both operands must be constants.
  // (2) One must be 1 and the other must be 0.
  ConstantSDNode *TValue = dyn_cast<ConstantSDNode>(Op.getOperand(0));
  ConstantSDNode *FValue = dyn_cast<ConstantSDNode>(Op.getOperand(1));

  // Check (1).
  if (!TValue || !FValue)
    return false;

  // Check (2).
  if (!TValue->isOne()) {
    // Update the comparison when we are interested in !cc.
    std::swap(TValue, FValue);
    SetCCInfo.Info.AArch64.CC =
        AArch64CC::getInvertedCondCode(SetCCInfo.Info.AArch64.CC);
  }
  return TValue->isOne() && FValue->isNullValue();
}

// Returns true if Op is setcc or zext of setcc.
static bool isSetCCOrZExtSetCC(const SDValue& Op, SetCCInfoAndKind &Info) {
  if (isSetCC(Op, Info))
    return true;
  return ((Op.getOpcode() == ISD::ZERO_EXTEND) &&
    isSetCC(Op->getOperand(0), Info));
}

// The folding we want to perform is:
// (add x, [zext] (setcc cc ...) )
//   -->
// (csel x, (add x, 1), !cc ...)
//
// The latter will get matched to a CSINC instruction.
static SDValue performSetccAddFolding(SDNode *Op, SelectionDAG &DAG) {
  assert(Op && Op->getOpcode() == ISD::ADD && "Unexpected operation!");
  SDValue LHS = Op->getOperand(0);
  SDValue RHS = Op->getOperand(1);
  SetCCInfoAndKind InfoAndKind;

  // If neither operand is a SET_CC, give up.
  if (!isSetCCOrZExtSetCC(LHS, InfoAndKind)) {
    std::swap(LHS, RHS);
    if (!isSetCCOrZExtSetCC(LHS, InfoAndKind))
      return SDValue();
  }

  // FIXME: This could be generatized to work for FP comparisons.
  EVT CmpVT = InfoAndKind.IsAArch64
                  ? InfoAndKind.Info.AArch64.Cmp->getOperand(0).getValueType()
                  : InfoAndKind.Info.Generic.Opnd0->getValueType();
  if (CmpVT != MVT::i32 && CmpVT != MVT::i64)
    return SDValue();

  SDValue CCVal;
  SDValue Cmp;
  SDLoc dl(Op);
  if (InfoAndKind.IsAArch64) {
    CCVal = DAG.getConstant(
        AArch64CC::getInvertedCondCode(InfoAndKind.Info.AArch64.CC), dl,
        MVT::i32);
    Cmp = *InfoAndKind.Info.AArch64.Cmp;
  } else
    Cmp = getAArch64Cmp(*InfoAndKind.Info.Generic.Opnd0,
                      *InfoAndKind.Info.Generic.Opnd1,
                      ISD::getSetCCInverse(InfoAndKind.Info.Generic.CC, true),
                      CCVal, DAG, dl);

  EVT VT = Op->getValueType(0);
  LHS = DAG.getNode(ISD::ADD, dl, VT, RHS, DAG.getConstant(1, dl, VT));
  return DAG.getNode(AArch64ISD::CSEL, dl, VT, RHS, LHS, CCVal, Cmp);
}

// The basic add/sub long vector instructions have variants with "2" on the end
// which act on the high-half of their inputs. They are normally matched by
// patterns like:
//
// (add (zeroext (extract_high LHS)),
//      (zeroext (extract_high RHS)))
// -> uaddl2 vD, vN, vM
//
// However, if one of the extracts is something like a duplicate, this
// instruction can still be used profitably. This function puts the DAG into a
// more appropriate form for those patterns to trigger.
static SDValue performAddSubLongCombine(SDNode *N,
                                        TargetLowering::DAGCombinerInfo &DCI,
                                        SelectionDAG &DAG) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  MVT VT = N->getSimpleValueType(0);
  if (!VT.is128BitVector()) {
    if (N->getOpcode() == ISD::ADD)
      return performSetccAddFolding(N, DAG);
    return SDValue();
  }

  // Make sure both branches are extended in the same way.
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  if ((LHS.getOpcode() != ISD::ZERO_EXTEND &&
       LHS.getOpcode() != ISD::SIGN_EXTEND) ||
      LHS.getOpcode() != RHS.getOpcode())
    return SDValue();

  unsigned ExtType = LHS.getOpcode();

  // It's not worth doing if at least one of the inputs isn't already an
  // extract, but we don't know which it'll be so we have to try both.
  if (isEssentiallyExtractSubvector(LHS.getOperand(0))) {
    RHS = tryExtendDUPToExtractHigh(RHS.getOperand(0), DAG);
    if (!RHS.getNode())
      return SDValue();

    RHS = DAG.getNode(ExtType, SDLoc(N), VT, RHS);
  } else if (isEssentiallyExtractSubvector(RHS.getOperand(0))) {
    LHS = tryExtendDUPToExtractHigh(LHS.getOperand(0), DAG);
    if (!LHS.getNode())
      return SDValue();

    LHS = DAG.getNode(ExtType, SDLoc(N), VT, LHS);
  }

  return DAG.getNode(N->getOpcode(), SDLoc(N), VT, LHS, RHS);
}

// Massage DAGs which we can use the high-half "long" operations on into
// something isel will recognize better. E.g.
//
// (aarch64_neon_umull (extract_high vec) (dupv64 scalar)) -->
//   (aarch64_neon_umull (extract_high (v2i64 vec)))
//                     (extract_high (v2i64 (dup128 scalar)))))
//
static SDValue tryCombineLongOpWithDup(SDNode *N,
                                       TargetLowering::DAGCombinerInfo &DCI,
                                       SelectionDAG &DAG) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  bool IsIntrinsic = N->getOpcode() == ISD::INTRINSIC_WO_CHAIN;
  SDValue LHS = N->getOperand(IsIntrinsic ? 1 : 0);
  SDValue RHS = N->getOperand(IsIntrinsic ? 2 : 1);
  assert(LHS.getValueType().is64BitVector() &&
         RHS.getValueType().is64BitVector() &&
         "unexpected shape for long operation");

  // Either node could be a DUP, but it's not worth doing both of them (you'd
  // just as well use the non-high version) so look for a corresponding extract
  // operation on the other "wing".
  if (isEssentiallyExtractSubvector(LHS)) {
    RHS = tryExtendDUPToExtractHigh(RHS, DAG);
    if (!RHS.getNode())
      return SDValue();
  } else if (isEssentiallyExtractSubvector(RHS)) {
    LHS = tryExtendDUPToExtractHigh(LHS, DAG);
    if (!LHS.getNode())
      return SDValue();
  }

  // N could either be an intrinsic or a sabsdiff/uabsdiff node.
  if (IsIntrinsic)
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, SDLoc(N), N->getValueType(0),
                       N->getOperand(0), LHS, RHS);
  else
    return DAG.getNode(N->getOpcode(), SDLoc(N), N->getValueType(0),
                       LHS, RHS);
}

static SDValue tryCombineShiftImm(unsigned IID, SDNode *N, SelectionDAG &DAG) {
  MVT ElemTy = N->getSimpleValueType(0).getScalarType();
  unsigned ElemBits = ElemTy.getSizeInBits();

  int64_t ShiftAmount;
  if (BuildVectorSDNode *BVN = dyn_cast<BuildVectorSDNode>(N->getOperand(2))) {
    APInt SplatValue, SplatUndef;
    unsigned SplatBitSize;
    bool HasAnyUndefs;
    if (!BVN->isConstantSplat(SplatValue, SplatUndef, SplatBitSize,
                              HasAnyUndefs, ElemBits) ||
        SplatBitSize != ElemBits)
      return SDValue();

    ShiftAmount = SplatValue.getSExtValue();
  } else if (ConstantSDNode *CVN = dyn_cast<ConstantSDNode>(N->getOperand(2))) {
    ShiftAmount = CVN->getSExtValue();
  } else
    return SDValue();

  unsigned Opcode;
  bool IsRightShift;
  switch (IID) {
  default:
    llvm_unreachable("Unknown shift intrinsic");
  case Intrinsic::aarch64_neon_sqshl:
    Opcode = AArch64ISD::SQSHL_I;
    IsRightShift = false;
    break;
  case Intrinsic::aarch64_neon_uqshl:
    Opcode = AArch64ISD::UQSHL_I;
    IsRightShift = false;
    break;
  case Intrinsic::aarch64_neon_srshl:
    Opcode = AArch64ISD::SRSHR_I;
    IsRightShift = true;
    break;
  case Intrinsic::aarch64_neon_urshl:
    Opcode = AArch64ISD::URSHR_I;
    IsRightShift = true;
    break;
  case Intrinsic::aarch64_neon_sqshlu:
    Opcode = AArch64ISD::SQSHLU_I;
    IsRightShift = false;
    break;
  }

  if (IsRightShift && ShiftAmount <= -1 && ShiftAmount >= -(int)ElemBits) {
    SDLoc dl(N);
    return DAG.getNode(Opcode, dl, N->getValueType(0), N->getOperand(1),
                       DAG.getConstant(-ShiftAmount, dl, MVT::i32));
  } else if (!IsRightShift && ShiftAmount >= 0 && ShiftAmount < ElemBits) {
    SDLoc dl(N);
    return DAG.getNode(Opcode, dl, N->getValueType(0), N->getOperand(1),
                       DAG.getConstant(ShiftAmount, dl, MVT::i32));
  }

  return SDValue();
}

// The CRC32[BH] instructions ignore the high bits of their data operand. Since
// the intrinsics must be legal and take an i32, this means there's almost
// certainly going to be a zext in the DAG which we can eliminate.
static SDValue tryCombineCRC32(unsigned Mask, SDNode *N, SelectionDAG &DAG) {
  SDValue AndN = N->getOperand(2);
  if (AndN.getOpcode() != ISD::AND)
    return SDValue();

  ConstantSDNode *CMask = dyn_cast<ConstantSDNode>(AndN.getOperand(1));
  if (!CMask || CMask->getZExtValue() != Mask)
    return SDValue();

  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, SDLoc(N), MVT::i32,
                     N->getOperand(0), N->getOperand(1), AndN.getOperand(0));
}

static SDValue combineAcrossLanesIntrinsic(unsigned Opc, SDNode *N,
                                           SelectionDAG &DAG) {
  SDLoc dl(N);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, N->getValueType(0),
                     DAG.getNode(Opc, dl,
                                 N->getOperand(1).getSimpleValueType(),
                                 N->getOperand(1)),
                     DAG.getConstant(0, dl, MVT::i64));
}

static SDValue performIntrinsicCombine(SDNode *N,
                                       TargetLowering::DAGCombinerInfo &DCI,
                                       const AArch64Subtarget *Subtarget) {
  SelectionDAG &DAG = DCI.DAG;
  unsigned IID = getIntrinsicID(N);
  switch (IID) {
  default:
    break;
  case Intrinsic::aarch64_neon_vcvtfxs2fp:
  case Intrinsic::aarch64_neon_vcvtfxu2fp:
    return tryCombineFixedPointConvert(N, DCI, DAG);
  case Intrinsic::aarch64_neon_saddv:
    return combineAcrossLanesIntrinsic(AArch64ISD::SADDV, N, DAG);
  case Intrinsic::aarch64_neon_uaddv:
    return combineAcrossLanesIntrinsic(AArch64ISD::UADDV, N, DAG);
  case Intrinsic::aarch64_neon_sminv:
    return combineAcrossLanesIntrinsic(AArch64ISD::SMINV, N, DAG);
  case Intrinsic::aarch64_neon_uminv:
    return combineAcrossLanesIntrinsic(AArch64ISD::UMINV, N, DAG);
  case Intrinsic::aarch64_neon_smaxv:
    return combineAcrossLanesIntrinsic(AArch64ISD::SMAXV, N, DAG);
  case Intrinsic::aarch64_neon_umaxv:
    return combineAcrossLanesIntrinsic(AArch64ISD::UMAXV, N, DAG);
  case Intrinsic::aarch64_neon_fmax:
    return DAG.getNode(ISD::FMAXNAN, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2));
  case Intrinsic::aarch64_neon_fmin:
    return DAG.getNode(ISD::FMINNAN, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2));
  case Intrinsic::aarch64_neon_sabd:
    return DAG.getNode(ISD::SABSDIFF, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2));
  case Intrinsic::aarch64_neon_uabd:
    return DAG.getNode(ISD::UABSDIFF, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2));
  case Intrinsic::aarch64_neon_fmaxnm:
    return DAG.getNode(ISD::FMAXNUM, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2));
  case Intrinsic::aarch64_neon_fminnm:
    return DAG.getNode(ISD::FMINNUM, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2));
  case Intrinsic::aarch64_neon_smull:
  case Intrinsic::aarch64_neon_umull:
  case Intrinsic::aarch64_neon_pmull:
  case Intrinsic::aarch64_neon_sqdmull:
    return tryCombineLongOpWithDup(N, DCI, DAG);
  case Intrinsic::aarch64_neon_sqshl:
  case Intrinsic::aarch64_neon_uqshl:
  case Intrinsic::aarch64_neon_sqshlu:
  case Intrinsic::aarch64_neon_srshl:
  case Intrinsic::aarch64_neon_urshl:
    return tryCombineShiftImm(IID, N, DAG);
  case Intrinsic::aarch64_crc32b:
  case Intrinsic::aarch64_crc32cb:
    return tryCombineCRC32(0xff, N, DAG);
  case Intrinsic::aarch64_crc32h:
  case Intrinsic::aarch64_crc32ch:
    return tryCombineCRC32(0xffff, N, DAG);
  }
  return SDValue();
}

static SDValue performExtendCombine(SDNode *N,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    SelectionDAG &DAG) {
  // If we see something like (zext (sabd (extract_high ...), (DUP ...))) then
  // we can convert that DUP into another extract_high (of a bigger DUP), which
  // helps the backend to decide that an sabdl2 would be useful, saving a real
  // extract_high operation.
  if (!DCI.isBeforeLegalizeOps() && N->getOpcode() == ISD::ZERO_EXTEND &&
      (N->getOperand(0).getOpcode() == ISD::SABSDIFF ||
       N->getOperand(0).getOpcode() == ISD::UABSDIFF)) {
    SDNode *ABDNode = N->getOperand(0).getNode();
    SDValue NewABD = tryCombineLongOpWithDup(ABDNode, DCI, DAG);
    if (!NewABD.getNode())
      return SDValue();

    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), N->getValueType(0),
                       NewABD);
  }

  // This is effectively a custom type legalization for AArch64.
  //
  // Type legalization will split an extend of a small, legal, type to a larger
  // illegal type by first splitting the destination type, often creating
  // illegal source types, which then get legalized in isel-confusing ways,
  // leading to really terrible codegen. E.g.,
  //   %result = v8i32 sext v8i8 %value
  // becomes
  //   %losrc = extract_subreg %value, ...
  //   %hisrc = extract_subreg %value, ...
  //   %lo = v4i32 sext v4i8 %losrc
  //   %hi = v4i32 sext v4i8 %hisrc
  // Things go rapidly downhill from there.
  //
  // For AArch64, the [sz]ext vector instructions can only go up one element
  // size, so we can, e.g., extend from i8 to i16, but to go from i8 to i32
  // take two instructions.
  //
  // This implies that the most efficient way to do the extend from v8i8
  // to two v4i32 values is to first extend the v8i8 to v8i16, then do
  // the normal splitting to happen for the v8i16->v8i32.

  // This is pre-legalization to catch some cases where the default
  // type legalization will create ill-tempered code.
  if (!DCI.isBeforeLegalizeOps())
    return SDValue();

  // We're only interested in cleaning things up for non-legal vector types
  // here. If both the source and destination are legal, things will just
  // work naturally without any fiddling.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT ResVT = N->getValueType(0);
  if (!ResVT.isVector() || TLI.isTypeLegal(ResVT))
    return SDValue();
  // If the vector type isn't a simple VT, it's beyond the scope of what
  // we're  worried about here. Let legalization do its thing and hope for
  // the best.
  SDValue Src = N->getOperand(0);
  EVT SrcVT = Src->getValueType(0);
  if (!ResVT.isSimple() || !SrcVT.isSimple())
    return SDValue();

  // If the source VT is a 64-bit vector, we can play games and get the
  // better results we want.
  if (SrcVT.getSizeInBits() != 64)
    return SDValue();

  unsigned SrcEltSize = SrcVT.getVectorElementType().getSizeInBits();
  unsigned ElementCount = SrcVT.getVectorNumElements();
  SrcVT = MVT::getVectorVT(MVT::getIntegerVT(SrcEltSize * 2), ElementCount);
  SDLoc DL(N);
  Src = DAG.getNode(N->getOpcode(), DL, SrcVT, Src);

  // Now split the rest of the operation into two halves, each with a 64
  // bit source.
  EVT LoVT, HiVT;
  SDValue Lo, Hi;
  unsigned NumElements = ResVT.getVectorNumElements();
  assert(!(NumElements & 1) && "Splitting vector, but not in half!");
  LoVT = HiVT = EVT::getVectorVT(*DAG.getContext(),
                                 ResVT.getVectorElementType(), NumElements / 2);

  EVT InNVT = EVT::getVectorVT(*DAG.getContext(), SrcVT.getVectorElementType(),
                               LoVT.getVectorNumElements());
  Lo = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, InNVT, Src,
                   DAG.getConstant(0, DL, MVT::i64));
  Hi = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, InNVT, Src,
                   DAG.getConstant(InNVT.getVectorNumElements(), DL, MVT::i64));
  Lo = DAG.getNode(N->getOpcode(), DL, LoVT, Lo);
  Hi = DAG.getNode(N->getOpcode(), DL, HiVT, Hi);

  // Now combine the parts back together so we still have a single result
  // like the combiner expects.
  return DAG.getNode(ISD::CONCAT_VECTORS, DL, ResVT, Lo, Hi);
}

/// Replace a splat of a scalar to a vector store by scalar stores of the scalar
/// value. The load store optimizer pass will merge them to store pair stores.
/// This has better performance than a splat of the scalar followed by a split
/// vector store. Even if the stores are not merged it is four stores vs a dup,
/// followed by an ext.b and two stores.
static SDValue replaceSplatVectorStore(SelectionDAG &DAG, StoreSDNode *St) {
  SDValue StVal = St->getValue();
  EVT VT = StVal.getValueType();

  // Don't replace floating point stores, they possibly won't be transformed to
  // stp because of the store pair suppress pass.
  if (VT.isFloatingPoint())
    return SDValue();

  // Check for insert vector elements.
  if (StVal.getOpcode() != ISD::INSERT_VECTOR_ELT)
    return SDValue();

  // We can express a splat as store pair(s) for 2 or 4 elements.
  unsigned NumVecElts = VT.getVectorNumElements();
  if (NumVecElts != 4 && NumVecElts != 2)
    return SDValue();
  SDValue SplatVal = StVal.getOperand(1);
  unsigned RemainInsertElts = NumVecElts - 1;

  // Check that this is a splat.
  while (--RemainInsertElts) {
    SDValue NextInsertElt = StVal.getOperand(0);
    if (NextInsertElt.getOpcode() != ISD::INSERT_VECTOR_ELT)
      return SDValue();
    if (NextInsertElt.getOperand(1) != SplatVal)
      return SDValue();
    StVal = NextInsertElt;
  }
  unsigned OrigAlignment = St->getAlignment();
  unsigned EltOffset = NumVecElts == 4 ? 4 : 8;
  unsigned Alignment = std::min(OrigAlignment, EltOffset);

  // Create scalar stores. This is at least as good as the code sequence for a
  // split unaligned store which is a dup.s, ext.b, and two stores.
  // Most of the time the three stores should be replaced by store pair
  // instructions (stp).
  SDLoc DL(St);
  SDValue BasePtr = St->getBasePtr();
  SDValue NewST1 =
      DAG.getStore(St->getChain(), DL, SplatVal, BasePtr, St->getPointerInfo(),
                   St->isVolatile(), St->isNonTemporal(), St->getAlignment());

  unsigned Offset = EltOffset;
  while (--NumVecElts) {
    SDValue OffsetPtr = DAG.getNode(ISD::ADD, DL, MVT::i64, BasePtr,
                                    DAG.getConstant(Offset, DL, MVT::i64));
    NewST1 = DAG.getStore(NewST1.getValue(0), DL, SplatVal, OffsetPtr,
                          St->getPointerInfo(), St->isVolatile(),
                          St->isNonTemporal(), Alignment);
    Offset += EltOffset;
  }
  return NewST1;
}

static SDValue split16BStores(SDNode *N, TargetLowering::DAGCombinerInfo &DCI,
                              SelectionDAG &DAG,
                              const AArch64Subtarget *Subtarget) {
  if (!DCI.isBeforeLegalize())
    return SDValue();

  StoreSDNode *S = cast<StoreSDNode>(N);
  if (S->isVolatile())
    return SDValue();

  // FIXME: The logic for deciding if an unaligned store should be split should
  // be included in TLI.allowsMisalignedMemoryAccesses(), and there should be
  // a call to that function here.

  // Cyclone has bad performance on unaligned 16B stores when crossing line and
  // page boundaries. We want to split such stores.
  if (!Subtarget->isCyclone())
    return SDValue();

  // Don't split at -Oz.
  if (DAG.getMachineFunction().getFunction()->optForMinSize())
    return SDValue();

  SDValue StVal = S->getValue();
  EVT VT = StVal.getValueType();

  // Don't split v2i64 vectors. Memcpy lowering produces those and splitting
  // those up regresses performance on micro-benchmarks and olden/bh.
  if (!VT.isVector() || VT.getVectorNumElements() < 2 || VT == MVT::v2i64)
    return SDValue();

  // Split unaligned 16B stores. They are terrible for performance.
  // Don't split stores with alignment of 1 or 2. Code that uses clang vector
  // extensions can use this to mark that it does not want splitting to happen
  // (by underspecifying alignment to be 1 or 2). Furthermore, the chance of
  // eliminating alignment hazards is only 1 in 8 for alignment of 2.
  if (VT.getSizeInBits() != 128 || S->getAlignment() >= 16 ||
      S->getAlignment() <= 2)
    return SDValue();

  // If we get a splat of a scalar convert this vector store to a store of
  // scalars. They will be merged into store pairs thereby removing two
  // instructions.
  if (SDValue ReplacedSplat = replaceSplatVectorStore(DAG, S))
    return ReplacedSplat;

  SDLoc DL(S);
  unsigned NumElts = VT.getVectorNumElements() / 2;
  // Split VT into two.
  EVT HalfVT =
      EVT::getVectorVT(*DAG.getContext(), VT.getVectorElementType(), NumElts);
  SDValue SubVector0 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, HalfVT, StVal,
                                   DAG.getConstant(0, DL, MVT::i64));
  SDValue SubVector1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, HalfVT, StVal,
                                   DAG.getConstant(NumElts, DL, MVT::i64));
  SDValue BasePtr = S->getBasePtr();
  SDValue NewST1 =
      DAG.getStore(S->getChain(), DL, SubVector0, BasePtr, S->getPointerInfo(),
                   S->isVolatile(), S->isNonTemporal(), S->getAlignment());
  SDValue OffsetPtr = DAG.getNode(ISD::ADD, DL, MVT::i64, BasePtr,
                                  DAG.getConstant(8, DL, MVT::i64));
  return DAG.getStore(NewST1.getValue(0), DL, SubVector1, OffsetPtr,
                      S->getPointerInfo(), S->isVolatile(), S->isNonTemporal(),
                      S->getAlignment());
}

/// Target-specific DAG combine function for post-increment LD1 (lane) and
/// post-increment LD1R.
static SDValue performPostLD1Combine(SDNode *N,
                                     TargetLowering::DAGCombinerInfo &DCI,
                                     bool IsLaneOp) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  unsigned LoadIdx = IsLaneOp ? 1 : 0;
  SDNode *LD = N->getOperand(LoadIdx).getNode();
  // If it is not LOAD, can not do such combine.
  if (LD->getOpcode() != ISD::LOAD)
    return SDValue();

  LoadSDNode *LoadSDN = cast<LoadSDNode>(LD);
  EVT MemVT = LoadSDN->getMemoryVT();
  // Check if memory operand is the same type as the vector element.
  if (MemVT != VT.getVectorElementType())
    return SDValue();

  // Check if there are other uses. If so, do not combine as it will introduce
  // an extra load.
  for (SDNode::use_iterator UI = LD->use_begin(), UE = LD->use_end(); UI != UE;
       ++UI) {
    if (UI.getUse().getResNo() == 1) // Ignore uses of the chain result.
      continue;
    if (*UI != N)
      return SDValue();
  }

  SDValue Addr = LD->getOperand(1);
  SDValue Vector = N->getOperand(0);
  // Search for a use of the address operand that is an increment.
  for (SDNode::use_iterator UI = Addr.getNode()->use_begin(), UE =
       Addr.getNode()->use_end(); UI != UE; ++UI) {
    SDNode *User = *UI;
    if (User->getOpcode() != ISD::ADD
        || UI.getUse().getResNo() != Addr.getResNo())
      continue;

    // Check that the add is independent of the load.  Otherwise, folding it
    // would create a cycle.
    if (User->isPredecessorOf(LD) || LD->isPredecessorOf(User))
      continue;
    // Also check that add is not used in the vector operand.  This would also
    // create a cycle.
    if (User->isPredecessorOf(Vector.getNode()))
      continue;

    // If the increment is a constant, it must match the memory ref size.
    SDValue Inc = User->getOperand(User->getOperand(0) == Addr ? 1 : 0);
    if (ConstantSDNode *CInc = dyn_cast<ConstantSDNode>(Inc.getNode())) {
      uint32_t IncVal = CInc->getZExtValue();
      unsigned NumBytes = VT.getScalarSizeInBits() / 8;
      if (IncVal != NumBytes)
        continue;
      Inc = DAG.getRegister(AArch64::XZR, MVT::i64);
    }

    // Finally, check that the vector doesn't depend on the load.
    // Again, this would create a cycle.
    // The load depending on the vector is fine, as that's the case for the
    // LD1*post we'll eventually generate anyway.
    if (LoadSDN->isPredecessorOf(Vector.getNode()))
      continue;

    SmallVector<SDValue, 8> Ops;
    Ops.push_back(LD->getOperand(0));  // Chain
    if (IsLaneOp) {
      Ops.push_back(Vector);           // The vector to be inserted
      Ops.push_back(N->getOperand(2)); // The lane to be inserted in the vector
    }
    Ops.push_back(Addr);
    Ops.push_back(Inc);

    EVT Tys[3] = { VT, MVT::i64, MVT::Other };
    SDVTList SDTys = DAG.getVTList(Tys);
    unsigned NewOp = IsLaneOp ? AArch64ISD::LD1LANEpost : AArch64ISD::LD1DUPpost;
    SDValue UpdN = DAG.getMemIntrinsicNode(NewOp, SDLoc(N), SDTys, Ops,
                                           MemVT,
                                           LoadSDN->getMemOperand());

    // Update the uses.
    SmallVector<SDValue, 2> NewResults;
    NewResults.push_back(SDValue(LD, 0));             // The result of load
    NewResults.push_back(SDValue(UpdN.getNode(), 2)); // Chain
    DCI.CombineTo(LD, NewResults);
    DCI.CombineTo(N, SDValue(UpdN.getNode(), 0));     // Dup/Inserted Result
    DCI.CombineTo(User, SDValue(UpdN.getNode(), 1));  // Write back register

    break;
  }
  return SDValue();
}

/// Simplify \Addr given that the top byte of it is ignored by HW during
/// address translation.
static bool performTBISimplification(SDValue Addr,
                                     TargetLowering::DAGCombinerInfo &DCI,
                                     SelectionDAG &DAG) {
  APInt DemandedMask = APInt::getLowBitsSet(64, 56);
  APInt KnownZero, KnownOne;
  TargetLowering::TargetLoweringOpt TLO(DAG, DCI.isBeforeLegalize(),
                                        DCI.isBeforeLegalizeOps());
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (TLI.SimplifyDemandedBits(Addr, DemandedMask, KnownZero, KnownOne, TLO)) {
    DCI.CommitTargetLoweringOpt(TLO);
    return true;
  }
  return false;
}

static SDValue performSTORECombine(SDNode *N,
                                   TargetLowering::DAGCombinerInfo &DCI,
                                   SelectionDAG &DAG,
                                   const AArch64Subtarget *Subtarget) {
  SDValue Split = split16BStores(N, DCI, DAG, Subtarget);
  if (Split.getNode())
    return Split;

  if (Subtarget->supportsAddressTopByteIgnored() &&
      performTBISimplification(N->getOperand(2), DCI, DAG))
    return SDValue(N, 0);

  return SDValue();
}

  /// This function handles the log2-shuffle pattern produced by the
/// LoopVectorizer for the across vector reduction. It consists of
/// log2(NumVectorElements) steps and, in each step, 2^(s) elements
/// are reduced, where s is an induction variable from 0 to
/// log2(NumVectorElements).
static SDValue tryMatchAcrossLaneShuffleForReduction(SDNode *N, SDValue OpV,
                                                     unsigned Op,
                                                     SelectionDAG &DAG) {
  EVT VTy = OpV->getOperand(0).getValueType();
  if (!VTy.isVector())
    return SDValue();

  int NumVecElts = VTy.getVectorNumElements();
  if (Op == ISD::FMAXNUM || Op == ISD::FMINNUM) {
    if (NumVecElts != 4)
      return SDValue();
  } else {
    if (NumVecElts != 4 && NumVecElts != 8 && NumVecElts != 16)
      return SDValue();
  }

  int NumExpectedSteps = APInt(8, NumVecElts).logBase2();
  SDValue PreOp = OpV;
  // Iterate over each step of the across vector reduction.
  for (int CurStep = 0; CurStep != NumExpectedSteps; ++CurStep) {
    SDValue CurOp = PreOp.getOperand(0);
    SDValue Shuffle = PreOp.getOperand(1);
    if (Shuffle.getOpcode() != ISD::VECTOR_SHUFFLE) {
      // Try to swap the 1st and 2nd operand as add and min/max instructions
      // are commutative.
      CurOp = PreOp.getOperand(1);
      Shuffle = PreOp.getOperand(0);
      if (Shuffle.getOpcode() != ISD::VECTOR_SHUFFLE)
        return SDValue();
    }

    // Check if the input vector is fed by the operator we want to handle,
    // except the last step; the very first input vector is not necessarily
    // the same operator we are handling.
    if (CurOp.getOpcode() != Op && (CurStep != (NumExpectedSteps - 1)))
      return SDValue();

    // Check if it forms one step of the across vector reduction.
    // E.g.,
    //   %cur = add %1, %0
    //   %shuffle = vector_shuffle %cur, <2, 3, u, u>
    //   %pre = add %cur, %shuffle
    if (Shuffle.getOperand(0) != CurOp)
      return SDValue();

    int NumMaskElts = 1 << CurStep;
    ArrayRef<int> Mask = cast<ShuffleVectorSDNode>(Shuffle)->getMask();
    // Check mask values in each step.
    // We expect the shuffle mask in each step follows a specific pattern
    // denoted here by the <M, U> form, where M is a sequence of integers
    // starting from NumMaskElts, increasing by 1, and the number integers
    // in M should be NumMaskElts. U is a sequence of UNDEFs and the number
    // of undef in U should be NumVecElts - NumMaskElts.
    // E.g., for <8 x i16>, mask values in each step should be :
    //   step 0 : <1,u,u,u,u,u,u,u>
    //   step 1 : <2,3,u,u,u,u,u,u>
    //   step 2 : <4,5,6,7,u,u,u,u>
    for (int i = 0; i < NumVecElts; ++i)
      if ((i < NumMaskElts && Mask[i] != (NumMaskElts + i)) ||
          (i >= NumMaskElts && !(Mask[i] < 0)))
        return SDValue();

    PreOp = CurOp;
  }
  unsigned Opcode;
  bool IsIntrinsic = false;

  switch (Op) {
  default:
    llvm_unreachable("Unexpected operator for across vector reduction");
  case ISD::ADD:
    Opcode = AArch64ISD::UADDV;
    break;
  case ISD::SMAX:
    Opcode = AArch64ISD::SMAXV;
    break;
  case ISD::UMAX:
    Opcode = AArch64ISD::UMAXV;
    break;
  case ISD::SMIN:
    Opcode = AArch64ISD::SMINV;
    break;
  case ISD::UMIN:
    Opcode = AArch64ISD::UMINV;
    break;
  case ISD::FMAXNUM:
    Opcode = Intrinsic::aarch64_neon_fmaxnmv;
    IsIntrinsic = true;
    break;
  case ISD::FMINNUM:
    Opcode = Intrinsic::aarch64_neon_fminnmv;
    IsIntrinsic = true;
    break;
  }
  SDLoc DL(N);

  return IsIntrinsic
             ? DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, N->getValueType(0),
                           DAG.getConstant(Opcode, DL, MVT::i32), PreOp)
             : DAG.getNode(
                   ISD::EXTRACT_VECTOR_ELT, DL, N->getValueType(0),
                   DAG.getNode(Opcode, DL, PreOp.getSimpleValueType(), PreOp),
                   DAG.getConstant(0, DL, MVT::i64));
}

/// Target-specific DAG combine for the across vector min/max reductions.
/// This function specifically handles the final clean-up step of the vector
/// min/max reductions produced by the LoopVectorizer. It is the log2-shuffle
/// pattern, which narrows down and finds the final min/max value from all
/// elements of the vector.
/// For example, for a <16 x i8> vector :
///   svn0 = vector_shuffle %0, undef<8,9,10,11,12,13,14,15,u,u,u,u,u,u,u,u>
///   %smax0 = smax %arr, svn0
///   %svn1 = vector_shuffle %smax0, undef<4,5,6,7,u,u,u,u,u,u,u,u,u,u,u,u>
///   %smax1 = smax %smax0, %svn1
///   %svn2 = vector_shuffle %smax1, undef<2,3,u,u,u,u,u,u,u,u,u,u,u,u,u,u>
///   %smax2 = smax %smax1, svn2
///   %svn3 = vector_shuffle %smax2, undef<1,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u>
///   %sc = setcc %smax2, %svn3, gt
///   %n0 = extract_vector_elt %sc, #0
///   %n1 = extract_vector_elt %smax2, #0
///   %n2 = extract_vector_elt $smax2, #1
///   %result = select %n0, %n1, n2
///     becomes :
///   %1 = smaxv %0
///   %result = extract_vector_elt %1, 0
static SDValue
performAcrossLaneMinMaxReductionCombine(SDNode *N, SelectionDAG &DAG,
                                        const AArch64Subtarget *Subtarget) {
  if (!Subtarget->hasNEON())
    return SDValue();

  SDValue N0 = N->getOperand(0);
  SDValue IfTrue = N->getOperand(1);
  SDValue IfFalse = N->getOperand(2);

  // Check if the SELECT merges up the final result of the min/max
  // from a vector.
  if (N0.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
      IfTrue.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
      IfFalse.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
    return SDValue();

  // Expect N0 is fed by SETCC.
  SDValue SetCC = N0.getOperand(0);
  EVT SetCCVT = SetCC.getValueType();
  if (SetCC.getOpcode() != ISD::SETCC || !SetCCVT.isVector() ||
      SetCCVT.getVectorElementType() != MVT::i1)
    return SDValue();

  SDValue VectorOp = SetCC.getOperand(0);
  unsigned Op = VectorOp->getOpcode();
  // Check if the input vector is fed by the operator we want to handle.
  if (Op != ISD::SMAX && Op != ISD::UMAX && Op != ISD::SMIN &&
      Op != ISD::UMIN && Op != ISD::FMAXNUM && Op != ISD::FMINNUM)
    return SDValue();

  EVT VTy = VectorOp.getValueType();
  if (!VTy.isVector())
    return SDValue();

  if (VTy.getSizeInBits() < 64)
    return SDValue();

  EVT EltTy = VTy.getVectorElementType();
  if (Op == ISD::FMAXNUM || Op == ISD::FMINNUM) {
    if (EltTy != MVT::f32)
      return SDValue();
  } else {
    if (EltTy != MVT::i32 && EltTy != MVT::i16 && EltTy != MVT::i8)
      return SDValue();
  }

  // Check if extracting from the same vector.
  // For example,
  //   %sc = setcc %vector, %svn1, gt
  //   %n0 = extract_vector_elt %sc, #0
  //   %n1 = extract_vector_elt %vector, #0
  //   %n2 = extract_vector_elt $vector, #1
  if (!(VectorOp == IfTrue->getOperand(0) &&
        VectorOp == IfFalse->getOperand(0)))
    return SDValue();

  // Check if the condition code is matched with the operator type.
  ISD::CondCode CC = cast<CondCodeSDNode>(SetCC->getOperand(2))->get();
  if ((Op == ISD::SMAX && CC != ISD::SETGT && CC != ISD::SETGE) ||
      (Op == ISD::UMAX && CC != ISD::SETUGT && CC != ISD::SETUGE) ||
      (Op == ISD::SMIN && CC != ISD::SETLT && CC != ISD::SETLE) ||
      (Op == ISD::UMIN && CC != ISD::SETULT && CC != ISD::SETULE) ||
      (Op == ISD::FMAXNUM && CC != ISD::SETOGT && CC != ISD::SETOGE &&
       CC != ISD::SETUGT && CC != ISD::SETUGE && CC != ISD::SETGT &&
       CC != ISD::SETGE) ||
      (Op == ISD::FMINNUM && CC != ISD::SETOLT && CC != ISD::SETOLE &&
       CC != ISD::SETULT && CC != ISD::SETULE && CC != ISD::SETLT &&
       CC != ISD::SETLE))
    return SDValue();

  // Expect to check only lane 0 from the vector SETCC.
  if (!isa<ConstantSDNode>(N0.getOperand(1)) ||
      cast<ConstantSDNode>(N0.getOperand(1))->getZExtValue() != 0)
    return SDValue();

  // Expect to extract the true value from lane 0.
  if (!isa<ConstantSDNode>(IfTrue.getOperand(1)) ||
      cast<ConstantSDNode>(IfTrue.getOperand(1))->getZExtValue() != 0)
    return SDValue();

  // Expect to extract the false value from lane 1.
  if (!isa<ConstantSDNode>(IfFalse.getOperand(1)) ||
      cast<ConstantSDNode>(IfFalse.getOperand(1))->getZExtValue() != 1)
    return SDValue();

  return tryMatchAcrossLaneShuffleForReduction(N, SetCC, Op, DAG);
}

/// Target-specific DAG combine for the across vector add reduction.
/// This function specifically handles the final clean-up step of the vector
/// add reduction produced by the LoopVectorizer. It is the log2-shuffle
/// pattern, which adds all elements of a vector together.
/// For example, for a <4 x i32> vector :
///   %1 = vector_shuffle %0, <2,3,u,u>
///   %2 = add %0, %1
///   %3 = vector_shuffle %2, <1,u,u,u>
///   %4 = add %2, %3
///   %result = extract_vector_elt %4, 0
/// becomes :
///   %0 = uaddv %0
///   %result = extract_vector_elt %0, 0
static SDValue
performAcrossLaneAddReductionCombine(SDNode *N, SelectionDAG &DAG,
                                     const AArch64Subtarget *Subtarget) {
  if (!Subtarget->hasNEON())
    return SDValue();
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // Check if the input vector is fed by the ADD.
  if (N0->getOpcode() != ISD::ADD)
    return SDValue();

  // The vector extract idx must constant zero because we only expect the final
  // result of the reduction is placed in lane 0.
  if (!isa<ConstantSDNode>(N1) || cast<ConstantSDNode>(N1)->getZExtValue() != 0)
    return SDValue();

  EVT VTy = N0.getValueType();
  if (!VTy.isVector())
    return SDValue();

  EVT EltTy = VTy.getVectorElementType();
  if (EltTy != MVT::i32 && EltTy != MVT::i16 && EltTy != MVT::i8)
    return SDValue();

  if (VTy.getSizeInBits() < 64)
    return SDValue();

  return tryMatchAcrossLaneShuffleForReduction(N, N0, ISD::ADD, DAG);
}

/// Target-specific DAG combine function for NEON load/store intrinsics
/// to merge base address updates.
static SDValue performNEONPostLDSTCombine(SDNode *N,
                                          TargetLowering::DAGCombinerInfo &DCI,
                                          SelectionDAG &DAG) {
  if (DCI.isBeforeLegalize() || DCI.isCalledByLegalizer())
    return SDValue();

  unsigned AddrOpIdx = N->getNumOperands() - 1;
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
    bool IsStore = false;
    bool IsLaneOp = false;
    bool IsDupOp = false;
    unsigned NewOpc = 0;
    unsigned NumVecs = 0;
    unsigned IntNo = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
    switch (IntNo) {
    default: llvm_unreachable("unexpected intrinsic for Neon base update");
    case Intrinsic::aarch64_neon_ld2:       NewOpc = AArch64ISD::LD2post;
      NumVecs = 2; break;
    case Intrinsic::aarch64_neon_ld3:       NewOpc = AArch64ISD::LD3post;
      NumVecs = 3; break;
    case Intrinsic::aarch64_neon_ld4:       NewOpc = AArch64ISD::LD4post;
      NumVecs = 4; break;
    case Intrinsic::aarch64_neon_st2:       NewOpc = AArch64ISD::ST2post;
      NumVecs = 2; IsStore = true; break;
    case Intrinsic::aarch64_neon_st3:       NewOpc = AArch64ISD::ST3post;
      NumVecs = 3; IsStore = true; break;
    case Intrinsic::aarch64_neon_st4:       NewOpc = AArch64ISD::ST4post;
      NumVecs = 4; IsStore = true; break;
    case Intrinsic::aarch64_neon_ld1x2:     NewOpc = AArch64ISD::LD1x2post;
      NumVecs = 2; break;
    case Intrinsic::aarch64_neon_ld1x3:     NewOpc = AArch64ISD::LD1x3post;
      NumVecs = 3; break;
    case Intrinsic::aarch64_neon_ld1x4:     NewOpc = AArch64ISD::LD1x4post;
      NumVecs = 4; break;
    case Intrinsic::aarch64_neon_st1x2:     NewOpc = AArch64ISD::ST1x2post;
      NumVecs = 2; IsStore = true; break;
    case Intrinsic::aarch64_neon_st1x3:     NewOpc = AArch64ISD::ST1x3post;
      NumVecs = 3; IsStore = true; break;
    case Intrinsic::aarch64_neon_st1x4:     NewOpc = AArch64ISD::ST1x4post;
      NumVecs = 4; IsStore = true; break;
    case Intrinsic::aarch64_neon_ld2r:      NewOpc = AArch64ISD::LD2DUPpost;
      NumVecs = 2; IsDupOp = true; break;
    case Intrinsic::aarch64_neon_ld3r:      NewOpc = AArch64ISD::LD3DUPpost;
      NumVecs = 3; IsDupOp = true; break;
    case Intrinsic::aarch64_neon_ld4r:      NewOpc = AArch64ISD::LD4DUPpost;
      NumVecs = 4; IsDupOp = true; break;
    case Intrinsic::aarch64_neon_ld2lane:   NewOpc = AArch64ISD::LD2LANEpost;
      NumVecs = 2; IsLaneOp = true; break;
    case Intrinsic::aarch64_neon_ld3lane:   NewOpc = AArch64ISD::LD3LANEpost;
      NumVecs = 3; IsLaneOp = true; break;
    case Intrinsic::aarch64_neon_ld4lane:   NewOpc = AArch64ISD::LD4LANEpost;
      NumVecs = 4; IsLaneOp = true; break;
    case Intrinsic::aarch64_neon_st2lane:   NewOpc = AArch64ISD::ST2LANEpost;
      NumVecs = 2; IsStore = true; IsLaneOp = true; break;
    case Intrinsic::aarch64_neon_st3lane:   NewOpc = AArch64ISD::ST3LANEpost;
      NumVecs = 3; IsStore = true; IsLaneOp = true; break;
    case Intrinsic::aarch64_neon_st4lane:   NewOpc = AArch64ISD::ST4LANEpost;
      NumVecs = 4; IsStore = true; IsLaneOp = true; break;
    }

    EVT VecTy;
    if (IsStore)
      VecTy = N->getOperand(2).getValueType();
    else
      VecTy = N->getValueType(0);

    // If the increment is a constant, it must match the memory ref size.
    SDValue Inc = User->getOperand(User->getOperand(0) == Addr ? 1 : 0);
    if (ConstantSDNode *CInc = dyn_cast<ConstantSDNode>(Inc.getNode())) {
      uint32_t IncVal = CInc->getZExtValue();
      unsigned NumBytes = NumVecs * VecTy.getSizeInBits() / 8;
      if (IsLaneOp || IsDupOp)
        NumBytes /= VecTy.getVectorNumElements();
      if (IncVal != NumBytes)
        continue;
      Inc = DAG.getRegister(AArch64::XZR, MVT::i64);
    }
    SmallVector<SDValue, 8> Ops;
    Ops.push_back(N->getOperand(0)); // Incoming chain
    // Load lane and store have vector list as input.
    if (IsLaneOp || IsStore)
      for (unsigned i = 2; i < AddrOpIdx; ++i)
        Ops.push_back(N->getOperand(i));
    Ops.push_back(Addr); // Base register
    Ops.push_back(Inc);

    // Return Types.
    EVT Tys[6];
    unsigned NumResultVecs = (IsStore ? 0 : NumVecs);
    unsigned n;
    for (n = 0; n < NumResultVecs; ++n)
      Tys[n] = VecTy;
    Tys[n++] = MVT::i64;  // Type of write back register
    Tys[n] = MVT::Other;  // Type of the chain
    SDVTList SDTys = DAG.getVTList(makeArrayRef(Tys, NumResultVecs + 2));

    MemIntrinsicSDNode *MemInt = cast<MemIntrinsicSDNode>(N);
    SDValue UpdN = DAG.getMemIntrinsicNode(NewOpc, SDLoc(N), SDTys, Ops,
                                           MemInt->getMemoryVT(),
                                           MemInt->getMemOperand());

    // Update the uses.
    std::vector<SDValue> NewResults;
    for (unsigned i = 0; i < NumResultVecs; ++i) {
      NewResults.push_back(SDValue(UpdN.getNode(), i));
    }
    NewResults.push_back(SDValue(UpdN.getNode(), NumResultVecs + 1));
    DCI.CombineTo(N, NewResults);
    DCI.CombineTo(User, SDValue(UpdN.getNode(), NumResultVecs));

    break;
  }
  return SDValue();
}

// Checks to see if the value is the prescribed width and returns information
// about its extension mode.
static
bool checkValueWidth(SDValue V, unsigned width, ISD::LoadExtType &ExtType) {
  ExtType = ISD::NON_EXTLOAD;
  switch(V.getNode()->getOpcode()) {
  default:
    return false;
  case ISD::LOAD: {
    LoadSDNode *LoadNode = cast<LoadSDNode>(V.getNode());
    if ((LoadNode->getMemoryVT() == MVT::i8 && width == 8)
       || (LoadNode->getMemoryVT() == MVT::i16 && width == 16)) {
      ExtType = LoadNode->getExtensionType();
      return true;
    }
    return false;
  }
  case ISD::AssertSext: {
    VTSDNode *TypeNode = cast<VTSDNode>(V.getNode()->getOperand(1));
    if ((TypeNode->getVT() == MVT::i8 && width == 8)
       || (TypeNode->getVT() == MVT::i16 && width == 16)) {
      ExtType = ISD::SEXTLOAD;
      return true;
    }
    return false;
  }
  case ISD::AssertZext: {
    VTSDNode *TypeNode = cast<VTSDNode>(V.getNode()->getOperand(1));
    if ((TypeNode->getVT() == MVT::i8 && width == 8)
       || (TypeNode->getVT() == MVT::i16 && width == 16)) {
      ExtType = ISD::ZEXTLOAD;
      return true;
    }
    return false;
  }
  case ISD::Constant:
  case ISD::TargetConstant: {
    if (std::abs(cast<ConstantSDNode>(V.getNode())->getSExtValue()) <
        1LL << (width - 1))
      return true;
    return false;
  }
  }

  return true;
}

// This function does a whole lot of voodoo to determine if the tests are
// equivalent without and with a mask. Essentially what happens is that given a
// DAG resembling:
//
//  +-------------+ +-------------+ +-------------+ +-------------+
//  |    Input    | | AddConstant | | CompConstant| |     CC      |
//  +-------------+ +-------------+ +-------------+ +-------------+
//           |           |           |               |
//           V           V           |    +----------+
//          +-------------+  +----+  |    |
//          |     ADD     |  |0xff|  |    |
//          +-------------+  +----+  |    |
//                  |           |    |    |
//                  V           V    |    |
//                 +-------------+   |    |
//                 |     AND     |   |    |
//                 +-------------+   |    |
//                      |            |    |
//                      +-----+      |    |
//                            |      |    |
//                            V      V    V
//                           +-------------+
//                           |     CMP     |
//                           +-------------+
//
// The AND node may be safely removed for some combinations of inputs. In
// particular we need to take into account the extension type of the Input,
// the exact values of AddConstant, CompConstant, and CC, along with the nominal
// width of the input (this can work for any width inputs, the above graph is
// specific to 8 bits.
//
// The specific equations were worked out by generating output tables for each
// AArch64CC value in terms of and AddConstant (w1), CompConstant(w2). The
// problem was simplified by working with 4 bit inputs, which means we only
// needed to reason about 24 distinct bit patterns: 8 patterns unique to zero
// extension (8,15), 8 patterns unique to sign extensions (-8,-1), and 8
// patterns present in both extensions (0,7). For every distinct set of
// AddConstant and CompConstants bit patterns we can consider the masked and
// unmasked versions to be equivalent if the result of this function is true for
// all 16 distinct bit patterns of for the current extension type of Input (w0).
//
//   sub      w8, w0, w1
//   and      w10, w8, #0x0f
//   cmp      w8, w2
//   cset     w9, AArch64CC
//   cmp      w10, w2
//   cset     w11, AArch64CC
//   cmp      w9, w11
//   cset     w0, eq
//   ret
//
// Since the above function shows when the outputs are equivalent it defines
// when it is safe to remove the AND. Unfortunately it only runs on AArch64 and
// would be expensive to run during compiles. The equations below were written
// in a test harness that confirmed they gave equivalent outputs to the above
// for all inputs function, so they can be used determine if the removal is
// legal instead.
//
// isEquivalentMaskless() is the code for testing if the AND can be removed
// factored out of the DAG recognition as the DAG can take several forms.

static
bool isEquivalentMaskless(unsigned CC, unsigned width,
                          ISD::LoadExtType ExtType, signed AddConstant,
                          signed CompConstant) {
  // By being careful about our equations and only writing the in term
  // symbolic values and well known constants (0, 1, -1, MaxUInt) we can
  // make them generally applicable to all bit widths.
  signed MaxUInt = (1 << width);

  // For the purposes of these comparisons sign extending the type is
  // equivalent to zero extending the add and displacing it by half the integer
  // width. Provided we are careful and make sure our equations are valid over
  // the whole range we can just adjust the input and avoid writing equations
  // for sign extended inputs.
  if (ExtType == ISD::SEXTLOAD)
    AddConstant -= (1 << (width-1));

  switch(CC) {
  case AArch64CC::LE:
  case AArch64CC::GT: {
    if ((AddConstant == 0) ||
        (CompConstant == MaxUInt - 1 && AddConstant < 0) ||
        (AddConstant >= 0 && CompConstant < 0) ||
        (AddConstant <= 0 && CompConstant <= 0 && CompConstant < AddConstant))
      return true;
  } break;
  case AArch64CC::LT:
  case AArch64CC::GE: {
    if ((AddConstant == 0) ||
        (AddConstant >= 0 && CompConstant <= 0) ||
        (AddConstant <= 0 && CompConstant <= 0 && CompConstant <= AddConstant))
      return true;
  } break;
  case AArch64CC::HI:
  case AArch64CC::LS: {
    if ((AddConstant >= 0 && CompConstant < 0) ||
       (AddConstant <= 0 && CompConstant >= -1 &&
        CompConstant < AddConstant + MaxUInt))
      return true;
  } break;
  case AArch64CC::PL:
  case AArch64CC::MI: {
    if ((AddConstant == 0) ||
        (AddConstant > 0 && CompConstant <= 0) ||
        (AddConstant < 0 && CompConstant <= AddConstant))
      return true;
  } break;
  case AArch64CC::LO:
  case AArch64CC::HS: {
    if ((AddConstant >= 0 && CompConstant <= 0) ||
        (AddConstant <= 0 && CompConstant >= 0 &&
         CompConstant <= AddConstant + MaxUInt))
      return true;
  } break;
  case AArch64CC::EQ:
  case AArch64CC::NE: {
    if ((AddConstant > 0 && CompConstant < 0) ||
        (AddConstant < 0 && CompConstant >= 0 &&
         CompConstant < AddConstant + MaxUInt) ||
        (AddConstant >= 0 && CompConstant >= 0 &&
         CompConstant >= AddConstant) ||
        (AddConstant <= 0 && CompConstant < 0 && CompConstant < AddConstant))

      return true;
  } break;
  case AArch64CC::VS:
  case AArch64CC::VC:
  case AArch64CC::AL:
  case AArch64CC::NV:
    return true;
  case AArch64CC::Invalid:
    break;
  }

  return false;
}

static
SDValue performCONDCombine(SDNode *N,
                           TargetLowering::DAGCombinerInfo &DCI,
                           SelectionDAG &DAG, unsigned CCIndex,
                           unsigned CmpIndex) {
  unsigned CC = cast<ConstantSDNode>(N->getOperand(CCIndex))->getSExtValue();
  SDNode *SubsNode = N->getOperand(CmpIndex).getNode();
  unsigned CondOpcode = SubsNode->getOpcode();

  if (CondOpcode != AArch64ISD::SUBS)
    return SDValue();

  // There is a SUBS feeding this condition. Is it fed by a mask we can
  // use?

  SDNode *AndNode = SubsNode->getOperand(0).getNode();
  unsigned MaskBits = 0;

  if (AndNode->getOpcode() != ISD::AND)
    return SDValue();

  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(AndNode->getOperand(1))) {
    uint32_t CNV = CN->getZExtValue();
    if (CNV == 255)
      MaskBits = 8;
    else if (CNV == 65535)
      MaskBits = 16;
  }

  if (!MaskBits)
    return SDValue();

  SDValue AddValue = AndNode->getOperand(0);

  if (AddValue.getOpcode() != ISD::ADD)
    return SDValue();

  // The basic dag structure is correct, grab the inputs and validate them.

  SDValue AddInputValue1 = AddValue.getNode()->getOperand(0);
  SDValue AddInputValue2 = AddValue.getNode()->getOperand(1);
  SDValue SubsInputValue = SubsNode->getOperand(1);

  // The mask is present and the provenance of all the values is a smaller type,
  // lets see if the mask is superfluous.

  if (!isa<ConstantSDNode>(AddInputValue2.getNode()) ||
      !isa<ConstantSDNode>(SubsInputValue.getNode()))
    return SDValue();

  ISD::LoadExtType ExtType;

  if (!checkValueWidth(SubsInputValue, MaskBits, ExtType) ||
      !checkValueWidth(AddInputValue2, MaskBits, ExtType) ||
      !checkValueWidth(AddInputValue1, MaskBits, ExtType) )
    return SDValue();

  if(!isEquivalentMaskless(CC, MaskBits, ExtType,
                cast<ConstantSDNode>(AddInputValue2.getNode())->getSExtValue(),
                cast<ConstantSDNode>(SubsInputValue.getNode())->getSExtValue()))
    return SDValue();

  // The AND is not necessary, remove it.

  SDVTList VTs = DAG.getVTList(SubsNode->getValueType(0),
                               SubsNode->getValueType(1));
  SDValue Ops[] = { AddValue, SubsNode->getOperand(1) };

  SDValue NewValue = DAG.getNode(CondOpcode, SDLoc(SubsNode), VTs, Ops);
  DAG.ReplaceAllUsesWith(SubsNode, NewValue.getNode());

  return SDValue(N, 0);
}

// Optimize compare with zero and branch.
static SDValue performBRCONDCombine(SDNode *N,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    SelectionDAG &DAG) {
  SDValue NV = performCONDCombine(N, DCI, DAG, 2, 3);
  if (NV.getNode())
    N = NV.getNode();
  SDValue Chain = N->getOperand(0);
  SDValue Dest = N->getOperand(1);
  SDValue CCVal = N->getOperand(2);
  SDValue Cmp = N->getOperand(3);

  assert(isa<ConstantSDNode>(CCVal) && "Expected a ConstantSDNode here!");
  unsigned CC = cast<ConstantSDNode>(CCVal)->getZExtValue();
  if (CC != AArch64CC::EQ && CC != AArch64CC::NE)
    return SDValue();

  unsigned CmpOpc = Cmp.getOpcode();
  if (CmpOpc != AArch64ISD::ADDS && CmpOpc != AArch64ISD::SUBS)
    return SDValue();

  // Only attempt folding if there is only one use of the flag and no use of the
  // value.
  if (!Cmp->hasNUsesOfValue(0, 0) || !Cmp->hasNUsesOfValue(1, 1))
    return SDValue();

  SDValue LHS = Cmp.getOperand(0);
  SDValue RHS = Cmp.getOperand(1);

  assert(LHS.getValueType() == RHS.getValueType() &&
         "Expected the value type to be the same for both operands!");
  if (LHS.getValueType() != MVT::i32 && LHS.getValueType() != MVT::i64)
    return SDValue();

  if (isa<ConstantSDNode>(LHS) && cast<ConstantSDNode>(LHS)->isNullValue())
    std::swap(LHS, RHS);

  if (!isa<ConstantSDNode>(RHS) || !cast<ConstantSDNode>(RHS)->isNullValue())
    return SDValue();

  if (LHS.getOpcode() == ISD::SHL || LHS.getOpcode() == ISD::SRA ||
      LHS.getOpcode() == ISD::SRL)
    return SDValue();

  // Fold the compare into the branch instruction.
  SDValue BR;
  if (CC == AArch64CC::EQ)
    BR = DAG.getNode(AArch64ISD::CBZ, SDLoc(N), MVT::Other, Chain, LHS, Dest);
  else
    BR = DAG.getNode(AArch64ISD::CBNZ, SDLoc(N), MVT::Other, Chain, LHS, Dest);

  // Do not add new nodes to DAG combiner worklist.
  DCI.CombineTo(N, BR, false);

  return SDValue();
}

// vselect (v1i1 setcc) ->
//     vselect (v1iXX setcc)  (XX is the size of the compared operand type)
// FIXME: Currently the type legalizer can't handle VSELECT having v1i1 as
// condition. If it can legalize "VSELECT v1i1" correctly, no need to combine
// such VSELECT.
static SDValue performVSelectCombine(SDNode *N, SelectionDAG &DAG) {
  SDValue N0 = N->getOperand(0);
  EVT CCVT = N0.getValueType();

  if (N0.getOpcode() != ISD::SETCC || CCVT.getVectorNumElements() != 1 ||
      CCVT.getVectorElementType() != MVT::i1)
    return SDValue();

  EVT ResVT = N->getValueType(0);
  EVT CmpVT = N0.getOperand(0).getValueType();
  // Only combine when the result type is of the same size as the compared
  // operands.
  if (ResVT.getSizeInBits() != CmpVT.getSizeInBits())
    return SDValue();

  SDValue IfTrue = N->getOperand(1);
  SDValue IfFalse = N->getOperand(2);
  SDValue SetCC =
      DAG.getSetCC(SDLoc(N), CmpVT.changeVectorElementTypeToInteger(),
                   N0.getOperand(0), N0.getOperand(1),
                   cast<CondCodeSDNode>(N0.getOperand(2))->get());
  return DAG.getNode(ISD::VSELECT, SDLoc(N), ResVT, SetCC,
                     IfTrue, IfFalse);
}

/// A vector select: "(select vL, vR, (setcc LHS, RHS))" is best performed with
/// the compare-mask instructions rather than going via NZCV, even if LHS and
/// RHS are really scalar. This replaces any scalar setcc in the above pattern
/// with a vector one followed by a DUP shuffle on the result.
static SDValue performSelectCombine(SDNode *N,
                                    TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  SDValue N0 = N->getOperand(0);
  EVT ResVT = N->getValueType(0);

  if (N0.getOpcode() != ISD::SETCC)
    return SDValue();

  // Make sure the SETCC result is either i1 (initial DAG), or i32, the lowered
  // scalar SetCCResultType. We also don't expect vectors, because we assume
  // that selects fed by vector SETCCs are canonicalized to VSELECT.
  assert((N0.getValueType() == MVT::i1 || N0.getValueType() == MVT::i32) &&
         "Scalar-SETCC feeding SELECT has unexpected result type!");

  // If NumMaskElts == 0, the comparison is larger than select result. The
  // largest real NEON comparison is 64-bits per lane, which means the result is
  // at most 32-bits and an illegal vector. Just bail out for now.
  EVT SrcVT = N0.getOperand(0).getValueType();

  // Don't try to do this optimization when the setcc itself has i1 operands.
  // There are no legal vectors of i1, so this would be pointless.
  if (SrcVT == MVT::i1)
    return SDValue();

  int NumMaskElts = ResVT.getSizeInBits() / SrcVT.getSizeInBits();
  if (!ResVT.isVector() || NumMaskElts == 0)
    return SDValue();

  SrcVT = EVT::getVectorVT(*DAG.getContext(), SrcVT, NumMaskElts);
  EVT CCVT = SrcVT.changeVectorElementTypeToInteger();

  // Also bail out if the vector CCVT isn't the same size as ResVT.
  // This can happen if the SETCC operand size doesn't divide the ResVT size
  // (e.g., f64 vs v3f32).
  if (CCVT.getSizeInBits() != ResVT.getSizeInBits())
    return SDValue();

  // Make sure we didn't create illegal types, if we're not supposed to.
  assert(DCI.isBeforeLegalize() ||
         DAG.getTargetLoweringInfo().isTypeLegal(SrcVT));

  // First perform a vector comparison, where lane 0 is the one we're interested
  // in.
  SDLoc DL(N0);
  SDValue LHS =
      DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, SrcVT, N0.getOperand(0));
  SDValue RHS =
      DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, SrcVT, N0.getOperand(1));
  SDValue SetCC = DAG.getNode(ISD::SETCC, DL, CCVT, LHS, RHS, N0.getOperand(2));

  // Now duplicate the comparison mask we want across all other lanes.
  SmallVector<int, 8> DUPMask(CCVT.getVectorNumElements(), 0);
  SDValue Mask = DAG.getVectorShuffle(CCVT, DL, SetCC, SetCC, DUPMask.data());
  Mask = DAG.getNode(ISD::BITCAST, DL,
                     ResVT.changeVectorElementTypeToInteger(), Mask);

  return DAG.getSelect(DL, ResVT, Mask, N->getOperand(1), N->getOperand(2));
}

/// Get rid of unnecessary NVCASTs (that don't change the type).
static SDValue performNVCASTCombine(SDNode *N) {
  if (N->getValueType(0) == N->getOperand(0).getValueType())
    return N->getOperand(0);

  return SDValue();
}

SDValue AArch64TargetLowering::PerformDAGCombine(SDNode *N,
                                                 DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::ADD:
  case ISD::SUB:
    return performAddSubLongCombine(N, DCI, DAG);
  case ISD::XOR:
    return performXorCombine(N, DAG, DCI, Subtarget);
  case ISD::MUL:
    return performMulCombine(N, DAG, DCI, Subtarget);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return performIntToFpCombine(N, DAG, Subtarget);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return performFpToIntCombine(N, DAG, Subtarget);
  case ISD::FDIV:
    return performFDivCombine(N, DAG, Subtarget);
  case ISD::OR:
    return performORCombine(N, DCI, Subtarget);
  case ISD::INTRINSIC_WO_CHAIN:
    return performIntrinsicCombine(N, DCI, Subtarget);
  case ISD::ANY_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
    return performExtendCombine(N, DCI, DAG);
  case ISD::BITCAST:
    return performBitcastCombine(N, DCI, DAG);
  case ISD::CONCAT_VECTORS:
    return performConcatVectorsCombine(N, DCI, DAG);
  case ISD::SELECT: {
    SDValue RV = performSelectCombine(N, DCI);
    if (!RV.getNode())
      RV = performAcrossLaneMinMaxReductionCombine(N, DAG, Subtarget);
    return RV;
  }
  case ISD::VSELECT:
    return performVSelectCombine(N, DCI.DAG);
  case ISD::LOAD:
    if (performTBISimplification(N->getOperand(1), DCI, DAG))
      return SDValue(N, 0);
    break;
  case ISD::STORE:
    return performSTORECombine(N, DCI, DAG, Subtarget);
  case AArch64ISD::BRCOND:
    return performBRCONDCombine(N, DCI, DAG);
  case AArch64ISD::CSEL:
    return performCONDCombine(N, DCI, DAG, 2, 3);
  case AArch64ISD::DUP:
    return performPostLD1Combine(N, DCI, false);
  case AArch64ISD::NVCAST:
    return performNVCASTCombine(N);
  case ISD::INSERT_VECTOR_ELT:
    return performPostLD1Combine(N, DCI, true);
  case ISD::EXTRACT_VECTOR_ELT:
    return performAcrossLaneAddReductionCombine(N, DAG, Subtarget);
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN:
    switch (cast<ConstantSDNode>(N->getOperand(1))->getZExtValue()) {
    case Intrinsic::aarch64_neon_ld2:
    case Intrinsic::aarch64_neon_ld3:
    case Intrinsic::aarch64_neon_ld4:
    case Intrinsic::aarch64_neon_ld1x2:
    case Intrinsic::aarch64_neon_ld1x3:
    case Intrinsic::aarch64_neon_ld1x4:
    case Intrinsic::aarch64_neon_ld2lane:
    case Intrinsic::aarch64_neon_ld3lane:
    case Intrinsic::aarch64_neon_ld4lane:
    case Intrinsic::aarch64_neon_ld2r:
    case Intrinsic::aarch64_neon_ld3r:
    case Intrinsic::aarch64_neon_ld4r:
    case Intrinsic::aarch64_neon_st2:
    case Intrinsic::aarch64_neon_st3:
    case Intrinsic::aarch64_neon_st4:
    case Intrinsic::aarch64_neon_st1x2:
    case Intrinsic::aarch64_neon_st1x3:
    case Intrinsic::aarch64_neon_st1x4:
    case Intrinsic::aarch64_neon_st2lane:
    case Intrinsic::aarch64_neon_st3lane:
    case Intrinsic::aarch64_neon_st4lane:
      return performNEONPostLDSTCombine(N, DCI, DAG);
    default:
      break;
    }
  }
  return SDValue();
}

// Check if the return value is used as only a return value, as otherwise
// we can't perform a tail-call. In particular, we need to check for
// target ISD nodes that are returns and any other "odd" constructs
// that the generic analysis code won't necessarily catch.
bool AArch64TargetLowering::isUsedByReturnOnly(SDNode *N,
                                               SDValue &Chain) const {
  if (N->getNumValues() != 1)
    return false;
  if (!N->hasNUsesOfValue(1, 0))
    return false;

  SDValue TCChain = Chain;
  SDNode *Copy = *N->use_begin();
  if (Copy->getOpcode() == ISD::CopyToReg) {
    // If the copy has a glue operand, we conservatively assume it isn't safe to
    // perform a tail call.
    if (Copy->getOperand(Copy->getNumOperands() - 1).getValueType() ==
        MVT::Glue)
      return false;
    TCChain = Copy->getOperand(0);
  } else if (Copy->getOpcode() != ISD::FP_EXTEND)
    return false;

  bool HasRet = false;
  for (SDNode *Node : Copy->uses()) {
    if (Node->getOpcode() != AArch64ISD::RET_FLAG)
      return false;
    HasRet = true;
  }

  if (!HasRet)
    return false;

  Chain = TCChain;
  return true;
}

// Return whether the an instruction can potentially be optimized to a tail
// call. This will cause the optimizers to attempt to move, or duplicate,
// return instructions to help enable tail call optimizations for this
// instruction.
bool AArch64TargetLowering::mayBeEmittedAsTailCall(CallInst *CI) const {
  if (!CI->isTailCall())
    return false;

  return true;
}

bool AArch64TargetLowering::getIndexedAddressParts(SDNode *Op, SDValue &Base,
                                                   SDValue &Offset,
                                                   ISD::MemIndexedMode &AM,
                                                   bool &IsInc,
                                                   SelectionDAG &DAG) const {
  if (Op->getOpcode() != ISD::ADD && Op->getOpcode() != ISD::SUB)
    return false;

  Base = Op->getOperand(0);
  // All of the indexed addressing mode instructions take a signed
  // 9 bit immediate offset.
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(Op->getOperand(1))) {
    int64_t RHSC = (int64_t)RHS->getZExtValue();
    if (RHSC >= 256 || RHSC <= -256)
      return false;
    IsInc = (Op->getOpcode() == ISD::ADD);
    Offset = Op->getOperand(1);
    return true;
  }
  return false;
}

bool AArch64TargetLowering::getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                                      SDValue &Offset,
                                                      ISD::MemIndexedMode &AM,
                                                      SelectionDAG &DAG) const {
  EVT VT;
  SDValue Ptr;
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    VT = LD->getMemoryVT();
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    VT = ST->getMemoryVT();
    Ptr = ST->getBasePtr();
  } else
    return false;

  bool IsInc;
  if (!getIndexedAddressParts(Ptr.getNode(), Base, Offset, AM, IsInc, DAG))
    return false;
  AM = IsInc ? ISD::PRE_INC : ISD::PRE_DEC;
  return true;
}

bool AArch64TargetLowering::getPostIndexedAddressParts(
    SDNode *N, SDNode *Op, SDValue &Base, SDValue &Offset,
    ISD::MemIndexedMode &AM, SelectionDAG &DAG) const {
  EVT VT;
  SDValue Ptr;
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    VT = LD->getMemoryVT();
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    VT = ST->getMemoryVT();
    Ptr = ST->getBasePtr();
  } else
    return false;

  bool IsInc;
  if (!getIndexedAddressParts(Op, Base, Offset, AM, IsInc, DAG))
    return false;
  // Post-indexing updates the base, so it's not a valid transform
  // if that's not the same as the load's pointer.
  if (Ptr != Base)
    return false;
  AM = IsInc ? ISD::POST_INC : ISD::POST_DEC;
  return true;
}

static void ReplaceBITCASTResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                                  SelectionDAG &DAG) {
  SDLoc DL(N);
  SDValue Op = N->getOperand(0);

  if (N->getValueType(0) != MVT::i16 || Op.getValueType() != MVT::f16)
    return;

  Op = SDValue(
      DAG.getMachineNode(TargetOpcode::INSERT_SUBREG, DL, MVT::f32,
                         DAG.getUNDEF(MVT::i32), Op,
                         DAG.getTargetConstant(AArch64::hsub, DL, MVT::i32)),
      0);
  Op = DAG.getNode(ISD::BITCAST, DL, MVT::i32, Op);
  Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, Op));
}

static void ReplaceReductionResults(SDNode *N,
                                    SmallVectorImpl<SDValue> &Results,
                                    SelectionDAG &DAG, unsigned InterOp,
                                    unsigned AcrossOp) {
  EVT LoVT, HiVT;
  SDValue Lo, Hi;
  SDLoc dl(N);
  std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(N->getValueType(0));
  std::tie(Lo, Hi) = DAG.SplitVectorOperand(N, 0);
  SDValue InterVal = DAG.getNode(InterOp, dl, LoVT, Lo, Hi);
  SDValue SplitVal = DAG.getNode(AcrossOp, dl, LoVT, InterVal);
  Results.push_back(SplitVal);
}

void AArch64TargetLowering::ReplaceNodeResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom expand this");
  case ISD::BITCAST:
    ReplaceBITCASTResults(N, Results, DAG);
    return;
  case AArch64ISD::SADDV:
    ReplaceReductionResults(N, Results, DAG, ISD::ADD, AArch64ISD::SADDV);
    return;
  case AArch64ISD::UADDV:
    ReplaceReductionResults(N, Results, DAG, ISD::ADD, AArch64ISD::UADDV);
    return;
  case AArch64ISD::SMINV:
    ReplaceReductionResults(N, Results, DAG, ISD::SMIN, AArch64ISD::SMINV);
    return;
  case AArch64ISD::UMINV:
    ReplaceReductionResults(N, Results, DAG, ISD::UMIN, AArch64ISD::UMINV);
    return;
  case AArch64ISD::SMAXV:
    ReplaceReductionResults(N, Results, DAG, ISD::SMAX, AArch64ISD::SMAXV);
    return;
  case AArch64ISD::UMAXV:
    ReplaceReductionResults(N, Results, DAG, ISD::UMAX, AArch64ISD::UMAXV);
    return;
  case ISD::FP_TO_UINT:
  case ISD::FP_TO_SINT:
    assert(N->getValueType(0) == MVT::i128 && "unexpected illegal conversion");
    // Let normal code take care of it by not adding anything to Results.
    return;
  }
}

bool AArch64TargetLowering::useLoadStackGuardNode() const {
  return true;
}

unsigned AArch64TargetLowering::combineRepeatedFPDivisors() const {
  // Combine multiple FDIVs with the same divisor into multiple FMULs by the
  // reciprocal if there are three or more FDIVs.
  return 3;
}

TargetLoweringBase::LegalizeTypeAction
AArch64TargetLowering::getPreferredVectorAction(EVT VT) const {
  MVT SVT = VT.getSimpleVT();
  // During type legalization, we prefer to widen v1i8, v1i16, v1i32  to v8i8,
  // v4i16, v2i32 instead of to promote.
  if (SVT == MVT::v1i8 || SVT == MVT::v1i16 || SVT == MVT::v1i32
      || SVT == MVT::v1f32)
    return TypeWidenVector;

  return TargetLoweringBase::getPreferredVectorAction(VT);
}

// Loads and stores less than 128-bits are already atomic; ones above that
// are doomed anyway, so defer to the default libcall and blame the OS when
// things go wrong.
bool AArch64TargetLowering::shouldExpandAtomicStoreInIR(StoreInst *SI) const {
  unsigned Size = SI->getValueOperand()->getType()->getPrimitiveSizeInBits();
  return Size == 128;
}

// Loads and stores less than 128-bits are already atomic; ones above that
// are doomed anyway, so defer to the default libcall and blame the OS when
// things go wrong.
TargetLowering::AtomicExpansionKind
AArch64TargetLowering::shouldExpandAtomicLoadInIR(LoadInst *LI) const {
  unsigned Size = LI->getType()->getPrimitiveSizeInBits();
  return Size == 128 ? AtomicExpansionKind::LLSC : AtomicExpansionKind::None;
}

// For the real atomic operations, we have ldxr/stxr up to 128 bits,
TargetLowering::AtomicExpansionKind
AArch64TargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const {
  unsigned Size = AI->getType()->getPrimitiveSizeInBits();
  return Size <= 128 ? AtomicExpansionKind::LLSC : AtomicExpansionKind::None;
}

bool AArch64TargetLowering::shouldExpandAtomicCmpXchgInIR(
    AtomicCmpXchgInst *AI) const {
  return true;
}

Value *AArch64TargetLowering::emitLoadLinked(IRBuilder<> &Builder, Value *Addr,
                                             AtomicOrdering Ord) const {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Type *ValTy = cast<PointerType>(Addr->getType())->getElementType();
  bool IsAcquire = isAtLeastAcquire(Ord);

  // Since i128 isn't legal and intrinsics don't get type-lowered, the ldrexd
  // intrinsic must return {i64, i64} and we have to recombine them into a
  // single i128 here.
  if (ValTy->getPrimitiveSizeInBits() == 128) {
    Intrinsic::ID Int =
        IsAcquire ? Intrinsic::aarch64_ldaxp : Intrinsic::aarch64_ldxp;
    Function *Ldxr = llvm::Intrinsic::getDeclaration(M, Int);

    Addr = Builder.CreateBitCast(Addr, Type::getInt8PtrTy(M->getContext()));
    Value *LoHi = Builder.CreateCall(Ldxr, Addr, "lohi");

    Value *Lo = Builder.CreateExtractValue(LoHi, 0, "lo");
    Value *Hi = Builder.CreateExtractValue(LoHi, 1, "hi");
    Lo = Builder.CreateZExt(Lo, ValTy, "lo64");
    Hi = Builder.CreateZExt(Hi, ValTy, "hi64");
    return Builder.CreateOr(
        Lo, Builder.CreateShl(Hi, ConstantInt::get(ValTy, 64)), "val64");
  }

  Type *Tys[] = { Addr->getType() };
  Intrinsic::ID Int =
      IsAcquire ? Intrinsic::aarch64_ldaxr : Intrinsic::aarch64_ldxr;
  Function *Ldxr = llvm::Intrinsic::getDeclaration(M, Int, Tys);

  return Builder.CreateTruncOrBitCast(
      Builder.CreateCall(Ldxr, Addr),
      cast<PointerType>(Addr->getType())->getElementType());
}

void AArch64TargetLowering::emitAtomicCmpXchgNoStoreLLBalance(
    IRBuilder<> &Builder) const {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  Builder.CreateCall(
      llvm::Intrinsic::getDeclaration(M, Intrinsic::aarch64_clrex));
}

Value *AArch64TargetLowering::emitStoreConditional(IRBuilder<> &Builder,
                                                   Value *Val, Value *Addr,
                                                   AtomicOrdering Ord) const {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  bool IsRelease = isAtLeastRelease(Ord);

  // Since the intrinsics must have legal type, the i128 intrinsics take two
  // parameters: "i64, i64". We must marshal Val into the appropriate form
  // before the call.
  if (Val->getType()->getPrimitiveSizeInBits() == 128) {
    Intrinsic::ID Int =
        IsRelease ? Intrinsic::aarch64_stlxp : Intrinsic::aarch64_stxp;
    Function *Stxr = Intrinsic::getDeclaration(M, Int);
    Type *Int64Ty = Type::getInt64Ty(M->getContext());

    Value *Lo = Builder.CreateTrunc(Val, Int64Ty, "lo");
    Value *Hi = Builder.CreateTrunc(Builder.CreateLShr(Val, 64), Int64Ty, "hi");
    Addr = Builder.CreateBitCast(Addr, Type::getInt8PtrTy(M->getContext()));
    return Builder.CreateCall(Stxr, {Lo, Hi, Addr});
  }

  Intrinsic::ID Int =
      IsRelease ? Intrinsic::aarch64_stlxr : Intrinsic::aarch64_stxr;
  Type *Tys[] = { Addr->getType() };
  Function *Stxr = Intrinsic::getDeclaration(M, Int, Tys);

  return Builder.CreateCall(Stxr,
                            {Builder.CreateZExtOrBitCast(
                                 Val, Stxr->getFunctionType()->getParamType(0)),
                             Addr});
}

bool AArch64TargetLowering::functionArgumentNeedsConsecutiveRegisters(
    Type *Ty, CallingConv::ID CallConv, bool isVarArg) const {
  return Ty->isArrayTy();
}

bool AArch64TargetLowering::shouldNormalizeToSelectSequence(LLVMContext &,
                                                            EVT) const {
  return false;
}

Value *AArch64TargetLowering::getSafeStackPointerLocation(IRBuilder<> &IRB) const {
  if (!Subtarget->isTargetAndroid())
    return TargetLowering::getSafeStackPointerLocation(IRB);

  // Android provides a fixed TLS slot for the SafeStack pointer. See the
  // definition of TLS_SLOT_SAFESTACK in
  // https://android.googlesource.com/platform/bionic/+/master/libc/private/bionic_tls.h
  const unsigned TlsOffset = 0x48;
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  Function *ThreadPointerFunc =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_thread_pointer);
  return IRB.CreatePointerCast(
      IRB.CreateConstGEP1_32(IRB.CreateCall(ThreadPointerFunc), TlsOffset),
      Type::getInt8PtrTy(IRB.getContext())->getPointerTo(0));
}
