//===-- SIISelLowering.cpp - SI DAG Lowering Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Custom DAG lowering for SI
//
//===----------------------------------------------------------------------===//

#ifdef _MSC_VER
// Provide M_PI.
#define _USE_MATH_DEFINES
#endif

#include "SIISelLowering.h"
#include "AMDGPU.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/DAGCombine.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetCallingConv.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetOptions.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "si-lower"

STATISTIC(NumTailCalls, "Number of tail calls");

static cl::opt<bool> EnableVGPRIndexMode(
  "amdgpu-vgpr-index-mode",
  cl::desc("Use GPR indexing mode instead of movrel for vector indexing"),
  cl::init(false));

static cl::opt<unsigned> AssumeFrameIndexHighZeroBits(
  "amdgpu-frame-index-zero-bits",
  cl::desc("High bits of frame index assumed to be zero"),
  cl::init(5),
  cl::ReallyHidden);

static unsigned findFirstFreeSGPR(CCState &CCInfo) {
  unsigned NumSGPRs = AMDGPU::SGPR_32RegClass.getNumRegs();
  for (unsigned Reg = 0; Reg < NumSGPRs; ++Reg) {
    if (!CCInfo.isAllocated(AMDGPU::SGPR0 + Reg)) {
      return AMDGPU::SGPR0 + Reg;
    }
  }
  llvm_unreachable("Cannot allocate sgpr");
}

SITargetLowering::SITargetLowering(const TargetMachine &TM,
                                   const SISubtarget &STI)
    : AMDGPUTargetLowering(TM, STI) {
  addRegisterClass(MVT::i1, &AMDGPU::VReg_1RegClass);
  addRegisterClass(MVT::i64, &AMDGPU::SReg_64RegClass);

  addRegisterClass(MVT::i32, &AMDGPU::SReg_32_XM0RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::VGPR_32RegClass);

  addRegisterClass(MVT::f64, &AMDGPU::VReg_64RegClass);
  addRegisterClass(MVT::v2i32, &AMDGPU::SReg_64RegClass);
  addRegisterClass(MVT::v2f32, &AMDGPU::VReg_64RegClass);

  addRegisterClass(MVT::v2i64, &AMDGPU::SReg_128RegClass);
  addRegisterClass(MVT::v2f64, &AMDGPU::SReg_128RegClass);

  addRegisterClass(MVT::v4i32, &AMDGPU::SReg_128RegClass);
  addRegisterClass(MVT::v4f32, &AMDGPU::VReg_128RegClass);

  addRegisterClass(MVT::v8i32, &AMDGPU::SReg_256RegClass);
  addRegisterClass(MVT::v8f32, &AMDGPU::VReg_256RegClass);

  addRegisterClass(MVT::v16i32, &AMDGPU::SReg_512RegClass);
  addRegisterClass(MVT::v16f32, &AMDGPU::VReg_512RegClass);

  if (Subtarget->has16BitInsts()) {
    addRegisterClass(MVT::i16, &AMDGPU::SReg_32_XM0RegClass);
    addRegisterClass(MVT::f16, &AMDGPU::SReg_32_XM0RegClass);
  }

  if (Subtarget->hasVOP3PInsts()) {
    addRegisterClass(MVT::v2i16, &AMDGPU::SReg_32_XM0RegClass);
    addRegisterClass(MVT::v2f16, &AMDGPU::SReg_32_XM0RegClass);
  }

  computeRegisterProperties(STI.getRegisterInfo());

  // We need to custom lower vector stores from local memory
  setOperationAction(ISD::LOAD, MVT::v2i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v8i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v16i32, Custom);
  setOperationAction(ISD::LOAD, MVT::i1, Custom);

  setOperationAction(ISD::STORE, MVT::v2i32, Custom);
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);
  setOperationAction(ISD::STORE, MVT::v8i32, Custom);
  setOperationAction(ISD::STORE, MVT::v16i32, Custom);
  setOperationAction(ISD::STORE, MVT::i1, Custom);

  setTruncStoreAction(MVT::v2i32, MVT::v2i16, Expand);
  setTruncStoreAction(MVT::v4i32, MVT::v4i16, Expand);
  setTruncStoreAction(MVT::v8i32, MVT::v8i16, Expand);
  setTruncStoreAction(MVT::v16i32, MVT::v16i16, Expand);
  setTruncStoreAction(MVT::v32i32, MVT::v32i16, Expand);
  setTruncStoreAction(MVT::v2i32, MVT::v2i8, Expand);
  setTruncStoreAction(MVT::v4i32, MVT::v4i8, Expand);
  setTruncStoreAction(MVT::v8i32, MVT::v8i8, Expand);
  setTruncStoreAction(MVT::v16i32, MVT::v16i8, Expand);
  setTruncStoreAction(MVT::v32i32, MVT::v32i8, Expand);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::ConstantPool, MVT::v2i64, Expand);

  setOperationAction(ISD::SELECT, MVT::i1, Promote);
  setOperationAction(ISD::SELECT, MVT::i64, Custom);
  setOperationAction(ISD::SELECT, MVT::f64, Promote);
  AddPromotedToType(ISD::SELECT, MVT::f64, MVT::i64);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i1, Expand);

  setOperationAction(ISD::SETCC, MVT::i1, Promote);
  setOperationAction(ISD::SETCC, MVT::v2i1, Expand);
  setOperationAction(ISD::SETCC, MVT::v4i1, Expand);
  AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);

  setOperationAction(ISD::TRUNCATE, MVT::v2i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v2f32, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i1, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i1, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i8, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i8, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::Other, Custom);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::f32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v4f32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v2f16, Custom);

  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::v2i16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::v2f16, Custom);

  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::BR_CC, MVT::i1, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::i64, Expand);
  setOperationAction(ISD::BR_CC, MVT::f32, Expand);
  setOperationAction(ISD::BR_CC, MVT::f64, Expand);

  setOperationAction(ISD::UADDO, MVT::i32, Legal);
  setOperationAction(ISD::USUBO, MVT::i32, Legal);

  setOperationAction(ISD::ADDCARRY, MVT::i32, Legal);
  setOperationAction(ISD::SUBCARRY, MVT::i32, Legal);

  // We only support LOAD/STORE and vector manipulation ops for vectors
  // with > 4 elements.
  for (MVT VT : {MVT::v8i32, MVT::v8f32, MVT::v16i32, MVT::v16f32,
        MVT::v2i64, MVT::v2f64}) {
    for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op) {
      switch (Op) {
      case ISD::LOAD:
      case ISD::STORE:
      case ISD::BUILD_VECTOR:
      case ISD::BITCAST:
      case ISD::EXTRACT_VECTOR_ELT:
      case ISD::INSERT_VECTOR_ELT:
      case ISD::INSERT_SUBVECTOR:
      case ISD::EXTRACT_SUBVECTOR:
      case ISD::SCALAR_TO_VECTOR:
        break;
      case ISD::CONCAT_VECTORS:
        setOperationAction(Op, VT, Custom);
        break;
      default:
        setOperationAction(Op, VT, Expand);
        break;
      }
    }
  }

  // TODO: For dynamic 64-bit vector inserts/extracts, should emit a pseudo that
  // is expanded to avoid having two separate loops in case the index is a VGPR.

  // Most operations are naturally 32-bit vector operations. We only support
  // load and store of i64 vectors, so promote v2i64 vector operations to v4i32.
  for (MVT Vec64 : { MVT::v2i64, MVT::v2f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v4i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v4i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v4i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v4i32);
  }

  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8f32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16f32, Expand);

  // Avoid stack access for these.
  // TODO: Generalize to more vector types.
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2i16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2f16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f16, Custom);

  // BUFFER/FLAT_ATOMIC_CMP_SWAP on GCN GPUs needs input marshalling,
  // and output demarshalling
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i64, Custom);

  // We can't return success/failure, only the old value,
  // let LLVM add the comparison
  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i64, Expand);

  if (getSubtarget()->hasFlatAddressSpace()) {
    setOperationAction(ISD::ADDRSPACECAST, MVT::i32, Custom);
    setOperationAction(ISD::ADDRSPACECAST, MVT::i64, Custom);
  }

  setOperationAction(ISD::BSWAP, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i32, Legal);

  // On SI this is s_memtime and s_memrealtime on VI.
  setOperationAction(ISD::READCYCLECOUNTER, MVT::i64, Legal);
  setOperationAction(ISD::TRAP, MVT::Other, Custom);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Custom);

  setOperationAction(ISD::FMINNUM, MVT::f64, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::f64, Legal);

  if (Subtarget->getGeneration() >= SISubtarget::SEA_ISLANDS) {
    setOperationAction(ISD::FTRUNC, MVT::f64, Legal);
    setOperationAction(ISD::FCEIL, MVT::f64, Legal);
    setOperationAction(ISD::FRINT, MVT::f64, Legal);
  }

  setOperationAction(ISD::FFLOOR, MVT::f64, Legal);

  setOperationAction(ISD::FSIN, MVT::f32, Custom);
  setOperationAction(ISD::FCOS, MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f64, Custom);

  if (Subtarget->has16BitInsts()) {
    setOperationAction(ISD::Constant, MVT::i16, Legal);

    setOperationAction(ISD::SMIN, MVT::i16, Legal);
    setOperationAction(ISD::SMAX, MVT::i16, Legal);

    setOperationAction(ISD::UMIN, MVT::i16, Legal);
    setOperationAction(ISD::UMAX, MVT::i16, Legal);

    setOperationAction(ISD::SIGN_EXTEND, MVT::i16, Promote);
    AddPromotedToType(ISD::SIGN_EXTEND, MVT::i16, MVT::i32);

    setOperationAction(ISD::ROTR, MVT::i16, Promote);
    setOperationAction(ISD::ROTL, MVT::i16, Promote);

    setOperationAction(ISD::SDIV, MVT::i16, Promote);
    setOperationAction(ISD::UDIV, MVT::i16, Promote);
    setOperationAction(ISD::SREM, MVT::i16, Promote);
    setOperationAction(ISD::UREM, MVT::i16, Promote);

    setOperationAction(ISD::BSWAP, MVT::i16, Promote);
    setOperationAction(ISD::BITREVERSE, MVT::i16, Promote);

    setOperationAction(ISD::CTTZ, MVT::i16, Promote);
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i16, Promote);
    setOperationAction(ISD::CTLZ, MVT::i16, Promote);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i16, Promote);

    setOperationAction(ISD::SELECT_CC, MVT::i16, Expand);

    setOperationAction(ISD::BR_CC, MVT::i16, Expand);

    setOperationAction(ISD::LOAD, MVT::i16, Custom);

    setTruncStoreAction(MVT::i64, MVT::i16, Expand);

    setOperationAction(ISD::FP16_TO_FP, MVT::i16, Promote);
    AddPromotedToType(ISD::FP16_TO_FP, MVT::i16, MVT::i32);
    setOperationAction(ISD::FP_TO_FP16, MVT::i16, Promote);
    AddPromotedToType(ISD::FP_TO_FP16, MVT::i16, MVT::i32);

    setOperationAction(ISD::FP_TO_SINT, MVT::i16, Promote);
    setOperationAction(ISD::FP_TO_UINT, MVT::i16, Promote);
    setOperationAction(ISD::SINT_TO_FP, MVT::i16, Promote);
    setOperationAction(ISD::UINT_TO_FP, MVT::i16, Promote);

    // F16 - Constant Actions.
    setOperationAction(ISD::ConstantFP, MVT::f16, Legal);

    // F16 - Load/Store Actions.
    setOperationAction(ISD::LOAD, MVT::f16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::f16, MVT::i16);
    setOperationAction(ISD::STORE, MVT::f16, Promote);
    AddPromotedToType(ISD::STORE, MVT::f16, MVT::i16);

    // F16 - VOP1 Actions.
    setOperationAction(ISD::FP_ROUND, MVT::f16, Custom);
    setOperationAction(ISD::FCOS, MVT::f16, Promote);
    setOperationAction(ISD::FSIN, MVT::f16, Promote);
    setOperationAction(ISD::FP_TO_SINT, MVT::f16, Promote);
    setOperationAction(ISD::FP_TO_UINT, MVT::f16, Promote);
    setOperationAction(ISD::SINT_TO_FP, MVT::f16, Promote);
    setOperationAction(ISD::UINT_TO_FP, MVT::f16, Promote);
    setOperationAction(ISD::FROUND, MVT::f16, Custom);

    // F16 - VOP2 Actions.
    setOperationAction(ISD::BR_CC, MVT::f16, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f16, Expand);
    setOperationAction(ISD::FMAXNUM, MVT::f16, Legal);
    setOperationAction(ISD::FMINNUM, MVT::f16, Legal);
    setOperationAction(ISD::FDIV, MVT::f16, Custom);

    // F16 - VOP3 Actions.
    setOperationAction(ISD::FMA, MVT::f16, Legal);
    if (!Subtarget->hasFP16Denormals())
      setOperationAction(ISD::FMAD, MVT::f16, Legal);
  }

  if (Subtarget->hasVOP3PInsts()) {
    for (MVT VT : {MVT::v2i16, MVT::v2f16}) {
      for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op) {
        switch (Op) {
        case ISD::LOAD:
        case ISD::STORE:
        case ISD::BUILD_VECTOR:
        case ISD::BITCAST:
        case ISD::EXTRACT_VECTOR_ELT:
        case ISD::INSERT_VECTOR_ELT:
        case ISD::INSERT_SUBVECTOR:
        case ISD::EXTRACT_SUBVECTOR:
        case ISD::SCALAR_TO_VECTOR:
          break;
        case ISD::CONCAT_VECTORS:
          setOperationAction(Op, VT, Custom);
          break;
        default:
          setOperationAction(Op, VT, Expand);
          break;
        }
      }
    }

    // XXX - Do these do anything? Vector constants turn into build_vector.
    setOperationAction(ISD::Constant, MVT::v2i16, Legal);
    setOperationAction(ISD::ConstantFP, MVT::v2f16, Legal);

    setOperationAction(ISD::STORE, MVT::v2i16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::STORE, MVT::v2f16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v2f16, MVT::i32);

    setOperationAction(ISD::LOAD, MVT::v2i16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::LOAD, MVT::v2f16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::v2f16, MVT::i32);

    setOperationAction(ISD::AND, MVT::v2i16, Promote);
    AddPromotedToType(ISD::AND, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::OR, MVT::v2i16, Promote);
    AddPromotedToType(ISD::OR, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::XOR, MVT::v2i16, Promote);
    AddPromotedToType(ISD::XOR, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::SELECT, MVT::v2i16, Promote);
    AddPromotedToType(ISD::SELECT, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::SELECT, MVT::v2f16, Promote);
    AddPromotedToType(ISD::SELECT, MVT::v2f16, MVT::i32);

    setOperationAction(ISD::ADD, MVT::v2i16, Legal);
    setOperationAction(ISD::SUB, MVT::v2i16, Legal);
    setOperationAction(ISD::MUL, MVT::v2i16, Legal);
    setOperationAction(ISD::SHL, MVT::v2i16, Legal);
    setOperationAction(ISD::SRL, MVT::v2i16, Legal);
    setOperationAction(ISD::SRA, MVT::v2i16, Legal);
    setOperationAction(ISD::SMIN, MVT::v2i16, Legal);
    setOperationAction(ISD::UMIN, MVT::v2i16, Legal);
    setOperationAction(ISD::SMAX, MVT::v2i16, Legal);
    setOperationAction(ISD::UMAX, MVT::v2i16, Legal);

    setOperationAction(ISD::FADD, MVT::v2f16, Legal);
    setOperationAction(ISD::FNEG, MVT::v2f16, Legal);
    setOperationAction(ISD::FMUL, MVT::v2f16, Legal);
    setOperationAction(ISD::FMA, MVT::v2f16, Legal);
    setOperationAction(ISD::FMINNUM, MVT::v2f16, Legal);
    setOperationAction(ISD::FMAXNUM, MVT::v2f16, Legal);

    // This isn't really legal, but this avoids the legalizer unrolling it (and
    // allows matching fneg (fabs x) patterns)
    setOperationAction(ISD::FABS, MVT::v2f16, Legal);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i16, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f16, Custom);

    setOperationAction(ISD::ANY_EXTEND, MVT::v2i32, Expand);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v2i32, Expand);
    setOperationAction(ISD::SIGN_EXTEND, MVT::v2i32, Expand);
    setOperationAction(ISD::FP_EXTEND, MVT::v2f32, Expand);
  } else {
    setOperationAction(ISD::SELECT, MVT::v2i16, Custom);
    setOperationAction(ISD::SELECT, MVT::v2f16, Custom);
  }

  for (MVT VT : { MVT::v4i16, MVT::v4f16, MVT::v2i8, MVT::v4i8, MVT::v8i8 }) {
    setOperationAction(ISD::SELECT, VT, Custom);
  }

  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::ADDCARRY);
  setTargetDAGCombine(ISD::SUB);
  setTargetDAGCombine(ISD::SUBCARRY);
  setTargetDAGCombine(ISD::FADD);
  setTargetDAGCombine(ISD::FSUB);
  setTargetDAGCombine(ISD::FMINNUM);
  setTargetDAGCombine(ISD::FMAXNUM);
  setTargetDAGCombine(ISD::SMIN);
  setTargetDAGCombine(ISD::SMAX);
  setTargetDAGCombine(ISD::UMIN);
  setTargetDAGCombine(ISD::UMAX);
  setTargetDAGCombine(ISD::SETCC);
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::XOR);
  setTargetDAGCombine(ISD::SINT_TO_FP);
  setTargetDAGCombine(ISD::UINT_TO_FP);
  setTargetDAGCombine(ISD::FCANONICALIZE);
  setTargetDAGCombine(ISD::SCALAR_TO_VECTOR);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT);
  setTargetDAGCombine(ISD::BUILD_VECTOR);

  // All memory operations. Some folding on the pointer operand is done to help
  // matching the constant offsets in the addressing modes.
  setTargetDAGCombine(ISD::LOAD);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::ATOMIC_LOAD);
  setTargetDAGCombine(ISD::ATOMIC_STORE);
  setTargetDAGCombine(ISD::ATOMIC_CMP_SWAP);
  setTargetDAGCombine(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS);
  setTargetDAGCombine(ISD::ATOMIC_SWAP);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_ADD);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_SUB);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_AND);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_OR);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_XOR);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_NAND);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_MIN);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_MAX);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_UMIN);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_UMAX);

  setSchedulingPreference(Sched::RegPressure);
}

const SISubtarget *SITargetLowering::getSubtarget() const {
  return static_cast<const SISubtarget *>(Subtarget);
}

//===----------------------------------------------------------------------===//
// TargetLowering queries
//===----------------------------------------------------------------------===//

bool SITargetLowering::isShuffleMaskLegal(ArrayRef<int>, EVT) const {
  // SI has some legal vector types, but no legal vector operations. Say no
  // shuffles are legal in order to prefer scalarizing some vector operations.
  return false;
}

bool SITargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                          const CallInst &CI,
                                          unsigned IntrID) const {
  switch (IntrID) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getType());
    Info.ptrVal = CI.getOperand(0);
    Info.align = 0;

    const ConstantInt *Vol = dyn_cast<ConstantInt>(CI.getOperand(4));
    Info.vol = !Vol || !Vol->isZero();
    Info.readMem = true;
    Info.writeMem = true;
    return true;
  }
  default:
    return false;
  }
}

bool SITargetLowering::getAddrModeArguments(IntrinsicInst *II,
                                            SmallVectorImpl<Value*> &Ops,
                                            Type *&AccessTy) const {
  switch (II->getIntrinsicID()) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec: {
    Value *Ptr = II->getArgOperand(0);
    AccessTy = II->getType();
    Ops.push_back(Ptr);
    return true;
  }
  default:
    return false;
  }
}

bool SITargetLowering::isLegalFlatAddressingMode(const AddrMode &AM) const {
  if (!Subtarget->hasFlatInstOffsets()) {
    // Flat instructions do not have offsets, and only have the register
    // address.
    return AM.BaseOffs == 0 && AM.Scale == 0;
  }

  // GFX9 added a 13-bit signed offset. When using regular flat instructions,
  // the sign bit is ignored and is treated as a 12-bit unsigned offset.

  // Just r + i
  return isUInt<12>(AM.BaseOffs) && AM.Scale == 0;
}

bool SITargetLowering::isLegalGlobalAddressingMode(const AddrMode &AM) const {
  if (Subtarget->hasFlatGlobalInsts())
    return isInt<13>(AM.BaseOffs) && AM.Scale == 0;

  if (!Subtarget->hasAddr64() || Subtarget->useFlatForGlobal()) {
      // Assume the we will use FLAT for all global memory accesses
      // on VI.
      // FIXME: This assumption is currently wrong.  On VI we still use
      // MUBUF instructions for the r + i addressing mode.  As currently
      // implemented, the MUBUF instructions only work on buffer < 4GB.
      // It may be possible to support > 4GB buffers with MUBUF instructions,
      // by setting the stride value in the resource descriptor which would
      // increase the size limit to (stride * 4GB).  However, this is risky,
      // because it has never been validated.
    return isLegalFlatAddressingMode(AM);
  }

  return isLegalMUBUFAddressingMode(AM);
}

bool SITargetLowering::isLegalMUBUFAddressingMode(const AddrMode &AM) const {
  // MUBUF / MTBUF instructions have a 12-bit unsigned byte offset, and
  // additionally can do r + r + i with addr64. 32-bit has more addressing
  // mode options. Depending on the resource constant, it can also do
  // (i64 r0) + (i32 r1) * (i14 i).
  //
  // Private arrays end up using a scratch buffer most of the time, so also
  // assume those use MUBUF instructions. Scratch loads / stores are currently
  // implemented as mubuf instructions with offen bit set, so slightly
  // different than the normal addr64.
  if (!isUInt<12>(AM.BaseOffs))
    return false;

  // FIXME: Since we can split immediate into soffset and immediate offset,
  // would it make sense to allow any immediate?

  switch (AM.Scale) {
  case 0: // r + i or just i, depending on HasBaseReg.
    return true;
  case 1:
    return true; // We have r + r or r + i.
  case 2:
    if (AM.HasBaseReg) {
      // Reject 2 * r + r.
      return false;
    }

    // Allow 2 * r as r + r
    // Or  2 * r + i is allowed as r + r + i.
    return true;
  default: // Don't allow n * r
    return false;
  }
}

bool SITargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                             const AddrMode &AM, Type *Ty,
                                             unsigned AS, Instruction *I) const {
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  if (AS == AMDGPUASI.GLOBAL_ADDRESS)
    return isLegalGlobalAddressingMode(AM);

  if (AS == AMDGPUASI.CONSTANT_ADDRESS) {
    // If the offset isn't a multiple of 4, it probably isn't going to be
    // correctly aligned.
    // FIXME: Can we get the real alignment here?
    if (AM.BaseOffs % 4 != 0)
      return isLegalMUBUFAddressingMode(AM);

    // There are no SMRD extloads, so if we have to do a small type access we
    // will use a MUBUF load.
    // FIXME?: We also need to do this if unaligned, but we don't know the
    // alignment here.
    if (DL.getTypeStoreSize(Ty) < 4)
      return isLegalGlobalAddressingMode(AM);

    if (Subtarget->getGeneration() == SISubtarget::SOUTHERN_ISLANDS) {
      // SMRD instructions have an 8-bit, dword offset on SI.
      if (!isUInt<8>(AM.BaseOffs / 4))
        return false;
    } else if (Subtarget->getGeneration() == SISubtarget::SEA_ISLANDS) {
      // On CI+, this can also be a 32-bit literal constant offset. If it fits
      // in 8-bits, it can use a smaller encoding.
      if (!isUInt<32>(AM.BaseOffs / 4))
        return false;
    } else if (Subtarget->getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
      // On VI, these use the SMEM format and the offset is 20-bit in bytes.
      if (!isUInt<20>(AM.BaseOffs))
        return false;
    } else
      llvm_unreachable("unhandled generation");

    if (AM.Scale == 0) // r + i or just i, depending on HasBaseReg.
      return true;

    if (AM.Scale == 1 && AM.HasBaseReg)
      return true;

    return false;

  } else if (AS == AMDGPUASI.PRIVATE_ADDRESS) {
    return isLegalMUBUFAddressingMode(AM);
  } else if (AS == AMDGPUASI.LOCAL_ADDRESS ||
             AS == AMDGPUASI.REGION_ADDRESS) {
    // Basic, single offset DS instructions allow a 16-bit unsigned immediate
    // field.
    // XXX - If doing a 4-byte aligned 8-byte type access, we effectively have
    // an 8-bit dword offset but we don't know the alignment here.
    if (!isUInt<16>(AM.BaseOffs))
      return false;

    if (AM.Scale == 0) // r + i or just i, depending on HasBaseReg.
      return true;

    if (AM.Scale == 1 && AM.HasBaseReg)
      return true;

    return false;
  } else if (AS == AMDGPUASI.FLAT_ADDRESS ||
             AS == AMDGPUASI.UNKNOWN_ADDRESS_SPACE) {
    // For an unknown address space, this usually means that this is for some
    // reason being used for pure arithmetic, and not based on some addressing
    // computation. We don't have instructions that compute pointers with any
    // addressing modes, so treat them as having no offset like flat
    // instructions.
    return isLegalFlatAddressingMode(AM);
  } else {
    llvm_unreachable("unhandled address space");
  }
}

bool SITargetLowering::canMergeStoresTo(unsigned AS, EVT MemVT,
                                        const SelectionDAG &DAG) const {
  if (AS == AMDGPUASI.GLOBAL_ADDRESS || AS == AMDGPUASI.FLAT_ADDRESS) {
    return (MemVT.getSizeInBits() <= 4 * 32);
  } else if (AS == AMDGPUASI.PRIVATE_ADDRESS) {
    unsigned MaxPrivateBits = 8 * getSubtarget()->getMaxPrivateElementSize();
    return (MemVT.getSizeInBits() <= MaxPrivateBits);
  } else if (AS == AMDGPUASI.LOCAL_ADDRESS) {
    return (MemVT.getSizeInBits() <= 2 * 32);
  }
  return true;
}

bool SITargetLowering::allowsMisalignedMemoryAccesses(EVT VT,
                                                      unsigned AddrSpace,
                                                      unsigned Align,
                                                      bool *IsFast) const {
  if (IsFast)
    *IsFast = false;

  // TODO: I think v3i32 should allow unaligned accesses on CI with DS_READ_B96,
  // which isn't a simple VT.
  // Until MVT is extended to handle this, simply check for the size and
  // rely on the condition below: allow accesses if the size is a multiple of 4.
  if (VT == MVT::Other || (VT != MVT::Other && VT.getSizeInBits() > 1024 &&
                           VT.getStoreSize() > 16)) {
    return false;
  }

  if (AddrSpace == AMDGPUASI.LOCAL_ADDRESS ||
      AddrSpace == AMDGPUASI.REGION_ADDRESS) {
    // ds_read/write_b64 require 8-byte alignment, but we can do a 4 byte
    // aligned, 8 byte access in a single operation using ds_read2/write2_b32
    // with adjacent offsets.
    bool AlignedBy4 = (Align % 4 == 0);
    if (IsFast)
      *IsFast = AlignedBy4;

    return AlignedBy4;
  }

  // FIXME: We have to be conservative here and assume that flat operations
  // will access scratch.  If we had access to the IR function, then we
  // could determine if any private memory was used in the function.
  if (!Subtarget->hasUnalignedScratchAccess() &&
      (AddrSpace == AMDGPUASI.PRIVATE_ADDRESS ||
       AddrSpace == AMDGPUASI.FLAT_ADDRESS)) {
    return false;
  }

  if (Subtarget->hasUnalignedBufferAccess()) {
    // If we have an uniform constant load, it still requires using a slow
    // buffer instruction if unaligned.
    if (IsFast) {
      *IsFast = (AddrSpace == AMDGPUASI.CONSTANT_ADDRESS) ?
        (Align % 4 == 0) : true;
    }

    return true;
  }

  // Smaller than dword value must be aligned.
  if (VT.bitsLT(MVT::i32))
    return false;

  // 8.1.6 - For Dword or larger reads or writes, the two LSBs of the
  // byte-address are ignored, thus forcing Dword alignment.
  // This applies to private, global, and constant memory.
  if (IsFast)
    *IsFast = true;

  return VT.bitsGT(MVT::i32) && Align % 4 == 0;
}

EVT SITargetLowering::getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                                          unsigned SrcAlign, bool IsMemset,
                                          bool ZeroMemset,
                                          bool MemcpyStrSrc,
                                          MachineFunction &MF) const {
  // FIXME: Should account for address space here.

  // The default fallback uses the private pointer size as a guess for a type to
  // use. Make sure we switch these to 64-bit accesses.

  if (Size >= 16 && DstAlign >= 4) // XXX: Should only do for global
    return MVT::v4i32;

  if (Size >= 8 && DstAlign >= 4)
    return MVT::v2i32;

  // Use the default.
  return MVT::Other;
}

static bool isFlatGlobalAddrSpace(unsigned AS, AMDGPUAS AMDGPUASI) {
  return AS == AMDGPUASI.GLOBAL_ADDRESS ||
         AS == AMDGPUASI.FLAT_ADDRESS ||
         AS == AMDGPUASI.CONSTANT_ADDRESS;
}

bool SITargetLowering::isNoopAddrSpaceCast(unsigned SrcAS,
                                           unsigned DestAS) const {
  return isFlatGlobalAddrSpace(SrcAS, AMDGPUASI) &&
         isFlatGlobalAddrSpace(DestAS, AMDGPUASI);
}

bool SITargetLowering::isMemOpHasNoClobberedMemOperand(const SDNode *N) const {
  const MemSDNode *MemNode = cast<MemSDNode>(N);
  const Value *Ptr = MemNode->getMemOperand()->getValue();
  const Instruction *I = dyn_cast<Instruction>(Ptr);
  return I && I->getMetadata("amdgpu.noclobber");
}

bool SITargetLowering::isCheapAddrSpaceCast(unsigned SrcAS,
                                            unsigned DestAS) const {
  // Flat -> private/local is a simple truncate.
  // Flat -> global is no-op
  if (SrcAS == AMDGPUASI.FLAT_ADDRESS)
    return true;

  return isNoopAddrSpaceCast(SrcAS, DestAS);
}

bool SITargetLowering::isMemOpUniform(const SDNode *N) const {
  const MemSDNode *MemNode = cast<MemSDNode>(N);

  return AMDGPU::isUniformMMO(MemNode->getMemOperand());
}

TargetLoweringBase::LegalizeTypeAction
SITargetLowering::getPreferredVectorAction(EVT VT) const {
  if (VT.getVectorNumElements() != 1 && VT.getScalarType().bitsLE(MVT::i16))
    return TypeSplitVector;

  return TargetLoweringBase::getPreferredVectorAction(VT);
}

bool SITargetLowering::shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                                         Type *Ty) const {
  // FIXME: Could be smarter if called for vector constants.
  return true;
}

bool SITargetLowering::isTypeDesirableForOp(unsigned Op, EVT VT) const {
  if (Subtarget->has16BitInsts() && VT == MVT::i16) {
    switch (Op) {
    case ISD::LOAD:
    case ISD::STORE:

    // These operations are done with 32-bit instructions anyway.
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
    case ISD::SELECT:
      // TODO: Extensions?
      return true;
    default:
      return false;
    }
  }

  // SimplifySetCC uses this function to determine whether or not it should
  // create setcc with i1 operands.  We don't have instructions for i1 setcc.
  if (VT == MVT::i1 && Op == ISD::SETCC)
    return false;

  return TargetLowering::isTypeDesirableForOp(Op, VT);
}

SDValue SITargetLowering::lowerKernArgParameterPtr(SelectionDAG &DAG,
                                                   const SDLoc &SL,
                                                   SDValue Chain,
                                                   uint64_t Offset) const {
  const DataLayout &DL = DAG.getDataLayout();
  MachineFunction &MF = DAG.getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  const ArgDescriptor *InputPtrReg;
  const TargetRegisterClass *RC;

  std::tie(InputPtrReg, RC)
    = Info->getPreloadedValue(AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);

  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
  MVT PtrVT = getPointerTy(DL, AMDGPUASI.CONSTANT_ADDRESS);
  SDValue BasePtr = DAG.getCopyFromReg(Chain, SL,
    MRI.getLiveInVirtReg(InputPtrReg->getRegister()), PtrVT);

  return DAG.getNode(ISD::ADD, SL, PtrVT, BasePtr,
                     DAG.getConstant(Offset, SL, PtrVT));
}

SDValue SITargetLowering::getImplicitArgPtr(SelectionDAG &DAG,
                                            const SDLoc &SL) const {
  auto MFI = DAG.getMachineFunction().getInfo<SIMachineFunctionInfo>();
  uint64_t Offset = getImplicitParameterOffset(MFI, FIRST_IMPLICIT);
  return lowerKernArgParameterPtr(DAG, SL, DAG.getEntryNode(), Offset);
}

SDValue SITargetLowering::convertArgType(SelectionDAG &DAG, EVT VT, EVT MemVT,
                                         const SDLoc &SL, SDValue Val,
                                         bool Signed,
                                         const ISD::InputArg *Arg) const {
  if (Arg && (Arg->Flags.isSExt() || Arg->Flags.isZExt()) &&
      VT.bitsLT(MemVT)) {
    unsigned Opc = Arg->Flags.isZExt() ? ISD::AssertZext : ISD::AssertSext;
    Val = DAG.getNode(Opc, SL, MemVT, Val, DAG.getValueType(VT));
  }

  if (MemVT.isFloatingPoint())
    Val = getFPExtOrFPTrunc(DAG, Val, SL, VT);
  else if (Signed)
    Val = DAG.getSExtOrTrunc(Val, SL, VT);
  else
    Val = DAG.getZExtOrTrunc(Val, SL, VT);

  return Val;
}

SDValue SITargetLowering::lowerKernargMemParameter(
  SelectionDAG &DAG, EVT VT, EVT MemVT,
  const SDLoc &SL, SDValue Chain,
  uint64_t Offset, bool Signed,
  const ISD::InputArg *Arg) const {
  const DataLayout &DL = DAG.getDataLayout();
  Type *Ty = MemVT.getTypeForEVT(*DAG.getContext());
  PointerType *PtrTy = PointerType::get(Ty, AMDGPUASI.CONSTANT_ADDRESS);
  MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));

  unsigned Align = DL.getABITypeAlignment(Ty);

  SDValue Ptr = lowerKernArgParameterPtr(DAG, SL, Chain, Offset);
  SDValue Load = DAG.getLoad(MemVT, SL, Chain, Ptr, PtrInfo, Align,
                             MachineMemOperand::MONonTemporal |
                             MachineMemOperand::MODereferenceable |
                             MachineMemOperand::MOInvariant);

  SDValue Val = convertArgType(DAG, VT, MemVT, SL, Load, Signed, Arg);
  return DAG.getMergeValues({ Val, Load.getValue(1) }, SL);
}

SDValue SITargetLowering::lowerStackParameter(SelectionDAG &DAG, CCValAssign &VA,
                                              const SDLoc &SL, SDValue Chain,
                                              const ISD::InputArg &Arg) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  if (Arg.Flags.isByVal()) {
    unsigned Size = Arg.Flags.getByValSize();
    int FrameIdx = MFI.CreateFixedObject(Size, VA.getLocMemOffset(), false);
    return DAG.getFrameIndex(FrameIdx, MVT::i32);
  }

  unsigned ArgOffset = VA.getLocMemOffset();
  unsigned ArgSize = VA.getValVT().getStoreSize();

  int FI = MFI.CreateFixedObject(ArgSize, ArgOffset, true);

  // Create load nodes to retrieve arguments from the stack.
  SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
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
    ExtType, SL, VA.getLocVT(), Chain, FIN,
    MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI),
    MemVT);
  return ArgValue;
}

SDValue SITargetLowering::getPreloadedValue(SelectionDAG &DAG,
  const SIMachineFunctionInfo &MFI,
  EVT VT,
  AMDGPUFunctionArgInfo::PreloadedValue PVID) const {
  const ArgDescriptor *Reg;
  const TargetRegisterClass *RC;

  std::tie(Reg, RC) = MFI.getPreloadedValue(PVID);
  return CreateLiveInRegister(DAG, RC, Reg->getRegister(), VT);
}

static void processShaderInputArgs(SmallVectorImpl<ISD::InputArg> &Splits,
                                   CallingConv::ID CallConv,
                                   ArrayRef<ISD::InputArg> Ins,
                                   BitVector &Skipped,
                                   FunctionType *FType,
                                   SIMachineFunctionInfo *Info) {
  for (unsigned I = 0, E = Ins.size(), PSInputNum = 0; I != E; ++I) {
    const ISD::InputArg &Arg = Ins[I];

    // First check if it's a PS input addr.
    if (CallConv == CallingConv::AMDGPU_PS && !Arg.Flags.isInReg() &&
        !Arg.Flags.isByVal() && PSInputNum <= 15) {

      if (!Arg.Used && !Info->isPSInputAllocated(PSInputNum)) {
        // We can safely skip PS inputs.
        Skipped.set(I);
        ++PSInputNum;
        continue;
      }

      Info->markPSInputAllocated(PSInputNum);
      if (Arg.Used)
        Info->markPSInputEnabled(PSInputNum);

      ++PSInputNum;
    }

    // Second split vertices into their elements.
    if (Arg.VT.isVector()) {
      ISD::InputArg NewArg = Arg;
      NewArg.Flags.setSplit();
      NewArg.VT = Arg.VT.getVectorElementType();

      // We REALLY want the ORIGINAL number of vertex elements here, e.g. a
      // three or five element vertex only needs three or five registers,
      // NOT four or eight.
      Type *ParamType = FType->getParamType(Arg.getOrigArgIndex());
      unsigned NumElements = ParamType->getVectorNumElements();

      for (unsigned J = 0; J != NumElements; ++J) {
        Splits.push_back(NewArg);
        NewArg.PartOffset += NewArg.VT.getStoreSize();
      }
    } else {
      Splits.push_back(Arg);
    }
  }
}

// Allocate special inputs passed in VGPRs.
static void allocateSpecialEntryInputVGPRs(CCState &CCInfo,
                                           MachineFunction &MF,
                                           const SIRegisterInfo &TRI,
                                           SIMachineFunctionInfo &Info) {
  if (Info.hasWorkItemIDX()) {
    unsigned Reg = AMDGPU::VGPR0;
    MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass);

    CCInfo.AllocateReg(Reg);
    Info.setWorkItemIDX(ArgDescriptor::createRegister(Reg));
  }

  if (Info.hasWorkItemIDY()) {
    unsigned Reg = AMDGPU::VGPR1;
    MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass);

    CCInfo.AllocateReg(Reg);
    Info.setWorkItemIDY(ArgDescriptor::createRegister(Reg));
  }

  if (Info.hasWorkItemIDZ()) {
    unsigned Reg = AMDGPU::VGPR2;
    MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass);

    CCInfo.AllocateReg(Reg);
    Info.setWorkItemIDZ(ArgDescriptor::createRegister(Reg));
  }
}

// Try to allocate a VGPR at the end of the argument list, or if no argument
// VGPRs are left allocating a stack slot.
static ArgDescriptor allocateVGPR32Input(CCState &CCInfo) {
  ArrayRef<MCPhysReg> ArgVGPRs
    = makeArrayRef(AMDGPU::VGPR_32RegClass.begin(), 32);
  unsigned RegIdx = CCInfo.getFirstUnallocated(ArgVGPRs);
  if (RegIdx == ArgVGPRs.size()) {
    // Spill to stack required.
    int64_t Offset = CCInfo.AllocateStack(4, 4);

    return ArgDescriptor::createStack(Offset);
  }

  unsigned Reg = ArgVGPRs[RegIdx];
  Reg = CCInfo.AllocateReg(Reg);
  assert(Reg != AMDGPU::NoRegister);

  MachineFunction &MF = CCInfo.getMachineFunction();
  MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass);
  return ArgDescriptor::createRegister(Reg);
}

static ArgDescriptor allocateSGPR32InputImpl(CCState &CCInfo,
                                             const TargetRegisterClass *RC,
                                             unsigned NumArgRegs) {
  ArrayRef<MCPhysReg> ArgSGPRs = makeArrayRef(RC->begin(), 32);
  unsigned RegIdx = CCInfo.getFirstUnallocated(ArgSGPRs);
  if (RegIdx == ArgSGPRs.size())
    report_fatal_error("ran out of SGPRs for arguments");

  unsigned Reg = ArgSGPRs[RegIdx];
  Reg = CCInfo.AllocateReg(Reg);
  assert(Reg != AMDGPU::NoRegister);

  MachineFunction &MF = CCInfo.getMachineFunction();
  MF.addLiveIn(Reg, RC);
  return ArgDescriptor::createRegister(Reg);
}

static ArgDescriptor allocateSGPR32Input(CCState &CCInfo) {
  return allocateSGPR32InputImpl(CCInfo, &AMDGPU::SGPR_32RegClass, 32);
}

static ArgDescriptor allocateSGPR64Input(CCState &CCInfo) {
  return allocateSGPR32InputImpl(CCInfo, &AMDGPU::SGPR_64RegClass, 16);
}

static void allocateSpecialInputVGPRs(CCState &CCInfo,
                                      MachineFunction &MF,
                                      const SIRegisterInfo &TRI,
                                      SIMachineFunctionInfo &Info) {
  if (Info.hasWorkItemIDX())
    Info.setWorkItemIDX(allocateVGPR32Input(CCInfo));

  if (Info.hasWorkItemIDY())
    Info.setWorkItemIDY(allocateVGPR32Input(CCInfo));

  if (Info.hasWorkItemIDZ())
    Info.setWorkItemIDZ(allocateVGPR32Input(CCInfo));
}

static void allocateSpecialInputSGPRs(CCState &CCInfo,
                                      MachineFunction &MF,
                                      const SIRegisterInfo &TRI,
                                      SIMachineFunctionInfo &Info) {
  auto &ArgInfo = Info.getArgInfo();

  // TODO: Unify handling with private memory pointers.

  if (Info.hasDispatchPtr())
    ArgInfo.DispatchPtr = allocateSGPR64Input(CCInfo);

  if (Info.hasQueuePtr())
    ArgInfo.QueuePtr = allocateSGPR64Input(CCInfo);

  if (Info.hasKernargSegmentPtr())
    ArgInfo.KernargSegmentPtr = allocateSGPR64Input(CCInfo);

  if (Info.hasDispatchID())
    ArgInfo.DispatchID = allocateSGPR64Input(CCInfo);

  // flat_scratch_init is not applicable for non-kernel functions.

  if (Info.hasWorkGroupIDX())
    ArgInfo.WorkGroupIDX = allocateSGPR32Input(CCInfo);

  if (Info.hasWorkGroupIDY())
    ArgInfo.WorkGroupIDY = allocateSGPR32Input(CCInfo);

  if (Info.hasWorkGroupIDZ())
    ArgInfo.WorkGroupIDZ = allocateSGPR32Input(CCInfo);

  if (Info.hasImplicitArgPtr())
    ArgInfo.ImplicitArgPtr = allocateSGPR64Input(CCInfo);
}

// Allocate special inputs passed in user SGPRs.
static void allocateHSAUserSGPRs(CCState &CCInfo,
                                 MachineFunction &MF,
                                 const SIRegisterInfo &TRI,
                                 SIMachineFunctionInfo &Info) {
  if (Info.hasImplicitBufferPtr()) {
    unsigned ImplicitBufferPtrReg = Info.addImplicitBufferPtr(TRI);
    MF.addLiveIn(ImplicitBufferPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(ImplicitBufferPtrReg);
  }

  // FIXME: How should these inputs interact with inreg / custom SGPR inputs?
  if (Info.hasPrivateSegmentBuffer()) {
    unsigned PrivateSegmentBufferReg = Info.addPrivateSegmentBuffer(TRI);
    MF.addLiveIn(PrivateSegmentBufferReg, &AMDGPU::SGPR_128RegClass);
    CCInfo.AllocateReg(PrivateSegmentBufferReg);
  }

  if (Info.hasDispatchPtr()) {
    unsigned DispatchPtrReg = Info.addDispatchPtr(TRI);
    MF.addLiveIn(DispatchPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(DispatchPtrReg);
  }

  if (Info.hasQueuePtr()) {
    unsigned QueuePtrReg = Info.addQueuePtr(TRI);
    MF.addLiveIn(QueuePtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(QueuePtrReg);
  }

  if (Info.hasKernargSegmentPtr()) {
    unsigned InputPtrReg = Info.addKernargSegmentPtr(TRI);
    MF.addLiveIn(InputPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(InputPtrReg);
  }

  if (Info.hasDispatchID()) {
    unsigned DispatchIDReg = Info.addDispatchID(TRI);
    MF.addLiveIn(DispatchIDReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(DispatchIDReg);
  }

  if (Info.hasFlatScratchInit()) {
    unsigned FlatScratchInitReg = Info.addFlatScratchInit(TRI);
    MF.addLiveIn(FlatScratchInitReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(FlatScratchInitReg);
  }

  // TODO: Add GridWorkGroupCount user SGPRs when used. For now with HSA we read
  // these from the dispatch pointer.
}

// Allocate special input registers that are initialized per-wave.
static void allocateSystemSGPRs(CCState &CCInfo,
                                MachineFunction &MF,
                                SIMachineFunctionInfo &Info,
                                CallingConv::ID CallConv,
                                bool IsShader) {
  if (Info.hasWorkGroupIDX()) {
    unsigned Reg = Info.addWorkGroupIDX();
    MF.addLiveIn(Reg, &AMDGPU::SReg_32_XM0RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupIDY()) {
    unsigned Reg = Info.addWorkGroupIDY();
    MF.addLiveIn(Reg, &AMDGPU::SReg_32_XM0RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupIDZ()) {
    unsigned Reg = Info.addWorkGroupIDZ();
    MF.addLiveIn(Reg, &AMDGPU::SReg_32_XM0RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupInfo()) {
    unsigned Reg = Info.addWorkGroupInfo();
    MF.addLiveIn(Reg, &AMDGPU::SReg_32_XM0RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasPrivateSegmentWaveByteOffset()) {
    // Scratch wave offset passed in system SGPR.
    unsigned PrivateSegmentWaveByteOffsetReg;

    if (IsShader) {
      PrivateSegmentWaveByteOffsetReg =
        Info.getPrivateSegmentWaveByteOffsetSystemSGPR();

      // This is true if the scratch wave byte offset doesn't have a fixed
      // location.
      if (PrivateSegmentWaveByteOffsetReg == AMDGPU::NoRegister) {
        PrivateSegmentWaveByteOffsetReg = findFirstFreeSGPR(CCInfo);
        Info.setPrivateSegmentWaveByteOffset(PrivateSegmentWaveByteOffsetReg);
      }
    } else
      PrivateSegmentWaveByteOffsetReg = Info.addPrivateSegmentWaveByteOffset();

    MF.addLiveIn(PrivateSegmentWaveByteOffsetReg, &AMDGPU::SGPR_32RegClass);
    CCInfo.AllocateReg(PrivateSegmentWaveByteOffsetReg);
  }
}

static void reservePrivateMemoryRegs(const TargetMachine &TM,
                                     MachineFunction &MF,
                                     const SIRegisterInfo &TRI,
                                     SIMachineFunctionInfo &Info) {
  // Now that we've figured out where the scratch register inputs are, see if
  // should reserve the arguments and use them directly.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool HasStackObjects = MFI.hasStackObjects();

  // Record that we know we have non-spill stack objects so we don't need to
  // check all stack objects later.
  if (HasStackObjects)
    Info.setHasNonSpillStackObjects(true);

  // Everything live out of a block is spilled with fast regalloc, so it's
  // almost certain that spilling will be required.
  if (TM.getOptLevel() == CodeGenOpt::None)
    HasStackObjects = true;

  // For now assume stack access is needed in any callee functions, so we need
  // the scratch registers to pass in.
  bool RequiresStackAccess = HasStackObjects || MFI.hasCalls();

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  if (ST.isAmdCodeObjectV2(MF)) {
    if (RequiresStackAccess) {
      // If we have stack objects, we unquestionably need the private buffer
      // resource. For the Code Object V2 ABI, this will be the first 4 user
      // SGPR inputs. We can reserve those and use them directly.

      unsigned PrivateSegmentBufferReg = Info.getPreloadedReg(
        AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER);
      Info.setScratchRSrcReg(PrivateSegmentBufferReg);

      if (MFI.hasCalls()) {
        // If we have calls, we need to keep the frame register in a register
        // that won't be clobbered by a call, so ensure it is copied somewhere.

        // This is not a problem for the scratch wave offset, because the same
        // registers are reserved in all functions.

        // FIXME: Nothing is really ensuring this is a call preserved register,
        // it's just selected from the end so it happens to be.
        unsigned ReservedOffsetReg
          = TRI.reservedPrivateSegmentWaveByteOffsetReg(MF);
        Info.setScratchWaveOffsetReg(ReservedOffsetReg);
      } else {
        unsigned PrivateSegmentWaveByteOffsetReg = Info.getPreloadedReg(
          AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);
        Info.setScratchWaveOffsetReg(PrivateSegmentWaveByteOffsetReg);
      }
    } else {
      unsigned ReservedBufferReg
        = TRI.reservedPrivateSegmentBufferReg(MF);
      unsigned ReservedOffsetReg
        = TRI.reservedPrivateSegmentWaveByteOffsetReg(MF);

      // We tentatively reserve the last registers (skipping the last two
      // which may contain VCC). After register allocation, we'll replace
      // these with the ones immediately after those which were really
      // allocated. In the prologue copies will be inserted from the argument
      // to these reserved registers.
      Info.setScratchRSrcReg(ReservedBufferReg);
      Info.setScratchWaveOffsetReg(ReservedOffsetReg);
    }
  } else {
    unsigned ReservedBufferReg = TRI.reservedPrivateSegmentBufferReg(MF);

    // Without HSA, relocations are used for the scratch pointer and the
    // buffer resource setup is always inserted in the prologue. Scratch wave
    // offset is still in an input SGPR.
    Info.setScratchRSrcReg(ReservedBufferReg);

    if (HasStackObjects && !MFI.hasCalls()) {
      unsigned ScratchWaveOffsetReg = Info.getPreloadedReg(
        AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);
      Info.setScratchWaveOffsetReg(ScratchWaveOffsetReg);
    } else {
      unsigned ReservedOffsetReg
        = TRI.reservedPrivateSegmentWaveByteOffsetReg(MF);
      Info.setScratchWaveOffsetReg(ReservedOffsetReg);
    }
  }
}

bool SITargetLowering::supportSplitCSR(MachineFunction *MF) const {
  const SIMachineFunctionInfo *Info = MF->getInfo<SIMachineFunctionInfo>();
  return !Info->isEntryFunction();
}

void SITargetLowering::initializeSplitCSR(MachineBasicBlock *Entry) const {

}

void SITargetLowering::insertCopiesSplitCSR(
  MachineBasicBlock *Entry,
  const SmallVectorImpl<MachineBasicBlock *> &Exits) const {
  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();

  const MCPhysReg *IStart = TRI->getCalleeSavedRegsViaCopy(Entry->getParent());
  if (!IStart)
    return;

  const TargetInstrInfo *TII = Subtarget->getInstrInfo();
  MachineRegisterInfo *MRI = &Entry->getParent()->getRegInfo();
  MachineBasicBlock::iterator MBBI = Entry->begin();
  for (const MCPhysReg *I = IStart; *I; ++I) {
    const TargetRegisterClass *RC = nullptr;
    if (AMDGPU::SReg_64RegClass.contains(*I))
      RC = &AMDGPU::SGPR_64RegClass;
    else if (AMDGPU::SReg_32RegClass.contains(*I))
      RC = &AMDGPU::SGPR_32RegClass;
    else
      llvm_unreachable("Unexpected register class in CSRsViaCopy!");

    unsigned NewVR = MRI->createVirtualRegister(RC);
    // Create copy from CSR to a virtual register.
    Entry->addLiveIn(*I);
    BuildMI(*Entry, MBBI, DebugLoc(), TII->get(TargetOpcode::COPY), NewVR)
      .addReg(*I);

    // Insert the copy-back instructions right before the terminator.
    for (auto *Exit : Exits)
      BuildMI(*Exit, Exit->getFirstTerminator(), DebugLoc(),
              TII->get(TargetOpcode::COPY), *I)
        .addReg(NewVR);
  }
}

SDValue SITargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();

  MachineFunction &MF = DAG.getMachineFunction();
  FunctionType *FType = MF.getFunction()->getFunctionType();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();

  if (Subtarget->isAmdHsaOS() && AMDGPU::isShader(CallConv)) {
    const Function *Fn = MF.getFunction();
    DiagnosticInfoUnsupported NoGraphicsHSA(
        *Fn, "unsupported non-compute shaders with HSA", DL.getDebugLoc());
    DAG.getContext()->diagnose(NoGraphicsHSA);
    return DAG.getEntryNode();
  }

  // Create stack objects that are used for emitting debugger prologue if
  // "amdgpu-debugger-emit-prologue" attribute was specified.
  if (ST.debuggerEmitPrologue())
    createDebuggerPrologueStackObjects(MF);

  SmallVector<ISD::InputArg, 16> Splits;
  SmallVector<CCValAssign, 16> ArgLocs;
  BitVector Skipped(Ins.size());
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  bool IsShader = AMDGPU::isShader(CallConv);
  bool IsKernel = AMDGPU::isKernel(CallConv);
  bool IsEntryFunc = AMDGPU::isEntryFunctionCC(CallConv);

  if (!IsEntryFunc) {
    // 4 bytes are reserved at offset 0 for the emergency stack slot. Skip over
    // this when allocating argument fixed offsets.
    CCInfo.AllocateStack(4, 4);
  }

  if (IsShader) {
    processShaderInputArgs(Splits, CallConv, Ins, Skipped, FType, Info);

    // At least one interpolation mode must be enabled or else the GPU will
    // hang.
    //
    // Check PSInputAddr instead of PSInputEnable. The idea is that if the user
    // set PSInputAddr, the user wants to enable some bits after the compilation
    // based on run-time states. Since we can't know what the final PSInputEna
    // will look like, so we shouldn't do anything here and the user should take
    // responsibility for the correct programming.
    //
    // Otherwise, the following restrictions apply:
    // - At least one of PERSP_* (0xF) or LINEAR_* (0x70) must be enabled.
    // - If POS_W_FLOAT (11) is enabled, at least one of PERSP_* must be
    //   enabled too.
    if (CallConv == CallingConv::AMDGPU_PS) {
      if ((Info->getPSInputAddr() & 0x7F) == 0 ||
           ((Info->getPSInputAddr() & 0xF) == 0 &&
            Info->isPSInputAllocated(11))) {
        CCInfo.AllocateReg(AMDGPU::VGPR0);
        CCInfo.AllocateReg(AMDGPU::VGPR1);
        Info->markPSInputAllocated(0);
        Info->markPSInputEnabled(0);
      }
      if (Subtarget->isAmdPalOS()) {
        // For isAmdPalOS, the user does not enable some bits after compilation
        // based on run-time states; the register values being generated here are
        // the final ones set in hardware. Therefore we need to apply the
        // workaround to PSInputAddr and PSInputEnable together.  (The case where
        // a bit is set in PSInputAddr but not PSInputEnable is where the
        // frontend set up an input arg for a particular interpolation mode, but
        // nothing uses that input arg. Really we should have an earlier pass
        // that removes such an arg.)
        unsigned PsInputBits = Info->getPSInputAddr() & Info->getPSInputEnable();
        if ((PsInputBits & 0x7F) == 0 ||
            ((PsInputBits & 0xF) == 0 &&
             (PsInputBits >> 11 & 1)))
          Info->markPSInputEnabled(
              countTrailingZeros(Info->getPSInputAddr(), ZB_Undefined));
      }
    }

    assert(!Info->hasDispatchPtr() &&
           !Info->hasKernargSegmentPtr() && !Info->hasFlatScratchInit() &&
           !Info->hasWorkGroupIDX() && !Info->hasWorkGroupIDY() &&
           !Info->hasWorkGroupIDZ() && !Info->hasWorkGroupInfo() &&
           !Info->hasWorkItemIDX() && !Info->hasWorkItemIDY() &&
           !Info->hasWorkItemIDZ());
  } else if (IsKernel) {
    assert(Info->hasWorkGroupIDX() && Info->hasWorkItemIDX());
  } else {
    Splits.append(Ins.begin(), Ins.end());
  }

  if (IsEntryFunc) {
    allocateSpecialEntryInputVGPRs(CCInfo, MF, *TRI, *Info);
    allocateHSAUserSGPRs(CCInfo, MF, *TRI, *Info);
  }

  if (IsKernel) {
    analyzeFormalArgumentsCompute(CCInfo, Ins);
  } else {
    CCAssignFn *AssignFn = CCAssignFnForCall(CallConv, isVarArg);
    CCInfo.AnalyzeFormalArguments(Splits, AssignFn);
  }

  SmallVector<SDValue, 16> Chains;

  for (unsigned i = 0, e = Ins.size(), ArgIdx = 0; i != e; ++i) {
    const ISD::InputArg &Arg = Ins[i];
    if (Skipped[i]) {
      InVals.push_back(DAG.getUNDEF(Arg.VT));
      continue;
    }

    CCValAssign &VA = ArgLocs[ArgIdx++];
    MVT VT = VA.getLocVT();

    if (IsEntryFunc && VA.isMemLoc()) {
      VT = Ins[i].VT;
      EVT MemVT = VA.getLocVT();

      const uint64_t Offset = Subtarget->getExplicitKernelArgOffset(MF) +
        VA.getLocMemOffset();
      Info->setABIArgOffset(Offset + MemVT.getStoreSize());

      // The first 36 bytes of the input buffer contains information about
      // thread group and global sizes.
      SDValue Arg = lowerKernargMemParameter(
        DAG, VT, MemVT, DL, Chain, Offset, Ins[i].Flags.isSExt(), &Ins[i]);
      Chains.push_back(Arg.getValue(1));

      auto *ParamTy =
        dyn_cast<PointerType>(FType->getParamType(Ins[i].getOrigArgIndex()));
      if (Subtarget->getGeneration() == SISubtarget::SOUTHERN_ISLANDS &&
          ParamTy && ParamTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
        // On SI local pointers are just offsets into LDS, so they are always
        // less than 16-bits.  On CI and newer they could potentially be
        // real pointers, so we can't guarantee their size.
        Arg = DAG.getNode(ISD::AssertZext, DL, Arg.getValueType(), Arg,
                          DAG.getValueType(MVT::i16));
      }

      InVals.push_back(Arg);
      continue;
    } else if (!IsEntryFunc && VA.isMemLoc()) {
      SDValue Val = lowerStackParameter(DAG, VA, DL, Chain, Arg);
      InVals.push_back(Val);
      if (!Arg.Flags.isByVal())
        Chains.push_back(Val.getValue(1));
      continue;
    }

    assert(VA.isRegLoc() && "Parameter must be in a register!");

    unsigned Reg = VA.getLocReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg, VT);
    EVT ValVT = VA.getValVT();

    Reg = MF.addLiveIn(Reg, RC);
    SDValue Val = DAG.getCopyFromReg(Chain, DL, Reg, VT);

    if (Arg.Flags.isSRet() && !getSubtarget()->enableHugePrivateBuffer()) {
      // The return object should be reasonably addressable.

      // FIXME: This helps when the return is a real sret. If it is a
      // automatically inserted sret (i.e. CanLowerReturn returns false), an
      // extra copy is inserted in SelectionDAGBuilder which obscures this.
      unsigned NumBits = 32 - AssumeFrameIndexHighZeroBits;
      Val = DAG.getNode(ISD::AssertZext, DL, VT, Val,
        DAG.getValueType(EVT::getIntegerVT(*DAG.getContext(), NumBits)));
    }

    // If this is an 8 or 16-bit value, it is really passed promoted
    // to 32 bits. Insert an assert[sz]ext to capture this, then
    // truncate to the right size.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, ValVT, Val);
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::AssertSext, DL, VT, Val,
                        DAG.getValueType(ValVT));
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::AssertZext, DL, VT, Val,
                        DAG.getValueType(ValVT));
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    if (IsShader && Arg.VT.isVector()) {
      // Build a vector from the registers
      Type *ParamType = FType->getParamType(Arg.getOrigArgIndex());
      unsigned NumElements = ParamType->getVectorNumElements();

      SmallVector<SDValue, 4> Regs;
      Regs.push_back(Val);
      for (unsigned j = 1; j != NumElements; ++j) {
        Reg = ArgLocs[ArgIdx++].getLocReg();
        Reg = MF.addLiveIn(Reg, RC);

        SDValue Copy = DAG.getCopyFromReg(Chain, DL, Reg, VT);
        Regs.push_back(Copy);
      }

      // Fill up the missing vector elements
      NumElements = Arg.VT.getVectorNumElements() - NumElements;
      Regs.append(NumElements, DAG.getUNDEF(VT));

      InVals.push_back(DAG.getBuildVector(Arg.VT, DL, Regs));
      continue;
    }

    InVals.push_back(Val);
  }

  if (!IsEntryFunc) {
    // Special inputs come after user arguments.
    allocateSpecialInputVGPRs(CCInfo, MF, *TRI, *Info);
  }

  // Start adding system SGPRs.
  if (IsEntryFunc) {
    allocateSystemSGPRs(CCInfo, MF, *Info, CallConv, IsShader);
  } else {
    CCInfo.AllocateReg(Info->getScratchRSrcReg());
    CCInfo.AllocateReg(Info->getScratchWaveOffsetReg());
    CCInfo.AllocateReg(Info->getFrameOffsetReg());
    allocateSpecialInputSGPRs(CCInfo, MF, *TRI, *Info);
  }

  auto &ArgUsageInfo =
    DAG.getPass()->getAnalysis<AMDGPUArgumentUsageInfo>();
  ArgUsageInfo.setFuncArgInfo(*MF.getFunction(), Info->getArgInfo());

  unsigned StackArgSize = CCInfo.getNextStackOffset();
  Info->setBytesInStackArgArea(StackArgSize);

  return Chains.empty() ? Chain :
    DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
}

// TODO: If return values can't fit in registers, we should return as many as
// possible in registers before passing on stack.
bool SITargetLowering::CanLowerReturn(
  CallingConv::ID CallConv,
  MachineFunction &MF, bool IsVarArg,
  const SmallVectorImpl<ISD::OutputArg> &Outs,
  LLVMContext &Context) const {
  // Replacing returns with sret/stack usage doesn't make sense for shaders.
  // FIXME: Also sort of a workaround for custom vector splitting in LowerReturn
  // for shaders. Vector types should be explicitly handled by CC.
  if (AMDGPU::isEntryFunctionCC(CallConv))
    return true;

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, CCAssignFnForReturn(CallConv, IsVarArg));
}

SDValue
SITargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals,
                              const SDLoc &DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  if (AMDGPU::isKernel(CallConv)) {
    return AMDGPUTargetLowering::LowerReturn(Chain, CallConv, isVarArg, Outs,
                                             OutVals, DL, DAG);
  }

  bool IsShader = AMDGPU::isShader(CallConv);

  Info->setIfReturnsVoid(Outs.size() == 0);
  bool IsWaveEnd = Info->returnsVoid() && IsShader;

  SmallVector<ISD::OutputArg, 48> Splits;
  SmallVector<SDValue, 48> SplitVals;

  // Split vectors into their elements.
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    const ISD::OutputArg &Out = Outs[i];

    if (IsShader && Out.VT.isVector()) {
      MVT VT = Out.VT.getVectorElementType();
      ISD::OutputArg NewOut = Out;
      NewOut.Flags.setSplit();
      NewOut.VT = VT;

      // We want the original number of vector elements here, e.g.
      // three or five, not four or eight.
      unsigned NumElements = Out.ArgVT.getVectorNumElements();

      for (unsigned j = 0; j != NumElements; ++j) {
        SDValue Elem = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, OutVals[i],
                                   DAG.getConstant(j, DL, MVT::i32));
        SplitVals.push_back(Elem);
        Splits.push_back(NewOut);
        NewOut.PartOffset += NewOut.VT.getStoreSize();
      }
    } else {
      SplitVals.push_back(OutVals[i]);
      Splits.push_back(Out);
    }
  }

  // CCValAssign - represent the assignment of the return value to a location.
  SmallVector<CCValAssign, 48> RVLocs;

  // CCState - Info about the registers and stack slots.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  // Analyze outgoing return values.
  CCInfo.AnalyzeReturn(Splits, CCAssignFnForReturn(CallConv, isVarArg));

  SDValue Flag;
  SmallVector<SDValue, 48> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)

  // Add return address for callable functions.
  if (!Info->isEntryFunction()) {
    const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();
    SDValue ReturnAddrReg = CreateLiveInRegister(
      DAG, &AMDGPU::SReg_64RegClass, TRI->getReturnAddressReg(MF), MVT::i64);

    // FIXME: Should be able to use a vreg here, but need a way to prevent it
    // from being allcoated to a CSR.

    SDValue PhysReturnAddrReg = DAG.getRegister(TRI->getReturnAddressReg(MF),
                                                MVT::i64);

    Chain = DAG.getCopyToReg(Chain, DL, PhysReturnAddrReg, ReturnAddrReg, Flag);
    Flag = Chain.getValue(1);

    RetOps.push_back(PhysReturnAddrReg);
  }

  // Copy the result values into the output registers.
  for (unsigned i = 0, realRVLocIdx = 0;
       i != RVLocs.size();
       ++i, ++realRVLocIdx) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    // TODO: Partially return in registers if return values don't fit.

    SDValue Arg = SplitVals[realRVLocIdx];

    // Copied from other backends.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Arg, Flag);
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  // FIXME: Does sret work properly?
  if (!Info->isEntryFunction()) {
    const SIRegisterInfo *TRI
      = static_cast<const SISubtarget *>(Subtarget)->getRegisterInfo();
    const MCPhysReg *I =
      TRI->getCalleeSavedRegsViaCopy(&DAG.getMachineFunction());
    if (I) {
      for (; *I; ++I) {
        if (AMDGPU::SReg_64RegClass.contains(*I))
          RetOps.push_back(DAG.getRegister(*I, MVT::i64));
        else if (AMDGPU::SReg_32RegClass.contains(*I))
          RetOps.push_back(DAG.getRegister(*I, MVT::i32));
        else
          llvm_unreachable("Unexpected register class in CSRsViaCopy!");
      }
    }
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Flag.getNode())
    RetOps.push_back(Flag);

  unsigned Opc = AMDGPUISD::ENDPGM;
  if (!IsWaveEnd)
    Opc = IsShader ? AMDGPUISD::RETURN_TO_EPILOG : AMDGPUISD::RET_FLAG;
  return DAG.getNode(Opc, DL, MVT::Other, RetOps);
}

SDValue SITargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals, bool IsThisReturn,
    SDValue ThisVal) const {
  CCAssignFn *RetCC = CCAssignFnForReturn(CallConv, IsVarArg);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];
    SDValue Val;

    if (VA.isRegLoc()) {
      Val = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), InFlag);
      Chain = Val.getValue(1);
      InFlag = Val.getValue(2);
    } else if (VA.isMemLoc()) {
      report_fatal_error("TODO: return values in memory");
    } else
      llvm_unreachable("unknown argument location type");

    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), Val,
                        DAG.getValueType(VA.getValVT()));
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), Val,
                        DAG.getValueType(VA.getValVT()));
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    InVals.push_back(Val);
  }

  return Chain;
}

// Add code to pass special inputs required depending on used features separate
// from the explicit user arguments present in the IR.
void SITargetLowering::passSpecialInputs(
    CallLoweringInfo &CLI,
    const SIMachineFunctionInfo &Info,
    SmallVectorImpl<std::pair<unsigned, SDValue>> &RegsToPass,
    SmallVectorImpl<SDValue> &MemOpChains,
    SDValue Chain,
    SDValue StackPtr) const {
  // If we don't have a call site, this was a call inserted by
  // legalization. These can never use special inputs.
  if (!CLI.CS)
    return;

  const Function *CalleeFunc = CLI.CS.getCalledFunction();
  assert(CalleeFunc);

  SelectionDAG &DAG = CLI.DAG;
  const SDLoc &DL = CLI.DL;

  const SISubtarget *ST = getSubtarget();
  const SIRegisterInfo *TRI = ST->getRegisterInfo();

  auto &ArgUsageInfo =
    DAG.getPass()->getAnalysis<AMDGPUArgumentUsageInfo>();
  const AMDGPUFunctionArgInfo &CalleeArgInfo
    = ArgUsageInfo.lookupFuncArgInfo(*CalleeFunc);

  const AMDGPUFunctionArgInfo &CallerArgInfo = Info.getArgInfo();

  // TODO: Unify with private memory register handling. This is complicated by
  // the fact that at least in kernels, the input argument is not necessarily
  // in the same location as the input.
  AMDGPUFunctionArgInfo::PreloadedValue InputRegs[] = {
    AMDGPUFunctionArgInfo::DISPATCH_PTR,
    AMDGPUFunctionArgInfo::QUEUE_PTR,
    AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR,
    AMDGPUFunctionArgInfo::DISPATCH_ID,
    AMDGPUFunctionArgInfo::WORKGROUP_ID_X,
    AMDGPUFunctionArgInfo::WORKGROUP_ID_Y,
    AMDGPUFunctionArgInfo::WORKGROUP_ID_Z,
    AMDGPUFunctionArgInfo::WORKITEM_ID_X,
    AMDGPUFunctionArgInfo::WORKITEM_ID_Y,
    AMDGPUFunctionArgInfo::WORKITEM_ID_Z,
    AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR
  };

  for (auto InputID : InputRegs) {
    const ArgDescriptor *OutgoingArg;
    const TargetRegisterClass *ArgRC;

    std::tie(OutgoingArg, ArgRC) = CalleeArgInfo.getPreloadedValue(InputID);
    if (!OutgoingArg)
      continue;

    const ArgDescriptor *IncomingArg;
    const TargetRegisterClass *IncomingArgRC;
    std::tie(IncomingArg, IncomingArgRC)
      = CallerArgInfo.getPreloadedValue(InputID);
    assert(IncomingArgRC == ArgRC);

    // All special arguments are ints for now.
    EVT ArgVT = TRI->getSpillSize(*ArgRC) == 8 ? MVT::i64 : MVT::i32;
    SDValue InputReg;

    if (IncomingArg) {
      InputReg = loadInputValue(DAG, ArgRC, ArgVT, DL, *IncomingArg);
    } else {
      // The implicit arg ptr is special because it doesn't have a corresponding
      // input for kernels, and is computed from the kernarg segment pointer.
      assert(InputID == AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR);
      InputReg = getImplicitArgPtr(DAG, DL);
    }

    if (OutgoingArg->isRegister()) {
      RegsToPass.emplace_back(OutgoingArg->getRegister(), InputReg);
    } else {
      SDValue ArgStore = storeStackInputValue(DAG, DL, Chain, StackPtr,
                                              InputReg,
                                              OutgoingArg->getStackOffset());
      MemOpChains.push_back(ArgStore);
    }
  }
}

static bool canGuaranteeTCO(CallingConv::ID CC) {
  return CC == CallingConv::Fast;
}

/// Return true if we might ever do TCO for calls with this calling convention.
static bool mayTailCallThisCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::C:
    return true;
  default:
    return canGuaranteeTCO(CC);
  }
}

bool SITargetLowering::isEligibleForTailCallOptimization(
    SDValue Callee, CallingConv::ID CalleeCC, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG &DAG) const {
  if (!mayTailCallThisCC(CalleeCC))
    return false;

  MachineFunction &MF = DAG.getMachineFunction();
  const Function *CallerF = MF.getFunction();
  CallingConv::ID CallerCC = CallerF->getCallingConv();
  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);

  // Kernels aren't callable, and don't have a live in return address so it
  // doesn't make sense to do a tail call with entry functions.
  if (!CallerPreserved)
    return false;

  bool CCMatch = CallerCC == CalleeCC;

  if (DAG.getTarget().Options.GuaranteedTailCallOpt) {
    if (canGuaranteeTCO(CalleeCC) && CCMatch)
      return true;
    return false;
  }

  // TODO: Can we handle var args?
  if (IsVarArg)
    return false;

  for (const Argument &Arg : CallerF->args()) {
    if (Arg.hasByValAttr())
      return false;
  }

  LLVMContext &Ctx = *DAG.getContext();

  // Check that the call results are passed in the same way.
  if (!CCState::resultsCompatible(CalleeCC, CallerCC, MF, Ctx, Ins,
                                  CCAssignFnForCall(CalleeCC, IsVarArg),
                                  CCAssignFnForCall(CallerCC, IsVarArg)))
    return false;

  // The callee has to preserve all registers the caller needs to preserve.
  if (!CCMatch) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }

  // Nothing more to check if the callee is taking no arguments.
  if (Outs.empty())
    return true;

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CalleeCC, IsVarArg, MF, ArgLocs, Ctx);

  CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForCall(CalleeCC, IsVarArg));

  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  // If the stack arguments for this call do not fit into our own save area then
  // the call cannot be made tail.
  // TODO: Is this really necessary?
  if (CCInfo.getNextStackOffset() > FuncInfo->getBytesInStackArgArea())
    return false;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  return parametersInCSRMatch(MRI, CallerPreserved, ArgLocs, OutVals);
}

bool SITargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  if (!CI->isTailCall())
    return false;

  const Function *ParentFn = CI->getParent()->getParent();
  if (AMDGPU::isEntryFunctionCC(ParentFn->getCallingConv()))
    return false;

  auto Attr = ParentFn->getFnAttribute("disable-tail-calls");
  return (Attr.getValueAsString() != "true");
}

// The wave scratch offset register is used as the global base pointer.
SDValue SITargetLowering::LowerCall(CallLoweringInfo &CLI,
                                    SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  const SDLoc &DL = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  bool IsSibCall = false;
  bool IsThisReturn = false;
  MachineFunction &MF = DAG.getMachineFunction();

  if (IsVarArg) {
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported call to variadic function ");
  }

  if (!CLI.CS.getCalledFunction()) {
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported indirect call to function ");
  }

  if (IsTailCall && MF.getTarget().Options.GuaranteedTailCallOpt) {
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported required tail call to function ");
  }

  // The first 4 bytes are reserved for the callee's emergency stack slot.
  const unsigned CalleeUsableStackOffset = 4;

  if (IsTailCall) {
    IsTailCall = isEligibleForTailCallOptimization(
      Callee, CallConv, IsVarArg, Outs, OutVals, Ins, DAG);
    if (!IsTailCall && CLI.CS && CLI.CS.isMustTailCall()) {
      report_fatal_error("failed to perform tail call elimination on a call "
                         "site marked musttail");
    }

    bool TailCallOpt = MF.getTarget().Options.GuaranteedTailCallOpt;

    // A sibling call is one where we're under the usual C ABI and not planning
    // to change that but can still do a tail call:
    if (!TailCallOpt && IsTailCall)
      IsSibCall = true;

    if (IsTailCall)
      ++NumTailCalls;
  }

  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Callee)) {
    // FIXME: Remove this hack for function pointer types after removing
    // support of old address space mapping. In the new address space
    // mapping the pointer in default address space is 64 bit, therefore
    // does not need this hack.
    if (Callee.getValueType() == MVT::i32) {
      const GlobalValue *GV = GA->getGlobal();
      Callee = DAG.getGlobalAddress(GV, DL, MVT::i64, GA->getOffset(), false,
                                    GA->getTargetFlags());
    }
  }
  assert(Callee.getValueType() == MVT::i64);

  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCAssignFn *AssignFn = CCAssignFnForCall(CallConv, IsVarArg);
  CCInfo.AnalyzeCallOperands(Outs, AssignFn);

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
  int32_t FPDiff = 0;
  MachineFrameInfo &MFI = MF.getFrameInfo();
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;

  SDValue CallerSavedFP;

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  if (!IsSibCall) {
    Chain = DAG.getCALLSEQ_START(Chain, 0, 0, DL);

    unsigned OffsetReg = Info->getScratchWaveOffsetReg();

    // In the HSA case, this should be an identity copy.
    SDValue ScratchRSrcReg
      = DAG.getCopyFromReg(Chain, DL, Info->getScratchRSrcReg(), MVT::v4i32);
    RegsToPass.emplace_back(AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, ScratchRSrcReg);

    // TODO: Don't hardcode these registers and get from the callee function.
    SDValue ScratchWaveOffsetReg
      = DAG.getCopyFromReg(Chain, DL, OffsetReg, MVT::i32);
    RegsToPass.emplace_back(AMDGPU::SGPR4, ScratchWaveOffsetReg);

    if (!Info->isEntryFunction()) {
      // Avoid clobbering this function's FP value. In the current convention
      // callee will overwrite this, so do save/restore around the call site.
      CallerSavedFP = DAG.getCopyFromReg(Chain, DL,
                                         Info->getFrameOffsetReg(), MVT::i32);
    }
  }

  // Stack pointer relative accesses are done by changing the offset SGPR. This
  // is just the VGPR offset component.
  SDValue StackPtr = DAG.getConstant(CalleeUsableStackOffset, DL, MVT::i32);

  SmallVector<SDValue, 8> MemOpChains;
  MVT PtrVT = MVT::i32;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, realArgIdx = 0, e = ArgLocs.size(); i != e;
       ++i, ++realArgIdx) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[realArgIdx];

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::FPExt:
      Arg = DAG.getNode(ISD::FP_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());

      SDValue DstAddr;
      MachinePointerInfo DstInfo;

      unsigned LocMemOffset = VA.getLocMemOffset();
      int32_t Offset = LocMemOffset;

      SDValue PtrOff = DAG.getObjectPtrOffset(DL, StackPtr, Offset);

      if (IsTailCall) {
        ISD::ArgFlagsTy Flags = Outs[realArgIdx].Flags;
        unsigned OpSize = Flags.isByVal() ?
          Flags.getByValSize() : VA.getValVT().getStoreSize();

        Offset = Offset + FPDiff;
        int FI = MFI.CreateFixedObject(OpSize, Offset, true);

        DstAddr = DAG.getObjectPtrOffset(DL, DAG.getFrameIndex(FI, PtrVT),
                                         StackPtr);
        DstInfo = MachinePointerInfo::getFixedStack(MF, FI);

        // Make sure any stack arguments overlapping with where we're storing
        // are loaded before this eventual operation. Otherwise they'll be
        // clobbered.

        // FIXME: Why is this really necessary? This seems to just result in a
        // lot of code to copy the stack and write them back to the same
        // locations, which are supposed to be immutable?
        Chain = addTokenForArgument(Chain, DAG, MFI, FI);
      } else {
        DstAddr = PtrOff;
        DstInfo = MachinePointerInfo::getStack(MF, LocMemOffset);
      }

      if (Outs[i].Flags.isByVal()) {
        SDValue SizeNode =
            DAG.getConstant(Outs[i].Flags.getByValSize(), DL, MVT::i32);
        SDValue Cpy = DAG.getMemcpy(
            Chain, DL, DstAddr, Arg, SizeNode, Outs[i].Flags.getByValAlign(),
            /*isVol = */ false, /*AlwaysInline = */ true,
            /*isTailCall = */ false, DstInfo,
            MachinePointerInfo(UndefValue::get(Type::getInt8PtrTy(
                *DAG.getContext(), AMDGPUASI.PRIVATE_ADDRESS))));

        MemOpChains.push_back(Cpy);
      } else {
        SDValue Store = DAG.getStore(Chain, DL, Arg, DstAddr, DstInfo);
        MemOpChains.push_back(Store);
      }
    }
  }

  // Copy special input registers after user input arguments.
  passSpecialInputs(CLI, *Info, RegsToPass, MemOpChains, Chain, StackPtr);

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


  SDValue PhysReturnAddrReg;
  if (IsTailCall) {
    // Since the return is being combined with the call, we need to pass on the
    // return address.

    const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();
    SDValue ReturnAddrReg = CreateLiveInRegister(
      DAG, &AMDGPU::SReg_64RegClass, TRI->getReturnAddressReg(MF), MVT::i64);

    PhysReturnAddrReg = DAG.getRegister(TRI->getReturnAddressReg(MF),
                                        MVT::i64);
    Chain = DAG.getCopyToReg(Chain, DL, PhysReturnAddrReg, ReturnAddrReg, InFlag);
    InFlag = Chain.getValue(1);
  }

  // We don't usually want to end the call-sequence here because we would tidy
  // the frame up *after* the call, however in the ABI-changing tail-call case
  // we've carefully laid out the parameters so that when sp is reset they'll be
  // in the correct location.
  if (IsTailCall && !IsSibCall) {
    Chain = DAG.getCALLSEQ_END(Chain,
                               DAG.getTargetConstant(NumBytes, DL, MVT::i32),
                               DAG.getTargetConstant(0, DL, MVT::i32),
                               InFlag, DL);
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

    Ops.push_back(PhysReturnAddrReg);
  }

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (auto &RegToPass : RegsToPass) {
    Ops.push_back(DAG.getRegister(RegToPass.first,
                                  RegToPass.second.getValueType()));
  }

  // Add a register mask operand representing the call-preserved registers.

  const AMDGPURegisterInfo *TRI = Subtarget->getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  // If we're doing a tall call, use a TC_RETURN here rather than an
  // actual call instruction.
  if (IsTailCall) {
    MFI.setHasTailCall();
    return DAG.getNode(AMDGPUISD::TC_RETURN, DL, NodeTys, Ops);
  }

  // Returns a chain and a flag for retval copy to use.
  SDValue Call = DAG.getNode(AMDGPUISD::CALL, DL, NodeTys, Ops);
  Chain = Call.getValue(0);
  InFlag = Call.getValue(1);

  if (CallerSavedFP) {
    SDValue FPReg = DAG.getRegister(Info->getFrameOffsetReg(), MVT::i32);
    Chain = DAG.getCopyToReg(Chain, DL, FPReg, CallerSavedFP, InFlag);
    InFlag = Chain.getValue(1);
  }

  uint64_t CalleePopBytes = NumBytes;
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getTargetConstant(0, DL, MVT::i32),
                             DAG.getTargetConstant(CalleePopBytes, DL, MVT::i32),
                             InFlag, DL);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, IsVarArg, Ins, DL, DAG,
                         InVals, IsThisReturn,
                         IsThisReturn ? OutVals[0] : SDValue());
}

unsigned SITargetLowering::getRegisterByName(const char* RegName, EVT VT,
                                             SelectionDAG &DAG) const {
  unsigned Reg = StringSwitch<unsigned>(RegName)
    .Case("m0", AMDGPU::M0)
    .Case("exec", AMDGPU::EXEC)
    .Case("exec_lo", AMDGPU::EXEC_LO)
    .Case("exec_hi", AMDGPU::EXEC_HI)
    .Case("flat_scratch", AMDGPU::FLAT_SCR)
    .Case("flat_scratch_lo", AMDGPU::FLAT_SCR_LO)
    .Case("flat_scratch_hi", AMDGPU::FLAT_SCR_HI)
    .Default(AMDGPU::NoRegister);

  if (Reg == AMDGPU::NoRegister) {
    report_fatal_error(Twine("invalid register name \""
                             + StringRef(RegName)  + "\"."));

  }

  if (Subtarget->getGeneration() == SISubtarget::SOUTHERN_ISLANDS &&
      Subtarget->getRegisterInfo()->regsOverlap(Reg, AMDGPU::FLAT_SCR)) {
    report_fatal_error(Twine("invalid register \""
                             + StringRef(RegName)  + "\" for subtarget."));
  }

  switch (Reg) {
  case AMDGPU::M0:
  case AMDGPU::EXEC_LO:
  case AMDGPU::EXEC_HI:
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
    if (VT.getSizeInBits() == 32)
      return Reg;
    break;
  case AMDGPU::EXEC:
  case AMDGPU::FLAT_SCR:
    if (VT.getSizeInBits() == 64)
      return Reg;
    break;
  default:
    llvm_unreachable("missing register type checking");
  }

  report_fatal_error(Twine("invalid type for register \""
                           + StringRef(RegName) + "\"."));
}

// If kill is not the last instruction, split the block so kill is always a
// proper terminator.
MachineBasicBlock *SITargetLowering::splitKillBlock(MachineInstr &MI,
                                                    MachineBasicBlock *BB) const {
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

  MachineBasicBlock::iterator SplitPoint(&MI);
  ++SplitPoint;

  if (SplitPoint == BB->end()) {
    // Don't bother with a new block.
    MI.setDesc(TII->getKillTerminatorFromPseudo(MI.getOpcode()));
    return BB;
  }

  MachineFunction *MF = BB->getParent();
  MachineBasicBlock *SplitBB
    = MF->CreateMachineBasicBlock(BB->getBasicBlock());

  MF->insert(++MachineFunction::iterator(BB), SplitBB);
  SplitBB->splice(SplitBB->begin(), BB, SplitPoint, BB->end());

  SplitBB->transferSuccessorsAndUpdatePHIs(BB);
  BB->addSuccessor(SplitBB);

  MI.setDesc(TII->getKillTerminatorFromPseudo(MI.getOpcode()));
  return SplitBB;
}

// Do a v_movrels_b32 or v_movreld_b32 for each unique value of \p IdxReg in the
// wavefront. If the value is uniform and just happens to be in a VGPR, this
// will only do one iteration. In the worst case, this will loop 64 times.
//
// TODO: Just use v_readlane_b32 if we know the VGPR has a uniform value.
static MachineBasicBlock::iterator emitLoadM0FromVGPRLoop(
  const SIInstrInfo *TII,
  MachineRegisterInfo &MRI,
  MachineBasicBlock &OrigBB,
  MachineBasicBlock &LoopBB,
  const DebugLoc &DL,
  const MachineOperand &IdxReg,
  unsigned InitReg,
  unsigned ResultReg,
  unsigned PhiReg,
  unsigned InitSaveExecReg,
  int Offset,
  bool UseGPRIdxMode) {
  MachineBasicBlock::iterator I = LoopBB.begin();

  unsigned PhiExec = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
  unsigned NewExec = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
  unsigned CurrentIdxReg = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  unsigned CondReg = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);

  BuildMI(LoopBB, I, DL, TII->get(TargetOpcode::PHI), PhiReg)
    .addReg(InitReg)
    .addMBB(&OrigBB)
    .addReg(ResultReg)
    .addMBB(&LoopBB);

  BuildMI(LoopBB, I, DL, TII->get(TargetOpcode::PHI), PhiExec)
    .addReg(InitSaveExecReg)
    .addMBB(&OrigBB)
    .addReg(NewExec)
    .addMBB(&LoopBB);

  // Read the next variant <- also loop target.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), CurrentIdxReg)
    .addReg(IdxReg.getReg(), getUndefRegState(IdxReg.isUndef()));

  // Compare the just read M0 value to all possible Idx values.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::V_CMP_EQ_U32_e64), CondReg)
    .addReg(CurrentIdxReg)
    .addReg(IdxReg.getReg(), 0, IdxReg.getSubReg());

  if (UseGPRIdxMode) {
    unsigned IdxReg;
    if (Offset == 0) {
      IdxReg = CurrentIdxReg;
    } else {
      IdxReg = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
      BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_ADD_I32), IdxReg)
        .addReg(CurrentIdxReg, RegState::Kill)
        .addImm(Offset);
    }

    MachineInstr *SetIdx =
      BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_SET_GPR_IDX_IDX))
      .addReg(IdxReg, RegState::Kill);
    SetIdx->getOperand(2).setIsUndef();
  } else {
    // Move index from VCC into M0
    if (Offset == 0) {
      BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
        .addReg(CurrentIdxReg, RegState::Kill);
    } else {
      BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
        .addReg(CurrentIdxReg, RegState::Kill)
        .addImm(Offset);
    }
  }

  // Update EXEC, save the original EXEC value to VCC.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_AND_SAVEEXEC_B64), NewExec)
    .addReg(CondReg, RegState::Kill);

  MRI.setSimpleHint(NewExec, CondReg);

  // Update EXEC, switch all done bits to 0 and all todo bits to 1.
  MachineInstr *InsertPt =
    BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_XOR_B64), AMDGPU::EXEC)
    .addReg(AMDGPU::EXEC)
    .addReg(NewExec);

  // XXX - s_xor_b64 sets scc to 1 if the result is nonzero, so can we use
  // s_cbranch_scc0?

  // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
    .addMBB(&LoopBB);

  return InsertPt->getIterator();
}

// This has slightly sub-optimal regalloc when the source vector is killed by
// the read. The register allocator does not understand that the kill is
// per-workitem, so is kept alive for the whole loop so we end up not re-using a
// subregister from it, using 1 more VGPR than necessary. This was saved when
// this was expanded after register allocation.
static MachineBasicBlock::iterator loadM0FromVGPR(const SIInstrInfo *TII,
                                                  MachineBasicBlock &MBB,
                                                  MachineInstr &MI,
                                                  unsigned InitResultReg,
                                                  unsigned PhiReg,
                                                  int Offset,
                                                  bool UseGPRIdxMode) {
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned SaveExec = MRI.createVirtualRegister(&AMDGPU::SReg_64_XEXECRegClass);
  unsigned TmpExec = MRI.createVirtualRegister(&AMDGPU::SReg_64_XEXECRegClass);

  BuildMI(MBB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), TmpExec);

  // Save the EXEC mask
  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B64), SaveExec)
    .addReg(AMDGPU::EXEC);

  // To insert the loop we need to split the block. Move everything after this
  // point to a new block, and insert a new empty block between the two.
  MachineBasicBlock *LoopBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *RemainderBB = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;

  MF->insert(MBBI, LoopBB);
  MF->insert(MBBI, RemainderBB);

  LoopBB->addSuccessor(LoopBB);
  LoopBB->addSuccessor(RemainderBB);

  // Move the rest of the block into a new block.
  RemainderBB->transferSuccessorsAndUpdatePHIs(&MBB);
  RemainderBB->splice(RemainderBB->begin(), &MBB, I, MBB.end());

  MBB.addSuccessor(LoopBB);

  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);

  auto InsPt = emitLoadM0FromVGPRLoop(TII, MRI, MBB, *LoopBB, DL, *Idx,
                                      InitResultReg, DstReg, PhiReg, TmpExec,
                                      Offset, UseGPRIdxMode);

  MachineBasicBlock::iterator First = RemainderBB->begin();
  BuildMI(*RemainderBB, First, DL, TII->get(AMDGPU::S_MOV_B64), AMDGPU::EXEC)
    .addReg(SaveExec);

  return InsPt;
}

// Returns subreg index, offset
static std::pair<unsigned, int>
computeIndirectRegAndOffset(const SIRegisterInfo &TRI,
                            const TargetRegisterClass *SuperRC,
                            unsigned VecReg,
                            int Offset) {
  int NumElts = TRI.getRegSizeInBits(*SuperRC) / 32;

  // Skip out of bounds offsets, or else we would end up using an undefined
  // register.
  if (Offset >= NumElts || Offset < 0)
    return std::make_pair(AMDGPU::sub0, Offset);

  return std::make_pair(AMDGPU::sub0 + Offset, 0);
}

// Return true if the index is an SGPR and was set.
static bool setM0ToIndexFromSGPR(const SIInstrInfo *TII,
                                 MachineRegisterInfo &MRI,
                                 MachineInstr &MI,
                                 int Offset,
                                 bool UseGPRIdxMode,
                                 bool IsIndirectSrc) {
  MachineBasicBlock *MBB = MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
  const TargetRegisterClass *IdxRC = MRI.getRegClass(Idx->getReg());

  assert(Idx->getReg() != AMDGPU::NoRegister);

  if (!TII->getRegisterInfo().isSGPRClass(IdxRC))
    return false;

  if (UseGPRIdxMode) {
    unsigned IdxMode = IsIndirectSrc ?
      VGPRIndexMode::SRC0_ENABLE : VGPRIndexMode::DST_ENABLE;
    if (Offset == 0) {
      MachineInstr *SetOn =
          BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_SET_GPR_IDX_ON))
              .add(*Idx)
              .addImm(IdxMode);

      SetOn->getOperand(3).setIsUndef();
    } else {
      unsigned Tmp = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
      BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_ADD_I32), Tmp)
          .add(*Idx)
          .addImm(Offset);
      MachineInstr *SetOn =
        BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_SET_GPR_IDX_ON))
        .addReg(Tmp, RegState::Kill)
        .addImm(IdxMode);

      SetOn->getOperand(3).setIsUndef();
    }

    return true;
  }

  if (Offset == 0) {
    BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
      .add(*Idx);
  } else {
    BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
      .add(*Idx)
      .addImm(Offset);
  }

  return true;
}

// Control flow needs to be inserted if indexing with a VGPR.
static MachineBasicBlock *emitIndirectSrc(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          const SISubtarget &ST) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned SrcReg = TII->getNamedOperand(MI, AMDGPU::OpName::src)->getReg();
  int Offset = TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm();

  const TargetRegisterClass *VecRC = MRI.getRegClass(SrcReg);

  unsigned SubReg;
  std::tie(SubReg, Offset)
    = computeIndirectRegAndOffset(TRI, VecRC, SrcReg, Offset);

  bool UseGPRIdxMode = ST.useVGPRIndexMode(EnableVGPRIndexMode);

  if (setM0ToIndexFromSGPR(TII, MRI, MI, Offset, UseGPRIdxMode, true)) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    if (UseGPRIdxMode) {
      // TODO: Look at the uses to avoid the copy. This may require rescheduling
      // to avoid interfering with other uses, so probably requires a new
      // optimization pass.
      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MOV_B32_e32), Dst)
        .addReg(SrcReg, RegState::Undef, SubReg)
        .addReg(SrcReg, RegState::Implicit)
        .addReg(AMDGPU::M0, RegState::Implicit);
      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_SET_GPR_IDX_OFF));
    } else {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MOVRELS_B32_e32), Dst)
        .addReg(SrcReg, RegState::Undef, SubReg)
        .addReg(SrcReg, RegState::Implicit);
    }

    MI.eraseFromParent();

    return &MBB;
  }

  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  unsigned PhiReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  unsigned InitReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  BuildMI(MBB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), InitReg);

  if (UseGPRIdxMode) {
    MachineInstr *SetOn = BuildMI(MBB, I, DL, TII->get(AMDGPU::S_SET_GPR_IDX_ON))
      .addImm(0) // Reset inside loop.
      .addImm(VGPRIndexMode::SRC0_ENABLE);
    SetOn->getOperand(3).setIsUndef();

    // Disable again after the loop.
    BuildMI(MBB, std::next(I), DL, TII->get(AMDGPU::S_SET_GPR_IDX_OFF));
  }

  auto InsPt = loadM0FromVGPR(TII, MBB, MI, InitReg, PhiReg, Offset, UseGPRIdxMode);
  MachineBasicBlock *LoopBB = InsPt->getParent();

  if (UseGPRIdxMode) {
    BuildMI(*LoopBB, InsPt, DL, TII->get(AMDGPU::V_MOV_B32_e32), Dst)
      .addReg(SrcReg, RegState::Undef, SubReg)
      .addReg(SrcReg, RegState::Implicit)
      .addReg(AMDGPU::M0, RegState::Implicit);
  } else {
    BuildMI(*LoopBB, InsPt, DL, TII->get(AMDGPU::V_MOVRELS_B32_e32), Dst)
      .addReg(SrcReg, RegState::Undef, SubReg)
      .addReg(SrcReg, RegState::Implicit);
  }

  MI.eraseFromParent();

  return LoopBB;
}

static unsigned getMOVRELDPseudo(const SIRegisterInfo &TRI,
                                 const TargetRegisterClass *VecRC) {
  switch (TRI.getRegSizeInBits(*VecRC)) {
  case 32: // 4 bytes
    return AMDGPU::V_MOVRELD_B32_V1;
  case 64: // 8 bytes
    return AMDGPU::V_MOVRELD_B32_V2;
  case 128: // 16 bytes
    return AMDGPU::V_MOVRELD_B32_V4;
  case 256: // 32 bytes
    return AMDGPU::V_MOVRELD_B32_V8;
  case 512: // 64 bytes
    return AMDGPU::V_MOVRELD_B32_V16;
  default:
    llvm_unreachable("unsupported size for MOVRELD pseudos");
  }
}

static MachineBasicBlock *emitIndirectDst(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          const SISubtarget &ST) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  unsigned Dst = MI.getOperand(0).getReg();
  const MachineOperand *SrcVec = TII->getNamedOperand(MI, AMDGPU::OpName::src);
  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
  const MachineOperand *Val = TII->getNamedOperand(MI, AMDGPU::OpName::val);
  int Offset = TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm();
  const TargetRegisterClass *VecRC = MRI.getRegClass(SrcVec->getReg());

  // This can be an immediate, but will be folded later.
  assert(Val->getReg());

  unsigned SubReg;
  std::tie(SubReg, Offset) = computeIndirectRegAndOffset(TRI, VecRC,
                                                         SrcVec->getReg(),
                                                         Offset);
  bool UseGPRIdxMode = ST.useVGPRIndexMode(EnableVGPRIndexMode);

  if (Idx->getReg() == AMDGPU::NoRegister) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    assert(Offset == 0);

    BuildMI(MBB, I, DL, TII->get(TargetOpcode::INSERT_SUBREG), Dst)
        .add(*SrcVec)
        .add(*Val)
        .addImm(SubReg);

    MI.eraseFromParent();
    return &MBB;
  }

  if (setM0ToIndexFromSGPR(TII, MRI, MI, Offset, UseGPRIdxMode, false)) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    if (UseGPRIdxMode) {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MOV_B32_indirect))
          .addReg(SrcVec->getReg(), RegState::Undef, SubReg) // vdst
          .add(*Val)
          .addReg(Dst, RegState::ImplicitDefine)
          .addReg(SrcVec->getReg(), RegState::Implicit)
          .addReg(AMDGPU::M0, RegState::Implicit);

      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_SET_GPR_IDX_OFF));
    } else {
      const MCInstrDesc &MovRelDesc = TII->get(getMOVRELDPseudo(TRI, VecRC));

      BuildMI(MBB, I, DL, MovRelDesc)
          .addReg(Dst, RegState::Define)
          .addReg(SrcVec->getReg())
          .add(*Val)
          .addImm(SubReg - AMDGPU::sub0);
    }

    MI.eraseFromParent();
    return &MBB;
  }

  if (Val->isReg())
    MRI.clearKillFlags(Val->getReg());

  const DebugLoc &DL = MI.getDebugLoc();

  if (UseGPRIdxMode) {
    MachineBasicBlock::iterator I(&MI);

    MachineInstr *SetOn = BuildMI(MBB, I, DL, TII->get(AMDGPU::S_SET_GPR_IDX_ON))
      .addImm(0) // Reset inside loop.
      .addImm(VGPRIndexMode::DST_ENABLE);
    SetOn->getOperand(3).setIsUndef();

    // Disable again after the loop.
    BuildMI(MBB, std::next(I), DL, TII->get(AMDGPU::S_SET_GPR_IDX_OFF));
  }

  unsigned PhiReg = MRI.createVirtualRegister(VecRC);

  auto InsPt = loadM0FromVGPR(TII, MBB, MI, SrcVec->getReg(), PhiReg,
                              Offset, UseGPRIdxMode);
  MachineBasicBlock *LoopBB = InsPt->getParent();

  if (UseGPRIdxMode) {
    BuildMI(*LoopBB, InsPt, DL, TII->get(AMDGPU::V_MOV_B32_indirect))
        .addReg(PhiReg, RegState::Undef, SubReg) // vdst
        .add(*Val)                               // src0
        .addReg(Dst, RegState::ImplicitDefine)
        .addReg(PhiReg, RegState::Implicit)
        .addReg(AMDGPU::M0, RegState::Implicit);
  } else {
    const MCInstrDesc &MovRelDesc = TII->get(getMOVRELDPseudo(TRI, VecRC));

    BuildMI(*LoopBB, InsPt, DL, MovRelDesc)
        .addReg(Dst, RegState::Define)
        .addReg(PhiReg)
        .add(*Val)
        .addImm(SubReg - AMDGPU::sub0);
  }

  MI.eraseFromParent();

  return LoopBB;
}

MachineBasicBlock *SITargetLowering::EmitInstrWithCustomInserter(
  MachineInstr &MI, MachineBasicBlock *BB) const {

  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  MachineFunction *MF = BB->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();

  if (TII->isMIMG(MI)) {
      if (!MI.memoperands_empty())
        return BB;
    // Add a memoperand for mimg instructions so that they aren't assumed to
    // be ordered memory instuctions.

    MachinePointerInfo PtrInfo(MFI->getImagePSV());
    MachineMemOperand::Flags Flags = MachineMemOperand::MODereferenceable;
    if (MI.mayStore())
      Flags |= MachineMemOperand::MOStore;

    if (MI.mayLoad())
      Flags |= MachineMemOperand::MOLoad;

    auto MMO = MF->getMachineMemOperand(PtrInfo, Flags, 0, 0);
    MI.addMemOperand(*MF, MMO);
    return BB;
  }

  switch (MI.getOpcode()) {
  case AMDGPU::S_ADD_U64_PSEUDO:
  case AMDGPU::S_SUB_U64_PSEUDO: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    const DebugLoc &DL = MI.getDebugLoc();

    MachineOperand &Dest = MI.getOperand(0);
    MachineOperand &Src0 = MI.getOperand(1);
    MachineOperand &Src1 = MI.getOperand(2);

    unsigned DestSub0 = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
    unsigned DestSub1 = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);

    MachineOperand Src0Sub0 = TII->buildExtractSubRegOrImm(MI, MRI,
     Src0, &AMDGPU::SReg_64RegClass, AMDGPU::sub0,
     &AMDGPU::SReg_32_XM0RegClass);
    MachineOperand Src0Sub1 = TII->buildExtractSubRegOrImm(MI, MRI,
      Src0, &AMDGPU::SReg_64RegClass, AMDGPU::sub1,
      &AMDGPU::SReg_32_XM0RegClass);

    MachineOperand Src1Sub0 = TII->buildExtractSubRegOrImm(MI, MRI,
      Src1, &AMDGPU::SReg_64RegClass, AMDGPU::sub0,
      &AMDGPU::SReg_32_XM0RegClass);
    MachineOperand Src1Sub1 = TII->buildExtractSubRegOrImm(MI, MRI,
      Src1, &AMDGPU::SReg_64RegClass, AMDGPU::sub1,
      &AMDGPU::SReg_32_XM0RegClass);

    bool IsAdd = (MI.getOpcode() == AMDGPU::S_ADD_U64_PSEUDO);

    unsigned LoOpc = IsAdd ? AMDGPU::S_ADD_U32 : AMDGPU::S_SUB_U32;
    unsigned HiOpc = IsAdd ? AMDGPU::S_ADDC_U32 : AMDGPU::S_SUBB_U32;
    BuildMI(*BB, MI, DL, TII->get(LoOpc), DestSub0)
      .add(Src0Sub0)
      .add(Src1Sub0);
    BuildMI(*BB, MI, DL, TII->get(HiOpc), DestSub1)
      .add(Src0Sub1)
      .add(Src1Sub1);
    BuildMI(*BB, MI, DL, TII->get(TargetOpcode::REG_SEQUENCE), Dest.getReg())
      .addReg(DestSub0)
      .addImm(AMDGPU::sub0)
      .addReg(DestSub1)
      .addImm(AMDGPU::sub1);
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::SI_INIT_M0: {
    BuildMI(*BB, MI.getIterator(), MI.getDebugLoc(),
            TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
        .add(MI.getOperand(0));
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::SI_INIT_EXEC:
    // This should be before all vector instructions.
    BuildMI(*BB, &*BB->begin(), MI.getDebugLoc(), TII->get(AMDGPU::S_MOV_B64),
            AMDGPU::EXEC)
        .addImm(MI.getOperand(0).getImm());
    MI.eraseFromParent();
    return BB;

  case AMDGPU::SI_INIT_EXEC_FROM_INPUT: {
    // Extract the thread count from an SGPR input and set EXEC accordingly.
    // Since BFM can't shift by 64, handle that case with CMP + CMOV.
    //
    // S_BFE_U32 count, input, {shift, 7}
    // S_BFM_B64 exec, count, 0
    // S_CMP_EQ_U32 count, 64
    // S_CMOV_B64 exec, -1
    MachineInstr *FirstMI = &*BB->begin();
    MachineRegisterInfo &MRI = MF->getRegInfo();
    unsigned InputReg = MI.getOperand(0).getReg();
    unsigned CountReg = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
    bool Found = false;

    // Move the COPY of the input reg to the beginning, so that we can use it.
    for (auto I = BB->begin(); I != &MI; I++) {
      if (I->getOpcode() != TargetOpcode::COPY ||
          I->getOperand(0).getReg() != InputReg)
        continue;

      if (I == FirstMI) {
        FirstMI = &*++BB->begin();
      } else {
        I->removeFromParent();
        BB->insert(FirstMI, &*I);
      }
      Found = true;
      break;
    }
    assert(Found);
    (void)Found;

    // This should be before all vector instructions.
    BuildMI(*BB, FirstMI, DebugLoc(), TII->get(AMDGPU::S_BFE_U32), CountReg)
        .addReg(InputReg)
        .addImm((MI.getOperand(1).getImm() & 0x7f) | 0x70000);
    BuildMI(*BB, FirstMI, DebugLoc(), TII->get(AMDGPU::S_BFM_B64),
            AMDGPU::EXEC)
        .addReg(CountReg)
        .addImm(0);
    BuildMI(*BB, FirstMI, DebugLoc(), TII->get(AMDGPU::S_CMP_EQ_U32))
        .addReg(CountReg, RegState::Kill)
        .addImm(64);
    BuildMI(*BB, FirstMI, DebugLoc(), TII->get(AMDGPU::S_CMOV_B64),
            AMDGPU::EXEC)
        .addImm(-1);
    MI.eraseFromParent();
    return BB;
  }

  case AMDGPU::GET_GROUPSTATICSIZE: {
    DebugLoc DL = MI.getDebugLoc();
    BuildMI(*BB, MI, DL, TII->get(AMDGPU::S_MOV_B32))
        .add(MI.getOperand(0))
        .addImm(MFI->getLDSSize());
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::SI_INDIRECT_SRC_V1:
  case AMDGPU::SI_INDIRECT_SRC_V2:
  case AMDGPU::SI_INDIRECT_SRC_V4:
  case AMDGPU::SI_INDIRECT_SRC_V8:
  case AMDGPU::SI_INDIRECT_SRC_V16:
    return emitIndirectSrc(MI, *BB, *getSubtarget());
  case AMDGPU::SI_INDIRECT_DST_V1:
  case AMDGPU::SI_INDIRECT_DST_V2:
  case AMDGPU::SI_INDIRECT_DST_V4:
  case AMDGPU::SI_INDIRECT_DST_V8:
  case AMDGPU::SI_INDIRECT_DST_V16:
    return emitIndirectDst(MI, *BB, *getSubtarget());
  case AMDGPU::SI_KILL_F32_COND_IMM_PSEUDO:
  case AMDGPU::SI_KILL_I1_PSEUDO:
    return splitKillBlock(MI, BB);
  case AMDGPU::V_CNDMASK_B64_PSEUDO: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

    unsigned Dst = MI.getOperand(0).getReg();
    unsigned Src0 = MI.getOperand(1).getReg();
    unsigned Src1 = MI.getOperand(2).getReg();
    const DebugLoc &DL = MI.getDebugLoc();
    unsigned SrcCond = MI.getOperand(3).getReg();

    unsigned DstLo = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    unsigned DstHi = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    unsigned SrcCondCopy = MRI.createVirtualRegister(&AMDGPU::SReg_64_XEXECRegClass);

    BuildMI(*BB, MI, DL, TII->get(AMDGPU::COPY), SrcCondCopy)
      .addReg(SrcCond);
    BuildMI(*BB, MI, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64), DstLo)
      .addReg(Src0, 0, AMDGPU::sub0)
      .addReg(Src1, 0, AMDGPU::sub0)
      .addReg(SrcCondCopy);
    BuildMI(*BB, MI, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64), DstHi)
      .addReg(Src0, 0, AMDGPU::sub1)
      .addReg(Src1, 0, AMDGPU::sub1)
      .addReg(SrcCondCopy);

    BuildMI(*BB, MI, DL, TII->get(AMDGPU::REG_SEQUENCE), Dst)
      .addReg(DstLo)
      .addImm(AMDGPU::sub0)
      .addReg(DstHi)
      .addImm(AMDGPU::sub1);
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::SI_BR_UNDEF: {
    const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
    const DebugLoc &DL = MI.getDebugLoc();
    MachineInstr *Br = BuildMI(*BB, MI, DL, TII->get(AMDGPU::S_CBRANCH_SCC1))
                           .add(MI.getOperand(0));
    Br->getOperand(1).setIsUndef(true); // read undef SCC
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::ADJCALLSTACKUP:
  case AMDGPU::ADJCALLSTACKDOWN: {
    const SIMachineFunctionInfo *Info = MF->getInfo<SIMachineFunctionInfo>();
    MachineInstrBuilder MIB(*MF, &MI);
    MIB.addReg(Info->getStackPtrOffsetReg(), RegState::ImplicitDefine)
        .addReg(Info->getStackPtrOffsetReg(), RegState::Implicit);
    return BB;
  }
  case AMDGPU::SI_CALL_ISEL:
  case AMDGPU::SI_TCRETURN_ISEL: {
    const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
    const DebugLoc &DL = MI.getDebugLoc();
    unsigned ReturnAddrReg = TII->getRegisterInfo().getReturnAddressReg(*MF);

    MachineRegisterInfo &MRI = MF->getRegInfo();
    unsigned GlobalAddrReg = MI.getOperand(0).getReg();
    MachineInstr *PCRel = MRI.getVRegDef(GlobalAddrReg);
    assert(PCRel->getOpcode() == AMDGPU::SI_PC_ADD_REL_OFFSET);

    const GlobalValue *G = PCRel->getOperand(1).getGlobal();

    MachineInstrBuilder MIB;
    if (MI.getOpcode() == AMDGPU::SI_CALL_ISEL) {
      MIB = BuildMI(*BB, MI, DL, TII->get(AMDGPU::SI_CALL), ReturnAddrReg)
        .add(MI.getOperand(0))
        .addGlobalAddress(G);
    } else {
      MIB = BuildMI(*BB, MI, DL, TII->get(AMDGPU::SI_TCRETURN))
        .add(MI.getOperand(0))
        .addGlobalAddress(G);

      // There is an additional imm operand for tcreturn, but it should be in the
      // right place already.
    }

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I)
      MIB.add(MI.getOperand(I));

    MIB.setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
    MI.eraseFromParent();
    return BB;
  }
  default:
    return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
  }
}

bool SITargetLowering::hasBitPreservingFPLogic(EVT VT) const {
  return isTypeLegal(VT.getScalarType());
}

bool SITargetLowering::enableAggressiveFMAFusion(EVT VT) const {
  // This currently forces unfolding various combinations of fsub into fma with
  // free fneg'd operands. As long as we have fast FMA (controlled by
  // isFMAFasterThanFMulAndFAdd), we should perform these.

  // When fma is quarter rate, for f64 where add / sub are at best half rate,
  // most of these combines appear to be cycle neutral but save on instruction
  // count / code size.
  return true;
}

EVT SITargetLowering::getSetCCResultType(const DataLayout &DL, LLVMContext &Ctx,
                                         EVT VT) const {
  if (!VT.isVector()) {
    return MVT::i1;
  }
  return EVT::getVectorVT(Ctx, MVT::i1, VT.getVectorNumElements());
}

MVT SITargetLowering::getScalarShiftAmountTy(const DataLayout &, EVT VT) const {
  // TODO: Should i16 be used always if legal? For now it would force VALU
  // shifts.
  return (VT == MVT::i16) ? MVT::i16 : MVT::i32;
}

// Answering this is somewhat tricky and depends on the specific device which
// have different rates for fma or all f64 operations.
//
// v_fma_f64 and v_mul_f64 always take the same number of cycles as each other
// regardless of which device (although the number of cycles differs between
// devices), so it is always profitable for f64.
//
// v_fma_f32 takes 4 or 16 cycles depending on the device, so it is profitable
// only on full rate devices. Normally, we should prefer selecting v_mad_f32
// which we can always do even without fused FP ops since it returns the same
// result as the separate operations and since it is always full
// rate. Therefore, we lie and report that it is not faster for f32. v_mad_f32
// however does not support denormals, so we do report fma as faster if we have
// a fast fma device and require denormals.
//
bool SITargetLowering::isFMAFasterThanFMulAndFAdd(EVT VT) const {
  VT = VT.getScalarType();

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
    // This is as fast on some subtargets. However, we always have full rate f32
    // mad available which returns the same result as the separate operations
    // which we should prefer over fma. We can't use this if we want to support
    // denormals, so only report this in these cases.
    return Subtarget->hasFP32Denormals() && Subtarget->hasFastFMAF32();
  case MVT::f64:
    return true;
  case MVT::f16:
    return Subtarget->has16BitInsts() && Subtarget->hasFP16Denormals();
  default:
    break;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Custom DAG Lowering Operations
//===----------------------------------------------------------------------===//

SDValue SITargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  case ISD::BRCOND: return LowerBRCOND(Op, DAG);
  case ISD::LOAD: {
    SDValue Result = LowerLOAD(Op, DAG);
    assert((!Result.getNode() ||
            Result.getNode()->getNumValues() == 2) &&
           "Load should return a value and a chain");
    return Result;
  }

  case ISD::FSIN:
  case ISD::FCOS:
    return LowerTrig(Op, DAG);
  case ISD::SELECT: return LowerSELECT(Op, DAG);
  case ISD::FDIV: return LowerFDIV(Op, DAG);
  case ISD::ATOMIC_CMP_SWAP: return LowerATOMIC_CMP_SWAP(Op, DAG);
  case ISD::STORE: return LowerSTORE(Op, DAG);
  case ISD::GlobalAddress: {
    MachineFunction &MF = DAG.getMachineFunction();
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    return LowerGlobalAddress(MFI, Op, DAG);
  }
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN: return LowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID: return LowerINTRINSIC_VOID(Op, DAG);
  case ISD::ADDRSPACECAST: return lowerADDRSPACECAST(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    return lowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return lowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::FP_ROUND:
    return lowerFP_ROUND(Op, DAG);
  case ISD::TRAP:
  case ISD::DEBUGTRAP:
    return lowerTRAP(Op, DAG);
  }
  return SDValue();
}

void SITargetLowering::ReplaceNodeResults(SDNode *N,
                                          SmallVectorImpl<SDValue> &Results,
                                          SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  case ISD::INSERT_VECTOR_ELT: {
    if (SDValue Res = lowerINSERT_VECTOR_ELT(SDValue(N, 0), DAG))
      Results.push_back(Res);
    return;
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    if (SDValue Res = lowerEXTRACT_VECTOR_ELT(SDValue(N, 0), DAG))
      Results.push_back(Res);
    return;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    if (IID == Intrinsic::amdgcn_cvt_pkrtz) {
      SDValue Src0 = N->getOperand(1);
      SDValue Src1 = N->getOperand(2);
      SDLoc SL(N);
      SDValue Cvt = DAG.getNode(AMDGPUISD::CVT_PKRTZ_F16_F32, SL, MVT::i32,
                                Src0, Src1);
      Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2f16, Cvt));
      return;
    }
    break;
  }
  case ISD::SELECT: {
    SDLoc SL(N);
    EVT VT = N->getValueType(0);
    EVT NewVT = getEquivalentMemType(*DAG.getContext(), VT);
    SDValue LHS = DAG.getNode(ISD::BITCAST, SL, NewVT, N->getOperand(1));
    SDValue RHS = DAG.getNode(ISD::BITCAST, SL, NewVT, N->getOperand(2));

    EVT SelectVT = NewVT;
    if (NewVT.bitsLT(MVT::i32)) {
      LHS = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, LHS);
      RHS = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, RHS);
      SelectVT = MVT::i32;
    }

    SDValue NewSelect = DAG.getNode(ISD::SELECT, SL, SelectVT,
                                    N->getOperand(0), LHS, RHS);

    if (NewVT != SelectVT)
      NewSelect = DAG.getNode(ISD::TRUNCATE, SL, NewVT, NewSelect);
    Results.push_back(DAG.getNode(ISD::BITCAST, SL, VT, NewSelect));
    return;
  }
  default:
    break;
  }
}

/// \brief Helper function for LowerBRCOND
static SDNode *findUser(SDValue Value, unsigned Opcode) {

  SDNode *Parent = Value.getNode();
  for (SDNode::use_iterator I = Parent->use_begin(), E = Parent->use_end();
       I != E; ++I) {

    if (I.getUse().get() != Value)
      continue;

    if (I->getOpcode() == Opcode)
      return *I;
  }
  return nullptr;
}

unsigned SITargetLowering::isCFIntrinsic(const SDNode *Intr) const {
  if (Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN) {
    switch (cast<ConstantSDNode>(Intr->getOperand(1))->getZExtValue()) {
    case Intrinsic::amdgcn_if:
      return AMDGPUISD::IF;
    case Intrinsic::amdgcn_else:
      return AMDGPUISD::ELSE;
    case Intrinsic::amdgcn_loop:
      return AMDGPUISD::LOOP;
    case Intrinsic::amdgcn_end_cf:
      llvm_unreachable("should not occur");
    default:
      return 0;
    }
  }

  // break, if_break, else_break are all only used as inputs to loop, not
  // directly as branch conditions.
  return 0;
}

void SITargetLowering::createDebuggerPrologueStackObjects(
    MachineFunction &MF) const {
  // Create stack objects that are used for emitting debugger prologue.
  //
  // Debugger prologue writes work group IDs and work item IDs to scratch memory
  // at fixed location in the following format:
  //   offset 0:  work group ID x
  //   offset 4:  work group ID y
  //   offset 8:  work group ID z
  //   offset 16: work item ID x
  //   offset 20: work item ID y
  //   offset 24: work item ID z
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  int ObjectIdx = 0;

  // For each dimension:
  for (unsigned i = 0; i < 3; ++i) {
    // Create fixed stack object for work group ID.
    ObjectIdx = MF.getFrameInfo().CreateFixedObject(4, i * 4, true);
    Info->setDebuggerWorkGroupIDStackObjectIndex(i, ObjectIdx);
    // Create fixed stack object for work item ID.
    ObjectIdx = MF.getFrameInfo().CreateFixedObject(4, i * 4 + 16, true);
    Info->setDebuggerWorkItemIDStackObjectIndex(i, ObjectIdx);
  }
}

bool SITargetLowering::shouldEmitFixup(const GlobalValue *GV) const {
  const Triple &TT = getTargetMachine().getTargetTriple();
  return GV->getType()->getAddressSpace() == AMDGPUASI.CONSTANT_ADDRESS &&
         AMDGPU::shouldEmitConstantsToTextSection(TT);
}

bool SITargetLowering::shouldEmitGOTReloc(const GlobalValue *GV) const {
  return (GV->getType()->getAddressSpace() == AMDGPUASI.GLOBAL_ADDRESS ||
              GV->getType()->getAddressSpace() == AMDGPUASI.CONSTANT_ADDRESS) &&
         !shouldEmitFixup(GV) &&
         !getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);
}

bool SITargetLowering::shouldEmitPCReloc(const GlobalValue *GV) const {
  return !shouldEmitFixup(GV) && !shouldEmitGOTReloc(GV);
}

/// This transforms the control flow intrinsics to get the branch destination as
/// last parameter, also switches branch target with BR if the need arise
SDValue SITargetLowering::LowerBRCOND(SDValue BRCOND,
                                      SelectionDAG &DAG) const {
  SDLoc DL(BRCOND);

  SDNode *Intr = BRCOND.getOperand(1).getNode();
  SDValue Target = BRCOND.getOperand(2);
  SDNode *BR = nullptr;
  SDNode *SetCC = nullptr;

  if (Intr->getOpcode() == ISD::SETCC) {
    // As long as we negate the condition everything is fine
    SetCC = Intr;
    Intr = SetCC->getOperand(0).getNode();

  } else {
    // Get the target from BR if we don't negate the condition
    BR = findUser(BRCOND, ISD::BR);
    Target = BR->getOperand(1);
  }

  // FIXME: This changes the types of the intrinsics instead of introducing new
  // nodes with the correct types.
  // e.g. llvm.amdgcn.loop

  // eg: i1,ch = llvm.amdgcn.loop t0, TargetConstant:i32<6271>, t3
  // =>     t9: ch = llvm.amdgcn.loop t0, TargetConstant:i32<6271>, t3, BasicBlock:ch<bb1 0x7fee5286d088>

  unsigned CFNode = isCFIntrinsic(Intr);
  if (CFNode == 0) {
    // This is a uniform branch so we don't need to legalize.
    return BRCOND;
  }

  bool HaveChain = Intr->getOpcode() == ISD::INTRINSIC_VOID ||
                   Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN;

  assert(!SetCC ||
        (SetCC->getConstantOperandVal(1) == 1 &&
         cast<CondCodeSDNode>(SetCC->getOperand(2).getNode())->get() ==
                                                             ISD::SETNE));

  // operands of the new intrinsic call
  SmallVector<SDValue, 4> Ops;
  if (HaveChain)
    Ops.push_back(BRCOND.getOperand(0));

  Ops.append(Intr->op_begin() + (HaveChain ?  2 : 1), Intr->op_end());
  Ops.push_back(Target);

  ArrayRef<EVT> Res(Intr->value_begin() + 1, Intr->value_end());

  // build the new intrinsic call
  SDNode *Result = DAG.getNode(CFNode, DL, DAG.getVTList(Res), Ops).getNode();

  if (!HaveChain) {
    SDValue Ops[] =  {
      SDValue(Result, 0),
      BRCOND.getOperand(0)
    };

    Result = DAG.getMergeValues(Ops, DL).getNode();
  }

  if (BR) {
    // Give the branch instruction our target
    SDValue Ops[] = {
      BR->getOperand(0),
      BRCOND.getOperand(2)
    };
    SDValue NewBR = DAG.getNode(ISD::BR, DL, BR->getVTList(), Ops);
    DAG.ReplaceAllUsesWith(BR, NewBR.getNode());
    BR = NewBR.getNode();
  }

  SDValue Chain = SDValue(Result, Result->getNumValues() - 1);

  // Copy the intrinsic results to registers
  for (unsigned i = 1, e = Intr->getNumValues() - 1; i != e; ++i) {
    SDNode *CopyToReg = findUser(SDValue(Intr, i), ISD::CopyToReg);
    if (!CopyToReg)
      continue;

    Chain = DAG.getCopyToReg(
      Chain, DL,
      CopyToReg->getOperand(1),
      SDValue(Result, i - 1),
      SDValue());

    DAG.ReplaceAllUsesWith(SDValue(CopyToReg, 0), CopyToReg->getOperand(0));
  }

  // Remove the old intrinsic from the chain
  DAG.ReplaceAllUsesOfValueWith(
    SDValue(Intr, Intr->getNumValues() - 1),
    Intr->getOperand(0));

  return Chain;
}

SDValue SITargetLowering::getFPExtOrFPTrunc(SelectionDAG &DAG,
                                            SDValue Op,
                                            const SDLoc &DL,
                                            EVT VT) const {
  return Op.getValueType().bitsLE(VT) ?
      DAG.getNode(ISD::FP_EXTEND, DL, VT, Op) :
      DAG.getNode(ISD::FTRUNC, DL, VT, Op);
}

SDValue SITargetLowering::lowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const {
  assert(Op.getValueType() == MVT::f16 &&
         "Do not know how to custom lower FP_ROUND for non-f16 type");

  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  if (SrcVT != MVT::f64)
    return Op;

  SDLoc DL(Op);

  SDValue FpToFp16 = DAG.getNode(ISD::FP_TO_FP16, DL, MVT::i32, Src);
  SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, FpToFp16);
  return DAG.getNode(ISD::BITCAST, DL, MVT::f16, Trunc);
}

SDValue SITargetLowering::lowerTRAP(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  MachineFunction &MF = DAG.getMachineFunction();
  SDValue Chain = Op.getOperand(0);

  unsigned TrapID = Op.getOpcode() == ISD::DEBUGTRAP ?
    SISubtarget::TrapIDLLVMDebugTrap : SISubtarget::TrapIDLLVMTrap;

  if (Subtarget->getTrapHandlerAbi() == SISubtarget::TrapHandlerAbiHsa &&
      Subtarget->isTrapHandlerEnabled()) {
    SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
    unsigned UserSGPR = Info->getQueuePtrUserSGPR();
    assert(UserSGPR != AMDGPU::NoRegister);

    SDValue QueuePtr = CreateLiveInRegister(
      DAG, &AMDGPU::SReg_64RegClass, UserSGPR, MVT::i64);

    SDValue SGPR01 = DAG.getRegister(AMDGPU::SGPR0_SGPR1, MVT::i64);

    SDValue ToReg = DAG.getCopyToReg(Chain, SL, SGPR01,
                                     QueuePtr, SDValue());

    SDValue Ops[] = {
      ToReg,
      DAG.getTargetConstant(TrapID, SL, MVT::i16),
      SGPR01,
      ToReg.getValue(1)
    };

    return DAG.getNode(AMDGPUISD::TRAP, SL, MVT::Other, Ops);
  }

  switch (TrapID) {
  case SISubtarget::TrapIDLLVMTrap:
    return DAG.getNode(AMDGPUISD::ENDPGM, SL, MVT::Other, Chain);
  case SISubtarget::TrapIDLLVMDebugTrap: {
    DiagnosticInfoUnsupported NoTrap(*MF.getFunction(),
                                     "debugtrap handler not supported",
                                     Op.getDebugLoc(),
                                     DS_Warning);
    LLVMContext &Ctx = MF.getFunction()->getContext();
    Ctx.diagnose(NoTrap);
    return Chain;
  }
  default:
    llvm_unreachable("unsupported trap handler type!");
  }

  return Chain;
}

SDValue SITargetLowering::getSegmentAperture(unsigned AS, const SDLoc &DL,
                                             SelectionDAG &DAG) const {
  // FIXME: Use inline constants (src_{shared, private}_base) instead.
  if (Subtarget->hasApertureRegs()) {
    unsigned Offset = AS == AMDGPUASI.LOCAL_ADDRESS ?
        AMDGPU::Hwreg::OFFSET_SRC_SHARED_BASE :
        AMDGPU::Hwreg::OFFSET_SRC_PRIVATE_BASE;
    unsigned WidthM1 = AS == AMDGPUASI.LOCAL_ADDRESS ?
        AMDGPU::Hwreg::WIDTH_M1_SRC_SHARED_BASE :
        AMDGPU::Hwreg::WIDTH_M1_SRC_PRIVATE_BASE;
    unsigned Encoding =
        AMDGPU::Hwreg::ID_MEM_BASES << AMDGPU::Hwreg::ID_SHIFT_ |
        Offset << AMDGPU::Hwreg::OFFSET_SHIFT_ |
        WidthM1 << AMDGPU::Hwreg::WIDTH_M1_SHIFT_;

    SDValue EncodingImm = DAG.getTargetConstant(Encoding, DL, MVT::i16);
    SDValue ApertureReg = SDValue(
        DAG.getMachineNode(AMDGPU::S_GETREG_B32, DL, MVT::i32, EncodingImm), 0);
    SDValue ShiftAmount = DAG.getTargetConstant(WidthM1 + 1, DL, MVT::i32);
    return DAG.getNode(ISD::SHL, DL, MVT::i32, ApertureReg, ShiftAmount);
  }

  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  unsigned UserSGPR = Info->getQueuePtrUserSGPR();
  assert(UserSGPR != AMDGPU::NoRegister);

  SDValue QueuePtr = CreateLiveInRegister(
    DAG, &AMDGPU::SReg_64RegClass, UserSGPR, MVT::i64);

  // Offset into amd_queue_t for group_segment_aperture_base_hi /
  // private_segment_aperture_base_hi.
  uint32_t StructOffset = (AS == AMDGPUASI.LOCAL_ADDRESS) ? 0x40 : 0x44;

  SDValue Ptr = DAG.getObjectPtrOffset(DL, QueuePtr, StructOffset);

  // TODO: Use custom target PseudoSourceValue.
  // TODO: We should use the value from the IR intrinsic call, but it might not
  // be available and how do we get it?
  Value *V = UndefValue::get(PointerType::get(Type::getInt8Ty(*DAG.getContext()),
                                              AMDGPUASI.CONSTANT_ADDRESS));

  MachinePointerInfo PtrInfo(V, StructOffset);
  return DAG.getLoad(MVT::i32, DL, QueuePtr.getValue(1), Ptr, PtrInfo,
                     MinAlign(64, StructOffset),
                     MachineMemOperand::MODereferenceable |
                         MachineMemOperand::MOInvariant);
}

SDValue SITargetLowering::lowerADDRSPACECAST(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc SL(Op);
  const AddrSpaceCastSDNode *ASC = cast<AddrSpaceCastSDNode>(Op);

  SDValue Src = ASC->getOperand(0);
  SDValue FlatNullPtr = DAG.getConstant(0, SL, MVT::i64);

  const AMDGPUTargetMachine &TM =
    static_cast<const AMDGPUTargetMachine &>(getTargetMachine());

  // flat -> local/private
  if (ASC->getSrcAddressSpace() == AMDGPUASI.FLAT_ADDRESS) {
    unsigned DestAS = ASC->getDestAddressSpace();

    if (DestAS == AMDGPUASI.LOCAL_ADDRESS ||
        DestAS == AMDGPUASI.PRIVATE_ADDRESS) {
      unsigned NullVal = TM.getNullPointerValue(DestAS);
      SDValue SegmentNullPtr = DAG.getConstant(NullVal, SL, MVT::i32);
      SDValue NonNull = DAG.getSetCC(SL, MVT::i1, Src, FlatNullPtr, ISD::SETNE);
      SDValue Ptr = DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, Src);

      return DAG.getNode(ISD::SELECT, SL, MVT::i32,
                         NonNull, Ptr, SegmentNullPtr);
    }
  }

  // local/private -> flat
  if (ASC->getDestAddressSpace() == AMDGPUASI.FLAT_ADDRESS) {
    unsigned SrcAS = ASC->getSrcAddressSpace();

    if (SrcAS == AMDGPUASI.LOCAL_ADDRESS ||
        SrcAS == AMDGPUASI.PRIVATE_ADDRESS) {
      unsigned NullVal = TM.getNullPointerValue(SrcAS);
      SDValue SegmentNullPtr = DAG.getConstant(NullVal, SL, MVT::i32);

      SDValue NonNull
        = DAG.getSetCC(SL, MVT::i1, Src, SegmentNullPtr, ISD::SETNE);

      SDValue Aperture = getSegmentAperture(ASC->getSrcAddressSpace(), SL, DAG);
      SDValue CvtPtr
        = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32, Src, Aperture);

      return DAG.getNode(ISD::SELECT, SL, MVT::i64, NonNull,
                         DAG.getNode(ISD::BITCAST, SL, MVT::i64, CvtPtr),
                         FlatNullPtr);
    }
  }

  // global <-> flat are no-ops and never emitted.

  const MachineFunction &MF = DAG.getMachineFunction();
  DiagnosticInfoUnsupported InvalidAddrSpaceCast(
    *MF.getFunction(), "invalid addrspacecast", SL.getDebugLoc());
  DAG.getContext()->diagnose(InvalidAddrSpaceCast);

  return DAG.getUNDEF(ASC->getValueType(0));
}

SDValue SITargetLowering::lowerINSERT_VECTOR_ELT(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDValue Idx = Op.getOperand(2);
  if (isa<ConstantSDNode>(Idx))
    return SDValue();

  // Avoid stack access for dynamic indexing.
  SDLoc SL(Op);
  SDValue Vec = Op.getOperand(0);
  SDValue Val = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Op.getOperand(1));

  // v_bfi_b32 (v_bfm_b32 16, (shl idx, 16)), val, vec
  SDValue ExtVal = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i32, Val);

  // Convert vector index to bit-index.
  SDValue ScaledIdx = DAG.getNode(ISD::SHL, SL, MVT::i32, Idx,
                                  DAG.getConstant(16, SL, MVT::i32));

  SDValue BCVec = DAG.getNode(ISD::BITCAST, SL, MVT::i32, Vec);

  SDValue BFM = DAG.getNode(ISD::SHL, SL, MVT::i32,
                            DAG.getConstant(0xffff, SL, MVT::i32),
                            ScaledIdx);

  SDValue LHS = DAG.getNode(ISD::AND, SL, MVT::i32, BFM, ExtVal);
  SDValue RHS = DAG.getNode(ISD::AND, SL, MVT::i32,
                            DAG.getNOT(SL, BFM, MVT::i32), BCVec);

  SDValue BFI = DAG.getNode(ISD::OR, SL, MVT::i32, LHS, RHS);
  return DAG.getNode(ISD::BITCAST, SL, Op.getValueType(), BFI);
}

SDValue SITargetLowering::lowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc SL(Op);

  EVT ResultVT = Op.getValueType();
  SDValue Vec = Op.getOperand(0);
  SDValue Idx = Op.getOperand(1);

  DAGCombinerInfo DCI(DAG, AfterLegalizeVectorOps, true, nullptr);

  // Make sure we we do any optimizations that will make it easier to fold
  // source modifiers before obscuring it with bit operations.

  // XXX - Why doesn't this get called when vector_shuffle is expanded?
  if (SDValue Combined = performExtractVectorEltCombine(Op.getNode(), DCI))
    return Combined;

  if (const ConstantSDNode *CIdx = dyn_cast<ConstantSDNode>(Idx)) {
    SDValue Result = DAG.getNode(ISD::BITCAST, SL, MVT::i32, Vec);

    if (CIdx->getZExtValue() == 1) {
      Result = DAG.getNode(ISD::SRL, SL, MVT::i32, Result,
                           DAG.getConstant(16, SL, MVT::i32));
    } else {
      assert(CIdx->getZExtValue() == 0);
    }

    if (ResultVT.bitsLT(MVT::i32))
      Result = DAG.getNode(ISD::TRUNCATE, SL, MVT::i16, Result);
    return DAG.getNode(ISD::BITCAST, SL, ResultVT, Result);
  }

  SDValue Sixteen = DAG.getConstant(16, SL, MVT::i32);

  // Convert vector index to bit-index.
  SDValue ScaledIdx = DAG.getNode(ISD::SHL, SL, MVT::i32, Idx, Sixteen);

  SDValue BC = DAG.getNode(ISD::BITCAST, SL, MVT::i32, Vec);
  SDValue Elt = DAG.getNode(ISD::SRL, SL, MVT::i32, BC, ScaledIdx);

  SDValue Result = Elt;
  if (ResultVT.bitsLT(MVT::i32))
    Result = DAG.getNode(ISD::TRUNCATE, SL, MVT::i16, Result);

  return DAG.getNode(ISD::BITCAST, SL, ResultVT, Result);
}

bool
SITargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // We can fold offsets for anything that doesn't require a GOT relocation.
  return (GA->getAddressSpace() == AMDGPUASI.GLOBAL_ADDRESS ||
              GA->getAddressSpace() == AMDGPUASI.CONSTANT_ADDRESS) &&
         !shouldEmitGOTReloc(GA->getGlobal());
}

static SDValue
buildPCRelGlobalAddress(SelectionDAG &DAG, const GlobalValue *GV,
                        const SDLoc &DL, unsigned Offset, EVT PtrVT,
                        unsigned GAFlags = SIInstrInfo::MO_NONE) {
  // In order to support pc-relative addressing, the PC_ADD_REL_OFFSET SDNode is
  // lowered to the following code sequence:
  //
  // For constant address space:
  //   s_getpc_b64 s[0:1]
  //   s_add_u32 s0, s0, $symbol
  //   s_addc_u32 s1, s1, 0
  //
  //   s_getpc_b64 returns the address of the s_add_u32 instruction and then
  //   a fixup or relocation is emitted to replace $symbol with a literal
  //   constant, which is a pc-relative offset from the encoding of the $symbol
  //   operand to the global variable.
  //
  // For global address space:
  //   s_getpc_b64 s[0:1]
  //   s_add_u32 s0, s0, $symbol@{gotpc}rel32@lo
  //   s_addc_u32 s1, s1, $symbol@{gotpc}rel32@hi
  //
  //   s_getpc_b64 returns the address of the s_add_u32 instruction and then
  //   fixups or relocations are emitted to replace $symbol@*@lo and
  //   $symbol@*@hi with lower 32 bits and higher 32 bits of a literal constant,
  //   which is a 64-bit pc-relative offset from the encoding of the $symbol
  //   operand to the global variable.
  //
  // What we want here is an offset from the value returned by s_getpc
  // (which is the address of the s_add_u32 instruction) to the global
  // variable, but since the encoding of $symbol starts 4 bytes after the start
  // of the s_add_u32 instruction, we end up with an offset that is 4 bytes too
  // small. This requires us to add 4 to the global variable offset in order to
  // compute the correct address.
  SDValue PtrLo = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset + 4,
                                             GAFlags);
  SDValue PtrHi = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset + 4,
                                             GAFlags == SIInstrInfo::MO_NONE ?
                                             GAFlags : GAFlags + 1);
  return DAG.getNode(AMDGPUISD::PC_ADD_REL_OFFSET, DL, PtrVT, PtrLo, PtrHi);
}

SDValue SITargetLowering::LowerGlobalAddress(AMDGPUMachineFunction *MFI,
                                             SDValue Op,
                                             SelectionDAG &DAG) const {
  GlobalAddressSDNode *GSD = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = GSD->getGlobal();

  if (GSD->getAddressSpace() != AMDGPUASI.CONSTANT_ADDRESS &&
      GSD->getAddressSpace() != AMDGPUASI.GLOBAL_ADDRESS &&
      // FIXME: It isn't correct to rely on the type of the pointer. This should
      // be removed when address space 0 is 64-bit.
      !GV->getType()->getElementType()->isFunctionTy())
    return AMDGPUTargetLowering::LowerGlobalAddress(MFI, Op, DAG);

  SDLoc DL(GSD);
  EVT PtrVT = Op.getValueType();

  if (shouldEmitFixup(GV))
    return buildPCRelGlobalAddress(DAG, GV, DL, GSD->getOffset(), PtrVT);
  else if (shouldEmitPCReloc(GV))
    return buildPCRelGlobalAddress(DAG, GV, DL, GSD->getOffset(), PtrVT,
                                   SIInstrInfo::MO_REL32);

  SDValue GOTAddr = buildPCRelGlobalAddress(DAG, GV, DL, 0, PtrVT,
                                            SIInstrInfo::MO_GOTPCREL32);

  Type *Ty = PtrVT.getTypeForEVT(*DAG.getContext());
  PointerType *PtrTy = PointerType::get(Ty, AMDGPUASI.CONSTANT_ADDRESS);
  const DataLayout &DataLayout = DAG.getDataLayout();
  unsigned Align = DataLayout.getABITypeAlignment(PtrTy);
  // FIXME: Use a PseudoSourceValue once those can be assigned an address space.
  MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));

  return DAG.getLoad(PtrVT, DL, DAG.getEntryNode(), GOTAddr, PtrInfo, Align,
                     MachineMemOperand::MODereferenceable |
                         MachineMemOperand::MOInvariant);
}

SDValue SITargetLowering::copyToM0(SelectionDAG &DAG, SDValue Chain,
                                   const SDLoc &DL, SDValue V) const {
  // We can't use S_MOV_B32 directly, because there is no way to specify m0 as
  // the destination register.
  //
  // We can't use CopyToReg, because MachineCSE won't combine COPY instructions,
  // so we will end up with redundant moves to m0.
  //
  // We use a pseudo to ensure we emit s_mov_b32 with m0 as the direct result.

  // A Null SDValue creates a glue result.
  SDNode *M0 = DAG.getMachineNode(AMDGPU::SI_INIT_M0, DL, MVT::Other, MVT::Glue,
                                  V, Chain);
  return SDValue(M0, 0);
}

SDValue SITargetLowering::lowerImplicitZextParam(SelectionDAG &DAG,
                                                 SDValue Op,
                                                 MVT VT,
                                                 unsigned Offset) const {
  SDLoc SL(Op);
  SDValue Param = lowerKernargMemParameter(DAG, MVT::i32, MVT::i32, SL,
                                           DAG.getEntryNode(), Offset, false);
  // The local size values will have the hi 16-bits as zero.
  return DAG.getNode(ISD::AssertZext, SL, MVT::i32, Param,
                     DAG.getValueType(VT));
}

static SDValue emitNonHSAIntrinsicError(SelectionDAG &DAG, const SDLoc &DL,
                                        EVT VT) {
  DiagnosticInfoUnsupported BadIntrin(*DAG.getMachineFunction().getFunction(),
                                      "non-hsa intrinsic with hsa target",
                                      DL.getDebugLoc());
  DAG.getContext()->diagnose(BadIntrin);
  return DAG.getUNDEF(VT);
}

static SDValue emitRemovedIntrinsicError(SelectionDAG &DAG, const SDLoc &DL,
                                         EVT VT) {
  DiagnosticInfoUnsupported BadIntrin(*DAG.getMachineFunction().getFunction(),
                                      "intrinsic not supported on subtarget",
                                      DL.getDebugLoc());
  DAG.getContext()->diagnose(BadIntrin);
  return DAG.getUNDEF(VT);
}

SDValue SITargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                  SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  auto MFI = MF.getInfo<SIMachineFunctionInfo>();

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();

  // TODO: Should this propagate fast-math-flags?

  switch (IntrinsicID) {
  case Intrinsic::amdgcn_implicit_buffer_ptr: {
    if (getSubtarget()->isAmdCodeObjectV2(MF))
      return emitNonHSAIntrinsicError(DAG, DL, VT);
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::IMPLICIT_BUFFER_PTR);
  }
  case Intrinsic::amdgcn_dispatch_ptr:
  case Intrinsic::amdgcn_queue_ptr: {
    if (!Subtarget->isAmdCodeObjectV2(MF)) {
      DiagnosticInfoUnsupported BadIntrin(
          *MF.getFunction(), "unsupported hsa intrinsic without hsa target",
          DL.getDebugLoc());
      DAG.getContext()->diagnose(BadIntrin);
      return DAG.getUNDEF(VT);
    }

    auto RegID = IntrinsicID == Intrinsic::amdgcn_dispatch_ptr ?
      AMDGPUFunctionArgInfo::DISPATCH_PTR : AMDGPUFunctionArgInfo::QUEUE_PTR;
    return getPreloadedValue(DAG, *MFI, VT, RegID);
  }
  case Intrinsic::amdgcn_implicitarg_ptr: {
    if (MFI->isEntryFunction())
      return getImplicitArgPtr(DAG, DL);
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR);
  }
  case Intrinsic::amdgcn_kernarg_segment_ptr: {
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);
  }
  case Intrinsic::amdgcn_dispatch_id: {
    return getPreloadedValue(DAG, *MFI, VT, AMDGPUFunctionArgInfo::DISPATCH_ID);
  }
  case Intrinsic::amdgcn_rcp:
    return DAG.getNode(AMDGPUISD::RCP, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rsq:
    return DAG.getNode(AMDGPUISD::RSQ, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rsq_legacy:
    if (Subtarget->getGeneration() >= SISubtarget::VOLCANIC_ISLANDS)
      return emitRemovedIntrinsicError(DAG, DL, VT);

    return DAG.getNode(AMDGPUISD::RSQ_LEGACY, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rcp_legacy:
    if (Subtarget->getGeneration() >= SISubtarget::VOLCANIC_ISLANDS)
      return emitRemovedIntrinsicError(DAG, DL, VT);
    return DAG.getNode(AMDGPUISD::RCP_LEGACY, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rsq_clamp: {
    if (Subtarget->getGeneration() < SISubtarget::VOLCANIC_ISLANDS)
      return DAG.getNode(AMDGPUISD::RSQ_CLAMP, DL, VT, Op.getOperand(1));

    Type *Type = VT.getTypeForEVT(*DAG.getContext());
    APFloat Max = APFloat::getLargest(Type->getFltSemantics());
    APFloat Min = APFloat::getLargest(Type->getFltSemantics(), true);

    SDValue Rsq = DAG.getNode(AMDGPUISD::RSQ, DL, VT, Op.getOperand(1));
    SDValue Tmp = DAG.getNode(ISD::FMINNUM, DL, VT, Rsq,
                              DAG.getConstantFP(Max, DL, VT));
    return DAG.getNode(ISD::FMAXNUM, DL, VT, Tmp,
                       DAG.getConstantFP(Min, DL, VT));
  }
  case Intrinsic::r600_read_ngroups_x:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::NGROUPS_X, false);
  case Intrinsic::r600_read_ngroups_y:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::NGROUPS_Y, false);
  case Intrinsic::r600_read_ngroups_z:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::NGROUPS_Z, false);
  case Intrinsic::r600_read_global_size_x:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::GLOBAL_SIZE_X, false);
  case Intrinsic::r600_read_global_size_y:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::GLOBAL_SIZE_Y, false);
  case Intrinsic::r600_read_global_size_z:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::GLOBAL_SIZE_Z, false);
  case Intrinsic::r600_read_local_size_x:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerImplicitZextParam(DAG, Op, MVT::i16,
                                  SI::KernelInputOffsets::LOCAL_SIZE_X);
  case Intrinsic::r600_read_local_size_y:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerImplicitZextParam(DAG, Op, MVT::i16,
                                  SI::KernelInputOffsets::LOCAL_SIZE_Y);
  case Intrinsic::r600_read_local_size_z:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerImplicitZextParam(DAG, Op, MVT::i16,
                                  SI::KernelInputOffsets::LOCAL_SIZE_Z);
  case Intrinsic::amdgcn_workgroup_id_x:
  case Intrinsic::r600_read_tgid_x:
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::WORKGROUP_ID_X);
  case Intrinsic::amdgcn_workgroup_id_y:
  case Intrinsic::r600_read_tgid_y:
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::WORKGROUP_ID_Y);
  case Intrinsic::amdgcn_workgroup_id_z:
  case Intrinsic::r600_read_tgid_z:
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::WORKGROUP_ID_Z);
  case Intrinsic::amdgcn_workitem_id_x: {
  case Intrinsic::r600_read_tidig_x:
    return loadInputValue(DAG, &AMDGPU::VGPR_32RegClass, MVT::i32,
                          SDLoc(DAG.getEntryNode()),
                          MFI->getArgInfo().WorkItemIDX);
  }
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::r600_read_tidig_y:
    return loadInputValue(DAG, &AMDGPU::VGPR_32RegClass, MVT::i32,
                          SDLoc(DAG.getEntryNode()),
                          MFI->getArgInfo().WorkItemIDY);
  case Intrinsic::amdgcn_workitem_id_z:
  case Intrinsic::r600_read_tidig_z:
    return loadInputValue(DAG, &AMDGPU::VGPR_32RegClass, MVT::i32,
                          SDLoc(DAG.getEntryNode()),
                          MFI->getArgInfo().WorkItemIDZ);
  case AMDGPUIntrinsic::SI_load_const: {
    SDValue Ops[] = {
      Op.getOperand(1),
      Op.getOperand(2)
    };

    MachineMemOperand *MMO = MF.getMachineMemOperand(
        MachinePointerInfo(),
        MachineMemOperand::MOLoad | MachineMemOperand::MODereferenceable |
            MachineMemOperand::MOInvariant,
        VT.getStoreSize(), 4);
    return DAG.getMemIntrinsicNode(AMDGPUISD::LOAD_CONSTANT, DL,
                                   Op->getVTList(), Ops, VT, MMO);
  }
  case Intrinsic::amdgcn_fdiv_fast:
    return lowerFDIV_FAST(Op, DAG);
  case Intrinsic::amdgcn_interp_mov: {
    SDValue M0 = copyToM0(DAG, DAG.getEntryNode(), DL, Op.getOperand(4));
    SDValue Glue = M0.getValue(1);
    return DAG.getNode(AMDGPUISD::INTERP_MOV, DL, MVT::f32, Op.getOperand(1),
                       Op.getOperand(2), Op.getOperand(3), Glue);
  }
  case Intrinsic::amdgcn_interp_p1: {
    SDValue M0 = copyToM0(DAG, DAG.getEntryNode(), DL, Op.getOperand(4));
    SDValue Glue = M0.getValue(1);
    return DAG.getNode(AMDGPUISD::INTERP_P1, DL, MVT::f32, Op.getOperand(1),
                       Op.getOperand(2), Op.getOperand(3), Glue);
  }
  case Intrinsic::amdgcn_interp_p2: {
    SDValue M0 = copyToM0(DAG, DAG.getEntryNode(), DL, Op.getOperand(5));
    SDValue Glue = SDValue(M0.getNode(), 1);
    return DAG.getNode(AMDGPUISD::INTERP_P2, DL, MVT::f32, Op.getOperand(1),
                       Op.getOperand(2), Op.getOperand(3), Op.getOperand(4),
                       Glue);
  }
  case Intrinsic::amdgcn_sin:
    return DAG.getNode(AMDGPUISD::SIN_HW, DL, VT, Op.getOperand(1));

  case Intrinsic::amdgcn_cos:
    return DAG.getNode(AMDGPUISD::COS_HW, DL, VT, Op.getOperand(1));

  case Intrinsic::amdgcn_log_clamp: {
    if (Subtarget->getGeneration() < SISubtarget::VOLCANIC_ISLANDS)
      return SDValue();

    DiagnosticInfoUnsupported BadIntrin(
      *MF.getFunction(), "intrinsic not supported on subtarget",
      DL.getDebugLoc());
      DAG.getContext()->diagnose(BadIntrin);
      return DAG.getUNDEF(VT);
  }
  case Intrinsic::amdgcn_ldexp:
    return DAG.getNode(AMDGPUISD::LDEXP, DL, VT,
                       Op.getOperand(1), Op.getOperand(2));

  case Intrinsic::amdgcn_fract:
    return DAG.getNode(AMDGPUISD::FRACT, DL, VT, Op.getOperand(1));

  case Intrinsic::amdgcn_class:
    return DAG.getNode(AMDGPUISD::FP_CLASS, DL, VT,
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::amdgcn_div_fmas:
    return DAG.getNode(AMDGPUISD::DIV_FMAS, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3),
                       Op.getOperand(4));

  case Intrinsic::amdgcn_div_fixup:
    return DAG.getNode(AMDGPUISD::DIV_FIXUP, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

  case Intrinsic::amdgcn_trig_preop:
    return DAG.getNode(AMDGPUISD::TRIG_PREOP, DL, VT,
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::amdgcn_div_scale: {
    // 3rd parameter required to be a constant.
    const ConstantSDNode *Param = dyn_cast<ConstantSDNode>(Op.getOperand(3));
    if (!Param)
      return DAG.getMergeValues({ DAG.getUNDEF(VT), DAG.getUNDEF(MVT::i1) }, DL);

    // Translate to the operands expected by the machine instruction. The
    // first parameter must be the same as the first instruction.
    SDValue Numerator = Op.getOperand(1);
    SDValue Denominator = Op.getOperand(2);

    // Note this order is opposite of the machine instruction's operations,
    // which is s0.f = Quotient, s1.f = Denominator, s2.f = Numerator. The
    // intrinsic has the numerator as the first operand to match a normal
    // division operation.

    SDValue Src0 = Param->isAllOnesValue() ? Numerator : Denominator;

    return DAG.getNode(AMDGPUISD::DIV_SCALE, DL, Op->getVTList(), Src0,
                       Denominator, Numerator);
  }
  case Intrinsic::amdgcn_icmp: {
    const auto *CD = dyn_cast<ConstantSDNode>(Op.getOperand(3));
    if (!CD)
      return DAG.getUNDEF(VT);

    int CondCode = CD->getSExtValue();
    if (CondCode < ICmpInst::Predicate::FIRST_ICMP_PREDICATE ||
        CondCode > ICmpInst::Predicate::LAST_ICMP_PREDICATE)
      return DAG.getUNDEF(VT);

    ICmpInst::Predicate IcInput = static_cast<ICmpInst::Predicate>(CondCode);
    ISD::CondCode CCOpcode = getICmpCondCode(IcInput);
    return DAG.getNode(AMDGPUISD::SETCC, DL, VT, Op.getOperand(1),
                       Op.getOperand(2), DAG.getCondCode(CCOpcode));
  }
  case Intrinsic::amdgcn_fcmp: {
    const auto *CD = dyn_cast<ConstantSDNode>(Op.getOperand(3));
    if (!CD)
      return DAG.getUNDEF(VT);

    int CondCode = CD->getSExtValue();
    if (CondCode < FCmpInst::Predicate::FIRST_FCMP_PREDICATE ||
        CondCode > FCmpInst::Predicate::LAST_FCMP_PREDICATE)
      return DAG.getUNDEF(VT);

    FCmpInst::Predicate IcInput = static_cast<FCmpInst::Predicate>(CondCode);
    ISD::CondCode CCOpcode = getFCmpCondCode(IcInput);
    return DAG.getNode(AMDGPUISD::SETCC, DL, VT, Op.getOperand(1),
                       Op.getOperand(2), DAG.getCondCode(CCOpcode));
  }
  case Intrinsic::amdgcn_fmed3:
    return DAG.getNode(AMDGPUISD::FMED3, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::amdgcn_fmul_legacy:
    return DAG.getNode(AMDGPUISD::FMUL_LEGACY, DL, VT,
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::amdgcn_sffbh:
    return DAG.getNode(AMDGPUISD::FFBH_I32, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_sbfe:
    return DAG.getNode(AMDGPUISD::BFE_I32, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::amdgcn_ubfe:
    return DAG.getNode(AMDGPUISD::BFE_U32, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::amdgcn_cvt_pkrtz: {
    // FIXME: Stop adding cast if v2f16 legal.
    EVT VT = Op.getValueType();
    SDValue Node = DAG.getNode(AMDGPUISD::CVT_PKRTZ_F16_F32, DL, MVT::i32,
                               Op.getOperand(1), Op.getOperand(2));
    return DAG.getNode(ISD::BITCAST, DL, VT, Node);
  }
  case Intrinsic::amdgcn_wqm: {
    SDValue Src = Op.getOperand(1);
    return SDValue(DAG.getMachineNode(AMDGPU::WQM, DL, Src.getValueType(), Src),
                   0);
  }
  case Intrinsic::amdgcn_wwm: {
    SDValue Src = Op.getOperand(1);
    return SDValue(DAG.getMachineNode(AMDGPU::WWM, DL, Src.getValueType(), Src),
                   0);
  }
  default:
    return Op;
  }
}

SDValue SITargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op,
                                                 SelectionDAG &DAG) const {
  unsigned IntrID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  SDLoc DL(Op);
  MachineFunction &MF = DAG.getMachineFunction();

  switch (IntrID) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec: {
    MemSDNode *M = cast<MemSDNode>(Op);
    unsigned Opc = (IntrID == Intrinsic::amdgcn_atomic_inc) ?
      AMDGPUISD::ATOMIC_INC : AMDGPUISD::ATOMIC_DEC;
    SDValue Ops[] = {
      M->getOperand(0), // Chain
      M->getOperand(2), // Ptr
      M->getOperand(3)  // Value
    };

    return DAG.getMemIntrinsicNode(Opc, SDLoc(Op), M->getVTList(), Ops,
                                   M->getMemoryVT(), M->getMemOperand());
  }
  case Intrinsic::amdgcn_buffer_load:
  case Intrinsic::amdgcn_buffer_load_format: {
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // rsrc
      Op.getOperand(3), // vindex
      Op.getOperand(4), // offset
      Op.getOperand(5), // glc
      Op.getOperand(6)  // slc
    };
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

    unsigned Opc = (IntrID == Intrinsic::amdgcn_buffer_load) ?
        AMDGPUISD::BUFFER_LOAD : AMDGPUISD::BUFFER_LOAD_FORMAT;
    EVT VT = Op.getValueType();
    EVT IntVT = VT.changeTypeToInteger();

    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(MFI->getBufferPSV()),
      MachineMemOperand::MOLoad,
      VT.getStoreSize(), VT.getStoreSize());

    return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops, IntVT, MMO);
  }
  case Intrinsic::amdgcn_tbuffer_load: {
    SDValue Ops[] = {
      Op.getOperand(0),  // Chain
      Op.getOperand(2),  // rsrc
      Op.getOperand(3),  // vindex
      Op.getOperand(4),  // voffset
      Op.getOperand(5),  // soffset
      Op.getOperand(6),  // offset
      Op.getOperand(7),  // dfmt
      Op.getOperand(8),  // nfmt
      Op.getOperand(9),  // glc
      Op.getOperand(10)   // slc
    };

    EVT VT = Op.getOperand(2).getValueType();

    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOLoad,
      VT.getStoreSize(), VT.getStoreSize());
    return DAG.getMemIntrinsicNode(AMDGPUISD::TBUFFER_LOAD_FORMAT, DL,
                                   Op->getVTList(), Ops, VT, MMO);
  }
  case Intrinsic::amdgcn_buffer_atomic_swap:
  case Intrinsic::amdgcn_buffer_atomic_add:
  case Intrinsic::amdgcn_buffer_atomic_sub:
  case Intrinsic::amdgcn_buffer_atomic_smin:
  case Intrinsic::amdgcn_buffer_atomic_umin:
  case Intrinsic::amdgcn_buffer_atomic_smax:
  case Intrinsic::amdgcn_buffer_atomic_umax:
  case Intrinsic::amdgcn_buffer_atomic_and:
  case Intrinsic::amdgcn_buffer_atomic_or:
  case Intrinsic::amdgcn_buffer_atomic_xor: {
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // vdata
      Op.getOperand(3), // rsrc
      Op.getOperand(4), // vindex
      Op.getOperand(5), // offset
      Op.getOperand(6)  // slc
    };
    EVT VT = Op.getOperand(3).getValueType();
    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOLoad |
      MachineMemOperand::MOStore |
      MachineMemOperand::MODereferenceable |
      MachineMemOperand::MOVolatile,
      VT.getStoreSize(), 4);
    unsigned Opcode = 0;

    switch (IntrID) {
    case Intrinsic::amdgcn_buffer_atomic_swap:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_SWAP;
      break;
    case Intrinsic::amdgcn_buffer_atomic_add:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_ADD;
      break;
    case Intrinsic::amdgcn_buffer_atomic_sub:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_SUB;
      break;
    case Intrinsic::amdgcn_buffer_atomic_smin:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_SMIN;
      break;
    case Intrinsic::amdgcn_buffer_atomic_umin:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_UMIN;
      break;
    case Intrinsic::amdgcn_buffer_atomic_smax:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_SMAX;
      break;
    case Intrinsic::amdgcn_buffer_atomic_umax:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_UMAX;
      break;
    case Intrinsic::amdgcn_buffer_atomic_and:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_AND;
      break;
    case Intrinsic::amdgcn_buffer_atomic_or:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_OR;
      break;
    case Intrinsic::amdgcn_buffer_atomic_xor:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_XOR;
      break;
    default:
      llvm_unreachable("unhandled atomic opcode");
    }

    return DAG.getMemIntrinsicNode(Opcode, DL, Op->getVTList(), Ops, VT, MMO);
  }

  case Intrinsic::amdgcn_buffer_atomic_cmpswap: {
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // src
      Op.getOperand(3), // cmp
      Op.getOperand(4), // rsrc
      Op.getOperand(5), // vindex
      Op.getOperand(6), // offset
      Op.getOperand(7)  // slc
    };
    EVT VT = Op.getOperand(4).getValueType();
    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOLoad |
      MachineMemOperand::MOStore |
      MachineMemOperand::MODereferenceable |
      MachineMemOperand::MOVolatile,
      VT.getStoreSize(), 4);

    return DAG.getMemIntrinsicNode(AMDGPUISD::BUFFER_ATOMIC_CMPSWAP, DL,
                                   Op->getVTList(), Ops, VT, MMO);
  }

  // Basic sample.
  case Intrinsic::amdgcn_image_sample:
  case Intrinsic::amdgcn_image_sample_cl:
  case Intrinsic::amdgcn_image_sample_d:
  case Intrinsic::amdgcn_image_sample_d_cl:
  case Intrinsic::amdgcn_image_sample_l:
  case Intrinsic::amdgcn_image_sample_b:
  case Intrinsic::amdgcn_image_sample_b_cl:
  case Intrinsic::amdgcn_image_sample_lz:
  case Intrinsic::amdgcn_image_sample_cd:
  case Intrinsic::amdgcn_image_sample_cd_cl:

  // Sample with comparison.
  case Intrinsic::amdgcn_image_sample_c:
  case Intrinsic::amdgcn_image_sample_c_cl:
  case Intrinsic::amdgcn_image_sample_c_d:
  case Intrinsic::amdgcn_image_sample_c_d_cl:
  case Intrinsic::amdgcn_image_sample_c_l:
  case Intrinsic::amdgcn_image_sample_c_b:
  case Intrinsic::amdgcn_image_sample_c_b_cl:
  case Intrinsic::amdgcn_image_sample_c_lz:
  case Intrinsic::amdgcn_image_sample_c_cd:
  case Intrinsic::amdgcn_image_sample_c_cd_cl:

  // Sample with offsets.
  case Intrinsic::amdgcn_image_sample_o:
  case Intrinsic::amdgcn_image_sample_cl_o:
  case Intrinsic::amdgcn_image_sample_d_o:
  case Intrinsic::amdgcn_image_sample_d_cl_o:
  case Intrinsic::amdgcn_image_sample_l_o:
  case Intrinsic::amdgcn_image_sample_b_o:
  case Intrinsic::amdgcn_image_sample_b_cl_o:
  case Intrinsic::amdgcn_image_sample_lz_o:
  case Intrinsic::amdgcn_image_sample_cd_o:
  case Intrinsic::amdgcn_image_sample_cd_cl_o:

  // Sample with comparison and offsets.
  case Intrinsic::amdgcn_image_sample_c_o:
  case Intrinsic::amdgcn_image_sample_c_cl_o:
  case Intrinsic::amdgcn_image_sample_c_d_o:
  case Intrinsic::amdgcn_image_sample_c_d_cl_o:
  case Intrinsic::amdgcn_image_sample_c_l_o:
  case Intrinsic::amdgcn_image_sample_c_b_o:
  case Intrinsic::amdgcn_image_sample_c_b_cl_o:
  case Intrinsic::amdgcn_image_sample_c_lz_o:
  case Intrinsic::amdgcn_image_sample_c_cd_o:
  case Intrinsic::amdgcn_image_sample_c_cd_cl_o:

  case Intrinsic::amdgcn_image_getlod: {
    // Replace dmask with everything disabled with undef.
    const ConstantSDNode *DMask = dyn_cast<ConstantSDNode>(Op.getOperand(5));
    if (!DMask || DMask->isNullValue()) {
      SDValue Undef = DAG.getUNDEF(Op.getValueType());
      return DAG.getMergeValues({ Undef, Op.getOperand(0) }, SDLoc(Op));
    }

    return SDValue();
  }
  default:
    return SDValue();
  }
}

SDValue SITargetLowering::LowerINTRINSIC_VOID(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  MachineFunction &MF = DAG.getMachineFunction();

  switch (IntrinsicID) {
  case Intrinsic::amdgcn_exp: {
    const ConstantSDNode *Tgt = cast<ConstantSDNode>(Op.getOperand(2));
    const ConstantSDNode *En = cast<ConstantSDNode>(Op.getOperand(3));
    const ConstantSDNode *Done = cast<ConstantSDNode>(Op.getOperand(8));
    const ConstantSDNode *VM = cast<ConstantSDNode>(Op.getOperand(9));

    const SDValue Ops[] = {
      Chain,
      DAG.getTargetConstant(Tgt->getZExtValue(), DL, MVT::i8), // tgt
      DAG.getTargetConstant(En->getZExtValue(), DL, MVT::i8),  // en
      Op.getOperand(4), // src0
      Op.getOperand(5), // src1
      Op.getOperand(6), // src2
      Op.getOperand(7), // src3
      DAG.getTargetConstant(0, DL, MVT::i1), // compr
      DAG.getTargetConstant(VM->getZExtValue(), DL, MVT::i1)
    };

    unsigned Opc = Done->isNullValue() ?
      AMDGPUISD::EXPORT : AMDGPUISD::EXPORT_DONE;
    return DAG.getNode(Opc, DL, Op->getVTList(), Ops);
  }
  case Intrinsic::amdgcn_exp_compr: {
    const ConstantSDNode *Tgt = cast<ConstantSDNode>(Op.getOperand(2));
    const ConstantSDNode *En = cast<ConstantSDNode>(Op.getOperand(3));
    SDValue Src0 = Op.getOperand(4);
    SDValue Src1 = Op.getOperand(5);
    const ConstantSDNode *Done = cast<ConstantSDNode>(Op.getOperand(6));
    const ConstantSDNode *VM = cast<ConstantSDNode>(Op.getOperand(7));

    SDValue Undef = DAG.getUNDEF(MVT::f32);
    const SDValue Ops[] = {
      Chain,
      DAG.getTargetConstant(Tgt->getZExtValue(), DL, MVT::i8), // tgt
      DAG.getTargetConstant(En->getZExtValue(), DL, MVT::i8),  // en
      DAG.getNode(ISD::BITCAST, DL, MVT::f32, Src0),
      DAG.getNode(ISD::BITCAST, DL, MVT::f32, Src1),
      Undef, // src2
      Undef, // src3
      DAG.getTargetConstant(1, DL, MVT::i1), // compr
      DAG.getTargetConstant(VM->getZExtValue(), DL, MVT::i1)
    };

    unsigned Opc = Done->isNullValue() ?
      AMDGPUISD::EXPORT : AMDGPUISD::EXPORT_DONE;
    return DAG.getNode(Opc, DL, Op->getVTList(), Ops);
  }
  case Intrinsic::amdgcn_s_sendmsg:
  case Intrinsic::amdgcn_s_sendmsghalt: {
    unsigned NodeOp = (IntrinsicID == Intrinsic::amdgcn_s_sendmsg) ?
      AMDGPUISD::SENDMSG : AMDGPUISD::SENDMSGHALT;
    Chain = copyToM0(DAG, Chain, DL, Op.getOperand(3));
    SDValue Glue = Chain.getValue(1);
    return DAG.getNode(NodeOp, DL, MVT::Other, Chain,
                       Op.getOperand(2), Glue);
  }
  case Intrinsic::amdgcn_init_exec: {
    return DAG.getNode(AMDGPUISD::INIT_EXEC, DL, MVT::Other, Chain,
                       Op.getOperand(2));
  }
  case Intrinsic::amdgcn_init_exec_from_input: {
    return DAG.getNode(AMDGPUISD::INIT_EXEC_FROM_INPUT, DL, MVT::Other, Chain,
                       Op.getOperand(2), Op.getOperand(3));
  }
  case AMDGPUIntrinsic::AMDGPU_kill: {
    SDValue Src = Op.getOperand(2);
    if (const ConstantFPSDNode *K = dyn_cast<ConstantFPSDNode>(Src)) {
      if (!K->isNegative())
        return Chain;

      SDValue NegOne = DAG.getTargetConstant(FloatToBits(-1.0f), DL, MVT::i32);
      return DAG.getNode(AMDGPUISD::KILL, DL, MVT::Other, Chain, NegOne);
    }

    SDValue Cast = DAG.getNode(ISD::BITCAST, DL, MVT::i32, Src);
    return DAG.getNode(AMDGPUISD::KILL, DL, MVT::Other, Chain, Cast);
  }
  case Intrinsic::amdgcn_s_barrier: {
    if (getTargetMachine().getOptLevel() > CodeGenOpt::None) {
      const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
      unsigned WGSize = ST.getFlatWorkGroupSizes(*MF.getFunction()).second;
      if (WGSize <= ST.getWavefrontSize())
        return SDValue(DAG.getMachineNode(AMDGPU::WAVE_BARRIER, DL, MVT::Other,
                                          Op.getOperand(0)), 0);
    }
    return SDValue();
  };
  case AMDGPUIntrinsic::SI_tbuffer_store: {

    // Extract vindex and voffset from vaddr as appropriate
    const ConstantSDNode *OffEn = cast<ConstantSDNode>(Op.getOperand(10));
    const ConstantSDNode *IdxEn = cast<ConstantSDNode>(Op.getOperand(11));
    SDValue VAddr = Op.getOperand(5);

    SDValue Zero = DAG.getTargetConstant(0, DL, MVT::i32);

    assert(!(OffEn->isOne() && IdxEn->isOne()) &&
           "Legacy intrinsic doesn't support both offset and index - use new version");

    SDValue VIndex = IdxEn->isOne() ? VAddr : Zero;
    SDValue VOffset = OffEn->isOne() ? VAddr : Zero;

    // Deal with the vec-3 case
    const ConstantSDNode *NumChannels = cast<ConstantSDNode>(Op.getOperand(4));
    auto Opcode = NumChannels->getZExtValue() == 3 ?
      AMDGPUISD::TBUFFER_STORE_FORMAT_X3 : AMDGPUISD::TBUFFER_STORE_FORMAT;

    SDValue Ops[] = {
     Chain,
     Op.getOperand(3),  // vdata
     Op.getOperand(2),  // rsrc
     VIndex,
     VOffset,
     Op.getOperand(6),  // soffset
     Op.getOperand(7),  // inst_offset
     Op.getOperand(8),  // dfmt
     Op.getOperand(9),  // nfmt
     Op.getOperand(12), // glc
     Op.getOperand(13), // slc
    };

    assert((cast<ConstantSDNode>(Op.getOperand(14)))->getZExtValue() == 0 &&
           "Value of tfe other than zero is unsupported");

    EVT VT = Op.getOperand(3).getValueType();
    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOStore,
      VT.getStoreSize(), 4);
    return DAG.getMemIntrinsicNode(Opcode, DL,
                                   Op->getVTList(), Ops, VT, MMO);
  }

  case Intrinsic::amdgcn_tbuffer_store: {
    SDValue Ops[] = {
      Chain,
      Op.getOperand(2),  // vdata
      Op.getOperand(3),  // rsrc
      Op.getOperand(4),  // vindex
      Op.getOperand(5),  // voffset
      Op.getOperand(6),  // soffset
      Op.getOperand(7),  // offset
      Op.getOperand(8),  // dfmt
      Op.getOperand(9),  // nfmt
      Op.getOperand(10), // glc
      Op.getOperand(11)  // slc
    };
    EVT VT = Op.getOperand(3).getValueType();
    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOStore,
      VT.getStoreSize(), 4);
    return DAG.getMemIntrinsicNode(AMDGPUISD::TBUFFER_STORE_FORMAT, DL,
                                   Op->getVTList(), Ops, VT, MMO);
  }

  case Intrinsic::amdgcn_buffer_store:
  case Intrinsic::amdgcn_buffer_store_format: {
    SDValue Ops[] = {
      Chain,
      Op.getOperand(2), // vdata
      Op.getOperand(3), // rsrc
      Op.getOperand(4), // vindex
      Op.getOperand(5), // offset
      Op.getOperand(6), // glc
      Op.getOperand(7)  // slc
    };
    EVT VT = Op.getOperand(3).getValueType();
    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOStore |
      MachineMemOperand::MODereferenceable,
      VT.getStoreSize(), 4);

    unsigned Opcode = IntrinsicID == Intrinsic::amdgcn_buffer_store ?
                        AMDGPUISD::BUFFER_STORE :
                        AMDGPUISD::BUFFER_STORE_FORMAT;
    return DAG.getMemIntrinsicNode(Opcode, DL, Op->getVTList(), Ops, VT, MMO);
  }

  default:
    return Op;
  }
}

SDValue SITargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  LoadSDNode *Load = cast<LoadSDNode>(Op);
  ISD::LoadExtType ExtType = Load->getExtensionType();
  EVT MemVT = Load->getMemoryVT();

  if (ExtType == ISD::NON_EXTLOAD && MemVT.getSizeInBits() < 32) {
    if (MemVT == MVT::i16 && isTypeLegal(MVT::i16))
      return SDValue();

    // FIXME: Copied from PPC
    // First, load into 32 bits, then truncate to 1 bit.

    SDValue Chain = Load->getChain();
    SDValue BasePtr = Load->getBasePtr();
    MachineMemOperand *MMO = Load->getMemOperand();

    EVT RealMemVT = (MemVT == MVT::i1) ? MVT::i8 : MVT::i16;

    SDValue NewLD = DAG.getExtLoad(ISD::EXTLOAD, DL, MVT::i32, Chain,
                                   BasePtr, RealMemVT, MMO);

    SDValue Ops[] = {
      DAG.getNode(ISD::TRUNCATE, DL, MemVT, NewLD),
      NewLD.getValue(1)
    };

    return DAG.getMergeValues(Ops, DL);
  }

  if (!MemVT.isVector())
    return SDValue();

  assert(Op.getValueType().getVectorElementType() == MVT::i32 &&
         "Custom lowering for non-i32 vectors hasn't been implemented.");

  unsigned AS = Load->getAddressSpace();
  if (!allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), MemVT,
                          AS, Load->getAlignment())) {
    SDValue Ops[2];
    std::tie(Ops[0], Ops[1]) = expandUnalignedLoad(Load, DAG);
    return DAG.getMergeValues(Ops, DL);
  }

  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  // If there is a possibilty that flat instruction access scratch memory
  // then we need to use the same legalization rules we use for private.
  if (AS == AMDGPUASI.FLAT_ADDRESS)
    AS = MFI->hasFlatScratchInit() ?
         AMDGPUASI.PRIVATE_ADDRESS : AMDGPUASI.GLOBAL_ADDRESS;

  unsigned NumElements = MemVT.getVectorNumElements();
  if (AS == AMDGPUASI.CONSTANT_ADDRESS) {
    if (isMemOpUniform(Load))
      return SDValue();
    // Non-uniform loads will be selected to MUBUF instructions, so they
    // have the same legalization requirements as global and private
    // loads.
    //
  }
  if (AS == AMDGPUASI.CONSTANT_ADDRESS || AS == AMDGPUASI.GLOBAL_ADDRESS) {
    if (Subtarget->getScalarizeGlobalBehavior() && isMemOpUniform(Load) &&
        !Load->isVolatile() && isMemOpHasNoClobberedMemOperand(Load))
      return SDValue();
    // Non-uniform loads will be selected to MUBUF instructions, so they
    // have the same legalization requirements as global and private
    // loads.
    //
  }
  if (AS == AMDGPUASI.CONSTANT_ADDRESS || AS == AMDGPUASI.GLOBAL_ADDRESS ||
      AS == AMDGPUASI.FLAT_ADDRESS) {
    if (NumElements > 4)
      return SplitVectorLoad(Op, DAG);
    // v4 loads are supported for private and global memory.
    return SDValue();
  }
  if (AS == AMDGPUASI.PRIVATE_ADDRESS) {
    // Depending on the setting of the private_element_size field in the
    // resource descriptor, we can only make private accesses up to a certain
    // size.
    switch (Subtarget->getMaxPrivateElementSize()) {
    case 4:
      return scalarizeVectorLoad(Load, DAG);
    case 8:
      if (NumElements > 2)
        return SplitVectorLoad(Op, DAG);
      return SDValue();
    case 16:
      // Same as global/flat
      if (NumElements > 4)
        return SplitVectorLoad(Op, DAG);
      return SDValue();
    default:
      llvm_unreachable("unsupported private_element_size");
    }
  } else if (AS == AMDGPUASI.LOCAL_ADDRESS) {
    if (NumElements > 2)
      return SplitVectorLoad(Op, DAG);

    if (NumElements == 2)
      return SDValue();

    // If properly aligned, if we split we might be able to use ds_read_b64.
    return SplitVectorLoad(Op, DAG);
  }
  return SDValue();
}

SDValue SITargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getValueType() != MVT::i64)
    return SDValue();

  SDLoc DL(Op);
  SDValue Cond = Op.getOperand(0);

  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue One = DAG.getConstant(1, DL, MVT::i32);

  SDValue LHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(1));
  SDValue RHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(2));

  SDValue Lo0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, Zero);
  SDValue Lo1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, Zero);

  SDValue Lo = DAG.getSelect(DL, MVT::i32, Cond, Lo0, Lo1);

  SDValue Hi0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, One);
  SDValue Hi1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, One);

  SDValue Hi = DAG.getSelect(DL, MVT::i32, Cond, Hi0, Hi1);

  SDValue Res = DAG.getBuildVector(MVT::v2i32, DL, {Lo, Hi});
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Res);
}

// Catch division cases where we can use shortcuts with rcp and rsq
// instructions.
SDValue SITargetLowering::lowerFastUnsafeFDIV(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  EVT VT = Op.getValueType();
  const SDNodeFlags Flags = Op->getFlags();
  bool Unsafe = DAG.getTarget().Options.UnsafeFPMath ||
                Flags.hasUnsafeAlgebra() || Flags.hasAllowReciprocal();

  if (!Unsafe && VT == MVT::f32 && Subtarget->hasFP32Denormals())
    return SDValue();

  if (const ConstantFPSDNode *CLHS = dyn_cast<ConstantFPSDNode>(LHS)) {
    if (Unsafe || VT == MVT::f32 || VT == MVT::f16) {
      if (CLHS->isExactlyValue(1.0)) {
        // v_rcp_f32 and v_rsq_f32 do not support denormals, and according to
        // the CI documentation has a worst case error of 1 ulp.
        // OpenCL requires <= 2.5 ulp for 1.0 / x, so it should always be OK to
        // use it as long as we aren't trying to use denormals.
        //
        // v_rcp_f16 and v_rsq_f16 DO support denormals.

        // 1.0 / sqrt(x) -> rsq(x)

        // XXX - Is UnsafeFPMath sufficient to do this for f64? The maximum ULP
        // error seems really high at 2^29 ULP.
        if (RHS.getOpcode() == ISD::FSQRT)
          return DAG.getNode(AMDGPUISD::RSQ, SL, VT, RHS.getOperand(0));

        // 1.0 / x -> rcp(x)
        return DAG.getNode(AMDGPUISD::RCP, SL, VT, RHS);
      }

      // Same as for 1.0, but expand the sign out of the constant.
      if (CLHS->isExactlyValue(-1.0)) {
        // -1.0 / x -> rcp (fneg x)
        SDValue FNegRHS = DAG.getNode(ISD::FNEG, SL, VT, RHS);
        return DAG.getNode(AMDGPUISD::RCP, SL, VT, FNegRHS);
      }
    }
  }

  if (Unsafe) {
    // Turn into multiply by the reciprocal.
    // x / y -> x * (1.0 / y)
    SDValue Recip = DAG.getNode(AMDGPUISD::RCP, SL, VT, RHS);
    return DAG.getNode(ISD::FMUL, SL, VT, LHS, Recip, Flags);
  }

  return SDValue();
}

static SDValue getFPBinOp(SelectionDAG &DAG, unsigned Opcode, const SDLoc &SL,
                          EVT VT, SDValue A, SDValue B, SDValue GlueChain) {
  if (GlueChain->getNumValues() <= 1) {
    return DAG.getNode(Opcode, SL, VT, A, B);
  }

  assert(GlueChain->getNumValues() == 3);

  SDVTList VTList = DAG.getVTList(VT, MVT::Other, MVT::Glue);
  switch (Opcode) {
  default: llvm_unreachable("no chain equivalent for opcode");
  case ISD::FMUL:
    Opcode = AMDGPUISD::FMUL_W_CHAIN;
    break;
  }

  return DAG.getNode(Opcode, SL, VTList, GlueChain.getValue(1), A, B,
                     GlueChain.getValue(2));
}

static SDValue getFPTernOp(SelectionDAG &DAG, unsigned Opcode, const SDLoc &SL,
                           EVT VT, SDValue A, SDValue B, SDValue C,
                           SDValue GlueChain) {
  if (GlueChain->getNumValues() <= 1) {
    return DAG.getNode(Opcode, SL, VT, A, B, C);
  }

  assert(GlueChain->getNumValues() == 3);

  SDVTList VTList = DAG.getVTList(VT, MVT::Other, MVT::Glue);
  switch (Opcode) {
  default: llvm_unreachable("no chain equivalent for opcode");
  case ISD::FMA:
    Opcode = AMDGPUISD::FMA_W_CHAIN;
    break;
  }

  return DAG.getNode(Opcode, SL, VTList, GlueChain.getValue(1), A, B, C,
                     GlueChain.getValue(2));
}

SDValue SITargetLowering::LowerFDIV16(SDValue Op, SelectionDAG &DAG) const {
  if (SDValue FastLowered = lowerFastUnsafeFDIV(Op, DAG))
    return FastLowered;

  SDLoc SL(Op);
  SDValue Src0 = Op.getOperand(0);
  SDValue Src1 = Op.getOperand(1);

  SDValue CvtSrc0 = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, Src0);
  SDValue CvtSrc1 = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, Src1);

  SDValue RcpSrc1 = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f32, CvtSrc1);
  SDValue Quot = DAG.getNode(ISD::FMUL, SL, MVT::f32, CvtSrc0, RcpSrc1);

  SDValue FPRoundFlag = DAG.getTargetConstant(0, SL, MVT::i32);
  SDValue BestQuot = DAG.getNode(ISD::FP_ROUND, SL, MVT::f16, Quot, FPRoundFlag);

  return DAG.getNode(AMDGPUISD::DIV_FIXUP, SL, MVT::f16, BestQuot, Src1, Src0);
}

// Faster 2.5 ULP division that does not support denormals.
SDValue SITargetLowering::lowerFDIV_FAST(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(1);
  SDValue RHS = Op.getOperand(2);

  SDValue r1 = DAG.getNode(ISD::FABS, SL, MVT::f32, RHS);

  const APFloat K0Val(BitsToFloat(0x6f800000));
  const SDValue K0 = DAG.getConstantFP(K0Val, SL, MVT::f32);

  const APFloat K1Val(BitsToFloat(0x2f800000));
  const SDValue K1 = DAG.getConstantFP(K1Val, SL, MVT::f32);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f32);

  EVT SetCCVT =
    getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);

  SDValue r2 = DAG.getSetCC(SL, SetCCVT, r1, K0, ISD::SETOGT);

  SDValue r3 = DAG.getNode(ISD::SELECT, SL, MVT::f32, r2, K1, One);

  // TODO: Should this propagate fast-math-flags?
  r1 = DAG.getNode(ISD::FMUL, SL, MVT::f32, RHS, r3);

  // rcp does not support denormals.
  SDValue r0 = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f32, r1);

  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHS, r0);

  return DAG.getNode(ISD::FMUL, SL, MVT::f32, r3, Mul);
}

SDValue SITargetLowering::LowerFDIV32(SDValue Op, SelectionDAG &DAG) const {
  if (SDValue FastLowered = lowerFastUnsafeFDIV(Op, DAG))
    return FastLowered;

  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f32);

  SDVTList ScaleVT = DAG.getVTList(MVT::f32, MVT::i1);

  SDValue DenominatorScaled = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT,
                                          RHS, RHS, LHS);
  SDValue NumeratorScaled = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT,
                                        LHS, RHS, LHS);

  // Denominator is scaled to not be denormal, so using rcp is ok.
  SDValue ApproxRcp = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f32,
                                  DenominatorScaled);
  SDValue NegDivScale0 = DAG.getNode(ISD::FNEG, SL, MVT::f32,
                                     DenominatorScaled);

  const unsigned Denorm32Reg = AMDGPU::Hwreg::ID_MODE |
                               (4 << AMDGPU::Hwreg::OFFSET_SHIFT_) |
                               (1 << AMDGPU::Hwreg::WIDTH_M1_SHIFT_);

  const SDValue BitField = DAG.getTargetConstant(Denorm32Reg, SL, MVT::i16);

  if (!Subtarget->hasFP32Denormals()) {
    SDVTList BindParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
    const SDValue EnableDenormValue = DAG.getConstant(FP_DENORM_FLUSH_NONE,
                                                      SL, MVT::i32);
    SDValue EnableDenorm = DAG.getNode(AMDGPUISD::SETREG, SL, BindParamVTs,
                                       DAG.getEntryNode(),
                                       EnableDenormValue, BitField);
    SDValue Ops[3] = {
      NegDivScale0,
      EnableDenorm.getValue(0),
      EnableDenorm.getValue(1)
    };

    NegDivScale0 = DAG.getMergeValues(Ops, SL);
  }

  SDValue Fma0 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32, NegDivScale0,
                             ApproxRcp, One, NegDivScale0);

  SDValue Fma1 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32, Fma0, ApproxRcp,
                             ApproxRcp, Fma0);

  SDValue Mul = getFPBinOp(DAG, ISD::FMUL, SL, MVT::f32, NumeratorScaled,
                           Fma1, Fma1);

  SDValue Fma2 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32, NegDivScale0, Mul,
                             NumeratorScaled, Mul);

  SDValue Fma3 = getFPTernOp(DAG, ISD::FMA,SL, MVT::f32, Fma2, Fma1, Mul, Fma2);

  SDValue Fma4 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32, NegDivScale0, Fma3,
                             NumeratorScaled, Fma3);

  if (!Subtarget->hasFP32Denormals()) {
    const SDValue DisableDenormValue =
        DAG.getConstant(FP_DENORM_FLUSH_IN_FLUSH_OUT, SL, MVT::i32);
    SDValue DisableDenorm = DAG.getNode(AMDGPUISD::SETREG, SL, MVT::Other,
                                        Fma4.getValue(1),
                                        DisableDenormValue,
                                        BitField,
                                        Fma4.getValue(2));

    SDValue OutputChain = DAG.getNode(ISD::TokenFactor, SL, MVT::Other,
                                      DisableDenorm, DAG.getRoot());
    DAG.setRoot(OutputChain);
  }

  SDValue Scale = NumeratorScaled.getValue(1);
  SDValue Fmas = DAG.getNode(AMDGPUISD::DIV_FMAS, SL, MVT::f32,
                             Fma4, Fma1, Fma3, Scale);

  return DAG.getNode(AMDGPUISD::DIV_FIXUP, SL, MVT::f32, Fmas, RHS, LHS);
}

SDValue SITargetLowering::LowerFDIV64(SDValue Op, SelectionDAG &DAG) const {
  if (DAG.getTarget().Options.UnsafeFPMath)
    return lowerFastUnsafeFDIV(Op, DAG);

  SDLoc SL(Op);
  SDValue X = Op.getOperand(0);
  SDValue Y = Op.getOperand(1);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f64);

  SDVTList ScaleVT = DAG.getVTList(MVT::f64, MVT::i1);

  SDValue DivScale0 = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT, Y, Y, X);

  SDValue NegDivScale0 = DAG.getNode(ISD::FNEG, SL, MVT::f64, DivScale0);

  SDValue Rcp = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f64, DivScale0);

  SDValue Fma0 = DAG.getNode(ISD::FMA, SL, MVT::f64, NegDivScale0, Rcp, One);

  SDValue Fma1 = DAG.getNode(ISD::FMA, SL, MVT::f64, Rcp, Fma0, Rcp);

  SDValue Fma2 = DAG.getNode(ISD::FMA, SL, MVT::f64, NegDivScale0, Fma1, One);

  SDValue DivScale1 = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT, X, Y, X);

  SDValue Fma3 = DAG.getNode(ISD::FMA, SL, MVT::f64, Fma1, Fma2, Fma1);
  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f64, DivScale1, Fma3);

  SDValue Fma4 = DAG.getNode(ISD::FMA, SL, MVT::f64,
                             NegDivScale0, Mul, DivScale1);

  SDValue Scale;

  if (Subtarget->getGeneration() == SISubtarget::SOUTHERN_ISLANDS) {
    // Workaround a hardware bug on SI where the condition output from div_scale
    // is not usable.

    const SDValue Hi = DAG.getConstant(1, SL, MVT::i32);

    // Figure out if the scale to use for div_fmas.
    SDValue NumBC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, X);
    SDValue DenBC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Y);
    SDValue Scale0BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, DivScale0);
    SDValue Scale1BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, DivScale1);

    SDValue NumHi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, NumBC, Hi);
    SDValue DenHi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, DenBC, Hi);

    SDValue Scale0Hi
      = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Scale0BC, Hi);
    SDValue Scale1Hi
      = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Scale1BC, Hi);

    SDValue CmpDen = DAG.getSetCC(SL, MVT::i1, DenHi, Scale0Hi, ISD::SETEQ);
    SDValue CmpNum = DAG.getSetCC(SL, MVT::i1, NumHi, Scale1Hi, ISD::SETEQ);
    Scale = DAG.getNode(ISD::XOR, SL, MVT::i1, CmpNum, CmpDen);
  } else {
    Scale = DivScale1.getValue(1);
  }

  SDValue Fmas = DAG.getNode(AMDGPUISD::DIV_FMAS, SL, MVT::f64,
                             Fma4, Fma3, Mul, Scale);

  return DAG.getNode(AMDGPUISD::DIV_FIXUP, SL, MVT::f64, Fmas, Y, X);
}

SDValue SITargetLowering::LowerFDIV(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();

  if (VT == MVT::f32)
    return LowerFDIV32(Op, DAG);

  if (VT == MVT::f64)
    return LowerFDIV64(Op, DAG);

  if (VT == MVT::f16)
    return LowerFDIV16(Op, DAG);

  llvm_unreachable("Unexpected type for fdiv");
}

SDValue SITargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  StoreSDNode *Store = cast<StoreSDNode>(Op);
  EVT VT = Store->getMemoryVT();

  if (VT == MVT::i1) {
    return DAG.getTruncStore(Store->getChain(), DL,
       DAG.getSExtOrTrunc(Store->getValue(), DL, MVT::i32),
       Store->getBasePtr(), MVT::i1, Store->getMemOperand());
  }

  assert(VT.isVector() &&
         Store->getValue().getValueType().getScalarType() == MVT::i32);

  unsigned AS = Store->getAddressSpace();
  if (!allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), VT,
                          AS, Store->getAlignment())) {
    return expandUnalignedStore(Store, DAG);
  }

  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  // If there is a possibilty that flat instruction access scratch memory
  // then we need to use the same legalization rules we use for private.
  if (AS == AMDGPUASI.FLAT_ADDRESS)
    AS = MFI->hasFlatScratchInit() ?
         AMDGPUASI.PRIVATE_ADDRESS : AMDGPUASI.GLOBAL_ADDRESS;

  unsigned NumElements = VT.getVectorNumElements();
  if (AS == AMDGPUASI.GLOBAL_ADDRESS ||
      AS == AMDGPUASI.FLAT_ADDRESS) {
    if (NumElements > 4)
      return SplitVectorStore(Op, DAG);
    return SDValue();
  } else if (AS == AMDGPUASI.PRIVATE_ADDRESS) {
    switch (Subtarget->getMaxPrivateElementSize()) {
    case 4:
      return scalarizeVectorStore(Store, DAG);
    case 8:
      if (NumElements > 2)
        return SplitVectorStore(Op, DAG);
      return SDValue();
    case 16:
      if (NumElements > 4)
        return SplitVectorStore(Op, DAG);
      return SDValue();
    default:
      llvm_unreachable("unsupported private_element_size");
    }
  } else if (AS == AMDGPUASI.LOCAL_ADDRESS) {
    if (NumElements > 2)
      return SplitVectorStore(Op, DAG);

    if (NumElements == 2)
      return Op;

    // If properly aligned, if we split we might be able to use ds_write_b64.
    return SplitVectorStore(Op, DAG);
  } else {
    llvm_unreachable("unhandled address space");
  }
}

SDValue SITargetLowering::LowerTrig(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue Arg = Op.getOperand(0);
  // TODO: Should this propagate fast-math-flags?
  SDValue FractPart = DAG.getNode(AMDGPUISD::FRACT, DL, VT,
                                  DAG.getNode(ISD::FMUL, DL, VT, Arg,
                                              DAG.getConstantFP(0.5/M_PI, DL,
                                                                VT)));

  switch (Op.getOpcode()) {
  case ISD::FCOS:
    return DAG.getNode(AMDGPUISD::COS_HW, SDLoc(Op), VT, FractPart);
  case ISD::FSIN:
    return DAG.getNode(AMDGPUISD::SIN_HW, SDLoc(Op), VT, FractPart);
  default:
    llvm_unreachable("Wrong trig opcode");
  }
}

SDValue SITargetLowering::LowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const {
  AtomicSDNode *AtomicNode = cast<AtomicSDNode>(Op);
  assert(AtomicNode->isCompareAndSwap());
  unsigned AS = AtomicNode->getAddressSpace();

  // No custom lowering required for local address space
  if (!isFlatGlobalAddrSpace(AS, AMDGPUASI))
    return Op;

  // Non-local address space requires custom lowering for atomic compare
  // and swap; cmp and swap should be in a v2i32 or v2i64 in case of _X2
  SDLoc DL(Op);
  SDValue ChainIn = Op.getOperand(0);
  SDValue Addr = Op.getOperand(1);
  SDValue Old = Op.getOperand(2);
  SDValue New = Op.getOperand(3);
  EVT VT = Op.getValueType();
  MVT SimpleVT = VT.getSimpleVT();
  MVT VecType = MVT::getVectorVT(SimpleVT, 2);

  SDValue NewOld = DAG.getBuildVector(VecType, DL, {New, Old});
  SDValue Ops[] = { ChainIn, Addr, NewOld };

  return DAG.getMemIntrinsicNode(AMDGPUISD::ATOMIC_CMP_SWAP, DL, Op->getVTList(),
                                 Ops, VT, AtomicNode->getMemOperand());
}

//===----------------------------------------------------------------------===//
// Custom DAG optimizations
//===----------------------------------------------------------------------===//

SDValue SITargetLowering::performUCharToFloatCombine(SDNode *N,
                                                     DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  EVT ScalarVT = VT.getScalarType();
  if (ScalarVT != MVT::f32)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  SDValue Src = N->getOperand(0);
  EVT SrcVT = Src.getValueType();

  // TODO: We could try to match extracting the higher bytes, which would be
  // easier if i8 vectors weren't promoted to i32 vectors, particularly after
  // types are legalized. v4i8 -> v4f32 is probably the only case to worry
  // about in practice.
  if (DCI.isAfterLegalizeVectorOps() && SrcVT == MVT::i32) {
    if (DAG.MaskedValueIsZero(Src, APInt::getHighBitsSet(32, 24))) {
      SDValue Cvt = DAG.getNode(AMDGPUISD::CVT_F32_UBYTE0, DL, VT, Src);
      DCI.AddToWorklist(Cvt.getNode());
      return Cvt;
    }
  }

  return SDValue();
}

// (shl (add x, c1), c2) -> add (shl x, c2), (shl c1, c2)

// This is a variant of
// (mul (add x, c1), c2) -> add (mul x, c2), (mul c1, c2),
//
// The normal DAG combiner will do this, but only if the add has one use since
// that would increase the number of instructions.
//
// This prevents us from seeing a constant offset that can be folded into a
// memory instruction's addressing mode. If we know the resulting add offset of
// a pointer can be folded into an addressing offset, we can replace the pointer
// operand with the add of new constant offset. This eliminates one of the uses,
// and may allow the remaining use to also be simplified.
//
SDValue SITargetLowering::performSHLPtrCombine(SDNode *N,
                                               unsigned AddrSpace,
                                               EVT MemVT,
                                               DAGCombinerInfo &DCI) const {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // We only do this to handle cases where it's profitable when there are
  // multiple uses of the add, so defer to the standard combine.
  if ((N0.getOpcode() != ISD::ADD && N0.getOpcode() != ISD::OR) ||
      N0->hasOneUse())
    return SDValue();

  const ConstantSDNode *CN1 = dyn_cast<ConstantSDNode>(N1);
  if (!CN1)
    return SDValue();

  const ConstantSDNode *CAdd = dyn_cast<ConstantSDNode>(N0.getOperand(1));
  if (!CAdd)
    return SDValue();

  // If the resulting offset is too large, we can't fold it into the addressing
  // mode offset.
  APInt Offset = CAdd->getAPIntValue() << CN1->getAPIntValue();
  Type *Ty = MemVT.getTypeForEVT(*DCI.DAG.getContext());

  AddrMode AM;
  AM.HasBaseReg = true;
  AM.BaseOffs = Offset.getSExtValue();
  if (!isLegalAddressingMode(DCI.DAG.getDataLayout(), AM, Ty, AddrSpace))
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  SDValue ShlX = DAG.getNode(ISD::SHL, SL, VT, N0.getOperand(0), N1);
  SDValue COffset = DAG.getConstant(Offset, SL, MVT::i32);

  SDNodeFlags Flags;
  Flags.setNoUnsignedWrap(N->getFlags().hasNoUnsignedWrap() &&
                          (N0.getOpcode() == ISD::OR ||
                           N0->getFlags().hasNoUnsignedWrap()));

  return DAG.getNode(ISD::ADD, SL, VT, ShlX, COffset, Flags);
}

SDValue SITargetLowering::performMemSDNodeCombine(MemSDNode *N,
                                                  DAGCombinerInfo &DCI) const {
  SDValue Ptr = N->getBasePtr();
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  // TODO: We could also do this for multiplies.
  if (Ptr.getOpcode() == ISD::SHL) {
    SDValue NewPtr = performSHLPtrCombine(Ptr.getNode(),  N->getAddressSpace(),
                                          N->getMemoryVT(), DCI);
    if (NewPtr) {
      SmallVector<SDValue, 8> NewOps(N->op_begin(), N->op_end());

      NewOps[N->getOpcode() == ISD::STORE ? 2 : 1] = NewPtr;
      return SDValue(DAG.UpdateNodeOperands(N, NewOps), 0);
    }
  }

  return SDValue();
}

static bool bitOpWithConstantIsReducible(unsigned Opc, uint32_t Val) {
  return (Opc == ISD::AND && (Val == 0 || Val == 0xffffffff)) ||
         (Opc == ISD::OR && (Val == 0xffffffff || Val == 0)) ||
         (Opc == ISD::XOR && Val == 0);
}

// Break up 64-bit bit operation of a constant into two 32-bit and/or/xor. This
// will typically happen anyway for a VALU 64-bit and. This exposes other 32-bit
// integer combine opportunities since most 64-bit operations are decomposed
// this way.  TODO: We won't want this for SALU especially if it is an inline
// immediate.
SDValue SITargetLowering::splitBinaryBitConstantOp(
  DAGCombinerInfo &DCI,
  const SDLoc &SL,
  unsigned Opc, SDValue LHS,
  const ConstantSDNode *CRHS) const {
  uint64_t Val = CRHS->getZExtValue();
  uint32_t ValLo = Lo_32(Val);
  uint32_t ValHi = Hi_32(Val);
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

    if ((bitOpWithConstantIsReducible(Opc, ValLo) ||
         bitOpWithConstantIsReducible(Opc, ValHi)) ||
        (CRHS->hasOneUse() && !TII->isInlineConstant(CRHS->getAPIntValue()))) {
    // If we need to materialize a 64-bit immediate, it will be split up later
    // anyway. Avoid creating the harder to understand 64-bit immediate
    // materialization.
    return splitBinaryBitConstantOpImpl(DCI, SL, Opc, LHS, ValLo, ValHi);
  }

  return SDValue();
}

// Returns true if argument is a boolean value which is not serialized into
// memory or argument and does not require v_cmdmask_b32 to be deserialized.
static bool isBoolSGPR(SDValue V) {
  if (V.getValueType() != MVT::i1)
    return false;
  switch (V.getOpcode()) {
  default: break;
  case ISD::SETCC:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case AMDGPUISD::FP_CLASS:
    return true;
  }
  return false;
}

SDValue SITargetLowering::performAndCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalize())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);


  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (VT == MVT::i64 && CRHS) {
    if (SDValue Split
        = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::AND, LHS, CRHS))
      return Split;
  }

  if (CRHS && VT == MVT::i32) {
    // and (srl x, c), mask => shl (bfe x, nb + c, mask >> nb), nb
    // nb = number of trailing zeroes in mask
    // It can be optimized out using SDWA for GFX8+ in the SDWA peephole pass,
    // given that we are selecting 8 or 16 bit fields starting at byte boundary.
    uint64_t Mask = CRHS->getZExtValue();
    unsigned Bits = countPopulation(Mask);
    if (getSubtarget()->hasSDWA() && LHS->getOpcode() == ISD::SRL &&
        (Bits == 8 || Bits == 16) && isShiftedMask_64(Mask) && !(Mask & 1)) {
      if (auto *CShift = dyn_cast<ConstantSDNode>(LHS->getOperand(1))) {
        unsigned Shift = CShift->getZExtValue();
        unsigned NB = CRHS->getAPIntValue().countTrailingZeros();
        unsigned Offset = NB + Shift;
        if ((Offset & (Bits - 1)) == 0) { // Starts at a byte or word boundary.
          SDLoc SL(N);
          SDValue BFE = DAG.getNode(AMDGPUISD::BFE_U32, SL, MVT::i32,
                                    LHS->getOperand(0),
                                    DAG.getConstant(Offset, SL, MVT::i32),
                                    DAG.getConstant(Bits, SL, MVT::i32));
          EVT NarrowVT = EVT::getIntegerVT(*DAG.getContext(), Bits);
          SDValue Ext = DAG.getNode(ISD::AssertZext, SL, VT, BFE,
                                    DAG.getValueType(NarrowVT));
          SDValue Shl = DAG.getNode(ISD::SHL, SDLoc(LHS), VT, Ext,
                                    DAG.getConstant(NB, SDLoc(CRHS), MVT::i32));
          return Shl;
        }
      }
    }
  }

  // (and (fcmp ord x, x), (fcmp une (fabs x), inf)) ->
  // fp_class x, ~(s_nan | q_nan | n_infinity | p_infinity)
  if (LHS.getOpcode() == ISD::SETCC && RHS.getOpcode() == ISD::SETCC) {
    ISD::CondCode LCC = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
    ISD::CondCode RCC = cast<CondCodeSDNode>(RHS.getOperand(2))->get();

    SDValue X = LHS.getOperand(0);
    SDValue Y = RHS.getOperand(0);
    if (Y.getOpcode() != ISD::FABS || Y.getOperand(0) != X)
      return SDValue();

    if (LCC == ISD::SETO) {
      if (X != LHS.getOperand(1))
        return SDValue();

      if (RCC == ISD::SETUNE) {
        const ConstantFPSDNode *C1 = dyn_cast<ConstantFPSDNode>(RHS.getOperand(1));
        if (!C1 || !C1->isInfinity() || C1->isNegative())
          return SDValue();

        const uint32_t Mask = SIInstrFlags::N_NORMAL |
                              SIInstrFlags::N_SUBNORMAL |
                              SIInstrFlags::N_ZERO |
                              SIInstrFlags::P_ZERO |
                              SIInstrFlags::P_SUBNORMAL |
                              SIInstrFlags::P_NORMAL;

        static_assert(((~(SIInstrFlags::S_NAN |
                          SIInstrFlags::Q_NAN |
                          SIInstrFlags::N_INFINITY |
                          SIInstrFlags::P_INFINITY)) & 0x3ff) == Mask,
                      "mask not equal");

        SDLoc DL(N);
        return DAG.getNode(AMDGPUISD::FP_CLASS, DL, MVT::i1,
                           X, DAG.getConstant(Mask, DL, MVT::i32));
      }
    }
  }

  if (VT == MVT::i32 &&
      (RHS.getOpcode() == ISD::SIGN_EXTEND || LHS.getOpcode() == ISD::SIGN_EXTEND)) {
    // and x, (sext cc from i1) => select cc, x, 0
    if (RHS.getOpcode() != ISD::SIGN_EXTEND)
      std::swap(LHS, RHS);
    if (isBoolSGPR(RHS.getOperand(0)))
      return DAG.getSelect(SDLoc(N), MVT::i32, RHS.getOperand(0),
                           LHS, DAG.getConstant(0, SDLoc(N), MVT::i32));
  }

  return SDValue();
}

SDValue SITargetLowering::performOrCombine(SDNode *N,
                                           DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  EVT VT = N->getValueType(0);
  if (VT == MVT::i1) {
    // or (fp_class x, c1), (fp_class x, c2) -> fp_class x, (c1 | c2)
    if (LHS.getOpcode() == AMDGPUISD::FP_CLASS &&
        RHS.getOpcode() == AMDGPUISD::FP_CLASS) {
      SDValue Src = LHS.getOperand(0);
      if (Src != RHS.getOperand(0))
        return SDValue();

      const ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(LHS.getOperand(1));
      const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS.getOperand(1));
      if (!CLHS || !CRHS)
        return SDValue();

      // Only 10 bits are used.
      static const uint32_t MaxMask = 0x3ff;

      uint32_t NewMask = (CLHS->getZExtValue() | CRHS->getZExtValue()) & MaxMask;
      SDLoc DL(N);
      return DAG.getNode(AMDGPUISD::FP_CLASS, DL, MVT::i1,
                         Src, DAG.getConstant(NewMask, DL, MVT::i32));
    }

    return SDValue();
  }

  if (VT != MVT::i64)
    return SDValue();

  // TODO: This could be a generic combine with a predicate for extracting the
  // high half of an integer being free.

  // (or i64:x, (zero_extend i32:y)) ->
  //   i64 (bitcast (v2i32 build_vector (or i32:y, lo_32(x)), hi_32(x)))
  if (LHS.getOpcode() == ISD::ZERO_EXTEND &&
      RHS.getOpcode() != ISD::ZERO_EXTEND)
    std::swap(LHS, RHS);

  if (RHS.getOpcode() == ISD::ZERO_EXTEND) {
    SDValue ExtSrc = RHS.getOperand(0);
    EVT SrcVT = ExtSrc.getValueType();
    if (SrcVT == MVT::i32) {
      SDLoc SL(N);
      SDValue LowLHS, HiBits;
      std::tie(LowLHS, HiBits) = split64BitValue(LHS, DAG);
      SDValue LowOr = DAG.getNode(ISD::OR, SL, MVT::i32, LowLHS, ExtSrc);

      DCI.AddToWorklist(LowOr.getNode());
      DCI.AddToWorklist(HiBits.getNode());

      SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32,
                                LowOr, HiBits);
      return DAG.getNode(ISD::BITCAST, SL, MVT::i64, Vec);
    }
  }

  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (CRHS) {
    if (SDValue Split
          = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::OR, LHS, CRHS))
      return Split;
  }

  return SDValue();
}

SDValue SITargetLowering::performXorCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  if (VT != MVT::i64)
    return SDValue();

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (CRHS) {
    if (SDValue Split
          = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::XOR, LHS, CRHS))
      return Split;
  }

  return SDValue();
}

// Instructions that will be lowered with a final instruction that zeros the
// high result bits.
// XXX - probably only need to list legal operations.
static bool fp16SrcZerosHighBits(unsigned Opc) {
  switch (Opc) {
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::FMA:
  case ISD::FMAD:
  case ISD::FCANONICALIZE:
  case ISD::FP_ROUND:
  case ISD::UINT_TO_FP:
  case ISD::SINT_TO_FP:
  case ISD::FABS:
    // Fabs is lowered to a bit operation, but it's an and which will clear the
    // high bits anyway.
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FPOWI:
  case ISD::FPOW:
  case ISD::FLOG:
  case ISD::FLOG2:
  case ISD::FLOG10:
  case ISD::FEXP:
  case ISD::FEXP2:
  case ISD::FCEIL:
  case ISD::FTRUNC:
  case ISD::FRINT:
  case ISD::FNEARBYINT:
  case ISD::FROUND:
  case ISD::FFLOOR:
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case AMDGPUISD::FRACT:
  case AMDGPUISD::CLAMP:
  case AMDGPUISD::COS_HW:
  case AMDGPUISD::SIN_HW:
  case AMDGPUISD::FMIN3:
  case AMDGPUISD::FMAX3:
  case AMDGPUISD::FMED3:
  case AMDGPUISD::FMAD_FTZ:
  case AMDGPUISD::RCP:
  case AMDGPUISD::RSQ:
  case AMDGPUISD::LDEXP:
    return true;
  default:
    // fcopysign, select and others may be lowered to 32-bit bit operations
    // which don't zero the high bits.
    return false;
  }
}

SDValue SITargetLowering::performZeroExtendCombine(SDNode *N,
                                                   DAGCombinerInfo &DCI) const {
  if (!Subtarget->has16BitInsts() ||
      DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i32)
    return SDValue();

  SDValue Src = N->getOperand(0);
  if (Src.getValueType() != MVT::i16)
    return SDValue();

  // (i32 zext (i16 (bitcast f16:$src))) -> fp16_zext $src
  // FIXME: It is not universally true that the high bits are zeroed on gfx9.
  if (Src.getOpcode() == ISD::BITCAST) {
    SDValue BCSrc = Src.getOperand(0);
    if (BCSrc.getValueType() == MVT::f16 &&
        fp16SrcZerosHighBits(BCSrc.getOpcode()))
      return DCI.DAG.getNode(AMDGPUISD::FP16_ZEXT, SDLoc(N), VT, BCSrc);
  }

  return SDValue();
}

SDValue SITargetLowering::performClassCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue Mask = N->getOperand(1);

  // fp_class x, 0 -> false
  if (const ConstantSDNode *CMask = dyn_cast<ConstantSDNode>(Mask)) {
    if (CMask->isNullValue())
      return DAG.getConstant(0, SDLoc(N), MVT::i1);
  }

  if (N->getOperand(0).isUndef())
    return DAG.getUNDEF(MVT::i1);

  return SDValue();
}

static bool isKnownNeverSNan(SelectionDAG &DAG, SDValue Op) {
  if (!DAG.getTargetLoweringInfo().hasFloatingPointExceptions())
    return true;

  return DAG.isKnownNeverNaN(Op);
}

static bool isCanonicalized(SelectionDAG &DAG, SDValue Op,
                            const SISubtarget *ST, unsigned MaxDepth=5) {
  // If source is a result of another standard FP operation it is already in
  // canonical form.

  switch (Op.getOpcode()) {
  default:
    break;

  // These will flush denorms if required.
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FSQRT:
  case ISD::FCEIL:
  case ISD::FFLOOR:
  case ISD::FMA:
  case ISD::FMAD:

  case ISD::FCANONICALIZE:
    return true;

  case ISD::FP_ROUND:
    return Op.getValueType().getScalarType() != MVT::f16 ||
           ST->hasFP16Denormals();

  case ISD::FP_EXTEND:
    return Op.getOperand(0).getValueType().getScalarType() != MVT::f16 ||
           ST->hasFP16Denormals();

  case ISD::FP16_TO_FP:
  case ISD::FP_TO_FP16:
    return ST->hasFP16Denormals();

  // It can/will be lowered or combined as a bit operation.
  // Need to check their input recursively to handle.
  case ISD::FNEG:
  case ISD::FABS:
    return (MaxDepth > 0) &&
           isCanonicalized(DAG, Op.getOperand(0), ST, MaxDepth - 1);

  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FSINCOS:
    return Op.getValueType().getScalarType() != MVT::f16;

  // In pre-GFX9 targets V_MIN_F32 and others do not flush denorms.
  // For such targets need to check their input recursively.
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case ISD::FMINNAN:
  case ISD::FMAXNAN:

    if (ST->supportsMinMaxDenormModes() &&
        DAG.isKnownNeverNaN(Op.getOperand(0)) &&
        DAG.isKnownNeverNaN(Op.getOperand(1)))
      return true;

    return (MaxDepth > 0) &&
           isCanonicalized(DAG, Op.getOperand(0), ST, MaxDepth - 1) &&
           isCanonicalized(DAG, Op.getOperand(1), ST, MaxDepth - 1);

  case ISD::ConstantFP: {
    auto F = cast<ConstantFPSDNode>(Op)->getValueAPF();
    return !F.isDenormal() && !(F.isNaN() && F.isSignaling());
  }
  }
  return false;
}

// Constant fold canonicalize.
SDValue SITargetLowering::performFCanonicalizeCombine(
  SDNode *N,
  DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  ConstantFPSDNode *CFP = isConstOrConstSplatFP(N->getOperand(0));

  if (!CFP) {
    SDValue N0 = N->getOperand(0);
    EVT VT = N0.getValueType().getScalarType();
    auto ST = getSubtarget();

    if (((VT == MVT::f32 && ST->hasFP32Denormals()) ||
         (VT == MVT::f64 && ST->hasFP64Denormals()) ||
         (VT == MVT::f16 && ST->hasFP16Denormals())) &&
        DAG.isKnownNeverNaN(N0))
      return N0;

    bool IsIEEEMode = Subtarget->enableIEEEBit(DAG.getMachineFunction());

    if ((IsIEEEMode || isKnownNeverSNan(DAG, N0)) &&
        isCanonicalized(DAG, N0, ST))
      return N0;

    return SDValue();
  }

  const APFloat &C = CFP->getValueAPF();

  // Flush denormals to 0 if not enabled.
  if (C.isDenormal()) {
    EVT VT = N->getValueType(0);
    EVT SVT = VT.getScalarType();
    if (SVT == MVT::f32 && !Subtarget->hasFP32Denormals())
      return DAG.getConstantFP(0.0, SDLoc(N), VT);

    if (SVT == MVT::f64 && !Subtarget->hasFP64Denormals())
      return DAG.getConstantFP(0.0, SDLoc(N), VT);

    if (SVT == MVT::f16 && !Subtarget->hasFP16Denormals())
      return DAG.getConstantFP(0.0, SDLoc(N), VT);
  }

  if (C.isNaN()) {
    EVT VT = N->getValueType(0);
    APFloat CanonicalQNaN = APFloat::getQNaN(C.getSemantics());
    if (C.isSignaling()) {
      // Quiet a signaling NaN.
      return DAG.getConstantFP(CanonicalQNaN, SDLoc(N), VT);
    }

    // Make sure it is the canonical NaN bitpattern.
    //
    // TODO: Can we use -1 as the canonical NaN value since it's an inline
    // immediate?
    if (C.bitcastToAPInt() != CanonicalQNaN.bitcastToAPInt())
      return DAG.getConstantFP(CanonicalQNaN, SDLoc(N), VT);
  }

  return N->getOperand(0);
}

static unsigned minMaxOpcToMin3Max3Opc(unsigned Opc) {
  switch (Opc) {
  case ISD::FMAXNUM:
    return AMDGPUISD::FMAX3;
  case ISD::SMAX:
    return AMDGPUISD::SMAX3;
  case ISD::UMAX:
    return AMDGPUISD::UMAX3;
  case ISD::FMINNUM:
    return AMDGPUISD::FMIN3;
  case ISD::SMIN:
    return AMDGPUISD::SMIN3;
  case ISD::UMIN:
    return AMDGPUISD::UMIN3;
  default:
    llvm_unreachable("Not a min/max opcode");
  }
}

SDValue SITargetLowering::performIntMed3ImmCombine(
  SelectionDAG &DAG, const SDLoc &SL,
  SDValue Op0, SDValue Op1, bool Signed) const {
  ConstantSDNode *K1 = dyn_cast<ConstantSDNode>(Op1);
  if (!K1)
    return SDValue();

  ConstantSDNode *K0 = dyn_cast<ConstantSDNode>(Op0.getOperand(1));
  if (!K0)
    return SDValue();

  if (Signed) {
    if (K0->getAPIntValue().sge(K1->getAPIntValue()))
      return SDValue();
  } else {
    if (K0->getAPIntValue().uge(K1->getAPIntValue()))
      return SDValue();
  }

  EVT VT = K0->getValueType(0);
  unsigned Med3Opc = Signed ? AMDGPUISD::SMED3 : AMDGPUISD::UMED3;
  if (VT == MVT::i32 || (VT == MVT::i16 && Subtarget->hasMed3_16())) {
    return DAG.getNode(Med3Opc, SL, VT,
                       Op0.getOperand(0), SDValue(K0, 0), SDValue(K1, 0));
  }

  // If there isn't a 16-bit med3 operation, convert to 32-bit.
  MVT NVT = MVT::i32;
  unsigned ExtOp = Signed ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;

  SDValue Tmp1 = DAG.getNode(ExtOp, SL, NVT, Op0->getOperand(0));
  SDValue Tmp2 = DAG.getNode(ExtOp, SL, NVT, Op0->getOperand(1));
  SDValue Tmp3 = DAG.getNode(ExtOp, SL, NVT, Op1);

  SDValue Med3 = DAG.getNode(Med3Opc, SL, NVT, Tmp1, Tmp2, Tmp3);
  return DAG.getNode(ISD::TRUNCATE, SL, VT, Med3);
}

static ConstantFPSDNode *getSplatConstantFP(SDValue Op) {
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op))
    return C;

  if (BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(Op)) {
    if (ConstantFPSDNode *C = BV->getConstantFPSplatNode())
      return C;
  }

  return nullptr;
}

SDValue SITargetLowering::performFPMed3ImmCombine(SelectionDAG &DAG,
                                                  const SDLoc &SL,
                                                  SDValue Op0,
                                                  SDValue Op1) const {
  ConstantFPSDNode *K1 = getSplatConstantFP(Op1);
  if (!K1)
    return SDValue();

  ConstantFPSDNode *K0 = getSplatConstantFP(Op0.getOperand(1));
  if (!K0)
    return SDValue();

  // Ordered >= (although NaN inputs should have folded away by now).
  APFloat::cmpResult Cmp = K0->getValueAPF().compare(K1->getValueAPF());
  if (Cmp == APFloat::cmpGreaterThan)
    return SDValue();

  // TODO: Check IEEE bit enabled?
  EVT VT = Op0.getValueType();
  if (Subtarget->enableDX10Clamp()) {
    // If dx10_clamp is enabled, NaNs clamp to 0.0. This is the same as the
    // hardware fmed3 behavior converting to a min.
    // FIXME: Should this be allowing -0.0?
    if (K1->isExactlyValue(1.0) && K0->isExactlyValue(0.0))
      return DAG.getNode(AMDGPUISD::CLAMP, SL, VT, Op0.getOperand(0));
  }

  // med3 for f16 is only available on gfx9+, and not available for v2f16.
  if (VT == MVT::f32 || (VT == MVT::f16 && Subtarget->hasMed3_16())) {
    // This isn't safe with signaling NaNs because in IEEE mode, min/max on a
    // signaling NaN gives a quiet NaN. The quiet NaN input to the min would
    // then give the other result, which is different from med3 with a NaN
    // input.
    SDValue Var = Op0.getOperand(0);
    if (!isKnownNeverSNan(DAG, Var))
      return SDValue();

    return DAG.getNode(AMDGPUISD::FMED3, SL, K0->getValueType(0),
                       Var, SDValue(K0, 0), SDValue(K1, 0));
  }

  return SDValue();
}

SDValue SITargetLowering::performMinMaxCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  EVT VT = N->getValueType(0);
  unsigned Opc = N->getOpcode();
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  // Only do this if the inner op has one use since this will just increases
  // register pressure for no benefit.


  if (Opc != AMDGPUISD::FMIN_LEGACY && Opc != AMDGPUISD::FMAX_LEGACY &&
      VT != MVT::f64 &&
      ((VT != MVT::f16 && VT != MVT::i16) || Subtarget->hasMin3Max3_16())) {
    // max(max(a, b), c) -> max3(a, b, c)
    // min(min(a, b), c) -> min3(a, b, c)
    if (Op0.getOpcode() == Opc && Op0.hasOneUse()) {
      SDLoc DL(N);
      return DAG.getNode(minMaxOpcToMin3Max3Opc(Opc),
                         DL,
                         N->getValueType(0),
                         Op0.getOperand(0),
                         Op0.getOperand(1),
                         Op1);
    }

    // Try commuted.
    // max(a, max(b, c)) -> max3(a, b, c)
    // min(a, min(b, c)) -> min3(a, b, c)
    if (Op1.getOpcode() == Opc && Op1.hasOneUse()) {
      SDLoc DL(N);
      return DAG.getNode(minMaxOpcToMin3Max3Opc(Opc),
                         DL,
                         N->getValueType(0),
                         Op0,
                         Op1.getOperand(0),
                         Op1.getOperand(1));
    }
  }

  // min(max(x, K0), K1), K0 < K1 -> med3(x, K0, K1)
  if (Opc == ISD::SMIN && Op0.getOpcode() == ISD::SMAX && Op0.hasOneUse()) {
    if (SDValue Med3 = performIntMed3ImmCombine(DAG, SDLoc(N), Op0, Op1, true))
      return Med3;
  }

  if (Opc == ISD::UMIN && Op0.getOpcode() == ISD::UMAX && Op0.hasOneUse()) {
    if (SDValue Med3 = performIntMed3ImmCombine(DAG, SDLoc(N), Op0, Op1, false))
      return Med3;
  }

  // fminnum(fmaxnum(x, K0), K1), K0 < K1 && !is_snan(x) -> fmed3(x, K0, K1)
  if (((Opc == ISD::FMINNUM && Op0.getOpcode() == ISD::FMAXNUM) ||
       (Opc == AMDGPUISD::FMIN_LEGACY &&
        Op0.getOpcode() == AMDGPUISD::FMAX_LEGACY)) &&
      (VT == MVT::f32 || VT == MVT::f64 ||
       (VT == MVT::f16 && Subtarget->has16BitInsts()) ||
       (VT == MVT::v2f16 && Subtarget->hasVOP3PInsts())) &&
      Op0.hasOneUse()) {
    if (SDValue Res = performFPMed3ImmCombine(DAG, SDLoc(N), Op0, Op1))
      return Res;
  }

  return SDValue();
}

static bool isClampZeroToOne(SDValue A, SDValue B) {
  if (ConstantFPSDNode *CA = dyn_cast<ConstantFPSDNode>(A)) {
    if (ConstantFPSDNode *CB = dyn_cast<ConstantFPSDNode>(B)) {
      // FIXME: Should this be allowing -0.0?
      return (CA->isExactlyValue(0.0) && CB->isExactlyValue(1.0)) ||
             (CA->isExactlyValue(1.0) && CB->isExactlyValue(0.0));
    }
  }

  return false;
}

// FIXME: Should only worry about snans for version with chain.
SDValue SITargetLowering::performFMed3Combine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  // v_med3_f32 and v_max_f32 behave identically wrt denorms, exceptions and
  // NaNs. With a NaN input, the order of the operands may change the result.

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  SDValue Src0 = N->getOperand(0);
  SDValue Src1 = N->getOperand(1);
  SDValue Src2 = N->getOperand(2);

  if (isClampZeroToOne(Src0, Src1)) {
    // const_a, const_b, x -> clamp is safe in all cases including signaling
    // nans.
    // FIXME: Should this be allowing -0.0?
    return DAG.getNode(AMDGPUISD::CLAMP, SL, VT, Src2);
  }

  // FIXME: dx10_clamp behavior assumed in instcombine. Should we really bother
  // handling no dx10-clamp?
  if (Subtarget->enableDX10Clamp()) {
    // If NaNs is clamped to 0, we are free to reorder the inputs.

    if (isa<ConstantFPSDNode>(Src0) && !isa<ConstantFPSDNode>(Src1))
      std::swap(Src0, Src1);

    if (isa<ConstantFPSDNode>(Src1) && !isa<ConstantFPSDNode>(Src2))
      std::swap(Src1, Src2);

    if (isa<ConstantFPSDNode>(Src0) && !isa<ConstantFPSDNode>(Src1))
      std::swap(Src0, Src1);

    if (isClampZeroToOne(Src1, Src2))
      return DAG.getNode(AMDGPUISD::CLAMP, SL, VT, Src0);
  }

  return SDValue();
}

SDValue SITargetLowering::performCvtPkRTZCombine(SDNode *N,
                                                 DAGCombinerInfo &DCI) const {
  SDValue Src0 = N->getOperand(0);
  SDValue Src1 = N->getOperand(1);
  if (Src0.isUndef() && Src1.isUndef())
    return DCI.DAG.getUNDEF(N->getValueType(0));
  return SDValue();
}

SDValue SITargetLowering::performExtractVectorEltCombine(
  SDNode *N, DAGCombinerInfo &DCI) const {
  SDValue Vec = N->getOperand(0);

  SelectionDAG &DAG = DCI.DAG;
  if (Vec.getOpcode() == ISD::FNEG && allUsesHaveSourceMods(N)) {
    SDLoc SL(N);
    EVT EltVT = N->getValueType(0);
    SDValue Idx = N->getOperand(1);
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                              Vec.getOperand(0), Idx);
    return DAG.getNode(ISD::FNEG, SL, EltVT, Elt);
  }

  return SDValue();
}

static bool convertBuildVectorCastElt(SelectionDAG &DAG,
                                      SDValue &Lo, SDValue &Hi) {
  if (Hi.getOpcode() == ISD::BITCAST &&
      Hi.getOperand(0).getValueType() == MVT::f16 &&
      (isa<ConstantSDNode>(Lo) || Lo.isUndef())) {
    Lo = DAG.getNode(ISD::BITCAST, SDLoc(Lo), MVT::f16, Lo);
    Hi = Hi.getOperand(0);
    return true;
  }

  return false;
}

SDValue SITargetLowering::performBuildVectorCombine(
  SDNode *N, DAGCombinerInfo &DCI) const {
  SDLoc SL(N);

  if (!isTypeLegal(MVT::v2i16))
    return SDValue();
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  if (VT == MVT::v2i16) {
    SDValue Lo = N->getOperand(0);
    SDValue Hi = N->getOperand(1);

    // v2i16 build_vector (const|undef), (bitcast f16:$x)
    // -> bitcast (v2f16 build_vector const|undef, $x
    if (convertBuildVectorCastElt(DAG, Lo, Hi)) {
      SDValue NewVec = DAG.getBuildVector(MVT::v2f16, SL, { Lo, Hi  });
      return DAG.getNode(ISD::BITCAST, SL, VT, NewVec);
    }

    if (convertBuildVectorCastElt(DAG, Hi, Lo)) {
      SDValue NewVec = DAG.getBuildVector(MVT::v2f16, SL, { Hi, Lo  });
      return DAG.getNode(ISD::BITCAST, SL, VT, NewVec);
    }
  }

  return SDValue();
}

unsigned SITargetLowering::getFusedOpcode(const SelectionDAG &DAG,
                                          const SDNode *N0,
                                          const SDNode *N1) const {
  EVT VT = N0->getValueType(0);

  // Only do this if we are not trying to support denormals. v_mad_f32 does not
  // support denormals ever.
  if ((VT == MVT::f32 && !Subtarget->hasFP32Denormals()) ||
      (VT == MVT::f16 && !Subtarget->hasFP16Denormals()))
    return ISD::FMAD;

  const TargetOptions &Options = DAG.getTarget().Options;
  if ((Options.AllowFPOpFusion == FPOpFusion::Fast || Options.UnsafeFPMath ||
       (N0->getFlags().hasUnsafeAlgebra() &&
        N1->getFlags().hasUnsafeAlgebra())) &&
      isFMAFasterThanFMulAndFAdd(VT)) {
    return ISD::FMA;
  }

  return 0;
}

static SDValue getMad64_32(SelectionDAG &DAG, const SDLoc &SL,
                           EVT VT,
                           SDValue N0, SDValue N1, SDValue N2,
                           bool Signed) {
  unsigned MadOpc = Signed ? AMDGPUISD::MAD_I64_I32 : AMDGPUISD::MAD_U64_U32;
  SDVTList VTs = DAG.getVTList(MVT::i64, MVT::i1);
  SDValue Mad = DAG.getNode(MadOpc, SL, VTs, N0, N1, N2);
  return DAG.getNode(ISD::TRUNCATE, SL, VT, Mad);
}

SDValue SITargetLowering::performAddCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  if ((LHS.getOpcode() == ISD::MUL || RHS.getOpcode() == ISD::MUL)
      && Subtarget->hasMad64_32() &&
      !VT.isVector() && VT.getScalarSizeInBits() > 32 &&
      VT.getScalarSizeInBits() <= 64) {
    if (LHS.getOpcode() != ISD::MUL)
      std::swap(LHS, RHS);

    SDValue MulLHS = LHS.getOperand(0);
    SDValue MulRHS = LHS.getOperand(1);
    SDValue AddRHS = RHS;

    // TODO: Maybe restrict if SGPR inputs.
    if (numBitsUnsigned(MulLHS, DAG) <= 32 &&
        numBitsUnsigned(MulRHS, DAG) <= 32) {
      MulLHS = DAG.getZExtOrTrunc(MulLHS, SL, MVT::i32);
      MulRHS = DAG.getZExtOrTrunc(MulRHS, SL, MVT::i32);
      AddRHS = DAG.getZExtOrTrunc(AddRHS, SL, MVT::i64);
      return getMad64_32(DAG, SL, VT, MulLHS, MulRHS, AddRHS, false);
    }

    if (numBitsSigned(MulLHS, DAG) < 32 && numBitsSigned(MulRHS, DAG) < 32) {
      MulLHS = DAG.getSExtOrTrunc(MulLHS, SL, MVT::i32);
      MulRHS = DAG.getSExtOrTrunc(MulRHS, SL, MVT::i32);
      AddRHS = DAG.getSExtOrTrunc(AddRHS, SL, MVT::i64);
      return getMad64_32(DAG, SL, VT, MulLHS, MulRHS, AddRHS, true);
    }

    return SDValue();
  }

  if (VT != MVT::i32)
    return SDValue();

  // add x, zext (setcc) => addcarry x, 0, setcc
  // add x, sext (setcc) => subcarry x, 0, setcc
  unsigned Opc = LHS.getOpcode();
  if (Opc == ISD::ZERO_EXTEND || Opc == ISD::SIGN_EXTEND ||
      Opc == ISD::ANY_EXTEND || Opc == ISD::ADDCARRY)
    std::swap(RHS, LHS);

  Opc = RHS.getOpcode();
  switch (Opc) {
  default: break;
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ANY_EXTEND: {
    auto Cond = RHS.getOperand(0);
    if (!isBoolSGPR(Cond))
      break;
    SDVTList VTList = DAG.getVTList(MVT::i32, MVT::i1);
    SDValue Args[] = { LHS, DAG.getConstant(0, SL, MVT::i32), Cond };
    Opc = (Opc == ISD::SIGN_EXTEND) ? ISD::SUBCARRY : ISD::ADDCARRY;
    return DAG.getNode(Opc, SL, VTList, Args);
  }
  case ISD::ADDCARRY: {
    // add x, (addcarry y, 0, cc) => addcarry x, y, cc
    auto C = dyn_cast<ConstantSDNode>(RHS.getOperand(1));
    if (!C || C->getZExtValue() != 0) break;
    SDValue Args[] = { LHS, RHS.getOperand(0), RHS.getOperand(2) };
    return DAG.getNode(ISD::ADDCARRY, SDLoc(N), RHS->getVTList(), Args);
  }
  }
  return SDValue();
}

SDValue SITargetLowering::performSubCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  if (VT != MVT::i32)
    return SDValue();

  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  unsigned Opc = LHS.getOpcode();
  if (Opc != ISD::SUBCARRY)
    std::swap(RHS, LHS);

  if (LHS.getOpcode() == ISD::SUBCARRY) {
    // sub (subcarry x, 0, cc), y => subcarry x, y, cc
    auto C = dyn_cast<ConstantSDNode>(LHS.getOperand(1));
    if (!C || C->getZExtValue() != 0)
      return SDValue();
    SDValue Args[] = { LHS.getOperand(0), RHS, LHS.getOperand(2) };
    return DAG.getNode(ISD::SUBCARRY, SDLoc(N), LHS->getVTList(), Args);
  }
  return SDValue();
}

SDValue SITargetLowering::performAddCarrySubCarryCombine(SDNode *N,
  DAGCombinerInfo &DCI) const {

  if (N->getValueType(0) != MVT::i32)
    return SDValue();

  auto C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!C || C->getZExtValue() != 0)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDValue LHS = N->getOperand(0);

  // addcarry (add x, y), 0, cc => addcarry x, y, cc
  // subcarry (sub x, y), 0, cc => subcarry x, y, cc
  unsigned LHSOpc = LHS.getOpcode();
  unsigned Opc = N->getOpcode();
  if ((LHSOpc == ISD::ADD && Opc == ISD::ADDCARRY) ||
      (LHSOpc == ISD::SUB && Opc == ISD::SUBCARRY)) {
    SDValue Args[] = { LHS.getOperand(0), LHS.getOperand(1), N->getOperand(2) };
    return DAG.getNode(Opc, SDLoc(N), N->getVTList(), Args);
  }
  return SDValue();
}

SDValue SITargetLowering::performFAddCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // These should really be instruction patterns, but writing patterns with
  // source modiifiers is a pain.

  // fadd (fadd (a, a), b) -> mad 2.0, a, b
  if (LHS.getOpcode() == ISD::FADD) {
    SDValue A = LHS.getOperand(0);
    if (A == LHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, LHS.getNode());
      if (FusedOp != 0) {
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, Two, RHS);
      }
    }
  }

  // fadd (b, fadd (a, a)) -> mad 2.0, a, b
  if (RHS.getOpcode() == ISD::FADD) {
    SDValue A = RHS.getOperand(0);
    if (A == RHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, RHS.getNode());
      if (FusedOp != 0) {
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, Two, LHS);
      }
    }
  }

  return SDValue();
}

SDValue SITargetLowering::performFSubCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  EVT VT = N->getValueType(0);
  assert(!VT.isVector());

  // Try to get the fneg to fold into the source modifier. This undoes generic
  // DAG combines and folds them into the mad.
  //
  // Only do this if we are not trying to support denormals. v_mad_f32 does
  // not support denormals ever.
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  if (LHS.getOpcode() == ISD::FADD) {
    // (fsub (fadd a, a), c) -> mad 2.0, a, (fneg c)
    SDValue A = LHS.getOperand(0);
    if (A == LHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, LHS.getNode());
      if (FusedOp != 0){
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        SDValue NegRHS = DAG.getNode(ISD::FNEG, SL, VT, RHS);

        return DAG.getNode(FusedOp, SL, VT, A, Two, NegRHS);
      }
    }
  }

  if (RHS.getOpcode() == ISD::FADD) {
    // (fsub c, (fadd a, a)) -> mad -2.0, a, c

    SDValue A = RHS.getOperand(0);
    if (A == RHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, RHS.getNode());
      if (FusedOp != 0){
        const SDValue NegTwo = DAG.getConstantFP(-2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, NegTwo, LHS);
      }
    }
  }

  return SDValue();
}

SDValue SITargetLowering::performSetCCCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  EVT VT = LHS.getValueType();
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();

  auto CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (!CRHS) {
    CRHS = dyn_cast<ConstantSDNode>(LHS);
    if (CRHS) {
      std::swap(LHS, RHS);
      CC = getSetCCSwappedOperands(CC);
    }
  }

  if (CRHS && VT == MVT::i32 && LHS.getOpcode() == ISD::SIGN_EXTEND &&
      isBoolSGPR(LHS.getOperand(0))) {
    // setcc (sext from i1 cc), -1, ne|sgt|ult) => not cc => xor cc, -1
    // setcc (sext from i1 cc), -1, eq|sle|uge) => cc
    // setcc (sext from i1 cc),  0, eq|sge|ule) => not cc => xor cc, -1
    // setcc (sext from i1 cc),  0, ne|ugt|slt) => cc
    if ((CRHS->isAllOnesValue() &&
         (CC == ISD::SETNE || CC == ISD::SETGT || CC == ISD::SETULT)) ||
        (CRHS->isNullValue() &&
         (CC == ISD::SETEQ || CC == ISD::SETGE || CC == ISD::SETULE)))
      return DAG.getNode(ISD::XOR, SL, MVT::i1, LHS.getOperand(0),
                         DAG.getConstant(-1, SL, MVT::i1));
    if ((CRHS->isAllOnesValue() &&
         (CC == ISD::SETEQ || CC == ISD::SETLE || CC == ISD::SETUGE)) ||
        (CRHS->isNullValue() &&
         (CC == ISD::SETNE || CC == ISD::SETUGT || CC == ISD::SETLT)))
      return LHS.getOperand(0);
  }

  if (VT != MVT::f32 && VT != MVT::f64 && (Subtarget->has16BitInsts() &&
                                           VT != MVT::f16))
    return SDValue();

  // Match isinf pattern
  // (fcmp oeq (fabs x), inf) -> (fp_class x, (p_infinity | n_infinity))
  if (CC == ISD::SETOEQ && LHS.getOpcode() == ISD::FABS) {
    const ConstantFPSDNode *CRHS = dyn_cast<ConstantFPSDNode>(RHS);
    if (!CRHS)
      return SDValue();

    const APFloat &APF = CRHS->getValueAPF();
    if (APF.isInfinity() && !APF.isNegative()) {
      unsigned Mask = SIInstrFlags::P_INFINITY | SIInstrFlags::N_INFINITY;
      return DAG.getNode(AMDGPUISD::FP_CLASS, SL, MVT::i1, LHS.getOperand(0),
                         DAG.getConstant(Mask, SL, MVT::i32));
    }
  }

  return SDValue();
}

SDValue SITargetLowering::performCvtF32UByteNCombine(SDNode *N,
                                                     DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  unsigned Offset = N->getOpcode() - AMDGPUISD::CVT_F32_UBYTE0;

  SDValue Src = N->getOperand(0);
  SDValue Srl = N->getOperand(0);
  if (Srl.getOpcode() == ISD::ZERO_EXTEND)
    Srl = Srl.getOperand(0);

  // TODO: Handle (or x, (srl y, 8)) pattern when known bits are zero.
  if (Srl.getOpcode() == ISD::SRL) {
    // cvt_f32_ubyte0 (srl x, 16) -> cvt_f32_ubyte2 x
    // cvt_f32_ubyte1 (srl x, 16) -> cvt_f32_ubyte3 x
    // cvt_f32_ubyte0 (srl x, 8) -> cvt_f32_ubyte1 x

    if (const ConstantSDNode *C =
        dyn_cast<ConstantSDNode>(Srl.getOperand(1))) {
      Srl = DAG.getZExtOrTrunc(Srl.getOperand(0), SDLoc(Srl.getOperand(0)),
                               EVT(MVT::i32));

      unsigned SrcOffset = C->getZExtValue() + 8 * Offset;
      if (SrcOffset < 32 && SrcOffset % 8 == 0) {
        return DAG.getNode(AMDGPUISD::CVT_F32_UBYTE0 + SrcOffset / 8, SL,
                           MVT::f32, Srl);
      }
    }
  }

  APInt Demanded = APInt::getBitsSet(32, 8 * Offset, 8 * Offset + 8);

  KnownBits Known;
  TargetLowering::TargetLoweringOpt TLO(DAG, !DCI.isBeforeLegalize(),
                                        !DCI.isBeforeLegalizeOps());
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (TLI.ShrinkDemandedConstant(Src, Demanded, TLO) ||
      TLI.SimplifyDemandedBits(Src, Demanded, Known, TLO)) {
    DCI.CommitTargetLoweringOpt(TLO);
  }

  return SDValue();
}

SDValue SITargetLowering::PerformDAGCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  default:
    return AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
  case ISD::ADD:
    return performAddCombine(N, DCI);
  case ISD::SUB:
    return performSubCombine(N, DCI);
  case ISD::ADDCARRY:
  case ISD::SUBCARRY:
    return performAddCarrySubCarryCombine(N, DCI);
  case ISD::FADD:
    return performFAddCombine(N, DCI);
  case ISD::FSUB:
    return performFSubCombine(N, DCI);
  case ISD::SETCC:
    return performSetCCCombine(N, DCI);
  case ISD::FMAXNUM:
  case ISD::FMINNUM:
  case ISD::SMAX:
  case ISD::SMIN:
  case ISD::UMAX:
  case ISD::UMIN:
  case AMDGPUISD::FMIN_LEGACY:
  case AMDGPUISD::FMAX_LEGACY: {
    if (DCI.getDAGCombineLevel() >= AfterLegalizeDAG &&
        getTargetMachine().getOptLevel() > CodeGenOpt::None)
      return performMinMaxCombine(N, DCI);
    break;
  }
  case ISD::LOAD:
  case ISD::STORE:
  case ISD::ATOMIC_LOAD:
  case ISD::ATOMIC_STORE:
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS:
  case ISD::ATOMIC_SWAP:
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case AMDGPUISD::ATOMIC_INC:
  case AMDGPUISD::ATOMIC_DEC: // TODO: Target mem intrinsics.
    if (DCI.isBeforeLegalize())
      break;
    return performMemSDNodeCombine(cast<MemSDNode>(N), DCI);
  case ISD::AND:
    return performAndCombine(N, DCI);
  case ISD::OR:
    return performOrCombine(N, DCI);
  case ISD::XOR:
    return performXorCombine(N, DCI);
  case ISD::ZERO_EXTEND:
    return performZeroExtendCombine(N, DCI);
  case AMDGPUISD::FP_CLASS:
    return performClassCombine(N, DCI);
  case ISD::FCANONICALIZE:
    return performFCanonicalizeCombine(N, DCI);
  case AMDGPUISD::FRACT:
  case AMDGPUISD::RCP:
  case AMDGPUISD::RSQ:
  case AMDGPUISD::RCP_LEGACY:
  case AMDGPUISD::RSQ_LEGACY:
  case AMDGPUISD::RSQ_CLAMP:
  case AMDGPUISD::LDEXP: {
    SDValue Src = N->getOperand(0);
    if (Src.isUndef())
      return Src;
    break;
  }
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return performUCharToFloatCombine(N, DCI);
  case AMDGPUISD::CVT_F32_UBYTE0:
  case AMDGPUISD::CVT_F32_UBYTE1:
  case AMDGPUISD::CVT_F32_UBYTE2:
  case AMDGPUISD::CVT_F32_UBYTE3:
    return performCvtF32UByteNCombine(N, DCI);
  case AMDGPUISD::FMED3:
    return performFMed3Combine(N, DCI);
  case AMDGPUISD::CVT_PKRTZ_F16_F32:
    return performCvtPkRTZCombine(N, DCI);
  case ISD::SCALAR_TO_VECTOR: {
    SelectionDAG &DAG = DCI.DAG;
    EVT VT = N->getValueType(0);

    // v2i16 (scalar_to_vector i16:x) -> v2i16 (bitcast (any_extend i16:x))
    if (VT == MVT::v2i16 || VT == MVT::v2f16) {
      SDLoc SL(N);
      SDValue Src = N->getOperand(0);
      EVT EltVT = Src.getValueType();
      if (EltVT == MVT::f16)
        Src = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Src);

      SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, Src);
      return DAG.getNode(ISD::BITCAST, SL, VT, Ext);
    }

    break;
  }
  case ISD::EXTRACT_VECTOR_ELT:
    return performExtractVectorEltCombine(N, DCI);
  case ISD::BUILD_VECTOR:
    return performBuildVectorCombine(N, DCI);
  }
  return AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
}

/// \brief Helper function for adjustWritemask
static unsigned SubIdx2Lane(unsigned Idx) {
  switch (Idx) {
  default: return 0;
  case AMDGPU::sub0: return 0;
  case AMDGPU::sub1: return 1;
  case AMDGPU::sub2: return 2;
  case AMDGPU::sub3: return 3;
  }
}

/// \brief Adjust the writemask of MIMG instructions
void SITargetLowering::adjustWritemask(MachineSDNode *&Node,
                                       SelectionDAG &DAG) const {
  SDNode *Users[4] = { };
  unsigned Lane = 0;
  unsigned DmaskIdx = (Node->getNumOperands() - Node->getNumValues() == 9) ? 2 : 3;
  unsigned OldDmask = Node->getConstantOperandVal(DmaskIdx);
  unsigned NewDmask = 0;

  // Try to figure out the used register components
  for (SDNode::use_iterator I = Node->use_begin(), E = Node->use_end();
       I != E; ++I) {

    // Don't look at users of the chain.
    if (I.getUse().getResNo() != 0)
      continue;

    // Abort if we can't understand the usage
    if (!I->isMachineOpcode() ||
        I->getMachineOpcode() != TargetOpcode::EXTRACT_SUBREG)
      return;

    // Lane means which subreg of %vgpra_vgprb_vgprc_vgprd is used.
    // Note that subregs are packed, i.e. Lane==0 is the first bit set
    // in OldDmask, so it can be any of X,Y,Z,W; Lane==1 is the second bit
    // set, etc.
    Lane = SubIdx2Lane(I->getConstantOperandVal(1));

    // Set which texture component corresponds to the lane.
    unsigned Comp;
    for (unsigned i = 0, Dmask = OldDmask; i <= Lane; i++) {
      assert(Dmask);
      Comp = countTrailingZeros(Dmask);
      Dmask &= ~(1 << Comp);
    }

    // Abort if we have more than one user per component
    if (Users[Lane])
      return;

    Users[Lane] = *I;
    NewDmask |= 1 << Comp;
  }

  // Abort if there's no change
  if (NewDmask == OldDmask)
    return;

  // Adjust the writemask in the node
  std::vector<SDValue> Ops;
  Ops.insert(Ops.end(), Node->op_begin(), Node->op_begin() + DmaskIdx);
  Ops.push_back(DAG.getTargetConstant(NewDmask, SDLoc(Node), MVT::i32));
  Ops.insert(Ops.end(), Node->op_begin() + DmaskIdx + 1, Node->op_end());
  Node = (MachineSDNode*)DAG.UpdateNodeOperands(Node, Ops);

  // If we only got one lane, replace it with a copy
  // (if NewDmask has only one bit set...)
  if (NewDmask && (NewDmask & (NewDmask-1)) == 0) {
    SDValue RC = DAG.getTargetConstant(AMDGPU::VGPR_32RegClassID, SDLoc(),
                                       MVT::i32);
    SDNode *Copy = DAG.getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                      SDLoc(), Users[Lane]->getValueType(0),
                                      SDValue(Node, 0), RC);
    DAG.ReplaceAllUsesWith(Users[Lane], Copy);
    return;
  }

  // Update the users of the node with the new indices
  for (unsigned i = 0, Idx = AMDGPU::sub0; i < 4; ++i) {
    SDNode *User = Users[i];
    if (!User)
      continue;

    SDValue Op = DAG.getTargetConstant(Idx, SDLoc(User), MVT::i32);
    DAG.UpdateNodeOperands(User, User->getOperand(0), Op);

    switch (Idx) {
    default: break;
    case AMDGPU::sub0: Idx = AMDGPU::sub1; break;
    case AMDGPU::sub1: Idx = AMDGPU::sub2; break;
    case AMDGPU::sub2: Idx = AMDGPU::sub3; break;
    }
  }
}

static bool isFrameIndexOp(SDValue Op) {
  if (Op.getOpcode() == ISD::AssertZext)
    Op = Op.getOperand(0);

  return isa<FrameIndexSDNode>(Op);
}

/// \brief Legalize target independent instructions (e.g. INSERT_SUBREG)
/// with frame index operands.
/// LLVM assumes that inputs are to these instructions are registers.
SDNode *SITargetLowering::legalizeTargetIndependentNode(SDNode *Node,
                                                        SelectionDAG &DAG) const {
  if (Node->getOpcode() == ISD::CopyToReg) {
    RegisterSDNode *DestReg = cast<RegisterSDNode>(Node->getOperand(1));
    SDValue SrcVal = Node->getOperand(2);

    // Insert a copy to a VReg_1 virtual register so LowerI1Copies doesn't have
    // to try understanding copies to physical registers.
    if (SrcVal.getValueType() == MVT::i1 &&
        TargetRegisterInfo::isPhysicalRegister(DestReg->getReg())) {
      SDLoc SL(Node);
      MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
      SDValue VReg = DAG.getRegister(
        MRI.createVirtualRegister(&AMDGPU::VReg_1RegClass), MVT::i1);

      SDNode *Glued = Node->getGluedNode();
      SDValue ToVReg
        = DAG.getCopyToReg(Node->getOperand(0), SL, VReg, SrcVal,
                         SDValue(Glued, Glued ? Glued->getNumValues() - 1 : 0));
      SDValue ToResultReg
        = DAG.getCopyToReg(ToVReg, SL, SDValue(DestReg, 0),
                           VReg, ToVReg.getValue(1));
      DAG.ReplaceAllUsesWith(Node, ToResultReg.getNode());
      DAG.RemoveDeadNode(Node);
      return ToResultReg.getNode();
    }
  }

  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0; i < Node->getNumOperands(); ++i) {
    if (!isFrameIndexOp(Node->getOperand(i))) {
      Ops.push_back(Node->getOperand(i));
      continue;
    }

    SDLoc DL(Node);
    Ops.push_back(SDValue(DAG.getMachineNode(AMDGPU::S_MOV_B32, DL,
                                     Node->getOperand(i).getValueType(),
                                     Node->getOperand(i)), 0));
  }

  return DAG.UpdateNodeOperands(Node, Ops);
}

/// \brief Fold the instructions after selecting them.
SDNode *SITargetLowering::PostISelFolding(MachineSDNode *Node,
                                          SelectionDAG &DAG) const {
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  unsigned Opcode = Node->getMachineOpcode();

  if (TII->isMIMG(Opcode) && !TII->get(Opcode).mayStore() &&
      !TII->isGather4(Opcode))
    adjustWritemask(Node, DAG);

  if (Opcode == AMDGPU::INSERT_SUBREG ||
      Opcode == AMDGPU::REG_SEQUENCE) {
    legalizeTargetIndependentNode(Node, DAG);
    return Node;
  }

  switch (Opcode) {
  case AMDGPU::V_DIV_SCALE_F32:
  case AMDGPU::V_DIV_SCALE_F64: {
    // Satisfy the operand register constraint when one of the inputs is
    // undefined. Ordinarily each undef value will have its own implicit_def of
    // a vreg, so force these to use a single register.
    SDValue Src0 = Node->getOperand(0);
    SDValue Src1 = Node->getOperand(1);
    SDValue Src2 = Node->getOperand(2);

    if ((Src0.isMachineOpcode() &&
         Src0.getMachineOpcode() != AMDGPU::IMPLICIT_DEF) &&
        (Src0 == Src1 || Src0 == Src2))
      break;

    MVT VT = Src0.getValueType().getSimpleVT();
    const TargetRegisterClass *RC = getRegClassFor(VT);

    MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
    SDValue UndefReg = DAG.getRegister(MRI.createVirtualRegister(RC), VT);

    SDValue ImpDef = DAG.getCopyToReg(DAG.getEntryNode(), SDLoc(Node),
                                      UndefReg, Src0, SDValue());

    // src0 must be the same register as src1 or src2, even if the value is
    // undefined, so make sure we don't violate this constraint.
    if (Src0.isMachineOpcode() &&
        Src0.getMachineOpcode() == AMDGPU::IMPLICIT_DEF) {
      if (Src1.isMachineOpcode() &&
          Src1.getMachineOpcode() != AMDGPU::IMPLICIT_DEF)
        Src0 = Src1;
      else if (Src2.isMachineOpcode() &&
               Src2.getMachineOpcode() != AMDGPU::IMPLICIT_DEF)
        Src0 = Src2;
      else {
        assert(Src1.getMachineOpcode() == AMDGPU::IMPLICIT_DEF);
        Src0 = UndefReg;
        Src1 = UndefReg;
      }
    } else
      break;

    SmallVector<SDValue, 4> Ops = { Src0, Src1, Src2 };
    for (unsigned I = 3, N = Node->getNumOperands(); I != N; ++I)
      Ops.push_back(Node->getOperand(I));

    Ops.push_back(ImpDef.getValue(1));
    return DAG.getMachineNode(Opcode, SDLoc(Node), Node->getVTList(), Ops);
  }
  default:
    break;
  }

  return Node;
}

/// \brief Assign the register class depending on the number of
/// bits set in the writemask
void SITargetLowering::AdjustInstrPostInstrSelection(MachineInstr &MI,
                                                     SDNode *Node) const {
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

  MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();

  if (TII->isVOP3(MI.getOpcode())) {
    // Make sure constant bus requirements are respected.
    TII->legalizeOperandsVOP3(MRI, MI);
    return;
  }

  if (TII->isMIMG(MI)) {
    unsigned VReg = MI.getOperand(0).getReg();
    const TargetRegisterClass *RC = MRI.getRegClass(VReg);
    // TODO: Need mapping tables to handle other cases (register classes).
    if (RC != &AMDGPU::VReg_128RegClass)
      return;

    unsigned DmaskIdx = MI.getNumOperands() == 12 ? 3 : 4;
    unsigned Writemask = MI.getOperand(DmaskIdx).getImm();
    unsigned BitsSet = 0;
    for (unsigned i = 0; i < 4; ++i)
      BitsSet += Writemask & (1 << i) ? 1 : 0;
    switch (BitsSet) {
    default: return;
    case 1:  RC = &AMDGPU::VGPR_32RegClass; break;
    case 2:  RC = &AMDGPU::VReg_64RegClass; break;
    case 3:  RC = &AMDGPU::VReg_96RegClass; break;
    }

    unsigned NewOpcode = TII->getMaskedMIMGOp(MI.getOpcode(), BitsSet);
    MI.setDesc(TII->get(NewOpcode));
    MRI.setRegClass(VReg, RC);
    return;
  }

  // Replace unused atomics with the no return version.
  int NoRetAtomicOp = AMDGPU::getAtomicNoRetOp(MI.getOpcode());
  if (NoRetAtomicOp != -1) {
    if (!Node->hasAnyUseOfValue(0)) {
      MI.setDesc(TII->get(NoRetAtomicOp));
      MI.RemoveOperand(0);
      return;
    }

    // For mubuf_atomic_cmpswap, we need to have tablegen use an extract_subreg
    // instruction, because the return type of these instructions is a vec2 of
    // the memory type, so it can be tied to the input operand.
    // This means these instructions always have a use, so we need to add a
    // special case to check if the atomic has only one extract_subreg use,
    // which itself has no uses.
    if ((Node->hasNUsesOfValue(1, 0) &&
         Node->use_begin()->isMachineOpcode() &&
         Node->use_begin()->getMachineOpcode() == AMDGPU::EXTRACT_SUBREG &&
         !Node->use_begin()->hasAnyUseOfValue(0))) {
      unsigned Def = MI.getOperand(0).getReg();

      // Change this into a noret atomic.
      MI.setDesc(TII->get(NoRetAtomicOp));
      MI.RemoveOperand(0);

      // If we only remove the def operand from the atomic instruction, the
      // extract_subreg will be left with a use of a vreg without a def.
      // So we need to insert an implicit_def to avoid machine verifier
      // errors.
      BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
              TII->get(AMDGPU::IMPLICIT_DEF), Def);
    }
    return;
  }
}

static SDValue buildSMovImm32(SelectionDAG &DAG, const SDLoc &DL,
                              uint64_t Val) {
  SDValue K = DAG.getTargetConstant(Val, DL, MVT::i32);
  return SDValue(DAG.getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32, K), 0);
}

MachineSDNode *SITargetLowering::wrapAddr64Rsrc(SelectionDAG &DAG,
                                                const SDLoc &DL,
                                                SDValue Ptr) const {
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

  // Build the half of the subregister with the constants before building the
  // full 128-bit register. If we are building multiple resource descriptors,
  // this will allow CSEing of the 2-component register.
  const SDValue Ops0[] = {
    DAG.getTargetConstant(AMDGPU::SGPR_64RegClassID, DL, MVT::i32),
    buildSMovImm32(DAG, DL, 0),
    DAG.getTargetConstant(AMDGPU::sub0, DL, MVT::i32),
    buildSMovImm32(DAG, DL, TII->getDefaultRsrcDataFormat() >> 32),
    DAG.getTargetConstant(AMDGPU::sub1, DL, MVT::i32)
  };

  SDValue SubRegHi = SDValue(DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL,
                                                MVT::v2i32, Ops0), 0);

  // Combine the constants and the pointer.
  const SDValue Ops1[] = {
    DAG.getTargetConstant(AMDGPU::SReg_128RegClassID, DL, MVT::i32),
    Ptr,
    DAG.getTargetConstant(AMDGPU::sub0_sub1, DL, MVT::i32),
    SubRegHi,
    DAG.getTargetConstant(AMDGPU::sub2_sub3, DL, MVT::i32)
  };

  return DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL, MVT::v4i32, Ops1);
}

/// \brief Return a resource descriptor with the 'Add TID' bit enabled
///        The TID (Thread ID) is multiplied by the stride value (bits [61:48]
///        of the resource descriptor) to create an offset, which is added to
///        the resource pointer.
MachineSDNode *SITargetLowering::buildRSRC(SelectionDAG &DAG, const SDLoc &DL,
                                           SDValue Ptr, uint32_t RsrcDword1,
                                           uint64_t RsrcDword2And3) const {
  SDValue PtrLo = DAG.getTargetExtractSubreg(AMDGPU::sub0, DL, MVT::i32, Ptr);
  SDValue PtrHi = DAG.getTargetExtractSubreg(AMDGPU::sub1, DL, MVT::i32, Ptr);
  if (RsrcDword1) {
    PtrHi = SDValue(DAG.getMachineNode(AMDGPU::S_OR_B32, DL, MVT::i32, PtrHi,
                                     DAG.getConstant(RsrcDword1, DL, MVT::i32)),
                    0);
  }

  SDValue DataLo = buildSMovImm32(DAG, DL,
                                  RsrcDword2And3 & UINT64_C(0xFFFFFFFF));
  SDValue DataHi = buildSMovImm32(DAG, DL, RsrcDword2And3 >> 32);

  const SDValue Ops[] = {
    DAG.getTargetConstant(AMDGPU::SReg_128RegClassID, DL, MVT::i32),
    PtrLo,
    DAG.getTargetConstant(AMDGPU::sub0, DL, MVT::i32),
    PtrHi,
    DAG.getTargetConstant(AMDGPU::sub1, DL, MVT::i32),
    DataLo,
    DAG.getTargetConstant(AMDGPU::sub2, DL, MVT::i32),
    DataHi,
    DAG.getTargetConstant(AMDGPU::sub3, DL, MVT::i32)
  };

  return DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL, MVT::v4i32, Ops);
}

//===----------------------------------------------------------------------===//
//                         SI Inline Assembly Support
//===----------------------------------------------------------------------===//

std::pair<unsigned, const TargetRegisterClass *>
SITargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                               StringRef Constraint,
                                               MVT VT) const {
  if (!isTypeLegal(VT))
    return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);

  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 's':
    case 'r':
      switch (VT.getSizeInBits()) {
      default:
        return std::make_pair(0U, nullptr);
      case 32:
      case 16:
        return std::make_pair(0U, &AMDGPU::SReg_32_XM0RegClass);
      case 64:
        return std::make_pair(0U, &AMDGPU::SGPR_64RegClass);
      case 128:
        return std::make_pair(0U, &AMDGPU::SReg_128RegClass);
      case 256:
        return std::make_pair(0U, &AMDGPU::SReg_256RegClass);
      case 512:
        return std::make_pair(0U, &AMDGPU::SReg_512RegClass);
      }

    case 'v':
      switch (VT.getSizeInBits()) {
      default:
        return std::make_pair(0U, nullptr);
      case 32:
      case 16:
        return std::make_pair(0U, &AMDGPU::VGPR_32RegClass);
      case 64:
        return std::make_pair(0U, &AMDGPU::VReg_64RegClass);
      case 96:
        return std::make_pair(0U, &AMDGPU::VReg_96RegClass);
      case 128:
        return std::make_pair(0U, &AMDGPU::VReg_128RegClass);
      case 256:
        return std::make_pair(0U, &AMDGPU::VReg_256RegClass);
      case 512:
        return std::make_pair(0U, &AMDGPU::VReg_512RegClass);
      }
    }
  }

  if (Constraint.size() > 1) {
    const TargetRegisterClass *RC = nullptr;
    if (Constraint[1] == 'v') {
      RC = &AMDGPU::VGPR_32RegClass;
    } else if (Constraint[1] == 's') {
      RC = &AMDGPU::SGPR_32RegClass;
    }

    if (RC) {
      uint32_t Idx;
      bool Failed = Constraint.substr(2).getAsInteger(10, Idx);
      if (!Failed && Idx < RC->getNumRegs())
        return std::make_pair(RC->getRegister(Idx), RC);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

SITargetLowering::ConstraintType
SITargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 's':
    case 'v':
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

// Figure out which registers should be reserved for stack access. Only after
// the function is legalized do we know all of the non-spill stack objects or if
// calls are present.
void SITargetLowering::finalizeLowering(MachineFunction &MF) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  if (Info->isEntryFunction()) {
    // Callable functions have fixed registers used for stack access.
    reservePrivateMemoryRegs(getTargetMachine(), MF, *TRI, *Info);
  }

  // We have to assume the SP is needed in case there are calls in the function
  // during lowering. Calls are only detected after the function is
  // lowered. We're about to reserve registers, so don't bother using it if we
  // aren't really going to use it.
  bool NeedSP = !Info->isEntryFunction() ||
    MFI.hasVarSizedObjects() ||
    MFI.hasCalls();

  if (NeedSP) {
    unsigned ReservedStackPtrOffsetReg = TRI->reservedStackPtrOffsetReg(MF);
    Info->setStackPtrOffsetReg(ReservedStackPtrOffsetReg);

    assert(Info->getStackPtrOffsetReg() != Info->getFrameOffsetReg());
    assert(!TRI->isSubRegister(Info->getScratchRSrcReg(),
                               Info->getStackPtrOffsetReg()));
    MRI.replaceRegWith(AMDGPU::SP_REG, Info->getStackPtrOffsetReg());
  }

  MRI.replaceRegWith(AMDGPU::PRIVATE_RSRC_REG, Info->getScratchRSrcReg());
  MRI.replaceRegWith(AMDGPU::FP_REG, Info->getFrameOffsetReg());
  MRI.replaceRegWith(AMDGPU::SCRATCH_WAVE_OFFSET_REG,
                     Info->getScratchWaveOffsetReg());

  TargetLoweringBase::finalizeLowering(MF);
}

void SITargetLowering::computeKnownBitsForFrameIndex(const SDValue Op,
                                                     KnownBits &Known,
                                                     const APInt &DemandedElts,
                                                     const SelectionDAG &DAG,
                                                     unsigned Depth) const {
  TargetLowering::computeKnownBitsForFrameIndex(Op, Known, DemandedElts,
                                                DAG, Depth);

  if (getSubtarget()->enableHugePrivateBuffer())
    return;

  // Technically it may be possible to have a dispatch with a single workitem
  // that uses the full private memory size, but that's not really useful. We
  // can't use vaddr in MUBUF instructions if we don't know the address
  // calculation won't overflow, so assume the sign bit is never set.
  Known.Zero.setHighBits(AssumeFrameIndexHighZeroBits);
}
