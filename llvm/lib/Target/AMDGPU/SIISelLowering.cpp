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
#include <cmath>
#endif

#include "AMDGPU.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"

using namespace llvm;

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

  addRegisterClass(MVT::i32, &AMDGPU::SReg_32RegClass);
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

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::FrameIndex, MVT::i32, Custom);
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

  setOperationAction(ISD::TRUNCATE, MVT::v2i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v2f32, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i1, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i1, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i8, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i8, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::Other, Custom);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::f32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v4f32, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);

  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::BR_CC, MVT::i1, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::i64, Expand);
  setOperationAction(ISD::BR_CC, MVT::f32, Expand);
  setOperationAction(ISD::BR_CC, MVT::f64, Expand);

  // We only support LOAD/STORE and vector manipulation ops for vectors
  // with > 4 elements.
  for (MVT VT : {MVT::v8i32, MVT::v8f32, MVT::v16i32, MVT::v16f32, MVT::v2i64, MVT::v2f64}) {
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
  setTargetDAGCombine(ISD::UINT_TO_FP);
  setTargetDAGCombine(ISD::FCANONICALIZE);

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

bool SITargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                          const CallInst &CI,
                                          unsigned IntrID) const {
  switch (IntrID) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getType());
    Info.ptrVal = CI.getOperand(0);
    Info.align = 0;
    Info.vol = false;
    Info.readMem = true;
    Info.writeMem = true;
    return true;
  default:
    return false;
  }
}

bool SITargetLowering::isShuffleMaskLegal(const SmallVectorImpl<int> &,
                                          EVT) const {
  // SI has some legal vector types, but no legal vector operations. Say no
  // shuffles are legal in order to prefer scalarizing some vector operations.
  return false;
}

bool SITargetLowering::isLegalFlatAddressingMode(const AddrMode &AM) const {
  // Flat instructions do not have offsets, and only have the register
  // address.
  return AM.BaseOffs == 0 && (AM.Scale == 0 || AM.Scale == 1);
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
                                             unsigned AS) const {
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  switch (AS) {
  case AMDGPUAS::GLOBAL_ADDRESS: {
    if (Subtarget->getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
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
  case AMDGPUAS::CONSTANT_ADDRESS: {
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
      return isLegalMUBUFAddressingMode(AM);

    if (Subtarget->getGeneration() == SISubtarget::SOUTHERN_ISLANDS) {
      // SMRD instructions have an 8-bit, dword offset on SI.
      if (!isUInt<8>(AM.BaseOffs / 4))
        return false;
    } else if (Subtarget->getGeneration() == SISubtarget::SEA_ISLANDS) {
      // On CI+, this can also be a 32-bit literal constant offset. If it fits
      // in 8-bits, it can use a smaller encoding.
      if (!isUInt<32>(AM.BaseOffs / 4))
        return false;
    } else if (Subtarget->getGeneration() == SISubtarget::VOLCANIC_ISLANDS) {
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
  }

  case AMDGPUAS::PRIVATE_ADDRESS:
    return isLegalMUBUFAddressingMode(AM);

  case AMDGPUAS::LOCAL_ADDRESS:
  case AMDGPUAS::REGION_ADDRESS: {
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
  }
  case AMDGPUAS::FLAT_ADDRESS:
  case AMDGPUAS::UNKNOWN_ADDRESS_SPACE:
    // For an unknown address space, this usually means that this is for some
    // reason being used for pure arithmetic, and not based on some addressing
    // computation. We don't have instructions that compute pointers with any
    // addressing modes, so treat them as having no offset like flat
    // instructions.
    return isLegalFlatAddressingMode(AM);

  default:
    llvm_unreachable("unhandled address space");
  }
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

  if (AddrSpace == AMDGPUAS::LOCAL_ADDRESS ||
      AddrSpace == AMDGPUAS::REGION_ADDRESS) {
    // ds_read/write_b64 require 8-byte alignment, but we can do a 4 byte
    // aligned, 8 byte access in a single operation using ds_read2/write2_b32
    // with adjacent offsets.
    bool AlignedBy4 = (Align % 4 == 0);
    if (IsFast)
      *IsFast = AlignedBy4;

    return AlignedBy4;
  }

  if (Subtarget->hasUnalignedBufferAccess()) {
    // If we have an uniform constant load, it still requires using a slow
    // buffer instruction if unaligned.
    if (IsFast) {
      *IsFast = (AddrSpace == AMDGPUAS::CONSTANT_ADDRESS) ?
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

static bool isFlatGlobalAddrSpace(unsigned AS) {
  return AS == AMDGPUAS::GLOBAL_ADDRESS ||
    AS == AMDGPUAS::FLAT_ADDRESS ||
    AS == AMDGPUAS::CONSTANT_ADDRESS;
}

bool SITargetLowering::isNoopAddrSpaceCast(unsigned SrcAS,
                                           unsigned DestAS) const {
  return isFlatGlobalAddrSpace(SrcAS) && isFlatGlobalAddrSpace(DestAS);
}

bool SITargetLowering::isMemOpUniform(const SDNode *N) const {
  const MemSDNode *MemNode = cast<MemSDNode>(N);
  const Value *Ptr = MemNode->getMemOperand()->getValue();

  // UndefValue means this is a load of a kernel input.  These are uniform.
  // Sometimes LDS instructions have constant pointers.
  // If Ptr is null, then that means this mem operand contains a
  // PseudoSourceValue like GOT.
  if (!Ptr || isa<UndefValue>(Ptr) || isa<Argument>(Ptr) ||
      isa<Constant>(Ptr) || isa<GlobalValue>(Ptr))
    return true;

  const Instruction *I = dyn_cast<Instruction>(Ptr);
  return I && I->getMetadata("amdgpu.uniform");
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

  // SimplifySetCC uses this function to determine whether or not it should
  // create setcc with i1 operands.  We don't have instructions for i1 setcc.
  if (VT == MVT::i1 && Op == ISD::SETCC)
    return false;

  return TargetLowering::isTypeDesirableForOp(Op, VT);
}

SDValue SITargetLowering::LowerParameterPtr(SelectionDAG &DAG,
                                            const SDLoc &SL, SDValue Chain,
                                            unsigned Offset) const {
  const DataLayout &DL = DAG.getDataLayout();
  MachineFunction &MF = DAG.getMachineFunction();
  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();
  unsigned InputPtrReg = TRI->getPreloadedValue(MF, SIRegisterInfo::KERNARG_SEGMENT_PTR);

  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
  MVT PtrVT = getPointerTy(DL, AMDGPUAS::CONSTANT_ADDRESS);
  SDValue BasePtr = DAG.getCopyFromReg(Chain, SL,
                                       MRI.getLiveInVirtReg(InputPtrReg), PtrVT);
  return DAG.getNode(ISD::ADD, SL, PtrVT, BasePtr,
                     DAG.getConstant(Offset, SL, PtrVT));
}
SDValue SITargetLowering::LowerParameter(SelectionDAG &DAG, EVT VT, EVT MemVT,
                                         const SDLoc &SL, SDValue Chain,
                                         unsigned Offset, bool Signed) const {
  const DataLayout &DL = DAG.getDataLayout();
  Type *Ty = VT.getTypeForEVT(*DAG.getContext());
  MVT PtrVT = getPointerTy(DL, AMDGPUAS::CONSTANT_ADDRESS);
  PointerType *PtrTy = PointerType::get(Ty, AMDGPUAS::CONSTANT_ADDRESS);
  SDValue PtrOffset = DAG.getUNDEF(PtrVT);
  MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));

  unsigned Align = DL.getABITypeAlignment(Ty);

  ISD::LoadExtType ExtTy = Signed ? ISD::SEXTLOAD : ISD::ZEXTLOAD;
  if (MemVT.isFloatingPoint())
    ExtTy = ISD::EXTLOAD;

  SDValue Ptr = LowerParameterPtr(DAG, SL, Chain, Offset);
  return DAG.getLoad(ISD::UNINDEXED, ExtTy, VT, SL, Chain, Ptr, PtrOffset,
                     PtrInfo, MemVT, Align, MachineMemOperand::MONonTemporal |
                                                MachineMemOperand::MOInvariant);
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
  BitVector Skipped(Ins.size());

  for (unsigned i = 0, e = Ins.size(), PSInputNum = 0; i != e; ++i) {
    const ISD::InputArg &Arg = Ins[i];

    // First check if it's a PS input addr
    if (CallConv == CallingConv::AMDGPU_PS && !Arg.Flags.isInReg() &&
        !Arg.Flags.isByVal() && PSInputNum <= 15) {

      if (!Arg.Used && !Info->isPSInputAllocated(PSInputNum)) {
        // We can safely skip PS inputs
        Skipped.set(i);
        ++PSInputNum;
        continue;
      }

      Info->markPSInputAllocated(PSInputNum);
      if (Arg.Used)
        Info->PSInputEna |= 1 << PSInputNum;

      ++PSInputNum;
    }

    if (AMDGPU::isShader(CallConv)) {
      // Second split vertices into their elements
      if (Arg.VT.isVector()) {
        ISD::InputArg NewArg = Arg;
        NewArg.Flags.setSplit();
        NewArg.VT = Arg.VT.getVectorElementType();

        // We REALLY want the ORIGINAL number of vertex elements here, e.g. a
        // three or five element vertex only needs three or five registers,
        // NOT four or eight.
        Type *ParamType = FType->getParamType(Arg.getOrigArgIndex());
        unsigned NumElements = ParamType->getVectorNumElements();

        for (unsigned j = 0; j != NumElements; ++j) {
          Splits.push_back(NewArg);
          NewArg.PartOffset += NewArg.VT.getStoreSize();
        }
      } else {
        Splits.push_back(Arg);
      }
    }
  }

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  // At least one interpolation mode must be enabled or else the GPU will hang.
  //
  // Check PSInputAddr instead of PSInputEna. The idea is that if the user set
  // PSInputAddr, the user wants to enable some bits after the compilation
  // based on run-time states. Since we can't know what the final PSInputEna
  // will look like, so we shouldn't do anything here and the user should take
  // responsibility for the correct programming.
  //
  // Otherwise, the following restrictions apply:
  // - At least one of PERSP_* (0xF) or LINEAR_* (0x70) must be enabled.
  // - If POS_W_FLOAT (11) is enabled, at least one of PERSP_* must be
  //   enabled too.
  if (CallConv == CallingConv::AMDGPU_PS &&
      ((Info->getPSInputAddr() & 0x7F) == 0 ||
       ((Info->getPSInputAddr() & 0xF) == 0 && Info->isPSInputAllocated(11)))) {
    CCInfo.AllocateReg(AMDGPU::VGPR0);
    CCInfo.AllocateReg(AMDGPU::VGPR1);
    Info->markPSInputAllocated(0);
    Info->PSInputEna |= 1;
  }

  if (!AMDGPU::isShader(CallConv)) {
    getOriginalFunctionArgs(DAG, DAG.getMachineFunction().getFunction(), Ins,
                            Splits);

    assert(Info->hasWorkGroupIDX() && Info->hasWorkItemIDX());
  } else {
    assert(!Info->hasPrivateSegmentBuffer() && !Info->hasDispatchPtr() &&
           !Info->hasKernargSegmentPtr() && !Info->hasFlatScratchInit() &&
           !Info->hasWorkGroupIDX() && !Info->hasWorkGroupIDY() &&
           !Info->hasWorkGroupIDZ() && !Info->hasWorkGroupInfo() &&
           !Info->hasWorkItemIDX() && !Info->hasWorkItemIDY() &&
           !Info->hasWorkItemIDZ());
  }

  // FIXME: How should these inputs interact with inreg / custom SGPR inputs?
  if (Info->hasPrivateSegmentBuffer()) {
    unsigned PrivateSegmentBufferReg = Info->addPrivateSegmentBuffer(*TRI);
    MF.addLiveIn(PrivateSegmentBufferReg, &AMDGPU::SReg_128RegClass);
    CCInfo.AllocateReg(PrivateSegmentBufferReg);
  }

  if (Info->hasDispatchPtr()) {
    unsigned DispatchPtrReg = Info->addDispatchPtr(*TRI);
    MF.addLiveIn(DispatchPtrReg, &AMDGPU::SReg_64RegClass);
    CCInfo.AllocateReg(DispatchPtrReg);
  }

  if (Info->hasQueuePtr()) {
    unsigned QueuePtrReg = Info->addQueuePtr(*TRI);
    MF.addLiveIn(QueuePtrReg, &AMDGPU::SReg_64RegClass);
    CCInfo.AllocateReg(QueuePtrReg);
  }

  if (Info->hasKernargSegmentPtr()) {
    unsigned InputPtrReg = Info->addKernargSegmentPtr(*TRI);
    MF.addLiveIn(InputPtrReg, &AMDGPU::SReg_64RegClass);
    CCInfo.AllocateReg(InputPtrReg);
  }

  if (Info->hasDispatchID()) {
    unsigned DispatchIDReg = Info->addDispatchID(*TRI);
    MF.addLiveIn(DispatchIDReg, &AMDGPU::SReg_64RegClass);
    CCInfo.AllocateReg(DispatchIDReg);
  }

  if (Info->hasFlatScratchInit()) {
    unsigned FlatScratchInitReg = Info->addFlatScratchInit(*TRI);
    MF.addLiveIn(FlatScratchInitReg, &AMDGPU::SReg_64RegClass);
    CCInfo.AllocateReg(FlatScratchInitReg);
  }

  AnalyzeFormalArguments(CCInfo, Splits);

  SmallVector<SDValue, 16> Chains;

  for (unsigned i = 0, e = Ins.size(), ArgIdx = 0; i != e; ++i) {

    const ISD::InputArg &Arg = Ins[i];
    if (Skipped[i]) {
      InVals.push_back(DAG.getUNDEF(Arg.VT));
      continue;
    }

    CCValAssign &VA = ArgLocs[ArgIdx++];
    MVT VT = VA.getLocVT();

    if (VA.isMemLoc()) {
      VT = Ins[i].VT;
      EVT MemVT = Splits[i].VT;
      const unsigned Offset = Subtarget->getExplicitKernelArgOffset() +
                              VA.getLocMemOffset();
      // The first 36 bytes of the input buffer contains information about
      // thread group and global sizes.
      SDValue Arg = LowerParameter(DAG, VT, MemVT,  DL, Chain,
                                   Offset, Ins[i].Flags.isSExt());
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
      Info->setABIArgOffset(Offset + MemVT.getStoreSize());
      continue;
    }
    assert(VA.isRegLoc() && "Parameter must be in a register!");

    unsigned Reg = VA.getLocReg();

    if (VT == MVT::i64) {
      // For now assume it is a pointer
      Reg = TRI->getMatchingSuperReg(Reg, AMDGPU::sub0,
                                     &AMDGPU::SReg_64RegClass);
      Reg = MF.addLiveIn(Reg, &AMDGPU::SReg_64RegClass);
      SDValue Copy = DAG.getCopyFromReg(Chain, DL, Reg, VT);
      InVals.push_back(Copy);
      continue;
    }

    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg, VT);

    Reg = MF.addLiveIn(Reg, RC);
    SDValue Val = DAG.getCopyFromReg(Chain, DL, Reg, VT);

    if (Arg.VT.isVector()) {

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

  // TODO: Add GridWorkGroupCount user SGPRs when used. For now with HSA we read
  // these from the dispatch pointer.

  // Start adding system SGPRs.
  if (Info->hasWorkGroupIDX()) {
    unsigned Reg = Info->addWorkGroupIDX();
    MF.addLiveIn(Reg, &AMDGPU::SReg_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info->hasWorkGroupIDY()) {
    unsigned Reg = Info->addWorkGroupIDY();
    MF.addLiveIn(Reg, &AMDGPU::SReg_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info->hasWorkGroupIDZ()) {
    unsigned Reg = Info->addWorkGroupIDZ();
    MF.addLiveIn(Reg, &AMDGPU::SReg_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info->hasWorkGroupInfo()) {
    unsigned Reg = Info->addWorkGroupInfo();
    MF.addLiveIn(Reg, &AMDGPU::SReg_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info->hasPrivateSegmentWaveByteOffset()) {
    // Scratch wave offset passed in system SGPR.
    unsigned PrivateSegmentWaveByteOffsetReg;

    if (AMDGPU::isShader(CallConv)) {
      PrivateSegmentWaveByteOffsetReg = findFirstFreeSGPR(CCInfo);
      Info->setPrivateSegmentWaveByteOffset(PrivateSegmentWaveByteOffsetReg);
    } else
      PrivateSegmentWaveByteOffsetReg = Info->addPrivateSegmentWaveByteOffset();

    MF.addLiveIn(PrivateSegmentWaveByteOffsetReg, &AMDGPU::SGPR_32RegClass);
    CCInfo.AllocateReg(PrivateSegmentWaveByteOffsetReg);
  }

  // Now that we've figured out where the scratch register inputs are, see if
  // should reserve the arguments and use them directly.
  bool HasStackObjects = MF.getFrameInfo().hasStackObjects();
  // Record that we know we have non-spill stack objects so we don't need to
  // check all stack objects later.
  if (HasStackObjects)
    Info->setHasNonSpillStackObjects(true);

  if (ST.isAmdHsaOS()) {
    // TODO: Assume we will spill without optimizations.
    if (HasStackObjects) {
      // If we have stack objects, we unquestionably need the private buffer
      // resource. For the HSA ABI, this will be the first 4 user SGPR
      // inputs. We can reserve those and use them directly.

      unsigned PrivateSegmentBufferReg = TRI->getPreloadedValue(
        MF, SIRegisterInfo::PRIVATE_SEGMENT_BUFFER);
      Info->setScratchRSrcReg(PrivateSegmentBufferReg);

      unsigned PrivateSegmentWaveByteOffsetReg = TRI->getPreloadedValue(
        MF, SIRegisterInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);
      Info->setScratchWaveOffsetReg(PrivateSegmentWaveByteOffsetReg);
    } else {
      unsigned ReservedBufferReg
        = TRI->reservedPrivateSegmentBufferReg(MF);
      unsigned ReservedOffsetReg
        = TRI->reservedPrivateSegmentWaveByteOffsetReg(MF);

      // We tentatively reserve the last registers (skipping the last two
      // which may contain VCC). After register allocation, we'll replace
      // these with the ones immediately after those which were really
      // allocated. In the prologue copies will be inserted from the argument
      // to these reserved registers.
      Info->setScratchRSrcReg(ReservedBufferReg);
      Info->setScratchWaveOffsetReg(ReservedOffsetReg);
    }
  } else {
    unsigned ReservedBufferReg = TRI->reservedPrivateSegmentBufferReg(MF);

    // Without HSA, relocations are used for the scratch pointer and the
    // buffer resource setup is always inserted in the prologue. Scratch wave
    // offset is still in an input SGPR.
    Info->setScratchRSrcReg(ReservedBufferReg);

    if (HasStackObjects) {
      unsigned ScratchWaveOffsetReg = TRI->getPreloadedValue(
        MF, SIRegisterInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);
      Info->setScratchWaveOffsetReg(ScratchWaveOffsetReg);
    } else {
      unsigned ReservedOffsetReg
        = TRI->reservedPrivateSegmentWaveByteOffsetReg(MF);
      Info->setScratchWaveOffsetReg(ReservedOffsetReg);
    }
  }

  if (Info->hasWorkItemIDX()) {
    unsigned Reg = TRI->getPreloadedValue(MF, SIRegisterInfo::WORKITEM_ID_X);
    MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info->hasWorkItemIDY()) {
    unsigned Reg = TRI->getPreloadedValue(MF, SIRegisterInfo::WORKITEM_ID_Y);
    MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info->hasWorkItemIDZ()) {
    unsigned Reg = TRI->getPreloadedValue(MF, SIRegisterInfo::WORKITEM_ID_Z);
    MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Chains.empty())
    return Chain;

  return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
}

SDValue
SITargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals,
                              const SDLoc &DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  if (!AMDGPU::isShader(CallConv))
    return AMDGPUTargetLowering::LowerReturn(Chain, CallConv, isVarArg, Outs,
                                             OutVals, DL, DAG);

  Info->setIfReturnsVoid(Outs.size() == 0);

  SmallVector<ISD::OutputArg, 48> Splits;
  SmallVector<SDValue, 48> SplitVals;

  // Split vectors into their elements.
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    const ISD::OutputArg &Out = Outs[i];

    if (Out.VT.isVector()) {
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
  AnalyzeReturn(CCInfo, Splits);

  SDValue Flag;
  SmallVector<SDValue, 48> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)

  // Copy the result values into the output registers.
  for (unsigned i = 0, realRVLocIdx = 0;
       i != RVLocs.size();
       ++i, ++realRVLocIdx) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    SDValue Arg = SplitVals[realRVLocIdx];

    // Copied from other backends.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    }

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Arg, Flag);
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Flag.getNode())
    RetOps.push_back(Flag);

  unsigned Opc = Info->returnsVoid() ? AMDGPUISD::ENDPGM : AMDGPUISD::RETURN;
  return DAG.getNode(Opc, DL, MVT::Other, RetOps);
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
    MI.setDesc(TII->get(AMDGPU::SI_KILL_TERMINATOR));
    return BB;
  }

  MachineFunction *MF = BB->getParent();
  MachineBasicBlock *SplitBB
    = MF->CreateMachineBasicBlock(BB->getBasicBlock());

  MF->insert(++MachineFunction::iterator(BB), SplitBB);
  SplitBB->splice(SplitBB->begin(), BB, SplitPoint, BB->end());

  SplitBB->transferSuccessorsAndUpdatePHIs(BB);
  BB->addSuccessor(SplitBB);

  MI.setDesc(TII->get(AMDGPU::SI_KILL_TERMINATOR));
  return SplitBB;
}

// Do a v_movrels_b32 or v_movreld_b32 for each unique value of \p IdxReg in the
// wavefront. If the value is uniform and just happens to be in a VGPR, this
// will only do one iteration. In the worst case, this will loop 64 times.
//
// TODO: Just use v_readlane_b32 if we know the VGPR has a uniform value.
static void emitLoadM0FromVGPRLoop(const SIInstrInfo *TII,
                                   MachineRegisterInfo &MRI,
                                   MachineBasicBlock &OrigBB,
                                   MachineBasicBlock &LoopBB,
                                   const DebugLoc &DL,
                                   MachineInstr *MovRel,
                                   const MachineOperand &IdxReg,
                                   unsigned InitReg,
                                   unsigned ResultReg,
                                   unsigned PhiReg,
                                   unsigned InitSaveExecReg,
                                   int Offset) {
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

  // Move index from VCC into M0
  if (Offset == 0) {
    BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
      .addReg(CurrentIdxReg, RegState::Kill);
  } else {
    BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
      .addReg(CurrentIdxReg, RegState::Kill)
      .addImm(Offset);
  }

  // Update EXEC, save the original EXEC value to VCC.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_AND_SAVEEXEC_B64), NewExec)
    .addReg(CondReg, RegState::Kill);

  MRI.setSimpleHint(NewExec, CondReg);

  // Do the actual move.
  LoopBB.insert(I, MovRel);

  // Update EXEC, switch all done bits to 0 and all todo bits to 1.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_XOR_B64), AMDGPU::EXEC)
    .addReg(AMDGPU::EXEC)
    .addReg(NewExec);

  // XXX - s_xor_b64 sets scc to 1 if the result is nonzero, so can we use
  // s_cbranch_scc0?

  // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
    .addMBB(&LoopBB);
}

// This has slightly sub-optimal regalloc when the source vector is killed by
// the read. The register allocator does not understand that the kill is
// per-workitem, so is kept alive for the whole loop so we end up not re-using a
// subregister from it, using 1 more VGPR than necessary. This was saved when
// this was expanded after register allocation.
static MachineBasicBlock *loadM0FromVGPR(const SIInstrInfo *TII,
                                         MachineBasicBlock &MBB,
                                         MachineInstr &MI,
                                         MachineInstr *MovRel,
                                         unsigned InitResultReg,
                                         unsigned PhiReg,
                                         int Offset) {
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned SaveExec = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
  unsigned TmpExec = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);

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

  emitLoadM0FromVGPRLoop(TII, MRI, MBB, *LoopBB, DL, MovRel, *Idx,
                         InitResultReg, DstReg, PhiReg, TmpExec, Offset);

  MachineBasicBlock::iterator First = RemainderBB->begin();
  BuildMI(*RemainderBB, First, DL, TII->get(AMDGPU::S_MOV_B64), AMDGPU::EXEC)
    .addReg(SaveExec);

  MI.eraseFromParent();

  return RemainderBB;
}

// Returns subreg index, offset
static std::pair<unsigned, int>
computeIndirectRegAndOffset(const SIRegisterInfo &TRI,
                            const TargetRegisterClass *SuperRC,
                            unsigned VecReg,
                            int Offset) {
  int NumElts = SuperRC->getSize() / 4;

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
                                 int Offset) {
  MachineBasicBlock *MBB = MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
  const TargetRegisterClass *IdxRC = MRI.getRegClass(Idx->getReg());

  assert(Idx->getReg() != AMDGPU::NoRegister);

  if (!TII->getRegisterInfo().isSGPRClass(IdxRC))
    return false;

  if (Offset == 0) {
    BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
      .addOperand(*Idx);
  } else {
    BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
      .addOperand(*Idx)
      .addImm(Offset);
  }

  return true;
}

// Control flow needs to be inserted if indexing with a VGPR.
static MachineBasicBlock *emitIndirectSrc(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          const SIInstrInfo *TII) {
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  unsigned Dst = MI.getOperand(0).getReg();
  const MachineOperand *SrcVec = TII->getNamedOperand(MI, AMDGPU::OpName::src);
  int Offset = TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm();

  const TargetRegisterClass *VecRC = MRI.getRegClass(SrcVec->getReg());

  unsigned SubReg;
  std::tie(SubReg, Offset)
    = computeIndirectRegAndOffset(TRI, VecRC, SrcVec->getReg(), Offset);

  if (setM0ToIndexFromSGPR(TII, MRI, MI, Offset)) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MOVRELS_B32_e32), Dst)
      .addReg(SrcVec->getReg(), RegState::Undef, SubReg)
      .addReg(SrcVec->getReg(), RegState::Implicit);
    MI.eraseFromParent();

    return &MBB;
  }

  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  unsigned PhiReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  unsigned InitReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  BuildMI(MBB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), InitReg);

  MachineInstr *MovRel =
    BuildMI(*MF, DL, TII->get(AMDGPU::V_MOVRELS_B32_e32), Dst)
    .addReg(SrcVec->getReg(), RegState::Undef, SubReg)
    .addReg(SrcVec->getReg(), RegState::Implicit);

  return loadM0FromVGPR(TII, MBB, MI, MovRel, InitReg, PhiReg, Offset);
}

static MachineBasicBlock *emitIndirectDst(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          const SIInstrInfo *TII) {
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
  if (Idx->getReg() == AMDGPU::NoRegister) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    assert(Offset == 0);

    BuildMI(MBB, I, DL, TII->get(TargetOpcode::INSERT_SUBREG), Dst)
      .addOperand(*SrcVec)
      .addOperand(*Val)
      .addImm(SubReg);

    MI.eraseFromParent();
    return &MBB;
  }

  const MCInstrDesc &MovRelDesc = TII->get(AMDGPU::V_MOVRELD_B32_e32);
  if (setM0ToIndexFromSGPR(TII, MRI, MI, Offset)) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    MachineInstr *MovRel =
      BuildMI(MBB, I, DL, MovRelDesc)
      .addReg(SrcVec->getReg(), RegState::Undef, SubReg) // vdst
      .addOperand(*Val)
      .addReg(Dst, RegState::ImplicitDefine)
      .addReg(SrcVec->getReg(), RegState::Implicit);

    const int ImpDefIdx = MovRelDesc.getNumOperands() +
      MovRelDesc.getNumImplicitUses();
    const int ImpUseIdx = ImpDefIdx + 1;

    MovRel->tieOperands(ImpDefIdx, ImpUseIdx);
    MI.eraseFromParent();
    return &MBB;
  }

  if (Val->isReg())
    MRI.clearKillFlags(Val->getReg());

  const DebugLoc &DL = MI.getDebugLoc();
  unsigned PhiReg = MRI.createVirtualRegister(VecRC);

  // vdst is not actually read and just provides the base register index.
  MachineInstr *MovRel =
    BuildMI(*MF, DL, MovRelDesc)
    .addReg(PhiReg, RegState::Undef, SubReg) // vdst
    .addOperand(*Val)
    .addReg(Dst, RegState::ImplicitDefine)
    .addReg(PhiReg, RegState::Implicit);

  const int ImpDefIdx = MovRelDesc.getNumOperands() +
    MovRelDesc.getNumImplicitUses();
  const int ImpUseIdx = ImpDefIdx + 1;

  MovRel->tieOperands(ImpDefIdx, ImpUseIdx);

  return loadM0FromVGPR(TII, MBB, MI, MovRel,
                        SrcVec->getReg(), PhiReg, Offset);
}

MachineBasicBlock *SITargetLowering::EmitInstrWithCustomInserter(
  MachineInstr &MI, MachineBasicBlock *BB) const {
  switch (MI.getOpcode()) {
  case AMDGPU::SI_INIT_M0: {
    const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
    BuildMI(*BB, MI.getIterator(), MI.getDebugLoc(),
            TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
      .addOperand(MI.getOperand(0));
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::GET_GROUPSTATICSIZE: {
    const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

    MachineFunction *MF = BB->getParent();
    SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
    DebugLoc DL = MI.getDebugLoc();
    BuildMI(*BB, MI, DL, TII->get(AMDGPU::S_MOV_B32))
      .addOperand(MI.getOperand(0))
      .addImm(MFI->getLDSSize());
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::SI_INDIRECT_SRC_V1:
  case AMDGPU::SI_INDIRECT_SRC_V2:
  case AMDGPU::SI_INDIRECT_SRC_V4:
  case AMDGPU::SI_INDIRECT_SRC_V8:
  case AMDGPU::SI_INDIRECT_SRC_V16:
    return emitIndirectSrc(MI, *BB, getSubtarget()->getInstrInfo());
  case AMDGPU::SI_INDIRECT_DST_V1:
  case AMDGPU::SI_INDIRECT_DST_V2:
  case AMDGPU::SI_INDIRECT_DST_V4:
  case AMDGPU::SI_INDIRECT_DST_V8:
  case AMDGPU::SI_INDIRECT_DST_V16:
    return emitIndirectDst(MI, *BB, getSubtarget()->getInstrInfo());
  case AMDGPU::SI_KILL:
    return splitKillBlock(MI, BB);
  default:
    return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
  }
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

MVT SITargetLowering::getScalarShiftAmountTy(const DataLayout &, EVT) const {
  return MVT::i32;
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

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
    // This is as fast on some subtargets. However, we always have full rate f32
    // mad available which returns the same result as the separate operations
    // which we should prefer over fma. We can't use this if we want to support
    // denormals, so only report this in these cases.
    return Subtarget->hasFP32Denormals() && Subtarget->hasFastFMAF32();
  case MVT::f64:
    return true;
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
  case ISD::FrameIndex: return LowerFrameIndex(Op, DAG);
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
  case ISD::TRAP: return lowerTRAP(Op, DAG);
  }
  return SDValue();
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

SDValue SITargetLowering::LowerFrameIndex(SDValue Op, SelectionDAG &DAG) const {

  SDLoc SL(Op);
  FrameIndexSDNode *FINode = cast<FrameIndexSDNode>(Op);
  unsigned FrameIndex = FINode->getIndex();

  // A FrameIndex node represents a 32-bit offset into scratch memory. If the
  // high bit of a frame index offset were to be set, this would mean that it
  // represented an offset of ~2GB * 64 = ~128GB from the start of the scratch
  // buffer, with 64 being the number of threads per wave.
  //
  // The maximum private allocation for the entire GPU is 4G, and we are
  // concerned with the largest the index could ever be for an individual
  // workitem. This will occur with the minmum dispatch size. If a program
  // requires more, the dispatch size will be reduced.
  //
  // With this limit, we can mark the high bit of the FrameIndex node as known
  // zero, which is important, because it means in most situations we can prove
  // that values derived from FrameIndex nodes are non-negative. This enables us
  // to take advantage of more addressing modes when accessing scratch buffers,
  // since for scratch reads/writes, the register offset must always be
  // positive.

  uint64_t MaxGPUAlloc = UINT64_C(4) * 1024 * 1024 * 1024;

  // XXX - It is unclear if partial dispatch works. Assume it works at half wave
  // granularity. It is probably a full wave.
  uint64_t MinGranularity = 32;

  unsigned KnownBits = Log2_64(MaxGPUAlloc / MinGranularity);
  EVT ExtVT = EVT::getIntegerVT(*DAG.getContext(), KnownBits);

  SDValue TFI = DAG.getTargetFrameIndex(FrameIndex, MVT::i32);
  return DAG.getNode(ISD::AssertZext, SL, MVT::i32, TFI,
                     DAG.getValueType(ExtVT));
}

bool SITargetLowering::isCFIntrinsic(const SDNode *Intr) const {
  if (Intr->getOpcode() != ISD::INTRINSIC_W_CHAIN)
    return false;

  switch (cast<ConstantSDNode>(Intr->getOperand(1))->getZExtValue()) {
  default: return false;
  case AMDGPUIntrinsic::amdgcn_if:
  case AMDGPUIntrinsic::amdgcn_else:
  case AMDGPUIntrinsic::amdgcn_break:
  case AMDGPUIntrinsic::amdgcn_if_break:
  case AMDGPUIntrinsic::amdgcn_else_break:
  case AMDGPUIntrinsic::amdgcn_loop:
  case AMDGPUIntrinsic::amdgcn_end_cf:
    return true;
  }
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

  if (!isCFIntrinsic(Intr)) {
    // This is a uniform branch so we don't need to legalize.
    return BRCOND;
  }

  assert(!SetCC ||
        (SetCC->getConstantOperandVal(1) == 1 &&
         cast<CondCodeSDNode>(SetCC->getOperand(2).getNode())->get() ==
                                                             ISD::SETNE));

  // Build the result and
  ArrayRef<EVT> Res(Intr->value_begin() + 1, Intr->value_end());

  // operands of the new intrinsic call
  SmallVector<SDValue, 4> Ops;
  Ops.push_back(BRCOND.getOperand(0));
  Ops.append(Intr->op_begin() + 1, Intr->op_end());
  Ops.push_back(Target);

  // build the new intrinsic call
  SDNode *Result = DAG.getNode(
    Res.size() > 1 ? ISD::INTRINSIC_W_CHAIN : ISD::INTRINSIC_VOID, DL,
    DAG.getVTList(Res), Ops).getNode();

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

SDValue SITargetLowering::getSegmentAperture(unsigned AS,
                                             SelectionDAG &DAG) const {
  SDLoc SL;
  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  unsigned UserSGPR = Info->getQueuePtrUserSGPR();
  assert(UserSGPR != AMDGPU::NoRegister);

  SDValue QueuePtr = CreateLiveInRegister(
    DAG, &AMDGPU::SReg_64RegClass, UserSGPR, MVT::i64);

  // Offset into amd_queue_t for group_segment_aperture_base_hi /
  // private_segment_aperture_base_hi.
  uint32_t StructOffset = (AS == AMDGPUAS::LOCAL_ADDRESS) ? 0x40 : 0x44;

  SDValue Ptr = DAG.getNode(ISD::ADD, SL, MVT::i64, QueuePtr,
                            DAG.getConstant(StructOffset, SL, MVT::i64));

  // TODO: Use custom target PseudoSourceValue.
  // TODO: We should use the value from the IR intrinsic call, but it might not
  // be available and how do we get it?
  Value *V = UndefValue::get(PointerType::get(Type::getInt8Ty(*DAG.getContext()),
                                              AMDGPUAS::CONSTANT_ADDRESS));

  MachinePointerInfo PtrInfo(V, StructOffset);
  return DAG.getLoad(MVT::i32, SL, QueuePtr.getValue(1), Ptr, PtrInfo,
                     MinAlign(64, StructOffset),
                     MachineMemOperand::MOInvariant);
}

SDValue SITargetLowering::lowerADDRSPACECAST(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc SL(Op);
  const AddrSpaceCastSDNode *ASC = cast<AddrSpaceCastSDNode>(Op);

  SDValue Src = ASC->getOperand(0);

  // FIXME: Really support non-0 null pointers.
  SDValue SegmentNullPtr = DAG.getConstant(-1, SL, MVT::i32);
  SDValue FlatNullPtr = DAG.getConstant(0, SL, MVT::i64);

  // flat -> local/private
  if (ASC->getSrcAddressSpace() == AMDGPUAS::FLAT_ADDRESS) {
    if (ASC->getDestAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
        ASC->getDestAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS) {
      SDValue NonNull = DAG.getSetCC(SL, MVT::i1, Src, FlatNullPtr, ISD::SETNE);
      SDValue Ptr = DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, Src);

      return DAG.getNode(ISD::SELECT, SL, MVT::i32,
                         NonNull, Ptr, SegmentNullPtr);
    }
  }

  // local/private -> flat
  if (ASC->getDestAddressSpace() == AMDGPUAS::FLAT_ADDRESS) {
    if (ASC->getSrcAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
        ASC->getSrcAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS) {
      SDValue NonNull
        = DAG.getSetCC(SL, MVT::i1, Src, SegmentNullPtr, ISD::SETNE);

      SDValue Aperture = getSegmentAperture(ASC->getSrcAddressSpace(), DAG);
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

static bool shouldEmitGOTReloc(const GlobalValue *GV,
                               const TargetMachine &TM) {
  return GV->getType()->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS &&
         !TM.shouldAssumeDSOLocal(*GV->getParent(), GV);
}

bool
SITargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // We can fold offsets for anything that doesn't require a GOT relocation.
  return GA->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS &&
         !shouldEmitGOTReloc(GA->getGlobal(), getTargetMachine());
}

static SDValue buildPCRelGlobalAddress(SelectionDAG &DAG, const GlobalValue *GV,
                                      SDLoc DL, unsigned Offset, EVT PtrVT,
                                      unsigned GAFlags = SIInstrInfo::MO_NONE) {
  // In order to support pc-relative addressing, the PC_ADD_REL_OFFSET SDNode is
  // lowered to the following code sequence:
  // s_getpc_b64 s[0:1]
  // s_add_u32 s0, s0, $symbol
  // s_addc_u32 s1, s1, 0
  //
  // s_getpc_b64 returns the address of the s_add_u32 instruction and then
  // a fixup or relocation is emitted to replace $symbol with a literal
  // constant, which is a pc-relative offset from the encoding of the $symbol
  // operand to the global variable.
  //
  // What we want here is an offset from the value returned by s_getpc
  // (which is the address of the s_add_u32 instruction) to the global
  // variable, but since the encoding of $symbol starts 4 bytes after the start
  // of the s_add_u32 instruction, we end up with an offset that is 4 bytes too
  // small. This requires us to add 4 to the global variable offset in order to
  // compute the correct address.
  SDValue GA = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset + 4,
                                          GAFlags);
  return DAG.getNode(AMDGPUISD::PC_ADD_REL_OFFSET, DL, PtrVT, GA);
}

SDValue SITargetLowering::LowerGlobalAddress(AMDGPUMachineFunction *MFI,
                                             SDValue Op,
                                             SelectionDAG &DAG) const {
  GlobalAddressSDNode *GSD = cast<GlobalAddressSDNode>(Op);

  if (GSD->getAddressSpace() != AMDGPUAS::CONSTANT_ADDRESS &&
      GSD->getAddressSpace() != AMDGPUAS::GLOBAL_ADDRESS)
    return AMDGPUTargetLowering::LowerGlobalAddress(MFI, Op, DAG);

  SDLoc DL(GSD);
  const GlobalValue *GV = GSD->getGlobal();
  EVT PtrVT = Op.getValueType();

  if (!shouldEmitGOTReloc(GV, getTargetMachine()))
    return buildPCRelGlobalAddress(DAG, GV, DL, GSD->getOffset(), PtrVT);

  SDValue GOTAddr = buildPCRelGlobalAddress(DAG, GV, DL, 0, PtrVT,
                                            SIInstrInfo::MO_GOTPCREL);

  Type *Ty = PtrVT.getTypeForEVT(*DAG.getContext());
  PointerType *PtrTy = PointerType::get(Ty, AMDGPUAS::CONSTANT_ADDRESS);
  const DataLayout &DataLayout = DAG.getDataLayout();
  unsigned Align = DataLayout.getABITypeAlignment(PtrTy);
  // FIXME: Use a PseudoSourceValue once those can be assigned an address space.
  MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));

  return DAG.getLoad(PtrVT, DL, DAG.getEntryNode(), GOTAddr, PtrInfo, Align,
                     MachineMemOperand::MOInvariant);
}

SDValue SITargetLowering::lowerTRAP(SDValue Op,
                                    SelectionDAG &DAG) const {
  const MachineFunction &MF = DAG.getMachineFunction();
  DiagnosticInfoUnsupported NoTrap(*MF.getFunction(),
                                   "trap handler not supported",
                                   Op.getDebugLoc(),
                                   DS_Warning);
  DAG.getContext()->diagnose(NoTrap);

  // Emit s_endpgm.

  // FIXME: This should really be selected to s_trap, but that requires
  // setting up the trap handler for it o do anything.
  return DAG.getNode(AMDGPUISD::ENDPGM, SDLoc(Op), MVT::Other,
                     Op.getOperand(0));
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
  SDValue Param = LowerParameter(DAG, MVT::i32, MVT::i32, SL,
                                 DAG.getEntryNode(), Offset, false);
  // The local size values will have the hi 16-bits as zero.
  return DAG.getNode(ISD::AssertZext, SL, MVT::i32, Param,
                     DAG.getValueType(VT));
}

static SDValue emitNonHSAIntrinsicError(SelectionDAG& DAG, SDLoc DL, EVT VT) {
  DiagnosticInfoUnsupported BadIntrin(*DAG.getMachineFunction().getFunction(),
                                      "non-hsa intrinsic with hsa target",
                                      DL.getDebugLoc());
  DAG.getContext()->diagnose(BadIntrin);
  return DAG.getUNDEF(VT);
}

static SDValue emitRemovedIntrinsicError(SelectionDAG& DAG, SDLoc DL, EVT VT) {
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
  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();

  // TODO: Should this propagate fast-math-flags?

  switch (IntrinsicID) {
  case Intrinsic::amdgcn_dispatch_ptr:
  case Intrinsic::amdgcn_queue_ptr: {
    if (!Subtarget->isAmdHsaOS()) {
      DiagnosticInfoUnsupported BadIntrin(
          *MF.getFunction(), "unsupported hsa intrinsic without hsa target",
          DL.getDebugLoc());
      DAG.getContext()->diagnose(BadIntrin);
      return DAG.getUNDEF(VT);
    }

    auto Reg = IntrinsicID == Intrinsic::amdgcn_dispatch_ptr ?
      SIRegisterInfo::DISPATCH_PTR : SIRegisterInfo::QUEUE_PTR;
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_64RegClass,
                                TRI->getPreloadedValue(MF, Reg), VT);
  }
  case Intrinsic::amdgcn_implicitarg_ptr: {
    unsigned offset = getImplicitParameterOffset(MFI, FIRST_IMPLICIT);
    return LowerParameterPtr(DAG, DL, DAG.getEntryNode(), offset);
  }
  case Intrinsic::amdgcn_kernarg_segment_ptr: {
    unsigned Reg
      = TRI->getPreloadedValue(MF, SIRegisterInfo::KERNARG_SEGMENT_PTR);
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_64RegClass, Reg, VT);
  }
  case Intrinsic::amdgcn_dispatch_id: {
    unsigned Reg = TRI->getPreloadedValue(MF, SIRegisterInfo::DISPATCH_ID);
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_64RegClass, Reg, VT);
  }
  case Intrinsic::amdgcn_rcp:
    return DAG.getNode(AMDGPUISD::RCP, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rsq:
  case AMDGPUIntrinsic::AMDGPU_rsq: // Legacy name
    return DAG.getNode(AMDGPUISD::RSQ, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rsq_legacy: {
    if (Subtarget->getGeneration() >= SISubtarget::VOLCANIC_ISLANDS)
      return emitRemovedIntrinsicError(DAG, DL, VT);

    return DAG.getNode(AMDGPUISD::RSQ_LEGACY, DL, VT, Op.getOperand(1));
  }
  case Intrinsic::amdgcn_rcp_legacy: {
    if (Subtarget->getGeneration() >= SISubtarget::VOLCANIC_ISLANDS)
      return emitRemovedIntrinsicError(DAG, DL, VT);
    return DAG.getNode(AMDGPUISD::RCP_LEGACY, DL, VT, Op.getOperand(1));
  }
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

    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::NGROUPS_X, false);
  case Intrinsic::r600_read_ngroups_y:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::NGROUPS_Y, false);
  case Intrinsic::r600_read_ngroups_z:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::NGROUPS_Z, false);
  case Intrinsic::r600_read_global_size_x:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::GLOBAL_SIZE_X, false);
  case Intrinsic::r600_read_global_size_y:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::GLOBAL_SIZE_Y, false);
  case Intrinsic::r600_read_global_size_z:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
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
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::WORKGROUP_ID_X), VT);
  case Intrinsic::amdgcn_workgroup_id_y:
  case Intrinsic::r600_read_tgid_y:
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::WORKGROUP_ID_Y), VT);
  case Intrinsic::amdgcn_workgroup_id_z:
  case Intrinsic::r600_read_tgid_z:
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::WORKGROUP_ID_Z), VT);
  case Intrinsic::amdgcn_workitem_id_x:
  case Intrinsic::r600_read_tidig_x:
    return CreateLiveInRegister(DAG, &AMDGPU::VGPR_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::WORKITEM_ID_X), VT);
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::r600_read_tidig_y:
    return CreateLiveInRegister(DAG, &AMDGPU::VGPR_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::WORKITEM_ID_Y), VT);
  case Intrinsic::amdgcn_workitem_id_z:
  case Intrinsic::r600_read_tidig_z:
    return CreateLiveInRegister(DAG, &AMDGPU::VGPR_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::WORKITEM_ID_Z), VT);
  case AMDGPUIntrinsic::SI_load_const: {
    SDValue Ops[] = {
      Op.getOperand(1),
      Op.getOperand(2)
    };

    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant,
      VT.getStoreSize(), 4);
    return DAG.getMemIntrinsicNode(AMDGPUISD::LOAD_CONSTANT, DL,
                                   Op->getVTList(), Ops, VT, MMO);
  }
  case AMDGPUIntrinsic::amdgcn_fdiv_fast: {
    return lowerFDIV_FAST(Op, DAG);
  }
  case AMDGPUIntrinsic::SI_vs_load_input:
    return DAG.getNode(AMDGPUISD::LOAD_INPUT, DL, VT,
                       Op.getOperand(1),
                       Op.getOperand(2),
                       Op.getOperand(3));

  case AMDGPUIntrinsic::SI_fs_constant: {
    SDValue M0 = copyToM0(DAG, DAG.getEntryNode(), DL, Op.getOperand(3));
    SDValue Glue = M0.getValue(1);
    return DAG.getNode(AMDGPUISD::INTERP_MOV, DL, MVT::f32,
                       DAG.getConstant(2, DL, MVT::i32), // P0
                       Op.getOperand(1), Op.getOperand(2), Glue);
  }
  case AMDGPUIntrinsic::SI_packf16:
    if (Op.getOperand(1).isUndef() && Op.getOperand(2).isUndef())
      return DAG.getUNDEF(MVT::i32);
    return Op;
  case AMDGPUIntrinsic::SI_fs_interp: {
    SDValue IJ = Op.getOperand(4);
    SDValue I = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, IJ,
                            DAG.getConstant(0, DL, MVT::i32));
    SDValue J = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, IJ,
                            DAG.getConstant(1, DL, MVT::i32));
    SDValue M0 = copyToM0(DAG, DAG.getEntryNode(), DL, Op.getOperand(3));
    SDValue Glue = M0.getValue(1);
    SDValue P1 = DAG.getNode(AMDGPUISD::INTERP_P1, DL,
                             DAG.getVTList(MVT::f32, MVT::Glue),
                             I, Op.getOperand(1), Op.getOperand(2), Glue);
    Glue = SDValue(P1.getNode(), 1);
    return DAG.getNode(AMDGPUISD::INTERP_P2, DL, MVT::f32, P1, J,
                             Op.getOperand(1), Op.getOperand(2), Glue);
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
      return DAG.getUNDEF(VT);

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
    int CondCode = CD->getSExtValue();

    if (CondCode < ICmpInst::Predicate::FIRST_ICMP_PREDICATE ||
	       CondCode >= ICmpInst::Predicate::BAD_ICMP_PREDICATE)
      return DAG.getUNDEF(VT);

    ICmpInst::Predicate IcInput =
	   static_cast<ICmpInst::Predicate>(CondCode);
    ISD::CondCode CCOpcode = getICmpCondCode(IcInput);
    return DAG.getNode(AMDGPUISD::SETCC, DL, VT, Op.getOperand(1),
                       Op.getOperand(2), DAG.getCondCode(CCOpcode));
  }
  case Intrinsic::amdgcn_fcmp: {
    const auto *CD = dyn_cast<ConstantSDNode>(Op.getOperand(3));
    int CondCode = CD->getSExtValue();

    if (CondCode <= FCmpInst::Predicate::FCMP_FALSE ||
	       CondCode >= FCmpInst::Predicate::FCMP_TRUE)
      return DAG.getUNDEF(VT);

    FCmpInst::Predicate IcInput =
	   static_cast<FCmpInst::Predicate>(CondCode);
    ISD::CondCode CCOpcode = getFCmpCondCode(IcInput);
    return DAG.getNode(AMDGPUISD::SETCC, DL, VT, Op.getOperand(1),
                       Op.getOperand(2), DAG.getCondCode(CCOpcode));
  }
  case Intrinsic::amdgcn_fmul_legacy:
    return DAG.getNode(AMDGPUISD::FMUL_LEGACY, DL, VT,
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::amdgcn_sffbh:
  case AMDGPUIntrinsic::AMDGPU_flbit_i32: // Legacy name.
    return DAG.getNode(AMDGPUISD::FFBH_I32, DL, VT, Op.getOperand(1));
  default:
    return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  }
}

SDValue SITargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op,
                                                 SelectionDAG &DAG) const {
  unsigned IntrID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
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
  default:
    return SDValue();
  }
}

SDValue SITargetLowering::LowerINTRINSIC_VOID(SDValue Op,
                                              SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();

  switch (IntrinsicID) {
  case AMDGPUIntrinsic::SI_sendmsg: {
    Chain = copyToM0(DAG, Chain, DL, Op.getOperand(3));
    SDValue Glue = Chain.getValue(1);
    return DAG.getNode(AMDGPUISD::SENDMSG, DL, MVT::Other, Chain,
                       Op.getOperand(2), Glue);
  }
  case AMDGPUIntrinsic::SI_tbuffer_store: {
    SDValue Ops[] = {
      Chain,
      Op.getOperand(2),
      Op.getOperand(3),
      Op.getOperand(4),
      Op.getOperand(5),
      Op.getOperand(6),
      Op.getOperand(7),
      Op.getOperand(8),
      Op.getOperand(9),
      Op.getOperand(10),
      Op.getOperand(11),
      Op.getOperand(12),
      Op.getOperand(13),
      Op.getOperand(14)
    };

    EVT VT = Op.getOperand(3).getValueType();

    MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOStore,
      VT.getStoreSize(), 4);
    return DAG.getMemIntrinsicNode(AMDGPUISD::TBUFFER_STORE_FORMAT, DL,
                                   Op->getVTList(), Ops, VT, MMO);
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
  default:
    return SDValue();
  }
}

SDValue SITargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  LoadSDNode *Load = cast<LoadSDNode>(Op);
  ISD::LoadExtType ExtType = Load->getExtensionType();
  EVT MemVT = Load->getMemoryVT();

  if (ExtType == ISD::NON_EXTLOAD && MemVT.getSizeInBits() < 32) {
    assert(MemVT == MVT::i1 && "Only i1 non-extloads expected");
    // FIXME: Copied from PPC
    // First, load into 32 bits, then truncate to 1 bit.

    SDValue Chain = Load->getChain();
    SDValue BasePtr = Load->getBasePtr();
    MachineMemOperand *MMO = Load->getMemOperand();

    SDValue NewLD = DAG.getExtLoad(ISD::EXTLOAD, DL, MVT::i32, Chain,
                                   BasePtr, MVT::i8, MMO);

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

  unsigned NumElements = MemVT.getVectorNumElements();
  switch (AS) {
  case AMDGPUAS::CONSTANT_ADDRESS:
    if (isMemOpUniform(Load))
      return SDValue();
    // Non-uniform loads will be selected to MUBUF instructions, so they
    // have the same legalization requires ments as global and private
    // loads.
    //
    LLVM_FALLTHROUGH;
  case AMDGPUAS::GLOBAL_ADDRESS:
  case AMDGPUAS::FLAT_ADDRESS:
    if (NumElements > 4)
      return SplitVectorLoad(Op, DAG);
    // v4 loads are supported for private and global memory.
    return SDValue();
  case AMDGPUAS::PRIVATE_ADDRESS: {
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
  }
  case AMDGPUAS::LOCAL_ADDRESS: {
    if (NumElements > 2)
      return SplitVectorLoad(Op, DAG);

    if (NumElements == 2)
      return SDValue();

    // If properly aligned, if we split we might be able to use ds_read_b64.
    return SplitVectorLoad(Op, DAG);
  }
  default:
    return SDValue();
  }
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
  bool Unsafe = DAG.getTarget().Options.UnsafeFPMath;

  if (const ConstantFPSDNode *CLHS = dyn_cast<ConstantFPSDNode>(LHS)) {
    if ((Unsafe || (VT == MVT::f32 && !Subtarget->hasFP32Denormals()))) {

      if (CLHS->isExactlyValue(1.0)) {
        // v_rcp_f32 and v_rsq_f32 do not support denormals, and according to
        // the CI documentation has a worst case error of 1 ulp.
        // OpenCL requires <= 2.5 ulp for 1.0 / x, so it should always be OK to
        // use it as long as we aren't trying to use denormals.

        // 1.0 / sqrt(x) -> rsq(x)
        //
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

  const SDNodeFlags *Flags = Op->getFlags();

  if (Unsafe || Flags->hasAllowReciprocal()) {
    // Turn into multiply by the reciprocal.
    // x / y -> x * (1.0 / y)
    SDNodeFlags Flags;
    Flags.setUnsafeAlgebra(true);
    SDValue Recip = DAG.getNode(AMDGPUISD::RCP, SL, VT, RHS);
    return DAG.getNode(ISD::FMUL, SL, VT, LHS, Recip, &Flags);
  }

  return SDValue();
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

  SDValue DenominatorScaled = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT, RHS, RHS, LHS);
  SDValue NumeratorScaled = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT, LHS, RHS, LHS);

  // Denominator is scaled to not be denormal, so using rcp is ok.
  SDValue ApproxRcp = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f32, DenominatorScaled);

  SDValue NegDivScale0 = DAG.getNode(ISD::FNEG, SL, MVT::f32, DenominatorScaled);

  SDValue Fma0 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDivScale0, ApproxRcp, One);
  SDValue Fma1 = DAG.getNode(ISD::FMA, SL, MVT::f32, Fma0, ApproxRcp, ApproxRcp);

  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f32, NumeratorScaled, Fma1);

  SDValue Fma2 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDivScale0, Mul, NumeratorScaled);
  SDValue Fma3 = DAG.getNode(ISD::FMA, SL, MVT::f32, Fma2, Fma1, Mul);
  SDValue Fma4 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDivScale0, Fma3, NumeratorScaled);

  SDValue Scale = NumeratorScaled.getValue(1);
  SDValue Fmas = DAG.getNode(AMDGPUISD::DIV_FMAS, SL, MVT::f32, Fma4, Fma1, Fma3, Scale);

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

  unsigned NumElements = VT.getVectorNumElements();
  switch (AS) {
  case AMDGPUAS::GLOBAL_ADDRESS:
  case AMDGPUAS::FLAT_ADDRESS:
    if (NumElements > 4)
      return SplitVectorStore(Op, DAG);
    return SDValue();
  case AMDGPUAS::PRIVATE_ADDRESS: {
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
  }
  case AMDGPUAS::LOCAL_ADDRESS: {
    if (NumElements > 2)
      return SplitVectorStore(Op, DAG);

    if (NumElements == 2)
      return Op;

    // If properly aligned, if we split we might be able to use ds_write_b64.
    return SplitVectorStore(Op, DAG);
  }
  default:
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
  if (!isFlatGlobalAddrSpace(AS))
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

/// \brief Return true if the given offset Size in bytes can be folded into
/// the immediate offsets of a memory instruction for the given address space.
static bool canFoldOffset(unsigned OffsetSize, unsigned AS,
                          const SISubtarget &STI) {
  switch (AS) {
  case AMDGPUAS::GLOBAL_ADDRESS: {
    // MUBUF instructions a 12-bit offset in bytes.
    return isUInt<12>(OffsetSize);
  }
  case AMDGPUAS::CONSTANT_ADDRESS: {
    // SMRD instructions have an 8-bit offset in dwords on SI and
    // a 20-bit offset in bytes on VI.
    if (STI.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS)
      return isUInt<20>(OffsetSize);
    else
      return (OffsetSize % 4 == 0) && isUInt<8>(OffsetSize / 4);
  }
  case AMDGPUAS::LOCAL_ADDRESS:
  case AMDGPUAS::REGION_ADDRESS: {
    // The single offset versions have a 16-bit offset in bytes.
    return isUInt<16>(OffsetSize);
  }
  case AMDGPUAS::PRIVATE_ADDRESS:
  // Indirect register addressing does not use any offsets.
  default:
    return 0;
  }
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
                                               DAGCombinerInfo &DCI) const {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  if (N0.getOpcode() != ISD::ADD)
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
  if (!canFoldOffset(Offset.getZExtValue(), AddrSpace, *getSubtarget()))
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  SDValue ShlX = DAG.getNode(ISD::SHL, SL, VT, N0.getOperand(0), N1);
  SDValue COffset = DAG.getConstant(Offset, SL, MVT::i32);

  return DAG.getNode(ISD::ADD, SL, VT, ShlX, COffset);
}

SDValue SITargetLowering::performAndCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalize())
    return SDValue();

  if (SDValue Base = AMDGPUTargetLowering::performAndCombine(N, DCI))
    return Base;

  SelectionDAG &DAG = DCI.DAG;

  // (and (fcmp ord x, x), (fcmp une (fabs x), inf)) ->
  // fp_class x, ~(s_nan | q_nan | n_infinity | p_infinity)
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  if (LHS.getOpcode() == ISD::SETCC &&
      RHS.getOpcode() == ISD::SETCC) {
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

  return SDValue();
}

SDValue SITargetLowering::performOrCombine(SDNode *N,
                                           DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  EVT VT = N->getValueType(0);
  if (VT == MVT::i64) {
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
  }

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

// Constant fold canonicalize.
SDValue SITargetLowering::performFCanonicalizeCombine(
  SDNode *N,
  DAGCombinerInfo &DCI) const {
  ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  if (!CFP)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  const APFloat &C = CFP->getValueAPF();

  // Flush denormals to 0 if not enabled.
  if (C.isDenormal()) {
    EVT VT = N->getValueType(0);
    if (VT == MVT::f32 && !Subtarget->hasFP32Denormals())
      return DAG.getConstantFP(0.0, SDLoc(N), VT);

    if (VT == MVT::f64 && !Subtarget->hasFP64Denormals())
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

  return SDValue(CFP, 0);
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

static SDValue performIntMed3ImmCombine(SelectionDAG &DAG, const SDLoc &SL,
                                        SDValue Op0, SDValue Op1, bool Signed) {
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
  return DAG.getNode(Signed ? AMDGPUISD::SMED3 : AMDGPUISD::UMED3, SL, VT,
                     Op0.getOperand(0), SDValue(K0, 0), SDValue(K1, 0));
}

static bool isKnownNeverSNan(SelectionDAG &DAG, SDValue Op) {
  if (!DAG.getTargetLoweringInfo().hasFloatingPointExceptions())
    return true;

  return DAG.isKnownNeverNaN(Op);
}

static SDValue performFPMed3ImmCombine(SelectionDAG &DAG, const SDLoc &SL,
                                       SDValue Op0, SDValue Op1) {
  ConstantFPSDNode *K1 = dyn_cast<ConstantFPSDNode>(Op1);
  if (!K1)
    return SDValue();

  ConstantFPSDNode *K0 = dyn_cast<ConstantFPSDNode>(Op0.getOperand(1));
  if (!K0)
    return SDValue();

  // Ordered >= (although NaN inputs should have folded away by now).
  APFloat::cmpResult Cmp = K0->getValueAPF().compare(K1->getValueAPF());
  if (Cmp == APFloat::cmpGreaterThan)
    return SDValue();

  // This isn't safe with signaling NaNs because in IEEE mode, min/max on a
  // signaling NaN gives a quiet NaN. The quiet NaN input to the min would then
  // give the other result, which is different from med3 with a NaN input.
  SDValue Var = Op0.getOperand(0);
  if (!isKnownNeverSNan(DAG, Var))
    return SDValue();

  return DAG.getNode(AMDGPUISD::FMED3, SL, K0->getValueType(0),
                     Var, SDValue(K0, 0), SDValue(K1, 0));
}

SDValue SITargetLowering::performMinMaxCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  unsigned Opc = N->getOpcode();
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  // Only do this if the inner op has one use since this will just increases
  // register pressure for no benefit.

  if (Opc != AMDGPUISD::FMIN_LEGACY && Opc != AMDGPUISD::FMAX_LEGACY) {
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
      N->getValueType(0) == MVT::f32 && Op0.hasOneUse()) {
    if (SDValue Res = performFPMed3ImmCombine(DAG, SDLoc(N), Op0, Op1))
      return Res;
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

  if (VT != MVT::f32 && VT != MVT::f64)
    return SDValue();

  // Match isinf pattern
  // (fcmp oeq (fabs x), inf) -> (fp_class x, (p_infinity | n_infinity))
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
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

SDValue SITargetLowering::PerformDAGCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  switch (N->getOpcode()) {
  default:
    return AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
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
        N->getValueType(0) != MVT::f64 &&
        getTargetMachine().getOptLevel() > CodeGenOpt::None)
      return performMinMaxCombine(N, DCI);
    break;
  }

  case AMDGPUISD::CVT_F32_UBYTE0:
  case AMDGPUISD::CVT_F32_UBYTE1:
  case AMDGPUISD::CVT_F32_UBYTE2:
  case AMDGPUISD::CVT_F32_UBYTE3: {
    unsigned Offset = N->getOpcode() - AMDGPUISD::CVT_F32_UBYTE0;
    SDValue Src = N->getOperand(0);

    // TODO: Handle (or x, (srl y, 8)) pattern when known bits are zero.
    if (Src.getOpcode() == ISD::SRL) {
      // cvt_f32_ubyte0 (srl x, 16) -> cvt_f32_ubyte2 x
      // cvt_f32_ubyte1 (srl x, 16) -> cvt_f32_ubyte3 x
      // cvt_f32_ubyte0 (srl x, 8) -> cvt_f32_ubyte1 x

      if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Src.getOperand(1))) {
        unsigned SrcOffset = C->getZExtValue() + 8 * Offset;
        if (SrcOffset < 32 && SrcOffset % 8 == 0) {
          return DAG.getNode(AMDGPUISD::CVT_F32_UBYTE0 + SrcOffset / 8, DL,
                             MVT::f32, Src.getOperand(0));
        }
      }
    }

    APInt Demanded = APInt::getBitsSet(32, 8 * Offset, 8 * Offset + 8);

    APInt KnownZero, KnownOne;
    TargetLowering::TargetLoweringOpt TLO(DAG, !DCI.isBeforeLegalize(),
                                          !DCI.isBeforeLegalizeOps());
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    if (TLO.ShrinkDemandedConstant(Src, Demanded) ||
        TLI.SimplifyDemandedBits(Src, Demanded, KnownZero, KnownOne, TLO)) {
      DCI.CommitTargetLoweringOpt(TLO);
    }

    break;
  }

  case ISD::UINT_TO_FP: {
    return performUCharToFloatCombine(N, DCI);
  }
  case ISD::FADD: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;

    EVT VT = N->getValueType(0);
    if (VT != MVT::f32)
      break;

    // Only do this if we are not trying to support denormals. v_mad_f32 does
    // not support denormals ever.
    if (Subtarget->hasFP32Denormals())
      break;

    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);

    // These should really be instruction patterns, but writing patterns with
    // source modiifiers is a pain.

    // fadd (fadd (a, a), b) -> mad 2.0, a, b
    if (LHS.getOpcode() == ISD::FADD) {
      SDValue A = LHS.getOperand(0);
      if (A == LHS.getOperand(1)) {
        const SDValue Two = DAG.getConstantFP(2.0, DL, MVT::f32);
        return DAG.getNode(ISD::FMAD, DL, VT, Two, A, RHS);
      }
    }

    // fadd (b, fadd (a, a)) -> mad 2.0, a, b
    if (RHS.getOpcode() == ISD::FADD) {
      SDValue A = RHS.getOperand(0);
      if (A == RHS.getOperand(1)) {
        const SDValue Two = DAG.getConstantFP(2.0, DL, MVT::f32);
        return DAG.getNode(ISD::FMAD, DL, VT, Two, A, LHS);
      }
    }

    return SDValue();
  }
  case ISD::FSUB: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;

    EVT VT = N->getValueType(0);

    // Try to get the fneg to fold into the source modifier. This undoes generic
    // DAG combines and folds them into the mad.
    //
    // Only do this if we are not trying to support denormals. v_mad_f32 does
    // not support denormals ever.
    if (VT == MVT::f32 &&
        !Subtarget->hasFP32Denormals()) {
      SDValue LHS = N->getOperand(0);
      SDValue RHS = N->getOperand(1);
      if (LHS.getOpcode() == ISD::FADD) {
        // (fsub (fadd a, a), c) -> mad 2.0, a, (fneg c)

        SDValue A = LHS.getOperand(0);
        if (A == LHS.getOperand(1)) {
          const SDValue Two = DAG.getConstantFP(2.0, DL, MVT::f32);
          SDValue NegRHS = DAG.getNode(ISD::FNEG, DL, VT, RHS);

          return DAG.getNode(ISD::FMAD, DL, VT, Two, A, NegRHS);
        }
      }

      if (RHS.getOpcode() == ISD::FADD) {
        // (fsub c, (fadd a, a)) -> mad -2.0, a, c

        SDValue A = RHS.getOperand(0);
        if (A == RHS.getOperand(1)) {
          const SDValue NegTwo = DAG.getConstantFP(-2.0, DL, MVT::f32);
          return DAG.getNode(ISD::FMAD, DL, VT, NegTwo, A, LHS);
        }
      }

      return SDValue();
    }

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
  case AMDGPUISD::ATOMIC_DEC: { // TODO: Target mem intrinsics.
    if (DCI.isBeforeLegalize())
      break;

    MemSDNode *MemNode = cast<MemSDNode>(N);
    SDValue Ptr = MemNode->getBasePtr();

    // TODO: We could also do this for multiplies.
    unsigned AS = MemNode->getAddressSpace();
    if (Ptr.getOpcode() == ISD::SHL && AS != AMDGPUAS::PRIVATE_ADDRESS) {
      SDValue NewPtr = performSHLPtrCombine(Ptr.getNode(), AS, DCI);
      if (NewPtr) {
        SmallVector<SDValue, 8> NewOps(MemNode->op_begin(), MemNode->op_end());

        NewOps[N->getOpcode() == ISD::STORE ? 2 : 1] = NewPtr;
        return SDValue(DAG.UpdateNodeOperands(MemNode, NewOps), 0);
      }
    }
    break;
  }
  case ISD::AND:
    return performAndCombine(N, DCI);
  case ISD::OR:
    return performOrCombine(N, DCI);
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

    // Abort if we can't understand the usage
    if (!I->isMachineOpcode() ||
        I->getMachineOpcode() != TargetOpcode::EXTRACT_SUBREG)
      return;

    // Lane means which subreg of %VGPRa_VGPRb_VGPRc_VGPRd is used.
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
void SITargetLowering::legalizeTargetIndependentNode(SDNode *Node,
                                                     SelectionDAG &DAG) const {

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

  DAG.UpdateNodeOperands(Node, Ops);
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
    unsigned DmaskIdx = MI.getNumOperands() == 12 ? 3 : 4;
    unsigned Writemask = MI.getOperand(DmaskIdx).getImm();
    unsigned BitsSet = 0;
    for (unsigned i = 0; i < 4; ++i)
      BitsSet += Writemask & (1 << i) ? 1 : 0;

    const TargetRegisterClass *RC;
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

SDValue SITargetLowering::CreateLiveInRegister(SelectionDAG &DAG,
                                               const TargetRegisterClass *RC,
                                               unsigned Reg, EVT VT) const {
  SDValue VReg = AMDGPUTargetLowering::CreateLiveInRegister(DAG, RC, Reg, VT);

  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(DAG.getEntryNode()),
                            cast<RegisterSDNode>(VReg)->getReg(), VT);
}

//===----------------------------------------------------------------------===//
//                         SI Inline Assembly Support
//===----------------------------------------------------------------------===//

std::pair<unsigned, const TargetRegisterClass *>
SITargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                               StringRef Constraint,
                                               MVT VT) const {

  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 's':
    case 'r':
      switch (VT.getSizeInBits()) {
      default:
        return std::make_pair(0U, nullptr);
      case 32:
        return std::make_pair(0U, &AMDGPU::SGPR_32RegClass);
      case 64:
        return std::make_pair(0U, &AMDGPU::SGPR_64RegClass);
      case 128:
        return std::make_pair(0U, &AMDGPU::SReg_128RegClass);
      case 256:
        return std::make_pair(0U, &AMDGPU::SReg_256RegClass);
      }

    case 'v':
      switch (VT.getSizeInBits()) {
      default:
        return std::make_pair(0U, nullptr);
      case 32:
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
