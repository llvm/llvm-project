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

#include "SIISelLowering.h"
#include "AMDGPU.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Function.h"
#include "llvm/ADT/SmallString.h"

using namespace llvm;

SITargetLowering::SITargetLowering(TargetMachine &TM) :
    AMDGPUTargetLowering(TM) {
  addRegisterClass(MVT::i1, &AMDGPU::VReg_1RegClass);
  addRegisterClass(MVT::i64, &AMDGPU::SReg_64RegClass);

  addRegisterClass(MVT::v32i8, &AMDGPU::SReg_256RegClass);
  addRegisterClass(MVT::v64i8, &AMDGPU::SReg_512RegClass);

  addRegisterClass(MVT::i32, &AMDGPU::SReg_32RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::VGPR_32RegClass);

  addRegisterClass(MVT::f64, &AMDGPU::VReg_64RegClass);
  addRegisterClass(MVT::v2i32, &AMDGPU::SReg_64RegClass);
  addRegisterClass(MVT::v2f32, &AMDGPU::VReg_64RegClass);

  addRegisterClass(MVT::v4i32, &AMDGPU::SReg_128RegClass);
  addRegisterClass(MVT::v4f32, &AMDGPU::VReg_128RegClass);

  addRegisterClass(MVT::v8i32, &AMDGPU::SReg_256RegClass);
  addRegisterClass(MVT::v8f32, &AMDGPU::VReg_256RegClass);

  addRegisterClass(MVT::v16i32, &AMDGPU::SReg_512RegClass);
  addRegisterClass(MVT::v16f32, &AMDGPU::VReg_512RegClass);

  computeRegisterProperties();

  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8f32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16f32, Expand);

  setOperationAction(ISD::ADD, MVT::i32, Legal);
  setOperationAction(ISD::ADDC, MVT::i32, Legal);
  setOperationAction(ISD::ADDE, MVT::i32, Legal);
  setOperationAction(ISD::SUBC, MVT::i32, Legal);
  setOperationAction(ISD::SUBE, MVT::i32, Legal);

  setOperationAction(ISD::FSIN, MVT::f32, Custom);
  setOperationAction(ISD::FCOS, MVT::f32, Custom);

  setOperationAction(ISD::FMINNUM, MVT::f32, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::f32, Legal);
  setOperationAction(ISD::FMINNUM, MVT::f64, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::f64, Legal);

  // We need to custom lower vector stores from local memory
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v8i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v16i32, Custom);

  setOperationAction(ISD::STORE, MVT::v8i32, Custom);
  setOperationAction(ISD::STORE, MVT::v16i32, Custom);

  setOperationAction(ISD::STORE, MVT::i1, Custom);
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);

  setOperationAction(ISD::SELECT, MVT::i64, Custom);
  setOperationAction(ISD::SELECT, MVT::f64, Promote);
  AddPromotedToType(ISD::SELECT, MVT::f64, MVT::i64);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);

  setOperationAction(ISD::SETCC, MVT::v2i1, Expand);
  setOperationAction(ISD::SETCC, MVT::v4i1, Expand);

  setOperationAction(ISD::BSWAP, MVT::i32, Legal);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i1, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i1, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i8, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i8, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i16, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::Other, Custom);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::f32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v16i8, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v4f32, Custom);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);

  for (MVT VT : MVT::integer_valuetypes()) {
    if (VT == MVT::i64)
      continue;

    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i32, Expand);

    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i32, Expand);

    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i32, Expand);
  }

  for (MVT VT : MVT::integer_vector_valuetypes()) {
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v8i16, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v16i16, Expand);
  }

  for (MVT VT : MVT::fp_valuetypes())
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f32, Expand);

  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::i64, MVT::i32, Expand);
  setTruncStoreAction(MVT::v8i32, MVT::v8i16, Expand);
  setTruncStoreAction(MVT::v16i32, MVT::v16i16, Expand);

  setOperationAction(ISD::LOAD, MVT::i1, Custom);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::FrameIndex, MVT::i32, Custom);

  // These should use UDIVREM, so set them to expand
  setOperationAction(ISD::UDIV, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);

  // We only support LOAD/STORE and vector manipulation ops for vectors
  // with > 4 elements.
  MVT VecTypes[] = {
    MVT::v8i32, MVT::v8f32, MVT::v16i32, MVT::v16f32
  };

  setOperationAction(ISD::SELECT_CC, MVT::i1, Expand);
  setOperationAction(ISD::SELECT, MVT::i1, Promote);

  for (MVT VT : VecTypes) {
    for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op) {
      switch(Op) {
      case ISD::LOAD:
      case ISD::STORE:
      case ISD::BUILD_VECTOR:
      case ISD::BITCAST:
      case ISD::EXTRACT_VECTOR_ELT:
      case ISD::INSERT_VECTOR_ELT:
      case ISD::INSERT_SUBVECTOR:
      case ISD::EXTRACT_SUBVECTOR:
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

  if (Subtarget->getGeneration() >= AMDGPUSubtarget::SEA_ISLANDS) {
    setOperationAction(ISD::FTRUNC, MVT::f64, Legal);
    setOperationAction(ISD::FCEIL, MVT::f64, Legal);
    setOperationAction(ISD::FFLOOR, MVT::f64, Legal);
    setOperationAction(ISD::FRINT, MVT::f64, Legal);
  }

  setOperationAction(ISD::FDIV, MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f64, Custom);

  setTargetDAGCombine(ISD::FADD);
  setTargetDAGCombine(ISD::FSUB);
  setTargetDAGCombine(ISD::FMINNUM);
  setTargetDAGCombine(ISD::FMAXNUM);
  setTargetDAGCombine(ISD::SELECT_CC);
  setTargetDAGCombine(ISD::SETCC);
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::UINT_TO_FP);

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

//===----------------------------------------------------------------------===//
// TargetLowering queries
//===----------------------------------------------------------------------===//

bool SITargetLowering::isShuffleMaskLegal(const SmallVectorImpl<int> &,
                                          EVT) const {
  // SI has some legal vector types, but no legal vector operations. Say no
  // shuffles are legal in order to prefer scalarizing some vector operations.
  return false;
}

// FIXME: This really needs an address space argument. The immediate offset
// size is different for different sets of memory instruction sets.

// The single offset DS instructions have a 16-bit unsigned byte offset.
//
// MUBUF / MTBUF have a 12-bit unsigned byte offset, and additionally can do r +
// r + i with addr64. 32-bit has more addressing mode options. Depending on the
// resource constant, it can also do (i64 r0) + (i32 r1) * (i14 i).
//
// SMRD instructions have an 8-bit, dword offset.
//
bool SITargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                             Type *Ty) const {
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // Allow a 16-bit unsigned immediate field, since this is what DS instructions
  // use.
  if (!isUInt<16>(AM.BaseOffs))
    return false;

  // Only support r+r,
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
  default: // Don't allow n * r
    return false;
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
  if (!VT.isSimple() || VT == MVT::Other)
    return false;

  // TODO - CI+ supports unaligned memory accesses, but this requires driver
  // support.

  // XXX - The only mention I see of this in the ISA manual is for LDS direct
  // reads the "byte address and must be dword aligned". Is it also true for the
  // normal loads and stores?
  if (AddrSpace == AMDGPUAS::LOCAL_ADDRESS) {
    // ds_read/write_b64 require 8-byte alignment, but we can do a 4 byte
    // aligned, 8 byte access in a single operation using ds_read2/write2_b32
    // with adjacent offsets.
    return Align % 4 == 0;
  }

  // Smaller than dword value must be aligned.
  // FIXME: This should be allowed on CI+
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

TargetLoweringBase::LegalizeTypeAction
SITargetLowering::getPreferredVectorAction(EVT VT) const {
  if (VT.getVectorNumElements() != 1 && VT.getScalarType().bitsLE(MVT::i16))
    return TypeSplitVector;

  return TargetLoweringBase::getPreferredVectorAction(VT);
}

bool SITargetLowering::shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                                         Type *Ty) const {
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      getTargetMachine().getSubtargetImpl()->getInstrInfo());
  return TII->isInlineConstant(Imm);
}

SDValue SITargetLowering::LowerParameter(SelectionDAG &DAG, EVT VT, EVT MemVT,
                                         SDLoc SL, SDValue Chain,
                                         unsigned Offset, bool Signed) const {
  const DataLayout *DL = getDataLayout();
  MachineFunction &MF = DAG.getMachineFunction();
  const SIRegisterInfo *TRI =
      static_cast<const SIRegisterInfo*>(Subtarget->getRegisterInfo());
  unsigned InputPtrReg = TRI->getPreloadedValue(MF, SIRegisterInfo::INPUT_PTR);

  Type *Ty = VT.getTypeForEVT(*DAG.getContext());

  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
  PointerType *PtrTy = PointerType::get(Ty, AMDGPUAS::CONSTANT_ADDRESS);
  SDValue BasePtr =  DAG.getCopyFromReg(Chain, SL,
                           MRI.getLiveInVirtReg(InputPtrReg), MVT::i64);
  SDValue Ptr = DAG.getNode(ISD::ADD, SL, MVT::i64, BasePtr,
                                             DAG.getConstant(Offset, MVT::i64));
  SDValue PtrOffset = DAG.getUNDEF(getPointerTy(AMDGPUAS::CONSTANT_ADDRESS));
  MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));

  return DAG.getLoad(ISD::UNINDEXED, Signed ? ISD::SEXTLOAD : ISD::ZEXTLOAD,
                     VT, SL, Chain, Ptr, PtrOffset, PtrInfo, MemVT,
                     false, // isVolatile
                     true, // isNonTemporal
                     true, // isInvariant
                     DL->getABITypeAlignment(Ty)); // Alignment
}

SDValue SITargetLowering::LowerFormalArguments(
                                      SDValue Chain,
                                      CallingConv::ID CallConv,
                                      bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SDLoc DL, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {

  const TargetMachine &TM = getTargetMachine();
  const SIRegisterInfo *TRI =
      static_cast<const SIRegisterInfo*>(TM.getSubtargetImpl()->getRegisterInfo());

  MachineFunction &MF = DAG.getMachineFunction();
  FunctionType *FType = MF.getFunction()->getFunctionType();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  assert(CallConv == CallingConv::C);

  SmallVector<ISD::InputArg, 16> Splits;
  BitVector Skipped(Ins.size());

  for (unsigned i = 0, e = Ins.size(), PSInputNum = 0; i != e; ++i) {
    const ISD::InputArg &Arg = Ins[i];

    // First check if it's a PS input addr
    if (Info->getShaderType() == ShaderType::PIXEL && !Arg.Flags.isInReg() &&
        !Arg.Flags.isByVal()) {

      assert((PSInputNum <= 15) && "Too many PS inputs!");

      if (!Arg.Used) {
        // We can savely skip PS inputs
        Skipped.set(i);
        ++PSInputNum;
        continue;
      }

      Info->PSInputAddr |= 1 << PSInputNum++;
    }

    // Second split vertices into their elements
    if (Info->getShaderType() != ShaderType::COMPUTE && Arg.VT.isVector()) {
      ISD::InputArg NewArg = Arg;
      NewArg.Flags.setSplit();
      NewArg.VT = Arg.VT.getVectorElementType();

      // We REALLY want the ORIGINAL number of vertex elements here, e.g. a
      // three or five element vertex only needs three or five registers,
      // NOT four or eigth.
      Type *ParamType = FType->getParamType(Arg.getOrigArgIndex());
      unsigned NumElements = ParamType->getVectorNumElements();

      for (unsigned j = 0; j != NumElements; ++j) {
        Splits.push_back(NewArg);
        NewArg.PartOffset += NewArg.VT.getStoreSize();
      }

    } else if (Info->getShaderType() != ShaderType::COMPUTE) {
      Splits.push_back(Arg);
    }
  }

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  // At least one interpolation mode must be enabled or else the GPU will hang.
  if (Info->getShaderType() == ShaderType::PIXEL &&
      (Info->PSInputAddr & 0x7F) == 0) {
    Info->PSInputAddr |= 1;
    CCInfo.AllocateReg(AMDGPU::VGPR0);
    CCInfo.AllocateReg(AMDGPU::VGPR1);
  }

  // The pointer to the list of arguments is stored in SGPR0, SGPR1
	// The pointer to the scratch buffer is stored in SGPR2, SGPR3
  if (Info->getShaderType() == ShaderType::COMPUTE) {
    if (Subtarget->isAmdHsaOS())
      Info->NumUserSGPRs = 2;  // FIXME: Need to support scratch buffers.
    else
      Info->NumUserSGPRs = 4;

    unsigned InputPtrReg =
        TRI->getPreloadedValue(MF, SIRegisterInfo::INPUT_PTR);
    unsigned InputPtrRegLo =
        TRI->getPhysRegSubReg(InputPtrReg, &AMDGPU::SReg_32RegClass, 0);
    unsigned InputPtrRegHi =
        TRI->getPhysRegSubReg(InputPtrReg, &AMDGPU::SReg_32RegClass, 1);

    unsigned ScratchPtrReg =
        TRI->getPreloadedValue(MF, SIRegisterInfo::SCRATCH_PTR);
    unsigned ScratchPtrRegLo =
        TRI->getPhysRegSubReg(ScratchPtrReg, &AMDGPU::SReg_32RegClass, 0);
    unsigned ScratchPtrRegHi =
        TRI->getPhysRegSubReg(ScratchPtrReg, &AMDGPU::SReg_32RegClass, 1);

    CCInfo.AllocateReg(InputPtrRegLo);
    CCInfo.AllocateReg(InputPtrRegHi);
    CCInfo.AllocateReg(ScratchPtrRegLo);
    CCInfo.AllocateReg(ScratchPtrRegHi);
    MF.addLiveIn(InputPtrReg, &AMDGPU::SReg_64RegClass);
    MF.addLiveIn(ScratchPtrReg, &AMDGPU::SReg_64RegClass);
  }

  if (Info->getShaderType() == ShaderType::COMPUTE) {
    getOriginalFunctionArgs(DAG, DAG.getMachineFunction().getFunction(), Ins,
                            Splits);
  }

  AnalyzeFormalArguments(CCInfo, Splits);

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
      const unsigned Offset = 36 + VA.getLocMemOffset();
      // The first 36 bytes of the input buffer contains information about
      // thread group and global sizes.
      SDValue Arg = LowerParameter(DAG, VT, MemVT,  DL, DAG.getRoot(),
                                   Offset, Ins[i].Flags.isSExt());

      const PointerType *ParamTy =
        dyn_cast<PointerType>(FType->getParamType(Ins[i].getOrigArgIndex()));
      if (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS &&
          ParamTy && ParamTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
        // On SI local pointers are just offsets into LDS, so they are always
        // less than 16-bits.  On CI and newer they could potentially be
        // real pointers, so we can't guarantee their size.
        Arg = DAG.getNode(ISD::AssertZext, DL, Arg.getValueType(), Arg,
                          DAG.getValueType(MVT::i16));
      }

      InVals.push_back(Arg);
      Info->ABIArgOffset = Offset + MemVT.getStoreSize();
      continue;
    }
    assert(VA.isRegLoc() && "Parameter must be in a register!");

    unsigned Reg = VA.getLocReg();

    if (VT == MVT::i64) {
      // For now assume it is a pointer
      Reg = TRI->getMatchingSuperReg(Reg, AMDGPU::sub0,
                                     &AMDGPU::SReg_64RegClass);
      Reg = MF.addLiveIn(Reg, &AMDGPU::SReg_64RegClass);
      InVals.push_back(DAG.getCopyFromReg(Chain, DL, Reg, VT));
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
        Regs.push_back(DAG.getCopyFromReg(Chain, DL, Reg, VT));
      }

      // Fill up the missing vector elements
      NumElements = Arg.VT.getVectorNumElements() - NumElements;
      for (unsigned j = 0; j != NumElements; ++j)
        Regs.push_back(DAG.getUNDEF(VT));

      InVals.push_back(DAG.getNode(ISD::BUILD_VECTOR, DL, Arg.VT, Regs));
      continue;
    }

    InVals.push_back(Val);
  }

  if (Info->getShaderType() != ShaderType::COMPUTE) {
    unsigned ScratchIdx = CCInfo.getFirstUnallocated(
        AMDGPU::SGPR_32RegClass.begin(), AMDGPU::SGPR_32RegClass.getNumRegs());
    Info->ScratchOffsetReg = AMDGPU::SGPR_32RegClass.getRegister(ScratchIdx);
  }
  return Chain;
}

MachineBasicBlock * SITargetLowering::EmitInstrWithCustomInserter(
    MachineInstr * MI, MachineBasicBlock * BB) const {

  MachineBasicBlock::iterator I = *MI;
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      getTargetMachine().getSubtargetImpl()->getInstrInfo());

  switch (MI->getOpcode()) {
  default:
    return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
  case AMDGPU::BRANCH: return BB;
  case AMDGPU::V_SUB_F64: {
    unsigned DestReg = MI->getOperand(0).getReg();
    BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::V_ADD_F64), DestReg)
      .addImm(0)  // SRC0 modifiers
      .addReg(MI->getOperand(1).getReg())
      .addImm(1)  // SRC1 modifiers
      .addReg(MI->getOperand(2).getReg())
      .addImm(0)  // CLAMP
      .addImm(0); // OMOD
    MI->eraseFromParent();
    break;
  }
  case AMDGPU::SI_RegisterStorePseudo: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    unsigned Reg = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
    MachineInstrBuilder MIB =
        BuildMI(*BB, I, MI->getDebugLoc(), TII->get(AMDGPU::SI_RegisterStore),
                Reg);
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
      MIB.addOperand(MI->getOperand(i));

    MI->eraseFromParent();
    break;
  }
  }
  return BB;
}

EVT SITargetLowering::getSetCCResultType(LLVMContext &Ctx, EVT VT) const {
  if (!VT.isVector()) {
    return MVT::i1;
  }
  return EVT::getVectorVT(Ctx, MVT::i1, VT.getVectorNumElements());
}

MVT SITargetLowering::getScalarShiftAmountTy(EVT VT) const {
  return MVT::i32;
}

bool SITargetLowering::isFMAFasterThanFMulAndFAdd(EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
    return false; /* There is V_MAD_F32 for f32 */
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
  case ISD::STORE: return LowerSTORE(Op, DAG);
  case ISD::GlobalAddress: {
    MachineFunction &MF = DAG.getMachineFunction();
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    return LowerGlobalAddress(MFI, Op, DAG);
  }
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID: return LowerINTRINSIC_VOID(Op, DAG);
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

  FrameIndexSDNode *FINode = cast<FrameIndexSDNode>(Op);
  unsigned FrameIndex = FINode->getIndex();

  return DAG.getTargetFrameIndex(FrameIndex, MVT::i32);
}

/// This transforms the control flow intrinsics to get the branch destination as
/// last parameter, also switches branch target with BR if the need arise
SDValue SITargetLowering::LowerBRCOND(SDValue BRCOND,
                                      SelectionDAG &DAG) const {

  SDLoc DL(BRCOND);

  SDNode *Intr = BRCOND.getOperand(1).getNode();
  SDValue Target = BRCOND.getOperand(2);
  SDNode *BR = nullptr;

  if (Intr->getOpcode() == ISD::SETCC) {
    // As long as we negate the condition everything is fine
    SDNode *SetCC = Intr;
    assert(SetCC->getConstantOperandVal(1) == 1);
    assert(cast<CondCodeSDNode>(SetCC->getOperand(2).getNode())->get() ==
           ISD::SETNE);
    Intr = SetCC->getOperand(0).getNode();

  } else {
    // Get the target from BR if we don't negate the condition
    BR = findUser(BRCOND, ISD::BR);
    Target = BR->getOperand(1);
  }

  assert(Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN);

  // Build the result and
  SmallVector<EVT, 4> Res;
  for (unsigned i = 1, e = Intr->getNumValues(); i != e; ++i)
    Res.push_back(Intr->getValueType(i));

  // operands of the new intrinsic call
  SmallVector<SDValue, 4> Ops;
  Ops.push_back(BRCOND.getOperand(0));
  for (unsigned i = 1, e = Intr->getNumOperands(); i != e; ++i)
    Ops.push_back(Intr->getOperand(i));
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

SDValue SITargetLowering::LowerGlobalAddress(AMDGPUMachineFunction *MFI,
                                             SDValue Op,
                                             SelectionDAG &DAG) const {
  GlobalAddressSDNode *GSD = cast<GlobalAddressSDNode>(Op);

  if (GSD->getAddressSpace() != AMDGPUAS::CONSTANT_ADDRESS)
    return AMDGPUTargetLowering::LowerGlobalAddress(MFI, Op, DAG);

  SDLoc DL(GSD);
  const GlobalValue *GV = GSD->getGlobal();
  MVT PtrVT = getPointerTy(GSD->getAddressSpace());

  SDValue Ptr = DAG.getNode(AMDGPUISD::CONST_DATA_PTR, DL, PtrVT);
  SDValue GA = DAG.getTargetGlobalAddress(GV, DL, MVT::i32);

  SDValue PtrLo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Ptr,
                              DAG.getConstant(0, MVT::i32));
  SDValue PtrHi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, MVT::i32, Ptr,
                              DAG.getConstant(1, MVT::i32));

  SDValue Lo = DAG.getNode(ISD::ADDC, DL, DAG.getVTList(MVT::i32, MVT::Glue),
                           PtrLo, GA);
  SDValue Hi = DAG.getNode(ISD::ADDE, DL, DAG.getVTList(MVT::i32, MVT::Glue),
                           PtrHi, DAG.getConstant(0, MVT::i32),
                           SDValue(Lo.getNode(), 1));
  return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, Lo, Hi);
}

SDValue SITargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                  SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  const SIRegisterInfo *TRI =
      static_cast<const SIRegisterInfo*>(MF.getSubtarget().getRegisterInfo());

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();

  switch (IntrinsicID) {
  case Intrinsic::r600_read_ngroups_x:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::NGROUPS_X, false);
  case Intrinsic::r600_read_ngroups_y:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::NGROUPS_Y, false);
  case Intrinsic::r600_read_ngroups_z:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::NGROUPS_Z, false);
  case Intrinsic::r600_read_global_size_x:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::GLOBAL_SIZE_X, false);
  case Intrinsic::r600_read_global_size_y:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::GLOBAL_SIZE_Y, false);
  case Intrinsic::r600_read_global_size_z:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::GLOBAL_SIZE_Z, false);
  case Intrinsic::r600_read_local_size_x:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::LOCAL_SIZE_X, false);
  case Intrinsic::r600_read_local_size_y:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::LOCAL_SIZE_Y, false);
  case Intrinsic::r600_read_local_size_z:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          SI::KernelInputOffsets::LOCAL_SIZE_Z, false);

  case Intrinsic::AMDGPU_read_workdim:
    return LowerParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                          MF.getInfo<SIMachineFunctionInfo>()->ABIArgOffset,
                          false);

  case Intrinsic::r600_read_tgid_x:
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::TGID_X), VT);
  case Intrinsic::r600_read_tgid_y:
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::TGID_Y), VT);
  case Intrinsic::r600_read_tgid_z:
    return CreateLiveInRegister(DAG, &AMDGPU::SReg_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::TGID_Z), VT);
  case Intrinsic::r600_read_tidig_x:
    return CreateLiveInRegister(DAG, &AMDGPU::VGPR_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::TIDIG_X), VT);
  case Intrinsic::r600_read_tidig_y:
    return CreateLiveInRegister(DAG, &AMDGPU::VGPR_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::TIDIG_Y), VT);
  case Intrinsic::r600_read_tidig_z:
    return CreateLiveInRegister(DAG, &AMDGPU::VGPR_32RegClass,
      TRI->getPreloadedValue(MF, SIRegisterInfo::TIDIG_Z), VT);
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
  case AMDGPUIntrinsic::SI_sample:
    return LowerSampleIntrinsic(AMDGPUISD::SAMPLE, Op, DAG);
  case AMDGPUIntrinsic::SI_sampleb:
    return LowerSampleIntrinsic(AMDGPUISD::SAMPLEB, Op, DAG);
  case AMDGPUIntrinsic::SI_sampled:
    return LowerSampleIntrinsic(AMDGPUISD::SAMPLED, Op, DAG);
  case AMDGPUIntrinsic::SI_samplel:
    return LowerSampleIntrinsic(AMDGPUISD::SAMPLEL, Op, DAG);
  case AMDGPUIntrinsic::SI_vs_load_input:
    return DAG.getNode(AMDGPUISD::LOAD_INPUT, DL, VT,
                       Op.getOperand(1),
                       Op.getOperand(2),
                       Op.getOperand(3));

  case AMDGPUIntrinsic::AMDGPU_fract:
  case AMDGPUIntrinsic::AMDIL_fraction: // Legacy name.
    return DAG.getNode(ISD::FSUB, DL, VT, Op.getOperand(1),
                       DAG.getNode(ISD::FFLOOR, DL, VT, Op.getOperand(1)));

  default:
    return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  }
}

SDValue SITargetLowering::LowerINTRINSIC_VOID(SDValue Op,
                                              SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SDValue Chain = Op.getOperand(0);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();

  switch (IntrinsicID) {
  case AMDGPUIntrinsic::SI_tbuffer_store: {
    SDLoc DL(Op);
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
  default:
    return SDValue();
  }
}

SDValue SITargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  LoadSDNode *Load = cast<LoadSDNode>(Op);

  if (Op.getValueType().isVector()) {
    assert(Op.getValueType().getVectorElementType() == MVT::i32 &&
           "Custom lowering for non-i32 vectors hasn't been implemented.");
    unsigned NumElements = Op.getValueType().getVectorNumElements();
    assert(NumElements != 2 && "v2 loads are supported for all address spaces.");
    switch (Load->getAddressSpace()) {
      default: break;
      case AMDGPUAS::GLOBAL_ADDRESS:
      case AMDGPUAS::PRIVATE_ADDRESS:
        // v4 loads are supported for private and global memory.
        if (NumElements <= 4)
          break;
        // fall-through
      case AMDGPUAS::LOCAL_ADDRESS:
        return ScalarizeVectorLoad(Op, DAG);
    }
  }

  return AMDGPUTargetLowering::LowerLOAD(Op, DAG);
}

SDValue SITargetLowering::LowerSampleIntrinsic(unsigned Opcode,
                                               const SDValue &Op,
                                               SelectionDAG &DAG) const {
  return DAG.getNode(Opcode, SDLoc(Op), Op.getValueType(), Op.getOperand(1),
                     Op.getOperand(2),
                     Op.getOperand(3),
                     Op.getOperand(4));
}

SDValue SITargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getValueType() != MVT::i64)
    return SDValue();

  SDLoc DL(Op);
  SDValue Cond = Op.getOperand(0);

  SDValue Zero = DAG.getConstant(0, MVT::i32);
  SDValue One = DAG.getConstant(1, MVT::i32);

  SDValue LHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(1));
  SDValue RHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(2));

  SDValue Lo0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, Zero);
  SDValue Lo1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, Zero);

  SDValue Lo = DAG.getSelect(DL, MVT::i32, Cond, Lo0, Lo1);

  SDValue Hi0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, One);
  SDValue Hi1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, One);

  SDValue Hi = DAG.getSelect(DL, MVT::i32, Cond, Hi0, Hi1);

  SDValue Res = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Res);
}

// Catch division cases where we can use shortcuts with rcp and rsq
// instructions.
SDValue SITargetLowering::LowerFastFDIV(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  EVT VT = Op.getValueType();
  bool Unsafe = DAG.getTarget().Options.UnsafeFPMath;

  if (const ConstantFPSDNode *CLHS = dyn_cast<ConstantFPSDNode>(LHS)) {
    if ((Unsafe || (VT == MVT::f32 && !Subtarget->hasFP32Denormals())) &&
        CLHS->isExactlyValue(1.0)) {
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
  }

  if (Unsafe) {
    // Turn into multiply by the reciprocal.
    // x / y -> x * (1.0 / y)
    SDValue Recip = DAG.getNode(AMDGPUISD::RCP, SL, VT, RHS);
    return DAG.getNode(ISD::FMUL, SL, VT, LHS, Recip);
  }

  return SDValue();
}

SDValue SITargetLowering::LowerFDIV32(SDValue Op, SelectionDAG &DAG) const {
  SDValue FastLowered = LowerFastFDIV(Op, DAG);
  if (FastLowered.getNode())
    return FastLowered;

  // This uses v_rcp_f32 which does not handle denormals. Let this hit a
  // selection error for now rather than do something incorrect.
  if (Subtarget->hasFP32Denormals())
    return SDValue();

  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);

  SDValue r1 = DAG.getNode(ISD::FABS, SL, MVT::f32, RHS);

  const APFloat K0Val(BitsToFloat(0x6f800000));
  const SDValue K0 = DAG.getConstantFP(K0Val, MVT::f32);

  const APFloat K1Val(BitsToFloat(0x2f800000));
  const SDValue K1 = DAG.getConstantFP(K1Val, MVT::f32);

  const SDValue One = DAG.getConstantFP(1.0, MVT::f32);

  EVT SetCCVT = getSetCCResultType(*DAG.getContext(), MVT::f32);

  SDValue r2 = DAG.getSetCC(SL, SetCCVT, r1, K0, ISD::SETOGT);

  SDValue r3 = DAG.getNode(ISD::SELECT, SL, MVT::f32, r2, K1, One);

  r1 = DAG.getNode(ISD::FMUL, SL, MVT::f32, RHS, r3);

  SDValue r0 = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f32, r1);

  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHS, r0);

  return DAG.getNode(ISD::FMUL, SL, MVT::f32, r3, Mul);
}

SDValue SITargetLowering::LowerFDIV64(SDValue Op, SelectionDAG &DAG) const {
  if (DAG.getTarget().Options.UnsafeFPMath)
    return LowerFastFDIV(Op, DAG);

  SDLoc SL(Op);
  SDValue X = Op.getOperand(0);
  SDValue Y = Op.getOperand(1);

  const SDValue One = DAG.getConstantFP(1.0, MVT::f64);

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

  if (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS) {
    // Workaround a hardware bug on SI where the condition output from div_scale
    // is not usable.

    const SDValue Hi = DAG.getConstant(1, MVT::i32);

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

  // These stores are legal.
  if (Store->getAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS) {
    if (VT.isVector() && VT.getVectorNumElements() > 4)
      return ScalarizeVectorStore(Op, DAG);
    return SDValue();
  }

  SDValue Ret = AMDGPUTargetLowering::LowerSTORE(Op, DAG);
  if (Ret.getNode())
    return Ret;

  if (VT.isVector() && VT.getVectorNumElements() >= 8)
      return ScalarizeVectorStore(Op, DAG);

  if (VT == MVT::i1)
    return DAG.getTruncStore(Store->getChain(), DL,
                        DAG.getSExtOrTrunc(Store->getValue(), DL, MVT::i32),
                        Store->getBasePtr(), MVT::i1, Store->getMemOperand());

  return SDValue();
}

SDValue SITargetLowering::LowerTrig(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDValue Arg = Op.getOperand(0);
  SDValue FractPart = DAG.getNode(AMDGPUISD::FRACT, SDLoc(Op), VT,
        DAG.getNode(ISD::FMUL, SDLoc(Op), VT, Arg,
          DAG.getConstantFP(0.5 / M_PI, VT)));

  switch (Op.getOpcode()) {
  case ISD::FCOS:
    return DAG.getNode(AMDGPUISD::COS_HW, SDLoc(Op), VT, FractPart);
  case ISD::FSIN:
    return DAG.getNode(AMDGPUISD::SIN_HW, SDLoc(Op), VT, FractPart);
  default:
    llvm_unreachable("Wrong trig opcode");
  }
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

  // We are primarily trying to catch operations on illegal vector types
  // before they are expanded.
  // For scalars, we can use the more flexible method of checking masked bits
  // after legalization.
  if (!DCI.isBeforeLegalize() ||
      !SrcVT.isVector() ||
      SrcVT.getVectorElementType() != MVT::i8) {
    return SDValue();
  }

  assert(DCI.isBeforeLegalize() && "Unexpected legal type");

  // Weird sized vectors are a pain to handle, but we know 3 is really the same
  // size as 4.
  unsigned NElts = SrcVT.getVectorNumElements();
  if (!SrcVT.isSimple() && NElts != 3)
    return SDValue();

  // Handle v4i8 -> v4f32 extload. Replace the v4i8 with a legal i32 load to
  // prevent a mess from expanding to v4i32 and repacking.
  if (ISD::isNormalLoad(Src.getNode()) && Src.hasOneUse()) {
    EVT LoadVT = getEquivalentMemType(*DAG.getContext(), SrcVT);
    EVT RegVT = getEquivalentLoadRegType(*DAG.getContext(), SrcVT);
    EVT FloatVT = EVT::getVectorVT(*DAG.getContext(), MVT::f32, NElts);
    LoadSDNode *Load = cast<LoadSDNode>(Src);

    unsigned AS = Load->getAddressSpace();
    unsigned Align = Load->getAlignment();
    Type *Ty = LoadVT.getTypeForEVT(*DAG.getContext());
    unsigned ABIAlignment = getDataLayout()->getABITypeAlignment(Ty);

    // Don't try to replace the load if we have to expand it due to alignment
    // problems. Otherwise we will end up scalarizing the load, and trying to
    // repack into the vector for no real reason.
    if (Align < ABIAlignment &&
        !allowsMisalignedMemoryAccesses(LoadVT, AS, Align, nullptr)) {
      return SDValue();
    }

    SDValue NewLoad = DAG.getExtLoad(ISD::ZEXTLOAD, DL, RegVT,
                                     Load->getChain(),
                                     Load->getBasePtr(),
                                     LoadVT,
                                     Load->getMemOperand());

    // Make sure successors of the original load stay after it by updating
    // them to use the new Chain.
    DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 1), NewLoad.getValue(1));

    SmallVector<SDValue, 4> Elts;
    if (RegVT.isVector())
      DAG.ExtractVectorElements(NewLoad, Elts);
    else
      Elts.push_back(NewLoad);

    SmallVector<SDValue, 4> Ops;

    unsigned EltIdx = 0;
    for (SDValue Elt : Elts) {
      unsigned ComponentsInElt = std::min(4u, NElts - 4 * EltIdx);
      for (unsigned I = 0; I < ComponentsInElt; ++I) {
        unsigned Opc = AMDGPUISD::CVT_F32_UBYTE0 + I;
        SDValue Cvt = DAG.getNode(Opc, DL, MVT::f32, Elt);
        DCI.AddToWorklist(Cvt.getNode());
        Ops.push_back(Cvt);
      }

      ++EltIdx;
    }

    assert(Ops.size() == NElts);

    return DAG.getNode(ISD::BUILD_VECTOR, DL, FloatVT, Ops);
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

  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      getTargetMachine().getSubtargetImpl()->getInstrInfo());

  // If the resulting offset is too large, we can't fold it into the addressing
  // mode offset.
  APInt Offset = CAdd->getAPIntValue() << CN1->getAPIntValue();
  if (!TII->canFoldOffset(Offset.getZExtValue(), AddrSpace))
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  SDValue ShlX = DAG.getNode(ISD::SHL, SL, VT, N0.getOperand(0), N1);
  SDValue COffset = DAG.getConstant(Offset, MVT::i32);

  return DAG.getNode(ISD::ADD, SL, VT, ShlX, COffset);
}

SDValue SITargetLowering::performAndCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalize())
    return SDValue();

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

        return DAG.getNode(AMDGPUISD::FP_CLASS, SDLoc(N), MVT::i1,
                           X, DAG.getConstant(Mask, MVT::i32));
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
    return DAG.getNode(AMDGPUISD::FP_CLASS, SDLoc(N), MVT::i1,
                       Src, DAG.getConstant(NewMask, MVT::i32));
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
      return DAG.getConstant(0, MVT::i1);
  }

  return SDValue();
}

static unsigned minMaxOpcToMin3Max3Opc(unsigned Opc) {
  switch (Opc) {
  case ISD::FMAXNUM:
    return AMDGPUISD::FMAX3;
  case AMDGPUISD::SMAX:
    return AMDGPUISD::SMAX3;
  case AMDGPUISD::UMAX:
    return AMDGPUISD::UMAX3;
  case ISD::FMINNUM:
    return AMDGPUISD::FMIN3;
  case AMDGPUISD::SMIN:
    return AMDGPUISD::SMIN3;
  case AMDGPUISD::UMIN:
    return AMDGPUISD::UMIN3;
  default:
    llvm_unreachable("Not a min/max opcode");
  }
}

SDValue SITargetLowering::performMin3Max3Combine(SDNode *N,
                                                 DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  unsigned Opc = N->getOpcode();
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  // Only do this if the inner op has one use since this will just increases
  // register pressure for no benefit.

  // max(max(a, b), c)
  if (Op0.getOpcode() == Opc && Op0.hasOneUse()) {
    SDLoc DL(N);
    return DAG.getNode(minMaxOpcToMin3Max3Opc(Opc),
                       DL,
                       N->getValueType(0),
                       Op0.getOperand(0),
                       Op0.getOperand(1),
                       Op1);
  }

  // max(a, max(b, c))
  if (Op1.getOpcode() == Opc && Op1.hasOneUse()) {
    SDLoc DL(N);
    return DAG.getNode(minMaxOpcToMin3Max3Opc(Opc),
                       DL,
                       N->getValueType(0),
                       Op0,
                       Op1.getOperand(0),
                       Op1.getOperand(1));
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
      return DAG.getNode(AMDGPUISD::FP_CLASS, SL, MVT::i1,
                         LHS.getOperand(0), DAG.getConstant(Mask, MVT::i32));
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
  case ISD::FMAXNUM: // TODO: What about fmax_legacy?
  case ISD::FMINNUM:
  case AMDGPUISD::SMAX:
  case AMDGPUISD::SMIN:
  case AMDGPUISD::UMAX:
  case AMDGPUISD::UMIN: {
    if (DCI.getDAGCombineLevel() >= AfterLegalizeDAG &&
        N->getValueType(0) != MVT::f64 &&
        getTargetMachine().getOptLevel() > CodeGenOpt::None)
      return performMin3Max3Combine(N, DCI);
    break;
  }

  case AMDGPUISD::CVT_F32_UBYTE0:
  case AMDGPUISD::CVT_F32_UBYTE1:
  case AMDGPUISD::CVT_F32_UBYTE2:
  case AMDGPUISD::CVT_F32_UBYTE3: {
    unsigned Offset = N->getOpcode() - AMDGPUISD::CVT_F32_UBYTE0;

    SDValue Src = N->getOperand(0);
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

  case ISD::FADD: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;

    EVT VT = N->getValueType(0);
    if (VT != MVT::f32)
      break;

    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);

    // These should really be instruction patterns, but writing patterns with
    // source modiifiers is a pain.

    // fadd (fadd (a, a), b) -> mad 2.0, a, b
    if (LHS.getOpcode() == ISD::FADD) {
      SDValue A = LHS.getOperand(0);
      if (A == LHS.getOperand(1)) {
        const SDValue Two = DAG.getConstantFP(2.0, MVT::f32);
        return DAG.getNode(AMDGPUISD::MAD, DL, VT, Two, A, RHS);
      }
    }

    // fadd (b, fadd (a, a)) -> mad 2.0, a, b
    if (RHS.getOpcode() == ISD::FADD) {
      SDValue A = RHS.getOperand(0);
      if (A == RHS.getOperand(1)) {
        const SDValue Two = DAG.getConstantFP(2.0, MVT::f32);
        return DAG.getNode(AMDGPUISD::MAD, DL, VT, Two, A, LHS);
      }
    }

    break;
  }
  case ISD::FSUB: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;

    EVT VT = N->getValueType(0);

    // Try to get the fneg to fold into the source modifier. This undoes generic
    // DAG combines and folds them into the mad.
    if (VT == MVT::f32) {
      SDValue LHS = N->getOperand(0);
      SDValue RHS = N->getOperand(1);

      if (LHS.getOpcode() == ISD::FMUL) {
        // (fsub (fmul a, b), c) -> mad a, b, (fneg c)

        SDValue A = LHS.getOperand(0);
        SDValue B = LHS.getOperand(1);
        SDValue C = DAG.getNode(ISD::FNEG, DL, VT, RHS);

        return DAG.getNode(AMDGPUISD::MAD, DL, VT, A, B, C);
      }

      if (RHS.getOpcode() == ISD::FMUL) {
        // (fsub c, (fmul a, b)) -> mad (fneg a), b, c

        SDValue A = DAG.getNode(ISD::FNEG, DL, VT, RHS.getOperand(0));
        SDValue B = RHS.getOperand(1);
        SDValue C = LHS;

        return DAG.getNode(AMDGPUISD::MAD, DL, VT, A, B, C);
      }

      if (LHS.getOpcode() == ISD::FADD) {
        // (fsub (fadd a, a), c) -> mad 2.0, a, (fneg c)

        SDValue A = LHS.getOperand(0);
        if (A == LHS.getOperand(1)) {
          const SDValue Two = DAG.getConstantFP(2.0, MVT::f32);
          SDValue NegRHS = DAG.getNode(ISD::FNEG, DL, VT, RHS);

          return DAG.getNode(AMDGPUISD::MAD, DL, VT, Two, A, NegRHS);
        }
      }

      if (RHS.getOpcode() == ISD::FADD) {
        // (fsub c, (fadd a, a)) -> mad -2.0, a, c

        SDValue A = RHS.getOperand(0);
        if (A == RHS.getOperand(1)) {
          const SDValue NegTwo = DAG.getConstantFP(-2.0, MVT::f32);
          return DAG.getNode(AMDGPUISD::MAD, DL, VT, NegTwo, A, LHS);
        }
      }
    }

    break;
  }
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
  case ISD::ATOMIC_LOAD_UMAX: { // TODO: Target mem intrinsics.
    if (DCI.isBeforeLegalize())
      break;

    MemSDNode *MemNode = cast<MemSDNode>(N);
    SDValue Ptr = MemNode->getBasePtr();

    // TODO: We could also do this for multiplies.
    unsigned AS = MemNode->getAddressSpace();
    if (Ptr.getOpcode() == ISD::SHL && AS != AMDGPUAS::PRIVATE_ADDRESS) {
      SDValue NewPtr = performSHLPtrCombine(Ptr.getNode(), AS, DCI);
      if (NewPtr) {
        SmallVector<SDValue, 8> NewOps;
        for (unsigned I = 0, E = MemNode->getNumOperands(); I != E; ++I)
          NewOps.push_back(MemNode->getOperand(I));

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
  }
  return AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
}

/// \brief Test if RegClass is one of the VSrc classes
static bool isVSrc(unsigned RegClass) {
  switch(RegClass) {
    default: return false;
    case AMDGPU::VS_32RegClassID:
    case AMDGPU::VS_64RegClassID:
      return true;
  }
}

/// \brief Analyze the possible immediate value Op
///
/// Returns -1 if it isn't an immediate, 0 if it's and inline immediate
/// and the immediate value if it's a literal immediate
int32_t SITargetLowering::analyzeImmediate(const SDNode *N) const {

  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
    getTargetMachine().getSubtargetImpl()->getInstrInfo());

  if (const ConstantSDNode *Node = dyn_cast<ConstantSDNode>(N)) {
    if (Node->getZExtValue() >> 32)
      return -1;

    if (TII->isInlineConstant(Node->getAPIntValue()))
      return 0;

    return Node->getZExtValue();
  }

  if (const ConstantFPSDNode *Node = dyn_cast<ConstantFPSDNode>(N)) {
    if (TII->isInlineConstant(Node->getValueAPF().bitcastToAPInt()))
      return 0;

    if (Node->getValueType(0) == MVT::f32)
      return FloatToBits(Node->getValueAPF().convertToFloat());

    return -1;
  }

  return -1;
}

const TargetRegisterClass *SITargetLowering::getRegClassForNode(
                                   SelectionDAG &DAG, const SDValue &Op) const {
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      getTargetMachine().getSubtargetImpl()->getInstrInfo());
  const SIRegisterInfo &TRI = TII->getRegisterInfo();

  if (!Op->isMachineOpcode()) {
    switch(Op->getOpcode()) {
    case ISD::CopyFromReg: {
      MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
      unsigned Reg = cast<RegisterSDNode>(Op->getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        return MRI.getRegClass(Reg);
      }
      return TRI.getPhysRegClass(Reg);
    }
    default:  return nullptr;
    }
  }
  const MCInstrDesc &Desc = TII->get(Op->getMachineOpcode());
  int OpClassID = Desc.OpInfo[Op.getResNo()].RegClass;
  if (OpClassID != -1) {
    return TRI.getRegClass(OpClassID);
  }
  switch(Op.getMachineOpcode()) {
  case AMDGPU::COPY_TO_REGCLASS:
    // Operand 1 is the register class id for COPY_TO_REGCLASS instructions.
    OpClassID = cast<ConstantSDNode>(Op->getOperand(1))->getZExtValue();

    // If the COPY_TO_REGCLASS instruction is copying to a VSrc register
    // class, then the register class for the value could be either a
    // VReg or and SReg.  In order to get a more accurate
    if (isVSrc(OpClassID))
      return getRegClassForNode(DAG, Op.getOperand(0));

    return TRI.getRegClass(OpClassID);
  case AMDGPU::EXTRACT_SUBREG: {
    int SubIdx = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    const TargetRegisterClass *SuperClass =
      getRegClassForNode(DAG, Op.getOperand(0));
    return TRI.getSubClassWithSubReg(SuperClass, SubIdx);
  }
  case AMDGPU::REG_SEQUENCE:
    // Operand 0 is the register class id for REG_SEQUENCE instructions.
    return TRI.getRegClass(
      cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue());
  default:
    return getRegClassFor(Op.getSimpleValueType());
  }
}

/// \brief Does "Op" fit into register class "RegClass" ?
bool SITargetLowering::fitsRegClass(SelectionDAG &DAG, const SDValue &Op,
                                    unsigned RegClass) const {
  const TargetRegisterInfo *TRI =
      getTargetMachine().getSubtargetImpl()->getRegisterInfo();
  const TargetRegisterClass *RC = getRegClassForNode(DAG, Op);
  if (!RC) {
    return false;
  }
  return TRI->getRegClass(RegClass)->hasSubClassEq(RC);
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
  unsigned OldDmask = Node->getConstantOperandVal(0);
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
  Ops.push_back(DAG.getTargetConstant(NewDmask, MVT::i32));
  for (unsigned i = 1, e = Node->getNumOperands(); i != e; ++i)
    Ops.push_back(Node->getOperand(i));
  Node = (MachineSDNode*)DAG.UpdateNodeOperands(Node, Ops);

  // If we only got one lane, replace it with a copy
  // (if NewDmask has only one bit set...)
  if (NewDmask && (NewDmask & (NewDmask-1)) == 0) {
    SDValue RC = DAG.getTargetConstant(AMDGPU::VGPR_32RegClassID, MVT::i32);
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

    SDValue Op = DAG.getTargetConstant(Idx, MVT::i32);
    DAG.UpdateNodeOperands(User, User->getOperand(0), Op);

    switch (Idx) {
    default: break;
    case AMDGPU::sub0: Idx = AMDGPU::sub1; break;
    case AMDGPU::sub1: Idx = AMDGPU::sub2; break;
    case AMDGPU::sub2: Idx = AMDGPU::sub3; break;
    }
  }
}

/// \brief Legalize target independent instructions (e.g. INSERT_SUBREG)
/// with frame index operands.
/// LLVM assumes that inputs are to these instructions are registers.
void SITargetLowering::legalizeTargetIndependentNode(SDNode *Node,
                                                     SelectionDAG &DAG) const {

  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0; i < Node->getNumOperands(); ++i) {
    if (!isa<FrameIndexSDNode>(Node->getOperand(i))) {
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
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      getTargetMachine().getSubtargetImpl()->getInstrInfo());
  Node = AdjustRegClass(Node, DAG);

  if (TII->isMIMG(Node->getMachineOpcode()))
    adjustWritemask(Node, DAG);

  if (Node->getMachineOpcode() == AMDGPU::INSERT_SUBREG ||
      Node->getMachineOpcode() == AMDGPU::REG_SEQUENCE) {
    legalizeTargetIndependentNode(Node, DAG);
    return Node;
  }
  return Node;
}

/// \brief Assign the register class depending on the number of
/// bits set in the writemask
void SITargetLowering::AdjustInstrPostInstrSelection(MachineInstr *MI,
                                                     SDNode *Node) const {
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      getTargetMachine().getSubtargetImpl()->getInstrInfo());

  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  TII->legalizeOperands(MI);

  if (TII->isMIMG(MI->getOpcode())) {
    unsigned VReg = MI->getOperand(0).getReg();
    unsigned Writemask = MI->getOperand(1).getImm();
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

    unsigned NewOpcode = TII->getMaskedMIMGOp(MI->getOpcode(), BitsSet);
    MI->setDesc(TII->get(NewOpcode));
    MRI.setRegClass(VReg, RC);
    return;
  }

  // Replace unused atomics with the no return version.
  int NoRetAtomicOp = AMDGPU::getAtomicNoRetOp(MI->getOpcode());
  if (NoRetAtomicOp != -1) {
    if (!Node->hasAnyUseOfValue(0)) {
      MI->setDesc(TII->get(NoRetAtomicOp));
      MI->RemoveOperand(0);
    }

    return;
  }
}

static SDValue buildSMovImm32(SelectionDAG &DAG, SDLoc DL, uint64_t Val) {
  SDValue K = DAG.getTargetConstant(Val, MVT::i32);
  return SDValue(DAG.getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32, K), 0);
}

MachineSDNode *SITargetLowering::wrapAddr64Rsrc(SelectionDAG &DAG,
                                                SDLoc DL,
                                                SDValue Ptr) const {
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      getTargetMachine().getSubtargetImpl()->getInstrInfo());
#if 1
    // XXX - Workaround for moveToVALU not handling different register class
    // inserts for REG_SEQUENCE.

    // Build the half of the subregister with the constants.
    const SDValue Ops0[] = {
      DAG.getTargetConstant(AMDGPU::SGPR_64RegClassID, MVT::i32),
      buildSMovImm32(DAG, DL, 0),
      DAG.getTargetConstant(AMDGPU::sub0, MVT::i32),
      buildSMovImm32(DAG, DL, TII->getDefaultRsrcDataFormat() >> 32),
      DAG.getTargetConstant(AMDGPU::sub1, MVT::i32)
    };

    SDValue SubRegHi = SDValue(DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL,
                                                  MVT::v2i32, Ops0), 0);

    // Combine the constants and the pointer.
    const SDValue Ops1[] = {
      DAG.getTargetConstant(AMDGPU::SReg_128RegClassID, MVT::i32),
      Ptr,
      DAG.getTargetConstant(AMDGPU::sub0_sub1, MVT::i32),
      SubRegHi,
      DAG.getTargetConstant(AMDGPU::sub2_sub3, MVT::i32)
    };

    return DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL, MVT::v4i32, Ops1);
#else
    const SDValue Ops[] = {
      DAG.getTargetConstant(AMDGPU::SReg_128RegClassID, MVT::i32),
      Ptr,
      DAG.getTargetConstant(AMDGPU::sub0_sub1, MVT::i32),
      buildSMovImm32(DAG, DL, 0),
      DAG.getTargetConstant(AMDGPU::sub2, MVT::i32),
      buildSMovImm32(DAG, DL, TII->getDefaultRsrcFormat() >> 32),
      DAG.getTargetConstant(AMDGPU::sub3, MVT::i32)
    };

    return DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL, MVT::v4i32, Ops);

#endif
}

/// \brief Return a resource descriptor with the 'Add TID' bit enabled
///        The TID (Thread ID) is multipled by the stride value (bits [61:48]
///        of the resource descriptor) to create an offset, which is added to the
///        resource ponter.
MachineSDNode *SITargetLowering::buildRSRC(SelectionDAG &DAG,
                                           SDLoc DL,
                                           SDValue Ptr,
                                           uint32_t RsrcDword1,
                                           uint64_t RsrcDword2And3) const {
  SDValue PtrLo = DAG.getTargetExtractSubreg(AMDGPU::sub0, DL, MVT::i32, Ptr);
  SDValue PtrHi = DAG.getTargetExtractSubreg(AMDGPU::sub1, DL, MVT::i32, Ptr);
  if (RsrcDword1) {
    PtrHi = SDValue(DAG.getMachineNode(AMDGPU::S_OR_B32, DL, MVT::i32, PtrHi,
                                     DAG.getConstant(RsrcDword1, MVT::i32)), 0);
  }

  SDValue DataLo = buildSMovImm32(DAG, DL,
                                  RsrcDword2And3 & UINT64_C(0xFFFFFFFF));
  SDValue DataHi = buildSMovImm32(DAG, DL, RsrcDword2And3 >> 32);

  const SDValue Ops[] = {
    DAG.getTargetConstant(AMDGPU::SReg_128RegClassID, MVT::i32),
    PtrLo,
    DAG.getTargetConstant(AMDGPU::sub0, MVT::i32),
    PtrHi,
    DAG.getTargetConstant(AMDGPU::sub1, MVT::i32),
    DataLo,
    DAG.getTargetConstant(AMDGPU::sub2, MVT::i32),
    DataHi,
    DAG.getTargetConstant(AMDGPU::sub3, MVT::i32)
  };

  return DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL, MVT::v4i32, Ops);
}

MachineSDNode *SITargetLowering::buildScratchRSRC(SelectionDAG &DAG,
                                                  SDLoc DL,
                                                  SDValue Ptr) const {
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      getTargetMachine().getSubtargetImpl()->getInstrInfo());
  uint64_t Rsrc = TII->getDefaultRsrcDataFormat() | AMDGPU::RSRC_TID_ENABLE |
                  0xffffffff; // Size

  return buildRSRC(DAG, DL, Ptr, 0, Rsrc);
}

MachineSDNode *SITargetLowering::AdjustRegClass(MachineSDNode *N,
                                                SelectionDAG &DAG) const {

  SDLoc DL(N);
  unsigned NewOpcode = N->getMachineOpcode();

  switch (N->getMachineOpcode()) {
  default: return N;
  case AMDGPU::S_LOAD_DWORD_IMM:
    NewOpcode = AMDGPU::BUFFER_LOAD_DWORD_ADDR64;
    // Fall-through
  case AMDGPU::S_LOAD_DWORDX2_SGPR:
    if (NewOpcode == N->getMachineOpcode()) {
      NewOpcode = AMDGPU::BUFFER_LOAD_DWORDX2_ADDR64;
    }
    // Fall-through
  case AMDGPU::S_LOAD_DWORDX4_IMM:
  case AMDGPU::S_LOAD_DWORDX4_SGPR: {
    if (NewOpcode == N->getMachineOpcode()) {
      NewOpcode = AMDGPU::BUFFER_LOAD_DWORDX4_ADDR64;
    }
    if (fitsRegClass(DAG, N->getOperand(0), AMDGPU::SReg_64RegClassID)) {
      return N;
    }
    ConstantSDNode *Offset = cast<ConstantSDNode>(N->getOperand(1));

    const SDValue Zero64 = DAG.getTargetConstant(0, MVT::i64);
    SDValue Ptr(DAG.getMachineNode(AMDGPU::S_MOV_B64, DL, MVT::i64, Zero64), 0);
    MachineSDNode *RSrc = wrapAddr64Rsrc(DAG, DL, Ptr);

    SmallVector<SDValue, 8> Ops;
    Ops.push_back(SDValue(RSrc, 0));
    Ops.push_back(N->getOperand(0));

    // The immediate offset is in dwords on SI and in bytes on VI.
    if (Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
      Ops.push_back(DAG.getTargetConstant(Offset->getSExtValue(), MVT::i32));
    else
      Ops.push_back(DAG.getTargetConstant(Offset->getSExtValue() << 2, MVT::i32));

    // Copy remaining operands so we keep any chain and glue nodes that follow
    // the normal operands.
    for (unsigned I = 2, E = N->getNumOperands(); I != E; ++I)
      Ops.push_back(N->getOperand(I));

    return DAG.getMachineNode(NewOpcode, DL, N->getVTList(), Ops);
  }
  }
}

SDValue SITargetLowering::CreateLiveInRegister(SelectionDAG &DAG,
                                               const TargetRegisterClass *RC,
                                               unsigned Reg, EVT VT) const {
  SDValue VReg = AMDGPUTargetLowering::CreateLiveInRegister(DAG, RC, Reg, VT);

  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(DAG.getEntryNode()),
                            cast<RegisterSDNode>(VReg)->getReg(), VT);
}
