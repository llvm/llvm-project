//===-- AGXISelLowering.cpp - AGX DAG lowering implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering of LLVM IR to the AGX selection DAG.
//
// Legal types are i16, i32, i64, f16 and f32. f64 values are softened to i64
// (their arithmetic is integer code from AGXSoftF64), and there are no
// vector types: vectors are scalarized. Booleans are 0/1 in 32-bit registers.
// 64-bit integers are native for add, sub, multiply and bitwise operations;
// shifts, compares and bit counts are split into 32-bit halves here.
//
// Kernel arguments arrive in uniform registers, see LowerFormalArguments and
// llvm/docs/AGXUsage.rst. Calls are not supported: everything is inlined.
//
//===----------------------------------------------------------------------===//

#include "AGXISelLowering.h"
#include "AGX.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXRegisterInfo.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower"

static void diagnose(SelectionDAG &DAG, const SDLoc &DL, const Twine &Msg) {
  const Function &F = DAG.getMachineFunction().getFunction();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(F, Msg, DL.getDebugLoc()));
}

AGXTargetLowering::AGXTargetLowering(const TargetMachine &TM,
                                     const AGXSubtarget &ST)
    : TargetLowering(TM, ST), ST(ST) {
  addRegisterClass(MVT::i16, &AGX::GPR16RegClass);
  addRegisterClass(MVT::f16, &AGX::GPR16RegClass);
  addRegisterClass(MVT::i32, &AGX::GPR32RegClass);
  addRegisterClass(MVT::f32, &AGX::GPR32RegClass);
  addRegisterClass(MVT::i64, &AGX::GPR64RegClass);
  computeRegisterProperties(ST.getRegisterInfo());

  setBooleanContents(ZeroOrOneBooleanContent);
  setSchedulingPreference(Sched::RegPressure);
  setMinFunctionAlignment(Align(2));
  setPrefFunctionAlignment(Align(2));
  setJumpIsExpensive(true);
  // Wider atomics are expanded with locks by AGXExpandWideAtomics, except
  // the 64-bit operations the chip has (shouldExpandAtomicRMWInIR).
  setMaxAtomicSizeInBitsSupported(ST.hasDeviceAtomicU64MinMax() ? 64 : 32);
  setMinCmpXchgSizeInBits(32);

  // There are no calls and no scalar branches: all control flow is
  // structured by AGXAnnotateControlFlow.
  setOperationAction(ISD::BR_CC, {MVT::i16, MVT::i32, MVT::i64, MVT::f16,
                                  MVT::f32}, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);
  setOperationAction({ISD::STACKSAVE, ISD::STACKRESTORE,
                      ISD::DYNAMIC_STACKALLOC},
                     {MVT::i32, MVT::Other}, Expand);
  setOperationAction(ISD::GlobalAddress, {MVT::i32, MVT::i64}, Custom);
  // trap is an instruction; there is no debugger to stop in.
  setOperationAction({ISD::TRAP, ISD::DEBUGTRAP}, MVT::Other, Legal);
  setOperationAction({ISD::INTRINSIC_WO_CHAIN, ISD::INTRINSIC_VOID},
                     MVT::Other, Custom);
  // Fences become the barrier sequences measured for their scope and ordering.
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i32, Custom);
  // llvm.agx.simd.ballot takes an i1, which is promoted by custom lowering.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i1, Custom);
  setTargetDAGCombine(ISD::BRCOND);

  //===--------------------------------------------------------------------===//
  // Integer operations
  //===--------------------------------------------------------------------===//

  for (MVT VT : {MVT::i16, MVT::i32, MVT::i64}) {
    setOperationAction({ISD::SDIV, ISD::UDIV, ISD::SREM, ISD::UREM,
                        ISD::SDIVREM, ISD::UDIVREM, ISD::SMUL_LOHI,
                        ISD::UMUL_LOHI, ISD::BSWAP, ISD::SELECT_CC,
                        ISD::SHL_PARTS, ISD::SRL_PARTS, ISD::SRA_PARTS},
                       VT, Expand);
    setOperationAction({ISD::SMIN, ISD::SMAX, ISD::UMIN, ISD::UMAX}, VT,
                       VT == MVT::i64 ? Expand : Legal);
  }
  setOperationAction({ISD::MULHU, ISD::MULHS}, MVT::i32, Legal);
  setOperationAction({ISD::MULHU, ISD::MULHS}, {MVT::i16, MVT::i64}, Expand);
  setOperationAction({ISD::ROTL, ISD::ROTR, ISD::FSHL}, {MVT::i16, MVT::i64},
                     Expand);
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::FSHL, MVT::i32, Expand);
  setOperationAction(ISD::FSHR, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i16, Promote);
  setOperationAction({ISD::CTPOP, ISD::CTLZ, ISD::CTTZ}, MVT::i16, Promote);
  setOperationAction({ISD::CTLZ, ISD::CTTZ}, MVT::i64, Custom);
  setOperationAction({ISD::SHL, ISD::SRL, ISD::SRA, ISD::MUL}, MVT::i64,
                     Custom);
  setOperationAction(ISD::SETCC, MVT::i64, Custom);
  setOperationAction({ISD::UADDSAT, ISD::USUBSAT, ISD::SADDSAT,
                      ISD::SSUBSAT},
                     MVT::i32, Legal);

  //===--------------------------------------------------------------------===//
  // Floating point operations
  //===--------------------------------------------------------------------===//

  for (MVT VT : {MVT::f16, MVT::f32}) {
    setOperationAction({ISD::FFLOOR, ISD::FCEIL, ISD::FTRUNC, ISD::FRINT,
                        ISD::FNEARBYINT, ISD::FROUNDEVEN, ISD::FMINNUM,
                        ISD::FMAXNUM, ISD::FEXP2, ISD::FLOG2},
                       VT, Legal);
    setOperationAction(ISD::FCOPYSIGN, VT, Custom);
    setOperationAction({ISD::FSUB, ISD::SELECT_CC,
                        ISD::FMINIMUM, ISD::FMAXIMUM, ISD::FMINNUM_IEEE,
                        ISD::FMAXNUM_IEEE, ISD::FSINCOS, ISD::FROUND},
                       VT, Expand);
    // Conditions without a direct hardware condition.
    setCondCodeAction({ISD::SETONE, ISD::SETUEQ, ISD::SETO, ISD::SETUO}, VT,
                      Expand);
  }
  setOperationAction({ISD::FDIV, ISD::FREM, ISD::FSIN, ISD::FCOS, ISD::FEXP,
                      ISD::FEXP10, ISD::FLOG, ISD::FLOG10, ISD::FPOW},
                     MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f16, Custom);
  setOperationAction({ISD::FREM, ISD::FSIN, ISD::FCOS, ISD::FEXP, ISD::FEXP10,
                      ISD::FLOG, ISD::FLOG10, ISD::FPOW},
                     MVT::f16, Promote);
  setOperationAction({ISD::FP16_TO_FP, ISD::FP_TO_FP16}, MVT::f32, Expand);
  // bfloat16 is not a legal type: conversions are integer operations.
  setOperationAction({ISD::BF16_TO_FP, ISD::FP_TO_BF16},
                     {MVT::i16, MVT::f16, MVT::f32}, Expand);
  // The generic expansion truncates; round to nearest even instead.
  setOperationAction(ISD::FP_TO_BF16, {MVT::i16, MVT::f16, MVT::f32}, Custom);
  // Constrained bfloat16 arithmetic (promoted to f32) narrows with the strict
  // form. Directed rounding is done in software, so the dynamic rounding
  // mode is round to nearest even and it is lowered the same way.
  setOperationAction(ISD::STRICT_FP_TO_BF16, {MVT::i16, MVT::f16, MVT::f32},
                     Custom);

  // 64-bit conversions are expanded with 32-bit conversions.
  setOperationAction({ISD::FP_TO_SINT, ISD::FP_TO_UINT, ISD::SINT_TO_FP,
                      ISD::UINT_TO_FP},
                     MVT::i64, Custom);

  //===--------------------------------------------------------------------===//
  // Memory
  //===--------------------------------------------------------------------===//

  for (MVT VT : {MVT::i16, MVT::i32, MVT::i64}) {
    setLoadExtAction({ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD}, VT,
                     MVT::i1, Promote);
    setLoadExtAction({ISD::SEXTLOAD}, VT, {MVT::i8, MVT::i16}, Expand);
  }
  setLoadExtAction({ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD}, MVT::i64,
                   {MVT::i8, MVT::i16, MVT::i32}, Expand);
  for (MVT MemVT : {MVT::i8, MVT::i16, MVT::i32})
    setTruncStoreAction(MVT::i64, MemVT, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, {MVT::f16, MVT::bf16}, Expand);
  setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  setTruncStoreAction(MVT::f32, MVT::bf16, Expand);

  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i32, Expand);

  // There is no library to call: memory intrinsics that AGXCodeGenPrepare
  // leaves alone are always expanded into loads and stores.
  MaxStoresPerMemcpy = MaxStoresPerMemcpyOptSize = ~0U;
  MaxStoresPerMemmove = MaxStoresPerMemmoveOptSize = ~0U;
  MaxStoresPerMemset = MaxStoresPerMemsetOptSize = ~0U;
}

EVT AGXTargetLowering::getSetCCResultType(const DataLayout &DL,
                                          LLVMContext &Context,
                                          EVT VT) const {
  if (VT.isVector())
    return EVT::getVectorVT(Context, MVT::i32, VT.getVectorElementCount());
  return MVT::i32;
}

MVT AGXTargetLowering::getScalarShiftAmountTy(const DataLayout &DL,
                                              EVT VT) const {
  return VT == MVT::i16 ? MVT::i16 : MVT::i32;
}

bool AGXTargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                   EVT VT) const {
  VT = VT.getScalarType();
  return VT == MVT::f32 || VT == MVT::f16;
}

bool AGXTargetLowering::isTruncateFree(EVT SrcVT, EVT DstVT) const {
  // Truncation reads a subregister.
  return SrcVT.isScalarInteger() && DstVT.isScalarInteger() &&
         SrcVT.getSizeInBits() > DstVT.getSizeInBits() &&
         DstVT.getSizeInBits() >= 16;
}

TargetLowering::AtomicExpansionKind
AGXTargetLowering::shouldExpandAtomicRMWInIR(const AtomicRMWInst *RMW) const {
  if (RMW->getType()->isIntegerTy(32)) {
    switch (RMW->getOperation()) {
    case AtomicRMWInst::Add:
    case AtomicRMWInst::Sub:
    case AtomicRMWInst::Xchg:
    case AtomicRMWInst::Min:
    case AtomicRMWInst::Max:
    case AtomicRMWInst::UMin:
    case AtomicRMWInst::UMax:
    case AtomicRMWInst::And:
    case AtomicRMWInst::Or:
    case AtomicRMWInst::Xor:
      return AtomicExpansionKind::None;
    default:
      break;
    }
  }
  // A native float add on device memory where the chip has one; the
  // threadgroup atomic has none.
  if (RMW->getOperation() == AtomicRMWInst::FAdd && RMW->getType()->isFloatTy() &&
      ST.hasDeviceAtomicFAdd() &&
      RMW->getPointerAddressSpace() != AGX::THREADGROUP_ADDRESS)
    return AtomicExpansionKind::None;
  // A native u64 min / max on device memory when the old value is unused.
  if ((RMW->getOperation() == AtomicRMWInst::UMin ||
       RMW->getOperation() == AtomicRMWInst::UMax) &&
      RMW->getType()->isIntegerTy(64) && RMW->use_empty() &&
      ST.hasDeviceAtomicU64MinMax() &&
      RMW->getPointerAddressSpace() != AGX::THREADGROUP_ADDRESS)
    return AtomicExpansionKind::None;
  return AtomicExpansionKind::CmpXChg;
}

TargetLowering::AtomicExpansionKind
AGXTargetLowering::shouldExpandAtomicLoadInIR(LoadInst *LI) const {
  // System-scope 32-bit loads of device memory stay loads, encoded coherent
  // after an f59d: the only form that sees host writes during a dispatch
  // (ptx-re-blockers B7).
  if (isSystemScopeDeviceAccess(LI))
    return AtomicExpansionKind::None;
  // Other atomic loads: a compare and swap that never succeeds in changing
  // memory.
  return AtomicExpansionKind::CmpXChg;
}

TargetLowering::AtomicExpansionKind
AGXTargetLowering::shouldExpandAtomicStoreInIR(StoreInst *SI) const {
  // System-scope 32-bit stores of device memory: a coherent store and an
  // f533, the only form the host sees during a dispatch.
  if (isSystemScopeDeviceAccess(SI))
    return AtomicExpansionKind::None;
  // Atomic stores become atomic exchanges.
  return AtomicExpansionKind::Expand;
}

bool AGXTargetLowering::isSystemScopeDeviceAccess(const Instruction *I) {
  Type *Ty = isa<LoadInst>(I) ? I->getType()
                              : cast<StoreInst>(I)->getValueOperand()->getType();
  const DataLayout &DL = I->getDataLayout();
  return DL.getTypeSizeInBits(Ty) == 32 && AGX::isHostCoherentAccess(*I) &&
         getLoadStoreAddressSpace(I) == AGX::DEVICE_ADDRESS;
}

//===----------------------------------------------------------------------===//
// Calling convention
//===----------------------------------------------------------------------===//

SDValue AGXTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  auto *MFI = MF.getInfo<AGXMachineFunctionInfo>();
  const MCRegisterInfo &MRI = *ST.getRegisterInfo();

  if (IsVarArg)
    diagnose(DAG, DL, "variadic AGX kernels are not supported");

  // Arguments are packed into consecutive uniform registers in 16-bit units,
  // each aligned to its size (64-bit values to 32 bits).
  unsigned Half = 0;
  bool Overflow = false;
  auto Allocate = [&](MVT VT) -> MCRegister {
    unsigned Bits = VT.getSizeInBits();
    AGX::RegKind Kind = Bits == 16   ? AGX::RegKind::UGPR16
                        : Bits == 32 ? AGX::RegKind::UGPR32
                                     : AGX::RegKind::UGPR64;
    Half = alignTo(Half, Bits == 16 ? 1 : 2);
    MCRegister Reg = AGX::getRegister(Kind, Half, MRI);
    Half += Bits / 16;
    if (!Reg)
      Overflow = true;
    return Reg;
  };

  for (const ISD::InputArg &In : Ins) {
    MVT VT = In.VT;
    if (In.Flags.isByVal() || !(VT == MVT::i16 || VT == MVT::f16 ||
                                VT == MVT::i32 || VT == MVT::f32 ||
                                VT == MVT::i64)) {
      diagnose(DAG, DL, "unsupported AGX kernel argument type");
      InVals.push_back(DAG.getUNDEF(VT));
      continue;
    }
    MCRegister PReg = Allocate(VT);
    if (In.isOrigArg()) {
      unsigned Arg = In.getOrigArgIndex();
      if (MFI->ArgHalves.size() <= Arg)
        MFI->ArgHalves.resize(Arg + 1, ~0u);
      if (MFI->ArgHalves[Arg] == ~0u)
        MFI->ArgHalves[Arg] = Half - VT.getSizeInBits() / 16;
    }
    if (!PReg) {
      InVals.push_back(DAG.getUNDEF(VT));
      continue;
    }
    Register VReg = MF.addLiveIn(PReg, getRegClassFor(VT));
    InVals.push_back(DAG.getCopyFromReg(Chain, DL, VReg, VT));
  }

  // Implicit arguments: threadgroups_per_grid.xyz.
  Half = alignTo(Half, 2);
  MFI->setImplicitArgHalfIndex(Half);
  for (unsigned Dim = 0; Dim < 3; ++Dim) {
    MCRegister PReg = Allocate(MVT::i32);
    if (PReg)
      MFI->setImplicitArgVReg(Dim,
                              MF.addLiveIn(PReg, &AGX::GPR32RegClass));
  }
  // The global data buffer address, if the module has device globals.
  if (!AGX::computeGlobalDataLayout(*MF.getFunction().getParent()).empty()) {
    unsigned GlobalHalf = alignTo(Half, 2);
    MCRegister PReg = Allocate(MVT::i64);
    if (PReg)
      MFI->setGlobalData(MF.addLiveIn(PReg, &AGX::GPR64RegClass), GlobalHalf);
  }
  MFI->setUniformHalfCount(Half);

  if (Overflow)
    diagnose(DAG, DL, "AGX kernel arguments exceed the uniform registers");
  return Chain;
}

SDValue
AGXTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               const SDLoc &DL, SelectionDAG &DAG) const {
  if (!Outs.empty())
    diagnose(DAG, DL, "AGX kernels cannot return values");
  return DAG.getNode(AGXISD::STOP, DL, MVT::Other, Chain);
}

SDValue AGXTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  StringRef Name = "<indirect>";
  if (auto *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee))
    Name = G->getGlobal()->getName();
  else if (auto *E = dyn_cast<ExternalSymbolSDNode>(CLI.Callee))
    Name = E->getSymbol();
  diagnose(DAG, CLI.DL, "unsupported call to '" + Name + "' (AGX has no calls)");
  for (const ISD::InputArg &In : CLI.Ins)
    InVals.push_back(DAG.getUNDEF(In.VT));
  return CLI.Chain;
}

//===----------------------------------------------------------------------===//
// Custom lowering
//===----------------------------------------------------------------------===//

/// Low and high halves of a 64-bit value.
static std::pair<SDValue, SDValue> split64(SelectionDAG &DAG, const SDLoc &DL,
                                           SDValue V) {
  return {DAG.getTargetExtractSubreg(AGX::sub0, DL, MVT::i32, V),
          DAG.getTargetExtractSubreg(AGX::sub1, DL, MVT::i32, V)};
}

static SDValue join64(SelectionDAG &DAG, const SDLoc &DL, SDValue Lo,
                      SDValue Hi) {
  SDValue Ops[] = {
      DAG.getTargetConstant(AGX::GPR64RegClassID, DL, MVT::i32), Lo,
      DAG.getTargetConstant(AGX::sub0, DL, MVT::i32), Hi,
      DAG.getTargetConstant(AGX::sub1, DL, MVT::i32)};
  return SDValue(
      DAG.getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::i64, Ops), 0);
}

SDValue AGXTargetLowering::LowerOperation(SDValue Op,
                                          SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN:
    return lowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID:
    return lowerINTRINSIC_VOID(Op, DAG);
  case ISD::ATOMIC_FENCE:
    return lowerATOMIC_FENCE(Op, DAG);
  case ISD::ATOMIC_LOAD_SUB:
    return lowerATOMIC_LOAD_SUB(Op, DAG);
  case ISD::BRCOND:
    return lowerBRCOND(Op, DAG);
  case ISD::SETCC:
    return lowerSETCC64(Op, DAG);
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    return lowerShift64(Op, DAG);
  case ISD::MUL:
    return lowerMUL64(Op, DAG);
  case ISD::CTLZ:
  case ISD::CTTZ:
    return lowerCTLZ_CTTZ64(Op, DAG);
  case ISD::FP_TO_BF16:
    return lowerFP_TO_BF16(Op, DAG);
  case ISD::STRICT_FP_TO_BF16:
    return DAG.getMergeValues({lowerFP_TO_BF16(Op, DAG), Op.getOperand(0)},
                              SDLoc(Op));
  case ISD::FDIV:
    return lowerFDIV(Op, DAG);
  case ISD::FREM:
    return lowerFREM(Op, DAG);
  case ISD::FCOPYSIGN:
    return lowerFCOPYSIGN(Op, DAG);
  case ISD::FSIN:
  case ISD::FCOS:
    return lowerTrig(Op, DAG);
  case ISD::FEXP:
  case ISD::FEXP10:
  case ISD::FLOG:
  case ISD::FLOG10:
  case ISD::FPOW:
    return lowerExpLog(Op, DAG);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return lowerINT_TO_FP64(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return lowerFP_TO_INT64(Op, DAG);
  default:
    llvm_unreachable("unexpected custom lowering");
  }
}

void AGXTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {
  // The custom operations do not have illegal result types.
}

SDValue AGXTargetLowering::lowerGlobalAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  auto *GA = cast<GlobalAddressSDNode>(Op);
  const auto *GV = dyn_cast<GlobalVariable>(GA->getGlobal());
  MachineFunction &MF = DAG.getMachineFunction();
  auto *MFI = MF.getInfo<AGXMachineFunctionInfo>();

  // Device and constant variables: an offset into the global data buffer.
  if (GV && AGX::isGlobalDataVariable(*GV)) {
    std::optional<uint64_t> Offset =
        AGX::computeGlobalDataLayout(*MF.getFunction().getParent())
            .getOffset(GV);
    if (!Offset || !MFI->getGlobalDataVReg()) {
      diagnose(DAG, DL,
               "unsupported external global '" + GV->getName() + "'");
      return DAG.getUNDEF(Op.getValueType());
    }
    SDValue Base = DAG.getCopyFromReg(DAG.getEntryNode(), DL,
                                      MFI->getGlobalDataVReg(), MVT::i64);
    SDValue Addr =
        DAG.getNode(ISD::ADD, DL, MVT::i64, Base,
                    DAG.getConstant(*Offset + GA->getOffset(), DL, MVT::i64));
    return DAG.getZExtOrTrunc(Addr, DL, Op.getValueType());
  }

  if (!GV || GA->getAddressSpace() != AGX::THREADGROUP_ADDRESS) {
    diagnose(DAG, DL,
             "unsupported global '" + GA->getGlobal()->getName() + "'");
    return DAG.getUNDEF(Op.getValueType());
  }
  if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer()))
    diagnose(DAG, DL, "threadgroup variables cannot have initializers");

  // Threadgroup variables are allocated statically per kernel; their
  // address is a constant offset into threadgroup memory.
  unsigned Offset =
      MFI->allocateThreadgroupVariable(DAG.getMachineFunction().getFunction(), *GV);
  return DAG.getConstant(Offset + GA->getOffset(), DL, Op.getValueType());
}

SDValue AGXTargetLowering::lowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  unsigned IID = Op.getConstantOperandVal(0);
  switch (IID) {
  case Intrinsic::agx_threadgroups_per_grid_x:
  case Intrinsic::agx_threadgroups_per_grid_y:
  case Intrinsic::agx_threadgroups_per_grid_z: {
    auto *MFI = DAG.getMachineFunction().getInfo<AGXMachineFunctionInfo>();
    Register VReg =
        MFI->getImplicitArgVReg(IID - Intrinsic::agx_threadgroups_per_grid_x);
    if (!VReg)
      return DAG.getUNDEF(MVT::i32);
    return DAG.getCopyFromReg(DAG.getEntryNode(), DL, VReg, MVT::i32);
  }
  case Intrinsic::agx_simd_ballot:
  case Intrinsic::agx_quad_ballot: {
    SDValue Cond = DAG.getZExtOrTrunc(Op.getOperand(1), DL, MVT::i32);
    return DAG.getNode(IID == Intrinsic::agx_simd_ballot
                           ? AGXISD::BALLOT
                           : AGXISD::QUAD_BALLOT,
                       DL, MVT::i32, Cond);
  }
  default:
    return SDValue();
  }
}

SDValue AGXTargetLowering::lowerINTRINSIC_VOID(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  switch (Op.getConstantOperandVal(1)) {
  case Intrinsic::agx_end_cf:
    return DAG.getNode(AGXISD::END_CF, DL, MVT::Other, Op.getOperand(0));
  case Intrinsic::agx_loop_begin:
    return DAG.getNode(AGXISD::LOOP_BEGIN, DL, MVT::Other, Op.getOperand(0));
  default:
    return SDValue();
  }
}

// The memory barriers of G13/G14, by their 16-bit encodings (memory_barrier
// a, b, c is 0xf5 | b << 8 | a << 10 | c << 12). Their effect was measured
// with cross-core litmus tests on G14C (M2 Max) under Asahi Linux
// (Apple-M2-Playground novel-results 2026-10-11-memory-ordering,
// 2026-10-18-sb-lb, 2026-10-18-atomic-release, 2026-10-18-fence-cost):
//   f596 (Mesa's memory_barrier) does not order stores between cores;
//   f533 is the store release (also needed before an atomic flag);
//   wait 0 then f59d is the acquire;
//   store buffering needs the whole sequence Mesa uses at queue-family scope.
// Threads of one threadgroup run on one core, where no reordering was seen.
enum AGXBarrier : unsigned { F596, F533, F59D, F59A, F59E, F503, F599 };

static SDValue emitBarrier(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                           AGXBarrier B) {
  static const unsigned Ops[][3] = {
      {1, 2, 9}, {0, 3, 3}, {3, 1, 9}, {2, 2, 9}, {3, 2, 9}, {0, 3, 0},
      {2, 1, 9}};
  SDValue Args[] = {
      Chain, DAG.getTargetConstant(Intrinsic::agx_memory_barrier, DL, MVT::i32),
      DAG.getTargetConstant(Ops[B][0], DL, MVT::i32),
      DAG.getTargetConstant(Ops[B][1], DL, MVT::i32),
      DAG.getTargetConstant(Ops[B][2], DL, MVT::i32)};
  return DAG.getNode(ISD::INTRINSIC_VOID, DL, MVT::Other, Args);
}

static SDValue emitWait(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain) {
  SDValue Args[] = {Chain,
                    DAG.getTargetConstant(Intrinsic::agx_wait, DL, MVT::i32),
                    DAG.getTargetConstant(0, DL, MVT::i32)};
  return DAG.getNode(ISD::INTRINSIC_VOID, DL, MVT::Other, Args);
}

SDValue AGXTargetLowering::lowerATOMIC_FENCE(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  auto Ordering = static_cast<AtomicOrdering>(Op.getConstantOperandVal(1));
  auto Scope = static_cast<SyncScope::ID>(Op.getConstantOperandVal(2));
  if (Scope == SyncScope::SingleThread)
    return Chain;
  LLVMContext &Ctx = *DAG.getContext();
  if (Scope == Ctx.getOrInsertSyncScopeID("workgroup") ||
      Scope == Ctx.getOrInsertSyncScopeID("workgroup-one-as"))
    return emitBarrier(DAG, DL, Chain, F596);
  // Device and system scope.
  switch (Ordering) {
  case AtomicOrdering::Acquire:
    return emitBarrier(DAG, DL, emitWait(DAG, DL, Chain), F59D);
  case AtomicOrdering::Release:
    return emitBarrier(DAG, DL, Chain, F533);
  case AtomicOrdering::AcquireRelease:
    Chain = emitBarrier(DAG, DL, Chain, F533);
    return emitBarrier(DAG, DL, emitWait(DAG, DL, Chain), F59D);
  default:
    for (AGXBarrier B : {F596, F533, F59A, F59E, F503, F599})
      Chain = emitBarrier(DAG, DL, Chain, B);
    return emitBarrier(DAG, DL, emitWait(DAG, DL, Chain), F59D);
  }
}

Instruction *AGXTargetLowering::emitLeadingFence(IRBuilderBase &Builder,
                                                 Instruction *Inst,
                                                 AtomicOrdering Ord) const {
  if (isReleaseOrStronger(Ord) && Inst->hasAtomicStore())
    return Builder.CreateFence(isAcquireOrStronger(Ord) ? AtomicOrdering::Release
                                                        : Ord,
                               getAtomicSyncScopeID(Inst).value_or(
                                   SyncScope::System));
  return nullptr;
}

Instruction *AGXTargetLowering::emitTrailingFence(IRBuilderBase &Builder,
                                                  Instruction *Inst,
                                                  AtomicOrdering Ord) const {
  if (isAcquireOrStronger(Ord))
    return Builder.CreateFence(
        Ord == AtomicOrdering::SequentiallyConsistent
            ? AtomicOrdering::SequentiallyConsistent
            : AtomicOrdering::Acquire,
        getAtomicSyncScopeID(Inst).value_or(SyncScope::System));
  return nullptr;
}

SDValue AGXTargetLowering::lowerATOMIC_LOAD_SUB(SDValue Op,
                                               SelectionDAG &DAG) const {
  // x - v is x + (-v). The add keeps the meaning of the opcode with the
  // coherency bit set, which turns the sub opcode into a float add on G14.
  auto *N = cast<AtomicSDNode>(Op.getNode());
  SDLoc DL(Op);
  SDValue Neg = DAG.getNode(ISD::SUB, DL, MVT::i32,
                            DAG.getConstant(0, DL, MVT::i32), N->getVal());
  return DAG.getAtomic(ISD::ATOMIC_LOAD_ADD, DL, N->getMemoryVT(),
                       N->getChain(), N->getBasePtr(), Neg,
                       N->getMemOperand());
}

SDValue AGXTargetLowering::lowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  // Every conditional branch is rewritten into execution mask operations by
  // performBRCONDCombine; anything else is unstructured control flow.
  diagnose(DAG, SDLoc(Op),
           "unstructured conditional branch (AGXAnnotateControlFlow did not "
           "run)");
  return Op.getOperand(0);
}

/// Rewrite
///
///   brcond (llvm.agx.if/else/loop ...), %a
///   br %b
///
/// to the target node, branching to the block executed when no thread
/// remains active, and an unconditional branch to the other block. This is
/// done before type legalization so that the i1 results of the control flow
/// intrinsics never need to be legalized.
SDValue AGXTargetLowering::performBRCONDCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  SDValue Cond = N->getOperand(1);
  SDValue Target = N->getOperand(2);

  bool Negated = false;
  if (Cond.getOpcode() == ISD::XOR && isOneConstant(Cond.getOperand(1))) {
    Negated = true;
    Cond = Cond.getOperand(0);
  } else if (Cond.getOpcode() == ISD::SETCC &&
             isa<ConstantSDNode>(Cond.getOperand(1))) {
    ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
    bool IsOne = isOneConstant(Cond.getOperand(1));
    bool IsZero = isNullConstant(Cond.getOperand(1));
    if ((CC == ISD::SETEQ && IsZero) || (CC == ISD::SETNE && IsOne))
      Negated = true;
    else if (!((CC == ISD::SETNE && IsZero) || (CC == ISD::SETEQ && IsOne)))
      return SDValue();
    Cond = Cond.getOperand(0);
  }

  if (Cond.getOpcode() != ISD::INTRINSIC_W_CHAIN)
    return SDValue();
  unsigned IID = Cond.getConstantOperandVal(1);
  unsigned Opc;
  switch (IID) {
  case Intrinsic::agx_if:
    Opc = AGXISD::IF;
    break;
  case Intrinsic::agx_else:
    Opc = AGXISD::ELSE;
    break;
  case Intrinsic::agx_loop:
    Opc = AGXISD::LOOP;
    break;
  default:
    return SDValue();
  }
  SDNode *Intr = Cond.getNode();

  // The block to skip to. Without negation, it is the destination of the
  // unconditional branch that follows, which is redirected to the original
  // destination.
  SDNode *BR = nullptr;
  if (!Negated) {
    for (SDNode *U : N->users())
      if (U->getOpcode() == ISD::BR)
        BR = U;
    if (!BR)
      return SDValue();
    Target = BR->getOperand(1);
  }

  SmallVector<SDValue, 3> Ops = {N->getOperand(0)};
  if (Opc != AGXISD::ELSE)
    Ops.push_back(DAG.getZExtOrTrunc(Intr->getOperand(2), DL, MVT::i32));
  Ops.push_back(Target);
  SDValue Result = DAG.getNode(Opc, DL, MVT::Other, Ops);

  if (BR) {
    SDValue NewBR = DAG.getNode(ISD::BR, SDLoc(BR), MVT::Other,
                                BR->getOperand(0), N->getOperand(2));
    DAG.ReplaceAllUsesWith(BR, NewBR.getNode());
  }

  // Remove the intrinsic from the chain.
  DAG.ReplaceAllUsesOfValueWith(SDValue(Intr, Intr->getNumValues() - 1),
                                Intr->getOperand(0));
  return Result;
}

SDValue AGXTargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  case ISD::BRCOND:
    return performBRCONDCombine(N, DCI);
  default:
    return SDValue();
  }
}

static ISD::CondCode getUnsignedCondCode(ISD::CondCode CC) {
  switch (CC) {
  case ISD::SETLT:
    return ISD::SETULT;
  case ISD::SETLE:
    return ISD::SETULE;
  case ISD::SETGT:
    return ISD::SETUGT;
  case ISD::SETGE:
    return ISD::SETUGE;
  default:
    return CC;
  }
}

SDValue AGXTargetLowering::lowerSETCC64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  auto [ALo, AHi] = split64(DAG, DL, Op.getOperand(0));
  auto [BLo, BHi] = split64(DAG, DL, Op.getOperand(1));

  if (CC == ISD::SETEQ || CC == ISD::SETNE) {
    SDValue Lo = DAG.getSetCC(DL, VT, ALo, BLo, CC);
    SDValue Hi = DAG.getSetCC(DL, VT, AHi, BHi, CC);
    return DAG.getNode(CC == ISD::SETEQ ? ISD::AND : ISD::OR, DL, VT, Lo, Hi);
  }

  // The high halves decide unless they are equal; the low halves always
  // compare unsigned.
  SDValue HiEq = DAG.getSetCC(DL, VT, AHi, BHi, ISD::SETEQ);
  SDValue HiCmp = DAG.getSetCC(DL, VT, AHi, BHi, CC);
  SDValue LoCmp = DAG.getSetCC(DL, VT, ALo, BLo, getUnsignedCondCode(CC));
  return DAG.getSelect(DL, VT, HiEq, LoCmp, HiCmp);
}

SDValue AGXTargetLowering::lowerShift64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  unsigned Opc = Op.getOpcode();
  auto [Lo, Hi] = split64(DAG, DL, Op.getOperand(0));
  SDValue Amt = DAG.getZExtOrTrunc(Op.getOperand(1), DL, MVT::i32);
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  auto C32 = [&](uint64_t V) { return DAG.getConstant(V, DL, MVT::i32); };

  // Shifts by a constant stay legal for address mode matching; they are
  // expanded during instruction selection.
  if (isa<ConstantSDNode>(Op.getOperand(1)))
    return Op;

  // Variable shifts: shift each half by the amount modulo 32, then select
  // on bit 5 of the amount.
  SDValue A5 = DAG.getNode(ISD::AND, DL, MVT::i32, Amt, C32(31));
  SDValue Big = DAG.getSetCC(
      DL, MVT::i32, DAG.getNode(ISD::AND, DL, MVT::i32, Amt, C32(32)), Zero,
      ISD::SETNE);
  SDValue NewLo, NewHi;
  if (Opc == ISD::SHL) {
    // hi = (hi << s) | ((lo >> 1) >> (31 - s)), avoiding a shift by 32.
    SDValue LoS = DAG.getNode(ISD::SHL, DL, MVT::i32, Lo, A5);
    SDValue Carry = DAG.getNode(
        ISD::SRL, DL, MVT::i32,
        DAG.getNode(ISD::SRL, DL, MVT::i32, Lo, C32(1)),
        DAG.getNode(ISD::XOR, DL, MVT::i32, A5, C32(31)));
    SDValue HiS = DAG.getNode(ISD::OR, DL, MVT::i32,
                              DAG.getNode(ISD::SHL, DL, MVT::i32, Hi, A5),
                              Carry);
    NewLo = DAG.getSelect(DL, MVT::i32, Big, Zero, LoS);
    NewHi = DAG.getSelect(DL, MVT::i32, Big, LoS, HiS);
  } else {
    SDValue HiS = DAG.getNode(Opc, DL, MVT::i32, Hi, A5);
    SDValue LoS = DAG.getNode(ISD::FSHR, DL, MVT::i32, Hi, Lo, A5);
    SDValue Fill = Opc == ISD::SRA
                       ? DAG.getNode(ISD::SRA, DL, MVT::i32, Hi, C32(31))
                       : Zero;
    NewLo = DAG.getSelect(DL, MVT::i32, Big, HiS, LoS);
    NewHi = DAG.getSelect(DL, MVT::i32, Big, Fill, HiS);
  }
  return join64(DAG, DL, NewLo, NewHi);
}

SDValue AGXTargetLowering::lowerMUL64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  auto IsExt32 = [](SDValue V, unsigned Opc) {
    return V.getOpcode() == Opc && V.getOperand(0).getValueType() == MVT::i32;
  };
  // 32 x 32 -> 64-bit products are a single imadd.
  if ((IsExt32(A, ISD::ZERO_EXTEND) && IsExt32(B, ISD::ZERO_EXTEND)) ||
      (IsExt32(A, ISD::SIGN_EXTEND) && IsExt32(B, ISD::SIGN_EXTEND)))
    return Op;

  // lo(a) * lo(b) as a 64-bit product, plus the cross products in the high
  // half.
  auto [ALo, AHi] = split64(DAG, DL, A);
  auto [BLo, BHi] = split64(DAG, DL, B);
  SDValue P = DAG.getNode(ISD::MUL, DL, MVT::i64,
                          DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, ALo),
                          DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, BLo));
  auto [PLo, PHi] = split64(DAG, DL, P);
  SDValue Cross = DAG.getNode(ISD::ADD, DL, MVT::i32,
                              DAG.getNode(ISD::MUL, DL, MVT::i32, ALo, BHi),
                              DAG.getNode(ISD::MUL, DL, MVT::i32, AHi, BLo));
  return join64(DAG, DL, PLo, DAG.getNode(ISD::ADD, DL, MVT::i32, PHi, Cross));
}

SDValue AGXTargetLowering::lowerCTLZ_CTTZ64(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  bool IsCTLZ = Op.getOpcode() == ISD::CTLZ;
  auto [Lo, Hi] = split64(DAG, DL, Op.getOperand(0));
  // Count in the half that decides; fall back to the other half plus 32.
  SDValue First = IsCTLZ ? Hi : Lo;
  SDValue Second = IsCTLZ ? Lo : Hi;
  unsigned Opc = Op.getOpcode();
  SDValue FirstZero =
      DAG.getSetCC(DL, MVT::i32, First, DAG.getConstant(0, DL, MVT::i32),
                   ISD::SETEQ);
  SDValue R = DAG.getSelect(
      DL, MVT::i32, FirstZero,
      DAG.getNode(ISD::ADD, DL, MVT::i32,
                  DAG.getNode(Opc, DL, MVT::i32, Second),
                  DAG.getConstant(32, DL, MVT::i32)),
      DAG.getNode(Opc, DL, MVT::i32, First));
  return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, R);
}

static SDValue getAGXIntrinsic(SelectionDAG &DAG, const SDLoc &DL,
                               Intrinsic::ID IID, EVT VT, SDValue Arg) {
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                     DAG.getTargetConstant(IID, DL, MVT::i32), Arg);
}

SDValue AGXTargetLowering::lowerFDIV(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDNodeFlags Flags = Op->getFlags();
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  SDValue R = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rcp, VT, B);

  auto *CA = dyn_cast<ConstantFPSDNode>(A);
  if (CA && CA->isExactlyValue(1.0))
    return R;
  SDValue Q = DAG.getNode(ISD::FMUL, DL, VT, A, R, Flags);
  if (Flags.hasAllowReciprocal() || Flags.hasApproximateFuncs() ||
      VT == MVT::f16)
    return Q;

  // One Newton-Raphson step on the quotient: q += (a - b * q) * rcp(b).
  SDValue E = DAG.getNode(ISD::FMA, DL, VT, DAG.getNode(ISD::FNEG, DL, VT, B),
                          Q, A);
  SDValue Refined = DAG.getNode(ISD::FMA, DL, VT, E, R, Q);
  // The step turns an infinite quotient into NaN (0 * inf) and a zero one
  // into +0; those quotients are already exact.
  EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  SDValue AbsQ = DAG.getNode(ISD::FABS, DL, VT, Q);
  SDValue Zero = DAG.getConstantFP(0.0, DL, VT);
  SDValue Inf = DAG.getConstantFP(APFloat::getInf(VT.getFltSemantics()), DL,
                                  VT);
  SDValue Exact =
      DAG.getNode(ISD::OR, DL, CCVT,
                  DAG.getSetCC(DL, CCVT, AbsQ, Zero, ISD::SETOEQ),
                  DAG.getSetCC(DL, CCVT, AbsQ, Inf, ISD::SETOEQ));
  return DAG.getSelect(DL, VT, Exact, Q, Refined);
}

/// copysign(m, s) = (m & ~sign) | (s & sign), on the bits.
SDValue AGXTargetLowering::lowerFCOPYSIGN(SDValue Op,
                                          SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  EVT IntVT = VT == MVT::f16 ? MVT::i16 : MVT::i32;
  SDValue Mag = Op.getOperand(0);
  SDValue Sign = Op.getOperand(1);
  if (Sign.getValueType() != VT)
    Sign = Sign.getValueType().bitsLT(VT)
               ? DAG.getNode(ISD::FP_EXTEND, DL, VT, Sign)
               : DAG.getNode(ISD::FP_ROUND, DL, VT, Sign,
                             DAG.getTargetConstant(0, DL, MVT::i32));
  APInt SignMask = APInt::getSignMask(IntVT.getSizeInBits());
  SDValue M = DAG.getNode(ISD::AND, DL, IntVT,
                          DAG.getBitcast(IntVT, Mag),
                          DAG.getConstant(~SignMask, DL, IntVT));
  SDValue S = DAG.getNode(ISD::AND, DL, IntVT, DAG.getBitcast(IntVT, Sign),
                          DAG.getConstant(SignMask, DL, IntVT));
  return DAG.getBitcast(VT, DAG.getNode(ISD::OR, DL, IntVT, M, S));
}

/// float to bfloat bits, rounded to nearest even: (x + 0x7fff + lsb) >> 16,
/// with NaNs quieted.
SDValue AGXTargetLowering::lowerFP_TO_BF16(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Src = Op.getOperand(Op->isStrictFPOpcode() ? 1 : 0);
  if (Src.getValueType() != MVT::f32)
    Src = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f32, Src);
  SDValue X = DAG.getBitcast(MVT::i32, Src);
  auto C = [&](uint32_t V) { return DAG.getConstant(V, DL, MVT::i32); };
  SDValue Lsb = DAG.getNode(ISD::AND, DL, MVT::i32,
                            DAG.getNode(ISD::SRL, DL, MVT::i32, X, C(16)), C(1));
  SDValue R = DAG.getNode(ISD::ADD, DL, MVT::i32, X,
                          DAG.getNode(ISD::ADD, DL, MVT::i32, Lsb, C(0x7fff)));
  R = DAG.getNode(ISD::SRL, DL, MVT::i32, R, C(16));
  SDValue Quiet = DAG.getNode(ISD::OR, DL, MVT::i32,
                              DAG.getNode(ISD::SRL, DL, MVT::i32, X, C(16)), C(0x40));
  EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::i32);
  SDValue IsNaN = DAG.getSetCC(DL, CCVT,
                               DAG.getNode(ISD::AND, DL, MVT::i32, X, C(0x7fffffff)),
                               C(0x7f800000), ISD::SETUGT);
  R = DAG.getSelect(DL, MVT::i32, IsNaN, Quiet, R);
  return DAG.getNode(ISD::TRUNCATE, DL, Op.getValueType(), R);
}

SDValue AGXTargetLowering::lowerFREM(SDValue Op, SelectionDAG &DAG) const {
  // a - trunc(a / b) * b
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  SDValue Q = DAG.getNode(ISD::FDIV, DL, VT, A, B, Op->getFlags());
  SDValue T = DAG.getNode(ISD::FTRUNC, DL, VT, Q);
  return DAG.getNode(ISD::FMA, DL, VT, DAG.getNode(ISD::FNEG, DL, VT, T), B,
                     A);
}

SDValue AGXTargetLowering::lowerTrig(SDValue Op, SelectionDAG &DAG) const {
  // The hardware evaluates sin(x * pi / 2) for x in [0, 4): reduce the
  // argument to turns, take the fraction and scale to quadrants. cos(x) is
  // sin(x + pi / 2), a quarter turn ahead.
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue Turns = DAG.getNode(ISD::FMUL, DL, VT, Op.getOperand(0),
                              DAG.getConstantFP(0.5 * M_1_PI, DL, VT));
  if (Op.getOpcode() == ISD::FCOS)
    Turns = DAG.getNode(ISD::FADD, DL, VT, Turns,
                        DAG.getConstantFP(0.25, DL, VT));
  SDValue Frac = DAG.getNode(ISD::FSUB, DL, VT, Turns,
                             DAG.getNode(ISD::FFLOOR, DL, VT, Turns));
  SDValue Quadrants =
      DAG.getNode(ISD::FMUL, DL, VT, Frac, DAG.getConstantFP(4.0, DL, VT));
  return DAG.getNode(AGXISD::SIN_QUADRANTS, DL, VT, Quadrants);
}

SDValue AGXTargetLowering::lowerExpLog(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue X = Op.getOperand(0);
  auto K = [&](double V) { return DAG.getConstantFP(V, DL, VT); };
  switch (Op.getOpcode()) {
  case ISD::FEXP:
    return DAG.getNode(ISD::FEXP2, DL, VT,
                       DAG.getNode(ISD::FMUL, DL, VT, X, K(M_LOG2E)));
  case ISD::FEXP10:
    return DAG.getNode(ISD::FEXP2, DL, VT,
                       DAG.getNode(ISD::FMUL, DL, VT, X, K(3.321928094887362)));
  case ISD::FLOG:
    return DAG.getNode(ISD::FMUL, DL, VT,
                       DAG.getNode(ISD::FLOG2, DL, VT, X), K(M_LN2));
  case ISD::FLOG10:
    return DAG.getNode(ISD::FMUL, DL, VT,
                       DAG.getNode(ISD::FLOG2, DL, VT, X),
                       K(0.301029995663981195));
  case ISD::FPOW:
    return DAG.getNode(
        ISD::FEXP2, DL, VT,
        DAG.getNode(ISD::FMUL, DL, VT, Op.getOperand(1),
                    DAG.getNode(ISD::FLOG2, DL, VT, X)));
  default:
    llvm_unreachable("unexpected opcode");
  }
}

SDValue AGXTargetLowering::lowerINT_TO_FP64(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  bool Signed = Op.getOpcode() == ISD::SINT_TO_FP;
  SDValue X = Op.getOperand(0);
  auto C32 = [&](uint64_t V) { return DAG.getConstant(V, DL, MVT::i32); };

  SDValue Sign;
  if (Signed) {
    // Convert |x| and apply the sign at the end.
    SDValue S = DAG.getNode(ISD::SRA, DL, MVT::i32, split64(DAG, DL, X).second,
                            C32(31));
    SDValue S64 = join64(DAG, DL, S, S);
    X = DAG.getNode(ISD::SUB, DL, MVT::i64,
                    DAG.getNode(ISD::XOR, DL, MVT::i64, X, S64), S64);
    Sign = DAG.getNode(ISD::AND, DL, MVT::i32, S, C32(0x80000000));
  }

  // Normalize so that the top bit is set, keep the high 32 bits with a
  // sticky bit for the rest, convert with round to nearest even and scale by
  // 2^(32 - lz). This rounds correctly since 32 bits exceed the 24 bit
  // significand by more than the two guard bits.
  SDValue LZ = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32,
                           DAG.getNode(ISD::CTLZ, DL, MVT::i64, X));
  SDValue Norm = DAG.getNode(ISD::SHL, DL, MVT::i64, X, LZ);
  auto [NLo, NHi] = split64(DAG, DL, Norm);
  SDValue Sticky = DAG.getSetCC(DL, MVT::i32, NLo, C32(0), ISD::SETNE);
  SDValue H = DAG.getNode(ISD::OR, DL, MVT::i32, NHi, Sticky);
  SDValue F = DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f32, H);
  SDValue ScaleBits = DAG.getNode(
      ISD::SHL, DL, MVT::i32,
      DAG.getNode(ISD::SUB, DL, MVT::i32, C32(127 + 32), LZ), C32(23));
  SDValue R = DAG.getNode(ISD::FMUL, DL, MVT::f32, F,
                          DAG.getBitcast(MVT::f32, ScaleBits));
  if (Signed)
    R = DAG.getBitcast(
        MVT::f32, DAG.getNode(ISD::OR, DL, MVT::i32,
                              DAG.getBitcast(MVT::i32, R), Sign));
  if (VT != MVT::f32)
    R = DAG.getNode(ISD::FP_ROUND, DL, VT, R,
                    DAG.getIntPtrConstant(0, DL, /*isTarget=*/true));
  return R;
}

SDValue AGXTargetLowering::lowerFP_TO_INT64(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  bool Signed = Op.getOpcode() == ISD::FP_TO_SINT;
  SDValue X = Op.getOperand(0);
  if (X.getValueType() != MVT::f32)
    X = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f32, X);

  SDValue T = DAG.getNode(ISD::FTRUNC, DL, MVT::f32, X);
  SDValue Sign;
  if (Signed) {
    Sign = DAG.getNode(ISD::SRA, DL, MVT::i32, DAG.getBitcast(MVT::i32, T),
                       DAG.getConstant(31, DL, MVT::i32));
    T = DAG.getNode(ISD::FABS, DL, MVT::f32, T);
  }

  // hi = floor(t * 2^-32), lo = t - hi * 2^32, both exact.
  SDValue K0 = DAG.getConstantFP(0x1p-32, DL, MVT::f32);
  SDValue K1 = DAG.getConstantFP(-0x1p+32, DL, MVT::f32);
  SDValue HiF = DAG.getNode(ISD::FFLOOR, DL, MVT::f32,
                            DAG.getNode(ISD::FMUL, DL, MVT::f32, T, K0));
  SDValue LoF = DAG.getNode(ISD::FMA, DL, MVT::f32, HiF, K1, T);
  SDValue Hi = DAG.getNode(ISD::FP_TO_UINT, DL, MVT::i32, HiF);
  SDValue Lo = DAG.getNode(ISD::FP_TO_UINT, DL, MVT::i32, LoF);
  SDValue R = join64(DAG, DL, Lo, Hi);
  if (Signed) {
    SDValue S64 = join64(DAG, DL, Sign, Sign);
    R = DAG.getNode(ISD::SUB, DL, MVT::i64,
                    DAG.getNode(ISD::XOR, DL, MVT::i64, R, S64), S64);
  }
  return R;
}
