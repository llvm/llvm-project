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
// llvm/docs/AGXUsage.rst. Calls are normally inlined. A narrow, opt-in leaf
// experiment exercises direct call/return code generation.
//
//===----------------------------------------------------------------------===//

#include "AGXISelLowering.h"
#include "AGXTargetTransformInfo.h"
#include "AGX.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXRegisterInfo.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>

using namespace llvm;

static cl::opt<bool> ExperimentalLeafCalls(
    "agx-experimental-leaf-calls", cl::Hidden, cl::init(false),
    cl::desc("Emit direct calls to argument-free void AGX leaf functions"));

bool llvm::agxExperimentalLeafCallsEnabled() { return ExperimentalLeafCalls; }

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

  // Scalar branches become structured mask operations in
  // AGXAnnotateControlFlow.
  setOperationAction(ISD::BR_CC, {MVT::i16, MVT::i32, MVT::i64, MVT::f16,
                                  MVT::f32}, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  // llvm.agx.texture.sample's <4 x i32> (ReplaceNodeResults)
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v4i32, Custom);
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
  setTargetDAGCombine({ISD::BRCOND, ISD::TRUNCATE, ISD::BUILD_PAIR, ISD::SRL,
                       ISD::ZERO_EXTEND});
  // Vectors are not legal types; 4-byte aligned loads and stores of 2 to 4
  // 32-bit elements still become one access (lowerVectorLoad/Store).
  for (MVT VT : {MVT::v2i32, MVT::v3i32, MVT::v4i32, MVT::v2f32, MVT::v3f32,
                 MVT::v4f32})
    setOperationAction({ISD::LOAD, ISD::STORE}, VT, Custom);
  // and naturally aligned device loads of 2 or 4 bytes become one access of
  // the byte format (ReplaceNodeResults)
  setOperationAction({ISD::LOAD, ISD::STORE}, {MVT::v2i8, MVT::v4i8}, Custom);
  // 4-byte aligned loads of 2 or 4 16-bit elements: one 32-bit access per
  // pair, whose halves are the elements (ReplaceNodeResults)
  setOperationAction(ISD::LOAD, {MVT::v2i16, MVT::v4i16, MVT::v2f16, MVT::v4f16},
                     Custom);

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
  // 64-bit logic and selects are done on the halves (BUILD_PAIR joins them),
  // so that constant or known halves fold.
  setOperationAction({ISD::AND, ISD::OR, ISD::XOR, ISD::SELECT}, MVT::i64,
                     Custom);
  setOperationAction(ISD::BUILD_PAIR, MVT::i64, Legal);
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
    setOperationAction({ISD::FMINIMUMNUM, ISD::FMAXIMUMNUM, ISD::FMINIMUM,
                        ISD::FMAXIMUM},
                       VT, Custom);
    setOperationAction({ISD::FSUB, ISD::SELECT_CC, ISD::FMINNUM_IEEE,
                        ISD::FMAXNUM_IEEE, ISD::FSINCOS},
                       VT, Expand);
    setOperationAction(ISD::FROUND, VT, Custom);
    // Conditions without a direct hardware condition.
    setCondCodeAction({ISD::SETONE, ISD::SETUEQ, ISD::SETO, ISD::SETUO}, VT,
                      Expand);
  }
  setOperationAction({ISD::FDIV, ISD::FREM, ISD::FSIN, ISD::FCOS, ISD::FEXP,
                      ISD::FEXP10, ISD::FLOG, ISD::FLOG10, ISD::FPOW},
                     MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f16, Custom);
  setOperationAction(ISD::FSQRT, MVT::f32, Custom);
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

  // Float-to-integer convert saturates and gives 0 for NaN (measured on G14C,
  // AGX Performance Simulator tasks/ptx-re-blockers.md, B27), so saturating
  // conversions to the convert's own width are legal; the others expand.
  setOperationAction({ISD::FP_TO_SINT_SAT, ISD::FP_TO_UINT_SAT},
                     {MVT::i16, MVT::i32}, Custom);

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

static cl::opt<unsigned> AGXAddressingModes(
    "agx-addressing-modes", cl::init(1), cl::Hidden,
    cl::desc("Addressing modes reported to IR passes (LSR): 0 the generic "
             "default, 1 the hardware's base + immediate only, 2 also base "
             "+ scaled index"));

/// The addresses loads and stores take (SelectDeviceAddr, SelectLocalAddr):
/// a device access is a 64-bit base plus either a constant offset in
/// elements (16-bit signed) or an extended 32-bit index scaled by the
/// element size times 1, 2 or 4; threadgroup and private ones a base plus
/// a constant offset.
bool AGXTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                              const AddrMode &AM, Type *Ty,
                                              unsigned AS,
                                              Instruction *I) const {
  if (AGXAddressingModes == 0)
    return TargetLowering::isLegalAddressingMode(DL, AM, Ty, AS, I);
  if (AM.BaseGV || AM.ScalableOffset)
    return false;
  uint64_t Elem = 1;
  if (Ty && Ty->isSized())
    Elem = std::max<uint64_t>(
        1, DL.getTypeStoreSize(Ty->getScalarType()).getFixedValue());
  bool Device = AS != AGX::THREADGROUP_ADDRESS && AS != AGX::PRIVATE_ADDRESS;
  if (AM.Scale == 0) {
    if (!Device)
      return isInt<16>(AM.BaseOffs);
    return AM.BaseOffs % int64_t(Elem) == 0 &&
           isInt<16>(AM.BaseOffs / int64_t(Elem));
  }
  // A register without a base: base + 0.
  if (AM.Scale == 1 && !AM.HasBaseReg)
    return Device ? AM.BaseOffs % int64_t(Elem) == 0 &&
                        isInt<16>(AM.BaseOffs / int64_t(Elem))
                  : isInt<16>(AM.BaseOffs);
  if (!Device || AGXAddressingModes < 2 || AM.BaseOffs != 0)
    return false;
  return AM.Scale == int64_t(Elem) || AM.Scale == int64_t(2 * Elem) ||
         AM.Scale == int64_t(4 * Elem);
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

static cl::opt<bool> AGXDecomposeMul(
    "agx-decompose-mul", cl::Hidden, cl::init(true),
    cl::desc("Multiply 32-bit values by 3, 5, 9 or 17 with a shifted iadd"));

bool AGXTargetLowering::decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                               SDValue C) const {
  // x * (2^k + 1), k <= 4: iadd x, x << k (an ALU instruction) instead of
  // imadd (SFU).
  auto *N = dyn_cast<ConstantSDNode>(C);
  if (!AGXDecomposeMul || VT != MVT::i32 || !N)
    return false;
  const APInt &V = N->getAPIntValue();
  return V == 3 || V == 5 || V == 9 || V == 17;
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

// Experimental device calls pass legalized values in r2..r29. Sixteen-bit
// values occupy one low half; i64 values start at an even register and occupy
// two. The same window carries return values. This is intentionally separate
// from the kernel uniform-register convention.
static MCRegister callValueReg(MVT VT, unsigned &Slot,
                               const MCRegisterInfo &MRI) {
  unsigned Words = VT == MVT::i64 ? 2 : 1;
  if (VT == MVT::i64)
    Slot = alignTo(Slot, 2);
  if (Slot + Words > 28)
    return MCRegister();
  AGX::RegKind Kind = VT == MVT::i16 || VT == MVT::f16
                          ? AGX::RegKind::GPR16
                      : VT == MVT::i32 || VT == MVT::f32
                          ? AGX::RegKind::GPR32
                      : VT == MVT::i64 ? AGX::RegKind::GPR64
                                       : AGX::RegKind::Invalid;
  if (Kind == AGX::RegKind::Invalid)
    return MCRegister();
  MCRegister Reg = AGX::getRegister(Kind, (2 + Slot) * 2, MRI);
  Slot += Words;
  return Reg;
}

template <typename Args>
static bool canUseCallRegisters(const Args &Values,
                                const MCRegisterInfo &MRI) {
  unsigned Slot = 0;
  for (const auto &Value : Values)
    if (!callValueReg(Value.VT, Slot, MRI))
      return false;
  return true;
}

SDValue AGXTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  auto *MFI = MF.getInfo<AGXMachineFunctionInfo>();
  const MCRegisterInfo &MRI = *ST.getRegisterInfo();

  if (CallConv != CallingConv::AGX_KERNEL && ExperimentalLeafCalls) {
    if (IsVarArg || !canUseCallRegisters(Ins, MRI)) {
      diagnose(DAG, DL, "experimental AGX call arguments exceed r2..r29");
      for (const ISD::InputArg &In : Ins)
        InVals.push_back(DAG.getUNDEF(In.VT));
      return Chain;
    }
    unsigned Slot = 0;
    for (const ISD::InputArg &In : Ins) {
      MCRegister PReg = callValueReg(In.VT, Slot, MRI);
      Register VReg = MF.addLiveIn(PReg, getRegClassFor(In.VT));
      InVals.push_back(DAG.getCopyFromReg(Chain, DL, VReg, In.VT));
    }
    return Chain;
  }

  if (IsVarArg)
    diagnose(DAG, DL, "variadic AGX kernels are not supported");

  // Arguments are packed into consecutive uniform registers in 16-bit units,
  // each aligned to its size (64-bit values to 32 bits), after the texture
  // heap address in a module that samples textures: texture_sample's bindless
  // base is a uniform below u64 at a multiple of four halves.
  unsigned Half = 0;
  if (AGX::usesTextureHeap(*MF.getFunction().getParent())) {
    MFI->setTextureHeap();
    Half = 4;
  }
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
  // The global data buffer address, if the kernel uses device globals (a
  // module's other globals, such as checked mode's site table, need none).
  if (AGX::usesGlobalData(MF.getFunction())) {
    unsigned GlobalHalf = alignTo(Half, 2);
    MCRegister PReg = Allocate(MVT::i64);
    if (PReg)
      MFI->setGlobalData(MF.addLiveIn(PReg, &AGX::GPR64RegClass), GlobalHalf);
  }
  // Values the launch computes from argument words (llvm.agx.launch.value).
  for (const Instruction &I : instructions(MF.getFunction())) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || II->getIntrinsicID() != Intrinsic::agx_launch_value)
      continue;
    unsigned Arg = cast<ConstantInt>(II->getArgOperand(0))->getZExtValue();
    unsigned Word = cast<ConstantInt>(II->getArgOperand(1))->getZExtValue();
    unsigned Kind = cast<ConstantInt>(II->getArgOperand(2))->getZExtValue();
    if (MFI->getLaunchValueVReg(Arg, Word, Kind))
      continue;
    Half = alignTo(Half, 2);
    unsigned At = Half;
    MCRegister PReg = Allocate(MVT::i32);
    if (PReg)
      MFI->LaunchValues.push_back(
          {Arg, Word, Kind, MF.addLiveIn(PReg, &AGX::GPR32RegClass), At});
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
  if (CallConv != CallingConv::AGX_KERNEL && ExperimentalLeafCalls) {
    const MCRegisterInfo &MRI = *ST.getRegisterInfo();
    if (!canUseCallRegisters(Outs, MRI))
      diagnose(DAG, DL, "experimental AGX call results exceed r2..r29");
    unsigned Slot = 0;
    for (unsigned I = 0; I < Outs.size(); ++I) {
      MCRegister PReg = callValueReg(Outs[I].VT, Slot, MRI);
      if (PReg)
        Chain = DAG.getCopyToReg(Chain, DL, PReg, OutVals[I]);
    }
    return DAG.getNode(AGXISD::RET, DL, MVT::Other, Chain);
  }
  if (!Outs.empty())
    diagnose(DAG, DL, "AGX kernels cannot return values");
  return DAG.getNode(AGXISD::STOP, DL, MVT::Other, Chain);
}

SDValue AGXTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  const MCRegisterInfo &MRI = *ST.getRegisterInfo();
  bool ABIShape = canUseCallRegisters(CLI.Outs, MRI) &&
                  canUseCallRegisters(CLI.Ins, MRI);
  if (ExperimentalLeafCalls && ABIShape) {
    if (auto *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee)) {
      if (const auto *F = dyn_cast<Function>(G->getGlobal())) {
        const Function &Caller = DAG.getMachineFunction().getFunction();
        bool HasCall = false;
        for (const BasicBlock &BB : Caller)
          for (const Instruction &I : BB)
            if (const auto *CB = dyn_cast<CallBase>(&I))
              if (CB->getCalledFunction() == F)
                HasCall = true;
        bool Supported = true;
        unsigned Returns = 0;
        for (const BasicBlock &BB : *F)
          for (const Instruction &I : BB)
            if (isa<ReturnInst>(I))
              ++Returns;
            else if (const auto *AI = dyn_cast<AllocaInst>(&I))
              Supported &= AI->isStaticAlloca();
            else if (const auto *CB = dyn_cast<CallBase>(&I))
              Supported &= CB->getCalledFunction() != nullptr;
        bool Signature = !F->isVarArg() && !CLI.IsVarArg &&
                         !CLI.IsTailCall &&
                         CLI.Outs.empty() == F->arg_empty() &&
                         CLI.Ins.empty() == F->getReturnType()->isVoidTy() &&
                         !F->hasStructRetAttr();
        if (!F->isDeclaration() && Returns == 1 && HasCall && Supported &&
            Signature && F->hasFnAttribute(Attribute::NoInline)) {
          SDValue Chain = CLI.Chain;
          unsigned Slot = 0;
          for (unsigned I = 0; I < CLI.OutVals.size(); ++I) {
            MCRegister PReg = callValueReg(CLI.Outs[I].VT, Slot, MRI);
            Chain = DAG.getCopyToReg(Chain, CLI.DL, PReg, CLI.OutVals[I]);
          }
          SDValue Target = DAG.getTargetGlobalAddress(F, CLI.DL, MVT::i64);
          const uint32_t *Mask = ST.getRegisterInfo()->getCallPreservedMask(
              DAG.getMachineFunction(), CallingConv::C);
          Chain = DAG.getNode(AGXISD::CALL, CLI.DL, MVT::Other, Chain, Target,
                              DAG.getRegisterMask(Mask));
          Slot = 0;
          for (const ISD::InputArg &In : CLI.Ins) {
            MCRegister PReg = callValueReg(In.VT, Slot, MRI);
            SDValue Result = DAG.getCopyFromReg(Chain, CLI.DL, PReg, In.VT);
            InVals.push_back(Result.getValue(0));
            Chain = Result.getValue(1);
          }
          return Chain;
        }
      }
    }
  }
  StringRef Name = "<indirect>";
  if (auto *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee))
    Name = G->getGlobal()->getName();
  else if (auto *E = dyn_cast<ExternalSymbolSDNode>(CLI.Callee))
    Name = E->getSymbol();
  diagnose(DAG, CLI.DL, "unsupported AGX call to '" + Name + "'");
  for (const ISD::InputArg &In : CLI.Ins)
    InVals.push_back(DAG.getUNDEF(In.VT));
  return CLI.Chain;
}

//===----------------------------------------------------------------------===//
// Custom lowering
//===----------------------------------------------------------------------===//

/// The high half of a 64-bit value.
static SDValue hi32(SelectionDAG &DAG, const SDLoc &DL, SDValue V) {
  if (auto *C = dyn_cast<ConstantSDNode>(V))
    return DAG.getConstant(C->getZExtValue() >> 32, DL, MVT::i32);
  switch (V.getOpcode()) {
  case ISD::BUILD_PAIR:
    return V.getOperand(1);
  case ISD::ZERO_EXTEND:
    return DAG.getConstant(0, DL, MVT::i32);
  case ISD::SIGN_EXTEND:
    if (V.getOperand(0).getValueType() == MVT::i32)
      return DAG.getNode(ISD::SRA, DL, MVT::i32, V.getOperand(0),
                         DAG.getConstant(31, DL, MVT::i32));
    break;
  case ISD::ANY_EXTEND:
    return DAG.getUNDEF(MVT::i32);
  default:
    break;
  }
  return DAG.getNode(AGXISD::HI32, DL, MVT::i32, V);
}

/// Low and high halves of a 64-bit value, as nodes the combiner folds.
static std::pair<SDValue, SDValue> split64(SelectionDAG &DAG, const SDLoc &DL,
                                           SDValue V) {
  if (V.getOpcode() == ISD::BUILD_PAIR)
    return {V.getOperand(0), V.getOperand(1)};
  return {DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, V), hi32(DAG, DL, V)};
}

static SDValue join64(SelectionDAG &DAG, const SDLoc &DL, SDValue Lo,
                      SDValue Hi) {
  return DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, Lo, Hi);
}

static SDValue lowerLogicSelect64(SDValue Op, SelectionDAG &DAG) {
  SDLoc DL(Op);
  unsigned Opc = Op.getOpcode();
  if (Opc == ISD::SELECT) {
    auto [XLo, XHi] = split64(DAG, DL, Op.getOperand(1));
    auto [YLo, YHi] = split64(DAG, DL, Op.getOperand(2));
    SDValue C = Op.getOperand(0);
    return join64(DAG, DL, DAG.getSelect(DL, MVT::i32, C, XLo, YLo),
                  DAG.getSelect(DL, MVT::i32, C, XHi, YHi));
  }
  auto [ALo, AHi] = split64(DAG, DL, Op.getOperand(0));
  auto [BLo, BHi] = split64(DAG, DL, Op.getOperand(1));
  return join64(DAG, DL, DAG.getNode(Opc, DL, MVT::i32, ALo, BLo),
                DAG.getNode(Opc, DL, MVT::i32, AHi, BHi));
}

SDValue AGXTargetLowering::LowerOperation(SDValue Op,
                                          SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::STORE:
    return lowerVectorStore(Op, DAG);
  case ISD::LOAD:
    // reached only for loads ReplaceNodeResults leaves as they are
    return SDValue();
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
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SELECT:
    return lowerLogicSelect64(Op, DAG);
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
  case ISD::FSQRT:
    return lowerFSQRT(Op, DAG);
  case ISD::FREM:
    return lowerFREM(Op, DAG);
  case ISD::FCOPYSIGN:
    return lowerFCOPYSIGN(Op, DAG);
  case ISD::FMINIMUMNUM:
  case ISD::FMAXIMUMNUM:
  case ISD::FMINIMUM:
  case ISD::FMAXIMUM:
    return lowerFMinMaxNum(Op, DAG);
  case ISD::FROUND:
    return lowerFROUND(Op, DAG);
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
  case ISD::FP_TO_SINT_SAT:
  case ISD::FP_TO_UINT_SAT: {
    EVT Src = Op.getOperand(0).getValueType();
    EVT Sat = cast<VTSDNode>(Op.getOperand(1))->getVT();
    if (Sat == Op.getValueType() && (Src == MVT::f32 || Src == MVT::f16))
      return Op;
    return SDValue();
  }
  default:
    llvm_unreachable("unexpected custom lowering");
  }
}

/// Whether a load or store of VT is one xy / xyz / xyzw access: 2 to 4
/// elements of 32 bits, 4-byte aligned, in device or threadgroup memory, and
/// neither volatile nor atomic (those keep the scalar path).
static bool isVectorAccess(EVT VT, const MemSDNode *M) {
  unsigned AS = M->getAddressSpace();
  return VT.isVector() && VT.getScalarSizeInBits() == 32 &&
         VT.getVectorNumElements() >= 2 && VT.getVectorNumElements() <= 4 &&
         M->isSimple() && M->getAlign() >= Align(4) &&
         (AS == AGX::DEVICE_ADDRESS || AS == AGX::THREADGROUP_ADDRESS);
}

static unsigned vectorNode(bool Load, unsigned N) {
  static const unsigned Loads[] = {AGXISD::LOAD_V2, AGXISD::LOAD_V3,
                                   AGXISD::LOAD_V4};
  static const unsigned Stores[] = {AGXISD::STORE_V2, AGXISD::STORE_V3,
                                    AGXISD::STORE_V4};
  return (Load ? Loads : Stores)[N - 2];
}

static cl::opt<bool> AGXElementAlignedVectors(
    "agx-element-aligned-vectors", cl::init(false), cl::Hidden,
    cl::desc("Vector accesses of 8- and 16-bit elements need only the "
             "element's alignment (not measured on hardware; 32-bit elements "
             "at 4 bytes are: ptx-re-blockers B33)"));

void AGXTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {
  // llvm.agx.texture.sample: one i32 result per channel.
  if (N->getOpcode() == ISD::INTRINSIC_W_CHAIN &&
      N->getConstantOperandVal(1) == Intrinsic::agx_texture_sample) {
    SDLoc DL(N);
    SDValue Sample = DAG.getNode(
        AGXISD::TEXTURE_SAMPLE, DL,
        DAG.getVTList({MVT::i32, MVT::i32, MVT::i32, MVT::i32, MVT::Other}),
        {N->getOperand(0), N->getOperand(2), N->getOperand(3), N->getOperand(4),
         N->getOperand(5),
         DAG.getTargetConstant(N->getConstantOperandVal(6), DL, MVT::i32)});
    Results.push_back(DAG.getBuildVector(
        MVT::v4i32, DL,
        {Sample.getValue(0), Sample.getValue(1), Sample.getValue(2),
         Sample.getValue(3)}));
    Results.push_back(Sample.getValue(4));
    return;
  }
  // Byte vector loads: one access of the byte format, a 16-bit result per
  // byte. The PTX alignment of ld.v2.u8 / ld.v4.u8 is the vector's size.
  if (auto *L = dyn_cast<LoadSDNode>(N);
      L && L->getExtensionType() == ISD::NON_EXTLOAD && !L->isIndexed() &&
      L->isSimple() && L->getAddressSpace() == AGX::DEVICE_ADDRESS &&
      (L->getValueType(0) == MVT::v2i8 || L->getValueType(0) == MVT::v4i8) &&
      (AGXElementAlignedVectors ||
       L->getAlign() >= Align(L->getValueType(0).getStoreSize()))) {
    SDLoc DL(N);
    EVT VT = L->getValueType(0);
    unsigned NumElts = VT.getVectorNumElements();
    SmallVector<EVT, 5> VTs(NumElts, MVT::i16);
    VTs.push_back(MVT::Other);
    SDValue Load = DAG.getMemIntrinsicNode(
        NumElts == 2 ? AGXISD::LOAD_V2_I8 : AGXISD::LOAD_V4_I8, DL,
        DAG.getVTList(VTs), {L->getChain(), L->getBasePtr()},
        L->getMemoryVT(), L->getMemOperand());
    SmallVector<SDValue, 4> Elts;
    for (unsigned I = 0; I < NumElts; ++I)
      Elts.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i8, Load.getValue(I)));
    Results.push_back(DAG.getBuildVector(VT, DL, Elts));
    Results.push_back(Load.getValue(NumElts));
    return;
  }
  // 16-bit elements below 4-byte alignment (-agx-element-aligned-vectors):
  // one device access of the i16 format into consecutive 16-bit halves.
  if (auto *L = dyn_cast<LoadSDNode>(N);
      AGXElementAlignedVectors && L &&
      L->getExtensionType() == ISD::NON_EXTLOAD && !L->isIndexed() &&
      L->isSimple() && L->getAlign() < Align(4) &&
      L->getAddressSpace() == AGX::DEVICE_ADDRESS &&
      (L->getValueType(0) == MVT::v2i16 || L->getValueType(0) == MVT::v4i16 ||
       L->getValueType(0) == MVT::v2f16 || L->getValueType(0) == MVT::v4f16)) {
    SDLoc DL(N);
    EVT VT = L->getValueType(0);
    unsigned NumElts = VT.getVectorNumElements();
    SmallVector<EVT, 5> VTs(NumElts, MVT::i16);
    VTs.push_back(MVT::Other);
    SDValue Load = DAG.getMemIntrinsicNode(
        NumElts == 2 ? AGXISD::LOAD_V2_I8 : AGXISD::LOAD_V4_I8, DL,
        DAG.getVTList(VTs), {L->getChain(), L->getBasePtr()},
        L->getMemoryVT(), L->getMemOperand());
    SmallVector<SDValue, 4> Elts;
    for (unsigned I = 0; I < NumElts; ++I)
      Elts.push_back(DAG.getBitcast(VT.getScalarType(), Load.getValue(I)));
    Results.push_back(DAG.getBuildVector(VT, DL, Elts));
    Results.push_back(Load.getValue(NumElts));
    return;
  }
  // Pairs of 16-bit elements: a 32-bit access each, its low half the first
  // element (a 16-bit half register, no instruction to split it).
  if (auto *L = dyn_cast<LoadSDNode>(N);
      L && L->getExtensionType() == ISD::NON_EXTLOAD && !L->isIndexed() &&
      L->isSimple() && L->getAlign() >= Align(4) &&
      (L->getAddressSpace() == AGX::DEVICE_ADDRESS ||
       L->getAddressSpace() == AGX::THREADGROUP_ADDRESS) &&
      (L->getValueType(0) == MVT::v2i16 || L->getValueType(0) == MVT::v4i16 ||
       L->getValueType(0) == MVT::v2f16 || L->getValueType(0) == MVT::v4f16)) {
    SDLoc DL(N);
    EVT VT = L->getValueType(0);
    unsigned Words = VT.getVectorNumElements() / 2;
    EVT WordsVT = Words == 1 ? EVT(MVT::i32) : EVT(MVT::v2i32);
    SDValue Load = DAG.getLoad(WordsVT, DL, L->getChain(), L->getBasePtr(),
                               L->getMemOperand());
    SmallVector<SDValue, 4> Elts;
    for (unsigned W = 0; W < Words; ++W) {
      SDValue Word = Words == 1 ? Load
                                : DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL,
                                              MVT::i32, Load,
                                              DAG.getVectorIdxConstant(W, DL));
      SDValue Hi = DAG.getNode(ISD::SRL, DL, MVT::i32, Word,
                               DAG.getConstant(16, DL, MVT::i32));
      for (SDValue Half : {Word, Hi})
        Elts.push_back(DAG.getBitcast(
            VT.getScalarType(), DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, Half)));
    }
    Results.push_back(DAG.getBuildVector(VT, DL, Elts));
    Results.push_back(Load.getValue(1));
    return;
  }
  // Vector loads: one access with a 32-bit result per element.
  auto *L = dyn_cast<LoadSDNode>(N);
  if (!L || L->getExtensionType() != ISD::NON_EXTLOAD || L->isIndexed() ||
      !isVectorAccess(L->getValueType(0), L))
    return;
  SDLoc DL(N);
  EVT VT = L->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();
  SmallVector<EVT, 5> VTs(NumElts, MVT::i32);
  VTs.push_back(MVT::Other);
  SDValue Load = DAG.getMemIntrinsicNode(
      vectorNode(true, NumElts), DL, DAG.getVTList(VTs),
      {L->getChain(), L->getBasePtr()}, L->getMemoryVT(), L->getMemOperand());
  SmallVector<SDValue, 4> Elts;
  for (unsigned I = 0; I < NumElts; ++I)
    Elts.push_back(DAG.getBitcast(VT.getScalarType(), Load.getValue(I)));
  Results.push_back(DAG.getBuildVector(VT, DL, Elts));
  Results.push_back(Load.getValue(NumElts));
}

/// Device and threadgroup accesses of up to 128 bits need only 4-byte
/// alignment: xy / xyzw accesses at every byte offset 0, 4, 8, 12 mod 16, also
/// across 64-byte lines and 4 KiB pages, match the byte-address model on the
/// M2 Max (ptx-re-blockers B33).
bool AGXTTIImpl::elementAlignedVectors() { return AGXElementAlignedVectors; }

bool AGXTargetLowering::allowsMisalignedMemoryAccesses(
    EVT VT, unsigned AS, Align Alignment, MachineMemOperand::Flags Flags,
    unsigned *Fast) const {
  Align Need(4);
  if (AGXElementAlignedVectors && VT.isVector() &&
      VT.getScalarSizeInBits() < 32)
    Need = Align(VT.getScalarSizeInBits() / 8);
  // The load/store vectorizer asks for its chain as one integer: with the
  // option, chains of 8- or 16-bit elements at their own alignment.
  if (AGXElementAlignedVectors && VT.isScalarInteger())
    Need = Align(1);
  bool OK = (AS == AGX::DEVICE_ADDRESS || AS == AGX::THREADGROUP_ADDRESS) &&
            Alignment >= Need && VT.getStoreSize() <= 16;
  if (Fast)
    *Fast = OK;
  return OK;
}

static cl::opt<bool> AGXByteVectorStores(
    "agx-byte-vector-stores", cl::init(true), cl::Hidden,
    cl::desc("Store 2 or 4 bytes with one device_store of the byte format"));

SDValue AGXTargetLowering::lowerVectorStore(SDValue Op,
                                            SelectionDAG &DAG) const {
  auto *S = cast<StoreSDNode>(Op);
  SDValue Val = S->getValue();
  EVT VT = Val.getValueType();
  SDLoc DL(Op);
  // 2 or 4 bytes, naturally aligned (PTX st.v2.u8 / st.v4.u8): one access of
  // the byte format from 16-bit values
  if (AGXByteVectorStores && (VT == MVT::v2i8 || VT == MVT::v4i8) &&
      !S->isTruncatingStore() &&
      !S->isIndexed() && S->isSimple() &&
      S->getAddressSpace() == AGX::DEVICE_ADDRESS &&
      S->getAlign() >= Align(VT.getStoreSize())) {
    unsigned NumElts = VT.getVectorNumElements();
    SmallVector<SDValue, 6> Ops{S->getChain()};
    for (unsigned I = 0; I < NumElts; ++I)
      Ops.push_back(DAG.getNode(
          ISD::ANY_EXTEND, DL, MVT::i16,
          DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i8, Val,
                      DAG.getVectorIdxConstant(I, DL))));
    Ops.push_back(S->getBasePtr());
    return DAG.getMemIntrinsicNode(
        NumElts == 2 ? AGXISD::STORE_V2_I8 : AGXISD::STORE_V4_I8, DL,
        DAG.getVTList(MVT::Other), Ops, S->getMemoryVT(), S->getMemOperand());
  }
  if (S->isTruncatingStore() || S->isIndexed() || !isVectorAccess(VT, S))
    return SDValue();
  SmallVector<SDValue, 6> Ops{S->getChain()};
  for (unsigned I = 0; I < VT.getVectorNumElements(); ++I)
    Ops.push_back(DAG.getBitcast(
        MVT::i32, DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT.getScalarType(),
                              Val, DAG.getVectorIdxConstant(I, DL))));
  Ops.push_back(S->getBasePtr());
  return DAG.getMemIntrinsicNode(vectorNode(false, VT.getVectorNumElements()),
                                 DL, DAG.getVTList(MVT::Other), Ops,
                                 S->getMemoryVT(), S->getMemOperand());
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
  case Intrinsic::agx_launch_value: {
    auto *MFI = DAG.getMachineFunction().getInfo<AGXMachineFunctionInfo>();
    Register VReg = MFI->getLaunchValueVReg(Op.getConstantOperandVal(1),
                                            Op.getConstantOperandVal(2),
                                            Op.getConstantOperandVal(3));
    if (!VReg)
      return DAG.getUNDEF(MVT::i32);
    return DAG.getCopyFromReg(DAG.getEntryNode(), DL, VReg, MVT::i32);
  }
  case Intrinsic::agx_sin:
  case Intrinsic::agx_cos:
  case Intrinsic::agx_exp2:
  case Intrinsic::agx_log2: {
    // the nodes of llvm.sin and the others, which the DAG does not fold
    unsigned Opc = IID == Intrinsic::agx_sin   ? ISD::FSIN
                   : IID == Intrinsic::agx_cos ? ISD::FCOS
                   : IID == Intrinsic::agx_exp2 ? ISD::FEXP2
                                                : ISD::FLOG2;
    return DAG.getNode(Opc, DL, Op.getValueType(), Op.getOperand(1));
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

/// Whether V (i16) is a boolean expression: compares, 0 / 1 constants,
/// truncations of 32-bit 0 / 1 values, and and / or / xor of those, each used
/// once.
static bool isBoolean16(SDValue V, SelectionDAG &DAG, unsigned Depth) {
  if (auto *C = dyn_cast<ConstantSDNode>(V))
    return C->getZExtValue() <= 1;
  if (Depth > 4 || !V.hasOneUse())
    return false;
  switch (V.getOpcode()) {
  case ISD::SETCC:
    return true;
  case ISD::TRUNCATE:
    // Of a value that is 0 or 1.
    return V.getOperand(0).getValueType() == MVT::i32 &&
           DAG.computeKnownBits(V.getOperand(0)).countMaxActiveBits() <= 1;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    return isBoolean16(V.getOperand(0), DAG, Depth + 1) &&
           isBoolean16(V.getOperand(1), DAG, Depth + 1);
  default:
    return false;
  }
}

/// A boolean expression accepted by isBoolean16, rebuilt in i32.
static SDValue widenBoolean(SDValue V, SelectionDAG &DAG, const SDLoc &DL) {
  if (auto *C = dyn_cast<ConstantSDNode>(V))
    return DAG.getConstant(C->getZExtValue(), DL, MVT::i32);
  if (V.getOpcode() == ISD::TRUNCATE)
    return V.getOperand(0);
  if (V.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::SETCC, DL, MVT::i32, V.getOperand(0),
                       V.getOperand(1), V.getOperand(2));
  return DAG.getNode(V.getOpcode(), DL, MVT::i32,
                     widenBoolean(V.getOperand(0), DAG, DL),
                     widenBoolean(V.getOperand(1), DAG, DL));
}

SDValue AGXTargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  case ISD::BRCOND:
    return performBRCONDCombine(N, DCI);
  case ISD::ZERO_EXTEND: {
    // Booleans (i1 promoted to i16) combined with and / or / xor and then
    // extended to 32 bits (for a branch): compute them in 32 bits, where
    // compares produce them directly, instead of a 16-bit chain and a move.
    SDValue A = N->getOperand(0);
    if (N->getValueType(0) != MVT::i32 || A.getValueType() != MVT::i16 ||
        !isBoolean16(A, DCI.DAG, 0))
      return SDValue();
    return widenBoolean(A, DCI.DAG, SDLoc(N));
  }
  case ISD::SRL: {
    // (x & (m << s)) >> s = (x >> s) & m: one bfeil (a field extract).
    SDValue A = N->getOperand(0);
    auto *S = dyn_cast<ConstantSDNode>(N->getOperand(1));
    auto *M = A.getOpcode() == ISD::AND
                  ? dyn_cast<ConstantSDNode>(A.getOperand(1))
                  : nullptr;
    if (!DCI.isAfterLegalizeDAG() || !S || !M || !A.hasOneUse())
      return SDValue();
    uint64_t Mask = M->getZExtValue(), Shift = S->getZExtValue();
    if (!isShiftedMask_64(Mask) || llvm::countr_zero(Mask) != Shift)
      return SDValue();
    SDLoc DL(N);
    EVT VT = N->getValueType(0);
    SDValue Shifted = DCI.DAG.getNode(ISD::SRL, DL, VT, A.getOperand(0),
                                      N->getOperand(1));
    return DCI.DAG.getNode(ISD::AND, DL, VT, Shifted,
                           DCI.DAG.getConstant(Mask >> Shift, DL, VT));
  }
  case ISD::TRUNCATE:
    // The low half of a pair.
    if (N->getOperand(0).getOpcode() == ISD::BUILD_PAIR) {
      SDValue Lo = N->getOperand(0).getOperand(0);
      return DCI.DAG.getZExtOrTrunc(Lo, SDLoc(N), N->getValueType(0));
    }
    return SDValue();
  case ISD::BUILD_PAIR: {
    // A value split and joined again.
    SDValue Lo = N->getOperand(0), Hi = N->getOperand(1);
    if (Lo.getOpcode() == ISD::TRUNCATE && Hi.getOpcode() == AGXISD::HI32 &&
        Lo.getOperand(0) == Hi.getOperand(0))
      return Lo.getOperand(0);
    return SDValue();
  }
  case AGXISD::IF:
  case AGXISD::LOOP: {
    // On (b ^ 1) for a boolean b: on b == 0, which the branch compares
    // itself (AGXFoldOperands), instead of an xor and a compare with zero.
    // Likewise on a ^ b, a != b, and on (a ^ b) == 0 or != 0, a == b or
    // a != b: one compare of the two instead of an xor and a compare.
    SDValue Cond = N->getOperand(1);
    SelectionDAG &DAG = DCI.DAG;
    SDLoc DL(N);
    auto Branch = [&](SDValue C) {
      return DAG.getNode(N->getOpcode(), DL, MVT::Other, N->getOperand(0), C,
                         N->getOperand(2));
    };
    auto IsBool = [&](SDValue V) {
      return DAG.computeKnownBits(V).countMaxActiveBits() <= 1;
    };
    if (Cond.getOpcode() == ISD::XOR && Cond.hasOneUse()) {
      SDValue A = Cond.getOperand(0), B = Cond.getOperand(1);
      if (isOneConstant(B) && IsBool(A))
        return Branch(DAG.getSetCC(DL, MVT::i32, A,
                                   DAG.getConstant(0, DL, MVT::i32),
                                   ISD::SETEQ));
      if (!isa<ConstantSDNode>(B) && IsBool(A) && IsBool(B))
        return Branch(DAG.getSetCC(DL, MVT::i32, A, B, ISD::SETNE));
      return SDValue();
    }
    if (Cond.getOpcode() == ISD::SETCC && Cond.hasOneUse() &&
        isNullConstant(Cond.getOperand(1)) &&
        Cond.getOperand(0).getOpcode() == ISD::XOR &&
        Cond.getOperand(0).hasOneUse()) {
      ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
      SDValue X = Cond.getOperand(0);
      if ((CC == ISD::SETEQ || CC == ISD::SETNE) &&
          X.getValueType() == MVT::i32)
        return Branch(DAG.getSetCC(DL, Cond.getValueType(), X.getOperand(0),
                                   X.getOperand(1), CC));
    }
    return SDValue();
  }
  case AGXISD::BFI7:
  case AGXISD::EXTR7:
  case AGXISD::SRA7:
    // Shifts of zero.
    if (all_of(drop_end(N->ops()), [](const SDUse &U) { return isNullConstant(U.get()); }))
      return DCI.DAG.getConstant(0, SDLoc(N), MVT::i32);
    return SDValue();
  case AGXISD::HI32: {
    SDValue V = N->getOperand(0);
    SDLoc DL(N);
    SDValue H = hi32(DCI.DAG, DL, V);
    if (H.getOpcode() != AGXISD::HI32)
      return H;
    KnownBits Known = DCI.DAG.computeKnownBits(SDValue(N, 0));
    if (Known.isConstant())
      return DCI.DAG.getConstant(Known.getConstant(), DL, MVT::i32);
    return SDValue();
  }
  default:
    return SDValue();
  }
}

void AGXTargetLowering::computeKnownBitsForTargetNode(
    const SDValue Op, KnownBits &Known, const APInt &DemandedElts,
    const SelectionDAG &DAG, unsigned Depth) const {
  Known.resetAll();
  switch (Op.getOpcode()) {
  case AGXISD::HI32:
    Known = DAG.computeKnownBits(Op.getOperand(0), Depth + 1).extractBits(32, 32);
    break;
  case AGXISD::MUL_WIDE_U:
  case AGXISD::MUL_WIDE_S: {
    bool S = Op.getOpcode() == AGXISD::MUL_WIDE_S;
    KnownBits A = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    KnownBits B = DAG.computeKnownBits(Op.getOperand(1), Depth + 1);
    Known = KnownBits::mul(S ? A.sext(64) : A.zext(64), S ? B.sext(64) : B.zext(64));
    break;
  }
  default:
    break;
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

  if (auto *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
    uint64_t S = C->getZExtValue();
    // Small left shifts stay for address mode matching (iadd with a shift);
    // instruction selection expands what is left.
    SDValue X = Op.getOperand(0);
    bool Address =
        X.getOpcode() == ISD::ZERO_EXTEND || X.getOpcode() == ISD::SIGN_EXTEND ||
        all_of(Op->users(), [](SDNode *U) { return U->getOpcode() == ISD::ADD; });
    if (S >= 64 || (Opc == ISD::SHL && S <= 4 && Address))
      return Op;
    if (S == 0)
      return X;
    SDValue NewLo, NewHi;
    if (S >= 32) {
      if (Opc == ISD::SHL) {
        NewLo = Zero;
        NewHi = DAG.getNode(ISD::SHL, DL, MVT::i32, Lo, C32(S - 32));
      } else {
        NewLo = DAG.getNode(Opc, DL, MVT::i32, Hi, C32(S - 32));
        NewHi = Opc == ISD::SRA ? DAG.getNode(ISD::SRA, DL, MVT::i32, Hi, C32(31))
                                : Zero;
      }
    } else if (Opc == ISD::SHL) {
      NewLo = DAG.getNode(ISD::SHL, DL, MVT::i32, Lo, C32(S));
      NewHi = DAG.getNode(ISD::FSHR, DL, MVT::i32, Hi, Lo, C32(32 - S));
    } else {
      NewLo = DAG.getNode(ISD::FSHR, DL, MVT::i32, Hi, Lo, C32(S));
      NewHi = DAG.getNode(Opc, DL, MVT::i32, Hi, C32(S));
    }
    return join64(DAG, DL, NewLo, NewHi);
  }

  // Variable shifts (by 0 to 63) with the 7-bit shift amounts of bfi, extr
  // and asr, which shift a register's bits out completely from 32 on:
  //   srl: lo = extr lo, hi, s            hi = extr hi, 0, s
  //   shl: lo = bfi 0, lo, s              hi = bfi (extr 0, lo, 64 - s), hi, s
  //   sra: lo = s < 32 ? extr lo, hi, s : asr hi, s - 32      hi = asr hi, s
  // In shl, extr 0, lo, 64 - s is the part of lo that moves into hi, for s
  // below and above 32 (and 0 for s = 0).
  SDValue NewLo, NewHi;
  if (Opc == ISD::SRL) {
    NewLo = DAG.getNode(AGXISD::EXTR7, DL, MVT::i32, Lo, Hi, Amt);
    NewHi = DAG.getNode(AGXISD::EXTR7, DL, MVT::i32, Hi, Zero, Amt);
  } else if (Opc == ISD::SHL) {
    NewLo = DAG.getNode(AGXISD::BFI7, DL, MVT::i32, Zero, Lo, Amt);
    SDValue Up = DAG.getNode(AGXISD::EXTR7, DL, MVT::i32, Zero, Lo,
                             DAG.getNode(ISD::SUB, DL, MVT::i32, C32(64), Amt));
    NewHi = DAG.getNode(AGXISD::BFI7, DL, MVT::i32, Up, Hi, Amt);
  } else {
    NewHi = DAG.getNode(AGXISD::SRA7, DL, MVT::i32, Hi, Amt);
    SDValue Big = DAG.getSetCC(DL, MVT::i32, Amt, C32(31), ISD::SETUGT);
    NewLo = DAG.getSelect(
        DL, MVT::i32, Big,
        DAG.getNode(AGXISD::SRA7, DL, MVT::i32, Hi,
                    DAG.getNode(ISD::SUB, DL, MVT::i32, Amt, C32(32))),
        DAG.getNode(AGXISD::EXTR7, DL, MVT::i32, Lo, Hi, Amt));
  }
  return join64(DAG, DL, NewLo, NewHi);
}

SDValue AGXTargetLowering::lowerMUL64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  auto [ALo, AHi] = split64(DAG, DL, A);
  auto [BLo, BHi] = split64(DAG, DL, B);
  // 32 x 32 -> 64-bit products are a single imadd.
  if (DAG.computeKnownBits(A).countMinLeadingZeros() >= 32 &&
      DAG.computeKnownBits(B).countMinLeadingZeros() >= 32)
    return DAG.getNode(AGXISD::MUL_WIDE_U, DL, MVT::i64, ALo, BLo);
  if (DAG.ComputeNumSignBits(A) > 32 && DAG.ComputeNumSignBits(B) > 32)
    return DAG.getNode(AGXISD::MUL_WIDE_S, DL, MVT::i64, ALo, BLo);

  // lo(a) * lo(b) as a 64-bit product, plus the cross products in the high
  // half.
  SDValue P = DAG.getNode(AGXISD::MUL_WIDE_U, DL, MVT::i64, ALo, BLo);
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

/// float division without arcp / afn, correctly rounded (with denormal
/// operands and results flushed, as the ALU does). The hardware's rcp is
/// within 1 ulp; the quotient of the significands, a1 in [1, 2) and d in
/// [2, 4) with the signs of a and b, takes the correctly rounded reciprocal
///   y = y0 + y0 (1 - d y0), y0 = rcp(d)
/// (correctly rounded for every significand with the recorded rcp table) and
/// one correction of q0 = a1 y:
///   q = q0 + (a1 - d q0) y,
/// correctly rounded by Markstein's theorem (y is the rounded 1 / d, q0
/// within 1 ulp, the remainder exact). The exponents come back as factors
///   pa = 2^ea, pb = 2^(128 - eb),
/// 0 / inf for a zero (or denormal) / infinite a, and inf / 0 for b, so that
/// zeros and infinities need no other test; the product is exact or rounds
/// once into the flushed or infinite range. pa first could flush a tiny a's
/// intermediate, pb first a huge b's, so a below 2^-124 takes pb first (both
/// at once flush the result anyway). NaNs are selected at the end.
SDValue AGXTargetLowering::lowerFDIV32(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const MVT VT = MVT::f32;
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  auto C = [&](uint32_t V) { return DAG.getConstant(V, DL, MVT::i32); };
  auto K = [&](float V) { return DAG.getConstantFP(V, DL, VT); };
  auto Bits = [&](SDValue X) { return DAG.getBitcast(MVT::i32, X); };
  auto Flt = [&](SDValue X) { return DAG.getBitcast(VT, X); };
  auto And = [&](SDValue X, uint32_t M) {
    return DAG.getNode(ISD::AND, DL, MVT::i32, X, C(M));
  };
  auto Fma = [&](SDValue X, SDValue Y, SDValue Z) {
    return DAG.getNode(ISD::FMA, DL, VT, X, Y, Z);
  };
  auto Mul = [&](SDValue X, SDValue Y) {
    return DAG.getNode(ISD::FMUL, DL, VT, X, Y);
  };
  EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);

  SDValue DB = Bits(B);
  SDValue D = Flt(DAG.getNode(ISD::OR, DL, MVT::i32, And(DB, 0x807fffff),
                              C(0x40000000)));
  SDValue NegD = DAG.getNode(ISD::FNEG, DL, VT, D);
  SDValue Y0 = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rcp, VT, D);
  SDValue Y = Fma(Fma(NegD, Y0, K(1.0f)), Y0, Y0);
  SDValue PB = Flt(DAG.getNode(ISD::SUB, DL, MVT::i32, C(0x7f800000),
                               And(DB, 0x7f800000)));
  SDValue R;
  SmallVector<SDValue, 2> NaNOperands = {B};
  auto *CA = dyn_cast<ConstantFPSDNode>(A);
  if (CA && CA->isExactlyValue(1.0)) {
    // 1 / b: y is already the rounded quotient.
    R = Mul(Y, PB);
  } else {
    SDValue AB = Bits(A);
    SDValue A1 = Flt(DAG.getNode(ISD::OR, DL, MVT::i32, And(AB, 0x807fffff),
                                 C(0x3f800000)));
    SDValue Q0 = Mul(A1, Y);
    SDValue Q = Fma(Fma(NegD, Q0, A1), Y, Q0);
    SDValue PA = Flt(And(AB, 0x7f800000));
    SDValue Small =
        DAG.getSetCC(DL, CCVT, DAG.getNode(ISD::FABS, DL, VT, A), K(0x1p-124f),
                     ISD::SETOLT);
    R = Mul(Mul(Q, DAG.getSelect(DL, VT, Small, PB, PA)),
            DAG.getSelect(DL, VT, Small, PA, PB));
    NaNOperands.push_back(A);
  }
  if (Op->getFlags().hasNoNaNs())
    return R;
  // One compare-and-select per operand.
  SDValue NaN =
      DAG.getConstantFP(APFloat::getQNaN(APFloat::IEEEsingle()), DL, VT);
  for (SDValue X : NaNOperands)
    R = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, X, X, ISD::SETUO), NaN,
                      R);
  return R;
}

SDValue AGXTargetLowering::lowerFDIV(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDNodeFlags Flags = Op->getFlags();
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  bool Approx = Flags.hasAllowReciprocal() || Flags.hasApproximateFuncs();
  if (VT == MVT::f32 && !Approx)
    return lowerFDIV32(Op, DAG);
  SDValue R = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rcp, VT, B);
  auto *CA = dyn_cast<ConstantFPSDNode>(A);
  if (CA && CA->isExactlyValue(1.0))
    return R;
  return DAG.getNode(ISD::FMUL, DL, VT, A, R, Flags);
}

/// float sqrt without afn, correctly rounded (denormals flushed): from
/// y = rsqrt_special(x) (within 1 ulp; 1 for 0 and infinity),
///   s0 = x y, s = s0 + (x - s0^2) y / 2
/// is correctly rounded for every x in [1, 4) with the recorded table, and
/// so for every x whose remainder x - s0^2 is not flushed: x below 2^-80 is
/// scaled by 2^64 first. +-0 gives +-0, and the NaN of +inf's remainder
/// (s0 = inf), negative values and NaNs are s0.
SDValue AGXTargetLowering::lowerFSQRT(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getValueType() != MVT::f32 || Op->getFlags().hasApproximateFuncs())
    return Op;
  SDLoc DL(Op);
  const MVT VT = MVT::f32;
  SDValue X = Op.getOperand(0);
  auto K = [&](float V) { return DAG.getConstantFP(V, DL, VT); };
  EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  SDValue Small = DAG.getSetCC(DL, CCVT, X, K(0x1p-80f), ISD::SETOLT);
  SDValue XS = DAG.getSelect(
      DL, VT, Small, DAG.getNode(ISD::FMUL, DL, VT, X, K(0x1p64f)), X);
  SDValue Y = getAGXIntrinsic(DAG, DL, Intrinsic::agx_rsqrt_special, VT, XS);
  SDValue S0 = DAG.getNode(ISD::FMUL, DL, VT, XS, Y);
  SDValue H = DAG.getNode(ISD::FMUL, DL, VT, Y, K(0.5f));
  SDValue Rem = DAG.getNode(ISD::FMA, DL, VT,
                            DAG.getNode(ISD::FNEG, DL, VT, S0), S0, XS);
  SDValue S = DAG.getNode(ISD::FMA, DL, VT, Rem, H, S0);
  S = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, S, S, ISD::SETUO), S0, S);
  return DAG.getSelect(DL, VT, Small,
                       DAG.getNode(ISD::FMUL, DL, VT, S, K(0x1p-32f)), S);
}

/// copysign(m, s) = (m & ~sign) | (s & sign), on the bits.
/// minimumnum / maximumnum: the hardware's NaN-ignoring min / max (fminnum /
/// fmaxnum, one fcmpsel with an "n" condition), except for operands that
/// compare equal: there the bits of both ORed (min) or ANDed (max), which
/// orders -0 below +0 and leaves equal nonzero values as they are.
/// minimum / maximum: the same, and the default NaN for a NaN operand.
SDValue AGXTargetLowering::lowerFMinMaxNum(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue A = Op.getOperand(0), B = Op.getOperand(1);
  unsigned Opc = Op.getOpcode();
  bool Max = Opc == ISD::FMAXIMUMNUM || Opc == ISD::FMAXIMUM;
  bool PropagateNaN = Opc == ISD::FMINIMUM || Opc == ISD::FMAXIMUM;
  EVT CCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  SDValue R = DAG.getNode(Max ? ISD::FMAXNUM : ISD::FMINNUM, DL, VT, A, B,
                          Op->getFlags());
  if (!Op->getFlags().hasNoSignedZeros()) {
    EVT IntVT = VT.changeTypeToInteger();
    SDValue Bits = DAG.getNode(Max ? ISD::AND : ISD::OR, DL, IntVT,
                               DAG.getBitcast(IntVT, A),
                               DAG.getBitcast(IntVT, B));
    SDValue Equal = DAG.getSetCC(DL, CCVT, A, B, ISD::SETOEQ);
    R = DAG.getSelect(DL, VT, Equal, DAG.getBitcast(VT, Bits), R);
  }
  if (!PropagateNaN || Op->getFlags().hasNoNaNs())
    return R;
  // minimum / maximum: the default NaN when either operand is a NaN.
  SDValue NaN = DAG.getConstantFP(
      APFloat::getQNaN(VT.getFltSemantics()), DL, VT);
  for (SDValue X : {B, A})
    R = DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, X, X, ISD::SETOEQ), R,
                      NaN);
  return R;
}

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

/// round(x), half away from zero (no libm on AGX): t = trunc(x), plus one
/// away from zero when |x - t| >= 0.5, with the sign of x (round(-0.3) and
/// round(-0) are -0). x - t is exact, and so is t + 1 for every x that is not
/// an integer; infinities give x - t = NaN, so t, and NaN stays NaN.
SDValue AGXTargetLowering::lowerFROUND(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue X = Op.getOperand(0);
  SDValue T = DAG.getNode(ISD::FTRUNC, DL, VT, X);
  SDValue D =
      DAG.getNode(ISD::FABS, DL, VT, DAG.getNode(ISD::FSUB, DL, VT, X, T));
  SDValue One = DAG.getNode(ISD::FCOPYSIGN, DL, VT,
                            DAG.getConstantFP(1.0, DL, VT), X);
  SDValue Step = DAG.getSelectCC(DL, D, DAG.getConstantFP(0.5, DL, VT), One,
                                 DAG.getConstantFP(0.0, DL, VT), ISD::SETOGE);
  return DAG.getNode(ISD::FCOPYSIGN, DL, VT,
                     DAG.getNode(ISD::FADD, DL, VT, T, Step), X);
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
