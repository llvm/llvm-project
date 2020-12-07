//===-- VEISelLowering.cpp - VE DAG Lowering Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interfaces that VE uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "VEISelLowering.h"
#include "MCTargetDesc/VEMCExpr.h"
#include "VEMachineFunctionInfo.h"
#include "VERegisterInfo.h"
#include "VETargetMachine.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
using namespace llvm;

#define DEBUG_TYPE "ve-lower"

//===----------------------------------------------------------------------===//
// Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "VEGenCallingConv.inc"

CCAssignFn *getReturnCC(CallingConv::ID CallConv) {
  switch (CallConv) {
  default:
    return RetCC_VE_C;
  case CallingConv::Fast:
    return RetCC_VE_Fast;
  }
}

CCAssignFn *getParamCC(CallingConv::ID CallConv, bool IsVarArg) {
  if (IsVarArg)
    return CC_VE2;
  switch (CallConv) {
  default:
    return CC_VE_C;
  case CallingConv::Fast:
    return CC_VE_Fast;
  }
}

bool VETargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  CCAssignFn *RetCC = getReturnCC(CallConv);
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC);
}

static const MVT AllVectorVTs[] = {MVT::v256i32, MVT::v512i32, MVT::v256i64,
                                   MVT::v256f32, MVT::v512f32, MVT::v256f64};

static const MVT AllMaskVTs[] = {MVT::v256i1, MVT::v512i1};

void VETargetLowering::initRegisterClasses() {
  // Set up the register classes.
  addRegisterClass(MVT::i32, &VE::I32RegClass);
  addRegisterClass(MVT::i64, &VE::I64RegClass);
  addRegisterClass(MVT::f32, &VE::F32RegClass);
  addRegisterClass(MVT::f64, &VE::I64RegClass);
  addRegisterClass(MVT::f128, &VE::F128RegClass);

  if (Subtarget->enableVPU()) {
    for (MVT VecVT : AllVectorVTs)
      addRegisterClass(VecVT, &VE::V64RegClass);
    for (MVT MaskVT : AllMaskVTs)
      addRegisterClass(MaskVT, &VE::VMRegClass);
  }
}

void VETargetLowering::initSPUActions() {
  const auto &TM = getTargetMachine();
  /// Load & Store {

  // VE doesn't have i1 sign extending load.
  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setTruncStoreAction(VT, MVT::i1, Expand);
  }

  // VE doesn't have floating point extload/truncstore, so expand them.
  for (MVT FPVT : MVT::fp_valuetypes()) {
    for (MVT OtherFPVT : MVT::fp_valuetypes()) {
      setLoadExtAction(ISD::EXTLOAD, FPVT, OtherFPVT, Expand);
      setTruncStoreAction(FPVT, OtherFPVT, Expand);
    }
  }

  // VE doesn't have fp128 load/store, so expand them in custom lower.
  setOperationAction(ISD::LOAD, MVT::f128, Custom);
  setOperationAction(ISD::STORE, MVT::f128, Custom);

  /// } Load & Store

  // Custom legalize address nodes into LO/HI parts.
  MVT PtrVT = MVT::getIntegerVT(TM.getPointerSizeInBits(0));
  setOperationAction(ISD::BlockAddress, PtrVT, Custom);
  setOperationAction(ISD::GlobalAddress, PtrVT, Custom);
  setOperationAction(ISD::GlobalTLSAddress, PtrVT, Custom);
  setOperationAction(ISD::ConstantPool, PtrVT, Custom);
  setOperationAction(ISD::JumpTable, PtrVT, Custom);

  /// VAARG handling {
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  // VAARG needs to be lowered to access with 8 bytes alignment.
  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  // Use the default implementation.
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);
  /// } VAARG handling

  /// Stack {
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Custom);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Custom);
  /// } Stack

  /// Branch {

  // VE doesn't have BRCOND
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);

  // BR_JT is not implemented yet.
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);

  /// } Branch

  /// Int Ops {
  for (MVT IntVT : {MVT::i32, MVT::i64}) {
    // VE has no REM or DIVREM operations.
    setOperationAction(ISD::UREM, IntVT, Expand);
    setOperationAction(ISD::SREM, IntVT, Expand);
    setOperationAction(ISD::SDIVREM, IntVT, Expand);
    setOperationAction(ISD::UDIVREM, IntVT, Expand);

    // VE has no SHL_PARTS/SRA_PARTS/SRL_PARTS operations.
    setOperationAction(ISD::SHL_PARTS, IntVT, Expand);
    setOperationAction(ISD::SRA_PARTS, IntVT, Expand);
    setOperationAction(ISD::SRL_PARTS, IntVT, Expand);

    // VE has no MULHU/S or U/SMUL_LOHI operations.
    // TODO: Use MPD instruction to implement SMUL_LOHI for i32 type.
    setOperationAction(ISD::MULHU, IntVT, Expand);
    setOperationAction(ISD::MULHS, IntVT, Expand);
    setOperationAction(ISD::UMUL_LOHI, IntVT, Expand);
    setOperationAction(ISD::SMUL_LOHI, IntVT, Expand);

    // VE has no CTTZ, ROTL, ROTR operations.
    setOperationAction(ISD::CTTZ, IntVT, Expand);
    setOperationAction(ISD::ROTL, IntVT, Expand);
    setOperationAction(ISD::ROTR, IntVT, Expand);

    // VE has 64 bits instruction which works as i64 BSWAP operation.  This
    // instruction works fine as i32 BSWAP operation with an additional
    // parameter.  Use isel patterns to lower BSWAP.
    setOperationAction(ISD::BSWAP, IntVT, Legal);

    // VE has only 64 bits instructions which work as i64 BITREVERSE/CTLZ/CTPOP
    // operations.  Use isel patterns for i64, promote for i32.
    LegalizeAction Act = (IntVT == MVT::i32) ? Promote : Legal;
    setOperationAction(ISD::BITREVERSE, IntVT, Act);
    setOperationAction(ISD::CTLZ, IntVT, Act);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, IntVT, Act);
    setOperationAction(ISD::CTPOP, IntVT, Act);

    // VE has only 64 bits instructions which work as i64 AND/OR/XOR operations.
    // Use isel patterns for i64, promote for i32.
    setOperationAction(ISD::AND, IntVT, Act);
    setOperationAction(ISD::OR, IntVT, Act);
    setOperationAction(ISD::XOR, IntVT, Act);
  }
  /// } Int Ops

  /// Conversion {
  // VE doesn't have instructions for fp<->uint, so expand them by llvm
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Promote); // use i64
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Promote); // use i64
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);

  // fp16 not supported
  for (MVT FPVT : MVT::fp_valuetypes()) {
    setOperationAction(ISD::FP16_TO_FP, FPVT, Expand);
    setOperationAction(ISD::FP_TO_FP16, FPVT, Expand);
  }
  /// } Conversion

  /// Floating-point Ops {
  /// Note: Floating-point operations are fneg, fadd, fsub, fmul, fdiv, frem,
  ///       and fcmp.

  // VE doesn't have following floating point operations.
  for (MVT VT : MVT::fp_valuetypes()) {
    setOperationAction(ISD::FNEG, VT, Expand);
    setOperationAction(ISD::FREM, VT, Expand);
  }

  // VE doesn't have fdiv of f128.
  setOperationAction(ISD::FDIV, MVT::f128, Expand);

  for (MVT FPVT : {MVT::f32, MVT::f64}) {
    // f32 and f64 uses ConstantFP.  f128 uses ConstantPool.
    setOperationAction(ISD::ConstantFP, FPVT, Legal);
  }
  /// } Floating-point Ops

  /// Floating-point math functions {

  // VE doesn't have following floating point math functions.
  for (MVT VT : MVT::fp_valuetypes()) {
    setOperationAction(ISD::FABS, VT, Expand);
    setOperationAction(ISD::FCOPYSIGN, VT, Expand);
    setOperationAction(ISD::FCOS, VT, Expand);
    setOperationAction(ISD::FSIN, VT, Expand);
    setOperationAction(ISD::FSQRT, VT, Expand);
  }

  /// } Floating-point math functions

  /// Atomic instructions {

  setMaxAtomicSizeInBitsSupported(64);
  setMinCmpXchgSizeInBits(32);
  setSupportsUnalignedAtomics(false);

  // Use custom inserter for ATOMIC_FENCE.
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);

  /// } Atomic isntructions
}

void VETargetLowering::initVPUActions() {
  for (MVT LegalVecVT : AllVectorVTs) {
    setOperationAction(ISD::BUILD_VECTOR, LegalVecVT, Custom);
    // Translate all vector instructions with legal element types to VVP_*
    // nodes.
    // TODO We will custom-widen into VVP_* nodes in the future. While we are
    // buildling the infrastructure for this, we only do this for legal vector
    // VTs.
#define ADD_VVP_OP(VVP_NAME, ISD_NAME)                                         \
  setOperationAction(ISD::ISD_NAME, LegalVecVT, Custom);
#include "VVPNodes.def"
  }
}

SDValue
VETargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool IsVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals,
                              const SDLoc &DL, SelectionDAG &DAG) const {
  // CCValAssign - represent the assignment of the return value to locations.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  // Analyze return values.
  CCInfo.AnalyzeReturn(Outs, getReturnCC(CallConv));

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    SDValue OutVal = OutVals[i];

    // Integer return values must be sign or zero extended by the callee.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      OutVal = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), OutVal);
      break;
    case CCValAssign::ZExt:
      OutVal = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), OutVal);
      break;
    case CCValAssign::AExt:
      OutVal = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), OutVal);
      break;
    case CCValAssign::BCvt: {
      // Convert a float return value to i64 with padding.
      //     63     31   0
      //    +------+------+
      //    | float|   0  |
      //    +------+------+
      assert(VA.getLocVT() == MVT::i64);
      assert(VA.getValVT() == MVT::f32);
      SDValue Undef = SDValue(
          DAG.getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, MVT::i64), 0);
      SDValue Sub_f32 = DAG.getTargetConstant(VE::sub_f32, DL, MVT::i32);
      OutVal = SDValue(DAG.getMachineNode(TargetOpcode::INSERT_SUBREG, DL,
                                          MVT::i64, Undef, OutVal, Sub_f32),
                       0);
      break;
    }
    default:
      llvm_unreachable("Unknown loc info!");
    }

    assert(!VA.needsCustom() && "Unexpected custom lowering");

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVal, Flag);

    // Guarantee that all emitted copies are stuck together with flags.
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain; // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(VEISD::RET_FLAG, DL, MVT::Other, RetOps);
}

SDValue VETargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();

  // Get the base offset of the incoming arguments stack space.
  unsigned ArgsBaseOffset = Subtarget->getRsaSize();
  // Get the size of the preserved arguments area
  unsigned ArgsPreserved = 64;

  // Analyze arguments according to CC_VE.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());
  // Allocate the preserved area first.
  CCInfo.AllocateStack(ArgsPreserved, Align(8));
  // We already allocated the preserved area, so the stack offset computed
  // by CC_VE would be correct now.
  CCInfo.AnalyzeFormalArguments(Ins, getParamCC(CallConv, false));

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    if (VA.isRegLoc()) {
      // This argument is passed in a register.
      // All integer register arguments are promoted by the caller to i64.

      // Create a virtual register for the promoted live-in value.
      unsigned VReg =
          MF.addLiveIn(VA.getLocReg(), getRegClassFor(VA.getLocVT()));
      SDValue Arg = DAG.getCopyFromReg(Chain, DL, VReg, VA.getLocVT());

      // Get the high bits for i32 struct elements.
      if (VA.getValVT() == MVT::i32 && VA.needsCustom())
        Arg = DAG.getNode(ISD::SRL, DL, VA.getLocVT(), Arg,
                          DAG.getConstant(32, DL, MVT::i32));

      // The caller promoted the argument, so insert an Assert?ext SDNode so we
      // won't promote the value again in this function.
      switch (VA.getLocInfo()) {
      case CCValAssign::SExt:
        Arg = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), Arg,
                          DAG.getValueType(VA.getValVT()));
        break;
      case CCValAssign::ZExt:
        Arg = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), Arg,
                          DAG.getValueType(VA.getValVT()));
        break;
      case CCValAssign::BCvt: {
        // Extract a float argument from i64 with padding.
        //     63     31   0
        //    +------+------+
        //    | float|   0  |
        //    +------+------+
        assert(VA.getLocVT() == MVT::i64);
        assert(VA.getValVT() == MVT::f32);
        SDValue Sub_f32 = DAG.getTargetConstant(VE::sub_f32, DL, MVT::i32);
        Arg = SDValue(DAG.getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL,
                                         MVT::f32, Arg, Sub_f32),
                      0);
        break;
      }
      default:
        break;
      }

      // Truncate the register down to the argument type.
      if (VA.isExtInLoc())
        Arg = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Arg);

      InVals.push_back(Arg);
      continue;
    }

    // The registers are exhausted. This argument was passed on the stack.
    assert(VA.isMemLoc());
    // The CC_VE_Full/Half functions compute stack offsets relative to the
    // beginning of the arguments area at %fp + the size of reserved area.
    unsigned Offset = VA.getLocMemOffset() + ArgsBaseOffset;
    unsigned ValSize = VA.getValVT().getSizeInBits() / 8;

    // Adjust offset for a float argument by adding 4 since the argument is
    // stored in 8 bytes buffer with offset like below.  LLVM generates
    // 4 bytes load instruction, so need to adjust offset here.  This
    // adjustment is required in only LowerFormalArguments.  In LowerCall,
    // a float argument is converted to i64 first, and stored as 8 bytes
    // data, which is required by ABI, so no need for adjustment.
    //    0      4
    //    +------+------+
    //    | empty| float|
    //    +------+------+
    if (VA.getValVT() == MVT::f32)
      Offset += 4;

    int FI = MF.getFrameInfo().CreateFixedObject(ValSize, Offset, true);
    InVals.push_back(
        DAG.getLoad(VA.getValVT(), DL, Chain,
                    DAG.getFrameIndex(FI, getPointerTy(MF.getDataLayout())),
                    MachinePointerInfo::getFixedStack(MF, FI)));
  }

  if (!IsVarArg)
    return Chain;

  // This function takes variable arguments, some of which may have been passed
  // in registers %s0-%s8.
  //
  // The va_start intrinsic needs to know the offset to the first variable
  // argument.
  // TODO: need to calculate offset correctly once we support f128.
  unsigned ArgOffset = ArgLocs.size() * 8;
  VEMachineFunctionInfo *FuncInfo = MF.getInfo<VEMachineFunctionInfo>();
  // Skip the reserved area at the top of stack.
  FuncInfo->setVarArgsFrameOffset(ArgOffset + ArgsBaseOffset);

  return Chain;
}

// FIXME? Maybe this could be a TableGen attribute on some registers and
// this table could be generated automatically from RegInfo.
Register VETargetLowering::getRegisterByName(const char *RegName, LLT VT,
                                             const MachineFunction &MF) const {
  Register Reg = StringSwitch<Register>(RegName)
                     .Case("sp", VE::SX11)    // Stack pointer
                     .Case("fp", VE::SX9)     // Frame pointer
                     .Case("sl", VE::SX8)     // Stack limit
                     .Case("lr", VE::SX10)    // Link register
                     .Case("tp", VE::SX14)    // Thread pointer
                     .Case("outer", VE::SX12) // Outer regiser
                     .Case("info", VE::SX17)  // Info area register
                     .Case("got", VE::SX15)   // Global offset table register
                     .Case("plt", VE::SX16) // Procedure linkage table register
                     .Default(0);

  if (Reg)
    return Reg;

  report_fatal_error("Invalid register name global variable");
}

//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

SDValue VETargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                    SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc DL = CLI.DL;
  SDValue Chain = CLI.Chain;
  auto PtrVT = getPointerTy(DAG.getDataLayout());

  // VE target does not yet support tail call optimization.
  CLI.IsTailCall = false;

  // Get the base offset of the outgoing arguments stack space.
  unsigned ArgsBaseOffset = Subtarget->getRsaSize();
  // Get the size of the preserved arguments area
  unsigned ArgsPreserved = 8 * 8u;

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CLI.CallConv, CLI.IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());
  // Allocate the preserved area first.
  CCInfo.AllocateStack(ArgsPreserved, Align(8));
  // We already allocated the preserved area, so the stack offset computed
  // by CC_VE would be correct now.
  CCInfo.AnalyzeCallOperands(CLI.Outs, getParamCC(CLI.CallConv, false));

  // VE requires to use both register and stack for varargs or no-prototyped
  // functions.
  bool UseBoth = CLI.IsVarArg;

  // Analyze operands again if it is required to store BOTH.
  SmallVector<CCValAssign, 16> ArgLocs2;
  CCState CCInfo2(CLI.CallConv, CLI.IsVarArg, DAG.getMachineFunction(),
                  ArgLocs2, *DAG.getContext());
  if (UseBoth)
    CCInfo2.AnalyzeCallOperands(CLI.Outs, getParamCC(CLI.CallConv, true));

  // Get the size of the outgoing arguments stack space requirement.
  unsigned ArgsSize = CCInfo.getNextStackOffset();

  // Keep stack frames 16-byte aligned.
  ArgsSize = alignTo(ArgsSize, 16);

  // Adjust the stack pointer to make room for the arguments.
  // FIXME: Use hasReservedCallFrame to avoid %sp adjustments around all calls
  // with more than 6 arguments.
  Chain = DAG.getCALLSEQ_START(Chain, ArgsSize, 0, DL);

  // Collect the set of registers to pass to the function and their values.
  // This will be emitted as a sequence of CopyToReg nodes glued to the call
  // instruction.
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;

  // Collect chains from all the memory opeations that copy arguments to the
  // stack. They must follow the stack pointer adjustment above and precede the
  // call instruction itself.
  SmallVector<SDValue, 8> MemOpChains;

  // VE needs to get address of callee function in a register
  // So, prepare to copy it to SX12 here.

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  SDValue Callee = CLI.Callee;

  bool IsPICCall = isPositionIndependent();

  // PC-relative references to external symbols should go through $stub.
  // If so, we need to prepare GlobalBaseReg first.
  const TargetMachine &TM = DAG.getTarget();
  const Module *Mod = DAG.getMachineFunction().getFunction().getParent();
  const GlobalValue *GV = nullptr;
  auto *CalleeG = dyn_cast<GlobalAddressSDNode>(Callee);
  if (CalleeG)
    GV = CalleeG->getGlobal();
  bool Local = TM.shouldAssumeDSOLocal(*Mod, GV);
  bool UsePlt = !Local;
  MachineFunction &MF = DAG.getMachineFunction();

  // Turn GlobalAddress/ExternalSymbol node into a value node
  // containing the address of them here.
  if (CalleeG) {
    if (IsPICCall) {
      if (UsePlt)
        Subtarget->getInstrInfo()->getGlobalBaseReg(&MF);
      Callee = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, 0);
      Callee = DAG.getNode(VEISD::GETFUNPLT, DL, PtrVT, Callee);
    } else {
      Callee =
          makeHiLoPair(Callee, VEMCExpr::VK_VE_HI32, VEMCExpr::VK_VE_LO32, DAG);
    }
  } else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    if (IsPICCall) {
      if (UsePlt)
        Subtarget->getInstrInfo()->getGlobalBaseReg(&MF);
      Callee = DAG.getTargetExternalSymbol(E->getSymbol(), PtrVT, 0);
      Callee = DAG.getNode(VEISD::GETFUNPLT, DL, PtrVT, Callee);
    } else {
      Callee =
          makeHiLoPair(Callee, VEMCExpr::VK_VE_HI32, VEMCExpr::VK_VE_LO32, DAG);
    }
  }

  RegsToPass.push_back(std::make_pair(VE::SX12, Callee));

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = CLI.OutVals[i];

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown location info!");
    case CCValAssign::Full:
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
    case CCValAssign::BCvt: {
      // Convert a float argument to i64 with padding.
      //     63     31   0
      //    +------+------+
      //    | float|   0  |
      //    +------+------+
      assert(VA.getLocVT() == MVT::i64);
      assert(VA.getValVT() == MVT::f32);
      SDValue Undef = SDValue(
          DAG.getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, MVT::i64), 0);
      SDValue Sub_f32 = DAG.getTargetConstant(VE::sub_f32, DL, MVT::i32);
      Arg = SDValue(DAG.getMachineNode(TargetOpcode::INSERT_SUBREG, DL,
                                       MVT::i64, Undef, Arg, Sub_f32),
                    0);
      break;
    }
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      if (!UseBoth)
        continue;
      VA = ArgLocs2[i];
    }

    assert(VA.isMemLoc());

    // Create a store off the stack pointer for this argument.
    SDValue StackPtr = DAG.getRegister(VE::SX11, PtrVT);
    // The argument area starts at %fp/%sp + the size of reserved area.
    SDValue PtrOff =
        DAG.getIntPtrConstant(VA.getLocMemOffset() + ArgsBaseOffset, DL);
    PtrOff = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr, PtrOff);
    MemOpChains.push_back(
        DAG.getStore(Chain, DL, Arg, PtrOff, MachinePointerInfo()));
  }

  // Emit all stores, make sure they occur before the call.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  // Build a sequence of CopyToReg nodes glued together with token chain and
  // glue operands which copy the outgoing args into registers. The InGlue is
  // necessary since all emitted instructions must be stuck together in order
  // to pass the live physical registers.
  SDValue InGlue;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, DL, RegsToPass[i].first,
                             RegsToPass[i].second, InGlue);
    InGlue = Chain.getValue(1);
  }

  // Build the operands for the call instruction itself.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  // Add a register mask operand representing the call-preserved registers.
  const VERegisterInfo *TRI = Subtarget->getRegisterInfo();
  const uint32_t *Mask =
      TRI->getCallPreservedMask(DAG.getMachineFunction(), CLI.CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  // Make sure the CopyToReg nodes are glued to the call instruction which
  // consumes the registers.
  if (InGlue.getNode())
    Ops.push_back(InGlue);

  // Now the call itself.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(VEISD::CALL, DL, NodeTys, Ops);
  InGlue = Chain.getValue(1);

  // Revert the stack pointer immediately after the call.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(ArgsSize, DL, true),
                             DAG.getIntPtrConstant(0, DL, true), InGlue, DL);
  InGlue = Chain.getValue(1);

  // Now extract the return values. This is more or less the same as
  // LowerFormalArguments.

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState RVInfo(CLI.CallConv, CLI.IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  // Set inreg flag manually for codegen generated library calls that
  // return float.
  if (CLI.Ins.size() == 1 && CLI.Ins[0].VT == MVT::f32 && !CLI.CB)
    CLI.Ins[0].Flags.setInReg();

  RVInfo.AnalyzeCallResult(CLI.Ins, getReturnCC(CLI.CallConv));

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    unsigned Reg = VA.getLocReg();

    // When returning 'inreg {i32, i32 }', two consecutive i32 arguments can
    // reside in the same register in the high and low bits. Reuse the
    // CopyFromReg previous node to avoid duplicate copies.
    SDValue RV;
    if (RegisterSDNode *SrcReg = dyn_cast<RegisterSDNode>(Chain.getOperand(1)))
      if (SrcReg->getReg() == Reg && Chain->getOpcode() == ISD::CopyFromReg)
        RV = Chain.getValue(0);

    // But usually we'll create a new CopyFromReg for a different register.
    if (!RV.getNode()) {
      RV = DAG.getCopyFromReg(Chain, DL, Reg, RVLocs[i].getLocVT(), InGlue);
      Chain = RV.getValue(1);
      InGlue = Chain.getValue(2);
    }

    // Get the high bits for i32 struct elements.
    if (VA.getValVT() == MVT::i32 && VA.needsCustom())
      RV = DAG.getNode(ISD::SRL, DL, VA.getLocVT(), RV,
                       DAG.getConstant(32, DL, MVT::i32));

    // The callee promoted the return value, so insert an Assert?ext SDNode so
    // we won't promote the value again in this function.
    switch (VA.getLocInfo()) {
    case CCValAssign::SExt:
      RV = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), RV,
                       DAG.getValueType(VA.getValVT()));
      break;
    case CCValAssign::ZExt:
      RV = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), RV,
                       DAG.getValueType(VA.getValVT()));
      break;
    case CCValAssign::BCvt: {
      // Extract a float return value from i64 with padding.
      //     63     31   0
      //    +------+------+
      //    | float|   0  |
      //    +------+------+
      assert(VA.getLocVT() == MVT::i64);
      assert(VA.getValVT() == MVT::f32);
      SDValue Sub_f32 = DAG.getTargetConstant(VE::sub_f32, DL, MVT::i32);
      RV = SDValue(DAG.getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL,
                                      MVT::f32, RV, Sub_f32),
                   0);
      break;
    }
    default:
      break;
    }

    // Truncate the register down to the return value type.
    if (VA.isExtInLoc())
      RV = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), RV);

    InVals.push_back(RV);
  }

  return Chain;
}

bool VETargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  // VE uses 64 bit addressing, so we need multiple instructions to generate
  // an address.  Folding address with offset increases the number of
  // instructions, so that we disable it here.  Offsets will be folded in
  // the DAG combine later if it worth to do so.
  return false;
}

/// isFPImmLegal - Returns true if the target can instruction select the
/// specified FP immediate natively. If false, the legalizer will
/// materialize the FP immediate as a load from a constant pool.
bool VETargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                    bool ForCodeSize) const {
  return VT == MVT::f32 || VT == MVT::f64;
}

/// Determine if the target supports unaligned memory accesses.
///
/// This function returns true if the target allows unaligned memory accesses
/// of the specified type in the given address space. If true, it also returns
/// whether the unaligned memory access is "fast" in the last argument by
/// reference. This is used, for example, in situations where an array
/// copy/move/set is converted to a sequence of store operations. Its use
/// helps to ensure that such replacements don't generate code that causes an
/// alignment error (trap) on the target machine.
bool VETargetLowering::allowsMisalignedMemoryAccesses(EVT VT,
                                                      unsigned AddrSpace,
                                                      unsigned Align,
                                                      MachineMemOperand::Flags,
                                                      bool *Fast) const {
  if (Fast) {
    // It's fast anytime on VE
    *Fast = true;
  }
  return true;
}

VETargetLowering::VETargetLowering(const TargetMachine &TM,
                                   const VESubtarget &STI)
    : TargetLowering(TM), Subtarget(&STI) {
  // Instructions which use registers as conditionals examine all the
  // bits (as does the pseudo SELECT_CC expansion). I don't think it
  // matters much whether it's ZeroOrOneBooleanContent, or
  // ZeroOrNegativeOneBooleanContent, so, arbitrarily choose the
  // former.
  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent);

  initRegisterClasses();
  initSPUActions();
  initVPUActions();

  setStackPointerRegisterToSaveRestore(VE::SX11);

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::TRUNCATE);

  // Set function alignment to 16 bytes
  setMinFunctionAlignment(Align(16));

  // VE stores all argument by 8 bytes alignment
  setMinStackArgumentAlignment(Align(8));

  computeRegisterProperties(Subtarget->getRegisterInfo());
}

const char *VETargetLowering::getTargetNodeName(unsigned Opcode) const {
#define TARGET_NODE_CASE(NAME)                                                 \
  case VEISD::NAME:                                                            \
    return "VEISD::" #NAME;
  switch ((VEISD::NodeType)Opcode) {
  case VEISD::FIRST_NUMBER:
    break;
    TARGET_NODE_CASE(Lo)
    TARGET_NODE_CASE(Hi)
    TARGET_NODE_CASE(GETFUNPLT)
    TARGET_NODE_CASE(GETSTACKTOP)
    TARGET_NODE_CASE(GETTLSADDR)
    TARGET_NODE_CASE(MEMBARRIER)
    TARGET_NODE_CASE(CALL)
    TARGET_NODE_CASE(VEC_BROADCAST)
    TARGET_NODE_CASE(RET_FLAG)
    TARGET_NODE_CASE(GLOBAL_BASE_REG)

    // Register the VVP_* SDNodes.
#define ADD_VVP_OP(VVP_NAME, ...) TARGET_NODE_CASE(VVP_NAME)
#include "VVPNodes.def"
  }
#undef TARGET_NODE_CASE
  return nullptr;
}

EVT VETargetLowering::getSetCCResultType(const DataLayout &, LLVMContext &,
                                         EVT VT) const {
  return MVT::i32;
}

// Convert to a target node and set target flags.
SDValue VETargetLowering::withTargetFlags(SDValue Op, unsigned TF,
                                          SelectionDAG &DAG) const {
  if (const GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op))
    return DAG.getTargetGlobalAddress(GA->getGlobal(), SDLoc(GA),
                                      GA->getValueType(0), GA->getOffset(), TF);

  if (const BlockAddressSDNode *BA = dyn_cast<BlockAddressSDNode>(Op))
    return DAG.getTargetBlockAddress(BA->getBlockAddress(), Op.getValueType(),
                                     0, TF);

  if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Op))
    return DAG.getTargetConstantPool(CP->getConstVal(), CP->getValueType(0),
                                     CP->getAlign(), CP->getOffset(), TF);

  if (const ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(Op))
    return DAG.getTargetExternalSymbol(ES->getSymbol(), ES->getValueType(0),
                                       TF);

  if (const JumpTableSDNode *JT = dyn_cast<JumpTableSDNode>(Op))
    return DAG.getTargetJumpTable(JT->getIndex(), JT->getValueType(0), TF);

  llvm_unreachable("Unhandled address SDNode");
}

// Split Op into high and low parts according to HiTF and LoTF.
// Return an ADD node combining the parts.
SDValue VETargetLowering::makeHiLoPair(SDValue Op, unsigned HiTF, unsigned LoTF,
                                       SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue Hi = DAG.getNode(VEISD::Hi, DL, VT, withTargetFlags(Op, HiTF, DAG));
  SDValue Lo = DAG.getNode(VEISD::Lo, DL, VT, withTargetFlags(Op, LoTF, DAG));
  return DAG.getNode(ISD::ADD, DL, VT, Hi, Lo);
}

// Build SDNodes for producing an address from a GlobalAddress, ConstantPool,
// or ExternalSymbol SDNode.
SDValue VETargetLowering::makeAddress(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT PtrVT = Op.getValueType();

  // Handle PIC mode first. VE needs a got load for every variable!
  if (isPositionIndependent()) {
    auto GlobalN = dyn_cast<GlobalAddressSDNode>(Op);

    if (isa<ConstantPoolSDNode>(Op) || isa<JumpTableSDNode>(Op) ||
        (GlobalN && GlobalN->getGlobal()->hasLocalLinkage())) {
      // Create following instructions for local linkage PIC code.
      //     lea %reg, label@gotoff_lo
      //     and %reg, %reg, (32)0
      //     lea.sl %reg, label@gotoff_hi(%reg, %got)
      SDValue HiLo = makeHiLoPair(Op, VEMCExpr::VK_VE_GOTOFF_HI32,
                                  VEMCExpr::VK_VE_GOTOFF_LO32, DAG);
      SDValue GlobalBase = DAG.getNode(VEISD::GLOBAL_BASE_REG, DL, PtrVT);
      return DAG.getNode(ISD::ADD, DL, PtrVT, GlobalBase, HiLo);
    }
    // Create following instructions for not local linkage PIC code.
    //     lea %reg, label@got_lo
    //     and %reg, %reg, (32)0
    //     lea.sl %reg, label@got_hi(%reg)
    //     ld %reg, (%reg, %got)
    SDValue HiLo = makeHiLoPair(Op, VEMCExpr::VK_VE_GOT_HI32,
                                VEMCExpr::VK_VE_GOT_LO32, DAG);
    SDValue GlobalBase = DAG.getNode(VEISD::GLOBAL_BASE_REG, DL, PtrVT);
    SDValue AbsAddr = DAG.getNode(ISD::ADD, DL, PtrVT, GlobalBase, HiLo);
    return DAG.getLoad(PtrVT, DL, DAG.getEntryNode(), AbsAddr,
                       MachinePointerInfo::getGOT(DAG.getMachineFunction()));
  }

  // This is one of the absolute code models.
  switch (getTargetMachine().getCodeModel()) {
  default:
    llvm_unreachable("Unsupported absolute code model");
  case CodeModel::Small:
  case CodeModel::Medium:
  case CodeModel::Large:
    // abs64.
    return makeHiLoPair(Op, VEMCExpr::VK_VE_HI32, VEMCExpr::VK_VE_LO32, DAG);
  }
}

/// Custom Lower {

// The mappings for emitLeading/TrailingFence for VE is designed by following
// http://www.cl.cam.ac.uk/~pes20/cpp/cpp0xmappings.html
Instruction *VETargetLowering::emitLeadingFence(IRBuilder<> &Builder,
                                                Instruction *Inst,
                                                AtomicOrdering Ord) const {
  switch (Ord) {
  case AtomicOrdering::NotAtomic:
  case AtomicOrdering::Unordered:
    llvm_unreachable("Invalid fence: unordered/non-atomic");
  case AtomicOrdering::Monotonic:
  case AtomicOrdering::Acquire:
    return nullptr; // Nothing to do
  case AtomicOrdering::Release:
  case AtomicOrdering::AcquireRelease:
    return Builder.CreateFence(AtomicOrdering::Release);
  case AtomicOrdering::SequentiallyConsistent:
    if (!Inst->hasAtomicStore())
      return nullptr; // Nothing to do
    return Builder.CreateFence(AtomicOrdering::SequentiallyConsistent);
  }
  llvm_unreachable("Unknown fence ordering in emitLeadingFence");
}

Instruction *VETargetLowering::emitTrailingFence(IRBuilder<> &Builder,
                                                 Instruction *Inst,
                                                 AtomicOrdering Ord) const {
  switch (Ord) {
  case AtomicOrdering::NotAtomic:
  case AtomicOrdering::Unordered:
    llvm_unreachable("Invalid fence: unordered/not-atomic");
  case AtomicOrdering::Monotonic:
  case AtomicOrdering::Release:
    return nullptr; // Nothing to do
  case AtomicOrdering::Acquire:
  case AtomicOrdering::AcquireRelease:
    return Builder.CreateFence(AtomicOrdering::Acquire);
  case AtomicOrdering::SequentiallyConsistent:
    return Builder.CreateFence(AtomicOrdering::SequentiallyConsistent);
  }
  llvm_unreachable("Unknown fence ordering in emitTrailingFence");
}

SDValue VETargetLowering::lowerATOMIC_FENCE(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  AtomicOrdering FenceOrdering = static_cast<AtomicOrdering>(
      cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue());
  SyncScope::ID FenceSSID = static_cast<SyncScope::ID>(
      cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue());

  // VE uses Release consistency, so need a fence instruction if it is a
  // cross-thread fence.
  if (FenceSSID == SyncScope::System) {
    switch (FenceOrdering) {
    case AtomicOrdering::NotAtomic:
    case AtomicOrdering::Unordered:
    case AtomicOrdering::Monotonic:
      // No need to generate fencem instruction here.
      break;
    case AtomicOrdering::Acquire:
      // Generate "fencem 2" as acquire fence.
      return SDValue(DAG.getMachineNode(VE::FENCEM, DL, MVT::Other,
                                        DAG.getTargetConstant(2, DL, MVT::i32),
                                        Op.getOperand(0)),
                     0);
    case AtomicOrdering::Release:
      // Generate "fencem 1" as release fence.
      return SDValue(DAG.getMachineNode(VE::FENCEM, DL, MVT::Other,
                                        DAG.getTargetConstant(1, DL, MVT::i32),
                                        Op.getOperand(0)),
                     0);
    case AtomicOrdering::AcquireRelease:
    case AtomicOrdering::SequentiallyConsistent:
      // Generate "fencem 3" as acq_rel and seq_cst fence.
      // FIXME: "fencem 3" doesn't wait for for PCIe deveices accesses,
      //        so  seq_cst may require more instruction for them.
      return SDValue(DAG.getMachineNode(VE::FENCEM, DL, MVT::Other,
                                        DAG.getTargetConstant(3, DL, MVT::i32),
                                        Op.getOperand(0)),
                     0);
    }
  }

  // MEMBARRIER is a compiler barrier; it codegens to a no-op.
  return DAG.getNode(VEISD::MEMBARRIER, DL, MVT::Other, Op.getOperand(0));
}

SDValue VETargetLowering::lowerGlobalAddress(SDValue Op,
                                             SelectionDAG &DAG) const {
  return makeAddress(Op, DAG);
}

SDValue VETargetLowering::lowerBlockAddress(SDValue Op,
                                            SelectionDAG &DAG) const {
  return makeAddress(Op, DAG);
}

SDValue VETargetLowering::lowerConstantPool(SDValue Op,
                                            SelectionDAG &DAG) const {
  return makeAddress(Op, DAG);
}

SDValue
VETargetLowering::lowerToTLSGeneralDynamicModel(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);

  // Generate the following code:
  //   t1: ch,glue = callseq_start t0, 0, 0
  //   t2: i64,ch,glue = VEISD::GETTLSADDR t1, label, t1:1
  //   t3: ch,glue = callseq_end t2, 0, 0, t2:2
  //   t4: i64,ch,glue = CopyFromReg t3, Register:i64 $sx0, t3:1
  SDValue Label = withTargetFlags(Op, 0, DAG);
  EVT PtrVT = Op.getValueType();

  // Lowering the machine isd will make sure everything is in the right
  // location.
  SDValue Chain = DAG.getEntryNode();
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  const uint32_t *Mask = Subtarget->getRegisterInfo()->getCallPreservedMask(
      DAG.getMachineFunction(), CallingConv::C);
  Chain = DAG.getCALLSEQ_START(Chain, 64, 0, DL);
  SDValue Args[] = {Chain, Label, DAG.getRegisterMask(Mask), Chain.getValue(1)};
  Chain = DAG.getNode(VEISD::GETTLSADDR, DL, NodeTys, Args);
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(64, DL, true),
                             DAG.getIntPtrConstant(0, DL, true),
                             Chain.getValue(1), DL);
  Chain = DAG.getCopyFromReg(Chain, DL, VE::SX0, PtrVT, Chain.getValue(1));

  // GETTLSADDR will be codegen'ed as call. Inform MFI that function has calls.
  MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();
  MFI.setHasCalls(true);

  // Also generate code to prepare a GOT register if it is PIC.
  if (isPositionIndependent()) {
    MachineFunction &MF = DAG.getMachineFunction();
    Subtarget->getInstrInfo()->getGlobalBaseReg(&MF);
  }

  return Chain;
}

SDValue VETargetLowering::lowerGlobalTLSAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  // The current implementation of nld (2.26) doesn't allow local exec model
  // code described in VE-tls_v1.1.pdf (*1) as its input. Instead, we always
  // generate the general dynamic model code sequence.
  //
  // *1: https://www.nec.com/en/global/prod/hpc/aurora/document/VE-tls_v1.1.pdf
  return lowerToTLSGeneralDynamicModel(Op, DAG);
}

SDValue VETargetLowering::lowerJumpTable(SDValue Op, SelectionDAG &DAG) const {
  return makeAddress(Op, DAG);
}

// Lower a f128 load into two f64 loads.
static SDValue lowerLoadF128(SDValue Op, SelectionDAG &DAG) {
  SDLoc DL(Op);
  LoadSDNode *LdNode = dyn_cast<LoadSDNode>(Op.getNode());
  assert(LdNode && LdNode->getOffset().isUndef() && "Unexpected node type");
  unsigned Alignment = LdNode->getAlign().value();
  if (Alignment > 8)
    Alignment = 8;

  SDValue Lo64 =
      DAG.getLoad(MVT::f64, DL, LdNode->getChain(), LdNode->getBasePtr(),
                  LdNode->getPointerInfo(), Alignment,
                  LdNode->isVolatile() ? MachineMemOperand::MOVolatile
                                       : MachineMemOperand::MONone);
  EVT AddrVT = LdNode->getBasePtr().getValueType();
  SDValue HiPtr = DAG.getNode(ISD::ADD, DL, AddrVT, LdNode->getBasePtr(),
                              DAG.getConstant(8, DL, AddrVT));
  SDValue Hi64 =
      DAG.getLoad(MVT::f64, DL, LdNode->getChain(), HiPtr,
                  LdNode->getPointerInfo(), Alignment,
                  LdNode->isVolatile() ? MachineMemOperand::MOVolatile
                                       : MachineMemOperand::MONone);

  SDValue SubRegEven = DAG.getTargetConstant(VE::sub_even, DL, MVT::i32);
  SDValue SubRegOdd = DAG.getTargetConstant(VE::sub_odd, DL, MVT::i32);

  // VE stores Hi64 to 8(addr) and Lo64 to 0(addr)
  SDNode *InFP128 =
      DAG.getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, MVT::f128);
  InFP128 = DAG.getMachineNode(TargetOpcode::INSERT_SUBREG, DL, MVT::f128,
                               SDValue(InFP128, 0), Hi64, SubRegEven);
  InFP128 = DAG.getMachineNode(TargetOpcode::INSERT_SUBREG, DL, MVT::f128,
                               SDValue(InFP128, 0), Lo64, SubRegOdd);
  SDValue OutChains[2] = {SDValue(Lo64.getNode(), 1),
                          SDValue(Hi64.getNode(), 1)};
  SDValue OutChain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  SDValue Ops[2] = {SDValue(InFP128, 0), OutChain};
  return DAG.getMergeValues(Ops, DL);
}

SDValue VETargetLowering::lowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  LoadSDNode *LdNode = cast<LoadSDNode>(Op.getNode());

  SDValue BasePtr = LdNode->getBasePtr();
  if (isa<FrameIndexSDNode>(BasePtr.getNode())) {
    // Do not expand store instruction with frame index here because of
    // dependency problems.  We expand it later in eliminateFrameIndex().
    return Op;
  }

  EVT MemVT = LdNode->getMemoryVT();
  if (MemVT == MVT::f128)
    return lowerLoadF128(Op, DAG);

  return Op;
}

// Lower a f128 store into two f64 stores.
static SDValue lowerStoreF128(SDValue Op, SelectionDAG &DAG) {
  SDLoc DL(Op);
  StoreSDNode *StNode = dyn_cast<StoreSDNode>(Op.getNode());
  assert(StNode && StNode->getOffset().isUndef() && "Unexpected node type");

  SDValue SubRegEven = DAG.getTargetConstant(VE::sub_even, DL, MVT::i32);
  SDValue SubRegOdd = DAG.getTargetConstant(VE::sub_odd, DL, MVT::i32);

  SDNode *Hi64 = DAG.getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL, MVT::i64,
                                    StNode->getValue(), SubRegEven);
  SDNode *Lo64 = DAG.getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL, MVT::i64,
                                    StNode->getValue(), SubRegOdd);

  unsigned Alignment = StNode->getAlign().value();
  if (Alignment > 8)
    Alignment = 8;

  // VE stores Hi64 to 8(addr) and Lo64 to 0(addr)
  SDValue OutChains[2];
  OutChains[0] =
      DAG.getStore(StNode->getChain(), DL, SDValue(Lo64, 0),
                   StNode->getBasePtr(), MachinePointerInfo(), Alignment,
                   StNode->isVolatile() ? MachineMemOperand::MOVolatile
                                        : MachineMemOperand::MONone);
  EVT AddrVT = StNode->getBasePtr().getValueType();
  SDValue HiPtr = DAG.getNode(ISD::ADD, DL, AddrVT, StNode->getBasePtr(),
                              DAG.getConstant(8, DL, AddrVT));
  OutChains[1] =
      DAG.getStore(StNode->getChain(), DL, SDValue(Hi64, 0), HiPtr,
                   MachinePointerInfo(), Alignment,
                   StNode->isVolatile() ? MachineMemOperand::MOVolatile
                                        : MachineMemOperand::MONone);
  return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
}

SDValue VETargetLowering::lowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  StoreSDNode *StNode = cast<StoreSDNode>(Op.getNode());
  assert(StNode && StNode->getOffset().isUndef() && "Unexpected node type");

  SDValue BasePtr = StNode->getBasePtr();
  if (isa<FrameIndexSDNode>(BasePtr.getNode())) {
    // Do not expand store instruction with frame index here because of
    // dependency problems.  We expand it later in eliminateFrameIndex().
    return Op;
  }

  EVT MemVT = StNode->getMemoryVT();
  if (MemVT == MVT::f128)
    return lowerStoreF128(Op, DAG);

  // Otherwise, ask llvm to expand it.
  return SDValue();
}

SDValue VETargetLowering::lowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  VEMachineFunctionInfo *FuncInfo = MF.getInfo<VEMachineFunctionInfo>();
  auto PtrVT = getPointerTy(DAG.getDataLayout());

  // Need frame address to find the address of VarArgsFrameIndex.
  MF.getFrameInfo().setFrameAddressIsTaken(true);

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  SDLoc DL(Op);
  SDValue Offset =
      DAG.getNode(ISD::ADD, DL, PtrVT, DAG.getRegister(VE::SX9, PtrVT),
                  DAG.getIntPtrConstant(FuncInfo->getVarArgsFrameOffset(), DL));
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, Offset, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue VETargetLowering::lowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getValueType(0);
  SDValue InChain = Node->getOperand(0);
  SDValue VAListPtr = Node->getOperand(1);
  EVT PtrVT = VAListPtr.getValueType();
  const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  SDLoc DL(Node);
  SDValue VAList =
      DAG.getLoad(PtrVT, DL, InChain, VAListPtr, MachinePointerInfo(SV));
  SDValue Chain = VAList.getValue(1);
  SDValue NextPtr;

  if (VT == MVT::f128) {
    // VE f128 values must be stored with 16 bytes alignment.  We doesn't
    // know the actual alignment of VAList, so we take alignment of it
    // dyanmically.
    int Align = 16;
    VAList = DAG.getNode(ISD::ADD, DL, PtrVT, VAList,
                         DAG.getConstant(Align - 1, DL, PtrVT));
    VAList = DAG.getNode(ISD::AND, DL, PtrVT, VAList,
                         DAG.getConstant(-Align, DL, PtrVT));
    // Increment the pointer, VAList, by 16 to the next vaarg.
    NextPtr =
        DAG.getNode(ISD::ADD, DL, PtrVT, VAList, DAG.getIntPtrConstant(16, DL));
  } else if (VT == MVT::f32) {
    // float --> need special handling like below.
    //    0      4
    //    +------+------+
    //    | empty| float|
    //    +------+------+
    // Increment the pointer, VAList, by 8 to the next vaarg.
    NextPtr =
        DAG.getNode(ISD::ADD, DL, PtrVT, VAList, DAG.getIntPtrConstant(8, DL));
    // Then, adjust VAList.
    unsigned InternalOffset = 4;
    VAList = DAG.getNode(ISD::ADD, DL, PtrVT, VAList,
                         DAG.getConstant(InternalOffset, DL, PtrVT));
  } else {
    // Increment the pointer, VAList, by 8 to the next vaarg.
    NextPtr =
        DAG.getNode(ISD::ADD, DL, PtrVT, VAList, DAG.getIntPtrConstant(8, DL));
  }

  // Store the incremented VAList to the legalized pointer.
  InChain = DAG.getStore(Chain, DL, NextPtr, VAListPtr, MachinePointerInfo(SV));

  // Load the actual argument out of the pointer VAList.
  // We can't count on greater alignment than the word size.
  return DAG.getLoad(VT, DL, InChain, VAList, MachinePointerInfo(),
                     std::min(PtrVT.getSizeInBits(), VT.getSizeInBits()) / 8);
}

SDValue VETargetLowering::lowerDYNAMIC_STACKALLOC(SDValue Op,
                                                  SelectionDAG &DAG) const {
  // Generate following code.
  //   (void)__llvm_grow_stack(size);
  //   ret = GETSTACKTOP;        // pseudo instruction
  SDLoc DL(Op);

  // Get the inputs.
  SDNode *Node = Op.getNode();
  SDValue Chain = Op.getOperand(0);
  SDValue Size = Op.getOperand(1);
  MaybeAlign Alignment(Op.getConstantOperandVal(2));
  EVT VT = Node->getValueType(0);

  // Chain the dynamic stack allocation so that it doesn't modify the stack
  // pointer when other instructions are using the stack.
  Chain = DAG.getCALLSEQ_START(Chain, 0, 0, DL);

  const TargetFrameLowering &TFI = *Subtarget->getFrameLowering();
  Align StackAlign = TFI.getStackAlign();
  bool NeedsAlign = Alignment.valueOrOne() > StackAlign;

  // Prepare arguments
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = Size;
  Entry.Ty = Entry.Node.getValueType().getTypeForEVT(*DAG.getContext());
  Args.push_back(Entry);
  if (NeedsAlign) {
    Entry.Node = DAG.getConstant(~(Alignment->value() - 1ULL), DL, VT);
    Entry.Ty = Entry.Node.getValueType().getTypeForEVT(*DAG.getContext());
    Args.push_back(Entry);
  }
  Type *RetTy = Type::getVoidTy(*DAG.getContext());

  EVT PtrVT = Op.getValueType();
  SDValue Callee;
  if (NeedsAlign) {
    Callee = DAG.getTargetExternalSymbol("__ve_grow_stack_align", PtrVT, 0);
  } else {
    Callee = DAG.getTargetExternalSymbol("__ve_grow_stack", PtrVT, 0);
  }

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(DL)
      .setChain(Chain)
      .setCallee(CallingConv::PreserveAll, RetTy, Callee, std::move(Args))
      .setDiscardResult(true);
  std::pair<SDValue, SDValue> pair = LowerCallTo(CLI);
  Chain = pair.second;
  SDValue Result = DAG.getNode(VEISD::GETSTACKTOP, DL, VT, Chain);
  if (NeedsAlign) {
    Result = DAG.getNode(ISD::ADD, DL, VT, Result,
                         DAG.getConstant((Alignment->value() - 1ULL), DL, VT));
    Result = DAG.getNode(ISD::AND, DL, VT, Result,
                         DAG.getConstant(~(Alignment->value() - 1ULL), DL, VT));
  }
  //  Chain = Result.getValue(1);
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(0, DL, true),
                             DAG.getIntPtrConstant(0, DL, true), SDValue(), DL);

  SDValue Ops[2] = {Result, Chain};
  return DAG.getMergeValues(Ops, DL);
}

static SDValue getSplatValue(SDNode *N) {
  if (auto *BuildVec = dyn_cast<BuildVectorSDNode>(N)) {
    return BuildVec->getSplatValue();
  }
  return SDValue();
}

SDValue VETargetLowering::lowerBUILD_VECTOR(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  unsigned NumEls = Op.getValueType().getVectorNumElements();
  MVT ElemVT = Op.getSimpleValueType().getVectorElementType();

  if (SDValue ScalarV = getSplatValue(Op.getNode())) {
    // lower to VEC_BROADCAST
    MVT LegalResVT = MVT::getVectorVT(ElemVT, 256);

    auto AVL = DAG.getConstant(NumEls, DL, MVT::i32);
    return DAG.getNode(VEISD::VEC_BROADCAST, DL, LegalResVT, Op.getOperand(0),
                       AVL);
  }

  // Expand
  return SDValue();
}

SDValue VETargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("Should not custom lower this!");
  case ISD::ATOMIC_FENCE:
    return lowerATOMIC_FENCE(Op, DAG);
  case ISD::BlockAddress:
    return lowerBlockAddress(Op, DAG);
  case ISD::ConstantPool:
    return lowerConstantPool(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return lowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:
    return lowerGlobalTLSAddress(Op, DAG);
  case ISD::JumpTable:
    return lowerJumpTable(Op, DAG);
  case ISD::LOAD:
    return lowerLOAD(Op, DAG);
  case ISD::BUILD_VECTOR:
    return lowerBUILD_VECTOR(Op, DAG);
  case ISD::STORE:
    return lowerSTORE(Op, DAG);
  case ISD::VASTART:
    return lowerVASTART(Op, DAG);
  case ISD::VAARG:
    return lowerVAARG(Op, DAG);

#define ADD_BINARY_VVP_OP(VVP_NAME, ISD_NAME) case ISD::ISD_NAME:
#include "VVPNodes.def"
    return lowerToVVP(Op, DAG);
  }
}
/// } Custom Lower

/// JumpTable for VE.
///
///   VE cannot generate relocatable symbol in jump table.  VE cannot
///   generate expressions using symbols in both text segment and data
///   segment like below.
///             .4byte  .LBB0_2-.LJTI0_0
///   So, we generate offset from the top of function like below as
///   a custom label.
///             .4byte  .LBB0_2-<function name>

unsigned VETargetLowering::getJumpTableEncoding() const {
  // Use custom label for PIC.
  if (isPositionIndependent())
    return MachineJumpTableInfo::EK_Custom32;

  // Otherwise, use the normal jump table encoding heuristics.
  return TargetLowering::getJumpTableEncoding();
}

const MCExpr *VETargetLowering::LowerCustomJumpTableEntry(
    const MachineJumpTableInfo *MJTI, const MachineBasicBlock *MBB,
    unsigned Uid, MCContext &Ctx) const {
  assert(isPositionIndependent());

  // Generate custom label for PIC like below.
  //    .4bytes  .LBB0_2-<function name>
  const auto *Value = MCSymbolRefExpr::create(MBB->getSymbol(), Ctx);
  MCSymbol *Sym = Ctx.getOrCreateSymbol(MBB->getParent()->getName().data());
  const auto *Base = MCSymbolRefExpr::create(Sym, Ctx);
  return MCBinaryExpr::createSub(Value, Base, Ctx);
}

SDValue VETargetLowering::getPICJumpTableRelocBase(SDValue Table,
                                                   SelectionDAG &DAG) const {
  assert(isPositionIndependent());
  SDLoc DL(Table);
  Function *Function = &DAG.getMachineFunction().getFunction();
  assert(Function != nullptr);
  auto PtrTy = getPointerTy(DAG.getDataLayout(), Function->getAddressSpace());

  // In the jump table, we have following values in PIC mode.
  //    .4bytes  .LBB0_2-<function name>
  // We need to add this value and the address of this function to generate
  // .LBB0_2 label correctly under PIC mode.  So, we want to generate following
  // instructions:
  //     lea %reg, fun@gotoff_lo
  //     and %reg, %reg, (32)0
  //     lea.sl %reg, fun@gotoff_hi(%reg, %got)
  // In order to do so, we need to genarate correctly marked DAG node using
  // makeHiLoPair.
  SDValue Op = DAG.getGlobalAddress(Function, DL, PtrTy);
  SDValue HiLo = makeHiLoPair(Op, VEMCExpr::VK_VE_GOTOFF_HI32,
                              VEMCExpr::VK_VE_GOTOFF_LO32, DAG);
  SDValue GlobalBase = DAG.getNode(VEISD::GLOBAL_BASE_REG, DL, PtrTy);
  return DAG.getNode(ISD::ADD, DL, PtrTy, GlobalBase, HiLo);
}

static bool isI32Insn(const SDNode *User, const SDNode *N) {
  switch (User->getOpcode()) {
  default:
    return false;
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SETCC:
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::SHL:
  case ISD::SRA:
  case ISD::BSWAP:
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
  case ISD::BR_CC:
  case ISD::BITCAST:
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_SWAP:
    return true;
  case ISD::SRL:
    if (N->getOperand(0).getOpcode() != ISD::SRL)
      return true;
    // (srl (trunc (srl ...))) may be optimized by combining srl, so
    // doesn't optimize trunc now.
    return false;
  case ISD::SELECT_CC:
    if (User->getOperand(2).getNode() != N &&
        User->getOperand(3).getNode() != N)
      return true;
    LLVM_FALLTHROUGH;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SELECT:
  case ISD::CopyToReg:
    // Check all use of selections, bit operations, and copies.  If all of them
    // are safe, optimize truncate to extract_subreg.
    for (SDNode::use_iterator UI = User->use_begin(), UE = User->use_end();
         UI != UE; ++UI) {
      switch ((*UI)->getOpcode()) {
      default:
        // If the use is an instruction which treats the source operand as i32,
        // it is safe to avoid truncate here.
        if (isI32Insn(*UI, N))
          continue;
        break;
      case ISD::ANY_EXTEND:
      case ISD::SIGN_EXTEND:
      case ISD::ZERO_EXTEND: {
        // Special optimizations to the combination of ext and trunc.
        // (ext ... (select ... (trunc ...))) is safe to avoid truncate here
        // since this truncate instruction clears higher 32 bits which is filled
        // by one of ext instructions later.
        assert(N->getValueType(0) == MVT::i32 &&
               "find truncate to not i32 integer");
        if (User->getOpcode() == ISD::SELECT_CC ||
            User->getOpcode() == ISD::SELECT)
          continue;
        break;
      }
      }
      return false;
    }
    return true;
  }
}

// Optimize TRUNCATE in DAG combining.  Optimizing it in CUSTOM lower is
// sometime too early.  Optimizing it in DAG pattern matching in VEInstrInfo.td
// is sometime too late.  So, doing it at here.
SDValue VETargetLowering::combineTRUNCATE(SDNode *N,
                                          DAGCombinerInfo &DCI) const {
  assert(N->getOpcode() == ISD::TRUNCATE &&
         "Should be called with a TRUNCATE node");

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  // We prefer to do this when all types are legal.
  if (!DCI.isAfterLegalizeDAG())
    return SDValue();

  // Skip combine TRUNCATE atm if the operand of TRUNCATE might be a constant.
  if (N->getOperand(0)->getOpcode() == ISD::SELECT_CC &&
      isa<ConstantSDNode>(N->getOperand(0)->getOperand(0)) &&
      isa<ConstantSDNode>(N->getOperand(0)->getOperand(1)))
    return SDValue();

  // Check all use of this TRUNCATE.
  for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end(); UI != UE;
       ++UI) {
    SDNode *User = *UI;

    // Make sure that we're not going to replace TRUNCATE for non i32
    // instructions.
    //
    // FIXME: Although we could sometimes handle this, and it does occur in
    // practice that one of the condition inputs to the select is also one of
    // the outputs, we currently can't deal with this.
    if (isI32Insn(User, N))
      continue;

    return SDValue();
  }

  SDValue SubI32 = DAG.getTargetConstant(VE::sub_i32, DL, MVT::i32);
  return SDValue(DAG.getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL, VT,
                                    N->getOperand(0), SubI32),
                 0);
}

SDValue VETargetLowering::PerformDAGCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::TRUNCATE:
    return combineTRUNCATE(N, DCI);
  }

  return SDValue();
}

//===----------------------------------------------------------------------===//
// VE Inline Assembly Support
//===----------------------------------------------------------------------===//

VETargetLowering::ConstraintType
VETargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'v': // vector registers
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
VETargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                               StringRef Constraint,
                                               MVT VT) const {
  const TargetRegisterClass *RC = nullptr;
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
    case 'r':
      RC = &VE::I64RegClass;
      break;
    case 'v':
      RC = &VE::V64RegClass;
      break;
    }
    return std::make_pair(0U, RC);
  }

  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

//===----------------------------------------------------------------------===//
// VE Target Optimization Support
//===----------------------------------------------------------------------===//

unsigned VETargetLowering::getMinimumJumpTableEntries() const {
  // Specify 8 for PIC model to relieve the impact of PIC load instructions.
  if (isJumpTableRelative())
    return 8;

  return TargetLowering::getMinimumJumpTableEntries();
}

bool VETargetLowering::hasAndNot(SDValue Y) const {
  EVT VT = Y.getValueType();

  // VE doesn't have vector and not instruction.
  if (VT.isVector())
    return false;

  // VE allows different immediate values for X and Y where ~X & Y.
  // Only simm7 works for X, and only mimm works for Y on VE.  However, this
  // function is used to check whether an immediate value is OK for and-not
  // instruction as both X and Y.  Generating additional instruction to
  // retrieve an immediate value is no good since the purpose of this
  // function is to convert a series of 3 instructions to another series of
  // 3 instructions with better parallelism.  Therefore, we return false
  // for all immediate values now.
  // FIXME: Change hasAndNot function to have two operands to make it work
  //        correctly with Aurora VE.
  if (isa<ConstantSDNode>(Y))
    return false;

  // It's ok for generic registers.
  return true;
}

/// \returns the VVP_* SDNode opcode corresponsing to \p OC.
static Optional<unsigned> getVVPOpcode(unsigned OC) {
  switch (OC) {
#define ADD_VVP_OP(VVPNAME, SDNAME)                                            \
  case VEISD::VVPNAME:                                                         \
  case ISD::SDNAME:                                                            \
    return VEISD::VVPNAME;
#include "VVPNodes.def"
  }
  return None;
}

SDValue VETargetLowering::lowerToVVP(SDValue Op, SelectionDAG &DAG) const {
  // Can we represent this as a VVP node.
  auto OCOpt = getVVPOpcode(Op->getOpcode());
  if (!OCOpt.hasValue())
    return SDValue();
  unsigned VVPOC = OCOpt.getValue();

  // The representative and legalized vector type of this operation.
  EVT OpVecVT = Op.getValueType();
  EVT LegalVecVT = getTypeToTransformTo(*DAG.getContext(), OpVecVT);

  // Materialize the VL parameter.
  SDLoc DL(Op);
  SDValue AVL = DAG.getConstant(OpVecVT.getVectorNumElements(), DL, MVT::i32);
  MVT MaskVT = MVT::v256i1;
  SDValue ConstTrue = DAG.getConstant(1, DL, MVT::i32);
  SDValue Mask = DAG.getNode(VEISD::VEC_BROADCAST, DL, MaskVT,
                             ConstTrue); // emit a VEISD::VEC_BROADCAST here.

  // Categories we are interested in.
  bool IsBinaryOp = false;

  switch (VVPOC) {
#define ADD_BINARY_VVP_OP(VVPNAME, ...)                                        \
  case VEISD::VVPNAME:                                                         \
    IsBinaryOp = true;                                                         \
    break;
#include "VVPNodes.def"
  }

  if (IsBinaryOp) {
    assert(LegalVecVT.isSimple());
    return DAG.getNode(VVPOC, DL, LegalVecVT, Op->getOperand(0),
                       Op->getOperand(1), Mask, AVL);
  }
  llvm_unreachable("lowerToVVP called for unexpected SDNode.");
}
