//=- WebAssemblyISelLowering.cpp - WebAssembly DAG Lowering Implementation -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the WebAssemblyTargetLowering class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyISelLowering.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-lower"

WebAssemblyTargetLowering::WebAssemblyTargetLowering(
    const TargetMachine &TM, const WebAssemblySubtarget &STI)
    : TargetLowering(TM), Subtarget(&STI) {
  auto MVTPtr = Subtarget->hasAddr64() ? MVT::i64 : MVT::i32;

  // Booleans always contain 0 or 1.
  setBooleanContents(ZeroOrOneBooleanContent);
  // WebAssembly does not produce floating-point exceptions on normal floating
  // point operations.
  setHasFloatingPointExceptions(false);
  // We don't know the microarchitecture here, so just reduce register pressure.
  setSchedulingPreference(Sched::RegPressure);
  // Tell ISel that we have a stack pointer.
  setStackPointerRegisterToSaveRestore(
      Subtarget->hasAddr64() ? WebAssembly::SP64 : WebAssembly::SP32);
  // Set up the register classes.
  addRegisterClass(MVT::i32, &WebAssembly::I32RegClass);
  addRegisterClass(MVT::i64, &WebAssembly::I64RegClass);
  addRegisterClass(MVT::f32, &WebAssembly::F32RegClass);
  addRegisterClass(MVT::f64, &WebAssembly::F64RegClass);
  if (Subtarget->hasSIMD128()) {
    addRegisterClass(MVT::v16i8, &WebAssembly::V128RegClass);
    addRegisterClass(MVT::v8i16, &WebAssembly::V128RegClass);
    addRegisterClass(MVT::v4i32, &WebAssembly::V128RegClass);
    addRegisterClass(MVT::v4f32, &WebAssembly::V128RegClass);
  }
  // Compute derived properties from the register classes.
  computeRegisterProperties(Subtarget->getRegisterInfo());

  setOperationAction(ISD::GlobalAddress, MVTPtr, Custom);
  setOperationAction(ISD::ExternalSymbol, MVTPtr, Custom);
  setOperationAction(ISD::JumpTable, MVTPtr, Custom);
  setOperationAction(ISD::BlockAddress, MVTPtr, Custom);
  setOperationAction(ISD::BRIND, MVT::Other, Custom);

  // Take the default expansion for va_arg, va_copy, and va_end. There is no
  // default action for va_start, so we do that custom.
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  for (auto T : {MVT::f32, MVT::f64}) {
    // Don't expand the floating-point types to constant pools.
    setOperationAction(ISD::ConstantFP, T, Legal);
    // Expand floating-point comparisons.
    for (auto CC : {ISD::SETO, ISD::SETUO, ISD::SETUEQ, ISD::SETONE,
                    ISD::SETULT, ISD::SETULE, ISD::SETUGT, ISD::SETUGE})
      setCondCodeAction(CC, T, Expand);
    // Expand floating-point library function operators.
    for (auto Op : {ISD::FSIN, ISD::FCOS, ISD::FSINCOS, ISD::FPOW, ISD::FREM,
                    ISD::FMA})
      setOperationAction(Op, T, Expand);
    // Note supported floating-point library function operators that otherwise
    // default to expand.
    for (auto Op :
         {ISD::FCEIL, ISD::FFLOOR, ISD::FTRUNC, ISD::FNEARBYINT, ISD::FRINT})
      setOperationAction(Op, T, Legal);
    // Support minnan and maxnan, which otherwise default to expand.
    setOperationAction(ISD::FMINNAN, T, Legal);
    setOperationAction(ISD::FMAXNAN, T, Legal);
    // WebAssembly currently has no builtin f16 support.
    setOperationAction(ISD::FP16_TO_FP, T, Expand);
    setOperationAction(ISD::FP_TO_FP16, T, Expand);
    setLoadExtAction(ISD::EXTLOAD, T, MVT::f16, Expand);
    setTruncStoreAction(T, MVT::f16, Expand);
  }

  for (auto T : {MVT::i32, MVT::i64}) {
    // Expand unavailable integer operations.
    for (auto Op :
         {ISD::BSWAP, ISD::SMUL_LOHI, ISD::UMUL_LOHI,
          ISD::MULHS, ISD::MULHU, ISD::SDIVREM, ISD::UDIVREM, ISD::SHL_PARTS,
          ISD::SRA_PARTS, ISD::SRL_PARTS, ISD::ADDC, ISD::ADDE, ISD::SUBC,
          ISD::SUBE}) {
      setOperationAction(Op, T, Expand);
    }
  }

  // As a special case, these operators use the type to mean the type to
  // sign-extend from.
  for (auto T : {MVT::i1, MVT::i8, MVT::i16, MVT::i32})
    setOperationAction(ISD::SIGN_EXTEND_INREG, T, Expand);

  // Dynamic stack allocation: use the default expansion.
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVTPtr, Expand);

  setOperationAction(ISD::FrameIndex, MVT::i32, Custom);
  setOperationAction(ISD::CopyToReg, MVT::Other, Custom);

  // Expand these forms; we pattern-match the forms that we can handle in isel.
  for (auto T : {MVT::i32, MVT::i64, MVT::f32, MVT::f64})
    for (auto Op : {ISD::BR_CC, ISD::SELECT_CC})
      setOperationAction(Op, T, Expand);

  // We have custom switch handling.
  setOperationAction(ISD::BR_JT, MVT::Other, Custom);

  // WebAssembly doesn't have:
  //  - Floating-point extending loads.
  //  - Floating-point truncating stores.
  //  - i1 extending loads.
  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  for (auto T : MVT::integer_valuetypes())
    for (auto Ext : {ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD})
      setLoadExtAction(Ext, T, MVT::i1, Promote);

  // Trap lowers to wasm unreachable
  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  setMaxAtomicSizeInBitsSupported(64);
}

FastISel *WebAssemblyTargetLowering::createFastISel(
    FunctionLoweringInfo &FuncInfo, const TargetLibraryInfo *LibInfo) const {
  return WebAssembly::createFastISel(FuncInfo, LibInfo);
}

bool WebAssemblyTargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode * /*GA*/) const {
  // All offsets can be folded.
  return true;
}

MVT WebAssemblyTargetLowering::getScalarShiftAmountTy(const DataLayout & /*DL*/,
                                                      EVT VT) const {
  unsigned BitWidth = NextPowerOf2(VT.getSizeInBits() - 1);
  if (BitWidth > 1 && BitWidth < 8) BitWidth = 8;

  if (BitWidth > 64) {
    // The shift will be lowered to a libcall, and compiler-rt libcalls expect
    // the count to be an i32.
    BitWidth = 32;
    assert(BitWidth >= Log2_32_Ceil(VT.getSizeInBits()) &&
           "32-bit shift counts ought to be enough for anyone");
  }

  MVT Result = MVT::getIntegerVT(BitWidth);
  assert(Result != MVT::INVALID_SIMPLE_VALUE_TYPE &&
         "Unable to represent scalar shift amount type");
  return Result;
}

const char *WebAssemblyTargetLowering::getTargetNodeName(
    unsigned Opcode) const {
  switch (static_cast<WebAssemblyISD::NodeType>(Opcode)) {
    case WebAssemblyISD::FIRST_NUMBER:
      break;
#define HANDLE_NODETYPE(NODE) \
  case WebAssemblyISD::NODE:  \
    return "WebAssemblyISD::" #NODE;
#include "WebAssemblyISD.def"
#undef HANDLE_NODETYPE
  }
  return nullptr;
}

std::pair<unsigned, const TargetRegisterClass *>
WebAssemblyTargetLowering::getRegForInlineAsmConstraint(
    const TargetRegisterInfo *TRI, StringRef Constraint, MVT VT) const {
  // First, see if this is a constraint that directly corresponds to a
  // WebAssembly register class.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
      case 'r':
        assert(VT != MVT::iPTR && "Pointer MVT not expected here");
        if (Subtarget->hasSIMD128() && VT.isVector()) {
          if (VT.getSizeInBits() == 128)
            return std::make_pair(0U, &WebAssembly::V128RegClass);
        }
        if (VT.isInteger() && !VT.isVector()) {
          if (VT.getSizeInBits() <= 32)
            return std::make_pair(0U, &WebAssembly::I32RegClass);
          if (VT.getSizeInBits() <= 64)
            return std::make_pair(0U, &WebAssembly::I64RegClass);
        }
        break;
      default:
        break;
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

bool WebAssemblyTargetLowering::isCheapToSpeculateCttz() const {
  // Assume ctz is a relatively cheap operation.
  return true;
}

bool WebAssemblyTargetLowering::isCheapToSpeculateCtlz() const {
  // Assume clz is a relatively cheap operation.
  return true;
}

bool WebAssemblyTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                      const AddrMode &AM,
                                                      Type *Ty,
                                                      unsigned AS,
                                                      Instruction *I) const {
  // WebAssembly offsets are added as unsigned without wrapping. The
  // isLegalAddressingMode gives us no way to determine if wrapping could be
  // happening, so we approximate this by accepting only non-negative offsets.
  if (AM.BaseOffs < 0) return false;

  // WebAssembly has no scale register operands.
  if (AM.Scale != 0) return false;

  // Everything else is legal.
  return true;
}

bool WebAssemblyTargetLowering::allowsMisalignedMemoryAccesses(
    EVT /*VT*/, unsigned /*AddrSpace*/, unsigned /*Align*/, bool *Fast) const {
  // WebAssembly supports unaligned accesses, though it should be declared
  // with the p2align attribute on loads and stores which do so, and there
  // may be a performance impact. We tell LLVM they're "fast" because
  // for the kinds of things that LLVM uses this for (merging adjacent stores
  // of constants, etc.), WebAssembly implementations will either want the
  // unaligned access or they'll split anyway.
  if (Fast) *Fast = true;
  return true;
}

bool WebAssemblyTargetLowering::isIntDivCheap(EVT VT,
                                              AttributeList Attr) const {
  // The current thinking is that wasm engines will perform this optimization,
  // so we can save on code size.
  return true;
}

//===----------------------------------------------------------------------===//
// WebAssembly Lowering private implementation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lowering Code
//===----------------------------------------------------------------------===//

static void fail(const SDLoc &DL, SelectionDAG &DAG, const char *msg) {
  MachineFunction &MF = DAG.getMachineFunction();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(*MF.getFunction(), msg, DL.getDebugLoc()));
}

// Test whether the given calling convention is supported.
static bool CallingConvSupported(CallingConv::ID CallConv) {
  // We currently support the language-independent target-independent
  // conventions. We don't yet have a way to annotate calls with properties like
  // "cold", and we don't have any call-clobbered registers, so these are mostly
  // all handled the same.
  return CallConv == CallingConv::C || CallConv == CallingConv::Fast ||
         CallConv == CallingConv::Cold ||
         CallConv == CallingConv::PreserveMost ||
         CallConv == CallingConv::PreserveAll ||
         CallConv == CallingConv::CXX_FAST_TLS;
}

SDValue WebAssemblyTargetLowering::LowerCall(
    CallLoweringInfo &CLI, SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc DL = CLI.DL;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  MachineFunction &MF = DAG.getMachineFunction();
  auto Layout = MF.getDataLayout();

  CallingConv::ID CallConv = CLI.CallConv;
  if (!CallingConvSupported(CallConv))
    fail(DL, DAG,
         "WebAssembly doesn't support language-specific or target-specific "
         "calling conventions yet");
  if (CLI.IsPatchPoint)
    fail(DL, DAG, "WebAssembly doesn't support patch point yet");

  // WebAssembly doesn't currently support explicit tail calls. If they are
  // required, fail. Otherwise, just disable them.
  if ((CallConv == CallingConv::Fast && CLI.IsTailCall &&
       MF.getTarget().Options.GuaranteedTailCallOpt) ||
      (CLI.CS && CLI.CS.isMustTailCall()))
    fail(DL, DAG, "WebAssembly doesn't support tail call yet");
  CLI.IsTailCall = false;

  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  if (Ins.size() > 1)
    fail(DL, DAG, "WebAssembly doesn't support more than 1 returned value yet");

  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  for (unsigned i = 0; i < Outs.size(); ++i) {
    const ISD::OutputArg &Out = Outs[i];
    SDValue &OutVal = OutVals[i];
    if (Out.Flags.isNest())
      fail(DL, DAG, "WebAssembly hasn't implemented nest arguments");
    if (Out.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca arguments");
    if (Out.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs arguments");
    if (Out.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs last arguments");
    if (Out.Flags.isByVal() && Out.Flags.getByValSize() != 0) {
      auto &MFI = MF.getFrameInfo();
      int FI = MFI.CreateStackObject(Out.Flags.getByValSize(),
                                     Out.Flags.getByValAlign(),
                                     /*isSS=*/false);
      SDValue SizeNode =
          DAG.getConstant(Out.Flags.getByValSize(), DL, MVT::i32);
      SDValue FINode = DAG.getFrameIndex(FI, getPointerTy(Layout));
      Chain = DAG.getMemcpy(
          Chain, DL, FINode, OutVal, SizeNode, Out.Flags.getByValAlign(),
          /*isVolatile*/ false, /*AlwaysInline=*/false,
          /*isTailCall*/ false, MachinePointerInfo(), MachinePointerInfo());
      OutVal = FINode;
    }
  }

  bool IsVarArg = CLI.IsVarArg;
  unsigned NumFixedArgs = CLI.NumFixedArgs;

  auto PtrVT = getPointerTy(Layout);

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (IsVarArg) {
    // Outgoing non-fixed arguments are placed in a buffer. First
    // compute their offsets and the total amount of buffer space needed.
    for (SDValue Arg :
         make_range(OutVals.begin() + NumFixedArgs, OutVals.end())) {
      EVT VT = Arg.getValueType();
      assert(VT != MVT::iPTR && "Legalized args should be concrete");
      Type *Ty = VT.getTypeForEVT(*DAG.getContext());
      unsigned Offset = CCInfo.AllocateStack(Layout.getTypeAllocSize(Ty),
                                             Layout.getABITypeAlignment(Ty));
      CCInfo.addLoc(CCValAssign::getMem(ArgLocs.size(), VT.getSimpleVT(),
                                        Offset, VT.getSimpleVT(),
                                        CCValAssign::Full));
    }
  }

  unsigned NumBytes = CCInfo.getAlignedCallFrameSize();

  SDValue FINode;
  if (IsVarArg && NumBytes) {
    // For non-fixed arguments, next emit stores to store the argument values
    // to the stack buffer at the offsets computed above.
    int FI = MF.getFrameInfo().CreateStackObject(NumBytes,
                                                 Layout.getStackAlignment(),
                                                 /*isSS=*/false);
    unsigned ValNo = 0;
    SmallVector<SDValue, 8> Chains;
    for (SDValue Arg :
         make_range(OutVals.begin() + NumFixedArgs, OutVals.end())) {
      assert(ArgLocs[ValNo].getValNo() == ValNo &&
             "ArgLocs should remain in order and only hold varargs args");
      unsigned Offset = ArgLocs[ValNo++].getLocMemOffset();
      FINode = DAG.getFrameIndex(FI, getPointerTy(Layout));
      SDValue Add = DAG.getNode(ISD::ADD, DL, PtrVT, FINode,
                                DAG.getConstant(Offset, DL, PtrVT));
      Chains.push_back(DAG.getStore(
          Chain, DL, Arg, Add,
          MachinePointerInfo::getFixedStack(MF, FI, Offset), 0));
    }
    if (!Chains.empty())
      Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
  } else if (IsVarArg) {
    FINode = DAG.getIntPtrConstant(0, DL);
  }

  // Compute the operands for the CALLn node.
  SmallVector<SDValue, 16> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add all fixed arguments. Note that for non-varargs calls, NumFixedArgs
  // isn't reliable.
  Ops.append(OutVals.begin(),
             IsVarArg ? OutVals.begin() + NumFixedArgs : OutVals.end());
  // Add a pointer to the vararg buffer.
  if (IsVarArg) Ops.push_back(FINode);

  SmallVector<EVT, 8> InTys;
  for (const auto &In : Ins) {
    assert(!In.Flags.isByVal() && "byval is not valid for return values");
    assert(!In.Flags.isNest() && "nest is not valid for return values");
    if (In.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca return values");
    if (In.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs return values");
    if (In.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG,
           "WebAssembly hasn't implemented cons regs last return values");
    // Ignore In.getOrigAlign() because all our arguments are passed in
    // registers.
    InTys.push_back(In.VT);
  }
  InTys.push_back(MVT::Other);
  SDVTList InTyList = DAG.getVTList(InTys);
  SDValue Res =
      DAG.getNode(Ins.empty() ? WebAssemblyISD::CALL0 : WebAssemblyISD::CALL1,
                  DL, InTyList, Ops);
  if (Ins.empty()) {
    Chain = Res;
  } else {
    InVals.push_back(Res);
    Chain = Res.getValue(1);
  }

  return Chain;
}

bool WebAssemblyTargetLowering::CanLowerReturn(
    CallingConv::ID /*CallConv*/, MachineFunction & /*MF*/, bool /*IsVarArg*/,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    LLVMContext & /*Context*/) const {
  // WebAssembly can't currently handle returning tuples.
  return Outs.size() <= 1;
}

SDValue WebAssemblyTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool /*IsVarArg*/,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
    SelectionDAG &DAG) const {
  assert(Outs.size() <= 1 && "WebAssembly can only return up to one value");
  if (!CallingConvSupported(CallConv))
    fail(DL, DAG, "WebAssembly doesn't support non-C calling conventions");

  SmallVector<SDValue, 4> RetOps(1, Chain);
  RetOps.append(OutVals.begin(), OutVals.end());
  Chain = DAG.getNode(WebAssemblyISD::RETURN, DL, MVT::Other, RetOps);

  // Record the number and types of the return values.
  for (const ISD::OutputArg &Out : Outs) {
    assert(!Out.Flags.isByVal() && "byval is not valid for return values");
    assert(!Out.Flags.isNest() && "nest is not valid for return values");
    assert(Out.IsFixed && "non-fixed return value is not valid");
    if (Out.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca results");
    if (Out.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs results");
    if (Out.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs last results");
  }

  return Chain;
}

SDValue WebAssemblyTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  if (!CallingConvSupported(CallConv))
    fail(DL, DAG, "WebAssembly doesn't support non-C calling conventions");

  MachineFunction &MF = DAG.getMachineFunction();
  auto *MFI = MF.getInfo<WebAssemblyFunctionInfo>();

  // Set up the incoming ARGUMENTS value, which serves to represent the liveness
  // of the incoming values before they're represented by virtual registers.
  MF.getRegInfo().addLiveIn(WebAssembly::ARGUMENTS);

  for (const ISD::InputArg &In : Ins) {
    if (In.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca arguments");
    if (In.Flags.isNest())
      fail(DL, DAG, "WebAssembly hasn't implemented nest arguments");
    if (In.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs arguments");
    if (In.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs last arguments");
    // Ignore In.getOrigAlign() because all our arguments are passed in
    // registers.
    InVals.push_back(
        In.Used
            ? DAG.getNode(WebAssemblyISD::ARGUMENT, DL, In.VT,
                          DAG.getTargetConstant(InVals.size(), DL, MVT::i32))
            : DAG.getUNDEF(In.VT));

    // Record the number and types of arguments.
    MFI->addParam(In.VT);
  }

  // Varargs are copied into a buffer allocated by the caller, and a pointer to
  // the buffer is passed as an argument.
  if (IsVarArg) {
    MVT PtrVT = getPointerTy(MF.getDataLayout());
    unsigned VarargVreg =
        MF.getRegInfo().createVirtualRegister(getRegClassFor(PtrVT));
    MFI->setVarargBufferVreg(VarargVreg);
    Chain = DAG.getCopyToReg(
        Chain, DL, VarargVreg,
        DAG.getNode(WebAssemblyISD::ARGUMENT, DL, PtrVT,
                    DAG.getTargetConstant(Ins.size(), DL, MVT::i32)));
    MFI->addParam(PtrVT);
  }

  // Record the number and types of results.
  SmallVector<MVT, 4> Params;
  SmallVector<MVT, 4> Results;
  ComputeSignatureVTs(*MF.getFunction(), DAG.getTarget(), Params, Results);
  for (MVT VT : Results)
    MFI->addResult(VT);

  return Chain;
}

//===----------------------------------------------------------------------===//
//  Custom lowering hooks.
//===----------------------------------------------------------------------===//

SDValue WebAssemblyTargetLowering::LowerOperation(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  switch (Op.getOpcode()) {
    default:
      llvm_unreachable("unimplemented operation lowering");
      return SDValue();
    case ISD::FrameIndex:
      return LowerFrameIndex(Op, DAG);
    case ISD::GlobalAddress:
      return LowerGlobalAddress(Op, DAG);
    case ISD::ExternalSymbol:
      return LowerExternalSymbol(Op, DAG);
    case ISD::JumpTable:
      return LowerJumpTable(Op, DAG);
    case ISD::BR_JT:
      return LowerBR_JT(Op, DAG);
    case ISD::VASTART:
      return LowerVASTART(Op, DAG);
    case ISD::BlockAddress:
    case ISD::BRIND:
      fail(DL, DAG, "WebAssembly hasn't implemented computed gotos");
      return SDValue();
    case ISD::RETURNADDR: // Probably nothing meaningful can be returned here.
      fail(DL, DAG, "WebAssembly hasn't implemented __builtin_return_address");
      return SDValue();
    case ISD::FRAMEADDR:
      return LowerFRAMEADDR(Op, DAG);
    case ISD::CopyToReg:
      return LowerCopyToReg(Op, DAG);
  }
}

SDValue WebAssemblyTargetLowering::LowerCopyToReg(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDValue Src = Op.getOperand(2);
  if (isa<FrameIndexSDNode>(Src.getNode())) {
    // CopyToReg nodes don't support FrameIndex operands. Other targets select
    // the FI to some LEA-like instruction, but since we don't have that, we
    // need to insert some kind of instruction that can take an FI operand and
    // produces a value usable by CopyToReg (i.e. in a vreg). So insert a dummy
    // copy_local between Op and its FI operand.
    SDValue Chain = Op.getOperand(0);
    SDLoc DL(Op);
    unsigned Reg = cast<RegisterSDNode>(Op.getOperand(1))->getReg();
    EVT VT = Src.getValueType();
    SDValue Copy(
        DAG.getMachineNode(VT == MVT::i32 ? WebAssembly::COPY_I32
                                          : WebAssembly::COPY_I64,
                           DL, VT, Src),
        0);
    return Op.getNode()->getNumValues() == 1
               ? DAG.getCopyToReg(Chain, DL, Reg, Copy)
               : DAG.getCopyToReg(Chain, DL, Reg, Copy, Op.getNumOperands() == 4
                                                            ? Op.getOperand(3)
                                                            : SDValue());
  }
  return SDValue();
}

SDValue WebAssemblyTargetLowering::LowerFrameIndex(SDValue Op,
                                                   SelectionDAG &DAG) const {
  int FI = cast<FrameIndexSDNode>(Op)->getIndex();
  return DAG.getTargetFrameIndex(FI, Op.getValueType());
}

SDValue WebAssemblyTargetLowering::LowerFRAMEADDR(SDValue Op,
                                                  SelectionDAG &DAG) const {
  // Non-zero depths are not supported by WebAssembly currently. Use the
  // legalizer's default expansion, which is to return 0 (what this function is
  // documented to do).
  if (Op.getConstantOperandVal(0) > 0)
    return SDValue();

  DAG.getMachineFunction().getFrameInfo().setFrameAddressIsTaken(true);
  EVT VT = Op.getValueType();
  unsigned FP =
      Subtarget->getRegisterInfo()->getFrameRegister(DAG.getMachineFunction());
  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(Op), FP, VT);
}

SDValue WebAssemblyTargetLowering::LowerGlobalAddress(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const auto *GA = cast<GlobalAddressSDNode>(Op);
  EVT VT = Op.getValueType();
  assert(GA->getTargetFlags() == 0 &&
         "Unexpected target flags on generic GlobalAddressSDNode");
  if (GA->getAddressSpace() != 0)
    fail(DL, DAG, "WebAssembly only expects the 0 address space");
  return DAG.getNode(
      WebAssemblyISD::Wrapper, DL, VT,
      DAG.getTargetGlobalAddress(GA->getGlobal(), DL, VT, GA->getOffset()));
}

SDValue WebAssemblyTargetLowering::LowerExternalSymbol(
    SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const auto *ES = cast<ExternalSymbolSDNode>(Op);
  EVT VT = Op.getValueType();
  assert(ES->getTargetFlags() == 0 &&
         "Unexpected target flags on generic ExternalSymbolSDNode");
  // Set the TargetFlags to 0x1 which indicates that this is a "function"
  // symbol rather than a data symbol. We do this unconditionally even though
  // we don't know anything about the symbol other than its name, because all
  // external symbols used in target-independent SelectionDAG code are for
  // functions.
  return DAG.getNode(WebAssemblyISD::Wrapper, DL, VT,
                     DAG.getTargetExternalSymbol(ES->getSymbol(), VT,
                                                 /*TargetFlags=*/0x1));
}

SDValue WebAssemblyTargetLowering::LowerJumpTable(SDValue Op,
                                                  SelectionDAG &DAG) const {
  // There's no need for a Wrapper node because we always incorporate a jump
  // table operand into a BR_TABLE instruction, rather than ever
  // materializing it in a register.
  const JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  return DAG.getTargetJumpTable(JT->getIndex(), Op.getValueType(),
                                JT->getTargetFlags());
}

SDValue WebAssemblyTargetLowering::LowerBR_JT(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  const auto *JT = cast<JumpTableSDNode>(Op.getOperand(1));
  SDValue Index = Op.getOperand(2);
  assert(JT->getTargetFlags() == 0 && "WebAssembly doesn't set target flags");

  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Index);

  MachineJumpTableInfo *MJTI = DAG.getMachineFunction().getJumpTableInfo();
  const auto &MBBs = MJTI->getJumpTables()[JT->getIndex()].MBBs;

  // Add an operand for each case.
  for (auto MBB : MBBs) Ops.push_back(DAG.getBasicBlock(MBB));

  // TODO: For now, we just pick something arbitrary for a default case for now.
  // We really want to sniff out the guard and put in the real default case (and
  // delete the guard).
  Ops.push_back(DAG.getBasicBlock(MBBs[0]));

  return DAG.getNode(WebAssemblyISD::BR_TABLE, DL, MVT::Other, Ops);
}

SDValue WebAssemblyTargetLowering::LowerVASTART(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT PtrVT = getPointerTy(DAG.getMachineFunction().getDataLayout());

  auto *MFI = DAG.getMachineFunction().getInfo<WebAssemblyFunctionInfo>();
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  SDValue ArgN = DAG.getCopyFromReg(DAG.getEntryNode(), DL,
                                    MFI->getVarargBufferVreg(), PtrVT);
  return DAG.getStore(Op.getOperand(0), DL, ArgN, Op.getOperand(1),
                      MachinePointerInfo(SV), 0);
}

//===----------------------------------------------------------------------===//
//                          WebAssembly Optimization Hooks
//===----------------------------------------------------------------------===//
