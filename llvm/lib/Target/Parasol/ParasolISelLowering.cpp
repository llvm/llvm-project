//===-- ParasolISelLowering.cpp - Parasol DAG Lowering Implementation -----===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Parasol uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//
#include "ParasolISelLowering.h"
#include "ParasolMachineFunction.h"
#include "ParasolSubtarget.h"
#include "ParasolTargetMachine.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/Debug.h"
#include <cassert>

// For Linux support to solve
// error: ‘unique_ptr’ in namespace ‘std’ does not name a template type
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "parasol-isellower"

#include "ParasolGenCallingConv.inc"

ParasolTargetLowering::ParasolTargetLowering(const TargetMachine &TM,
                                             const ParasolSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  // Set up the register classes
  // addRegisterClass(MVT::i32, &Parasol::GPRRegClass);

  // Add our custom register classes
  addRegisterClass(MVT::i1, &Parasol::IRRegClass);
  addRegisterClass(MVT::i8, &Parasol::IRRegClass);
  addRegisterClass(MVT::i16, &Parasol::IRRegClass);
  addRegisterClass(MVT::i32, &Parasol::IRRegClass);

  // Must, computeRegisterProperties - Once all of the register classes are
  // added, this allows us to compute derived properties we expose.
  computeRegisterProperties(Subtarget.getRegisterInfo());

  // SUNSCREEN TODO: Remove?
  setStackPointerRegisterToSaveRestore(Parasol::X2);

  // Set scheduling preference. There are a few options:
  //    - None: No preference
  //    - Source: Follow source order
  //    - RegPressure: Scheduling for lowest register pressure
  //    - Hybrid: Scheduling for both latency and register pressure
  // Source (the option used by XCore) is no good when there are few registers
  // because the compiler will try to keep a lot more things into the register
  // which eventually results in a lot of stack spills for no good reason. So
  // use either RegPressure or Hybrid
  setSchedulingPreference(Sched::RegPressure);

  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::BlockAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool, MVT::i32, Custom);

  // Parasol has no select or setcc: expand to SELECT_CC.
  setOperationAction({ISD::SELECT_CC}, {MVT::i1, MVT::i8, MVT::i16, MVT::i32},
                     Expand);

  setLoadExtAction({ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD}, MVT::i32,
                   {MVT::i8, MVT::i16}, Expand);

  setTruncStoreAction(MVT::i32, MVT::i8, Expand);
  setTruncStoreAction(MVT::i32, MVT::i16, Expand);

  // Branches
  // setOperationAction(ISD::BRCOND, MVT::i1, Custom);
  setOperationAction(ISD::BR_CC, {MVT::i1, MVT::i8, MVT::i16, MVT::i32},
                     Expand);

  // Set minimum and preferred function alignment (log2)
  setMinFunctionAlignment(Align(8));
  setPrefFunctionAlignment(Align(8));

  // Set preferred loop alignment (log2)
  setPrefLoopAlignment(Align(8));
}

const char *ParasolTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case ParasolISD::Ret:
    return "ParasolISD::Ret";
  case ParasolISD::BR_CC:
    return "ParasolISD::BR_CC";
  default:
    return NULL;
  }
}

void ParasolTargetLowering::ReplaceNodeResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom expand this!");
  }
}

//===----------------------------------------------------------------------===//
//@            Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//

// The BeyondRISC calling convention parameter registers.
static const MCPhysReg GPRArgRegs[] = {Parasol::X10, Parasol::X11, Parasol::X12,
                                       Parasol::X13, Parasol::X14, Parasol::X15,
                                       Parasol::X16, Parasol::X17};

bool assignArg(const DataLayout &DL, unsigned ValNo, MVT ValVT, MVT LocVT,
               CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
               CCState &State, bool IsRet, Type *OrigTy,
               const ParasolTargetLowering &TLI) {
  // We use RISCV's RV32 calling convention.
  unsigned XLen = 4;
  unsigned TwoXLen = 8;

  SmallVectorImpl<CCValAssign> &PendingLocs = State.getPendingLocs();
  SmallVectorImpl<ISD::ArgFlagsTy> &PendingArgFlags =
      State.getPendingArgFlags();

  unsigned ValSize = ValVT.getStoreSize();
  Align ValAlign = Align(ValSize);

  // Handle 8-byte values split over 2 registers.
  if (ValVT.isScalarInteger() && (ArgFlags.isSplit() || !PendingLocs.empty())) {
    // If this isn't the last value, store it for later and move to the next
    // one.
    if (!ArgFlags.isSplitEnd()) {
      LocVT = MVT::i32;
      LocInfo = CCValAssign::Indirect;
      PendingLocs.push_back(
          CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));
      PendingArgFlags.push_back(ArgFlags);

      return false;
    } else {
      assert(PendingLocs.size() == 1 && "Unexpected PendingLocs.size()");
      CCValAssign VALLo = PendingLocs[0];
      ISD::ArgFlagsTy AFLo = PendingArgFlags[0];
      PendingLocs.clear();
      PendingArgFlags.clear();

      // Try placing the lo word in a register. Failing that, it goes on the
      // stack.
      if (Register Reg = State.AllocateReg(GPRArgRegs)) {
        State.addLoc(CCValAssign::getReg(VALLo.getValNo(), VALLo.getValVT(),
                                         Reg, VALLo.getLocVT(),
                                         CCValAssign::Full));
      } else {
        State.addLoc(CCValAssign::getMem(
            VALLo.getValNo(), VALLo.getValVT(),
            State.AllocateStack(XLen, Align(AFLo.getNonZeroOrigAlign())),
            VALLo.getLocVT(), CCValAssign::Full));
      }

      // Now try placing the hi word. Again, it goes on the stack if no
      // registers are available.
      if (Register Reg = State.AllocateReg(GPRArgRegs)) {
        State.addLoc(
            CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, CCValAssign::Full));
      } else {
        State.addLoc(CCValAssign::getMem(ValNo, ValVT,
                                         State.AllocateStack(XLen, Align(XLen)),
                                         LocVT, CCValAssign::Full));
      }
    }
  } else if (ValSize <= 4) { // <= 4-byte values go in a register/stack
    if (Register Reg = State.AllocateReg(GPRArgRegs)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    } else {
      State.addLoc(CCValAssign::getMem(ValNo, ValVT,
                                       State.AllocateStack(ValSize, ValAlign),
                                       LocVT, LocInfo));
    }
  } else { // Larger values are passed by reference.
    assert("Unimplemented");
  }

  return false;
}

void ParasolTargetLowering::analyzeInputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet) const {
  unsigned NumArgs = Ins.size();
  FunctionType *FType = MF.getFunction().getFunctionType();

  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ArgVT = Ins[i].VT;
    ISD::ArgFlagsTy ArgFlags = Ins[i].Flags;

    Type *ArgTy = nullptr;
    if (IsRet)
      ArgTy = FType->getReturnType();
    else if (Ins[i].isOrigArg())
      ArgTy = FType->getParamType(Ins[i].getOrigArgIndex());

    if (assignArg(MF.getDataLayout(), i, ArgVT, ArgVT, CCValAssign::Full,
                  ArgFlags, CCInfo, IsRet, ArgTy, *this)) {
      LLVM_DEBUG(dbgs() << "InputArg #" << i << " has unhandled type " << ArgVT
                        << '\n');
      llvm_unreachable(nullptr);
    }
  }
}

void ParasolTargetLowering::analyzeOutputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::OutputArg> &Outs) const {

  for (unsigned i = 0; i < Outs.size(); i++) {
    MVT ArgVT = Outs[i].VT;

    Register Reg = CCInfo.AllocateReg(GPRArgRegs);
    CCInfo.addLoc(CCValAssign::getReg(i, ArgVT, Reg, ArgVT, CCValAssign::Full));
  }
}

// Convert Val to a ValVT. Should not be called for CCValAssign::Indirect
// values.
static SDValue convertLocVTToValVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL) {
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    break;
  case CCValAssign::BCvt:
    Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromRegLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL,
                                const ISD::InputArg &In,
                                const ParasolTargetLowering &TLI) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  EVT LocVT = VA.getLocVT();
  SDValue Val;
  const TargetRegisterClass *RC = TLI.getRegClassFor(LocVT.getSimpleVT());
  Register VReg = RegInfo.createVirtualRegister(RC);
  RegInfo.addLiveIn(VA.getLocReg(), VReg);

  // Our arguments are packed as 32-bit values, so we need to start there and
  // truncate to the actual size below.
  Val = DAG.getCopyFromReg(Chain, DL, VReg, MVT::i32);

  // If input is sign extended from 32 bits, note it for the SExtWRemoval pass.
  if (In.isOrigArg()) {
    Argument *OrigArg = MF.getFunction().getArg(In.getOrigArgIndex());
    if (OrigArg->getType()->isIntegerTy()) {
      unsigned BitWidth = OrigArg->getType()->getIntegerBitWidth();
      // An input zero extended from i31 can also be considered sign extended.
      if ((BitWidth <= 32 && In.Flags.isSExt()) ||
          (BitWidth < 32 && In.Flags.isZExt())) {
        ParasolFunctionInfo *PFI = MF.getInfo<ParasolFunctionInfo>();
        Val = DAG.getZExtOrTrunc(Val, DL, VA.getValVT());
        // PFI->addSExt32Register(VReg);
      }
    }
  }

  if (VA.getLocInfo() == CCValAssign::Indirect) {
    return Val;
  }

  return convertLocVTToValVT(DAG, Val, VA, DL);
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

/// LowerFormalArguments - transform physical registers into virtual registers
/// and generate load operations for arguments places on the stack.
SDValue ParasolTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  assert((CallingConv::C == CallConv || CallingConv::Fast == CallConv) &&
         "Unsupported CallingConv to FORMAL_ARGS");

  if (isVarArg) {
    llvm_unreachable("Variadic arguments not supported");
  }

  MachineFunction &MF = DAG.getMachineFunction();
  Function &F = MF.getFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, MF, ArgLocs, *DAG.getContext());

  SmallVector<SDValue, 16> ArgValues;
  SDValue ArgValue;
  Function::const_arg_iterator CurOrigArg = MF.getFunction().arg_begin();

  analyzeInputArgs(MF, CCInfo, Ins, false);

  for (unsigned i = 0, e = ArgLocs.size(), InsIdx = 0; i != e; ++i, ++InsIdx) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue;

    if (VA.isRegLoc()) {
      ArgValue = unpackFromRegLoc(DAG, Chain, VA, DL, Ins[InsIdx], *this);
    } else {
      ArgValue = unpackFromMemLoc(DAG, Chain, VA, DL);
    }

    InVals.push_back(ArgValue);
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//@              Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

bool ParasolTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  switch (Outs.size()) {
  case 0:
    return true;
  case 1:
    return Outs[0].VT.getStoreSize() <= 8;
  case 2:
    return Outs[0].VT.getStoreSize() + Outs[1].VT.getStoreSize() <= 8;
  default:
    return false;
  }

  return false;
}

/// LowerMemOpCallTo - Store the argument to the stack.
SDValue ParasolTargetLowering::LowerMemOpCallTo(SDValue Chain, SDValue Arg,
                                                const SDLoc &dl,
                                                SelectionDAG &DAG,
                                                const CCValAssign &VA,
                                                ISD::ArgFlagsTy Flags) const {
  llvm_unreachable("Cannot store arguments to stack");
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
SDValue ParasolTargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals, bool isThisReturn,
    SDValue ThisVal) const {
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, Parasol_CRetConv);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];

    // Pass 'this' value directly from the argument to return value, to avoid
    // reg unit interference
    if (i == 0 && isThisReturn) {
      assert(!VA.needsCustom() && VA.getLocVT() == MVT::i32 &&
             "unexpected return calling convention register assignment");
      InVals.push_back(ThisVal);
      continue;
    }

    SDValue Val;
    if (VA.needsCustom()) {
      llvm_unreachable("Vector and floating point values not supported yet");
    } else {
      Val =
          DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), VA.getLocVT(), InFlag);
      Chain = Val.getValue(1);
      InFlag = Val.getValue(2);
    }

    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, dl, VA.getValVT(), Val);
      break;
    }

    InVals.push_back(Val);
  }

  return Chain;
}

SDValue
ParasolTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                 SmallVectorImpl<SDValue> &InVals) const {
  llvm_unreachable("Cannot lower call");
}

/// HandleByVal - Every parameter *after* a byval parameter is passed
/// on the stack.  Remember the next parameter register to allocate,
/// and then confiscate the rest of the parameter registers to insure
/// this.

void ParasolTargetLowering::HandleByVal(CCState *State, unsigned &Size,
                                        Align align) const {
  // Byval (as with any stack) slots are always at least 4 byte aligned.
  // TODO: Look into how to replicate this but with the Align type.
  // align = std::max(align, 4U);
  align = Align(std::max(align.value(), (uint64_t)4));

  unsigned Reg = State->AllocateReg(GPRArgRegs);
  if (!Reg)
    return;

  // TODO: Use of .value() may be wrong here
  unsigned AlignInRegs = align.value() / 4;
  unsigned Waste = (Parasol::X4 - Reg) % AlignInRegs;
  for (unsigned i = 0; i < Waste; ++i)
    Reg = State->AllocateReg(GPRArgRegs);

  if (!Reg)
    return;

  unsigned Excess = 4 * (Parasol::X4 - Reg);

  // Special case when NSAA != SP and parameter size greater than size of
  // all remained GPR regs. In that case we can't split parameter, we must
  // send it to stack. We also must set NCRN to X4, so waste all
  // remained registers.
  // TODO: Check if getStackSize is the replacement for getNextStackOffset
  // https://reviews.llvm.org/D149566
  const unsigned NSAAOffset = State->getStackSize();
  if (NSAAOffset != 0 && Size > Excess) {
    while (State->AllocateReg(GPRArgRegs))
      ;
    return;
  }

  // First register for byval parameter is the first register that wasn't
  // allocated before this method call, so it would be "reg".
  // If parameter is small enough to be saved in range [reg, r4), then
  // the end (first after last) register would be reg + param-size-in-regs,
  // else parameter would be splitted between registers and stack,
  // end register would be r4 in this case.
  unsigned ByValRegBegin = Reg;
  unsigned ByValRegEnd = std::min<unsigned>(Reg + Size / 4, Parasol::X4);
  State->addInRegsParamInfo(ByValRegBegin, ByValRegEnd);
  // Note, first register is allocated in the beginning of function already,
  // allocate remained amount of registers we need.
  for (unsigned i = Reg + 1; i != ByValRegEnd; ++i)
    State->AllocateReg(GPRArgRegs);
  // A byval parameter that is split between registers and memory needs its
  // size truncated here.
  // In the case where the entire structure fits in registers, we set the
  // size in memory to zero.
  Size = std::max<int>(Size - Excess, 0);
}

SDValue
ParasolTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                   bool isVarArg,
                                   const SmallVectorImpl<ISD::OutputArg> &Outs,
                                   const SmallVectorImpl<SDValue> &OutVals,
                                   const SDLoc &DL, SelectionDAG &DAG) const {
  SmallVector<CCValAssign, 16> RVLocs;
  SmallVector<SDValue, 4> RetOps(1, Chain);
  MachineFunction &MF = DAG.getMachineFunction();
  CCState CCInfo(CallConv, isVarArg, MF, RVLocs, *DAG.getContext());
  SDValue Glue;

  // The caller has split up > 4-byte values on our behalf.
  if (OutVals.size() > 2) {
    assert("Cannot return more than 2 values in registers.");
  }

  analyzeOutputArgs(MF, CCInfo, Outs);

  for (size_t i = 0; i < RVLocs.size(); i++) {
    SDValue Val = OutVals[i];
    CCValAssign &VA = RVLocs[i];

    // Explicitly zero-extend values too small for 4-bytes.
    if (VA.getValVT().getStoreSize() < 4) {
      Val = DAG.getZeroExtendInReg(Val, DL, MVT::i32);
    }

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain;

  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }

  return DAG.getNode(ParasolISD::RetGlue, DL, MVT::Other, RetOps);
}

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//

SDValue ParasolTargetLowering::getGlobalAddressWrapper(
    SDValue GA, const GlobalValue *GV, SelectionDAG &DAG) const {
  llvm_unreachable("Unhandled global variable");
}

// Parasol supports boolean registers so setcc result type is i1
EVT ParasolTargetLowering::getSetCCResultType(const DataLayout &DL,
                                              LLVMContext &Context,
                                              EVT VT) const {
  // We do support single bits
  return MVT::i1;
}

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//

SDValue ParasolTargetLowering::LowerGlobalAddress(SDValue Op,
                                                  SelectionDAG &DAG) const {
  llvm_unreachable("Unsupported global address");
}

SDValue ParasolTargetLowering::LowerConstantPool(SDValue Op,
                                                 SelectionDAG &DAG) const {
  llvm_unreachable("Unsupported constant pool");
}

SDValue ParasolTargetLowering::LowerBlockAddress(SDValue Op,
                                                 SelectionDAG &DAG) const {
  llvm_unreachable("Unsupported block address");
}

SDValue ParasolTargetLowering::LowerRETURNADDR(SDValue Op,
                                               SelectionDAG &DAG) const {
  return SDValue();
}

SDValue ParasolTargetLowering::LowerOperation(SDValue Op,
                                              SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::BlockAddress:
    return LowerBlockAddress(Op, DAG);
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG);
  case ISD::RETURNADDR:
    return LowerRETURNADDR(Op, DAG);
  default:
    llvm_unreachable("unimplemented operand");
  }
}
