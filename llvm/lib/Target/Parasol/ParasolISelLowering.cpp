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

  addRegisterClass(MVT::i32, &Parasol::PRRegClass);

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

  // Set minimum and preferred function alignment (log2)
  setMinFunctionAlignment(Align(1));
  setPrefFunctionAlignment(Align(1));

  // Set preferred loop alignment (log2)
  setPrefLoopAlignment(Align(1));
}

const char *ParasolTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case ParasolISD::Ret:
    return "ParasolISD::Ret";
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
static const MCPhysReg GPRArgRegs[] = {Parasol::P0, Parasol::P1, Parasol::P2,
                                       Parasol::P3};

/// LowerFormalArguments - transform physical registers into virtual registers
/// and generate load operations for arguments places on the stack.
SDValue ParasolTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
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
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_Parasol);

  SmallVector<SDValue, 16> ArgValues;
  SDValue ArgValue;
  Function::const_arg_iterator CurOrigArg = MF.getFunction().arg_begin();
  unsigned CurArgIdx = 0;

  // SUNSCREEN TODO: Commented out this code since we are not handling cases of
  // variadic arguments or pointer arguments.
  //
  // Calculate the amount of stack space that we need to allocate to store
  // byval and variadic arguments that are passed in registers.
  // We need to know this before we allocate the first byval or variadic
  // argument, as they will be allocated a stack slot below the CFA (Canonical
  // Frame Address, the stack pointer at entry to the function).
  unsigned ArgRegBegin = Parasol::P32;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    if (CCInfo.getInRegsParamsProcessed() >= CCInfo.getInRegsParamsCount())
      break;

    CCValAssign &VA = ArgLocs[i];
    unsigned Index = VA.getValNo();
    ISD::ArgFlagsTy Flags = Ins[Index].Flags;
    if (!Flags.isByVal())
      continue;

    assert(VA.isMemLoc() && "unexpected byval pointer in reg");
    unsigned RBegin, REnd;
    CCInfo.getInRegsParamInfo(CCInfo.getInRegsParamsProcessed(), RBegin, REnd);
    ArgRegBegin = std::min(ArgRegBegin, RBegin);

    CCInfo.nextInRegsParam();
  }
  CCInfo.rewindByValRegsInfo();

  int lastInsIndex = -1;

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    if (Ins[VA.getValNo()].isOrigArg()) {
      std::advance(CurOrigArg,
                   Ins[VA.getValNo()].getOrigArgIndex() - CurArgIdx);
      CurArgIdx = Ins[VA.getValNo()].getOrigArgIndex();
    }

    unsigned Index = VA.getValNo();
    Argument *Arg = F.getArg(i);
    ISD::ArgFlagsTy Flags = Ins[Index].Flags;

    // Arguments stored in registers.
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();

      if (VA.needsCustom()) {
        llvm_unreachable("Custom val assignment not supported by "
                         "FORMAL_ARGUMENTS Lowering");
      } else {
        const TargetRegisterClass *RC;

        // SUNSCREEN TODO: This gets triggered but the resulting registers are
        // still of type r instead of e.  This also seems to double allocate the
        // registers, as now the ParasolInstrInfo::copyPhysReg reports the
        // register list as X, X, E, E, R instead of X, X, R when using
        // non-encrypted arguments.
        if (Flags.isEncrypted()) {
          RC = &Parasol::PRRegClass;
          RegVT.setIsEncrypted(true);
        }
        if (RegVT == MVT::i32) {
          RC = &Parasol::PRRegClass;
        } else {
          llvm_unreachable("RegVT not supported by FORMAL_ARGUMENTS Lowering");
        }

        // Transform the arguments in physical registers into virtual ones.
        unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
        ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);

        unsigned bind_op;

        if (Arg->onlyReadsMemory() == 1) {
          bind_op =
              Flags.isEncrypted() ? ISD::BINDREADENCRYPTED : ISD::BINDREADPLAIN;
        } else {
          bind_op = Flags.isEncrypted() ? ISD::BINDREADWRITEENCRYPTED
                                        : ISD::BINDREADWRITEPLAIN;
        }

        SDValue BindId = DAG.getConstant(i, dl, EVT(MVT::SimpleValueType::i32));

        ArgValue = DAG.getNode(bind_op, dl, VA.getValVT(), ArgValue, BindId);
        ArgValue.getNode()->setHasEncryptedValue(Flags.isEncrypted());
      }

      // If this is an 8 or 16-bit value, it is really passed promoted
      // to 32 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      switch (VA.getLocInfo()) {
      default:
        llvm_unreachable("Unknown loc info!");
      case CCValAssign::Full:
        break;
      case CCValAssign::BCvt:
        ArgValue = DAG.getNode(ISD::BITCAST, dl, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::SExt:
        ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
        break;
      case CCValAssign::ZExt:
        ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
        break;
      }
      InVals.push_back(ArgValue);
    } else { // VA.isRegLoc()
      // sanity check
      assert(VA.isMemLoc());
      assert(VA.getValVT() != MVT::i64 && "i64 should already be lowered");

      int index = VA.getValNo();

      // Some Ins[] entries become multiple ArgLoc[] entries.
      // Process them only once.
      if (index != lastInsIndex) {
        llvm_unreachable("Cannot retrieve arguments from the stack");
      }
    }
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
  CCState CCInfo(CallConv, isVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, Parasol_CRetConv);
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
                                   const SDLoc &dl, SelectionDAG &DAG) const {
  // CCValAssign - represent the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slots.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  // Analyze outgoing return values.
  CCInfo.AnalyzeReturn(Outs, Parasol_CRetConv);

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)

  // Copy the result values into the output registers.
  for (unsigned i = 0, realRVLocIdx = 0; i != RVLocs.size();
       ++i, ++realRVLocIdx) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    SDValue Arg = OutVals[realRVLocIdx];
    bool ReturnF16 = false;

    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      if (!ReturnF16)
        Arg = DAG.getNode(ISD::BITCAST, dl, VA.getLocVT(), Arg);
      break;
    }

    if (VA.needsCustom()) {
      llvm_unreachable("Custom val assignment not supported by "
                       "RETURN Lowering");
    } else {
      Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), Arg, Flag);
    }

    // Guarantee that all emitted copies are stuck together, avoiding something
    // bad.
    Flag = Chain.getValue(1);
    RetOps.push_back(
        DAG.getRegister(VA.getLocReg(), ReturnF16 ? MVT::f16 : VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(ParasolISD::Ret, dl, MVT::Other, RetOps);
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
