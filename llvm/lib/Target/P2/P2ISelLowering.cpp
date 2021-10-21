//===-- P2ISelLowering.cpp - P2 DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that P2 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//
#include "P2ISelLowering.h"

#include "P2MachineFunctionInfo.h"
#include "P2TargetMachine.h"
#include "P2TargetObjectFile.h"
#include "MCTargetDesc/P2BaseInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

/*
it feels like there's a lot of redundant code here, so we should dig through what ever piece does and clean things up

How function calling works:

1. LowerCall is used to copy arguments to there destination registers/stack space
2. LowerFormalArguments is used to copy passed argument by the callee into the callee's stack frame
3. LowerReturn is used to copy the return value to r31 or the stack
4. LowerCallResult used to copy the return value out of r31 or the stack into the caller's space

Byval arguments: byvals will always be passed by the stack. easier this way for now, since most structs will be
greater than 4 ints anyway

*/

using namespace llvm;

#define DEBUG_TYPE "p2-isel-lower"

// addLiveIn - This helper function adds the specified physical register to the
// MachineFunction as a live in value.  It also creates a corresponding
// virtual register for it.
static unsigned addLiveIn(MachineFunction &MF, unsigned PReg, const TargetRegisterClass *RC) {
    unsigned VReg = MF.getRegInfo().createVirtualRegister(RC);
    MF.getRegInfo().addLiveIn(PReg, VReg);
    return VReg;
}

const char *P2TargetLowering::getTargetNodeName(unsigned Opcode) const {

    switch (Opcode) {
        case P2ISD::RET: return "P2RET";
        case P2ISD::CALL: return "P2CALL";
        case P2ISD::GAWRAPPER: return "P2GAWRAPPER";
        default:
            return nullptr;
    }
    #undef NODE
}

P2TargetLowering::P2TargetLowering(const P2TargetMachine &TM) : TargetLowering(TM), target_machine(TM) {
    addRegisterClass(MVT::i32, &P2::P2GPRRegClass);

    //  computeRegisterProperties - Once all of the register classes are
    //  added, this allows us to compute derived properties we expose.
    computeRegisterProperties(TM.getRegisterInfo());

    // See https://llvm.org/doxygen/TargetLowering_8h_source.html#l00192 for the various action
    setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);

    setOperationAction(ISD::MULHS, MVT::i32, Expand);
    setOperationAction(ISD::MULHU, MVT::i32, Expand);

    setOperationAction(ISD::VASTART, MVT::Other, Custom);
    setOperationAction(ISD::VAARG, MVT::Other, Custom);
    setOperationAction(ISD::VACOPY, MVT::Other, Expand);
    setOperationAction(ISD::VAEND, MVT::Other, Expand);

    setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
    setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

    setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
    setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
    setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

    // can expand mul instead of a libcall and it will just become mullo/hi, which we've already created
    // a lowering for
    setOperationAction(ISD::MUL, MVT::i32, Expand);

    for (MVT VT : MVT::integer_valuetypes()) {
        setOperationAction(ISD::ATOMIC_SWAP, VT, Expand);
        setOperationAction(ISD::ATOMIC_CMP_SWAP, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_NAND, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_MAX, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_MIN, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_UMAX, VT, Expand);
        setOperationAction(ISD::ATOMIC_LOAD_UMIN, VT, Expand);
    }

    setOperationAction(ISD::SETCC, MVT::i32, Expand);
    setOperationAction(ISD::BR_JT, MVT::Other, Expand);
    setOperationAction(ISD::JumpTable, MVT::i32, Custom);
    setOperationAction(ISD::BSWAP, MVT::i32, Expand);

    // setup all the functions that will be libcalls.
    setOperationAction(ISD::SDIV, MVT::i32, LibCall);
    setOperationAction(ISD::SREM, MVT::i32, LibCall);

    setLibcallName(RTLIB::SDIV_I32, "__sdiv");
    setLibcallName(RTLIB::SREM_I32, "__srem");
    setLibcallName(RTLIB::MEMCPY, "__memcpy");
    setLibcallName(RTLIB::MEMSET, "__memset");
}

SDValue P2TargetLowering::lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const {
    auto DL = DAG.getDataLayout();

    const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
    int64_t Offset = cast<GlobalAddressSDNode>(Op)->getOffset();

    // Create the TargetGlobalAddress node, folding in the constant offset.
    SDValue Result = DAG.getTargetGlobalAddress(GV, SDLoc(Op), getPointerTy(DL), Offset);
    return DAG.getNode(P2ISD::GAWRAPPER, SDLoc(Op), getPointerTy(DL), Result);
}

SDValue P2TargetLowering::lowerVASTART(SDValue Op, SelectionDAG &DAG) const {
    const MachineFunction &MF = DAG.getMachineFunction();
    const P2FunctionInfo *P2FI = MF.getInfo<P2FunctionInfo>();
    const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
    auto DL = DAG.getDataLayout();
    SDLoc dl(Op);

    // Vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    SDValue FI = DAG.getFrameIndex(P2FI->getVarArgsFrameIndex(), getPointerTy(DL));

    return DAG.getStore(Op.getOperand(0), dl, FI, Op.getOperand(1), MachinePointerInfo(SV), 0);
}

SDValue P2TargetLowering::lowerVAARG(SDValue Op, SelectionDAG &DAG) const {
    SDNode *Node = Op.getNode();
    EVT VT = Node->getValueType(0);
    SDValue Chain = Node->getOperand(0);
    SDValue VAListPtr = Node->getOperand(1);
    const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
    SDLoc DL(Node);
    EVT vt = VAListPtr.getValueType();

    SDValue VAList = DAG.getLoad(vt, DL, Chain, VAListPtr, MachinePointerInfo(SV));

    // decrement the pointer, VAList, to the next vaarg,
    // store the decremented VAList to the legalized pointer,
    // and load the actual argument out of the pointer VAList
    SDValue cond = DAG.getTargetConstant(P2::ALWAYS, DL, MVT::i32);
    SDValue eff = DAG.getTargetConstant(P2::NOEFF, DL, MVT::i32);
    SDValue ops[] = {VAList, DAG.getIntPtrConstant(VT.getSizeInBits()/8, DL, true), cond, eff};
    SDValue adj = SDValue(DAG.getMachineNode(P2::SUBri, DL, vt, ops), 0);
    Chain = DAG.getStore(VAList.getValue(1), DL, adj, VAListPtr, MachinePointerInfo(SV));

    return DAG.getLoad(VT, DL, Chain, VAList, MachinePointerInfo(), std::min(vt.getSizeInBits(), VT.getSizeInBits())/8);
}

SDValue P2TargetLowering::lowerJumpTable(SDValue Op, SelectionDAG &DAG) const {
    auto *N = cast<JumpTableSDNode>(Op);
    SDValue GA = DAG.getTargetJumpTable(N->getIndex(), MVT::i32);
    return DAG.getNode(P2ISD::GAWRAPPER, SDLoc(N), MVT::i32, GA);
}

#include "P2GenCallingConv.inc"

//===----------------------------------------------------------------------===//
//                  CALL Calling Convention Implementation
//===----------------------------------------------------------------------===//

SDValue P2TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
    switch (Op.getOpcode()) {
        case ISD::GlobalAddress:
            return lowerGlobalAddress(Op, DAG);
        case ISD::VASTART:
            return lowerVASTART(Op, DAG);
        case ISD::VAARG:
            return lowerVAARG(Op, DAG);
        case ISD::JumpTable:
            return lowerJumpTable(Op, DAG);
    }

    return SDValue();
}

/// LowerCall - functions arguments are copied from virtual regs to
/// physical regs and/or the stack frame, CALLSEQ_START and CALLSEQ_END are emitted.
SDValue P2TargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                              SmallVectorImpl<SDValue> &InVals) const {
    SelectionDAG &DAG                     = CLI.DAG;
    SDLoc DL                              = CLI.DL;
    SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
    SmallVectorImpl<SDValue> &OutVals     = CLI.OutVals;
    SmallVectorImpl<ISD::InputArg> &Ins   = CLI.Ins;
    SDValue Chain                         = CLI.Chain;
    SDValue Callee                        = CLI.Callee;
    CallingConv::ID CallConv              = CLI.CallConv;
    bool IsVarArg                         = CLI.IsVarArg;
    bool &isTailCall                      = CLI.IsTailCall;

    MachineFunction &MF = DAG.getMachineFunction();
    // MachineFrameInfo *MFI = &MF.getFrameInfo();
    const TargetFrameLowering *TFL = MF.getSubtarget().getFrameLowering();
    //P2FunctionInfo *FuncInfo = MF.getInfo<P2FunctionInfo>();
    //P2FunctionInfo *P2FI = MF.getInfo<P2FunctionInfo>();

    LLVM_DEBUG(errs() << "=== Lower Call\n");

    // P2 does not yet support tail call optimization.
    isTailCall = false;

    // Analyze operands of the call, assigning locations to each operand.
    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs, *DAG.getContext());
    if (IsVarArg) {
        CCInfo.AnalyzeCallOperands(Outs, CC_P2_Vararg);
    } else {
        CCInfo.AnalyzeCallOperands(Outs, CC_P2); // This doesn't seem to properly compute byval sizes
    }

    // Get a count of how many bytes are to be pushed on the stack.
    unsigned NextStackOffset = CCInfo.getNextStackOffset();

    LLVM_DEBUG(errs() << "Caller: " << MF.getName() << ". Next stack offset is " << NextStackOffset << "\n");

    // Chain is the output chain of the last Load/Store or CopyToReg node.
    unsigned StackAlignment = TFL->getStackAlignment();
    NextStackOffset = alignTo(NextStackOffset, StackAlignment);

    // start the call sequence.
    Chain = DAG.getCALLSEQ_START(Chain, NextStackOffset, 0, DL);

    // get the current stack pointer value
    SDValue StackPtr = DAG.getCopyFromReg(Chain, DL, P2::PTRA, getPointerTy(DAG.getDataLayout()));

    // we have 4 args on registers
    std::deque<std::pair<unsigned, SDValue>> RegsToPass;
    SmallVector<SDValue, 8> MemOpChains;

    CCInfo.rewindByValRegsInfo();

    // iterate over the argument locations.
    // at each location, push a reg/value pair onto RegsToPass, promoting the type if needed.
    // if the location is a memory location, load it from memory
    for (unsigned i = 0; i < ArgLocs.size(); i++) {
        SDValue Arg = OutVals[i];
        CCValAssign &VA = ArgLocs[i];
        MVT LocVT = VA.getLocVT();

        // Promote the value if needed.
        switch (VA.getLocInfo()) {
            default: llvm_unreachable("Unknown loc info!");
            case CCValAssign::Full:
                break;
            case CCValAssign::SExt:
                Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, LocVT, Arg);
                break;
            case CCValAssign::ZExt:
                Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, LocVT, Arg);
                break;
            case CCValAssign::AExt:
                Arg = DAG.getNode(ISD::ANY_EXTEND, DL, LocVT, Arg);
                break;
        }

        // Arguments that can be passed on register must be kept at
        // RegsToPass vector
        if (VA.isRegLoc()) {
            RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
        } else {
            // Register can't get to this point...
            assert(VA.isMemLoc());

            LLVM_DEBUG(errs() << "Stack argument location offset is " << VA.getLocMemOffset() << "\n");
            LLVM_DEBUG(errs() << "argument: ");
            LLVM_DEBUG(Arg.dump());

            SDValue mem_op;
            ISD::ArgFlagsTy Flags = Outs[i].Flags;
            int arg_size = 0;
            if (Flags.isByVal()) 
                arg_size = Flags.getByValSize();
            else
                arg_size = 4; 

            int off = (NextStackOffset-VA.getLocMemOffset()-arg_size);
            LLVM_DEBUG(errs() << "stack offset for argument: " << off << "\n");
            EVT vt = getPointerTy(DAG.getDataLayout());
            SDValue cond = DAG.getTargetConstant(P2::ALWAYS, DL, MVT::i32);
            SDValue eff = DAG.getTargetConstant(P2::NOEFF, DL, MVT::i32);
            SDValue ops[] = {StackPtr, DAG.getConstant(off, DL, MVT::i32, true), cond, eff};
            SDValue PtrOff = SDValue(DAG.getMachineNode(P2::ADDri, DL, vt, ops), 0);

            if (Flags.isByVal()) {
                LLVM_DEBUG(errs() << "Argument is byval of size " << Flags.getByValSize() << "\n");
                SDValue size_val = DAG.getConstant(arg_size, DL, MVT::i32);
                mem_op = DAG.getMemcpy(
                    Chain, DL, PtrOff, Arg, size_val, Flags.getNonZeroByValAlign(),
                    /*isVolatile*/ false,
                    /*AlwaysInline=*/true,
                    /*isTailCall=*/false, MachinePointerInfo(), MachinePointerInfo());
            } else {
                mem_op = DAG.getStore(Chain, DL, Arg, PtrOff, MachinePointerInfo());
            }

            MemOpChains.push_back(mem_op);
        }
    }

    LLVM_DEBUG(errs() << "Finished processing arguments\n");

    // Transform all store nodes into one single node because all store
    // nodes are independent of each other.
    if (!MemOpChains.empty())
        Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

    // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
    // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
    // node so that legalize doesn't hack it.
    bool GlobalOrExternal = false, InternalLinkage = false;
    // SDValue CalleeLo;

    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
        Callee = DAG.getTargetGlobalAddress(G->getGlobal(), DL, getPointerTy(DAG.getDataLayout()), 0);
        GlobalOrExternal = true;
        LLVM_DEBUG(errs() << "Callee is a global address\n");
    }  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
        const char *Sym = S->getSymbol();

        Callee = DAG.getTargetExternalSymbol(Sym, getPointerTy(DAG.getDataLayout()));

        LLVM_DEBUG(errs() << "Callee is an external symbol\n");
        GlobalOrExternal = true;
    }

    SmallVector<SDValue, 8> Ops(1, Chain);
    SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

    LLVM_DEBUG(errs() << "callee: "; Callee.dump());

    // buit the list of CopyToReg operations.
    getOpndList(Ops, RegsToPass, false, GlobalOrExternal, InternalLinkage, CLI, Callee, Chain);

    // call the function
    Chain = DAG.getNode(P2ISD::CALL, DL, NodeTys, Ops);
    SDValue InFlag = Chain.getValue(1);

    // end the call sequence
    Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NextStackOffset, DL, true), DAG.getIntPtrConstant(0, DL, true), InFlag, DL);
    InFlag = Chain.getValue(1);

    // Handle result values, copying them out of physregs into vregs that we
    // return.
    return LowerCallResult(Chain, InFlag, CallConv, IsVarArg, Ins, DL, DAG, InVals, CLI.Callee.getNode(), CLI.RetTy);
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
SDValue P2TargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                            CallingConv::ID CallConv, bool IsVarArg,
                                            const SmallVectorImpl<ISD::InputArg> &Ins,
                                            const SDLoc &dl, SelectionDAG &DAG,
                                            SmallVectorImpl<SDValue> &InVals,
                                            const SDNode *CallNode,
                                            const Type *RetTy) const {
    // Assign locations to each value returned by this call.
    SmallVector<CCValAssign, 16> RVLocs;
    CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs, *DAG.getContext());
    LLVM_DEBUG(errs() << "=== Lower Call Result\n");
    CCInfo.AnalyzeCallResult(Ins, RetCC_P2);

    SmallVector<std::pair<int, unsigned>, 4> ResultMemLocs;

    // Copy results out of physical registers.
    for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
        const CCValAssign &VA = RVLocs[i];
        if (VA.isRegLoc()) {
            SDValue RetValue;

            // Transform the arguments stored on
            // physical registers into virtual ones
            DAG.getMachineFunction().getRegInfo().addLiveIn(VA.getLocReg());
            RetValue = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), VA.getValVT(), InFlag);
            InVals.push_back(RetValue);

            Chain = RetValue.getValue(1);
            InFlag = RetValue.getValue(2);
        } else {
            assert(VA.isMemLoc() && "Must be memory location.");
            llvm_unreachable("returning values via memory not yet supported!");
        }
    }

    return Chain;
}

/// LowerFormalArguments - transform physical registers into virtual registers
/// and generate load operations for arguments places on the stack.
SDValue P2TargetLowering::LowerFormalArguments(SDValue Chain,
                                                CallingConv::ID CallConv,
                                                bool IsVarArg,
                                                const SmallVectorImpl<ISD::InputArg> &Ins,
                                                const SDLoc &DL, SelectionDAG &DAG,
                                                SmallVectorImpl<SDValue> &InVals) const {
    MachineFunction &MF = DAG.getMachineFunction();
    MachineFrameInfo *MFI = &MF.getFrameInfo();
    P2FunctionInfo *P2FI = MF.getInfo<P2FunctionInfo>();

    LLVM_DEBUG(errs() << "=== Lower Formal Arguments\n");

    SmallVector<SDValue, 12> MemOpChains;

    // Assign locations to all of the incoming arguments.
    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs, *DAG.getContext());

    if (IsVarArg) {
        CCInfo.AnalyzeFormalArguments(Ins, CC_P2_Vararg);
    } else {
        CCInfo.AnalyzeFormalArguments(Ins, CC_P2);
    }

    // creat a frame index for the PC saved by CALL
    int stack_size = CCInfo.getNextStackOffset();
    LLVM_DEBUG(errs() << " - next stack offset: " << stack_size << "\n");

    int fi = MFI->CreateFixedObject(4, stack_size, true); // frame index for where CALL saved the PC
    MFI->mapLocalFrameObject(fi, MFI->getObjectOffset(fi)); // sets it as pre-allocated, I think
    P2FI->setCallRetIdx(fi);

    // this will hold the memory offset of the last argument for use by var args, if needed
    int last_formal_arg_offset = 0;

    // save arg info for future use
    P2FI->setFormalArgInfo(CCInfo.getNextStackOffset(), CCInfo.getInRegsParamsCount() > 0);
    //const Function &Func = DAG.getMachineFunction().getFunction();

    CCInfo.rewindByValRegsInfo();

    // iterate over the argument locations and create copies to virtual registers
    for (unsigned i = 0; i < ArgLocs.size(); i++) {
        CCValAssign &VA = ArgLocs[i];
        EVT ValVT = VA.getValVT();
        ISD::ArgFlagsTy Flags = Ins[i].Flags;

        if (Flags.isByVal()) {
            LLVM_DEBUG(errs() << "have a byval argument\n");
            assert(Flags.getByValSize() && "ByVal args of size 0 should have been ignored by front-end.");
        }

        LLVM_DEBUG(errs() << " - Loading argument: " << i << "\n");

        // Arguments stored on registers
        if (VA.isRegLoc()) {
            MVT RegVT = VA.getLocVT();
            unsigned ArgReg = VA.getLocReg();
            const TargetRegisterClass *RC = getRegClassFor(RegVT);

            // create a new virtual register for this frame and copy to it from
            // the current argument register
            unsigned Reg = addLiveIn(DAG.getMachineFunction(), ArgReg, RC);
            SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegVT);

            // If this is an 8 or 16-bit value, it has been passed promoted
            // to 32 bits.  Insert an assert[sz]ext to capture this, then
            // truncate to the right size.
            if (VA.getLocInfo() != CCValAssign::Full) {
                unsigned Opcode = 0;
                if (VA.getLocInfo() == CCValAssign::SExt) {
                    Opcode = ISD::AssertSext;
                } else if (VA.getLocInfo() == CCValAssign::ZExt) {
                    Opcode = ISD::AssertZext;
                }

                if (Opcode) {
                    ArgValue = DAG.getNode(Opcode, DL, RegVT, ArgValue, DAG.getValueType(ValVT));
                }

                ArgValue = DAG.getNode(ISD::TRUNCATE, DL, ValVT, ArgValue);
            }
            InVals.push_back(ArgValue);
        } else { // values stored on the stack
            // sanity check
            assert(VA.isMemLoc());

            SDValue ArgValue;
            ISD::ArgFlagsTy Flags = Ins[i].Flags;
            
            int arg_size = 0;
            if (Flags.isByVal()) 
                arg_size = Flags.getByValSize();
            else
                arg_size = 4; 

            last_formal_arg_offset = stack_size-VA.getLocMemOffset()-arg_size;

            if (Flags.isByVal()) {
                LLVM_DEBUG(errs() << "byval mem offset: " << VA.getLocMemOffset() << "\n");

                int fi = MFI->CreateFixedObject(arg_size, last_formal_arg_offset, true);
                ArgValue = DAG.getFrameIndex(fi, getPointerTy(DAG.getDataLayout()));
            } else {
                // Load the argument to a virtual register
                auto obj_size = VA.getLocVT().getStoreSize();
                assert((obj_size <= 4) && "Unhandled argument--object size > stack slot size (4 bytes) for non-byval argument");
                LLVM_DEBUG(errs() << " - location offset: " << VA.getLocMemOffset() << "\n");

                // Create the frame index object for this incoming parameter
                fi = MFI->CreateFixedObject(obj_size, last_formal_arg_offset, true);

                LLVM_DEBUG(errs() << " - Loading argument from index " << fi << "\n");

                // Create the SelectionDAG nodes corresponding to a load from this parameter
                SDValue FIN = DAG.getFrameIndex(fi, MVT::i32);
                ArgValue = DAG.getLoad(VA.getLocVT(), DL, Chain, FIN, MachinePointerInfo::getFixedStack(MF, fi));
            }
            
            InVals.push_back(ArgValue);
        }
    }

    if (IsVarArg) {
        P2FI->setVarArgsFrameIndex(MFI->CreateFixedObject(4, last_formal_arg_offset-4, true));
    }

    for (unsigned i = 0; i < ArgLocs.size(); i++) {
        if (Ins[i].Flags.isSRet()) {
            unsigned Reg = P2FI->getSRetReturnReg();
            if (!Reg) {
                Reg = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i32));
                P2FI->setSRetReturnReg(Reg);
            }

            SDValue Copy = DAG.getCopyToReg(DAG.getEntryNode(), DL, Reg, InVals[i]);
            Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Copy, Chain);
        }
    }

    return Chain;
}

SDValue P2TargetLowering::LowerReturn(SDValue Chain,
                                        CallingConv::ID CallConv, bool IsVarArg,
                                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                                        const SmallVectorImpl<SDValue> &OutVals,
                                        const SDLoc &DL, SelectionDAG &DAG) const {

    // CCValAssign - represent the assignment of
    // the return value to a location
    SmallVector<CCValAssign, 16> RVLocs;
    MachineFunction &MF = DAG.getMachineFunction();
    // MachineFrameInfo MFI = MF.getFrameInfo();

    // CCState - Info about the registers and stack slot.
    CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());

    // Analyze return values.
    LLVM_DEBUG(errs() << "=== Lower Return: "; MF.dump());
    CCInfo.AnalyzeReturn(Outs, RetCC_P2);

    SDValue Flag;
    SmallVector<SDValue, 4> RetOps(1, Chain);

    // Copy the result values into the output register.
    for (unsigned i = 0; i != RVLocs.size(); ++i) {
        SDValue Val = OutVals[i];
        CCValAssign &VA = RVLocs[i];
        if (!VA.isRegLoc())
            continue;

        // sanity check
        assert(VA.isRegLoc() && "Can only return in registers!");

        if (RVLocs[i].getValVT() != RVLocs[i].getLocVT())
            Val = DAG.getNode(ISD::BITCAST, DL, RVLocs[i].getLocVT(), Val);

        Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Flag);

        // Guarantee that all emitted copies are stuck together with flags.
        Flag = Chain.getValue(1);
        RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
    }

    if (MF.getFunction().hasStructRetAttr()) {
        P2FunctionInfo *P2FI = MF.getInfo<P2FunctionInfo>();
        unsigned reg = P2FI->getSRetReturnReg();

        if (!reg)
            llvm_unreachable("sret virtual register not created in entry block");

        SDValue Val = DAG.getCopyFromReg(Chain, DL, reg, getPointerTy(DAG.getDataLayout()));
        unsigned R31 = P2::R31;

        Chain = DAG.getCopyToReg(Chain, DL, R31, Val, Flag);
        Flag = Chain.getValue(1);
        RetOps.push_back(DAG.getRegister(R31, getPointerTy(DAG.getDataLayout())));
    }

    RetOps[0] = Chain;  // Update chain.

    // Add the flag if we have it.
    if (Flag.getNode())
        RetOps.push_back(Flag);

    // Return on P2 is always a "ret"
    return DAG.getNode(P2ISD::RET, DL, MVT::Other, RetOps);
}

void P2TargetLowering::getOpndList(SmallVectorImpl<SDValue> &Ops,
            std::deque< std::pair<unsigned, SDValue> > &RegsToPass,
            bool IsPICCall, bool GlobalOrExternal, bool InternalLinkage,
            CallLoweringInfo &CLI, SDValue Callee, SDValue Chain) const {

    Ops.push_back(Callee);

    // Build a sequence of copy-to-reg nodes chained together with token
    // chain and flag operands which copy the outgoing args into registers.
    // The InFlag in necessary since all emitted instructions must be
    // stuck together.
    SDValue InFlag;

    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
        Chain = CLI.DAG.getCopyToReg(Chain, CLI.DL, RegsToPass[i].first, RegsToPass[i].second, InFlag);
        InFlag = Chain.getValue(1);
    }

    // Add argument registers to the end of the list so that they are
    // known live into the call.
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
        Ops.push_back(CLI.DAG.getRegister(RegsToPass[i].first, RegsToPass[i].second.getValueType()));

    // Add a register mask operand representing the call-preserved registers.
    const TargetRegisterInfo *TRI = target_machine.getRegisterInfo();
    const uint32_t *Mask = TRI->getCallPreservedMask(CLI.DAG.getMachineFunction(), CLI.CallConv);
    assert(Mask && "Missing call preserved mask for calling convention");

    Ops.push_back(CLI.DAG.getRegisterMask(Mask));

    if (InFlag.getNode())
        Ops.push_back(InFlag);
}

//===----------------------------------------------------------------------===//
//                           P2 Inline Assembly Support
//===----------------------------------------------------------------------===//

Register P2TargetLowering::getRegisterByName(const char *RegName, LLT VT, const MachineFunction &MF) const {
    Register Reg  = StringSwitch<unsigned>(RegName)
            .Case("r0",     P2::R0)
            .Case("r1",     P2::R1)
            .Case("r2",     P2::R2)
            .Case("r3",     P2::R3)
            .Case("r4",     P2::R4)
            .Case("r5",     P2::R5)
            .Case("r6",     P2::R6)
            .Case("r7",     P2::R7)
            .Case("r8",     P2::R8)
            .Case("r9",     P2::R9)
            .Case("r10",    P2::R10)
            .Case("r11",    P2::R11)
            .Case("r12",    P2::R12)
            .Case("r13",    P2::R13)
            .Case("r14",    P2::R14)
            .Case("r15",    P2::R15)
            .Case("r16",    P2::R16)
            .Case("r17",    P2::R17)
            .Case("r18",    P2::R18)
            .Case("r19",    P2::R19)
            .Case("r20",    P2::R20)
            .Case("r21",    P2::R21)
            .Case("r22",    P2::R22)
            .Case("r23",    P2::R23)
            .Case("r24",    P2::R24)
            .Case("r25",    P2::R25)
            .Case("r26",    P2::R26)
            .Case("r27",    P2::R27)
            .Case("r28",    P2::R28)
            .Case("r29",    P2::R29)
            .Case("r30",    P2::R30)
            .Case("r31",    P2::R31)
            .Case("ijmp3",  P2::IJMP3)
            .Case("iret3",  P2::IRET3)
            .Case("ijmp2",  P2::IJMP2)
            .Case("iret2",  P2::IRET2)
            .Case("ijmp1",  P2::IJMP1)
            .Case("iret1",  P2::IRET1)
            .Case("pa",     P2::PA)
            .Case("pb",     P2::PB)
            .Case("ptra",   P2::PTRA)
            .Case("ptrb",   P2::PTRB)
            .Case("dira",   P2::DIRA)
            .Case("dirb",   P2::DIRB)
            .Case("outa",   P2::OUTA)
            .Case("outb",   P2::OUTB)
            .Case("ina",    P2::INA)
            .Case("inb",    P2::INB)
            .Default(0);

    if (Reg) return Reg;

    report_fatal_error("Invalid register name global variable");
}

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
P2TargetLowering::ConstraintType P2TargetLowering::getConstraintType(StringRef Constraint) const {
    return TargetLowering::getConstraintType(Constraint);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight P2TargetLowering::getSingleConstraintMatchWeight(AsmOperandInfo &info, const char *constraint) const {
    return CW_Default;
}

/// This is a helper function to parse a physical register string and split it
/// into non-numeric and numeric parts (Prefix and Reg). The first boolean flag
/// that is returned indicates whether parsing was successful. The second flag
/// is true if the numeric part exists.
static std::pair<bool, bool> parsePhysicalReg(const StringRef &C, std::string &Prefix, unsigned long long &Reg) {
  if (C.front() != '{' || C.back() != '}')
    return std::make_pair(false, false);

  // Search for the first numeric character.
  StringRef::const_iterator I, B = C.begin() + 1, E = C.end() - 1;
  I = std::find_if(B, E, std::ptr_fun(isdigit));

  Prefix.assign(B, I - B);

  // The second flag is set to false if no numeric characters were found.
  if (I == E)
    return std::make_pair(true, false);

  // Parse the numeric characters.
  return std::make_pair(!getAsUnsignedInteger(StringRef(I, E - I), 10, Reg), true);
}

std::pair<unsigned, const TargetRegisterClass *> P2TargetLowering::parseRegForInlineAsmConstraint(const StringRef &C, MVT VT) const {
    const TargetRegisterClass *RC;
    std::string Prefix;
    unsigned long long Reg;

    std::pair<bool, bool> R = parsePhysicalReg(C, Prefix, Reg);

    LLVM_DEBUG(errs() << "for constraint " << C << ":\n");
    LLVM_DEBUG(errs() << " prefix is " << Prefix << "\n");
    LLVM_DEBUG(errs() << " reg is " << Reg << "\n");

    if (!R.first)
        return std::make_pair(0U, nullptr);
    if (!R.second)
        return std::make_pair(0U, nullptr);

    assert(Prefix == "$");
    RC = getRegClassFor((VT == MVT::Other) ? MVT::i32 : VT);

    assert(Reg < RC->getNumRegs());
    return std::make_pair(*(RC->begin() + Reg), RC);
}

/// Given a register class constraint, like 'r', if this corresponds directly
/// to an LLVM register class, return a register of 0 and the register class
/// pointer.
std::pair<unsigned, const TargetRegisterClass *> P2TargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                                                                StringRef Constraint,
                                                                                                MVT VT) const {
    if (Constraint.size() == 1) {
        switch (Constraint[0]) {
        case 'r':
            if (VT == MVT::i32 || VT == MVT::i16 || VT == MVT::i8) {
                return std::make_pair(0U, &P2::P2GPRRegClass);
            } else {
                llvm_unreachable("Unexpected type for constraint r");
            }
            break;
        default:
            llvm_unreachable("Unexpected type.");
        }
    }

    std::pair<unsigned, const TargetRegisterClass *> R;
    R = parseRegForInlineAsmConstraint(Constraint, VT);

    if (R.second)
        return R;

    return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void P2TargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                     std::string &Constraint,
                                                     std::vector<SDValue>&Ops,
                                                     SelectionDAG &DAG) const {
    TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}