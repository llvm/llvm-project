//===-- P2ISelDAGToDAG.cpp - A Dag to Dag Inst Selector for P2 --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the P2 target.
//
//===----------------------------------------------------------------------===//

#include "P2ISelDAGToDAG.h"
#include "P2.h"

#include "P2MachineFunctionInfo.h"
#include "P2RegisterInfo.h"
#include "P2TargetMachine.h"
#include "MCTargetDesc/P2BaseInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "p2-isel"

bool P2DAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
    return SelectionDAGISel::runOnMachineFunction(MF);
}

void P2DAGToDAGISel::selectMultiplication(SDNode *N) {
    SDLoc DL(N);
    MVT vt = N->getSimpleValueType(0);

    assert(vt == MVT::i32 && "unexpected value type");
    bool isSigned = N->getOpcode() == ISD::SMUL_LOHI;
    unsigned op = isSigned ? P2::QMULrr : P2::QMULrr; // FIXME: replace with signed multiplication node (which will be a pseudo maybe?) (it might not be needed)

    SDValue cond = CurDAG->getTargetConstant(P2::ALWAYS, DL, vt);

    SDValue lhs = N->getOperand(0);
    SDValue rhs = N->getOperand(1);
    SDNode *mul = CurDAG->getMachineNode(op, DL, MVT::Glue, lhs, rhs, cond);
    SDValue in_chain = CurDAG->getEntryNode();
    SDValue in_glue = SDValue(mul, 0);

    // Copy the low half of the result, if it is needed.
    if (N->hasAnyUseOfValue(0)) {
        SDValue res = CurDAG->getCopyFromReg(in_chain, DL, P2::QX, vt, in_glue);
        ReplaceUses(SDValue(N, 0), res);

        in_chain = res.getValue(1);
        in_glue = res.getValue(2);
    }

    // Copy the high half of the result, if it is needed.
    if (N->hasAnyUseOfValue(1)) {
        SDValue res = CurDAG->getCopyFromReg(in_chain, DL, P2::QY, vt, in_glue);
        ReplaceUses(SDValue(N, 1), res);

        in_chain = res.getValue(1);
        in_glue = res.getValue(2);
    }

    CurDAG->RemoveDeadNode(N);
}

bool P2DAGToDAGISel::selectAddr(SDValue addr, SDValue &addr_result) {
    EVT vt = addr.getValueType();
    SDLoc DL(addr);

    if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(addr)) {
        LLVM_DEBUG(errs() << "select addr: value is a frame index\n");
        addr_result = CurDAG->getTargetFrameIndex(FIN->getIndex(), TLI->getPointerTy(CurDAG->getDataLayout()));
        return true;
    }

    if (addr.getOpcode() == P2ISD::GAWRAPPER) {
        addr_result = addr;
        return true;
    }

    if ((addr.getOpcode() == ISD::TargetExternalSymbol || addr.getOpcode() == ISD::TargetGlobalAddress)) {
        return false;
    }

    if (CurDAG->isBaseWithConstantOffset(addr)) {
        LLVM_DEBUG(errs() << "select addr: value is base with offset\n");
        ConstantSDNode *CN = dyn_cast<ConstantSDNode>(addr.getOperand(1));

        SDValue off = CurDAG->getTargetConstant(CN->getSExtValue(), DL, MVT::i32);
        SDValue cond = CurDAG->getTargetConstant(P2::ALWAYS, DL, MVT::i32);
        SDValue eff = CurDAG->getTargetConstant(P2::NOEFF, DL, MVT::i32);
        SDValue base = addr.getOperand(0);
        SDNode *add;

        LLVM_DEBUG(errs() << "Address node is ");
        LLVM_DEBUG(addr.dump());

        if (!isInt<9>(CN->getSExtValue())) {
            SDValue mov = SDValue(CurDAG->getMachineNode(P2::MOVri32, DL, MVT::i32, off), 0);
            SDValue ops[] = {base, mov, cond, eff};
            add = CurDAG->getMachineNode(P2::ADDrr, DL, vt, ops);
        } else {    
            SDValue ops[] = {base, off, cond, eff};
            add = CurDAG->getMachineNode(P2::ADDri, DL, vt, ops);
        }

        LLVM_DEBUG(errs() << "...base is: ");
        LLVM_DEBUG(base.dump());
        
        addr_result = SDValue(add, 0);

        return true;
    }

    addr_result = addr;
    return true;
}

void P2DAGToDAGISel::Select(SDNode *N) {
    // Dump information about the Node being selected
    LLVM_DEBUG(errs() << "<-------->\n");
    LLVM_DEBUG(errs() << "Selecting: "; N->dump(CurDAG); errs() << "\n");

    // this is already a machine op
    if (N->isMachineOpcode()) {
        LLVM_DEBUG(errs() << "== "; N->dump(CurDAG); errs() << "\n");
        N->setNodeId(-1);
        return;
    }

    /*
     * Instruction Selection not handled by the auto-generated
     * tablegen selection should be handled here.
     */
    unsigned Opcode = N->getOpcode();
    auto DL = CurDAG->getDataLayout();

    switch(Opcode) {
        default: break;

        case ISD::FrameIndex: {
            LLVM_DEBUG(errs() << "frame index node is being selected\n");
            FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(N);
            SDValue TFI = CurDAG->getTargetFrameIndex(FIN->getIndex(), getTargetLowering()->getPointerTy(DL));

            CurDAG->SelectNodeTo(N, P2::FRMIDX, getTargetLowering()->getPointerTy(DL), TFI);
            return;
        }

        // Mul with two results
        case ISD::SMUL_LOHI:
        case ISD::UMUL_LOHI: {
            selectMultiplication(N);
            return;
        }
    }

    // Select the default instruction
    SelectCode(N);
    LLVM_DEBUG(errs() << "Done selecting, chose "; N->dump());
}

FunctionPass *llvm::createP2ISelDag(P2TargetMachine &TM, CodeGenOpt::Level OptLevel) {
    return new P2DAGToDAGISel(TM, OptLevel);
}