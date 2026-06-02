//===-- EZHISelDAGToDAG.cpp - A dag to dag inst selector for EZH ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHTargetMachine.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "ezh-isel"
#define PASS_NAME "EZH DAG->DAG Pattern Instruction Selection"

namespace {

class EZHDAGToDAGISel : public SelectionDAGISel {
public:
  EZHDAGToDAGISel() = delete;

  explicit EZHDAGToDAGISel(EZHTargetMachine &TargetMachine)
      : SelectionDAGISel(TargetMachine) {}

private:
#include "EZHGenDAGISel.inc"

  void Select(SDNode *N) override;
  bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                    InlineAsm::ConstraintCode ConstraintID,
                                    std::vector<SDValue> &OutOps) override;
  void selectFrameIndex(SDNode *N);
  bool tryIndexedLoadStore(SDNode *Node);
};

class EZHDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;
  explicit EZHDAGToDAGISelLegacy(EZHTargetMachine &TM)
      : SelectionDAGISelLegacy(ID, std::make_unique<EZHDAGToDAGISel>(TM)) {}
};

} // namespace

char EZHDAGToDAGISelLegacy::ID = 0;

INITIALIZE_PASS(EZHDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)

bool EZHDAGToDAGISel::tryIndexedLoadStore(SDNode *Node) {
  unsigned Opcode = Node->getOpcode();
  bool IsLoad = (Opcode == ISD::LOAD);
  ISD::MemIndexedMode AM;
  SDValue Base, Offset;

  if (IsLoad) {
    LoadSDNode *LD = cast<LoadSDNode>(Node);
    AM = LD->getAddressingMode();
    Base = LD->getBasePtr();
    Offset = LD->getOffset();
  } else {
    StoreSDNode *ST = cast<StoreSDNode>(Node);
    AM = ST->getAddressingMode();
    Base = ST->getBasePtr();
    Offset = ST->getOffset();
  }

  if (AM != ISD::POST_INC && AM != ISD::PRE_INC && AM != ISD::POST_DEC &&
      AM != ISD::PRE_DEC)
    return false;

  MemSDNode *MemNode = cast<MemSDNode>(Node);
  SDLoc DL(Node);
  EVT MemVT = MemNode->getMemoryVT();

  auto *C = dyn_cast<ConstantSDNode>(Offset);
  if (!C)
    return false;
  int64_t OffImm = C->getSExtValue();
  if (AM == ISD::POST_DEC || AM == ISD::PRE_DEC) {
    OffImm = -OffImm;
  }

  unsigned TargetOpcode = 0;

  if (MemVT == MVT::i32) {
    if ((OffImm & 3) != 0 || OffImm < -512 || OffImm > 508)
      return false;
    if (IsLoad) {
      TargetOpcode = (AM == ISD::POST_INC || AM == ISD::POST_DEC)
                         ? EZH::LDR_POST
                         : EZH::LDR_PRE;
    } else {
      TargetOpcode = (AM == ISD::POST_INC || AM == ISD::POST_DEC)
                         ? EZH::STR_POST
                         : EZH::STR_PRE;
    }
  } else if (MemVT == MVT::i8) {
    if (OffImm < -128 || OffImm > 255)
      return false;
    if (IsLoad) {
      if (cast<LoadSDNode>(Node)->getExtensionType() == ISD::SEXTLOAD) {
        TargetOpcode = (AM == ISD::POST_INC || AM == ISD::POST_DEC)
                           ? EZH::LDRBS_POST
                           : EZH::LDRBS_PRE;
      } else {
        TargetOpcode = (AM == ISD::POST_INC || AM == ISD::POST_DEC)
                           ? EZH::LDRB_POST
                           : EZH::LDRB_PRE;
      }
    } else {
      TargetOpcode = (AM == ISD::POST_INC || AM == ISD::POST_DEC)
                         ? EZH::STRB_POST
                         : EZH::STRB_PRE;
    }
  } else {
    return false; // i16 is not supported
  }

  SDValue TargetImm =
      CurDAG->getTargetConstant(static_cast<uint32_t>(OffImm), DL, MVT::i32);

  if (IsLoad) {
    SDValue Ops[] = {Base, TargetImm, MemNode->getChain()};
    SDNode *ResNode = CurDAG->getMachineNode(
        TargetOpcode, DL, CurDAG->getVTList(MVT::i32, MVT::i32, MVT::Other),
        Ops);
    ReplaceUses(SDValue(Node, 0), SDValue(ResNode, 0)); // Value
    ReplaceUses(SDValue(Node, 1), SDValue(ResNode, 1)); // New Ptr
    ReplaceUses(SDValue(Node, 2), SDValue(ResNode, 2)); // Chain
    CurDAG->RemoveDeadNode(Node);
  } else {
    SDValue Val = cast<StoreSDNode>(Node)->getValue();
    SDValue Ops[] = {Val, Base, TargetImm, MemNode->getChain()};
    SDNode *ResNode = CurDAG->getMachineNode(
        TargetOpcode, DL, CurDAG->getVTList(MVT::i32, MVT::Other), Ops);
    ReplaceUses(SDValue(Node, 0), SDValue(ResNode, 0)); // New Ptr
    ReplaceUses(SDValue(Node, 1), SDValue(ResNode, 1)); // Chain
    CurDAG->RemoveDeadNode(Node);
  }

  return true;
}

void EZHDAGToDAGISel::Select(SDNode *Node) {
  if (Node->isMachineOpcode()) {
    Node->setNodeId(-1);
    return;
  }

  if (Node->getOpcode() == ISD::CTLZ ||
      Node->getOpcode() == ISD::CTLZ_ZERO_POISON) {
    SDLoc dl(Node);
    SDValue Src = Node->getOperand(0);

    const TargetLowering &TLI = CurDAG->getTargetLoweringInfo();

    // Select appropriate CLZ library helper based on operand size
    EVT VT = Src.getValueType();
    const char *LibcallName = (VT == MVT::i64) ? "__clzdi2" : "__clzsi2";
    SDValue Callee = CurDAG->getExternalSymbol(
        LibcallName, TLI.getPointerTy(CurDAG->getDataLayout()));

    SDValue Chain = CurDAG->getEntryNode();
    SDValue InGlue;

    if (VT == MVT::i64) {
      // Explicitly generate EXTRACT_SUBREG nodes to let register allocator
      // handle virtual registers correctly!
      SDValue SrcLo =
          CurDAG->getNode(TargetOpcode::EXTRACT_SUBREG, dl, MVT::i32, Src,
                          CurDAG->getTargetConstant(sub_even, dl, MVT::i32));
      SDValue SrcHi =
          CurDAG->getNode(TargetOpcode::EXTRACT_SUBREG, dl, MVT::i32, Src,
                          CurDAG->getTargetConstant(sub_odd, dl, MVT::i32));

      Chain = CurDAG->getCopyToReg(Chain, dl, EZH::R0, SrcLo, InGlue);
      InGlue = Chain.getValue(1);
      Chain = CurDAG->getCopyToReg(Chain, dl, EZH::R1, SrcHi, InGlue);
      InGlue = Chain.getValue(1);
    } else {
      Chain = CurDAG->getCopyToReg(Chain, dl, EZH::R0, Src, InGlue);
      InGlue = Chain.getValue(1);
    }

    SmallVector<SDValue, 8> Ops;
    Ops.push_back(Callee);
    Ops.push_back(CurDAG->getRegister(EZH::R0, MVT::i32));
    if (VT == MVT::i64) {
      Ops.push_back(CurDAG->getRegister(EZH::R1, MVT::i32));
    }
    Ops.push_back(Chain);
    Ops.push_back(InGlue);

    SDVTList NodeTys = CurDAG->getVTList(MVT::Other, MVT::Glue);
    SDNode *CallNode = CurDAG->getMachineNode(EZH::CALLExt, dl, NodeTys, Ops);

    Chain = SDValue(CallNode, 0);
    InGlue = SDValue(CallNode, 1);

    SDValue Result =
        CurDAG->getCopyFromReg(Chain, dl, EZH::R0, MVT::i32, InGlue);

    ReplaceNode(Node, Result.getNode());
    return;
  }

  unsigned Opcode = Node->getOpcode();

  switch (Opcode) {
  case ISD::Constant: {
    auto *C = cast<ConstantSDNode>(Node);
    int64_t Val = C->getSExtValue();
    if (!isInt<11>(Val)) {
      SDLoc DL(Node);
      SDValue CPIdx = CurDAG->getTargetConstantPool(
          ConstantInt::get(Type::getInt32Ty(*CurDAG->getContext()), Val),
          TLI->getPointerTy(CurDAG->getDataLayout()));

      SDValue Ops[] = {CPIdx, CurDAG->getEntryNode()};
      SDNode *ResNode = CurDAG->getMachineNode(EZH::LOAD_CONSTANT, DL, MVT::i32,
                                               MVT::Other, Ops);

      ReplaceUses(SDValue(Node, 0), SDValue(ResNode, 0));
      CurDAG->RemoveDeadNode(Node);
      return;
    }
    break;
  }
  case ISD::FrameIndex:
    selectFrameIndex(Node);
    return;
  case ISD::LOAD:
  case ISD::STORE:
    if (tryIndexedLoadStore(Node))
      return;
    break;
  default:
    break;
  }

  SelectCode(Node);
}

void EZHDAGToDAGISel::selectFrameIndex(SDNode *Node) {
  SDLoc DL(Node);
  SDValue Imm = CurDAG->getTargetConstant(0, DL, MVT::i32);
  int FI = cast<FrameIndexSDNode>(Node)->getIndex();
  EVT VT = Node->getValueType(0);
  SDValue TFI = CurDAG->getTargetFrameIndex(FI, VT);
  unsigned Opc = EZH::ADDri__;
  if (Node->hasOneUse()) {
    CurDAG->SelectNodeTo(Node, Opc, VT, TFI, Imm);
    return;
  }
  ReplaceNode(Node, CurDAG->getMachineNode(Opc, DL, VT, TFI, Imm));
}

FunctionPass *llvm::createEZHISelDag(EZHTargetMachine &TM) {
  return new EZHDAGToDAGISelLegacy(TM);
}

bool EZHDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, InlineAsm::ConstraintCode ConstraintID,
    std::vector<SDValue> &OutOps) {
  OutOps.push_back(Op);
  return false;
}
