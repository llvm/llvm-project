/* --- PEISelDAGToDAG.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/7/2025
------------------------------------------ */

// 指令选择pass
#include "MCTargetDesc/PEMCTargetDesc.h"
#include "PE.h"
#include "PESubtarget.h"
#include "PETargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"

using namespace llvm;

#define DEBUG_TYPE "pe-isel"
#define PASS_NAME "PE DAG->DAG Pattern Instruction Selection"
class PEDAGToDAGISel : public SelectionDAGISel {
public:
  PEDAGToDAGISel() = delete;
  explicit PEDAGToDAGISel(PETargetMachine &TM, CodeGenOptLevel OL)
      : SelectionDAGISel(TM), Subtarget(nullptr) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool SelectAddrFI(SDValue AddrFI, SDValue &Base, SDValue &Offset);

private:
#include "PEGenDAGISel.inc"
  const PESubtarget *Subtarget;

  void Select(SDNode *Node) override;
  const PETargetMachine &getTargetMachine() {
    return static_cast<const PETargetMachine &>(TM);
  }
};

class PEDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;
  explicit PEDAGToDAGISelLegacy(PETargetMachine &TargetMachine,
                                CodeGenOptLevel OptLevel);
};

bool PEDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<PESubtarget>();
  return SelectionDAGISel::runOnMachineFunction(MF);
}

bool PEDAGToDAGISel::SelectAddrFI(SDValue AddrFI, SDValue &Base,
                                  SDValue &Offset) {
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(AddrFI)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), AddrFI.getValueType());
    Offset = CurDAG->getTargetConstant(0, SDLoc(AddrFI), AddrFI.getValueType());
    return true;
  }
  if (CurDAG->isBaseWithConstantOffset(AddrFI)) {
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(AddrFI.getOperand(1));
    if (FrameIndexSDNode *FI =
            dyn_cast<FrameIndexSDNode>(AddrFI.getOperand(0))) {
      Base = CurDAG->getTargetFrameIndex(FI->getIndex(), AddrFI.getValueType());
    } else {
      Base = AddrFI.getOperand(0);
    }
    Offset = CurDAG->getTargetConstant(CN->getZExtValue(), SDLoc(AddrFI),
                                       AddrFI.getValueType());
    return true;
  }
  return false;
}


void PEDAGToDAGISel::Select(SDNode *Node) {
  //   unsigned Opcode = Node->getOpcode();
  SDLoc DL(Node);
  switch (Node->getOpcode()) {
  case ISD::BRCOND: {
    SDValue Cond = Node->getOperand(1);
    // 检查第二个操作数是否为常量0
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Cond)) {
      if (C->getZExtValue() == 0) {
        // 直接删除该节点
        SDValue Chain = Node->getOperand(0);
        ReplaceNode(Node, Chain.getNode());
        return;
      } else {
        // 非0，直接生成跳转指令
        SDValue Chain = Node->getOperand(0);
        SDValue Dest = Node->getOperand(2);
        // 生成你的跳转指令，比如 JUMP
        MachineSDNode *J =
            CurDAG->getMachineNode(PE::J, SDLoc(Node), MVT::Other, Chain, Dest);
        ReplaceNode(Node, J);
        return;
      }
    }
    break;
  }
  default:
    break;
  }

  LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
  SelectCode(Node);
}

FunctionPass *llvm::createPEISelDag(PETargetMachine &TM,CodeGenOptLevel OptLevel) {
  return new PEDAGToDAGISelLegacy(TM, OptLevel);
}

char PEDAGToDAGISelLegacy::ID = 0;

PEDAGToDAGISelLegacy::PEDAGToDAGISelLegacy(PETargetMachine &TM,
                                           CodeGenOptLevel OptLevel)
    : SelectionDAGISelLegacy(
          ID, std::make_unique<PEDAGToDAGISel>(TM, TM.getOptLevel())) {}

// 注册一个pass
INITIALIZE_PASS(PEDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)