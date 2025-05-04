/* --- PEISelDAGToDAG.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/7/2025
------------------------------------------ */

//指令选择pass
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "PE.h"
#include "PETargetMachine.h"
#include "MCTargetDesc/PEMCTargetDesc.h"
#include "PESubtarget.h"

using namespace llvm;

#define DEBUG_TYPE "pe-isel"
#define PASS_NAME "PE DAG->DAG Pattern Instruction Selection"

class PEDAGToDAGISel : public SelectionDAGISel {
    const PESubtarget *Subtarget = nullptr;

public:
    PEDAGToDAGISel() = delete;
    explicit PEDAGToDAGISel(PETargetMachine &TM) : SelectionDAGISel(TM){}

    bool runOnMachineFunction(MachineFunction &MF) override;

    bool SelectAddrFI(SDNode *Parent,SDValue AddrFI, SDValue &Base, SDValue &Offset);

private:
#include "PEGenDAGISel.inc"
    void Select(SDNode *Node) override;
    const PETargetMachine &getTargetMachine() {
        return static_cast<const PETargetMachine &>(TM);
    }
};

class PEDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
    static char ID;
    explicit PEDAGToDAGISelLegacy(PETargetMachine &TargetMachine, CodeGenOptLevel OptLevel);
};

bool PEDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
    Subtarget = &MF.getSubtarget<PESubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
}

bool PEDAGToDAGISel::SelectAddrFI(SDNode *Parent, SDValue AddrFI, SDValue &Base,
                                  SDValue &Offset) {
    if(FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(AddrFI)){
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(),
        AddrFI.getValueType());
        Offset = CurDAG->getTargetConstant(0, SDLoc(AddrFI),
        AddrFI.getValueType());
        return true;
    }
  return false;
}

void PEDAGToDAGISel::Select(SDNode *Node) {
  unsigned Opcode = Node->getOpcode();
  SDLoc DL(Node);

  LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");

  SelectCode(Node);
}

FunctionPass *llvm::createPEISelDag(PETargetMachine &TM, CodeGenOptLevel OptLevel) {
    return new PEDAGToDAGISelLegacy(TM, OptLevel);
}

char PEDAGToDAGISelLegacy::ID = 0;

PEDAGToDAGISelLegacy::PEDAGToDAGISelLegacy(PETargetMachine &TM, CodeGenOptLevel OptLevel)
    : SelectionDAGISelLegacy(ID, std::make_unique<PEDAGToDAGISel>(TM)) {}

//注册一个pass
INITIALIZE_PASS(PEDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)