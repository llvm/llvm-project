//===-- Toy64ISelDAGToDAG.cpp - Toy64 DAG Instruction Selector -----------===//

#include "Toy64.h"
#include "Toy64ISelLowering.h"
#include "Toy64TargetMachine.h"

#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Support/MathExtras.h"

#include <memory>

using namespace llvm;

#define DEBUG_TYPE "toy64-isel"
#define PASS_NAME "Toy64 DAG->DAG Instruction Selection"

namespace {

class Toy64DAGToDAGISel : public SelectionDAGISel {
public:
  explicit Toy64DAGToDAGISel(Toy64TargetMachine &TM)
      : SelectionDAGISel(TM, TM.getOptLevel()) {}

private:
  bool SelectAddr(SDValue Addr, SDValue &Base, SDValue &Offset);
#include "Toy64GenDAGISel.inc"

  void Select(SDNode *Node) override {
    // Already selected.
    if (Node->isMachineOpcode()) {
      Node->setNodeId(-1);
      return;
    }

    // Invoke the TableGen-generated pattern matcher.
    SelectCode(Node);
  }
};

class Toy64DAGToDAGISelLegacy : public SelectionDAGISelLegacy {

public:
  static char ID;

  explicit Toy64DAGToDAGISelLegacy(Toy64TargetMachine &TM)
      : SelectionDAGISelLegacy(ID, std::make_unique<Toy64DAGToDAGISel>(TM)) {}
};

} // end anonymous namespace

bool Toy64DAGToDAGISel::SelectAddr(SDValue Addr, SDValue &Base,
                                   SDValue &Offset) {

  SDLoc DL(Addr);

  // ------------------------------------------------------------
  // Case 1: Stack/frame object.
  //
  // SelectionDAG represents an alloca/spill location using a
  // FrameIndex node.
  //
  // Convert:
  //
  //     FrameIndex FI
  //
  // into:
  //
  //     TargetFrameIndex FI, 0
  //
  // PEI will later call eliminateFrameIndex() and turn this into
  // X2 + the final stack offset.
  // ------------------------------------------------------------

  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {

    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i64);

    Offset = CurDAG->getTargetConstant(0, DL, MVT::i64);

    return true;
  }

  // ------------------------------------------------------------
  // Case 2:
  //
  //     Base + constant
  //
  // where the constant fits the Toy64 signed 12-bit displacement.
  // ------------------------------------------------------------

  if (Addr.getOpcode() == ISD::ADD) {

    SDValue LHS = Addr.getOperand(0);
    SDValue RHS = Addr.getOperand(1);

    if (auto *CN = dyn_cast<ConstantSDNode>(RHS)) {

      int64_t Imm = CN->getSExtValue();

      if (isInt<12>(Imm)) {

        Base = LHS;

        Offset = CurDAG->getTargetConstant(Imm, DL, MVT::i64);

        return true;
      }
    }

    // ADD is commutative, so handle:
    //
    //     constant + Base
    //
    // too.
    if (auto *CN = dyn_cast<ConstantSDNode>(LHS)) {

      int64_t Imm = CN->getSExtValue();

      if (isInt<12>(Imm)) {

        Base = RHS;

        Offset = CurDAG->getTargetConstant(Imm, DL, MVT::i64);

        return true;
      }
    }
  }

  // ------------------------------------------------------------
  // Case 3:
  //
  // Plain register address.
  //
  //     Base
  //
  // becomes:
  //
  //     Base + 0
  // ------------------------------------------------------------

  Base = Addr;

  Offset = CurDAG->getTargetConstant(0, DL, MVT::i64);

  return true;
}

char Toy64DAGToDAGISelLegacy::ID = 0;

INITIALIZE_PASS(Toy64DAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createToy64ISelDagLegacyPass(Toy64TargetMachine &TM) {

  return new Toy64DAGToDAGISelLegacy(TM);
}