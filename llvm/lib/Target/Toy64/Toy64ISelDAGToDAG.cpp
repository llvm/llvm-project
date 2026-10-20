//===-- Toy64ISelDAGToDAG.cpp - Toy64 DAG Instruction Selector -----------===//

#include "Toy64.h"
#include "Toy64ISelLowering.h"
#include "Toy64TargetMachine.h"

#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include <memory>

using namespace llvm;

#define DEBUG_TYPE "toy64-isel"
#define PASS_NAME "Toy64 DAG->DAG Instruction Selection"

namespace {

class Toy64DAGToDAGISel : public SelectionDAGISel {
public:
  explicit Toy64DAGToDAGISel(
      Toy64TargetMachine &TM)
      : SelectionDAGISel(
            TM,
            TM.getOptLevel()) {}

private:
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

class Toy64DAGToDAGISelLegacy
    : public SelectionDAGISelLegacy {

public:
  static char ID;

  explicit Toy64DAGToDAGISelLegacy(
      Toy64TargetMachine &TM)
      : SelectionDAGISelLegacy(
            ID,
            std::make_unique<Toy64DAGToDAGISel>(TM)) {}
};

} // end anonymous namespace

char Toy64DAGToDAGISelLegacy::ID = 0;

INITIALIZE_PASS(
    Toy64DAGToDAGISelLegacy,
    DEBUG_TYPE,
    PASS_NAME,
    false,
    false)

FunctionPass *
llvm::createToy64ISelDagLegacyPass(
    Toy64TargetMachine &TM) {

  return new Toy64DAGToDAGISelLegacy(TM);
}