#include "llvm/CodeGen/SelectionDAGNodes.h"

using namespace llvm;

// From llvm/lib/Target/Mips/MipsSEISelLowering.cpp
static bool isSplatVector(const BuildVectorSDNode *N) {
  unsigned int nOps = N->getNumOperands();
  assert(nOps > 1 && "isSplatVector(): N is 0 or 1 sized build vector");

  SDValue Operand0 = N->getOperand(0);

  for (unsigned int i = 1; i < nOps; ++i) {
    if (N->getOperand(i) != Operand0)
      return false;
  }

  return true;
}
