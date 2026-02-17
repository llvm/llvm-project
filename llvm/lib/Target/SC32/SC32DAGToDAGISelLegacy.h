#ifndef LLVM_LIB_TARGET_SC32_SC32DAGTODAGISELLEGACY_H
#define LLVM_LIB_TARGET_SC32_SC32DAGTODAGISELLEGACY_H

#include "llvm/CodeGen/SelectionDAGISel.h"

namespace llvm {

class SC32DAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;

  SC32DAGToDAGISelLegacy(TargetMachine &TM);
};

} // namespace llvm

#endif
