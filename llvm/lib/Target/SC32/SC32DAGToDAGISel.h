#ifndef LLVM_LIB_TARGET_SC32_SC32DAGTODAGISEL_H
#define LLVM_LIB_TARGET_SC32_SC32DAGTODAGISEL_H

#include "llvm/CodeGen/SelectionDAGISel.h"

namespace llvm {

class SC32DAGToDAGISel : public SelectionDAGISel {
public:
#define GET_DAGISEL_DECL
#include "SC32GenDAGISel.inc"

  using SelectionDAGISel::SelectionDAGISel;

  void Select(SDNode *N) override;
};

} // namespace llvm

#endif
