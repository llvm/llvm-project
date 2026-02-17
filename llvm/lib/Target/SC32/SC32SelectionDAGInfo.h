#ifndef LLVM_LIB_TARGET_SC32_SC32SELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_SC32_SC32SELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "SC32GenSDNodeInfo.inc"

namespace llvm {

class SC32SelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  SC32SelectionDAGInfo();
};

} // namespace llvm

#endif
