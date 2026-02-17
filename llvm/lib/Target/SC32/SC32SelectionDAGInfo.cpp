#include "SC32SelectionDAGInfo.h"

using namespace llvm;

#define GET_SDNODE_DESC
#include "SC32GenSDNodeInfo.inc"

SC32SelectionDAGInfo::SC32SelectionDAGInfo()
    : SelectionDAGGenTargetInfo(SC32GenSDNodeInfo) {}
