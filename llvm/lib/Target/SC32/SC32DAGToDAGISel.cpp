#include "SC32DAGToDAGISel.h"
#include "MCTargetDesc/SC32MCTargetDesc.h"
#include "SC32SelectionDAGInfo.h"

using namespace llvm;

#define GET_DAGISEL_BODY SC32DAGToDAGISel
#include "SC32GenDAGISel.inc"

void SC32DAGToDAGISel::Select(SDNode *N) {
  switch (N->getOpcode()) {
  case ISD::FrameIndex:
    FrameIndexSDNode *FIN = cast<FrameIndexSDNode>(N);
    SDValue FI =
        CurDAG->getTargetFrameIndex(FIN->getIndex(), FIN->getValueType(0));
    ReplaceNode(N, FI.getNode());
    return;
  }

  SelectCode(N);
}
