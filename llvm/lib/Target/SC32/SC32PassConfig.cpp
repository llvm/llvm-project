#include "SC32PassConfig.h"
#include "SC32DAGToDAGISelLegacy.h"

using namespace llvm;

bool SC32PassConfig::addInstSelector() {
  addPass(new SC32DAGToDAGISelLegacy(*TM));
  return false;
}
